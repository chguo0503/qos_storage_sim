# 给模板 full24 / semi24 / near35 增加 OD baseline

本目录直接复制 `template/qos_experiments_20260919` 的九份原始冻结 manifest；不调用输入生成器，不改变请求、NPU 分配、卡内顺序、到达时间或逐块落盘位置。

重要区别：模板这三组使用旧条带 `(block_index+npu_id)%3`。2026-09-18 的 OD diverse 结果已改为 Ring Hash，完整输入指纹不同，因此没有复用。此次仅重放冻结布局，不向当前项目重新加入条带生成策略。

配置：32 NPU，3 SSU × 40 GiB/s，NPU 接收链路 50 GiB/s，8 层，batch=1，全部 arrival=0，固定卡内 FIFO，跨请求首层预取开启，5 ms 拥塞快照，seed7/19/43。full 和 near35 每卡 42 条，semi 每卡 30 条；每组含 24 个 data 画像。

OD 每盘使用 32 条独立 Path，每个 NPU 固定独占一条，CIR=1.25 GiB/s，PIR 不设上限，空闲份额可以借用；不动态改 CIR，不重排请求。ASU 使用共享 Path0。

统计沿用模板：主窗口 [2,4) 秒；利用率为窗口内计算卡时间 / (32 × 2 秒)。TTFT 使用窗口内接纳请求，逐个跟踪到最终完成，SLO 基准为该请求 8 层纯计算时间，阈值为基准 ×1.5。三种子分别计算后等权平均；CDF 同样等权。另保留 [2,6) 秒和完整输入排空统计。

`metrics.py` 逐字节复制原模板的旧指标实现。需求为当前已接纳请求的每盘 V/C，含其等待驻留时间；下一请求预取不重复叠加为需求。真实供给积分使用 SSD 服务区间 `[ssd_activation_time, link_enqueue_time)`，在接收完成回调中计量，完整排空保证不丢失晚到回调。输出 2 ms、10 ms 每盘平均和 warm 每卡平均。

`runtime/` 是本次执行开始前冻结的新包代码。`version_audit.json` 保存旧版/新版结构差异；三个场景 seed7 各自完整重放 ASU，并在 `runs/*_asu_baseline_seed7/parity.json` 核验原模板全部请求/层时序、U/SLO、cohort、SSD 10ms曲线。只有全部通过才生成最终汇总。

- `input_audit.json`：九份原始 SHA / 指纹、旧参考路径、完整运行配置及冻结源码 SHA。
- `runs/{scenario}_od_baseline_seed{seed}/`：原样 manifest、完整 raw、命令/源码/守恒记录、进度和 warm 预览。
- `execution_logs/`：逐任务 stdout 与队列状态；`local_plan.json` / `remote_plan.json` 为调度计划。
- `progress_summary.json`：最新同步状态；warm 预览不是最终三种子结果。
- `data/od_*.csv`、`data/od_metrics.json`、`validation.json`：所有运行和 parity 完成后由 `summarize_od.py` 生成。

复现单个 OD：

```bash
python run_trial.py --scenario full --seed 7 --policy od_baseline
```

目标目录存在时拒绝覆盖；可通过 `--label` 指定新的结果目录。`--smoke` 只用于每卡第一条请求的启动检查，不能用于正式统计。正式结果在各自进程中排空；不修改旧实验。

## 已完成的 OD 结果

九个 OD 正式案例和三个 ASU 全量回放均已完成。三组 ASU 的全部请求/层指标 SHA 与原模板相同，SSD 10ms 曲线最大误差为 0。OD 共完成 10,944 条请求、76,213,248 个块；全部来源/输入/路径归属审计通过，见 `audit_final.json`。

| 原实验 | NPU 利用率 | TTFT SLO×1.5 |
|---|---:|---:|
| full | 62.5274% | 38.9137% |
| semi | 98.8843% | 95.1804% |
| near35 | 99.2681% | 97.0447% |

表中为主 warm [2,4) 秒、seed7/19/43 等权平均；不是三种子请求合并后的达标率。扩展窗口和全程结果在 `data/od_summary.csv`。
