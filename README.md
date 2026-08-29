# QoS + SSD 路由仿真

这个仓库保留 Path 路由刷新频率实验、Scheme B one-shot CIR 分配，以及一个保持物理约束的最优参考候选。

## 保留的策略

前五个策略共用同一套 Static QoS、CIR、Path 布局和 SSD→NPU 数据面；唯一差异是 NPU 如何选择 Path：

| 策略 | Path 选择 | 读取 pressure |
|---|---|---:|
| `baseline` | 所有 I/O 固定进入 Path 0 | 0 |
| `path_rr` | 在请求类别允许的 Path 内确定性轮转 | 0 |
| `layer_once` | 每个 `(request, layer, SSU)` 读取一次后完成该层选路 | 每层一次 |
| `refresh8` | 每规划 8 条 I/O 刷新一次 | 每 8 条 I/O |
| `refresh1` | 每规划一条 I/O 前刷新一次 | 每条 I/O |

`capacity_constrained_oracle` 使用相同 workload、block placement、到达时间、层依赖、SSD 40 GB/s 和 NPU 50 GB/s 上限，以可见请求的全局信息调度。

`scheme_b_prefill` 把一次 admission batch 的 128 个请求 manifest 视为启动前已知，按 ring placement 计算 `NPU × SSU` demand，求满足每盘 40 GB/s、每 NPU 50 GB/s 的等权 max-min grant。每个 NPU 在每块 SSU 上使用一条专属 Path，CIR 只配置一次并复用 16 层；原配对实验中的 0–5 ms 到达时间保留为 NPU launch jitter。该策略不读取 Path pressure，SSD 和 NPU 数据面与前五个策略相同。

主图中的 `Best feasible` 是每个 `(seed, SSU)` 上，五个路由策略、Scheme B 与 oracle 候选中实测最好的结果；它是可执行参考，不是带数学证明的精确理论上界。

## Block placement

每个 token block 用 `(request_id, block_index)` 做 consistent ring hash：

- 同一 token block 的 16 层始终位于同一个 SSU；
- 不同 token block 独立落在 ring 上，允许落到同一个 SSU；
- ring 映射是确定性的，placement seed 只保留为配对结果元数据，不参与映射；
- 所有策略复用完全相同的 placement，便于严格配对比较。

## 数据面

```text
NPU 选择 Path
    ↓
Static QoS/CIR 仲裁
    ↓
SSD 单命令、不可抢占：t = size / 40 GB/s
    ↓
目标 NPU 的独立 FCFS 接收队列：t = size / 50 GB/s
    ↓
block 对计算侧可见
```

CIR 决定命令获得 SSD 服务的机会，不把一条已选中的命令限速为 CIR；获胜命令仍使用 SSD 的 40 GB/s 单命令服务能力。多个 SSD 同时向同一 NPU 返回时，会在该 NPU 的 50 GB/s 接收队列中形成 incast 排队。

## 正式配置与结果

- 128 NPU，16 层
- SSU：`8, 16, 28, 40, 56, 80, 112`
- seed：`42, 43`
- 每个 NPU 独立的 `0–5 ms` 到达延迟
- I/O batch 为 1，相邻发行间隔为 `0.1 µs`
- SS/SL/LS/LL 的 CIR 为 `20/6/8/6 GB/s`
- 每组 Path 数为 `12/4/12/4`，共 8 组

主图纵轴是每个请求的

```text
16 层计算时间 / (16 层计算时间 + 暴露的 I/O stall)
```

再对 128 个请求等权平均。它不等于包含全局 makespan 的 fleet utilization。

| 策略 | SSU 8 | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 80 | SSU 112 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: Path 0 | 29.885% | 44.646% | 56.561% | 65.701% | 74.872% | 84.999% | 91.017% |
| No-refresh Path RR | 33.610% | 49.279% | 61.485% | 70.397% | 78.662% | 86.106% | 90.141% |
| Layer once | 34.063% | 50.883% | 68.203% | 79.220% | 85.758% | 88.993% | 90.756% |
| Refresh every 8 I/Os | 34.065% | 50.887% | 69.925% | 79.126% | 85.714% | 88.937% | 90.584% |
| Refresh every I/O | 34.068% | 50.905% | 67.916% | 80.702% | 86.585% | 89.842% | 90.699% |
| Scheme B (one-shot) | 36.032% | 50.422% | 62.817% | 72.048% | 80.591% | 88.199% | 90.994% |
| Best feasible | 47.646% | 64.225% | 75.737% | 82.171% | 86.947% | 90.125% | 91.687% |

![正式结果](results/routing_refresh_concurrency/01_routing_refresh_finite_issue.png)

## Full-prefill microbatch 端到端实验

`continuous_prefill_experiment.py` 现在使用固定成员、逐层同步的 Full-prefill
microbatch，而不是把 batch 解释成 8 个可独立推进的 resident slot：

- 同一 NPU 在空闲时从 FIFO 冻结最多 8 个请求；有未来请求时等待凑满，trace 尾部允许唯一 final partial batch；
- batch 的第 k 层只有在所有成员第 k 层数据均通过 SSD40 和 NPU50 后才能开始；
- 第 k 层只产生一个 joint batch compute 事件，并在开始时为所有成员预取第 k+1 层；
- 第 15 层结束时所有成员同刻完成，整批释放；运行中到达的新请求不能插入当前 batch；
- batch 层计算时间默认是成员 singleton 层耗时之和，保证计算工作守恒，不假设未经测量的 batch 加速；
- 所有策略共用同一 microbatch membership、ring placement、单命令 SSD40 和每 NPU 独立 NPU50 数据面。

正式 trace 是 128 NPU、batch 8、16 层、28 SSU：每个 NPU 初始 8 个请求，另有
1 个请求随机到达。因此会形成 128 个初始 batch8 和 128 个尾部 singleton；这是有限
Full-prefill trace，不代表长期稳定的 continuous batching。

Scheme B 在 microbatch membership 改变时重新计算 `NPU × SSU` demand-capped
max-min grant；周期版本另外在每 8/4/2/1 个真实 completed batch-layer equivalent
后评估。sticky placement 和固定成员使稳态各层需求相同，因此没有 membership/shape
变化时，更高更新频率通常只增加评估而不改变目标 CIR。

Layer 0 没有上一层计算窗口，是单独的 cold-start burst；仿真让它在真实命令队列中排队，
不会用 `layer bytes / layer compute` 把它伪装成可隐藏的稳态需求。

新结果写入 [`results/full_prefill_microbatch/`](results/full_prefill_microbatch/)，分析器会
拒绝没有 `full_prefill_layer_synchronous_microbatch_v1` 标签的旧 JSON。

| 策略 | Fleet NPU 利用率 | Active-window 利用率 | Makespan | P99 latency |
|---|---:|---:|---:|---:|
| Baseline / Path 0 | 44.943% | 94.322% | 6453.774 ms | 5317.867 ms |
| Path RR | 44.901% | 92.821% | 6459.771 ms | 5391.892 ms |
| Layer once | 44.826% | 93.066% | 6470.581 ms | 5392.078 ms |
| Refresh 8 | 44.921% | 93.240% | 6456.891 ms | 5382.161 ms |
| Refresh 1 | 44.926% | 93.232% | 6456.145 ms | 5382.090 ms |
| Scheme B（所有更新频率） | 43.553% | 93.236% | 6659.748 ms | 5570.189 ms |
| Causal Layer 0 Path0 → Scheme B | 44.943% | 92.648% | 6453.774 ms | 5510.491 ms |
| Full-info EDF reference | 44.773% | 95.679% | 6478.293 ms | 5339.606 ms |

Compute-only fleet 上界是 45.138%；baseline 已达到该上界的 99.567%。原 Scheme B
把稳态 CIR 用于 layer-0 cold start，关键 NPU 119 的首层 barrier 从
22.911 ms 增至 226.243 ms。严格因果混合方案的 Layer 0 固定使用 Path0；每个 NPU
完成自己的上一层 I/O 后，只上报该层实际 bytes-by-SSU，控制器再为下一层计算
max-min CIR。控制输入不含仿真时钟、未来 placement、Path pressure 或 SSD 队列，也没有
global cold fence。它将关键 NPU 119 的 Layer-0 等待恢复到 22.911 ms，因此
fleet 利用率和 makespan 与 baseline 相同；但初始满 batch 的平均 Layer-0 等待为
215.910 ms，因此 P99 仍比 baseline 高 192.624 ms。原因是 baseline 中 Path0 虽只配置
0.208333 GB/s CIR，但作为唯一活跃 Path 可以获得全部 work-conserving 剩余带宽；混合
方案激活 warm 专属 Path 后，这些 Path 的 max-min CIR 接近占满每盘 40 GB/s，尚未完成
Layer 0 的公共 Path0 失去剩余带宽。完整矩阵、cohort、逐层、实际最后完成者和控制开销见
[`report.md`](results/full_prefill_microbatch/report.md) 和
[`causal_full_matrix_report.md`](results/full_prefill_microbatch/causal_full_matrix_report.md)。

![Full-prefill microbatch 结果](results/full_prefill_microbatch/01_full_prefill_microbatch_strategies.png)

`results/continuous_prefill/`、`results/continuous_prefill_capacity_scan/` 和
`results/continuous_batch_control/` 是旧 request-interleaved/slot-replacement 模型的
历史产物，只能用于追溯，不能再作为 Full-prefill microbatch 的硬件结论。

## 连续 batch=1 的跨请求 Layer0 预取

`cross_request_layer0_prefetch=True` 会在请求 k 的最后一层开始计算时，预取已到达的
请求 k+1 的 Layer0；请求 k+1 仍在请求 k 完成时正式 admission。baseline 和 Scheme B
使用相同触发时刻。Scheme B 会先由 ring-hash manifest 计算下一请求的
`NPU × SSU` demand、配置专用 Path/CIR，再提交 Layer0，因此后续请求不再进入
0.208333 GB/s 的公共 cold Path。每个 NPU 的第一次请求仍保留真实冷启动。

正式矩阵为 128 NPU、每 NPU 6 个 batch=1 请求、16/24/56/80 层、SSU 40/56/70。
同一完整 trace 同时输出两个视图：

- cold：请求0 admission 到请求5 completion，包含全部6个请求；
- warm：请求1 admission 到请求5 completion，只统计后5个请求。

TTFT 使用 `completion - admission`，不包含外部 arrival queue wait；利用率先按每个 NPU
计算 `compute busy / cohort window`，再对128个 NPU 等权平均。Scheme B 相对 baseline
的 warm 利用率增益如下，粗体为达到 `+10 pp` 目标的配置：

| Layers | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|
| 16 | **+24.50 pp** | +9.43 pp | +1.18 pp |
| 24 | **+17.62 pp** | +6.51 pp | +1.09 pp |
| 56 | +7.94 pp | +2.72 pp | +0.81 pp |
| 80 | +6.19 pp | +2.19 pp | +0.51 pp |

![Cold/warm 跨请求预取结果](results/cold_warm_modified/01_cold_warm.png)

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 五个路由策略：2 seed × 7 SSU × 5 策略
python routing_refresh_concurrency_experiment.py --workers 10 --rerun

# 受约束 oracle 候选：2 seed × 7 SSU
python capacity_constrained_oracle_experiment.py --workers 10

# Scheme B one-shot：2 seed × 7 SSU
python scheme_b_prefill_experiment.py --workers 10

# 严格校验配对数据并生成 JSON、Markdown 和主图
python analyze_routing_refresh_concurrency.py

# Scheme-B continuous-batch 控制面：40 个严格配对 case
python continuous_batch_experiment.py --workers 10 --rerun
python analyze_continuous_batch.py

# Full-prefill microbatch：5 个路由策略、5 个 Scheme-B 频率和 full-info reference
python continuous_prefill_experiment.py --workers 8
python analyze_continuous_prefill.py

# 配对比较 baseline、原 Scheme B 与 Layer-0 Path0 混合方案
python continuous_prefill_experiment.py \
  --output results/full_prefill_microbatch/causal_layer0_comparison_results.json \
  --workers 3 --case baseline --case scheme_b_once \
  --case scheme_b_after_l0 --rerun
python analyze_scheme_b_cold_start.py

# 对完整 12 策略结果做 cohort、分层、尾部和控制开销分析
python analyze_causal_full_matrix.py

# 均衡六请求 batch=1：3 策略 × 3 SSU × 4 层数
# SSU = 16/24/56
# 截止于最快 NPU 完成第 6 次推理；TTFT SLO 固定统计前 5 次
python six_request_experiment.py --workers 8
python analyze_six_request.py

# 跨请求 Layer0 预取：2 策略 × 3 SSU × 4 层数
python cold_warm_experiment.py --workers 8
python analyze_cold_warm.py
```

runner 会逐 case checkpoint；不带 `--rerun` 时，只在代码、数据和配置指纹完全一致时复用缓存。

## 文件

- `sim.py`：精简离散事件引擎、ring-hash placement、Static QoS 与两级数据面。
- `strategy_profiles.py`：唯一保留的 Static CIR/Path 配置。
- `experiment.py`：结果压缩和原始 Matplotlib 画图函数。
- `routing_refresh_concurrency_experiment.py`：五策略正式矩阵。
- `capacity_constrained_oracle_experiment.py`：保持容量约束的 oracle 候选。
- `analyze_routing_refresh_concurrency.py`：配对校验、聚合、报告和主图。
- `continuous_batch_control.py`：方案 B 的容量受限 grant 与原子 CIR 提交控制器。
- `scheme_b_prefill.py`：batch=1 one-shot demand、max-min grant 和 per-SSU CIR 计划。
- `scheme_b_prefill_experiment.py`：Scheme B 的 14 个端到端配对 case。
- `test_scheme_b_prefill.py`：grant、Path 映射、旧接口等价和数据面约束测试。
- `continuous_batch_experiment.py`：稳态 continuous-batch、batch-size 和 KV 增长 trace。
- `analyze_continuous_batch.py`：控制面覆盖、提交频率、写入量和尾部保护分析。
- `continuous_prefill_workload.py`：batch8、随机尾部请求、NPU jitter 与 sticky ring placement。
- `continuous_prefill_client.py`：五个旧客户端配置和 Scheme B grant/CIR 生成。
- `continuous_batch_sim.py`：固定 Full-prefill microbatch、层 barrier、联合 compute、SSD40、NPU50 的端到端 DES。
- `continuous_prefill_experiment.py`：28 SSU 单点策略矩阵与真实 batch-layer 更新频率扫描。
- `analyze_continuous_prefill.py`：执行模型/配对/守恒校验、compute-only 上界、报告和结果图。
- `analyze_scheme_b_cold_start.py`：分解 Layer 0 与后 15 层等待，输出三策略 cold-start 配对报告。
- `analyze_causal_full_matrix.py`：校验并分解完整 12 策略 Full-prefill 矩阵。
- `six_request_workload.py`：为 128 个 NPU 构造计算量和 KV 量误差低于 0.1% 的六请求均衡压力输入。
- `six_request_experiment.py`：比较 baseline、因果 Scheme B 和 Full-info EDF，并在最快第 6 次完成处裁剪全系统利用率。
- `analyze_six_request.py`：严格检查前五请求 SLO cohort，生成 16/24/56/80 层对比图和报告。
- `cold_warm_metrics.py`：从同一完整 trace 计算 cold/warm TTFT SLO 与每 NPU 利用率。
- `cold_warm_experiment.py`：运行修改后的 baseline/Scheme B 跨请求预取矩阵并逐 case checkpoint。
- `analyze_cold_warm.py`：校验配对输入、输出 cold/warm 差值、报告和结果图。
- `test_continuous_batch_sim.py`：动态 max-min 一致性、wall-clock 更新与端到端守恒测试。
- `test_full_prefill_microbatch_sim.py`：固定成员、层 barrier、整批预取/完成和计算工作守恒测试。
- `data`：请求画像输入。
- `requirements.txt`：运行依赖。
- `results/routing_refresh_concurrency/`：正式原始结果与生成物。

## 模型边界

- Path pressure 读取视为零延迟且无丢失。
- 未建模固定 NAND/协议延迟、channel/die 并行、QD 吞吐曲线和 buffer 反压。
- QoS 使用命令级 CIR/WRR 近似，不对应某款 SSD 的完整微架构。
- Full-prefill 实验固定 batch 成员，未建模 chunked prefill 或 decode iteration-level continuous batching。
- batch compute 使用 singleton 工作量相加的代理；仓库没有真实 batch kernel 的硬件测量。
- 画像 KV 字段实际按 GiB 计算，但为保持历史配对暂未转换为十进制 GB；绝对容量需按该单位假设解读。
- 动态 CIR 原子影响后续命令仲裁，不抢占正在服务的 SSD 命令；没有逐 token 模拟 bucket depth、credit/debt 或配置传播延迟。
