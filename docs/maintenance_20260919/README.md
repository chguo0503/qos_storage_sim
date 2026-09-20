# 2026-09-19 代码拆分与清理

现在根目录不再混放 Python 实现。仿真器在 `simulator/`，输入构造及实验入口在 `inputs/`，回归测试在 `tests/`。原始 `data`、用户提供的 `trace/` 和实验数据继续保留。

## 拆了哪些内容

| 内容 | 新位置 |
|---|---|
| SSU/NPU 数据面、事件、预取、统计 | `simulator/core/` |
| ASU、OD、Once、NewOnce、SLO 池、Scheme B、coflow 等决策 | `simulator/policies/` |
| 周期观测、策略快照和决策接口 | `simulator/telemetry.py`、`simulator/contracts.py` |
| 策略连接、状态转换和写寄存器 | `simulator/adapters/` |
| 对外运行入口 | `simulator/api.py` |
| data 读取、合成输入、manifest 读写 | `inputs/` |
| 原有 `run_*.py` / `sweep_*.py` | `inputs/runners/` |

完整旧模块到新模块映射见 [migration_map.json](evidence/migration_map.json)。核心中原来的三种 Scheme B 控制器、两个 multi-SSU 决策算法已经提取；SLO 候选池策略也从实验目录移到正式策略文件。旧实验中的 `policy.py` 只保留显式导入接口。

各策略提供 `main()` 自检，例如：

```bash
python -m simulator.policies.asu_baseline
python -m simulator.policies.od_baseline
python -m simulator.policies.once
python -m simulator.policies.slo_pool
python -m pytest -q
```

所有 Python 调用使用规范包名，不通过复制模块或 `import *` 生成另一份核心。现有 adapter 仍使用作用域内的补丁，保留安装/恢复顺序；并行仿真使用独立进程。新输入接口可读取合成请求、精确 data 画像或冻结清单，详见 [输入说明](../../inputs/README.md)。

新运行的来源校验递归包含实际执行的包源码。历史结果的 JSON、CSV、图片和输入没有重写；旧源码哈希继续代表旧版本。9月14日的旧根目录清单已移入该日期的维护目录，并标明是历史快照。

## 是否改变实验结果

迁移前先保存工作树源码和一份确定性输入，再运行 ASU、OD、Once。该输入为 **32 NPU、3 SSU、8 层、128 个请求、47,104 个 I/O**，只用于验证重构；不是之前持续欠载或过载研究的新结果。

| 策略 | 迁移前完成时间 ms | 迁移后完成时间 ms | 利用率变化（百分点） |
|---|---:|---:|---:|
| ASU | 70.295418603516 | 70.295418603516 | 0 |
| OD | 72.032372338867 | 72.032372338867 | 0 |
| Once | 72.169326074219 | 72.169326074219 | 0 |

完整请求、层、I/O 完成顺序及其时间签名逐字节一致，两个小窗口的统计也完全相同：[对比证据](evidence/comparison.json)。这些验证窗口是毫秒尺度，不能替代原研究 warm `[2,4)` 秒的结论。

还核对了 400 次新旧控制器决定、108 组基线配置，以及保留核心函数的 AST；见 [数值核对](evidence/policy_extraction_numeric_audit.json)和[提取核对](evidence/policy_extraction_ast_audit.json)。

最终测试与全部策略入口结果见 [tests_final.json](evidence/tests_final.json) 和 [policy_mains.json](evidence/policy_mains.json)。测试覆盖：

- 小型冻结输入的真实时序、新 API 与原 runner 一致。
- 只复制 `simulator/`，在没有 `inputs/`、`results/`、`data` 的临时目录运行三策略。
- 策略异常后恢复全部补丁，公共接口类型保持同一对象。
- manifest 读写及指纹、真实源码覆盖、catalog 显式/隐式输入一致。
- manifest 默认沿用保存的 seed，显式覆盖 seed 不改变冻结输入；拒绝非法物理带宽。

迁移前源码压缩归档在 [pre_refactor_sources.tar.gz](evidence/pre_refactor_sources.tar.gz)，重构前的紧凑输入与期望值在 [tests/fixtures](../../tests/fixtures/)。正常回归不需要临时目录中的迁移快照。详细一次性捕获步骤在 [capture_notes.md](evidence/capture_notes.md)，其中绝对路径描述捕获当时的环境。

## 慢在哪里

小型相同输入、不带 profiler 的真实执行时间：

| 策略 | 迁移前秒 | 迁移后秒 |
|---|---:|---:|
| ASU | 1.216 | 1.248 |
| OD | 1.333 | 1.361 |
| Once | 3.368 | 3.417 |

这是单次小测，存在机器负载波动；此次整理**没有证明仿真提速**。Once 需要给许多 I/O 评估候选 Path。带 profiler 的运行中，`_projection_choices` 调用 47,104 次、累计约 3.269 秒；另有事件队列与 QoS 仲裁开销。累计时间有嵌套，不能简单相加。见 [原始 profile 摘要](evidence/once_profile.txt)。

核心和输入代码不读取 `.learnings/ERRORS.md` 或 `LEARNINGS.md`，这些日志不是测到的仿真 CPU 瓶颈。代码搜索限定为 `simulator inputs tests`、默认测试仅收集 `tests/`，可以避免无意遍历整个实验树；并不改变每个 I/O 的仿真成本。

## 清理范围

- 常读错误/经验记录由约 **95 KB 缩至 3.9 KB**，保留持续适用的规则；历史全文无损压缩归档并验证还原一致。
- 清理项目内可再生成的 Python/pytest 缓存，以及空的辅助目录；精确数量见 [cleanup.json](cleanup.json)。
- 9月14日旧清单移到日期对应的维护目录，避免误当作当前源码清单。
- 原始实验运行日志体积很小，且包含实验执行证据，保留。`results/` 约 4.1 GiB；`.git/` 约 4.9 GiB，主要是可达历史对象。此次没有删除 Git 历史或实验结果来腾空间。

新源码来源记录和测试报告是本次维护证据，不替换任何历史实验审计。
