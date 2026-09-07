# 原生 Deadline + Fluid 分配：单 SSU 集中与到达顺序反事实

本补充直接调用现有 `run_multi_ssu_stall_experiments.run_case(strategy="deadline", assignment="fluid")`，没有修改模拟器、控制器或分配算法。所有请求参数、物理 placement 与输入 fingerprint 均来自冻结 manifest；结果通过全部 invariants、输入不变性和源文件 hash 检查。

| 冻结输入 | 策略 | 1000–2000 ms NPU 利用率 | 2000–3000 ms NPU 利用率 | Admission SLO |
|---|---|---:|---:|---:|
| strong32_local，全部 t=0 | Deadline + Fluid | 4.2306% | 11.8828% | 68.1111% |
| raw32_local，全部 t=0 | Baseline native | 86.9938% | 86.7386% | 88.0117% |
| raw32_local，全部 t=0 | Deadline + Fluid | 10.9408% | 12.9890% | 53.1067% |
| raw32_local_interleaved_arrivals | Baseline native | 86.9938% | 86.7386% | 88.0117% |
| raw32_local_interleaved_arrivals | Deadline + Fluid | 99.1064% | 94.8309% | 99.3056% |

表中每个窗口都验证了 **32 张 NPU 全程 active**：每张卡始终处于已 admission 请求的执行或等待 I/O 区间，低利用率不是有限输入耗尽后的空闲。Admission SLO 使用全部完成请求，判定为 `completion − admission <= 1.5 × 8 × per_layer_compute_ms`；不包含 admission 前排队时间。

## 确认的损失机制

模拟器对同时到达请求按 `(arrival_time_ms, original_npu_id, request_id)` 排序，并以原 NPU ID 作为 arrival 事件的资源键。因此原输入先完整处理原 NPU0 的 backlog，再处理 NPU1，依此类推。Fluid 改变执行 NPU，但不移动 SSD 上的块；与每张执行 NPU 的 FCFS 请求队列结合后，同一 SSU 的请求会同时占据许多执行队列的前端。

strong32_local 的首个窗口中，**全部 32000 NPU-ms 都在运行原 NPU3 的长请求，物理读取只涉及 SSU0**。其中 warm-layer I/O stall 为 30646.2195 NPU-ms，Layer0 stall 为 **0**。因此这个窗口已经确认的主因是持续的单 SSU 争用，而不是跨请求 Layer0 切换。第二窗口仍有 28890.5552 NPU-ms 来自 SSU0，其余才逐步进入 SSU1、SSU2。[逐 SSU 与停顿类型证据](../native_assignment/strong32_fluid_phase_analysis.json)

raw32_local 的两个窗口也都将全部 32000 NPU-ms 集中在 SSU0；首窗是在全部 32 张执行卡上处理原 NPU0、NPU1 的短请求。此时短请求每卡理想需求约 11.4251 GiB/s，单 SSU 稳态容量比例 `40/(32×11.4251) ≈ 10.9408%`，与实测利用率相符。这个比例用于解释该集中相位，不是忽略窗口边界在途数据的任意有限窗口严格上界。[原始 raw 相位证据](../native_assignment/raw32_fluid_phase_analysis.json)

## 到达顺序反事实

新输入把同一批 2736 个请求按 `(generation, original_npu_id)` 排序，再令 `arrival_ms = rank × 1e-6`，最大到达时间仅 **0.002735 ms，即 2.735 µs**。逐请求核对了请求数、ID、原 NPU 绑定、C/V、placement、每条 lane 内顺序和所有非 arrival load 字段不变。两个 load arrival 字段与实际 arrival 时间发生变化，生成了新的 fingerprint；因此新旧输入不能混称为同一条 trace。[变更与保留字段证明](../native_assignment/interleaved_arrivals_provenance.json)

同一新输入内，Deadline + Fluid 的两窗口达到 99.1064% / 94.8309%，8 个 SSU 各自对应约 3800–4334 active NPU-ms；原输入却全部集中在 SSU0。Baseline 新旧两窗只相差约 ±0.0000007 个百分点，SLO 不变。Fluid makespan 从 15723.3400 ms 降到 4966.8251 ms，Baseline 从 5779.778333 ms 变为 5779.778361 ms。[四格配对结果与分 SSU 证据](../native_assignment/arrival_tie_comparison.json)

这支持的结论是：该 Fluid 实现在大量初始 backlog 和特定 arrival tie 顺序下，会形成很差的请求队首分布；改动初始到达顺序及后续调度反馈可以消除这一具体负例。它不证明 Fluid 在一般线上到达中总是差，也不证明微小时间扰动在任意输入中都安全。

## 指标边界与运行成本

分析文件中的 `read_lifetime_overlap_ms_by_ssu` 是每层 **I/O release 到 HBM ready** 区间与观察窗的交叠长度之和，包含排队、服务和链路等待，**不能当作 SSD busy time 或 SSD 利用率**。本输入每层只位于一个 SSU，因此某 SSU 的该值为零，才可据此说明没有该 SSU 的层 I/O 生命周期与窗口交叠。strong 首窗和 raw 原输入两窗的 SSU1–7 均为零。

Strong 总运行墙钟为 1728.9444 s，其中累计 assignment 回调墙钟为 1152.8160 s；raw 原输入分别为 273.2947 s、14.8112 s。回调字段由 `perf_counter` 记录，可能包含并行运行时被调度出去的时间，不能严格称作 CPU 时间。这些开销没有计入模拟数据面时间，也不能据此承诺实际硬件部署速度。

结果由 [独立包装脚本](../run_native_assignment_probe.py) 生成；原 stress runner 和核心源码保持冻结。

四格到达反事实已统一使用本机 Python 3.10 的 native Baseline / Deadline + Fluid。对原始 raw32_local 的精准桥接还验证了：本机 native、本机 5 ms shared Baseline、远端 Python 3.14 的 screen Baseline，其完整 2736 条 request metrics、全部 microbatch/layer 时序及 makespan 均逐项相等。本机两包装器的窗口和 SLO 也完全相等；跨 Python 环境仅第二窗口利用率汇总相差约 2.22e-16，属于浮点求和舍入，不是调度效果。[递归比较证据](../native_assignment/raw32_baseline_wrapper_bridge.json)
