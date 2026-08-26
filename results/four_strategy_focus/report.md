# 四策略正式对比

固定合同：128 NPU、16 层、SSU=40/56/80、每 NPU 0–5 ms 随机到达延迟；三个可运行策略共享 workload、block placement 和 SSD40→NPU50 数据面。

## 四条曲线分别是什么

1. **Baseline (NPU-RR)**：每块 SSD 在 NPU 队列之间轮转，不使用 QoS Path/CIR。
2. **Static CIR before (20/4/12/4)**：当前 `qos_static_cir`，SS/SL/LS/LL 的每盘 CIR 总预算为 20/4/12/4 GB/s，组内 Path 数为 12/4/12/4，Path 压力每 8 条读取一次。
3. **Static CIR after (20/6/8/6)**：调优后的固定配置；Path 数仍为 12/4/12/4，只把类别 CIR 调成 20/6/8/6 GB/s。
4. **Ideal fluid bound**：删除所有跨 NPU SSD 竞争，并允许 SSD 与 NPU fluid cut-through 的不可运行 request 指标上界；它不是硬件策略，没有联合 makespan，所以没有 fleet 数值。

## 结果

每格为 `request compute fraction / fleet compute utilization`。

| SSU | Baseline | Static before | Static after | Ideal fluid bound |
|---:|---:|---:|---:|---:|
| 40 | 72.869% / 14.553% | 81.952% / 14.396% | 82.436% / 14.422% | 91.232% / N/A |
| 56 | 81.268% / 14.593% | 86.870% / 14.493% | 87.030% / 14.522% | 91.232% / N/A |
| 80 | 88.286% / 14.620% | 89.844% / 14.584% | 90.114% / 14.591% | 91.232% / N/A |

图片只画用户指定的四条 request compute-fraction 曲线。它不是 fleet 指标；fluid bound 没有合法的 fleet/makespan。

![Four strategy overview](01_strategy_overview.png)

无竞争上界的完整定义见 [IDEAL_NO_CONTENTION_BOUND_CN.md](../../doc/IDEAL_NO_CONTENTION_BOUND_CN.md)。
