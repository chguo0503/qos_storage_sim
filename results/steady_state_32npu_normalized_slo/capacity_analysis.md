# 32-NPU steady-state capacity/input audit

本报告只分析真实 trace、ring placement、SSD40 和 NPU50 容量；不运行或比较任何调度策略。

## 四类请求与 NPU50

| Category | Raw demand | 2× target | Compute/layer | Work/layer | NPU50 raw bottleneck | NPU50 util ceiling |
|---|---:|---:|---:|---:|---:|---:|
| SS | 83.346 GB/s | 41.673 GB/s | 1.288 ms | 0.107338 GB | yes | 59.99% |
| SL | 13.812 GB/s | 6.906 GB/s | 7.729 ms | 0.106750 GB | no | 100.00% |
| LS | 28.831 GB/s | 14.415 GB/s | 6.695 ms | 0.193024 GB | no | 100.00% |
| LL | 14.712 GB/s | 7.356 GB/s | 13.097 ms | 0.192688 GB | no | 100.00% |

在无限 SSD、理想 warm pipeline 下，NPU50 给出的 fleet mean NPU utilization 上限为 `97.105%`。

## Fleet average 和 placement

长期 fleet raw demand 为 `666.236 GB/s`；raw knee=`16.656` SSU，2× knee=`8.328` SSU。

| SSU | Long-run max raw/SSU | Long-run max 2× target | Raw feasible | 2× feasible | Synchronized feasible sequences | Dephased 2× infeasible |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 121.866 GB/s | 60.933 GB/s | no | no | 0/32 | 100.00% |
| 10 | 70.349 GB/s | 35.175 GB/s | no | yes | 0/32 | 16.31% |
| 18 | 39.187 GB/s | 19.593 GB/s | yes | yes | 32/32 | 0.00% |

## 口径和限制

- 长期 placement 负载对每个 NPU 使用完整32请求周期的 `Σwork/Σcompute`，再跨 NPU 求和。
- synchronized 行逐一检查32个 sequence 的每盘 `ΣD/2 <= 40` 和每 NPU `ΣD/2 <= 50`。
- dephased 行是固定 seed `20260830` 的 `20000` 次独立 phase 抽样，不是 DES 稳态结果；真实 residence time 还会受到 SSD/NPU stall 和策略影响。
- 所有 SSU 共用同一组 phase indices，fingerprint=`826e473636d92a390cc984025e9c4b7a345a52b369126050916caf74f6d3fff3`。
- `2×` 指 processing TTFT 的流体带宽条件，不包含外部 arrival-to-admission 排队。

Source fingerprint: `0f0b3dcd684ae16ee990f540e7a36e7f80f31841c925db49c1e83ccdad712d15`
