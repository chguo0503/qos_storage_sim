# 32-NPU deadline/barrier Scheme B V3

真实 `data`、batch=1、16 层复用固定 placement；所有 V3 行使用周期控制，且设置最小间隔与周期相同。

| SSU | Case | Interval | Margin | Wait floor | NPU util | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Controls | Commits | Writes | Wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | deadline_v3_25ms | 25 ms | 1.15 | 0.65 | 56.68% | 74.92% | 74.92% | 24.39% | 98.85% | 77.78% | 100.00% | 149 | 149 | 46945 | 194.6s |

## 与同输入 baseline 的差值

| SSU | Case | Paired | NPU util | Equal-NPU SLO | Request SLO |
|---:|---|:---:|---:|---:|---:|
| 10 | deadline_v3_25ms | yes | -0.56 pp | -0.08 pp | -0.08 pp |

`Paired=yes` 要求 assignment/workload/placement/trace/simulator input 五个指纹全部相同。
