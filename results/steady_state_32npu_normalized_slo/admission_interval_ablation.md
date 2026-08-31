# 32-NPU SLO-admission minimum-interval ablation

Only `min_interval_ms` changes across rows. Control remains batch-membership-event driven with `on_batch_boundary=True`; these are minimum spacings, not periodic timers.

Complete: `true`; source stable: `true`; config stable: `true`.

Controller: target=0.52, required=0.50, background reserve=5.00%, SSD cap=40 GB/s, NPU cap=50 GB/s.

| SSU | Min interval | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Requests | Evaluations | Commits | Path writes | Wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 25 ms | 50.38% | 53.25% | 57.00% | 84.57% | 84.59% | 43.75% | 100.00% | 95.06% | 100.00% | 318 | 154 | 151 | 48010 | 195.8s |
| 10 | 50 ms | 47.45% | 52.76% | 57.13% | 82.77% | 82.61% | 37.97% | 100.00% | 91.36% | 100.00% | 322 | 78 | 78 | 24650 | 199.2s |
| 10 | 100 ms | 47.26% | 49.81% | 56.89% | 76.03% | 76.10% | 3.85% | 100.00% | 98.73% | 100.00% | 318 | 38 | 38 | 11850 | 193.3s |
| 18 | 25 ms | 92.40% | 92.66% | 93.38% | 98.85% | 98.85% | 95.42% | 100.00% | 100.00% | 100.00% | 523 | 126 | 124 | 37158 | 296.1s |
| 18 | 50 ms | 92.12% | 92.71% | 94.23% | 89.60% | 89.58% | 58.78% | 100.00% | 100.00% | 100.00% | 518 | 65 | 65 | 26498 | 294.5s |
| 18 | 100 ms | 91.46% | 92.10% | 93.26% | 86.49% | 86.63% | 46.09% | 100.00% | 100.00% | 100.00% | 516 | 33 | 33 | 15265 | 295.5s |

## Delta from 25-ms row

Percentage points; only SSUs with a completed 25-ms reference are shown.

| SSU | Interval | Mean NPU util | Equal-NPU SLO | Request SLO |
|---:|---:|---:|---:|---:|
| 10 | 25 ms | +0.00 pp | +0.00 pp | +0.00 pp |
| 10 | 50 ms | +0.13 pp | -1.80 pp | -1.98 pp |
| 10 | 100 ms | -0.11 pp | -8.53 pp | -8.49 pp |
| 18 | 25 ms | +0.00 pp | +0.00 pp | +0.00 pp |
| 18 | 50 ms | +0.85 pp | -9.25 pp | -9.28 pp |
| 18 | 100 ms | -0.11 pp | -12.36 pp | -12.22 pp |
