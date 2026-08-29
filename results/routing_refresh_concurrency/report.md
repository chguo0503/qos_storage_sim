# Ring-hash refresh strategy results

All rows use 128 NPUs, 16 layers, 0–5 ms arrival delay and the same block ring-hash placement. One block stays on the same SSU for all layers.
Scheme B computes one max-min NPU×SSU grant for the admitted batch, writes it once, and reuses it across all prefill layers.

Values below are the two-seed mean per-request NPU compute fraction.

| Strategy | SSU 8 | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 80 | SSU 112 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (Path 0) | 29.885% | 44.646% | 56.561% | 65.701% | 74.872% | 84.999% | 91.017% |
| Refresh every 8 I/Os | 34.065% | 50.887% | 69.925% | 79.126% | 85.714% | 88.937% | 90.584% |
| Scheme B (one-shot) | 36.032% | 50.422% | 62.817% | 72.048% | 80.591% | 88.199% | 90.994% |
| Best feasible reference | 47.646% | 64.225% | 75.737% | 82.171% | 86.947% | 90.125% | 91.687% |

## Scheme B paired deltas

| SSU | vs baseline | vs Refresh8 | SS class vs Refresh8 | saturated SSUs |
|---:|---:|---:|---:|---:|
| 8 | +6.146 pp | +1.967 pp | -5.947 pp | 8.0/8 |
| 16 | +5.776 pp | -0.465 pp | -10.370 pp | 16.0/16 |
| 28 | +6.257 pp | -7.107 pp | -39.995 pp | 28.0/28 |
| 40 | +6.347 pp | -7.078 pp | -35.414 pp | 40.0/40 |
| 56 | +5.719 pp | -5.123 pp | -22.307 pp | 56.0/56 |
| 80 | +3.201 pp | -0.738 pp | -3.172 pp | 4.5/80 |
| 112 | -0.023 pp | +0.411 pp | +1.045 pp | 0.0/112 |

Scheme B uses one per-NPU Path on every SSU and configures the max-min grant once before the admitted batch launches. The 0–5 ms arrival vector is retained as launch jitter; no Path-pressure table is read during execution.

At 8 SSUs, equal max-min sharing improves the mean under extreme contention. From 28 through 56 SSUs every disk remains saturated, and equal flow fairness removes the original 20/40 GB/s protection for the latency-sensitive SS class; the resulting last-block wait makes Scheme B slower than pressure-aware routing. The executable oracle reference has the highest measured mean at every SSU count.

`Best feasible reference` is an executable capacity-preserving oracle candidate, not a proven exact optimum.
