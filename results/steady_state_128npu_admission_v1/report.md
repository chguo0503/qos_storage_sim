# 128-NPU warm/full-load admission experiment

All rows use seed 42, 16 layers, batch 1, four warmup completions per NPU, 500-ms settle, and the same 2,000-ms measurement window. Admission 25/50 ms denotes event-gated minimum spacing, not a periodic timer.

Complete: `false`; selected complete: `true`; source stable: `true`; config stable: `true`.

| SSU | Strategy | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Requests | Evals | Commits | Path writes | Wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 40 | baseline | 48.06% | 48.29% | 51.60% | 50.00% | 50.00% | 0.00% | 100.00% | 0.00% | 100.00% | 1152 | 0 | 0 | 0 | 909.2s |
| 40 | current_scheme_b | 43.52% | 47.35% | 51.57% | 50.64% | 50.74% | 0.00% | 98.94% | 3.57% | 99.32% | 1149 | 47492 | 2902 | 4041483 | 1459.8s |
| 40 | admission_25ms | 40.01% | 44.97% | 51.57% | 82.45% | 82.87% | 36.59% | 100.00% | 95.22% | 100.00% | 1144 | 166 | 166 | 844631 | 1150.9s |
| 40 | admission_50ms | 35.18% | 44.29% | 51.70% | 75.23% | 75.57% | 24.55% | 99.65% | 76.55% | 100.00% | 1138 | 85 | 85 | 430120 | 1174.1s |
| 70 | baseline | 84.57% | 84.79% | 88.93% | 77.33% | 77.41% | 12.11% | 100.00% | 100.00% | 100.00% | 1992 | 0 | 0 | 0 | 1555.6s |
| 70 | current_scheme_b | 83.82% | 85.79% | 89.06% | 81.94% | 82.15% | 26.58% | 100.00% | 100.00% | 100.00% | 1950 | 53902 | 3344 | 6019200 | 2353.2s |
| 70 | admission_25ms | 87.68% | 89.19% | 90.85% | 98.88% | 98.89% | 95.51% | 100.00% | 100.00% | 100.00% | 1977 | 131 | 131 | 611236 | 2289.9s |
| 70 | admission_50ms | 86.76% | 88.91% | 90.78% | 92.26% | 92.31% | 69.29% | 100.00% | 100.00% | 100.00% | 1976 | 65 | 65 | 362115 | 2162.0s |
