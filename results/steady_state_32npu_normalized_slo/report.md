# 32-NPU warm/full-load Scheme-B analysis

All utilization values use each strategy's fixed 2,000-ms steady window. The paired-request section additionally restricts TTFT to request IDs that occur inside all three strategies' measurement cohorts.

| SSU | Strategy | NPU util | TTFT SLO | Category-balanced SLO | Admissions/s | SSD util | NPU-link util | Control evals |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | baseline | 32.69% | 25.00% | 25.00% | 96.0 | 90.74% | 13.61% | 0 |
| 6 | current_scheme_b | 32.57% | 9.15% | 9.08% | 91.0 | 90.91% | 13.64% | 10245 |
| 6 | new_scheme_b | 32.76% | 0.00% | 0.00% | 92.0 | 90.78% | 13.62% | 179 |
| 10 | baseline | 57.24% | 75.00% | 75.00% | 160.0 | 95.38% | 23.84% | 0 |
| 10 | current_scheme_b | 56.94% | 50.77% | 50.32% | 156.5 | 95.07% | 23.77% | 10110 |
| 10 | new_scheme_b | 56.69% | 61.51% | 63.12% | 164.0 | 95.29% | 23.82% | 186 |
| 18 | baseline | 93.29% | 79.38% | 79.30% | 258.0 | 86.15% | 38.76% | 0 |
| 18 | current_scheme_b | 92.78% | 82.28% | 81.76% | 254.5 | 85.20% | 38.34% | 13683 |
| 18 | new_scheme_b | 91.14% | 89.00% | 88.66% | 247.0 | 83.67% | 37.65% | 187 |

## Paired-request TTFT cohort

| SSU | Common requests | Strategy | Equal-NPU SLO | Category-balanced SLO | Mean normalized TTFT |
|---:|---:|---|---:|---:|---:|
| 6 | 41 | baseline | 31.94% | 25.00% | 5.327 |
| 6 | 41 | current_scheme_b | 9.86% | 11.17% | 4.626 |
| 6 | 41 | new_scheme_b | 0.00% | 0.00% | 3.386 |
| 10 | 277 | baseline | 74.76% | 75.00% | 3.426 |
| 10 | 277 | current_scheme_b | 50.17% | 50.36% | 2.887 |
| 10 | 277 | new_scheme_b | 60.98% | 61.23% | 2.127 |
| 18 | 478 | baseline | 79.96% | 79.49% | 1.379 |
| 18 | 478 | current_scheme_b | 82.25% | 81.84% | 1.346 |
| 18 | 478 | new_scheme_b | 89.08% | 88.89% | 1.271 |

## New Scheme B deltas

| SSU | Reference | NPU util | TTFT SLO | Category-balanced SLO | Admissions/s |
|---:|---|---:|---:|---:|---:|
| 6 | baseline | +0.06 pp | -25.00 pp | -25.00 pp | -4.17% |
| 6 | current_scheme_b | +0.19 pp | -9.15 pp | -9.08 pp | +1.10% |
| 10 | baseline | -0.55 pp | -13.49 pp | -11.88 pp | +2.50% |
| 10 | current_scheme_b | -0.25 pp | +10.74 pp | +12.81 pp | +4.79% |
| 18 | baseline | -2.16 pp | +9.62 pp | +9.36 pp | -4.26% |
| 18 | current_scheme_b | -1.64 pp | +6.73 pp | +6.89 pp | -2.95% |
