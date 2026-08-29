# Modified strategies: cold and warm six-request results

Both views come from the same fully completed trace. Cold uses each NPU's request-0 admission through request-5 completion; warm starts at request-1 admission. TTFT is completion minus admission, so external arrival queue wait is not included.

Primary SLO: `TTFT <= 2 × compute-only TTFT`.

A strategy meets the requested utilization criterion only when its gain over baseline is at least `+10 pp`.

## Scheme B minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|
| 16 | **+11.27 pp** | +4.37 pp | +0.08 pp |
| 24 | +8.52 pp | +3.34 pp | +0.23 pp |
| 56 | +5.00 pp | +1.82 pp | +0.22 pp |
| 80 | +4.29 pp | +1.54 pp | +0.02 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|
| 16 | +20.57 pp | +3.26 pp | +8.20 pp |
| 24 | +16.15 pp | +0.78 pp | +7.81 pp |
| 56 | +10.03 pp | -3.12 pp | +9.11 pp |
| 80 | +7.94 pp | -3.78 pp | +10.55 pp |

## Scheme B minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|
| 16 | **+24.50 pp** | +9.43 pp | +1.18 pp |
| 24 | **+17.62 pp** | +6.51 pp | +1.09 pp |
| 56 | +7.94 pp | +2.72 pp | +0.81 pp |
| 80 | +6.19 pp | +2.19 pp | +0.51 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|
| 16 | +24.22 pp | +5.16 pp | +8.91 pp |
| 24 | +18.75 pp | +2.03 pp | +8.75 pp |
| 56 | +10.78 pp | -2.34 pp | +10.16 pp |
| 80 | +8.28 pp | -3.12 pp | +11.88 pp |

## 16 layers

### Mean NPU utilization

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 51.16% | 72.74% | 91.95% |
| Baseline + L0 prefetch — warm | 51.24% | 72.64% | 93.17% |
| Scheme B + manifest/CIR prefetch — cold | 62.44% | 77.11% | 92.03% |
| Scheme B + manifest/CIR prefetch — warm | 75.74% | 82.07% | 94.35% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 50.26% | 75.00% | 80.86% |
| Baseline + L0 prefetch — warm | 50.31% | 74.84% | 81.88% |
| Scheme B + manifest/CIR prefetch — cold | 70.83% | 78.26% | 89.06% |
| Scheme B + manifest/CIR prefetch — warm | 74.53% | 80.00% | 90.78% |


## 24 layers

### Mean NPU utilization

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 51.13% | 72.80% | 91.90% |
| Baseline + L0 prefetch — warm | 51.24% | 72.69% | 92.95% |
| Scheme B + manifest/CIR prefetch — cold | 59.64% | 76.14% | 92.13% |
| Scheme B + manifest/CIR prefetch — warm | 68.86% | 79.20% | 94.05% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 50.00% | 75.00% | 78.39% |
| Baseline + L0 prefetch — warm | 50.00% | 74.84% | 78.91% |
| Scheme B + manifest/CIR prefetch — cold | 66.15% | 75.78% | 86.20% |
| Scheme B + manifest/CIR prefetch — warm | 68.75% | 76.88% | 87.66% |


## 56 layers

### Mean NPU utilization

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 51.08% | 72.87% | 91.83% |
| Baseline + L0 prefetch — warm | 51.24% | 72.76% | 92.70% |
| Scheme B + manifest/CIR prefetch — cold | 56.08% | 74.68% | 92.05% |
| Scheme B + manifest/CIR prefetch — warm | 59.18% | 75.48% | 93.51% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 50.00% | 75.00% | 76.69% |
| Baseline + L0 prefetch — warm | 50.00% | 74.84% | 76.88% |
| Scheme B + manifest/CIR prefetch — cold | 60.03% | 71.88% | 85.81% |
| Scheme B + manifest/CIR prefetch — warm | 60.78% | 72.50% | 87.03% |


## 80 layers

### Mean NPU utilization

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 51.07% | 72.88% | 91.83% |
| Baseline + L0 prefetch — warm | 51.24% | 72.78% | 92.66% |
| Scheme B + manifest/CIR prefetch — cold | 55.37% | 74.42% | 91.85% |
| Scheme B + manifest/CIR prefetch — warm | 57.44% | 74.97% | 93.17% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 50.00% | 75.00% | 75.91% |
| Baseline + L0 prefetch — warm | 50.00% | 74.84% | 75.94% |
| Scheme B + manifest/CIR prefetch — cold | 57.94% | 71.22% | 86.46% |
| Scheme B + manifest/CIR prefetch — warm | 58.28% | 71.72% | 87.81% |
