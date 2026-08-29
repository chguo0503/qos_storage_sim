# Modified strategies: cold and warm six-request results

Both views come from the same fully completed trace. Cold uses each NPU's request-0 admission through request-5 completion; warm starts at request-1 admission. TTFT is completion minus admission, so external arrival queue wait is not included.

Primary SLO: `TTFT <= 2 × compute-only TTFT`.

A strategy meets the requested utilization criterion only when its gain over baseline is at least `+10 pp`.

## Refresh8 + L0 prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 40 |
|---:|---:|
| 16 | +4.18 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 |
|---:|---:|
| 16 | +15.89 pp |

## Refresh8 + L0 prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 40 |
|---:|---:|
| 16 | +8.62 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 |
|---:|---:|
| 16 | +27.03 pp |

## Scheme B + manifest/CIR prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 40 |
|---:|---:|
| 16 | **+11.27 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 |
|---:|---:|
| 16 | +20.57 pp |

## Scheme B + manifest/CIR prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 40 |
|---:|---:|
| 16 | **+24.50 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 40 |
|---:|---:|
| 16 | +24.22 pp |

## 16 layers

### Mean NPU utilization

| Strategy / cohort | SSU 40 |
|---|---:|
| Baseline + L0 prefetch — cold | 51.16% |
| Baseline + L0 prefetch — warm | 51.24% |
| Refresh8 + L0 prefetch — cold | 55.34% |
| Refresh8 + L0 prefetch — warm | 59.87% |
| Scheme B + manifest/CIR prefetch — cold | 62.44% |
| Scheme B + manifest/CIR prefetch — warm | 75.74% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 40 |
|---|---:|
| Baseline + L0 prefetch — cold | 50.26% |
| Baseline + L0 prefetch — warm | 50.31% |
| Refresh8 + L0 prefetch — cold | 66.15% |
| Refresh8 + L0 prefetch — warm | 77.34% |
| Scheme B + manifest/CIR prefetch — cold | 70.83% |
| Scheme B + manifest/CIR prefetch — warm | 74.53% |
