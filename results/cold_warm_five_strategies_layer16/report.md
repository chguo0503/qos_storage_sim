# Modified strategies: cold and warm six-request results

Both views come from the same fully completed trace. The main NPU utilization metric is computed independently per NPU, then averaged equally across 128 NPUs. Cold uses that NPU's request-0 admission through request-5 completion; warm uses its request-1 admission through request-5 completion. Time after an NPU finishes its own six-request stream is not charged to that NPU.

A shared fleet-window utilization is retained below as a diagnostic for cross-NPU serialization and makespan imbalance; it is not the requested mean per-NPU metric. TTFT remains completion minus admission, so external arrival queue wait is not included.

The shared warm envelope is still not a strict fixed-clock steady-state metric because its earliest request-1 admission depends on the strategy. A future steady-state experiment should use a common burn-in boundary and a fixed wall-clock measurement interval.

Primary SLO: `TTFT <= 2 × compute-only TTFT`.

A strategy meets the requested utilization criterion only when its gain over baseline is at least `+10 pp`.

Provenance: the analyzer verified experiment metadata and paired workload fingerprints, then preserved a separate source fingerprint for every result row. It does not infer behavior compatibility across source versions. Compatibility note: Manual audit: one-command submission, 0.1-us spacing, identical L0-prefetch trigger rule, compatible shared data plane; SSU56/70 freshly rerun with current simulator source. See `comparison_results.json` for file hashes and row sources.

`Best feasible` is the retained demand-weighted shortest-visible-layer-work candidate. Each SSD independently ranks only its currently enqueued pending I/O. The candidate preserves SSD/NPU capacity, placement, the client issue stream, and layer-release constraints, but cannot see another SSD's queue or released commands that have not been submitted. It is not a proof of the mathematical optimum for NPU utilization or TTFT SLO.

## Read once/layer + L0 prefetch minus baseline — cold

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +3.91 pp | +4.88 pp | +4.18 pp | +2.25 pp | -0.58 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +48.96 pp | +32.81 pp | +15.23 pp | +24.74 pp | +17.58 pp |

## Read once/layer + L0 prefetch minus baseline — warm

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +9.66 pp | **+10.48 pp** | +8.66 pp | +5.57 pp | +0.41 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +58.75 pp | +44.22 pp | +27.03 pp | +25.16 pp | +16.25 pp |

## Refresh8 + L0 prefetch minus baseline — cold

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +3.91 pp | +4.88 pp | +4.18 pp | +2.21 pp | -0.75 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +50.26 pp | +32.68 pp | +15.89 pp | +25.00 pp | +17.45 pp |

## Refresh8 + L0 prefetch minus baseline — warm

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +9.71 pp | **+10.46 pp** | +8.62 pp | +5.52 pp | +0.20 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +60.31 pp | +44.06 pp | +27.03 pp | +25.16 pp | +16.09 pp |

## Scheme B + manifest/CIR prefetch minus baseline — cold

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | **+16.77 pp** | **+21.12 pp** | **+11.27 pp** | +4.37 pp | +0.08 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +61.72 pp | +49.09 pp | +20.57 pp | +3.26 pp | +8.20 pp |

## Scheme B + manifest/CIR prefetch minus baseline — warm

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | **+49.36 pp** | **+50.59 pp** | **+24.50 pp** | +9.43 pp | +1.18 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +69.53 pp | +56.56 pp | +24.22 pp | +5.16 pp | +8.91 pp |

## Best feasible reference + L0 prefetch minus baseline — cold

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | **+11.40 pp** | **+15.15 pp** | **+13.92 pp** | +8.16 pp | -0.94 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +71.61 pp | +54.69 pp | +35.68 pp | +19.92 pp | +18.75 pp |

## Best feasible reference + L0 prefetch minus baseline — warm

### Mean per-NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | **+27.39 pp** | **+29.01 pp** | **+23.97 pp** | **+14.42 pp** | +0.71 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---:|---:|---:|---:|---:|---:|
| 16 | +80.78 pp | +61.25 pp | +40.62 pp | +21.56 pp | +17.81 pp |

## 16 layers

### Mean per-NPU utilization

| Strategy / cohort | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|---:|---:|
| Baseline + L0 prefetch — cold | 22.65% | 36.66% | 51.16% | 72.74% | 91.95% |
| Baseline + L0 prefetch — warm | 22.57% | 36.67% | 51.24% | 72.64% | 93.17% |
| Read once/layer + L0 prefetch — cold | 26.56% | 41.55% | 55.35% | 74.99% | 91.37% |
| Read once/layer + L0 prefetch — warm | 32.24% | 47.15% | 59.90% | 78.21% | 93.58% |
| Refresh8 + L0 prefetch — cold | 26.57% | 41.55% | 55.34% | 74.95% | 91.20% |
| Refresh8 + L0 prefetch — warm | 32.28% | 47.12% | 59.87% | 78.16% | 93.37% |
| Scheme B + manifest/CIR prefetch — cold | 39.42% | 57.78% | 62.44% | 77.11% | 92.03% |
| Scheme B + manifest/CIR prefetch — warm | 71.94% | 87.26% | 75.74% | 82.07% | 94.35% |
| Best feasible reference + L0 prefetch — cold | 34.06% | 51.82% | 65.09% | 80.90% | 91.02% |
| Best feasible reference + L0 prefetch — warm | 49.96% | 65.68% | 75.22% | 87.06% | 93.88% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|---:|---:|
| Baseline + L0 prefetch — cold | 0.00% | 25.00% | 50.26% | 75.00% | 80.86% |
| Baseline + L0 prefetch — warm | 0.00% | 25.16% | 50.31% | 74.84% | 81.88% |
| Read once/layer + L0 prefetch — cold | 48.96% | 57.81% | 65.49% | 99.74% | 98.44% |
| Read once/layer + L0 prefetch — warm | 58.75% | 69.38% | 77.34% | 100.00% | 98.12% |
| Refresh8 + L0 prefetch — cold | 50.26% | 57.68% | 66.15% | 100.00% | 98.31% |
| Refresh8 + L0 prefetch — warm | 60.31% | 69.22% | 77.34% | 100.00% | 97.97% |
| Scheme B + manifest/CIR prefetch — cold | 61.72% | 74.09% | 70.83% | 78.26% | 89.06% |
| Scheme B + manifest/CIR prefetch — warm | 69.53% | 81.72% | 74.53% | 80.00% | 90.78% |
| Best feasible reference + L0 prefetch — cold | 71.61% | 79.69% | 85.94% | 94.92% | 99.61% |
| Best feasible reference + L0 prefetch — warm | 80.78% | 86.41% | 90.94% | 96.41% | 99.69% |

### Shared fleet-window utilization (diagnostic only)

This uses the earliest cohort admission and latest stream completion across all 128 NPUs. It exposes cross-NPU serialization, but charges an NPU for time after it has finished its own assigned stream, so it is not used as the requested mean per-NPU metric.

| Strategy / cohort | SSU 16 | SSU 28 | SSU 40 | SSU 56 | SSU 70 |
|---|---:|---:|---:|---:|---:|
| Baseline + L0 prefetch — cold | 22.48% | 36.31% | 50.54% | 68.44% | 83.29% |
| Baseline + L0 prefetch — warm | 22.05% | 35.68% | 49.62% | 65.37% | 77.22% |
| Read once/layer + L0 prefetch — cold | 22.09% | 35.21% | 47.51% | 66.14% | 87.29% |
| Read once/layer + L0 prefetch — warm | 19.41% | 30.93% | 40.71% | 57.07% | 76.18% |
| Refresh8 + L0 prefetch — cold | 22.07% | 35.16% | 47.57% | 66.00% | 87.04% |
| Refresh8 + L0 prefetch — warm | 19.39% | 30.86% | 40.66% | 56.97% | 75.95% |
| Scheme B + manifest/CIR prefetch — cold | 19.89% | 29.99% | 38.88% | 48.17% | 80.27% |
| Scheme B + manifest/CIR prefetch — warm | 16.75% | 25.39% | 33.06% | 41.17% | 69.82% |
| Best feasible reference + L0 prefetch — cold | 20.47% | 31.13% | 40.61% | 46.76% | 64.20% |
| Best feasible reference + L0 prefetch — warm | 17.26% | 26.38% | 34.57% | 39.93% | 55.34% |
