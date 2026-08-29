# Modified strategies: cold and warm six-request results

Both views come from the same fully completed trace. Cold uses each NPU's request-0 admission through request-5 completion; warm starts at request-1 admission. TTFT is completion minus admission, so external arrival queue wait is not included.

Primary SLO: `TTFT <= 2 × compute-only TTFT`.

A strategy meets the requested utilization criterion only when its gain over baseline is at least `+10 pp`.

Provenance: the analyzer verified experiment metadata and paired workload fingerprints, then preserved a separate source fingerprint for every result row. It does not infer behavior compatibility across source versions. Compatibility note: Manual audit: one-command submission, 0.1-us spacing, identical L0-prefetch trigger rule, compatible shared data plane. See `comparison_results.json` for file hashes and row sources.

`Best feasible` is the retained demand-weighted shortest-visible-layer-work candidate. Each SSD independently ranks only its currently enqueued pending I/O. The candidate preserves SSD/NPU capacity, placement, the client issue stream, and layer-release constraints, but cannot see another SSD's queue or released commands that have not been submitted. It is not a proof of the mathematical optimum for NPU utilization or TTFT SLO.

## Read once/layer + L0 prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +3.91 pp | +4.88 pp | +4.18 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +48.96 pp | +32.81 pp | +15.23 pp |

## Read once/layer + L0 prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +9.66 pp | **+10.48 pp** | +8.66 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +58.75 pp | +44.22 pp | +27.03 pp |

## Refresh8 + L0 prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +3.91 pp | +4.88 pp | +4.18 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +50.26 pp | +32.68 pp | +15.89 pp |

## Refresh8 + L0 prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +9.71 pp | **+10.46 pp** | +8.62 pp |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +60.31 pp | +44.06 pp | +27.03 pp |

## Scheme B + manifest/CIR prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | **+16.77 pp** | **+21.12 pp** | **+11.27 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +61.72 pp | +49.09 pp | +20.57 pp |

## Scheme B + manifest/CIR prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | **+49.36 pp** | **+50.59 pp** | **+24.50 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +69.53 pp | +56.56 pp | +24.22 pp |

## Best feasible reference + L0 prefetch minus baseline — cold

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | **+11.40 pp** | **+15.15 pp** | **+13.92 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +71.61 pp | +54.69 pp | +35.68 pp |

## Best feasible reference + L0 prefetch minus baseline — warm

### Mean NPU utilization delta

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | **+27.39 pp** | **+29.01 pp** | **+23.97 pp** |

### TTFT SLO attainment delta @ 2x

| Layers | SSU 16 | SSU 28 | SSU 40 |
|---:|---:|---:|---:|
| 16 | +80.78 pp | +61.25 pp | +40.62 pp |

## 16 layers

### Mean NPU utilization

| Strategy / cohort | SSU 16 | SSU 28 | SSU 40 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 22.65% | 36.66% | 51.16% |
| Baseline + L0 prefetch — warm | 22.57% | 36.67% | 51.24% |
| Read once/layer + L0 prefetch — cold | 26.56% | 41.55% | 55.35% |
| Read once/layer + L0 prefetch — warm | 32.24% | 47.15% | 59.90% |
| Refresh8 + L0 prefetch — cold | 26.57% | 41.55% | 55.34% |
| Refresh8 + L0 prefetch — warm | 32.28% | 47.12% | 59.87% |
| Scheme B + manifest/CIR prefetch — cold | 39.42% | 57.78% | 62.44% |
| Scheme B + manifest/CIR prefetch — warm | 71.94% | 87.26% | 75.74% |
| Best feasible reference + L0 prefetch — cold | 34.06% | 51.82% | 65.09% |
| Best feasible reference + L0 prefetch — warm | 49.96% | 65.68% | 75.22% |

### TTFT SLO attainment @ 2x

| Strategy / cohort | SSU 16 | SSU 28 | SSU 40 |
|---|---:|---:|---:|
| Baseline + L0 prefetch — cold | 0.00% | 25.00% | 50.26% |
| Baseline + L0 prefetch — warm | 0.00% | 25.16% | 50.31% |
| Read once/layer + L0 prefetch — cold | 48.96% | 57.81% | 65.49% |
| Read once/layer + L0 prefetch — warm | 58.75% | 69.38% | 77.34% |
| Refresh8 + L0 prefetch — cold | 50.26% | 57.68% | 66.15% |
| Refresh8 + L0 prefetch — warm | 60.31% | 69.22% | 77.34% |
| Scheme B + manifest/CIR prefetch — cold | 61.72% | 74.09% | 70.83% |
| Scheme B + manifest/CIR prefetch — warm | 69.53% | 81.72% | 74.53% |
| Best feasible reference + L0 prefetch — cold | 71.61% | 79.69% | 85.94% |
| Best feasible reference + L0 prefetch — warm | 80.78% | 86.41% | 90.94% |
