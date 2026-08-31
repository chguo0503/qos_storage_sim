# 128-NPU vs 32-NPU warm/full-load scale analysis

Verdict: `policy_promising_but_queue_drift_requires_validation`. Completed 128-NPU rows: `16/16`. This report is safe to regenerate while the runner checkpoints.

Capacity-equivalent points use `SSU × 32 / NPU`: 128×24 ↔ 32×6, 128×40 ↔ 32×10, and 128×70 lies between 32×17 and 32×18.

## 128-NPU completed rows

| Source | SSU | Strategy | NPU util (min/p10/mean) | Equal-NPU / request SLO | Equal-category SLO | SSD mean/max | SSD >=99% | NPU-link mean/max | Queue start→end | Queue check |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 128_v1_supplemental | 24 | baseline | 30.21%/30.48%/32.76% | 25.00%/25.00% | 25.00% | 90.94%/100.00% | 1 | 13.64%/14.61% | 5580→6904 | compatible_not_proven |
| 128_v1_supplemental | 24 | current_scheme_b | 27.83%/29.94%/32.38% | 17.01%/17.40% | 17.68% | 89.86%/100.00% | 1 | 13.48%/15.37% | 7943→8001 | compatible_not_proven |
| 128_v2 | 24 | admission_v2_25ms | 16.66%/21.18%/32.51% | 52.10%/53.88% | 52.18% | 90.37%/100.00% | 1 | 13.55%/21.12% | 3525→3525 | compatible_not_proven |
| 128_adaptive_v2_1 | 24 | adaptive_v2_1_25ms | 16.66%/21.18%/32.51% | 52.10%/53.88% | 52.18% | 90.37%/100.00% | 1 | 13.55%/21.12% | 3525→3525 | compatible_not_proven |
| 128_v1 | 40 | baseline | 48.06%/48.29%/51.60% | 50.00%/50.00% | 50.00% | 85.91%/100.00% | 2 | 21.48%/22.23% | 3667→4864 | inconclusive |
| 128_v1 | 40 | current_scheme_b | 43.52%/47.35%/51.57% | 50.64%/50.74% | 50.36% | 85.86%/100.00% | 1 | 21.47%/24.04% | 4610→1708 | drift_detected |
| 128_v1 | 40 | admission_25ms | 40.01%/44.97%/51.57% | 82.45%/82.87% | 82.62% | 85.89%/99.79% | 2 | 21.47%/24.83% | 3459→3506 | compatible_not_proven |
| 128_v1 | 40 | admission_50ms | 35.18%/44.29%/51.70% | 75.23%/75.57% | 75.03% | 86.03%/99.98% | 2 | 21.51%/25.52% | 4071→5099 | inconclusive |
| 128_v2 | 40 | admission_v2_25ms | 43.95%/47.75%/51.66% | 79.20%/79.81% | 78.94% | 85.99%/99.81% | 2 | 21.50%/24.36% | 1974→10865 | drift_detected |
| 128_adaptive_v2_1 | 40 | adaptive_v2_1_25ms | 40.01%/44.97%/51.57% | 82.45%/82.87% | 82.62% | 85.89%/99.79% | 2 | 21.47%/24.83% | 3459→3506 | compatible_not_proven |
| 128_v1 | 70 | baseline | 84.57%/84.79%/88.93% | 77.33%/77.41% | 78.03% | 84.93%/97.82% | 0 | 37.15%/38.69% | 13330→3022 | drift_detected |
| 128_v1 | 70 | current_scheme_b | 83.82%/85.79%/89.06% | 81.94%/82.15% | 81.18% | 84.08%/96.98% | 0 | 36.79%/38.97% | 1956→34 | drift_detected |
| 128_v1 | 70 | admission_25ms | 87.68%/89.19%/90.85% | 98.88%/98.89% | 98.84% | 86.19%/99.24% | 1 | 37.71%/38.82% | 3475→2534 | inconclusive |
| 128_v1 | 70 | admission_50ms | 86.76%/88.91%/90.78% | 92.26%/92.31% | 92.23% | 86.21%/99.29% | 1 | 37.72%/38.98% | 5529→5762 | compatible_not_proven |
| 128_v2 | 70 | admission_v2_25ms | 88.16%/89.23%/90.89% | 98.55%/98.57% | 98.49% | 86.14%/99.18% | 1 | 37.69%/38.91% | 1983→575 | drift_detected |
| 128_adaptive_v2_1 | 70 | adaptive_v2_1_25ms | 87.68%/89.19%/90.85% | 98.88%/98.89% | 98.84% | 86.19%/99.24% | 1 | 37.71%/38.82% | 3475→2534 | inconclusive |

## Exact capacity-ratio anchors

An anchor compares the same `SSU×32/NPU` capacity ratio. It is a policy scaling pair only for baseline; V2 has no 32-NPU V2 run yet.

| 128 SSU | Strategy | Equivalent 32 SSU | Lower baseline util/SLO | Upper baseline util/SLO | Status |
|---:|---|---:|---:|---:|---|
| 24 | baseline | 6.0 | 32×6 32.69%/25.00% | 32×6 32.69%/25.00% | exact_capacity_baseline_anchor |
| 24 | current_scheme_b | 6.0 | 32×6 32.69%/25.00% | 32×6 32.69%/25.00% | exact_capacity_baseline_anchor |
| 24 | admission_v2_25ms | 6.0 | 32×6 32.69%/25.00% | 32×6 32.69%/25.00% | exact_capacity_baseline_anchor |
| 24 | adaptive_v2_1_25ms | 6.0 | 32×6 32.69%/25.00% | 32×6 32.69%/25.00% | exact_capacity_baseline_anchor |
| 40 | baseline | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 40 | current_scheme_b | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 40 | admission_25ms | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 40 | admission_50ms | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 40 | admission_v2_25ms | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 40 | adaptive_v2_1_25ms | 10.0 | 32×10 57.24%/75.00% | 32×10 57.24%/75.00% | exact_capacity_baseline_anchor |
| 70 | baseline | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |
| 70 | current_scheme_b | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |
| 70 | admission_25ms | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |
| 70 | admission_50ms | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |
| 70 | admission_v2_25ms | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |
| 70 | adaptive_v2_1_25ms | 17.5 | 32×17 91.61%/80.76% | 32×18 93.29%/79.38% | capacity_bracket_baseline_anchor |

## Policy minus baseline

| Candidate source | Baseline source | NPU | SSU | Strategy | NPU util | Equal-NPU SLO | Equal-category SLO | Joint goal |
|---|---|---:|---:|---|---:|---:|---:|---|
| 128_v1_supplemental | 128_v1_supplemental | 128 | 24 | current_scheme_b | -0.39 pp | -7.99 pp | -7.32 pp | retune_or_reject |
| 128_v2 | 128_v1_supplemental | 128 | 24 | admission_v2_25ms | -0.25 pp | +27.10 pp | +27.18 pp | strong_slo_win_util_preserved |
| 128_adaptive_v2_1 | 128_v1_supplemental | 128 | 24 | adaptive_v2_1_25ms | -0.25 pp | +27.10 pp | +27.18 pp | strong_slo_win_util_preserved |
| 128_v1 | 128_v1 | 128 | 40 | current_scheme_b | -0.03 pp | +0.64 pp | +0.36 pp | not_far_superior |
| 128_v1 | 128_v1 | 128 | 40 | admission_25ms | -0.03 pp | +32.45 pp | +32.62 pp | strong_slo_win_util_preserved |
| 128_v1 | 128_v1 | 128 | 40 | admission_50ms | +0.11 pp | +25.23 pp | +25.03 pp | strong_slo_win_util_preserved |
| 128_v2 | 128_v1 | 128 | 40 | admission_v2_25ms | +0.06 pp | +29.20 pp | +28.94 pp | strong_slo_win_util_preserved |
| 128_adaptive_v2_1 | 128_v1 | 128 | 40 | adaptive_v2_1_25ms | -0.03 pp | +32.45 pp | +32.62 pp | strong_slo_win_util_preserved |
| 128_v1 | 128_v1 | 128 | 70 | current_scheme_b | +0.13 pp | +4.60 pp | +3.16 pp | not_far_superior |
| 128_v1 | 128_v1 | 128 | 70 | admission_25ms | +1.92 pp | +21.54 pp | +20.82 pp | strong_slo_win_util_preserved |
| 128_v1 | 128_v1 | 128 | 70 | admission_50ms | +1.85 pp | +14.93 pp | +14.20 pp | strong_slo_win_util_preserved |
| 128_v2 | 128_v1 | 128 | 70 | admission_v2_25ms | +1.96 pp | +21.22 pp | +20.46 pp | strong_slo_win_util_preserved |
| 128_adaptive_v2_1 | 128_v1 | 128 | 70 | adaptive_v2_1_25ms | +1.92 pp | +21.54 pp | +20.82 pp | strong_slo_win_util_preserved |

## 128/32 trend comparison

| 128 SSU | Strategy | 32 reference | Util Δ (128 / 32) | SLO Δ (128 / 32) | Consistency |
|---:|---|---:|---:|---:|---|
| 24 | adaptive_v2_1_25ms | 32×6 | -0.25 pp / +0.32 pp | +27.10 pp / +50.24 pp | consistent |
| 24 | admission_v2_25ms | — | — | — | no_32_reference |
| 24 | current_scheme_b | 32×6 | -0.39 pp / -0.13 pp | -7.99 pp / -15.85 pp | consistent |
| 40 | adaptive_v2_1_25ms | 32×10 | -0.03 pp / -0.24 pp | +32.45 pp / +9.57 pp | consistent |
| 40 | admission_25ms | 32×10 | -0.03 pp / -0.24 pp | +32.45 pp / +9.57 pp | consistent |
| 40 | admission_50ms | 32×10 | +0.11 pp / -0.11 pp | +25.23 pp / +7.77 pp | consistent |
| 40 | admission_v2_25ms | — | — | — | no_32_reference |
| 40 | current_scheme_b | 32×10 | -0.03 pp / -0.30 pp | +0.64 pp / -24.23 pp | compatible_with_flat_tolerance |
| 70 | adaptive_v2_1_25ms | 32×18 | +1.92 pp / +0.08 pp | +21.54 pp / +19.47 pp | compatible_with_flat_tolerance |
| 70 | admission_25ms | 32×18 | +1.92 pp / +0.08 pp | +21.54 pp / +19.47 pp | compatible_with_flat_tolerance |
| 70 | admission_50ms | 32×18 | +1.85 pp / +0.93 pp | +14.93 pp / +10.22 pp | consistent |
| 70 | admission_v2_25ms | — | — | — | no_32_reference |
| 70 | current_scheme_b | 32×18 | +0.13 pp / -0.51 pp | +4.60 pp / +2.90 pp | compatible_with_flat_tolerance |

## Provenance, invariants, and input pairing

Live validation recomputes each runner's source, config, and case fingerprints and checks every completed summary invariant.

| Source | Available | Completed rows | Valid |
|---|---|---:|---|
| 128_v1 | True | 8 | True |
| 128_v2 | True | 3 | True |
| 128_v1_supplemental | True | 2 | True |
| 128_adaptive_v2_1 | True | 3 | True |
| 32_adaptive_v2_1 | True | 3 | True |

| SSU | Completed sources/strategies | Cross-strategy comparison | All pairing hashes match |
|---:|---|---|---|
| 24 | adaptive_v2_1_25ms, admission_v2_25ms, baseline, current_scheme_b | True | True |
| 40 | adaptive_v2_1_25ms, admission_25ms, admission_50ms, admission_v2_25ms, baseline, current_scheme_b | True | True |
| 70 | adaptive_v2_1_25ms, admission_25ms, admission_50ms, admission_v2_25ms, baseline, current_scheme_b | True | True |

## Absolute category equal-NPU SLO

| Source | SSU | Strategy | SS | SL | LS | LL |
|---|---:|---|---:|---:|---:|---:|
| 128_v1_supplemental | 24 | baseline | 0.00% | 0.00% | 0.00% | 100.00% |
| 128_v1_supplemental | 24 | current_scheme_b | 0.00% | 46.09% | 0.00% | 24.61% |
| 128_v2 | 24 | admission_v2_25ms | 7.94% | 87.80% | 60.08% | 52.91% |
| 128_adaptive_v2_1 | 24 | adaptive_v2_1_25ms | 7.94% | 87.80% | 60.08% | 52.91% |
| 128_v1 | 40 | baseline | 0.00% | 100.00% | 0.00% | 100.00% |
| 128_v1 | 40 | current_scheme_b | 0.00% | 98.83% | 3.39% | 99.22% |
| 128_v1 | 40 | admission_25ms | 35.29% | 100.00% | 95.18% | 100.00% |
| 128_v1 | 40 | admission_50ms | 24.22% | 99.61% | 76.30% | 100.00% |
| 128_v2 | 40 | admission_v2_25ms | 29.43% | 100.00% | 86.33% | 100.00% |
| 128_adaptive_v2_1 | 40 | adaptive_v2_1_25ms | 35.29% | 100.00% | 95.18% | 100.00% |
| 128_v1 | 70 | baseline | 12.11% | 100.00% | 100.00% | 100.00% |
| 128_v1 | 70 | current_scheme_b | 24.74% | 100.00% | 100.00% | 100.00% |
| 128_v1 | 70 | admission_25ms | 95.38% | 100.00% | 100.00% | 100.00% |
| 128_v1 | 70 | admission_50ms | 68.92% | 100.00% | 100.00% | 100.00% |
| 128_v2 | 70 | admission_v2_25ms | 93.95% | 100.00% | 100.00% | 100.00% |
| 128_adaptive_v2_1 | 70 | adaptive_v2_1_25ms | 95.38% | 100.00% | 100.00% | 100.00% |

## Ring ownership association

The tagged-cohort manifest is reconstructed from the verified immutable ring placement. It excludes warmup or untagged I/O that can overlap the SSD measurement window, so correlation is diagnostic rather than causal.

| Source | SSU | Strategy | Placement verified | Manifest↔SSD util r | Manifest↔queue-start r | Ring-hot / util-hot SSU | Util-hot manifest percentile | Mean/max dominant-NPU share |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 128_v1_supplemental | 24 | baseline | True | 0.99841992868816 | 0.3972866832248572 | 20/20 | 100.00% | 0.98%/1.04% |
| 128_v1_supplemental | 24 | current_scheme_b | True | 0.9993500431834815 | 0.6037133805978568 | 20/20 | 100.00% | 1.02%/1.07% |
| 128_v2 | 24 | admission_v2_25ms | True | 0.9993617101673676 | 0.5167900818131316 | 20/20 | 100.00% | 1.30%/1.42% |
| 128_adaptive_v2_1 | 24 | adaptive_v2_1_25ms | True | 0.9993617101673676 | 0.5167900818131316 | 20/20 | 100.00% | 1.30%/1.42% |
| 128_v1 | 40 | baseline | True | 0.9994326763944762 | 0.3783050821347066 | 39/4 | 97.50% | 0.94%/1.00% |
| 128_v1 | 40 | current_scheme_b | True | 0.9996638810274784 | 0.546256247646133 | 4/4 | 100.00% | 1.00%/1.07% |
| 128_v1 | 40 | admission_25ms | True | 0.9997499800745697 | 0.46309903592211193 | 39/39 | 100.00% | 1.05%/1.12% |
| 128_v1 | 40 | admission_50ms | True | 0.9996630644228762 | 0.480900764862621 | 39/39 | 100.00% | 1.05%/1.16% |
| 128_v2 | 40 | admission_v2_25ms | True | 0.9997324916465469 | 0.377443762697799 | 39/39 | 100.00% | 1.04%/1.12% |
| 128_adaptive_v2_1 | 40 | adaptive_v2_1_25ms | True | 0.9997499800745697 | 0.46309903592211193 | 39/39 | 100.00% | 1.05%/1.12% |
| 128_v1 | 70 | baseline | True | 0.9997219642499693 | 0.6981310205704907 | 11/11 | 100.00% | 0.94%/0.99% |
| 128_v1 | 70 | current_scheme_b | True | 0.9998115303534748 | 0.45125195047251354 | 11/11 | 100.00% | 0.96%/1.03% |
| 128_v1 | 70 | admission_25ms | True | 0.9998104761655224 | 0.5889951434042673 | 11/11 | 100.00% | 0.95%/1.04% |
| 128_v1 | 70 | admission_50ms | True | 0.9998000645902728 | 0.7045630936619094 | 11/11 | 100.00% | 0.95%/1.03% |
| 128_v2 | 70 | admission_v2_25ms | True | 0.999739729327752 | 0.4846499062932984 | 11/11 | 100.00% | 0.95%/1.01% |
| 128_adaptive_v2_1 | 70 | adaptive_v2_1_25ms | True | 0.9998104761655224 | 0.5889951434042673 | 11/11 | 100.00% | 0.95%/1.04% |

### Exact-ratio ring geometry

Capacity ratio equality does not preserve the number of SSUs touched by one request. This table uses the strategy-independent full 32-request prefix and exact topology ratios only.

| 128/32 topology | Mean request fanout (128 / 32) | P95 fanout (128 / 32) | Manifest SSD CV (128 / 32) | Max/mean manifest (128 / 32) | Status |
|---|---:|---:|---:|---:|---|
| 128×24 / 32×6 | 24.000 / 6.000 | 24.000 / 6.000 | 0.0554 / 0.0588 | 1.1063 / 1.0975 | exact_capacity_ratio |
| 128×40 / 32×10 | 40.000 / 10.000 | 40.000 / 10.000 | 0.0670 / 0.0319 | 1.1709 / 1.0559 | exact_capacity_ratio |
| 128×70 / 32×— | — | — | — | — | no_exact_32_topology_ring_reference |

## Stationarity diagnostics

| Source | SSU | Strategy | Outstanding start→end (ratio) | Queue Δ blocks/s | 500-ms NPU-util range | Classification |
|---|---:|---|---:|---:|---:|---|
| 128_v1_supplemental | 24 | baseline | 5580→6904 (1.2372334707041748) | 662.0 | +0.56 pp | compatible_not_proven |
| 128_v1_supplemental | 24 | current_scheme_b | 7943→8001 (1.00730110775428) | 29.0 | +2.30 pp | compatible_not_proven |
| 128_v2 | 24 | admission_v2_25ms | 3525→3525 (1.0) | 0.0 | +1.34 pp | compatible_not_proven |
| 128_adaptive_v2_1 | 24 | adaptive_v2_1_25ms | 3525→3525 (1.0) | 0.0 | +1.34 pp | compatible_not_proven |
| 128_v1 | 40 | baseline | 3667→4864 (1.3263358778625953) | 598.5 | +1.25 pp | inconclusive |
| 128_v1 | 40 | current_scheme_b | 4610→1708 (0.3706354369984819) | -1451.0 | +1.09 pp | drift_detected |
| 128_v1 | 40 | admission_25ms | 3459→3506 (1.0135838150289018) | 23.5 | +1.71 pp | compatible_not_proven |
| 128_v1 | 40 | admission_50ms | 4071→5099 (1.2524557956777995) | 514.0 | +1.24 pp | inconclusive |
| 128_v2 | 40 | admission_v2_25ms | 1974→10865 (5.501772151898734) | 4445.5 | +1.85 pp | drift_detected |
| 128_adaptive_v2_1 | 40 | adaptive_v2_1_25ms | 3459→3506 (1.0135838150289018) | 23.5 | +1.71 pp | compatible_not_proven |
| 128_v1 | 70 | baseline | 13330→3022 (0.22676468381966844) | -5154.0 | +1.88 pp | drift_detected |
| 128_v1 | 70 | current_scheme_b | 1956→34 (0.01788451711803781) | -961.0 | +2.32 pp | drift_detected |
| 128_v1 | 70 | admission_25ms | 3475→2534 (0.7292865362485615) | -470.5 | +1.81 pp | inconclusive |
| 128_v1 | 70 | admission_50ms | 5529→5762 (1.0421338155515372) | 116.5 | +1.62 pp | compatible_not_proven |
| 128_v2 | 70 | admission_v2_25ms | 1983→575 (0.2903225806451613) | -704.0 | +1.26 pp | drift_detected |
| 128_adaptive_v2_1 | 70 | adaptive_v2_1_25ms | 3475→2534 (0.7292865362485615) | -470.5 | +1.81 pp | inconclusive |

## Control frequency and Adaptive V2.1 mode audit

Configured admission intervals are event-gated minimum spacing, not periodic polling. Average spacing below is `drain_stop/evaluations` over the complete simulated lifetime.

| Source | SSU | Strategy | Min interval | Effective avg spacing | Evaluations / commits | Path writes | Adaptive V1/V2 mode counts |
|---|---:|---|---:|---:|---:|---:|---|
| 128_v1_supplemental | 24 | current_scheme_b | event-only | 0.1636954043000671 | 43759/854 | 410103 | — |
| 128_v2 | 24 | admission_v2_25ms | 25.0 | 25.77496903426766 | 249/248 | 754209 | — |
| 128_adaptive_v2_1 | 24 | adaptive_v2_1_25ms | 25.0 | 25.77496903426766 | 249/248 | 754209 | 1/248 |
| 128_v1 | 40 | current_scheme_b | event-only | 0.10451441890848549 | 47492/2902 | 4041483 | — |
| 128_v1 | 40 | admission_25ms | 25.0 | 24.99055307312119 | 166/166 | 844631 | — |
| 128_v1 | 40 | admission_50ms | 50.0 | 49.981237745219794 | 85/85 | 430120 | — |
| 128_v2 | 40 | admission_v2_25ms | 25.0 | 24.894664353217443 | 162/162 | 821977 | — |
| 128_adaptive_v2_1 | 40 | adaptive_v2_1_25ms | 25.0 | 24.99055307312119 | 166/166 | 844631 | 166/0 |
| 128_v1 | 70 | current_scheme_b | event-only | 0.06153880436655269 | 53902/3344 | 6019200 | — |
| 128_v1 | 70 | admission_25ms | 25.0 | 25.075474672917103 | 131/131 | 611236 | — |
| 128_v1 | 70 | admission_50ms | 50.0 | 49.95824628658308 | 65/65 | 362115 | — |
| 128_v2 | 70 | admission_v2_25ms | 25.0 | 25.037534612249825 | 131/130 | 585135 | — |
| 128_adaptive_v2_1 | 70 | adaptive_v2_1_25ms | 25.0 | 25.075474672917103 | 131/131 | 611236 | 131/0 |

## Cross-NPU / NPU-id bias audit

The first-vs-last NPU-id quartile and NPU-id correlation are tie/pinning signals, not causal proof. They are compared against the paired baseline.

| SSU | Strategy | Equal−request SLO | SLO min/p10/p90 | SLO first−last id quartile | Util min/p10/p90 | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 24 | admission_v2_25ms | -1.78 pp | 0.00%/25.00%/75.00% | -7.18 pp | 16.66%/21.18%/43.69% | cross_npu_inequality_regression_without_clear_id_order |
| 24 | adaptive_v2_1_25ms | -1.78 pp | 0.00%/25.00%/75.00% | -7.18 pp | 16.66%/21.18%/43.69% | cross_npu_inequality_regression_without_clear_id_order |
| 40 | admission_25ms | -0.42 pp | 62.50%/71.00%/93.64% | +0.07 pp | 40.01%/44.97%/56.30% | cross_npu_inequality_regression_without_clear_id_order |
| 40 | admission_50ms | -0.34 pp | 44.44%/62.50%/90.00% | -5.92 pp | 35.18%/44.29%/57.85% | cross_npu_inequality_regression_without_clear_id_order |
| 40 | admission_v2_25ms | -0.61 pp | 50.00%/62.50%/93.64% | +5.16 pp | 43.95%/47.75%/55.65% | cross_npu_inequality_regression_without_clear_id_order |
| 40 | adaptive_v2_1_25ms | -0.42 pp | 62.50%/71.00%/93.64% | +0.07 pp | 40.01%/44.97%/56.30% | cross_npu_inequality_regression_without_clear_id_order |
| 70 | admission_25ms | -0.01 pp | 86.67%/93.62%/100.00% | -1.63 pp | 87.68%/89.19%/92.42% | no_new_large_npu_id_bias_signal |
| 70 | admission_50ms | -0.04 pp | 80.00%/85.71%/100.00% | +0.27 pp | 86.76%/88.91%/92.55% | no_new_large_npu_id_bias_signal |
| 70 | admission_v2_25ms | -0.02 pp | 86.67%/93.33%/100.00% | +1.48 pp | 88.16%/89.23%/92.65% | no_new_large_npu_id_bias_signal |
| 70 | adaptive_v2_1_25ms | -0.01 pp | 86.67%/93.62%/100.00% | -1.63 pp | 87.68%/89.19%/92.42% | no_new_large_npu_id_bias_signal |

## Category equal-NPU SLO deltas vs baseline

| NPU | SSU | Strategy | SS | SL | LS | LL |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 24 | current_scheme_b | +0.00 pp | +46.09 pp | +0.00 pp | -75.39 pp |
| 128 | 24 | admission_v2_25ms | +7.94 pp | +87.80 pp | +60.08 pp | -47.09 pp |
| 128 | 24 | adaptive_v2_1_25ms | +7.94 pp | +87.80 pp | +60.08 pp | -47.09 pp |
| 128 | 40 | current_scheme_b | +0.00 pp | -1.17 pp | +3.39 pp | -0.78 pp |
| 128 | 40 | admission_25ms | +35.29 pp | +0.00 pp | +95.18 pp | +0.00 pp |
| 128 | 40 | admission_50ms | +24.22 pp | -0.39 pp | +76.30 pp | +0.00 pp |
| 128 | 40 | admission_v2_25ms | +29.43 pp | +0.00 pp | +86.33 pp | +0.00 pp |
| 128 | 40 | adaptive_v2_1_25ms | +35.29 pp | +0.00 pp | +95.18 pp | +0.00 pp |
| 128 | 70 | current_scheme_b | +12.63 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 128 | 70 | admission_25ms | +83.27 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 128 | 70 | admission_50ms | +56.81 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 128 | 70 | admission_v2_25ms | +81.84 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| 128 | 70 | adaptive_v2_1_25ms | +83.27 pp | +0.00 pp | +0.00 pp | +0.00 pp |

## Interpretation guardrails

- `SSD mean/max` and the count at >=99% expose fixed-placement hotspots; aggregate SSU capacity alone is not sufficient.
- NPU-link mean/max tests whether the 50-GB/s receive link, rather than SSD allocation, is the visible bottleneck.
- Queue `drift_detected` rejects a steady-state interpretation. `compatible_not_proven` does not prove stationarity because only the measurement-window endpoints are available.
- A strong joint win means SLO is at least +10 pp while mean NPU utilization is no worse than -1 pp relative to the paired baseline.
