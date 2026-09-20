# Discovered OD/Once screening jobs

Updated UTC: 2026-09-18T17:09:54.118203+00:00

Warm window: [2,4) seconds. Bandwidth: GiB/s. Stall: summed NPU milliseconds.

| Case | Status | U | SLO×1.5 | Strict underload | A/B cards | Internal stall | Cross-request L0 stall | Internal share |
|---|---|---:|---:|---|---:|---:|---:|---:|
| boundary_od_seed7_local | complete | 97.9693% | 100.00% | True | 32/32 | 0.000 | 1299.631 | 0.00% |
| fixedssu1_safe38_od_local | complete | 97.2811% | 96.37% | True | 32/32 | 1105.949 | 634.148 | 63.56% |
| fixedssu1_safe38_once_remote | complete | 99.5461% | 100.00% | True | 32/32 | 93.584 | 196.899 | 32.22% |
| fixedssu1_safe38_ordinary_addresses_od_remote | complete | 97.5349% | 96.07% | False | 32/32 | 1125.468 | 452.220 | 71.34% |
| fixedssu1_safe38_random_od_remote | complete | 99.4861% | 100.00% | False | 32/32 | 0.000 | 328.879 | 0.00% |
| fixedssu1_safe38_spread_od_remote | complete | 97.5070% | 100.00% | True | 32/32 | 238.916 | 1356.578 | 14.97% |
| fixedssu1_safe39_od_local | complete | 97.0112% | 98.81% | True | 32/32 | 1190.204 | 722.610 | 62.22% |
| fixedssu1_tight40_od_local | complete | 96.9371% | 98.95% | True | 32/32 | 1215.025 | 745.231 | 61.98% |
| native_phase_lock_a101_b0995_od_local | complete | 94.0580% | 73.75% | True | 32/32 | 3114.788 | 688.100 | 81.91% |
| native_phase_lock_a101_b0995_once_remote | complete | 99.4978% | 100.00% | True | 32/32 | 107.254 | 214.144 | 33.37% |
| native_phase_lock_a102_od_local | complete | 95.9035% | 86.88% | True | 32/32 | 2063.002 | 558.788 | 78.69% |
| native_phase_lock_a102_once_remote | complete | 99.0648% | 100.00% | False | 32/32 | 165.474 | 433.077 | 27.65% |
| raw32_128_a3b1_p3_od_local | complete | 98.8225% | 100.00% | True | 32/32 | 73.785 | 679.830 | 9.79% |
| raw32_128_a4b1_p3_od_remote | complete | 99.1395% | 100.00% | True | 32/32 | 102.132 | 448.609 | 18.54% |
| raw32_128_a5b1_p3_od_remote | complete | 99.0421% | 100.00% | True | 32/32 | 64.567 | 548.496 | 10.53% |
| raw32_200_a5b1_p3_od_remote | complete | 98.5624% | 100.00% | True | 32/32 | 121.786 | 798.282 | 13.24% |
| raw32_200_a7b1_p3_od_remote | complete | 98.0448% | 100.00% | True | 32/32 | 303.602 | 947.703 | 24.26% |
| raw32_200_proxy_best_od_local | complete | 99.0575% | 100.00% | True | 32/32 | 71.840 | 531.390 | 11.91% |
| raw32_32_a3b1_p4_od_local | complete | 99.6835% | 100.00% | False | 32/32 | 186.743 | 15.815 | 92.19% |
| raw_boundary_32m2048_200m4096_sync_once_remote | complete | 98.4911% | 100.00% | True | 32/32 | 104.514 | 861.190 | 10.82% |
| synth_phase180_1039_c103_od_remote | complete | 98.7913% | 100.00% | False | 32/32 | 370.247 | 403.341 | 47.86% |
| synth_phase180_1039_od_local | complete | 99.0127% | 100.00% | False | 32/32 | 299.278 | 332.563 | 47.37% |
| synth_phase80_1210_c103_od_remote | complete | 98.0730% | 97.86% | False | 32/32 | 570.329 | 662.979 | 46.24% |
| synth_phase80_1210_od_local | complete | 98.1512% | 98.75% | False | 32/32 | 612.690 | 570.517 | 51.78% |
| synth_phase80_1210_once_remote | complete | 98.7862% | 100.00% | False | 32/32 | 286.746 | 490.115 | 36.91% |

A/B coverage counts actual overlapping computation, not only admission. Strict underload refers to current admitted-request V/C on every disk at every event interval. Completed files are verified against recorded input/result hashes and post-run source checks.

Completed: 25/25. Errors: 0.
