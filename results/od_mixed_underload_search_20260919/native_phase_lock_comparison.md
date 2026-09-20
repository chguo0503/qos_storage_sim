# Fully drained phase-lock comparisons

Completed policies: 4/4.

| Candidate | Policy | Window | U | SLO×1.5 | Strict underload | A/B cards |
|---|---|---|---:|---:|---|---:|
| native_phase_lock_a102 | od_baseline | 2-4s | 95.90345% | 86.87943% | True | 32/32 |
| native_phase_lock_a102 | od_baseline | 2-6s | 96.91459% | 92.96188% | True | 32/32 |
| native_phase_lock_a102 | od_baseline | 4-8s | 98.45776% | 99.47705% | True | 32/32 |
| native_phase_lock_a102 | od_baseline | 8-12s | 99.43984% | 100.00000% | True | 32/32 |
| native_phase_lock_a102 | od_baseline | 12-16s | 99.34161% | 100.00000% | True | 32/32 |
| native_phase_lock_a102 | od_baseline | full | 93.07917% | 96.09544% | True | 32/32 |
| native_phase_lock_a102 | once | 2-4s | 99.06476% | 100.00000% | False | 32/32 |
| native_phase_lock_a102 | once | 2-6s | 99.23451% | 100.00000% | False | 32/32 |
| native_phase_lock_a102 | once | 4-8s | 99.08633% | 99.93990% | False | 32/32 |
| native_phase_lock_a102 | once | 8-12s | 99.29579% | 99.94176% | False | 32/32 |
| native_phase_lock_a102 | once | 12-16s | 99.24458% | 100.00000% | False | 32/32 |
| native_phase_lock_a102 | once | full | 92.75482% | 99.97289% | False | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | 2-4s | 94.05799% | 73.75297% | True | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | 2-6s | 96.05276% | 86.70277% | True | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | 4-8s | 98.28915% | 99.76608% | True | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | 8-12s | 99.37819% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | 12-16s | 99.48214% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | od_baseline | full | 92.62503% | 94.78037% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | 2-4s | 99.49781% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | 2-6s | 99.45523% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | 4-8s | 99.28727% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | 8-12s | 99.42791% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | 12-16s | 99.48279% | 100.00000% | True | 32/32 |
| native_phase_lock_a101_b0995 | once | full | 93.38726% | 100.00000% | True | 32/32 |

Window admission cohorts differ between policies. Full SLO uses the identical fully drained request population; full U includes inactive drain-tail time.
