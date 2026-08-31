# 128-NPU warm/full-load admission experiment

All rows use seed 42, 16 layers, batch 1, four warmup completions per NPU, 500-ms settle, and the same 2,000-ms measurement window. Admission 25/50 ms denotes event-gated minimum spacing, not a periodic timer.

Complete: `false`; selected complete: `true`; source stable: `true`; config stable: `true`.

| SSU | Strategy | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Requests | Evals | Commits | Path writes | Wall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | baseline | 30.21% | 30.48% | 32.76% | 25.00% | 25.00% | 0.00% | 0.00% | 0.00% | 100.00% | 768 | 0 | 0 | 0 | 604.4s |
| 24 | current_scheme_b | 27.83% | 29.94% | 32.38% | 17.01% | 17.40% | 0.00% | 45.74% | 0.00% | 23.98% | 730 | 43759 | 854 | 410103 | 1154.5s |
