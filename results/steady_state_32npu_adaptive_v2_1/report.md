# 32-NPU Adaptive Admission V2.1 validation

Adaptive V2.1 uses the current active-manifest selected fraction. Fractions strictly below 0.75 select explicit V2 spill; all others retain V1 coflow residual. The signal is causal, while 0.75 is a calibrated operating-point heuristic rather than a universal optimum.

Complete: `true`; selected complete: `true`; source stable: `true`; config stable: `true`.

Source fingerprint: `78c5eb9435b67b41495b88888676d2869ab2830a61dcc524cc090a54682c5c41`

Config fingerprint: `f973e11ace94dcca61ac5b33a774078993649cd590b3c09701e01732339a6711`

| SSU | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Explicit/coflow evals | Last selected | Requests | Path writes | Wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 13.70% | 16.07% | 33.01% | 75.24% | 77.05% | 12.77% | 100.00% | 97.87% | 100.00% | 243/2 | 0.5938 | 183 | 39378 | 188.0s |
| 10 | 50.38% | 53.25% | 57.00% | 84.57% | 84.59% | 43.75% | 100.00% | 95.06% | 100.00% | 0/154 | 0.9375 | 318 | 48010 | 206.3s |
| 18 | 92.40% | 92.66% | 93.38% | 98.85% | 98.85% | 95.42% | 100.00% | 100.00% | 100.00% | 0/126 | 1.0000 | 523 | 37158 | 309.8s |
