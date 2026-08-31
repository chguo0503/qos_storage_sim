# 128-NPU Adaptive Admission V2.1 real-data matrix

The current active-manifest selected fraction is causal. Fractions strictly below 0.75 use explicit V2 spill; all others retain V1 coflow residual. The cutoff is a calibrated heuristic, not a universal optimum.

Complete: `true`; selected complete: `true`; source stable: `true`; config stable: `true`.

Source fingerprint: `5a2894a23f050a383c9d7a16c12f1097c78e12b9e8f7ffcbdc25e2abdd933072`

Config fingerprint: `f09150497f9555662ce9ace4a6d673e7fe3c1bbb39df37b59584494ae8cd0842`

| SSU | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Explicit/coflow evals | Last selected | Requests | Path writes | Wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 16.66% | 21.18% | 32.51% | 52.10% | 53.88% | 8.33% | 90.91% | 59.09% | 55.87% | 248/1 | 0.5859 | 722 | 754209 | 1040.5s |
| 40 | 40.01% | 44.97% | 51.57% | 82.45% | 82.87% | 36.59% | 100.00% | 95.22% | 100.00% | 0/166 | 0.9219 | 1144 | 844631 | 1202.2s |
| 70 | 87.68% | 89.19% | 90.85% | 98.88% | 98.89% | 95.51% | 100.00% | 100.00% | 100.00% | 0/131 | 1.0000 | 1977 | 611236 | 2142.4s |
