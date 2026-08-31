# Consistent-hash ring ownership scale audit

Method: exact integration of all adjacent SHA-256 ring intervals; no block-key sampling. Ownership is normalized so an ideally balanced SSU has `1.0x`.

Virtual nodes per SSU: `256`

| SSU count | Min ownership | Min SSU | Max ownership | Max SSU | Population stddev | Conservation error |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.943378x (-5.66%) | 3 | 1.055528x (+5.55%) | 4 | 0.031237 | 0.000e+00 |
| 18 | 0.895580x (-10.44%) | 7 | 1.059218x (+5.92%) | 5 | 0.041541 | 1.110e-16 |
| 24 | 0.903955x (-9.60%) | 7 | 1.102909x (+10.29%) | 20 | 0.055313 | 1.110e-16 |
| 40 | 0.836575x (-16.34%) | 37 | 1.169584x (+16.96%) | 4 | 0.066797 | 0.000e+00 |
| 70 | 0.872811x (-12.72%) | 65 | 1.150396x (+15.04%) | 11 | 0.058924 | 0.000e+00 |

## Virtual-node sensitivity (placement-only counterfactual)

placement-only counterfactual; the completed simulations and their fingerprints remain fixed at 256 virtual nodes per SSU

| SSU count | Virtual nodes / SSU | Min ownership | Max ownership | Population stddev |
|---:|---:|---:|---:|---:|
| 40 | 256 | 0.836575x | 1.169584x | 0.066797 |
| 40 | 1024 | 0.949598x | 1.058502x | 0.026771 |
| 40 | 4096 | 0.970775x | 1.034628x | 0.015789 |
| 70 | 256 | 0.872811x | 1.150396x | 0.058924 |
| 70 | 1024 | 0.888092x | 1.077290x | 0.030796 |
| 70 | 4096 | 0.971487x | 1.033362x | 0.012687 |

Increasing virtual nodes is a placement-balance candidate for a new experiment, not a reinterpretation of current results

## Scale conclusion

Keeping NPU:SSU capacity ratio fixed does not keep fixed-placement hotspot severity fixed: the maximum ownership rises from 1.0555x at SSU10 to 1.1696x at SSU40, and from 1.0592x at SSU18 to 1.1504x at SSU70. Thus 32-NPU ratio points are capacity brackets, not placement-equivalent predictions for 128 NPU.

In the completed 128-NPU SSU40 row, the persistent queue hotspot is SSU4, which is also the exact ring-ownership maximum (1.169584x). This alignment is evidence that the observed scale loss is driven by fixed placement, although it does not by itself prove every end-to-end latency effect.

Source fingerprint: `5e5ec64d754c3862b3539748d722a21547ab5baef5f0226183233873e490faf5`
