# 128-NPU real-data paired experiment: SSU16

This independent run compares baseline with admission_25ms using the same seed-42 real-data trace, 16-layer workload, placement, warmup, settle period, and 2,000-ms measurement window. The 25-ms setting is event-gated minimum spacing, not a periodic timer.

Complete: `false`; selected complete: `false`; source stable: `true`; config stable: `true`.

Source fingerprint: `eabd7e4c427a5a216004060e1d8da00f13a99912d4401c9574ce13eccfea2f45`

Config fingerprint: `5bdcbfd47422a718310c064e3fa14e8c7a614538c15a5a8f3470a3e1550813d3`

| Strategy | NPU util min | p10 | mean | Equal-NPU SLO | Request SLO | SS | SL | LS | LL | Requests | Evals | Commits | Path writes | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 22.17% | 22.34% | 22.64% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 416 | 0 | 0 | 0 | 481.0s |

## Paired input fingerprints

- assignment: `f426e5cf68b8ad7209cc475e3d7a59e04b0ecc81bc86fb69db2049d305aa5289`
- workload: `3562f9d737034f97efcb3141f36b012ad39cd847dc0e82b6c6c88a8c20982f2b`
- placement: `b27df01b3bfa3ff4b963a56f03d56ad9cab2f7218aed72313832e75267f9b920`
- trace: `feec5925423477f19f4cb678dc171b9881a3f8f1ea695d00bb0b84e894d8fe55`
- simulator: `5691497d7e1e0491e868e749871ac05db6e590041e3d3e040144af516aeef9fb`
