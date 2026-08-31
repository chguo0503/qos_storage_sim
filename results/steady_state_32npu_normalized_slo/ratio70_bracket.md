# SSU70 steady-state bracket (32 NPU)

Status: `complete`. 

Under the same per-NPU demand and SSD40/NPU50 data plane, scaling both resource counts by four preserves the NPU:SSU capacity ratio:

| Role | 32-NPU point | Equivalent 128-NPU point |
|---|---:|---:|
| Lower bracket (this run) | 32 NPU / 17 SSU | 128 NPU / 68 SSU |
| Requested target | — | 128 NPU / 70 SSU |
| Upper bracket (existing point) | 32 NPU / 18 SSU | 128 NPU / 72 SSU |

Thus `68 < 70 < 72`: 32x17 and the existing 32x18 point bracket the 128-NPU SSU70 operating point. This is a capacity-ratio bracket, not a claim that finite-fleet scheduling variance is identical.

The admission controller uses the full remaining manifest and a batch-boundary event-gated controller with a 10-ms minimum interval (`interval_ms=None`); it is not a fixed 10-ms periodic controller.

## Exact-window results

| Strategy | Status | NPU util | Equal-NPU TTFT SLO | Request-weighted SLO | Mean TTFT (ms) | P99 TTFT (ms) | SSD util | NPU-link util | Requests | Control evals |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | ok | 91.61% | 80.76% | 80.90% | 126.528 | 209.792 | 89.69% | 38.12% | 513 | 0 |
| scheme_b_slo_admission | ok | 90.57% | 98.12% | 98.15% | 128.673 | 232.362 | 87.98% | 37.39% | 486 | 265 |

## Admission minus baseline

| Metric | Delta |
|---|---:|
| NPU utilization | -1.05 pp |
| Equal-NPU TTFT SLO | +17.37 pp |
| Request-weighted TTFT SLO | +17.25 pp |
| Mean TTFT | +2.144 ms |
| P99 TTFT | +22.570 ms |

## Validity and provenance

- Source stable during run: `True`
- Source fingerprint: `3492b7a86e1cb50a0686913dc8d5395514613f1566961f54a01197ac478dbc95`
- Ending source fingerprint: `3492b7a86e1cb50a0686913dc8d5395514613f1566961f54a01197ac478dbc95`
- Config fingerprint: `1a82e5f59af3bb18bea2ecf4ae2cee4cd9c2e6b78ee08f4af4ee494acb41b815`
- All simulator/runner invariants: `True`
- Paired input fields all match: `True`
- assignment_hash: `8fd46b083f896992d736e44019b22136750ea4b5d59f4f79e8faa67097ea8939`
- workload_hash: `f7156362d26c4e29e3468fb47a9aff7af77e6435c7d2e9389ad15a1cd3c3f56a`
- placement_hash: `172c280db45fa75013b38e8888aaf232991cfd78d7eec1c065a0b1767d9603d3`
- trace_hash: `0c4ffa40f9456a0418a5aacda42597b8ff1dd92e35a20c3ccf649b29ef1460d4`
- simulator_input_fingerprint: `6c6a70256d55cdb5cef139223188502217c8b636e8d8699ae37111a62389a41f`

## Measurement-block stability

| Strategy | Block NPU-util range | SSD outstanding blocks start -> end |
|---|---:|---:|
| baseline | 0.67 pp | 6492 -> 74 |
| scheme_b_slo_admission | 0.46 pp | 6815 -> 5709 |
