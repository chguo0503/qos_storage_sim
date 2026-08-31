# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B planned satisfaction | Planned 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---|
| hotspot_raw_infeasible | 10.00-60.00 | 20.00 | no | yes | 0.714 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |

## Full data-plane steady window

| Case | Strategy | SSD satisfaction | Link satisfaction | SSD util | Link util | NPU compute util | TTFT 2x SLO | Mean TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hotspot_raw_infeasible | layer_once | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID |

## Invalid data-plane runs

- `hotspot_raw_infeasible / layer_once`: `AssertionError`; see the JSON `failure` field for invariant and per-NPU drain diagnostics.

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link delivered/target demand. Both use exact overlap at the common measurement boundaries; completion-only byte counting is not used.
