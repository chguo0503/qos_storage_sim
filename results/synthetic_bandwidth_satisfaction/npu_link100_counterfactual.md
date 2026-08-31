# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B planned satisfaction | Planned 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---|
| npu50_link_stress | 10.00-10.00 | 80.00 | no | yes | 0.625 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |

## Full data-plane steady window

| Case | Strategy | SSD satisfaction | Link satisfaction | SSD util | Link util | NPU compute util | TTFT 2x SLO | Mean TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| npu50_link_stress | baseline | 1.000 | 1.000 | 0.250 | 0.800 | 1.000 | 1.000 | 160.00 |

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link delivered/target demand. Both use exact overlap at the common measurement boundaries; completion-only byte counting is not used.
