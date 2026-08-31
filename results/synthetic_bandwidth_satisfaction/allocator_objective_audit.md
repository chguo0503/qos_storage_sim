# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B satisfaction | Scheme-B 2x SLO | SLO-aware satisfaction | SLO-aware 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---|
| deadline_feasible_scheme_objective | 80.00-80.00 | 60.00 | no | yes | 0.500 | 0.667 | 0.500 | 1.000 | capacity_allows_warm_2x_but_scheme_b_objective_can_miss_it |
| npu50_control | 30.00-30.00 | 60.00 | no | yes | 0.833 | 1.000 | 0.833 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| npu50_link_stress | 10.00-10.00 | 80.00 | no | yes | 0.625 | 1.000 | 0.625 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link service/target demand. Both use exact overlap at the common measurement boundaries; partial service from an in-flight link command is not compute-visible until that command completes.
