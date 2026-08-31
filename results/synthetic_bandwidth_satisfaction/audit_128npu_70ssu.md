# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B planned satisfaction | Planned 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---|
| uniform_dense_ssu_5 | 5.00-5.00 | 2.73 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_20 | 20.00-20.00 | 10.94 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_39 | 39.00-39.00 | 21.33 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_40 | 40.00-40.00 | 21.88 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_41 | 41.00-41.00 | 22.42 | no | yes | 0.976 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_60 | 60.00-60.00 | 32.81 | no | yes | 0.667 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_80 | 80.00-80.00 | 43.75 | no | yes | 0.500 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_84 | 84.00-84.00 | 45.94 | no | no | 0.476 | 0.000 | input_capacity_prevents_100pct_warm_2x |
| uniform_dense_ssu_90 | 90.00-90.00 | 49.22 | no | no | 0.444 | 0.000 | input_capacity_prevents_100pct_warm_2x |

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link delivered/target demand. Both use exact overlap at the common measurement boundaries; completion-only byte counting is not used.
