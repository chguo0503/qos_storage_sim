# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B satisfaction | Scheme-B 2x SLO | SLO-aware satisfaction | SLO-aware 2x SLO | Admission satisfaction | Admission 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---|
| uniform_striped_ssu_32 | 32.00-32.00 | 8.00 | yes | yes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_striped_ssu_36 | 36.00-36.00 | 9.00 | yes | yes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_striped_ssu_39 | 39.00-39.00 | 9.75 | yes | yes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_striped_ssu_40 | 40.00-40.00 | 10.00 | yes | yes | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_striped_ssu_41 | 41.00-41.00 | 10.25 | no | yes | 0.976 | 1.000 | 0.976 | 1.000 | 0.976 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_striped_ssu_42 | 42.00-42.00 | 10.50 | no | yes | 0.952 | 1.000 | 0.952 | 1.000 | 0.952 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_striped_ssu_44 | 44.00-44.00 | 11.00 | no | yes | 0.909 | 1.000 | 0.909 | 1.000 | 0.909 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| skewed_striped_ssu_41 | 41.00-41.00 | 16.40 | no | yes | 0.976 | 1.000 | 0.976 | 1.000 | 0.976 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |

## Full data-plane steady window

| Case | Strategy | SSD satisfaction | Link satisfaction | SSD util | Link util | NPU compute util | TTFT 2x SLO | Mean TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_striped_ssu_32 | baseline | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_32 | layer_once | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_32 | refresh8 | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_32 | scheme_b | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_32 | scheme_b_slo | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_32 | scheme_b_admission | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | baseline | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | layer_once | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | refresh8 | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | scheme_b | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | scheme_b_slo | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_36 | scheme_b_admission | 1.000 | 1.000 | 0.900 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | baseline | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | layer_once | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | refresh8 | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | scheme_b | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | scheme_b_slo | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_39 | scheme_b_admission | 1.000 | 1.000 | 0.975 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | baseline | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | layer_once | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | refresh8 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | scheme_b | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | scheme_b_slo | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.00 |
| uniform_striped_ssu_40 | scheme_b_admission | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 160.02 |
| uniform_striped_ssu_41 | baseline | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.00 |
| uniform_striped_ssu_41 | layer_once | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.03 |
| uniform_striped_ssu_41 | refresh8 | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 163.95 |
| uniform_striped_ssu_41 | scheme_b | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.00 |
| uniform_striped_ssu_41 | scheme_b_slo | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.00 |
| uniform_striped_ssu_41 | scheme_b_admission | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.02 |
| uniform_striped_ssu_42 | baseline | 0.952 | 0.953 | 1.000 | 0.000 | 0.952 | 1.000 | 168.00 |
| uniform_striped_ssu_42 | layer_once | 0.952 | 0.953 | 1.000 | 0.000 | 0.953 | 1.000 | 168.04 |
| uniform_striped_ssu_42 | refresh8 | 0.952 | 0.953 | 1.000 | 0.000 | 0.952 | 1.000 | 168.06 |
| uniform_striped_ssu_42 | scheme_b | 0.952 | 0.953 | 1.000 | 0.000 | 0.952 | 1.000 | 168.00 |
| uniform_striped_ssu_42 | scheme_b_slo | 0.952 | 0.953 | 1.000 | 0.000 | 0.952 | 1.000 | 168.00 |
| uniform_striped_ssu_42 | scheme_b_admission | 0.952 | 0.953 | 1.000 | 0.000 | 0.952 | 1.000 | 168.06 |
| uniform_striped_ssu_44 | baseline | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.00 |
| uniform_striped_ssu_44 | layer_once | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.00 |
| uniform_striped_ssu_44 | refresh8 | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.11 |
| uniform_striped_ssu_44 | scheme_b | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.00 |
| uniform_striped_ssu_44 | scheme_b_slo | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.00 |
| uniform_striped_ssu_44 | scheme_b_admission | 0.909 | 0.909 | 1.000 | 0.000 | 0.909 | 1.000 | 176.09 |
| skewed_striped_ssu_41 | baseline | 0.976 | 0.976 | 1.000 | 0.000 | 0.976 | 1.000 | 164.00 |
| skewed_striped_ssu_41 | layer_once | 0.976 | 0.976 | 1.000 | 0.000 | 0.986 | 1.000 | 162.53 |
| skewed_striped_ssu_41 | refresh8 | 0.976 | 0.976 | 1.000 | 0.000 | 0.985 | 1.000 | 162.65 |
| skewed_striped_ssu_41 | scheme_b | 0.976 | 0.975 | 1.000 | 0.000 | 0.983 | 1.000 | 162.85 |
| skewed_striped_ssu_41 | scheme_b_slo | 0.976 | 0.976 | 1.000 | 0.000 | 0.981 | 1.000 | 163.15 |
| skewed_striped_ssu_41 | scheme_b_admission | 0.976 | 0.976 | 1.000 | 0.000 | 0.983 | 1.000 | 162.95 |

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link service/target demand. Both use exact overlap at the common measurement boundaries; partial service from an in-flight link command is not compute-visible until that command completes.
