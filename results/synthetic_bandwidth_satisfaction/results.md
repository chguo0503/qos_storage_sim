# Synthetic bandwidth-satisfaction results

## Allocator/input audit

| Case | SSU offered GB/s (min-max) | Max NPU GB/s | Raw feasible | Warm 2x feasible | Scheme-B planned satisfaction | Planned 2x SLO | Diagnosis |
|---|---:|---:|:---:|:---:|---:|---:|---|
| uniform_dense_ssu_5 | 5.00-5.00 | 2.50 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_20 | 20.00-20.00 | 10.00 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_39 | 39.00-39.00 | 19.50 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_40 | 40.00-40.00 | 20.00 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| uniform_dense_ssu_41 | 41.00-41.00 | 20.50 | no | yes | 0.976 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_60 | 60.00-60.00 | 30.00 | no | yes | 0.667 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_80 | 80.00-80.00 | 40.00 | no | yes | 0.500 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| uniform_dense_ssu_84 | 84.00-84.00 | 42.00 | no | no | 0.476 | 0.000 | input_capacity_prevents_100pct_warm_2x |
| uniform_dense_ssu_90 | 90.00-90.00 | 45.00 | no | no | 0.444 | 0.000 | input_capacity_prevents_100pct_warm_2x |
| heterogeneous_raw_feasible | 40.00-40.00 | 30.00 | yes | yes | 1.000 | 1.000 | raw_full_hide_input_feasible |
| deadline_feasible_scheme_objective | 80.00-80.00 | 60.00 | no | yes | 0.500 | 0.667 | capacity_allows_warm_2x_but_scheme_b_objective_can_miss_it |
| hotspot_raw_infeasible | 10.00-60.00 | 20.00 | no | yes | 0.714 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| hotspot_deadline_infeasible | 10.00-90.00 | 30.00 | no | no | 0.500 | 0.250 | input_capacity_prevents_100pct_warm_2x |
| single_flow_60 | 60.00-60.00 | 60.00 | no | yes | 0.667 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| single_flow_90 | 90.00-90.00 | 90.00 | no | no | 0.444 | 0.000 | input_capacity_prevents_100pct_warm_2x |
| npu50_control | 30.00-30.00 | 60.00 | no | yes | 0.833 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |
| npu50_link_stress | 10.00-10.00 | 80.00 | no | yes | 0.625 | 1.000 | raw_infeasible_but_warm_2x_capacity_feasible |

## Full data-plane steady window

| Case | Strategy | SSD satisfaction | Link satisfaction | SSD util | Link util | NPU compute util | TTFT 2x SLO | Mean TTFT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform_dense_ssu_5 | baseline | 1.000 | 1.000 | 0.125 | 0.050 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_5 | layer_once | 1.000 | 1.000 | 0.125 | 0.050 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_5 | refresh8 | 1.000 | 1.000 | 0.125 | 0.050 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_5 | scheme_b | 1.000 | 1.000 | 0.125 | 0.050 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_20 | baseline | 1.000 | 1.000 | 0.500 | 0.200 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_20 | layer_once | 1.000 | 1.000 | 0.500 | 0.200 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_20 | refresh8 | 1.000 | 1.000 | 0.500 | 0.200 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_20 | scheme_b | 1.000 | 1.000 | 0.500 | 0.200 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_39 | baseline | 1.000 | 1.000 | 0.975 | 0.390 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_39 | layer_once | 1.000 | 1.000 | 0.975 | 0.390 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_39 | refresh8 | 1.000 | 1.000 | 0.975 | 0.390 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_39 | scheme_b | 1.000 | 1.000 | 0.975 | 0.390 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_40 | baseline | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_40 | layer_once | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_40 | refresh8 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_40 | scheme_b | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| uniform_dense_ssu_41 | baseline | 0.976 | 0.976 | 1.000 | 0.400 | 0.976 | 1.000 | 164.00 |
| uniform_dense_ssu_41 | layer_once | 0.971 | 0.971 | 0.995 | 0.398 | 0.970 | 1.000 | 164.59 |
| uniform_dense_ssu_41 | refresh8 | 0.973 | 0.973 | 0.998 | 0.399 | 0.974 | 1.000 | 164.42 |
| uniform_dense_ssu_41 | scheme_b | 0.976 | 0.976 | 1.000 | 0.400 | 0.976 | 1.000 | 164.00 |
| uniform_dense_ssu_60 | baseline | 0.667 | 0.666 | 1.000 | 0.400 | 0.667 | 1.000 | 240.00 |
| uniform_dense_ssu_60 | layer_once | 0.666 | 0.666 | 0.999 | 0.399 | 0.667 | 1.000 | 240.69 |
| uniform_dense_ssu_60 | refresh8 | 0.666 | 0.666 | 0.998 | 0.399 | 0.666 | 1.000 | 241.96 |
| uniform_dense_ssu_60 | scheme_b | 0.667 | 0.667 | 1.000 | 0.400 | 0.669 | 1.000 | 240.00 |
| uniform_dense_ssu_80 | baseline | 0.500 | 0.500 | 1.000 | 0.400 | 0.500 | 0.333 | 320.00 |
| uniform_dense_ssu_80 | layer_once | 0.500 | 0.500 | 1.000 | 0.400 | 0.499 | 0.417 | 317.23 |
| uniform_dense_ssu_80 | refresh8 | 0.500 | 0.500 | 1.000 | 0.400 | 0.502 | 0.500 | 319.88 |
| uniform_dense_ssu_80 | scheme_b | 0.500 | 0.500 | 1.000 | 0.400 | 0.500 | 0.333 | 320.00 |
| uniform_dense_ssu_84 | baseline | 0.476 | 0.476 | 1.000 | 0.400 | 0.476 | 0.000 | 336.00 |
| uniform_dense_ssu_84 | layer_once | 0.476 | 0.477 | 1.000 | 0.400 | 0.477 | 0.417 | 335.18 |
| uniform_dense_ssu_84 | refresh8 | 0.476 | 0.476 | 0.999 | 0.400 | 0.478 | 0.083 | 339.13 |
| uniform_dense_ssu_84 | scheme_b | 0.476 | 0.476 | 1.000 | 0.400 | 0.479 | 0.000 | 336.00 |
| uniform_dense_ssu_90 | baseline | 0.444 | 0.444 | 1.000 | 0.400 | 0.444 | 0.000 | 360.00 |
| uniform_dense_ssu_90 | layer_once | 0.444 | 0.445 | 1.000 | 0.400 | 0.447 | 0.167 | 358.34 |
| uniform_dense_ssu_90 | refresh8 | 0.444 | 0.444 | 1.000 | 0.400 | 0.445 | 0.083 | 361.03 |
| uniform_dense_ssu_90 | scheme_b | 0.444 | 0.444 | 1.000 | 0.400 | 0.442 | 0.000 | 360.00 |
| heterogeneous_raw_feasible | baseline | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| heterogeneous_raw_feasible | layer_once | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| heterogeneous_raw_feasible | refresh8 | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| heterogeneous_raw_feasible | scheme_b | 1.000 | 1.000 | 1.000 | 0.400 | 1.000 | 1.000 | 160.00 |
| deadline_feasible_scheme_objective | baseline | 0.500 | 0.500 | 1.000 | 0.267 | 0.500 | 0.000 | 320.00 |
| deadline_feasible_scheme_objective | layer_once | 0.500 | 0.500 | 1.000 | 0.267 | 0.780 | 0.667 | 205.71 |
| deadline_feasible_scheme_objective | refresh8 | 0.500 | 0.500 | 1.000 | 0.267 | 0.780 | 0.667 | 205.71 |
| deadline_feasible_scheme_objective | scheme_b | 0.500 | 0.500 | 1.000 | 0.267 | 0.780 | 0.667 | 205.71 |
| hotspot_raw_infeasible | baseline | 0.714 | 0.714 | 0.625 | 0.250 | 0.750 | 1.000 | 210.53 |
| hotspot_raw_infeasible | layer_once | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID | INVALID |
| hotspot_raw_infeasible | refresh8 | 0.714 | 0.714 | 0.625 | 0.250 | 0.750 | 0.917 | 213.12 |
| hotspot_raw_infeasible | scheme_b | 0.714 | 0.714 | 0.625 | 0.250 | 0.751 | 1.000 | 210.53 |
| hotspot_deadline_infeasible | baseline | 0.500 | 0.500 | 0.625 | 0.250 | 0.584 | 0.250 | 260.00 |
| hotspot_deadline_infeasible | layer_once | 0.500 | 0.500 | 0.625 | 0.250 | 0.583 | 0.417 | 267.06 |
| hotspot_deadline_infeasible | refresh8 | 0.500 | 0.500 | 0.625 | 0.250 | 0.581 | 0.500 | 264.28 |
| hotspot_deadline_infeasible | scheme_b | 0.500 | 0.500 | 0.625 | 0.250 | 0.583 | 0.250 | 260.00 |
| single_flow_60 | baseline | 0.612 | 0.613 | 0.919 | 0.735 | 0.610 | 1.000 | 261.33 |
| single_flow_60 | layer_once | 0.612 | 0.613 | 0.919 | 0.735 | 0.610 | 1.000 | 261.33 |
| single_flow_60 | refresh8 | 0.612 | 0.613 | 0.919 | 0.735 | 0.610 | 1.000 | 261.33 |
| single_flow_60 | scheme_b | 0.612 | 0.613 | 0.919 | 0.735 | 0.610 | 1.000 | 261.33 |
| single_flow_90 | baseline | 0.408 | 0.408 | 0.918 | 0.735 | 0.406 | 0.000 | 392.00 |
| single_flow_90 | layer_once | 0.408 | 0.408 | 0.918 | 0.735 | 0.406 | 0.000 | 392.00 |
| single_flow_90 | refresh8 | 0.408 | 0.408 | 0.918 | 0.735 | 0.406 | 0.000 | 392.00 |
| single_flow_90 | scheme_b | 0.408 | 0.408 | 0.918 | 0.735 | 0.406 | 0.000 | 392.00 |
| npu50_control | baseline | 0.780 | 0.779 | 0.585 | 0.935 | 0.779 | 1.000 | 205.33 |
| npu50_control | layer_once | 0.780 | 0.779 | 0.585 | 0.935 | 0.779 | 1.000 | 205.33 |
| npu50_control | refresh8 | 0.780 | 0.779 | 0.585 | 0.935 | 0.779 | 1.000 | 205.33 |
| npu50_control | scheme_b | 0.780 | 0.779 | 0.585 | 0.935 | 0.779 | 1.000 | 205.33 |
| npu50_link_stress | baseline | 0.618 | 0.614 | 0.154 | 0.983 | 0.612 | 1.000 | 260.44 |
| npu50_link_stress | layer_once | 0.618 | 0.614 | 0.154 | 0.983 | 0.612 | 1.000 | 260.44 |
| npu50_link_stress | refresh8 | 0.618 | 0.614 | 0.154 | 0.983 | 0.612 | 1.000 | 260.44 |
| npu50_link_stress | scheme_b | 0.618 | 0.614 | 0.154 | 0.983 | 0.612 | 1.000 | 260.44 |

## Invalid data-plane runs

- `hotspot_raw_infeasible / layer_once`: `AssertionError`; see the JSON `failure` field for invariant and per-NPU drain diagnostics.

SSD satisfaction is backend service/target demand. Link satisfaction is NPU-link delivered/target demand. Both use exact overlap at the common measurement boundaries; completion-only byte counting is not used.
