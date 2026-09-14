# 根目录测试精简记录

Executed: four focused root test files remain; 12 old files deleted; 9 test functions / 10 pytest parameterized cases retained. No new test inputs or assertions.

## 实际运行链

- results/baseline_ab128_32_ratio12_20260912/experiment.py:227 baseline fixed -> run_baseline_npu32_stress.run_case
- results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu3_seed7/run_once.py:83 once fixed -> original.run_case
- results/baseline_ab128_32_ratio12_20260912/once_per_layer_ssu4_seed7/run_once.py:85 once fixed -> original.run_case
- results/baseline_random_near_capacity_20260914/experiment.py:203 baseline/once fixed -> run_case; :257 CLI only baseline/once
- run_baseline_npu32_stress.py:293-299 -> run_coflow_experiments.run_case; :245-246 -> shared summarize_window
- run_coflow_experiments.py:160-163 REFERENCE_STRATEGIES -> shared_path_adapter; :178-184 simulate with control=None, no steady_state
- run_coflow_experiments.py:173 and :188-193 placement fingerprints checked before/after actual run
- shared_path_sim_adapter.py:191-200 baseline_path_ids / once_path_ids; :178 and :162 periodic collector; :231,:266,:305 branch out assignment/JIT/reorder
- shared_path_sim_adapter.py:168-170 equal176KiB guard applies every strategy
- run_shared_path_experiments.py:174 summarizes actual windows; distinct from run_multi_ssu_stall_experiments.summarize

`run_baseline_npu32_stress.py:296` 明确导入 `run_coflow_experiments.run_case as run_coflow`；所以保留的 coflow 小型运行检查直接覆盖两组现用运行包装器，并非仅因导入链保留。

## 逐文件处理

| 文件 | 已执行 | 依据 / 剩余内容 |
| --- | --- | --- |
| test_coflow_capacity_analysis.py | deleted | capacity_overloads / solo_read_lower_bound_ms 的区间容量证书；两组运行器及分析/画图脚本均未调用这些函数。公式的概念关联不等于生成现有结果所用的计算路径。 15项均为旧容量证书验证。 |
| test_coflow_capacity_simulation.py | deleted | 一层读完成、外置deadline、特定submit_batch_size及coflow_disk_adapter(strategy2)；现有两组使用八层逐层预取、baseline/once且无coflow重排。 5个测试函数（含参数化）均为独立旧容量实验。 |
| test_coflow_client_policy.py | deleted | choose_npu / choose_npu_pipeline / plan_global_batch / priority；两组固定NPU且策略仅baseline/once，不进入coflow全局授信分支。 9项均删。 |
| test_coflow_disk_adapter.py | deleted | coflow_disk_adapter重排、跨Path选择、动态coflow映射；两组run_coflow.run_case走REFERENCE_STRATEGIES分支，不加载该适配器。 10项均删；通用I/O守恒由保留的shared_path集成测试覆盖。 |
| test_coflow_disk_policy.py | deleted | choose_path / local_priority / coflow_priority；当前baseline原生Path0 FIFO，once用静态CIR原生后端，不启用这些coflow优先级函数。 8项均删。 |
| test_coflow_experiment_inputs.py | trimmed | placement_fingerprint、run_case确为现有两组调用；其余主要测试coflow_global旧704请求配额、218GiB/s到达重定时、5/6/7盘sweep。 保留原有placement和baseline运行包装器smoke两项；删除原类setUpClass全量旧输入构造、其余12项以及多余imports。每项原有小输入fixture和断言保持。未新增用例。 |
| test_coflow_sim_adapter.py | deleted | 所有仿真测试均进入coflow_adapter(strategy1/2/3)；归档重放也使用strategy3。固定选卡测试也不能证明其分支用于当前baseline/once。 10个测试函数均删。该文件导入test_shared_path_sim_adapter.example_trace，只是测试互相依赖，不是生产依赖。 |
| test_continuous_batch_profile_cycle_frontier.py | deleted | SteadyStateConfig.profile_cycle_probe_npu_ids / profile_cycle_period的周期前沿探针；两组run_coflow.run_case未传steady_state，独立随机有限输入也未启用周期探针。 3项均删。 |
| test_multi_ssu_experiment_inputs.py | deleted | build_workload为旧32卡6/7盘、初始每卡4请求+正到达时刻；run_case为旧native/controller分支；summarize是run_multi_ssu_stall_experiments.summarize，当前用run_shared_path_experiments.summarize_window。 7项均删，包括看似相关但被测函数不同的test_fixed_window_distinguishes_io_stall_from_no_request_idle。 |
| test_multi_ssu_npu_assignment.py | deleted | MultiNPUBacklog / assign_requests_on_arrival / choose_npu；两组固定NPU，请求不迁移、不在到达时跨卡重分配。 10项均删。 |
| test_multi_ssu_qos_controller.py | deleted | MultiSSUController / choose_coflow_rates / multi_ssu_control_events；两组control=None、无动态CIR写入。 13项均删。 |
| test_shared_path_jit_causality.py | deleted | 三个测试函数通过strategy1/2运行，验证JIT保持/延迟发布；当前baseline/once的prefetch_stats.enabled为false。 3个函数均删。one_step_prefetch的某些断言概念上通用，但当前测试实际走未启用策略，不能把整文件归为当前实验直接验证。 |
| test_shared_path_policies.py | trimmed | baseline_path_ids和once_path_ids直接调用当前路由函数；new_once预留感知、strategy1分配和strategy2重排均未用于两组。 保留 baseline_path_ids 与 once_path_ids 两项原测试；另保留原压力视图计数/weight/CIR重建测试，重命名为test_pressure_view_rebuilds_rates_as_well_as_counts，输入与断言不变。该第三项验证Once单测的合成快照夹具，不声称Once生产代码调用pressure_from_counts。其余12项删除。 |
| test_shared_path_reorder_integration.py | deleted | 前两项专门向原生后端注入固定队列验证strategy1/2重排；最后non176KiB检查对应shared_path_adapter.initialize所有策略共用的输入约束。 删除两项strategy1/2后端重排测试和strategy2上下文中的非176KiB测试；不新增或迁移策略用例。共同176KiB约束仍由生产适配器执行。 |
| test_shared_path_sim_adapter.py | trimmed | 直接运行shared_path_adapter和simulate_continuous_batch：当前两组使用同一adapter、8层、batch1、50GiB/s链路、5ms采集、跨请求L0预取。 仅保留原有baseline和once两参数I/O守恒/5ms采样/无重排测试，以及baseline观测透明性测试；删除new_once、strategy1、strategy2参数和JIT测试。未迁移其他测试，未新增输入/断言。 |
| test_shared_ssu_state.py | trimmed | PeriodicSSUState.collect/get是当前5ms快照通路，当前无write_cir。 保留快照只读和原minimum-period测试中interval_ms=4.9分支；删除write_cir测试及cir_min_interval_ms=99分支，函数改名反映采样周期含义。 |

## 历史记录与范围

Past 51-file / 159-test audit remains true at its earlier revision. Record new cleanup separately; do not rewrite its hashes or outcome. Update root README/current manifest/current file checksums to current 4-test-file layout. Removing tests does not require removing runtime source files that remain source-archive/import dependencies.

pressure_from_counts is not called by current Once production routing: shared_path_sim_adapter.py:194 calls policy_logic.pressure_snapshot. The retained pressure-view test validates the synthetic snapshots used in the retained Once routing unit test. Its old ledger-prefixed name is changed, with identical inputs and assertions.

本轮未改动生产代码、冻结运行器、数据、图像、历史清理审计；仅精简根目录测试。保留10个原有pytest例（9个函数），由主agent运行核验。

## 当前执行验证

精简后的4文件、10个测试已运行全部通过（0.73秒）。35项生产运行文件（含data）与本次修改前的Git版本逐字节相同。
