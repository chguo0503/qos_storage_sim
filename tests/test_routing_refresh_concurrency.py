import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import routing_refresh_concurrency_experiment as experiment
import sim


SIMULATION_STRATEGIES = (
    "baseline",
    "path_rr_baseline",
    "refresh1",
    "refresh8",
    "layer_once",
)
ALL_STRATEGIES = SIMULATION_STRATEGIES


def _two_request_prepared():
    block_size_gb = 0.001
    request_loads = []
    placement = {}
    for request_id in range(2):
        request_loads.append(
            {
                "request_id": request_id,
                "npu_id": request_id,
                "seq_len_k": 3,
                "nql": 0,
                "per_layer_us": 1_000.0 + 100.0 * request_id,
                "per_layer_kv_gb": 4 * block_size_gb,
                "required_bw_input_gbps": 10.0,
                "category": "SS",
                "arrival_time": 0.05 * request_id,
            }
        )
        placement[request_id] = {
            0: tuple((0, block_size_gb) for _ in range(4))
        }
    loads = tuple(request_loads)
    return sim.PreparedSimulationInputs(
        request_loads=loads,
        placement_by_request=placement,
        workload_seed=1,
        placement_seed=2,
        workload_hash=sim.workload_fingerprint(loads),
        placement_hash=sim.placement_fingerprint(placement),
        n_layers=1,
        num_disk=1,
        placement_mode="manual",
        arrival_delay_seed=3,
        arrival_delay_max_ms=0.05,
    )


def _run_fixed_path_zero(name):
    _, result = sim.simulate_continuous(
        {},
        policy=sim.POLICY_QOS_STATIC_CIR,
        num_npu=2,
        num_disk=1,
        n_layers=1,
        qos_config=experiment.STATIC_PROFILE.hardware_config(),
        client_io_config=sim.ClientIOConfig(
            name=name,
            pressure_window_io=None,
            submit_batch_size=1,
            issue_interval_us=0.1,
            path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
        ),
        submit_order_seed=4,
        prepared_inputs=_two_request_prepared(),
    )
    return result


class RoutingRefreshContractTests(unittest.TestCase):
    def test_registry_contains_exactly_the_five_routing_strategies(self):
        strategies = experiment.strategy_specs()
        self.assertEqual(
            [strategy.name for strategy in strategies], list(ALL_STRATEGIES)
        )
        self.assertEqual(
            [strategy.policy for strategy in strategies],
            [sim.POLICY_QOS_STATIC_CIR] * 5,
        )
        self.assertEqual(
            [strategy.path_selection_mode for strategy in strategies],
            [
                sim.PATH_SELECTION_FIXED_PATH_ZERO,
                sim.PATH_SELECTION_STATELESS_RR,
                sim.PATH_SELECTION_PRESSURE_AWARE,
                sim.PATH_SELECTION_PRESSURE_AWARE,
                sim.PATH_SELECTION_PRESSURE_AWARE,
            ],
        )
        self.assertEqual(
            [strategy.pressure_window_io for strategy in strategies],
            [None, None, 1, 8, None],
        )

    def test_formal_runtime_and_finite_issue_contract(self):
        runtime = experiment.runtime_config()
        self.assertEqual(runtime["num_npu"], 128)
        self.assertEqual(runtime["n_layers"], 16)
        self.assertEqual(
            runtime["ssu_list"], [8, 16, 28, 40, 56, 80, 112]
        )
        self.assertEqual(runtime["seeds"], [42, 43])
        self.assertEqual(runtime["arrival_delay_ms"], [0.0, 5.0])
        self.assertEqual(runtime["client_submit_batch_size"], 1)
        self.assertEqual(runtime["client_issue_interval_us"], 0.1)
        for strategy in experiment.strategy_specs():
            client = strategy.client_config()
            self.assertEqual(client.submit_batch_size, 1)
            self.assertEqual(client.issue_interval_us, 0.1)

    def test_all_simulations_share_optimized_static_hardware_and_policy(self):
        profile = experiment.STATIC_PROFILE
        self.assertEqual(profile.category_cir_gbps, (20.0, 6.0, 8.0, 6.0))
        self.assertEqual(profile.category_paths_per_group, (12, 4, 12, 4))
        self.assertAlmostEqual(sum(profile.hardware_config().path_cirs), 40.0)

        simulations = list(experiment.strategy_specs())
        self.assertEqual(len(simulations), 5)
        self.assertEqual(
            {strategy.policy for strategy in simulations},
            {sim.POLICY_QOS_STATIC_CIR},
        )
        configs = [strategy.config() for strategy in simulations]
        self.assertEqual(
            {config["profile_name"] for config in configs}, {profile.name}
        )
        self.assertEqual(
            {tuple(config["category_cir_gbps"]) for config in configs},
            {(20.0, 6.0, 8.0, 6.0)},
        )
        self.assertEqual(
            {tuple(config["category_paths_per_group"]) for config in configs},
            {(12, 4, 12, 4)},
        )

    def test_experiment_metadata_isolates_only_npu_path_selection(self):
        runtime = experiment.runtime_config()
        spec = experiment.experiment_spec({(1, 2): (3.0, 4.0)}, runtime)
        control = spec["controlled_comparison"]
        self.assertEqual(control["isolated_difference"], "NPU Path-ID selection")
        self.assertEqual(
            control["shared_simulation_policy"], sim.POLICY_QOS_STATIC_CIR
        )
        self.assertEqual(control["final_baseline_path"], 0)
        self.assertEqual(control["simulation_submit_batch_size"], 1)
        self.assertEqual(control["simulation_issue_interval_us"], 0.1)
        self.assertEqual(
            [row["name"] for row in spec["available_strategies"]],
            list(ALL_STRATEGIES),
        )

    def test_fixed_path_zero_enqueues_only_path_zero(self):
        result = _run_fixed_path_zero("final_baseline")
        self.assertEqual(result["path_selection"], "fixed_path_zero")
        self.assertEqual(
            result["qos_client_routing"],
            "all_npus_all_io_fixed_to_path_zero",
        )
        self.assertEqual(result["pressure_read_interval"], None)
        self.assertEqual(
            sum(row["pressure_reports"] for row in result["disk_stats"]), 0
        )
        self.assertEqual(
            sorted(
                {
                    path_id
                    for row in result["disk_stats"]
                    for path_id in row["enqueued_path_ids"]
                }
            ),
            [0],
        )

    def test_same_path_selection_produces_identical_core_results(self):
        baseline = _run_fixed_path_zero("final_baseline")
        forced_static = _run_fixed_path_zero("qos_static_forced_path_zero")

        self.assertNotEqual(
            baseline["pressure_read_mode"], forced_static["pressure_read_mode"]
        )
        baseline_without_label = dict(baseline)
        forced_without_label = dict(forced_static)
        baseline_without_label.pop("pressure_read_mode")
        forced_without_label.pop("pressure_read_mode")
        self.assertEqual(baseline_without_label, forced_without_label)


class RoutingRefreshCheckpointTests(unittest.TestCase):
    def test_checkpoint_runs_70_cases_and_reuses_matching_cache(self):
        def fake_prepare(table, runtime, seed, num_ssu):
            del table, runtime
            return mock.Mock(
                workload_hash=f"workload-{seed}",
                placement_hash=f"placement-{seed}-{num_ssu}",
            )

        def fake_case(table, runtime, seed, num_ssu, strategy, prepared=None):
            del table, runtime
            return {
                "seed": seed,
                "num_ssu": num_ssu,
                "strategy": strategy.name,
                "kind": "simulation",
                "config": strategy.config(),
                "seeds": experiment.seed_bundle(seed),
                "workload_fingerprint": prepared.workload_hash,
                "placement_hash": prepared.placement_hash,
                "summary": {},
                "request_metrics": [],
                "wall_time_s": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with mock.patch.object(
                experiment.sim,
                "load_bw_table_cache",
                return_value={(1, 2): (3.0, 4.0)},
            ), mock.patch.object(
                experiment, "prepare", side_effect=fake_prepare
            ) as prepare_mock, mock.patch.object(
                experiment, "run_strategy_case", side_effect=fake_case
            ) as case_mock, mock.patch.object(experiment, "_print_completed"):
                result = experiment.run_matrix(path, workers=1)

            self.assertTrue(result["complete"])
            self.assertEqual(len(result["results"]), 70)
            self.assertEqual(prepare_mock.call_count, 14)
            self.assertEqual(case_mock.call_count, 70)
            self.assertEqual(json.loads(path.read_text()), result)

            with mock.patch.object(
                experiment.sim,
                "load_bw_table_cache",
                return_value={(1, 2): (3.0, 4.0)},
            ), mock.patch.object(experiment, "prepare") as prepare_again:
                cached = experiment.run_matrix(path, workers=1)
            self.assertEqual(cached, result)
            prepare_again.assert_not_called()


if __name__ == "__main__":
    unittest.main()
