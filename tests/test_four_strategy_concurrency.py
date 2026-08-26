import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import four_strategy_concurrency_experiment as experiment
import sim


class FourStrategyContractTests(unittest.TestCase):
    def test_registry_contains_exactly_the_requested_four_strategies(self):
        strategies = experiment.strategy_specs()
        self.assertEqual(
            [row.name for row in strategies],
            [
                "baseline",
                "static_cir_before",
                "static_cir_after",
                "fluid_no_contention_bound",
            ],
        )
        self.assertEqual(
            [row.kind for row in strategies],
            ["simulation", "simulation", "simulation", "upper_bound"],
        )
        self.assertEqual(
            [row.policy for row in strategies[:3]],
            [
                sim.POLICY_BASELINE_BYPASS,
                sim.POLICY_QOS_STATIC_CIR,
                sim.POLICY_QOS_STATIC_CIR,
            ],
        )

    def test_formal_runtime_is_fixed_to_both_seeds_and_requested_matrix(self):
        runtime = experiment.runtime_config()
        self.assertEqual(runtime["mode"], "formal")
        self.assertEqual(runtime["num_npu"], 128)
        self.assertEqual(runtime["n_layers"], 16)
        self.assertEqual(runtime["ssu_list"], [40, 56, 80])
        self.assertEqual(runtime["seeds"], [42, 43])
        self.assertEqual(runtime["arrival_delay_ms"], [0.0, 5.0])
        self.assertEqual(runtime["client_submit_batch_size"], 1)
        self.assertEqual(runtime["client_issue_interval_us"], 0.1)

    def test_simulations_share_finite_issue_and_static_refresh8_contract(self):
        baseline, before, after, bound = experiment.strategy_specs()
        baseline_client = baseline.client_config()
        self.assertEqual(baseline_client.submit_batch_size, 1)
        self.assertEqual(baseline_client.issue_interval_us, 0.1)
        self.assertIsNone(baseline_client.pressure_window_io)

        for strategy in (before, after):
            client = strategy.client_config()
            self.assertEqual(client.submit_batch_size, 1)
            self.assertEqual(client.issue_interval_us, 0.1)
            self.assertEqual(client.pressure_window_io, 8)
            self.assertEqual(client.path_pool_mode, "category_shared")
        self.assertIsNone(bound.client_config())

    def test_static_profiles_isolate_cir_and_keep_path_allocation_equal(self):
        before = experiment.BEFORE_PROFILE
        after = experiment.AFTER_PROFILE
        self.assertEqual(before.category_cir_gbps, (20.0, 4.0, 12.0, 4.0))
        self.assertEqual(after.category_cir_gbps, (20.0, 6.0, 8.0, 6.0))
        self.assertEqual(
            before.category_paths_per_group,
            after.category_paths_per_group,
        )
        self.assertEqual(before.category_paths_per_group, (12, 4, 12, 4))
        self.assertAlmostEqual(sum(before.hardware_config().path_cirs), 40.0)
        self.assertAlmostEqual(sum(after.hardware_config().path_cirs), 40.0)

    def test_experiment_metadata_records_controlled_comparison(self):
        runtime = experiment.runtime_config()
        spec = experiment.experiment_spec({(1, 2): (3.0, 4.0)}, runtime)
        self.assertEqual(spec["formal_contract"]["seeds"], [42, 43])
        control = spec["controlled_comparison"]
        self.assertEqual(control["simulation_submit_batch_size"], 1)
        self.assertEqual(control["simulation_issue_interval_us"], 0.1)
        self.assertEqual(
            control["static_routing"],
            "refresh_path_pressure_every_8_ios",
        )
        self.assertEqual(
            [row["name"] for row in spec["available_strategies"]],
            [row.name for row in experiment.strategy_specs()],
        )


class PairingAndCheckpointTests(unittest.TestCase):
    def test_pair_validation_rejects_fingerprint_divergence(self):
        selected = experiment.strategy_specs()
        runtime = {"seeds": [42], "ssu_list": [40]}
        rows = [
            {
                "seed": 42,
                "num_ssu": 40,
                "strategy": strategy.name,
                "workload_fingerprint": "workload",
                "placement_hash": "placement",
            }
            for strategy in selected
        ]
        experiment._validate_paired_inputs(rows, runtime, selected)
        rows[-1]["placement_hash"] = "different"
        with self.assertRaises(AssertionError):
            experiment._validate_paired_inputs(rows, runtime, selected)

    def test_checkpoint_is_complete_and_reused_when_fingerprint_matches(self):
        prepared_by_pair = {}

        def fake_prepare(table, runtime, seed, num_ssu):
            del table, runtime
            key = (seed, num_ssu)
            prepared = mock.Mock(
                workload_hash=f"workload-{seed}",
                placement_hash=f"placement-{seed}-{num_ssu}",
            )
            prepared_by_pair[key] = prepared
            return prepared

        def fake_case(table, runtime, seed, num_ssu, strategy, prepared=None):
            del table, runtime
            return {
                "seed": seed,
                "num_ssu": num_ssu,
                "strategy": strategy.name,
                "kind": strategy.kind,
                "config": strategy.config(),
                "seeds": experiment.seed_bundle(seed),
                "workload_fingerprint": prepared.workload_hash,
                "placement_hash": prepared.placement_hash,
                "summary": {
                    "avg_request_compute_fraction_upper_bound": 0.9
                },
                "request_metrics": [],
                "wall_time_s": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with mock.patch.object(
                experiment.sim, "load_bw_table_cache", return_value={(1, 2): (3, 4)}
            ), mock.patch.object(
                experiment, "prepare", side_effect=fake_prepare
            ) as prepare_mock, mock.patch.object(
                experiment, "run_strategy_case", side_effect=fake_case
            ) as case_mock, mock.patch.object(
                experiment, "_print_completed"
            ):
                result = experiment.run_matrix(path, workers=1)

            self.assertTrue(result["complete"])
            self.assertEqual(len(result["results"]), 24)
            self.assertEqual(prepare_mock.call_count, 6)
            self.assertEqual(case_mock.call_count, 24)
            self.assertEqual(
                result["selected_strategies"],
                [row.name for row in experiment.strategy_specs()],
            )
            self.assertEqual(json.loads(path.read_text()), result)

            with mock.patch.object(
                experiment.sim, "load_bw_table_cache", return_value={(1, 2): (3, 4)}
            ), mock.patch.object(
                experiment, "prepare"
            ) as prepare_again, mock.patch.object(
                experiment, "run_strategy_case"
            ) as case_again:
                cached = experiment.run_matrix(path, workers=1)
            self.assertEqual(cached, result)
            prepare_again.assert_not_called()
            case_again.assert_not_called()


if __name__ == "__main__":
    unittest.main()
