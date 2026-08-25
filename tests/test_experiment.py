import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import experiment


PROFILE = (5.0, 1_000.0, 10.0, 0.017)
TABLE = {
    (3, 0): PROFILE,
    (3, 1_023): PROFILE,
    (81, 0): PROFILE,
    (81, 80_895): PROFILE,
}


class ExperimentTests(unittest.TestCase):
    def test_case_contains_only_paired_baseline_and_qos(self):
        with patch.object(experiment, "NUM_NPU", 1):
            row = experiment.run_routing_comparison_case(TABLE, 1, n_layers=1, seed=13)
        self.assertEqual(set(row), {"num_ssu", "seeds", "baseline", "qos"})
        self.assertAlmostEqual(
            row["baseline"]["avg_request_compute_fraction"],
            0.7017543859649124,
        )
        self.assertEqual(
            row["baseline"]["placement_hash"], row["qos"]["placement_hash"]
        )
        self.assertTrue(all(row["baseline"]["invariants"].values()))
        self.assertTrue(all(row["qos"]["invariants"].values()))
        self.assertEqual(row["qos"]["pressure_reports"], 3)

    def test_spec_records_fixed_8_per_io_semantics(self):
        spec = experiment.routing_comparison_spec(TABLE, (40, 56), 16, 42)
        self.assertEqual(spec["num_ssu"], [40, 56])
        self.assertEqual(spec["npu_cap_gbps"], 50.0)
        self.assertEqual(spec["qos"]["pressure_read_interval"], 8)
        self.assertEqual(spec["qos"]["path_selection"], "per_io")
        self.assertEqual(spec["qos"]["client_submit_batch_size"], 8)

    def test_cached_parallel_sweep_does_not_create_zero_worker_pool(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            spec = experiment.routing_comparison_spec(TABLE, (1,), 1, 13)
            cached = {
                "schema_version": experiment.SCHEMA_VERSION,
                "experiment": spec,
                "results": [{"num_ssu": 1}],
            }
            path.write_text(json.dumps(cached))
            with patch.object(experiment, "NUM_NPU", 1), patch.object(
                experiment, "load_bw_table_cache", return_value=TABLE
            ), patch.object(experiment, "routing_comparison_spec", return_value=spec):
                result = experiment.run_sweep(
                    path, ssu_list=(1,), n_layers=1, seed=13, workers=2
                )
            self.assertEqual(result["results"], [{"num_ssu": 1}])

    def test_plot_writes_two_strategy_comparison(self):
        data = {
            "experiment": {"n_layers": 1},
            "results": [
                {
                    "num_ssu": 1,
                    "baseline": {"avg_request_compute_fraction": 0.7},
                    "qos": {"avg_request_compute_fraction": 0.6},
                }
            ],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "plot.png"
            experiment.plot_results(data, output)
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
