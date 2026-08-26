import copy
import json
from pathlib import Path
import tempfile
import unittest

from validation_analysis import (
    EXPECTED_STRATEGIES,
    FORMAL_SSUS,
    analyze_sources,
    expected_seed_bundle,
    run_validation,
    validate_sources,
)


REQUEST_GAINS = {
    "baseline": 0.0,
    "current_refresh8": 0.01,
    "tune__low_protect_cir_20_6_8_6_current_paths": 0.06,
    "tune__low_protect_cir_20_5_10_5_paths_12_5_10_5": 0.04,
    "demand_maxmin": 0.05,
}
FLEET_GAINS = {
    "baseline": 0.0,
    "current_refresh8": 0.01,
    "tune__low_protect_cir_20_6_8_6_current_paths": 0.04,
    "tune__low_protect_cir_20_5_10_5_paths_12_5_10_5": 0.03,
    "demand_maxmin": 0.05,
}


def make_row(seed, ssu, strategy, request_value, fleet_value):
    workload_hash = "workload-%d" % seed
    placement_hash = "placement-%d-%d" % (seed, ssu)
    return {
        "num_ssu": ssu,
        "strategy": strategy,
        "family": "synthetic",
        "kind": "simulation",
        "config": {},
        "seeds": expected_seed_bundle(seed),
        "workload_fingerprint": workload_hash,
        "placement_hash": placement_hash,
        "summary": {
            "avg_request_compute_fraction": request_value,
            "fleet_npu_compute_utilization": fleet_value,
            "workload_fingerprint": workload_hash,
            "placement_hash": placement_hash,
        },
        "request_metrics": [
            {"request_id": request_id} for request_id in range(128)
        ],
        "wall_time_s": 0.0,
    }


def make_result(seed, exact_selected):
    selected = list(EXPECTED_STRATEGIES)
    if not exact_selected:
        selected.append("unrelated_full_matrix_strategy")
    rows = []
    seed_scale = 1.0 if seed == 42 else 0.8
    for ssu in FORMAL_SSUS:
        baseline_request = 0.50 + (ssu - 40) * 0.001
        baseline_fleet = 0.40 + (ssu - 40) * 0.0005
        for strategy in EXPECTED_STRATEGIES:
            rows.append(
                make_row(
                    seed,
                    ssu,
                    strategy,
                    baseline_request + seed_scale * REQUEST_GAINS[strategy],
                    baseline_fleet + seed_scale * FLEET_GAINS[strategy],
                )
            )
        if not exact_selected:
            rows.append(
                make_row(
                    seed,
                    ssu,
                    "unrelated_full_matrix_strategy",
                    baseline_request,
                    baseline_fleet,
                )
            )
    return {
        "schema_version": 2,
        "experiment": {
            "runtime": {
                "mode": "formal",
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": [40, 56, 80],
                "seeds": expected_seed_bundle(seed),
            }
        },
        "selected_strategies": selected,
        "complete": True,
        "results": rows,
    }


class ValidationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.seed42 = make_result(42, exact_selected=False)
        self.seed43 = make_result(43, exact_selected=True)

    def test_analysis_finds_winner_and_seed43_retains_direction_and_rank(self):
        analysis = analyze_sources(self.seed42, self.seed43)
        robustness = analysis["seed43_robustness"]
        self.assertEqual(
            robustness["seed42_request_winner"],
            "tune__low_protect_cir_20_6_8_6_current_paths",
        )
        self.assertTrue(robustness["request_direction_held_all_ssus"])
        self.assertTrue(robustness["request_rank_held"])
        self.assertEqual(robustness["request_rank"], {"42": 1, "43": 1})
        self.assertTrue(robustness["fleet_direction_held_all_ssus"])
        self.assertTrue(robustness["fleet_rank_held"])
        self.assertEqual(len(analysis["comparisons"]), 30)
        self.assertIn("do not establish statistical significance", analysis["interpretation"])

    def test_contract_strategy_and_pairing_failures_are_rejected(self):
        cases = []

        incomplete = copy.deepcopy(self.seed42)
        incomplete["complete"] = False
        cases.append((incomplete, self.seed43, "not complete"))

        wrong_contract = copy.deepcopy(self.seed42)
        wrong_contract["experiment"]["runtime"]["num_npu"] = 127
        cases.append((wrong_contract, self.seed43, "num_npu"))

        wrong_seed = copy.deepcopy(self.seed43)
        wrong_seed["experiment"]["runtime"]["seeds"]["workload"] = 42
        cases.append((self.seed42, wrong_seed, "seed bundle"))

        wrong_scope = copy.deepcopy(self.seed43)
        wrong_scope["selected_strategies"].append("extra")
        cases.append((self.seed42, wrong_scope, "exactly the five"))

        unpaired = copy.deepcopy(self.seed43)
        target = next(
            row
            for row in unpaired["results"]
            if row["num_ssu"] == 56 and row["strategy"] == "demand_maxmin"
        )
        target["placement_hash"] = "wrong-placement"
        target["summary"]["placement_hash"] = "wrong-placement"
        cases.append((self.seed42, unpaired, "placement hashes are not paired"))

        for seed42, seed43, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_sources(seed42, seed43)

    def test_run_writes_json_markdown_and_plot_from_temporary_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed42_path = root / "seed42.json"
            seed43_path = root / "seed43.json"
            output_dir = root / "output"
            seed42_path.write_text(json.dumps(self.seed42))
            seed43_path.write_text(json.dumps(self.seed43))

            run_validation(seed42_path, seed43_path, output_dir)

            json_path = output_dir / "validation_analysis.json"
            report_path = output_dir / "validation_report.md"
            figure_path = output_dir / "12_seed_validation.png"
            self.assertTrue(json_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertGreater(figure_path.stat().st_size, 1_000)
            written = json.loads(json_path.read_text())
            self.assertEqual(written["contract"]["seeds"], [42, 43])
            report = report_path.read_text()
            self.assertIn("does not establish statistical significance", report)
            self.assertIn("12_seed_validation.png", report)


if __name__ == "__main__":
    unittest.main()
