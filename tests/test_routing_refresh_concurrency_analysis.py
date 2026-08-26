import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt

import analyze_routing_refresh_concurrency as analysis
import routing_refresh_concurrency_experiment as experiment


REQUEST_OFFSETS = {
    "baseline": 0.00,
    "path_rr_baseline": 0.01,
    "refresh1": 0.04,
    "refresh8": 0.02,
    "layer_once": 0.03,
    analysis.ORACLE_CANDIDATE_STRATEGY: 0.08,
}


def _request_value(seed, ssu, strategy):
    return (
        0.40
        + 0.01 * analysis.SSUS.index(ssu)
        + REQUEST_OFFSETS[strategy]
        + 0.001 * (seed - 42)
    )


def _simulation_row(seed, ssu, strategy, spec):
    request = _request_value(seed, ssu, strategy)
    pressure = {
        "baseline": 0,
        "path_rr_baseline": 0,
        "layer_once": 100,
        "refresh8": 800,
        "refresh1": 6400,
    }[strategy]
    return {
        "seed": seed,
        "num_ssu": ssu,
        "strategy": strategy,
        "kind": "simulation",
        "config": spec.config(),
        "workload_fingerprint": f"workload-{seed}",
        "placement_hash": f"placement-{seed}-{ssu}",
        "summary": {
            "avg_request_compute_fraction": request,
            "fleet_npu_compute_utilization": request / 4.0,
            "makespan_ms": 1000.0 - request,
            "pressure_reports": pressure,
            "pressure_telemetry_mb": pressure * 0.001024,
            "avg_queue_wait_ms_per_block": 2.0 - request,
            "avg_npu_link_queue_wait_ms": 1.0 + request,
            "request_compute_fraction_jain": 0.9 + request / 100,
            "category_metrics": {
                category: {"avg_request_compute_fraction": request}
                for category in analysis.CATEGORIES
            },
            "enqueued_path_ids": [0] if strategy == "baseline" else [0, 1],
            "invariants": {"blocks_conserved": True},
        },
        "request_metrics": [
            {
                "request_id": request_id,
                "arrival_delay_ms": request_id / 100.0,
                "request_npu_utilization": request,
            }
            for request_id in range(128)
        ],
    }


def _synthetic_matrix():
    specs = {strategy.name: strategy for strategy in experiment.strategy_specs()}
    rows = []
    for seed in analysis.SEEDS:
        for ssu in analysis.SSUS:
            for strategy in analysis.INPUT_STRATEGIES:
                rows.append(
                    _simulation_row(seed, ssu, strategy, specs[strategy])
                )
    return {
        "complete": True,
        "selected_strategies": list(analysis.INPUT_STRATEGIES),
        "experiment": {"runtime": experiment.runtime_config()},
        "results": rows,
    }


def _oracle_candidate_row(seed, ssu):
    request = _request_value(
        seed, ssu, analysis.ORACLE_CANDIDATE_STRATEGY
    )
    row = _simulation_row(
        seed,
        ssu,
        "baseline",
        {spec.name: spec for spec in experiment.strategy_specs()}["baseline"],
    )
    row["strategy"] = analysis.ORACLE_CANDIDATE_STRATEGY
    row["kind"] = "feasible_oracle_candidate"
    row.pop("config")
    row["summary"].update(
        {
            "policy": "per_ssd_full_visible_edf",
            "avg_request_compute_fraction": request,
            "fleet_npu_compute_utilization": request / 4.0,
            "makespan_ms": 1000.0 - request,
            "client_submit_batch_size": 1,
            "client_submit_interval_us": 0.1,
            "exact_optimum_proven": False,
            "block_conservation": {"placement_targets_preserved": True},
        }
    )
    for category in analysis.CATEGORIES:
        row["summary"]["category_metrics"][category][
            "avg_request_compute_fraction"
        ] = request
    for metric in row["request_metrics"]:
        metric["request_npu_utilization"] = request
    return row


def _synthetic_oracle_matrix():
    return {
        "complete": True,
        "experiment": {
            "runtime": experiment.runtime_config(),
            "strategy": analysis.ORACLE_CANDIDATE_STRATEGY,
            "optimality": {"exact_optimum_proven": False},
        },
        "results": [
            _oracle_candidate_row(seed, ssu)
            for seed in analysis.SEEDS
            for ssu in analysis.SSUS
        ],
    }


def _validated_envelope():
    index = analysis.validate(_synthetic_matrix())
    candidates = analysis.validate_oracle_candidates(
        _synthetic_oracle_matrix(), index
    )
    selections = analysis.add_oracle_envelope(index, candidates)
    return index, selections


class RoutingRefreshAnalysisTests(unittest.TestCase):
    def test_validate_accepts_exact_70_row_contract(self):
        data = _synthetic_matrix()
        index = analysis.validate(data)
        self.assertEqual(len(index), 70)

        invalid = _synthetic_matrix()
        invalid["results"].pop()
        with self.assertRaisesRegex(AssertionError, "matrix key mismatch"):
            analysis.validate(invalid)

    def test_validate_rejects_route_cadence_profile_and_path0_drift(self):
        invalid = _synthetic_matrix()
        row = next(
            row for row in invalid["results"] if row["strategy"] == "refresh1"
        )
        row["config"]["client_io"]["pressure_window_io"] = 8
        with self.assertRaisesRegex(AssertionError, "pressure cadence mismatch"):
            analysis.validate(invalid)

        invalid = _synthetic_matrix()
        row = next(
            row
            for row in invalid["results"]
            if row["strategy"] == "path_rr_baseline"
        )
        row["config"]["client_io"]["path_selection_mode"] = "fixed_path_zero"
        with self.assertRaisesRegex(AssertionError, "Path-selection mode mismatch"):
            analysis.validate(invalid)

        invalid = _synthetic_matrix()
        row = next(
            row for row in invalid["results"] if row["strategy"] == "refresh8"
        )
        row["config"]["category_cir_gbps"] = [20.0, 4.0, 12.0, 4.0]
        with self.assertRaisesRegex(AssertionError, "static CIR mismatch"):
            analysis.validate(invalid)

        invalid = _synthetic_matrix()
        row = next(
            row for row in invalid["results"] if row["strategy"] == "baseline"
        )
        row["summary"]["enqueued_path_ids"] = [0, 1]
        with self.assertRaisesRegex(AssertionError, "Path other than Path 0"):
            analysis.validate(invalid)

    def test_aggregate_and_comparisons_use_both_seeds(self):
        index, selections = _validated_envelope()
        values = analysis.aggregate(index)
        compared = analysis.comparisons(values, index)
        row = values[(8, "refresh1")]
        expected = (
            _request_value(42, 8, "refresh1")
            + _request_value(43, 8, "refresh1")
        ) / 2.0
        self.assertAlmostEqual(row["request_mean"], expected)
        self.assertEqual(row["pressure_reports_mean"], 6400)
        self.assertAlmostEqual(
            row["pressure_telemetry_gib_mean"],
            6400 * 256 * 4 / (1024**3),
        )
        oracle = values[(8, analysis.ORACLE_STRATEGY)]
        self.assertEqual(oracle["kind"], "feasible_oracle_envelope")
        self.assertAlmostEqual(
            oracle["request_mean"],
            (
                _request_value(42, 8, analysis.ORACLE_CANDIDATE_STRATEGY)
                + _request_value(43, 8, analysis.ORACLE_CANDIDATE_STRATEGY)
            )
            / 2.0,
        )
        self.assertEqual(
            selections["42:8"]["chosen_candidate"],
            analysis.ORACLE_CANDIDATE_STRATEGY,
        )
        self.assertAlmostEqual(
            compared["refresh1_minus_refresh8"][
                "cross_ssu_request_delta_pp"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            compared["rr_minus_final_baseline"][
                "cross_ssu_request_delta_pp"
            ],
            1.0,
        )
        paired = compared["refresh1_minus_refresh8"]["by_ssu"]["8"][
            "paired_requests"
        ]
        self.assertEqual(paired["count"], 256)
        self.assertEqual(paired["improved"], 256)

    def test_overview_contains_exactly_the_final_six_curves(self):
        index, _ = _validated_envelope()
        values = analysis.aggregate(index)
        figure, axis = plt.subplots()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            analysis.plt, "subplots", return_value=(figure, axis)
        ), mock.patch.object(analysis.plt, "close"):
            analysis.plot_overview(values, Path(directory) / "plot.png")
        self.assertEqual(len(axis.lines), 6)
        self.assertEqual(
            [line.get_label() for line in axis.lines],
            [analysis.LABELS[strategy] for strategy in analysis.STRATEGIES],
        )
        self.assertEqual(axis.lines[-1].get_linestyle(), "-")
        self.assertEqual(len(axis.collections), 0)
        self.assertEqual(axis.get_ylabel(), "Average NPU Utilization (%)")
        self.assertEqual(axis.get_ylim(), (0.0, 100.0))
        self.assertNotIn("0.1", axis.get_title())
        self.assertNotIn("0–5", axis.get_title())
        plt.close(figure)

    def test_system_plot_includes_feasible_oracle_metrics(self):
        index, _ = _validated_envelope()
        values = analysis.aggregate(index)
        figure, axes = plt.subplots(1, 3)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            analysis.plt, "subplots", return_value=(figure, axes)
        ), mock.patch.object(analysis.plt, "close"):
            analysis.plot_system_metrics(values, Path(directory) / "plot.png")
        for axis in axes:
            self.assertEqual(len(axis.lines), 6)
            self.assertEqual(
                [line.get_label() for line in axis.lines],
                [analysis.LABELS[strategy] for strategy in analysis.STRATEGIES],
            )
        plt.close(figure)

    def test_end_to_end_writes_json_report_and_three_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "results.json"
            output_dir = root / "output"
            input_path.write_text(json.dumps(_synthetic_matrix()))
            oracle_path = root / "capacity_constrained_oracle_results.json"
            oracle_path.write_text(json.dumps(_synthetic_oracle_matrix()))
            result = analysis.analyze(input_path, output_dir)
            saved = json.loads((output_dir / "analysis.json").read_text())
            report = (output_dir / "report.md").read_text()
            self.assertTrue(
                (output_dir / "01_routing_refresh_finite_issue.png").exists()
            )
            self.assertTrue(
                (output_dir / "02_simulation_system_metrics.png").exists()
            )
            self.assertTrue(
                (
                    output_dir
                    / "03_strategy_deltas_vs_final_baseline.png"
                ).exists()
            )
        self.assertEqual(result["strategies"], list(analysis.STRATEGIES))
        self.assertEqual(
            result["simulation_strategies"],
            list(analysis.SIMULATION_STRATEGIES),
        )
        self.assertEqual(saved["ssu_list"], list(analysis.SSUS))
        self.assertIn("Final Path0", report)
        self.assertIn("RR", report)
        self.assertIn("Capacity-constrained Oracle", report)
        self.assertNotIn("91.842%", report)
        self.assertIn("fleet", report)


if __name__ == "__main__":
    unittest.main()
