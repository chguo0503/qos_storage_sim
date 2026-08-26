import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib.pyplot as plt

import analyze_four_strategy_concurrency as analysis
import four_strategy_concurrency_experiment as experiment


def _request_value(seed, num_ssu, strategy):
    ssu_index = analysis.SSUS.index(num_ssu)
    strategy_index = analysis.STRATEGIES.index(strategy)
    return 0.40 + 0.01 * ssu_index + 0.02 * strategy_index + 0.001 * (
        seed - analysis.SEEDS[0]
    )


def _synthetic_matrix():
    rows = []
    specs = {strategy.name: strategy for strategy in experiment.strategy_specs()}
    for seed in analysis.SEEDS:
        for num_ssu in analysis.SSUS:
            for strategy in analysis.STRATEGIES:
                request = _request_value(seed, num_ssu, strategy)
                upper_bound = strategy == "fluid_no_contention_bound"
                summary = (
                    {"avg_request_compute_fraction_upper_bound": request}
                    if upper_bound
                    else {
                        "avg_request_compute_fraction": request,
                        "fleet_npu_compute_utilization": request / 2.0,
                        "makespan_ms": 1_000.0 + 2.0 * (seed - 42),
                        "category_metrics": {
                            category: {
                                "avg_request_compute_fraction": request,
                                "avg_io_wait_total_ms": 1.0 / request,
                            }
                            for category in analysis.CATEGORIES
                        },
                        "invariants": {"command_conservation": True},
                    }
                )
                rows.append(
                    {
                        "seed": seed,
                        "num_ssu": num_ssu,
                        "strategy": strategy,
                        "kind": "upper_bound" if upper_bound else "simulation",
                        "config": specs[strategy].config(),
                        "workload_fingerprint": f"workload-{seed}",
                        "placement_hash": f"placement-{seed}-{num_ssu}",
                        "summary": summary,
                        "request_metrics": (
                            []
                            if upper_bound
                            else [
                                {
                                    "request_id": 0,
                                    "category": "LL",
                                    "arrival_delay_ms": 0.0,
                                    "request_compute_ms": 100.0,
                                    "io_wait_total_ms": 900.0
                                    + 2.0 * (seed - 42),
                                }
                            ]
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "complete": True,
        "experiment": {
            "runtime": {
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": list(analysis.SSUS),
                "seeds": list(analysis.SEEDS),
                "arrival_delay_ms": [0.0, 5.0],
            }
        },
        "selected_strategies": list(analysis.STRATEGIES),
        "results": rows,
    }


def _synthetic_zero_issue():
    source_names = {
        "baseline": "baseline",
        "static_cir_before": "current_refresh8",
        "static_cir_after": "tune__low_protect_cir_20_6_8_6_current_paths",
    }
    rows = []
    for num_ssu in analysis.SSUS:
        for strategy, source in source_names.items():
            strategy_index = analysis.STRATEGIES.index(strategy)
            rows.append(
                {
                    "num_ssu": num_ssu,
                    "strategy": source,
                    "summary": {
                        "avg_request_compute_fraction": (
                            _request_value(42, num_ssu, strategy)
                            - 0.001 * (strategy_index + 1)
                        )
                    },
                }
            )
        rows.append(
            {
                "num_ssu": num_ssu,
                "strategy": "isolated_no_contention_bound",
                "summary": {
                    "avg_request_compute_fraction_upper_bound": 0.99
                },
            }
        )
    return {
        "complete": True,
        "experiment": {
            "runtime": {
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": list(analysis.SSUS),
                "seeds": {"workload": 42},
            }
        },
        "selected_strategies": [
            "baseline",
            "current_refresh8",
            "tune__low_protect_cir_20_6_8_6_current_paths",
            "isolated_no_contention_bound",
        ],
        "results": rows,
    }


class FourStrategyConcurrencyAnalysisTests(unittest.TestCase):
    def test_validate_accepts_exact_formal_24_row_contract(self):
        data = _synthetic_matrix()
        index = analysis.validate(data)

        expected = {
            (seed, num_ssu, strategy)
            for seed in analysis.SEEDS
            for num_ssu in analysis.SSUS
            for strategy in analysis.STRATEGIES
        }
        self.assertEqual(len(data["results"]), 24)
        self.assertEqual(set(index), expected)

        invalid = _synthetic_matrix()
        invalid["selected_strategies"] = list(reversed(analysis.STRATEGIES))
        with self.assertRaisesRegex(AssertionError, "strategy set mismatch"):
            analysis.validate(invalid)

        invalid = _synthetic_matrix()
        invalid["results"].pop()
        with self.assertRaisesRegex(AssertionError, "matrix key mismatch"):
            analysis.validate(invalid)

    def test_validate_rejects_strategy_identity_and_client_contract_drift(self):
        mutations = (
            (
                "wrong kind",
                "baseline",
                lambda row: row.__setitem__("kind", "upper_bound"),
                "strategy kind mismatch",
            ),
            (
                "wrong static policy",
                "static_cir_before",
                lambda row: row["config"].__setitem__(
                    "policy", "baseline_bypass"
                ),
                "static policy mismatch",
            ),
            (
                "wrong refresh cadence",
                "static_cir_after",
                lambda row: row["config"]["client_io"].__setitem__(
                    "pressure_window_io", 1
                ),
                "static pressure cadence mismatch",
            ),
            (
                "baseline pressure read",
                "baseline",
                lambda row: row["config"]["client_io"].__setitem__(
                    "pressure_window_io", 8
                ),
                "baseline read Path pressure",
            ),
        )
        for label, strategy, mutate, message in mutations:
            with self.subTest(label=label):
                invalid = _synthetic_matrix()
                row = next(
                    row
                    for row in invalid["results"]
                    if row["strategy"] == strategy
                )
                mutate(row)
                with self.assertRaisesRegex(AssertionError, message):
                    analysis.validate(invalid)

    def test_aggregate_uses_both_seeds_and_preserves_bound_na(self):
        index = analysis.validate(_synthetic_matrix())
        values = analysis.aggregate(index)

        for num_ssu in analysis.SSUS:
            for strategy in analysis.STRATEGIES:
                row = values[(num_ssu, strategy)]
                seed_42 = _request_value(42, num_ssu, strategy)
                seed_43 = _request_value(43, num_ssu, strategy)
                self.assertEqual(
                    row["request_by_seed"], {42: seed_42, 43: seed_43}
                )
                self.assertAlmostEqual(
                    row["request_mean"], (seed_42 + seed_43) / 2.0
                )
                self.assertEqual(row["request_min"], seed_42)
                self.assertEqual(row["request_max"], seed_43)

                if strategy == "fluid_no_contention_bound":
                    self.assertIsNone(row["fleet_mean"])
                    self.assertIsNone(row["makespan_mean_ms"])
                else:
                    self.assertAlmostEqual(
                        row["fleet_mean"], (seed_42 + seed_43) / 4.0
                    )
                    self.assertAlmostEqual(row["makespan_mean_ms"], 1_001.0)
                    self.assertEqual(
                        row["category_request_mean"],
                        {
                            category: (seed_42 + seed_43) / 2.0
                            for category in analysis.CATEGORIES
                        },
                    )
                    self.assertEqual(
                        row["category_io_wait_mean_ms"],
                        {
                            category: (
                                1.0 / seed_42 + 1.0 / seed_43
                            )
                            / 2.0
                            for category in analysis.CATEGORIES
                        },
                    )

    def test_comparisons_and_tail_diagnostics_are_paired(self):
        index = analysis.validate(_synthetic_matrix())
        values = analysis.aggregate(index)
        compared = analysis.comparisons(values)
        tails = analysis.tail_diagnostics(index)

        after_before = compared["static_after_minus_before"]
        after_baseline = compared["static_after_minus_baseline"]
        self.assertAlmostEqual(
            after_before["cross_ssu_request_delta_pp"], 2.0
        )
        self.assertAlmostEqual(
            after_baseline["cross_ssu_request_delta_pp"], 4.0
        )
        self.assertEqual(len(tails), 18)
        self.assertEqual(tails["42:40:baseline"]["request_id"], 0)

    def test_plot_contains_only_the_requested_four_strategy_curves(self):
        values = analysis.aggregate(analysis.validate(_synthetic_matrix()))
        figure, axis = plt.subplots()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            analysis.plt, "subplots", return_value=(figure, axis)
        ), mock.patch.object(analysis.plt, "close"):
            analysis.plot(values, Path(directory) / "plot.png")

        self.assertEqual(len(axis.lines), 4)
        self.assertEqual(
            [line.get_label() for line in axis.lines],
            [analysis.LABELS[strategy] for strategy in analysis.STRATEGIES],
        )
        self.assertEqual(
            {line.get_label() for line in axis.lines},
            {analysis.LABELS[strategy] for strategy in analysis.STRATEGIES},
        )
        plt.close(figure)

    def test_end_to_end_report_uses_seed42_zero_issue_paired_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "finite.json"
            zero_path = root / "zero.json"
            output_dir = root / "output"
            input_path.write_text(json.dumps(_synthetic_matrix()))
            zero_path.write_text(json.dumps(_synthetic_zero_issue()))

            result = analysis.analyze(input_path, zero_path, output_dir)
            report = (output_dir / "report.md").read_text()
            saved = json.loads((output_dir / "analysis.json").read_text())

        self.assertEqual(result["strategies"], list(analysis.STRATEGIES))
        self.assertEqual(saved["strategies"], list(analysis.STRATEGIES))
        self.assertEqual(len(saved["aggregated"]), 12)
        self.assertIn("comparisons", saved)
        self.assertEqual(len(saved["tail_diagnostics"]), 18)
        self.assertEqual(report.count("+0.100 pp"), 3)
        self.assertEqual(report.count("+0.200 pp"), 3)
        self.assertEqual(report.count("+0.300 pp"), 3)


if __name__ == "__main__":
    unittest.main()
