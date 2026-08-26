import copy
from pathlib import Path
import tempfile
import unittest

import analysis_experiment
import analyze_results
from strategy_profiles import (
    PRIMARY_STATIC_CANDIDATES,
    REFINEMENT_STATIC_CANDIDATES,
)


def simulation_row(ssu, strategy, family, request_fraction, fleet_fraction):
    return {
        "num_ssu": ssu,
        "strategy": strategy,
        "family": family,
        "kind": "simulation",
        "summary": {
            "avg_request_compute_fraction": request_fraction,
            "fleet_npu_compute_utilization": fleet_fraction,
        },
    }


def request_input(request_id):
    categories = ("SS", "SL", "LS", "LL")
    category = categories[request_id % len(categories)]
    actual_demand = 10.0 + 10.0 * (request_id % 6)
    per_layer_us = 1_000.0
    return {
        "request_id": request_id,
        "category": category,
        "seq_len_k": 3.0 if category[0] == "S" else 81.0,
        "nql": 0.0 if category[1] == "S" else 1_023.0,
        "required_bw_input_gbps": actual_demand + 2.0,
        "arrival_delay_ms": (request_id + 0.5) * 5.0 / 128.0,
        "per_layer_us": per_layer_us,
        "per_layer_kv_gb": actual_demand * per_layer_us / 1_000_000.0,
    }


def simulation_requests(compute_fraction):
    rows = []
    for request_id in range(128):
        row = request_input(request_id)
        row.update(
            {
                "request_npu_utilization": compute_fraction,
                "io_wait_total_ms": 16.0 * (1.0 - compute_fraction),
                "avg_ssd_queue_wait_ms": 0.2 * (1.0 - compute_fraction),
                "avg_npu_link_queue_wait_ms": 0.1 * (1.0 - compute_fraction),
                "avg_end_to_end_io_latency_ms": 0.5,
                "io_wait_L0_ms": 0.2,
                "io_wait_L1_ms": 0.1,
                "io_wait_L2plus_ms": 1.4,
            }
        )
        rows.append(row)
    return rows


def strategy_metrics(name, family):
    if name == "baseline":
        return 0.50, 0.20
    if name.startswith("tune__"):
        return 0.82, 0.18
    if name == "current_refresh8":
        return 0.75, 0.10
    if name == "current_layer_snapshot":
        return 0.73, 0.09
    if name == "current_per_io":
        return 0.77, 0.11
    if name == "demand_maxmin":
        return 0.70, 0.45
    if family == "ticket":
        return 0.74, 0.30
    if name == "per_ssd_full_visible_edf":
        return 0.86, 0.34
    if name == "global_link_aware_online":
        return 0.90, 0.36
    return 0.68, 0.24


def formal_data():
    specs = analysis_experiment.strategy_specs()
    runtime = analysis_experiment.runtime_config(False, 42)
    rows = []
    for ssu in (40, 56, 80):
        for spec in specs:
            base = {
                "num_ssu": ssu,
                "strategy": spec.name,
                "family": spec.family,
                "kind": spec.kind,
                "config": spec.config(),
                "seeds": copy.deepcopy(runtime["seeds"]),
                "workload_fingerprint": "paired-workload",
                "placement_hash": "placement-%d" % ssu,
                "wall_time_s": 1.0,
            }
            if spec.kind == "upper_bound":
                base["summary"] = {
                    "avg_request_compute_fraction_upper_bound": 0.98
                }
                base["request_metrics"] = [
                    {
                        "request_id": request_id,
                        "category": request_input(request_id)["category"],
                        "compute_fraction_upper_bound": 0.98,
                    }
                    for request_id in range(128)
                ]
            else:
                request_fraction, fleet_fraction = strategy_metrics(
                    spec.name, spec.family
                )
                base["summary"] = {
                    "policy": spec.policy,
                    "backend_model": analyze_results.EXPECTED_FORMAL_BACKEND[
                        "model"
                    ],
                    "data_plane_stages": copy.deepcopy(
                        analyze_results.EXPECTED_DATA_PLANE_STAGES
                    ),
                    "backend_capacity_gbps": 40.0,
                    "npu_bw_limit_gbps": 50.0,
                    "avg_request_compute_fraction": request_fraction,
                    "fleet_npu_compute_utilization": fleet_fraction,
                    "request_compute_fraction_jain": 0.95,
                    "makespan_ms": 100.0 + ssu,
                    "throughput_requests_per_s": 1_000.0,
                    "npu_link_utilization": 0.40,
                    "avg_npu_link_queue_wait_ms": 0.02,
                    "ssu_active_time_utilization": 0.70,
                    "ssu_effective_bandwidth_utilization": 0.65,
                    "pressure_reports": 1_000,
                    "pressure_telemetry_mb": 1.024,
                    "invariants": {"all_conserved": True},
                }
                base["request_metrics"] = simulation_requests(request_fraction)
            rows.append(base)
    return {
        "schema_version": analysis_experiment.SCHEMA_VERSION,
        "experiment": {
            "schema_version": analysis_experiment.SCHEMA_VERSION,
            "code_fingerprint": analysis_experiment._code_fingerprint(),
            "data_fingerprint": "0" * 64,
            "runtime": runtime,
            "formal_contract": {
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": [40, 56, 80],
            },
            "backend": copy.deepcopy(
                analyze_results.EXPECTED_FORMAL_BACKEND
            ),
            "available_strategies": [spec.config() for spec in specs],
        },
        "selected_strategies": [spec.name for spec in specs],
        "complete": True,
        "results": rows,
    }


class AnalysisPipelineTests(unittest.TestCase):
    def test_formal_contract_is_fixed_to_requested_matrix(self):
        runtime = analysis_experiment.runtime_config(False, 42)
        self.assertEqual(runtime["mode"], "formal")
        self.assertEqual(runtime["num_npu"], 128)
        self.assertEqual(runtime["n_layers"], 16)
        self.assertEqual(runtime["ssu_list"], [40, 56, 80])
        self.assertEqual(runtime["arrival_delay_ms"], [0.0, 5.0])

    def test_every_selected_static_profile_fills_one_40_gbps_256_path_ssd(self):
        for profile in PRIMARY_STATIC_CANDIDATES + REFINEMENT_STATIC_CANDIDATES:
            with self.subTest(profile=profile.name):
                config = profile.hardware_config()
                self.assertEqual(len(config.path_cirs), 256)
                self.assertEqual(sum(profile.category_paths_per_group), 32)
                self.assertAlmostEqual(sum(config.path_cirs), 40.0, places=12)

    def test_request_and_fleet_metrics_select_independent_winners(self):
        rows = []
        for ssu in (40, 56, 80):
            rows.extend(
                [
                    simulation_row(ssu, "baseline", "core", 0.60, 0.20),
                    simulation_row(
                        ssu, "current_layer_snapshot", "cadence", 0.81, 0.12
                    ),
                    simulation_row(
                        ssu, "current_refresh8", "cadence", 0.80, 0.10
                    ),
                    simulation_row(
                        ssu, "current_per_io", "cadence", 0.91, 0.11
                    ),
                    simulation_row(
                        ssu, "current_refresh8_batch16", "batch", 0.79, 0.40
                    ),
                    simulation_row(
                        ssu, "current_refresh8_batch32", "batch", 0.78, 0.55
                    ),
                    simulation_row(
                        ssu, "ticket_refresh8", "ticket", 0.77, 0.45
                    ),
                    simulation_row(
                        ssu, "demand_maxmin", "advanced", 0.70, 0.30
                    ),
                    simulation_row(
                        ssu, "per_ssd_full_visible_edf", "advanced", 0.99, 0.99
                    ),
                    simulation_row(
                        ssu, "global_link_aware_online", "advanced", 1.00, 1.00
                    ),
                ]
            )
        selection = analyze_results.choose_fixed_strategies(
            {"results": rows}, [40, 56, 80]
        )
        self.assertEqual(
            selection["best_nonideal_practical_request"], "current_per_io"
        )
        self.assertEqual(
            selection["best_nonideal_practical_fleet"],
            "current_refresh8_batch32",
        )
        self.assertEqual(
            selection["best_nonideal_practical"], "current_per_io"
        )
        practical_names = {
            row["strategy"] for row in selection["practical_scores"]
        }
        self.assertTrue(
            {
                "current_layer_snapshot",
                "current_refresh8",
                "current_per_io",
                "current_refresh8_batch16",
                "current_refresh8_batch32",
                "ticket_refresh8",
                "demand_maxmin",
            }.issubset(practical_names)
        )
        self.assertTrue(
            {
                "per_ssd_full_visible_edf",
                "global_link_aware_online",
            }.isdisjoint(practical_names)
        )

    def test_complete_24_by_3_matrix_passes_strict_validation(self):
        analyze_results.require_complete(formal_data())

    def test_incomplete_duplicate_and_wrong_runtime_matrices_are_rejected(self):
        incomplete = formal_data()
        incomplete["results"].pop()
        with self.assertRaisesRegex(RuntimeError, "missing or duplicate"):
            analyze_results.require_complete(incomplete)

        duplicate = formal_data()
        duplicate["results"][-1] = copy.deepcopy(duplicate["results"][0])
        with self.assertRaisesRegex(RuntimeError, "missing or duplicate"):
            analyze_results.require_complete(duplicate)

        wrong_runtime = formal_data()
        wrong_runtime["experiment"]["runtime"]["n_layers"] = 8
        with self.assertRaisesRegex(RuntimeError, "formal runtime contract"):
            analyze_results.require_complete(wrong_runtime)

    def test_schema_fingerprint_registry_and_row_tampering_are_rejected(self):
        data_schema = formal_data()
        data_schema["schema_version"] += 1
        with self.assertRaisesRegex(RuntimeError, "schema versions"):
            analyze_results.require_complete(data_schema)

        experiment_schema = formal_data()
        experiment_schema["experiment"]["schema_version"] += 1
        with self.assertRaisesRegex(RuntimeError, "schema versions"):
            analyze_results.require_complete(experiment_schema)

        fingerprint = formal_data()
        fingerprint["experiment"]["code_fingerprint"] = "stale"
        with self.assertRaisesRegex(RuntimeError, "code fingerprint"):
            analyze_results.require_complete(fingerprint)

        available = formal_data()
        available["experiment"]["available_strategies"][1][
            "description"
        ] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "available strategy configs"):
            analyze_results.require_complete(available)

        selected = formal_data()
        selected["selected_strategies"][1:3] = reversed(
            selected["selected_strategies"][1:3]
        )
        with self.assertRaisesRegex(RuntimeError, "selected strategy order"):
            analyze_results.require_complete(selected)

        row_config = formal_data()
        row_config["results"][1]["config"]["description"] = "tampered"
        with self.assertRaisesRegex(RuntimeError, "config/kind/family"):
            analyze_results.require_complete(row_config)

        row_kind = formal_data()
        row_kind["results"][1]["kind"] = "upper_bound"
        with self.assertRaisesRegex(RuntimeError, "config/kind/family"):
            analyze_results.require_complete(row_kind)

        row_family = formal_data()
        row_family["results"][1]["family"] = "advanced"
        with self.assertRaisesRegex(RuntimeError, "config/kind/family"):
            analyze_results.require_complete(row_family)

    def test_backend_stage_capacity_and_metric_tampering_are_rejected(self):
        backend = formal_data()
        backend["experiment"]["backend"]["ssd_max_active_io"] = 2
        with self.assertRaisesRegex(RuntimeError, "backend metadata"):
            analyze_results.require_complete(backend)

        stages = formal_data()
        stages["results"][0]["summary"]["data_plane_stages"]["ssd"][
            "max_active_io"
        ] = 2
        with self.assertRaisesRegex(RuntimeError, "data-plane contract"):
            analyze_results.require_complete(stages)

        capacity = formal_data()
        capacity["results"][0]["summary"]["backend_capacity_gbps"] = 39.0
        with self.assertRaisesRegex(RuntimeError, "data-plane contract"):
            analyze_results.require_complete(capacity)

        simulation_average = formal_data()
        simulation_average["results"][0]["summary"][
            "avg_request_compute_fraction"
        ] += 0.01
        with self.assertRaisesRegex(RuntimeError, "simulation summary average"):
            analyze_results.require_complete(simulation_average)

        bound_average = formal_data()
        bound_row = next(
            row
            for row in bound_average["results"]
            if row["num_ssu"] == 40
            and row["strategy"] == "isolated_no_contention_bound"
        )
        bound_row["summary"]["avg_request_compute_fraction_upper_bound"] -= 0.01
        with self.assertRaisesRegex(RuntimeError, "bound summary average"):
            analyze_results.require_complete(bound_average)

        request_bound = formal_data()
        global_row = next(
            row
            for row in request_bound["results"]
            if row["num_ssu"] == 40
            and row["strategy"] == "global_link_aware_online"
        )
        global_row["request_metrics"][0]["request_npu_utilization"] = 0.99
        global_row["request_metrics"][1]["request_npu_utilization"] = 0.81
        with self.assertRaisesRegex(RuntimeError, "paired fluid upper bound"):
            analyze_results.require_complete(request_bound)

    def test_hash_request_jitter_and_upper_bound_violations_are_rejected(self):
        wrong_hash = formal_data()
        wrong_hash["results"][1]["workload_fingerprint"] = "other"
        with self.assertRaisesRegex(RuntimeError, "paired workload"):
            analyze_results.require_complete(wrong_hash)

        wrong_input = formal_data()
        candidate = next(
            row
            for row in wrong_input["results"]
            if row["num_ssu"] == 40 and row["strategy"] == "current_refresh8"
        )
        candidate["request_metrics"][0]["nql"] = 99.0
        with self.assertRaisesRegex(RuntimeError, "input fields"):
            analyze_results.require_complete(wrong_input)

        duplicate_jitter = formal_data()
        for row in duplicate_jitter["results"]:
            if row["kind"] == "simulation":
                row["request_metrics"][1]["arrival_delay_ms"] = row[
                    "request_metrics"
                ][0]["arrival_delay_ms"]
        with self.assertRaisesRegex(RuntimeError, "unique"):
            analyze_results.require_complete(duplicate_jitter)

        bound_violation = formal_data()
        global_row = next(
            row
            for row in bound_violation["results"]
            if row["num_ssu"] == 40
            and row["strategy"] == "global_link_aware_online"
        )
        global_row["summary"]["avg_request_compute_fraction"] = 0.99
        for request in global_row["request_metrics"]:
            request["request_npu_utilization"] = 0.99
        with self.assertRaisesRegex(RuntimeError, "upper bound"):
            analyze_results.require_complete(bound_violation)

    def test_actual_and_capped_demand_drive_paired_classification(self):
        data = formal_data()
        index = analyze_results.row_index(data)
        pairs = analyze_results.paired_request_rows(
            index[(40, "baseline")], index[(40, "current_refresh8")]
        )
        cap_hit = next(
            row for row in pairs if row["actual_layer_demand_gbps"] == 60.0
        )
        self.assertEqual(cap_hit["capped_layer_demand_gbps"], 50.0)
        self.assertEqual(cap_hit["capped_layer_demand_bin"], "50-cap")

    def test_analysis_contains_paired_cadence_system_and_allocation_contracts(self):
        analysis = analyze_results.analyze(formal_data(), Path("synthetic.json"))
        self.assertEqual(
            analysis["selection"]["best_nonideal_practical_fleet"],
            "demand_maxmin",
        )
        cadence = analysis["pressure_cadence_paired"]["comparisons"]
        self.assertAlmostEqual(
            cadence["current_per_io"]["all_ssus"]["mean_compute_delta_pp"],
            4.0,
        )
        self.assertTrue(analysis["system_metrics"])
        alignment = analysis["allocation_alignment"]["rows"]
        for field in (
            "actual_demand_share",
            "capped_demand_share",
            "kv_bytes_share",
            "current_cir_share",
            "current_path_share",
            "best_cir_share",
            "best_path_share",
        ):
            self.assertAlmostEqual(sum(row[field] for row in alignment), 1.0)

        target = analysis["target_15pp"]
        self.assertEqual(
            target["request_selected"]["strategy"],
            analysis["selection"]["best_nonideal_practical_request"],
        )
        self.assertEqual(target["fleet_selected"]["strategy"], "demand_maxmin")
        report = analyze_results.build_report(analysis)
        self.assertIn("in-sample", report)
        self.assertIn("full current layer in one submission", report)
        self.assertIn(
            "best runnable <= unknown optimum <= loose fluid bound", report
        )
        causal_section = report.split(
            "## Why requests became faster or slower", 1
        )[1].split("### Current static CIR", 1)[0]
        for winner_field in (
            "best_nonideal_practical_request",
            "best_nonideal_practical_fleet",
        ):
            self.assertIn(
                "`%s`" % analysis["selection"][winner_field], causal_section
            )

    def test_all_eleven_figures_render_from_a_complete_analysis(self):
        analysis = analyze_results.analyze(formal_data(), Path("synthetic.json"))
        with tempfile.TemporaryDirectory() as directory:
            rendered = analyze_results.render_plots(analysis, Path(directory))
            self.assertEqual(len(rendered), 11)
            self.assertTrue(
                all(
                    (Path(directory) / filename).exists()
                    for filename in rendered.values()
                )
            )


if __name__ == "__main__":
    unittest.main()
