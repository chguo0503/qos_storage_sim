import copy
import json
from pathlib import Path
from statistics import fmean
import tempfile
import unittest

from analyze_joint_dynamic import (
    DYNAMIC_STRATEGIES,
    FORMAL_SSUS,
    STRATEGY_ORDER,
    analyze_results,
    run_analysis,
    validate_result,
)
from joint_dynamic_experiment import (
    SCHEMA_VERSION,
    code_fingerprint,
    seed_bundle,
    strategies,
)


REQUEST_OFFSETS = {
    "baseline": 0.00,
    "current_static": 0.02,
    "best_fixed_static": 0.04,
    "ticket_static": 0.03,
    "demand_maxmin": 0.05,
    "dynamic_demand_fixed_path": 0.06,
    "joint_demand_path_cir": 0.07,
    "dynamic_slack_fixed_path": 0.08,
    "joint_slack_path_cir": 0.10,
}
FLEET_OFFSETS = {
    "baseline": 0.000,
    "current_static": -0.001,
    "best_fixed_static": 0.002,
    "ticket_static": 0.001,
    "demand_maxmin": 0.003,
    "dynamic_demand_fixed_path": 0.004,
    "joint_demand_path_cir": 0.005,
    "dynamic_slack_fixed_path": 0.006,
    "joint_slack_path_cir": 0.008,
}
CATEGORY_ORDER = ("SS", "SL", "LS", "LL")


def request_rows(seed, strategy, scale):
    rows = []
    for request_id in range(128):
        category_index = request_id % 4
        category = CATEGORY_ORDER[category_index]
        baseline = 0.45 + 0.03 * category_index + 0.001 * (request_id % 5)
        utilization = baseline + scale * REQUEST_OFFSETS[strategy]
        request_compute_ms = 32.0
        io_wait_total_ms = request_compute_ms * (1.0 - utilization) / utilization
        rows.append(
            {
                "request_id": request_id,
                "category": category,
                "npu_id": request_id,
                "seq_len_k": 20 + category_index * 100,
                "nql": 64 + category_index * 256,
                "per_layer_us": 2_000.0 + category_index * 4_000.0,
                "per_layer_kv_gb": 0.05 + category_index * 0.02,
                "required_bw_input_gbps": 5.0 + category_index * 12.0,
                "arrival_delay_ms": (request_id % 50) / 10.0,
                "queueing_delay_ms": 0.0,
                "processing_ttft_ms": 100.0,
                "ttft_ms": 100.0 + (request_id % 50) / 10.0,
                "io_wait_L0_ms": io_wait_total_ms * 0.5,
                "io_wait_L1_ms": io_wait_total_ms * 0.2,
                "io_wait_L2plus_ms": io_wait_total_ms * 0.3,
                "io_wait_total_ms": io_wait_total_ms,
                "avg_ssd_queue_wait_ms": (1.0 - utilization) * 2.0,
                "max_ssd_queue_wait_ms": 4.0,
                "avg_npu_link_queue_wait_ms": (1.0 - utilization),
                "max_npu_link_queue_wait_ms": 2.0,
                "avg_end_to_end_io_latency_ms": 3.0,
                "io_count": 64,
                "request_compute_ms": request_compute_ms,
                "request_npu_utilization": utilization,
            }
        )
    return rows


def dynamic_control(strategy, num_ssu):
    spec = next(item for item in strategies() if item.name == strategy)
    if spec.dynamic_mode is None:
        return None
    return {
        "mode": spec.dynamic_mode,
        "paths_per_npu": 2,
        "routing_mode": spec.routing_mode,
        "epochs": 1_000 + num_ssu,
        "final_total_cir_gbps": [39.0] * num_ssu,
        "max_total_cir_gbps": 40.0,
    }


def make_row(seed, num_ssu, strategy, scale):
    spec = next(item for item in strategies() if item.name == strategy)
    workload_hash = "workload-%d" % seed
    placement_hash = "placement-%d-%d" % (seed, num_ssu)
    requests = request_rows(seed, strategy, scale)
    request_mean = fmean(row["request_npu_utilization"] for row in requests)
    fleet = 0.14 + num_ssu / 100_000.0 + scale * FLEET_OFFSETS[strategy]
    makespan_ms = sum(row["request_compute_ms"] for row in requests) / (
        128 * fleet
    )
    block_conservation = {
        "expected": 100_000,
        "submitted": 100_000,
        "completed": 100_000,
        "placement_targets_preserved": True,
        "expected_read_gb": 100.0,
        "ssd_completed_read_gb": 100.0,
        "completed_read_gb": 100.0,
    }
    return {
        "num_ssu": num_ssu,
        "strategy": strategy,
        "config": spec.config(),
        "seeds": seed_bundle(seed),
        "workload_fingerprint": workload_hash,
        "placement_hash": placement_hash,
        "summary": {
            "policy": spec.policy,
            "backend_model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
            "data_plane_stages": {
                "ssd": {
                    "discipline": "policy_select_then_one_nonpreemptive_command",
                    "max_active_io": 1,
                    "service_time": "io_size_gb / disk_bw_gbps",
                },
                "npu_link": {
                    "discipline": "fcfs_store_and_forward",
                    "max_active_io": 1,
                    "service_time": "io_size_gb / npu_bw_limit_gbps",
                },
                "intermediate_buffer": "unbounded_store_and_forward",
                "block_visible_after": "npu_link_completion",
                "path_pressure_released_after": "ssd_completion",
            },
            "backend_capacity_gbps": 40.0,
            "npu_bw_limit_gbps": 50.0,
            "avg_request_compute_fraction": request_mean,
            "fleet_npu_compute_utilization": fleet,
            "makespan_ms": makespan_ms,
            "workload_fingerprint": workload_hash,
            "placement_hash": placement_hash,
            "invariants": {
                "requests_completed": True,
                "blocks_conserved": True,
                "placement_preserved": True,
                "bytes_conserved": True,
                "npu_cap_respected": True,
                "single_backend_io": True,
                "single_npu_link_io": True,
                "queues_drained": True,
            },
            "block_conservation": block_conservation,
            "blocks_enqueued": 100_000,
            "dynamic_cir_control": dynamic_control(strategy, num_ssu),
        },
        "request_metrics": requests,
        "wall_time_s": 1.0,
    }


def make_result(seed):
    scale = 1.0 if seed == 42 else 0.8
    registry = strategies()
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "experiment": {
            "schema_version": SCHEMA_VERSION,
            "code_fingerprint": code_fingerprint(),
            "data_fingerprint": "synthetic-data",
            "runtime": {
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": [40, 56, 80],
                "ls_ratio": 0.5,
                "seeds": seed_bundle(seed),
                "arrival_delay_ms": [0.0, 5.0],
                "disk_bw_gbps": 40.0,
                "npu_bw_limit_gbps": 50.0,
            },
            "backend": {
                "ssd": "one nonpreemptive command, size/40GBps",
                "npu": "one FCFS receive command per NPU, size/50GBps",
                "visible_after": "NPU receive completion",
            },
            "selected_strategies": [item.config() for item in registry],
        },
        "selected_strategies": [item.name for item in registry],
        "results": [
            make_row(seed, num_ssu, strategy, scale)
            for num_ssu in FORMAL_SSUS
            for strategy in STRATEGY_ORDER
        ],
    }


class JointDynamicAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.seed42 = make_result(42)
        self.seed43 = make_result(43)

    def test_analysis_covers_routing_dynamic_best_categories_and_inputs(self):
        analysis = analyze_results({42: self.seed42})
        self.assertEqual(
            analysis["winners"]["42"]["overall_request"],
            "joint_slack_path_cir",
        )
        self.assertEqual(
            analysis["winners"]["42"]["best_dynamic_request"],
            "joint_slack_path_cir",
        )
        self.assertEqual(len(analysis["strategy_comparisons"]), 27)
        self.assertEqual(len(analysis["category_analysis"]), 108)
        self.assertEqual(len(analysis["request_fastest_slowest_inputs"]), 27)
        self.assertEqual(len(analysis["route_ablations"]), 6)
        self.assertEqual(len(analysis["dynamic_vs_best_fixed"]), 12)
        self.assertEqual(len(analysis["control_epochs"]), 12)
        demand_route = next(
            row
            for row in analysis["route_ablations"]
            if row["num_ssu"] == 40 and row["mode"] == "demand_proportional"
        )
        self.assertAlmostEqual(demand_route["request_route_gain_pp"], 1.0)
        slack_route = next(
            row
            for row in analysis["route_ablations"]
            if row["num_ssu"] == 40 and row["mode"] == "slack_link_guarded"
        )
        self.assertAlmostEqual(slack_route["request_route_gain_pp"], 2.0)
        versus_best = next(
            row
            for row in analysis["dynamic_vs_best_fixed"]
            if row["num_ssu"] == 40 and row["strategy"] == "joint_slack_path_cir"
        )
        self.assertAlmostEqual(versus_best["request_gain_vs_best_fixed_pp"], 6.0)
        self.assertGreater(versus_best["improved_request_count"], 0)
        self.assertEqual(analysis["cross_seed"], None)

    def test_strict_contract_pairing_invariants_and_cir_are_rejected(self):
        cases = []

        incomplete = copy.deepcopy(self.seed42)
        incomplete["complete"] = False
        cases.append((incomplete, "not complete"))

        wrong_contract = copy.deepcopy(self.seed42)
        wrong_contract["experiment"]["runtime"]["n_layers"] = 15
        cases.append((wrong_contract, "n_layers"))

        unpaired = copy.deepcopy(self.seed42)
        target = next(
            row
            for row in unpaired["results"]
            if row["num_ssu"] == 56 and row["strategy"] == "joint_demand_path_cir"
        )
        target["placement_hash"] = "wrong"
        target["summary"]["placement_hash"] = "wrong"
        cases.append((unpaired, "placement hashes are not paired"))

        bad_invariant = copy.deepcopy(self.seed42)
        target = next(
            row
            for row in bad_invariant["results"]
            if row["num_ssu"] == 40 and row["strategy"] == "baseline"
        )
        target["summary"]["invariants"]["single_backend_io"] = False
        cases.append((bad_invariant, "invariants failed"))

        bad_cir = copy.deepcopy(self.seed42)
        target = next(
            row
            for row in bad_cir["results"]
            if row["num_ssu"] == 80 and row["strategy"] == "joint_slack_path_cir"
        )
        target["summary"]["dynamic_cir_control"]["max_total_cir_gbps"] = 40.01
        cases.append((bad_cir, "max CIR sum exceeds"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_result(data, 42)

    def test_backend_arrival_conservation_and_derived_metric_tampering_is_rejected(self):
        cases = []

        wrong_arrival = copy.deepcopy(self.seed42)
        wrong_arrival["experiment"]["runtime"]["arrival_delay_ms"] = [0.0, 0.0]
        cases.append((wrong_arrival, "arrival delay"))

        wrong_policy = copy.deepcopy(self.seed42)
        wrong_policy["results"][0]["summary"]["policy"] = "bogus"
        cases.append((wrong_policy, "summary policy"))

        wrong_backend = copy.deepcopy(self.seed42)
        wrong_backend["results"][0]["summary"]["backend_model"] = "bogus"
        cases.append((wrong_backend, "backend model"))

        wrong_stage = copy.deepcopy(self.seed42)
        wrong_stage["results"][0]["summary"]["data_plane_stages"]["ssd"][
            "max_active_io"
        ] = 2
        cases.append((wrong_stage, "data plane stages"))

        wrong_capacity = copy.deepcopy(self.seed42)
        wrong_capacity["results"][0]["summary"]["backend_capacity_gbps"] = 39.0
        cases.append((wrong_capacity, "capacity contract"))

        wrong_invariant_keys = copy.deepcopy(self.seed42)
        wrong_invariant_keys["results"][0]["summary"]["invariants"][
            "unexpected"
        ] = True
        cases.append((wrong_invariant_keys, "invariant keys"))

        wrong_blocks = copy.deepcopy(self.seed42)
        wrong_blocks["results"][0]["summary"]["block_conservation"][
            "completed"
        ] -= 1
        cases.append((wrong_blocks, "block counts"))

        wrong_bytes = copy.deepcopy(self.seed42)
        wrong_bytes["results"][0]["summary"]["block_conservation"][
            "completed_read_gb"
        ] += 1.0
        cases.append((wrong_bytes, "read bytes"))

        wrong_wait = copy.deepcopy(self.seed42)
        wrong_wait["results"][0]["request_metrics"][0]["io_wait_total_ms"] += 1.0
        cases.append((wrong_wait, "wait decomposition"))

        wrong_utilization = copy.deepcopy(self.seed42)
        wrong_utilization["results"][0]["request_metrics"][0][
            "request_npu_utilization"
        ] += 0.01
        cases.append((wrong_utilization, "per-request utilization"))

        wrong_fleet = copy.deepcopy(self.seed42)
        wrong_fleet["results"][0]["summary"][
            "fleet_npu_compute_utilization"
        ] += 0.01
        cases.append((wrong_fleet, "fleet metric"))

        wrong_makespan = copy.deepcopy(self.seed42)
        wrong_makespan["results"][0]["summary"]["makespan_ms"] = -1.0
        cases.append((wrong_makespan, "makespan"))

        for data, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_result(data, 42)

    def test_run_writes_two_figures_chinese_report_and_cross_seed_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed42_path = root / "seed42.json"
            seed43_path = root / "seed43.json"
            output_dir = root / "output"
            seed42_path.write_text(json.dumps(self.seed42))
            seed43_path.write_text(json.dumps(self.seed43))

            analysis = run_analysis(seed42_path, output_dir, seed43_path)

            self.assertIsNotNone(analysis["cross_seed"])
            self.assertTrue(analysis["cross_seed"]["rank_held"])
            self.assertTrue(analysis["cross_seed"]["direction_held_all_ssus"])
            self.assertTrue((output_dir / "analysis.json").is_file())
            self.assertGreater(
                (output_dir / "01_joint_dynamic_strategy_comparison.png").stat().st_size,
                1_000,
            )
            self.assertGreater(
                (output_dir / "02_category_path_cir_epochs.png").stat().st_size,
                1_000,
            )
            report = (output_dir / "report.md").read_text()
            self.assertIn("联合动态 Path + CIR 分析", report)
            self.assertIn("不能建立统计显著性", report)


if __name__ == "__main__":
    unittest.main()
