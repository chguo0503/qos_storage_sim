import copy
import json
import tempfile
import unittest
from pathlib import Path

import analyze_marginal_probe as analyzer
import joint_dynamic_experiment as joint_experiment
import marginal_utility_probe as probe
import sim


FIXTURE_CODE_FINGERPRINT = "fixture-probe-code"
FIXTURE_DATA_FINGERPRINT = "fixture-data"


def request_metrics(utilization):
    compute_ms = 10.0
    wait_ms = compute_ms / utilization - compute_ms
    categories = ("SS", "SL", "LS", "LL")
    return [
        {
            "request_id": request_id,
            "category": categories[request_id // 32],
            "npu_id": request_id,
            "seq_len_k": 32 + request_id,
            "nql": 64 + request_id,
            "per_layer_us": 1000.0 + request_id,
            "per_layer_kv_gb": 0.01 + request_id / 10000.0,
            "required_bw_input_gbps": 10.0 + request_id / 10.0,
            "arrival_delay_ms": (request_id % 5) / 10.0,
            "request_compute_ms": compute_ms,
            "io_count": 16,
            "request_npu_utilization": utilization,
            "io_wait_total_ms": wait_ms,
            "io_wait_L0_ms": wait_ms,
            "io_wait_L1_ms": 0.0,
            "io_wait_L2plus_ms": 0.0,
            "avg_ssd_queue_wait_ms": wait_ms / 16.0,
            "max_ssd_queue_wait_ms": wait_ms,
            "avg_npu_link_queue_wait_ms": 0.0,
            "max_npu_link_queue_wait_ms": 0.0,
            "avg_end_to_end_io_latency_ms": wait_ms / 16.0,
        }
        for request_id in range(128)
    ]


def block_conservation():
    return {
        "expected": 128,
        "submitted": 128,
        "completed": 128,
        "placement_targets_preserved": True,
        "expected_read_gb": 1.0,
        "ssd_completed_read_gb": 1.0,
        "completed_read_gb": 1.0,
    }


def summary(
    *,
    strategy,
    policy,
    request_fraction,
    fleet_fraction,
    workload_hash,
    placement_hash,
    dynamic_control,
):
    return {
        "policy": policy,
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
        "avg_request_compute_fraction": request_fraction,
        "fleet_npu_compute_utilization": fleet_fraction,
        "makespan_ms": 10.0 / fleet_fraction,
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
        "block_conservation": block_conservation(),
        "dynamic_cir_control": dynamic_control,
    }


def metric_values(name, seed, ssu):
    offset = (seed - 42) * 0.001 + (ssu - 40) * 0.0001
    base = {
        "baseline": (0.70, 0.1000),
        "current_static": (0.76, 0.1010),
        "best_fixed_static": (0.80, 0.1020),
        "joint_slack_path_cir": (0.79, 0.1015),
        "dynamic_marginal_fixed_path": (0.82, 0.1030),
        "joint_marginal_path_cir": (0.83, 0.1040),
    }.get(name, (0.72, 0.1005))
    return base[0] + offset, base[1] + offset / 10.0


def joint_control(strategy, ssu):
    if strategy.dynamic_mode is None:
        return None
    return {
        "mode": strategy.dynamic_mode,
        "paths_per_npu": 2,
        "routing_mode": strategy.routing_mode,
        "epochs": 10,
        "final_total_cir_gbps": [0.0] * ssu,
        "max_total_cir_gbps": 40.0,
    }


def make_joint_fixture(seed):
    strategies = joint_experiment.strategies()
    rows = []
    for ssu in analyzer.FORMAL_SSUS:
        workload_hash = "workload-%d" % seed
        placement_hash = "placement-%d-%d" % (seed, ssu)
        for strategy in strategies:
            request_fraction, fleet_fraction = metric_values(
                strategy.name,
                seed,
                ssu,
            )
            rows.append(
                {
                    "num_ssu": ssu,
                    "strategy": strategy.name,
                    "config": strategy.config(),
                    "seeds": joint_experiment.seed_bundle(seed),
                    "workload_fingerprint": workload_hash,
                    "placement_hash": placement_hash,
                    "summary": summary(
                        strategy=strategy.name,
                        policy=strategy.policy,
                        request_fraction=request_fraction,
                        fleet_fraction=fleet_fraction,
                        workload_hash=workload_hash,
                        placement_hash=placement_hash,
                        dynamic_control=joint_control(strategy, ssu),
                    ),
                    "request_metrics": request_metrics(request_fraction),
                    "wall_time_s": 1.0,
                }
            )
    return {
        "schema_version": joint_experiment.SCHEMA_VERSION,
        "complete": True,
        "experiment": {
            "schema_version": joint_experiment.SCHEMA_VERSION,
            "code_fingerprint": joint_experiment.code_fingerprint(),
            "data_fingerprint": FIXTURE_DATA_FINGERPRINT,
            "runtime": joint_experiment.runtime(seed),
            "backend": analyzer._expected_backend(),
            "selected_strategies": [strategy.config() for strategy in strategies],
        },
        "selected_strategies": [strategy.name for strategy in strategies],
        "results": rows,
    }


def marginal_control(strategy, ssu):
    control = probe.marginal_control_metadata(
        probe.DEFAULT_MARGINAL_CONFIG,
        strategy.routing_mode,
    )
    control.update(
        {
            "epochs": 10,
            "final_total_cir_gbps": [0.0] * ssu,
            "max_total_cir_gbps": 40.0,
            "desired_cir_evaluations": 100,
            "min_desired_cir_gbps": 5.1,
            "max_desired_cir_gbps": 49.0,
            "min_horizon_ms": 0.1,
            "max_horizon_ms": 100.0,
        }
    )
    return control


def make_probe_fixture():
    strategies = probe.strategies()
    rows = []
    for seed in analyzer.FORMAL_SEEDS:
        for ssu in analyzer.FORMAL_SSUS:
            workload_hash = "workload-%d" % seed
            placement_hash = "placement-%d-%d" % (seed, ssu)
            for strategy in strategies:
                request_fraction, fleet_fraction = metric_values(
                    strategy.name,
                    seed,
                    ssu,
                )
                rows.append(
                    {
                        "seed": seed,
                        "num_ssu": ssu,
                        "strategy": strategy.name,
                        "config": strategy.config(probe.DEFAULT_MARGINAL_CONFIG),
                        "seeds": probe.runtime_for_seed(seed)["seeds"],
                        "workload_fingerprint": workload_hash,
                        "placement_hash": placement_hash,
                        "summary": summary(
                            strategy=strategy.name,
                            policy=sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
                            request_fraction=request_fraction,
                            fleet_fraction=fleet_fraction,
                            workload_hash=workload_hash,
                            placement_hash=placement_hash,
                            dynamic_control=marginal_control(strategy, ssu),
                        ),
                        "request_metrics": request_metrics(request_fraction),
                        "wall_time_s": 1.0,
                    }
                )
    return {
        "schema_version": probe.SCHEMA_VERSION,
        "complete": True,
        "experiment": {
            "schema_version": probe.SCHEMA_VERSION,
            "code_fingerprint": FIXTURE_CODE_FINGERPRINT,
            "data_fingerprint": FIXTURE_DATA_FINGERPRINT,
            "seeds": list(analyzer.FORMAL_SEEDS),
            "runtime": {
                "num_npu": 128,
                "n_layers": 16,
                "ssu_list": [40, 56, 80],
                "ls_ratio": 0.5,
                "arrival_delay_ms": [0.0, sim.ARRIVAL_DELAY_MAX_MS],
                "disk_bw_gbps": 40.0,
                "npu_bw_limit_gbps": 50.0,
            },
            "backend": analyzer._expected_backend(),
            "marginal_config": {
                "tau_ms": 1.0,
                "rmin_gbps": 5.0,
                "npu_cap_gbps": 50.0,
            },
            "selected_strategies": [
                strategy.config(probe.DEFAULT_MARGINAL_CONFIG)
                for strategy in strategies
            ],
        },
        "selected_strategies": [strategy.name for strategy in strategies],
        "results": rows,
    }


def analyze_fixture(probe_data=None, joint_data=None):
    return analyzer.analyze_documents(
        make_probe_fixture() if probe_data is None else probe_data,
        (
            {seed: make_joint_fixture(seed) for seed in analyzer.FORMAL_SEEDS}
            if joint_data is None
            else joint_data
        ),
        expected_probe_code_fingerprint=FIXTURE_CODE_FINGERPRINT,
        expected_data_fingerprint=FIXTURE_DATA_FINGERPRINT,
    )


class AnalyzeMarginalProbeTests(unittest.TestCase):
    def test_complete_fixture_produces_exact_paired_comparisons_and_artifacts(self):
        analysis = analyze_fixture()

        self.assertTrue(analysis["validated"])
        self.assertEqual(len(analysis["scenarios"]), 6)
        first = analysis["scenarios"][0]
        self.assertAlmostEqual(
            first["marginal_vs_references"]["dynamic_marginal_fixed_path"][
                "baseline"
            ]["request_pp"],
            12.0,
        )
        self.assertAlmostEqual(
            first["marginal_vs_references"]["joint_marginal_path_cir"][
                "best_fixed_static"
            ]["request_pp"],
            3.0,
        )
        self.assertAlmostEqual(first["joint_minus_fixed"]["request_pp"], 1.0)
        self.assertAlmostEqual(first["joint_minus_fixed"]["fleet_pp"], 0.1)
        self.assertEqual(
            analysis["conclusions"]["joint_beats_fixed_request_count"],
            6,
        )

        with tempfile.TemporaryDirectory() as directory:
            json_path, report_path, plot_path = analyzer.write_outputs(
                analysis,
                Path(directory),
            )
            written = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(written, analysis)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("严格配对分析", report)
            self.assertIn("联合选路消融", report)
            self.assertGreater(plot_path.stat().st_size, 10_000)

    def test_stale_fingerprint_incomplete_matrix_and_bad_cap_are_rejected(self):
        stale = make_probe_fixture()
        stale["experiment"]["code_fingerprint"] = "stale"
        with self.assertRaisesRegex(ValueError, "code fingerprint"):
            analyze_fixture(stale)

        incomplete = make_probe_fixture()
        incomplete["results"].pop()
        with self.assertRaisesRegex(ValueError, "2x3x2"):
            analyze_fixture(incomplete)

        cap = make_probe_fixture()
        cap["results"][0]["summary"]["dynamic_cir_control"][
            "max_total_cir_gbps"
        ] = 40.1
        with self.assertRaisesRegex(ValueError, "40 GB/s"):
            analyze_fixture(cap)

    def test_invariant_and_cross_source_request_tampering_are_rejected(self):
        invariant = make_probe_fixture()
        invariant["results"][0]["summary"]["invariants"][
            "bytes_conserved"
        ] = False
        with self.assertRaisesRegex(ValueError, "invariants"):
            analyze_fixture(invariant)

        request_tamper = make_probe_fixture()
        for row in request_tamper["results"]:
            if row["seed"] == 42 and row["num_ssu"] == 40:
                row["request_metrics"][0]["nql"] += 1
        with self.assertRaisesRegex(ValueError, "references"):
            analyze_fixture(request_tamper)

    def test_reference_data_fingerprint_and_fleet_tampering_are_rejected(self):
        references = {
            seed: make_joint_fixture(seed) for seed in analyzer.FORMAL_SEEDS
        }
        references[43]["experiment"]["data_fingerprint"] = "wrong"
        with self.assertRaisesRegex(ValueError, "data fingerprint"):
            analyze_fixture(joint_data=references)

        fleet = make_probe_fixture()
        fleet["results"][0]["summary"][
            "fleet_npu_compute_utilization"
        ] += 0.01
        with self.assertRaisesRegex(ValueError, "fleet metric"):
            analyze_fixture(fleet)

        wrong_policy = {
            seed: make_joint_fixture(seed) for seed in analyzer.FORMAL_SEEDS
        }
        wrong_policy[42]["results"][0]["summary"]["policy"] = "wrong"
        with self.assertRaisesRegex(ValueError, "summary policy"):
            analyze_fixture(joint_data=wrong_policy)


if __name__ == "__main__":
    unittest.main()
