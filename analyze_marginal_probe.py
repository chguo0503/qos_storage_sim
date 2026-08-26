"""Strict paired analysis for the bounded marginal-utility CIR probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_joint_dynamic import validate_result as validate_joint_result
import joint_dynamic_experiment as joint_experiment
import marginal_utility_probe as probe
import sim


ANALYSIS_SCHEMA_VERSION = 1
FORMAL_SEEDS = (42, 43)
FORMAL_SSUS = (40, 56, 80)
FORMAL_NUM_NPU = 128
FORMAL_LAYERS = 16
PROBE_STRATEGIES = (
    "dynamic_marginal_fixed_path",
    "joint_marginal_path_cir",
)
REFERENCE_STRATEGIES = (
    "baseline",
    "current_static",
    "best_fixed_static",
    "joint_slack_path_cir",
)
REQUIRED_INVARIANTS = {
    "requests_completed",
    "blocks_conserved",
    "placement_preserved",
    "bytes_conserved",
    "npu_cap_respected",
    "single_backend_io",
    "single_npu_link_io",
    "queues_drained",
}
DISPLAY_STRATEGIES = REFERENCE_STRATEGIES + PROBE_STRATEGIES
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "current_static": "Current static",
    "best_fixed_static": "Best fixed",
    "joint_slack_path_cir": "Joint slack",
    "dynamic_marginal_fixed_path": "Marginal fixed",
    "joint_marginal_path_cir": "Marginal joint",
}
DEFAULT_PROBE_RESULT = Path("results/marginal_utility_probe/results.json")
DEFAULT_JOINT_SEED42 = Path("results/joint_dynamic_cir/results_seed42.json")
DEFAULT_JOINT_SEED43 = Path("results/joint_dynamic_cir/results_seed43.json")
DEFAULT_OUTPUT_DIR = Path("results/marginal_utility_probe")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expected_backend():
    return {
        "ssd": "one nonpreemptive command, size/40GBps",
        "npu": "one FCFS receive command per NPU, size/50GBps",
        "visible_after": "NPU receive completion",
    }


def _input_signature(request):
    return (
        request["request_id"],
        request["category"],
        request["npu_id"],
        request["seq_len_k"],
        request["nql"],
        request["per_layer_us"],
        request["per_layer_kv_gb"],
        request["required_bw_input_gbps"],
        request["arrival_delay_ms"],
        request["request_compute_ms"],
        request["io_count"],
    )


def _request_signatures(row):
    requests = row["request_metrics"]
    _require(
        len(requests) == FORMAL_NUM_NPU,
        "%s must contain 128 requests" % row["strategy"],
    )
    by_id = {request["request_id"]: request for request in requests}
    _require(
        len(by_id) == FORMAL_NUM_NPU
        and set(by_id) == set(range(FORMAL_NUM_NPU)),
        "%s request IDs must be unique 0..127" % row["strategy"],
    )
    return tuple(_input_signature(by_id[index]) for index in range(128))


def _finite_number(value):
    return not isinstance(value, bool) and math.isfinite(float(value))


def _validate_block_conservation(row):
    conservation = row["summary"]["block_conservation"]
    _require(
        conservation["expected"]
        == conservation["submitted"]
        == conservation["completed"],
        "%s block counts are not conserved" % row["strategy"],
    )
    _require(
        conservation["placement_targets_preserved"] is True,
        "%s placement targets were not preserved" % row["strategy"],
    )
    expected_gb = conservation["expected_read_gb"]
    _require(
        abs(conservation["ssd_completed_read_gb"] - expected_gb) <= 1e-9
        and abs(conservation["completed_read_gb"] - expected_gb) <= 1e-9,
        "%s read bytes are not conserved" % row["strategy"],
    )


def _validate_invariants(row):
    invariants = row["summary"]["invariants"]
    _require(
        set(invariants) == REQUIRED_INVARIANTS,
        "%s invariant keys do not match the formal contract" % row["strategy"],
    )
    _require(
        all(value is True for value in invariants.values()),
        "%s invariants failed" % row["strategy"],
    )


def _validate_metric_summary(row):
    requests = row["request_metrics"]
    summary = row["summary"]
    for request in requests:
        for key in (
            "request_npu_utilization",
            "request_compute_ms",
            "io_wait_total_ms",
            "io_wait_L0_ms",
            "io_wait_L1_ms",
            "io_wait_L2plus_ms",
        ):
            _require(
                _finite_number(request[key]),
                "%s request metric %s is not finite" % (row["strategy"], key),
            )
        _require(
            request["request_compute_ms"] >= 0.0
            and request["io_wait_total_ms"] >= 0.0
            and request["io_wait_L0_ms"] >= 0.0
            and request["io_wait_L1_ms"] >= 0.0
            and request["io_wait_L2plus_ms"] >= 0.0,
            "%s request time is negative" % row["strategy"],
        )
        decomposed_wait = (
            request["io_wait_L0_ms"]
            + request["io_wait_L1_ms"]
            + request["io_wait_L2plus_ms"]
        )
        _require(
            abs(decomposed_wait - request["io_wait_total_ms"]) <= 1e-9,
            "%s request IO wait decomposition mismatch" % row["strategy"],
        )
        request_total_ms = (
            request["request_compute_ms"] + request["io_wait_total_ms"]
        )
        _require(
            request_total_ms > 0.0,
            "%s request total time must be positive" % row["strategy"],
        )
        expected_utilization = request["request_compute_ms"] / request_total_ms
        _require(
            abs(expected_utilization - request["request_npu_utilization"])
            <= 1e-12,
            "%s per-request utilization mismatch" % row["strategy"],
        )
    request_mean = fmean(
        request["request_npu_utilization"] for request in requests
    )
    _require(
        abs(request_mean - summary["avg_request_compute_fraction"]) <= 1e-12,
        "%s request mean does not match summary" % row["strategy"],
    )
    total_compute_ms = sum(request["request_compute_ms"] for request in requests)
    _require(
        _finite_number(summary["avg_request_compute_fraction"])
        and 0.0 <= summary["avg_request_compute_fraction"] <= 1.0,
        "%s request summary metric is outside [0,1]" % row["strategy"],
    )
    _require(
        _finite_number(summary["fleet_npu_compute_utilization"])
        and 0.0 <= summary["fleet_npu_compute_utilization"] <= 1.0,
        "%s fleet summary metric is outside [0,1]" % row["strategy"],
    )
    _require(
        _finite_number(summary["makespan_ms"])
        and summary["makespan_ms"] > 0.0,
        "%s makespan is invalid" % row["strategy"],
    )
    expected_fleet = total_compute_ms / (
        FORMAL_NUM_NPU * summary["makespan_ms"]
    )
    _require(
        abs(expected_fleet - summary["fleet_npu_compute_utilization"])
        <= 1e-12,
        "%s fleet metric does not match requests/makespan" % row["strategy"],
    )


def _validate_probe_control(row, strategy):
    control = row["summary"]["dynamic_cir_control"]
    expected_config = strategy.config(probe.DEFAULT_MARGINAL_CONFIG)[
        "dynamic_cir"
    ]
    _require(control is not None, "%s has no marginal telemetry" % strategy.name)
    for key in (
        "mode",
        "formula_version",
        "decision_owner",
        "future_information",
        "formula",
        "tau_ms",
        "rmin_gbps",
        "npu_cap_gbps",
        "units",
        "disk_split",
        "ssd_capacity_rule",
        "configuration_latency_ms",
        "routing_mode",
        "paths_per_npu_per_ssu",
        "apply_boundary",
    ):
        _require(
            control[key] == expected_config[key],
            "%s marginal telemetry mismatch: %s" % (strategy.name, key),
        )
    _require(control["epochs"] > 0, "%s has no CIR epochs" % strategy.name)
    _require(
        control["desired_cir_evaluations"] > 0,
        "%s has no marginal evaluations" % strategy.name,
    )
    _require(
        probe.DEFAULT_MARGINAL_CONFIG.rmin_gbps - 1e-12
        <= control["min_desired_cir_gbps"]
        <= control["max_desired_cir_gbps"]
        <= sim.NPU_BW_LIMIT + 1e-12,
        "%s marginal desired CIR is outside [5,50]" % strategy.name,
    )
    _require(
        0.0 <= control["min_horizon_ms"] <= control["max_horizon_ms"],
        "%s marginal horizon range is invalid" % strategy.name,
    )
    _require(
        control["max_total_cir_gbps"] <= sim.DISK_BW + 1e-9,
        "%s exceeded the 40 GB/s SSD CIR cap" % strategy.name,
    )
    final_cirs = control["final_total_cir_gbps"]
    _require(
        len(final_cirs) == row["num_ssu"],
        "%s final CIR vector length mismatch" % strategy.name,
    )
    _require(
        all(0.0 <= value <= sim.DISK_BW + 1e-9 for value in final_cirs),
        "%s final CIR vector exceeds 40 GB/s" % strategy.name,
    )
    _require(
        max(final_cirs, default=0.0)
        <= control["max_total_cir_gbps"] + 1e-9,
        "%s final CIR is larger than recorded maximum" % strategy.name,
    )


def validate_probe_result(
    data,
    *,
    expected_code_fingerprint,
    expected_data_fingerprint,
):
    """Validate the exact 2-seed x 3-SSU x 2-route probe contract."""

    expected_strategies = probe.strategies()
    strategies_by_name = {
        strategy.name: strategy for strategy in expected_strategies
    }
    expected_configs = {
        strategy.name: strategy.config(probe.DEFAULT_MARGINAL_CONFIG)
        for strategy in expected_strategies
    }
    _require(
        data["schema_version"] == probe.SCHEMA_VERSION,
        "probe top-level schema version mismatch",
    )
    _require(data["complete"] is True, "marginal probe is incomplete")
    experiment = data["experiment"]
    _require(
        experiment["schema_version"] == probe.SCHEMA_VERSION,
        "probe experiment schema version mismatch",
    )
    _require(
        experiment["code_fingerprint"] == expected_code_fingerprint,
        "probe code fingerprint is stale",
    )
    _require(
        experiment["data_fingerprint"] == expected_data_fingerprint,
        "probe data fingerprint is stale",
    )
    _require(
        tuple(experiment["seeds"]) == FORMAL_SEEDS,
        "probe seeds must be exactly 42,43",
    )
    runtime = experiment["runtime"]
    _require(runtime["num_npu"] == FORMAL_NUM_NPU, "probe num_npu must be 128")
    _require(runtime["n_layers"] == FORMAL_LAYERS, "probe n_layers must be 16")
    _require(
        tuple(runtime["ssu_list"]) == FORMAL_SSUS,
        "probe SSUs must be exactly 40,56,80",
    )
    _require(runtime["ls_ratio"] == 0.5, "probe ls_ratio must be 0.5")
    _require(
        runtime["arrival_delay_ms"] == [0.0, sim.ARRIVAL_DELAY_MAX_MS],
        "probe arrival delay contract mismatch",
    )
    _require(runtime["disk_bw_gbps"] == 40.0, "probe SSD must be 40 GB/s")
    _require(runtime["npu_bw_limit_gbps"] == 50.0, "probe NPU must be 50 GB/s")
    _require(
        experiment["backend"] == _expected_backend(),
        "probe backend contract mismatch",
    )
    _require(
        experiment["marginal_config"] == asdict(probe.DEFAULT_MARGINAL_CONFIG),
        "probe marginal parameters are not tau=1ms/rmin=5GBps",
    )
    _require(
        tuple(data["selected_strategies"]) == PROBE_STRATEGIES,
        "probe selected strategy names/order mismatch",
    )
    _require(
        experiment["selected_strategies"]
        == [
            strategy.config(probe.DEFAULT_MARGINAL_CONFIG)
            for strategy in expected_strategies
        ],
        "probe strategy registry/config mismatch",
    )

    indexed = {}
    for row in data["results"]:
        key = (int(row["seed"]), int(row["num_ssu"]), row["strategy"])
        _require(key not in indexed, "duplicate probe row %s" % (key,))
        indexed[key] = row
    expected_keys = {
        (seed, ssu, strategy)
        for seed in FORMAL_SEEDS
        for ssu in FORMAL_SSUS
        for strategy in PROBE_STRATEGIES
    }
    _require(set(indexed) == expected_keys, "probe matrix is not exactly 2x3x2")

    pairing = {}
    for seed in FORMAL_SEEDS:
        seed_workloads = set()
        for ssu in FORMAL_SSUS:
            rows = [indexed[(seed, ssu, name)] for name in PROBE_STRATEGIES]
            workload_hashes = {row["workload_fingerprint"] for row in rows}
            placement_hashes = {row["placement_hash"] for row in rows}
            _require(
                len(workload_hashes) == 1,
                "probe seed=%d SSU=%d workload hashes are not paired" % (seed, ssu),
            )
            _require(
                len(placement_hashes) == 1,
                "probe seed=%d SSU=%d placement hashes are not paired" % (seed, ssu),
            )
            workload_hash = next(iter(workload_hashes))
            placement_hash = next(iter(placement_hashes))
            seed_workloads.add(workload_hash)
            pairing["%d/%d" % (seed, ssu)] = {
                "workload_fingerprint": workload_hash,
                "placement_hash": placement_hash,
            }
            signatures = None
            for row in rows:
                name = row["strategy"]
                strategy = strategies_by_name[name]
                _require(
                    row["config"] == expected_configs[name],
                    "%s row config mismatch" % name,
                )
                _require(
                    row["seeds"] == probe.runtime_for_seed(seed)["seeds"],
                    "%s seed bundle mismatch" % name,
                )
                summary = row["summary"]
                _require(
                    summary["workload_fingerprint"] == workload_hash,
                    "%s summary workload hash mismatch" % name,
                )
                _require(
                    summary["placement_hash"] == placement_hash,
                    "%s summary placement hash mismatch" % name,
                )
                _validate_invariants(row)
                _require(
                    summary["policy"] == sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
                    "%s policy mismatch" % name,
                )
                _require(
                    summary["backend_capacity_gbps"] == sim.DISK_BW
                    and summary["npu_bw_limit_gbps"] == sim.NPU_BW_LIMIT,
                    "%s capacity contract mismatch" % name,
                )
                current_signatures = _request_signatures(row)
                if signatures is None:
                    signatures = current_signatures
                _require(
                    current_signatures == signatures,
                    "probe seed=%d SSU=%d request inputs are not paired"
                    % (seed, ssu),
                )
                _validate_metric_summary(row)
                _validate_block_conservation(row)
                _validate_probe_control(row, strategy)
        _require(
            len(seed_workloads) == 1,
            "probe seed=%d workload changes across SSU counts" % seed,
        )
    return indexed, pairing


def validate_reference_results(
    seed_data,
    *,
    expected_data_fingerprint,
):
    _require(
        set(seed_data) == set(FORMAL_SEEDS),
        "joint references must contain exactly seed42 and seed43",
    )
    indexes = {}
    pairing = {}
    expected_strategies = {
        strategy.name: strategy for strategy in joint_experiment.strategies()
    }
    for seed in FORMAL_SEEDS:
        data = seed_data[seed]
        runtime = data["experiment"]["runtime"]
        _require(runtime["ls_ratio"] == 0.5, "joint seed%d ls_ratio mismatch" % seed)
        _require(
            runtime["arrival_delay_ms"] == [0.0, sim.ARRIVAL_DELAY_MAX_MS],
            "joint seed%d arrival delay mismatch" % seed,
        )
        _require(
            data["experiment"]["data_fingerprint"]
            == expected_data_fingerprint,
            "joint seed%d data fingerprint is stale" % seed,
        )
        indexes[seed], pairing[str(seed)] = validate_joint_result(data, seed)
        for (_, name), row in indexes[seed].items():
            expected_strategy = expected_strategies[name]
            _require(
                row["summary"]["policy"] == expected_strategy.policy,
                "joint seed%d %s summary policy mismatch" % (seed, name),
            )
            _require(
                row["summary"]["backend_capacity_gbps"] == sim.DISK_BW
                and row["summary"]["npu_bw_limit_gbps"]
                == sim.NPU_BW_LIMIT,
                "joint seed%d %s capacity contract mismatch" % (seed, name),
            )
            _validate_invariants(row)
            _request_signatures(row)
            _validate_metric_summary(row)
            _validate_block_conservation(row)
    return indexes, pairing


def validate_cross_pairing(probe_index, reference_indexes):
    pairing = {}
    for seed in FORMAL_SEEDS:
        for ssu in FORMAL_SSUS:
            reference = reference_indexes[seed][(ssu, "baseline")]
            reference_signatures = _request_signatures(reference)
            for strategy in PROBE_STRATEGIES:
                row = probe_index[(seed, ssu, strategy)]
                _require(
                    row["workload_fingerprint"]
                    == reference["workload_fingerprint"],
                    "seed=%d SSU=%d %s workload differs from references"
                    % (seed, ssu, strategy),
                )
                _require(
                    row["placement_hash"] == reference["placement_hash"],
                    "seed=%d SSU=%d %s placement differs from references"
                    % (seed, ssu, strategy),
                )
                _require(
                    _request_signatures(row) == reference_signatures,
                    "seed=%d SSU=%d %s request inputs differ from references"
                    % (seed, ssu, strategy),
                )
            pairing["%d/%d" % (seed, ssu)] = {
                "workload_fingerprint": reference["workload_fingerprint"],
                "placement_hash": reference["placement_hash"],
                "request_input_pairing": True,
            }
    return pairing


def _metrics(row):
    summary = row["summary"]
    return {
        "request_compute_pct": 100.0
        * summary["avg_request_compute_fraction"],
        "fleet_compute_pct": 100.0
        * summary["fleet_npu_compute_utilization"],
        "makespan_ms": summary["makespan_ms"],
    }


def _delta(left, right):
    return {
        "request_pp": right["request_compute_pct"]
        - left["request_compute_pct"],
        "fleet_pp": right["fleet_compute_pct"] - left["fleet_compute_pct"],
        "makespan_ms": right["makespan_ms"] - left["makespan_ms"],
    }


def analyze_documents(
    probe_data,
    joint_seed_data,
    *,
    expected_probe_code_fingerprint,
    expected_data_fingerprint,
):
    probe_index, probe_pairing = validate_probe_result(
        probe_data,
        expected_code_fingerprint=expected_probe_code_fingerprint,
        expected_data_fingerprint=expected_data_fingerprint,
    )
    reference_indexes, reference_pairing = validate_reference_results(
        joint_seed_data,
        expected_data_fingerprint=expected_data_fingerprint,
    )
    cross_pairing = validate_cross_pairing(probe_index, reference_indexes)

    scenarios = []
    for seed in FORMAL_SEEDS:
        for ssu in FORMAL_SSUS:
            rows = {
                name: (
                    probe_index[(seed, ssu, name)]
                    if name in PROBE_STRATEGIES
                    else reference_indexes[seed][(ssu, name)]
                )
                for name in DISPLAY_STRATEGIES
            }
            metrics = {name: _metrics(row) for name, row in rows.items()}
            comparisons = {
                marginal: {
                    reference: _delta(
                        metrics[reference],
                        metrics[marginal],
                    )
                    for reference in REFERENCE_STRATEGIES
                }
                for marginal in PROBE_STRATEGIES
            }
            scenarios.append(
                {
                    "seed": seed,
                    "num_ssu": ssu,
                    "metrics": metrics,
                    "marginal_vs_references": comparisons,
                    "joint_minus_fixed": _delta(
                        metrics["dynamic_marginal_fixed_path"],
                        metrics["joint_marginal_path_cir"],
                    ),
                }
            )

    aggregate_by_ssu = []
    for ssu in FORMAL_SSUS:
        selected = [row for row in scenarios if row["num_ssu"] == ssu]
        aggregate_by_ssu.append(
            {
                "num_ssu": ssu,
                "seed_count": len(selected),
                "metrics": {
                    name: {
                        metric: fmean(
                            row["metrics"][name][metric] for row in selected
                        )
                        for metric in (
                            "request_compute_pct",
                            "fleet_compute_pct",
                            "makespan_ms",
                        )
                    }
                    for name in DISPLAY_STRATEGIES
                },
                "joint_minus_fixed": {
                    metric: fmean(
                        row["joint_minus_fixed"][metric] for row in selected
                    )
                    for metric in ("request_pp", "fleet_pp", "makespan_ms")
                },
            }
        )

    def positive_count(strategy, reference, metric):
        return sum(
            row["marginal_vs_references"][strategy][reference][metric] > 0.0
            for row in scenarios
        )

    conclusions = {
        "scenario_count": len(scenarios),
        "fixed_request_beats_baseline_count": positive_count(
            "dynamic_marginal_fixed_path", "baseline", "request_pp"
        ),
        "joint_request_beats_baseline_count": positive_count(
            "joint_marginal_path_cir", "baseline", "request_pp"
        ),
        "fixed_request_beats_best_fixed_count": positive_count(
            "dynamic_marginal_fixed_path", "best_fixed_static", "request_pp"
        ),
        "joint_request_beats_best_fixed_count": positive_count(
            "joint_marginal_path_cir", "best_fixed_static", "request_pp"
        ),
        "joint_beats_fixed_request_count": sum(
            row["joint_minus_fixed"]["request_pp"] > 0.0
            for row in scenarios
        ),
        "joint_beats_fixed_fleet_count": sum(
            row["joint_minus_fixed"]["fleet_pp"] > 0.0
            for row in scenarios
        ),
    }
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "validated": True,
        "contract": {
            "seeds": list(FORMAL_SEEDS),
            "num_npu": FORMAL_NUM_NPU,
            "n_layers": FORMAL_LAYERS,
            "ssu_list": list(FORMAL_SSUS),
            "probe_strategies": list(PROBE_STRATEGIES),
            "reference_strategies": list(REFERENCE_STRATEGIES),
            "probe_code_fingerprint": expected_probe_code_fingerprint,
            "data_fingerprint": expected_data_fingerprint,
        },
        "pairing": {
            "probe": probe_pairing,
            "references": reference_pairing,
            "cross_source": cross_pairing,
        },
        "scenarios": scenarios,
        "aggregate_by_ssu": aggregate_by_ssu,
        "conclusions": conclusions,
    }


def _format_pair(metrics):
    return "%.3f / %.3f" % (
        metrics["request_compute_pct"],
        metrics["fleet_compute_pct"],
    )


def render_report(analysis):
    lines = [
        "# Marginal utility CIR 严格配对分析",
        "",
        "已严格验证 128 NPU、16 层、40/56/80 SSU、seed42/43、两种 Path 路由的完整矩阵。",
        "Probe 与 joint reference 的 workload、placement、128 个 request 输入完全配对；所有 invariants 通过，且每盘最大 CIR 不超过 40 GB/s。",
        "",
        "下表每格为 `request 利用率 % / fleet 利用率 %`。request 是逐请求平均，fleet 由总计算量与 makespan 计算，两者不可混用。",
        "",
        "| Seed | SSU | Baseline | Current | Best fixed | Joint slack | Marginal fixed | Marginal joint |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["scenarios"]:
        metrics = row["metrics"]
        lines.append(
            "| %d | %d | %s | %s | %s | %s | %s | %s |"
            % (
                row["seed"],
                row["num_ssu"],
                _format_pair(metrics["baseline"]),
                _format_pair(metrics["current_static"]),
                _format_pair(metrics["best_fixed_static"]),
                _format_pair(metrics["joint_slack_path_cir"]),
                _format_pair(metrics["dynamic_marginal_fixed_path"]),
                _format_pair(metrics["joint_marginal_path_cir"]),
            )
        )

    lines.extend(
        [
            "",
            "## 相对 baseline 与 best-fixed 的差值",
            "",
            "差值单位为百分点（pp），顺序为 `Δrequest / Δfleet`。",
            "",
            "| Seed | SSU | Fixed vs baseline | Joint vs baseline | Fixed vs best | Joint vs best |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["scenarios"]:
        comparison = row["marginal_vs_references"]

        def value(strategy, reference):
            delta = comparison[strategy][reference]
            return "%+.3f / %+.3f" % (
                delta["request_pp"],
                delta["fleet_pp"],
            )

        lines.append(
            "| %d | %d | %s | %s | %s | %s |"
            % (
                row["seed"],
                row["num_ssu"],
                value("dynamic_marginal_fixed_path", "baseline"),
                value("joint_marginal_path_cir", "baseline"),
                value("dynamic_marginal_fixed_path", "best_fixed_static"),
                value("joint_marginal_path_cir", "best_fixed_static"),
            )
        )

    lines.extend(
        [
            "",
            "## 联合选路消融",
            "",
            "`joint - fixed`；makespan 为负表示联合选路缩短总完成时间。",
            "",
            "| Seed | SSU | Δrequest pp | Δfleet pp | Δmakespan ms |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["scenarios"]:
        delta = row["joint_minus_fixed"]
        lines.append(
            "| %d | %d | %+.6f | %+.6f | %+.6f |"
            % (
                row["seed"],
                row["num_ssu"],
                delta["request_pp"],
                delta["fleet_pp"],
                delta["makespan_ms"],
            )
        )

    conclusion = analysis["conclusions"]
    count = conclusion["scenario_count"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- Marginal fixed 在 %d/%d 个场景的 request 指标高于 baseline，在 %d/%d 个场景高于 best-fixed。"
            % (
                conclusion["fixed_request_beats_baseline_count"],
                count,
                conclusion["fixed_request_beats_best_fixed_count"],
                count,
            ),
            "- Marginal joint 在 %d/%d 个场景的 request 指标高于 baseline，在 %d/%d 个场景高于 best-fixed。"
            % (
                conclusion["joint_request_beats_baseline_count"],
                count,
                conclusion["joint_request_beats_best_fixed_count"],
                count,
            ),
            "- 联合 Path 选路在 %d/%d 个场景提高 request 指标，在 %d/%d 个场景提高 fleet 指标。"
            % (
                conclusion["joint_beats_fixed_request_count"],
                count,
                conclusion["joint_beats_fixed_fleet_count"],
                count,
            ),
            "- 这是零配置延迟、命令边界重配置的在线 heuristic 对比，不应称为理论最优。",
            "",
        ]
    )
    return "\n".join(lines)


def render_plot(analysis, output_path):
    scenarios = analysis["scenarios"]
    labels = ["s%d/%d" % (row["seed"], row["num_ssu"]) for row in scenarios]
    positions = list(range(len(scenarios)))
    width = 0.36
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
    comparisons = (
        ("baseline", "vs baseline"),
        ("best_fixed_static", "vs best fixed"),
    )
    metric_rows = (
        ("request_pp", "Request utilization delta (pp)"),
        ("fleet_pp", "Fleet utilization delta (pp)"),
    )
    for row_index, (metric, ylabel) in enumerate(metric_rows):
        for column_index, (reference, title) in enumerate(comparisons):
            axis = axes[row_index][column_index]
            fixed = [
                row["marginal_vs_references"][
                    "dynamic_marginal_fixed_path"
                ][reference][metric]
                for row in scenarios
            ]
            joint = [
                row["marginal_vs_references"]["joint_marginal_path_cir"][
                    reference
                ][metric]
                for row in scenarios
            ]
            axis.bar(
                [position - width / 2.0 for position in positions],
                fixed,
                width,
                label="Marginal fixed",
                color="#4C78A8",
            )
            axis.bar(
                [position + width / 2.0 for position in positions],
                joint,
                width,
                label="Marginal joint",
                color="#F58518",
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        ablation_axis = axes[row_index][2]
        ablation_axis.bar(
            positions,
            [row["joint_minus_fixed"][metric] for row in scenarios],
            0.62,
            color="#54A24B",
            label="Joint - fixed",
        )
        ablation_axis.axhline(0.0, color="black", linewidth=0.8)
        ablation_axis.grid(axis="y", alpha=0.25)
        ablation_axis.set_title("joint Path - fixed Path")
        ablation_axis.set_ylabel(ylabel)
    for axis in axes[1]:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=35, ha="right")
    axes[0][0].legend(loc="best")
    axes[0][2].legend(loc="best")
    figure.suptitle(
        "Bounded marginal-CIR: paired seed42/43 comparison",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_outputs(analysis, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis.json"
    report_path = output_dir / "report.md"
    plot_path = output_dir / "analysis.png"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    report_path.write_text(render_report(analysis), encoding="utf-8")
    render_plot(analysis, plot_path)
    return json_path, report_path, plot_path


def run_analysis(
    *,
    probe_path=DEFAULT_PROBE_RESULT,
    joint_seed42_path=DEFAULT_JOINT_SEED42,
    joint_seed43_path=DEFAULT_JOINT_SEED43,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    table = sim.load_bw_table_cache(num_npu=FORMAL_NUM_NPU)
    current_probe_code = probe._code_fingerprint()
    current_probe_data = probe._data_fingerprint(table)
    current_joint_data = joint_experiment.data_fingerprint(table)
    _require(
        current_probe_data == current_joint_data,
        "probe and joint runners use different current data fingerprints",
    )
    analysis = analyze_documents(
        load_json(probe_path),
        {
            42: load_json(joint_seed42_path),
            43: load_json(joint_seed43_path),
        },
        expected_probe_code_fingerprint=current_probe_code,
        expected_data_fingerprint=current_probe_data,
    )
    analysis["sources"] = {
        "probe": str(probe_path),
        "joint_seed42": str(joint_seed42_path),
        "joint_seed43": str(joint_seed43_path),
    }
    paths = write_outputs(analysis, output_dir)
    return analysis, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE_RESULT)
    parser.add_argument(
        "--joint-seed42", type=Path, default=DEFAULT_JOINT_SEED42
    )
    parser.add_argument(
        "--joint-seed43", type=Path, default=DEFAULT_JOINT_SEED43
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analysis, paths = run_analysis(
        probe_path=args.probe,
        joint_seed42_path=args.joint_seed42,
        joint_seed43_path=args.joint_seed43,
        output_dir=args.output_dir,
    )
    print("validated scenarios:", len(analysis["scenarios"]))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
