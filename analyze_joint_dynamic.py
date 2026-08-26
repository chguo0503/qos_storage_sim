"""Validate and analyze the joint dynamic Path-routing/CIR experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joint_dynamic_experiment import (
    SCHEMA_VERSION as INPUT_SCHEMA_VERSION,
    code_fingerprint,
    seed_bundle,
    strategies,
)


FORMAL_NUM_NPU = 128
FORMAL_LAYERS = 16
FORMAL_SSUS = (40, 56, 80)
DISK_CIR_CAP_GBPS = 40.0
NPU_LINK_CAP_GBPS = 50.0
CATEGORIES = ("SS", "SL", "LS", "LL")
DEFAULT_SEED42 = Path("results/joint_dynamic_cir/results_seed42.json")
DEFAULT_OUTPUT_DIR = Path("results/joint_dynamic_cir")

STRATEGY_ORDER = tuple(strategy.name for strategy in strategies())
DYNAMIC_STRATEGIES = (
    "dynamic_demand_fixed_path",
    "joint_demand_path_cir",
    "dynamic_slack_fixed_path",
    "joint_slack_path_cir",
)
ROUTE_ABLATIONS = (
    (
        "demand_proportional",
        "dynamic_demand_fixed_path",
        "joint_demand_path_cir",
    ),
    (
        "slack_link_guarded",
        "dynamic_slack_fixed_path",
        "joint_slack_path_cir",
    ),
)
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "current_static": "Current static",
    "best_fixed_static": "Best fixed static",
    "ticket_static": "Ticket static",
    "demand_maxmin": "Demand max-min",
    "dynamic_demand_fixed_path": "Dynamic demand / fixed Path",
    "joint_demand_path_cir": "Joint demand Path+CIR",
    "dynamic_slack_fixed_path": "Dynamic slack / fixed Path",
    "joint_slack_path_cir": "Joint slack Path+CIR",
}
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
EXPECTED_BACKEND_MODEL = "shared_two_stage_ssd40_then_npu50_single_server_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def _expected_backend():
    return {
        "ssd": "one nonpreemptive command, size/40GBps",
        "npu": "one FCFS receive command per NPU, size/50GBps",
        "visible_after": "NPU receive completion",
    }


def _expected_data_plane_stages():
    return {
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
    }


def _finite_number(value):
    return not isinstance(value, bool) and math.isfinite(float(value))


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
        _finite_number(expected_gb)
        and _finite_number(conservation["ssd_completed_read_gb"])
        and _finite_number(conservation["completed_read_gb"])
        and math.isclose(
            conservation["ssd_completed_read_gb"],
            expected_gb,
            rel_tol=1e-10,
            abs_tol=1e-9,
        )
        and math.isclose(
            conservation["completed_read_gb"],
            expected_gb,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ),
        "%s read bytes are not conserved" % row["strategy"],
    )


def _validate_metric_summary(row):
    requests = row["request_metrics"]
    summary = row["summary"]
    wait_keys = (
        "io_wait_total_ms",
        "io_wait_L0_ms",
        "io_wait_L1_ms",
        "io_wait_L2plus_ms",
        "avg_ssd_queue_wait_ms",
        "max_ssd_queue_wait_ms",
        "avg_npu_link_queue_wait_ms",
        "max_npu_link_queue_wait_ms",
        "avg_end_to_end_io_latency_ms",
    )
    for request in requests:
        _require(
            _finite_number(request["request_compute_ms"])
            and request["request_compute_ms"] >= 0.0,
            "%s request compute time is invalid" % row["strategy"],
        )
        for key in wait_keys:
            _require(
                _finite_number(request[key]) and request[key] >= 0.0,
                "%s request wait metric %s is invalid"
                % (row["strategy"], key),
            )
        decomposed_wait = (
            request["io_wait_L0_ms"]
            + request["io_wait_L1_ms"]
            + request["io_wait_L2plus_ms"]
        )
        _require(
            math.isclose(
                decomposed_wait,
                request["io_wait_total_ms"],
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "%s request IO wait decomposition mismatch" % row["strategy"],
        )
        request_total_ms = (
            request["request_compute_ms"] + request["io_wait_total_ms"]
        )
        _require(
            request_total_ms > 0.0
            and _finite_number(request["request_npu_utilization"]),
            "%s request utilization inputs are invalid" % row["strategy"],
        )
        expected_utilization = request["request_compute_ms"] / request_total_ms
        _require(
            math.isclose(
                expected_utilization,
                request["request_npu_utilization"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "%s per-request utilization mismatch" % row["strategy"],
        )

    request_mean = fmean(
        request["request_npu_utilization"] for request in requests
    )
    _require(
        _finite_number(summary["avg_request_compute_fraction"])
        and 0.0 <= summary["avg_request_compute_fraction"] <= 1.0
        and math.isclose(
            request_mean,
            summary["avg_request_compute_fraction"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "%s request mean does not match summary" % row["strategy"],
    )
    _require(
        _finite_number(summary["makespan_ms"])
        and summary["makespan_ms"] > 0.0,
        "%s makespan is invalid" % row["strategy"],
    )
    expected_fleet = sum(
        request["request_compute_ms"] for request in requests
    ) / (FORMAL_NUM_NPU * summary["makespan_ms"])
    _require(
        _finite_number(summary["fleet_npu_compute_utilization"])
        and 0.0 <= summary["fleet_npu_compute_utilization"] <= 1.0
        and math.isclose(
            expected_fleet,
            summary["fleet_npu_compute_utilization"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "%s fleet metric does not match requests/makespan" % row["strategy"],
    )


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
    )


def _validate_cir(row, expected_strategy):
    config = row["config"]
    if "category_cir_gbps" in config:
        _require(
            sum(config["category_cir_gbps"]) <= DISK_CIR_CAP_GBPS + 1e-9,
            "%s static CIR sum exceeds 40 GB/s" % row["strategy"],
        )
    control = row["summary"]["dynamic_cir_control"]
    if expected_strategy.dynamic_mode is None:
        _require(
            control is None,
            "%s unexpectedly contains dynamic CIR control" % row["strategy"],
        )
        return
    _require(control is not None, "%s is missing dynamic CIR control" % row["strategy"])
    _require(
        control["mode"] == expected_strategy.dynamic_mode,
        "%s dynamic CIR mode mismatch" % row["strategy"],
    )
    _require(
        control["routing_mode"] == expected_strategy.routing_mode,
        "%s dynamic routing mode mismatch" % row["strategy"],
    )
    _require(control["paths_per_npu"] == 2, "dynamic Paths per NPU must be two")
    _require(control["epochs"] > 0, "%s has no control epochs" % row["strategy"])
    _require(
        control["max_total_cir_gbps"] <= DISK_CIR_CAP_GBPS + 1e-9,
        "%s max CIR sum exceeds 40 GB/s" % row["strategy"],
    )
    final_cirs = control["final_total_cir_gbps"]
    _require(
        len(final_cirs) == row["num_ssu"],
        "%s final CIR vector length does not equal SSU count" % row["strategy"],
    )
    _require(
        all(value <= DISK_CIR_CAP_GBPS + 1e-9 for value in final_cirs),
        "%s final CIR sum exceeds 40 GB/s" % row["strategy"],
    )


def validate_result(data, expected_seed):
    """Strictly validate one complete formal seed and return indexed rows."""
    expected_strategies = strategies()
    expected_strategy_objects = {
        strategy.name: strategy for strategy in expected_strategies
    }
    expected_configs = {
        strategy.name: strategy.config() for strategy in expected_strategies
    }
    _require(
        data["schema_version"] == INPUT_SCHEMA_VERSION,
        "top-level schema version mismatch",
    )
    _require(data["complete"] is True, "joint dynamic result is not complete")
    experiment = data["experiment"]
    _require(
        experiment["schema_version"] == INPUT_SCHEMA_VERSION,
        "experiment schema version mismatch",
    )
    _require(
        experiment["code_fingerprint"] == code_fingerprint(),
        "joint dynamic code fingerprint is stale",
    )
    _require(bool(experiment["data_fingerprint"]), "data fingerprint is empty")
    runtime = experiment["runtime"]
    _require(runtime["num_npu"] == FORMAL_NUM_NPU, "num_npu must be 128")
    _require(runtime["n_layers"] == FORMAL_LAYERS, "n_layers must be 16")
    _require(tuple(runtime["ssu_list"]) == FORMAL_SSUS, "SSUs must be 40,56,80")
    _require(runtime["ls_ratio"] == 0.5, "ls_ratio must be 0.5")
    _require(runtime["seeds"] == seed_bundle(expected_seed), "seed bundle mismatch")
    _require(
        runtime["arrival_delay_ms"] == [0.0, 5.0],
        "arrival delay must be Uniform[0,5ms)",
    )
    _require(runtime["disk_bw_gbps"] == 40.0, "SSD bandwidth must be 40 GB/s")
    _require(runtime["npu_bw_limit_gbps"] == 50.0, "NPU bandwidth must be 50 GB/s")
    _require(
        experiment["backend"] == _expected_backend(),
        "shared SSD40->NPU50 backend contract mismatch",
    )
    _require(
        tuple(data["selected_strategies"]) == STRATEGY_ORDER,
        "selected strategy names/order mismatch",
    )
    _require(
        experiment["selected_strategies"]
        == [strategy.config() for strategy in expected_strategies],
        "experiment strategy registry/config mismatch",
    )

    indexed = {}
    for row in data["results"]:
        key = (int(row["num_ssu"]), row["strategy"])
        _require(key not in indexed, "duplicate result row %s" % (key,))
        indexed[key] = row
    expected_keys = {
        (ssu, strategy) for ssu in FORMAL_SSUS for strategy in STRATEGY_ORDER
    }
    _require(set(indexed) == expected_keys, "result matrix is not exactly 3x9")

    workload_hashes = set()
    pairing = {}
    for ssu in FORMAL_SSUS:
        rows = [indexed[(ssu, name)] for name in STRATEGY_ORDER]
        workloads = {row["workload_fingerprint"] for row in rows}
        placements = {row["placement_hash"] for row in rows}
        _require(len(workloads) == 1, "SSU=%d workload hashes are not paired" % ssu)
        _require(len(placements) == 1, "SSU=%d placement hashes are not paired" % ssu)
        workload_hash = next(iter(workloads))
        placement_hash = next(iter(placements))
        workload_hashes.add(workload_hash)
        pairing[str(ssu)] = {
            "workload_fingerprint": workload_hash,
            "placement_hash": placement_hash,
        }
        reference_signatures = None
        for row in rows:
            name = row["strategy"]
            _require(row["config"] == expected_configs[name], "%s row config mismatch" % name)
            _require(row["seeds"] == seed_bundle(expected_seed), "%s row seeds mismatch" % name)
            summary = row["summary"]
            expected_strategy = expected_strategy_objects[name]
            _require(
                summary["policy"] == expected_strategy.policy,
                "%s summary policy mismatch" % name,
            )
            _require(
                summary["backend_model"] == EXPECTED_BACKEND_MODEL,
                "%s backend model mismatch" % name,
            )
            _require(
                summary["data_plane_stages"] == _expected_data_plane_stages(),
                "%s data plane stages mismatch" % name,
            )
            _require(
                summary["backend_capacity_gbps"] == DISK_CIR_CAP_GBPS
                and summary["npu_bw_limit_gbps"] == NPU_LINK_CAP_GBPS,
                "%s capacity contract mismatch" % name,
            )
            _require(
                summary["workload_fingerprint"] == workload_hash,
                "%s summary workload hash mismatch" % name,
            )
            _require(
                summary["placement_hash"] == placement_hash,
                "%s summary placement hash mismatch" % name,
            )
            _validate_invariants(row)
            requests = row["request_metrics"]
            _require(len(requests) == FORMAL_NUM_NPU, "%s must have 128 requests" % name)
            by_id = {request["request_id"]: request for request in requests}
            _require(
                set(by_id) == set(range(FORMAL_NUM_NPU)),
                "%s request IDs must be 0..127" % name,
            )
            signatures = tuple(_input_signature(by_id[index]) for index in range(128))
            if reference_signatures is None:
                reference_signatures = signatures
            _require(signatures == reference_signatures, "SSU=%d request inputs are not paired" % ssu)
            _validate_metric_summary(row)
            _validate_block_conservation(row)
            _validate_cir(row, expected_strategy)
    _require(len(workload_hashes) == 1, "workload hash changes across SSU counts")
    return indexed, pairing


def _summary_metric(row, name):
    return float(row["summary"][name])


def _request_index(row):
    return {request["request_id"]: request for request in row["request_metrics"]}


def _input_view(request):
    return {
        "request_id": request["request_id"],
        "category": request["category"],
        "seq_len_k": request["seq_len_k"],
        "nql": request["nql"],
        "required_bw_input_gbps": request["required_bw_input_gbps"],
        "per_layer_compute_ms": request["per_layer_us"] / 1000.0,
        "per_layer_kv_gb": request["per_layer_kv_gb"],
        "request_compute_fraction": request["request_npu_utilization"],
        "io_wait_total_ms": request["io_wait_total_ms"],
        "avg_ssd_queue_wait_ms": request["avg_ssd_queue_wait_ms"],
        "avg_npu_link_queue_wait_ms": request["avg_npu_link_queue_wait_ms"],
    }


def _request_delta_extremes(left_row, right_row):
    left = _request_index(left_row)
    right = _request_index(right_row)
    deltas = []
    for request_id in range(FORMAL_NUM_NPU):
        delta_pp = 100.0 * (
            right[request_id]["request_npu_utilization"]
            - left[request_id]["request_npu_utilization"]
        )
        deltas.append((delta_pp, request_id))
    fastest = max(deltas, key=lambda value: (value[0], -value[1]))
    slowest = min(deltas, key=lambda value: (value[0], value[1]))
    return {
        "improved_request_count": sum(delta > 1e-12 for delta, _ in deltas),
        "regressed_request_count": sum(delta < -1e-12 for delta, _ in deltas),
        "largest_gain": {
            "delta_pp": fastest[0],
            "input": _input_view(right[fastest[1]]),
        },
        "largest_regression": {
            "delta_pp": slowest[0],
            "input": _input_view(right[slowest[1]]),
        },
    }


def _category_rows(row):
    grouped = defaultdict(list)
    for request in row["request_metrics"]:
        grouped[request["category"]].append(request)
    result = {}
    for category in CATEGORIES:
        requests = grouped[category]
        result[category] = {
            "count": len(requests),
            "avg_request_compute_fraction": fmean(
                request["request_npu_utilization"] for request in requests
            ),
            "avg_io_wait_total_ms": fmean(
                request["io_wait_total_ms"] for request in requests
            ),
            "avg_ssd_queue_wait_ms": fmean(
                request["avg_ssd_queue_wait_ms"] for request in requests
            ),
            "avg_npu_link_queue_wait_ms": fmean(
                request["avg_npu_link_queue_wait_ms"] for request in requests
            ),
        }
    return result


def _rank_scores(scores):
    return sorted(
        scores,
        key=lambda row: (
            -row["mean_request_compute_fraction"],
            -row["mean_fleet_npu_compute_utilization"],
            row["strategy"],
        ),
    )


def analyze_results(seed_data):
    """Validate one or two seeds and build all paired analyses."""
    indexes = {}
    pairing = {}
    for seed in sorted(seed_data):
        indexes[seed], pairing[str(seed)] = validate_result(seed_data[seed], seed)

    comparisons = []
    strategy_scores = {}
    request_extremes = []
    category_analysis = []
    route_ablations = []
    dynamic_vs_best = []
    control_epochs = []
    winners = {}

    for seed, index in indexes.items():
        for ssu in FORMAL_SSUS:
            baseline = index[(ssu, "baseline")]
            best_fixed = index[(ssu, "best_fixed_static")]
            baseline_request = _summary_metric(baseline, "avg_request_compute_fraction")
            baseline_fleet = _summary_metric(baseline, "fleet_npu_compute_utilization")
            best_request = _summary_metric(best_fixed, "avg_request_compute_fraction")
            best_fleet = _summary_metric(best_fixed, "fleet_npu_compute_utilization")
            baseline_categories = _category_rows(baseline)
            best_categories = _category_rows(best_fixed)
            for strategy in STRATEGY_ORDER:
                row = index[(ssu, strategy)]
                request_value = _summary_metric(row, "avg_request_compute_fraction")
                fleet_value = _summary_metric(row, "fleet_npu_compute_utilization")
                comparisons.append(
                    {
                        "seed": seed,
                        "num_ssu": ssu,
                        "strategy": strategy,
                        "request_compute_fraction": request_value,
                        "fleet_npu_compute_utilization": fleet_value,
                        "request_gain_vs_baseline_pp": 100.0
                        * (request_value - baseline_request),
                        "fleet_gain_vs_baseline_pp": 100.0
                        * (fleet_value - baseline_fleet),
                        "request_gain_vs_best_fixed_pp": 100.0
                        * (request_value - best_request),
                        "fleet_gain_vs_best_fixed_pp": 100.0
                        * (fleet_value - best_fleet),
                    }
                )
                requests = row["request_metrics"]
                fastest = max(
                    requests,
                    key=lambda request: (
                        request["request_npu_utilization"],
                        -request["request_id"],
                    ),
                )
                slowest = min(
                    requests,
                    key=lambda request: (
                        request["request_npu_utilization"],
                        request["request_id"],
                    ),
                )
                request_extremes.append(
                    {
                        "seed": seed,
                        "num_ssu": ssu,
                        "strategy": strategy,
                        "fastest_input": _input_view(fastest),
                        "slowest_input": _input_view(slowest),
                    }
                )
                categories = _category_rows(row)
                for category in CATEGORIES:
                    item = dict(categories[category])
                    item.update(
                        {
                            "seed": seed,
                            "num_ssu": ssu,
                            "strategy": strategy,
                            "category": category,
                            "gain_vs_baseline_pp": 100.0
                            * (
                                item["avg_request_compute_fraction"]
                                - baseline_categories[category][
                                    "avg_request_compute_fraction"
                                ]
                            ),
                            "gain_vs_best_fixed_pp": 100.0
                            * (
                                item["avg_request_compute_fraction"]
                                - best_categories[category][
                                    "avg_request_compute_fraction"
                                ]
                            ),
                        }
                    )
                    category_analysis.append(item)

                if strategy in DYNAMIC_STRATEGIES:
                    control = row["summary"]["dynamic_cir_control"]
                    final_cirs = control["final_total_cir_gbps"]
                    blocks = row["summary"]["blocks_enqueued"]
                    control_epochs.append(
                        {
                            "seed": seed,
                            "num_ssu": ssu,
                            "strategy": strategy,
                            "mode": control["mode"],
                            "routing_mode": control["routing_mode"],
                            "epochs": control["epochs"],
                            "epochs_per_million_blocks": control["epochs"]
                            / blocks
                            * 1_000_000.0,
                            "max_total_cir_gbps": control["max_total_cir_gbps"],
                            "final_total_cir_min_gbps": min(final_cirs),
                            "final_total_cir_mean_gbps": fmean(final_cirs),
                            "final_total_cir_max_gbps": max(final_cirs),
                        }
                    )

            for mode, fixed_name, joint_name in ROUTE_ABLATIONS:
                fixed = index[(ssu, fixed_name)]
                joint = index[(ssu, joint_name)]
                route_item = {
                    "seed": seed,
                    "num_ssu": ssu,
                    "mode": mode,
                    "fixed_strategy": fixed_name,
                    "joint_strategy": joint_name,
                    "request_route_gain_pp": 100.0
                    * (
                        _summary_metric(joint, "avg_request_compute_fraction")
                        - _summary_metric(fixed, "avg_request_compute_fraction")
                    ),
                    "fleet_route_gain_pp": 100.0
                    * (
                        _summary_metric(joint, "fleet_npu_compute_utilization")
                        - _summary_metric(fixed, "fleet_npu_compute_utilization")
                    ),
                }
                route_item.update(_request_delta_extremes(fixed, joint))
                route_ablations.append(route_item)

            for dynamic_name in DYNAMIC_STRATEGIES:
                dynamic = index[(ssu, dynamic_name)]
                item = {
                    "seed": seed,
                    "num_ssu": ssu,
                    "strategy": dynamic_name,
                    "request_gain_vs_best_fixed_pp": 100.0
                    * (
                        _summary_metric(dynamic, "avg_request_compute_fraction")
                        - best_request
                    ),
                    "fleet_gain_vs_best_fixed_pp": 100.0
                    * (
                        _summary_metric(dynamic, "fleet_npu_compute_utilization")
                        - best_fleet
                    ),
                }
                item.update(_request_delta_extremes(best_fixed, dynamic))
                dynamic_vs_best.append(item)

        seed_scores = []
        comparison_lookup = {
            (row["num_ssu"], row["strategy"]): row
            for row in comparisons
            if row["seed"] == seed
        }
        for strategy in STRATEGY_ORDER:
            points = [comparison_lookup[(ssu, strategy)] for ssu in FORMAL_SSUS]
            seed_scores.append(
                {
                    "strategy": strategy,
                    "mean_request_compute_fraction": fmean(
                        point["request_compute_fraction"] for point in points
                    ),
                    "mean_fleet_npu_compute_utilization": fmean(
                        point["fleet_npu_compute_utilization"] for point in points
                    ),
                    "mean_request_gain_vs_baseline_pp": fmean(
                        point["request_gain_vs_baseline_pp"] for point in points
                    ),
                    "mean_fleet_gain_vs_baseline_pp": fmean(
                        point["fleet_gain_vs_baseline_pp"] for point in points
                    ),
                }
            )
        ranked = _rank_scores(seed_scores)
        for rank, score in enumerate(ranked, start=1):
            score["request_rank"] = rank
        strategy_scores[str(seed)] = ranked
        dynamic_ranked = [score for score in ranked if score["strategy"] in DYNAMIC_STRATEGIES]
        winners[str(seed)] = {
            "overall_request": ranked[0]["strategy"],
            "best_dynamic_request": dynamic_ranked[0]["strategy"],
            "fleet": max(
                ranked,
                key=lambda score: (
                    score["mean_fleet_npu_compute_utilization"],
                    score["strategy"],
                ),
            )["strategy"],
        }

    cross_seed = None
    if 42 in indexes and 43 in indexes:
        score_lookup = {
            seed: {row["strategy"]: row for row in strategy_scores[str(seed)]}
            for seed in (42, 43)
        }
        winner42 = winners["42"]["overall_request"]
        rank43 = score_lookup[43][winner42]["request_rank"]
        by_ssu = {}
        comparison_lookup = {
            (row["seed"], row["num_ssu"], row["strategy"]): row
            for row in comparisons
        }
        for ssu in FORMAL_SSUS:
            gain42 = comparison_lookup[(42, ssu, winner42)][
                "request_gain_vs_baseline_pp"
            ]
            gain43 = comparison_lookup[(43, ssu, winner42)][
                "request_gain_vs_baseline_pp"
            ]
            by_ssu[str(ssu)] = {
                "seed42_gain_pp": gain42,
                "seed43_gain_pp": gain43,
                "direction_held": (gain42 >= 0.0) == (gain43 >= 0.0),
            }
        cross_seed = {
            "seed42_winner": winner42,
            "seed43_rank": rank43,
            "rank_held": rank43 == 1,
            "direction_by_ssu": by_ssu,
            "direction_held_all_ssus": all(
                item["direction_held"] for item in by_ssu.values()
            ),
            "strategy_mean_request_gain_change_pp": {
                strategy: score_lookup[43][strategy][
                    "mean_request_gain_vs_baseline_pp"
                ]
                - score_lookup[42][strategy]["mean_request_gain_vs_baseline_pp"]
                for strategy in STRATEGY_ORDER
            },
            "interpretation": (
                "Two deterministic seeds are a sensitivity check, not evidence "
                "of statistical significance."
            ),
        }

    return {
        "schema_version": 1,
        "contract": {
            "num_npu": FORMAL_NUM_NPU,
            "n_layers": FORMAL_LAYERS,
            "ssu_list": list(FORMAL_SSUS),
            "strategies": list(STRATEGY_ORDER),
            "cir_sum_cap_gbps": DISK_CIR_CAP_GBPS,
            "seeds": sorted(indexes),
        },
        "pairing": pairing,
        "strategy_comparisons": comparisons,
        "strategy_scores": strategy_scores,
        "winners": winners,
        "category_analysis": category_analysis,
        "request_fastest_slowest_inputs": request_extremes,
        "route_ablations": route_ablations,
        "dynamic_vs_best_fixed": dynamic_vs_best,
        "control_epochs": control_epochs,
        "cross_seed": cross_seed,
    }


def _plot_strategy_comparison(analysis, output_path):
    rows = analysis["strategy_comparisons"]
    seeds = analysis["contract"]["seeds"]
    groups = [(seed, ssu) for seed in seeds for ssu in FORMAL_SSUS]
    lookup = {
        (row["seed"], row["num_ssu"], row["strategy"]): row for row in rows
    }
    x = list(range(len(groups)))
    figure, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), sharex=True)
    colors = plt.get_cmap("tab10")
    for index, strategy in enumerate(STRATEGY_ORDER):
        axes[0].plot(
            x,
            [100.0 * lookup[(seed, ssu, strategy)]["request_compute_fraction"] for seed, ssu in groups],
            marker="o",
            linewidth=1.6,
            color=colors(index),
            label=STRATEGY_LABELS[strategy],
        )
        axes[1].plot(
            x,
            [100.0 * lookup[(seed, ssu, strategy)]["fleet_npu_compute_utilization"] for seed, ssu in groups],
            marker="o",
            linewidth=1.6,
            color=colors(index),
        )
    axes[0].set_ylabel("Avg request compute fraction (%)")
    axes[1].set_ylabel("Fleet NPU compute utilization (%)")
    axes[1].set_xlabel("Seed / SSU")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["seed %d\n%d SSU" % group for group in groups])
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Joint dynamic Path+CIR strategy comparison")
    figure.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_ablations(analysis, output_path):
    seeds = analysis["contract"]["seeds"]
    groups = [(seed, ssu) for seed in seeds for ssu in FORMAL_SSUS]
    x = list(range(len(groups)))
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))

    seed42 = seeds[0]
    best_dynamic = analysis["winners"][str(seed42)]["best_dynamic_request"]
    category_lookup = {
        (row["seed"], row["num_ssu"], row["strategy"], row["category"]): row
        for row in analysis["category_analysis"]
    }
    width = 0.18
    for category_index, category in enumerate(CATEGORIES):
        axes[0, 0].bar(
            [position + (category_index - 1.5) * width for position in x],
            [
                category_lookup[(seed, ssu, best_dynamic, category)][
                    "gain_vs_best_fixed_pp"
                ]
                for seed, ssu in groups
            ],
            width,
            label=category,
        )
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Best dynamic category gain vs best fixed")
    axes[0, 0].set_ylabel("Request gain (pp)")
    axes[0, 0].legend(ncol=4)

    route_lookup = {
        (row["seed"], row["num_ssu"], row["mode"]): row
        for row in analysis["route_ablations"]
    }
    for mode_index, (mode, _, _) in enumerate(ROUTE_ABLATIONS):
        offset = (mode_index - 0.5) * 0.32
        axes[0, 1].bar(
            [position + offset for position in x],
            [route_lookup[(seed, ssu, mode)]["request_route_gain_pp"] for seed, ssu in groups],
            0.32,
            label=mode,
        )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Least-work Path routing ablation")
    axes[0, 1].set_ylabel("Joint - fixed request gain (pp)")
    axes[0, 1].legend()

    dynamic_lookup = {
        (row["seed"], row["num_ssu"], row["strategy"]): row
        for row in analysis["dynamic_vs_best_fixed"]
    }
    dynamic_width = 0.19
    for strategy_index, strategy in enumerate(DYNAMIC_STRATEGIES):
        axes[1, 0].bar(
            [position + (strategy_index - 1.5) * dynamic_width for position in x],
            [dynamic_lookup[(seed, ssu, strategy)]["request_gain_vs_best_fixed_pp"] for seed, ssu in groups],
            dynamic_width,
            label=STRATEGY_LABELS[strategy],
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Dynamic Path+CIR vs best fixed static")
    axes[1, 0].set_ylabel("Request gain (pp)")
    axes[1, 0].legend(fontsize=8)

    epoch_lookup = {
        (row["seed"], row["num_ssu"], row["strategy"]): row
        for row in analysis["control_epochs"]
    }
    for strategy in DYNAMIC_STRATEGIES:
        axes[1, 1].plot(
            x,
            [epoch_lookup[(seed, ssu, strategy)]["epochs_per_million_blocks"] for seed, ssu in groups],
            marker="o",
            label=STRATEGY_LABELS[strategy],
        )
    axes[1, 1].set_title("Dynamic control epoch density")
    axes[1, 1].set_ylabel("Epochs per million blocks")
    axes[1, 1].legend(fontsize=8)

    labels = ["S%d/%d" % group for group in groups]
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=30, ha="right")
    figure.suptitle(
        "Category, Path-routing, CIR policy, and control-epoch analysis\n"
        "Best dynamic: %s" % STRATEGY_LABELS[best_dynamic]
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _format_pp(value):
    return "%+.3fpp" % value


def build_report(analysis):
    seeds = analysis["contract"]["seeds"]
    lines = [
        "# 联合动态 Path + CIR 分析",
        "",
        "严格合同校验已通过：128 NPU、16 层、40/56/80 SSU、配对 workload/placement、"
        "全部 invariants，且动态控制记录的历史最大 ΣCIR 不超过 40 GB/s。",
        "",
        "动态策略的控制语义是：NPU 上报当前层 demand 并选择自己的两条专属 Path；"
        "每块 SSU 在下一条非抢占命令前原子归一化并安装最终 CIR。配置延迟按要求设为 0，"
        "而且可以逐命令边界更新，因此结果是零成本控制的乐观在线 heuristic，不等同于"
        "已经验证过寄存器写入周期的成熟硬件实现。",
        "",
        "## 策略总览",
        "",
    ]
    comparisons = {
        (row["seed"], row["num_ssu"], row["strategy"]): row
        for row in analysis["strategy_comparisons"]
    }
    for seed in seeds:
        lines.extend(
            [
                "### Seed %d" % seed,
                "",
                "| Strategy | 40 request/fleet | 56 request/fleet | 80 request/fleet | Mean request gain |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        score_lookup = {
            row["strategy"]: row for row in analysis["strategy_scores"][str(seed)]
        }
        for strategy in STRATEGY_ORDER:
            points = [comparisons[(seed, ssu, strategy)] for ssu in FORMAL_SSUS]
            lines.append(
                "| %s | %s | %s | %s | %s |"
                % (
                    STRATEGY_LABELS[strategy],
                    "%.2f%% / %.2f%%" % (
                        100.0 * points[0]["request_compute_fraction"],
                        100.0 * points[0]["fleet_npu_compute_utilization"],
                    ),
                    "%.2f%% / %.2f%%" % (
                        100.0 * points[1]["request_compute_fraction"],
                        100.0 * points[1]["fleet_npu_compute_utilization"],
                    ),
                    "%.2f%% / %.2f%%" % (
                        100.0 * points[2]["request_compute_fraction"],
                        100.0 * points[2]["fleet_npu_compute_utilization"],
                    ),
                    _format_pp(
                        score_lookup[strategy]["mean_request_gain_vs_baseline_pp"]
                    ),
                )
            )
        winner = analysis["winners"][str(seed)]
        lines.extend(
            [
                "",
                "按三个 SSU 等权平均，Request 第一：`%s`；最佳动态策略：`%s`；"
                "fleet 第一：`%s`。"
                % (
                    winner["overall_request"],
                    winner["best_dynamic_request"],
                    winner["fleet"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Path 选路增益",
            "",
            "| Seed | SSU | CIR mode | Request joint-fixed | Fleet joint-fixed | Improved/regressed requests |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in analysis["route_ablations"]:
        lines.append(
            "| %d | %d | %s | %s | %s | %d / %d |"
            % (
                row["seed"],
                row["num_ssu"],
                row["mode"],
                _format_pp(row["request_route_gain_pp"]),
                _format_pp(row["fleet_route_gain_pp"]),
                row["improved_request_count"],
                row["regressed_request_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Dynamic vs best fixed static",
            "",
            "| Seed | SSU | Dynamic strategy | Request | Fleet | Largest request gain | Largest regression |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["dynamic_vs_best_fixed"]:
        lines.append(
            "| %d | %d | %s | %s | %s | %s (#%d) | %s (#%d) |"
            % (
                row["seed"],
                row["num_ssu"],
                STRATEGY_LABELS[row["strategy"]],
                _format_pp(row["request_gain_vs_best_fixed_pp"]),
                _format_pp(row["fleet_gain_vs_best_fixed_pp"]),
                _format_pp(row["largest_gain"]["delta_pp"]),
                row["largest_gain"]["input"]["request_id"],
                _format_pp(row["largest_regression"]["delta_pp"]),
                row["largest_regression"]["input"]["request_id"],
            )
        )

    lines.extend(["", "## 类别归因", ""])
    category_lookup = {
        (row["seed"], row["num_ssu"], row["strategy"], row["category"]): row
        for row in analysis["category_analysis"]
    }
    for seed in seeds:
        best_dynamic = analysis["winners"][str(seed)]["best_dynamic_request"]
        lines.extend(
            [
                "Seed %d 最佳动态策略 `%s` 相对 best fixed（三个 SSU 等权平均）："
                % (seed, best_dynamic),
                "",
                "`Request gain` 是相对 best fixed 的差值；三个 wait 列是该动态策略的绝对时间。",
                "",
                "| Category | Request gain | I/O wait (absolute) | SSD queue wait (absolute) | NPU link wait (absolute) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for category in CATEGORIES:
            rows = [
                category_lookup[(seed, ssu, best_dynamic, category)]
                for ssu in FORMAL_SSUS
            ]
            lines.append(
                "| %s | %s | %.3fms | %.4fms | %.4fms |"
                % (
                    category,
                    _format_pp(fmean(row["gain_vs_best_fixed_pp"] for row in rows)),
                    fmean(row["avg_io_wait_total_ms"] for row in rows),
                    fmean(row["avg_ssd_queue_wait_ms"] for row in rows),
                    fmean(row["avg_npu_link_queue_wait_ms"] for row in rows),
                )
            )
        lines.append("")

    lines.extend(["## 最快/最慢输入", ""])
    extremes = {
        (row["seed"], row["num_ssu"], row["strategy"]): row
        for row in analysis["request_fastest_slowest_inputs"]
    }
    for seed in seeds:
        best_dynamic = analysis["winners"][str(seed)]["best_dynamic_request"]
        lines.extend(
            [
                "Seed %d 的最佳动态策略 `%s`：" % (seed, best_dynamic),
                "",
                "| SSU | Fastest input | Fraction | Slowest input | Fraction |",
                "|---:|---|---:|---|---:|",
            ]
        )
        for ssu in FORMAL_SSUS:
            row = extremes[(seed, ssu, best_dynamic)]
            fast = row["fastest_input"]
            slow = row["slowest_input"]
            lines.append(
                "| %d | #%d %s, BW %.2f | %.2f%% | #%d %s, BW %.2f | %.2f%% |"
                % (
                    ssu,
                    fast["request_id"],
                    fast["category"],
                    fast["required_bw_input_gbps"],
                    100.0 * fast["request_compute_fraction"],
                    slow["request_id"],
                    slow["category"],
                    slow["required_bw_input_gbps"],
                    100.0 * slow["request_compute_fraction"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "## 控制 epoch 与 CIR",
            "",
            "| Seed | SSU | Strategy | Epochs | Epochs/M blocks | Max ΣCIR |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in analysis["control_epochs"]:
        lines.append(
            "| %d | %d | %s | %d | %.1f | %.6f |"
            % (
                row["seed"],
                row["num_ssu"],
                STRATEGY_LABELS[row["strategy"]],
                row["epochs"],
                row["epochs_per_million_blocks"],
                row["max_total_cir_gbps"],
            )
        )

    if analysis["cross_seed"] is not None:
        cross = analysis["cross_seed"]
        lines.extend(
            [
                "",
                "## 跨 seed 敏感性",
                "",
                "Seed42 赢家 `%s` 在 seed43 排名 %d；排名保持：**%s**，"
                "40/56/80 SSU 增益方向全部保持：**%s**。"
                % (
                    cross["seed42_winner"],
                    cross["seed43_rank"],
                    cross["rank_held"],
                    cross["direction_held_all_ssus"],
                ),
                "",
                "这只是两个确定性 seed 的敏感性检查，不能建立统计显著性。",
            ]
        )

    lines.extend(
        [
            "",
            "## 图",
            "",
            "![Strategy comparison](01_joint_dynamic_strategy_comparison.png)",
            "",
            "![Ablations and epochs](02_category_path_cir_epochs.png)",
            "",
        ]
    )
    return "\n".join(lines)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def run_analysis(seed42_path, output_dir, seed43_path=None):
    seed_data = {42: load_json(seed42_path)}
    if seed43_path is not None:
        seed_data[43] = load_json(seed43_path)
    analysis = analyze_results(seed_data)
    analysis["inputs"] = {
        "seed42": str(Path(seed42_path)),
        "seed43": str(Path(seed43_path)) if seed43_path is not None else None,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_strategy_comparison(
        analysis, output_dir / "01_joint_dynamic_strategy_comparison.png"
    )
    _plot_ablations(analysis, output_dir / "02_category_path_cir_epochs.png")
    write_json(output_dir / "analysis.json", analysis)
    (output_dir / "report.md").write_text(build_report(analysis))
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed42", type=Path, default=DEFAULT_SEED42)
    parser.add_argument("--seed43", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analysis = run_analysis(args.seed42, args.output_dir, args.seed43)
    print("analysis: %s" % (args.output_dir / "analysis.json"))
    print("report:   %s" % (args.output_dir / "report.md"))
    print("figures:  2")
    print("winner seed42: %s" % analysis["winners"]["42"]["overall_request"])


if __name__ == "__main__":
    main()
