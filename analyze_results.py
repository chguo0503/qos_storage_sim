"""Analyze the paired full QoS experiment and render publication-ready plots.

The input is the ``results.json`` produced by ``analysis_experiment.py``.  The
normal simulation summary and per-request records intentionally remain separate;
the fluid upper-bound row has its own metric names and is handled explicitly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_experiment import (
    SCHEMA_VERSION as EXPERIMENT_SCHEMA_VERSION,
    _code_fingerprint as current_code_fingerprint,
    strategy_specs as current_strategy_specs,
)
from allocation_calibration import run_two_npu_calibration


DEFAULT_INPUT = Path("results/full_analysis/results.json")
DEFAULT_OUTPUT_DIR = Path("results/full_analysis")
FORMAL_SSUS = (40, 56, 80)
FORMAL_NUM_NPU = 128
FORMAL_LAYERS = 16
FORMAL_STRATEGY_COUNT = 24
CATEGORY_ORDER = ("SS", "SL", "LS", "LL")
BW_BIN_EDGES = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, math.inf)
FIGURE_DPI = 180

EXPECTED_FORMAL_BACKEND = {
    "model": "shared_two_stage_ssd40_then_npu50_single_server_v1",
    "ssd_service": "io_size_gb / 40 GB/s",
    "ssd_max_active_io": 1,
    "npu_service": "io_size_gb / 50 GB/s",
    "npu_max_active_io": 1,
    "block_visible_after": "npu_link_completion",
}
EXPECTED_DATA_PLANE_STAGES = {
    "ssd": {
        "discipline": "policy_select_then_one_nonpreemptive_command",
        "service_time": "io_size_gb / disk_bw_gbps",
        "max_active_io": 1,
    },
    "npu_link": {
        "discipline": "fcfs_store_and_forward",
        "service_time": "io_size_gb / npu_bw_limit_gbps",
        "max_active_io": 1,
    },
    "block_visible_after": "npu_link_completion",
    "path_pressure_released_after": "ssd_completion",
    "intermediate_buffer": "unbounded_store_and_forward",
}

EXPECTED_FORMAL_RUNTIME_KEYS = {
    "mode",
    "num_npu",
    "n_layers",
    "ssu_list",
    "ls_ratio",
    "seeds",
    "arrival_delay_ms",
    "disk_bw_gbps",
    "npu_bw_limit_gbps",
}
EXPECTED_SEED_KEYS = {
    "workload",
    "placement",
    "submit_order",
    "arrival_delay",
}


def load_results(path):
    return json.loads(path.read_text())


def require_formal_provenance(data):
    experiment = data["experiment"]
    if (
        data["schema_version"] != EXPERIMENT_SCHEMA_VERSION
        or experiment["schema_version"] != EXPERIMENT_SCHEMA_VERSION
    ):
        raise RuntimeError("formal data and experiment schema versions must match")
    if experiment["code_fingerprint"] != current_code_fingerprint():
        raise RuntimeError("formal result code fingerprint is not current")

    specs = current_strategy_specs()
    if len(specs) != FORMAL_STRATEGY_COUNT:
        raise RuntimeError("current strategy registry is not the formal 24-strategy set")
    expected_configs = [spec.config() for spec in specs]
    if experiment["available_strategies"] != expected_configs:
        raise RuntimeError("formal available strategy configs do not match current code")
    expected_names = [spec.name for spec in specs]
    if data["selected_strategies"] != expected_names:
        raise RuntimeError("formal selected strategy order does not match current code")

    runtime = experiment["runtime"]
    seeds = runtime["seeds"]
    if (
        set(runtime) != EXPECTED_FORMAL_RUNTIME_KEYS
        or set(seeds) != EXPECTED_SEED_KEYS
        or runtime["mode"] != "formal"
        or runtime["num_npu"] != FORMAL_NUM_NPU
        or runtime["n_layers"] != FORMAL_LAYERS
        or tuple(runtime["ssu_list"]) != FORMAL_SSUS
        or runtime["ls_ratio"] != 0.5
        or runtime["arrival_delay_ms"] != [0.0, 5.0]
        or runtime["disk_bw_gbps"] != 40.0
        or runtime["npu_bw_limit_gbps"] != 50.0
        or seeds["placement"] != seeds["workload"] + 1_000_003
        or seeds["submit_order"] != seeds["workload"] + 2_000_003
        or seeds["arrival_delay"] != seeds["workload"] + 3_000_003
    ):
        raise RuntimeError("formal runtime contract is not the current 128/16/40,56,80 setup")
    if experiment["formal_contract"] != {
        "num_npu": FORMAL_NUM_NPU,
        "n_layers": FORMAL_LAYERS,
        "ssu_list": list(FORMAL_SSUS),
    }:
        raise RuntimeError("formal experiment contract metadata is not current")
    if experiment["backend"] != EXPECTED_FORMAL_BACKEND:
        raise RuntimeError("formal experiment backend metadata is not current")
    return specs


def require_row_contract(row, spec, runtime):
    if (
        row["config"] != spec.config()
        or row["kind"] != spec.kind
        or row["family"] != spec.family
    ):
        raise RuntimeError("result row config/kind/family does not match its strategy spec")
    if row["seeds"] != runtime["seeds"]:
        raise RuntimeError("result row seeds do not match the formal runtime")
    if spec.kind == "upper_bound":
        return

    summary = row["summary"]
    if (
        summary["policy"] != spec.policy
        or summary["backend_model"] != EXPECTED_FORMAL_BACKEND["model"]
        or summary["data_plane_stages"] != EXPECTED_DATA_PLANE_STAGES
        or summary["backend_capacity_gbps"] != runtime["disk_bw_gbps"]
        or summary["npu_bw_limit_gbps"] != runtime["npu_bw_limit_gbps"]
    ):
        raise RuntimeError("simulation row does not match the formal data-plane contract")


def require_request_metric_conservation(row):
    requests = row["request_metrics"]
    request_mean = float(
        np.mean([request_metric_value(row, item) for item in requests])
    )
    if not math.isclose(
        metric_value(row), request_mean, rel_tol=1e-12, abs_tol=1e-12
    ):
        metric_name = "bound" if row["kind"] == "upper_bound" else "simulation"
        raise RuntimeError(
            "%s summary average does not equal its 128-request mean" % metric_name
        )


def require_complete(data):
    if not data.get("complete", False):
        raise RuntimeError("refusing to analyze an incomplete formal result matrix")
    specs = require_formal_provenance(data)
    runtime = data["experiment"]["runtime"]
    selected = list(data["selected_strategies"])
    spec_by_name = {spec.name: spec for spec in specs}

    rows = data["results"]
    expected_keys = {
        (ssu, strategy) for ssu in FORMAL_SSUS for strategy in selected
    }
    keys = [(int(row["num_ssu"]), row["strategy"]) for row in rows]
    if len(keys) != len(expected_keys) or len(set(keys)) != len(keys):
        raise RuntimeError("formal result matrix contains missing or duplicate rows")
    if set(keys) != expected_keys:
        raise RuntimeError("formal result matrix does not match the 24 x 3 contract")

    index = {(int(row["num_ssu"]), row["strategy"]): row for row in rows}
    for row in rows:
        require_row_contract(row, spec_by_name[row["strategy"]], runtime)
    workload_hashes = {row["workload_fingerprint"] for row in rows}
    if len(workload_hashes) != 1:
        raise RuntimeError("formal SSU cases do not share one paired workload")
    for ssu in FORMAL_SSUS:
        placement_hashes = {
            index[(ssu, strategy)]["placement_hash"] for strategy in selected
        }
        if len(placement_hashes) != 1:
            raise RuntimeError("strategies within an SSU do not share one placement")

    baseline_by_ssu = {
        ssu: request_map(index[(ssu, "baseline")]) for ssu in FORMAL_SSUS
    }
    immutable_fields = (
        "category",
        "seq_len_k",
        "nql",
        "required_bw_input_gbps",
        "arrival_delay_ms",
        "per_layer_us",
        "per_layer_kv_gb",
    )
    expected_ids = set(range(FORMAL_NUM_NPU))
    reference_inputs = baseline_by_ssu[FORMAL_SSUS[0]]
    for ssu, baseline in baseline_by_ssu.items():
        if (
            len(index[(ssu, "baseline")]["request_metrics"]) != FORMAL_NUM_NPU
            or set(baseline) != expected_ids
        ):
            raise RuntimeError("baseline request IDs are not exactly 0..127")
        for request_id in expected_ids:
            if any(
                baseline[request_id][field] != reference_inputs[request_id][field]
                for field in immutable_fields
            ):
                raise RuntimeError("request inputs differ across SSU cases")

        delays = [float(baseline[request_id]["arrival_delay_ms"]) for request_id in expected_ids]
        if min(delays) < 0.0 or max(delays) > 5.0 or len(set(delays)) != FORMAL_NUM_NPU:
            raise RuntimeError("formal run does not contain 128 unique [0,5] ms jitters")

        for strategy in selected:
            row = index[(ssu, strategy)]
            candidate = request_map(row)
            if (
                len(row["request_metrics"]) != FORMAL_NUM_NPU
                or set(candidate) != expected_ids
            ):
                raise RuntimeError("paired strategy request IDs differ from baseline")
            for request_id in expected_ids:
                fields = ("category",) if row["kind"] == "upper_bound" else immutable_fields
                if any(
                    candidate[request_id][field] != baseline[request_id][field]
                    for field in fields
                ):
                    raise RuntimeError("paired request input fields differ from baseline")
            if row["kind"] == "simulation" and not all(
                row["summary"]["invariants"].values()
            ):
                raise RuntimeError("a simulation row violates its data-plane invariants")
            require_request_metric_conservation(row)

        bound_row = index[(ssu, "isolated_no_contention_bound")]
        bound = metric_value(bound_row)
        bound_requests = request_map(bound_row)
        for strategy in selected:
            row = index[(ssu, strategy)]
            if row["kind"] == "simulation" and metric_value(row) > bound + 1e-12:
                raise RuntimeError("a runnable strategy exceeds the declared fluid upper bound")
            if row["kind"] == "simulation":
                for request_id, request in request_map(row).items():
                    if (
                        request_metric_value(row, request)
                        > request_metric_value(
                            bound_row, bound_requests[request_id]
                        )
                        + 1e-12
                    ):
                        raise RuntimeError(
                            "a runnable request exceeds its paired fluid upper bound"
                        )


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def metric_value(row):
    summary = row["summary"]
    if row["kind"] == "upper_bound":
        return float(summary["avg_request_compute_fraction_upper_bound"])
    return float(summary["avg_request_compute_fraction"])


def fleet_metric_value(row):
    if row["kind"] == "upper_bound":
        return None
    return float(row["summary"]["fleet_npu_compute_utilization"])


def request_metric_value(row, request):
    if row["kind"] == "upper_bound":
        return float(request["compute_fraction_upper_bound"])
    return float(request["request_npu_utilization"])


def row_index(data):
    keys = [(int(row["num_ssu"]), row["strategy"]) for row in data["results"]]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate (SSU, strategy) result rows")
    return {key: row for key, row in zip(keys, data["results"])}


def actual_ssus(data):
    return sorted({int(row["num_ssu"]) for row in data["results"]})


def selection_ssus(data):
    present = actual_ssus(data)
    runtime = data.get("experiment", {}).get("runtime", {})
    if runtime.get("mode") == "formal" and all(ssu in present for ssu in FORMAL_SSUS):
        return list(FORMAL_SSUS)
    return present


def strategy_rows(data):
    result = defaultdict(dict)
    for row in data["results"]:
        result[row["strategy"]][int(row["num_ssu"])] = row
    return result


def fixed_strategy_scores(data, names, ssus):
    by_strategy = strategy_rows(data)
    scores = []
    for name in sorted(set(names)):
        rows = by_strategy.get(name, {})
        if all(ssu in rows for ssu in ssus):
            values = [metric_value(rows[ssu]) for ssu in ssus]
            fleet_values = [fleet_metric_value(rows[ssu]) for ssu in ssus]
            scores.append(
                {
                    "strategy": name,
                    "mean_compute_fraction": float(np.mean(values)),
                    "min_compute_fraction": float(np.min(values)),
                    "values_by_ssu": {
                        str(ssu): metric_value(rows[ssu]) for ssu in ssus
                    },
                    "mean_fleet_npu_compute_utilization": float(
                        np.mean(fleet_values)
                    ),
                    "fleet_values_by_ssu": {
                        str(ssu): fleet_metric_value(rows[ssu]) for ssu in ssus
                    },
                }
            )
    scores.sort(
        key=lambda item: (
            -item["mean_compute_fraction"],
            -item["min_compute_fraction"],
            item["strategy"],
        )
    )
    return scores


def choose_fixed_strategies(data, ssus):
    rows = data["results"]
    static_names = {
        row["strategy"]
        for row in rows
        if row.get("family") == "tuning"
    }
    static_names.add("current_refresh8")
    ticket_names = {
        row["strategy"] for row in rows if row.get("family") == "ticket"
    }
    practical_names = {
        row["strategy"]
        for row in rows
        if row.get("family") in {"cadence", "batch", "tuning", "ticket"}
    }
    practical_names.add("demand_maxmin")
    static_scores = fixed_strategy_scores(data, static_names, ssus)
    ticket_scores = fixed_strategy_scores(data, ticket_names, ssus)
    practical_scores = fixed_strategy_scores(data, practical_names, ssus)
    practical_fleet_scores = sorted(
        practical_scores,
        key=lambda item: (
            -item["mean_fleet_npu_compute_utilization"],
            item["strategy"],
        ),
    )
    request_winner = practical_scores[0]["strategy"] if practical_scores else None
    fleet_winner = (
        practical_fleet_scores[0]["strategy"] if practical_fleet_scores else None
    )
    return {
        "rule": (
            "Choose one strategy name for every selected SSU; maximize the "
            "unweighted mean request compute fraction across those SSUs, then "
            "maximize the worst-SSU value, then use lexical name order. Static "
            "selection is restricted to refresh8/batch8 CIR/path candidates so "
            "demand-aware and ticket policies remain independently identifiable. "
            "Practical selection covers every runnable cadence, batch, tuning, "
            "ticket, and demand policy while excluding the information-advantaged "
            "EDF/global policies and the infeasible bound; it also receives a "
            "separate fleet-utilization ranking."
        ),
        "selection_ssus": list(ssus),
        "best_static": static_scores[0]["strategy"] if static_scores else None,
        "best_ticket": ticket_scores[0]["strategy"] if ticket_scores else None,
        "best_nonideal_practical": request_winner,
        "best_nonideal_practical_request": request_winner,
        "best_nonideal_practical_fleet": fleet_winner,
        "static_scores": static_scores,
        "ticket_scores": ticket_scores,
        "practical_scores": practical_scores,
        "practical_fleet_scores": practical_fleet_scores,
        "selection_is_in_sample": True,
    }


def strategy_label(name, selection):
    labels = {
        "baseline": "Baseline NPU-RR",
        "current_refresh8": "Current static CIR",
        "demand_maxmin": "Demand max-min",
        "per_ssd_full_visible_edf": "Per-SSD full-visible EDF",
        "global_link_aware_online": "Global link-aware online",
        "isolated_no_contention_bound": "Fluid upper bound",
        "tune__low_protect_cir_20_6_8_6_current_paths": (
            "Tuned static CIR 20/6/8/6"
        ),
        "tune__low_protect_cir_20_5_10_5_paths_12_5_10_5": (
            "Tuned static CIR 20/5/10/5, paths 12/5/10/5"
        ),
    }
    base = labels.get(name, name.replace("_", " "))
    roles = []
    if name == selection.get("best_static"):
        roles.append("best fixed")
    if name == selection.get("best_ticket"):
        roles.append("best ticket")
    if name == selection.get("best_nonideal_practical_request"):
        roles.append("request winner")
    if name == selection.get("best_nonideal_practical_fleet"):
        roles.append("fleet winner")
    return "%s (%s)" % (base, "; ".join(roles)) if roles else base


def overview_strategy_names(selection):
    names = ["baseline", "current_refresh8"]
    names.append(selection.get("best_static"))
    names.append(selection.get("best_nonideal_practical"))
    names.append(selection.get("best_nonideal_practical_fleet"))
    names.extend(
        [
            "demand_maxmin",
            selection.get("best_ticket"),
            "per_ssd_full_visible_edf",
            "global_link_aware_online",
            "isolated_no_contention_bound",
        ]
    )
    result = []
    for name in names:
        if name is not None and name not in result:
            result.append(name)
    return result


def overview_table(data, selection, ssus):
    by_strategy = strategy_rows(data)
    table = []
    for name in overview_strategy_names(selection):
        rows = by_strategy.get(name, {})
        values = {
            str(ssu): metric_value(rows[ssu]) if ssu in rows else None
            for ssu in ssus
        }
        fleet_values = {
            str(ssu): fleet_metric_value(rows[ssu]) if ssu in rows else None
            for ssu in ssus
        }
        available = [value for value in values.values() if value is not None]
        table.append(
            {
                "strategy": name,
                "label": strategy_label(name, selection),
                "kind": next(
                    (row["kind"] for row in rows.values()), "missing"
                ),
                "compute_fraction_by_ssu": values,
                "fleet_npu_compute_utilization_by_ssu": fleet_values,
                "mean_compute_fraction": (
                    float(np.mean(available)) if available else None
                ),
                "mean_fleet_npu_compute_utilization": (
                    float(
                        np.mean(
                            [value for value in fleet_values.values() if value is not None]
                        )
                    )
                    if any(value is not None for value in fleet_values.values())
                    else None
                ),
            }
        )
    return table


def request_map(row):
    return {int(item["request_id"]): item for item in row["request_metrics"]}


def pearson(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def bw_bin_label(value):
    for low, high in zip(BW_BIN_EDGES[:-1], BW_BIN_EDGES[1:]):
        if low <= value < high:
            if math.isinf(high):
                return "50+"
            return "%g-%g" % (low, high)
    raise AssertionError("required bandwidth is outside the configured bins")


def input_bin_label(request):
    return "seq=%gk,nql=%g" % (
        float(request["seq_len_k"]),
        float(request["nql"]),
    )


def actual_layer_demand_gbps(request):
    return float(request["per_layer_kv_gb"]) / (
        float(request["per_layer_us"]) / 1_000_000.0
    )


def capped_layer_demand_gbps(request):
    return min(actual_layer_demand_gbps(request), 50.0)


def capped_demand_bin_label(request):
    value = capped_layer_demand_gbps(request)
    return "50-cap" if value == 50.0 else bw_bin_label(value)


def paired_request_rows(baseline_row, candidate_row):
    baseline = request_map(baseline_row)
    candidate = request_map(candidate_row)
    paired = []
    for request_id in sorted(baseline):
        before = baseline[request_id]
        after = candidate[request_id]
        delta_pp = 100.0 * (
            request_metric_value(candidate_row, after)
            - request_metric_value(baseline_row, before)
        )
        row = {
            "num_ssu": int(baseline_row["num_ssu"]),
            "strategy": candidate_row["strategy"],
            "request_id": request_id,
            "category": before["category"],
            "seq_len_k": float(before["seq_len_k"]),
            "nql": float(before["nql"]),
            "required_bw_gbps": float(before["required_bw_input_gbps"]),
            "required_bw_bin": bw_bin_label(
                float(before["required_bw_input_gbps"])
            ),
            "actual_layer_demand_gbps": actual_layer_demand_gbps(before),
            "actual_layer_demand_bin": bw_bin_label(
                actual_layer_demand_gbps(before)
            ),
            "capped_layer_demand_gbps": capped_layer_demand_gbps(before),
            "capped_layer_demand_bin": capped_demand_bin_label(before),
            "input_bin": input_bin_label(before),
            "arrival_delay_ms": float(before["arrival_delay_ms"]),
            "per_layer_us": float(before["per_layer_us"]),
            "per_layer_kv_gb": float(before["per_layer_kv_gb"]),
            "baseline_compute_fraction": request_metric_value(
                baseline_row, before
            ),
            "candidate_compute_fraction": request_metric_value(
                candidate_row, after
            ),
            "compute_delta_pp": delta_pp,
            "baseline_io_wait_ms": float(before["io_wait_total_ms"]),
            "candidate_io_wait_ms": float(after["io_wait_total_ms"]),
            "io_wait_delta_ms": float(after["io_wait_total_ms"])
            - float(before["io_wait_total_ms"]),
            "ssd_queue_delta_ms_per_io": float(after["avg_ssd_queue_wait_ms"])
            - float(before["avg_ssd_queue_wait_ms"]),
            "npu_queue_delta_ms_per_io": float(
                after["avg_npu_link_queue_wait_ms"]
            )
            - float(before["avg_npu_link_queue_wait_ms"]),
        }
        paired.append(row)
    return paired


def aggregate_pairs(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    result = []
    for key in sorted(groups):
        values = groups[key]
        deltas = np.asarray([item["compute_delta_pp"] for item in values])
        result.append(
            {
                "bin": key,
                "count": len(values),
                "mean_compute_delta_pp": float(np.mean(deltas)),
                "median_compute_delta_pp": float(np.median(deltas)),
                "p10_compute_delta_pp": float(np.percentile(deltas, 10)),
                "p90_compute_delta_pp": float(np.percentile(deltas, 90)),
                "improved_fraction": float(np.mean(deltas > 1e-12)),
                "slower_fraction": float(np.mean(deltas < -1e-12)),
                "mean_io_wait_delta_ms": float(
                    np.mean([item["io_wait_delta_ms"] for item in values])
                ),
                "mean_ssd_queue_delta_ms_per_io": float(
                    np.mean(
                        [item["ssd_queue_delta_ms_per_io"] for item in values]
                    )
                ),
                "mean_npu_queue_delta_ms_per_io": float(
                    np.mean(
                        [item["npu_queue_delta_ms_per_io"] for item in values]
                    )
                ),
            }
        )
    return result


def summarize_pairs(rows):
    deltas = np.asarray([row["compute_delta_pp"] for row in rows], dtype=float)
    result = {
        "count": len(rows),
        "mean_compute_delta_pp": float(np.mean(deltas)),
        "median_compute_delta_pp": float(np.median(deltas)),
        "p10_compute_delta_pp": float(np.percentile(deltas, 10)),
        "p90_compute_delta_pp": float(np.percentile(deltas, 90)),
        "improved_count": int(np.sum(deltas > 1e-12)),
        "slower_count": int(np.sum(deltas < -1e-12)),
        "unchanged_count": int(np.sum(np.abs(deltas) <= 1e-12)),
        "mean_io_wait_delta_ms": float(
            np.mean([row["io_wait_delta_ms"] for row in rows])
        ),
        "mean_ssd_queue_delta_ms_per_io": float(
            np.mean([row["ssd_queue_delta_ms_per_io"] for row in rows])
        ),
        "mean_npu_queue_delta_ms_per_io": float(
            np.mean([row["npu_queue_delta_ms_per_io"] for row in rows])
        ),
        "correlation_delta_vs_required_bw": pearson(
            [row["required_bw_gbps"] for row in rows], deltas
        ),
        "correlation_delta_vs_arrival_delay": pearson(
            [row["arrival_delay_ms"] for row in rows], deltas
        ),
        "correlation_delta_vs_io_wait_delta": pearson(
            [row["io_wait_delta_ms"] for row in rows], deltas
        ),
    }
    result["by_category"] = aggregate_pairs(rows, "category")
    result["by_required_bw_bin"] = aggregate_pairs(rows, "required_bw_bin")
    result["by_actual_demand_bin"] = aggregate_pairs(
        rows, "actual_layer_demand_bin"
    )
    result["by_capped_demand_bin"] = aggregate_pairs(
        rows, "capped_layer_demand_bin"
    )
    result["by_input_bin"] = aggregate_pairs(rows, "input_bin")
    result["fastest_inputs"] = sorted(
        rows, key=lambda row: (-row["compute_delta_pp"], row["request_id"])
    )[:10]
    result["slowest_inputs"] = sorted(
        rows, key=lambda row: (row["compute_delta_pp"], row["request_id"])
    )[:10]
    return result


def build_paired_analysis(data, selection, ssus):
    index = row_index(data)
    names = [
        "current_refresh8",
        selection.get("best_static"),
        selection.get("best_ticket"),
        selection.get("best_nonideal_practical"),
        selection.get("best_nonideal_practical_fleet"),
        "demand_maxmin",
    ]
    names = [name for index_, name in enumerate(names) if name and name not in names[:index_]]
    all_pairs = []
    comparisons = {}
    for name in names:
        strategy_pairs = []
        by_ssu = {}
        for ssu in ssus:
            baseline_key = (ssu, "baseline")
            candidate_key = (ssu, name)
            if baseline_key not in index or candidate_key not in index:
                continue
            pairs = paired_request_rows(index[baseline_key], index[candidate_key])
            strategy_pairs.extend(pairs)
            all_pairs.extend(pairs)
            by_ssu[str(ssu)] = summarize_pairs(pairs)
        if strategy_pairs:
            comparisons[name] = {
                "all_ssus": summarize_pairs(strategy_pairs),
                "by_ssu": by_ssu,
            }
    return {"comparisons": comparisons, "paired_requests": all_pairs}


def category_wait_analysis(data, selection, ssus):
    index = row_index(data)
    names = [
        "current_refresh8",
        selection.get("best_static"),
        selection.get("best_nonideal_practical"),
        selection.get("best_nonideal_practical_fleet"),
        "demand_maxmin",
        "global_link_aware_online",
    ]
    names = [name for i, name in enumerate(names) if name and name not in names[:i]]
    rows = []
    for ssu in ssus:
        baseline_row = index.get((ssu, "baseline"))
        if baseline_row is None:
            continue
        baseline_requests = request_map(baseline_row)
        for name in names:
            candidate_row = index.get((ssu, name))
            if candidate_row is None:
                continue
            candidate_requests = request_map(candidate_row)
            for category in CATEGORY_ORDER:
                ids = [
                    request_id
                    for request_id, request in baseline_requests.items()
                    if request["category"] == category
                ]

                def mean_for(records, field):
                    return float(np.mean([float(records[i][field]) for i in ids]))

                baseline_compute = float(
                    np.mean(
                        [
                            float(baseline_requests[i]["request_npu_utilization"])
                            for i in ids
                        ]
                    )
                )
                candidate_compute = float(
                    np.mean(
                        [
                            float(candidate_requests[i]["request_npu_utilization"])
                            for i in ids
                        ]
                    )
                )
                rows.append(
                    {
                        "num_ssu": ssu,
                        "strategy": name,
                        "category": category,
                        "count": len(ids),
                        "baseline_compute_fraction": baseline_compute,
                        "candidate_compute_fraction": candidate_compute,
                        "compute_delta_pp": 100.0
                        * (candidate_compute - baseline_compute),
                        "baseline_ssd_queue_ms_per_io": mean_for(
                            baseline_requests, "avg_ssd_queue_wait_ms"
                        ),
                        "candidate_ssd_queue_ms_per_io": mean_for(
                            candidate_requests, "avg_ssd_queue_wait_ms"
                        ),
                        "ssd_queue_delta_ms_per_io": mean_for(
                            candidate_requests, "avg_ssd_queue_wait_ms"
                        )
                        - mean_for(baseline_requests, "avg_ssd_queue_wait_ms"),
                        "baseline_npu_queue_ms_per_io": mean_for(
                            baseline_requests, "avg_npu_link_queue_wait_ms"
                        ),
                        "candidate_npu_queue_ms_per_io": mean_for(
                            candidate_requests, "avg_npu_link_queue_wait_ms"
                        ),
                        "npu_queue_delta_ms_per_io": mean_for(
                            candidate_requests, "avg_npu_link_queue_wait_ms"
                        )
                        - mean_for(
                            baseline_requests, "avg_npu_link_queue_wait_ms"
                        ),
                        "io_wait_delta_ms": mean_for(
                            candidate_requests, "io_wait_total_ms"
                        )
                        - mean_for(baseline_requests, "io_wait_total_ms"),
                    }
                )
    return rows


def cadence_analysis(data, ssus):
    index = row_index(data)
    names = (
        "current_layer_snapshot",
        "current_refresh8",
        "current_per_io",
    )
    rows = []
    for ssu in ssus:
        for name in names:
            row = index.get((ssu, name))
            if row is None:
                continue
            rows.append(
                {
                    "num_ssu": ssu,
                    "strategy": name,
                    "compute_fraction": metric_value(row),
                    "pressure_reports": int(row["summary"]["pressure_reports"]),
                    "pressure_telemetry_mb": float(
                        row["summary"]["pressure_telemetry_mb"]
                    ),
                    "wall_time_s": float(row["wall_time_s"]),
                }
            )
    return rows


def cadence_paired_analysis(data, ssus):
    index = row_index(data)
    reference_name = "current_layer_snapshot"
    comparisons = {}
    for candidate_name in ("current_refresh8", "current_per_io"):
        all_pairs = []
        by_ssu = {}
        for ssu in ssus:
            pairs = paired_request_rows(
                index[(ssu, reference_name)], index[(ssu, candidate_name)]
            )
            all_pairs.extend(pairs)
            by_ssu[str(ssu)] = summarize_pairs(pairs)
        comparisons[candidate_name] = {
            "reference": reference_name,
            "all_ssus": summarize_pairs(all_pairs),
            "by_ssu": by_ssu,
        }
    return {"reference": reference_name, "comparisons": comparisons}


def static_tuning_analysis(data, selection, ssus):
    by_strategy = strategy_rows(data)
    score_names = [item["strategy"] for item in selection["static_scores"]]
    result = []
    for name in score_names:
        rows = by_strategy[name]
        sample = rows[ssus[0]]
        config = sample["config"]
        values = {
            str(ssu): metric_value(rows[ssu]) for ssu in ssus if ssu in rows
        }
        baseline_values = {
            str(ssu): metric_value(by_strategy["baseline"][ssu])
            for ssu in ssus
            if ssu in by_strategy.get("baseline", {})
        }
        deltas = [
            100.0 * (values[str(ssu)] - baseline_values[str(ssu)])
            for ssu in ssus
            if str(ssu) in values and str(ssu) in baseline_values
        ]
        result.append(
            {
                "strategy": name,
                "selected": name == selection["best_static"],
                "category_cir_gbps": config.get("category_cir_gbps"),
                "category_paths_per_group": config.get(
                    "category_paths_per_group"
                ),
                "compute_fraction_by_ssu": values,
                "mean_delta_vs_baseline_pp": float(np.mean(deltas)),
            }
        )
    return result


def batch_analysis(data, ssus):
    index = row_index(data)
    names = {
        8: "current_refresh8",
        16: "current_refresh8_batch16",
        32: "current_refresh8_batch32",
    }
    rows = []
    for ssu in ssus:
        for batch, name in names.items():
            row = index.get((ssu, name))
            if row is None:
                continue
            requests = row["request_metrics"]
            l0 = float(np.mean([request["io_wait_L0_ms"] for request in requests]))
            l1 = float(np.mean([request["io_wait_L1_ms"] for request in requests]))
            l2 = float(
                np.mean([request["io_wait_L2plus_ms"] for request in requests])
            )
            layers = int(
                data.get("experiment", {}).get("runtime", {}).get("n_layers", 16)
            )
            steady_per_layer = l2 / max(layers - 2, 1)
            overall_per_layer = (l0 + l1 + l2) / layers
            rows.append(
                {
                    "num_ssu": ssu,
                    "batch_size": batch,
                    "strategy": name,
                    "compute_fraction": metric_value(row),
                    "avg_wait_L0_ms": l0,
                    "avg_wait_L1_ms": l1,
                    "avg_wait_L2plus_ms": l2,
                    "steady_L2plus_wait_ms_per_layer": steady_per_layer,
                    "overall_wait_ms_per_layer": overall_per_layer,
                    "more_layers_extrapolated_direction": (
                        "improve"
                        if steady_per_layer < overall_per_layer
                        else "degrade"
                    ),
                }
            )
    fixed_scores = []
    for batch in sorted(names):
        batch_rows = [row for row in rows if row["batch_size"] == batch]
        if len(batch_rows) == len(ssus):
            fixed_scores.append(
                {
                    "batch_size": batch,
                    "mean_compute_fraction": float(
                        np.mean([row["compute_fraction"] for row in batch_rows])
                    ),
                }
            )
    fixed_scores.sort(
        key=lambda item: (-item["mean_compute_fraction"], item["batch_size"])
    )
    return {
        "rows": rows,
        "fixed_batch_scores": fixed_scores,
        "best_fixed_batch_size": (
            fixed_scores[0]["batch_size"] if fixed_scores else None
        ),
        "layer_count_note": (
            "L2+ contains layers 2 through L-1. The more-layer direction is an "
            "extrapolation from layer-16 steady-state wait, not a separate run."
        ),
    }


def workload_analysis(data, ssus):
    index = row_index(data)
    reference = index[(ssus[0], "baseline")]
    requests = reference["request_metrics"]
    delays = np.asarray([row["arrival_delay_ms"] for row in requests], dtype=float)
    bandwidths = np.asarray(
        [row["required_bw_input_gbps"] for row in requests], dtype=float
    )
    actual_demands = np.asarray(
        [actual_layer_demand_gbps(row) for row in requests], dtype=float
    )
    capped_demands = np.minimum(actual_demands, 50.0)
    kv = np.asarray([row["per_layer_kv_gb"] for row in requests], dtype=float)
    compute = np.asarray([row["per_layer_us"] / 1000.0 for row in requests])
    category_counts = {
        category: sum(row["category"] == category for row in requests)
        for category in CATEGORY_ORDER
    }
    unique_delays = len({float(row["arrival_delay_ms"]) for row in requests})
    return {
        "reference_ssu": ssus[0],
        "request_count": len(requests),
        "arrival_delay_ms": {
            "min": float(np.min(delays)),
            "mean": float(np.mean(delays)),
            "p50": float(np.percentile(delays, 50)),
            "p95": float(np.percentile(delays, 95)),
            "max": float(np.max(delays)),
            "unique_count": unique_delays,
            "in_required_range_0_to_5": bool(
                np.all(delays >= 0.0) and np.all(delays <= 5.0)
            ),
        },
        "required_bw_gbps": {
            "min": float(np.min(bandwidths)),
            "mean": float(np.mean(bandwidths)),
            "p50": float(np.percentile(bandwidths, 50)),
            "p95": float(np.percentile(bandwidths, 95)),
            "max": float(np.max(bandwidths)),
            "over_50_count": int(np.sum(bandwidths > 50.0)),
        },
        "actual_layer_demand_gbps": {
            "min": float(np.min(actual_demands)),
            "mean": float(np.mean(actual_demands)),
            "p50": float(np.percentile(actual_demands, 50)),
            "p95": float(np.percentile(actual_demands, 95)),
            "max": float(np.max(actual_demands)),
            "over_50_count": int(np.sum(actual_demands > 50.0)),
        },
        "capped_layer_demand_gbps": {
            "min": float(np.min(capped_demands)),
            "mean": float(np.mean(capped_demands)),
            "p50": float(np.percentile(capped_demands, 50)),
            "p95": float(np.percentile(capped_demands, 95)),
            "max": float(np.max(capped_demands)),
        },
        "per_layer_kv_gb": {
            "min": float(np.min(kv)),
            "mean": float(np.mean(kv)),
            "max": float(np.max(kv)),
        },
        "per_layer_compute_ms": {
            "min": float(np.min(compute)),
            "mean": float(np.mean(compute)),
            "max": float(np.max(compute)),
        },
        "category_counts": category_counts,
        "requests": [
            {
                "request_id": int(row["request_id"]),
                "category": row["category"],
                "arrival_delay_ms": float(row["arrival_delay_ms"]),
                "required_bw_gbps": float(row["required_bw_input_gbps"]),
                "actual_layer_demand_gbps": actual_layer_demand_gbps(row),
                "capped_layer_demand_gbps": capped_layer_demand_gbps(row),
                "per_layer_kv_gb": float(row["per_layer_kv_gb"]),
                "per_layer_compute_ms": float(row["per_layer_us"]) / 1000.0,
            }
            for row in requests
        ],
    }


def system_analysis(data, selection, ssus):
    index = row_index(data)
    names = [
        name
        for name in overview_strategy_names(selection)
        if name != "isolated_no_contention_bound"
    ]
    rows = []
    for ssu in ssus:
        for name in names:
            row = index[(ssu, name)]
            summary = row["summary"]
            rows.append(
                {
                    "num_ssu": ssu,
                    "strategy": name,
                    "label": strategy_label(name, selection),
                    "request_compute_fraction": metric_value(row),
                    "fleet_npu_compute_utilization": fleet_metric_value(row),
                    "ssu_active_time_utilization": float(
                        summary["ssu_active_time_utilization"]
                    ),
                    "ssu_effective_bandwidth_utilization": float(
                        summary["ssu_effective_bandwidth_utilization"]
                    ),
                    "npu_link_utilization": float(summary["npu_link_utilization"]),
                    "avg_npu_link_queue_wait_ms": float(
                        summary["avg_npu_link_queue_wait_ms"]
                    ),
                    "makespan_ms": float(summary["makespan_ms"]),
                    "throughput_requests_per_s": float(
                        summary["throughput_requests_per_s"]
                    ),
                    "request_compute_fraction_jain": float(
                        summary["request_compute_fraction_jain"]
                    ),
                }
            )
    return rows


def allocation_alignment_analysis(data, selection, ssus):
    index = row_index(data)
    requests = index[(ssus[0], "baseline")]["request_metrics"]
    totals = {
        category: {
            "actual_demand": sum(
                actual_layer_demand_gbps(row)
                for row in requests
                if row["category"] == category
            ),
            "capped_demand": sum(
                capped_layer_demand_gbps(row)
                for row in requests
                if row["category"] == category
            ),
            "kv_bytes": sum(
                float(row["per_layer_kv_gb"])
                for row in requests
                if row["category"] == category
            ),
        }
        for category in CATEGORY_ORDER
    }
    denominators = {
        field: sum(totals[category][field] for category in CATEGORY_ORDER)
        for field in ("actual_demand", "capped_demand", "kv_bytes")
    }

    current_config = index[(ssus[0], "current_refresh8")]["config"]
    best_name = selection["best_static"]
    best_config = index[(ssus[0], best_name)]["config"]

    def config_shares(config):
        cir = config["category_cir_gbps"]
        paths = config["category_paths_per_group"]
        return {
            "cir": [value / sum(cir) for value in cir],
            "paths": [value / sum(paths) for value in paths],
        }

    current = config_shares(current_config)
    best = config_shares(best_config)
    rows = []
    for position, category in enumerate(CATEGORY_ORDER):
        rows.append(
            {
                "category": category,
                "actual_demand_share": totals[category]["actual_demand"]
                / denominators["actual_demand"],
                "capped_demand_share": totals[category]["capped_demand"]
                / denominators["capped_demand"],
                "kv_bytes_share": totals[category]["kv_bytes"]
                / denominators["kv_bytes"],
                "current_cir_share": current["cir"][position],
                "current_path_share": current["paths"][position],
                "best_cir_share": best["cir"][position],
                "best_path_share": best["paths"][position],
            }
        )
    return {
        "workload_reference_ssu": ssus[0],
        "best_static_strategy": best_name,
        "rows": rows,
    }


def gap_analysis(data, selection, ssus):
    index = row_index(data)
    practical_name = selection.get("best_nonideal_practical")
    rows = []
    for ssu in ssus:
        names = {
            "baseline": "baseline",
            "practical": practical_name,
            "per_ssd_edf": "per_ssd_full_visible_edf",
            "global_online": "global_link_aware_online",
            "fluid_upper_bound": "isolated_no_contention_bound",
        }
        values = {}
        for role, name in names.items():
            row = index.get((ssu, name)) if name is not None else None
            values[role] = metric_value(row) if row is not None else None
        entry = {"num_ssu": ssu, "practical_strategy": practical_name}
        entry.update(values)
        if all(values[key] is not None for key in values):
            entry.update(
                {
                    "practical_gain_vs_baseline_pp": 100.0
                    * (values["practical"] - values["baseline"]),
                    "edf_gain_vs_practical_pp": 100.0
                    * (values["per_ssd_edf"] - values["practical"]),
                    "global_gain_vs_practical_pp": 100.0
                    * (values["global_online"] - values["practical"]),
                    "bound_headroom_vs_practical_pp": 100.0
                    * (values["fluid_upper_bound"] - values["practical"]),
                    "bound_headroom_vs_global_pp": 100.0
                    * (values["fluid_upper_bound"] - values["global_online"]),
                }
            )
        rows.append(entry)
    return rows


def target_analysis(data, selection, ssus):
    index = row_index(data)
    request_winner = selection["best_nonideal_practical_request"]
    fleet_winner = selection["best_nonideal_practical_fleet"]

    def evaluate(strategy):
        rows = []
        for ssu in ssus:
            baseline_row = index[(ssu, "baseline")]
            strategy_row = index[(ssu, strategy)]
            request_gain = 100.0 * (
                metric_value(strategy_row) - metric_value(baseline_row)
            )
            fleet_gain = 100.0 * (
                fleet_metric_value(strategy_row) - fleet_metric_value(baseline_row)
            )
            rows.append(
                {
                    "num_ssu": ssu,
                    "request_compute_fraction_gain_pp": request_gain,
                    "fleet_npu_compute_utilization_gain_pp": fleet_gain,
                    "request_target_15pp_met": request_gain >= 15.0,
                    "fleet_target_15pp_met": fleet_gain >= 15.0,
                }
            )
        mean_request = float(
            np.mean([row["request_compute_fraction_gain_pp"] for row in rows])
        )
        mean_fleet = float(
            np.mean([row["fleet_npu_compute_utilization_gain_pp"] for row in rows])
        )
        return {
            "strategy": strategy,
            "rows": rows,
            "mean_request_gain_pp": mean_request,
            "mean_fleet_gain_pp": mean_fleet,
            "mean_request_target_15pp_met": mean_request >= 15.0,
            "mean_fleet_target_15pp_met": mean_fleet >= 15.0,
        }

    request_selected = evaluate(request_winner)
    fleet_selected = evaluate(fleet_winner)
    return {
        # Legacy fields retain the former request-selected strategy contract.
        "strategy": request_selected["strategy"],
        "rows": request_selected["rows"],
        "mean_request_gain_pp": request_selected["mean_request_gain_pp"],
        "mean_fleet_gain_pp": request_selected["mean_fleet_gain_pp"],
        "mean_request_target_15pp_met": request_selected[
            "mean_request_target_15pp_met"
        ],
        "mean_fleet_target_15pp_met": request_selected[
            "mean_fleet_target_15pp_met"
        ],
        "request_selected": request_selected,
        "fleet_selected": fleet_selected,
    }


def max_min_allocate(capacity, demands):
    remaining = list(range(len(demands)))
    allocations = [0.0 for _ in demands]
    remaining_capacity = float(capacity)
    while remaining:
        share = remaining_capacity / len(remaining)
        satisfied = [index for index in remaining if demands[index] <= share]
        if not satisfied:
            for index in remaining:
                allocations[index] = share
            break
        for index in satisfied:
            allocations[index] = float(demands[index])
            remaining_capacity -= float(demands[index])
            remaining.remove(index)
    return allocations


def maxmin_analysis(paired):
    calibration = run_two_npu_calibration()
    formal = paired["comparisons"].get("demand_maxmin", {})
    return {
        "controlled_two_npu_case": {
            **calibration,
            "interpretation": (
                "The real command schedulers were run for 40 equal-sized commands. "
                "Baseline measured 20/20 GB/s and demand max-min measured 10/30."
            ),
        },
        "formal_paired_result": formal,
        "formal_caveat": (
            "The formal model uses discrete non-preemptive commands, per-SSD "
            "placement, 128 NPUs, and a separate 50 GB/s per-NPU receive queue; "
            "the analytic 10/30 allocation is a calibration, not its prediction."
        ),
    }


def build_warnings(data, selection, workload, gaps):
    warnings = []
    runtime = data.get("experiment", {}).get("runtime", {})
    if runtime.get("mode") != "formal":
        warnings.append("Input is not marked as the formal experiment.")
    if selection["selection_ssus"] != list(FORMAL_SSUS):
        warnings.append(
            "Fixed strategy selection did not use the required 40/56/80 SSUs."
        )
    if not workload["arrival_delay_ms"]["in_required_range_0_to_5"]:
        warnings.append("At least one NPU arrival delay is outside [0, 5] ms.")
    if workload["arrival_delay_ms"]["unique_count"] != FORMAL_NUM_NPU:
        warnings.append("The formal workload does not have one unique jitter per NPU.")
    for row in gaps:
        if (
            row.get("fluid_upper_bound") is not None
            and row.get("practical") is not None
            and row["fluid_upper_bound"] + 1e-12 < row["practical"]
        ):
            warnings.append(
                "SSU %d practical result exceeds the declared fluid bound."
                % row["num_ssu"]
            )
    return warnings


def analyze(data, source_path):
    require_complete(data)
    ssus = selection_ssus(data)
    selection = choose_fixed_strategies(data, ssus)
    paired = build_paired_analysis(data, selection, ssus)
    workload = workload_analysis(data, ssus)
    gaps = gap_analysis(data, selection, ssus)
    analysis = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "input_schema_version": data.get("schema_version"),
        "experiment": data.get("experiment", {}),
        "result_complete": bool(data.get("complete", False)),
        "selection": selection,
        "overview": overview_table(data, selection, ssus),
        "pressure_cadence": cadence_analysis(data, ssus),
        "pressure_cadence_paired": cadence_paired_analysis(data, ssus),
        "static_tuning": static_tuning_analysis(data, selection, ssus),
        "paired_causality": paired,
        "category_wait_decomposition": category_wait_analysis(
            data, selection, ssus
        ),
        "workload_and_delay": workload,
        "system_metrics": system_analysis(data, selection, ssus),
        "allocation_alignment": allocation_alignment_analysis(
            data, selection, ssus
        ),
        "practical_ideal_bound_gaps": gaps,
        "target_15pp": target_analysis(data, selection, ssus),
        "batch_and_layer_waits": batch_analysis(data, ssus),
        "maxmin_calibration": maxmin_analysis(paired),
        "metric_contract": {
            "selection_metric": "avg_request_compute_fraction",
            "request_selection_metric": "avg_request_compute_fraction",
            "fleet_selection_metric": "fleet_npu_compute_utilization",
            "selection_reason": (
                "Request pipeline efficiency and fleet utilization have separate "
                "fixed-strategy winners; neither metric is substituted for the other."
            ),
            "fleet_metric": "fleet_npu_compute_utilization",
            "fleet_metric_note": (
                "Total compute divided by 128 times global makespan; heterogeneous "
                "tail requests can make it move opposite to the request mean."
            ),
        },
    }
    analysis["warnings"] = build_warnings(data, selection, workload, gaps)
    return analysis


def style_axis(axis):
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure, path):
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def plot_strategy_overview(analysis, output_path):
    ssus = analysis["selection"]["selection_ssus"]
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    markers = ("o", "s", "^", "D", "P", "X", "v", "*", "<", ">")
    for marker, row in zip(markers, analysis["overview"]):
        values = [row["compute_fraction_by_ssu"].get(str(ssu)) for ssu in ssus]
        if not all(value is not None for value in values):
            continue
        line_style = "--" if row["kind"] == "upper_bound" else "-"
        axes[0].plot(
            ssus,
            100.0 * np.asarray(values),
            marker=marker,
            linewidth=2.0,
            markersize=6,
            linestyle=line_style,
            label=row["label"],
        )
        fleet_values = [
            row["fleet_npu_compute_utilization_by_ssu"].get(str(ssu))
            for ssu in ssus
        ]
        if all(value is not None for value in fleet_values):
            axes[1].plot(
                ssus,
                100.0 * np.asarray(fleet_values),
                marker=marker,
                linewidth=2.0,
                markersize=6,
                label=row["label"],
            )
    axes[0].set_title("Mean per-request pipeline efficiency")
    axes[0].set_ylabel("Mean request compute fraction (%)")
    axes[1].set_title("Fleet utilization over global makespan")
    axes[1].set_ylabel("Fleet NPU compute utilization (%)")
    for axis in axes:
        axis.set_xlabel("Number of SSUs")
        axis.set_xticks(ssus)
        style_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        fontsize=7.5,
    )
    figure.text(
        0.5,
        0.01,
        "128 NPUs, 16 layers, paired workload and placement",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    save_figure(figure, output_path)


def plot_pressure_cadence(analysis, output_path):
    rows = analysis["pressure_cadence"]
    ssus = analysis["selection"]["selection_ssus"]
    names = (
        "current_layer_snapshot",
        "current_refresh8",
        "current_per_io",
    )
    labels = (
        "One snapshot / request / layer / SSD",
        "Refresh every 8",
        "Per-I/O live",
    )
    colors = ("#4c78a8", "#f58518", "#54a24b")
    index = {(row["num_ssu"], row["strategy"]): row for row in rows}
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    for name, label, color in zip(names, labels, colors):
        available = [index.get((ssu, name)) for ssu in ssus]
        if not all(available):
            continue
        axes[0].plot(
            ssus,
            [100.0 * row["compute_fraction"] for row in available],
            marker="o",
            label=label,
            color=color,
        )
        axes[1].plot(
            ssus,
            [row["pressure_telemetry_mb"] for row in available],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_title("Compute result")
    axes[0].set_ylabel("Mean request compute fraction (%)")
    axes[1].set_title("Telemetry traffic (256 counters x 4 bytes/report)")
    axes[1].set_ylabel("Pressure telemetry (MB)")
    axes[1].set_yscale("log")
    paired = analysis["pressure_cadence_paired"]["comparisons"]
    paired_names = ("current_refresh8", "current_per_io")
    paired_labels = ("Refresh8 - snapshot", "Per-I/O - snapshot")
    for name, label, color in zip(paired_names, paired_labels, colors[1:]):
        values = [
            paired[name]["by_ssu"][str(ssu)]["mean_compute_delta_pp"]
            for ssu in ssus
        ]
        axes[2].plot(ssus, values, marker="o", label=label, color=color)
    axes[2].axhline(0.0, color="black", linewidth=0.9)
    axes[2].set_title("Request-ID paired effect")
    axes[2].set_ylabel("Compute-fraction delta vs snapshot (pp)")
    for axis in axes:
        axis.set_xlabel("Number of SSUs")
        axis.set_xticks(ssus)
        style_axis(axis)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    figure.suptitle("Path-pressure read cadence: outcome versus telemetry cost")
    figure.tight_layout()
    save_figure(figure, output_path)


def compact_static_label(row):
    cir = row.get("category_cir_gbps")
    paths = row.get("category_paths_per_group")
    if cir is None or paths is None:
        return row["strategy"]
    cir_text = "/".join("%g" % value for value in cir)
    path_text = "/".join(str(value) for value in paths)
    prefix = "★ " if row.get("selected") else ""
    return "%sCIR %s | paths %s" % (prefix, cir_text, path_text)


def plot_static_tuning(analysis, output_path):
    rows = analysis["static_tuning"]
    ssus = analysis["selection"]["selection_ssus"]
    if not rows:
        return
    matrix = np.asarray(
        [
            [100.0 * row["compute_fraction_by_ssu"][str(ssu)] for ssu in ssus]
            for row in rows
        ]
    )
    labels = [compact_static_label(row) for row in rows]
    height = max(6.5, 0.48 * len(rows) + 1.8)
    figure, axes = plt.subplots(
        1, 2, figsize=(15.5, height), gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    image = axes[0].imshow(matrix, aspect="auto", cmap="viridis")
    axes[0].set_xticks(np.arange(len(ssus)), [str(ssu) for ssu in ssus])
    axes[0].set_yticks(np.arange(len(rows)), labels, fontsize=8)
    axes[0].set_xlabel("Number of SSUs")
    axes[0].set_title("Compute fraction (%)")
    for y in range(len(rows)):
        for x in range(len(ssus)):
            axes[0].text(
                x,
                y,
                "%.2f" % matrix[y, x],
                ha="center",
                va="center",
                color="white" if matrix[y, x] < np.mean(matrix) else "black",
                fontsize=7.5,
            )
    figure.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    deltas = [row["mean_delta_vs_baseline_pp"] for row in rows]
    colors = ["#e45756" if row["selected"] else "#4c78a8" for row in rows]
    axes[1].barh(np.arange(len(rows)), deltas, color=colors)
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set_yticks(np.arange(len(rows)), labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean delta vs baseline (pp), fixed across SSUs")
    axes[1].set_title("Cross-SSU selection score")
    style_axis(axes[1])
    figure.suptitle("Static CIR and path-count tuning (SS/SL/LS/LL order)")
    figure.tight_layout()
    save_figure(figure, output_path)


def aggregates_for_plot(analysis, strategy, field):
    comparison = analysis["paired_causality"]["comparisons"].get(strategy)
    if comparison is None:
        return []
    return comparison["all_ssus"][field]


def ordered_bin_rows(rows, kind):
    mapping = {row["bin"]: row for row in rows}
    if kind == "category":
        labels = [category for category in CATEGORY_ORDER if category in mapping]
    elif kind == "bandwidth":
        labels = [
            "0-10",
            "10-20",
            "20-30",
            "30-40",
            "40-50",
            "50-cap",
            "50+",
        ]
        labels = [label for label in labels if label in mapping]
    else:
        labels = sorted(mapping)
    return labels, [mapping[label] for label in labels]


def plot_paired_input_causality(analysis, output_path):
    current = "current_refresh8"
    best = analysis["selection"].get("best_static")
    strategies = [current]
    if best and best != current:
        strategies.append(best)
    labels = [strategy_label(name, analysis["selection"]) for name in strategies]
    fields = (
        (
            "by_capped_demand_bin",
            "bandwidth",
            "Actual demand bin after 50 GB/s NPU cap",
        ),
        ("by_category", "category", "Workload category"),
        ("by_input_bin", "input", "Exact input bin (seq_len, NQL)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.0))
    width = 0.8 / max(len(strategies), 1)
    for axis, (field, kind, title) in zip(axes, fields):
        first_rows = aggregates_for_plot(analysis, strategies[0], field)
        bin_labels, _ = ordered_bin_rows(first_rows, kind)
        if kind == "input":
            first_by_bin = {row["bin"]: row for row in first_rows}
            bin_labels = sorted(
                bin_labels,
                key=lambda value: abs(
                    first_by_bin[value]["mean_compute_delta_pp"]
                ),
                reverse=True,
            )[:16]
            bin_labels.sort(
                key=lambda value: first_by_bin[value]["mean_compute_delta_pp"]
            )
            y = np.arange(len(bin_labels))
            for position, (strategy, label) in enumerate(zip(strategies, labels)):
                rows = aggregates_for_plot(analysis, strategy, field)
                values_by_bin = {
                    row["bin"]: row["mean_compute_delta_pp"] for row in rows
                }
                values = [
                    values_by_bin.get(bin_label, np.nan)
                    for bin_label in bin_labels
                ]
                offset = (position - (len(strategies) - 1) / 2.0) * width
                axis.barh(y + offset, values, height=width, label=label)
            axis.axvline(0.0, color="black", linewidth=0.9)
            axis.set_yticks(y, bin_labels)
            axis.set_title("16 exact input bins with largest |effect|")
            axis.set_xlabel("Paired compute-fraction delta (pp)")
            axis.grid(True, axis="x", alpha=0.25, linewidth=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            continue
        x = np.arange(len(bin_labels))
        for position, (strategy, label) in enumerate(zip(strategies, labels)):
            rows = aggregates_for_plot(analysis, strategy, field)
            values_by_bin = {
                row["bin"]: row["mean_compute_delta_pp"] for row in rows
            }
            values = [values_by_bin.get(bin_label, np.nan) for bin_label in bin_labels]
            offset = (position - (len(strategies) - 1) / 2.0) * width
            axis.bar(x + offset, values, width=width, label=label)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_xticks(x, bin_labels)
        axis.set_title(title)
        axis.set_ylabel("Paired compute-fraction delta (pp)")
        style_axis(axis)
    axes[0].legend(fontsize=9)
    figure.suptitle(
        "Request-ID paired causality: static QoS relative to the same baseline inputs"
    )
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_category_wait_decomposition(analysis, output_path):
    rows = analysis["category_wait_decomposition"]
    strategies = []
    for row in rows:
        if row["strategy"] not in strategies:
            strategies.append(row["strategy"])
    ssus = analysis["selection"]["selection_ssus"]
    figure, axes = plt.subplots(len(ssus), 3, figsize=(15.5, 4.0 * len(ssus)), squeeze=False)
    width = 0.8 / max(len(strategies), 1)
    fields = (
        ("ssd_queue_delta_ms_per_io", "SSD queue delta (ms / I/O)"),
        ("npu_queue_delta_ms_per_io", "NPU receive-queue delta (ms / I/O)"),
        ("compute_delta_pp", "Compute-fraction delta (pp)"),
    )
    for row_index_, ssu in enumerate(ssus):
        for column, (field, title) in enumerate(fields):
            axis = axes[row_index_, column]
            x = np.arange(len(CATEGORY_ORDER))
            for position, strategy in enumerate(strategies):
                mapping = {
                    row["category"]: row[field]
                    for row in rows
                    if row["num_ssu"] == ssu and row["strategy"] == strategy
                }
                values = [mapping.get(category, np.nan) for category in CATEGORY_ORDER]
                offset = (position - (len(strategies) - 1) / 2.0) * width
                axis.bar(
                    x + offset,
                    values,
                    width=width,
                    label=strategy_label(strategy, analysis["selection"]),
                )
            axis.axhline(0.0, color="black", linewidth=0.9)
            axis.set_xticks(x, CATEGORY_ORDER)
            axis.set_title("%d SSUs: %s" % (ssu, title))
            style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(strategies),
        fontsize=7.5,
    )
    figure.suptitle(
        "Category stage diagnostics versus baseline (queue means are non-additive)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    save_figure(figure, output_path)


def plot_workload_delay(analysis, output_path):
    requests = analysis["workload_and_delay"]["requests"]
    delays = [row["arrival_delay_ms"] for row in requests]
    bandwidth = [row["required_bw_gbps"] for row in requests]
    actual_demand = [row["actual_layer_demand_gbps"] for row in requests]
    capped_demand = [row["capped_layer_demand_gbps"] for row in requests]
    kv = [row["per_layer_kv_gb"] for row in requests]
    compute = [row["per_layer_compute_ms"] for row in requests]
    colors = {
        "SS": "#4c78a8",
        "SL": "#f58518",
        "LS": "#54a24b",
        "LL": "#e45756",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    axes[0, 0].hist(delays, bins=np.linspace(0.0, 5.0, 11), color="#4c78a8")
    axes[0, 0].set_title("Independent NPU start jitter")
    axes[0, 0].set_xlabel("Arrival delay (ms)")
    axes[0, 0].set_ylabel("Requests")
    axes[0, 1].hist(bandwidth, bins=16, alpha=0.35, label="Input table BW")
    axes[0, 1].hist(
        actual_demand, bins=16, alpha=0.35, label="Actual KV / compute"
    )
    axes[0, 1].hist(
        capped_demand, bins=16, alpha=0.35, label="Demand after NPU cap"
    )
    axes[0, 1].axvline(50.0, color="black", linestyle="--", label="NPU 50 GB/s")
    axes[0, 1].set_title("Input, actual, and capped layer demand")
    axes[0, 1].set_xlabel("Required bandwidth (GB/s)")
    axes[0, 1].legend(fontsize=9)
    for category in CATEGORY_ORDER:
        selected = [row for row in requests if row["category"] == category]
        axes[1, 0].scatter(
            [row["per_layer_compute_ms"] for row in selected],
            [row["per_layer_kv_gb"] for row in selected],
            label=category,
            color=colors[category],
            alpha=0.75,
            s=28,
        )
    axes[1, 0].set_title("Workload shape by category")
    axes[1, 0].set_xlabel("Compute time per layer (ms)")
    axes[1, 0].set_ylabel("KV read per layer (GB)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].hist(compute, bins=16, alpha=0.7, label="Compute ms")
    twin = axes[1, 1].twinx()
    twin.hist(kv, bins=16, alpha=0.35, color="#e45756", label="KV GB")
    axes[1, 1].set_title("Marginal workload distributions")
    axes[1, 1].set_xlabel("Value (different units; see legends)")
    axes[1, 1].set_ylabel("Compute-time count")
    twin.set_ylabel("KV-size count")
    handles_left, labels_left = axes[1, 1].get_legend_handles_labels()
    handles_right, labels_right = twin.get_legend_handles_labels()
    axes[1, 1].legend(
        handles_left + handles_right, labels_left + labels_right, fontsize=9
    )
    for axis in axes.flat:
        style_axis(axis)
    figure.suptitle("Paired formal workload and 0-5 ms NPU arrival delay")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_gap(analysis, output_path):
    rows = analysis["practical_ideal_bound_gaps"]
    ssus = [row["num_ssu"] for row in rows]
    practical_name = analysis["selection"]["best_nonideal_practical_request"]
    roles = (
        ("baseline", "Baseline", "#9d9da1", "-"),
        (
            "practical",
            strategy_label(practical_name, analysis["selection"]),
            "#4c78a8",
            "-",
        ),
        ("per_ssd_edf", "Per-SSD EDF heuristic", "#f58518", "-"),
        ("global_online", "Global link-aware online", "#b279a2", "-"),
        ("fluid_upper_bound", "Fluid upper bound", "#54a24b", "--"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for field, label, color, line_style in roles:
        values = [row.get(field) for row in rows]
        if all(value is not None for value in values):
            axes[0].plot(
                ssus,
                100.0 * np.asarray(values),
                marker="o",
                label=label,
                color=color,
                linestyle=line_style,
            )
    axes[0].set_title("Achieved compute fraction")
    axes[0].set_xlabel("Number of SSUs")
    axes[0].set_ylabel("Mean request compute fraction (%)")
    axes[0].set_xticks(ssus)
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    gap_fields = (
        ("practical_gain_vs_baseline_pp", "Practical - baseline"),
        ("edf_gain_vs_practical_pp", "EDF - practical"),
        ("global_gain_vs_practical_pp", "Global - practical"),
        ("bound_headroom_vs_practical_pp", "Bound - practical"),
    )
    x = np.arange(len(ssus))
    width = 0.19
    for position, (field, label) in enumerate(gap_fields):
        values = [row.get(field, np.nan) for row in rows]
        offset = (position - (len(gap_fields) - 1) / 2.0) * width
        axes[1].bar(x + offset, values, width=width, label=label)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_xticks(x, [str(ssu) for ssu in ssus])
    axes[1].set_xlabel("Number of SSUs")
    axes[1].set_ylabel("Gap (percentage points)")
    axes[1].set_title("Practical, idealized heuristic, and relaxation gaps")
    axes[1].legend(fontsize=8)
    for axis in axes:
        style_axis(axis)
    figure.legend(
        line_handles,
        line_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(line_handles),
        fontsize=7.5,
    )
    figure.suptitle("Achieved performance versus idealized scheduling and fluid bound")
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    save_figure(figure, output_path)


def plot_system_metrics(analysis, output_path):
    rows = analysis["system_metrics"]
    ssus = analysis["selection"]["selection_ssus"]
    names = []
    for row in rows:
        if row["strategy"] not in names:
            names.append(row["strategy"])
    index = {(row["num_ssu"], row["strategy"]): row for row in rows}
    panels = (
        (
            "ssu_effective_bandwidth_utilization",
            "Mean SSU effective bandwidth utilization",
            "Utilization (%)",
            100.0,
        ),
        ("npu_link_utilization", "NPU receive-link utilization", "Utilization (%)", 100.0),
        (
            "avg_npu_link_queue_wait_ms",
            "NPU incast queueing",
            "Mean queue wait (ms / I/O)",
            1.0,
        ),
        ("makespan_ms", "Global completion tail", "Makespan (ms)", 1.0),
        (
            "fleet_npu_compute_utilization",
            "Fleet NPU compute utilization",
            "Utilization (%)",
            100.0,
        ),
        (
            "request_compute_fraction_jain",
            "Per-request compute-fraction fairness",
            "Jain index",
            1.0,
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))
    for axis, (field, title, ylabel, scale) in zip(axes.flat, panels):
        for name in names:
            values = [scale * index[(ssu, name)][field] for ssu in ssus]
            axis.plot(
                ssus,
                values,
                marker="o",
                label=strategy_label(name, analysis["selection"]),
            )
        axis.set_title(title)
        axis.set_xlabel("Number of SSUs")
        axis.set_ylabel(ylabel)
        axis.set_xticks(ssus)
        style_axis(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        fontsize=7.0,
    )
    figure.suptitle("System utilization, NPU incast, tail time, and fairness")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    save_figure(figure, output_path)


def plot_allocation_alignment(analysis, output_path):
    details = analysis["allocation_alignment"]
    rows = details["rows"]
    x = np.arange(len(CATEGORY_ORDER))
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    workload_fields = (
        ("actual_demand_share", "Actual demand"),
        ("capped_demand_share", "Demand after NPU cap"),
        ("kv_bytes_share", "KV bytes"),
    )
    width = 0.8 / len(workload_fields)
    for position, (field, label) in enumerate(workload_fields):
        offset = (position - 1) * width
        axes[0].bar(
            x + offset,
            [100.0 * row[field] for row in rows],
            width=width,
            label=label,
        )
    axes[0].set_title("Formal workload shares")

    allocation_fields = (
        ("current_cir_share", "Current CIR"),
        ("current_path_share", "Current paths"),
        ("best_cir_share", "Best static CIR"),
        ("best_path_share", "Best static paths"),
    )
    width = 0.8 / len(allocation_fields)
    for position, (field, label) in enumerate(allocation_fields):
        offset = (position - (len(allocation_fields) - 1) / 2.0) * width
        axes[1].bar(
            x + offset,
            [100.0 * row[field] for row in rows],
            width=width,
            label=label,
        )
    axes[1].set_title(
        "Static allocation shares: %s"
        % strategy_label(
            details["best_static_strategy"], analysis["selection"]
        )
    )
    for axis in axes:
        axis.set_xticks(x, CATEGORY_ORDER)
        axis.set_xlabel("Workload category")
        axis.set_ylabel("Share (%)")
        axis.legend(fontsize=8)
        style_axis(axis)
    figure.suptitle("Does CIR/path allocation follow offered demand? (SS/SL/LS/LL)")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_batch_layer_waits(analysis, output_path):
    details = analysis["batch_and_layer_waits"]
    rows = details["rows"]
    ssus = analysis["selection"]["selection_ssus"]
    batch_sizes = (8, 16, 32)
    index = {(row["num_ssu"], row["batch_size"]): row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 5.7))
    for ssu in ssus:
        available = [index.get((ssu, batch)) for batch in batch_sizes]
        if all(available):
            axes[0].plot(
                batch_sizes,
                [100.0 * row["compute_fraction"] for row in available],
                marker="o",
                label="%d SSUs" % ssu,
            )
    axes[0].set_title("Submission batch size")
    axes[0].set_xlabel("I/Os submitted per client batch")
    axes[0].set_ylabel("Mean request compute fraction (%)")
    axes[0].set_xticks(batch_sizes)
    axes[0].legend(fontsize=9)
    labels = []
    l0_values = []
    l1_values = []
    l2_values = []
    for ssu in ssus:
        for batch in batch_sizes:
            row = index.get((ssu, batch))
            if row is None:
                continue
            labels.append("%d/%d" % (ssu, batch))
            l0_values.append(row["avg_wait_L0_ms"])
            l1_values.append(row["avg_wait_L1_ms"])
            l2_values.append(row["avg_wait_L2plus_ms"])
    x = np.arange(len(labels))
    axes[1].bar(x, l0_values, label="L0 wait")
    axes[1].bar(x, l1_values, bottom=l0_values, label="L1 wait")
    axes[1].bar(
        x,
        l2_values,
        bottom=np.asarray(l0_values) + np.asarray(l1_values),
        label="L2+ wait (14 layers)",
    )
    axes[1].set_xticks(x, labels, rotation=45)
    axes[1].set_xlabel("SSUs / batch size")
    axes[1].set_ylabel("Mean request I/O stall (ms)")
    axes[1].set_title("Layer wait composition at 16 layers")
    axes[1].legend(fontsize=9)
    for axis in axes:
        style_axis(axis)
    figure.suptitle("Batch sensitivity and startup versus steady-state layer waits")
    figure.tight_layout()
    save_figure(figure, output_path)


def plot_maxmin(analysis, output_path):
    calibration = analysis["maxmin_calibration"]["controlled_two_npu_case"]
    baseline = calibration["baseline"]["achieved_ssd_service_gbps"]
    aware = calibration["demand_maxmin"]["achieved_ssd_service_gbps"]
    demands = calibration["npu_demands_gbps"]
    paired = analysis["paired_causality"]["paired_requests"]
    demand_pairs = [row for row in paired if row["strategy"] == "demand_maxmin"]
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))
    x = np.arange(2)
    width = 0.26
    axes[0].bar(x - width, demands, width=width, label="Demand")
    axes[0].bar(x, baseline, width=width, label="Equal active share")
    axes[0].bar(x + width, aware, width=width, label="Demand-aware max-min")
    axes[0].set_xticks(x, ["NPU A", "NPU B"])
    axes[0].set_ylabel("Bandwidth (GB/s)")
    axes[0].set_title("Measured 10/30 calibration using the real SSD schedulers")
    axes[0].legend(fontsize=8)
    category_markers = {"SS": "o", "SL": "s", "LS": "^", "LL": "D"}
    ssu_colors = {40: "#4c78a8", 56: "#f58518", 80: "#54a24b"}
    for ssu in sorted({row["num_ssu"] for row in demand_pairs}):
        for category in CATEGORY_ORDER:
            selected = [
                row
                for row in demand_pairs
                if row["num_ssu"] == ssu and row["category"] == category
            ]
            if not selected:
                continue
            axes[1].scatter(
                [row["capped_layer_demand_gbps"] for row in selected],
                [row["compute_delta_pp"] for row in selected],
                color=ssu_colors.get(ssu),
                marker=category_markers[category],
                alpha=0.65,
                s=28,
                label="%d SSU / %s" % (ssu, category),
            )
    if demand_pairs:
        x_values = np.asarray(
            [row["capped_layer_demand_gbps"] for row in demand_pairs]
        )
        y_values = np.asarray([row["compute_delta_pp"] for row in demand_pairs])
        bin_edges = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 9)
        centers = []
        means = []
        for bin_index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if bin_index == len(bin_edges) - 2:
                mask = (x_values >= low) & (x_values <= high)
            else:
                mask = (x_values >= low) & (x_values < high)
            if np.any(mask):
                centers.append((low + high) / 2.0)
                means.append(float(np.mean(y_values[mask])))
        axes[1].plot(centers, means, color="black", linewidth=2.2, label="Binned mean")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axvline(10.0, color="gray", linestyle=":", linewidth=0.8)
    axes[1].axvline(30.0, color="gray", linestyle=":", linewidth=0.8)
    axes[1].set_xlabel("Actual layer demand after 50 GB/s NPU cap")
    axes[1].set_ylabel("Demand max-min delta vs baseline (pp)")
    axes[1].set_title("Formal request-ID paired demand-aware result")
    axes[1].legend(ncol=2, fontsize=6.8)
    for axis in axes:
        style_axis(axis)
    figure.suptitle("From the 10/30 allocation mechanism to the full discrete simulation")
    figure.tight_layout()
    save_figure(figure, output_path)


def render_plots(analysis, output_dir):
    figures = {
        "strategy_overview": "01_strategy_overview.png",
        "pressure_cadence": "02_pressure_cadence.png",
        "static_tuning": "03_static_cir_path_tuning.png",
        "paired_input_causality": "04_paired_input_causality.png",
        "category_wait_decomposition": "05_category_wait_decomposition.png",
        "delay_workload": "06_delay_workload_distribution.png",
        "practical_ideal_bound_gap": "07_practical_ideal_bound_gap.png",
        "batch_layer_waits": "08_batch_layer_wait_composition.png",
        "maxmin_calibration": "09_maxmin_calibration.png",
        "system_metrics": "10_system_utilization_and_incast.png",
        "allocation_alignment": "11_cir_path_demand_alignment.png",
    }
    plotters = {
        "strategy_overview": plot_strategy_overview,
        "pressure_cadence": plot_pressure_cadence,
        "static_tuning": plot_static_tuning,
        "paired_input_causality": plot_paired_input_causality,
        "category_wait_decomposition": plot_category_wait_decomposition,
        "delay_workload": plot_workload_delay,
        "practical_ideal_bound_gap": plot_gap,
        "batch_layer_waits": plot_batch_layer_waits,
        "maxmin_calibration": plot_maxmin,
        "system_metrics": plot_system_metrics,
        "allocation_alignment": plot_allocation_alignment,
    }
    rendered = {}
    for key, filename in figures.items():
        path = output_dir / filename
        if path.exists():
            path.unlink()
        plotters[key](analysis, path)
        if path.exists():
            rendered[key] = filename
    return rendered


def format_percent(value):
    if value is None:
        return "n/a"
    return "%.2f%%" % (100.0 * value)


def format_pp(value):
    if value is None:
        return "n/a"
    return "%+.2f pp" % value


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def causal_sentence(strategy, comparison):
    overall = comparison["all_ssus"]
    direction = "improves" if overall["mean_compute_delta_pp"] >= 0.0 else "reduces"
    dominant = sorted(
        overall["by_category"],
        key=lambda row: abs(row["mean_compute_delta_pp"]),
        reverse=True,
    )[0]
    wait_direction = (
        "less" if overall["mean_io_wait_delta_ms"] < 0.0 else "more"
    )
    return (
        "`%s` %s mean compute fraction by %.2f pp. The largest category "
        "effect is %s (%+.2f pp); paired requests see %.2f ms %s total I/O "
        "stall on average. The compute delta versus stall delta correlation is "
        "%.3f (fixed compute work makes this relationship mechanically strong)."
        % (
            strategy,
            direction,
            abs(overall["mean_compute_delta_pp"]),
            dominant["bin"],
            dominant["mean_compute_delta_pp"],
            abs(overall["mean_io_wait_delta_ms"]),
            wait_direction,
            overall["correlation_delta_vs_io_wait_delta"],
        )
    )


def top_input_table(rows):
    return markdown_table(
        [
            "SSU",
            "request",
            "cat",
            "input BW",
            "actual/capped BW",
            "seq/NQL",
            "delta",
            "wait delta",
        ],
        [
            [
                row["num_ssu"],
                row["request_id"],
                row["category"],
                "%.2f" % row["required_bw_gbps"],
                "%.2f/%.2f"
                % (
                    row["actual_layer_demand_gbps"],
                    row["capped_layer_demand_gbps"],
                ),
                "%g/%g" % (row["seq_len_k"], row["nql"]),
                format_pp(row["compute_delta_pp"]),
                "%+.2f ms" % row["io_wait_delta_ms"],
            ]
            for row in rows[:8]
        ],
    )


def build_report(analysis):
    selection = analysis["selection"]
    ssus = selection["selection_ssus"]
    lines = [
        "# QoS storage simulation analysis",
        "",
        "Generated from `%s`." % analysis["source"],
        "",
        "## Experiment and selection contract",
        "",
        "The analyzed SSUs are **%s**. The formal contract is 128 NPUs, 16 "
        "layers, paired workload/placement, and one independent 0-5 ms arrival "
        "delay per NPU." % "/".join(str(ssu) for ssu in ssus),
        "",
        selection["rule"],
        "",
        "Selected fixed static strategy: `%s`. Selected fixed ticket strategy: `%s`. "
        "Best practical by request metric: `%s`; best practical by fleet metric: `%s`."
        % (
            selection.get("best_static"),
            selection.get("best_ticket"),
            selection.get("best_nonideal_practical_request"),
            selection.get("best_nonideal_practical_fleet"),
        ),
        "",
        "All tuning and winner reporting is **in-sample**: the same fixed seed is "
        "used to select and evaluate configurations. It is a controlled paired "
        "comparison, not an independent-seed generalization claim.",
        "",
    ]
    if analysis["warnings"]:
        lines.extend(
            ["Warnings: " + "; ".join(analysis["warnings"]), ""]
        )
    overview_rows = []
    for row in analysis["overview"]:
        overview_rows.append(
            [
                row["label"],
                *[
                    "%s / %s"
                    % (
                        format_percent(
                            row["compute_fraction_by_ssu"].get(str(ssu))
                        ),
                        format_percent(
                            row["fleet_npu_compute_utilization_by_ssu"].get(
                                str(ssu)
                            )
                        ),
                    )
                    for ssu in ssus
                ],
                "%s / %s"
                % (
                    format_percent(row["mean_compute_fraction"]),
                    format_percent(
                        row["mean_fleet_npu_compute_utilization"]
                    ),
                ),
            ]
        )
    lines.extend(
        [
            "## Main outcome",
            "",
            markdown_table(
                ["Strategy"]
                + ["%d SSU request/fleet" % ssu for ssu in ssus]
                + ["Mean request/fleet"],
                overview_rows,
            ),
            "",
            "Every cell reports request compute fraction first and fleet NPU "
            "utilization second. Separate winners are selected for the two metrics; "
            "they may differ because fleet utilization includes the global tail "
            "makespan.",
            "",
        ]
    )
    comparisons = analysis["paired_causality"]["comparisons"]
    lines.extend(["## Why requests became faster or slower", ""])
    causal_names = (
        "current_refresh8",
        selection.get("best_static"),
        "demand_maxmin",
        selection.get("best_ticket"),
        selection.get("best_nonideal_practical_request"),
        selection.get("best_nonideal_practical_fleet"),
    )
    for position, strategy in enumerate(causal_names):
        if strategy in causal_names[:position]:
            continue
        if strategy in comparisons:
            lines.extend([causal_sentence(strategy, comparisons[strategy]), ""])
    current = comparisons.get("current_refresh8")
    if current:
        category_rows = current["all_ssus"]["by_category"]
        bandwidth_rows = current["all_ssus"]["by_capped_demand_bin"]
        lines.extend(
            [
                "### Current static CIR: exact paired input groups",
                "",
                markdown_table(
                    ["Category", "N", "compute delta", "improved", "I/O stall delta"],
                    [
                        [
                            row["bin"],
                            row["count"],
                            format_pp(row["mean_compute_delta_pp"]),
                            "%.1f%%" % (100.0 * row["improved_fraction"]),
                            "%+.2f ms" % row["mean_io_wait_delta_ms"],
                        ]
                        for row in category_rows
                    ],
                ),
                "",
                markdown_table(
                    ["Capped actual demand", "N", "compute delta", "improved", "I/O stall delta"],
                    [
                        [
                            row["bin"],
                            row["count"],
                            format_pp(row["mean_compute_delta_pp"]),
                            "%.1f%%" % (100.0 * row["improved_fraction"]),
                            "%+.2f ms" % row["mean_io_wait_delta_ms"],
                        ]
                        for row in bandwidth_rows
                    ],
                ),
                "",
                "Fastest paired inputs:",
                "",
                top_input_table(current["all_ssus"]["fastest_inputs"]),
                "",
                "Slowest paired inputs:",
                "",
                top_input_table(current["all_ssus"]["slowest_inputs"]),
                "",
            ]
        )
    target = analysis["target_15pp"]
    request_target = target["request_selected"]
    fleet_target = target["fleet_selected"]
    lines.extend(
        [
            "## 15 percentage-point target",
            "",
            "The request-metric winner `%s` gains %+.2f pp in mean request "
            "compute fraction; its 15 pp target is **%s**. The independently "
            "selected fleet-metric winner `%s` gains %+.2f pp in fleet NPU "
            "compute utilization; its 15 pp target is **%s**."
            % (
                request_target["strategy"],
                request_target["mean_request_gain_pp"],
                request_target["mean_request_target_15pp_met"],
                fleet_target["strategy"],
                fleet_target["mean_fleet_gain_pp"],
                fleet_target["mean_fleet_target_15pp_met"],
            ),
            "",
        ]
    )
    cadence = analysis["pressure_cadence"]
    lines.extend(["## Pressure read cadence", ""])
    cadence_table = []
    for ssu in ssus:
        for name in (
            "current_layer_snapshot",
            "current_refresh8",
            "current_per_io",
        ):
            row = next(
                (item for item in cadence if item["num_ssu"] == ssu and item["strategy"] == name),
                None,
            )
            if row:
                cadence_table.append(
                    [
                        ssu,
                        name,
                        format_percent(row["compute_fraction"]),
                        row["pressure_reports"],
                        "%.2f" % row["pressure_telemetry_mb"],
                    ]
                )
    lines.extend(
        [
            markdown_table(
                ["SSU", "mode", "compute", "reports", "telemetry MB"],
                cadence_table,
            ),
            "",
        ]
    )
    cadence_paired = analysis["pressure_cadence_paired"]["comparisons"]
    lines.extend(
        [
            markdown_table(
                ["Mode vs snapshot", "mean paired delta", "improved", "slower"],
                [
                    [
                        name,
                        format_pp(details["all_ssus"]["mean_compute_delta_pp"]),
                        details["all_ssus"]["improved_count"],
                        details["all_ssus"]["slower_count"],
                    ]
                    for name, details in cadence_paired.items()
                ],
            ),
            "",
            "Snapshot means one pressure read per request/layer/SSD, not one global "
            "read per layer. Aggregate equality is not treated as equivalence unless "
            "the request-ID paired deltas are also zero; telemetry cost is reported "
            "separately.",
            "",
        ]
    )
    tuning_rows = analysis["static_tuning"]
    lines.extend(["## Static CIR/path allocation", ""])
    lines.append(
        markdown_table(
            ["Selected", "CIR SS/SL/LS/LL", "paths/group", "mean delta vs baseline"],
            [
                [
                    "yes" if row["selected"] else "no",
                    "/".join("%g" % value for value in row["category_cir_gbps"]),
                    "/".join(str(value) for value in row["category_paths_per_group"]),
                    format_pp(row["mean_delta_vs_baseline_pp"]),
                ]
                for row in tuning_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "The fixed winner is selected on all SSUs together, not separately per "
            "SSU. CIR reserves category bandwidth while path count controls admission/"
            "sharing granularity; comparing one-factor CIR-only and path-only rows in "
            "the table separates these effects.",
            "",
        ]
    )
    allocation = analysis["allocation_alignment"]
    lines.extend(
        [
            markdown_table(
                [
                    "Category",
                    "actual demand",
                    "capped demand",
                    "KV bytes",
                    "current CIR/paths",
                    "best CIR/paths",
                ],
                [
                    [
                        row["category"],
                        format_percent(row["actual_demand_share"]),
                        format_percent(row["capped_demand_share"]),
                        format_percent(row["kv_bytes_share"]),
                        "%s / %s"
                        % (
                            format_percent(row["current_cir_share"]),
                            format_percent(row["current_path_share"]),
                        ),
                        "%s / %s"
                        % (
                            format_percent(row["best_cir_share"]),
                            format_percent(row["best_path_share"]),
                        ),
                    ]
                    for row in allocation["rows"]
                ],
            ),
            "",
            "Actual demand is KV bytes divided by compute time; capped demand applies "
            "the 50 GB/s NPU limit before comparing the workload with CIR/path shares.",
            "",
            "## Batch size and layer-count inference",
            "",
        ]
    )
    batch = analysis["batch_and_layer_waits"]
    lines.append(
        markdown_table(
            ["Batch", "cross-SSU mean compute"],
            [
                [row["batch_size"], format_percent(row["mean_compute_fraction"])]
                for row in batch["fixed_batch_scores"]
            ],
        )
    )
    directions = defaultdict(list)
    for row in batch["rows"]:
        if row["batch_size"] == 8:
            directions[row["more_layers_extrapolated_direction"]].append(row["num_ssu"])
    lines.extend(
        [
            "",
            "Best one-size batch: **%s**. At batch 8, the layer-16 steady-state "
            "extrapolation predicts improvement for SSUs %s and degradation for SSUs "
            "%s if layers increase. This is an extrapolation, because every formal "
            "simulation remains fixed at 16 layers."
            % (
                batch["best_fixed_batch_size"],
                directions.get("improve", []),
                directions.get("degrade", []),
            ),
            "",
        ]
    )
    system_rows = analysis["system_metrics"]
    lines.extend(
        [
            "## System utilization, incast, tail, and fairness",
            "",
            markdown_table(
                [
                    "SSU",
                    "strategy",
                    "SSU effective",
                    "NPU link",
                    "NPU queue",
                    "makespan",
                    "Jain",
                ],
                [
                    [
                        row["num_ssu"],
                        row["strategy"],
                        format_percent(row["ssu_effective_bandwidth_utilization"]),
                        format_percent(row["npu_link_utilization"]),
                        "%.4f ms" % row["avg_npu_link_queue_wait_ms"],
                        "%.2f ms" % row["makespan_ms"],
                        "%.4f" % row["request_compute_fraction_jain"],
                    ]
                    for row in system_rows
                ],
            ),
            "",
            "SSU utilization shows whether extra scheduling actually keeps media busy; "
            "NPU receive-queue wait isolates downstream incast. A higher request mean "
            "can coexist with a longer makespan or lower fleet utilization.",
            "",
            "## Demand-aware calibration and theoretical headroom",
            "",
            "For two active NPUs demanding 10 and 30 GB/s from one 40 GB/s SSD, "
            "the actual command schedulers measure 20/20 GB/s for baseline and "
            "10/30 GB/s for demand max-min. Formal demand plots use actual KV/compute "
            "demand after the 50 GB/s NPU cap; placement, 128 NPUs, and receive queues "
            "explain why the controlled gain can be diluted.",
            "",
        ]
    )
    gap_rows = []
    for row in analysis["practical_ideal_bound_gaps"]:
        gap_rows.append(
            [
                row["num_ssu"],
                format_pp(row.get("practical_gain_vs_baseline_pp")),
                format_pp(row.get("edf_gain_vs_practical_pp")),
                format_pp(row.get("global_gain_vs_practical_pp")),
                format_pp(row.get("bound_headroom_vs_practical_pp")),
            ]
        )
    lines.extend(
        [
            markdown_table(
                [
                    "SSU",
                    "practical-baseline",
                    "per-SSD EDF-practical",
                    "global-practical",
                    "bound-practical",
                ],
                gap_rows,
            ),
            "",
            "Per-SSD EDF is an independent-disk heuristic. Global link-aware online "
            "coordinates committed cross-SSD work but still sees no future layer. "
            "Both idealized heuristics receive a full current layer in one submission, "
            "whereas the selected practical winner uses batch 8, so their information advantage "
            "must not be attributed only to the arbitration rule. Neither proves "
            "optimality. The fluid result is a loose, infeasible upper bound, not the "
            "unknown exact optimum: best runnable <= unknown optimum <= loose fluid bound.",
            "",
            "## Arrival delay verification",
            "",
        ]
    )
    delay = analysis["workload_and_delay"]["arrival_delay_ms"]
    lines.extend(
        [
            "N=%d, min=%.4f ms, mean=%.4f ms, p95=%.4f ms, max=%.4f ms, "
            "unique=%d. Range check [0,5] ms: **%s**."
            % (
                analysis["workload_and_delay"]["request_count"],
                delay["min"],
                delay["mean"],
                delay["p95"],
                delay["max"],
                delay["unique_count"],
                delay["in_required_range_0_to_5"],
            ),
            "",
            "These values are independent release/start jitters. They stagger when an "
            "NPU request enters the simulation and are not added to processing TTFT or "
            "request compute fraction.",
            "",
            "## Figures",
            "",
        ]
    )
    for title, filename in analysis.get("figures", {}).items():
        lines.extend(["### " + title.replace("_", " ").title(), "", "![](%s)" % filename, ""])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.input)
    require_complete(data)
    analysis = analyze(data, args.input)
    analysis["figures"] = render_plots(analysis, args.output_dir)
    write_json(args.output_dir / "analysis.json", analysis)
    (args.output_dir / "report.md").write_text(build_report(analysis))
    print("analysis: %s" % (args.output_dir / "analysis.json"))
    print("report:   %s" % (args.output_dir / "report.md"))
    print("figures:  %d" % len(analysis["figures"]))
    print("best fixed static: %s" % analysis["selection"]["best_static"])


if __name__ == "__main__":
    main()
