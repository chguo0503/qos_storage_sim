"""Compare partial/full 128-NPU admission runs with the 32-NPU controls.

The 128-NPU runner checkpoints atomically after every completed case.  This
analyzer therefore treats a missing input, an incomplete matrix, and rows that
have not yet acquired a ``steady_summary`` as normal states.  It never imports
the experiment runner or simulator, so running it cannot change an experiment
fingerprint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_128_INPUT = (
    ROOT / "results" / "steady_state_128npu_admission_v1" / "results.json"
)
DEFAULT_128_V2_INPUT = (
    ROOT / "results" / "steady_state_128npu_admission_v2" / "results.json"
)
DEFAULT_128_SUPPLEMENTAL_INPUT = (
    ROOT / "results" / "steady_state_128npu_admission_ssu24_pair" / "results.json"
)
DEFAULT_128_ADAPTIVE_INPUT = (
    ROOT / "results" / "steady_state_128npu_adaptive_v2_1" / "results.json"
)
DEFAULT_32_PRIMARY = (
    ROOT / "results" / "steady_state_32npu_normalized_slo" / "results.json"
)
DEFAULT_32_ADMISSION = DEFAULT_32_PRIMARY.with_name(
    "admission_interval_ablation.json"
)
DEFAULT_32_BRACKET = DEFAULT_32_PRIMARY.with_name("ratio70_bracket.json")
DEFAULT_32_ADAPTIVE = (
    ROOT / "results" / "steady_state_32npu_adaptive_v2_1" / "results.json"
)
DEFAULT_ANALYSIS = DEFAULT_128_INPUT.with_name("scale_analysis.json")
DEFAULT_REPORT = DEFAULT_128_INPUT.with_name("scale_analysis.md")

SCHEMA_VERSION = 1
STRATEGIES = (
    "baseline",
    "current_scheme_b",
    "admission_25ms",
    "admission_50ms",
    "admission_v2_25ms",
    "adaptive_v2_1_25ms",
)
V1_STRATEGIES = (
    "baseline",
    "current_scheme_b",
    "admission_25ms",
    "admission_50ms",
)
V1_SUPPLEMENTAL_STRATEGIES = ("baseline", "current_scheme_b")
V2_STRATEGIES = ("admission_v2_25ms",)
ADAPTIVE_STRATEGIES = ("adaptive_v2_1_25ms",)
CATEGORIES = ("SS", "SL", "LS", "LL")
PAIRING_FIELDS = (
    "assignment_hash",
    "workload_hash",
    "placement_hash",
    "trace_hash",
    "simulator_input_fingerprint",
)

# Direction labels intentionally have a dead band: sub-percentage-point
# differences should not be presented as a scalable policy effect.
UTIL_DIRECTION_TOLERANCE = 0.005
SLO_DIRECTION_TOLERANCE = 0.01


def _percentile(values: Iterable[float], percentile: float):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_mean(values):
    values = [float(value) for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _id_bias_metrics(id_values):
    """Describe dispersion and monotonic NPU-id bias without assuming causality."""
    ordered = sorted((int(identity), float(value)) for identity, value in id_values)
    if not ordered:
        return {"available": False}
    identities = [identity for identity, _ in ordered]
    values = [value for _, value in ordered]
    mean = statistics.fmean(values)
    quartile_count = max(1, len(values) // 4)
    first = statistics.fmean(values[:quartile_count])
    last = statistics.fmean(values[-quartile_count:])
    if len(values) >= 2 and statistics.pstdev(identities) > 0 and statistics.pstdev(values) > 0:
        id_mean = statistics.fmean(identities)
        covariance = statistics.fmean(
            (identity - id_mean) * (value - mean)
            for identity, value in ordered
        )
        correlation = covariance / (
            statistics.pstdev(identities) * statistics.pstdev(values)
        )
    else:
        correlation = 0.0
    squared_sum = sum(value * value for value in values)
    return {
        "available": True,
        "count": len(values),
        "minimum": min(values),
        "p10": _percentile(values, 10),
        "mean": mean,
        "p90": _percentile(values, 90),
        "maximum": max(values),
        "p90_minus_p10": _percentile(values, 90) - _percentile(values, 10),
        "max_minus_min": max(values) - min(values),
        "first_id_quartile_mean": first,
        "last_id_quartile_mean": last,
        "first_minus_last_id_quartile": first - last,
        "pearson_npu_id_correlation": correlation,
        "jain_fairness": (
            sum(values) ** 2 / (len(values) * squared_sum)
            if squared_sum > 0
            else None
        ),
    }


def _load_optional(path: Path):
    if not path.exists():
        return None, {"path": str(path), "status": "missing"}
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, {
            "path": str(path),
            "status": "unreadable",
            "error": f"{type(error).__name__}: {error}",
        }
    if not isinstance(payload, dict):
        return None, {
            "path": str(path),
            "status": "invalid_root",
            "error": "JSON root must be an object",
        }
    return payload, {
        "path": str(path),
        "status": "ok",
        "declared_complete": bool(payload.get("complete", False)),
        "declared_result_count": len(payload.get("results", ()))
        if isinstance(payload.get("results", ()), list)
        else None,
    }


def _strategy_name(row):
    strategy = row.get("strategy")
    if strategy in STRATEGIES:
        return strategy
    # This historical name is a 10-ms controller and is deliberately *not*
    # relabeled as the requested 25/50-ms controls.
    return None


def _infer_num_npu(row, payload, default=None):
    summary = row.get("steady_summary") or {}
    experiment = payload.get("experiment") or payload.get("experiment_spec") or {}
    for value in (
        row.get("num_npu"),
        summary.get("num_npu"),
        experiment.get("num_npu"),
        default,
    ):
        if value is not None:
            return int(value)
    return None


def _request_metrics(request_rows):
    valid = [
        row
        for row in request_rows
        if isinstance(row, dict)
        and row.get("npu_id") is not None
        and row.get("slo_met") is not None
    ]
    by_npu = {}
    for row in valid:
        by_npu.setdefault(int(row["npu_id"]), []).append(row)

    per_npu = [
        statistics.fmean(bool(row["slo_met"]) for row in rows)
        for rows in by_npu.values()
        if rows
    ]
    per_npu_by_id = [
        (
            npu_id,
            statistics.fmean(bool(row["slo_met"]) for row in rows),
        )
        for npu_id, rows in by_npu.items()
        if rows
    ]
    category = {}
    for name in CATEGORIES:
        rows = [row for row in valid if row.get("category") == name]
        by_category_npu = {}
        for row in rows:
            by_category_npu.setdefault(int(row["npu_id"]), []).append(row)
        equal_npu_values = [
            statistics.fmean(bool(row["slo_met"]) for row in npu_rows)
            for npu_rows in by_category_npu.values()
            if npu_rows
        ]
        category[name] = {
            "request_count": len(rows),
            "npu_count": len(by_category_npu),
            "request_weighted_slo": (
                statistics.fmean(bool(row["slo_met"]) for row in rows)
                if rows
                else None
            ),
            "equal_npu_slo": _safe_mean(equal_npu_values),
        }
    populated_category_slos = [
        metrics["equal_npu_slo"]
        for metrics in category.values()
        if metrics["equal_npu_slo"] is not None
    ]
    return {
        "request_count": len(valid),
        "npu_count": len(by_npu),
        "request_weighted_slo": (
            statistics.fmean(bool(row["slo_met"]) for row in valid)
            if valid
            else None
        ),
        "equal_npu_slo": _safe_mean(per_npu),
        "per_npu_slo": _id_bias_metrics(per_npu_by_id),
        "equal_category_slo": _safe_mean(populated_category_slos),
        "category": category,
    }


def _queue_diagnostics(summary):
    start = [
        int(value)
        for value in summary.get("measurement_ssd_outstanding_blocks_at_start", ())
    ]
    end = [
        int(value)
        for value in summary.get("measurement_ssd_outstanding_blocks_at_end", ())
    ]
    block_utils = [
        float(row["npu_utilization"])
        for row in summary.get("measurement_blocks", ())
        if isinstance(row, dict) and row.get("npu_utilization") is not None
    ]
    if not start or len(start) != len(end):
        return {
            "available": False,
            "stationarity_class": "unavailable",
            "stationarity_note": "queue endpoint arrays are missing or mismatched",
            "block_npu_utilization_range": (
                max(block_utils) - min(block_utils) if block_utils else None
            ),
        }

    total_start = sum(start)
    total_end = sum(end)
    delta = total_end - total_start
    ratio = (total_end + 1.0) / (total_start + 1.0)
    block_range = max(block_utils) - min(block_utils) if block_utils else None

    # Endpoint equality cannot prove stationarity.  The labels distinguish
    # strong evidence of drift from endpoints that merely do not reject it.
    if ratio < 0.5 or ratio > 2.0 or (block_range is not None and block_range > 0.10):
        stationarity_class = "drift_detected"
        note = "endpoint queue changed by >2x or utilization blocks span >10 pp"
    elif 0.8 <= ratio <= 1.25 and (
        block_range is None or block_range <= 0.05
    ):
        stationarity_class = "compatible_not_proven"
        note = "endpoints and utilization blocks are compatible with stationarity"
    else:
        stationarity_class = "inconclusive"
        note = "two queue endpoints are insufficient to establish stationarity"
    duration_ms = float(summary.get("measurement_duration_ms") or 0.0)
    return {
        "available": True,
        "total_start_blocks": total_start,
        "total_end_blocks": total_end,
        "total_delta_blocks": delta,
        "end_to_start_ratio": ratio,
        "delta_blocks_per_second": (
            1000.0 * delta / duration_ms if duration_ms > 0 else None
        ),
        "max_abs_per_ssd_delta_blocks": max(
            abs(after - before) for before, after in zip(start, end)
        ),
        "growing_ssd_count": sum(after > before for before, after in zip(start, end)),
        "shrinking_ssd_count": sum(after < before for before, after in zip(start, end)),
        "block_npu_utilization_range": block_range,
        "stationarity_class": stationarity_class,
        "stationarity_note": note,
    }


def _utilization_distribution(values, served_gbps=()):
    values = [float(value) for value in values]
    served_gbps = [float(value) for value in served_gbps]
    if not values:
        return {"available": False}
    mean = statistics.fmean(values)
    maximum = max(values)
    return {
        "available": True,
        "count": len(values),
        "mean": mean,
        "minimum": min(values),
        "p95": _percentile(values, 95),
        "maximum": maximum,
        "max_index": values.index(maximum),
        "max_minus_mean": maximum - mean,
        "coefficient_of_variation": (
            statistics.pstdev(values) / mean if mean > 0 else None
        ),
        "saturated_count_ge_99pct": sum(value >= 0.99 for value in values),
        "max_served_gbps": max(served_gbps) if served_gbps else None,
    }


def _pearson(left, right):
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    if len(pairs) < 2:
        return None
    a_values = [value for value, _ in pairs]
    b_values = [value for _, value in pairs]
    a_std = statistics.pstdev(a_values)
    b_std = statistics.pstdev(b_values)
    if a_std <= 0.0 or b_std <= 0.0:
        return None
    a_mean = statistics.fmean(a_values)
    b_mean = statistics.fmean(b_values)
    covariance = statistics.fmean(
        (a - a_mean) * (b - b_mean) for a, b in pairs
    )
    return covariance / (a_std * b_std)


def _ring_manifest_stats(requests, *, num_npu, num_ssu, n_layers):
    matrix = [[0.0] * num_ssu for _ in range(num_npu)]
    missing_npu = 0
    request_fanout = []
    for request in requests:
        npu_id = int(request.npu_id)
        if npu_id < 0 or npu_id >= num_npu:
            missing_npu += 1
            continue
        request_fanout.append(
            sum(float(amount) > 1e-12 for amount in request.work_by_ssu_gb)
        )
        for ssu_id, amount in enumerate(request.work_by_ssu_gb):
            matrix[npu_id][ssu_id] += float(amount) * n_layers
    ssu_total = [sum(row[ssu] for row in matrix) for ssu in range(num_ssu)]
    active_owner_count = []
    dominant_owner_npu = []
    dominant_owner_share = []
    owner_hhi = []
    for ssu_id, total in enumerate(ssu_total):
        ownership = [row[ssu_id] for row in matrix]
        active_owner_count.append(sum(amount > 1e-12 for amount in ownership))
        if total <= 1e-12:
            dominant_owner_npu.append(None)
            dominant_owner_share.append(None)
            owner_hhi.append(None)
            continue
        maximum = max(ownership)
        dominant_owner_npu.append(ownership.index(maximum))
        dominant_owner_share.append(maximum / total)
        owner_hhi.append(sum((amount / total) ** 2 for amount in ownership))
    npu_total = [sum(row) for row in matrix]
    npu_ssu_count = [sum(amount > 1e-12 for amount in row) for row in matrix]
    npu_ssu_hhi = [
        sum((amount / total) ** 2 for amount in row) if total > 1e-12 else None
        for row, total in zip(matrix, npu_total)
    ]
    return {
        "request_count": len(requests),
        "missing_npu_count": missing_npu,
        "total_manifest_gb": sum(ssu_total),
        "mean_request_ssu_fanout": _safe_mean(request_fanout),
        "p95_request_ssu_fanout": _percentile(request_fanout, 95),
        "max_request_ssu_fanout": max(request_fanout, default=None),
        "ssu_manifest_gb": ssu_total,
        "ssu_active_owner_npu_count": active_owner_count,
        "ssu_dominant_owner_npu": dominant_owner_npu,
        "ssu_dominant_owner_share": dominant_owner_share,
        "ssu_owner_hhi": owner_hhi,
        "npu_manifest_gb": npu_total,
        "npu_owned_ssu_count": npu_ssu_count,
        "npu_ssu_hhi": npu_ssu_hhi,
    }


def _ring_association_one(stats, summary):
    if not stats or not stats["ssu_manifest_gb"]:
        return {"available": False}
    manifest = stats["ssu_manifest_gb"]
    utilization = [float(value) for value in summary.get("measurement_ssd_utilizations", ())]
    queue_start = [
        float(value)
        for value in summary.get("measurement_ssd_outstanding_blocks_at_start", ())
    ]
    queue_end = [
        float(value)
        for value in summary.get("measurement_ssd_outstanding_blocks_at_end", ())
    ]
    queue_delta = [after - before for before, after in zip(queue_start, queue_end)]
    npu_util = [float(value) for value in summary.get("npu_utilizations", ())]
    hot_manifest = manifest.index(max(manifest)) if manifest else None
    hot_observed = utilization.index(max(utilization)) if utilization else None
    sorted_manifest = sorted(manifest)
    observed_manifest_percentile = None
    if hot_observed is not None and len(manifest) == len(utilization):
        observed_amount = manifest[hot_observed]
        observed_manifest_percentile = (
            sum(value <= observed_amount for value in manifest) / len(manifest)
        )
    dominant_values = stats["ssu_dominant_owner_share"]
    hhi_values = stats["ssu_owner_hhi"]
    return {
        "available": True,
        "request_count": stats["request_count"],
        "total_manifest_gb": stats["total_manifest_gb"],
        "mean_request_ssu_fanout": stats["mean_request_ssu_fanout"],
        "p95_request_ssu_fanout": stats["p95_request_ssu_fanout"],
        "max_request_ssu_fanout": stats["max_request_ssu_fanout"],
        "manifest_ssu_coefficient_of_variation": (
            statistics.pstdev(manifest) / statistics.fmean(manifest)
            if statistics.fmean(manifest) > 0.0
            else None
        ),
        "manifest_ssu_max_over_mean": (
            max(manifest) / statistics.fmean(manifest)
            if statistics.fmean(manifest) > 0.0
            else None
        ),
        "hottest_manifest_ssu": hot_manifest,
        "hottest_observed_util_ssu": hot_observed,
        "hot_ssu_matches": hot_manifest == hot_observed,
        "observed_hot_ssu_manifest_percentile": observed_manifest_percentile,
        "manifest_vs_ssd_utilization_correlation": _pearson(manifest, utilization),
        "manifest_vs_queue_start_correlation": _pearson(manifest, queue_start),
        "manifest_vs_queue_end_correlation": _pearson(manifest, queue_end),
        "manifest_vs_queue_delta_correlation": _pearson(manifest, queue_delta),
        "owner_count_vs_ssd_utilization_correlation": _pearson(
            stats["ssu_active_owner_npu_count"], utilization
        ),
        "dominant_owner_share_vs_ssd_utilization_correlation": _pearson(
            dominant_values, utilization
        ),
        "owner_hhi_vs_ssd_utilization_correlation": _pearson(
            hhi_values, utilization
        ),
        "mean_dominant_owner_share": _safe_mean(dominant_values),
        "max_dominant_owner_share": max(
            (value for value in dominant_values if value is not None),
            default=None,
        ),
        "mean_active_owner_npu_count": _safe_mean(
            stats["ssu_active_owner_npu_count"]
        ),
        "mean_npu_owned_ssu_count": _safe_mean(stats["npu_owned_ssu_count"]),
        "npu_manifest_vs_compute_utilization_correlation": _pearson(
            stats["npu_manifest_gb"], npu_util
        ),
        "npu_owned_ssu_count_vs_compute_utilization_correlation": _pearson(
            stats["npu_owned_ssu_count"], npu_util
        ),
        "npu_ssu_hhi_vs_compute_utilization_correlation": _pearson(
            stats["npu_ssu_hhi"], npu_util
        ),
    }


def _ring_manifest_association(summary, catalog_entry, *, num_npu, num_ssu):
    if catalog_entry is None:
        return {
            "available": False,
            "reason": "verified ring workload catalog unavailable",
        }
    request_by_id = catalog_entry["request_by_id"]
    request_ids = {
        int(row["request_id"])
        for row in summary.get("request_rows", ())
        if isinstance(row, dict) and row.get("request_id") is not None
    }
    missing_ids = sorted(request_ids - set(request_by_id))
    cohort_requests = [
        request_by_id[request_id]
        for request_id in sorted(request_ids)
        if request_id in request_by_id
    ]
    n_layers = int(summary.get("n_layers") or catalog_entry["n_layers"])
    cohort = _ring_manifest_stats(
        cohort_requests,
        num_npu=num_npu,
        num_ssu=num_ssu,
        n_layers=n_layers,
    )
    return {
        "available": not missing_ids,
        "placement_hash_verified": catalog_entry["hashes_match"],
        "missing_tagged_request_ids": missing_ids,
        "definition": (
            "immutable ring-owned full manifest of tagged measurement requests; "
            "it excludes warmup/untagged I/O that may overlap the SSD window"
        ),
        "tagged_cohort": _ring_association_one(cohort, summary),
        "full_prefix": _ring_association_one(catalog_entry["full_stats"], summary),
    }


def _build_ring_catalog(*payloads):
    raw_by_topology = {}
    specs = {}
    for payload in payloads:
        if not payload:
            continue
        spec = payload.get("experiment_spec") or payload.get("experiment") or {}
        for row in payload.get("results", ()) if isinstance(payload.get("results", ()), list) else ():
            if not isinstance(row, dict) or not isinstance(row.get("steady_summary"), dict):
                continue
            num_npu = _infer_num_npu(row, payload, 128)
            num_ssu = int(row.get("num_ssu", row["steady_summary"].get("num_ssu")))
            raw_by_topology.setdefault((num_npu, num_ssu), []).append(row)
            specs[(num_npu, num_ssu)] = spec
    if not raw_by_topology:
        return {}, {"status": "no_completed_128_rows", "topologies": {}}

    try:
        import sim
        from steady_state_workload import prepare_steady_state_workload
    except Exception as error:  # pragma: no cover - environment-level failure
        return {}, {
            "status": "import_failed",
            "error": f"{type(error).__name__}: {error}",
            "topologies": {},
        }

    catalog = {}
    topology_status = {}
    tables = {}
    for topology, raw_rows in sorted(raw_by_topology.items()):
        num_npu, num_ssu = topology
        spec = specs[topology]
        try:
            if num_npu not in tables:
                tables[num_npu] = sim.load_bw_table_cache(num_npu=num_npu)
            table = tables[num_npu]
            requests_per_npu = int(spec.get("requests_per_npu_prefix", 32))
            n_layers = int(spec.get("n_layers", 16))
            seed = int(spec.get("seed", 42))
            workload = prepare_steady_state_workload(
                table,
                num_npu=num_npu,
                num_ssu=num_ssu,
                n_layers=n_layers,
                requests_per_npu=requests_per_npu,
                seed=seed,
            )
            expected = {
                "assignment_hash": workload.statistics["assignment_hash"],
                "workload_hash": workload.workload_hash,
                "placement_hash": workload.placement_hash,
                "trace_hash": workload.trace_hash,
            }
            mismatches = {
                field: [row.get(field) for row in raw_rows]
                for field, value in expected.items()
                if any(row.get(field) != value for row in raw_rows)
            }
            hashes_match = not mismatches
            if not hashes_match:
                topology_status[f"{num_npu}x{num_ssu}"] = {
                    "status": "hash_mismatch",
                    "expected": expected,
                    "observed_mismatches": mismatches,
                }
                continue
            request_by_id = {
                int(request.request_id): request for request in workload.requests
            }
            full_stats = _ring_manifest_stats(
                workload.requests,
                num_npu=num_npu,
                num_ssu=num_ssu,
                n_layers=n_layers,
            )
            catalog[topology] = {
                "request_by_id": request_by_id,
                "n_layers": n_layers,
                "hashes_match": True,
                "full_stats": full_stats,
            }
            topology_status[f"{num_npu}x{num_ssu}"] = {
                "status": "verified",
                "completed_rows": len(raw_rows),
                "request_count": len(workload.requests),
                "hashes": expected,
            }
        except Exception as error:  # analyzer must remain partial-safe
            topology_status[f"{num_npu}x{num_ssu}"] = {
                "status": "reconstruction_failed",
                "error": f"{type(error).__name__}: {error}",
            }
    return catalog, {
        "status": "verified" if len(catalog) == len(raw_by_topology) else "partial",
        "topologies": topology_status,
    }


def _row_metrics(
    row,
    payload,
    source_label,
    default_num_npu=None,
    ring_catalog=None,
):
    summary = row.get("steady_summary")
    strategy = _strategy_name(row)
    if strategy is None:
        return None, f"unsupported strategy {row.get('strategy')!r}"
    if not isinstance(summary, dict):
        return None, f"{strategy} SSU={row.get('num_ssu')}: steady_summary missing"
    num_npu = _infer_num_npu(row, payload, default_num_npu)
    num_ssu = row.get("num_ssu", summary.get("num_ssu"))
    if num_npu is None or num_ssu is None:
        return None, f"{strategy}: topology missing"
    num_ssu = int(num_ssu)

    requests = _request_metrics(summary.get("request_rows", ()))
    equal_npu = requests["equal_npu_slo"]
    if equal_npu is None:
        equal_npu = summary.get("ttft_slo_attainment")
    ssd = _utilization_distribution(
        summary.get("measurement_ssd_utilizations", ()),
        summary.get("measurement_ssd_served_gbps_by_ssu", ()),
    )
    npu_link = _utilization_distribution(
        summary.get("measurement_npu_link_utilizations", ())
    )
    mean_util = summary.get("mean_npu_utilization")
    if mean_util is None:
        mean_util = _safe_mean(summary.get("npu_utilizations", ()))
    npu_utilizations = [
        float(value) for value in summary.get("npu_utilizations", ())
    ]
    request_weighted_slo = requests["request_weighted_slo"]
    if request_weighted_slo is None:
        request_weighted_slo = summary.get("request_weighted_slo_attainment")
    result = {
        "source": source_label,
        "strategy": strategy,
        "num_npu": num_npu,
        "num_ssu": num_ssu,
        "ssu_per_32_npu": 32.0 * num_ssu / num_npu,
        "ssd_capacity_gbps": 40.0 * num_ssu,
        "ssd_capacity_per_npu_gbps": 40.0 * num_ssu / num_npu,
        "mean_npu_utilization": (
            float(mean_util) if mean_util is not None else None
        ),
        "equal_npu_slo": float(equal_npu) if equal_npu is not None else None,
        "summary_equal_npu_slo": summary.get("ttft_slo_attainment"),
        "request_weighted_slo": request_weighted_slo,
        "equal_minus_request_weighted_slo_pp": _difference_pp(
            equal_npu, request_weighted_slo
        ),
        "equal_category_slo": requests["equal_category_slo"],
        "category_slo": requests["category"],
        "cross_npu_slo": requests["per_npu_slo"],
        "cross_npu_utilization": _id_bias_metrics(enumerate(npu_utilizations)),
        "measurement_request_count": summary.get("measurement_request_count"),
        "ssd_hotspot": ssd,
        "npu_link": npu_link,
        "queue": _queue_diagnostics(summary),
        "control_evaluations": summary.get("control_evaluations"),
        "control_min_interval_ms": summary.get("control_min_interval_ms"),
        "cir_commits": summary.get("cir_commits"),
        "cir_path_writes": summary.get("cir_path_writes"),
        "drain_stop_ms": summary.get("drain_stop_ms"),
        "effective_average_evaluation_spacing_ms": (
            float(summary["drain_stop_ms"]) / int(summary["control_evaluations"])
            if summary.get("drain_stop_ms") is not None
            and int(summary.get("control_evaluations") or 0) > 0
            else None
        ),
        "adaptive_residual_mode_evaluations": summary.get(
            "adaptive_residual_mode_evaluations"
        ),
        "adaptive_last_selected_fraction": summary.get(
            "adaptive_last_selected_fraction"
        ),
        "invariants_all_hold": bool(summary.get("invariants"))
        and all(bool(value) for value in summary["invariants"].values()),
    }
    catalog_entry = (ring_catalog or {}).get((num_npu, num_ssu))
    result["ring_ownership"] = _ring_manifest_association(
        summary, catalog_entry, num_npu=num_npu, num_ssu=num_ssu
    )
    return result, None


def _collect_rows(
    payload,
    source_label,
    default_num_npu=None,
    ring_catalog=None,
):
    if payload is None:
        return [], []
    raw_rows = payload.get("results", ())
    if not isinstance(raw_rows, list):
        return [], ["results is not a list"]
    rows = []
    issues = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            issues.append(f"row {index}: not an object")
            continue
        metrics, issue = _row_metrics(
            raw,
            payload,
            source_label,
            default_num_npu=default_num_npu,
            ring_catalog=ring_catalog,
        )
        if metrics is not None:
            rows.append(metrics)
        elif raw.get("strategy") in STRATEGIES or issue != "unsupported strategy None":
            issues.append(f"row {index}: {issue}")
    return rows, issues


def _deduplicate_32(*row_groups):
    rows = {}
    duplicates = []
    for group in row_groups:
        for row in group:
            key = (row["strategy"], row["num_npu"], row["num_ssu"])
            if key in rows:
                duplicates.append(key)
                continue
            rows[key] = row
    return list(rows.values()), duplicates


def _paired_input_status(*raw_payloads):
    groups = {}
    for raw_payload in raw_payloads:
        raw_rows = raw_payload.get("results", ()) if raw_payload else ()
        for row in raw_rows if isinstance(raw_rows, list) else ():
            if not isinstance(row, dict) or _strategy_name(row) is None:
                continue
            if not isinstance(row.get("steady_summary"), dict):
                continue
            groups.setdefault(int(row["num_ssu"]), []).append(row)
    result = {}
    for num_ssu, rows in groups.items():
        mismatches = []
        for field in PAIRING_FIELDS:
            values = [row.get(field) for row in rows]
            if any(value is None for value in values) or len(set(values)) != 1:
                mismatches.append(field)
        result[str(num_ssu)] = {
            "completed_strategy_count": len(rows),
            "strategies": sorted(row["strategy"] for row in rows),
            "cross_strategy_comparison_available": len(rows) >= 2,
            "all_available_rows_paired": not mismatches if len(rows) >= 2 else None,
            "mismatched_or_missing_fields": mismatches,
        }
    return result


def _provenance_status(payload):
    if not payload:
        return {"available": False, "internally_consistent": None}
    source = payload.get("source_fingerprint")
    config = payload.get("config_fingerprint")
    raw_rows = payload.get("results", ())
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    checks = {
        "source_stable_flag": payload.get("source_stable_during_run") is True,
        "config_stable_flag": payload.get("config_stable_during_run") is True,
        "source_equals_ending": source is not None
        and source == payload.get("ending_source_fingerprint"),
        "config_equals_ending": config is not None
        and config == payload.get("ending_config_fingerprint"),
        "all_rows_match_top_source": bool(rows)
        and all(row.get("source_fingerprint") == source for row in rows),
        "all_rows_match_top_config": bool(rows)
        and all(row.get("config_fingerprint") == config for row in rows),
        "all_rows_have_case_fingerprint": bool(rows)
        and all(bool(row.get("case_fingerprint")) for row in rows),
    }
    return {
        "available": True,
        "checks": checks,
        "internally_consistent": all(checks.values()),
        "note": (
            "internal checkpoint consistency only; live source/case hash "
            "recomputation remains the runner's responsibility"
        ),
    }


def _live_provenance_status(payload, module_name):
    """Recompute runner fingerprints and every completed row's case hash."""
    if not payload:
        return {"available": False, "valid": None, "module": module_name}
    try:
        import importlib

        module = importlib.import_module(module_name)
        live_source = module._source_fingerprint()
        live_spec = module.experiment_spec()
        live_config = module._config_fingerprint(live_spec)
        raw_rows = payload.get("results", ())
        rows = [row for row in raw_rows if isinstance(row, dict)]
        checks = {
            "top_source_matches_live": payload.get("source_fingerprint")
            == live_source,
            "ending_source_matches_live": payload.get("ending_source_fingerprint")
            == live_source,
            "top_config_matches_live": payload.get("config_fingerprint")
            == live_config,
            "ending_config_matches_live": payload.get("ending_config_fingerprint")
            == live_config,
            "source_stable_flag": payload.get("source_stable_during_run") is True,
            "config_stable_flag": payload.get("config_stable_during_run") is True,
            "all_summary_invariants": bool(rows)
            and all(
                isinstance(row.get("steady_summary"), dict)
                and bool(row["steady_summary"].get("invariants"))
                and all(row["steady_summary"]["invariants"].values())
                for row in rows
            ),
            "all_pairing_hashes_present": bool(rows)
            and all(
                all(row.get(field) is not None for field in PAIRING_FIELDS)
                for row in rows
            ),
        }
        case_hash_checks = []
        for row in rows:
            strategy = row.get("strategy")
            num_ssu = int(row.get("num_ssu"))
            if hasattr(module, "CASE_BY_KEY"):
                case = module.CASE_BY_KEY[(strategy, num_ssu)]
            else:
                case = module.CASE_BY_NAME[strategy]
            import inspect

            case_fingerprint_parameters = inspect.signature(
                module._case_fingerprint
            ).parameters
            if len(case_fingerprint_parameters) == 3:
                expected_case = module._case_fingerprint(
                    case, live_source, live_config
                )
            else:
                expected_case = module._case_fingerprint(
                    case, num_ssu, live_source, live_config
                )
            case_hash_checks.append(
                row.get("source_fingerprint") == live_source
                and row.get("config_fingerprint") == live_config
                and row.get("case_fingerprint") == expected_case
            )
        checks["all_row_source_config_case_hashes"] = bool(case_hash_checks) and all(
            case_hash_checks
        )
        return {
            "available": True,
            "module": module_name,
            "completed_rows": len(rows),
            "checks": checks,
            "valid": all(checks.values()),
        }
    except Exception as error:
        return {
            "available": True,
            "module": module_name,
            "valid": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _strategy_deltas(rows):
    by_topology = {}
    for row in rows:
        by_topology.setdefault((row["num_npu"], row["num_ssu"]), {})[
            row["strategy"]
        ] = row
    output = []
    for (num_npu, num_ssu), group in sorted(by_topology.items()):
        baseline = group.get("baseline")
        if baseline is None:
            continue
        for strategy in STRATEGIES[1:]:
            candidate = group.get(strategy)
            if candidate is None:
                continue
            category_deltas = {}
            for category in CATEGORIES:
                before = baseline["category_slo"][category]["equal_npu_slo"]
                after = candidate["category_slo"][category]["equal_npu_slo"]
                category_deltas[category] = (
                    100.0 * (after - before)
                    if before is not None and after is not None
                    else None
                )
            util_delta = _difference_pp(
                candidate["mean_npu_utilization"],
                baseline["mean_npu_utilization"],
            )
            slo_delta = _difference_pp(
                candidate["equal_npu_slo"], baseline["equal_npu_slo"]
            )
            output.append(
                {
                    "num_npu": num_npu,
                    "num_ssu": num_ssu,
                    "ssu_per_32_npu": 32.0 * num_ssu / num_npu,
                    "strategy": strategy,
                    "candidate_source": candidate["source"],
                    "baseline_source": baseline["source"],
                    "mean_npu_utilization_pp": util_delta,
                    "equal_npu_slo_pp": slo_delta,
                    "equal_category_slo_pp": _difference_pp(
                        candidate["equal_category_slo"],
                        baseline["equal_category_slo"],
                    ),
                    "candidate_equal_minus_request_weighted_slo_pp": candidate[
                        "equal_minus_request_weighted_slo_pp"
                    ],
                    "baseline_equal_minus_request_weighted_slo_pp": baseline[
                        "equal_minus_request_weighted_slo_pp"
                    ],
                    "npu_util_p10_pp": _difference_pp(
                        candidate["cross_npu_utilization"].get("p10"),
                        baseline["cross_npu_utilization"].get("p10"),
                    ),
                    "per_npu_slo_p10_pp": _difference_pp(
                        candidate["cross_npu_slo"].get("p10"),
                        baseline["cross_npu_slo"].get("p10"),
                    ),
                    "category_equal_npu_slo_pp": category_deltas,
                    "joint_goal": _joint_goal(util_delta, slo_delta),
                }
            )
    return output


def _bias_assessments(rows):
    by_topology = {}
    for row in rows:
        by_topology.setdefault(row["num_ssu"], {})[row["strategy"]] = row
    assessments = []
    for num_ssu, group in sorted(by_topology.items()):
        baseline = group.get("baseline")
        if baseline is None:
            continue
        for strategy in (
            "admission_25ms",
            "admission_50ms",
            "admission_v2_25ms",
            "adaptive_v2_1_25ms",
        ):
            candidate = group.get(strategy)
            if candidate is None:
                continue
            base_slo = baseline["cross_npu_slo"]
            cand_slo = candidate["cross_npu_slo"]
            base_util = baseline["cross_npu_utilization"]
            cand_util = candidate["cross_npu_utilization"]
            if not all(
                metrics.get("available")
                for metrics in (base_slo, cand_slo, base_util, cand_util)
            ):
                verdict = "insufficient_per_npu_data"
            else:
                slo_id_bias_growth = abs(
                    cand_slo["first_minus_last_id_quartile"]
                ) - abs(base_slo["first_minus_last_id_quartile"])
                util_id_bias_growth = abs(
                    cand_util["first_minus_last_id_quartile"]
                ) - abs(base_util["first_minus_last_id_quartile"])
                slo_spread_growth = (
                    cand_slo["p90_minus_p10"] - base_slo["p90_minus_p10"]
                )
                if (
                    abs(cand_slo["first_minus_last_id_quartile"]) >= 0.10
                    and slo_id_bias_growth >= 0.05
                    and abs(cand_slo["pearson_npu_id_correlation"]) >= 0.25
                ) or (
                    abs(cand_util["first_minus_last_id_quartile"]) >= 0.05
                    and util_id_bias_growth >= 0.025
                    and abs(cand_util["pearson_npu_id_correlation"]) >= 0.25
                ):
                    verdict = "new_npu_id_bias_signal"
                elif slo_spread_growth >= 0.10:
                    verdict = "cross_npu_inequality_regression_without_clear_id_order"
                else:
                    verdict = "no_new_large_npu_id_bias_signal"
            assessments.append(
                {
                    "num_ssu": num_ssu,
                    "strategy": strategy,
                    "verdict": verdict,
                    "equal_minus_request_weighted_slo_pp": candidate[
                        "equal_minus_request_weighted_slo_pp"
                    ],
                    "baseline_equal_minus_request_weighted_slo_pp": baseline[
                        "equal_minus_request_weighted_slo_pp"
                    ],
                    "candidate_slo_min": cand_slo.get("minimum"),
                    "candidate_slo_p10": cand_slo.get("p10"),
                    "candidate_slo_p90": cand_slo.get("p90"),
                    "candidate_slo_id_quartile_gap": cand_slo.get(
                        "first_minus_last_id_quartile"
                    ),
                    "baseline_slo_id_quartile_gap": base_slo.get(
                        "first_minus_last_id_quartile"
                    ),
                    "candidate_slo_id_correlation": cand_slo.get(
                        "pearson_npu_id_correlation"
                    ),
                    "candidate_util_min": cand_util.get("minimum"),
                    "candidate_util_p10": cand_util.get("p10"),
                    "candidate_util_p90": cand_util.get("p90"),
                    "candidate_util_id_quartile_gap": cand_util.get(
                        "first_minus_last_id_quartile"
                    ),
                    "baseline_util_id_quartile_gap": base_util.get(
                        "first_minus_last_id_quartile"
                    ),
                    "candidate_util_id_correlation": cand_util.get(
                        "pearson_npu_id_correlation"
                    ),
                }
            )
    return assessments


def _difference_pp(after, before):
    if after is None or before is None:
        return None
    return 100.0 * (float(after) - float(before))


def _joint_goal(util_delta_pp, slo_delta_pp):
    if util_delta_pp is None or slo_delta_pp is None:
        return "inconclusive"
    if slo_delta_pp >= 10.0 and util_delta_pp >= -1.0:
        return "strong_slo_win_util_preserved"
    if slo_delta_pp >= 5.0 and util_delta_pp >= -2.0:
        return "moderate_slo_win"
    if slo_delta_pp < 0.0 or util_delta_pp < -2.0:
        return "retune_or_reject"
    return "not_far_superior"


def _direction(delta_pp, tolerance_fraction):
    if delta_pp is None:
        return "unavailable"
    tolerance_pp = 100.0 * tolerance_fraction
    if delta_pp > tolerance_pp:
        return "improved"
    if delta_pp < -tolerance_pp:
        return "degraded"
    return "flat"


def _scale_trends(rows_128, rows_32, deltas_128, deltas_32):
    delta_index_128 = {
        (row["num_ssu"], row["strategy"]): row for row in deltas_128
    }
    delta_index_32 = {
        (row["num_ssu"], row["strategy"]): row for row in deltas_32
    }
    available_32_by_strategy = {}
    for row in deltas_32:
        available_32_by_strategy.setdefault(row["strategy"], []).append(row)

    trends = []
    for (num_ssu, strategy), large in sorted(delta_index_128.items()):
        target_ratio = large["ssu_per_32_npu"]
        candidates = available_32_by_strategy.get(strategy, ())
        if not candidates:
            trends.append(
                {
                    "num_ssu_128": num_ssu,
                    "strategy": strategy,
                    "status": "no_32_reference",
                }
            )
            continue
        reference = min(
            candidates,
            key=lambda row: (abs(row["num_ssu"] - target_ratio), row["num_ssu"]),
        )
        gap = abs(reference["num_ssu"] - target_ratio)
        util_128 = _direction(
            large["mean_npu_utilization_pp"], UTIL_DIRECTION_TOLERANCE
        )
        util_32 = _direction(
            reference["mean_npu_utilization_pp"], UTIL_DIRECTION_TOLERANCE
        )
        slo_128 = _direction(large["equal_npu_slo_pp"], SLO_DIRECTION_TOLERANCE)
        slo_32 = _direction(
            reference["equal_npu_slo_pp"], SLO_DIRECTION_TOLERANCE
        )
        if gap > 1.0:
            consistency = "inconclusive_capacity_gap"
        elif (
            util_128 not in (util_32, "flat")
            and util_32 != "flat"
        ) or (slo_128 not in (slo_32, "flat") and slo_32 != "flat"):
            consistency = "inconsistent"
        elif util_128 == util_32 and slo_128 == slo_32:
            consistency = "consistent"
        else:
            consistency = "compatible_with_flat_tolerance"
        trends.append(
            {
                "num_ssu_128": num_ssu,
                "ssu_per_32_npu": target_ratio,
                "strategy": strategy,
                "candidate_source": large.get("candidate_source"),
                "baseline_source": large.get("baseline_source"),
                "reference_num_ssu_32": reference["num_ssu"],
                "capacity_gap_ssu_per_32": gap,
                "util_delta_pp_128": large["mean_npu_utilization_pp"],
                "util_delta_pp_32": reference["mean_npu_utilization_pp"],
                "util_direction_128": util_128,
                "util_direction_32": util_32,
                "equal_npu_slo_delta_pp_128": large["equal_npu_slo_pp"],
                "equal_npu_slo_delta_pp_32": reference["equal_npu_slo_pp"],
                "slo_direction_128": slo_128,
                "slo_direction_32": slo_32,
                "consistency": consistency,
                "joint_goal_128": large["joint_goal"],
            }
        )
    return trends


def _capacity_references(rows_128, rows_32):
    references = []
    for large in sorted(
        rows_128, key=lambda row: (row["num_ssu"], STRATEGIES.index(row["strategy"]))
    ):
        candidates = [
            row for row in rows_32 if row["strategy"] == large["strategy"]
        ]
        if not candidates:
            references.append(
                {
                    "num_ssu_128": large["num_ssu"],
                    "strategy": large["strategy"],
                    "status": "no_32_reference",
                }
            )
            continue
        target = large["ssu_per_32_npu"]
        lower = max(
            (row for row in candidates if row["num_ssu"] <= target),
            key=lambda row: row["num_ssu"],
            default=None,
        )
        upper = min(
            (row for row in candidates if row["num_ssu"] >= target),
            key=lambda row: row["num_ssu"],
            default=None,
        )
        nearest = min(candidates, key=lambda row: abs(row["num_ssu"] - target))
        references.append(
            {
                "num_ssu_128": large["num_ssu"],
                "strategy": large["strategy"],
                "ssu_per_32_npu": target,
                "status": "exact"
                if math.isclose(nearest["num_ssu"], target, abs_tol=1e-12)
                else "bracketed"
                if lower is not None and upper is not None and lower is not upper
                else "nearest_only",
                "lower_num_ssu_32": lower["num_ssu"] if lower else None,
                "upper_num_ssu_32": upper["num_ssu"] if upper else None,
                "nearest_num_ssu_32": nearest["num_ssu"],
                "nearest_mean_npu_utilization": nearest["mean_npu_utilization"],
                "nearest_equal_npu_slo": nearest["equal_npu_slo"],
                "mean_npu_utilization_delta_pp_vs_nearest": _difference_pp(
                    large["mean_npu_utilization"], nearest["mean_npu_utilization"]
                ),
                "equal_npu_slo_delta_pp_vs_nearest": _difference_pp(
                    large["equal_npu_slo"], nearest["equal_npu_slo"]
                ),
            }
        )
    return references


def _baseline_capacity_anchors(rows_128, rows_32):
    """Attach a baseline capacity anchor without calling it a policy pair."""
    baselines = [row for row in rows_32 if row["strategy"] == "baseline"]
    anchors = []
    for row in sorted(
        rows_128, key=lambda item: (item["num_ssu"], STRATEGIES.index(item["strategy"]))
    ):
        target = row["ssu_per_32_npu"]
        if not baselines:
            anchors.append(
                {
                    "num_ssu_128": row["num_ssu"],
                    "strategy": row["strategy"],
                    "status": "unavailable",
                }
            )
            continue
        lower = max(
            (base for base in baselines if base["num_ssu"] <= target),
            key=lambda base: base["num_ssu"],
            default=None,
        )
        upper = min(
            (base for base in baselines if base["num_ssu"] >= target),
            key=lambda base: base["num_ssu"],
            default=None,
        )
        exact = next(
            (
                base
                for base in baselines
                if math.isclose(base["num_ssu"], target, abs_tol=1e-12)
            ),
            None,
        )
        anchors.append(
            {
                "num_ssu_128": row["num_ssu"],
                "strategy": row["strategy"],
                "ssu_per_32_npu": target,
                "status": "exact_capacity_baseline_anchor"
                if exact is not None
                else "capacity_bracket_baseline_anchor"
                if lower is not None and upper is not None
                else "nearest_baseline_anchor",
                "lower_num_ssu_32": lower["num_ssu"] if lower else None,
                "lower_mean_npu_utilization": (
                    lower["mean_npu_utilization"] if lower else None
                ),
                "lower_equal_npu_slo": lower["equal_npu_slo"] if lower else None,
                "upper_num_ssu_32": upper["num_ssu"] if upper else None,
                "upper_mean_npu_utilization": (
                    upper["mean_npu_utilization"] if upper else None
                ),
                "upper_equal_npu_slo": upper["equal_npu_slo"] if upper else None,
                "policy_pair": row["strategy"] == "baseline",
                "caveat": (
                    "capacity anchor only; policy-matched scaling requires the "
                    "same strategy at 32 NPU"
                ),
            }
        )
    return anchors


def _topology_ring_scale(rows_128, rows_32):
    """Compare strategy-independent full-prefix ring geometry by capacity ratio."""
    representative_128 = {}
    for row in rows_128:
        if row["ring_ownership"].get("available"):
            representative_128.setdefault(row["num_ssu"], row)
    baseline_32 = {
        row["num_ssu"]: row
        for row in rows_32
        if row["strategy"] == "baseline" and row["ring_ownership"].get("available")
    }
    output = []
    for num_ssu, large in sorted(representative_128.items()):
        target = large["ssu_per_32_npu"]
        exact_ssu = int(round(target))
        reference = (
            baseline_32.get(exact_ssu)
            if math.isclose(target, exact_ssu, abs_tol=1e-12)
            else None
        )
        large_ring = large["ring_ownership"]["full_prefix"]
        if reference is None:
            output.append(
                {
                    "num_ssu_128": num_ssu,
                    "ssu_per_32_npu": target,
                    "status": "no_exact_32_topology_ring_reference",
                }
            )
            continue
        small_ring = reference["ring_ownership"]["full_prefix"]
        output.append(
            {
                "num_ssu_128": num_ssu,
                "num_ssu_32": exact_ssu,
                "ssu_per_32_npu": target,
                "status": "exact_capacity_ratio",
                "mean_request_ssu_fanout_128": large_ring[
                    "mean_request_ssu_fanout"
                ],
                "mean_request_ssu_fanout_32": small_ring[
                    "mean_request_ssu_fanout"
                ],
                "p95_request_ssu_fanout_128": large_ring[
                    "p95_request_ssu_fanout"
                ],
                "p95_request_ssu_fanout_32": small_ring[
                    "p95_request_ssu_fanout"
                ],
                "manifest_ssu_cv_128": large_ring[
                    "manifest_ssu_coefficient_of_variation"
                ],
                "manifest_ssu_cv_32": small_ring[
                    "manifest_ssu_coefficient_of_variation"
                ],
                "manifest_ssu_max_over_mean_128": large_ring[
                    "manifest_ssu_max_over_mean"
                ],
                "manifest_ssu_max_over_mean_32": small_ring[
                    "manifest_ssu_max_over_mean"
                ],
                "note": (
                    "capacity ratio is exact, but request fanout and ring "
                    "geometry need not scale identically"
                ),
            }
        )
    return output


def analyze(
    payload_128,
    payload_32_primary,
    payload_32_admission=None,
    payload_32_bracket=None,
    payload_128_v2=None,
    payload_128_supplemental=None,
    payload_128_adaptive=None,
    payload_32_adaptive=None,
    *,
    input_status=None,
    ring_catalog=None,
    ring_catalog_status=None,
    live_validation=None,
):
    rows_128, issues_128 = _collect_rows(
        payload_128,
        "128_v1",
        default_num_npu=128,
        ring_catalog=ring_catalog,
    )
    rows_v2, issues_v2 = _collect_rows(
        payload_128_v2,
        "128_v2",
        default_num_npu=128,
        ring_catalog=ring_catalog,
    )
    rows_128.extend(rows_v2)
    rows_supplemental, issues_supplemental = _collect_rows(
        payload_128_supplemental,
        "128_v1_supplemental",
        default_num_npu=128,
        ring_catalog=ring_catalog,
    )
    rows_adaptive, issues_adaptive = _collect_rows(
        payload_128_adaptive,
        "128_adaptive_v2_1",
        default_num_npu=128,
        ring_catalog=ring_catalog,
    )
    rows_128.extend(rows_supplemental)
    rows_128.extend(rows_adaptive)
    primary, issues_primary = _collect_rows(
        payload_32_primary,
        "32_primary",
        default_num_npu=32,
        ring_catalog=ring_catalog,
    )
    admission, issues_admission = _collect_rows(
        payload_32_admission,
        "32_admission",
        default_num_npu=32,
        ring_catalog=ring_catalog,
    )
    bracket, issues_bracket = _collect_rows(
        payload_32_bracket,
        "32_bracket",
        default_num_npu=32,
        ring_catalog=ring_catalog,
    )
    adaptive_32, issues_adaptive_32 = _collect_rows(
        payload_32_adaptive,
        "32_adaptive_v2_1",
        default_num_npu=32,
        ring_catalog=ring_catalog,
    )
    rows_32, duplicate_32 = _deduplicate_32(
        primary, admission, bracket, adaptive_32
    )
    rows_128.sort(key=lambda row: (row["num_ssu"], STRATEGIES.index(row["strategy"])))
    rows_32.sort(key=lambda row: (row["num_ssu"], STRATEGIES.index(row["strategy"])))

    deltas_128 = _strategy_deltas(rows_128)
    deltas_32 = _strategy_deltas(rows_32)
    expected_v1 = {
        (num_ssu, strategy)
        # The frozen primary run covers SSU40/70.  SSU24 baseline/current are
        # intentionally supplied by the separately checkpointed paired run.
        for num_ssu in (40, 70)
        for strategy in V1_STRATEGIES
    }
    expected_v2 = {
        (num_ssu, strategy)
        for num_ssu in (24, 40, 70)
        for strategy in V2_STRATEGIES
    }
    expected_supplemental = {
        (24, strategy) for strategy in V1_SUPPLEMENTAL_STRATEGIES
    }
    expected_adaptive = {
        (num_ssu, strategy)
        for num_ssu in (24, 40, 70)
        for strategy in ADAPTIVE_STRATEGIES
    }
    present_v1 = {
        (row["num_ssu"], row["strategy"])
        for row in rows_128
        if row["source"] == "128_v1"
    }
    present_v2 = {
        (row["num_ssu"], row["strategy"])
        for row in rows_128
        if row["source"] == "128_v2"
    }
    present_supplemental = {
        (row["num_ssu"], row["strategy"])
        for row in rows_128
        if row["source"] == "128_v1_supplemental"
    }
    present_adaptive = {
        (row["num_ssu"], row["strategy"])
        for row in rows_128
        if row["source"] == "128_adaptive_v2_1"
    }
    missing_v1 = sorted(expected_v1 - present_v1)
    missing_v2 = sorted(expected_v2 - present_v2)
    missing_supplemental = sorted(expected_supplemental - present_supplemental)
    missing_adaptive = sorted(expected_adaptive - present_adaptive)
    missing_128 = [
        ("v1", num_ssu, strategy) for num_ssu, strategy in missing_v1
    ] + [
        ("v2", num_ssu, strategy) for num_ssu, strategy in missing_v2
    ] + [
        ("v1_supplemental", num_ssu, strategy)
        for num_ssu, strategy in missing_supplemental
    ] + [
        ("adaptive_v2_1", num_ssu, strategy)
        for num_ssu, strategy in missing_adaptive
    ]
    trend = _scale_trends(rows_128, rows_32, deltas_128, deltas_32)
    bias_128 = _bias_assessments(rows_128)

    conclusive_trends = [
        row
        for row in trend
        if row.get("consistency")
        not in (None, "inconclusive_capacity_gap", "no_32_reference")
    ]
    inconsistent = [
        row for row in conclusive_trends if row["consistency"] == "inconsistent"
    ]
    joint_failures = [
        row
        for row in trend
        if row.get("joint_goal_128") in ("retune_or_reject", "not_far_superior")
        and (
            row.get("strategy", "").startswith("admission_")
            or row.get("strategy", "").startswith("adaptive_")
        )
    ]
    drift_rows = [
        {
            "num_ssu": row["num_ssu"],
            "strategy": row["strategy"],
            "stationarity_class": row["queue"]["stationarity_class"],
        }
        for row in rows_128
        if row["queue"]["stationarity_class"] == "drift_detected"
    ]
    if not rows_128:
        verdict = "waiting_for_128_results"
    elif missing_128:
        verdict = "partial_128_results"
    elif inconsistent or joint_failures:
        verdict = "retune_or_investigate"
    elif drift_rows:
        verdict = "policy_promising_but_queue_drift_requires_validation"
    else:
        verdict = "scale_trend_consistent_no_retune_signal"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": {
            "verdict": verdict,
            "input_files": input_status or {},
            "completed_128_rows": len(rows_128),
            "expected_128_rows": (
                len(expected_v1)
                + len(expected_v2)
                + len(expected_supplemental)
                + len(expected_adaptive)
            ),
            "completed_v1_rows": len(present_v1),
            "expected_v1_rows": len(expected_v1),
            "completed_v2_rows": len(present_v2),
            "expected_v2_rows": len(expected_v2),
            "completed_v1_supplemental_rows": len(present_supplemental),
            "expected_v1_supplemental_rows": len(expected_supplemental),
            "completed_adaptive_v2_1_rows": len(present_adaptive),
            "expected_adaptive_v2_1_rows": len(expected_adaptive),
            "missing_128_rows": [
                {"source": source, "num_ssu": num_ssu, "strategy": strategy}
                for source, num_ssu, strategy in missing_128
            ],
            "analysis_is_partial": bool(missing_128),
            "issues": {
                "128": issues_128,
                "128_v2": issues_v2,
                "128_v1_supplemental": issues_supplemental,
                "128_adaptive_v2_1": issues_adaptive,
                "32_primary": issues_primary,
                "32_admission": issues_admission,
                "32_bracket": issues_bracket,
                "32_adaptive_v2_1": issues_adaptive_32,
                "duplicate_32_rows_ignored": [list(key) for key in duplicate_32],
            },
            "inconsistent_scale_trends": len(inconsistent),
            "admission_joint_goal_failures": len(joint_failures),
            "queue_drift_rows": drift_rows,
        },
        "method": {
            "capacity_equivalence": "num_ssu * 32 / num_npu",
            "128_ssu_mapping": {"24": 6.0, "40": 10.0, "70": 17.5},
            "util_direction_tolerance_pp": 100.0 * UTIL_DIRECTION_TOLERANCE,
            "slo_direction_tolerance_pp": 100.0 * SLO_DIRECTION_TOLERANCE,
            "queue_caveat": (
                "two endpoints cannot prove stationarity; drift_detected is a "
                "rejection heuristic, compatible_not_proven is not proof"
            ),
            "joint_goal": (
                "strong: equal-NPU SLO >= baseline +10 pp and mean NPU util "
                ">= baseline -1 pp"
            ),
        },
        "paired_input_128": _paired_input_status(
            payload_128,
            payload_128_v2,
            payload_128_supplemental,
            payload_128_adaptive,
        ),
        "provenance_128": _provenance_status(payload_128),
        "provenance_128_v2": _provenance_status(payload_128_v2),
        "provenance_128_v1_supplemental": _provenance_status(
            payload_128_supplemental
        ),
        "provenance_128_adaptive_v2_1": _provenance_status(
            payload_128_adaptive
        ),
        "live_validation": live_validation or {},
        "ring_catalog": ring_catalog_status or {"status": "not_requested"},
        "rows_128": rows_128,
        "rows_32": rows_32,
        "capacity_references": _capacity_references(rows_128, rows_32),
        "baseline_capacity_anchors_32": _baseline_capacity_anchors(
            rows_128, rows_32
        ),
        "topology_ring_scale": _topology_ring_scale(rows_128, rows_32),
        "baseline_deltas_128": deltas_128,
        "baseline_deltas_32": deltas_32,
        "scale_trends": trend,
        "cross_npu_bias_assessment_128": bias_128,
    }


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _pp(value):
    return "—" if value is None else f"{float(value):+.2f} pp"


def render_markdown(analysis):
    status = analysis["status"]
    lines = [
        "# 128-NPU vs 32-NPU warm/full-load scale analysis",
        "",
        f"Verdict: `{status['verdict']}`. Completed 128-NPU rows: "
        f"`{status['completed_128_rows']}/{status['expected_128_rows']}`. "
        "This report is safe to regenerate while the runner checkpoints.",
        "",
        "Capacity-equivalent points use `SSU × 32 / NPU`: 128×24 ↔ 32×6, "
        "128×40 ↔ 32×10, and 128×70 lies between 32×17 and 32×18.",
        "",
        "## 128-NPU completed rows",
        "",
        "| Source | SSU | Strategy | NPU util (min/p10/mean) | Equal-NPU / request SLO | Equal-category SLO | "
        "SSD mean/max | SSD >=99% | NPU-link mean/max | Queue start→end | "
        "Queue check |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in analysis["rows_128"]:
        ssd = row["ssd_hotspot"]
        link = row["npu_link"]
        queue = row["queue"]
        util_npu = row["cross_npu_utilization"]
        lines.append(
            "| {source} | {ssu} | {strategy} | {util} | {slo} | {cat} | {ssd_mean}/{ssd_max} "
            "| {sat} | {link_mean}/{link_max} | {qstart}→{qend} | {qclass} |".format(
                source=row["source"],
                ssu=row["num_ssu"],
                strategy=row["strategy"],
                util="/".join(
                    _pct(value)
                    for value in (
                        util_npu.get("minimum"),
                        util_npu.get("p10"),
                        row["mean_npu_utilization"],
                    )
                ),
                slo=f"{_pct(row['equal_npu_slo'])}/{_pct(row['request_weighted_slo'])}",
                cat=_pct(row["equal_category_slo"]),
                ssd_mean=_pct(ssd.get("mean")),
                ssd_max=_pct(ssd.get("maximum")),
                sat=ssd.get("saturated_count_ge_99pct", "—"),
                link_mean=_pct(link.get("mean")),
                link_max=_pct(link.get("maximum")),
                qstart=queue.get("total_start_blocks", "—"),
                qend=queue.get("total_end_blocks", "—"),
                qclass=queue["stationarity_class"],
            )
        )

    lines.extend(
        [
            "",
            "## Exact capacity-ratio anchors",
            "",
            "An anchor compares the same `SSU×32/NPU` capacity ratio. It is a "
            "policy scaling pair only for baseline; V2 has no 32-NPU V2 run yet.",
            "",
            "| 128 SSU | Strategy | Equivalent 32 SSU | Lower baseline util/SLO | "
            "Upper baseline util/SLO | Status |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis["baseline_capacity_anchors_32"]:
        lower_ssu = row.get("lower_num_ssu_32")
        upper_ssu = row.get("upper_num_ssu_32")
        lines.append(
            f"| {row['num_ssu_128']} | {row['strategy']} | "
            f"{row.get('ssu_per_32_npu', '—')} | "
            f"{('32×' + str(lower_ssu) + ' ' + _pct(row.get('lower_mean_npu_utilization')) + '/' + _pct(row.get('lower_equal_npu_slo'))) if lower_ssu is not None else '—'} | "
            f"{('32×' + str(upper_ssu) + ' ' + _pct(row.get('upper_mean_npu_utilization')) + '/' + _pct(row.get('upper_equal_npu_slo'))) if upper_ssu is not None else '—'} | "
            f"{row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Policy minus baseline",
            "",
            "| Candidate source | Baseline source | NPU | SSU | Strategy | NPU util | Equal-NPU SLO | "
            "Equal-category SLO | Joint goal |",
            "|---|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis["baseline_deltas_128"]:
        lines.append(
            f"| {row['candidate_source']} | {row['baseline_source']} | "
            f"{row['num_npu']} | {row['num_ssu']} | {row['strategy']} | "
            f"{_pp(row['mean_npu_utilization_pp'])} | "
            f"{_pp(row['equal_npu_slo_pp'])} | "
            f"{_pp(row['equal_category_slo_pp'])} | {row['joint_goal']} |"
        )

    lines.extend(
        [
            "",
            "## 128/32 trend comparison",
            "",
            "| 128 SSU | Strategy | 32 reference | Util Δ (128 / 32) | "
            "SLO Δ (128 / 32) | Consistency |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis["scale_trends"]:
        if row.get("reference_num_ssu_32") is None:
            lines.append(
                f"| {row['num_ssu_128']} | {row['strategy']} | — | — | — | "
                f"{row['status']} |"
            )
            continue
        lines.append(
            f"| {row['num_ssu_128']} | {row['strategy']} | "
            f"32×{row['reference_num_ssu_32']} | "
            f"{_pp(row['util_delta_pp_128'])} / {_pp(row['util_delta_pp_32'])} | "
            f"{_pp(row['equal_npu_slo_delta_pp_128'])} / "
            f"{_pp(row['equal_npu_slo_delta_pp_32'])} | "
            f"{row['consistency']} |"
        )

    lines.extend(
        [
            "",
            "## Provenance, invariants, and input pairing",
            "",
            "Live validation recomputes each runner's source, config, and case "
            "fingerprints and checks every completed summary invariant.",
            "",
            "| Source | Available | Completed rows | Valid |",
            "|---|---|---:|---|",
        ]
    )
    for source, validation in analysis.get("live_validation", {}).items():
        lines.append(
            f"| {source} | {validation.get('available')} | "
            f"{validation.get('completed_rows', '—')} | {validation.get('valid')} |"
        )
    lines.extend(
        [
            "",
            "| SSU | Completed sources/strategies | Cross-strategy comparison | "
            "All pairing hashes match |",
            "|---:|---|---|---|",
        ]
    )
    for num_ssu, pairing in sorted(
        analysis["paired_input_128"].items(), key=lambda item: int(item[0])
    ):
        lines.append(
            f"| {num_ssu} | {', '.join(pairing['strategies'])} | "
            f"{pairing['cross_strategy_comparison_available']} | "
            f"{pairing['all_available_rows_paired']} |"
        )

    lines.extend(
        [
            "",
            "## Absolute category equal-NPU SLO",
            "",
            "| Source | SSU | Strategy | SS | SL | LS | LL |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["rows_128"]:
        lines.append(
            f"| {row['source']} | {row['num_ssu']} | {row['strategy']} | "
            + " | ".join(
                _pct(row["category_slo"][name]["equal_npu_slo"])
                for name in CATEGORIES
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Ring ownership association",
            "",
            "The tagged-cohort manifest is reconstructed from the verified immutable "
            "ring placement. It excludes warmup or untagged I/O that can overlap the "
            "SSD measurement window, so correlation is diagnostic rather than causal.",
            "",
            "| Source | SSU | Strategy | Placement verified | Manifest↔SSD util r | "
            "Manifest↔queue-start r | Ring-hot / util-hot SSU | Util-hot manifest percentile | "
            "Mean/max dominant-NPU share |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["rows_128"]:
        ring = row["ring_ownership"]
        cohort = ring.get("tagged_cohort", {})
        if not ring.get("available") or not cohort.get("available"):
            lines.append(
                f"| {row['source']} | {row['num_ssu']} | {row['strategy']} | "
                f"{ring.get('placement_hash_verified', False)} | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {row['source']} | {row['num_ssu']} | {row['strategy']} | "
            f"{ring['placement_hash_verified']} | "
            f"{cohort['manifest_vs_ssd_utilization_correlation'] if cohort['manifest_vs_ssd_utilization_correlation'] is not None else '—'} | "
            f"{cohort['manifest_vs_queue_start_correlation'] if cohort['manifest_vs_queue_start_correlation'] is not None else '—'} | "
            f"{cohort['hottest_manifest_ssu']}/{cohort['hottest_observed_util_ssu']} | "
            f"{_pct(cohort['observed_hot_ssu_manifest_percentile'])} | "
            f"{_pct(cohort['mean_dominant_owner_share'])}/{_pct(cohort['max_dominant_owner_share'])} |"
        )

    lines.extend(
        [
            "",
            "### Exact-ratio ring geometry",
            "",
            "Capacity ratio equality does not preserve the number of SSUs touched by "
            "one request. This table uses the strategy-independent full 32-request "
            "prefix and exact topology ratios only.",
            "",
            "| 128/32 topology | Mean request fanout (128 / 32) | P95 fanout "
            "(128 / 32) | Manifest SSD CV (128 / 32) | Max/mean manifest "
            "(128 / 32) | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in analysis["topology_ring_scale"]:
        if row["status"] != "exact_capacity_ratio":
            lines.append(
                f"| 128×{row['num_ssu_128']} / 32×— | — | — | — | — | "
                f"{row['status']} |"
            )
            continue
        lines.append(
            f"| 128×{row['num_ssu_128']} / 32×{row['num_ssu_32']} | "
            f"{row['mean_request_ssu_fanout_128']:.3f} / "
            f"{row['mean_request_ssu_fanout_32']:.3f} | "
            f"{row['p95_request_ssu_fanout_128']:.3f} / "
            f"{row['p95_request_ssu_fanout_32']:.3f} | "
            f"{row['manifest_ssu_cv_128']:.4f} / {row['manifest_ssu_cv_32']:.4f} | "
            f"{row['manifest_ssu_max_over_mean_128']:.4f} / "
            f"{row['manifest_ssu_max_over_mean_32']:.4f} | exact_capacity_ratio |"
        )

    lines.extend(
        [
            "",
            "## Stationarity diagnostics",
            "",
            "| Source | SSU | Strategy | Outstanding start→end (ratio) | "
            "Queue Δ blocks/s | 500-ms NPU-util range | Classification |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in analysis["rows_128"]:
        queue = row["queue"]
        lines.append(
            f"| {row['source']} | {row['num_ssu']} | {row['strategy']} | "
            f"{queue.get('total_start_blocks', '—')}→{queue.get('total_end_blocks', '—')} "
            f"({queue.get('end_to_start_ratio', '—')}) | "
            f"{queue.get('delta_blocks_per_second', '—')} | "
            f"{_pp(100.0 * queue['block_npu_utilization_range'] if queue.get('block_npu_utilization_range') is not None else None)} | "
            f"{queue['stationarity_class']} |"
        )

    lines.extend(
        [
            "",
            "## Control frequency and Adaptive V2.1 mode audit",
            "",
            "Configured admission intervals are event-gated minimum spacing, not "
            "periodic polling. Average spacing below is `drain_stop/evaluations` "
            "over the complete simulated lifetime.",
            "",
            "| Source | SSU | Strategy | Min interval | Effective avg spacing | "
            "Evaluations / commits | Path writes | Adaptive V1/V2 mode counts |",
            "|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in analysis["rows_128"]:
        if not row.get("control_evaluations"):
            continue
        modes = row.get("adaptive_residual_mode_evaluations")
        mode_text = (
            "—"
            if modes is None
            else f"{modes.get('v1_coflow_residual', 0)}/"
            f"{modes.get('v2_explicit_selected_spill', 0)}"
        )
        lines.append(
            f"| {row['source']} | {row['num_ssu']} | {row['strategy']} | "
            f"{row.get('control_min_interval_ms') if row.get('control_min_interval_ms') is not None else 'event-only'} | "
            f"{row.get('effective_average_evaluation_spacing_ms', '—')} | "
            f"{row['control_evaluations']}/{row['cir_commits']} | "
            f"{row['cir_path_writes']} | {mode_text} |"
        )

    lines.extend(
        [
            "",
            "## Cross-NPU / NPU-id bias audit",
            "",
            "The first-vs-last NPU-id quartile and NPU-id correlation are tie/pinning "
            "signals, not causal proof. They are compared against the paired baseline.",
            "",
            "| SSU | Strategy | Equal−request SLO | SLO min/p10/p90 | "
            "SLO first−last id quartile | Util min/p10/p90 | Verdict |",
            "|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in analysis["cross_npu_bias_assessment_128"]:
        lines.append(
            f"| {row['num_ssu']} | {row['strategy']} | "
            f"{_pp(row['equal_minus_request_weighted_slo_pp'])} | "
            f"{_pct(row['candidate_slo_min'])}/{_pct(row['candidate_slo_p10'])}/"
            f"{_pct(row['candidate_slo_p90'])} | "
            f"{_pp(100.0 * row['candidate_slo_id_quartile_gap'] if row['candidate_slo_id_quartile_gap'] is not None else None)} | "
            f"{_pct(row['candidate_util_min'])}/{_pct(row['candidate_util_p10'])}/"
            f"{_pct(row['candidate_util_p90'])} | {row['verdict']} |"
        )

    lines.extend(["", "## Category equal-NPU SLO deltas vs baseline", ""])
    lines.extend(
        [
            "| NPU | SSU | Strategy | SS | SL | LS | LL |",
            "|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["baseline_deltas_128"]:
        category = row["category_equal_npu_slo_pp"]
        lines.append(
            f"| {row['num_npu']} | {row['num_ssu']} | {row['strategy']} | "
            + " | ".join(_pp(category[name]) for name in CATEGORIES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `SSD mean/max` and the count at >=99% expose fixed-placement "
            "hotspots; aggregate SSU capacity alone is not sufficient.",
            "- NPU-link mean/max tests whether the 50-GB/s receive link, rather "
            "than SSD allocation, is the visible bottleneck.",
            "- Queue `drift_detected` rejects a steady-state interpretation. "
            "`compatible_not_proven` does not prove stationarity because only "
            "the measurement-window endpoints are available.",
            "- A strong joint win means SLO is at least +10 pp while mean NPU "
            "utilization is no worse than -1 pp relative to the paired baseline.",
        ]
    )
    if status["missing_128_rows"]:
        lines.extend(["", "## Pending 128-NPU cases", ""])
        lines.extend(
            f"- {row['source']} SSU {row['num_ssu']}: `{row['strategy']}`"
            for row in status["missing_128_rows"]
        )
    return "\n".join(lines) + "\n"


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def run(
    input_128=DEFAULT_128_INPUT,
    input_128_v2=DEFAULT_128_V2_INPUT,
    input_128_supplemental=DEFAULT_128_SUPPLEMENTAL_INPUT,
    input_128_adaptive=DEFAULT_128_ADAPTIVE_INPUT,
    input_32_primary=DEFAULT_32_PRIMARY,
    input_32_admission=DEFAULT_32_ADMISSION,
    input_32_bracket=DEFAULT_32_BRACKET,
    input_32_adaptive=DEFAULT_32_ADAPTIVE,
    output=DEFAULT_ANALYSIS,
    report=DEFAULT_REPORT,
):
    payload_128, status_128 = _load_optional(Path(input_128))
    payload_128_v2, status_128_v2 = _load_optional(Path(input_128_v2))
    payload_128_supplemental, status_128_supplemental = _load_optional(
        Path(input_128_supplemental)
    )
    payload_128_adaptive, status_128_adaptive = _load_optional(
        Path(input_128_adaptive)
    )
    payload_primary, status_primary = _load_optional(Path(input_32_primary))
    payload_admission, status_admission = _load_optional(Path(input_32_admission))
    payload_bracket, status_bracket = _load_optional(Path(input_32_bracket))
    payload_32_adaptive, status_32_adaptive = _load_optional(
        Path(input_32_adaptive)
    )
    ring_catalog, ring_catalog_status = _build_ring_catalog(
        payload_128,
        payload_128_v2,
        payload_128_supplemental,
        payload_128_adaptive,
        payload_primary,
        payload_admission,
        payload_bracket,
        payload_32_adaptive,
    )
    live_validation = {
        "128_v1": _live_provenance_status(
            payload_128, "steady_state_128npu_admission_experiment"
        ),
        "128_v2": _live_provenance_status(
            payload_128_v2, "steady_state_128npu_admission_v2_experiment"
        ),
        "128_v1_supplemental": _live_provenance_status(
            payload_128_supplemental,
            "steady_state_128npu_admission_experiment",
        ),
        "128_adaptive_v2_1": _live_provenance_status(
            payload_128_adaptive,
            "steady_state_128npu_adaptive_v2_1_experiment",
        ),
        "32_adaptive_v2_1": _live_provenance_status(
            payload_32_adaptive,
            "steady_state_32npu_adaptive_v2_1_experiment",
        ),
    }
    analysis = analyze(
        payload_128,
        payload_primary,
        payload_admission,
        payload_bracket,
        payload_128_v2,
        payload_128_supplemental,
        payload_128_adaptive,
        payload_32_adaptive,
        input_status={
            "128": status_128,
            "128_v2": status_128_v2,
            "128_v1_supplemental": status_128_supplemental,
            "128_adaptive_v2_1": status_128_adaptive,
            "32_primary": status_primary,
            "32_admission": status_admission,
            "32_bracket": status_bracket,
            "32_adaptive_v2_1": status_32_adaptive,
        },
        ring_catalog=ring_catalog,
        ring_catalog_status=ring_catalog_status,
        live_validation=live_validation,
    )
    _write_atomic(Path(output), json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    _write_atomic(Path(report), render_markdown(analysis))
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-128", type=Path, default=DEFAULT_128_INPUT)
    parser.add_argument("--input-128-v2", type=Path, default=DEFAULT_128_V2_INPUT)
    parser.add_argument(
        "--input-128-supplemental",
        type=Path,
        default=DEFAULT_128_SUPPLEMENTAL_INPUT,
    )
    parser.add_argument(
        "--input-128-adaptive", type=Path, default=DEFAULT_128_ADAPTIVE_INPUT
    )
    parser.add_argument("--input-32-primary", type=Path, default=DEFAULT_32_PRIMARY)
    parser.add_argument("--input-32-admission", type=Path, default=DEFAULT_32_ADMISSION)
    parser.add_argument("--input-32-bracket", type=Path, default=DEFAULT_32_BRACKET)
    parser.add_argument(
        "--input-32-adaptive", type=Path, default=DEFAULT_32_ADAPTIVE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    analysis = run(
        args.input_128,
        args.input_128_v2,
        args.input_128_supplemental,
        args.input_128_adaptive,
        args.input_32_primary,
        args.input_32_admission,
        args.input_32_bracket,
        args.input_32_adaptive,
        args.output,
        args.report,
    )
    print(
        f"wrote {args.output} and {args.report}: "
        f"{analysis['status']['verdict']} "
        f"({analysis['status']['completed_128_rows']}/"
        f"{analysis['status']['expected_128_rows']} 128-NPU rows)"
    )


if __name__ == "__main__":
    main()
