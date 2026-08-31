"""Strictly merge and analyze millisecond-scale QoS-control experiment shards.

This module is post-processing only.  It never imports the experiment runner
and never invokes the simulator.  Compatible, individually complete shards may
be analyzed before the full 42-row grid has arrived; every generated artifact
then carries an explicit ``complete`` flag and a list of missing rows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCHEMA_VERSION = 1
EXPERIMENT_NAME = "32npu_ms_scale_ssu_qos_control_v1"
EXPECTED_SSUS = (6, 10, 18)
EXPECTED_RESULT_COUNT = 42
PRIMARY_SLO_ALPHA = 2.0
SENSITIVITY_SLO_ALPHA = 1.5
SLO_CLASSIFICATION_EPSILON = 1e-12
STATIC_BASELINE = "baseline"
TTL_ANCHOR = "layer_once_ttl_0ms"
ADAPTIVE_ANCHOR = "adaptive_t0_i25ms"
EXPECTED_CASE_NAMES = (
    "baseline",
    "layer_once_ttl_0ms",
    "layer_once_ttl_0p25ms",
    "layer_once_ttl_1ms",
    "layer_once_ttl_2ms",
    "layer_once_ttl_5ms",
    "adaptive_t0_i25ms",
    "adaptive_t0p005_i25ms",
    "adaptive_t0p01_i25ms",
    "adaptive_t0p02_i25ms",
    "adaptive_t0p05_i25ms",
    "adaptive_t0_i50ms",
    "adaptive_t0_i100ms",
    "adaptive_t0_i200ms",
)
EXPECTED_CASE_GRID = {
    "baseline": ("baseline", "baseline", 0.0, 0.0, 0.0),
    "layer_once_ttl_0ms": ("ttl", "layer_once", 0.0, 0.0, 0.0),
    "layer_once_ttl_0p25ms": ("ttl", "layer_once", 0.25, 0.0, 0.0),
    "layer_once_ttl_1ms": ("ttl", "layer_once", 1.0, 0.0, 0.0),
    "layer_once_ttl_2ms": ("ttl", "layer_once", 2.0, 0.0, 0.0),
    "layer_once_ttl_5ms": ("ttl", "layer_once", 5.0, 0.0, 0.0),
    "adaptive_t0_i25ms": ("threshold_interval", "adaptive", 0.0, 0.0, 25.0),
    "adaptive_t0p005_i25ms": ("threshold", "adaptive", 0.0, 0.005, 25.0),
    "adaptive_t0p01_i25ms": ("threshold", "adaptive", 0.0, 0.01, 25.0),
    "adaptive_t0p02_i25ms": ("threshold", "adaptive", 0.0, 0.02, 25.0),
    "adaptive_t0p05_i25ms": ("threshold", "adaptive", 0.0, 0.05, 25.0),
    "adaptive_t0_i50ms": ("interval", "adaptive", 0.0, 0.0, 50.0),
    "adaptive_t0_i100ms": ("interval", "adaptive", 0.0, 0.0, 100.0),
    "adaptive_t0_i200ms": ("interval", "adaptive", 0.0, 0.0, 200.0),
}
STEADY_STATE_FIELDS = {
    "seed",
    "requests_per_npu",
    "warmup_requests_per_npu",
    "settle_ms",
    "measurement_ms",
    "block_ms",
    "slo_alpha",
}
EXPECTED_ADAPTIVE = {
    "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
    "explicit_spill_threshold": 0.75,
    "target_ratio": 0.52,
    "required_ratio": 0.50,
    "background_reserve_fraction": 0.05,
    "ssd_cap_gbps": 40.0,
    "npu_cap_gbps": 50.0,
}
INPUT_FINGERPRINT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
    "workload",
    "placement",
    "trace",
    "simulator",
)
CROSS_TOPOLOGY_FINGERPRINT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
)
CATEGORIES = ("SS", "SL", "LS", "LL")
DEMAND_BINS = (
    "le_10",
    "gt_10_le_20",
    "gt_20_le_40",
    "gt_40_le_50",
    "gt_50_le_80",
    "gt_80",
)
FAMILY_COLORS = {
    6: "#2563EB",
    10: "#EA580C",
    18: "#059669",
}


class ValidationError(ValueError):
    """Raised when shard contents are unsafe to merge."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _canonical_hash(value, namespace: bytes) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"{name} is not finite: {number!r}")
    return number


def _nonnegative(value, name: str) -> float:
    number = _finite(value, name)
    _require(number >= -1e-12, f"{name} is negative: {number!r}")
    return max(0.0, number)


def _fraction(value, name: str) -> float:
    number = _finite(value, name)
    _require(-1e-12 <= number <= 1.0 + 1e-12, f"{name} is outside [0,1]")
    return min(1.0, max(0.0, number))


def _integer(value, name: str, *, minimum: int = 0) -> int:
    _require(not isinstance(value, bool), f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} is not an integer: {value!r}") from error
    _require(number == value, f"{name} is not an exact integer: {value!r}")
    _require(number >= minimum, f"{name} is smaller than {minimum}")
    return number


def _close(actual, expected, name: str, *, tolerance: float = 1e-9) -> None:
    actual_number = _finite(actual, name)
    expected_number = _finite(expected, f"{name} expected")
    _require(
        math.isclose(
            actual_number,
            expected_number,
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        f"{name} mismatch: {actual_number!r} != {expected_number!r}",
    )


def _equivalent_derived_metadata(
    actual, expected, *, absolute_tolerance: float = 1e-12
) -> bool:
    """Compare derived diagnostics without rejecting last-bit FP variation.

    The cryptographic input fingerprints remain exact.  This comparator is
    only for deterministic statistics re-derived from those authenticated
    inputs: summation order and libm details may differ by a few ulps between
    hosts even when the materialized workload is byte-for-byte identical.
    """

    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        actual_number = float(actual)
        expected_number = float(expected)
        return (
            math.isfinite(actual_number)
            and math.isfinite(expected_number)
            and math.isclose(
                actual_number,
                expected_number,
                rel_tol=1e-12,
                abs_tol=absolute_tolerance,
            )
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equivalent_derived_metadata(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent_derived_metadata(left, right)
            for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _vector(value, length: int, name: str, *, fractions: bool = False) -> list[float]:
    _require(isinstance(value, list), f"{name} must be a list")
    _require(len(value) == length, f"{name} must contain {length} entries")
    converter = _fraction if fractions else _nonnegative
    return [converter(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _case_key(row: Mapping[str, object]) -> tuple[str, int]:
    return str(row.get("case")), _integer(row.get("num_ssu"), "row num_ssu", minimum=1)


def _case_fingerprint(case_spec, num_ssu, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": case_spec,
            "num_ssu": int(num_ssu),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"ms-scale-control-case:v1\0",
    )


def _validate_spec(spec) -> tuple[dict[str, dict], tuple[int, ...]]:
    _require(isinstance(spec, dict), "experiment_spec is missing")
    _require(spec.get("experiment") == EXPERIMENT_NAME, "wrong experiment name")
    _require(
        _integer(spec.get("schema_version"), "spec schema") == 1, "wrong spec schema"
    )
    _require(_integer(spec.get("num_npu"), "spec num_npu") == 32, "expected 32 NPUs")
    _require(
        _integer(spec.get("n_layers"), "spec n_layers") == 16, "expected 16 layers"
    )
    _require(
        _integer(spec.get("batch_size"), "spec batch_size") == 1, "expected batch 1"
    )

    raw_ssus = spec.get("default_ssu_list")
    _require(isinstance(raw_ssus, list), "default_ssu_list is missing")
    ssus = tuple(_integer(value, "default SSU", minimum=1) for value in raw_ssus)
    _require(ssus == EXPECTED_SSUS, f"expected SSUs {EXPECTED_SSUS}, got {ssus}")

    raw_cases = spec.get("cases")
    _require(isinstance(raw_cases, list), "spec cases are missing")
    case_specs = {}
    for index, case in enumerate(raw_cases):
        _require(isinstance(case, dict), f"case spec {index} is not an object")
        required = {
            "name",
            "family",
            "kind",
            "pressure_ttl_ms",
            "cir_write_threshold_gbps",
            "min_interval_ms",
        }
        _require(set(case) == required, f"case spec {index} has unexpected fields")
        name = str(case["name"])
        _require(name not in case_specs, f"duplicate case name {name!r}")
        _require(
            str(case["kind"]) in {"baseline", "layer_once", "adaptive"},
            f"bad kind for {name}",
        )
        _nonnegative(case["pressure_ttl_ms"], f"{name} TTL")
        _nonnegative(case["cir_write_threshold_gbps"], f"{name} threshold")
        _nonnegative(case["min_interval_ms"], f"{name} interval")
        (
            expected_family,
            expected_kind,
            expected_ttl,
            expected_threshold,
            expected_interval,
        ) = EXPECTED_CASE_GRID.get(name, (None, None, None, None, None))
        _require(
            case["family"] == expected_family and case["kind"] == expected_kind,
            f"{name}: family/kind differs from the formal main grid",
        )
        _close(case["pressure_ttl_ms"], expected_ttl, f"{name}: formal TTL")
        _close(
            case["cir_write_threshold_gbps"],
            expected_threshold,
            f"{name}: formal CIR threshold",
        )
        _close(
            case["min_interval_ms"],
            expected_interval,
            f"{name}: formal control interval",
        )
        case_specs[name] = case
    _require(
        tuple(case_specs) == EXPECTED_CASE_NAMES,
        "case grid/order differs from the 42-row main spec",
    )
    _require(
        len(case_specs) * len(ssus) == EXPECTED_RESULT_COUNT, "main grid is not 42 rows"
    )

    _require(
        case_specs[STATIC_BASELINE]["kind"] == "baseline",
        "static baseline kind mismatch",
    )
    _require(case_specs[TTL_ANCHOR]["kind"] == "layer_once", "TTL anchor kind mismatch")
    _close(case_specs[TTL_ANCHOR]["pressure_ttl_ms"], 0.0, "TTL anchor")
    _require(
        case_specs[ADAPTIVE_ANCHOR]["kind"] == "adaptive",
        "Adaptive anchor kind mismatch",
    )
    _close(
        case_specs[ADAPTIVE_ANCHOR]["cir_write_threshold_gbps"],
        0.0,
        "Adaptive anchor threshold",
    )
    _close(
        case_specs[ADAPTIVE_ANCHOR]["min_interval_ms"], 25.0, "Adaptive anchor interval"
    )

    workload = spec.get("workload")
    _require(isinstance(workload, dict), "workload spec is missing")
    _require(
        workload.get("mode") == "iid_uniform_profile_catalog_v1", "wrong workload mode"
    )
    workload_seed = _integer(workload.get("seed"), "workload seed")
    _require(workload_seed < 2**64, "workload seed does not fit uint64")
    backing_requests = _integer(
        workload.get("requests_per_npu"),
        "backing requests per NPU",
        minimum=32,
    )
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _require(_is_sha256(workload.get(field)), f"invalid workload {field} hash")
    _require(
        _is_sha256(workload.get("prefix_32_assignment_hash")),
        "invalid workload prefix_32_assignment_hash",
    )
    _require(
        workload.get("full_assignment_hash") == workload.get("assignment"),
        "workload full-assignment alias differs from assignment",
    )
    _require(
        _integer(
            workload.get("scientific_prefix_requests_per_npu"),
            "scientific prefix length",
        )
        == 32,
        "scientific prefix must be 32 requests/NPU",
    )
    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), "steady-state spec is missing")
    _require(
        set(steady) == STEADY_STATE_FIELDS,
        "steady-state fields differ from the runner schema",
    )
    _require(
        _integer(steady.get("seed"), "steady-state seed") == workload_seed,
        "steady-state and workload seeds differ",
    )
    _require(
        _integer(
            steady.get("requests_per_npu"),
            "steady-state backing requests",
            minimum=32,
        )
        == backing_requests,
        "steady-state and workload backing-prefix lengths differ",
    )
    warmup_requests = _integer(
        steady.get("warmup_requests_per_npu"),
        "warmup requests",
        minimum=1,
    )
    _require(
        warmup_requests < backing_requests,
        "warmup must leave at least one request/NPU in the finite backing prefix",
    )
    settle_ms = _nonnegative(steady.get("settle_ms"), "settle_ms")
    measurement_ms = _nonnegative(steady.get("measurement_ms"), "measurement_ms")
    block_ms = _nonnegative(steady.get("block_ms"), "block_ms")
    slo_alpha = _nonnegative(steady.get("slo_alpha"), "slo_alpha")
    _require(measurement_ms > 0.0, "measurement_ms must be positive")
    _require(block_ms > 0.0, "block_ms must be positive")
    _require(slo_alpha > 0.0, "slo_alpha must be positive")
    _close(slo_alpha, PRIMARY_SLO_ALPHA, "primary slo_alpha")
    _require(math.isfinite(settle_ms), "settle_ms must be finite")
    adaptive = spec.get("adaptive")
    _require(isinstance(adaptive, dict), "adaptive spec is missing")
    _require(
        set(adaptive) == set(EXPECTED_ADAPTIVE),
        "Adaptive fields differ from the formal main configuration",
    )
    _require(
        adaptive.get("controller") == EXPECTED_ADAPTIVE["controller"],
        "formal Adaptive controller mismatch",
    )
    for field in (
        "explicit_spill_threshold",
        "target_ratio",
        "required_ratio",
        "background_reserve_fraction",
        "ssd_cap_gbps",
        "npu_cap_gbps",
    ):
        _close(
            adaptive.get(field), EXPECTED_ADAPTIVE[field], f"formal Adaptive {field}"
        )
    _require(
        spec.get("cross_request_layer0_prefetch") is True,
        "formal main run must enable cross-request layer-0 prefetch",
    )
    _require(
        spec.get("placement") == "token-block ring hash reused across all 16 layers",
        "formal placement mode mismatch",
    )
    return case_specs, ssus


def _read_shard(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValidationError(f"cannot read {resolved}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"invalid JSON in {resolved}: {error}") from error
    _require(isinstance(payload, dict), f"{resolved}: root must be an object")
    payload["_analysis_path"] = str(resolved)
    return payload


def _validate_shard_root(payload: Mapping[str, object]) -> None:
    path = payload["_analysis_path"]
    _require(
        _integer(payload.get("schema_version"), f"{path}: shard schema")
        == SCHEMA_VERSION,
        f"{path}: unsupported shard schema",
    )
    _require(
        payload.get("selected_complete") is True,
        f"{path}: selected_complete is not true",
    )
    _require(
        payload.get("source_stable_during_run") is True, f"{path}: source was unstable"
    )
    _require(
        payload.get("config_stable_during_run") is True, f"{path}: config was unstable"
    )
    source = payload.get("source_fingerprint")
    config = payload.get("config_fingerprint")
    _require(_is_sha256(source), f"{path}: invalid source fingerprint")
    _require(_is_sha256(config), f"{path}: invalid config fingerprint")
    _require(
        payload.get("ending_source_fingerprint") == source,
        f"{path}: source endpoints differ",
    )
    _require(
        payload.get("ending_config_fingerprint") == config,
        f"{path}: config endpoints differ",
    )

    manifest = payload.get("source_manifest")
    _require(
        isinstance(manifest, dict) and manifest, f"{path}: source manifest missing"
    )
    _require(
        all(
            isinstance(name, str) and _is_sha256(value)
            for name, value in manifest.items()
        ),
        f"{path}: invalid source manifest",
    )
    _require(
        _canonical_hash(manifest, b"ms-scale-control-source:v1\0") == source,
        f"{path}: source fingerprint does not authenticate its manifest",
    )
    spec = payload.get("experiment_spec")
    _validate_spec(spec)
    _require(
        _canonical_hash(spec, b"ms-scale-control-config:v1\0") == config,
        f"{path}: config fingerprint does not authenticate experiment_spec",
    )

    schedule = payload.get("schedule_metadata")
    _require(isinstance(schedule, dict), f"{path}: schedule_metadata missing")
    workload = spec["workload"]
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _require(
            schedule.get(field) == workload[field],
            f"{path}: schedule {field} differs from spec",
        )
    for field in ("mode", "seed", "requests_per_npu"):
        _require(
            schedule.get(field) == workload[field],
            f"{path}: schedule {field} differs from spec",
        )
    _require(
        _integer(schedule.get("num_npu"), f"{path}: schedule num_npu") == 32,
        f"{path}: wrong schedule NPU count",
    )

    results = payload.get("results")
    _require(isinstance(results, list), f"{path}: results must be a list")
    selected = payload.get("selected_keys")
    _require(isinstance(selected, list) and selected, f"{path}: selected_keys missing")
    selected_keys = []
    for index, key in enumerate(selected):
        _require(
            isinstance(key, list) and len(key) == 2,
            f"{path}: malformed selected key {index}",
        )
        selected_keys.append(
            (str(key[0]), _integer(key[1], f"{path}: selected SSU", minimum=1))
        )
    _require(
        len(selected_keys) == len(set(selected_keys)),
        f"{path}: duplicate selected keys",
    )
    available = {_case_key(row) for row in results if isinstance(row, dict)}
    _require(
        set(selected_keys) <= available,
        f"{path}: selected_complete contradicts missing selected rows",
    )


def _validate_group_metric(metric, name: str) -> tuple[int, int, float | None]:
    _require(isinstance(metric, dict), f"{name} must be an object")
    count = _integer(metric.get("count"), f"{name} count")
    passed = _integer(metric.get("passed"), f"{name} passed")
    _require(passed <= count, f"{name} passed exceeds count")
    attainment = metric.get("slo_attainment")
    if count == 0:
        _require(attainment is None, f"{name} zero-count attainment must be null")
        return count, passed, None
    value = _fraction(attainment, f"{name} attainment")
    _close(value, passed / count, f"{name} attainment recomputation")
    return count, passed, value


def _validate_cohort_metrics(row, prefix: str, request_count: int) -> None:
    metrics = row.get("cohort_profile_metrics")
    _require(isinstance(metrics, dict), f"{prefix}: cohort_profile_metrics missing")
    category = metrics.get("category")
    _require(isinstance(category, dict), f"{prefix}: category metrics missing")
    _require(set(category) <= set(CATEGORIES), f"{prefix}: unknown category metric")
    category_total = sum(
        _validate_group_metric(metric, f"{prefix}: category {name}")[0]
        for name, metric in category.items()
    )
    _require(
        category_total == request_count, f"{prefix}: category cohort count mismatch"
    )

    bins = metrics.get("raw_demand_bins")
    _require(
        isinstance(bins, dict) and set(bins) == set(DEMAND_BINS),
        f"{prefix}: demand bins differ",
    )
    bin_total = sum(
        _validate_group_metric(bins[name], f"{prefix}: bin {name}")[0]
        for name in DEMAND_BINS
    )
    _require(bin_total == request_count, f"{prefix}: demand-bin cohort count mismatch")

    profiles = metrics.get("profile")
    _require(
        isinstance(profiles, dict) and profiles, f"{prefix}: profile metrics missing"
    )
    profile_total = 0
    observed = 0
    for name, metric in profiles.items():
        _nonnegative(metric.get("raw_demand_gbps"), f"{prefix}: profile {name} demand")
        count, _, _ = _validate_group_metric(metric, f"{prefix}: profile {name}")
        profile_total += count
        observed += int(count > 0)
    _require(profile_total == request_count, f"{prefix}: profile cohort count mismatch")
    _require(
        _integer(metrics.get("profiles_observed"), f"{prefix}: profiles observed")
        == observed,
        f"{prefix}: profiles_observed mismatch",
    )

    realized = metrics.get("realized_cohort")
    _require(isinstance(realized, dict), f"{prefix}: realized cohort missing")
    _require(
        _integer(realized.get("request_count"), f"{prefix}: realized request count")
        == request_count,
        f"{prefix}: realized cohort count mismatch",
    )
    demand = realized.get("per_npu_raw_demand_gbps")
    _require(isinstance(demand, dict), f"{prefix}: realized per-NPU demand missing")
    demand_min = _nonnegative(demand.get("min"), f"{prefix}: realized demand min")
    demand_max = _nonnegative(demand.get("max"), f"{prefix}: realized demand max")
    demand_mean = _nonnegative(demand.get("mean"), f"{prefix}: realized demand mean")
    _require(
        demand_min <= demand_mean <= demand_max,
        f"{prefix}: realized demand range invalid",
    )
    _nonnegative(
        demand.get("coefficient_of_variation"),
        f"{prefix}: realized demand CV",
    )
    fleet_demand = _nonnegative(
        realized.get("fleet_raw_demand_gbps"),
        f"{prefix}: realized fleet demand",
    )
    _close(
        fleet_demand,
        32.0 * demand_mean,
        f"{prefix}: realized fleet demand aggregation",
        tolerance=1e-7,
    )
    ms_per_gb = realized.get("per_npu_ms_per_gb")
    _require(isinstance(ms_per_gb, dict), f"{prefix}: realized ms/GB missing")
    ms_min = _nonnegative(ms_per_gb.get("min"), f"{prefix}: realized ms/GB min")
    ms_max = _nonnegative(ms_per_gb.get("max"), f"{prefix}: realized ms/GB max")
    ms_mean = _nonnegative(ms_per_gb.get("mean"), f"{prefix}: realized ms/GB mean")
    _require(ms_min <= ms_mean <= ms_max, f"{prefix}: realized ms/GB range invalid")
    _close(
        ms_per_gb.get("spread"),
        ms_max - ms_min,
        f"{prefix}: realized ms/GB spread",
    )


def _validate_rate_count(
    summary, count_field: str, rate_field: str, duration_s: float, prefix: str
) -> int:
    count = _integer(summary.get(count_field), f"{prefix}: {count_field}")
    _close(
        summary.get(rate_field),
        count / duration_s,
        f"{prefix}: {rate_field}",
        tolerance=1e-8,
    )
    return count


def _validate_count_vector(
    summary, total_field: str, vector_field: str, num_ssu: int, prefix: str
) -> list[int]:
    values = summary.get(vector_field)
    _require(
        isinstance(values, list) and len(values) == num_ssu,
        f"{prefix}: {vector_field} shape",
    )
    integers = [
        _integer(value, f"{prefix}: {vector_field}[{index}]")
        for index, value in enumerate(values)
    ]
    _require(
        sum(integers) == _integer(summary.get(total_field), f"{prefix}: {total_field}"),
        f"{prefix}: {vector_field} sum mismatch",
    )
    return integers


def _slo_metrics_at_alpha(request_rows, alpha: float, prefix: str) -> dict[str, float]:
    """Reclassify one wall-time cohort at ``alpha`` without changing its members."""
    _require(isinstance(request_rows, list) and request_rows, f"{prefix}: no requests")
    alpha = _nonnegative(alpha, f"{prefix}: alpha")
    _require(alpha > 0.0, f"{prefix}: alpha must be positive")
    by_npu = defaultdict(list)
    outcomes = []
    for index, request in enumerate(request_rows):
        _require(isinstance(request, dict), f"{prefix}: request row {index} malformed")
        npu_id = _integer(request.get("npu_id"), f"{prefix}: request NPU")
        _require(npu_id < 32, f"{prefix}: request NPU out of range")
        ttft_ms = _nonnegative(request.get("ttft_ms"), f"{prefix}: request TTFT")
        ideal_ttft_ms = _finite(
            request.get("ideal_ttft_ms"), f"{prefix}: request ideal TTFT"
        )
        _require(ideal_ttft_ms > 0.0, f"{prefix}: ideal TTFT must be positive")
        passed = ttft_ms <= alpha * ideal_ttft_ms + SLO_CLASSIFICATION_EPSILON
        by_npu[npu_id].append(passed)
        outcomes.append(passed)
    _require(
        set(by_npu) == set(range(32)),
        f"{prefix}: not every NPU has SLO samples",
    )
    return {
        "equal_npu": statistics.fmean(
            statistics.fmean(values) for values in by_npu.values()
        ),
        "request_weighted": statistics.fmean(outcomes),
    }


def _validate_row(row, *, shard: str, case_specs, ssus, source, config, spec) -> None:
    _require(isinstance(row, dict), f"{shard}: result row is not an object")
    case_name, num_ssu = _case_key(row)
    prefix = f"{shard}: {case_name}/SSU{num_ssu}"
    _require(case_name in case_specs, f"{prefix}: unknown case")
    _require(num_ssu in ssus, f"{prefix}: unexpected SSU")
    case_spec = case_specs[case_name]
    _require(row.get("status") == "ok", f"{prefix}: status is not ok")
    _require(row.get("case_spec") == case_spec, f"{prefix}: case_spec mismatch")
    _require(row.get("family") == case_spec["family"], f"{prefix}: family mismatch")
    _require(row.get("kind") == case_spec["kind"], f"{prefix}: kind mismatch")
    _require(row.get("source_fingerprint") == source, f"{prefix}: source mismatch")
    _require(row.get("config_fingerprint") == config, f"{prefix}: config mismatch")
    _require(
        row.get("case_fingerprint")
        == _case_fingerprint(case_spec, num_ssu, source, config),
        f"{prefix}: invalid case fingerprint",
    )

    fingerprints = row.get("input_fingerprints")
    _require(isinstance(fingerprints, dict), f"{prefix}: input_fingerprints missing")
    _require(
        set(fingerprints) == set(INPUT_FINGERPRINT_FIELDS),
        f"{prefix}: input fingerprint fields differ",
    )
    _require(
        all(_is_sha256(value) for value in fingerprints.values()),
        f"{prefix}: malformed input fingerprint",
    )
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _require(
            fingerprints[field] == spec["workload"][field],
            f"{prefix}: {field} differs from spec",
        )
    _require(
        fingerprints["prefix_32_assignment"]
        == spec["workload"]["prefix_32_assignment_hash"],
        f"{prefix}: prefix assignment differs from spec",
    )
    _require(
        fingerprints["full_assignment"] == spec["workload"]["full_assignment_hash"],
        f"{prefix}: full assignment differs from spec",
    )
    _require(
        _is_sha256(row.get("measurement_cohort_fingerprint")),
        f"{prefix}: measurement cohort fingerprint missing or malformed",
    )

    prefix_stats = row.get("prefix_32_workload_statistics")
    _require(
        isinstance(prefix_stats, dict),
        f"{prefix}: prefix_32_workload_statistics missing",
    )
    backing_stats = row.get("workload_statistics")
    _require(isinstance(backing_stats, dict), f"{prefix}: workload statistics missing")
    prefix_materialized = row.get("prefix_32_materialized_fingerprints")
    _require(
        isinstance(prefix_materialized, dict)
        and set(prefix_materialized) == {"workload", "placement", "trace"},
        f"{prefix}: prefix_32 materialized fingerprints differ",
    )
    _require(
        all(_is_sha256(value) for value in prefix_materialized.values()),
        f"{prefix}: malformed prefix_32 materialized fingerprint",
    )
    _require(
        _integer(prefix_stats.get("requests_per_npu"), f"{prefix}: prefix requests")
        == 32,
        f"{prefix}: prefix statistics are not for 32 requests/NPU",
    )
    _require(
        prefix_stats.get("assignment") == fingerprints["prefix_32_assignment"]
        and prefix_stats.get("full_assignment_hash")
        == fingerprints["prefix_32_assignment"],
        f"{prefix}: prefix assignment hashes do not match",
    )
    _require(
        backing_stats.get("prefix_32_assignment_hash")
        == fingerprints["prefix_32_assignment"],
        f"{prefix}: backing-prefix hash does not match prefix materialization",
    )

    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{prefix}: steady_summary missing")
    _require(
        _integer(summary.get("schema_version"), f"{prefix}: summary schema") == 1,
        f"{prefix}: unsupported steady summary schema",
    )
    _require(
        summary.get("input_fingerprint") == fingerprints["simulator"],
        f"{prefix}: simulator fingerprint differs from summary",
    )
    _require(summary.get("mode") == "steady_state_full_load", f"{prefix}: wrong mode")
    _require(
        _integer(summary.get("num_npu"), f"{prefix}: num_npu") == 32,
        f"{prefix}: wrong NPU count",
    )
    _require(
        _integer(summary.get("num_ssu"), f"{prefix}: num_ssu") == num_ssu,
        f"{prefix}: SSU mismatch",
    )
    _require(
        _integer(summary.get("n_layers"), f"{prefix}: n_layers") == 16,
        f"{prefix}: layer mismatch",
    )
    _require(
        _integer(summary.get("batch_size"), f"{prefix}: batch_size") == 1,
        f"{prefix}: batch mismatch",
    )
    invariants = summary.get("invariants")
    _require(
        isinstance(invariants, dict) and invariants, f"{prefix}: invariants missing"
    )
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{prefix}: failed invariants {failed}")

    steady = spec["steady_state"]
    _require(
        _integer(summary.get("warmup_requests_per_npu"), f"{prefix}: warmup requests")
        == _integer(steady["warmup_requests_per_npu"], "spec warmup requests"),
        f"{prefix}: warmup config mismatch",
    )
    _close(summary.get("settle_ms"), steady["settle_ms"], f"{prefix}: settle_ms")
    _close(
        summary.get("measurement_duration_ms"),
        steady["measurement_ms"],
        f"{prefix}: measurement duration",
    )
    _close(summary.get("slo_alpha"), steady["slo_alpha"], f"{prefix}: SLO alpha")
    warm = _nonnegative(summary.get("warmup_reached_ms"), f"{prefix}: warmup reached")
    start = _nonnegative(
        summary.get("measurement_start_ms"), f"{prefix}: measurement start"
    )
    end = _nonnegative(summary.get("measurement_end_ms"), f"{prefix}: measurement end")
    drain = _nonnegative(summary.get("drain_stop_ms"), f"{prefix}: drain stop")
    _require(warm <= start <= end <= drain, f"{prefix}: invalid phase ordering")
    _close(end - start, steady["measurement_ms"], f"{prefix}: exact window")
    _close(summary.get("tail_drain_ms"), drain - end, f"{prefix}: tail drain")

    _close(
        summary.get("pressure_ttl_ms"), case_spec["pressure_ttl_ms"], f"{prefix}: TTL"
    )
    _close(
        summary.get("cir_write_threshold_gbps"),
        case_spec["cir_write_threshold_gbps"],
        f"{prefix}: CIR threshold",
    )
    if case_spec["kind"] == "adaptive":
        _close(
            summary.get("control_min_interval_ms"),
            case_spec["min_interval_ms"],
            f"{prefix}: control interval",
        )
    else:
        _require(
            summary.get("control_min_interval_ms") is None,
            f"{prefix}: static case has control interval",
        )

    npu_utils = _vector(
        summary.get("npu_utilizations"),
        32,
        f"{prefix}: NPU utilizations",
        fractions=True,
    )
    _close(
        summary.get("mean_npu_utilization"),
        statistics.fmean(npu_utils),
        f"{prefix}: mean NPU utilization",
    )
    ssd_utils = _vector(
        summary.get("measurement_ssd_utilizations"),
        num_ssu,
        f"{prefix}: SSD utilizations",
        fractions=True,
    )
    _close(
        summary.get("measurement_ssd_mean_utilization"),
        statistics.fmean(ssd_utils),
        f"{prefix}: mean SSD utilization",
    )

    request_rows = summary.get("request_rows")
    _require(
        isinstance(request_rows, list) and request_rows,
        f"{prefix}: request_rows missing",
    )
    request_count = _integer(
        summary.get("measurement_request_count"), f"{prefix}: request count", minimum=1
    )
    _require(request_count == len(request_rows), f"{prefix}: request count mismatch")
    ids = []
    by_npu = defaultdict(list)
    category_outcomes = defaultdict(list)
    for index, request in enumerate(request_rows):
        _require(isinstance(request, dict), f"{prefix}: request row {index} malformed")
        request_id = _integer(request.get("request_id"), f"{prefix}: request ID")
        npu_id = _integer(request.get("npu_id"), f"{prefix}: request NPU")
        _require(npu_id < 32, f"{prefix}: request NPU out of range")
        _integer(request.get("sequence"), f"{prefix}: request sequence")
        category = request.get("category")
        _require(
            category in CATEGORIES, f"{prefix}: unknown request category {category!r}"
        )
        _require(
            isinstance(request.get("slo_met"), bool),
            f"{prefix}: slo_met is not boolean",
        )
        admission_time_ms = _nonnegative(
            request.get("admission_time_ms"),
            f"{prefix}: request admission time",
        )
        completion_time_ms = _nonnegative(
            request.get("completion_time_ms"),
            f"{prefix}: request completion time",
        )
        _require(
            completion_time_ms >= admission_time_ms,
            f"{prefix}: request completes before admission",
        )
        ttft_ms = _nonnegative(request.get("ttft_ms"), f"{prefix}: request TTFT")
        ideal_ttft_ms = _finite(
            request.get("ideal_ttft_ms"), f"{prefix}: request ideal TTFT"
        )
        _require(ideal_ttft_ms > 0.0, f"{prefix}: ideal TTFT must be positive")
        _close(
            ttft_ms,
            completion_time_ms - admission_time_ms,
            f"{prefix}: request TTFT duration",
            tolerance=1e-8,
        )
        expected_primary_slo = (
            ttft_ms <= PRIMARY_SLO_ALPHA * ideal_ttft_ms + SLO_CLASSIFICATION_EPSILON
        )
        _require(
            request["slo_met"] == expected_primary_slo,
            f"{prefix}: stored slo_met differs from alpha={PRIMARY_SLO_ALPHA:g}",
        )
        ids.append(request_id)
        by_npu[npu_id].append(request["slo_met"])
        category_outcomes[category].append(request["slo_met"])
    _require(len(ids) == len(set(ids)), f"{prefix}: duplicate measurement request IDs")
    _require(set(by_npu) == set(range(32)), f"{prefix}: not every NPU has SLO samples")
    request_counts_by_npu = [len(by_npu[npu_id]) for npu_id in range(32)]
    raw_request_counts_by_npu = summary.get("request_counts_by_npu")
    _require(
        isinstance(raw_request_counts_by_npu, list)
        and len(raw_request_counts_by_npu) == 32,
        f"{prefix}: request_counts_by_npu shape",
    )
    reported_request_counts_by_npu = [
        _integer(value, f"{prefix}: request count NPU{npu_id}", minimum=1)
        for npu_id, value in enumerate(raw_request_counts_by_npu)
    ]
    _require(
        reported_request_counts_by_npu == request_counts_by_npu,
        f"{prefix}: request_counts_by_npu differs from request_rows",
    )
    primary_slo = _slo_metrics_at_alpha(
        request_rows,
        PRIMARY_SLO_ALPHA,
        f"{prefix}: alpha={PRIMARY_SLO_ALPHA:g} recomputation",
    )
    equal_npu_slo = primary_slo["equal_npu"]
    request_slo = primary_slo["request_weighted"]
    _close(
        summary.get("ttft_slo_attainment"), equal_npu_slo, f"{prefix}: equal-NPU SLO"
    )
    _close(
        summary.get("request_weighted_slo_attainment"),
        request_slo,
        f"{prefix}: request-weighted SLO",
    )

    duration_s = _finite(steady["measurement_ms"], "measurement duration") / 1000.0
    _require(
        summary.get("measurement_control_counter_window")
        == "half-open [measurement_start_ms, measurement_end_ms)",
        f"{prefix}: control counters are not scoped to the measurement window",
    )
    reports = _validate_rate_count(
        summary,
        "measurement_pressure_reports",
        "measurement_pressure_report_rate_hz",
        duration_s,
        prefix,
    )
    evaluations = _validate_rate_count(
        summary,
        "measurement_control_evaluations",
        "measurement_control_evaluation_rate_hz",
        duration_s,
        prefix,
    )
    commits = _validate_rate_count(
        summary,
        "measurement_cir_commits",
        "measurement_cir_commit_rate_hz",
        duration_s,
        prefix,
    )
    transactions = _validate_rate_count(
        summary,
        "measurement_cir_write_transactions",
        "measurement_cir_write_transaction_rate_hz",
        duration_s,
        prefix,
    )
    writes = _validate_rate_count(
        summary,
        "measurement_cir_path_writes",
        "measurement_cir_path_write_rate_hz",
        duration_s,
        prefix,
    )
    pressure_by_ssu = _validate_count_vector(
        summary,
        "measurement_pressure_reports",
        "measurement_pressure_reports_by_ssu",
        num_ssu,
        prefix,
    )
    cache_hits = _integer(
        summary.get("measurement_pressure_cache_hits"),
        f"{prefix}: measurement pressure cache hits",
    )
    cache_hits_by_ssu = _validate_count_vector(
        summary,
        "measurement_pressure_cache_hits",
        "measurement_pressure_cache_hits_by_ssu",
        num_ssu,
        prefix,
    )
    pressure_requests = _integer(
        summary.get("measurement_pressure_requests"),
        f"{prefix}: measurement pressure requests",
    )
    pressure_requests_by_ssu = _validate_count_vector(
        summary,
        "measurement_pressure_requests",
        "measurement_pressure_requests_by_ssu",
        num_ssu,
        prefix,
    )
    _require(
        pressure_requests == reports + cache_hits,
        f"{prefix}: pressure requests != reads + cache hits",
    )
    _require(
        pressure_requests_by_ssu
        == [report + hit for report, hit in zip(pressure_by_ssu, cache_hits_by_ssu)],
        f"{prefix}: per-SSU pressure requests mismatch",
    )
    _close(
        summary.get("measurement_pressure_cache_hit_fraction"),
        cache_hits / pressure_requests if pressure_requests else 0.0,
        f"{prefix}: pressure cache-hit fraction",
    )
    fractions_by_ssu = _vector(
        summary.get("measurement_pressure_cache_hit_fraction_by_ssu"),
        num_ssu,
        f"{prefix}: pressure cache-hit fractions by SSU",
        fractions=True,
    )
    for index, (fraction, hit, request_count_at_ssu) in enumerate(
        zip(fractions_by_ssu, cache_hits_by_ssu, pressure_requests_by_ssu)
    ):
        _close(
            fraction,
            hit / request_count_at_ssu if request_count_at_ssu else 0.0,
            f"{prefix}: pressure cache-hit fraction SSU{index}",
        )
    transaction_by_ssu = _validate_count_vector(
        summary,
        "measurement_cir_write_transactions",
        "measurement_cir_write_transactions_by_ssu",
        num_ssu,
        prefix,
    )
    writes_by_ssu = _validate_count_vector(
        summary,
        "measurement_cir_path_writes",
        "measurement_cir_path_writes_by_ssu",
        num_ssu,
        prefix,
    )
    for vector_field, values in (
        ("measurement_pressure_report_rate_hz_by_ssu", pressure_by_ssu),
        ("measurement_cir_write_transaction_rate_hz_by_ssu", transaction_by_ssu),
        ("measurement_cir_path_write_rate_hz_by_ssu", writes_by_ssu),
    ):
        rates = _vector(summary.get(vector_field), num_ssu, f"{prefix}: {vector_field}")
        for index, (rate, count) in enumerate(zip(rates, values)):
            _close(
                rate,
                count / duration_s,
                f"{prefix}: {vector_field}[{index}]",
                tolerance=1e-8,
            )
    _close(
        summary.get("measurement_cir_entries_per_transaction"),
        writes / transactions if transactions else 0.0,
        f"{prefix}: entries/transaction",
    )
    entries_per_transaction_by_ssu = _vector(
        summary.get("measurement_cir_entries_per_transaction_by_ssu"),
        num_ssu,
        f"{prefix}: entries/transaction by SSU",
    )
    for index, (ratio, entry_count, transaction_count) in enumerate(
        zip(entries_per_transaction_by_ssu, writes_by_ssu, transaction_by_ssu)
    ):
        _close(
            ratio,
            entry_count / transaction_count if transaction_count else 0.0,
            f"{prefix}: entries/transaction SSU{index}",
        )
    _require(
        transactions <= writes and commits <= transactions and commits <= evaluations,
        f"{prefix}: CIR evaluation/commit/transaction/entry counters are inconsistent",
    )

    if case_spec["kind"] == "baseline":
        _require(
            pressure_requests == writes == evaluations == transactions == commits == 0,
            f"{prefix}: baseline incurred control operations",
        )
    elif case_spec["kind"] == "layer_once":
        _require(
            reports > 0
            and pressure_requests > 0
            and writes == evaluations == transactions == commits == 0,
            f"{prefix}: layer_once counters invalid",
        )
    else:
        _require(
            pressure_requests == 0 and evaluations > 0,
            f"{prefix}: Adaptive counters invalid",
        )

    disk_cap = _finite(spec["adaptive"]["ssd_cap_gbps"], "SSD cap")
    npu_cap = _finite(spec["adaptive"]["npu_cap_gbps"], "NPU cap")
    max_ssu_cir = _vector(
        summary.get("max_actual_cir_sum_gbps_by_ssu"),
        num_ssu,
        f"{prefix}: max actual SSU CIR",
    )
    _require(
        max(max_ssu_cir) <= disk_cap + 1e-9,
        f"{prefix}: actual SSU CIR exceeds capacity",
    )
    max_npu_cir = summary.get("max_actual_npu_cir_sum_gbps_by_npu")
    _require(
        summary.get("actual_cir_per_npu_capacity_applicable")
        is (max_npu_cir is not None),
        f"{prefix}: actual per-NPU CIR applicability mismatch",
    )
    if max_npu_cir is not None:
        npu_cirs = _vector(max_npu_cir, 32, f"{prefix}: max actual NPU CIR")
        _require(
            max(npu_cirs) <= npu_cap + 1e-9,
            f"{prefix}: actual NPU CIR exceeds capacity",
        )

    raw_outstanding_start = summary.get("measurement_ssd_outstanding_blocks_at_start")
    raw_outstanding_end = summary.get("measurement_ssd_outstanding_blocks_at_end")
    _require(
        isinstance(raw_outstanding_start, list)
        and len(raw_outstanding_start) == num_ssu,
        f"{prefix}: queue start shape",
    )
    _require(
        isinstance(raw_outstanding_end, list) and len(raw_outstanding_end) == num_ssu,
        f"{prefix}: queue end shape",
    )
    outstanding_start = [
        _integer(value, f"{prefix}: queue start[{index}]")
        for index, value in enumerate(raw_outstanding_start)
    ]
    outstanding_end = [
        _integer(value, f"{prefix}: queue end[{index}]")
        for index, value in enumerate(raw_outstanding_end)
    ]
    blocks = summary.get("measurement_blocks")
    _require(
        isinstance(blocks, list) and blocks, f"{prefix}: measurement blocks missing"
    )
    block_utils = [
        _fraction(block.get("npu_utilization"), f"{prefix}: block utilization")
        for block in blocks
    ]
    block_requests = [
        _integer(block.get("request_count"), f"{prefix}: block requests")
        for block in blocks
    ]
    _require(
        sum(block_requests) == request_count,
        f"{prefix}: measurement-block request counts do not cover request_rows",
    )
    stationarity = row.get("stationarity_diagnostics")
    _require(
        isinstance(stationarity, dict), f"{prefix}: stationarity diagnostics missing"
    )
    _require(
        stationarity.get("block_npu_utilizations") == block_utils,
        f"{prefix}: block utilization diagnostics mismatch",
    )
    _require(
        stationarity.get("block_request_counts") == block_requests,
        f"{prefix}: block request diagnostics mismatch",
    )
    _close(
        stationarity.get("block_utilization_range"),
        max(block_utils) - min(block_utils),
        f"{prefix}: block utilization range",
    )
    _close(
        stationarity.get("first_last_utilization_delta"),
        block_utils[-1] - block_utils[0],
        f"{prefix}: block first/last delta",
    )
    drift = [
        end_value - start_value
        for start_value, end_value in zip(outstanding_start, outstanding_end)
    ]
    _require(
        stationarity.get("outstanding_blocks_drift_by_ssu") == drift,
        f"{prefix}: queue drift vector mismatch",
    )
    _require(
        _integer(
            stationarity.get("fleet_outstanding_blocks_drift"),
            f"{prefix}: fleet drift",
            minimum=-(10**18),
        )
        == sum(drift),
        f"{prefix}: fleet queue drift mismatch",
    )

    _require(
        _integer(backing_stats.get("requests_per_npu"), f"{prefix}: backing requests")
        == _integer(spec["workload"]["requests_per_npu"], "spec backing requests"),
        f"{prefix}: backing statistics length mismatch",
    )
    for stats_name, stats_value, expected_requests, expected_assignment in (
        ("prefix32", prefix_stats, 32, fingerprints["prefix_32_assignment"]),
        (
            "backing",
            backing_stats,
            _integer(spec["workload"]["requests_per_npu"], "spec backing requests"),
            fingerprints["full_assignment"],
        ),
    ):
        _require(
            stats_value.get("catalog") == fingerprints["catalog"],
            f"{prefix}: {stats_name} catalog fingerprint mismatch",
        )
        if stats_name == "backing":
            for field in ("recipe", "schedule"):
                _require(
                    stats_value.get(field) == fingerprints[field],
                    f"{prefix}: backing {field} fingerprint mismatch",
                )
        else:
            _require(
                _is_sha256(stats_value.get("recipe"))
                and _is_sha256(stats_value.get("schedule")),
                f"{prefix}: malformed prefix32 recipe/schedule fingerprint",
            )
        _require(
            stats_value.get("assignment") == expected_assignment
            and stats_value.get("full_assignment_hash") == expected_assignment,
            f"{prefix}: {stats_name} assignment fingerprint mismatch",
        )
        _require(
            stats_value.get("prefix_32_assignment_hash")
            == fingerprints["prefix_32_assignment"],
            f"{prefix}: {stats_name} prefix assignment fingerprint mismatch",
        )
        _require(
            _integer(
                stats_value.get("requests_per_npu"), f"{prefix}: {stats_name} requests"
            )
            == expected_requests,
            f"{prefix}: {stats_name} request-prefix length mismatch",
        )
        _require(
            _integer(
                stats_value.get("request_count"),
                f"{prefix}: {stats_name} request count",
            )
            == 32 * expected_requests,
            f"{prefix}: {stats_name} fleet request count mismatch",
        )
        _require(
            _integer(stats_value.get("seed"), f"{prefix}: {stats_name} seed")
            == _integer(spec["workload"]["seed"], "spec workload seed"),
            f"{prefix}: {stats_name} seed mismatch",
        )
        fleet_demand = _nonnegative(
            stats_value.get("fleet_demand_gbps"),
            f"{prefix}: {stats_name} fleet demand",
        )
        _require(
            fleet_demand > 0.0,
            f"{prefix}: bad {stats_name} fleet demand",
        )
        _close(
            stats_value.get("capacity_knee_ssu"),
            fleet_demand / disk_cap,
            f"{prefix}: {stats_name} raw knee",
            tolerance=1e-8,
        )
        demands = _vector(
            stats_value.get("demand_gbps_by_ssu"),
            num_ssu,
            f"{prefix}: {stats_name} demand by SSU",
        )
        _close(
            sum(demands),
            fleet_demand,
            f"{prefix}: {stats_name} placement demand aggregation",
            tolerance=1e-7,
        )
        _close(
            stats_value.get("max_ssu_demand_gbps"),
            max(demands),
            f"{prefix}: {stats_name} max SSU demand",
        )
        _require(
            _integer(
                stats_value.get("ssu_over_40_count"),
                f"{prefix}: {stats_name} over-40 count",
            )
            == sum(value > disk_cap for value in demands),
            f"{prefix}: {stats_name} over-40 count mismatch",
        )
    _validate_cohort_metrics(row, prefix, request_count)
    category_metrics = row["cohort_profile_metrics"]["category"]
    for category, outcomes in category_outcomes.items():
        metric = category_metrics.get(category)
        _require(
            isinstance(metric, dict)
            and _integer(metric.get("count"), f"{prefix}: category {category} count")
            == len(outcomes)
            and _integer(metric.get("passed"), f"{prefix}: category {category} passed")
            == sum(outcomes),
            f"{prefix}: category {category} does not match request_rows",
        )

    completed = summary.get("completed_by_npu_at_stop")
    _require(
        isinstance(completed, list) and len(completed) == 32,
        f"{prefix}: completed_by_npu_at_stop shape",
    )
    completed_counts = [
        _integer(value, f"{prefix}: completed count NPU{index}")
        for index, value in enumerate(completed)
    ]
    _require(
        max(completed_counts)
        <= _integer(spec["workload"]["requests_per_npu"], "spec backing requests"),
        f"{prefix}: completed count exceeds finite backing prefix",
    )


def _validate_shard_consistency(
    shards: Sequence[dict],
) -> tuple[dict, dict, tuple[int, ...], str, str]:
    reference = shards[0]
    spec = reference["experiment_spec"]
    case_specs, ssus = _validate_spec(spec)
    source = reference["source_fingerprint"]
    config = reference["config_fingerprint"]
    for shard in shards:
        path = shard["_analysis_path"]
        _require(
            shard["source_fingerprint"] == source,
            f"{path}: source differs across shards",
        )
        _require(
            shard["config_fingerprint"] == config,
            f"{path}: config differs across shards",
        )
        _require(
            shard["experiment_spec"] == spec,
            f"{path}: experiment_spec differs across shards",
        )
        _require(
            shard["source_manifest"] == reference["source_manifest"],
            f"{path}: source manifest differs",
        )
        _require(
            shard["schedule_metadata"] == reference["schedule_metadata"],
            f"{path}: schedule metadata differs",
        )
    return spec, case_specs, ssus, source, config


def _scientific_row(row: Mapping[str, object]) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"runtime", "wall_time_s", "_analysis_source_shards"}
    }


def _collect_rows(shards, case_specs, ssus, source, config, spec):
    rows = {}
    provenance = defaultdict(list)
    for shard in shards:
        path = shard["_analysis_path"]
        local_keys = set()
        for row in shard["results"]:
            key = _case_key(row)
            _require(key not in local_keys, f"{path}: duplicate row {key} inside shard")
            local_keys.add(key)
            _validate_row(
                row,
                shard=path,
                case_specs=case_specs,
                ssus=ssus,
                source=source,
                config=config,
                spec=spec,
            )
            if key in rows:
                _require(
                    _scientific_row(rows[key]) == _scientific_row(row),
                    f"duplicate row {key} differs scientifically across shards",
                )
            else:
                rows[key] = row
            provenance[key].append(path)
    return rows, provenance


def _pairing_audit(rows, case_specs, ssus, spec):
    audit = {}
    for num_ssu in ssus:
        group = [row for (name, ssu), row in rows.items() if ssu == num_ssu]
        available = sorted(row["case"] for row in group)
        field_status = {}
        for field in INPUT_FINGERPRINT_FIELDS:
            field_status[field] = (
                bool(group)
                and len({row["input_fingerprints"][field] for row in group}) == 1
            )
        workload_statistics_paired = bool(group) and all(
            _equivalent_derived_metadata(
                group[0]["workload_statistics"], row["workload_statistics"]
            )
            for row in group[1:]
        )
        prefix_statistics_paired = bool(group) and all(
            _equivalent_derived_metadata(
                group[0]["prefix_32_workload_statistics"],
                row["prefix_32_workload_statistics"],
            )
            for row in group[1:]
        )
        prefix_materialized_paired = (
            bool(group)
            and len(
                {
                    json.dumps(
                        row["prefix_32_materialized_fingerprints"],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for row in group
                }
            )
            == 1
        )
        cohort_fingerprints = {row["measurement_cohort_fingerprint"] for row in group}
        complete_cases = set(available) == set(case_specs)
        audit[str(num_ssu)] = {
            "available_cases": available,
            "available_count": len(available),
            "expected_count": len(case_specs),
            "complete_case_set": complete_cases,
            "fingerprint_fields": field_status,
            "workload_statistics_paired": workload_statistics_paired,
            "prefix_32_workload_statistics_paired": prefix_statistics_paired,
            "prefix_32_materialized_fingerprints_paired": prefix_materialized_paired,
            "distinct_measurement_cohort_count": len(cohort_fingerprints),
            "measurement_cohorts_identical": bool(group)
            and len(cohort_fingerprints) == 1,
            "all_available_rows_paired": (
                all(field_status.values())
                and workload_statistics_paired
                and prefix_statistics_paired
                and prefix_materialized_paired
            ),
        }
        _require(
            audit[str(num_ssu)]["all_available_rows_paired"] or not group,
            f"SSU{num_ssu}: available rows are not input-paired",
        )

    for field in CROSS_TOPOLOGY_FINGERPRINT_FIELDS:
        values = {row["input_fingerprints"][field] for row in rows.values()}
        _require(len(values) <= 1, f"cross-topology fingerprint {field} differs")
        if values and field in {"catalog", "recipe", "schedule", "assignment"}:
            _require(
                next(iter(values)) == spec["workload"][field],
                f"cross-topology {field} differs from spec",
            )
    return audit


def _anchor_for(case_spec) -> str:
    if case_spec["kind"] == "baseline":
        return STATIC_BASELINE
    if case_spec["kind"] == "layer_once":
        return TTL_ANCHOR
    return ADAPTIVE_ANCHOR


def _group_values(row, group_name: str, labels: Sequence[str]) -> dict[str, object]:
    group = row["cohort_profile_metrics"][group_name]
    values = {}
    for label in labels:
        metric = group.get(label, {"count": 0, "passed": 0, "slo_attainment": None})
        values[f"{group_name}_{label}_count"] = int(metric["count"])
        values[f"{group_name}_{label}_passed"] = int(metric["passed"])
        values[f"{group_name}_{label}_slo_pct"] = (
            None
            if metric["slo_attainment"] is None
            else 100.0 * float(metric["slo_attainment"])
        )
    return values


def _common_request_slo(current_row, anchor_row) -> dict[str, object]:
    """Compute a secondary, request-ID-intersection SLO comparison.

    This deliberately does not replace each row's operational wall-time cohort.
    The coverage fields expose the selection introduced by taking an
    intersection of two closed-loop streams.
    """
    current_requests = {
        int(row["request_id"]): (int(row["npu_id"]), bool(row["slo_met"]))
        for row in current_row["steady_summary"]["request_rows"]
    }
    anchor_requests = {
        int(row["request_id"]): (int(row["npu_id"]), bool(row["slo_met"]))
        for row in anchor_row["steady_summary"]["request_rows"]
    }
    common_ids = sorted(set(current_requests).intersection(anchor_requests))
    if not common_ids:
        return {
            "common_with_anchor_request_count": 0,
            "common_with_anchor_npu_count": 0,
            "common_coverage_current_pct": 0.0,
            "common_coverage_anchor_pct": 0.0,
            "common_request_slo_pct": None,
            "anchor_common_request_slo_pct": None,
            "common_request_slo_delta_pp": None,
            "common_equal_npu_slo_pct": None,
            "anchor_common_equal_npu_slo_pct": None,
            "common_equal_npu_slo_delta_pp": None,
        }

    current_by_npu = defaultdict(list)
    anchor_by_npu = defaultdict(list)
    for request_id in common_ids:
        current_npu, current_slo = current_requests[request_id]
        anchor_npu, anchor_slo = anchor_requests[request_id]
        _require(
            current_npu == anchor_npu,
            f"common request {request_id} changed NPU between a row and its anchor",
        )
        current_by_npu[current_npu].append(current_slo)
        anchor_by_npu[anchor_npu].append(anchor_slo)
    _require(
        set(current_by_npu) == set(anchor_by_npu),
        "common-request NPU sets differ",
    )
    current_request_slo = statistics.fmean(
        current_requests[request_id][1] for request_id in common_ids
    )
    anchor_request_slo = statistics.fmean(
        anchor_requests[request_id][1] for request_id in common_ids
    )
    current_equal = statistics.fmean(
        statistics.fmean(values) for values in current_by_npu.values()
    )
    anchor_equal = statistics.fmean(
        statistics.fmean(values) for values in anchor_by_npu.values()
    )
    return {
        "common_with_anchor_request_count": len(common_ids),
        "common_with_anchor_npu_count": len(current_by_npu),
        "common_coverage_current_pct": 100.0 * len(common_ids) / len(current_requests),
        "common_coverage_anchor_pct": 100.0 * len(common_ids) / len(anchor_requests),
        "common_request_slo_pct": 100.0 * current_request_slo,
        "anchor_common_request_slo_pct": 100.0 * anchor_request_slo,
        "common_request_slo_delta_pp": 100.0
        * (current_request_slo - anchor_request_slo),
        "common_equal_npu_slo_pct": 100.0 * current_equal,
        "anchor_common_equal_npu_slo_pct": 100.0 * anchor_equal,
        "common_equal_npu_slo_delta_pp": 100.0 * (current_equal - anchor_equal),
    }


def _compact_rows(rows, case_specs, ssus, spec):
    compact = {}
    requests_per_npu = int(spec["workload"]["requests_per_npu"])
    disk_cap = float(spec["adaptive"]["ssd_cap_gbps"])
    for key, row in rows.items():
        case_name, num_ssu = key
        case = case_specs[case_name]
        summary = row["steady_summary"]
        stationarity = row["stationarity_diagnostics"]
        stats = row["prefix_32_workload_statistics"]
        backing_stats = row["workload_statistics"]
        realized = row["cohort_profile_metrics"]["realized_cohort"]
        realized_demand = realized["per_npu_raw_demand_gbps"]
        realized_ms_per_gb = realized["per_npu_ms_per_gb"]
        completed = [int(value) for value in summary["completed_by_npu_at_stop"]]
        npu_cirs = summary.get("max_actual_npu_cir_sum_gbps_by_npu")
        queue_start_by_ssu = [
            int(value)
            for value in summary["measurement_ssd_outstanding_blocks_at_start"]
        ]
        queue_end_by_ssu = [
            int(value) for value in summary["measurement_ssd_outstanding_blocks_at_end"]
        ]
        queue_drift_by_ssu = [
            int(value) for value in stationarity["outstanding_blocks_drift_by_ssu"]
        ]
        pressure_read_rates_by_ssu = [
            float(value)
            for value in summary["measurement_pressure_report_rate_hz_by_ssu"]
        ]
        cir_entry_write_rates_by_ssu = [
            float(value)
            for value in summary["measurement_cir_path_write_rate_hz_by_ssu"]
        ]
        cir_transaction_rates_by_ssu = [
            float(value)
            for value in summary["measurement_cir_write_transaction_rate_hz_by_ssu"]
        ]
        busiest_pressure_rate = max(pressure_read_rates_by_ssu)
        request_counts_by_npu = [
            int(value) for value in summary["request_counts_by_npu"]
        ]
        minimum_npu_request_count = min(request_counts_by_npu)
        alpha_1p5_slo = _slo_metrics_at_alpha(
            summary["request_rows"],
            SENSITIVITY_SLO_ALPHA,
            f"{case_name}/SSU{num_ssu}: alpha={SENSITIVITY_SLO_ALPHA:g} sensitivity",
        )
        equal_npu_slo_pct = 100.0 * float(summary["ttft_slo_attainment"])
        request_weighted_slo_pct = 100.0 * float(
            summary["request_weighted_slo_attainment"]
        )
        alpha_1p5_equal_npu_slo_pct = 100.0 * alpha_1p5_slo["equal_npu"]
        alpha_1p5_request_weighted_slo_pct = 100.0 * alpha_1p5_slo["request_weighted"]
        _require(
            alpha_1p5_equal_npu_slo_pct <= equal_npu_slo_pct + 1e-9,
            f"{case_name}/SSU{num_ssu}: stricter alpha improved equal-NPU SLO",
        )
        _require(
            alpha_1p5_request_weighted_slo_pct <= request_weighted_slo_pct + 1e-9,
            f"{case_name}/SSU{num_ssu}: stricter alpha improved request-weighted SLO",
        )
        item = {
            "case": case_name,
            "family": case["family"],
            "kind": case["kind"],
            "num_ssu": num_ssu,
            "pressure_ttl_ms": float(case["pressure_ttl_ms"]),
            "cir_write_threshold_gbps": float(case["cir_write_threshold_gbps"]),
            "min_interval_ms": float(case["min_interval_ms"]),
            "family_anchor_case": _anchor_for(case),
            "mean_npu_utilization_pct": 100.0 * float(summary["mean_npu_utilization"]),
            "primary_slo_alpha": PRIMARY_SLO_ALPHA,
            "sensitivity_slo_alpha": SENSITIVITY_SLO_ALPHA,
            "equal_npu_slo_pct": equal_npu_slo_pct,
            "request_weighted_slo_pct": request_weighted_slo_pct,
            "alpha_1p5_equal_npu_slo_pct": alpha_1p5_equal_npu_slo_pct,
            "alpha_1p5_request_weighted_slo_pct": (alpha_1p5_request_weighted_slo_pct),
            "alpha_1p5_equal_npu_change_vs_alpha_2_pp": (
                alpha_1p5_equal_npu_slo_pct - equal_npu_slo_pct
            ),
            "alpha_1p5_request_weighted_change_vs_alpha_2_pp": (
                alpha_1p5_request_weighted_slo_pct - request_weighted_slo_pct
            ),
            "measurement_request_count": int(summary["measurement_request_count"]),
            "measurement_requests_per_npu_min": minimum_npu_request_count,
            "measurement_requests_per_npu_median": statistics.median(
                request_counts_by_npu
            ),
            "measurement_requests_per_npu_max": max(request_counts_by_npu),
            "single_outcome_weight_at_min_sample_pp": 100.0
            / (32.0 * minimum_npu_request_count),
            "measurement_cohort_fingerprint": row["measurement_cohort_fingerprint"],
            "realized_cohort_fleet_raw_demand_gbps": float(
                realized["fleet_raw_demand_gbps"]
            ),
            "realized_cohort_per_npu_demand_min_gbps": float(realized_demand["min"]),
            "realized_cohort_per_npu_demand_max_gbps": float(realized_demand["max"]),
            "realized_cohort_per_npu_demand_mean_gbps": float(realized_demand["mean"]),
            "realized_cohort_per_npu_demand_cv": float(
                realized_demand["coefficient_of_variation"]
            ),
            "realized_cohort_per_npu_ms_per_gb_min": float(realized_ms_per_gb["min"]),
            "realized_cohort_per_npu_ms_per_gb_max": float(realized_ms_per_gb["max"]),
            "ssd_mean_utilization_pct": 100.0
            * float(summary["measurement_ssd_mean_utilization"]),
            "ssd_max_utilization_pct": 100.0
            * max(float(value) for value in summary["measurement_ssd_utilizations"]),
            "true_pressure_reads": int(summary["measurement_pressure_reports"]),
            "true_pressure_read_rate_hz": float(
                summary["measurement_pressure_report_rate_hz"]
            ),
            "true_pressure_read_rate_hz_per_ssu": float(
                summary["measurement_pressure_report_rate_hz"]
            )
            / num_ssu,
            "true_pressure_read_rate_hz_max_ssu": busiest_pressure_rate,
            "true_pressure_read_interval_ms_busiest_ssu": (
                None if busiest_pressure_rate == 0.0 else 1000.0 / busiest_pressure_rate
            ),
            "pressure_cache_hit_fraction_pct": 100.0
            * float(summary["measurement_pressure_cache_hit_fraction"]),
            "cir_entry_writes": int(summary["measurement_cir_path_writes"]),
            "cir_entry_write_rate_hz": float(
                summary["measurement_cir_path_write_rate_hz"]
            ),
            "cir_entry_write_rate_hz_per_ssu": float(
                summary["measurement_cir_path_write_rate_hz"]
            )
            / num_ssu,
            "cir_entry_write_rate_hz_max_ssu": max(cir_entry_write_rates_by_ssu),
            "cir_write_transactions": int(
                summary["measurement_cir_write_transactions"]
            ),
            "cir_write_transaction_rate_hz": float(
                summary["measurement_cir_write_transaction_rate_hz"]
            ),
            "cir_write_transaction_rate_hz_per_ssu": float(
                summary["measurement_cir_write_transaction_rate_hz"]
            )
            / num_ssu,
            "cir_write_transaction_rate_hz_max_ssu": max(cir_transaction_rates_by_ssu),
            "cir_entries_per_transaction": float(
                summary["measurement_cir_entries_per_transaction"]
            ),
            "control_evaluation_rate_hz": float(
                summary["measurement_control_evaluation_rate_hz"]
            ),
            "warmup_reached_ms": float(summary["warmup_reached_ms"]),
            "measurement_start_ms": float(summary["measurement_start_ms"]),
            "measurement_end_ms": float(summary["measurement_end_ms"]),
            "tail_drain_ms": float(summary["tail_drain_ms"]),
            "fleet_queue_start_blocks": sum(
                int(value)
                for value in summary["measurement_ssd_outstanding_blocks_at_start"]
            ),
            "fleet_queue_end_blocks": sum(
                int(value)
                for value in summary["measurement_ssd_outstanding_blocks_at_end"]
            ),
            "fleet_queue_drift_blocks": int(
                stationarity["fleet_outstanding_blocks_drift"]
            ),
            "queue_start_by_ssu_json": json.dumps(
                queue_start_by_ssu, separators=(",", ":")
            ),
            "queue_end_by_ssu_json": json.dumps(
                queue_end_by_ssu, separators=(",", ":")
            ),
            "queue_drift_by_ssu_json": json.dumps(
                queue_drift_by_ssu, separators=(",", ":")
            ),
            "max_abs_queue_drift_blocks": max(
                (abs(value) for value in queue_drift_by_ssu), default=0
            ),
            "block_utilization_range_pp": 100.0
            * float(stationarity["block_utilization_range"]),
            "block_first_last_utilization_delta_pp": 100.0
            * float(stationarity["first_last_utilization_delta"]),
            "completed_by_npu_min": min(completed),
            "completed_by_npu_median": statistics.median(completed),
            "completed_by_npu_max": max(completed),
            "prefix_margin_at_fastest_npu": requests_per_npu - max(completed),
            "fleet_raw_demand_gbps": float(stats["fleet_demand_gbps"]),
            "raw_capacity_knee_ssu": float(stats["capacity_knee_ssu"]),
            "global_raw_load_fraction": float(stats["fleet_demand_gbps"])
            / (disk_cap * num_ssu),
            "max_ssu_raw_demand_gbps": float(stats["max_ssu_demand_gbps"]),
            "ssu_over_40_count": int(stats["ssu_over_40_count"]),
            "backing_prefix_requests_per_npu": int(backing_stats["requests_per_npu"]),
            "backing_fleet_raw_demand_gbps": float(backing_stats["fleet_demand_gbps"]),
            "backing_raw_capacity_knee_ssu": float(backing_stats["capacity_knee_ssu"]),
            "max_actual_ssu_cir_gbps": max(
                float(value) for value in summary["max_actual_cir_sum_gbps_by_ssu"]
            ),
            "max_actual_npu_cir_gbps": None
            if npu_cirs is None
            else max(float(value) for value in npu_cirs),
            "all_invariants_passed": all(summary["invariants"].values()),
            "wall_time_s": float(row.get("wall_time_s", 0.0)),
        }
        item.update(_group_values(row, "category", CATEGORIES))
        item.update(_group_values(row, "raw_demand_bins", DEMAND_BINS))
        compact[key] = item

    for key, item in compact.items():
        case_name, num_ssu = key
        anchor = compact.get((item["family_anchor_case"], num_ssu))
        static = compact.get((STATIC_BASELINE, num_ssu))
        if anchor is not None:
            item["slo_delta_vs_family_anchor_pp"] = (
                item["equal_npu_slo_pct"] - anchor["equal_npu_slo_pct"]
            )
            item["alpha_1p5_equal_npu_delta_vs_family_anchor_pp"] = (
                item["alpha_1p5_equal_npu_slo_pct"]
                - anchor["alpha_1p5_equal_npu_slo_pct"]
            )
            item["alpha_1p5_request_weighted_delta_vs_family_anchor_pp"] = (
                item["alpha_1p5_request_weighted_slo_pct"]
                - anchor["alpha_1p5_request_weighted_slo_pct"]
            )
            item["util_delta_vs_family_anchor_pp"] = (
                item["mean_npu_utilization_pct"] - anchor["mean_npu_utilization_pct"]
            )
            if item["kind"] == "layer_once":
                rate = item["true_pressure_read_rate_hz_per_ssu"]
                anchor_rate = anchor["true_pressure_read_rate_hz_per_ssu"]
            elif item["kind"] == "adaptive":
                rate = item["cir_entry_write_rate_hz_per_ssu"]
                anchor_rate = anchor["cir_entry_write_rate_hz_per_ssu"]
            else:
                rate = anchor_rate = 0.0
            item["control_rate_reduction_vs_family_anchor_pct"] = (
                100.0 * (1.0 - rate / anchor_rate) if anchor_rate > 0.0 else 0.0
            )
            item.update(
                _common_request_slo(
                    rows[key],
                    rows[(item["family_anchor_case"], num_ssu)],
                )
            )
        else:
            item["slo_delta_vs_family_anchor_pp"] = None
            item["alpha_1p5_equal_npu_delta_vs_family_anchor_pp"] = None
            item["alpha_1p5_request_weighted_delta_vs_family_anchor_pp"] = None
            item["util_delta_vs_family_anchor_pp"] = None
            item["control_rate_reduction_vs_family_anchor_pct"] = None
            item.update(
                {
                    "common_with_anchor_request_count": 0,
                    "common_with_anchor_npu_count": 0,
                    "common_coverage_current_pct": 0.0,
                    "common_coverage_anchor_pct": 0.0,
                    "common_request_slo_pct": None,
                    "anchor_common_request_slo_pct": None,
                    "common_request_slo_delta_pp": None,
                    "common_equal_npu_slo_pct": None,
                    "anchor_common_equal_npu_slo_pct": None,
                    "common_equal_npu_slo_delta_pp": None,
                }
            )
        if static is not None:
            item["static_baseline_equal_npu_slo_pct"] = static["equal_npu_slo_pct"]
            item["static_baseline_alpha_1p5_equal_npu_slo_pct"] = static[
                "alpha_1p5_equal_npu_slo_pct"
            ]
            item["static_baseline_alpha_1p5_request_weighted_slo_pct"] = static[
                "alpha_1p5_request_weighted_slo_pct"
            ]
            item["static_baseline_mean_npu_utilization_pct"] = static[
                "mean_npu_utilization_pct"
            ]
            item["slo_delta_vs_static_baseline_pp"] = (
                item["equal_npu_slo_pct"] - static["equal_npu_slo_pct"]
            )
            item["alpha_1p5_equal_npu_delta_vs_static_baseline_pp"] = (
                item["alpha_1p5_equal_npu_slo_pct"]
                - static["alpha_1p5_equal_npu_slo_pct"]
            )
            item["alpha_1p5_request_weighted_delta_vs_static_baseline_pp"] = (
                item["alpha_1p5_request_weighted_slo_pct"]
                - static["alpha_1p5_request_weighted_slo_pct"]
            )
            item["util_delta_vs_static_baseline_pp"] = (
                item["mean_npu_utilization_pct"] - static["mean_npu_utilization_pct"]
            )
        else:
            item["static_baseline_equal_npu_slo_pct"] = None
            item["static_baseline_alpha_1p5_equal_npu_slo_pct"] = None
            item["static_baseline_alpha_1p5_request_weighted_slo_pct"] = None
            item["static_baseline_mean_npu_utilization_pct"] = None
            item["slo_delta_vs_static_baseline_pp"] = None
            item["alpha_1p5_equal_npu_delta_vs_static_baseline_pp"] = None
            item["alpha_1p5_request_weighted_delta_vs_static_baseline_pp"] = None
            item["util_delta_vs_static_baseline_pp"] = None
    return compact


def _ordered_keys(rows, case_specs, ssus):
    order = {name: index for index, name in enumerate(case_specs)}
    return sorted(rows, key=lambda key: (ssus.index(key[1]), order[key[0]]))


def _family_cases(case_specs, view: str) -> list[str]:
    if view == "ttl":
        return [
            name for name, case in case_specs.items() if case["kind"] == "layer_once"
        ]
    if view == "threshold":
        return [ADAPTIVE_ANCHOR] + [
            name for name, case in case_specs.items() if case["family"] == "threshold"
        ]
    if view == "interval":
        return [ADAPTIVE_ANCHOR] + [
            name for name, case in case_specs.items() if case["family"] == "interval"
        ]
    raise ValueError(view)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, payload) -> None:
    text = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    _atomic_write_text(path, text)


def _csv_text(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_csvs(output_dir: Path, compact, rows, ordered_keys) -> None:
    base_fields = [
        "case",
        "family",
        "kind",
        "num_ssu",
        "pressure_ttl_ms",
        "cir_write_threshold_gbps",
        "min_interval_ms",
        "family_anchor_case",
        "mean_npu_utilization_pct",
        "primary_slo_alpha",
        "sensitivity_slo_alpha",
        "equal_npu_slo_pct",
        "request_weighted_slo_pct",
        "alpha_1p5_equal_npu_slo_pct",
        "alpha_1p5_request_weighted_slo_pct",
        "alpha_1p5_equal_npu_change_vs_alpha_2_pp",
        "alpha_1p5_request_weighted_change_vs_alpha_2_pp",
        "slo_delta_vs_family_anchor_pp",
        "alpha_1p5_equal_npu_delta_vs_family_anchor_pp",
        "alpha_1p5_request_weighted_delta_vs_family_anchor_pp",
        "util_delta_vs_family_anchor_pp",
        "control_rate_reduction_vs_family_anchor_pct",
        "static_baseline_equal_npu_slo_pct",
        "static_baseline_alpha_1p5_equal_npu_slo_pct",
        "static_baseline_alpha_1p5_request_weighted_slo_pct",
        "static_baseline_mean_npu_utilization_pct",
        "slo_delta_vs_static_baseline_pp",
        "alpha_1p5_equal_npu_delta_vs_static_baseline_pp",
        "alpha_1p5_request_weighted_delta_vs_static_baseline_pp",
        "util_delta_vs_static_baseline_pp",
        "measurement_request_count",
        "measurement_requests_per_npu_min",
        "measurement_requests_per_npu_median",
        "measurement_requests_per_npu_max",
        "single_outcome_weight_at_min_sample_pp",
        "measurement_cohort_fingerprint",
        "realized_cohort_fleet_raw_demand_gbps",
        "realized_cohort_per_npu_demand_min_gbps",
        "realized_cohort_per_npu_demand_max_gbps",
        "realized_cohort_per_npu_demand_mean_gbps",
        "realized_cohort_per_npu_demand_cv",
        "realized_cohort_per_npu_ms_per_gb_min",
        "realized_cohort_per_npu_ms_per_gb_max",
        "common_with_anchor_request_count",
        "common_with_anchor_npu_count",
        "common_coverage_current_pct",
        "common_coverage_anchor_pct",
        "common_request_slo_pct",
        "anchor_common_request_slo_pct",
        "common_request_slo_delta_pp",
        "common_equal_npu_slo_pct",
        "anchor_common_equal_npu_slo_pct",
        "common_equal_npu_slo_delta_pp",
        "true_pressure_read_rate_hz_per_ssu",
        "true_pressure_read_rate_hz_max_ssu",
        "true_pressure_read_interval_ms_busiest_ssu",
        "pressure_cache_hit_fraction_pct",
        "cir_entry_write_rate_hz_per_ssu",
        "cir_entry_write_rate_hz_max_ssu",
        "cir_write_transaction_rate_hz_per_ssu",
        "cir_write_transaction_rate_hz_max_ssu",
        "cir_entries_per_transaction",
        "control_evaluation_rate_hz",
        "warmup_reached_ms",
        "measurement_start_ms",
        "tail_drain_ms",
        "fleet_queue_start_blocks",
        "fleet_queue_end_blocks",
        "fleet_queue_drift_blocks",
        "queue_start_by_ssu_json",
        "queue_end_by_ssu_json",
        "queue_drift_by_ssu_json",
        "max_abs_queue_drift_blocks",
        "block_utilization_range_pp",
        "block_first_last_utilization_delta_pp",
        "prefix_margin_at_fastest_npu",
        "fleet_raw_demand_gbps",
        "raw_capacity_knee_ssu",
        "global_raw_load_fraction",
        "max_ssu_raw_demand_gbps",
        "ssu_over_40_count",
        "max_actual_ssu_cir_gbps",
        "backing_prefix_requests_per_npu",
        "backing_fleet_raw_demand_gbps",
        "backing_raw_capacity_knee_ssu",
        "max_actual_npu_cir_gbps",
        "all_invariants_passed",
        "wall_time_s",
    ]
    group_fields = []
    for group_name, labels in (
        ("category", CATEGORIES),
        ("raw_demand_bins", DEMAND_BINS),
    ):
        for label in labels:
            group_fields.extend(
                f"{group_name}_{label}_{suffix}"
                for suffix in ("count", "passed", "slo_pct")
            )
    summary_rows = [compact[key] for key in ordered_keys]
    _atomic_write_text(
        output_dir / "summary.csv", _csv_text(summary_rows, base_fields + group_fields)
    )
    sensitivity_fields = (
        "case",
        "family",
        "kind",
        "num_ssu",
        "family_anchor_case",
        "measurement_cohort_fingerprint",
        "measurement_request_count",
        "measurement_requests_per_npu_min",
        "measurement_requests_per_npu_median",
        "measurement_requests_per_npu_max",
        "primary_slo_alpha",
        "sensitivity_slo_alpha",
        "equal_npu_slo_pct",
        "request_weighted_slo_pct",
        "alpha_1p5_equal_npu_slo_pct",
        "alpha_1p5_request_weighted_slo_pct",
        "alpha_1p5_equal_npu_change_vs_alpha_2_pp",
        "alpha_1p5_request_weighted_change_vs_alpha_2_pp",
        "alpha_1p5_equal_npu_delta_vs_family_anchor_pp",
        "alpha_1p5_request_weighted_delta_vs_family_anchor_pp",
        "static_baseline_alpha_1p5_equal_npu_slo_pct",
        "static_baseline_alpha_1p5_request_weighted_slo_pct",
        "alpha_1p5_equal_npu_delta_vs_static_baseline_pp",
        "alpha_1p5_request_weighted_delta_vs_static_baseline_pp",
    )
    _atomic_write_text(
        output_dir / "slo_alpha_sensitivity.csv",
        _csv_text(summary_rows, sensitivity_fields),
    )

    breakdown = []
    for key in ordered_keys:
        row = rows[key]
        for group_name, labels in (
            ("category", CATEGORIES),
            ("raw_demand_bins", DEMAND_BINS),
        ):
            group = row["cohort_profile_metrics"][group_name]
            for label in labels:
                metric = group.get(
                    label, {"count": 0, "passed": 0, "slo_attainment": None}
                )
                breakdown.append(
                    {
                        "case": key[0],
                        "num_ssu": key[1],
                        "group_type": group_name,
                        "group": label,
                        "count": metric["count"],
                        "passed": metric["passed"],
                        "slo_attainment_pct": None
                        if metric["slo_attainment"] is None
                        else 100.0 * float(metric["slo_attainment"]),
                    }
                )
    _atomic_write_text(
        output_dir / "slo_breakdown.csv",
        _csv_text(
            breakdown,
            (
                "case",
                "num_ssu",
                "group_type",
                "group",
                "count",
                "passed",
                "slo_attainment_pct",
            ),
        ),
    )


def _fmt(value, digits=2, suffix="") -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}{suffix}"


def _metric_cell(metric) -> str:
    if metric is None or int(metric.get("count", 0)) == 0:
        return "— (n=0)"
    return f"{100.0 * float(metric['slo_attainment']):.2f}% (n={int(metric['count'])})"


def _report_text(
    *,
    complete,
    expected_keys,
    missing_keys,
    shards,
    rows,
    compact,
    case_specs,
    ssus,
    pairing,
    source,
    config,
    spec,
) -> str:
    status = (
        "COMPLETE（42/42）" if complete else f"PARTIAL（{len(rows)}/{len(expected_keys)}）"
    )
    lines = [
        "# 毫秒级 QoS 控制实验报告",
        "",
        f"> 状态：**{status}**。`complete = {str(complete).lower()}`。",
        "",
        "本报告仅合并通过严格 source/config/spec、稳态 invariant 与同 SSU 输入配对校验的 shard。"
        "策略间的测量窗口请求 cohort 可能不同，因此主比较使用各自稳态总体指标；"
        "另以共同 request-ID 交集提供有覆盖率标注的 secondary sensitivity check。",
        "",
        "## 完整性与配对",
        "",
        f"- 输入 shard：{len(shards)} 个。",
        f"- 已获得结果：{len(rows)}/{len(expected_keys)}。",
        f"- source fingerprint：`{source}`。",
        f"- config fingerprint：`{config}`。",
        f"- workload：84-profile IID、有放回、每 NPU 独立；seed={spec['workload']['seed']}。"
        f"任务输入统计使用前 32 requests/NPU；仿真 backing prefix={spec['workload']['requests_per_npu']} 仅用于避免队列耗尽。",
    ]
    if missing_keys:
        lines.extend(
            [
                f"- 缺失结果：{len(missing_keys)} 个；本报告不得作为最终选型结论。",
                "",
                "缺失键：" + "、".join(f"{name}/SSU{ssu}" for name, ssu in missing_keys),
            ]
        )
    lines.extend(
        [
            "",
            "| SSU | available/expected | case set complete | finite input paired | distinct measurement cohorts |",
            "|---:|---:|:---:|:---:|---:|",
        ]
    )
    for ssu in ssus:
        item = pairing[str(ssu)]
        lines.append(
            f"| {ssu} | {item['available_count']}/{item['expected_count']} | "
            f"{item['complete_case_set']} | {item['all_available_rows_paired']} | "
            f"{item['distinct_measurement_cohort_count']} |"
        )

    lines.extend(
        [
            "",
            "## Workload 与实际容量检查",
            "",
            "下表的 workload 数字全部来自确定的前 32-request prefix，即任务要求的原始输入特征。"
            f"{spec['workload']['requests_per_npu']}-request backing prefix 只提供足够的饱和请求，不用来替换任务输入统计；"
            "measurement cohort 则由每行 `request_rows` 单独统计。`fleet raw demand` 是所有 NPU "
            "100% compute 时的反事实需求，`max SSU demand` 来自固定 placement。"
            "实际 CIR 最大值来自仿真中已经写入的表，而不是理论目标值。",
            "",
            "| SSU | prefix32 fleet D | global load | prefix32 max SSU D | >40 SSU | max actual SSU CIR | max actual NPU CIR | invariants |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for ssu in ssus:
        available = [compact[key] for key in compact if key[1] == ssu]
        if not available:
            lines.append(f"| {ssu} | — | — | — | — | — | — | — |")
            continue
        item = available[0]
        max_actual_ssu = max(value["max_actual_ssu_cir_gbps"] for value in available)
        npu_values = [
            value["max_actual_npu_cir_gbps"]
            for value in available
            if value["max_actual_npu_cir_gbps"] is not None
        ]
        max_actual_npu = max(npu_values) if npu_values else None
        lines.append(
            f"| {ssu} | {_fmt(item['fleet_raw_demand_gbps'])} GB/s | "
            f"{_fmt(100.0 * item['global_raw_load_fraction'])}% | "
            f"{_fmt(item['max_ssu_raw_demand_gbps'])} GB/s | {item['ssu_over_40_count']} | "
            f"{_fmt(max_actual_ssu)} GB/s | {_fmt(max_actual_npu)} GB/s | "
            f"{all(value['all_invariants_passed'] for value in available)} |"
        )

    lines.extend(
        [
            "",
            "### Closed-loop measurement cohort",
            "",
            "同 SSU 的有限输入、placement 与 simulator fingerprint 必须相同；"
            "但固定 wall-time 窗口中，不同策略会推进到不同 request-ID 集合。"
            "下面的 cohort fingerprint 因而允许不同，raw demand 也按每行实际 `request_rows` 单独计算。",
            "",
            "| SSU | case | cohort hash | requests | admissions/NPU min–median–max | one min-NPU outcome weight | realized fleet D | per-NPU D range | CV |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        item = compact[key]
        lines.append(
            f"| {key[1]} | `{key[0]}` | `{item['measurement_cohort_fingerprint'][:12]}…` | "
            f"{item['measurement_request_count']} | "
            f"{item['measurement_requests_per_npu_min']}–"
            f"{_fmt(item['measurement_requests_per_npu_median'], 1)}–"
            f"{item['measurement_requests_per_npu_max']} | "
            f"{_fmt(item['single_outcome_weight_at_min_sample_pp'], 3)} pp | "
            f"{_fmt(item['realized_cohort_fleet_raw_demand_gbps'])} GB/s | "
            f"{_fmt(item['realized_cohort_per_npu_demand_min_gbps'])}–"
            f"{_fmt(item['realized_cohort_per_npu_demand_max_gbps'])} | "
            f"{_fmt(100.0 * item['realized_cohort_per_npu_demand_cv'])}% |"
        )

    lines.extend(
        [
            "",
            "## Static baseline 绝对锚点",
            "",
            "Static baseline 不属于 TTL 或 Adaptive 的降频曲线；它只提供绝对 SLO/利用率参照。",
            "",
            "| SSU | equal-NPU SLO | request-weighted SLO | NPU util | SSD util | requests |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ssu in ssus:
        item = compact.get((STATIC_BASELINE, ssu))
        if item is None:
            lines.append(f"| {ssu} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {ssu} | {_fmt(item['equal_npu_slo_pct'])}% | "
                f"{_fmt(item['request_weighted_slo_pct'])}% | "
                f"{_fmt(item['mean_npu_utilization_pct'])}% | "
                f"{_fmt(item['ssd_mean_utilization_pct'])}% | {item['measurement_request_count']} |"
            )

    lines.extend(
        [
            "",
            "## TTFT SLO α=1.5 后处理敏感性",
            "",
            "α=2.0 仍是正式主指标。这里不重跑仿真，也不改变任何策略的 wall-time measurement cohort；"
            "仅对每行原有 `request_rows` 按 `ttft_ms <= 1.5 × ideal_ttft_ms`（含与仿真一致的数值 epsilon）重新判定。"
            "每个 NPU 至少有一个窗内样本仍是硬校验，因此同时给出 equal-NPU 与 request-weighted 两种口径；"
            "利用率、控制率及队列指标不会因该后处理阈值而改变。后续 category、raw-demand-bin 表和 `slo_breakdown.csv` "
            "仍采用 α=2.0 主定义。",
            "",
            "| SSU | case | requests (min/NPU) | α2 equal-NPU | α1.5 equal-NPU | Δ(1.5−2.0) | α2 req-weighted | α1.5 req-weighted | Δα1.5 vs family anchor | Δα1.5 vs static |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        item = compact[key]
        lines.append(
            f"| {key[1]} | `{key[0]}` | {item['measurement_request_count']} "
            f"({item['measurement_requests_per_npu_min']}) | "
            f"{_fmt(item['equal_npu_slo_pct'])}% | "
            f"{_fmt(item['alpha_1p5_equal_npu_slo_pct'])}% | "
            f"{_fmt(item['alpha_1p5_equal_npu_change_vs_alpha_2_pp'], suffix=' pp')} | "
            f"{_fmt(item['request_weighted_slo_pct'])}% | "
            f"{_fmt(item['alpha_1p5_request_weighted_slo_pct'])}% | "
            f"{_fmt(item['alpha_1p5_equal_npu_delta_vs_family_anchor_pp'], suffix=' pp')} | "
            f"{_fmt(item['alpha_1p5_equal_npu_delta_vs_static_baseline_pp'], suffix=' pp')} |"
        )

    views = (
        (
            "TTL：真实压力表读取率",
            "ttl",
            "pressure_ttl_ms",
            "true_pressure_read_rate_hz_per_ssu",
            "true_pressure_read_rate_hz_max_ssu",
            "TTL (ms)",
        ),
        (
            "CIR threshold：真实 entry 写率",
            "threshold",
            "cir_write_threshold_gbps",
            "cir_entry_write_rate_hz_per_ssu",
            "cir_entry_write_rate_hz_max_ssu",
            "threshold (GB/s)",
        ),
        (
            "Adaptive interval：真实 entry 写率",
            "interval",
            "min_interval_ms",
            "cir_entry_write_rate_hz_per_ssu",
            "cir_entry_write_rate_hz_max_ssu",
            "interval (ms)",
        ),
    )
    for title, view, config_field, rate_field, max_rate_field, config_label in views:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"变化量均相对该机制自身 anchor：`{TTL_ANCHOR if view == 'ttl' else ADAPTIVE_ANCHOR}`。",
                "",
                f"| SSU | case | {config_label} | fleet-mean /s/SSU | busiest SSU /s | rate reduction | SLO | ΔSLO | util | Δutil | common IDs | current coverage | common ΔSLO* |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for ssu in ssus:
            for name in _family_cases(case_specs, view):
                item = compact.get((name, ssu))
                if item is None:
                    continue
                lines.append(
                    f"| {ssu} | `{name}` | {_fmt(item[config_field], 3)} | "
                    f"{_fmt(item[rate_field], 2)} | "
                    f"{_fmt(item[max_rate_field], 2)} | "
                    f"{_fmt(item['control_rate_reduction_vs_family_anchor_pct'])}% | "
                    f"{_fmt(item['equal_npu_slo_pct'])}% | "
                    f"{_fmt(item['slo_delta_vs_family_anchor_pp'])} pp | "
                    f"{_fmt(item['mean_npu_utilization_pct'])}% | "
                    f"{_fmt(item['util_delta_vs_family_anchor_pp'])} pp | "
                    f"{item['common_with_anchor_request_count']} | "
                    f"{_fmt(item['common_coverage_current_pct'])}% | "
                    f"{_fmt(item['common_request_slo_delta_pp'])} pp |"
                )
        lines.extend(
            [
                "",
                "\\* `common ΔSLO` 是当前行与自身 anchor 的 request-ID 交集上的 request-weighted secondary 指标；"
                "它受交集选择影响，不能替代各自 wall-time cohort 的主 SLO。",
            ]
        )

    lines.extend(
        [
            "",
            "## 控制面真实动作频率总表",
            "",
            "这张表把三种不同单位分开：pressure read 是每块 SSU 的真实压力表读取；"
            "evaluation 是 fleet controller 的决策次数；transaction 是某块 SSU 发生一次 CIR 表写事务；"
            "entry write 是事务中真正改写的 Path CIR 寄存器数。不能把 entry/s 当成 transaction/s，"
            "也不能把 fleet evaluation/s 再除以 SSU 数。mean 与 max 均按 SSU 统计，便于同时看平均盘和最忙盘。",
            "",
            "| SSU | case | pressure reads/s/SSU mean–max | fleet eval/s | CIR tx/s/SSU mean–max | CIR entries/s/SSU mean–max | entries/tx |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        item = compact[key]
        lines.append(
            f"| {key[1]} | `{key[0]}` | "
            f"{_fmt(item['true_pressure_read_rate_hz_per_ssu'], 2)}–"
            f"{_fmt(item['true_pressure_read_rate_hz_max_ssu'], 2)} | "
            f"{_fmt(item['control_evaluation_rate_hz'], 2)} | "
            f"{_fmt(item['cir_write_transaction_rate_hz_per_ssu'], 2)}–"
            f"{_fmt(item['cir_write_transaction_rate_hz_max_ssu'], 2)} | "
            f"{_fmt(item['cir_entry_write_rate_hz_per_ssu'], 2)}–"
            f"{_fmt(item['cir_entry_write_rate_hz_max_ssu'], 2)} | "
            f"{_fmt(item['cir_entries_per_transaction'], 2)} |"
        )

    lines.extend(
        [
            "",
            "## Category SLO",
            "",
            "单元格为 SLO（样本数）。小样本只能作描述，不能据此排序策略。",
            "",
            "| SSU | case | SS | SL | LS | LL |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        metrics = rows[key]["cohort_profile_metrics"]["category"]
        cells = [_metric_cell(metrics.get(category)) for category in CATEGORIES]
        lines.append(f"| {key[1]} | `{key[0]}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Raw-demand bin SLO",
            "",
            "| SSU | case | ≤10 | 10–20 | 20–40 | 40–50 | 50–80 | >80 GB/s |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        metrics = rows[key]["cohort_profile_metrics"]["raw_demand_bins"]
        cells = [_metric_cell(metrics.get(name)) for name in DEMAND_BINS]
        lines.append(f"| {key[1]} | `{key[0]}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Warm、queue drift 与 prefix 余量",
            "",
            "fleet queue drift 可能因不同 SSU 的正负变化互相抵消，因此同时报告 per-SSU drift 与最大绝对 drift。",
            "",
            "| SSU | case | warm reached | measurement start | tail drain | fleet queue start→end (Δ) | per-SSU Δ | max abs Δ | block util range | fastest prefix margin |",
            "|---:|---|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for key in _ordered_keys(rows, case_specs, ssus):
        item = compact[key]
        lines.append(
            f"| {key[1]} | `{key[0]}` | {_fmt(item['warmup_reached_ms'], 1)} ms | "
            f"{_fmt(item['measurement_start_ms'], 1)} ms | {_fmt(item['tail_drain_ms'], 1)} ms | "
            f"{item['fleet_queue_start_blocks']}→{item['fleet_queue_end_blocks']} "
            f"({item['fleet_queue_drift_blocks']:+d}) | "
            f"`{item['queue_drift_by_ssu_json']}` | "
            f"{item['max_abs_queue_drift_blocks']} | "
            f"{_fmt(item['block_utilization_range_pp'])} pp | "
            f"{item['prefix_margin_at_fastest_npu']} |"
        )

    lines.extend(
        [
            "",
            "## 解释纪律",
            "",
            "- TTL 横轴是真实 SSU 压力表读取次数率，不含 cache hit；threshold/interval 横轴是真实 CIR entry 写率，不是 evaluation 或 transaction 数。",
            "- 曲线横轴是 fleet-mean /s/SSU；判断每块盘是否满足频率下限时必须看表中的 busiest SSU，不能用均值代替最坏盘。",
            "- TTL 与 CIR 写是不同硬件动作，未给定真实单次成本前，不应把二者的“下降百分比”直接横向换算。",
            "- `ΔSLO/Δutil` 使用各自机制 anchor；static baseline 只作为绝对锚点。",
            "- 固定时间窗下，各策略完成的 category/profile cohort 可能不同，所以同时给出 category 和 demand-bin 样本数。",
            "- `all_npus_sampled_for_slo` 只保证每个 NPU 至少 1 个 admission，使 equal-NPU SLO 有定义；若按 2 pp 选型，建议校准到每 NPU 至少 4 个样本。min=1 时单个 outcome 可改变 equal-NPU SLO 3.125 pp。",
            "- equal-NPU SLO 是策略公平性的主指标；request-weighted SLO 只在全 NPU 已覆盖后作为吞吐加权辅指标，不能用来挽救缺失 NPU 的窗口。",
            "- α=2.0 是主 SLO 定义；α=1.5 只是在完全相同 request cohort 上重分类的更严格敏感性分析，不能与另一行不同 cohort 的结果伪装成逐请求配对因果效应。",
            "- `measurement_cohort_fingerprint` 不要求跨策略相同；finite-input fingerprints 必须相同。"
            "common request-ID SLO 只作 secondary sensitivity check，并同时报告交集覆盖率。",
            "- `complete=false` 时，曲线和表格只用于检查趋势与异常，不得形成最终配置推荐。",
            "",
            "生成文件：`results.json`、`summary.csv`、`slo_alpha_sensitivity.csv`、"
            "`slo_breakdown.csv`、三张控制率曲线。",
            "",
        ]
    )
    return "\n".join(lines)


def _save_figure_atomic(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    temporary.replace(path)
    plt.close(fig)


def _plot_family(
    output_dir: Path, *, view: str, compact, case_specs, ssus, complete: bool
) -> None:
    if view == "ttl":
        rate_field = "true_pressure_read_rate_hz_per_ssu"
        x_label = "Fleet-mean true pressure-table reads / s / SSU"
        knob_field = "pressure_ttl_ms"
        knob_label = "TTL"
        filename = "ttl_true_read_rate_vs_quality.png"
    elif view == "threshold":
        rate_field = "cir_entry_write_rate_hz_per_ssu"
        x_label = "Fleet-mean true CIR entry writes / s / SSU"
        knob_field = "cir_write_threshold_gbps"
        knob_label = "threshold"
        filename = "threshold_entry_write_rate_vs_quality.png"
    else:
        rate_field = "cir_entry_write_rate_hz_per_ssu"
        x_label = "Fleet-mean true CIR entry writes / s / SSU"
        knob_field = "min_interval_ms"
        knob_label = "interval"
        filename = "interval_entry_write_rate_vs_quality.png"

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    cases = _family_cases(case_specs, view)
    any_rows = False
    for ssu in ssus:
        points = [compact[(name, ssu)] for name in cases if (name, ssu) in compact]
        if not points:
            continue
        any_rows = True
        points.sort(key=lambda row: (row[rate_field], row[knob_field]))
        x = [row[rate_field] for row in points]
        color = FAMILY_COLORS.get(ssu)
        axes[0].plot(
            x,
            [row["equal_npu_slo_pct"] for row in points],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"SSU{ssu}",
        )
        axes[1].plot(
            x,
            [row["mean_npu_utilization_pct"] for row in points],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"SSU{ssu}",
        )
        for axis, metric in (
            (axes[0], "equal_npu_slo_pct"),
            (axes[1], "mean_npu_utilization_pct"),
        ):
            for row in points:
                axis.annotate(
                    f"{knob_label}={row[knob_field]:g}",
                    (row[rate_field], row[metric]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )
        static = compact.get((STATIC_BASELINE, ssu))
        if static is not None and x:
            axes[0].axhline(
                static["equal_npu_slo_pct"], color=color, linestyle=":", alpha=0.25
            )
            axes[1].axhline(
                static["mean_npu_utilization_pct"],
                color=color,
                linestyle=":",
                alpha=0.25,
            )

    if not any_rows:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No available rows",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    axes[0].set_ylabel("Equal-NPU TTFT SLO (%)")
    axes[1].set_ylabel("Mean NPU utilization (%)")
    axes[1].set_xlabel(x_label)
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    for axis in axes:
        axis.grid(True, alpha=0.25)
    if any_rows:
        axes[0].legend(loc="best")
    fig.suptitle(
        f"{view.upper()} actual control rate vs quality — {'COMPLETE' if complete else 'PARTIAL'}"
    )
    fig.tight_layout()
    _save_figure_atomic(fig, output_dir / filename)


def _build_merged_payload(
    *,
    complete,
    expected_keys,
    missing_keys,
    shards,
    rows,
    provenance,
    compact,
    pairing,
    spec,
    source,
    config,
):
    ordered_keys = _ordered_keys(
        rows,
        {case["name"]: case for case in spec["cases"]},
        tuple(spec["default_ssu_list"]),
    )
    raw_results = []
    for key in ordered_keys:
        raw = dict(rows[key])
        raw["analysis_source_shards"] = provenance[key]
        raw_results.append(raw)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "ms_scale_control_analysis_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "expected_result_count": len(expected_keys),
        "result_count": len(rows),
        "missing_results": [[name, ssu] for name, ssu in missing_keys],
        "source_fingerprint": source,
        "config_fingerprint": config,
        "experiment_spec": spec,
        "input_shards": [shard["_analysis_path"] for shard in shards],
        "pairing_audit": pairing,
        "analysis_rows": [compact[key] for key in ordered_keys],
        "results": raw_results,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path, help="runner shard JSON files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ms_scale_control"),
        help="directory for merged JSON, report, CSV, and plots",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="write partial diagnostics, then return a non-zero status unless all 42 rows exist",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    resolved_paths = [path.expanduser().resolve() for path in args.shards]
    _require(
        len(resolved_paths) == len(set(resolved_paths)),
        "the same shard path was supplied twice",
    )
    shards = [_read_shard(path) for path in resolved_paths]
    for shard in shards:
        _validate_shard_root(shard)
    spec, case_specs, ssus, source, config = _validate_shard_consistency(shards)
    rows, provenance = _collect_rows(shards, case_specs, ssus, source, config, spec)
    pairing = _pairing_audit(rows, case_specs, ssus, spec)

    expected_keys = tuple((name, ssu) for ssu in ssus for name in case_specs)
    missing_keys = tuple(key for key in expected_keys if key not in rows)
    complete = (
        len(rows) == EXPECTED_RESULT_COUNT
        and not missing_keys
        and all(pairing[str(ssu)]["complete_case_set"] for ssu in ssus)
        and all(pairing[str(ssu)]["all_available_rows_paired"] for ssu in ssus)
    )
    compact = _compact_rows(rows, case_specs, ssus, spec)
    ordered_keys = _ordered_keys(rows, case_specs, ssus)
    output_dir = args.output_dir.expanduser().resolve()
    merged = _build_merged_payload(
        complete=complete,
        expected_keys=expected_keys,
        missing_keys=missing_keys,
        shards=shards,
        rows=rows,
        provenance=provenance,
        compact=compact,
        pairing=pairing,
        spec=spec,
        source=source,
        config=config,
    )
    _atomic_write_json(output_dir / "results.json", merged)
    _write_csvs(output_dir, compact, rows, ordered_keys)
    report = _report_text(
        complete=complete,
        expected_keys=expected_keys,
        missing_keys=missing_keys,
        shards=shards,
        rows=rows,
        compact=compact,
        case_specs=case_specs,
        ssus=ssus,
        pairing=pairing,
        source=source,
        config=config,
        spec=spec,
    )
    _atomic_write_text(output_dir / "report.md", report)
    for view in ("ttl", "threshold", "interval"):
        _plot_family(
            output_dir,
            view=view,
            compact=compact,
            case_specs=case_specs,
            ssus=ssus,
            complete=complete,
        )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "complete": complete,
                "result_count": len(rows),
                "expected_result_count": len(expected_keys),
                "missing_count": len(missing_keys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        raise SystemExit(f"validation error: {error}") from error
