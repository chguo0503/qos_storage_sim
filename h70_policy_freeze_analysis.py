"""Formal, simulation-free audit and policy freeze for 32-NPU PFO H70.

The analyzer accepts one or more schema-3 ``ms_scale_control_experiment.py``
JSON shards.  It does not import the runner, the simulator, or a controller.
It independently authenticates the frozen campaign document, source closure,
configuration and input pairing, recomputes request/SLO/resource/stationarity
metrics, and replays the observable PFO audit ledger.  A successful complete
campaign produces the 25-field policy-freeze artifact consumed by
``npu128_scale_analysis.py``.

No simulation is started by this program.  ``--self-test`` exercises only
pure analysis and failure-injection paths in memory.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping, Sequence


ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_NAME = "h70_policy_freeze_analysis_v1"
SHARD_SCHEMA_VERSION = 3
SUMMARY_SCHEMA_VERSION = 2
CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_NAME = "pfo_h70_policy_freeze32_v1"
EXPERIMENT = "32npu_protected_floor_h70_freeze_v2"
DEFINITION = "pfo32"
NUM_NPU = 32
N_LAYERS = 16
BATCH_SIZE = 1
FORMAL_SEEDS = (42, 43, 44)
FORMAL_SSUS = (6, 10, 18)
ROLE_ORDER = ("B", "A0", "H70")
ROLE_TO_CASE = {
    "B": "baseline",
    "A0": "adaptive_t0_i25ms",
    "H70": "pfo_astar_h70",
}
CASE_TO_ROLE = {case: role for role, case in ROLE_TO_CASE.items()}
SELECTION_RULE_VERSION = "h70-confirm32-preregistered-freeze-v1"
STATIONARITY_RULE_VERSION = "h70-confirm32-preregistered-stationarity-v1"
EXPECTED_STATIONARITY_SEMANTICS = (
    "read-only left-limit snapshot before workload events at the same time"
)
EXPECTED_CONTROL_WINDOW = "half-open [measurement_start_ms, measurement_end_ms)"
PRIMARY_ALPHA = 2.0
SENSITIVITY_ALPHA = 1.5
SLO_EPSILON = 1e-12
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
FORECAST_HOT_FRACTION = 0.70
FORECAST_REQUESTS_PER_NPU = 32
MIN_FORMAL_MEASUREMENT_MS = 8_000.0
FORMAL_BLOCK_MS = 500.0
FORMAL_WARMUP_REQUESTS = 8
FORMAL_SETTLE_MS = 500.0
MIN_FORMAL_BACKING = 128
MIN_BACKING_MARGIN = 32
MIN_REQUESTS_PER_NPU = 8
UTIL_HALF_LIMIT_PP = 1.0
UTIL_TREND_LIMIT_PP = 2.0
SERVED_HALF_RELATIVE_LIMIT = 0.02
UTIL_NONINFERIOR_MARGIN_PP = 0.5
SLO_NONINFERIOR_MARGIN_PP = 1.0
P99_NONINFERIOR_RELATIVE_MARGIN = 0.01
WRITE_REDUCTION_MIN = 0.50
FAR_BETTER_SLO_PP = 5.0
SHA256_HEX = frozenset("0123456789abcdef")

EXPECTED_CASES = {
    "baseline": {
        "name": "baseline",
        "family": "pfo32",
        "kind": "baseline",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 0.0,
    },
    "adaptive_t0_i25ms": {
        "name": "adaptive_t0_i25ms",
        "family": "pfo32",
        "kind": "adaptive",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 25.0,
    },
    "pfo_astar_h70": {
        "name": "pfo_astar_h70",
        "family": "pfo32",
        "kind": "pfo",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 25.0,
        "pfo_internal_deadband_gbps": 0.05,
        "forecast_hot_fraction": 0.70,
        "forecast_requests_per_npu": 32,
    },
}

EXPECTED_INVARIANTS = frozenset(
    {
        "warmup_reached_all_npus",
        "measurement_window_closed",
        "measurement_duration_exact",
        "all_npus_sampled_for_slo",
        "all_tagged_requests_completed",
        "tagged_admissions_inside_window",
        "all_window_admissions_tagged",
        "no_backlog_exhaustion",
        "compute_overlap_bounds",
        "ssd_overlap_bounds",
        "ssd_service_attribution",
        "npu_link_overlap_bounds",
        "npu_link_service_attribution",
        "one_command_per_ssd",
        "cir_capacity",
        "actual_cir_per_ssu_capacity",
        "actual_cir_per_npu_capacity",
        "cir_entry_write_counter_consistency",
        "stationarity_boundary_count",
        "stationarity_boundary_times_exact",
        "stationarity_snapshot_shapes",
        "stationarity_values_nonnegative",
        "stationarity_cumulative_monotonic",
        "stationarity_cumulative_service_consistency",
        "stationarity_block_sums_match_whole",
        "stationarity_independent_compute_match",
        "stationarity_legacy_edges_match",
        "stationarity_whole_window_match",
        "stationarity_block_resource_bounds",
    }
)

INPUT_FINGERPRINT_FIELDS = frozenset(
    {
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
    }
)
CROSS_SSU_INPUT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
)

SNAPSHOT_FIELDS = frozenset(
    {
        "boundary",
        "time_ms",
        "ssd_cumulative_busy_ms_by_ssu",
        "ssd_cumulative_served_gb_by_ssu",
        "ssd_outstanding_blocks_by_ssu",
        "ssd_outstanding_gb_by_ssu",
        "npu_compute_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_served_gb_by_npu",
        "npu_link_outstanding_blocks_by_npu",
        "npu_link_outstanding_gb_by_npu",
    }
)

BLOCK_FIELDS = frozenset(
    {
        "block",
        "start_ms",
        "end_ms",
        "duration_ms",
        "npu_utilization",
        "request_count",
        "request_weighted_slo_attainment",
        "ssd_busy_ms_by_ssu",
        "ssd_served_gb_by_ssu",
        "ssd_utilizations",
        "ssd_mean_utilization",
        "ssd_outstanding_blocks_at_start",
        "ssd_outstanding_blocks_at_end",
        "ssd_outstanding_blocks_delta",
        "ssd_outstanding_gb_at_start",
        "ssd_outstanding_gb_at_end",
        "ssd_outstanding_gb_delta",
        "compute_ms_by_npu",
        "npu_utilizations",
        "npu_link_busy_ms_by_npu",
        "npu_link_served_gb_by_npu",
        "npu_link_utilizations",
        "npu_link_mean_utilization",
        "npu_link_outstanding_blocks_at_start",
        "npu_link_outstanding_blocks_at_end",
        "npu_link_outstanding_blocks_delta",
        "npu_link_outstanding_gb_at_start",
        "npu_link_outstanding_gb_at_end",
        "npu_link_outstanding_gb_delta",
    }
)

AUDIT_RECORD_FIELDS = frozenset(
    {
        "time_ms",
        "evaluation",
        "pre_state_hash",
        "ideal_state_hash",
        "required_state_hash",
        "install_state_hash",
        "changed_entries",
        "changed_entries_hash",
        "decrease_writes",
        "increase_writes",
        "change_reason_counts",
        "ordered_sequence_capacity_safe",
        "maximum_ordered_prefix_ssu_excess_gbps",
        "maximum_ordered_prefix_npu_excess_gbps",
        "maximum_required_floor_shortfall_gbps",
        "active_set_hold_violations",
        "maximum_deadband_held_delta_gbps",
        "post_des_state_verified",
        "post_des_verification",
        "materialized_ssu_mask",
        "cold_ssu_install_zero",
        "writes_by_ssu",
        "transactions_by_ssu",
        "decrease_transactions_by_ssu",
        "increase_transactions_by_ssu",
        "safety_forced_writes_by_ssu",
        "safety_forced_writes",
        "required_floor_increases",
        "capacity_compensation_decreases",
        "deadband_holds",
        "selected_npu_count",
    }
)


class ValidationError(ValueError):
    """Raised when an artifact is unsafe for a report-ready H70 freeze."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def _canonical_hash(value, namespace: bytes = b"") -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(namespace + encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in SHA256_HEX for character in value)
    )


def _finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"{name} is not finite: {number!r}")
    return number


def _nonnegative(value, name: str, *, tolerance: float = 1e-8) -> float:
    number = _finite(value, name)
    _require(number >= -tolerance, f"{name} is negative: {number!r}")
    return max(0.0, number)


def _fraction(value, name: str) -> float:
    number = _finite(value, name)
    _require(-1e-8 <= number <= 1.0 + 1e-8, f"{name} is outside [0,1]")
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


def _close(
    actual,
    expected,
    name: str,
    *,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-10,
) -> None:
    left = _finite(actual, name)
    right = _finite(expected, f"{name} expected")
    _require(
        math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{name} mismatch: {left!r} != {right!r}",
    )


def _vector(
    value,
    length: int,
    name: str,
    *,
    integer: bool = False,
    fraction: bool = False,
) -> list:
    _require(isinstance(value, list) and len(value) == length, f"{name} shape")
    if integer:
        return [_integer(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if fraction:
        return [_fraction(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return [_nonnegative(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _compare_vectors(actual: Sequence, expected: Sequence, name: str) -> None:
    _require(len(actual) == len(expected), f"{name} length")
    for index, (left, right) in enumerate(zip(actual, expected)):
        _close(left, right, f"{name}[{index}]")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile requires samples")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _theil_sen_slope(times_ms: Sequence[float], values: Sequence[float]) -> float:
    _require(len(times_ms) == len(values) and len(values) >= 2, "trend shape")
    slopes = [
        (float(values[right]) - float(values[left]))
        / ((float(times_ms[right]) - float(times_ms[left])) / 1000.0)
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if float(times_ms[right]) > float(times_ms[left])
    ]
    _require(bool(slopes), "stationarity times are not increasing")
    return statistics.median(slopes)


def _equivalent_derived(actual, expected, *, tolerance: float = 1e-10) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=tolerance
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent_derived(left, right, tolerance=tolerance)
            for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _equivalent_derived(actual[key], expected[key], tolerance=tolerance)
            for key in actual
        )
    return type(actual) is type(expected) and actual == expected


def _read_json(path: Path) -> tuple[dict, bytes]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise ValidationError(f"cannot read {resolved}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON in {resolved}: {error}") from error
    _require(isinstance(payload, dict), f"{resolved}: root must be an object")
    return payload, raw


def _read_campaign_spec(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    spec, raw = _read_json(resolved)
    expected_fields = {
        "schema_version",
        "campaign",
        "experiment",
        "definition",
        "num_npu",
        "seeds",
        "ssus",
        "roles",
        "backing_requests_per_npu",
        "warmup_requests_per_npu",
        "settle_ms",
        "measurement_ms",
        "block_ms",
        "definition_fingerprint",
        "source_fingerprint",
        "analysis_source_sha256",
        "stationarity_rule_version",
        "selection_rule_version",
    }
    _require(set(spec) == expected_fields, "formal campaign spec fields changed")
    _require(
        spec["schema_version"] == CAMPAIGN_SCHEMA_VERSION
        and spec["campaign"] == CAMPAIGN_NAME
        and spec["experiment"] == EXPERIMENT
        and spec["definition"] == DEFINITION
        and spec["num_npu"] == NUM_NPU,
        "formal campaign identity/topology differs",
    )
    _require(tuple(spec["seeds"]) == FORMAL_SEEDS, "formal seed plan differs")
    _require(tuple(spec["ssus"]) == FORMAL_SSUS, "formal SSU plan differs")
    _require(spec["roles"] == ROLE_TO_CASE, "formal role/case plan differs")
    _require(
        spec["selection_rule_version"] == SELECTION_RULE_VERSION,
        "formal selection rule differs",
    )
    _require(
        spec["stationarity_rule_version"] == STATIONARITY_RULE_VERSION,
        "formal stationarity rule differs",
    )
    backing = _integer(
        spec["backing_requests_per_npu"], "formal backing", minimum=MIN_FORMAL_BACKING
    )
    warmup = _integer(spec["warmup_requests_per_npu"], "formal warmup", minimum=1)
    _require(warmup == FORMAL_WARMUP_REQUESTS, "formal warmup must be 8")
    settle_ms = _finite(spec["settle_ms"], "formal settle_ms")
    _close(settle_ms, FORMAL_SETTLE_MS, "formal settle_ms")
    measurement_ms = _finite(spec["measurement_ms"], "formal measurement_ms")
    _require(
        measurement_ms >= MIN_FORMAL_MEASUREMENT_MS,
        "formal measurement is shorter than 8000 ms",
    )
    block_ms = _finite(spec["block_ms"], "formal block_ms")
    _close(block_ms, FORMAL_BLOCK_MS, "formal block_ms")
    ratio = measurement_ms / block_ms
    block_count = int(round(ratio))
    _require(
        math.isclose(ratio, block_count, rel_tol=0.0, abs_tol=1e-12)
        and block_count >= 16
        and block_count % 2 == 0,
        "measurement must contain an even number of at least 16 full blocks",
    )
    _require(backing - warmup >= MIN_BACKING_MARGIN, "formal backing is too short")
    for name in (
        "definition_fingerprint",
        "source_fingerprint",
        "analysis_source_sha256",
    ):
        _require(_is_sha256(spec[name]), f"formal {name} is malformed")
    _require(
        spec["analysis_source_sha256"] == _file_sha256(Path(__file__).resolve()),
        "campaign names a different analyzer source",
    )
    return {
        **spec,
        "path": str(resolved),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "file_size_bytes": len(raw),
        "block_count": block_count,
        "boundary_count": block_count + 1,
        "expected_result_count": len(FORMAL_SEEDS)
        * len(FORMAL_SSUS)
        * len(ROLE_ORDER),
    }


def _parse_local_data(path: Path) -> tuple[dict, list[list]]:
    try:
        raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValidationError(f"cannot directly parse authenticated data: {error}") from error
    _require(isinstance(raw, dict) and raw, "data must contain profiles")
    table = {}
    for raw_key, raw_values in raw.items():
        key_value = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        _require(
            isinstance(key_value, (tuple, list)) and len(key_value) == 2,
            f"malformed profile key {raw_key!r}",
        )
        key = (
            _integer(key_value[0], "seq_len_k", minimum=1),
            _integer(key_value[1], "nql", minimum=1),
        )
        _require(key not in table, f"duplicate profile {key}")
        _require(isinstance(raw_values, (tuple, list)), f"profile {key} values")
        values = tuple(raw_values)
        if len(values) == 3:
            required_bw, per_layer_us, ttft_ms = values
            values = (
                required_bw,
                per_layer_us,
                ttft_ms,
                required_bw * per_layer_us / 1e6,
            )
        _require(len(values) == 4, f"profile {key} must have 3/4 values")
        converted = tuple(_finite(value, f"profile {key}") for value in values)
        _require(all(value > 0.0 for value in converted), f"profile {key} nonpositive")
        table[key] = converted
    rows = [[key[0], key[1], list(table[key])] for key in sorted(table)]
    return table, rows


def _discover_source_closure(root: Path) -> tuple[str, ...]:
    pending = ["ms_scale_control_experiment.py"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / name
        _require(path.is_file(), f"source closure file missing: {name}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise ValidationError(f"cannot parse source closure {name}: {error}") from error
        seen.add(name)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in modules:
            candidate = module.split(".", 1)[0] + ".py"
            if (root / candidate).is_file() and candidate not in seen:
                pending.append(candidate)
    return tuple(sorted(seen))


def _validate_local_source_closure(payloads: Sequence[dict], source_root: Path) -> dict:
    root = source_root.expanduser().resolve()
    _require(root.is_dir(), f"source root is not a directory: {root}")
    manifests = [payload.get("source_manifest") for payload in payloads]
    _require(
        manifests and isinstance(manifests[0], dict) and manifests[0],
        "source manifest missing",
    )
    _require(all(value == manifests[0] for value in manifests), "source manifests differ")
    manifest = manifests[0]
    expected_names = set(_discover_source_closure(root)) | {"data"}
    _require(set(manifest) == expected_names, "source manifest/import closure differs")
    for relative_name, expected_hash in manifest.items():
        _require(
            isinstance(relative_name, str)
            and relative_name
            and not Path(relative_name).is_absolute()
            and ".." not in Path(relative_name).parts,
            f"unsafe source path {relative_name!r}",
        )
        _require(_is_sha256(expected_hash), f"malformed source hash for {relative_name}")
        local = (root / relative_name).resolve()
        try:
            local.relative_to(root)
        except ValueError as error:
            raise ValidationError(f"source path escapes root: {relative_name}") from error
        _require(local.is_file(), f"source file absent: {relative_name}")
        _require(_file_sha256(local) == expected_hash, f"source changed: {relative_name}")
    source_fingerprint = _canonical_hash(manifest, b"ms-scale-control-source:v1\0")
    table, catalog_rows = _parse_local_data(root / "data")
    _require(len(table) == 84, "authenticated data must have 84 profiles")
    catalog_hash_rows = [[[row[0], row[1]], row[2]] for row in catalog_rows]
    catalog_hash = _canonical_hash(
        catalog_hash_rows, b"random-steady-state:data-catalog:v1\0"
    )
    table_hash = _canonical_hash(catalog_rows, b"authenticated-bw-table:v1\0")
    data_hash = _file_sha256(root / "data")
    authentication = payloads[0].get("input_authentication")
    expected_auth = {
        "source": "data",
        "source_sha256": data_hash,
        "catalog_hash": catalog_hash,
        "table_fingerprint": table_hash,
        "profile_count": len(table),
    }
    _require(authentication == expected_auth, "local data authentication differs")
    for payload in payloads:
        _require(payload.get("input_authentication") == expected_auth, "input auth differs")
        schedule = payload.get("schedule_metadata")
        _require(
            isinstance(schedule, dict) and schedule.get("catalog_rows") == catalog_rows,
            "embedded catalog differs from local data",
        )
    return {
        "source_root": str(root),
        "verified_file_count": len(manifest),
        "source_fingerprint": source_fingerprint,
        "source_manifest": manifest,
        "data_authentication": expected_auth,
        "max_single_request_layer_gb": max(values[3] for values in table.values()),
        "fleet_layer_burst_bound_gb": NUM_NPU
        * max(values[3] for values in table.values()),
        "catalog": {f"{key[0]},{key[1]}": list(values) for key, values in table.items()},
    }


def _case_fingerprint(case: dict, num_ssu: int, source: str, config: str) -> str:
    return _canonical_hash(
        {
            "case": case,
            "num_ssu": num_ssu,
            "source_fingerprint": source,
            "config_fingerprint": config,
        },
        b"ms-scale-control-case:v1\0",
    )


def _validate_path_abi(value, prefix: str) -> None:
    _require(isinstance(value, dict), f"{prefix}: path ABI missing")
    _require(
        value.get("path_count") == 256
        and value.get("group_count") == 8
        and value.get("paths_per_group") == 32
        and value.get("max_npu") == 128
        and value.get("assigned_count") == NUM_NPU
        and value.get("assigned_unique") == NUM_NPU
        and value.get("path_zero_reserved") is True
        and _is_sha256(value.get("assigned_paths_sha256")),
        f"{prefix}: path ABI differs",
    )


def _validate_schedule_metadata(schedule, spec: dict, plan: dict, prefix: str) -> dict:
    _require(isinstance(schedule, dict), f"{prefix}: schedule metadata missing")
    seed = _integer(schedule.get("seed"), f"{prefix}: schedule seed")
    _require(seed in FORMAL_SEEDS, f"{prefix}: schedule seed outside plan")
    _require(
        schedule.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{prefix}: workload mode",
    )
    _require(schedule.get("num_npu") == NUM_NPU, f"{prefix}: schedule num_npu")
    backing = _integer(
        schedule.get("requests_per_npu"), f"{prefix}: schedule backing", minimum=1
    )
    _require(backing == plan["backing_requests_per_npu"], f"{prefix}: schedule backing")
    _require(
        schedule.get("request_id_formula") == "sequence * num_npu + npu_id",
        f"{prefix}: request-id formula",
    )
    assignments = schedule.get("assignment_rows")
    expected_count = NUM_NPU * backing
    _require(
        isinstance(assignments, list) and len(assignments) == expected_count,
        f"{prefix}: assignment row count",
    )
    by_request = {}
    for index, row in enumerate(assignments):
        _require(isinstance(row, list) and len(row) == 5, f"{prefix}: assignment {index}")
        request_id = _integer(row[0], f"{prefix}: request id")
        npu_id = _integer(row[1], f"{prefix}: NPU id")
        sequence = _integer(row[2], f"{prefix}: sequence")
        _require(request_id == index, f"{prefix}: request IDs are not contiguous")
        _require(0 <= npu_id < NUM_NPU, f"{prefix}: invalid NPU id")
        _require(
            request_id == sequence * NUM_NPU + npu_id,
            f"{prefix}: request-id formula mismatch",
        )
        _require(row[3] in ("SS", "SL", "LS", "LL"), f"{prefix}: category")
        _require(
            isinstance(row[4], list)
            and len(row[4]) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in row[4]),
            f"{prefix}: profile key",
        )
        by_request[request_id] = {
            "npu_id": npu_id,
            "sequence": sequence,
            "category": row[3],
            "profile_key": list(row[4]),
        }
    workload = spec.get("workload")
    _require(isinstance(workload, dict), f"{prefix}: workload spec missing")
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _require(
            schedule.get(field) == workload.get(field),
            f"{prefix}: schedule/spec {field} differs",
        )
    return {"seed": seed, "by_request": by_request, "assignments": assignments}


def _validate_spec(spec, plan: dict, prefix: str) -> tuple[dict[str, dict], int]:
    _require(isinstance(spec, dict), f"{prefix}: experiment_spec missing")
    _require(spec.get("schema_version") == SHARD_SCHEMA_VERSION, f"{prefix}: spec schema")
    _require(
        spec.get("experiment") == EXPERIMENT
        and spec.get("definition") == DEFINITION
        and spec.get("num_npu") == NUM_NPU
        and spec.get("n_layers") == N_LAYERS
        and spec.get("batch_size") == BATCH_SIZE,
        f"{prefix}: experiment identity/topology",
    )
    _require(
        spec.get("definition_fingerprint") == plan["definition_fingerprint"],
        f"{prefix}: definition fingerprint differs from campaign",
    )
    _require(
        spec.get("campaign_spec_sha256") == plan["file_sha256"],
        f"{prefix}: spec campaign SHA differs",
    )
    scale = spec.get("scale_semantics")
    _require(
        isinstance(scale, dict)
        and scale.get("num_npu") == NUM_NPU
        and scale.get("backing_requests_per_npu") == plan["backing_requests_per_npu"]
        and scale.get("total_assignment_count")
        == NUM_NPU * plan["backing_requests_per_npu"],
        f"{prefix}: scale semantics differ",
    )
    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), f"{prefix}: steady-state spec missing")
    expected_steady = {
        "requests_per_npu": plan["backing_requests_per_npu"],
        "warmup_requests_per_npu": plan["warmup_requests_per_npu"],
        "settle_ms": plan["settle_ms"],
        "measurement_ms": plan["measurement_ms"],
        "block_ms": plan["block_ms"],
        "slo_alpha": PRIMARY_ALPHA,
    }
    for field, expected in expected_steady.items():
        _require(steady.get(field) == expected, f"{prefix}: steady {field} differs")
    seed = _integer(steady.get("seed"), f"{prefix}: seed")
    _require(seed in FORMAL_SEEDS, f"{prefix}: seed outside plan")
    workload = spec.get("workload")
    _require(
        isinstance(workload, dict)
        and workload.get("seed") == seed
        and workload.get("requests_per_npu") == plan["backing_requests_per_npu"]
        and workload.get("scientific_prefix_requests_per_npu")
        == FORECAST_REQUESTS_PER_NPU,
        f"{prefix}: workload identity/shape",
    )
    _require(
        workload.get("authentication") is not None,
        f"{prefix}: workload authentication missing",
    )
    cases = spec.get("cases")
    _require(isinstance(cases, list), f"{prefix}: cases missing")
    case_by_name = {}
    for case in cases:
        _require(isinstance(case, dict) and isinstance(case.get("name"), str), f"{prefix}: case")
        _require(case["name"] not in case_by_name, f"{prefix}: duplicate case")
        case_by_name[case["name"]] = case
    for case_name, expected in EXPECTED_CASES.items():
        _require(case_by_name.get(case_name) == expected, f"{prefix}: case {case_name} differs")
    report_roles = spec.get("report_roles")
    _require(isinstance(report_roles, dict), f"{prefix}: report roles missing")
    for role, case_name in ROLE_TO_CASE.items():
        _require(report_roles.get(role) == case_name, f"{prefix}: role {role} differs")
    adaptive = spec.get("adaptive")
    _require(
        isinstance(adaptive, dict)
        and adaptive.get("controller") == "AdaptiveAdmissionSchemeBControllerV2_1"
        and adaptive.get("target_ratio") == 0.52
        and adaptive.get("required_ratio") == 0.5
        and adaptive.get("background_reserve_fraction") == 0.05
        and adaptive.get("ssd_cap_gbps") == SSD_CAP_GBPS
        and adaptive.get("npu_cap_gbps") == NPU_CAP_GBPS,
        f"{prefix}: adaptive controller/caps differ",
    )
    pfo = spec.get("pfo")
    _require(
        isinstance(pfo, dict)
        and pfo.get("controller") == "ProtectedFloorSchemeBController"
        and pfo.get("materialized_allocation_stages") == ["selected_protected_floor"]
        and pfo.get("request_pin") == "stable (npu_id, request_id)"
        and pfo.get("target_ratio") == 0.52
        and pfo.get("required_ratio") == 0.5
        and pfo.get("background_reserve_fraction_for_admission_only") == 0.05
        and pfo.get("internal_deadband_gbps_by_case", {}).get("pfo_astar_h70") == 0.05
        and pfo.get("downstream_cir_write_threshold_gbps") == 0.0
        and pfo.get("path_pressure_reads") == 0
        and pfo.get("trigger_ownership") == "CIRControlConfig.min_interval_ms"
        and pfo.get("real_register_order") == "all decreases before any increase",
        f"{prefix}: PFO controller contract differs",
    )
    _require(
        spec.get("measurement_cost_scope")
        == "true SSU pressure-table reads and CIR writes",
        f"{prefix}: measurement cost scope differs",
    )
    return case_by_name, seed


def _read_shard(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    payload, raw = _read_json(resolved)
    payload["_analysis_path"] = str(resolved)
    payload["_analysis_sha256"] = hashlib.sha256(raw).hexdigest()
    payload["_analysis_size_bytes"] = len(raw)
    return payload


def _validate_shard_root(payload: dict, plan: dict) -> dict:
    label = payload["_analysis_path"]
    _require(payload.get("schema_version") == SHARD_SCHEMA_VERSION, f"{label}: schema3 required")
    _require(
        payload.get("definition") == DEFINITION
        and payload.get("num_npu") == NUM_NPU
        and payload.get("backing_requests_per_npu")
        == plan["backing_requests_per_npu"],
        f"{label}: root identity/scale differs",
    )
    _require(
        payload.get("source_stable_during_run") is True
        and payload.get("source_fingerprint") == payload.get("ending_source_fingerprint")
        and payload.get("source_fingerprint") == plan["source_fingerprint"],
        f"{label}: source changed or differs from campaign",
    )
    source_manifest = payload.get("source_manifest")
    _require(isinstance(source_manifest, dict) and source_manifest, f"{label}: source manifest")
    _require(
        _canonical_hash(source_manifest, b"ms-scale-control-source:v1\0")
        == payload["source_fingerprint"],
        f"{label}: source fingerprint does not authenticate manifest",
    )
    _require(
        payload.get("definition_fingerprint") == plan["definition_fingerprint"],
        f"{label}: definition fingerprint differs",
    )
    _require(
        payload.get("config_stable_during_run") is True
        and payload.get("config_fingerprint") == payload.get("ending_config_fingerprint"),
        f"{label}: config changed during run",
    )
    _require(
        payload.get("campaign_spec_sha256") == plan["file_sha256"]
        and payload.get("campaign_spec_stable_during_run") is True,
        f"{label}: campaign SHA/stability differs",
    )
    expected_campaign_auth = {
        "sha256": plan["file_sha256"],
        "size_bytes": plan["file_size_bytes"],
    }
    _require(
        payload.get("campaign_spec_authentication") == expected_campaign_auth
        and payload.get("ending_campaign_spec_authentication") == expected_campaign_auth,
        f"{label}: raw campaign authentication differs",
    )
    spec = payload.get("experiment_spec")
    case_by_name, seed = _validate_spec(spec, plan, label)
    expected_config = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
    _require(payload.get("config_fingerprint") == expected_config, f"{label}: config fingerprint")
    authentication = payload.get("input_authentication")
    _require(
        isinstance(authentication, dict)
        and authentication == spec.get("workload", {}).get("authentication"),
        f"{label}: input authentication differs from spec",
    )
    _validate_path_abi(payload.get("path_abi"), label)
    schedule = _validate_schedule_metadata(payload.get("schedule_metadata"), spec, plan, label)
    _require(schedule["seed"] == seed, f"{label}: schedule/spec seed differs")
    execution = payload.get("execution")
    _require(
        isinstance(execution, dict)
        and execution.get("multiprocessing_start_method") in ("spawn", "forkserver"),
        f"{label}: unsafe/missing execution provenance",
    )
    _require(payload.get("selected_complete") is True, f"{label}: selected campaign incomplete")
    results = payload.get("results")
    _require(isinstance(results, list) and results, f"{label}: results missing")
    return {
        "path": label,
        "sha256": payload["_analysis_sha256"],
        "size_bytes": payload["_analysis_size_bytes"],
        "payload": payload,
        "spec": spec,
        "case_by_name": case_by_name,
        "seed": seed,
        "schedule": schedule,
        "source_fingerprint": payload["source_fingerprint"],
        "source_manifest": source_manifest,
        "config_fingerprint": expected_config,
        "authentication": authentication,
    }


def _slo_metrics(request_rows: Sequence[dict], alpha: float) -> dict:
    by_npu: dict[int, list[bool]] = defaultdict(list)
    outcomes = []
    for row in request_rows:
        outcome = (
            float(row["ttft_ms"])
            <= alpha * float(row["ideal_ttft_ms"]) + SLO_EPSILON
        )
        by_npu[int(row["npu_id"])].append(outcome)
        outcomes.append(outcome)
    _require(set(by_npu) == set(range(NUM_NPU)), "SLO cohort misses an NPU")
    return {
        "equal_npu": statistics.fmean(
            statistics.fmean(by_npu[npu_id]) for npu_id in range(NUM_NPU)
        ),
        "request_weighted": statistics.fmean(outcomes),
        "counts": [len(by_npu[npu_id]) for npu_id in range(NUM_NPU)],
    }


def _validate_requests(summary: dict, schedule: dict, prefix: str) -> dict:
    rows = summary.get("request_rows")
    _require(isinstance(rows, list) and rows, f"{prefix}: request_rows missing")
    _require(
        summary.get("measurement_request_count") == len(rows),
        f"{prefix}: measurement request count",
    )
    start_ms = _finite(summary.get("measurement_start_ms"), f"{prefix}: start")
    end_ms = _finite(summary.get("measurement_end_ms"), f"{prefix}: end")
    by_request = schedule["by_request"]
    seen = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{prefix}: request {index}")
        request_id = _integer(row.get("request_id"), f"{prefix}: request id")
        _require(request_id not in seen and request_id in by_request, f"{prefix}: request identity")
        seen.add(request_id)
        expected = by_request[request_id]
        npu_id = _integer(row.get("npu_id"), f"{prefix}: request NPU")
        sequence = _integer(row.get("sequence"), f"{prefix}: request sequence")
        _require(
            npu_id == expected["npu_id"]
            and sequence == expected["sequence"]
            and row.get("category") == expected["category"]
            and row.get("profile_key") == expected["profile_key"],
            f"{prefix}: request differs from authenticated assignment",
        )
        admission = _finite(row.get("admission_time_ms"), f"{prefix}: admission")
        completion = _finite(row.get("completion_time_ms"), f"{prefix}: completion")
        ttft = _nonnegative(row.get("ttft_ms"), f"{prefix}: TTFT")
        ideal = _finite(row.get("ideal_ttft_ms"), f"{prefix}: ideal TTFT")
        _require(ideal > 0.0, f"{prefix}: nonpositive ideal TTFT")
        _require(
            start_ms <= admission < end_ms and completion >= admission,
            f"{prefix}: tagged request outside half-open window",
        )
        _close(ttft, completion - admission, f"{prefix}: request TTFT")
        expected_slo = ttft <= PRIMARY_ALPHA * ideal + SLO_EPSILON
        _require(row.get("slo_met") is expected_slo, f"{prefix}: request SLO flag")
    alpha2 = _slo_metrics(rows, PRIMARY_ALPHA)
    alpha15 = _slo_metrics(rows, SENSITIVITY_ALPHA)
    counts = alpha2["counts"]
    _require(min(counts) >= MIN_REQUESTS_PER_NPU, f"{prefix}: fewer than 8 samples/NPU")
    _require(summary.get("request_counts_by_npu") == counts, f"{prefix}: per-NPU counts")
    _close(summary.get("ttft_slo_attainment"), alpha2["equal_npu"], f"{prefix}: alpha2 SLO")
    _close(
        summary.get("request_weighted_slo_attainment"),
        alpha2["request_weighted"],
        f"{prefix}: request-weighted SLO",
    )
    ttfts = [float(row["ttft_ms"]) for row in rows]
    _close(summary.get("mean_ttft_ms"), statistics.fmean(ttfts), f"{prefix}: mean TTFT")
    _close(summary.get("p99_ttft_ms"), _percentile(ttfts, 99.0), f"{prefix}: p99 TTFT")
    cohort_rows = [
        [
            int(row["request_id"]),
            int(row["npu_id"]),
            int(row["sequence"]),
            list(row["profile_key"]),
        ]
        for row in rows
    ]
    cohort_hash = _canonical_hash(
        cohort_rows, b"ms-scale-control-measurement-cohort:v1\0"
    )
    completed = _vector(
        summary.get("completed_by_npu_at_stop"),
        NUM_NPU,
        f"{prefix}: completed counts",
        integer=True,
    )
    for npu_id in range(NUM_NPU):
        minimum_completed = max(
            int(row["sequence"]) for row in rows if int(row["npu_id"]) == npu_id
        ) + 1
        _require(completed[npu_id] >= minimum_completed, f"{prefix}: completion count contradiction")
    _require(summary.get("all_input_requests_completed") is False, f"{prefix}: backing exhausted")
    backing = schedule["assignments"][-1][2] + 1
    _require(backing > 0, f"{prefix}: backing derivation")
    backing_margin = backing - max(completed)
    _require(backing_margin >= MIN_BACKING_MARGIN, f"{prefix}: backing margin < 32")
    return {
        "rows": rows,
        "counts": counts,
        "alpha2": alpha2,
        "alpha15": alpha15,
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 99.0),
        "cohort_hash": cohort_hash,
        "backing_margin": backing_margin,
        "max_equal_npu_single_outcome_weight_pp": max(
            100.0 / (NUM_NPU * count) for count in counts
        ),
    }


def _validate_count_rate(
    summary: dict, count_field: str, rate_field: str, duration_s: float, prefix: str
) -> int:
    count = _integer(summary.get(count_field), f"{prefix}: {count_field}")
    _close(summary.get(rate_field), count / duration_s, f"{prefix}: {rate_field}")
    return count


def _validate_control_metrics(
    summary: dict, case: dict, num_ssu: int, duration_ms: float, prefix: str
) -> dict:
    duration_s = duration_ms / 1000.0
    _require(
        summary.get("measurement_control_counter_window") == EXPECTED_CONTROL_WINDOW,
        f"{prefix}: control counter window",
    )
    evaluations = _validate_count_rate(
        summary,
        "measurement_control_evaluations",
        "measurement_control_evaluation_rate_hz",
        duration_s,
        prefix,
    )
    writes = _validate_count_rate(
        summary,
        "measurement_cir_path_writes",
        "measurement_cir_path_write_rate_hz",
        duration_s,
        prefix,
    )
    transactions = _validate_count_rate(
        summary,
        "measurement_cir_write_transactions",
        "measurement_cir_write_transaction_rate_hz",
        duration_s,
        prefix,
    )
    commits = _integer(summary.get("measurement_cir_commits"), f"{prefix}: commits")
    write_by_ssu = _vector(
        summary.get("measurement_cir_path_writes_by_ssu"),
        num_ssu,
        f"{prefix}: writes by SSU",
        integer=True,
    )
    tx_by_ssu = _vector(
        summary.get("measurement_cir_write_transactions_by_ssu"),
        num_ssu,
        f"{prefix}: transactions by SSU",
        integer=True,
    )
    _require(sum(write_by_ssu) == writes, f"{prefix}: write conservation")
    _require(sum(tx_by_ssu) == transactions, f"{prefix}: transaction conservation")
    rate_by_ssu = _vector(
        summary.get("measurement_cir_path_write_rate_hz_by_ssu"),
        num_ssu,
        f"{prefix}: write rates by SSU",
    )
    tx_rate_by_ssu = _vector(
        summary.get("measurement_cir_write_transaction_rate_hz_by_ssu"),
        num_ssu,
        f"{prefix}: transaction rates by SSU",
    )
    _compare_vectors(rate_by_ssu, [value / duration_s for value in write_by_ssu], f"{prefix}: write rates")
    _compare_vectors(tx_rate_by_ssu, [value / duration_s for value in tx_by_ssu], f"{prefix}: tx rates")
    _require(0 <= commits <= evaluations, f"{prefix}: commit/evaluation relation")
    _require(transactions <= writes or writes == 0, f"{prefix}: transactions exceed writes")
    _close(summary.get("pressure_ttl_ms"), case["pressure_ttl_ms"], f"{prefix}: pressure TTL")
    _close(
        summary.get("cir_write_threshold_gbps"),
        case["cir_write_threshold_gbps"],
        f"{prefix}: downstream threshold",
    )
    _close(summary.get("control_min_interval_ms"), case["min_interval_ms"], f"{prefix}: interval")
    if case["kind"] == "baseline":
        _require(
            evaluations == writes == transactions == commits == 0,
            f"{prefix}: baseline performed dynamic control",
        )
    if case["kind"] == "adaptive":
        _require(evaluations > 0 and writes > 0, f"{prefix}: A0 control is inactive")
    return {
        "control_evaluation_rate_hz": evaluations / duration_s,
        "cir_entry_write_rate_hz": writes / duration_s,
        "cir_transaction_rate_hz": transactions / duration_s,
        "measurement_control_evaluations": evaluations,
        "measurement_cir_path_writes": writes,
        "measurement_cir_write_transactions": transactions,
        "measurement_cir_commits": commits,
    }


def _matrix(value, rows: int, columns: int, name: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == rows, f"{name} rows")
    return [_vector(row, columns, f"{name}[{index}]") for index, row in enumerate(value)]


def _validate_resources(summary: dict, num_ssu: int, duration_ms: float, prefix: str) -> dict:
    compute = _vector(summary.get("compute_ms_by_npu"), NUM_NPU, f"{prefix}: compute")
    npu_utils = _vector(
        summary.get("npu_utilizations"), NUM_NPU, f"{prefix}: NPU utils", fraction=True
    )
    _compare_vectors(npu_utils, [value / duration_ms for value in compute], f"{prefix}: NPU utilization")
    mean_util = statistics.fmean(npu_utils)
    _close(summary.get("mean_npu_utilization"), mean_util, f"{prefix}: mean NPU util")
    ssd_busy = _vector(
        summary.get("measurement_ssd_busy_ms_by_ssu"), num_ssu, f"{prefix}: SSD busy"
    )
    ssd_served = _vector(
        summary.get("measurement_ssd_served_gb_by_ssu"), num_ssu, f"{prefix}: SSD served"
    )
    ssd_utils = _vector(
        summary.get("measurement_ssd_utilizations"),
        num_ssu,
        f"{prefix}: SSD utils",
        fraction=True,
    )
    _compare_vectors(ssd_utils, [value / duration_ms for value in ssd_busy], f"{prefix}: SSD utilization")
    _compare_vectors(
        ssd_served,
        [value * SSD_CAP_GBPS / 1000.0 for value in ssd_busy],
        f"{prefix}: SSD service",
    )
    _close(
        summary.get("measurement_ssd_mean_utilization"),
        statistics.fmean(ssd_utils),
        f"{prefix}: mean SSD util",
    )
    link_busy = _vector(
        summary.get("measurement_npu_link_busy_ms_by_npu"),
        NUM_NPU,
        f"{prefix}: link busy",
    )
    link_utils = _vector(
        summary.get("measurement_npu_link_utilizations"),
        NUM_NPU,
        f"{prefix}: link utils",
        fraction=True,
    )
    _compare_vectors(link_utils, [value / duration_ms for value in link_busy], f"{prefix}: link utilization")
    _close(
        summary.get("measurement_npu_link_mean_utilization"),
        statistics.fmean(link_utils),
        f"{prefix}: mean link util",
    )
    ssd_matrix = _matrix(
        summary.get("measurement_npu_ssu_ssd_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: NPU/SSU SSD service",
    )
    link_matrix = _matrix(
        summary.get("measurement_npu_ssu_link_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: NPU/SSU link service",
    )
    _compare_vectors(
        [math.fsum(ssd_matrix[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)],
        ssd_served,
        f"{prefix}: SSD matrix columns",
    )
    _compare_vectors(
        [math.fsum(row) for row in link_matrix],
        [value * NPU_CAP_GBPS / 1000.0 for value in link_busy],
        f"{prefix}: link matrix rows",
    )
    for field, length, cap, tolerance in (
        ("actual_cir_sum_gbps_by_ssu_at_stop", num_ssu, SSD_CAP_GBPS, 1e-12),
        ("max_actual_cir_sum_gbps_by_ssu", num_ssu, SSD_CAP_GBPS, 1e-12),
        ("measurement_actual_cir_sum_gbps_by_ssu_at_start", num_ssu, SSD_CAP_GBPS, 1e-12),
        ("measurement_actual_cir_sum_gbps_by_ssu_at_end", num_ssu, SSD_CAP_GBPS, 1e-12),
        ("actual_npu_cir_sum_gbps_at_stop", NUM_NPU, NPU_CAP_GBPS, 1e-9),
        ("max_actual_npu_cir_sum_gbps_by_npu", NUM_NPU, NPU_CAP_GBPS, 1e-9),
        ("measurement_actual_npu_cir_sum_gbps_at_start", NUM_NPU, NPU_CAP_GBPS, 1e-9),
        ("measurement_actual_npu_cir_sum_gbps_at_end", NUM_NPU, NPU_CAP_GBPS, 1e-9),
    ):
        values = _vector(summary.get(field), length, f"{prefix}: {field}")
        _require(max(values, default=0.0) <= cap + tolerance, f"{prefix}: {field} exceeds capacity")
    return {
        "mean_npu_utilization_pct": 100.0 * mean_util,
        "measurement_ssd_mean_utilization_pct": 100.0 * statistics.fmean(ssd_utils),
        "measurement_npu_link_mean_utilization_pct": 100.0 * statistics.fmean(link_utils),
    }


def _parse_stationarity_snapshots(
    summary: dict, num_ssu: int, plan: dict, prefix: str
) -> list[dict]:
    _require(
        summary.get("stationarity_boundary_semantics")
        == EXPECTED_STATIONARITY_SEMANTICS,
        f"{prefix}: stationarity boundary semantics",
    )
    snapshots = summary.get("measurement_stationarity_boundaries")
    _require(
        isinstance(snapshots, list) and len(snapshots) == plan["boundary_count"],
        f"{prefix}: stationarity boundary count",
    )
    _require(
        summary.get("measurement_stationarity_boundary_count")
        == plan["boundary_count"],
        f"{prefix}: reported stationarity boundary count",
    )
    start_ms = float(summary["measurement_start_ms"])
    parsed = []
    for boundary, snapshot in enumerate(snapshots):
        _require(
            isinstance(snapshot, dict) and set(snapshot) == SNAPSHOT_FIELDS,
            f"{prefix}: snapshot {boundary} fields",
        )
        _require(snapshot.get("boundary") == boundary, f"{prefix}: boundary index")
        expected_time = start_ms + boundary * plan["block_ms"]
        _close(
            snapshot.get("time_ms"),
            expected_time,
            f"{prefix}: boundary time",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        item = {
            "time_ms": float(snapshot["time_ms"]),
            "ssd_busy": _vector(
                snapshot["ssd_cumulative_busy_ms_by_ssu"],
                num_ssu,
                f"{prefix}: boundary SSD busy",
            ),
            "ssd_served": _vector(
                snapshot["ssd_cumulative_served_gb_by_ssu"],
                num_ssu,
                f"{prefix}: boundary SSD served",
            ),
            "ssd_blocks": _vector(
                snapshot["ssd_outstanding_blocks_by_ssu"],
                num_ssu,
                f"{prefix}: boundary SSD blocks",
                integer=True,
            ),
            "ssd_gb": _vector(
                snapshot["ssd_outstanding_gb_by_ssu"],
                num_ssu,
                f"{prefix}: boundary SSD GB",
            ),
            "compute": _vector(
                snapshot["npu_compute_cumulative_busy_ms_by_npu"],
                NUM_NPU,
                f"{prefix}: boundary compute",
            ),
            "link_busy": _vector(
                snapshot["npu_link_cumulative_busy_ms_by_npu"],
                NUM_NPU,
                f"{prefix}: boundary link busy",
            ),
            "link_served": _vector(
                snapshot["npu_link_cumulative_served_gb_by_npu"],
                NUM_NPU,
                f"{prefix}: boundary link served",
            ),
            "link_blocks": _vector(
                snapshot["npu_link_outstanding_blocks_by_npu"],
                NUM_NPU,
                f"{prefix}: boundary link blocks",
                integer=True,
            ),
            "link_gb": _vector(
                snapshot["npu_link_outstanding_gb_by_npu"],
                NUM_NPU,
                f"{prefix}: boundary link GB",
            ),
        }
        _compare_vectors(
            item["ssd_served"],
            [value * SSD_CAP_GBPS / 1000.0 for value in item["ssd_busy"]],
            f"{prefix}: boundary SSD service",
        )
        _compare_vectors(
            item["link_served"],
            [value * NPU_CAP_GBPS / 1000.0 for value in item["link_busy"]],
            f"{prefix}: boundary link service",
        )
        parsed.append(item)
    for previous, current in zip(parsed, parsed[1:]):
        for field in ("ssd_busy", "ssd_served", "compute", "link_busy", "link_served"):
            _require(
                all(right + 1e-8 >= left for left, right in zip(previous[field], current[field])),
                f"{prefix}: cumulative {field} moved backwards",
            )
    return parsed


def _validate_stationarity(
    summary: dict,
    diagnostic: dict,
    request_info: dict,
    num_ssu: int,
    plan: dict,
    burst_bound_gb: float,
    prefix: str,
) -> dict:
    snapshots = _parse_stationarity_snapshots(summary, num_ssu, plan, prefix)
    blocks = summary.get("measurement_blocks")
    _require(
        isinstance(blocks, list) and len(blocks) == plan["block_count"],
        f"{prefix}: stationarity block count",
    )
    block_utils = []
    block_request_counts = []
    block_fleet_served_gb = []
    resource_sums = {
        "ssd_busy": [0.0] * num_ssu,
        "ssd_served": [0.0] * num_ssu,
        "compute": [0.0] * NUM_NPU,
        "link_busy": [0.0] * NUM_NPU,
    }
    start_ms = float(summary["measurement_start_ms"])
    for block_index, (block, left, right) in enumerate(
        zip(blocks, snapshots, snapshots[1:])
    ):
        _require(
            isinstance(block, dict) and set(block) == BLOCK_FIELDS,
            f"{prefix}: block {block_index} fields",
        )
        expected_start = start_ms + block_index * plan["block_ms"]
        expected_end = expected_start + plan["block_ms"]
        _require(block.get("block") == block_index, f"{prefix}: block index")
        _close(block.get("start_ms"), expected_start, f"{prefix}: block start", abs_tol=1e-9, rel_tol=0.0)
        _close(block.get("end_ms"), expected_end, f"{prefix}: block end", abs_tol=1e-9, rel_tol=0.0)
        _close(block.get("duration_ms"), plan["block_ms"], f"{prefix}: block duration")
        deltas = {
            "ssd_busy": [b - a for a, b in zip(left["ssd_busy"], right["ssd_busy"])],
            "ssd_served": [b - a for a, b in zip(left["ssd_served"], right["ssd_served"])],
            "compute": [b - a for a, b in zip(left["compute"], right["compute"])],
            "link_busy": [b - a for a, b in zip(left["link_busy"], right["link_busy"])],
            "link_served": [b - a for a, b in zip(left["link_served"], right["link_served"])],
        }
        for field, values in deltas.items():
            _require(all(value >= -1e-8 for value in values), f"{prefix}: negative block {field}")
            if field in resource_sums:
                for index, value in enumerate(values):
                    resource_sums[field][index] += max(0.0, value)
        for block_field, delta_field in (
            ("ssd_busy_ms_by_ssu", "ssd_busy"),
            ("ssd_served_gb_by_ssu", "ssd_served"),
            ("compute_ms_by_npu", "compute"),
            ("npu_link_busy_ms_by_npu", "link_busy"),
            ("npu_link_served_gb_by_npu", "link_served"),
        ):
            values = _vector(block[block_field], len(deltas[delta_field]), f"{prefix}: {block_field}")
            _compare_vectors(values, deltas[delta_field], f"{prefix}: block {block_field}")
        ssd_utils = _vector(block["ssd_utilizations"], num_ssu, f"{prefix}: block SSD utils", fraction=True)
        _compare_vectors(ssd_utils, [value / plan["block_ms"] for value in deltas["ssd_busy"]], f"{prefix}: block SSD utils")
        _close(block["ssd_mean_utilization"], statistics.fmean(ssd_utils), f"{prefix}: block SSD mean")
        npu_utils = _vector(block["npu_utilizations"], NUM_NPU, f"{prefix}: block NPU utils", fraction=True)
        _compare_vectors(npu_utils, [value / plan["block_ms"] for value in deltas["compute"]], f"{prefix}: block NPU utils")
        block_util = statistics.fmean(npu_utils)
        _close(block["npu_utilization"], block_util, f"{prefix}: block NPU mean")
        link_utils = _vector(block["npu_link_utilizations"], NUM_NPU, f"{prefix}: block link utils", fraction=True)
        _compare_vectors(link_utils, [value / plan["block_ms"] for value in deltas["link_busy"]], f"{prefix}: block link utils")
        _close(block["npu_link_mean_utilization"], statistics.fmean(link_utils), f"{prefix}: block link mean")
        for field, expected, integer_values in (
            ("ssd_outstanding_blocks_at_start", left["ssd_blocks"], True),
            ("ssd_outstanding_blocks_at_end", right["ssd_blocks"], True),
            ("ssd_outstanding_gb_at_start", left["ssd_gb"], False),
            ("ssd_outstanding_gb_at_end", right["ssd_gb"], False),
            ("npu_link_outstanding_blocks_at_start", left["link_blocks"], True),
            ("npu_link_outstanding_blocks_at_end", right["link_blocks"], True),
            ("npu_link_outstanding_gb_at_start", left["link_gb"], False),
            ("npu_link_outstanding_gb_at_end", right["link_gb"], False),
        ):
            values = _vector(block[field], len(expected), f"{prefix}: {field}", integer=integer_values)
            if integer_values:
                _require(values == expected, f"{prefix}: {field} edge")
            else:
                _compare_vectors(values, expected, f"{prefix}: {field} edge")
        for base, left_values, right_values, integer_values in (
            ("ssd_outstanding_blocks", left["ssd_blocks"], right["ssd_blocks"], True),
            ("ssd_outstanding_gb", left["ssd_gb"], right["ssd_gb"], False),
            ("npu_link_outstanding_blocks", left["link_blocks"], right["link_blocks"], True),
            ("npu_link_outstanding_gb", left["link_gb"], right["link_gb"], False),
        ):
            expected_delta = [right_value - left_value for left_value, right_value in zip(left_values, right_values)]
            raw_delta = block[f"{base}_delta"]
            if integer_values:
                _require(raw_delta == expected_delta, f"{prefix}: {base} delta")
            else:
                _compare_vectors(raw_delta, expected_delta, f"{prefix}: {base} delta")
        admitted = [
            row
            for row in request_info["rows"]
            if expected_start <= float(row["admission_time_ms"]) < expected_end
        ]
        _require(block["request_count"] == len(admitted), f"{prefix}: block request count")
        if admitted:
            _close(
                block["request_weighted_slo_attainment"],
                statistics.fmean(bool(row["slo_met"]) for row in admitted),
                f"{prefix}: block SLO",
            )
        else:
            _require(block["request_weighted_slo_attainment"] is None, f"{prefix}: empty block SLO")
        block_utils.append(block_util)
        block_request_counts.append(len(admitted))
        block_fleet_served_gb.append(math.fsum(deltas["ssd_served"]))
    _require(sum(block_request_counts) == len(request_info["rows"]), f"{prefix}: blocks do not cover cohort")
    for summary_field, resource_field in (
        ("measurement_ssd_busy_ms_by_ssu", "ssd_busy"),
        ("measurement_ssd_served_gb_by_ssu", "ssd_served"),
        ("compute_ms_by_npu", "compute"),
        ("measurement_npu_link_busy_ms_by_npu", "link_busy"),
    ):
        values = _vector(summary[summary_field], len(resource_sums[resource_field]), f"{prefix}: {summary_field}")
        _compare_vectors(values, resource_sums[resource_field], f"{prefix}: {summary_field} block sum")
    expected_diagnostic_fields = {
        "block_npu_utilizations",
        "block_request_counts",
        "block_utilization_range",
        "first_last_utilization_delta",
        "outstanding_blocks_drift_by_ssu",
        "fleet_outstanding_blocks_drift",
    }
    _require(isinstance(diagnostic, dict) and set(diagnostic) == expected_diagnostic_fields, f"{prefix}: stationarity diagnostics")
    _compare_vectors(diagnostic["block_npu_utilizations"], block_utils, f"{prefix}: diagnostic block utils")
    _require(diagnostic["block_request_counts"] == block_request_counts, f"{prefix}: diagnostic request counts")
    _close(diagnostic["block_utilization_range"], max(block_utils) - min(block_utils), f"{prefix}: util range")
    _close(diagnostic["first_last_utilization_delta"], block_utils[-1] - block_utils[0], f"{prefix}: first/last util")
    first, last = snapshots[0], snapshots[-1]
    block_drift = [end - start for start, end in zip(first["ssd_blocks"], last["ssd_blocks"])]
    _require(diagnostic["outstanding_blocks_drift_by_ssu"] == block_drift, f"{prefix}: queue drift")
    _require(diagnostic["fleet_outstanding_blocks_drift"] == sum(block_drift), f"{prefix}: fleet queue drift")
    half = plan["block_count"] // 2
    first_util = statistics.fmean(block_utils[:half])
    second_util = statistics.fmean(block_utils[half:])
    util_half_delta_pp = 100.0 * (second_util - first_util)
    midpoints = [start_ms + (index + 0.5) * plan["block_ms"] for index in range(plan["block_count"])]
    util_slope = _theil_sen_slope(midpoints, block_utils)
    util_projected_change_pp = 100.0 * util_slope * (plan["measurement_ms"] / 1000.0)
    half_seconds = half * plan["block_ms"] / 1000.0
    first_served = math.fsum(block_fleet_served_gb[:half]) / half_seconds
    second_served = math.fsum(block_fleet_served_gb[half:]) / half_seconds
    served_midpoint = (first_served + second_served) / 2.0
    served_relative_delta = abs(second_served - first_served) / served_midpoint if served_midpoint > 0.0 else 0.0
    times = [snapshot["time_ms"] for snapshot in snapshots]

    def queue_diagnostic(values: list[float]) -> dict:
        deltas = [right - left for left, right in zip(values, values[1:])]
        slope = _theil_sen_slope(times, values)
        nondecreasing = sum(delta >= -1e-8 for delta in deltas) / len(deltas)
        growth = values[-1] - values[0]
        persistent = slope > 0.0 and nondecreasing >= 0.75 and growth > burst_bound_gb + 1e-8
        return {
            "start_gb": values[0],
            "end_gb": values[-1],
            "net_growth_gb": growth,
            "theil_sen_slope_gbps": slope,
            "nondecreasing_step_fraction": nondecreasing,
            "persistent_growth_over_one_layer_burst": persistent,
        }

    fleet_queue = queue_diagnostic([math.fsum(snapshot["ssd_gb"]) for snapshot in snapshots])
    per_ssu_queue = [queue_diagnostic([snapshot["ssd_gb"][ssu] for snapshot in snapshots]) for ssu in range(num_ssu)]
    rules = {
        "utilization_half_delta_le_1pp": abs(util_half_delta_pp) <= UTIL_HALF_LIMIT_PP + 1e-12,
        "utilization_projected_trend_le_2pp": abs(util_projected_change_pp) <= UTIL_TREND_LIMIT_PP + 1e-12,
        "fleet_served_half_relative_delta_le_2pct": served_relative_delta <= SERVED_HALF_RELATIVE_LIMIT + 1e-12,
        "no_persistent_queue_growth_over_one_layer_burst": not (
            fleet_queue["persistent_growth_over_one_layer_burst"]
            or any(value["persistent_growth_over_one_layer_burst"] for value in per_ssu_queue)
        ),
    }
    return {
        "stationarity_rule_version": STATIONARITY_RULE_VERSION,
        "block_count": plan["block_count"],
        "boundary_count": plan["boundary_count"],
        "utilization_first_half": first_util,
        "utilization_second_half": second_util,
        "utilization_half_delta_pp": util_half_delta_pp,
        "utilization_theil_sen_slope_fraction_per_s": util_slope,
        "utilization_projected_change_pp": util_projected_change_pp,
        "fleet_served_first_half_gbps": first_served,
        "fleet_served_second_half_gbps": second_served,
        "fleet_served_half_relative_delta": served_relative_delta,
        "fleet_queue": fleet_queue,
        "per_ssu_queue": per_ssu_queue,
        "stationarity_rules": rules,
        "stationarity_gate_passed": all(rules.values()),
    }


def _forecast_from_manifest(demand_values: Sequence[float], num_ssu: int) -> dict:
    demand = tuple(_nonnegative(value, "forecast demand") for value in demand_values)
    _require(len(demand) == num_ssu, "forecast demand topology differs")
    capacities = (SSD_CAP_GBPS,) * num_ssu
    fractions = tuple(value / cap for value, cap in zip(demand, capacities))
    fleet_fraction = math.fsum(demand) / math.fsum(capacities)
    fallback = fleet_fraction >= FORECAST_HOT_FRACTION
    mask = tuple(fallback or value >= FORECAST_HOT_FRACTION for value in fractions)
    classifications = tuple(
        "fleet_full_protection_fallback"
        if fallback
        else "hot_ssu"
        if selected
        else "cold_ssu_zero_cir"
        for selected in mask
    )
    canonical_input = {
        "capacity_gbps_by_ssu": capacities,
        "demand_gbps_by_ssu": demand,
        "forecast_requests_per_npu": FORECAST_REQUESTS_PER_NPU,
        "hot_fraction": FORECAST_HOT_FRACTION,
        "policy": "frozen_manifest_hotspot_v1",
    }
    fingerprint = _canonical_hash(canonical_input, b"frozen-manifest-hotspot:v1\0")
    return {
        "demand_gbps_by_ssu": list(demand),
        "capacity_gbps_by_ssu": list(capacities),
        "load_fraction_by_ssu": list(fractions),
        "fleet_load_fraction": fleet_fraction,
        "hot_fraction": FORECAST_HOT_FRACTION,
        "materialized_ssu_mask": list(mask),
        "classification_by_ssu": list(classifications),
        "full_protection_fallback": fallback,
        "forecast_requests_per_npu": FORECAST_REQUESTS_PER_NPU,
        "input_fingerprint": fingerprint,
        "policy": "frozen_manifest_hotspot_v1",
        "frozen_for_measurement": True,
        "materialized_ssu_count": sum(mask),
        "cold_ssu_count": len(mask) - sum(mask),
        "forecast_input_fingerprint": fingerprint,
    }


def _audit_record(record, index: int, num_ssu: int, mask: list[bool], prefix: str) -> dict:
    _require(
        isinstance(record, dict) and set(record) == AUDIT_RECORD_FIELDS,
        f"{prefix}: PFO audit record {index} fields",
    )
    time_ms = _finite(record["time_ms"], f"{prefix}: audit time")
    evaluation = _integer(record["evaluation"], f"{prefix}: audit evaluation", minimum=1)
    for field in (
        "pre_state_hash",
        "ideal_state_hash",
        "required_state_hash",
        "install_state_hash",
        "changed_entries_hash",
    ):
        _require(_is_sha256(record[field]), f"{prefix}: audit {field}")
    _require(record["materialized_ssu_mask"] == mask, f"{prefix}: audit mask changed")
    _require(record["cold_ssu_install_zero"] is True, f"{prefix}: cold CIR install nonzero")
    _require(
        record["ordered_sequence_capacity_safe"] is True
        and record["post_des_state_verified"] is True,
        f"{prefix}: PFO ordered/DES safety failed",
    )
    _require(
        _finite(record["maximum_ordered_prefix_ssu_excess_gbps"], f"{prefix}: SSU excess")
        <= 1e-12
        and _finite(record["maximum_ordered_prefix_npu_excess_gbps"], f"{prefix}: NPU excess")
        <= 1e-9
        and _finite(record["maximum_required_floor_shortfall_gbps"], f"{prefix}: floor shortfall")
        <= 1e-12,
        f"{prefix}: capacity/floor audit exceeded tolerance",
    )
    _require(
        _integer(record["active_set_hold_violations"], f"{prefix}: active holds") == 0,
        f"{prefix}: active-set deadband violation",
    )
    _nonnegative(record["maximum_deadband_held_delta_gbps"], f"{prefix}: held delta")
    changed = record["changed_entries"]
    _require(isinstance(changed, list), f"{prefix}: changed entries")
    reason_counts = Counter()
    writes_by_ssu = [0] * num_ssu
    decrease_by_ssu = [0] * num_ssu
    increase_by_ssu = [0] * num_ssu
    safety_by_ssu = [0] * num_ssu
    required_increases = 0
    capacity_decreases = 0
    for change_index, change in enumerate(changed):
        _require(
            isinstance(change, list) and len(change) == 5,
            f"{prefix}: changed entry {change_index}",
        )
        ssu_id = _integer(change[0], f"{prefix}: changed SSU")
        path_id = _integer(change[1], f"{prefix}: changed Path")
        direction = _integer(change[2], f"{prefix}: changed direction", minimum=-1)
        _require(0 <= ssu_id < num_ssu and 0 <= path_id < 256, f"{prefix}: changed target")
        _require(direction in (-1, 1), f"{prefix}: changed direction must be +/-1")
        reason = change[3]
        _require(isinstance(reason, str) and reason, f"{prefix}: change reason")
        safety_forced = _integer(change[4], f"{prefix}: safety flag")
        _require(safety_forced in (0, 1), f"{prefix}: safety flag is not boolean")
        reason_counts[reason] += 1
        writes_by_ssu[ssu_id] += 1
        decrease_by_ssu[ssu_id] += direction < 0
        increase_by_ssu[ssu_id] += direction > 0
        safety_by_ssu[ssu_id] += safety_forced
        required_increases += direction > 0 and reason == "required_hard_floor"
        capacity_decreases += direction < 0 and reason in (
            "ssd_capacity_compensation",
            "npu_capacity_compensation",
        )
    _require(
        record["changed_entries_hash"]
        == _canonical_hash(changed, b"pfo-ordered-changed-entries:v1\0"),
        f"{prefix}: changed-entry hash",
    )
    _require(record["change_reason_counts"] == dict(sorted(reason_counts.items())), f"{prefix}: reason counts")
    decrease_writes = sum(decrease_by_ssu)
    increase_writes = sum(increase_by_ssu)
    _require(
        record["decrease_writes"] == decrease_writes
        and record["increase_writes"] == increase_writes,
        f"{prefix}: phase write counts",
    )
    _require(record["writes_by_ssu"] == writes_by_ssu, f"{prefix}: writes by SSU")
    expected_decrease_tx = [int(value > 0) for value in decrease_by_ssu]
    expected_increase_tx = [int(value > 0) for value in increase_by_ssu]
    expected_tx = [left + right for left, right in zip(expected_decrease_tx, expected_increase_tx)]
    _require(
        record["decrease_transactions_by_ssu"] == expected_decrease_tx
        and record["increase_transactions_by_ssu"] == expected_increase_tx
        and record["transactions_by_ssu"] == expected_tx,
        f"{prefix}: phase transaction counts",
    )
    _require(
        record["safety_forced_writes_by_ssu"] == safety_by_ssu
        and record["safety_forced_writes"] == sum(safety_by_ssu),
        f"{prefix}: safety-forced write counts",
    )
    _require(
        record["required_floor_increases"] == required_increases
        and record["capacity_compensation_decreases"] == capacity_decreases,
        f"{prefix}: safety reason counts",
    )
    _integer(record["deadband_holds"], f"{prefix}: deadband holds")
    selected_count = _integer(record["selected_npu_count"], f"{prefix}: selected count")
    _require(selected_count <= NUM_NPU, f"{prefix}: selected count exceeds fleet")
    return {
        "time_ms": time_ms,
        "evaluation": evaluation,
        "writes_by_ssu": writes_by_ssu,
        "transactions_by_ssu": expected_tx,
        "decrease_transactions_by_ssu": expected_decrease_tx,
        "increase_transactions_by_ssu": expected_increase_tx,
        "safety_by_ssu": safety_by_ssu,
        "required_increases": required_increases,
        "capacity_decreases": capacity_decreases,
        "deadband_holds": int(record["deadband_holds"]),
        "active_violations": int(record["active_set_hold_violations"]),
        "floor_shortfall": float(record["maximum_required_floor_shortfall_gbps"]),
        "ssu_excess": float(record["maximum_ordered_prefix_ssu_excess_gbps"]),
        "npu_excess": float(record["maximum_ordered_prefix_npu_excess_gbps"]),
    }


def _sum_record_vectors(records: Sequence[dict], field: str, num_ssu: int) -> list[int]:
    return [sum(record[field][ssu] for record in records) for ssu in range(num_ssu)]


def _min_gap(records: Sequence[dict]) -> float | None:
    return min(
        (
            later["time_ms"] - earlier["time_ms"]
            for earlier, later in zip(records, records[1:])
        ),
        default=None,
    )


def _validate_pfo(
    summary: dict, prefix_stats: dict, num_ssu: int, prefix: str
) -> dict:
    expected_forecast = _forecast_from_manifest(
        prefix_stats.get("demand_gbps_by_ssu", ()), num_ssu
    )
    reported_forecast = summary.get("pfo_materialization_forecast")
    _require(
        _equivalent_derived(reported_forecast, expected_forecast),
        f"{prefix}: H70 forecast differs from frozen manifest policy",
    )
    mask = expected_forecast["materialized_ssu_mask"]
    _require(summary.get("pfo_materialized_ssu_mask") == mask, f"{prefix}: summary mask")
    _require(
        summary.get("pfo_materialized_ssu_count") == sum(mask),
        f"{prefix}: materialized SSU count",
    )
    _require(
        summary.get("pfo_controller") == "ProtectedFloorSchemeBController"
        and summary.get("pfo_materialized_allocation_stages")
        == ["selected_protected_floor"]
        and summary.get("pfo_path_table_ownership") == "exclusive_complete_table"
        and summary.get("pfo_required_downstream_deadband_gbps") == 0.0
        and summary.get("pfo_internal_deadband_gbps") == 0.05,
        f"{prefix}: PFO controller identity/ownership differs",
    )
    pressure_fields = (
        "pfo_pressure_reads",
        "pressure_reports",
        "pressure_cache_hits",
        "measurement_pressure_reports",
        "measurement_pressure_cache_hits",
        "measurement_pressure_requests",
    )
    _require(
        all(summary.get(field) == 0 for field in pressure_fields),
        f"{prefix}: H70 read Path pressure",
    )
    for field in (
        "pressure_reports_by_ssu",
        "pressure_cache_hits_by_ssu",
        "measurement_pressure_reports_by_ssu",
        "measurement_pressure_cache_hits_by_ssu",
        "measurement_pressure_requests_by_ssu",
    ):
        _require(summary.get(field) == [0] * num_ssu, f"{prefix}: {field} nonzero")
    records = summary.get("pfo_evaluation_audit_records")
    _require(isinstance(records, list) and records, f"{prefix}: PFO audit records missing")
    _require(summary.get("pfo_evaluation_audit_schema") == "pfo_update_audit_v2", f"{prefix}: PFO audit schema")
    parsed = [_audit_record(record, index, num_ssu, mask, prefix) for index, record in enumerate(records)]
    _require(
        [record["evaluation"] for record in parsed] == list(range(1, len(parsed) + 1)),
        f"{prefix}: PFO evaluation sequence",
    )
    _require(
        all(later["time_ms"] > earlier["time_ms"] for earlier, later in zip(parsed, parsed[1:])),
        f"{prefix}: PFO evaluation time order",
    )
    _require(
        all(
            left["install_state_hash"] == right["pre_state_hash"]
            for left, right in zip(records, records[1:])
        ),
        f"{prefix}: installed CIR is not the next observed actual CIR",
    )
    for index, record in enumerate(records):
        expected_verification = (
            "final_des_accept_zero_threshold"
            if index == len(records) - 1
            else "next_snapshot_exact"
        )
        _require(
            record["post_des_verification"] == expected_verification,
            f"{prefix}: DES verification boundary",
        )
    _require(
        summary.get("pfo_evaluation_sequence_contiguous") is True
        and summary.get("pfo_all_update_safety_gates_pass") is True,
        f"{prefix}: reported PFO safety gates failed",
    )
    _require(
        summary.get("pfo_evaluation_diagnostics_hash")
        == _canonical_hash(records, b"pfo-evaluation-diagnostics:v1\0"),
        f"{prefix}: PFO audit hash",
    )
    start_ms = float(summary["measurement_start_ms"])
    end_ms = float(summary["measurement_end_ms"])
    measurement_pairs = [
        (record, item)
        for record, item in zip(records, parsed)
        if start_ms <= item["time_ms"] < end_ms
    ]
    measurement_raw = [pair[0] for pair in measurement_pairs]
    measurement = [pair[1] for pair in measurement_pairs]
    _require(bool(measurement), f"{prefix}: no PFO evaluations in measurement")
    _require(
        summary.get("pfo_measurement_evaluation_diagnostics_hash")
        == _canonical_hash(measurement_raw, b"pfo-measurement-diagnostics:v1\0"),
        f"{prefix}: measurement PFO audit hash",
    )
    all_gap = _min_gap(parsed)
    measurement_gap = _min_gap(measurement)
    if all_gap is not None:
        _require(all_gap >= 25.0 - 1e-9, f"{prefix}: PFO interval below 25 ms")
    _require(summary.get("pfo_control_interval_respected") is True, f"{prefix}: interval gate")
    if summary.get("pfo_min_observed_control_interval_ms") is not None:
        _close(summary["pfo_min_observed_control_interval_ms"], all_gap, f"{prefix}: min PFO gap")
    if summary.get("pfo_measurement_min_observed_control_interval_ms") is not None:
        _close(summary["pfo_measurement_min_observed_control_interval_ms"], measurement_gap, f"{prefix}: measurement min gap")

    def aggregate(selected: Sequence[dict], scope: str) -> dict:
        writes_by_ssu = _sum_record_vectors(selected, "writes_by_ssu", num_ssu)
        tx_by_ssu = _sum_record_vectors(selected, "transactions_by_ssu", num_ssu)
        decrease_by_ssu = _sum_record_vectors(selected, "decrease_transactions_by_ssu", num_ssu)
        increase_by_ssu = _sum_record_vectors(selected, "increase_transactions_by_ssu", num_ssu)
        safety_by_ssu = _sum_record_vectors(selected, "safety_by_ssu", num_ssu)
        prefix_name = "pfo_total" if scope == "total" else "pfo_measurement"
        _require(summary[f"{prefix_name}_control_evaluations"] == len(selected), f"{prefix}: {scope} evaluations")
        _require(summary[f"{prefix_name}_planned_cir_path_writes_by_ssu"] == writes_by_ssu, f"{prefix}: {scope} writes by SSU")
        _require(summary[f"{prefix_name}_planned_cir_path_writes"] == sum(writes_by_ssu), f"{prefix}: {scope} writes")
        _require(summary[f"{prefix_name}_planned_cir_write_transactions_by_ssu"] == tx_by_ssu, f"{prefix}: {scope} transactions by SSU")
        _require(summary[f"{prefix_name}_planned_cir_write_transactions"] == sum(tx_by_ssu), f"{prefix}: {scope} transactions")
        _require(summary[f"{prefix_name}_decrease_phase_transactions_by_ssu"] == decrease_by_ssu, f"{prefix}: {scope} decrease transactions")
        _require(summary[f"{prefix_name}_increase_phase_transactions_by_ssu"] == increase_by_ssu, f"{prefix}: {scope} increase transactions")
        _require(summary[f"{prefix_name}_decrease_phase_transactions"] == sum(decrease_by_ssu), f"{prefix}: {scope} decrease total")
        _require(summary[f"{prefix_name}_increase_phase_transactions"] == sum(increase_by_ssu), f"{prefix}: {scope} increase total")
        expected_commits = sum(any(record["writes_by_ssu"]) for record in selected)
        _require(summary[f"{prefix_name}_planned_cir_commits"] == expected_commits, f"{prefix}: {scope} commits")
        if scope == "measurement":
            _require(summary["pfo_measurement_safety_forced_cir_path_writes_by_ssu"] == safety_by_ssu, f"{prefix}: safety writes by SSU")
            _require(summary["pfo_measurement_safety_forced_cir_path_writes"] == sum(safety_by_ssu), f"{prefix}: safety writes")
            _require(summary["pfo_measurement_required_floor_increases"] == sum(item["required_increases"] for item in selected), f"{prefix}: required increases")
            _require(summary["pfo_measurement_capacity_compensation_decreases"] == sum(item["capacity_decreases"] for item in selected), f"{prefix}: capacity decreases")
            _require(summary["pfo_measurement_deadband_holds"] == sum(item["deadband_holds"] for item in selected), f"{prefix}: deadband holds")
            _require(summary["pfo_measurement_active_set_hold_violations"] == 0, f"{prefix}: active holds")
        return {
            "writes_by_ssu": writes_by_ssu,
            "transactions_by_ssu": tx_by_ssu,
            "commits": expected_commits,
        }

    total = aggregate(parsed, "total")
    measured = aggregate(measurement, "measurement")
    _require(summary["pfo_total_safety_forced_cir_path_writes"] == sum(sum(item["safety_by_ssu"]) for item in parsed), f"{prefix}: total safety writes")
    _require(
        summary["control_evaluations"] == len(parsed)
        == summary["pfo_controller_evaluations"]
        == summary["pfo_controller_decisions"]
        and summary["cir_path_writes"] == sum(total["writes_by_ssu"])
        and summary["cir_write_transactions"] == sum(total["transactions_by_ssu"])
        and summary["cir_commits"] == total["commits"],
        f"{prefix}: total PFO planned/DES counters differ",
    )
    _require(
        summary["measurement_control_evaluations"] == len(measurement)
        and summary["measurement_cir_path_writes"] == sum(measured["writes_by_ssu"])
        and summary["measurement_cir_write_transactions"] == sum(measured["transactions_by_ssu"])
        and summary["measurement_cir_commits"] == measured["commits"],
        f"{prefix}: measurement PFO planned/DES counters differ",
    )
    hot_writes = sum(value for selected, value in zip(mask, measured["writes_by_ssu"]) if selected)
    cold_writes = sum(value for selected, value in zip(mask, measured["writes_by_ssu"]) if not selected)
    _require(
        summary["pfo_measurement_hot_ssu_cir_path_writes"] == hot_writes
        and summary["pfo_measurement_cold_ssu_cir_path_writes"] == cold_writes == 0,
        f"{prefix}: cold SSU received a CIR write",
    )
    for field in (
        "actual_cir_sum_gbps_by_ssu_at_stop",
        "max_actual_cir_sum_gbps_by_ssu",
        "measurement_actual_cir_sum_gbps_by_ssu_at_start",
        "measurement_actual_cir_sum_gbps_by_ssu_at_end",
    ):
        values = summary[field]
        _require(
            all(selected or abs(float(value)) <= 1e-12 for selected, value in zip(mask, values)),
            f"{prefix}: cold SSU actual CIR nonzero in {field}",
        )
    return {
        "forecast": expected_forecast,
        "materialized_ssu_mask": mask,
        "materialized_ssu_count": sum(mask),
        "cold_ssu_count": len(mask) - sum(mask),
        "measurement_cold_ssu_cir_path_writes": cold_writes,
        "measurement_hot_ssu_cir_path_writes": hot_writes,
        "pfo_audit_record_count": len(parsed),
        "pfo_measurement_audit_record_count": len(measurement),
        "pfo_audit_gate_passed": True,
    }


def _validate_row(
    row: dict,
    shard: dict,
    plan: dict,
    source_audit: dict,
) -> dict:
    prefix = f"{shard['path']}:{row.get('case')}/SSU{row.get('num_ssu')}"
    _require(isinstance(row, dict), f"{prefix}: row must be an object")
    case_name = row.get("case")
    _require(case_name in CASE_TO_ROLE, f"{prefix}: row outside formal matrix")
    role = CASE_TO_ROLE[case_name]
    case = EXPECTED_CASES[case_name]
    num_ssu = _integer(row.get("num_ssu"), f"{prefix}: num_ssu", minimum=1)
    _require(num_ssu in FORMAL_SSUS, f"{prefix}: SSU outside formal matrix")
    _require(
        row.get("status") == "ok"
        and row.get("role") == role
        and row.get("family") == case["family"]
        and row.get("kind") == case["kind"]
        and row.get("case_spec") == case
        and row.get("definition") == DEFINITION
        and row.get("definition_fingerprint") == plan["definition_fingerprint"]
        and row.get("num_npu") == NUM_NPU
        and row.get("backing_requests_per_npu")
        == plan["backing_requests_per_npu"],
        f"{prefix}: row identity/shape differs",
    )
    _require(
        row.get("source_fingerprint") == shard["source_fingerprint"]
        and row.get("config_fingerprint") == shard["config_fingerprint"]
        and row.get("campaign_spec_sha256") == plan["file_sha256"],
        f"{prefix}: row provenance differs",
    )
    _require(
        row.get("case_fingerprint")
        == _case_fingerprint(
            case,
            num_ssu,
            shard["source_fingerprint"],
            shard["config_fingerprint"],
        ),
        f"{prefix}: case fingerprint",
    )
    inputs = row.get("input_fingerprints")
    _require(
        isinstance(inputs, dict)
        and set(inputs) == INPUT_FINGERPRINT_FIELDS
        and all(_is_sha256(value) for value in inputs.values()),
        f"{prefix}: input fingerprints",
    )
    workload = shard["spec"]["workload"]
    expected_input = {
        "catalog": workload["catalog"],
        "recipe": workload["recipe"],
        "schedule": workload["schedule"],
        "assignment": workload["assignment"],
        "prefix_32_assignment": workload["prefix_32_assignment_hash"],
        "full_assignment": workload["full_assignment_hash"],
    }
    _require(
        all(inputs[field] == value for field, value in expected_input.items()),
        f"{prefix}: row/spec schedule inputs differ",
    )
    prefix_materialized = row.get("prefix_32_materialized_fingerprints")
    _require(
        isinstance(prefix_materialized, dict)
        and set(prefix_materialized) == {"workload", "placement", "trace"}
        and all(_is_sha256(value) for value in prefix_materialized.values()),
        f"{prefix}: prefix fingerprints",
    )
    full_stats = row.get("workload_statistics")
    prefix_stats = row.get("prefix_32_workload_statistics")
    _require(isinstance(full_stats, dict) and isinstance(prefix_stats, dict), f"{prefix}: workload statistics")
    _require(
        full_stats.get("seed") == shard["seed"]
        and full_stats.get("requests_per_npu") == plan["backing_requests_per_npu"]
        and full_stats.get("request_count")
        == NUM_NPU * plan["backing_requests_per_npu"],
        f"{prefix}: full workload shape",
    )
    _require(
        prefix_stats.get("seed") == shard["seed"]
        and prefix_stats.get("requests_per_npu") == FORECAST_REQUESTS_PER_NPU
        and prefix_stats.get("request_count") == NUM_NPU * FORECAST_REQUESTS_PER_NPU,
        f"{prefix}: scientific prefix shape",
    )
    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{prefix}: steady summary missing")
    _require(
        summary.get("schema_version") == SUMMARY_SCHEMA_VERSION
        and summary.get("mode") == "steady_state_full_load"
        and summary.get("num_npu") == NUM_NPU
        and summary.get("num_ssu") == num_ssu
        and summary.get("n_layers") == N_LAYERS
        and summary.get("batch_size") == BATCH_SIZE
        and summary.get("warmup_requests_per_npu")
        == plan["warmup_requests_per_npu"]
        and summary.get("settle_ms") == plan["settle_ms"]
        and summary.get("slo_alpha") == PRIMARY_ALPHA,
        f"{prefix}: summary identity/shape",
    )
    start_ms = _finite(summary.get("measurement_start_ms"), f"{prefix}: measurement start")
    end_ms = _finite(summary.get("measurement_end_ms"), f"{prefix}: measurement end")
    _close(end_ms - start_ms, plan["measurement_ms"], f"{prefix}: measurement endpoints", abs_tol=1e-8, rel_tol=0.0)
    _close(summary.get("measurement_duration_ms"), plan["measurement_ms"], f"{prefix}: measurement duration")
    _require(
        inputs["simulator"] == summary.get("input_fingerprint"),
        f"{prefix}: simulator fingerprint differs",
    )
    invariants = summary.get("invariants")
    _require(
        isinstance(invariants, dict) and set(invariants) == EXPECTED_INVARIANTS,
        f"{prefix}: invariant catalog differs",
    )
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{prefix}: failed invariants {failed}")
    request_info = _validate_requests(summary, shard["schedule"], prefix)
    _require(
        row.get("measurement_cohort_fingerprint") == request_info["cohort_hash"],
        f"{prefix}: cohort fingerprint",
    )
    controls = _validate_control_metrics(summary, case, num_ssu, plan["measurement_ms"], prefix)
    resources = _validate_resources(summary, num_ssu, plan["measurement_ms"], prefix)
    stationarity = _validate_stationarity(
        summary,
        row.get("stationarity_diagnostics"),
        request_info,
        num_ssu,
        plan,
        source_audit["fleet_layer_burst_bound_gb"],
        prefix,
    )
    _require(stationarity["stationarity_gate_passed"], f"{prefix}: stationarity gate failed")
    pfo = _validate_pfo(summary, prefix_stats, num_ssu, prefix) if role == "H70" else None
    row_gates = {
        "schema3_campaign_binding": True,
        "source_config_input_pairing": True,
        "all_29_invariants": True,
        "all_32_npus_sampled": True,
        "sample_target_min_8": min(request_info["counts"]) >= MIN_REQUESTS_PER_NPU,
        "backing_margin_min_32": request_info["backing_margin"] >= MIN_BACKING_MARGIN,
        "stationarity": stationarity["stationarity_gate_passed"],
        "pfo_audit_if_applicable": pfo is None or pfo["pfo_audit_gate_passed"],
    }
    return {
        "seed": shard["seed"],
        "num_ssu": num_ssu,
        "case": case_name,
        "role": role,
        "case_spec": case,
        "config_fingerprint": shard["config_fingerprint"],
        "input_fingerprints": inputs,
        "prefix_materialized_fingerprints": prefix_materialized,
        "workload_statistics": full_stats,
        "prefix_workload_statistics": prefix_stats,
        "measurement_cohort_fingerprint": request_info["cohort_hash"],
        "measurement_request_count": len(request_info["rows"]),
        "requests_per_npu_min": min(request_info["counts"]),
        "requests_per_npu_median": statistics.median(request_info["counts"]),
        "requests_per_npu_max": max(request_info["counts"]),
        "backing_margin_fastest_npu": request_info["backing_margin"],
        "equal_npu_slo_alpha2_pct": 100.0 * request_info["alpha2"]["equal_npu"],
        "request_weighted_slo_alpha2_pct": 100.0
        * request_info["alpha2"]["request_weighted"],
        "equal_npu_slo_alpha15_pct": 100.0 * request_info["alpha15"]["equal_npu"],
        "request_weighted_slo_alpha15_pct": 100.0
        * request_info["alpha15"]["request_weighted"],
        "mean_ttft_ms": request_info["mean_ttft_ms"],
        "p99_ttft_ms": request_info["p99_ttft_ms"],
        "max_equal_npu_single_outcome_weight_pp": request_info[
            "max_equal_npu_single_outcome_weight_pp"
        ],
        "wall_time_s": _nonnegative(row.get("wall_time_s"), f"{prefix}: wall time"),
        "stationarity": stationarity,
        "pfo": pfo,
        "row_gates": row_gates,
        "row_gate_passed": all(row_gates.values()),
        **controls,
        **resources,
    }


def _campaign_projection(spec: dict) -> dict:
    value = json.loads(json.dumps(spec))
    for field in (
        "seed",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment_hash",
        "full_assignment_hash",
    ):
        value["workload"].pop(field, None)
    value["steady_state"].pop("seed", None)
    return value


def _validate_campaign(
    shards: Sequence[dict], plan: dict, source_audit: dict
) -> dict:
    _require(bool(shards), "at least one shard is required")
    _require(
        all(shard["source_fingerprint"] == source_audit["source_fingerprint"] for shard in shards),
        "shard source differs from independently audited closure",
    )
    _require(
        all(shard["authentication"] == source_audit["data_authentication"] for shard in shards),
        "shard data authentication differs",
    )
    projections = [_campaign_projection(shard["spec"]) for shard in shards]
    _require(
        all(_equivalent_derived(value, projections[0]) for value in projections),
        "experiment configuration differs beyond seed-bound workload fields",
    )
    rows = {}
    provenance = {}
    seed_configs: dict[int, str] = {}
    seed_specs: dict[int, dict] = {}
    seed_schedules: dict[int, list] = {}
    for shard in shards:
        seed = shard["seed"]
        if seed in seed_configs:
            _require(seed_configs[seed] == shard["config_fingerprint"], f"seed{seed}: config fingerprint differs")
            _require(shard["spec"] == seed_specs[seed], f"seed{seed}: spec differs")
            _require(shard["schedule"]["assignments"] == seed_schedules[seed], f"seed{seed}: schedule differs")
        else:
            seed_configs[seed] = shard["config_fingerprint"]
            seed_specs[seed] = shard["spec"]
            seed_schedules[seed] = shard["schedule"]["assignments"]
        for raw_row in shard["payload"]["results"]:
            row = _validate_row(raw_row, shard, plan, source_audit)
            key = (row["seed"], row["num_ssu"], row["role"])
            _require(key not in rows, f"duplicate formal row {key}")
            rows[key] = row
            provenance[key] = {"path": shard["path"], "sha256": shard["sha256"]}
    expected = {
        (seed, num_ssu, role)
        for seed in FORMAL_SEEDS
        for num_ssu in FORMAL_SSUS
        for role in ROLE_ORDER
    }
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    _require(not extra, f"campaign has rows outside formal plan: {extra}")
    _require(not missing, f"campaign is incomplete; missing {missing}")
    _require(len(rows) == plan["expected_result_count"], "formal result count differs")
    for seed in FORMAL_SEEDS:
        seed_rows = [row for key, row in rows.items() if key[0] == seed]
        for field in CROSS_SSU_INPUT_FIELDS:
            _require(
                len({row["input_fingerprints"][field] for row in seed_rows}) == 1,
                f"seed{seed}: cross-SSU {field} differs",
            )
        for num_ssu in FORMAL_SSUS:
            cell = [rows[(seed, num_ssu, role)] for role in ROLE_ORDER]
            for field in INPUT_FINGERPRINT_FIELDS:
                _require(
                    len({row["input_fingerprints"][field] for row in cell}) == 1,
                    f"seed{seed}/SSU{num_ssu}: unpaired {field}",
                )
            reference = cell[0]
            for current in cell[1:]:
                _require(
                    _equivalent_derived(current["workload_statistics"], reference["workload_statistics"]),
                    f"seed{seed}/SSU{num_ssu}: backing statistics differ",
                )
                _require(
                    _equivalent_derived(current["prefix_workload_statistics"], reference["prefix_workload_statistics"]),
                    f"seed{seed}/SSU{num_ssu}: prefix statistics differ",
                )
                _require(
                    current["prefix_materialized_fingerprints"]
                    == reference["prefix_materialized_fingerprints"],
                    f"seed{seed}/SSU{num_ssu}: prefix materialization differs",
                )
    _require(
        len({rows[(seed, FORMAL_SSUS[0], "B")]["input_fingerprints"]["assignment"] for seed in FORMAL_SEEDS})
        == len(FORMAL_SEEDS),
        "different seeds reused the same assignment",
    )
    return {
        "rows": rows,
        "provenance": provenance,
        "seed_configs": {str(seed): seed_configs[seed] for seed in FORMAL_SEEDS},
        "complete": True,
        "expected_result_count": len(expected),
        "validated_result_count": len(rows),
    }
