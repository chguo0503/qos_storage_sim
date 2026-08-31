"""Strict, simulation-independent audit of confirm32 experiment shards.

The program never imports the experiment runner or simulator.  It rebuilds
the authenticated 84-profile catalog, IID assignments, measurement metrics,
control rates, resource conservation, and 500-ms stationarity blocks from the
JSON artifacts.  Only a complete 3-seed x 3-SSU x 5-case campaign can freeze
an A* choice.
"""

from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import struct
import tarfile
from typing import Mapping, Sequence

import numpy as np


ANALYSIS_SCHEMA_VERSION = 1
SHARD_SCHEMA_VERSION = 2
SUMMARY_SCHEMA_VERSION = 2
ANALYSIS_NAME = "confirm32_analysis_v1"
SELECTION_RULE_VERSION = "confirm32-preregistered-selection-v1"
EXPERIMENT = "32npu_threshold_interval_factorial_v1"
DEFINITION = "confirm32"
NUM_NPU = 32
N_LAYERS = 16
BATCH_SIZE = 1
FORMAL_SEEDS = (42, 43, 44)
FORMAL_SSUS = (6, 10, 18)
ROLE_ORDER = ("B", "A0", "threshold-only", "interval-only", "combined")
ROLE_TO_CASE = {
    "B": "baseline",
    "A0": "adaptive_t0_i25ms",
    "threshold-only": "adaptive_t0p05_i25ms",
    "interval-only": "adaptive_t0_i50ms",
    "combined": "adaptive_t0p05_i50ms",
}
CASE_TO_ROLE = {case: role for role, case in ROLE_TO_CASE.items()}
EXPECTED_CASES = (
    {
        "name": "baseline",
        "family": "baseline",
        "kind": "baseline",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 0.0,
    },
    {
        "name": "adaptive_t0_i25ms",
        "family": "factorial",
        "kind": "adaptive",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 25.0,
    },
    {
        "name": "adaptive_t0p05_i25ms",
        "family": "factorial",
        "kind": "adaptive",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.05,
        "min_interval_ms": 25.0,
    },
    {
        "name": "adaptive_t0_i50ms",
        "family": "factorial",
        "kind": "adaptive",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 50.0,
    },
    {
        "name": "adaptive_t0p05_i50ms",
        "family": "factorial",
        "kind": "adaptive",
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.05,
        "min_interval_ms": 50.0,
    },
)
EXPECTED_CASE_BY_NAME = {case["name"]: case for case in EXPECTED_CASES}
EXPECTED_ADAPTIVE_DEFINITION = {
    "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
    "explicit_spill_threshold": 0.75,
    "target_ratio": 0.52,
    "required_ratio": 0.5,
    "background_reserve_fraction": 0.05,
}
EXPECTED_ADAPTIVE_SPEC = {
    **EXPECTED_ADAPTIVE_DEFINITION,
    "ssd_cap_gbps": 40.0,
    "npu_cap_gbps": 50.0,
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
CATEGORIES = ("SS", "SL", "LS", "LL")
PRIMARY_ALPHA = 2.0
SENSITIVITY_ALPHA = 1.5
SLO_EPSILON = 1e-12
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
DEFINITION_DEFAULT_BACKING_REQUESTS = 128
WARMUP_REQUESTS = 8
SETTLE_MS = 500.0
DEFINITION_DEFAULT_MEASUREMENT_MS = 8_000.0
MIN_FORMAL_MEASUREMENT_MS = 8_000.0
FORMAL_BLOCK_MS = 500.0
ACTIVE_BACKING_REQUESTS = DEFINITION_DEFAULT_BACKING_REQUESTS
ACTIVE_MEASUREMENT_MS = DEFINITION_DEFAULT_MEASUREMENT_MS
ACTIVE_BLOCK_MS = FORMAL_BLOCK_MS
ACTIVE_BLOCKS = 16
ACTIVE_BOUNDARIES = 17
MIN_REQUESTS_HARD = 4
MIN_REQUESTS_FREEZE_TARGET = 8
MIN_BACKING_MARGIN = 32
UTIL_NONINFERIOR_MARGIN_PP = 0.5
SLO_NONINFERIOR_MARGIN_PP = 1.0
P99_NONINFERIOR_RELATIVE_MARGIN = 0.01
WRITE_REDUCTION_MIN = 0.50
EVALUATION_REDUCTION_MIN = 0.40
FAR_BETTER_SLO_PP = 5.0
UTIL_HALF_LIMIT_PP = 1.0
UTIL_TREND_LIMIT_PP = 2.0
SERVED_HALF_RELATIVE_LIMIT = 0.02
EXPECTED_STATIONARITY_SEMANTICS = (
    "read-only left-limit snapshot before workload events at the same time"
)
EXPECTED_CONTROL_WINDOW = "half-open [measurement_start_ms, measurement_end_ms)"
SHA256_HEX = frozenset("0123456789abcdef")
RNG_KEY = struct.Struct("!QIII")
IID_NAMESPACE = b"random-steady-state:iid-uniform-profile:v1\0"
INITIAL_JITTER_NAMESPACE = b"steady-state:initial-jitter:v1\0"
BLOCK_VNODE_NAMESPACE = b"qos_storage_sim:block_ring_hash:vnode:v1\0"
BLOCK_KEY_NAMESPACE = b"qos_storage_sim:block_ring_hash:block:v1\0"
SIMULATOR_INPUT_NAMESPACE = b"full-prefill-microbatch-des-input-v2\0"
BLOCK_SIZE = 128
BLOCK_RING_VIRTUAL_NODES = 256
MATERIALIZATION_CACHE: dict[tuple[str, int, int], dict] = {}


class ValidationError(ValueError):
    """An artifact is unsafe for the confirm32 conclusion."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _configure_formal_plan(
    payloads: Sequence[dict],
    *,
    expected_measurement_ms: float | None,
    expected_backing: int | None,
) -> dict:
    """Freeze one measurement/backing shape before validating any shard.

    The confirm32 definition fingerprint intentionally authenticates its 8-s /
    128-request defaults, while a longer calibration run may override both in
    the immutable experiment spec.  Every shard supplied to one audit must use
    exactly the same override.  This early pass prevents a valid 8-s shard and
    a valid 16-s shard from being silently combined into one campaign.
    """

    shapes = []
    for index, payload in enumerate(payloads):
        label = str(payload.get("_analysis_path", f"shard[{index}]"))
        spec = payload.get("experiment_spec")
        _require(isinstance(spec, dict), f"{label}: experiment_spec missing")
        workload = spec.get("workload")
        steady = spec.get("steady_state")
        _require(
            isinstance(workload, dict) and isinstance(steady, dict),
            f"{label}: workload/steady_state spec missing",
        )
        backing = _integer(
            workload.get("requests_per_npu"), f"{label}: plan backing", minimum=1
        )
        steady_backing = _integer(
            steady.get("requests_per_npu"),
            f"{label}: steady plan backing",
            minimum=1,
        )
        _require(backing == steady_backing, f"{label}: two backing values differ")
        measurement_ms = _finite(
            steady.get("measurement_ms"), f"{label}: plan measurement"
        )
        block_ms = _finite(steady.get("block_ms"), f"{label}: plan block")
        warmup = _integer(
            steady.get("warmup_requests_per_npu"),
            f"{label}: plan warmup",
            minimum=1,
        )
        settle_ms = _finite(steady.get("settle_ms"), f"{label}: plan settle")
        shapes.append((backing, measurement_ms, block_ms, warmup, settle_ms))
    _require(
        len(set(shapes)) == 1,
        "mixed formal plans are forbidden (backing/measurement/block/warmup/settle differ)",
    )
    backing, measurement_ms, block_ms, warmup, settle_ms = shapes[0]
    _require(
        backing >= DEFINITION_DEFAULT_BACKING_REQUESTS,
        "formal backing cannot be shorter than the confirm32 definition default",
    )
    _require(
        measurement_ms >= MIN_FORMAL_MEASUREMENT_MS,
        "formal measurement window cannot be shorter than 8000 ms",
    )
    _close(block_ms, FORMAL_BLOCK_MS, "formal stationarity block width")
    _require(warmup == WARMUP_REQUESTS, "formal warmup differs from confirm32")
    _close(settle_ms, SETTLE_MS, "formal settle interval")
    block_ratio = measurement_ms / block_ms
    block_count = int(round(block_ratio))
    _require(
        math.isclose(block_ratio, block_count, rel_tol=0.0, abs_tol=1e-12),
        "measurement window is not an exact multiple of block_ms",
    )
    _require(block_count >= 16 and block_count % 2 == 0, "invalid block count")
    if expected_measurement_ms is not None:
        _close(
            measurement_ms,
            expected_measurement_ms,
            "CLI --measurement-ms expectation",
        )
    if expected_backing is not None:
        _require(backing == expected_backing, "CLI --backing expectation differs")

    global ACTIVE_BACKING_REQUESTS
    global ACTIVE_MEASUREMENT_MS
    global ACTIVE_BLOCK_MS
    global ACTIVE_BLOCKS
    global ACTIVE_BOUNDARIES
    ACTIVE_BACKING_REQUESTS = backing
    ACTIVE_MEASUREMENT_MS = measurement_ms
    ACTIVE_BLOCK_MS = block_ms
    ACTIVE_BLOCKS = block_count
    ACTIVE_BOUNDARIES = block_count + 1
    MATERIALIZATION_CACHE.clear()
    return {
        "backing_requests_per_npu": backing,
        "measurement_ms": measurement_ms,
        "block_ms": block_ms,
        "stationarity_blocks": block_count,
        "stationarity_boundaries": block_count + 1,
        "warmup_requests_per_npu": warmup,
        "settle_ms": settle_ms,
        "source": "identical immutable experiment_spec values from every input shard",
    }


def _canonical_hash(value, namespace: bytes = b"") -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_local_data(path: Path) -> tuple[dict, list[list]]:
    """Directly parse tracked ``data`` without importing sim or a cache loader."""

    try:
        raw = ast.literal_eval(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise ValidationError(
            f"cannot directly parse authenticated data: {error}"
        ) from error
    _require(isinstance(raw, dict) and raw, "data must contain a profile dictionary")
    table = {}
    for raw_key, raw_values in raw.items():
        key_value = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        _require(
            isinstance(key_value, (tuple, list)) and len(key_value) == 2,
            f"data profile key is malformed: {raw_key!r}",
        )
        key = (
            _integer(key_value[0], "data seq_len_k", minimum=1),
            _integer(key_value[1], "data nql", minimum=1),
        )
        _require(key not in table, f"duplicate data profile {key}")
        _require(
            isinstance(raw_values, (tuple, list)),
            f"data profile {key} values are malformed",
        )
        values = tuple(raw_values)
        if len(values) == 3:
            required_bw, per_layer_us, ttft_ms = values
            values = (
                required_bw,
                per_layer_us,
                ttft_ms,
                required_bw * per_layer_us / 1e6,
            )
        _require(len(values) == 4, f"data profile {key} must have 3/4 values")
        converted = tuple(_finite(value, f"data profile {key}") for value in values)
        _require(
            all(value > 0.0 for value in converted), f"data profile {key} nonpositive"
        )
        table[key] = converted
    rows = [
        [key[0], key[1], [float(value) for value in table[key]]]
        for key in sorted(table)
    ]
    return table, rows


def _validate_local_source_closure(
    payloads: Sequence[dict],
    source_root: Path,
    source_archive: Path | None,
) -> dict:
    """Bind artifact provenance to the exact local report source/data closure."""

    root = source_root.expanduser().resolve()
    _require(root.is_dir(), f"source root is not a directory: {root}")
    manifests = [payload.get("source_manifest") for payload in payloads]
    _require(
        manifests and isinstance(manifests[0], dict) and manifests[0],
        "source manifest missing before local closure audit",
    )
    _require(
        all(manifest == manifests[0] for manifest in manifests),
        "source manifests differ across supplied shards",
    )
    manifest = manifests[0]
    for relative_name, expected_hash in manifest.items():
        _require(
            isinstance(relative_name, str)
            and relative_name
            and not Path(relative_name).is_absolute()
            and ".." not in Path(relative_name).parts,
            f"unsafe source manifest path: {relative_name!r}",
        )
        path = (root / relative_name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValidationError(
                f"source manifest path escapes root: {relative_name!r}"
            ) from error
        _require(path.is_file(), f"source file is absent: {path}")
        _require(
            _file_sha256(path) == expected_hash,
            f"local source differs from shard manifest: {relative_name}",
        )

    data_path = root / "data"
    table, catalog_rows = _parse_local_data(data_path)
    _require(len(table) == 84, "direct local data parse did not produce 84 profiles")
    catalog_hash_rows = [[[row[0], row[1]], row[2]] for row in catalog_rows]
    catalog_hash = _canonical_hash(
        catalog_hash_rows, b"random-steady-state:data-catalog:v1\0"
    )
    table_hash = _canonical_hash(catalog_rows, b"authenticated-bw-table:v1\0")
    source_hash = _file_sha256(data_path)
    for payload in payloads:
        label = str(payload.get("_analysis_path", "shard"))
        authentication = payload.get("input_authentication")
        _require(isinstance(authentication, dict), f"{label}: input auth missing")
        _require(
            authentication.get("source_sha256") == source_hash
            and authentication.get("catalog_hash") == catalog_hash
            and authentication.get("table_fingerprint") == table_hash
            and authentication.get("profile_count") == 84,
            f"{label}: local direct data parse differs from authentication",
        )
        schedule = payload.get("schedule_metadata")
        _require(
            isinstance(schedule, dict) and schedule.get("catalog_rows") == catalog_rows,
            f"{label}: embedded catalog differs from local direct data parse",
        )
    archive_audit = None
    if source_archive is not None:
        archive_path = source_archive.expanduser().resolve()
        _require(archive_path.is_file(), f"source archive is absent: {archive_path}")
        try:
            with tarfile.open(archive_path, "r:*") as archive:
                members = archive.getmembers()
                _require(
                    all(member.isfile() for member in members),
                    "source archive contains a non-regular member",
                )
                names = [member.name for member in members]
                _require(
                    len(names) == len(set(names)) and set(names) == set(manifest),
                    "source archive closure differs from source manifest",
                )
                for member in members:
                    handle = archive.extractfile(member)
                    _require(
                        handle is not None, f"cannot read archive member {member.name}"
                    )
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                    _require(
                        digest.hexdigest() == manifest[member.name],
                        f"source archive member differs: {member.name}",
                    )
        except (OSError, tarfile.TarError) as error:
            raise ValidationError(f"cannot audit source archive: {error}") from error
        archive_audit = {
            "path": str(archive_path),
            "sha256": _file_sha256(archive_path),
            "size_bytes": archive_path.stat().st_size,
            "verified_regular_file_count": len(manifest),
            "all_member_hashes_match_source_manifest": True,
        }
    return {
        "source_root": str(root),
        "verified_file_count": len(manifest),
        "source_fingerprint": _canonical_hash(
            manifest, b"ms-scale-control-source:v1\0"
        ),
        "data_sha256": source_hash,
        "catalog_hash": catalog_hash,
        "table_fingerprint": table_hash,
        "profile_count": len(table),
        "cache_used": False,
        "method": "stdlib ast.literal_eval direct parse of data; no sim/cache import",
        "source_archive": archive_audit,
    }


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
    _require(math.isfinite(number), f"{name} is not finite")
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
    _require(isinstance(value, list), f"{name} must be a list")
    _require(len(value) == length, f"{name} must contain {length} entries")
    converter = _integer if integer else (_fraction if fraction else _nonnegative)
    return [converter(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _matrix(value, rows: int, columns: int, name: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == rows, f"{name} row shape")
    return [
        _vector(row, columns, f"{name}[{index}]") for index, row in enumerate(value)
    ]


def _compare_vectors(
    actual: Sequence,
    expected: Sequence,
    name: str,
    *,
    abs_tol: float = 1e-8,
) -> None:
    _require(len(actual) == len(expected), f"{name}: vector length differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        _close(left, right, f"{name}[{index}]", abs_tol=abs_tol)


def _equivalent_derived(actual, expected) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        left, right = float(actual), float(expected)
        return (
            math.isfinite(left)
            and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equivalent_derived(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equivalent_derived(left, right) for left, right in zip(actual, expected)
        )
    return type(actual) is type(expected) and actual == expected


def _classify_profile(key: tuple[int, int]) -> str:
    seq_len_k, nql = key
    return ("S" if seq_len_k <= 80 else "L") + ("L" if nql >= 512 else "S")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "cannot take percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _theil_sen_slope(times_ms: Sequence[float], values: Sequence[float]) -> float:
    _require(len(times_ms) == len(values) and len(values) >= 2, "invalid slope series")
    slopes = [
        (float(values[right]) - float(values[left]))
        * 1000.0
        / (float(times_ms[right]) - float(times_ms[left]))
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if float(times_ms[right]) > float(times_ms[left])
    ]
    _require(bool(slopes), "stationarity times are not increasing")
    return statistics.median(slopes)


def _expected_definition_dict() -> dict:
    return {
        "key": DEFINITION,
        "experiment_name": EXPERIMENT,
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssus": list(FORMAL_SSUS),
        "cases": list(EXPECTED_CASES),
        "default_requests_per_npu": DEFINITION_DEFAULT_BACKING_REQUESTS,
        "default_measurement_ms": DEFINITION_DEFAULT_MEASUREMENT_MS,
        "adaptive": EXPECTED_ADAPTIVE_DEFINITION,
        "report_roles": [[role, ROLE_TO_CASE[role]] for role in ROLE_ORDER],
        "require_single_ssu_simulation": False,
    }


EXPECTED_DEFINITION_FINGERPRINT = _canonical_hash(
    _expected_definition_dict(), b"ms-scale-control-definition:v1\0"
)


def _validate_spec(spec, prefix: str) -> tuple[int, dict]:
    _require(isinstance(spec, dict), f"{prefix}: experiment_spec missing")
    expected_top = {
        "schema_version",
        "experiment",
        "definition",
        "definition_fingerprint",
        "num_npu",
        "n_layers",
        "batch_size",
        "default_ssu_list",
        "cases",
        "report_roles",
        "workload",
        "steady_state",
        "adaptive",
        "cross_request_layer0_prefetch",
        "placement",
        "measurement_cost_scope",
        "pairing_scope",
        "diagnostics",
        "source_files",
    }
    _require(set(spec) == expected_top, f"{prefix}: experiment_spec fields changed")
    _require(spec["schema_version"] == SHARD_SCHEMA_VERSION, f"{prefix}: spec schema")
    _require(spec["experiment"] == EXPERIMENT, f"{prefix}: wrong experiment")
    _require(spec["definition"] == DEFINITION, f"{prefix}: wrong definition")
    _require(spec["num_npu"] == NUM_NPU, f"{prefix}: expected 32 NPU")
    _require(
        spec["n_layers"] == N_LAYERS and spec["batch_size"] == BATCH_SIZE,
        f"{prefix}: model shape",
    )
    _require(tuple(spec["default_ssu_list"]) == FORMAL_SSUS, f"{prefix}: SSU grid")
    _require(
        spec["cases"] == list(EXPECTED_CASES),
        f"{prefix}: case definitions/order changed",
    )
    _require(spec["report_roles"] == ROLE_TO_CASE, f"{prefix}: report roles changed")
    _require(
        spec["definition_fingerprint"] == EXPECTED_DEFINITION_FINGERPRINT,
        f"{prefix}: definition fingerprint",
    )
    _require(
        spec["adaptive"] == EXPECTED_ADAPTIVE_SPEC,
        f"{prefix}: Adaptive constants changed",
    )
    _require(
        spec["cross_request_layer0_prefetch"] is True, f"{prefix}: prefetch disabled"
    )
    _require(
        spec["placement"] == "token-block ring hash reused across all 16 layers",
        f"{prefix}: placement changed",
    )
    _require(
        spec["measurement_cost_scope"]
        == "true SSU pressure-table reads and CIR writes",
        f"{prefix}: measurement scope changed",
    )
    diagnostics = spec["diagnostics"]
    _require(
        isinstance(diagnostics, dict)
        and diagnostics.get("source_stable_false") == "invalidates the shard",
        f"{prefix}: diagnostics contract",
    )
    source_files = spec["source_files"]
    _require(
        isinstance(source_files, list) and len(source_files) == len(set(source_files)),
        f"{prefix}: source_files",
    )
    required_sources = {
        "data",
        "ms_scale_control_experiment.py",
        "continuous_batch_sim.py",
        "sim.py",
        "random_steady_state_workload.py",
        "authenticated_workload_inputs.py",
        "adaptive_admission_scheme_b_v2_1.py",
    }
    _require(required_sources <= set(source_files), f"{prefix}: critical source absent")

    workload = spec["workload"]
    _require(isinstance(workload, dict), f"{prefix}: workload spec missing")
    expected_workload_fields = {
        "mode",
        "seed",
        "requests_per_npu",
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment_hash",
        "full_assignment_hash",
        "sampling",
        "per_npu_streams",
        "scientific_prefix_requests_per_npu",
        "backing_prefix_reason",
        "authentication",
    }
    _require(
        set(workload) == expected_workload_fields, f"{prefix}: workload fields changed"
    )
    _require(
        workload["mode"] == "iid_uniform_profile_catalog_v1", f"{prefix}: workload mode"
    )
    seed = _integer(workload["seed"], f"{prefix}: seed")
    _require(seed in FORMAL_SEEDS, f"{prefix}: seed {seed} outside frozen plan")
    _require(
        workload["requests_per_npu"] == ACTIVE_BACKING_REQUESTS,
        f"{prefix}: backing differs from the frozen campaign plan",
    )
    for field in (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment_hash",
        "full_assignment_hash",
    ):
        _require(_is_sha256(workload[field]), f"{prefix}: invalid workload {field}")
    _require(
        workload["assignment"] == workload["full_assignment_hash"],
        f"{prefix}: assignment alias",
    )
    _require(
        workload["scientific_prefix_requests_per_npu"] == 32,
        f"{prefix}: scientific prefix",
    )
    _require(
        workload["sampling"]
        == "IID uniform with replacement over all 84 data profiles",
        f"{prefix}: sampling contract",
    )
    _require(
        workload["per_npu_streams"] == "independent and prefix-stable",
        f"{prefix}: stream contract",
    )
    authentication = workload["authentication"]
    _require(
        isinstance(authentication, dict)
        and set(authentication)
        == {
            "source",
            "source_sha256",
            "catalog_hash",
            "table_fingerprint",
            "profile_count",
        },
        f"{prefix}: authentication fields",
    )
    _require(
        authentication["source"] == "data" and authentication["profile_count"] == 84,
        f"{prefix}: unauthenticated/non-84 input",
    )
    for field in ("source_sha256", "catalog_hash", "table_fingerprint"):
        _require(_is_sha256(authentication[field]), f"{prefix}: authentication {field}")
    _require(
        authentication["catalog_hash"] == workload["catalog"],
        f"{prefix}: catalog authentication",
    )

    steady = spec["steady_state"]
    expected_steady_fields = {
        "seed",
        "requests_per_npu",
        "warmup_requests_per_npu",
        "settle_ms",
        "measurement_ms",
        "block_ms",
        "slo_alpha",
    }
    _require(
        isinstance(steady, dict) and set(steady) == expected_steady_fields,
        f"{prefix}: steady fields",
    )
    _require(
        steady["seed"] == seed
        and steady["requests_per_npu"] == ACTIVE_BACKING_REQUESTS,
        f"{prefix}: steady input differs",
    )
    _require(
        steady["warmup_requests_per_npu"] == WARMUP_REQUESTS,
        f"{prefix}: warmup changed",
    )
    _close(steady["settle_ms"], SETTLE_MS, f"{prefix}: settle")
    _close(
        steady["measurement_ms"],
        ACTIVE_MEASUREMENT_MS,
        f"{prefix}: measurement",
    )
    _close(steady["block_ms"], ACTIVE_BLOCK_MS, f"{prefix}: block")
    _close(steady["slo_alpha"], PRIMARY_ALPHA, f"{prefix}: alpha")
    return seed, authentication


def _recipe(seed: int, catalog_hash: str, requests_per_npu: int | None = None) -> dict:
    if requests_per_npu is None:
        requests_per_npu = ACTIVE_BACKING_REQUESTS
    return {
        "schema_version": 1,
        "mode": "iid_uniform_profile_catalog_v1",
        "seed": seed,
        "num_npu": NUM_NPU,
        "requests_per_npu": requests_per_npu,
        "categories": list(CATEGORIES),
        "category_policy": "category derived from an IID uniformly selected catalog profile",
        "profile_policy": (
            "IID uniform over the sorted data catalog with replacement per "
            "(seed, NPU, sequence); prefix-stable when extended"
        ),
        "request_order": "sequence-major",
        "request_id": "sequence * num_npu + npu_id",
        "initial_jitter": "existing deterministic 0-5 ms rule",
        "catalog_hash": catalog_hash,
    }


def _assignment_hash(rows: list[list]) -> str:
    return _canonical_hash(rows, b"random-steady-state:assignments:v1\0")


def _expected_profile_index(seed: int, npu_id: int, sequence: int, size: int) -> int:
    digest = hashlib.sha256(IID_NAMESPACE)
    digest.update(RNG_KEY.pack(seed, npu_id, sequence, 0))
    rng = np.random.RandomState(int.from_bytes(digest.digest()[:4], "big"))
    return int(rng.randint(size))


def _validate_seed_manifest(manifest, spec: dict, prefix: str) -> dict:
    _require(isinstance(manifest, dict), f"{prefix}: schedule_metadata missing")
    required = {
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "mode",
        "seed",
        "num_npu",
        "requests_per_npu",
        "request_id_formula",
        "catalog_rows",
        "assignment_rows",
    }
    _require(set(manifest) == required, f"{prefix}: seed manifest fields changed")
    workload = spec["workload"]
    for field in (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "mode",
        "seed",
        "requests_per_npu",
    ):
        _require(
            manifest[field] == workload[field], f"{prefix}: manifest {field} differs"
        )
    _require(manifest["num_npu"] == NUM_NPU, f"{prefix}: manifest NPU count")
    _require(
        manifest["request_id_formula"] == "sequence * num_npu + npu_id",
        f"{prefix}: request ID formula",
    )

    rows = manifest["catalog_rows"]
    _require(
        isinstance(rows, list) and len(rows) == 84,
        f"{prefix}: expected 84 catalog rows",
    )
    catalog: dict[tuple[int, int], tuple[float, ...]] = {}
    catalog_hash_rows = []
    for index, row in enumerate(rows):
        _require(
            isinstance(row, list) and len(row) == 3, f"{prefix}: catalog row {index}"
        )
        key = (
            _integer(row[0], f"{prefix}: catalog seq_len", minimum=1),
            _integer(row[1], f"{prefix}: catalog nql", minimum=1),
        )
        _require(key not in catalog, f"{prefix}: duplicate catalog profile {key}")
        _require(
            isinstance(row[2], list) and len(row[2]) == 4,
            f"{prefix}: profile {key} shape",
        )
        values = tuple(_finite(value, f"{prefix}: profile {key}") for value in row[2])
        _require(
            all(value > 0.0 for value in values), f"{prefix}: profile {key} nonpositive"
        )
        catalog[key] = values
        catalog_hash_rows.append([list(key), list(values)])
    _require(list(catalog) == sorted(catalog), f"{prefix}: catalog is not sorted")
    catalog_hash = _canonical_hash(
        catalog_hash_rows, b"random-steady-state:data-catalog:v1\0"
    )
    table_hash = _canonical_hash(rows, b"authenticated-bw-table:v1\0")
    authentication = workload["authentication"]
    _require(
        catalog_hash == workload["catalog"] == authentication["catalog_hash"],
        f"{prefix}: catalog hash mismatch",
    )
    _require(
        table_hash == authentication["table_fingerprint"],
        f"{prefix}: table fingerprint mismatch",
    )

    recipe_hash = _canonical_hash(
        _recipe(int(workload["seed"]), catalog_hash),
        b"random-steady-state:recipe:v1\0",
    )
    _require(recipe_hash == workload["recipe"], f"{prefix}: recipe hash mismatch")
    assignments = manifest["assignment_rows"]
    expected_count = NUM_NPU * ACTIVE_BACKING_REQUESTS
    _require(
        isinstance(assignments, list) and len(assignments) == expected_count,
        f"{prefix}: assignment count",
    )
    profile_keys = tuple(sorted(catalog))
    seed = int(workload["seed"])
    for request_id, row in enumerate(assignments):
        _require(
            isinstance(row, list) and len(row) == 5,
            f"{prefix}: assignment {request_id}",
        )
        _require(row[0] == request_id, f"{prefix}: non-contiguous request ID")
        npu_id = _integer(row[1], f"{prefix}: assignment NPU")
        sequence = _integer(row[2], f"{prefix}: assignment sequence")
        _require(
            npu_id < NUM_NPU and sequence < ACTIVE_BACKING_REQUESTS,
            f"{prefix}: assignment coordinate range",
        )
        _require(
            npu_id == request_id % NUM_NPU and sequence == request_id // NUM_NPU,
            f"{prefix}: request ID mapping {request_id}",
        )
        _require(row[3] in CATEGORIES, f"{prefix}: assignment category")
        _require(
            isinstance(row[4], list) and len(row[4]) == 2,
            f"{prefix}: assignment profile",
        )
        key = (
            _integer(row[4][0], f"{prefix}: assignment profile seq", minimum=1),
            _integer(row[4][1], f"{prefix}: assignment profile nql", minimum=1),
        )
        _require(key in catalog, f"{prefix}: assignment profile absent")
        _require(
            row[3] == _classify_profile(key), f"{prefix}: category/profile mismatch"
        )
        expected_key = profile_keys[
            _expected_profile_index(seed, npu_id, sequence, len(profile_keys))
        ]
        _require(
            key == expected_key,
            f"{prefix}: assignment {request_id} was not generated by frozen IID RNG",
        )
    assignment_hash = _assignment_hash(assignments)
    _require(
        assignment_hash == workload["assignment"], f"{prefix}: assignment hash mismatch"
    )
    schedule_hash = _canonical_hash(
        {"recipe_hash": recipe_hash, "assignments": assignments},
        b"random-steady-state:schedule:v1\0",
    )
    _require(schedule_hash == workload["schedule"], f"{prefix}: schedule hash mismatch")
    prefix_rows = [row for row in assignments if row[2] < 32]
    _require(
        _assignment_hash(prefix_rows) == workload["prefix_32_assignment_hash"],
        f"{prefix}: prefix assignment hash",
    )
    _require(
        assignment_hash == workload["full_assignment_hash"],
        f"{prefix}: full assignment hash",
    )
    return {
        "catalog": catalog,
        "assignments": assignments,
        "assignment_hash": assignment_hash,
        "seed": seed,
        "max_single_request_layer_gb": max(values[3] for values in catalog.values()),
        "fleet_layer_burst_bound_gb": NUM_NPU
        * max(values[3] for values in catalog.values()),
    }


def _json_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_array_digest() -> tuple[hashlib._Hash, list[bytes]]:
    digest = hashlib.sha256()
    digest.update(b"[")
    return digest, [b""]


def _json_array_digest_add(digest, separator: list[bytes], value: object) -> None:
    digest.update(separator[0])
    digest.update(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    separator[0] = b","


def _json_array_digest_finish(digest) -> str:
    digest.update(b"]")
    return digest.hexdigest()


def _addressed_rng(
    seed: int,
    npu_id: int,
    stream_id: int,
    generation: int,
    namespace: bytes,
) -> np.random.RandomState:
    digest = hashlib.sha256(namespace)
    digest.update(RNG_KEY.pack(seed, npu_id, stream_id, generation))
    return np.random.RandomState(int.from_bytes(digest.digest()[:4], "big"))


def _ring_position(namespace: bytes, first: int, second: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            namespace + struct.pack("!QQ", int(first), int(second))
        ).digest(),
        "big",
    )


@lru_cache(maxsize=None)
def _block_hash_ring(num_ssu: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    entries = sorted(
        (
            _ring_position(BLOCK_VNODE_NAMESPACE, ssu_id, virtual_node),
            ssu_id,
        )
        for ssu_id in range(num_ssu)
        for virtual_node in range(BLOCK_RING_VIRTUAL_NODES)
    )
    return (
        tuple(position for position, _ in entries),
        tuple(ssu_id for _, ssu_id in entries),
    )


def _block_ring_ssu(request_id: int, block_index: int, num_ssu: int) -> int:
    positions, ssu_ids = _block_hash_ring(num_ssu)
    block_position = _ring_position(BLOCK_KEY_NAMESPACE, request_id, block_index)
    ring_index = bisect.bisect_left(positions, block_position) % len(positions)
    return ssu_ids[ring_index]


def _sticky_placement(
    request_id: int,
    seq_len_k: int,
    nql: int,
    per_layer_kv_gb: float,
    num_ssu: int,
) -> tuple[tuple[int, float], ...]:
    total_tokens = int(round(float(seq_len_k) * 1024.0))
    ssd_tokens = total_tokens - int(round(float(nql)))
    _require(ssd_tokens > 0, "authenticated profile has no SSD-resident tokens")
    gb_per_token = float(per_layer_kv_gb) / ssd_tokens
    block_count = int(math.ceil(ssd_tokens / BLOCK_SIZE))
    return tuple(
        (
            _block_ring_ssu(request_id, block_index, num_ssu),
            float(
                min(BLOCK_SIZE, ssd_tokens - block_index * BLOCK_SIZE) * gb_per_token
            ),
        )
        for block_index in range(block_count)
    )


def _materialize_inputs(
    *,
    assignments: list[list],
    assignment_hash: str,
    catalog: dict[tuple[int, int], tuple[float, ...]],
    seed: int,
    num_ssu: int,
    requests_per_npu: int,
) -> dict:
    """Rebuild finite requests and sticky placement without runner/simulator imports."""

    cache_key = (assignment_hash, num_ssu, requests_per_npu)
    cached = MATERIALIZATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _require(
        len(assignments) == NUM_NPU * requests_per_npu,
        "materialization assignment length differs",
    )
    initial_starts = tuple(
        float(
            _addressed_rng(seed, npu_id, 0, 0, INITIAL_JITTER_NAMESPACE).uniform(
                0.0, 5.0
            )
        )
        for npu_id in range(NUM_NPU)
    )
    compute_us_by_npu = [0.0] * NUM_NPU
    for assignment in assignments:
        key = (int(assignment[4][0]), int(assignment[4][1]))
        compute_us_by_npu[int(assignment[1])] += float(catalog[key][1])
    compute_s_by_npu = [value / 1e6 for value in compute_us_by_npu]

    workload_digest, workload_separator = _json_array_digest()
    placement_digest, placement_separator = _json_array_digest()
    simulator_digest = hashlib.sha256(SIMULATOR_INPUT_NAMESPACE)
    demand_by_ssu = [0.0] * num_ssu
    for assignment in assignments:
        request_id, npu_id, sequence = map(int, assignment[:3])
        category = str(assignment[3])
        key = (int(assignment[4][0]), int(assignment[4][1]))
        required_bw, per_layer_us, _, per_layer_kv_gb = map(float, catalog[key])
        arrival_ms = initial_starts[npu_id]
        placement = _sticky_placement(
            request_id, key[0], key[1], per_layer_kv_gb, num_ssu
        )
        work_by_ssu = [0.0] * num_ssu
        for block_index, (ssu_id, block_gb) in enumerate(placement):
            work_by_ssu[ssu_id] += block_gb
            _json_array_digest_add(
                placement_digest,
                placement_separator,
                (request_id, block_index, ssu_id, block_gb),
            )
        for ssu_id, work_gb in enumerate(work_by_ssu):
            demand_by_ssu[ssu_id] += work_gb / compute_s_by_npu[npu_id]
        request_dict = {
            "request_id": request_id,
            "npu_id": npu_id,
            "stream_id": sequence,
            "generation": 0,
            "profile_key": key,
            "seq_len_k": key[0],
            "nql": key[1],
            "category": category,
            "required_bw_input_gbps": required_bw,
            "per_layer_us": per_layer_us,
            "per_layer_kv_gb": per_layer_kv_gb,
            "arrival_ms": arrival_ms,
            "arrival_time": arrival_ms,
            "initial": sequence == 0,
            "work_by_ssu_gb": tuple(work_by_ssu),
        }
        _json_array_digest_add(workload_digest, workload_separator, request_dict)
        simulator_digest.update(
            repr(
                (
                    request_id,
                    npu_id,
                    arrival_ms,
                    category,
                    per_layer_us,
                    (placement,),
                )
            ).encode()
        )
    workload_hash = _json_array_digest_finish(workload_digest)
    placement_hash = _json_array_digest_finish(placement_digest)
    result = {
        "workload": workload_hash,
        "placement": placement_hash,
        "trace": _json_hash(
            {
                "workload": workload_hash,
                "placement": placement_hash,
                "placement_reuse": "all_layers",
            }
        ),
        "simulator": simulator_digest.hexdigest(),
        "initial_npu_start_ms": list(initial_starts),
        "demand_gbps_by_ssu": demand_by_ssu,
    }
    MATERIALIZATION_CACHE[cache_key] = result
    return result


def _path_mapping() -> list[int]:
    return [(npu_id % 8) * 32 + 16 + (npu_id // 8) for npu_id in range(NUM_NPU)]


def _validate_path_abi(value, prefix: str) -> None:
    paths = _path_mapping()
    expected = {
        "path_count": 256,
        "group_count": 8,
        "paths_per_group": 32,
        "max_npu": 128,
        "assigned_count": NUM_NPU,
        "assigned_unique": NUM_NPU,
        "assigned_min": min(paths),
        "assigned_max": max(paths),
        "path_zero_reserved": True,
        "assigned_paths_sha256": _canonical_hash(
            paths, b"ms-scale-control-path-abi:v1\0"
        ),
    }
    _require(value == expected, f"{prefix}: Path ABI differs")


RUNTIME_SCIENTIFIC_FIELDS = (
    "hostname",
    "python",
    "python_full",
    "python_implementation",
    "numpy",
    "platform",
    "multiprocessing_start_method",
    "cpu_count",
)
RUNTIME_CAMPAIGN_FIELDS = (
    "python",
    "python_implementation",
    "numpy",
    "multiprocessing_start_method",
)


def _runtime_scientific(runtime, prefix: str, *, require_process: bool = True) -> dict:
    _require(isinstance(runtime, dict), f"{prefix}: runtime missing")
    _require(
        all(field in runtime for field in RUNTIME_SCIENTIFIC_FIELDS),
        f"{prefix}: runtime incomplete",
    )
    for field in (
        "hostname",
        "python",
        "python_full",
        "python_implementation",
        "numpy",
        "platform",
    ):
        _require(
            isinstance(runtime[field], str) and runtime[field],
            f"{prefix}: runtime {field}",
        )
    _require(
        runtime["python_implementation"] == "CPython",
        f"{prefix}: unsupported Python implementation",
    )
    _integer(runtime["cpu_count"], f"{prefix}: CPU count", minimum=1)
    _require(
        runtime["multiprocessing_start_method"] in ("spawn", "forkserver"),
        f"{prefix}: unsafe multiprocessing method",
    )
    if require_process:
        for field in ("pid", "rss_current_bytes", "rss_peak_bytes"):
            _require(field in runtime, f"{prefix}: process runtime {field} missing")
        _integer(runtime["pid"], f"{prefix}: pid", minimum=1)
        peak = _integer(runtime["rss_peak_bytes"], f"{prefix}: peak RSS", minimum=1)
        current_value = runtime["rss_current_bytes"]
        if current_value is not None:
            current = _integer(current_value, f"{prefix}: current RSS", minimum=1)
            _require(peak >= current, f"{prefix}: peak RSS below current RSS")
    return {field: runtime[field] for field in RUNTIME_SCIENTIFIC_FIELDS}


def _runtime_campaign_signature(runtime: Mapping[str, object]) -> dict:
    return {field: runtime[field] for field in RUNTIME_CAMPAIGN_FIELDS}


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


def _slo_metrics(request_rows: list[dict], alpha: float) -> dict:
    by_npu: dict[int, list[bool]] = defaultdict(list)
    outcomes = []
    for row in request_rows:
        outcome = (
            float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + SLO_EPSILON
        )
        by_npu[int(row["npu_id"])].append(outcome)
        outcomes.append(outcome)
    all_npus = set(by_npu) == set(range(NUM_NPU))
    return {
        "all_npus_sampled": all_npus,
        "equal_npu": (
            statistics.fmean(statistics.fmean(by_npu[npu]) for npu in range(NUM_NPU))
            if all_npus
            else None
        ),
        "request_weighted": statistics.fmean(outcomes) if outcomes else None,
        "count": len(outcomes),
        "counts_by_npu": [len(by_npu[npu]) for npu in range(NUM_NPU)],
    }


def _group_metric(outcomes: Sequence[bool]) -> dict:
    passed = sum(bool(value) for value in outcomes)
    return {
        "count": len(outcomes),
        "passed": passed,
        "slo_attainment": passed / len(outcomes) if outcomes else None,
    }


def _cohort_alpha2_metrics(
    request_rows: list[dict], catalog: dict[tuple[int, int], tuple[float, ...]]
) -> dict:
    categories: dict[str, list[bool]] = defaultdict(list)
    profiles: dict[tuple[int, int], list[bool]] = defaultdict(list)
    bins: dict[str, list[bool]] = defaultdict(list)
    profiles_by_npu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    demand_bins = (
        ("le_10", -math.inf, 10.0),
        ("gt_10_le_20", 10.0, 20.0),
        ("gt_20_le_40", 20.0, 40.0),
        ("gt_40_le_50", 40.0, 50.0),
        ("gt_50_le_80", 50.0, 80.0),
        ("gt_80", 80.0, math.inf),
    )
    for row in request_rows:
        key = tuple(row["profile_key"])
        outcome = bool(row["slo_met"])
        demand = catalog[key][0]
        categories[str(row["category"])].append(outcome)
        profiles[key].append(outcome)
        profiles_by_npu[int(row["npu_id"])].append(key)
        for name, lower, upper in demand_bins:
            if lower < demand <= upper:
                bins[name].append(outcome)
                break
    per_npu_demand = []
    per_npu_ms_per_gb = []
    for npu_id in range(NUM_NPU):
        keys = profiles_by_npu[npu_id]
        _require(bool(keys), f"cohort NPU{npu_id} has no requests")
        compute_s = math.fsum(catalog[key][1] for key in keys) / 1e6
        kv_gb = math.fsum(catalog[key][3] for key in keys)
        per_npu_demand.append(kv_gb / compute_s)
        per_npu_ms_per_gb.append(1000.0 * compute_s / kv_gb)
    demand_mean = statistics.fmean(per_npu_demand)
    ms_mean = statistics.fmean(per_npu_ms_per_gb)
    return {
        "category": {
            category: _group_metric(categories[category])
            for category in sorted(categories)
        },
        "raw_demand_bins": {
            name: _group_metric(bins.get(name, ())) for name, _, _ in demand_bins
        },
        "profile": {
            f"{key[0]},{key[1]}": {
                "raw_demand_gbps": catalog[key][0],
                **_group_metric(profiles.get(key, ())),
            }
            for key in sorted(catalog)
        },
        "profiles_observed": sum(bool(values) for values in profiles.values()),
        "realized_cohort": {
            "request_count": len(request_rows),
            "per_npu_raw_demand_gbps": {
                "min": min(per_npu_demand),
                "max": max(per_npu_demand),
                "mean": demand_mean,
                "coefficient_of_variation": (
                    statistics.pstdev(per_npu_demand) / demand_mean
                    if demand_mean > 0.0
                    else 0.0
                ),
            },
            "per_npu_ms_per_gb": {
                "min": min(per_npu_ms_per_gb),
                "max": max(per_npu_ms_per_gb),
                "mean": ms_mean,
                "spread": max(per_npu_ms_per_gb) - min(per_npu_ms_per_gb),
            },
            "fleet_raw_demand_gbps": math.fsum(per_npu_demand),
        },
    }


REQUEST_ROW_FIELDS = {
    "request_id",
    "npu_id",
    "sequence",
    "category",
    "profile_id",
    "profile_key",
    "profile_name",
    "raw_demand_gbps",
    "admission_time_ms",
    "completion_time_ms",
    "ttft_ms",
    "ideal_ttft_ms",
    "slo_met",
}


def _validate_requests(summary: dict, manifest_data: dict, prefix: str) -> dict:
    rows = summary.get("request_rows")
    _require(isinstance(rows, list) and rows, f"{prefix}: request_rows missing")
    _require(
        summary.get("measurement_request_count") == len(rows),
        f"{prefix}: request count",
    )
    start = _finite(summary.get("measurement_start_ms"), f"{prefix}: window start")
    end = _finite(summary.get("measurement_end_ms"), f"{prefix}: window end")
    catalog = manifest_data["catalog"]
    assignments = manifest_data["assignments"]
    ids = []
    by_npu: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        _require(
            isinstance(row, dict) and set(row) == REQUEST_ROW_FIELDS,
            f"{prefix}: request row {index} fields",
        )
        request_id = _integer(row["request_id"], f"{prefix}: request ID")
        _require(
            request_id < len(assignments), f"{prefix}: request ID outside manifest"
        )
        npu_id = _integer(row["npu_id"], f"{prefix}: request NPU")
        sequence = _integer(row["sequence"], f"{prefix}: request sequence")
        manifest = assignments[request_id]
        _require(
            npu_id == request_id % NUM_NPU
            and sequence == request_id // NUM_NPU
            and manifest[:3] == [request_id, npu_id, sequence],
            f"{prefix}: request {request_id} identity differs",
        )
        key_value = row["profile_key"]
        _require(
            isinstance(key_value, list) and len(key_value) == 2,
            f"{prefix}: profile key",
        )
        key = (
            _integer(key_value[0], f"{prefix}: profile seq", minimum=1),
            _integer(key_value[1], f"{prefix}: profile nql", minimum=1),
        )
        _require(
            list(key) == manifest[4] and key in catalog,
            f"{prefix}: request profile differs",
        )
        category = _classify_profile(key)
        _require(
            row["category"] == manifest[3] == category, f"{prefix}: request category"
        )
        _require(row["profile_id"] == f"{key[0]},{key[1]}", f"{prefix}: profile_id")
        _require(
            row["profile_name"] == f"seq_len_k={key[0]},nql={key[1]}",
            f"{prefix}: profile_name",
        )
        _close(row["raw_demand_gbps"], catalog[key][0], f"{prefix}: raw demand")
        ideal = N_LAYERS * catalog[key][1] / 1000.0
        _close(row["ideal_ttft_ms"], ideal, f"{prefix}: ideal TTFT")
        admission = _finite(row["admission_time_ms"], f"{prefix}: admission")
        completion = _finite(row["completion_time_ms"], f"{prefix}: completion")
        _require(
            start <= admission < end, f"{prefix}: admission outside half-open window"
        )
        _require(completion >= admission, f"{prefix}: completion before admission")
        _close(row["ttft_ms"], completion - admission, f"{prefix}: TTFT")
        expected_slo = float(row["ttft_ms"]) <= PRIMARY_ALPHA * ideal + SLO_EPSILON
        _require(
            isinstance(row["slo_met"], bool) and row["slo_met"] == expected_slo,
            f"{prefix}: alpha2 outcome",
        )
        ids.append(request_id)
        by_npu[npu_id].append(row)
    _require(
        ids == sorted(ids) and len(ids) == len(set(ids)),
        f"{prefix}: request IDs not unique/sorted",
    )
    _require(set(by_npu) == set(range(NUM_NPU)), f"{prefix}: not every NPU sampled")
    counts = [len(by_npu[npu]) for npu in range(NUM_NPU)]
    _require(
        min(counts) >= MIN_REQUESTS_HARD,
        f"{prefix}: fewer than {MIN_REQUESTS_HARD} completed tagged requests on an NPU",
    )
    _require(
        summary.get("request_counts_by_npu") == counts,
        f"{prefix}: per-NPU request counts",
    )
    alpha2 = _slo_metrics(rows, PRIMARY_ALPHA)
    alpha15 = _slo_metrics(rows, SENSITIVITY_ALPHA)
    _close(
        summary.get("ttft_slo_attainment"),
        alpha2["equal_npu"],
        f"{prefix}: equal-NPU alpha2",
    )
    _close(
        summary.get("request_weighted_slo_attainment"),
        alpha2["request_weighted"],
        f"{prefix}: request alpha2",
    )
    ttfts = [float(row["ttft_ms"]) for row in rows]
    _close(summary.get("mean_ttft_ms"), statistics.fmean(ttfts), f"{prefix}: mean TTFT")
    _close(summary.get("p99_ttft_ms"), _percentile(ttfts, 99), f"{prefix}: p99 TTFT")
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
    return {
        "rows": rows,
        "counts": counts,
        "minimum_completed_by_npu_from_tagged_sequences": [
            max(int(row["sequence"]) for row in by_npu[npu_id]) + 1
            for npu_id in range(NUM_NPU)
        ],
        "alpha2": alpha2,
        "alpha15": alpha15,
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p99_ttft_ms": _percentile(ttfts, 99),
        "cohort_hash": cohort_hash,
        "cohort_alpha2": _cohort_alpha2_metrics(rows, catalog),
        "sample_target_met": min(counts) >= MIN_REQUESTS_FREEZE_TARGET,
        "max_equal_npu_single_outcome_weight_pp": max(
            100.0 / (NUM_NPU * count) for count in counts
        ),
    }


def _validate_control_metrics(
    summary: dict, case: dict, num_ssu: int, duration_ms: float, prefix: str
) -> dict:
    duration_s = duration_ms / 1000.0

    def count_rate(count_field: str, rate_field: str) -> int:
        count = _integer(summary.get(count_field), f"{prefix}: {count_field}")
        _close(summary.get(rate_field), count / duration_s, f"{prefix}: {rate_field}")
        return count

    def count_vector(total_field: str, vector_field: str) -> list[int]:
        values = _vector(
            summary.get(vector_field),
            num_ssu,
            f"{prefix}: {vector_field}",
            integer=True,
        )
        _require(
            sum(values) == int(summary[total_field]), f"{prefix}: {vector_field} sum"
        )
        return values

    reports = count_rate(
        "measurement_pressure_reports", "measurement_pressure_report_rate_hz"
    )
    cache_hits = _integer(
        summary.get("measurement_pressure_cache_hits"), f"{prefix}: cache hits"
    )
    requests = _integer(
        summary.get("measurement_pressure_requests"), f"{prefix}: pressure requests"
    )
    evaluations = count_rate(
        "measurement_control_evaluations", "measurement_control_evaluation_rate_hz"
    )
    commits = count_rate("measurement_cir_commits", "measurement_cir_commit_rate_hz")
    transactions = count_rate(
        "measurement_cir_write_transactions",
        "measurement_cir_write_transaction_rate_hz",
    )
    writes = count_rate(
        "measurement_cir_path_writes", "measurement_cir_path_write_rate_hz"
    )
    report_by_ssu = count_vector(
        "measurement_pressure_reports", "measurement_pressure_reports_by_ssu"
    )
    cache_by_ssu = count_vector(
        "measurement_pressure_cache_hits", "measurement_pressure_cache_hits_by_ssu"
    )
    request_by_ssu = count_vector(
        "measurement_pressure_requests", "measurement_pressure_requests_by_ssu"
    )
    transaction_by_ssu = count_vector(
        "measurement_cir_write_transactions",
        "measurement_cir_write_transactions_by_ssu",
    )
    write_by_ssu = count_vector(
        "measurement_cir_path_writes", "measurement_cir_path_writes_by_ssu"
    )
    _require(
        summary.get("legacy_control_counter_scope")
        == "cumulative from simulation start through tagged-request drain",
        f"{prefix}: legacy counter scope",
    )

    def cumulative_total_vector(
        total_field: str, vector_field: str
    ) -> tuple[int, list[int]]:
        total = _integer(summary.get(total_field), f"{prefix}: {total_field}")
        vector = _vector(
            summary.get(vector_field),
            num_ssu,
            f"{prefix}: {vector_field}",
            integer=True,
        )
        _require(sum(vector) == total, f"{prefix}: cumulative {vector_field} sum")
        return total, vector

    legacy_reports, legacy_report_by_ssu = cumulative_total_vector(
        "pressure_reports", "pressure_reports_by_ssu"
    )
    legacy_cache, legacy_cache_by_ssu = cumulative_total_vector(
        "pressure_cache_hits", "pressure_cache_hits_by_ssu"
    )
    legacy_transactions, legacy_transaction_by_ssu = cumulative_total_vector(
        "cir_write_transactions", "cir_write_transactions_by_ssu"
    )
    legacy_writes, legacy_write_by_ssu = cumulative_total_vector(
        "cir_path_writes", "cir_path_writes_by_ssu"
    )
    legacy_evaluations = _integer(
        summary.get("control_evaluations"), f"{prefix}: cumulative evaluations"
    )
    legacy_commits = _integer(
        summary.get("cir_commits"), f"{prefix}: cumulative commits"
    )
    for measurement, cumulative, name in (
        (reports, legacy_reports, "reports"),
        (cache_hits, legacy_cache, "cache hits"),
        (evaluations, legacy_evaluations, "evaluations"),
        (commits, legacy_commits, "commits"),
        (transactions, legacy_transactions, "transactions"),
        (writes, legacy_writes, "path writes"),
    ):
        _require(
            measurement <= cumulative,
            f"{prefix}: measurement {name} exceeds cumulative counter",
        )
    for measurement, cumulative, name in (
        (report_by_ssu, legacy_report_by_ssu, "reports"),
        (cache_by_ssu, legacy_cache_by_ssu, "cache hits"),
        (transaction_by_ssu, legacy_transaction_by_ssu, "transactions"),
        (write_by_ssu, legacy_write_by_ssu, "path writes"),
    ):
        _require(
            all(left <= right for left, right in zip(measurement, cumulative)),
            f"{prefix}: measurement per-SSU {name} exceeds cumulative counter",
        )
    _require(
        requests == reports + cache_hits, f"{prefix}: reads + cache hits != requests"
    )
    _require(
        request_by_ssu
        == [left + right for left, right in zip(report_by_ssu, cache_by_ssu)],
        f"{prefix}: per-SSU pressure requests",
    )
    _require(
        transactions <= writes and commits <= transactions and commits <= evaluations,
        f"{prefix}: control counter ordering",
    )
    for field, counts in (
        ("measurement_pressure_report_rate_hz_by_ssu", report_by_ssu),
        ("measurement_cir_write_transaction_rate_hz_by_ssu", transaction_by_ssu),
        ("measurement_cir_path_write_rate_hz_by_ssu", write_by_ssu),
    ):
        rates = _vector(summary.get(field), num_ssu, f"{prefix}: {field}")
        _compare_vectors(
            rates, [count / duration_s for count in counts], f"{prefix}: {field}"
        )
    _close(
        summary.get("measurement_pressure_cache_hit_fraction"),
        cache_hits / requests if requests else 0.0,
        f"{prefix}: cache fraction",
    )
    hit_fractions = _vector(
        summary.get("measurement_pressure_cache_hit_fraction_by_ssu"),
        num_ssu,
        f"{prefix}: cache fractions",
        fraction=True,
    )
    for index, (value, hits, request_count) in enumerate(
        zip(hit_fractions, cache_by_ssu, request_by_ssu)
    ):
        _close(
            value,
            hits / request_count if request_count else 0.0,
            f"{prefix}: cache fraction SSU{index}",
        )
    _close(
        summary.get("measurement_cir_entries_per_transaction"),
        writes / transactions if transactions else 0.0,
        f"{prefix}: entries/transaction",
    )
    entries = _vector(
        summary.get("measurement_cir_entries_per_transaction_by_ssu"),
        num_ssu,
        f"{prefix}: entries/transaction by SSU",
    )
    for index, (value, entry_count, transaction_count) in enumerate(
        zip(entries, write_by_ssu, transaction_by_ssu)
    ):
        _close(
            value,
            entry_count / transaction_count if transaction_count else 0.0,
            f"{prefix}: entries/transaction SSU{index}",
        )
    if case["kind"] == "baseline":
        _require(
            requests == writes == transactions == commits == evaluations == 0,
            f"{prefix}: baseline used control plane",
        )
        _require(
            summary.get("control_min_interval_ms") is None,
            f"{prefix}: baseline interval",
        )
    else:
        _require(requests == 0 and evaluations > 0, f"{prefix}: Adaptive counters")
        _close(
            summary.get("control_min_interval_ms"),
            case["min_interval_ms"],
            f"{prefix}: Adaptive interval",
        )
        maximum_evaluations = (
            int(math.ceil(duration_ms / float(case["min_interval_ms"]))) + 1
        )
        _require(
            evaluations <= maximum_evaluations,
            f"{prefix}: evaluations exceed min-interval physical upper bound",
        )
    return {
        "pressure_reads": reports,
        "pressure_read_rate_hz": reports / duration_s,
        "pressure_read_rate_hz_per_ssu": reports / duration_s / num_ssu,
        "cir_entry_writes": writes,
        "cir_entry_write_rate_hz": writes / duration_s,
        "cir_entry_write_rate_hz_per_ssu": writes / duration_s / num_ssu,
        "cir_transactions": transactions,
        "cir_transaction_rate_hz_per_ssu": transactions / duration_s / num_ssu,
        "control_evaluations": evaluations,
        "control_evaluation_rate_hz": evaluations / duration_s,
        "control_evaluation_rate_hz_per_ssu": evaluations / duration_s / num_ssu,
    }


def _validate_resources(
    summary: dict, num_ssu: int, duration_ms: float, prefix: str
) -> dict:
    npu_utils = _vector(
        summary.get("npu_utilizations"), NUM_NPU, f"{prefix}: NPU utils", fraction=True
    )
    compute_ms = _vector(
        summary.get("compute_ms_by_npu"), NUM_NPU, f"{prefix}: compute ms"
    )
    _compare_vectors(
        compute_ms,
        [value * duration_ms for value in npu_utils],
        f"{prefix}: compute/util",
    )
    _close(
        summary.get("mean_npu_utilization"),
        statistics.fmean(npu_utils),
        f"{prefix}: mean NPU util",
    )
    ssd_utils = _vector(
        summary.get("measurement_ssd_utilizations"),
        num_ssu,
        f"{prefix}: SSD utils",
        fraction=True,
    )
    ssd_busy = _vector(
        summary.get("measurement_ssd_busy_ms_by_ssu"), num_ssu, f"{prefix}: SSD busy"
    )
    ssd_served = _vector(
        summary.get("measurement_ssd_served_gb_by_ssu"),
        num_ssu,
        f"{prefix}: SSD served",
    )
    ssd_rates = _vector(
        summary.get("measurement_ssd_served_gbps_by_ssu"),
        num_ssu,
        f"{prefix}: SSD rates",
    )
    _compare_vectors(
        ssd_busy,
        [value * duration_ms for value in ssd_utils],
        f"{prefix}: SSD busy/util",
    )
    _compare_vectors(
        ssd_served,
        [value * SSD_CAP_GBPS / 1000.0 for value in ssd_busy],
        f"{prefix}: SSD service/busy",
    )
    _compare_vectors(
        ssd_rates,
        [value * 1000.0 / duration_ms for value in ssd_served],
        f"{prefix}: SSD rates",
    )
    _close(
        summary.get("measurement_ssd_mean_utilization"),
        statistics.fmean(ssd_utils),
        f"{prefix}: mean SSD util",
    )
    ssd_matrix = _matrix(
        summary.get("measurement_npu_ssu_ssd_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: NPU-SSU SSD service",
    )
    ssd_rate_matrix = _matrix(
        summary.get("measurement_npu_ssu_ssd_served_gbps"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: NPU-SSU SSD rates",
    )
    for npu in range(NUM_NPU):
        _compare_vectors(
            ssd_rate_matrix[npu],
            [value * 1000.0 / duration_ms for value in ssd_matrix[npu]],
            f"{prefix}: SSD matrix rates NPU{npu}",
        )
    _compare_vectors(
        [math.fsum(row[ssu] for row in ssd_matrix) for ssu in range(num_ssu)],
        ssd_served,
        f"{prefix}: SSD attribution",
    )

    link_utils = _vector(
        summary.get("measurement_npu_link_utilizations"),
        NUM_NPU,
        f"{prefix}: link utils",
        fraction=True,
    )
    link_busy = _vector(
        summary.get("measurement_npu_link_busy_ms_by_npu"),
        NUM_NPU,
        f"{prefix}: link busy",
    )
    _compare_vectors(
        link_busy,
        [value * duration_ms for value in link_utils],
        f"{prefix}: link busy/util",
    )
    _close(
        summary.get("measurement_npu_link_mean_utilization"),
        statistics.fmean(link_utils),
        f"{prefix}: mean link util",
    )
    link_matrix = _matrix(
        summary.get("measurement_npu_ssu_link_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: link service",
    )
    link_rate_matrix = _matrix(
        summary.get("measurement_npu_ssu_link_served_gbps"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: link rates",
    )
    for npu in range(NUM_NPU):
        _compare_vectors(
            link_rate_matrix[npu],
            [value * 1000.0 / duration_ms for value in link_matrix[npu]],
            f"{prefix}: link matrix rates NPU{npu}",
        )
        _close(
            math.fsum(link_matrix[npu]),
            link_busy[npu] * NPU_CAP_GBPS / 1000.0,
            f"{prefix}: link attribution NPU{npu}",
        )

    actual_ssu = _vector(
        summary.get("max_actual_cir_sum_gbps_by_ssu"),
        num_ssu,
        f"{prefix}: max actual SSU CIR",
    )
    _require(
        max(actual_ssu) <= SSD_CAP_GBPS + 1e-9, f"{prefix}: actual SSU CIR exceeds 40"
    )
    actual_npu = summary.get("max_actual_npu_cir_sum_gbps_by_npu")
    applicable = summary.get("actual_cir_per_npu_capacity_applicable")
    _require(applicable is (actual_npu is not None), f"{prefix}: NPU CIR applicability")
    if actual_npu is not None:
        actual_npu_values = _vector(
            actual_npu, NUM_NPU, f"{prefix}: max actual NPU CIR"
        )
        _require(
            max(actual_npu_values) <= NPU_CAP_GBPS + 1e-9,
            f"{prefix}: actual NPU CIR exceeds 50",
        )
    else:
        actual_npu_values = None
    for field in (
        "actual_cir_sum_gbps_by_ssu_at_stop",
        "measurement_actual_cir_sum_gbps_by_ssu_at_start",
        "measurement_actual_cir_sum_gbps_by_ssu_at_end",
    ):
        values = _vector(summary.get(field), num_ssu, f"{prefix}: {field}")
        _require(max(values) <= SSD_CAP_GBPS + 1e-9, f"{prefix}: {field} cap")
    for field in (
        "actual_npu_cir_sum_gbps_at_stop",
        "measurement_actual_npu_cir_sum_gbps_at_start",
        "measurement_actual_npu_cir_sum_gbps_at_end",
    ):
        if applicable:
            values = _vector(summary.get(field), NUM_NPU, f"{prefix}: {field}")
            _require(max(values) <= NPU_CAP_GBPS + 1e-9, f"{prefix}: {field} cap")
        else:
            _require(summary.get(field) is None, f"{prefix}: {field} should be null")
    return {
        "mean_npu_utilization_pct": 100.0 * statistics.fmean(npu_utils),
        "mean_ssd_utilization_pct": 100.0 * statistics.fmean(ssd_utils),
        "max_ssd_utilization_pct": 100.0 * max(ssd_utils),
        "mean_npu_link_utilization_pct": 100.0 * statistics.fmean(link_utils),
        "max_actual_ssu_cir_gbps": max(actual_ssu),
        "max_actual_npu_cir_gbps": (
            max(actual_npu_values) if actual_npu_values is not None else None
        ),
    }


SNAPSHOT_FIELDS = {
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


def _parse_stationarity_snapshots(
    summary: dict, num_ssu: int, prefix: str
) -> list[dict]:
    _require(
        summary.get("stationarity_boundary_semantics")
        == EXPECTED_STATIONARITY_SEMANTICS,
        f"{prefix}: stationarity semantics",
    )
    snapshots = summary.get("measurement_stationarity_boundaries")
    _require(
        isinstance(snapshots, list) and len(snapshots) == ACTIVE_BOUNDARIES,
        f"{prefix}: expected exactly {ACTIVE_BOUNDARIES} boundaries",
    )
    _require(
        summary.get("measurement_stationarity_boundary_count") == ACTIVE_BOUNDARIES,
        f"{prefix}: reported boundary count",
    )
    start_ms = float(summary["measurement_start_ms"])
    expected_times = [
        start_ms + index * ACTIVE_BLOCK_MS for index in range(ACTIVE_BOUNDARIES)
    ]
    parsed = []
    for boundary, (snapshot, expected_time) in enumerate(
        zip(snapshots, expected_times)
    ):
        _require(
            isinstance(snapshot, dict) and set(snapshot) == SNAPSHOT_FIELDS,
            f"{prefix}: snapshot {boundary} fields",
        )
        _require(snapshot["boundary"] == boundary, f"{prefix}: snapshot boundary index")
        _close(
            snapshot["time_ms"],
            expected_time,
            f"{prefix}: snapshot time",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        item = {
            "time_ms": float(snapshot["time_ms"]),
            "ssd_busy": _vector(
                snapshot["ssd_cumulative_busy_ms_by_ssu"],
                num_ssu,
                f"{prefix}: snapshot SSD busy",
            ),
            "ssd_served": _vector(
                snapshot["ssd_cumulative_served_gb_by_ssu"],
                num_ssu,
                f"{prefix}: snapshot SSD served",
            ),
            "ssd_blocks": _vector(
                snapshot["ssd_outstanding_blocks_by_ssu"],
                num_ssu,
                f"{prefix}: snapshot SSD blocks",
                integer=True,
            ),
            "ssd_gb": _vector(
                snapshot["ssd_outstanding_gb_by_ssu"],
                num_ssu,
                f"{prefix}: snapshot SSD GB",
            ),
            "compute": _vector(
                snapshot["npu_compute_cumulative_busy_ms_by_npu"],
                NUM_NPU,
                f"{prefix}: snapshot compute",
            ),
            "link_busy": _vector(
                snapshot["npu_link_cumulative_busy_ms_by_npu"],
                NUM_NPU,
                f"{prefix}: snapshot link busy",
            ),
            "link_served": _vector(
                snapshot["npu_link_cumulative_served_gb_by_npu"],
                NUM_NPU,
                f"{prefix}: snapshot link served",
            ),
            "link_blocks": _vector(
                snapshot["npu_link_outstanding_blocks_by_npu"],
                NUM_NPU,
                f"{prefix}: snapshot link blocks",
                integer=True,
            ),
            "link_gb": _vector(
                snapshot["npu_link_outstanding_gb_by_npu"],
                NUM_NPU,
                f"{prefix}: snapshot link GB",
            ),
        }
        for index, (busy, served) in enumerate(
            zip(item["ssd_busy"], item["ssd_served"])
        ):
            _close(
                served,
                busy * SSD_CAP_GBPS / 1000.0,
                f"{prefix}: cumulative SSD service {boundary}/{index}",
            )
        for index, (busy, served) in enumerate(
            zip(item["link_busy"], item["link_served"])
        ):
            _close(
                served,
                busy * NPU_CAP_GBPS / 1000.0,
                f"{prefix}: cumulative link service {boundary}/{index}",
            )
        parsed.append(item)
    for previous, current in zip(parsed, parsed[1:]):
        for field in ("ssd_busy", "ssd_served", "compute", "link_busy", "link_served"):
            _require(
                all(
                    right + 1e-8 >= left
                    for left, right in zip(previous[field], current[field])
                ),
                f"{prefix}: cumulative {field} moved backwards",
            )
    return parsed


BLOCK_FIELDS = {
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


def _validate_stationarity(
    summary: dict,
    diagnostic: dict,
    request_info: dict,
    manifest_data: dict,
    num_ssu: int,
    prefix: str,
) -> dict:
    snapshots = _parse_stationarity_snapshots(summary, num_ssu, prefix)
    blocks = summary.get("measurement_blocks")
    _require(
        isinstance(blocks, list) and len(blocks) == ACTIVE_BLOCKS,
        f"{prefix}: expected exactly {ACTIVE_BLOCKS} blocks",
    )
    request_rows = request_info["rows"]
    block_utils = []
    block_request_counts = []
    block_fleet_served_gb = []
    block_resource_sums = {
        "ssd_busy": [0.0] * num_ssu,
        "ssd_served": [0.0] * num_ssu,
        "compute": [0.0] * NUM_NPU,
        "link_busy": [0.0] * NUM_NPU,
        "link_served": [0.0] * NUM_NPU,
    }
    for block_index, (block, left, right) in enumerate(
        zip(blocks, snapshots, snapshots[1:])
    ):
        _require(
            isinstance(block, dict) and set(block) == BLOCK_FIELDS,
            f"{prefix}: block {block_index} fields",
        )
        expected_start = (
            float(summary["measurement_start_ms"]) + block_index * ACTIVE_BLOCK_MS
        )
        expected_end = expected_start + ACTIVE_BLOCK_MS
        _require(block["block"] == block_index, f"{prefix}: block index")
        _close(
            block["start_ms"],
            expected_start,
            f"{prefix}: block start",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        _close(
            block["end_ms"],
            expected_end,
            f"{prefix}: block end",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        _close(block["duration_ms"], ACTIVE_BLOCK_MS, f"{prefix}: block duration")
        deltas = {
            "ssd_busy": [b - a for a, b in zip(left["ssd_busy"], right["ssd_busy"])],
            "ssd_served": [
                b - a for a, b in zip(left["ssd_served"], right["ssd_served"])
            ],
            "compute": [b - a for a, b in zip(left["compute"], right["compute"])],
            "link_busy": [b - a for a, b in zip(left["link_busy"], right["link_busy"])],
            "link_served": [
                b - a for a, b in zip(left["link_served"], right["link_served"])
            ],
        }
        for field, values in deltas.items():
            _require(
                all(value >= -1e-8 for value in values),
                f"{prefix}: negative block {field}",
            )
            for index, value in enumerate(values):
                block_resource_sums[field][index] += max(0.0, value)
        for block_field, delta_field in (
            ("ssd_busy_ms_by_ssu", "ssd_busy"),
            ("ssd_served_gb_by_ssu", "ssd_served"),
            ("compute_ms_by_npu", "compute"),
            ("npu_link_busy_ms_by_npu", "link_busy"),
            ("npu_link_served_gb_by_npu", "link_served"),
        ):
            values = _vector(
                block[block_field], len(deltas[delta_field]), f"{prefix}: {block_field}"
            )
            _compare_vectors(
                values,
                deltas[delta_field],
                f"{prefix}: block {block_index} {block_field}",
            )
        ssd_utils = _vector(
            block["ssd_utilizations"],
            num_ssu,
            f"{prefix}: block SSD utils",
            fraction=True,
        )
        _compare_vectors(
            ssd_utils,
            [value / ACTIVE_BLOCK_MS for value in deltas["ssd_busy"]],
            f"{prefix}: block SSD utils",
        )
        _close(
            block["ssd_mean_utilization"],
            statistics.fmean(ssd_utils),
            f"{prefix}: block SSD mean",
        )
        npu_utils = _vector(
            block["npu_utilizations"],
            NUM_NPU,
            f"{prefix}: block NPU utils",
            fraction=True,
        )
        _compare_vectors(
            npu_utils,
            [value / ACTIVE_BLOCK_MS for value in deltas["compute"]],
            f"{prefix}: block NPU utils",
        )
        block_util = statistics.fmean(npu_utils)
        _close(block["npu_utilization"], block_util, f"{prefix}: block NPU util")
        link_utils = _vector(
            block["npu_link_utilizations"],
            NUM_NPU,
            f"{prefix}: block link utils",
            fraction=True,
        )
        _compare_vectors(
            link_utils,
            [value / ACTIVE_BLOCK_MS for value in deltas["link_busy"]],
            f"{prefix}: block link utils",
        )
        _close(
            block["npu_link_mean_utilization"],
            statistics.fmean(link_utils),
            f"{prefix}: block link mean",
        )
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
            values = _vector(
                block[field],
                len(expected),
                f"{prefix}: {field}",
                integer=integer_values,
            )
            if integer_values:
                _require(values == expected, f"{prefix}: {field} edge")
            else:
                _compare_vectors(values, expected, f"{prefix}: {field} edge")
        for base, left_values, right_values, integer_values in (
            ("ssd_outstanding_blocks", left["ssd_blocks"], right["ssd_blocks"], True),
            ("ssd_outstanding_gb", left["ssd_gb"], right["ssd_gb"], False),
            (
                "npu_link_outstanding_blocks",
                left["link_blocks"],
                right["link_blocks"],
                True,
            ),
            ("npu_link_outstanding_gb", left["link_gb"], right["link_gb"], False),
        ):
            expected_delta = [b - a for a, b in zip(left_values, right_values)]
            raw_delta = block[f"{base}_delta"]
            _require(
                isinstance(raw_delta, list) and len(raw_delta) == len(expected_delta),
                f"{prefix}: {base} delta shape",
            )
            if integer_values:
                _require(raw_delta == expected_delta, f"{prefix}: {base} delta")
            else:
                _compare_vectors(raw_delta, expected_delta, f"{prefix}: {base} delta")
        admitted = [
            row
            for row in request_rows
            if expected_start <= float(row["admission_time_ms"]) < expected_end
        ]
        _require(
            block["request_count"] == len(admitted), f"{prefix}: block request count"
        )
        if admitted:
            _close(
                block["request_weighted_slo_attainment"],
                statistics.fmean(bool(row["slo_met"]) for row in admitted),
                f"{prefix}: block SLO",
            )
        else:
            _require(
                block["request_weighted_slo_attainment"] is None,
                f"{prefix}: empty block SLO",
            )
        block_utils.append(block_util)
        block_request_counts.append(len(admitted))
        block_fleet_served_gb.append(math.fsum(deltas["ssd_served"]))
    _require(
        sum(block_request_counts) == len(request_rows),
        f"{prefix}: blocks do not cover cohort",
    )

    for summary_field, resource_field in (
        ("measurement_ssd_busy_ms_by_ssu", "ssd_busy"),
        ("measurement_ssd_served_gb_by_ssu", "ssd_served"),
        ("compute_ms_by_npu", "compute"),
        ("measurement_npu_link_busy_ms_by_npu", "link_busy"),
    ):
        values = _vector(
            summary[summary_field],
            len(block_resource_sums[resource_field]),
            f"{prefix}: {summary_field}",
        )
        _compare_vectors(
            values,
            block_resource_sums[resource_field],
            f"{prefix}: {summary_field} block sum",
        )
    first, last = snapshots[0], snapshots[-1]
    for field, expected, integer_values in (
        ("measurement_ssd_outstanding_blocks_at_start", first["ssd_blocks"], True),
        ("measurement_ssd_outstanding_blocks_at_end", last["ssd_blocks"], True),
        ("measurement_ssd_outstanding_gb_at_start", first["ssd_gb"], False),
        ("measurement_ssd_outstanding_gb_at_end", last["ssd_gb"], False),
        (
            "measurement_npu_link_outstanding_blocks_at_start",
            first["link_blocks"],
            True,
        ),
        ("measurement_npu_link_outstanding_blocks_at_end", last["link_blocks"], True),
        ("measurement_npu_link_outstanding_gb_at_start", first["link_gb"], False),
        ("measurement_npu_link_outstanding_gb_at_end", last["link_gb"], False),
    ):
        values = _vector(
            summary[field], len(expected), f"{prefix}: {field}", integer=integer_values
        )
        if integer_values:
            _require(values == expected, f"{prefix}: {field}")
        else:
            _compare_vectors(values, expected, f"{prefix}: {field}")
    for base, start_values, end_values, integer_values in (
        (
            "measurement_ssd_outstanding_blocks",
            first["ssd_blocks"],
            last["ssd_blocks"],
            True,
        ),
        ("measurement_ssd_outstanding_gb", first["ssd_gb"], last["ssd_gb"], False),
        (
            "measurement_npu_link_outstanding_blocks",
            first["link_blocks"],
            last["link_blocks"],
            True,
        ),
        (
            "measurement_npu_link_outstanding_gb",
            first["link_gb"],
            last["link_gb"],
            False,
        ),
    ):
        expected = [end - start for start, end in zip(start_values, end_values)]
        raw = summary[f"{base}_drift"]
        if integer_values:
            _require(raw == expected, f"{prefix}: {base} drift")
        else:
            _compare_vectors(raw, expected, f"{prefix}: {base} drift")

    expected_diagnostic_fields = {
        "block_npu_utilizations",
        "block_request_counts",
        "block_utilization_range",
        "first_last_utilization_delta",
        "outstanding_blocks_drift_by_ssu",
        "fleet_outstanding_blocks_drift",
    }
    _require(
        isinstance(diagnostic, dict) and set(diagnostic) == expected_diagnostic_fields,
        f"{prefix}: stationarity_diagnostics fields",
    )
    _compare_vectors(
        diagnostic["block_npu_utilizations"],
        block_utils,
        f"{prefix}: diagnostic block utils",
    )
    _require(
        diagnostic["block_request_counts"] == block_request_counts,
        f"{prefix}: diagnostic block counts",
    )
    _close(
        diagnostic["block_utilization_range"],
        max(block_utils) - min(block_utils),
        f"{prefix}: diagnostic util range",
    )
    _close(
        diagnostic["first_last_utilization_delta"],
        block_utils[-1] - block_utils[0],
        f"{prefix}: diagnostic first/last",
    )
    block_drift = [
        end - start for start, end in zip(first["ssd_blocks"], last["ssd_blocks"])
    ]
    _require(
        diagnostic["outstanding_blocks_drift_by_ssu"] == block_drift,
        f"{prefix}: diagnostic queue drift",
    )
    _require(
        diagnostic["fleet_outstanding_blocks_drift"] == sum(block_drift),
        f"{prefix}: diagnostic fleet drift",
    )

    half = ACTIVE_BLOCKS // 2
    first_util = statistics.fmean(block_utils[:half])
    second_util = statistics.fmean(block_utils[half:])
    util_half_delta_pp = 100.0 * (second_util - first_util)
    midpoints = [
        float(summary["measurement_start_ms"]) + (index + 0.5) * ACTIVE_BLOCK_MS
        for index in range(ACTIVE_BLOCKS)
    ]
    util_slope_per_s = _theil_sen_slope(midpoints, block_utils)
    util_projected_change_pp = (
        100.0 * util_slope_per_s * (ACTIVE_MEASUREMENT_MS / 1000.0)
    )
    first_served_rate = math.fsum(block_fleet_served_gb[:half]) / (
        half * ACTIVE_BLOCK_MS / 1000.0
    )
    second_served_rate = math.fsum(block_fleet_served_gb[half:]) / (
        half * ACTIVE_BLOCK_MS / 1000.0
    )
    served_midpoint = (first_served_rate + second_served_rate) / 2.0
    served_half_relative_delta = (
        abs(second_served_rate - first_served_rate) / served_midpoint
        if served_midpoint > 0.0
        else 0.0
    )

    times = [snapshot["time_ms"] for snapshot in snapshots]
    fleet_queue = [math.fsum(snapshot["ssd_gb"]) for snapshot in snapshots]
    per_ssu_queue = [
        [snapshot["ssd_gb"][ssu] for snapshot in snapshots] for ssu in range(num_ssu)
    ]
    burst_bound = manifest_data["fleet_layer_burst_bound_gb"]

    def queue_diagnostics(values: list[float]) -> dict:
        deltas = [right - left for left, right in zip(values, values[1:])]
        slope = _theil_sen_slope(times, values)
        nondecreasing_fraction = sum(delta >= -1e-8 for delta in deltas) / len(deltas)
        net_growth = values[-1] - values[0]
        persistent_growth = (
            slope > 0.0
            and nondecreasing_fraction >= 0.75
            and net_growth > burst_bound + 1e-8
        )
        return {
            "start_gb": values[0],
            "end_gb": values[-1],
            "net_growth_gb": net_growth,
            "first_half_median_gb": statistics.median(values[: half + 1]),
            "second_half_median_gb": statistics.median(values[half:]),
            "theil_sen_slope_gbps": slope,
            "nondecreasing_step_fraction": nondecreasing_fraction,
            "max_positive_boundary_burst_gb": max([0.0, *deltas]),
            "persistent_growth_over_one_layer_burst": persistent_growth,
        }

    fleet_queue_diag = queue_diagnostics(fleet_queue)
    per_ssu_queue_diag = [queue_diagnostics(values) for values in per_ssu_queue]
    stationarity_rules = {
        "utilization_half_delta_le_1pp": abs(util_half_delta_pp)
        <= UTIL_HALF_LIMIT_PP + 1e-12,
        "utilization_projected_trend_le_2pp": abs(util_projected_change_pp)
        <= UTIL_TREND_LIMIT_PP + 1e-12,
        "fleet_served_half_relative_delta_le_2pct": served_half_relative_delta
        <= SERVED_HALF_RELATIVE_LIMIT + 1e-12,
        "no_persistent_queue_growth_over_one_layer_burst": not (
            fleet_queue_diag["persistent_growth_over_one_layer_burst"]
            or any(
                item["persistent_growth_over_one_layer_burst"]
                for item in per_ssu_queue_diag
            )
        ),
    }
    return {
        "block_count": ACTIVE_BLOCKS,
        "boundary_count": ACTIVE_BOUNDARIES,
        "block_npu_utilizations": block_utils,
        "utilization_first_half": first_util,
        "utilization_second_half": second_util,
        "utilization_half_delta_pp": util_half_delta_pp,
        "utilization_theil_sen_slope_fraction_per_s": util_slope_per_s,
        "utilization_projected_change_pp": util_projected_change_pp,
        "fleet_served_first_half_gbps": first_served_rate,
        "fleet_served_second_half_gbps": second_served_rate,
        "fleet_served_half_relative_delta": served_half_relative_delta,
        "queue_burst_bound_derivation": {
            "method": "32 NPUs multiplied by the maximum authenticated catalog per-layer KV GB; this conservatively allows every NPU's one layer to land on one SSU",
            "num_npu": NUM_NPU,
            "max_authenticated_profile_per_layer_gb": manifest_data[
                "max_single_request_layer_gb"
            ],
            "fleet_one_layer_burst_bound_gb": burst_bound,
            "persistent_definition": "Theil-Sen slope > 0, at least 75% nondecreasing boundary steps, and net growth exceeds the independent one-layer burst bound",
        },
        "fleet_queue": fleet_queue_diag,
        "per_ssu_queue": per_ssu_queue_diag,
        "stationarity_rules": stationarity_rules,
        "stationarity_gate_passed": all(stationarity_rules.values()),
    }


def _expected_assignment_statistics(
    assignments: list[list], catalog: dict[tuple[int, int], tuple[float, ...]]
) -> dict:
    profile_counts = Counter((row[4][0], row[4][1]) for row in assignments)
    category_counts = Counter(str(row[3]) for row in assignments)
    by_npu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in assignments:
        by_npu[int(row[1])].append((row[4][0], row[4][1]))
    per_npu_demands = []
    per_npu_ms_per_gb = []
    for npu in range(NUM_NPU):
        keys = by_npu[npu]
        compute_s = math.fsum(catalog[key][1] for key in keys) / 1e6
        kv_gb = math.fsum(catalog[key][3] for key in keys)
        per_npu_demands.append(kv_gb / compute_s)
        per_npu_ms_per_gb.append(1000.0 * compute_s / kv_gb)
    return {
        "profile_counts": profile_counts,
        "category_counts": category_counts,
        "per_npu_demands": per_npu_demands,
        "per_npu_ms_per_gb": per_npu_ms_per_gb,
        "fleet_demand": math.fsum(per_npu_demands),
    }


def _validate_workload_statistics(
    stats: dict,
    *,
    assignments: list[list],
    catalog: dict[tuple[int, int], tuple[float, ...]],
    expected_requests_per_npu: int,
    expected_assignment: str,
    expected_recipe: str,
    expected_schedule: str,
    materialized: dict,
    spec: dict,
    num_ssu: int,
    prefix: str,
) -> dict:
    _require(isinstance(stats, dict), f"{prefix}: workload statistics missing")
    workload = spec["workload"]
    _require(stats.get("catalog") == workload["catalog"], f"{prefix}: stats catalog")
    _require(
        stats.get("recipe") == expected_recipe
        and stats.get("schedule") == expected_schedule,
        f"{prefix}: independently rebuilt recipe/schedule",
    )
    _require(
        stats.get("assignment") == expected_assignment, f"{prefix}: stats assignment"
    )
    _require(
        stats.get("full_assignment_hash") == expected_assignment,
        f"{prefix}: stats full assignment",
    )
    _require(
        stats.get("prefix_32_assignment_hash") == workload["prefix_32_assignment_hash"],
        f"{prefix}: stats prefix assignment",
    )
    _require(
        stats.get("prefix_hash_requests_per_npu") == 32, f"{prefix}: prefix metadata"
    )
    _require(
        stats.get("profile_sampling") == "iid_uniform_profile_catalog_v1",
        f"{prefix}: sampling mode",
    )
    _require(
        stats.get("profile_sampling_uniform_over_catalog") is True
        and stats.get("profile_sampling_with_replacement") is True
        and stats.get("profile_sequence_prefix_stable") is True,
        f"{prefix}: sampling flags",
    )
    _require(
        stats.get("request_id_formula") == "sequence * num_npu + npu_id",
        f"{prefix}: request ID formula",
    )
    _require(
        stats.get("placement_mode") == "block_ring_hash"
        and stats.get("placement_reuse_layers") == N_LAYERS,
        f"{prefix}: placement metadata",
    )
    _require(
        stats.get("requests_per_npu") == expected_requests_per_npu,
        f"{prefix}: requests/NPU",
    )
    _require(
        stats.get("request_count") == NUM_NPU * expected_requests_per_npu,
        f"{prefix}: request count",
    )
    _require(stats.get("seed") == workload["seed"], f"{prefix}: seed")
    _require(stats.get("catalog_profile_count") == 84, f"{prefix}: catalog count")
    starts = _vector(
        stats.get("initial_npu_start_ms"), NUM_NPU, f"{prefix}: initial starts"
    )
    _require(max(starts) <= 5.0 + 1e-12, f"{prefix}: initial jitter exceeds 5 ms")
    _compare_vectors(
        starts,
        materialized["initial_npu_start_ms"],
        f"{prefix}: independently rebuilt initial jitter",
    )
    expected = _expected_assignment_statistics(assignments, catalog)
    profile_counts = {
        f"{key[0]},{key[1]}": count
        for key, count in sorted(expected["profile_counts"].items())
    }
    profile_counts_all = {
        f"{key[0]},{key[1]}": expected["profile_counts"].get(key, 0)
        for key in sorted(catalog)
    }
    _require(
        stats.get("fleet_profile_counts") == profile_counts, f"{prefix}: profile counts"
    )
    _require(
        stats.get("fleet_profile_counts_all") == profile_counts_all,
        f"{prefix}: all profile counts",
    )
    _require(
        stats.get("profiles_used") == len(profile_counts), f"{prefix}: profiles used"
    )
    category_counts = {
        category: expected["category_counts"].get(category, 0)
        for category in CATEGORIES
    }
    _require(
        stats.get("fleet_category_counts_all") == category_counts,
        f"{prefix}: category counts",
    )
    _require(
        sum(stats.get("fleet_category_counts", {}).values()) == len(assignments),
        f"{prefix}: category count sum",
    )
    demands = expected["per_npu_demands"]
    ms_values = expected["per_npu_ms_per_gb"]
    demand_meta = stats.get("per_npu_raw_demand_gbps")
    ms_meta = stats.get("per_npu_ms_per_gb")
    _require(
        isinstance(demand_meta, dict) and isinstance(ms_meta, dict),
        f"{prefix}: per-NPU demand metadata",
    )
    for field, value in (
        ("min", min(demands)),
        ("max", max(demands)),
        ("mean", statistics.fmean(demands)),
    ):
        _close(demand_meta.get(field), value, f"{prefix}: demand {field}")
    demand_mean = statistics.fmean(demands)
    _close(
        demand_meta.get("coefficient_of_variation"),
        statistics.pstdev(demands) / demand_mean,
        f"{prefix}: demand CV",
    )
    for field, value in (
        ("min", min(ms_values)),
        ("max", max(ms_values)),
        ("mean", statistics.fmean(ms_values)),
        ("spread", max(ms_values) - min(ms_values)),
    ):
        _close(ms_meta.get(field), value, f"{prefix}: ms/GB {field}")
    fleet = expected["fleet_demand"]
    _close(stats.get("fleet_demand_gbps"), fleet, f"{prefix}: fleet demand")
    _close(
        stats.get("capacity_knee_ssu"), fleet / SSD_CAP_GBPS, f"{prefix}: capacity knee"
    )
    placed = _vector(
        stats.get("demand_gbps_by_ssu"), num_ssu, f"{prefix}: demand by SSU"
    )
    _compare_vectors(
        placed,
        materialized["demand_gbps_by_ssu"],
        f"{prefix}: independently rebuilt placement demand",
    )
    _close(math.fsum(placed), fleet, f"{prefix}: placed demand sum")
    _close(stats.get("max_ssu_demand_gbps"), max(placed), f"{prefix}: max SSU demand")
    _require(
        stats.get("ssu_over_40_count") == sum(value > SSD_CAP_GBPS for value in placed),
        f"{prefix}: SSUs over 40",
    )
    return {
        "fleet_demand_gbps": fleet,
        "capacity_knee_ssu": fleet / SSD_CAP_GBPS,
        "global_load_fraction": fleet / (SSD_CAP_GBPS * num_ssu),
        "max_ssu_demand_gbps": max(placed),
        "ssu_over_40_count": sum(value > SSD_CAP_GBPS for value in placed),
    }


INPUT_FINGERPRINT_FIELDS = {
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


def _validate_row(
    row: dict,
    *,
    seed: int,
    source: str,
    config: str,
    spec: dict,
    manifest_data: dict,
    root_runtime: dict,
    shard_path: str,
) -> dict:
    _require(isinstance(row, dict), f"{shard_path}: result row malformed")
    case_name = str(row.get("case"))
    _require(
        case_name in EXPECTED_CASE_BY_NAME, f"{shard_path}: unknown case {case_name}"
    )
    case = EXPECTED_CASE_BY_NAME[case_name]
    role = CASE_TO_ROLE[case_name]
    num_ssu = _integer(row.get("num_ssu"), f"{shard_path}: row SSU", minimum=1)
    _require(num_ssu in FORMAL_SSUS, f"{shard_path}: SSU{num_ssu} outside frozen grid")
    prefix = f"{shard_path}: seed{seed}/SSU{num_ssu}/{role}"
    _require(row.get("status") == "ok", f"{prefix}: status")
    _require(
        row.get("family") == case["family"] and row.get("kind") == case["kind"],
        f"{prefix}: family/kind",
    )
    _require(row.get("role") == role, f"{prefix}: role")
    _require(row.get("num_npu") == NUM_NPU, f"{prefix}: NPU count")
    _require(
        row.get("definition") == DEFINITION
        and row.get("definition_fingerprint") == EXPECTED_DEFINITION_FINGERPRINT,
        f"{prefix}: definition",
    )
    _require(row.get("case_spec") == case, f"{prefix}: case spec")
    _require(
        row.get("source_fingerprint") == source
        and row.get("config_fingerprint") == config,
        f"{prefix}: provenance",
    )
    _require(
        row.get("case_fingerprint") == _case_fingerprint(case, num_ssu, source, config),
        f"{prefix}: case fingerprint",
    )
    inputs = row.get("input_fingerprints")
    _require(
        isinstance(inputs, dict) and set(inputs) == INPUT_FINGERPRINT_FIELDS,
        f"{prefix}: input fingerprint fields",
    )
    _require(
        all(_is_sha256(value) for value in inputs.values()),
        f"{prefix}: invalid input fingerprint",
    )
    workload = spec["workload"]
    expected_inputs = {
        "catalog": workload["catalog"],
        "recipe": workload["recipe"],
        "schedule": workload["schedule"],
        "assignment": workload["assignment"],
        "prefix_32_assignment": workload["prefix_32_assignment_hash"],
        "full_assignment": workload["full_assignment_hash"],
    }
    _require(
        all(inputs[field] == value for field, value in expected_inputs.items()),
        f"{prefix}: schedule fingerprints",
    )
    full_materialized = _materialize_inputs(
        assignments=manifest_data["assignments"],
        assignment_hash=workload["assignment"],
        catalog=manifest_data["catalog"],
        seed=seed,
        num_ssu=num_ssu,
        requests_per_npu=ACTIVE_BACKING_REQUESTS,
    )
    for field in ("workload", "placement", "trace", "simulator"):
        _require(
            inputs[field] == full_materialized[field],
            f"{prefix}: independently rebuilt {field} fingerprint",
        )
    prefix_assignments = [
        assignment for assignment in manifest_data["assignments"] if assignment[2] < 32
    ]
    prefix_assignment_hash = _assignment_hash(prefix_assignments)
    _require(
        prefix_assignment_hash == workload["prefix_32_assignment_hash"],
        f"{prefix}: independently rebuilt prefix assignment",
    )
    prefix_recipe_hash = _canonical_hash(
        _recipe(seed, workload["catalog"], 32),
        b"random-steady-state:recipe:v1\0",
    )
    prefix_schedule_hash = _canonical_hash(
        {
            "recipe_hash": prefix_recipe_hash,
            "assignments": prefix_assignments,
        },
        b"random-steady-state:schedule:v1\0",
    )
    prefix_materialized_expected = _materialize_inputs(
        assignments=prefix_assignments,
        assignment_hash=prefix_assignment_hash,
        catalog=manifest_data["catalog"],
        seed=seed,
        num_ssu=num_ssu,
        requests_per_npu=32,
    )
    row_runtime = _runtime_scientific(row.get("runtime"), prefix)
    _require(
        row_runtime == root_runtime, f"{prefix}: worker/root scientific runtime differs"
    )
    _require(
        _finite(row.get("wall_time_s"), f"{prefix}: wall time") > 0.0,
        f"{prefix}: wall time must be positive",
    )

    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{prefix}: steady summary missing")
    _require(
        summary.get("schema_version") == SUMMARY_SCHEMA_VERSION,
        f"{prefix}: summary schema",
    )
    _require(summary.get("mode") == "steady_state_full_load", f"{prefix}: mode")
    _require(
        summary.get("num_npu") == NUM_NPU and summary.get("num_ssu") == num_ssu,
        f"{prefix}: topology",
    )
    _require(
        summary.get("n_layers") == N_LAYERS and summary.get("batch_size") == BATCH_SIZE,
        f"{prefix}: model",
    )
    _require(
        summary.get("input_fingerprint") == inputs["simulator"],
        f"{prefix}: simulator fingerprint",
    )
    _require(
        summary.get("warmup_requests_per_npu") == WARMUP_REQUESTS, f"{prefix}: warmup"
    )
    _close(summary.get("settle_ms"), SETTLE_MS, f"{prefix}: settle")
    _close(
        summary.get("measurement_duration_ms"),
        ACTIVE_MEASUREMENT_MS,
        f"{prefix}: measurement duration",
    )
    _close(summary.get("slo_alpha"), PRIMARY_ALPHA, f"{prefix}: alpha")
    _close(
        summary.get("pressure_ttl_ms"),
        case["pressure_ttl_ms"],
        f"{prefix}: pressure TTL",
    )
    _close(
        summary.get("cir_write_threshold_gbps"),
        case["cir_write_threshold_gbps"],
        f"{prefix}: CIR threshold",
    )
    warm = _nonnegative(summary.get("warmup_reached_ms"), f"{prefix}: warm reached")
    start = _nonnegative(
        summary.get("measurement_start_ms"), f"{prefix}: measurement start"
    )
    end = _nonnegative(summary.get("measurement_end_ms"), f"{prefix}: measurement end")
    drain = _nonnegative(summary.get("drain_stop_ms"), f"{prefix}: drain stop")
    _require(warm <= start <= end <= drain, f"{prefix}: phase ordering")
    _close(start - warm, SETTLE_MS, f"{prefix}: warm-to-measurement settle")
    _close(end - start, ACTIVE_MEASUREMENT_MS, f"{prefix}: window width")
    _close(summary.get("tail_drain_ms"), drain - end, f"{prefix}: tail drain")
    _require(
        summary.get("measurement_control_counter_window") == EXPECTED_CONTROL_WINDOW,
        f"{prefix}: counter window",
    )
    invariants = summary.get("invariants")
    _require(isinstance(invariants, dict), f"{prefix}: invariants missing")
    _require(
        set(invariants) == EXPECTED_INVARIANTS and len(invariants) == 29,
        f"{prefix}: invariant catalog is not the frozen 29-item set",
    )
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{prefix}: failed invariants {failed}")

    request_info = _validate_requests(summary, manifest_data, prefix)
    _require(
        row.get("measurement_cohort_fingerprint") == request_info["cohort_hash"],
        f"{prefix}: cohort fingerprint",
    )
    _require(
        _equivalent_derived(
            row.get("cohort_profile_metrics"), request_info["cohort_alpha2"]
        ),
        f"{prefix}: cohort profile metrics",
    )
    controls = _validate_control_metrics(
        summary, case, num_ssu, ACTIVE_MEASUREMENT_MS, prefix
    )
    resources = _validate_resources(summary, num_ssu, ACTIVE_MEASUREMENT_MS, prefix)
    stationarity = _validate_stationarity(
        summary,
        row.get("stationarity_diagnostics"),
        request_info,
        manifest_data,
        num_ssu,
        prefix,
    )
    full_stats = _validate_workload_statistics(
        row.get("workload_statistics"),
        assignments=manifest_data["assignments"],
        catalog=manifest_data["catalog"],
        expected_requests_per_npu=ACTIVE_BACKING_REQUESTS,
        expected_assignment=workload["assignment"],
        expected_recipe=workload["recipe"],
        expected_schedule=workload["schedule"],
        materialized=full_materialized,
        spec=spec,
        num_ssu=num_ssu,
        prefix=f"{prefix}: backing",
    )
    prefix_stats = _validate_workload_statistics(
        row.get("prefix_32_workload_statistics"),
        assignments=prefix_assignments,
        catalog=manifest_data["catalog"],
        expected_requests_per_npu=32,
        expected_assignment=workload["prefix_32_assignment_hash"],
        expected_recipe=prefix_recipe_hash,
        expected_schedule=prefix_schedule_hash,
        materialized=prefix_materialized_expected,
        spec=spec,
        num_ssu=num_ssu,
        prefix=f"{prefix}: scientific prefix",
    )
    prefix_materialized = row.get("prefix_32_materialized_fingerprints")
    _require(
        isinstance(prefix_materialized, dict)
        and set(prefix_materialized) == {"workload", "placement", "trace"},
        f"{prefix}: prefix materialized fields",
    )
    _require(
        all(_is_sha256(value) for value in prefix_materialized.values()),
        f"{prefix}: prefix materialized fingerprints",
    )
    for field in ("workload", "placement", "trace"):
        _require(
            prefix_materialized[field] == prefix_materialized_expected[field],
            f"{prefix}: independently rebuilt prefix {field} fingerprint",
        )
    completed = _vector(
        summary.get("completed_by_npu_at_stop"),
        NUM_NPU,
        f"{prefix}: completed counts",
        integer=True,
    )
    _require(
        max(completed) <= ACTIVE_BACKING_REQUESTS,
        f"{prefix}: completed beyond backing",
    )
    _require(
        all(value >= WARMUP_REQUESTS for value in completed),
        f"{prefix}: completed below warmup",
    )
    for npu_id, (reported, independently_required) in enumerate(
        zip(
            completed,
            request_info["minimum_completed_by_npu_from_tagged_sequences"],
        )
    ):
        _require(
            reported >= independently_required,
            f"{prefix}: completed count NPU{npu_id} contradicts tagged sequence",
        )
    _require(
        summary.get("all_input_requests_completed") is False,
        f"{prefix}: finite backing was exhausted/all input completed",
    )
    backing_margin = ACTIVE_BACKING_REQUESTS - max(completed)
    _require(
        backing_margin >= MIN_BACKING_MARGIN,
        f"{prefix}: backing margin {backing_margin} < {MIN_BACKING_MARGIN}",
    )
    row_gates = {
        "sample_target_min_8": request_info["sample_target_met"],
        "backing_margin_min_32": backing_margin >= MIN_BACKING_MARGIN,
        "stationarity": stationarity["stationarity_gate_passed"],
        "all_29_invariants": True,
        "all_32_npus_sampled": True,
    }
    return {
        "seed": seed,
        "num_ssu": num_ssu,
        "case": case_name,
        "role": role,
        "pressure_ttl_ms": float(case["pressure_ttl_ms"]),
        "cir_write_threshold_gbps": float(case["cir_write_threshold_gbps"]),
        "min_interval_ms": float(case["min_interval_ms"]),
        "equal_npu_slo_alpha2_pct": 100.0 * request_info["alpha2"]["equal_npu"],
        "request_weighted_slo_alpha2_pct": 100.0
        * request_info["alpha2"]["request_weighted"],
        "equal_npu_slo_alpha15_pct": 100.0 * request_info["alpha15"]["equal_npu"],
        "request_weighted_slo_alpha15_pct": 100.0
        * request_info["alpha15"]["request_weighted"],
        "mean_ttft_ms": request_info["mean_ttft_ms"],
        "p99_ttft_ms": request_info["p99_ttft_ms"],
        "measurement_request_count": len(request_info["rows"]),
        "requests_per_npu_min": min(request_info["counts"]),
        "requests_per_npu_median": statistics.median(request_info["counts"]),
        "requests_per_npu_max": max(request_info["counts"]),
        "max_equal_npu_single_outcome_weight_pp": request_info[
            "max_equal_npu_single_outcome_weight_pp"
        ],
        "measurement_cohort_fingerprint": request_info["cohort_hash"],
        "backing_margin_fastest_npu": backing_margin,
        "completed_by_npu_min": min(completed),
        "completed_by_npu_max": max(completed),
        "wall_time_s": float(row["wall_time_s"]),
        "input_fingerprints": inputs,
        "prefix_materialized_fingerprints": prefix_materialized,
        "workload_statistics": row["workload_statistics"],
        "prefix_workload_statistics": row["prefix_32_workload_statistics"],
        "row_gates": row_gates,
        "row_gate_passed": all(row_gates.values()),
        **controls,
        **resources,
        **stationarity,
        **{f"prefix_{name}": value for name, value in prefix_stats.items()},
        **{f"backing_{name}": value for name, value in full_stats.items()},
    }


def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValidationError(f"non-finite JSON constant: {value}")


def _read_shard(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise ValidationError(f"cannot read {resolved}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON in {resolved}: {error}") from error
    _require(isinstance(payload, dict), f"{resolved}: root must be an object")
    payload["_analysis_path"] = str(resolved)
    payload["_analysis_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _validate_shard(payload: dict) -> dict:
    path = payload["_analysis_path"]
    _require(
        payload.get("schema_version") == SHARD_SCHEMA_VERSION, f"{path}: shard schema"
    )
    _require(
        payload.get("selected_complete") is True,
        f"{path}: selected_complete is not true",
    )
    _require(
        payload.get("source_stable_during_run") is True,
        f"{path}: source changed during run",
    )
    _require(
        payload.get("config_stable_during_run") is True,
        f"{path}: config changed during run",
    )
    source = payload.get("source_fingerprint")
    config = payload.get("config_fingerprint")
    _require(
        _is_sha256(source) and _is_sha256(config),
        f"{path}: invalid source/config fingerprint",
    )
    _require(
        payload.get("ending_source_fingerprint") == source,
        f"{path}: source endpoints differ",
    )
    _require(
        payload.get("ending_config_fingerprint") == config,
        f"{path}: config endpoints differ",
    )
    source_manifest = payload.get("source_manifest")
    _require(
        isinstance(source_manifest, dict) and source_manifest,
        f"{path}: source manifest missing",
    )
    _require(
        all(
            isinstance(name, str) and _is_sha256(value)
            for name, value in source_manifest.items()
        ),
        f"{path}: source manifest malformed",
    )
    _require(
        _canonical_hash(source_manifest, b"ms-scale-control-source:v1\0") == source,
        f"{path}: source fingerprint does not authenticate manifest",
    )

    spec = payload.get("experiment_spec")
    seed, authentication = _validate_spec(spec, path)
    _require(
        set(source_manifest) == set(spec["source_files"]),
        f"{path}: source manifest/spec closure differs",
    )
    _require(
        source_manifest.get("data") == authentication["source_sha256"],
        f"{path}: data file authentication",
    )
    _require(
        _canonical_hash(spec, b"ms-scale-control-config:v1\0") == config,
        f"{path}: config fingerprint mismatch",
    )
    _require(payload.get("definition") == DEFINITION, f"{path}: root definition")
    _require(
        payload.get("definition_fingerprint") == EXPECTED_DEFINITION_FINGERPRINT,
        f"{path}: root definition fingerprint",
    )
    _require(payload.get("num_npu") == NUM_NPU, f"{path}: root NPU count")
    _require(
        payload.get("input_authentication") == authentication,
        f"{path}: root input authentication",
    )
    loader = payload.get("input_loader_environment")
    _require(
        isinstance(loader, dict)
        and set(loader) == {"cache_path", "cache_present", "cache_verified_equal"},
        f"{path}: loader environment",
    )
    _require(
        isinstance(loader["cache_present"], bool)
        and isinstance(loader["cache_verified_equal"], bool),
        f"{path}: loader booleans",
    )
    _require(
        loader["cache_present"] == loader["cache_verified_equal"],
        f"{path}: cache present but not verified equal",
    )
    _require(
        loader["cache_path"] == "results/bw_table_cache_v2_32npu.npz",
        f"{path}: unexpected cache path",
    )
    _validate_path_abi(payload.get("path_abi"), path)
    root_runtime = _runtime_scientific(payload.get("runtime"), path)
    execution = payload.get("execution")
    _require(
        isinstance(execution, dict)
        and set(execution)
        == {
            "multiprocessing_start_method",
            "requested_max_workers",
            "single_ssu_process_pool_required",
        },
        f"{path}: execution metadata",
    )
    _require(
        execution["multiprocessing_start_method"]
        == root_runtime["multiprocessing_start_method"],
        f"{path}: execution/runtime start method",
    )
    _require(
        execution["single_ssu_process_pool_required"] is False,
        f"{path}: wrong confirm32 pool contract",
    )
    _integer(
        execution["requested_max_workers"], f"{path}: requested workers", minimum=1
    )
    manifest_data = _validate_seed_manifest(
        payload.get("schedule_metadata"), spec, path
    )

    selected = payload.get("selected_keys")
    _require(isinstance(selected, list) and selected, f"{path}: selected_keys missing")
    selected_keys = []
    for index, key in enumerate(selected):
        _require(
            isinstance(key, list) and len(key) == 2, f"{path}: selected key {index}"
        )
        parsed = (str(key[0]), _integer(key[1], f"{path}: selected SSU", minimum=1))
        _require(
            parsed[0] in EXPECTED_CASE_BY_NAME and parsed[1] in FORMAL_SSUS,
            f"{path}: selected key outside plan {parsed}",
        )
        selected_keys.append(parsed)
    _require(
        len(selected_keys) == len(set(selected_keys)), f"{path}: duplicate selected key"
    )
    raw_results = payload.get("results")
    _require(isinstance(raw_results, list) and raw_results, f"{path}: results missing")
    raw_keys = []
    compact_rows = {}
    raw_by_key = {}
    for row in raw_results:
        compact = _validate_row(
            row,
            seed=seed,
            source=source,
            config=config,
            spec=spec,
            manifest_data=manifest_data,
            root_runtime=root_runtime,
            shard_path=path,
        )
        key = (compact["case"], compact["num_ssu"])
        _require(key not in compact_rows, f"{path}: duplicate result {key}")
        raw_keys.append(key)
        compact_rows[key] = compact
        raw_by_key[key] = row
    _require(
        set(selected_keys) <= set(raw_keys),
        f"{path}: selected_complete contradicts results",
    )
    expected_seed_keys = {
        (case["name"], num_ssu) for case in EXPECTED_CASES for num_ssu in FORMAL_SSUS
    }
    recomputed_complete = set(raw_keys) == expected_seed_keys
    _require(
        payload.get("complete") is recomputed_complete,
        f"{path}: root complete flag is inconsistent",
    )

    pairing_root = payload.get("pairing_audit")
    _require(isinstance(pairing_root, dict), f"{path}: runner pairing audit missing")
    for num_ssu in sorted({key[1] for key in raw_keys}):
        group = [compact_rows[key] for key in compact_rows if key[1] == num_ssu]
        root_entry = pairing_root.get(str(num_ssu))
        _require(
            isinstance(root_entry, dict)
            and root_entry.get("all_available_rows_paired") is True,
            f"{path}: runner pairing audit SSU{num_ssu}",
        )
        for field in INPUT_FINGERPRINT_FIELDS:
            _require(
                len({row["input_fingerprints"][field] for row in group}) == 1,
                f"{path}: unpaired {field} at SSU{num_ssu}",
            )
        reference = group[0]
        for current in group[1:]:
            _require(
                _equivalent_derived(
                    current["workload_statistics"], reference["workload_statistics"]
                ),
                f"{path}: unpaired workload statistics at SSU{num_ssu}",
            )
            _require(
                _equivalent_derived(
                    current["prefix_workload_statistics"],
                    reference["prefix_workload_statistics"],
                ),
                f"{path}: unpaired prefix statistics at SSU{num_ssu}",
            )
            _require(
                current["prefix_materialized_fingerprints"]
                == reference["prefix_materialized_fingerprints"],
                f"{path}: unpaired prefix materialization at SSU{num_ssu}",
            )
    return {
        "path": path,
        "sha256": payload["_analysis_sha256"],
        "seed": seed,
        "source_fingerprint": source,
        "source_manifest": source_manifest,
        "definition_fingerprint": EXPECTED_DEFINITION_FINGERPRINT,
        "config_fingerprint": config,
        "spec": spec,
        "campaign_projection": _campaign_projection(spec),
        "authentication": authentication,
        "path_abi": payload["path_abi"],
        "root_runtime": root_runtime,
        "runtime_campaign_signature": _runtime_campaign_signature(root_runtime),
        "loader_environment": loader,
        "schedule_metadata": payload["schedule_metadata"],
        "selected_keys": selected_keys,
        "rows": compact_rows,
        "raw_rows": raw_by_key,
    }


def _validate_campaign(shards: list[dict]) -> dict:
    _require(bool(shards), "at least one shard is required")
    reference = shards[0]
    seed_reference: dict[int, dict] = {}
    assignment_by_seed = {}
    rows: dict[tuple[int, int, str], dict] = {}
    raw_rows = {}
    provenance = {}
    for shard in shards:
        for field in (
            "source_fingerprint",
            "source_manifest",
            "definition_fingerprint",
            "campaign_projection",
            "authentication",
            "path_abi",
            "runtime_campaign_signature",
        ):
            _require(
                shard[field] == reference[field],
                f"{shard['path']}: campaign {field} differs",
            )
        seed = shard["seed"]
        if seed in seed_reference:
            previous = seed_reference[seed]
            _require(
                shard["config_fingerprint"] == previous["config_fingerprint"],
                f"seed{seed}: config differs across shards",
            )
            _require(
                shard["spec"] == previous["spec"],
                f"seed{seed}: spec differs across shards",
            )
            _require(
                shard["schedule_metadata"] == previous["schedule_metadata"],
                f"seed{seed}: schedule differs across shards",
            )
        else:
            seed_reference[seed] = shard
            assignment_by_seed[seed] = shard["spec"]["workload"]["assignment"]
        for (case, num_ssu), row in shard["rows"].items():
            key = (seed, num_ssu, CASE_TO_ROLE[case])
            _require(
                key not in rows,
                f"duplicate campaign row seed{seed}/SSU{num_ssu}/{key[2]}",
            )
            rows[key] = row
            raw_rows[key] = shard["raw_rows"][(case, num_ssu)]
            provenance[key] = {"path": shard["path"], "sha256": shard["sha256"]}
    _require(
        len(set(assignment_by_seed.values())) == len(assignment_by_seed),
        "different seeds reused the same assignment",
    )

    # Exact finite inputs must pair across all five strategies in a cell.  The
    # schedule component must also remain identical across SSU topologies.
    for seed in FORMAL_SEEDS:
        seed_rows = [row for key, row in rows.items() if key[0] == seed]
        if not seed_rows:
            continue
        for field in (
            "catalog",
            "recipe",
            "schedule",
            "assignment",
            "prefix_32_assignment",
            "full_assignment",
        ):
            _require(
                len({row["input_fingerprints"][field] for row in seed_rows}) == 1,
                f"seed{seed}: cross-SSU {field} differs",
            )
        for num_ssu in FORMAL_SSUS:
            group = [
                rows[(seed, num_ssu, role)]
                for role in ROLE_ORDER
                if (seed, num_ssu, role) in rows
            ]
            if not group:
                continue
            for field in INPUT_FINGERPRINT_FIELDS:
                _require(
                    len({row["input_fingerprints"][field] for row in group}) == 1,
                    f"seed{seed}/SSU{num_ssu}: unpaired {field}",
                )
            reference_row = group[0]
            for current in group[1:]:
                _require(
                    _equivalent_derived(
                        current["workload_statistics"],
                        reference_row["workload_statistics"],
                    ),
                    f"seed{seed}/SSU{num_ssu}: workload statistics differ",
                )
                _require(
                    _equivalent_derived(
                        current["prefix_workload_statistics"],
                        reference_row["prefix_workload_statistics"],
                    ),
                    f"seed{seed}/SSU{num_ssu}: prefix statistics differ",
                )
                _require(
                    current["prefix_materialized_fingerprints"]
                    == reference_row["prefix_materialized_fingerprints"],
                    f"seed{seed}/SSU{num_ssu}: prefix fingerprints differ",
                )
    expected = {
        (seed, num_ssu, role)
        for seed in FORMAL_SEEDS
        for num_ssu in FORMAL_SSUS
        for role in ROLE_ORDER
    }
    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    _require(not extra, f"campaign contains rows outside frozen plan: {extra}")
    return {
        "rows": rows,
        "raw_rows": raw_rows,
        "provenance": provenance,
        "expected": sorted(expected),
        "missing": missing,
        "complete": not missing and len(rows) == 45,
        "seed_configs": {
            seed: shard["config_fingerprint"]
            for seed, shard in sorted(seed_reference.items())
        },
        "runtime_hosts": sorted(
            {shard["root_runtime"]["hostname"] for shard in shards}
        ),
        "runtime_platforms": sorted(
            {shard["root_runtime"]["platform"] for shard in shards}
        ),
    }


DEPLOYMENT_CANDIDATES = ("threshold-only", "interval-only", "combined")


def _cell_decision(candidate: dict, anchor: dict) -> dict:
    _require(
        candidate["seed"] == anchor["seed"]
        and candidate["num_ssu"] == anchor["num_ssu"],
        "cell decision pairing",
    )
    _require(anchor["role"] == "A0", "cell anchor must be A0")
    _require(anchor["cir_entry_write_rate_hz"] > 0.0, "A0 has zero CIR write rate")
    _require(anchor["control_evaluation_rate_hz"] > 0.0, "A0 has zero evaluation rate")
    discrete_margin_pp = max(
        SLO_NONINFERIOR_MARGIN_PP,
        candidate["max_equal_npu_single_outcome_weight_pp"]
        + anchor["max_equal_npu_single_outcome_weight_pp"],
    )
    util_delta = (
        candidate["mean_npu_utilization_pct"] - anchor["mean_npu_utilization_pct"]
    )
    alpha2_delta = (
        candidate["equal_npu_slo_alpha2_pct"] - anchor["equal_npu_slo_alpha2_pct"]
    )
    alpha15_delta = (
        candidate["equal_npu_slo_alpha15_pct"] - anchor["equal_npu_slo_alpha15_pct"]
    )
    p99_ratio = candidate["p99_ttft_ms"] / anchor["p99_ttft_ms"]
    write_ratio = (
        candidate["cir_entry_write_rate_hz"] / anchor["cir_entry_write_rate_hz"]
    )
    evaluation_ratio = (
        candidate["control_evaluation_rate_hz"] / anchor["control_evaluation_rate_hz"]
    )
    gates = {
        "candidate_row_valid": candidate["row_gate_passed"],
        "anchor_row_valid": anchor["row_gate_passed"],
        "utilization_noninferior": util_delta >= -UTIL_NONINFERIOR_MARGIN_PP - 1e-12,
        "alpha2_noninferior": alpha2_delta >= -discrete_margin_pp - 1e-12,
        "alpha15_noninferior": alpha15_delta >= -discrete_margin_pp - 1e-12,
        "p99_noninferior": p99_ratio <= 1.0 + P99_NONINFERIOR_RELATIVE_MARGIN + 1e-12,
        "entry_write_reduction_at_least_50pct": write_ratio
        <= 1.0 - WRITE_REDUCTION_MIN + 1e-12,
        "evaluation_reduction_at_least_40pct": evaluation_ratio
        <= 1.0 - EVALUATION_REDUCTION_MIN + 1e-12,
    }
    return {
        "seed": candidate["seed"],
        "num_ssu": candidate["num_ssu"],
        "candidate_role": candidate["role"],
        "utilization_delta_pp_vs_a0": util_delta,
        "alpha2_equal_npu_slo_delta_pp_vs_a0": alpha2_delta,
        "alpha15_equal_npu_slo_delta_pp_vs_a0": alpha15_delta,
        "p99_ttft_delta_ms_vs_a0": candidate["p99_ttft_ms"] - anchor["p99_ttft_ms"],
        "p99_ttft_ratio_vs_a0": p99_ratio,
        "cir_entry_write_rate_ratio_vs_a0": write_ratio,
        "control_evaluation_rate_ratio_vs_a0": evaluation_ratio,
        "candidate_single_outcome_weight_pp": candidate[
            "max_equal_npu_single_outcome_weight_pp"
        ],
        "anchor_single_outcome_weight_pp": anchor[
            "max_equal_npu_single_outcome_weight_pp"
        ],
        "slo_noninferiority_margin_pp": discrete_margin_pp,
        "gates": gates,
        "qualified": all(gates.values()),
    }


def _build_selection(campaign: dict) -> dict:
    rows = campaign["rows"]
    decisions = []
    for seed in FORMAL_SEEDS:
        for num_ssu in FORMAL_SSUS:
            anchor = rows.get((seed, num_ssu, "A0"))
            if anchor is None:
                continue
            for role in DEPLOYMENT_CANDIDATES:
                candidate = rows.get((seed, num_ssu, role))
                if candidate is not None:
                    decisions.append(_cell_decision(candidate, anchor))
    global_candidates = []
    for role in DEPLOYMENT_CANDIDATES:
        role_decisions = [item for item in decisions if item["candidate_role"] == role]
        complete_cells = len(role_decisions) == len(FORMAL_SEEDS) * len(FORMAL_SSUS)
        globally_qualified = complete_cells and all(
            item["qualified"] for item in role_decisions
        )
        global_candidates.append(
            {
                "role": role,
                "complete_cell_count": len(role_decisions),
                "expected_cell_count": len(FORMAL_SEEDS) * len(FORMAL_SSUS),
                "globally_qualified": globally_qualified,
                "failed_cells": [
                    [item["seed"], item["num_ssu"]]
                    for item in role_decisions
                    if not item["qualified"]
                ],
                "worst_entry_write_ratio": (
                    max(
                        item["cir_entry_write_rate_ratio_vs_a0"]
                        for item in role_decisions
                    )
                    if role_decisions
                    else None
                ),
                "worst_evaluation_ratio": (
                    max(
                        item["control_evaluation_rate_ratio_vs_a0"]
                        for item in role_decisions
                    )
                    if role_decisions
                    else None
                ),
                "worst_alpha2_slo_delta_pp": (
                    min(
                        item["alpha2_equal_npu_slo_delta_pp_vs_a0"]
                        for item in role_decisions
                    )
                    if role_decisions
                    else None
                ),
            }
        )
    qualified = [item for item in global_candidates if item["globally_qualified"]]
    qualified.sort(
        key=lambda item: (
            item["worst_entry_write_ratio"],
            item["worst_evaluation_ratio"],
            -item["worst_alpha2_slo_delta_pp"],
            DEPLOYMENT_CANDIDATES.index(item["role"]),
        )
    )
    a0_rows = [row for key, row in rows.items() if key[2] == "A0"]
    a0_fallback_qualified = len(a0_rows) == len(FORMAL_SEEDS) * len(
        FORMAL_SSUS
    ) and all(row["row_gate_passed"] for row in a0_rows)
    selected_role = (
        qualified[0]["role"] if qualified else ("A0" if a0_fallback_qualified else None)
    )

    far_better_by_ssu = []
    if selected_role is not None:
        for num_ssu in (6, 10):
            slo_deltas = []
            util_deltas = []
            for seed in FORMAL_SEEDS:
                candidate = rows.get((seed, num_ssu, selected_role))
                baseline = rows.get((seed, num_ssu, "B"))
                if candidate is not None and baseline is not None:
                    slo_deltas.append(
                        candidate["equal_npu_slo_alpha2_pct"]
                        - baseline["equal_npu_slo_alpha2_pct"]
                    )
                    util_deltas.append(
                        candidate["mean_npu_utilization_pct"]
                        - baseline["mean_npu_utilization_pct"]
                    )
            complete = len(slo_deltas) == len(FORMAL_SEEDS)
            far_better_by_ssu.append(
                {
                    "num_ssu": num_ssu,
                    "complete": complete,
                    "median_alpha2_slo_delta_pp_vs_baseline": statistics.median(
                        slo_deltas
                    )
                    if slo_deltas
                    else None,
                    "median_utilization_delta_pp_vs_baseline": statistics.median(
                        util_deltas
                    )
                    if util_deltas
                    else None,
                    "far_better_gate": (
                        complete
                        and statistics.median(slo_deltas) >= FAR_BETTER_SLO_PP
                        and statistics.median(util_deltas) >= -1e-12
                    ),
                }
            )
    all_rows_report_ready = campaign["complete"] and all(
        row["row_gate_passed"] for row in rows.values()
    )
    astar_frozen = all_rows_report_ready and selected_role is not None
    far_better_claim_allowed = astar_frozen and any(
        item["far_better_gate"] for item in far_better_by_ssu
    )
    return {
        "selection_rule_version": SELECTION_RULE_VERSION,
        "cell_decisions": decisions,
        "global_candidates": global_candidates,
        "a0_fallback_qualified": a0_fallback_qualified,
        "selected_role": selected_role,
        "selected_case": ROLE_TO_CASE.get(selected_role),
        "all_45_rows_report_ready": all_rows_report_ready,
        "astar_frozen": astar_frozen,
        "far_better_by_ssu": far_better_by_ssu,
        "far_better_baseline_claim_allowed": far_better_claim_allowed,
        "allowed_claim": (
            "far superior to Baseline"
            if far_better_claim_allowed
            else (
                "quality noninferior with lower control cost"
                if astar_frozen and selected_role != "A0"
                else (
                    "A0 retained; no deployable control-cost reduction qualified"
                    if astar_frozen
                    else "no frozen claim"
                )
            )
        ),
    }


def _build_freeze_artifact(campaign: dict, selection: dict, shards: list[dict]) -> dict:
    reference = shards[0]
    artifact = {
        "schema_version": 1,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "frozen": selection["astar_frozen"],
        "selected_role": selection["selected_role"],
        "selected_case": selection["selected_case"],
        "selected_function_parameters": (
            {
                "case": EXPECTED_CASE_BY_NAME[selection["selected_case"]],
                "adaptive_controller_and_caps": (
                    EXPECTED_ADAPTIVE_SPEC
                    if EXPECTED_CASE_BY_NAME[selection["selected_case"]]["kind"]
                    == "adaptive"
                    else None
                ),
            }
            if selection["selected_case"] is not None
            else None
        ),
        "source_fingerprint": reference["source_fingerprint"],
        "source_manifest": reference["source_manifest"],
        "data_authentication": reference["authentication"],
        "definition_fingerprint": reference["definition_fingerprint"],
        "config_fingerprints_by_seed": campaign["seed_configs"],
        "seeds": list(FORMAL_SEEDS),
        "ssus": list(FORMAL_SSUS),
        "num_npu": NUM_NPU,
        "warmup_requests_per_npu": WARMUP_REQUESTS,
        "settle_ms": SETTLE_MS,
        "measurement_ms": ACTIVE_MEASUREMENT_MS,
        "block_ms": ACTIVE_BLOCK_MS,
        "backing_requests_per_npu": ACTIVE_BACKING_REQUESTS,
        "cell_decisions": selection["cell_decisions"],
        "global_candidates": selection["global_candidates"],
        "far_better_by_ssu": selection["far_better_by_ssu"],
        "schema_v2_evidence_limitations": {
            "measurement_control_counters": (
                "rates are recomputed from reported half-open-window deltas and "
                "checked for fleet/per-SSU conservation plus cumulative upper "
                "bounds; schema v2 has no start/end counter snapshots or event "
                "digest for independent delta replay"
            ),
            "completed_by_npu_at_stop": (
                "checked against every tagged request sequence, finite backing, "
                "all_input_requests_completed=false, and the configured margin; "
                "schema v2 has no full completion trace to independently recover "
                "the exact untagged completion count"
            ),
        },
        "input_shards": [
            {"path": shard["path"], "sha256": shard["sha256"]} for shard in shards
        ],
    }
    artifact["freeze_fingerprint"] = _canonical_hash(
        artifact, b"confirm32-freeze-artifact:v1\0"
    )
    return artifact


def _public_row(row: dict) -> dict:
    omitted = {
        "workload_statistics",
        "prefix_workload_statistics",
        "input_fingerprints",
        "prefix_materialized_fingerprints",
    }
    return {key: value for key, value in row.items() if key not in omitted}


def _build_analysis(
    campaign: dict,
    selection: dict,
    freeze: dict,
    shards: list[dict],
    formal_plan: dict,
    local_source_audit: dict,
) -> dict:
    ordered_rows = [
        _public_row(campaign["rows"][key]) for key in sorted(campaign["rows"])
    ]
    reference = shards[0]
    failed_row_gates = [
        {
            "seed": row["seed"],
            "num_ssu": row["num_ssu"],
            "role": row["role"],
            "failed": [name for name, passed in row["row_gates"].items() if not passed],
        }
        for row in ordered_rows
        if not row["row_gate_passed"]
    ]
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": ANALYSIS_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": campaign["complete"],
        "validated_result_count": len(campaign["rows"]),
        "expected_result_count": 45,
        "missing_results": [list(key) for key in campaign["missing"]],
        "hard_validation_passed": True,
        "formal_plan": formal_plan,
        "local_source_and_data_audit": local_source_audit,
        "failed_report_readiness_row_gates": failed_row_gates,
        "source_fingerprint": reference["source_fingerprint"],
        "source_manifest": reference["source_manifest"],
        "definition_fingerprint": reference["definition_fingerprint"],
        "data_authentication": reference["authentication"],
        "config_fingerprints_by_seed": campaign["seed_configs"],
        "path_abi": reference["path_abi"],
        "runtime_campaign_signature": reference["runtime_campaign_signature"],
        "runtime_hosts": campaign["runtime_hosts"],
        "runtime_platforms": campaign["runtime_platforms"],
        "input_shards": [
            {"path": shard["path"], "sha256": shard["sha256"]} for shard in shards
        ],
        "rules": {
            "coverage_hard_min_completed_tagged_requests_per_npu": MIN_REQUESTS_HARD,
            "coverage_freeze_target_per_npu": MIN_REQUESTS_FREEZE_TARGET,
            "minimum_backing_margin": MIN_BACKING_MARGIN,
            "expected_invariant_count": len(EXPECTED_INVARIANTS),
            "stationarity_blocks": ACTIVE_BLOCKS,
            "stationarity_boundaries": ACTIVE_BOUNDARIES,
            "utilization_half_delta_limit_pp": UTIL_HALF_LIMIT_PP,
            "utilization_projected_trend_limit_pp": UTIL_TREND_LIMIT_PP,
            "fleet_served_half_relative_limit": SERVED_HALF_RELATIVE_LIMIT,
            "utilization_noninferiority_margin_pp": UTIL_NONINFERIOR_MARGIN_PP,
            "alpha2_base_noninferiority_margin_pp": SLO_NONINFERIOR_MARGIN_PP,
            "alpha15_noninferiority": "same conservative discrete margin as alpha2",
            "p99_relative_noninferiority_margin": P99_NONINFERIOR_RELATIVE_MARGIN,
            "entry_write_reduction_minimum": WRITE_REDUCTION_MIN,
            "control_evaluation_reduction_minimum": EVALUATION_REDUCTION_MIN,
        },
        "schema_v2_evidence_limitations": freeze["schema_v2_evidence_limitations"],
        "rows": ordered_rows,
        "selection": selection,
        "freeze_artifact": freeze,
        "analysis_source_sha256": _file_sha256(Path(__file__).resolve()),
        "analysis_source_stable_during_audit": True,
        "local_source_closure_stable_during_audit": True,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _csv_text(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = {
            field: (
                json.dumps(
                    row.get(field),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if isinstance(row.get(field), (dict, list))
                else row.get(field)
            )
            for field in fields
        }
        writer.writerow(normalized)
    return stream.getvalue()


ROW_CSV_FIELDS = (
    "seed",
    "num_ssu",
    "role",
    "case",
    "equal_npu_slo_alpha2_pct",
    "request_weighted_slo_alpha2_pct",
    "equal_npu_slo_alpha15_pct",
    "request_weighted_slo_alpha15_pct",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_npu_utilization_pct",
    "mean_ssd_utilization_pct",
    "mean_npu_link_utilization_pct",
    "measurement_request_count",
    "requests_per_npu_min",
    "requests_per_npu_median",
    "requests_per_npu_max",
    "backing_margin_fastest_npu",
    "pressure_read_rate_hz_per_ssu",
    "cir_entry_write_rate_hz_per_ssu",
    "cir_transaction_rate_hz_per_ssu",
    "control_evaluation_rate_hz_per_ssu",
    "utilization_half_delta_pp",
    "utilization_projected_change_pp",
    "fleet_served_half_relative_delta",
    "stationarity_rules",
    "row_gates",
    "row_gate_passed",
)


CELL_CSV_FIELDS = (
    "seed",
    "num_ssu",
    "candidate_role",
    "utilization_delta_pp_vs_a0",
    "alpha2_equal_npu_slo_delta_pp_vs_a0",
    "alpha15_equal_npu_slo_delta_pp_vs_a0",
    "p99_ttft_delta_ms_vs_a0",
    "p99_ttft_ratio_vs_a0",
    "cir_entry_write_rate_ratio_vs_a0",
    "control_evaluation_rate_ratio_vs_a0",
    "slo_noninferiority_margin_pp",
    "gates",
    "qualified",
)


def _write_outputs(output_dir: Path, analysis: dict) -> None:
    output_dir = output_dir.expanduser().resolve()
    _atomic_write_json(output_dir / "audit.json", analysis)
    _atomic_write_json(output_dir / "freeze_artifact.json", analysis["freeze_artifact"])
    _atomic_write_text(
        output_dir / "rows.csv", _csv_text(analysis["rows"], ROW_CSV_FIELDS)
    )
    _atomic_write_text(
        output_dir / "cell_decisions.csv",
        _csv_text(analysis["selection"]["cell_decisions"], CELL_CSV_FIELDS),
    )
    _atomic_write_text(output_dir / "report.md", _report_text(analysis))


def _fmt(value, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _report_text(analysis: dict) -> str:
    selection = analysis["selection"]
    plan = analysis["formal_plan"]
    if analysis["complete"]:
        status = "COMPLETE"
    elif analysis["failed_report_readiness_row_gates"]:
        window_s = float(plan["measurement_ms"]) / 1000.0
        status = (
            f"{window_s:g}s CALIBRATION FAILURE EVIDENCE — " "PARTIAL — NOT FREEZABLE"
        )
    else:
        status = "PARTIAL — NOT FREEZABLE"
    archive = analysis["local_source_and_data_audit"].get("source_archive")
    archive_line = (
        f"- immutable source archive：`{archive['sha256']}`（{archive['size_bytes']} bytes，{archive['verified_regular_file_count']} 个成员逐一匹配 manifest）。"
        if archive is not None
        else "- 未提供 immutable source archive；本次仅验证 source-root closure。"
    )
    lines = [
        f"# confirm32 严格审计 — {status}",
        "",
        f"- 已验证行：`{analysis['validated_result_count']}/45`；缺失：`{len(analysis['missing_results'])}`。",
        f"- 正式形状：measurement=`{plan['measurement_ms']:.0f} ms`，backing=`{plan['backing_requests_per_npu']}` requests/NPU，blocks/boundaries=`{plan['stationarity_blocks']}/{plan['stationarity_boundaries']}`；所有输入 shard 必须完全一致，禁止混合窗口。",
        "- 29 项 simulator invariant 的目录和值被逐行严格核对；可由 artifact 观测量独立重建的资源守恒、block/boundary、placement 和输入哈希另行重算。",
        f"- A* 冻结：`{selection['astar_frozen']}`；选择：`{selection['selected_role']}` / `{selection['selected_case']}`。",
        f"- 可使用的表述：**{selection['allowed_claim']}**。",
        f"- source：`{analysis['source_fingerprint']}`；data catalog：`{analysis['data_authentication']['catalog_hash']}`。",
        f"- 本地 source/data closure：`{analysis['local_source_and_data_audit']['verified_file_count']}` 个文件逐一 SHA-256 匹配；`data` 由 stdlib 直接解析，未导入 sim、未读取 cache。",
        archive_line,
        f"- runtime scientific signature：`{json.dumps(analysis['runtime_campaign_signature'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 规则口径",
        "",
        "- 每 NPU tagged-completion 硬门槛为 4，冻结目标为 8；drain stop 的 backing 余量至少 32。",
        "- 相对 A0：utilization 非劣界 -0.5 pp；α=2 和 α=1.5 使用 `max(1 pp, 两行各一个 binary outcome 权重之和)`；p99 最多 +1%。",
        "- 每 cell 的 CIR entry-write rate 至少下降 50%，control-evaluation rate 至少下降 40%。",
        "- stationarity：上下半窗 utilization ≤1 pp、Theil–Sen 全窗投影 ≤2 pp、fleet served GB/s 上下半窗相对差 ≤2%。",
        "- queue burst 不是魔数：从已认证 84-profile catalog 取最大 per-layer KV GB，再乘 32 NPU，得到最保守的 fleet 单层 burst 上界；只有稳健斜率为正、至少 75% 边界步不下降且净增长超过该上界才判持续增长失败。",
        "- 整批封存纪律：即便某一候选未被选中，45 行中任一行未通过 report-readiness gate，整批也不能冻结 A*；逐候选资格仍在 cell_decisions 单独给出。",
        "",
        "## Schema v2 证据边界",
        "",
        "- control write/read/evaluation 频率：analyzer 从 measurement delta/count 重新计算频率，并核对 fleet/per-SSU 守恒、计数顺序、min-interval 物理上界及 legacy 累计上界；artifact 没有 counter 起止快照或 control-event digest，因此这不是独立事件重放。",
        "- completed/backing：analyzer 要求 stop count 不小于所有 tagged request 的最大 sequence+1、未耗尽全部输入且余量≥32；artifact 没有完整 completion trace，因此无法独立恢复窗口外未 tagged completion 的精确数量。",
        "- 以上字段仍受运行时 source/config fingerprint、source_stable、29 invariant 和最终 shard SHA 约束，但 schema v2 本身不能抵抗对这些字段及全部派生字段的同步自洽篡改。",
        "",
    ]
    if analysis["missing_results"]:
        lines.extend(["## 缺失行", "", "```text"])
        lines.extend(
            f"seed={seed} SSU={ssu} role={role}"
            for seed, ssu, role in analysis["missing_results"]
        )
        lines.extend(["```", ""])
    if analysis["failed_report_readiness_row_gates"]:
        lines.extend(
            [
                "## 未通过冻结就绪 gate 的行",
                "",
                "| seed | SSU | role | failed |",
                "|---:|---:|:---|:---|",
            ]
        )
        for item in analysis["failed_report_readiness_row_gates"]:
            lines.append(
                f"| {item['seed']} | {item['num_ssu']} | {item['role']} | {', '.join(item['failed'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 逐 cell 预注册判定",
            "",
            "| seed | SSU | candidate | Δutil pp | Δα2 SLO pp | Δα1.5 SLO pp | p99 ratio | write ratio | eval ratio | qualified |",
            "|---:|---:|:---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for item in selection["cell_decisions"]:
        lines.append(
            f"| {item['seed']} | {item['num_ssu']} | {item['candidate_role']} | "
            f"{_fmt(item['utilization_delta_pp_vs_a0'])} | "
            f"{_fmt(item['alpha2_equal_npu_slo_delta_pp_vs_a0'])} | "
            f"{_fmt(item['alpha15_equal_npu_slo_delta_pp_vs_a0'])} | "
            f"{_fmt(item['p99_ttft_ratio_vs_a0'], 4)} | "
            f"{_fmt(item['cir_entry_write_rate_ratio_vs_a0'], 4)} | "
            f"{_fmt(item['control_evaluation_rate_ratio_vs_a0'], 4)} | "
            f"{item['qualified']} |"
        )
    lines.extend(
        [
            "",
            "## 全局候选",
            "",
            "| role | cells | globally qualified | worst write ratio | worst eval ratio | worst Δα2 SLO pp |",
            "|:---|---:|:---:|---:|---:|---:|",
        ]
    )
    for item in selection["global_candidates"]:
        lines.append(
            f"| {item['role']} | {item['complete_cell_count']}/{item['expected_cell_count']} | "
            f"{item['globally_qualified']} | {_fmt(item['worst_entry_write_ratio'], 4)} | "
            f"{_fmt(item['worst_evaluation_ratio'], 4)} | {_fmt(item['worst_alpha2_slo_delta_pp'])} |"
        )
    lines.extend(
        [
            "",
            "## Stationarity 原始量",
            "",
            "| seed | SSU | role | util half Δ pp | util projected Δ pp | served half Δ % | fleet queue net GB | fleet queue slope GB/s | row gate |",
            "|---:|---:|:---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in analysis["rows"]:
        lines.append(
            f"| {row['seed']} | {row['num_ssu']} | {row['role']} | "
            f"{_fmt(row['utilization_half_delta_pp'])} | {_fmt(row['utilization_projected_change_pp'])} | "
            f"{_fmt(100.0 * row['fleet_served_half_relative_delta'])} | "
            f"{_fmt(row['fleet_queue']['net_growth_gb'])} | {_fmt(row['fleet_queue']['theil_sen_slope_gbps'])} | "
            f"{row['row_gate_passed']} |"
        )
    lines.extend(
        [
            "",
            "## 冻结与来源",
            "",
            f"- freeze fingerprint：`{analysis['freeze_artifact']['freeze_fingerprint']}`",
            f"- analyzer SHA-256：`{analysis['analysis_source_sha256']}`",
            "- analyzer 本身及 immutable source/data closure 均在审计开始与结束各重哈希一次，结果一致。",
            "- `complete=false`、任一 report-readiness gate 失败或任一候选 cell 失败时，均不得把该候选写成已冻结 A*。",
            "",
        ]
    )
    return "\n".join(lines)


def _self_test() -> None:
    _require(len(EXPECTED_INVARIANTS) == 29, "self-test invariant count")
    paths = _path_mapping()
    _require(len(paths) == len(set(paths)) == NUM_NPU, "self-test Path uniqueness")
    _require(
        min(paths) == 16 and max(paths) == 243 and 0 not in paths,
        "self-test Path range",
    )
    expected_path = {
        "path_count": 256,
        "group_count": 8,
        "paths_per_group": 32,
        "max_npu": 128,
        "assigned_count": 32,
        "assigned_unique": 32,
        "assigned_min": 16,
        "assigned_max": 243,
        "path_zero_reserved": True,
        "assigned_paths_sha256": _canonical_hash(
            paths, b"ms-scale-control-path-abi:v1\0"
        ),
    }
    _validate_path_abi(expected_path, "self-test")
    first = _expected_profile_index(42, 0, 0, 84)
    _require(
        first == _expected_profile_index(42, 0, 0, 84), "self-test deterministic RNG"
    )
    _require(0 <= first < 84, "self-test RNG range")
    try:
        json.loads('{"a":1,"a":2}', object_pairs_hook=_reject_duplicate_pairs)
    except ValidationError:
        pass
    else:
        raise ValidationError("self-test duplicate JSON was accepted")

    rows = {}
    for seed in FORMAL_SEEDS:
        for num_ssu in FORMAL_SSUS:
            for role in ROLE_ORDER:
                if role == "B":
                    alpha2, alpha15, util, p99, writes, evaluations = (
                        70.0,
                        55.0,
                        49.0,
                        110.0,
                        0.0,
                        0.0,
                    )
                elif role == "A0":
                    alpha2, alpha15, util, p99, writes, evaluations = (
                        80.0,
                        65.0,
                        50.0,
                        100.0,
                        100.0,
                        100.0,
                    )
                elif role == "threshold-only":
                    alpha2, alpha15, util, p99, writes, evaluations = (
                        81.0,
                        66.0,
                        50.0,
                        100.5,
                        40.0,
                        50.0,
                    )
                elif role == "interval-only":
                    alpha2, alpha15, util, p99, writes, evaluations = (
                        81.0,
                        66.0,
                        50.0,
                        100.5,
                        45.0,
                        55.0,
                    )
                else:
                    alpha2, alpha15, util, p99, writes, evaluations = (
                        82.0,
                        67.0,
                        51.0,
                        100.0,
                        30.0,
                        45.0,
                    )
                rows[(seed, num_ssu, role)] = {
                    "seed": seed,
                    "num_ssu": num_ssu,
                    "role": role,
                    "row_gate_passed": True,
                    "max_equal_npu_single_outcome_weight_pp": 0.390625,
                    "equal_npu_slo_alpha2_pct": alpha2,
                    "equal_npu_slo_alpha15_pct": alpha15,
                    "mean_npu_utilization_pct": util,
                    "p99_ttft_ms": p99,
                    "cir_entry_write_rate_hz": writes,
                    "control_evaluation_rate_hz": evaluations,
                }
    selection = _build_selection({"rows": rows, "complete": True})
    _require(
        selection["selected_role"] == "combined" and selection["astar_frozen"],
        "self-test combined selection",
    )
    _require(
        selection["far_better_baseline_claim_allowed"], "self-test far-better claim"
    )
    tampered = {key: dict(value) for key, value in rows.items()}
    tampered[(42, 6, "combined")]["equal_npu_slo_alpha2_pct"] = 70.0
    fallback = _build_selection({"rows": tampered, "complete": True})
    _require(
        fallback["selected_role"] == "threshold-only",
        "self-test deterministic fallback ranking",
    )
    partial = _build_selection({"rows": rows, "complete": False})
    _require(not partial["astar_frozen"], "self-test partial campaign froze A*")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shards", nargs="*", type=Path, help="confirm32 runner shard JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for audit JSON/CSV/report; omitted means read-only stdout summary",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "repository source root used to hash the complete source manifest and "
            "directly parse data (default: analyzer directory)"
        ),
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        help=(
            "optional immutable tar archive; every regular member and its hash "
            "must exactly equal the shard source manifest"
        ),
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return status 2 unless all 45 frozen-plan rows are present",
    )
    parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="return status 3 unless the complete campaign freezes an A* role",
    )
    parser.add_argument(
        "--measurement-ms",
        type=float,
        help=(
            "optional exact gate for the common immutable measurement window; "
            "for example 16000"
        ),
    )
    parser.add_argument(
        "--backing",
        "--requests-per-npu",
        dest="backing",
        type=int,
        help=(
            "optional exact gate for the common finite backing requests/NPU; "
            "for example 256"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic pure-analysis tests without reading shards",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _require(
            not args.shards
            and args.output_dir is None
            and args.measurement_ms is None
            and args.backing is None,
            "--self-test does not accept shards/output/formal-plan gates",
        )
        _self_test()
        print(
            json.dumps(
                {"self_test": "passed", "analysis": ANALYSIS_NAME}, ensure_ascii=False
            )
        )
        return 0
    _require(bool(args.shards), "at least one shard is required")
    analyzer_source_start = _file_sha256(Path(__file__).resolve())
    paths = [path.expanduser().resolve() for path in args.shards]
    _require(len(paths) == len(set(paths)), "the same shard path was supplied twice")
    raw_shards = [_read_shard(path) for path in paths]
    formal_plan = _configure_formal_plan(
        raw_shards,
        expected_measurement_ms=args.measurement_ms,
        expected_backing=args.backing,
    )
    local_source_audit = _validate_local_source_closure(
        raw_shards, args.source_root, args.source_archive
    )
    shards = [_validate_shard(payload) for payload in raw_shards]
    campaign = _validate_campaign(shards)
    selection = _build_selection(campaign)
    freeze = _build_freeze_artifact(campaign, selection, shards)
    analysis = _build_analysis(
        campaign,
        selection,
        freeze,
        shards,
        formal_plan,
        local_source_audit,
    )
    ending_local_source_audit = _validate_local_source_closure(
        raw_shards, args.source_root, args.source_archive
    )
    _require(
        ending_local_source_audit == local_source_audit,
        "local source/data/archive closure changed during the audit",
    )
    analyzer_source_end = _file_sha256(Path(__file__).resolve())
    _require(
        analyzer_source_end == analyzer_source_start,
        "confirm32_analysis.py changed during the audit",
    )
    _require(
        analysis["analysis_source_sha256"] == analyzer_source_end,
        "analysis source hash was not captured coherently",
    )
    if args.output_dir is not None:
        _write_outputs(args.output_dir, analysis)
    print(
        json.dumps(
            {
                "analysis": ANALYSIS_NAME,
                "complete": analysis["complete"],
                "validated_result_count": analysis["validated_result_count"],
                "missing_count": len(analysis["missing_results"]),
                "failed_report_readiness_rows": len(
                    analysis["failed_report_readiness_row_gates"]
                ),
                "formal_plan": formal_plan,
                "selected_role": selection["selected_role"],
                "astar_frozen": selection["astar_frozen"],
                "allowed_claim": selection["allowed_claim"],
                "output_dir": str(args.output_dir.expanduser().resolve())
                if args.output_dir is not None
                else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_complete and not analysis["complete"]:
        return 2
    if args.require_frozen and not selection["astar_frozen"]:
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        raise SystemExit(f"validation error: {error}") from error
