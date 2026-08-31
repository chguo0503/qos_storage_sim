"""Strict multi-seed consistency audit for report128 experiment shards.

The program does not import the runner or simulator.  It independently
recomputes the metrics and conservation relations observable in each JSON
artifact; with ``--source-root`` it also binds embedded hashes to a local source
closure.  This is strong internal-consistency evidence, not a replay/proof of
the simulator.  One shard must contain the five B/L0/L*/A0/A* cases for one
``(seed, SSU)`` cell.
"""

from __future__ import annotations

import argparse
import ast
import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import tarfile
from functools import lru_cache
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ANALYSIS_SCHEMA_VERSION = 3
LEGACY_SHARD_SCHEMA_VERSION = 2
FORMAL_SHARD_SCHEMA_VERSION = 3
SUMMARY_SCHEMA_VERSION = 2
LEGACY_EXPERIMENT = "128npu_report_curve_random_real_data_v1"
FORMAL_EXPERIMENT = "128npu_report_curve_random_real_data_h70_v2"
DEFINITION = "report128"
NUM_NPU = 128
N_LAYERS = 16
BATCH_SIZE = 1
FORMAL_SSUS = (8, 12, 16, 20, 24, 32, 40, 48, 72)
FORMAL_SEEDS = (42, 43, 44)
ROLE_ORDER = ("B", "L0", "L*", "A0", "A*")
LEGACY_ROLE_TO_CASE = {
    "B": "baseline",
    "L0": "layer_once_ttl_0ms",
    "L*": "layer_once_lstar",
    "A0": "adaptive_t0_i25ms",
    "A*": "adaptive_astar",
}
ROLE_TO_CASE = {
    "B": "baseline",
    "L0": "layer_once_ttl_0ms",
    "L*": "layer_once_lstar",
    "A0": "adaptive_t0_i25ms",
    "A*": "pfo_astar_h70",
}
CASE_TO_ROLE = {
    **{case: role for role, case in LEGACY_ROLE_TO_CASE.items()},
    **{case: role for role, case in ROLE_TO_CASE.items()},
}
H70_FREEZE_SELECTION_RULE = "h70-confirm32-preregistered-freeze-v1"
H70_FREEZE_ROLE = "H70"
H70_FREEZE_CASE = "pfo_astar_h70"
CATEGORIES = ("SS", "SL", "LS", "LL")
DEMAND_BINS = (
    ("le_10", -math.inf, 10.0),
    ("gt_10_le_20", 10.0, 20.0),
    ("gt_20_le_40", 20.0, 40.0),
    ("gt_40_le_50", 40.0, 50.0),
    ("gt_50_le_80", 50.0, 80.0),
    ("gt_80", 80.0, math.inf),
)
PRIMARY_ALPHA = 2.0
SENSITIVITY_ALPHA = 1.5
SLO_EPSILON = 1e-12
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
MIN_REQUESTS_PER_NPU = 8
MIN_BACKING_MARGIN = 32
MIN_STATIONARITY_BLOCKS = 16
UTIL_HALF_DELTA_LIMIT_PP = 1.0
UTIL_TREND_PROJECTED_LIMIT_PP = 2.0
SERVED_HALF_RELATIVE_LIMIT = 0.02
STATIONARITY_RULE_VERSION = "report128-preregistered-stationarity-v1"
FORMAL_CAMPAIGN_SCHEMA_VERSION = 3
FORMAL_CAMPAIGN_NAME = "report128_formal_curve_h70_v2"
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
ROLE_COLORS = {
    "B": "#475569",
    "L0": "#DC2626",
    "L*": "#EA580C",
    "A0": "#2563EB",
    "A*": "#059669",
}


class ValidationError(ValueError):
    """An artifact is not safe to include in the formal report."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


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


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _parse_local_data(path: Path) -> tuple[dict, list[list]]:
    """Parse the tracked profile table without importing the simulator/cache."""

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
            all(value > 0.0 for value in converted),
            f"data profile {key} contains a nonpositive value",
        )
        table[key] = converted
    rows = [
        [key[0], key[1], [float(value) for value in table[key]]]
        for key in sorted(table)
    ]
    return table, rows


def _independent_local_source_closure(root: Path) -> tuple[str, ...]:
    """Rebuild the runner's flat local-Python import closure from source text."""

    pending = ["ms_scale_control_experiment.py"]
    seen = set()
    while pending:
        relative_name = pending.pop()
        if relative_name in seen:
            continue
        path = root / relative_name
        _require(path.is_file(), f"source closure file is absent: {path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_name)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            raise ValidationError(
                f"cannot parse source closure member {relative_name}: {error}"
            ) from error
        seen.add(relative_name)
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
    return tuple(sorted(seen)) + ("data",)


def _validate_local_source_closure(
    payloads: Sequence[dict], source_root: Path, source_archive: Path | None
) -> dict:
    """Bind embedded hashes to an independently supplied local source closure."""

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
    independently_discovered = _independent_local_source_closure(root)
    _require(
        set(manifest) == set(independently_discovered),
        "runner source manifest differs from independently rebuilt local import closure",
    )
    for relative_name, expected_hash in manifest.items():
        _require(
            isinstance(relative_name, str)
            and relative_name
            and not Path(relative_name).is_absolute()
            and ".." not in Path(relative_name).parts
            and _is_sha256(expected_hash),
            f"unsafe or malformed source manifest entry: {relative_name!r}",
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
    catalog_hash = _canonical_hash(
        [[[row[0], row[1]], row[2]] for row in catalog_rows],
        b"random-steady-state:data-catalog:v1\0",
    )
    table_hash = _canonical_hash(catalog_rows, b"authenticated-bw-table:v1\0")
    data_sha256 = _file_sha256(data_path)
    for payload in payloads:
        label = str(payload.get("_analysis_path", "shard"))
        authentication = payload.get("input_authentication")
        _require(isinstance(authentication, dict), f"{label}: input auth missing")
        _require(
            authentication.get("source_sha256") == data_sha256
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
        "root_manifest_all_files_match": True,
        "independent_import_closure": list(independently_discovered),
        "independent_import_closure_matches_manifest": True,
        "source_fingerprint": _canonical_hash(
            manifest, b"ms-scale-control-source:v1\0"
        ),
        "data_sha256": data_sha256,
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


def _read_formal_campaign_spec(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        spec = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(
            f"cannot read formal campaign spec {resolved}: {error}"
        ) from error
    expected_fields = {
        "schema_version",
        "campaign",
        "definition",
        "num_npu",
        "seeds",
        "ssus",
        "roles",
        "backing_requests_per_npu",
        "measurement_ms",
        "block_ms",
        "warmup_requests_per_npu",
        "settle_ms",
        "definition_fingerprint",
        "source_fingerprint",
        "source_archive_sha256",
        "policy_freeze_artifact_sha256",
        "analysis_source_sha256",
        "stationarity_rule_version",
    }
    _require(
        isinstance(spec, dict) and set(spec) == expected_fields,
        "formal campaign spec fields changed",
    )
    _require(
        spec["schema_version"] == FORMAL_CAMPAIGN_SCHEMA_VERSION
        and spec["campaign"] == FORMAL_CAMPAIGN_NAME
        and spec["definition"] == DEFINITION
        and spec["num_npu"] == NUM_NPU,
        "formal campaign identity/topology differs",
    )
    _require(
        tuple(spec["seeds"]) == FORMAL_SEEDS,
        "formal seed plan must be exactly 42,43,44",
    )
    _require(
        tuple(spec["ssus"]) == FORMAL_SSUS,
        "formal SSU plan differs from preregistration",
    )
    _require(spec["roles"] == ROLE_TO_CASE, "formal role/case plan differs")
    _require(
        spec["stationarity_rule_version"] == STATIONARITY_RULE_VERSION,
        "formal stationarity rule version differs",
    )
    backing = _integer(
        spec["backing_requests_per_npu"], "formal backing requests/NPU", minimum=32
    )
    measurement_ms = _finite(spec["measurement_ms"], "formal measurement_ms")
    block_ms = _finite(spec["block_ms"], "formal block_ms")
    _close(block_ms, 500.0, "formal block_ms")
    warmup = _integer(spec["warmup_requests_per_npu"], "formal warmup", minimum=1)
    settle_ms = _nonnegative(spec["settle_ms"], "formal settle_ms")
    _require(backing > warmup, "formal backing must exceed warmup")
    _require(
        measurement_ms > 0.0 and block_ms > 0.0, "formal durations must be positive"
    )
    ratio = measurement_ms / block_ms
    block_count = int(round(ratio))
    _require(
        math.isclose(ratio, block_count, rel_tol=0.0, abs_tol=1e-12)
        and block_count >= MIN_STATIONARITY_BLOCKS
        and block_count % 2 == 0,
        "formal measurement must contain an even number of at least 16 full blocks",
    )
    _require(settle_ms >= 500.0, "formal settle_ms must be at least 500 ms")
    for name in (
        "definition_fingerprint",
        "source_fingerprint",
        "source_archive_sha256",
        "policy_freeze_artifact_sha256",
        "analysis_source_sha256",
    ):
        _require(_is_sha256(spec[name]), f"formal {name} is not SHA-256")
    _require(
        spec["analysis_source_sha256"] == _file_sha256(Path(__file__).resolve()),
        "formal campaign spec names a different analyzer source",
    )
    result = dict(spec)
    result["path"] = str(resolved)
    result["block_count"] = block_count
    result["expected_cell_count"] = len(FORMAL_SEEDS) * len(FORMAL_SSUS)
    result["expected_result_count"] = (
        len(FORMAL_SEEDS) * len(FORMAL_SSUS) * len(ROLE_ORDER)
    )
    result["fingerprint"] = _canonical_hash(
        spec, b"report128-formal-campaign-spec:v3\0"
    )
    result["file_sha256"] = hashlib.sha256(raw).hexdigest()
    result["file_size_bytes"] = len(raw)
    return result


def _validate_policy_freeze_artifact(
    path: Path, expected_sha256: str, reference_cell: dict
) -> dict:
    """Bind report128 A* parameters to a frozen formal32 H70 artifact."""

    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        artifact = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except OSError as error:
        raise ValidationError(f"cannot read policy freeze artifact {resolved}: {error}")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid policy freeze artifact {resolved}: {error}")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    _require(
        actual_sha256 == expected_sha256,
        "policy freeze artifact SHA-256 differs from frozen campaign spec",
    )
    _require(isinstance(artifact, dict), "policy freeze artifact must be an object")
    expected_fields = {
        "schema_version",
        "selection_rule_version",
        "frozen",
        "selected_role",
        "selected_case",
        "selected_function_parameters",
        "source_fingerprint",
        "source_manifest",
        "data_authentication",
        "definition_fingerprint",
        "config_fingerprints_by_seed",
        "seeds",
        "ssus",
        "num_npu",
        "warmup_requests_per_npu",
        "settle_ms",
        "measurement_ms",
        "block_ms",
        "backing_requests_per_npu",
        "cell_decisions",
        "global_candidates",
        "far_better_by_ssu",
        "schema_v2_evidence_limitations",
        "input_shards",
        "freeze_fingerprint",
    }
    _require(
        set(artifact) == expected_fields,
        "policy freeze artifact fields differ from the audited schema",
    )
    _require(
        artifact.get("schema_version") == 1
        and artifact.get("selection_rule_version") == H70_FREEZE_SELECTION_RULE
        and artifact.get("frozen") is True,
        "policy freeze artifact is not a successful frozen H70 confirmation",
    )
    freeze_fingerprint = artifact.get("freeze_fingerprint")
    _require(_is_sha256(freeze_fingerprint), "policy freeze fingerprint is malformed")
    fingerprint_payload = dict(artifact)
    fingerprint_payload.pop("freeze_fingerprint")
    _require(
        _canonical_hash(fingerprint_payload, b"confirm32-freeze-artifact:v1\0")
        == freeze_fingerprint,
        "policy freeze fingerprint does not authenticate its artifact",
    )
    _require(
        artifact.get("num_npu") == 32
        and artifact.get("seeds") == list(FORMAL_SEEDS)
        and artifact.get("ssus") == [6, 10, 18],
        "policy freeze artifact has the wrong confirmation topology/plan",
    )
    confirmation_warmup = _integer(
        artifact.get("warmup_requests_per_npu"), "policy freeze warmup", minimum=1
    )
    confirmation_settle_ms = _finite(
        artifact.get("settle_ms"), "policy freeze settle_ms"
    )
    confirmation_block_ms = _finite(artifact.get("block_ms"), "policy freeze block_ms")
    confirmation_measurement_ms = _finite(
        artifact.get("measurement_ms"), "policy freeze measurement_ms"
    )
    confirmation_backing = _integer(
        artifact.get("backing_requests_per_npu"),
        "policy freeze backing requests/NPU",
        minimum=1,
    )
    _require(
        confirmation_warmup == 8
        and confirmation_settle_ms >= 500.0
        and confirmation_block_ms == 500.0
        and confirmation_measurement_ms >= 8000.0
        and confirmation_backing >= 128,
        "policy freeze artifact has an invalid confirmation run shape",
    )
    confirmation_blocks = confirmation_measurement_ms / confirmation_block_ms
    _require(
        confirmation_blocks.is_integer()
        and int(confirmation_blocks) >= MIN_STATIONARITY_BLOCKS
        and int(confirmation_blocks) % 2 == 0,
        "policy freeze artifact has an invalid stationarity block plan",
    )
    config_fingerprints = artifact.get("config_fingerprints_by_seed")
    _require(
        isinstance(config_fingerprints, dict)
        and set(config_fingerprints) == {str(seed) for seed in FORMAL_SEEDS}
        and all(_is_sha256(value) for value in config_fingerprints.values()),
        "policy freeze config fingerprints are incomplete",
    )
    _require(
        _is_sha256(artifact.get("definition_fingerprint")),
        "policy freeze definition fingerprint is malformed",
    )
    selected_role = artifact.get("selected_role")
    selected_case_name = artifact.get("selected_case")
    _require(
        selected_role == H70_FREEZE_ROLE and selected_case_name == H70_FREEZE_CASE,
        "policy freeze selected role/case identity is inconsistent",
    )
    _require(
        artifact.get("source_fingerprint") == reference_cell["source_fingerprint"]
        and artifact.get("source_manifest") == reference_cell["source_manifest"],
        "policy freeze source closure differs from report128 source closure",
    )
    _require(
        artifact.get("data_authentication") == reference_cell["authentication"],
        "policy freeze data authentication differs from report128 data",
    )
    parameters = artifact.get("selected_function_parameters")
    _require(
        isinstance(parameters, dict)
        and set(parameters)
        == {
            "case",
            "adaptive_controller_and_caps",
            "pfo_controller_and_caps",
            "materialization_policy",
        },
        "frozen H70 function parameters are absent or malformed",
    )
    selected_case = parameters.get("case")
    _require(isinstance(selected_case, dict), "frozen policy case is absent")
    _require(
        selected_case.get("name") == selected_case_name,
        "frozen function case differs from selected_case",
    )
    report_cases = reference_cell["spec"]["cases"]
    report_astar = next(
        (case for case in report_cases if case.get("name") == ROLE_TO_CASE["A*"]),
        None,
    )
    _require(report_astar is not None, "report128 A* case is absent")
    _require(
        selected_case == report_astar,
        "frozen H70 deployment PFOCase is not the exact report128 A* case",
    )
    _require(
        parameters.get("adaptive_controller_and_caps")
        == reference_cell["spec"]["adaptive"],
        "report128 Adaptive controller/caps differ from the frozen selection",
    )
    report_pfo = reference_cell["spec"].get("pfo")
    _require(isinstance(report_pfo, dict), "report128 PFO contract is absent")
    expected_pfo_controller = {
        "controller": report_pfo["controller"],
        "materialized_allocation_stages": report_pfo["materialized_allocation_stages"],
        "request_pin": report_pfo["request_pin"],
        "target_ratio": report_pfo["target_ratio"],
        "required_ratio": report_pfo["required_ratio"],
        "background_reserve_fraction_for_admission_only": report_pfo[
            "background_reserve_fraction_for_admission_only"
        ],
        "internal_deadband_gbps": report_astar["pfo_internal_deadband_gbps"],
        "downstream_cir_write_threshold_gbps": report_pfo[
            "downstream_cir_write_threshold_gbps"
        ],
        "ssd_cap_gbps": reference_cell["spec"]["adaptive"]["ssd_cap_gbps"],
        "npu_cap_gbps": reference_cell["spec"]["adaptive"]["npu_cap_gbps"],
        "path_pressure_reads": report_pfo["path_pressure_reads"],
        "trigger_ownership": report_pfo["trigger_ownership"],
        "real_register_order": report_pfo["real_register_order"],
    }
    _require(
        parameters.get("pfo_controller_and_caps") == expected_pfo_controller,
        "report128 protected-floor controller/caps differ from frozen H70",
    )
    expected_materialization = {
        "policy": "frozen_manifest_hotspot_v1",
        "hot_fraction": report_astar["forecast_hot_fraction"],
        "forecast_requests_per_npu": report_astar["forecast_requests_per_npu"],
        "fleet_full_protection_fraction": report_astar["forecast_hot_fraction"],
        "local_hot_fraction": report_astar["forecast_hot_fraction"],
        "cold_ssu_cir_gbps": 0.0,
        "frozen_for_measurement": True,
        "path_pressure_reads": 0,
    }
    _require(
        parameters.get("materialization_policy") == expected_materialization,
        "report128 H70 materialization mask semantics differ from the freeze",
    )
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "freeze_fingerprint": freeze_fingerprint,
        "selection_rule_version": artifact["selection_rule_version"],
        "selected_role": artifact.get("selected_role"),
        "selected_case": artifact.get("selected_case"),
        "selected_function_parameters": parameters,
        "frozen": True,
        "selected_identity_consistent": True,
        "source_and_data_lineage_matches_report128": True,
        "report128_astar_parameters_match": True,
    }


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


def _equivalent_derived(actual, expected) -> bool:
    """Permit only last-bit host differences in authenticated derived metadata."""
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


def _demand_bin(demand: float) -> str:
    for name, lower, upper in DEMAND_BINS:
        if lower < demand <= upper:
            return name
    raise AssertionError(demand)


def _expected_definition_dict(cases: list[dict], shard_schema_version: int) -> dict:
    adaptive = {
        "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
        "explicit_spill_threshold": 0.75,
        "target_ratio": 0.52,
        "required_ratio": 0.5,
        "background_reserve_fraction": 0.05,
    }
    return {
        "key": DEFINITION,
        "experiment_name": (
            FORMAL_EXPERIMENT
            if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
            else LEGACY_EXPERIMENT
        ),
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssus": list(FORMAL_SSUS),
        "cases": cases,
        "default_requests_per_npu": 128,
        "default_measurement_ms": 8000.0,
        "adaptive": adaptive,
        "report_roles": [
            [
                role,
                (
                    ROLE_TO_CASE[role]
                    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
                    else LEGACY_ROLE_TO_CASE[role]
                ),
            ]
            for role in ROLE_ORDER
        ],
        "require_single_ssu_simulation": True,
    }


def _validate_case_specs(
    spec: Mapping[str, object], prefix: str, shard_schema_version: int
) -> dict[str, dict]:
    raw = spec.get("cases")
    _require(isinstance(raw, list) and len(raw) == 5, f"{prefix}: expected five cases")
    cases: dict[str, dict] = {}
    role_to_case = (
        ROLE_TO_CASE
        if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
        else LEGACY_ROLE_TO_CASE
    )
    expected_names = tuple(role_to_case[role] for role in ROLE_ORDER)
    common_fields = {
        "name",
        "family",
        "kind",
        "pressure_ttl_ms",
        "cir_write_threshold_gbps",
        "min_interval_ms",
    }
    for index, case in enumerate(raw):
        _require(isinstance(case, dict), f"{prefix}: case {index} malformed")
        expected_fields = set(common_fields)
        if (
            shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
            and case.get("name") == ROLE_TO_CASE["A*"]
        ):
            expected_fields.update(
                {
                    "pfo_internal_deadband_gbps",
                    "forecast_hot_fraction",
                    "forecast_requests_per_npu",
                }
            )
        _require(
            set(case) == expected_fields,
            f"{prefix}: case {index} fields changed",
        )
        name = str(case["name"])
        _require(name not in cases, f"{prefix}: duplicate case {name}")
        cases[name] = case
        for field in (
            "pressure_ttl_ms",
            "cir_write_threshold_gbps",
            "min_interval_ms",
        ):
            _nonnegative(case[field], f"{prefix}: {name} {field}")
    _require(tuple(cases) == expected_names, f"{prefix}: case order/names changed")
    expected_fixed = {
        "baseline": ("baseline", "baseline", 0.0, 0.0, 0.0),
        "layer_once_ttl_0ms": ("ttl", "layer_once", 0.0, 0.0, 0.0),
        "adaptive_t0_i25ms": ("adaptive", "adaptive", 0.0, 0.0, 25.0),
    }
    for name, expected in expected_fixed.items():
        case = cases[name]
        actual = (
            case["family"],
            case["kind"],
            float(case["pressure_ttl_ms"]),
            float(case["cir_write_threshold_gbps"]),
            float(case["min_interval_ms"]),
        )
        _require(actual == expected, f"{prefix}: frozen case {name} changed")
    lstar = cases["layer_once_lstar"]
    _require(
        lstar["family"] == "ttl"
        and lstar["kind"] == "layer_once"
        and float(lstar["pressure_ttl_ms"]) >= 1.0
        and float(lstar["cir_write_threshold_gbps"]) == 0.0
        and float(lstar["min_interval_ms"]) == 0.0,
        f"{prefix}: invalid L*",
    )
    astar = cases[role_to_case["A*"]]
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        pfo_deadband = _nonnegative(
            astar["pfo_internal_deadband_gbps"], f"{prefix}: A* PFO deadband"
        )
        forecast_hot_fraction = _finite(
            astar["forecast_hot_fraction"], f"{prefix}: A* forecast hot fraction"
        )
        forecast_requests = _integer(
            astar["forecast_requests_per_npu"],
            f"{prefix}: A* forecast requests/NPU",
            minimum=1,
        )
        _require(
            astar["family"] == "pfo"
            and astar["kind"] == "pfo"
            and float(astar["pressure_ttl_ms"]) == 0.0
            and float(astar["cir_write_threshold_gbps"]) == 0.0
            and float(astar["min_interval_ms"]) >= 1.0
            and pfo_deadband <= 0.05
            and forecast_hot_fraction == 0.70
            and forecast_requests == 32,
            f"{prefix}: invalid frozen H70 A*",
        )
    else:
        _require(
            astar["family"] == "adaptive"
            and astar["kind"] == "adaptive"
            and float(astar["pressure_ttl_ms"]) == 0.0
            and 0.0 <= float(astar["cir_write_threshold_gbps"]) <= 0.05
            and float(astar["min_interval_ms"]) >= 1.0,
            f"{prefix}: invalid legacy A*",
        )
    return cases


def _validate_spec(
    spec, prefix: str, shard_schema_version: int
) -> tuple[dict[str, dict], int, int]:
    _require(isinstance(spec, dict), f"{prefix}: experiment_spec missing")
    expected_spec_fields = {
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
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        expected_spec_fields.update({"campaign_spec_sha256", "scale_semantics", "pfo"})
    _require(
        set(spec) == expected_spec_fields,
        f"{prefix}: experiment spec fields changed",
    )
    _require(
        spec.get("schema_version") == shard_schema_version,
        f"{prefix}: spec schema",
    )
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        _require(
            spec.get("campaign_spec_sha256") is None
            or _is_sha256(spec.get("campaign_spec_sha256")),
            f"{prefix}: spec campaign_spec_sha256 is malformed",
        )
    expected_experiment = (
        FORMAL_EXPERIMENT
        if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
        else LEGACY_EXPERIMENT
    )
    _require(
        spec.get("experiment") == expected_experiment,
        f"{prefix}: wrong experiment",
    )
    _require(spec.get("definition") == DEFINITION, f"{prefix}: wrong definition")
    _require(spec.get("num_npu") == NUM_NPU, f"{prefix}: expected 128 NPU")
    _require(spec.get("n_layers") == N_LAYERS, f"{prefix}: expected 16 layers")
    _require(spec.get("batch_size") == BATCH_SIZE, f"{prefix}: expected batch 1")
    _require(
        tuple(spec.get("default_ssu_list", ())) == FORMAL_SSUS, f"{prefix}: SSU grid"
    )
    cases = _validate_case_specs(spec, prefix, shard_schema_version)
    role_to_case = (
        ROLE_TO_CASE
        if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
        else LEGACY_ROLE_TO_CASE
    )
    _require(
        spec.get("report_roles") == role_to_case,
        f"{prefix}: report roles changed",
    )
    definition_hash = _canonical_hash(
        _expected_definition_dict(list(cases.values()), shard_schema_version),
        b"ms-scale-control-definition:v1\0",
    )
    _require(
        spec.get("definition_fingerprint") == definition_hash,
        f"{prefix}: definition hash",
    )
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        pfo = spec.get("pfo")
        astar = cases[ROLE_TO_CASE["A*"]]
        _require(
            pfo
            == {
                "controller": "ProtectedFloorSchemeBController",
                "materialized_allocation_stages": ["selected_protected_floor"],
                "request_pin": "stable (npu_id, request_id)",
                "target_ratio": 0.52,
                "required_ratio": 0.5,
                "background_reserve_fraction_for_admission_only": 0.05,
                "internal_deadband_gbps_by_case": {
                    ROLE_TO_CASE["A*"]: astar["pfo_internal_deadband_gbps"]
                },
                "downstream_cir_write_threshold_gbps": 0.0,
                "path_pressure_reads": 0,
                "trigger_ownership": "CIRControlConfig.min_interval_ms",
                "real_register_order": "all decreases before any increase",
            },
            f"{prefix}: frozen H70 PFO controller contract changed",
        )

    workload = spec.get("workload")
    _require(isinstance(workload, dict), f"{prefix}: workload spec missing")
    _require(
        set(workload)
        == {
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
        },
        f"{prefix}: workload spec fields changed",
    )
    _require(
        workload.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{prefix}: workload mode",
    )
    seed = _integer(workload.get("seed"), f"{prefix}: seed")
    _require(seed < 2**64, f"{prefix}: seed exceeds uint64")
    backing = _integer(
        workload.get("requests_per_npu"), f"{prefix}: backing", minimum=32
    )
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        _require(
            spec.get("scale_semantics")
            == {
                "num_npu": NUM_NPU,
                "naked_128_means": "128 NPU",
                "backing_requests_per_npu": backing,
                "total_assignment_count": NUM_NPU * backing,
                "rule": (
                    "NPU count is fixed by --definition; --requests-per-npu is "
                    "finite input backing only"
                ),
            },
            f"{prefix}: scale semantics confuse 128 NPU with finite backing",
        )
    for name in (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment_hash",
        "full_assignment_hash",
    ):
        _require(_is_sha256(workload.get(name)), f"{prefix}: invalid workload {name}")
    _require(
        workload["full_assignment_hash"] == workload["assignment"],
        f"{prefix}: assignment alias",
    )
    _require(
        workload.get("scientific_prefix_requests_per_npu") == 32,
        f"{prefix}: scientific prefix",
    )
    _require(
        workload.get("sampling")
        == "IID uniform with replacement over all 84 data profiles"
        and workload.get("per_npu_streams") == "independent and prefix-stable",
        f"{prefix}: workload sampling semantics changed",
    )
    _require(
        workload.get("backing_prefix_reason")
        == (
            f"{backing} requests preserve the first-32 assignment; any suffix "
            "only prevents full-load queue exhaustion during measurement/drain"
        ),
        f"{prefix}: finite backing semantics changed",
    )
    authentication = workload.get("authentication")
    _require(isinstance(authentication, dict), f"{prefix}: authentication missing")
    _require(
        set(authentication)
        == {
            "source",
            "source_sha256",
            "catalog_hash",
            "table_fingerprint",
            "profile_count",
        },
        f"{prefix}: authentication fields changed",
    )
    _require(
        authentication.get("source") == "data", f"{prefix}: unauthenticated source"
    )
    for name in ("source_sha256", "catalog_hash", "table_fingerprint"):
        _require(
            _is_sha256(authentication.get(name)), f"{prefix}: bad authentication {name}"
        )
    _require(
        authentication.get("catalog_hash") == workload["catalog"],
        f"{prefix}: catalog authentication",
    )
    _require(
        authentication.get("profile_count") == 84, f"{prefix}: expected 84 profiles"
    )

    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), f"{prefix}: steady config missing")
    _require(
        set(steady)
        == {
            "seed",
            "requests_per_npu",
            "warmup_requests_per_npu",
            "settle_ms",
            "measurement_ms",
            "block_ms",
            "slo_alpha",
        },
        f"{prefix}: steady config fields changed",
    )
    _require(steady.get("seed") == seed, f"{prefix}: steady/workload seed differs")
    _require(steady.get("requests_per_npu") == backing, f"{prefix}: backing differs")
    warmup = _integer(
        steady.get("warmup_requests_per_npu"), f"{prefix}: warmup", minimum=1
    )
    _require(warmup < backing, f"{prefix}: warmup exhausts backing")
    settle = _nonnegative(steady.get("settle_ms"), f"{prefix}: settle")
    measurement = _nonnegative(steady.get("measurement_ms"), f"{prefix}: measurement")
    block = _nonnegative(steady.get("block_ms"), f"{prefix}: block")
    _require(measurement > 0 and block > 0, f"{prefix}: invalid measurement/block")
    _close(
        steady.get("slo_alpha"),
        PRIMARY_ALPHA,
        f"{prefix}: primary alpha",
        abs_tol=1e-12,
    )
    max_interval = max(float(case["min_interval_ms"]) for case in cases.values())
    _require(
        settle + 1e-9 >= max(500.0, 5.0 * max_interval), f"{prefix}: settle too short"
    )

    adaptive = spec.get("adaptive")
    expected_adaptive = {
        "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
        "explicit_spill_threshold": 0.75,
        "target_ratio": 0.52,
        "required_ratio": 0.5,
        "background_reserve_fraction": 0.05,
        "ssd_cap_gbps": SSD_CAP_GBPS,
        "npu_cap_gbps": NPU_CAP_GBPS,
    }
    _require(adaptive == expected_adaptive, f"{prefix}: Adaptive constants changed")
    _require(
        spec.get("cross_request_layer0_prefetch") is True,
        f"{prefix}: prefetch disabled",
    )
    _require(
        spec.get("placement") == "token-block ring hash reused across all 16 layers",
        f"{prefix}: placement changed",
    )
    _require(
        spec.get("measurement_cost_scope")
        == "true SSU pressure-table reads and CIR writes",
        f"{prefix}: measurement cost scope changed",
    )
    _require(
        spec.get("pairing_scope")
        == (
            "all strategies share the exact finite schedule/placement/simulator "
            "input within each SSU; wall-time measurement cohorts may differ in "
            "closed loop and receive a separate membership fingerprint"
        ),
        f"{prefix}: pairing scope changed",
    )
    _require(
        spec.get("diagnostics")
        == {
            "control_evaluations": (
                "reported but not required to equal the threshold=0 closed-loop "
                "anchor because filtered CIR writes can change event timing"
            ),
            "source_stable_false": "invalidates the shard",
        },
        f"{prefix}: diagnostic semantics changed",
    )
    _require(
        isinstance(spec.get("source_files"), list)
        and spec["source_files"]
        and spec["source_files"][-1] == "data"
        and len(spec["source_files"]) == len(set(spec["source_files"])),
        f"{prefix}: source file list malformed",
    )
    return cases, seed, backing


def _recipe(seed: int, backing: int, catalog_hash: str) -> dict:
    return {
        "schema_version": 1,
        "mode": "iid_uniform_profile_catalog_v1",
        "seed": seed,
        "num_npu": NUM_NPU,
        "requests_per_npu": backing,
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


def _json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _json_array_digest() -> tuple[object, list[bytes]]:
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
    """Independently rebuild finite requests, placement, trace, and DES input."""

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
    for name in (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "mode",
        "seed",
        "requests_per_npu",
    ):
        _require(
            manifest.get(name) == workload.get(name),
            f"{prefix}: manifest {name} differs",
        )
    _require(manifest.get("num_npu") == NUM_NPU, f"{prefix}: manifest NPU count")
    _require(
        manifest.get("request_id_formula") == "sequence * num_npu + npu_id",
        f"{prefix}: request ID formula",
    )

    catalog_rows = manifest.get("catalog_rows")
    _require(
        isinstance(catalog_rows, list) and len(catalog_rows) == 84,
        f"{prefix}: catalog rows",
    )
    catalog: dict[tuple[int, int], tuple[float, ...]] = {}
    normalized_for_catalog_hash = []
    for index, row in enumerate(catalog_rows):
        _require(
            isinstance(row, list) and len(row) == 3, f"{prefix}: catalog row {index}"
        )
        key = (
            _integer(row[0], f"{prefix}: seq_len", minimum=1),
            _integer(row[1], f"{prefix}: nql", minimum=1),
        )
        _require(key not in catalog, f"{prefix}: duplicate profile {key}")
        values = tuple(
            _finite(value, f"{prefix}: profile {key} value") for value in row[2]
        )
        _require(
            len(values) == 4 and all(value > 0 for value in values),
            f"{prefix}: invalid profile {key}",
        )
        catalog[key] = values
        normalized_for_catalog_hash.append([list(key), list(values)])
    _require(list(catalog) == sorted(catalog), f"{prefix}: catalog not sorted")
    catalog_hash = _canonical_hash(
        normalized_for_catalog_hash, b"random-steady-state:data-catalog:v1\0"
    )
    table_hash = _canonical_hash(catalog_rows, b"authenticated-bw-table:v1\0")
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
        _recipe(int(workload["seed"]), int(workload["requests_per_npu"]), catalog_hash),
        b"random-steady-state:recipe:v1\0",
    )
    _require(recipe_hash == workload["recipe"], f"{prefix}: recipe hash mismatch")
    assignments = manifest.get("assignment_rows")
    expected_count = NUM_NPU * int(workload["requests_per_npu"])
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
            npu_id == request_id % NUM_NPU and sequence == request_id // NUM_NPU,
            f"{prefix}: request ID mapping {request_id}",
        )
        _require(row[3] in CATEGORIES, f"{prefix}: assignment category")
        _require(
            isinstance(row[4], list) and len(row[4]) == 2,
            f"{prefix}: assignment profile",
        )
        key = (int(row[4][0]), int(row[4][1]))
        _require(key in catalog, f"{prefix}: assignment profile absent")
        _require(
            row[3] == _classify_profile(key),
            f"{prefix}: assignment category/profile mismatch",
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
    max_layer_gb = max(values[3] for values in catalog.values())
    return {
        "catalog": catalog,
        "assignments": assignments,
        "max_single_request_layer_gb": max_layer_gb,
        "fleet_layer_burst_bound_gb": NUM_NPU * max_layer_gb,
    }


def _path_mapping() -> list[int]:
    return [(npu_id % 8) * 32 + 16 + (npu_id // 8) for npu_id in range(NUM_NPU)]


def _validate_path_abi(value, prefix: str) -> None:
    expected = {
        "path_count": 256,
        "group_count": 8,
        "paths_per_group": 32,
        "max_npu": 128,
        "assigned_count": 128,
        "assigned_unique": 128,
        "assigned_min": 16,
        "assigned_max": 255,
        "path_zero_reserved": True,
        "assigned_paths_sha256": _canonical_hash(
            _path_mapping(), b"ms-scale-control-path-abi:v1\0"
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


def _runtime_process_audit(runtime, prefix: str) -> dict:
    for field in ("pid", "rss_current_bytes", "rss_peak_bytes"):
        _require(field in runtime, f"{prefix}: process runtime {field} missing")
    pid = _integer(runtime["pid"], f"{prefix}: process pid", minimum=1)
    peak_rss = _integer(
        runtime["rss_peak_bytes"], f"{prefix}: process peak RSS", minimum=1
    )
    current_rss = runtime["rss_current_bytes"]
    if current_rss is not None:
        current_rss = _integer(current_rss, f"{prefix}: process current RSS", minimum=1)
        _require(
            peak_rss >= current_rss,
            f"{prefix}: process peak RSS is below current RSS",
        )
    return {
        "pid": pid,
        "rss_current_bytes": current_rss,
        "rss_peak_bytes": peak_rss,
        "peak_not_below_current": current_rss is None or peak_rss >= current_rss,
    }


def _runtime_scientific(runtime, prefix: str) -> dict:
    _require(isinstance(runtime, dict), f"{prefix}: runtime missing")
    _require(
        all(field in runtime for field in RUNTIME_SCIENTIFIC_FIELDS),
        f"{prefix}: runtime incomplete",
    )
    _require(
        all(
            isinstance(runtime[field], str) and bool(runtime[field])
            for field in (
                "hostname",
                "python",
                "python_full",
                "python_implementation",
                "numpy",
                "platform",
            )
        ),
        f"{prefix}: runtime identity malformed",
    )
    _require(
        runtime["python_implementation"] == "CPython",
        f"{prefix}: unsupported Python implementation",
    )
    _require(
        runtime["multiprocessing_start_method"] in ("spawn", "forkserver"),
        f"{prefix}: unsafe multiprocessing",
    )
    _integer(runtime["cpu_count"], f"{prefix}: runtime CPU count", minimum=1)
    _runtime_process_audit(runtime, prefix)
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


def _slo_metrics(request_rows: list[dict], alpha: float, prefix: str) -> dict:
    by_npu: dict[int, list[bool]] = defaultdict(list)
    outcomes = []
    for row in request_rows:
        outcome = (
            float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + SLO_EPSILON
        )
        by_npu[int(row["npu_id"])].append(outcome)
        outcomes.append(outcome)
    all_npus = set(by_npu) == set(range(NUM_NPU))
    equal = (
        statistics.fmean(statistics.fmean(by_npu[npu]) for npu in range(NUM_NPU))
        if all_npus
        else None
    )
    return {
        "all_npus_sampled": all_npus,
        "equal_npu": equal,
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


def _cohort_metrics(
    request_rows: list[dict],
    catalog: dict[tuple[int, int], tuple[float, ...]],
    alpha: float,
) -> dict:
    categories: dict[str, list[bool]] = defaultdict(list)
    profiles: dict[tuple[int, int], list[bool]] = defaultdict(list)
    bins: dict[str, list[bool]] = defaultdict(list)
    profiles_by_npu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in request_rows:
        key = tuple(row["profile_key"])
        outcome = (
            float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + SLO_EPSILON
        )
        demand = catalog[key][0]
        categories[str(row["category"])].append(outcome)
        profiles[key].append(outcome)
        bins[_demand_bin(demand)].append(outcome)
        profiles_by_npu[int(row["npu_id"])].append(key)
    per_npu_demand = []
    per_npu_ms_per_gb = []
    for npu_id in range(NUM_NPU):
        keys = profiles_by_npu[npu_id]
        _require(bool(keys), f"cohort: NPU{npu_id} has no requests")
        compute_s = math.fsum(catalog[key][1] for key in keys) / 1e6
        kv_gb = math.fsum(catalog[key][3] for key in keys)
        per_npu_demand.append(kv_gb / compute_s)
        per_npu_ms_per_gb.append(1000.0 * compute_s / kv_gb)
    demand_mean = statistics.fmean(per_npu_demand)
    ms_mean = statistics.fmean(per_npu_ms_per_gb)
    return {
        "category": {
            category: _group_metric(categories.get(category, ()))
            for category in CATEGORIES
            if categories.get(category)
        },
        "raw_demand_bins": {
            name: _group_metric(bins.get(name, ())) for name, _, _ in DEMAND_BINS
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


def _validate_requests(
    summary: dict,
    manifest_data: dict,
    seed: int,
    prefix: str,
) -> dict:
    rows = summary.get("request_rows")
    _require(isinstance(rows, list) and rows, f"{prefix}: request_rows missing")
    _require(
        summary.get("measurement_request_count") == len(rows),
        f"{prefix}: request count",
    )
    start = _finite(summary.get("measurement_start_ms"), f"{prefix}: window start")
    end = _finite(summary.get("measurement_end_ms"), f"{prefix}: window end")
    drain_stop = _finite(summary.get("drain_stop_ms"), f"{prefix}: drain stop")
    _require(drain_stop >= end, f"{prefix}: drain stops before measurement ends")
    catalog = manifest_data["catalog"]
    assignments = manifest_data["assignments"]
    ids = []
    by_npu = defaultdict(list)
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{prefix}: request row {index}")
        required_fields = {
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
        _require(set(row) == required_fields, f"{prefix}: request row fields changed")
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
            and manifest[0] == request_id
            and manifest[1] == npu_id
            and manifest[2] == sequence,
            f"{prefix}: request {request_id} ID mapping differs",
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
            f"{prefix}: request profile differs from manifest",
        )
        expected_category = _classify_profile(key)
        _require(
            row["category"] == manifest[3] == expected_category,
            f"{prefix}: request category",
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
        _require(
            completion <= drain_stop + 1e-8, f"{prefix}: completion after drain stop"
        )
        _require(
            float(row["ttft_ms"]) + 1e-8 >= ideal,
            f"{prefix}: TTFT is below the authenticated pure-compute lower bound",
        )
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
        min(counts) >= MIN_REQUESTS_PER_NPU,
        f"{prefix}: fewer than 8 requests on an NPU",
    )
    _require(
        summary.get("request_counts_by_npu") == counts,
        f"{prefix}: per-NPU request counts",
    )
    alpha2 = _slo_metrics(rows, PRIMARY_ALPHA, prefix)
    alpha15 = _slo_metrics(rows, SENSITIVITY_ALPHA, prefix)
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

    midpoint = start + (end - start) / 2.0
    first_rows = [row for row in rows if float(row["admission_time_ms"]) < midpoint]
    second_rows = [row for row in rows if float(row["admission_time_ms"]) >= midpoint]
    half_metrics = {
        "first_alpha2": _slo_metrics(first_rows, PRIMARY_ALPHA, prefix),
        "second_alpha2": _slo_metrics(second_rows, PRIMARY_ALPHA, prefix),
        "first_alpha15": _slo_metrics(first_rows, SENSITIVITY_ALPHA, prefix),
        "second_alpha15": _slo_metrics(second_rows, SENSITIVITY_ALPHA, prefix),
    }
    half_all_sampled = all(item["all_npus_sampled"] for item in half_metrics.values())
    alpha2_half_delta = (
        100.0
        * (
            half_metrics["second_alpha2"]["equal_npu"]
            - half_metrics["first_alpha2"]["equal_npu"]
        )
        if half_all_sampled
        else None
    )
    alpha15_half_delta = (
        100.0
        * (
            half_metrics["second_alpha15"]["equal_npu"]
            - half_metrics["first_alpha15"]["equal_npu"]
        )
        if half_all_sampled
        else None
    )
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
        "alpha2": alpha2,
        "alpha15": alpha15,
        "alpha2_half_delta_pp": alpha2_half_delta,
        "alpha15_half_delta_pp": alpha15_half_delta,
        "half_all_npus_sampled": half_all_sampled,
        "cohort_hash": cohort_hash,
        "cohort_alpha2": _cohort_metrics(rows, catalog, PRIMARY_ALPHA),
        "cohort_alpha15": _cohort_metrics(rows, catalog, SENSITIVITY_ALPHA),
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
    _require(requests == reports + cache_hits, f"{prefix}: reads+hits != requests")
    _require(
        request_by_ssu
        == [left + right for left, right in zip(report_by_ssu, cache_by_ssu)],
        f"{prefix}: per-SSU pressure requests",
    )
    _require(
        transactions <= writes and commits <= transactions and commits <= evaluations,
        f"{prefix}: control counter ordering",
    )
    for vector_field, counts in (
        ("measurement_pressure_report_rate_hz_by_ssu", report_by_ssu),
        ("measurement_cir_write_transaction_rate_hz_by_ssu", transaction_by_ssu),
        ("measurement_cir_path_write_rate_hz_by_ssu", write_by_ssu),
    ):
        rates = _vector(summary.get(vector_field), num_ssu, f"{prefix}: {vector_field}")
        for index, (rate, count) in enumerate(zip(rates, counts)):
            _close(rate, count / duration_s, f"{prefix}: {vector_field}[{index}]")
    _close(
        summary.get("measurement_pressure_cache_hit_fraction"),
        cache_hits / requests if requests else 0.0,
        f"{prefix}: cache hit fraction",
    )
    hit_fractions = _vector(
        summary.get("measurement_pressure_cache_hit_fraction_by_ssu"),
        num_ssu,
        f"{prefix}: cache hit fraction by SSU",
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
    ratios = _vector(
        summary.get("measurement_cir_entries_per_transaction_by_ssu"),
        num_ssu,
        f"{prefix}: entries/transaction by SSU",
    )
    for index, (ratio, entry_count, transaction_count) in enumerate(
        zip(ratios, write_by_ssu, transaction_by_ssu)
    ):
        _close(
            ratio,
            entry_count / transaction_count if transaction_count else 0.0,
            f"{prefix}: entries/transaction SSU{index}",
        )

    kind = case["kind"]
    if kind == "baseline":
        _require(
            requests == writes == transactions == commits == evaluations == 0,
            f"{prefix}: baseline control operation",
        )
        _require(
            summary.get("control_min_interval_ms") is None,
            f"{prefix}: baseline interval",
        )
    elif kind == "layer_once":
        _require(
            reports > 0
            and requests > 0
            and writes == transactions == commits == evaluations == 0,
            f"{prefix}: L control counters",
        )
        _require(
            summary.get("control_min_interval_ms") is None, f"{prefix}: L interval"
        )
    else:
        _require(requests == 0 and evaluations > 0, f"{prefix}: Adaptive counters")
        _close(
            summary.get("control_min_interval_ms"),
            case["min_interval_ms"],
            f"{prefix}: Adaptive interval",
        )
    return {
        "pressure_reads": reports,
        "pressure_read_rate_hz_per_ssu": reports / duration_s / num_ssu,
        "cir_entry_writes": writes,
        "cir_entry_write_rate_hz_per_ssu": writes / duration_s / num_ssu,
        "cir_transactions": transactions,
        "cir_transaction_rate_hz_per_ssu": transactions / duration_s / num_ssu,
        "control_evaluations": evaluations,
    }


def _compare_vectors(
    actual: Sequence, expected: Sequence, name: str, *, abs_tol=1e-8
) -> None:
    _require(len(actual) == len(expected), f"{name}: vector length differs")
    for index, (left, right) in enumerate(zip(actual, expected)):
        _close(left, right, f"{name}[{index}]", abs_tol=abs_tol)


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


def _stationarity_rule_results(
    *,
    half_all_npus_sampled: bool,
    util_half_delta_pp: float,
    util_projected_change_pp: float,
    served_half_relative_delta: float,
    persistent_queue_growth: bool,
) -> dict[str, bool]:
    return {
        "all_npus_sampled_in_both_halves": bool(half_all_npus_sampled),
        "utilization_half_delta_le_1pp": abs(util_half_delta_pp)
        <= UTIL_HALF_DELTA_LIMIT_PP + 1e-12,
        "utilization_projected_trend_le_2pp": abs(util_projected_change_pp)
        <= UTIL_TREND_PROJECTED_LIMIT_PP + 1e-12,
        "fleet_served_half_relative_delta_le_2pct": served_half_relative_delta
        <= SERVED_HALF_RELATIVE_LIMIT + 1e-12,
        "no_persistent_queue_growth_over_one_layer_burst": not persistent_queue_growth,
    }


def _validate_stationarity(
    summary: dict,
    diagnostic: dict,
    request_info: dict,
    manifest_data: dict,
    num_ssu: int,
    duration_ms: float,
    block_ms: float,
    prefix: str,
) -> dict:
    _require(
        summary.get("stationarity_boundary_semantics")
        == EXPECTED_STATIONARITY_SEMANTICS,
        f"{prefix}: stationarity semantics",
    )
    snapshots = summary.get("measurement_stationarity_boundaries")
    blocks = summary.get("measurement_blocks")
    _require(
        isinstance(blocks, list) and blocks, f"{prefix}: measurement blocks missing"
    )
    block_ratio = duration_ms / block_ms
    expected_blocks = int(round(block_ratio))
    _require(
        math.isclose(block_ratio, expected_blocks, rel_tol=0.0, abs_tol=1e-12)
        and expected_blocks >= MIN_STATIONARITY_BLOCKS
        and expected_blocks % 2 == 0,
        f"{prefix}: stationarity requires an even number of at least 16 full blocks",
    )
    _require(
        len(blocks) == expected_blocks, f"{prefix}: block count differs from config"
    )
    _require(
        isinstance(snapshots, list) and len(snapshots) == expected_blocks + 1,
        f"{prefix}: boundary count",
    )
    _require(
        summary.get("measurement_stationarity_boundary_count") == len(snapshots),
        f"{prefix}: reported boundary count",
    )
    snapshot_fields = {
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
    start_ms = float(summary["measurement_start_ms"])
    end_ms = float(summary["measurement_end_ms"])
    expected_times = [start_ms] + [
        end_ms if block + 1 == expected_blocks else start_ms + (block + 1) * block_ms
        for block in range(expected_blocks)
    ]
    parsed_snapshots = []
    cumulative_fields = (
        "ssd_cumulative_busy_ms_by_ssu",
        "ssd_cumulative_served_gb_by_ssu",
        "npu_compute_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_served_gb_by_npu",
    )
    for boundary, (snapshot, expected_time) in enumerate(
        zip(snapshots, expected_times)
    ):
        _require(
            isinstance(snapshot, dict) and set(snapshot) == snapshot_fields,
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
        parsed = {
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
            zip(parsed["ssd_busy"], parsed["ssd_served"])
        ):
            _close(
                served,
                busy * SSD_CAP_GBPS / 1000.0,
                f"{prefix}: cumulative SSD service {boundary}/{index}",
            )
        for index, (busy, served) in enumerate(
            zip(parsed["link_busy"], parsed["link_served"])
        ):
            _close(
                served,
                busy * NPU_CAP_GBPS / 1000.0,
                f"{prefix}: cumulative link service {boundary}/{index}",
            )
        parsed_snapshots.append(parsed)
    for previous, current in zip(parsed_snapshots, parsed_snapshots[1:]):
        for field in ("ssd_busy", "ssd_served", "compute", "link_busy", "link_served"):
            _require(
                all(
                    right + 1e-8 >= left
                    for left, right in zip(previous[field], current[field])
                ),
                f"{prefix}: {field} cumulative counter moved backwards",
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
        zip(blocks, parsed_snapshots, parsed_snapshots[1:])
    ):
        _require(isinstance(block, dict), f"{prefix}: block {block_index}")
        expected_start = expected_times[block_index]
        expected_end = expected_times[block_index + 1]
        block_duration = expected_end - expected_start
        _require(block.get("block") == block_index, f"{prefix}: block index")
        _close(
            block.get("start_ms"),
            expected_start,
            f"{prefix}: block start",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        _close(
            block.get("end_ms"),
            expected_end,
            f"{prefix}: block end",
            abs_tol=1e-9,
            rel_tol=0.0,
        )
        _close(block.get("duration_ms"), block_duration, f"{prefix}: block duration")
        deltas = {
            "ssd_busy": [
                right_value - left_value
                for left_value, right_value in zip(left["ssd_busy"], right["ssd_busy"])
            ],
            "ssd_served": [
                right_value - left_value
                for left_value, right_value in zip(
                    left["ssd_served"], right["ssd_served"]
                )
            ],
            "compute": [
                right_value - left_value
                for left_value, right_value in zip(left["compute"], right["compute"])
            ],
            "link_busy": [
                right_value - left_value
                for left_value, right_value in zip(
                    left["link_busy"], right["link_busy"]
                )
            ],
            "link_served": [
                right_value - left_value
                for left_value, right_value in zip(
                    left["link_served"], right["link_served"]
                )
            ],
        }
        for field, values in deltas.items():
            _require(
                all(value >= -1e-8 for value in values),
                f"{prefix}: negative block {field}",
            )
            for index, value in enumerate(values):
                block_resource_sums[field][index] += max(0.0, value)
        expected_fields = (
            ("ssd_busy_ms_by_ssu", "ssd_busy"),
            ("ssd_served_gb_by_ssu", "ssd_served"),
            ("compute_ms_by_npu", "compute"),
            ("npu_link_busy_ms_by_npu", "link_busy"),
            ("npu_link_served_gb_by_npu", "link_served"),
        )
        for block_field, delta_field in expected_fields:
            values = _vector(
                block.get(block_field),
                len(deltas[delta_field]),
                f"{prefix}: {block_field}",
            )
            _compare_vectors(
                values,
                deltas[delta_field],
                f"{prefix}: block {block_index} {block_field}",
            )
        ssd_utils = _vector(
            block.get("ssd_utilizations"),
            num_ssu,
            f"{prefix}: block SSD utils",
            fraction=True,
        )
        expected_ssd_utils = [
            max(0.0, value) / block_duration for value in deltas["ssd_busy"]
        ]
        _compare_vectors(ssd_utils, expected_ssd_utils, f"{prefix}: block SSD utils")
        _close(
            block.get("ssd_mean_utilization"),
            statistics.fmean(ssd_utils),
            f"{prefix}: block SSD mean",
        )
        npu_utils = _vector(
            block.get("npu_utilizations"),
            NUM_NPU,
            f"{prefix}: block NPU utils",
            fraction=True,
        )
        expected_npu_utils = [
            max(0.0, value) / block_duration for value in deltas["compute"]
        ]
        _compare_vectors(npu_utils, expected_npu_utils, f"{prefix}: block NPU utils")
        block_util = statistics.fmean(npu_utils)
        _close(
            block.get("npu_utilization"), block_util, f"{prefix}: block fleet NPU util"
        )
        stored_block_util = float(block["npu_utilization"])
        link_utils = _vector(
            block.get("npu_link_utilizations"),
            NUM_NPU,
            f"{prefix}: block link utils",
            fraction=True,
        )
        _compare_vectors(
            link_utils,
            [max(0.0, value) / block_duration for value in deltas["link_busy"]],
            f"{prefix}: block link utils",
        )
        _close(
            block.get("npu_link_mean_utilization"),
            statistics.fmean(link_utils),
            f"{prefix}: block link mean",
        )
        edge_fields = (
            ("ssd_outstanding_blocks_at_start", left["ssd_blocks"], True),
            ("ssd_outstanding_blocks_at_end", right["ssd_blocks"], True),
            ("ssd_outstanding_gb_at_start", left["ssd_gb"], False),
            ("ssd_outstanding_gb_at_end", right["ssd_gb"], False),
            ("npu_link_outstanding_blocks_at_start", left["link_blocks"], True),
            ("npu_link_outstanding_blocks_at_end", right["link_blocks"], True),
            ("npu_link_outstanding_gb_at_start", left["link_gb"], False),
            ("npu_link_outstanding_gb_at_end", right["link_gb"], False),
        )
        for field, expected, integer_values in edge_fields:
            values = _vector(
                block.get(field),
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
            expected_delta = [
                right_value - left_value
                for left_value, right_value in zip(left_values, right_values)
            ]
            raw_delta = block.get(f"{base}_delta")
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
            block.get("request_count") == len(admitted),
            f"{prefix}: block request count",
        )
        expected_block_slo = (
            statistics.fmean(bool(row["slo_met"]) for row in admitted)
            if admitted
            else None
        )
        if expected_block_slo is None:
            _require(
                block.get("request_weighted_slo_attainment") is None,
                f"{prefix}: empty block SLO",
            )
        else:
            _close(
                block.get("request_weighted_slo_attainment"),
                expected_block_slo,
                f"{prefix}: block SLO",
            )
        block_utils.append(stored_block_util)
        block_request_counts.append(len(admitted))
        block_fleet_served_gb.append(math.fsum(deltas["ssd_served"]))
    _require(
        sum(block_request_counts) == len(request_rows),
        f"{prefix}: blocks do not cover cohort",
    )

    whole_fields = (
        ("measurement_ssd_busy_ms_by_ssu", "ssd_busy"),
        ("measurement_ssd_served_gb_by_ssu", "ssd_served"),
        ("compute_ms_by_npu", "compute"),
        ("measurement_npu_link_busy_ms_by_npu", "link_busy"),
    )
    for summary_field, total_field in whole_fields:
        values = _vector(
            summary.get(summary_field),
            len(block_resource_sums[total_field]),
            f"{prefix}: {summary_field}",
        )
        _compare_vectors(
            values,
            block_resource_sums[total_field],
            f"{prefix}: {summary_field} block sum",
        )
    link_matrix = _matrix(
        summary.get("measurement_npu_ssu_link_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{prefix}: stationarity link service matrix",
    )
    _compare_vectors(
        [math.fsum(row) for row in link_matrix],
        block_resource_sums["link_served"],
        f"{prefix}: block link service sum",
    )
    first, last = parsed_snapshots[0], parsed_snapshots[-1]
    summary_edge_fields = (
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
    )
    for field, expected, integer_values in summary_edge_fields:
        values = _vector(
            summary.get(field),
            len(expected),
            f"{prefix}: {field}",
            integer=integer_values,
        )
        if integer_values:
            _require(values == expected, f"{prefix}: {field}")
        else:
            _compare_vectors(values, expected, f"{prefix}: {field}")
        drift_field = field.replace("_at_start", "_drift")
        if field.endswith("_at_start"):
            end_field = field.replace("_at_start", "_at_end")
            expected_drift = [
                right - left for left, right in zip(expected, summary[end_field])
            ]
            raw_drift = summary.get(drift_field)
            _require(
                isinstance(raw_drift, list) and len(raw_drift) == len(expected_drift),
                f"{prefix}: {drift_field}",
            )
            if integer_values:
                _require(raw_drift == expected_drift, f"{prefix}: {drift_field}")
            else:
                _compare_vectors(raw_drift, expected_drift, f"{prefix}: {drift_field}")

    _require(
        isinstance(diagnostic, dict), f"{prefix}: stationarity_diagnostics missing"
    )
    _require(
        diagnostic.get("block_npu_utilizations") == block_utils,
        f"{prefix}: diagnostic block utils",
    )
    _require(
        diagnostic.get("block_request_counts") == block_request_counts,
        f"{prefix}: diagnostic block counts",
    )
    _close(
        diagnostic.get("block_utilization_range"),
        max(block_utils) - min(block_utils),
        f"{prefix}: diagnostic util range",
    )
    _close(
        diagnostic.get("first_last_utilization_delta"),
        block_utils[-1] - block_utils[0],
        f"{prefix}: diagnostic first/last util",
    )
    block_drift = [
        right - left for left, right in zip(first["ssd_blocks"], last["ssd_blocks"])
    ]
    _require(
        diagnostic.get("outstanding_blocks_drift_by_ssu") == block_drift,
        f"{prefix}: diagnostic queue drift",
    )
    _require(
        diagnostic.get("fleet_outstanding_blocks_drift") == sum(block_drift),
        f"{prefix}: diagnostic fleet drift",
    )

    times = [snapshot["time_ms"] for snapshot in parsed_snapshots]
    per_ssu_queue_series = [
        [snapshot["ssd_gb"][ssu] for snapshot in parsed_snapshots]
        for ssu in range(num_ssu)
    ]
    fleet_queue_series = [
        math.fsum(snapshot["ssd_gb"]) for snapshot in parsed_snapshots
    ]
    burst_bound = float(manifest_data["fleet_layer_burst_bound_gb"])

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
        half_boundary = len(values) // 2
        return {
            "start_gb": values[0],
            "end_gb": values[-1],
            "net_growth_gb": net_growth,
            "first_half_median_gb": statistics.median(values[: half_boundary + 1]),
            "second_half_median_gb": statistics.median(values[half_boundary:]),
            "theil_sen_slope_gbps": slope,
            "nondecreasing_step_fraction": nondecreasing_fraction,
            "max_positive_boundary_burst_gb": max([0.0, *deltas]),
            "persistent_growth_over_one_layer_burst": persistent_growth,
        }

    fleet_queue = queue_diagnostics(fleet_queue_series)
    per_ssu_queue = [queue_diagnostics(values) for values in per_ssu_queue_series]
    per_ssu_slopes = [item["theil_sen_slope_gbps"] for item in per_ssu_queue]
    fleet_slope = float(fleet_queue["theil_sen_slope_gbps"])
    fleet_normalized = fleet_slope / (SSD_CAP_GBPS * num_ssu)
    max_abs_per_ssu_normalized = (
        max(abs(value) for value in per_ssu_slopes) / SSD_CAP_GBPS
    )

    half = len(block_utils) // 2
    util_first_half = statistics.fmean(block_utils[:half])
    util_second_half = statistics.fmean(block_utils[half:])
    util_half_delta_pp = 100.0 * (util_second_half - util_first_half)
    block_midpoints = [
        start_ms + (index + 0.5) * block_ms for index in range(expected_blocks)
    ]
    util_slope_per_s = _theil_sen_slope(block_midpoints, block_utils)
    util_projected_change_pp = 100.0 * util_slope_per_s * (duration_ms / 1000.0)
    first_served_rate = math.fsum(block_fleet_served_gb[:half]) / (
        half * block_ms / 1000.0
    )
    second_served_rate = math.fsum(block_fleet_served_gb[half:]) / (
        half * block_ms / 1000.0
    )
    served_midpoint = (first_served_rate + second_served_rate) / 2.0
    served_half_relative_delta = (
        abs(second_served_rate - first_served_rate) / served_midpoint
        if served_midpoint > 0.0
        else 0.0
    )
    persistent_growth = fleet_queue["persistent_growth_over_one_layer_burst"] or any(
        item["persistent_growth_over_one_layer_burst"] for item in per_ssu_queue
    )
    stationarity_rules = _stationarity_rule_results(
        half_all_npus_sampled=request_info["half_all_npus_sampled"],
        util_half_delta_pp=util_half_delta_pp,
        util_projected_change_pp=util_projected_change_pp,
        served_half_relative_delta=served_half_relative_delta,
        persistent_queue_growth=persistent_growth,
    )
    queue_bounded = not persistent_growth
    temporal_quality = all(
        stationarity_rules[name]
        for name in (
            "all_npus_sampled_in_both_halves",
            "utilization_half_delta_le_1pp",
            "utilization_projected_trend_le_2pp",
            "fleet_served_half_relative_delta_le_2pct",
        )
    )
    if persistent_growth:
        queue_regime = "overloaded_growing"
    elif not temporal_quality:
        queue_regime = "window_nonstationary"
    else:
        queue_regime = "bounded"
    return {
        "stationarity_rule_version": STATIONARITY_RULE_VERSION,
        "block_count": len(blocks),
        "boundary_count": len(parsed_snapshots),
        "utilization_first_half": util_first_half,
        "utilization_second_half": util_second_half,
        "util_half_delta_pp": util_half_delta_pp,
        "utilization_theil_sen_slope_fraction_per_s": util_slope_per_s,
        "utilization_projected_change_pp": util_projected_change_pp,
        "fleet_served_first_half_gbps": first_served_rate,
        "fleet_served_second_half_gbps": second_served_rate,
        "fleet_served_half_relative_delta": served_half_relative_delta,
        "alpha2_slo_half_delta_pp": request_info["alpha2_half_delta_pp"],
        "alpha15_slo_half_delta_pp": request_info["alpha15_half_delta_pp"],
        "half_all_npus_sampled": request_info["half_all_npus_sampled"],
        "queue_burst_bound_derivation": {
            "method": "128 NPUs multiplied by the maximum authenticated catalog per-layer KV GB; this conservatively permits every NPU's one layer on one SSU",
            "num_npu": NUM_NPU,
            "max_authenticated_profile_per_layer_gb": manifest_data[
                "max_single_request_layer_gb"
            ],
            "fleet_one_layer_burst_bound_gb": burst_bound,
            "persistent_definition": "Theil-Sen slope > 0, at least 75% nondecreasing boundary steps, and net growth exceeds the independent one-layer burst bound",
        },
        "fleet_queue": fleet_queue,
        "per_ssu_queue": per_ssu_queue,
        "stationarity_rules": stationarity_rules,
        "stationarity_gate_passed": all(stationarity_rules.values()),
        "fleet_queue_slope_gbps": fleet_slope,
        "fleet_queue_slope_capacity_fraction": fleet_normalized,
        "max_abs_per_ssu_queue_slope_gbps": max(abs(value) for value in per_ssu_slopes),
        "max_abs_per_ssu_queue_slope_capacity_fraction": max_abs_per_ssu_normalized,
        "queue_regime": queue_regime,
        "queue_bounded": queue_bounded,
        "temporal_quality_stable": temporal_quality,
        "steady_state_qualified": all(stationarity_rules.values()),
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
        values = _vector(actual_npu, NUM_NPU, f"{prefix}: max actual NPU CIR")
        _require(
            max(values) <= NPU_CAP_GBPS + 1e-9, f"{prefix}: actual NPU CIR exceeds 50"
        )
    for field, length, cap in (
        ("actual_cir_sum_gbps_by_ssu_at_stop", num_ssu, SSD_CAP_GBPS),
        ("measurement_actual_cir_sum_gbps_by_ssu_at_start", num_ssu, SSD_CAP_GBPS),
        ("measurement_actual_cir_sum_gbps_by_ssu_at_end", num_ssu, SSD_CAP_GBPS),
    ):
        values = _vector(summary.get(field), length, f"{prefix}: {field}")
        _require(max(values) <= cap + 1e-9, f"{prefix}: {field} cap")
    for field in (
        "actual_npu_cir_sum_gbps_at_stop",
        "measurement_actual_npu_cir_sum_gbps_at_start",
        "measurement_actual_npu_cir_sum_gbps_at_end",
    ):
        values = summary.get(field)
        if applicable:
            parsed = _vector(values, NUM_NPU, f"{prefix}: {field}")
            _require(max(parsed) <= NPU_CAP_GBPS + 1e-9, f"{prefix}: {field} cap")
        else:
            _require(values is None, f"{prefix}: {field} should be null")
    return {
        "mean_npu_utilization_pct": 100.0 * statistics.fmean(npu_utils),
        "mean_ssd_utilization_pct": 100.0 * statistics.fmean(ssd_utils),
        "max_ssd_utilization_pct": 100.0 * max(ssd_utils),
        "mean_npu_link_utilization_pct": 100.0 * statistics.fmean(link_utils),
        "max_actual_ssu_cir_gbps": max(actual_ssu),
        "max_actual_npu_cir_gbps": max(actual_npu) if actual_npu is not None else None,
    }


def _expected_assignment_statistics(
    assignments: list[list], catalog: dict[tuple[int, int], tuple[float, ...]]
) -> dict:
    profile_counts = Counter((row[4][0], row[4][1]) for row in assignments)
    category_counts = Counter(str(row[3]) for row in assignments)
    by_npu: dict[int, list[tuple[int, int]]] = defaultdict(list)
    categories_by_npu: dict[int, list[str]] = defaultdict(list)
    for row in assignments:
        by_npu[int(row[1])].append((row[4][0], row[4][1]))
        categories_by_npu[int(row[1])].append(str(row[3]))
    per_npu_compute_s = []
    per_npu_kv_gb = []
    per_npu_demands = []
    per_npu_ms_per_gb = []
    for npu in range(NUM_NPU):
        keys = by_npu[npu]
        compute_s = math.fsum(catalog[key][1] for key in keys) / 1e6
        kv_gb = math.fsum(catalog[key][3] for key in keys)
        per_npu_compute_s.append(compute_s)
        per_npu_kv_gb.append(kv_gb)
        per_npu_demands.append(kv_gb / compute_s)
        per_npu_ms_per_gb.append(1000.0 * compute_s / kv_gb)
    category_ranges = {}
    for category in CATEGORIES:
        counts = [
            sum(value == category for value in categories_by_npu[npu])
            for npu in range(NUM_NPU)
        ]
        category_ranges[category] = {
            "min": min(counts),
            "mean": statistics.fmean(counts),
            "max": max(counts),
        }
    required_bw = [catalog[(row[4][0], row[4][1])][0] for row in assignments]
    return {
        "profile_counts": profile_counts,
        "category_counts": category_counts,
        "category_ranges": category_ranges,
        "per_npu_compute_s": per_npu_compute_s,
        "per_npu_kv_gb": per_npu_kv_gb,
        "per_npu_demands": per_npu_demands,
        "per_npu_ms_per_gb": per_npu_ms_per_gb,
        "fleet_demand": math.fsum(per_npu_demands),
        "required_bw_profile_bins": {
            "le_50": sum(value <= NPU_CAP_GBPS for value in required_bw),
            "gt_50_le_100": sum(
                NPU_CAP_GBPS < value <= 2.0 * NPU_CAP_GBPS for value in required_bw
            ),
            "gt_100": sum(value > 2.0 * NPU_CAP_GBPS for value in required_bw),
        },
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
        stats.get("prefix_hash_requests_per_npu") == 32,
        f"{prefix}: prefix length metadata",
    )
    _require(
        stats.get("profile_sampling") == "iid_uniform_profile_catalog_v1",
        f"{prefix}: profile sampling",
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
    _require(
        stats.get("catalog_profile_count") == 84, f"{prefix}: catalog profile count"
    )
    starts = _vector(
        stats.get("initial_npu_start_ms"), NUM_NPU, f"{prefix}: initial NPU starts"
    )
    _require(max(starts) <= 5.0 + 1e-12, f"{prefix}: initial jitter exceeds 5 ms")
    _compare_vectors(
        starts,
        materialized["initial_npu_start_ms"],
        f"{prefix}: independently rebuilt initial jitter",
    )
    expected = _expected_assignment_statistics(assignments, catalog)
    expected_profile_counts = {
        f"{key[0]},{key[1]}": count
        for key, count in sorted(expected["profile_counts"].items())
    }
    expected_profile_counts_all = {
        f"{key[0]},{key[1]}": expected["profile_counts"].get(key, 0)
        for key in sorted(catalog)
    }
    _require(
        stats.get("fleet_profile_counts") == expected_profile_counts,
        f"{prefix}: profile counts",
    )
    _require(
        stats.get("fleet_profile_counts_all") == expected_profile_counts_all,
        f"{prefix}: all profile counts",
    )
    _require(
        stats.get("profiles_used") == len(expected_profile_counts),
        f"{prefix}: profiles used",
    )
    expected_categories = {
        category: expected["category_counts"].get(category, 0)
        for category in CATEGORIES
    }
    expected_present_categories = {
        category: count for category, count in expected_categories.items() if count
    }
    _require(
        stats.get("fleet_category_counts") == expected_present_categories,
        f"{prefix}: present category counts",
    )
    _require(
        stats.get("fleet_category_counts_all") == expected_categories,
        f"{prefix}: category counts",
    )
    _require(
        _equivalent_derived(
            stats.get("per_npu_category_count_ranges"), expected["category_ranges"]
        ),
        f"{prefix}: per-NPU category ranges",
    )
    for field, values in (
        ("per_npu_compute_s_range", expected["per_npu_compute_s"]),
        ("per_npu_kv_gb_range", expected["per_npu_kv_gb"]),
        ("per_npu_demand_gbps_range", expected["per_npu_demands"]),
    ):
        _compare_vectors(
            _vector(stats.get(field), 2, f"{prefix}: {field}"),
            [min(values), max(values)],
            f"{prefix}: {field}",
        )
    demand_values = expected["per_npu_demands"]
    ms_values = expected["per_npu_ms_per_gb"]
    demand_meta = stats.get("per_npu_raw_demand_gbps")
    ms_meta = stats.get("per_npu_ms_per_gb")
    _require(
        isinstance(demand_meta, dict) and isinstance(ms_meta, dict),
        f"{prefix}: per-NPU demand metadata",
    )
    for field, value in (
        ("min", min(demand_values)),
        ("max", max(demand_values)),
        ("mean", statistics.fmean(demand_values)),
    ):
        _close(demand_meta.get(field), value, f"{prefix}: demand {field}")
    demand_mean = statistics.fmean(demand_values)
    _close(
        stats.get("per_npu_demand_gbps_mean"),
        demand_mean,
        f"{prefix}: demand mean alias",
    )
    demand_cv = statistics.pstdev(demand_values) / demand_mean
    _close(
        demand_meta.get("coefficient_of_variation"),
        demand_cv,
        f"{prefix}: demand CV",
    )
    _close(
        stats.get("per_npu_demand_gbps_cv"),
        demand_cv,
        f"{prefix}: demand CV alias",
    )
    for field, value in (
        ("min", min(ms_values)),
        ("max", max(ms_values)),
        ("mean", statistics.fmean(ms_values)),
        ("spread", max(ms_values) - min(ms_values)),
    ):
        _close(ms_meta.get(field), value, f"{prefix}: ms/GB {field}")
    _close(
        ms_meta.get("spread_pct_of_mean"),
        100.0 * (max(ms_values) - min(ms_values)) / statistics.fmean(ms_values),
        f"{prefix}: ms/GB spread percent",
    )
    _require(
        stats.get("required_bw_profile_bins") == expected["required_bw_profile_bins"],
        f"{prefix}: required bandwidth bins",
    )
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


def _validate_row(
    row: dict,
    *,
    cases: dict[str, dict],
    seed: int,
    backing: int,
    num_ssu: int,
    source: str,
    config: str,
    definition_fingerprint: str,
    spec: dict,
    manifest_data: dict,
    root_runtime: dict,
    shard_path: str,
) -> dict:
    _require(isinstance(row, dict), f"{shard_path}: row malformed")
    case_name = str(row.get("case"))
    _require(case_name in cases, f"{shard_path}: unknown case {case_name}")
    case = cases[case_name]
    role = CASE_TO_ROLE[case_name]
    prefix = f"{shard_path}: seed{seed}/SSU{num_ssu}/{role}"
    _require(row.get("status") == "ok", f"{prefix}: row status")
    _require(
        row.get("family") == case["family"] and row.get("kind") == case["kind"],
        f"{prefix}: family/kind",
    )
    _require(row.get("role") == role, f"{prefix}: role")
    _require(
        row.get("num_ssu") == num_ssu and row.get("num_npu") == NUM_NPU,
        f"{prefix}: topology",
    )
    _require(
        row.get("definition") == DEFINITION
        and row.get("definition_fingerprint") == definition_fingerprint,
        f"{prefix}: definition",
    )
    _require(row.get("case_spec") == case, f"{prefix}: case spec")
    _require(
        row.get("source_fingerprint") == source
        and row.get("config_fingerprint") == config,
        f"{prefix}: row provenance",
    )
    _require(
        row.get("case_fingerprint") == _case_fingerprint(case, num_ssu, source, config),
        f"{prefix}: case fingerprint",
    )
    inputs = row.get("input_fingerprints")
    input_fields = {
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
    _require(
        isinstance(inputs, dict)
        and set(inputs) == input_fields
        and all(_is_sha256(value) for value in inputs.values()),
        f"{prefix}: input fingerprints",
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
        all(inputs[name] == value for name, value in expected_inputs.items()),
        f"{prefix}: schedule input fingerprint",
    )
    full_materialized = _materialize_inputs(
        assignments=manifest_data["assignments"],
        assignment_hash=workload["assignment"],
        catalog=manifest_data["catalog"],
        seed=seed,
        num_ssu=num_ssu,
        requests_per_npu=backing,
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
        _recipe(seed, 32, workload["catalog"]),
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
    raw_row_runtime = row.get("runtime")
    row_runtime = _runtime_scientific(raw_row_runtime, prefix)
    worker_process_runtime = _runtime_process_audit(raw_row_runtime, prefix)
    _require(row_runtime == root_runtime, f"{prefix}: worker/root runtime differs")
    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{prefix}: steady summary missing")
    _require(
        summary.get("schema_version") == SUMMARY_SCHEMA_VERSION,
        f"{prefix}: summary schema",
    )
    _require(summary.get("mode") == "steady_state_full_load", f"{prefix}: mode")
    _require(
        summary.get("num_npu") == NUM_NPU and summary.get("num_ssu") == num_ssu,
        f"{prefix}: summary topology",
    )
    _require(
        summary.get("n_layers") == N_LAYERS and summary.get("batch_size") == BATCH_SIZE,
        f"{prefix}: summary model",
    )
    _require(
        summary.get("input_fingerprint") == inputs["simulator"],
        f"{prefix}: simulator fingerprint",
    )
    steady = spec["steady_state"]
    _require(
        summary.get("warmup_requests_per_npu") == steady["warmup_requests_per_npu"],
        f"{prefix}: warmup config",
    )
    _close(summary.get("settle_ms"), steady["settle_ms"], f"{prefix}: settle")
    _close(
        summary.get("measurement_duration_ms"),
        steady["measurement_ms"],
        f"{prefix}: measurement duration",
    )
    _close(summary.get("slo_alpha"), PRIMARY_ALPHA, f"{prefix}: SLO alpha")
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
    _close(end - start, steady["measurement_ms"], f"{prefix}: window width")
    _close(summary.get("tail_drain_ms"), drain - end, f"{prefix}: tail drain")
    _require(
        summary.get("measurement_control_counter_window") == EXPECTED_CONTROL_WINDOW,
        f"{prefix}: counter window",
    )
    invariants = summary.get("invariants")
    _require(
        isinstance(invariants, dict) and set(invariants) == EXPECTED_INVARIANTS,
        f"{prefix}: invariant catalog differs from the frozen 29-check schema",
    )
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{prefix}: failed invariants {failed}")

    request_info = _validate_requests(summary, manifest_data, seed, prefix)
    _require(
        row.get("measurement_cohort_fingerprint") == request_info["cohort_hash"],
        f"{prefix}: cohort fingerprint",
    )
    _require(
        _equivalent_derived(
            row.get("cohort_profile_metrics"), request_info["cohort_alpha2"]
        ),
        f"{prefix}: cohort metrics do not match request rows",
    )
    controls = _validate_control_metrics(
        summary, case, num_ssu, float(steady["measurement_ms"]), prefix
    )
    resources = _validate_resources(
        summary, num_ssu, float(steady["measurement_ms"]), prefix
    )
    stationarity = _validate_stationarity(
        summary,
        row.get("stationarity_diagnostics"),
        request_info,
        manifest_data,
        num_ssu,
        float(steady["measurement_ms"]),
        float(steady["block_ms"]),
        prefix,
    )

    full_stats = _validate_workload_statistics(
        row.get("workload_statistics"),
        assignments=manifest_data["assignments"],
        catalog=manifest_data["catalog"],
        expected_requests_per_npu=backing,
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
        and set(prefix_materialized) == {"workload", "placement", "trace"}
        and all(_is_sha256(value) for value in prefix_materialized.values()),
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
    _require(max(completed) <= backing, f"{prefix}: completed beyond backing")
    _require(
        summary.get("all_input_requests_completed") is False,
        f"{prefix}: finite backing was exhausted before drain stop",
    )
    tagged_completed_lower_bound = [0] * NUM_NPU
    for request in request_info["rows"]:
        npu_id = int(request["npu_id"])
        tagged_completed_lower_bound[npu_id] = max(
            tagged_completed_lower_bound[npu_id], int(request["sequence"]) + 1
        )
    _require(
        all(
            observed >= lower_bound
            for observed, lower_bound in zip(completed, tagged_completed_lower_bound)
        ),
        f"{prefix}: completed_by_npu_at_stop is below a completed tagged sequence",
    )
    backing_margin = backing - max(completed)
    _require(
        backing_margin >= MIN_BACKING_MARGIN,
        f"{prefix}: backing margin {backing_margin} < {MIN_BACKING_MARGIN}",
    )
    _require(
        _finite(row.get("wall_time_s", 0.0), f"{prefix}: wall time") >= 0.0,
        f"{prefix}: wall time negative",
    )
    return {
        "seed": seed,
        "num_npu": NUM_NPU,
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
        "measurement_request_count": len(request_info["rows"]),
        "backing_requests_per_npu": backing,
        "requests_per_npu_min": min(request_info["counts"]),
        "requests_per_npu_median": statistics.median(request_info["counts"]),
        "requests_per_npu_max": max(request_info["counts"]),
        "measurement_cohort_fingerprint": request_info["cohort_hash"],
        "backing_margin_fastest_npu": backing_margin,
        "worker_peak_rss_bytes": worker_process_runtime["rss_peak_bytes"],
        "wall_time_s": float(row.get("wall_time_s", 0.0)),
        "input_fingerprints": inputs,
        "prefix_materialized_fingerprints": prefix_materialized,
        "workload_statistics": row["workload_statistics"],
        "prefix_workload_statistics": row["prefix_32_workload_statistics"],
        "cohort_alpha2": request_info["cohort_alpha2"],
        "cohort_alpha15": request_info["cohort_alpha15"],
        **controls,
        **resources,
        **stationarity,
        **{f"prefix_{name}": value for name, value in prefix_stats.items()},
        **{f"backing_{name}": value for name, value in full_stats.items()},
    }


def _read_shard(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except OSError as error:
        raise ValidationError(f"cannot read {resolved}: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValidationError(f"invalid JSON in {resolved}: {error}") from error
    _require(isinstance(payload, dict), f"{resolved}: root must be an object")
    payload["_analysis_path"] = str(resolved)
    payload["_analysis_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def _campaign_spec_binding_audit(
    payloads: Sequence[dict], formal_spec: dict | None, *, required: bool
) -> dict:
    """Check the raw campaign-spec file hash embedded in every input shard.

    Partial/calibration artifacts from the older schema may omit the field, but
    omission or mismatch can never satisfy formal/report-ready gates.  In
    strict publication mode this runs before expensive input reconstruction.
    """

    expected_sha256 = formal_spec.get("file_sha256") if formal_spec else None
    expected_size_bytes = formal_spec.get("file_size_bytes") if formal_spec else None
    _require(
        expected_sha256 is None or _is_sha256(expected_sha256),
        "external campaign spec raw-byte SHA-256 is malformed",
    )
    entries = []
    for payload in payloads:
        path = str(payload.get("_analysis_path", "shard"))
        schema_version = payload.get("schema_version")
        embedded = payload.get("campaign_spec_sha256")
        spec = payload.get("experiment_spec")
        spec_embedded = (
            spec.get("campaign_spec_sha256") if isinstance(spec, dict) else None
        )
        workload = spec.get("workload") if isinstance(spec, dict) else None
        steady_state = spec.get("steady_state") if isinstance(spec, dict) else None
        schedule = payload.get("schedule_metadata")
        backing = payload.get("backing_requests_per_npu")
        topology_and_backing_consistent = bool(
            payload.get("num_npu") == NUM_NPU
            and isinstance(spec, dict)
            and spec.get("num_npu") == NUM_NPU
            and isinstance(backing, int)
            and not isinstance(backing, bool)
            and backing >= 32
            and payload.get("total_assignment_count") == NUM_NPU * backing
            and isinstance(workload, dict)
            and workload.get("requests_per_npu") == backing
            and isinstance(steady_state, dict)
            and steady_state.get("requests_per_npu") == backing
            and isinstance(schedule, dict)
            and schedule.get("num_npu") == NUM_NPU
            and schedule.get("requests_per_npu") == backing
            and formal_spec is not None
            and formal_spec.get("num_npu") == NUM_NPU
            and formal_spec.get("backing_requests_per_npu") == backing
        )
        rows = payload.get("results")
        row_embedded = (
            [row.get("campaign_spec_sha256") for row in rows]
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
            else None
        )
        row_scale_consistent = bool(
            isinstance(rows, list)
            and bool(rows)
            and all(
                isinstance(row, dict)
                and row.get("num_npu") == NUM_NPU
                and row.get("backing_requests_per_npu") == backing
                for row in rows
            )
        )
        authentication = payload.get("campaign_spec_authentication")
        ending_authentication = payload.get("ending_campaign_spec_authentication")
        expected_authentication = (
            {"sha256": expected_sha256, "size_bytes": expected_size_bytes}
            if expected_sha256 is not None
            else None
        )
        present = embedded is not None
        well_formed = _is_sha256(embedded)
        all_layers_match = bool(
            schema_version == FORMAL_SHARD_SCHEMA_VERSION
            and well_formed
            and expected_sha256 is not None
            and embedded == expected_sha256
            and spec_embedded == expected_sha256
            and isinstance(row_embedded, list)
            and bool(row_embedded)
            and all(value == expected_sha256 for value in row_embedded)
            and authentication == expected_authentication
            and ending_authentication == expected_authentication
            and payload.get("campaign_spec_stable_during_run") is True
            and topology_and_backing_consistent
            and row_scale_consistent
        )
        entries.append(
            {
                "path": path,
                "shard_schema_version": schema_version,
                "embedded_campaign_spec_sha256": embedded,
                "spec_campaign_spec_sha256": spec_embedded,
                "row_campaign_spec_sha256_values": row_embedded,
                "campaign_spec_authentication": authentication,
                "ending_campaign_spec_authentication": ending_authentication,
                "campaign_spec_stable_during_run": payload.get(
                    "campaign_spec_stable_during_run"
                ),
                "num_npu_is_exactly_128_and_backing_is_independent": (
                    topology_and_backing_consistent and row_scale_consistent
                ),
                "present": present,
                "well_formed": well_formed,
                "all_schema3_layers_match_external_campaign_spec_raw_bytes": (
                    all_layers_match
                ),
            }
        )
        if required:
            _require(
                formal_spec is not None,
                "--require-report-ready requires --campaign-spec",
            )
            _require(
                schema_version == FORMAL_SHARD_SCHEMA_VERSION,
                f"{path}: report-ready analysis requires shard schema 3; schema 2 is calibration-only",
            )
            _require(
                present,
                f"{path}: campaign_spec_sha256 is required for report-ready analysis",
            )
            _require(
                well_formed,
                f"{path}: campaign_spec_sha256 is not a lowercase SHA-256",
            )
            _require(
                all_layers_match,
                f"{path}: schema3 campaign binding/authentication differs from the external campaign spec raw bytes",
            )
    all_present = bool(entries) and all(entry["present"] for entry in entries)
    all_well_formed = bool(entries) and all(entry["well_formed"] for entry in entries)
    all_match = bool(entries) and all(
        entry["all_schema3_layers_match_external_campaign_spec_raw_bytes"]
        for entry in entries
    )
    return {
        "expected_external_campaign_spec_sha256": expected_sha256,
        "expected_external_campaign_spec_size_bytes": expected_size_bytes,
        "shard_count": len(entries),
        "all_shards_embed_campaign_spec_sha256": all_present,
        "all_embedded_hashes_well_formed": all_well_formed,
        "all_schema3_layers_match_external_campaign_spec_raw_bytes": all_match,
        "verified": bool(
            formal_spec is not None and all_present and all_well_formed and all_match
        ),
        "entries": entries,
    }


def _validate_shard(payload: dict) -> dict:
    path = payload["_analysis_path"]
    shard_schema_version = payload.get("schema_version")
    _require(
        shard_schema_version
        in (LEGACY_SHARD_SCHEMA_VERSION, FORMAL_SHARD_SCHEMA_VERSION),
        f"{path}: shard schema must be legacy 2 or formal 3",
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
    embedded_campaign_spec_sha256 = payload.get("campaign_spec_sha256")
    campaign_spec_authentication = payload.get("campaign_spec_authentication")
    ending_campaign_spec_authentication = payload.get(
        "ending_campaign_spec_authentication"
    )
    campaign_spec_stable = payload.get("campaign_spec_stable_during_run")
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        _require(
            embedded_campaign_spec_sha256 is None
            or _is_sha256(embedded_campaign_spec_sha256),
            f"{path}: campaign_spec_sha256 is malformed",
        )
        if embedded_campaign_spec_sha256 is None:
            _require(
                campaign_spec_authentication is None
                and ending_campaign_spec_authentication is None,
                f"{path}: absent campaign spec has non-null authentication",
            )
        else:
            _require(
                isinstance(campaign_spec_authentication, dict)
                and set(campaign_spec_authentication) == {"sha256", "size_bytes"}
                and campaign_spec_authentication.get("sha256")
                == embedded_campaign_spec_sha256,
                f"{path}: campaign spec authentication is inconsistent",
            )
            _integer(
                campaign_spec_authentication.get("size_bytes"),
                f"{path}: campaign spec size",
                minimum=1,
            )
            _require(
                ending_campaign_spec_authentication == campaign_spec_authentication,
                f"{path}: campaign spec authentication changed during run",
            )
        _require(
            campaign_spec_stable is True,
            f"{path}: campaign spec was not stable during run",
        )
    else:
        _require(
            "campaign_spec_sha256" not in payload
            and "campaign_spec_authentication" not in payload
            and "ending_campaign_spec_authentication" not in payload
            and "campaign_spec_stable_during_run" not in payload,
            f"{path}: schema2 artifact contains schema3 campaign-binding fields",
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
    _require("data" in source_manifest, f"{path}: data absent from source manifest")

    spec = payload.get("experiment_spec")
    cases, seed, backing = _validate_spec(spec, path, shard_schema_version)
    if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
        _require(
            spec["campaign_spec_sha256"] == embedded_campaign_spec_sha256,
            f"{path}: payload/spec campaign_spec_sha256 differs",
        )
        _require(
            payload.get("backing_requests_per_npu") == backing,
            f"{path}: root backing_requests_per_npu differs from the run config",
        )
        _require(
            payload.get("total_assignment_count") == NUM_NPU * backing,
            f"{path}: total_assignment_count must equal 128 NPUs times backing_requests_per_npu",
        )
    else:
        _require(
            "backing_requests_per_npu" not in payload
            and "total_assignment_count" not in payload,
            f"{path}: schema2 artifact contains schema3 topology/backing fields",
        )
    _require(
        spec["source_files"] == list(source_manifest),
        f"{path}: experiment source file list differs from source manifest",
    )
    _require(
        _canonical_hash(spec, b"ms-scale-control-config:v1\0") == config,
        f"{path}: config fingerprint mismatch",
    )
    definition_fingerprint = spec["definition_fingerprint"]
    _require(payload.get("definition") == DEFINITION, f"{path}: root definition")
    _require(
        payload.get("definition_fingerprint") == definition_fingerprint,
        f"{path}: root definition fingerprint",
    )
    _require(payload.get("num_npu") == NUM_NPU, f"{path}: root NPU count")
    authentication = spec["workload"]["authentication"]
    _require(
        payload.get("input_authentication") == authentication,
        f"{path}: root input authentication",
    )
    _require(
        source_manifest["data"] == authentication["source_sha256"],
        f"{path}: data file authentication",
    )
    loader = payload.get("input_loader_environment")
    _require(
        isinstance(loader, dict)
        and set(loader) == {"cache_path", "cache_present", "cache_verified_equal"},
        f"{path}: input loader environment",
    )
    _require(
        loader["cache_path"] == "results/bw_table_cache_v2_128npu.npz"
        and isinstance(loader["cache_present"], bool)
        and isinstance(loader["cache_verified_equal"], bool),
        f"{path}: input loader values changed",
    )
    _require(
        loader["cache_present"] == loader["cache_verified_equal"],
        f"{path}: cache was present but not verified",
    )
    _validate_path_abi(payload.get("path_abi"), path)
    raw_root_runtime = payload.get("runtime")
    root_runtime = _runtime_scientific(raw_root_runtime, path)
    root_process_runtime = _runtime_process_audit(raw_root_runtime, path)
    execution = payload.get("execution")
    _require(isinstance(execution, dict), f"{path}: execution metadata missing")
    _require(
        execution.get("multiprocessing_start_method")
        == root_runtime["multiprocessing_start_method"],
        f"{path}: execution/runtime start method",
    )
    _require(
        execution.get("single_ssu_process_pool_required") is True,
        f"{path}: single-SSU pool requirement disabled",
    )
    _integer(
        execution.get("requested_max_workers"), f"{path}: requested workers", minimum=1
    )

    manifest_data = _validate_seed_manifest(
        payload.get("schedule_metadata"), spec, path
    )
    selected = payload.get("selected_keys")
    _require(
        isinstance(selected, list) and len(selected) == 5,
        f"{path}: selected_keys must contain five cases",
    )
    selected_keys = []
    for index, key in enumerate(selected):
        _require(
            isinstance(key, list) and len(key) == 2, f"{path}: selected key {index}"
        )
        selected_keys.append(
            (str(key[0]), _integer(key[1], f"{path}: selected SSU", minimum=1))
        )
    _require(len(set(selected_keys)) == 5, f"{path}: duplicate selected key")
    selected_ssus = {key[1] for key in selected_keys}
    _require(len(selected_ssus) == 1, f"{path}: report shard must contain one SSU")
    num_ssu = next(iter(selected_ssus))
    _require(num_ssu in FORMAL_SSUS, f"{path}: SSU outside formal grid")
    expected_role_to_case = (
        ROLE_TO_CASE
        if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION
        else LEGACY_ROLE_TO_CASE
    )
    _require(
        {key[0] for key in selected_keys} == set(expected_role_to_case.values()),
        f"{path}: selected case set differs",
    )
    raw_results = payload.get("results")
    _require(
        isinstance(raw_results, list) and len(raw_results) == 5,
        f"{path}: results must contain five rows",
    )
    result_keys = [
        (str(row.get("case")), row.get("num_ssu"))
        for row in raw_results
        if isinstance(row, dict)
    ]
    _require(
        set(result_keys) == set(selected_keys)
        and len(result_keys) == len(set(result_keys)),
        f"{path}: result/selection key mismatch",
    )
    root_pairing = payload.get("pairing_audit")
    _require(
        isinstance(root_pairing, dict)
        and isinstance(root_pairing.get(str(num_ssu)), dict),
        f"{path}: pairing audit missing",
    )
    _require(
        root_pairing[str(num_ssu)].get("all_available_rows_paired") is True,
        f"{path}: runner pairing audit failed",
    )

    compact_rows = {}
    raw_by_role = {}
    for row in raw_results:
        if shard_schema_version == FORMAL_SHARD_SCHEMA_VERSION:
            _require(
                row.get("campaign_spec_sha256") == embedded_campaign_spec_sha256,
                f"{path}: result row campaign_spec_sha256 differs",
            )
            _require(
                row.get("num_npu") == NUM_NPU
                and row.get("backing_requests_per_npu") == backing,
                f"{path}: result row confuses 128 NPU with backing requests/NPU",
            )
        else:
            _require(
                "campaign_spec_sha256" not in row,
                f"{path}: schema2 row contains schema3 campaign binding",
            )
        compact = _validate_row(
            row,
            cases=cases,
            seed=seed,
            backing=backing,
            num_ssu=num_ssu,
            source=source,
            config=config,
            definition_fingerprint=definition_fingerprint,
            spec=spec,
            manifest_data=manifest_data,
            root_runtime=root_runtime,
            shard_path=path,
        )
        compact_rows[compact["role"]] = compact
        raw_by_role[compact["role"]] = row
    _require(set(compact_rows) == set(ROLE_ORDER), f"{path}: compact role set")
    input_fields = tuple(next(iter(compact_rows.values()))["input_fingerprints"])
    for field in input_fields:
        _require(
            len(
                {compact_rows[role]["input_fingerprints"][field] for role in ROLE_ORDER}
            )
            == 1,
            f"{path}: unpaired input {field}",
        )
    reference = compact_rows["B"]
    for role in ROLE_ORDER[1:]:
        current = compact_rows[role]
        _require(
            _equivalent_derived(
                current["workload_statistics"], reference["workload_statistics"]
            ),
            f"{path}: backing statistics not paired",
        )
        _require(
            _equivalent_derived(
                current["prefix_workload_statistics"],
                reference["prefix_workload_statistics"],
            ),
            f"{path}: prefix statistics not paired",
        )
        _require(
            current["prefix_materialized_fingerprints"]
            == reference["prefix_materialized_fingerprints"],
            f"{path}: prefix materialization not paired",
        )
    return {
        "path": path,
        "sha256": payload["_analysis_sha256"],
        "shard_schema_version": shard_schema_version,
        "seed": seed,
        "num_ssu": num_ssu,
        "backing": backing,
        "source_fingerprint": source,
        "source_manifest": source_manifest,
        "definition_fingerprint": definition_fingerprint,
        "config_fingerprint": config,
        "campaign_spec_sha256": embedded_campaign_spec_sha256,
        "campaign_spec_authentication": campaign_spec_authentication,
        "campaign_spec_stable_during_run": campaign_spec_stable,
        "spec": spec,
        "campaign_projection": _campaign_projection(spec),
        "authentication": authentication,
        "path_abi": payload["path_abi"],
        "root_runtime": root_runtime,
        "root_process_runtime": root_process_runtime,
        "runtime_campaign_signature": _runtime_campaign_signature(root_runtime),
        "loader_environment": loader,
        "schedule_metadata": payload["schedule_metadata"],
        "rows": compact_rows,
    }


def _validate_campaign(cells: list[dict]) -> None:
    _require(bool(cells), "at least one shard is required")
    reference = cells[0]
    seen_cells = set()
    seed_reference = {}
    assignments_by_seed = {}
    for cell in cells:
        key = (cell["seed"], cell["num_ssu"])
        _require(
            key not in seen_cells, f"duplicate report cell seed{key[0]}/SSU{key[1]}"
        )
        seen_cells.add(key)
        for field in (
            "shard_schema_version",
            "source_fingerprint",
            "source_manifest",
            "definition_fingerprint",
            "campaign_spec_sha256",
            "campaign_spec_authentication",
            "campaign_spec_stable_during_run",
            "campaign_projection",
            "authentication",
            "path_abi",
            "runtime_campaign_signature",
        ):
            _require(
                cell[field] == reference[field],
                f"{cell['path']}: campaign {field} differs",
            )
        seed = cell["seed"]
        if seed in seed_reference:
            previous = seed_reference[seed]
            _require(
                cell["config_fingerprint"] == previous["config_fingerprint"],
                f"seed{seed}: config differs across SSUs",
            )
            _require(
                cell["spec"] == previous["spec"],
                f"seed{seed}: spec differs across SSUs",
            )
            _require(
                cell["schedule_metadata"] == previous["schedule_metadata"],
                f"seed{seed}: schedule manifest differs across SSUs",
            )
        else:
            seed_reference[seed] = cell
            assignments_by_seed[seed] = cell["spec"]["workload"]["assignment"]
    _require(
        len(set(assignments_by_seed.values())) == len(assignments_by_seed),
        "different seeds reused the same assignment",
    )


T_CRITICAL_975 = {
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.200985160,
    12: 2.178812830,
    13: 2.160368656,
    14: 2.144786688,
    15: 2.131449546,
    16: 2.119905299,
    17: 2.109815578,
    18: 2.100922040,
    19: 2.093024054,
    20: 2.085963447,
    21: 2.079613845,
    22: 2.073873068,
    23: 2.068657610,
    24: 2.063898562,
    25: 2.059538553,
    26: 2.055529439,
    27: 2.051830516,
    28: 2.048407142,
    29: 2.045229642,
    30: 2.042272456,
}


def _summary_statistics(values: Sequence[float]) -> dict:
    parsed = [float(value) for value in values]
    _require(bool(parsed), "cannot summarize an empty metric")
    count = len(parsed)
    mean = statistics.fmean(parsed)
    sample_sd = statistics.stdev(parsed) if count >= 2 else None
    ci_low = ci_high = None
    if count >= 3:
        critical = T_CRITICAL_975.get(count - 1, 1.959963985)
        half_width = critical * sample_sd / math.sqrt(count)
        ci_low, ci_high = mean - half_width, mean + half_width
    return {
        "n": count,
        "mean": mean,
        "sample_sd": sample_sd,
        "median": statistics.median(parsed),
        "min": min(parsed),
        "max": max(parsed),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _build_analysis(
    cells: list[dict], planned_seeds: tuple[int, ...], planned_ssus: tuple[int, ...]
) -> dict:
    rows = {}
    cell_by_key = {(cell["seed"], cell["num_ssu"]): cell for cell in cells}
    for cell in cells:
        for role, row in cell["rows"].items():
            key = (cell["seed"], cell["num_ssu"], role)
            _require(key not in rows, f"duplicate run row {key}")
            rows[key] = dict(row)
    expected = {
        (seed, num_ssu, role)
        for seed in planned_seeds
        for num_ssu in planned_ssus
        for role in ROLE_ORDER
    }
    _require(
        set(rows) <= expected,
        "one or more shards are outside the explicit seed/SSU plan",
    )
    missing = sorted(
        expected - set(rows), key=lambda key: (key[0], key[1], ROLE_ORDER.index(key[2]))
    )

    paired_seed_rows = []
    for seed in planned_seeds:
        for num_ssu in planned_ssus:
            baseline = rows.get((seed, num_ssu, "B"))
            for role in ROLE_ORDER[1:]:
                current = rows.get((seed, num_ssu, role))
                if baseline is None or current is None:
                    continue
                paired_seed_rows.append(
                    {
                        "seed": seed,
                        "num_ssu": num_ssu,
                        "role": role,
                        "case": current["case"],
                        "delta_equal_npu_slo_alpha2_pp": current[
                            "equal_npu_slo_alpha2_pct"
                        ]
                        - baseline["equal_npu_slo_alpha2_pct"],
                        "delta_equal_npu_slo_alpha15_pp": current[
                            "equal_npu_slo_alpha15_pct"
                        ]
                        - baseline["equal_npu_slo_alpha15_pct"],
                        "delta_mean_npu_utilization_pp": current[
                            "mean_npu_utilization_pct"
                        ]
                        - baseline["mean_npu_utilization_pct"],
                        "both_steady_state_qualified": current["steady_state_qualified"]
                        and baseline["steady_state_qualified"],
                        "strategy_queue_regime": current["queue_regime"],
                        "baseline_queue_regime": baseline["queue_regime"],
                    }
                )

    curve_summary = []
    for num_ssu in planned_ssus:
        for role in ROLE_ORDER:
            group = [
                rows[(seed, num_ssu, role)]
                for seed in planned_seeds
                if (seed, num_ssu, role) in rows
            ]
            if not group:
                continue
            item = {
                "num_ssu": num_ssu,
                "role": role,
                "case": group[0]["case"],
                "planned_seed_count": len(planned_seeds),
                "available_seeds": [row["seed"] for row in group],
                "available_seed_count": len(group),
                "steady_state_qualified_count": sum(
                    row["steady_state_qualified"] for row in group
                ),
                "overloaded_growing_count": sum(
                    row["queue_regime"] == "overloaded_growing" for row in group
                ),
            }
            for metric in (
                "equal_npu_slo_alpha2_pct",
                "equal_npu_slo_alpha15_pct",
                "mean_npu_utilization_pct",
                "pressure_read_rate_hz_per_ssu",
                "cir_entry_write_rate_hz_per_ssu",
                "fleet_queue_slope_capacity_fraction",
            ):
                summary = _summary_statistics([row[metric] for row in group])
                item.update(
                    {f"{metric}_{name}": value for name, value in summary.items()}
                )
            curve_summary.append(item)

    paired_summary = []
    for num_ssu in planned_ssus:
        for role in ROLE_ORDER[1:]:
            group = [
                row
                for row in paired_seed_rows
                if row["num_ssu"] == num_ssu and row["role"] == role
            ]
            if not group:
                continue
            item = {
                "num_ssu": num_ssu,
                "role": role,
                "case": group[0]["case"],
                "planned_seed_count": len(planned_seeds),
                "available_seeds": [row["seed"] for row in group],
                "available_seed_count": len(group),
                "all_pairs_steady_state_qualified": all(
                    row["both_steady_state_qualified"] for row in group
                ),
            }
            for metric in (
                "delta_equal_npu_slo_alpha2_pp",
                "delta_equal_npu_slo_alpha15_pp",
                "delta_mean_npu_utilization_pp",
            ):
                values = [row[metric] for row in group]
                summary = _summary_statistics(values)
                item.update(
                    {f"{metric}_{name}": value for name, value in summary.items()}
                )
                item[f"{metric}_positive_seed_count"] = sum(
                    value > 0.0 for value in values
                )
                item[f"{metric}_nonnegative_seed_count"] = sum(
                    value >= 0.0 for value in values
                )
            paired_summary.append(item)
    return {
        "rows": rows,
        "cell_by_key": cell_by_key,
        "expected": expected,
        "missing": missing,
        "plan_complete": not missing and len(rows) == len(expected),
        "paired_seed_rows": paired_seed_rows,
        "curve_summary": curve_summary,
        "paired_summary": paired_summary,
    }


def _apply_formal_campaign_status(
    analysis: dict,
    cells: list[dict],
    planned_seeds: tuple[int, ...],
    planned_ssus: tuple[int, ...],
    formal_spec: dict | None,
    local_source_audit: dict | None,
    policy_freeze_audit: dict | None,
    campaign_spec_binding_audit: dict | None,
) -> None:
    reasons = []
    exact_preregistered_plan = (
        planned_seeds == FORMAL_SEEDS and planned_ssus == FORMAL_SSUS
    )
    if not exact_preregistered_plan:
        reasons.append("seed/SSU plan is a calibration subset, not the frozen 3x9 plan")
    if formal_spec is None:
        reasons.append("no frozen --campaign-spec was supplied")
    if not analysis["plan_complete"]:
        reasons.append("one or more rows are missing from the caller's plan")
    campaign_spec_binding_verified = bool(
        formal_spec is not None
        and isinstance(campaign_spec_binding_audit, dict)
        and campaign_spec_binding_audit.get("verified") is True
        and campaign_spec_binding_audit.get("expected_external_campaign_spec_sha256")
        == formal_spec["file_sha256"]
        and campaign_spec_binding_audit.get("shard_count") == len(cells)
    )
    if formal_spec is not None and not campaign_spec_binding_verified:
        reasons.append(
            "one or more shards do not bind the external campaign spec raw-byte SHA-256"
        )

    spec_matches = formal_spec is not None
    if formal_spec is not None:
        reference = cells[0]
        steady = reference["spec"]["steady_state"]
        comparisons = (
            (
                reference["source_fingerprint"],
                formal_spec["source_fingerprint"],
                "source fingerprint",
            ),
            (
                reference["definition_fingerprint"],
                formal_spec["definition_fingerprint"],
                "definition fingerprint",
            ),
            (
                reference["backing"],
                formal_spec["backing_requests_per_npu"],
                "backing requests/NPU",
            ),
            (
                float(steady["measurement_ms"]),
                float(formal_spec["measurement_ms"]),
                "measurement_ms",
            ),
            (float(steady["block_ms"]), float(formal_spec["block_ms"]), "block_ms"),
            (
                int(steady["warmup_requests_per_npu"]),
                int(formal_spec["warmup_requests_per_npu"]),
                "warmup requests/NPU",
            ),
            (float(steady["settle_ms"]), float(formal_spec["settle_ms"]), "settle_ms"),
        )
        for actual, expected, label in comparisons:
            equal = (
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                if isinstance(actual, float) or isinstance(expected, float)
                else actual == expected
            )
            if not equal:
                spec_matches = False
                reasons.append(f"artifact {label} differs from frozen campaign spec")

    exact_expected = {
        (seed, num_ssu, role)
        for seed in FORMAL_SEEDS
        for num_ssu in FORMAL_SSUS
        for role in ROLE_ORDER
    }
    exact_rows = set(analysis["rows"]) == exact_expected
    exact_cells = {(cell["seed"], cell["num_ssu"]) for cell in cells} == {
        (seed, num_ssu) for seed in FORMAL_SEEDS for num_ssu in FORMAL_SSUS
    }
    if not exact_rows or not exact_cells:
        reasons.append("formal campaign is not exactly 27 cells / 135 strategy rows")
    formal_complete = bool(
        formal_spec is not None
        and exact_preregistered_plan
        and analysis["plan_complete"]
        and campaign_spec_binding_verified
        and spec_matches
        and exact_rows
        and exact_cells
    )
    all_stationarity = all(
        row["steady_state_qualified"] for row in analysis["rows"].values()
    )
    if not all_stationarity:
        reasons.append("one or more rows failed the preregistered stationarity gate")
    policy_freeze_verified = bool(
        formal_spec is not None
        and isinstance(policy_freeze_audit, dict)
        and policy_freeze_audit.get("frozen") is True
        and policy_freeze_audit.get("selected_identity_consistent") is True
        and policy_freeze_audit.get("source_and_data_lineage_matches_report128") is True
        and policy_freeze_audit.get("report128_astar_parameters_match") is True
        and policy_freeze_audit.get("sha256")
        == formal_spec["policy_freeze_artifact_sha256"]
    )
    if not policy_freeze_verified:
        reasons.append(
            "frozen formal32 H70 policy-selection lineage was not independently verified"
        )
    local_root_verified = bool(
        isinstance(local_source_audit, dict)
        and local_source_audit.get("root_manifest_all_files_match") is True
        and local_source_audit.get("independent_import_closure_matches_manifest")
        is True
        and local_source_audit.get("source_fingerprint")
        == cells[0]["source_fingerprint"]
        and local_source_audit.get("verified_file_count")
        == len(cells[0]["source_manifest"])
        and local_source_audit.get("data_sha256")
        == cells[0]["authentication"]["source_sha256"]
        and local_source_audit.get("catalog_hash")
        == cells[0]["authentication"]["catalog_hash"]
        and local_source_audit.get("table_fingerprint")
        == cells[0]["authentication"]["table_fingerprint"]
        and local_source_audit.get("profile_count") == 84
        and local_source_audit.get("cache_used") is False
    )
    archive_audit = (
        local_source_audit.get("source_archive") if local_root_verified else None
    )
    source_archive_verified = bool(
        isinstance(archive_audit, dict)
        and _is_sha256(archive_audit.get("sha256"))
        and archive_audit.get("all_member_hashes_match_source_manifest") is True
        and archive_audit.get("verified_regular_file_count")
        == local_source_audit.get("verified_file_count")
    )
    source_archive_matches_campaign = bool(
        source_archive_verified
        and formal_spec is not None
        and archive_audit["sha256"] == formal_spec["source_archive_sha256"]
    )
    source_closure_verified = (
        local_root_verified
        and source_archive_verified
        and source_archive_matches_campaign
    )
    if not local_root_verified:
        reasons.append("local source/data root closure was not independently verified")
    if not source_archive_verified:
        reasons.append(
            "an exact immutable source archive was not independently verified"
        )
    elif formal_spec is None:
        reasons.append("source archive cannot be bound without a frozen campaign spec")
    elif not source_archive_matches_campaign:
        reasons.append("source archive SHA-256 differs from the frozen campaign spec")
    report_ready = (
        formal_complete
        and all_stationarity
        and source_closure_verified
        and policy_freeze_verified
    )
    analysis.update(
        {
            "formal_plan_selected": exact_preregistered_plan,
            "formal_campaign_spec": formal_spec,
            "campaign_spec_binding_verified": campaign_spec_binding_verified,
            "campaign_spec_binding_audit": campaign_spec_binding_audit,
            "formal_complete": formal_complete,
            "all_rows_stationarity_qualified": all_stationarity,
            "policy_freeze_verified": policy_freeze_verified,
            "policy_freeze_audit": policy_freeze_audit,
            "local_source_root_verified": local_root_verified,
            "source_archive_verified": source_archive_verified,
            "source_archive_matches_campaign": source_archive_matches_campaign,
            "source_closure_verified": source_closure_verified,
            "report_ready": report_ready,
            "formal_blockers": reasons,
            "analysis_status": (
                "REPORT_READY"
                if report_ready
                else (
                    "FORMAL_INPUT_COMPLETE_NOT_REPORT_READY"
                    if formal_complete
                    else "CALIBRATION_OR_PARTIAL"
                )
            ),
        }
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path: Path, payload) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n",
    )


def _csv_text(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


RUN_FIELDS = (
    "seed",
    "num_npu",
    "num_ssu",
    "role",
    "case",
    "pressure_ttl_ms",
    "cir_write_threshold_gbps",
    "min_interval_ms",
    "equal_npu_slo_alpha2_pct",
    "request_weighted_slo_alpha2_pct",
    "equal_npu_slo_alpha15_pct",
    "request_weighted_slo_alpha15_pct",
    "mean_npu_utilization_pct",
    "mean_ssd_utilization_pct",
    "max_ssd_utilization_pct",
    "mean_npu_link_utilization_pct",
    "measurement_request_count",
    "backing_requests_per_npu",
    "requests_per_npu_min",
    "requests_per_npu_median",
    "requests_per_npu_max",
    "backing_margin_fastest_npu",
    "worker_peak_rss_bytes",
    "pressure_reads",
    "pressure_read_rate_hz_per_ssu",
    "cir_entry_writes",
    "cir_entry_write_rate_hz_per_ssu",
    "cir_transactions",
    "cir_transaction_rate_hz_per_ssu",
    "control_evaluations",
    "fleet_queue_slope_gbps",
    "fleet_queue_slope_capacity_fraction",
    "max_abs_per_ssu_queue_slope_gbps",
    "max_abs_per_ssu_queue_slope_capacity_fraction",
    "util_half_delta_pp",
    "utilization_projected_change_pp",
    "fleet_served_first_half_gbps",
    "fleet_served_second_half_gbps",
    "fleet_served_half_relative_delta",
    "alpha2_slo_half_delta_pp",
    "alpha15_slo_half_delta_pp",
    "half_all_npus_sampled",
    "queue_regime",
    "queue_bounded",
    "temporal_quality_stable",
    "steady_state_qualified",
    "stationarity_rule_version",
    "stationarity_rules",
    "prefix_fleet_demand_gbps",
    "prefix_capacity_knee_ssu",
    "prefix_global_load_fraction",
    "prefix_max_ssu_demand_gbps",
    "prefix_ssu_over_40_count",
    "backing_fleet_demand_gbps",
    "backing_capacity_knee_ssu",
    "measurement_cohort_fingerprint",
    "wall_time_s",
)


def _public_run(row: dict) -> dict:
    return {field: row.get(field) for field in RUN_FIELDS}


def _write_tables(output_dir: Path, analysis: dict) -> None:
    analysis_status = analysis.get("analysis_status", "CALIBRATION_OR_PARTIAL")
    ordered_runs = [
        analysis["rows"][key]
        for key in sorted(
            analysis["rows"], key=lambda key: (key[0], key[1], ROLE_ORDER.index(key[2]))
        )
    ]
    public_runs = [
        {"analysis_status": analysis_status, **_public_run(row)} for row in ordered_runs
    ]
    _atomic_write_text(
        output_dir / "run_metrics.csv",
        _csv_text(public_runs, ("analysis_status",) + RUN_FIELDS),
    )
    stationarity_fields = (
        "analysis_status",
        "seed",
        "num_ssu",
        "role",
        "case",
        "block_count",
        "util_half_delta_pp",
        "utilization_projected_change_pp",
        "fleet_served_first_half_gbps",
        "fleet_served_second_half_gbps",
        "fleet_served_half_relative_delta",
        "alpha2_slo_half_delta_pp",
        "alpha15_slo_half_delta_pp",
        "half_all_npus_sampled",
        "fleet_queue_slope_gbps",
        "fleet_queue_slope_capacity_fraction",
        "max_abs_per_ssu_queue_slope_gbps",
        "max_abs_per_ssu_queue_slope_capacity_fraction",
        "queue_regime",
        "queue_bounded",
        "temporal_quality_stable",
        "steady_state_qualified",
        "stationarity_rule_version",
        "stationarity_rules",
    )
    _atomic_write_text(
        output_dir / "stationarity.csv", _csv_text(public_runs, stationarity_fields)
    )
    alpha_fields = (
        "analysis_status",
        "seed",
        "num_ssu",
        "role",
        "case",
        "equal_npu_slo_alpha2_pct",
        "request_weighted_slo_alpha2_pct",
        "equal_npu_slo_alpha15_pct",
        "request_weighted_slo_alpha15_pct",
        "measurement_request_count",
        "requests_per_npu_min",
    )
    _atomic_write_text(
        output_dir / "slo_alpha_sensitivity.csv", _csv_text(public_runs, alpha_fields)
    )
    paired_fields = (
        ("analysis_status",) + tuple(analysis["paired_seed_rows"][0])
        if analysis["paired_seed_rows"]
        else (
            "analysis_status",
            "seed",
            "num_ssu",
            "role",
            "case",
            "delta_equal_npu_slo_alpha2_pp",
            "delta_equal_npu_slo_alpha15_pp",
            "delta_mean_npu_utilization_pp",
            "both_steady_state_qualified",
            "strategy_queue_regime",
            "baseline_queue_regime",
        )
    )
    _atomic_write_text(
        output_dir / "seed_paired_deltas.csv",
        _csv_text(
            [
                {"analysis_status": analysis_status, **row}
                for row in analysis["paired_seed_rows"]
            ],
            paired_fields,
        ),
    )
    curve_fields = ["analysis_status"] + sorted(
        {key for row in analysis["curve_summary"] for key in row}
    )
    paired_summary_fields = sorted(
        {key for row in analysis["paired_summary"] for key in row}
    )
    paired_summary_fields = ["analysis_status"] + paired_summary_fields
    curve_rows = [
        {
            "analysis_status": analysis_status,
            **row,
            "available_seeds": json.dumps(
                row["available_seeds"], separators=(",", ":")
            ),
        }
        for row in analysis["curve_summary"]
    ]
    paired_rows = [
        {
            "analysis_status": analysis_status,
            **row,
            "available_seeds": json.dumps(
                row["available_seeds"], separators=(",", ":")
            ),
        }
        for row in analysis["paired_summary"]
    ]
    _atomic_write_text(
        output_dir / "curve_summary.csv", _csv_text(curve_rows, curve_fields)
    )
    _atomic_write_text(
        output_dir / "paired_delta_summary.csv",
        _csv_text(paired_rows, paired_summary_fields),
    )

    breakdown = []
    for row in ordered_runs:
        for alpha_name, metrics in (
            ("2.0", row["cohort_alpha2"]),
            ("1.5", row["cohort_alpha15"]),
        ):
            for group_type, names in (
                ("category", CATEGORIES),
                ("raw_demand_bins", tuple(name for name, _, _ in DEMAND_BINS)),
            ):
                for name in names:
                    metric = metrics[group_type].get(
                        name, {"count": 0, "passed": 0, "slo_attainment": None}
                    )
                    breakdown.append(
                        {
                            "analysis_status": analysis_status,
                            "seed": row["seed"],
                            "num_ssu": row["num_ssu"],
                            "role": row["role"],
                            "case": row["case"],
                            "alpha": alpha_name,
                            "group_type": group_type,
                            "group": name,
                            "count": metric["count"],
                            "passed": metric["passed"],
                            "slo_attainment_pct": None
                            if metric["slo_attainment"] is None
                            else 100.0 * metric["slo_attainment"],
                        }
                    )
    breakdown_fields = tuple(breakdown[0]) if breakdown else ()
    _atomic_write_text(
        output_dir / "cohort_breakdown.csv", _csv_text(breakdown, breakdown_fields)
    )


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    temporary.replace(path)
    plt.close(fig)


def _plot_quality(
    output_dir: Path, analysis: dict, planned_seeds: tuple[int, ...]
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 11.5), sharex=True)
    metrics = (
        ("equal_npu_slo_alpha2_pct", "Equal-NPU TTFT SLO, α=2 (%)"),
        ("equal_npu_slo_alpha15_pct", "Equal-NPU TTFT SLO, α=1.5 (%)"),
        ("mean_npu_utilization_pct", "Mean NPU utilization (%)"),
    )
    rows = analysis["rows"]
    for role in ROLE_ORDER:
        color = ROLE_COLORS[role]
        for seed in planned_seeds:
            points = sorted(
                (
                    row
                    for (row_seed, _, row_role), row in rows.items()
                    if row_seed == seed and row_role == role
                ),
                key=lambda row: row["num_ssu"],
            )
            if len(points) >= 2:
                for axis, (metric, _) in zip(axes, metrics):
                    axis.plot(
                        [row["num_ssu"] for row in points],
                        [row[metric] for row in points],
                        color=color,
                        linewidth=0.8,
                        alpha=0.18,
                    )
        summaries = sorted(
            (row for row in analysis["curve_summary"] if row["role"] == role),
            key=lambda row: row["num_ssu"],
        )
        if not summaries:
            continue
        x = [row["num_ssu"] for row in summaries]
        for axis, (metric, _) in zip(axes, metrics):
            y = [row[f"{metric}_mean"] for row in summaries]
            axis.plot(x, y, marker="o", linewidth=2.0, color=color, label=role)
            for point_x, point_y, row in zip(x, y, summaries):
                low, high = row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"]
                if low is not None and high is not None:
                    axis.errorbar(
                        [point_x],
                        [point_y],
                        yerr=[[point_y - low], [high - point_y]],
                        fmt="none",
                        ecolor=color,
                        capsize=3,
                        alpha=0.8,
                    )
                if row["steady_state_qualified_count"] < row["available_seed_count"]:
                    axis.scatter(
                        [point_x], [point_y], marker="x", s=55, color="black", zorder=5
                    )
    for axis, (_, label) in zip(axes, metrics):
        axis.set_ylabel(label)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("SSU count")
    axes[0].legend(ncol=5, loc="best")
    fig.suptitle(
        "128-NPU full-load quality curve — "
        + analysis.get("analysis_status", "CALIBRATION_OR_PARTIAL")
        + "\nThin lines: individual seeds; error bars: seed-level 95% t CI; ×: not all runs steady-qualified"
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "quality_curve.png")


def _plot_paired(output_dir: Path, analysis: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.5), sharex=True)
    metrics = (
        ("delta_equal_npu_slo_alpha2_pp", "Δ Equal-NPU SLO α=2 (pp)"),
        ("delta_equal_npu_slo_alpha15_pp", "Δ Equal-NPU SLO α=1.5 (pp)"),
        ("delta_mean_npu_utilization_pp", "Δ Mean NPU utilization (pp)"),
    )
    for role in ROLE_ORDER[1:]:
        points = sorted(
            (row for row in analysis["paired_summary"] if row["role"] == role),
            key=lambda row: row["num_ssu"],
        )
        if not points:
            continue
        x = [row["num_ssu"] for row in points]
        color = ROLE_COLORS[role]
        for axis, (metric, _) in zip(axes, metrics):
            y = [row[f"{metric}_mean"] for row in points]
            axis.plot(x, y, marker="o", linewidth=2.0, color=color, label=role)
            for point_x, point_y, row in zip(x, y, points):
                low, high = row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"]
                if low is not None and high is not None:
                    axis.errorbar(
                        [point_x],
                        [point_y],
                        yerr=[[point_y - low], [high - point_y]],
                        fmt="none",
                        ecolor=color,
                        capsize=3,
                    )
                if not row["all_pairs_steady_state_qualified"]:
                    axis.scatter(
                        [point_x], [point_y], marker="x", s=55, color="black", zorder=5
                    )
    for axis, (_, label) in zip(axes, metrics):
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("SSU count")
    axes[0].legend(ncol=4, loc="best")
    fig.suptitle(
        "Paired seed deltas versus Baseline — "
        + analysis.get("analysis_status", "CALIBRATION_OR_PARTIAL")
        + "\nCI unit is seed, never request or NPU"
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "paired_gain_vs_baseline.png")


def _plot_control_stationarity(output_dir: Path, analysis: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 10.5), sharex=True)
    views = (
        (
            axes[0],
            ("L0", "L*"),
            "pressure_read_rate_hz_per_ssu",
            "Modeled pressure-table reads/s/SSU",
        ),
        (
            axes[1],
            ("A0", "A*"),
            "cir_entry_write_rate_hz_per_ssu",
            "Modeled CIR entry writes/s/SSU",
        ),
        (
            axes[2],
            ROLE_ORDER,
            "fleet_queue_slope_capacity_fraction",
            "Fleet queue slope / total SSU capacity (%)",
        ),
    )
    for axis, roles, metric, label in views:
        for role in roles:
            points = sorted(
                (row for row in analysis["curve_summary"] if row["role"] == role),
                key=lambda row: row["num_ssu"],
            )
            if not points:
                continue
            multiplier = (
                100.0 if metric == "fleet_queue_slope_capacity_fraction" else 1.0
            )
            axis.plot(
                [row["num_ssu"] for row in points],
                [multiplier * row[f"{metric}_mean"] for row in points],
                marker="o",
                linewidth=2.0,
                color=ROLE_COLORS[role],
                label=role,
            )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=5)
    axes[2].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("SSU count")
    fig.suptitle(
        "Control cost and stationarity — "
        + analysis.get("analysis_status", "CALIBRATION_OR_PARTIAL")
        + "\nPressure reads and CIR writes are distinct hardware operations"
    )
    fig.tight_layout()
    _save_figure(fig, output_dir / "control_cost_and_stationarity.png")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _report_text(
    cells: list[dict],
    analysis: dict,
    planned_seeds: tuple[int, ...],
    planned_ssus: tuple[int, ...],
) -> str:
    status = analysis.get("analysis_status", "CALIBRATION_OR_PARTIAL")
    local_audit = analysis.get("local_source_and_data_audit") or {}
    archive_audit = local_audit.get("source_archive") or {}
    lines = [
        f"# 128-NPU report curve analysis — {status}",
        "",
        f"- Explicit seed plan: `{list(planned_seeds)}`",
        f"- Explicit SSU plan: `{list(planned_ssus)}`",
        f"- Topology: `{NUM_NPU} NPU`; finite backing: `{cells[0]['backing']} requests/NPU`; total assignments: `{NUM_NPU * cells[0]['backing']}`",
        f"- Validated cells: `{len(cells)}/{len(planned_seeds) * len(planned_ssus)}`",
        f"- Validated rows: `{len(analysis['rows'])}/{len(analysis['expected'])}`",
        f"- Missing rows: `{len(analysis['missing'])}`",
        f"- Plan complete: `{analysis['plan_complete']}`; formal complete: `{analysis['formal_complete']}`; report ready: `{analysis['report_ready']}`",
        f"- Frozen campaign fingerprint: `{(analysis.get('formal_campaign_spec') or {}).get('fingerprint', 'not supplied')}`",
        f"- Every shard binds the external campaign-spec raw-byte SHA256: `{analysis['campaign_spec_binding_verified']}`",
        f"- Frozen formal32 H70 policy-selection lineage verified: `{analysis['policy_freeze_verified']}`",
        "- Primary SLO is equal-NPU α=2. For both α values, `ideal_ttft_ms = 16 × authenticated_catalog[profile_key].per_layer_compute_us / 1000`; a request passes iff `ttft_ms <= α × ideal_ttft_ms + 1e-12`. The catalog's ad-hoc third value is never used as ideal TTFT.",
        "- Curve intervals use one value per independent seed; gain intervals first pair strategy and Baseline within each seed, then apply the t interval across seed-level deltas. Fewer than three seeds produce no formal CI.",
        "",
    ]
    if analysis["formal_blockers"]:
        lines.extend(
            [
                "## Formal/report blockers",
                "",
                *[f"- {reason}" for reason in analysis["formal_blockers"]],
                "",
            ]
        )
    if analysis["missing"]:
        lines.extend(
            [
                "## Missing planned rows",
                "",
                "```text",
                *[
                    f"seed={seed} SSU={ssu} role={role}"
                    for seed, ssu, role in analysis["missing"]
                ],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Quality curve",
            "",
            "| SSU | role | seeds | α2 equal-NPU SLO | α1.5 equal-NPU SLO | NPU util | steady-qualified | growing queue |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["curve_summary"]:
        lines.append(
            f"| {row['num_ssu']} | {row['role']} | {row['available_seed_count']} | "
            f"{_fmt(row['equal_npu_slo_alpha2_pct_mean'])}% | "
            f"{_fmt(row['equal_npu_slo_alpha15_pct_mean'])}% | "
            f"{_fmt(row['mean_npu_utilization_pct_mean'])}% | "
            f"{row['steady_state_qualified_count']}/{row['available_seed_count']} | "
            f"{row['overloaded_growing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Paired gains versus Baseline",
            "",
            "| SSU | role | paired seeds | ΔSLO α2 mean [95% CI] | ΔSLO α1.5 mean [95% CI] | Δutil mean [95% CI] | all steady |",
            "|---:|:---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in analysis["paired_summary"]:
        cells_text = []
        for metric in (
            "delta_equal_npu_slo_alpha2_pp",
            "delta_equal_npu_slo_alpha15_pp",
            "delta_mean_npu_utilization_pp",
        ):
            cells_text.append(
                f"{_fmt(row[f'{metric}_mean'])} [{_fmt(row[f'{metric}_ci95_low'])}, {_fmt(row[f'{metric}_ci95_high'])}]"
            )
        lines.append(
            f"| {row['num_ssu']} | {row['role']} | {row['available_seed_count']} | "
            + " | ".join(cells_text)
            + f" | {row['all_pairs_steady_state_qualified']} |"
        )
    lines.extend(
        [
            "",
            "## Stationarity interpretation",
            "",
            "A row is steady-qualified only with at least 16 complete 500-ms-style blocks, all 128 NPUs sampled in both halves, half-window utilization delta ≤1 pp, Theil–Sen utilization trend projected over the full window ≤2 pp, fleet served-rate half delta ≤2%, and no persistent queue growth beyond the independently derived one-layer fleet burst bound.",
            "",
            "`overloaded_growing` rows remain valid finite-window capacity observations, but they are not evidence of an infinite-horizon bounded-queue TTFT steady state. They are never silently discarded from paired statistics.",
            "",
            "## Provenance",
            "",
            f"- Source fingerprint: `{cells[0]['source_fingerprint']}`",
            f"- Definition fingerprint: `{cells[0]['definition_fingerprint']}`",
            f"- Data catalog fingerprint: `{cells[0]['authentication']['catalog_hash']}`",
            f"- Analysis source fingerprint: `{_file_sha256(Path(__file__).resolve())}`",
            f"- External campaign-spec raw-byte SHA256: `{(analysis.get('formal_campaign_spec') or {}).get('file_sha256', 'not supplied')}`",
            f"- Local source/data root verified: `{analysis['local_source_root_verified']}`",
            f"- Immutable source archive verified: `{analysis['source_archive_verified']}`; archive SHA256: `{archive_audit.get('sha256', 'not supplied')}`",
            f"- Archive matches frozen campaign SHA256: `{analysis['source_archive_matches_campaign']}`",
            f"- Full root + archive closure verified: `{analysis['source_closure_verified']}`",
            f"- Policy freeze artifact verified: `{analysis['policy_freeze_verified']}`; artifact SHA256: `{(analysis.get('policy_freeze_audit') or {}).get('sha256', 'not supplied')}`",
            f"- Policy freeze fingerprint: `{(analysis.get('policy_freeze_audit') or {}).get('freeze_fingerprint', 'not supplied')}`",
            "",
            "| seed | SSU | shard SHA256 | runtime | root / max-worker peak RSS |",
            "|---:|---:|:---|:---|---:|",
        ]
    )
    for cell in sorted(cells, key=lambda item: (item["seed"], item["num_ssu"])):
        runtime = cell["root_runtime"]
        peak_rss_gib = cell["root_process_runtime"]["rss_peak_bytes"] / (1024**3)
        worker_peak_rss_gib = max(
            row["worker_peak_rss_bytes"] for row in cell["rows"].values()
        ) / (1024**3)
        lines.append(
            f"| {cell['seed']} | {cell['num_ssu']} | `{cell['sha256']}` | "
            f"{runtime['hostname']}; Python {runtime['python']}; NumPy {runtime['numpy']}; {runtime['multiprocessing_start_method']} | "
            f"{peak_rss_gib:.3f} / {worker_peak_rss_gib:.3f} GiB |"
        )
    lines.extend(
        [
            "",
            "Generated artifacts: `results.json`, six analysis CSVs, `cohort_breakdown.csv`, and three PNG figures.",
            "",
        ]
    )
    return "\n".join(lines)


def _expect_validation_failure(callback, label: str) -> None:
    try:
        callback()
    except ValidationError:
        return
    raise AssertionError(f"fault injection was not rejected: {label}")


def _synthetic_manifest_fixture() -> tuple[dict, dict]:
    seq_values = (32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 200)
    nql_values = (64, 128, 256, 512, 1024, 2048, 4096)
    catalog_rows = []
    for index, (seq_len, nql) in enumerate(
        (
            pair
            for seq_len in seq_values
            for pair in ((seq_len, nql) for nql in nql_values)
        )
    ):
        required = 5.0 + index
        per_layer_us = 1000.0 + index
        catalog_rows.append(
            [
                seq_len,
                nql,
                [
                    required,
                    per_layer_us,
                    16.0 * per_layer_us / 1000.0,
                    required * per_layer_us / 1e6,
                ],
            ]
        )
    normalized = [[row[:2], row[2]] for row in catalog_rows]
    catalog_hash = _canonical_hash(normalized, b"random-steady-state:data-catalog:v1\0")
    table_hash = _canonical_hash(catalog_rows, b"authenticated-bw-table:v1\0")
    seed = 7
    backing = 32
    recipe_hash = _canonical_hash(
        _recipe(seed, backing, catalog_hash), b"random-steady-state:recipe:v1\0"
    )
    assignments = []
    keys = [(row[0], row[1]) for row in catalog_rows]
    for sequence in range(backing):
        for npu in range(NUM_NPU):
            request_id = sequence * NUM_NPU + npu
            key = keys[_expected_profile_index(seed, npu, sequence, len(keys))]
            assignments.append(
                [request_id, npu, sequence, _classify_profile(key), list(key)]
            )
    assignment_hash = _assignment_hash(assignments)
    schedule_hash = _canonical_hash(
        {"recipe_hash": recipe_hash, "assignments": assignments},
        b"random-steady-state:schedule:v1\0",
    )
    workload = {
        "mode": "iid_uniform_profile_catalog_v1",
        "seed": seed,
        "requests_per_npu": backing,
        "catalog": catalog_hash,
        "recipe": recipe_hash,
        "schedule": schedule_hash,
        "assignment": assignment_hash,
        "prefix_32_assignment_hash": assignment_hash,
        "full_assignment_hash": assignment_hash,
        "authentication": {
            "source": "data",
            "source_sha256": "0" * 64,
            "catalog_hash": catalog_hash,
            "table_fingerprint": table_hash,
            "profile_count": 84,
        },
    }
    manifest = {
        "catalog": catalog_hash,
        "recipe": recipe_hash,
        "schedule": schedule_hash,
        "assignment": assignment_hash,
        "mode": workload["mode"],
        "seed": seed,
        "num_npu": NUM_NPU,
        "requests_per_npu": backing,
        "request_id_formula": "sequence * num_npu + npu_id",
        "catalog_rows": catalog_rows,
        "assignment_rows": assignments,
    }
    return manifest, {"workload": workload}


def _run_self_test(shard_paths: Sequence[Path]) -> dict:
    manifest, spec = _synthetic_manifest_fixture()
    _validate_seed_manifest(manifest, spec, "self-test")
    corrupted = json.loads(json.dumps(manifest))
    original_category = corrupted["assignment_rows"][0][3]
    corrupted["assignment_rows"][0][3] = next(
        category for category in CATEGORIES if category != original_category
    )
    _expect_validation_failure(
        lambda: _validate_seed_manifest(corrupted, spec, "self-test/category"),
        "assignment category",
    )
    corrupted = json.loads(json.dumps(manifest))
    corrupted["catalog_rows"][0][2][0] += 1.0
    _expect_validation_failure(
        lambda: _validate_seed_manifest(corrupted, spec, "self-test/catalog"),
        "catalog value",
    )

    # Small known-answer vector for the independent 128-NPU input rebuild.  It
    # fixes the request-ID ABI, block-ring placement, initial jitter, trace, and
    # simulator-input serialization without running the simulator.
    materialization_catalog = {(1, 64): (5.0, 1000.0, 16.0, 0.005)}
    materialization_assignments = [
        [npu, npu, 0, "SS", [1, 64]] for npu in range(NUM_NPU)
    ]
    materialization_assignment_hash = _assignment_hash(materialization_assignments)
    _require(
        materialization_assignment_hash
        == "d2e81099797cbcbe74c355495dc356f16f08ea46c6e5bd22b6142def7d52160f",
        "self-test materialization assignment known answer",
    )
    materialized = _materialize_inputs(
        assignments=materialization_assignments,
        assignment_hash=materialization_assignment_hash,
        catalog=materialization_catalog,
        seed=123,
        num_ssu=2,
        requests_per_npu=1,
    )
    _require(
        {
            name: materialized[name]
            for name in ("workload", "placement", "trace", "simulator")
        }
        == {
            "workload": "869b8e4de3241695d2372fb82215835ee07cd87afdde80944d93893ed89b3500",
            "placement": "d906f375b931fb0bcbe9c81093d2acc6d608a0a37a4adf6145121f52e41bfa8b",
            "trace": "9ae7eaec1cd8a978f46a763cdbd96c5d8d0cfde365b5892c53d2bb1f8908c15b",
            "simulator": "1c27ae48911afa5c7e1d844c478bb4ac0636259777225095ee1697127e9a398c",
        },
        "self-test materialization fingerprints changed",
    )
    _close(
        math.fsum(materialized["demand_gbps_by_ssu"]),
        NUM_NPU * 5.0,
        "self-test materialization fleet demand",
    )
    _compare_vectors(
        materialized["demand_gbps_by_ssu"],
        [301.0, 339.0],
        "self-test materialization placed demand",
    )
    stats = _summary_statistics((1.0, 2.0, 3.0))
    _close(stats["mean"], 2.0, "self-test mean")
    _close(stats["sample_sd"], 1.0, "self-test sample SD")
    expected_half = T_CRITICAL_975[2] / math.sqrt(3.0)
    _close(stats["ci95_low"], 2.0 - expected_half, "self-test CI low")
    _close(stats["ci95_high"], 2.0 + expected_half, "self-test CI high")
    _validate_path_abi(
        {
            "path_count": 256,
            "group_count": 8,
            "paths_per_group": 32,
            "max_npu": 128,
            "assigned_count": 128,
            "assigned_unique": 128,
            "assigned_min": 16,
            "assigned_max": 255,
            "path_zero_reserved": True,
            "assigned_paths_sha256": _canonical_hash(
                _path_mapping(), b"ms-scale-control-path-abi:v1\0"
            ),
        },
        "self-test",
    )
    runtime_fixture = {
        "hostname": "self-test-host",
        "python": "3.11.0",
        "python_full": "3.11.0 self-test",
        "python_implementation": "CPython",
        "numpy": "1.26.0",
        "platform": "Linux-self-test",
        "multiprocessing_start_method": "spawn",
        "cpu_count": 8,
        "pid": 123,
        "rss_current_bytes": 1024,
        "rss_peak_bytes": 2048,
    }
    parsed_runtime = _runtime_scientific(runtime_fixture, "self-test/runtime")
    _require(
        _runtime_campaign_signature(parsed_runtime)
        == {
            "python": "3.11.0",
            "python_implementation": "CPython",
            "numpy": "1.26.0",
            "multiprocessing_start_method": "spawn",
        },
        "self-test runtime campaign signature",
    )
    invalid_runtime = dict(runtime_fixture)
    invalid_runtime["rss_peak_bytes"] = 512
    _expect_validation_failure(
        lambda: _runtime_scientific(invalid_runtime, "self-test/runtime-rss"),
        "runtime peak RSS below current RSS",
    )
    boundary_rules = _stationarity_rule_results(
        half_all_npus_sampled=True,
        util_half_delta_pp=1.0,
        util_projected_change_pp=2.0,
        served_half_relative_delta=0.02,
        persistent_queue_growth=False,
    )
    _require(all(boundary_rules.values()), "self-test stationarity boundary acceptance")
    rejected_rules = _stationarity_rule_results(
        half_all_npus_sampled=True,
        util_half_delta_pp=1.0001,
        util_projected_change_pp=2.0001,
        served_half_relative_delta=0.020001,
        persistent_queue_growth=True,
    )
    _require(
        not any(
            rejected_rules[name]
            for name in (
                "utilization_half_delta_le_1pp",
                "utilization_projected_trend_le_2pp",
                "fleet_served_half_relative_delta_le_2pct",
                "no_persistent_queue_growth_over_one_layer_burst",
            )
        ),
        "self-test stationarity threshold rejection",
    )

    source_hash = "a" * 64
    definition_hash = "b" * 64
    data_hash = "6" * 64
    catalog_hash = "7" * 64
    table_hash = "8" * 64
    synthetic_source_manifest = {
        f"file_{index}.py": f"{index:x}" * 64 for index in range(16)
    }
    formal_spec = {
        "num_npu": NUM_NPU,
        "source_fingerprint": source_hash,
        "source_archive_sha256": "d" * 64,
        "policy_freeze_artifact_sha256": "9" * 64,
        "analysis_source_sha256": _file_sha256(Path(__file__).resolve()),
        "definition_fingerprint": definition_hash,
        "backing_requests_per_npu": 256,
        "measurement_ms": 32000.0,
        "block_ms": 500.0,
        "warmup_requests_per_npu": 8,
        "settle_ms": 500.0,
        "fingerprint": "c" * 64,
        "file_sha256": "e" * 64,
        "file_size_bytes": 1234,
    }
    binding_payload = {
        "_analysis_path": "self-test/schema3-shard.json",
        "schema_version": FORMAL_SHARD_SCHEMA_VERSION,
        "num_npu": NUM_NPU,
        "backing_requests_per_npu": 256,
        "total_assignment_count": NUM_NPU * 256,
        "campaign_spec_sha256": formal_spec["file_sha256"],
        "campaign_spec_authentication": {
            "sha256": formal_spec["file_sha256"],
            "size_bytes": formal_spec["file_size_bytes"],
        },
        "ending_campaign_spec_authentication": {
            "sha256": formal_spec["file_sha256"],
            "size_bytes": formal_spec["file_size_bytes"],
        },
        "campaign_spec_stable_during_run": True,
        "experiment_spec": {
            "campaign_spec_sha256": formal_spec["file_sha256"],
            "num_npu": NUM_NPU,
            "workload": {"requests_per_npu": 256},
            "steady_state": {"requests_per_npu": 256},
        },
        "schedule_metadata": {"num_npu": NUM_NPU, "requests_per_npu": 256},
        "results": [
            {
                "campaign_spec_sha256": formal_spec["file_sha256"],
                "num_npu": NUM_NPU,
                "backing_requests_per_npu": 256,
            }
            for _ in ROLE_ORDER
        ],
    }
    valid_campaign_binding_audit = _campaign_spec_binding_audit(
        [binding_payload for _ in range(len(FORMAL_SEEDS) * len(FORMAL_SSUS))],
        formal_spec,
        required=True,
    )
    _require(
        valid_campaign_binding_audit["verified"],
        "self-test valid schema3 campaign binding",
    )
    missing_binding_payload = json.loads(json.dumps(binding_payload))
    missing_binding_payload.pop("campaign_spec_sha256")
    missing_binding_audit = _campaign_spec_binding_audit(
        [missing_binding_payload], formal_spec, required=False
    )
    _require(
        not missing_binding_audit["verified"],
        "self-test missing campaign binding must remain unverified",
    )
    _expect_validation_failure(
        lambda: _campaign_spec_binding_audit(
            [missing_binding_payload], formal_spec, required=True
        ),
        "missing strict campaign binding",
    )
    mismatched_binding_payload = json.loads(json.dumps(binding_payload))
    mismatched_binding_payload["campaign_spec_sha256"] = "f" * 64
    _expect_validation_failure(
        lambda: _campaign_spec_binding_audit(
            [mismatched_binding_payload], formal_spec, required=True
        ),
        "mismatched strict campaign binding",
    )
    confused_scale_payload = json.loads(json.dumps(binding_payload))
    confused_scale_payload["total_assignment_count"] = 256
    _expect_validation_failure(
        lambda: _campaign_spec_binding_audit(
            [confused_scale_payload], formal_spec, required=True
        ),
        "128 NPU confused with backing in root assignment count",
    )
    confused_row_payload = json.loads(json.dumps(binding_payload))
    confused_row_payload["results"][0]["backing_requests_per_npu"] = NUM_NPU
    _expect_validation_failure(
        lambda: _campaign_spec_binding_audit(
            [confused_row_payload], formal_spec, required=True
        ),
        "row backing confused with 128-NPU topology",
    )
    legacy_binding_payload = {
        "_analysis_path": "self-test/schema2-shard.json",
        "schema_version": LEGACY_SHARD_SCHEMA_VERSION,
    }
    _expect_validation_failure(
        lambda: _campaign_spec_binding_audit(
            [legacy_binding_payload], formal_spec, required=True
        ),
        "legacy schema2 strict campaign binding",
    )
    formal_rows = {
        (seed_value, ssu, role): {"steady_state_qualified": True}
        for seed_value in FORMAL_SEEDS
        for ssu in FORMAL_SSUS
        for role in ROLE_ORDER
    }
    formal_cells = [
        {
            "seed": seed_value,
            "num_ssu": ssu,
            "source_fingerprint": source_hash,
            "source_manifest": synthetic_source_manifest,
            "authentication": {
                "source_sha256": data_hash,
                "catalog_hash": catalog_hash,
                "table_fingerprint": table_hash,
            },
            "definition_fingerprint": definition_hash,
            "backing": 256,
            "spec": {
                "steady_state": {
                    "measurement_ms": 32000.0,
                    "block_ms": 500.0,
                    "warmup_requests_per_npu": 8,
                    "settle_ms": 500.0,
                }
            },
        }
        for seed_value in FORMAL_SEEDS
        for ssu in FORMAL_SSUS
    ]
    formal_analysis = {
        "plan_complete": True,
        "rows": formal_rows,
    }
    valid_source_audit = {
        "root_manifest_all_files_match": True,
        "independent_import_closure_matches_manifest": True,
        "source_fingerprint": source_hash,
        "data_sha256": data_hash,
        "catalog_hash": catalog_hash,
        "table_fingerprint": table_hash,
        "profile_count": 84,
        "cache_used": False,
        "verified_file_count": 16,
        "source_archive": {
            "sha256": "d" * 64,
            "verified_regular_file_count": 16,
            "all_member_hashes_match_source_manifest": True,
        },
    }
    valid_policy_freeze_audit = {
        "sha256": "9" * 64,
        "frozen": True,
        "selected_identity_consistent": True,
        "source_and_data_lineage_matches_report128": True,
        "report128_astar_parameters_match": True,
    }
    _apply_formal_campaign_status(
        formal_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        valid_source_audit,
        valid_policy_freeze_audit,
        valid_campaign_binding_audit,
    )
    _require(
        formal_analysis["formal_complete"] and formal_analysis["report_ready"],
        "self-test exact 27-cell/135-row formal status",
    )
    root_only_analysis = {
        "plan_complete": True,
        "rows": formal_rows,
    }
    root_only_audit = dict(valid_source_audit)
    root_only_audit["source_archive"] = None
    _apply_formal_campaign_status(
        root_only_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        root_only_audit,
        valid_policy_freeze_audit,
        valid_campaign_binding_audit,
    )
    _require(
        root_only_analysis["formal_complete"]
        and not root_only_analysis["report_ready"]
        and not root_only_analysis["source_archive_verified"],
        "self-test formal report must require an immutable source archive",
    )
    no_policy_analysis = {"plan_complete": True, "rows": formal_rows}
    _apply_formal_campaign_status(
        no_policy_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        valid_source_audit,
        None,
        valid_campaign_binding_audit,
    )
    _require(
        no_policy_analysis["formal_complete"]
        and not no_policy_analysis["report_ready"]
        and not no_policy_analysis["policy_freeze_verified"],
        "self-test formal report must require frozen policy-selection lineage",
    )
    unbound_campaign_analysis = {"plan_complete": True, "rows": formal_rows}
    _apply_formal_campaign_status(
        unbound_campaign_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        valid_source_audit,
        valid_policy_freeze_audit,
        missing_binding_audit,
    )
    _require(
        not unbound_campaign_analysis["formal_complete"]
        and not unbound_campaign_analysis["report_ready"]
        and not unbound_campaign_analysis["campaign_spec_binding_verified"],
        "self-test missing shard campaign binding must block formal/report status",
    )
    stale_policy_analysis = {"plan_complete": True, "rows": formal_rows}
    stale_policy_audit = dict(valid_policy_freeze_audit)
    stale_policy_audit["source_and_data_lineage_matches_report128"] = False
    _apply_formal_campaign_status(
        stale_policy_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        valid_source_audit,
        stale_policy_audit,
        valid_campaign_binding_audit,
    )
    _require(
        stale_policy_analysis["formal_complete"]
        and not stale_policy_analysis["report_ready"]
        and not stale_policy_analysis["policy_freeze_verified"],
        "self-test stale confirm32 source lineage must block report readiness",
    )
    forged_audit_analysis = {
        "plan_complete": True,
        "rows": formal_rows,
    }
    forged_audit = dict(valid_source_audit)
    forged_audit["source_fingerprint"] = "e" * 64
    _apply_formal_campaign_status(
        forged_audit_analysis,
        formal_cells,
        FORMAL_SEEDS,
        FORMAL_SSUS,
        formal_spec,
        forged_audit,
        valid_policy_freeze_audit,
        valid_campaign_binding_audit,
    )
    _require(
        forged_audit_analysis["formal_complete"]
        and not forged_audit_analysis["report_ready"]
        and not forged_audit_analysis["local_source_root_verified"],
        "self-test report readiness must bind the audited source fingerprint",
    )
    subset_analysis = {
        "plan_complete": True,
        "rows": {
            (FORMAL_SEEDS[0], FORMAL_SSUS[0], role): {"steady_state_qualified": True}
            for role in ROLE_ORDER
        },
    }
    _apply_formal_campaign_status(
        subset_analysis,
        [formal_cells[0]],
        (FORMAL_SEEDS[0],),
        (FORMAL_SSUS[0],),
        None,
        None,
        None,
        None,
    )
    _require(
        not subset_analysis["formal_complete"]
        and not subset_analysis["report_ready"]
        and subset_analysis["analysis_status"] == "CALIBRATION_OR_PARTIAL",
        "self-test subset must remain calibration/partial",
    )
    shard_faults = 0
    if shard_paths:
        original = _read_shard(shard_paths[0])
        _validate_shard(original)
        for label, mutate in (
            (
                "source stability",
                lambda value: value.__setitem__("source_stable_during_run", False),
            ),
            ("five-case selection", lambda value: value["selected_keys"].pop()),
            (
                "request raw demand",
                lambda value: value["results"][0]["steady_summary"]["request_rows"][
                    0
                ].__setitem__(
                    "raw_demand_gbps",
                    value["results"][0]["steady_summary"]["request_rows"][0][
                        "raw_demand_gbps"
                    ]
                    + 1.0,
                ),
            ),
        ):
            candidate = json.loads(json.dumps(original))
            mutate(candidate)
            _expect_validation_failure(
                lambda candidate=candidate: _validate_shard(candidate), label
            )
            shard_faults += 1
    return {
        "self_test": "passed",
        "synthetic_faults_rejected": 2,
        "shard_faults_rejected": shard_faults,
        "t_ci_checked": True,
        "path_abi_checked": True,
        "materialization_known_answer_checked": True,
        "schema3_campaign_binding_checked": True,
        "runtime_rss_checked": True,
        "stationarity_rules_checked": True,
        "formal_status_checked": True,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", type=Path, help="report128 JSON shards")
    parser.add_argument(
        "--seed", action="append", type=int, help="planned seed; repeat explicitly"
    )
    parser.add_argument(
        "--ssu", action="append", type=int, help="planned SSU; repeat explicitly"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "analysis directory; default is "
            "results/ms_scale_control/npu128_scale_analysis_128npu_backing_per_npuN"
        ),
    )
    parser.add_argument(
        "--campaign-spec",
        type=Path,
        help=(
            "frozen formal campaign JSON; it must name exactly seeds 42/43/44, "
            "all nine preregistered SSUs, all five roles, and the immutable run shape"
        ),
    )
    parser.add_argument(
        "--policy-freeze-artifact",
        type=Path,
        help=(
            "frozen formal32 H70 selection artifact whose exact SHA-256 and "
            "normalized A* parameters must match report128"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="source snapshot root whose complete manifest and data are independently hashed",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="optional immutable tar archive that must exactly match the source manifest",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="backward-compatible: fail unless the caller-supplied seed/SSU plan is complete",
    )
    parser.add_argument(
        "--require-formal",
        action="store_true",
        help=(
            "fail unless the exact frozen 27-cell/135-row formal input matrix is "
            "complete; this does not require stationarity or source/archive closure"
        ),
    )
    parser.add_argument(
        "--require-report-ready",
        action="store_true",
        help=(
            "strict publication/CI gate: require formal completeness, every row's "
            "stationarity gate, schema3 shard/spec/row binding to the external "
            "campaign-spec raw-byte SHA256, frozen source-root/archive closure, "
            "and frozen formal32 H70 selection lineage"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run in-memory fault injection; an optional shard adds end-to-end corruptions",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _require(
            len(args.shards) <= 1
            and not args.seed
            and not args.ssu
            and args.output_dir is None
            and not args.require_complete
            and not args.require_formal
            and not args.require_report_ready
            and args.campaign_spec is None
            and args.policy_freeze_artifact is None
            and args.source_root is None
            and args.source_archive is None,
            "--self-test accepts at most one optional shard and no plans, output, or formal gates",
        )
        result = _run_self_test(args.shards)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    _require(bool(args.shards), "at least one shard path is required")
    _require(
        args.source_archive is None or args.source_root is not None,
        "--source-archive requires --source-root",
    )
    formal_spec = (
        _read_formal_campaign_spec(args.campaign_spec)
        if args.campaign_spec is not None
        else None
    )
    _require(
        args.policy_freeze_artifact is None or formal_spec is not None,
        "--policy-freeze-artifact requires --campaign-spec",
    )
    if args.require_report_ready:
        _require(
            formal_spec is not None,
            "--require-report-ready requires --campaign-spec",
        )
    if formal_spec is None:
        _require(
            bool(args.seed) and bool(args.ssu),
            "explicit --seed and --ssu plans are required for partial/calibration mode",
        )
    else:
        if args.seed:
            _require(
                tuple(sorted(args.seed)) == FORMAL_SEEDS,
                "--seed differs from frozen campaign spec",
            )
        if args.ssu:
            _require(
                tuple(sorted(args.ssu)) == FORMAL_SSUS,
                "--ssu differs from frozen campaign spec",
            )
        args.seed = list(FORMAL_SEEDS)
        args.ssu = list(FORMAL_SSUS)
    _require(len(args.seed) == len(set(args.seed)), "duplicate --seed in plan")
    _require(len(args.ssu) == len(set(args.ssu)), "duplicate --ssu in plan")
    planned_seeds = tuple(sorted(_integer(seed, "planned seed") for seed in args.seed))
    _require(
        all(seed < 2**64 for seed in planned_seeds), "planned seed exceeds uint64"
    )
    planned_ssus = tuple(
        sorted(_integer(ssu, "planned SSU", minimum=1) for ssu in args.ssu)
    )
    _require(
        set(planned_ssus) <= set(FORMAL_SSUS), f"planned SSU must be in {FORMAL_SSUS}"
    )
    analyzer_source_start = _file_sha256(Path(__file__).resolve())
    resolved = [path.expanduser().resolve() for path in args.shards]
    _require(len(resolved) == len(set(resolved)), "same shard path supplied twice")
    raw_payloads = [_read_shard(path) for path in resolved]
    campaign_spec_binding_audit = _campaign_spec_binding_audit(
        raw_payloads, formal_spec, required=args.require_report_ready
    )
    local_source_audit = (
        _validate_local_source_closure(
            raw_payloads, args.source_root, args.source_archive
        )
        if args.source_root is not None
        else None
    )
    cells = [_validate_shard(payload) for payload in raw_payloads]
    _validate_campaign(cells)
    policy_freeze_audit = (
        _validate_policy_freeze_artifact(
            args.policy_freeze_artifact,
            formal_spec["policy_freeze_artifact_sha256"],
            cells[0],
        )
        if args.policy_freeze_artifact is not None
        else None
    )
    analysis = _build_analysis(cells, planned_seeds, planned_ssus)
    _apply_formal_campaign_status(
        analysis,
        cells,
        planned_seeds,
        planned_ssus,
        formal_spec,
        local_source_audit,
        policy_freeze_audit,
        campaign_spec_binding_audit,
    )
    if local_source_audit is not None:
        ending_source_audit = _validate_local_source_closure(
            raw_payloads, args.source_root, args.source_archive
        )
        _require(
            ending_source_audit == local_source_audit,
            "local source/data/archive closure changed during analysis",
        )
    if formal_spec is not None:
        _require(
            _read_formal_campaign_spec(args.campaign_spec) == formal_spec,
            "formal campaign spec changed during analysis",
        )
    if policy_freeze_audit is not None:
        _require(
            _validate_policy_freeze_artifact(
                args.policy_freeze_artifact,
                formal_spec["policy_freeze_artifact_sha256"],
                cells[0],
            )
            == policy_freeze_audit,
            "policy freeze artifact changed during analysis",
        )
    _require(
        _file_sha256(Path(__file__).resolve()) == analyzer_source_start,
        "npu128_scale_analysis.py changed during analysis",
    )
    analysis["local_source_and_data_audit"] = local_source_audit
    analysis["policy_freeze_audit"] = policy_freeze_audit
    analysis["campaign_spec_binding_audit"] = campaign_spec_binding_audit
    final_output_dir = (
        (
            args.output_dir
            if args.output_dir is not None
            else Path(
                "results/ms_scale_control/"
                "npu128_scale_analysis_128npu_"
                f"backing_per_npu{cells[0]['backing']}"
            )
        )
        .expanduser()
        .resolve()
    )
    _require(
        not final_output_dir.exists(),
        f"refusing to overwrite an existing analysis directory: {final_output_dir}",
    )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = final_output_dir.with_name(
        f".{final_output_dir.name}.{os.getpid()}.staging"
    )
    _require(
        not output_dir.exists(),
        f"stale analysis staging directory exists: {output_dir}",
    )
    output_dir.mkdir()
    _write_tables(output_dir, analysis)
    _plot_quality(output_dir, analysis, planned_seeds)
    _plot_paired(output_dir, analysis)
    _plot_control_stationarity(output_dir, analysis)
    report = _report_text(cells, analysis, planned_seeds, planned_ssus)
    _atomic_write_text(output_dir / "report.md", report)
    merged = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis": "npu128_scale_analysis_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": analysis["report_ready"],
        "complete_scope": "formal_report_ready",
        "plan_complete": analysis["plan_complete"],
        "formal_complete": analysis["formal_complete"],
        "report_ready": analysis["report_ready"],
        "campaign_spec_binding_verified": analysis["campaign_spec_binding_verified"],
        "policy_freeze_verified": analysis["policy_freeze_verified"],
        "local_source_root_verified": analysis["local_source_root_verified"],
        "source_archive_verified": analysis["source_archive_verified"],
        "source_archive_matches_campaign": analysis["source_archive_matches_campaign"],
        "analysis_status": analysis["analysis_status"],
        "formal_blockers": analysis["formal_blockers"],
        "watermark": None
        if analysis["report_ready"]
        else "CALIBRATION/PARTIAL — NOT A FORMAL REPORT",
        "planned_seeds": list(planned_seeds),
        "planned_ssus": list(planned_ssus),
        "roles": list(ROLE_ORDER),
        "expected_result_count": len(analysis["expected"]),
        "result_count": len(analysis["rows"]),
        "missing_results": [list(key) for key in analysis["missing"]],
        "source_fingerprint": cells[0]["source_fingerprint"],
        "definition_fingerprint": cells[0]["definition_fingerprint"],
        "num_npu": NUM_NPU,
        "backing_requests_per_npu": cells[0]["backing"],
        "total_assignment_count": NUM_NPU * cells[0]["backing"],
        "data_authentication": cells[0]["authentication"],
        "formal_campaign_spec": formal_spec,
        "campaign_spec_binding_audit": campaign_spec_binding_audit,
        "local_source_and_data_audit": local_source_audit,
        "policy_freeze_audit": policy_freeze_audit,
        "slo_definitions": {
            "ideal_ttft_ms": (
                "16 * authenticated_catalog[profile_key].per_layer_compute_us / 1000"
            ),
            "classification": "ttft_ms <= alpha * ideal_ttft_ms + 1e-12",
            "alphas": [PRIMARY_ALPHA, SENSITIVITY_ALPHA],
            "primary_aggregation": "equal-NPU mean of per-NPU request pass fractions",
            "request_weighted_reported_as_secondary": True,
            "catalog_third_value_used_as_ideal_ttft": False,
        },
        "analysis_source_fingerprint": _file_sha256(Path(__file__).resolve()),
        "stationarity_thresholds": {
            "rule_version": STATIONARITY_RULE_VERSION,
            "minimum_blocks": MIN_STATIONARITY_BLOCKS,
            "util_half_delta_limit_pp": UTIL_HALF_DELTA_LIMIT_PP,
            "util_projected_trend_limit_pp": UTIL_TREND_PROJECTED_LIMIT_PP,
            "fleet_served_half_relative_limit": SERVED_HALF_RELATIVE_LIMIT,
            "queue_growth_rule": "positive Theil-Sen slope AND >=75% nondecreasing boundary steps AND net growth > authenticated 128-NPU one-layer burst bound",
        },
        "input_shards": [
            {
                "path": cell["path"],
                "sha256": cell["sha256"],
                "seed": cell["seed"],
                "num_ssu": cell["num_ssu"],
                "config_fingerprint": cell["config_fingerprint"],
                "shard_schema_version": cell["shard_schema_version"],
                "campaign_spec_sha256": cell["campaign_spec_sha256"],
                "num_npu": NUM_NPU,
                "backing_requests_per_npu": cell["backing"],
                "total_assignment_count": NUM_NPU * cell["backing"],
                "runtime": cell["root_runtime"],
                "root_process_runtime": cell["root_process_runtime"],
                "worker_peak_rss_bytes_by_role": {
                    role: cell["rows"][role]["worker_peak_rss_bytes"]
                    for role in ROLE_ORDER
                },
            }
            for cell in sorted(cells, key=lambda item: (item["seed"], item["num_ssu"]))
        ],
        "run_metrics": [
            _public_run(analysis["rows"][key])
            for key in sorted(
                analysis["rows"],
                key=lambda key: (key[0], key[1], ROLE_ORDER.index(key[2])),
            )
        ],
        "curve_summary": analysis["curve_summary"],
        "seed_paired_deltas": analysis["paired_seed_rows"],
        "paired_delta_summary": analysis["paired_summary"],
    }
    _atomic_write_json(output_dir / "results.json", merged)
    expected_artifacts = {
        "run_metrics.csv",
        "stationarity.csv",
        "slo_alpha_sensitivity.csv",
        "seed_paired_deltas.csv",
        "curve_summary.csv",
        "paired_delta_summary.csv",
        "cohort_breakdown.csv",
        "quality_curve.png",
        "paired_gain_vs_baseline.png",
        "control_cost_and_stationarity.png",
        "report.md",
        "results.json",
    }
    _require(
        {path.name for path in output_dir.iterdir()} == expected_artifacts
        and all((output_dir / name).is_file() for name in expected_artifacts),
        "staged analysis artifact set is incomplete or contains unexpected files",
    )
    if local_source_audit is not None:
        _require(
            _validate_local_source_closure(
                raw_payloads, args.source_root, args.source_archive
            )
            == local_source_audit,
            "local source/data/archive closure changed while writing analysis",
        )
    if formal_spec is not None:
        _require(
            _read_formal_campaign_spec(args.campaign_spec) == formal_spec,
            "formal campaign spec changed while writing analysis",
        )
    if policy_freeze_audit is not None:
        _require(
            _validate_policy_freeze_artifact(
                args.policy_freeze_artifact,
                formal_spec["policy_freeze_artifact_sha256"],
                cells[0],
            )
            == policy_freeze_audit,
            "policy freeze artifact changed while writing analysis",
        )
    _require(
        _file_sha256(Path(__file__).resolve()) == analyzer_source_start,
        "npu128_scale_analysis.py changed while writing analysis",
    )
    _require(
        not final_output_dir.exists(),
        f"analysis output target appeared during staging: {final_output_dir}",
    )
    output_dir.rename(final_output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(final_output_dir),
                "plan_complete": analysis["plan_complete"],
                "formal_complete": analysis["formal_complete"],
                "report_ready": analysis["report_ready"],
                "campaign_spec_binding_verified": analysis[
                    "campaign_spec_binding_verified"
                ],
                "analysis_status": analysis["analysis_status"],
                "validated_cells": len(cells),
                "result_count": len(analysis["rows"]),
                "expected_result_count": len(analysis["expected"]),
                "missing_count": len(analysis["missing"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.require_report_ready and not analysis["report_ready"]:
        return 4
    if args.require_formal and not analysis["formal_complete"]:
        return 3
    return 2 if args.require_complete and not analysis["plan_complete"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        raise SystemExit(f"validation error: {error}") from error
