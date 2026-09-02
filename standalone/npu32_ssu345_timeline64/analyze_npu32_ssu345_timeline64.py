#!/usr/bin/env python3
"""Strict 64-second timeline analysis for the 32-NPU / SSU-3,4,5 campaign.

This analyzer is intentionally independent of the simulator.  It accepts one
or more runner JSON shards, validates the complete 3-strategy x 3-SSU matrix,
reconstructs exact 500-ms NPU/SSD/link service from left-limit boundaries, and
writes portable CSV/JSON/Markdown evidence plus publication-style figures.

The three bandwidth concepts are kept separate throughout:

* controller demand is a workload-derived controller input;
* installed CIR is an arbitration guarantee (Adaptive only), never throughput;
* interval-average attributed SSD service is differenced only from the v3
  stable completed-command-plus-immutable-active-prefix counter; direct
  physical outstanding bytes are never inferred from a fragmented settle
  counter;
* NPU-link delivery is an independently differenced physical counter and is
  the bandwidth that actually reached an NPU.

Run ``python analyze_npu32_ssu345_timeline64.py --self-test`` to exercise the
full validator and output pipeline against a deterministic synthetic schema.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from adaptive_admission_scheme_b_v2_1 import (
    allocate_adaptive_admission_grants,
    replay_admission_selection,
)
from authenticated_workload_inputs import load_authenticated_bw_table
from continuous_batch_sim import (
    continuous_batch_input_fingerprint,
    requests_from_continuous_prefill_workload,
)
from continuous_prefill_client import static_qos_config
from policy_logic import (
    PathPressureSnapshot,
    QoSHardwareView,
    category_path_ids,
    cold_start_hybrid_path_id,
    pressure_aware_path_ids,
    hardware_view,
)
from random_steady_state_workload import (
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path(
    "results/ms_scale_control/npu32_ssu345_timeline64_analysis"
)
CASES = (
    "baseline",
    "layer_once_ttl_5ms",
    "adaptive_t0_i100ms",
)
SOURCE_FINGERPRINT_NAMESPACE = b"ms-scale-control-source:v1\0"
CONFIG_FINGERPRINT_NAMESPACE = b"ms-scale-control-config:v1\0"
DEFINITION_FINGERPRINT_NAMESPACE = b"ms-scale-control-definition:v1\0"
CASE_FINGERPRINT_NAMESPACE = b"ms-scale-control-case:v1\0"


def _legacy_case(
    name: str,
    family: str,
    kind: str,
    pressure_ttl_ms: float = 0.0,
    cir_write_threshold_gbps: float = 0.0,
    min_interval_ms: float = 0.0,
) -> dict:
    return {
        "name": name,
        "family": family,
        "kind": kind,
        "pressure_ttl_ms": pressure_ttl_ms,
        "cir_write_threshold_gbps": cir_write_threshold_gbps,
        "min_interval_ms": min_interval_ms,
    }


LEGACY32_CASE_SPECS = (
    _legacy_case("baseline", "baseline", "baseline"),
    _legacy_case("layer_once_ttl_0ms", "ttl", "layer_once", 0.0),
    _legacy_case("layer_once_ttl_0p25ms", "ttl", "layer_once", 0.25),
    _legacy_case("layer_once_ttl_1ms", "ttl", "layer_once", 1.0),
    _legacy_case("layer_once_ttl_2ms", "ttl", "layer_once", 2.0),
    _legacy_case("layer_once_ttl_5ms", "ttl", "layer_once", 5.0),
    _legacy_case(
        "adaptive_t0_i25ms", "threshold_interval", "adaptive", 0.0, 0.0, 25.0
    ),
    _legacy_case(
        "adaptive_t0p005_i25ms", "threshold", "adaptive", 0.0, 0.005, 25.0
    ),
    _legacy_case(
        "adaptive_t0p01_i25ms", "threshold", "adaptive", 0.0, 0.01, 25.0
    ),
    _legacy_case(
        "adaptive_t0p02_i25ms", "threshold", "adaptive", 0.0, 0.02, 25.0
    ),
    _legacy_case(
        "adaptive_t0p05_i25ms", "threshold", "adaptive", 0.0, 0.05, 25.0
    ),
    _legacy_case(
        "adaptive_t0_i50ms", "interval", "adaptive", 0.0, 0.0, 50.0
    ),
    _legacy_case(
        "adaptive_t0_i100ms", "interval", "adaptive", 0.0, 0.0, 100.0
    ),
    _legacy_case(
        "adaptive_t0_i200ms", "interval", "adaptive", 0.0, 0.0, 200.0
    ),
)
LEGACY32_DEFINITION_CANONICAL = {
    "key": "legacy32",
    "experiment_name": "32npu_ms_scale_ssu_qos_control_v1",
    "num_npu": 32,
    "n_layers": 16,
    "batch_size": 1,
    "default_ssus": [6, 10, 18],
    "cases": list(LEGACY32_CASE_SPECS),
    "default_requests_per_npu": 64,
    "default_measurement_ms": 2000.0,
    "adaptive": {
        "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
        "explicit_spill_threshold": 0.75,
        "target_ratio": 0.52,
        "required_ratio": 0.5,
        "background_reserve_fraction": 0.05,
    },
    "report_roles": [],
    "require_single_ssu_simulation": False,
}
SSU_COUNTS = (3, 4, 5)
CASE_LABELS = {
    "baseline": "Baseline",
    "layer_once_ttl_5ms": "Layer once: TTL 5 ms",
    "adaptive_t0_i100ms": "Adaptive: event-driven, min 100 ms",
}
CASE_COLORS = {
    "baseline": "#1f77b4",
    "layer_once_ttl_5ms": "#d62728",
    "adaptive_t0_i100ms": "#8c564b",
}
CASE_MARKERS = {
    "baseline": "s",
    "layer_once_ttl_5ms": "D",
    "adaptive_t0_i100ms": "x",
}
NUM_NPU = 32
N_LAYERS = 16
BATCH_SIZE = 1
MEASUREMENT_MS = 64_000.0
BLOCK_MS = 500.0
BLOCK_COUNT = 128
BOUNDARY_COUNT = 129
SSD_CAP_GBPS = 40.0
NPU_LINK_CAP_GBPS = 50.0
HORIZONS_S = (0.5, 1.0, 2.0, 3.0, 8.0, 16.0, 32.0, 64.0)
SLO_ALPHAS = (1.5, 2.0)
BOUNDARY_SEMANTICS = (
    "read-only left-limit snapshot before workload events at the same time"
)
CONTROL_WINDOW = "half-open [measurement_start_ms, measurement_end_ms)"
TIMELINE_SCHEMA = "steady_timeline_boundary_v3"
SSD_ACCOUNTING_SCHEMA = "steady_ssd_accounting_residuals_v1"
SSD_SERVICE_ABSOLUTE_TOLERANCE_GB = 1e-8
PAIR_FIELDS_WITHIN_SSU = (
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
CROSS_SSU_INPUT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
)
REQUIRED_TIMELINE_INVARIANTS = frozenset(
    {
        "timeline_snapshot_shapes",
        "timeline_values_nonnegative",
        "timeline_service_attribution",
        "timeline_independent_queue_attribution",
        "timeline_io_stage_conservation",
        "timeline_remaining_work_bounds",
        "timeline_cumulative_monotonic",
        "timeline_ssd_link_queue_conservation",
        "timeline_compute_inventory_conservation",
        "timeline_dispatch_replay_exact",
        "timeline_dispatch_probe_nonempty",
        "timeline_route_probe_nonempty",
        "timeline_state_duration_partition",
        "timeline_state_compute_matches_utilization",
        "timeline_block_state_duration_partition",
        "timeline_block_state_compute_matches_stationarity",
        "timeline_carry_in_definition_exact",
        "timeline_carry_in_unique_per_npu",
        "timeline_carry_in_batch_size_one",
        "timeline_carry_in_layer_shape_exact",
        "timeline_carry_in_request_identity_exact",
        "timeline_carry_in_compute_budget_exact",
        "timeline_carry_in_interval_closure",
    }
)
TOL = 1e-8
MAX_PUBLIC_FILE_BYTES = 50 * 1024 * 1024
EXPECTED_PATH_ABI = {
    "path_count": 256,
    "group_count": 8,
    "paths_per_group": 32,
    "max_npu": 128,
    "assigned_count": 32,
    "assigned_unique": 32,
    "assigned_min": 16,
    "assigned_max": 243,
    "path_zero_reserved": True,
    "assigned_paths_sha256": (
        "529078ba90ec4ca915b24039deb5b18454883ea20bc6df074d0343d8df44ad3b"
    ),
}
SCIENTIFIC_INPUT_AUTHENTICATION_KEYS = (
    "source",
    "source_sha256",
    "catalog_hash",
    "table_fingerprint",
    "profile_count",
)


class AnalysisError(ValueError):
    """Raised when raw data cannot support the requested conclusion."""


@dataclass(frozen=True)
class CaseData:
    case: str
    num_ssu: int
    source_path: Path
    payload: dict
    row: dict
    summary: dict
    boundaries: tuple[dict, ...]
    blocks: tuple[dict, ...]
    requests: tuple[dict, ...]

    @property
    def key(self) -> tuple[int, str]:
        return self.num_ssu, self.case

    @property
    def measurement_start_ms(self) -> float:
        return float(self.summary["measurement_start_ms"])


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _number(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context}: expected a number") from error
    _require(math.isfinite(result), f"{context}: expected a finite number")
    return result


def _integer(value: object, context: str) -> int:
    _require(not isinstance(value, bool), f"{context}: expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context}: expected an integer") from error
    _require(float(result) == float(value), f"{context}: expected an integer")
    return result


def _optional_number(value: object, context: str) -> float | None:
    return None if value is None else _number(value, context)


def _close(left: float, right: float, *, tolerance: float = TOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=tolerance)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _runner_canonical_hash(value: object, namespace: bytes) -> str:
    """Reproduce ``ms_scale_control_experiment._canonical_hash`` exactly."""
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _is_lower_hex_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transitive_runner_sources(root: Path = PROJECT_ROOT) -> tuple[str, ...]:
    """Independently reproduce the runner's root-local Python import closure."""
    pending = ["ms_scale_control_experiment.py"]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = root / name
        _require(path.is_file(), f"source closure member is missing: {name}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise AnalysisError(f"cannot parse source closure member {name}: {error}") from error
        seen.add(name)
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                modules.append(node.module)
        for module in modules:
            candidate = module.split(".", 1)[0] + ".py"
            if (root / candidate).is_file() and candidate not in seen:
                pending.append(candidate)
    return tuple(sorted(seen))


def _current_runner_source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    names = _transitive_runner_sources(root) + ("data",)
    return {name: _sha256(root / name) for name in names}


def _validate_source_manifest_against_checkout(
    manifest: Mapping[str, object],
    label: str,
    *,
    root: Path = PROJECT_ROOT,
) -> None:
    root_resolved = root.resolve()
    expected_names = set(_transitive_runner_sources(root)) | {"data"}
    _require(
        set(manifest) == expected_names,
        f"{label}: source manifest key set differs from current runner closure",
    )
    for name, reported_digest in manifest.items():
        relative = Path(name)
        _require(
            not relative.is_absolute()
            and name not in ("", ".")
            and ".." not in relative.parts,
            f"{label}: unsafe source manifest path {name!r}",
        )
        candidate = (root_resolved / relative).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError as error:
            raise AnalysisError(
                f"{label}: source manifest path escapes checkout: {name!r}"
            ) from error
        _require(candidate.is_file(), f"{label}: source file missing: {name}")
        _require(
            _sha256(candidate) == reported_digest,
            f"{label}: checkout bytes differ from recorded source: {name}",
        )


def _publication_path(path: Path) -> str:
    """Never publish a private workspace prefix or hostname-bearing path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {_publication_path(path)}: {error}") from error
    _require(isinstance(value, dict), f"{_publication_path(path)}: top level must be an object")
    return value


def _vector(
    value: object,
    size: int,
    context: str,
    *,
    integer: bool = False,
    nonnegative: bool = True,
) -> list[float] | list[int]:
    _require(isinstance(value, (list, tuple)), f"{context}: expected a vector")
    _require(len(value) == size, f"{context}: expected {size} values, got {len(value)}")
    convert = _integer if integer else _number
    result = [convert(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if nonnegative:
        _require(min(result, default=0.0) >= -TOL, f"{context}: negative value")
    return result


def _matrix(
    value: object,
    rows: int,
    columns: int,
    context: str,
    *,
    integer: bool = False,
    nonnegative: bool = True,
    allow_none: bool = False,
) -> list[list[float]] | list[list[int]] | None:
    if value is None and allow_none:
        return None
    _require(isinstance(value, (list, tuple)), f"{context}: expected a matrix")
    _require(len(value) == rows, f"{context}: expected {rows} rows")
    return [
        list(
            _vector(
                row,
                columns,
                f"{context}[{index}]",
                integer=integer,
                nonnegative=nonnegative,
            )
        )
        for index, row in enumerate(value)
    ]


def _vectors_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))


def _matrices_close(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> bool:
    return len(left) == len(right) and all(
        _vectors_close(a, b) for a, b in zip(left, right)
    )


def _flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _safe_mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _case_spec(spec: Mapping[str, object], case: str) -> dict:
    rows = spec.get("cases")
    _require(isinstance(rows, list), "experiment_spec.cases missing")
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == case]
    _require(len(matches) == 1, f"experiment_spec: expected one {case} definition")
    return matches[0]


def _discover_payload_paths(inputs: Sequence[Path]) -> list[Path]:
    _require(inputs, "at least one --input PATH is required")
    paths: list[Path] = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        _require(path.exists(), f"input does not exist: {_publication_path(path)}")
        if path.is_file():
            paths.append(path)
            continue
        candidates = sorted(path.rglob("*.json"))
        _require(candidates, f"input directory has no JSON: {_publication_path(path)}")
        paths.extend(candidate for candidate in candidates if candidate.is_file())
    unique = list(dict.fromkeys(paths))
    _require(unique, "no JSON input files discovered")
    return unique


def _runtime_signature(runtime: object, context: str) -> dict:
    _require(isinstance(runtime, dict), f"{context}: runtime provenance missing")
    # Hostname participates in the in-memory equality gate requested for this
    # formal campaign.  Only a one-way hash of this signature is ever exported;
    # PID, CPU count, and memory readings remain excluded.
    fields = (
        "hostname",
        "python",
        "python_implementation",
        "numpy",
        "numpy_blas_identity",
        "platform",
        "multiprocessing_start_method",
        "thread_limit_environment",
    )
    result = {field: runtime.get(field) for field in fields}
    _require(
        result["hostname"] is not None and result["python"] is not None,
        f"{context}: hostname/Python version missing",
    )
    return result


def _runtime_merge_identity(runtime: object, context: str) -> dict:
    """Match the runner's cross-host scientific runtime merge identity."""
    _require(isinstance(runtime, dict), f"{context}: runtime provenance missing")
    blas = runtime.get("numpy_blas_identity")
    _require(isinstance(blas, dict), f"{context}: BLAS identity missing")
    identity = {
        "python_implementation": runtime.get("python_implementation"),
        "python_version": runtime.get("python"),
        "numpy_version": runtime.get("numpy"),
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "openblas_configuration": blas.get("openblas_configuration"),
    }
    _require(
        all(identity[field] is not None for field in (
            "python_implementation",
            "python_version",
            "numpy_version",
            "blas_name",
            "blas_version",
        )),
        f"{context}: runtime merge identity is incomplete",
    )
    return identity


def _legacy_definition_fingerprint() -> str:
    return _runner_canonical_hash(
        LEGACY32_DEFINITION_CANONICAL,
        DEFINITION_FINGERPRINT_NAMESPACE,
    )


@lru_cache(maxsize=1)
def _current_scientific_input_authentication() -> dict:
    _table, authentication = load_authenticated_bw_table(NUM_NPU)
    return {
        key: authentication[key] for key in SCIENTIFIC_INPUT_AUTHENTICATION_KEYS
    }


@lru_cache(maxsize=1)
def _current_formal_schedule_fingerprints() -> dict:
    table, _authentication = load_authenticated_bw_table(NUM_NPU)
    schedule = build_steady_state_profile_schedule(
        table,
        mode="iid_uniform_profile_catalog_v1",
        seed=42,
        num_npu=NUM_NPU,
        requests_per_npu=256,
    )
    return {
        **schedule.as_fingerprint_dict(),
        "prefix_32_assignment_hash": schedule.prefix_32_assignment_hash,
        "full_assignment_hash": schedule.full_assignment_hash,
    }


@lru_cache(maxsize=len(SSU_COUNTS))
def _current_materialized_analysis_input(
    num_ssu: int,
) -> tuple[dict, dict[int, dict]]:
    _require(num_ssu in SSU_COUNTS, f"unsupported materialized SSU count {num_ssu}")
    table, _authentication = load_authenticated_bw_table(NUM_NPU)
    schedule = build_steady_state_profile_schedule(
        table,
        mode="iid_uniform_profile_catalog_v1",
        seed=42,
        num_npu=NUM_NPU,
        requests_per_npu=256,
    )
    workload = prepare_random_steady_state_workload(
        table,
        schedule=schedule,
        num_ssu=num_ssu,
        n_layers=N_LAYERS,
    )
    requests = requests_from_continuous_prefill_workload(workload)
    fingerprints = {
        "workload": workload.workload_hash,
        "placement": workload.placement_hash,
        "trace": workload.trace_hash,
        "simulator": continuous_batch_input_fingerprint(requests),
    }
    metadata: dict[int, dict] = {}
    for request in workload.requests:
        placement = workload.placement_by_request[int(request.request_id)]
        placement_blocks_by_ssu: dict[int, list[tuple[int, float]]] = {
            ssu_id: [] for ssu_id in range(num_ssu)
        }
        for block_index, (ssu_id, size_gb) in enumerate(placement):
            placement_blocks_by_ssu[int(ssu_id)].append(
                (block_index, float(size_gb))
            )
        metadata[int(request.request_id)] = {
            "npu_id": int(request.npu_id),
            "sequence": int(request.stream_id),
            "category": str(request.category),
            "profile_id": f"{int(request.seq_len_k)},{int(request.nql)}",
            "profile_name": (
                f"seq_len_k={int(request.seq_len_k)},nql={int(request.nql)}"
            ),
            "profile_key": (int(request.seq_len_k), int(request.nql)),
            "raw_demand_gbps": float(request.required_bw_input_gbps),
            "ideal_ttft_ms": N_LAYERS * float(request.per_layer_us) / 1000.0,
            "per_layer_work_gb_by_ssu": tuple(
                float(value) for value in request.work_by_ssu_gb
            ),
            "placement_blocks_by_ssu": {
                ssu_id: tuple(blocks)
                for ssu_id, blocks in placement_blocks_by_ssu.items()
            },
        }
    return fingerprints, metadata


def _current_materialized_input_fingerprints(num_ssu: int) -> dict:
    return _current_materialized_analysis_input(num_ssu)[0]


def _current_materialized_request_metadata(num_ssu: int) -> dict[int, dict]:
    return _current_materialized_analysis_input(num_ssu)[1]


def _authenticated_remaining_layer_count(
    *,
    num_ssu: int,
    npu_id: int,
    request_id: int,
    remaining_compute_ms: float,
    remaining_work_gb_by_ssu: Sequence[float],
    context: str,
) -> int:
    """Bind a controller inventory to one authenticated 1..16-layer slice."""

    metadata = _current_materialized_request_metadata(num_ssu).get(request_id)
    _require(
        metadata is not None and int(metadata["npu_id"]) == npu_id,
        f"{context}: controller inventory request is not authenticated to its NPU",
    )
    _require(
        len(remaining_work_gb_by_ssu) == num_ssu,
        f"{context}: controller inventory SSU shape mismatch",
    )
    per_layer_compute_ms = float(metadata["ideal_ttft_ms"]) / N_LAYERS
    matching_layer_counts = [
        layer_count
        for layer_count in range(1, N_LAYERS + 1)
        if _close(
            remaining_compute_ms,
            layer_count * per_layer_compute_ms,
            tolerance=2e-9,
        )
        and all(
            _close(
                float(remaining_work_gb_by_ssu[ssu_id]),
                layer_count
                * float(metadata["per_layer_work_gb_by_ssu"][ssu_id]),
                tolerance=2e-8,
            )
            for ssu_id in range(num_ssu)
        )
    ]
    _require(
        len(matching_layer_counts) == 1,
        f"{context}: remaining work/compute is not an authenticated integer layer inventory",
    )
    return matching_layer_counts[0]


def _expected_experiment_spec() -> dict:
    schedule = _current_formal_schedule_fingerprints()
    authentication = _current_scientific_input_authentication()
    return {
        "schema_version": 3,
        "experiment": LEGACY32_DEFINITION_CANONICAL["experiment_name"],
        "definition": "legacy32",
        "definition_fingerprint": _legacy_definition_fingerprint(),
        "num_npu": NUM_NPU,
        "scale_semantics": {
            "num_npu": NUM_NPU,
            "naked_128_means": None,
            "backing_requests_per_npu": 256,
            "total_assignment_count": NUM_NPU * 256,
            "rule": (
                "NPU count is fixed by --definition; --requests-per-npu is "
                "finite input backing only"
            ),
        },
        "campaign_spec_sha256": None,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssu_list": list(LEGACY32_DEFINITION_CANONICAL["default_ssus"]),
        "cases": [dict(case) for case in LEGACY32_CASE_SPECS],
        "report_roles": {},
        "workload": {
            "mode": "iid_uniform_profile_catalog_v1",
            "seed": 42,
            "requests_per_npu": 256,
            **schedule,
            "sampling": "IID uniform with replacement over all 84 data profiles",
            "per_npu_streams": "independent and prefix-stable",
            "scientific_prefix_requests_per_npu": 32,
            "backing_prefix_reason": (
                "256 requests preserve the first-32 assignment; any suffix "
                "only prevents full-load queue exhaustion during measurement/drain"
            ),
            "authentication": authentication,
        },
        "steady_state": {
            "seed": 42,
            "requests_per_npu": 256,
            "warmup_requests_per_npu": 8,
            "settle_ms": 500.0,
            "measurement_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "slo_alpha": 2.0,
            "timeline_diagnostics": True,
            "timeline_dispatch_probe_ms": 50.0,
            "timeline_dispatch_probe_limit": 10_000,
            "calibration_mode": False,
        },
        "adaptive": {
            **LEGACY32_DEFINITION_CANONICAL["adaptive"],
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_LINK_CAP_GBPS,
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash reused across all 16 layers",
        "measurement_cost_scope": "true SSU pressure-table reads and CIR writes",
        "pairing_scope": (
            "all strategies share the exact finite schedule/placement/simulator "
            "input within each SSU; wall-time measurement cohorts may differ in "
            "closed loop and receive a separate membership fingerprint"
        ),
        "diagnostics": {
            "control_evaluations": (
                "reported but not required to equal the threshold=0 closed-loop "
                "anchor because filtered CIR writes can change event timing"
            ),
            "source_stable_false": "invalidates the shard",
        },
        "source_files": list(_transitive_runner_sources(PROJECT_ROOT)) + ["data"],
    }


def _case_fingerprint(
    case_spec: Mapping[str, object],
    num_ssu: int,
    source_fingerprint: str,
    config_fingerprint: str,
) -> str:
    return _runner_canonical_hash(
        {
            "case": dict(case_spec),
            "num_ssu": int(num_ssu),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        CASE_FINGERPRINT_NAMESPACE,
    )


def _validate_experiment_spec(spec: object, label: str) -> None:
    _require(isinstance(spec, dict), f"{label}: experiment_spec missing")
    _require(
        spec == _expected_experiment_spec(),
        f"{label}: experiment_spec differs from the complete local runner contract",
    )
    _require(
        _integer(spec.get("schema_version"), f"{label}.schema_version") == 3,
        f"{label}: experiment spec schema mismatch",
    )
    _require(
        spec.get("experiment")
        == LEGACY32_DEFINITION_CANONICAL["experiment_name"]
        and spec.get("definition") == "legacy32",
        f"{label}: legacy32 experiment identity mismatch",
    )
    _require(
        spec.get("campaign_spec_sha256") is None,
        f"{label}: formal campaign unexpectedly uses a campaign-spec override",
    )
    expected_definition_fingerprint = _legacy_definition_fingerprint()
    _require(
        spec.get("definition_fingerprint") == expected_definition_fingerprint,
        f"{label}: spec definition fingerprint is not runner-derived",
    )
    _require(_integer(spec.get("num_npu"), f"{label}.num_npu") == NUM_NPU, f"{label}: expected 32 NPUs")
    _require(_integer(spec.get("n_layers"), f"{label}.n_layers") == N_LAYERS, f"{label}: expected 16 layers")
    _require(_integer(spec.get("batch_size"), f"{label}.batch_size") == BATCH_SIZE, f"{label}: expected batch size 1")
    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), f"{label}: steady_state spec missing")
    expected_steady = {
        "seed": 42,
        "requests_per_npu": 256,
        "warmup_requests_per_npu": 8,
        "settle_ms": 500.0,
        "measurement_ms": MEASUREMENT_MS,
        "block_ms": BLOCK_MS,
        "slo_alpha": 2.0,
        "timeline_diagnostics": True,
        "timeline_dispatch_probe_ms": 50.0,
        "timeline_dispatch_probe_limit": 10_000,
        "calibration_mode": False,
    }
    _require(
        steady == expected_steady,
        f"{label}: formal 64-second steady-state spec is not exact",
    )
    timeline_flag = steady.get("timeline_diagnostics", spec.get("diagnostics", {}).get("timeline_diagnostics") if isinstance(spec.get("diagnostics"), dict) else None)
    _require(timeline_flag is True, f"{label}: timeline diagnostics were not enabled")
    workload = spec.get("workload")
    _require(isinstance(workload, dict), f"{label}: workload spec missing")
    backing = _integer(workload.get("requests_per_npu"), f"{label}.requests_per_npu")
    _require(
        backing == 256
        and workload.get("seed") == 42
        and workload.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{label}: formal workload mode/seed/backing mismatch",
    )
    _require(
        spec.get("cross_request_layer0_prefetch") is True,
        f"{label}: cross-request layer-0 prefetch must be enabled",
    )
    _require(
        spec.get("placement")
        == "token-block ring hash reused across all 16 layers",
        f"{label}: placement definition mismatch",
    )
    expected_sources = list(_transitive_runner_sources(PROJECT_ROOT)) + ["data"]
    _require(
        spec.get("source_files") == expected_sources,
        f"{label}: experiment source_files differs from the runner closure",
    )
    _require(
        workload.get("authentication")
        == _current_scientific_input_authentication(),
        f"{label}: workload input authentication differs from current tracked data",
    )
    expected_schedule = _current_formal_schedule_fingerprints()
    _require(
        all(workload.get(field) == value for field, value in expected_schedule.items()),
        f"{label}: config-authenticated workload schedule fingerprints mismatch",
    )
    _require(
        workload["authentication"]["catalog_hash"] == workload["catalog"],
        f"{label}: authenticated catalog hash differs from schedule catalog",
    )

    _require(
        spec.get("default_ssu_list")
        == LEGACY32_DEFINITION_CANONICAL["default_ssus"],
        f"{label}: legacy32 default SSU definition mismatch",
    )
    _require(
        spec.get("cases") == list(LEGACY32_CASE_SPECS),
        f"{label}: legacy32 case definition table mismatch",
    )
    _require(
        spec.get("report_roles") == {},
        f"{label}: legacy32 report roles mismatch",
    )

    baseline = _case_spec(spec, "baseline")
    layer = _case_spec(spec, "layer_once_ttl_5ms")
    adaptive = _case_spec(spec, "adaptive_t0_i100ms")
    _require(baseline.get("kind") == "baseline", "baseline kind mismatch")
    _require(layer.get("kind") == "layer_once", "layer-once kind mismatch")
    _require(adaptive.get("kind") == "adaptive", "Adaptive kind mismatch")
    _require(_close(_number(layer.get("pressure_ttl_ms"), "layer TTL"), 5.0), "layer-once TTL must be 5 ms")
    _require(_close(_number(adaptive.get("min_interval_ms"), "Adaptive interval"), 100.0), "Adaptive interval must be 100 ms")
    adaptive_spec = spec.get("adaptive")
    _require(isinstance(adaptive_spec, dict), f"{label}: adaptive spec missing")
    expected_adaptive = {
        **LEGACY32_DEFINITION_CANONICAL["adaptive"],
        "ssd_cap_gbps": SSD_CAP_GBPS,
        "npu_cap_gbps": NPU_LINK_CAP_GBPS,
    }
    _require(
        adaptive_spec == expected_adaptive,
        f"{label}: legacy32 Adaptive definition mismatch",
    )
    _require(_close(_number(adaptive_spec.get("ssd_cap_gbps"), "SSD cap"), SSD_CAP_GBPS), "SSD cap mismatch")
    _require(_close(_number(adaptive_spec.get("npu_cap_gbps"), "NPU cap"), NPU_LINK_CAP_GBPS), "NPU-link cap mismatch")
    # required_ratio=0.5 is the alpha=2 feasibility target.  It does not mean
    # the controller reads TTFT deadline/slack; that is checked in each summary.
    if "required_ratio" in adaptive_spec:
        _require(_close(_number(adaptive_spec["required_ratio"], "required_ratio"), 0.5), "Adaptive required_ratio must be 0.5 (alpha=2 configuration)")


def _validate_request_rows(summary: dict, case_label: str) -> tuple[dict, ...]:
    rows = summary.get("request_rows")
    _require(isinstance(rows, list) and rows, f"{case_label}: request_rows missing")
    start = _number(summary.get("measurement_start_ms"), f"{case_label}.start")
    end = _number(summary.get("measurement_end_ms"), f"{case_label}.end")
    seen_ids: set[int] = set()
    seen_streams: set[tuple[int, int]] = set()
    counts = [0] * NUM_NPU
    primary_outcomes: list[list[bool]] = [[] for _ in range(NUM_NPU)]
    ttfts: list[float] = []
    num_ssu = _integer(summary.get("num_ssu"), f"{case_label}.num_ssu")
    authenticated_requests = _current_materialized_request_metadata(num_ssu)
    for index, row in enumerate(rows):
        context = f"{case_label}.request_rows[{index}]"
        _require(isinstance(row, dict), f"{context}: expected an object")
        request_id = _integer(row.get("request_id"), f"{context}.request_id")
        npu_id = _integer(row.get("npu_id"), f"{context}.npu_id")
        sequence = _integer(row.get("sequence"), f"{context}.sequence")
        _require(0 <= npu_id < NUM_NPU, f"{context}: NPU out of range")
        _require(request_id not in seen_ids, f"{context}: duplicate request ID")
        _require((npu_id, sequence) not in seen_streams, f"{context}: duplicate NPU stream sequence")
        expected_request = authenticated_requests.get(request_id)
        _require(
            expected_request is not None
            and npu_id == expected_request["npu_id"]
            and sequence == expected_request["sequence"]
            and row.get("category") == expected_request["category"]
            and row.get("profile_id") == expected_request["profile_id"]
            and row.get("profile_name") == expected_request["profile_name"],
            f"{context}: request identity/profile differs from authenticated materialized workload",
        )
        seen_ids.add(request_id)
        seen_streams.add((npu_id, sequence))
        admission = _number(row.get("admission_time_ms"), f"{context}.admission")
        completion = _number(row.get("completion_time_ms"), f"{context}.completion")
        ttft = _number(row.get("ttft_ms"), f"{context}.ttft")
        ideal = _number(row.get("ideal_ttft_ms"), f"{context}.ideal")
        _require(
            _close(ideal, float(expected_request["ideal_ttft_ms"])),
            f"{context}: ideal TTFT differs from authenticated request profile",
        )
        _require(start - TOL <= admission < end - TOL, f"{context}: admission is outside the half-open measurement window")
        _require(completion + TOL >= admission, f"{context}: completion precedes admission")
        _require(ideal > 0.0 and ttft >= ideal - 1e-6, f"{context}: invalid ideal/actual TTFT")
        _require(_close(completion - admission, ttft, tolerance=1e-6), f"{context}: TTFT endpoint mismatch")
        primary = ttft <= 2.0 * ideal + TOL
        _require(row.get("slo_met") is primary, f"{context}: primary alpha=2 outcome mismatch")
        barrier = _number(row.get("io_barrier_ms"), f"{context}.io_barrier_ms")
        error = _number(row.get("ttft_accounting_error_ms"), f"{context}.ttft_accounting_error_ms")
        _require(barrier >= -TOL, f"{context}: negative IO barrier")
        _require(_close(ttft - ideal - barrier, error, tolerance=2e-6), f"{context}: TTFT/compute/barrier accounting mismatch")
        _require(abs(error) <= 2e-6, f"{context}: nonzero TTFT accounting residual: {error}")

        layers = row.get("timeline_layers")
        _require(isinstance(layers, dict), f"{context}: per-layer timeline missing")
        layer_vectors = {}
        for field in (
            "io_start_time_ms",
            "io_ready_time_ms",
            "compute_start_ms",
            "compute_end_ms",
            "io_barrier_wait_ms",
        ):
            layer_vectors[field] = _vector(
                layers.get(field), N_LAYERS, f"{context}.timeline_layers.{field}"
            )
        work = _vector(
            layers.get("per_layer_work_gb_by_ssu"),
            num_ssu,
            f"{context}.timeline_layers.per_layer_work",
        )
        _require(
            any(value > 0.0 for value in work)
            and _vectors_close(
                work, expected_request["per_layer_work_gb_by_ssu"]
            ),
            f"{context}: per-layer IO work differs from authenticated placement",
        )
        barrier_sum = 0.0
        compute_sum = 0.0
        for layer in range(N_LAYERS):
            io_start = layer_vectors["io_start_time_ms"][layer]
            io_ready = layer_vectors["io_ready_time_ms"][layer]
            compute_start = layer_vectors["compute_start_ms"][layer]
            compute_end = layer_vectors["compute_end_ms"][layer]
            wait = layer_vectors["io_barrier_wait_ms"][layer]
            _require(io_ready + TOL >= io_start, f"{context}: layer {layer} IO ends before it starts")
            _require(compute_start + TOL >= io_ready, f"{context}: layer {layer} compute starts before IO readiness")
            _require(compute_end > compute_start, f"{context}: layer {layer} compute interval is empty")
            prior_end = (
                layer_vectors["compute_end_ms"][layer - 1]
                if layer
                else admission
            )
            _require(
                compute_start + TOL >= prior_end,
                f"{context}: compute layers overlap",
            )
            expected_wait = max(0.0, io_ready - prior_end)
            expected_compute_start = max(prior_end, io_ready)
            expected_layer_compute = ideal / N_LAYERS
            _require(
                _close(compute_start, expected_compute_start, tolerance=2e-6),
                f"{context}: layer {layer} contains an unexplained pre-compute gap",
            )
            _require(
                _close(
                    compute_end - compute_start,
                    expected_layer_compute,
                    tolerance=2e-6,
                ),
                f"{context}: layer {layer} compute duration differs from authenticated fixed per-layer compute",
            )
            _require(_close(wait, expected_wait, tolerance=2e-6), f"{context}: layer {layer} barrier mismatch")
            barrier_sum += wait
            compute_sum += compute_end - compute_start
        _require(_close(barrier_sum, barrier, tolerance=2e-6), f"{context}: layer barriers do not sum to request barrier")
        _require(
            _close(compute_sum, ideal, tolerance=2e-6),
            f"{context}: per-layer compute durations do not sum to ideal TTFT",
        )
        _require(
            _close(
                layer_vectors["compute_end_ms"][-1],
                completion,
                tolerance=2e-6,
            ),
            f"{context}: final layer compute end != request completion",
        )
        _require(
            _close(barrier_sum + compute_sum, ttft, tolerance=2e-6),
            f"{context}: barrier + compute does not close to TTFT",
        )

        counts[npu_id] += 1
        primary_outcomes[npu_id].append(primary)
        ttfts.append(ttft)

    reported_count = _integer(summary.get("measurement_request_count"), f"{case_label}.measurement_request_count")
    _require(reported_count == len(rows), f"{case_label}: request count mismatch")
    _require(
        counts == _vector(summary.get("request_counts_by_npu"), NUM_NPU, f"{case_label}.request_counts_by_npu", integer=True),
        f"{case_label}: per-NPU request counts mismatch",
    )
    _require(all(primary_outcomes), f"{case_label}: an NPU has no measured request")
    weighted = statistics.fmean(outcome for values in primary_outcomes for outcome in values)
    equal_npu = statistics.fmean(statistics.fmean(values) for values in primary_outcomes)
    _require(_close(weighted, _number(summary.get("request_weighted_slo_attainment"), f"{case_label}.weighted_slo")), f"{case_label}: weighted SLO mismatch")
    _require(_close(equal_npu, _number(summary.get("ttft_slo_attainment"), f"{case_label}.equal_npu_slo")), f"{case_label}: equal-NPU SLO mismatch")
    _require(_close(statistics.fmean(ttfts), _number(summary.get("mean_ttft_ms"), f"{case_label}.mean_ttft")), f"{case_label}: mean TTFT mismatch")
    _require(_close(float(np.percentile(ttfts, 99)), _number(summary.get("p99_ttft_ms"), f"{case_label}.p99_ttft")), f"{case_label}: p99 TTFT mismatch")
    return tuple(rows)


def _validate_blocks(summary: dict, case_label: str, num_ssu: int) -> tuple[dict, ...]:
    start = _number(summary.get("measurement_start_ms"), f"{case_label}.start")
    blocks = summary.get("measurement_blocks")
    _require(isinstance(blocks, list) and len(blocks) == BLOCK_COUNT, f"{case_label}: expected 128 x 500-ms blocks")
    sums = {
        "compute_ms_by_npu": [0.0] * NUM_NPU,
        "ssd_busy_ms_by_ssu": [0.0] * num_ssu,
        "ssd_served_gb_by_ssu": [0.0] * num_ssu,
        "npu_link_busy_ms_by_npu": [0.0] * NUM_NPU,
        "npu_link_served_gb_by_npu": [0.0] * NUM_NPU,
    }
    for block_index, block in enumerate(blocks):
        context = f"{case_label}.measurement_blocks[{block_index}]"
        _require(isinstance(block, dict), f"{context}: expected an object")
        _require(_integer(block.get("block"), f"{context}.block") == block_index, f"{context}: index mismatch")
        block_start = _number(block.get("start_ms"), f"{context}.start")
        block_end = _number(block.get("end_ms"), f"{context}.end")
        duration = _number(block.get("duration_ms"), f"{context}.duration")
        _require(_close(block_start, start + block_index * BLOCK_MS), f"{context}: start mismatch")
        _require(_close(block_end, block_start + BLOCK_MS), f"{context}: end mismatch")
        _require(_close(duration, BLOCK_MS), f"{context}: duration mismatch")
        compute = _vector(block.get("compute_ms_by_npu"), NUM_NPU, f"{context}.compute")
        npu_utils = _vector(block.get("npu_utilizations"), NUM_NPU, f"{context}.npu_utils")
        ssd_busy = _vector(block.get("ssd_busy_ms_by_ssu"), num_ssu, f"{context}.ssd_busy")
        ssd_service = _vector(block.get("ssd_served_gb_by_ssu"), num_ssu, f"{context}.ssd_service")
        ssd_utils = _vector(block.get("ssd_utilizations"), num_ssu, f"{context}.ssd_utils")
        link_busy = _vector(block.get("npu_link_busy_ms_by_npu"), NUM_NPU, f"{context}.link_busy")
        link_service = _vector(block.get("npu_link_served_gb_by_npu"), NUM_NPU, f"{context}.link_service")
        link_utils = _vector(block.get("npu_link_utilizations"), NUM_NPU, f"{context}.link_utils")
        _require(max(compute + ssd_busy + link_busy) <= BLOCK_MS + TOL, f"{context}: resource overlap exceeds block")
        _require(_vectors_close(npu_utils, [value / BLOCK_MS for value in compute]), f"{context}: NPU utilization mismatch")
        _require(_close(_number(block.get("npu_utilization"), f"{context}.fleet_util"), sum(compute) / (NUM_NPU * BLOCK_MS)), f"{context}: fleet utilization mismatch")
        _require(_vectors_close(ssd_utils, [value / BLOCK_MS for value in ssd_busy]), f"{context}: SSD utilization mismatch")
        _require(_vectors_close(ssd_service, [value * SSD_CAP_GBPS / 1000.0 for value in ssd_busy]), f"{context}: SSD service/capacity mismatch")
        _require(_vectors_close(link_utils, [value / BLOCK_MS for value in link_busy]), f"{context}: link utilization mismatch")
        _require(_vectors_close(link_service, [value * NPU_LINK_CAP_GBPS / 1000.0 for value in link_busy]), f"{context}: link service/capacity mismatch")
        for field, values in (
            ("compute_ms_by_npu", compute),
            ("ssd_busy_ms_by_ssu", ssd_busy),
            ("ssd_served_gb_by_ssu", ssd_service),
            ("npu_link_busy_ms_by_npu", link_busy),
            ("npu_link_served_gb_by_npu", link_service),
        ):
            sums[field] = [left + right for left, right in zip(sums[field], values)]

    summary_fields = {
        "compute_ms_by_npu": "compute_ms_by_npu",
        "ssd_busy_ms_by_ssu": "measurement_ssd_busy_ms_by_ssu",
        "ssd_served_gb_by_ssu": "measurement_ssd_served_gb_by_ssu",
        "npu_link_busy_ms_by_npu": "measurement_npu_link_busy_ms_by_npu",
    }
    for source, target in summary_fields.items():
        size = num_ssu if "ssu" in source else NUM_NPU
        reported = _vector(summary.get(target), size, f"{case_label}.{target}")
        _require(_vectors_close(sums[source], reported), f"{case_label}: block sum != {target}")
    fleet_util = sum(sums["compute_ms_by_npu"]) / (NUM_NPU * MEASUREMENT_MS)
    _require(_close(fleet_util, _number(summary.get("mean_npu_utilization"), f"{case_label}.mean_npu_utilization")), f"{case_label}: whole-window NPU utilization mismatch")
    return tuple(blocks)


def _validate_active_command_projection(
    command: Mapping[str, object],
    *,
    boundary_time_ms: float,
    total_gb: float,
    context: str,
) -> None:
    """Validate immutable non-preemptive command progress at a left limit."""

    command_remaining = _number(
        command.get("remaining_gb"), f"{context}.remaining_gb"
    )
    command_start_time = _number(
        command.get("command_start_time_ms"),
        f"{context}.command_start_time_ms",
    )
    command_age = _number(
        command.get("command_age_ms"), f"{context}.command_age_ms"
    )
    expected_command_age = boundary_time_ms - command_start_time
    expected_command_remaining = max(
        0.0,
        total_gb - SSD_CAP_GBPS * expected_command_age / 1000.0,
    )
    _require(
        0.0 <= command_remaining <= total_gb + TOL
        and command_start_time <= boundary_time_ms
        and abs(command_age - expected_command_age) <= 1e-9
        and _close(
            command_remaining,
            expected_command_remaining,
            tolerance=SSD_SERVICE_ABSOLUTE_TOLERANCE_GB,
        ),
        f"{context}: immutable command start/age/remaining projection mismatch",
    )
    _require(
        _close(
            _number(
                command.get("physical_service_gbps"),
                f"{context}.physical_service_gbps",
            ),
            SSD_CAP_GBPS,
        )
        and command.get("non_preemptive") is True,
        f"{context}: instantaneous command rate/non-preemption mismatch",
    )


def _validate_timeline(
    summary: dict,
    blocks: Sequence[dict],
    case: str,
    num_ssu: int,
    case_label: str,
) -> tuple[dict, ...]:
    start = _number(summary.get("measurement_start_ms"), f"{case_label}.start")
    boundaries = summary.get("measurement_stationarity_boundaries")
    _require(_integer(summary.get("measurement_stationarity_boundary_count"), f"{case_label}.boundary_count") == BOUNDARY_COUNT, f"{case_label}: boundary count field mismatch")
    _require(isinstance(boundaries, list) and len(boundaries) == BOUNDARY_COUNT, f"{case_label}: expected 129 boundaries")
    authenticated_requests = _current_materialized_request_metadata(num_ssu)

    previous: dict | None = None
    for boundary_index, boundary in enumerate(boundaries):
        context = f"{case_label}.boundary[{boundary_index}]"
        _require(isinstance(boundary, dict), f"{context}: expected an object")
        _require(_integer(boundary.get("boundary"), f"{context}.boundary") == boundary_index, f"{context}: index mismatch")
        boundary_time = _number(boundary.get("time_ms"), f"{context}.time")
        _require(_close(boundary_time, start + boundary_index * BLOCK_MS), f"{context}: time mismatch")
        base_ssd = _vector(boundary.get("ssd_cumulative_served_gb_by_ssu"), num_ssu, f"{context}.base_ssd")
        base_compute = _vector(boundary.get("npu_compute_cumulative_busy_ms_by_npu"), NUM_NPU, f"{context}.base_compute")
        base_link = _vector(boundary.get("npu_link_cumulative_served_gb_by_npu"), NUM_NPU, f"{context}.base_link")
        timeline = boundary.get("timeline")
        _require(isinstance(timeline, dict), f"{context}: timeline missing")
        _require(timeline.get("schema") == TIMELINE_SCHEMA, f"{context}: timeline schema mismatch")
        npu_rows = timeline.get("npu_rows")
        _require(isinstance(npu_rows, list) and len(npu_rows) == NUM_NPU, f"{context}: expected 32 NPU rows")
        for npu_id, npu_row in enumerate(npu_rows):
            npu_context = f"{context}.npu_rows[{npu_id}]"
            _require(isinstance(npu_row, dict), f"{npu_context}: expected object")
            _require(_integer(npu_row.get("npu_id"), f"{npu_context}.npu_id") == npu_id, f"{npu_context}: ID mismatch")
            pipeline_state = npu_row.get("pipeline_state")
            _require(
                pipeline_state
                in {
                    "compute",
                    "ready_not_running",
                    "io_barrier",
                    "between_batches",
                    "waiting_arrival",
                    "drained",
                },
                f"{npu_context}: invalid pipeline state",
            )
            current_layer = npu_row.get("current_compute_layer")
            next_layer = npu_row.get("next_compute_layer")
            waiting_layer = npu_row.get("waiting_on_io_layer")
            next_ready = npu_row.get("next_compute_layer_io_ready")
            for field, value in (
                ("current_compute_layer", current_layer),
                ("next_compute_layer", next_layer),
                ("waiting_on_io_layer", waiting_layer),
            ):
                if value is not None:
                    parsed = _integer(value, f"{npu_context}.{field}")
                    _require(0 <= parsed < N_LAYERS, f"{npu_context}.{field}: out of range")
            _require(next_ready is None or type(next_ready) is bool, f"{npu_context}.next_compute_layer_io_ready: expected bool/null")
            active_request = npu_row.get("active_request_id")
            active_batch = npu_row.get("active_batch_id")
            done_up_to = npu_row.get("compute_done_up_to")
            compute_start = npu_row.get("compute_start_ms")
            compute_end = npu_row.get("compute_end_ms")
            if pipeline_state == "compute":
                _require(
                    active_request is not None
                    and active_batch is not None
                    and current_layer is not None,
                    f"{npu_context}: compute state lacks active batch/layer",
                )
                current = _integer(current_layer, f"{npu_context}.current_layer")
                done = _integer(done_up_to, f"{npu_context}.compute_done_up_to")
                interval_start = _number(compute_start, f"{npu_context}.compute_start_ms")
                interval_end = _number(compute_end, f"{npu_context}.compute_end_ms")
                _require(
                    done == current - 1
                    and interval_start <= boundary_time + TOL
                    and boundary_time <= interval_end + TOL
                    and interval_end > interval_start,
                    f"{npu_context}: compute state/layer/interval mismatch",
                )
                expected_next = current + 1 if current + 1 < N_LAYERS else None
                _require(
                    next_layer == expected_next
                    and waiting_layer is None
                    and (expected_next is not None or next_ready is None),
                    f"{npu_context}: compute next/waiting layer mismatch",
                )
            else:
                _require(
                    current_layer is None
                    and compute_start is None
                    and compute_end is None,
                    f"{npu_context}: non-compute state has compute interval",
                )
            if pipeline_state == "io_barrier":
                _require(waiting_layer is not None and waiting_layer == next_layer, f"{npu_context}: IO-barrier layer mismatch")
                _require(next_ready is False, f"{npu_context}: IO barrier claims next layer ready")
            if pipeline_state in {"ready_not_running", "io_barrier"}:
                _require(
                    active_request is not None
                    and active_batch is not None
                    and done_up_to is not None,
                    f"{npu_context}: active non-compute state lacks batch progress",
                )
                done = _integer(done_up_to, f"{npu_context}.compute_done_up_to")
                expected_next = done + 1 if done + 1 < N_LAYERS else None
                _require(
                    next_layer == expected_next and expected_next is not None,
                    f"{npu_context}: active next layer mismatch",
                )
                if pipeline_state == "ready_not_running":
                    _require(
                        next_ready is True and waiting_layer is None,
                        f"{npu_context}: ready state fields mismatch",
                    )
            elif pipeline_state in {"between_batches", "waiting_arrival", "drained"}:
                _require(
                    active_request is None
                    and active_batch is None
                    and done_up_to is None
                    and next_layer is None
                    and next_ready is None
                    and waiting_layer is None,
                    f"{npu_context}: inactive state exposes active batch fields",
                )
            q_ms = _number(npu_row.get("compute_inventory_q_ms"), f"{npu_context}.Q")
            activated_compute = _number(
                npu_row.get("activated_compute_cumulative_ms"),
                f"{npu_context}.activated_compute",
            )
            _require(
                q_ms >= -TOL and activated_compute >= -TOL,
                f"{npu_context}: negative Q/activated compute",
            )
            controller_request = npu_row.get("controller_request_id")
            if controller_request is not None:
                parsed_controller_request = _integer(
                    controller_request, f"{npu_context}.controller_request_id"
                )
                _require(
                    parsed_controller_request in authenticated_requests
                    and authenticated_requests[parsed_controller_request]["npu_id"]
                    == npu_id,
                    f"{npu_context}: controller request is not authenticated to this NPU",
                )
            prefetch_request = npu_row.get("prefetch_request_id")
            if prefetch_request is not None:
                parsed_prefetch_request = _integer(
                    prefetch_request, f"{npu_context}.prefetch_request_id"
                )
                _require(
                    parsed_prefetch_request in authenticated_requests
                    and authenticated_requests[parsed_prefetch_request]["npu_id"]
                    == npu_id,
                    f"{npu_context}: prefetch request is not authenticated to this NPU",
                )
            controller_prefetch = npu_row.get("controller_prefetch_only")
            _require(
                controller_prefetch is None or type(controller_prefetch) is bool,
                f"{npu_context}.controller_prefetch_only: expected bool/null",
            )
            if active_request is None:
                _require(
                    _close(q_ms, 0.0),
                    f"{npu_context}: inactive NPU has nonzero compute inventory Q",
                )
            else:
                _require(
                    q_ms
                    <= float(authenticated_requests[int(active_request)]["ideal_ttft_ms"])
                    + TOL,
                    f"{npu_context}: Q exceeds active request compute budget",
                )
            if controller_request is None:
                _require(
                    controller_prefetch is None,
                    f"{npu_context}: absent controller request has prefetch flag",
                )
            elif controller_prefetch is True:
                _require(
                    prefetch_request is not None
                    and int(controller_request) == int(prefetch_request),
                    f"{npu_context}: prefetch controller request identity mismatch",
                )
            else:
                _require(
                    type(controller_prefetch) is bool
                    and
                    active_request is not None
                    and int(controller_request) == int(active_request),
                    f"{npu_context}: active controller request identity mismatch",
                )
            if prefetch_request is not None:
                if active_request is None:
                    _require(
                        controller_prefetch is True
                        and controller_request is not None
                        and int(controller_request) == int(prefetch_request),
                        f"{npu_context}: idle prefetch identity is not the controller prefetch view",
                    )
                elif int(prefetch_request) == int(active_request):
                    # Admission can occur while its previously launched
                    # cross-request layer-0 prefetch is still in flight.  The
                    # producer clears this identity only when layer-0 becomes
                    # ready, so the left-limit snapshot legitimately exposes
                    # the same ID in both fields.
                    _require(
                        next_layer == 0 and next_ready is False,
                        f"{npu_context}: admitted in-flight layer-0 prefetch state mismatch",
                    )
                else:
                    _require(
                        authenticated_requests[int(prefetch_request)]["sequence"]
                        == authenticated_requests[int(active_request)]["sequence"]
                        + 1,
                        f"{npu_context}: prefetch request is not the active stream successor",
                    )
            ttft_fields = (
                "admission_time_ms",
                "elapsed_ttft_ms",
                "ideal_ttft_ms",
                "slo_alpha1p5_slack_ms",
                "slo_alpha2_slack_ms",
            )
            if active_request is None:
                _require(
                    all(npu_row.get(field) is None for field in ttft_fields),
                    f"{npu_context}: inactive NPU exposes TTFT clock/slack metadata",
                )
            else:
                parsed_active_request = _integer(
                    active_request, f"{npu_context}.active_request_id"
                )
                active_metadata = authenticated_requests.get(parsed_active_request)
                _require(
                    active_metadata is not None
                    and active_metadata["npu_id"] == npu_id
                    and npu_row.get("category") == active_metadata["category"]
                    and _integer(
                        npu_row.get("sequence"), f"{npu_context}.sequence"
                    )
                    == active_metadata["sequence"]
                    and npu_row.get("profile_id")
                    == active_metadata["profile_id"]
                    and npu_row.get("profile_name")
                    == active_metadata["profile_name"],
                    f"{npu_context}: active request/profile is not authenticated to this NPU",
                )
                admission = _number(
                    npu_row.get("admission_time_ms"),
                    f"{npu_context}.admission_time_ms",
                )
                elapsed = _number(
                    npu_row.get("elapsed_ttft_ms"),
                    f"{npu_context}.elapsed_ttft_ms",
                )
                ideal = _number(
                    npu_row.get("ideal_ttft_ms"),
                    f"{npu_context}.ideal_ttft_ms",
                )
                _require(
                    _close(ideal, float(active_metadata["ideal_ttft_ms"])),
                    f"{npu_context}: active request ideal TTFT differs from authenticated profile",
                )
                _require(
                    admission <= boundary_time + TOL and ideal > 0.0,
                    f"{npu_context}: invalid admission time or ideal TTFT",
                )
                expected_elapsed = max(0.0, boundary_time - admission)
                _require(
                    _close(elapsed, expected_elapsed),
                    f"{npu_context}: elapsed TTFT clock mismatch",
                )
                for alpha, field in (
                    (1.5, "slo_alpha1p5_slack_ms"),
                    (2.0, "slo_alpha2_slack_ms"),
                ):
                    _require(
                        _close(
                            _number(npu_row.get(field), f"{npu_context}.{field}"),
                            alpha * ideal - elapsed,
                        ),
                        f"{npu_context}: alpha={alpha:g} diagnostic slack mismatch",
                    )

        active_commands = timeline.get("active_command_by_ssu")
        _require(
            isinstance(active_commands, list) and len(active_commands) == num_ssu,
            f"{context}: active-command vector shape mismatch",
        )
        for ssu_id, command in enumerate(active_commands):
            if command is None:
                continue
            command_context = f"{context}.active_command_by_ssu[{ssu_id}]"
            _require(isinstance(command, dict), f"{command_context}: expected object/null")
            _require(
                _integer(command.get("ssu_id"), f"{command_context}.ssu_id")
                == ssu_id,
                f"{command_context}: SSU ID mismatch",
            )
            owner = _integer(command.get("npu_id"), f"{command_context}.npu_id")
            _require(0 <= owner < NUM_NPU, f"{command_context}: owner NPU out of range")
            command_request = _integer(
                command.get("request_id"), f"{command_context}.request_id"
            )
            _require(
                command_request in authenticated_requests
                and authenticated_requests[command_request]["npu_id"] == owner,
                f"{command_context}: command request is not authenticated to owner NPU",
            )
            path_id = _integer(
                command.get("path_id"), f"{command_context}.path_id"
            )
            _require(
                0 <= path_id < EXPECTED_PATH_ABI["path_count"],
                f"{command_context}: path ID out of range",
            )
            if case == "baseline":
                _require(path_id == 0, f"{command_context}: baseline did not use Path 0")
            elif case == "adaptive_t0_i100ms":
                _require(
                    path_id == cold_start_hybrid_path_id(owner),
                    f"{command_context}: Adaptive command is not on its NPU-dedicated Path",
                )
            layer = _integer(command.get("layer"), f"{command_context}.layer")
            _require(0 <= layer < N_LAYERS, f"{command_context}: layer out of range")
            command_block_idx = _integer(
                command.get("block_idx"), f"{command_context}.block_idx"
            )
            command_block_sizes = {
                int(block_idx): float(size_gb)
                for block_idx, size_gb in authenticated_requests[
                    command_request
                ]["placement_blocks_by_ssu"][ssu_id]
            }
            _require(
                command_block_idx in command_block_sizes,
                f"{command_context}: command block is not in authenticated request/SSU placement",
            )
            _validate_active_command_projection(
                command,
                boundary_time_ms=boundary_time,
                total_gb=command_block_sizes[command_block_idx],
                context=command_context,
            )

        cells = timeline.get("npu_ssu")
        _require(isinstance(cells, dict), f"{context}: npu_ssu timeline missing")
        float_fields = (
            "ssd_enqueued_cumulative_gb",
            "ssd_served_cumulative_gb",
            "ssd_served_fragmented_diagnostic_cumulative_gb",
            "ssd_outstanding_gb",
            "link_enqueued_cumulative_gb",
            "link_served_cumulative_gb",
            "link_outstanding_gb",
            "ssd_served_awaiting_link_enqueue_gb",
            "client_unissued_gb",
            "activated_io_cumulative_gb",
            "physical_remaining_gb",
            "controller_declared_remaining_gb",
            "physical_demand_gbps",
            "controller_demand_gbps",
        )
        matrices: dict[str, list[list[float]]] = {}
        for field in float_fields:
            matrix = _matrix(cells.get(field), NUM_NPU, num_ssu, f"{context}.{field}")
            assert matrix is not None
            matrices[field] = matrix
        for field in (
            "ssd_outstanding_blocks",
            "link_outstanding_blocks",
            "route_plans_cumulative",
            "route_pressure_fresh_cumulative",
            "route_pressure_cache_cumulative",
        ):
            _matrix(cells.get(field), NUM_NPU, num_ssu, f"{context}.{field}", integer=True)
        residual_rows = timeline.get("ssd_accounting_residuals_by_ssu")
        _require(
            isinstance(residual_rows, list) and len(residual_rows) == num_ssu,
            f"{context}: v3 SSD accounting residual vector is missing",
        )
        base_ssd_outstanding_gb = _vector(
            boundary.get("ssd_outstanding_gb_by_ssu"),
            num_ssu,
            f"{context}.base_ssd_outstanding_gb",
        )
        base_ssd_outstanding_blocks = _vector(
            boundary.get("ssd_outstanding_blocks_by_ssu"),
            num_ssu,
            f"{context}.base_ssd_outstanding_blocks",
            integer=True,
        )
        for ssu_id, residual_row in enumerate(residual_rows):
            residual_context = (
                f"{context}.ssd_accounting_residuals_by_ssu[{ssu_id}]"
            )
            _require(
                isinstance(residual_row, dict)
                and _integer(
                    residual_row.get("ssu_id"), f"{residual_context}.ssu_id"
                )
                == ssu_id,
                f"{residual_context}: SSU identity mismatch",
            )
            _require(
                set(residual_row)
                == {
                    "ssu_id",
                    "stable_service_minus_busy_counter_gb",
                    "fragmented_service_minus_stable_gb",
                    "physical_queue_minus_scheduler_gb",
                    "enqueue_minus_service_minus_physical_queue_gb",
                    "counter_queue_minus_physical_queue_gb",
                    "fragmented_counter_queue_minus_physical_queue_gb",
                    "maximum_abs_npu_queue_identity_residual_gb",
                    "maximum_abs_npu_queue_identity_residual_npu_id",
                    "physical_queue_block_minus_scheduler_blocks",
                    "counter_queue_block_minus_physical_blocks",
                },
                f"{residual_context}: v3 residual field set mismatch",
            )
            enqueued_total = math.fsum(
                matrices["ssd_enqueued_cumulative_gb"][npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            stable_service_total = math.fsum(
                matrices["ssd_served_cumulative_gb"][npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            fragmented_service_total = math.fsum(
                matrices[
                    "ssd_served_fragmented_diagnostic_cumulative_gb"
                ][npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            physical_queue_total = math.fsum(
                matrices["ssd_outstanding_gb"][npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            counter_queue_total = math.fsum(
                max(
                    0.0,
                    matrices["ssd_enqueued_cumulative_gb"][npu_id][ssu_id]
                    - matrices["ssd_served_cumulative_gb"][npu_id][ssu_id],
                )
                for npu_id in range(NUM_NPU)
            )
            fragmented_counter_queue_total = math.fsum(
                max(
                    0.0,
                    matrices["ssd_enqueued_cumulative_gb"][npu_id][ssu_id]
                    - matrices[
                        "ssd_served_fragmented_diagnostic_cumulative_gb"
                    ][npu_id][ssu_id],
                )
                for npu_id in range(NUM_NPU)
            )
            per_npu_identity = [
                math.fsum(
                    (
                        matrices["ssd_enqueued_cumulative_gb"][npu_id][
                            ssu_id
                        ],
                        -matrices["ssd_served_cumulative_gb"][npu_id][ssu_id],
                        -matrices["ssd_outstanding_gb"][npu_id][ssu_id],
                    )
                )
                for npu_id in range(NUM_NPU)
            ]
            maximum_identity_npu = max(
                range(NUM_NPU), key=lambda npu_id: abs(per_npu_identity[npu_id])
            )
            expected_float_residuals = {
                "stable_service_minus_busy_counter_gb": (
                    stable_service_total - base_ssd[ssu_id]
                ),
                "fragmented_service_minus_stable_gb": (
                    fragmented_service_total - stable_service_total
                ),
                "physical_queue_minus_scheduler_gb": (
                    physical_queue_total - base_ssd_outstanding_gb[ssu_id]
                ),
                "enqueue_minus_service_minus_physical_queue_gb": math.fsum(
                    (enqueued_total, -stable_service_total, -physical_queue_total)
                ),
                "counter_queue_minus_physical_queue_gb": (
                    counter_queue_total - physical_queue_total
                ),
                "fragmented_counter_queue_minus_physical_queue_gb": (
                    fragmented_counter_queue_total - physical_queue_total
                ),
                "maximum_abs_npu_queue_identity_residual_gb": abs(
                    per_npu_identity[maximum_identity_npu]
                ),
            }
            for field, expected in expected_float_residuals.items():
                observed = _number(
                    residual_row.get(field), f"{residual_context}.{field}"
                )
                _require(
                    _close(observed, expected, tolerance=2e-10),
                    f"{residual_context}: {field} does not independently recompute",
                )
                if field not in {
                    "fragmented_service_minus_stable_gb",
                    "fragmented_counter_queue_minus_physical_queue_gb",
                }:
                    _require(
                        abs(observed)
                        <= SSD_SERVICE_ABSOLUTE_TOLERANCE_GB + 1e-15,
                        f"{residual_context}: {field} exceeds v3 accounting tolerance",
                    )
            _require(
                _integer(
                    residual_row.get(
                        "maximum_abs_npu_queue_identity_residual_npu_id"
                    ),
                    f"{residual_context}.maximum_identity_npu",
                )
                == maximum_identity_npu,
                f"{residual_context}: maximum residual NPU identity mismatch",
            )
            expected_physical_blocks_residual = int(
                math.fsum(
                    int(cells["ssd_outstanding_blocks"][npu_id][ssu_id])
                    for npu_id in range(NUM_NPU)
                )
                - base_ssd_outstanding_blocks[ssu_id]
            )
            _require(
                _integer(
                    residual_row.get(
                        "physical_queue_block_minus_scheduler_blocks"
                    ),
                    f"{residual_context}.physical_queue_block_residual",
                )
                == expected_physical_blocks_residual
                == 0,
                f"{residual_context}: direct physical block enumeration mismatch",
            )
            _require(
                _integer(
                    residual_row.get(
                        "counter_queue_block_minus_physical_blocks"
                    ),
                    f"{residual_context}.counter_queue_block_residual",
                )
                == 0,
                f"{residual_context}: legacy block counter differs from direct physical enumeration",
            )
        compute_denominator = _vector(cells.get("controller_remaining_compute_ms"), NUM_NPU, f"{context}.controller_remaining_compute_ms")
        for npu_id, npu_row in enumerate(npu_rows):
            controller_request = npu_row.get("controller_request_id")
            if controller_request is None:
                _require(
                    _close(compute_denominator[npu_id], 0.0)
                    and all(
                        _close(value, 0.0)
                        for value in matrices[
                            "controller_declared_remaining_gb"
                        ][npu_id]
                    )
                    and all(
                        _close(value, 0.0)
                        for value in matrices["controller_demand_gbps"][npu_id]
                    ),
                    f"{context}: absent controller request exposes demand/compute",
                )
            else:
                _require(
                    compute_denominator[npu_id] > 0.0
                    and any(
                        value > 0.0
                        for value in matrices[
                            "controller_declared_remaining_gb"
                        ][npu_id]
                    ),
                    f"{context}: controller request lacks remaining work/compute",
                )
                _authenticated_remaining_layer_count(
                    num_ssu=num_ssu,
                    npu_id=npu_id,
                    request_id=_integer(
                        controller_request,
                        f"{context}.npu[{npu_id}].controller_request_id",
                    ),
                    remaining_compute_ms=compute_denominator[npu_id],
                    remaining_work_gb_by_ssu=matrices[
                        "controller_declared_remaining_gb"
                    ][npu_id],
                    context=f"{context}.npu[{npu_id}].controller_inventory",
                )
        installed = _matrix(
            cells.get("installed_dedicated_path_cir_gbps"),
            NUM_NPU,
            num_ssu,
            f"{context}.installed_cir",
            allow_none=True,
        )
        if case == "adaptive_t0_i100ms":
            _require(installed is not None, f"{context}: Adaptive CIR matrix missing")
            assert installed is not None
            for ssu_id in range(num_ssu):
                _require(sum(row[ssu_id] for row in installed) <= SSD_CAP_GBPS + TOL, f"{context}: installed CIR exceeds SSU capacity")
            for npu_id, row_values in enumerate(installed):
                _require(sum(row_values) <= NPU_LINK_CAP_GBPS + TOL, f"{context}: NPU {npu_id} CIR exceeds link capacity")
        else:
            _require(installed is None, f"{context}: non-Adaptive strategy unexpectedly exposes dedicated CIR")

        path_rows = timeline.get("sparse_ssu_path_rows")
        _require(isinstance(path_rows, list), f"{context}: sparse Path state missing")
        paths_by_ssu: dict[int, dict[int, dict]] = {
            ssu_id: {} for ssu_id in range(num_ssu)
        }
        for path_index, path_row in enumerate(path_rows):
            path_context = f"{context}.sparse_ssu_path_rows[{path_index}]"
            _require(isinstance(path_row, dict), f"{path_context}: expected object")
            ssu_id = _integer(path_row.get("ssu_id"), f"{path_context}.ssu_id")
            path_id = _integer(path_row.get("path_id"), f"{path_context}.path_id")
            group_id = _integer(path_row.get("group_id"), f"{path_context}.group_id")
            _require(
                0 <= ssu_id < num_ssu
                and 0 <= path_id < EXPECTED_PATH_ABI["path_count"]
                and group_id == path_id // EXPECTED_PATH_ABI["paths_per_group"],
                f"{path_context}: invalid SSU/Path/group mapping",
            )
            _require(
                path_id not in paths_by_ssu[ssu_id],
                f"{path_context}: duplicate Path row",
            )
            paths_by_ssu[ssu_id][path_id] = path_row
            cir = _number(path_row.get("cir_gbps"), f"{path_context}.cir_gbps")
            weight = _number(path_row.get("path_weight"), f"{path_context}.path_weight")
            virtual_finish = _number(
                path_row.get("virtual_finish"), f"{path_context}.virtual_finish"
            )
            _require(
                virtual_finish >= -TOL,
                f"{path_context}: virtual finish is negative",
            )
            rate = _number(
                path_row.get("estimated_next_arbitration_rate_gbps"),
                f"{path_context}.estimated_next_arbitration_rate_gbps",
            )
            pending_blocks = _integer(
                path_row.get("pending_blocks"), f"{path_context}.pending_blocks"
            )
            pending_gb = _number(path_row.get("pending_gb"), f"{path_context}.pending_gb")
            active_remaining = _number(
                path_row.get("active_remaining_gb"),
                f"{path_context}.active_remaining_gb",
            )
            _require(
                min(cir, weight, rate, pending_blocks, pending_gb, active_remaining)
                >= -TOL,
                f"{path_context}: negative Path arbitration/queue state",
            )
            is_active_command_path = (
                isinstance(active_commands[ssu_id], dict)
                and path_id == int(active_commands[ssu_id]["path_id"])
            )
            _require(
                pending_blocks > 0 or is_active_command_path,
                f"{path_context}: sparse Path row is neither pending nor active",
            )
            _require(
                is_active_command_path or active_remaining == 0.0,
                f"{path_context}: non-command Path exposes active remaining bytes",
            )
            head_fields = (
                "head_wait_age_ms",
                "head_npu_id",
                "head_request_id",
                "head_layer",
            )
            if pending_blocks == 0:
                _require(
                    _close(pending_gb, 0.0)
                    and all(path_row.get(field) is None for field in head_fields),
                    f"{path_context}: empty pending queue exposes head metadata/bytes",
                )
            else:
                _require(pending_gb > 0.0, f"{path_context}: pending blocks lack bytes")
                _require(
                    _number(
                        path_row.get("head_wait_age_ms"),
                        f"{path_context}.head_wait_age_ms",
                    )
                    >= -TOL,
                    f"{path_context}: negative head wait age",
                )
                head_npu = _integer(
                    path_row.get("head_npu_id"), f"{path_context}.head_npu_id"
                )
                head_request = _integer(
                    path_row.get("head_request_id"),
                    f"{path_context}.head_request_id",
                )
                head_layer = _integer(
                    path_row.get("head_layer"), f"{path_context}.head_layer"
                )
                _require(
                    0 <= head_npu < NUM_NPU and 0 <= head_layer < N_LAYERS,
                    f"{path_context}: invalid pending-head NPU/layer",
                )
                _require(
                    head_request in authenticated_requests
                    and authenticated_requests[head_request]["npu_id"]
                    == head_npu,
                    f"{path_context}: pending-head request is not authenticated to head NPU",
                )
            if case == "baseline":
                _require(path_id == 0, f"{path_context}: baseline queue is not Path 0")
            elif case == "adaptive_t0_i100ms":
                dedicated_by_path = {
                    cold_start_hybrid_path_id(npu_id): npu_id
                    for npu_id in range(NUM_NPU)
                }
                _require(
                    path_id in dedicated_by_path,
                    f"{path_context}: Adaptive queue is not a dedicated Path",
                )
                dedicated_npu = dedicated_by_path[path_id]
                assert installed is not None
                _require(
                    _close(cir, installed[dedicated_npu][ssu_id]),
                    f"{path_context}: Path CIR differs from installed NPU/SSU CIR",
                )
                _require(
                    _close(weight, 1.0),
                    f"{path_context}: Adaptive immutable Path weight mismatch",
                )
            else:
                formal_qos = static_qos_config()
                _require(
                    _close(cir, formal_qos.path_cirs[path_id])
                    and _close(weight, formal_qos.path_weights[path_id]),
                    f"{path_context}: Path CIR/weight differs from formal static QoS registers",
                )

        for ssu_id in range(num_ssu):
            ssu_paths = paths_by_ssu[ssu_id]
            path_outstanding_gb = math.fsum(
                    _number(row["pending_gb"], f"{context}.path_pending_gb")
                    + _number(
                        row["active_remaining_gb"],
                        f"{context}.path_active_remaining_gb",
                    )
                    for row in ssu_paths.values()
                )
            attributed_outstanding_gb = math.fsum(
                matrices["ssd_outstanding_gb"][npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            _require(
                _close(
                    path_outstanding_gb,
                    attributed_outstanding_gb,
                    tolerance=SSD_SERVICE_ABSOLUTE_TOLERANCE_GB,
                ),
                f"{context}: sparse Path bytes do not close to aggregate SSD queue",
            )
            command = active_commands[ssu_id]
            active_path_ids = (
                []
                if command is None
                else [
                    path_id
                    for path_id in ssu_paths
                    if path_id == int(command["path_id"])
                ]
            )
            if command is None:
                _require(
                    not active_path_ids,
                    f"{context}: sparse Path exposes active work without active command",
                )
            else:
                _require(
                    active_path_ids == [int(command["path_id"])],
                    f"{context}: active command does not have exactly one active sparse Path",
                )
            sparse_outstanding_blocks = sum(
                _integer(row["pending_blocks"], f"{context}.pending_blocks")
                for row in ssu_paths.values()
            ) + len(active_path_ids)
            direct_cell_blocks = sum(
                int(cells["ssd_outstanding_blocks"][npu_id][ssu_id])
                for npu_id in range(NUM_NPU)
            )
            _require(
                sparse_outstanding_blocks
                == direct_cell_blocks
                == int(base_ssd_outstanding_blocks[ssu_id]),
                f"{context}: sparse Path block counts do not close to direct physical queue",
            )
            _require(
                math.fsum(
                    _number(
                        row["estimated_next_arbitration_rate_gbps"],
                        f"{context}.path_estimated_rate",
                    )
                    for row in ssu_paths.values()
                )
                <= SSD_CAP_GBPS + TOL,
                f"{context}: estimated next-arbitration Path rates exceed SSD capacity",
            )
            pending_paths = [
                (path_id, row)
                for path_id, row in ssu_paths.items()
                if _integer(row["pending_blocks"], f"{context}.pending_blocks")
                > 0
            ]
            expected_rates = {path_id: 0.0 for path_id in ssu_paths}
            if pending_paths:
                assigned = {
                    path_id: _number(row["cir_gbps"], f"{context}.path_cir")
                    for path_id, row in pending_paths
                }
                remaining = max(0.0, SSD_CAP_GBPS - math.fsum(assigned.values()))
                by_group: dict[int, list[tuple[int, dict]]] = defaultdict(list)
                for path_id, row in pending_paths:
                    by_group[path_id // EXPECTED_PATH_ABI["paths_per_group"]].append(
                        (path_id, row)
                    )
                active_groups = [
                    group_id
                    for group_id, members in by_group.items()
                    if any(
                        _number(member["path_weight"], f"{context}.path_weight")
                        > 1e-12
                        for _path_id, member in members
                    )
                ]
                active_group_weight = float(len(active_groups))
                for group_id in active_groups:
                    members = by_group[group_id]
                    path_weight_sum = math.fsum(
                        _number(member["path_weight"], f"{context}.path_weight")
                        for _path_id, member in members
                        if _number(
                            member["path_weight"], f"{context}.path_weight"
                        )
                        > 1e-12
                    )
                    group_grant = (
                        remaining / active_group_weight
                        if active_group_weight > 0.0
                        else 0.0
                    )
                    for path_id, member in members:
                        weight_value = _number(
                            member["path_weight"], f"{context}.path_weight"
                        )
                        expected_rates[path_id] = assigned[path_id] + (
                            group_grant * weight_value / path_weight_sum
                            if weight_value > 1e-12 and path_weight_sum > 0.0
                            else 0.0
                        )
            _require(
                all(
                    _close(
                        _number(
                            row["estimated_next_arbitration_rate_gbps"],
                            f"{context}.estimated_next_rate",
                        ),
                        expected_rates[path_id],
                    )
                    for path_id, row in ssu_paths.items()
                ),
                f"{context}: next-arbitration rates differ from independent CIR/group/path-weight allocation",
            )
            if command is not None:
                command_path = _integer(
                    command["path_id"], f"{context}.active_command.path_id"
                )
                _require(
                    command_path in ssu_paths
                    and _close(
                        _number(
                            ssu_paths[command_path]["active_remaining_gb"],
                            f"{context}.active_path_remaining",
                        ),
                        _number(
                            command["remaining_gb"],
                            f"{context}.active_command.remaining_gb",
                        ),
                    ),
                    f"{context}: active command does not close to sparse Path state",
                )

        pressure_rows = timeline.get("pressure_state_by_ssu")
        _require(
            isinstance(pressure_rows, list) and len(pressure_rows) == num_ssu,
            f"{context}: pressure-state vector shape mismatch",
        )
        for ssu_id, pressure in enumerate(pressure_rows):
            pressure_context = f"{context}.pressure_state_by_ssu[{ssu_id}]"
            _require(isinstance(pressure, dict), f"{pressure_context}: expected object")
            _require(
                _integer(pressure.get("ssu_id"), f"{pressure_context}.ssu_id")
                == ssu_id,
                f"{pressure_context}: SSU ID mismatch",
            )
            reports = _integer(
                pressure.get("reports_cumulative"),
                f"{pressure_context}.reports_cumulative",
            )
            hits = _integer(
                pressure.get("cache_hits_cumulative"),
                f"{pressure_context}.cache_hits_cumulative",
            )
            _require(min(reports, hits) >= 0, f"{pressure_context}: negative counter")
            _require(
                reports
                == sum(
                    int(
                        cells["route_pressure_fresh_cumulative"][npu_id][ssu_id]
                    )
                    for npu_id in range(NUM_NPU)
                )
                and hits
                == sum(
                    int(
                        cells["route_pressure_cache_cumulative"][npu_id][ssu_id]
                    )
                    for npu_id in range(NUM_NPU)
                ),
                f"{pressure_context}: aggregate pressure counters do not close to NPU/SSU cells",
            )
            _require(
                _close(
                    _number(pressure.get("ttl_ms"), f"{pressure_context}.ttl_ms"),
                    5.0 if case == "layer_once_ttl_5ms" else 0.0,
                ),
                f"{pressure_context}: TTL differs from strategy",
            )
            cache_time = pressure.get("cache_time_ms")
            cache_age = pressure.get("cache_age_ms")
            if cache_time is None:
                _require(cache_age is None, f"{pressure_context}: cache age without cache time")
            else:
                parsed_cache_time = _number(
                    cache_time, f"{pressure_context}.cache_time_ms"
                )
                _require(
                    parsed_cache_time <= boundary_time + TOL
                    and _close(
                        _number(cache_age, f"{pressure_context}.cache_age_ms"),
                        max(0.0, boundary_time - parsed_cache_time),
                    ),
                    f"{pressure_context}: cache age/time mismatch",
                )

        route_groups = cells.get("route_blocks_by_group_cumulative")
        _require(isinstance(route_groups, list) and len(route_groups) == NUM_NPU, f"{context}: route group tensor shape")
        for npu_id, row in enumerate(route_groups):
            _require(isinstance(row, list) and len(row) == num_ssu, f"{context}: route group row {npu_id} shape")
            for ssu_id, groups in enumerate(row):
                _vector(
                    groups,
                    EXPECTED_PATH_ABI["group_count"],
                    f"{context}.route_groups[{npu_id}][{ssu_id}]",
                    integer=True,
                )

        _require(_vectors_close([sum(row[ssu] for row in matrices["ssd_served_cumulative_gb"]) for ssu in range(num_ssu)], base_ssd), f"{context}: NPU x SSU SSD attribution does not equal physical SSD counter")
        _require(_vectors_close([sum(row) for row in matrices["link_served_cumulative_gb"]], base_link), f"{context}: NPU x SSU link attribution does not equal physical link counter")
        for npu_id in range(NUM_NPU):
            compute_s = compute_denominator[npu_id] / 1000.0
            for ssu_id in range(num_ssu):
                _require(
                    _close(
                        matrices["ssd_enqueued_cumulative_gb"][npu_id][ssu_id]
                        - matrices["ssd_served_cumulative_gb"][npu_id][ssu_id],
                        matrices["ssd_outstanding_gb"][npu_id][ssu_id],
                        tolerance=2e-7,
                    ),
                    f"{context}: SSD queue-byte conservation mismatch",
                )
                _require(
                    _close(
                        matrices["link_enqueued_cumulative_gb"][npu_id][ssu_id]
                        - matrices["link_served_cumulative_gb"][npu_id][ssu_id],
                        matrices["link_outstanding_gb"][npu_id][ssu_id],
                        tolerance=2e-7,
                    ),
                    f"{context}: link queue-byte conservation mismatch",
                )
                undelivered_activated = (
                    matrices["client_unissued_gb"][npu_id][ssu_id]
                    + matrices["ssd_outstanding_gb"][npu_id][ssu_id]
                    + matrices["ssd_served_awaiting_link_enqueue_gb"][npu_id][ssu_id]
                    + matrices["link_outstanding_gb"][npu_id][ssu_id]
                )
                _require(
                    _close(
                        matrices["activated_io_cumulative_gb"][npu_id][ssu_id],
                        undelivered_activated
                        + matrices["link_served_cumulative_gb"][npu_id][ssu_id],
                        tolerance=2e-7,
                    ),
                    f"{context}: activated IO stage conservation mismatch",
                )
                _require(
                    matrices["physical_remaining_gb"][npu_id][ssu_id]
                    + TOL
                    >= undelivered_activated,
                    f"{context}: physical remaining work is below activated undelivered IO",
                )
                _require(
                    matrices["controller_declared_remaining_gb"][npu_id][ssu_id]
                    + TOL
                    >= matrices["physical_remaining_gb"][npu_id][ssu_id],
                    f"{context}: controller remaining work is below physical remaining work",
                )
                expected_controller = (
                    matrices["controller_declared_remaining_gb"][npu_id][ssu_id] / compute_s
                    if compute_s > 0.0
                    else 0.0
                )
                expected_physical = (
                    matrices["physical_remaining_gb"][npu_id][ssu_id] / compute_s
                    if compute_s > 0.0
                    else 0.0
                )
                _require(_close(expected_controller, matrices["controller_demand_gbps"][npu_id][ssu_id]), f"{context}: controller demand formula mismatch")
                _require(_close(expected_physical, matrices["physical_demand_gbps"][npu_id][ssu_id]), f"{context}: physical demand formula mismatch")

        if previous is not None:
            prior_timeline = previous["timeline"]
            prior_cells = prior_timeline["npu_ssu"]
            current_compute = base_compute
            prior_compute = _vector(previous["npu_compute_cumulative_busy_ms_by_npu"], NUM_NPU, f"{context}.prior_compute")
            block = blocks[boundary_index - 1]
            block_compute = _vector(block.get("compute_ms_by_npu"), NUM_NPU, f"{context}.block_compute")
            _require(_vectors_close([now - before for before, now in zip(prior_compute, current_compute)], block_compute), f"{context}: boundary compute difference != block compute")

            for field in (
                "ssd_enqueued_cumulative_gb",
                "ssd_served_cumulative_gb",
                "ssd_served_fragmented_diagnostic_cumulative_gb",
                "link_enqueued_cumulative_gb",
                "link_served_cumulative_gb",
                "activated_io_cumulative_gb",
                "route_plans_cumulative",
                "route_pressure_fresh_cumulative",
                "route_pressure_cache_cumulative",
            ):
                now = cells[field]
                before = prior_cells[field]
                _require(all(float(now[npu][ssu]) + TOL >= float(before[npu][ssu]) for npu in range(NUM_NPU) for ssu in range(num_ssu)), f"{context}: {field} decreased")
            prior_route_groups = prior_cells["route_blocks_by_group_cumulative"]
            _require(
                all(
                    int(route_groups[npu][ssu][group])
                    >= int(prior_route_groups[npu][ssu][group])
                    for npu in range(NUM_NPU)
                    for ssu in range(num_ssu)
                    for group in range(EXPECTED_PATH_ABI["group_count"])
                ),
                f"{context}: route_blocks_by_group_cumulative decreased",
            )
            for npu_id in range(NUM_NPU):
                expected_adaptive_group = (
                    cold_start_hybrid_path_id(npu_id)
                    // EXPECTED_PATH_ABI["paths_per_group"]
                )
                for ssu_id in range(num_ssu):
                    plan_delta = int(cells["route_plans_cumulative"][npu_id][ssu_id]) - int(
                        prior_cells["route_plans_cumulative"][npu_id][ssu_id]
                    )
                    fresh_delta = int(cells["route_pressure_fresh_cumulative"][npu_id][ssu_id]) - int(
                        prior_cells["route_pressure_fresh_cumulative"][npu_id][ssu_id]
                    )
                    cache_delta = int(cells["route_pressure_cache_cumulative"][npu_id][ssu_id]) - int(
                        prior_cells["route_pressure_cache_cumulative"][npu_id][ssu_id]
                    )
                    group_delta = [
                        int(route_groups[npu_id][ssu_id][group_id])
                        - int(prior_route_groups[npu_id][ssu_id][group_id])
                        for group_id in range(EXPECTED_PATH_ABI["group_count"])
                    ]
                    _require(
                        min([plan_delta, fresh_delta, cache_delta] + group_delta)
                        >= 0,
                        f"{context}: route counter delta is negative",
                    )
                    if case == "layer_once_ttl_5ms":
                        _require(
                            plan_delta == fresh_delta + cache_delta,
                            f"{context}: layer-once plans != fresh + cache reads",
                        )
                    else:
                        _require(
                            fresh_delta == 0 and cache_delta == 0,
                            f"{context}: non-pressure policy changed pressure counters",
                        )
                    _require(
                        (plan_delta == 0) == (sum(group_delta) == 0)
                        and sum(group_delta) >= plan_delta,
                        f"{context}: route plans and routed-block group counts do not close",
                    )
                    if case == "baseline":
                        _require(
                            all(
                                count == 0
                                for group_id, count in enumerate(group_delta)
                                if group_id != 0
                            ),
                            f"{context}: baseline routed outside reserved Path-0 group",
                        )
                    elif case == "adaptive_t0_i100ms":
                        _require(
                            all(
                                count == 0
                                for group_id, count in enumerate(group_delta)
                                if group_id != expected_adaptive_group
                            ),
                            f"{context}: Adaptive routed outside its NPU-dedicated Path group",
                        )

            ssd_delta = [
                [
                    matrices["ssd_served_cumulative_gb"][npu][ssu]
                    - float(prior_cells["ssd_served_cumulative_gb"][npu][ssu])
                    for ssu in range(num_ssu)
                ]
                for npu in range(NUM_NPU)
            ]
            link_delta = [
                [
                    matrices["link_served_cumulative_gb"][npu][ssu]
                    - float(prior_cells["link_served_cumulative_gb"][npu][ssu])
                    for ssu in range(num_ssu)
                ]
                for npu in range(NUM_NPU)
            ]
            _require(min(_flatten(ssd_delta) + _flatten(link_delta)) >= -TOL, f"{context}: physical service counter decreased")
            _require(all(sum(ssd_delta[npu][ssu] for npu in range(NUM_NPU)) <= SSD_CAP_GBPS * BLOCK_MS / 1000.0 + TOL for ssu in range(num_ssu)), f"{context}: SSD service exceeds capacity")
            _require(all(sum(link_delta[npu]) <= NPU_LINK_CAP_GBPS * BLOCK_MS / 1000.0 + TOL for npu in range(NUM_NPU)), f"{context}: NPU-link delivery exceeds capacity")
            block_ssd = _vector(block.get("ssd_served_gb_by_ssu"), num_ssu, f"{context}.block_ssd")
            block_link = _vector(block.get("npu_link_served_gb_by_npu"), NUM_NPU, f"{context}.block_link")
            _require(_vectors_close([sum(row[ssu] for row in ssd_delta) for ssu in range(num_ssu)], block_ssd), f"{context}: exact SSD cell differences != block totals")
            _require(_vectors_close([sum(row) for row in link_delta], block_link), f"{context}: exact link cell differences != block totals")

            for npu_id in range(NUM_NPU):
                before_row = prior_timeline["npu_rows"][npu_id]
                now_row = npu_rows[npu_id]
                activated = _number(now_row["activated_compute_cumulative_ms"], f"{context}.activated") - _number(before_row["activated_compute_cumulative_ms"], f"{context}.prior_activated")
                _require(
                    activated >= -1e-9,
                    f"{context}: NPU {npu_id} activated-compute counter decreased",
                )
                q_start = _number(before_row["compute_inventory_q_ms"], f"{context}.q_start")
                q_end = _number(now_row["compute_inventory_q_ms"], f"{context}.q_end")
                reconstructed = activated + q_start - q_end
                _require(_close(reconstructed, block_compute[npu_id], tolerance=2e-6), f"{context}: NPU {npu_id} Q/activation conservation mismatch")
        previous = boundary

    first, last = boundaries[0], boundaries[-1]
    first_cells = first["timeline"]["npu_ssu"]
    last_cells = last["timeline"]["npu_ssu"]
    whole_ssd = [
        [
            float(last_cells["ssd_served_cumulative_gb"][npu][ssu])
            - float(first_cells["ssd_served_cumulative_gb"][npu][ssu])
            for ssu in range(num_ssu)
        ]
        for npu in range(NUM_NPU)
    ]
    whole_link = [
        [
            float(last_cells["link_served_cumulative_gb"][npu][ssu])
            - float(first_cells["link_served_cumulative_gb"][npu][ssu])
            for ssu in range(num_ssu)
        ]
        for npu in range(NUM_NPU)
    ]
    whole_fragmented = [
        [
            float(
                last_cells["ssd_served_fragmented_diagnostic_cumulative_gb"][
                    npu
                ][ssu]
            )
            - float(
                first_cells[
                    "ssd_served_fragmented_diagnostic_cumulative_gb"
                ][npu][ssu]
            )
            for ssu in range(num_ssu)
        ]
        for npu in range(NUM_NPU)
    ]
    reported_ssd = _matrix(summary.get("measurement_npu_ssu_ssd_served_gb"), NUM_NPU, num_ssu, f"{case_label}.whole_ssd")
    reported_link = _matrix(summary.get("measurement_npu_ssu_link_served_gb"), NUM_NPU, num_ssu, f"{case_label}.whole_link")
    reported_fragmented = _matrix(
        summary.get("measurement_fragmented_npu_ssu_ssd_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{case_label}.whole_fragmented_ssd_diagnostic",
    )
    assert reported_ssd is not None and reported_link is not None and reported_fragmented is not None
    _require(_matrices_close(whole_ssd, reported_ssd), f"{case_label}: whole-window SSD cell reconstruction mismatch")
    _require(_matrices_close(whole_link, reported_link), f"{case_label}: whole-window link cell reconstruction mismatch")
    _require(
        _matrices_close(whole_fragmented, reported_fragmented),
        f"{case_label}: whole-window fragmented diagnostic reconstruction mismatch",
    )
    return tuple(boundaries)


def _expected_measurement_ssd_accounting_residuals(
    stable_matrix: Sequence[Sequence[float]],
    fragmented_matrix: Sequence[Sequence[float]],
    busy_service: Sequence[float],
    num_ssu: int,
) -> dict[str, list[float]]:
    """Reproduce the producer's exact whole-window reduction order."""

    return {
        "stable_service_minus_busy_counter_gb_by_ssu": [
            math.fsum(
                stable_matrix[npu_id][ssu_id] for npu_id in range(NUM_NPU)
            )
            - busy_service[ssu_id]
            for ssu_id in range(num_ssu)
        ],
        "fragmented_service_minus_stable_gb_by_ssu": [
            # The fragmented observer is diagnostics-only.  Keep the exact
            # producer order: reduce each column, then subtract the totals.
            math.fsum(
                fragmented_matrix[npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            - math.fsum(
                stable_matrix[npu_id][ssu_id]
                for npu_id in range(NUM_NPU)
            )
            for ssu_id in range(num_ssu)
        ],
    }


def _validate_decimal_byte_identity(
    observed_gb: Sequence[float],
    observed_decimal_bytes: Sequence[float],
    context: str,
) -> None:
    """Authenticate the producer's direct serialized GB-to-byte identity."""

    _require(
        _vectors_close(
            observed_decimal_bytes, [value * 1e9 for value in observed_gb]
        ),
        f"{context}: conversion mismatch",
    )


def _validate_ssd_accounting_summary(
    summary: dict,
    boundaries: Sequence[dict],
    case_label: str,
    num_ssu: int,
) -> None:
    semantics = summary.get("timeline_ssd_accounting_semantics")
    _require(
        isinstance(semantics, dict)
        and semantics.get("schema") == TIMELINE_SCHEMA
        and "whole-command" in str(semantics.get("ssd_served_cumulative_gb"))
        and "active command prefix" in str(
            semantics.get("ssd_served_cumulative_gb")
        )
        and "direct math.fsum enumeration" in str(
            semantics.get("ssd_outstanding_gb")
        )
        and "not a scientific output" in str(
            semantics.get("fragmented_service_diagnostic")
        )
        and "observer-fragmentation-dependent" in str(
            semantics.get("fragmented_service_diagnostic")
        ),
        f"{case_label}: v3 stable-service/direct-outstanding semantics missing",
    )
    accounting = summary.get("measurement_ssd_accounting_residuals")
    _require(
        isinstance(accounting, dict)
        and accounting.get("schema") == SSD_ACCOUNTING_SCHEMA
        and _close(
            _number(
                accounting.get("service_absolute_tolerance_gb"),
                f"{case_label}.service_absolute_tolerance_gb",
            ),
            SSD_SERVICE_ABSOLUTE_TOLERANCE_GB,
        )
        and _integer(
            accounting.get("block_tolerance"), f"{case_label}.block_tolerance"
        )
        == 0
        and _close(
            _number(
                accounting.get("decimal_bytes_per_gb"),
                f"{case_label}.decimal_bytes_per_gb",
            ),
            1e9,
        ),
        f"{case_label}: measurement SSD accounting schema/tolerances mismatch",
    )
    stable_matrix = _matrix(
        summary.get("measurement_npu_ssu_ssd_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{case_label}.measurement_stable_ssd_service",
    )
    fragmented_matrix = _matrix(
        summary.get("measurement_fragmented_npu_ssu_ssd_served_gb"),
        NUM_NPU,
        num_ssu,
        f"{case_label}.measurement_fragmented_ssd_service",
    )
    busy_service = _vector(
        summary.get("measurement_ssd_served_gb_by_ssu"),
        num_ssu,
        f"{case_label}.measurement_busy_ssd_service",
    )
    assert stable_matrix is not None and fragmented_matrix is not None
    expected_measurement = _expected_measurement_ssd_accounting_residuals(
        stable_matrix, fragmented_matrix, busy_service, num_ssu
    )
    for field, expected in expected_measurement.items():
        observed = _vector(
            accounting.get(field), num_ssu, f"{case_label}.{field}", nonnegative=False
        )
        _require(
            _vectors_close(observed, expected),
            f"{case_label}: {field} does not recompute",
        )
        decimal_field = field.replace("_gb_by_ssu", "_decimal_bytes_by_ssu")
        decimals = _vector(
            accounting.get(decimal_field),
            num_ssu,
            f"{case_label}.{decimal_field}",
            nonnegative=False,
        )
        # The producer serializes decimal bytes by multiplying the already
        # computed/reported GB residual.  Keep this representation identity
        # separate from the independent GB reconstruction gate above.
        _validate_decimal_byte_identity(
            observed,
            decimals,
            f"{case_label}: {decimal_field}",
        )
    _require(
        max(
            abs(value)
            for value in expected_measurement[
                "stable_service_minus_busy_counter_gb_by_ssu"
            ]
        )
        <= SSD_SERVICE_ABSOLUTE_TOLERANCE_GB + 1e-15,
        f"{case_label}: whole-window stable service differs from busy counter",
    )
    _vector(
        accounting.get("busy_time_compensation_ms_by_ssu_at_stop"),
        num_ssu,
        f"{case_label}.busy_time_compensation",
        nonnegative=False,
    )

    measurement_maxima = accounting.get("measurement_maxima")
    _require(
        isinstance(measurement_maxima, dict)
        and set(measurement_maxima)
        == {"stable_service_minus_busy_counter", "fragmented_service_minus_stable"},
        f"{case_label}: measurement accounting maxima shape mismatch",
    )
    for name, field in (
        (
            "stable_service_minus_busy_counter",
            "stable_service_minus_busy_counter_gb_by_ssu",
        ),
        (
            "fragmented_service_minus_stable",
            "fragmented_service_minus_stable_gb_by_ssu",
        ),
    ):
        values = expected_measurement[field]
        ssu_id = max(range(num_ssu), key=lambda index: abs(values[index]))
        expected = values[ssu_id]
        maximum = measurement_maxima[name]
        _require(isinstance(maximum, dict), f"{case_label}: {name} maximum missing")
        _require(
            _integer(maximum.get("ssu_id"), f"{case_label}.{name}.ssu_id")
            == ssu_id
            and _close(
                _number(maximum.get("signed_gb"), f"{case_label}.{name}.signed"),
                expected,
            )
            and _close(
                _number(
                    maximum.get("absolute_gb"),
                    f"{case_label}.{name}.absolute",
                ),
                abs(expected),
            )
            and _close(
                _number(
                    maximum.get("signed_decimal_bytes"),
                    f"{case_label}.{name}.signed_bytes",
                ),
                expected * 1e9,
            )
            and _close(
                _number(
                    maximum.get("absolute_decimal_bytes"),
                    f"{case_label}.{name}.absolute_bytes",
                ),
                abs(expected) * 1e9,
            ),
            f"{case_label}: {name} measurement maximum mismatch",
        )

    timeline_maxima = accounting.get("timeline_maxima")
    float_fields = (
        "stable_service_minus_busy_counter_gb",
        "fragmented_service_minus_stable_gb",
        "physical_queue_minus_scheduler_gb",
        "enqueue_minus_service_minus_physical_queue_gb",
        "counter_queue_minus_physical_queue_gb",
        "fragmented_counter_queue_minus_physical_queue_gb",
        "maximum_abs_npu_queue_identity_residual_gb",
    )
    block_fields = (
        "physical_queue_block_minus_scheduler_blocks",
        "counter_queue_block_minus_physical_blocks",
    )
    _require(
        isinstance(timeline_maxima, dict)
        and set(timeline_maxima) == set(float_fields) | set(block_fields),
        f"{case_label}: timeline accounting maxima shape mismatch",
    )
    for field in float_fields:
        candidates = [
            (
                boundary_index,
                ssu_id,
                float(
                    boundary["timeline"]["ssd_accounting_residuals_by_ssu"][
                        ssu_id
                    ][field]
                ),
            )
            for boundary_index, boundary in enumerate(boundaries)
            for ssu_id in range(num_ssu)
        ]
        boundary_index, ssu_id, expected = max(
            candidates, key=lambda candidate: abs(candidate[2])
        )
        maximum = timeline_maxima[field]
        _require(isinstance(maximum, dict), f"{case_label}: {field} maximum missing")
        _require(
            _integer(maximum.get("boundary"), f"{case_label}.{field}.boundary")
            == boundary_index
            and _integer(maximum.get("ssu_id"), f"{case_label}.{field}.ssu_id")
            == ssu_id
            and _close(
                _number(maximum.get("elapsed_ms"), f"{case_label}.{field}.elapsed"),
                boundary_index * BLOCK_MS,
            )
            and _close(
                _number(maximum.get("signed_gb"), f"{case_label}.{field}.signed"),
                expected,
            )
            and _close(
                _number(
                    maximum.get("absolute_gb"), f"{case_label}.{field}.absolute"
                ),
                abs(expected),
            )
            and _close(
                _number(
                    maximum.get("signed_decimal_bytes"),
                    f"{case_label}.{field}.signed_bytes",
                ),
                expected * 1e9,
            )
            and _close(
                _number(
                    maximum.get("absolute_decimal_bytes"),
                    f"{case_label}.{field}.absolute_bytes",
                ),
                abs(expected) * 1e9,
            ),
            f"{case_label}: {field} timeline maximum mismatch",
        )
        if field == "maximum_abs_npu_queue_identity_residual_gb":
            expected_npu = boundaries[boundary_index]["timeline"][
                "ssd_accounting_residuals_by_ssu"
            ][ssu_id]["maximum_abs_npu_queue_identity_residual_npu_id"]
            _require(
                _integer(maximum.get("npu_id"), f"{case_label}.{field}.npu_id")
                == int(expected_npu),
                f"{case_label}: maximum queue-identity NPU mismatch",
            )
    for field in block_fields:
        candidates = [
            (
                boundary_index,
                ssu_id,
                int(
                    boundary["timeline"]["ssd_accounting_residuals_by_ssu"][
                        ssu_id
                    ][field]
                ),
            )
            for boundary_index, boundary in enumerate(boundaries)
            for ssu_id in range(num_ssu)
        ]
        boundary_index, ssu_id, expected = max(
            candidates, key=lambda candidate: abs(candidate[2])
        )
        maximum = timeline_maxima[field]
        _require(
            isinstance(maximum, dict)
            and _integer(maximum.get("boundary"), f"{case_label}.{field}.boundary")
            == boundary_index
            and _integer(maximum.get("ssu_id"), f"{case_label}.{field}.ssu_id")
            == ssu_id
            and _integer(
                maximum.get("signed_blocks"), f"{case_label}.{field}.signed"
            )
            == expected
            and _integer(
                maximum.get("absolute_blocks"), f"{case_label}.{field}.absolute"
            )
            == abs(expected),
            f"{case_label}: {field} timeline maximum mismatch",
        )


def _validate_state_durations(summary: dict, case_label: str) -> None:
    semantics = summary.get("timeline_state_duration_semantics")
    _require(
        isinstance(semantics, str)
        and "includes carry-in" in semantics
        and "exact window complement" in semantics,
        f"{case_label}: exact state-duration semantics missing",
    )
    rows = summary.get("timeline_state_durations_ms_by_npu")
    _require(
        isinstance(rows, list) and len(rows) == NUM_NPU,
        f"{case_label}: expected 32 state-duration rows",
    )
    compute_reported = _vector(
        summary.get("compute_ms_by_npu"),
        NUM_NPU,
        f"{case_label}.compute_ms_by_npu",
    )
    for npu_id, row in enumerate(rows):
        context = f"{case_label}.timeline_state_durations[{npu_id}]"
        _require(isinstance(row, dict), f"{context}: expected object")
        _require(_integer(row.get("npu_id"), f"{context}.npu_id") == npu_id, f"{context}: ID mismatch")
        compute = _number(row.get("compute_ms"), f"{context}.compute_ms")
        barrier = _number(row.get("io_barrier_ms"), f"{context}.io_barrier_ms")
        other = _number(row.get("other_ms"), f"{context}.other_ms")
        measurement = _number(row.get("measurement_ms"), f"{context}.measurement_ms")
        _require(min(compute, barrier, other) >= -TOL, f"{context}: negative state duration")
        _require(_close(measurement, MEASUREMENT_MS), f"{context}: measurement duration mismatch")
        _require(_close(compute + barrier + other, measurement, tolerance=2e-7), f"{context}: state partition does not close")
        _require(_close(compute, compute_reported[npu_id], tolerance=2e-7), f"{context}: compute duration != utilization numerator")
        for field, value in (
            ("compute_fraction", compute / measurement),
            ("io_barrier_fraction", barrier / measurement),
            ("other_fraction", other / measurement),
        ):
            _require(_close(_number(row.get(field), f"{context}.{field}"), value), f"{context}: {field} mismatch")


def _validate_block_state_durations(
    summary: dict, blocks: Sequence[dict], case_label: str
) -> None:
    rows = summary.get("timeline_block_state_durations_ms")
    _require(
        isinstance(rows, list) and len(rows) == BLOCK_COUNT,
        f"{case_label}: expected 128 exact block-state rows",
    )
    whole_rows = summary.get("timeline_state_durations_ms_by_npu")
    _require(
        isinstance(whole_rows, list) and len(whole_rows) == NUM_NPU,
        f"{case_label}: whole-window state rows unavailable",
    )
    accumulated = {
        "compute_ms": [0.0] * NUM_NPU,
        "io_barrier_ms": [0.0] * NUM_NPU,
        "other_ms": [0.0] * NUM_NPU,
    }
    measurement_start = _number(
        summary.get("measurement_start_ms"), f"{case_label}.measurement_start"
    )
    for block_index, row in enumerate(rows):
        context = f"{case_label}.timeline_block_state_durations[{block_index}]"
        _require(isinstance(row, dict), f"{context}: expected object")
        _require(
            _integer(row.get("block"), f"{context}.block") == block_index,
            f"{context}: block index mismatch",
        )
        start_ms = _number(row.get("start_ms"), f"{context}.start_ms")
        end_ms = _number(row.get("end_ms"), f"{context}.end_ms")
        duration_ms = _number(row.get("duration_ms"), f"{context}.duration_ms")
        _require(
            _close(start_ms, measurement_start + block_index * BLOCK_MS),
            f"{context}: start time mismatch",
        )
        _require(
            _close(end_ms, start_ms + BLOCK_MS)
            and _close(duration_ms, BLOCK_MS),
            f"{context}: duration mismatch",
        )
        vectors = {
            field: _vector(
                row.get(field), NUM_NPU, f"{context}.{field}"
            )
            for field in (
                "compute_ms_by_npu",
                "io_barrier_ms_by_npu",
                "other_ms_by_npu",
            )
        }
        expected_compute = _vector(
            blocks[block_index].get("compute_ms_by_npu"),
            NUM_NPU,
            f"{context}.stationarity_compute",
        )
        _require(
            _vectors_close(vectors["compute_ms_by_npu"], expected_compute),
            f"{context}: state compute != stationarity compute",
        )
        for npu_id in range(NUM_NPU):
            compute = vectors["compute_ms_by_npu"][npu_id]
            barrier = vectors["io_barrier_ms_by_npu"][npu_id]
            other = vectors["other_ms_by_npu"][npu_id]
            _require(
                min(compute, barrier, other) >= -TOL,
                f"{context}: NPU {npu_id} has negative state duration",
            )
            _require(
                _close(compute + barrier + other, duration_ms, tolerance=2e-7),
                f"{context}: NPU {npu_id} state partition does not close",
            )
            accumulated["compute_ms"][npu_id] += compute
            accumulated["io_barrier_ms"][npu_id] += barrier
            accumulated["other_ms"][npu_id] += other

    for npu_id, whole in enumerate(whole_rows):
        context = f"{case_label}.state_block_to_whole[{npu_id}]"
        for field in ("compute_ms", "io_barrier_ms", "other_ms"):
            _require(
                _close(
                    accumulated[field][npu_id],
                    _number(whole.get(field), f"{context}.{field}"),
                    tolerance=2e-7,
                ),
                f"{context}: block sum != whole-window {field}",
            )


def _interval_overlap_ms(
    interval_start_ms: float,
    interval_end_ms: float,
    window_start_ms: float,
    window_end_ms: float,
) -> float:
    return max(
        0.0,
        min(interval_end_ms, window_end_ms)
        - max(interval_start_ms, window_start_ms),
    )


def _validate_state_durations_from_lifecycles(
    summary: dict,
    blocks: Sequence[dict],
    request_rows: Sequence[dict],
    case_label: str,
) -> None:
    """Independently reproduce exact compute/barrier/other state durations.

    Measurement-cohort request rows contain every batch admitted in the
    half-open 64-second window.  ``timeline_carry_in_batches`` contains the
    only additional batches that can intersect it.  We deliberately rebuild
    both the whole window and every 500-ms block from those layer intervals;
    the producer's state-duration aggregates are not used as inputs.
    """

    start = _number(summary.get("measurement_start_ms"), f"{case_label}.start")
    end = _number(summary.get("measurement_end_ms"), f"{case_label}.end")
    _require(
        summary.get("timeline_carry_in_batches_schema")
        == "steady_timeline_carry_in_batch_v1",
        f"{case_label}: carry-in schema mismatch",
    )
    semantics = summary.get("timeline_carry_in_batch_semantics")
    _require(
        isinstance(semantics, str)
        and "admission_time_ms < measurement_start_ms <= completion_time_ms"
        in semantics
        and "at most one per NPU" in semantics,
        f"{case_label}: carry-in semantics missing",
    )
    carry_rows = summary.get("timeline_carry_in_batches")
    _require(
        isinstance(carry_rows, list) and len(carry_rows) <= NUM_NPU,
        f"{case_label}: invalid carry-in batch collection",
    )
    authenticated = _current_materialized_request_metadata(
        _integer(summary.get("num_ssu"), f"{case_label}.num_ssu")
    )
    seen_batch_ids: set[int] = set()
    seen_request_ids = {
        _integer(row.get("request_id"), f"{case_label}.request.request_id")
        for row in request_rows
    }
    seen_carry_npus: set[int] = set()
    lifecycle_rows: list[dict] = []

    for index, row in enumerate(carry_rows):
        context = f"{case_label}.timeline_carry_in_batches[{index}]"
        _require(isinstance(row, dict), f"{context}: expected object")
        batch_id = _integer(row.get("batch_id"), f"{context}.batch_id")
        npu_id = _integer(row.get("npu_id"), f"{context}.npu_id")
        _require(
            batch_id not in seen_batch_ids and npu_id not in seen_carry_npus,
            f"{context}: duplicate carry-in batch or NPU",
        )
        seen_batch_ids.add(batch_id)
        seen_carry_npus.add(npu_id)
        _require(0 <= npu_id < NUM_NPU, f"{context}: NPU out of range")
        request_ids_raw = row.get("request_ids")
        _require(
            isinstance(request_ids_raw, list)
            and len(request_ids_raw) == BATCH_SIZE,
            f"{context}: carry-in request membership mismatch",
        )
        request_ids = [
            _integer(value, f"{context}.request_ids")
            for value in request_ids_raw
        ]
        _require(
            len(set(request_ids)) == len(request_ids)
            and not (set(request_ids) & seen_request_ids)
            and all(
                request_id in authenticated
                and authenticated[request_id]["npu_id"] == npu_id
                for request_id in request_ids
            ),
            f"{context}: carry-in request identity is duplicate or unauthenticated",
        )
        seen_request_ids.update(request_ids)
        admission = _number(row.get("admission_time_ms"), f"{context}.admission")
        completion = _number(
            row.get("completion_time_ms"), f"{context}.completion"
        )
        _require(
            admission < start and completion >= start,
            f"{context}: row does not satisfy the exact left-limit carry-in definition",
        )
        _require(
            _integer(row.get("layer_count"), f"{context}.layer_count")
            == N_LAYERS,
            f"{context}: layer count mismatch",
        )
        vectors = {
            field: _vector(row.get(field), N_LAYERS, f"{context}.{field}")
            for field in (
                "io_ready_time_ms",
                "compute_start_ms",
                "compute_end_ms",
                "compute_duration_ms",
                "io_barrier_wait_ms",
            )
        }
        previous_end = admission
        compute_sum = 0.0
        for layer in range(N_LAYERS):
            compute_start = vectors["compute_start_ms"][layer]
            compute_end = vectors["compute_end_ms"][layer]
            io_ready = vectors["io_ready_time_ms"][layer]
            duration = vectors["compute_duration_ms"][layer]
            wait = vectors["io_barrier_wait_ms"][layer]
            _require(
                _close(
                    compute_start,
                    max(previous_end, io_ready),
                    tolerance=2e-6,
                )
                and _close(
                    wait,
                    compute_start - previous_end,
                    tolerance=2e-6,
                )
                and _close(
                    duration,
                    compute_end - compute_start,
                    tolerance=2e-6,
                )
                and compute_end > compute_start,
                f"{context}: layer {layer} interval closure mismatch",
            )
            compute_sum += duration
            previous_end = compute_end
        expected_ideal = float(authenticated[request_ids[0]]["ideal_ttft_ms"])
        reported_per_layer_compute = _number(
            row.get("per_layer_compute_ms"),
            f"{context}.per_layer_compute_ms",
        )
        reported_ideal_compute = _number(
            row.get("ideal_compute_ms"), f"{context}.ideal_compute_ms"
        )
        _require(
            _close(compute_sum, expected_ideal, tolerance=2e-6)
            and _close(previous_end, completion, tolerance=2e-6)
            and _close(
                reported_per_layer_compute,
                expected_ideal / N_LAYERS,
                tolerance=2e-9,
            )
            and _close(
                reported_ideal_compute, expected_ideal, tolerance=2e-6
            ),
            f"{context}: carry-in compute/completion closure mismatch",
        )
        _require(
            all(
                _close(
                    duration,
                    expected_ideal / N_LAYERS,
                    tolerance=2e-6,
                )
                for duration in vectors["compute_duration_ms"]
            ),
            f"{context}: carry-in per-layer compute differs from authenticated fixed profile",
        )
        lifecycle_rows.append(
            {
                "npu_id": npu_id,
                "admission": admission,
                "completion": completion,
                "compute_start": vectors["compute_start_ms"],
                "compute_end": vectors["compute_end_ms"],
            }
        )

    # The left-boundary snapshot independently authenticates the exact carry-in
    # set.  Omitting a carry-in row and editing the aggregates together must not
    # be able to pass this validator.
    first_boundary = summary["measurement_stationarity_boundaries"][0]
    first_npu_rows = first_boundary["timeline"]["npu_rows"]
    expected_carry = {
        (
            _integer(row["npu_id"], f"{case_label}.left_boundary.npu_id"),
            _integer(
                row["active_request_id"],
                f"{case_label}.left_boundary.active_request_id",
            ),
        )
        for row in first_npu_rows
        if row.get("active_request_id") is not None
        and row.get("admission_time_ms") is not None
        and _number(
            row["admission_time_ms"], f"{case_label}.left_boundary.admission"
        )
        < start
    }
    actual_carry = {
        (int(row["npu_id"]), int(row["request_ids"][0])) for row in carry_rows
    }
    _require(
        actual_carry == expected_carry,
        f"{case_label}: carry-in rows do not match the left-boundary active set",
    )
    for row in carry_rows:
        if float(row["completion_time_ms"]) == start:
            for layer_start, layer_end in zip(
                row["compute_start_ms"], row["compute_end_ms"]
            ):
                _require(
                    _interval_overlap_ms(
                        float(layer_start), float(layer_end), start, end
                    )
                    == 0.0,
                    f"{case_label}: equality carry-in contributes nonzero compute to the half-open window",
                )

    requests_by_npu: dict[int, list[dict]] = defaultdict(list)
    for row in request_rows:
        requests_by_npu[int(row["npu_id"])].append(row)
    carry_by_npu = {int(row["npu_id"]): row for row in carry_rows}
    for npu_id in range(NUM_NPU):
        ordered = sorted(
            requests_by_npu[npu_id], key=lambda row: int(row["sequence"])
        )
        sequences = [int(row["sequence"]) for row in ordered]
        _require(
            sequences
            and sequences
            == list(range(sequences[0], sequences[0] + len(sequences))),
            f"{case_label}: NPU {npu_id} measurement request sequence has a gap",
        )
        if npu_id in carry_by_npu:
            carry_request_id = int(carry_by_npu[npu_id]["request_ids"][0])
            _require(
                int(authenticated[carry_request_id]["sequence"]) + 1
                == sequences[0],
                f"{case_label}: NPU {npu_id} carry-in is not the measurement predecessor",
            )
        final_row = summary["measurement_stationarity_boundaries"][-1][
            "timeline"
        ]["npu_rows"][npu_id]
        final_active = final_row.get("active_request_id")
        if final_active is not None:
            _require(
                int(final_active) == int(ordered[-1]["request_id"]),
                f"{case_label}: NPU {npu_id} right-boundary active request is not the final measurement lifecycle",
            )
        first_activated = _number(
            first_npu_rows[npu_id]["activated_compute_cumulative_ms"],
            f"{case_label}.left_boundary.activated_compute",
        )
        last_activated = _number(
            final_row["activated_compute_cumulative_ms"],
            f"{case_label}.right_boundary.activated_compute",
        )
        expected_window_activated = math.fsum(
            float(row["ideal_ttft_ms"]) for row in ordered
        )
        _require(
            _close(
                last_activated - first_activated,
                expected_window_activated,
                tolerance=2e-6,
            ),
            f"{case_label}: NPU {npu_id} activated-compute boundary delta does not cover exactly the measurement request cohort",
        )

    for row in request_rows:
        layers = row["timeline_layers"]
        lifecycle_rows.append(
            {
                "npu_id": int(row["npu_id"]),
                "admission": float(row["admission_time_ms"]),
                "completion": float(row["completion_time_ms"]),
                "compute_start": [float(value) for value in layers["compute_start_ms"]],
                "compute_end": [float(value) for value in layers["compute_end_ms"]],
            }
        )

    expected_compute = [[0.0] * NUM_NPU for _ in range(BLOCK_COUNT)]
    expected_barrier = [[0.0] * NUM_NPU for _ in range(BLOCK_COUNT)]
    intervals_by_npu: list[list[tuple[float, float, str]]] = [
        [] for _ in range(NUM_NPU)
    ]
    for lifecycle in lifecycle_rows:
        npu_id = int(lifecycle["npu_id"])
        barrier_start = float(lifecycle["admission"])
        for compute_start, compute_end in zip(
            lifecycle["compute_start"], lifecycle["compute_end"]
        ):
            intervals_by_npu[npu_id].append(
                (barrier_start, float(compute_start), "io_barrier")
            )
            intervals_by_npu[npu_id].append(
                (float(compute_start), float(compute_end), "compute")
            )
            for block_index in range(BLOCK_COUNT):
                block_start = start + block_index * BLOCK_MS
                block_end = block_start + BLOCK_MS
                expected_barrier[block_index][npu_id] += _interval_overlap_ms(
                    barrier_start, float(compute_start), block_start, block_end
                )
                expected_compute[block_index][npu_id] += _interval_overlap_ms(
                    float(compute_start),
                    float(compute_end),
                    block_start,
                    block_end,
                )
            barrier_start = float(compute_end)

    for npu_id, intervals in enumerate(intervals_by_npu):
        clipped = sorted(
            (max(left, start), min(right, end), kind)
            for left, right, kind in intervals
            if right > start and left < end and right > left
        )
        for previous_interval, current_interval in zip(clipped, clipped[1:]):
            _require(
                current_interval[0] >= previous_interval[1] - 2e-6,
                f"{case_label}: independently reconstructed NPU {npu_id} lifecycles overlap",
            )

    state_blocks = summary["timeline_block_state_durations_ms"]
    for block_index, state_block in enumerate(state_blocks):
        reported_compute = _vector(
            state_block["compute_ms_by_npu"],
            NUM_NPU,
            f"{case_label}.independent_state.block_compute",
        )
        reported_barrier = _vector(
            state_block["io_barrier_ms_by_npu"],
            NUM_NPU,
            f"{case_label}.independent_state.block_barrier",
        )
        reported_other = _vector(
            state_block["other_ms_by_npu"],
            NUM_NPU,
            f"{case_label}.independent_state.block_other",
        )
        expected_other = [
            BLOCK_MS
            - expected_compute[block_index][npu_id]
            - expected_barrier[block_index][npu_id]
            for npu_id in range(NUM_NPU)
        ]
        _require(
            _vectors_close(reported_compute, expected_compute[block_index])
            and _vectors_close(reported_barrier, expected_barrier[block_index])
            and _vectors_close(reported_other, expected_other),
            f"{case_label}: 500-ms state durations differ from independent lifecycle reconstruction",
        )
        _require(
            _vectors_close(
                _vector(
                    blocks[block_index]["compute_ms_by_npu"],
                    NUM_NPU,
                    f"{case_label}.stationarity_compute",
                ),
                expected_compute[block_index],
            ),
            f"{case_label}: stationarity compute differs from independent lifecycle reconstruction",
        )

    whole_rows = summary["timeline_state_durations_ms_by_npu"]
    for npu_id in range(NUM_NPU):
        compute = math.fsum(row[npu_id] for row in expected_compute)
        barrier = math.fsum(row[npu_id] for row in expected_barrier)
        other = MEASUREMENT_MS - compute - barrier
        reported = whole_rows[npu_id]
        _require(
            _close(_number(reported["compute_ms"], "state.compute"), compute)
            and _close(_number(reported["io_barrier_ms"], "state.barrier"), barrier)
            and _close(_number(reported["other_ms"], "state.other"), other),
            f"{case_label}: whole-window state durations differ from independent lifecycle reconstruction",
        )


def _replay_route_probe(
    probe: Mapping[str, object], case: str, context: str
) -> tuple[int, ...]:
    rule = probe.get("rule")
    sizes = tuple(
        _number(value, f"{context}.block_sizes_gb")
        for value in probe.get("block_sizes_gb", [])
    )
    allowed = tuple(
        _integer(value, f"{context}.allowed_path_ids")
        for value in probe.get("allowed_path_ids", [])
    )
    _require(sizes and allowed, f"{context}: empty route replay input")
    if rule == "fixed_path_zero":
        return (0,) * len(sizes)
    if rule == "npu_dedicated_path":
        _require(len(allowed) == 1, f"{context}: dedicated route has !=1 allowed path")
        return (allowed[0],) * len(sizes)
    _require(
        case == "layer_once_ttl_5ms"
        and rule == "pressure_aware_once_per_layer",
        f"{context}: unsupported pressure-aware route rule",
    )
    path_count = _integer(probe.get("path_count"), f"{context}.path_count")
    paths_per_group_reported = _integer(
        probe.get("paths_per_group"), f"{context}.paths_per_group"
    )
    group_weights = tuple(
        _number(value, f"{context}.group_weights")
        for value in probe.get("group_weights", [])
    )
    _require(
        group_weights and path_count % len(group_weights) == 0,
        f"{context}: invalid group weights/path count",
    )
    paths_per_group = path_count // len(group_weights)
    _require(
        paths_per_group_reported == paths_per_group,
        f"{context}: paths-per-group mismatch",
    )
    allowed_counts = _vector(
        probe.get("allowed_path_pressure_counts"),
        len(allowed),
        f"{context}.allowed_path_pressure_counts",
        integer=True,
    )
    allowed_cirs = _vector(
        probe.get("allowed_path_cir_gbps"),
        len(allowed),
        f"{context}.allowed_path_cir_gbps",
    )
    pir_raw = probe.get("allowed_path_pir_gbps_or_null")
    _require(isinstance(pir_raw, list) and len(pir_raw) == len(allowed), f"{context}: PIR vector shape")
    allowed_pirs = [
        math.inf if value is None else _number(value, f"{context}.allowed_path_pir")
        for value in pir_raw
    ]
    allowed_weights = _vector(
        probe.get("allowed_path_weights"),
        len(allowed),
        f"{context}.allowed_path_weights",
    )
    allowed_groups = _vector(
        probe.get("allowed_path_group_ids"),
        len(allowed),
        f"{context}.allowed_path_group_ids",
        integer=True,
    )
    _require(
        all(group == path // paths_per_group for path, group in zip(allowed, allowed_groups)),
        f"{context}: allowed path/group mapping mismatch",
    )
    path_cirs = [0.0] * path_count
    path_pirs = [math.inf] * path_count
    path_weights = [0.0] * path_count
    counts = [0] * path_count
    for path, count, cir, pir, weight in zip(
        allowed, allowed_counts, allowed_cirs, allowed_pirs, allowed_weights
    ):
        _require(0 <= path < path_count, f"{context}: path ID out of range")
        counts[path] = int(count)
        path_cirs[path] = float(cir)
        path_pirs[path] = float(pir)
        path_weights[path] = float(weight)
    group_io_counts = tuple(
        _integer(value, f"{context}.group_io_counts")
        for value in probe.get("group_io_counts", [])
    )
    active_paths_per_group = tuple(
        _integer(value, f"{context}.active_paths_per_group")
        for value in probe.get("active_paths_per_group", [])
    )
    active_path_weights = tuple(
        _number(value, f"{context}.active_path_weights")
        for value in probe.get("active_path_weights", [])
    )
    _require(
        len(group_io_counts)
        == len(active_paths_per_group)
        == len(active_path_weights)
        == len(group_weights),
        f"{context}: group aggregate shape mismatch",
    )
    snapshot = PathPressureSnapshot(
        counts=tuple(counts),
        group_io_counts=group_io_counts,
        active_paths_per_group=active_paths_per_group,
        active_path_weights=active_path_weights,
        active_group_weight_sum=_number(
            probe.get("active_group_weight_sum"),
            f"{context}.active_group_weight_sum",
        ),
        active_cir_sum=_number(
            probe.get("active_cir_sum_gbps"),
            f"{context}.active_cir_sum_gbps",
        ),
    )
    qos = QoSHardwareView(
        path_cirs=tuple(path_cirs),
        path_pirs=tuple(path_pirs),
        path_weights=tuple(path_weights),
        group_weights=group_weights,
        category_paths_per_group=(paths_per_group,),
        category_labels=("probe",),
    )
    return pressure_aware_path_ids(
        sizes,
        snapshot,
        allowed,
        qos,
        disk_bw_gbps=_number(
            probe.get("disk_bw_gbps"), f"{context}.disk_bw_gbps"
        ),
        start_offset=_integer(
            probe.get("start_offset"), f"{context}.start_offset"
        ),
    )


def _validate_previous_pinned_order(
    *,
    pinned: Sequence[int],
    previous_pairs: Sequence[tuple[int, int]],
    request_map: Mapping[int, int],
    previous_selected_order: Sequence[int],
    context: str,
) -> None:
    """Bind pins to the prior allocation order, filtered by request identity."""

    previous_map = dict(previous_pairs)
    _require(
        len(previous_map) == len(previous_pairs)
        and len(previous_selected_order) == len(set(previous_selected_order))
        and set(previous_selected_order) == set(previous_map),
        f"{context}: previous selected order/map mismatch",
    )
    expected_pinned = [
        npu_id
        for npu_id in previous_selected_order
        if request_map.get(npu_id) == previous_map[npu_id]
    ]
    _require(
        list(pinned) == expected_pinned,
        f"{context}: pinned NPUs do not preserve prior selection order for continuing requests",
    )


def _validate_admission_diagnostic(
    record: Mapping[str, object],
    num_ssu: int,
    context: str,
    *,
    previous_selected_order: Sequence[int] | None = None,
) -> None:
    def npu_vector(field: str) -> list[int]:
        raw = record.get(field)
        _require(isinstance(raw, list), f"{context}.{field}: expected list")
        values = [_integer(value, f"{context}.{field}") for value in raw]
        _require(
            len(values) == len(set(values))
            and all(0 <= value < NUM_NPU for value in values),
            f"{context}.{field}: duplicate/out-of-range NPU",
        )
        return values

    active = npu_vector("active_npu_ids")
    selected = npu_vector("selected_npu_ids")
    rejected = npu_vector("rejected_npu_ids")

    work = _matrix(
        record.get("remaining_work_gb_by_npu_ssu"),
        NUM_NPU,
        num_ssu,
        f"{context}.remaining_work",
    )
    compute_s = _vector(
        record.get("remaining_compute_s_by_npu"),
        NUM_NPU,
        f"{context}.remaining_compute_s",
    )
    demand = _matrix(
        record.get("controller_demand_gbps_by_npu_ssu"),
        NUM_NPU,
        num_ssu,
        f"{context}.controller_demand",
    )
    grants = _matrix(
        record.get("grants_gbps_by_npu_ssu"),
        NUM_NPU,
        num_ssu,
        f"{context}.grants",
    )
    assert work is not None and demand is not None and grants is not None
    expected_demand = [
        [
            amount / compute_s[npu_id] if compute_s[npu_id] > 0.0 else 0.0
            for amount in work[npu_id]
        ]
        for npu_id in range(NUM_NPU)
    ]
    _require(
        _matrices_close(demand, expected_demand),
        f"{context}: controller demand != remaining work / remaining compute",
    )
    for npu_id in range(NUM_NPU):
        has_work = any(value > 0.0 for value in work[npu_id])
        has_compute = compute_s[npu_id] > 0.0
        has_demand = any(value > 0.0 for value in demand[npu_id])
        _require(
            has_work == has_compute == has_demand,
            f"{context}: NPU {npu_id} work/compute/demand active-state equivalence mismatch",
        )
        if not has_work:
            _require(
                compute_s[npu_id] == 0.0
                and all(value == 0.0 for value in work[npu_id])
                and all(value == 0.0 for value in demand[npu_id]),
                f"{context}: NPU {npu_id} inactive work/compute/demand row is not exact zero",
            )
    expected_active = [
        npu_id
        for npu_id, row in enumerate(demand)
        if math.fsum(row) > 1e-12
    ]
    _require(
        active == expected_active,
        f"{context}: active NPU IDs do not match positive controller demand",
    )

    def sorted_pairs(field: str) -> list[tuple[int, int]]:
        raw = record.get(field)
        _require(isinstance(raw, list), f"{context}.{field}: expected pair list")
        parsed: list[tuple[int, int]] = []
        for index, item in enumerate(raw):
            pair_context = f"{context}.{field}[{index}]"
            _require(
                isinstance(item, (list, tuple)) and len(item) == 2,
                f"{pair_context}: expected [npu_id, request_id]",
            )
            parsed.append(
                (
                    _integer(item[0], f"{pair_context}.npu_id"),
                    _integer(item[1], f"{pair_context}.request_id"),
                )
            )
        _require(
            parsed == sorted(parsed) and len(parsed) == len(set(parsed)),
            f"{context}.{field}: pairs are not unique and sorted",
        )
        return parsed

    request_pairs = sorted_pairs("request_by_npu")
    previous_pairs = sorted_pairs("previous_selected_request_by_npu")
    authenticated_requests = _current_materialized_request_metadata(num_ssu)
    for field, pairs in (
        ("request_by_npu", request_pairs),
        ("previous_selected_request_by_npu", previous_pairs),
    ):
        _require(
            all(
                request_id in authenticated_requests
                and authenticated_requests[request_id]["npu_id"] == npu_id
                for npu_id, request_id in pairs
            ),
            f"{context}: {field} contains a request not authenticated to its NPU",
        )
    request_map = dict(request_pairs)
    _require(
        len(request_map) == len(request_pairs)
        and set(request_map) == set(expected_active),
        f"{context}: request mapping does not cover the active demand rows",
    )
    prefetch_raw = record.get("prefetch_only_by_npu")
    _require(
        isinstance(prefetch_raw, list),
        f"{context}.prefetch_only_by_npu: expected pair list",
    )
    prefetch_map: dict[int, bool] = {}
    for index, item in enumerate(prefetch_raw):
        item_context = f"{context}.prefetch_only_by_npu[{index}]"
        _require(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and type(item[1]) is bool,
            f"{item_context}: expected [npu_id, bool]",
        )
        npu_id = _integer(item[0], f"{item_context}.npu_id")
        _require(
            npu_id not in prefetch_map,
            f"{item_context}: duplicate NPU",
        )
        prefetch_map[npu_id] = bool(item[1])
    _require(
        list(prefetch_map) == sorted(prefetch_map)
        and set(prefetch_map) == set(request_map),
        f"{context}: prefetch classification does not exactly cover request mapping",
    )
    for npu_id, request_id in request_pairs:
        _authenticated_remaining_layer_count(
            num_ssu=num_ssu,
            npu_id=npu_id,
            request_id=request_id,
            remaining_compute_ms=compute_s[npu_id] * 1000.0,
            remaining_work_gb_by_ssu=work[npu_id],
            context=f"{context}: NPU {npu_id}",
        )
        # A pre-admission request can already have layer-0 I/O ready because
        # cross-request prefetch runs before admission.  Therefore prefetch
        # identity masks TTFT diagnostics, but does not imply a fixed 16-layer
        # remaining inventory.  The unique authenticated integer inventory
        # above is the exact producer contract for both admitted and prefetch
        # views.
    pinned = npu_vector("previous_pinned_npu_ids")
    _require(
        previous_selected_order is not None or not previous_pairs,
        f"{context}: prior selected order is required for a non-initial decision",
    )
    _validate_previous_pinned_order(
        pinned=pinned,
        previous_pairs=previous_pairs,
        request_map=request_map,
        previous_selected_order=(
            []
            if previous_selected_order is None
            else previous_selected_order
        ),
        context=context,
    )

    replay = replay_admission_selection(
        demand,
        target_ratio=0.52,
        required_ratio=0.5,
        background_reserve_fraction=0.05,
        pinned_npu_ids=pinned,
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=NPU_LINK_CAP_GBPS,
    )
    expected_replay_fields: dict[str, object] = {
        "selection_mode": replay.selection_mode,
        "effective_target_ratio": replay.effective_target_ratio,
        "active_npu_ids": replay.active_npu_ids,
        "candidate_normalized_scores": tuple(
            asdict(item) for item in replay.candidate_scores
        ),
        "candidate_order": replay.candidate_order,
        "admission_attempts": tuple(asdict(item) for item in replay.attempts),
        "selected_npu_ids": replay.selected_npu_ids,
        "rejected_npu_ids": replay.rejected_npu_ids,
        "capacity_rejections": tuple(
            asdict(item) for item in replay.capacity_rejections
        ),
    }
    for field, expected in expected_replay_fields.items():
        _require(
            _canonical(record.get(field)) == _canonical(expected),
            f"{context}: {field} differs from exact causal admission replay",
        )

    allocation = allocate_adaptive_admission_grants(
        demand,
        explicit_spill_threshold=0.75,
        target_ratio=0.52,
        required_ratio=0.5,
        background_reserve_fraction=0.05,
        pinned_npu_ids=pinned,
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=NPU_LINK_CAP_GBPS,
    )
    _require(
        _matrices_close(grants, allocation.grants_gbps),
        f"{context}: grants differ from exact causal allocation replay",
    )
    v2 = allocation.v2_allocation
    component_fields = (
        ("v2_floor_grants_gbps", "floor_grants_gbps"),
        ("v2_background_grants_gbps", "background_grants_gbps"),
        ("v2_selected_tail_grants_gbps", "selected_tail_grants_gbps"),
        ("v2_spill_tail_grants_gbps", "spill_tail_grants_gbps"),
    )
    if v2 is None:
        _require(
            record.get("v2_effective_floor_ratio") is None
            and all(record.get(field) is None for field, _ in component_fields),
            f"{context}: V1 residual mode unexpectedly exposes V2 components",
        )
    else:
        _require(
            _close(
                _number(
                    record.get("v2_effective_floor_ratio"),
                    f"{context}.v2_effective_floor_ratio",
                ),
                v2.effective_floor_ratio,
            ),
            f"{context}: V2 effective floor ratio differs from replay",
        )
        component_matrices: list[list[list[float]]] = []
        for raw_field, allocation_field in component_fields:
            raw_matrix = _matrix(
                record.get(raw_field),
                NUM_NPU,
                num_ssu,
                f"{context}.{raw_field}",
            )
            assert raw_matrix is not None
            expected_matrix = getattr(v2, allocation_field)
            _require(
                _matrices_close(raw_matrix, expected_matrix),
                f"{context}: {raw_field} differs from exact V2 allocation replay",
            )
            component_matrices.append(raw_matrix)
        _require(
            all(
                _close(
                    math.fsum(
                        component[npu_id][ssu_id]
                        for component in component_matrices
                    ),
                    grants[npu_id][ssu_id],
                )
                for npu_id in range(NUM_NPU)
                for ssu_id in range(num_ssu)
            ),
            f"{context}: V2 grant components do not sum to final grant",
        )
    _require(
        _close(
            _number(record.get("selected_fraction"), f"{context}.selected_fraction"),
            allocation.selected_fraction,
        )
        and record.get("residual_mode") == allocation.residual_mode,
        f"{context}: selected fraction/residual mode differs from allocation replay",
    )
    _require(
        all(
            math.fsum(grants[npu_id]) <= NPU_LINK_CAP_GBPS + TOL
            for npu_id in range(NUM_NPU)
        )
        and all(
            math.fsum(grants[npu_id][ssu_id] for npu_id in range(NUM_NPU))
            <= SSD_CAP_GBPS + TOL
            for ssu_id in range(num_ssu)
        ),
        f"{context}: grants exceed NPU-link or SSD capacity",
    )
    _require(
        not (set(selected) & set(rejected))
        and set(selected) | set(rejected) == set(active),
        f"{context}: selected/rejected are not a disjoint closure of active NPUs",
    )
    candidate_order = npu_vector("candidate_order")
    scores_raw = record.get("candidate_normalized_scores")
    _require(isinstance(scores_raw, list), f"{context}: candidate scores missing")
    score_by_npu: dict[int, tuple[float, float]] = {}
    for index, score in enumerate(scores_raw):
        score_context = f"{context}.candidate_normalized_scores[{index}]"
        _require(isinstance(score, dict), f"{score_context}: expected object")
        npu_id = _integer(score.get("npu_id"), f"{score_context}.npu_id")
        _require(
            npu_id in active and npu_id not in score_by_npu,
            f"{score_context}: duplicate/non-active candidate",
        )
        score_by_npu[npu_id] = (
            _number(score.get("normalized_total"), f"{score_context}.total"),
            _number(
                score.get("normalized_dominant"), f"{score_context}.dominant"
            ),
        )
    _require(
        min((value for pair in score_by_npu.values() for value in pair), default=0.0)
        >= -TOL,
        f"{context}: negative candidate score",
    )
    mode = record.get("selection_mode")
    _require(
        mode
        in (
            "all_preferred_targets_feasible",
            "all_required_targets_feasible",
            "greedy_overload",
        ),
        f"{context}: unknown selection mode",
    )
    attempts = record.get("admission_attempts")
    _require(isinstance(attempts, list), f"{context}: admission attempts missing")
    if mode != "greedy_overload":
        _require(
            not candidate_order
            and not attempts
            and not rejected
            and set(selected) == set(active),
            f"{context}: all-feasible selection unexpectedly has admission attempts/rejections",
        )
        return

    _require(
        set(candidate_order) == set(score_by_npu)
        and candidate_order
        == sorted(
            candidate_order,
            key=lambda npu: (
                score_by_npu[npu][0],
                score_by_npu[npu][1],
                npu,
            ),
        ),
        f"{context}: candidate order does not match normalized score order",
    )
    parsed_attempts: list[dict] = []
    prior_remaining_after: list[float] | None = None
    for index, attempt in enumerate(attempts):
        attempt_context = f"{context}.admission_attempts[{index}]"
        _require(isinstance(attempt, dict), f"{attempt_context}: expected object")
        _require(
            _integer(attempt.get("attempt_index"), f"{attempt_context}.index")
            == index,
            f"{attempt_context}: attempt indices are not contiguous",
        )
        npu_id = _integer(attempt.get("npu_id"), f"{attempt_context}.npu_id")
        _require(npu_id in active, f"{attempt_context}: attempt NPU is not active")
        stage = attempt.get("stage")
        _require(
            stage in ("pinned", "greedy_candidate"),
            f"{attempt_context}: unknown admission stage",
        )
        accepted = attempt.get("accepted")
        _require(type(accepted) is bool, f"{attempt_context}: accepted is not bool")
        reason = attempt.get("rejection_reason")
        _require(
            (accepted and reason is None)
            or (
                not accepted
                and reason
                in (
                    "empty_demand",
                    "npu_target_exceeds_capacity",
                    "ssu_admission_capacity_exceeded",
                )
            ),
            f"{attempt_context}: acceptance/rejection reason mismatch",
        )
        target = _vector(
            attempt.get("target_gbps_by_ssu"),
            num_ssu,
            f"{attempt_context}.target",
        )
        before = _vector(
            attempt.get("admission_remaining_before_gbps_by_ssu"),
            num_ssu,
            f"{attempt_context}.remaining_before",
        )
        after = _vector(
            attempt.get("admission_remaining_after_gbps_by_ssu"),
            num_ssu,
            f"{attempt_context}.remaining_after",
        )
        _require(
            _close(
                math.fsum(target),
                _number(attempt.get("target_sum_gbps"), f"{attempt_context}.target_sum"),
            ),
            f"{attempt_context}: target sum mismatch",
        )
        target_sum = math.fsum(target)
        npu_capacity = _number(
            attempt.get("npu_capacity_gbps"),
            f"{attempt_context}.npu_capacity_gbps",
        )
        _require(npu_capacity > 0.0, f"{attempt_context}: NPU capacity is not positive")
        if reason == "npu_target_exceeds_capacity":
            _require(
                target_sum > npu_capacity + 1e-12,
                f"{attempt_context}: NPU-capacity rejection is unexplained",
            )
        elif reason != "empty_demand":
            _require(
                target_sum <= npu_capacity + 1e-12,
                f"{attempt_context}: target exceeds NPU capacity without matching rejection",
            )
        if prior_remaining_after is not None:
            _require(
                _vectors_close(before, prior_remaining_after),
                f"{attempt_context}: admission remaining is discontinuous",
            )
        expected_after = (
            [left - amount for left, amount in zip(before, target)]
            if accepted
            else before
        )
        _require(
            _vectors_close(after, expected_after),
            f"{attempt_context}: capacity remaining transition mismatch",
        )
        prior_remaining_after = list(after)
        violating_raw = attempt.get("violating_ssu_ids")
        _require(isinstance(violating_raw, list), f"{attempt_context}: violating SSUs missing")
        violating = [
            _integer(value, f"{attempt_context}.violating_ssu_ids")
            for value in violating_raw
        ]
        _require(
            violating == sorted(set(violating))
            and all(0 <= value < num_ssu for value in violating),
            f"{attempt_context}: invalid violating SSU list",
        )
        expected_violating = [
            ssu for ssu in range(num_ssu) if target[ssu] > before[ssu] + 1e-12
        ]
        if reason == "ssu_admission_capacity_exceeded":
            _require(
                violating == expected_violating and violating,
                f"{attempt_context}: violating SSUs do not explain rejection",
            )
        else:
            _require(not violating, f"{attempt_context}: unexpected violating SSUs")
        parsed_attempts.append(
            {
                "source": attempt,
                "npu_id": npu_id,
                "stage": stage,
                "accepted": accepted,
            }
        )

    _require(
        [row["npu_id"] for row in parsed_attempts if row["stage"] == "pinned"]
        == pinned,
        f"{context}: pinned admission-attempt order mismatch",
    )
    _require(
        [
            row["npu_id"]
            for row in parsed_attempts
            if row["stage"] == "greedy_candidate"
        ]
        == candidate_order,
        f"{context}: greedy attempt order != candidate order",
    )
    last_attempt_by_npu: dict[int, dict] = {}
    accepted_order: list[int] = []
    for row in parsed_attempts:
        last_attempt_by_npu[int(row["npu_id"])] = row
        if row["accepted"]:
            accepted_order.append(int(row["npu_id"]))
    _require(
        set(last_attempt_by_npu) == set(active)
        and all(last_attempt_by_npu[npu]["accepted"] for npu in selected)
        and all(not last_attempt_by_npu[npu]["accepted"] for npu in rejected)
        and accepted_order == selected,
        f"{context}: attempt outcomes do not close to selected/rejected order",
    )
    capacity_rejections = record.get("capacity_rejections")
    _require(
        isinstance(capacity_rejections, list)
        and len(capacity_rejections) == len(rejected),
        f"{context}: capacity rejection count mismatch",
    )
    rejection_by_npu = {
        _integer(item.get("npu_id"), f"{context}.capacity_rejections.npu_id"): item
        for item in capacity_rejections
        if isinstance(item, dict)
    }
    _require(
        set(rejection_by_npu) == set(rejected)
        and all(
            _canonical(rejection_by_npu[npu])
            == _canonical(last_attempt_by_npu[npu]["source"])
            for npu in rejected
        ),
        f"{context}: capacity rejections != final rejected attempts",
    )


def _adaptive_grants_at_time(
    summary: Mapping[str, object], time_ms: float, num_ssu: int
) -> list[list[float]]:
    installed = [[0.0] * num_ssu for _ in range(NUM_NPU)]
    records = summary.get("adaptive_decision_diagnostics")
    _require(isinstance(records, list), "Adaptive decision diagnostics missing")
    for index, record in enumerate(records):
        _require(isinstance(record, dict), "Adaptive decision record malformed")
        snapshot_time = _number(
            record.get("snapshot_time_ms"),
            f"adaptive_decision[{index}].snapshot_time_ms",
        )
        if snapshot_time > time_ms:
            break
        target = _matrix(
            record.get("grants_gbps_by_npu_ssu"),
            NUM_NPU,
            num_ssu,
            f"adaptive_decision[{index}].grants",
        )
        assert target is not None
        installed = target
    return installed


def _validate_bounded_probe_count(
    count: int, truncated: object, limit: int, context: str
) -> None:
    """Validate the producer's exact capped-record-stream contract."""

    _require(
        type(truncated) is bool
        and 0 < count <= limit
        and truncated is (count == limit),
        f"{context}: bounded probe count/cap-reached flag mismatch",
    )


def _validate_summary(row: dict, source_path: Path) -> CaseData:
    case = str(row.get("case"))
    num_ssu = _integer(row.get("num_ssu"), f"{case}.num_ssu")
    case_label = f"SSU{num_ssu}/{case}"
    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{case_label}: steady_summary missing")
    _require(
        _integer(summary.get("schema_version"), f"{case_label}.schema_version")
        == 2
        and summary.get("mode") == "steady_state_full_load",
        f"{case_label}: steady summary producer schema/mode mismatch",
    )
    _require(
        _integer(
            summary.get("warmup_requests_per_npu"),
            f"{case_label}.warmup_requests_per_npu",
        )
        == 8
        and _close(
            _number(summary.get("settle_ms"), f"{case_label}.settle_ms"),
            500.0,
        )
        and _close(
            _number(summary.get("slo_alpha"), f"{case_label}.slo_alpha"),
            2.0,
        ),
        f"{case_label}: steady summary warmup/settle/SLO contract mismatch",
    )
    for field, expected in (
        ("num_npu", NUM_NPU),
        ("num_ssu", num_ssu),
        ("n_layers", N_LAYERS),
        ("batch_size", BATCH_SIZE),
    ):
        _require(_integer(summary.get(field), f"{case_label}.{field}") == expected, f"{case_label}: {field} mismatch")
    _require(_close(_number(summary.get("measurement_duration_ms"), f"{case_label}.duration"), MEASUREMENT_MS), f"{case_label}: measurement is not 64 seconds")
    start = _number(summary.get("measurement_start_ms"), f"{case_label}.start")
    end = _number(summary.get("measurement_end_ms"), f"{case_label}.end")
    _require(_close(end - start, MEASUREMENT_MS), f"{case_label}: endpoint duration mismatch")
    _require(summary.get("stationarity_boundary_semantics") == BOUNDARY_SEMANTICS, f"{case_label}: boundary semantics mismatch")
    _require(summary.get("measurement_control_counter_window") == CONTROL_WINDOW, f"{case_label}: control window semantics mismatch")
    _require(summary.get("timeline_diagnostics_enabled") is True, f"{case_label}: timeline diagnostics disabled")
    _require(summary.get("timeline_adaptive_deadline_input") is False, f"{case_label}: deadline-input declaration is not false")
    semantics = summary.get("timeline_demand_semantics")
    _require(isinstance(semantics, dict) and all(field in semantics for field in ("controller_demand", "physical_demand", "installed_cir", "realized_service", "ssd_served_awaiting_link_enqueue")), f"{case_label}: bandwidth semantics missing")
    _require(
        _close(
            _number(
                summary.get("timeline_dispatch_probe_ms"),
                f"{case_label}.timeline_dispatch_probe_ms",
            ),
            50.0,
        )
        and _integer(
            summary.get("timeline_dispatch_probe_limit"),
            f"{case_label}.timeline_dispatch_probe_limit",
        )
        == 10_000,
        f"{case_label}: bounded probe scope is not the formal 50 ms / 10000 record contract",
    )

    expected_knobs = {
        "baseline": (0.0, None),
        "layer_once_ttl_5ms": (5.0, None),
        "adaptive_t0_i100ms": (0.0, 100.0),
    }
    ttl, interval = expected_knobs[case]
    _require(_close(_number(summary.get("pressure_ttl_ms"), f"{case_label}.ttl"), ttl), f"{case_label}: pressure TTL mismatch")
    actual_interval = summary.get("control_min_interval_ms")
    if interval is None:
        _require(actual_interval is None, f"{case_label}: unexpected controller interval")
    else:
        _require(_close(_number(actual_interval, f"{case_label}.interval"), interval), f"{case_label}: controller interval mismatch")

    trigger_records = summary.get("timeline_control_trigger_records")
    _require(
        isinstance(trigger_records, list),
        f"{case_label}: control trigger records missing",
    )
    expected_case_spec = next(
        item for item in LEGACY32_CASE_SPECS if item["name"] == case
    )
    _require(
        row.get("case_spec") == expected_case_spec,
        f"{case_label}: case spec does not authenticate control scheduling",
    )
    if case == "adaptive_t0_i100ms":
        _require(
            expected_case_spec["kind"] == "adaptive"
            and _close(float(expected_case_spec["min_interval_ms"]), 100.0)
            and trigger_records,
            f"{case_label}: Adaptive control is not event-driven batch-boundary-only",
        )
    else:
        _require(
            expected_case_spec["kind"] != "adaptive" and not trigger_records,
            f"{case_label}: non-Adaptive policy exposes control scheduling",
        )

    invariants = summary.get("invariants")
    _require(isinstance(invariants, dict) and invariants, f"{case_label}: invariants missing")
    _require(REQUIRED_TIMELINE_INVARIANTS <= set(invariants), f"{case_label}: timeline invariants missing: {sorted(REQUIRED_TIMELINE_INVARIANTS - set(invariants))}")
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{case_label}: failed simulator invariants: {failed}")
    blocks = _validate_blocks(summary, case_label, num_ssu)
    requests = _validate_request_rows(summary, case_label)
    boundaries = _validate_timeline(summary, blocks, case, num_ssu, case_label)
    _validate_ssd_accounting_summary(summary, boundaries, case_label, num_ssu)
    _validate_state_durations(summary, case_label)
    _validate_block_state_durations(summary, blocks, case_label)
    _validate_state_durations_from_lifecycles(
        summary, blocks, requests, case_label
    )

    probes = summary.get("timeline_dispatch_probe_records")
    _require(isinstance(probes, list) and probes, f"{case_label}: dispatch probe is empty")
    _validate_bounded_probe_count(
        len(probes),
        summary.get("timeline_dispatch_probe_truncated"),
        10_000,
        f"{case_label}.dispatch_probe",
    )
    dispatch_times = [
        _number(probe.get("time_ms"), f"{case_label}.dispatch_probe.time")
        for probe in probes
    ]
    _require(
        dispatch_times == sorted(dispatch_times),
        f"{case_label}: dispatch probe times are not nondecreasing",
    )
    for index, probe in enumerate(probes):
        context = f"{case_label}.dispatch_probe[{index}]"
        _require(isinstance(probe, dict), f"{context}: expected object")
        probe_time = _number(probe.get("time_ms"), f"{context}.time_ms")
        _require(
            start <= probe_time < start + 50.0,
            f"{context}: dispatch probe lies outside the formal first-50-ms window",
        )
        ssu_id = _integer(probe.get("ssu_id"), f"{context}.ssu_id")
        _require(0 <= ssu_id < num_ssu, f"{context}: SSU ID out of range")
        rr_cursor = _integer(
            probe.get("rr_cursor_before"), f"{context}.rr_cursor_before"
        )
        candidate_count = _integer(
            probe.get("candidate_path_count"), f"{context}.candidate_count"
        )
        tie_count = _integer(
            probe.get("finish_tie_count"), f"{context}.finish_tie_count"
        )
        _require(
            0 <= rr_cursor < EXPECTED_PATH_ABI["path_count"]
            and 1 <= tie_count <= candidate_count <= EXPECTED_PATH_ABI["path_count"],
            f"{context}: invalid RR/candidate/tie cardinality",
        )
        minimum_finish = _number(
            probe.get("minimum_finish_tag"), f"{context}.minimum_finish_tag"
        )
        winner_finish = _number(
            probe.get("winner_finish_tag_before"),
            f"{context}.winner_finish_tag_before",
        )
        virtual_after = _number(
            probe.get("winner_virtual_finish_after"),
            f"{context}.winner_virtual_finish_after",
        )
        _require(
            minimum_finish - 1e-12 <= winner_finish <= minimum_finish + 1e-12
            and abs(winner_finish - virtual_after) <= 1e-12,
            f"{context}: winning finish tag/tie/runtime virtual-finish mismatch",
        )
        expected_path = _integer(
            probe.get("expected_path_id"), f"{context}.expected_path"
        )
        actual_path = _integer(
            probe.get("actual_path_id"), f"{context}.actual_path"
        )
        _require(
            0 <= expected_path < EXPECTED_PATH_ABI["path_count"],
            f"{context}: winning Path ID out of range",
        )
        _require(probe.get("prediction_matches_actual") is True, f"{context}: replay mismatch")
        _require(expected_path == actual_path, f"{context}: expected/actual path mismatch")
        _require(probe.get("selection_rule") == "minimum_virtual_finish_then_round_robin", f"{context}: unknown dispatch rule")
        winner_npu = _integer(
            probe.get("winner_npu_id"), f"{context}.winner_npu_id"
        )
        winner_request = _integer(
            probe.get("winner_request_id"), f"{context}.winner_request_id"
        )
        authenticated_winner = _current_materialized_request_metadata(
            num_ssu
        ).get(winner_request)
        _require(
            0 <= winner_npu < NUM_NPU
            and authenticated_winner is not None
            and authenticated_winner["npu_id"] == winner_npu,
            f"{context}: dispatch winner request is not authenticated to winner NPU",
        )
        winner_group = _integer(
            probe.get("winner_group_id"), f"{context}.winner_group_id"
        )
        _require(
            winner_group
            == expected_path // EXPECTED_PATH_ABI["paths_per_group"],
            f"{context}: winning Path/group mapping mismatch",
        )
        if case == "baseline":
            _require(expected_path == 0, f"{context}: baseline winner is not Path 0")
        elif case == "adaptive_t0_i100ms":
            _require(
                expected_path == cold_start_hybrid_path_id(winner_npu),
                f"{context}: Adaptive winner is not the exact NPU-dedicated Path",
            )
        rate = _number(
            probe.get("winner_estimated_arbitration_rate_gbps"),
            f"{context}.winner_estimated_rate",
        )
        cir = _number(probe.get("winner_cir_gbps"), f"{context}.winner_cir")
        weight = _number(
            probe.get("winner_path_weight"), f"{context}.winner_path_weight"
        )
        if case == "adaptive_t0_i100ms":
            expected_grants = _adaptive_grants_at_time(
                summary, probe_time, num_ssu
            )
            _require(
                _close(cir, expected_grants[winner_npu][ssu_id])
                and _close(weight, 1.0),
                f"{context}: winner CIR/weight differs from the latest Adaptive control state",
            )
        else:
            qos = static_qos_config()
            _require(
                _close(cir, qos.path_cirs[expected_path])
                and _close(weight, qos.path_weights[expected_path]),
                f"{context}: winner CIR/weight differs from formal static QoS registers",
            )
        pending_blocks = _integer(
            probe.get("winner_pending_blocks_before"),
            f"{context}.winner_pending_blocks",
        )
        pending_gb = _number(
            probe.get("winner_pending_gb_before"),
            f"{context}.winner_pending_gb",
        )
        command_gb = _number(
            probe.get("winner_command_gb"), f"{context}.winner_command_gb"
        )
        winner_layer = _integer(
            probe.get("winner_layer"), f"{context}.winner_layer"
        )
        winner_block_idx = _integer(
            probe.get("winner_block_idx"), f"{context}.winner_block_idx"
        )
        authenticated_block_sizes = {
            int(block_idx): float(size_gb)
            for block_idx, size_gb in authenticated_winner[
                "placement_blocks_by_ssu"
            ][ssu_id]
        }
        _require(
            0.0 < rate <= SSD_CAP_GBPS + TOL
            and cir >= -TOL
            and weight > 0.0
            and pending_blocks >= 1
            and pending_gb + TOL >= command_gb > 0.0
            and _number(
                probe.get("winner_queue_wait_ms"), f"{context}.queue_wait_ms"
            )
            >= -TOL
            and 0 <= winner_layer < N_LAYERS
            and winner_block_idx in authenticated_block_sizes
            and _close(
                command_gb, authenticated_block_sizes[winner_block_idx]
            ),
            f"{context}: invalid winner rate/queue/work/layer state",
        )
        _require(
            _close(
                _number(
                    probe.get("physical_command_service_gbps"),
                    f"{context}.physical_command_service_gbps",
                ),
                SSD_CAP_GBPS,
            )
            and probe.get("physical_command_non_preemptive") is True,
            f"{context}: runtime physical command rate/non-preemption mismatch",
        )

    route_probes = summary.get("timeline_route_probe_records")
    _require(isinstance(route_probes, list) and route_probes, f"{case_label}: route probe is empty")
    _validate_bounded_probe_count(
        len(route_probes),
        summary.get("timeline_route_probe_truncated"),
        10_000,
        f"{case_label}.route_probe",
    )
    route_times = [
        _number(probe.get("time_ms"), f"{case_label}.route_probe.time")
        for probe in route_probes
    ]
    _require(
        route_times == sorted(route_times),
        f"{case_label}: route probe times are not nondecreasing",
    )
    expected_route_rule = {
        "baseline": "fixed_path_zero",
        "layer_once_ttl_5ms": "pressure_aware_once_per_layer",
        "adaptive_t0_i100ms": "npu_dedicated_path",
    }[case]
    route_probe_ms = _number(
        summary.get("timeline_dispatch_probe_ms"),
        f"{case_label}.timeline_dispatch_probe_ms",
    )
    _require(
        type(summary.get("timeline_route_probe_truncated")) is bool,
        f"{case_label}: route-probe truncation declaration missing",
    )
    seen_route_plans: set[tuple[int, int, int]] = set()
    for index, probe in enumerate(route_probes):
        context = f"{case_label}.route_probe[{index}]"
        _require(isinstance(probe, dict), f"{context}: expected object")
        _require(probe.get("rule") == expected_route_rule, f"{context}: unexpected route rule")
        probe_time = _number(probe.get("time_ms"), f"{context}.time_ms")
        _require(
            start <= probe_time < min(end, start + route_probe_ms),
            f"{context}: probe outside declared bounded window",
        )
        npu_id = _integer(probe.get("npu_id"), f"{context}.npu_id")
        ssu_id = _integer(probe.get("ssu_id"), f"{context}.ssu_id")
        _require(0 <= npu_id < NUM_NPU, f"{context}: NPU ID out of range")
        route_request_id = _integer(
            probe.get("request_id"), f"{context}.request_id"
        )
        authenticated_route_request = _current_materialized_request_metadata(
            num_ssu
        ).get(route_request_id)
        _require(
            authenticated_route_request is not None
            and authenticated_route_request["npu_id"] == npu_id,
            f"{context}: route request is not authenticated to its NPU",
        )
        route_category = probe.get("category")
        _require(
            route_category == authenticated_route_request["category"],
            f"{context}: route category differs from authenticated request",
        )
        _require(0 <= ssu_id < num_ssu, f"{context}: SSU ID out of range")
        layer = _integer(probe.get("layer"), f"{context}.layer")
        _require(0 <= layer < N_LAYERS, f"{context}: layer out of range")
        route_identity = (route_request_id, layer, ssu_id)
        _require(
            route_identity not in seen_route_plans,
            f"{context}: duplicate request/layer/SSU route plan",
        )
        seen_route_plans.add(route_identity)
        selected_paths = probe.get("selected_path_ids")
        selected_groups = probe.get("selected_group_ids")
        block_indices = probe.get("block_indices")
        block_sizes = probe.get("block_sizes_gb")
        _require(
            isinstance(selected_paths, list)
            and isinstance(selected_groups, list)
            and isinstance(block_indices, list)
            and isinstance(block_sizes, list)
            and len(selected_paths)
            == len(selected_groups)
            == len(block_indices)
            == len(block_sizes)
            and len(selected_paths) > 0,
            f"{context}: route-plan vector shape mismatch",
        )
        path_count = _integer(probe.get("path_count"), f"{context}.path_count")
        paths_per_group = _integer(
            probe.get("paths_per_group"), f"{context}.paths_per_group"
        )
        group_weights = _vector(
            probe.get("group_weights"),
            path_count // paths_per_group if paths_per_group > 0 else 0,
            f"{context}.group_weights",
        )
        _require(
            path_count == EXPECTED_PATH_ABI["path_count"]
            and paths_per_group == EXPECTED_PATH_ABI["paths_per_group"]
            and len(group_weights) == EXPECTED_PATH_ABI["group_count"]
            and _vectors_close(
                group_weights, static_qos_config().group_weights
            ),
            f"{context}: route hardware shape/registers differ from formal ABI",
        )
        selected_path_values = [
            _integer(value, f"{context}.selected_path_ids")
            for value in selected_paths
        ]
        selected_group_values = [
            _integer(value, f"{context}.selected_group_ids")
            for value in selected_groups
        ]
        _require(
            all(0 <= value < path_count for value in selected_path_values),
            f"{context}: selected path out of range",
        )
        _require(
            selected_group_values
            == [value // paths_per_group for value in selected_path_values],
            f"{context}: selected path/group mapping mismatch",
        )
        for value in block_indices:
            _require(
                _integer(value, f"{context}.block_indices") >= 0,
                f"{context}: negative block index",
            )
        _require(
            all(
                _number(value, f"{context}.block_sizes_gb") > 0.0
                for value in block_sizes
            ),
            f"{context}: non-positive route block size",
        )
        allowed = probe.get("allowed_path_ids")
        _require(
            isinstance(allowed, list) and allowed,
            f"{context}: allowed paths missing",
        )
        allowed_values = [
            _integer(value, f"{context}.allowed_path_ids") for value in allowed
        ]
        _require(
            len(set(allowed_values)) == len(allowed_values)
            and all(0 <= value < path_count for value in allowed_values),
            f"{context}: invalid or duplicate allowed paths",
        )
        static_allowed = list(
            category_path_ids(
                str(route_category), hardware_view(static_qos_config())
            )
        )
        if case == "baseline":
            _require(
                allowed_values == static_allowed
                and all(value == 0 for value in selected_path_values),
                f"{context}: baseline route inputs differ from authenticated static pool or selected Path 0",
            )
        elif case == "adaptive_t0_i100ms":
            dedicated = cold_start_hybrid_path_id(npu_id)
            _require(
                allowed_values == [dedicated]
                and all(value == dedicated for value in selected_path_values),
                f"{context}: Adaptive route is not the exact dedicated NPU Path",
            )
        else:
            _require(
                allowed_values == static_allowed,
                f"{context}: layer-once allowed Path pool differs from authenticated category pool",
            )
        expected_blocks = authenticated_route_request[
            "placement_blocks_by_ssu"
        ][ssu_id]
        _require(
            [int(value) for value in block_indices]
            == [int(block_index) for block_index, _size in expected_blocks]
            and _vectors_close(
                [float(value) for value in block_sizes],
                [float(size) for _block_index, size in expected_blocks],
            )
            and expected_blocks,
            f"{context}: routed block indices/sizes differ from authenticated placement",
        )
        _require(
            _integer(probe.get("start_offset"), f"{context}.start_offset")
            == (route_request_id + 13 * layer + 29 * ssu_id)
            % len(allowed_values),
            f"{context}: route start offset differs from the authenticated formula",
        )
        for field in (
            "allowed_path_cir_gbps",
            "allowed_path_pir_gbps_or_null",
            "allowed_path_weights",
            "allowed_path_group_ids",
        ):
            value = probe.get(field)
            _require(
                isinstance(value, list) and len(value) == len(allowed_values),
                f"{context}: {field} shape mismatch",
            )
        allowed_cirs = _vector(
            probe.get("allowed_path_cir_gbps"),
            len(allowed_values),
            f"{context}.allowed_path_cir_gbps",
        )
        allowed_weights = _vector(
            probe.get("allowed_path_weights"),
            len(allowed_values),
            f"{context}.allowed_path_weights",
        )
        allowed_group_values = _vector(
            probe.get("allowed_path_group_ids"),
            len(allowed_values),
            f"{context}.allowed_path_group_ids",
            integer=True,
        )
        _require(
            allowed_group_values
            == [value // paths_per_group for value in allowed_values],
            f"{context}: allowed path/group mapping mismatch",
        )
        pir_values = probe.get("allowed_path_pir_gbps_or_null")
        assert isinstance(pir_values, list)
        for value in pir_values:
            if value is not None:
                _require(
                    _number(value, f"{context}.allowed_path_pir") >= 0.0,
                    f"{context}: negative PIR",
                )
        _require(
            min(allowed_cirs + allowed_weights) >= -TOL,
            f"{context}: negative allowed-path register",
        )
        if case != "adaptive_t0_i100ms":
            formal_qos = static_qos_config()
            _require(
                _vectors_close(
                    allowed_cirs,
                    [formal_qos.path_cirs[path_id] for path_id in allowed_values],
                )
                and _vectors_close(
                    allowed_weights,
                    [
                        formal_qos.path_weights[path_id]
                        for path_id in allowed_values
                    ],
                )
                and all(value is None for value in pir_values),
                f"{context}: route Path registers differ from formal static QoS config",
            )
        else:
            latest_grants = _adaptive_grants_at_time(
                summary, probe_time, num_ssu
            )
            _require(
                _vectors_close(
                    allowed_cirs,
                    [latest_grants[npu_id][ssu_id]],
                )
                and _vectors_close(allowed_weights, [1.0] * len(allowed_values))
                and all(value is None for value in pir_values),
                f"{context}: Adaptive route CIR/immutable Path registers differ from latest control state",
            )
        _require(
            _close(
                _number(
                    probe.get("pressure_ttl_ms"),
                    f"{context}.pressure_ttl_ms",
                ),
                _number(
                    summary.get("pressure_ttl_ms"),
                    f"{case_label}.pressure_ttl_ms",
                ),
            ),
            f"{context}: route-probe TTL != case TTL",
        )

        if case == "layer_once_ttl_5ms":
            source = probe.get("pressure_source")
            _require(source in ("fresh", "cache"), f"{context}: layer-once pressure source missing")
            snapshot_time = _number(
                probe.get("pressure_snapshot_time_ms"),
                f"{context}.pressure_snapshot_time_ms",
            )
            pressure_age = _number(
                probe.get("pressure_age_ms"), f"{context}.pressure_age_ms"
            )
            pressure_ttl = _number(
                probe.get("pressure_ttl_ms"), f"{context}.pressure_ttl_ms"
            )
            _require(
                _close(probe_time - snapshot_time, pressure_age),
                f"{context}: pressure age != probe time - snapshot time",
            )
            _require(
                pressure_age >= 0.0
                and probe_time < snapshot_time + pressure_ttl,
                f"{context}: pressure snapshot is outside its TTL",
            )
            if source == "fresh":
                _require(
                    _close(pressure_age, 0.0),
                    f"{context}: fresh pressure snapshot has nonzero age",
                )
            counts = _vector(
                probe.get("allowed_path_pressure_counts"),
                len(allowed_values),
                f"{context}.allowed_path_pressure_counts",
                integer=True,
            )
            group_io = _vector(
                probe.get("group_io_counts"),
                len(group_weights),
                f"{context}.group_io_counts",
                integer=True,
            )
            active_paths = _vector(
                probe.get("active_paths_per_group"),
                len(group_weights),
                f"{context}.active_paths_per_group",
                integer=True,
            )
            active_weights = _vector(
                probe.get("active_path_weights"),
                len(group_weights),
                f"{context}.active_path_weights",
            )
            _require(
                min(counts + group_io + active_paths + active_weights) >= -TOL,
                f"{context}: negative pressure aggregate",
            )
            for group_id in range(len(group_weights)):
                allowed_members = [
                    index
                    for index, value in enumerate(allowed_group_values)
                    if value == group_id
                ]
                _require(
                    group_io[group_id]
                    >= sum(int(counts[index]) for index in allowed_members),
                    f"{context}: group IO count below allowed-path subtotal",
                )
                active_allowed = [
                    index for index in allowed_members if counts[index] > 0
                ]
                _require(
                    active_paths[group_id] >= len(active_allowed),
                    f"{context}: active-path count below allowed-path subtotal",
                )
                _require(
                    active_weights[group_id] + TOL
                    >= sum(allowed_weights[index] for index in active_allowed),
                    f"{context}: active path weight below allowed subtotal",
                )
            expected_active_group_weight = sum(
                group_weights[group_id]
                for group_id, count in enumerate(active_paths)
                if count > 0
            )
            _require(
                _close(
                    _number(
                        probe.get("active_group_weight_sum"),
                        f"{context}.active_group_weight_sum",
                    ),
                    expected_active_group_weight,
                ),
                f"{context}: active group-weight sum mismatch",
            )
            active_cir_sum = _number(
                probe.get("active_cir_sum_gbps"),
                f"{context}.active_cir_sum_gbps",
            )
            allowed_active_cir = 0.0
            for item, count in enumerate(counts):
                if count <= 0:
                    continue
                pir = pir_values[item]
                allowed_active_cir += min(
                    allowed_cirs[item],
                    math.inf
                    if pir is None
                    else _number(pir, f"{context}.allowed_path_pir"),
                )
            _require(
                active_cir_sum + TOL >= allowed_active_cir,
                f"{context}: active CIR below allowed-path subtotal",
            )
        else:
            _require(probe.get("pressure_source") is None, f"{context}: non-pressure route has pressure source")
            for field in (
                "pressure_snapshot_time_ms",
                "pressure_age_ms",
                "allowed_path_pressure_counts",
                "group_io_counts",
                "active_paths_per_group",
                "active_path_weights",
                "active_group_weight_sum",
                "active_cir_sum_gbps",
            ):
                _require(
                    probe.get(field) is None,
                    f"{context}: non-pressure route unexpectedly has {field}",
                )
        try:
            replayed_paths = _replay_route_probe(probe, case, context)
        except (TypeError, ValueError, IndexError, ZeroDivisionError) as error:
            raise AnalysisError(f"{context}: portable route replay failed: {error}") from error
        _require(
            replayed_paths == tuple(selected_path_values),
            f"{context}: portable pressure-aware path replay mismatch",
        )

    if case == "adaptive_t0_i100ms":
        profile = summary.get("adaptive_controller_profile")
        expected_profile = {
            "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
            "explicit_spill_threshold": 0.75,
            "target_ratio": 0.52,
            "required_ratio": 0.5,
            "background_reserve_fraction": 0.05,
        }
        _require(
            profile == expected_profile,
            f"{case_label}: Adaptive controller profile mismatch",
        )
        records = summary.get("adaptive_decision_diagnostics")
        _require(summary.get("adaptive_decision_diagnostic_schema") == "adaptive_admission_decision_v1", f"{case_label}: decision schema mismatch")
        _require(isinstance(records, list) and records, f"{case_label}: Adaptive decisions missing")
        _require(len(records) == _integer(summary.get("control_evaluations"), f"{case_label}.control_evaluations"), f"{case_label}: full decision diagnostic count mismatch")
        trigger_reasons_by_effective_time: dict[float, set[str]] = defaultdict(set)
        seen_effective_times: set[float] = set()
        prior_raw_time: float | None = None
        for trigger_index, trigger in enumerate(trigger_records):
            trigger_context = f"{case_label}.control_trigger[{trigger_index}]"
            _require(isinstance(trigger, dict), f"{trigger_context}: expected object")
            raw_time = _number(trigger.get("raw_time_ms"), f"{trigger_context}.raw_time_ms")
            effective_time = _number(
                trigger.get("effective_time_ms"),
                f"{trigger_context}.effective_time_ms",
            )
            reason = trigger.get("reason")
            rate_limited = trigger.get("rate_limited")
            coalesced = trigger.get("coalesced")
            _require(
                reason in {"initial", "batch_boundary"}
                and type(rate_limited) is bool
                and type(coalesced) is bool
                and raw_time <= effective_time
                and _close(
                    _number(
                        trigger.get("min_interval_ms"),
                        f"{trigger_context}.min_interval_ms",
                    ),
                    100.0,
                ),
                f"{trigger_context}: invalid/non-causal/wall-clock trigger record",
            )
            _require(
                rate_limited is (effective_time > raw_time + 1e-12)
                and coalesced is (effective_time in seen_effective_times),
                f"{trigger_context}: rate-limit/coalescing flags do not recompute",
            )
            if prior_raw_time is not None:
                _require(
                    raw_time >= prior_raw_time,
                    f"{trigger_context}: trigger raw times are not nondecreasing",
                )
            prior_raw_time = raw_time
            seen_effective_times.add(effective_time)
            trigger_reasons_by_effective_time[effective_time].add(str(reason))
        prior_snapshot_time: float | None = None
        expected_previous_selected_pairs: list[list[int]] = []
        expected_previous_selected_order: list[int] = []
        for index, record in enumerate(records, start=1):
            _require(_integer(record.get("snapshot_evaluation"), f"{case_label}.decision[{index}]") == index, f"{case_label}: decision evaluations are not contiguous")
            snapshot_time = _number(
                record.get("snapshot_time_ms"),
                f"{case_label}.decision[{index}].snapshot_time_ms",
            )
            if prior_snapshot_time is not None:
                _require(
                    snapshot_time > prior_snapshot_time
                    and snapshot_time - prior_snapshot_time >= 100.0 - TOL,
                    f"{case_label}.decision[{index}]: snapshots violate the 100-ms minimum interval",
                )
            prior_snapshot_time = snapshot_time
            _require(
                set(record.get("trigger_reasons", []))
                == trigger_reasons_by_effective_time.get(snapshot_time, set()),
                f"{case_label}.decision[{index}]: decision trigger reasons/time do not match executed event-driven triggers",
            )
            _require(
                record.get("previous_selected_request_by_npu")
                == expected_previous_selected_pairs,
                f"{case_label}.decision[{index}]: previous-selected request map is not the preceding decision's selected map",
            )
            _validate_admission_diagnostic(
                record,
                num_ssu,
                f"{case_label}.decision[{index}]",
                previous_selected_order=expected_previous_selected_order,
            )
            request_map = {
                int(npu): int(request_id)
                for npu, request_id in record.get("request_by_npu", [])
            }
            expected_previous_selected_pairs = [
                [npu_id, request_map[npu_id]]
                for npu_id in sorted(record.get("selected_npu_ids", []))
            ]
            expected_previous_selected_order = [
                int(npu_id) for npu_id in record.get("selected_npu_ids", [])
            ]
            _require(
                _decision_prefetch_map(
                    record,
                    request_map,
                    f"{case_label}.decision[{index}]",
                )
                is not None,
                f"{case_label}.decision[{index}]: explicit prefetch classification missing",
            )
        decision_times = {float(record["snapshot_time_ms"]) for record in records}
        last_decision_time = max(decision_times)
        _require(
            decision_times <= seen_effective_times
            and all(
                trigger_time > last_decision_time
                for trigger_time in seen_effective_times - decision_times
            ),
            f"{case_label}: executed effective trigger times do not exactly cover decisions before drain",
        )
        measurement_records = [
            record
            for record in records
            if start <= _number(record.get("snapshot_time_ms"), f"{case_label}.decision_time") < end
        ]
        _require(len(measurement_records) == _integer(summary.get("measurement_control_evaluations"), f"{case_label}.measurement_control_evaluations"), f"{case_label}: measurement decision count mismatch")
        _require(
            len(measurement_records) <= math.ceil(MEASUREMENT_MS / 100.0) + 1,
            f"{case_label}: too many controller evaluations for a 100-ms minimum interval",
        )
    else:
        _require(not summary.get("adaptive_decision_diagnostics"), f"{case_label}: non-Adaptive strategy has Adaptive decisions")

    return CaseData(
        case=case,
        num_ssu=num_ssu,
        source_path=source_path,
        payload={},
        row=row,
        summary=summary,
        boundaries=boundaries,
        blocks=blocks,
        requests=requests,
    )


def _load_and_validate(inputs: Sequence[Path]) -> tuple[dict[tuple[int, str], CaseData], dict]:
    paths = _discover_payload_paths(inputs)
    selected_rows: dict[tuple[int, str], tuple[Path, dict, dict]] = {}
    used_payloads: dict[Path, dict] = {}
    source_fingerprints: set[str] = set()
    config_fingerprints: set[str] = set()
    definition_fingerprints: set[str] = set()
    campaign_fingerprints: set[object] = set()
    specs: set[str] = set()
    manifests: set[str] = set()
    schedules: set[str] = set()
    runtime_merge_identities: set[str] = set()
    full_runtime_signatures: set[str] = set()
    payload_ssu_by_path: dict[Path, int] = {}

    for path in paths:
        payload = _load_json(path)
        results = payload.get("results")
        if not isinstance(results, list) or not isinstance(payload.get("experiment_spec"), dict):
            continue
        targeted = [
            row
            for row in results
            if isinstance(row, dict)
            and row.get("case") in CASES
            and row.get("num_ssu") in SSU_COUNTS
        ]
        if not targeted:
            continue
        label = _publication_path(path)
        _require(
            payload.get("schema_version") == 3
            and payload.get("definition") == "legacy32"
            and payload.get("num_npu") == NUM_NPU
            and payload.get("backing_requests_per_npu") == 256
            and payload.get("total_assignment_count") == NUM_NPU * 256,
            f"{label}: top-level runner schema/definition mismatch",
        )
        _require(
            payload.get("path_abi") == EXPECTED_PATH_ABI,
            f"{label}: top-level dedicated Path ABI mismatch",
        )
        _require(
            len(results) == len(CASES) and all(isinstance(row, dict) for row in results),
            f"{label}: selected shard must contain exactly three result rows and no irrelevant row",
        )
        shard_keys: list[tuple[str, int]] = []
        for index, row in enumerate(results):
            case = row.get("case")
            _require(case in CASES, f"{label}: results[{index}] is an irrelevant case")
            num_ssu = _integer(row.get("num_ssu"), f"{label}.results[{index}].num_ssu")
            _require(num_ssu in SSU_COUNTS, f"{label}: results[{index}] has an unexpected SSU")
            _require(
                isinstance(row.get("steady_summary"), dict)
                and row["steady_summary"].get("timeline_diagnostics_enabled") is True,
                f"{label}: results[{index}] lacks v3 timeline diagnostics",
            )
            shard_keys.append((str(case), num_ssu))
        _require(
            len(set(shard_keys)) == len(shard_keys),
            f"{label}: duplicate result key in selected shard",
        )
        shard_ssus = {num_ssu for _case, num_ssu in shard_keys}
        _require(
            len(shard_ssus) == 1,
            f"{label}: a selected shard must contain exactly one SSU",
        )
        shard_ssu = next(iter(shard_ssus))
        expected_shard_keys = {(case, shard_ssu) for case in CASES}
        _require(
            set(shard_keys) == expected_shard_keys,
            f"{label}: selected shard is not the atomic three-strategy SSU{shard_ssu} matrix",
        )
        _require(
            payload.get("selected_ssus") == [shard_ssu],
            f"{label}: selected_ssus is not exactly [{shard_ssu}]",
        )
        selected_cases = payload.get("selected_cases")
        _require(
            isinstance(selected_cases, list)
            and len(selected_cases) == len(CASES)
            and set(selected_cases) == set(CASES),
            f"{label}: selected_cases is not exactly the three audited strategies",
        )
        selected_keys = payload.get("selected_keys")
        _require(isinstance(selected_keys, list), f"{label}: selected_keys missing")
        normalized_selected_keys = {
            (str(key[0]), _integer(key[1], f"{label}.selected_keys"))
            for key in selected_keys
            if isinstance(key, list) and len(key) == 2
        }
        _require(
            len(selected_keys) == len(expected_shard_keys)
            and len(normalized_selected_keys) == len(selected_keys)
            and normalized_selected_keys == expected_shard_keys,
            f"{label}: selected_keys is not exactly the atomic three-strategy matrix",
        )
        _require(payload.get("selected_complete") is True, f"{label}: selected shard is incomplete")
        _require(payload.get("source_stable_during_run") is True, f"{label}: source changed during run")
        _require(payload.get("config_stable_during_run") is True, f"{label}: config changed during run")
        _require(payload.get("campaign_spec_stable_during_run") is True, f"{label}: campaign spec changed during run")
        _require(
            payload.get("campaign_spec_sha256") is None
            and payload.get("campaign_spec_authentication") is None
            and payload.get("ending_campaign_spec_authentication") is None,
            f"{label}: formal run must not contain a campaign-spec override",
        )
        source = payload.get("source_fingerprint")
        config = payload.get("config_fingerprint")
        definition = payload.get("definition_fingerprint")
        _require(_is_lower_hex_sha256(source), f"{label}: invalid source fingerprint")
        _require(_is_lower_hex_sha256(config), f"{label}: invalid config fingerprint")
        _require(_is_lower_hex_sha256(definition), f"{label}: invalid definition fingerprint")
        _require(payload.get("ending_source_fingerprint") == source, f"{label}: ending source fingerprint differs")
        _require(payload.get("ending_config_fingerprint") == config, f"{label}: ending config fingerprint differs")
        _require(payload.get("ending_campaign_spec_authentication") == payload.get("campaign_spec_authentication"), f"{label}: ending campaign authentication differs")
        source_manifest = payload.get("source_manifest")
        _require(
            isinstance(source_manifest, dict) and source_manifest,
            f"{label}: source manifest missing",
        )
        _require(
            all(
                isinstance(name, str)
                and name
                and _is_lower_hex_sha256(digest)
                for name, digest in source_manifest.items()
            ),
            f"{label}: source manifest contains malformed entries",
        )
        _validate_source_manifest_against_checkout(source_manifest, label)
        _require(
            source
            == _runner_canonical_hash(
                source_manifest, SOURCE_FINGERPRINT_NAMESPACE
            ),
            f"{label}: source fingerprint is not derived from source_manifest",
        )
        spec = payload["experiment_spec"]
        _validate_experiment_spec(spec, label)
        input_authentication = payload.get("input_authentication")
        _require(
            input_authentication == _current_scientific_input_authentication()
            and spec.get("workload", {}).get("authentication")
            == input_authentication,
            f"{label}: top/spec scientific input authentication mismatch",
        )
        _require(
            input_authentication.get("source") == "data"
            and input_authentication.get("source_sha256")
            == source_manifest.get("data"),
            f"{label}: authenticated data bytes differ from source_manifest",
        )
        _require(
            config == _runner_canonical_hash(spec, CONFIG_FINGERPRINT_NAMESPACE),
            f"{label}: config fingerprint is not derived from experiment_spec",
        )
        expected_definition = _legacy_definition_fingerprint()
        _require(
            definition == expected_definition
            and spec.get("definition_fingerprint") == expected_definition,
            f"{label}: definition fingerprint is not derived from the legacy32 definition",
        )
        top_runtime_merge = _runtime_merge_identity(
            payload.get("runtime"), f"{label}.runtime"
        )
        runtime_merge_identities.add(_canonical(top_runtime_merge))
        full_runtime_signatures.add(
            _canonical(_runtime_signature(payload.get("runtime"), f"{label}.runtime"))
        )
        source_fingerprints.add(source)
        config_fingerprints.add(config)
        definition_fingerprints.add(definition)
        campaign_fingerprints.add(payload.get("campaign_spec_sha256"))
        specs.add(_canonical(spec))
        manifests.add(_canonical(source_manifest))
        schedules.add(_canonical(payload.get("schedule_metadata")))
        used_payloads[path] = payload
        payload_ssu_by_path[path] = shard_ssu
        spec_case_by_name = {
            str(case_spec["name"]): case_spec for case_spec in spec["cases"]
        }
        for row in results:
            key = (_integer(row["num_ssu"], f"{label}.num_ssu"), str(row["case"]))
            _require(row.get("status") == "ok", f"{label}: {key} status is not ok")
            _require(
                row.get("campaign_spec_sha256") is None,
                f"{label}: {key} unexpectedly uses a campaign-spec override",
            )
            _require(key not in selected_rows, f"duplicate selected row {key}")
            _require(
                row.get("definition") == "legacy32"
                and row.get("num_npu") == NUM_NPU
                and row.get("backing_requests_per_npu") == 256,
                f"{label}: {key} row definition/topology mismatch",
            )
            _require(row.get("source_fingerprint") == source, f"{label}: {key} row/source fingerprint mismatch")
            _require(row.get("config_fingerprint") == config, f"{label}: {key} row/config fingerprint mismatch")
            _require(row.get("definition_fingerprint") == expected_definition, f"{label}: {key} row/definition fingerprint mismatch")
            expected_case_spec = spec_case_by_name[str(row["case"])]
            _require(
                row.get("case_spec") == expected_case_spec,
                f"{label}: {key} row case spec differs from experiment_spec",
            )
            _require(
                row.get("case_fingerprint")
                == _case_fingerprint(
                    expected_case_spec,
                    key[0],
                    str(source),
                    str(config),
                ),
                f"{label}: {key} case fingerprint is not runner-derived",
            )
            row_runtime_merge = _runtime_merge_identity(
                row.get("runtime"), f"{label}.{key}.runtime"
            )
            runtime_merge_identities.add(_canonical(row_runtime_merge))
            full_runtime_signatures.add(
                _canonical(
                    _runtime_signature(row.get("runtime"), f"{label}.{key}.runtime")
                )
            )
            selected_rows[key] = (path, payload, row)

        audit = payload.get("pairing_audit")
        _require(isinstance(audit, dict), f"{label}: pairing audit missing")
        item = audit.get(str(shard_ssu))
        _require(isinstance(item, dict), f"{label}: SSU{shard_ssu} pairing audit missing")
        _require(
            item.get("has_rows") is True
            and item.get("all_available_rows_paired") is True
            and isinstance(item.get("cases"), list)
            and len(item["cases"]) == len(CASES)
            and set(item["cases"]) == set(CASES),
            f"{label}: SSU{shard_ssu} atomic pairing audit failed",
        )

    expected = {(num_ssu, case) for num_ssu in SSU_COUNTS for case in CASES}
    _require(set(selected_rows) == expected, f"incomplete 3 x 3 matrix: missing={sorted(expected - set(selected_rows))}, extra={sorted(set(selected_rows) - expected)}")
    _require(
        len(used_payloads) == len(SSU_COUNTS)
        and set(payload_ssu_by_path.values()) == set(SSU_COUNTS)
        and len(set(payload_ssu_by_path.values())) == len(payload_ssu_by_path),
        "campaign must use exactly three atomic one-SSU shards",
    )
    for num_ssu in SSU_COUNTS:
        _require(
            len({selected_rows[(num_ssu, case)][0] for case in CASES}) == 1,
            f"SSU{num_ssu}: three strategies do not come from the same payload shard",
        )
    _require(len(source_fingerprints) == 1, "shards use different source fingerprints")
    _require(len(config_fingerprints) == 1, "shards use different config fingerprints")
    _require(len(definition_fingerprints) == 1, "shards use different experiment definitions")
    _require(len(campaign_fingerprints) == 1, "shards use different campaign specs")
    _require(len(specs) == 1 and len(manifests) == 1 and len(schedules) == 1, "shard source/spec/schedule metadata differs")
    _require(
        len(runtime_merge_identities) == 1,
        "local/remote shards do not share the runner runtime merge identity",
    )
    _require(
        len(full_runtime_signatures) == 1,
        "three shards do not share the exact formal hostname/platform/thread/Python/NumPy/BLAS runtime signature",
    )

    cases: dict[tuple[int, str], CaseData] = {}
    runtime_hash_by_key: dict[tuple[int, str], str] = {}
    input_fingerprints_by_key: dict[tuple[int, str], dict] = {}
    for key in sorted(expected):
        path, payload, row = selected_rows[key]
        data = _validate_summary(row, path)
        data = CaseData(
            case=data.case,
            num_ssu=data.num_ssu,
            source_path=data.source_path,
            payload=payload,
            row=data.row,
            summary=data.summary,
            boundaries=data.boundaries,
            blocks=data.blocks,
            requests=data.requests,
        )
        inputs_for_row = row.get("input_fingerprints")
        _require(
            isinstance(inputs_for_row, dict)
            and set(inputs_for_row) == set(PAIR_FIELDS_WITHIN_SSU),
            f"{key}: input fingerprint field set is not exact",
        )
        for field in PAIR_FIELDS_WITHIN_SSU:
            value = inputs_for_row.get(field)
            _require(_is_lower_hex_sha256(value), f"{key}: invalid {field} fingerprint")
        authenticated_workload = payload["experiment_spec"]["workload"]
        for field in ("catalog", "recipe", "schedule", "assignment"):
            _require(
                inputs_for_row[field] == authenticated_workload[field],
                f"{key}: row {field} fingerprint differs from experiment_spec",
            )
        _require(
            inputs_for_row["prefix_32_assignment"]
            == authenticated_workload["prefix_32_assignment_hash"]
            and inputs_for_row["full_assignment"]
            == authenticated_workload["full_assignment_hash"],
            f"{key}: row prefix/full assignment is not config-authenticated",
        )
        expected_materialized = _current_materialized_input_fingerprints(key[0])
        for field, expected_value in expected_materialized.items():
            _require(
                inputs_for_row[field] == expected_value,
                f"{key}: row {field} differs from independent materialization",
            )
        _require(data.summary.get("input_fingerprint") == inputs_for_row.get("simulator"), f"{key}: simulator input fingerprint mismatch")
        runtime = _runtime_signature(row.get("runtime"), f"{key}.runtime")
        runtime_hash_by_key[key] = _canonical_sha256(runtime)
        input_fingerprints_by_key[key] = inputs_for_row
        cases[key] = data

    for num_ssu in SSU_COUNTS:
        group = [input_fingerprints_by_key[(num_ssu, case)] for case in CASES]
        for field in PAIR_FIELDS_WITHIN_SSU:
            _require(len({row[field] for row in group}) == 1, f"SSU{num_ssu}: paired strategies differ in {field}")
        _require(len({runtime_hash_by_key[(num_ssu, case)] for case in CASES}) == 1, f"SSU{num_ssu}: paired strategies use different scientific runtimes")
        request_identity = []
        for case in CASES:
            data = cases[(num_ssu, case)]
            request_identity.append(
                {
                    (int(row["npu_id"]), int(row["sequence"])): (
                        row.get("profile_id"),
                        float(row["ideal_ttft_ms"]),
                    )
                    for row in data.requests
                }
            )
        common = set.intersection(*(set(values) for values in request_identity))
        _require(common, f"SSU{num_ssu}: no common measured requests across strategies")
        for identity in common:
            _require(len({values[identity] for values in request_identity}) == 1, f"SSU{num_ssu}: common request profile/ideal compute mismatch")

    for field in CROSS_SSU_INPUT_FIELDS:
        # Materialized workload/placement/trace/simulator inputs legitimately
        # change with SSU count.  The catalog recipe and sampled assignments do
        # not, and therefore provide the strict cross-SSU prefix bridge.
        values = {
            input_fingerprints_by_key[(num_ssu, "baseline")][field]
            for num_ssu in SSU_COUNTS
        }
        _require(len(values) == 1, f"cross-SSU scientific input differs in {field}")

    invariant_counts = {
        f"ssu{num_ssu}/{case}": len(data.summary["invariants"])
        for (num_ssu, case), data in sorted(cases.items())
    }
    validation = {
        "passed": True,
        "analysis": "npu32_ssu345_timeline64_v1",
        "input_files": [
            {"path": _publication_path(path), "sha256": _sha256(path)}
            for path in sorted(used_payloads)
        ],
        "source_fingerprint": next(iter(source_fingerprints)),
        "config_fingerprint": next(iter(config_fingerprints)),
        "definition_fingerprint": next(iter(definition_fingerprints)),
        "campaign_spec_sha256": next(iter(campaign_fingerprints)),
        "runtime_merge_identity_sha256": hashlib.sha256(
            next(iter(runtime_merge_identities)).encode("utf-8")
        ).hexdigest(),
        "full_runtime_signature_sha256": hashlib.sha256(
            next(iter(full_runtime_signatures)).encode("utf-8")
        ).hexdigest(),
        "runtime_signature_sha256_by_ssu": {
            str(num_ssu): runtime_hash_by_key[(num_ssu, "baseline")]
            for num_ssu in SSU_COUNTS
        },
        "paired_input_fingerprints_by_ssu": {
            str(num_ssu): {
                field: input_fingerprints_by_key[(num_ssu, "baseline")][field]
                for field in PAIR_FIELDS_WITHIN_SSU
            }
            for num_ssu in SSU_COUNTS
        },
        "simulator_invariant_count_by_case": invariant_counts,
        "checks": {
            "complete_3_strategies_x_3_ssus": True,
            "exactly_3_atomic_one_ssu_shards": True,
            "each_ssu_three_strategies_from_same_payload": True,
            "no_irrelevant_or_duplicate_result_rows": True,
            "npu32_layers16_batch1": True,
            "measurement_64s": True,
            "steady_summary_schema_mode_warmup_settle_slo_exact": True,
            "blocks_128_x_500ms": True,
            "boundaries_129_left_limit": True,
            "source_config_campaign_stable_and_identical": True,
            "source_fingerprint_recomputed_from_manifest": True,
            "source_manifest_raw_bytes_match_upload_checkout": True,
            "config_fingerprint_recomputed_from_experiment_spec": True,
            "definition_and_case_fingerprints_recomputed": True,
            "runtime_merge_identity_identical_across_local_remote_shards": True,
            "full_hostname_platform_thread_python_numpy_blas_runtime_signature_identical": True,
            "runtime_identical_within_each_ssu_pair": True,
            "input_fingerprints_identical_within_each_ssu_pair": True,
            "workload_prefix_identical_across_ssus": True,
            "all_simulator_invariants_true": True,
            "exact_npu_ssu_ssd_service_reconstructed": True,
            "timeline_schema_v3_stable_service_and_direct_physical_outstanding": True,
            "v3_ssd_accounting_residuals_recomputed_with_declared_tolerance": True,
            "exact_npu_ssu_link_delivery_reconstructed": True,
            "actual_bandwidth_is_adjacent_boundary_counter_difference": True,
            "demand_is_left_boundary_state_and_is_not_differenced": True,
            "controller_manifest_inventory_authenticated_to_unique_1_to_16_layer_slice": True,
            "compute_q_activation_conservation_recomputed": True,
            "exact_full_window_compute_io_barrier_other_partition_including_carry_in": True,
            "state_duration_compute_matches_npu_utilization": True,
            "block_state_128x500ms_partition_and_whole_window_sum_exact": True,
            "simulator_predispatch_prediction_runtime_assertion_matches": True,
            "portable_route_plan_replay_matches": True,
            "route_pressure_fresh_cache_age_and_group_mapping_valid": True,
            "adaptive_deadline_is_diagnostic_only": True,
            "adaptive_event_driven_chain_bound_to_experiment_case_summary_triggers_and_decisions": True,
            "adaptive_decision_cir_and_write_counters_reconstructed": True,
            "probe_truncated_flag_means_record_cap_reached": True,
            "all_exported_ttft_slack_is_diagnostic_only": True,
            "all_public_artifacts_strictly_below_50_mib": True,
            "raw_runner_shards_not_copied_to_public_output": True,
        },
        "semantic_guards": {
            "installed_cir_is_actual_bandwidth": False,
            "npu_link_delivered_gbps_is_actual_received_bandwidth": True,
            "fragmented_settle_service_used_as_scientific_bandwidth": False,
            "ssd_outstanding_derived_from_fragmented_counter_subtraction": False,
            "adaptive_reads_ttft_deadline_or_slack": False,
            "adaptive_required_ratio": 0.5,
            "adaptive_required_ratio_configuration": "alpha=2 feasibility target",
        },
    }
    return cases, validation


def _matrix_delta(
    before: Sequence[Sequence[float]], after: Sequence[Sequence[float]]
) -> list[list[float]]:
    return [
        [float(right) - float(left) for left, right in zip(left_row, right_row)]
        for left_row, right_row in zip(before, after)
    ]


def _build_timeline_rows(cases: Mapping[tuple[int, str], CaseData]) -> list[dict]:
    rows: list[dict] = []
    for (num_ssu, case), data in sorted(cases.items()):
        for block_index in range(BLOCK_COUNT):
            before = data.boundaries[block_index]
            after = data.boundaries[block_index + 1]
            before_timeline = before["timeline"]
            before_cells = before_timeline["npu_ssu"]
            after_cells = after["timeline"]["npu_ssu"]
            duration_s = (float(after["time_ms"]) - float(before["time_ms"])) / 1000.0
            ssd_service = _matrix_delta(
                before_cells["ssd_served_cumulative_gb"],
                after_cells["ssd_served_cumulative_gb"],
            )
            link_delivery = _matrix_delta(
                before_cells["link_served_cumulative_gb"],
                after_cells["link_served_cumulative_gb"],
            )
            route_plans = _matrix_delta(
                before_cells["route_plans_cumulative"],
                after_cells["route_plans_cumulative"],
            )
            route_fresh = _matrix_delta(
                before_cells["route_pressure_fresh_cumulative"],
                after_cells["route_pressure_fresh_cumulative"],
            )
            route_cache = _matrix_delta(
                before_cells["route_pressure_cache_cumulative"],
                after_cells["route_pressure_cache_cumulative"],
            )
            route_groups = [
                [
                    [
                        int(
                            after_cells["route_blocks_by_group_cumulative"][npu_id][ssu_id][group_id]
                        )
                        - int(
                            before_cells["route_blocks_by_group_cumulative"][npu_id][ssu_id][group_id]
                        )
                        for group_id in range(EXPECTED_PATH_ABI["group_count"])
                    ]
                    for ssu_id in range(num_ssu)
                ]
                for npu_id in range(NUM_NPU)
            ]
            base_compute_before = before["npu_compute_cumulative_busy_ms_by_npu"]
            base_compute_after = after["npu_compute_cumulative_busy_ms_by_npu"]
            installed = before_cells["installed_dedicated_path_cir_gbps"]
            active_by_ssu = before_timeline.get("active_command_by_ssu", [None] * num_ssu)
            for npu_id in range(NUM_NPU):
                npu_row = before_timeline["npu_rows"][npu_id]
                npu_util = (
                    float(base_compute_after[npu_id])
                    - float(base_compute_before[npu_id])
                ) / BLOCK_MS
                deadline_slack15 = npu_row.get("slo_alpha1p5_slack_ms")
                deadline_slack2 = npu_row.get("slo_alpha2_slack_ms")
                active_request_id = npu_row.get("active_request_id")
                controller_request_id = npu_row.get("controller_request_id")
                controller_prefetch = npu_row.get("controller_prefetch_only")
                slack_eligible = (
                    active_request_id is not None
                    and controller_request_id is not None
                    and int(active_request_id) == int(controller_request_id)
                    and controller_prefetch is False
                )
                if not slack_eligible:
                    deadline_slack15 = None
                    deadline_slack2 = None
                q_ms = float(npu_row["compute_inventory_q_ms"])
                for ssu_id in range(num_ssu):
                    controller = float(before_cells["controller_demand_gbps"][npu_id][ssu_id])
                    physical = float(before_cells["physical_demand_gbps"][npu_id][ssu_id])
                    ssd_gbps = ssd_service[npu_id][ssu_id] / duration_s
                    link_gbps = link_delivery[npu_id][ssu_id] / duration_s
                    command = active_by_ssu[ssu_id]
                    group_counts = route_groups[npu_id][ssu_id]
                    group_total = sum(group_counts)
                    dominant_group = (
                        None
                        if group_total == 0
                        else min(
                            group_id
                            for group_id, count in enumerate(group_counts)
                            if count == max(group_counts)
                        )
                    )
                    rows.append(
                        {
                            "num_ssu": num_ssu,
                            "case": case,
                            "case_label": CASE_LABELS[case],
                            "block": block_index,
                            "relative_start_s": block_index * BLOCK_MS / 1000.0,
                            "relative_end_s": (block_index + 1) * BLOCK_MS / 1000.0,
                            "npu_id": npu_id,
                            "ssu_id": ssu_id,
                            "pipeline_state_at_left_boundary": npu_row["pipeline_state"],
                            "active_request_id": active_request_id,
                            "controller_request_id": controller_request_id,
                            "controller_prefetch_only": controller_prefetch,
                            "slack_eligible_admitted_nonprefetch": slack_eligible,
                            "prefetch_requests_excluded_from_slack": True,
                            "current_compute_layer": npu_row.get("current_compute_layer"),
                            "next_compute_layer": npu_row.get("next_compute_layer"),
                            "next_compute_layer_io_ready": npu_row.get("next_compute_layer_io_ready"),
                            "waiting_on_io_layer": npu_row.get("waiting_on_io_layer"),
                            "npu_utilization": npu_util,
                            "compute_inventory_q_ms": q_ms,
                            "deadline_slack_alpha1p5_ms_diagnostic_only": deadline_slack15,
                            "deadline_slack_alpha2_ms_diagnostic_only": deadline_slack2,
                            "slack_after_remaining_compute_alpha1p5_ms_diagnostic_only": (
                                None if deadline_slack15 is None else float(deadline_slack15) - q_ms
                            ),
                            "slack_after_remaining_compute_alpha2_ms_diagnostic_only": (
                                None if deadline_slack2 is None else float(deadline_slack2) - q_ms
                            ),
                            "manifest_demand_gbps_adaptive_input_otherwise_diagnostic": controller,
                            "manifest_demand_policy_role": (
                                "adaptive_input"
                                if case == "adaptive_t0_i100ms"
                                else "counterfactual_diagnostic"
                            ),
                            "physical_remaining_demand_gbps_diagnostic_only": physical,
                            "installed_cir_gbps_not_actual": (
                                None if installed is None else float(installed[npu_id][ssu_id])
                            ),
                            "interval_average_attributed_ssd_service_gbps": ssd_gbps,
                            "npu_link_delivered_gbps_actual_received": link_gbps,
                            "link_minus_controller_demand_gbps": link_gbps - controller,
                            "link_minus_physical_demand_gbps": link_gbps - physical,
                            "interval_average_ssd_service_minus_link_delivery_gbps": ssd_gbps - link_gbps,
                            "ssd_outstanding_gb_at_left_boundary": float(before_cells["ssd_outstanding_gb"][npu_id][ssu_id]),
                            "link_outstanding_gb_at_left_boundary": float(before_cells["link_outstanding_gb"][npu_id][ssu_id]),
                            "ssd_served_awaiting_nonstreaming_link_enqueue_gb_at_left_boundary": float(before_cells["ssd_served_awaiting_link_enqueue_gb"][npu_id][ssu_id]),
                            "client_unissued_gb_at_left_boundary": float(before_cells["client_unissued_gb"][npu_id][ssu_id]),
                            "route_plans_in_block": int(round(route_plans[npu_id][ssu_id])),
                            "fresh_pressure_reads_in_block": int(round(route_fresh[npu_id][ssu_id])),
                            "pressure_cache_hits_in_block": int(round(route_cache[npu_id][ssu_id])),
                            "route_blocks_by_group_in_block": _canonical(group_counts),
                            "route_block_count_in_block": group_total,
                            "dominant_route_group_in_block": dominant_group,
                            "dominant_route_group_share_in_block": (
                                None
                                if dominant_group is None
                                else group_counts[dominant_group] / group_total
                            ),
                            "left_boundary_ssd_command_owned_by_cell": bool(
                                isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                            ),
                            "left_boundary_ssd_command_request_id": (
                                command.get("request_id")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_physical_service_gbps": (
                                command.get("physical_service_gbps")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_path_id": (
                                command.get("path_id")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_layer": (
                                command.get("layer")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_block_idx": (
                                command.get("block_idx")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_remaining_gb": (
                                command.get("remaining_gb")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_age_ms": (
                                command.get("command_age_ms")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_start_time_ms": (
                                command.get("command_start_time_ms")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "left_boundary_inflight_ssd_command_non_preemptive": (
                                command.get("non_preemptive")
                                if isinstance(command, dict)
                                and int(command.get("npu_id", -1)) == npu_id
                                else None
                            ),
                            "allocation_mechanism_code": {
                                "baseline": "fixed_path0_nonpreemptive",
                                "layer_once_ttl_5ms": "layer_once_ttl5_pressure_route",
                                "adaptive_t0_i100ms": "admission_plus_dedicated_cir",
                            }[case],
                        }
                    )
    return rows


def _build_ssd_accounting_residual_rows(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    float_fields = (
        "stable_service_minus_busy_counter_gb",
        "fragmented_service_minus_stable_gb",
        "physical_queue_minus_scheduler_gb",
        "enqueue_minus_service_minus_physical_queue_gb",
        "counter_queue_minus_physical_queue_gb",
        "fragmented_counter_queue_minus_physical_queue_gb",
        "maximum_abs_npu_queue_identity_residual_gb",
    )
    block_fields = (
        "physical_queue_block_minus_scheduler_blocks",
        "counter_queue_block_minus_physical_blocks",
    )
    for (num_ssu, case), data in sorted(cases.items()):
        for boundary_index, boundary in enumerate(data.boundaries):
            for residual in boundary["timeline"][
                "ssd_accounting_residuals_by_ssu"
            ]:
                row = {
                    "num_ssu": num_ssu,
                    "case": case,
                    "boundary": boundary_index,
                    "relative_time_s": boundary_index * BLOCK_MS / 1000.0,
                    "ssu_id": int(residual["ssu_id"]),
                    "timeline_schema": TIMELINE_SCHEMA,
                    "service_absolute_tolerance_gb": (
                        SSD_SERVICE_ABSOLUTE_TOLERANCE_GB
                    ),
                    "block_tolerance": 0,
                    "scientific_ssd_service_source": (
                        "stable completed-command cumulative plus immutable active-prefix"
                    ),
                    "ssd_outstanding_source": (
                        "direct physical pending/active enumeration"
                    ),
                    "fragmented_service_is_diagnostic_only_not_scientific_output": True,
                    "maximum_abs_npu_queue_identity_residual_npu_id": int(
                        residual[
                            "maximum_abs_npu_queue_identity_residual_npu_id"
                        ]
                    ),
                }
                for field in float_fields:
                    value = float(residual[field])
                    row[field] = value
                    row[field.replace("_gb", "_decimal_bytes")] = value * 1e9
                    diagnostic_only = field in {
                        "fragmented_service_minus_stable_gb",
                        "fragmented_counter_queue_minus_physical_queue_gb",
                    }
                    row[f"{field}_scientific_gate_applicable"] = (
                        not diagnostic_only
                    )
                    row[f"{field}_within_declared_tolerance"] = (
                        None
                        if diagnostic_only
                        else abs(value)
                        <= SSD_SERVICE_ABSOLUTE_TOLERANCE_GB + 1e-15
                    )
                for field in block_fields:
                    value = int(residual[field])
                    row[field] = value
                    row[f"{field}_within_declared_tolerance"] = value == 0
                rows.append(row)
    _require(
        len(rows)
        == sum(BOUNDARY_COUNT * num_ssu * len(CASES) for num_ssu in SSU_COUNTS),
        "SSD accounting residual export row count mismatch",
    )
    return rows


def _request_window_metrics(
    data: CaseData, relative_start_s: float, relative_end_s: float
) -> dict:
    absolute_start = data.measurement_start_ms + relative_start_s * 1000.0
    absolute_end = data.measurement_start_ms + relative_end_s * 1000.0
    rows = [
        row
        for row in data.requests
        if absolute_start <= float(row["admission_time_ms"]) < absolute_end
    ]
    by_npu: list[list[dict]] = [[] for _ in range(NUM_NPU)]
    for row in rows:
        by_npu[int(row["npu_id"])].append(row)
    result: dict[str, object] = {
        "request_count": len(rows),
        "npus_with_requests": sum(bool(values) for values in by_npu),
        "mean_io_barrier_ms": _safe_mean([float(row["io_barrier_ms"]) for row in rows]),
        "p99_io_barrier_ms": _safe_percentile([float(row["io_barrier_ms"]) for row in rows], 99),
        "mean_ttft_ms": _safe_mean([float(row["ttft_ms"]) for row in rows]),
    }
    for alpha in SLO_ALPHAS:
        suffix = "alpha1p5" if alpha == 1.5 else "alpha2"
        outcomes = [float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + TOL for row in rows]
        per_npu = [
            statistics.fmean(
                float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + TOL
                for row in values
            )
            for values in by_npu
            if values
        ]
        result[f"request_weighted_slo_{suffix}"] = _safe_mean(outcomes)
        result[f"equal_observed_npu_slo_{suffix}"] = _safe_mean(per_npu)
    return result


def _one_window_metric(
    data: CaseData,
    *,
    kind: str,
    requested_duration_s: float,
    window_index: int,
    start_block: int,
    end_block: int,
) -> dict:
    _require(0 <= start_block < end_block <= BLOCK_COUNT, "invalid window block range")
    before = data.boundaries[start_block]
    after = data.boundaries[end_block]
    duration_ms = (end_block - start_block) * BLOCK_MS
    relative_start_s = start_block * BLOCK_MS / 1000.0
    relative_end_s = end_block * BLOCK_MS / 1000.0
    compute_before = before["npu_compute_cumulative_busy_ms_by_npu"]
    compute_after = after["npu_compute_cumulative_busy_ms_by_npu"]
    compute_ms = sum(float(right) - float(left) for left, right in zip(compute_before, compute_after))
    ssd_before = before["ssd_cumulative_busy_ms_by_ssu"]
    ssd_after = after["ssd_cumulative_busy_ms_by_ssu"]
    ssd_busy_ms = sum(float(right) - float(left) for left, right in zip(ssd_before, ssd_after))
    link_before = before["npu_link_cumulative_busy_ms_by_npu"]
    link_after = after["npu_link_cumulative_busy_ms_by_npu"]
    link_busy_ms = sum(float(right) - float(left) for left, right in zip(link_before, link_after))
    return {
        "num_ssu": data.num_ssu,
        "case": data.case,
        "case_label": CASE_LABELS[data.case],
        "window_kind": kind,
        "requested_duration_s": requested_duration_s,
        "window_index": window_index,
        "relative_start_s": relative_start_s,
        "relative_end_s": relative_end_s,
        "actual_duration_s": duration_ms / 1000.0,
        "full_requested_window": _close(duration_ms / 1000.0, requested_duration_s),
        "npu_utilization": compute_ms / (NUM_NPU * duration_ms),
        "ssd_mean_utilization": ssd_busy_ms / (data.num_ssu * duration_ms),
        "npu_link_mean_utilization": link_busy_ms / (NUM_NPU * duration_ms),
        "utilization_window_semantics": (
            "exact resource-time integral over [relative_start_s,relative_end_s)"
        ),
        "request_cohort_semantics": (
            "requests admitted in [relative_start_s,relative_end_s); TTFT completion may occur after relative_end_s"
        ),
        **_request_window_metrics(data, relative_start_s, relative_end_s),
    }


def _build_window_metrics(cases: Mapping[tuple[int, str], CaseData]) -> list[dict]:
    rows: list[dict] = []
    for data in cases.values():
        for horizon_s in HORIZONS_S:
            block_count = int(round(horizon_s * 1000.0 / BLOCK_MS))
            _require(_close(block_count * BLOCK_MS, horizon_s * 1000.0), "horizon is not boundary aligned")
            rows.append(
                _one_window_metric(
                    data,
                    kind="cumulative",
                    requested_duration_s=horizon_s,
                    window_index=0,
                    start_block=0,
                    end_block=block_count,
                )
            )
            window_index = 0
            for start_block in range(0, BLOCK_COUNT, block_count):
                end_block = min(BLOCK_COUNT, start_block + block_count)
                rows.append(
                    _one_window_metric(
                        data,
                        kind="disjoint",
                        requested_duration_s=horizon_s,
                        window_index=window_index,
                        start_block=start_block,
                        end_block=end_block,
                    )
                )
                window_index += 1
    for data in cases.values():
        whole = next(
            row
            for row in rows
            if row["num_ssu"] == data.num_ssu
            and row["case"] == data.case
            and row["window_kind"] == "cumulative"
            and row["requested_duration_s"] == 64.0
        )
        _require(_close(float(whole["npu_utilization"]), float(data.summary["mean_npu_utilization"])), f"{data.key}: 64-s independent utilization mismatch")
        _require(_close(float(whole["request_weighted_slo_alpha2"]), float(data.summary["request_weighted_slo_attainment"])), f"{data.key}: 64-s independent alpha=2 SLO mismatch")
        _require(_close(float(whole["equal_observed_npu_slo_alpha2"]), float(data.summary["ttft_slo_attainment"])), f"{data.key}: 64-s independent equal-NPU alpha=2 SLO mismatch")
    return rows


def _window_compute_components(data: CaseData, start_block: int, end_block: int) -> dict:
    before = data.boundaries[start_block]
    after = data.boundaries[end_block]
    before_rows = before["timeline"]["npu_rows"]
    after_rows = after["timeline"]["npu_rows"]
    compute_ms = math.fsum(
        float(after["npu_compute_cumulative_busy_ms_by_npu"][npu])
        - float(before["npu_compute_cumulative_busy_ms_by_npu"][npu])
        for npu in range(NUM_NPU)
    )
    activated_ms = math.fsum(
        float(after_rows[npu]["activated_compute_cumulative_ms"])
        - float(before_rows[npu]["activated_compute_cumulative_ms"])
        for npu in range(NUM_NPU)
    )
    q_start_ms = math.fsum(float(row["compute_inventory_q_ms"]) for row in before_rows)
    q_end_ms = math.fsum(float(row["compute_inventory_q_ms"]) for row in after_rows)
    reconstructed_ms = activated_ms + q_start_ms - q_end_ms
    _require(_close(compute_ms, reconstructed_ms, tolerance=5e-6), f"{data.key}: aggregate Q/activation closure failed")
    return {
        "compute_busy_npu_s": compute_ms / 1000.0,
        "activated_compute_npu_s": activated_ms / 1000.0,
        "q_start_npu_s": q_start_ms / 1000.0,
        "q_end_npu_s": q_end_ms / 1000.0,
        "reconstructed_compute_npu_s": reconstructed_ms / 1000.0,
        "closure_error_npu_s": (compute_ms - reconstructed_ms) / 1000.0,
    }


def _build_compute_decomposition(
    cases: Mapping[tuple[int, str], CaseData], window_rows: Sequence[dict]
) -> list[dict]:
    rows: list[dict] = []
    for window in window_rows:
        data = cases[(int(window["num_ssu"]), str(window["case"]))]
        start_block = int(round(float(window["relative_start_s"]) * 1000.0 / BLOCK_MS))
        end_block = int(round(float(window["relative_end_s"]) * 1000.0 / BLOCK_MS))
        absolute = _window_compute_components(data, start_block, end_block)
        baseline = cases[(data.num_ssu, "baseline")]
        baseline_values = _window_compute_components(baseline, start_block, end_block)
        delta = {
            f"delta_vs_baseline_{field}": absolute[field] - baseline_values[field]
            for field in (
                "compute_busy_npu_s",
                "activated_compute_npu_s",
                "q_start_npu_s",
                "q_end_npu_s",
            )
        }
        delta_reconstructed = (
            delta["delta_vs_baseline_activated_compute_npu_s"]
            + delta["delta_vs_baseline_q_start_npu_s"]
            - delta["delta_vs_baseline_q_end_npu_s"]
        )
        _require(_close(delta["delta_vs_baseline_compute_busy_npu_s"], delta_reconstructed, tolerance=1e-8), f"{data.key}: policy-vs-baseline decomposition does not close")
        rows.append(
            {
                "num_ssu": data.num_ssu,
                "case": data.case,
                "window_kind": window["window_kind"],
                "requested_duration_s": window["requested_duration_s"],
                "window_index": window["window_index"],
                "relative_start_s": window["relative_start_s"],
                "relative_end_s": window["relative_end_s"],
                "full_requested_window": window["full_requested_window"],
                **absolute,
                **delta,
                "delta_vs_baseline_reconstructed_compute_npu_s": delta_reconstructed,
                "delta_vs_baseline_closure_error_npu_s": delta["delta_vs_baseline_compute_busy_npu_s"] - delta_reconstructed,
            }
        )
    return rows


def _build_state_duration_tables(
    cases: Mapping[tuple[int, str], CaseData]
) -> tuple[list[dict], list[dict]]:
    per_npu_rows: list[dict] = []
    summary_rows: list[dict] = []
    for num_ssu in SSU_COUNTS:
        baseline_rows = cases[(num_ssu, "baseline")].summary[
            "timeline_state_durations_ms_by_npu"
        ]
        for case in CASES:
            data = cases[(num_ssu, case)]
            source_rows = data.summary["timeline_state_durations_ms_by_npu"]
            for npu_id, (source, baseline) in enumerate(
                zip(source_rows, baseline_rows)
            ):
                per_npu_rows.append(
                    {
                        "num_ssu": num_ssu,
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "npu_id": npu_id,
                        "compute_ms": float(source["compute_ms"]),
                        "io_barrier_ms": float(source["io_barrier_ms"]),
                        "other_ms": float(source["other_ms"]),
                        "measurement_ms": float(source["measurement_ms"]),
                        "compute_fraction": float(source["compute_fraction"]),
                        "io_barrier_fraction": float(source["io_barrier_fraction"]),
                        "other_fraction": float(source["other_fraction"]),
                        "idle_fraction": float(source["io_barrier_fraction"])
                        + float(source["other_fraction"]),
                        "delta_vs_baseline_compute_ms": float(source["compute_ms"])
                        - float(baseline["compute_ms"]),
                        "delta_vs_baseline_io_barrier_ms": float(
                            source["io_barrier_ms"]
                        )
                        - float(baseline["io_barrier_ms"]),
                        "delta_vs_baseline_other_ms": float(source["other_ms"])
                        - float(baseline["other_ms"]),
                        "includes_carry_in": True,
                        "full_window_idle_root_cause_evidence": True,
                    }
                )
            compute_ms = math.fsum(float(row["compute_ms"]) for row in source_rows)
            barrier_ms = math.fsum(
                float(row["io_barrier_ms"]) for row in source_rows
            )
            other_ms = math.fsum(float(row["other_ms"]) for row in source_rows)
            total_ms = NUM_NPU * MEASUREMENT_MS
            _require(
                _close(compute_ms + barrier_ms + other_ms, total_ms, tolerance=2e-6),
                f"SSU{num_ssu}/{case}: fleet state partition does not close",
            )
            _require(
                _close(
                    compute_ms / total_ms,
                    float(data.summary["mean_npu_utilization"]),
                ),
                f"SSU{num_ssu}/{case}: state-duration compute != NPU utilization",
            )
            baseline_compute = math.fsum(
                float(row["compute_ms"]) for row in baseline_rows
            )
            baseline_barrier = math.fsum(
                float(row["io_barrier_ms"]) for row in baseline_rows
            )
            baseline_other = math.fsum(
                float(row["other_ms"]) for row in baseline_rows
            )
            summary_rows.append(
                {
                    "num_ssu": num_ssu,
                    "case": case,
                    "case_label": CASE_LABELS[case],
                    "compute_npu_s": compute_ms / 1000.0,
                    "io_barrier_npu_s": barrier_ms / 1000.0,
                    "other_npu_s": other_ms / 1000.0,
                    "compute_fraction": compute_ms / total_ms,
                    "io_barrier_fraction": barrier_ms / total_ms,
                    "other_fraction": other_ms / total_ms,
                    "idle_fraction": (barrier_ms + other_ms) / total_ms,
                    "io_barrier_share_of_idle": (
                        barrier_ms / (barrier_ms + other_ms)
                        if barrier_ms + other_ms > 0.0
                        else 0.0
                    ),
                    "delta_vs_baseline_compute_npu_s": (
                        compute_ms - baseline_compute
                    )
                    / 1000.0,
                    "delta_vs_baseline_io_barrier_npu_s": (
                        barrier_ms - baseline_barrier
                    )
                    / 1000.0,
                    "delta_vs_baseline_other_npu_s": (
                        other_ms - baseline_other
                    )
                    / 1000.0,
                    "delta_partition_closure_npu_s": (
                        compute_ms
                        - baseline_compute
                        + barrier_ms
                        - baseline_barrier
                        + other_ms
                        - baseline_other
                    )
                    / 1000.0,
                    "includes_carry_in": True,
                    "full_window_idle_root_cause_evidence": True,
                }
            )
    return per_npu_rows, summary_rows


def _build_block_state_duration_rows(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    for (num_ssu, case), data in sorted(cases.items()):
        for source in data.summary["timeline_block_state_durations_ms"]:
            block = int(source["block"])
            duration_ms = float(source["duration_ms"])
            for npu_id in range(NUM_NPU):
                compute = float(source["compute_ms_by_npu"][npu_id])
                barrier = float(source["io_barrier_ms_by_npu"][npu_id])
                other = float(source["other_ms_by_npu"][npu_id])
                rows.append(
                    {
                        "num_ssu": num_ssu,
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "block": block,
                        "relative_start_s": block * BLOCK_MS / 1000.0,
                        "relative_end_s": (block + 1) * BLOCK_MS / 1000.0,
                        "npu_id": npu_id,
                        "compute_ms": compute,
                        "io_barrier_ms": barrier,
                        "other_ms": other,
                        "duration_ms": duration_ms,
                        "compute_fraction": compute / duration_ms,
                        "io_barrier_fraction": barrier / duration_ms,
                        "other_fraction": other / duration_ms,
                        "includes_carry_in_at_measurement_start": True,
                        "exact_microbatch_interval_intersection": True,
                    }
                )
    return rows


def _request_key(row: Mapping[str, object]) -> tuple[int, int]:
    return int(row["npu_id"]), int(row["sequence"])


def _build_matched_requests(
    cases: Mapping[tuple[int, str], CaseData]
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    wide_rows: list[dict] = []
    layer_rows: list[dict] = []
    summary_rows: list[dict] = []
    common_counts: list[dict] = []
    for num_ssu in SSU_COUNTS:
        maps = {
            case: {_request_key(row): row for row in cases[(num_ssu, case)].requests}
            for case in CASES
        }
        common = set.intersection(*(set(mapping) for mapping in maps.values()))
        union = set.union(*(set(mapping) for mapping in maps.values()))
        _require(common, f"SSU{num_ssu}: no common request identities")
        for npu_id in range(NUM_NPU):
            identities = {
                case: {key for key in mapping if key[0] == npu_id}
                for case, mapping in maps.items()
            }
            common_npu = set.intersection(*identities.values())
            union_npu = set.union(*identities.values())
            common_counts.append(
                {
                    "num_ssu": num_ssu,
                    "npu_id": npu_id,
                    "common_request_count": len(common_npu),
                    "union_request_count": len(union_npu),
                    "intersection_over_union": (
                        len(common_npu) / len(union_npu) if union_npu else None
                    ),
                    **{
                        f"{case}_measurement_request_count": len(
                            identities[case]
                        )
                        for case in CASES
                    },
                    "cohort_evidence_only_not_full_window_idle": True,
                }
            )
        for key in sorted(common):
            source = {case: maps[case][key] for case in CASES}
            ideals = {float(row["ideal_ttft_ms"]) for row in source.values()}
            profiles = {row.get("profile_id") for row in source.values()}
            _require(len(ideals) == 1 and len(profiles) == 1, f"SSU{num_ssu}/{key}: matched profile differs")
            baseline = source["baseline"]
            wide: dict[str, object] = {
                "num_ssu": num_ssu,
                "npu_id": key[0],
                "sequence": key[1],
                "profile_id": baseline.get("profile_id"),
                "profile_name": baseline.get("profile_name"),
                "category": baseline.get("category"),
                "ideal_ttft_ms": float(baseline["ideal_ttft_ms"]),
            }
            for case, row in source.items():
                prefix = case
                ttft = float(row["ttft_ms"])
                barrier = float(row["io_barrier_ms"])
                wide[f"{prefix}_request_id"] = int(row["request_id"])
                wide[f"{prefix}_admission_relative_s"] = (
                    float(row["admission_time_ms"])
                    - cases[(num_ssu, case)].measurement_start_ms
                ) / 1000.0
                wide[f"{prefix}_completion_relative_s"] = (
                    float(row["completion_time_ms"])
                    - cases[(num_ssu, case)].measurement_start_ms
                ) / 1000.0
                wide[f"{prefix}_ttft_ms"] = ttft
                wide[f"{prefix}_io_barrier_ms"] = barrier
                wide[f"{prefix}_slo_alpha1p5"] = ttft <= 1.5 * float(row["ideal_ttft_ms"]) + TOL
                wide[f"{prefix}_slo_alpha2"] = ttft <= 2.0 * float(row["ideal_ttft_ms"]) + TOL
                wide[f"{prefix}_barrier_delta_vs_baseline_ms"] = barrier - float(baseline["io_barrier_ms"])
                layers = row["timeline_layers"]
                for layer in range(N_LAYERS):
                    layer_rows.append(
                        {
                            "num_ssu": num_ssu,
                            "case": case,
                            "npu_id": key[0],
                            "sequence": key[1],
                            "request_id": int(row["request_id"]),
                            "layer": layer,
                            "io_start_relative_to_admission_ms": float(layers["io_start_time_ms"][layer]) - float(row["admission_time_ms"]),
                            "io_ready_relative_to_admission_ms": float(layers["io_ready_time_ms"][layer]) - float(row["admission_time_ms"]),
                            "compute_start_relative_to_admission_ms": float(layers["compute_start_ms"][layer]) - float(row["admission_time_ms"]),
                            "compute_end_relative_to_admission_ms": float(layers["compute_end_ms"][layer]) - float(row["admission_time_ms"]),
                            "io_barrier_wait_ms": float(layers["io_barrier_wait_ms"][layer]),
                            **{
                                f"work_gb_ssu{ssu_id}": float(layers["per_layer_work_gb_by_ssu"][ssu_id])
                                for ssu_id in range(num_ssu)
                            },
                        }
                    )
            wide_rows.append(wide)

        for policy in CASES[1:]:
            matched = [row for row in wide_rows if row["num_ssu"] == num_ssu]
            deltas = [float(row[f"{policy}_barrier_delta_vs_baseline_ms"]) for row in matched]
            summary: dict[str, object] = {
                "num_ssu": num_ssu,
                "policy": policy,
                "matched_request_count": len(matched),
                "mean_barrier_delta_vs_baseline_ms": statistics.fmean(deltas),
                "median_barrier_delta_vs_baseline_ms": float(np.median(deltas)),
                "p99_barrier_delta_vs_baseline_ms": float(np.percentile(deltas, 99)),
                "total_barrier_delta_vs_baseline_s": math.fsum(deltas) / 1000.0,
                "barrier_improved_request_count": sum(delta < -TOL for delta in deltas),
                "barrier_regressed_request_count": sum(delta > TOL for delta in deltas),
                "barrier_unchanged_request_count": sum(abs(delta) <= TOL for delta in deltas),
            }
            for alpha_suffix in ("alpha1p5", "alpha2"):
                baseline_field = f"baseline_slo_{alpha_suffix}"
                policy_field = f"{policy}_slo_{alpha_suffix}"
                summary[f"baseline_matched_slo_{alpha_suffix}"] = statistics.fmean(bool(row[baseline_field]) for row in matched)
                summary[f"policy_matched_slo_{alpha_suffix}"] = statistics.fmean(bool(row[policy_field]) for row in matched)
                summary[f"fail_to_pass_{alpha_suffix}"] = sum(not bool(row[baseline_field]) and bool(row[policy_field]) for row in matched)
                summary[f"pass_to_fail_{alpha_suffix}"] = sum(bool(row[baseline_field]) and not bool(row[policy_field]) for row in matched)
            summary_rows.append(summary)
    return wide_rows, layer_rows, summary_rows, common_counts


def _request_metadata_for_slack(data: CaseData) -> dict[int, tuple[float, float]]:
    metadata: dict[int, tuple[float, float]] = {}
    for row in data.requests:
        metadata[int(row["request_id"])] = (
            float(row["admission_time_ms"]),
            float(row["ideal_ttft_ms"]),
        )
    for boundary in data.boundaries:
        for row in boundary["timeline"]["npu_rows"]:
            request_id = row.get("active_request_id")
            admission = row.get("admission_time_ms")
            ideal = row.get("ideal_ttft_ms")
            if request_id is None or admission is None or ideal is None:
                continue
            value = (float(admission), float(ideal))
            previous = metadata.get(int(request_id))
            _require(previous is None or _vectors_close(previous, value), f"{data.key}: inconsistent request slack metadata")
            metadata[int(request_id)] = value
    return metadata


def _changed_cir_entries(
    installed: Sequence[Sequence[float]], target: Sequence[Sequence[float]]
) -> tuple[list[list[float]], list[list[bool]]]:
    next_installed: list[list[float]] = []
    changed: list[list[bool]] = []
    for old_row, target_row in zip(installed, target):
        new_row: list[float] = []
        changed_row: list[bool] = []
        for old, new in zip(old_row, target_row):
            selected = abs(float(new) - float(old)) > 1e-12
            changed_row.append(selected)
            new_row.append(float(new) if selected else float(old))
        next_installed.append(new_row)
        changed.append(changed_row)
    return next_installed, changed


def _decision_prefetch_map(
    record: Mapping[str, object],
    request_by_npu: Mapping[int, int],
    context: str,
) -> dict[int, bool] | None:
    """Parse the v2 sorted ``(npu_id, bool)`` prefetch classification."""
    raw = record.get("prefetch_only_by_npu")
    if raw is None:
        return None
    _require(isinstance(raw, (list, tuple)), f"{context}.prefetch_only_by_npu: expected a list")
    parsed: dict[int, bool] = {}
    for index, item in enumerate(raw):
        item_context = f"{context}.prefetch_only_by_npu[{index}]"
        _require(isinstance(item, (list, tuple)) and len(item) == 2, f"{item_context}: expected (npu,bool)")
        npu = _integer(item[0], f"{item_context}.npu")
        _require(type(item[1]) is bool, f"{item_context}.value: expected bool")
        _require(npu not in parsed, f"{item_context}: duplicate NPU")
        parsed[npu] = bool(item[1])
    _require(list(parsed) == sorted(parsed), f"{context}.prefetch_only_by_npu: pairs are not sorted")
    _require(set(parsed) == set(request_by_npu), f"{context}.prefetch_only_by_npu: pair coverage differs from request_by_npu")
    return parsed


def _build_adaptive_decisions(
    cases: Mapping[tuple[int, str], CaseData]
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    decision_rows: list[dict] = []
    npu_rows: list[dict] = []
    boundary_slack_rows: list[dict] = []
    attempt_rows: list[dict] = []
    for num_ssu in SSU_COUNTS:
        data = cases[(num_ssu, "adaptive_t0_i100ms")]
        records = data.summary["adaptive_decision_diagnostics"]
        start = data.measurement_start_ms
        end = float(data.summary["measurement_end_ms"])
        metadata = _request_metadata_for_slack(data)
        installed = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        replay: list[dict] = []
        measurement_commits = 0
        measurement_transactions = [0] * num_ssu
        measurement_writes = [0] * num_ssu
        for record_index, record in enumerate(records):
            context = f"SSU{num_ssu}.adaptive_decision[{record_index}]"
            target = _matrix(record.get("grants_gbps_by_npu_ssu"), NUM_NPU, num_ssu, f"{context}.grants")
            demand = _matrix(record.get("controller_demand_gbps_by_npu_ssu"), NUM_NPU, num_ssu, f"{context}.demand")
            assert target is not None and demand is not None
            old = [list(row) for row in installed]
            installed, changed = _changed_cir_entries(installed, target)
            changed_by_ssu = [sum(changed[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)]
            changed_total = sum(changed_by_ssu)
            time_ms = _number(record.get("snapshot_time_ms"), f"{context}.time")
            in_measurement = start <= time_ms < end
            if in_measurement:
                measurement_commits += int(changed_total > 0)
                for ssu_id, count in enumerate(changed_by_ssu):
                    measurement_transactions[ssu_id] += int(count > 0)
                    measurement_writes[ssu_id] += count
            request_by_npu = {
                int(npu): int(request_id)
                for npu, request_id in record.get("request_by_npu", [])
            }
            _require(len(request_by_npu) == len(record.get("request_by_npu", [])), f"{context}: duplicate request_by_npu entry")
            prefetch_by_npu = _decision_prefetch_map(
                record, request_by_npu, context
            )
            selected = {int(value) for value in record.get("selected_npu_ids", [])}
            rejected = {int(value) for value in record.get("rejected_npu_ids", [])}
            pinned = {int(value) for value in record.get("previous_pinned_npu_ids", [])}
            candidate_order = [
                int(value) for value in record.get("candidate_order", [])
            ]
            candidate_rank = {
                npu_id: rank for rank, npu_id in enumerate(candidate_order)
            }
            scores = {
                int(item["npu_id"]): item
                for item in record.get("candidate_normalized_scores", [])
            }
            attempts_by_npu: dict[int, list[dict]] = defaultdict(list)
            for attempt in record.get("admission_attempts", []):
                attempts_by_npu[int(attempt["npu_id"])].append(attempt)
            remaining_compute = _vector(record.get("remaining_compute_s_by_npu"), NUM_NPU, f"{context}.remaining_compute_s")
            slacks15: dict[int, float] = {}
            slacks2: dict[int, float] = {}
            slack_eligible: dict[int, bool] = {}
            for npu_id, request_id in request_by_npu.items():
                if request_id not in metadata:
                    slack_eligible[npu_id] = False
                    continue
                admission, ideal = metadata[request_id]
                explicit_prefetch = (
                    None
                    if prefetch_by_npu is None
                    else prefetch_by_npu[npu_id]
                )
                inferred_prefetch = time_ms < admission - TOL
                if explicit_prefetch is not None:
                    if explicit_prefetch:
                        _require(
                            time_ms <= admission + TOL,
                            f"{context}: explicit prefetch request was already admitted",
                        )
                    else:
                        _require(
                            not inferred_prefetch,
                            f"{context}: non-prefetch decision precedes admission",
                        )
                is_prefetch = bool(explicit_prefetch) or inferred_prefetch
                slack_eligible[npu_id] = not is_prefetch
                if is_prefetch:
                    continue
                elapsed = time_ms - admission
                slacks15[npu_id] = 1.5 * ideal - elapsed
                slacks2[npu_id] = 2.0 * ideal - elapsed
            triggers = [str(value) for value in record.get("trigger_reasons", [])]
            aggregate = {
                "num_ssu": num_ssu,
                "evaluation": _integer(record.get("snapshot_evaluation"), f"{context}.evaluation"),
                "relative_time_s": (time_ms - start) / 1000.0,
                "inside_measurement_window": in_measurement,
                "trigger_reasons": "|".join(triggers),
                "layer_jobs_since_previous": _integer(record.get("layer_jobs_since_previous"), f"{context}.layer_jobs"),
                "selection_mode": record.get("selection_mode"),
                "candidate_order": "|".join(map(str, candidate_order)),
                "candidate_count": len(candidate_order),
                "residual_mode": record.get("residual_mode"),
                "effective_target_ratio": _number(record.get("effective_target_ratio"), f"{context}.target_ratio"),
                "active_npu_count": len(record.get("active_npu_ids", [])),
                "selected_npu_count": len(selected),
                "rejected_npu_count": len(rejected),
                "previous_pinned_npu_count": len(pinned),
                "selected_fraction": _number(record.get("selected_fraction"), f"{context}.selected_fraction"),
                "changed_cir_entry_count": changed_total,
                "changed_ssu_transaction_count": sum(count > 0 for count in changed_by_ssu),
                "deadline_or_slack_used_by_controller": False,
                "prefetch_requests_excluded_from_slack": True,
                "deadline_slack_coverage_count": len(slacks2),
                "prefetch_only_npu_count": (
                    None
                    if prefetch_by_npu is None
                    else sum(prefetch_by_npu.values())
                ),
                "selected_deadline_slack_alpha1p5_median_ms_diagnostic_only": _safe_percentile([value for npu, value in slacks15.items() if npu in selected], 50),
                "rejected_deadline_slack_alpha1p5_median_ms_diagnostic_only": _safe_percentile([value for npu, value in slacks15.items() if npu in rejected], 50),
                "selected_deadline_slack_alpha2_median_ms_diagnostic_only": _safe_percentile([value for npu, value in slacks2.items() if npu in selected], 50),
                "rejected_deadline_slack_alpha2_median_ms_diagnostic_only": _safe_percentile([value for npu, value in slacks2.items() if npu in rejected], 50),
                **{f"target_grant_sum_ssu{ssu}_gbps": math.fsum(target[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)},
                **{f"old_installed_cir_sum_ssu{ssu}_gbps": math.fsum(old[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)},
                **{f"installed_after_sum_ssu{ssu}_gbps": math.fsum(installed[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)},
                **{f"changed_entries_ssu{ssu}": changed_by_ssu[ssu] for ssu in range(num_ssu)},
            }
            replay.append(
                {
                    "time_ms": time_ms,
                    "installed_after": [list(row) for row in installed],
                    "request_by_npu": dict(request_by_npu),
                    "prefetch_only_by_npu": dict(prefetch_by_npu or {}),
                    "selected_request_by_npu": {
                        npu_id: request_by_npu[npu_id]
                        for npu_id in selected
                        if npu_id in request_by_npu
                    },
                    "rejected_request_by_npu": {
                        npu_id: request_by_npu[npu_id]
                        for npu_id in rejected
                        if npu_id in request_by_npu
                    },
                }
            )
            if in_measurement:
                decision_rows.append(aggregate)
                for attempt in record.get("admission_attempts", []):
                    attempt_row = {
                        "num_ssu": num_ssu,
                        "evaluation": aggregate["evaluation"],
                        "relative_time_s": aggregate["relative_time_s"],
                        "selection_mode": aggregate["selection_mode"],
                        "attempt_index": int(attempt["attempt_index"]),
                        "npu_id": int(attempt["npu_id"]),
                        "stage": attempt["stage"],
                        "accepted": bool(attempt["accepted"]),
                        "rejection_reason": attempt.get("rejection_reason"),
                        "target_sum_gbps": float(attempt["target_sum_gbps"]),
                        "npu_capacity_gbps": float(
                            attempt["npu_capacity_gbps"]
                        ),
                        "target_gbps_by_ssu": _canonical(
                            attempt["target_gbps_by_ssu"]
                        ),
                        "admission_remaining_before_gbps_by_ssu": _canonical(
                            attempt[
                                "admission_remaining_before_gbps_by_ssu"
                            ]
                        ),
                        "admission_remaining_after_gbps_by_ssu": _canonical(
                            attempt[
                                "admission_remaining_after_gbps_by_ssu"
                            ]
                        ),
                        "violating_ssu_ids": "|".join(
                            map(str, attempt.get("violating_ssu_ids", []))
                        ),
                    }
                    for ssu_id in range(num_ssu):
                        attempt_row[f"target_ssu{ssu_id}_gbps"] = float(
                            attempt["target_gbps_by_ssu"][ssu_id]
                        )
                        attempt_row[
                            f"admission_remaining_before_ssu{ssu_id}_gbps"
                        ] = float(
                            attempt[
                                "admission_remaining_before_gbps_by_ssu"
                            ][ssu_id]
                        )
                        attempt_row[
                            f"admission_remaining_after_ssu{ssu_id}_gbps"
                        ] = float(
                            attempt[
                                "admission_remaining_after_gbps_by_ssu"
                            ][ssu_id]
                        )
                    attempt_rows.append(attempt_row)
                for npu_id, request_id in sorted(request_by_npu.items()):
                    final_attempt = attempts_by_npu.get(npu_id, [])[-1] if attempts_by_npu.get(npu_id) else None
                    admission, ideal = metadata.get(request_id, (math.nan, math.nan))
                    explicit_prefetch = (
                        None
                        if prefetch_by_npu is None
                        else prefetch_by_npu[npu_id]
                    )
                    inferred_prefetch = (
                        math.isfinite(admission) and time_ms < admission - TOL
                    )
                    is_prefetch = bool(explicit_prefetch) or inferred_prefetch
                    eligible = bool(slack_eligible.get(npu_id, False))
                    elapsed = (
                        time_ms - admission
                        if eligible and math.isfinite(admission)
                        else math.nan
                    )
                    score = scores.get(npu_id, {})
                    npu_rows.append(
                        {
                            "num_ssu": num_ssu,
                            "evaluation": aggregate["evaluation"],
                            "relative_time_s": aggregate["relative_time_s"],
                            "npu_id": npu_id,
                            "request_id": request_id,
                            "selected": npu_id in selected,
                            "rejected": npu_id in rejected,
                            "previously_pinned": npu_id in pinned,
                            "selection_mode": aggregate["selection_mode"],
                            "residual_mode": aggregate["residual_mode"],
                            "effective_target_ratio": aggregate[
                                "effective_target_ratio"
                            ],
                            "candidate_normalized_total": score.get("normalized_total"),
                            "candidate_normalized_dominant": score.get("normalized_dominant"),
                            "candidate_order_rank_zero_based": candidate_rank.get(
                                npu_id
                            ),
                            "admission_stage": None if final_attempt is None else final_attempt.get("stage"),
                            "admission_accepted": None if final_attempt is None else final_attempt.get("accepted"),
                            "rejection_reason": None if final_attempt is None else final_attempt.get("rejection_reason"),
                            "violating_ssu_ids": "" if final_attempt is None else "|".join(map(str, final_attempt.get("violating_ssu_ids", []))),
                            "remaining_controller_compute_ms": float(remaining_compute[npu_id]) * 1000.0,
                            "prefetch_only_at_decision": is_prefetch,
                            "prefetch_classification_source": (
                                "explicit_prefetch_only_by_npu"
                                if explicit_prefetch is not None
                                else "inferred_from_snapshot_before_admission"
                            ),
                            "slack_eligible_admitted_nonprefetch": eligible,
                            "elapsed_ttft_ms_diagnostic_only": None if not eligible else elapsed,
                            "deadline_slack_alpha1p5_ms_diagnostic_only": None if not eligible or not math.isfinite(ideal) else 1.5 * ideal - elapsed,
                            "deadline_slack_alpha2_ms_diagnostic_only": None if not eligible or not math.isfinite(ideal) else 2.0 * ideal - elapsed,
                            "prefetch_requests_excluded_from_slack": True,
                            "deadline_or_slack_used_by_controller": False,
                            **{f"controller_demand_ssu{ssu}_gbps": demand[npu_id][ssu] for ssu in range(num_ssu)},
                            **{f"old_installed_cir_ssu{ssu}_gbps": old[npu_id][ssu] for ssu in range(num_ssu)},
                            **{f"target_grant_ssu{ssu}_gbps": target[npu_id][ssu] for ssu in range(num_ssu)},
                            **{f"installed_after_ssu{ssu}_gbps": installed[npu_id][ssu] for ssu in range(num_ssu)},
                            **{f"cir_entry_written_ssu{ssu}": changed[npu_id][ssu] for ssu in range(num_ssu)},
                        }
                    )

        _require(measurement_commits == _integer(data.summary.get("measurement_cir_commits"), f"SSU{num_ssu}.measurement_commits"), f"SSU{num_ssu}: reconstructed CIR commit count mismatch")
        _require(measurement_transactions == _vector(data.summary.get("measurement_cir_write_transactions_by_ssu"), num_ssu, f"SSU{num_ssu}.transactions", integer=True), f"SSU{num_ssu}: reconstructed CIR transaction count mismatch")
        _require(measurement_writes == _vector(data.summary.get("measurement_cir_path_writes_by_ssu"), num_ssu, f"SSU{num_ssu}.writes", integer=True), f"SSU{num_ssu}: reconstructed CIR path-write count mismatch")

        replay_index = 0
        expected_installed = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        latest_decision_request_by_npu: dict[int, int] = {}
        latest_decision_prefetch_by_npu: dict[int, bool] = {}
        selected_request_at_left: dict[int, int] = {}
        rejected_request_at_left: dict[int, int] = {}
        for boundary_index, boundary in enumerate(data.boundaries):
            boundary_time = float(boundary["time_ms"])
            while replay_index < len(replay) and float(replay[replay_index]["time_ms"]) < boundary_time:
                expected_installed = replay[replay_index]["installed_after"]
                latest_decision_request_by_npu = replay[replay_index][
                    "request_by_npu"
                ]
                latest_decision_prefetch_by_npu = replay[replay_index][
                    "prefetch_only_by_npu"
                ]
                selected_request_at_left = replay[replay_index][
                    "selected_request_by_npu"
                ]
                rejected_request_at_left = replay[replay_index][
                    "rejected_request_by_npu"
                ]
                replay_index += 1
            actual = boundary["timeline"]["npu_ssu"]["installed_dedicated_path_cir_gbps"]
            _require(_matrices_close(expected_installed, actual), f"SSU{num_ssu}: reconstructed old CIR != boundary {boundary_index} installed CIR")
            for npu_row in boundary["timeline"]["npu_rows"]:
                if npu_row.get("active_request_id") is None:
                    continue
                controller_request_id = npu_row.get("controller_request_id")
                controller_prefetch = npu_row.get("controller_prefetch_only")
                _require(
                    controller_prefetch is None
                    or type(controller_prefetch) is bool,
                    f"SSU{num_ssu}: boundary controller_prefetch_only is not bool/null",
                )
                request_matches = (
                    controller_request_id is not None
                    and int(controller_request_id)
                    == int(npu_row["active_request_id"])
                )
                active_request_id = int(npu_row["active_request_id"])
                npu_id = int(npu_row["npu_id"])
                decision_request_id = latest_decision_request_by_npu.get(npu_id)
                decision_prefetch_only = latest_decision_prefetch_by_npu.get(
                    npu_id
                )
                decision_request_matches = (
                    request_matches
                    and decision_request_id is not None
                    and decision_request_id == active_request_id
                    and decision_prefetch_only is False
                    and controller_prefetch is False
                )
                if (
                    decision_request_matches
                    and selected_request_at_left.get(npu_id) == active_request_id
                ):
                    selection_classification = "selected"
                    selected_value: bool | None = True
                elif (
                    decision_request_matches
                    and rejected_request_at_left.get(npu_id) == active_request_id
                ):
                    selection_classification = "rejected"
                    selected_value = False
                else:
                    selection_classification = (
                        "unknown_prefetch_decision"
                        if request_matches
                        and decision_request_id == active_request_id
                        and decision_prefetch_only is True
                        else "unknown_or_stale_decision"
                    )
                    selected_value = None
                slack_eligible = decision_request_matches and selected_value is not None
                q_ms = float(npu_row["compute_inventory_q_ms"])
                deadline15 = (
                    npu_row.get("slo_alpha1p5_slack_ms")
                    if slack_eligible
                    else None
                )
                deadline2 = (
                    npu_row.get("slo_alpha2_slack_ms")
                    if slack_eligible
                    else None
                )
                boundary_slack_rows.append(
                    {
                        "num_ssu": num_ssu,
                        "boundary": boundary_index,
                        "relative_time_s": (boundary_time - start) / 1000.0,
                        "npu_id": npu_id,
                        "request_id": active_request_id,
                        "controller_request_id": controller_request_id,
                        "latest_strictly_prior_decision_request_id": decision_request_id,
                        "latest_strictly_prior_decision_prefetch_only": (
                            decision_prefetch_only
                        ),
                        "latest_decision_request_matches_boundary_active": decision_request_matches,
                        "selection_classification": selection_classification,
                        "controller_prefetch_only": controller_prefetch,
                        "slack_request_matches_controller_request": request_matches,
                        "slack_eligible_admitted_nonprefetch": slack_eligible,
                        "prefetch_requests_excluded_from_slack": True,
                        "selected_by_latest_strictly_prior_decision": selected_value,
                        "compute_inventory_q_ms": q_ms,
                        "deadline_slack_alpha1p5_ms_diagnostic_only": deadline15,
                        "deadline_slack_alpha2_ms_diagnostic_only": deadline2,
                        "slack_after_remaining_compute_alpha1p5_ms_diagnostic_only": None if deadline15 is None else float(deadline15) - q_ms,
                        "slack_after_remaining_compute_alpha2_ms_diagnostic_only": None if deadline2 is None else float(deadline2) - q_ms,
                        "deadline_or_slack_used_by_controller": False,
                    }
                )
    return decision_rows, npu_rows, boundary_slack_rows, attempt_rows


def _build_adaptive_grant_components(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    component_fields = (
        ("v2_floor_grants_gbps", "floor_grant_gbps"),
        ("v2_background_grants_gbps", "background_grant_gbps"),
        ("v2_selected_tail_grants_gbps", "selected_tail_grant_gbps"),
        ("v2_spill_tail_grants_gbps", "spill_tail_grant_gbps"),
    )
    for num_ssu in SSU_COUNTS:
        data = cases[(num_ssu, "adaptive_t0_i100ms")]
        start = data.measurement_start_ms
        end = float(data.summary["measurement_end_ms"])
        for record in data.summary["adaptive_decision_diagnostics"]:
            time_ms = float(record["snapshot_time_ms"])
            if not start <= time_ms < end:
                continue
            final = record["grants_gbps_by_npu_ssu"]
            raw_components = [record.get(field) for field, _ in component_fields]
            has_v2 = all(component is not None for component in raw_components)
            for npu_id in range(NUM_NPU):
                for ssu_id in range(num_ssu):
                    values = [
                        None
                        if component is None
                        else float(component[npu_id][ssu_id])
                        for component in raw_components
                    ]
                    row = {
                        "num_ssu": num_ssu,
                        "evaluation": int(record["snapshot_evaluation"]),
                        "relative_time_s": (time_ms - start) / 1000.0,
                        "npu_id": npu_id,
                        "ssu_id": ssu_id,
                        "selection_mode": record["selection_mode"],
                        "residual_mode": record["residual_mode"],
                        "v2_components_present": has_v2,
                        "v2_effective_floor_ratio": record.get(
                            "v2_effective_floor_ratio"
                        ),
                        "final_grant_gbps": float(final[npu_id][ssu_id]),
                    }
                    for (_raw_field, export_field), value in zip(
                        component_fields, values
                    ):
                        row[export_field] = value
                    row["v1_coflow_grant_gbps"] = (
                        0.0 if has_v2 else row["final_grant_gbps"]
                    )
                    component_sum = math.fsum(
                        [value for value in values if value is not None]
                        + [row["v1_coflow_grant_gbps"]]
                    )
                    _require(
                        _close(component_sum, row["final_grant_gbps"]),
                        "Adaptive exported grant decomposition does not close",
                    )
                    row["component_sum_gbps"] = component_sum
                    row["component_closure_error_gbps"] = (
                        component_sum - row["final_grant_gbps"]
                    )
                    rows.append(row)
    return rows


def _build_dispatch_probe_rows(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    for (num_ssu, case), data in sorted(cases.items()):
        start = data.measurement_start_ms
        for index, source in enumerate(data.summary["timeline_dispatch_probe_records"]):
            row = {
                "num_ssu": num_ssu,
                "case": case,
                "probe_index": index,
                "relative_time_ms": float(source["time_ms"]) - start,
                "probe_window_ms": float(data.summary["timeline_dispatch_probe_ms"]),
                "probe_stream_truncated": bool(
                    data.summary["timeline_dispatch_probe_truncated"]
                ),
            }
            for field, value in source.items():
                if field == "time_ms":
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[field] = value
                else:
                    row[field] = _canonical(value)
            rows.append(row)
    return rows


def _build_route_probe_rows(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    for (num_ssu, case), data in sorted(cases.items()):
        start = data.measurement_start_ms
        for index, source in enumerate(data.summary["timeline_route_probe_records"]):
            replayed = _replay_route_probe(
                source,
                case,
                f"SSU{num_ssu}/{case}.route_probe[{index}]",
            )
            selected = tuple(int(value) for value in source["selected_path_ids"])
            row = {
                "num_ssu": num_ssu,
                "case": case,
                "probe_index": index,
                "relative_time_ms": float(source["time_ms"]) - start,
                "probe_window_ms": float(data.summary["timeline_dispatch_probe_ms"]),
                "probe_stream_truncated": bool(
                    data.summary["timeline_route_probe_truncated"]
                ),
                "allocation_reason_semantics": {
                    "baseline": "fixed Path 0",
                    "layer_once_ttl_5ms": "fresh-or-TTL-cached pressure snapshot, then pressure-aware once-per-layer path plan",
                    "adaptive_t0_i100ms": "dedicated NPU path; CIR decision audited separately",
                }[case],
                "portable_replayed_path_ids": _canonical(replayed),
                "portable_replay_exact_match": replayed == selected,
                "ttft_deadline_or_slack_used_for_route": False,
            }
            for field, value in source.items():
                if field == "time_ms":
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[field] = value
                else:
                    row[field] = _canonical(value)
            rows.append(row)
    return rows


def _build_path_state_rows(
    cases: Mapping[tuple[int, str], CaseData]
) -> list[dict]:
    rows: list[dict] = []
    for (num_ssu, case), data in sorted(cases.items()):
        start = data.measurement_start_ms
        for boundary in data.boundaries:
            timeline = boundary["timeline"]
            cells = timeline["npu_ssu"]
            path_rows_by_ssu: dict[int, list[dict]] = defaultdict(list)
            for path_row in timeline["sparse_ssu_path_rows"]:
                path_rows_by_ssu[int(path_row["ssu_id"])].append(path_row)
            pressure_by_ssu = {
                int(row["ssu_id"]): row
                for row in timeline["pressure_state_by_ssu"]
            }
            active_by_ssu = timeline["active_command_by_ssu"]
            for ssu_id in range(num_ssu):
                pressure = pressure_by_ssu[ssu_id]
                active = active_by_ssu[ssu_id]
                source_rows: list[dict | None] = path_rows_by_ssu.get(
                    ssu_id, []
                ) or [None]
                aggregate_outstanding = math.fsum(
                    float(cells["ssd_outstanding_gb"][npu_id][ssu_id])
                    for npu_id in range(NUM_NPU)
                )
                for source in source_rows:
                    path_id = None if source is None else int(source["path_id"])
                    row = {
                        "num_ssu": num_ssu,
                        "case": case,
                        "boundary": int(boundary["boundary"]),
                        "relative_time_s": (float(boundary["time_ms"]) - start)
                        / 1000.0,
                        "ssu_id": ssu_id,
                        "read_only_left_boundary_snapshot": True,
                        "estimated_next_arbitration_rate_is_not_historical_service": True,
                        "path_state_present": source is not None,
                        "path_is_active_command": bool(
                            source is not None
                            and active is not None
                            and path_id == int(active["path_id"])
                        ),
                        "active_command_npu_id": (
                            None if active is None else active["npu_id"]
                        ),
                        "active_command_request_id": (
                            None if active is None else active["request_id"]
                        ),
                        "active_command_layer": (
                            None if active is None else active["layer"]
                        ),
                        "active_command_path_id": (
                            None if active is None else active["path_id"]
                        ),
                        "active_command_remaining_gb": (
                            None if active is None else active["remaining_gb"]
                        ),
                        "active_command_start_time_ms": (
                            None
                            if active is None
                            else active["command_start_time_ms"]
                        ),
                        "active_command_age_ms": (
                            None if active is None else active["command_age_ms"]
                        ),
                        "active_command_physical_service_gbps": (
                            None
                            if active is None
                            else active["physical_service_gbps"]
                        ),
                        "aggregate_ssd_outstanding_gb": aggregate_outstanding,
                        "pressure_reports_cumulative": pressure[
                            "reports_cumulative"
                        ],
                        "pressure_cache_hits_cumulative": pressure[
                            "cache_hits_cumulative"
                        ],
                        "pressure_cache_time_ms": pressure["cache_time_ms"],
                        "pressure_cache_age_ms": pressure["cache_age_ms"],
                        "pressure_ttl_ms": pressure["ttl_ms"],
                    }
                    if source is not None:
                        row.update(source)
                    rows.append(row)
    return rows


def _build_route_probe_summary(route_rows: Sequence[dict]) -> list[dict]:
    summary: list[dict] = []
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            rows = [
                row
                for row in route_rows
                if int(row["num_ssu"]) == num_ssu and row["case"] == case
            ]
            _require(rows, f"SSU{num_ssu}/{case}: route rows missing")
            sources: dict[str, int] = defaultdict(int)
            path_counts: dict[int, int] = defaultdict(int)
            group_counts: dict[int, int] = defaultdict(int)
            for row in rows:
                source = row.get("pressure_source")
                sources["none" if source is None else str(source)] += 1
                for value in json.loads(str(row["selected_path_ids"])):
                    path_counts[int(value)] += 1
                for value in json.loads(str(row["selected_group_ids"])):
                    group_counts[int(value)] += 1
            summary.append(
                {
                    "num_ssu": num_ssu,
                    "case": case,
                    "probe_record_count": len(rows),
                    "planned_block_count": sum(path_counts.values()),
                    "pressure_source_counts": dict(sorted(sources.items())),
                    "selected_path_counts": dict(sorted(path_counts.items())),
                    "selected_group_counts": dict(sorted(group_counts.items())),
                    "portable_replay_all_exact": all(
                        bool(row["portable_replay_exact_match"]) for row in rows
                    ),
                    "probe_stream_truncated": any(
                        bool(row["probe_stream_truncated"]) for row in rows
                    ),
                    "ttft_deadline_or_slack_used_for_route": False,
                }
            )
    return summary


def _select_forensic_requests(matched_rows: Sequence[dict]) -> list[dict]:
    selected: list[dict] = []
    field = "adaptive_t0_i100ms_barrier_delta_vs_baseline_ms"
    for num_ssu in SSU_COUNTS:
        candidates = [
            row
            for row in matched_rows
            if int(row["num_ssu"]) == num_ssu
            and all(
                float(row[f"{case}_admission_relative_s"]) >= -TOL
                and float(row[f"{case}_completion_relative_s"])
                <= MEASUREMENT_MS / 1000.0 + TOL
                for case in CASES
            )
        ]
        _require(candidates, f"SSU{num_ssu}: no matched cohort for forensic zoom")
        absolute_deltas = [abs(float(row[field])) for row in candidates]
        target = float(np.median(absolute_deltas))
        chosen = min(
            candidates,
            key=lambda row: (
                abs(abs(float(row[field])) - target),
                int(row["npu_id"]),
                int(row["sequence"]),
            ),
        )
        selected.append(
            {
                **chosen,
                "selection_rule": (
                    "among requests common to all three policies, choose abs(baseline-to-Adaptive IO-barrier delta) closest to the SSU-specific median; ties by npu_id then sequence"
                ),
                "absolute_barrier_delta_target_median_ms": target,
                "absolute_barrier_delta_distance_from_median_ms": abs(
                    abs(float(chosen[field])) - target
                ),
                "matched_cohort_evidence_only": True,
                "full_request_lifecycle_inside_64s_timeline_all_policies": True,
            }
        )
    return selected


def _build_forensic_timeline_rows(
    selections: Sequence[Mapping[str, object]],
    timeline_rows: Sequence[dict],
) -> list[dict]:
    """Bind deterministic forensic requests to aggregate 500-ms NPU evidence.

    SSD/link counters are deliberately *not* request-attributed: cross-request
    layer-0 prefetch can overlap the active request.  Only manifest demand is
    exposed as request-attributable, and only where the controller request ID
    exactly matches the selected request.
    """
    rows: list[dict] = []
    for selection in selections:
        num_ssu = int(selection["num_ssu"])
        npu_id = int(selection["npu_id"])
        sequence = int(selection["sequence"])
        for case in CASES:
            selected_request_id = int(selection[f"{case}_request_id"])
            admission_s = float(selection[f"{case}_admission_relative_s"])
            completion_s = float(selection[f"{case}_completion_relative_s"])
            selected = [
                row
                for row in timeline_rows
                if int(row["num_ssu"]) == num_ssu
                and row["case"] == case
                and int(row["npu_id"]) == npu_id
                and float(row["relative_end_s"]) > admission_s - 0.5 - TOL
                and float(row["relative_start_s"]) < completion_s + 0.5 + TOL
            ]
            _require(
                selected,
                f"SSU{num_ssu}/{case}: forensic timeline binding is empty",
            )
            for source in selected:
                active_matches = (
                    source.get("active_request_id") is not None
                    and int(source["active_request_id"]) == selected_request_id
                )
                controller_matches = (
                    source.get("controller_request_id") is not None
                    and int(source["controller_request_id"])
                    == selected_request_id
                )
                command_matches = (
                    source.get("left_boundary_ssd_command_request_id") is not None
                    and int(source["left_boundary_ssd_command_request_id"])
                    == selected_request_id
                )
                rows.append(
                    {
                        "num_ssu": num_ssu,
                        "case": case,
                        "npu_id": npu_id,
                        "sequence": sequence,
                        "selected_request_id": selected_request_id,
                        "block": int(source["block"]),
                        "relative_start_s": float(source["relative_start_s"]),
                        "relative_end_s": float(source["relative_end_s"]),
                        "ssu_id": int(source["ssu_id"]),
                        "active_request_id_at_left_boundary": source.get(
                            "active_request_id"
                        ),
                        "controller_request_id_at_left_boundary": source.get(
                            "controller_request_id"
                        ),
                        "left_boundary_ssd_command_request_id": source.get(
                            "left_boundary_ssd_command_request_id"
                        ),
                        "active_request_matches_selected": active_matches,
                        "controller_request_matches_selected": controller_matches,
                        "left_boundary_ssd_command_request_matches_selected": (
                            command_matches
                        ),
                        "selected_request_manifest_demand_gbps_if_controller_match": (
                            source[
                                "manifest_demand_gbps_adaptive_input_otherwise_diagnostic"
                            ]
                            if controller_matches
                            else None
                        ),
                        "installed_cir_gbps_npu_ssu_not_request_attributed": source[
                            "installed_cir_gbps_not_actual"
                        ],
                        "interval_average_attributed_ssd_service_gbps_npu_aggregate_not_request_attributed": source[
                            "interval_average_attributed_ssd_service_gbps"
                        ],
                        "npu_link_delivered_gbps_npu_aggregate_not_request_attributed": source[
                            "npu_link_delivered_gbps_actual_received"
                        ],
                        "deadline_slack_alpha1p5_ms_if_active_request_matches_diagnostic_only": (
                            source[
                                "deadline_slack_alpha1p5_ms_diagnostic_only"
                            ]
                            if active_matches
                            else None
                        ),
                        "deadline_slack_alpha2_ms_if_active_request_matches_diagnostic_only": (
                            source["deadline_slack_alpha2_ms_diagnostic_only"]
                            if active_matches
                            else None
                        ),
                        "boundary_state_time_semantics": (
                            "demand/CIR/slack are left-boundary state at relative_start_s"
                        ),
                        "interval_bandwidth_time_semantics": (
                            "SSD/link are NPU-aggregate interval averages over [relative_start_s,relative_end_s); not request-attributed"
                        ),
                    }
                )
    return rows


def _plot_forensic_zoom(
    selection: Mapping[str, object],
    matched_layer_rows: Sequence[dict],
    forensic_timeline_rows: Sequence[dict],
    output: Path,
    *,
    compact: bool,
) -> None:
    num_ssu = int(selection["num_ssu"])
    npu_id = int(selection["npu_id"])
    sequence = int(selection["sequence"])
    fig, axes = plt.subplots(
        6,
        len(CASES),
        figsize=(12.0, 12.0) if compact else (20.0, 18.0),
        squeeze=False,
    )
    ssu_colors = plt.get_cmap("tab10").colors
    metric_rows = (
        (
            "selected_request_manifest_demand_gbps_if_controller_match",
            "Selected-request manifest demand\n(GB/s; ID-matched only)",
            False,
        ),
        (
            "installed_cir_gbps_npu_ssu_not_request_attributed",
            "Installed CIR (GB/s; NPU/SSU, not request)",
            False,
        ),
        (
            "interval_average_attributed_ssd_service_gbps_npu_aggregate_not_request_attributed",
            "Interval SSD service (GB/s; NPU aggregate)",
            True,
        ),
        (
            "npu_link_delivered_gbps_npu_aggregate_not_request_attributed",
            "NPU-link delivery (GB/s; NPU aggregate)",
            True,
        ),
    )
    for column, case in enumerate(CASES):
        layers = sorted(
            (
                row
                for row in matched_layer_rows
                if int(row["num_ssu"]) == num_ssu
                and row["case"] == case
                and int(row["npu_id"]) == npu_id
                and int(row["sequence"]) == sequence
            ),
            key=lambda row: int(row["layer"]),
        )
        _require(len(layers) == N_LAYERS, f"SSU{num_ssu}/{case}: forensic layer rows incomplete")
        gantt = axes[0, column]
        for row in layers:
            layer = int(row["layer"])
            compute_start = float(row["compute_start_relative_to_admission_ms"])
            compute_end = float(row["compute_end_relative_to_admission_ms"])
            wait = float(row["io_barrier_wait_ms"])
            if wait > 0.0:
                gantt.barh(
                    layer,
                    wait,
                    left=compute_start - wait,
                    height=0.72,
                    color="#f58518",
                    label="I/O barrier" if layer == 0 else None,
                )
            gantt.barh(
                layer,
                compute_end - compute_start,
                left=compute_start,
                height=0.72,
                color="#4c78a8",
                label="compute" if layer == 0 else None,
            )
        gantt.set_yticks(range(N_LAYERS))
        gantt.invert_yaxis()
        gantt.set_xlabel("Time from request admission (ms)")
        gantt.set_title(
            f"{CASE_LABELS[case]}\nTTFT={float(selection[f'{case}_ttft_ms']):.1f} ms"
        )
        gantt.grid(True, axis="x")

        admission_s = float(selection[f"{case}_admission_relative_s"])
        completion_s = float(selection[f"{case}_completion_relative_s"])
        request_timeline = [
            row
            for row in forensic_timeline_rows
            if int(row["num_ssu"]) == num_ssu
            and row["case"] == case
            and int(row["npu_id"]) == npu_id
            and int(row["sequence"]) == sequence
        ]
        _require(request_timeline, f"SSU{num_ssu}/{case}: forensic 500-ms timeline empty")
        for metric_index, (field, ylabel, is_interval_average) in enumerate(
            metric_rows, start=1
        ):
            axis = axes[metric_index, column]
            has_values = False
            for ssu_id in range(num_ssu):
                values_by_block = sorted(
                    (
                        row
                        for row in request_timeline
                        if int(row["ssu_id"]) == ssu_id
                    ),
                    key=lambda row: int(row["block"]),
                )
                values = [row[field] for row in values_by_block]
                if all(value is None for value in values):
                    continue
                has_values = True
                axis.step(
                    [
                        (
                            (
                                0.5
                                * (
                                    float(row["relative_start_s"])
                                    + float(row["relative_end_s"])
                                )
                                if is_interval_average
                                else float(row["relative_start_s"])
                            )
                            - admission_s
                        )
                        for row in values_by_block
                    ],
                    [math.nan if value is None else float(value) for value in values],
                    where="mid" if is_interval_average else "post",
                    color=ssu_colors[ssu_id % len(ssu_colors)],
                    linewidth=1.3,
                    label=f"SSU {ssu_id}",
                )
            if not has_values:
                axis.text(
                    0.5,
                    0.5,
                    "N/A for this policy",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            axis.axvline(0.0, color="black", linewidth=0.7)
            axis.axvline(
                completion_s - admission_s,
                color="black",
                linestyle="--",
                linewidth=0.7,
            )
            axis.grid(True)
            axis.set_xlabel(
                "Time from admission (s; interval midpoint)"
                if is_interval_average
                else "Time from admission (s; left-boundary state)"
            )
            if column == 0:
                axis.set_ylabel(ylabel)

        slack_axis = axes[5, column]
        slack_rows = sorted(
            (
                row
                for row in request_timeline
                if int(row["ssu_id"]) == 0
            ),
            key=lambda row: int(row["block"]),
        )
        x_values = [
            float(row["relative_start_s"]) - admission_s
            for row in slack_rows
        ]
        for field, label, color in (
            (
                "deadline_slack_alpha1p5_ms_if_active_request_matches_diagnostic_only",
                "alpha=1.5 slack",
                "#e45756",
            ),
            (
                "deadline_slack_alpha2_ms_if_active_request_matches_diagnostic_only",
                "alpha=2 slack",
                "#54a24b",
            ),
        ):
            slack_axis.step(
                x_values,
                [math.nan if row[field] is None else float(row[field]) for row in slack_rows],
                where="post",
                linewidth=1.1,
                color=color,
                label=label,
            )
        slack_axis.axhline(0.0, color="black", linewidth=0.7)
        slack_axis.axvline(0.0, color="black", linewidth=0.7)
        slack_axis.grid(True)
        slack_axis.set_xlabel("Time from request admission (s; diagnostic only)")
        if column == 0:
            slack_axis.set_ylabel("TTFT deadline slack (ms)\ndiagnostic only")

    axes[0, 0].legend(fontsize=7, loc="best")
    axes[1, 0].legend(fontsize=6, ncol=2, loc="best")
    axes[5, 0].legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Deterministic forensic zoom — SSU{num_ssu}, NPU{npu_id}, sequence {sequence}",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Selection: full-lifecycle common-cohort |baseline-to-Adaptive barrier delta| closest to its SSU median (deterministic ties). Demand/CIR/slack are left-boundary states; SSD/link are 500-ms interval averages at interval midpoint. Demand/slack are shown only when request IDs match. SSD/link remain NPU-aggregate, not request-attributed (cross-request prefetch may overlap). Slack is diagnostic-only and never an input.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97))
    _save_figure(
        fig,
        output,
        "Deterministically selected matched-request 16-layer Gantt and 500-ms per-NPU/SSU forensic bandwidth/slack zoom",
        compact=compact,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.45,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.fancybox": True,
        }
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _require_source_bytes_unchanged(
    path: Path, expected_sha256: str, context: str
) -> None:
    _require(
        _sha256(path) == expected_sha256,
        f"{context}: analyzer source bytes changed during analysis",
    )


def _save_figure(
    fig: plt.Figure,
    path: Path,
    description: str,
    *,
    compact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=55 if compact else 150,
        facecolor="white",
        metadata={
            "Title": path.stem,
            "Description": description,
        },
    )
    plt.close(fig)
    temporary.replace(path)


def _timeline_arrays(
    rows: Sequence[dict], num_ssu: int, case: str
) -> dict[str, np.ndarray]:
    selected = [row for row in rows if row["num_ssu"] == num_ssu and row["case"] == case]
    lookup = {
        (int(row["block"]), int(row["npu_id"]), int(row["ssu_id"])): row
        for row in selected
    }
    _require(len(lookup) == BLOCK_COUNT * NUM_NPU * num_ssu, f"SSU{num_ssu}/{case}: timeline cell table incomplete")
    cell_fields = (
        "manifest_demand_gbps_adaptive_input_otherwise_diagnostic",
        "physical_remaining_demand_gbps_diagnostic_only",
        "interval_average_attributed_ssd_service_gbps",
        "npu_link_delivered_gbps_actual_received",
        "link_minus_controller_demand_gbps",
        "interval_average_ssd_service_minus_link_delivery_gbps",
    )
    arrays = {
        field: np.array(
            [
                [
                    float(lookup[(block, npu, ssu)][field])
                    for block in range(BLOCK_COUNT)
                ]
                for npu in range(NUM_NPU)
                for ssu in range(num_ssu)
            ],
            dtype=float,
        )
        for field in cell_fields
    }
    arrays["installed_cir_gbps_not_actual"] = np.array(
        [
            [
                (
                    math.nan
                    if lookup[(block, npu, ssu)][
                        "installed_cir_gbps_not_actual"
                    ]
                    is None
                    else float(
                        lookup[(block, npu, ssu)][
                            "installed_cir_gbps_not_actual"
                        ]
                    )
                )
                for block in range(BLOCK_COUNT)
            ]
            for npu in range(NUM_NPU)
            for ssu in range(num_ssu)
        ],
        dtype=float,
    )
    arrays["npu_utilization"] = np.array(
        [
            [float(lookup[(block, npu, 0)]["npu_utilization"]) for block in range(BLOCK_COUNT)]
            for npu in range(NUM_NPU)
        ],
        dtype=float,
    )
    return arrays


def _plot_timeline_heatmap(
    timeline_rows: Sequence[dict], num_ssu: int, output: Path, *, compact: bool
) -> None:
    arrays = {case: _timeline_arrays(timeline_rows, num_ssu, case) for case in CASES}
    cell_count = NUM_NPU * num_ssu
    metric_specs = (
        (
            "npu_utilization",
            "NPU compute utilization",
            "viridis",
            Normalize(0.0, 1.0),
            [f"N{npu:02d}" for npu in range(NUM_NPU)],
        ),
        (
            "manifest_demand_gbps_adaptive_input_otherwise_diagnostic",
            "Manifest demand (GB/s; Adaptive input, otherwise diagnostic)",
            "magma",
            Normalize(
                0.0,
                max(
                    1e-12,
                    max(float(np.max(arrays[case]["manifest_demand_gbps_adaptive_input_otherwise_diagnostic"])) for case in CASES),
                ),
            ),
            [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
        ),
        (
            "physical_remaining_demand_gbps_diagnostic_only",
            "Physical remaining demand (GB/s; diagnostic)",
            "magma",
            Normalize(
                0.0,
                max(
                    1e-12,
                    max(
                        float(
                            np.max(
                                arrays[case][
                                    "physical_remaining_demand_gbps_diagnostic_only"
                                ]
                            )
                        )
                        for case in CASES
                    ),
                ),
            ),
            [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
        ),
        (
            "installed_cir_gbps_not_actual",
            "Installed CIR (GB/s; N/A for non-Adaptive; not actual)",
            "YlGnBu",
            Normalize(
                0.0,
                max(
                    1e-12,
                    float(
                        np.nanmax(
                            arrays["adaptive_t0_i100ms"][
                                "installed_cir_gbps_not_actual"
                            ]
                        )
                    ),
                ),
            ),
            [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
        ),
        (
            "interval_average_attributed_ssd_service_gbps",
            "Interval-average attributed SSD service (GB/s)",
            "cividis",
            Normalize(0.0, SSD_CAP_GBPS),
            [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
        ),
        (
            "npu_link_delivered_gbps_actual_received",
            "NPU-link delivered: actual received (GB/s)",
            "cividis",
            Normalize(0.0, NPU_LINK_CAP_GBPS),
            [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
        ),
    )
    gap_fields = (
        "link_minus_controller_demand_gbps",
        "interval_average_ssd_service_minus_link_delivery_gbps",
    )
    gap_specs = []
    for field, title in (
        ("link_minus_controller_demand_gbps", "Actual received - manifest demand (GB/s)"),
        (
            "interval_average_ssd_service_minus_link_delivery_gbps",
            "Interval-average SSD service - link delivery (GB/s)",
        ),
    ):
        limit = max(
            1e-12,
            max(float(np.max(np.abs(arrays[case][field]))) for case in CASES),
        )
        gap_specs.append(
            (
                field,
                title,
                "coolwarm",
                TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                [f"N{npu:02d}/S{ssu}" for npu in range(NUM_NPU) for ssu in range(num_ssu)],
            )
        )
    specs = metric_specs + tuple(gap_specs)
    height_ratios = [NUM_NPU] + [cell_count] * (len(specs) - 1)
    if compact:
        figsize = (13.0, 13.0)
    else:
        figsize = (25.0, min(43.0, 8.0 + 0.045 * sum(height_ratios)))
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        len(specs),
        len(CASES),
        height_ratios=height_ratios,
        hspace=0.16,
        wspace=0.08,
        left=0.10 if not compact else 0.08,
        right=0.92,
        bottom=0.05,
        top=0.965,
    )
    axes_by_metric: list[list[plt.Axes]] = []
    for metric_index, (field, title, cmap, norm, labels) in enumerate(specs):
        row_axes: list[plt.Axes] = []
        image = None
        for case_index, case in enumerate(CASES):
            axis = fig.add_subplot(grid[metric_index, case_index])
            row_axes.append(axis)
            values = arrays[case][field]
            plotted_cmap = plt.get_cmap(cmap).copy()
            plotted_cmap.set_bad("#dedede")
            image = axis.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap=plotted_cmap,
                norm=norm,
                extent=(0.0, 64.0, values.shape[0], 0.0),
                rasterized=True,
            )
            if np.all(np.isnan(values)):
                axis.text(
                    0.5,
                    0.5,
                    "N/A\n(no dedicated CIR in this policy)",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#333333",
                    bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
                )
            for horizon in (1.0, 2.0, 3.0, 8.0, 16.0, 32.0):
                axis.axvline(horizon, color="white", linewidth=0.35, alpha=0.50)
            if metric_index == 0:
                axis.set_title(CASE_LABELS[case], fontsize=11, pad=5)
            if case_index == 0:
                axis.set_ylabel(title, fontsize=8)
                if compact:
                    step = max(1, len(labels) // 16)
                    ticks = np.arange(0.5, len(labels), step)
                    shown = labels[::step]
                else:
                    ticks = np.arange(0.5, len(labels), 1.0)
                    shown = labels
                axis.set_yticks(ticks, shown, fontsize=3.2 if len(labels) > 32 else 5.2)
            else:
                axis.set_yticks([])
            if metric_index == len(specs) - 1:
                axis.set_xlabel("Time since measurement start (s)")
                axis.set_xticks([0, 1, 2, 3, 8, 16, 32, 48, 64])
            else:
                axis.set_xticklabels([])
        assert image is not None
        colorbar = fig.colorbar(image, ax=row_axes, fraction=0.012, pad=0.008)
        colorbar.ax.tick_params(labelsize=6)
        axes_by_metric.append(row_axes)
    fig.suptitle(
        f"32-NPU / {num_ssu}-SSU exact 500-ms timeline — shared scales across strategies",
        fontsize=14,
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Demand is left-boundary state; physical demand is diagnostic. CIR is N/A outside Adaptive and is never actual bandwidth. SSD/link are 500-ms interval averages from exact attributed-byte counter differences.",
        ha="center",
        fontsize=8,
    )
    _save_figure(
        fig,
        output,
        f"SSU{num_ssu} exact NPU-by-SSU timeline; interval-average SSD attribution and link delivery are distinct",
        compact=compact,
    )


def _plot_ssu_summary_metric(
    window_rows: Sequence[dict],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
    *,
    compact: bool,
) -> None:
    whole = {
        (int(row["num_ssu"]), str(row["case"])): row
        for row in window_rows
        if row["window_kind"] == "cumulative"
        and _close(float(row["requested_duration_s"]), 64.0)
    }
    _require(
        len(whole) == len(SSU_COUNTS) * len(CASES),
        "whole-window SSU summary matrix is incomplete",
    )
    fig, axis = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for case in CASES:
        values = [float(whole[(num_ssu, case)][metric]) * 100.0 for num_ssu in SSU_COUNTS]
        axis.plot(
            SSU_COUNTS,
            values,
            label=CASE_LABELS[case],
            color=CASE_COLORS[case],
            marker=CASE_MARKERS[case],
            linewidth=2.2,
            markersize=7.0,
        )
    axis.set_xticks(SSU_COUNTS)
    axis.set_xlabel("Number of SSUs")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_ylim(0.0, 100.0)
    axis.grid(True, color="#d9d9d9", linewidth=0.75, alpha=0.8)
    axis.legend(frameon=True, ncol=1, loc="lower right")
    _save_figure(
        fig,
        output,
        title + "; 64-second whole-window, equal-observed-NPU SLO where applicable",
        compact=compact,
    )


def _plot_cumulative_metrics(
    window_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    rows = [row for row in window_rows if row["window_kind"] == "cumulative"]
    metrics = (
        ("npu_utilization", "Average NPU utilization (%)"),
        ("equal_observed_npu_slo_alpha1p5", "Equal-NPU TTFT SLO, alpha=1.5 (%)"),
        ("equal_observed_npu_slo_alpha2", "Equal-NPU TTFT SLO, alpha=2 (%)"),
    )
    fig, axes = plt.subplots(
        len(metrics),
        len(SSU_COUNTS),
        figsize=(12.0, 8.0) if compact else (17.0, 12.0),
        sharex=True,
        squeeze=False,
    )
    for metric_index, (field, ylabel) in enumerate(metrics):
        for ssu_index, num_ssu in enumerate(SSU_COUNTS):
            axis = axes[metric_index, ssu_index]
            for case in CASES:
                selected = sorted(
                    (row for row in rows if row["num_ssu"] == num_ssu and row["case"] == case),
                    key=lambda row: float(row["requested_duration_s"]),
                )
                axis.plot(
                    [float(row["requested_duration_s"]) for row in selected],
                    [math.nan if row[field] is None else 100.0 * float(row[field]) for row in selected],
                    label=CASE_LABELS[case],
                    color=CASE_COLORS[case],
                    marker=CASE_MARKERS[case],
                    linewidth=2.0,
                    markersize=5.0,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(HORIZONS_S, [str(value).rstrip("0").rstrip(".") for value in HORIZONS_S])
            axis.set_ylim(0.0, 100.0)
            axis.grid(True)
            if metric_index == 0:
                axis.set_title(f"{num_ssu} SSUs")
            if ssu_index == 0:
                axis.set_ylabel(ylabel)
            if metric_index == len(metrics) - 1:
                axis.set_xlabel("Cumulative horizon from measurement start (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("64-second cumulative-window behavior (500-ms exact boundaries)", y=1.01, fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Utilization is the exact resource-time integral in each horizon. SLO cohorts are selected by admission in the half-open horizon; their TTFT may complete later.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    _save_figure(fig, output, "Cumulative NPU utilization and TTFT SLO at 0.5-64 second horizons", compact=compact)


def _plot_disjoint_variability(
    window_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    rows = [
        row
        for row in window_rows
        if row["window_kind"] == "disjoint" and row["full_requested_window"]
    ]
    metrics = (
        ("npu_utilization", "NPU utilization (%)"),
        ("equal_observed_npu_slo_alpha1p5", "Equal-NPU TTFT SLO alpha=1.5 (%)"),
        ("equal_observed_npu_slo_alpha2", "Equal-NPU TTFT SLO alpha=2 (%)"),
    )
    fig, axes = plt.subplots(
        len(metrics),
        len(SSU_COUNTS),
        figsize=(12.0, 8.0) if compact else (17.0, 12.0),
        sharex=True,
        squeeze=False,
    )
    for metric_index, (field, ylabel) in enumerate(metrics):
        for ssu_index, num_ssu in enumerate(SSU_COUNTS):
            axis = axes[metric_index, ssu_index]
            for case in CASES:
                centers, lows, highs = [], [], []
                for horizon in HORIZONS_S:
                    values = [
                        100.0 * float(row[field])
                        for row in rows
                        if row["num_ssu"] == num_ssu
                        and row["case"] == case
                        and row["requested_duration_s"] == horizon
                        and row[field] is not None
                    ]
                    centers.append(float(np.median(values)) if values else math.nan)
                    lows.append(min(values) if values else math.nan)
                    highs.append(max(values) if values else math.nan)
                center_array = np.asarray(centers)
                axis.errorbar(
                    HORIZONS_S,
                    center_array,
                    yerr=np.vstack((center_array - np.asarray(lows), np.asarray(highs) - center_array)),
                    label=CASE_LABELS[case],
                    color=CASE_COLORS[case],
                    marker=CASE_MARKERS[case],
                    linewidth=1.8,
                    capsize=2.5,
                    markersize=5.0,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(HORIZONS_S, [str(value).rstrip("0").rstrip(".") for value in HORIZONS_S])
            axis.set_ylim(0.0, 100.0)
            axis.grid(True)
            if metric_index == 0:
                axis.set_title(f"{num_ssu} SSUs")
            if ssu_index == 0:
                axis.set_ylabel(ylabel)
            if metric_index == len(metrics) - 1:
                axis.set_xlabel("Disjoint full-window width (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Window effect: median and exact min-max across disjoint windows", y=1.01, fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Utilization is integrated inside each disjoint window. SLO is for requests admitted in that half-open window, even if TTFT completes after its end.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    _save_figure(fig, output, "Disjoint-window variability for NPU utilization and TTFT SLO", compact=compact)


def _plot_adaptive_causal_timeline(
    decision_rows: Sequence[dict],
    boundary_slack_rows: Sequence[dict],
    timeline_rows: Sequence[dict],
    grant_component_rows: Sequence[dict],
    output: Path,
    *,
    compact: bool,
) -> None:
    modes = sorted({str(row["selection_mode"]) for row in decision_rows})
    mode_id = {mode: index for index, mode in enumerate(modes)}
    fig, axes = plt.subplots(
        6,
        len(SSU_COUNTS),
        figsize=(13.0, 13.0) if compact else (19.0, 19.0),
        sharex="col",
        squeeze=False,
    )
    for column, num_ssu in enumerate(SSU_COUNTS):
        decisions = sorted(
            (row for row in decision_rows if row["num_ssu"] == num_ssu),
            key=lambda row: float(row["relative_time_s"]),
        )
        times = [float(row["relative_time_s"]) for row in decisions]
        axes[0, column].step(times, [int(row["active_npu_count"]) for row in decisions], where="post", label="Active", color="#7f7f7f", linewidth=1.3)
        axes[0, column].step(times, [int(row["selected_npu_count"]) for row in decisions], where="post", label="Selected", color=CASE_COLORS["adaptive_t0_i100ms"], linewidth=2.0)
        axes[0, column].set_ylim(-0.5, NUM_NPU + 0.5)
        axes[0, column].set_title(f"Adaptive 100 ms — {num_ssu} SSUs")
        axes[0, column].grid(True)

        axes[1, column].scatter(
            times,
            [mode_id[str(row["selection_mode"])] for row in decisions],
            c=[int(row["changed_cir_entry_count"]) for row in decisions],
            cmap="viridis",
            s=8 if compact else 13,
            alpha=0.85,
        )
        axes[1, column].set_yticks(range(len(modes)), modes, fontsize=7)
        axes[1, column].grid(True, axis="x")

        adaptive_timeline = [
            row
            for row in timeline_rows
            if int(row["num_ssu"]) == num_ssu
            and row["case"] == "adaptive_t0_i100ms"
        ]
        left_boundary_times = np.arange(BLOCK_COUNT, dtype=float) * 0.5
        interval_times = left_boundary_times + 0.25
        ssu_colors = plt.get_cmap("tab10").colors
        for ssu_id in range(num_ssu):
            for field, label, linestyle in (
                (
                    "manifest_demand_gbps_adaptive_input_otherwise_diagnostic",
                    "manifest demand (controller input)",
                    ":",
                ),
                ("installed_cir_gbps_not_actual", "installed CIR", "-"),
            ):
                values = [
                    math.fsum(
                        float(row[field])
                        for row in adaptive_timeline
                        if int(row["block"]) == block
                        and int(row["ssu_id"]) == ssu_id
                    )
                    for block in range(BLOCK_COUNT)
                ]
                axes[2, column].step(
                    left_boundary_times,
                    values,
                    where="post",
                    color=ssu_colors[ssu_id % len(ssu_colors)],
                    linestyle=linestyle,
                    linewidth=1.0,
                    label=f"S{ssu_id} {label}",
                )
            link_values = [
                math.fsum(
                    float(row["npu_link_delivered_gbps_actual_received"])
                    for row in adaptive_timeline
                    if int(row["block"]) == block
                    and int(row["ssu_id"]) == ssu_id
                )
                for block in range(BLOCK_COUNT)
            ]
            axes[2, column].plot(
                interval_times,
                link_values,
                color=ssu_colors[ssu_id % len(ssu_colors)],
                linestyle="--",
                linewidth=1.0,
                label=f"S{ssu_id} link actual delivery (interval midpoint)",
            )
        axes[2, column].axhline(SSD_CAP_GBPS, color="black", linestyle="--", linewidth=0.8, label="40 GB/s cap")
        axes[2, column].grid(True)

        components = [
            row
            for row in grant_component_rows
            if int(row["num_ssu"]) == num_ssu
        ]
        component_times = sorted({float(row["relative_time_s"]) for row in components})
        component_fields = (
            ("floor_grant_gbps", "floor", "#4c78a8"),
            ("background_grant_gbps", "background", "#f58518"),
            ("selected_tail_grant_gbps", "selected tail", "#54a24b"),
            ("spill_tail_grant_gbps", "spill tail", "#e45756"),
            ("v1_coflow_grant_gbps", "V1 coflow grant", "#b279a2"),
        )
        component_values = [
            [
                math.fsum(
                    (0.0 if row[field] is None else float(row[field]))
                    for row in components
                    if float(row["relative_time_s"]) == time_s
                )
                for time_s in component_times
            ]
            for field, _label, _color in component_fields
        ]
        axes[3, column].stackplot(
            component_times,
            component_values,
            labels=[label for _field, label, _color in component_fields],
            colors=[color for _field, _label, color in component_fields],
            alpha=0.82,
            step="post",
        )
        final_values = [
            math.fsum(
                float(row["final_grant_gbps"])
                for row in components
                if float(row["relative_time_s"]) == time_s
            )
            for time_s in component_times
        ]
        axes[3, column].step(
            component_times,
            final_values,
            where="post",
            color="black",
            linewidth=1.0,
            label="final grant",
        )
        axes[3, column].grid(True)

        slack = [row for row in boundary_slack_rows if row["num_ssu"] == num_ssu]
        for alpha_index, alpha_suffix in enumerate(("alpha1p5", "alpha2"), start=4):
            boundary_values: dict[float, list[dict]] = defaultdict(list)
            for row in slack:
                boundary_values[float(row["relative_time_s"])].append(row)
            boundary_times = sorted(boundary_values)
            for selected, color, label_prefix in (
                (True, CASE_COLORS["adaptive_t0_i100ms"], "Selected"),
                (False, "#7f7f7f", "Not selected"),
            ):
                deadline_values = []
                after_compute_values = []
                for time_s in boundary_times:
                    group = [
                        row
                        for row in boundary_values[time_s]
                        if row["selected_by_latest_strictly_prior_decision"]
                        is selected
                    ]
                    deadline = [
                        float(row[f"deadline_slack_{alpha_suffix}_ms_diagnostic_only"])
                        for row in group
                        if row[f"deadline_slack_{alpha_suffix}_ms_diagnostic_only"] is not None
                    ]
                    after_compute = [
                        float(row[f"slack_after_remaining_compute_{alpha_suffix}_ms_diagnostic_only"])
                        for row in group
                        if row[f"slack_after_remaining_compute_{alpha_suffix}_ms_diagnostic_only"] is not None
                    ]
                    deadline_values.append(float(np.median(deadline)) if deadline else math.nan)
                    after_compute_values.append(float(np.median(after_compute)) if after_compute else math.nan)
                axes[alpha_index, column].plot(
                    boundary_times,
                    deadline_values,
                    color=color,
                    linewidth=1.5,
                    label=f"{label_prefix}: deadline slack",
                )
                axes[alpha_index, column].plot(
                    boundary_times,
                    after_compute_values,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{label_prefix}: slack after Q",
                )
            axes[alpha_index, column].axhline(0.0, color="black", linewidth=0.8)
            axes[alpha_index, column].grid(True)
            axes[alpha_index, column].set_xlabel("Time since measurement start (s)")

        if column == 0:
            axes[0, column].set_ylabel("NPU count")
            axes[1, column].set_ylabel("Selection mode\n(color = CIR writes)")
            axes[2, column].set_ylabel("Per-SSU fleet totals (GB/s)\ndemand / CIR / actual")
            axes[3, column].set_ylabel("Fleet grant components (GB/s)\nV2 stages or V1 coflow; sum=final")
            axes[4, column].set_ylabel("alpha=1.5 slack (ms)\ndiagnostic only")
            axes[5, column].set_ylabel("alpha=2 slack (ms)\ndiagnostic only")
        axes[0, column].legend(fontsize=7, loc="lower right")
        axes[2, column].legend(fontsize=5.2, ncol=2, loc="upper right")
        axes[3, column].legend(fontsize=6, ncol=2, loc="best")
        axes[4, column].legend(fontsize=6, ncol=2, loc="best")
        axes[5, column].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(
        "Adaptive causal decision timeline — slack is an after-the-fact diagnostic, never a controller input",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "required_ratio=0.5 is the alpha=2 feasibility configuration, not a wall-clock TTFT deadline check. Demand/CIR are left-boundary step states; link is a 500-ms interval average at interval midpoint. Dashed slack subtracts exact boundary Q(t).",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.021,
        "Prefetch-only / not-yet-admitted controller requests are excluded: elapsed TTFT and both slack definitions are NA.",
        ha="center",
        fontsize=8,
    )
    fig.text(
        0.5,
        0.034,
        "A boundary is classified selected/rejected only when its active/controller request exactly matches the latest strictly-prior decision request; request-turnover gaps are unknown/stale and excluded.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.98))
    _save_figure(fig, output, "Adaptive selected count, exact causal admission and grant components, demand/CIR/link delivery, and diagnostic-only TTFT slack", compact=compact)


def _plot_dispatch_probes(
    probe_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    waits = [float(row["winner_queue_wait_ms"]) for row in probe_rows]
    color_max = max(1e-9, float(np.percentile(waits, 99)))
    norm = Normalize(0.0, color_max)
    fig, axes = plt.subplots(
        len(SSU_COUNTS),
        len(CASES),
        figsize=(12.0, 8.0) if compact else (18.0, 12.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    scatter = None
    for row_index, num_ssu in enumerate(SSU_COUNTS):
        for column, case in enumerate(CASES):
            axis = axes[row_index, column]
            selected = [row for row in probe_rows if row["num_ssu"] == num_ssu and row["case"] == case]
            scatter = axis.scatter(
                [float(row["relative_time_ms"]) for row in selected],
                [int(row["winner_npu_id"]) for row in selected],
                c=[float(row["winner_queue_wait_ms"]) for row in selected],
                norm=norm,
                cmap="plasma",
                s=[8.0 + 2.0 * min(20, int(row["candidate_path_count"])) for row in selected],
                alpha=0.75,
                edgecolors="black",
                linewidths=0.15,
            )
            axis.grid(True)
            matches = sum(bool(row["prediction_matches_actual"]) for row in selected)
            if row_index == 0:
                axis.set_title(CASE_LABELS[case])
            if column == 0:
                axis.set_ylabel(f"{num_ssu} SSUs\nwinning NPU")
            if row_index == len(SSU_COUNTS) - 1:
                axis.set_xlabel("Time into first-50-ms capped dispatch prefix (ms)")
            axis.text(0.99, 0.98, f"runtime assertions {matches}/{len(selected)}", transform=axis.transAxes, ha="right", va="top", fontsize=7)
            axis.set_ylim(-1, NUM_NPU)
    assert scatter is not None
    colorbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.015, pad=0.015)
    colorbar.set_label("Winner queue wait (ms), shared scale clipped at global p99")
    fig.suptitle("Microscopic non-preemptive SSD dispatch: capped recorded prefix and runtime assertion", y=0.995, fontsize=14)
    fig.text(0.5, 0.01, "At most 10,000 records from the first 50 ms; cap-reached flag is in CSV. Marker size encodes candidate-path count.", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.07, top=0.95, hspace=0.16, wspace=0.10)
    _save_figure(fig, output, "Simulator pre-dispatch read-only predictions with runtime winner assertions", compact=compact)


def _plot_matched_barriers(
    matched_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    fig, axes = plt.subplots(1, len(SSU_COUNTS), figsize=(11.0, 3.5) if compact else (18.0, 5.5), sharey=True)
    for axis, num_ssu in zip(axes, SSU_COUNTS):
        selected = [row for row in matched_rows if row["num_ssu"] == num_ssu]
        for policy in CASES[1:]:
            values = np.sort([float(row[f"{policy}_barrier_delta_vs_baseline_ms"]) for row in selected])
            cdf = np.arange(1, len(values) + 1) / len(values)
            axis.plot(values, cdf, label=CASE_LABELS[policy], color=CASE_COLORS[policy], linewidth=2.0)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_title(f"{num_ssu} SSUs; n={len(selected)} common requests")
        axis.set_xlabel("Matched request IO-barrier delta vs baseline (ms)")
        axis.grid(True)
    axes[0].set_ylabel("Empirical CDF")
    axes[0].legend(loc="lower right")
    fig.suptitle("Matched-request barrier redistribution (negative is improvement)", y=1.02, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output, "Matched request IO-barrier deltas against baseline", compact=compact)


def _plot_state_duration_partition(
    state_summary: Sequence[dict], output: Path, *, compact: bool
) -> None:
    fig, axes = plt.subplots(
        1,
        len(SSU_COUNTS),
        figsize=(11.0, 3.8) if compact else (18.0, 6.2),
        sharey=True,
        squeeze=False,
    )
    colors = {
        "compute_fraction": "#4c78a8",
        "io_barrier_fraction": "#f58518",
        "other_fraction": "#bab0ac",
    }
    labels = {
        "compute_fraction": "Compute (NPU utilization)",
        "io_barrier_fraction": "I/O barrier idle",
        "other_fraction": "Other window complement",
    }
    for column, num_ssu in enumerate(SSU_COUNTS):
        axis = axes[0, column]
        rows = [row for row in state_summary if row["num_ssu"] == num_ssu]
        rows.sort(key=lambda row: CASES.index(str(row["case"])))
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for field in ("compute_fraction", "io_barrier_fraction", "other_fraction"):
            values = 100.0 * np.asarray([float(row[field]) for row in rows])
            axis.bar(
                x,
                values,
                bottom=bottom,
                color=colors[field],
                width=0.66,
                label=labels[field],
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += values
        for index, row in enumerate(rows):
            axis.text(
                index,
                100.8,
                f"barrier={100.0 * float(row['io_barrier_share_of_idle']):.1f}% of idle",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=20,
            )
        axis.set_xticks(
            x,
            [CASE_LABELS[str(row["case"])] for row in rows],
            rotation=14,
            ha="right",
        )
        axis.set_title(f"{num_ssu} SSUs")
        axis.set_ylim(0.0, 112.0)
        axis.grid(True, axis="y")
    axes[0, 0].set_ylabel("Exact share of 32 x 64-s NPU time (%)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        "Primary low-utilization evidence: exact compute / I/O-barrier / other partition (includes carry-in)",
        fontsize=14,
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(
        fig,
        output,
        "Exact full-window NPU state-duration partition including carry-in",
        compact=compact,
    )


def _plot_block_state_duration_timeline(
    block_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    fig, axes = plt.subplots(
        len(SSU_COUNTS),
        len(CASES),
        figsize=(12.0, 7.5) if compact else (19.0, 11.5),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = ("#4c78a8", "#f58518", "#bab0ac")
    labels = ("compute", "I/O barrier", "other")
    for row_index, num_ssu in enumerate(SSU_COUNTS):
        for column, case in enumerate(CASES):
            axis = axes[row_index, column]
            selected = [
                row
                for row in block_rows
                if int(row["num_ssu"]) == num_ssu and row["case"] == case
            ]
            lookup = {
                (int(row["block"]), int(row["npu_id"])): row
                for row in selected
            }
            _require(
                len(lookup) == BLOCK_COUNT * NUM_NPU,
                f"SSU{num_ssu}/{case}: block state plot table incomplete",
            )
            state = np.zeros((3, BLOCK_COUNT), dtype=float)
            compute_ms = np.zeros(BLOCK_COUNT, dtype=float)
            for block in range(BLOCK_COUNT):
                values = [lookup[(block, npu)] for npu in range(NUM_NPU)]
                for state_index, field in enumerate(
                    ("compute_fraction", "io_barrier_fraction", "other_fraction")
                ):
                    state[state_index, block] = statistics.fmean(
                        float(row[field]) for row in values
                    )
                compute_ms[block] = math.fsum(
                    float(row["compute_ms"]) for row in values
                )
            time_s = np.arange(BLOCK_COUNT, dtype=float) * 0.5 + 0.25
            axis.stackplot(
                time_s,
                100.0 * state,
                colors=colors,
                labels=labels,
                alpha=0.88,
                step="mid",
            )
            cumulative_compute = np.cumsum(compute_ms) / (
                NUM_NPU * BLOCK_MS * np.arange(1, BLOCK_COUNT + 1)
            )
            axis.plot(
                time_s,
                100.0 * cumulative_compute,
                color="black",
                linewidth=1.15,
                label="cumulative utilization",
            )
            for horizon in (1.0, 2.0, 3.0, 8.0, 16.0, 32.0):
                axis.axvline(
                    horizon, color="white", linewidth=0.55, alpha=0.75
                )
            if row_index == 0:
                axis.set_title(CASE_LABELS[case])
            if column == 0:
                axis.set_ylabel(f"{num_ssu} SSUs\nNPU-time share (%)")
            if row_index == len(SSU_COUNTS) - 1:
                axis.set_xlabel("Time in measurement window (s)")
            axis.set_xlim(0.0, 64.0)
            axis.set_ylim(0.0, 100.0)
            axis.grid(True, alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Exact 500-ms state timeline: instantaneous phase mix and cumulative utilization",
        fontsize=14,
        y=1.035,
    )
    fig.text(
        0.5,
        0.01,
        "Every colored column is an exact 32-NPU partition; vertical guides are the requested 1/2/3/8/16/32-s horizons.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.955))
    _save_figure(
        fig,
        output,
        "Exact 500-ms compute, IO-barrier and other state timeline",
        compact=compact,
    )


def _plot_layer_once_route_causality(
    route_rows: Sequence[dict],
    timeline_rows: Sequence[dict],
    output: Path,
    *,
    compact: bool,
) -> None:
    fig, axes = plt.subplots(
        3,
        len(SSU_COUNTS),
        figsize=(11.5, 8.5) if compact else (18.0, 13.0),
        squeeze=False,
    )
    source_style = {
        "fresh": ("#4c78a8", "o"),
        "cache": ("#f58518", "x"),
    }
    layer_timeline = [
        row
        for row in timeline_rows
        if row["case"] == "layer_once_ttl_5ms"
    ]
    group_matrices: dict[int, np.ndarray] = {}
    for num_ssu in SSU_COUNTS:
        matrix = np.zeros(
            (num_ssu * EXPECTED_PATH_ABI["group_count"], BLOCK_COUNT),
            dtype=float,
        )
        for row in layer_timeline:
            if int(row["num_ssu"]) != num_ssu:
                continue
            counts = json.loads(str(row["route_blocks_by_group_in_block"]))
            for group_id, count in enumerate(counts):
                matrix[
                    int(row["ssu_id"]) * EXPECTED_PATH_ABI["group_count"]
                    + group_id,
                    int(row["block"]),
                ] += int(count)
        group_matrices[num_ssu] = matrix
    group_norm = Normalize(
        0.0,
        max(1.0, max(float(np.max(matrix)) for matrix in group_matrices.values())),
    )
    group_image = None
    for column, num_ssu in enumerate(SSU_COUNTS):
        group_axis = axes[0, column]
        matrix = group_matrices[num_ssu]
        group_image = group_axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, 64.0, matrix.shape[0], 0.0),
            cmap="Blues",
            norm=group_norm,
            rasterized=True,
        )
        group_ticks = [
            ssu_id * EXPECTED_PATH_ABI["group_count"] + group_id + 0.5
            for ssu_id in range(num_ssu)
            for group_id in range(EXPECTED_PATH_ABI["group_count"])
        ]
        group_axis.set_yticks(
            group_ticks,
            [
                f"S{ssu_id}/G{group_id}"
                for ssu_id in range(num_ssu)
                for group_id in range(EXPECTED_PATH_ABI["group_count"])
            ],
            fontsize=5.5,
        )
        group_axis.set_title(f"{num_ssu} SSUs")
        group_axis.set_xlabel("Time in 64-s measurement (s; 500-ms bins)")

        counter_axis = axes[1, column]
        selected_timeline = [
            row
            for row in layer_timeline
            if int(row["num_ssu"]) == num_ssu
        ]
        time_s = np.arange(BLOCK_COUNT, dtype=float) * 0.5 + 0.25
        for field, label, color in (
            ("fresh_pressure_reads_in_block", "fresh pressure reads", "#4c78a8"),
            ("pressure_cache_hits_in_block", "TTL cache hits", "#f58518"),
            ("route_plans_in_block", "route plans", "#54a24b"),
        ):
            values = [
                sum(
                    int(row[field])
                    for row in selected_timeline
                    if int(row["block"]) == block
                )
                for block in range(BLOCK_COUNT)
            ]
            counter_axis.step(
                time_s,
                values,
                where="mid",
                color=color,
                linewidth=1.25,
                label=label,
            )
        counter_axis.set_xlabel("Time in 64-s measurement (s; 500-ms bins)")
        counter_axis.set_ylabel("Fleet count / bin")
        counter_axis.grid(True)

        axis = axes[2, column]
        selected = [
            row
            for row in route_rows
            if int(row["num_ssu"]) == num_ssu
            and row["case"] == "layer_once_ttl_5ms"
        ]
        _require(selected, f"SSU{num_ssu}: layer-once route plot has no probes")
        group_count = len(json.loads(str(selected[0]["group_weights"])))
        for source, (color, marker) in source_style.items():
            x_values: list[float] = []
            y_values: list[int] = []
            sizes: list[float] = []
            for row in selected:
                if row.get("pressure_source") != source:
                    continue
                groups = json.loads(str(row["selected_group_ids"]))
                pressure_age = float(row["pressure_age_ms"])
                for group in groups:
                    x_values.append(float(row["relative_time_ms"]))
                    y_values.append(
                        int(row["ssu_id"]) * group_count + int(group)
                    )
                    sizes.append(12.0 + 8.0 * pressure_age)
            axis.scatter(
                x_values,
                y_values,
                s=sizes,
                c=color,
                marker=marker,
                alpha=0.72,
                linewidths=0.8,
                label=f"{source} pressure",
            )
        ticks = [
            ssu * group_count + group
            for ssu in range(num_ssu)
            for group in range(group_count)
        ]
        axis.set_yticks(
            ticks,
            [
                f"S{ssu}/G{group}"
                for ssu in range(num_ssu)
                for group in range(group_count)
            ],
            fontsize=6,
        )
        axis.set_title(
            "Bounded 50-ms Path evidence; exact replay "
            f"{sum(bool(row['portable_replay_exact_match']) for row in selected)}/{len(selected)}"
        )
        axis.set_xlabel("Time into bounded route probe (ms)")
        axis.grid(True)
    assert group_image is not None
    colorbar = fig.colorbar(
        group_image, ax=axes[0, :].ravel().tolist(), fraction=0.015, pad=0.012
    )
    colorbar.set_label("Route blocks assigned to SSU/group in each 500-ms interval")
    axes[0, 0].set_ylabel("64-s counter evidence\nSSU / QoS group")
    axes[1, 0].legend(loc="best", fontsize=7)
    axes[2, 0].set_ylabel("50-ms probe\nselected SSU/group")
    axes[2, 0].legend(loc="best", fontsize=7)
    fig.suptitle(
        "Layer-once route causality: full-window group/fresh/cache counters plus bounded Path replay",
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Top/middle: exact 500-ms cumulative-counter differences across all 64 s. Bottom: capped route records (at most 10,000) from the first 50 ms have per-plan portable Path replay; cap flag is in CSV. TTFT slack is never a routing input.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    _save_figure(
        fig,
        output,
        "Layer-once pressure source and selected group route causality",
        compact=compact,
    )


def _plot_compute_decomposition(
    decomposition_rows: Sequence[dict], output: Path, *, compact: bool
) -> None:
    selected = [
        row
        for row in decomposition_rows
        if row["window_kind"] == "cumulative"
        and row["requested_duration_s"] == 64.0
        and row["case"] != "baseline"
    ]
    fig, axes = plt.subplots(1, len(SSU_COUNTS), figsize=(11.0, 3.7) if compact else (18.0, 6.0), sharey=True)
    for axis, num_ssu in zip(axes, SSU_COUNTS):
        rows = [row for row in selected if row["num_ssu"] == num_ssu]
        x = np.arange(len(rows))
        width = 0.18
        component_specs = (
            ("delta_vs_baseline_activated_compute_npu_s", "Delta activated compute", "#4c78a8", 0),
            ("delta_vs_baseline_q_start_npu_s", "Delta Q(start)", "#f58518", 1),
            ("delta_vs_baseline_q_end_npu_s", "- Delta Q(end)", "#54a24b", 2),
        )
        for field, label, color, offset in component_specs:
            values = [float(row[field]) * (-1.0 if field.endswith("q_end_npu_s") else 1.0) for row in rows]
            axis.bar(x + (offset - 1) * width, values, width=width, color=color, label=label)
        actual = [float(row["delta_vs_baseline_compute_busy_npu_s"]) for row in rows]
        reconstructed = [float(row["delta_vs_baseline_reconstructed_compute_npu_s"]) for row in rows]
        axis.scatter(x + 2 * width, actual, marker="D", color="black", label="Observed delta compute busy", zorder=4)
        axis.scatter(x + 2 * width, reconstructed, marker="x", color="#e45756", label="Reconstructed sum", zorder=5)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, [CASE_LABELS[str(row["case"])] for row in rows], rotation=12, ha="right")
        axis.set_title(f"{num_ssu} SSUs")
        axis.grid(True, axis="y")
    axes[0].set_ylabel("Policy - baseline over 64 s (NPU-seconds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.99), fontsize=8)
    fig.suptitle("Exact utilization conservation: delta C = delta activated + delta Q(start) - delta Q(end)", y=1.04, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_figure(fig, output, "Exact 64-second compute inventory decomposition against baseline", compact=compact)


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _summarize_matched_coverage(rows: Sequence[dict]) -> list[dict]:
    result: list[dict] = []
    for num_ssu in SSU_COUNTS:
        selected = [row for row in rows if int(row["num_ssu"]) == num_ssu]
        _require(
            len(selected) == NUM_NPU,
            f"SSU{num_ssu}: expected per-NPU matched coverage rows",
        )
        common = [int(row["common_request_count"]) for row in selected]
        ious = [
            float(row["intersection_over_union"])
            for row in selected
            if row["intersection_over_union"] is not None
        ]
        result.append(
            {
                "num_ssu": num_ssu,
                "npu_count": NUM_NPU,
                "common_request_count_min": min(common),
                "common_request_count_median": float(np.median(common)),
                "common_request_count_max": max(common),
                "zero_common_coverage_npu_count": sum(value == 0 for value in common),
                "intersection_over_union_min": min(ious) if ious else None,
                "intersection_over_union_median": (
                    float(np.median(ious)) if ious else None
                ),
                "intersection_over_union_max": max(ious) if ious else None,
                "cohort_evidence_only_not_full_window_idle": True,
            }
        )
    return result


def _build_results_summary(
    window_rows: Sequence[dict],
    state_duration_summary: Sequence[dict],
    decomposition_rows: Sequence[dict],
    matched_summary: Sequence[dict],
    decision_rows: Sequence[dict],
    route_summary: Sequence[dict],
    matched_coverage_summary: Sequence[dict],
) -> dict:
    whole = [
        row
        for row in window_rows
        if row["window_kind"] == "cumulative"
        and row["requested_duration_s"] == 64.0
    ]
    decomposition = [
        row
        for row in decomposition_rows
        if row["window_kind"] == "cumulative"
        and row["requested_duration_s"] == 64.0
    ]
    decisions = {}
    for num_ssu in SSU_COUNTS:
        selected = [row for row in decision_rows if row["num_ssu"] == num_ssu]
        relative_times_ms = sorted(
            float(row["relative_time_s"]) * 1000.0 for row in selected
        )
        gaps_ms = [
            right - left
            for left, right in zip(relative_times_ms, relative_times_ms[1:])
        ]
        measurement_commits = sum(
            int(int(row["changed_cir_entry_count"]) > 0) for row in selected
        )
        decisions[str(num_ssu)] = {
            "measurement_evaluations": len(selected),
            "measurement_commits_reconstructed": measurement_commits,
            "inter_evaluation_gap_ms_min": min(gaps_ms) if gaps_ms else None,
            "inter_evaluation_gap_ms_median": (
                statistics.median(gaps_ms) if gaps_ms else None
            ),
            "inter_evaluation_gap_ms_max": max(gaps_ms) if gaps_ms else None,
            "all_observed_measurement_gaps_equal_100ms": bool(gaps_ms)
            and all(_close(value, 100.0) for value in gaps_ms),
            "all_measurement_evaluations_changed_at_least_one_cir_entry": (
                measurement_commits == len(selected)
            ),
            "trigger_reasons": sorted(
                {
                    reason
                    for row in selected
                    for reason in str(row["trigger_reasons"]).split("|")
                    if reason
                }
            ),
            "selected_npu_count_mean": statistics.fmean(float(row["selected_npu_count"]) for row in selected),
            "selected_npu_count_min": min(int(row["selected_npu_count"]) for row in selected),
            "selected_npu_count_max": max(int(row["selected_npu_count"]) for row in selected),
            "cir_entry_writes_reconstructed": sum(int(row["changed_cir_entry_count"]) for row in selected),
            "deadline_or_slack_used_by_controller": False,
        }
    utilization_spread_by_ssu = []
    for num_ssu in SSU_COUNTS:
        values = [
            float(row["npu_utilization"])
            for row in whole
            if int(row["num_ssu"]) == num_ssu
        ]
        utilization_spread_by_ssu.append(
            {
                "num_ssu": num_ssu,
                "minimum": min(values),
                "maximum": max(values),
                "spread_fraction": max(values) - min(values),
                "spread_percentage_points": 100.0 * (max(values) - min(values)),
            }
        )
    nonbaseline_state = [
        row for row in state_duration_summary if row["case"] != "baseline"
    ]
    maximum_abs_other_fraction = max(
        abs(float(row["other_fraction"])) for row in state_duration_summary
    )
    utilization_delta_explained_by_barrier = all(
        abs(float(row["delta_vs_baseline_other_npu_s"])) <= 2e-9
        and abs(
            float(row["delta_vs_baseline_compute_npu_s"])
            + float(row["delta_vs_baseline_io_barrier_npu_s"])
        )
        <= 2e-9
        for row in nonbaseline_state
    )
    return {
        "analysis": "npu32_ssu345_timeline64_v1",
        "bandwidth_semantics": {
            "actual_received": "NPU-link delivered GB/s from exact cumulative counter differences",
            "interval_average_attributed_ssd_service": (
                "NPU-by-SSU v3 stable completed-command-plus-immutable-active-prefix counter difference divided by the 500-ms interval; not an instantaneous in-flight command rate; fragmented settle accumulation is excluded"
            ),
            "ssd_outstanding": (
                "direct physical pending/active enumeration; never derived by cumulative-counter subtraction"
            ),
            "instantaneous_inflight_ssd_command_rate": (
                "available only in active_command/probe physical_service_gbps fields; normally 40 GB/s for the one active non-preemptive command"
            ),
            "installed_cir": "arbitration guarantee, not actual throughput",
            "controller_demand": "remaining manifest IO / remaining controller compute",
            "physical_demand": "undelivered physical IO / same compute denominator; diagnostic only",
        },
        "adaptive_deadline_input": False,
        "adaptive_required_ratio": 0.5,
        "adaptive_required_ratio_interpretation": "alpha=2 feasibility configuration, not a live deadline check",
        "adaptive_tuning_scope": (
            "alpha=2 tuned; alpha=1.5 is a same-trajectory sensitivity metric, "
            "not a separately alpha=1.5-tuned run"
        ),
        "adaptive_prefetch_slack_policy": "prefetch-only or not-yet-admitted requests have NA elapsed/slack and never enter slack plots",
        "utilization_spread_by_ssu": utilization_spread_by_ssu,
        "maximum_strategy_utilization_spread_percentage_points": max(
            row["spread_percentage_points"] for row in utilization_spread_by_ssu
        ),
        "maximum_abs_state_other_fraction": maximum_abs_other_fraction,
        "all_strategy_utilization_deltas_explained_by_io_barrier_when_other_zero": (
            utilization_delta_explained_by_barrier
        ),
        "whole_64s": whole,
        "state_duration_summary_64s": list(state_duration_summary),
        "compute_decomposition_64s": decomposition,
        "matched_request_summary": list(matched_summary),
        "matched_request_coverage_summary": list(matched_coverage_summary),
        "adaptive_decision_summary": decisions,
        "route_probe_summary": list(route_summary),
    }


def _build_report(
    validation: dict,
    results: dict,
    window_rows: Sequence[dict],
    matched_counts: Sequence[dict],
    probe_rows: Sequence[dict],
    route_rows: Sequence[dict],
    decision_npu_rows: Sequence[dict],
    admission_attempt_rows: Sequence[dict],
    grant_component_rows: Sequence[dict],
    forensic_selection_rows: Sequence[dict],
) -> str:
    whole = {
        (int(row["num_ssu"]), str(row["case"])): row
        for row in results["whole_64s"]
    }
    decomposition = {
        (int(row["num_ssu"]), str(row["case"])): row
        for row in results["compute_decomposition_64s"]
    }
    state_durations = {
        (int(row["num_ssu"]), str(row["case"])): row
        for row in results["state_duration_summary_64s"]
    }
    matched = {
        (int(row["num_ssu"]), str(row["policy"])): row
        for row in results["matched_request_summary"]
    }
    route_summary = {
        (int(row["num_ssu"]), str(row["case"])): row
        for row in results["route_probe_summary"]
    }
    matched_coverage = {
        int(row["num_ssu"]): row
        for row in results["matched_request_coverage_summary"]
    }
    control_summary = results["adaptive_decision_summary"]
    control_is_saturated_100ms = all(
        int(control_summary[str(num_ssu)]["measurement_evaluations"]) == 640
        and int(
            control_summary[str(num_ssu)][
                "measurement_commits_reconstructed"
            ]
        )
        == 640
        and bool(
            control_summary[str(num_ssu)][
                "all_observed_measurement_gaps_equal_100ms"
            ]
        )
        for num_ssu in SSU_COUNTS
    )
    maximum_utilization_spread_pp = float(
        results["maximum_strategy_utilization_spread_percentage_points"]
    )
    maximum_abs_other_fraction = float(
        results["maximum_abs_state_other_fraction"]
    )
    barrier_only_delta = bool(
        results[
            "all_strategy_utilization_deltas_explained_by_io_barrier_when_other_zero"
        ]
    )
    lines = [
        "# 32 NPU / SSU 3–5 / 64 秒微观时间线审计",
        "",
        "## 验收结论",
        "",
        "9 个点（3 个策略 × SSU 3/4/5）全部通过独立验收：每点 128 个 500 ms 块、129 个左极限边界，仿真器报告的全部 invariant 均为 `true`。分析器又独立重算了 NPU×SSU 的 interval-average attributed SSD service、NPU-link delivery、资源容量和 `C = activated + Q(start) - Q(end)` 闭合。",
        "",
        "严格配对范围是同一 SSU 内三策略的完整输入 fingerprint 和科学运行时；跨 SSU 则严格要求 profile catalog、采样 recipe、schedule 和 assignment 相同。placement/trace/simulator fingerprint 随 SSU 数变化是预期行为。报告没有写入 hostname、PID、token 或私有绝对路径。",
        "三个 SSU shard 的完整 hostname/platform/thread/CPython/NumPy/BLAS runtime signature 只在内存中比较，公开结果仅保留不可逆 SHA-256。原始 runner shard 不复制进公开目录；每个公开文件必须严格小于 50 MiB。",
        "",
        "## 领导汇报摘要",
        "",
        "- Adaptive 本次配置是 **α=2 tuned**：`required_ratio=0.5` 是 α=2 容量可行性目标，不是运行时读取 TTFT deadline/slack。α=1.5 图仅是同一条 α=2-tuned 轨迹的补充敏感性统计，不是另跑的 α=1.5-tuned 控制器。",
        f"- 64 秒全窗内，同一 SSU 三策略的 NPU 利用率最大精确跨度为 `{maximum_utilization_spread_pp:.6f}` 个百分点；不是逐位完全相等。",
        (
            f"- 完整 state partition 的最大 `other` 占比仅 `{100.0 * maximum_abs_other_fraction:.9f}%`，且所有策略相对 baseline 的 compute 差均由反向 IO-barrier 差精确闭合；因此该模型里的利用率差来自 IO barrier，而不是未解释的 idle。"
            if barrier_only_delta
            else f"- 完整 state partition 的最大 `other` 占比为 `{100.0 * maximum_abs_other_fraction:.9f}%`；逐策略精确数值见下表，不能把全部利用率差都简化为 IO barrier。"
        ),
        (
            "- 三个 SSU 的正式 measurement 均观测到 `640 evaluations / 640 commits`，相邻 evaluation 均为 `100 ms`。这是持续 batch-boundary 事件让 100-ms 最小间隔在这条高负载轨迹上饱和；trigger 仍只有事件来源，并不存在 wall-clock 周期 trigger，不能把一般语义写成‘每 100 ms 定时改 CIR’。"
            if control_is_saturated_100ms
            else "- Adaptive evaluation/commit 数及相邻间隔由决策记录逐条重建，当前输入未同时满足三个 SSU 都是 `640/640` 且每个 gap 恰为 100 ms；具体动态值见后表。event-driven 的一般语义仍不是 wall-clock 周期改 CIR。"
        ),
        "",
        "| SSU | measurement evaluations | commits | gap min/median/max | trigger reasons |",
        "|---:|---:|---:|---:|---|",
        *[
            (
                f"| {num_ssu} | {int(control_summary[str(num_ssu)]['measurement_evaluations'])} | "
                f"{int(control_summary[str(num_ssu)]['measurement_commits_reconstructed'])} | "
                f"{float(control_summary[str(num_ssu)]['inter_evaluation_gap_ms_min']):.6f}/"
                f"{float(control_summary[str(num_ssu)]['inter_evaluation_gap_ms_median']):.6f}/"
                f"{float(control_summary[str(num_ssu)]['inter_evaluation_gap_ms_max']):.6f} ms | "
                f"`{', '.join(control_summary[str(num_ssu)]['trigger_reasons'])}` |"
            )
            for num_ssu in SSU_COUNTS
        ],
        "",
        "## 64 秒总体结果",
        "",
        "| SSU | 策略 | NPU 利用率 | equal-NPU TTFT SLO α=1.5 | equal-NPU TTFT SLO α=2 | 平均 IO barrier | 请求数 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            row = whole[(num_ssu, case)]
            lines.append(
                f"| {num_ssu} | {CASE_LABELS[case]} | {_pct(row['npu_utilization'])} | "
                f"{_pct(row['equal_observed_npu_slo_alpha1p5'])} | {_pct(row['equal_observed_npu_slo_alpha2'])} | "
                f"{float(row['mean_io_barrier_ms']):.3f} ms | {int(row['request_count'])} |"
            )

    lines.extend(
        [
            "",
            "## TTFT 窗口与启动阶段口径",
            "",
            "`TTFT = request completion_time - admission_time`，不包含 arrival→admission 的等待。正式 measurement 在每个 NPU 已 warm up 8 个请求后再 settle 500 ms 才开始；SLO cohort 只包含 admission 落在 `[measurement_start, measurement_end)` 的请求，因此不包含仿真最初请求的启动瞬态。measurement start 已在途的 carry-in 不进入 SLO cohort，但其与窗口的交集会完整进入 utilization/state partition。这里描述的是仿真口径，不能称为真实硬件 cold-start 测量。",
            "",
            "layer 0 I/O 若在 admission 后才 ready，其等待会进入 TTFT 的 IO barrier；cross-request prefetch 若在 admission 前已经完成，则提前完成的那一段不会计入 TTFT。逐层导出的 `compute_start=max(previous_compute_end, io_ready)` 与 barrier 之和已经逐请求闭合到 TTFT。",
        ]
    )

    lines.extend(
        [
            "",
            "## 低利用率的主证据：完整 64 秒状态时间分解",
            "",
            "下面直接对每个 NPU 的所有 microbatch layer 区间与严格 64 秒窗口求交，包含 measurement start 时已经在途的 carry-in。每个 NPU 都精确满足 `compute + io_barrier + other = 64 s`，其中 compute 与利用率分子独立一致；因此这张表覆盖完整窗口的 idle，而不是只看 measurement 内新 admission 的请求 cohort。",
            "",
            "| SSU | 策略 | compute | IO-barrier idle | other complement | barrier / all idle | vs baseline: Δbarrier | vs baseline: Δother |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            row = state_durations[(num_ssu, case)]
            lines.append(
                f"| {num_ssu} | {CASE_LABELS[case]} | {_pct(row['compute_fraction'])} | "
                f"{_pct(row['io_barrier_fraction'])} | {_pct(row['other_fraction'])} | "
                f"{_pct(row['io_barrier_share_of_idle'])} | "
                f"{float(row['delta_vs_baseline_io_barrier_npu_s']):+.6f} NPU-s | "
                f"{float(row['delta_vs_baseline_other_npu_s']):+.6f} NPU-s |"
            )

    lines.extend(
        [
            "",
            "## 为什么 SLO 可以更高，但平均 NPU 利用率不一定更高",
            "",
            "两者统计对象不同。上面的完整时间分解给出 idle 究竟落在 IO barrier 还是 other；NPU 利用率本身是 64 秒内 compute busy time 的积分，而 TTFT SLO 是逐请求的阈值计数。策略可能把等待在请求之间重新分布，从而增加 fail→pass 个数而不增加总 compute busy time；但这句话本身不是整体 SLO 差异的因果证明。闭环运行中，各策略按半开测量窗 admission 得到的整体 SLO cohort 成员和数量可能不同，因此不能仅凭整体 SLO 直接归因。matched common cohort 只佐证其覆盖到的交集请求；零 common NPU 和非交集请求均不支持该因果外推。完整 64 秒 state partition 才是利用率/idle 根因的全窗证据。",
            "",
            "下面的 Q/activated 等式进一步解释 compute busy 总量为什么变化，但它不是 idle 类型的替代指标；等式由逐 NPU、逐 500 ms 边界精确闭合后再汇总：",
            "",
            "`Δ compute_busy = Δ activated_compute + Δ Q(start) - Δ Q(end)`",
            "",
            "| SSU | 策略 vs baseline | Δcompute busy | Δactivated | ΔQ(start) | -ΔQ(end) | 闭合误差 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES[1:]:
            row = decomposition[(num_ssu, case)]
            lines.append(
                f"| {num_ssu} | {CASE_LABELS[case]} | {float(row['delta_vs_baseline_compute_busy_npu_s']):+.6f} NPU-s | "
                f"{float(row['delta_vs_baseline_activated_compute_npu_s']):+.6f} | "
                f"{float(row['delta_vs_baseline_q_start_npu_s']):+.6f} | "
                f"{-float(row['delta_vs_baseline_q_end_npu_s']):+.6f} | "
                f"{float(row['delta_vs_baseline_closure_error_npu_s']):+.3e} |"
            )

    lines.extend(
        [
            "",
            "matched-request 只作为 TTFT cohort 的佐证：它比较三策略测量窗中共同的 `(NPU, sequence)` 请求，可展示阈值重分配，但不包含所有 carry-in，也绝不用于声称覆盖完整 64 秒 idle：",
            "",
            "| SSU | 每NPU common请求 min/median/max | 零 common coverage NPU数 | 每NPU IoU min/median/max |",
            "|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in SSU_COUNTS:
        row = matched_coverage[num_ssu]
        lines.append(
            f"| {num_ssu} | {int(row['common_request_count_min'])}/"
            f"{float(row['common_request_count_median']):.1f}/"
            f"{int(row['common_request_count_max'])} | "
            f"{int(row['zero_common_coverage_npu_count'])} | "
            f"{float(row['intersection_over_union_min']):.3f}/"
            f"{float(row['intersection_over_union_median']):.3f}/"
            f"{float(row['intersection_over_union_max']):.3f} |"
        )
    lines.extend(
        [
            "",
            "| SSU | 策略 vs baseline | common 请求 | barrier 改善/恶化 | α1.5 fail→pass / pass→fail | α2 fail→pass / pass→fail | barrier 总变化 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES[1:]:
            row = matched[(num_ssu, case)]
            lines.append(
                f"| {num_ssu} | {CASE_LABELS[case]} | {int(row['matched_request_count'])} | "
                f"{int(row['barrier_improved_request_count'])}/{int(row['barrier_regressed_request_count'])} | "
                f"{int(row['fail_to_pass_alpha1p5'])}/{int(row['pass_to_fail_alpha1p5'])} | "
                f"{int(row['fail_to_pass_alpha2'])}/{int(row['pass_to_fail_alpha2'])} | "
                f"{float(row['total_barrier_delta_vs_baseline_s']):+.6f} s |"
            )

    lines.extend(
        [
            "",
            "## 带宽时间线应如何读",
            "",
            "- `controller_demand` 是 remaining manifest IO / remaining controller compute。它是 Adaptive 的输入；对 baseline/layer-once 只作为同口径反事实诊断。",
            "- `physical_remaining_demand` 是 producer 直接导出的物理未交付工作除以同一 compute denominator，只是诊断，不是控制器输入。分析器独立验证 activated IO 各 stage 守恒、direct SSD/link queue、`physical >= activated-undelivered`、`controller >= physical` 及 demand 除法，但现有导出不足以从零完全重建全部未来尚未 activate 的 physical work，因此不把它冒充成独立 oracle。",
            "- `installed CIR` 是 dedicated Path 的仲裁保证，不是实际吞吐。主图没有把 CIR 画成 actual。",
            "- `interval-average attributed SSD service` 是相邻 500 ms 边界间 NPU×SSU SSD-served 累计字节差除以 0.5 s；它不是某一时刻正在执行命令的瞬时速率。同一窗口内可先后服务多个 cell。",
            "- 只有 `active_command_by_ssu.physical_service_gbps` 与 dispatch probe 的 `physical_command_service_gbps` 描述当时唯一在途非抢占命令的服务速率（本配置为 40 GB/s）。`NPU-link delivered` 才是该 NPU 从该 SSU 在区间内平均实际收到的带宽。二者的区间平均差刻画 SSD→link 管线填充/排空与排队。",
            "- v3 边界把层状态拆成 `current_compute_layer`、`next_compute_layer`、`next_compute_layer_io_ready` 和 `waiting_on_io_layer`；不再把正在计算的当前层误写成 next-layer IO 状态。SSD scientific service 只采用 compensated completed-command 累计加 immutable active-prefix；历史 fragmented-settle 累计只留作 residual 诊断，绝不用于带宽图或结论。SSD outstanding 则直接枚举 physical pending/active，不由两个累计数相减构造。",
            "- Adaptive 的每个旧 CIR 都由前一次实际安装值重建（首次为 0），threshold=0 时按 `abs(target-old)>1e-12` 重放。重建后的 commit/transaction/path-write 计数和 129 个边界的 installed CIR 已全部一致。",
            "",
            "## Adaptive 是否因为‘快到 TTFT SLO’才分配带宽",
            "",
            "不是。当前控制器的因果输入没有 admission time、elapsed TTFT、deadline 或 slack。`required_ratio=0.5` 是 α=2 的容量可行性配置，不是实时 deadline 检查。`min_interval_ms=100` 是 event-driven controller evaluation/commit 的最小间隔（节流），并不表示每 100 ms 周期性必改一次 CIR；只有事件触发、距上次 evaluation 至少 100 ms，并且 `abs(target-old)>threshold` 时对应 CIR entry 才写入。`adaptive_decisions.csv` 的相邻 time/trigger/changed-entry 可逐次核对。图里同时给出 `deadline_slack = α×ideal - elapsed` 与 `slack_after_remaining_compute = deadline_slack - Q(t)`；它们均由边界状态事后派生，并明确标注 diagnostic-only。decision-time 的 prefetch-only/尚未 admission 请求没有 TTFT 时钟，因此 elapsed、slack 和 slack-Q 一律为 NA，也不进入 slack 曲线。若 selected 与 slack 有相关性，也不能解释为控制器读取了 slack。",
            "",
            "Adaptive 真正的选择理由记录在 decision CSV：controller demand、normalized total/dominant score、candidate order、pinned 状态、admission stage、capacity rejection、violating SSU 和最终 grant。admission 决定‘谁被选中’，grant decomposition 决定‘每个 NPU×SSU 最终给多少’；explicit-spill/V2 使用 floor/background/selected-tail/spill-tail 四项，coflow/V1 则单列 V1 grant，所有 cell 均逐元素闭合到 final grant。Layer-once 的理由则由每块 route plan、fresh pressure read、TTL cache hit 和 dispatch probe 的 virtual-finish/RR 选择共同刻画。",
            "",
            "## Layer-once 的 route plan 为什么选择这些 Path/group",
            "",
            "route probe 保留了纯策略函数所需的完整输入：allowed Path 的 pressure count/CIR/PIR/weight/group、全 group 活跃聚合、block 大小和 start offset。分析器用 `policy_logic.pressure_aware_path_ids` 独立重放，要求 selected Path 逐 block 完全相同；`fresh` 必须 age=0，`cache` 必须仍在 5 ms TTL 内。这里同样没有 TTFT deadline/slack 输入。",
            "",
            "| SSU | 策略 | route records | planned blocks | fresh/cache/none | selected groups | portable exact replay | truncated |",
            "|---:|---|---:|---:|---|---|---:|---:|",
        ]
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            row = route_summary[(num_ssu, case)]
            lines.append(
                f"| {num_ssu} | {CASE_LABELS[case]} | {int(row['probe_record_count'])} | "
                f"{int(row['planned_block_count'])} | `{_canonical(row['pressure_source_counts'])}` | "
                f"`{_canonical(row['selected_group_counts'])}` | "
                f"{bool(row['portable_replay_all_exact'])} | {bool(row['probe_stream_truncated'])} |"
            )

    lines.extend(
        [
            "",
            "## 可逐式复算的 Adaptive 决策样例",
            "",
            "下面每个 SSU 确定性取测量窗内最早一个有 admission attempt 的候选（若该模式无需 attempt，则取最早 selected NPU）。`target = effective_ratio × manifest demand`；`normalized_total = Σ(target_ssu / 40)`，`normalized_dominant = max(target_ssu / 40)`。attempt 的 before/after 和 violating SSU 解释‘为何选中/拒绝’，grant 分量解释‘最终给多少’。slack 不参与任何一步。",
            "",
        ]
    )
    for num_ssu in SSU_COUNTS:
        candidates = [
            row
            for row in decision_npu_rows
            if int(row["num_ssu"]) == num_ssu
            and row.get("candidate_normalized_total") is not None
        ]
        attempted = {
            (int(row["evaluation"]), int(row["npu_id"]))
            for row in admission_attempt_rows
            if int(row["num_ssu"]) == num_ssu
        }
        sample = min(
            (
                row
                for row in candidates
                if (int(row["evaluation"]), int(row["npu_id"])) in attempted
            ),
            key=lambda row: (
                int(row["evaluation"]),
                int(row["candidate_order_rank_zero_based"] or 0),
                int(row["npu_id"]),
            ),
            default=min(
                (row for row in candidates if bool(row["selected"])),
                key=lambda row: (int(row["evaluation"]), int(row["npu_id"])),
                default=None,
            ),
        )
        _require(sample is not None, f"SSU{num_ssu}: no Adaptive decision sample")
        evaluation = int(sample["evaluation"])
        npu_id = int(sample["npu_id"])
        demand = [
            float(sample[f"controller_demand_ssu{ssu}_gbps"])
            for ssu in range(num_ssu)
        ]
        ratio = float(sample["effective_target_ratio"])
        target = [ratio * value for value in demand]
        expected_total = math.fsum(value / SSD_CAP_GBPS for value in target)
        expected_dominant = max((value / SSD_CAP_GBPS for value in target), default=0.0)
        _require(
            _close(expected_total, float(sample["candidate_normalized_total"]))
            and _close(
                expected_dominant,
                float(sample["candidate_normalized_dominant"]),
            ),
            f"SSU{num_ssu}: exported decision sample score does not recompute",
        )
        attempts = [
            row
            for row in admission_attempt_rows
            if int(row["num_ssu"]) == num_ssu
            and int(row["evaluation"]) == evaluation
            and int(row["npu_id"]) == npu_id
        ]
        attempt = max(attempts, key=lambda row: int(row["attempt_index"]), default=None)
        components = [
            row
            for row in grant_component_rows
            if int(row["num_ssu"]) == num_ssu
            and int(row["evaluation"]) == evaluation
            and int(row["npu_id"]) == npu_id
        ]
        _require(
            len(components) == num_ssu,
            f"SSU{num_ssu}: Adaptive sample grant components incomplete",
        )
        grant = [float(row["final_grant_gbps"]) for row in components]
        component_mode = str(components[0]["residual_mode"])
        lines.extend(
            [
                f"### SSU {num_ssu}",
                "",
                f"evaluation `{evaluation}`，t=`{float(sample['relative_time_s']):.3f}s`，NPU `{npu_id}` / request `{int(sample['request_id'])}`；mode=`{sample['selection_mode']}`，residual=`{component_mode}`，selected=`{bool(sample['selected'])}`。",
                "",
                f"- demand=`{_canonical(demand)}` GB/s；ratio=`{ratio:.6g}`；target=`{_canonical(target)}` GB/s。",
                f"- normalized total=`{expected_total:.9g}`，dominant=`{expected_dominant:.9g}`；candidate rank=`{sample['candidate_order_rank_zero_based']}`。",
                f"- final grant=`{_canonical(grant)}` GB/s；逐 cell decomposition closure 见 `adaptive_grant_components.csv`。",
                (
                    "- 此模式无需逐候选 admission attempt（全体 preferred/required 可行）。"
                    if attempt is None
                    else f"- attempt stage=`{attempt['stage']}`，accepted=`{bool(attempt['accepted'])}`，before=`{attempt['admission_remaining_before_gbps_by_ssu']}`，after=`{attempt['admission_remaining_after_gbps_by_ssu']}`，violating SSU=`{attempt['violating_ssu_ids'] or 'none'}`。"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## 窗口效应",
            "",
            "0.5/1/2/3/8/16/32/64 秒均同时输出 cumulative 和 disjoint 统计。利用率是该时间窗内严格 resource-time 积分；SLO 则按 admission time 落在半开窗内选择 cohort，TTFT 允许在该窗结束之后完成，所以它不是‘截至 3 秒已完成请求’的在线 SLO。3 秒不能整除 64 秒，CSV 保留最后 1 秒并标记 `full_requested_window=false`；min–max 图只使用完整 disjoint 窗，避免把 1 秒尾窗伪装成 3 秒窗。短窗差异是系统相位、carry-in 库存 Q、请求 profile 和排队状态的混合；不能仅凭一个短窗断言长期吞吐改变。",
            "",
            "## 微观 dispatch 证据与限制",
            "",
            f"共导出 {len(probe_rows)} 条 first-50-ms capped dispatch prefix 记录；每个 case 最多 10000 条，`probe_stream_truncated=True` 仅表示 record cap reached，不能据此声称观察到了第 10001 条，也不能把该前缀称为完整 50 ms dispatch 序列。这些记录是 simulator 在 dispatch 前的 read-only winner prediction，随后由 runtime assertion 与实际 Path 核对，并不是分析器独立重演完整调度器。CSV 保留 cap flag、candidate path count、virtual finish、RR cursor、estimated arbitration rate、CIR、group、queue wait、request/layer/block 和 physical command rate。",
            f"另导出 {len(route_rows)} 条 first-50-ms capped route prefix 记录；每条已记录 route plan 的纯策略函数重放均逐 block exact-match，fresh/cache age 与 selected Path/group 映射均已硬验收。具体 Path 的 portable replay 只覆盖 measurement 开头 50 ms 内已记录、最多 10000 条的前缀；若 cap reached 则不声称完整覆盖这 50 ms，更不能外推为 64 秒逐 dispatch Path 重放。全 64 秒仅有每 500 ms 的 group/fresh/cache 累计计数及其差分证据。",
            "",
            "## 模型真实性边界",
            "",
            "这里的 NPU utilization 是仿真器对 compute interval 的 binary duty-cycle 积分，不是芯片性能计数器。每个 SSD 同时最多执行一个 40 GB/s 非抢占 command；CIR 只影响下一条离散 command 的仲裁，不是多个 IO 并发时按比例持续限速。每个 NPU link 也是单流 50 GB/s。层计算时长与 profile 固定，模型没有真实 HBM、网络、缓存和内核执行抖动。因而这些时间线可以严格证明仿真内部的状态、守恒和决策链，却不能替代真实硬件的外部测量；这些离散且固定的结构也会让不同策略的长期总 NPU duty cycle 更容易接近。",
            "",
            "本报告能严格证明这一个固定 workload/seed/64 秒轨迹中的守恒关系和调度机制；它不能把单 seed 结果提升为总体概率结论。若用于最终对外结论，仍应追加多 seed 置信区间和更长稳态窗。",
            "",
            "## 产物索引",
            "",
            "- `00a_mean_npu_utilization_vs_ssu.png`、`00b_ttft_slo_alpha1p5_vs_ssu.png`、`00c_ttft_slo_alpha2_vs_ssu.png`: 64 秒全窗的三策略 SSU3/4/5 汇总曲线；SLO 是 equal-observed-NPU 口径。",
            "- `timeline_npu_ssu_500ms_ssu{3,4,5}_{case}.csv`: 每个策略、NPU、SSU、500 ms 的 demand/CIR/interval-average attributed SSD service/link delivery/slack/queue/route 证据；按 SSU×策略拆分以保持单文件适合 GitHub。",
            "- `window_metrics.csv`: cumulative + disjoint 多尺度利用率与 α1.5/α2 SLO。",
            "- `state_durations_per_npu.csv`、`state_duration_summary.csv` 与 `state_durations_500ms.csv`: 包含 carry-in 的完整窗口及 128 个精确块 compute/IO-barrier/other 主根因分解。",
            "- `compute_inventory_decomposition.csv`: Q/activated compute 精确分解。",
            "- `matched_requests.csv` 与 `matched_request_layers_ssu{N}_{case}_npu{lo}_{hi}.csv`: 同请求及逐层 barrier 时间；逐 SSU×策略×8-NPU 分片，保证 GitHub 单文件体积。",
            "- `forensic_selection.csv`、`forensic_timeline.csv` 与 `09_forensic_zoom_ssu{3,4,5}.png`: 确定性 full-lifecycle matched 请求的 16 层 Gantt 与 request-ID 绑定后的局部证据；SSD/link 明确是 NPU aggregate，不能误称单请求归因。",
            "- `adaptive_decisions.csv`, `adaptive_decision_npu.csv`, `adaptive_admission_attempts_ssu{3,4,5}.csv`, `adaptive_grant_components.csv`, `adaptive_boundary_slack.csv`: 决策因果输入、candidate order/rank、attempt/capacity rejection、V1/V2 grant 分量、重建 CIR 和 diagnostic-only slack。",
            "- `path_state_500ms_ssu{3,4,5}_{case}_disk{physical_ssu}.csv`: 按拓扑×策略×物理 SSU 拆分的 500-ms 左边界 sparse Path queue/CIR/virtual-finish/下一次仲裁速率估计；这是边界状态与 next-arbitration estimate，不是前一个区间历史 winner。拆分后即使 256 Path 全部活跃，单文件也有严格行数/体积预算。",
            "- `dispatch_probe.csv`: 开头 bounded 窗内 simulator pre-dispatch read-only prediction 与 runtime winner assertion。",
            "- `route_probe_ssu{3,4,5}.csv`: 开头 50 ms 内、最多 10000 条的 capped route prefix，含 cap flag、路径规划输入、fresh/cache TTL、selected Path/group 与逐记录纯策略函数 exact replay；若 cap reached，不声称完整覆盖该 50 ms。",
            "- `ssd_accounting_residuals.csv`: v3 stable-service/direct-physical-queue 的逐边界逐 SSU residual 及容差证明。",
            "- `analyze_npu32_ssu345_timeline64.py`: 与 validation/results/manifest 中 analyzer SHA-256 完全一致的分析器字节。",
            "",
            f"验证指纹：source `{validation['source_fingerprint']}`；config `{validation['config_fingerprint']}`；analyzer `{validation['analyzer_sha256']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def _assert_portable_text_outputs(output_dir: Path) -> None:
    forbidden = (
        "ghp_",
        "\"hostname\"",
        str(PROJECT_ROOT),
        "/home/",
    )
    for path in output_dir.iterdir():
        if path.suffix not in {".json", ".csv", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            _require(token not in text, f"portable-output guard rejected {path.name}: contains {token!r}")


def _assert_public_file_sizes(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        _require(
            path.stat().st_size < MAX_PUBLIC_FILE_BYTES,
            f"public artifact exceeds the strict <50 MiB gate: {path.name}",
        )


def analyze(
    inputs: Sequence[Path],
    output_dir: Path,
    *,
    make_plots: bool = True,
    compact_plots: bool = False,
) -> dict:
    analyzer_path = Path(__file__).resolve()
    analyzer_bytes = analyzer_path.read_bytes()
    analyzer_sha256 = hashlib.sha256(analyzer_bytes).hexdigest()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        _require(output_dir.is_dir(), "output path exists and is not a directory")
        _require(
            not any(output_dir.iterdir()),
            "output directory must be absent or empty; refusing stale artifacts/raw inputs",
        )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    cases, validation = _load_and_validate(inputs)
    validation["analyzer_sha256"] = analyzer_sha256
    validation["checks"]["analyzer_source_stable_and_published_exactly"] = True
    timeline_rows = _build_timeline_rows(cases)
    ssd_accounting_residual_rows = _build_ssd_accounting_residual_rows(cases)
    window_rows = _build_window_metrics(cases)
    state_duration_rows, state_duration_summary = _build_state_duration_tables(
        cases
    )
    block_state_duration_rows = _build_block_state_duration_rows(cases)
    decomposition_rows = _build_compute_decomposition(cases, window_rows)
    matched_rows, matched_layer_rows, matched_summary, matched_counts = _build_matched_requests(cases)
    forensic_selection_rows = _select_forensic_requests(matched_rows)
    forensic_timeline_rows = _build_forensic_timeline_rows(
        forensic_selection_rows, timeline_rows
    )
    matched_coverage_summary = _summarize_matched_coverage(matched_counts)
    (
        decision_rows,
        decision_npu_rows,
        boundary_slack_rows,
        admission_attempt_rows,
    ) = _build_adaptive_decisions(cases)
    grant_component_rows = _build_adaptive_grant_components(cases)
    probe_rows = _build_dispatch_probe_rows(cases)
    route_rows = _build_route_probe_rows(cases)
    path_state_rows = _build_path_state_rows(cases)
    route_summary = _build_route_probe_summary(route_rows)
    results = _build_results_summary(
        window_rows,
        state_duration_summary,
        decomposition_rows,
        matched_summary,
        decision_rows,
        route_summary,
        matched_coverage_summary,
    )
    results["analyzer_sha256"] = analyzer_sha256
    report = _build_report(
        validation,
        results,
        window_rows,
        matched_counts,
        probe_rows,
        route_rows,
        decision_npu_rows,
        admission_attempt_rows,
        grant_component_rows,
        forensic_selection_rows,
    )

    _write_json(output_dir / "validation.json", validation)
    _write_json(output_dir / "results.json", results)
    _write_text(output_dir / "report.md", report)
    _write_bytes(
        output_dir / "analyze_npu32_ssu345_timeline64.py", analyzer_bytes
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            _write_csv(
                output_dir
                / f"timeline_npu_ssu_500ms_ssu{num_ssu}_{case}.csv",
                [
                    row
                    for row in timeline_rows
                    if row["num_ssu"] == num_ssu and row["case"] == case
                ],
            )
    _write_csv(output_dir / "window_metrics.csv", window_rows)
    _write_csv(
        output_dir / "ssd_accounting_residuals.csv",
        ssd_accounting_residual_rows,
    )
    _write_csv(output_dir / "state_durations_per_npu.csv", state_duration_rows)
    _write_csv(output_dir / "state_duration_summary.csv", state_duration_summary)
    _write_csv(output_dir / "state_durations_500ms.csv", block_state_duration_rows)
    _write_csv(output_dir / "compute_inventory_decomposition.csv", decomposition_rows)
    _write_csv(output_dir / "matched_requests.csv", matched_rows)
    _require(
        8 * 256 * N_LAYERS * 1024 < MAX_PUBLIC_FILE_BYTES,
        "worst-case matched-layer CSV shard budget is unsafe",
    )
    for num_ssu in SSU_COUNTS:
        for case in CASES:
            for npu_lo in range(0, NUM_NPU, 8):
                npu_hi = npu_lo + 7
                _write_csv(
                    output_dir
                    / (
                        f"matched_request_layers_ssu{num_ssu}_{case}_"
                        f"npu{npu_lo:02d}_{npu_hi:02d}.csv"
                    ),
                    [
                        row
                        for row in matched_layer_rows
                        if row["num_ssu"] == num_ssu
                        and row["case"] == case
                        and npu_lo <= int(row["npu_id"]) <= npu_hi
                    ],
                )
    _write_csv(output_dir / "matched_request_summary.csv", matched_summary)
    _write_csv(output_dir / "matched_request_coverage.csv", matched_counts)
    _write_csv(output_dir / "forensic_selection.csv", forensic_selection_rows)
    _write_csv(output_dir / "forensic_timeline.csv", forensic_timeline_rows)
    _write_csv(output_dir / "adaptive_decisions.csv", decision_rows)
    _write_csv(output_dir / "adaptive_decision_npu.csv", decision_npu_rows)
    _write_csv(output_dir / "adaptive_boundary_slack.csv", boundary_slack_rows)
    _write_csv(
        output_dir / "adaptive_grant_components.csv", grant_component_rows
    )
    _write_csv(output_dir / "dispatch_probe.csv", probe_rows)
    for num_ssu in SSU_COUNTS:
        _write_csv(
            output_dir / f"adaptive_admission_attempts_ssu{num_ssu}.csv",
            [
                row
                for row in admission_attempt_rows
                if int(row["num_ssu"]) == num_ssu
            ],
        )
        _write_csv(
            output_dir / f"route_probe_ssu{num_ssu}.csv",
            [row for row in route_rows if int(row["num_ssu"]) == num_ssu],
        )
        for case in CASES:
            for physical_ssu_id in range(num_ssu):
                path_state_slice = [
                    row
                    for row in path_state_rows
                    if int(row["num_ssu"]) == num_ssu
                    and row["case"] == case
                    and int(row["ssu_id"]) == physical_ssu_id
                ]
                _require(
                    len(path_state_slice)
                    <= BOUNDARY_COUNT * EXPECTED_PATH_ABI["path_count"],
                    "sparse Path export exceeds the one-row-per-Path boundary bound",
                )
                _require(
                    BOUNDARY_COUNT
                    * EXPECTED_PATH_ABI["path_count"]
                    * 1024
                    < MAX_PUBLIC_FILE_BYTES,
                    "worst-case split Path-state CSV size budget is unsafe",
                )
                _write_csv(
                    output_dir
                    / (
                        f"path_state_500ms_ssu{num_ssu}_{case}_"
                        f"disk{physical_ssu_id}.csv"
                    ),
                    path_state_slice,
                )

    if make_plots:
        _style()
        _plot_ssu_summary_metric(
            window_rows,
            "npu_utilization",
            "Mean NPU utilization (%)",
            "32-NPU mean utilization vs. SSU count (64 s)",
            output_dir / "00a_mean_npu_utilization_vs_ssu.png",
            compact=compact_plots,
        )
        _plot_ssu_summary_metric(
            window_rows,
            "equal_observed_npu_slo_alpha1p5",
            "TTFT SLO attainment, alpha=1.5 (%)",
            "Equal-observed-NPU TTFT SLO alpha=1.5 vs. SSU count",
            output_dir / "00b_ttft_slo_alpha1p5_vs_ssu.png",
            compact=compact_plots,
        )
        _plot_ssu_summary_metric(
            window_rows,
            "equal_observed_npu_slo_alpha2",
            "TTFT SLO attainment, alpha=2 (%)",
            "Equal-observed-NPU TTFT SLO alpha=2 vs. SSU count",
            output_dir / "00c_ttft_slo_alpha2_vs_ssu.png",
            compact=compact_plots,
        )
        for num_ssu in SSU_COUNTS:
            _plot_timeline_heatmap(
                timeline_rows,
                num_ssu,
                output_dir / f"01_timeline_heatmap_ssu{num_ssu}.png",
                compact=compact_plots,
            )
        _plot_adaptive_causal_timeline(
            decision_rows,
            boundary_slack_rows,
            timeline_rows,
            grant_component_rows,
            output_dir / "02_adaptive_causal_decision_timeline.png",
            compact=compact_plots,
        )
        _plot_layer_once_route_causality(
            route_rows,
            timeline_rows,
            output_dir / "02b_layer_once_route_causality.png",
            compact=compact_plots,
        )
        _plot_cumulative_metrics(
            window_rows,
            output_dir / "03_cumulative_multiscale_util_slo.png",
            compact=compact_plots,
        )
        _plot_disjoint_variability(
            window_rows,
            output_dir / "04_disjoint_window_variability.png",
            compact=compact_plots,
        )
        _plot_compute_decomposition(
            decomposition_rows,
            output_dir / "06_compute_inventory_decomposition.png",
            compact=compact_plots,
        )
        _plot_state_duration_partition(
            state_duration_summary,
            output_dir / "05_state_duration_partition.png",
            compact=compact_plots,
        )
        _plot_block_state_duration_timeline(
            block_state_duration_rows,
            output_dir / "05b_state_duration_timeline.png",
            compact=compact_plots,
        )
        _plot_dispatch_probes(
            probe_rows,
            output_dir / "07_micro_dispatch_probe.png",
            compact=compact_plots,
        )
        _plot_matched_barriers(
            matched_rows,
            output_dir / "08_matched_request_barrier_ecdf.png",
            compact=compact_plots,
        )
        for selection in forensic_selection_rows:
            num_ssu = int(selection["num_ssu"])
            _plot_forensic_zoom(
                selection,
                matched_layer_rows,
                forensic_timeline_rows,
                output_dir / f"09_forensic_zoom_ssu{num_ssu}.png",
                compact=compact_plots,
            )

    _assert_portable_text_outputs(output_dir)
    _assert_public_file_sizes(output_dir)
    _require_source_bytes_unchanged(
        analyzer_path, analyzer_sha256, "pre-manifest stability gate"
    )
    _require(
        _sha256(output_dir / "analyze_npu32_ssu345_timeline64.py")
        == analyzer_sha256,
        "published analyzer bytes differ from the executing analyzer",
    )

    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest_rows.append(
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema": "portable_analysis_manifest_v1",
        "analysis": "npu32_ssu345_timeline64_v1",
        "contains_hostname": False,
        "contains_token": False,
        "raw_inputs_included": False,
        "strict_max_file_size_bytes_exclusive": MAX_PUBLIC_FILE_BYTES,
        "analyzer_sha256": analyzer_sha256,
        "files": manifest_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _assert_public_file_sizes(output_dir)
    _require_source_bytes_unchanged(
        analyzer_path, analyzer_sha256, "post-manifest stability gate"
    )
    return {
        "validation": validation,
        "results": results,
        "output_dir": _publication_path(output_dir),
        "file_count": len(manifest_rows) + 1,
    }


def _synthetic_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_requests(case: str, num_ssu: int, start_ms: float) -> list[dict]:
    rows = []
    relative_improvement = {
        "baseline": 0.0,
        "layer_once_ttl_5ms": 0.18,
        "adaptive_t0_i100ms": 0.26,
    }[case]
    authenticated = _current_materialized_request_metadata(num_ssu)
    for npu_id in range(NUM_NPU):
        for sequence in range(1, 5):
            admission = start_ms + sequence * 6000.0
            request_id = sequence * NUM_NPU + npu_id
            expected = authenticated[request_id]
            ideal = float(expected["ideal_ttft_ms"])
            baseline_ttft = ideal * (2.05 if sequence % 4 == 0 else 1.58)
            ttft = baseline_ttft - relative_improvement * ideal
            if npu_id == 0 and sequence == 4:
                # Keep one authenticated request in an IO barrier through the
                # right boundary so the compact fixture exercises non-empty
                # direct physical queue/Path state without moving attribution
                # between NPU cells.
                ttft = 45_000.0
            barrier = ttft - ideal
            per_layer_compute = ideal / N_LAYERS
            io_start, io_ready, compute_start, compute_end, waits = [], [], [], [], []
            current = admission
            for layer in range(N_LAYERS):
                if layer == 0:
                    start_io = admission
                    ready_io = admission + barrier
                    start_compute = ready_io
                    wait = barrier
                else:
                    start_io = compute_start[-1]
                    ready_io = compute_end[-1]
                    start_compute = compute_end[-1]
                    wait = 0.0
                end_compute = start_compute + per_layer_compute
                io_start.append(start_io)
                io_ready.append(ready_io)
                compute_start.append(start_compute)
                compute_end.append(end_compute)
                waits.append(wait)
                current = end_compute
            _require(_close(current, admission + ttft, tolerance=2e-6), "synthetic layer construction failed")
            rows.append(
                {
                    "request_id": request_id,
                    "npu_id": npu_id,
                    "sequence": sequence,
                    "category": expected["category"],
                    "profile_id": expected["profile_id"],
                    "profile_name": expected["profile_name"],
                    "raw_demand_gbps": 10.0 + sequence % 4,
                    "admission_time_ms": admission,
                    "completion_time_ms": admission + ttft,
                    "ttft_ms": ttft,
                    "ideal_ttft_ms": ideal,
                    "io_barrier_ms": barrier,
                    "ttft_accounting_error_ms": 0.0,
                    "slo_met": ttft <= 2.0 * ideal,
                    "timeline_layers": {
                        "io_start_time_ms": io_start,
                        "io_ready_time_ms": io_ready,
                        "compute_start_ms": compute_start,
                        "compute_end_ms": compute_end,
                        "io_barrier_wait_ms": waits,
                        "per_layer_work_gb_by_ssu": list(
                            expected["per_layer_work_gb_by_ssu"]
                        ),
                    },
                }
            )
    return rows


def _synthetic_adaptive_demand(num_ssu: int) -> list[list[float]]:
    metadata = _current_materialized_request_metadata(num_ssu)
    return [
        [
            16.0 * float(metadata[NUM_NPU + npu]["per_layer_work_gb_by_ssu"][ssu])
            / (float(metadata[NUM_NPU + npu]["ideal_ttft_ms"]) / 1000.0)
            for ssu in range(num_ssu)
        ]
        for npu in range(NUM_NPU)
    ]


def _synthetic_adaptive_record(
    evaluation: int,
    time_ms: float,
    num_ssu: int,
    previous_selected_pairs: Sequence[Sequence[int]] = (),
    previous_selected_order: Sequence[int] = (),
) -> dict:
    request_sequence = 1
    request_by_npu = [
        [npu, request_sequence * NUM_NPU + npu] for npu in range(NUM_NPU)
    ]
    previous_pairs = [
        [int(item[0]), int(item[1])] for item in previous_selected_pairs
    ]
    previous_map = {npu: request_id for npu, request_id in previous_pairs}
    selected_order = [int(npu) for npu in previous_selected_order]
    _require(
        len(previous_map) == len(previous_pairs)
        and len(selected_order) == len(set(selected_order))
        and set(selected_order) == set(previous_map),
        "synthetic adaptive previous selected order/map mismatch",
    )
    current_request_map = {npu: request_id for npu, request_id in request_by_npu}
    pinned = [
        npu
        for npu in selected_order
        if current_request_map.get(npu) == previous_map[npu]
    ]
    request_admission_ms = 1000.0 + request_sequence * 6000.0
    metadata = _current_materialized_request_metadata(num_ssu)
    demands = _synthetic_adaptive_demand(num_ssu)
    remaining_layer_count = 15 if time_ms < request_admission_ms else N_LAYERS
    remaining_work = [
        [
            remaining_layer_count
            * float(
                metadata[request_sequence * NUM_NPU + npu][
                    "per_layer_work_gb_by_ssu"
                ][ssu]
            )
            for ssu in range(num_ssu)
        ]
        for npu in range(NUM_NPU)
    ]
    remaining_compute_s = [
        remaining_layer_count
        * float(metadata[request_sequence * NUM_NPU + npu]["ideal_ttft_ms"])
        / N_LAYERS
        / 1000.0
        for npu in range(NUM_NPU)
    ]
    replay = replay_admission_selection(
        demands,
        target_ratio=0.52,
        required_ratio=0.5,
        background_reserve_fraction=0.05,
        pinned_npu_ids=pinned,
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=NPU_LINK_CAP_GBPS,
    )
    allocation = allocate_adaptive_admission_grants(
        demands,
        explicit_spill_threshold=0.75,
        target_ratio=0.52,
        required_ratio=0.5,
        background_reserve_fraction=0.05,
        pinned_npu_ids=pinned,
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=NPU_LINK_CAP_GBPS,
    )
    v2 = allocation.v2_allocation
    return {
        "snapshot_time_ms": time_ms,
        "snapshot_evaluation": evaluation,
        "trigger_reasons": (
            ["initial", "batch_boundary"]
            if evaluation == 1
            else ["batch_boundary"]
        ),
        "layer_jobs_since_previous": 1,
        "request_by_npu": request_by_npu,
        "prefetch_only_by_npu": [
            [npu, bool(time_ms < request_admission_ms)]
            for npu in range(NUM_NPU)
        ],
        "remaining_work_gb_by_npu_ssu": remaining_work,
        "remaining_compute_s_by_npu": remaining_compute_s,
        "controller_demand_gbps_by_npu_ssu": demands,
        "previous_selected_request_by_npu": previous_pairs,
        "previous_pinned_npu_ids": pinned,
        "selection_mode": replay.selection_mode,
        "effective_target_ratio": replay.effective_target_ratio,
        "active_npu_ids": list(replay.active_npu_ids),
        "candidate_normalized_scores": [
            asdict(item) for item in replay.candidate_scores
        ],
        "candidate_order": list(replay.candidate_order),
        "admission_attempts": [asdict(item) for item in replay.attempts],
        "selected_npu_ids": list(replay.selected_npu_ids),
        "rejected_npu_ids": list(replay.rejected_npu_ids),
        "capacity_rejections": [
            asdict(item) for item in replay.capacity_rejections
        ],
        "selected_fraction": allocation.selected_fraction,
        "residual_mode": allocation.residual_mode,
        "grants_gbps_by_npu_ssu": [list(row) for row in allocation.grants_gbps],
        "v2_effective_floor_ratio": (
            None if v2 is None else v2.effective_floor_ratio
        ),
        "v2_floor_grants_gbps": (
            None if v2 is None else [list(row) for row in v2.floor_grants_gbps]
        ),
        "v2_background_grants_gbps": (
            None
            if v2 is None
            else [list(row) for row in v2.background_grants_gbps]
        ),
        "v2_selected_tail_grants_gbps": (
            None
            if v2 is None
            else [list(row) for row in v2.selected_tail_grants_gbps]
        ),
        "v2_spill_tail_grants_gbps": (
            None
            if v2 is None
            else [list(row) for row in v2.spill_tail_grants_gbps]
        ),
    }


def _synthetic_route_probes(
    case: str, num_ssu: int, start_ms: float
) -> list[dict]:
    """Build a tiny route fixture with an oracle independent of replay code.

    The authenticated request/SSU pairs below were chosen because they have
    the smallest non-empty placement slices for their NPU in each campaign.
    Baseline and Adaptive expectations follow directly from their one-Path
    rules.  The two Layer-once vectors per SSU were calculated by hand from
    the immutable pressure table (Path 12 has 20 queued commands; all other
    legal SL Paths are empty), start offset, equal registers, and the
    largest-block-first planning order.  Keeping these literal vectors is
    intentional: the fixture must not call the portable replay being tested
    to manufacture its own expected answer.
    """
    records: list[dict] = []
    metadata = _current_materialized_request_metadata(num_ssu)
    formal_qos = static_qos_config()
    formal_view = hardware_view(formal_qos)
    adaptive_grants = allocate_adaptive_admission_grants(
        _synthetic_adaptive_demand(num_ssu),
        explicit_spill_threshold=0.75,
        target_ratio=0.52,
        required_ratio=0.5,
        background_reserve_fraction=0.05,
        pinned_npu_ids=(),
        ssd_caps=SSD_CAP_GBPS,
        npu_caps=NPU_LINK_CAP_GBPS,
    ).grants_gbps
    route_fixture = {
        3: (294, 0),
        4: (4970, 3),
        5: (2349, 2),
    }
    layer_once_oracle = {
        3: (
            (204, 236, 45, 77, 109, 141, 173, 205, 237, 46, 78, 110,
             142, 174, 13, 206, 238, 47, 79, 111, 143, 175, 14, 207,
             239, 44, 76, 108, 140, 172, 15, 204, 236, 45, 77, 109,
             141, 173, 205, 237, 46, 78, 110, 142, 174, 206, 238, 47,
             79, 111, 143, 175, 207),
            (110, 142, 174, 206, 238, 47, 79, 111, 143, 175, 207, 239,
             44, 76, 15, 108, 140, 172, 204, 236, 45, 77, 13, 109,
             141, 173, 205, 237, 46, 78, 14, 110, 142, 174, 206, 238,
             47, 79, 111, 143, 175, 207, 239, 44, 76, 108, 140, 172,
             204, 236, 45, 77, 109),
        ),
        4: (
            (44, 76, 108, 140, 172, 204, 236, 45, 77, 109, 141, 173,
             205, 237, 13, 46, 78, 110, 142, 174, 206, 238, 14, 47,
             79, 111, 143, 175, 207, 239, 15, 44, 76, 108, 140),
            (205, 237, 46, 78, 110, 142, 174, 206, 238, 47, 79, 111,
             143, 175, 14, 207, 239, 44, 76, 108, 140, 172, 15, 204,
             236, 45, 77, 109, 141, 173, 13, 205, 237, 46, 78),
        ),
        5: (
            (236, 45, 77, 109, 141, 173, 205, 237, 46, 78, 110, 142,
             174, 206, 13, 238, 47, 79, 111, 143, 175, 207, 14, 239,
             44, 76, 108),
            (142, 174, 206, 238, 47, 79, 111, 143, 175, 207, 239, 44,
             76, 108, 15, 140, 172, 204, 236, 45, 77, 109, 13, 141,
             173, 205, 237),
        ),
    }
    request_id, ssu_id = route_fixture[num_ssu]
    request = metadata[request_id]
    npu_id = int(request["npu_id"])
    for index in range(2):
        request = metadata[request_id]
        time_ms = start_ms + index * 2.0
        category = str(request["category"])
        layer = index
        static_allowed = list(category_path_ids(category, formal_view))
        if case == "baseline":
            rule = "fixed_path_zero"
            pressure_source = None
            allowed_paths = static_allowed
        elif case == "adaptive_t0_i100ms":
            rule = "npu_dedicated_path"
            pressure_source = None
            allowed_paths = [cold_start_hybrid_path_id(npu_id)]
        else:
            rule = "pressure_aware_once_per_layer"
            pressure_source = "fresh" if index % 2 == 0 else "cache"
            allowed_paths = static_allowed
        pressure_age = (
            None
            if pressure_source is None
            else 0.0
            if pressure_source == "fresh"
            else 1.0
        )
        allowed_cirs = (
            [float(adaptive_grants[npu_id][ssu_id])]
            if case == "adaptive_t0_i100ms"
            else [float(formal_qos.path_cirs[path]) for path in allowed_paths]
        )
        allowed_weights = [
            float(formal_qos.path_weights[path]) for path in allowed_paths
        ]
        pressure_counts = None
        group_io = None
        active_paths = None
        active_weights = None
        active_group_weight_sum = None
        active_cir_sum = None
        if pressure_source is not None:
            pressure_counts = [0] * len(allowed_paths)
            pressure_counts[0] = 20
            group_io = [0] * EXPECTED_PATH_ABI["group_count"]
            active_paths = [0] * EXPECTED_PATH_ABI["group_count"]
            active_weights = [0.0] * EXPECTED_PATH_ABI["group_count"]
            for path, count, cir, weight in zip(
                allowed_paths, pressure_counts, allowed_cirs, allowed_weights
            ):
                if count <= 0:
                    continue
                group = path // EXPECTED_PATH_ABI["paths_per_group"]
                group_io[group] += count
                active_paths[group] += 1
                active_weights[group] += weight
            active_group_weight_sum = math.fsum(
                formal_qos.group_weights[group]
                for group, count in enumerate(active_paths)
                if count > 0
            )
            active_cir_sum = math.fsum(
                cir for cir, count in zip(allowed_cirs, pressure_counts) if count > 0
            )
        authenticated_blocks = request["placement_blocks_by_ssu"][ssu_id]
        record = {
                "time_ms": time_ms,
                "rule": rule,
                "npu_id": npu_id,
                "ssu_id": ssu_id,
                "request_id": request_id,
                "layer": layer,
                "category": category,
                "start_offset": (
                    request_id + 13 * layer + 29 * ssu_id
                )
                % len(allowed_paths),
                "pressure_source": pressure_source,
                "pressure_snapshot_time_ms": (
                    None if pressure_age is None else time_ms - pressure_age
                ),
                "pressure_age_ms": pressure_age,
                "pressure_ttl_ms": (
                    5.0 if case == "layer_once_ttl_5ms" else 0.0
                ),
                "allowed_path_ids": allowed_paths,
                "allowed_path_pressure_counts": pressure_counts,
                "allowed_path_cir_gbps": allowed_cirs,
                "allowed_path_pir_gbps_or_null": [None] * len(allowed_paths),
                "allowed_path_weights": allowed_weights,
                "allowed_path_group_ids": [path // 32 for path in allowed_paths],
                "disk_bw_gbps": SSD_CAP_GBPS,
                "path_count": 256,
                "paths_per_group": 32,
                "group_weights": [1.0] * 8,
                "group_io_counts": group_io,
                "active_paths_per_group": active_paths,
                "active_path_weights": active_weights,
                "active_group_weight_sum": active_group_weight_sum,
                "active_cir_sum_gbps": active_cir_sum,
                "block_indices": [int(block) for block, _size in authenticated_blocks],
                "block_sizes_gb": [float(size) for _block, size in authenticated_blocks],
                "selected_path_ids": [],
                "selected_group_ids": [],
            }
        if case == "baseline":
            selected = (0,) * len(authenticated_blocks)
        elif case == "adaptive_t0_i100ms":
            selected = (cold_start_hybrid_path_id(npu_id),) * len(
                authenticated_blocks
            )
        else:
            selected = layer_once_oracle[num_ssu][index]
            _require(
                len(selected) == len(authenticated_blocks),
                "synthetic literal route oracle length mismatch",
            )
        record["selected_path_ids"] = list(selected)
        record["selected_group_ids"] = [
            path // EXPECTED_PATH_ABI["paths_per_group"] for path in selected
        ]
        records.append(record)
    return records


def _synthetic_summary(case: str, num_ssu: int, simulator_hash: str) -> dict:
    start_ms = 1000.0
    requests = _synthetic_requests(case, num_ssu, start_ms)
    authenticated_requests = _current_materialized_request_metadata(num_ssu)
    carry_request_id = 0
    carry_metadata = authenticated_requests[carry_request_id]
    carry_ideal = float(carry_metadata["ideal_ttft_ms"])
    carry_admission = start_ms - 0.5 * carry_ideal
    carry_per_layer_compute = carry_ideal / N_LAYERS
    carry_io_ready: list[float] = []
    carry_compute_start: list[float] = []
    carry_compute_end: list[float] = []
    carry_wait: list[float] = []
    carry_previous_end = carry_admission
    for layer in range(N_LAYERS):
        ready = start_ms if layer == 0 else carry_previous_end
        compute_start = max(carry_previous_end, ready)
        compute_end = compute_start + carry_per_layer_compute
        carry_io_ready.append(ready)
        carry_compute_start.append(compute_start)
        carry_compute_end.append(compute_end)
        carry_wait.append(compute_start - carry_previous_end)
        carry_previous_end = compute_end
    synthetic_carry_rows = [
        {
            "batch_id": carry_request_id,
            "request_ids": [carry_request_id],
            "npu_id": 0,
            "admission_time_ms": carry_admission,
            "completion_time_ms": carry_previous_end,
            "layer_count": N_LAYERS,
            "per_layer_compute_ms": carry_per_layer_compute,
            "ideal_compute_ms": carry_ideal,
            "io_ready_time_ms": carry_io_ready,
            "compute_start_ms": carry_compute_start,
            "compute_end_ms": carry_compute_end,
            "compute_duration_ms": [carry_per_layer_compute] * N_LAYERS,
            "io_barrier_wait_ms": carry_wait,
        }
    ]
    equality_request_id = 2
    equality_metadata = authenticated_requests[equality_request_id]
    equality_ideal = float(equality_metadata["ideal_ttft_ms"])
    equality_per_layer = equality_ideal / N_LAYERS
    equality_admission = start_ms - equality_ideal
    equality_starts = [
        equality_admission + layer * equality_per_layer
        for layer in range(N_LAYERS)
    ]
    equality_ends = [value + equality_per_layer for value in equality_starts]
    synthetic_carry_rows.append(
        {
            "batch_id": equality_request_id,
            "request_ids": [equality_request_id],
            "npu_id": 2,
            "admission_time_ms": equality_admission,
            "completion_time_ms": start_ms,
            "layer_count": N_LAYERS,
            "per_layer_compute_ms": equality_per_layer,
            "ideal_compute_ms": equality_ideal,
            "io_ready_time_ms": list(equality_starts),
            "compute_start_ms": list(equality_starts),
            "compute_end_ms": list(equality_ends),
            "compute_duration_ms": [equality_per_layer] * N_LAYERS,
            "io_barrier_wait_ms": [0.0] * N_LAYERS,
        }
    )
    requests = [
        row
        for row in requests
        if not (int(row["npu_id"]) == 0 and int(row["sequence"]) == 0)
    ]
    synthetic_compute_blocks = [[0.0] * NUM_NPU for _ in range(BLOCK_COUNT)]
    synthetic_barrier_blocks = [[0.0] * NUM_NPU for _ in range(BLOCK_COUNT)]
    synthetic_lifecycles: list[dict] = list(requests)
    synthetic_lifecycles.append(
        {
            "request_id": carry_request_id,
            "npu_id": 0,
            "sequence": int(carry_metadata["sequence"]),
            "category": carry_metadata["category"],
            "profile_id": carry_metadata["profile_id"],
            "profile_name": carry_metadata["profile_name"],
            "admission_time_ms": carry_admission,
            "completion_time_ms": carry_previous_end,
            "ideal_ttft_ms": carry_ideal,
            "timeline_layers": {
                "io_ready_time_ms": carry_io_ready,
                "compute_start_ms": carry_compute_start,
                "compute_end_ms": carry_compute_end,
            },
        }
    )
    synthetic_lifecycles.append(
        {
            "request_id": equality_request_id,
            "npu_id": 2,
            "sequence": int(equality_metadata["sequence"]),
            "category": equality_metadata["category"],
            "profile_id": equality_metadata["profile_id"],
            "profile_name": equality_metadata["profile_name"],
            "admission_time_ms": equality_admission,
            "completion_time_ms": start_ms,
            "ideal_ttft_ms": equality_ideal,
            "timeline_layers": {
                "io_ready_time_ms": list(equality_starts),
                "compute_start_ms": list(equality_starts),
                "compute_end_ms": list(equality_ends),
            },
        }
    )
    for request in synthetic_lifecycles:
        npu_id = int(request["npu_id"])
        previous_end = float(request["admission_time_ms"])
        layers = request["timeline_layers"]
        for compute_start, compute_end in zip(
            layers["compute_start_ms"], layers["compute_end_ms"]
        ):
            for block_index in range(BLOCK_COUNT):
                block_start = start_ms + block_index * BLOCK_MS
                block_end = block_start + BLOCK_MS
                synthetic_barrier_blocks[block_index][npu_id] += (
                    _interval_overlap_ms(
                        previous_end, float(compute_start), block_start, block_end
                    )
                )
                synthetic_compute_blocks[block_index][npu_id] += (
                    _interval_overlap_ms(
                        float(compute_start),
                        float(compute_end),
                        block_start,
                        block_end,
                    )
                )
            previous_end = float(compute_end)
    if case == "adaptive_t0_i100ms":
        synthetic_allocation = allocate_adaptive_admission_grants(
            _synthetic_adaptive_demand(num_ssu),
            explicit_spill_threshold=0.75,
            target_ratio=0.52,
            required_ratio=0.5,
            background_reserve_fraction=0.05,
            pinned_npu_ids=(),
            ssd_caps=SSD_CAP_GBPS,
            npu_caps=NPU_LINK_CAP_GBPS,
        )
        installed = [list(row) for row in synthetic_allocation.grants_gbps]
    else:
        installed = [[0.0] * num_ssu for _ in range(NUM_NPU)]
    compute_cumulative = [0.0] * NUM_NPU
    ssd_cumulative = [[0.0] * num_ssu for _ in range(NUM_NPU)]
    link_cumulative = [[0.0] * num_ssu for _ in range(NUM_NPU)]
    blocks = []
    boundaries = []

    def make_boundary(index: int) -> dict:
        time_ms = start_ms + index * BLOCK_MS
        controller_remaining = [0.0] * NUM_NPU
        controller_demand = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        physical_demand = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        controller_remaining_gb = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        physical_remaining_gb = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        ssd_outstanding = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        ssd_outstanding_blocks = [[0] * num_ssu for _ in range(NUM_NPU)]
        link_outstanding = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        link_outstanding_blocks = [[0] * num_ssu for _ in range(NUM_NPU)]
        npu_rows = []
        sequence = min(15, index // 8)
        for npu in range(NUM_NPU):
            active_candidates = [
                lifecycle
                for lifecycle in synthetic_lifecycles
                if int(lifecycle["npu_id"]) == npu
                and float(lifecycle["admission_time_ms"]) < time_ms
                and time_ms <= float(lifecycle["completion_time_ms"]) + TOL
            ]
            _require(
                len(active_candidates) <= 1,
                "synthetic NPU lifecycles overlap at a boundary",
            )
            if not active_candidates:
                npu_rows.append(
                    {
                        "npu_id": npu,
                        "pipeline_state": (
                            "waiting_arrival"
                            if time_ms < start_ms + MEASUREMENT_MS
                            else "drained"
                        ),
                        "compute_inventory_q_ms": 0.0,
                        "activated_compute_cumulative_ms": math.fsum(
                            float(lifecycle["ideal_ttft_ms"])
                            for lifecycle in synthetic_lifecycles
                            if int(lifecycle["npu_id"]) == npu
                            and float(lifecycle["admission_time_ms"]) < time_ms
                        ),
                        "active_request_id": None,
                        "active_batch_id": None,
                        "current_compute_layer": None,
                        "next_compute_layer": None,
                        "compute_done_up_to": None,
                        "compute_start_ms": None,
                        "compute_end_ms": None,
                        "next_compute_layer_io_ready": None,
                        "waiting_on_io_layer": None,
                        "prefetch_request_id": None,
                        "controller_request_id": None,
                        "controller_prefetch_only": None,
                        "admission_time_ms": None,
                        "elapsed_ttft_ms": None,
                        "ideal_ttft_ms": None,
                        "slo_alpha1p5_slack_ms": None,
                        "slo_alpha2_slack_ms": None,
                        "category": None,
                        "sequence": None,
                        "profile_id": None,
                        "profile_name": None,
                        "raw_demand_gbps": None,
                    }
                )
                continue
            lifecycle = active_candidates[0]
            request_id = int(lifecycle["request_id"])
            request_metadata = authenticated_requests[request_id]
            admission = float(lifecycle["admission_time_ms"])
            elapsed = max(0.0, time_ms - admission)
            ideal = float(lifecycle["ideal_ttft_ms"])
            layer_vectors = lifecycle["timeline_layers"]
            current_layer = None
            waiting_layer = None
            done_up_to = -1
            pipeline_state = "io_barrier"
            for layer, (layer_start, layer_end) in enumerate(
                zip(
                    layer_vectors["compute_start_ms"],
                    layer_vectors["compute_end_ms"],
                )
            ):
                layer_start = float(layer_start)
                layer_end = float(layer_end)
                prior_end = (
                    admission
                    if layer == 0
                    else float(layer_vectors["compute_end_ms"][layer - 1])
                )
                if layer_start < time_ms <= layer_end + TOL:
                    pipeline_state = "compute"
                    current_layer = layer
                    done_up_to = layer - 1
                    break
                if prior_end < time_ms <= layer_start + TOL:
                    pipeline_state = "io_barrier"
                    waiting_layer = layer
                    done_up_to = layer - 1
                    break
            if pipeline_state == "compute":
                next_layer = (
                    current_layer + 1
                    if current_layer is not None
                    and current_layer + 1 < N_LAYERS
                    else None
                )
                next_ready = (
                    None
                    if next_layer is None
                    else time_ms
                    > float(layer_vectors["io_ready_time_ms"][next_layer])
                )
                remaining_current = max(
                    0.0,
                    float(layer_vectors["compute_end_ms"][current_layer])
                    - time_ms,
                )
                q_ms = remaining_current + (
                    N_LAYERS - int(current_layer) - 1
                ) * (ideal / N_LAYERS)
                remaining_layer_count = N_LAYERS - int(current_layer)
            else:
                _require(
                    waiting_layer is not None,
                    "synthetic active lifecycle has no boundary state",
                )
                next_layer = waiting_layer
                next_ready = False
                q_ms = (N_LAYERS - int(waiting_layer)) * (ideal / N_LAYERS)
                remaining_layer_count = N_LAYERS - int(waiting_layer)
            controller_remaining[npu] = remaining_layer_count * (
                ideal / N_LAYERS
            )
            per_layer_work = request_metadata["per_layer_work_gb_by_ssu"]
            controller_remaining_gb[npu] = [
                remaining_layer_count * float(value) for value in per_layer_work
            ]
            physical_remaining_gb[npu] = [
                0.9 * value for value in controller_remaining_gb[npu]
            ]
            compute_s = controller_remaining[npu] / 1000.0
            controller_demand[npu] = [
                value / compute_s for value in controller_remaining_gb[npu]
            ]
            physical_demand[npu] = [
                value / compute_s for value in physical_remaining_gb[npu]
            ]
            npu_rows.append(
                {
                    "npu_id": npu,
                    "pipeline_state": pipeline_state,
                    "compute_inventory_q_ms": q_ms,
                    "activated_compute_cumulative_ms": math.fsum(
                        float(item["ideal_ttft_ms"])
                        for item in synthetic_lifecycles
                        if int(item["npu_id"]) == npu
                        and float(item["admission_time_ms"]) < time_ms
                    ),
                    "active_request_id": request_id,
                    "active_batch_id": request_id,
                    "current_compute_layer": current_layer,
                    "next_compute_layer": next_layer,
                    "compute_done_up_to": done_up_to,
                    "compute_start_ms": (
                        float(layer_vectors["compute_start_ms"][current_layer])
                        if current_layer is not None
                        else None
                    ),
                    "compute_end_ms": (
                        float(layer_vectors["compute_end_ms"][current_layer])
                        if current_layer is not None
                        else None
                    ),
                    "next_compute_layer_io_ready": next_ready,
                    "waiting_on_io_layer": waiting_layer,
                    "prefetch_request_id": None,
                    "controller_request_id": request_id,
                    "controller_prefetch_only": False,
                    "admission_time_ms": admission,
                    "elapsed_ttft_ms": elapsed,
                    "ideal_ttft_ms": ideal,
                    "slo_alpha1p5_slack_ms": 1.5 * ideal - elapsed,
                    "slo_alpha2_slack_ms": 2.0 * ideal - elapsed,
                    "category": request_metadata["category"],
                    "sequence": request_metadata["sequence"],
                    "profile_id": request_metadata["profile_id"],
                    "profile_name": request_metadata["profile_name"],
                    "raw_demand_gbps": 10.0 + int(request_metadata["sequence"]) % 4,
                }
            )
        queue_owner_by_ssu: list[int | None] = []
        queue_command_by_ssu: list[tuple[int, int, float] | None] = []
        for ssu_id in range(num_ssu):
            candidates = [
                npu_id
                for npu_id, row in enumerate(npu_rows)
                if npu_id == 0
                and row["active_request_id"] == 4 * NUM_NPU
                and physical_remaining_gb[npu_id][ssu_id] > TOL
                and authenticated_requests[int(row["active_request_id"])][
                    "placement_blocks_by_ssu"
                ][ssu_id]
            ]
            if not candidates:
                queue_owner_by_ssu.append(None)
                queue_command_by_ssu.append(None)
                continue
            owner = candidates[(index + ssu_id) % len(candidates)]
            request_id = int(npu_rows[owner]["active_request_id"])
            block_idx, block_gb = authenticated_requests[request_id][
                "placement_blocks_by_ssu"
            ][ssu_id][0]
            remaining_gb = float(block_gb)
            _require(remaining_gb > 0.0, "synthetic queue remainder is empty")
            ssd_outstanding[owner][ssu_id] = remaining_gb
            ssd_outstanding_blocks[owner][ssu_id] = 1
            queue_owner_by_ssu.append(owner)
            queue_command_by_ssu.append(
                (request_id, int(block_idx), remaining_gb)
            )
        timeline_cells = {
            "ssd_enqueued_cumulative_gb": [[ssd_cumulative[npu][ssu] + ssd_outstanding[npu][ssu] for ssu in range(num_ssu)] for npu in range(NUM_NPU)],
            "ssd_served_cumulative_gb": [list(row) for row in ssd_cumulative],
            "ssd_served_fragmented_diagnostic_cumulative_gb": [
                list(row) for row in ssd_cumulative
            ],
            "ssd_outstanding_gb": ssd_outstanding,
            "ssd_outstanding_blocks": ssd_outstanding_blocks,
            "link_enqueued_cumulative_gb": [list(row) for row in link_cumulative],
            "link_served_cumulative_gb": [list(row) for row in link_cumulative],
            "link_outstanding_gb": link_outstanding,
            "link_outstanding_blocks": link_outstanding_blocks,
            "ssd_served_awaiting_link_enqueue_gb": [[0.0] * num_ssu for _ in range(NUM_NPU)],
            "client_unissued_gb": [[0.0] * num_ssu for _ in range(NUM_NPU)],
            "activated_io_cumulative_gb": [
                [link_cumulative[npu][ssu] + ssd_outstanding[npu][ssu] for ssu in range(num_ssu)]
                for npu in range(NUM_NPU)
            ],
            "physical_remaining_gb": physical_remaining_gb,
            "controller_declared_remaining_gb": controller_remaining_gb,
            "controller_remaining_compute_ms": controller_remaining,
            "physical_demand_gbps": physical_demand,
            "controller_demand_gbps": controller_demand,
            "installed_dedicated_path_cir_gbps": [list(row) for row in installed] if case == "adaptive_t0_i100ms" else None,
            "route_plans_cumulative": [[index for _ in range(num_ssu)] for _ in range(NUM_NPU)],
            "route_pressure_fresh_cumulative": [[index // 2 if case == "layer_once_ttl_5ms" else 0 for _ in range(num_ssu)] for _ in range(NUM_NPU)],
            "route_pressure_cache_cumulative": [[index - index // 2 if case == "layer_once_ttl_5ms" else 0 for _ in range(num_ssu)] for _ in range(NUM_NPU)],
            "route_blocks_by_group_cumulative": [
                [
                    [
                        index
                        if group_id
                        == (
                            cold_start_hybrid_path_id(npu)
                            // EXPECTED_PATH_ABI["paths_per_group"]
                            if case == "adaptive_t0_i100ms"
                            else 0
                        )
                        else 0
                        for group_id in range(EXPECTED_PATH_ABI["group_count"])
                    ]
                    for _ in range(num_ssu)
                ]
                for npu in range(NUM_NPU)
            ],
        }
        ssd_by_ssu = [math.fsum(ssd_cumulative[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)]
        link_by_npu = [math.fsum(link_cumulative[npu]) for npu in range(NUM_NPU)]
        active_commands = []
        sparse_path_rows = []
        for ssu in range(num_ssu):
            owner = queue_owner_by_ssu[ssu]
            command_tuple = queue_command_by_ssu[ssu]
            if owner is None or command_tuple is None:
                active_commands.append(None)
                continue
            command_request_id, command_block_idx, command_remaining_gb = (
                command_tuple
            )
            active_commands.append(
                {
                    "ssu_id": ssu,
                    "npu_id": owner,
                    "request_id": command_request_id,
                    "layer": index % N_LAYERS,
                    "block_idx": int(command_block_idx),
                    "path_id": (
                        cold_start_hybrid_path_id(owner)
                        if case == "adaptive_t0_i100ms"
                        else 0
                    ),
                    "remaining_gb": command_remaining_gb,
                    "command_start_time_ms": time_ms,
                    "command_age_ms": 0.0,
                    "physical_service_gbps": SSD_CAP_GBPS,
                    "non_preemptive": True,
                }
            )
            path_id = int(active_commands[-1]["path_id"])
            sparse_path_rows.append(
                {
                    "ssu_id": ssu,
                    "path_id": path_id,
                    "group_id": path_id // EXPECTED_PATH_ABI["paths_per_group"],
                    "cir_gbps": (
                        installed[owner][ssu]
                        if case == "adaptive_t0_i100ms"
                        else float(static_qos_config().path_cirs[path_id])
                    ),
                    "path_weight": float(
                        static_qos_config().path_weights[path_id]
                    ),
                    "virtual_finish": float(index),
                    "estimated_next_arbitration_rate_gbps": 0.0,
                    "pending_blocks": 0,
                    "pending_gb": 0.0,
                    "active_remaining_gb": command_remaining_gb,
                    "head_wait_age_ms": None,
                    "head_npu_id": None,
                    "head_request_id": None,
                    "head_layer": None,
                }
            )
        return {
            "boundary": index,
            "time_ms": time_ms,
            "ssd_cumulative_busy_ms_by_ssu": [value / SSD_CAP_GBPS * 1000.0 for value in ssd_by_ssu],
            "ssd_cumulative_served_gb_by_ssu": ssd_by_ssu,
            "ssd_outstanding_blocks_by_ssu": [
                sum(ssd_outstanding_blocks[npu][ssu] for npu in range(NUM_NPU))
                for ssu in range(num_ssu)
            ],
            "ssd_outstanding_gb_by_ssu": [
                math.fsum(ssd_outstanding[npu][ssu] for npu in range(NUM_NPU))
                for ssu in range(num_ssu)
            ],
            "npu_compute_cumulative_busy_ms_by_npu": list(compute_cumulative),
            "npu_link_cumulative_busy_ms_by_npu": [value / NPU_LINK_CAP_GBPS * 1000.0 for value in link_by_npu],
            "npu_link_cumulative_served_gb_by_npu": link_by_npu,
            "npu_link_outstanding_blocks_by_npu": [0] * NUM_NPU,
            "npu_link_outstanding_gb_by_npu": [0.0] * NUM_NPU,
            "timeline": {
                "schema": TIMELINE_SCHEMA,
                "ssd_accounting_residuals_by_ssu": [
                    {
                        "ssu_id": ssu,
                        "stable_service_minus_busy_counter_gb": 0.0,
                        "fragmented_service_minus_stable_gb": 0.0,
                        "physical_queue_minus_scheduler_gb": 0.0,
                        "enqueue_minus_service_minus_physical_queue_gb": 0.0,
                        "counter_queue_minus_physical_queue_gb": 0.0,
                        "fragmented_counter_queue_minus_physical_queue_gb": 0.0,
                        "maximum_abs_npu_queue_identity_residual_gb": 0.0,
                        "maximum_abs_npu_queue_identity_residual_npu_id": 0,
                        "physical_queue_block_minus_scheduler_blocks": 0,
                        "counter_queue_block_minus_physical_blocks": 0,
                    }
                    for ssu in range(num_ssu)
                ],
                "npu_rows": npu_rows,
                "npu_ssu": timeline_cells,
                "sparse_ssu_path_rows": sparse_path_rows,
                "active_command_by_ssu": active_commands,
                "pressure_state_by_ssu": [
                    {
                        "ssu_id": ssu,
                        "reports_cumulative": (
                            NUM_NPU * (index // 2)
                            if case == "layer_once_ttl_5ms"
                            else 0
                        ),
                        "cache_hits_cumulative": (
                            NUM_NPU * (index - index // 2)
                            if case == "layer_once_ttl_5ms"
                            else 0
                        ),
                        "cache_time_ms": (
                            time_ms if case == "layer_once_ttl_5ms" else None
                        ),
                        "cache_age_ms": (
                            0.0 if case == "layer_once_ttl_5ms" else None
                        ),
                        "ttl_ms": 5.0 if case == "layer_once_ttl_5ms" else 0.0,
                    }
                    for ssu in range(num_ssu)
                ],
            },
        }

    boundaries.append(make_boundary(0))
    for block in range(BLOCK_COUNT):
        compute_delta = list(synthetic_compute_blocks[block])
        for npu in range(NUM_NPU):
            compute_cumulative[npu] += compute_delta[npu]
        ssd_delta = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        link_delta = [[0.0] * num_ssu for _ in range(NUM_NPU)]
        for ssu in range(num_ssu):
            ssd_owner = (block * 7 + ssu * 5) % NUM_NPU
            ssd_delta[ssd_owner][ssu] = 15.0
            ssd_cumulative[ssd_owner][ssu] += 15.0
            if block > 0:
                link_owner = ((block - 1) * 7 + ssu * 5) % NUM_NPU
                link_delta[link_owner][ssu] = 15.0
                link_cumulative[link_owner][ssu] += 15.0
        ssd_service = [math.fsum(ssd_delta[npu][ssu] for npu in range(NUM_NPU)) for ssu in range(num_ssu)]
        link_service = [math.fsum(link_delta[npu]) for npu in range(NUM_NPU)]
        ssd_busy = [value / SSD_CAP_GBPS * 1000.0 for value in ssd_service]
        link_busy = [value / NPU_LINK_CAP_GBPS * 1000.0 for value in link_service]
        block_start = start_ms + block * BLOCK_MS
        block_end = block_start + BLOCK_MS
        admitted = [row for row in requests if block_start <= float(row["admission_time_ms"]) < block_end]
        blocks.append(
            {
                "block": block,
                "start_ms": block_start,
                "end_ms": block_end,
                "duration_ms": BLOCK_MS,
                "npu_utilization": math.fsum(compute_delta) / (NUM_NPU * BLOCK_MS),
                "request_count": len(admitted),
                "request_weighted_slo_attainment": statistics.fmean(bool(row["slo_met"]) for row in admitted) if admitted else None,
                "ssd_busy_ms_by_ssu": ssd_busy,
                "ssd_served_gb_by_ssu": ssd_service,
                "ssd_utilizations": [value / BLOCK_MS for value in ssd_busy],
                "ssd_mean_utilization": statistics.fmean(value / BLOCK_MS for value in ssd_busy),
                "compute_ms_by_npu": compute_delta,
                "npu_utilizations": [value / BLOCK_MS for value in compute_delta],
                "npu_link_busy_ms_by_npu": link_busy,
                "npu_link_served_gb_by_npu": link_service,
                "npu_link_utilizations": [value / BLOCK_MS for value in link_busy],
                "npu_link_mean_utilization": statistics.fmean(value / BLOCK_MS for value in link_busy),
            }
        )
        boundaries.append(make_boundary(block + 1))

    ttfts = [float(row["ttft_ms"]) for row in requests]
    outcomes = [bool(row["slo_met"]) for row in requests]
    request_counts = [
        sum(int(row["npu_id"]) == npu_id for row in requests)
        for npu_id in range(NUM_NPU)
    ]
    compute_total = [float(boundaries[-1]["npu_compute_cumulative_busy_ms_by_npu"][npu]) for npu in range(NUM_NPU)]
    whole_ssd = _matrix_delta(
        boundaries[0]["timeline"]["npu_ssu"]["ssd_served_cumulative_gb"],
        boundaries[-1]["timeline"]["npu_ssu"]["ssd_served_cumulative_gb"],
    )
    whole_link = _matrix_delta(
        boundaries[0]["timeline"]["npu_ssu"]["link_served_cumulative_gb"],
        boundaries[-1]["timeline"]["npu_ssu"]["link_served_cumulative_gb"],
    )
    measurement_ssd_accounting_residuals = {
        "schema": SSD_ACCOUNTING_SCHEMA,
        "service_absolute_tolerance_gb": SSD_SERVICE_ABSOLUTE_TOLERANCE_GB,
        "block_tolerance": 0,
        "decimal_bytes_per_gb": 1e9,
        "stable_service_minus_busy_counter_gb_by_ssu": [0.0] * num_ssu,
        "stable_service_minus_busy_counter_decimal_bytes_by_ssu": [
            0.0
        ]
        * num_ssu,
        "fragmented_service_minus_stable_gb_by_ssu": [0.0] * num_ssu,
        "fragmented_service_minus_stable_decimal_bytes_by_ssu": [0.0]
        * num_ssu,
        "busy_time_compensation_ms_by_ssu_at_stop": [0.0] * num_ssu,
        "measurement_maxima": {
            name: {
                "signed_gb": 0.0,
                "signed_decimal_bytes": 0.0,
                "absolute_gb": 0.0,
                "absolute_decimal_bytes": 0.0,
                "ssu_id": 0,
            }
            for name in (
                "stable_service_minus_busy_counter",
                "fragmented_service_minus_stable",
            )
        },
        "timeline_maxima": {
            **{
                field: {
                    "signed_gb": 0.0,
                    "signed_decimal_bytes": 0.0,
                    "absolute_gb": 0.0,
                    "absolute_decimal_bytes": 0.0,
                    "boundary": 0,
                    "elapsed_ms": 0.0,
                    "ssu_id": 0,
                    **(
                        {"npu_id": 0}
                        if field
                        == "maximum_abs_npu_queue_identity_residual_gb"
                        else {}
                    ),
                }
                for field in (
                    "stable_service_minus_busy_counter_gb",
                    "fragmented_service_minus_stable_gb",
                    "physical_queue_minus_scheduler_gb",
                    "enqueue_minus_service_minus_physical_queue_gb",
                    "counter_queue_minus_physical_queue_gb",
                    "fragmented_counter_queue_minus_physical_queue_gb",
                    "maximum_abs_npu_queue_identity_residual_gb",
                )
            },
            **{
                field: {
                    "signed_blocks": 0,
                    "absolute_blocks": 0,
                    "boundary": 0,
                    "elapsed_ms": 0.0,
                    "ssu_id": 0,
                }
                for field in (
                    "physical_queue_block_minus_scheduler_blocks",
                    "counter_queue_block_minus_physical_blocks",
                )
            },
        },
    }
    probes = []
    authenticated_probe_requests = _current_materialized_request_metadata(
        num_ssu
    )
    for index in range(20):
        winner_npu = index % NUM_NPU
        winner_request = authenticated_probe_requests[winner_npu]
        winner_ssus = [
            ssu_id
            for ssu_id, placement in winner_request[
                "placement_blocks_by_ssu"
            ].items()
            if placement
        ]
        winner_ssu = winner_ssus[index % len(winner_ssus)]
        winner_block_idx, winner_command_gb = winner_request[
            "placement_blocks_by_ssu"
        ][winner_ssu][0]
        winner_path = (
            cold_start_hybrid_path_id(winner_npu)
            if case == "adaptive_t0_i100ms"
            else 0
        )
        probes.append({
            "ssu_id": winner_ssu,
            "time_ms": start_ms + index * 2.0,
            "rr_cursor_before": index % 4,
            "candidate_path_count": 2 + index % 5,
            "minimum_finish_tag": 1.0 + index,
            "finish_tie_count": 1,
            "expected_path_id": winner_path,
            "winner_finish_tag_before": 1.0 + index,
            "winner_estimated_arbitration_rate_gbps": 10.0,
            "winner_cir_gbps": (
                float(static_qos_config().path_cirs[winner_path])
                if case != "adaptive_t0_i100ms"
                else float(installed[winner_npu][winner_ssu])
            ),
            "winner_group_id": winner_path // EXPECTED_PATH_ABI["paths_per_group"],
            "winner_path_weight": float(
                static_qos_config().path_weights[winner_path]
            ),
            "winner_pending_blocks_before": 4,
            "winner_pending_gb_before": max(0.4, float(winner_command_gb)),
            "selection_rule": "minimum_virtual_finish_then_round_robin",
            "actual_path_id": winner_path,
            "winner_npu_id": winner_npu,
            "winner_request_id": winner_npu,
            "winner_layer": index % N_LAYERS,
            "winner_block_idx": int(winner_block_idx),
            "winner_queue_wait_ms": 0.1 * index,
            "winner_command_gb": float(winner_command_gb),
            "winner_virtual_finish_after": 1.0 + index,
            "physical_command_service_gbps": SSD_CAP_GBPS,
            "physical_command_non_preemptive": True,
            "prediction_matches_actual": True,
        })
    diagnostics = []
    if case == "adaptive_t0_i100ms":
        previous_selected_pairs: list[list[int]] = []
        previous_selected_order: list[int] = []
        decision_times = [start_ms - 100.0] + [
            start_ms + index * BLOCK_MS + 100.0
            for index in range(BLOCK_COUNT)
        ]
        for evaluation, decision_time in enumerate(decision_times, start=1):
            record = _synthetic_adaptive_record(
                evaluation,
                decision_time,
                num_ssu,
                previous_selected_pairs,
                previous_selected_order,
            )
            diagnostics.append(record)
            request_map = {
                int(npu): int(request_id)
                for npu, request_id in record["request_by_npu"]
            }
            previous_selected_pairs = [
                [npu, request_map[npu]]
                for npu in sorted(record["selected_npu_ids"])
            ]
            previous_selected_order = [
                int(npu) for npu in record["selected_npu_ids"]
            ]
    trigger_records = []
    for record in diagnostics:
        for reason_index, reason in enumerate(record["trigger_reasons"]):
            trigger_records.append(
                {
                    "raw_time_ms": float(record["snapshot_time_ms"]),
                    "effective_time_ms": float(record["snapshot_time_ms"]),
                    "reason": str(reason),
                    "rate_limited": False,
                    "coalesced": reason_index > 0,
                    "min_interval_ms": 100.0,
                }
            )
    invariants = {name: True for name in REQUIRED_TIMELINE_INVARIANTS}
    invariants.update(
        {
            "warmup_reached_all_npus": True,
            "measurement_window_closed": True,
            "measurement_duration_exact": True,
            "all_npus_sampled_for_slo": True,
            "all_tagged_requests_completed": True,
            "no_backlog_exhaustion": True,
            "timeline_dispatch_probe_nonempty": True,
        }
    )
    ssd_busy_whole = [float(boundaries[-1]["ssd_cumulative_busy_ms_by_ssu"][ssu]) for ssu in range(num_ssu)]
    link_busy_whole = [float(boundaries[-1]["npu_link_cumulative_busy_ms_by_npu"][npu]) for npu in range(NUM_NPU)]
    state_durations = []
    for npu in range(NUM_NPU):
        compute_ms = compute_total[npu]
        barrier_ms = math.fsum(
            synthetic_barrier_blocks[block][npu]
            for block in range(BLOCK_COUNT)
        )
        other_ms = MEASUREMENT_MS - compute_ms - barrier_ms
        state_durations.append(
            {
                "npu_id": npu,
                "compute_ms": compute_ms,
                "io_barrier_ms": barrier_ms,
                "other_ms": other_ms,
                "measurement_ms": MEASUREMENT_MS,
                "compute_fraction": compute_ms / MEASUREMENT_MS,
                "io_barrier_fraction": barrier_ms / MEASUREMENT_MS,
                "other_fraction": other_ms / MEASUREMENT_MS,
            }
        )
    block_state_durations = []
    for block_index, block in enumerate(blocks):
        compute_values = [float(value) for value in block["compute_ms_by_npu"]]
        barrier_values = list(synthetic_barrier_blocks[block_index])
        other_values = [
            BLOCK_MS - compute - barrier
            for compute, barrier in zip(compute_values, barrier_values)
        ]
        block_state_durations.append(
            {
                "block": block_index,
                "start_ms": float(block["start_ms"]),
                "end_ms": float(block["end_ms"]),
                "duration_ms": BLOCK_MS,
                "compute_ms_by_npu": compute_values,
                "io_barrier_ms_by_npu": barrier_values,
                "other_ms_by_npu": other_values,
            }
        )
    route_probes = _synthetic_route_probes(case, num_ssu, start_ms)
    summary = {
        "schema_version": 2,
        "mode": "steady_state_full_load",
        "num_npu": NUM_NPU,
        "num_ssu": num_ssu,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "warmup_requests_per_npu": 8,
        "warmup_reached_ms": 500.0,
        "settle_ms": 500.0,
        "measurement_start_ms": start_ms,
        "measurement_end_ms": start_ms + MEASUREMENT_MS,
        "measurement_duration_ms": MEASUREMENT_MS,
        "stationarity_boundary_semantics": BOUNDARY_SEMANTICS,
        "timeline_diagnostics_enabled": True,
        "timeline_demand_semantics": {
            "controller_demand": "synthetic controller demand",
            "physical_demand": "synthetic physical demand",
            "installed_cir": "guarantee, not throughput",
            "realized_service": "boundary counter difference",
            "ssd_served_awaiting_link_enqueue": "non-streaming command-stage bytes",
        },
        "timeline_ssd_accounting_semantics": {
            "schema": TIMELINE_SCHEMA,
            "ssd_served_cumulative_gb": (
                "compensated whole-command completion totals per NPU and SSU, plus the at-most-one active command prefix reconstructed from its immutable activation time and total_gb"
            ),
            "ssd_outstanding_gb": (
                "direct math.fsum enumeration of every physical pending command total_gb plus the active command's immutable projected remainder; never derived by subtracting cumulative counters"
            ),
            "ssd_outstanding_blocks": (
                "exact integer enumeration of every pending and active command's block_count"
            ),
            "busy_service_reference": (
                "independent compensated SSD busy-time counter multiplied by the physical SSD bandwidth"
            ),
            "fragmented_service_diagnostic": (
                "historical observer-fragmentation-dependent per-settle service accumulation retained only in accounting residuals; it is not a scientific output or gate input"
            ),
            "queue_identity": (
                "compensated enqueued_gb minus stable cumulative service minus direct physical outstanding_gb"
            ),
        },
        "timeline_adaptive_deadline_input": False,
        "timeline_adaptive_deadline_note": "diagnostic only",
        "timeline_state_durations_ms_by_npu": state_durations,
        "timeline_block_state_durations_ms": block_state_durations,
        "timeline_carry_in_batches_schema": (
            "steady_timeline_carry_in_batch_v1"
        ),
        "timeline_carry_in_batches": synthetic_carry_rows,
        "timeline_carry_in_batch_semantics": (
            "exact batches satisfying admission_time_ms < measurement_start_ms <= completion_time_ms; "
            "at most one per NPU; half-open interval "
            "intersections use exact clipping; admissions inside the "
            "measurement window remain exclusively in request_rows"
        ),
        "timeline_state_duration_semantics": (
            "exact intersections of every microbatch layer compute interval and "
            "its preceding I/O-barrier interval with the measurement window; "
            "includes carry-in, and other is the exact window complement"
        ),
        "timeline_control_trigger_records": trigger_records,
        "timeline_route_probe_records": route_probes,
        "timeline_dispatch_probe_ms": 50.0,
        "timeline_dispatch_probe_limit": 10_000,
        "timeline_dispatch_probe_truncated": False,
        "timeline_route_probe_truncated": False,
        "timeline_dispatch_probe_records": probes,
        "measurement_stationarity_boundary_count": BOUNDARY_COUNT,
        "measurement_stationarity_boundaries": boundaries,
        "measurement_control_counter_window": CONTROL_WINDOW,
        "legacy_control_counter_scope": "synthetic",
        "pressure_ttl_ms": 5.0 if case == "layer_once_ttl_5ms" else 0.0,
        "cir_write_threshold_gbps": 0.0,
        "slo_alpha": 2.0,
        "mean_npu_utilization": math.fsum(compute_total) / (NUM_NPU * MEASUREMENT_MS),
        "npu_utilizations": [value / MEASUREMENT_MS for value in compute_total],
        "compute_ms_by_npu": compute_total,
        "measurement_ssd_busy_ms_by_ssu": ssd_busy_whole,
        "measurement_ssd_served_gb_by_ssu": [value * SSD_CAP_GBPS / 1000.0 for value in ssd_busy_whole],
        "measurement_npu_link_busy_ms_by_npu": link_busy_whole,
        "measurement_npu_ssu_ssd_served_gb": whole_ssd,
        "measurement_fragmented_npu_ssu_ssd_served_gb": whole_ssd,
        "measurement_ssd_accounting_residuals": (
            measurement_ssd_accounting_residuals
        ),
        "measurement_npu_ssu_link_served_gb": whole_link,
        "ttft_slo_attainment": statistics.fmean(
            statistics.fmean(
                bool(row["slo_met"])
                for row in requests
                if int(row["npu_id"]) == npu_id
            )
            for npu_id in range(NUM_NPU)
        ),
        "request_weighted_slo_attainment": statistics.fmean(outcomes),
        "mean_ttft_ms": statistics.fmean(ttfts),
        "p99_ttft_ms": float(np.percentile(ttfts, 99)),
        "measurement_request_count": len(requests),
        "request_counts_by_npu": request_counts,
        "request_rows": requests,
        "measurement_blocks": blocks,
        "input_fingerprint": simulator_hash,
        "control_min_interval_ms": 100.0 if case == "adaptive_t0_i100ms" else None,
        "control_evaluations": len(diagnostics),
        "measurement_control_evaluations": BLOCK_COUNT if case == "adaptive_t0_i100ms" else 0,
        "measurement_cir_commits": 0,
        "measurement_cir_write_transactions": 0,
        "measurement_cir_write_transactions_by_ssu": [0] * num_ssu,
        "measurement_cir_path_writes": 0,
        "measurement_cir_path_writes_by_ssu": [0] * num_ssu,
        "invariants": invariants,
        "adaptive_controller_profile": (
            {
                "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
                "explicit_spill_threshold": 0.75,
                "target_ratio": 0.52,
                "required_ratio": 0.5,
                "background_reserve_fraction": 0.05,
            }
            if case == "adaptive_t0_i100ms"
            else None
        ),
        "adaptive_decision_diagnostic_schema": "adaptive_admission_decision_v1" if case == "adaptive_t0_i100ms" else None,
        "adaptive_decision_diagnostics": diagnostics,
    }
    return summary


def _synthetic_payload(num_ssu: int) -> dict:
    source_manifest = _current_runner_source_manifest()
    source = _runner_canonical_hash(
        source_manifest, SOURCE_FINGERPRINT_NAMESPACE
    )
    definition = _legacy_definition_fingerprint()
    input_authentication = _current_scientific_input_authentication()
    schedule_fingerprints = _current_formal_schedule_fingerprints()
    runtime = {
        "hostname": "synthetic-private-host-must-not-escape",
        "python": "3.14.0",
        "python_implementation": "CPython",
        "numpy": "2.5.0",
        "numpy_blas_identity": {
            "name": "synthetic-blas",
            "version": "1",
            "openblas_configuration": "synthetic-common-runtime",
        },
        "platform": "Linux-synthetic",
        "multiprocessing_start_method": "spawn",
        "thread_limit_environment": {"OMP_NUM_THREADS": "1"},
        "pid": 12345,
        "rss_peak_bytes": 123456,
    }
    cases_spec = [dict(case) for case in LEGACY32_CASE_SPECS]
    spec = {
        "schema_version": 3,
        "experiment": LEGACY32_DEFINITION_CANONICAL["experiment_name"],
        "definition": "legacy32",
        "definition_fingerprint": definition,
        "campaign_spec_sha256": None,
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssu_list": list(LEGACY32_DEFINITION_CANONICAL["default_ssus"]),
        "cases": cases_spec,
        "report_roles": {},
        "workload": {
            "mode": "iid_uniform_profile_catalog_v1",
            "seed": 42,
            "requests_per_npu": 256,
            "authentication": input_authentication,
            **schedule_fingerprints,
        },
        "steady_state": {
            "seed": 42,
            "requests_per_npu": 256,
            "warmup_requests_per_npu": 8,
            "settle_ms": 500.0,
            "measurement_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "slo_alpha": 2.0,
            "timeline_diagnostics": True,
            "timeline_dispatch_probe_ms": 50.0,
            "timeline_dispatch_probe_limit": 10_000,
            "calibration_mode": False,
        },
        "adaptive": {
            "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
            "explicit_spill_threshold": 0.75,
            "target_ratio": 0.52,
            "required_ratio": 0.5,
            "background_reserve_fraction": 0.05,
            "ssd_cap_gbps": SSD_CAP_GBPS,
            "npu_cap_gbps": NPU_LINK_CAP_GBPS,
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash reused across all 16 layers",
        "source_files": list(_transitive_runner_sources(PROJECT_ROOT))
        + ["data"],
        "diagnostics": {"timeline_diagnostics": True},
    }
    # The synthetic fixture uses the exact same complete spec constructor as
    # the validator; no reduced self-test-only schema is accepted.
    spec = _expected_experiment_spec()
    config = _runner_canonical_hash(spec, CONFIG_FINGERPRINT_NAMESPACE)
    shared = {
        "catalog": schedule_fingerprints["catalog"],
        "recipe": schedule_fingerprints["recipe"],
        "schedule": schedule_fingerprints["schedule"],
        "assignment": schedule_fingerprints["assignment"],
        "prefix_32_assignment": schedule_fingerprints[
            "prefix_32_assignment_hash"
        ],
        "full_assignment": schedule_fingerprints["full_assignment_hash"],
    }
    fingerprints = {
        **shared,
        **_current_materialized_input_fingerprints(num_ssu),
    }
    results = []
    for case in CASES:
        results.append(
            {
                "status": "ok",
                "case": case,
                "family": _case_spec(spec, case)["family"],
                "kind": _case_spec(spec, case)["kind"],
                "num_ssu": num_ssu,
                "num_npu": NUM_NPU,
                "backing_requests_per_npu": 256,
                "definition": "legacy32",
                "definition_fingerprint": definition,
                "case_spec": _case_spec(spec, case),
                "source_fingerprint": source,
                "config_fingerprint": config,
                "case_fingerprint": _case_fingerprint(
                    _case_spec(spec, case), num_ssu, source, config
                ),
                "campaign_spec_sha256": None,
                "input_fingerprints": dict(fingerprints),
                "runtime": dict(runtime),
                "steady_summary": _synthetic_summary(case, num_ssu, fingerprints["simulator"]),
            }
        )
    return {
        "schema_version": 3,
        "complete": False,
        "selected_complete": True,
        "source_stable_during_run": True,
        "config_stable_during_run": True,
        "campaign_spec_stable_during_run": True,
        "source_fingerprint": source,
        "ending_source_fingerprint": source,
        "source_manifest": source_manifest,
        "definition": "legacy32",
        "definition_fingerprint": definition,
        "num_npu": NUM_NPU,
        "backing_requests_per_npu": 256,
        "total_assignment_count": NUM_NPU * 256,
        "path_abi": dict(EXPECTED_PATH_ABI),
        "input_authentication": input_authentication,
        "runtime": runtime,
        "config_fingerprint": config,
        "ending_config_fingerprint": config,
        "campaign_spec_sha256": None,
        "campaign_spec_authentication": None,
        "ending_campaign_spec_authentication": None,
        "experiment_spec": spec,
        "selected_ssus": [num_ssu],
        "selected_cases": list(CASES),
        "selected_keys": [[case, num_ssu] for case in CASES],
        "schedule_metadata": {"seed": 42, "assignment": _synthetic_hash("assignment")},
        "pairing_audit": {
            str(num_ssu): {
                "cases": list(CASES),
                "has_rows": True,
                "all_available_rows_paired": True,
            }
        },
        "results": results,
    }


def _self_test_independent_adaptive_oracles() -> None:
    """Pin admission/allocation branches to three hand-computed examples."""

    examples = (
        (
            [[40.0], [40.0]],
            "all_required_targets_feasible",
            (0, 1),
            (),
            "v1_coflow_residual",
            [[20.0], [20.0]],
            None,
        ),
        (
            [[40.0, 40.0]],
            "all_preferred_targets_feasible",
            (0,),
            (),
            "v1_coflow_residual",
            [[25.0, 25.0]],
            None,
        ),
        (
            [[40.0], [100.0]],
            "greedy_overload",
            (0,),
            (1,),
            "v2_explicit_selected_spill",
            [[38.0], [2.0]],
            {
                "floor": [[20.8], [0.0]],
                "background": [[0.0], [2.0]],
                "selected_tail": [[17.2], [0.0]],
                "spill_tail": [[0.0], [0.0]],
            },
        ),
    )
    for (
        demand,
        expected_mode,
        expected_selected,
        expected_rejected,
        expected_residual_mode,
        expected_grants,
        expected_v2,
    ) in examples:
        replay = replay_admission_selection(
            demand,
            target_ratio=0.52,
            required_ratio=0.5,
            background_reserve_fraction=0.05,
            pinned_npu_ids=(),
            ssd_caps=SSD_CAP_GBPS,
            npu_caps=NPU_LINK_CAP_GBPS,
        )
        allocation = allocate_adaptive_admission_grants(
            demand,
            explicit_spill_threshold=0.75,
            target_ratio=0.52,
            required_ratio=0.5,
            background_reserve_fraction=0.05,
            pinned_npu_ids=(),
            ssd_caps=SSD_CAP_GBPS,
            npu_caps=NPU_LINK_CAP_GBPS,
        )
        _require(
            replay.selection_mode == expected_mode
            and tuple(replay.selected_npu_ids) == expected_selected
            and tuple(replay.rejected_npu_ids) == expected_rejected,
            "independent Adaptive admission oracle mismatch",
        )
        _require(
            allocation.residual_mode == expected_residual_mode
            and tuple(allocation.selected_npu_ids) == expected_selected
            and _matrices_close(allocation.grants_gbps, expected_grants),
            "independent Adaptive allocation oracle mismatch",
        )
        if expected_v2 is None:
            _require(
                allocation.v2_allocation is None,
                "independent Adaptive V1 oracle unexpectedly exposed V2",
            )
        else:
            v2 = allocation.v2_allocation
            _require(v2 is not None, "independent Adaptive V2 oracle missing")
            _require(
                _matrices_close(v2.floor_grants_gbps, expected_v2["floor"])
                and _matrices_close(
                    v2.background_grants_gbps, expected_v2["background"]
                )
                and _matrices_close(
                    v2.selected_tail_grants_gbps,
                    expected_v2["selected_tail"],
                )
                and _matrices_close(
                    v2.spill_tail_grants_gbps, expected_v2["spill_tail"]
                ),
                "independent Adaptive V2 component oracle mismatch",
            )


def _self_test_completion_boundary_active_command() -> None:
    """Exercise completion==boundary under the producer's left-limit order."""

    total_gb = 0.04
    boundary_time_ms = 1234.5
    exact_age_ms = total_gb / SSD_CAP_GBPS * 1000.0
    command = {
        "remaining_gb": 0.0,
        "command_start_time_ms": boundary_time_ms - exact_age_ms,
        "command_age_ms": exact_age_ms,
        "physical_service_gbps": SSD_CAP_GBPS,
        "non_preemptive": True,
    }
    _validate_active_command_projection(
        command,
        boundary_time_ms=boundary_time_ms,
        total_gb=total_gb,
        context="positive.completion_boundary_active_command",
    )


def _self_test_fragmented_measurement_reduction_order() -> None:
    """Pin diagnostic residual/decimal conversion to producer FP order."""

    stable = [[1e8] for _ in range(NUM_NPU)]
    fragmented = [[1e8] for _ in range(NUM_NPU)]
    fragmented[0][0] += 1e-8
    fragmented[1][0] += 1e-8
    fragmented[2][0] -= 1e-8
    busy = [math.fsum(row[0] for row in stable)]
    expected = _expected_measurement_ssd_accounting_residuals(
        stable, fragmented, busy, 1
    )
    producer_order = expected[
        "fragmented_service_minus_stable_gb_by_ssu"
    ][0]
    pairwise_difference_order = math.fsum(
        fragmented[npu_id][0] - stable[npu_id][0]
        for npu_id in range(NUM_NPU)
    )
    _require(
        abs(producer_order - pairwise_difference_order) * 1e9 > 1.0,
        "fragmented reduction-order regression does not expose byte-scale amplification",
    )
    _validate_decimal_byte_identity(
        [producer_order],
        [producer_order * 1e9],
        "positive.fragmented_decimal",
    )
    try:
        _validate_decimal_byte_identity(
            [producer_order],
            [producer_order * 1e9 + 1e-4],
            "negative.fragmented_decimal",
        )
    except AnalysisError as error:
        _require(
            "conversion mismatch" in str(error),
            "fragmented decimal negative regression hit the wrong gate",
        )
    else:
        raise AnalysisError(
            "fragmented decimal negative regression accepted a forged conversion"
        )


def _self_test_previous_pinned_selection_order() -> None:
    """Pin continuing requests in the producer's prior allocation order."""

    previous_pairs = ((1, 101), (2, 102), (3, 103))
    request_map = {1: 101, 2: 202, 3: 103}
    previous_selected_order = (3, 1, 2)
    _validate_previous_pinned_order(
        pinned=(3, 1),
        previous_pairs=previous_pairs,
        request_map=request_map,
        previous_selected_order=previous_selected_order,
        context="positive.previous_pinned_order",
    )
    try:
        _validate_previous_pinned_order(
            pinned=(1, 3),
            previous_pairs=previous_pairs,
            request_map=request_map,
            previous_selected_order=previous_selected_order,
            context="negative.previous_pinned_order",
        )
    except AnalysisError as error:
        _require(
            "do not preserve prior selection order" in str(error),
            "pinned-order negative regression hit the wrong gate",
        )
    else:
        raise AnalysisError(
            "pinned-order negative regression accepted a sorted permutation"
        )


def _run_corruption_suite(root: Path, inputs: Sequence[Path]) -> list[str]:
    """Targeted integrated negative tests with exact expected gate messages."""

    passed: list[str] = []

    def expect(
        name: str, expected_message: str, action: Callable[[], object]
    ) -> None:
        try:
            action()
        except AnalysisError as error:
            _require(
                expected_message in str(error),
                f"negative test {name!r} hit the wrong gate: {error}",
            )
        else:
            raise AnalysisError(
                f"negative test {name!r} did not reject corrupted evidence"
            )
        passed.append(name)

    payload = json.loads(inputs[0].read_text(encoding="utf-8"))
    rows = {str(row["case"]): row for row in payload["results"]}
    baseline = rows["baseline"]
    layer = rows["layer_once_ttl_5ms"]
    adaptive = rows["adaptive_t0_i100ms"]

    _validate_bounded_probe_count(9_999, False, 10_000, "positive.probe")
    _validate_bounded_probe_count(10_000, True, 10_000, "positive.probe")
    expect(
        "probe_cap_flag_true_below_limit",
        "bounded probe count/cap-reached flag mismatch",
        lambda: _validate_bounded_probe_count(
            9_999, True, 10_000, "negative.probe"
        ),
    )
    expect(
        "probe_cap_flag_false_at_limit",
        "bounded probe count/cap-reached flag mismatch",
        lambda: _validate_bounded_probe_count(
            10_000, False, 10_000, "negative.probe"
        ),
    )
    expect(
        "probe_count_above_limit",
        "bounded probe count/cap-reached flag mismatch",
        lambda: _validate_bounded_probe_count(
            10_001, True, 10_000, "negative.probe"
        ),
    )

    # Complete config-authenticated experiment contract.
    spec = payload["experiment_spec"]
    original_scope = spec["measurement_cost_scope"]
    spec["measurement_cost_scope"] = "forged"
    try:
        expect(
            "complete_experiment_spec",
            "complete local runner contract",
            lambda: _validate_experiment_spec(spec, "negative.spec"),
        )
    finally:
        spec["measurement_cost_scope"] = original_scope

    forged_auth = dict(spec["workload"]["authentication"])
    original_auth = spec["workload"]["authentication"]
    forged_auth["catalog_hash"] = "0" * 64
    spec["workload"]["authentication"] = forged_auth
    try:
        expect(
            "joint_input_authentication_forgery",
            "complete local runner contract",
            lambda: _validate_experiment_spec(spec, "negative.auth"),
        )
    finally:
        spec["workload"]["authentication"] = original_auth

    forged_spec_paths: list[Path] = []
    for shard_index, input_path in enumerate(inputs):
        forged_payload = json.loads(input_path.read_text(encoding="utf-8"))
        forged_payload["experiment_spec"]["measurement_cost_scope"] = "forged"
        forged_config = _runner_canonical_hash(
            forged_payload["experiment_spec"], CONFIG_FINGERPRINT_NAMESPACE
        )
        forged_payload["config_fingerprint"] = forged_config
        forged_payload["ending_config_fingerprint"] = forged_config
        for forged_row in forged_payload["results"]:
            forged_row["config_fingerprint"] = forged_config
            forged_row["case_fingerprint"] = _case_fingerprint(
                forged_row["case_spec"],
                int(forged_row["num_ssu"]),
                str(forged_row["source_fingerprint"]),
                forged_config,
            )
        forged_path = root / f"forged_spec_shard_{shard_index}.json"
        _write_json(forged_path, forged_payload)
        forged_spec_paths.append(forged_path)
    expect(
        "integrated_spec_config_case_rehash_forgery",
        "complete local runner contract",
        lambda: _load_and_validate(forged_spec_paths),
    )

    deleted_field = spec.pop("pairing_scope")
    try:
        expect(
            "complete_experiment_spec_missing_field",
            "complete local runner contract",
            lambda: _validate_experiment_spec(spec, "negative.spec_missing"),
        )
    finally:
        spec["pairing_scope"] = deleted_field

    # Carry-in definition, fixed compute budget, cohort completeness, and
    # independent state reconstruction.
    summary = baseline["steady_summary"]
    original_summary_mode = summary["mode"]
    summary["mode"] = "forged_mode"
    try:
        expect(
            "steady_summary_producer_schema_mode",
            "steady summary producer schema/mode mismatch",
            lambda: _validate_summary(baseline, inputs[0]),
        )
    finally:
        summary["mode"] = original_summary_mode

    original_summary_settle = summary["settle_ms"]
    summary["settle_ms"] = 0.0
    try:
        expect(
            "steady_summary_run_contract",
            "steady summary warmup/settle/SLO contract mismatch",
            lambda: _validate_summary(baseline, inputs[0]),
        )
    finally:
        summary["settle_ms"] = original_summary_settle

    carry = summary["timeline_carry_in_batches"]
    removed_carry = carry.pop(0)
    try:
        expect(
            "missing_carry_in",
            "carry-in rows do not match the left-boundary active set",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.carry",
            ),
        )
    finally:
        carry.insert(0, removed_carry)

    equality_carry = next(
        row
        for row in carry
        if float(row["completion_time_ms"])
        == float(summary["measurement_start_ms"])
    )
    original_equality_completion = equality_carry["completion_time_ms"]
    equality_carry["completion_time_ms"] = (
        float(summary["measurement_start_ms"]) - 0.001
    )
    try:
        expect(
            "carry_left_limit_definition",
            "exact left-limit carry-in definition",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.carry_left_limit",
            ),
        )
    finally:
        equality_carry["completion_time_ms"] = original_equality_completion

    original_layer_duration = carry[0]["compute_duration_ms"][0]
    carry[0]["compute_duration_ms"][0] = original_layer_duration + 0.01
    try:
        expect(
            "carry_layer_compute_budget",
            "layer 0 interval closure mismatch",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.carry_compute",
            ),
        )
    finally:
        carry[0]["compute_duration_ms"][0] = original_layer_duration

    original_ideal_compute = carry[0]["ideal_compute_ms"]
    carry[0]["ideal_compute_ms"] = original_ideal_compute + 0.1
    try:
        expect(
            "carry_authenticated_ideal_compute",
            "carry-in compute/completion closure mismatch",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.carry_ideal",
            ),
        )
    finally:
        carry[0]["ideal_compute_ms"] = original_ideal_compute

    final_active_row = summary["measurement_stationarity_boundaries"][-1][
        "timeline"
    ]["npu_rows"][0]
    original_final_activated = final_active_row[
        "activated_compute_cumulative_ms"
    ]
    final_active_row["activated_compute_cumulative_ms"] = (
        original_final_activated + 1.0
    )
    try:
        expect(
            "measurement_lifecycle_activation_coverage",
            "activated-compute boundary delta does not cover exactly the measurement request cohort",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.activation_coverage",
            ),
        )
    finally:
        final_active_row["activated_compute_cumulative_ms"] = (
            original_final_activated
        )

    original_carry_invariant = summary["invariants"].pop(
        "timeline_carry_in_compute_budget_exact"
    )
    try:
        expect(
            "producer_carry_in_invariant_missing",
            "timeline invariants missing",
            lambda: _validate_summary(baseline, inputs[0]),
        )
    finally:
        summary["invariants"]["timeline_carry_in_compute_budget_exact"] = (
            original_carry_invariant
        )

    original_carry_invariant_value = summary["invariants"][
        "timeline_carry_in_compute_budget_exact"
    ]
    summary["invariants"]["timeline_carry_in_compute_budget_exact"] = False
    try:
        expect(
            "producer_carry_in_invariant_false",
            "failed simulator invariants",
            lambda: _validate_summary(baseline, inputs[0]),
        )
    finally:
        summary["invariants"]["timeline_carry_in_compute_budget_exact"] = (
            original_carry_invariant_value
        )

    state_row = summary["timeline_block_state_durations_ms"][0]
    state_row["io_barrier_ms_by_npu"][0] += 0.25
    state_row["other_ms_by_npu"][0] -= 0.25
    try:
        expect(
            "paired_state_aggregate_corruption",
            "500-ms state durations differ from independent lifecycle reconstruction",
            lambda: _validate_state_durations_from_lifecycles(
                summary,
                summary["measurement_blocks"],
                summary["request_rows"],
                "negative.state",
            ),
        )
    finally:
        state_row["io_barrier_ms_by_npu"][0] -= 0.25
        state_row["other_ms_by_npu"][0] += 0.25

    activated_row = summary["measurement_stationarity_boundaries"][1][
        "timeline"
    ]["npu_rows"][0]
    original_activated = activated_row["activated_compute_cumulative_ms"]
    activated_row["activated_compute_cumulative_ms"] = -1.0
    try:
        expect(
            "activated_compute_decrease",
            "negative Q/activated compute",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.activated",
            ),
        )
    finally:
        activated_row["activated_compute_cumulative_ms"] = original_activated

    # Stable/fragmented SSD accounting and immutable active command projection.
    residual = summary["measurement_stationarity_boundaries"][1]["timeline"][
        "ssd_accounting_residuals_by_ssu"
    ][0]
    original_residual = residual["stable_service_minus_busy_counter_gb"]
    residual["stable_service_minus_busy_counter_gb"] = 1.5e-8
    try:
        expect(
            "stable_ssd_residual_1p5e8",
            "stable_service_minus_busy_counter_gb does not independently recompute",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.stable_residual",
            ),
        )
    finally:
        residual["stable_service_minus_busy_counter_gb"] = original_residual

    stable_cell = summary["measurement_stationarity_boundaries"][1][
        "timeline"
    ]["npu_ssu"]["ssd_served_cumulative_gb"]
    original_stable_cell = stable_cell[0][0]
    stable_cell[0][0] = original_stable_cell + 1.5e-8
    try:
        expect(
            "stable_ssd_counter_1p5e8",
            "stable_service_minus_busy_counter_gb does not independently recompute",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.stable_counter",
            ),
        )
    finally:
        stable_cell[0][0] = original_stable_cell

    link_cell = summary["measurement_stationarity_boundaries"][1][
        "timeline"
    ]["npu_ssu"]["link_served_cumulative_gb"]
    original_link_cell = link_cell[0][0]
    link_cell[0][0] = original_link_cell + 1e-5
    try:
        expect(
            "link_delivery_counter_corruption",
            "NPU x SSU link attribution does not equal physical link counter",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.link_counter",
            ),
        )
    finally:
        link_cell[0][0] = original_link_cell

    demand_boundary_index, demand_npu_id = next(
        (boundary_index, npu_id)
        for boundary_index, boundary in enumerate(
            summary["measurement_stationarity_boundaries"]
        )
        for npu_id, npu_row in enumerate(boundary["timeline"]["npu_rows"])
        if npu_row["controller_request_id"] is not None
    )
    demand_cell = summary["measurement_stationarity_boundaries"][
        demand_boundary_index
    ]["timeline"]["npu_ssu"]["controller_demand_gbps"]
    original_demand_cell = demand_cell[demand_npu_id][0]
    demand_cell[demand_npu_id][0] = original_demand_cell + 0.01
    try:
        expect(
            "boundary_controller_demand_corruption",
            "controller demand formula mismatch",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.boundary_demand",
            ),
        )
    finally:
        demand_cell[demand_npu_id][0] = original_demand_cell

    demand_boundary_cells = summary["measurement_stationarity_boundaries"][
        demand_boundary_index
    ]["timeline"]["npu_ssu"]
    demand_boundary_npu = summary["measurement_stationarity_boundaries"][
        demand_boundary_index
    ]["timeline"]["npu_rows"][demand_npu_id]
    original_controller_compute = demand_boundary_cells[
        "controller_remaining_compute_ms"
    ][demand_npu_id]
    original_controller_work = list(
        demand_boundary_cells["controller_declared_remaining_gb"][demand_npu_id]
    )
    original_layer_count = _authenticated_remaining_layer_count(
        num_ssu=3,
        npu_id=demand_npu_id,
        request_id=int(demand_boundary_npu["controller_request_id"]),
        remaining_compute_ms=float(original_controller_compute),
        remaining_work_gb_by_ssu=original_controller_work,
        context="positive.boundary_inventory",
    )
    fractional_layer_scale = (original_layer_count + 0.5) / original_layer_count
    demand_boundary_cells["controller_remaining_compute_ms"][demand_npu_id] = (
        original_controller_compute * fractional_layer_scale
    )
    demand_boundary_cells["controller_declared_remaining_gb"][demand_npu_id] = [
        value * fractional_layer_scale for value in original_controller_work
    ]
    try:
        expect(
            "boundary_controller_inventory_joint_corruption",
            "authenticated integer layer inventory",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.boundary_inventory",
            ),
        )
    finally:
        demand_boundary_cells["controller_remaining_compute_ms"][
            demand_npu_id
        ] = original_controller_compute
        demand_boundary_cells["controller_declared_remaining_gb"][
            demand_npu_id
        ] = original_controller_work

    physical_cell = summary["measurement_stationarity_boundaries"][1][
        "timeline"
    ]["npu_ssu"]["ssd_outstanding_gb"]
    original_physical = physical_cell[0][0]
    physical_cell[0][0] = original_physical + 0.01
    try:
        expect(
            "direct_physical_outstanding_corruption",
            "physical_queue_minus_scheduler_gb does not independently recompute",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.direct_physical",
            ),
        )
    finally:
        physical_cell[0][0] = original_physical

    final_cells = summary["measurement_stationarity_boundaries"][-1][
        "timeline"
    ]["npu_ssu"]
    original_fragmented = final_cells[
        "ssd_served_fragmented_diagnostic_cumulative_gb"
    ][0][0]
    final_cells["ssd_served_fragmented_diagnostic_cumulative_gb"][0][0] += 1e-5
    try:
        expect(
            "fragmented_diagnostic_counter_corruption",
            "fragmented_service_minus_stable_gb does not independently recompute",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.fragmented",
            ),
        )
    finally:
        final_cells["ssd_served_fragmented_diagnostic_cumulative_gb"][0][0] = (
            original_fragmented
        )

    command_boundary = next(
        boundary
        for boundary in summary["measurement_stationarity_boundaries"]
        if any(
            command is not None
            for command in boundary["timeline"]["active_command_by_ssu"]
        )
    )
    command = next(
        command
        for command in command_boundary["timeline"]["active_command_by_ssu"]
        if command is not None
    )
    original_age = command["command_age_ms"]
    command["command_age_ms"] = original_age + 0.001
    try:
        expect(
            "active_command_age_projection",
            "immutable command start/age/remaining projection mismatch",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.command",
            ),
        )
    finally:
        command["command_age_ms"] = original_age

    original_start = command["command_start_time_ms"]
    command["command_start_time_ms"] = original_start - 0.001
    try:
        expect(
            "active_command_start_projection",
            "immutable command start/age/remaining projection mismatch",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.command_start",
            ),
        )
    finally:
        command["command_start_time_ms"] = original_start

    original_remaining = command["remaining_gb"]
    command["remaining_gb"] = original_remaining + 2.0e-8
    try:
        expect(
            "active_command_remaining_projection",
            "immutable command start/age/remaining projection mismatch",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.command_remaining",
            ),
        )
    finally:
        command["remaining_gb"] = original_remaining

    command_path_id = int(command["path_id"])
    command_ssu_id = next(
        ssu_id
        for ssu_id, item in enumerate(
            command_boundary["timeline"]["active_command_by_ssu"]
        )
        if item is command
    )
    sparse_row = next(
        row
        for row in command_boundary["timeline"]["sparse_ssu_path_rows"]
        if int(row["ssu_id"]) == command_ssu_id
        and int(row["path_id"]) == command_path_id
    )
    original_sparse_active = sparse_row["active_remaining_gb"]
    sparse_row["active_remaining_gb"] = original_sparse_active + 1.5e-8
    try:
        expect(
            "sparse_path_bytes_1p5e8",
            "sparse Path bytes do not close to aggregate SSD queue",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.sparse_path_bytes",
            ),
        )
    finally:
        sparse_row["active_remaining_gb"] = original_sparse_active

    layer_command_boundary = next(
        boundary
        for boundary in layer["steady_summary"][
            "measurement_stationarity_boundaries"
        ]
        if any(
            command is not None
            for command in boundary["timeline"]["active_command_by_ssu"]
        )
    )
    existing_path = layer_command_boundary["timeline"]["sparse_ssu_path_rows"][0]
    stray_path = dict(existing_path)
    stray_path_id = (int(existing_path["path_id"]) + 1) % EXPECTED_PATH_ABI[
        "path_count"
    ]
    qos = static_qos_config()
    stray_path.update(
        {
            "path_id": stray_path_id,
            "group_id": stray_path_id // EXPECTED_PATH_ABI["paths_per_group"],
            "cir_gbps": float(qos.path_cirs[stray_path_id]),
            "path_weight": float(qos.path_weights[stray_path_id]),
            "pending_blocks": 1,
            "pending_gb": 0.01,
            "active_remaining_gb": 0.001,
            "head_wait_age_ms": 0.0,
        }
    )
    layer_command_boundary["timeline"]["sparse_ssu_path_rows"].append(stray_path)
    try:
        expect(
            "noncommand_path_stray_active_remaining",
            "non-command Path exposes active remaining bytes",
            lambda: _validate_timeline(
                layer["steady_summary"],
                layer["steady_summary"]["measurement_blocks"],
                "layer_once_ttl_5ms",
                3,
                "negative.stray_active",
            ),
        )
    finally:
        layer_command_boundary["timeline"]["sparse_ssu_path_rows"].pop()

    # Full-64s route group tensor monotonicity.
    group_before = summary["measurement_stationarity_boundaries"][2]["timeline"][
        "npu_ssu"
    ]["route_blocks_by_group_cumulative"][0][0][0]
    summary["measurement_stationarity_boundaries"][2]["timeline"]["npu_ssu"][
        "route_blocks_by_group_cumulative"
    ][0][0][0] = 0
    try:
        expect(
            "route_group_counter_decrease",
            "route_blocks_by_group_cumulative decreased",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.route_group",
            ),
        )
    finally:
        summary["measurement_stationarity_boundaries"][2]["timeline"][
            "npu_ssu"
        ]["route_blocks_by_group_cumulative"][0][0][0] = group_before

    prior_group = summary["measurement_stationarity_boundaries"][1]["timeline"][
        "npu_ssu"
    ]["route_blocks_by_group_cumulative"][0][0][0]
    summary["measurement_stationarity_boundaries"][2]["timeline"]["npu_ssu"][
        "route_blocks_by_group_cumulative"
    ][0][0][0] = prior_group
    try:
        expect(
            "route_plan_without_group_blocks",
            "route plans and routed-block group counts do not close",
            lambda: _validate_timeline(
                summary,
                summary["measurement_blocks"],
                "baseline",
                3,
                "negative.route_plan_group",
            ),
        )
    finally:
        summary["measurement_stationarity_boundaries"][2]["timeline"][
            "npu_ssu"
        ]["route_blocks_by_group_cumulative"][0][0][0] = group_before

    # Adaptive decision causal binding, inactive exact-zero, cross-record
    # continuity, and V2 grant decomposition.
    adaptive_summary = adaptive["steady_summary"]
    decisions = adaptive_summary["adaptive_decision_diagnostics"]

    def prior_selected_order(records: Sequence[dict], target: dict) -> list[int]:
        index = next(
            position for position, record in enumerate(records) if record is target
        )
        return (
            []
            if index == 0
            else [int(value) for value in records[index - 1]["selected_npu_ids"]]
        )
    original_control_min_interval = adaptive_summary["control_min_interval_ms"]
    adaptive_summary["control_min_interval_ms"] = 200.0
    try:
        expect(
            "adaptive_periodic_interval_rejected",
            "controller interval mismatch",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        adaptive_summary["control_min_interval_ms"] = (
            original_control_min_interval
        )
    original_case_interval = adaptive["case_spec"]["min_interval_ms"]
    adaptive["case_spec"]["min_interval_ms"] = 200.0
    try:
        expect(
            "adaptive_case_spec_interval_rejected",
            "case spec does not authenticate control scheduling",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        adaptive["case_spec"]["min_interval_ms"] = original_case_interval

    experiment_adaptive_case = next(
        case_spec
        for case_spec in spec["cases"]
        if case_spec["name"] == "adaptive_t0_i100ms"
    )
    original_experiment_interval = experiment_adaptive_case["min_interval_ms"]
    experiment_adaptive_case["min_interval_ms"] = 200.0
    try:
        expect(
            "adaptive_complete_experiment_spec_interval_rejected",
            "complete local runner contract",
            lambda: _validate_experiment_spec(
                spec, "negative.adaptive_experiment_spec"
            ),
        )
    finally:
        experiment_adaptive_case["min_interval_ms"] = (
            original_experiment_interval
        )
    original_trigger_reason = adaptive_summary["timeline_control_trigger_records"][
        1
    ]["reason"]
    adaptive_summary["timeline_control_trigger_records"][1]["reason"] = (
        "wall_clock"
    )
    try:
        expect(
            "adaptive_wall_clock_trigger_rejected",
            "invalid/non-causal/wall-clock trigger record",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        adaptive_summary["timeline_control_trigger_records"][1]["reason"] = (
            original_trigger_reason
        )

    trigger_for_flag_test = adaptive_summary[
        "timeline_control_trigger_records"
    ][1]
    original_rate_limited = trigger_for_flag_test["rate_limited"]
    trigger_for_flag_test["rate_limited"] = not original_rate_limited
    try:
        expect(
            "adaptive_trigger_effective_flag_rejected",
            "rate-limit/coalescing flags do not recompute",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        trigger_for_flag_test["rate_limited"] = original_rate_limited

    original_decision_trigger_reasons = list(decisions[0]["trigger_reasons"])
    decisions[0]["trigger_reasons"] = ["batch_boundary"]
    try:
        expect(
            "adaptive_decision_trigger_binding_rejected",
            "decision trigger reasons/time do not match executed event-driven triggers",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        decisions[0]["trigger_reasons"] = original_decision_trigger_reasons

    original_second_snapshot_time = decisions[1]["snapshot_time_ms"]
    decisions[1]["snapshot_time_ms"] = (
        float(decisions[0]["snapshot_time_ms"]) + 50.0
    )
    try:
        expect(
            "adaptive_decision_min_spacing_rejected",
            "snapshots violate the 100-ms minimum interval",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        decisions[1]["snapshot_time_ms"] = original_second_snapshot_time
    decision = decisions[0]
    original_compute = decision["remaining_compute_s_by_npu"][0]
    original_work = list(decision["remaining_work_gb_by_npu_ssu"][0])
    decision["remaining_compute_s_by_npu"][0] *= 2.0
    decision["remaining_work_gb_by_npu_ssu"][0] = [
        2.0 * value for value in original_work
    ]
    try:
        expect(
            "adaptive_authenticated_work_compute",
            "authenticated integer layer inventory",
            lambda: _validate_admission_diagnostic(
                decision, 3, "negative.adaptive_work"
            ),
        )
    finally:
        decision["remaining_compute_s_by_npu"][0] = original_compute
        decision["remaining_work_gb_by_npu_ssu"][0] = original_work

    inactive_record = json.loads(json.dumps(decision))
    inactive_record["remaining_work_gb_by_npu_ssu"][0] = [1e-6, 0.0, 0.0]
    inactive_record["remaining_compute_s_by_npu"][0] = 0.0
    inactive_record["controller_demand_gbps_by_npu_ssu"][0] = [0.0, 0.0, 0.0]
    expect(
        "adaptive_inactive_tiny_work",
        "work/compute/demand active-state equivalence mismatch",
        lambda: _validate_admission_diagnostic(
            inactive_record, 3, "negative.adaptive_inactive"
        ),
    )
    compute_only_record = json.loads(json.dumps(decision))
    compute_only_record["remaining_work_gb_by_npu_ssu"][0] = [0.0, 0.0, 0.0]
    compute_only_record["controller_demand_gbps_by_npu_ssu"][0] = [0.0, 0.0, 0.0]
    expect(
        "adaptive_inactive_compute_only",
        "work/compute/demand active-state equivalence mismatch",
        lambda: _validate_admission_diagnostic(
            compute_only_record, 3, "negative.adaptive_compute_only"
        ),
    )

    original_previous = decisions[1]["previous_selected_request_by_npu"]
    decisions[1]["previous_selected_request_by_npu"] = []
    try:
        expect(
            "adaptive_previous_selected_continuity",
            "previous-selected request map is not the preceding decision",
            lambda: _validate_summary(adaptive, inputs[0]),
        )
    finally:
        decisions[1]["previous_selected_request_by_npu"] = original_previous

    v2_record = next(
        record for record in decisions if record["v2_floor_grants_gbps"] is not None
    )
    original_component = v2_record["v2_floor_grants_gbps"][0][0]
    v2_record["v2_floor_grants_gbps"][0][0] = original_component + 0.01
    try:
        expect(
            "adaptive_v2_grant_component",
            "differs from exact V2 allocation replay",
            lambda: _validate_admission_diagnostic(
                v2_record,
                3,
                "negative.adaptive_v2",
                previous_selected_order=prior_selected_order(
                    decisions, v2_record
                ),
            ),
        )
    finally:
        v2_record["v2_floor_grants_gbps"][0][0] = original_component

    ssu4_payload = json.loads(inputs[1].read_text(encoding="utf-8"))
    ssu5_payload = json.loads(inputs[2].read_text(encoding="utf-8"))
    ssu4_decisions = next(
        row
        for row in ssu4_payload["results"]
        if row["case"] == "adaptive_t0_i100ms"
    )["steady_summary"]["adaptive_decision_diagnostics"]
    ssu5_decisions = next(
        row
        for row in ssu5_payload["results"]
        if row["case"] == "adaptive_t0_i100ms"
    )["steady_summary"]["adaptive_decision_diagnostics"]
    _require(
        any(record["v2_floor_grants_gbps"] is not None for record in decisions)
        and any(
            record["v2_floor_grants_gbps"] is not None
            for record in ssu4_decisions
        )
        and any(
            record["v2_floor_grants_gbps"] is None
            and record["residual_mode"] == "v1_coflow_residual"
            for record in ssu5_decisions
        ),
        "synthetic campaign does not exercise SSU3/4 V2 and SSU5 V1 branches",
    )
    v1_record = next(
        record
        for record in ssu5_decisions
        if record["v2_floor_grants_gbps"] is None
    )
    v1_record["v2_floor_grants_gbps"] = [
        [0.0] * 5 for _ in range(NUM_NPU)
    ]
    try:
        expect(
            "adaptive_v1_rejects_v2_component",
            "V1 residual mode unexpectedly exposes V2 components",
            lambda: _validate_admission_diagnostic(
                v1_record,
                5,
                "negative.adaptive_v1",
                previous_selected_order=prior_selected_order(
                    ssu5_decisions, v1_record
                ),
            ),
        )
    finally:
        v1_record["v2_floor_grants_gbps"] = None
    del ssu4_payload, ssu5_payload, ssu4_decisions, ssu5_decisions, v1_record

    validated_cases, _validated_provenance = _load_and_validate(inputs)
    (
        _positive_decisions,
        _positive_decision_npu,
        positive_boundary_slack,
        _positive_attempts,
    ) = _build_adaptive_decisions(validated_cases)

    # A linked boundary-register forgery remains internally consistent at the
    # sparse Path/timeline layer, but must still be rejected by replaying the
    # strictly-prior controller decisions.  This proves the causal gate is not
    # merely a duplicate single-field shape check.
    adaptive_case_data = validated_cases[(3, "adaptive_t0_i100ms")]
    causal_boundary_index, causal_path_row = next(
        (boundary_index, path_row)
        for boundary_index, boundary in enumerate(adaptive_case_data.boundaries)
        for path_row in boundary["timeline"]["sparse_ssu_path_rows"]
        if float(path_row["cir_gbps"]) > 1e-6
    )
    causal_boundary = adaptive_case_data.boundaries[causal_boundary_index]
    causal_ssu_id = int(causal_path_row["ssu_id"])
    causal_path_id = int(causal_path_row["path_id"])
    causal_npu_id = next(
        npu_id
        for npu_id in range(NUM_NPU)
        if cold_start_hybrid_path_id(npu_id) == causal_path_id
    )
    causal_installed = causal_boundary["timeline"]["npu_ssu"][
        "installed_dedicated_path_cir_gbps"
    ]
    original_boundary_cir = causal_installed[causal_npu_id][causal_ssu_id]
    original_sparse_cir = causal_path_row["cir_gbps"]
    linked_delta = -min(0.001, float(original_boundary_cir) / 2.0)
    causal_installed[causal_npu_id][causal_ssu_id] += linked_delta
    causal_path_row["cir_gbps"] += linked_delta
    try:
        _validate_timeline(
            adaptive_case_data.summary,
            adaptive_case_data.blocks,
            "adaptive_t0_i100ms",
            3,
            "positive.linked_boundary_register_forgery",
        )
        expect(
            "adaptive_decision_to_boundary_cir_binding",
            "reconstructed old CIR != boundary",
            lambda: _build_adaptive_decisions(validated_cases),
        )
    finally:
        causal_installed[causal_npu_id][causal_ssu_id] = original_boundary_cir
        causal_path_row["cir_gbps"] = original_sparse_cir

    original_measurement_commits = adaptive_case_data.summary[
        "measurement_cir_commits"
    ]
    adaptive_case_data.summary["measurement_cir_commits"] += 1
    try:
        expect(
            "adaptive_measurement_commit_counter_binding",
            "reconstructed CIR commit count mismatch",
            lambda: _build_adaptive_decisions(validated_cases),
        )
    finally:
        adaptive_case_data.summary["measurement_cir_commits"] = (
            original_measurement_commits
        )

    original_measurement_transactions = adaptive_case_data.summary[
        "measurement_cir_write_transactions_by_ssu"
    ][0]
    adaptive_case_data.summary[
        "measurement_cir_write_transactions_by_ssu"
    ][0] += 1
    try:
        expect(
            "adaptive_measurement_transaction_counter_binding",
            "reconstructed CIR transaction count mismatch",
            lambda: _build_adaptive_decisions(validated_cases),
        )
    finally:
        adaptive_case_data.summary[
            "measurement_cir_write_transactions_by_ssu"
        ][0] = original_measurement_transactions

    original_measurement_writes = adaptive_case_data.summary[
        "measurement_cir_path_writes_by_ssu"
    ][0]
    adaptive_case_data.summary["measurement_cir_path_writes_by_ssu"][0] += 1
    try:
        expect(
            "adaptive_measurement_path_write_counter_binding",
            "reconstructed CIR path-write count mismatch",
            lambda: _build_adaptive_decisions(validated_cases),
        )
    finally:
        adaptive_case_data.summary[
            "measurement_cir_path_writes_by_ssu"
        ][0] = original_measurement_writes

    masked_boundary_rows = [
        row
        for row in positive_boundary_slack
        if row["selection_classification"]
        in {"unknown_prefetch_decision", "unknown_or_stale_decision"}
    ]
    _require(
        masked_boundary_rows
        and all(
            row["selected_by_latest_strictly_prior_decision"] is None
            and row["deadline_slack_alpha1p5_ms_diagnostic_only"] is None
            and row["deadline_slack_alpha2_ms_diagnostic_only"] is None
            and row[
                "slack_after_remaining_compute_alpha1p5_ms_diagnostic_only"
            ]
            is None
            and row[
                "slack_after_remaining_compute_alpha2_ms_diagnostic_only"
            ]
            is None
            for row in masked_boundary_rows
        ),
        "synthetic stale/prefetch decision rows leaked causal slack labels",
    )
    validated_adaptive = validated_cases[(3, "adaptive_t0_i100ms")]
    prefetch_record = next(
        record
        for record in validated_adaptive.summary["adaptive_decision_diagnostics"]
        if any(bool(pair[1]) for pair in record["prefetch_only_by_npu"])
    )
    prefetch_pair = next(
        pair for pair in prefetch_record["prefetch_only_by_npu"] if bool(pair[1])
    )
    prefetch_pair[1] = False
    try:
        expect(
            "adaptive_false_prefetch_before_admission",
            "non-prefetch decision precedes admission",
            lambda: _build_adaptive_decisions(validated_cases),
        )
    finally:
        prefetch_pair[1] = True
    del validated_cases, validated_adaptive, _validated_provenance

    # Bounded route/dispatch inputs and authenticated placement.
    layer_summary = layer["steady_summary"]
    route = layer_summary["timeline_route_probe_records"][1]
    last_route = layer_summary["timeline_route_probe_records"][-1]
    original_last_route_time = last_route["time_ms"]
    original_last_snapshot = last_route["pressure_snapshot_time_ms"]
    last_age = last_route["pressure_age_ms"]
    last_route["time_ms"] = (
        layer_summary["measurement_start_ms"] + 50.0 + 2.0 * TOL
    )
    last_route["pressure_snapshot_time_ms"] = last_route["time_ms"] - last_age
    try:
        expect(
            "route_probe_half_open_50ms",
            "probe outside declared bounded window",
            lambda: _validate_summary(layer, inputs[0]),
        )
    finally:
        last_route["time_ms"] = original_last_route_time
        last_route["pressure_snapshot_time_ms"] = original_last_snapshot

    original_selected_path = route["selected_path_ids"][0]
    original_selected_group = route["selected_group_ids"][0]
    allowed = route["allowed_path_ids"]
    replacement = next(
        path for path in allowed if int(path) != int(original_selected_path)
    )
    route["selected_path_ids"][0] = replacement
    route["selected_group_ids"][0] = (
        int(replacement) // EXPECTED_PATH_ABI["paths_per_group"]
    )
    try:
        expect(
            "route_selected_path_corruption",
            "portable pressure-aware path replay mismatch",
            lambda: _validate_summary(layer, inputs[0]),
        )
    finally:
        route["selected_path_ids"][0] = original_selected_path
        route["selected_group_ids"][0] = original_selected_group

    original_snapshot = route["pressure_snapshot_time_ms"]
    original_age = route["pressure_age_ms"]
    route["pressure_age_ms"] = route["pressure_ttl_ms"] + 2e-12
    route["pressure_snapshot_time_ms"] = (
        route["time_ms"] - route["pressure_age_ms"]
    )
    try:
        expect(
            "route_ttl_exact_expiry",
            "pressure snapshot is outside its TTL",
            lambda: _validate_summary(layer, inputs[0]),
        )
    finally:
        route["pressure_snapshot_time_ms"] = original_snapshot
        route["pressure_age_ms"] = original_age

    original_block_size = route["block_sizes_gb"][0]
    route["block_sizes_gb"][0] = original_block_size + 0.01
    try:
        expect(
            "route_authenticated_placement",
            "routed block indices/sizes differ from authenticated placement",
            lambda: _validate_summary(layer, inputs[0]),
        )
    finally:
        route["block_sizes_gb"][0] = original_block_size

    original_route_path_count = route["path_count"]
    original_route_paths_per_group = route["paths_per_group"]
    route["path_count"] = 128
    route["paths_per_group"] = 16
    try:
        expect(
            "route_local_path_abi",
            "route hardware shape/registers differ from formal ABI",
            lambda: _validate_summary(layer, inputs[0]),
        )
    finally:
        route["path_count"] = original_route_path_count
        route["paths_per_group"] = original_route_paths_per_group

    dispatch = summary["timeline_dispatch_probe_records"][0]
    original_finish = dispatch["winner_finish_tag_before"]
    original_virtual = dispatch["winner_virtual_finish_after"]
    dispatch["winner_finish_tag_before"] = dispatch["minimum_finish_tag"] - 1e-6
    dispatch["winner_virtual_finish_after"] = dispatch[
        "winner_finish_tag_before"
    ]
    try:
        expect(
            "dispatch_finish_below_minimum",
            "winning finish tag/tie/runtime virtual-finish mismatch",
            lambda: _validate_summary(baseline, inputs[0]),
        )
    finally:
        dispatch["winner_finish_tag_before"] = original_finish
        dispatch["winner_virtual_finish_after"] = original_virtual

    # Row hashes forged together must still fail independent materialization.
    original_hashes = []
    for row in payload["results"]:
        original_hashes.append(row["input_fingerprints"]["workload"])
        row["input_fingerprints"]["workload"] = "f" * 64
    try:
        bad_path = root / "bad_joint_fingerprint.json"
        _write_json(bad_path, payload)
        expect(
            "joint_row_fingerprint_forgery",
            "row workload differs from independent materialization",
            lambda: _load_and_validate([bad_path, *inputs[1:]]),
        )
    finally:
        for row, value in zip(payload["results"], original_hashes):
            row["input_fingerprints"]["workload"] = value

    original_top_path_count = payload["path_abi"]["path_count"]
    payload["path_abi"]["path_count"] = 255
    try:
        bad_abi_path = root / "bad_path_abi.json"
        _write_json(bad_abi_path, payload)
        expect(
            "top_level_path_abi",
            "top-level dedicated Path ABI mismatch",
            lambda: _load_and_validate([bad_abi_path, *inputs[1:]]),
        )
    finally:
        payload["path_abi"]["path_count"] = original_top_path_count

    source_name = next(iter(payload["source_manifest"]))
    original_source_digest = payload["source_manifest"][source_name]
    payload["source_manifest"][source_name] = "0" * 64
    try:
        bad_source_path = root / "bad_source_manifest.json"
        _write_json(bad_source_path, payload)
        expect(
            "source_manifest_checkout_bytes",
            "checkout bytes differ from recorded source",
            lambda: _load_and_validate([bad_source_path, *inputs[1:]]),
        )
    finally:
        payload["source_manifest"][source_name] = original_source_digest

    payload["results"].append(json.loads(json.dumps(payload["results"][0])))
    try:
        extra_row_path = root / "bad_extra_row.json"
        _write_json(extra_row_path, payload)
        expect(
            "atomic_shard_extra_row",
            "selected shard must contain exactly three result rows",
            lambda: _load_and_validate([extra_row_path, *inputs[1:]]),
        )
    finally:
        payload["results"].pop()

    split_paths: list[Path] = []
    for index, source_row in enumerate(payload["results"]):
        split_payload = json.loads(json.dumps(payload))
        split_payload["results"] = [json.loads(json.dumps(source_row))]
        split_payload["selected_cases"] = [source_row["case"]]
        split_payload["selected_keys"] = [[source_row["case"], 3]]
        split_path = root / f"bad_split_shard_{index}.json"
        _write_json(split_path, split_payload)
        split_paths.append(split_path)
    expect(
        "atomic_shard_split_cases",
        "selected shard must contain exactly three result rows",
        lambda: _load_and_validate([*split_paths, *inputs[1:]]),
    )

    # Artifact publication must reject stale files before reading raw inputs.
    stale = root / "stale-output"
    stale.mkdir()
    _write_text(stale / "sentinel.txt", "stale")
    expect(
        "stale_output_directory",
        "refusing stale artifacts/raw inputs",
        lambda: analyze(inputs, stale, make_plots=False),
    )

    expect(
        "analyzer_source_stability",
        "analyzer source bytes changed",
        lambda: _require_source_bytes_unchanged(
            Path(__file__).resolve(), "0" * 64, "negative analyzer mutation"
        ),
    )
    return passed


def _self_test() -> dict:
    _self_test_independent_adaptive_oracles()
    _self_test_completion_boundary_active_command()
    _self_test_fragmented_measurement_reduction_order()
    _self_test_previous_pinned_selection_order()
    with tempfile.TemporaryDirectory(prefix="timeline64-selftest-") as temporary:
        root = Path(temporary)
        inputs = []
        for num_ssu in SSU_COUNTS:
            path = root / f"synthetic_ssu{num_ssu}.json"
            _write_json(path, _synthetic_payload(num_ssu))
            inputs.append(path)
        synthetic_ssu3 = _load_json(inputs[0])
        synthetic_adaptive = next(
            row
            for row in synthetic_ssu3["results"]
            if row["case"] == "adaptive_t0_i100ms"
        )["steady_summary"]
        first_decision = synthetic_adaptive["adaptive_decision_diagnostics"][0]
        first_request = int(first_decision["request_by_npu"][0][1])
        first_npu = int(first_decision["request_by_npu"][0][0])
        first_metadata = _current_materialized_request_metadata(3)[first_request]
        observed_remaining_layers = (
            float(first_decision["remaining_compute_s_by_npu"][first_npu])
            * 1000.0
            / (float(first_metadata["ideal_ttft_ms"]) / N_LAYERS)
        )
        _require(
            first_decision["prefetch_only_by_npu"][0][1] is True
            and _close(observed_remaining_layers, 15.0, tolerance=2e-9),
            "self-test did not exercise authenticated prefetch with 15 remaining layers",
        )
        first_trigger_records = [
            record
            for record in synthetic_adaptive["timeline_control_trigger_records"]
            if float(record["effective_time_ms"])
            == float(first_decision["snapshot_time_ms"])
        ]
        _require(
            {record["reason"] for record in first_trigger_records}
            == {"initial", "batch_boundary"}
            and [record["coalesced"] for record in first_trigger_records]
            == [False, True],
            "self-test did not exercise coalesced event-driven trigger reasons",
        )
        output = root / "analysis"
        result = analyze(inputs, output, make_plots=True, compact_plots=True)
        required = {
            "validation.json",
            "results.json",
            "report.md",
            "window_metrics.csv",
            "state_durations_per_npu.csv",
            "state_duration_summary.csv",
            "state_durations_500ms.csv",
            "compute_inventory_decomposition.csv",
            "matched_requests.csv",
            "adaptive_decisions.csv",
            "adaptive_decision_npu.csv",
            "adaptive_boundary_slack.csv",
            "adaptive_grant_components.csv",
            "adaptive_admission_attempts_ssu3.csv",
            "adaptive_admission_attempts_ssu4.csv",
            "adaptive_admission_attempts_ssu5.csv",
            "ssd_accounting_residuals.csv",
            "forensic_selection.csv",
            "forensic_timeline.csv",
            "dispatch_probe.csv",
            "route_probe_ssu3.csv",
            "route_probe_ssu4.csv",
            "route_probe_ssu5.csv",
            "00a_mean_npu_utilization_vs_ssu.png",
            "00b_ttft_slo_alpha1p5_vs_ssu.png",
            "00c_ttft_slo_alpha2_vs_ssu.png",
            "01_timeline_heatmap_ssu3.png",
            "01_timeline_heatmap_ssu4.png",
            "01_timeline_heatmap_ssu5.png",
            "02_adaptive_causal_decision_timeline.png",
            "02b_layer_once_route_causality.png",
            "05_state_duration_partition.png",
            "05b_state_duration_timeline.png",
            "07_micro_dispatch_probe.png",
            "09_forensic_zoom_ssu3.png",
            "09_forensic_zoom_ssu4.png",
            "09_forensic_zoom_ssu5.png",
            "analyze_npu32_ssu345_timeline64.py",
            "manifest.json",
        }
        required.update(
            f"timeline_npu_ssu_500ms_ssu{num_ssu}_{case}.csv"
            for num_ssu in SSU_COUNTS
            for case in CASES
        )
        required.update(
            (
                f"matched_request_layers_ssu{num_ssu}_{case}_"
                f"npu{npu_lo:02d}_{npu_lo + 7:02d}.csv"
            )
            for num_ssu in SSU_COUNTS
            for case in CASES
            for npu_lo in range(0, NUM_NPU, 8)
        )
        _require(required <= {path.name for path in output.iterdir()}, "self-test output set incomplete")
        validation = json.loads(
            (output / "validation.json").read_text(encoding="utf-8")
        )
        _require(validation["passed"] is True, "self-test validation did not pass")
        _require(
            validation["checks"]["portable_route_plan_replay_matches"] is True
            and validation["checks"][
                "block_state_128x500ms_partition_and_whole_window_sum_exact"
            ]
            is True,
            "self-test did not exercise route/block-state hard gates",
        )
        exported_route_rows = []
        for num_ssu in SSU_COUNTS:
            with (output / f"route_probe_ssu{num_ssu}.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                exported_route_rows.extend(csv.DictReader(handle))
        _require(
            exported_route_rows
            and all(
                row["portable_replay_exact_match"] == "True"
                and row["ttft_deadline_or_slack_used_for_route"] == "False"
                for row in exported_route_rows
            ),
            "self-test route replay/export mismatch",
        )
        with (output / "adaptive_decision_npu.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            exported_decision_npu_rows = list(csv.DictReader(handle))
        prefetch_rows = [
            row
            for row in exported_decision_npu_rows
            if row["prefetch_only_at_decision"] == "True"
        ]
        _require(prefetch_rows, "self-test did not exercise prefetch decisions")
        _require(
            all(
                row["elapsed_ttft_ms_diagnostic_only"] == ""
                and row[
                    "deadline_slack_alpha1p5_ms_diagnostic_only"
                ]
                == ""
                and row[
                    "deadline_slack_alpha2_ms_diagnostic_only"
                ]
                == ""
                for row in prefetch_rows
            ),
            "prefetch-only decision leaked elapsed/slack diagnostics",
        )
        with (output / "adaptive_boundary_slack.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            exported_boundary_slack = list(csv.DictReader(handle))
        unknown_slack_rows = [
            row
            for row in exported_boundary_slack
            if row["selection_classification"]
            in {"unknown_prefetch_decision", "unknown_or_stale_decision"}
        ]
        _require(
            unknown_slack_rows
            and all(
                row["selected_by_latest_strictly_prior_decision"] == ""
                and row["deadline_slack_alpha1p5_ms_diagnostic_only"] == ""
                and row["deadline_slack_alpha2_ms_diagnostic_only"] == ""
                and row[
                    "slack_after_remaining_compute_alpha1p5_ms_diagnostic_only"
                ]
                == ""
                and row[
                    "slack_after_remaining_compute_alpha2_ms_diagnostic_only"
                ]
                == ""
                for row in unknown_slack_rows
            ),
            "stale/prefetch boundary classification leaked slack or selected labels",
        )
        results_payload = json.loads(
            (output / "results.json").read_text(encoding="utf-8")
        )
        whole_result_rows = results_payload["whole_64s"]
        recomputed_maximum_spread_pp = max(
            100.0
            * (
                max(
                    float(row["npu_utilization"])
                    for row in whole_result_rows
                    if int(row["num_ssu"]) == num_ssu
                )
                - min(
                    float(row["npu_utilization"])
                    for row in whole_result_rows
                    if int(row["num_ssu"]) == num_ssu
                )
            )
            for num_ssu in SSU_COUNTS
        )
        _require(
            _close(
                recomputed_maximum_spread_pp,
                float(
                    results_payload[
                        "maximum_strategy_utilization_spread_percentage_points"
                    ]
                ),
            ),
            "self-test dynamic utilization-spread summary does not recompute",
        )
        with (output / "adaptive_decisions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            exported_decisions = list(csv.DictReader(handle))
        for num_ssu in SSU_COUNTS:
            selected_decisions = [
                row
                for row in exported_decisions
                if int(row["num_ssu"]) == num_ssu
            ]
            decision_summary = results_payload["adaptive_decision_summary"][
                str(num_ssu)
            ]
            times_ms = sorted(
                float(row["relative_time_s"]) * 1000.0
                for row in selected_decisions
            )
            gaps_ms = [
                right - left for left, right in zip(times_ms, times_ms[1:])
            ]
            _require(
                int(decision_summary["measurement_evaluations"])
                == len(selected_decisions)
                and int(
                    decision_summary["measurement_commits_reconstructed"]
                )
                == sum(
                    int(int(row["changed_cir_entry_count"]) > 0)
                    for row in selected_decisions
                )
                and _close(
                    float(decision_summary["inter_evaluation_gap_ms_min"]),
                    min(gaps_ms),
                )
                and _close(
                    float(decision_summary["inter_evaluation_gap_ms_median"]),
                    statistics.median(gaps_ms),
                )
                and _close(
                    float(decision_summary["inter_evaluation_gap_ms_max"]),
                    max(gaps_ms),
                ),
                f"self-test SSU{num_ssu} dynamic control summary does not recompute",
            )
        report_text = (output / "report.md").read_text(encoding="utf-8")
        _require(
            "α=2 tuned" in report_text
            and "TTFT = request completion_time - admission_time" in report_text
            and "不把它冒充成独立 oracle" in report_text,
            "self-test report omitted tuning/TTFT/physical-demand scope guards",
        )
        manifest_payload = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        executing_analyzer_sha = _sha256(Path(__file__).resolve())
        copied_analyzer_sha = _sha256(
            output / "analyze_npu32_ssu345_timeline64.py"
        )
        manifest_analyzer_row = next(
            row
            for row in manifest_payload["files"]
            if row["path"] == "analyze_npu32_ssu345_timeline64.py"
        )
        _require(
            executing_analyzer_sha
            == copied_analyzer_sha
            == validation["analyzer_sha256"]
            == results_payload["analyzer_sha256"]
            == manifest_payload["analyzer_sha256"]
            == manifest_analyzer_row["sha256"]
            and executing_analyzer_sha
            in (output / "report.md").read_text(encoding="utf-8"),
            "analyzer SHA is not identical across execution/copy/validation/results/report/manifest",
        )
        with (output / "state_durations_500ms.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            block_state_count = sum(1 for _ in csv.DictReader(handle))
        _require(
            block_state_count
            == len(SSU_COUNTS) * len(CASES) * BLOCK_COUNT * NUM_NPU,
            "self-test block-state export row count mismatch",
        )
        for path in output.iterdir():
            if path.suffix not in {".json", ".csv", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            _require("synthetic-private-host-must-not-escape" not in text, f"private hostname escaped into {path.name}")
            _require("ghp_" not in text, f"token-like text escaped into {path.name}")
        negative_tests = _run_corruption_suite(root, inputs)
        return {
            "passed": True,
            "analysis": "npu32_ssu345_timeline64_v1",
            "synthetic_cases": 9,
            "synthetic_boundaries_per_case": BOUNDARY_COUNT,
            "synthetic_blocks_per_case": BLOCK_COUNT,
            "full_plot_pipeline_exercised": True,
            "independent_adaptive_v1_v2_oracles_passed": True,
            "completion_boundary_zero_remaining_active_command_exercised": True,
            "fragmented_measurement_reduction_and_decimal_identity_exercised": True,
            "previous_pinned_selection_order_exercised": True,
            "prefetch_15_layer_inventory_exercised": True,
            "coalesced_event_driven_trigger_exercised": True,
            "portable_metadata_scan_passed": True,
            "negative_corruption_tests_passed": len(negative_tests),
            "negative_corruption_test_names": negative_tests,
            "generated_file_count": result["file_count"],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="runner JSON shard or directory; repeat for multiple shards",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plots", action="store_true", help="validate and write tables/report without PNGs")
    parser.add_argument("--self-test", action="store_true", help="run the deterministic synthetic full-pipeline self-test")
    parser.add_argument("--compact-plots", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            result = _self_test()
        else:
            _require(args.input, "--input is required unless --self-test is used")
            result = analyze(
                args.input,
                args.output_dir,
                make_plots=not args.no_plots,
                compact_plots=args.compact_plots,
            )
    except AnalysisError as error:
        raise SystemExit(f"analysis failed: {error}") from error
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
