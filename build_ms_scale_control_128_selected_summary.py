"""Strictly audit selected128 shards and build the 128-NPU curve summary.

This post-processor delegates the formal shard gate to the runner's public
validator, then independently recomputes fingerprints and SLO metrics from the
serialized schema-v3 shards.  The dependency points from this file to the
runner, so adding or running it cannot expand the runner's source closure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
from typing import Mapping, Sequence

from ms_scale_control_experiment import (
    SELECTED128_EXPECTED_RUNTIME_IDENTITY,
    validate_selected128_campaign_document,
    validate_selected128_formal_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CAMPAIGN_SPEC = Path("campaigns/selected128_alpha_tuned_v1.json")
DEFAULT_RAW_DIR = Path("results/ms_scale_control/selected128_alpha_tuned_v1_raw")
DEFAULT_OUTPUT_DIR = Path(
    "results/ms_scale_control/selected128_alpha_tuned_v1_analysis"
)
EXPERIMENT = "128npu_selected_cir_control_alpha_tuned_v1"
DEFINITION = "selected128"
NUM_NPU = 128
N_LAYERS = 16
BATCH_SIZE = 1
SSUS = (8, 12, 16, 20, 24, 40, 72)
INTERVALS_MS = (25.0, 100.0, 200.0)
SEED = 42
BACKING_REQUESTS_PER_NPU = 128
WARMUP_REQUESTS_PER_NPU = 8
SETTLE_MS = 500.0
MEASUREMENT_MS = 8_000.0
BLOCK_MS = 500.0
PRIMARY_ALPHA = 2.0
SENSITIVITY_ALPHA = 1.5
SLO_EPSILON = 1e-12
THREAD_LIMIT_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPECTED_THREAD_LIMITS = {name: "1" for name in THREAD_LIMIT_NAMES}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CATEGORIES = frozenset(("SS", "SL", "LS", "LL"))
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
GLOBAL_INPUT_FINGERPRINT_FIELDS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
)
EXPECTED_INVARIANTS = frozenset(
    (
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
    )
)


class SummaryError(ValueError):
    """Raised when a shard cannot support the formal selected128 curve."""


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    kind: str
    pressure_ttl_ms: float
    min_interval_ms: float
    tuning_slo_alpha: float | None = None
    explicit_spill_threshold: float | None = None
    target_ratio: float | None = None
    required_ratio: float | None = None
    background_reserve_fraction: float | None = None

    def case_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "pressure_ttl_ms": self.pressure_ttl_ms,
            "cir_write_threshold_gbps": 0.0,
            "min_interval_ms": self.min_interval_ms,
        }
        if self.kind == "adaptive":
            result.update(
                {
                    "tuning_slo_alpha": self.tuning_slo_alpha,
                    "explicit_spill_threshold": self.explicit_spill_threshold,
                    "target_ratio": self.target_ratio,
                    "required_ratio": self.required_ratio,
                    "background_reserve_fraction": (self.background_reserve_fraction),
                }
            )
        return result

    def adaptive_profile(self) -> dict[str, object] | None:
        if self.kind != "adaptive":
            return None
        return {
            "tuning_slo_alpha": self.tuning_slo_alpha,
            "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
            "explicit_spill_threshold": self.explicit_spill_threshold,
            "target_ratio": self.target_ratio,
            "required_ratio": self.required_ratio,
            "background_reserve_fraction": self.background_reserve_fraction,
        }


def _case_specs() -> tuple[CaseSpec, ...]:
    specs = [
        CaseSpec("baseline", "baseline", "baseline", 0.0, 0.0),
        CaseSpec("layer_once_ttl_0ms", "ttl", "layer_once", 0.0, 0.0),
        CaseSpec("layer_once_ttl_2ms", "ttl", "layer_once", 2.0, 0.0),
        CaseSpec("layer_once_ttl_5ms", "ttl", "layer_once", 5.0, 0.0),
    ]
    profiles = (
        (
            "a1p5",
            1.5,
            0.75,
            1.0 / 1.5 + 0.02,
            1.0 / 1.5,
        ),
        ("a2", 2.0, 0.75, 0.52, 0.50),
    )
    for alpha_name, alpha, spill, target, required in profiles:
        for interval in INTERVALS_MS:
            specs.append(
                CaseSpec(
                    f"adaptive_{alpha_name}_t0_i{int(interval)}ms",
                    f"adaptive_alpha_{alpha_name}",
                    "adaptive",
                    0.0,
                    interval,
                    alpha,
                    spill,
                    target,
                    required,
                    0.05,
                )
            )
    return tuple(specs)


CASES = _case_specs()
CASE_BY_NAME = {case.name: case for case in CASES}
CASE_INDEX = {case.name: index for index, case in enumerate(CASES)}
CASE_NAMES = tuple(case.name for case in CASES)

OUTPUT_FIELDS = (
    "case",
    "family",
    "kind",
    "tuning_slo_alpha",
    "num_ssu",
    "pressure_ttl_ms",
    "cir_write_threshold_gbps",
    "min_interval_ms",
    "explicit_spill_threshold",
    "target_ratio",
    "required_ratio",
    "background_reserve_fraction",
    "mean_npu_utilization_pct",
    "primary_slo_alpha",
    "sensitivity_slo_alpha",
    "alpha_1p5_equal_npu_slo_pct",
    "alpha_1p5_request_weighted_slo_pct",
    "alpha_2_equal_npu_slo_pct",
    "alpha_2_request_weighted_slo_pct",
    "measurement_request_count",
    "measurement_requests_per_npu_min",
    "measurement_requests_per_npu_median",
    "measurement_requests_per_npu_max",
    "measurement_cohort_fingerprint",
    "input_simulator_fingerprint",
    "source_fingerprint",
    "config_fingerprint",
    "definition_fingerprint",
    "campaign_spec_sha256",
    "all_invariants_passed",
    "source_artifact",
)


@dataclass(frozen=True)
class ShardDocument:
    path: Path
    payload: dict[str, object]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CampaignDocument:
    path: Path
    sha256: str
    size_bytes: int

    @property
    def authentication(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def _integer(value: object, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise SummaryError(f"{context}: bool is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise SummaryError(f"{context}: expected an integer")
    if minimum is not None:
        _require(result >= minimum, f"{context}: must be at least {minimum}")
    return result


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryError(f"{context}: expected a number") from error
    _require(math.isfinite(result), f"{context}: expected a finite number")
    return result


def _close(
    actual: object,
    expected: float,
    context: str,
    *,
    tolerance: float = 1e-10,
) -> float:
    result = _finite(actual, context)
    _require(
        math.isclose(result, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{context}: {result!r} != {expected!r}",
    )
    return result


def _fraction(value: object, context: str) -> float:
    result = _finite(value, context)
    _require(0.0 <= result <= 1.0, f"{context}: outside [0, 1]")
    return result


def _sha256_value(value: object, context: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{context}: expected lowercase SHA-256",
    )
    return value


def _canonical_hash(value: object, namespace: bytes = b"") -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(namespace + encoded).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise SummaryError(
            f"path is outside the project and cannot be recorded portably: {path}"
        ) from error


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise SummaryError(f"non-finite JSON constant: {value}")


def _read_shard(path: Path) -> ShardDocument:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SummaryError(f"cannot read shard {path}: {error}") from error
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"invalid UTF-8 JSON shard {path}: {error}") from error
    _require(isinstance(payload, dict), f"{path}: top level is not an object")
    return ShardDocument(path.resolve(), payload, sha256(raw).hexdigest(), len(raw))


def _read_campaign_spec(path: Path) -> CampaignDocument:
    resolved = path.expanduser().resolve()
    _portable_path(resolved)
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise SummaryError(f"cannot read campaign spec {resolved}: {error}") from error
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SummaryError(
            f"campaign spec must be duplicate-free UTF-8 JSON: {resolved}: {error}"
        ) from error
    _require(isinstance(document, dict), "campaign spec must be a JSON object")
    try:
        validate_selected128_campaign_document(document)
    except (KeyError, TypeError, ValueError) as error:
        raise SummaryError(f"formal campaign spec is invalid: {error}") from error
    return CampaignDocument(
        path=resolved,
        sha256=sha256(raw).hexdigest(),
        size_bytes=len(raw),
    )


def _expected_definition() -> dict[str, object]:
    return {
        "key": DEFINITION,
        "experiment_name": EXPERIMENT,
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssus": list(SSUS),
        "cases": [case.case_dict() for case in CASES],
        "default_requests_per_npu": BACKING_REQUESTS_PER_NPU,
        "default_measurement_ms": MEASUREMENT_MS,
        "adaptive": {
            "controller": "AdaptiveAdmissionSchemeBControllerV2_1",
            "explicit_spill_threshold": 0.75,
            "target_ratio": 0.52,
            "required_ratio": 0.50,
            "background_reserve_fraction": 0.05,
        },
        "report_roles": [],
        "require_single_ssu_simulation": True,
    }


def _expected_definition_fingerprint() -> str:
    return _canonical_hash(_expected_definition(), b"ms-scale-control-definition:v1\0")


def _validate_thread_limits(value: object, context: str) -> None:
    _require(
        value == EXPECTED_THREAD_LIMITS,
        f"{context}: every BLAS/OpenMP thread limit must be the string '1'; "
        f"got {value!r}",
    )


def _validate_source_manifest(payload: Mapping[str, object], context: str) -> None:
    manifest = payload.get("source_manifest")
    _require(
        isinstance(manifest, dict) and manifest, f"{context}: source manifest missing"
    )
    for relative_name, digest in manifest.items():
        _require(
            isinstance(relative_name, str)
            and relative_name
            and not Path(relative_name).is_absolute()
            and ".." not in Path(relative_name).parts,
            f"{context}: unsafe source manifest path {relative_name!r}",
        )
        _sha256_value(digest, f"{context}/source_manifest/{relative_name}")
    _require("data" in manifest, f"{context}: authenticated data file is absent")
    expected = _canonical_hash(manifest, b"ms-scale-control-source:v1\0")
    _require(
        payload.get("source_fingerprint") == expected,
        f"{context}: source fingerprint does not authenticate source_manifest",
    )


def _validate_experiment_spec(
    payload: Mapping[str, object], context: str
) -> dict[str, object]:
    spec = payload.get("experiment_spec")
    _require(isinstance(spec, dict), f"{context}: experiment_spec missing")
    _require(spec.get("schema_version") == 3, f"{context}: spec schema is not 3")
    _require(spec.get("experiment") == EXPERIMENT, f"{context}: wrong experiment")
    _require(spec.get("definition") == DEFINITION, f"{context}: wrong definition")
    _require(spec.get("num_npu") == NUM_NPU, f"{context}: expected 128 NPUs")
    _require(spec.get("n_layers") == N_LAYERS, f"{context}: wrong layer count")
    _require(spec.get("batch_size") == BATCH_SIZE, f"{context}: wrong batch size")
    _require(
        spec.get("default_ssu_list") == list(SSUS),
        f"{context}: SSU definition differs from {SSUS}",
    )
    _require(
        spec.get("cases") == [case.case_dict() for case in CASES],
        f"{context}: ten-case selected128 definition differs",
    )
    expected_profiles = {
        case.name: case.adaptive_profile() for case in CASES if case.kind == "adaptive"
    }
    _require(
        spec.get("adaptive_case_profiles") == expected_profiles,
        f"{context}: SLO-tuned Adaptive profiles differ",
    )
    _require(spec.get("report_roles") == {}, f"{context}: unexpected report roles")
    _require(
        spec.get("definition_fingerprint") == _expected_definition_fingerprint(),
        f"{context}: definition fingerprint mismatch",
    )
    _validate_thread_limits(
        spec.get("thread_limit_environment"), f"{context}/spec/thread limits"
    )
    steady = spec.get("steady_state")
    expected_steady = {
        "seed": SEED,
        "requests_per_npu": BACKING_REQUESTS_PER_NPU,
        "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
        "settle_ms": SETTLE_MS,
        "measurement_ms": MEASUREMENT_MS,
        "block_ms": BLOCK_MS,
        "slo_alpha": PRIMARY_ALPHA,
        "calibration_mode": False,
    }
    _require(steady == expected_steady, f"{context}: steady-state shape differs")
    scale = spec.get("scale_semantics")
    _require(isinstance(scale, dict), f"{context}: scale_semantics missing")
    _require(scale.get("num_npu") == NUM_NPU, f"{context}: scale NPU mismatch")
    _require(
        scale.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: backing mismatch",
    )
    _require(
        scale.get("total_assignment_count") == NUM_NPU * BACKING_REQUESTS_PER_NPU,
        f"{context}: total assignment count mismatch",
    )
    workload = spec.get("workload")
    _require(isinstance(workload, dict), f"{context}: workload spec missing")
    _require(
        workload.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{context}: wrong workload mode",
    )
    _require(workload.get("seed") == SEED, f"{context}: workload seed mismatch")
    _require(
        workload.get("requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: workload backing mismatch",
    )
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _sha256_value(workload.get(field), f"{context}/workload/{field}")
    _sha256_value(
        workload.get("prefix_32_assignment_hash"),
        f"{context}/workload/prefix_32_assignment_hash",
    )
    _sha256_value(
        workload.get("full_assignment_hash"),
        f"{context}/workload/full_assignment_hash",
    )
    _require(
        workload.get("sampling")
        == "IID uniform with replacement over all 84 data profiles",
        f"{context}: sampling contract differs",
    )
    _require(
        workload.get("per_npu_streams") == "independent and prefix-stable",
        f"{context}: per-NPU stream contract differs",
    )
    _require(
        workload.get("scientific_prefix_requests_per_npu") == 32,
        f"{context}: scientific prefix length mismatch",
    )
    _require(
        spec.get("cross_request_layer0_prefetch") is True,
        f"{context}: cross-request Layer-0 prefetch is not enabled",
    )
    _require(
        spec.get("placement") == "token-block ring hash reused across all 16 layers",
        f"{context}: placement contract differs",
    )
    source_files = spec.get("source_files")
    source_manifest = payload.get("source_manifest")
    _require(
        isinstance(source_files, list)
        and isinstance(source_manifest, dict)
        and set(source_files) == set(source_manifest),
        f"{context}: source_files differ from source_manifest",
    )
    campaign = _sha256_value(
        spec.get("campaign_spec_sha256"), f"{context}/campaign spec"
    )
    _require(
        payload.get("campaign_spec_sha256") == campaign,
        f"{context}: payload/spec campaign mismatch",
    )
    expected_config = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
    _require(
        payload.get("config_fingerprint") == expected_config,
        f"{context}: config fingerprint does not authenticate experiment_spec",
    )
    return spec


def _validate_campaign(
    payload: Mapping[str, object],
    expected_authentication: Mapping[str, object],
    context: str,
) -> None:
    campaign = _sha256_value(
        payload.get("campaign_spec_sha256"), f"{context}/campaign SHA"
    )
    authentication = payload.get("campaign_spec_authentication")
    _require(
        isinstance(authentication, dict),
        f"{context}: campaign authentication missing",
    )
    _require(
        authentication == expected_authentication,
        f"{context}: campaign authentication differs from local frozen bytes",
    )
    _integer(
        authentication.get("size_bytes"),
        f"{context}/campaign size",
        minimum=1,
    )
    _require(
        payload.get("ending_campaign_spec_authentication") == authentication,
        f"{context}: campaign bytes changed during run",
    )
    _require(
        payload.get("campaign_spec_stable_during_run") is True,
        f"{context}: campaign was not stable",
    )


def _catalog_and_schedule(
    payload: Mapping[str, object], context: str
) -> tuple[dict[tuple[int, int], tuple[float, ...]], dict[int, tuple]]:
    metadata = payload.get("schedule_metadata")
    _require(isinstance(metadata, dict), f"{context}: schedule metadata missing")
    _require(
        metadata.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{context}: schedule mode",
    )
    _require(metadata.get("seed") == SEED, f"{context}: schedule seed")
    _require(metadata.get("num_npu") == NUM_NPU, f"{context}: schedule NPU count")
    _require(
        metadata.get("requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: schedule backing",
    )
    _require(
        metadata.get("request_id_formula") == "sequence * num_npu + npu_id",
        f"{context}: request ID formula differs",
    )
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _sha256_value(metadata.get(field), f"{context}/schedule/{field}")

    catalog_rows = metadata.get("catalog_rows")
    _require(
        isinstance(catalog_rows, list) and len(catalog_rows) == 84,
        f"{context}: expected 84 catalog rows",
    )
    catalog: dict[tuple[int, int], tuple[float, ...]] = {}
    for index, row in enumerate(catalog_rows):
        _require(
            isinstance(row, list) and len(row) == 3,
            f"{context}/catalog[{index}]: malformed row",
        )
        key = (
            _integer(row[0], f"{context}/catalog[{index}]/seq", minimum=1),
            _integer(row[1], f"{context}/catalog[{index}]/nql", minimum=1),
        )
        _require(key not in catalog, f"{context}: duplicate catalog profile {key}")
        values = row[2]
        _require(
            isinstance(values, list) and len(values) == 4,
            f"{context}/catalog[{index}]: expected four values",
        )
        converted = tuple(
            _finite(value, f"{context}/catalog[{index}]/value") for value in values
        )
        _require(
            all(value > 0.0 for value in converted),
            f"{context}: nonpositive catalog value",
        )
        catalog[key] = converted

    assignment_rows = metadata.get("assignment_rows")
    expected_count = NUM_NPU * BACKING_REQUESTS_PER_NPU
    _require(
        isinstance(assignment_rows, list) and len(assignment_rows) == expected_count,
        f"{context}: expected {expected_count} assignment rows",
    )
    assignments: dict[int, tuple] = {}
    for request_id, row in enumerate(assignment_rows):
        _require(
            isinstance(row, list) and len(row) == 5,
            f"{context}/assignment[{request_id}]: malformed row",
        )
        _require(row[0] == request_id, f"{context}: assignment IDs are not contiguous")
        expected_npu = request_id % NUM_NPU
        expected_sequence = request_id // NUM_NPU
        _require(row[1] == expected_npu, f"{context}: assignment NPU formula mismatch")
        _require(
            row[2] == expected_sequence,
            f"{context}: assignment sequence formula mismatch",
        )
        _require(row[3] in CATEGORIES, f"{context}: invalid assignment category")
        _require(
            isinstance(row[4], list) and len(row[4]) == 2,
            f"{context}: malformed profile key",
        )
        profile = (_integer(row[4][0], context), _integer(row[4][1], context))
        _require(
            profile in catalog, f"{context}: assignment uses unknown profile {profile}"
        )
        assignments[request_id] = (
            expected_npu,
            expected_sequence,
            row[3],
            profile,
        )
    return catalog, assignments


def _validate_path_abi(value: object, context: str) -> None:
    _require(isinstance(value, dict), f"{context}: path ABI missing")
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
    }
    for field, expected_value in expected.items():
        _require(
            value.get(field) == expected_value, f"{context}: path ABI {field} mismatch"
        )
    _sha256_value(value.get("assigned_paths_sha256"), f"{context}/assigned paths")


def _validate_input_authentication(value: object, context: str) -> None:
    _require(isinstance(value, dict), f"{context}: input authentication missing")
    _require(value.get("source") == "data", f"{context}: workload source is not data")
    _require(value.get("profile_count") == 84, f"{context}: profile count is not 84")
    for field in ("source_sha256", "catalog_hash", "table_fingerprint"):
        _sha256_value(value.get(field), f"{context}/{field}")


def _slo_metrics(
    request_rows: Sequence[Mapping[str, object]],
    alpha: float,
    context: str,
) -> tuple[float, float, list[int]]:
    outcomes: dict[int, list[float]] = {npu: [] for npu in range(NUM_NPU)}
    weighted: list[float] = []
    for index, request in enumerate(request_rows):
        npu = _integer(request.get("npu_id"), f"{context}/request[{index}]/npu")
        _require(0 <= npu < NUM_NPU, f"{context}: NPU outside 128-NPU fleet")
        ttft = _finite(request.get("ttft_ms"), f"{context}/request[{index}]/ttft")
        ideal = _finite(
            request.get("ideal_ttft_ms"), f"{context}/request[{index}]/ideal"
        )
        _require(ttft >= 0.0 and ideal > 0.0, f"{context}: invalid TTFT/ideal")
        passed = float(ttft <= alpha * ideal + SLO_EPSILON)
        outcomes[npu].append(passed)
        weighted.append(passed)
    _require(weighted, f"{context}: empty measurement cohort")
    missing = [npu for npu, values in outcomes.items() if not values]
    _require(not missing, f"{context}: NPUs without measured requests: {missing}")
    counts = [len(outcomes[npu]) for npu in range(NUM_NPU)]
    equal_npu = fmean(fmean(outcomes[npu]) for npu in range(NUM_NPU))
    request_weighted = fmean(weighted)
    return equal_npu, request_weighted, counts


def _validate_summary(
    summary: object,
    case: CaseSpec,
    num_ssu: int,
    catalog: Mapping[tuple[int, int], tuple[float, ...]],
    assignments: Mapping[int, tuple],
    stored_cohort_hash: str,
    context: str,
) -> dict[str, object]:
    _require(isinstance(summary, dict), f"{context}: steady_summary missing")
    _require(summary.get("schema_version") == 2, f"{context}: summary schema is not 2")
    _require(summary.get("mode") == "steady_state_full_load", f"{context}: wrong mode")
    _require(summary.get("num_npu") == NUM_NPU, f"{context}: wrong NPU count")
    _require(summary.get("num_ssu") == num_ssu, f"{context}: wrong SSU count")
    _require(summary.get("n_layers") == N_LAYERS, f"{context}: wrong layer count")
    _require(summary.get("batch_size") == BATCH_SIZE, f"{context}: wrong batch size")
    _require(
        summary.get("warmup_requests_per_npu") == WARMUP_REQUESTS_PER_NPU,
        f"{context}: wrong warmup count",
    )
    _close(summary.get("settle_ms"), SETTLE_MS, f"{context}/settle")
    _close(summary.get("slo_alpha"), PRIMARY_ALPHA, f"{context}/primary alpha")
    _close(
        summary.get("measurement_duration_ms"),
        MEASUREMENT_MS,
        f"{context}/measurement duration",
    )
    _close(
        summary.get("pressure_ttl_ms"),
        case.pressure_ttl_ms,
        f"{context}/pressure TTL",
    )
    _close(
        summary.get("cir_write_threshold_gbps"),
        0.0,
        f"{context}/CIR write threshold",
    )
    if case.kind == "adaptive":
        _close(
            summary.get("control_min_interval_ms"),
            case.min_interval_ms,
            f"{context}/control interval",
        )
    else:
        _require(
            summary.get("control_min_interval_ms") is None,
            f"{context}: static strategy unexpectedly has CIR control",
        )

    invariants = summary.get("invariants")
    _require(isinstance(invariants, dict), f"{context}: invariants missing")
    _require(
        EXPECTED_INVARIANTS <= set(invariants),
        f"{context}: required invariants missing: {sorted(EXPECTED_INVARIANTS - set(invariants))}",
    )
    _require(
        all(value is True for value in invariants.values()),
        f"{context}: at least one simulator invariant failed: {invariants}",
    )

    start = _finite(summary.get("measurement_start_ms"), f"{context}/start")
    end = _finite(summary.get("measurement_end_ms"), f"{context}/end")
    warmup = _finite(summary.get("warmup_reached_ms"), f"{context}/warmup")
    drain = _finite(summary.get("drain_stop_ms"), f"{context}/drain")
    tail = _finite(summary.get("tail_drain_ms"), f"{context}/tail drain")
    _require(warmup + SETTLE_MS <= start + 1e-9, f"{context}: settle window is short")
    _close(end - start, MEASUREMENT_MS, f"{context}/window length", tolerance=1e-8)
    _require(drain >= end and tail >= 0.0, f"{context}: invalid drain timing")
    _close(drain - end, tail, f"{context}/tail drain relation", tolerance=1e-8)

    request_rows = summary.get("request_rows")
    _require(
        isinstance(request_rows, list) and request_rows,
        f"{context}: request rows missing",
    )
    request_ids: list[int] = []
    cohort_material = []
    for index, request in enumerate(request_rows):
        _require(
            isinstance(request, dict), f"{context}/request[{index}]: not an object"
        )
        request_id = _integer(request.get("request_id"), f"{context}/request ID")
        _require(
            request_id in assignments, f"{context}: request outside frozen schedule"
        )
        npu, sequence, category, profile = assignments[request_id]
        _require(
            request.get("npu_id") == npu,
            f"{context}: request NPU differs from schedule",
        )
        _require(
            request.get("sequence") == sequence,
            f"{context}: sequence differs from schedule",
        )
        _require(
            request.get("category") == category,
            f"{context}: category differs from schedule",
        )
        _require(
            request.get("profile_key") == list(profile),
            f"{context}: profile differs from schedule",
        )
        _require(
            request.get("profile_id") == f"{profile[0]},{profile[1]}",
            f"{context}: profile_id mismatch",
        )
        _require(
            request.get("profile_name") == f"seq_len_k={profile[0]},nql={profile[1]}",
            f"{context}: profile_name mismatch",
        )
        raw_demand, per_layer_us, _ttft_table, _work_gb = catalog[profile]
        _close(request.get("raw_demand_gbps"), raw_demand, f"{context}/raw demand")
        ideal = N_LAYERS * per_layer_us / 1000.0
        _close(
            request.get("ideal_ttft_ms"), ideal, f"{context}/ideal TTFT", tolerance=1e-8
        )
        admission = _finite(request.get("admission_time_ms"), f"{context}/admission")
        completion = _finite(request.get("completion_time_ms"), f"{context}/completion")
        ttft = _finite(request.get("ttft_ms"), f"{context}/TTFT")
        _require(
            start <= admission < end, f"{context}: admission outside half-open window"
        )
        _require(completion >= admission, f"{context}: completion before admission")
        _require(completion <= drain + 1e-8, f"{context}: completion after drain stop")
        _close(completion - admission, ttft, f"{context}/TTFT relation", tolerance=1e-8)
        _require(
            ttft + 1e-8 >= ideal, f"{context}: TTFT below pure-compute lower bound"
        )
        expected_primary = ttft <= PRIMARY_ALPHA * ideal + SLO_EPSILON
        _require(
            request.get("slo_met") is expected_primary,
            f"{context}: stored alpha=2 outcome mismatch",
        )
        request_ids.append(request_id)
        cohort_material.append([request_id, npu, sequence, list(profile)])
    _require(
        request_ids == sorted(request_ids)
        and len(request_ids) == len(set(request_ids)),
        f"{context}: request IDs are not unique and sorted",
    )
    cohort_hash = _canonical_hash(
        cohort_material, b"ms-scale-control-measurement-cohort:v1\0"
    )
    _require(
        cohort_hash == stored_cohort_hash, f"{context}: cohort fingerprint mismatch"
    )

    alpha_1p5 = _slo_metrics(request_rows, SENSITIVITY_ALPHA, context)
    alpha_2 = _slo_metrics(request_rows, PRIMARY_ALPHA, context)
    counts = alpha_2[2]
    _require(
        summary.get("request_counts_by_npu") == counts,
        f"{context}: request_counts_by_npu mismatch",
    )
    _require(
        summary.get("measurement_request_count") == len(request_rows),
        f"{context}: measurement request count mismatch",
    )
    _close(
        summary.get("ttft_slo_attainment"),
        alpha_2[0],
        f"{context}/stored equal-NPU alpha2",
    )
    _close(
        summary.get("request_weighted_slo_attainment"),
        alpha_2[1],
        f"{context}/stored request-weighted alpha2",
    )

    utilizations = summary.get("npu_utilizations")
    _require(
        isinstance(utilizations, list) and len(utilizations) == NUM_NPU,
        f"{context}: expected 128 NPU utilizations",
    )
    converted_utilizations = [
        _fraction(value, f"{context}/NPU utilization") for value in utilizations
    ]
    mean_utilization = fmean(converted_utilizations)
    _close(
        summary.get("mean_npu_utilization"),
        mean_utilization,
        f"{context}/mean NPU utilization",
    )
    compute_ms = summary.get("compute_ms_by_npu")
    _require(
        isinstance(compute_ms, list) and len(compute_ms) == NUM_NPU,
        f"{context}: compute_ms_by_npu missing",
    )
    for npu, (busy, utilization) in enumerate(zip(compute_ms, converted_utilizations)):
        _close(
            _finite(busy, f"{context}/compute[{npu}]") / MEASUREMENT_MS,
            utilization,
            f"{context}/compute utilization[{npu}]",
            tolerance=1e-9,
        )

    measurement_pressure = _integer(
        summary.get("measurement_pressure_reports"),
        f"{context}/measurement pressure reports",
        minimum=0,
    )
    measurement_evaluations = _integer(
        summary.get("measurement_control_evaluations"),
        f"{context}/measurement control evaluations",
        minimum=0,
    )
    measurement_writes = _integer(
        summary.get("measurement_cir_path_writes"),
        f"{context}/measurement CIR writes",
        minimum=0,
    )
    if case.kind == "baseline":
        _require(measurement_pressure == 0, f"{context}: baseline read pressure table")
    elif case.kind == "layer_once":
        _require(
            measurement_pressure > 0,
            f"{context}: layer_once made no true pressure read",
        )
    else:
        _require(measurement_pressure == 0, f"{context}: Adaptive read pressure table")
        _require(measurement_evaluations > 0, f"{context}: Adaptive never evaluated")
    if case.kind != "adaptive":
        _require(
            measurement_evaluations == 0, f"{context}: static strategy evaluated CIR"
        )
        _require(measurement_writes == 0, f"{context}: static strategy wrote CIR")

    return {
        "mean_npu_utilization_pct": 100.0 * mean_utilization,
        "alpha_1p5_equal_npu_slo_pct": 100.0 * alpha_1p5[0],
        "alpha_1p5_request_weighted_slo_pct": 100.0 * alpha_1p5[1],
        "alpha_2_equal_npu_slo_pct": 100.0 * alpha_2[0],
        "alpha_2_request_weighted_slo_pct": 100.0 * alpha_2[1],
        "measurement_request_count": len(request_rows),
        "measurement_requests_per_npu_min": min(counts),
        "measurement_requests_per_npu_median": median(counts),
        "measurement_requests_per_npu_max": max(counts),
    }


def _validate_row(
    row: object,
    case: CaseSpec,
    num_ssu: int,
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    catalog: Mapping[tuple[int, int], tuple[float, ...]],
    assignments: Mapping[int, tuple],
    context: str,
) -> dict[str, object]:
    _require(isinstance(row, dict), f"{context}: result row is not an object")
    _require(row.get("status") == "ok", f"{context}: row status is not ok")
    _require(row.get("case") == case.name, f"{context}: case name mismatch")
    _require(row.get("family") == case.family, f"{context}: family mismatch")
    _require(row.get("kind") == case.kind, f"{context}: kind mismatch")
    _require(row.get("role") is None, f"{context}: selected128 row has a role")
    _require(row.get("num_ssu") == num_ssu, f"{context}: row SSU mismatch")
    _require(row.get("num_npu") == NUM_NPU, f"{context}: row NPU mismatch")
    _require(row.get("definition") == DEFINITION, f"{context}: row definition mismatch")
    _require(
        row.get("definition_fingerprint") == _expected_definition_fingerprint(),
        f"{context}: row definition fingerprint mismatch",
    )
    _require(
        row.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: row backing mismatch",
    )
    _require(row.get("case_spec") == case.case_dict(), f"{context}: case spec mismatch")
    for field in ("source_fingerprint", "config_fingerprint", "campaign_spec_sha256"):
        _require(
            row.get(field) == payload.get(field), f"{context}: row {field} mismatch"
        )
    expected_case_fingerprint = _canonical_hash(
        {
            "case": case.case_dict(),
            "num_ssu": num_ssu,
            "source_fingerprint": payload["source_fingerprint"],
            "config_fingerprint": payload["config_fingerprint"],
        },
        b"ms-scale-control-case:v1\0",
    )
    _require(
        row.get("case_fingerprint") == expected_case_fingerprint,
        f"{context}: case fingerprint mismatch",
    )
    inputs = row.get("input_fingerprints")
    _require(
        isinstance(inputs, dict) and set(inputs) == set(INPUT_FINGERPRINT_FIELDS),
        f"{context}: input fingerprint fields differ",
    )
    for field in INPUT_FINGERPRINT_FIELDS:
        _sha256_value(inputs[field], f"{context}/input/{field}")
    workload = spec["workload"]
    expected_global = {
        "catalog": workload["catalog"],
        "recipe": workload["recipe"],
        "schedule": workload["schedule"],
        "assignment": workload["assignment"],
        "prefix_32_assignment": workload["prefix_32_assignment_hash"],
        "full_assignment": workload["full_assignment_hash"],
    }
    for field, expected in expected_global.items():
        _require(inputs[field] == expected, f"{context}: global input {field} mismatch")
    summary = row.get("steady_summary")
    _require(
        isinstance(summary, dict)
        and inputs["simulator"] == summary.get("input_fingerprint"),
        f"{context}: simulator input fingerprint mismatch",
    )
    cohort_hash = _sha256_value(
        row.get("measurement_cohort_fingerprint"), f"{context}/cohort fingerprint"
    )
    _require(
        isinstance(row.get("workload_statistics"), dict),
        f"{context}: workload stats missing",
    )
    _require(
        isinstance(row.get("prefix_32_workload_statistics"), dict),
        f"{context}: prefix workload stats missing",
    )
    prefix_materialized = row.get("prefix_32_materialized_fingerprints")
    _require(
        isinstance(prefix_materialized, dict)
        and set(prefix_materialized) == {"workload", "placement", "trace"},
        f"{context}: prefix materialized fingerprints differ",
    )
    for field, value in prefix_materialized.items():
        _sha256_value(value, f"{context}/prefix/{field}")
    _require(
        isinstance(row.get("cohort_profile_metrics"), dict),
        f"{context}: cohort metrics missing",
    )
    _require(
        isinstance(row.get("stationarity_diagnostics"), dict),
        f"{context}: stationarity diagnostics missing",
    )
    runtime = row.get("runtime")
    _require(isinstance(runtime, dict), f"{context}: worker runtime missing")
    _validate_thread_limits(
        runtime.get("thread_limit_environment"), f"{context}/worker threads"
    )
    _require(
        runtime.get("multiprocessing_start_method")
        == payload.get("execution", {}).get("multiprocessing_start_method"),
        f"{context}: worker multiprocessing mode mismatch",
    )
    _require(
        _finite(row.get("wall_time_s"), f"{context}/wall time") >= 0.0,
        f"{context}: negative wall time",
    )

    metrics = _validate_summary(
        summary,
        case,
        num_ssu,
        catalog,
        assignments,
        cohort_hash,
        context,
    )
    return {
        "case": case.name,
        "family": case.family,
        "kind": case.kind,
        "tuning_slo_alpha": (
            "" if case.tuning_slo_alpha is None else case.tuning_slo_alpha
        ),
        "num_ssu": num_ssu,
        "pressure_ttl_ms": case.pressure_ttl_ms,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": case.min_interval_ms,
        "explicit_spill_threshold": (
            ""
            if case.explicit_spill_threshold is None
            else case.explicit_spill_threshold
        ),
        "target_ratio": "" if case.target_ratio is None else case.target_ratio,
        "required_ratio": ("" if case.required_ratio is None else case.required_ratio),
        "background_reserve_fraction": (
            ""
            if case.background_reserve_fraction is None
            else case.background_reserve_fraction
        ),
        **metrics,
        "primary_slo_alpha": PRIMARY_ALPHA,
        "sensitivity_slo_alpha": SENSITIVITY_ALPHA,
        "measurement_cohort_fingerprint": cohort_hash,
        "input_simulator_fingerprint": inputs["simulator"],
        "source_fingerprint": payload["source_fingerprint"],
        "config_fingerprint": payload["config_fingerprint"],
        "definition_fingerprint": payload["definition_fingerprint"],
        "campaign_spec_sha256": payload["campaign_spec_sha256"],
        "all_invariants_passed": True,
    }


def _canonical_formal_gate(
    payload: Mapping[str, object],
    expected_campaign_sha256: str,
    context: str,
) -> tuple[int, dict[str, object]]:
    selected_ssus = payload.get("selected_ssus")
    _require(
        isinstance(selected_ssus, list)
        and len(selected_ssus) == 1
        and type(selected_ssus[0]) is int
        and selected_ssus[0] in SSUS,
        f"{context}: shard must select exactly one frozen SSU",
    )
    num_ssu = selected_ssus[0]
    try:
        identity = validate_selected128_formal_payload(
            payload,
            expected_ssu=num_ssu,
            expected_campaign_sha256=expected_campaign_sha256,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise SummaryError(
            f"{context}: canonical formal validation failed: {error}"
        ) from error
    _require(
        isinstance(identity, dict),
        f"{context}: canonical formal validator returned no identity",
    )
    return num_ssu, identity


def _validate_shard_root(
    shard: ShardDocument,
    campaign: CampaignDocument,
) -> tuple[
    int,
    dict[str, object],
    dict[tuple[str, int], dict[str, object]],
    dict[str, object],
]:
    payload = shard.payload
    context = _portable_path(shard.path)
    num_ssu, formal_identity = _canonical_formal_gate(payload, campaign.sha256, context)
    _require(payload.get("schema_version") == 3, f"{context}: shard schema is not 3")
    _require(payload.get("definition") == DEFINITION, f"{context}: wrong definition")
    _require(payload.get("num_npu") == NUM_NPU, f"{context}: expected 128 NPUs")
    _require(
        payload.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: backing is not 128 requests/NPU",
    )
    _require(
        payload.get("total_assignment_count") == NUM_NPU * BACKING_REQUESTS_PER_NPU,
        f"{context}: total assignment count mismatch",
    )
    _require(
        payload.get("selected_complete") is True,
        f"{context}: selected shard incomplete",
    )
    _require(
        payload.get("source_stable_during_run") is True, f"{context}: source changed"
    )
    _require(
        payload.get("config_stable_during_run") is True, f"{context}: config changed"
    )
    _require(
        payload.get("ending_source_fingerprint") == payload.get("source_fingerprint"),
        f"{context}: ending source fingerprint mismatch",
    )
    _require(
        payload.get("ending_config_fingerprint") == payload.get("config_fingerprint"),
        f"{context}: ending config fingerprint mismatch",
    )
    _require(
        payload.get("definition_fingerprint") == _expected_definition_fingerprint(),
        f"{context}: definition fingerprint mismatch",
    )
    _validate_source_manifest(payload, context)
    spec = _validate_experiment_spec(payload, context)
    _validate_campaign(payload, campaign.authentication, context)
    _validate_path_abi(payload.get("path_abi"), context)
    _validate_input_authentication(payload.get("input_authentication"), context)
    runtime = payload.get("runtime")
    _require(isinstance(runtime, dict), f"{context}: parent runtime missing")
    _validate_thread_limits(
        runtime.get("thread_limit_environment"), f"{context}/parent threads"
    )
    execution = payload.get("execution")
    _require(isinstance(execution, dict), f"{context}: execution metadata missing")
    _require(
        execution.get("multiprocessing_start_method") == "spawn",
        f"{context}: selected128 requires the frozen spawn mode",
    )
    _integer(execution.get("requested_max_workers"), f"{context}/workers", minimum=1)
    _require(
        execution.get("single_ssu_process_pool_required") is True,
        f"{context}: single-SSU process-pool contract absent",
    )

    _require(
        payload.get("selected_cases") == sorted(CASE_NAMES),
        f"{context}: selected case list differs",
    )
    expected_keys = [[name, num_ssu] for name in sorted(CASE_NAMES)]
    _require(
        payload.get("selected_keys") == expected_keys,
        f"{context}: selected keys differ",
    )
    pairing = payload.get("pairing_audit")
    _require(
        pairing
        == {
            str(num_ssu): {
                "cases": sorted(CASE_NAMES),
                "has_rows": True,
                "all_available_rows_paired": True,
            }
        },
        f"{context}: runner pairing audit differs",
    )

    catalog, assignments = _catalog_and_schedule(payload, context)
    rows = payload.get("results")
    _require(
        isinstance(rows, list) and len(rows) == len(CASES),
        f"{context}: expected ten result rows",
    )
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    compact_rows = {}
    for row in rows:
        _require(isinstance(row, dict), f"{context}: malformed result row")
        case_name = row.get("case")
        _require(case_name in CASE_BY_NAME, f"{context}: unknown case {case_name!r}")
        key = (case_name, num_ssu)
        _require(key not in by_key, f"{context}: duplicate result {key}")
        by_key[key] = row
        compact = _validate_row(
            row,
            CASE_BY_NAME[case_name],
            num_ssu,
            payload,
            spec,
            catalog,
            assignments,
            f"{context}/{case_name}/SSU{num_ssu}",
        )
        compact["source_artifact"] = context
        compact_rows[key] = compact
    _require(
        set(by_key) == {(name, num_ssu) for name in CASE_NAMES},
        f"{context}: incomplete case grid",
    )

    for field in INPUT_FINGERPRINT_FIELDS:
        values = {row["input_fingerprints"][field] for row in by_key.values()}
        _require(
            len(values) == 1, f"{context}: {field} is not paired within SSU{num_ssu}"
        )
    for field in (
        "workload_statistics",
        "prefix_32_workload_statistics",
        "prefix_32_materialized_fingerprints",
    ):
        values = {_canonical_hash(row[field]) for row in by_key.values()}
        _require(len(values) == 1, f"{context}: {field} differs within SSU{num_ssu}")
    return num_ssu, spec, compact_rows, formal_identity


def _validate_shards(
    shards: Sequence[ShardDocument],
    campaign: CampaignDocument,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _require(len(shards) == len(SSUS), f"expected seven shards, got {len(shards)}")
    compact_by_key: dict[tuple[str, int], dict[str, object]] = {}
    shard_by_ssu: dict[int, ShardDocument] = {}
    reference: ShardDocument | None = None
    reference_spec = None
    reference_identity = None
    for shard in shards:
        num_ssu, spec, compact, identity = _validate_shard_root(shard, campaign)
        _require(num_ssu not in shard_by_ssu, f"duplicate shard for SSU{num_ssu}")
        shard_by_ssu[num_ssu] = shard
        compact_by_key.update(compact)
        if reference is None:
            reference = shard
            reference_spec = spec
            reference_identity = identity
            continue
        _require(
            identity == reference_identity,
            f"{_portable_path(shard.path)}: cross-shard formal identity mismatch",
        )
        for field in (
            "source_fingerprint",
            "ending_source_fingerprint",
            "source_manifest",
            "definition_fingerprint",
            "config_fingerprint",
            "ending_config_fingerprint",
            "campaign_spec_sha256",
            "campaign_spec_authentication",
            "ending_campaign_spec_authentication",
            "experiment_spec",
            "input_authentication",
            "schedule_metadata",
        ):
            _require(
                shard.payload.get(field) == reference.payload.get(field),
                f"{_portable_path(shard.path)}: cross-shard {field} mismatch",
            )
    _require(set(shard_by_ssu) == set(SSUS), f"SSU shards differ from {SSUS}")
    _require(
        reference is not None
        and reference_spec is not None
        and reference_identity is not None,
        "no reference shard",
    )

    expected_global = None
    for num_ssu in SSUS:
        shard = shard_by_ssu[num_ssu]
        rows = shard.payload["results"]
        current = {
            field: rows[0]["input_fingerprints"][field]
            for field in GLOBAL_INPUT_FINGERPRINT_FIELDS
        }
        if expected_global is None:
            expected_global = current
        else:
            _require(
                current == expected_global,
                f"SSU{num_ssu}: global input fingerprints differ",
            )

    expected_keys = {(name, ssu) for ssu in SSUS for name in CASE_NAMES}
    _require(
        set(compact_by_key) == expected_keys, "70-row selected128 grid is incomplete"
    )
    ordered = [compact_by_key[(case.name, ssu)] for ssu in SSUS for case in CASES]
    provenance = {
        "source_fingerprint": reference.payload["source_fingerprint"],
        "source_manifest": reference.payload["source_manifest"],
        "definition_fingerprint": reference.payload["definition_fingerprint"],
        "config_fingerprint": reference.payload["config_fingerprint"],
        "campaign_spec_sha256": reference.payload["campaign_spec_sha256"],
        "campaign_spec": {
            "path": _portable_path(campaign.path),
            **campaign.authentication,
        },
        "campaign_spec_authentication": reference.payload[
            "campaign_spec_authentication"
        ],
        "input_authentication": reference.payload["input_authentication"],
        "global_input_fingerprints": expected_global,
        "thread_limit_environment": EXPECTED_THREAD_LIMITS,
        "runtime_identity": reference_identity["runtime_identity"],
        "experiment_spec": reference_spec,
        "shards": [
            {
                "num_ssu": num_ssu,
                "path": _portable_path(shard_by_ssu[num_ssu].path),
                "sha256": shard_by_ssu[num_ssu].sha256,
                "size_bytes": shard_by_ssu[num_ssu].size_bytes,
                "row_count": len(shard_by_ssu[num_ssu].payload["results"]),
            }
            for num_ssu in SSUS
        ],
    }
    return ordered, provenance


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    value = stream.getvalue()
    _require("/home/" not in value, "summary CSV contains an absolute home path")
    return value


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _assert_portable_manifest(value: object, context: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_portable_manifest(item, f"{context}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable_manifest(item, f"{context}[{index}]")
    elif isinstance(value, str):
        _require("/home/" not in value, f"{context}: contains an absolute home path")
        _require(not value.startswith("/"), f"{context}: contains an absolute path")


def _build_outputs(
    shards: Sequence[ShardDocument],
    campaign: CampaignDocument,
    output_dir: Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    rows, provenance = _validate_shards(shards, campaign)
    output_dir = output_dir.expanduser().resolve()
    _portable_path(output_dir)
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "manifest.json"
    summary_text = _csv_text(rows)
    _atomic_write_text(summary_path, summary_text)
    summary_digest = sha256(summary_text.encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "analysis": "selected128_alpha_tuned_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "num_npu": NUM_NPU,
        "seed": SEED,
        "ssu_counts": list(SSUS),
        "case_count": len(CASES),
        "row_count": len(rows),
        "cases": [case.case_dict() for case in CASES],
        "run_shape": {
            "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
            "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
            "settle_ms": SETTLE_MS,
            "measurement_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "primary_slo_alpha": PRIMARY_ALPHA,
            "calibration_mode": False,
        },
        "metric_definitions": {
            "request_slo": "ttft_ms <= alpha * ideal_ttft_ms + 1e-12",
            "equal_npu_slo": "mean of per-NPU request SLO attainment across 128 NPUs",
            "mean_npu_utilization": "mean of 128 measurement-window NPU compute utilizations",
            "reported_alphas": [SENSITIVITY_ALPHA, PRIMARY_ALPHA],
        },
        "summary_csv": {
            "path": _portable_path(summary_path),
            "sha256": summary_digest,
            "size_bytes": len(summary_text.encode()),
        },
        **provenance,
    }
    _assert_portable_manifest(manifest)
    manifest_text = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write_text(manifest_path, manifest_text)
    return summary_path, manifest_path, rows


def _discover_shards(raw_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        paths = [path.expanduser().resolve() for path in explicit]
    else:
        resolved = raw_dir.expanduser().resolve()
        _require(
            resolved.is_dir(),
            "raw shard directory does not exist: "
            f"{resolved}; expected seven schema-v3 selected128 JSON shards "
            f"(one per SSU: {', '.join(map(str, SSUS))})",
        )
        paths = sorted(resolved.glob("*.json"))
    _require(
        len(paths) == len(SSUS),
        f"expected exactly seven JSON shards, found {len(paths)}: "
        + ", ".join(path.name for path in paths),
    )
    _require(len(set(paths)) == len(paths), "the same shard path was supplied twice")
    for path in paths:
        _portable_path(path)
        _require(path.is_file(), f"shard is not a regular file: {path}")
    return paths


def _fake_sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _synthetic_documents() -> tuple[list[ShardDocument], CampaignDocument]:
    catalog_rows = []
    profile_keys = []
    for index in range(84):
        key = (32 + index, 64 + index)
        profile_keys.append(key)
        catalog_rows.append([key[0], key[1], [10.0 + index, 1000.0, 16.0, 0.01]])
    assignment_rows = []
    for request_id in range(NUM_NPU * BACKING_REQUESTS_PER_NPU):
        npu = request_id % NUM_NPU
        sequence = request_id // NUM_NPU
        category = tuple(sorted(CATEGORIES))[request_id % 4]
        profile = profile_keys[request_id % len(profile_keys)]
        assignment_rows.append([request_id, npu, sequence, category, list(profile)])
    global_hashes = {
        field: _fake_sha(field)
        for field in ("catalog", "recipe", "schedule", "assignment")
    }
    schedule_metadata = {
        **global_hashes,
        "mode": "iid_uniform_profile_catalog_v1",
        "seed": SEED,
        "num_npu": NUM_NPU,
        "requests_per_npu": BACKING_REQUESTS_PER_NPU,
        "request_id_formula": "sequence * num_npu + npu_id",
        "catalog_rows": catalog_rows,
        "assignment_rows": assignment_rows,
    }
    source_manifest = {"dummy.py": _fake_sha("dummy"), "data": _fake_sha("data")}
    source_fingerprint = _canonical_hash(
        source_manifest, b"ms-scale-control-source:v1\0"
    )
    campaign = _fake_sha("campaign")
    cases = [case.case_dict() for case in CASES]
    spec = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "definition": DEFINITION,
        "definition_fingerprint": _expected_definition_fingerprint(),
        "num_npu": NUM_NPU,
        "scale_semantics": {
            "num_npu": NUM_NPU,
            "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
            "total_assignment_count": NUM_NPU * BACKING_REQUESTS_PER_NPU,
        },
        "campaign_spec_sha256": campaign,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "default_ssu_list": list(SSUS),
        "cases": cases,
        "report_roles": {},
        "workload": {
            "mode": "iid_uniform_profile_catalog_v1",
            "seed": SEED,
            "requests_per_npu": BACKING_REQUESTS_PER_NPU,
            **global_hashes,
            "prefix_32_assignment_hash": _fake_sha("prefix"),
            "full_assignment_hash": global_hashes["assignment"],
            "sampling": "IID uniform with replacement over all 84 data profiles",
            "per_npu_streams": "independent and prefix-stable",
            "scientific_prefix_requests_per_npu": 32,
        },
        "steady_state": {
            "seed": SEED,
            "requests_per_npu": BACKING_REQUESTS_PER_NPU,
            "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
            "settle_ms": SETTLE_MS,
            "measurement_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "slo_alpha": PRIMARY_ALPHA,
            "calibration_mode": False,
        },
        "adaptive": _expected_definition()["adaptive"],
        "adaptive_case_profiles": {
            case.name: case.adaptive_profile()
            for case in CASES
            if case.kind == "adaptive"
        },
        "thread_limit_environment": EXPECTED_THREAD_LIMITS,
        "runtime_identity": SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash reused across all 16 layers",
        "source_files": list(source_manifest),
    }
    config_fingerprint = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
    input_authentication = {
        "source": "data",
        "source_sha256": source_manifest["data"],
        "catalog_hash": global_hashes["catalog"],
        "table_fingerprint": _fake_sha("table"),
        "profile_count": 84,
    }
    path_abi = {
        "path_count": 256,
        "group_count": 8,
        "paths_per_group": 32,
        "max_npu": 128,
        "assigned_count": 128,
        "assigned_unique": 128,
        "assigned_min": 16,
        "assigned_max": 255,
        "path_zero_reserved": True,
        "assigned_paths_sha256": _fake_sha("paths"),
    }
    runtime = {
        "python_implementation": SELECTED128_EXPECTED_RUNTIME_IDENTITY[
            "python_implementation"
        ],
        "python": SELECTED128_EXPECTED_RUNTIME_IDENTITY["python_version"],
        "numpy": SELECTED128_EXPECTED_RUNTIME_IDENTITY["numpy_version"],
        "numpy_blas_identity": {
            "name": SELECTED128_EXPECTED_RUNTIME_IDENTITY["blas_name"],
            "version": SELECTED128_EXPECTED_RUNTIME_IDENTITY["blas_version"],
            "openblas_configuration": SELECTED128_EXPECTED_RUNTIME_IDENTITY[
                "openblas_configuration"
            ],
        },
        "multiprocessing_start_method": "spawn",
        "thread_limit_environment": EXPECTED_THREAD_LIMITS,
    }
    campaign_authentication = {"sha256": campaign, "size_bytes": 100}
    documents = []
    for num_ssu in SSUS:
        simulator_input = _fake_sha(f"simulator-{num_ssu}")
        workload_input = _fake_sha(f"workload-{num_ssu}")
        placement_input = _fake_sha(f"placement-{num_ssu}")
        trace_input = _fake_sha(f"trace-{num_ssu}")
        results = []
        for case in CASES:
            request_rows = []
            for npu in range(NUM_NPU):
                request_id = 8 * NUM_NPU + npu
                assignment = assignment_rows[request_id]
                profile = assignment[4]
                ideal = 16.0
                ttft = ideal * (1.4 if npu % 2 == 0 else 1.8)
                admission = 1_000.0 + 0.01 * npu
                request_rows.append(
                    {
                        "request_id": request_id,
                        "npu_id": npu,
                        "sequence": 8,
                        "category": assignment[3],
                        "profile_id": f"{profile[0]},{profile[1]}",
                        "profile_key": profile,
                        "profile_name": f"seq_len_k={profile[0]},nql={profile[1]}",
                        "raw_demand_gbps": 10.0 + request_id % 84,
                        "admission_time_ms": admission,
                        "completion_time_ms": admission + ttft,
                        "ttft_ms": ttft,
                        "ideal_ttft_ms": ideal,
                        "slo_met": True,
                    }
                )
            cohort_material = [
                [row["request_id"], row["npu_id"], row["sequence"], row["profile_key"]]
                for row in request_rows
            ]
            cohort_hash = _canonical_hash(
                cohort_material, b"ms-scale-control-measurement-cohort:v1\0"
            )
            if case.kind == "adaptive":
                pressure = 0
                evaluations = 10
                writes = 100
                control_interval = case.min_interval_ms
            elif case.kind == "layer_once":
                pressure = 100
                evaluations = 0
                writes = 0
                control_interval = None
            else:
                pressure = evaluations = writes = 0
                control_interval = None
            summary = {
                "schema_version": 2,
                "mode": "steady_state_full_load",
                "num_npu": NUM_NPU,
                "num_ssu": num_ssu,
                "n_layers": N_LAYERS,
                "batch_size": BATCH_SIZE,
                "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
                "settle_ms": SETTLE_MS,
                "slo_alpha": PRIMARY_ALPHA,
                "measurement_duration_ms": MEASUREMENT_MS,
                "pressure_ttl_ms": case.pressure_ttl_ms,
                "cir_write_threshold_gbps": 0.0,
                "control_min_interval_ms": control_interval,
                "invariants": {name: True for name in EXPECTED_INVARIANTS},
                "warmup_reached_ms": 500.0,
                "measurement_start_ms": 1_000.0,
                "measurement_end_ms": 9_000.0,
                "drain_stop_ms": 9_100.0,
                "tail_drain_ms": 100.0,
                "request_rows": request_rows,
                "request_counts_by_npu": [1] * NUM_NPU,
                "measurement_request_count": NUM_NPU,
                "ttft_slo_attainment": 1.0,
                "request_weighted_slo_attainment": 1.0,
                "npu_utilizations": [0.5] * NUM_NPU,
                "compute_ms_by_npu": [4_000.0] * NUM_NPU,
                "mean_npu_utilization": 0.5,
                "measurement_pressure_reports": pressure,
                "measurement_control_evaluations": evaluations,
                "measurement_cir_path_writes": writes,
                "input_fingerprint": simulator_input,
            }
            if case.kind == "adaptive":
                profile = case.adaptive_profile()
                summary["adaptive_controller_profile"] = {
                    key: value
                    for key, value in profile.items()
                    if key != "tuning_slo_alpha"
                }
            case_dict = case.case_dict()
            case_fingerprint = _canonical_hash(
                {
                    "case": case_dict,
                    "num_ssu": num_ssu,
                    "source_fingerprint": source_fingerprint,
                    "config_fingerprint": config_fingerprint,
                },
                b"ms-scale-control-case:v1\0",
            )
            results.append(
                {
                    "status": "ok",
                    "case": case.name,
                    "family": case.family,
                    "kind": case.kind,
                    "role": None,
                    "num_ssu": num_ssu,
                    "num_npu": NUM_NPU,
                    "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
                    "definition": DEFINITION,
                    "definition_fingerprint": _expected_definition_fingerprint(),
                    "case_spec": case_dict,
                    "source_fingerprint": source_fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "campaign_spec_sha256": campaign,
                    "case_fingerprint": case_fingerprint,
                    "input_fingerprints": {
                        **global_hashes,
                        "prefix_32_assignment": _fake_sha("prefix"),
                        "full_assignment": global_hashes["assignment"],
                        "workload": workload_input,
                        "placement": placement_input,
                        "trace": trace_input,
                        "simulator": simulator_input,
                    },
                    "workload_statistics": {"num_ssu": num_ssu},
                    "prefix_32_workload_statistics": {"num_ssu": num_ssu},
                    "prefix_32_materialized_fingerprints": {
                        "workload": _fake_sha(f"prefix-workload-{num_ssu}"),
                        "placement": _fake_sha(f"prefix-placement-{num_ssu}"),
                        "trace": _fake_sha(f"prefix-trace-{num_ssu}"),
                    },
                    "cohort_profile_metrics": {},
                    "measurement_cohort_fingerprint": cohort_hash,
                    "stationarity_diagnostics": {},
                    "runtime": runtime,
                    "wall_time_s": 1.0,
                    "steady_summary": summary,
                }
            )
        payload = {
            "schema_version": 3,
            "complete": False,
            "selected_complete": True,
            "source_stable_during_run": True,
            "config_stable_during_run": True,
            "campaign_spec_stable_during_run": True,
            "source_fingerprint": source_fingerprint,
            "ending_source_fingerprint": source_fingerprint,
            "source_manifest": source_manifest,
            "definition": DEFINITION,
            "definition_fingerprint": _expected_definition_fingerprint(),
            "num_npu": NUM_NPU,
            "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
            "total_assignment_count": NUM_NPU * BACKING_REQUESTS_PER_NPU,
            "path_abi": path_abi,
            "input_authentication": input_authentication,
            "runtime": runtime,
            "execution": {
                "multiprocessing_start_method": "spawn",
                "requested_max_workers": 1,
                "single_ssu_process_pool_required": True,
            },
            "config_fingerprint": config_fingerprint,
            "ending_config_fingerprint": config_fingerprint,
            "campaign_spec_sha256": campaign,
            "campaign_spec_authentication": campaign_authentication,
            "ending_campaign_spec_authentication": campaign_authentication,
            "experiment_spec": spec,
            "selected_ssus": [num_ssu],
            "selected_cases": sorted(CASE_NAMES),
            "selected_keys": [[name, num_ssu] for name in sorted(CASE_NAMES)],
            "schedule_metadata": schedule_metadata,
            "pairing_audit": {
                str(num_ssu): {
                    "cases": sorted(CASE_NAMES),
                    "has_rows": True,
                    "all_available_rows_paired": True,
                }
            },
            "results": results,
        }
        path = PROJECT_ROOT / "results" / "synthetic" / f"ssu{num_ssu}.json"
        documents.append(ShardDocument(path, payload, _fake_sha(str(num_ssu)), 1))
    campaign_document = CampaignDocument(
        path=PROJECT_ROOT / "campaigns" / "synthetic_selected128.json",
        sha256=campaign,
        size_bytes=campaign_authentication["size_bytes"],
    )
    return documents, campaign_document


def _self_test() -> dict[str, object]:
    documents, campaign = _synthetic_documents()
    rows, provenance = _validate_shards(documents, campaign)
    _require(len(rows) == 70, "self-test did not produce 70 rows")
    _assert_portable_manifest(provenance)
    _csv_text(rows)
    _require(
        {row["alpha_1p5_equal_npu_slo_pct"] for row in rows} == {50.0},
        "self-test alpha1.5 metric differs",
    )
    _require(
        {row["alpha_2_equal_npu_slo_pct"] for row in rows} == {100.0},
        "self-test alpha2 metric differs",
    )
    first = documents[0]
    bad_payload = dict(first.payload)
    bad_results = list(first.payload["results"])
    bad_row = dict(bad_results[0])
    bad_summary = dict(bad_row["steady_summary"])
    bad_invariants = dict(bad_summary["invariants"])
    bad_invariants["no_backlog_exhaustion"] = False
    bad_summary["invariants"] = bad_invariants
    bad_row["steady_summary"] = bad_summary
    bad_results[0] = bad_row
    bad_payload["results"] = bad_results
    bad_documents = [
        ShardDocument(first.path, bad_payload, first.sha256, first.size_bytes),
        *documents[1:],
    ]
    rejected = False
    try:
        _validate_shards(bad_documents, campaign)
    except SummaryError:
        rejected = True
    _require(rejected, "self-test failed to reject a false invariant")
    return {
        "self_test": "passed",
        "valid_rows": len(rows),
        "ssu_count": len(SSUS),
        "case_count": len(CASES),
        "canonical_formal_validator": True,
        "false_invariant_rejected": True,
        "portable_provenance_checked": bool(provenance["shards"]),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-spec",
        type=Path,
        default=DEFAULT_CAMPAIGN_SPEC,
        help=(
            "frozen campaign JSON whose raw SHA-256 is independently matched "
            "against every shard"
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="explicit shard path; repeat exactly seven times instead of --raw-dir discovery",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        _require(not args.shard, "--self-test does not accept --shard")
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return 0
    campaign = _read_campaign_spec(args.campaign_spec)
    paths = _discover_shards(args.raw_dir, tuple(args.shard or ()))
    shards = [_read_shard(path) for path in paths]
    summary_path, manifest_path, rows = _build_outputs(
        shards, campaign, args.output_dir
    )
    print(
        json.dumps(
            {
                "complete": True,
                "row_count": len(rows),
                "summary_csv": _portable_path(summary_path),
                "manifest": _portable_path(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SummaryError as error:
        raise SystemExit(f"ERROR: {error}") from error
