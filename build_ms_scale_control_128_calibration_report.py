"""Audit selected128 alpha-1.5 calibration v2 and apply its frozen rule.

The report mechanically selects an evidence target under the preregistered
rule, but never reads or writes a formal campaign setting.  The dependency
direction is from this analyzer to the experiment runner, so this file cannot
expand the runner's source closure.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
from statistics import fmean, median
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from ms_scale_control_experiment import (
    AdaptiveCase,
    SELECTED128_EXPECTED_RUNTIME_IDENTITY,
    THREAD_LIMIT_ENVIRONMENT,
    _definition_fingerprint,
    _selected128_definition,
    runtime_merge_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(
    "results/ms_scale_control/selected128_alpha1p5_calibration_v2_raw"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/ms_scale_control/selected128_alpha1p5_calibration_v2_analysis"
)
DEFAULT_RULE_PATH = Path("campaigns/selected128_alpha1p5_calibration_rule_v2.json")
EXPERIMENT = "128npu_selected_cir_control_alpha_tuned_v1"
DEFINITION = "selected128"
CASE_NAME = "adaptive_a1p5_t0_i25ms"
TARGETS = (0.68, 0.6866666666666666, 0.70)
TARGET_TOKEN = {
    0.68: "0680",
    0.6866666666666666: "0686667",
    0.70: "0700",
}
HISTORICAL_DEFINITION_FINGERPRINTS = {
    0.68: "b3a74eb21c0ea953b06fd95e6c61bd1233b19714f8a0f523edf422a4604ae191",
    0.6866666666666666: "d2535e6b4860ed6e1a574236ae3caa85476d6ee9f35350dd9668ed2cd0525266",
    0.70: "3aa89bc0eb82c553951ee056e84f68fa050f1cdec030425b20c82f6a8fffa497",
}
HISTORICAL_SOURCE_FINGERPRINT = (
    "7fc63b4110c9a7161be79945e03fa06c037f883ead61379b96cb51c7cc3ec900"
)
HISTORICAL_SOURCE_COMMIT = "426a3d843a66cface373b291c4e1ecc7f55774d5"
HISTORICAL_SOURCE_MANIFEST = {
    "adaptive_admission_scheme_b_v2_1.py": "7b336d145d898e2bcb646a560529b04da01231afa31a6f5136dda199727a18fb",
    "authenticated_workload_inputs.py": "d3b2f9917108916164006b70a6f7f2687911484f3231da1cd3328211b86ac95f",
    "continuous_batch_control.py": "fe7ee3933ba37362cf7b67e1c39c9f4720bd682a66bd008b57c06aa72076591f",
    "continuous_batch_sim.py": "1aaa11b3d2b4cf3e8ad85786e41a7288f511ab355271cc2ce864e9a8e953a546",
    "continuous_prefill_client.py": "04a8c4dc2d6430ff803b43127007c6fcb58d1279b4c260468425c3899ed0c9da",
    "continuous_prefill_workload.py": "65c901ed1d63df88b4ccf35d309e586d563e470a3874ae0d1da2b221f2b5b5de",
    "data": "fd197b79865b4c1f42d400100c5e05349ca1ba5f2d42b904af8a1759aabeb04b",
    "forecast_hotspot_policy.py": "60504451631cadfbca01be4ff636d8f08e3c3e6c20027cc801216dabb2f83139",
    "ms_scale_control_experiment.py": "85b5719c655e34b7e041da6980137afbb205f62e0d5e07b59283ccfa0d5f1b54",
    "policy_logic.py": "fb72249119635558a8db6efbc12e9dbc33e4a065327c558b15e49bb996f5d229",
    "protected_floor_scheme_b.py": "517fb28e589a25117483b45ec17d37137768489e44d9af17ee74ffeb1af481ed",
    "random_steady_state_workload.py": "70284f3ede4097453338e396ec0a0d252909be6999ec158d854d36b7df015607",
    "scheme_b_prefill.py": "4cb650e8d8e9a0b08c8928fee3a9e25ea1c6204dd825a77b14a57764912ea740",
    "sim.py": "4e94f0ca96786621b75bfe46a16183ab9696706b8ecbb3391bfd0813b67a5ff7",
    "six_request_workload.py": "f5d3654d5ce3aa86858d4b6d8695054354d98b1ed84863dc63e05d3ab3fb2e26",
    "slo_admission_scheme_b.py": "64774141a2d0c576a1643703d923baf23193327f2ed2153516e1f37bc8a174d8",
    "slo_admission_scheme_b_v2.py": "48b9801f706f310d7592040ff6c077495a8ae56d16fd3047e4de5a240f9d6195",
    "strategy_profiles.py": "d72d471d08fb0d28ab2412cc24fa69b52fd15677c859fd43bafbcf66bdddc2fb",
}
FORMAL_TARGET = 1.0 / 1.5 + 0.02
SSUS = (16, 20, 24)
NUM_NPU = 128
N_LAYERS = 16
BATCH_SIZE = 1
SEED = 43
BACKING_REQUESTS_PER_NPU = 128
WARMUP_REQUESTS_PER_NPU = 8
SETTLE_MS = 500.0
BLOCK_MS = 500.0
PRIMARY_ALPHA = 2.0
SENSITIVITY_ALPHA = 1.5
SLO_EPSILON = 1e-12
FRACTION_TOLERANCE = 1e-12
EXPECTED_THREADS = {name: "1" for name in THREAD_LIMIT_ENVIRONMENT}
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

CSV_FIELDS = (
    "target_ratio",
    "is_current_formal_target",
    "num_ssu",
    "case",
    "seed",
    "measurement_ms",
    "mean_npu_utilization_pct",
    "alpha_1p5_equal_npu_slo_pct",
    "alpha_1p5_request_weighted_slo_pct",
    "alpha_2_equal_npu_slo_pct",
    "alpha_2_request_weighted_slo_pct",
    "measurement_request_count",
    "measurement_requests_per_npu_min",
    "measurement_requests_per_npu_median",
    "measurement_requests_per_npu_max",
    "measurement_control_evaluations",
    "measurement_cir_path_writes",
    "measurement_cir_write_transactions",
    "measurement_cir_commits",
    "measurement_cir_path_write_rate_hz",
    "delta_alpha_1p5_attainment_vs_default",
    "delta_utilization_vs_default",
    "qualified_challenger",
    "selected_by_preregistered_rule",
    "measurement_cohort_fingerprint",
    "input_simulator_fingerprint",
    "source_fingerprint",
    "config_fingerprint",
    "definition_fingerprint",
    "all_invariants_passed",
    "source_artifact",
)


class CalibrationReportError(ValueError):
    """Raised when a shard cannot support the calibration evidence report."""


@dataclass(frozen=True)
class ShardDocument:
    path: Path
    payload: dict[str, object]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RuleDocument:
    path: Path
    document: dict[str, object]
    sha256: str
    size_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationReportError(message)


def _integer(value: object, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise CalibrationReportError(f"{context}: bool is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = int(value)
    else:
        raise CalibrationReportError(f"{context}: expected an integer")
    if minimum is not None:
        _require(result >= minimum, f"{context}: must be at least {minimum}")
    return result


def _finite(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationReportError(f"{context}: expected a number") from error
    _require(math.isfinite(result), f"{context}: expected a finite number")
    return result


def _close(
    value: object,
    expected: float,
    context: str,
    *,
    tolerance: float = 1e-10,
) -> float:
    result = _finite(value, context)
    _require(
        math.isclose(result, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{context}: {result!r} != {expected!r}",
    )
    return result


def _fraction(value: object, context: str) -> float:
    result = _finite(value, context)
    _require(
        -FRACTION_TOLERANCE <= result <= 1.0 + FRACTION_TOLERANCE,
        f"{context}: outside [0, 1] beyond floating tolerance",
    )
    return min(1.0, max(0.0, result))


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
        raise CalibrationReportError(
            f"path is outside the project and cannot be recorded portably: {path}"
        ) from error


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationReportError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise CalibrationReportError(f"non-finite JSON constant: {value}")


def _read_shard(path: Path) -> ShardDocument:
    resolved = path.expanduser().resolve()
    _portable_path(resolved)
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise CalibrationReportError(
            f"cannot read shard {resolved}: {error}"
        ) from error
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationReportError(
            f"invalid UTF-8 JSON shard {resolved}: {error}"
        ) from error
    _require(isinstance(payload, dict), f"{resolved}: top level is not an object")
    return ShardDocument(resolved, payload, sha256(raw).hexdigest(), len(raw))


def _require_exact_keys(value: object, expected: set[str], context: str) -> None:
    _require(isinstance(value, dict), f"{context}: expected an object")
    _require(
        set(value) == expected,
        f"{context}: keys differ; expected {sorted(expected)}, got {sorted(value)}",
    )


def _read_rule(path: Path) -> RuleDocument:
    resolved = path.expanduser().resolve()
    _portable_path(resolved)
    try:
        raw = resolved.read_bytes()
    except OSError as error:
        raise CalibrationReportError(
            f"cannot read calibration rule {resolved}: {error}"
        ) from error
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationReportError(
            f"calibration rule must be duplicate-free UTF-8 JSON: {resolved}: {error}"
        ) from error
    _require_exact_keys(
        document,
        {
            "schema_version",
            "rule",
            "preregistered_at",
            "formal_source_commit",
            "supersedes",
            "supersession_reason",
            "scope",
            "matrix",
            "validity_gate",
            "metrics",
            "challenger_qualification_against_default",
            "selection",
        },
        "calibration rule",
    )
    _require(document["schema_version"] == 1, "calibration rule schema is not 1")
    _require(
        document["rule"] == "selected128_alpha1p5_calibration_rule_v2",
        "unexpected calibration rule name",
    )
    for field in ("preregistered_at", "supersession_reason", "scope"):
        _require(
            isinstance(document[field], str) and document[field].strip(),
            f"calibration rule {field} is empty",
        )
    _require(
        isinstance(document["formal_source_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", document["formal_source_commit"]) is not None,
        "calibration rule formal_source_commit is malformed",
    )
    _require(
        document["supersedes"] == "selected128_alpha1p5_calibration_rule_v1",
        "calibration rule does not supersede v1",
    )

    matrix = document["matrix"]
    _require_exact_keys(
        matrix,
        {
            "seed",
            "measurement_ms",
            "case",
            "interval_ms",
            "target_ratios",
            "ssu_counts",
            "default_target_ratio",
        },
        "calibration rule/matrix",
    )
    _require(
        matrix
        == {
            "seed": SEED,
            "measurement_ms": 4_000.0,
            "case": CASE_NAME,
            "interval_ms": 25.0,
            "target_ratios": list(TARGETS),
            "ssu_counts": list(SSUS),
            "default_target_ratio": FORMAL_TARGET,
        },
        "calibration rule matrix differs from the frozen v2 grid",
    )

    validity = document["validity_gate"]
    _require_exact_keys(
        validity,
        {
            "require_all_nine_cells",
            "require_exact_matrix_configuration",
            "require_status_ok",
            "require_all_simulator_invariants_true",
            "require_positive_measurement_requests_for_all_128_npus",
            "require_same_ten_input_fingerprints_across_targets_within_each_ssu",
            "failure_decision",
        },
        "calibration rule/validity_gate",
    )
    _require(
        all(
            validity[field] is True for field in validity if field != "failure_decision"
        )
        and validity["failure_decision"] == "NO_DECISION",
        "calibration validity gate differs from preregistered v2",
    )

    metrics = document["metrics"]
    _require_exact_keys(
        metrics,
        {
            "alpha1p5_equal_npu_attainment",
            "utilization",
            "comparisons_use_unrounded_values",
        },
        "calibration rule/metrics",
    )
    _require(
        isinstance(metrics["alpha1p5_equal_npu_attainment"], str)
        and metrics["alpha1p5_equal_npu_attainment"].strip()
        and metrics["utilization"] == "mean_npu_utilization"
        and metrics["comparisons_use_unrounded_values"] is True,
        "calibration metric contract differs from preregistered v2",
    )

    qualification = document["challenger_qualification_against_default"]
    expected_qualification = {
        "minimum_delta_alpha1p5_attainment_at_every_ssu": -0.005,
        "minimum_ssu_points_with_delta_attainment_at_least_0p005": 2,
        "minimum_mean_delta_alpha1p5_attainment": 0.005,
        "minimum_mean_delta_utilization": -0.005,
    }
    _require(
        qualification == expected_qualification,
        "challenger qualification thresholds differ from preregistered v2",
    )

    selection = document["selection"]
    _require(
        selection
        == {
            "no_qualified_challenger": "default_target_ratio",
            "one_qualified_challenger": "qualified_challenger",
            "two_qualified_challengers_tiebreakers": [
                "larger minimum per-SSU delta alpha1p5 attainment",
                "if that difference is below 0.005, closer target ratio to the default",
                "higher mean utilization",
                "lower mean CIR path writes per second",
                "default target ratio",
            ],
        },
        "selection procedure differs from preregistered v2",
    )
    return RuleDocument(resolved, document, sha256(raw).hexdigest(), len(raw))


def _definition(target: float):
    return _selected128_definition(alpha1p5_target_ratio=target)


def _case(target: float) -> AdaptiveCase:
    case = _definition(target).case_by_name[CASE_NAME]
    _require(isinstance(case, AdaptiveCase), "calibration case is not AdaptiveCase")
    return case


def _actual_profile(case: AdaptiveCase) -> dict[str, object]:
    return asdict(case.controller_definition())


def _expected_filename(target: float, num_ssu: int) -> str:
    return f"cal_t{TARGET_TOKEN[target]}_ssu{num_ssu}.json"


def _target_from_payload(payload: Mapping[str, object], context: str) -> float:
    rows = payload.get("results")
    _require(
        isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict),
        f"{context}: expected exactly one result row",
    )
    case_spec = rows[0].get("case_spec")
    _require(isinstance(case_spec, dict), f"{context}: case_spec missing")
    target = _finite(case_spec.get("target_ratio"), f"{context}/target ratio")
    _require(target in TARGETS, f"{context}: unexpected target ratio {target!r}")
    return target


def _validate_runtime(value: object, context: str) -> None:
    _require(isinstance(value, dict), f"{context}: runtime missing")
    _require(
        runtime_merge_identity(value) == SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        f"{context}: runtime identity differs from the frozen selected128 runtime",
    )
    _require(
        value.get("thread_limit_environment") == EXPECTED_THREADS,
        f"{context}: every BLAS/OpenMP thread limit must equal the string '1'",
    )
    _require(
        value.get("multiprocessing_start_method") == "spawn",
        f"{context}: multiprocessing start method is not spawn",
    )


def _validate_source(
    payload: Mapping[str, object],
    expected_manifest: Mapping[str, str],
    expected_fingerprint: str,
    context: str,
) -> None:
    manifest = payload.get("source_manifest")
    _require(
        manifest == expected_manifest,
        f"{context}: source manifest differs from frozen calibration source",
    )
    recomputed = _canonical_hash(manifest, b"ms-scale-control-source:v1\0")
    _require(
        recomputed == expected_fingerprint, f"{context}: source manifest hash mismatch"
    )
    _require(
        payload.get("source_fingerprint") == expected_fingerprint
        and payload.get("ending_source_fingerprint") == expected_fingerprint
        and payload.get("source_stable_during_run") is True,
        f"{context}: source fingerprint was not stable",
    )


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


def _validate_input_authentication(value: object, context: str) -> dict[str, object]:
    _require(isinstance(value, dict), f"{context}: input authentication missing")
    _require(value.get("source") == "data", f"{context}: workload source is not data")
    _require(value.get("profile_count") == 84, f"{context}: profile count is not 84")
    for field in ("source_sha256", "catalog_hash", "table_fingerprint"):
        _sha256_value(value.get(field), f"{context}/{field}")
    return value


def _validate_spec(
    payload: Mapping[str, object],
    target: float,
    expected_manifest: Mapping[str, str],
    input_authentication: Mapping[str, object],
    context: str,
) -> tuple[dict[str, object], float]:
    spec = payload.get("experiment_spec")
    _require(isinstance(spec, dict), f"{context}: experiment_spec missing")
    definition = _definition(target)
    expected_definition_fingerprint = _definition_fingerprint(definition)
    expected_cases = [asdict(case) for case in definition.cases]
    expected_profiles = {
        case.name: {
            "tuning_slo_alpha": case.tuning_slo_alpha,
            **asdict(case.controller_definition()),
        }
        for case in definition.cases
        if isinstance(case, AdaptiveCase)
    }
    _require(spec.get("schema_version") == 3, f"{context}: spec schema is not 3")
    _require(spec.get("experiment") == EXPERIMENT, f"{context}: wrong experiment")
    _require(spec.get("definition") == DEFINITION, f"{context}: wrong definition")
    _require(spec.get("num_npu") == NUM_NPU, f"{context}: expected 128 NPUs")
    _require(spec.get("n_layers") == N_LAYERS, f"{context}: wrong layer count")
    _require(spec.get("batch_size") == BATCH_SIZE, f"{context}: wrong batch size")
    _require(
        spec.get("default_ssu_list") == list(definition.default_ssus),
        f"{context}: selected128 SSU definition differs",
    )
    _require(
        spec.get("cases") == expected_cases, f"{context}: target-specific cases differ"
    )
    _require(
        spec.get("adaptive_case_profiles") == expected_profiles,
        f"{context}: target-specific Adaptive profiles differ",
    )
    _require(spec.get("report_roles") == {}, f"{context}: unexpected report roles")
    _require(
        spec.get("definition_fingerprint") == expected_definition_fingerprint,
        f"{context}: definition fingerprint mismatch",
    )
    _require(
        spec.get("campaign_spec_sha256") is None,
        f"{context}: calibration unexpectedly binds a campaign",
    )
    _require(
        spec.get("thread_limit_environment") == EXPECTED_THREADS,
        f"{context}: spec thread limits differ",
    )
    _require(
        spec.get("runtime_identity") == SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        f"{context}: spec runtime identity differs",
    )

    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), f"{context}: steady_state missing")
    expected_steady_keys = {
        "seed",
        "requests_per_npu",
        "warmup_requests_per_npu",
        "settle_ms",
        "measurement_ms",
        "block_ms",
        "slo_alpha",
        "calibration_mode",
    }
    _require(
        set(steady) == expected_steady_keys, f"{context}: steady-state keys differ"
    )
    _require(steady.get("seed") == SEED, f"{context}: calibration seed is not 43")
    _require(
        steady.get("requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: backing is not 128 requests/NPU",
    )
    _require(
        steady.get("warmup_requests_per_npu") == WARMUP_REQUESTS_PER_NPU,
        f"{context}: warmup is not 8 requests/NPU",
    )
    _close(steady.get("settle_ms"), SETTLE_MS, f"{context}/settle")
    measurement_ms = _finite(steady.get("measurement_ms"), f"{context}/measurement")
    _close(measurement_ms, 4_000.0, f"{context}/v2 measurement")
    _close(steady.get("block_ms"), BLOCK_MS, f"{context}/block")
    _close(steady.get("slo_alpha"), PRIMARY_ALPHA, f"{context}/primary alpha")
    _require(
        steady.get("calibration_mode") is True,
        f"{context}: calibration_mode is not true",
    )

    scale = spec.get("scale_semantics")
    _require(isinstance(scale, dict), f"{context}: scale_semantics missing")
    _require(scale.get("num_npu") == NUM_NPU, f"{context}: scale NPU mismatch")
    _require(
        scale.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: scale backing mismatch",
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
    for field in ("prefix_32_assignment_hash", "full_assignment_hash"):
        _sha256_value(workload.get(field), f"{context}/workload/{field}")
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
        f"{context}: scientific prefix length differs",
    )
    _require(
        workload.get("authentication") == input_authentication,
        f"{context}: workload authentication differs from payload",
    )
    _require(
        spec.get("cross_request_layer0_prefetch") is True,
        f"{context}: cross-request Layer-0 prefetch is not enabled",
    )
    _require(
        spec.get("placement") == "token-block ring hash reused across all 16 layers",
        f"{context}: placement contract differs",
    )
    _require(
        isinstance(spec.get("source_files"), list)
        and set(spec["source_files"]) == set(expected_manifest),
        f"{context}: source_files differ from authenticated source manifest",
    )
    adaptive = spec.get("adaptive")
    _require(isinstance(adaptive, dict), f"{context}: base Adaptive definition missing")
    expected_base = asdict(definition.adaptive)
    for field, expected in expected_base.items():
        _require(
            adaptive.get(field) == expected, f"{context}: base Adaptive {field} differs"
        )

    expected_config = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
    _require(
        payload.get("config_fingerprint") == expected_config
        and payload.get("ending_config_fingerprint") == expected_config
        and payload.get("config_stable_during_run") is True,
        f"{context}: config fingerprint was not stable or does not authenticate spec",
    )
    return spec, measurement_ms


def _catalog_and_schedule(
    payload: Mapping[str, object],
    spec: Mapping[str, object],
    context: str,
) -> tuple[dict[tuple[int, int], tuple[float, ...]], dict[int, tuple]]:
    metadata = payload.get("schedule_metadata")
    _require(isinstance(metadata, dict), f"{context}: schedule metadata missing")
    _require(
        metadata.get("mode") == "iid_uniform_profile_catalog_v1",
        f"{context}: schedule mode differs",
    )
    _require(metadata.get("seed") == SEED, f"{context}: schedule seed is not 43")
    _require(
        metadata.get("num_npu") == NUM_NPU, f"{context}: schedule NPU count differs"
    )
    _require(
        metadata.get("requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: schedule backing differs",
    )
    _require(
        metadata.get("request_id_formula") == "sequence * num_npu + npu_id",
        f"{context}: request ID formula differs",
    )
    workload = spec["workload"]
    for field in ("catalog", "recipe", "schedule", "assignment"):
        _require(
            metadata.get(field) == workload.get(field),
            f"{context}: schedule {field} differs from spec",
        )

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
            f"{context}: nonpositive profile value",
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
        npu = request_id % NUM_NPU
        sequence = request_id // NUM_NPU
        _require(row[1] == npu, f"{context}: assignment NPU formula mismatch")
        _require(row[2] == sequence, f"{context}: assignment sequence formula mismatch")
        _require(row[3] in CATEGORIES, f"{context}: invalid request category")
        _require(
            isinstance(row[4], list) and len(row[4]) == 2,
            f"{context}: malformed profile key",
        )
        profile = (
            _integer(row[4][0], f"{context}/profile key"),
            _integer(row[4][1], f"{context}/profile key"),
        )
        _require(
            profile in catalog, f"{context}: assignment uses unknown profile {profile}"
        )
        assignments[request_id] = (npu, sequence, row[3], profile)
    return catalog, assignments


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
    case: AdaptiveCase,
    num_ssu: int,
    measurement_ms: float,
    catalog: Mapping[tuple[int, int], tuple[float, ...]],
    assignments: Mapping[int, tuple],
    cohort_fingerprint: str,
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
        measurement_ms,
        f"{context}/measurement duration",
    )
    _close(summary.get("pressure_ttl_ms"), 0.0, f"{context}/pressure TTL")
    _close(
        summary.get("cir_write_threshold_gbps"),
        0.0,
        f"{context}/CIR write threshold",
    )
    _close(
        summary.get("control_min_interval_ms"),
        25.0,
        f"{context}/control interval",
    )
    _require(
        summary.get("adaptive_controller_profile") == _actual_profile(case),
        f"{context}: actual Adaptive controller profile differs from target",
    )

    invariants = summary.get("invariants")
    _require(
        isinstance(invariants, dict) and invariants, f"{context}: invariants missing"
    )
    _require(
        EXPECTED_INVARIANTS <= set(invariants),
        f"{context}: required invariants missing: "
        f"{sorted(EXPECTED_INVARIANTS - set(invariants))}",
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
    _close(end - start, measurement_ms, f"{context}/window length", tolerance=1e-8)
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
            start <= admission < end, f"{context}: admission outside measurement window"
        )
        _require(completion >= admission, f"{context}: completion before admission")
        _require(completion <= drain + 1e-8, f"{context}: completion after drain stop")
        _close(completion - admission, ttft, f"{context}/TTFT relation", tolerance=1e-8)
        _require(ttft + 1e-8 >= ideal, f"{context}: TTFT below compute lower bound")
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
    recomputed_cohort = _canonical_hash(
        cohort_material, b"ms-scale-control-measurement-cohort:v1\0"
    )
    _require(
        recomputed_cohort == cohort_fingerprint,
        f"{context}: measurement cohort fingerprint mismatch",
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
            _finite(busy, f"{context}/compute[{npu}]") / measurement_ms,
            utilization,
            f"{context}/compute utilization[{npu}]",
            tolerance=1e-9,
        )

    pressure = _integer(
        summary.get("measurement_pressure_reports"),
        f"{context}/measurement pressure reports",
        minimum=0,
    )
    pressure_requests = _integer(
        summary.get("measurement_pressure_requests"),
        f"{context}/measurement pressure requests",
        minimum=0,
    )
    cache_hits = _integer(
        summary.get("measurement_pressure_cache_hits"),
        f"{context}/measurement pressure cache hits",
        minimum=0,
    )
    evaluations = _integer(
        summary.get("measurement_control_evaluations"),
        f"{context}/measurement control evaluations",
        minimum=1,
    )
    writes = _integer(
        summary.get("measurement_cir_path_writes"),
        f"{context}/measurement CIR writes",
        minimum=0,
    )
    transactions = _integer(
        summary.get("measurement_cir_write_transactions"),
        f"{context}/measurement CIR transactions",
        minimum=0,
    )
    commits = _integer(
        summary.get("measurement_cir_commits"),
        f"{context}/measurement CIR commits",
        minimum=0,
    )
    _require(
        pressure == 0
        and cache_hits == 0
        and pressure_requests == pressure + cache_hits,
        f"{context}: Adaptive unexpectedly read the SSU pressure table",
    )
    _require(
        commits <= transactions <= writes and commits <= evaluations,
        f"{context}: control-plane counters are inconsistent",
    )
    for field, total in (
        ("measurement_pressure_reports_by_ssu", pressure),
        ("measurement_pressure_cache_hits_by_ssu", cache_hits),
        ("measurement_pressure_requests_by_ssu", pressure_requests),
        ("measurement_cir_path_writes_by_ssu", writes),
        ("measurement_cir_write_transactions_by_ssu", transactions),
    ):
        values = summary.get(field)
        _require(
            isinstance(values, list)
            and len(values) == num_ssu
            and all(type(value) is int and value >= 0 for value in values)
            and sum(values) == total,
            f"{context}: {field} does not reconcile to its total",
        )

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
        "measurement_control_evaluations": evaluations,
        "measurement_cir_path_writes": writes,
        "measurement_cir_write_transactions": transactions,
        "measurement_cir_commits": commits,
        "measurement_cir_path_write_rate_hz": writes / (measurement_ms / 1_000.0),
    }


def _normalized_spec(spec: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(spec)
    normalized["definition_fingerprint"] = "<target-specific>"
    for case in normalized.get("cases", ()):  # validated as dictionaries earlier
        if case.get("family") == "adaptive_alpha_a1p5":
            case["target_ratio"] = "<target-specific>"
    for name, profile in normalized.get("adaptive_case_profiles", {}).items():
        if name.startswith("adaptive_a1p5_"):
            profile["target_ratio"] = "<target-specific>"
    return normalized


def _validate_shard(
    shard: ShardDocument,
    expected_manifest: Mapping[str, str],
    expected_source_fingerprint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = shard.payload
    context = _portable_path(shard.path)
    target = _target_from_payload(payload, context)
    definition = _definition(target)
    case = _case(target)
    expected_definition_fingerprint = _definition_fingerprint(definition)

    _require(payload.get("schema_version") == 3, f"{context}: shard schema is not 3")
    _require(
        payload.get("complete") is False,
        f"{context}: single-cell calibration marked complete",
    )
    _require(
        payload.get("selected_complete") is True,
        f"{context}: selected shard incomplete",
    )
    _require(payload.get("definition") == DEFINITION, f"{context}: wrong definition")
    _require(payload.get("num_npu") == NUM_NPU, f"{context}: expected 128 NPUs")
    _require(
        payload.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: backing is not 128 requests/NPU",
    )
    _require(
        payload.get("total_assignment_count") == NUM_NPU * BACKING_REQUESTS_PER_NPU,
        f"{context}: total assignment count differs",
    )
    _require(
        payload.get("definition_fingerprint") == expected_definition_fingerprint,
        f"{context}: target-specific definition fingerprint differs",
    )
    _validate_source(payload, expected_manifest, expected_source_fingerprint, context)
    _validate_path_abi(payload.get("path_abi"), context)
    input_authentication = _validate_input_authentication(
        payload.get("input_authentication"), context
    )

    _require(
        payload.get("campaign_spec_sha256") is None
        and payload.get("campaign_spec_authentication") is None
        and payload.get("ending_campaign_spec_authentication") is None
        and payload.get("campaign_spec_stable_during_run") is True,
        f"{context}: calibration must not bind the formal campaign",
    )
    _validate_runtime(payload.get("runtime"), f"{context}/parent runtime")
    execution = payload.get("execution")
    _require(isinstance(execution, dict), f"{context}: execution metadata missing")
    _require(
        execution.get("multiprocessing_start_method") == "spawn",
        f"{context}: execution did not use spawn",
    )
    workers = _integer(execution.get("requested_max_workers"), f"{context}/workers")
    _require(1 <= workers <= 3, f"{context}: worker count is outside [1, 3]")
    _require(
        execution.get("single_ssu_process_pool_required") is True,
        f"{context}: single-SSU process-pool contract absent",
    )

    spec, measurement_ms = _validate_spec(
        payload,
        target,
        expected_manifest,
        input_authentication,
        context,
    )
    selected_ssus = payload.get("selected_ssus")
    _require(
        isinstance(selected_ssus, list)
        and len(selected_ssus) == 1
        and type(selected_ssus[0]) is int
        and selected_ssus[0] in SSUS,
        f"{context}: shard must select one of SSU {SSUS}",
    )
    num_ssu = selected_ssus[0]
    _require(
        shard.path.name == _expected_filename(target, num_ssu),
        f"{context}: filename does not match target/SSU evidence cell",
    )
    _require(
        payload.get("selected_cases") == [CASE_NAME],
        f"{context}: calibration must select only {CASE_NAME}",
    )
    _require(
        payload.get("selected_keys") == [[CASE_NAME, num_ssu]],
        f"{context}: selected key differs",
    )
    _require(
        payload.get("pairing_audit")
        == {
            str(num_ssu): {
                "cases": [CASE_NAME],
                "has_rows": True,
                "all_available_rows_paired": True,
            }
        },
        f"{context}: runner pairing audit differs",
    )

    catalog, assignments = _catalog_and_schedule(payload, spec, context)
    rows = payload["results"]
    row = rows[0]
    _require(row.get("status") == "ok", f"{context}: result status is not ok")
    _require(row.get("case") == CASE_NAME, f"{context}: wrong case")
    _require(row.get("family") == case.family, f"{context}: wrong family")
    _require(row.get("kind") == "adaptive", f"{context}: wrong case kind")
    _require(row.get("role") is None, f"{context}: calibration row has a report role")
    _require(row.get("num_ssu") == num_ssu, f"{context}: row SSU mismatch")
    _require(row.get("num_npu") == NUM_NPU, f"{context}: row NPU mismatch")
    _require(row.get("definition") == DEFINITION, f"{context}: row definition mismatch")
    _require(
        row.get("definition_fingerprint") == expected_definition_fingerprint,
        f"{context}: row definition fingerprint mismatch",
    )
    _require(
        row.get("backing_requests_per_npu") == BACKING_REQUESTS_PER_NPU,
        f"{context}: row backing mismatch",
    )
    _require(
        row.get("case_spec") == asdict(case),
        f"{context}: target-specific case spec differs",
    )
    for field in ("source_fingerprint", "config_fingerprint", "campaign_spec_sha256"):
        _require(
            row.get(field) == payload.get(field), f"{context}: row {field} mismatch"
        )
    expected_case_fingerprint = _canonical_hash(
        {
            "case": asdict(case),
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
        f"{context}: cohort profile metrics missing",
    )
    _require(
        isinstance(row.get("stationarity_diagnostics"), dict),
        f"{context}: stationarity diagnostics missing",
    )
    _validate_runtime(row.get("runtime"), f"{context}/worker runtime")
    _require(
        _finite(row.get("wall_time_s"), f"{context}/wall time") >= 0.0,
        f"{context}: negative wall time",
    )
    cohort_fingerprint = _sha256_value(
        row.get("measurement_cohort_fingerprint"), f"{context}/cohort fingerprint"
    )
    summary = row.get("steady_summary")
    _require(
        isinstance(summary, dict)
        and inputs["simulator"] == summary.get("input_fingerprint"),
        f"{context}: simulator input fingerprint mismatch",
    )
    metrics = _validate_summary(
        summary,
        case,
        num_ssu,
        measurement_ms,
        catalog,
        assignments,
        cohort_fingerprint,
        context,
    )

    cell = {
        "target_ratio": target,
        "is_current_formal_target": target == FORMAL_TARGET,
        "num_ssu": num_ssu,
        "case": CASE_NAME,
        "seed": SEED,
        "measurement_ms": measurement_ms,
        **metrics,
        "measurement_cohort_fingerprint": cohort_fingerprint,
        "input_simulator_fingerprint": inputs["simulator"],
        "source_fingerprint": payload["source_fingerprint"],
        "config_fingerprint": payload["config_fingerprint"],
        "definition_fingerprint": expected_definition_fingerprint,
        "all_invariants_passed": True,
        "source_artifact": context,
    }
    evidence = {
        "target_ratio": target,
        "num_ssu": num_ssu,
        "path": context,
        "sha256": shard.sha256,
        "size_bytes": shard.size_bytes,
        "source_fingerprint": payload["source_fingerprint"],
        "config_fingerprint": payload["config_fingerprint"],
        "definition_fingerprint": expected_definition_fingerprint,
        "runtime_identity": SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "normalized_spec_sha256": _canonical_hash(_normalized_spec(spec)),
        "schedule_metadata_sha256": _canonical_hash(payload["schedule_metadata"]),
        "input_authentication_sha256": _canonical_hash(input_authentication),
        "path_abi_sha256": _canonical_hash(payload["path_abi"]),
        "global_inputs": {
            field: inputs[field] for field in GLOBAL_INPUT_FINGERPRINT_FIELDS
        },
        "all_inputs": dict(inputs),
        "workload_statistics_sha256": _canonical_hash(row["workload_statistics"]),
        "prefix_workload_statistics_sha256": _canonical_hash(
            row["prefix_32_workload_statistics"]
        ),
        "prefix_materialized_sha256": _canonical_hash(prefix_materialized),
    }
    return cell, evidence


def _validate_grid(
    shards: Sequence[ShardDocument],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    _require(
        len(shards) == 9, f"expected exactly nine calibration shards, got {len(shards)}"
    )
    expected_manifest = HISTORICAL_SOURCE_MANIFEST
    expected_source_fingerprint = HISTORICAL_SOURCE_FINGERPRINT
    cells: dict[tuple[float, int], dict[str, object]] = {}
    evidence: dict[tuple[float, int], dict[str, object]] = {}
    for shard in shards:
        cell, cell_evidence = _validate_shard(
            shard,
            expected_manifest,
            expected_source_fingerprint,
        )
        key = (cell["target_ratio"], cell["num_ssu"])
        _require(key not in cells, f"duplicate calibration cell {key}")
        cells[key] = cell
        evidence[key] = cell_evidence

    expected_keys = {(target, num_ssu) for target in TARGETS for num_ssu in SSUS}
    _require(
        set(cells) == expected_keys,
        f"calibration grid differs from {sorted(expected_keys)}",
    )
    ordered_cells = [cells[(target, num_ssu)] for target in TARGETS for num_ssu in SSUS]
    ordered_evidence = [
        evidence[(target, num_ssu)] for target in TARGETS for num_ssu in SSUS
    ]

    _require(
        {item["source_fingerprint"] for item in ordered_evidence}
        == {expected_source_fingerprint},
        "calibration shards do not share the frozen checkout source",
    )
    for field in (
        "normalized_spec_sha256",
        "schedule_metadata_sha256",
        "input_authentication_sha256",
        "path_abi_sha256",
    ):
        _require(
            len({item[field] for item in ordered_evidence}) == 1,
            f"cross-shard {field} mismatch",
        )
    _require(
        len({cell["measurement_ms"] for cell in ordered_cells}) == 1,
        "calibration cells use different measurement durations",
    )

    for target in TARGETS:
        target_evidence = [evidence[(target, num_ssu)] for num_ssu in SSUS]
        _require(
            len({item["config_fingerprint"] for item in target_evidence}) == 1,
            f"target {target}: config differs across SSUs",
        )
        _require(
            len({item["definition_fingerprint"] for item in target_evidence}) == 1,
            f"target {target}: definition differs across SSUs",
        )
    _require(
        len({cell["config_fingerprint"] for cell in ordered_cells}) == len(TARGETS),
        "target-specific config fingerprints are not distinct",
    )
    _require(
        len({cell["definition_fingerprint"] for cell in ordered_cells}) == len(TARGETS),
        "target-specific definition fingerprints are not distinct",
    )

    for num_ssu in SSUS:
        ssu_evidence = [evidence[(target, num_ssu)] for target in TARGETS]
        _require(
            len({_canonical_hash(item["all_inputs"]) for item in ssu_evidence}) == 1,
            f"SSU{num_ssu}: simulator inputs differ across target ratios",
        )
        for field in (
            "workload_statistics_sha256",
            "prefix_workload_statistics_sha256",
            "prefix_materialized_sha256",
        ):
            _require(
                len({item[field] for item in ssu_evidence}) == 1,
                f"SSU{num_ssu}: {field} differs across target ratios",
            )

    target_summaries = [_target_summary(target, ordered_cells) for target in TARGETS]
    common = {
        "source_fingerprint": expected_source_fingerprint,
        "source_manifest": expected_manifest,
        "runtime_identity": SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "thread_limit_environment": EXPECTED_THREADS,
        "seed": SEED,
        "measurement_ms": ordered_cells[0]["measurement_ms"],
        "schedule_metadata_sha256": ordered_evidence[0]["schedule_metadata_sha256"],
        "input_authentication_sha256": ordered_evidence[0][
            "input_authentication_sha256"
        ],
        "path_abi_sha256": ordered_evidence[0]["path_abi_sha256"],
        "normalized_spec_sha256": ordered_evidence[0]["normalized_spec_sha256"],
        "target_summaries": target_summaries,
    }
    return ordered_cells, ordered_evidence, common


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    _require(len(values) == len(SSUS), "target aggregate does not contain three SSUs")
    return {
        "mean": fmean(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def _target_summary(
    target: float,
    cells: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected = [cell for cell in cells if cell["target_ratio"] == target]
    _require(
        [cell["num_ssu"] for cell in selected] == list(SSUS),
        f"target {target}: SSU evidence is incomplete",
    )
    metric_fields = (
        "mean_npu_utilization_pct",
        "alpha_1p5_equal_npu_slo_pct",
        "alpha_1p5_request_weighted_slo_pct",
        "alpha_2_equal_npu_slo_pct",
        "alpha_2_request_weighted_slo_pct",
    )
    return {
        "target_ratio": target,
        "is_current_formal_target": target == FORMAL_TARGET,
        "ssu_counts": list(SSUS),
        "cell_count": len(selected),
        "statistics_across_ssus": {
            field: _metric_stats([float(cell[field]) for cell in selected])
            for field in metric_fields
        },
        "config_fingerprint": selected[0]["config_fingerprint"],
        "definition_fingerprint": selected[0]["definition_fingerprint"],
        "all_invariants_passed": all(
            cell["all_invariants_passed"] is True for cell in selected
        ),
    }


def _evaluate_rule(
    cells: Sequence[Mapping[str, object]],
    rule: RuleDocument,
    *,
    gate_failures: Sequence[str] = (),
) -> dict[str, object]:
    validity_checks = {
        "require_all_nine_cells": len(cells) == 9,
        "require_exact_matrix_configuration": (
            {(cell["target_ratio"], cell["num_ssu"]) for cell in cells}
            == {(target, num_ssu) for target in TARGETS for num_ssu in SSUS}
        ),
        "require_status_ok": len(cells) == 9,
        "require_all_simulator_invariants_true": (
            len(cells) == 9
            and all(cell["all_invariants_passed"] is True for cell in cells)
        ),
        "require_positive_measurement_requests_for_all_128_npus": (
            len(cells) == 9
            and all(cell["measurement_requests_per_npu_min"] > 0 for cell in cells)
        ),
        "require_same_ten_input_fingerprints_across_targets_within_each_ssu": (
            len(cells) == 9
        ),
    }
    failures = list(gate_failures) + [
        name for name, passed in validity_checks.items() if not passed
    ]
    gate = {
        "passed": not failures,
        "checks": validity_checks,
        "failures": failures,
        "failure_decision": rule.document["validity_gate"]["failure_decision"],
    }
    if failures:
        return {
            "validity_gate": gate,
            "decision": {
                "outcome": "NO_DECISION",
                "selected_target_ratio": None,
                "selection_branch": "validity_gate_failure",
                "formal_campaign_modified": False,
            },
            "qualified_challengers": [],
            "challenger_evidence": [],
            "deltas": [],
            "tiebreak": {"applied": False, "reason": "validity gate failed"},
            "comparisons_use_unrounded_values": True,
        }

    default_target = float(rule.document["matrix"]["default_target_ratio"])
    by_key = {
        (float(cell["target_ratio"]), int(cell["num_ssu"])): cell for cell in cells
    }
    qualification = rule.document["challenger_qualification_against_default"]
    every_threshold = float(
        qualification["minimum_delta_alpha1p5_attainment_at_every_ssu"]
    )
    point_threshold = 0.005
    required_points = int(
        qualification["minimum_ssu_points_with_delta_attainment_at_least_0p005"]
    )
    mean_slo_threshold = float(qualification["minimum_mean_delta_alpha1p5_attainment"])
    mean_util_threshold = float(qualification["minimum_mean_delta_utilization"])
    challenger_evidence = []
    deltas = []
    for target in TARGETS:
        if target == default_target:
            continue
        target_deltas = []
        for num_ssu in SSUS:
            challenger = by_key[(target, num_ssu)]
            default = by_key[(default_target, num_ssu)]
            delta_slo = (
                float(challenger["alpha_1p5_equal_npu_slo_pct"])
                - float(default["alpha_1p5_equal_npu_slo_pct"])
            ) / 100.0
            delta_util = (
                float(challenger["mean_npu_utilization_pct"])
                - float(default["mean_npu_utilization_pct"])
            ) / 100.0
            delta = {
                "target_ratio": target,
                "default_target_ratio": default_target,
                "num_ssu": num_ssu,
                "delta_alpha1p5_equal_npu_attainment": delta_slo,
                "delta_mean_npu_utilization": delta_util,
                "challenger_cir_path_writes_per_second": challenger[
                    "measurement_cir_path_write_rate_hz"
                ],
                "default_cir_path_writes_per_second": default[
                    "measurement_cir_path_write_rate_hz"
                ],
            }
            target_deltas.append(delta)
            deltas.append(delta)
        slo_deltas = [
            item["delta_alpha1p5_equal_npu_attainment"] for item in target_deltas
        ]
        util_deltas = [item["delta_mean_npu_utilization"] for item in target_deltas]
        target_cells = [by_key[(target, num_ssu)] for num_ssu in SSUS]
        criteria = {
            "minimum_delta_at_every_ssu": {
                "observed": min(slo_deltas),
                "threshold": every_threshold,
                "passed": min(slo_deltas) >= every_threshold,
            },
            "ssu_points_with_delta_at_least_0p005": {
                "observed": sum(delta >= point_threshold for delta in slo_deltas),
                "threshold": required_points,
                "passed": sum(delta >= point_threshold for delta in slo_deltas)
                >= required_points,
            },
            "mean_delta_alpha1p5_attainment": {
                "observed": fmean(slo_deltas),
                "threshold": mean_slo_threshold,
                "passed": fmean(slo_deltas) >= mean_slo_threshold,
            },
            "mean_delta_utilization": {
                "observed": fmean(util_deltas),
                "threshold": mean_util_threshold,
                "passed": fmean(util_deltas) >= mean_util_threshold,
            },
        }
        challenger_evidence.append(
            {
                "target_ratio": target,
                "qualified": all(item["passed"] for item in criteria.values()),
                "criteria": criteria,
                "mean_utilization": fmean(
                    float(cell["mean_npu_utilization_pct"]) / 100.0
                    for cell in target_cells
                ),
                "mean_cir_path_writes_per_second": fmean(
                    float(cell["measurement_cir_path_write_rate_hz"])
                    for cell in target_cells
                ),
            }
        )

    qualified = [item for item in challenger_evidence if item["qualified"]]
    qualified_targets = [item["target_ratio"] for item in qualified]
    tiebreak: dict[str, object] = {"applied": False, "stages": []}
    if not qualified:
        selected = default_target
        branch = "no_qualified_challenger"
    elif len(qualified) == 1:
        selected = qualified_targets[0]
        branch = "one_qualified_challenger"
    else:
        _require(len(qualified) == 2, "unexpected number of qualified challengers")
        branch = "two_qualified_challengers_tiebreakers"
        tiebreak["applied"] = True
        first, second = qualified
        first_min = first["criteria"]["minimum_delta_at_every_ssu"]["observed"]
        second_min = second["criteria"]["minimum_delta_at_every_ssu"]["observed"]
        difference = abs(first_min - second_min)
        tiebreak["stages"].append(
            {
                "stage": "larger minimum per-SSU delta alpha1p5 attainment",
                "values": {
                    str(first["target_ratio"]): first_min,
                    str(second["target_ratio"]): second_min,
                },
                "absolute_difference": difference,
                "minimum_difference_to_resolve": 0.005,
            }
        )
        if difference >= 0.005:
            selected = (
                first["target_ratio"]
                if first_min > second_min
                else second["target_ratio"]
            )
            tiebreak["resolved_by"] = tiebreak["stages"][-1]["stage"]
        else:
            first_distance = abs(first["target_ratio"] - default_target)
            second_distance = abs(second["target_ratio"] - default_target)
            tiebreak["stages"].append(
                {
                    "stage": "closer target ratio to the default",
                    "values": {
                        str(first["target_ratio"]): first_distance,
                        str(second["target_ratio"]): second_distance,
                    },
                }
            )
            if first_distance != second_distance:
                selected = (
                    first["target_ratio"]
                    if first_distance < second_distance
                    else second["target_ratio"]
                )
                tiebreak["resolved_by"] = tiebreak["stages"][-1]["stage"]
            elif first["mean_utilization"] != second["mean_utilization"]:
                tiebreak["stages"].append(
                    {
                        "stage": "higher mean utilization",
                        "values": {
                            str(first["target_ratio"]): first["mean_utilization"],
                            str(second["target_ratio"]): second["mean_utilization"],
                        },
                    }
                )
                selected = (
                    first["target_ratio"]
                    if first["mean_utilization"] > second["mean_utilization"]
                    else second["target_ratio"]
                )
                tiebreak["resolved_by"] = tiebreak["stages"][-1]["stage"]
            elif (
                first["mean_cir_path_writes_per_second"]
                != second["mean_cir_path_writes_per_second"]
            ):
                tiebreak["stages"].append(
                    {
                        "stage": "lower mean CIR path writes per second",
                        "values": {
                            str(first["target_ratio"]): first[
                                "mean_cir_path_writes_per_second"
                            ],
                            str(second["target_ratio"]): second[
                                "mean_cir_path_writes_per_second"
                            ],
                        },
                    }
                )
                selected = (
                    first["target_ratio"]
                    if first["mean_cir_path_writes_per_second"]
                    < second["mean_cir_path_writes_per_second"]
                    else second["target_ratio"]
                )
                tiebreak["resolved_by"] = tiebreak["stages"][-1]["stage"]
            else:
                tiebreak["stages"].append(
                    {
                        "stage": "default target ratio",
                        "values": {"default_target_ratio": default_target},
                    }
                )
                selected = default_target
                tiebreak["resolved_by"] = tiebreak["stages"][-1]["stage"]
        tiebreak["selected_target_ratio"] = selected

    return {
        "validity_gate": gate,
        "decision": {
            "outcome": "TARGET_SELECTED",
            "selected_target_ratio": selected,
            "selected_role": (
                "default_target_ratio"
                if selected == default_target
                else "qualified_challenger"
            ),
            "selection_branch": branch,
            "formal_campaign_modified": False,
        },
        "qualified_challengers": qualified_targets,
        "challenger_evidence": challenger_evidence,
        "deltas": deltas,
        "tiebreak": tiebreak,
        "comparisons_use_unrounded_values": True,
    }


def _decorate_cells(
    cells: Sequence[Mapping[str, object]], decision_evidence: Mapping[str, object]
) -> list[dict[str, object]]:
    default_target = FORMAL_TARGET
    qualified = set(decision_evidence["qualified_challengers"])
    selected = decision_evidence["decision"]["selected_target_ratio"]
    deltas = {
        (item["target_ratio"], item["num_ssu"]): item
        for item in decision_evidence["deltas"]
    }
    decorated = []
    for source in cells:
        cell = dict(source)
        key = (cell["target_ratio"], cell["num_ssu"])
        if cell["target_ratio"] == default_target:
            cell["delta_alpha_1p5_attainment_vs_default"] = 0.0
            cell["delta_utilization_vs_default"] = 0.0
            cell["qualified_challenger"] = ""
        else:
            delta = deltas[key]
            cell["delta_alpha_1p5_attainment_vs_default"] = delta[
                "delta_alpha1p5_equal_npu_attainment"
            ]
            cell["delta_utilization_vs_default"] = delta["delta_mean_npu_utilization"]
            cell["qualified_challenger"] = cell["target_ratio"] in qualified
        cell["selected_by_preregistered_rule"] = cell["target_ratio"] == selected
        decorated.append(cell)
    return decorated


def _discover_shards(input_dir: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        paths = [path.expanduser().resolve() for path in explicit]
    else:
        resolved = input_dir.expanduser().resolve()
        _require(
            resolved.is_dir(),
            "calibration input directory does not exist: "
            f"{resolved}; expected nine files named cal_t{{0680,0686667,0700}}_"
            "ssu{16,20,24}.json",
        )
        paths = sorted(resolved.glob("*.json"))
    _require(
        len(paths) == 9,
        f"expected exactly nine calibration JSON files, found {len(paths)}: "
        + ", ".join(path.name for path in paths),
    )
    _require(len(set(paths)) == len(paths), "the same shard path was supplied twice")
    expected_names = {
        _expected_filename(target, num_ssu) for target in TARGETS for num_ssu in SSUS
    }
    _require(
        {path.name for path in paths} == expected_names,
        "calibration filenames differ from frozen nine-cell grid: "
        + ", ".join(sorted(expected_names)),
    )
    for path in paths:
        _portable_path(path)
        _require(path.is_file(), f"shard is not a regular file: {path}")
    return paths


def _csv_text(cells: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(cells)
    value = stream.getvalue()
    _require("/home/" not in value, "calibration CSV contains an absolute home path")
    return value


def _markdown_text(
    cells: Sequence[Mapping[str, object]],
    common: Mapping[str, object],
    decision_evidence: Mapping[str, object],
    rule: RuleDocument,
    report_json_path: str,
    report_json_sha256: str,
    rows_csv_path: str,
    rows_csv_sha256: str,
) -> str:
    lines = [
        "# selected128 α=1.5 calibration v2 decision evidence",
        "",
        "> Mechanical application of the preregistered v2 rule. The decision is "
        "reported as evidence and is not applied to the formal campaign.",
        "",
        "## Mechanical decision",
        "",
        f"- Outcome: `{decision_evidence['decision']['outcome']}`",
        f"- Selected target ratio: `{decision_evidence['decision']['selected_target_ratio']}`",
        f"- Selection branch: `{decision_evidence['decision']['selection_branch']}`",
        f"- Qualified challengers: `{decision_evidence['qualified_challengers']}`",
        f"- Validity gate passed: `{decision_evidence['validity_gate']['passed']}`",
        "- Formal campaign modified: `False`",
        "",
        "## Contract",
        "",
        f"- Case: `{CASE_NAME}`",
        f"- Seed: `{SEED}`",
        f"- Measurement duration: `{common['measurement_ms']:g} ms`",
        f"- SSUs: `{', '.join(map(str, SSUS))}`",
        f"- Candidate target ratios: `{', '.join(format(value, '.16g') for value in TARGETS)}`",
        f"- Current formal target shown for context only: `{FORMAL_TARGET:.16g}`",
        "- SLO aggregation: equal weight per NPU; request-weighted values are secondary evidence.",
        f"- Preregistered rule: `{_portable_path(rule.path)}` (`{rule.sha256}`)",
        "",
        "## Per-cell evidence",
        "",
        "| target | SSUs | mean NPU util % | α1.5 equal-NPU SLO % | α2 equal-NPU SLO % | requests |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| {target:.16g} | {ssu} | {util:.6f} | {a1:.6f} | {a2:.6f} | {count} |".format(
                target=cell["target_ratio"],
                ssu=cell["num_ssu"],
                util=cell["mean_npu_utilization_pct"],
                a1=cell["alpha_1p5_equal_npu_slo_pct"],
                a2=cell["alpha_2_equal_npu_slo_pct"],
                count=cell["measurement_request_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Statistics across SSUs 16/20/24",
            "",
            "| target | formal context | util mean/min/max % | α1.5 SLO mean/min/max % | α2 SLO mean/min/max % |",
            "|---:|:---:|:---|:---|:---|",
        ]
    )
    for summary in common["target_summaries"]:
        stats = summary["statistics_across_ssus"]
        util = stats["mean_npu_utilization_pct"]
        alpha_1p5 = stats["alpha_1p5_equal_npu_slo_pct"]
        alpha_2 = stats["alpha_2_equal_npu_slo_pct"]
        lines.append(
            "| {target:.16g} | {formal} | {um:.6f}/{umin:.6f}/{umax:.6f} | "
            "{a1m:.6f}/{a1min:.6f}/{a1max:.6f} | "
            "{a2m:.6f}/{a2min:.6f}/{a2max:.6f} |".format(
                target=summary["target_ratio"],
                formal="yes" if summary["is_current_formal_target"] else "no",
                um=util["mean"],
                umin=util["min"],
                umax=util["max"],
                a1m=alpha_1p5["mean"],
                a1min=alpha_1p5["min"],
                a1max=alpha_1p5["max"],
                a2m=alpha_2["mean"],
                a2min=alpha_2["min"],
                a2max=alpha_2["max"],
            )
        )
    lines.extend(
        [
            "",
            "## Challenger qualification and deltas",
            "",
            "| challenger | qualified | min Δα1.5 | points Δ≥0.005 | mean Δα1.5 | mean Δutil |",
            "|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for challenger in decision_evidence["challenger_evidence"]:
        criteria = challenger["criteria"]
        lines.append(
            "| {target:.16g} | {qualified} | {minimum:.9f} | {points} | {mean_slo:.9f} | {mean_util:.9f} |".format(
                target=challenger["target_ratio"],
                qualified=challenger["qualified"],
                minimum=criteria["minimum_delta_at_every_ssu"]["observed"],
                points=criteria["ssu_points_with_delta_at_least_0p005"]["observed"],
                mean_slo=criteria["mean_delta_alpha1p5_attainment"]["observed"],
                mean_util=criteria["mean_delta_utilization"]["observed"],
            )
        )
    lines.extend(
        [
            "",
            "Deltas above are unrounded fractions (challenger minus default); thresholds are applied before display rounding.",
            "",
            "## Tiebreak evidence",
            "",
            "```json",
            json.dumps(
                decision_evidence["tiebreak"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Source fingerprint: `{common['source_fingerprint']}`",
            f"- Runtime identity: `{json.dumps(common['runtime_identity'], sort_keys=True)}`",
            f"- Thread limits: `{json.dumps(common['thread_limit_environment'], sort_keys=True)}`",
            f"- JSON evidence: `{report_json_path}` (`{report_json_sha256}`)",
            f"- CSV rows: `{rows_csv_path}` (`{rows_csv_sha256}`)",
            "",
            "The mechanical v2 decision is not a campaign mutation; applying any change to the formal campaign requires a separate explicit action.",
            "",
        ]
    )
    value = "\n".join(lines)
    _require("/home/" not in value, "Markdown contains an absolute home path")
    return value


def _assert_portable(value: object, context: str = "report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_portable(item, f"{context}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_portable(item, f"{context}[{index}]")
    elif isinstance(value, str):
        _require("/home/" not in value, f"{context}: contains an absolute home path")
        _require(not value.startswith("/"), f"{context}: contains an absolute path")


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _build_outputs(
    shards: Sequence[ShardDocument], rule: RuleDocument, output_dir: Path
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    cells, evidence, common = _validate_grid(shards)
    decision_evidence = _evaluate_rule(cells, rule)
    _require(
        decision_evidence["validity_gate"]["passed"] is True,
        "validated grid unexpectedly failed the preregistered validity gate",
    )
    cells = _decorate_cells(cells, decision_evidence)
    output_dir = output_dir.expanduser().resolve()
    _portable_path(output_dir)
    rows_path = output_dir / "rows.csv"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"

    rows_text = _csv_text(cells)
    rows_sha256 = sha256(rows_text.encode()).hexdigest()
    report = {
        "schema_version": 1,
        "analysis": "selected128_alpha1p5_calibration_decision_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "preregistered_rule_applied_mechanically": True,
        "formal_campaign_modified": False,
        "statement": (
            "The preregistered v2 rule is applied mechanically to held-out seed-43 "
            "evidence. Its decision is reported but not applied to the formal campaign."
        ),
        "rule_authentication": {
            "path": _portable_path(rule.path),
            "sha256": rule.sha256,
            "size_bytes": rule.size_bytes,
            "rule": rule.document["rule"],
            "preregistered_at": rule.document["preregistered_at"],
            "formal_source_commit": rule.document["formal_source_commit"],
            "supersedes": rule.document["supersedes"],
            "supersession_reason": rule.document["supersession_reason"],
        },
        "preregistered_rule": rule.document,
        "contract": {
            "definition": DEFINITION,
            "case": CASE_NAME,
            "seed": SEED,
            "targets": list(TARGETS),
            "current_formal_target_for_context_only": FORMAL_TARGET,
            "ssu_counts": list(SSUS),
            "num_npu": NUM_NPU,
            "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
            "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
            "settle_ms": SETTLE_MS,
            "measurement_ms": common["measurement_ms"],
            "block_ms": BLOCK_MS,
            "calibration_mode": True,
            "campaign_spec_sha256": None,
        },
        "metric_definitions": {
            "request_slo": "ttft_ms <= alpha * ideal_ttft_ms + 1e-12",
            "equal_npu_slo": "mean of per-NPU request SLO attainment across 128 NPUs",
            "mean_npu_utilization": "mean of 128 measurement-window compute utilizations",
            "reported_alphas": [SENSITIVITY_ALPHA, PRIMARY_ALPHA],
            "target_statistics": "mean, min, max, and spread across SSU 16/20/24",
        },
        "common_evidence": {
            key: value for key, value in common.items() if key != "target_summaries"
        },
        "target_summaries": common["target_summaries"],
        "validity_gate": decision_evidence["validity_gate"],
        "decision": decision_evidence["decision"],
        "qualified_challengers": decision_evidence["qualified_challengers"],
        "challenger_evidence": decision_evidence["challenger_evidence"],
        "deltas": decision_evidence["deltas"],
        "tiebreak": decision_evidence["tiebreak"],
        "comparisons_use_unrounded_values": decision_evidence[
            "comparisons_use_unrounded_values"
        ],
        "cells": cells,
        "input_artifacts": evidence,
        "outputs": {
            "rows_csv": {
                "path": _portable_path(rows_path),
                "sha256": rows_sha256,
                "size_bytes": len(rows_text.encode()),
            },
            "markdown": {"path": _portable_path(markdown_path)},
        },
    }
    _assert_portable(report)
    report_text = (
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    report_sha256 = sha256(report_text.encode()).hexdigest()
    markdown_text = _markdown_text(
        cells,
        common,
        decision_evidence,
        rule,
        _portable_path(report_path),
        report_sha256,
        _portable_path(rows_path),
        rows_sha256,
    )
    _atomic_write(rows_path, rows_text)
    _atomic_write(report_path, report_text)
    _atomic_write(markdown_path, markdown_text)
    return rows_path, report_path, markdown_path, cells


def _portable_failure(message: str) -> str:
    project_prefix = PROJECT_ROOT.as_posix().rstrip("/") + "/"
    value = message.replace(project_prefix, "")
    _require("/home/" not in value, "gate failure contains an unportable path")
    return value


def _build_no_decision_outputs(
    rule: RuleDocument,
    output_dir: Path,
    failures: Sequence[str],
    shards: Sequence[ShardDocument] = (),
) -> tuple[Path, Path, Path]:
    portable_failures = [_portable_failure(message) for message in failures]
    evidence = _evaluate_rule([], rule, gate_failures=portable_failures)
    _require(
        evidence["decision"]["outcome"] == "NO_DECISION",
        "validity failure did not map to the preregistered NO_DECISION",
    )
    output_dir = output_dir.expanduser().resolve()
    _portable_path(output_dir)
    rows_path = output_dir / "rows.csv"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    rows_text = _csv_text([])
    rows_sha256 = sha256(rows_text.encode()).hexdigest()
    report = {
        "schema_version": 1,
        "analysis": "selected128_alpha1p5_calibration_decision_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "preregistered_rule_applied_mechanically": True,
        "formal_campaign_modified": False,
        "rule_authentication": {
            "path": _portable_path(rule.path),
            "sha256": rule.sha256,
            "size_bytes": rule.size_bytes,
            "rule": rule.document["rule"],
            "preregistered_at": rule.document["preregistered_at"],
            "supersedes": rule.document["supersedes"],
            "supersession_reason": rule.document["supersession_reason"],
        },
        "preregistered_rule": rule.document,
        "validity_gate": evidence["validity_gate"],
        "decision": evidence["decision"],
        "qualified_challengers": [],
        "challenger_evidence": [],
        "deltas": [],
        "tiebreak": evidence["tiebreak"],
        "input_artifacts_read_before_failure": [
            {
                "path": _portable_path(shard.path),
                "sha256": shard.sha256,
                "size_bytes": shard.size_bytes,
            }
            for shard in shards
        ],
        "outputs": {
            "rows_csv": {
                "path": _portable_path(rows_path),
                "sha256": rows_sha256,
                "size_bytes": len(rows_text.encode()),
            },
            "markdown": {"path": _portable_path(markdown_path)},
        },
    }
    _assert_portable(report)
    report_text = (
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    report_sha256 = sha256(report_text.encode()).hexdigest()
    markdown_lines = [
        "# selected128 α=1.5 calibration v2 — NO_DECISION",
        "",
        "> The preregistered validity gate failed. Per the frozen rule, no target "
        "decision is permitted and the formal campaign remains untouched.",
        "",
        f"- Rule: `{_portable_path(rule.path)}` (`{rule.sha256}`)",
        "- Outcome: `NO_DECISION`",
        "- Formal campaign modified: `False`",
        "",
        "## Gate failures",
        "",
        *[f"- {message}" for message in portable_failures],
        "",
        "## Outputs",
        "",
        f"- JSON evidence: `{_portable_path(report_path)}` (`{report_sha256}`)",
        f"- CSV rows: `{_portable_path(rows_path)}` (`{rows_sha256}`)",
        "",
    ]
    markdown_text = "\n".join(markdown_lines)
    _require("/home/" not in markdown_text, "NO_DECISION Markdown is not portable")
    _atomic_write(rows_path, rows_text)
    _atomic_write(report_path, report_text)
    _atomic_write(markdown_path, markdown_text)
    return rows_path, report_path, markdown_path


def _fake_sha(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _synthetic_runtime() -> dict[str, object]:
    return {
        "hostname": "synthetic-host",
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
        "thread_limit_environment": EXPECTED_THREADS,
    }


def _synthetic_documents() -> list[ShardDocument]:
    source_manifest = dict(HISTORICAL_SOURCE_MANIFEST)
    source_fingerprint = HISTORICAL_SOURCE_FINGERPRINT
    catalog_rows = []
    profile_keys = []
    for index in range(84):
        key = (32 + index, 64 + index)
        profile_keys.append(key)
        catalog_rows.append([key[0], key[1], [10.0 + index, 1000.0, 16.0, 0.01]])
    assignment_rows = []
    categories = tuple(sorted(CATEGORIES))
    for request_id in range(NUM_NPU * BACKING_REQUESTS_PER_NPU):
        npu = request_id % NUM_NPU
        sequence = request_id // NUM_NPU
        profile = profile_keys[request_id % len(profile_keys)]
        assignment_rows.append(
            [request_id, npu, sequence, categories[request_id % 4], list(profile)]
        )
    global_hashes = {
        field: _fake_sha(field)
        for field in ("catalog", "recipe", "schedule", "assignment")
    }
    prefix_hash = _fake_sha("prefix_32_assignment")
    full_hash = global_hashes["assignment"]
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
    runtime = _synthetic_runtime()
    documents = []
    for target_index, target in enumerate(TARGETS):
        definition = _definition(target)
        case = _case(target)
        definition_fingerprint = _definition_fingerprint(definition)
        spec = {
            "schema_version": 3,
            "experiment": EXPERIMENT,
            "definition": DEFINITION,
            "definition_fingerprint": definition_fingerprint,
            "num_npu": NUM_NPU,
            "scale_semantics": {
                "num_npu": NUM_NPU,
                "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
                "total_assignment_count": NUM_NPU * BACKING_REQUESTS_PER_NPU,
            },
            "campaign_spec_sha256": None,
            "n_layers": N_LAYERS,
            "batch_size": BATCH_SIZE,
            "default_ssu_list": list(definition.default_ssus),
            "cases": [asdict(item) for item in definition.cases],
            "report_roles": {},
            "workload": {
                "mode": "iid_uniform_profile_catalog_v1",
                "seed": SEED,
                "requests_per_npu": BACKING_REQUESTS_PER_NPU,
                **global_hashes,
                "prefix_32_assignment_hash": prefix_hash,
                "full_assignment_hash": full_hash,
                "sampling": "IID uniform with replacement over all 84 data profiles",
                "per_npu_streams": "independent and prefix-stable",
                "scientific_prefix_requests_per_npu": 32,
                "authentication": input_authentication,
            },
            "steady_state": {
                "seed": SEED,
                "requests_per_npu": BACKING_REQUESTS_PER_NPU,
                "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
                "settle_ms": SETTLE_MS,
                "measurement_ms": 4_000.0,
                "block_ms": BLOCK_MS,
                "slo_alpha": PRIMARY_ALPHA,
                "calibration_mode": True,
            },
            "adaptive": {
                **asdict(definition.adaptive),
                "ssd_cap_gbps": 16.0,
                "npu_cap_gbps": 16.0,
            },
            "adaptive_case_profiles": {
                item.name: {
                    "tuning_slo_alpha": item.tuning_slo_alpha,
                    **asdict(item.controller_definition()),
                }
                for item in definition.cases
                if isinstance(item, AdaptiveCase)
            },
            "thread_limit_environment": EXPECTED_THREADS,
            "runtime_identity": SELECTED128_EXPECTED_RUNTIME_IDENTITY,
            "cross_request_layer0_prefetch": True,
            "placement": "token-block ring hash reused across all 16 layers",
            "source_files": list(source_manifest),
        }
        config_fingerprint = _canonical_hash(spec, b"ms-scale-control-config:v1\0")
        for num_ssu in SSUS:
            simulator_input = _fake_sha(f"simulator-{num_ssu}")
            workload_input = _fake_sha(f"workload-{num_ssu}")
            placement_input = _fake_sha(f"placement-{num_ssu}")
            trace_input = _fake_sha(f"trace-{num_ssu}")
            request_rows = []
            for npu in range(NUM_NPU):
                request_id = 8 * NUM_NPU + npu
                assignment = assignment_rows[request_id]
                profile = assignment[4]
                ideal = 16.0
                ratio = 1.42 + 0.16 * ((npu + num_ssu) % 4) - 0.02 * target_index
                ttft = ideal * ratio
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
                        "slo_met": ttft <= PRIMARY_ALPHA * ideal + SLO_EPSILON,
                    }
                )
            cohort_material = [
                [row["request_id"], row["npu_id"], row["sequence"], row["profile_key"]]
                for row in request_rows
            ]
            cohort_fingerprint = _canonical_hash(
                cohort_material, b"ms-scale-control-measurement-cohort:v1\0"
            )
            alpha_2 = _slo_metrics(request_rows, PRIMARY_ALPHA, "synthetic")
            utilization = 0.40 + 0.03 * target_index + num_ssu / 1_000.0
            evaluations = 12
            writes = 128
            transactions = 4
            commits = 4
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
                "measurement_duration_ms": 4_000.0,
                "pressure_ttl_ms": 0.0,
                "cir_write_threshold_gbps": 0.0,
                "control_min_interval_ms": 25.0,
                "adaptive_controller_profile": _actual_profile(case),
                "invariants": {name: True for name in EXPECTED_INVARIANTS},
                "warmup_reached_ms": 500.0,
                "measurement_start_ms": 1_000.0,
                "measurement_end_ms": 5_000.0,
                "drain_stop_ms": 5_100.0,
                "tail_drain_ms": 100.0,
                "request_rows": request_rows,
                "request_counts_by_npu": [1] * NUM_NPU,
                "measurement_request_count": NUM_NPU,
                "ttft_slo_attainment": alpha_2[0],
                "request_weighted_slo_attainment": alpha_2[1],
                "npu_utilizations": [utilization] * NUM_NPU,
                "compute_ms_by_npu": [4_000.0 * utilization] * NUM_NPU,
                "mean_npu_utilization": utilization,
                "measurement_pressure_reports": 0,
                "measurement_pressure_reports_by_ssu": [0] * num_ssu,
                "measurement_pressure_cache_hits": 0,
                "measurement_pressure_cache_hits_by_ssu": [0] * num_ssu,
                "measurement_pressure_requests": 0,
                "measurement_pressure_requests_by_ssu": [0] * num_ssu,
                "measurement_control_evaluations": evaluations,
                "measurement_cir_path_writes": writes,
                "measurement_cir_path_writes_by_ssu": [writes] + [0] * (num_ssu - 1),
                "measurement_cir_write_transactions": transactions,
                "measurement_cir_write_transactions_by_ssu": [transactions]
                + [0] * (num_ssu - 1),
                "measurement_cir_commits": commits,
                "input_fingerprint": simulator_input,
            }
            case_fingerprint = _canonical_hash(
                {
                    "case": asdict(case),
                    "num_ssu": num_ssu,
                    "source_fingerprint": source_fingerprint,
                    "config_fingerprint": config_fingerprint,
                },
                b"ms-scale-control-case:v1\0",
            )
            row = {
                "status": "ok",
                "case": CASE_NAME,
                "family": case.family,
                "kind": case.kind,
                "role": None,
                "num_ssu": num_ssu,
                "num_npu": NUM_NPU,
                "backing_requests_per_npu": BACKING_REQUESTS_PER_NPU,
                "definition": DEFINITION,
                "definition_fingerprint": definition_fingerprint,
                "case_spec": asdict(case),
                "source_fingerprint": source_fingerprint,
                "config_fingerprint": config_fingerprint,
                "campaign_spec_sha256": None,
                "case_fingerprint": case_fingerprint,
                "input_fingerprints": {
                    **global_hashes,
                    "prefix_32_assignment": prefix_hash,
                    "full_assignment": full_hash,
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
                "measurement_cohort_fingerprint": cohort_fingerprint,
                "stationarity_diagnostics": {},
                "runtime": runtime,
                "wall_time_s": 1.0,
                "steady_summary": summary,
            }
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
                "definition_fingerprint": definition_fingerprint,
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
                "campaign_spec_sha256": None,
                "campaign_spec_authentication": None,
                "ending_campaign_spec_authentication": None,
                "experiment_spec": spec,
                "selected_ssus": [num_ssu],
                "selected_cases": [CASE_NAME],
                "selected_keys": [[CASE_NAME, num_ssu]],
                "schedule_metadata": schedule_metadata,
                "pairing_audit": {
                    str(num_ssu): {
                        "cases": [CASE_NAME],
                        "has_rows": True,
                        "all_available_rows_paired": True,
                    }
                },
                "results": [row],
            }
            path = (
                PROJECT_ROOT
                / "results"
                / "synthetic_calibration"
                / _expected_filename(target, num_ssu)
            )
            documents.append(
                ShardDocument(path, payload, _fake_sha(path.name), size_bytes=1)
            )
    return documents


def _self_test() -> dict[str, object]:
    _require(
        _canonical_hash(HISTORICAL_SOURCE_MANIFEST, b"ms-scale-control-source:v1\0")
        == HISTORICAL_SOURCE_FINGERPRINT,
        "historical calibration source manifest fingerprint changed",
    )
    for target, expected_fingerprint in HISTORICAL_DEFINITION_FINGERPRINTS.items():
        definition = _definition(target)
        _require(
            definition.experiment_name == EXPERIMENT
            and definition.default_measurement_ms == 8_000.0
            and _definition_fingerprint(definition) == expected_fingerprint,
            f"historical definition identity changed for target {target}",
        )
    _require(
        _fraction(math.nextafter(1.0, math.inf), "self-test upper roundoff") == 1.0
        and _fraction(math.nextafter(0.0, -math.inf), "self-test lower roundoff")
        == 0.0,
        "self-test did not clamp ulp-scale fraction roundoff",
    )
    material_fraction_rejected = False
    try:
        _fraction(1.0 + 1e-6, "self-test material fraction violation")
    except CalibrationReportError:
        material_fraction_rejected = True
    _require(
        material_fraction_rejected,
        "self-test accepted a material fraction-domain violation",
    )
    rule = _read_rule(DEFAULT_RULE_PATH)
    _require(
        rule.document["formal_source_commit"] == HISTORICAL_SOURCE_COMMIT,
        "calibration rule no longer binds the frozen source commit",
    )
    documents = _synthetic_documents()
    cells, evidence, common = _validate_grid(documents)
    _require(len(cells) == 9, "self-test did not produce nine cells")
    _require(
        len(common["target_summaries"]) == 3, "self-test did not produce three targets"
    )
    decision_evidence = _evaluate_rule(cells, rule)
    _require(
        decision_evidence["decision"]["selected_target_ratio"] == FORMAL_TARGET,
        "self-test no-qualified branch did not retain the default",
    )
    decorated_cells = _decorate_cells(cells, decision_evidence)
    csv_text = _csv_text(decorated_cells)
    _assert_portable({"cells": cells, "evidence": evidence, "common": common})
    markdown = _markdown_text(
        decorated_cells,
        common,
        decision_evidence,
        rule,
        "results/synthetic/report.json",
        "a" * 64,
        "results/synthetic/rows.csv",
        sha256(csv_text.encode()).hexdigest(),
    )
    _require("formal campaign" in markdown, "self-test campaign disclaimer missing")
    _require("Mechanical decision" in markdown, "self-test decision evidence missing")

    two_qualified_cells = copy.deepcopy(cells)
    for cell in two_qualified_cells:
        if cell["target_ratio"] == FORMAL_TARGET:
            cell["alpha_1p5_equal_npu_slo_pct"] = 50.0
            cell["mean_npu_utilization_pct"] = 50.0
        elif cell["target_ratio"] == 0.68:
            cell["alpha_1p5_equal_npu_slo_pct"] = 51.0
            cell["mean_npu_utilization_pct"] = 50.1
        else:
            cell["alpha_1p5_equal_npu_slo_pct"] = 50.8
            cell["mean_npu_utilization_pct"] = 50.2
    tiebreak_evidence = _evaluate_rule(two_qualified_cells, rule)
    _require(
        tiebreak_evidence["qualified_challengers"] == [0.68, 0.70]
        and tiebreak_evidence["decision"]["selected_target_ratio"] == 0.68
        and tiebreak_evidence["tiebreak"]["resolved_by"]
        == "closer target ratio to the default",
        "self-test two-challenger tiebreak differs from preregistered rule",
    )
    no_decision = _evaluate_rule([], rule, gate_failures=["synthetic gate failure"])
    _require(
        no_decision["decision"]["outcome"] == "NO_DECISION",
        "self-test validity failure did not produce NO_DECISION",
    )
    with TemporaryDirectory(
        prefix="selected128-calibration-report-", dir=PROJECT_ROOT / "results"
    ) as temporary:
        rows_path, report_path, markdown_path, written_cells = _build_outputs(
            documents, rule, Path(temporary)
        )
        _require(len(written_cells) == 9, "self-test output lost calibration cells")
        for path in (rows_path, report_path, markdown_path):
            _require(
                path.is_file() and path.stat().st_size > 0, f"{path}: output missing"
            )
            _require(
                "/home/" not in path.read_text(encoding="utf-8"),
                f"{path}: not portable",
            )
        no_rows, no_report, no_markdown = _build_no_decision_outputs(
            rule,
            Path(temporary) / "no_decision",
            ["synthetic validity failure"],
        )
        no_payload = json.loads(no_report.read_text(encoding="utf-8"))
        _require(
            no_payload["decision"]["outcome"] == "NO_DECISION"
            and no_payload["complete"] is False,
            "self-test NO_DECISION artifacts differ from the frozen failure rule",
        )
        for path in (no_rows, no_report, no_markdown):
            _require(path.is_file(), f"{path}: NO_DECISION output missing")

    bad_documents = list(documents)
    first = documents[0]
    bad_payload = copy.deepcopy(first.payload)
    bad_payload["results"][0]["steady_summary"]["adaptive_controller_profile"][
        "target_ratio"
    ] = 0.99
    bad_documents[0] = ShardDocument(
        first.path,
        bad_payload,
        first.sha256,
        first.size_bytes,
    )
    bad_profile_rejected = False
    try:
        _validate_grid(bad_documents)
    except CalibrationReportError:
        bad_profile_rejected = True
    _require(bad_profile_rejected, "self-test accepted the wrong actual profile")
    return {
        "self_test": "passed",
        "cell_count": len(cells),
        "target_count": len(common["target_summaries"]),
        "ssu_count": len(SSUS),
        "bad_actual_profile_rejected": True,
        "preregistered_rule_sha256": rule.sha256,
        "no_qualified_branch_checked": True,
        "two_challenger_tiebreak_checked": True,
        "gate_failure_no_decision_checked": True,
        "portable_json_csv_markdown_checked": True,
        "fraction_roundoff_clamped": True,
        "material_fraction_violation_rejected": True,
        "historical_definition_fingerprints_checked": True,
        "historical_source_fingerprint_checked": True,
        "historical_source_commit_checked": True,
        "formal_campaign_modified": False,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rule",
        type=Path,
        default=DEFAULT_RULE_PATH,
        help="preregistered v2 decision rule; raw bytes are authenticated in the report",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--shard",
        action="append",
        type=Path,
        help="explicit shard path; repeat exactly nine times instead of --input-dir",
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
    rule = _read_rule(args.rule)
    if not args.shard and not args.input_dir.expanduser().resolve().is_dir():
        _discover_shards(args.input_dir, ())
    shards = []
    try:
        paths = _discover_shards(args.input_dir, tuple(args.shard or ()))
        for path in paths:
            shards.append(_read_shard(path))
        rows_path, report_path, markdown_path, cells = _build_outputs(
            shards, rule, args.output_dir
        )
    except CalibrationReportError as error:
        rows_path, report_path, markdown_path = _build_no_decision_outputs(
            rule,
            args.output_dir,
            [str(error)],
            shards,
        )
        cells = []
    decision = json.loads(report_path.read_text(encoding="utf-8"))["decision"]
    print(
        json.dumps(
            {
                "complete": decision["outcome"] != "NO_DECISION",
                "preregistered_rule_applied_mechanically": True,
                "formal_campaign_modified": False,
                "cell_count": len(cells),
                "decision": decision,
                "rows_csv": _portable_path(rows_path),
                "report_json": _portable_path(report_path),
                "report_markdown": _portable_path(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CalibrationReportError as error:
        raise SystemExit(f"ERROR: {error}") from error
