"""Build a verified seven-SSU summary for the selected CIR-control settings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, median


PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_SOURCE_FINGERPRINT = (
    "7e994146eeab627dde5d5cacf75945e4d611de64cd3b109c5716b9fe8c75ea83"
)
EXPECTED_CONFIG_FINGERPRINT = (
    "a3a9d2b4962629478e7d2550effa881b53d8b422ec3a1fa6e95544348b1864bd"
)
EXPECTED_FORMAL_SUMMARY_SHA256 = (
    "25905e8ee9f4d88f440ac9bd3991880149bddbabfe98d0d851d808b447ecbb5b"
)
FORMAL_SSUS = (6, 10, 18)
SUPPLEMENTAL_SSUS = (2, 3, 4, 5)
SSUS = SUPPLEMENTAL_SSUS + FORMAL_SSUS
CASES = (
    "baseline",
    "layer_once_ttl_0ms",
    "layer_once_ttl_2ms",
    "layer_once_ttl_5ms",
    "adaptive_t0_i25ms",
    "adaptive_t0_i100ms",
    "adaptive_t0_i200ms",
)
CASE_SPECS = {
    "baseline": ("baseline", "baseline", 0.0, 0.0, 0.0),
    "layer_once_ttl_0ms": ("ttl", "layer_once", 0.0, 0.0, 0.0),
    "layer_once_ttl_2ms": ("ttl", "layer_once", 2.0, 0.0, 0.0),
    "layer_once_ttl_5ms": ("ttl", "layer_once", 5.0, 0.0, 0.0),
    "adaptive_t0_i25ms": (
        "threshold_interval",
        "adaptive",
        0.0,
        0.0,
        25.0,
    ),
    "adaptive_t0_i100ms": ("interval", "adaptive", 0.0, 0.0, 100.0),
    "adaptive_t0_i200ms": ("interval", "adaptive", 0.0, 0.0, 200.0),
}
EXPECTED_INVARIANTS = {
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
}
REQUIRED_INPUT_FINGERPRINTS = {
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
GLOBAL_PAIRED_FINGERPRINTS = (
    "catalog",
    "recipe",
    "schedule",
    "assignment",
    "prefix_32_assignment",
    "full_assignment",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
DEFAULT_FORMAL_SUMMARY = Path(
    "results/ms_scale_control/final_measure8_backing128_analysis/summary.csv"
)
DEFAULT_SUPPLEMENTAL_SHARDS = (
    Path("results/ms_scale_control/frozen32_ssu2_selected7_measure8_backing128.json"),
    Path("results/ms_scale_control/frozen32_ssu3_selected7_measure8_backing128.json"),
    Path("results/ms_scale_control/remote_aux_ssu5.json"),
    Path("results/ms_scale_control/frozen32_ssu4_selected7_measure8_backing128.json"),
    Path("results/ms_scale_control/frozen32_ssu5_ttl2_measure8_backing128.json"),
)
DEFAULT_OUTPUT = Path(
    "results/ms_scale_control/selected_settings_alpha1p5_ssu2_5_analysis/summary.csv"
)
OUTPUT_FIELDS = (
    "case",
    "family",
    "kind",
    "num_ssu",
    "pressure_ttl_ms",
    "cir_write_threshold_gbps",
    "min_interval_ms",
    "mean_npu_utilization_pct",
    "primary_slo_alpha",
    "sensitivity_slo_alpha",
    "equal_npu_slo_pct",
    "request_weighted_slo_pct",
    "alpha_1p5_equal_npu_slo_pct",
    "alpha_1p5_request_weighted_slo_pct",
    "measurement_request_count",
    "measurement_requests_per_npu_min",
    "measurement_requests_per_npu_median",
    "measurement_requests_per_npu_max",
    "measurement_cohort_fingerprint",
    "input_simulator_fingerprint",
    "all_invariants_passed",
    "source_artifact",
)


class SummaryError(ValueError):
    """Raised when an input artifact cannot support a comparable curve."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--supplemental-shard", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def _as_float(value: object, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SummaryError(f"{context}: expected a number") from error
    _require(math.isfinite(result), f"{context}: expected a finite number")
    return result


def _as_int(value: object, context: str) -> int:
    if isinstance(value, bool):
        raise SummaryError(f"{context}: expected an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise SummaryError(f"{context}: expected an integer")


def _close(left: float, right: float, *, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    """Return a repository-relative provenance path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _canonical_hash(value: object, namespace: bytes) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _require_sha256(value: object, context: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{context}: expected a lowercase SHA-256 fingerprint",
    )
    return value


def _require_percentage(value: object, context: str) -> float:
    result = _as_float(value, context)
    _require(0.0 <= result <= 100.0, f"{context}: percentage outside [0, 100]")
    return result


def _expected_case_spec(case: str) -> dict[str, object]:
    family, kind, ttl, threshold, interval = CASE_SPECS[case]
    return {
        "name": case,
        "family": family,
        "kind": kind,
        "pressure_ttl_ms": ttl,
        "cir_write_threshold_gbps": threshold,
        "min_interval_ms": interval,
    }


def _slo_metrics(
    request_rows: list[dict[str, object]], alpha: float, context: str
) -> tuple[float, float, list[int]]:
    outcomes_by_npu: dict[int, list[float]] = {npu: [] for npu in range(32)}
    all_outcomes: list[float] = []
    request_ids: set[int] = set()
    stream_positions: set[tuple[int, int]] = set()
    for index, request in enumerate(request_rows):
        _require(isinstance(request, dict), f"{context}/request[{index}]: not a dict")
        request_id = _as_int(
            request.get("request_id"), f"{context}/request[{index}]/request_id"
        )
        _require(
            request_id not in request_ids,
            f"{context}: duplicate measured request ID {request_id}",
        )
        request_ids.add(request_id)
        npu = _as_int(request.get("npu_id"), f"{context}/request[{index}]/npu")
        _require(npu in outcomes_by_npu, f"{context}: unexpected NPU {npu}")
        sequence = _as_int(
            request.get("sequence"), f"{context}/request[{index}]/sequence"
        )
        _require(0 <= sequence < 128, f"{context}: sequence outside backing prefix")
        stream_position = (npu, sequence)
        _require(
            stream_position not in stream_positions,
            f"{context}: duplicate NPU/sequence position {stream_position}",
        )
        stream_positions.add(stream_position)
        _require(
            request.get("category") in {"SS", "SL", "LS", "LL"},
            f"{context}: invalid request category",
        )
        ttft_ms = _as_float(request.get("ttft_ms"), f"{context}/request[{index}]/ttft")
        admission_ms = _as_float(
            request.get("admission_time_ms"),
            f"{context}/request[{index}]/admission_time",
        )
        completion_ms = _as_float(
            request.get("completion_time_ms"),
            f"{context}/request[{index}]/completion_time",
        )
        ideal_ms = _as_float(
            request.get("ideal_ttft_ms"),
            f"{context}/request[{index}]/ideal_ttft",
        )
        _require(
            completion_ms >= admission_ms, f"{context}: completion before admission"
        )
        _require(
            _close(completion_ms - admission_ms, ttft_ms, tolerance=1e-8),
            f"{context}: TTFT does not match completion-admission",
        )
        _require(ttft_ms >= 0.0, f"{context}: negative TTFT")
        _require(ideal_ms > 0.0, f"{context}: non-positive ideal TTFT")
        passed = ttft_ms <= alpha * ideal_ms + 1e-12
        if alpha == 2.0:
            _require(
                request.get("slo_met") is passed,
                f"{context}: stored per-request alpha2 outcome mismatch",
            )
        outcome = float(passed)
        outcomes_by_npu[npu].append(outcome)
        all_outcomes.append(outcome)

    _require(all_outcomes, f"{context}: measurement cohort is empty")
    missing = [npu for npu, values in outcomes_by_npu.items() if not values]
    _require(not missing, f"{context}: no measured request on NPUs {missing}")
    counts = [len(outcomes_by_npu[npu]) for npu in range(32)]
    equal_npu_pct = 100.0 * fmean(fmean(outcomes_by_npu[npu]) for npu in range(32))
    request_weighted_pct = 100.0 * fmean(all_outcomes)
    return equal_npu_pct, request_weighted_pct, counts


def _load_formal_rows(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    try:
        summary_sha256 = _sha256(path)
        with path.open(encoding="utf-8", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise SummaryError(f"cannot read formal summary {path}: {error}") from error
    _require(
        summary_sha256 == EXPECTED_FORMAL_SUMMARY_SHA256,
        f"{path}: frozen formal summary SHA-256 mismatch",
    )

    rows: dict[tuple[str, int], dict[str, object]] = {}
    for source in source_rows:
        case = source.get("case")
        if case not in CASES:
            continue
        num_ssu = _as_int(source.get("num_ssu"), f"{case}/num_ssu")
        if num_ssu not in FORMAL_SSUS:
            continue
        key = (case, num_ssu)
        _require(key not in rows, f"formal summary has duplicate row {key}")
        family, kind, ttl, threshold, interval = CASE_SPECS[case]
        context = f"formal/{case}/SSU{num_ssu}"
        _require(source.get("family") == family, f"{context}: family mismatch")
        _require(source.get("kind") == kind, f"{context}: kind mismatch")
        actual_knobs = tuple(
            _as_float(source.get(field), f"{context}/{field}")
            for field in (
                "pressure_ttl_ms",
                "cir_write_threshold_gbps",
                "min_interval_ms",
            )
        )
        _require(
            actual_knobs == (ttl, threshold, interval),
            f"{context}: control knob mismatch {actual_knobs}",
        )
        _require(
            source.get("all_invariants_passed") == "True",
            f"{context}: formal invariants did not pass",
        )
        _require(
            _as_float(source.get("primary_slo_alpha"), context) == 2.0,
            f"{context}: primary alpha mismatch",
        )
        _require(
            _as_float(source.get("sensitivity_slo_alpha"), context) == 1.5,
            f"{context}: sensitivity alpha mismatch",
        )
        for field in (
            "mean_npu_utilization_pct",
            "equal_npu_slo_pct",
            "request_weighted_slo_pct",
            "alpha_1p5_equal_npu_slo_pct",
            "alpha_1p5_request_weighted_slo_pct",
        ):
            _require_percentage(source.get(field), f"{context}/{field}")
        request_count = _as_int(
            source.get("measurement_request_count"),
            f"{context}/measurement_request_count",
        )
        minimum_count = _as_int(
            source.get("measurement_requests_per_npu_min"),
            f"{context}/measurement_requests_per_npu_min",
        )
        maximum_count = _as_int(
            source.get("measurement_requests_per_npu_max"),
            f"{context}/measurement_requests_per_npu_max",
        )
        median_count = _as_float(
            source.get("measurement_requests_per_npu_median"),
            f"{context}/measurement_requests_per_npu_median",
        )
        _require(request_count > 0, f"{context}: empty measurement cohort")
        _require(
            0 < minimum_count <= median_count <= maximum_count,
            f"{context}: invalid per-NPU measurement counts",
        )
        _require(
            32 * minimum_count <= request_count <= 32 * maximum_count,
            f"{context}: fleet/per-NPU request counts disagree",
        )
        _require_sha256(
            source.get("measurement_cohort_fingerprint"),
            f"{context}/measurement_cohort_fingerprint",
        )
        rows[key] = {field: source.get(field, "") for field in OUTPUT_FIELDS}
        rows[key].update(
            {
                "family": family,
                "kind": kind,
                "input_simulator_fingerprint": "",
                "source_artifact": _portable_path(path),
            }
        )
    expected = {(case, ssu) for case in CASES for ssu in FORMAL_SSUS}
    _require(set(rows) == expected, "formal summary is missing selected rows")
    return rows


def _validate_shard_identity(payload: dict[str, object], path: Path) -> None:
    context = str(path)
    _require(payload.get("schema_version") == 1, f"{context}: schema mismatch")
    _require(
        payload.get("selected_complete") is True,
        f"{context}: selected shard is incomplete",
    )
    source_manifest = payload.get("source_manifest")
    _require(
        isinstance(source_manifest, dict) and source_manifest,
        f"{context}: source manifest missing",
    )
    _require(
        all(
            isinstance(name, str)
            and name
            and SHA256_PATTERN.fullmatch(value) is not None
            for name, value in source_manifest.items()
            if isinstance(value, str)
        )
        and all(isinstance(value, str) for value in source_manifest.values()),
        f"{context}: invalid source manifest entries",
    )
    authenticated_source = _canonical_hash(
        source_manifest, b"ms-scale-control-source:v1\0"
    )
    _require(
        authenticated_source == EXPECTED_SOURCE_FINGERPRINT,
        f"{context}: source manifest does not authenticate the frozen source",
    )
    experiment_spec = payload.get("experiment_spec")
    _require(
        isinstance(experiment_spec, dict) and experiment_spec,
        f"{context}: experiment specification missing",
    )
    authenticated_config = _canonical_hash(
        experiment_spec, b"ms-scale-control-config:v1\0"
    )
    _require(
        authenticated_config == EXPECTED_CONFIG_FINGERPRINT,
        f"{context}: experiment specification does not authenticate the config",
    )
    workload_spec = experiment_spec.get("workload")
    _require(
        isinstance(workload_spec, dict),
        f"{context}: authenticated workload specification missing",
    )
    authenticated_global_inputs = {
        "catalog": workload_spec.get("catalog"),
        "recipe": workload_spec.get("recipe"),
        "schedule": workload_spec.get("schedule"),
        "assignment": workload_spec.get("assignment"),
        "prefix_32_assignment": workload_spec.get("prefix_32_assignment_hash"),
        "full_assignment": workload_spec.get("full_assignment_hash"),
    }
    for name, value in authenticated_global_inputs.items():
        _require_sha256(value, f"{context}/experiment_spec/workload/{name}")
    expected_schedule_metadata = {
        "catalog": authenticated_global_inputs["catalog"],
        "recipe": authenticated_global_inputs["recipe"],
        "schedule": authenticated_global_inputs["schedule"],
        "assignment": authenticated_global_inputs["assignment"],
        "mode": workload_spec.get("mode"),
        "seed": workload_spec.get("seed"),
        "num_npu": experiment_spec.get("num_npu"),
        "requests_per_npu": workload_spec.get("requests_per_npu"),
    }
    _require(
        payload.get("schedule_metadata") == expected_schedule_metadata,
        f"{context}: schedule metadata is not bound to the authenticated config",
    )
    _require(
        payload.get("source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        f"{context}: source fingerprint mismatch",
    )
    _require(
        payload.get("ending_source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        f"{context}: ending source fingerprint mismatch",
    )
    _require(
        payload.get("source_stable_during_run") is True,
        f"{context}: source was not stable",
    )
    _require(
        payload.get("config_fingerprint") == EXPECTED_CONFIG_FINGERPRINT,
        f"{context}: config fingerprint mismatch",
    )
    _require(
        payload.get("ending_config_fingerprint") == EXPECTED_CONFIG_FINGERPRINT,
        f"{context}: ending config fingerprint mismatch",
    )
    _require(
        payload.get("config_stable_during_run") is True,
        f"{context}: config was not stable",
    )
    pairing_audit = payload.get("pairing_audit")
    _require(
        isinstance(pairing_audit, dict) and pairing_audit,
        f"{context}: pairing audit missing",
    )
    _require(
        all(
            isinstance(entry, dict) and entry.get("all_available_rows_paired") is True
            for entry in pairing_audit.values()
        ),
        f"{context}: shard pairing audit failed",
    )
    selected_keys = payload.get("selected_keys")
    results = payload.get("results")
    _require(isinstance(selected_keys, list), f"{context}: selected keys missing")
    _require(isinstance(results, list), f"{context}: result rows missing")
    normalized_selected = {
        (_as_int(key[1], f"{context}/selected_ssu"), str(key[0]))
        for key in selected_keys
        if isinstance(key, list) and len(key) == 2
    }
    _require(
        len(normalized_selected) == len(selected_keys),
        f"{context}: malformed or duplicate selected keys",
    )
    normalized_results = {
        (
            _as_int(result.get("num_ssu"), f"{context}/result_ssu"),
            str(result.get("case")),
        )
        for result in results
        if isinstance(result, dict)
    }
    _require(
        len(normalized_results) == len(results),
        f"{context}: malformed or duplicate result keys",
    )
    _require(
        normalized_results == normalized_selected,
        f"{context}: selected/result key mismatch",
    )
    for result in results:
        result_inputs = result.get("input_fingerprints")
        _require(
            isinstance(result_inputs, dict),
            f"{context}: result input fingerprints missing",
        )
        _require(
            all(
                result_inputs.get(name) == value
                for name, value in authenticated_global_inputs.items()
            ),
            f"{context}: row inputs are not bound to the authenticated config",
        )


def _row_from_supplemental(
    row: dict[str, object], path: Path
) -> tuple[tuple[str, int], dict[str, object], dict[str, str]]:
    case = row.get("case")
    num_ssu = _as_int(row.get("num_ssu"), f"{path}/num_ssu")
    _require(case in CASES, f"{path}: unexpected selected case {case!r}")
    _require(num_ssu in SUPPLEMENTAL_SSUS, f"{path}: unexpected SSU {num_ssu}")
    context = f"{path}/{case}/SSU{num_ssu}"
    _require(row.get("status") == "ok", f"{context}: status is not ok")
    _require(
        row.get("source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        f"{context}: row source mismatch",
    )
    _require(
        row.get("config_fingerprint") == EXPECTED_CONFIG_FINGERPRINT,
        f"{context}: row config mismatch",
    )
    _require(
        row.get("case_spec") == _expected_case_spec(case),
        f"{context}: case specification mismatch",
    )
    family, kind, ttl, threshold, interval = CASE_SPECS[case]
    _require(row.get("family") == family, f"{context}: row family mismatch")
    _require(row.get("kind") == kind, f"{context}: row kind mismatch")
    expected_case_fingerprint = _canonical_hash(
        {
            "case": _expected_case_spec(case),
            "num_ssu": num_ssu,
            "source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
            "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        },
        b"ms-scale-control-case:v1\0",
    )
    _require(
        row.get("case_fingerprint") == expected_case_fingerprint,
        f"{context}: case fingerprint mismatch",
    )

    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{context}: steady summary missing")
    _require(summary.get("num_npu") == 32, f"{context}: NPU count mismatch")
    _require(summary.get("num_ssu") == num_ssu, f"{context}: SSU mismatch")
    _require(summary.get("n_layers") == 16, f"{context}: layer count mismatch")
    _require(
        _as_float(summary.get("measurement_duration_ms"), context) == 8000.0,
        f"{context}: measurement duration mismatch",
    )
    _require(
        _as_float(summary.get("settle_ms"), context) == 500.0,
        f"{context}: settle duration mismatch",
    )
    _require(
        summary.get("warmup_requests_per_npu") == 8,
        f"{context}: warmup count mismatch",
    )
    _require(
        _as_float(summary.get("slo_alpha"), context) == 2.0,
        f"{context}: primary alpha mismatch",
    )
    invariants = summary.get("invariants")
    _require(
        isinstance(invariants, dict) and set(invariants) == EXPECTED_INVARIANTS,
        f"{context}: simulator invariant key set mismatch",
    )
    failed = sorted(name for name, passed in invariants.items() if passed is not True)
    _require(not failed, f"{context}: failed invariants {failed}")

    blocks = summary.get("measurement_blocks")
    _require(isinstance(blocks, list) and len(blocks) == 16, f"{context}: block count")
    for index, block in enumerate(blocks):
        _require(isinstance(block, dict), f"{context}: invalid block {index}")
        start_ms = _as_float(block.get("start_ms"), f"{context}/block{index}")
        end_ms = _as_float(block.get("end_ms"), f"{context}/block{index}")
        _require(_close(end_ms - start_ms, 500.0), f"{context}: block width")
        if index:
            prior_end = _as_float(
                blocks[index - 1].get("end_ms"), f"{context}/block{index - 1}"
            )
            _require(_close(start_ms, prior_end), f"{context}: block gap")

    request_rows = summary.get("request_rows")
    _require(isinstance(request_rows, list), f"{context}: request rows missing")
    equal_2, weighted_2, counts = _slo_metrics(request_rows, 2.0, context)
    equal_1p5, weighted_1p5, counts_1p5 = _slo_metrics(request_rows, 1.5, context)
    _require(counts == counts_1p5, f"{context}: inconsistent NPU counts")
    _require(
        len(request_rows)
        == _as_int(
            summary.get("measurement_request_count"),
            f"{context}/measurement_request_count",
        ),
        f"{context}: request count mismatch",
    )
    stored_counts = summary.get("request_counts_by_npu")
    _require(stored_counts == counts, f"{context}: per-NPU counts mismatch")
    stored_equal_2 = 100.0 * _as_float(
        summary.get("ttft_slo_attainment"), f"{context}/equal_alpha2"
    )
    stored_weighted_2 = 100.0 * _as_float(
        summary.get("request_weighted_slo_attainment"),
        f"{context}/weighted_alpha2",
    )
    _require(_close(equal_2, stored_equal_2), f"{context}: alpha2 equal-NPU mismatch")
    _require(
        _close(weighted_2, stored_weighted_2),
        f"{context}: alpha2 request-weighted mismatch",
    )

    npu_utilizations = summary.get("npu_utilizations")
    _require(
        isinstance(npu_utilizations, list) and len(npu_utilizations) == 32,
        f"{context}: NPU utilization vector mismatch",
    )
    utilization_values = [
        _as_float(value, f"{context}/npu_utilization") for value in npu_utilizations
    ]
    _require(
        all(0.0 <= value <= 1.0 for value in utilization_values),
        f"{context}: NPU utilization outside [0, 1]",
    )
    mean_util_pct = 100.0 * fmean(utilization_values)
    stored_mean_utilization = _as_float(
        summary.get("mean_npu_utilization"), f"{context}/mean_utilization"
    )
    _require(
        0.0 <= stored_mean_utilization <= 1.0,
        f"{context}: stored mean utilization outside [0, 1]",
    )
    stored_mean_util_pct = 100.0 * stored_mean_utilization
    _require(
        _close(mean_util_pct, stored_mean_util_pct),
        f"{context}: mean utilization mismatch",
    )

    input_fingerprints = row.get("input_fingerprints")
    _require(
        isinstance(input_fingerprints, dict),
        f"{context}: input fingerprints missing",
    )
    _require(
        set(input_fingerprints) == REQUIRED_INPUT_FINGERPRINTS,
        f"{context}: input fingerprint key set mismatch",
    )
    for name, value in input_fingerprints.items():
        _require_sha256(value, f"{context}/input_fingerprints/{name}")
    simulator_fingerprint = input_fingerprints["simulator"]
    _require(
        summary.get("input_fingerprint") == simulator_fingerprint,
        f"{context}: simulator input fingerprint binding mismatch",
    )
    cohort_fingerprint = _require_sha256(
        row.get("measurement_cohort_fingerprint"),
        f"{context}/measurement_cohort_fingerprint",
    )
    output = {
        "case": case,
        "family": family,
        "kind": kind,
        "num_ssu": num_ssu,
        "pressure_ttl_ms": ttl,
        "cir_write_threshold_gbps": threshold,
        "min_interval_ms": interval,
        "mean_npu_utilization_pct": mean_util_pct,
        "primary_slo_alpha": 2.0,
        "sensitivity_slo_alpha": 1.5,
        "equal_npu_slo_pct": equal_2,
        "request_weighted_slo_pct": weighted_2,
        "alpha_1p5_equal_npu_slo_pct": equal_1p5,
        "alpha_1p5_request_weighted_slo_pct": weighted_1p5,
        "measurement_request_count": len(request_rows),
        "measurement_requests_per_npu_min": min(counts),
        "measurement_requests_per_npu_median": median(counts),
        "measurement_requests_per_npu_max": max(counts),
        "measurement_cohort_fingerprint": cohort_fingerprint,
        "input_simulator_fingerprint": simulator_fingerprint,
        "all_invariants_passed": True,
        "source_artifact": _portable_path(path),
    }
    return (case, num_ssu), output, input_fingerprints


def _load_supplemental_rows(
    paths: tuple[Path, ...],
) -> tuple[
    dict[tuple[str, int], dict[str, object]], dict[tuple[str, int], dict[str, str]]
]:
    rows: dict[tuple[str, int], dict[str, object]] = {}
    fingerprints: dict[tuple[str, int], dict[str, str]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SummaryError(
                f"cannot read supplemental shard {path}: {error}"
            ) from error
        _require(isinstance(payload, dict), f"{path}: invalid top-level payload")
        _validate_shard_identity(payload, path)
        source_rows = payload.get("results")
        _require(isinstance(source_rows, list), f"{path}: results missing")
        for source_row in source_rows:
            _require(isinstance(source_row, dict), f"{path}: invalid result row")
            if source_row.get("case") not in CASES:
                continue
            num_ssu = _as_int(source_row.get("num_ssu"), f"{path}/num_ssu")
            if num_ssu not in SUPPLEMENTAL_SSUS:
                continue
            key, output, input_fingerprints = _row_from_supplemental(source_row, path)
            _require(key not in rows, f"duplicate supplemental row {key}")
            rows[key] = output
            fingerprints[key] = input_fingerprints

    expected = {(case, ssu) for case in CASES for ssu in SUPPLEMENTAL_SSUS}
    missing = sorted(expected - set(rows))
    _require(not missing, f"missing supplemental rows: {missing}")
    unexpected = sorted(set(rows) - expected)
    _require(not unexpected, f"unexpected supplemental rows: {unexpected}")
    for num_ssu in SUPPLEMENTAL_SSUS:
        paired = [fingerprints[(case, num_ssu)] for case in CASES]
        _require(
            all(value == paired[0] for value in paired[1:]),
            f"SSU{num_ssu}: selected strategy inputs are not paired",
        )
    reference = fingerprints[(CASES[0], SUPPLEMENTAL_SSUS[0])]
    for key in GLOBAL_PAIRED_FINGERPRINTS:
        _require(
            all(value[key] == reference[key] for value in fingerprints.values()),
            f"supplemental SSUs disagree on global input fingerprint {key}",
        )
    return rows, fingerprints


def _write_csv(path: Path, rows: dict[tuple[str, int], dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for num_ssu in SSUS:
            for case in CASES:
                writer.writerow(rows[(case, num_ssu)])
    temporary.replace(path)


def _write_manifest(
    path: Path,
    *,
    input_hashes: dict[Path, str],
    builder_path: Path,
    builder_sha256: str,
    output: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "source_fingerprint": EXPECTED_SOURCE_FINGERPRINT,
        "config_fingerprint": EXPECTED_CONFIG_FINGERPRINT,
        "num_npu": 32,
        "ssus": list(SSUS),
        "cases": list(CASES),
        "slo_alpha": 1.5,
        "slo_weighting": "equal_npu",
        "inputs": [
            {"path": _portable_path(input_path), "sha256": sha256}
            for input_path, sha256 in input_hashes.items()
        ],
        "builder": {
            "path": _portable_path(builder_path),
            "sha256": builder_sha256,
        },
        "output": {"path": _portable_path(output), "sha256": _sha256(output)},
        "row_count": len(SSUS) * len(CASES),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    formal_summary = args.formal_summary.expanduser().resolve()
    supplemental_shards = tuple(
        path.expanduser().resolve()
        for path in (args.supplemental_shard or DEFAULT_SUPPLEMENTAL_SHARDS)
    )
    output = args.output.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output.with_name("manifest.json")
    )
    inputs = (formal_summary,) + supplemental_shards
    _require(len(set(inputs)) == len(inputs), "input artifact paths are not unique")
    _require(output != manifest, "summary output and manifest paths must differ")
    _require(
        output not in inputs and manifest not in inputs,
        "derived output paths must not overwrite an input artifact",
    )
    builder_path = Path(__file__).resolve()
    _require(
        output != builder_path and manifest != builder_path,
        "derived output paths must not overwrite the builder source",
    )
    try:
        input_hashes = {path: _sha256(path) for path in inputs}
        builder_sha256 = _sha256(builder_path)
    except OSError as error:
        raise SummaryError(f"cannot hash input artifact: {error}") from error

    rows = _load_formal_rows(formal_summary)
    supplemental, _ = _load_supplemental_rows(supplemental_shards)
    overlap = sorted(set(rows) & set(supplemental))
    _require(not overlap, f"formal/supplemental row overlap: {overlap}")
    rows.update(supplemental)
    expected = {(case, ssu) for case in CASES for ssu in SSUS}
    _require(set(rows) == expected, "combined selected-setting matrix is incomplete")
    _require(
        all(_sha256(path) == sha256 for path, sha256 in input_hashes.items()),
        "an input artifact changed while the summary was being built",
    )
    _require(
        _sha256(builder_path) == builder_sha256,
        "the builder source changed while the summary was being built",
    )

    _write_csv(output, rows)
    _write_manifest(
        manifest,
        input_hashes=input_hashes,
        builder_path=builder_path,
        builder_sha256=builder_sha256,
        output=output,
    )
    print(f"wrote {output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
