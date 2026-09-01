"""Strict convergence analysis for the paired 32-NPU / 4-SSU 32-s run.

The analyzer is deliberately independent of the simulator.  It accepts either
one JSON artifact containing both selected cases, or two shards supplied with
repeated ``--input`` arguments.  It validates the paired inputs and rebuilds
all 500-ms, prefix, rolling-window, per-NPU, resource, and admitted-work
statistics before writing any result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    "results/ms_scale_control/"
    "npu32_ssu4_convergence32s_seed42_backing256_warm8_settle500_v1.json"
)
DEFAULT_OUTPUT = Path(
    "results/ms_scale_control/npu32_ssu4_convergence32s_analysis"
)
CASES = ("baseline", "adaptive_t0_i100ms")
LABELS = {"baseline": "Baseline", "adaptive_t0_i100ms": "Adaptive 100 ms"}
COLORS = {"baseline": "#1f77b4", "adaptive_t0_i100ms": "#d62728"}
NUM_NPU = 32
NUM_SSU = 4
N_LAYERS = 16
BATCH_SIZE = 1
MEASUREMENT_MS = 32_000.0
BLOCK_MS = 500.0
BLOCK_COUNT = 64
WARMUP_REQUESTS = 8
SETTLE_MS = 500.0
SSD_CAP_GBPS = 40.0
NPU_LINK_CAP_GBPS = 50.0
PREFIX_SECONDS = (2.0, 3.0, 8.0, 16.0, 32.0)
FIGURE_PREFIX_SECONDS = (3.0, 8.0, 16.0, 32.0)
HISTORICAL_8S_START_MS = {
    "baseline": 10136.618340562687,
    "adaptive_t0_i100ms": 10906.062311812053,
}
HISTORICAL_8S_NPU_UTILIZATION = {
    "baseline": 0.7863877341361012,
    "adaptive_t0_i100ms": 0.7818647786723211,
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
PAIR_FIELDS = (
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
BOUNDARY_SEMANTICS = (
    "read-only left-limit snapshot before workload events at the same time"
)
CONTROL_WINDOW = "half-open [measurement_start_ms, measurement_end_ms)"
TOL = 1e-8


class AnalysisError(ValueError):
    """Raised when an input cannot support the requested conclusion."""


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


def _close(left: float, right: float, *, tolerance: float = TOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=tolerance)


def _vector(
    value: object,
    size: int,
    context: str,
    *,
    integer: bool = False,
    nonnegative: bool = True,
) -> list[float] | list[int]:
    _require(isinstance(value, list), f"{context}: expected a list")
    _require(len(value) == size, f"{context}: expected {size} values")
    converter = _integer if integer else _number
    result = [converter(item, f"{context}[{index}]") for index, item in enumerate(value)]
    if nonnegative:
        _require(min(result, default=0.0) >= -TOL, f"{context}: negative value")
    return result


def _vectors_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(_close(a, b) for a, b in zip(left, right))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publication_path(path: Path) -> str:
    """Return a reproducible path without publishing the local workspace prefix."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path}: top level must be an object")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _case_spec(spec: Mapping[str, object], case: str) -> dict:
    cases = spec.get("cases")
    _require(isinstance(cases, list), "experiment_spec.cases missing")
    matches = [item for item in cases if isinstance(item, dict) and item.get("name") == case]
    _require(len(matches) == 1, f"experiment_spec: expected one definition for {case}")
    return matches[0]


def _validate_experiment_spec(spec: object, label: str) -> None:
    _require(isinstance(spec, dict), f"{label}: experiment_spec missing")
    _require(_integer(spec.get("num_npu"), f"{label}.num_npu") == NUM_NPU, f"{label}: expected 32 NPUs")
    _require(_integer(spec.get("n_layers"), f"{label}.n_layers") == N_LAYERS, f"{label}: expected 16 layers")
    _require(_integer(spec.get("batch_size"), f"{label}.batch_size") == BATCH_SIZE, f"{label}: expected batch size 1")
    steady = spec.get("steady_state")
    _require(isinstance(steady, dict), f"{label}: steady_state spec missing")
    expected_numbers = {
        "warmup_requests_per_npu": WARMUP_REQUESTS,
        "settle_ms": SETTLE_MS,
        "measurement_ms": MEASUREMENT_MS,
        "block_ms": BLOCK_MS,
        "slo_alpha": 2.0,
    }
    for field, expected in expected_numbers.items():
        actual = _number(steady.get(field), f"{label}.steady_state.{field}")
        _require(_close(actual, expected), f"{label}: unexpected {field}: {actual}")
    workload = spec.get("workload")
    _require(isinstance(workload, dict), f"{label}: workload spec missing")
    backing = _integer(workload.get("requests_per_npu"), f"{label}.workload.requests_per_npu")
    _require(backing >= 256, f"{label}: expected at least 256 backing requests/NPU")
    baseline = _case_spec(spec, "baseline")
    adaptive = _case_spec(spec, "adaptive_t0_i100ms")
    for field, expected in {
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 0.0,
    }.items():
        _require(_close(_number(baseline.get(field), f"baseline.{field}"), expected), f"baseline: unexpected {field}")
    _require(baseline.get("kind") == "baseline", "baseline: unexpected kind")
    for field, expected in {
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
        "min_interval_ms": 100.0,
    }.items():
        _require(_close(_number(adaptive.get(field), f"adaptive.{field}"), expected), f"adaptive: unexpected {field}")
    _require(adaptive.get("kind") == "adaptive", "adaptive: unexpected kind")
    adaptive_spec = spec.get("adaptive")
    _require(isinstance(adaptive_spec, dict), f"{label}: adaptive spec missing")
    _require(_close(_number(adaptive_spec.get("ssd_cap_gbps"), "ssd_cap_gbps"), SSD_CAP_GBPS), "unexpected SSD capacity")
    _require(_close(_number(adaptive_spec.get("npu_cap_gbps"), "npu_cap_gbps"), NPU_LINK_CAP_GBPS), "unexpected NPU-link capacity")


def _validate_request_rows(summary: dict, blocks: list[dict], case: str) -> dict:
    rows = summary.get("request_rows")
    _require(isinstance(rows, list) and rows, f"{case}: request_rows missing")
    start = _number(summary["measurement_start_ms"], f"{case}.measurement_start_ms")
    end = _number(summary["measurement_end_ms"], f"{case}.measurement_end_ms")
    ids: set[int] = set()
    per_npu = [0] * NUM_NPU
    ttfts: list[float] = []
    slo_by_npu: list[list[bool]] = [[] for _ in range(NUM_NPU)]
    by_block: list[list[dict]] = [[] for _ in range(BLOCK_COUNT)]
    for index, row in enumerate(rows):
        context = f"{case}.request_rows[{index}]"
        _require(isinstance(row, dict), f"{context}: expected an object")
        request_id = _integer(row.get("request_id"), f"{context}.request_id")
        _require(request_id not in ids, f"{case}: duplicate request_id {request_id}")
        ids.add(request_id)
        npu = _integer(row.get("npu_id"), f"{context}.npu_id")
        _require(0 <= npu < NUM_NPU, f"{context}: NPU out of range")
        admission = _number(row.get("admission_time_ms"), f"{context}.admission")
        completion = _number(row.get("completion_time_ms"), f"{context}.completion")
        ttft = _number(row.get("ttft_ms"), f"{context}.ttft")
        ideal = _number(row.get("ideal_ttft_ms"), f"{context}.ideal")
        raw_demand = _number(row.get("raw_demand_gbps"), f"{context}.raw_demand")
        _require(start - TOL <= admission < end - TOL, f"{context}: admission outside half-open window")
        _require(completion >= admission, f"{context}: completion before admission")
        _require(_close(ttft, completion - admission), f"{context}: TTFT mismatch")
        _require(ideal > 0.0 and raw_demand > 0.0, f"{context}: invalid profile work")
        met = ttft <= 2.0 * ideal + 1e-9
        _require(row.get("slo_met") is met, f"{context}: alpha=2 SLO flag mismatch")
        block = min(BLOCK_COUNT - 1, int((admission - start) // BLOCK_MS))
        by_block[block].append(row)
        per_npu[npu] += 1
        slo_by_npu[npu].append(met)
        ttfts.append(ttft)
    _require(len(rows) == _integer(summary.get("measurement_request_count"), f"{case}.request_count"), f"{case}: request count mismatch")
    _require(per_npu == _vector(summary.get("request_counts_by_npu"), NUM_NPU, f"{case}.request_counts_by_npu", integer=True), f"{case}: per-NPU request counts mismatch")
    weighted = statistics.fmean([flag for values in slo_by_npu for flag in values])
    equal_npu = statistics.fmean(statistics.fmean(values) for values in slo_by_npu)
    _require(_close(weighted, _number(summary.get("request_weighted_slo_attainment"), f"{case}.weighted_slo")), f"{case}: weighted SLO mismatch")
    _require(_close(equal_npu, _number(summary.get("ttft_slo_attainment"), f"{case}.equal_npu_slo")), f"{case}: equal-NPU SLO mismatch")
    _require(_close(statistics.fmean(ttfts), _number(summary.get("mean_ttft_ms"), f"{case}.mean_ttft")), f"{case}: mean TTFT mismatch")
    _require(_close(float(np.percentile(ttfts, 99)), _number(summary.get("p99_ttft_ms"), f"{case}.p99_ttft")), f"{case}: p99 TTFT mismatch")
    for index, (block, admitted) in enumerate(zip(blocks, by_block)):
        _require(_integer(block.get("request_count"), f"{case}.block[{index}].request_count") == len(admitted), f"{case}: block {index} request count mismatch")
        expected = statistics.fmean(bool(row["slo_met"]) for row in admitted) if admitted else None
        actual = block.get("request_weighted_slo_attainment")
        _require((expected is None and actual is None) or (expected is not None and _close(expected, _number(actual, f"{case}.block[{index}].slo"))), f"{case}: block {index} SLO mismatch")
    return {"rows": rows, "by_block": by_block}


def _validate_summary(row: dict, case: str) -> dict:
    summary = row.get("steady_summary")
    _require(isinstance(summary, dict), f"{case}: steady_summary missing")
    _require(_integer(summary.get("schema_version"), f"{case}.summary.schema") == 2, f"{case}: rich summary schema v2 required")
    for field, expected in {
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "warmup_requests_per_npu": WARMUP_REQUESTS,
    }.items():
        _require(_integer(summary.get(field), f"{case}.{field}") == expected, f"{case}: unexpected {field}")
    for field, expected in {
        "settle_ms": SETTLE_MS,
        "measurement_duration_ms": MEASUREMENT_MS,
        "slo_alpha": 2.0,
        "pressure_ttl_ms": 0.0,
        "cir_write_threshold_gbps": 0.0,
    }.items():
        _require(_close(_number(summary.get(field), f"{case}.{field}"), expected), f"{case}: unexpected {field}")
    interval = summary.get("control_min_interval_ms")
    if case == "baseline":
        _require(interval is None, f"{case}: baseline unexpectedly has a controller interval")
    else:
        _require(_close(_number(interval, f"{case}.control_min_interval_ms"), 100.0), f"{case}: controller interval mismatch")
    start = _number(summary.get("measurement_start_ms"), f"{case}.start")
    end = _number(summary.get("measurement_end_ms"), f"{case}.end")
    _require(_close(end - start, MEASUREMENT_MS), f"{case}: measurement endpoints mismatch")
    _require(summary.get("stationarity_boundary_semantics") == BOUNDARY_SEMANTICS, f"{case}: unexpected boundary semantics")
    _require(summary.get("measurement_control_counter_window") == CONTROL_WINDOW, f"{case}: unexpected control counter window")
    invariants = summary.get("invariants")
    _require(isinstance(invariants, dict), f"{case}: invariants missing")
    _require(set(invariants) == EXPECTED_INVARIANTS, f"{case}: invariant set mismatch: missing={sorted(EXPECTED_INVARIANTS - set(invariants))}, extra={sorted(set(invariants) - EXPECTED_INVARIANTS)}")
    failed = sorted(name for name, value in invariants.items() if value is not True)
    _require(not failed, f"{case}: failed simulator invariants: {failed}")

    blocks = summary.get("measurement_blocks")
    _require(isinstance(blocks, list) and len(blocks) == BLOCK_COUNT, f"{case}: expected 64 measurement blocks")
    previous: dict | None = None
    sums = {
        "compute_ms_by_npu": [0.0] * NUM_NPU,
        "ssd_busy_ms_by_ssu": [0.0] * NUM_SSU,
        "ssd_served_gb_by_ssu": [0.0] * NUM_SSU,
        "npu_link_busy_ms_by_npu": [0.0] * NUM_NPU,
        "npu_link_served_gb_by_npu": [0.0] * NUM_NPU,
    }
    for index, block in enumerate(blocks):
        context = f"{case}.measurement_blocks[{index}]"
        _require(isinstance(block, dict), f"{context}: expected an object")
        _require(_integer(block.get("block"), f"{context}.block") == index, f"{context}: index mismatch")
        block_start = _number(block.get("start_ms"), f"{context}.start")
        block_end = _number(block.get("end_ms"), f"{context}.end")
        duration = _number(block.get("duration_ms"), f"{context}.duration")
        _require(_close(block_start, start + index * BLOCK_MS), f"{context}: start mismatch")
        _require(_close(block_end, start + (index + 1) * BLOCK_MS), f"{context}: end mismatch")
        _require(_close(duration, BLOCK_MS), f"{context}: duration mismatch")
        vectors: dict[str, list[float] | list[int]] = {}
        for field, size, integer in (
            ("compute_ms_by_npu", NUM_NPU, False),
            ("npu_utilizations", NUM_NPU, False),
            ("ssd_busy_ms_by_ssu", NUM_SSU, False),
            ("ssd_served_gb_by_ssu", NUM_SSU, False),
            ("ssd_utilizations", NUM_SSU, False),
            ("ssd_outstanding_blocks_at_start", NUM_SSU, True),
            ("ssd_outstanding_blocks_at_end", NUM_SSU, True),
            ("ssd_outstanding_blocks_delta", NUM_SSU, True),
            ("ssd_outstanding_gb_at_start", NUM_SSU, False),
            ("ssd_outstanding_gb_at_end", NUM_SSU, False),
            ("ssd_outstanding_gb_delta", NUM_SSU, False),
            ("npu_link_busy_ms_by_npu", NUM_NPU, False),
            ("npu_link_served_gb_by_npu", NUM_NPU, False),
            ("npu_link_utilizations", NUM_NPU, False),
            ("npu_link_outstanding_blocks_at_start", NUM_NPU, True),
            ("npu_link_outstanding_blocks_at_end", NUM_NPU, True),
            ("npu_link_outstanding_blocks_delta", NUM_NPU, True),
            ("npu_link_outstanding_gb_at_start", NUM_NPU, False),
            ("npu_link_outstanding_gb_at_end", NUM_NPU, False),
            ("npu_link_outstanding_gb_delta", NUM_NPU, False),
        ):
            vectors[field] = _vector(block.get(field), size, f"{context}.{field}", integer=integer, nonnegative="delta" not in field)
        for value in vectors["compute_ms_by_npu"] + vectors["ssd_busy_ms_by_ssu"] + vectors["npu_link_busy_ms_by_npu"]:
            _require(value <= duration + TOL, f"{context}: resource overlap exceeds block")
        _require(_vectors_close(vectors["npu_utilizations"], [value / duration for value in vectors["compute_ms_by_npu"]]), f"{context}: NPU utilization vector mismatch")
        _require(_close(_number(block.get("npu_utilization"), f"{context}.npu_utilization"), sum(vectors["compute_ms_by_npu"]) / (NUM_NPU * duration)), f"{context}: fleet NPU utilization mismatch")
        _require(_vectors_close(vectors["ssd_utilizations"], [value / duration for value in vectors["ssd_busy_ms_by_ssu"]]), f"{context}: SSD utilization vector mismatch")
        _require(_close(_number(block.get("ssd_mean_utilization"), f"{context}.ssd_mean"), statistics.fmean(vectors["ssd_utilizations"])), f"{context}: SSD mean mismatch")
        _require(_vectors_close(vectors["ssd_served_gb_by_ssu"], [value * SSD_CAP_GBPS / 1000.0 for value in vectors["ssd_busy_ms_by_ssu"]]), f"{context}: SSD service/capacity mismatch")
        _require(_vectors_close(vectors["npu_link_utilizations"], [value / duration for value in vectors["npu_link_busy_ms_by_npu"]]), f"{context}: link utilization vector mismatch")
        _require(_close(_number(block.get("npu_link_mean_utilization"), f"{context}.link_mean"), statistics.fmean(vectors["npu_link_utilizations"])), f"{context}: link mean mismatch")
        _require(_vectors_close(vectors["npu_link_served_gb_by_npu"], [value * NPU_LINK_CAP_GBPS / 1000.0 for value in vectors["npu_link_busy_ms_by_npu"]]), f"{context}: link service/capacity mismatch")
        for prefix in ("ssd", "npu_link"):
            for unit in ("blocks", "gb"):
                before = vectors[f"{prefix}_outstanding_{unit}_at_start"]
                after = vectors[f"{prefix}_outstanding_{unit}_at_end"]
                delta = vectors[f"{prefix}_outstanding_{unit}_delta"]
                _require(_vectors_close(delta, [b - a for a, b in zip(before, after)]), f"{context}: {prefix} {unit} queue delta mismatch")
                if previous is not None:
                    prior = previous[f"{prefix}_outstanding_{unit}_at_end"]
                    _require(_vectors_close(before, prior), f"{context}: {prefix} {unit} queue boundary discontinuity")
        for field in sums:
            sums[field] = [a + b for a, b in zip(sums[field], vectors[field])]
        previous = vectors

    request_info = _validate_request_rows(summary, blocks, case)
    summary_pairs = {
        "compute_ms_by_npu": "compute_ms_by_npu",
        "ssd_busy_ms_by_ssu": "measurement_ssd_busy_ms_by_ssu",
        "ssd_served_gb_by_ssu": "measurement_ssd_served_gb_by_ssu",
        "npu_link_busy_ms_by_npu": "measurement_npu_link_busy_ms_by_npu",
    }
    for accumulated, summary_field in summary_pairs.items():
        size = NUM_SSU if "ssu" in accumulated else NUM_NPU
        actual = _vector(summary.get(summary_field), size, f"{case}.{summary_field}")
        _require(_vectors_close(sums[accumulated], actual), f"{case}: block sum != {summary_field}")
    link_matrix = summary.get("measurement_npu_ssu_link_served_gb")
    _require(isinstance(link_matrix, list) and len(link_matrix) == NUM_NPU, f"{case}: link attribution matrix shape")
    link_totals = [sum(_vector(values, NUM_SSU, f"{case}.link_matrix[{npu}]")) for npu, values in enumerate(link_matrix)]
    _require(_vectors_close(link_totals, sums["npu_link_served_gb_by_npu"]), f"{case}: link attribution mismatch")
    ssd_matrix = summary.get("measurement_npu_ssu_ssd_served_gb")
    _require(isinstance(ssd_matrix, list) and len(ssd_matrix) == NUM_NPU, f"{case}: SSD attribution matrix shape")
    ssd_rows = [_vector(values, NUM_SSU, f"{case}.ssd_matrix[{npu}]") for npu, values in enumerate(ssd_matrix)]
    ssd_totals = [sum(values[ssu] for values in ssd_rows) for ssu in range(NUM_SSU)]
    _require(_vectors_close(ssd_totals, sums["ssd_served_gb_by_ssu"]), f"{case}: SSD attribution mismatch")
    whole_npu = sum(sums["compute_ms_by_npu"]) / (NUM_NPU * MEASUREMENT_MS)
    whole_ssd = sum(sums["ssd_busy_ms_by_ssu"]) / (NUM_SSU * MEASUREMENT_MS)
    whole_link = sum(sums["npu_link_busy_ms_by_npu"]) / (NUM_NPU * MEASUREMENT_MS)
    _require(_close(whole_npu, _number(summary.get("mean_npu_utilization"), f"{case}.mean_npu")), f"{case}: whole NPU utilization mismatch")
    _require(_close(whole_ssd, _number(summary.get("measurement_ssd_mean_utilization"), f"{case}.mean_ssd")), f"{case}: whole SSD utilization mismatch")
    _require(_close(whole_link, _number(summary.get("measurement_npu_link_mean_utilization"), f"{case}.mean_link")), f"{case}: whole link utilization mismatch")
    _require(_vectors_close(_vector(summary.get("measurement_ssd_outstanding_blocks_at_start"), NUM_SSU, f"{case}.ssd_q_start", integer=True), blocks[0]["ssd_outstanding_blocks_at_start"]), f"{case}: SSD initial queue mismatch")
    _require(_vectors_close(_vector(summary.get("measurement_ssd_outstanding_blocks_at_end"), NUM_SSU, f"{case}.ssd_q_end", integer=True), blocks[-1]["ssd_outstanding_blocks_at_end"]), f"{case}: SSD final queue mismatch")

    boundaries = summary.get("measurement_stationarity_boundaries")
    _require(_integer(summary.get("measurement_stationarity_boundary_count"), f"{case}.boundary_count") == BLOCK_COUNT + 1, f"{case}: boundary count field mismatch")
    _require(isinstance(boundaries, list) and len(boundaries) == BLOCK_COUNT + 1, f"{case}: expected 65 stationarity boundaries")
    for index, boundary in enumerate(boundaries):
        context = f"{case}.boundary[{index}]"
        _require(isinstance(boundary, dict), f"{context}: expected object")
        _require(_integer(boundary.get("boundary"), f"{context}.boundary") == index, f"{context}: index mismatch")
        _require(_close(_number(boundary.get("time_ms"), f"{context}.time"), start + index * BLOCK_MS), f"{context}: time mismatch")
        _vector(boundary.get("ssd_outstanding_gb_by_ssu"), NUM_SSU, f"{context}.ssd_queue_gb")
        _vector(boundary.get("npu_link_outstanding_gb_by_npu"), NUM_NPU, f"{context}.link_queue_gb")

    return {
        "summary": summary,
        "blocks": blocks,
        "requests": request_info["rows"],
        "requests_by_block": request_info["by_block"],
        "whole_npu_utilization": whole_npu,
        "whole_ssd_utilization": whole_ssd,
        "whole_link_utilization": whole_link,
    }


def _load_and_validate(
    paths: Sequence[Path],
    *,
    enforce_historical_bridge: bool = True,
) -> tuple[dict[str, dict], dict]:
    _require(paths, "at least one --input is required")
    payloads: list[tuple[Path, dict]] = []
    selected: dict[str, tuple[Path, dict]] = {}
    source_fingerprints: set[str] = set()
    config_fingerprints: set[str] = set()
    specs: set[str] = set()
    manifests: set[str] = set()
    schedules: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        payload = _load_json(path)
        label = path.as_posix()
        _require(payload.get("source_stable_during_run") is True, f"{label}: source was not stable")
        _require(payload.get("config_stable_during_run") is True, f"{label}: config was not stable")
        _require(payload.get("selected_complete") is True, f"{label}: selected run is incomplete")
        source = payload.get("source_fingerprint")
        config = payload.get("config_fingerprint")
        _require(isinstance(source, str) and len(source) == 64, f"{label}: invalid source fingerprint")
        _require(isinstance(config, str) and len(config) == 64, f"{label}: invalid config fingerprint")
        _require(payload.get("ending_source_fingerprint") == source, f"{label}: ending source changed")
        _require(payload.get("ending_config_fingerprint") == config, f"{label}: ending config changed")
        _validate_experiment_spec(payload.get("experiment_spec"), label)
        audit = payload.get("pairing_audit")
        _require(isinstance(audit, dict) and isinstance(audit.get("4"), dict), f"{label}: SSU4 pairing audit missing")
        _require(audit["4"].get("has_rows") is True and audit["4"].get("all_available_rows_paired") is True, f"{label}: SSU4 pairing audit failed")
        source_fingerprints.add(source)
        config_fingerprints.add(config)
        specs.add(_canonical(payload["experiment_spec"]))
        manifests.add(_canonical(payload.get("source_manifest")))
        schedules.add(_canonical(payload.get("schedule_metadata")))
        results = payload.get("results")
        _require(isinstance(results, list), f"{label}: results missing")
        for row in results:
            if not isinstance(row, dict) or row.get("case") not in CASES or row.get("num_ssu") != NUM_SSU:
                continue
            case = str(row["case"])
            _require(row.get("status") == "ok", f"{label}: {case} status is not ok")
            _require(case not in selected, f"duplicate selected row for {case}")
            _require(row.get("source_fingerprint") == source, f"{case}: row/source fingerprint mismatch")
            _require(row.get("config_fingerprint") == config, f"{case}: row/config fingerprint mismatch")
            selected[case] = (path, row)
        payloads.append((path, payload))
    _require(set(selected) == set(CASES), f"missing paired cases: {sorted(set(CASES) - set(selected))}")
    _require(len(source_fingerprints) == 1, "paired shards use different source fingerprints")
    _require(len(config_fingerprints) == 1, "paired shards use different config fingerprints")
    _require(len(specs) == 1 and len(manifests) == 1 and len(schedules) == 1, "paired shard metadata differs")
    paired_values: dict[str, set[str]] = {field: set() for field in PAIR_FIELDS}
    prefix_materialized: set[str] = set()
    summary_inputs: set[str] = set()
    cases: dict[str, dict] = {}
    for case in CASES:
        path, row = selected[case]
        inputs = row.get("input_fingerprints")
        _require(isinstance(inputs, dict), f"{case}: input_fingerprints missing")
        for field in PAIR_FIELDS:
            value = inputs.get(field)
            _require(isinstance(value, str) and len(value) == 64, f"{case}: invalid paired fingerprint {field}")
            paired_values[field].add(value)
        prefix = row.get("prefix_32_materialized_fingerprints")
        _require(isinstance(prefix, dict), f"{case}: prefix_32 materialized fingerprints missing")
        prefix_materialized.add(_canonical(prefix))
        data = _validate_summary(row, case)
        summary_input = data["summary"].get("input_fingerprint")
        _require(isinstance(summary_input, str) and len(summary_input) == 64, f"{case}: invalid simulator input fingerprint")
        summary_inputs.add(summary_input)
        data.update({"path": path, "row": row})
        cases[case] = data
    mismatched = sorted(field for field, values in paired_values.items() if len(values) != 1)
    _require(not mismatched, f"paired cases have different inputs: {mismatched}")
    _require(len(prefix_materialized) == 1, "paired prefix-32 materialization differs")
    _require(len(summary_inputs) == 1, "paired simulator input fingerprint differs")
    bridge: dict[str, dict[str, float]] = {}
    for case in CASES:
        summary = cases[case]["summary"]
        actual_start = float(summary["measurement_start_ms"])
        first_eight = cases[case]["blocks"][:16]
        actual_utilization = sum(
            sum(float(value) for value in block["compute_ms_by_npu"])
            for block in first_eight
        ) / (NUM_NPU * 8000.0)
        if enforce_historical_bridge:
            _require(
                _close(actual_start, HISTORICAL_8S_START_MS[case], tolerance=1e-6),
                f"{case}: 32-s run does not bridge historical measurement start",
            )
            _require(
                _close(
                    actual_utilization,
                    HISTORICAL_8S_NPU_UTILIZATION[case],
                    tolerance=1e-12,
                ),
                f"{case}: first 8 s does not reproduce the historical formal result",
            )
        bridge[case] = {
            "measurement_start_ms": actual_start,
            "first_8s_npu_utilization": actual_utilization,
            "expected_first_8s_npu_utilization": HISTORICAL_8S_NPU_UTILIZATION[case],
        }
    validation = {
        "passed": True,
        "analysis": "npu32_ssu4_convergence32s_v1",
        "input_files": [
            {"path": _publication_path(path), "sha256": _sha256(path)}
            for path, _ in payloads
        ],
        "source_fingerprint": next(iter(source_fingerprints)),
        "config_fingerprint": next(iter(config_fingerprints)),
        "paired_input_fingerprints": {field: next(iter(values)) for field, values in paired_values.items()},
        "checks": {
            "npu_32": True,
            "ssu_4": True,
            "layers_16": True,
            "measurement_32s": True,
            "blocks_64_x_500ms": True,
            "source_stable_and_identical": True,
            "config_stable_and_identical": True,
            "pairing_audit_and_fingerprints": True,
            "all_29_simulator_invariants_per_case": True,
            "independent_resource_reconstruction": True,
            "request_and_profile_reconstruction": True,
            "historical_8s_semantic_bridge": enforce_historical_bridge,
        },
        "historical_8s_bridge": bridge,
        "unsupported": {
            "active_remaining_compute_inventory_at_boundaries": (
                "schema v2 records cumulative compute and IO queues, but not active "
                "request/layer remaining compute at each boundary; only its window "
                "change is inferred from C - admitted_work, not independently observed"
            )
        },
    }
    return cases, validation


def _raw_demand_bin(value: float) -> str:
    if value <= 10.0:
        return "le_10"
    if value <= 20.0:
        return "gt_10_le_20"
    if value <= 40.0:
        return "gt_20_le_40"
    if value <= 50.0:
        return "gt_40_le_50"
    if value <= 80.0:
        return "gt_50_le_80"
    return "gt_80"


def _admitted_work(rows: Sequence[dict]) -> dict[str, float | int]:
    compute_ms = sum(float(row["ideal_ttft_ms"]) for row in rows)
    io_gb = sum(float(row["raw_demand_gbps"]) * float(row["ideal_ttft_ms"]) / 1000.0 for row in rows)
    demands = [float(row["raw_demand_gbps"]) for row in rows]
    return {
        "admitted_request_count": len(rows),
        "admitted_compute_ms": compute_ms,
        "admitted_io_gb": io_gb,
        "admitted_compute_ms_per_io_gb": compute_ms / io_gb if io_gb else math.nan,
        "admitted_io_per_compute_gbps": io_gb / (compute_ms / 1000.0) if compute_ms else math.nan,
        "admitted_mean_raw_demand_gbps": statistics.fmean(demands) if demands else math.nan,
    }


def _append_profile_groups(
    output: list[dict],
    *,
    case: str,
    window_kind: str,
    start_s: float,
    end_s: float,
    rows: Sequence[dict],
) -> None:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for request in rows:
        groups[("overall", "all")].append(request)
        groups[("category", str(request["category"]))].append(request)
        groups[("raw_demand_bin", _raw_demand_bin(float(request["raw_demand_gbps"])))].append(request)
        groups[("profile", str(request["profile_id"]))].append(request)
    for (dimension, group), values in sorted(groups.items()):
        work = _admitted_work(values)
        ttfts = [float(item["ttft_ms"]) for item in values]
        output.append(
            {
                "case": case,
                "label": LABELS[case],
                "window_kind": window_kind,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "dimension": dimension,
                "group": group,
                **work,
                "mean_ttft_ms": statistics.fmean(ttfts),
                "p99_ttft_ms": float(np.percentile(ttfts, 99)),
                "slo_alpha1p5_pct": 100.0 * statistics.fmean(float(item["ttft_ms"]) <= 1.5 * float(item["ideal_ttft_ms"]) + 1e-9 for item in values),
                "slo_alpha2_pct": 100.0 * statistics.fmean(float(item["ttft_ms"]) <= 2.0 * float(item["ideal_ttft_ms"]) + 1e-9 for item in values),
            }
        )


def _matched_request_rows(cases: dict[str, dict]) -> list[dict]:
    """Compare identical request IDs while retaining the different queue contexts."""
    relative: dict[str, dict[int, tuple[float, dict]]] = {}
    for case in CASES:
        start = float(cases[case]["summary"]["measurement_start_ms"])
        relative[case] = {
            int(row["request_id"]): ((float(row["admission_time_ms"]) - start) / 1000.0, row)
            for row in cases[case]["requests"]
        }
    output: list[dict] = []
    for prefix_s in PREFIX_SECONDS:
        eligible = {
            case: {
                request_id: row
                for request_id, (admission_s, row) in relative[case].items()
                if admission_s < prefix_s - TOL
            }
            for case in CASES
        }
        matched = sorted(set(eligible["baseline"]) & set(eligible["adaptive_t0_i100ms"]))
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for request_id in matched:
            baseline = eligible["baseline"][request_id]
            adaptive = eligible["adaptive_t0_i100ms"][request_id]
            _require(baseline.get("profile_id") == adaptive.get("profile_id"), f"matched request {request_id}: profile mismatch")
            _require(_close(float(baseline["ideal_ttft_ms"]), float(adaptive["ideal_ttft_ms"])), f"matched request {request_id}: ideal compute mismatch")
            groups[("overall", "all")].append(request_id)
            groups[("category", str(baseline["category"]))].append(request_id)
            groups[("raw_demand_bin", _raw_demand_bin(float(baseline["raw_demand_gbps"])))].append(request_id)
        for (dimension, group), request_ids in sorted(groups.items()):
            baseline_stalls = [float(eligible["baseline"][request_id]["ttft_ms"]) - float(eligible["baseline"][request_id]["ideal_ttft_ms"]) for request_id in request_ids]
            adaptive_stalls = [float(eligible["adaptive_t0_i100ms"][request_id]["ttft_ms"]) - float(eligible["adaptive_t0_i100ms"][request_id]["ideal_ttft_ms"]) for request_id in request_ids]
            paired_deltas = [adaptive - baseline for baseline, adaptive in zip(baseline_stalls, adaptive_stalls)]
            output.append(
                {
                    "prefix_s": prefix_s,
                    "dimension": dimension,
                    "group": group,
                    "matched_request_count": len(request_ids),
                    "baseline_mean_stall_ms": statistics.fmean(baseline_stalls),
                    "adaptive_mean_stall_ms": statistics.fmean(adaptive_stalls),
                    "adaptive_minus_baseline_mean_stall_ms": statistics.fmean(paired_deltas),
                    "adaptive_minus_baseline_median_stall_ms": statistics.median(paired_deltas),
                    "adaptive_minus_baseline_p99_stall_ms": float(np.percentile(paired_deltas, 99)),
                    "baseline_admitted_request_count": len(eligible["baseline"]),
                    "adaptive_admitted_request_count": len(eligible["adaptive_t0_i100ms"]),
                }
            )
    return output


def _build_tables(cases: dict[str, dict]) -> dict[str, list[dict]]:
    blocks_rows: list[dict] = []
    cumulative_rows: list[dict] = []
    rolling: dict[int, list[dict]] = {8: [], 16: []}
    resource_rows: list[dict] = []
    prefix_rows_by_case: dict[str, dict[float, dict]] = defaultdict(dict)
    per_npu_rows: list[dict] = []
    profile_rows: list[dict] = []
    disjoint_rows: list[dict] = []
    for case in CASES:
        data = cases[case]
        blocks = data["blocks"]
        request_blocks = data["requests_by_block"]
        cumulative_compute = np.zeros(NUM_NPU, dtype=float)
        cumulative_ssd_busy = np.zeros(NUM_SSU, dtype=float)
        cumulative_link_busy = np.zeros(NUM_NPU, dtype=float)
        cumulative_ssd_gb = 0.0
        cumulative_link_gb = 0.0
        cumulative_requests: list[dict] = []
        for index, (block, admitted) in enumerate(zip(blocks, request_blocks)):
            prefix_s = (index + 1) * BLOCK_MS / 1000.0
            compute = np.asarray(block["compute_ms_by_npu"], dtype=float)
            ssd_busy = np.asarray(block["ssd_busy_ms_by_ssu"], dtype=float)
            link_busy = np.asarray(block["npu_link_busy_ms_by_npu"], dtype=float)
            ssd_gb = float(sum(block["ssd_served_gb_by_ssu"]))
            link_gb = float(sum(block["npu_link_served_gb_by_npu"]))
            cumulative_compute += compute
            cumulative_ssd_busy += ssd_busy
            cumulative_link_busy += link_busy
            cumulative_ssd_gb += ssd_gb
            cumulative_link_gb += link_gb
            cumulative_requests.extend(admitted)
            block_work = _admitted_work(admitted)
            cumulative_work = _admitted_work(cumulative_requests)
            common = {
                "case": case,
                "label": LABELS[case],
                "block": index,
                "start_s": index * BLOCK_MS / 1000.0,
                "end_s": prefix_s,
                "duration_ms": BLOCK_MS,
                "npu_utilization_pct": 100.0 * float(block["npu_utilization"]),
                "ssd_mean_utilization_pct": 100.0 * float(block["ssd_mean_utilization"]),
                "npu_link_mean_utilization_pct": 100.0 * float(block["npu_link_mean_utilization"]),
                "compute_busy_npu_ms": float(compute.sum()),
                "ssd_busy_ssu_ms": float(ssd_busy.sum()),
                "ssd_served_gb": ssd_gb,
                "npu_link_busy_npu_ms": float(link_busy.sum()),
                "npu_link_served_gb": link_gb,
                **block_work,
                "inferred_compute_inventory_drop_ms": float(compute.sum()) - float(block_work["admitted_compute_ms"]),
                "request_weighted_slo_alpha2_pct": (
                    100.0 * float(block["request_weighted_slo_attainment"])
                    if block["request_weighted_slo_attainment"] is not None
                    else math.nan
                ),
            }
            blocks_rows.append(common)
            resource_rows.append(
                {
                    **common,
                    "ssd_outstanding_blocks_start": sum(block["ssd_outstanding_blocks_at_start"]),
                    "ssd_outstanding_blocks_end": sum(block["ssd_outstanding_blocks_at_end"]),
                    "ssd_outstanding_blocks_delta": sum(block["ssd_outstanding_blocks_delta"]),
                    "ssd_outstanding_gb_start": sum(block["ssd_outstanding_gb_at_start"]),
                    "ssd_outstanding_gb_end": sum(block["ssd_outstanding_gb_at_end"]),
                    "ssd_outstanding_gb_delta": sum(block["ssd_outstanding_gb_delta"]),
                    "npu_link_outstanding_blocks_start": sum(block["npu_link_outstanding_blocks_at_start"]),
                    "npu_link_outstanding_blocks_end": sum(block["npu_link_outstanding_blocks_at_end"]),
                    "npu_link_outstanding_blocks_delta": sum(block["npu_link_outstanding_blocks_delta"]),
                    "npu_link_outstanding_gb_start": sum(block["npu_link_outstanding_gb_at_start"]),
                    "npu_link_outstanding_gb_end": sum(block["npu_link_outstanding_gb_at_end"]),
                    "npu_link_outstanding_gb_delta": sum(block["npu_link_outstanding_gb_delta"]),
                    "active_remaining_compute_inventory_supported": False,
                    "active_remaining_compute_ms": None,
                }
            )
            cumulative = {
                "case": case,
                "label": LABELS[case],
                "prefix_s": prefix_s,
                "blocks": index + 1,
                "npu_utilization_pct": 100.0 * float(cumulative_compute.sum()) / (NUM_NPU * prefix_s * 1000.0),
                "ssd_mean_utilization_pct": 100.0 * float(cumulative_ssd_busy.sum()) / (NUM_SSU * prefix_s * 1000.0),
                "npu_link_mean_utilization_pct": 100.0 * float(cumulative_link_busy.sum()) / (NUM_NPU * prefix_s * 1000.0),
                "compute_busy_npu_ms": float(cumulative_compute.sum()),
                "ssd_served_gb": cumulative_ssd_gb,
                "npu_link_served_gb": cumulative_link_gb,
                **cumulative_work,
                "inferred_compute_inventory_drop_ms": float(cumulative_compute.sum()) - float(cumulative_work["admitted_compute_ms"]),
                "ssd_outstanding_blocks_end": sum(block["ssd_outstanding_blocks_at_end"]),
                "ssd_outstanding_gb_end": sum(block["ssd_outstanding_gb_at_end"]),
                "npu_link_outstanding_blocks_end": sum(block["npu_link_outstanding_blocks_at_end"]),
                "npu_link_outstanding_gb_end": sum(block["npu_link_outstanding_gb_at_end"]),
            }
            for alpha in (1.5, 2.0):
                hits = [float(row["ttft_ms"]) <= alpha * float(row["ideal_ttft_ms"]) + 1e-9 for row in cumulative_requests]
                cumulative[f"request_weighted_slo_alpha{str(alpha).replace('.', 'p')}_pct"] = 100.0 * statistics.fmean(hits) if hits else math.nan
            cumulative_rows.append(cumulative)
            _append_profile_groups(
                profile_rows,
                case=case,
                window_kind="chunk_500ms",
                start_s=prefix_s - BLOCK_MS / 1000.0,
                end_s=prefix_s,
                rows=admitted,
            )
            if prefix_s in PREFIX_SECONDS:
                prefix_rows_by_case[case][prefix_s] = cumulative
                for npu in range(NUM_NPU):
                    per_npu_rows.append(
                        {
                            "case": case,
                            "label": LABELS[case],
                            "prefix_s": prefix_s,
                            "npu_id": npu,
                            "compute_ms": float(cumulative_compute[npu]),
                            "utilization_pct": 100.0 * float(cumulative_compute[npu]) / (prefix_s * 1000.0),
                            "idle_ms": prefix_s * 1000.0 - float(cumulative_compute[npu]),
                        }
                    )
            for seconds in rolling:
                window_blocks = int(seconds * 1000.0 / BLOCK_MS)
                if index + 1 < window_blocks:
                    continue
                selected_blocks = blocks[index + 1 - window_blocks : index + 1]
                selected_requests = [row for values in request_blocks[index + 1 - window_blocks : index + 1] for row in values]
                window_compute = sum(sum(item["compute_ms_by_npu"]) for item in selected_blocks)
                window_ssd = sum(sum(item["ssd_busy_ms_by_ssu"]) for item in selected_blocks)
                window_link = sum(sum(item["npu_link_busy_ms_by_npu"]) for item in selected_blocks)
                rolling[seconds].append(
                    {
                        "case": case,
                        "label": LABELS[case],
                        "window_s": float(seconds),
                        "start_s": prefix_s - seconds,
                        "end_s": prefix_s,
                        "npu_utilization_pct": 100.0 * window_compute / (NUM_NPU * seconds * 1000.0),
                        "ssd_mean_utilization_pct": 100.0 * window_ssd / (NUM_SSU * seconds * 1000.0),
                        "npu_link_mean_utilization_pct": 100.0 * window_link / (NUM_NPU * seconds * 1000.0),
                        "compute_busy_npu_ms": window_compute,
                        **_admitted_work(selected_requests),
                        "inferred_compute_inventory_drop_ms": window_compute - float(_admitted_work(selected_requests)["admitted_compute_ms"]),
                    }
                )

        for prefix_s in PREFIX_SECONDS:
            selected_requests = [row for row in data["requests"] if float(row["admission_time_ms"]) < float(data["summary"]["measurement_start_ms"]) + prefix_s * 1000.0 - TOL]
            _append_profile_groups(
                profile_rows,
                case=case,
                window_kind="cumulative_prefix",
                start_s=0.0,
                end_s=prefix_s,
                rows=selected_requests,
            )
        for seconds in (8, 16):
            window_blocks = int(seconds * 1000.0 / BLOCK_MS)
            for first in range(0, BLOCK_COUNT, window_blocks):
                last = first + window_blocks
                selected_blocks = blocks[first:last]
                selected_requests = [
                    request
                    for values in request_blocks[first:last]
                    for request in values
                ]
                compute = sum(sum(item["compute_ms_by_npu"]) for item in selected_blocks)
                ssd_busy = sum(sum(item["ssd_busy_ms_by_ssu"]) for item in selected_blocks)
                link_busy = sum(sum(item["npu_link_busy_ms_by_npu"]) for item in selected_blocks)
                work = _admitted_work(selected_requests)
                disjoint_rows.append(
                    {
                        "case": case,
                        "label": LABELS[case],
                        "window_s": float(seconds),
                        "start_s": first * BLOCK_MS / 1000.0,
                        "end_s": last * BLOCK_MS / 1000.0,
                        "npu_utilization_pct": 100.0 * compute / (NUM_NPU * seconds * 1000.0),
                        "compute_busy_npu_ms": compute,
                        "ssd_mean_utilization_pct": 100.0 * ssd_busy / (NUM_SSU * seconds * 1000.0),
                        "ssd_served_gb": sum(sum(item["ssd_served_gb_by_ssu"]) for item in selected_blocks),
                        "npu_link_mean_utilization_pct": 100.0 * link_busy / (NUM_NPU * seconds * 1000.0),
                        "npu_link_served_gb": sum(sum(item["npu_link_served_gb_by_npu"]) for item in selected_blocks),
                        **work,
                        "inferred_compute_inventory_drop_ms": compute - float(work["admitted_compute_ms"]),
                        "ssd_outstanding_blocks_start": sum(selected_blocks[0]["ssd_outstanding_blocks_at_start"]),
                        "ssd_outstanding_blocks_end": sum(selected_blocks[-1]["ssd_outstanding_blocks_at_end"]),
                        "ssd_outstanding_gb_start": sum(selected_blocks[0]["ssd_outstanding_gb_at_start"]),
                        "ssd_outstanding_gb_end": sum(selected_blocks[-1]["ssd_outstanding_gb_at_end"]),
                        "npu_link_outstanding_blocks_start": sum(selected_blocks[0]["npu_link_outstanding_blocks_at_start"]),
                        "npu_link_outstanding_blocks_end": sum(selected_blocks[-1]["npu_link_outstanding_blocks_at_end"]),
                        "npu_link_outstanding_gb_start": sum(selected_blocks[0]["npu_link_outstanding_gb_at_start"]),
                        "npu_link_outstanding_gb_end": sum(selected_blocks[-1]["npu_link_outstanding_gb_at_end"]),
                    }
                )
                _append_profile_groups(
                    profile_rows,
                    case=case,
                    window_kind=f"disjoint_{seconds}s",
                    start_s=first * BLOCK_MS / 1000.0,
                    end_s=last * BLOCK_MS / 1000.0,
                    rows=selected_requests,
                )

    per_npu_wide: list[dict] = []
    index = {(row["case"], row["prefix_s"], row["npu_id"]): row for row in per_npu_rows}
    for prefix_s in PREFIX_SECONDS:
        for npu in range(NUM_NPU):
            baseline = index[("baseline", prefix_s, npu)]
            adaptive = index[("adaptive_t0_i100ms", prefix_s, npu)]
            per_npu_wide.append(
                {
                    "prefix_s": prefix_s,
                    "npu_id": npu,
                    "baseline_compute_ms": baseline["compute_ms"],
                    "adaptive_compute_ms": adaptive["compute_ms"],
                    "baseline_utilization_pct": baseline["utilization_pct"],
                    "adaptive_utilization_pct": adaptive["utilization_pct"],
                    "adaptive_minus_baseline_pp": adaptive["utilization_pct"] - baseline["utilization_pct"],
                    "baseline_idle_ms": baseline["idle_ms"],
                    "adaptive_idle_ms": adaptive["idle_ms"],
                }
            )
    summary_rows: list[dict] = []
    for prefix_s in PREFIX_SECONDS:
        baseline = prefix_rows_by_case["baseline"][prefix_s]
        adaptive = prefix_rows_by_case["adaptive_t0_i100ms"][prefix_s]
        actual_delta = adaptive["compute_busy_npu_ms"] - baseline["compute_busy_npu_ms"]
        admitted_delta = adaptive["admitted_compute_ms"] - baseline["admitted_compute_ms"]
        residual = actual_delta - admitted_delta
        for case in CASES:
            row = dict(prefix_rows_by_case[case][prefix_s])
            row.update(
                {
                    "adaptive_minus_baseline_npu_utilization_pp": adaptive["npu_utilization_pct"] - baseline["npu_utilization_pct"],
                    "adaptive_minus_baseline_actual_compute_busy_ms": actual_delta,
                    "adaptive_minus_baseline_admitted_compute_ms": admitted_delta,
                    "adaptive_minus_baseline_inferred_inventory_drop_ms": residual,
                    "admitted_compute_share_of_actual_delta_pct": (100.0 * admitted_delta / actual_delta if abs(actual_delta) > TOL else math.nan),
                    "active_remaining_compute_inventory_supported": False,
                }
            )
            summary_rows.append(row)
    return {
        "blocks": blocks_rows,
        "cumulative": cumulative_rows,
        "rolling8": rolling[8],
        "rolling16": rolling[16],
        "resources": resource_rows,
        "per_npu": per_npu_wide,
        "profiles": profile_rows,
        "matched_stall": _matched_request_rows(cases),
        "disjoint": disjoint_rows,
        "summary": summary_rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _require(rows, f"cannot write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.88,
        }
    )


def _save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, format="png", dpi=180, facecolor="white", bbox_inches="tight", metadata={"Title": path.stem, "Description": description})
    plt.close(fig)
    temporary.replace(path)


def _case_series(rows: Sequence[dict], case: str) -> list[dict]:
    return sorted((row for row in rows if row["case"] == case), key=lambda row: row.get("prefix_s", row.get("end_s", 0.0)))


def _plot_convergence(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.2), sharex=True)
    for case in CASES:
        raw = _case_series(tables["blocks"], case)
        cumulative = _case_series(tables["cumulative"], case)
        axes[0].plot([row["end_s"] for row in raw], [row["npu_utilization_pct"] for row in raw], color=COLORS[case], linewidth=1.2, alpha=0.72, label=LABELS[case])
        axes[1].plot([row["prefix_s"] for row in cumulative], [row["npu_utilization_pct"] for row in cumulative], color=COLORS[case], linewidth=2.3, label=LABELS[case])
    for axis in axes:
        for horizon in PREFIX_SECONDS:
            axis.axvline(horizon, color="#777777", linestyle=":" if horizon not in (8.0, 16.0, 32.0) else "--", linewidth=0.8, alpha=0.65)
        axis.grid(True, alpha=0.45)
        axis.set_ylabel("NPU compute busy-duty (%)")
        axis.legend(loc="best")
    axes[0].set_title("500-ms NPU utilization (phase-sensitive)")
    axes[1].set_title("Cumulative prefix utilization")
    axes[1].set_xlabel("Time since measurement start (s)")
    axes[1].set_xlim(0.0, 32.0)
    fig.suptitle("32 NPUs / 4 SSUs: utilization convergence", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "01_npu_utilization_convergence.png", "Raw 500-ms and cumulative NPU compute duty-cycle")


def _paired_by_time(rows: Sequence[dict], time_field: str) -> list[dict]:
    by_key = {(row["case"], row[time_field]): row for row in rows}
    times = sorted({row[time_field] for row in rows})
    return [
        {
            time_field: time,
            "baseline": by_key[("baseline", time)],
            "adaptive": by_key[("adaptive_t0_i100ms", time)],
        }
        for time in times
    ]


def _plot_delta(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    cumulative = _paired_by_time(tables["cumulative"], "prefix_s")
    rolling8 = _paired_by_time(tables["rolling8"], "end_s")
    rolling16 = _paired_by_time(tables["rolling16"], "end_s")
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.0), sharex=True)
    axes[0].plot([row["prefix_s"] for row in cumulative], [row["adaptive"]["npu_utilization_pct"] - row["baseline"]["npu_utilization_pct"] for row in cumulative], color="#2ca02c", linewidth=2.3, label="Cumulative prefix A - B")
    axes[0].plot([row["end_s"] for row in rolling8], [row["adaptive"]["npu_utilization_pct"] - row["baseline"]["npu_utilization_pct"] for row in rolling8], color="#9467bd", linewidth=1.8, label="Trailing 8 s A - B")
    axes[0].plot([row["end_s"] for row in rolling16], [row["adaptive"]["npu_utilization_pct"] - row["baseline"]["npu_utilization_pct"] for row in rolling16], color="#ff7f0e", linewidth=1.8, label="Trailing 16 s A - B")
    advantage = [(row["adaptive"]["compute_busy_npu_ms"] - row["baseline"]["compute_busy_npu_ms"]) / 1000.0 for row in cumulative]
    axes[1].plot([row["prefix_s"] for row in cumulative], advantage, color="#17becf", linewidth=2.3, label="Cumulative compute-busy advantage")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.9)
        for horizon in PREFIX_SECONDS:
            axis.axvline(horizon, color="#888888", linestyle="--", linewidth=0.75, alpha=0.55)
        axis.grid(True, alpha=0.45)
        axis.legend(loc="best")
    axes[0].set_ylabel("Adaptive - Baseline (pp)")
    axes[0].set_title("Utilization difference: cumulative and trailing windows")
    axes[1].set_ylabel("Adaptive - Baseline (NPU-s)")
    axes[1].set_xlabel("Time since measurement start (s)")
    axes[1].set_title("Integral of the compute-busy difference")
    axes[1].set_xlim(0.0, 32.0)
    fig.suptitle("Does the early Adaptive advantage persist?", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "02_adaptive_minus_baseline_convergence.png", "Cumulative and rolling Adaptive-minus-Baseline utilization")


def _plot_per_npu(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex=True, sharey=True)
    for axis, horizon in zip(axes.ravel(), FIGURE_PREFIX_SECONDS):
        rows = sorted((row for row in tables["per_npu"] if row["prefix_s"] == horizon), key=lambda row: row["npu_id"])
        x = [row["npu_id"] for row in rows]
        baseline = [row["baseline_utilization_pct"] for row in rows]
        adaptive = [row["adaptive_utilization_pct"] for row in rows]
        axis.plot(x, baseline, marker="o", markersize=3.0, color=COLORS["baseline"], linewidth=1.4, label=LABELS["baseline"])
        axis.plot(x, adaptive, marker="o", markersize=3.0, color=COLORS["adaptive_t0_i100ms"], linewidth=1.4, label=LABELS["adaptive_t0_i100ms"])
        axis.fill_between(x, baseline, adaptive, color="#999999", alpha=0.14)
        delta = statistics.fmean(row["adaptive_minus_baseline_pp"] for row in rows)
        axis.set_title(f"Prefix {horizon:g} s  |  fleet delta {delta:+.2f} pp")
        axis.grid(True, alpha=0.4)
        axis.set_ylim(0.0, 102.0)
    axes[0, 0].set_ylabel("NPU busy-duty (%)")
    axes[1, 0].set_ylabel("NPU busy-duty (%)")
    axes[1, 0].set_xlabel("NPU ID")
    axes[1, 1].set_xlabel("NPU ID")
    axes[0, 0].legend(loc="lower right", ncol=2, fontsize=8)
    fig.suptitle("Per-NPU redistribution across prefix lengths", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "03_per_npu_prefix_utilization.png", "Per-NPU utilization at 3, 8, 16, and 32 seconds")


def _plot_resources(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), sharex=True)
    fields = (
        ("ssd_mean_utilization_pct", "SSD busy-duty (%)", "500-ms SSD utilization"),
        ("npu_link_mean_utilization_pct", "NPU-link busy-duty (%)", "500-ms NPU-link utilization"),
        ("ssd_outstanding_gb_end", "Outstanding GB", "SSD queue at block boundary"),
        ("npu_link_outstanding_gb_end", "Outstanding GB", "NPU-link queue at block boundary"),
    )
    for axis, (field, ylabel, title) in zip(axes.ravel(), fields):
        for case in CASES:
            rows = _case_series(tables["resources"], case)
            axis.plot([row["end_s"] for row in rows], [row[field] for row in rows], color=COLORS[case], linewidth=1.6, label=LABELS[case])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.42)
    axes[1, 0].set_xlabel("Time since measurement start (s)")
    axes[1, 1].set_xlabel("Time since measurement start (s)")
    axes[0, 0].legend(loc="best")
    fig.suptitle("IO service and queue evolution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "04_resource_and_queue_timeseries.png", "SSD and NPU-link service utilization and outstanding queues")


def _plot_admitted_work(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
    for case in CASES:
        rows = _case_series(tables["blocks"], case)
        axes[0, 0].plot([row["end_s"] for row in rows], [row["admitted_compute_ms"] for row in rows], color=COLORS[case], linewidth=1.4, label=LABELS[case])
        axes[0, 1].plot([row["end_s"] for row in rows], [row["admitted_io_gb"] for row in rows], color=COLORS[case], linewidth=1.4, label=LABELS[case])
    cumulative = _paired_by_time(tables["cumulative"], "prefix_s")
    x = [row["prefix_s"] for row in cumulative]
    actual_delta = [row["adaptive"]["compute_busy_npu_ms"] - row["baseline"]["compute_busy_npu_ms"] for row in cumulative]
    admitted_delta = [row["adaptive"]["admitted_compute_ms"] - row["baseline"]["admitted_compute_ms"] for row in cumulative]
    axes[1, 0].plot(x, actual_delta, color="#17becf", linewidth=2.0, label="Actual compute-busy delta")
    axes[1, 0].plot(x, admitted_delta, color="#e377c2", linewidth=2.0, label="Admitted compute-work delta")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    for case in CASES:
        rows = _case_series(tables["cumulative"], case)
        axes[1, 1].plot([row["prefix_s"] for row in rows], [row["admitted_compute_ms_per_io_gb"] for row in rows], color=COLORS[case], linewidth=2.0, label=LABELS[case])
    titles = ("Admitted compute work per 500 ms", "Admitted IO work per 500 ms", "Cumulative work-mix delta", "Cumulative admitted compute/IO mix")
    ylabels = ("Ideal compute (ms)", "IO (GB)", "Adaptive - Baseline (ms)", "Compute (ms) / IO (GB)")
    for axis, title, ylabel in zip(axes.ravel(), titles, ylabels):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.42)
        axis.legend(loc="best", fontsize=8)
    axes[1, 0].set_xlabel("Time since measurement start (s)")
    axes[1, 1].set_xlabel("Time since measurement start (s)")
    fig.suptitle("Admitted workload mix versus realized compute busy time", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "05_admitted_work_mix.png", "Admitted compute and IO profile work compared with actual compute busy time")


def _plot_matched_stall(tables: dict[str, list[dict]], output: Path) -> None:
    _style()
    rows = sorted(
        (
            row
            for row in tables["matched_stall"]
            if row["dimension"] == "overall" and row["group"] == "all"
        ),
        key=lambda row: row["prefix_s"],
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.6), sharex=True)
    x = [row["prefix_s"] for row in rows]
    axes[0].plot(
        x,
        [row["adaptive_minus_baseline_mean_stall_ms"] for row in rows],
        color="#2ca02c",
        marker="o",
        linewidth=2.2,
        label="Mean paired stall delta",
    )
    axes[0].plot(
        x,
        [row["adaptive_minus_baseline_median_stall_ms"] for row in rows],
        color="#9467bd",
        marker="s",
        linewidth=1.8,
        label="Median paired stall delta",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("Adaptive - Baseline stall (ms)")
    axes[0].set_title("Same request IDs, different queue states (negative favors Adaptive)")
    axes[0].legend(loc="best")
    axes[1].plot(
        x,
        [row["matched_request_count"] for row in rows],
        color="#8c564b",
        marker="o",
        linewidth=2.2,
        label="Matched requests",
    )
    axes[1].plot(
        x,
        [row["baseline_admitted_request_count"] for row in rows],
        color=COLORS["baseline"],
        marker=".",
        linewidth=1.4,
        label="Baseline admitted",
    )
    axes[1].plot(
        x,
        [row["adaptive_admitted_request_count"] for row in rows],
        color=COLORS["adaptive_t0_i100ms"],
        marker=".",
        linewidth=1.4,
        label="Adaptive admitted",
    )
    axes[1].set_ylabel("Request count")
    axes[1].set_xlabel("Cumulative admission prefix (s)")
    axes[1].set_title("Matched-cohort coverage")
    axes[1].legend(loc="best")
    for axis in axes:
        axis.grid(True, alpha=0.42)
    fig.suptitle("Profile selection and matched-request stall", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output / "06_matched_request_stall.png", "Matched request-ID stall comparison controls request identity but not the global queue state")


def _lookup(rows: Sequence[dict], case: str, prefix_s: float) -> dict:
    matches = [row for row in rows if row["case"] == case and row["prefix_s"] == prefix_s]
    _require(len(matches) == 1, f"missing summary {case}/{prefix_s}s")
    return matches[0]


def _build_report(tables: dict[str, list[dict]], validation: dict) -> tuple[str, dict]:
    summaries = tables["summary"]
    paired_cumulative = _paired_by_time(tables["cumulative"], "prefix_s")
    deltas = [row["adaptive"]["npu_utilization_pct"] - row["baseline"]["npu_utilization_pct"] for row in paired_cumulative]
    busy_advantages = [row["adaptive"]["compute_busy_npu_ms"] - row["baseline"]["compute_busy_npu_ms"] for row in paired_cumulative]
    peak_index = max(range(len(busy_advantages)), key=busy_advantages.__getitem__)
    peak = paired_cumulative[peak_index]
    rolling8 = _paired_by_time(tables["rolling8"], "end_s")
    final_rolling8 = rolling8[-1]
    final_rolling8_delta = final_rolling8["adaptive"]["npu_utilization_pct"] - final_rolling8["baseline"]["npu_utilization_pct"]
    matched_overall = {
        row["prefix_s"]: row
        for row in tables["matched_stall"]
        if row["dimension"] == "overall" and row["group"] == "all"
    }
    disjoint8 = _paired_by_time(
        [row for row in tables["disjoint"] if row["window_s"] == 8.0],
        "end_s",
    )
    disjoint_deltas = [
        row["adaptive"]["npu_utilization_pct"]
        - row["baseline"]["npu_utilization_pct"]
        for row in disjoint8
    ]
    rows = []
    for horizon in PREFIX_SECONDS:
        baseline = _lookup(summaries, "baseline", horizon)
        adaptive = _lookup(summaries, "adaptive_t0_i100ms", horizon)
        rows.append(
            {
                "prefix_s": horizon,
                "baseline_util_pct": baseline["npu_utilization_pct"],
                "adaptive_util_pct": adaptive["npu_utilization_pct"],
                "delta_pp": adaptive["npu_utilization_pct"] - baseline["npu_utilization_pct"],
                "actual_busy_delta_ms": adaptive["compute_busy_npu_ms"] - baseline["compute_busy_npu_ms"],
                "admitted_compute_delta_ms": adaptive["admitted_compute_ms"] - baseline["admitted_compute_ms"],
                "inferred_inventory_drop_delta_ms": baseline["adaptive_minus_baseline_inferred_inventory_drop_ms"],
                "ssd_delta_pp": adaptive["ssd_mean_utilization_pct"] - baseline["ssd_mean_utilization_pct"],
                "link_delta_pp": adaptive["npu_link_mean_utilization_pct"] - baseline["npu_link_mean_utilization_pct"],
                "matched_mean_stall_delta_ms": matched_overall[horizon]["adaptive_minus_baseline_mean_stall_ms"],
                "matched_request_count": matched_overall[horizon]["matched_request_count"],
            }
        )
    short = next(row for row in rows if row["prefix_s"] == 3.0)
    full = next(row for row in rows if row["prefix_s"] == 32.0)
    per_npu_3 = [row for row in tables["per_npu"] if row["prefix_s"] == 3.0]
    improved = sum(row["adaptive_minus_baseline_pp"] > 0.0 for row in per_npu_3)
    baseline_sorted = sorted(per_npu_3, key=lambda row: row["baseline_utilization_pct"])
    low_quartile = baseline_sorted[: NUM_NPU // 4]
    low_gain = statistics.fmean(row["adaptive_minus_baseline_pp"] for row in low_quartile)
    early_profile_fraction = (
        short["admitted_compute_delta_ms"] / short["actual_busy_delta_ms"]
        if abs(short["actual_busy_delta_ms"]) > TOL
        else math.nan
    )
    initial_advantage_erased_by_8s = (
        short["delta_pp"] > 1.0
        and next(row for row in rows if row["prefix_s"] == 8.0)["delta_pp"] <= 0.0
    )
    long_cumulative_close = abs(full["delta_pp"]) <= 1.0
    last_window_prevents_convergence_claim = abs(final_rolling8_delta) > 1.0
    support = initial_advantage_erased_by_8s and 0.75 <= early_profile_fraction <= 1.25
    result = {
        "early_advantage_then_decay_supported": support,
        "initial_advantage_erased_by_8s": initial_advantage_erased_by_8s,
        "early_actual_delta_explained_by_admitted_compute_fraction": early_profile_fraction,
        "full_32s_cumulative_difference_within_1pp": long_cumulative_close,
        "last_8s_prevents_convergence_claim": last_window_prevents_convergence_claim,
        "disjoint_8s_adaptive_minus_baseline_pp": disjoint_deltas,
        "prefix_metrics": rows,
        "peak_compute_busy_advantage": {
            "prefix_s": peak["prefix_s"],
            "npu_seconds": busy_advantages[peak_index] / 1000.0,
        },
        "final_trailing_8s_delta_pp": final_rolling8_delta,
        "initial_3s_npus_improved": improved,
        "initial_3s_bottom_quartile_mean_gain_pp": low_gain,
        "matched_request_stall": {
            str(horizon): {
                "matched_request_count": matched_overall[horizon]["matched_request_count"],
                "adaptive_minus_baseline_mean_stall_ms": matched_overall[horizon]["adaptive_minus_baseline_mean_stall_ms"],
                "adaptive_minus_baseline_median_stall_ms": matched_overall[horizon]["adaptive_minus_baseline_median_stall_ms"],
            }
            for horizon in PREFIX_SECONDS
        },
        "active_remaining_compute_inventory_supported": False,
    }
    validation["mechanism_diagnostics"] = result
    table_lines = [
        "| 前缀 | Baseline NPU | Adaptive NPU | A-B | A-B actual busy | A-B admitted compute | A-B inferred inventory drop | matched stall A-B |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['prefix_s']:g}s | {row['baseline_util_pct']:.3f}% | {row['adaptive_util_pct']:.3f}% | {row['delta_pp']:+.3f} pp | {row['actual_busy_delta_ms']/1000.0:+.3f} NPU-s | {row['admitted_compute_delta_ms']/1000.0:+.3f} NPU-s | {row['inferred_inventory_drop_delta_ms']/1000.0:+.3f} NPU-s | {row['matched_mean_stall_delta_ms']:+.3f} ms (n={row['matched_request_count']}) |"
        )
    mechanism_sentence = (
        "这组数据证明 3 秒的领先没有持续到 8/16 秒；其主要来源是该短窗口准入了 compute 更重、IO 更轻的请求画像。3 秒 matched cohort 也观察到较低 stall，但长窗口的方向和幅度并不稳定。"
        if support
        else "这组数据未通过‘3 秒领先在 8 秒消失且 admitted-compute 能解释主要差值’判据，不能把短期差异归因于请求画像前移。"
    )
    chunk_lines = [
        "| 独立窗口 | Baseline NPU | Adaptive NPU | A-B | A-B admitted compute | A-B admitted IO |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in disjoint8:
        chunk_lines.append(
            f"| {row['end_s']-8:g}–{row['end_s']:g}s | {row['baseline']['npu_utilization_pct']:.3f}% | {row['adaptive']['npu_utilization_pct']:.3f}% | {row['adaptive']['npu_utilization_pct']-row['baseline']['npu_utilization_pct']:+.3f} pp | {(row['adaptive']['admitted_compute_ms']-row['baseline']['admitted_compute_ms'])/1000.0:+.3f} NPU-s | {row['adaptive']['admitted_io_gb']-row['baseline']['admitted_io_gb']:+.3f} GB |"
        )
    report = f"""# 32 NPU / SSU=4 / 32 秒收敛与机制审计

## 结论

{mechanism_sentence}

短窗口和长窗口并不是两种利用率定义：它们都是 compute interval 在窗口内的积分，只是积分边界不同。3 秒时 Adaptive 相对 Baseline 为 **{short['delta_pp']:+.3f} pp**，32 秒时为 **{full['delta_pp']:+.3f} pp**，最后一个 trailing-8s 窗口为 **{final_rolling8_delta:+.3f} pp**。Adaptive 的累计 compute-busy 优势在 {peak['prefix_s']:g}s 达到峰值 **{busy_advantages[peak_index]/1000.0:+.3f} NPU-s**；峰值随后回落，是与“前移工作而非创造长期算力”一致的积分证据。由于边界 Q 未独立记录，它不是单独充分的因果证明。

3 秒 actual busy 差为 **{short['actual_busy_delta_ms']/1000.0:+.3f} NPU-s**，admitted compute 差为 **{short['admitted_compute_delta_ms']/1000.0:+.3f} NPU-s**，比例为 **{100.0*early_profile_fraction:.1f}%**。这表示短期差的主体来自 measurement window 选中了不同请求画像；它不等价于 Adaptive 增加了硬件能力。

## 不同前缀的闭合量

{chr(10).join(table_lines)}

`actual busy` 是仿真实际 compute interval 的严格积分。`admitted compute` 是该前缀新准入请求的 `sum(16 × per-layer compute)`；`admitted IO` 则由 `raw_demand_gbps × ideal_ttft_s` 重建。对每个策略和每个窗口都有守恒式 `Q(start)-Q(end) = actual busy - admitted compute`，其中 Q 是 active request 的剩余 compute 库存。

当前 schema v2 **没有独立记录**每个边界的 Q，因此只能由上式报告 `inferred inventory drop`，不能拿它反过来宣称库存被独立验证。表中的 A-B inventory drop 是两策略该推断量之差；SSD queue 或 request count 没有被冒充为 compute 库存。

## 四个互不重叠的 8 秒窗口

{chr(10).join(chunk_lines)}

四段 A-B 分别为 **{', '.join(f'{value:+.3f} pp' for value in disjoint_deltas)}**。最后一段突然变为 {disjoint_deltas[-1]:+.3f} pp，而不是围绕一个小值稳定波动，因此 **32 秒累计均值虽只差 {full['delta_pp']:+.3f} pp，仍不足以证明已经收敛**。它说明更长窗口平均掉了大部分画像/相位差；单 seed 的末段波动本身不能证明底层非稳态，只能阻止我们宣称已经达到稳态。

## 为什么短期可能高、长期却不高

1. Adaptive 改的是 SSU 命令的 CIR 排队优先级。它可以更早把 IO 送到原本等待的 NPU，减少某一段时间内的 compute starvation；3 秒内有 **{improved}/32** 个 NPU 的累计利用率上升，Baseline 初始利用率最低四分之一 NPU 的平均变化为 **{low_gain:+.3f} pp**。
2. 控制器没有改变每层固定 compute 时间，也没有增加 SSD 的 40 GB/s 或 NPU-link 的 50 GB/s 容量。被提前完成的工作会改变后续队列和可运行层的数量；如果没有新增长期吞吐，累计优势会持平或回落。
3. 两策略使用相同 workload、placement 和 trace 指纹，但 warmup 完成时刻和 measurement 起始动态状态并不相同。相对时间 0 对齐的是“各自 warmup+settle 后”，不是完全相同的 SSD/link/active-compute 库存快照。
4. 这是 closed-loop 连续输入。不同策略在同一 3 秒内准入的 request 数和 profile mix 可以不同。因此必须同时看 admitted compute/IO，而不能只看请求个数。详细结果在 `profile_diagnostics.csv` 和图 05。
5. `matched_request_stall.csv` 只比较两个策略都出现的同一 request ID，并使用 `stall = TTFT - ideal_ttft`，从而排除 request identity 差异；但两边的绝对时刻和全局队列状态仍不同，因此它不是纯调度效率指标。3 秒 matched cohort 的平均 stall 变化为 **{matched_overall[3.0]['adaptive_minus_baseline_mean_stall_ms']:+.3f} ms**（负数表示 Adaptive 更好），32 秒为 **{matched_overall[32.0]['adaptive_minus_baseline_mean_stall_ms']:+.3f} ms**，且中间窗口会换符号；见图 06。
6. NPU utilization 在这里是二值 compute busy-duty，不是 tensor-core、FLOPS 或 HBM 硬件计数器。

## IO 与队列检查

每个 500ms block 已独立验证：逐 NPU compute busy、逐 SSU SSD busy/served、逐 NPU link busy/served均与 32 秒汇总闭合；SSD 以 40 GB/s、link 以 50 GB/s 的固定活动速率换算也逐块闭合。SSD/link outstanding 的块数和 GB 在相邻边界连续。资源曲线和队列见图 04；它们能解释 IO 何时可用，但不能替代缺失的 active compute 库存。

## 有效性与限制

- 输入严格为 32 NPU、4 SSU、16 层、warmup=8、settle=500ms、32 秒、64×500ms。
- baseline 与 Adaptive 100ms 的 source/config、十个输入指纹、prefix-32 materialization 和 simulator input fingerprint 完全一致。
- 每个 case 的 29 个 simulator invariants 全部为真，并由本分析再次重建关键资源与请求统计。
- 前 8 秒聚合利用率桥接旧正式实验：Baseline {HISTORICAL_8S_NPU_UTILIZATION['baseline']*100.0:.6f}%，Adaptive {HISTORICAL_8S_NPU_UTILIZATION['adaptive_t0_i100ms']*100.0:.6f}%；measurement start 也精确一致。
- 这仍是单 seed。32 秒能检验窗口收敛，但不能替代多 seed，也不能仅凭累计均值宣称已经达到稳态；应结合 rolling-8s/16s 和队列漂移判断。

## 输出

- `summary.csv`：2/3/8/16/32 秒前缀的主指标与 work-mix 残差
- `blocks_500ms.csv`、`cumulative_prefix.csv`、`rolling_8s.csv`、`rolling_16s.csv`、`disjoint_windows.csv`
- `per_npu_prefix_utilization.csv`
- `resource_diagnostics.csv`、`profile_diagnostics.csv`、`matched_request_stall.csv`
- `validation.json`：输入哈希、严格校验和机制判据
- 图 01–06：利用率收敛、差值积分、逐 NPU、IO/queue、admitted work mix、matched-request stall
"""
    return report, result


def analyze(
    paths: Sequence[Path],
    output: Path,
    *,
    enforce_historical_bridge: bool = True,
) -> None:
    cases, validation = _load_and_validate(
        paths,
        enforce_historical_bridge=enforce_historical_bridge,
    )
    tables = _build_tables(cases)
    report, _ = _build_report(tables, validation)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "summary.csv", tables["summary"])
    _write_csv(output / "blocks_500ms.csv", tables["blocks"])
    _write_csv(output / "cumulative_prefix.csv", tables["cumulative"])
    _write_csv(output / "rolling_8s.csv", tables["rolling8"])
    _write_csv(output / "rolling_16s.csv", tables["rolling16"])
    _write_csv(output / "disjoint_windows.csv", tables["disjoint"])
    _write_csv(output / "per_npu_prefix_utilization.csv", tables["per_npu"])
    _write_csv(output / "resource_diagnostics.csv", tables["resources"])
    _write_csv(output / "profile_diagnostics.csv", tables["profiles"])
    _write_csv(output / "matched_request_stall.csv", tables["matched_stall"])
    _plot_convergence(tables, output)
    _plot_delta(tables, output)
    _plot_per_npu(tables, output)
    _plot_resources(tables, output)
    _plot_admitted_work(tables, output)
    _plot_matched_stall(tables, output)
    validation["outputs"] = sorted(
        {path.name for path in output.iterdir()} | {"validation.json", "report.md"}
    )
    _write_json(output / "validation.json", validation)
    _write_text(output / "report.md", report)


def _synthetic_payload() -> dict:
    """Create a self-consistent schema-v2 fixture without reading experiment raw."""
    fingerprints = {field: hashlib.sha256(field.encode()).hexdigest() for field in PAIR_FIELDS}
    source = "1" * 64
    config = "2" * 64
    rows = []
    for case_index, case in enumerate(CASES):
        start = 10_000.0 + case_index * 777.0
        blocks = []
        boundaries = []
        cumulative_ssd_busy = np.zeros(NUM_SSU)
        cumulative_ssd_gb = np.zeros(NUM_SSU)
        cumulative_compute = np.zeros(NUM_NPU)
        cumulative_link_busy = np.zeros(NUM_NPU)
        cumulative_link_gb = np.zeros(NUM_NPU)
        ssd_queue = np.full(NUM_SSU, 0.25 + 0.03 * case_index)
        link_queue = np.zeros(NUM_NPU)
        request_rows = []
        for boundary in range(BLOCK_COUNT + 1):
            boundaries.append(
                {
                    "boundary": boundary,
                    "time_ms": start + boundary * BLOCK_MS,
                    "ssd_cumulative_busy_ms_by_ssu": cumulative_ssd_busy.tolist(),
                    "ssd_cumulative_served_gb_by_ssu": cumulative_ssd_gb.tolist(),
                    "ssd_outstanding_blocks_by_ssu": [int(round(value * 6000)) for value in ssd_queue],
                    "ssd_outstanding_gb_by_ssu": ssd_queue.tolist(),
                    "npu_compute_cumulative_busy_ms_by_npu": cumulative_compute.tolist(),
                    "npu_link_cumulative_busy_ms_by_npu": cumulative_link_busy.tolist(),
                    "npu_link_cumulative_served_gb_by_npu": cumulative_link_gb.tolist(),
                    "npu_link_outstanding_blocks_by_npu": [int(round(value * 6000)) for value in link_queue],
                    "npu_link_outstanding_gb_by_npu": link_queue.tolist(),
                }
            )
            if boundary == BLOCK_COUNT:
                break
            index = boundary
            phase = 0.03 * math.sin(index / 3.0)
            early = 0.08 if case_index and index < 6 else (-0.008 if case_index else 0.0)
            npu_utils = [min(0.99, max(0.2, 0.76 + phase + early + 0.06 * math.sin((npu + index) / 5.0))) for npu in range(NUM_NPU)]
            compute = [value * BLOCK_MS for value in npu_utils]
            ssd_utils = [0.92 + 0.015 * math.sin((index + ssu) / 4.0) for ssu in range(NUM_SSU)]
            ssd_busy = [value * BLOCK_MS for value in ssd_utils]
            ssd_gb = [value * SSD_CAP_GBPS / 1000.0 for value in ssd_busy]
            link_utils = [0.10 + 0.015 * math.sin((index + npu) / 7.0) for npu in range(NUM_NPU)]
            link_busy = [value * BLOCK_MS for value in link_utils]
            link_gb = [value * NPU_LINK_CAP_GBPS / 1000.0 for value in link_busy]
            ssd_start = ssd_queue.copy()
            link_start = link_queue.copy()
            ssd_queue = np.maximum(0.01, ssd_queue + 0.002 * np.sin(index + np.arange(NUM_SSU)))
            link_queue = np.maximum(0.0, 0.002 * np.sin(index / 2.0 + np.arange(NUM_NPU)))
            ssd_blocks_start = [int(round(value * 6000)) for value in ssd_start]
            ssd_blocks_end = [int(round(value * 6000)) for value in ssd_queue]
            link_blocks_start = [int(round(value * 6000)) for value in link_start]
            link_blocks_end = [int(round(value * 6000)) for value in link_queue]
            admitted = []
            for npu in range(NUM_NPU):
                request_id = index * NUM_NPU + npu
                ideal = 320.0 + 16.0 * ((npu + index) % 5)
                demand = 8.0 + 4.0 * ((npu + index) % 6)
                admission = start + index * BLOCK_MS + 10.0 + npu
                row = {
                    "request_id": request_id,
                    "npu_id": npu,
                    "sequence": index,
                    "category": ("SS", "SL", "LS", "LL")[(npu + index) % 4],
                    "profile_id": f"fixture-{(npu + index) % 6}",
                    "profile_key": [32 + 16 * ((npu + index) % 6), 64],
                    "profile_name": "synthetic fixture",
                    "raw_demand_gbps": demand,
                    "admission_time_ms": admission,
                    "completion_time_ms": admission + ideal * (1.05 + 0.02 * case_index),
                    "ttft_ms": ideal * (1.05 + 0.02 * case_index),
                    "ideal_ttft_ms": ideal,
                    "slo_met": True,
                }
                admitted.append(row)
                request_rows.append(row)
            block = {
                "block": index,
                "start_ms": start + index * BLOCK_MS,
                "end_ms": start + (index + 1) * BLOCK_MS,
                "duration_ms": BLOCK_MS,
                "npu_utilization": statistics.fmean(npu_utils),
                "request_count": len(admitted),
                "request_weighted_slo_attainment": 1.0,
                "ssd_busy_ms_by_ssu": ssd_busy,
                "ssd_served_gb_by_ssu": ssd_gb,
                "ssd_utilizations": ssd_utils,
                "ssd_mean_utilization": statistics.fmean(ssd_utils),
                "ssd_outstanding_blocks_at_start": ssd_blocks_start,
                "ssd_outstanding_blocks_at_end": ssd_blocks_end,
                "ssd_outstanding_blocks_delta": [b - a for a, b in zip(ssd_blocks_start, ssd_blocks_end)],
                "ssd_outstanding_gb_at_start": ssd_start.tolist(),
                "ssd_outstanding_gb_at_end": ssd_queue.tolist(),
                "ssd_outstanding_gb_delta": (ssd_queue - ssd_start).tolist(),
                "compute_ms_by_npu": compute,
                "npu_utilizations": npu_utils,
                "npu_link_busy_ms_by_npu": link_busy,
                "npu_link_served_gb_by_npu": link_gb,
                "npu_link_utilizations": link_utils,
                "npu_link_mean_utilization": statistics.fmean(link_utils),
                "npu_link_outstanding_blocks_at_start": link_blocks_start,
                "npu_link_outstanding_blocks_at_end": link_blocks_end,
                "npu_link_outstanding_blocks_delta": [b - a for a, b in zip(link_blocks_start, link_blocks_end)],
                "npu_link_outstanding_gb_at_start": link_start.tolist(),
                "npu_link_outstanding_gb_at_end": link_queue.tolist(),
                "npu_link_outstanding_gb_delta": (link_queue - link_start).tolist(),
            }
            blocks.append(block)
            cumulative_ssd_busy += ssd_busy
            cumulative_ssd_gb += ssd_gb
            cumulative_compute += compute
            cumulative_link_busy += link_busy
            cumulative_link_gb += link_gb
        request_counts = [BLOCK_COUNT] * NUM_NPU
        ssd_attribution = [[float(cumulative_ssd_gb[ssu]) / NUM_NPU for ssu in range(NUM_SSU)] for _ in range(NUM_NPU)]
        link_attribution = [[float(cumulative_link_gb[npu]) / NUM_SSU for _ in range(NUM_SSU)] for npu in range(NUM_NPU)]
        summary = {
            "schema_version": 2,
            "mode": "steady_state_full_load",
            "num_npu": NUM_NPU,
            "num_ssu": NUM_SSU,
            "n_layers": N_LAYERS,
            "batch_size": BATCH_SIZE,
            "warmup_requests_per_npu": WARMUP_REQUESTS,
            "settle_ms": SETTLE_MS,
            "measurement_start_ms": start,
            "measurement_end_ms": start + MEASUREMENT_MS,
            "measurement_duration_ms": MEASUREMENT_MS,
            "stationarity_boundary_semantics": BOUNDARY_SEMANTICS,
            "measurement_stationarity_boundary_count": BLOCK_COUNT + 1,
            "measurement_stationarity_boundaries": boundaries,
            "measurement_control_counter_window": CONTROL_WINDOW,
            "pressure_ttl_ms": 0.0,
            "cir_write_threshold_gbps": 0.0,
            "control_min_interval_ms": None if case == "baseline" else 100.0,
            "slo_alpha": 2.0,
            "mean_npu_utilization": float(cumulative_compute.sum()) / (NUM_NPU * MEASUREMENT_MS),
            "npu_utilizations": (cumulative_compute / MEASUREMENT_MS).tolist(),
            "compute_ms_by_npu": cumulative_compute.tolist(),
            "measurement_ssd_mean_utilization": float(cumulative_ssd_busy.sum()) / (NUM_SSU * MEASUREMENT_MS),
            "measurement_ssd_busy_ms_by_ssu": cumulative_ssd_busy.tolist(),
            "measurement_ssd_served_gb_by_ssu": cumulative_ssd_gb.tolist(),
            "measurement_npu_ssu_ssd_served_gb": ssd_attribution,
            "measurement_npu_link_mean_utilization": float(cumulative_link_busy.sum()) / (NUM_NPU * MEASUREMENT_MS),
            "measurement_npu_link_busy_ms_by_npu": cumulative_link_busy.tolist(),
            "measurement_npu_ssu_link_served_gb": link_attribution,
            "measurement_ssd_outstanding_blocks_at_start": blocks[0]["ssd_outstanding_blocks_at_start"],
            "measurement_ssd_outstanding_blocks_at_end": blocks[-1]["ssd_outstanding_blocks_at_end"],
            "measurement_request_count": len(request_rows),
            "request_counts_by_npu": request_counts,
            "request_rows": request_rows,
            "request_weighted_slo_attainment": 1.0,
            "ttft_slo_attainment": 1.0,
            "mean_ttft_ms": statistics.fmean(row["ttft_ms"] for row in request_rows),
            "p99_ttft_ms": float(np.percentile([row["ttft_ms"] for row in request_rows], 99)),
            "measurement_blocks": blocks,
            "input_fingerprint": "3" * 64,
            "invariants": {name: True for name in EXPECTED_INVARIANTS},
        }
        rows.append(
            {
                "status": "ok",
                "case": case,
                "kind": "baseline" if case == "baseline" else "adaptive",
                "num_ssu": NUM_SSU,
                "source_fingerprint": source,
                "config_fingerprint": config,
                "input_fingerprints": fingerprints,
                "prefix_32_materialized_fingerprints": {"workload": "4" * 64, "placement": "5" * 64, "trace": "6" * 64},
                "steady_summary": summary,
            }
        )
    spec = {
        "experiment": "synthetic_fixture",
        "num_npu": NUM_NPU,
        "n_layers": N_LAYERS,
        "batch_size": BATCH_SIZE,
        "cases": [
            {"name": "baseline", "kind": "baseline", "pressure_ttl_ms": 0.0, "cir_write_threshold_gbps": 0.0, "min_interval_ms": 0.0},
            {"name": "adaptive_t0_i100ms", "kind": "adaptive", "pressure_ttl_ms": 0.0, "cir_write_threshold_gbps": 0.0, "min_interval_ms": 100.0},
        ],
        "workload": {"requests_per_npu": 256},
        "steady_state": {"warmup_requests_per_npu": WARMUP_REQUESTS, "settle_ms": SETTLE_MS, "measurement_ms": MEASUREMENT_MS, "block_ms": BLOCK_MS, "slo_alpha": 2.0},
        "adaptive": {"ssd_cap_gbps": SSD_CAP_GBPS, "npu_cap_gbps": NPU_LINK_CAP_GBPS},
    }
    return {
        "schema_version": 2,
        "selected_complete": True,
        "source_stable_during_run": True,
        "config_stable_during_run": True,
        "source_fingerprint": source,
        "ending_source_fingerprint": source,
        "source_manifest": {"synthetic": "fixture"},
        "config_fingerprint": config,
        "ending_config_fingerprint": config,
        "experiment_spec": spec,
        "schedule_metadata": {"mode": "synthetic", "seed": 42, "num_npu": NUM_NPU},
        "pairing_audit": {"4": {"cases": list(CASES), "has_rows": True, "all_available_rows_paired": True}},
        "results": rows,
    }


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="npu32-ssu4-convergence-selftest-") as temporary:
        root = Path(temporary)
        fixture = root / "fixture.json"
        fixture.write_text(json.dumps(_synthetic_payload()), encoding="utf-8")
        output = root / "analysis"
        analyze([fixture], output, enforce_historical_bridge=False)
        # The publication path is routinely regenerated in place; verify that
        # its manifest and atomic writers remain idempotent on a second pass.
        analyze([fixture], output, enforce_historical_bridge=False)
        expected = {
            "summary.csv",
            "blocks_500ms.csv",
            "cumulative_prefix.csv",
            "rolling_8s.csv",
            "rolling_16s.csv",
            "disjoint_windows.csv",
            "per_npu_prefix_utilization.csv",
            "resource_diagnostics.csv",
            "profile_diagnostics.csv",
            "matched_request_stall.csv",
            "validation.json",
            "report.md",
            "01_npu_utilization_convergence.png",
            "02_adaptive_minus_baseline_convergence.png",
            "03_per_npu_prefix_utilization.png",
            "04_resource_and_queue_timeseries.png",
            "05_admitted_work_mix.png",
            "06_matched_request_stall.png",
        }
        actual = {path.name for path in output.iterdir()}
        _require(expected == actual, f"self-test output mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        _require(validation.get("passed") is True, "self-test validation did not pass")
    print("synthetic self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, help="paired JSON or one case shard; repeat for two shards")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true", help="run only an isolated synthetic fixture test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    inputs = args.input if args.input else [DEFAULT_INPUT]
    analyze(inputs, args.output_dir)
    print(f"analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
