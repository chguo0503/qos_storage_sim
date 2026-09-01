#!/usr/bin/env python3
"""Verify and visualize a paired 32-NPU, 4-or-5-SSU microtrace diagnostic.

The input is produced by :mod:`run_npu32_ssu5_microtrace`.  This program does
not re-simulate anything.  It independently integrates the exact half-open
service intervals into 0.5 ms bins, checks instantaneous capacity and the
simulator's 5 ms accounting, and writes auditable CSV files plus plots.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_INPUT_ROOT = Path(
    "results/ms_scale_control/npu32_ssu4_microtrace_baseline_adaptive100_v1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "analysis"
PROJECT_ROOT = Path(__file__).resolve().parent
EXPECTED_CASES = ("baseline", "adaptive_t0_i100ms")
CASE_LABELS = {
    "baseline": "Baseline",
    "adaptive_t0_i100ms": "Adaptive: interval 100 ms",
}
CASE_COLORS = {
    "baseline": "#1f77b4",
    "adaptive_t0_i100ms": "#9467bd",
}
EPS = 1e-9


class AnalysisError(ValueError):
    """Raised when a raw artifact cannot support the requested audit."""


@dataclass(frozen=True)
class CaseData:
    name: str
    case_dir: Path
    summary: dict
    trace: dict

    @property
    def num_ssu(self) -> int:
        return int(self.summary["num_ssu"])


@dataclass(frozen=True)
class ServiceInterval:
    case: str
    resource: str
    start_ms: float
    end_ms: float
    rate: float
    npu_id: int | None
    ssu_id: int | None
    request_id: int | None
    batch_id: int | None
    layer: int | None
    block_idx: int | None
    size_gb: float | None
    source: dict

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class Bin:
    index: int
    start_ms: float
    end_ms: float
    trace_start_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    @property
    def relative_start_ms(self) -> float:
        return self.start_ms - self.trace_start_ms

    @property
    def relative_end_ms(self) -> float:
        return self.end_ms - self.trace_start_ms

    @property
    def relative_mid_ms(self) -> float:
        return (self.relative_start_ms + self.relative_end_ms) / 2.0


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
    if isinstance(value, bool):
        raise AnalysisError(f"{context}: expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"{context}: expected an integer") from error
    _require(float(result) == float(value), f"{context}: expected an integer")
    return result


def _optional_integer(value: object, context: str) -> int | None:
    return None if value is None else _integer(value, context)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path}: top-level JSON must be an object")
    return value


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _discover_case_dirs(input_roots: Sequence[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for input_root in input_roots:
        root = input_root.expanduser().resolve()
        candidates: list[Path] = []
        if (root / "summary.json").is_file() and (root / "microtrace.json").is_file():
            candidates.append(root)
        else:
            for case in EXPECTED_CASES:
                candidates.extend(
                    path.parent
                    for path in root.glob(f"**/{case}/microtrace.json")
                    if (path.parent / "summary.json").is_file()
                )
        for case_dir in candidates:
            trace = _load_json(case_dir / "microtrace.json")
            case = trace.get("case", case_dir.name)
            if case not in EXPECTED_CASES:
                continue
            previous = found.get(case)
            _require(
                previous is None or previous == case_dir,
                f"duplicate raw input for {case}: {previous} and {case_dir}",
            )
            found[case] = case_dir
    missing = [case for case in EXPECTED_CASES if case not in found]
    _require(not missing, f"missing paired case directories: {missing}")
    return found


def _load_cases(input_roots: Sequence[Path]) -> list[CaseData]:
    case_dirs = _discover_case_dirs(input_roots)
    cases = []
    for name in EXPECTED_CASES:
        case_dir = case_dirs[name]
        summary = _load_json(case_dir / "summary.json")
        trace = _load_json(case_dir / "microtrace.json")
        _require(trace.get("case") == name, f"{case_dir}: trace case mismatch")
        _require(summary.get("num_npu") == 32, f"{name}: expected 32 NPUs")
        _require(summary.get("num_ssu") in (4, 5), f"{name}: expected 4 or 5 SSUs")
        _require(summary.get("batch_size") == 1, f"{name}: expected batch size 1")
        _require(
            all(bool(value) for value in summary.get("invariants", {}).values()),
            f"{name}: simulator invariant failed",
        )
        validation = trace.get("validation", {})
        _require(validation.get("passed") is True, f"{name}: runner trace validation failed")
        cases.append(CaseData(name, case_dir, summary, trace))

    fingerprints = {item.summary.get("input_fingerprint") for item in cases}
    _require(len(fingerprints) == 1 and None not in fingerprints, "paired inputs differ")
    _require(
        len({item.num_ssu for item in cases}) == 1,
        "paired cases use different SSU counts",
    )
    starts = {_number(item.trace["trace_start_ms"], item.name) for item in cases}
    durations = {_number(item.trace["trace_duration_ms"], item.name) for item in cases}
    _require(len(durations) == 1, "paired trace durations differ")
    # Absolute DES time can differ by strategy because warmup completion differs.
    _require(all(value >= 0.0 for value in starts), "invalid trace start")
    return cases


def _normalise_intervals(case: CaseData) -> list[ServiceInterval]:
    specs = (
        (
            "ssd",
            "ssd_dispatch_intervals",
            "ssd_start_time_ms",
            "ssd_end_time_ms",
            "actual_ssd_bw_gbps",
        ),
        (
            "link",
            "npu_link_intervals",
            "link_start_time_ms",
            "link_end_time_ms",
            "actual_npu_link_bw_gbps",
        ),
        (
            "compute",
            "compute_intervals",
            "compute_start_time_ms",
            "compute_end_time_ms",
            None,
        ),
    )
    intervals: list[ServiceInterval] = []
    trace_start = _number(case.trace["trace_start_ms"], f"{case.name}.trace_start")
    trace_end = _number(case.trace["trace_end_ms"], f"{case.name}.trace_end")
    for resource, field, start_field, end_field, rate_field in specs:
        rows = case.trace.get(field)
        _require(isinstance(rows, list), f"{case.name}.{field}: expected a list")
        for index, row in enumerate(rows):
            context = f"{case.name}.{field}[{index}]"
            _require(isinstance(row, dict), f"{context}: expected an object")
            start = _number(row.get(start_field), f"{context}.{start_field}")
            end = _number(row.get(end_field), f"{context}.{end_field}")
            _require(end > start, f"{context}: non-positive interval")
            _require(start < trace_end + EPS and end > trace_start - EPS, f"{context}: no trace overlap")
            rate = 1.0 if rate_field is None else _number(row.get(rate_field), f"{context}.{rate_field}")
            _require(rate >= 0.0, f"{context}: negative service rate")
            intervals.append(
                ServiceInterval(
                    case=case.name,
                    resource=resource,
                    start_ms=start,
                    end_ms=end,
                    rate=rate,
                    npu_id=_optional_integer(row.get("npu_id"), f"{context}.npu_id"),
                    ssu_id=_optional_integer(row.get("ssu_id"), f"{context}.ssu_id"),
                    request_id=_optional_integer(row.get("request_id"), f"{context}.request_id"),
                    batch_id=_optional_integer(row.get("batch_id"), f"{context}.batch_id"),
                    layer=_optional_integer(row.get("layer"), f"{context}.layer"),
                    block_idx=_optional_integer(row.get("block_idx"), f"{context}.block_idx"),
                    size_gb=(
                        None
                        if row.get("size_gb") is None
                        else _number(row["size_gb"], f"{context}.size_gb")
                    ),
                    source=row,
                )
            )
    return intervals


def _make_bins(start_ms: float, end_ms: float, bin_ms: float) -> list[Bin]:
    _require(bin_ms > 0.0, "bin size must be positive")
    count_float = (end_ms - start_ms) / bin_ms
    count = int(round(count_float))
    _require(math.isclose(count_float, count, abs_tol=1e-8), "trace duration must be divisible by bin size")
    return [
        Bin(index, start_ms + index * bin_ms, start_ms + (index + 1) * bin_ms, start_ms)
        for index in range(count)
    ]


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _intervals_for(
    intervals: Iterable[ServiceInterval],
    resource: str,
    *,
    npu_id: int | None = None,
    ssu_id: int | None = None,
) -> list[ServiceInterval]:
    return [
        item
        for item in intervals
        if item.resource == resource
        and (npu_id is None or item.npu_id == npu_id)
        and (ssu_id is None or item.ssu_id == ssu_id)
    ]


def _integrate(
    intervals: Iterable[ServiceInterval], start_ms: float, end_ms: float
) -> tuple[float, float, set[int], set[int]]:
    busy_ms = 0.0
    volume = 0.0
    requests: set[int] = set()
    layers: set[int] = set()
    for item in intervals:
        overlap_ms = _overlap(item.start_ms, item.end_ms, start_ms, end_ms)
        if overlap_ms <= 0.0:
            continue
        busy_ms += overlap_ms
        volume += item.rate * overlap_ms / 1000.0
        if item.request_id is not None:
            requests.add(item.request_id)
        if item.layer is not None:
            layers.add(item.layer)
    return busy_ms, volume, requests, layers


def _json_cell(values: Iterable[int]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def _build_binned_resource_rows(
    case: CaseData, intervals: list[ServiceInterval], bins: Sequence[Bin]
) -> tuple[list[dict], list[dict], list[dict]]:
    npu_rows: list[dict] = []
    ssu_rows: list[dict] = []
    fleet_rows: list[dict] = []
    by_resource = {
        resource: [item for item in intervals if item.resource == resource]
        for resource in ("compute", "link", "ssd")
    }

    for bin_row in bins:
        fleet_compute_ms = 0.0
        fleet_link_ms = 0.0
        fleet_link_gb = 0.0
        npu_utils: list[float] = []
        for npu_id in range(32):
            compute = [item for item in by_resource["compute"] if item.npu_id == npu_id]
            links = [item for item in by_resource["link"] if item.npu_id == npu_id]
            compute_ms, _, compute_requests, compute_layers = _integrate(
                compute, bin_row.start_ms, bin_row.end_ms
            )
            link_ms, link_gb, link_requests, link_layers = _integrate(
                links, bin_row.start_ms, bin_row.end_ms
            )
            util = compute_ms / bin_row.duration_ms
            link_util = link_ms / bin_row.duration_ms
            fleet_compute_ms += compute_ms
            fleet_link_ms += link_ms
            fleet_link_gb += link_gb
            npu_utils.append(util)
            npu_rows.append(
                {
                    "case": case.name,
                    "num_ssu": case.num_ssu,
                    "bin": bin_row.index,
                    "relative_start_ms": bin_row.relative_start_ms,
                    "relative_end_ms": bin_row.relative_end_ms,
                    "absolute_start_ms": bin_row.start_ms,
                    "absolute_end_ms": bin_row.end_ms,
                    "duration_ms": bin_row.duration_ms,
                    "npu_id": npu_id,
                    "compute_busy_ms": compute_ms,
                    "compute_utilization": util,
                    "compute_utilization_pct": 100.0 * util,
                    "npu_link_busy_ms": link_ms,
                    "npu_link_utilization": link_util,
                    "npu_link_utilization_pct": 100.0 * link_util,
                    "npu_link_served_gb": link_gb,
                    "actual_npu_link_bandwidth_gbps": link_gb * 1000.0 / bin_row.duration_ms,
                    "compute_request_ids": _json_cell(compute_requests),
                    "compute_layers": _json_cell(compute_layers),
                    "link_request_ids": _json_cell(link_requests),
                    "link_layers": _json_cell(link_layers),
                }
            )

        fleet_ssd_ms = 0.0
        fleet_ssd_gb = 0.0
        ssu_bandwidths: list[float] = []
        for ssu_id in range(case.num_ssu):
            ssd = [item for item in by_resource["ssd"] if item.ssu_id == ssu_id]
            busy_ms, served_gb, request_ids, layers = _integrate(
                ssd, bin_row.start_ms, bin_row.end_ms
            )
            bandwidth = served_gb * 1000.0 / bin_row.duration_ms
            fleet_ssd_ms += busy_ms
            fleet_ssd_gb += served_gb
            ssu_bandwidths.append(bandwidth)
            ssu_rows.append(
                {
                    "case": case.name,
                    "num_ssu": case.num_ssu,
                    "bin": bin_row.index,
                    "relative_start_ms": bin_row.relative_start_ms,
                    "relative_end_ms": bin_row.relative_end_ms,
                    "absolute_start_ms": bin_row.start_ms,
                    "absolute_end_ms": bin_row.end_ms,
                    "duration_ms": bin_row.duration_ms,
                    "ssu_id": ssu_id,
                    "ssd_busy_ms": busy_ms,
                    "ssd_utilization": busy_ms / bin_row.duration_ms,
                    "ssd_utilization_pct": 100.0 * busy_ms / bin_row.duration_ms,
                    "ssd_served_gb": served_gb,
                    "actual_ssd_bandwidth_gbps": bandwidth,
                    "request_ids": _json_cell(request_ids),
                    "layers": _json_cell(layers),
                }
            )

        fleet_rows.append(
            {
                "case": case.name,
                "num_ssu": case.num_ssu,
                "bin": bin_row.index,
                "relative_start_ms": bin_row.relative_start_ms,
                "relative_end_ms": bin_row.relative_end_ms,
                "absolute_start_ms": bin_row.start_ms,
                "absolute_end_ms": bin_row.end_ms,
                "duration_ms": bin_row.duration_ms,
                "mean_npu_compute_utilization": fleet_compute_ms / (32 * bin_row.duration_ms),
                "mean_npu_compute_utilization_pct": 100.0 * fleet_compute_ms / (32 * bin_row.duration_ms),
                "min_npu_compute_utilization_pct": 100.0 * min(npu_utils),
                "max_npu_compute_utilization_pct": 100.0 * max(npu_utils),
                "mean_npu_link_utilization": fleet_link_ms / (32 * bin_row.duration_ms),
                "mean_npu_link_utilization_pct": 100.0 * fleet_link_ms / (32 * bin_row.duration_ms),
                "fleet_npu_link_served_gb": fleet_link_gb,
                "fleet_npu_link_bandwidth_gbps": fleet_link_gb * 1000.0 / bin_row.duration_ms,
                "mean_ssd_utilization": fleet_ssd_ms
                / (case.num_ssu * bin_row.duration_ms),
                "mean_ssd_utilization_pct": 100.0
                * fleet_ssd_ms
                / (case.num_ssu * bin_row.duration_ms),
                "fleet_ssd_served_gb": fleet_ssd_gb,
                "fleet_ssd_bandwidth_gbps": fleet_ssd_gb * 1000.0 / bin_row.duration_ms,
                "max_single_ssu_bandwidth_gbps": max(ssu_bandwidths),
            }
        )
    return npu_rows, ssu_rows, fleet_rows


def _build_full_measurement_rows(
    case: CaseData,
    *,
    rolling_ms: float = 100.0,
) -> tuple[list[dict], list[dict], dict]:
    """Reconstruct the full 3 s measurement from native 5 ms blocks."""
    blocks = sorted(case.summary["measurement_blocks"], key=lambda row: row["start_ms"])
    _require(blocks, f"{case.name}: no measurement blocks")
    measurement_start = _number(
        case.summary["measurement_start_ms"], f"{case.name}.measurement_start_ms"
    )
    measurement_end = _number(
        case.summary["measurement_end_ms"], f"{case.name}.measurement_end_ms"
    )
    _require(
        math.isclose(float(blocks[0]["start_ms"]), measurement_start, abs_tol=EPS)
        and math.isclose(float(blocks[-1]["end_ms"]), measurement_end, abs_tol=EPS),
        f"{case.name}: measurement blocks do not span the full window",
    )
    _require(
        all(
            math.isclose(
                float(previous["end_ms"]),
                float(current["start_ms"]),
                abs_tol=EPS,
            )
            for previous, current in zip(blocks, blocks[1:])
        ),
        f"{case.name}: measurement blocks are not contiguous",
    )

    count = len(blocks)
    duration_prefix = [0.0]
    compute_prefix = [[0.0] for _ in range(32)]
    ssd_busy_prefix = [0.0]
    ssd_served_prefix = [0.0]
    link_busy_prefix = [0.0]
    link_served_prefix = [0.0]
    for block_index, block in enumerate(blocks):
        duration = _number(block["duration_ms"], f"{case.name}.block[{block_index}]")
        compute = list(map(float, block["compute_ms_by_npu"]))
        ssd_busy = list(map(float, block["ssd_busy_ms_by_ssu"]))
        ssd_served = list(map(float, block["ssd_served_gb_by_ssu"]))
        link_busy = list(map(float, block["npu_link_busy_ms_by_npu"]))
        link_served = list(map(float, block["npu_link_served_gb_by_npu"]))
        _require(len(compute) == 32 and len(link_busy) == 32, "full block NPU shape")
        _require(
            len(ssd_busy) == case.num_ssu
            and len(ssd_served) == case.num_ssu,
            "full block SSU shape",
        )
        duration_prefix.append(duration_prefix[-1] + duration)
        for npu_id in range(32):
            compute_prefix[npu_id].append(
                compute_prefix[npu_id][-1] + compute[npu_id]
            )
        ssd_busy_prefix.append(ssd_busy_prefix[-1] + math.fsum(ssd_busy))
        ssd_served_prefix.append(ssd_served_prefix[-1] + math.fsum(ssd_served))
        link_busy_prefix.append(link_busy_prefix[-1] + math.fsum(link_busy))
        link_served_prefix.append(link_served_prefix[-1] + math.fsum(link_served))

    fleet_rows: list[dict] = []
    npu_rows: list[dict] = []
    window_start_index = 0
    for index, block in enumerate(blocks):
        absolute_start = float(block["start_ms"])
        absolute_end = float(block["end_ms"])
        duration = float(block["duration_ms"])
        rolling_start_time = absolute_end - rolling_ms
        while (
            window_start_index < index
            and float(blocks[window_start_index]["end_ms"])
            <= rolling_start_time + EPS
        ):
            window_start_index += 1
        rolling_duration = (
            duration_prefix[index + 1] - duration_prefix[window_start_index]
        )
        cumulative_duration = duration_prefix[index + 1]
        raw_compute_ms = math.fsum(
            compute_prefix[npu_id][index + 1]
            - compute_prefix[npu_id][index]
            for npu_id in range(32)
        )
        rolling_compute_ms = math.fsum(
            compute_prefix[npu_id][index + 1]
            - compute_prefix[npu_id][window_start_index]
            for npu_id in range(32)
        )
        cumulative_compute_ms = math.fsum(
            compute_prefix[npu_id][index + 1] for npu_id in range(32)
        )
        raw_ssd_busy_ms = ssd_busy_prefix[index + 1] - ssd_busy_prefix[index]
        rolling_ssd_busy_ms = (
            ssd_busy_prefix[index + 1] - ssd_busy_prefix[window_start_index]
        )
        cumulative_ssd_busy_ms = ssd_busy_prefix[index + 1]
        raw_link_busy_ms = link_busy_prefix[index + 1] - link_busy_prefix[index]
        rolling_link_busy_ms = (
            link_busy_prefix[index + 1] - link_busy_prefix[window_start_index]
        )
        cumulative_link_busy_ms = link_busy_prefix[index + 1]
        raw_ssd_gb = ssd_served_prefix[index + 1] - ssd_served_prefix[index]
        rolling_ssd_gb = (
            ssd_served_prefix[index + 1] - ssd_served_prefix[window_start_index]
        )
        cumulative_ssd_gb = ssd_served_prefix[index + 1]
        raw_link_gb = link_served_prefix[index + 1] - link_served_prefix[index]
        rolling_link_gb = (
            link_served_prefix[index + 1] - link_served_prefix[window_start_index]
        )
        cumulative_link_gb = link_served_prefix[index + 1]
        fleet_rows.append(
            {
                "case": case.name,
                "num_ssu": case.num_ssu,
                "block": index,
                "relative_start_ms": absolute_start - measurement_start,
                "relative_end_ms": absolute_end - measurement_start,
                "relative_mid_ms": (absolute_start + absolute_end) / 2.0
                - measurement_start,
                "absolute_start_ms": absolute_start,
                "absolute_end_ms": absolute_end,
                "duration_ms": duration,
                "rolling_window_ms": rolling_duration,
                "raw_5ms_mean_npu_compute_utilization_pct": 100.0
                * raw_compute_ms
                / (32.0 * duration),
                "rolling_100ms_mean_npu_compute_utilization_pct": 100.0
                * rolling_compute_ms
                / (32.0 * rolling_duration),
                "cumulative_mean_npu_compute_utilization_pct": 100.0
                * cumulative_compute_ms
                / (32.0 * cumulative_duration),
                "raw_5ms_mean_ssd_utilization_pct": 100.0
                * raw_ssd_busy_ms
                / (case.num_ssu * duration),
                "rolling_100ms_mean_ssd_utilization_pct": 100.0
                * rolling_ssd_busy_ms
                / (case.num_ssu * rolling_duration),
                "cumulative_mean_ssd_utilization_pct": 100.0
                * cumulative_ssd_busy_ms
                / (case.num_ssu * cumulative_duration),
                "raw_5ms_mean_npu_link_utilization_pct": 100.0
                * raw_link_busy_ms
                / (32.0 * duration),
                "rolling_100ms_mean_npu_link_utilization_pct": 100.0
                * rolling_link_busy_ms
                / (32.0 * rolling_duration),
                "cumulative_mean_npu_link_utilization_pct": 100.0
                * cumulative_link_busy_ms
                / (32.0 * cumulative_duration),
                "raw_5ms_fleet_ssd_bandwidth_gbps": raw_ssd_gb
                * 1000.0
                / duration,
                "rolling_100ms_fleet_ssd_bandwidth_gbps": rolling_ssd_gb
                * 1000.0
                / rolling_duration,
                "cumulative_fleet_ssd_bandwidth_gbps": cumulative_ssd_gb
                * 1000.0
                / cumulative_duration,
                "raw_5ms_fleet_npu_link_bandwidth_gbps": raw_link_gb
                * 1000.0
                / duration,
                "rolling_100ms_fleet_npu_link_bandwidth_gbps": rolling_link_gb
                * 1000.0
                / rolling_duration,
                "cumulative_fleet_npu_link_bandwidth_gbps": cumulative_link_gb
                * 1000.0
                / cumulative_duration,
                "request_count": block["request_count"],
                "request_weighted_slo_attainment": block[
                    "request_weighted_slo_attainment"
                ],
            }
        )
        for npu_id in range(32):
            raw_busy = (
                compute_prefix[npu_id][index + 1]
                - compute_prefix[npu_id][index]
            )
            rolling_busy = (
                compute_prefix[npu_id][index + 1]
                - compute_prefix[npu_id][window_start_index]
            )
            cumulative_busy = compute_prefix[npu_id][index + 1]
            npu_rows.append(
                {
                    "case": case.name,
                    "num_ssu": case.num_ssu,
                    "block": index,
                    "relative_start_ms": absolute_start - measurement_start,
                    "relative_end_ms": absolute_end - measurement_start,
                    "absolute_start_ms": absolute_start,
                    "absolute_end_ms": absolute_end,
                    "duration_ms": duration,
                    "rolling_window_ms": rolling_duration,
                    "npu_id": npu_id,
                    "raw_5ms_compute_utilization_pct": 100.0
                    * raw_busy
                    / duration,
                    "rolling_100ms_compute_utilization_pct": 100.0
                    * rolling_busy
                    / rolling_duration,
                    "cumulative_compute_utilization_pct": 100.0
                    * cumulative_busy
                    / cumulative_duration,
                }
            )

    final = fleet_rows[-1]
    final_npu = [
        row
        for row in npu_rows
        if int(row["block"]) == count - 1
    ]
    summary_npu = list(map(float, case.summary["npu_utilizations"]))
    residuals = {
        "fleet_compute_utilization": abs(
            float(final["cumulative_mean_npu_compute_utilization_pct"]) / 100.0
            - float(case.summary["mean_npu_utilization"])
        ),
        "fleet_ssd_utilization": abs(
            float(final["cumulative_mean_ssd_utilization_pct"]) / 100.0
            - float(case.summary["measurement_ssd_mean_utilization"])
        ),
        "fleet_npu_link_utilization": abs(
            float(final["cumulative_mean_npu_link_utilization_pct"]) / 100.0
            - float(case.summary["measurement_npu_link_mean_utilization"])
        ),
        "per_npu_compute_utilization": max(
            abs(
                float(row["cumulative_compute_utilization_pct"]) / 100.0
                - summary_npu[int(row["npu_id"])]
            )
            for row in final_npu
        ),
    }
    audit = {
        "passed": len(blocks) == 600
        and math.isclose(duration_prefix[-1], 3000.0, abs_tol=1e-8)
        and max(residuals.values()) <= 1e-10,
        "block_count": len(blocks),
        "covered_duration_ms": duration_prefix[-1],
        "rolling_window_target_ms": rolling_ms,
        "maximum_absolute_residuals": residuals,
    }
    return fleet_rows, npu_rows, audit


def _cir_segments(case: CaseData) -> list[dict]:
    events = case.trace.get("cir_table_events")
    _require(isinstance(events, list) and events, f"{case.name}: missing CIR snapshots")
    events = sorted(events, key=lambda row: (float(row["time_ms"]), int(row["event_index"])))
    trace_start = _number(case.trace["trace_start_ms"], case.name)
    trace_end = _number(case.trace["trace_end_ms"], case.name)
    _require(math.isclose(float(events[0]["time_ms"]), trace_start, abs_tol=EPS), f"{case.name}: no CIR snapshot at trace start")
    segments = []
    for index, event in enumerate(events):
        start = max(trace_start, float(event["time_ms"]))
        end = trace_end if index + 1 == len(events) else min(trace_end, float(events[index + 1]["time_ms"]))
        if end <= start + EPS:
            continue
        table = event.get("installed_tables_by_ssu_gbps")
        paths = event.get("npu_dedicated_paths")
        _require(
            isinstance(table, list) and len(table) == case.num_ssu,
            f"{case.name}: invalid CIR table",
        )
        _require(
            all(isinstance(row, list) and row for row in table),
            f"{case.name}: invalid CIR table rows",
        )
        _require(
            len({len(row) for row in table}) == 1,
            f"{case.name}: ragged CIR table",
        )
        if paths is not None:
            _require(
                isinstance(paths, list) and len(paths) == 32,
                f"{case.name}: invalid NPU path mapping",
            )
            _require(
                all(len(row) > max(paths) for row in table),
                f"{case.name}: CIR table/path mismatch",
            )
        segments.append({"start_ms": start, "end_ms": end, "table": table, "paths": paths, "event": event})
    _require(segments and math.isclose(segments[-1]["end_ms"], trace_end, abs_tol=EPS), f"{case.name}: CIR segments do not cover trace")
    return segments


def _build_cir_rows(case: CaseData, bins: Sequence[Bin]) -> list[dict]:
    segments = _cir_segments(case)
    rows: list[dict] = []
    table_width = len(segments[0]["table"][0])
    dedicated_paths = segments[0]["paths"]
    _require(
        all(segment["paths"] == dedicated_paths for segment in segments),
        f"{case.name}: NPU path mapping changed inside trace",
    )
    path_to_npu = (
        {}
        if dedicated_paths is None
        else {int(path_id): npu_id for npu_id, path_id in enumerate(dedicated_paths)}
    )
    observed_paths = {
        int(row["path_id"])
        for row in case.trace.get("ssd_dispatch_intervals", [])
        if row.get("path_id") is not None
    }
    nonzero_paths = {
        path_id
        for segment in segments
        for table_row in segment["table"]
        for path_id, value in enumerate(table_row)
        if abs(float(value)) > EPS
    }
    relevant_paths = sorted(observed_paths | nonzero_paths | set(path_to_npu))
    _require(relevant_paths, f"{case.name}: no CIR paths to report")
    _require(
        min(relevant_paths) >= 0 and max(relevant_paths) < table_width,
        f"{case.name}: observed path outside CIR table",
    )
    for bin_row in bins:
        for ssu_id in range(case.num_ssu):
            for path_id in relevant_paths:
                npu_id = path_to_npu.get(path_id)
                integrated = 0.0
                ssu_total_integrated = 0.0
                npu_total_integrated = 0.0
                change_events = 0
                for segment in segments:
                    overlap_ms = _overlap(segment["start_ms"], segment["end_ms"], bin_row.start_ms, bin_row.end_ms)
                    if overlap_ms <= 0.0:
                        continue
                    table = segment["table"]
                    integrated += float(table[ssu_id][path_id]) * overlap_ms
                    ssu_total_integrated += math.fsum(map(float, table[ssu_id])) * overlap_ms
                    if npu_id is not None:
                        npu_total_integrated += math.fsum(
                            float(table[current_ssu][path_id])
                            for current_ssu in range(case.num_ssu)
                        ) * overlap_ms
                    change_events += int(segment["event"].get("changed_entry_count", 0) > 0)
                rows.append(
                    {
                        "case": case.name,
                        "num_ssu": case.num_ssu,
                        "bin": bin_row.index,
                        "relative_start_ms": bin_row.relative_start_ms,
                        "relative_end_ms": bin_row.relative_end_ms,
                        "absolute_start_ms": bin_row.start_ms,
                        "absolute_end_ms": bin_row.end_ms,
                        "duration_ms": bin_row.duration_ms,
                        "npu_id": npu_id,
                        "ssu_id": ssu_id,
                        "path_id": path_id,
                        "path_scope": "nonzero_or_observed_or_dedicated",
                        "installed_cir_gbps": integrated / bin_row.duration_ms,
                        "installed_total_cir_gbps_on_ssu": ssu_total_integrated / bin_row.duration_ms,
                        "installed_total_cir_gbps_for_npu": (
                            None
                            if npu_id is None
                            else npu_total_integrated / bin_row.duration_ms
                        ),
                        "cir_change_segments_overlapping_bin": change_events,
                    }
                )
    return rows


def _max_piecewise_rate(intervals: Sequence[ServiceInterval], key_field: str) -> dict[int, float]:
    grouped: dict[int, list[tuple[float, int, float]]] = {}
    for item in intervals:
        key = getattr(item, key_field)
        _require(key is not None, f"{item.case}.{item.resource}: missing {key_field}")
        # End sorts before start at the same time: intervals are half-open.
        grouped.setdefault(key, []).append((item.start_ms, 1, item.rate))
        grouped[key].append((item.end_ms, 0, -item.rate))
    maxima: dict[int, float] = {}
    for key, events in grouped.items():
        current = 0.0
        maximum = 0.0
        for _, _, delta in sorted(events, key=lambda event: (event[0], event[1])):
            current += delta
            _require(current >= -1e-7, f"negative sweep state for {key_field}={key}")
            maximum = max(maximum, current)
        _require(abs(current) <= 1e-7, f"unclosed sweep state for {key_field}={key}")
        maxima[key] = maximum
    return maxima


def _capacity_validation(case: CaseData, intervals: Sequence[ServiceInterval]) -> dict:
    compute = _intervals_for(intervals, "compute")
    links = _intervals_for(intervals, "link")
    ssds = _intervals_for(intervals, "ssd")
    compute_max = _max_piecewise_rate(compute, "npu_id")
    link_max = _max_piecewise_rate(links, "npu_id")
    ssd_max = _max_piecewise_rate(ssds, "ssu_id")
    cir_ssu_max = [0.0] * case.num_ssu
    cir_npu_max: list[float] | None = [0.0] * 32
    for segment in _cir_segments(case):
        table = segment["table"]
        paths = segment["paths"]
        for ssu_id in range(case.num_ssu):
            cir_ssu_max[ssu_id] = max(cir_ssu_max[ssu_id], math.fsum(map(float, table[ssu_id])))
        if paths is None:
            cir_npu_max = None
        else:
            assert cir_npu_max is not None
            for npu_id, path_id in enumerate(paths):
                cir_npu_max[npu_id] = max(
                    cir_npu_max[npu_id],
                    math.fsum(
                        float(table[ssu_id][path_id])
                        for ssu_id in range(case.num_ssu)
                    ),
                )
    checks = {
        "compute_per_npu_at_most_one": max(compute_max.values(), default=0.0) <= 1.0 + 1e-8,
        "npu_link_per_npu_at_most_50_gbps": max(link_max.values(), default=0.0) <= 50.0 + 1e-8,
        "ssd_per_ssu_at_most_40_gbps": max(ssd_max.values(), default=0.0) <= 40.0 + 1e-8,
        "cir_sum_per_ssu_at_most_40_gbps": max(cir_ssu_max, default=0.0) <= 40.0 + 1e-8,
        "dedicated_cir_sum_per_npu_at_most_50_gbps": (
            cir_npu_max is None
            or max(cir_npu_max, default=0.0) <= 50.0 + 1e-8
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_compute_concurrency_by_npu": compute_max,
        "max_npu_link_bandwidth_gbps_by_npu": link_max,
        "max_ssd_bandwidth_gbps_by_ssu": ssd_max,
        "max_installed_cir_sum_gbps_by_ssu": cir_ssu_max,
        "max_installed_dedicated_cir_sum_gbps_by_npu": cir_npu_max,
        "dedicated_npu_cir_check_applicable": cir_npu_max is not None,
    }


def _vector_integral(
    intervals: Sequence[ServiceInterval],
    resource: str,
    count: int,
    id_field: str,
    start_ms: float,
    end_ms: float,
    *,
    volume: bool = False,
) -> list[float]:
    values = [0.0] * count
    for item in intervals:
        if item.resource != resource:
            continue
        identifier = getattr(item, id_field)
        _require(identifier is not None and 0 <= identifier < count, f"bad {id_field}")
        overlap_ms = _overlap(item.start_ms, item.end_ms, start_ms, end_ms)
        values[identifier] += overlap_ms * (item.rate / 1000.0 if volume else 1.0)
    return values


def _max_abs_residual(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "residual vector shape differs")
    return max([0.0] + [abs(float(a) - float(b)) for a, b in zip(left, right)])


def _closure_validation(
    case: CaseData,
    intervals: Sequence[ServiceInterval],
    bins: Sequence[Bin],
    npu_rows: Sequence[dict],
    ssu_rows: Sequence[dict],
) -> dict:
    trace_start = float(case.trace["trace_start_ms"])
    trace_end = float(case.trace["trace_end_ms"])
    blocks = [
        row
        for row in case.summary["measurement_blocks"]
        if float(row["start_ms"]) >= trace_start - EPS
        and float(row["end_ms"]) <= trace_end + EPS
    ]
    expected_compute = [
        math.fsum(float(row["compute_ms_by_npu"][index]) for row in blocks)
        for index in range(32)
    ]
    expected_link_ms = [
        math.fsum(float(row["npu_link_busy_ms_by_npu"][index]) for row in blocks)
        for index in range(32)
    ]
    expected_link_gb = [
        math.fsum(float(row["npu_link_served_gb_by_npu"][index]) for row in blocks)
        for index in range(32)
    ]
    expected_ssd_ms = [
        math.fsum(float(row["ssd_busy_ms_by_ssu"][index]) for row in blocks)
        for index in range(case.num_ssu)
    ]
    expected_ssd_gb = [
        math.fsum(float(row["ssd_served_gb_by_ssu"][index]) for row in blocks)
        for index in range(case.num_ssu)
    ]
    actual_compute = _vector_integral(intervals, "compute", 32, "npu_id", trace_start, trace_end)
    actual_link_ms = _vector_integral(intervals, "link", 32, "npu_id", trace_start, trace_end)
    actual_link_gb = _vector_integral(intervals, "link", 32, "npu_id", trace_start, trace_end, volume=True)
    actual_ssd_ms = _vector_integral(
        intervals, "ssd", case.num_ssu, "ssu_id", trace_start, trace_end
    )
    actual_ssd_gb = _vector_integral(
        intervals,
        "ssd",
        case.num_ssu,
        "ssu_id",
        trace_start,
        trace_end,
        volume=True,
    )

    binned_compute = [
        math.fsum(float(row["compute_busy_ms"]) for row in npu_rows if int(row["npu_id"]) == npu_id)
        for npu_id in range(32)
    ]
    binned_link_ms = [
        math.fsum(float(row["npu_link_busy_ms"]) for row in npu_rows if int(row["npu_id"]) == npu_id)
        for npu_id in range(32)
    ]
    binned_link_gb = [
        math.fsum(float(row["npu_link_served_gb"]) for row in npu_rows if int(row["npu_id"]) == npu_id)
        for npu_id in range(32)
    ]
    binned_ssd_ms = [
        math.fsum(float(row["ssd_busy_ms"]) for row in ssu_rows if int(row["ssu_id"]) == ssu_id)
        for ssu_id in range(case.num_ssu)
    ]
    binned_ssd_gb = [
        math.fsum(float(row["ssd_served_gb"]) for row in ssu_rows if int(row["ssu_id"]) == ssu_id)
        for ssu_id in range(case.num_ssu)
    ]
    coverage_ms = math.fsum(float(row["duration_ms"]) for row in blocks)
    trace_duration = trace_end - trace_start
    residuals = {
        "trace_vs_5ms_compute_ms": _max_abs_residual(actual_compute, expected_compute),
        "trace_vs_5ms_npu_link_busy_ms": _max_abs_residual(actual_link_ms, expected_link_ms),
        "trace_vs_5ms_npu_link_served_gb": _max_abs_residual(actual_link_gb, expected_link_gb),
        "trace_vs_5ms_ssd_busy_ms": _max_abs_residual(actual_ssd_ms, expected_ssd_ms),
        "trace_vs_5ms_ssd_served_gb": _max_abs_residual(actual_ssd_gb, expected_ssd_gb),
        "trace_vs_0p5ms_compute_ms": _max_abs_residual(actual_compute, binned_compute),
        "trace_vs_0p5ms_npu_link_busy_ms": _max_abs_residual(actual_link_ms, binned_link_ms),
        "trace_vs_0p5ms_npu_link_served_gb": _max_abs_residual(actual_link_gb, binned_link_gb),
        "trace_vs_0p5ms_ssd_busy_ms": _max_abs_residual(actual_ssd_ms, binned_ssd_ms),
        "trace_vs_0p5ms_ssd_served_gb": _max_abs_residual(actual_ssd_gb, binned_ssd_gb),
    }
    checks = {
        "5ms_blocks_exactly_cover_trace": math.isclose(coverage_ms, trace_duration, abs_tol=1e-8),
        "exact_trace_matches_5ms_summary": max(value for key, value in residuals.items() if "5ms" in key) <= 1e-7,
        "0p5ms_bins_reintegrate_exact_trace": max(value for key, value in residuals.items() if "0p5ms" in key) <= 1e-9,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "trace_duration_ms": trace_duration,
        "summary_block_coverage_ms": coverage_ms,
        "summary_block_count": len(blocks),
        "trace_mean_npu_compute_utilization": math.fsum(actual_compute)
        / (32.0 * trace_duration),
        "trace_mean_npu_link_utilization": math.fsum(actual_link_ms)
        / (32.0 * trace_duration),
        "trace_mean_ssd_utilization": math.fsum(actual_ssd_ms)
        / (case.num_ssu * trace_duration),
        "trace_fleet_npu_link_bandwidth_gbps": math.fsum(actual_link_gb)
        * 1000.0
        / trace_duration,
        "trace_fleet_ssd_bandwidth_gbps": math.fsum(actual_ssd_gb)
        * 1000.0
        / trace_duration,
        "maximum_absolute_residuals": residuals,
    }


def _request_rows(case: CaseData, intervals: Sequence[ServiceInterval]) -> list[dict]:
    trace_start = float(case.trace["trace_start_ms"])
    trace_end = float(case.trace["trace_end_ms"])
    summary_rows = {int(row["request_id"]): row for row in case.summary.get("request_rows", [])}
    trace_ids = {item.request_id for item in intervals if item.request_id is not None}
    all_ids = sorted(set(summary_rows) | trace_ids)
    rows = []
    for request_id in all_ids:
        source = summary_rows.get(request_id, {})
        matching = [item for item in intervals if item.request_id == request_id]
        compute = [item for item in matching if item.resource == "compute"]
        links = [item for item in matching if item.resource == "link"]
        ssds = [item for item in matching if item.resource == "ssd"]
        admission = source.get("admission_time_ms")
        if admission is None and compute:
            admission = compute[0].source.get("admission_time_ms")
        completion = source.get("completion_time_ms")
        ttft = source.get("ttft_ms")
        ideal = source.get("ideal_ttft_ms")
        rows.append(
            {
                "case": case.name,
                "request_id": request_id,
                "npu_id": source.get("npu_id", matching[0].npu_id if matching else None),
                "sequence": source.get("sequence"),
                "category": source.get("category"),
                "profile_id": source.get("profile_id"),
                "admission_time_ms": admission,
                "completion_time_ms": completion,
                "admission_relative_ms": None if admission is None else float(admission) - trace_start,
                "completion_relative_ms": None if completion is None else float(completion) - trace_start,
                "ttft_ms": ttft,
                "ideal_ttft_ms": ideal,
                "io_stall_ms": None if ttft is None or ideal is None else float(ttft) - float(ideal),
                "request_compute_fraction": None if ttft in (None, 0) or ideal is None else float(ideal) / float(ttft),
                "slo_met_alpha1p5": None if ttft is None or ideal is None else float(ttft) <= 1.5 * float(ideal) + EPS,
                "slo_met_alpha2": None if ttft is None or ideal is None else float(ttft) <= 2.0 * float(ideal) + EPS,
                "trace_compute_interval_count": len(compute),
                "trace_link_interval_count": len(links),
                "trace_ssd_interval_count": len(ssds),
                "has_any_trace_activity": bool(matching),
                "request_fully_inside_trace": admission is not None and completion is not None and trace_start <= float(admission) and float(completion) <= trace_end,
            }
        )
    return rows


def _slo_metrics(summary: dict) -> dict:
    """Recompute both SLO thresholds from raw TTFT, equal-weighting NPUs."""
    outcomes = {
        1.5: [[] for _ in range(32)],
        2.0: [[] for _ in range(32)],
    }
    request_rows = summary.get("request_rows", [])
    _require(isinstance(request_rows, list) and request_rows, "summary has no requests")
    for index, row in enumerate(request_rows):
        npu_id = _integer(row.get("npu_id"), f"request_rows[{index}].npu_id")
        _require(0 <= npu_id < 32, f"request_rows[{index}]: invalid NPU")
        ttft = _number(row.get("ttft_ms"), f"request_rows[{index}].ttft_ms")
        ideal = _number(
            row.get("ideal_ttft_ms"), f"request_rows[{index}].ideal_ttft_ms"
        )
        _require(ideal > 0.0 and ttft >= 0.0, "invalid TTFT/ideal TTFT")
        for alpha in outcomes:
            outcomes[alpha][npu_id].append(ttft <= alpha * ideal + EPS)
    _require(
        all(rows for by_npu in outcomes.values() for rows in by_npu),
        "SLO equal-NPU recomputation requires at least one request per NPU",
    )

    def mean(values: Sequence[bool | float]) -> float:
        return math.fsum(float(value) for value in values) / len(values)

    result = {"request_count": len(request_rows)}
    for alpha, by_npu in outcomes.items():
        suffix = "1p5" if alpha == 1.5 else "2"
        per_npu = [mean(rows) for rows in by_npu]
        result[f"alpha{suffix}_equal_npu_slo_attainment"] = mean(per_npu)
        result[f"alpha{suffix}_request_weighted_slo_attainment"] = mean(
            [value for rows in by_npu for value in rows]
        )
        result[f"alpha{suffix}_per_npu_slo_attainment"] = per_npu
    primary = summary.get("slo_alpha")
    result["summary_primary_slo_alpha"] = primary
    result["summary_primary_equal_npu_slo_attainment"] = summary.get(
        "ttft_slo_attainment"
    )
    result["summary_alpha2_matches_recomputation"] = (
        float(primary) == 2.0
        and math.isclose(
            float(summary["ttft_slo_attainment"]),
            result["alpha2_equal_npu_slo_attainment"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    return result


def _per_npu_measurement_stats(cases: Sequence[CaseData]) -> dict:
    summaries = {item.name: item.summary for item in cases}
    baseline = list(map(float, summaries["baseline"]["npu_utilizations"]))
    adaptive = list(
        map(float, summaries["adaptive_t0_i100ms"]["npu_utilizations"])
    )
    _require(len(baseline) == 32 and len(adaptive) == 32, "per-NPU shape mismatch")
    deltas = [right - left for left, right in zip(baseline, adaptive)]

    def mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    def population_std(values: Sequence[float]) -> float:
        center = mean(values)
        return math.sqrt(math.fsum((value - center) ** 2 for value in values) / len(values))

    baseline_mean = mean(baseline)
    adaptive_mean = mean(adaptive)
    baseline_std = population_std(baseline)
    adaptive_std = population_std(adaptive)
    covariance = math.fsum(
        (left - baseline_mean) * (right - adaptive_mean)
        for left, right in zip(baseline, adaptive)
    ) / len(baseline)
    correlation = (
        None
        if baseline_std <= EPS or adaptive_std <= EPS
        else covariance / (baseline_std * adaptive_std)
    )
    return {
        "baseline_mean": baseline_mean,
        "adaptive_mean": adaptive_mean,
        "mean_delta_adaptive_minus_baseline": mean(deltas),
        "mean_absolute_per_npu_delta": mean([abs(value) for value in deltas]),
        "maximum_absolute_per_npu_delta": max(abs(value) for value in deltas),
        "maximum_increase": max(deltas),
        "maximum_decrease": min(deltas),
        "npu_count_increased": sum(value > 1e-12 for value in deltas),
        "npu_count_decreased": sum(value < -1e-12 for value in deltas),
        "npu_count_unchanged": sum(abs(value) <= 1e-12 for value in deltas),
        "baseline_population_std": baseline_std,
        "adaptive_population_std": adaptive_std,
        "pearson_correlation": correlation,
        "baseline_by_npu": baseline,
        "adaptive_by_npu": adaptive,
        "delta_by_npu": deltas,
    }


def _historical_formal_rows(
    path: Path,
    cases: Sequence[CaseData],
) -> tuple[list[dict], dict]:
    """Load the frozen 8 s run and prove that the new 3 s run is its prefix."""
    source_path = path.expanduser().resolve()
    source_label = _portable_path(source_path)
    artifact = _load_json(source_path)
    results = artifact.get("results")
    _require(isinstance(results, list), f"{source_path}: missing results")
    num_ssu = cases[0].num_ssu
    current = {item.name: item.summary for item in cases}
    rows: list[dict] = []
    case_audits: dict[str, dict] = {}
    for case_name in EXPECTED_CASES:
        matches = [
            row
            for row in results
            if row.get("case") == case_name
            and int(row.get("num_ssu", -1)) == num_ssu
        ]
        _require(
            len(matches) == 1,
            f"{source_path}: expected one {case_name}/SSU{num_ssu} result",
        )
        result = matches[0]
        _require(result.get("status") == "ok", f"{case_name}: historical status")
        summary = result.get("steady_summary")
        _require(isinstance(summary, dict), f"{case_name}: missing historical summary")
        blocks = summary.get("measurement_blocks")
        _require(
            isinstance(blocks, list) and len(blocks) == 16,
            f"{case_name}: expected 16 historical blocks",
        )
        durations = [
            float(block["end_ms"]) - float(block["start_ms"]) for block in blocks
        ]
        _require(
            all(math.isclose(value, 500.0, abs_tol=1e-8) for value in durations),
            f"{case_name}: historical blocks are not 500 ms",
        )
        _require(
            all(
                math.isclose(
                    float(previous["end_ms"]),
                    float(next_block["start_ms"]),
                    abs_tol=EPS,
                )
                for previous, next_block in zip(blocks, blocks[1:])
            ),
            f"{case_name}: historical blocks are not contiguous",
        )
        historical_fingerprint = summary.get("input_fingerprint")
        current_fingerprint = current[case_name].get("input_fingerprint")
        input_fingerprint_equal = historical_fingerprint == current_fingerprint
        measurement_start_equal = math.isclose(
            float(summary["measurement_start_ms"]),
            float(current[case_name]["measurement_start_ms"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        cumulative_busy_fraction_ms = 0.0
        cumulative_duration_ms = 0.0
        case_rows = []
        for index, (block, duration_ms) in enumerate(zip(blocks, durations)):
            raw = _number(
                block.get("npu_utilization"),
                f"historical.{case_name}.block[{index}].npu_utilization",
            )
            _require(0.0 <= raw <= 1.0 + EPS, "historical utilization bounds")
            cumulative_busy_fraction_ms += raw * duration_ms
            cumulative_duration_ms += duration_ms
            cumulative = cumulative_busy_fraction_ms / cumulative_duration_ms
            row = {
                "case": case_name,
                "num_ssu": num_ssu,
                "block": index,
                "relative_start_ms": 500.0 * index,
                "relative_end_ms": 500.0 * (index + 1),
                "relative_mid_ms": 500.0 * (index + 0.5),
                "absolute_start_ms": block["start_ms"],
                "absolute_end_ms": block["end_ms"],
                "duration_ms": duration_ms,
                "raw_500ms_mean_npu_utilization_pct": 100.0 * raw,
                "cumulative_mean_npu_utilization_pct": 100.0 * cumulative,
                "is_3s_checkpoint": index == 5,
                "is_8s_endpoint": index == 15,
                "new_3s_summary_mean_npu_utilization_pct": 100.0
                * float(current[case_name]["mean_npu_utilization"]),
                "source_artifact": source_label,
            }
            case_rows.append(row)
            rows.append(row)

        first_3s = float(
            next(row for row in case_rows if row["is_3s_checkpoint"])[
                "cumulative_mean_npu_utilization_pct"
            ]
        ) / 100.0
        formal_8s = float(case_rows[-1]["cumulative_mean_npu_utilization_pct"]) / 100.0
        new_3s = float(current[case_name]["mean_npu_utilization"])
        reported_8s = float(summary["mean_npu_utilization"])
        case_audits[case_name] = {
            "input_fingerprint_equal": input_fingerprint_equal,
            "measurement_start_equal": measurement_start_equal,
            "new_3s_mean_npu_utilization": new_3s,
            "historical_first_6_blocks_mean_npu_utilization": first_3s,
            "new_3s_minus_historical_prefix_residual": new_3s - first_3s,
            "historical_8s_cumulative_mean_npu_utilization": formal_8s,
            "historical_reported_mean_npu_utilization": reported_8s,
            "historical_8s_cumulative_residual": formal_8s - reported_8s,
            "passed": input_fingerprint_equal
            and measurement_start_equal
            and math.isclose(new_3s, first_3s, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(formal_8s, reported_8s, rel_tol=0.0, abs_tol=1e-12),
        }

    baseline = case_audits["baseline"]
    adaptive = case_audits["adaptive_t0_i100ms"]
    audit = {
        "passed": all(value["passed"] for value in case_audits.values()),
        "source_artifact": source_label,
        "num_ssu": num_ssu,
        "block_count_per_case": 16,
        "block_duration_ms": 500.0,
        "cases": case_audits,
        "adaptive_minus_baseline_3s_percentage_points": 100.0
        * (
            adaptive["new_3s_mean_npu_utilization"]
            - baseline["new_3s_mean_npu_utilization"]
        ),
        "adaptive_minus_baseline_8s_percentage_points": 100.0
        * (
            adaptive["historical_8s_cumulative_mean_npu_utilization"]
            - baseline["historical_8s_cumulative_mean_npu_utilization"]
        ),
    }
    return rows, audit


def _layer_rows(case: CaseData, intervals: Sequence[ServiceInterval]) -> list[dict]:
    trace_start = float(case.trace["trace_start_ms"])
    trace_end = float(case.trace["trace_end_ms"])
    compute = [item for item in intervals if item.resource == "compute"]
    by_request_layer = {(item.request_id, item.layer): item for item in compute}
    rows = []
    for item in sorted(compute, key=lambda value: (value.npu_id, value.start_ms, value.layer)):
        links = [
            value
            for value in intervals
            if value.resource == "link"
            and value.request_id == item.request_id
            and value.layer == item.layer
        ]
        ssds = [
            value
            for value in intervals
            if value.resource == "ssd"
            and value.request_id == item.request_id
            and value.layer == item.layer
        ]
        source = item.source
        io_ready_reported = _number(source.get("io_ready_time_ms"), "compute.io_ready")
        max_link_end = max((value.end_ms for value in links), default=float("nan"))
        if item.layer == 0:
            previous_compute_end = _number(source.get("admission_time_ms"), "compute.admission")
        else:
            previous = by_request_layer.get((item.request_id, item.layer - 1))
            previous_compute_end = None if previous is None else previous.end_ms
        reconstructed_barrier = (
            None if previous_compute_end is None else item.start_ms - previous_compute_end
        )
        reported_barrier = _number(source.get("io_barrier_wait_ms"), "compute.barrier")
        complete = (
            bool(links)
            and bool(ssds)
            and previous_compute_end is not None
            and min(value.source.get("ssd_enqueue_time_ms", value.start_ms) for value in ssds) >= trace_start - EPS
            and item.end_ms <= trace_end + EPS
            and all(value.start_ms >= trace_start - EPS and value.end_ms <= trace_end + EPS for value in links + ssds)
        )
        expected_start = (
            None
            if previous_compute_end is None or not links
            else max(max_link_end, previous_compute_end)
        )
        rows.append(
            {
                "case": case.name,
                "request_id": item.request_id,
                "batch_id": item.batch_id,
                "npu_id": item.npu_id,
                "layer": item.layer,
                "io_submit_time_ms": min((float(value.source["ssd_enqueue_time_ms"]) for value in ssds), default=None),
                "io_submit_relative_ms": min((float(value.source["ssd_enqueue_time_ms"]) - trace_start for value in ssds), default=None),
                "ssd_first_start_ms": min((value.start_ms for value in ssds), default=None),
                "ssd_last_end_ms": max((value.end_ms for value in ssds), default=None),
                "max_link_end_ms": None if not links else max_link_end,
                "io_ready_reported_ms": io_ready_reported,
                "previous_compute_end_ms": previous_compute_end,
                "compute_start_ms": item.start_ms,
                "compute_end_ms": item.end_ms,
                "reported_barrier_ms": reported_barrier,
                "reconstructed_barrier_ms": reconstructed_barrier,
                "io_ready_minus_max_link_end_ms": None if not links else io_ready_reported - max_link_end,
                "compute_start_minus_max_ready_previous_ms": None if expected_start is None else item.start_ms - expected_start,
                "barrier_accounting_error_ms": None if reconstructed_barrier is None else reported_barrier - reconstructed_barrier,
                "ssd_interval_count": len(ssds),
                "link_interval_count": len(links),
                "complete_inside_trace": complete,
            }
        )
    return rows


def _select_layer(rows: Sequence[dict], case_name: str) -> dict:
    _require(case_name in EXPECTED_CASES, f"unknown selection case: {case_name}")
    candidates = [
        row
        for row in rows
        if row["complete_inside_trace"]
        and row["case"] == case_name
        and row["io_ready_minus_max_link_end_ms"] is not None
        and abs(float(row["io_ready_minus_max_link_end_ms"])) <= 1e-7
        and abs(float(row["compute_start_minus_max_ready_previous_ms"])) <= 1e-7
        and abs(float(row["barrier_accounting_error_ms"])) <= 1e-7
    ]
    _require(
        candidates,
        f"{case_name}: no complete trace layer satisfies the causal equations",
    )
    return max(
        candidates,
        key=lambda row: (
            float(row["reported_barrier_ms"]),
            int(row["layer"]),
        ),
    )


def _service_rows(intervals: Sequence[ServiceInterval], trace_starts: dict[str, float]) -> list[dict]:
    rows = []
    for item in sorted(intervals, key=lambda value: (value.case, value.start_ms, value.resource)):
        clipped_start = max(item.start_ms, trace_starts[item.case])
        clipped_end = min(item.end_ms, float(item.source["trace_overlap_end_ms"]))
        rows.append(
            {
                "case": item.case,
                "resource": item.resource,
                "request_id": item.request_id,
                "batch_id": item.batch_id,
                "npu_id": item.npu_id,
                "ssu_id": item.ssu_id,
                "layer": item.layer,
                "block_idx": item.block_idx,
                "start_time_ms": item.start_ms,
                "end_time_ms": item.end_ms,
                "duration_ms": item.duration_ms,
                "relative_start_ms": item.start_ms - trace_starts[item.case],
                "relative_end_ms": item.end_ms - trace_starts[item.case],
                "trace_overlap_start_ms": clipped_start,
                "trace_overlap_end_ms": clipped_end,
                "trace_overlap_ms": max(0.0, clipped_end - clipped_start),
                "actual_rate_gbps_or_compute_units": item.rate,
                "trace_served_gb": None if item.resource == "compute" else item.rate * max(0.0, clipped_end - clipped_start) / 1000.0,
                "size_gb": item.size_gb,
                "path_id": item.source.get("path_id"),
                "enqueue_time_ms": item.source.get("link_enqueue_time_ms", item.source.get("ssd_enqueue_time_ms")),
                "queue_wait_ms": item.source.get("link_queue_wait_ms", item.source.get("ssd_queue_wait_ms")),
                "installed_path_cir_gbps_at_dispatch": item.source.get("installed_path_cir_gbps_at_ssd_dispatch", item.source.get("installed_path_cir_gbps_at_dispatch")),
            }
        )
    return rows


def _selected_layer_service_rows(
    intervals: Sequence[ServiceInterval],
    selected_layers: dict[str, dict],
    trace_starts: dict[str, float],
) -> list[dict]:
    keys = {
        (
            case_name,
            int(selected["request_id"]),
            int(selected["layer"]),
        )
        for case_name, selected in selected_layers.items()
    }
    selected_intervals = [
        item
        for item in intervals
        if item.request_id is not None
        and item.layer is not None
        and (item.case, item.request_id, item.layer) in keys
    ]
    _require(
        {item.resource for item in selected_intervals}
        == {"ssd", "link", "compute"},
        "selected layer lightweight trace lacks a resource type",
    )
    return _service_rows(selected_intervals, trace_starts)


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    _require(bool(rows), f"refusing to write empty CSV: {path.name}")
    fields = list(rows[0])
    for row in rows:
        _require(set(row) == set(fields), f"{path.name}: inconsistent row shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
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
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#b0b0b0",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.fancybox": True,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, format="png", dpi=180, facecolor="white", metadata={"Title": path.stem})
    plt.close(fig)
    temporary.replace(path)


def _plot_fleet(
    fleet_rows: Sequence[dict], num_ssu: int, output: Path
) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.2), sharex=True)
    for case in EXPECTED_CASES:
        rows = [row for row in fleet_rows if row["case"] == case]
        x = [(float(row["relative_start_ms"]) + float(row["relative_end_ms"])) / 2.0 for row in rows]
        axes[0].plot(x, [row["mean_npu_compute_utilization_pct"] for row in rows], label=CASE_LABELS[case], color=CASE_COLORS[case], linewidth=2.0)
        axes[1].plot(x, [row["mean_ssd_utilization_pct"] for row in rows], label=CASE_LABELS[case], color=CASE_COLORS[case], linewidth=2.0)
    axes[0].set(
        ylabel="Mean NPU utilization (%)",
        ylim=(0, 101),
        title=f"32-NPU / {num_ssu}-SSU exact 0.5 ms resource trace",
    )
    axes[1].set(xlabel="Time since measurement start (ms)", ylabel="Mean SSD utilization (%)", ylim=(0, 101))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, output)


def _plot_heatmap(npu_rows: Sequence[dict], output: Path) -> None:
    _style()
    fig = plt.figure(figsize=(10.0, 8.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 0.025),
        left=0.08,
        right=0.94,
        bottom=0.08,
        top=0.91,
        hspace=0.24,
        wspace=0.08,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    axes[1].sharex(axes[0])
    colorbar_axis = fig.add_subplot(grid[:, 1])
    image = None
    for axis, case in zip(axes, EXPECTED_CASES):
        rows = [row for row in npu_rows if row["case"] == case]
        bin_count = 1 + max(int(row["bin"]) for row in rows)
        matrix = [[0.0] * bin_count for _ in range(32)]
        for row in rows:
            matrix[int(row["npu_id"])][int(row["bin"])] = float(row["compute_utilization_pct"])
        duration = max(float(row["relative_end_ms"]) for row in rows)
        image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=100.0, extent=(0.0, duration, -0.5, 31.5))
        axis.set(ylabel="NPU ID", title=CASE_LABELS[case])
    axes[-1].set_xlabel("Time since measurement start (ms)")
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Compute utilization (%)")
    fig.suptitle("Per-NPU compute occupancy (exact interval overlap)", y=0.99)
    _save_figure(fig, output)


def _micro_bins(start_ms: float, end_ms: float, width_ms: float = 0.02) -> list[tuple[float, float]]:
    count = max(1, int(math.ceil((end_ms - start_ms) / width_ms)))
    return [(start_ms + index * width_ms, min(end_ms, start_ms + (index + 1) * width_ms)) for index in range(count)]


def _cir_at(case: CaseData, time_ms: float, ssu_id: int, path_id: int) -> float:
    segments = _cir_segments(case)
    for segment in reversed(segments):
        if segment["start_ms"] <= time_ms + EPS and time_ms < segment["end_ms"]:
            return float(segment["table"][ssu_id][path_id])
    raise AnalysisError("selected time is outside CIR trace")


def _plot_selected_layer(
    selected: dict,
    cases: Sequence[CaseData],
    all_intervals: Sequence[ServiceInterval],
    output: Path,
) -> None:
    case = next(item for item in cases if item.name == selected["case"])
    request_id = int(selected["request_id"])
    layer = int(selected["layer"])
    npu_id = int(selected["npu_id"])
    matching = [item for item in all_intervals if item.case == case.name and item.request_id == request_id and item.layer == layer]
    start = min(item.start_ms for item in matching) - 0.15
    end = max(item.end_ms for item in matching) + 0.15
    start = max(start, float(case.trace["trace_start_ms"]))
    end = min(end, float(case.trace["trace_end_ms"]))
    bins = _micro_bins(start, end)
    x = [((left + right) / 2.0) - float(case.trace["trace_start_ms"]) for left, right in bins]
    links = [item for item in matching if item.resource == "link"]
    ssds = [item for item in matching if item.resource == "ssd"]
    compute = [item for item in matching if item.resource == "compute"]
    relevant_ssu_paths = sorted(
        {
            (item.ssu_id, int(item.source["path_id"]))
            for item in ssds
            if item.ssu_id is not None and item.source.get("path_id") is not None
        }
    )

    def bandwidth(rows: Sequence[ServiceInterval], left: float, right: float) -> float:
        return math.fsum(item.rate * _overlap(item.start_ms, item.end_ms, left, right) / (right - left) for item in rows)

    link_y = [bandwidth(links, left, right) for left, right in bins]
    compute_y = [100.0 * math.fsum(_overlap(item.start_ms, item.end_ms, left, right) for item in compute) / (right - left) for left, right in bins]
    _style()
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.4), sharex=True)
    axes[0].plot(x, link_y, color="#1f77b4", linewidth=2.0, label=f"NPU {npu_id} link")
    for offset, (ssu_id, path_id) in enumerate(relevant_ssu_paths):
        rows = [
            item
            for item in ssds
            if item.ssu_id == ssu_id and int(item.source["path_id"]) == path_id
        ]
        axes[0].plot(x, [bandwidth(rows, left, right) for left, right in bins], linewidth=1.4, label=f"SSU {ssu_id}", color=plt.cm.tab10(offset))
    axes[0].set(
        ylabel="Selected-layer service\n(GB/s; 0.02 ms bin avg)",
        title=(
            f"Selected complete layer: {CASE_LABELS[case.name]}, request "
            f"{request_id}, NPU {npu_id}, layer {layer}"
        ),
    )
    axes[0].text(
        0.99,
        0.04,
        (
            "Selected layer only; 0.02 ms bin average\n"
            "Not whole-device instantaneous bandwidth\n"
            "Active SSD command: 40 GB/s | active NPU link: 50 GB/s"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.82},
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].fill_between(x, compute_y, step="mid", color="#2ca02c", alpha=0.7)
    axes[1].set(ylabel="NPU compute\noccupancy (%)", ylim=(0, 105))
    axes[1].grid(alpha=0.3)
    for offset, (ssu_id, path_id) in enumerate(relevant_ssu_paths):
        axes[2].plot(
            x,
            [
                _cir_at(case, (left + right) / 2.0, ssu_id, path_id)
                for left, right in bins
            ],
            linewidth=1.8,
            label=f"SSU {ssu_id} / Path {path_id}",
            color=plt.cm.tab10(offset),
        )
    axes[2].set(xlabel="Time since measurement start (ms)", ylabel="Installed CIR\n(GB/s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(ncol=3, fontsize=8)
    marker_points = sorted(
        (
            (float(selected["max_link_end_ms"]), "I/O ready"),
            (float(selected["previous_compute_end_ms"]), "Previous compute end"),
            (float(selected["compute_start_ms"]), "Compute start"),
        )
    )
    marker_groups: list[tuple[float, list[str]]] = []
    for absolute, label in marker_points:
        if marker_groups and math.isclose(
            marker_groups[-1][0], absolute, rel_tol=0.0, abs_tol=1e-8
        ):
            marker_groups[-1][1].append(label)
        else:
            marker_groups.append((absolute, [label]))
    for absolute, labels in marker_groups:
        label_order = {
            "Previous compute end": 0,
            "I/O ready": 1,
            "Compute start": 2,
        }
        labels.sort(key=label_order.__getitem__)
        label = " = ".join(labels)
        color = (
            "#2ca02c"
            if "Compute start" in labels
            else "#ff7f0e"
            if "Previous compute end" in labels
            else "#d62728"
        )
        relative = float(absolute) - float(case.trace["trace_start_ms"])
        for axis in axes:
            axis.axvline(relative, color=color, linestyle="--", linewidth=1.1, alpha=0.9)
        axes[0].annotate(label, xy=(relative, 0.98), xycoords=("data", "axes fraction"), rotation=90, va="top", ha="right", color=color, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output)


def _plot_full_measurement(
    fleet_rows: Sequence[dict],
    output: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 7.6), sharex=True)
    metric_specs = (
        (
            "raw_5ms_mean_npu_compute_utilization_pct",
            "rolling_100ms_mean_npu_compute_utilization_pct",
            "cumulative_mean_npu_compute_utilization_pct",
            "Mean NPU utilization (%)",
        ),
        (
            "raw_5ms_mean_ssd_utilization_pct",
            "rolling_100ms_mean_ssd_utilization_pct",
            "cumulative_mean_ssd_utilization_pct",
            "Mean SSD utilization (%)",
        ),
    )
    for axis, (raw_field, rolling_field, cumulative_field, ylabel) in zip(
        axes, metric_specs
    ):
        endpoints: list[tuple[float, float, str]] = []
        for case in EXPECTED_CASES:
            rows = [row for row in fleet_rows if row["case"] == case]
            x = [float(row["relative_mid_ms"]) for row in rows]
            color = CASE_COLORS[case]
            axis.plot(
                x,
                [float(row[raw_field]) for row in rows],
                color=color,
                linewidth=0.65,
                alpha=0.22,
            )
            axis.plot(
                x,
                [float(row[rolling_field]) for row in rows],
                color=color,
                linewidth=1.8,
            )
            cumulative = [float(row[cumulative_field]) for row in rows]
            axis.plot(
                x,
                cumulative,
                color=color,
                linewidth=2.1,
                linestyle="--",
            )
            endpoints.append((x[-1], cumulative[-1], color))
        for rank, (last_x, last_value, color) in enumerate(
            sorted(endpoints, key=lambda item: item[1])
        ):
            axis.annotate(
                f"{last_value:.2f}%",
                xy=(last_x, last_value),
                xytext=(-4, -10 if rank == 0 else 6),
                textcoords="offset points",
                ha="right",
                va="top" if rank == 0 else "bottom",
                color=color,
                fontsize=8,
            )
        axis.set(ylabel=ylabel, ylim=(0.0, 101.0))
        axis.grid(alpha=0.3)

    case_handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.2, label=CASE_LABELS[case])
        for case in EXPECTED_CASES
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", linewidth=0.8, alpha=0.35, label="Raw 5 ms"),
        Line2D([0], [0], color="#555555", linewidth=1.8, label="Trailing 100 ms"),
        Line2D([0], [0], color="#555555", linewidth=2.1, linestyle="--", label="Cumulative"),
    ]
    case_legend = axes[0].legend(handles=case_handles, loc="lower left")
    axes[0].add_artist(case_legend)
    axes[0].legend(handles=style_handles, loc="lower right", ncol=3, fontsize=8)
    axes[0].set_title(
        "Full 3 s measurement: native 5 ms blocks, 100 ms rolling, and cumulative duty"
    )
    axes[-1].set_xlabel("Time since measurement start (ms)")
    fig.tight_layout()
    _save_figure(fig, output)


def _plot_per_npu_measurement(stats: dict, output: Path) -> None:
    _style()
    baseline = [100.0 * value for value in stats["baseline_by_npu"]]
    adaptive = [100.0 * value for value in stats["adaptive_by_npu"]]
    deltas = [100.0 * value for value in stats["delta_by_npu"]]
    npu_ids = list(range(32))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    axes[0].plot(
        npu_ids,
        baseline,
        color=CASE_COLORS["baseline"],
        marker="s",
        markersize=4.0,
        linewidth=1.5,
        label=CASE_LABELS["baseline"],
    )
    axes[0].plot(
        npu_ids,
        adaptive,
        color=CASE_COLORS["adaptive_t0_i100ms"],
        marker="o",
        markersize=4.0,
        linewidth=1.5,
        label=CASE_LABELS["adaptive_t0_i100ms"],
    )
    for case, mean in (
        ("baseline", 100.0 * stats["baseline_mean"]),
        ("adaptive_t0_i100ms", 100.0 * stats["adaptive_mean"]),
    ):
        axes[0].axhline(
            mean,
            color=CASE_COLORS[case],
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            label=f"{CASE_LABELS[case]} mean = {mean:.2f}%",
        )
    axes[0].set(
        ylabel="3 s NPU utilization (%)",
        ylim=(
            max(
                0.0,
                5.0
                * math.floor((min(baseline + adaptive) - 5.0) / 5.0),
            ),
            101.0,
        ),
        title="Fleet mean hides per-NPU utilization redistribution",
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", ncol=2, fontsize=8)

    colors = ["#2ca02c" if value >= 0.0 else "#d62728" for value in deltas]
    axes[1].bar(npu_ids, deltas, color=colors, width=0.78, alpha=0.85)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(
        100.0 * stats["mean_delta_adaptive_minus_baseline"],
        color=CASE_COLORS["adaptive_t0_i100ms"],
        linestyle="--",
        linewidth=1.2,
        label=(
            "Fleet-mean delta = "
            f"{100.0 * stats['mean_delta_adaptive_minus_baseline']:+.2f} pp"
        ),
    )
    maximum = max(0.1, max(abs(value) for value in deltas))
    axes[1].set(
        xlabel="NPU ID",
        ylabel="Adaptive - Baseline\n(percentage points)",
        xticks=npu_ids,
        ylim=(-1.15 * maximum, 1.15 * maximum),
    )
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    _save_figure(fig, output)


def _plot_historical_convergence(
    rows: Sequence[dict],
    output: Path,
) -> None:
    _style()
    fig, axis = plt.subplots(figsize=(10.0, 6.2))
    all_values: list[float] = []
    endpoints: list[tuple[float, float, str, str]] = []
    checkpoints: list[tuple[float, float, str, str]] = []
    for case in EXPECTED_CASES:
        case_rows = [row for row in rows if row["case"] == case]
        _require(len(case_rows) == 16, f"{case}: incomplete historical plot rows")
        color = CASE_COLORS[case]
        raw = [float(row["raw_500ms_mean_npu_utilization_pct"]) for row in case_rows]
        cumulative = [
            float(row["cumulative_mean_npu_utilization_pct"]) for row in case_rows
        ]
        raw_x = [float(row["relative_mid_ms"]) for row in case_rows]
        cumulative_x = [float(row["relative_end_ms"]) for row in case_rows]
        all_values.extend(raw + cumulative)
        axis.plot(
            raw_x,
            raw,
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=1.0,
            alpha=0.38,
        )
        axis.plot(
            cumulative_x,
            cumulative,
            color=color,
            marker="s",
            markersize=4.0,
            linewidth=2.2,
        )
        checkpoints.append((3000.0, cumulative[5], color, CASE_LABELS[case]))
        endpoints.append((8000.0, cumulative[-1], color, CASE_LABELS[case]))

    axis.axvline(3000.0, color="#d62728", linestyle="--", linewidth=1.2)
    axis.axvline(8000.0, color="#333333", linestyle=":", linewidth=1.2)
    axis.text(
        3000.0,
        0.02,
        "new 3 s checkpoint",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="bottom",
        color="#d62728",
        fontsize=8,
    )
    axis.text(
        8000.0,
        0.02,
        "formal 8 s endpoint",
        transform=axis.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="bottom",
        color="#333333",
        fontsize=8,
    )
    for points, horizontal_offset in ((checkpoints, 5), (endpoints, -5)):
        for rank, (x, value, color, label) in enumerate(
            sorted(points, key=lambda item: item[1])
        ):
            axis.annotate(
                f"{label}: {value:.3f}%",
                xy=(x, value),
                xytext=(horizontal_offset, -12 if rank == 0 else 8),
                textcoords="offset points",
                ha="left" if horizontal_offset > 0 else "right",
                va="top" if rank == 0 else "bottom",
                color=color,
                fontsize=8,
            )

    case_handles = [
        Line2D([0], [0], color=CASE_COLORS[case], linewidth=2.2, label=CASE_LABELS[case])
        for case in EXPECTED_CASES
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linewidth=1.0,
            alpha=0.45,
            label="Raw 500 ms block",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="s",
            linewidth=2.2,
            label="Cumulative from measurement start",
        ),
    ]
    case_legend = axis.legend(handles=case_handles, loc="lower left")
    axis.add_artist(case_legend)
    axis.legend(handles=style_handles, loc="upper right")
    lower = max(0.0, 5.0 * math.floor((min(all_values) - 5.0) / 5.0))
    upper = min(100.0, 5.0 * math.ceil((max(all_values) + 5.0) / 5.0))
    axis.set(
        xlabel="Time since formal measurement start (ms)",
        ylabel="Mean NPU utilization (%)",
        xlim=(0.0, 8250.0),
        ylim=(lower, upper),
        title="Formal 8 s convergence: the new 3 s result is the exact first-six-block prefix",
    )
    axis.grid(alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output)


def _report(
    cases: Sequence[CaseData],
    validations: dict[str, dict],
    selected_layers: dict[str, dict],
    full_fleet_rows: Sequence[dict],
    per_npu_stats: dict,
    historical_audit: dict | None,
    output: Path,
) -> None:
    summaries = {item.name: item.summary for item in cases}
    num_ssu = cases[0].num_ssu
    baseline = summaries["baseline"]
    adaptive = summaries["adaptive_t0_i100ms"]
    util_base = 100.0 * float(baseline["mean_npu_utilization"])
    util_adaptive = 100.0 * float(adaptive["mean_npu_utilization"])
    trace_util_base = 100.0 * float(
        validations["baseline"]["closure"]["trace_mean_npu_compute_utilization"]
    )
    trace_util_adaptive = 100.0 * float(
        validations["adaptive_t0_i100ms"]["closure"][
            "trace_mean_npu_compute_utilization"
        ]
    )
    trace_ssd_base = 100.0 * float(
        validations["baseline"]["closure"]["trace_mean_ssd_utilization"]
    )
    trace_ssd_adaptive = 100.0 * float(
        validations["adaptive_t0_i100ms"]["closure"][
            "trace_mean_ssd_utilization"
        ]
    )
    full_ssd_base = 100.0 * float(baseline["measurement_ssd_mean_utilization"])
    full_ssd_adaptive = 100.0 * float(
        adaptive["measurement_ssd_mean_utilization"]
    )
    trace_starts = {
        item.name: float(item.trace["trace_start_ms"]) for item in cases
    }

    def cumulative_tail_range(case: str, field: str) -> float:
        rows = [
            row
            for row in full_fleet_rows
            if row["case"] == case and float(row["relative_end_ms"]) >= 2500.0
        ]
        values = [float(row[field]) for row in rows]
        return max(values) - min(values)

    compute_tail_ranges = {
        case: cumulative_tail_range(
            case, "cumulative_mean_npu_compute_utilization_pct"
        )
        for case in EXPECTED_CASES
    }
    ssd_tail_ranges = {
        case: cumulative_tail_range(case, "cumulative_mean_ssd_utilization_pct")
        for case in EXPECTED_CASES
    }
    historical_lines: list[str] = []
    if historical_audit is not None:
        historical_baseline = historical_audit["cases"]["baseline"]
        historical_adaptive = historical_audit["cases"][
            "adaptive_t0_i100ms"
        ]
        historical_lines = [
            "## 3 s 与 frozen formal 8 s 的关系",
            "",
            (
                "新 3 s 结果不是一条不同随机轨迹：它与 frozen formal artifact 的 input fingerprint、"
                "measurement start 完全一致，而且 NPU 利用率数值精确等于旧 8 s 结果前 6 个 "
                "500 ms block 的累计值。"
            ),
            "",
            (
                f"在 3 s 截点，Baseline 为 "
                f"**{100.0 * historical_baseline['new_3s_mean_npu_utilization']:.3f}%**、"
                f"Adaptive 为 **{100.0 * historical_adaptive['new_3s_mean_npu_utilization']:.3f}%**，"
                f"差 **{historical_audit['adaptive_minus_baseline_3s_percentage_points']:+.3f} pp**；"
                f"因此这 {historical_audit['adaptive_minus_baseline_3s_percentage_points']:+.3f} pp "
                "是选取前 3 s 窗口的结果，不是新 trace 改变了仿真。"
            ),
            "",
            (
                f"继续积分到 formal 8 s，Baseline 收敛到 "
                f"**{100.0 * historical_baseline['historical_8s_cumulative_mean_npu_utilization']:.3f}%**、"
                f"Adaptive 收敛到 **{100.0 * historical_adaptive['historical_8s_cumulative_mean_npu_utilization']:.3f}%**，"
                f"差变为 **{historical_audit['adaptive_minus_baseline_8s_percentage_points']:+.3f} pp**。"
                "`06_formal_8s_convergence.png` 展示 500 ms 原始波动及累计平均如何改变排序。"
            ),
            "",
        ]

    causal_lines = [
        "## 请求级因果链示例",
        "",
        "Baseline 与 Adaptive 各自从本策略 trace 自动选取一个完整 layer；两者 measurement 起点和 active cohort 不同，因此不把它们称为同一请求配对。",
        "",
        "`03`/`03b` 顶图只画所选 request/layer 对各资源的服务贡献，并按 0.02 ms interval overlap 求 bin 平均；它不是整台 SSD 或整个 NPU link 的瞬时总带宽。原始模型中，一个正在服务的 SSD 命令固定为 40 GB/s，一个正在服务的 NPU link flow 固定为 50 GB/s。",
        "",
    ]
    for case_name in EXPECTED_CASES:
        selected = selected_layers[case_name]
        causal_lines.extend(
            [
                (
                    f"### {CASE_LABELS[case_name]}：request={selected['request_id']}，"
                    f"NPU={selected['npu_id']}，layer={selected['layer']}"
                ),
                "",
                f"- `io_ready = max(link_end) = {float(selected['max_link_end_ms']):.9f} ms`，与 simulator 字段的误差为 {float(selected['io_ready_minus_max_link_end_ms']):.3e} ms。",
                f"- `compute_start = max(io_ready, previous_compute_end) = {float(selected['compute_start_ms']):.9f} ms`，重建误差为 {float(selected['compute_start_minus_max_ready_previous_ms']):.3e} ms。",
                f"- `barrier = compute_start - previous_compute_end = {float(selected['reconstructed_barrier_ms']):.9f} ms`，与 simulator 字段的误差为 {float(selected['barrier_accounting_error_ms']):.3e} ms。",
                "",
            ]
        )
    slo_base = validations["baseline"]["slo_recalculation"]
    slo_adaptive = validations["adaptive_t0_i100ms"]["slo_recalculation"]
    lines = [
        f"# 32 NPU / {num_ssu} SSU 微观仿真审计",
        "",
        "## 结论",
        "",
        (
            f"精确微观 trace 内的 NPU 平均利用率分别为 Baseline **{trace_util_base:.3f}%**、"
            f"Adaptive 100 ms **{trace_util_adaptive:.3f}%**（差 {trace_util_adaptive - trace_util_base:+.3f} 个百分点）。"
        ),
        "",
        (
            f"这段 50 ms trace 的 SSD 平均利用率差异更大：Baseline **{trace_ssd_base:.3f}%**、"
            f"Adaptive **{trace_ssd_adaptive:.3f}%**。这不是稳态均值的逐点证据，而是两个流水线在各自 measurement 起点的短窗口相位切片。"
        ),
        "",
        (
            f"完整 measurement summary 的 NPU 平均利用率分别为 **{util_base:.3f}%** 和 "
            f"**{util_adaptive:.3f}%**（差 {util_adaptive - util_base:+.3f} 个百分点）；"
            f"3 s SSD 平均利用率分别为 **{full_ssd_base:.3f}%** 和 "
            f"**{full_ssd_adaptive:.3f}%**。"
        ),
        "",
        (
            f"从 `request_rows` 后处理重算的按 NPU 等权 TTFT SLO 为：α=1.5 时 "
            f"**{100.0 * slo_base['alpha1p5_equal_npu_slo_attainment']:.3f}%** 和 "
            f"**{100.0 * slo_adaptive['alpha1p5_equal_npu_slo_attainment']:.3f}%**；"
            f"α=2 时 **{100.0 * slo_base['alpha2_equal_npu_slo_attainment']:.3f}%** 和 "
            f"**{100.0 * slo_adaptive['alpha2_equal_npu_slo_attainment']:.3f}%**。"
        ),
        "",
        "本次仿真的 primary SLO 是 α=2；α=1.5 与旧 SSU 曲线相同，都是在不改变控制器和仿真事件的前提下，使用 raw `ttft_ms <= 1.5 * ideal_ttft_ms` 后处理得到。",
        "",
        "## 为什么 50 ms 不能作逐点相同比较",
        "",
        (
            f"Baseline 与 Adaptive 的稳态 measurement 分别从绝对仿真时间 "
            f"**{trace_starts['baseline']:.6f} ms** 和 "
            f"**{trace_starts['adaptive_t0_i100ms']:.6f} ms** 开始。warmup 达标时间由策略决定，"
            "因此两个相对时间 0 ms 并非同一流水线相位；compute、SSD dispatch 和 link completion 的短周期峰谷可以错开。"
        ),
        "",
        (
            "`04_full_3s_timeseries.png` 同时保留原生 5 ms 点、100 ms trailing rolling 和从 measurement 起点累计的 duty。"
            f"最后 500 ms 内累计 NPU 曲线的摆幅仅为 Baseline **{compute_tail_ranges['baseline']:.3f} pp**、"
            f"Adaptive **{compute_tail_ranges['adaptive_t0_i100ms']:.3f} pp**；"
            f"累计 SSD 曲线摆幅为 **{ssd_tail_ranges['baseline']:.3f} pp** 和 "
            f"**{ssd_tail_ranges['adaptive_t0_i100ms']:.3f} pp**，并在 3 s 终点精确闭合 summary。"
        ),
        "",
        "因此，0.5 ms/50 ms trace 用来证明事件因果关系和计数是否真实闭合；策略的平均资源占用应看完整 3 s 积分，而不是要求两个短 trace 逐点相同。",
        "",
        *historical_lines,
        "## Fleet 平均值掩盖的逐 NPU 重分配",
        "",
        (
            f"Adaptive 相对 Baseline 的 fleet 平均变化是 "
            f"**{100.0 * per_npu_stats['mean_delta_adaptive_minus_baseline']:+.3f} pp**，"
            f"但逐 NPU 变化的平均绝对值为 "
            f"**{100.0 * per_npu_stats['mean_absolute_per_npu_delta']:.3f} pp**，"
            f"最大绝对变化为 **{100.0 * per_npu_stats['maximum_absolute_per_npu_delta']:.3f} pp**。"
        ),
        "",
        (
            f"32 个 NPU 中，{per_npu_stats['npu_count_increased']} 个利用率上升、"
            f"{per_npu_stats['npu_count_decreased']} 个下降、"
            f"{per_npu_stats['npu_count_unchanged']} 个不变。"
            "`05_per_npu_measurement_utilization.png` 展示了这种重分配，说明相近的 fleet mean 不代表每个 NPU 行为相同。"
        ),
        "",
        "微观记录能够闭合仿真内部的资源统计：精确服务区间重新积分后，与 5 ms summary block 及独立的 0.5 ms 分箱一致；瞬时每个 SSU 不超过 40 GB/s、每个 NPU link 不超过 50 GB/s、每个 NPU 同时最多一个 compute 区间。因而“利用率接近”不是 summary 把策略写成同一个值造成的。",
        "",
        "但这不是对真实硬件计数器的验证。此 DES 把每个 SSU 建模为一次只运行一个、不可抢占、以 40 GB/s 执行的命令，把每个 NPU link 建模为一次只运行一个、以 50 GB/s 执行的流；CIR 改变离散命令的排序，而不是把同时在途 I/O 按比例限速。连续满负载下，各策略的总字节量和计算量接近，所以 fleet 平均利用率天然接近；策略更明显地改变的是谁先得到 I/O、单请求 barrier 和 TTFT/SLO。这个抽象内部自洽，但不能单独证明与真实 SSD/NVMe 并发和带宽整形足够接近。",
        "",
        "Baseline 使用 category/generic Path，没有 NPU→dedicated Path 映射，因此不能定义逐 NPU 的 installed CIR；报告保留逐 SSU×Path 的 CIR、每条 flow 的实际 path_id 以及 dispatch 时该 Path 的 CIR。Adaptive 100 ms 才额外报告 dedicated Path 映射下的逐 NPU CIR 总和。分析没有为 Baseline 虚构 NPU CIR。",
        "",
        *causal_lines,
        "## 校验",
        "",
    ]
    for case in EXPECTED_CASES:
        validation = validations[case]
        closure = validation["closure"]
        capacity = validation["capacity"]
        full_measurement = validation["full_measurement_5ms"]
        maximum = max(closure["maximum_absolute_residuals"].values())
        lines.extend(
            [
                f"- {CASE_LABELS[case]}：capacity checks={'PASS' if capacity['passed'] else 'FAIL'}；50 ms closure={'PASS' if closure['passed'] else 'FAIL'}；full 3 s/5 ms reconstruction={'PASS' if full_measurement['passed'] else 'FAIL'}；最大绝对闭合残差={maximum:.3e}。",
            ]
        )
    if historical_audit is not None:
        maximum_prefix_residual = max(
            abs(
                float(row["new_3s_minus_historical_prefix_residual"])
            )
            for row in historical_audit["cases"].values()
        )
        lines.append(
            "- Frozen formal 8 s prefix："
            f"{'PASS' if historical_audit['passed'] else 'FAIL'}；"
            f"新 3 s 与旧前六 block 最大残差={maximum_prefix_residual:.3e}。"
        )
    historical_file_lines = (
        [
            "- `historical_8s_convergence.csv`：frozen formal 的 16×500 ms 原始利用率与累计利用率。",
            "- `06_formal_8s_convergence.png`：3 s 截点到 8 s 终点的窗口收敛图。",
        ]
        if historical_audit is not None
        else []
    )
    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `service_intervals.csv`：每条 SSD、NPU link、compute 的原始精确起止时间、实际速率、请求/NPU/SSU/layer 标识。",
            "- `npu_timeseries.csv`：0.5 ms、逐 NPU 的 compute 利用率和实际 link 带宽。",
            "- `ssu_timeseries.csv`：0.5 ms、逐 SSU 的实际带宽和占用率。",
            "- `fleet_timeseries.csv`：0.5 ms 的 fleet 聚合值。",
            "- `cir_timeseries.csv`：0.5 ms、逐 SSU×相关 Path 的已安装 CIR 时间加权值；Adaptive 的 dedicated Path 行带 NPU ID，Baseline 的 NPU ID 为 N/A。",
            "- `request_details.csv`、`layer_details.csv`：请求级与 layer 因果分解。",
            "- `selected_layer_service_intervals.csv`：自动选出的 Baseline/Adaptive 两个完整 layer 的轻量原始 SSD/link/compute 区间。",
            "- `03_selected_layer_micro_timeline.png`、`03b_baseline_selected_layer_micro_timeline.png`：Adaptive 与 Baseline 各自的请求级因果时间线；带宽面板是 selected-layer contribution 的 0.02 ms bin 平均。",
            "- `case_summary.csv`：新 3 s measurement 的利用率，以及从 raw request rows 重算的 α=1.5/α=2 两组 SLO。",
            "- `full_measurement_5ms_timeseries.csv`：完整 3 s 的 fleet 原生 5 ms、100 ms rolling 与累计资源占用。",
            "- `full_measurement_npu_5ms_timeseries.csv`：完整 3 s 的逐 NPU 5 ms、rolling 与累计 compute duty。",
            "- `validation.json`：容量、输入配对和两级积分闭合的机器可读证据。",
            *historical_file_lines,
            "",
            "时间区间统一采用半开语义 `[start, end)`；图中的 0.5 ms 点是精确 interval overlap 的平均值，不是采样时刻的瞬时猜测。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        help=(
            "raw root or individual case directory; repeat for independently "
            "run baseline/adaptive roots"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bin-ms", type=float, default=0.5)
    parser.add_argument(
        "--historical-formal-json",
        type=Path,
        help=(
            "optional frozen formal artifact whose 16x500 ms blocks are used "
            "to prove 3 s-prefix and 8 s convergence"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_roots = args.input_root or [DEFAULT_INPUT_ROOT]
    output_dir = args.output_dir.expanduser().resolve()
    cases = _load_cases(input_roots)
    all_intervals: list[ServiceInterval] = []
    npu_rows: list[dict] = []
    ssu_rows: list[dict] = []
    fleet_rows: list[dict] = []
    cir_rows: list[dict] = []
    request_rows: list[dict] = []
    layer_rows: list[dict] = []
    case_summary_rows: list[dict] = []
    full_measurement_rows: list[dict] = []
    full_measurement_npu_rows: list[dict] = []
    validations: dict[str, dict] = {}
    trace_starts = {item.name: float(item.trace["trace_start_ms"]) for item in cases}

    for case in cases:
        intervals = _normalise_intervals(case)
        all_intervals.extend(intervals)
        bins = _make_bins(float(case.trace["trace_start_ms"]), float(case.trace["trace_end_ms"]), args.bin_ms)
        current_npu, current_ssu, current_fleet = _build_binned_resource_rows(case, intervals, bins)
        npu_rows.extend(current_npu)
        ssu_rows.extend(current_ssu)
        fleet_rows.extend(current_fleet)
        cir_rows.extend(_build_cir_rows(case, bins))
        request_rows.extend(_request_rows(case, intervals))
        layer_rows.extend(_layer_rows(case, intervals))
        current_full, current_full_npu, full_measurement_audit = (
            _build_full_measurement_rows(case)
        )
        full_measurement_rows.extend(current_full)
        full_measurement_npu_rows.extend(current_full_npu)
        slo_metrics = _slo_metrics(case.summary)
        validations[case.name] = {
            "raw_runner_validation": case.trace["validation"],
            "capacity": _capacity_validation(case, intervals),
            "closure": _closure_validation(case, intervals, bins, current_npu, current_ssu),
            "full_measurement_5ms": full_measurement_audit,
            "slo_recalculation": slo_metrics,
        }
        _require(validations[case.name]["capacity"]["passed"], f"{case.name}: capacity validation failed")
        _require(validations[case.name]["closure"]["passed"], f"{case.name}: closure validation failed")
        _require(
            full_measurement_audit["passed"],
            f"{case.name}: full 3 s/5 ms reconstruction failed",
        )
        _require(
            slo_metrics["summary_alpha2_matches_recomputation"],
            f"{case.name}: summary alpha=2 SLO does not match request-row recomputation",
        )
        closure = validations[case.name]["closure"]
        case_summary_rows.append(
            {
                "case": case.name,
                "num_ssu": case.num_ssu,
                "measurement_duration_ms": case.summary["measurement_duration_ms"],
                "trace_duration_ms": case.trace["trace_duration_ms"],
                "measurement_request_count": slo_metrics["request_count"],
                "measurement_mean_npu_utilization": case.summary[
                    "mean_npu_utilization"
                ],
                "measurement_mean_npu_utilization_pct": 100.0
                * float(case.summary["mean_npu_utilization"]),
                "trace_mean_npu_utilization": closure[
                    "trace_mean_npu_compute_utilization"
                ],
                "trace_mean_npu_utilization_pct": 100.0
                * float(closure["trace_mean_npu_compute_utilization"]),
                "measurement_mean_ssd_utilization_pct": 100.0
                * float(case.summary["measurement_ssd_mean_utilization"]),
                "trace_mean_ssd_utilization_pct": 100.0
                * float(closure["trace_mean_ssd_utilization"]),
                "alpha1p5_equal_npu_slo_attainment": slo_metrics[
                    "alpha1p5_equal_npu_slo_attainment"
                ],
                "alpha1p5_equal_npu_slo_pct": 100.0
                * float(slo_metrics["alpha1p5_equal_npu_slo_attainment"]),
                "alpha2_equal_npu_slo_attainment": slo_metrics[
                    "alpha2_equal_npu_slo_attainment"
                ],
                "alpha2_equal_npu_slo_pct": 100.0
                * float(slo_metrics["alpha2_equal_npu_slo_attainment"]),
                "alpha1p5_request_weighted_slo_pct": 100.0
                * float(
                    slo_metrics["alpha1p5_request_weighted_slo_attainment"]
                ),
                "alpha2_request_weighted_slo_pct": 100.0
                * float(slo_metrics["alpha2_request_weighted_slo_attainment"]),
            }
        )

    selected_layers = {
        case_name: _select_layer(layer_rows, case_name)
        for case_name in EXPECTED_CASES
    }
    per_npu_stats = _per_npu_measurement_stats(cases)
    historical_rows: list[dict] = []
    historical_audit: dict | None = None
    if args.historical_formal_json is not None:
        historical_rows, historical_audit = _historical_formal_rows(
            args.historical_formal_json,
            cases,
        )
        _require(
            historical_audit["passed"],
            "historical formal 3 s-prefix/8 s convergence validation failed",
        )
        validations["historical_formal"] = historical_audit
    base_checks_passed = all(
        validations[case][kind]["passed"]
        for case in EXPECTED_CASES
        for kind in ("capacity", "closure", "full_measurement_5ms")
    )
    validations["paired"] = {
        "input_fingerprints_equal": len({item.summary["input_fingerprint"] for item in cases}) == 1,
        "selected_layer": selected_layers["adaptive_t0_i100ms"],
        "selected_layers_by_case": selected_layers,
        "all_checks_passed": base_checks_passed
        and (historical_audit is None or historical_audit["passed"]),
        "num_ssu": cases[0].num_ssu,
        "per_npu_measurement_stats": per_npu_stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "service_intervals.csv", _service_rows(all_intervals, trace_starts))
    _write_csv(output_dir / "npu_timeseries.csv", npu_rows)
    _write_csv(output_dir / "ssu_timeseries.csv", ssu_rows)
    _write_csv(output_dir / "fleet_timeseries.csv", fleet_rows)
    _write_csv(output_dir / "cir_timeseries.csv", cir_rows)
    _write_csv(output_dir / "request_details.csv", request_rows)
    _write_csv(output_dir / "layer_details.csv", layer_rows)
    _write_csv(
        output_dir / "selected_layer_service_intervals.csv",
        _selected_layer_service_rows(
            all_intervals,
            selected_layers,
            trace_starts,
        ),
    )
    _write_csv(output_dir / "case_summary.csv", case_summary_rows)
    _write_csv(
        output_dir / "full_measurement_5ms_timeseries.csv",
        full_measurement_rows,
    )
    _write_csv(
        output_dir / "full_measurement_npu_5ms_timeseries.csv",
        full_measurement_npu_rows,
    )
    if historical_audit is not None:
        _write_csv(
            output_dir / "historical_8s_convergence.csv",
            historical_rows,
        )
    _write_json(output_dir / "validation.json", validations)
    _plot_fleet(
        fleet_rows,
        cases[0].num_ssu,
        output_dir / "01_fleet_npu_ssd_timeseries.png",
    )
    _plot_heatmap(npu_rows, output_dir / "02_npu_compute_heatmap.png")
    _plot_selected_layer(
        selected_layers["adaptive_t0_i100ms"],
        cases,
        all_intervals,
        output_dir / "03_selected_layer_micro_timeline.png",
    )
    _plot_selected_layer(
        selected_layers["baseline"],
        cases,
        all_intervals,
        output_dir / "03b_baseline_selected_layer_micro_timeline.png",
    )
    _plot_full_measurement(
        full_measurement_rows,
        output_dir / "04_full_3s_timeseries.png",
    )
    _plot_per_npu_measurement(
        per_npu_stats,
        output_dir / "05_per_npu_measurement_utilization.png",
    )
    if historical_audit is not None:
        _plot_historical_convergence(
            historical_rows,
            output_dir / "06_formal_8s_convergence.png",
        )
    _report(
        cases,
        validations,
        selected_layers,
        full_measurement_rows,
        per_npu_stats,
        historical_audit,
        output_dir / "report.md",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
