#!/usr/bin/env python3
"""Reproduce a low-utilization 4-NPU/1-SSU Baseline Path0 workload.

The primary result always uses the repository's Baseline client: every block
is submitted to QoS Path 0.  A read-only event observer records the same three
physical views used by the historical ``baseline_layer0_32npu...`` bundle.
The optional four-Path run changes only the Path mapping and is a mechanism
counterfactual; it is not part of the Baseline search objective.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from authenticated_workload_inputs import load_authenticated_bw_table
import continuous_batch_sim as batch_sim
from continuous_batch_sim import (
    ContinuousBatchRequest,
    SteadyStateConfig,
    continuous_batch_input_fingerprint,
    simulate_continuous_batch,
)
from continuous_prefill_client import routing_strategy_specs, static_qos_config
import sim


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "baseline_4npu_ssu1_low_utilization"
NUM_NPU = 4
NUM_SSU = 1
N_LAYERS = 8
DISK_BW_GIBPS = 40.0
NPU_LINK_BW_GIBPS = 50.0
SUBMIT_ORDER_SEED = 42
WARMUP_REQUESTS_PER_NPU = 8
SETTLE_MS = 500.0
MEASUREMENT_MS = 1_000.0
BLOCK_MS = 100.0
SLO_ALPHA = 8.0
TIME_TOLERANCE_MS = 1e-8

# The final bounded-search winner.  ``seq_len_k`` uses the simulator's
# historical Ki-token convention: 1 means 1,024 total tokens.
FINAL_KEYS = ((1, 169), (192, 368), (192, 896), (192, 896))
# A second Baseline input retained only for the unit audit.  Its GiB/s sum is
# below the GiB/s equivalent of a literal decimal 40 GB/s limit.
DECIMAL_40GBPS_KEYS = ((1, 160), (192, 400), (192, 960), (192, 960))
FINAL_ROLES = ("short_victim", "large_fast", "large_moderate_a", "large_moderate_b")
NQL_GRID = (64, 128, 256, 512, 1024, 2048, 4096)

NPU_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
IO_COLOR = "#2C7FB8"
STALL_COLOR = "#D62728"
COMPUTE_COLOR = "#72B7B2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _savefig(fig, path: Path, description: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        metadata={"Title": path.stem, "Description": description},
    )
    plt.close(fig)
    temporary.replace(path)


def _load_table():
    with redirect_stdout(io.StringIO()):
        return load_authenticated_bw_table(NUM_NPU)


def _interpolated_compute_us(table, seq_len_k: int, nql: int) -> tuple[float, dict]:
    """Piecewise-linear NQL interpolation, plus the documented 1K extrapolation.

    At every tabulated NQL, compute time is exactly linear in ``seq_len_k``
    across the tracked 32K--192K grid (up to floating-point noise).  The 1K
    profile extends the 32K/48K line.  NQLs between two tracked values are then
    linearly interpolated.  This is a model construction, not a hardware datum.
    """

    if nql < NQL_GRID[0] or nql > NQL_GRID[-1]:
        raise ValueError(f"NQL {nql} is outside the authenticated table's NQL range")
    position = bisect.bisect_left(NQL_GRID, nql)
    if position < len(NQL_GRID) and NQL_GRID[position] == nql:
        lower = upper = nql
    else:
        lower, upper = NQL_GRID[position - 1], NQL_GRID[position]

    def at_grid(grid_nql: int) -> float:
        direct = table.get((seq_len_k, grid_nql))
        if direct is not None:
            return float(direct[1])
        if seq_len_k != 1:
            raise ValueError(f"only seq_len_k=1 extrapolation is defined: {(seq_len_k, nql)}")
        value_32 = float(table[(32, grid_nql)][1])
        slope_per_k = (float(table[(48, grid_nql)][1]) - value_32) / 16.0
        return value_32 + slope_per_k * (seq_len_k - 32)

    lower_us = at_grid(lower)
    upper_us = at_grid(upper)
    fraction = 0.0 if lower == upper else (nql - lower) / (upper - lower)
    compute_us = lower_us + fraction * (upper_us - lower_us)
    return compute_us, {
        "method": "piecewise_linear_nql_then_linear_seq_extrapolation",
        "nql_bounds": [lower, upper],
        "nql_fraction": fraction,
        "seq_extrapolated_from_k": ([32, 48] if seq_len_k == 1 else None),
    }


def _profile(table, npu_id: int, key: tuple[int, int], role: str) -> dict:
    seq_len_k, nql = key
    if nql >= seq_len_k * 1024:
        raise ValueError(f"NQL leaves no SSD-resident prefix: {key}")
    direct = table.get(key)
    if direct is not None:
        required_bw, per_layer_us, source_ttft_ms, per_layer_kv_gib = map(float, direct)
        construction = {"method": "direct_data_row"}
    else:
        per_layer_us, construction = _interpolated_compute_us(table, seq_len_k, nql)
        per_layer_kv_gib = (seq_len_k * 1024 - nql) * 1408.0 / (2**30)
        required_bw = per_layer_kv_gib / (per_layer_us / 1e6)
        source_ttft_ms = 78.0 * per_layer_us / 1000.0
    recomputed = per_layer_kv_gib / (per_layer_us / 1e6)
    if not math.isclose(required_bw, recomputed, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"profile bandwidth identity failed: {key}")
    return {
        "npu_id": npu_id,
        "role": role,
        "seq_len_k": seq_len_k,
        "total_tokens": seq_len_k * 1024,
        "nql": nql,
        "ssd_prefix_tokens": seq_len_k * 1024 - nql,
        "per_layer_kv_gib": per_layer_kv_gib,
        "per_layer_compute_us": per_layer_us,
        "required_bandwidth_gibps": required_bw,
        "source_equivalent_ttft_78_layers_ms": source_ttft_ms,
        "simulated_ideal_compute_8_layers_ms": N_LAYERS * per_layer_us / 1000.0,
        "construction": construction,
    }


def _profiles(table, keys=FINAL_KEYS) -> tuple[dict, ...]:
    return tuple(
        _profile(table, npu_id, key, FINAL_ROLES[npu_id])
        for npu_id, key in enumerate(keys)
    )


def _placement(profile: dict) -> tuple[tuple[int, float], ...]:
    seq_len_k = int(profile["seq_len_k"])
    nql = int(profile["nql"])
    _, _, ssd_tokens = sim.calculate_token_partition(seq_len_k, nql)
    gib_per_token = float(profile["per_layer_kv_gib"]) / ssd_tokens
    return tuple(
        (
            0,
            float(
                min(sim.BLOCK_SIZE, ssd_tokens - block_idx * sim.BLOCK_SIZE)
                * gib_per_token
            ),
        )
        for block_idx in range(math.ceil(ssd_tokens / sim.BLOCK_SIZE))
    )


def _build_requests(profiles: Sequence[dict]) -> tuple[ContinuousBatchRequest, ...]:
    requests = []
    # The short lane completes far more requests while the fleet waits for the
    # 192K lanes to warm up.  Unequal finite counts prevent artificial backlog
    # exhaustion while preserving one saturated stream per NPU.
    counts = (3000, 150, 150, 150)
    for profile, request_count in zip(profiles, counts):
        npu_id = int(profile["npu_id"])
        placement = _placement(profile)
        key = (int(profile["seq_len_k"]), int(profile["nql"]))
        for generation in range(request_count):
            request_id = npu_id * 100_000 + generation
            load = {
                "request_id": request_id,
                "npu_id": npu_id,
                "stream_id": generation,
                "generation": generation,
                "profile_key": key,
                "seq_len_k": key[0],
                "nql": key[1],
                "category": sim.classify_request(*key),
                "required_bw_input_gbps": float(profile["required_bandwidth_gibps"]),
                "per_layer_us": float(profile["per_layer_compute_us"]),
                "per_layer_kv_gb": float(profile["per_layer_kv_gib"]),
                "source_ttft_ms": float(profile["source_equivalent_ttft_78_layers_ms"]),
                "arrival_ms": 0.0,
                "arrival_time": 0.0,
                "initial": generation == 0,
                "role": profile["role"],
                "constructed_profile": profile["construction"]["method"]
                != "direct_data_row",
            }
            requests.append(
                ContinuousBatchRequest(
                    request_id=request_id,
                    npu_id=npu_id,
                    arrival_time_ms=0.0,
                    load=load,
                    placement=(placement,),
                )
            )
    return tuple(requests)


def _baseline_client():
    return next(spec for spec in routing_strategy_specs() if spec.name == "baseline")


def _simulate(requests, *, four_path_diagnostic=False):
    return simulate_continuous_batch(
        requests,
        num_npu=NUM_NPU,
        num_ssu=NUM_SSU,
        n_layers=N_LAYERS,
        batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR,
        qos_config=static_qos_config(),
        npu_dedicated_paths=((0, 1, 2, 3) if four_path_diagnostic else None),
        cross_request_layer0_prefetch=True,
        client_io_config=_baseline_client().client_config(),
        steady_state=SteadyStateConfig(
            warmup_requests_per_npu=WARMUP_REQUESTS_PER_NPU,
            settle_ms=SETTLE_MS,
            measurement_ms=MEASUREMENT_MS,
            slo_alpha=SLO_ALPHA,
            block_ms=BLOCK_MS,
            timeline_diagnostics=False,
        ),
        disk_bw_gbps=DISK_BW_GIBPS,
        npu_bw_gbps=NPU_LINK_BW_GIBPS,
        pressure_ttl_ms=0.0,
        cir_write_threshold_gbps=0.0,
        submit_order_seed=SUBMIT_ORDER_SEED,
    )


class WindowTimelineTrace:
    """Reversible, read-only wrappers around authoritative simulator events."""

    def __init__(self, window_start_ms: float, window_end_ms: float):
        self.window_start_ms = float(window_start_ms)
        self.window_end_ms = float(window_end_ms)
        self.block_rows: list[dict] = []
        self.layer_rows: list[dict] = []
        self.pending: dict[int, dict] = {}
        self.global_enqueue_order = 0
        self.ssu_enqueue_order = defaultdict(int)
        self.ssu_service_order = defaultdict(int)
        self.path0_fifo_mismatches = 0
        self._enqueue = sim.DiskIOScheduler.enqueue_many
        self._activate = sim.DiskIOScheduler._activate_flow
        self._complete = sim.DiskIOScheduler.complete_ready_flows
        self._compute = batch_sim._handle_compute_schedule

    def __enter__(self):
        collector = self

        def enqueue_many(scheduler, flows, current_time_ms):
            flows = tuple(flows)
            for flow in flows:
                identity = id(flow)
                if identity in collector.pending:
                    raise AssertionError("duplicate observed flow object")
                ssu_id = int(flow.disk_id)
                collector.pending[identity] = {
                    "_flow": flow,
                    "global_enqueue_order": collector.global_enqueue_order,
                    "ssu_enqueue_order": collector.ssu_enqueue_order[ssu_id],
                    "ssd_service_order": None,
                    "ssu_id": ssu_id,
                    "npu_id": int(flow.npu_id),
                    "request_id": int(flow.request_id),
                    "layer": int(flow.layer),
                    "block_idx": int(flow.block_idx),
                    "size_gib": float(flow.total_gb),
                    "path_id": int(flow.queue_id),
                    "enqueue_time_ms": float(current_time_ms),
                    "ssd_start_time_ms": None,
                    "ssd_end_time_ms": None,
                    "ssd_queue_wait_ms": None,
                }
                collector.global_enqueue_order += 1
                collector.ssu_enqueue_order[ssu_id] += 1
            return collector._enqueue(scheduler, flows, current_time_ms)

        def activate_flow(scheduler, flow, current_time_ms):
            result = collector._activate(scheduler, flow, current_time_ms)
            row = collector.pending[id(flow)]
            ssu_id = int(flow.disk_id)
            order = collector.ssu_service_order[ssu_id]
            row["ssd_service_order"] = order
            row["ssd_start_time_ms"] = float(current_time_ms)
            row["ssd_queue_wait_ms"] = float(flow.ssd_queue_wait_ms)
            collector.ssu_service_order[ssu_id] += 1
            if row["path_id"] == 0 and order != row["ssu_enqueue_order"]:
                collector.path0_fifo_mismatches += 1
            return result

        def complete_ready_flows(scheduler, current_time_ms):
            completed = collector._complete(scheduler, current_time_ms)
            for flow in completed:
                row = collector.pending.pop(id(flow))
                row.pop("_flow")
                row["ssd_end_time_ms"] = float(current_time_ms)
                if (
                    row["ssd_start_time_ms"] < collector.window_end_ms
                    and row["ssd_end_time_ms"] > collector.window_start_ms
                ):
                    collector.block_rows.append(row)
            return completed

        def handle_compute_schedule(context, npu_id, current_time_ms):
            before = context.npus[npu_id].compute_active
            result = collector._compute(context, npu_id, current_time_ms)
            active = context.npus[npu_id].compute_active
            if active is None or active == before:
                return result
            batch_id, layer = active
            batch = context.microbatches[batch_id]
            metric = batch.layer_metrics[layer]
            if len(batch.member_request_ids) != 1:
                raise AssertionError("timeline observer requires batch size one")
            request_id = int(batch.member_request_ids[0])
            manifest = context.requests[request_id].manifest
            compute_start = float(metric.compute_start_ms)
            compute_end = float(metric.compute_end_ms)
            barrier_start = compute_start - float(metric.io_barrier_wait_ms)
            io_start = float(metric.io_start_time_ms)
            io_ready = float(metric.io_ready_time_ms)
            intervals = (
                (io_start, io_ready),
                (barrier_start, compute_start),
                (compute_start, compute_end),
            )
            if any(
                left < collector.window_end_ms and right > collector.window_start_ms
                for left, right in intervals
            ):
                collector.layer_rows.append(
                    {
                        "npu_id": int(npu_id),
                        "request_id": request_id,
                        "role": str(manifest.load["role"]),
                        "seq_len_k": int(manifest.load["seq_len_k"]),
                        "nql": int(manifest.load["nql"]),
                        "layer": int(layer),
                        "io_release_time_ms": io_start,
                        "io_ready_time_ms": io_ready,
                        "comparison_deadline_ms": barrier_start,
                        "compute_start_time_ms": compute_start,
                        "compute_end_time_ms": compute_end,
                        "compute_duration_ms": float(metric.compute_duration_ms),
                        "io_barrier_wait_ms": float(metric.io_barrier_wait_ms),
                        "overrun_ms": max(0.0, io_ready - barrier_start),
                    }
                )
            return result

        sim.DiskIOScheduler.enqueue_many = enqueue_many
        sim.DiskIOScheduler._activate_flow = activate_flow
        sim.DiskIOScheduler.complete_ready_flows = complete_ready_flows
        batch_sim._handle_compute_schedule = handle_compute_schedule
        return self

    def __exit__(self, exc_type, exc, traceback):
        # The steady-state runner may stop after all tagged requests drain
        # while an untagged command crossing the measurement end is still
        # active.  Its scheduled physical end is already fixed and is needed
        # to account the portion inside the half-open measurement window.
        if exc_type is None:
            for row in self.pending.values():
                flow = row.get("_flow")
                if (
                    row["ssd_start_time_ms"] is not None
                    and row["ssd_start_time_ms"] < self.window_end_ms
                    and math.isfinite(float(flow.end_time))
                    and float(flow.end_time) > self.window_start_ms
                ):
                    copied = {key: value for key, value in row.items() if key != "_flow"}
                    copied["ssd_end_time_ms"] = float(flow.end_time)
                    self.block_rows.append(copied)
        sim.DiskIOScheduler.enqueue_many = self._enqueue
        sim.DiskIOScheduler._activate_flow = self._activate
        sim.DiskIOScheduler.complete_ready_flows = self._complete
        batch_sim._handle_compute_schedule = self._compute
        return False


def _overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(float(right), end) - max(float(left), start))


def _percentiles(values: Sequence[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }


def _analyze_timeline(summary, trace: WindowTimelineTrace) -> tuple[dict, dict]:
    start = float(summary["measurement_start_ms"])
    end = float(summary["measurement_end_ms"])
    compute_ms = [0.0] * NUM_NPU
    barrier_ms = [0.0] * NUM_NPU
    for row in trace.layer_rows:
        npu_id = row["npu_id"]
        compute_ms[npu_id] += _overlap(
            row["compute_start_time_ms"], row["compute_end_time_ms"], start, end
        )
        barrier_ms[npu_id] += _overlap(
            row["comparison_deadline_ms"], row["compute_start_time_ms"], start, end
        )

    ssd_busy_ms = math.fsum(
        _overlap(row["ssd_start_time_ms"], row["ssd_end_time_ms"], start, end)
        for row in trace.block_rows
    )
    stall_service = [defaultdict(float) for _ in range(NUM_NPU)]
    service_rows = sorted(trace.block_rows, key=lambda row: row["ssd_start_time_ms"])
    service_ends = [float(row["ssd_end_time_ms"]) for row in service_rows]
    for layer in trace.layer_rows:
        waiting_npu = layer["npu_id"]
        left = max(start, layer["comparison_deadline_ms"])
        right = min(end, layer["compute_start_time_ms"])
        if right <= left:
            continue
        first = bisect.bisect_right(service_ends, left)
        for block in service_rows[first:]:
            if block["ssd_start_time_ms"] >= right:
                break
            amount = _overlap(
                block["ssd_start_time_ms"], block["ssd_end_time_ms"], left, right
            )
            if amount:
                stall_service[waiting_npu][block["npu_id"]] += amount

    queue_wait = {
        str(npu_id): _percentiles(
            [
                float(row["ssd_queue_wait_ms"])
                for row in trace.block_rows
                if row["npu_id"] == npu_id
            ]
        )
        for npu_id in range(NUM_NPU)
    }
    attribution = {}
    for waiting_npu in range(NUM_NPU):
        by_source = {
            str(serving_npu): float(stall_service[waiting_npu].get(serving_npu, 0.0))
            for serving_npu in range(NUM_NPU)
        }
        busy = sum(by_source.values())
        attribution[str(waiting_npu)] = {
            "barrier_ms": barrier_ms[waiting_npu],
            "ssd_busy_overlap_ms": busy,
            "ssd_busy_overlap_fraction": (
                busy / barrier_ms[waiting_npu] if barrier_ms[waiting_npu] else None
            ),
            "ssd_service_overlap_ms_by_serving_npu": by_source,
            "other_npu_service_overlap_ms": sum(
                value for source, value in by_source.items() if int(source) != waiting_npu
            ),
            "idle_or_link_tail_ms": max(0.0, barrier_ms[waiting_npu] - busy),
        }

    checks = {
        "simulator_invariants": all(summary["invariants"].values()),
        "all_npus_have_completed_window_requests": all(
            int(value) > 0 for value in summary["request_counts_by_npu"]
        ),
        "all_npus_have_continuous_compute_or_barrier_coverage": all(
            math.isclose(
                compute_ms[npu_id] + barrier_ms[npu_id],
                MEASUREMENT_MS,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for npu_id in range(NUM_NPU)
        ),
        "observer_compute_matches_summary": all(
            math.isclose(
                compute_ms[npu_id],
                float(summary["compute_ms_by_npu"][npu_id]),
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            for npu_id in range(NUM_NPU)
        ),
        "observer_ssd_busy_matches_summary": math.isclose(
            ssd_busy_ms,
            float(summary["measurement_ssd_busy_ms_by_ssu"][0]),
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        "one_nonpreemptive_command_per_ssd": all(
            right["ssd_start_time_ms"] + TIME_TOLERANCE_MS
            >= left["ssd_end_time_ms"]
            for left, right in zip(
                service_rows,
                service_rows[1:],
            )
        ),
        "every_observed_command_uses_path0": all(
            row["path_id"] == 0 for row in trace.block_rows
        ),
        "path0_service_order_equals_enqueue_order": trace.path0_fifo_mismatches == 0,
        "physical_service_duration_is_40_gibps": all(
            math.isclose(
                row["ssd_end_time_ms"] - row["ssd_start_time_ms"],
                1000.0 * row["size_gib"] / DISK_BW_GIBPS,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for row in trace.block_rows
        ),
    }
    analysis = {
        "compute_ms_by_npu_from_trace": compute_ms,
        "barrier_ms_by_npu_from_trace": barrier_ms,
        "compute_plus_barrier_ms_by_npu": [
            compute_ms[npu_id] + barrier_ms[npu_id] for npu_id in range(NUM_NPU)
        ],
        "ssd_busy_ms_from_trace": ssd_busy_ms,
        "physical_block_count_overlapping_window": len(trace.block_rows),
        "path0_fifo_mismatch_count_full_run": trace.path0_fifo_mismatches,
        "ssd_queue_wait_ms_by_npu": queue_wait,
        "stall_attribution": attribution,
    }
    return analysis, checks


def _plot_npu_timeline(rows: Sequence[dict], summary: dict, output: Path) -> None:
    start = float(summary["measurement_start_ms"])
    end = float(summary["measurement_end_ms"])
    fig, ax = plt.subplots(figsize=(16, 5.8))
    io_segments, stall_segments, compute_segments = [], [], []
    for row in rows:
        y = NUM_NPU - 1 - int(row["npu_id"])
        io_left, io_right = max(start, row["io_release_time_ms"]), min(end, row["io_ready_time_ms"])
        if io_right > io_left:
            io_segments.append(((io_left - start, io_right - start), y + 0.18))
        stall_left = max(start, row["comparison_deadline_ms"])
        stall_right = min(end, row["compute_start_time_ms"])
        if stall_right > stall_left:
            stall_segments.append(((stall_left - start, stall_right - start), y))
        comp_left = max(start, row["compute_start_time_ms"])
        comp_right = min(end, row["compute_end_time_ms"])
        if comp_right > comp_left:
            compute_segments.append(((comp_left - start, comp_right - start), y - 0.18))

    def add(segments, color, width, zorder):
        if not segments:
            return
        collection = LineCollection(
            [[(left, y), (right, y)] for (left, right), y in segments],
            colors=color,
            linewidths=width,
            capstyle="butt",
            zorder=zorder,
        )
        ax.add_collection(collection)

    add(io_segments, IO_COLOR, 2.0, 1)
    add(stall_segments, STALL_COLOR, 5.0, 3)
    add(compute_segments, COMPUTE_COLOR, 3.2, 2)
    ax.set_xlim(0, MEASUREMENT_MS)
    ax.set_ylim(-0.6, NUM_NPU - 0.4)
    ax.set_yticks(range(NUM_NPU))
    ax.set_yticklabels([f"NPU {npu_id}" for npu_id in reversed(range(NUM_NPU))])
    ax.set_xlabel("Time in the measured middle window (ms)")
    ax.set_ylabel("NPU")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(
        "Baseline Path0: logical I/O, exposed barrier stall, and compute\n"
        f"1 s exact window; mean NPU utilization = {100*summary['mean_npu_utilization']:.2f}%"
    )
    ax.legend(
        handles=(
            Patch(color=IO_COLOR, label="logical I/O release → ready"),
            Patch(color=STALL_COLOR, label="exposed I/O barrier stall"),
            Patch(color=COMPUTE_COLOR, label="NPU compute"),
        ),
        loc="upper right",
        ncol=3,
    )
    _savefig(fig, output, "Exact middle-1-second NPU logical I/O, barrier, and compute intervals")


def _plot_enqueue_order(rows: Sequence[dict], output: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["ssu_enqueue_order"])
    values = np.asarray([[row["npu_id"] for row in ordered]], dtype=np.int16)
    cmap = ListedColormap(NPU_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, NUM_NPU + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(16, 2.8))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=(ordered[0]["ssu_enqueue_order"], ordered[-1]["ssu_enqueue_order"] + 1, 0, 1),
    )
    ax.set_yticks([0.5])
    ax.set_yticklabels([f"SSU 0 ({len(rows):,} serviced blocks overlapping window)"])
    ax.set_xlabel("Absolute FCFS enqueue position on QoS Path 0")
    ax.set_title("Path0 enqueue order for every physical block represented in the 1 s service window")
    colorbar = fig.colorbar(image, ax=ax, ticks=range(NUM_NPU), pad=0.015)
    colorbar.set_label("NPU ID")
    _savefig(fig, output, "Path0 FCFS enqueue-rank raster colored by source NPU")


def _plot_service(rows: Sequence[dict], summary: dict, output: Path) -> None:
    start = float(summary["measurement_start_ms"])
    end = float(summary["measurement_end_ms"])
    segments, colors = [], []
    for row in rows:
        left = max(start, row["ssd_start_time_ms"]) - start
        right = min(end, row["ssd_end_time_ms"]) - start
        if right > left:
            segments.append([(left, 0.0), (right, 0.0)])
            colors.append(NPU_COLORS[row["npu_id"]])
    fig, ax = plt.subplots(figsize=(16, 3.2))
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=8.0, capstyle="butt"))
    ax.set_xlim(0, MEASUREMENT_MS)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([0])
    ax.set_yticklabels(["SSU 0 / Path0"])
    ax.set_xlabel("Physical SSD service time in the measured window (ms)")
    ax.set_title(
        "Physical non-preemptive SSD command service timeline\n"
        f"SSD utilization = {100*summary['measurement_ssd_mean_utilization']:.2f}%; blank regions are idle"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[Patch(color=NPU_COLORS[n], label=f"NPU {n}") for n in range(NUM_NPU)],
        ncol=4,
        loc="upper right",
    )
    _savefig(fig, output, "Every physical non-preemptive SSD command overlapping the 1-second window")


def _plot_blocks(summary: dict, output: Path) -> None:
    blocks = summary["measurement_blocks"]
    x = np.asarray([float(row["start_ms"]) - summary["measurement_start_ms"] for row in blocks])
    width = np.asarray([float(row["duration_ms"]) for row in blocks])
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for npu_id in range(NUM_NPU):
        values = [100.0 * float(row["compute_ms_by_npu"][npu_id]) / float(row["duration_ms"]) for row in blocks]
        ax.step(np.append(x, x[-1] + width[-1]), np.append(values, values[-1]), where="post", color=NPU_COLORS[npu_id], label=f"NPU {npu_id}")
    fleet = [100.0 * float(row["npu_utilization"]) for row in blocks]
    ax.step(np.append(x, x[-1] + width[-1]), np.append(fleet, fleet[-1]), where="post", color="black", linewidth=2.0, label="fleet mean")
    ax.axhline(80.0, color="#777777", linestyle="--", linewidth=1.2, label="80% target")
    ax.set_xlim(0, MEASUREMENT_MS)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Time in measured window (ms)")
    ax.set_ylabel("NPU compute utilization (%)")
    ax.set_title("Exact 100 ms sub-window utilization (same 1 s accounting window)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, loc="lower right")
    _savefig(fig, output, "Per-NPU and fleet utilization in ten exact 100-ms blocks")


def _profile_csv_rows(profiles: Sequence[dict]) -> list[dict]:
    rows = []
    for profile in profiles:
        rows.append(
            {
                "npu_id": profile["npu_id"],
                "role": profile["role"],
                "seq_len_k": profile["seq_len_k"],
                "total_tokens": profile["total_tokens"],
                "nql": profile["nql"],
                "ssd_prefix_tokens": profile["ssd_prefix_tokens"],
                "per_layer_kv_gib": profile["per_layer_kv_gib"],
                "per_layer_compute_us": profile["per_layer_compute_us"],
                "required_bandwidth_gibps": profile["required_bandwidth_gibps"],
                "source_equivalent_ttft_78_layers_ms": profile["source_equivalent_ttft_78_layers_ms"],
                "simulated_ideal_compute_8_layers_ms": profile["simulated_ideal_compute_8_layers_ms"],
                "construction_method": profile["construction"]["method"],
            }
        )
    return rows


def _report(result: dict) -> str:
    base = result["baseline"]
    diag = result["four_path_diagnostic"]
    profiles = result["input"]["profiles"]
    attribution = result["timeline_analysis"]["stall_attribution"]
    decimal_case = result["literal_decimal_40gbps_baseline"]
    search = result.get("bounded_search")
    search_sentence = (
        f"最终统一口径又对短流的整数 NQL=153..192 共 {len(search['rows'])} 个点逐一复跑；"
        f"NQL={search['winner']['victim_nql']} 是该有限网格最低点。"
        if search is not None
        else "搜索明细未随本次运行载入；最终 profile 仍可独立复现。"
    )
    return f"""# 4 NPU / 1 SSU Baseline Path0 最低利用率实验

## 结论

在统一的 1 秒中间窗口内，Baseline Path0 的平均 NPU 利用率为 **{100*base['mean_npu_utilization']:.4f}%**，四张卡分别为 **{', '.join(f'{100*x:.4f}%' for x in base['npu_utilizations'])}**。四条固定流的需求和为 **{result['input']['bandwidth_contract']['instantaneous_fixed_stream_sum_gibps']:.6f} GiB/s**，小于 40 GiB/s；所有 NPU 都有持续的 active request，且计算或 I/O barrier 覆盖完整 1 秒。

这不是已证明的连续参数全局最小值，而是报告所列有界搜索中的最低候选。{search_sentence}`data` 没有概率权重，因此“满足 data 分布”不能解释成统计抽样；这里保留 GLM-5.1 KV 关系，并在相邻 NQL 点间线性插值。NPU 0 的 1K 长度还使用了 32K/48K 计算时间斜率外推，所以它是模型构造，不是实测 profile。

## 输入与带宽定义

| NPU | role | total tokens | NQL | KV/layer (GiB) | compute/layer (us) | B (GiB/s) |
|---:|---|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {p['npu_id']} | {p['role']} | {p['total_tokens']} | {p['nql']} | {p['per_layer_kv_gib']:.9f} | {p['per_layer_compute_us']:.6f} | {p['required_bandwidth_gibps']:.6f} |"
        for p in profiles
    ) + f"""

单流定义为 `B = 每层 SSD KV GiB / 每层理想计算秒数`。由于每个请求固定运行 8 层，分子和分母同时乘 8，B 不变。四条流各自固定 profile，因此任意时刻的输入需求和就是四个 B 的和，不使用周期平均来绕过 40 GiB/s 约束。代码沿用历史 `_gb/_gbps` 字段名，但 KV 由 `2^30` 归一化，实际单位是 GiB/GiB/s。

这也意味着字面量 **40 GB/s（十进制）只等于 {decimal_case['limit_gibps']:.6f} GiB/s**，主候选不满足这个更窄的上限。单位审计另给出一组 Baseline 输入：需求和 **{decimal_case['total_bandwidth_gibps']:.6f} GiB/s = {decimal_case['total_bandwidth_decimal_gbps']:.6f} GB/s**，平均 NPU 利用率仍为 **{100*decimal_case['summary']['mean_npu_utilization']:.4f}%**。它证明结论不依赖把 GB 与 GiB 混用，但不是主搜索网格的最低点。

## 为什么 Path0 会降低利用率

Baseline 将全部物理块放入同一个 Path0 FCFS 队列，命令不可抢占。三个 192K 流每层约有 1,500 个物理块；NPU 0 每层只有数个块，但它们一旦排在已入队的大流块之后就不能越过。跨请求 Layer-0 预取已经开启，但模拟器仍只提前一层，无法为短流建立多层缓冲。

NPU 0 在窗口中的 barrier 为 **{attribution['0']['barrier_ms']:.3f} ms**；其中 SSD 正在服务其他 NPU 的重叠时间为 **{attribution['0']['other_npu_service_overlap_ms']:.3f} ms**。NPU 1 也出现 **{attribution['1']['barrier_ms']:.3f} ms** barrier。因此平均值低不是因为某张卡没有请求，而是 active request 被 I/O barrier 阻塞。

只用于因果诊断的四 Path 对照保持输入、SSD、QoS 表和预取不变，只把 NPU 映射到 Path 0--3。其平均利用率为 **{100*diag['mean_npu_utilization']:.4f}%**，比 Path0 高 **{100*(diag['mean_npu_utilization']-base['mean_npu_utilization']):.4f} 个百分点**；NPU 0 从 **{100*base['npu_utilizations'][0]:.4f}%** 恢复到 **{100*diag['npu_utilizations'][0]:.4f}%**。这支持“单 Path FCFS 队头阻塞是关键原因”。但 NPU 1 在四 Path 下仍约 {100*diag['npu_utilizations'][1]:.2f}%，说明物理 SSD 突发与单层预取深度也是共同原因，不能把全部损失都归因于 Path 数量。

## 审计边界

- GLM-5.1 的 78 层只用于保持 `data` 中 source TTFT = 78 × per-layer compute 的关系；本实验按要求实际执行 8 层。中间窗口消除了首次冷启动 Layer-0 的影响，但每个新请求仍有 Layer-0，跨请求预取会改变 Path0 排队，所以不能说 Layer0 完全无关。
- `{ROOT / 'data'}` 的最大 200K 行超过 [GLM-5.1 官方 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json) 中的 202,752 token 上限（200×1024=204,800），本实验没有使用 200K。
- 当前结果是离散事件模型预测。1K profile 超出 `data` 的 32K--200K 长度网格，真实 GLM-5.1 硬件结论必须补测其 per-layer compute 和 KV 读流量。
- SSD 实际服务率为 **{base['measurement_ssd_served_gbps_by_ssu'][0]:.6f} GiB/s**，不是名义需求和；二者分别回答输入理想需求与物理执行吞吐。

## 文件

- `input_profiles.csv`：四条固定流。
- `request_layer_timeline.csv`：窗口相交的逻辑 I/O、barrier 与 compute。
- `physical_block_trace.csv`：窗口相交的物理 Path0 块服务。
- `01_npu_io_compute_timeline.png`、`02_ssu_path0_enqueue_order.png`、`03_ssu_path0_service_timeline.png`、`04_100ms_npu_utilization.png`：与历史 timeline 方法对应的图。
- `simulator_summary.json`：未经裁剪的 Baseline 稳态摘要。
- `result.json`：审计、归因和四 Path 机制对照。
- `search_summary.csv`、`search_result.json`：统一口径的 NQL=153..192 有界细网格。
"""


def _compact_summary(summary: dict) -> dict:
    return {
        "measurement_start_ms": summary["measurement_start_ms"],
        "measurement_end_ms": summary["measurement_end_ms"],
        "measurement_duration_ms": summary["measurement_duration_ms"],
        "mean_npu_utilization": summary["mean_npu_utilization"],
        "npu_utilizations": summary["npu_utilizations"],
        "compute_ms_by_npu": summary["compute_ms_by_npu"],
        "measurement_ssd_mean_utilization": summary["measurement_ssd_mean_utilization"],
        "measurement_ssd_served_gbps_by_ssu": summary["measurement_ssd_served_gbps_by_ssu"],
        "measurement_npu_ssu_ssd_served_gbps": summary["measurement_npu_ssu_ssd_served_gbps"],
        "request_counts_by_npu": summary["request_counts_by_npu"],
        "measurement_request_count": summary["measurement_request_count"],
        "measurement_blocks": summary["measurement_blocks"],
        "invariants": summary["invariants"],
        "no_backlog_exhaustion": summary["invariants"]["no_backlog_exhaustion"],
    }


def _search_worker(victim_nql: int) -> dict:
    table, _ = _load_table()
    keys = ((1, victim_nql), (192, 368), (192, 896), (192, 896))
    profiles = _profiles(table, keys)
    total = math.fsum(p["required_bandwidth_gibps"] for p in profiles)
    if total > DISK_BW_GIBPS + 1e-12:
        return {"victim_nql": victim_nql, "total_bandwidth_gibps": total, "status": "over_cap"}
    summary = _simulate(_build_requests(profiles))
    return {
        "victim_nql": victim_nql,
        "total_bandwidth_gibps": total,
        "status": "valid",
        "mean_npu_utilization": summary["mean_npu_utilization"],
        "npu0_utilization": summary["npu_utilizations"][0],
        "npu1_utilization": summary["npu_utilizations"][1],
        "npu2_utilization": summary["npu_utilizations"][2],
        "npu3_utilization": summary["npu_utilizations"][3],
        "ssd_utilization": summary["measurement_ssd_mean_utilization"],
        "all_invariants": all(summary["invariants"].values()),
    }


def run_search(output: Path, jobs: int) -> list[dict]:
    # This is intentionally bounded and explicit.  It is a fine search around
    # the best blocker combination found by the broader exploratory screen.
    victim_nqls = tuple(range(153, 193))
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(_search_worker, nql): nql for nql in victim_nqls}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"search nql={row['victim_nql']} status={row['status']} "
                f"mean={row.get('mean_npu_utilization')}",
                flush=True,
            )
    rows.sort(key=lambda row: row["victim_nql"])
    fields = (
        "victim_nql",
        "total_bandwidth_gibps",
        "status",
        "mean_npu_utilization",
        "npu0_utilization",
        "npu1_utilization",
        "npu2_utilization",
        "npu3_utilization",
        "ssd_utilization",
        "all_invariants",
    )
    _atomic_csv(output / "search_summary.csv", fields, rows)
    valid = [row for row in rows if row["status"] == "valid"]
    winner = min(valid, key=lambda row: row["mean_npu_utilization"])
    _atomic_json(
        output / "search_result.json",
        {
            "schema_version": 1,
            "scope": {
                "victim": "seq_len_k=1; integer NQL 153..192",
                "blockers": [[192, 368], [192, 896], [192, 896]],
                "common_measurement": {
                    "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
                    "settle_ms": SETTLE_MS,
                    "measurement_ms": MEASUREMENT_MS,
                },
                "claim": "minimum only within this explicit finite grid",
            },
            "winner": winner,
            "rows": rows,
        },
    )
    return rows


def run_result(output: Path, *, skip_counterfactual=False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    table, authentication = _load_table()
    profiles = _profiles(table)
    total_bandwidth = math.fsum(p["required_bandwidth_gibps"] for p in profiles)
    if total_bandwidth > DISK_BW_GIBPS + 1e-12:
        raise AssertionError("fixed-stream bandwidth sum exceeds 40 GiB/s")
    requests = _build_requests(profiles)
    input_fingerprint = continuous_batch_input_fingerprint(requests)

    print("running uninstrumented Baseline Path0", flush=True)
    started = time.monotonic()
    baseline = _simulate(requests)
    baseline_wall_s = time.monotonic() - started
    baseline_hash = _json_hash(baseline)
    if baseline["input_fingerprint"] != input_fingerprint:
        raise AssertionError("simulator input fingerprint drift")

    print("running instrumented Baseline Path0", flush=True)
    with WindowTimelineTrace(
        baseline["measurement_start_ms"], baseline["measurement_end_ms"]
    ) as trace:
        observed = _simulate(requests)
    observed_hash = _json_hash(observed)
    observer_noninterference = observed_hash == baseline_hash
    if not observer_noninterference:
        raise AssertionError("read-only observer changed simulator summary")
    timeline_analysis, timeline_checks = _analyze_timeline(baseline, trace)
    timeline_checks["observer_noninterference"] = observer_noninterference

    if skip_counterfactual:
        four_path = None
        compact_four = {
            "skipped": True,
            "mean_npu_utilization": None,
            "npu_utilizations": [None] * NUM_NPU,
        }
    else:
        print("running four-Path mechanism diagnostic", flush=True)
        four_path = _simulate(requests, four_path_diagnostic=True)
        compact_four = _compact_summary(four_path)

    print("running literal-decimal-40-GB/s Baseline unit audit", flush=True)
    decimal_profiles = _profiles(table, DECIMAL_40GBPS_KEYS)
    decimal_total_gibps = math.fsum(
        profile["required_bandwidth_gibps"] for profile in decimal_profiles
    )
    decimal_limit_gibps = 40.0 * 1e9 / (2**30)
    decimal_summary = _simulate(_build_requests(decimal_profiles))

    checks = {
        **timeline_checks,
        "bandwidth_sum_at_or_below_40_gibps": total_bandwidth <= DISK_BW_GIBPS + 1e-12,
        "baseline_mean_npu_utilization_below_80_percent": baseline["mean_npu_utilization"] < 0.8,
        "cross_request_layer0_prefetch_enabled": True,
        "eight_layers": N_LAYERS == 8,
        "single_ssu": NUM_SSU == 1,
        "four_npus": NUM_NPU == 4,
        "literal_decimal_40gbps_audit_passes_bandwidth": (
            decimal_total_gibps <= decimal_limit_gibps + 1e-12
        ),
        "literal_decimal_40gbps_audit_below_80_percent": (
            decimal_summary["mean_npu_utilization"] < 0.8
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError("result validation failed: " + ", ".join(failed))

    profile_rows = _profile_csv_rows(profiles)
    _atomic_csv(output / "input_profiles.csv", tuple(profile_rows[0]), profile_rows)
    layer_rows = sorted(
        trace.layer_rows,
        key=lambda row: (row["compute_start_time_ms"], row["npu_id"], row["layer"]),
    )
    _atomic_csv(output / "request_layer_timeline.csv", tuple(layer_rows[0]), layer_rows)
    block_rows = sorted(trace.block_rows, key=lambda row: row["ssd_service_order"])
    _atomic_csv(output / "physical_block_trace.csv", tuple(block_rows[0]), block_rows)
    _atomic_json(output / "simulator_summary.json", baseline)

    _plot_npu_timeline(layer_rows, baseline, output / "01_npu_io_compute_timeline.png")
    _plot_enqueue_order(block_rows, output / "02_ssu_path0_enqueue_order.png")
    _plot_service(block_rows, baseline, output / "03_ssu_path0_service_timeline.png")
    _plot_blocks(baseline, output / "04_100ms_npu_utilization.png")

    search_path = output / "search_result.json"
    bounded_search = (
        json.loads(search_path.read_text(encoding="utf-8")) if search_path.exists() else None
    )
    result = {
        "schema_version": 1,
        "experiment": {
            "policy": "Baseline static QoS; every I/O uses Path 0",
            "num_npu": NUM_NPU,
            "num_ssu": NUM_SSU,
            "n_layers": N_LAYERS,
            "batch_size": 1,
            "cross_request_layer0_prefetch": True,
            "disk_bandwidth_gibps": DISK_BW_GIBPS,
            "npu_link_bandwidth_gibps": NPU_LINK_BW_GIBPS,
            "warmup_requests_per_npu": WARMUP_REQUESTS_PER_NPU,
            "settle_ms": SETTLE_MS,
            "measurement_ms": MEASUREMENT_MS,
            "block_ms": BLOCK_MS,
            "submit_order_seed": SUBMIT_ORDER_SEED,
        },
        "input": {
            "profiles": profiles,
            "request_counts_by_input_lane": [3000, 150, 150, 150],
            "input_fingerprint": input_fingerprint,
            "authentication": authentication,
            "bandwidth_contract": {
                "profile_formula": "per_layer_kv_GiB / (per_layer_compute_us / 1e6)",
                "aggregation": "sum of four fixed, concurrently active stream profile demands",
                "instantaneous_fixed_stream_sum_gibps": total_bandwidth,
                "limit_gibps": DISK_BW_GIBPS,
                "passes": total_bandwidth <= DISK_BW_GIBPS + 1e-12,
                "legacy_field_name_warning": "simulator *_gb/*_gbps names carry GiB/GiB/s numerics",
            },
        },
        "baseline": _compact_summary(baseline),
        "baseline_summary_sha256": baseline_hash,
        "instrumented_summary_sha256": observed_hash,
        "instrumented_observer_noninterference": observer_noninterference,
        "timeline_analysis": timeline_analysis,
        "bounded_search": bounded_search,
        "four_path_diagnostic": compact_four,
        "four_path_diagnostic_interpretation": (
            "Same simulator, input, SSD, static QoS table, and prefetch; only fixed routing changes "
            "from one shared Path0 to NPU-dedicated Paths 0..3. It is not a Baseline search case."
        ),
        "literal_decimal_40gbps_baseline": {
            "keys": [list(key) for key in DECIMAL_40GBPS_KEYS],
            "profiles": decimal_profiles,
            "limit_decimal_gbps": 40.0,
            "limit_gibps": decimal_limit_gibps,
            "total_bandwidth_gibps": decimal_total_gibps,
            "total_bandwidth_decimal_gbps": decimal_total_gibps * (2**30) / 1e9,
            "summary": _compact_summary(decimal_summary),
            "interpretation": (
                "Separate Baseline validity witness for a literal decimal 40 GB/s cap; "
                "not part of the main 40-numeric-unit bounded search."
            ),
        },
        "validation": checks,
        "runtime": {"uninstrumented_baseline_wall_s": baseline_wall_s},
        "source_sha256": {
            path.name: _sha256(path)
            for path in (
                ROOT / "sim.py",
                ROOT / "continuous_batch_sim.py",
                ROOT / "continuous_prefill_client.py",
                ROOT / "authenticated_workload_inputs.py",
                ROOT / "data",
                Path(__file__).resolve(),
            )
        },
    }
    _atomic_json(output / "result.json", result)
    if not skip_counterfactual:
        _atomic_text(output / "report.md", _report(result))
    print(
        f"done: mean={baseline['mean_npu_utilization']:.9f}, "
        f"bandwidth={total_bandwidth:.9f} GiB/s, output={output}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-counterfactual", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.search_only:
        run_search(args.output.resolve(), args.jobs)
    else:
        run_result(args.output.resolve(), skip_counterfactual=args.skip_counterfactual)


if __name__ == "__main__":
    main()
