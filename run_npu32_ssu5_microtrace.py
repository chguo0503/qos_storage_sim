#!/usr/bin/env python3
"""Capture an exact 32-NPU/4-or-5-SSU microtrace without changing DES semantics.

This is a diagnostic-only runner for the legacy32 baseline and
``adaptive_t0_i100ms`` cases.  It reuses the frozen workload/configuration code
from :mod:`ms_scale_control_experiment` and temporarily wraps event handlers in
the current process.  The wrappers are restored after every case and never
modify the simulator source or its scheduling decisions.

The trace window is the half-open interval beginning at the steady-state
measurement boundary.  Full service intervals are retained when they overlap
that window, including commands/compute that started before the measurement
boundary or end after the trace window.  ``trace_overlap_*`` fields contain the
exact clipped contribution used for cross-checking the simulator's cumulative
accounting.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

from authenticated_workload_inputs import load_authenticated_bw_table
import continuous_batch_sim as cb
from continuous_batch_sim import (
    continuous_batch_input_fingerprint,
    requests_from_continuous_prefill_workload,
)
from ms_scale_control_experiment import (
    IID_UNIFORM_PROFILE_CATALOG_V1,
    LEGACY32_DEFINITION,
    RunConfig,
    _config_fingerprint,
    _experiment_spec,
    _simulate,
    _source_fingerprint,
)
from random_steady_state_workload import (
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)
import sim


SCHEMA_VERSION = 1
DEFAULT_CASES = ("baseline", "adaptive_t0_i100ms")
DEFAULT_OUTPUT = Path(
    "results/ms_scale_control/npu32_ssu4_microtrace_baseline_adaptive100_v1"
)
EPS = 1e-9
SEMANTIC_REPLAY_OF = "7e994146"
EXPECTED_BY_NUM_SSU = {
    4: {
        "input_fingerprint": (
            "a97bf59f0850daa06469a7ecc458ed2110a4ae96f2aecb33da25b374fe36fc9d"
        ),
        "workload_hash": (
            "267adff09227a4f2620963cb1152ec2d257b2d2d4672b543229a45ae64f87dcc"
        ),
        "placement_hash": (
            "2d81460ca3cff20ecb2c5a098fc7bc8513969bef6d2656b048d2485022473257"
        ),
        "trace_hash": (
            "371ce3e7017aa7baafe0d7efd1b7eced08807f3b8f1cb7853c9fa454f2b4ef6d"
        ),
        "measurement_start_ms": {
            "baseline": 10136.618340562687,
            "adaptive_t0_i100ms": 10906.062311812053,
        },
    },
    5: {
        "input_fingerprint": (
            "f3cf060f79ebca9284c143be891d4cc6cbf34988a576c6501c4a4511f84af6a0"
        ),
        "workload_hash": (
            "5a8f2df85915fcbad4e8ebe874d66598eb563aaf3a634d1d46d745e2ae8038c3"
        ),
        "placement_hash": (
            "921a020b04599be808e3c62d4872be736582bc9f2865b2e0a675dd5d3e0e9923"
        ),
        "trace_hash": (
            "062694374dedd96e3b12a90bf8ba0e2a2087e0e1b7d9956930145124f2f5bdeb"
        ),
        "measurement_start_ms": {
            "baseline": 9533.21367413627,
            "adaptive_t0_i100ms": 9863.429933979758,
        },
    },
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def _atomic_write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _atomic_write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {name: _csv_value(row.get(name)) for name in fieldnames}
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overlap_fields(start_ms, end_ms, trace_start_ms, trace_end_ms, bw_gbps=None):
    overlap_start_ms = max(float(start_ms), trace_start_ms)
    overlap_end_ms = min(float(end_ms), trace_end_ms)
    overlap_ms = max(0.0, overlap_end_ms - overlap_start_ms)
    return {
        "trace_overlap_start_ms": overlap_start_ms,
        "trace_overlap_end_ms": overlap_end_ms,
        "trace_overlap_ms": overlap_ms,
        "trace_served_gb": (
            None if bw_gbps is None else float(bw_gbps) * overlap_ms / 1000.0
        ),
        "straddles_measurement_start": bool(start_ms < trace_start_ms < end_ms),
        "straddles_trace_end": bool(start_ms < trace_end_ms < end_ms),
    }


class MicrotraceRecorder:
    """Mutable sink used only by temporary wrappers around the DES."""

    def __init__(self, case_name: str, trace_ms: float):
        self.case_name = case_name
        self.trace_ms = float(trace_ms)
        self.trace_start_ms = None
        self.trace_end_ms = None
        self.context = None
        self.ssd_dispatch_intervals = []
        self.npu_link_intervals = []
        self.compute_intervals = []
        self.cir_table_events = []
        self._ssd_keys = set()
        self._link_keys = set()
        self._compute_keys = set()

    @property
    def active(self):
        return self.trace_start_ms is not None

    def contains_event_time(self, time_ms):
        return self.active and self.trace_start_ms <= time_ms < self.trace_end_ms

    def overlaps_trace(self, start_ms, end_ms):
        return (
            self.active
            and float(start_ms) < self.trace_end_ms
            and float(end_ms) > self.trace_start_ms
        )

    def _flow_key(self, flow):
        return (
            int(flow.request_id),
            int(flow.npu_id),
            int(flow.layer),
            int(flow.block_idx),
            int(flow.disk_id),
        )

    def begin_measurement(self, context, current_time_ms):
        if self.active:
            raise AssertionError("microtrace measurement boundary occurred twice")
        self.context = context
        self.trace_start_ms = float(current_time_ms)
        self.trace_end_ms = self.trace_start_ms + self.trace_ms

        # Capture intervals already in flight.  The SSD wrapper stores the
        # original dispatch time because scheduler.settle() advances
        # BlockIOFlow.start_time while preserving the non-preemptive end time.
        for disk in context.disks:
            if disk.active_flows:
                flow = disk.active_flows[0]
                start_ms = getattr(
                    flow, "_microtrace_ssd_dispatch_start_ms", flow.start_time
                )
                self.record_ssd_dispatch(disk.scheduler, flow, start_ms)
        for npu in context.npus:
            if npu.link_active_flow is not None:
                self.record_npu_link(
                    context,
                    npu,
                    npu.link_active_flow,
                    npu.link_active_flow.link_start_time,
                )
            batch = npu.active_batch
            if batch is not None and batch.compute_active_layer >= 0:
                self.record_compute(context, npu, batch)
        self.record_cir_table_event(
            context,
            current_time_ms,
            event_kind="measurement_start_snapshot",
            before_tables=None,
            decision_present=None,
        )

    def record_ssd_dispatch(self, scheduler, flow, start_ms):
        end_ms = float(flow.end_time)
        if not self.overlaps_trace(start_ms, end_ms):
            return
        key = self._flow_key(flow)
        if key in self._ssd_keys:
            return
        self._ssd_keys.add(key)
        path_cir = getattr(flow, "_microtrace_ssd_path_cir_gbps", None)
        row = {
            "case": self.case_name,
            "event_index": len(self.ssd_dispatch_intervals),
            "request_id": int(flow.request_id),
            "npu_id": int(flow.npu_id),
            "layer": int(flow.layer),
            "block_idx": int(flow.block_idx),
            "ssu_id": int(flow.disk_id),
            "path_id": int(flow.queue_id),
            "size_gb": float(flow.total_gb),
            "ssd_enqueue_time_ms": float(flow.enqueue_time),
            "ssd_start_time_ms": float(start_ms),
            "ssd_end_time_ms": end_ms,
            "ssd_service_time_ms": end_ms - float(start_ms),
            "ssd_queue_wait_ms": float(start_ms) - float(flow.enqueue_time),
            "actual_ssd_bw_gbps": float(flow.bw),
            "installed_path_cir_gbps_at_dispatch": (
                None if path_cir is None else float(path_cir)
            ),
            **_overlap_fields(
                start_ms,
                end_ms,
                self.trace_start_ms,
                self.trace_end_ms,
                flow.bw,
            ),
        }
        if row["trace_overlap_ms"] <= 0.0:
            raise AssertionError("recorded an SSD interval outside the trace")
        self.ssd_dispatch_intervals.append(row)

    def record_npu_link(self, context, npu, flow, start_ms):
        end_ms = float(flow.link_end_time)
        if not self.overlaps_trace(start_ms, end_ms):
            return
        key = self._flow_key(flow)
        if key in self._link_keys:
            return
        self._link_keys.add(key)
        bw_gbps = float(context.npu_bw_gbps)
        row = {
            "case": self.case_name,
            "event_index": len(self.npu_link_intervals),
            "request_id": int(flow.request_id),
            "npu_id": int(flow.npu_id),
            "layer": int(flow.layer),
            "block_idx": int(flow.block_idx),
            "ssu_id": int(flow.disk_id),
            "path_id": int(flow.queue_id),
            "size_gb": float(flow.total_gb),
            "ssd_enqueue_time_ms": float(flow.enqueue_time),
            "ssd_start_time_ms": float(
                getattr(flow, "_microtrace_ssd_dispatch_start_ms", flow.start_time)
            ),
            "ssd_end_time_ms": float(flow.end_time),
            "link_enqueue_time_ms": float(flow.link_enqueue_time),
            "link_start_time_ms": float(start_ms),
            "link_end_time_ms": end_ms,
            "link_service_time_ms": end_ms - float(start_ms),
            "link_queue_wait_ms": float(start_ms) - float(flow.link_enqueue_time),
            "actual_npu_link_bw_gbps": bw_gbps,
            "installed_path_cir_gbps_at_ssd_dispatch": getattr(
                flow, "_microtrace_ssd_path_cir_gbps", None
            ),
            **_overlap_fields(
                start_ms,
                end_ms,
                self.trace_start_ms,
                self.trace_end_ms,
                bw_gbps,
            ),
        }
        if row["trace_overlap_ms"] <= 0.0:
            raise AssertionError("recorded an NPU-link interval outside the trace")
        self.npu_link_intervals.append(row)

    def record_compute(self, context, npu, batch):
        layer = int(batch.compute_active_layer)
        if layer < 0:
            return
        metric = batch.layer_metrics[layer]
        start_ms = float(metric.compute_start_ms)
        end_ms = float(metric.compute_end_ms)
        if not self.overlaps_trace(start_ms, end_ms):
            return
        key = (int(batch.batch_id), int(npu.npu_id), layer)
        if key in self._compute_keys:
            return
        self._compute_keys.add(key)
        member_ids = [int(value) for value in batch.member_request_ids]
        request_states = [context.requests[value] for value in member_ids]
        row = {
            "case": self.case_name,
            "event_index": len(self.compute_intervals),
            "batch_id": int(batch.batch_id),
            "request_id": member_ids[0] if len(member_ids) == 1 else None,
            "member_request_ids": member_ids,
            "npu_id": int(npu.npu_id),
            "layer": layer,
            "categories": [state.category for state in request_states],
            "admission_time_ms": min(
                float(state.admission_time_ms) for state in request_states
            ),
            "io_ready_time_ms": float(metric.io_ready_time_ms),
            "io_barrier_wait_ms": float(metric.io_barrier_wait_ms),
            "compute_start_time_ms": start_ms,
            "compute_end_time_ms": end_ms,
            "compute_duration_ms": end_ms - start_ms,
            **_overlap_fields(
                start_ms,
                end_ms,
                self.trace_start_ms,
                self.trace_end_ms,
            ),
        }
        if row["trace_overlap_ms"] <= 0.0:
            raise AssertionError("recorded a compute interval outside the trace")
        self.compute_intervals.append(row)

    @staticmethod
    def _installed_tables(context):
        return [
            [float(path.cir) for path in disk.scheduler.paths.values()]
            for disk in context.disks
        ]

    @staticmethod
    def _installed_npu_sums(context, tables):
        if context.npu_dedicated_paths is None:
            return None
        return [
            math.fsum(table[path_id] for table in tables)
            for path_id in context.npu_dedicated_paths
        ]

    def record_cir_table_event(
        self,
        context,
        current_time_ms,
        *,
        event_kind,
        before_tables,
        decision_present,
    ):
        if not self.contains_event_time(float(current_time_ms)):
            return
        after_tables = self._installed_tables(context)
        changed_entries = []
        if before_tables is not None:
            for ssu_id, (before, after) in enumerate(
                zip(before_tables, after_tables)
            ):
                for path_id, (old, new) in enumerate(zip(before, after)):
                    if old != new:
                        changed_entries.append(
                            {
                                "ssu_id": ssu_id,
                                "path_id": path_id,
                                "old_cir_gbps": old,
                                "new_cir_gbps": new,
                            }
                        )
        event = {
            "case": self.case_name,
            "event_index": len(self.cir_table_events),
            "event_kind": event_kind,
            "time_ms": float(current_time_ms),
            "relative_time_ms": float(current_time_ms) - self.trace_start_ms,
            "control_evaluation": int(context.control_evaluations),
            "decision_present": decision_present,
            "changed_entry_count": len(changed_entries),
            "changed_entries": changed_entries,
            "installed_tables_by_ssu_gbps": after_tables,
            "installed_sum_gbps_by_ssu": [math.fsum(row) for row in after_tables],
            "installed_sum_gbps_by_npu": self._installed_npu_sums(
                context, after_tables
            ),
            "npu_dedicated_paths": (
                None
                if context.npu_dedicated_paths is None
                else list(context.npu_dedicated_paths)
            ),
        }
        self.cir_table_events.append(event)

    def as_dict(self):
        if not self.active:
            raise AssertionError("steady-state measurement never started")
        return {
            "schema_version": SCHEMA_VERSION,
            "case": self.case_name,
            "window_semantics": "half-open [trace_start_ms, trace_end_ms)",
            "interval_semantics": (
                "full exact intervals that overlap the trace; clipped contribution "
                "is in trace_overlap_ms/trace_served_gb"
            ),
            "trace_start_ms": self.trace_start_ms,
            "trace_end_ms": self.trace_end_ms,
            "trace_duration_ms": self.trace_ms,
            "ssd_dispatch_intervals": self.ssd_dispatch_intervals,
            "npu_link_intervals": self.npu_link_intervals,
            "compute_intervals": self.compute_intervals,
            "cir_table_events": self.cir_table_events,
        }


@contextmanager
def _installed_trace_wrappers(recorder: MicrotraceRecorder):
    """Install process-local wrappers and restore every original on exit."""

    original_measurement_start = cb._handle_steady_measurement_start
    original_ssd_dispatch_one = sim.DiskIOScheduler._dispatch_one
    original_link_start = cb._start_next_link_io
    original_compute_schedule = cb._handle_compute_schedule
    original_apply_cir = cb._apply_cir_decision

    def measurement_start_wrapper(context, current_time_ms):
        recorder.begin_measurement(context, current_time_ms)
        return original_measurement_start(context, current_time_ms)

    def ssd_dispatch_wrapper(scheduler, current_time_ms):
        before = scheduler.state.active_flows[0] if scheduler.state.active_flows else None
        flow = original_ssd_dispatch_one(scheduler, current_time_ms)
        if flow is not None and flow is not before and math.isfinite(flow.end_time):
            # Preserve facts that scheduler.settle() later mutates or whose
            # installed CIR can change before the measurement boundary.
            flow._microtrace_ssd_dispatch_start_ms = float(current_time_ms)
            flow._microtrace_ssd_path_cir_gbps = (
                float(scheduler.paths[flow.queue_id].cir)
                if flow.queue_id in scheduler.paths
                else None
            )
            recorder.record_ssd_dispatch(scheduler, flow, current_time_ms)
        return flow

    def link_start_wrapper(context, npu, current_time_ms):
        before = npu.link_active_flow
        result = original_link_start(context, npu, current_time_ms)
        after = npu.link_active_flow
        if after is not None and after is not before:
            recorder.record_npu_link(context, npu, after, after.link_start_time)
        return result

    def compute_schedule_wrapper(context, npu_id, current_time_ms):
        npu = context.npus[npu_id]
        before = npu.compute_active
        result = original_compute_schedule(context, npu_id, current_time_ms)
        if npu.compute_active is not None and npu.compute_active != before:
            recorder.record_compute(context, npu, npu.active_batch)
        return result

    def apply_cir_wrapper(context, decision, current_time_ms):
        should_record = recorder.contains_event_time(float(current_time_ms))
        before = recorder._installed_tables(context) if should_record else None
        result = original_apply_cir(context, decision, current_time_ms)
        if should_record:
            recorder.record_cir_table_event(
                context,
                current_time_ms,
                event_kind="atomic_fleet_apply",
                before_tables=before,
                decision_present=decision is not None,
            )
        return result

    cb._handle_steady_measurement_start = measurement_start_wrapper
    sim.DiskIOScheduler._dispatch_one = ssd_dispatch_wrapper
    cb._start_next_link_io = link_start_wrapper
    cb._handle_compute_schedule = compute_schedule_wrapper
    cb._apply_cir_decision = apply_cir_wrapper
    try:
        yield
    finally:
        cb._handle_steady_measurement_start = original_measurement_start
        sim.DiskIOScheduler._dispatch_one = original_ssd_dispatch_one
        cb._start_next_link_io = original_link_start
        cb._handle_compute_schedule = original_compute_schedule
        cb._apply_cir_decision = original_apply_cir


def _sum_by(rows, resource_count, resource_field, value_field):
    values = [0.0] * resource_count
    for row in rows:
        values[int(row[resource_field])] += float(row[value_field])
    return values


def _vectors_close(left, right, tolerance=1e-7):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def _validate_trace(summary, trace):
    start_ms = float(trace["trace_start_ms"])
    end_ms = float(trace["trace_end_ms"])
    duration_ms = end_ms - start_ms
    summary_start_matches = math.isclose(
        start_ms, float(summary["measurement_start_ms"]), abs_tol=EPS
    )
    if not summary_start_matches:
        raise AssertionError("microtrace and summary measurement starts differ")
    expected_historical_start_ms = EXPECTED_BY_NUM_SSU[int(summary["num_ssu"])][
        "measurement_start_ms"
    ][trace["case"]]
    historical_start_matches = math.isclose(
        start_ms,
        expected_historical_start_ms,
        rel_tol=0.0,
        abs_tol=EPS,
    )

    blocks = [
        row
        for row in summary["measurement_blocks"]
        if row["start_ms"] >= start_ms - EPS and row["end_ms"] <= end_ms + EPS
    ]
    covered_ms = math.fsum(float(row["duration_ms"]) for row in blocks)
    exact_block_coverage = math.isclose(covered_ms, duration_ms, abs_tol=EPS)

    observed_ssd_ms = _sum_by(
        trace["ssd_dispatch_intervals"],
        summary["num_ssu"],
        "ssu_id",
        "trace_overlap_ms",
    )
    observed_compute_ms = _sum_by(
        trace["compute_intervals"],
        summary["num_npu"],
        "npu_id",
        "trace_overlap_ms",
    )
    observed_link_ms = _sum_by(
        trace["npu_link_intervals"],
        summary["num_npu"],
        "npu_id",
        "trace_overlap_ms",
    )
    expected_ssd_ms = [
        math.fsum(float(row["ssd_busy_ms_by_ssu"][index]) for row in blocks)
        for index in range(summary["num_ssu"])
    ]
    expected_compute_ms = [
        math.fsum(float(row["compute_ms_by_npu"][index]) for row in blocks)
        for index in range(summary["num_npu"])
    ]
    expected_link_ms = [
        math.fsum(float(row["npu_link_busy_ms_by_npu"][index]) for row in blocks)
        for index in range(summary["num_npu"])
    ]

    ssd_service_identity = all(
        math.isclose(
            row["trace_served_gb"],
            row["actual_ssd_bw_gbps"] * row["trace_overlap_ms"] / 1000.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in trace["ssd_dispatch_intervals"]
    )
    link_service_identity = all(
        math.isclose(
            row["trace_served_gb"],
            row["actual_npu_link_bw_gbps"]
            * row["trace_overlap_ms"]
            / 1000.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in trace["npu_link_intervals"]
    )
    checks = {
        "trace_start_matches_summary": summary_start_matches,
        "measurement_start_matches_historical_bridge": historical_start_matches,
        "trace_exactly_covered_by_summary_blocks": exact_block_coverage,
        "ssd_interval_overlap_matches_summary_blocks": (
            exact_block_coverage and _vectors_close(observed_ssd_ms, expected_ssd_ms)
        ),
        "compute_interval_overlap_matches_summary_blocks": (
            exact_block_coverage
            and _vectors_close(observed_compute_ms, expected_compute_ms)
        ),
        "npu_link_interval_overlap_matches_summary_blocks": (
            exact_block_coverage and _vectors_close(observed_link_ms, expected_link_ms)
        ),
        "ssd_service_time_bandwidth_identity": ssd_service_identity,
        "npu_link_service_time_bandwidth_identity": link_service_identity,
        "ssd_single_command_bound": all(value <= duration_ms + EPS for value in observed_ssd_ms),
        "npu_single_compute_bound": all(
            value <= duration_ms + EPS for value in observed_compute_ms
        ),
        "npu_single_link_command_bound": all(
            value <= duration_ms + EPS for value in observed_link_ms
        ),
        "cir_snapshot_at_trace_start": bool(trace["cir_table_events"])
        and trace["cir_table_events"][0]["event_kind"]
        == "measurement_start_snapshot",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary_block_coverage_ms": covered_ms,
        "historical_bridge": {
            "semantic_replay_of": SEMANTIC_REPLAY_OF,
            "expected_measurement_start_ms": expected_historical_start_ms,
            "actual_measurement_start_ms": start_ms,
            "absolute_residual_ms": abs(start_ms - expected_historical_start_ms),
            "absolute_tolerance_ms": EPS,
        },
        "observed_ssd_busy_ms_by_ssu": observed_ssd_ms,
        "expected_ssd_busy_ms_by_ssu": expected_ssd_ms,
        "observed_compute_ms_by_npu": observed_compute_ms,
        "expected_compute_ms_by_npu": expected_compute_ms,
        "observed_npu_link_busy_ms_by_npu": observed_link_ms,
        "expected_npu_link_busy_ms_by_npu": expected_link_ms,
        "maximum_absolute_residual_ms": max(
            [0.0]
            + [abs(a - b) for a, b in zip(observed_ssd_ms, expected_ssd_ms)]
            + [abs(a - b) for a, b in zip(observed_compute_ms, expected_compute_ms)]
            + [abs(a - b) for a, b in zip(observed_link_ms, expected_link_ms)]
        ),
    }


SSD_FIELDS = (
    "case",
    "event_index",
    "request_id",
    "npu_id",
    "layer",
    "block_idx",
    "ssu_id",
    "path_id",
    "size_gb",
    "ssd_enqueue_time_ms",
    "ssd_start_time_ms",
    "ssd_end_time_ms",
    "ssd_service_time_ms",
    "ssd_queue_wait_ms",
    "actual_ssd_bw_gbps",
    "installed_path_cir_gbps_at_dispatch",
    "trace_overlap_start_ms",
    "trace_overlap_end_ms",
    "trace_overlap_ms",
    "trace_served_gb",
    "straddles_measurement_start",
    "straddles_trace_end",
)
LINK_FIELDS = (
    "case",
    "event_index",
    "request_id",
    "npu_id",
    "layer",
    "block_idx",
    "ssu_id",
    "path_id",
    "size_gb",
    "ssd_enqueue_time_ms",
    "ssd_start_time_ms",
    "ssd_end_time_ms",
    "link_enqueue_time_ms",
    "link_start_time_ms",
    "link_end_time_ms",
    "link_service_time_ms",
    "link_queue_wait_ms",
    "actual_npu_link_bw_gbps",
    "installed_path_cir_gbps_at_ssd_dispatch",
    "trace_overlap_start_ms",
    "trace_overlap_end_ms",
    "trace_overlap_ms",
    "trace_served_gb",
    "straddles_measurement_start",
    "straddles_trace_end",
)
COMPUTE_FIELDS = (
    "case",
    "event_index",
    "batch_id",
    "request_id",
    "member_request_ids",
    "npu_id",
    "layer",
    "categories",
    "admission_time_ms",
    "io_ready_time_ms",
    "io_barrier_wait_ms",
    "compute_start_time_ms",
    "compute_end_time_ms",
    "compute_duration_ms",
    "trace_overlap_start_ms",
    "trace_overlap_end_ms",
    "trace_overlap_ms",
    "straddles_measurement_start",
    "straddles_trace_end",
)
CIR_FIELDS = (
    "case",
    "event_index",
    "event_kind",
    "time_ms",
    "relative_time_ms",
    "control_evaluation",
    "decision_present",
    "changed_entry_count",
    "ssu_id",
    "path_id",
    "npu_id",
    "installed_cir_gbps",
    "old_cir_gbps",
    "new_cir_gbps",
    "changed",
    "installed_sum_gbps_by_ssu",
    "installed_sum_gbps_by_npu",
)


def _flatten_cir_events(events):
    rows = []
    for event in events:
        changes = {
            (row["ssu_id"], row["path_id"]): row
            for row in event["changed_entries"]
        }
        paths = event["npu_dedicated_paths"]
        path_to_npu = (
            {} if paths is None else {path_id: npu_id for npu_id, path_id in enumerate(paths)}
        )
        for ssu_id, table in enumerate(event["installed_tables_by_ssu_gbps"]):
            for path_id, cir in enumerate(table):
                change = changes.get((ssu_id, path_id))
                rows.append(
                    {
                        "case": event["case"],
                        "event_index": event["event_index"],
                        "event_kind": event["event_kind"],
                        "time_ms": event["time_ms"],
                        "relative_time_ms": event["relative_time_ms"],
                        "control_evaluation": event["control_evaluation"],
                        "decision_present": event["decision_present"],
                        "changed_entry_count": event["changed_entry_count"],
                        "ssu_id": ssu_id,
                        "path_id": path_id,
                        "npu_id": path_to_npu.get(path_id),
                        "installed_cir_gbps": cir,
                        "old_cir_gbps": None if change is None else change["old_cir_gbps"],
                        "new_cir_gbps": None if change is None else change["new_cir_gbps"],
                        "changed": change is not None,
                        "installed_sum_gbps_by_ssu": event[
                            "installed_sum_gbps_by_ssu"
                        ],
                        "installed_sum_gbps_by_npu": event[
                            "installed_sum_gbps_by_npu"
                        ],
                    }
                )
    return rows


def _write_case_outputs(output_dir, case_name, summary, trace, validation):
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(case_dir / "summary.json", summary)
    _atomic_write_json(
        case_dir / "microtrace.json",
        {**trace, "validation": validation},
    )
    _atomic_write_csv(
        case_dir / "ssd_dispatch_intervals.csv",
        trace["ssd_dispatch_intervals"],
        SSD_FIELDS,
    )
    _atomic_write_csv(
        case_dir / "npu_link_intervals.csv",
        trace["npu_link_intervals"],
        LINK_FIELDS,
    )
    _atomic_write_csv(
        case_dir / "compute_intervals.csv",
        trace["compute_intervals"],
        COMPUTE_FIELDS,
    )
    _atomic_write_csv(
        case_dir / "cir_installed_tables.csv",
        _flatten_cir_events(trace["cir_table_events"]),
        CIR_FIELDS,
    )


def _write_combined_outputs(output_dir, completed):
    for filename, key, fields in (
        ("ssd_dispatch_intervals.csv", "ssd_dispatch_intervals", SSD_FIELDS),
        ("npu_link_intervals.csv", "npu_link_intervals", LINK_FIELDS),
        ("compute_intervals.csv", "compute_intervals", COMPUTE_FIELDS),
    ):
        rows = [row for item in completed for row in item["trace"][key]]
        _atomic_write_csv(output_dir / filename, rows, fields)
    cir_rows = [
        row
        for item in completed
        for row in _flatten_cir_events(item["trace"]["cir_table_events"])
    ]
    _atomic_write_csv(output_dir / "cir_installed_tables.csv", cir_rows, CIR_FIELDS)


def _build_inputs(args):
    definition = LEGACY32_DEFINITION
    if definition.num_npu != 32 or definition.n_layers != 16 or definition.batch_size != 1:
        raise AssertionError("legacy32 topology unexpectedly changed")
    table, authentication = load_authenticated_bw_table(definition.num_npu)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=IID_UNIFORM_PROFILE_CATALOG_V1,
        seed=args.seed,
        num_npu=definition.num_npu,
        requests_per_npu=args.requests_per_npu,
    )
    workload = prepare_random_steady_state_workload(
        table,
        schedule=schedule,
        num_ssu=args.num_ssu,
        n_layers=definition.n_layers,
    )
    requests = requests_from_continuous_prefill_workload(workload)
    observed_gate = {
        "input_fingerprint": continuous_batch_input_fingerprint(requests),
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
    }
    expected = EXPECTED_BY_NUM_SSU[args.num_ssu]
    expected_gate = {
        key: expected[key]
        for key in (
            "input_fingerprint",
            "workload_hash",
            "placement_hash",
            "trace_hash",
        )
    }
    if observed_gate != expected_gate:
        raise AssertionError(
            "historical input bridge gate failed: "
            f"observed={observed_gate}, expected={expected_gate}"
        )
    config = RunConfig(
        seed=args.seed,
        requests_per_npu=args.requests_per_npu,
        warmup_requests_per_npu=args.warmup_requests,
        settle_ms=args.settle_ms,
        measurement_ms=args.measurement_ms,
        block_ms=args.block_ms,
        slo_alpha=args.slo_alpha,
    )
    spec = _experiment_spec(definition, schedule, config, authentication)
    return definition, workload, requests, config, authentication, spec


def _initial_manifest(args, definition, workload, requests, config, authentication, spec):
    runner_path = Path(__file__).resolve()
    expected = EXPECTED_BY_NUM_SSU[args.num_ssu]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "preflight" if args.preflight else "running",
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "runner": str(runner_path),
        "runner_sha256": _sha256(runner_path),
        "source_fingerprint": _source_fingerprint(),
        "config_fingerprint": _config_fingerprint(spec),
        "definition": definition.key,
        "cases_requested": list(args.case),
        "completed_cases": [],
        "num_npu": definition.num_npu,
        "num_ssu": args.num_ssu,
        "n_layers": definition.n_layers,
        "batch_size": definition.batch_size,
        "seed": config.seed,
        "requests_per_npu": config.requests_per_npu,
        "warmup_requests_per_npu": config.warmup_requests_per_npu,
        "settle_ms": config.settle_ms,
        "measurement_ms": config.measurement_ms,
        "block_ms": config.block_ms,
        "slo_alpha": config.slo_alpha,
        "trace_ms": args.trace_ms,
        "input_fingerprint": continuous_batch_input_fingerprint(requests),
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "input_authentication": authentication,
        "historical_bridge": {
            "provenance": f"semantic_replay_of_{SEMANTIC_REPLAY_OF}",
            "is_original_historical_trace": False,
            "meaning": (
                "current-HEAD diagnostic instrumentation replaying the exact "
                "historical input and pre-measurement semantics"
            ),
            "expected_input_fingerprint": expected["input_fingerprint"],
            "expected_workload_hash": expected["workload_hash"],
            "expected_placement_hash": expected["placement_hash"],
            "expected_trace_hash": expected["trace_hash"],
            "expected_measurement_start_ms_by_case": expected[
                "measurement_start_ms"
            ],
            "absolute_time_tolerance_ms": EPS,
        },
        "event_model": {
            "ssd": (
                "one non-preemptive command per SSU at actual_ssd_bw_gbps; CIR "
                "changes discrete command arbitration order, not in-flight bandwidth"
            ),
            "npu_link": "one FCFS command per NPU at actual_npu_link_bw_gbps",
            "compute": "one layer compute interval per NPU",
            "cir": "coherent installed fleet table after each atomic DES apply",
            "trace_window": "half-open and includes full boundary-straddling intervals",
        },
        "experiment_spec": spec,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--case",
        action="append",
        choices=DEFAULT_CASES,
        help="repeat to select cases; default runs the paired baseline and Adaptive",
    )
    parser.add_argument("--num-ssu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests-per-npu", type=int, default=128)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--settle-ms", type=float, default=500.0)
    parser.add_argument("--measurement-ms", type=float, default=3_000.0)
    parser.add_argument("--block-ms", type=float, default=5.0)
    parser.add_argument("--slo-alpha", type=float, default=2.0)
    parser.add_argument("--trace-ms", type=float, default=50.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="materialize and authenticate the paired input without simulating",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace outputs for selected cases in an existing diagnostic directory",
    )
    args = parser.parse_args(argv)
    if args.case is None:
        args.case = list(DEFAULT_CASES)
    if len(set(args.case)) != len(args.case):
        parser.error("--case values must be unique")
    if args.num_ssu not in EXPECTED_BY_NUM_SSU:
        parser.error("this diagnostic has authenticated bridge gates for --num-ssu 4 or 5")
    if args.seed != 42:
        parser.error("this diagnostic is intentionally frozen to --seed 42")
    if args.requests_per_npu != 128:
        parser.error("this diagnostic is intentionally frozen to 128 requests/NPU")
    if args.warmup_requests != 8 or args.settle_ms != 500.0:
        parser.error("this diagnostic is frozen to warmup=8 and settle=500ms")
    if args.measurement_ms != 3_000.0:
        parser.error("this diagnostic is frozen to a 3000ms measurement")
    if args.trace_ms <= 0.0 or args.trace_ms > args.measurement_ms:
        parser.error("--trace-ms must be in (0, measurement-ms]")
    if args.block_ms <= 0.0 or args.trace_ms % args.block_ms != 0.0:
        parser.error("--block-ms must divide --trace-ms exactly")
    return args


def main(argv=None):
    args = parse_args(argv)
    definition, workload, requests, config, authentication, spec = _build_inputs(args)
    manifest = _initial_manifest(
        args,
        definition,
        workload,
        requests,
        config,
        authentication,
        spec,
    )
    if args.preflight:
        print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
        return 0

    output_dir = args.output_dir.resolve()
    existing_targets = [
        output_dir / case_name
        for case_name in args.case
        if (output_dir / case_name).exists()
    ]
    if existing_targets and not args.force:
        joined = ", ".join(str(path) for path in existing_targets)
        raise FileExistsError(f"case output already exists (use --force): {joined}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / "manifest.json", manifest)

    completed = []
    reference_input_fingerprint = manifest["input_fingerprint"]
    expected_input_fingerprint = EXPECTED_BY_NUM_SSU[args.num_ssu][
        "input_fingerprint"
    ]
    for case_name in args.case:
        case = definition.case_by_name[case_name]
        recorder = MicrotraceRecorder(case_name, args.trace_ms)
        started = time.monotonic()
        with _installed_trace_wrappers(recorder):
            summary = _simulate(
                definition,
                config,
                case,
                args.num_ssu,
                requests,
            )
        elapsed_s = time.monotonic() - started
        if (
            summary["input_fingerprint"] != reference_input_fingerprint
            or summary["input_fingerprint"] != expected_input_fingerprint
        ):
            raise AssertionError("strategy changed the paired simulator input")
        trace = recorder.as_dict()
        validation = _validate_trace(summary, trace)
        _write_case_outputs(
            output_dir,
            case_name,
            summary,
            trace,
            validation,
        )
        completed.append(
            {
                "case": case_name,
                "summary": summary,
                "trace": trace,
                "validation": validation,
                "wall_time_seconds": elapsed_s,
            }
        )
        manifest["completed_cases"].append(
            {
                "case": case_name,
                "wall_time_seconds": elapsed_s,
                "mean_npu_utilization": summary["mean_npu_utilization"],
                "ttft_slo_attainment": summary["ttft_slo_attainment"],
                "trace_validation_passed": validation["passed"],
                "trace_counts": {
                    "ssd_dispatch_intervals": len(trace["ssd_dispatch_intervals"]),
                    "npu_link_intervals": len(trace["npu_link_intervals"]),
                    "compute_intervals": len(trace["compute_intervals"]),
                    "cir_table_events": len(trace["cir_table_events"]),
                },
            }
        )
        manifest["updated_utc"] = _utc_now()
        _atomic_write_json(output_dir / "manifest.json", manifest)
        if not validation["passed"]:
            raise AssertionError(
                f"{case_name} microtrace failed cumulative-accounting validation: "
                f"{validation['checks']}"
            )

    _write_combined_outputs(output_dir, completed)
    manifest["status"] = "complete"
    manifest["updated_utc"] = _utc_now()
    manifest["paired_input_fingerprints_equal"] = (
        len({item["summary"]["input_fingerprint"] for item in completed}) == 1
    )
    _atomic_write_json(output_dir / "manifest.json", manifest)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
