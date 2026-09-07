#!/usr/bin/env python3
"""Validate the independent multi-SSU equations against unchanged native code.

Cold cases supply only manifests, environment and the public run seed, never
native output traces. Warm snapshots discard SSD FIFO order and RNG position;
their differences are intentionally reported rather than relabeled exact.
"""
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from unittest.mock import patch

import continuous_batch_sim as native
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from multi_ssu_stall_predictor import (
    IO_BYTES, NPUState, Request, forecast_baseline, impact_order_sensitivity, io_service_ms,
    predict_request_impact, screen_current_layer, screen_pending_layer,
)
from validate_stall_predictor import comparison
import sim


def portable_request(request):
    return Request(request.request_id,
                   tuple(tuple(ssu for ssu, _ in layer) for layer in request.placement),
                   request.load["per_layer_us"] / 1000.0, 8)


def make_request(rid, npu, blocks, compute_ms, num_ssu, *, arrival_ms=0.0,
                 placement_offset=0, placement_mode="stripe", nql=128):
    if placement_mode == "hotspot":
        disks = tuple(0 if k % 3 else (k + npu) % num_ssu for k in range(blocks))
    else:
        disks = tuple((k + placement_offset) % num_ssu for k in range(blocks))
    seq_len_k = (blocks * 128 + nql) / 1024.0
    load = {"request_id": rid, "npu_id": npu, "nql": nql, "seq_len_k": seq_len_k,
            "category": sim.classify_request(seq_len_k, nql),
            "per_layer_us": compute_ms * 1000.0,
            "per_layer_kv_gb": blocks * IO_BYTES / 2**30,
            "required_bw_input_gbps": blocks * IO_BYTES / 2**30 * 1000.0 / compute_ms,
            "arrival_ms": arrival_ms, "arrival_time": arrival_ms, "initial": arrival_ms == 0}
    return native.ContinuousBatchRequest(rid, npu, arrival_ms, load,
                                         (tuple((s, IO_BYTES / 2**30) for s in disks),))


def run_native(requests, num_npu, num_ssu, seed=42):
    spec = next(s for s in routing_strategy_specs() if s.name == "baseline")
    return native.simulate_continuous_batch(
        requests, num_npu=num_npu, num_ssu=num_ssu, n_layers=8, batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR, qos_config=static_qos_config(),
        cross_request_layer0_prefetch=True, client_io_config=spec.client_config(),
        disk_bw_gbps=40.0, npu_bw_gbps=50.0, submit_order_seed=seed)


def cold_states(requests, num_npu):
    lanes = [[] for _ in range(num_npu)]
    for request in sorted(requests, key=lambda r: (r.arrival_time_ms, r.npu_id, r.request_id)):
        lanes[request.npu_id].append(portable_request(request))
    return tuple(NPUState(lane[0], waiting_requests=tuple(lane[1:])) if lane else NPUState()
                 for lane in lanes)


def cold_case(name, requests, num_npu, num_ssu, seed=42):
    started = time.perf_counter()
    prediction = forecast_baseline(cold_states(requests, num_npu), num_ssu=num_ssu,
                                   issue_order_seed=seed)
    predictor_seconds = time.perf_counter() - started
    started = time.perf_counter()
    summary = run_native(requests, num_npu, num_ssu, seed)
    native_seconds = time.perf_counter() - started
    return {"name": name, "num_npu": num_npu, "num_ssu": num_ssu, "seed": seed,
            "input_fingerprint": native.continuous_batch_input_fingerprint(requests),
            "request_count": len(requests),
            "predictor_wall_seconds": predictor_seconds, "native_wall_seconds": native_seconds,
            "prediction_scope": "cold closed-loop manifests and public seed, not trace replay",
            "assumptions": prediction["assumptions"],
            "comparison": comparison(prediction, summary)}


class CountSnapshotObserver:
    """Per-owner counters + client emitter state; deliberately no SSD FIFO read."""
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.submitted = defaultdict(int)
        self.disk_done = defaultdict(int)
        self.link_done = defaultdict(int)
        self.snapshot = None
        self.time_ms = None

    def capture(self, context, now):
        states = []
        for lane in context.npus:
            waiting = tuple(portable_request(context.requests[rid].manifest)
                            for rid in lane.admission_queue)
            current, next_layer, end = None, 0, 0.0
            head_id, head_layer = None, 0
            batch = lane.active_batch
            if batch is not None:
                current = portable_request(context.requests[batch.member_request_ids[0]].manifest)
                if lane.compute_active is not None:
                    active_layer = lane.compute_active[1]
                    next_layer = active_layer + 1
                    end = batch.layer_metrics[active_layer].compute_end_ms - now
                else:
                    next_layer = batch.compute_done_up_to + 1
                    end = batch.previous_compute_end_ms - now
                if next_layer < current.layers:
                    head_id, head_layer = current.request_id, next_layer
            if head_id is None and waiting:
                head_id = waiting[0].request_id
            queued, unissued, link_count, link_remaining, emitter = (), None, 0, None, ()
            if head_id is not None:
                request = context.requests[head_id]
                if request.io_started[head_layer]:
                    placement = portable_request(request.manifest).placement(head_layer)
                    queued = tuple(self.submitted[head_id, head_layer, s] - self.disk_done[head_id, head_layer, s]
                                   for s in range(context.num_ssu))
                    unissued = tuple(placement.count(s) - self.submitted[head_id, head_layer, s]
                                     for s in range(context.num_ssu))
                    link_count = sum(self.disk_done[head_id, head_layer, s] - self.link_done[head_id, head_layer, s]
                                     for s in range(context.num_ssu))
                    if lane.link_active_flow is not None:
                        link_remaining = max(0.0, lane.link_active_flow.link_end_time - now) / 1000.0 * context.npu_bw_gbps * 2**30
                    emitter = tuple(context.submission_states[state_id].disk_id
                                    for state_id in context.submission_queues.get(lane.npu_id, ()))
            states.append(NPUState(current, next_layer, end, queued, unissued, link_count,
                                   link_remaining, 0.0, max(0.0, context.client_next_issue_ms[lane.npu_id] - now),
                                   waiting, emitter))
        self.snapshot, self.time_ms = tuple(states), now

    def run(self, requests, num_npu, num_ssu, seed):
        submit, disk_done = native._register_submit, native._enqueue_link_io
        link_done, arrival = native._register_complete, native._handle_arrival

        def submitted(context, flow):
            self.submitted[flow.request_id, flow.layer, flow.disk_id] += 1
            return submit(context, flow)

        def disk_completed(context, flow, now):
            self.disk_done[flow.request_id, flow.layer, flow.disk_id] += 1
            return disk_done(context, flow, now)

        def link_completed(context, flow):
            self.link_done[flow.request_id, flow.layer, flow.disk_id] += 1
            return link_done(context, flow)

        def arrived(context, request_id, now):
            if request_id == self.candidate_id:
                self.capture(context, now)
            return arrival(context, request_id, now)

        with patch.object(native, "_register_submit", submitted), \
                patch.object(native, "_enqueue_link_io", disk_completed), \
                patch.object(native, "_register_complete", link_completed), \
                patch.object(native, "_handle_arrival", arrived):
            return run_native(requests, num_npu, num_ssu, seed)


def validate_strict_bounds(requests, num_npu, num_ssu, seed=42):
    """Observe public Path counts before release and compare every final ready.

    Uses no FIFO ownership order, active SSD state, native RNG position, or
    future events. Bounds are allowed to be loose; an underestimate of a
    deadline miss is not a bound failure, but a lower bound > truth is.
    """
    records = {}
    start, ready = native._start_layer_io, native._mark_layer_io_ready

    def started(context, request, layer, now, deadline, demand_window, **kwargs):
        if 0 <= layer < context.n_layers and not request.io_started[layer]:
            manifest = portable_request(request.manifest).placement(layer)
            counts = [manifest.count(s) for s in range(context.num_ssu)]
            queued = [disk.scheduler.report_path_pressure_analysis(now).counts[0]
                      for disk in context.disks]
            lane = context.npus[request.manifest.npu_id]
            result = screen_current_layer(counts, deadline - now, queued,
                                           link_pending_io=len(lane.link_pending) + int(lane.link_active_flow is not None))
            records[request.manifest.request_id, layer] = {
                "request_id": request.manifest.request_id, "layer": layer,
                "release_ms": now, "deadline_ms": deadline,
                "path_outstanding_io": queued, "kv_blocks_by_ssu": counts,
                "bound": result}
        return start(context, request, layer, now, deadline, demand_window, **kwargs)

    def marked_ready(context, request, layer, now):
        record = records[request.manifest.request_id, layer]
        record["actual_ready_after_release_ms"] = now - record["release_ms"]
        record["actual_deadline_miss_ms"] = max(0.0, now - record["deadline_ms"])
        return ready(context, request, layer, now)

    with patch.object(native, "_start_layer_io", started), \
            patch.object(native, "_mark_layer_io_ready", marked_ready):
        run_native(requests, num_npu, num_ssu, seed)
    rows = list(records.values())
    violations = [r for r in rows if r["bound"]["ready_lower_ms"] > r["actual_ready_after_release_ms"] + 1e-9]
    certified = [r for r in rows if r["bound"]["verdict"] == "must_stall"]
    return {"num_npu": num_npu, "num_ssu": num_ssu, "seed": seed,
            "checked_layers": len(rows), "bound_violation_count": len(violations),
            "certified_deadline_miss_count": len(certified),
            "actual_deadline_miss_count": sum(r["actual_deadline_miss_ms"] > 1e-9 for r in rows),
            "certified_false_positive_count": sum(r["actual_deadline_miss_ms"] <= 1e-9 for r in certified),
            "max_bound_slack_ms": max(r["actual_ready_after_release_ms"] - r["bound"]["ready_lower_ms"] for r in rows),
            "layers": rows}


def warm_case(num_ssu, seed=42, arrival_ms=0.04):
    num_npu = 32
    requests = tuple(make_request(i, i, 64 + 8 * (i % 5), 0.3 + (i % 7) * 0.07,
                                  num_ssu, placement_offset=i) for i in range(num_npu))
    candidate = make_request(99, 0, 192, 0.8, num_ssu, arrival_ms=arrival_ms)
    observer = CountSnapshotObserver(99)
    after = observer.run(requests + (candidate,), num_npu, num_ssu, seed)
    before = run_native(requests, num_npu, num_ssu, seed)
    started = time.perf_counter()
    paired = predict_request_impact(observer.snapshot, portable_request(candidate), 0,
                                    num_ssu=num_ssu, issue_order_seed=seed)
    paired_seconds = time.perf_counter() - started
    before_ends = {r["request_id"]: r["completion_time_ms"] for r in before["request_metrics"]}
    after_ends = {r["request_id"]: r["completion_time_ms"] for r in after["request_metrics"]}
    impacts = [{"request_id": rid, "actual_completion_delay_ms": after_ends[rid] - end,
                "predicted_completion_delay_ms": paired["existing_request_completion_delay_ms"][rid]}
               for rid, end in before_ends.items()]
    sensitivity = impact_order_sensitivity(observer.snapshot, portable_request(candidate), 0,
                                           num_ssu=num_ssu, issue_order_seed=seed)
    first_predicted = {}
    for row in paired["without_candidate"]["layers"]:
        first_predicted.setdefault(row["npu_id"], row)
    from validate_stall_predictor import truth_rows
    actual_before = truth_rows(before, arrival_ms)
    pending_bounds = []
    for npu, state in enumerate(observer.snapshot):
        if state.unissued_by_ssu is None:
            continue
        predicted = first_predicted[npu]
        actual = actual_before[predicted["request_id"], predicted["layer"]]
        bound = screen_pending_layer(state.queued_io_by_ssu, state.unissued_by_ssu,
                                      state.compute_end_in_ms, link_pending_io=state.link_pending_io,
                                      link_active_remaining_bytes=state.link_active_remaining_bytes)
        pending_bounds.append({"npu_id": npu, "request_id": predicted["request_id"],
                               "layer": predicted["layer"], "bound": bound,
                               "actual_ready_in_ms": actual["io_ready_ms"]})
    return {"name": "warm_count_only_candidate_arrival", "num_npu": num_npu,
            "num_ssu": num_ssu, "seed": seed, "snapshot_ms": arrival_ms,
            "paired_predictor_wall_seconds": paired_seconds,
            "snapshot": [asdict(s) for s in observer.snapshot],
            "telemetry": {"uses": ["per request/layer/SSU submitted and SSD/link completed counts",
                                     "NPU current compute end", "client emitter rotation and next issue",
                                     "NPU link active residual", "already-arrived manifests and admission order"],
                          "never_reads": ["SSU FIFO order", "active SSD owner or residual", "future event heap",
                                          "native RNG continuation"],
                          "warning": "Per-layer SSD completion feedback is additional telemetry, not only Path-total counts"},
            "with_candidate": comparison(paired["with_candidate"], after, arrival_ms),
            "without_candidate": comparison(paired["without_candidate"], before, arrival_ms),
            "existing_request_impact": impacts,
            "max_impact_error_ms": max(abs(r["predicted_completion_delay_ms"] - r["actual_completion_delay_ms"])
                                       for r in impacts),
            "order_sensitivity": sensitivity,
            "computing_npu_count_at_snapshot": sum(s.compute_end_in_ms > 0 for s in observer.snapshot),
            "total_unissued_io_at_snapshot": sum(sum(s.unissued_by_ssu or ()) for s in observer.snapshot),
            "total_link_io_at_snapshot": sum(s.link_pending_io for s in observer.snapshot),
            "pending_layer_bounds": pending_bounds,
            "pending_bound_violation_count": sum(r["bound"]["ready_lower_ms"] > r["actual_ready_in_ms"] + 1e-9
                                                 for r in pending_bounds),
            "range_is_guaranteed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "results/multi_ssu_stall_experiments/data/validation_multi_ssu_predictor.json"))
    args = parser.parse_args()
    core_files = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in core_files}
    from authenticated_workload_inputs import load_authenticated_bw_table
    with redirect_stdout(io.StringIO()):
        table, table_metadata = load_authenticated_bw_table(32)
    source_keys = ((32, 128), (64, 512), (96, 1024), (192, 512), (192, 1024), (192, 2048))
    cold, bounds = [], []
    for num_ssu in (6, 7):
        for seed in (7, 42):
            requests = tuple(make_request(i * 10 + j, i, 12 + (i * 7 + j * 19) % 85,
                                          0.08 + ((i + j) % 9) * 0.055,
                                          num_ssu, placement_offset=i)
                             for i in range(32) for j in range(2))
            cold.append(cold_case("32_lane_variable_two_requests", requests, 32, num_ssu, seed))
        hotspot = tuple(make_request(i, i, 12 + i * 3, 0.25 + (i % 3) * 0.12,
                                     num_ssu, placement_mode="hotspot") for i in range(32))
        cold.append(cold_case("32_lane_uneven_ssu_placement", hotspot, 32, num_ssu))
        bounds.append(validate_strict_bounds(hotspot, 32, num_ssu))
        raw_requests = []
        for lane in range(32):
            seq, nql = source_keys[lane % len(source_keys)]
            raw_requests.append(make_request(lane, lane, (seq * 1024 - nql) // 128,
                                              float(table[seq, nql][1]) / 1000.0,
                                              num_ssu, placement_offset=lane, nql=nql))
        raw_case = cold_case("32_lane_distinct_authenticated_data_requests", tuple(raw_requests), 32, num_ssu)
        raw_case["source_keys"] = [list(key) for key in source_keys]
        raw_case["authenticated_source"] = table_metadata
        cold.append(raw_case)
        compute_ms = float(table[192, 128][1]) / 1000.0
        single = make_request(0, 0, 1535, compute_ms, num_ssu)
        regression = cold_case("192K_nql128_single_active_receive_bottleneck", (single,), 32, num_ssu)
        counts = [sum(s == disk for s, _ in single.placement[0]) for disk in range(num_ssu)]
        regression["strict_lower_bound"] = screen_current_layer(counts, compute_ms, [0] * num_ssu)
        cold.append(regression)
    warm = [warm_case(s, arrival_ms=t) for s in (6, 7) for t in (0.002, 0.04, 2.0)]
    report = {"schema_version": 1, "io_bytes": IO_BYTES, "layers_per_request": 8,
              "predictor_sha256": hashlib.sha256(Path("multi_ssu_stall_predictor.py").read_bytes()).hexdigest(),
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "core_files_unchanged": all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items()),
              "core_sha256": hashes, "cold_cases": cold, "warm_cases": warm,
              "strict_bound_cases": bounds,
              "conclusion_scope": "cold algorithm agreement is not a real-SSD timing or unknown-order guarantee"}
    if not report["core_files_unchanged"]:
        raise AssertionError("Core simulator source changed during validation")
    if any(c["bound_violation_count"] for c in bounds):
        raise AssertionError("A strict lower bound exceeded observed completion time")
    if any(c["pending_bound_violation_count"] for c in warm):
        raise AssertionError("A pending-layer lower bound exceeded observed completion time")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "cold": [{"name": c["name"], "ssu": c["num_ssu"], "seed": c["seed"],
                                "errors": c["comparison"]["max_absolute_error_ms"]} for c in cold],
                      "warm": [{"ssu": c["num_ssu"], "snapshot_ms": c["snapshot_ms"],
                                 "errors": c["with_candidate"]["max_absolute_error_ms"],
                                 "impact_error": c["max_impact_error_ms"]} for c in warm],
                      "strict_bounds": [{k: v for k, v in c.items() if k != "layers"} for c in bounds]}, indent=2))


if __name__ == "__main__":
    main()
