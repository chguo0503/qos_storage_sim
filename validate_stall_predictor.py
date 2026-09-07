#!/usr/bin/env python3
"""Compare the conditional predictor with the UNMODIFIED project simulator.

The warm-arrival observer exposes counts and observed completions, not the
FIFO order, active command identity/remainder, or future simulator events.
Per-request/layer SSD-completion counters are additional telemetry: this
validation does not claim that the existing Path-total counter supplies them.
Temporary Python observation wrappers call the original handlers unchanged.
"""

from collections import defaultdict
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import continuous_batch_sim as native
from npu_stall_predictor import (
    NPUState, Request, forecast_path0, io_service_ms, predict_request_impact,
)
from run_stall_policy_experiments import Profile, build_requests, run_case


def portable_request(request):
    return Request(request.request_id, len(request.placement[0]),
                   request.load["per_layer_us"] / 1000, layers=8)


def truth_rows(summary, snapshot_ms=0.0):
    result = {}
    for batch in summary["microbatch_metrics"]:
        deadline = batch["admission_time_ms"]
        for metric in batch["layer_metrics"]:
            start = metric["compute_start_ms"]
            row = {
                "npu_id": batch["npu_id"], "request_id": batch["member_request_ids"][0],
                "layer": metric["layer"], "deadline_ms": deadline - snapshot_ms,
                "io_ready_ms": max(0.0, metric["io_ready_time_ms"] - snapshot_ms),
                "compute_start_ms": start - snapshot_ms,
                "compute_end_ms": metric["compute_end_ms"] - snapshot_ms,
                "future_stall_ms": max(0.0, start - max(snapshot_ms, deadline)),
                "is_layer0": metric["layer"] == 0,
            }
            result[row["request_id"], row["layer"]] = row
            deadline = metric["compute_end_ms"]
    return result


def comparison(prediction, summary, snapshot_ms=0.0):
    actual = truth_rows(summary, snapshot_ms)
    records = []
    for row in prediction["layers"]:
        truth = actual[row["request_id"], row["layer"]]
        errors = {field: row[field] - truth[field] for field in
                  ("io_ready_ms", "compute_start_ms", "future_stall_ms")}
        records.append({"request_id": row["request_id"], "layer": row["layer"],
                        "npu_id": row["npu_id"], "predicted": row,
                        "actual": truth, "prediction_minus_actual": errors})
    fields = ("io_ready_ms", "compute_start_ms", "future_stall_ms")
    warm = [r for r in records if r["layer"] > 0]
    return {
        "comparison_layer_count": len(records),
        "ready_times_before_snapshot_clamped_to_zero": True,
        "max_absolute_error_ms": {
            field: max((abs(r["prediction_minus_actual"][field]) for r in records), default=0.0)
            for field in fields},
        "mean_absolute_error_ms": {
            field: sum(abs(r["prediction_minus_actual"][field]) for r in records) / len(records)
            if records else 0.0 for field in fields},
        "non_layer0_stall_false_negative_count": sum(
            r["actual"]["future_stall_ms"] > 1e-8 and r["predicted"]["future_stall_ms"] <= 1e-8
            for r in warm),
        "non_layer0_stall_false_positive_count": sum(
            r["actual"]["future_stall_ms"] <= 1e-8 and r["predicted"]["future_stall_ms"] > 1e-8
            for r in warm),
        "layers": records,
    }


class CountSnapshotObserver:
    """An instrumentation-only adapter, with no retained submission ordering."""

    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.submitted = defaultdict(int)
        self.ssd_completed = defaultdict(int)
        self.last_ssd_completion_ms = {}
        self.snapshot = None
        self.snapshot_time_ms = None
        self.path_total = None

    def capture(self, context, now):
        states = []
        for lane in context.npus:
            waiting = tuple(portable_request(context.requests[rid].manifest)
                            for rid in lane.admission_queue)
            batch = lane.active_batch
            current, next_layer, compute_end = None, 0, 0.0
            head_id, head_layer = None, 0
            if batch is not None:
                current = portable_request(context.requests[batch.member_request_ids[0]].manifest)
                if lane.compute_active is not None:
                    active_layer = lane.compute_active[1]
                    next_layer = active_layer + 1
                    compute_end = batch.layer_metrics[active_layer].compute_end_ms - now
                else:
                    next_layer = batch.compute_done_up_to + 1
                    compute_end = batch.previous_compute_end_ms - now
                if next_layer < current.layers:
                    head_id, head_layer = current.request_id, next_layer
            if head_id is None and waiting:
                head_id = waiting[0].request_id
            queued, unsent, ready_in = 0, None, 0.0
            if head_id is not None:
                request = context.requests[head_id]
                if request.io_started[head_layer]:
                    key = head_id, head_layer
                    block_count = len(request.manifest.placement[0])
                    queued = self.submitted[key] - self.ssd_completed[key]
                    unsent = block_count - self.submitted[key]
                    if not queued and not unsent:
                        ready_in = max(0.0, self.last_ssd_completion_ms.get(key, now)
                                       + io_service_ms(context.npu_bw_gbps) - now)
                        if request.io_ready[head_layer]:
                            ready_in = 0.0
            states.append(NPUState(
                current, next_layer, compute_end, queued, unsent, ready_in,
                max(0.0, context.client_next_issue_ms[lane.npu_id] - now), waiting))
        pressure = context.disks[0].scheduler.report_path_pressure_analysis(now)
        self.path_total = int(pressure.counts[0])
        if sum(s.queued_io for s in states) != self.path_total:
            raise AssertionError("count-only snapshot lost some outstanding SSD work")
        self.snapshot, self.snapshot_time_ms = tuple(states), now

    def run(self, profiles, requests, seed):
        submit_original = native._register_submit
        complete_original = native._enqueue_link_io
        arrival_original = native._handle_arrival

        def submitted(context, flow):
            self.submitted[flow.request_id, flow.layer] += flow.block_count
            return submit_original(context, flow)

        def completed(context, flow, now):
            key = flow.request_id, flow.layer
            self.ssd_completed[key] += flow.block_count
            self.last_ssd_completion_ms[key] = now
            return complete_original(context, flow, now)

        def arrived(context, request_id, now):
            if request_id == self.candidate_id:
                self.capture(context, now)
            return arrival_original(context, request_id, now)

        with patch.object(native, "_register_submit", submitted), \
                patch.object(native, "_enqueue_link_io", completed), \
                patch.object(native, "_handle_arrival", arrived):
            return run_case(profiles, "baseline", seed=seed, requests=requests, complete_all=True)


def cold_case(name, profiles, seed, only_npu0=False):
    requests = build_requests(profiles, count_per_npu=1)
    if only_npu0:
        requests = tuple(r for r in requests if r.npu_id == 0)
    states = [NPUState() for _ in range(4)]
    for request in requests:
        states[request.npu_id] = NPUState(portable_request(request))
    predicted = forecast_path0(states)
    actual = run_case(profiles, "baseline", seed=seed, requests=requests, complete_all=True)
    return {"name": name, "seed": seed,
            "profiles": [asdict(p) for p in profiles],
            "input_fingerprint": actual["input_fingerprint"],
            "conditional_assumptions": predicted["assumptions"],
            "comparison": comparison(predicted, actual["summary"])}


def arrival_case(profiles, seed, *, arrival_ms=1.25):
    existing = build_requests(profiles, count_per_npu=1)
    # Replacement lane demand remains below capacity: 39.39 GiB/s total,
    # rather than explaining misses by introducing a sustained overload.
    candidate_profile = Profile(256, 6)
    prototype = build_requests((candidate_profile,) * 4, count_per_npu=1)[0]
    load = dict(prototype.load, request_id=99, arrival_ms=arrival_ms,
                arrival_time=arrival_ms, initial=False)
    candidate = replace(prototype, request_id=99, arrival_time_ms=arrival_ms, load=load)
    observer = CountSnapshotObserver(candidate.request_id)
    after = observer.run(profiles, existing + (candidate,), seed)
    before = run_case(profiles, "baseline", seed=seed, requests=existing, complete_all=True)
    snapshot = observer.snapshot
    paired = predict_request_impact(snapshot, portable_request(candidate), candidate.npu_id)
    before_end = {r["request_id"]: r["completion_time_ms"]
                  for r in before["summary"]["request_metrics"]}
    after_end = {r["request_id"]: r["completion_time_ms"]
                 for r in after["summary"]["request_metrics"]}
    delay_rows = []
    for rid, end in before_end.items():
        actual_delta = after_end[rid] - end
        forecast_delta = paired["existing_request_completion_delay_ms"][rid]
        delay_rows.append({"request_id": rid, "actual_completion_delay_ms": actual_delta,
                           "predicted_completion_delay_ms": forecast_delta,
                           "prediction_minus_actual_ms": forecast_delta - actual_delta})
    raw_before = truth_rows(before["summary"])
    raw_after = truth_rows(after["summary"])
    prefix_equal = all(
        abs(row[field] - raw_after[key][field]) < 1e-9
        for key, row in raw_before.items() for field in ("compute_start_ms", "io_ready_ms")
        if row[field] < arrival_ms)
    scenario_rows = []
    for layout in ("round_robin", "grouped"):
        for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
            scenario = predict_request_impact(snapshot, portable_request(candidate), candidate.npu_id,
                                              queue_layout=layout, queue_order=order, tie_order=order)
            scenario_rows.append({"layout": layout, "order": list(order),
                                  "existing_completion_delay_ms": scenario["existing_request_completion_delay_ms"],
                                  "candidate_completion_ms": scenario["with_candidate"]["request_completion_ms"][99]})
    return {
        "name": "native_arrival_count_only_snapshot", "seed": seed,
        "snapshot_ms": arrival_ms, "path0_outstanding_io": observer.path_total,
        "snapshot": [asdict(state) for state in snapshot],
        "candidate": asdict(portable_request(candidate)),
        "candidate_target_npu": candidate.npu_id,
        "nominal_demand_after_lane_replacement_gib_s": (
            sum(p.demand_gib_s for p in profiles[1:]) + candidate_profile.demand_gib_s),
        "pre_arrival_completed_event_times_identical": prefix_equal,
        "telemetry": {
            "requires": ["NPU current compute/end and admission queue", "arrived request manifest",
                         "per-request/layer submitted count", "per-request/layer SSD-completed count",
                         "last observed SSD completion timestamp", "client next issue time", "Path0 total count"],
            "never_read": ["FIFO sequence", "active SSD command identity", "active remaining service", "future event heap"],
            "deployment_warning": "Per-layer SSD completion counters/timestamps need SSU feedback or equivalent instrumentation; Path0 total alone is insufficient.",
        },
        "without_candidate": comparison(paired["without_candidate"], before["summary"], arrival_ms),
        "with_candidate": comparison(paired["with_candidate"], after["summary"], arrival_ms),
        "existing_request_impact": delay_rows,
        "actual_candidate_completion_relative_ms": after_end[99] - arrival_ms,
        "predicted_candidate_completion_relative_ms": paired["with_candidate"]["request_completion_ms"][99],
        "alternative_scenarios_not_bounds": scenario_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "results/stall_prediction_experiments/data/validation_stall_predictor_v1.json"))
    args = parser.parse_args()
    core_paths = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")
    hashes_before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in core_paths}
    profiles = (Profile(8, 0.2), Profile(128, 1), Profile(64, 2), Profile(96, 3))
    cases = [cold_case("single_npu_no_submission_tie", profiles, 42, only_npu0=True)]
    cases += [cold_case("four_npu_original_random_ties", profiles, seed) for seed in (0, 7, 42)]
    arrivals = [arrival_case(profiles, seed) for seed in (7, 42)]
    if any(error > 1e-9 for error in cases[0]["comparison"]["max_absolute_error_ms"].values()):
        raise AssertionError("Single-lane exact arithmetic no longer agrees with native simulator")
    if not all(case["pre_arrival_completed_event_times_identical"] for case in arrivals):
        raise AssertionError("Candidate changed native execution before its arrival")
    hashes_after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in core_paths}
    if hashes_before != hashes_after:
        raise AssertionError("Core simulator source changed during validation")
    report = {"schema_version": 1, "method": "conditional predictor versus original native simulator",
              "core_files_unchanged": True, "core_sha256": hashes_after,
              "predictor_sha256": hashlib.sha256(Path("npu_stall_predictor.py").read_bytes()).hexdigest(),
              "nominal_four_npu_demand_gib_s": sum(p.demand_gib_s for p in profiles),
              "io_bytes": 176 * 1024, "npu_count": 4, "ssu_count": 1,
              "ssd_gib_s": 40, "npu_link_gib_s": 50,
              "issue_interval_us": 0.1, "layers_per_request": 8,
              "cross_request_prefetch": True,
              "no_claim_of_exact_order_or_no_stall_guarantee": True,
              "cold_start_cases": cases, "arrival_cases": arrivals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output),
                      "cold_start_max_errors_ms": [c["comparison"]["max_absolute_error_ms"] for c in cases],
                      "arrival_max_errors_ms": [c["with_candidate"]["max_absolute_error_ms"] for c in arrivals],
                      "arrival_impact": [c["existing_request_impact"] for c in arrivals]}, indent=2))


if __name__ == "__main__":
    main()
