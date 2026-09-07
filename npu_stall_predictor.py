"""4 NPU / 1 SSD stall arithmetic, without importing the project simulator.

One I/O is one complete GLM KV block: 128 tokens, 176 KiB. Times are ms
relative to the snapshot; bandwidth is GiB/s. No KV payload is stored in DRAM.

screen_current_layer and fifo_burst_certificate provide mathematical bounds
under their documented FIFO assumptions. forecast_path0 is a CONDITIONAL
closed-loop prediction: reconstructed initial queue order is NOT an observed
SSD fact. It must not be used as a proof of no stall in a real device.

Only equal blocks, one non-preemptive work-conserving SSD, independent NPU
receive links at least as fast as the SSD, and one-layer lookahead are modeled.
No unknown future arrivals, finite PIR, hardware overhead or compute jitter.
Callers supply valid physical inputs; validation is limited to model scope.
"""

from dataclasses import dataclass, replace
import heapq
import itertools
import json


IO_BYTES = 176 * 1024
EPS_MS = 1e-10


def io_service_ms(bandwidth_gib_s=40.0):
    return IO_BYTES / (bandwidth_gib_s * 2**30) * 1000


def screen_current_layer(
    kv_blocks, deadline_ms, path_outstanding_io, *, bandwidth_gib_s=40.0,
    link_gib_s=50.0, active_remaining_bytes=None, atomic_enqueue=False,
):
    """Screen an entirely UNSUBMITTED layer appended to the only active Path.

    path_outstanding_io INCLUDES the one possible active SSD I/O, excludes
    the target and excludes I/Os already on the receive link. This differs
    from glm_path0_deadline.queued_io_count, which EXCLUDES active I/O.
    active_remaining_bytes=None means unknown, bounded by [0, IO_BYTES].
    atomic_enqueue=False permits competitors to interleave during submission;
    then only a lower completion bound is available. No issue delay is added
    to that optimistic bound. Do not use on a partially submitted target.
    """
    if link_gib_s < bandwidth_gib_s:
        raise ValueError("This bound requires an independent link >= SSD bandwidth")
    service, tail = io_service_ms(bandwidth_gib_s), io_service_ms(link_gib_s)
    if not kv_blocks:
        lower = upper = 0.0
    else:
        q = path_outstanding_io
        if q and active_remaining_bytes is None:
            ahead_low, ahead_high = (q - 1) * service, q * service
        else:
            active = (active_remaining_bytes or 0) / IO_BYTES * service
            ahead_low = ahead_high = (max(q - 1, 0) * service + active) if q else 0.0
        lower = ahead_low + kv_blocks * service + tail
        upper = ahead_high + kv_blocks * service + tail if atomic_enqueue else None
    verdict = (
        "must_stall" if lower > deadline_ms + EPS_MS
        else "meets_deadline" if upper is not None and upper <= deadline_ms + EPS_MS
        else "unknown"
    )
    return {
        "verdict": verdict, "ready_lower_ms": lower, "ready_upper_ms": upper,
        "stall_lower_ms": max(0.0, lower - deadline_ms),
        "stall_upper_ms": None if upper is None else max(0.0, upper - deadline_ms),
        "atomic_enqueue": atomic_enqueue, "io_bytes": IO_BYTES,
        "scope": "FIFO current-layer bound; target has not submitted any I/O",
    }


def fifo_burst_certificate(
    burst_kv_blocks, burst_submit_ms, victim_kv_blocks, victim_compute_ms,
    *, bandwidth_gib_s=40.0, link_gib_s=50.0,
):
    """Sufficient, NOT necessary, condition for phase-independent FIFO stall.

    At the end of a fully submitted burst its remaining work is at least H.
    A computing victim with constant C and enough future layers must miss at
    least one of its next two transitions if H + own_service + tail > 2*C.
    The excess bounds their CUMULATIVE stall, not necessarily one layer's
    stall: an earlier miss may postpone the next release and its deadline.
    No bypass of the committed burst is allowed. This is not an infinite-
    period proof for a finite eight-layer job.
    """
    service = io_service_ms(bandwidth_gib_s)
    h = max(0.0, burst_kv_blocks * service - burst_submit_ms)
    excess = (h + victim_kv_blocks * service + io_service_ms(link_gib_s) - 2 * victim_compute_ms
              if victim_kv_blocks else 0.0)
    return {
        "committed_work_lower_ms": h,
        "phase_independent_stall_certified": excess > EPS_MS,
        "certified_excess_lower_ms": max(0.0, excess),
        "excess_scope": "cumulative stall across the next two compute transitions",
        "scope": "Requires currently computing victim, two future transitions, constant C and strict FIFO",
    }


def required_rate_gib_s(kv_blocks, budget_ms, *, ahead_io=0, latency_ms=0.0):
    """Fluid service budget for a layer-owned Path, NOT a CIR latency guarantee.

    With a VALID service curve r*(t-latency)^+, this is the sufficient rate.
    Without such a measured/proven curve, it is only a planning estimate;
    the project's virtual-finish CIR scheduler needs separate verification.
    """
    work = (ahead_io + kv_blocks) * IO_BYTES
    if not work:
        return 0.0
    usable = budget_ms - latency_ms
    return work / 2**30 / (usable / 1000) if usable > 0 else float("inf")


@dataclass(frozen=True)
class Request:
    request_id: int
    kv_blocks: int
    compute_ms: float
    layers: int = 8


@dataclass(frozen=True)
class NPUState:
    """Client-visible lane state at time zero, no FIFO ordering field.

    request is the current/admitted request; next_layer is its next layer
    whose computation has NOT started. next_layer==layers permits a current
    final-layer computation followed by queued requests. waiting_requests
    have already arrived, in admission order; no future manifests are read.

    compute_end_in_ms>0: current computation will finish at that relative time.
    <=0: waiting/idle, with the preceding compute end retained (can be past).
    The following fields describe the FIRST not-yet-computed layer, including
    queued request Layer0 if the active request is computing its last layer:
      queued_io: SSD outstanding count, including active if it belongs here;
      unissued_io=None: prefetch not released; otherwise unsent command count;
      ready_in_ms: residual HBM arrival time if queued_io==unissued_io==0.
    next_issue_in_ms is the observed client emitter's next available time.
    When all I/O is ready, explicitly use unissued_io=0, not None.
    """
    request: Request | None = None
    next_layer: int = 0
    compute_end_in_ms: float = 0.0
    queued_io: int = 0
    unissued_io: int | None = None
    ready_in_ms: float = 0.0
    next_issue_in_ms: float = 0.0
    waiting_requests: tuple[Request, ...] = ()


def forecast_path0(
    npus, *, bandwidth_gib_s=40.0, link_gib_s=50.0,
    issue_interval_us=0.1, cross_request_prefetch=True,
    queue_order=(0, 1, 2, 3), queue_layout="round_robin",
    tie_order=(0, 1, 2, 3), active_remaining_bytes=None,
    active_npu_id=None,
):
    """Finite max-plus rollout of all known remaining layers/requests.

    Unknown initial FIFO order is reconstructed from per-lane counts using
    round_robin or grouped ownership. queue_order/tie_order are SCENARIO
    choices, never required SSD telemetry. Unknown active remainder uses one
    full block for this scenario; this does NOT upper-bound the whole rollout.
    Unlike the project's randomized same-time emitter, ties here use the
    declared deterministic NPU order. Exact simulator agreement is not claimed.
    Inputs are immutable and times in all results are snapshot-relative.
    """
    if len(npus) != 4 or link_gib_s < bandwidth_gib_s:
        raise ValueError("Model scope: four NPUs, one SSD, link bandwidth >= SSD")
    if queue_layout not in ("round_robin", "grouped"):
        raise ValueError("queue_layout must be round_robin or grouped")
    service, tail = io_service_ms(bandwidth_gib_s), io_service_ms(link_gib_s)
    jobs, positions, running, compute_end = [], [0]*4, [], []
    next_issue = [max(0.0, n.next_issue_in_ms) for n in npus]
    rows, events, request_ends = [], [], {}
    sequence, issue_generation, pending_issue, fifo_tail = 0, 0, None, 0.0

    def push(time, kind, npu, payload=None):
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (time, kind, npu, sequence, payload))

    for i, state in enumerate(npus):
        lane = []
        manifests = ([] if state.request is None else [(state.request, state.next_layer)])
        manifests += [(r, 0) for r in state.waiting_requests]
        for request, first in manifests:
            for layer in range(first, request.layers):
                lane.append({"request_id": request.request_id, "layer": layer,
                             "kv_blocks": request.kv_blocks, "compute_ms": request.compute_ms,
                             "last_layer": layer == request.layers - 1,
                             "released": False, "unsent": request.kv_blocks,
                             "ready_ms": None, "deadline_ms": None})
        if lane and state.unissued_io is not None:
            lane[0].update(released=True, unsent=state.unissued_io,
                           deadline_ms=state.compute_end_in_ms)
        jobs.append(lane)
        running.append(state.compute_end_in_ms > 0)
        compute_end.append(state.compute_end_in_ms)
        if running[i]:
            finished_id = (state.request.request_id if state.request is not None
                           and state.next_layer == state.request.layers else None)
            push(state.compute_end_in_ms, 0, i, finished_id)

    def head(i):
        return jobs[i][positions[i]] if positions[i] < len(jobs[i]) else None

    def schedule_issue(now):
        nonlocal issue_generation, pending_issue
        ready = [max(now, next_issue[i]) for i in range(4)
                 if head(i) is not None and head(i)["released"] and head(i)["unsent"] > 0]
        if not ready:
            return
        when = min(ready)
        if pending_issue is not None and pending_issue <= when:
            return
        issue_generation += 1
        pending_issue = when
        push(when, 3, -1, issue_generation)

    def release(i, now, deadline):
        job = head(i)
        job.update(released=True, deadline_ms=deadline)
        if not job["unsent"]:
            job["ready_ms"] = now
            push(now, 2, i, job)
        schedule_issue(now)

    def start(i, now):
        job = head(i)
        if running[i] or job is None:
            return
        if not job["released"]:
            # Initial/new request admission. A late-arriving queued request is
            # not retroactively prefetched at the previous compute start.
            release(i, now, now)
        if job["unsent"] or job["ready_ms"] is None or job["ready_ms"] > now + EPS_MS:
            return
        deadline = job["deadline_ms"]
        end = now + job["compute_ms"]
        rows.append({"npu_id": i, "request_id": job["request_id"], "layer": job["layer"],
                     "deadline_ms": deadline, "io_ready_ms": job["ready_ms"],
                     "compute_start_ms": now, "compute_end_ms": end,
                     "stall_ms": max(0.0, now - deadline),
                     "future_stall_ms": max(0.0, now - max(0.0, deadline)),
                     "is_layer0": job["layer"] == 0})
        positions[i] += 1
        running[i], compute_end[i] = True, end
        push(end, 0, i, job["request_id"] if job["last_layer"] else None)
        nxt = head(i)
        if nxt is not None and (cross_request_prefetch or nxt["request_id"] == job["request_id"]):
            release(i, now, end)

    # Reconstruct ONLY a possible initial queue, explicitly not hidden truth.
    counts = [n.queued_io for n in npus]
    owners = []
    if active_npu_id is not None and counts[active_npu_id]:
        owners.append(active_npu_id)
        counts[active_npu_id] -= 1
    if queue_layout == "grouped":
        owners.extend(i for i in queue_order for _ in range(counts[i]))
    else:
        while any(counts):
            for i in queue_order:
                if counts[i]:
                    owners.append(i)
                    counts[i] -= 1
    for number, i in enumerate(owners):
        duration = service
        if number == 0 and active_remaining_bytes is not None:
            duration = active_remaining_bytes / IO_BYTES * service
        fifo_tail += duration
        head(i)["ready_ms"] = fifo_tail + tail
    for i, state in enumerate(npus):
        job = head(i)
        if job is not None and job["released"] and job["unsent"] == 0:
            job["ready_ms"] = max(job["ready_ms"] or 0.0, state.ready_in_ms)
            push(job["ready_ms"], 2, i, job)
        push(0.0, 2.75, i)
    schedule_issue(0.0)

    while events:
        now, kind, i, _, payload = heapq.heappop(events)
        if kind == 0:
            running[i] = False
            if payload is not None:
                request_ends[payload] = now
            push(now, 2.75, i)
        elif kind == 2:
            push(now, 2.75, i)
        elif kind == 2.75:
            start(i, now)
        else:
            if payload != issue_generation:
                continue
            pending_issue = None
            for i in tie_order:
                job = head(i)
                if job is None or not job["released"] or not job["unsent"] or next_issue[i] > now + EPS_MS:
                    continue
                fifo_tail = max(now, fifo_tail) + service
                job["ready_ms"] = fifo_tail + tail
                job["unsent"] -= 1
                next_issue[i] = now + issue_interval_us / 1000
                if not job["unsent"]:
                    push(job["ready_ms"], 2, i, job)
            schedule_issue(now)
    return {
        "kind": "conditional_prediction", "guaranteed": False,
        "assumptions": {"unknown_future_arrivals": "none", "queue_layout": queue_layout,
                        "queue_order": list(queue_order), "tie_order": list(tie_order),
                        "unknown_active_remainder": "one full I/O", "issue_interval_us": issue_interval_us,
                        "cross_request_prefetch": cross_request_prefetch},
        "layers": rows, "request_completion_ms": request_ends,
        "future_stall_ms_by_npu": [sum(r["future_stall_ms"] for r in rows if r["npu_id"] == i) for i in range(4)],
        "non_layer0_stall_ms_by_npu": [sum(r["future_stall_ms"] for r in rows if r["npu_id"] == i and not r["is_layer0"]) for i in range(4)],
    }


def predict_request_impact(npus, new_request, target_npu, **environment):
    """Paired forecasts from the SAME snapshot; insertion cannot erase old work."""
    before = forecast_path0(npus, **environment)
    changed = list(npus)
    target = changed[target_npu]
    has_old_head = (target.request is not None and target.next_layer < target.request.layers
                    or bool(target.waiting_requests))
    if not has_old_head:
        # A newly created head has not issued any reads, even if an empty
        # lane reported zero unissued commands for its previous work.
        target = replace(target, queued_io=0, unissued_io=None, ready_in_ms=0.0)
    if target.request is None and not target.waiting_requests:
        changed[target_npu] = replace(target, request=new_request)
    else:
        changed[target_npu] = replace(target, waiting_requests=target.waiting_requests + (new_request,))
    after = forecast_path0(changed, **environment)
    old_rows = {(r["request_id"], r["layer"]): r for r in before["layers"]}
    deltas = []
    for row in after["layers"]:
        key = (row["request_id"], row["layer"])
        if key in old_rows:
            deltas.append({"request_id": key[0], "layer": key[1], "npu_id": row["npu_id"],
                           "compute_start_delay_ms": row["compute_start_ms"] - old_rows[key]["compute_start_ms"],
                           "extra_future_stall_ms": row["future_stall_ms"] - old_rows[key]["future_stall_ms"]})
    return {
        "kind": "conditional_prediction", "guaranteed": False,
        "candidate_request_id": new_request.request_id, "target_npu": target_npu,
        "candidate_layers": [r for r in after["layers"] if r["request_id"] == new_request.request_id],
        "existing_layer_deltas": deltas,
        "existing_request_completion_delay_ms": {
            rid: after["request_completion_ms"][rid] - end
            for rid, end in before["request_completion_ms"].items()},
        "without_candidate": before, "with_candidate": after,
    }


def impact_order_sensitivity(npus, new_request, target_npu, **environment):
    """24 paired order scenarios; their range is NOT a rigorous error bound."""
    samples = []
    for order in itertools.permutations(range(4)):
        report = predict_request_impact(npus, new_request, target_npu,
                                        queue_order=order, tie_order=order, **environment)
        samples.append({
            "order": list(order),
            "candidate_non_layer0_stall_ms": sum(r["future_stall_ms"] for r in report["candidate_layers"] if not r["is_layer0"]),
            "largest_existing_completion_delay_ms": max([0.0, *report["existing_request_completion_delay_ms"].values()]),
        })
    return {"kind": "scenario_sensitivity", "range_is_guaranteed": False, "samples": samples}


if __name__ == "__main__":
    states = [NPUState(), NPUState(Request(10, 64, 0.5), next_layer=1,
                                  compute_end_in_ms=0.02, unissued_io=0),
              NPUState(), NPUState()]
    impact = predict_request_impact(states, Request(99, 128, 0.8), 0)
    print(json.dumps({
        "current_layer": screen_current_layer(10, 0.5, 100),
        "burst_certificate": fifo_burst_certificate(1534, 0.1533, 7, 0.582149491632),
        "candidate_layers": impact["candidate_layers"],
        "existing_request_completion_delay_ms": impact["existing_request_completion_delay_ms"],
        "prediction_kind": impact["kind"],
    }, ensure_ascii=False, indent=2))
