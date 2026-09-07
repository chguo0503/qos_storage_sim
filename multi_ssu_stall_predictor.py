"""Independent, stdlib-only baseline predictor for N NPUs and M SSUs.

One I/O is 176 KiB. Each SSU has one work-conserving FIFO Path0; each NPU
has ONE FCFS receive link shared by all SSUs. Times are snapshot-relative ms.
The strict current-layer lower bound does not need FIFO ownership order.
The closed-loop forecast DOES depend on an assumed initial order: its output
is a conditional prediction, never a hardware deadline guarantee.

Only arrived requests are inputs. One-layer and already-arrived cross-request
prefetch are supported, with one request computing on each NPU. No batching,
DRAM payload, finite PIR, future unknown arrivals, or hardware timing noise.
"""

from collections import deque
from dataclasses import dataclass, replace
import heapq
import json

IO_BYTES = 176 * 1024
EPS_MS = 1e-12


def io_service_ms(bandwidth_gib_s):
    return IO_BYTES / (bandwidth_gib_s * 2**30) * 1000.0


def screen_current_layer(kv_blocks_by_ssu, deadline_ms, path_outstanding_io,
                         *, disk_gib_s=40.0, link_gib_s=50.0,
                         disk_active_remaining_bytes=None, link_pending_io=0,
                         link_active_remaining_bytes=None):
    """Strict optimistic bound for a wholly UNSUBMITTED layer on FIFO Path0.

    Each Path count includes its active I/O and excludes the target. Link
    count also includes its active I/O. An unknown active residual is allowed
    to be zero when constructing a LOWER bound. Future competitors can only
    make this bound more optimistic. No finite upper bound is claimed.
    """
    service, tail = io_service_ms(disk_gib_s), io_service_ms(link_gib_s)
    residuals = (disk_active_remaining_bytes if disk_active_remaining_bytes is not None
                 else (None,) * len(kv_blocks_by_ssu))
    if not len(kv_blocks_by_ssu) == len(path_outstanding_io) == len(residuals):
        raise ValueError("Supply one target count, Path count and optional residual per SSU")
    first, last = [], []
    for blocks, queued, residual in zip(kv_blocks_by_ssu, path_outstanding_io, residuals):
        if not blocks:
            continue
        ahead = max(0, queued - 1) * service
        if queued and residual is not None:
            ahead += residual / IO_BYTES * service
        first.append(ahead + service)
        last.append(ahead + blocks * service)
    if not last:
        lower = disk_lower = link_lower = 0.0
    else:
        link_ahead = max(0, link_pending_io - 1) * tail
        if link_pending_io and link_active_remaining_bytes is not None:
            link_ahead += link_active_remaining_bytes / IO_BYTES * tail
        disk_lower = max(last) + tail
        link_lower = max(link_ahead, min(first)) + sum(kv_blocks_by_ssu) * tail
        lower = max(disk_lower, link_lower)
    return {"verdict": "must_stall" if lower > deadline_ms + EPS_MS else "unknown",
            "ready_lower_ms": lower, "ready_upper_ms": None,
            "disk_barrier_lower_ms": disk_lower, "receive_link_lower_ms": link_lower,
            "stall_lower_ms": max(0.0, lower - deadline_ms), "io_bytes": IO_BYTES,
            "scope": "Strict lower bound; entirely unsubmitted target on each SSU FIFO Path0"}


def fifo_burst_certificate(burst_kv_blocks_by_ssu, burst_submit_ms,
                           victim_kv_blocks_by_ssu, victim_compute_ms,
                           *, disk_gib_s=40.0, link_gib_s=50.0):
    """A per-SSU sufficient certificate, not a necessary condition.

    At burst submission end H_s >= max(0, K_burst,s*b/B_s - J). If an SSU
    used by the victim has H_s + K_victim,s*b/B_s + b/L_i > 2*C_i, at least
    one of its next two compute transitions must stall under strict FIFO.
    Excess is a lower bound on their CUMULATIVE stall, not one layer's stall.
    Requires the victim currently computing, two future transitions and the
    same compute/read profile on those transitions. Finite eight-layer jobs
    do not by themselves imply an infinite recurrent-stall theorem.
    """
    service, tail = io_service_ms(disk_gib_s), io_service_ms(link_gib_s)
    committed = [max(0.0, blocks * service - burst_submit_ms)
                 for blocks in burst_kv_blocks_by_ssu]
    excess = [max(0.0, h + blocks * service + tail - 2 * victim_compute_ms)
              if blocks else 0.0 for h, blocks in zip(committed, victim_kv_blocks_by_ssu)]
    return {"committed_work_lower_ms_by_ssu": committed,
            "phase_independent_stall_certified": max(excess, default=0.0) > EPS_MS,
            "certified_cumulative_stall_lower_ms": max(excess, default=0.0),
            "excess_lower_ms_by_ssu": excess,
            "scope": "cumulative stall across next two transitions; strict FIFO and constant victim profile"}


def screen_pending_layer(queued_io_by_ssu, unissued_by_ssu, deadline_ms, *,
                         disk_gib_s=40.0, link_gib_s=50.0, link_pending_io=0,
                         link_active_remaining_bytes=None):
    """Strict optimistic bound for a PARTLY submitted current layer.

    Counts here belong ONLY to this layer, unlike screen_current_layer's
    ahead-of-target Path totals. Unknown competitors' position is ignored.
    If this layer has queued work on an SSU, optimistically its first command
    is active and almost finished. This makes the result a lower bound even
    without knowing the active owner/residual. Its remaining commands still
    consume SSD and the NPU's shared receive-link service.

    link_pending_io and an optional active residual must also belong to THIS
    layer. With no remaining I/O the layer is already ready, but its historical
    ready timestamp cannot be reconstructed; a past deadline must not be
    called an I/O miss merely because snapshot-relative zero exceeds it.
    """
    if len(queued_io_by_ssu) != len(unissued_by_ssu):
        raise ValueError("Supply queued and unissued target counts for every SSU")
    service, tail = io_service_ms(disk_gib_s), io_service_ms(link_gib_s)
    if not sum(queued_io_by_ssu) + sum(unissued_by_ssu) + link_pending_io:
        return {"verdict": "already_ready", "ready_lower_ms": 0.0,
                "ready_upper_ms": 0.0, "stall_lower_ms": 0.0,
                "scope": "No remaining I/O; past readiness/deadline history is unknown"}
    first, last = [], []
    for queued, unissued in zip(queued_io_by_ssu, unissued_by_ssu):
        total = queued + unissued
        if total:
            first.append(0.0 if queued else service)
            last.append((total - bool(queued)) * service)
    link_remaining = max(0, link_pending_io - 1) * tail
    if link_pending_io and link_active_remaining_bytes is not None:
        link_remaining += link_active_remaining_bytes / IO_BYTES * tail
    future_link = (sum(queued_io_by_ssu) + sum(unissued_by_ssu)) * tail
    disk_lower = max(last) + tail if last else 0.0
    link_lower = max(link_remaining, min(first) if first else 0.0) + future_link
    lower = max(disk_lower, link_lower)
    return {"verdict": "must_stall" if lower > deadline_ms + EPS_MS else "unknown",
            "ready_lower_ms": lower, "ready_upper_ms": None,
            "stall_lower_ms": max(0.0, lower - deadline_ms),
            "scope": "Strict optimistic bound for one partially submitted layer; ignores unknown competitor ordering"}


@dataclass(frozen=True)
class Request:
    request_id: int
    io_ssus_by_layer: tuple[tuple[int, ...], ...]
    compute_ms: float
    layers: int = 8

    def placement(self, layer):
        return self.io_ssus_by_layer[0 if len(self.io_ssus_by_layer) == 1 else layer]


@dataclass(frozen=True)
class NPUState:
    """Client-visible partial snapshot; ownership counts, not SSD FIFO order.

    next_layer is the next computation not yet started. All queued/unissued/
    link counts describe that first pending layer, including a prefetched
    next request's L0 during the active request's final compute. None for
    unissued_by_ssu means this layer has NOT been activated. A tuple of zeros
    means it was activated and all commands have been issued. queued_io_by_ssu
    excludes link-only I/Os; link_pending_io includes active plus waiting.

    emitter_ssu_order is client-owned round-robin submission-state order, NOT
    the SSU service order. Supply it for a warm snapshot; otherwise sorted
    nonempty SSUs are an explicit scenario assumption. ready_in_ms is used
    only if all stages have drained. Negative compute_end preserves elapsed
    stall, while future_stall_ms only counts the time after this snapshot.
    """
    request: Request | None = None
    next_layer: int = 0
    compute_end_in_ms: float = 0.0
    queued_io_by_ssu: tuple[int, ...] = ()
    unissued_by_ssu: tuple[int, ...] | None = None
    link_pending_io: int = 0
    link_active_remaining_bytes: float | None = None
    ready_in_ms: float = 0.0
    next_issue_in_ms: float = 0.0
    waiting_requests: tuple[Request, ...] = ()
    emitter_ssu_order: tuple[int, ...] = ()


class LegacyMT19937:
    """Tiny MT19937 + mask/reject shuffle matching NumPy RandomState integers.

    This optional cold-start reproducibility mechanism is not device telemetry.
    A warm rollout without the original RNG position must remain a scenario.
    """
    def __init__(self, seed):
        self.words = [seed & 0xffffffff]
        for i in range(1, 624):
            previous = self.words[-1]
            self.words.append((1812433253 * (previous ^ (previous >> 30)) + i) & 0xffffffff)
        self.index = 624

    def uint32(self):
        if self.index == 624:
            for i in range(624):
                y = (self.words[i] & 0x80000000) | (self.words[(i + 1) % 624] & 0x7fffffff)
                self.words[i] = (self.words[(i + 397) % 624] ^ (y >> 1)
                                 ^ (0x9908b0df if y & 1 else 0))
            self.index = 0
        y = self.words[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        return (y ^ (y >> 18)) & 0xffffffff

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            mask = (1 << i.bit_length()) - 1
            j = self.uint32() & mask
            while j > i:
                j = self.uint32() & mask
            items[i], items[j] = items[j], items[i]


def forecast_baseline(npus, *, num_ssu, disk_gib_s=40.0, link_gib_s=50.0,
                      issue_interval_us=0.1, cross_request_prefetch=True,
                      issue_order_seed=None, queue_order=None, queue_layout="round_robin",
                      disk_active_remaining_bytes=None, disk_active_npu_ids=None):
    """Finite closed-loop max-plus forecast, with explicit receive queues.

    Native submission groups each layer's manifest by SSU (preserving block
    indices inside each group), then round-robins nonempty SSUs per NPU.
    issue_order_seed reproduces RandomState same-time NPU shuffles from its
    initial position. With a warm snapshot it is only a chosen future order.
    Initial FIFO ownership is reconstructed from counts, never secretly read.
    Complexity is proportional to remaining I/O events, not NPU factorial.
    """
    if queue_layout not in ("round_robin", "grouped"):
        raise ValueError("queue_layout must be round_robin or grouped")
    n = len(npus)
    order = tuple(range(n)) if queue_order is None else tuple(queue_order)
    residuals = ((None,) * num_ssu if disk_active_remaining_bytes is None
                 else disk_active_remaining_bytes)
    active_ids = ((None,) * num_ssu if disk_active_npu_ids is None else disk_active_npu_ids)
    rng = None if issue_order_seed is None else LegacyMT19937(issue_order_seed)
    disk_service, link_service = io_service_ms(disk_gib_s), io_service_ms(link_gib_s)
    disk_tail, link_tail = [0.0] * num_ssu, [0.0] * n
    jobs, positions, running = [], [0] * n, []
    next_issue = [max(0.0, state.next_issue_in_ms) for state in npus]
    rows, events, completions = [], [], {}
    sequence, generation, pending_issue = 0, 0, None

    def push(time, kind, resource, payload=None):
        nonlocal sequence
        sequence += 1
        heapq.heappush(events, (time, kind, resource, sequence, payload))

    def head(i):
        return jobs[i][positions[i]] if positions[i] < len(jobs[i]) else None

    def make_groups(placement):
        groups = {}
        for block, ssu in enumerate(placement):
            groups.setdefault(ssu, deque()).append(block)
        return groups

    for i, state in enumerate(npus):
        lane = []
        manifests = ([] if state.request is None else [(state.request, state.next_layer)])
        manifests += [(request, 0) for request in state.waiting_requests]
        for request, first_layer in manifests:
            for layer in range(first_layer, request.layers):
                groups = make_groups(request.placement(layer))
                lane.append({"request_id": request.request_id, "layer": layer,
                             "compute_ms": request.compute_ms, "last": layer == request.layers - 1,
                             "released": False, "groups": groups, "emit": deque(sorted(groups)),
                             "remaining": len(request.placement(layer)), "ready_ms": None,
                             "io_blocks_by_ssu": tuple(len(groups.get(s, ())) for s in range(num_ssu)),
                             "ssd_ready_ms_by_ssu": [0.0] * num_ssu,
                             "deadline_ms": None, "release_ms": None})
        jobs.append(lane)
        running.append(state.compute_end_in_ms > 0)
        if running[i]:
            end_id = (state.request.request_id if state.request is not None
                      and state.next_layer == state.request.layers else None)
            push(state.compute_end_in_ms, 0, i, end_id)
        if lane and state.unissued_by_ssu is not None:
            job = lane[0]
            for ssu, group in job["groups"].items():
                count = state.unissued_by_ssu[ssu]
                job["groups"][ssu] = deque(list(group)[-count:]) if count else deque()
            emit = state.emitter_ssu_order or tuple(sorted(job["groups"]))
            job.update(released=True, deadline_ms=state.compute_end_in_ms, release_ms=0.0,
                       emit=deque(ssu for ssu in emit if job["groups"][ssu]),
                       remaining=(sum(state.queued_io_by_ssu) + sum(state.unissued_by_ssu)
                                  + state.link_pending_io))

    def schedule_issue(now):
        nonlocal generation, pending_issue
        ready = [max(now, next_issue[i]) for i in range(n)
                 if head(i) is not None and head(i)["released"] and head(i)["emit"]]
        if not ready:
            return
        when = min(ready)
        if pending_issue is not None and pending_issue <= when + EPS_MS:
            return
        generation += 1
        pending_issue = when
        push(when, 3, 0, generation)

    def release(i, now, deadline):
        job = head(i)
        job.update(released=True, deadline_ms=deadline, release_ms=now)
        if not job["remaining"]:
            job["ready_ms"] = now
        schedule_issue(now)

    def start(i, now):
        job = head(i)
        if running[i] or job is None:
            return
        if not job["released"]:
            release(i, now, now)
        if job["remaining"] or job["ready_ms"] is None or job["ready_ms"] > now + EPS_MS:
            return
        deadline, end = job["deadline_ms"], now + job["compute_ms"]
        rows.append({"npu_id": i, "request_id": job["request_id"], "layer": job["layer"],
                     "io_blocks_by_ssu": job["io_blocks_by_ssu"],
                     "ssd_ready_ms_by_ssu": tuple(job["ssd_ready_ms_by_ssu"]),
                     "deadline_ms": deadline, "io_release_ms": job["release_ms"],
                     "io_ready_ms": job["ready_ms"], "compute_start_ms": now,
                     "compute_end_ms": end, "stall_ms": max(0.0, now - deadline),
                     "future_stall_ms": max(0.0, now - max(0.0, deadline)),
                     "is_layer0": job["layer"] == 0})
        positions[i] += 1
        running[i] = True
        push(end, 0, i, job["request_id"] if job["last"] else None)
        following = head(i)
        if following is not None and (cross_request_prefetch
                                     or following["request_id"] == job["request_id"]):
            release(i, now, end)

    # Each initial link contains only the first pending layer for this NPU.
    for i, state in enumerate(npus):
        job = head(i)
        if job is None or not job["released"]:
            continue
        if state.link_pending_io:
            first = (IO_BYTES if state.link_active_remaining_bytes is None
                     else state.link_active_remaining_bytes)
            link_tail[i] = (first / IO_BYTES + state.link_pending_io - 1) * link_service
            push(link_tail[i], 2, i, (job, state.link_pending_io))
        if job["remaining"] == 0:
            job["ready_ms"] = max(0.0, state.ready_in_ms)
            push(job["ready_ms"], 2.75, i)

    # A scenario queue consistent with public per-owner counts, not an oracle.
    for ssu in range(num_ssu):
        counts = [(state.queued_io_by_ssu[ssu] if state.queued_io_by_ssu else 0)
                  for state in npus]
        owners = []
        active = active_ids[ssu]
        if active is not None and counts[active]:
            owners.append(active)
            counts[active] -= 1
        if queue_layout == "grouped":
            owners.extend(i for i in order for _ in range(counts[i]))
        else:
            while any(counts):
                for i in order:
                    if counts[i]:
                        owners.append(i)
                        counts[i] -= 1
        for k, i in enumerate(owners):
            duration = disk_service
            if not k and residuals[ssu] is not None:
                duration *= residuals[ssu] / IO_BYTES
            disk_tail[ssu] += duration
            push(disk_tail[ssu], 1, ssu, (i, head(i), k, ssu))
    for i in range(n):
        push(0.0, 2.75, i)
    schedule_issue(0.0)

    while events:
        now, kind, resource, _, payload = heapq.heappop(events)
        if kind == 0:
            running[resource] = False
            if payload is not None:
                completions[payload] = now
            push(now, 2.75, resource)
        elif kind == 1:
            simultaneous = [payload]
            # Native groups SSD completions within its numerical tolerance,
            # sorting simultaneous arrivals by request/layer/block/SSU.
            while events and events[0][1] == 1 and abs(events[0][0] - now) <= EPS_MS:
                next_event = heapq.heappop(events)
                now = max(now, next_event[0])
                simultaneous.append(next_event[4])
            simultaneous.sort(key=lambda flow: (flow[0], flow[1]["request_id"],
                                                 flow[1]["layer"], flow[2], flow[3]))
            for i, job, _, ssu in simultaneous:
                job["ssd_ready_ms_by_ssu"][ssu] = now
                link_tail[i] = max(now, link_tail[i]) + link_service
                push(link_tail[i], 2, i, (job, 1))
        elif kind == 2:
            job, count = payload
            job["remaining"] -= count
            if not job["remaining"]:
                job["ready_ms"] = now
                push(now, 2.75, resource)
        elif kind == 2.75:
            start(resource, now)
        else:
            if payload != generation:
                continue
            pending_issue = None
            ready = [i for i in range(n) if head(i) is not None and head(i)["released"]
                     and head(i)["emit"] and next_issue[i] <= now + EPS_MS]
            if rng is not None:
                rng.shuffle(ready)
            for i in ready:
                job = head(i)
                ssu = job["emit"].popleft()
                block = job["groups"][ssu].popleft()
                if job["groups"][ssu]:
                    job["emit"].append(ssu)
                disk_tail[ssu] = max(now, disk_tail[ssu]) + disk_service
                push(disk_tail[ssu], 1, ssu, (i, job, block, ssu))
                next_issue[i] = now + issue_interval_us / 1000.0
            schedule_issue(now)
    return {"kind": "conditional_prediction", "guaranteed": False,
            "assumptions": {"unknown_future_arrivals": "none", "num_npu": n,
                            "num_ssu": num_ssu, "queue_layout": queue_layout,
                            "queue_order": list(order), "issue_order_seed": issue_order_seed,
                            "warm_seed_position": "initial, NOT observed native RNG continuation",
                            "unknown_active_residual": "one full I/O scenario",
                            "receive_model": "one shared FCFS link per NPU",
                            "cross_request_prefetch": cross_request_prefetch},
            "layers": rows, "request_completion_ms": completions,
            "future_stall_ms_by_npu": [sum(r["future_stall_ms"] for r in rows if r["npu_id"] == i)
                                       for i in range(n)],
            "non_layer0_stall_ms_by_npu": [sum(r["future_stall_ms"] for r in rows
                                                if r["npu_id"] == i and not r["is_layer0"])
                                           for i in range(n)]}


def predict_request_impact(npus, new_request, target_npu, **environment):
    """Paired conditional forecasts; a new arrival cannot erase queued work."""
    before = forecast_baseline(npus, **environment)
    changed = list(npus)
    target = changed[target_npu]
    old_head = (target.request is not None and target.next_layer < target.request.layers
                or bool(target.waiting_requests))
    if not old_head:
        target = replace(target, queued_io_by_ssu=(), unissued_by_ssu=None,
                         link_pending_io=0, ready_in_ms=0.0, emitter_ssu_order=())
    if target.request is None and not target.waiting_requests:
        changed[target_npu] = replace(target, request=new_request)
    else:
        changed[target_npu] = replace(target, waiting_requests=target.waiting_requests + (new_request,))
    after = forecast_baseline(changed, **environment)
    old_rows = {(r["request_id"], r["layer"]): r for r in before["layers"]}
    deltas = []
    for row in after["layers"]:
        key = row["request_id"], row["layer"]
        if key in old_rows:
            deltas.append({"request_id": key[0], "layer": key[1], "npu_id": row["npu_id"],
                           "compute_start_delay_ms": row["compute_start_ms"] - old_rows[key]["compute_start_ms"],
                           "extra_future_stall_ms": row["future_stall_ms"] - old_rows[key]["future_stall_ms"]})
    return {"kind": "conditional_prediction", "guaranteed": False,
            "candidate_request_id": new_request.request_id, "target_npu": target_npu,
            "candidate_layers": [r for r in after["layers"] if r["request_id"] == new_request.request_id],
            "existing_layer_deltas": deltas,
            "existing_request_completion_delay_ms": {
                rid: after["request_completion_ms"][rid] - end
                for rid, end in before["request_completion_ms"].items()},
            "without_candidate": before, "with_candidate": after}


def impact_order_sensitivity(npus, new_request, target_npu, *, scenarios=None, **environment):
    """A small declared sample, not a combinatorial bound or admission proof.

    Default uses two FIFO layouts and forward/reverse owner order (four
    rollouts pairs), so 32 lanes do not imply 32! work. Add measured plausible
    scenarios if needed. The sampled min/max cannot certify no stall.
    """
    n = len(npus)
    if scenarios is None:
        scenarios = tuple({"queue_layout": layout, "queue_order": order}
                          for layout in ("round_robin", "grouped")
                          for order in (tuple(range(n)), tuple(reversed(range(n)))))
    samples = []
    for scenario in scenarios:
        result = predict_request_impact(npus, new_request, target_npu,
                                         **dict(environment, **scenario))
        samples.append({"scenario": scenario,
                        "candidate_completion_ms": result["with_candidate"]["request_completion_ms"][new_request.request_id],
                        "candidate_non_layer0_stall_ms": sum(r["future_stall_ms"] for r in result["candidate_layers"]
                                                               if not r["is_layer0"]),
                        "largest_existing_completion_delay_ms": max(
                            result["existing_request_completion_delay_ms"].values(), default=0.0)})
    return {"kind": "scenario_sensitivity", "range_is_guaranteed": False, "samples": samples}


if __name__ == "__main__":
    placement = tuple(i % 6 for i in range(1535))
    states = [NPUState(Request(0, (placement,), 4.533469))] + [NPUState()] * 31
    report = forecast_baseline(states, num_ssu=6, issue_order_seed=42)
    print(json.dumps({"current_layer": screen_current_layer(
        [placement.count(s) for s in range(6)], 4.533469, [0] * 6),
        "first_two_layers": report["layers"][:2], "assumptions": report["assumptions"]}, indent=2))
