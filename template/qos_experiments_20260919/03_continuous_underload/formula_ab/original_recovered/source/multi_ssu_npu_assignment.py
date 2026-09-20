"""Causal NPU placement with per-SSD and per-NPU-link capacity estimates.

One I/O is one 176 KiB KV block. This is an online, whole-backlog heuristic,
not an exact deadline predictor. It observes only already-arrived requests,
their immutable SSD placement, and NPU-side compute/completion counters.
Assigning an NPU does not move SSD blocks, migrate an old request, or change
arrival times. Use the simulator adapter in separate processes, not threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import inf, isclose
from time import perf_counter


IO_SIZE_BYTES = 176 * 1024
IO_SIZE_GIB = IO_SIZE_BYTES / 1024**3


@dataclass(frozen=True)
class MultiNPUBacklog:
    npu_id: int
    remaining_compute_ms: float
    remaining_io_by_ssu: tuple[int, ...]
    unfinished_request_count: int = 0


@dataclass(frozen=True)
class MultiAssignmentDecision:
    npu_id: int
    selection_scores_ms_by_npu: tuple[float, ...]
    estimated_finish_ms_by_npu: tuple[float, ...]
    disk_overload_factors_by_candidate: tuple[float, ...]
    link_overload_factors_by_candidate: tuple[float, ...]
    selection_objective: str


def _rate(block_count, compute_ms):
    if not block_count:
        return 0.0
    return block_count * IO_SIZE_GIB * 1000.0 / compute_ms if compute_ms > 0 else inf


def choose_npu(
    now_ms,
    backlogs,
    new_io_by_ssu,
    new_compute_ms,
    *,
    disk_bandwidths_gib_s,
    npu_bandwidths_gib_s=None,
    policy="fluid",
):
    """Choose using *total remaining* work, including unissued future layers.

    Both incoming arguments cover the complete new request, not one layer.
    Bandwidth and output arrays follow the order of ``backlogs``; IDs need
    not equal array indices. For each candidate assignment form C[n] and
    V[n,s] (GiB), then score:

      now + max_n C[n] * max(1,
          max_s sum_n (1000 V[n,s]/C[n]) / B[s],
          max_n (1000 sum_s V[n,s]/C[n]) / Link[n]).

    The compute-only ablation scores the selected NPU's remaining compute.
    The fluid score is a coarse fleet makespan estimate, not an actual
    request finish, deadline guarantee, or proof of optimal assignment.
    In particular, burst order and prefetch overlap are not modeled here.
    """
    backlogs = tuple(backlogs)
    disks = tuple(disk_bandwidths_gib_s)
    links = tuple(npu_bandwidths_gib_s) if npu_bandwidths_gib_s is not None else (50.0,) * len(backlogs)
    incoming = tuple(new_io_by_ssu)
    if not backlogs or len(links) != len(backlogs) or len(incoming) != len(disks):
        raise ValueError("provide one link capacity per NPU and one work count per SSU")
    if any(len(b.remaining_io_by_ssu) != len(disks) for b in backlogs):
        raise ValueError("every NPU backlog needs one work count per SSU")
    if policy not in ("fluid", "compute"):
        raise ValueError("policy must be 'fluid' or 'compute'")

    selection, finishes, disk_factors, link_factors = [], [], [], []
    for candidate, backlog in enumerate(backlogs):
        compute = [b.remaining_compute_ms for b in backlogs]
        compute[candidate] += new_compute_ms
        work = [b.remaining_io_by_ssu for b in backlogs]
        work[candidate] = tuple(a + b for a, b in zip(work[candidate], incoming))
        disk_demand = [sum(_rate(work[n][s], compute[n]) for n in range(len(backlogs)))
                       for s in range(len(disks))]
        disk_factor = max(demand / capacity for demand, capacity in zip(disk_demand, disks))
        link_factor = max(_rate(sum(counts), duration) / capacity
                          for counts, duration, capacity in zip(work, compute, links))
        own_io_ms = max(
            max(count * IO_SIZE_GIB / capacity * 1000
                for count, capacity in zip(work[candidate], disks)),
            sum(work[candidate]) * IO_SIZE_GIB / links[candidate] * 1000,
        )
        finish = now_ms + max(compute[candidate], own_io_ms)
        score = (now_ms + max(compute) * max(1.0, disk_factor, link_factor)
                 if policy == "fluid" else now_ms + compute[candidate])
        selection.append(score)
        finishes.append(finish)
        disk_factors.append(disk_factor)
        link_factors.append(link_factor)

    chosen = min(range(len(backlogs)), key=lambda n: (
        selection[n], finishes[n] if policy == "fluid" else 0.0,
        backlogs[n].unfinished_request_count, backlogs[n].npu_id,
    ))
    return MultiAssignmentDecision(
        backlogs[chosen].npu_id, tuple(selection), tuple(finishes),
        tuple(disk_factors), tuple(link_factors),
        "estimated_fleet_fluid_makespan_per_ssu_and_link" if policy == "fluid"
        else "remaining_compute_on_selected_npu",
    )


def request_layer_io_by_ssu(manifest, n_layers, num_ssu):
    """Read immutable client manifest; reject a different I/O unit explicitly."""
    rows = []
    for placement in manifest.placement:
        counts = [0] * num_ssu
        for ssu, size_gib in placement:
            if not isclose(size_gib, IO_SIZE_GIB, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("assignment model requires one whole 176 KiB I/O per KV block")
            counts[ssu] += 1
        rows.append(tuple(counts))
    if len(rows) == 1:
        return tuple(rows) * n_layers
    if len(rows) != n_layers:
        raise ValueError("manifest must contain one repeated placement or every layer")
    return tuple(rows)


def snapshot_npu_backlogs(context, now_ms, layer_io_cache=None):
    """NPU-visible residual work, including link-only uncompleted commands.

    Per-SSU completed bytes are client completion counters, not SSD internal
    FIFO telemetry. Counting link-only data again as SSD work is conservative
    and can overestimate the fluid load; it is not an exact queue snapshot.
    The optional cache is populated only for requests that already arrived.
    """
    cache = {} if layer_io_cache is None else layer_io_cache
    result = []
    for npu in context.npus:
        ids = list(npu.admission_queue)
        compute_ms = 0.0
        batch = npu.active_batch
        if batch is not None:
            rid = batch.member_request_ids[0]
            ids.append(rid)
            per_layer = context.requests[rid].per_layer_compute_ms
            if npu.compute_active is not None:
                _, layer = npu.compute_active
                compute_ms += max(0.0, batch.layer_metrics[layer].compute_end_ms - now_ms)
                compute_ms += (context.n_layers - layer - 1) * per_layer
            else:
                compute_ms += (context.n_layers - batch.compute_done_up_to - 1) * per_layer
        for rid in npu.admission_queue:
            compute_ms += context.n_layers * context.requests[rid].per_layer_compute_ms

        remaining = [0] * context.num_ssu
        for rid in ids:
            request = context.requests[rid]
            if rid not in cache:
                cache[rid] = request_layer_io_by_ssu(request.manifest, context.n_layers, context.num_ssu)
            completed = request.completed_gb_by_layer_ssu
            for layer, counts in enumerate(cache[rid]):
                if request.io_ready[layer]:
                    continue
                for ssu, count in enumerate(counts):
                    done = round(completed[layer][ssu] / IO_SIZE_GIB) if completed else 0
                    remaining[ssu] += count - done
        result.append(MultiNPUBacklog(npu.npu_id, compute_ms, tuple(remaining), len(ids)))
    return tuple(result)


@contextmanager
def assign_requests_on_arrival(*, policy="fluid"):
    """Temporarily assign each request at its actual simulator arrival.

    The audit log records actual assignments, measured decision CPU time,
    and conditional scores. Original input manifests remain untouched. This
    wrapper maintains the simulator's per-NPU future-arrival bookkeeping but
    never reads that count when deciding, nor scans future request profiles.
    """
    import continuous_batch_sim as native

    original = native._handle_arrival
    decisions, cache = [], {}

    def arrival(context, request_id, now_ms):
        if context.batch_size != 1 or context.profile_cycle_probe_npu_ids:
            raise ValueError("assignment requires batch_size=1 and no profile-cycle probe")
        started = perf_counter()
        request = context.requests[request_id]
        old = request.manifest
        backlogs = snapshot_npu_backlogs(context, now_ms, cache)
        cache[request_id] = request_layer_io_by_ssu(old, context.n_layers, context.num_ssu)
        incoming = tuple(sum(row[s] for row in cache[request_id]) for s in range(context.num_ssu))
        decision = choose_npu(
            now_ms, backlogs, incoming, context.n_layers * request.per_layer_compute_ms,
            disk_bandwidths_gib_s=(context.disk_bw_gbps,) * context.num_ssu,
            npu_bandwidths_gib_s=(context.npu_bw_gbps,) * context.num_npu,
            policy=policy,
        )
        if decision.npu_id != old.npu_id:
            context.npus[old.npu_id].future_arrivals -= 1
            context.npus[decision.npu_id].future_arrivals += 1
            request.manifest = replace(old, npu_id=decision.npu_id,
                                       load={**old.load, "npu_id": decision.npu_id})
        decisions.append({
            "request_id": request_id, "arrival_time_ms": now_ms,
            "original_npu_id": old.npu_id, "assigned_npu_id": decision.npu_id,
            "policy": policy, "selection_objective": decision.selection_objective,
            "selection_scores_ms_by_npu": decision.selection_scores_ms_by_npu,
            "estimated_finish_ms_by_npu": decision.estimated_finish_ms_by_npu,
            "disk_overload_factors_by_candidate": decision.disk_overload_factors_by_candidate,
            "link_overload_factors_by_candidate": decision.link_overload_factors_by_candidate,
            "known_remaining_compute_ms_by_npu": [b.remaining_compute_ms for b in backlogs],
            "known_remaining_io_by_npu_ssu": [b.remaining_io_by_ssu for b in backlogs],
            "incoming_total_io_by_ssu": incoming,
            "decision_wall_time_us": (perf_counter() - started) * 1e6,
        })
        original(context, request_id, now_ms)

    native._handle_arrival = arrival
    try:
        yield decisions
    finally:
        native._handle_arrival = original
