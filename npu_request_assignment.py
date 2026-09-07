"""Causal, arrival-time NPU assignment for independent prefill requests.

One I/O is one 176 KiB KV block.  The pure selector uses only unfinished
compute and uncompleted read counts reported by each NPU.  It does NOT
inspect SSD queue order, an in-service command's remainder, future arrivals,
or future simulation events.  Its completion estimates are heuristics, not
deadline guarantees: shared-SSD interference and cross-request prefetch are
approximated by overlapping compute and an assumed per-NPU SSD share.

``assign_requests_on_arrival`` adapts the selector to this repository without
changing the simulator.  Use it around one simulation in a single process;
the temporary arrival-handler wrapper is restored even after an exception.
Assignment never migrates an arrived request or changes its arrival time.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Sequence


IO_SIZE_BYTES = 176 * 1024


@dataclass(frozen=True)
class NPUBacklog:
    npu_id: int
    remaining_compute_ms: float
    remaining_io_count: int
    unfinished_request_count: int = 0


@dataclass(frozen=True)
class AssignmentDecision:
    npu_id: int
    predicted_finish_ms_by_npu: tuple[float, ...]
    assumed_share_gib_s: float
    selection_scores_ms_by_npu: tuple[float, ...] = ()
    selection_objective: str = "estimated_new_request_finish"


def choose_npu(
    now_ms: float,
    backlogs: Sequence[NPUBacklog],
    kv_block_count: int,
    per_layer_compute_ms: float,
    n_layers: int = 8,
    bandwidth_gib_s: float = 40.0,
    *,
    policy: str = "pipeline",
) -> AssignmentDecision:
    """Choose the earliest estimated finish among already observed backlogs.

    For N NPUs and one shared disk use b=B/N and tau=176 KiB/b.
    ``pipeline`` scores max(R_compute, R_io*tau) +
    max(L*C, L*K*tau).  ``compute`` instead scores R_compute + L*C;
    the latter is a useful ablation. ``fluid`` evaluates all candidate
    placements using max(total compute per NPU) multiplied by the implied
    fleet bandwidth overload factor. This is a coarse whole-backlog
    makespan objective, not the new request's own finish time. None of these
    predictions is a bound.

    Counts include all unfinished reads of already assigned requests, even
    reads not yet submitted.  They exclude the incoming request, which is
    added here once.  NPU IDs are returned unchanged, not list indices.
    """
    share = bandwidth_gib_s / len(backlogs)
    io_ms = IO_SIZE_BYTES / (share * 1024**3) * 1000.0
    own_compute = n_layers * per_layer_compute_ms
    own_io_ms = n_layers * kv_block_count * io_ms
    scores = []
    selection_scores = []
    for backlog in backlogs:
        if policy == "compute":
            duration = backlog.remaining_compute_ms + own_compute
        elif policy in ("pipeline", "fluid"):
            duration = max(
                backlog.remaining_compute_ms, backlog.remaining_io_count * io_ms
            ) + max(own_compute, own_io_ms)
        else:
            raise ValueError("policy must be 'pipeline', 'compute', or 'fluid'")
        scores.append(now_ms + duration)
        if policy == "fluid":
            compute = [b.remaining_compute_ms for b in backlogs]
            io_counts = [b.remaining_io_count for b in backlogs]
            candidate_index = len(scores) - 1
            compute[candidate_index] += own_compute
            io_counts[candidate_index] += n_layers * kv_block_count
            total_demand = sum(
                count * IO_SIZE_BYTES / 1024**3 * 1000 / compute_ms
                for count, compute_ms in zip(io_counts, compute)
                if compute_ms > 0
            )
            selection_scores.append(
                now_ms + max(compute) * max(1.0, total_demand / bandwidth_gib_s)
            )
        else:
            selection_scores.append(scores[-1])
    selected = min(
        range(len(backlogs)),
        key=lambda index: (
            selection_scores[index],
            scores[index],
            backlogs[index].unfinished_request_count,
            backlogs[index].npu_id,
        ),
    )
    return AssignmentDecision(
        backlogs[selected].npu_id, tuple(scores), share,
        tuple(selection_scores),
        "estimated_whole_backlog_fluid_makespan" if policy == "fluid"
        else "estimated_new_request_finish",
    )


def _read_count(request, n_layers: int) -> int:
    """Uncompleted whole I/Os, using client-visible completion counters."""
    placement = request.manifest.placement
    count = 0
    for layer in range(n_layers):
        if request.io_ready[layer]:
            continue
        count += (
            request.pending_blocks[layer]
            if request.io_started[layer]
            else len(placement[0 if len(placement) == 1 else layer])
        )
    return count


def snapshot_npu_backlogs(context, now_ms: float) -> tuple[NPUBacklog, ...]:
    """Adapt NPU progress/admission queues; never traverse SSU queues/events.

    This adapter is for batch_size=1.  Local NPU admission-queue order is
    available to its runtime; it is distinct from the unavailable SSU FIFO
    ordering.  The selector itself only receives aggregate counters.
    """
    snapshots = []
    for npu in context.npus:
        ids = list(npu.admission_queue)
        compute_ms = 0.0
        batch = npu.active_batch
        if batch is not None:
            request_id = batch.member_request_ids[0]
            ids.append(request_id)
            compute_per_layer = context.requests[request_id].per_layer_compute_ms
            if npu.compute_active is not None:
                _, layer = npu.compute_active
                compute_ms += max(
                    0.0, batch.layer_metrics[layer].compute_end_ms - now_ms
                ) + (context.n_layers - layer - 1) * compute_per_layer
            else:
                compute_ms += (
                    context.n_layers - batch.compute_done_up_to - 1
                ) * compute_per_layer
        for request_id in npu.admission_queue:
            compute_ms += (
                context.n_layers * context.requests[request_id].per_layer_compute_ms
            )
        snapshots.append(NPUBacklog(
            npu.npu_id,
            compute_ms,
            sum(_read_count(context.requests[rid], context.n_layers) for rid in ids),
            len(ids),
        ))
    return tuple(snapshots)


@contextmanager
def assign_requests_on_arrival(*, policy: str = "pipeline"):
    """Yield an audit log while assigning at actual simulator arrival events.

    Example::

        with assign_requests_on_arrival() as decisions:
            result = simulate_continuous_batch(requests, batch_size=1, ...)

    Use independent OS processes for parallel runs, not threads.  Profile
    cycle probes depend on the original NPU assignment and are unsupported;
    ordinary steady-state/timeline measurements continue to use actual NPUs.
    The input manifests remain unchanged, and the output request rows expose
    the actual chosen NPU.  Hardware placement and request IDs are preserved.
    """
    import continuous_batch_sim as batch_sim

    original_arrival = batch_sim._handle_arrival
    decisions = []

    def arrival(context, request_id: int, now_ms: float):
        if context.batch_size != 1 or context.profile_cycle_probe_npu_ids:
            raise ValueError("assignment requires batch_size=1 and no profile-cycle probe")
        started = perf_counter()
        request = context.requests[request_id]
        old = request.manifest
        backlogs = snapshot_npu_backlogs(context, now_ms)
        # This project workload repeats one per-layer read count for 8 layers.
        # Selecting a heterogeneous-layer request would need its full profile.
        decision = choose_npu(
            now_ms,
            backlogs,
            len(old.placement[0]),
            request.per_layer_compute_ms,
            context.n_layers,
            context.disk_bw_gbps,
            policy=policy,
        )
        if decision.npu_id != old.npu_id:
            context.npus[old.npu_id].future_arrivals -= 1
            context.npus[decision.npu_id].future_arrivals += 1
            request.manifest = replace(
                old, npu_id=decision.npu_id,
                load={**old.load, "npu_id": decision.npu_id},
            )
        decisions.append({
            "request_id": request_id,
            "arrival_time_ms": now_ms,
            "original_npu_id": old.npu_id,
            "assigned_npu_id": decision.npu_id,
            "policy": policy,
            "estimated_finish_ms_by_npu": list(decision.predicted_finish_ms_by_npu),
            "selection_scores_ms_by_npu": list(decision.selection_scores_ms_by_npu),
            "selection_objective": decision.selection_objective,
            "assumed_share_gib_s": decision.assumed_share_gib_s,
            "known_remaining_compute_ms_by_npu": [b.remaining_compute_ms for b in backlogs],
            "known_remaining_io_by_npu": [b.remaining_io_count for b in backlogs],
            "decision_wall_time_us": (perf_counter() - started) * 1e6,
        })
        original_arrival(context, request_id, now_ms)

    batch_sim._handle_arrival = arrival
    try:
        yield decisions
    finally:
        batch_sim._handle_arrival = original_arrival


def fixed_window_metrics(summary, start_ms=1000.0, end_ms=2000.0):
    """Clip completed simulation intervals to one identical wall-clock window."""
    def overlap(start, end):
        return max(0.0, min(end, end_ms) - max(start, start_ms))

    compute = [0.0] * 4
    active = [0.0] * 4
    stall = [0.0] * 4
    for batch in summary["microbatch_metrics"]:
        npu = batch["npu_id"]
        active[npu] += overlap(batch["admission_time_ms"], batch["completion_time_ms"])
        for layer in batch["layer_metrics"]:
            compute[npu] += overlap(layer["compute_start_ms"], layer["compute_end_ms"])
            stall[npu] += overlap(
                layer["compute_start_ms"] - layer["io_barrier_wait_ms"],
                layer["compute_start_ms"],
            )
    duration = end_ms - start_ms
    return {
        "start_ms": start_ms, "end_ms": end_ms,
        "mean_npu_utilization": sum(compute) / (4 * duration),
        "npu_utilizations": [value / duration for value in compute],
        "compute_ms_by_npu": compute, "io_stall_ms_by_npu": stall,
        "active_request_ms_by_npu": active,
        "idle_without_active_request_ms_by_npu": [duration - value for value in active],
        "all_npus_have_active_requests_throughout": all(value >= duration - 1e-6 for value in active),
    }


def main():
    """Reproduce assignment experiments on the shared raw-data variable trace."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from run_stall_policy_experiments import build_variable_requests, run_case

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--policy", choices=("pipeline", "compute", "fluid"), default="pipeline")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--arrival-gib-s", type=float, default=39.2)
    parser.add_argument("--duration-ms", type=float, default=2000.0)
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument("--mix", choices=("long", "broad"), default="long")
    parser.add_argument("--output", type=Path, default=Path("results/stall_prediction_experiments/data/strategy_b"))
    args = parser.parse_args()
    requests, pool, metadata = build_variable_requests(
        seed=args.seed, arrival_gib_s=args.arrival_gib_s,
        duration_ms=args.duration_ms, burst_size=args.burst_size, mix=args.mix,
    )
    with assign_requests_on_arrival(policy=args.policy) as decisions:
        result = run_case(pool, "dedicated_demand", seed=args.seed,
                          requests=requests, complete_all=True)
    result["variable_input"] = metadata
    result["assignment_policy"] = args.policy
    result["assignment_decisions"] = decisions
    result["assignment_implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["fixed_window"] = fixed_window_metrics(result["summary"])
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"assignment_variable_seed{args.seed}_burst{args.burst_size}_{args.policy}"
    if args.mix != "long":
        name += f"_{args.mix}"
    # Preserve the earlier exploratory files whose native submit seed was 42.
    # Same request fingerprints alone do not match simultaneous issue order.
    name += "_seedmatched"
    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"case": name, "fixed_window": result["fixed_window"],
                      "makespan_ms": result["summary"]["makespan_ms"],
                      "all_complete_utilization": result["summary"]["fleet_npu_compute_utilization"],
                      "mean_arrival_latency_ms": result["summary"]["avg_request_latency_ms"],
                      "wall_seconds": result["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
