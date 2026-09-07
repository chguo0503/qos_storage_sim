"""Development-only arrival placement and optional just-in-time prefetch.

These candidates leave the frozen shared-Path policies, static CIR/PIR and
native SSD arbitration unchanged. They read only arrived/known client work;
they neither know future requests nor predict the SSD's exact FIFO order.
Scores are fluid heuristics in milliseconds, not completion-time guarantees.
"""

from shared_path_common import IO_GIB, solo_io_ms
from shared_path_strategy1 import AssignmentChoice, KnownNPU


def _ssu_counts(counts):
    """Accept either per-SSU totals or per-SSU rows of sampled Path counts."""
    return tuple(row if isinstance(row, (int, float)) else sum(row) for row in counts)


def choose_candidate(now_ms, backlogs, incoming_layer_io_by_ssu, compute_ms,
                     n_layers=8, snapshot_counts_by_ssu=(),
                     disk_bw_gib_s=40.0, link_bw_gib_s=50.0, *,
                     mode="fair_pipeline"):
    """Same inputs/return type as shared_path_strategy1.choose_npu.

    ``compute`` minimizes the lane's remaining compute after assignment. It is
    deliberately a compute-only ablation, not a coflow-aware policy.

    ``fair_pipeline`` minimizes max(C_j, sum(V_j)/L, max_s A_s V_js/B).
    A_s counts lanes with *known unfinished work* on SSD s after this candidate
    assignment, not lanes currently issuing I/O. Equal sharing is an estimate;
    the real static category CIR scheduler does not promise it.

    ``compute_guarded_mix`` allows only lanes whose existing compute backlog is
    within one incoming layer's compute time of the minimum. Among those it
    minimizes own_pipeline * max(1, peak_projected_SSD_demand/B), where each
    lane's projected demand uses max(compute, isolated I/O time) as denominator.
    The guard limits extra *compute* backlog, not real admission waiting time.

    Full known request work is included, including unsubmitted future layers
    of already-arrived requests. No deadline or exact current-layer release
    state is available in KnownNPU, so none of these tests proves no stall.
    The global sampled queue is recorded but not a common score floor: such a
    floor can hide differences between placement candidates. Scores/details
    follow input order, while npu_id preserves the caller's actual identifiers.
    """
    if mode not in ("compute", "fair_pipeline", "compute_guarded_mix"):
        raise ValueError(f"unknown assignment candidate: {mode}")
    backlogs = tuple(backlogs)
    incoming = tuple(incoming_layer_io_by_ssu)
    new_work = tuple(n_layers * count for count in incoming)
    sampled_counts = _ssu_counts(snapshot_counts_by_ssu)
    guard_limit = min(row.remaining_compute_ms for row in backlogs) + compute_ms
    scores, details, eligible = [], [], []
    for candidate, state in enumerate(backlogs):
        compute = [row.remaining_compute_ms for row in backlogs]
        work = [row.unfinished_io_by_ssu for row in backlogs]
        compute[candidate] += n_layers * compute_ms
        work[candidate] = tuple(old + new for old, new in zip(work[candidate], new_work))
        own_work = work[candidate]
        own_solo_ms = solo_io_ms(own_work, disk_bw_gib_s, link_bw_gib_s)
        detail = {"candidate_npu_id": state.npu_id, "mode": mode,
                  "projected_compute_ms": compute[candidate],
                  "isolated_io_ms": own_solo_ms,
                  "sampled_pending_counts_by_ssu": sampled_counts,
                  "is_completion_guarantee": False}
        allowed = True
        if mode == "compute":
            duration = compute[candidate]
        elif mode == "fair_pipeline":
            active = tuple(sum(row[s] > 0 for row in work) for s in range(len(incoming)))
            disk_ms = tuple(1000 * IO_GIB * count * lanes / disk_bw_gib_s
                            for count, lanes in zip(own_work, active))
            link_ms = 1000 * IO_GIB * sum(own_work) / link_bw_gib_s
            duration = max(compute[candidate], link_ms, max(disk_ms, default=0.0))
            detail.update(known_work_lanes_by_ssu=active,
                          fair_disk_ms_by_ssu=disk_ms, receive_link_ms=link_ms)
        else:
            pipeline = [max(c, solo_io_ms(w, disk_bw_gib_s, link_bw_gib_s))
                        for c, w in zip(compute, work)]
            demand = tuple(sum(1000 * IO_GIB * row[s] / duration if duration else 0.0
                               for row, duration in zip(work, pipeline))
                           for s in range(len(incoming)))
            contention = max(1.0, max(demand, default=0.0) / disk_bw_gib_s)
            duration = pipeline[candidate] * contention
            allowed = state.remaining_compute_ms <= guard_limit
            detail.update(own_pipeline_ms=pipeline[candidate],
                          projected_demand_gib_s_by_ssu=demand,
                          disk_contention_factor=contention,
                          compute_guard_limit_ms=guard_limit)
        detail["eligible"] = allowed
        eligible.append(allowed)
        scores.append(now_ms + duration)
        details.append(detail)
    chosen = min((i for i in range(len(backlogs)) if eligible[i]), key=lambda i: (
        scores[i], backlogs[i].unfinished_request_count,
        backlogs[i].remaining_compute_ms, backlogs[i].npu_id))
    return AssignmentChoice(backlogs[chosen].npu_id, tuple(scores), tuple(details))


def prefetch_release_ms(now_ms, deadline_ms, layer_io_by_ssu,
                        pending_counts_by_ssu, guard_ms=5.0,
                        disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Return an optional common release time for a wholly unsubmitted coflow.

    estimated_ms = 1000*b*max(max_s((Q_s+K_s)/B), sum_s(K_s)/L)
    release_ms = max(now_ms, deadline_ms-estimated_ms-guard_ms).

    b is one 176 KiB I/O in GiB. Q may be cached per-Path rows or per-SSD total
    counts and must exclude this new coflow K. Disk components run in parallel;
    the NPU has one receive link, so link bytes are summed across disks.
    This delays only previously unsubmitted I/O, never already-queued I/O or
    compute. Snapshot age, category CIR, other arrivals and release order make
    the estimate heuristic, not an upper bound or a deadline guarantee.
    An empty coflow has no submission to postpone and returns now_ms.
    """
    incoming = tuple(layer_io_by_ssu)
    if not any(incoming):
        return float(now_ms)
    pending = _ssu_counts(pending_counts_by_ssu)
    disk_ms = max((1000 * IO_GIB * (q + k) / disk_bw_gib_s
                   for q, k in zip(pending, incoming)), default=0.0)
    link_ms = 1000 * IO_GIB * sum(incoming) / link_bw_gib_s
    return max(now_ms, deadline_ms - max(disk_ms, link_ms) - guard_ms)


if __name__ == "__main__":
    lanes = (KnownNPU(0, 10, (100, 100), 1), KnownNPU(1, 3, (10, 10), 1))
    for candidate_mode in ("compute", "fair_pipeline", "compute_guarded_mix"):
        result = choose_candidate(5, lanes, (64, 64), 1, mode=candidate_mode)
        print(candidate_mode, result.npu_id, result.scores_ms)
    print("prefetch release ms:", prefetch_release_ms(5, 20, (64, 64), ((8, 4), (3, 2))))
