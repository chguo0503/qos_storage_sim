"""Shared-Path Strategy 1: fair-pipeline placement and JIT prefetch.

Reads use New Once with static CIR/PIR. Only a just-arrived request may change
its NPU assignment; existing requests never migrate. A caller may defer a
wholly unsubmitted layer coflow using prefetch_release_ms. Neither function
accesses simulator state, changes queued I/O, or writes CIR/PIR.

Selected on development seed 20260906; held-out improvement is not guaranteed.
Both formulas are fluid heuristics, not exact deadline/no-stall tests.
"""

from dataclasses import dataclass

from shared_path_common import IO_GIB, solo_io_ms


@dataclass(frozen=True)
class KnownNPU:
    """Client-known work of already-arrived active and admission-queued requests.

    Compute includes the current compute remainder and later known layers.
    I/O includes unsubmitted known layers and submitted work not yet HBM-ACKed,
    grouped by SSU. This is not a snapshot of the SSD's actual pending queue.
    """

    npu_id: int
    remaining_compute_ms: float
    unfinished_io_by_ssu: tuple[int, ...]
    unfinished_request_count: int = 0


@dataclass(frozen=True)
class AssignmentChoice:
    npu_id: int
    scores_ms: tuple[float, ...]
    details: tuple[dict, ...]


def _ssu_counts(counts):
    """Accept per-SSU totals or per-SSU rows of cached Path counts."""
    return tuple(row if isinstance(row, (int, float)) else sum(row) for row in counts)


def choose_npu(now_ms, backlogs, incoming_layer_io_by_ssu, compute_ms, n_layers=8,
               snapshot_counts_by_ssu=(), disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Minimize now + max(C_j, 1000*sum(V_j)/L, max_s(1000*A_s*V_js/B)).

    C_j and V_js are candidate j's remaining compute ms and per-SSU GiB after
    adding all n_layers of the incoming request. B is each disk's GiB/s, L is
    the NPU's single receive-link GiB/s. A_s counts NPUs with known unfinished
    work on SSD s after this candidate assignment, not NPUs currently issuing.

    The disk term estimates equal sharing; actual static category CIR/PIR
    arbitration does not promise it. max assumes fluid I/O/compute overlap,
    not an exact multi-layer schedule. No SSD order, active service remainder,
    future arrival or per-layer deadline is inferred from KnownNPU.

    Cached SSU counts are diagnostics here, not a candidate-independent floor
    masking load differences. They enter the separate JIT formula below.
    Scores/details follow backlogs' order; returned NPU IDs need not be indices.
    """
    backlogs = tuple(backlogs)
    incoming = tuple(incoming_layer_io_by_ssu)
    total_new = tuple(n_layers * count for count in incoming)
    queue_counts = _ssu_counts(snapshot_counts_by_ssu)
    scores, details = [], []
    for candidate, state in enumerate(backlogs):
        work = [row.unfinished_io_by_ssu for row in backlogs]
        compute = state.remaining_compute_ms + n_layers * compute_ms
        work[candidate] = tuple(old + new for old, new in zip(work[candidate], total_new))
        own_work = work[candidate]
        active = tuple(sum(row[s] > 0 for row in work) for s in range(len(incoming)))
        disk_ms = tuple(1000 * IO_GIB * count * lanes / disk_bw_gib_s
                        for count, lanes in zip(own_work, active))
        link_ms = 1000 * IO_GIB * sum(own_work) / link_bw_gib_s
        predicted = max(compute, link_ms, max(disk_ms, default=0.0))
        scores.append(now_ms + predicted)
        details.append({"candidate_npu_id": state.npu_id,
                        "mode": "fair_pipeline", "eligible": True,
                        "projected_compute_ms": compute,
                        "isolated_io_ms": solo_io_ms(own_work, disk_bw_gib_s, link_bw_gib_s),
                        "known_work_lanes_by_ssu": active,
                        "fair_disk_ms_by_ssu": disk_ms,
                        "receive_link_ms": link_ms,
                        "sampled_pending_counts_by_ssu": queue_counts,
                        "is_completion_guarantee": False})
    chosen = min(range(len(backlogs)), key=lambda i: (
        scores[i], backlogs[i].unfinished_request_count,
        backlogs[i].remaining_compute_ms, backlogs[i].npu_id))
    return AssignmentChoice(backlogs[chosen].npu_id, tuple(scores), tuple(details))


def prefetch_release_ms(now_ms, deadline_ms, layer_io_by_ssu,
                        pending_counts_by_ssu, guard_ms=10.0,
                        disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Return a common release time for an entirely unsubmitted layer coflow.

    estimated_ms = 1000*b*max(max_s((Q_s+K_s)/B), sum_s(K_s)/L)
    release_ms = max(now_ms, deadline_ms-estimated_ms-guard_ms).

    b is one 176 KiB I/O in GiB. K is this layer's per-SSU I/O count. Q is
    periodically cached SSD Path counts (rows or per-SSU totals), excluding K.
    Each disk works in parallel, but all returns share the NPU's one link.
    deadline_ms is the caller's currently known layer-data-needed timestamp.

    The 10 ms guard makes release earlier, not later. It is a development-set
    parameter, not a delay limit or guarantee. Stale Q, future arrivals, static
    CIR, release order and receive backlog can invalidate this estimate. The
    caller only defers unsubmitted I/O; it must not move queued I/O or compute.
    Empty coflows return now_ms because there is no read to postpone.
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
    print(choose_npu(5, lanes, (64, 64), 1, snapshot_counts_by_ssu=((8, 4), (3, 2))))
    print("prefetch release ms:", prefetch_release_ms(5, 30, (64, 64), ((8, 4), (3, 2))))
