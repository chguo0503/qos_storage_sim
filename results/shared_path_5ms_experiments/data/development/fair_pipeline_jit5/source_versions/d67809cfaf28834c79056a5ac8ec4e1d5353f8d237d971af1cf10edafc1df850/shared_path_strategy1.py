"""Causal arrival-time placement for the shared-Path client policy.

This does not reuse a dedicated-Path CIR controller. Reads still use New Once
and the static Path pool. Only the just-arrived request can be assigned; an
existing request and its SSD placement never move. The score is a fluid
heuristic, not a deadline/no-stall guarantee or a forecast of unknown arrivals.
"""

from dataclasses import dataclass

from shared_path_common import IO_GIB, solo_io_ms


@dataclass(frozen=True)
class KnownNPU:
    npu_id: int
    remaining_compute_ms: float
    unfinished_io_by_ssu: tuple[int, ...]
    unfinished_request_count: int = 0


@dataclass(frozen=True)
class AssignmentChoice:
    npu_id: int
    scores_ms: tuple[float, ...]
    details: tuple[dict, ...]


def choose_npu(now_ms, backlogs, incoming_layer_io_by_ssu, compute_ms, n_layers=8,
               snapshot_counts_by_ssu=(), disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Balance own pipeline finish and the simultaneous coflow profile mix.

    For each candidate, combine its known remaining request work with the new
    request. Each NPU's denominator max(compute, isolated I/O service) gives a
    physically capped fluid work rate. Sum those rates per SSD to estimate a
    fleet contention factor, then apply it to the candidate's pipeline time.

    The periodically sampled *whole-SSU* queue provides a separate congestion
    floor. It is combined by max, not added to the same known backlog again.
    Because it ignores QoS bypass, snapshot age, future submissions and release
    order, this floor is just a score component, NOT a mathematical time bound.
    Rows in scores/details follow backlogs' order; NPU IDs need not be indices.
    """
    backlogs = tuple(backlogs)
    incoming = tuple(incoming_layer_io_by_ssu)
    total_new = tuple(n_layers * count for count in incoming)
    ssus = len(incoming)
    queue_counts = tuple(sum(row) for row in snapshot_counts_by_ssu)
    queue_hint_ms = max((1000 * IO_GIB * (count + incoming[s]) / disk_bw_gib_s
                         for s, count in enumerate(queue_counts)), default=0.0)
    scores, details = [], []
    for candidate, state in enumerate(backlogs):
        compute = [row.remaining_compute_ms for row in backlogs]
        work = [row.unfinished_io_by_ssu for row in backlogs]
        compute[candidate] += n_layers * compute_ms
        work[candidate] = tuple(old + new for old, new in zip(work[candidate], total_new))
        pipeline = [max(c, solo_io_ms(w, disk_bw_gib_s, link_bw_gib_s))
                    for c, w in zip(compute, work)]
        demand_by_ssu = tuple(sum(1000 * IO_GIB * row[s] / duration
                                  if duration else 0.0
                                  for row, duration in zip(work, pipeline))
                              for s in range(ssus))
        contention = max(1.0, max(demand_by_ssu, default=0.0) / disk_bw_gib_s)
        predicted = max(pipeline[candidate] * contention, queue_hint_ms + compute_ms)
        scores.append(now_ms + predicted)
        details.append({"candidate_npu_id": state.npu_id,
                        "own_pipeline_ms": pipeline[candidate],
                        "projected_demand_gib_s_by_ssu": demand_by_ssu,
                        "disk_contention_factor": contention,
                        "sampled_queue_hint_ms": queue_hint_ms,
                        "is_completion_guarantee": False})
    chosen = min(range(len(backlogs)), key=lambda i: (
        scores[i], backlogs[i].unfinished_request_count,
        backlogs[i].remaining_compute_ms, backlogs[i].npu_id))
    return AssignmentChoice(backlogs[chosen].npu_id, tuple(scores), tuple(details))


if __name__ == "__main__":
    lanes = (KnownNPU(0, 10, (100, 100), 1), KnownNPU(1, 3, (10, 10), 1))
    print(choose_npu(5, lanes, (64, 64), 1, snapshot_counts_by_ssu=((8, 4), (3, 2))))
