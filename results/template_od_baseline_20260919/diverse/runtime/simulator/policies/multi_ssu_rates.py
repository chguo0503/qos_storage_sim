"""Pure multi-SSU coflow CIR proposals from released client-known work.

The serial-stall objective is a heuristic model, not an event simulation or
a guarantee about measured NPU utilization. The adapter applies decisions.
"""

from dataclasses import dataclass
from simulator.policies.allocation import allocate_coflow_grants


IO_GIB = 176 * 1024 / 2**30
_EPS = 1e-12


@dataclass(frozen=True)
class PendingCoflow:
    npu_id: int
    request_id: int
    layer: int
    remaining_gib_by_ssu: tuple[float, ...]
    deadline_ms: float
    compute_ms: float

    @property
    def remaining_blocks(self):
        return tuple(round(value / IO_GIB) for value in self.remaining_gib_by_ssu)


def coflow_solo_ms(work_gib_by_ssu, ssd_cap_gib_s=40.0, npu_cap_gib_s=50.0):
    """Fluid lower bound, excluding command startup/tail and queued work."""
    return 1000 * max(max(work_gib_by_ssu, default=0.0) / ssd_cap_gib_s,
                      sum(work_gib_by_ssu) / npu_cap_gib_s)


def serial_stall_objective(jobs, now_ms, *, ssd_cap_gib_s=40.0,
                           npu_cap_gib_s=50.0):
    """Sum future stall in the ordering heuristic's exclusive-bottleneck model.

    Every coflow occupies a *hypothetical shared* bottleneck for its fluid solo
    lower bound T_i. The cumulative finish f_i incurs max(0, f_i-d_i), where
    d_i=max(0, deadline_i-now). This deliberately excludes already accrued
    stall. It is exact for the stated serial model, NOT a prediction of the
    native multi-SSU scheduler or of subsequent layer releases.
    """
    prefix, total = 0.0, 0.0
    for job in jobs:
        prefix += coflow_solo_ms(job.remaining_gib_by_ssu,
                                ssd_cap_gib_s, npu_cap_gib_s)
        total += max(0.0, prefix - max(0.0, job.deadline_ms - now_ms))
    return total


def stall_interchange_order(jobs, now_ms, *, ssd_cap_gib_s=40.0,
                            npu_cap_gib_s=50.0):
    """Improve an existing EDF order with at most two adjacent-swap passes.

    A swap uses the FULL elapsed prefix, not an independent t=0 for each pair.
    It must strictly decrease that pair's hinge-loss sum. Earlier jobs do not
    change, and the pair's total duration is unchanged, so all later starts
    also remain unchanged: the complete serial objective decreases at each
    accepted swap. Two passes bound work to <=2*(N-1) pair comparisons.

    Multi-SSU parallelism and CIR behavior violate this serial abstraction;
    monotonicity is only for the surrogate objective, not measured utilization.
    Short-coflow bias can worsen a large request's tail latency. Two passes
    limit planning work, not starvation; no fairness guarantee is asserted.
    """
    ordered = list(jobs)
    durations = {job.npu_id: coflow_solo_ms(job.remaining_gib_by_ssu,
                                         ssd_cap_gib_s, npu_cap_gib_s)
                 for job in ordered}
    for _ in range(2):
        prefix, changed = 0.0, False
        for index in range(len(ordered) - 1):
            left, right = ordered[index:index + 2]
            ti, tj = durations[left.npu_id], durations[right.npu_id]
            di, dj = max(0.0, left.deadline_ms - now_ms), max(0.0, right.deadline_ms - now_ms)
            forward = max(0.0, prefix + ti - di) + max(0.0, prefix + ti + tj - dj)
            reverse = max(0.0, prefix + tj - dj) + max(0.0, prefix + tj + ti - di)
            if reverse < forward - _EPS:
                ordered[index:index + 2] = right, left
                changed = True
            prefix += durations[ordered[index].npu_id]
        if not changed:
            break
    return tuple(ordered)


def choose_coflow_rates(jobs, now_ms, *, num_npu, num_ssu, mode="deadline",
                        ssd_cap_gib_s=40.0, npu_cap_gib_s=50.0):
    """Return r[NPU][SSU], with each column <= B and each row <= L.

    ``demand`` is a synchronized max-min V/C comparator, not EDF. ``deadline``
    greedily fills each whole remaining-work vector in earliest-deadline order;
    all its SSU components receive the same progress-per-second. Spare resources
    can serve later coflows. ``deadline_reserve`` first tries W/(D-now), then
    fills spare capacity in the same order. ``least_slack`` substitutes D-Tsolo.
    ``stall_interchange`` makes at most two locally improving adjacent-swap
    passes over EDF using an exclusive-bottleneck total-stall surrogate.
    None of these heuristics is an optimum of long-horizon NPU utilization.

    One already-released layer per NPU is supported, as in batch_size=1 with
    one-layer lookahead.  Unreleased future layers must not be passed as jobs.
    """
    if not 1 <= num_npu <= 128:
        raise ValueError("dedicated Paths support 1 through 128 NPUs")
    if mode not in ("demand", "deadline", "deadline_reserve", "least_slack",
                    "stall_interchange"):
        raise ValueError("unknown coflow planning mode")
    jobs = tuple(job for job in jobs if sum(job.remaining_gib_by_ssu) > _EPS)
    if len({job.npu_id for job in jobs}) != len(jobs):
        raise ValueError("pass only the currently released coflow per NPU")
    rates = [[0.0] * num_ssu for _ in range(num_npu)]
    if mode == "demand":
        for job in jobs:
            rates[job.npu_id] = [value * 1000 / job.compute_ms
                                 for value in job.remaining_gib_by_ssu]
        return allocate_coflow_grants(rates, target_ratios=1.0,
                                     ssd_caps=ssd_cap_gib_s,
                                     npu_caps=npu_cap_gib_s)

    def priority(job):
        solo = coflow_solo_ms(job.remaining_gib_by_ssu,
                             ssd_cap_gib_s, npu_cap_gib_s)
        return (job.deadline_ms - (solo if mode == "least_slack" else 0),
                solo, job.npu_id)

    ordered = sorted(jobs, key=priority)
    if mode == "stall_interchange":
        ordered = stall_interchange_order(ordered, now_ms,
                                         ssd_cap_gib_s=ssd_cap_gib_s,
                                         npu_cap_gib_s=npu_cap_gib_s)
    remaining_ssd = [ssd_cap_gib_s] * num_ssu
    remaining_link = [npu_cap_gib_s] * num_npu

    def fill(job, progress_limit=float("inf")):
        work = job.remaining_gib_by_ssu
        # One shared progress scalar avoids serving an easy SSU far ahead of
        # the bottleneck SSU of this same layer. It may deliberately leave an
        # irrelevant SSU idle if every remaining coflow is bottlenecked elsewhere.
        progress = min(progress_limit,
                       remaining_link[job.npu_id] / sum(work),
                       *(remaining_ssd[s] / value for s, value in enumerate(work)
                         if value > _EPS))
        for s, value in enumerate(work):
            grant = max(0.0, progress) * value
            rates[job.npu_id][s] += grant
            remaining_ssd[s] = max(0.0, remaining_ssd[s] - grant)
            remaining_link[job.npu_id] = max(0.0,
                                           remaining_link[job.npu_id] - grant)

    if mode == "deadline_reserve":
        for job in ordered:
            budget_ms = job.deadline_ms - now_ms
            fill(job, 1000 / budget_ms if budget_ms > 0 else float("inf"))
    for job in ordered:
        fill(job)
    return tuple(tuple(row) for row in rates)


def main():
    jobs = (PendingCoflow(0, 1, 0, (.6, .1), 5, 10),
            PendingCoflow(1, 2, 0, (.1, .6), 8, 10))
    for mode in ("demand", "deadline", "deadline_reserve", "least_slack", "stall_interchange"):
        rates = choose_coflow_rates(jobs, 0, num_npu=2, num_ssu=2, mode=mode)
        assert all(sum(row) <= 50 + 1e-9 for row in rates)
        assert all(sum(row[s] for row in rates) <= 40 + 1e-9 for s in range(2))
        assert all(v >= 0 for row in rates for v in row)
    ordered = stall_interchange_order(jobs, 0)
    assert serial_stall_objective(ordered, 0) <= serial_stall_objective(jobs, 0) + 1e-9
    assert choose_coflow_rates((), 0, num_npu=2, num_ssu=2) == ((0.0, 0.0), (0.0, 0.0))
    print("multi_ssu_rates: PASS (all five rate modes satisfy capacities; serial surrogate does not increase)")


if __name__ == "__main__":
    main()
