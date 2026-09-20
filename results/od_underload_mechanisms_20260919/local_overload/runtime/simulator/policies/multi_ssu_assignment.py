"""Pure causal assignment from known per-NPU compute and per-SSD work.

No manifest loader, event queue, future arrival, or context is required.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import inf


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


def main():
    backlogs = (MultiNPUBacklog(10, 100, (100, 100), 1),
                MultiNPUBacklog(20, 1, (10, 10), 1))
    for mode in ("compute", "fluid"):
        decision = choose_npu(5, backlogs, (20, 20), 10,
            disk_bandwidths_gib_s=(40, 40), policy=mode)
        assert decision.npu_id == 20
        assert len(decision.selection_scores_ms_by_npu) == 2
    assert _rate(0, 0) == 0
    print("multi_ssu_assignment: PASS (compute/fluid choices retain actual NPU IDs)")


if __name__ == "__main__":
    main()
