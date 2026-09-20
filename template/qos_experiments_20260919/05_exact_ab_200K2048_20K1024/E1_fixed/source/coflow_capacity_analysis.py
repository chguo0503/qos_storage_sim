"""Independent necessary-capacity checks for a fixed activated-coflow state.

One I/O is 176 KiB. This is an offline arithmetic audit, not a simulator or a
packetized exact predictor. An overload proves that the supplied remaining
work cannot all meet the supplied fixed releases/deadlines. No overload does
NOT prove feasibility. A certificate after a particular scheduling history
does not prove that every earlier phase, placement or client policy fails.

SSD remaining counts must describe bytes that still need SSD service; link
remaining counts must separately describe bytes not yet transferred to HBM.
Active partial I/O may use fractional I/O units. If an active remainder is
unknown, omit that command to obtain a valid (weaker) lower-bound witness;
counting its whole original size can falsely assert overload. Do not pass a
host planned/sent-unacknowledged ledger as exact SSD remaining work.
"""

from dataclasses import dataclass

from shared_path_common import IO_GIB


@dataclass(frozen=True)
class ActivatedCoflow:
    request_id: int
    layer: int
    npu_id: int
    release_ms: float
    deadline_ms: float
    remaining_ssd_io: tuple[float, ...]
    remaining_link_io: float


@dataclass(frozen=True)
class CapacityOverload:
    resource: str
    resource_id: int
    start_ms: float
    end_ms: float
    required_gib: float
    capacity_gib: float
    excess_gib: float
    required_service_ms: float
    coflow_ids: tuple[tuple[int, int], ...]
    kind: str


def solo_read_lower_bound_ms(io_by_ssu, disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Fluid isolated-read lower bound, not actual two-stage completion time.

    All I/O must traverse its fixed SSD and the NPU's one receive link.
    Packet pipeline startup/drain, queues, CIR and client issue delay can only
    add time. A value greater than a given compute window disproves hiding a
    wholly unread layer in that window, under that activation/deadline pair.
    """
    counts = tuple(io_by_ssu)
    return 1000 * IO_GIB * max(max(counts, default=0) / disk_bw_gib_s,
                              sum(counts) / link_bw_gib_s)


def capacity_overloads(coflows, now_ms, *, disk_bw_gib_s=40.0,
                       link_bw_gib_s=50.0, release_intervals=True,
                       tolerance_gib=1e-12):
    """Return fixed-state interval overload witnesses, not a feasible schedule.

    For every deadline D, the prefix [now,D] counts remaining work due by D.
    If release_intervals is True, also check [a,D] at each known effective
    release a: only coflows with release >= a and deadline <= D must execute
    wholly inside that interval. Past releases become now because these are
    *remaining* bytes. release_ms is the fixed earliest allowed release of
    this work, not an unknown future request; all coflows must already be
    activated/known to the caller. Later-layer work must not be charged to the
    current layer's deadline.

    Each SSD has capacity B*(D-a)/1000 GiB; each NPU's independent receive link
    has capacity L*(D-a)/1000. The relaxed link release is the coflow release,
    even though real packets first need SSD service. Thus these checks ignore
    precedence, non-preemption, QoS and packet serialization and are necessary
    but not sufficient. already_due_unfinished means a deadline has already
    expired; it is not a forward-looking physical-overload discovery. Expired
    work is omitted from later future-deadline checks so an already missed
    deadline is not reused as circular proof of a new unavoidable miss.

    Inputs are nonnegative actual remaining amounts or lower bounds on them;
    resource vectors have one entry per SSU. The small GiB tolerance only
    suppresses floating-point equality noise. Returned witnesses can overlap.
    """
    coflows = tuple(coflows)
    if not coflows:
        return ()
    deadlines = sorted({max(now_ms, row.deadline_ms) for row in coflows})
    starts = {now_ms}
    if release_intervals:
        # A release after its own deadline is itself impossible; the zero-
        # length interval at that deadline exposes this fixed-input conflict.
        starts.update(max(now_ms, min(row.release_ms, row.deadline_ms)) for row in coflows)
    witnesses = []
    for start in sorted(starts):
        for end in deadlines:
            if end < start:
                continue
            selected = tuple(row for row in coflows
                             if max(now_ms, row.release_ms) >= start and row.deadline_ms <= end
                             and (end == now_ms or row.deadline_ms > now_ms))
            resources = [("ssu", s, disk_bw_gib_s,
                          [(row, row.remaining_ssd_io[s]) for row in selected])
                         for s in range(len(coflows[0].remaining_ssd_io))]
            resources.extend(("npu_link", n, link_bw_gib_s,
                              [(row, row.remaining_link_io) for row in selected if row.npu_id == n])
                             for n in sorted({row.npu_id for row in selected}))
            for resource, resource_id, bandwidth, work in resources:
                positive = [(row, count) for row, count in work if count > 0]
                required = IO_GIB * sum(count for _, count in positive)
                capacity = bandwidth * (end - start) / 1000
                if required <= capacity + tolerance_gib:
                    continue
                kind = ("already_due_unfinished" if end <= now_ms else
                        "ready_prefix" if start == now_ms else "release_interval")
                witnesses.append(CapacityOverload(
                    resource, resource_id, start, end, required, capacity,
                    required - capacity, 1000 * required / bandwidth,
                    tuple((row.request_id, row.layer) for row, _ in positive), kind))
    return tuple(witnesses)


if __name__ == "__main__":
    reads = (ActivatedCoflow(1, 1, 0, 0, 1, (150,), 150),
             ActivatedCoflow(2, 1, 1, 0, 1, (150,), 150))
    for witness in capacity_overloads(reads, 0):
        print(witness)
