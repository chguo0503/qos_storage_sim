"""Ideal causal reordering of pending commands INSIDE one selected QoS Path.

The caller retains the native cross-Path CIR arbitration and non-preemptive
active command. Only that selected Path's already-arrived PENDING commands are
passed here. No command changes Path/SSU or reads an unknown future request.
The SSD is allowed immediate local queue/metadata visibility, unlike the
client's periodic 5-ms telemetry; that extra permission must be reported.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PendingIO:
    io_id: int
    npu_id: int
    request_id: int
    layer: int
    path_id: int
    ssu_id: int
    arrival_ms: float
    deadline_ms: float
    layer_io_count: int = 1


def reorder_pending(ios, now_ms):
    """Return IDs in deadline order; break ties toward smaller read components.

    'deadline' is the data-ready deadline carried by the released I/O, not a
    forecast of a future layer or the entire request's TTFT. layer_io_count is
    the immutable total of this layer's read component on this SSU, carried at
    submission (flow.layer_work_gb / IO_GIB). It is NOT a live remaining count
    or knowledge of other SSUs. This feasible ideal capability comparator does
    not claim globally optimal scheduling or exact coflow-barrier completion.
    """
    ios = tuple(ios)
    if len({(io.ssu_id, io.path_id) for io in ios}) > 1:
        raise ValueError("only one SSU's one Path pending queue may be reordered")
    if any(io.arrival_ms > now_ms for io in ios):
        raise ValueError("future commands cannot enter the pending reorder input")
    return tuple(io.io_id for io in sorted(ios, key=lambda io: (
        io.deadline_ms, io.layer_io_count, io.arrival_ms, io.io_id)))


if __name__ == "__main__":
    commands = (PendingIO(10, 0, 1, 2, 7, 0, 0, 10),
                PendingIO(11, 1, 2, 1, 7, 0, .1, 1))
    print(reorder_pending(commands, .5))
