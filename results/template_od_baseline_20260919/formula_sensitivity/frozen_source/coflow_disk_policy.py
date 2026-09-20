"""Pure choices inside the native WFQ minimum-finish equivalence set.

This changes a legal tie-break, not the native service rates, CIR table or
virtual-finish accounting. It is not unrestricted cross-Path EDF.
"""

from dataclasses import dataclass

from shared_path_common import IO_GIB


@dataclass(frozen=True)
class PathCandidate:
    path_id: int
    finish_tag: float
    priority: tuple


def local_priority(flow, now_ms):
    """Only metadata of an already-enqueued command, not another SSD's state.

    Component size is immutable work on this SSD, not live coflow remainder.
    now_ms is accepted for a common callback interface; absolute deadline
    order itself does not change merely because the clock advances.
    """
    return (flow.deadline_time, round(flow.layer_work_gb / IO_GIB), flow.enqueue_time)


def choose_path(candidates, rr_cursor, path_count=256, epsilon=1e-12):
    """Choose by priority only among native-equivalent minimum finish tags."""
    candidates = tuple(candidates)
    minimum = min(candidate.finish_tag for candidate in candidates)
    return min((candidate for candidate in candidates
                if candidate.finish_tag <= minimum + epsilon), key=lambda candidate: (
        candidate.priority, (candidate.path_id - rr_cursor) % path_count,
        candidate.path_id, candidate.finish_tag))


if __name__ == "__main__":
    print(choose_path((PathCandidate(0, 1.0, (20.0,)),
                       PathCandidate(32, 1.0, (2.0,)),
                       PathCandidate(64, 2.0, (0.0,))), 0))
