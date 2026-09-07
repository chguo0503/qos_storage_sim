"""Pure global client dispatcher: one KV I/O = 176 KiB.

All counters are client-owned plans, actual submissions and HBM ACKs. They
are not instantaneous SSD queue measurements. Device telemetry stays at 5 ms.
Only activated, wholly or partially unsubmitted layers may enter this planner.
"""

from dataclasses import dataclass
from shared_path_common import IO_GIB, solo_io_ms


@dataclass(frozen=True)
class ClientRead:
    request_id: int
    layer: int
    npu_id: int
    deadline_ms: float
    unsent_by_ssu: tuple[int, ...]
    remaining_by_ssu: tuple[int, ...]


def choose_npu(compute_backlog_ms, original_npu_id, request_counts=()):
    """Choose a lane by already-arrived remaining compute, not future traffic.

    Incoming compute is identical for all candidate lanes, so it cancels.
    Compute-only placement deliberately avoids treating a fair-share fluid
    estimate of all future layers as an exact admission finish time.
    """
    counts = request_counts or (0,) * len(compute_backlog_ms)
    return min(range(len(compute_backlog_ms)), key=lambda n: (
        compute_backlog_ms[n], counts[n], n != original_npu_id, n))


def priority(read, now_ms, disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Urgent coflows: shorter remaining work first; others: earlier deadline.

    Urgent means isolated remaining service no longer fits the current slack.
    This is a heuristic, not a no-stall certificate or optimality theorem.
    """
    service = solo_io_ms(read.remaining_by_ssu, disk_bw_gib_s, link_bw_gib_s)
    urgent = read.deadline_ms - now_ms <= service
    return (not urgent, service if urgent else read.deadline_ms,
            read.deadline_ms, read.request_id, read.layer)


def plan_global_batch(reads, sent_unacked_by_ssu, sent_unacked_by_npu, now_ms,
                      *, queue_window_ms=0.25, max_commands=64,
                      disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Return (request_id, layer, npu_id, ssu_id) grants for the next batch.

    Each SSU and NPU has a bounded client outstanding window B*window_ms.
    Reservations are local to this plan: the caller must execute grants in
    order or discard the unissued suffix before planning again. Actual sends
    consume credits; HBM ACKs return them. A grant is one equal-sized I/O.
    No data is migrated and no queued SSD command is moved here.

    An empty result means wait for an ACK or a newly activated layer. This
    wait is scheduling, not a new SSU read. Idle credits are work-conserving:
    even a distant-deadline coflow may use them when no urgent work needs them.
    """
    disk_limit = max(1, int(disk_bw_gib_s * queue_window_ms / (1000 * IO_GIB)))
    link_limit = max(1, int(link_bw_gib_s * queue_window_ms / (1000 * IO_GIB)))
    disk_room = [max(0, disk_limit - q) for q in sent_unacked_by_ssu]
    link_room = [max(0, link_limit - q) for q in sent_unacked_by_npu]
    if not any(disk_room) or not any(link_room):
        return ()
    grants = []
    for read in sorted(reads, key=lambda r: priority(r, now_ms, disk_bw_gib_s, link_bw_gib_s)):
        remaining = list(read.unsent_by_ssu)
        while len(grants) < max_commands and link_room[read.npu_id] > 0:
            available = [s for s, count in enumerate(remaining) if count and disk_room[s] > 0]
            if not available:
                break
            # Keep components moving together, favouring the larger remainder.
            s = min(available, key=lambda s: (-remaining[s], s))
            grants.append((read.request_id, read.layer, read.npu_id, s))
            remaining[s] -= 1
            disk_room[s] -= 1
            link_room[read.npu_id] -= 1
        if len(grants) == max_commands:
            break
    return tuple(grants)


if __name__ == "__main__":
    reads = (ClientRead(1, 1, 0, 20, (5, 5), (5, 5)),
             ClientRead(2, 1, 1, .02, (2, 2), (2, 2)))
    print("assigned NPU:", choose_npu((10, 2, 8, 5), 0))
    print("global I/O grants:", plan_global_batch(reads, (0, 0), (0, 0, 0, 0), 0,
                                                max_commands=6))
