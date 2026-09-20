"""Shared-pool Once with a global client reservation/completion ledger.

The scope is explicit: ALL traffic to these SSUs is managed by this client
system. client_pending_counts includes every planned-but-not-submitted I/O and
every submitted I/O not yet acknowledged at HBM. It therefore upper-bounds
current SSD outstanding work without reading an SSD FIFO or active remainder.

The caller must atomically reserve the returned Path IDs before planning the
next client, and subtract only actual HBM acknowledgements (or explicitly
cancelled unsubmitted reservations). This function never mutates that ledger.
An old device count is not max'ed back in after its work has completed: under
the all-managed assumption, doing that would retain fictitious old backlog.
"""

from simulator.policies.common import pressure_from_counts
from simulator.policies.once import once_path_ids


def new_once_path_ids(io_count, snapshot, client_pending_counts, allowed_path_ids,
                      qos, start_offset=0, disk_bw_gib_s=40.0):
    if len(client_pending_counts) != len(snapshot.counts):
        raise ValueError("the global ledger must cover the same SSU Path table")
    if any(count < 0 for count in client_pending_counts):
        raise ValueError("a client reservation/completion count cannot be negative")
    planned_view = pressure_from_counts(client_pending_counts, qos)
    return once_path_ids(io_count, planned_view, allowed_path_ids, qos,
                         start_offset=start_offset, disk_bw_gib_s=disk_bw_gib_s)


def main():
    from simulator.policies.policy_logic import category_path_ids
    from simulator.policies.profiles import FINAL_STATIC
    config = FINAL_STATIC.hardware_config()
    snapshot = pressure_from_counts((0,) * 256, config)
    ledger = [0] * 256
    candidates = category_path_ids("SS", config)
    first = new_once_path_ids(8, snapshot, ledger, candidates, config)
    assert ledger == [0] * 256  # The caller, not the pure decision, reserves work.
    for path in first:
        ledger[path] += 1
    second = new_once_path_ids(8, snapshot, ledger, candidates, config)
    assert len(second) == 8 and set(second).issubset(candidates)
    assert first != second
    assert snapshot.counts == (0,) * 256
    assert sum(ledger) == 8
    print("new_once: PASS (client reservations affect the next decision; stale snapshot unchanged)")


if __name__ == "__main__":
    main()
