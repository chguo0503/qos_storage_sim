"""A small joint-coflow priority computed from client-provided progress.

The disk adapter does not collect these inputs. Its caller supplies only an
already-activated coflow's known unfinished work, from the agreed 5-ms SSU
telemetry and/or host HBM acknowledgements. No data migration is involved.
"""

from shared_path_common import IO_GIB


def coflow_priority(deadline_ms, unfinished_io_by_ssu, now_ms,
                    disk_bw_gib_s=40.0, link_bw_gib_s=50.0, rule="least_slack"):
    """Joint heuristic; identical key for every SSD component.

    unfinished_io_by_ssu is the already-activated coflow's client-known count,
    not all future layers. If it includes work already in the receive link,
    the per-SSD term is an estimate, not actual SSD service still required.
    Likewise max(disk, link) is not an exact packetized completion prediction.
    least_slack preserves the first prototype. urgent_short instead matches
    the client's objective: urgent coflows sort by smaller remaining service;
    nonurgent coflows sort by earlier deadline. Neither rule guarantees the
    least mean stall, freedom from starvation, or improvement on every input.
    """
    counts = tuple(unfinished_io_by_ssu)
    remaining_ms = 1000 * IO_GIB * max(
        max(counts, default=0) / disk_bw_gib_s,
        sum(counts) / link_bw_gib_s)
    if rule == "least_slack":
        return (deadline_ms - now_ms - remaining_ms, remaining_ms, deadline_ms)
    if rule == "urgent_short":
        urgent = deadline_ms - now_ms <= remaining_ms
        return (not urgent, remaining_ms if urgent else deadline_ms, deadline_ms)
    raise ValueError("coflow rule must be least_slack or urgent_short")


if __name__ == "__main__":
    print(coflow_priority(20, (1000, 200), 5))
