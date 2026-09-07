"""Small immutable values for the shared-Path policy experiments.

One I/O is one 176 KiB GLM KV block. No event queue, SSD service remainder,
future arrival, or simulator context is an input to this module.
"""

from dataclasses import dataclass

IO_BYTES = 176 * 1024
IO_GIB = IO_BYTES / 2**30


@dataclass(frozen=True)
class PathSnapshot:
    counts: tuple[int, ...]
    group_io_counts: tuple[int, ...]
    active_paths_per_group: tuple[int, ...]
    active_path_weights: tuple[float, ...]
    active_group_weight_sum: float
    active_cir_sum: float


def pressure_from_counts(counts, qos):
    """Construct the existing routing engine's view from a count-only ledger.

    With unsubmitted reservations included, 'active' means planned/unfinished
    client work, not proof that the SSD currently has that Path backlogged.
    It is a conservative planning view, not newly sampled device telemetry.
    """
    counts = tuple(counts)
    width = qos.paths_per_group
    group_counts, active_paths, active_weights = [], [], []
    for group in range(len(qos.group_weights)):
        indices = range(group * width, (group + 1) * width)
        group_counts.append(sum(counts[p] for p in indices))
        active_paths.append(sum(counts[p] > 0 for p in indices))
        active_weights.append(sum(qos.path_weights[p] for p in indices if counts[p] > 0))
    return PathSnapshot(
        counts, tuple(group_counts), tuple(active_paths), tuple(active_weights),
        sum(weight for weight, active in zip(qos.group_weights, active_paths) if active),
        sum(min(cir, pir) for count, cir, pir in zip(counts, qos.path_cirs, qos.path_pirs)
            if count > 0),
    )


def solo_io_ms(io_by_ssu, disk_bw_gib_s=40.0, link_bw_gib_s=50.0):
    """Fluid isolated-service lower bound, not an end-to-end prediction."""
    return 1000 * IO_GIB * max(max(io_by_ssu, default=0) / disk_bw_gib_s,
                               sum(io_by_ssu) / link_bw_gib_s)
