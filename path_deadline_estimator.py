"""Pure deadline estimates for a frozen, work-conserving FCFS SSD path.

This module deliberately separates an inexpensive snapshot estimate from the
authoritative discrete-event simulator.  The estimate is exact for the current
simulator's *SSD stage* only when all work named by the caller is already in a
known FCFS order on one Path and no command can be inserted ahead of it.

Counts alone are not enough unless every command has the same known size.  The
aggregate API therefore accepts bytes first and uses command counts only when
an IOPS cap or a non-overlapped per-command overhead is requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


BYTES_PER_GIB = 2**30


@dataclass(frozen=True)
class PathDeadlineEstimate:
    """Result of one FCFS SSD deadline estimate.

    Positive ``slack_ms`` means the SSD-stage completion is early.  Negative
    ``slack_ms`` means it misses the supplied budget by ``shortfall_ms``.
    """

    meets_deadline: bool
    time_budget_ms: float
    estimated_ssd_completion_ms: float
    slack_ms: float
    early_by_ms: float
    shortfall_ms: float
    queue_ahead_ms: float
    target_incremental_service_ms: float
    path_bytes_ahead: int
    required_bytes: int
    path_io_ahead: int | None
    required_io_count: int | None
    bandwidth_gib_s: float
    iops_limit: float | None
    fixed_command_latency_us: float
    bandwidth_bound_ms: float
    iops_bound_ms: float | None
    fixed_latency_ms: float
    model_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _nonnegative_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _cumulative_service_ms(
    *,
    byte_count: int,
    io_count: int | None,
    bandwidth_bytes_s: float,
    iops_limit: float | None,
    fixed_command_latency_us: float,
) -> tuple[float, float, float | None, float]:
    bandwidth_ms = byte_count / bandwidth_bytes_s * 1000.0
    iops_ms = None if iops_limit is None else io_count / iops_limit * 1000.0
    throughput_ms = bandwidth_ms if iops_ms is None else max(bandwidth_ms, iops_ms)
    fixed_latency_ms = (
        0.0
        if io_count is None
        else io_count * fixed_command_latency_us / 1000.0
    )
    return throughput_ms + fixed_latency_ms, bandwidth_ms, iops_ms, fixed_latency_ms


def estimate_fcfs_path_deadline(
    *,
    path_bytes_ahead: int,
    required_bytes: int,
    time_budget_ms: float,
    bandwidth_gib_s: float = 40.0,
    path_io_ahead: int | None = None,
    required_io_count: int | None = None,
    iops_limit: float | None = None,
    fixed_command_latency_us: float = 0.0,
) -> PathDeadlineEstimate:
    """Estimate whether target I/O at the tail of one frozen FCFS Path is ready.

    Parameters
    ----------
    path_bytes_ahead:
        Remaining bytes in the active SSD command plus queued commands that
        are guaranteed to stay ahead of the target I/O.
    required_bytes:
        Total target-layer bytes that must finish at the SSD stage.
    time_budget_ms:
        Time from the snapshot to the target deadline.
    bandwidth_gib_s:
        Binary GiB/s, matching this repository's simulator numerics.
    path_io_ahead / required_io_count:
        Command counts.  They are optional in the bandwidth-only model, but
        required if ``iops_limit`` or ``fixed_command_latency_us`` is used.
    iops_limit:
        Optional aggregate throughput ceiling.  The cumulative service time
        uses ``max(bytes / bandwidth, command_count / IOPS)``.
    fixed_command_latency_us:
        Optional additional, non-overlapped serial overhead per command.  Do
        not set it when the supplied IOPS limit already includes that same
        overhead unless deliberate conservatism is desired.

    Notes
    -----
    The function predicts SSD completion, not final NPU ``io_ready``.  It does
    not model later arrivals interleaving during client submission, other Path
    arbitration, the 50-GiB/s NPU receive link, caches, or device latency
    distributions.  Use the discrete-event simulator when those matter.
    """

    path_bytes_ahead = _nonnegative_int("path_bytes_ahead", path_bytes_ahead)
    required_bytes = _nonnegative_int("required_bytes", required_bytes)
    time_budget_ms = _nonnegative_float("time_budget_ms", time_budget_ms)
    bandwidth_gib_s = _positive_float("bandwidth_gib_s", bandwidth_gib_s)
    fixed_command_latency_us = _nonnegative_float(
        "fixed_command_latency_us", fixed_command_latency_us
    )

    if path_io_ahead is not None:
        path_io_ahead = _nonnegative_int("path_io_ahead", path_io_ahead)
    if required_io_count is not None:
        required_io_count = _nonnegative_int(
            "required_io_count", required_io_count
        )
    if iops_limit is not None:
        iops_limit = _positive_float("iops_limit", iops_limit)
    count_dependent_model = iops_limit is not None or fixed_command_latency_us > 0.0
    if count_dependent_model and (
        path_io_ahead is None or required_io_count is None
    ):
        raise ValueError(
            "path_io_ahead and required_io_count are required when using "
            "iops_limit or fixed_command_latency_us"
        )

    bandwidth_bytes_s = bandwidth_gib_s * BYTES_PER_GIB
    ahead_ms, _, _, _ = _cumulative_service_ms(
        byte_count=path_bytes_ahead,
        io_count=path_io_ahead,
        bandwidth_bytes_s=bandwidth_bytes_s,
        iops_limit=iops_limit,
        fixed_command_latency_us=fixed_command_latency_us,
    )
    total_io_count = (
        None
        if path_io_ahead is None or required_io_count is None
        else path_io_ahead + required_io_count
    )
    total_ms, bandwidth_ms, iops_ms, fixed_latency_ms = _cumulative_service_ms(
        byte_count=path_bytes_ahead + required_bytes,
        io_count=total_io_count,
        bandwidth_bytes_s=bandwidth_bytes_s,
        iops_limit=iops_limit,
        fixed_command_latency_us=fixed_command_latency_us,
    )
    slack_ms = time_budget_ms - total_ms
    tolerance_ms = 1e-12
    return PathDeadlineEstimate(
        meets_deadline=slack_ms >= -tolerance_ms,
        time_budget_ms=time_budget_ms,
        estimated_ssd_completion_ms=total_ms,
        slack_ms=slack_ms,
        early_by_ms=max(0.0, slack_ms),
        shortfall_ms=max(0.0, -slack_ms),
        queue_ahead_ms=ahead_ms,
        target_incremental_service_ms=max(0.0, total_ms - ahead_ms),
        path_bytes_ahead=path_bytes_ahead,
        required_bytes=required_bytes,
        path_io_ahead=path_io_ahead,
        required_io_count=required_io_count,
        bandwidth_gib_s=bandwidth_gib_s,
        iops_limit=iops_limit,
        fixed_command_latency_us=fixed_command_latency_us,
        bandwidth_bound_ms=bandwidth_ms,
        iops_bound_ms=iops_ms,
        fixed_latency_ms=fixed_latency_ms,
        model_scope="frozen single-Path FCFS SSD stage; target appended after named backlog",
    )


def estimate_uniform_fcfs_path_deadline(
    *,
    path_io_ahead: int,
    required_io_count: int,
    io_size_bytes: int,
    time_budget_ms: float,
    bandwidth_gib_s: float = 40.0,
    iops_limit: float | None = None,
    fixed_command_latency_us: float = 0.0,
) -> PathDeadlineEstimate:
    """Count-based convenience wrapper for equal-size commands.

    This is the form suggested in the question.  ``io_size_bytes`` is required
    because the same count could mean 4-KiB or 176-KiB commands and therefore
    radically different transfer time.
    """

    path_io_ahead = _nonnegative_int("path_io_ahead", path_io_ahead)
    required_io_count = _nonnegative_int(
        "required_io_count", required_io_count
    )
    io_size_bytes = _nonnegative_int("io_size_bytes", io_size_bytes)
    if io_size_bytes == 0 and path_io_ahead + required_io_count > 0:
        raise ValueError("io_size_bytes must be positive when any I/O is present")
    return estimate_fcfs_path_deadline(
        path_bytes_ahead=path_io_ahead * io_size_bytes,
        required_bytes=required_io_count * io_size_bytes,
        time_budget_ms=time_budget_ms,
        bandwidth_gib_s=bandwidth_gib_s,
        path_io_ahead=path_io_ahead,
        required_io_count=required_io_count,
        iops_limit=iops_limit,
        fixed_command_latency_us=fixed_command_latency_us,
    )


__all__ = [
    "BYTES_PER_GIB",
    "PathDeadlineEstimate",
    "estimate_fcfs_path_deadline",
    "estimate_uniform_fcfs_path_deadline",
]
