"""Causal request-SLO-aware path selection; no SSD or simulator dependency.

This experimental heuristic changes the *candidate pool* of newly submitted
I/O, then calls the unchanged Once routing engine. A request with substantial
waiting budget uses one (aggressive) or two (mild) legal paths per hardware
group. Requests with little budget retain their complete category-legal pool.
Packing slack-rich traffic activates fewer weighted paths, leaving more of the
shared service opportunity to urgent traffic. It is not a reservation, an EDF
queue, or an optimal scheduling algorithm. Every submitted path remains FIFO.

The caller supplies only upper-layer state belonging to this request, the
latest periodically collected path counts, and the known QoS configuration.
No future arrivals, live queues, simulated finish times, CIR/PIR mutation,
request reordering, or intentional I/O-release delay is used here. Requests
whose waiting budget is already exhausted fall back to Once; they are neither
discarded nor indefinitely relegated to a slow path. Unknown request deadlines
(e.g. pre-admission first-layer prefetch) also fall back to Once.

The budget is an admission-to-completion request budget, not the next layer's
I/O deadline. ``remaining_compute_ms`` includes the current compute remainder
and all subsequent own compute. ``remaining_layers`` counts remaining I/O
target layers, including the layer currently being routed, over which the
remaining waiting budget is shared (for target ``layer``, use ``8 - layer``).
These are estimates: positive budget does not prove that the request is
salvageable, and negative budget does not authorize dropping work.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from shared_path_once import once_path_ids


@dataclass(frozen=True)
class RequestBudget:
    remaining_slo_ms: float
    remaining_compute_ms: float
    remaining_layers: int
    layer_work_gib_by_ssu: tuple[float, ...]


@dataclass(frozen=True)
class RouterConfig:
    slack_budget_threshold_ms: float = 6.0
    slack_paths_per_group: int = 1

    def __post_init__(self):
        if not isfinite(self.slack_budget_threshold_ms) or self.slack_budget_threshold_ms < 0:
            raise ValueError("slack budget threshold must be finite and nonnegative")
        if self.slack_paths_per_group < 1:
            raise ValueError("slack requests must retain at least one path per group")


VARIANTS = {
    "mild": RouterConfig(slack_paths_per_group=2),
    "aggressive": RouterConfig(slack_paths_per_group=1),
}


@dataclass(frozen=True)
class PoolDecision:
    path_ids: tuple[int, ...]
    mode: str
    waiting_budget_ms: float | None
    waiting_budget_per_layer_ms: float | None
    isolated_layer_read_lower_bound_ms: float | None


def choose_path_pool(allowed_path_ids, qos, *, budget: RequestBudget | None,
                     config: RouterConfig = VARIANTS["aggressive"],
                     disk_bw_gib_s: float = 40.0,
                     npu_link_bw_gib_s: float = 50.0) -> PoolDecision:
    """Select a shared, group-balanced legal prefix using request-owned state.

    The physical isolated-read lower bound is only a guard against treating a
    large read's tiny waiting allowance as abundant slack. It does not predict
    queueing or assume I/O and compute are serialized. Actual count-based path
    congestion is considered by the unchanged Once engine after pool selection.
    """
    allowed = tuple(allowed_path_ids)
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("allowed path pool must be nonempty and contain no duplicates")
    if any(p < 0 or p >= qos.path_count for p in allowed):
        raise ValueError("path ID outside hardware range")
    if disk_bw_gib_s <= 0 or npu_link_bw_gib_s <= 0:
        raise ValueError("physical bandwidths must be positive")
    if budget is None:
        return PoolDecision(allowed, "unknown_budget_once", None, None, None)
    values = (budget.remaining_slo_ms, budget.remaining_compute_ms,
              *budget.layer_work_gib_by_ssu)
    if any(not isfinite(v) for v in values):
        raise ValueError("request budget must contain finite values")
    if budget.remaining_compute_ms < 0 or any(v < 0 for v in budget.layer_work_gib_by_ssu):
        raise ValueError("remaining compute and read sizes cannot be negative")
    if budget.remaining_layers < 1:
        raise ValueError("routing requires at least one unfinished layer")

    wait_ms = budget.remaining_slo_ms - budget.remaining_compute_ms
    per_layer_ms = wait_ms / budget.remaining_layers
    work = budget.layer_work_gib_by_ssu
    solo_ms = 1000.0 * max(max(work, default=0.0) / disk_bw_gib_s,
                          sum(work) / npu_link_bw_gib_s)
    threshold_ms = max(config.slack_budget_threshold_ms, solo_ms)
    if per_layer_ms <= threshold_ms:
        mode = "exhausted_budget_once" if wait_ms <= 0 else "urgent_full_pool"
        return PoolDecision(allowed, mode, wait_ms, per_layer_ms, solo_ms)

    # A shared prefix is deliberate: rotating the prefix per request would
    # activate the whole pool again and would not borrow slack at all. Preserve
    # the caller's established legal ordering, including Once tie-breaking.
    counts = {}
    selected = []
    for path_id in allowed:
        group_id = path_id // qos.paths_per_group
        if counts.get(group_id, 0) < config.slack_paths_per_group:
            selected.append(path_id)
            counts[group_id] = counts.get(group_id, 0) + 1
    return PoolDecision(tuple(selected), "slack_shared_prefix", wait_ms,
                        per_layer_ms, solo_ms)


def slo_path_ids(io_count, snapshot, allowed_path_ids, qos, *,
                 budget: RequestBudget | None,
                 config: RouterConfig = VARIANTS["aggressive"],
                 start_offset: int = 0, disk_bw_gib_s: float = 40.0,
                 npu_link_bw_gib_s: float = 50.0) -> tuple[int, ...]:
    """Route new equal-sized I/O using only the supplied sampled snapshot."""
    if io_count < 0:
        raise ValueError("I/O count must be nonnegative")
    if io_count == 0:
        return ()
    decision = choose_path_pool(
        allowed_path_ids, qos, budget=budget, config=config,
        disk_bw_gib_s=disk_bw_gib_s, npu_link_bw_gib_s=npu_link_bw_gib_s,
    )
    return once_path_ids(io_count, snapshot, decision.path_ids, qos,
                         start_offset=start_offset, disk_bw_gib_s=disk_bw_gib_s)
