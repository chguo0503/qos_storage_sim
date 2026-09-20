"""Current L3 strategies generalized to diverse profiles without role labels.

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
remaining waiting budget is shared (for target ``layer``, use ``n_layers - layer``).
These are estimates: positive budget does not prove that the request is
salvageable, and negative budget does not authorize dropping work.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    "static": RouterConfig(slack_paths_per_group=1),
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


POLICIES = ("baseline", "mild", "aggressive", "static")


def simulator_strategy(policy_name):
    """Original adapter backbone; the extension only replaces Once routing."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    return "baseline" if policy_name == "baseline" else "once"


def make_stats():
    """Bounded categorical aggregates, not a per-block event log."""
    return dict(calls=0, io_blocks=0, unique_paths_sum=0, candidate_paths_sum=0,
                mode_counts={}, decision_counts={}, examples=[])


def upper_budget(context, state, now_ms, policy_name="aggressive"):
    """Read only this request's host-visible profile, admission, and progress.

    Static freezes the initial budget for *any* data profile. The deciding
    initial waiting budget per layer is 0.5*C, independent of category/role.
    Category only determines the legal pool supplied by the existing adapter.
    No current SSD queue, future event, or future request is inspected.
    """
    if policy_name not in VARIANTS:
        raise ValueError("upper_budget is for mild/aggressive/static")
    request = context.requests[state.request_id]
    if not request.admitted:
        return None
    assert request.batch_size == 1
    c = request.per_layer_compute_ms
    if policy_name == "static":
        remaining_compute = context.n_layers * c
        remaining_slo = 1.5 * remaining_compute
        remaining_layers = context.n_layers
    else:
        batch = context.microbatches[request.batch_id]
        if batch.compute_active_layer >= 0:
            k = batch.compute_active_layer
            elapsed = max(0., now_ms - batch.layer_metrics[k].compute_start_ms)
            remaining_compute = max(0., c - elapsed) + (context.n_layers-k-1)*c
        else:
            remaining_compute = (context.n_layers-batch.compute_done_up_to-1)*c
        remaining_slo = request.admission_time_ms + 1.5*context.n_layers*c - now_ms
        remaining_layers = context.n_layers-state.layer
    layout = request.manifest.placement[0 if len(request.manifest.placement)==1 else state.layer]
    work = [0.] * context.num_ssu
    for ssu, size in layout:
        work[ssu] += size
    return RequestBudget(remaining_slo, remaining_compute, remaining_layers, tuple(work))


@contextmanager
def install_policy(policy_name, stats=None):
    """Temporarily install the extension at the original adapter boundary.

    Usage::

        with install_policy(name, stats):
            run_case(..., strategy=simulator_strategy(name))

    The caller still drives the ordinary simulator and original shared adapter.
    Its FIFO, static QoS, sampled count table, L1/L2, submit times, and initial
    unadmitted prefetch are preserved. Policy compute latency remains the same
    as the reference (zero simulated overhead). Each process runs one policy.
    """
    import continuous_batch_sim as native
    import shared_path_sim_adapter as shared
    import shared_path_once as once_module

    strategy = simulator_strategy(policy_name)
    if stats is None:
        stats = make_stats()
    original_manager = shared.shared_path_adapter

    @contextmanager
    def manager(**kwargs):
        assert kwargs.get("strategy", "once") == strategy
        if policy_name == "baseline":
            with original_manager(**kwargs) as adapter:
                yield adapter
            return
        config = VARIANTS[policy_name]
        decision = None

        def route(count, snapshot, allowed_path_ids, qos, start_offset=0,
                  disk_bw_gib_s=40.0):
            assert decision is not None
            ids = once_path_ids(count, snapshot, decision.path_ids, qos,
                                start_offset=start_offset, disk_bw_gib_s=disk_bw_gib_s)
            assert len(ids) == count and set(ids).issubset(allowed_path_ids)
            stats["io_blocks"] += count
            stats["unique_paths_sum"] += len(set(ids))
            return ids

        with patch.object(once_module, "once_path_ids", route):
            with original_manager(**kwargs) as adapter:
                original_plan = native._plan_paths

                def plan(context, state, now_ms):
                    nonlocal decision
                    budget = upper_budget(context, state, now_ms, policy_name)
                    decision = choose_path_pool(
                        state.allowed_path_ids, adapter.qos[state.disk_id],
                        budget=budget, config=config, disk_bw_gib_s=context.disk_bw_gbps,
                        npu_link_bw_gib_s=context.npu_bw_gbps,
                    )
                    load = context.requests[state.request_id].manifest.load
                    category = load["category"]
                    profile = f"{load.get('seq_len_k', '?')}:{load.get('nql', '?')}"
                    key = f"{category}|{profile}|layer{state.layer}|{decision.mode}"
                    stats["calls"] += 1
                    stats["candidate_paths_sum"] += len(decision.path_ids)
                    stats["mode_counts"][decision.mode] = stats["mode_counts"].get(decision.mode, 0) + 1
                    stats["decision_counts"][key] = stats["decision_counts"].get(key, 0) + 1
                    if len(stats["examples"]) < 32:
                        stats["examples"].append(dict(time_ms=now_ms,
                            request_id=state.request_id, npu_id=state.npu_id,
                            layer=state.layer, ssu_id=state.disk_id, category=category,
                            profile=profile, mode=decision.mode,
                            candidate_path_count=len(decision.path_ids),
                            waiting_budget_ms=decision.waiting_budget_ms,
                            waiting_budget_per_layer_ms=decision.waiting_budget_per_layer_ms))
                    return original_plan(context, state, now_ms)

                with patch.object(native, "_plan_paths", plan):
                    yield adapter

    with patch.object(shared, "shared_path_adapter", manager):
        yield stats

