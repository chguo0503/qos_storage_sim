"""Scheme-B control plane for continuous-batch CIR allocation.

The allocator treats every ``(NPU, SSU)`` pair as a flow consuming one NPU
receive link and one SSU.  Grants are weighted max-min fair, demand capped,
and obey both sets of capacities.  The commit helper keeps those target grants
off the SSU fast path until a material change or a bounded epoch deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Optional, Sequence, Tuple, Union

import numpy as np


Matrix = Sequence[Sequence[float]]
CapSpec = Union[float, Sequence[float]]
GrantMatrix = Tuple[Tuple[float, ...], ...]
_EPS = 1e-12


@dataclass(frozen=True)
class CIRChange:
    npu: int
    ssu: int
    old_gbps: float
    new_gbps: float


@dataclass(frozen=True)
class CIRCommitPlan:
    commit: bool
    reason: str
    changes: Tuple[CIRChange, ...]


def _matrix(values: Matrix, name: str) -> GrantMatrix:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    if any(value < 0.0 or not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain finite non-negative values")
    return rows


def _capacities(spec: CapSpec, count: int, name: str) -> Tuple[float, ...]:
    if isinstance(spec, Real):
        values = (float(spec),) * count
    else:
        values = tuple(float(value) for value in spec)
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def _resource_steps(
    headroom: np.ndarray,
    active: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    """Water level at which each row consumes its residual capacity."""
    values = np.where(active, headroom, np.inf)
    values.sort(axis=1)
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    values[~valid] = 0.0
    prefix = np.cumsum(values, axis=1)
    denominator = counts[:, None] - np.arange(values.shape[1])
    candidates = np.full_like(values, -np.inf)
    np.divide(
        residual[:, None] - (prefix - values),
        denominator,
        out=candidates,
        where=denominator > 0,
    )
    steps = np.maximum(candidates.max(axis=1), 0.0)
    steps[(counts == 0) | (prefix[:, -1] < residual)] = np.inf
    steps[(residual <= _EPS) & (counts > 0)] = 0.0
    return steps


def _allocate_equal(
    demand: np.ndarray,
    ssd_limits: Tuple[float, ...],
    npu_limits: Tuple[float, ...],
) -> GrantMatrix:
    """Jump directly between row/column saturation events for unit weights."""
    grants = np.zeros_like(demand)
    npu_caps = np.asarray(npu_limits, dtype=float)
    ssd_caps = np.asarray(ssd_limits, dtype=float)
    open_npus = npu_caps > 0.0
    open_ssus = ssd_caps > 0.0
    npu_used = np.zeros_like(npu_caps)
    ssd_used = np.zeros_like(ssd_caps)
    scale = max(1.0, float(npu_caps.max()), float(ssd_caps.max()))

    while open_npus.any() and open_ssus.any():
        npu_ids = np.flatnonzero(open_npus)
        ssu_ids = np.flatnonzero(open_ssus)
        block = np.ix_(npu_ids, ssu_ids)
        current = grants[block]
        target = demand[block]
        headroom = target - current
        active = headroom > 0.0
        if not active.any():
            break

        npu_steps = _resource_steps(
            headroom,
            active,
            npu_caps[npu_ids] - npu_used[npu_ids],
        )
        ssu_steps = _resource_steps(
            headroom.T,
            active.T,
            ssd_caps[ssu_ids] - ssd_used[ssu_ids],
        )
        step = min(float(npu_steps.min()), float(ssu_steps.min()))
        if not math.isfinite(step):
            increment = np.where(active, headroom, 0.0)
            grants[block] = current + increment
            npu_used[npu_ids] += increment.sum(axis=1)
            ssd_used[ssu_ids] += increment.sum(axis=0)
            break

        increment = np.where(
            active,
            np.minimum(headroom, step),
            0.0,
        )
        grants[block] = current + increment
        npu_used[npu_ids] += increment.sum(axis=1)
        ssd_used[ssu_ids] += increment.sum(axis=0)
        tolerance = _EPS * max(scale, abs(step))
        saturated_npus = np.isfinite(npu_steps) & (
            npu_steps <= step + tolerance
        )
        saturated_ssus = np.isfinite(ssu_steps) & (
            ssu_steps <= step + tolerance
        )
        open_npus[npu_ids[saturated_npus]] = False
        open_ssus[ssu_ids[saturated_ssus]] = False

    return tuple(tuple(float(value) for value in row) for row in grants)


def allocate_grants(
    demand: Matrix,
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
    weights: Optional[Matrix] = None,
) -> GrantMatrix:
    """Return deterministic demand-capped weighted progressive-fill grants.

    ``weights[n][s]`` is the relative growth rate of that flow.  A flow with
    weight two receives twice the grant of an equally situated weight-one flow
    until a demand, NPU, or SSU constraint becomes tight.
    """
    if weights is None:
        demand_array = np.asarray(demand, dtype=float)
        if demand_array.size == 0:
            return tuple(tuple() for _ in range(len(demand)))
        if demand_array.ndim != 2:
            raise ValueError("demand must be rectangular")
        if np.any(demand_array < 0.0) or not np.all(np.isfinite(demand_array)):
            raise ValueError("demand must contain finite non-negative values")
        npu_count, ssu_count = demand_array.shape
        npu_limits = _capacities(npu_caps, npu_count, "npu_caps")
        ssd_limits = _capacities(ssd_caps, ssu_count, "ssd_caps")
        return _allocate_equal(demand_array, ssd_limits, npu_limits)

    demands = _matrix(demand, "demand")
    npu_count = len(demands)
    ssu_count = len(demands[0]) if demands else 0
    npu_limits = _capacities(npu_caps, npu_count, "npu_caps")
    ssd_limits = _capacities(ssd_caps, ssu_count, "ssd_caps")
    if npu_count == 0 or ssu_count == 0:
        return demands
    rates = _matrix(weights, "weights")
    if len(rates) != npu_count or any(len(row) != ssu_count for row in rates):
        raise ValueError("weights must have the same shape as demand")
    if any(
        demands[npu][ssu] > 0.0 and rates[npu][ssu] <= 0.0
        for npu in range(npu_count)
        for ssu in range(ssu_count)
    ):
        raise ValueError("positive-demand flows must have positive weights")

    grants = [[0.0] * ssu_count for _ in range(npu_count)]
    active = [
        [
            demands[npu][ssu] > 0.0
            and npu_limits[npu] > 0.0
            and ssd_limits[ssu] > 0.0
            for ssu in range(ssu_count)
        ]
        for npu in range(npu_count)
    ]

    while any(any(row) for row in active):
        row_used = [sum(row) for row in grants]
        col_used = [
            sum(grants[npu][ssu] for npu in range(npu_count))
            for ssu in range(ssu_count)
        ]
        row_rate = [
            sum(rates[npu][ssu] for ssu in range(ssu_count) if active[npu][ssu])
            for npu in range(npu_count)
        ]
        col_rate = [
            sum(rates[npu][ssu] for npu in range(npu_count) if active[npu][ssu])
            for ssu in range(ssu_count)
        ]

        steps = [
            (demands[npu][ssu] - grants[npu][ssu]) / rates[npu][ssu]
            for npu in range(npu_count)
            for ssu in range(ssu_count)
            if active[npu][ssu]
        ]
        steps.extend(
            (npu_limits[npu] - row_used[npu]) / row_rate[npu]
            for npu in range(npu_count)
            if row_rate[npu] > 0.0
        )
        steps.extend(
            (ssd_limits[ssu] - col_used[ssu]) / col_rate[ssu]
            for ssu in range(ssu_count)
            if col_rate[ssu] > 0.0
        )
        step = max(0.0, min(steps))

        for npu in range(npu_count):
            for ssu in range(ssu_count):
                if active[npu][ssu]:
                    grants[npu][ssu] = min(
                        demands[npu][ssu],
                        grants[npu][ssu] + step * rates[npu][ssu],
                    )

        row_used = [sum(row) for row in grants]
        col_used = [
            sum(grants[npu][ssu] for npu in range(npu_count))
            for ssu in range(ssu_count)
        ]
        for npu in range(npu_count):
            for ssu in range(ssu_count):
                if not active[npu][ssu]:
                    continue
                scale = max(
                    1.0,
                    demands[npu][ssu],
                    npu_limits[npu],
                    ssd_limits[ssu],
                )
                if (
                    demands[npu][ssu] - grants[npu][ssu] <= _EPS * scale
                    or npu_limits[npu] - row_used[npu] <= _EPS * scale
                    or ssd_limits[ssu] - col_used[ssu] <= _EPS * scale
                ):
                    active[npu][ssu] = False

    return tuple(tuple(row) for row in grants)


def batch_cir_diff(
    current: Matrix,
    target: Matrix,
    abs_tol_gbps: float = 1e-9,
) -> Tuple[CIRChange, ...]:
    """Return a stable, row-major batch of changed ``(NPU, SSU)`` grants."""
    old = _matrix(current, "current")
    new = _matrix(target, "target")
    if len(old) != len(new) or any(len(a) != len(b) for a, b in zip(old, new)):
        raise ValueError("current and target must have the same shape")
    return tuple(
        CIRChange(npu, ssu, old[npu][ssu], new[npu][ssu])
        for npu in range(len(old))
        for ssu in range(len(old[npu]))
        if abs(old[npu][ssu] - new[npu][ssu]) > abs_tol_gbps
    )


def plan_cir_commit(
    current: Matrix,
    target: Matrix,
    epoch: int,
    last_commit_epoch: int,
    relative_threshold: float = 0.10,
    max_epoch_interval: int = 8,
    relative_floor_gbps: float = 1.0,
    abs_tol_gbps: float = 1e-9,
) -> CIRCommitPlan:
    """Plan an atomic CIR update, while deferring harmless batch churn.

    Activation/deactivation commits immediately.  Otherwise a commit happens
    when any relative change exceeds ``relative_threshold`` or when pending
    changes have waited ``max_epoch_interval`` epochs.  A committed plan carries
    every pending change so the SSUs can switch to one coherent target epoch.
    """
    if epoch < last_commit_epoch:
        raise ValueError("epoch cannot precede last_commit_epoch")
    if relative_threshold < 0.0 or max_epoch_interval <= 0:
        raise ValueError("threshold must be non-negative and interval positive")
    changes = batch_cir_diff(current, target, abs_tol_gbps)
    if not changes:
        return CIRCommitPlan(False, "unchanged", ())

    if any(
        (change.old_gbps <= abs_tol_gbps) != (change.new_gbps <= abs_tol_gbps)
        for change in changes
    ):
        reason = "activation_change"
    elif any(
        abs(change.new_gbps - change.old_gbps)
        / max(abs(change.old_gbps), relative_floor_gbps)
        > relative_threshold
        for change in changes
    ):
        reason = "relative_change"
    elif epoch - last_commit_epoch >= max_epoch_interval:
        reason = "max_epoch_interval"
    else:
        return CIRCommitPlan(False, "deferred", ())
    return CIRCommitPlan(True, reason, changes)


def plan_global_cir_commit(
    current: Matrix,
    target: Matrix,
    epoch: int,
    pending_since_epoch: Optional[int],
    global_deficit_threshold: float = 0.10,
    max_epoch_interval: int = 8,
    minimum_npu_coverage: Optional[float] = None,
    minimum_ssu_coverage: Optional[float] = None,
    abs_tol_gbps: float = 1e-9,
) -> CIRCommitPlan:
    """Plan a whole-matrix commit from aggregate target coverage deficit.

    The trigger is ``sum(max(target-current, 0)) / sum(target)``.  Individual
    Path activation is therefore deferred unless its contribution makes the
    global deficit material; optional NPU/SSU coverage floors protect tail
    resources.  A triggered commit still carries the entire diff.  The
    deadline starts at ``pending_since_epoch``; ``None`` disables it.
    """
    if pending_since_epoch is not None and epoch < pending_since_epoch:
        raise ValueError("epoch cannot precede pending_since_epoch")
    if global_deficit_threshold < 0.0 or max_epoch_interval <= 0:
        raise ValueError("threshold must be non-negative and interval positive")
    old = _matrix(current, "current")
    new = _matrix(target, "target")
    changes = batch_cir_diff(old, new, abs_tol_gbps)
    if not changes:
        return CIRCommitPlan(False, "unchanged", ())

    target_sum = sum(sum(row) for row in new)
    deficit = (
        sum(
            max(new[npu][ssu] - old[npu][ssu], 0.0)
            for npu in range(len(new))
            for ssu in range(len(new[npu]))
        )
        / target_sum
        if target_sum > 0.0
        else 0.0
    )
    def minimum_coverage(by_npu: bool) -> float:
        if by_npu:
            totals = [sum(row) for row in new]
            covered = [
                sum(min(old[npu][ssu], new[npu][ssu]) for ssu in range(len(new[npu])))
                for npu in range(len(new))
            ]
        else:
            width = len(new[0]) if new else 0
            totals = [sum(row[ssu] for row in new) for ssu in range(width)]
            covered = [
                sum(min(old[npu][ssu], new[npu][ssu]) for npu in range(len(new)))
                for ssu in range(width)
            ]
        return min(
            (have / want for have, want in zip(covered, totals) if want > 0.0),
            default=1.0,
        )

    if (
        minimum_npu_coverage is not None
        and minimum_coverage(True) < minimum_npu_coverage
    ):
        reason = "npu_coverage"
    elif (
        minimum_ssu_coverage is not None
        and minimum_coverage(False) < minimum_ssu_coverage
    ):
        reason = "ssu_coverage"
    elif deficit > global_deficit_threshold:
        reason = "global_deficit"
    elif (
        pending_since_epoch is not None
        and epoch - pending_since_epoch >= max_epoch_interval
    ):
        reason = "max_epoch_interval"
    else:
        return CIRCommitPlan(False, "deferred", ())
    return CIRCommitPlan(True, reason, changes)


def _self_check() -> None:
    assert allocate_grants(((10.0,), (30.0,))) == ((10.0,), (30.0,))
    demand = ((10.0, 50.0), (30.0, 30.0), (40.0, 10.0))
    grants = allocate_grants(demand)
    assert all(
        0.0 <= grants[npu][ssu] <= demand[npu][ssu] + 1e-9
        for npu in range(3)
        for ssu in range(2)
    )
    assert all(sum(row) <= 50.0 + 1e-9 for row in grants)
    assert all(sum(row[ssu] for row in grants) <= 40.0 + 1e-9 for ssu in range(2))
    assert grants == allocate_grants(demand)

    current = ((10.0, 20.0),)
    target = ((10.5, 19.5),)
    small = plan_cir_commit(current, target, 3, 0)
    assert not small.commit and small.reason == "deferred"
    timeout = plan_cir_commit(current, target, 8, 0)
    assert timeout.commit and timeout.reason == "max_epoch_interval"
    startup = plan_cir_commit(((0.0,),), ((1.0,),), 1, 0)
    assert startup.commit and startup.reason == "activation_change"

    small_activation = plan_global_cir_commit(
        ((20.0, 0.0, 5.0),),
        ((20.0, 1.0, 5.0),),
        1,
        None,
    )
    assert not small_activation.commit and small_activation.reason == "deferred"
    global_change = plan_global_cir_commit(
        ((20.0, 0.0, 5.0),),
        ((20.0, 5.0, 4.5),),
        1,
        None,
    )
    assert global_change.reason == "global_deficit" and len(global_change.changes) == 2
    global_timeout = plan_global_cir_commit(
        ((20.0, 0.0, 5.0),),
        ((20.0, 1.0, 5.0),),
        8,
        0,
    )
    assert global_timeout.commit and global_timeout.reason == "max_epoch_interval"
    guarded = plan_global_cir_commit(
        ((5.0, 20.0),),
        ((10.0, 15.0),),
        2,
        2,
        global_deficit_threshold=0.5,
        minimum_npu_coverage=0.9,
    )
    assert guarded.commit and guarded.reason == "npu_coverage"
    no_pending_deadline = plan_global_cir_commit(
        ((20.0, 0.0, 5.0),),
        ((20.0, 1.0, 5.0),),
        100,
        None,
    )
    assert not no_pending_deadline.commit
    print("continuous_batch_control self-check: OK")


if __name__ == "__main__":
    _self_check()
