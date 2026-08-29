"""Demand-capped max-min grant allocation for Scheme B.

Every ``(NPU, SSU)`` pair is a flow that consumes one NPU receive link and one
SSU.  The allocator returns equal-progressive-fill grants that obey both sets
of capacities.  CIR update timing and simulator events intentionally live
outside this module.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence, Tuple, Union

import numpy as np


Matrix = Sequence[Sequence[float]]
CapSpec = Union[float, Sequence[float]]
GrantMatrix = Tuple[Tuple[float, ...], ...]
_EPS = 1e-12


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
) -> GrantMatrix:
    """Return deterministic, demand-capped equal max-min grants."""
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
