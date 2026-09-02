"""SLO-cardinality admission extension for overloaded Scheme B epochs.

The normalized coflow allocator is the right fairness primitive when every
request can reach its warm SLO ratio.  When that target set is infeasible,
however, equalizing everybody below the threshold produces zero SLO passes.
This module adds a stable, request-lifetime admission layer: reserve a small
background share, pack as many low-cost request targets as possible, pin those
requests until completion, and use every residual byte with the original
work-conserving max-min allocator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np

import sim
from continuous_batch_control import (
    CapSpec,
    GrantMatrix,
    Matrix,
    allocate_coflow_grants,
    allocate_grants,
)
from continuous_batch_sim import (
    CIRControlDecision,
    CIRControlSnapshot,
    ControlRequestView,
)


_EPS = 1e-12


@dataclass(frozen=True)
class AdmissionAllocation:
    grants_gbps: GrantMatrix
    selected_npu_ids: tuple[int, ...]
    target_ratio: float
    required_ratio: float
    background_reserve_fraction: float


def _caps(spec: CapSpec, count: int, name: str) -> tuple[float, ...]:
    if isinstance(spec, Real):
        values = (float(spec),) * count
    else:
        values = tuple(float(value) for value in spec)
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def allocate_slo_admission_grants(
    demand: Matrix,
    *,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
) -> AdmissionAllocation:
    """Pack stable SLO targets, then spend all residual capacity.

    Selection is deterministic and count-oriented: already admitted requests
    are attempted first; new candidates are ordered by total normalized SSD
    footprint, dominant footprint, then NPU ID.  The small admission reserve
    prevents the target set from consuming every SSD CIR.  Residual allocation
    remains demand-capped and work-conserving, and advances each request's
    remaining SSU vector at one coflow ratio.  This avoids spending the scarce
    background share on a small fragment while its request remains blocked on
    a different SSU.

    This greedy multidimensional packing is deployable and stable, but is not
    claimed to be an exact maximum-cardinality knapsack solver.
    """
    demand_array = np.asarray(demand, dtype=float)
    if demand_array.size == 0:
        return AdmissionAllocation(
            tuple(tuple() for _ in range(len(demand))),
            (),
            float(target_ratio),
            float(required_ratio),
            float(background_reserve_fraction),
        )
    if demand_array.ndim != 2:
        raise ValueError("demand must be rectangular")
    if np.any(demand_array < 0.0) or not np.all(np.isfinite(demand_array)):
        raise ValueError("demand must contain finite non-negative values")
    ratio = float(target_ratio)
    required = float(required_ratio)
    reserve = float(background_reserve_fraction)
    if not 0.0 < ratio <= 1.0 or not math.isfinite(ratio):
        raise ValueError("target_ratio must be finite and in (0, 1]")
    if not 0.0 < required <= ratio or not math.isfinite(required):
        raise ValueError("required_ratio must be finite and in (0, target_ratio]")
    if not 0.0 <= reserve < 1.0 or not math.isfinite(reserve):
        raise ValueError("background_reserve_fraction must be finite and in [0, 1)")

    num_npu, num_ssu = demand_array.shape
    ssd_limits = np.asarray(_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float)
    npu_limits = np.asarray(_caps(npu_caps, num_npu, "npu_caps"), dtype=float)
    pinned = tuple(dict.fromkeys(int(npu) for npu in pinned_npu_ids))
    if any(npu < 0 or npu >= num_npu for npu in pinned):
        raise ValueError("pinned_npu_ids contains an invalid NPU")

    preferred_targets = ratio * demand_array
    required_targets = required * demand_array
    admission_remaining = (1.0 - reserve) * ssd_limits
    selected = []
    selected_set = set()
    all_requests_admitted = False

    def all_fit(targets):
        return bool(
            np.all(targets.sum(axis=0) <= ssd_limits + _EPS)
            and np.all(targets.sum(axis=1) <= npu_limits + _EPS)
        )

    # A background reserve is only an overload safety valve.  It must not turn
    # an actually feasible all-request SLO epoch into an artificial admission
    # failure.  Prefer the small operational margin; fall back to the exact
    # mathematical SLO floor when only that full set fits.
    if all_fit(preferred_targets):
        targets = preferred_targets
        selected = list(np.flatnonzero(demand_array.sum(axis=1) > _EPS))
        selected_set = set(selected)
        all_requests_admitted = True
    elif all_fit(required_targets):
        targets = required_targets
        selected = list(np.flatnonzero(demand_array.sum(axis=1) > _EPS))
        selected_set = set(selected)
        all_requests_admitted = True
    else:
        targets = preferred_targets

    def admit(npu_id):
        target = targets[npu_id]
        if demand_array[npu_id].sum() <= _EPS:
            return False
        if target.sum() > npu_limits[npu_id] + _EPS:
            return False
        if np.any(target > admission_remaining + _EPS):
            return False
        admission_remaining[:] -= target
        selected.append(npu_id)
        selected_set.add(npu_id)
        return True

    if not selected:
        for npu_id in pinned:
            admit(npu_id)

        candidates = []
        safe_ssd_limits = np.maximum(ssd_limits, _EPS)
        for npu_id in range(num_npu):
            if npu_id in selected_set or demand_array[npu_id].sum() <= _EPS:
                continue
            normalized = targets[npu_id] / safe_ssd_limits
            candidates.append(
                (
                    float(normalized.sum()),
                    float(normalized.max(initial=0.0)),
                    npu_id,
                )
            )
        for _, _, npu_id in sorted(candidates):
            admit(npu_id)

    base = np.zeros_like(demand_array)
    if selected:
        base[selected] = targets[selected]
    residual_ssd = tuple(
        max(0.0, limit - used) for limit, used in zip(ssd_limits, base.sum(axis=0))
    )
    residual_npu = tuple(
        max(0.0, limit - used) for limit, used in zip(npu_limits, base.sum(axis=1))
    )
    residual_demand = np.maximum(0.0, demand_array - base)
    if all_requests_admitted:
        # Every request already owns its complete SLO floor.  The remaining
        # bytes are no longer a threshold-cardinality decision, so use the
        # work-conserving absolute fill to maximize useful throughput.
        extra_grants = allocate_grants(
            residual_demand,
            ssd_caps=residual_ssd,
            npu_caps=residual_npu,
        )
    else:
        # During overload, advance every non-admitted request as one coflow so
        # its small fragments cannot consume the background share while a
        # barrier-critical fragment remains starved.
        extra_grants = allocate_coflow_grants(
            residual_demand,
            target_ratios=1.0,
            ssd_caps=residual_ssd,
            npu_caps=residual_npu,
        )
    extra = np.asarray(extra_grants, dtype=float)
    grants = base + extra
    return AdmissionAllocation(
        tuple(tuple(float(value) for value in row) for row in grants),
        tuple(selected),
        ratio,
        required,
        reserve,
    )


class SLOAdmissionSchemeBController:
    """Stateful full-manifest controller that pins admitted SLO targets."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        target_ratio: float = 0.52,
        required_ratio: float = 0.5,
        background_reserve_fraction: float = 0.05,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        if len(set(self.path_by_npu)) != len(self.path_by_npu):
            raise ValueError("SLO admission requires one unique Path per NPU")
        self.target_ratio = float(target_ratio)
        self.required_ratio = float(required_ratio)
        self.background_reserve_fraction = float(background_reserve_fraction)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self.selected_request_by_npu = {}
        self.last_allocation = None
        self.evaluations = 0

    @staticmethod
    def _remaining_manifest(request: ControlRequestView):
        work = request.remaining_work_gb_by_ssu
        compute_ms = request.remaining_compute_budget_ms
        if not work and request.remaining_layers > 0:
            work = tuple(
                request.remaining_layers * amount
                for amount in request.next_layer_work_gb_by_ssu
            )
            compute_ms = request.remaining_layers * request.per_layer_compute_ms
        return tuple(work), float(compute_ms)

    def __call__(self, snapshot: CIRControlSnapshot):
        if len(self.path_by_npu) != snapshot.num_npu:
            raise ValueError("dedicated Path mapping does not cover the fleet")
        work = [[0.0] * snapshot.num_ssu for _ in range(snapshot.num_npu)]
        compute_s = [0.0] * snapshot.num_npu
        request_by_npu = {}
        for request in snapshot.active_requests:
            remaining_work, compute_ms = self._remaining_manifest(request)
            if compute_ms <= 0.0 or not any(value > 0.0 for value in remaining_work):
                continue
            if request.npu_id in request_by_npu:
                raise ValueError("admission controller requires batch-1 active coflows")
            request_by_npu[request.npu_id] = request.request_id
            compute_s[request.npu_id] = compute_ms / 1000.0
            work[request.npu_id] = list(remaining_work)
        demands = tuple(
            tuple(
                amount / compute_s[npu] if compute_s[npu] > 0.0 else 0.0
                for amount in work[npu]
            )
            for npu in range(snapshot.num_npu)
        )
        pinned = tuple(
            npu
            for npu, request_id in self.selected_request_by_npu.items()
            if request_by_npu.get(npu) == request_id
        )
        allocation = allocate_slo_admission_grants(
            demands,
            target_ratio=self.target_ratio,
            required_ratio=self.required_ratio,
            background_reserve_fraction=self.background_reserve_fraction,
            pinned_npu_ids=pinned,
            ssd_caps=self.ssd_cap_gbps,
            npu_caps=self.npu_cap_gbps,
        )
        self.selected_request_by_npu = {
            npu: request_by_npu[npu]
            for npu in allocation.selected_npu_ids
            if npu in request_by_npu
        }
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        tables = []
        for ssu_id in range(snapshot.num_ssu):
            cirs = [0.0] * path_count
            for npu_id, path_id in enumerate(self.path_by_npu):
                cirs[path_id] = allocation.grants_gbps[npu_id][ssu_id]
            tables.append(tuple(cirs))
        self.last_allocation = allocation
        self.evaluations += 1
        return CIRControlDecision(tuple(tables))
