"""Explicit-tail SLO-admission Scheme B prototype.

This module is deliberately separate from :mod:`slo_admission_scheme_b` so the
frozen 32/128-NPU runners and their source fingerprints remain unchanged.  V2
keeps the original threshold admission decision, but makes the capacity left
after the selected-request floor explicit:

1. selected requests receive an immutable request-level SLO floor;
2. rejected requests receive a small proportional coflow background grant;
3. selected requests consume feasible tail capacity first; and
4. an absolute max-min spill consumes any tail that selected requests cannot.

The last two stages prevent residual SSD capacity from being assigned only by
the hardware WRR surplus rule.  Every stage remains demand-capped and obeys
both SSD and NPU-link capacities.
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
class AdmissionAllocationV2:
    """A V2 grant plus its independently auditable allocation stages."""

    grants_gbps: GrantMatrix
    selected_npu_ids: tuple[int, ...]
    target_ratio: float
    required_ratio: float
    effective_floor_ratio: float
    background_reserve_fraction: float
    floor_grants_gbps: GrantMatrix
    background_grants_gbps: GrantMatrix
    selected_tail_grants_gbps: GrantMatrix
    spill_tail_grants_gbps: GrantMatrix


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


def _matrix_tuple(values: np.ndarray) -> GrantMatrix:
    return tuple(tuple(float(value) for value in row) for row in values)


def _remaining_caps(
    grants: np.ndarray,
    ssd_limits: np.ndarray,
    npu_limits: np.ndarray,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(
            max(0.0, float(limit - used))
            for limit, used in zip(ssd_limits, grants.sum(axis=0))
        ),
        tuple(
            max(0.0, float(limit - used))
            for limit, used in zip(npu_limits, grants.sum(axis=1))
        ),
    )


def allocate_slo_admission_grants_v2(
    demand: Matrix,
    *,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
) -> AdmissionAllocationV2:
    """Pack SLO floors and explicitly allocate every feasible tail byte.

    Admission order intentionally matches V1: feasible pinned requests first,
    then increasing total normalized SSD footprint, dominant footprint, and NPU
    ID.  The background reserve applies only to rejected rows.  Tail allocation
    can never reduce either the selected floor or the background component.
    """

    demand_array = np.asarray(demand, dtype=float)
    if demand_array.size == 0:
        empty = tuple(tuple() for _ in range(len(demand)))
        return AdmissionAllocationV2(
            grants_gbps=empty,
            selected_npu_ids=(),
            target_ratio=float(target_ratio),
            required_ratio=float(required_ratio),
            effective_floor_ratio=float(target_ratio),
            background_reserve_fraction=float(background_reserve_fraction),
            floor_grants_gbps=empty,
            background_grants_gbps=empty,
            selected_tail_grants_gbps=empty,
            spill_tail_grants_gbps=empty,
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
        raise ValueError(
            "background_reserve_fraction must be finite and in [0, 1)"
        )

    num_npu, num_ssu = demand_array.shape
    ssd_limits = np.asarray(_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float)
    npu_limits = np.asarray(_caps(npu_caps, num_npu, "npu_caps"), dtype=float)
    pinned = tuple(dict.fromkeys(int(npu) for npu in pinned_npu_ids))
    if any(npu < 0 or npu >= num_npu for npu in pinned):
        raise ValueError("pinned_npu_ids contains an invalid NPU")

    preferred_targets = ratio * demand_array
    required_targets = required * demand_array
    admission_remaining = (1.0 - reserve) * ssd_limits
    selected: list[int] = []
    selected_set: set[int] = set()
    all_requests_admitted = False

    def all_fit(targets: np.ndarray) -> bool:
        return bool(
            np.all(targets.sum(axis=0) <= ssd_limits + _EPS)
            and np.all(targets.sum(axis=1) <= npu_limits + _EPS)
        )

    if all_fit(preferred_targets):
        targets = preferred_targets
        effective_floor_ratio = ratio
        selected = list(np.flatnonzero(demand_array.sum(axis=1) > _EPS))
        selected_set = set(selected)
        all_requests_admitted = True
    elif all_fit(required_targets):
        targets = required_targets
        effective_floor_ratio = required
        selected = list(np.flatnonzero(demand_array.sum(axis=1) > _EPS))
        selected_set = set(selected)
        all_requests_admitted = True
    else:
        targets = preferred_targets
        effective_floor_ratio = ratio

    def admit(npu_id: int) -> bool:
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

    if not all_requests_admitted:
        for npu_id in pinned:
            admit(npu_id)

        safe_ssd_limits = np.maximum(ssd_limits, _EPS)
        candidates = []
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

    floor = np.zeros_like(demand_array)
    if selected:
        floor[selected] = targets[selected]

    # Reserve a bounded, request-proportional progress pool only for rejected
    # requests.  Any reserve that coflow fill cannot use remains available to
    # the explicit tail stages below.
    residual_ssd, residual_npu = _remaining_caps(
        floor, ssd_limits, npu_limits
    )
    background_caps = tuple(
        min(remaining, reserve * limit)
        for remaining, limit in zip(residual_ssd, ssd_limits)
    )
    rejected_demand = np.zeros_like(demand_array)
    if not all_requests_admitted:
        rejected = [npu for npu in range(num_npu) if npu not in selected_set]
        if rejected:
            rejected_demand[rejected] = np.maximum(
                0.0, demand_array[rejected] - floor[rejected]
            )
    background = np.asarray(
        allocate_coflow_grants(
            rejected_demand,
            target_ratios=1.0,
            ssd_caps=background_caps,
            npu_caps=residual_npu,
        ),
        dtype=float,
    )

    grants = floor + background
    residual_ssd, residual_npu = _remaining_caps(
        grants, ssd_limits, npu_limits
    )
    selected_residual = np.zeros_like(demand_array)
    if selected:
        selected_residual[selected] = np.maximum(
            0.0, demand_array[selected] - grants[selected]
        )
    selected_tail = np.asarray(
        allocate_grants(
            selected_residual,
            ssd_caps=residual_ssd,
            npu_caps=residual_npu,
        ),
        dtype=float,
    )

    grants = grants + selected_tail
    residual_ssd, residual_npu = _remaining_caps(
        grants, ssd_limits, npu_limits
    )
    spill_tail = np.asarray(
        allocate_grants(
            np.maximum(0.0, demand_array - grants),
            ssd_caps=residual_ssd,
            npu_caps=residual_npu,
        ),
        dtype=float,
    )
    grants = grants + spill_tail

    return AdmissionAllocationV2(
        grants_gbps=_matrix_tuple(grants),
        selected_npu_ids=tuple(selected),
        target_ratio=ratio,
        required_ratio=required,
        effective_floor_ratio=effective_floor_ratio,
        background_reserve_fraction=reserve,
        floor_grants_gbps=_matrix_tuple(floor),
        background_grants_gbps=_matrix_tuple(background),
        selected_tail_grants_gbps=_matrix_tuple(selected_tail),
        spill_tail_grants_gbps=_matrix_tuple(spill_tail),
    )


class SLOAdmissionSchemeBControllerV2:
    """Full-manifest adapter for the explicit-tail V2 allocator."""

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
        self.selected_request_by_npu: dict[int, int] = {}
        self.last_allocation: AdmissionAllocationV2 | None = None
        self.evaluations = 0

    @staticmethod
    def _remaining_manifest(
        request: ControlRequestView,
    ) -> tuple[tuple[float, ...], float]:
        work = request.remaining_work_gb_by_ssu
        compute_ms = request.remaining_compute_budget_ms
        if not work and request.remaining_layers > 0:
            work = tuple(
                request.remaining_layers * amount
                for amount in request.next_layer_work_gb_by_ssu
            )
            compute_ms = request.remaining_layers * request.per_layer_compute_ms
        return tuple(work), float(compute_ms)

    def __call__(self, snapshot: CIRControlSnapshot) -> CIRControlDecision:
        if len(self.path_by_npu) != snapshot.num_npu:
            raise ValueError("dedicated Path mapping does not cover the fleet")
        work = [[0.0] * snapshot.num_ssu for _ in range(snapshot.num_npu)]
        compute_s = [0.0] * snapshot.num_npu
        request_by_npu: dict[int, int] = {}
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
        allocation = allocate_slo_admission_grants_v2(
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

