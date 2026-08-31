"""Adaptive residual-mode admission controller (V2.1 prototype).

The admission-selected fraction is computed causally from the current active
manifest and current SSD/NPU capacities.  It is used only to choose how bytes
left after the selected SLO floors are spent:

* below ``explicit_spill_threshold``: use V2 selected-first explicit spill;
* at or above the threshold: retain V1 request-proportional coflow residual.

The 0.75 default is an operating-point heuristic, not a universal optimum.  It
lies in the measured gap between the severe-overload sequence-0 points
(0.6875 at 32x6 and 128x24) and moderate points (0.8125 at 32x10 and 0.8047 at
128x40).  The decision uses no future completion or simulator-private state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

import sim
from continuous_batch_control import CapSpec, GrantMatrix, Matrix
from continuous_batch_sim import (
    CIRControlDecision,
    CIRControlSnapshot,
    ControlRequestView,
)
from slo_admission_scheme_b import (
    AdmissionAllocation,
    allocate_slo_admission_grants,
)
from slo_admission_scheme_b_v2 import (
    AdmissionAllocationV2,
    allocate_slo_admission_grants_v2,
)


RESIDUAL_MODE_COFLOW = "v1_coflow_residual"
RESIDUAL_MODE_EXPLICIT = "v2_explicit_selected_spill"


@dataclass(frozen=True)
class AdaptiveAdmissionAllocation:
    grants_gbps: GrantMatrix
    selected_npu_ids: tuple[int, ...]
    active_npu_count: int
    selected_fraction: float
    explicit_spill_threshold: float
    residual_mode: str
    v1_allocation: AdmissionAllocation
    v2_allocation: AdmissionAllocationV2 | None


def allocate_adaptive_admission_grants(
    demand: Matrix,
    *,
    explicit_spill_threshold: float = 0.75,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
) -> AdaptiveAdmissionAllocation:
    """Choose V1/V2 residual service from the current selected fraction."""
    threshold = float(explicit_spill_threshold)
    if not 0.0 < threshold <= 1.0 or not math.isfinite(threshold):
        raise ValueError("explicit_spill_threshold must be finite and in (0, 1]")
    demand_array = np.asarray(demand, dtype=float)
    if demand_array.size and demand_array.ndim != 2:
        raise ValueError("demand must be rectangular")
    if demand_array.size and (
        np.any(demand_array < 0.0) or not np.all(np.isfinite(demand_array))
    ):
        raise ValueError("demand must contain finite non-negative values")

    common = {
        "target_ratio": target_ratio,
        "required_ratio": required_ratio,
        "background_reserve_fraction": background_reserve_fraction,
        "pinned_npu_ids": pinned_npu_ids,
        "ssd_caps": ssd_caps,
        "npu_caps": npu_caps,
    }
    v1 = allocate_slo_admission_grants(demand, **common)
    active_count = (
        int(np.count_nonzero(demand_array.sum(axis=1) > 1e-12))
        if demand_array.size
        else 0
    )
    selected_fraction = (
        len(v1.selected_npu_ids) / active_count if active_count else 1.0
    )

    if selected_fraction < threshold:
        v2 = allocate_slo_admission_grants_v2(demand, **common)
        if v2.selected_npu_ids != v1.selected_npu_ids:
            raise AssertionError("V1 and V2 admission sets diverged")
        grants = v2.grants_gbps
        residual_mode = RESIDUAL_MODE_EXPLICIT
    else:
        v2 = None
        grants = v1.grants_gbps
        residual_mode = RESIDUAL_MODE_COFLOW
    return AdaptiveAdmissionAllocation(
        grants_gbps=grants,
        selected_npu_ids=v1.selected_npu_ids,
        active_npu_count=active_count,
        selected_fraction=float(selected_fraction),
        explicit_spill_threshold=threshold,
        residual_mode=residual_mode,
        v1_allocation=v1,
        v2_allocation=v2,
    )


class AdaptiveAdmissionSchemeBControllerV2_1:
    """Full-manifest controller for adaptive V1/V2 residual selection."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        explicit_spill_threshold: float = 0.75,
        target_ratio: float = 0.52,
        required_ratio: float = 0.5,
        background_reserve_fraction: float = 0.05,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        if len(set(self.path_by_npu)) != len(self.path_by_npu):
            raise ValueError("adaptive admission requires one unique Path per NPU")
        threshold = float(explicit_spill_threshold)
        if not 0.0 < threshold <= 1.0 or not math.isfinite(threshold):
            raise ValueError(
                "explicit_spill_threshold must be finite and in (0, 1]"
            )
        self.explicit_spill_threshold = threshold
        self.target_ratio = float(target_ratio)
        self.required_ratio = float(required_ratio)
        self.background_reserve_fraction = float(background_reserve_fraction)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self.selected_request_by_npu: dict[int, int] = {}
        self.last_allocation: AdaptiveAdmissionAllocation | None = None
        self.evaluations = 0
        self.residual_mode_evaluations = {
            RESIDUAL_MODE_COFLOW: 0,
            RESIDUAL_MODE_EXPLICIT: 0,
        }

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
        allocation = allocate_adaptive_admission_grants(
            demands,
            explicit_spill_threshold=self.explicit_spill_threshold,
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
        self.residual_mode_evaluations[allocation.residual_mode] += 1
        return CIRControlDecision(tuple(tables))

