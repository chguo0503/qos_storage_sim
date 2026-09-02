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
from numbers import Real
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

SELECTION_MODE_PREFERRED = "all_preferred_targets_feasible"
SELECTION_MODE_REQUIRED = "all_required_targets_feasible"
SELECTION_MODE_GREEDY = "greedy_overload"

REJECTION_EMPTY_DEMAND = "empty_demand"
REJECTION_NPU_CAPACITY = "npu_target_exceeds_capacity"
REJECTION_SSU_CAPACITY = "ssu_admission_capacity_exceeded"

_EPS = 1e-12


@dataclass(frozen=True)
class AdmissionCandidateScore:
    """Stable multidimensional-packing score for one greedy candidate."""

    npu_id: int
    normalized_total: float
    normalized_dominant: float


@dataclass(frozen=True)
class AdmissionAttemptDiagnostic:
    """One exact invocation of the allocator's inner ``admit`` operation."""

    attempt_index: int
    npu_id: int
    stage: str
    accepted: bool
    rejection_reason: str | None
    target_gbps_by_ssu: tuple[float, ...]
    target_sum_gbps: float
    npu_capacity_gbps: float
    admission_remaining_before_gbps_by_ssu: tuple[float, ...]
    admission_remaining_after_gbps_by_ssu: tuple[float, ...]
    violating_ssu_ids: tuple[int, ...]


@dataclass(frozen=True)
class AdmissionSelectionReplay:
    """Pure replay of only the V1/V2 shared admission-selection stage."""

    selection_mode: str
    effective_target_ratio: float
    active_npu_ids: tuple[int, ...]
    candidate_scores: tuple[AdmissionCandidateScore, ...]
    candidate_order: tuple[int, ...]
    attempts: tuple[AdmissionAttemptDiagnostic, ...]
    selected_npu_ids: tuple[int, ...]
    rejected_npu_ids: tuple[int, ...]
    capacity_rejections: tuple[AdmissionAttemptDiagnostic, ...]


@dataclass(frozen=True)
class AdaptiveAdmissionDiagnostic:
    """One causally available, independently replayed controller evaluation."""

    snapshot_time_ms: float
    snapshot_evaluation: int
    trigger_reasons: tuple[str, ...]
    layer_jobs_since_previous: int
    request_by_npu: tuple[tuple[int, int], ...]
    prefetch_only_by_npu: tuple[tuple[int, bool], ...]
    remaining_work_gb_by_npu_ssu: Matrix
    remaining_compute_s_by_npu: tuple[float, ...]
    controller_demand_gbps_by_npu_ssu: Matrix
    previous_selected_request_by_npu: tuple[tuple[int, int], ...]
    previous_pinned_npu_ids: tuple[int, ...]
    selection_mode: str
    effective_target_ratio: float
    active_npu_ids: tuple[int, ...]
    candidate_normalized_scores: tuple[AdmissionCandidateScore, ...]
    candidate_order: tuple[int, ...]
    admission_attempts: tuple[AdmissionAttemptDiagnostic, ...]
    selected_npu_ids: tuple[int, ...]
    rejected_npu_ids: tuple[int, ...]
    capacity_rejections: tuple[AdmissionAttemptDiagnostic, ...]
    selected_fraction: float
    residual_mode: str
    grants_gbps_by_npu_ssu: GrantMatrix
    v2_effective_floor_ratio: float | None
    v2_floor_grants_gbps: GrantMatrix | None
    v2_background_grants_gbps: GrantMatrix | None
    v2_selected_tail_grants_gbps: GrantMatrix | None
    v2_spill_tail_grants_gbps: GrantMatrix | None


def _expanded_caps(spec: CapSpec, count: int, name: str) -> tuple[float, ...]:
    if isinstance(spec, Real):
        values = (float(spec),) * count
    else:
        values = tuple(float(value) for value in spec)
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    return values


def replay_admission_selection(
    demand: Matrix,
    *,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
) -> AdmissionSelectionReplay:
    """Replay the shared V1/V2 admission stage without changing any state.

    The implementation deliberately mirrors ``allocate_slo_admission_grants``
    and ``allocate_slo_admission_grants_v2``.  It stops before residual grant
    allocation and exposes every capacity decision made by their inner
    ``admit`` operation.
    """

    demand_array = np.asarray(demand, dtype=float)
    if demand_array.size == 0:
        return AdmissionSelectionReplay(
            selection_mode=SELECTION_MODE_PREFERRED,
            effective_target_ratio=float(target_ratio),
            active_npu_ids=(),
            candidate_scores=(),
            candidate_order=(),
            attempts=(),
            selected_npu_ids=(),
            rejected_npu_ids=(),
            capacity_rejections=(),
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
    ssd_limits = np.asarray(
        _expanded_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float
    )
    npu_limits = np.asarray(
        _expanded_caps(npu_caps, num_npu, "npu_caps"), dtype=float
    )
    pinned = tuple(dict.fromkeys(int(npu) for npu in pinned_npu_ids))
    if any(npu < 0 or npu >= num_npu for npu in pinned):
        raise ValueError("pinned_npu_ids contains an invalid NPU")

    preferred_targets = ratio * demand_array
    required_targets = required * demand_array
    active_npu_ids = tuple(
        int(npu) for npu in np.flatnonzero(demand_array.sum(axis=1) > _EPS)
    )

    def all_fit(targets: np.ndarray) -> bool:
        return bool(
            np.all(targets.sum(axis=0) <= ssd_limits + _EPS)
            and np.all(targets.sum(axis=1) <= npu_limits + _EPS)
        )

    if all_fit(preferred_targets):
        targets = preferred_targets
        selection_mode = SELECTION_MODE_PREFERRED
        effective_target_ratio = ratio
        selected = list(active_npu_ids)
    elif all_fit(required_targets):
        targets = required_targets
        selection_mode = SELECTION_MODE_REQUIRED
        effective_target_ratio = required
        selected = list(active_npu_ids)
    else:
        targets = preferred_targets
        selection_mode = SELECTION_MODE_GREEDY
        effective_target_ratio = ratio
        selected = []

    safe_ssd_limits = np.maximum(ssd_limits, _EPS)
    all_scores = tuple(
        AdmissionCandidateScore(
            npu_id=npu,
            normalized_total=float((targets[npu] / safe_ssd_limits).sum()),
            normalized_dominant=float(
                (targets[npu] / safe_ssd_limits).max(initial=0.0)
            ),
        )
        for npu in active_npu_ids
    )

    if selection_mode != SELECTION_MODE_GREEDY:
        return AdmissionSelectionReplay(
            selection_mode=selection_mode,
            effective_target_ratio=float(effective_target_ratio),
            active_npu_ids=active_npu_ids,
            candidate_scores=all_scores,
            candidate_order=(),
            attempts=(),
            selected_npu_ids=tuple(selected),
            rejected_npu_ids=(),
            capacity_rejections=(),
        )

    admission_remaining = (1.0 - reserve) * ssd_limits
    selected_set: set[int] = set()
    attempts: list[AdmissionAttemptDiagnostic] = []
    last_rejection_by_npu: dict[int, AdmissionAttemptDiagnostic] = {}

    def admit(npu_id: int, stage: str) -> bool:
        target = targets[npu_id]
        remaining_before = tuple(float(value) for value in admission_remaining)
        target_sum = float(target.sum())
        violating_ssu_ids: tuple[int, ...] = ()
        if demand_array[npu_id].sum() <= _EPS:
            rejection_reason = REJECTION_EMPTY_DEMAND
        elif target_sum > npu_limits[npu_id] + _EPS:
            rejection_reason = REJECTION_NPU_CAPACITY
        else:
            violating_ssu_ids = tuple(
                int(ssu)
                for ssu in np.flatnonzero(target > admission_remaining + _EPS)
            )
            rejection_reason = (
                REJECTION_SSU_CAPACITY if violating_ssu_ids else None
            )
        accepted = rejection_reason is None
        remaining_after = (
            tuple(
                float(remaining - amount)
                for remaining, amount in zip(admission_remaining, target)
            )
            if accepted
            else remaining_before
        )
        attempt = AdmissionAttemptDiagnostic(
            attempt_index=len(attempts),
            npu_id=npu_id,
            stage=stage,
            accepted=accepted,
            rejection_reason=rejection_reason,
            target_gbps_by_ssu=tuple(float(value) for value in target),
            target_sum_gbps=target_sum,
            npu_capacity_gbps=float(npu_limits[npu_id]),
            admission_remaining_before_gbps_by_ssu=remaining_before,
            admission_remaining_after_gbps_by_ssu=remaining_after,
            violating_ssu_ids=violating_ssu_ids,
        )
        attempts.append(attempt)
        if not accepted:
            last_rejection_by_npu[npu_id] = attempt
            return False
        admission_remaining[:] -= target
        selected.append(npu_id)
        selected_set.add(npu_id)
        last_rejection_by_npu.pop(npu_id, None)
        return True

    for npu_id in pinned:
        admit(npu_id, "pinned")

    candidate_scores = tuple(
        score
        for score in all_scores
        if score.npu_id not in selected_set
    )
    ordered_scores = tuple(
        sorted(
            candidate_scores,
            key=lambda score: (
                score.normalized_total,
                score.normalized_dominant,
                score.npu_id,
            ),
        )
    )
    candidate_order = tuple(score.npu_id for score in ordered_scores)
    for npu_id in candidate_order:
        admit(npu_id, "greedy_candidate")

    rejected_npu_ids = tuple(
        npu for npu in active_npu_ids if npu not in selected_set
    )
    capacity_rejections = tuple(
        last_rejection_by_npu[npu] for npu in rejected_npu_ids
    )
    return AdmissionSelectionReplay(
        selection_mode=selection_mode,
        effective_target_ratio=float(effective_target_ratio),
        active_npu_ids=active_npu_ids,
        candidate_scores=candidate_scores,
        candidate_order=candidate_order,
        attempts=tuple(attempts),
        selected_npu_ids=tuple(selected),
        rejected_npu_ids=rejected_npu_ids,
        capacity_rejections=capacity_rejections,
    )


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
        record_diagnostics: bool = False,
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
        self.record_diagnostics = bool(record_diagnostics)
        self.diagnostics: list[AdaptiveAdmissionDiagnostic] = []
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
        prefetch_only_by_npu: dict[int, bool] = {}
        for request in snapshot.active_requests:
            remaining_work, compute_ms = self._remaining_manifest(request)
            if compute_ms <= 0.0 or not any(value > 0.0 for value in remaining_work):
                continue
            if request.npu_id in request_by_npu:
                raise ValueError("admission controller requires batch-1 active coflows")
            request_by_npu[request.npu_id] = request.request_id
            prefetch_only_by_npu[request.npu_id] = bool(request.prefetch_only)
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
        previous_selected_request_by_npu = (
            tuple(
                sorted(
                    (int(npu), int(request_id))
                    for npu, request_id in self.selected_request_by_npu.items()
                )
            )
            if self.record_diagnostics
            else ()
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
        diagnostic: AdaptiveAdmissionDiagnostic | None = None
        if self.record_diagnostics:
            replay = replay_admission_selection(
                demands,
                target_ratio=self.target_ratio,
                required_ratio=self.required_ratio,
                background_reserve_fraction=self.background_reserve_fraction,
                pinned_npu_ids=pinned,
                ssd_caps=self.ssd_cap_gbps,
                npu_caps=self.npu_cap_gbps,
            )
            if replay.selected_npu_ids != allocation.selected_npu_ids:
                raise AssertionError(
                    "diagnostic admission replay diverged from allocator: "
                    f"replay={replay.selected_npu_ids}, "
                    f"allocator={allocation.selected_npu_ids}"
                )
            trigger_reasons_value = getattr(snapshot, "trigger_reasons", ())
            if trigger_reasons_value is None:
                trigger_reasons = ()
            elif isinstance(trigger_reasons_value, str):
                trigger_reasons = (trigger_reasons_value,)
            else:
                try:
                    trigger_reasons = tuple(
                        str(reason) for reason in trigger_reasons_value
                    )
                except TypeError:
                    trigger_reasons = (str(trigger_reasons_value),)
            v2 = allocation.v2_allocation
            diagnostic = AdaptiveAdmissionDiagnostic(
                snapshot_time_ms=float(snapshot.time_ms),
                snapshot_evaluation=int(snapshot.evaluation),
                trigger_reasons=trigger_reasons,
                layer_jobs_since_previous=int(snapshot.layer_jobs_since_previous),
                request_by_npu=tuple(
                    sorted(
                        (int(npu), int(request_id))
                        for npu, request_id in request_by_npu.items()
                    )
                ),
                prefetch_only_by_npu=tuple(
                    sorted(
                        (int(npu), bool(prefetch_only))
                        for npu, prefetch_only in prefetch_only_by_npu.items()
                    )
                ),
                remaining_work_gb_by_npu_ssu=tuple(
                    tuple(float(value) for value in row) for row in work
                ),
                remaining_compute_s_by_npu=tuple(
                    float(value) for value in compute_s
                ),
                controller_demand_gbps_by_npu_ssu=demands,
                previous_selected_request_by_npu=previous_selected_request_by_npu,
                previous_pinned_npu_ids=tuple(int(npu) for npu in pinned),
                selection_mode=replay.selection_mode,
                effective_target_ratio=replay.effective_target_ratio,
                active_npu_ids=replay.active_npu_ids,
                candidate_normalized_scores=replay.candidate_scores,
                candidate_order=replay.candidate_order,
                admission_attempts=replay.attempts,
                selected_npu_ids=tuple(
                    int(npu) for npu in allocation.selected_npu_ids
                ),
                rejected_npu_ids=replay.rejected_npu_ids,
                capacity_rejections=replay.capacity_rejections,
                selected_fraction=allocation.selected_fraction,
                residual_mode=allocation.residual_mode,
                grants_gbps_by_npu_ssu=allocation.grants_gbps,
                v2_effective_floor_ratio=(
                    v2.effective_floor_ratio if v2 is not None else None
                ),
                v2_floor_grants_gbps=(
                    v2.floor_grants_gbps if v2 is not None else None
                ),
                v2_background_grants_gbps=(
                    v2.background_grants_gbps if v2 is not None else None
                ),
                v2_selected_tail_grants_gbps=(
                    v2.selected_tail_grants_gbps if v2 is not None else None
                ),
                v2_spill_tail_grants_gbps=(
                    v2.spill_tail_grants_gbps if v2 is not None else None
                ),
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
        decision = CIRControlDecision(tuple(tables))
        if diagnostic is not None:
            self.diagnostics.append(diagnostic)
        return decision
