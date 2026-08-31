"""Deadline/barrier-aware Scheme B V3 prototype.

V1/V2 make a request-level threshold decision from an average remaining
manifest.  V3 keeps that deployment-visible manifest, but recomputes the
floor from the request's remaining TTFT time and gives the earliest floors to
requests that are already blocked at an I/O barrier.  The allocator is fully
separate from the frozen V1/V2 experiments.

The controller requires only :class:`ControlRequestView` fields and the
snapshot clock.  It never reads simulator-private queue state or byte
progress.  A production client can keep the same per-request clock when a
request is admitted.  This simulator adapter conservatively starts the clock
at the first periodic observation (which can be the cross-request Layer-0
prefetch observation).
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
class DeadlineBarrierAllocationV3:
    """An auditable floor/background/spill allocation."""

    grants_gbps: GrantMatrix
    selected_npu_ids: tuple[int, ...]
    rejected_npu_ids: tuple[int, ...]
    floor_grants_gbps: GrantMatrix
    background_grants_gbps: GrantMatrix
    spill_grants_gbps: GrantMatrix
    background_reserve_fraction: float


@dataclass(frozen=True)
class DeadlineRequestPlanV3:
    request_id: int
    npu_id: int
    prefetch_only: bool
    waiting_for_io: bool
    admission_clock_ms: float
    deadline_ms: float
    time_to_deadline_ms: float
    remaining_compute_ms: float
    target_ratio: float
    row_target_gbps: float
    row_raw_demand_gbps: float
    selected: bool


@dataclass(frozen=True)
class DeadlineBarrierPlanV3:
    time_ms: float
    request_plans: tuple[DeadlineRequestPlanV3, ...]
    allocation: DeadlineBarrierAllocationV3
    priority_npu_ids: tuple[int, ...]


@dataclass
class _RequestClock:
    npu_id: int
    first_observed_ms: float
    total_layers: int
    per_layer_compute_ms: float


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


def _matrix(values: np.ndarray) -> GrantMatrix:
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


def allocate_deadline_barrier_grants_v3(
    demand: Matrix,
    deadline_targets: Matrix,
    *,
    priority_npu_ids: Sequence[int] = (),
    background_reserve_fraction: float = 0.05,
    ssd_caps: CapSpec = 40.0,
    npu_caps: CapSpec = 50.0,
) -> DeadlineBarrierAllocationV3:
    """Pack deadline floors in priority order, then fill every useful tail.

    ``deadline_targets`` is already the controller's deadline/barrier floor.
    A row is capped proportionally at its NPU receive limit before packing.
    During overload, at most ``1-reserve`` of each SSD is consumed by selected
    floors.  Rejected coflows receive the bounded reserve, and a final absolute
    max-min spill consumes any capacity that either prior stage cannot use.
    """

    demand_array = np.asarray(demand, dtype=float)
    target_array = np.asarray(deadline_targets, dtype=float)
    if demand_array.size == 0:
        empty = tuple(tuple() for _ in range(len(demand)))
        return DeadlineBarrierAllocationV3(
            grants_gbps=empty,
            selected_npu_ids=(),
            rejected_npu_ids=(),
            floor_grants_gbps=empty,
            background_grants_gbps=empty,
            spill_grants_gbps=empty,
            background_reserve_fraction=float(background_reserve_fraction),
        )
    if demand_array.ndim != 2 or target_array.shape != demand_array.shape:
        raise ValueError("demand and deadline_targets must have one rectangular shape")
    if (
        np.any(demand_array < 0.0)
        or np.any(target_array < 0.0)
        or not np.all(np.isfinite(demand_array))
        or not np.all(np.isfinite(target_array))
    ):
        raise ValueError("demand and targets must contain finite non-negative values")

    reserve = float(background_reserve_fraction)
    if not 0.0 <= reserve < 1.0 or not math.isfinite(reserve):
        raise ValueError("background_reserve_fraction must be finite and in [0, 1)")

    num_npu, num_ssu = demand_array.shape
    ssd_limits = np.asarray(_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float)
    npu_limits = np.asarray(_caps(npu_caps, num_npu, "npu_caps"), dtype=float)
    priorities = tuple(dict.fromkeys(int(npu) for npu in priority_npu_ids))
    if any(npu < 0 or npu >= num_npu for npu in priorities):
        raise ValueError("priority_npu_ids contains an invalid NPU")
    priorities += tuple(npu for npu in range(num_npu) if npu not in priorities)

    targets = np.minimum(demand_array, target_array).copy()
    row_targets = targets.sum(axis=1)
    for npu_id, total in enumerate(row_targets):
        if total > npu_limits[npu_id] + _EPS:
            targets[npu_id] *= npu_limits[npu_id] / total

    active = tuple(np.flatnonzero(demand_array.sum(axis=1) > _EPS))
    selected: list[int] = []
    selected_set: set[int] = set()
    floor = np.zeros_like(demand_array)

    all_targets_fit = bool(
        np.all(targets.sum(axis=0) <= ssd_limits + _EPS)
        and np.all(targets.sum(axis=1) <= npu_limits + _EPS)
    )
    if all_targets_fit:
        selected = [int(npu) for npu in active]
        selected_set = set(selected)
        if selected:
            floor[selected] = targets[selected]
    else:
        floor_ssd_remaining = (1.0 - reserve) * ssd_limits
        for npu_id in priorities:
            target = targets[npu_id]
            if demand_array[npu_id].sum() <= _EPS or target.sum() <= _EPS:
                continue
            if np.any(target > floor_ssd_remaining + _EPS):
                continue
            floor[npu_id] = target
            floor_ssd_remaining -= target
            selected.append(npu_id)
            selected_set.add(npu_id)

    rejected = tuple(int(npu) for npu in active if npu not in selected_set)
    residual_ssd, residual_npu = _remaining_caps(floor, ssd_limits, npu_limits)
    background_caps = tuple(
        min(remaining, reserve * limit)
        for remaining, limit in zip(residual_ssd, ssd_limits)
    )
    rejected_demand = np.zeros_like(demand_array)
    if rejected:
        rejected_demand[list(rejected)] = demand_array[list(rejected)]
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
    residual_ssd, residual_npu = _remaining_caps(grants, ssd_limits, npu_limits)
    spill = np.asarray(
        allocate_grants(
            np.maximum(0.0, demand_array - grants),
            ssd_caps=residual_ssd,
            npu_caps=residual_npu,
        ),
        dtype=float,
    )
    grants += spill

    return DeadlineBarrierAllocationV3(
        grants_gbps=_matrix(grants),
        selected_npu_ids=tuple(selected),
        rejected_npu_ids=rejected,
        floor_grants_gbps=_matrix(floor),
        background_grants_gbps=_matrix(background),
        spill_grants_gbps=_matrix(spill),
        background_reserve_fraction=reserve,
    )


class DeadlineBarrierSchemeBControllerV3:
    """Periodic full-manifest controller for binary warm TTFT SLOs."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        slo_alpha: float = 2.0,
        safety_margin: float = 1.15,
        waiting_boost_ratio: float = 0.65,
        prefetch_target_ratio: float = 0.65,
        background_reserve_fraction: float = 0.05,
        min_decision_interval_ms: float = 25.0,
        hysteresis_gbps: float = 0.25,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        if len(set(self.path_by_npu)) != len(self.path_by_npu):
            raise ValueError("V3 requires one unique Path per NPU")
        values = {
            "slo_alpha": slo_alpha,
            "safety_margin": safety_margin,
            "waiting_boost_ratio": waiting_boost_ratio,
            "prefetch_target_ratio": prefetch_target_ratio,
            "min_decision_interval_ms": min_decision_interval_ms,
            "ssd_cap_gbps": ssd_cap_gbps,
            "npu_cap_gbps": npu_cap_gbps,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("V3 controller parameters must be finite")
        if slo_alpha <= 1.0:
            raise ValueError("slo_alpha must exceed one")
        if safety_margin <= 0.0:
            raise ValueError("safety_margin must be positive")
        if not 0.0 <= waiting_boost_ratio <= 1.0:
            raise ValueError("waiting_boost_ratio must be in [0, 1]")
        if not 0.0 <= prefetch_target_ratio <= 1.0:
            raise ValueError("prefetch_target_ratio must be in [0, 1]")
        if min_decision_interval_ms < 25.0:
            raise ValueError("V3 decision interval must be at least 25 ms")
        if hysteresis_gbps < 0.0 or not math.isfinite(float(hysteresis_gbps)):
            raise ValueError("hysteresis_gbps must be finite and non-negative")
        if ssd_cap_gbps <= 0.0 or npu_cap_gbps <= 0.0:
            raise ValueError("V3 physical capacities must be positive")

        self.slo_alpha = float(slo_alpha)
        self.safety_margin = float(safety_margin)
        self.waiting_boost_ratio = float(waiting_boost_ratio)
        self.prefetch_target_ratio = float(prefetch_target_ratio)
        self.background_reserve_fraction = float(background_reserve_fraction)
        self.min_decision_interval_ms = float(min_decision_interval_ms)
        self.hysteresis_gbps = float(hysteresis_gbps)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self._clocks: dict[int, _RequestClock] = {}
        self._selection_count_by_npu = [0] * len(self.path_by_npu)
        self._last_decision_ms: float | None = None
        self.last_plan: DeadlineBarrierPlanV3 | None = None
        self.evaluations = 0
        self.decisions = 0
        self.hysteresis_skips = 0
        self.rate_limit_skips = 0

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
        return tuple(float(value) for value in work), float(compute_ms)

    def _request_clock(
        self,
        request: ControlRequestView,
        time_ms: float,
    ) -> _RequestClock:
        clock = self._clocks.get(request.request_id)
        if clock is None:
            total_layers = max(
                1,
                int(request.remaining_layers) + int(request.compute_done_up_to) + 1,
            )
            clock = _RequestClock(
                npu_id=int(request.npu_id),
                first_observed_ms=float(time_ms),
                total_layers=total_layers,
                per_layer_compute_ms=float(request.per_layer_compute_ms),
            )
            self._clocks[request.request_id] = clock
        return clock

    def __call__(self, snapshot: CIRControlSnapshot):
        self.evaluations += 1
        if len(self.path_by_npu) != snapshot.num_npu:
            raise ValueError("dedicated Path mapping does not cover the fleet")

        active_ids = {request.request_id for request in snapshot.active_requests}
        self._clocks = {
            request_id: clock
            for request_id, clock in self._clocks.items()
            if request_id in active_ids
        }

        demand = np.zeros((snapshot.num_npu, snapshot.num_ssu), dtype=float)
        targets = np.zeros_like(demand)
        rows = []
        priority_records = []
        request_by_npu: dict[int, ControlRequestView] = {}
        derived = {}

        for request in snapshot.active_requests:
            remaining_work, remaining_compute_ms = self._remaining_manifest(request)
            if (
                remaining_compute_ms <= 0.0
                or len(remaining_work) != snapshot.num_ssu
                or not any(value > _EPS for value in remaining_work)
            ):
                continue
            if request.npu_id in request_by_npu:
                raise ValueError("V3 requires batch-1 active coflows")
            request_by_npu[request.npu_id] = request
            clock = self._request_clock(request, snapshot.time_ms)
            layer_s = request.per_layer_compute_ms / 1000.0
            if layer_s <= 0.0:
                continue
            raw = np.asarray(request.next_layer_work_gb_by_ssu, dtype=float) / layer_s
            if raw.shape != (snapshot.num_ssu,) or not np.any(raw > _EPS):
                remaining_layers = max(1.0, remaining_compute_ms / request.per_layer_compute_ms)
                raw = np.asarray(remaining_work, dtype=float) / remaining_layers / layer_s

            deadline_ms = clock.first_observed_ms + (
                self.slo_alpha
                * clock.total_layers
                * clock.per_layer_compute_ms
            )
            time_left_ms = deadline_ms - snapshot.time_ms
            if request.prefetch_only:
                target = self.prefetch_target_ratio * raw
                target_ratio = self.prefetch_target_ratio
            else:
                # The 25-ms observation uncertainty is intentionally handled by
                # clamping at one period, which turns near-deadline requests into
                # a full-rate target rather than declaring them irrecoverable.
                horizon_ms = max(time_left_ms, self.min_decision_interval_ms)
                target = (
                    self.safety_margin
                    * np.asarray(remaining_work, dtype=float)
                    / (horizon_ms / 1000.0)
                )
                target = np.minimum(raw, target)
                if request.waiting_for_io:
                    target = np.maximum(target, self.waiting_boost_ratio * raw)
                target_ratio = float(
                    target.sum() / raw.sum() if raw.sum() > _EPS else 0.0
                )

            demand[request.npu_id] = raw
            targets[request.npu_id] = target
            laxity_ms = time_left_ms - remaining_compute_ms
            # Active barrier wait first, then earliest remaining laxity/deadline.
            # Prefetch work uses spare admission slots after live TTFT clocks.
            priority_records.append(
                (
                    int(request.prefetch_only),
                    int(not request.waiting_for_io),
                    float(laxity_ms),
                    float(deadline_ms),
                    self._selection_count_by_npu[request.npu_id],
                    int(request.npu_id),
                )
            )
            derived[request.npu_id] = (
                request,
                clock,
                deadline_ms,
                time_left_ms,
                remaining_compute_ms,
                target_ratio,
                float(target.sum()),
                float(raw.sum()),
            )

        priorities = tuple(record[-1] for record in sorted(priority_records))
        allocation = allocate_deadline_barrier_grants_v3(
            demand,
            targets,
            priority_npu_ids=priorities,
            background_reserve_fraction=self.background_reserve_fraction,
            ssd_caps=self.ssd_cap_gbps,
            npu_caps=self.npu_cap_gbps,
        )
        selected = set(allocation.selected_npu_ids)
        for npu_id in selected:
            self._selection_count_by_npu[npu_id] += 1
        for npu_id in sorted(derived):
            (
                request,
                clock,
                deadline_ms,
                time_left_ms,
                remaining_compute_ms,
                target_ratio,
                row_target,
                row_raw,
            ) = derived[npu_id]
            rows.append(
                DeadlineRequestPlanV3(
                    request_id=int(request.request_id),
                    npu_id=npu_id,
                    prefetch_only=bool(request.prefetch_only),
                    waiting_for_io=bool(request.waiting_for_io),
                    admission_clock_ms=clock.first_observed_ms,
                    deadline_ms=deadline_ms,
                    time_to_deadline_ms=time_left_ms,
                    remaining_compute_ms=remaining_compute_ms,
                    target_ratio=target_ratio,
                    row_target_gbps=row_target,
                    row_raw_demand_gbps=row_raw,
                    selected=npu_id in selected,
                )
            )
        self.last_plan = DeadlineBarrierPlanV3(
            time_ms=float(snapshot.time_ms),
            request_plans=tuple(rows),
            allocation=allocation,
            priority_npu_ids=priorities,
        )

        if (
            self._last_decision_ms is not None
            and snapshot.time_ms
            < self._last_decision_ms + self.min_decision_interval_ms - _EPS
        ):
            self.rate_limit_skips += 1
            return None

        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        tables = []
        for ssu_id in range(snapshot.num_ssu):
            cirs = [0.0] * path_count
            for npu_id, path_id in enumerate(self.path_by_npu):
                cirs[path_id] = allocation.grants_gbps[npu_id][ssu_id]
            tables.append(tuple(cirs))
        decision = CIRControlDecision(tuple(tables))

        if self._last_decision_ms is not None:
            max_change = max(
                abs(new - old)
                for new_table, old_table in zip(
                    decision.path_cirs_by_ssu,
                    snapshot.current_path_cirs_by_ssu,
                )
                for new, old in zip(new_table, old_table)
            )
            if max_change < self.hysteresis_gbps - _EPS:
                self.hysteresis_skips += 1
                return None

        self._last_decision_ms = float(snapshot.time_ms)
        self.decisions += 1
        return decision

