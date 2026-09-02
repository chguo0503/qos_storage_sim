"""Protected-floor-only Scheme B (PFO-SB).

PFO-SB deliberately reuses the admission decision made by
``allocate_slo_admission_grants_v2`` but installs only that allocator's
``floor_grants_gbps`` on an immutable, controller-configured SSU mask.  The
default mask covers every SSU and therefore preserves the original PFO-SB
behavior.  A false mask entry leaves that SSU's complete CIR table at zero so
its dedicated Paths use work-conserving residual WRR.  In particular,
rejected-request background grants, selected-request tail grants, and residual
spill grants are never copied into the CIR target.  This makes the policy
question precise: can a small set of request-level SLO floors outperform
work-conserving residual allocation?

The controller uses only the public :class:`CIRControlSnapshot`: active request
manifests and the *actually installed* CIR table.  It performs no Path-pressure
read and has no timer.  Evaluation frequency must be supplied by the existing
``CIRControlConfig`` at integration time.

PFO-SB requires exclusive ownership of the complete Path CIR table.  Every Path
outside ``path_by_npu`` must already have a numerically-zero CIR.  Encountering
a positive non-dedicated CIR is treated as an ownership conflict and rejected;
the controller never silently erases another policy's reservation.

Deadband ownership is an important integration contract
---------------------------------------------------------
The 0.05 GB/s deadband is applied exactly once, in this module.  A small
increase is retained when the installed value is below a selected request's
``required_ratio`` hard floor.  Small decreases are retained when they are
needed to keep either an SSU or an NPU link within capacity.  The returned
``CIRControlDecision`` is therefore the complete, already-reconciled table.

``CIRControlDecision`` cannot carry per-entry ``forced`` flags or a register
write order.  Consequently, an integration using this controller **must set
the downstream simulator/hardware abstraction deadband to zero**.  Applying a
second 0.05 threshold could suppress a safety-forced small change.  The
``ordered_changes`` diagnostic exposes the deployable sequence for a real
non-atomic control plane: issue every decrease before any increase.  The DES
may still install the complete decision atomically.

All allocators and reconciliation helpers below are pure functions.  They do
not import or inspect simulator-private state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

import numpy as np

from continuous_batch_control import CapSpec, GrantMatrix, Matrix
from continuous_batch_sim import (
    CIRControlDecision,
    CIRControlSnapshot,
    ControlRequestView,
)
from slo_admission_scheme_b_v2 import allocate_slo_admission_grants_v2


DEFAULT_SSD_CAP_GBPS = 40.0
DEFAULT_NPU_CAP_GBPS = 50.0
DEFAULT_DEADBAND_GBPS = 0.05
MAX_DEADBAND_GBPS = 0.05
REQUIRED_DOWNSTREAM_DEADBAND_GBPS = 0.0

_EPS = 1e-12
_FLOOR_TOLERANCE = 1e-9
# Match the two independent downstream checks exactly: DiskIOScheduler uses
# sim._EPS for one SSU table, while the fleet-wide NPU-link check uses 1e-9.
_SSD_CAP_TOLERANCE = 1e-12
_NPU_CAP_TOLERANCE = 1e-9

PathCIRTable = tuple[tuple[float, ...], ...]
ChangePhase = Literal["decrease", "increase"]


@dataclass(frozen=True)
class ProtectedFloorAllocation:
    """Admission result whose only materialized grant is the selected floor.

    ``protected_floor_grants_gbps`` is byte-for-byte the V2 allocator's
    ``floor_grants_gbps``.  ``required_floor_grants_gbps`` is the lower hard
    floor used only by safety-aware deadband reconciliation.  Both matrices are
    NPU-major: ``matrix[npu_id][ssu_id]``.
    """

    selected_npu_ids: tuple[int, ...]
    target_ratio: float
    required_ratio: float
    effective_floor_ratio: float
    background_reserve_fraction: float
    protected_floor_grants_gbps: GrantMatrix
    required_floor_grants_gbps: GrantMatrix

    @property
    def grants_gbps(self) -> GrantMatrix:
        """Compatibility name: PFO grants are exactly the protected floors."""

        return self.protected_floor_grants_gbps


@dataclass(frozen=True)
class CIRRegisterChange:
    """One real register write in a capacity-safe global order."""

    sequence_index: int
    phase: ChangePhase
    ssu_id: int
    path_id: int
    npu_id: int | None
    old_gbps: float
    new_gbps: float
    delta_gbps: float
    reason: str
    safety_forced: bool
    would_pass_ordinary_deadband: bool


@dataclass(frozen=True)
class CIRDeadbandHold:
    """An ideal change deliberately held at its installed value."""

    ssu_id: int
    path_id: int
    npu_id: int | None
    installed_gbps: float
    ideal_gbps: float
    delta_gbps: float


@dataclass(frozen=True)
class PFOReconciliation:
    """Auditable result of reconciling ideal floors with installed CIRs."""

    deadband_gbps: float
    actual_path_cirs_by_ssu: PathCIRTable
    ideal_path_cirs_by_ssu: PathCIRTable
    required_path_cirs_by_ssu: PathCIRTable
    install_path_cirs_by_ssu: PathCIRTable
    ordered_changes: tuple[CIRRegisterChange, ...]
    deadband_holds: tuple[CIRDeadbandHold, ...]
    logical_forced_path_ids_by_ssu: tuple[tuple[int, ...], ...]
    required_floor_increase_count: int
    capacity_compensation_decrease_count: int
    actual_ssu_totals_gbps: tuple[float, ...]
    install_ssu_totals_gbps: tuple[float, ...]
    actual_npu_totals_gbps: tuple[float, ...]
    install_npu_totals_gbps: tuple[float, ...]
    pressure_reads: int = 0
    required_downstream_deadband_gbps: float = REQUIRED_DOWNSTREAM_DEADBAND_GBPS


@dataclass(frozen=True)
class PFORequestPlan:
    """Controller-visible request row used in one admission evaluation."""

    request_id: int
    npu_id: int
    raw_demand_gbps: float
    protected_floor_gbps: float
    required_floor_gbps: float
    was_pinned: bool
    selected: bool


@dataclass(frozen=True)
class PFOControllerPlan:
    """Complete public diagnostic for one PFO-SB callback."""

    time_ms: float
    evaluation: int
    materialized_ssu_mask: tuple[bool, ...]
    requests: tuple[PFORequestPlan, ...]
    allocation: ProtectedFloorAllocation
    reconciliation: PFOReconciliation


def _matrix_tuple(values: np.ndarray) -> GrantMatrix:
    return tuple(tuple(float(value) for value in row) for row in values)


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


def _rectangular_table(
    table: Sequence[Sequence[float]],
    *,
    row_count: int,
    column_count: int | None,
    name: str,
) -> tuple[PathCIRTable, int]:
    rows = tuple(tuple(float(value) for value in row) for row in table)
    if len(rows) != row_count:
        raise ValueError(f"{name} must have {row_count} SSU rows")
    if column_count is None:
        if not rows:
            raise ValueError(f"{name} cannot infer a Path count from zero SSUs")
        column_count = len(rows[0])
    if column_count <= 0 or any(len(row) != column_count for row in rows):
        raise ValueError(f"{name} must be rectangular with non-empty Path rows")
    if any(value < 0.0 or not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain finite non-negative values")
    return rows, column_count


def _ssu_capacity_totals(values: np.ndarray) -> np.ndarray:
    """Return SSU row totals with the downstream summation semantics."""

    return np.asarray(
        [math.fsum(float(value) for value in row) for row in values],
        dtype=float,
    )


def _npu_capacity_totals(
    values: np.ndarray,
    paths: Sequence[int],
) -> np.ndarray:
    """Return one cross-SSU total for every NPU-dedicated Path."""

    return np.asarray(
        [
            math.fsum(float(values[ssu_id, path_id]) for ssu_id in range(len(values)))
            for path_id in paths
        ],
        dtype=float,
    )


def _reject_non_dedicated_cirs(
    values: np.ndarray,
    paths: Sequence[int],
    *,
    name: str,
) -> None:
    """Reject a table that conflicts with PFO's exclusive Path ownership."""

    dedicated = frozenset(paths)
    for ssu_id, row in enumerate(values):
        for path_id, value in enumerate(row):
            if path_id not in dedicated and float(value) > _EPS:
                raise ValueError(
                    "PFO-SB requires exclusive ownership of the complete Path CIR "
                    f"table; {name}[{ssu_id}][{path_id}]={float(value)!r} is a "
                    "positive non-dedicated CIR"
                )


def allocate_protected_floor_grants(
    demand: Matrix,
    *,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps: CapSpec = DEFAULT_SSD_CAP_GBPS,
    npu_caps: CapSpec = DEFAULT_NPU_CAP_GBPS,
) -> ProtectedFloorAllocation:
    """Select requests exactly as Scheme B V2, then retain only its floor.

    ``background_reserve_fraction`` still participates in V2 admission
    packing, but no background bandwidth is materialized.  This is intentional:
    reserving headroom during selection keeps admission behavior comparable,
    while PFO's CIR objective remains floor-only.
    """

    demand_rows = tuple(tuple(float(value) for value in row) for row in demand)
    if demand_rows:
        num_ssu = len(demand_rows[0])
        if num_ssu <= 0 or any(len(row) != num_ssu for row in demand_rows):
            raise ValueError("demand must be a non-empty rectangular SSU matrix")
    else:
        num_ssu = 0

    source = allocate_slo_admission_grants_v2(
        demand_rows,
        target_ratio=target_ratio,
        required_ratio=required_ratio,
        background_reserve_fraction=background_reserve_fraction,
        pinned_npu_ids=pinned_npu_ids,
        ssd_caps=ssd_caps,
        npu_caps=npu_caps,
    )
    demand_array = np.asarray(demand_rows, dtype=float)
    if not demand_rows:
        required = np.empty((0, 0), dtype=float)
    else:
        required = np.zeros_like(demand_array)
        if source.selected_npu_ids:
            selected = list(source.selected_npu_ids)
            required[selected] = float(source.required_ratio) * demand_array[selected]

    protected = (
        np.asarray(source.floor_grants_gbps, dtype=float)
        if demand_rows
        else np.empty((0, 0), dtype=float)
    )
    if protected.shape != required.shape:
        raise AssertionError("V2 floor shape diverged from PFO demand shape")
    if np.any(required > protected + _FLOOR_TOLERANCE):
        raise AssertionError("protected floor fell below its required hard floor")
    if demand_rows:
        protected_ssu = _ssu_capacity_totals(protected.T)
        protected_npu = np.asarray(
            [math.fsum(float(value) for value in row) for row in protected],
            dtype=float,
        )
        ssd_limits = np.asarray(_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float)
        npu_limits = np.asarray(
            _caps(npu_caps, len(demand_rows), "npu_caps"),
            dtype=float,
        )
        if np.any(protected_ssu > ssd_limits + _SSD_CAP_TOLERANCE):
            raise AssertionError("V2 protected floor exceeds an SSU capacity")
        if np.any(protected_npu > npu_limits + _NPU_CAP_TOLERANCE):
            raise AssertionError("V2 protected floor exceeds an NPU capacity")

    # PFO intentionally copies no value from V2 background/tail/spill stages.
    return ProtectedFloorAllocation(
        selected_npu_ids=tuple(int(npu_id) for npu_id in source.selected_npu_ids),
        target_ratio=float(source.target_ratio),
        required_ratio=float(source.required_ratio),
        effective_floor_ratio=float(source.effective_floor_ratio),
        background_reserve_fraction=float(source.background_reserve_fraction),
        protected_floor_grants_gbps=_matrix_tuple(protected),
        required_floor_grants_gbps=_matrix_tuple(required),
    )


def _ordinary_deadband_selects(
    old_gbps: float,
    new_gbps: float,
    deadband_gbps: float,
) -> bool:
    delta = new_gbps - old_gbps
    if abs(delta) <= _EPS:
        return False
    if deadband_gbps <= 0.0:
        return True
    active_set_change = (old_gbps > 0.0) != (new_gbps > 0.0)
    return active_set_change or abs(delta) > deadband_gbps


def reconcile_protected_floor_cirs(
    actual_path_cirs_by_ssu: Sequence[Sequence[float]],
    ideal_path_cirs_by_ssu: Sequence[Sequence[float]],
    required_path_cirs_by_ssu: Sequence[Sequence[float]],
    *,
    path_by_npu: Sequence[int],
    deadband_gbps: float = DEFAULT_DEADBAND_GBPS,
    ssd_caps: CapSpec = DEFAULT_SSD_CAP_GBPS,
    npu_caps: CapSpec = DEFAULT_NPU_CAP_GBPS,
) -> PFOReconciliation:
    """Apply one safety-aware deadband to complete Path CIR tables.

    The returned install table is a component-wise mixture of the actual and
    ideal tables.  Ordinary changes pass on active-set transitions or when
    ``abs(delta) > deadband``.  Two changes bypass that rule:

    * an increase needed because actual CIR is below a selected hard floor;
    * a suppressed decrease needed to compensate accepted increases.

    Compensation is deterministic: largest release first, Path/SSU ID as the
    tie-breaker.  SSU constraints are repaired first; NPU-link repairs can only
    decrease an SSU total, so they cannot invalidate the first pass.
    """

    deadband = float(deadband_gbps)
    if not math.isfinite(deadband) or deadband < 0.0 or deadband > MAX_DEADBAND_GBPS:
        raise ValueError(
            f"deadband_gbps must be finite and in [0, {MAX_DEADBAND_GBPS}]"
        )
    paths = tuple(int(path_id) for path_id in path_by_npu)
    if len(set(paths)) != len(paths) or any(path_id < 0 for path_id in paths):
        raise ValueError("path_by_npu must contain unique non-negative Path IDs")

    actual, path_count = _rectangular_table(
        actual_path_cirs_by_ssu,
        row_count=len(actual_path_cirs_by_ssu),
        column_count=None,
        name="actual_path_cirs_by_ssu",
    )
    num_ssu = len(actual)
    ideal, _ = _rectangular_table(
        ideal_path_cirs_by_ssu,
        row_count=num_ssu,
        column_count=path_count,
        name="ideal_path_cirs_by_ssu",
    )
    required, _ = _rectangular_table(
        required_path_cirs_by_ssu,
        row_count=num_ssu,
        column_count=path_count,
        name="required_path_cirs_by_ssu",
    )
    if any(path_id >= path_count for path_id in paths):
        raise ValueError("path_by_npu references a Path outside the CIR table")

    ssd_limits = np.asarray(_caps(ssd_caps, num_ssu, "ssd_caps"), dtype=float)
    npu_limits = np.asarray(_caps(npu_caps, len(paths), "npu_caps"), dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    ideal_array = np.asarray(ideal, dtype=float)
    required_array = np.asarray(required, dtype=float)
    if np.any(required_array > ideal_array + _FLOOR_TOLERANCE):
        raise ValueError("required hard floor cannot exceed the ideal floor")

    # PFO owns a complete table but materializes grants only on the one
    # dedicated Path assigned to each NPU.  A positive value elsewhere means
    # another policy or tenant already owns that register; rejecting it is safer
    # than silently converting the complete desired table to zero there.
    _reject_non_dedicated_cirs(
        actual_array,
        paths,
        name="actual_path_cirs_by_ssu",
    )
    _reject_non_dedicated_cirs(
        ideal_array,
        paths,
        name="ideal_path_cirs_by_ssu",
    )
    _reject_non_dedicated_cirs(
        required_array,
        paths,
        name="required_path_cirs_by_ssu",
    )

    actual_ssu = _ssu_capacity_totals(actual_array)
    ideal_ssu = _ssu_capacity_totals(ideal_array)
    actual_npu = _npu_capacity_totals(actual_array, paths)
    ideal_npu = _npu_capacity_totals(ideal_array, paths)
    if np.any(actual_ssu > ssd_limits + _SSD_CAP_TOLERANCE):
        raise ValueError("actual installed table exceeds an SSU capacity")
    if np.any(ideal_ssu > ssd_limits + _SSD_CAP_TOLERANCE):
        raise ValueError("complete ideal table exceeds an SSU capacity")
    if np.any(actual_npu > npu_limits + _NPU_CAP_TOLERANCE):
        raise ValueError("actual installed table exceeds an NPU capacity")
    if np.any(ideal_npu > npu_limits + _NPU_CAP_TOLERANCE):
        raise ValueError("complete ideal table exceeds an NPU capacity")

    selected = np.zeros_like(actual_array, dtype=bool)
    reasons: dict[tuple[int, int], str] = {}
    safety_forced: set[tuple[int, int]] = set()
    ordinary_pass: dict[tuple[int, int], bool] = {}

    for ssu_id in range(num_ssu):
        for path_id in range(path_count):
            old = float(actual_array[ssu_id, path_id])
            new = float(ideal_array[ssu_id, path_id])
            passes = _ordinary_deadband_selects(old, new, deadband)
            ordinary_pass[(ssu_id, path_id)] = passes
            hard_floor_needed = old + _EPS < required_array[ssu_id, path_id]
            if hard_floor_needed:
                if new <= old + _EPS:
                    raise AssertionError("hard-floor repair is not an increase")
                selected[ssu_id, path_id] = True
                reasons[(ssu_id, path_id)] = "required_hard_floor"
                safety_forced.add((ssu_id, path_id))
            elif passes:
                selected[ssu_id, path_id] = True
                active_set_change = (old > 0.0) != (new > 0.0)
                reasons[(ssu_id, path_id)] = (
                    "active_set_change" if active_set_change else "deadband_exceeded"
                )

    install = np.where(selected, ideal_array, actual_array)

    def force_decrease(ssu_id: int, path_id: int, reason: str) -> None:
        if selected[ssu_id, path_id]:
            return
        if ideal_array[ssu_id, path_id] >= actual_array[ssu_id, path_id] - _EPS:
            raise AssertionError("capacity compensation must release bandwidth")
        selected[ssu_id, path_id] = True
        install[ssu_id, path_id] = ideal_array[ssu_id, path_id]
        reasons[(ssu_id, path_id)] = reason
        safety_forced.add((ssu_id, path_id))

    # First repair each SSU.  Candidate releases that ordinary deadband already
    # selected are already reflected in ``install`` and need no forced flag.
    for ssu_id in range(num_ssu):
        if (
            math.fsum(float(value) for value in install[ssu_id])
            <= ssd_limits[ssu_id] + _SSD_CAP_TOLERANCE
        ):
            continue
        candidates = [
            path_id
            for path_id in range(path_count)
            if not selected[ssu_id, path_id]
            and ideal_array[ssu_id, path_id] < actual_array[ssu_id, path_id] - _EPS
        ]
        candidates.sort(
            key=lambda path_id: (
                -(actual_array[ssu_id, path_id] - ideal_array[ssu_id, path_id]),
                path_id,
            )
        )
        for path_id in candidates:
            force_decrease(ssu_id, path_id, "ssd_capacity_compensation")
            if (
                math.fsum(float(value) for value in install[ssu_id])
                <= ssd_limits[ssu_id] + _SSD_CAP_TOLERANCE
            ):
                break
        if (
            math.fsum(float(value) for value in install[ssu_id])
            > ssd_limits[ssu_id] + _SSD_CAP_TOLERANCE
        ):
            raise AssertionError("deadband reconciliation cannot repair SSU capacity")

    # A later NPU repair only lowers SSU totals, so the pass ordering is safe.
    for npu_id, path_id in enumerate(paths):
        if (
            math.fsum(float(install[ssu_id, path_id]) for ssu_id in range(num_ssu))
            <= npu_limits[npu_id] + _NPU_CAP_TOLERANCE
        ):
            continue
        candidates = [
            ssu_id
            for ssu_id in range(num_ssu)
            if not selected[ssu_id, path_id]
            and ideal_array[ssu_id, path_id] < actual_array[ssu_id, path_id] - _EPS
        ]
        candidates.sort(
            key=lambda ssu_id: (
                -(actual_array[ssu_id, path_id] - ideal_array[ssu_id, path_id]),
                ssu_id,
            )
        )
        for ssu_id in candidates:
            force_decrease(ssu_id, path_id, "npu_capacity_compensation")
            if (
                math.fsum(
                    float(install[check_ssu, path_id]) for check_ssu in range(num_ssu)
                )
                <= npu_limits[npu_id] + _NPU_CAP_TOLERANCE
            ):
                break
        if (
            math.fsum(float(install[ssu_id, path_id]) for ssu_id in range(num_ssu))
            > npu_limits[npu_id] + _NPU_CAP_TOLERANCE
        ):
            raise AssertionError("deadband reconciliation cannot repair NPU capacity")

    install_ssu = _ssu_capacity_totals(install)
    install_npu = _npu_capacity_totals(install, paths)
    if np.any(install_ssu > ssd_limits + _SSD_CAP_TOLERANCE):
        raise AssertionError("reconciled install table exceeds an SSU capacity")
    if np.any(install_npu > npu_limits + _NPU_CAP_TOLERANCE):
        raise AssertionError("reconciled install table exceeds an NPU capacity")
    if np.any(install + _FLOOR_TOLERANCE < required_array):
        raise AssertionError("reconciled install table misses a required hard floor")

    path_to_npu = {path_id: npu_id for npu_id, path_id in enumerate(paths)}
    raw_changes = []
    holds = []
    for ssu_id in range(num_ssu):
        for path_id in range(path_count):
            old = float(actual_array[ssu_id, path_id])
            new = float(install[ssu_id, path_id])
            ideal_value = float(ideal_array[ssu_id, path_id])
            if abs(new - old) > _EPS:
                delta = new - old
                phase: ChangePhase = "decrease" if delta < 0.0 else "increase"
                raw_changes.append(
                    (
                        0 if phase == "decrease" else 1,
                        ssu_id,
                        path_id,
                        phase,
                        old,
                        new,
                        delta,
                        reasons[(ssu_id, path_id)],
                        (ssu_id, path_id) in safety_forced,
                        ordinary_pass[(ssu_id, path_id)],
                    )
                )
            elif abs(ideal_value - old) > _EPS:
                holds.append(
                    CIRDeadbandHold(
                        ssu_id=ssu_id,
                        path_id=path_id,
                        npu_id=path_to_npu.get(path_id),
                        installed_gbps=old,
                        ideal_gbps=ideal_value,
                        delta_gbps=ideal_value - old,
                    )
                )

    changes = tuple(
        CIRRegisterChange(
            sequence_index=index,
            phase=phase,
            ssu_id=ssu_id,
            path_id=path_id,
            npu_id=path_to_npu.get(path_id),
            old_gbps=old,
            new_gbps=new,
            delta_gbps=delta,
            reason=reason,
            safety_forced=forced,
            would_pass_ordinary_deadband=passes,
        )
        for index, (
            _,
            ssu_id,
            path_id,
            phase,
            old,
            new,
            delta,
            reason,
            forced,
            passes,
        ) in enumerate(sorted(raw_changes))
    )
    forced_by_ssu = tuple(
        tuple(
            sorted(
                path_id for forced_ssu, path_id in safety_forced if forced_ssu == ssu_id
            )
        )
        for ssu_id in range(num_ssu)
    )
    required_increases = sum(
        change.reason == "required_hard_floor" for change in changes
    )
    compensating_decreases = sum(
        change.reason in ("ssd_capacity_compensation", "npu_capacity_compensation")
        for change in changes
    )

    return PFOReconciliation(
        deadband_gbps=deadband,
        actual_path_cirs_by_ssu=_matrix_tuple(actual_array),
        ideal_path_cirs_by_ssu=_matrix_tuple(ideal_array),
        required_path_cirs_by_ssu=_matrix_tuple(required_array),
        install_path_cirs_by_ssu=_matrix_tuple(install),
        ordered_changes=changes,
        deadband_holds=tuple(holds),
        logical_forced_path_ids_by_ssu=forced_by_ssu,
        required_floor_increase_count=required_increases,
        capacity_compensation_decrease_count=compensating_decreases,
        actual_ssu_totals_gbps=tuple(float(value) for value in actual_ssu),
        install_ssu_totals_gbps=tuple(float(value) for value in install_ssu),
        actual_npu_totals_gbps=tuple(float(value) for value in actual_npu),
        install_npu_totals_gbps=tuple(float(value) for value in install_npu),
    )


class ProtectedFloorSchemeBController:
    """Public-snapshot adapter for PFO-SB.

    The callback performs one evaluation whenever its caller invokes it.  It
    intentionally contains no wall-clock gate, layer counter, event hook, or
    pressure read.  Configure cadence with ``CIRControlConfig``.  Request pinning
    is stable by ``(npu_id, request_id)`` and lasts until that request leaves the
    active manifest or can no longer be selected feasibly.

    Integration must use ``cir_write_threshold_gbps=0`` downstream because the
    controller already emits an internally deadbanded table.  See the module
    docstring and ``last_plan.reconciliation.ordered_changes``.

    ``materialized_ssu_mask`` is copied to a tuple at construction and must
    contain exactly one boolean per SSU when supplied.  ``None`` is the
    backwards-compatible all-true mask, resolved against the snapshot topology
    and recorded explicitly in every :class:`PFOControllerPlan`.
    """

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        target_ratio: float = 0.52,
        required_ratio: float = 0.5,
        background_reserve_fraction: float = 0.05,
        deadband_gbps: float = DEFAULT_DEADBAND_GBPS,
        ssd_cap_gbps: CapSpec = DEFAULT_SSD_CAP_GBPS,
        npu_cap_gbps: CapSpec = DEFAULT_NPU_CAP_GBPS,
        materialized_ssu_mask: Sequence[bool] | None = None,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        if len(set(self.path_by_npu)) != len(self.path_by_npu) or any(
            path_id < 0 for path_id in self.path_by_npu
        ):
            raise ValueError("PFO-SB requires one unique non-negative Path per NPU")
        self.target_ratio = float(target_ratio)
        self.required_ratio = float(required_ratio)
        self.background_reserve_fraction = float(background_reserve_fraction)
        self.deadband_gbps = float(deadband_gbps)
        if (
            not math.isfinite(self.deadband_gbps)
            or self.deadband_gbps < 0.0
            or self.deadband_gbps > MAX_DEADBAND_GBPS
        ):
            raise ValueError(
                f"deadband_gbps must be finite and in [0, {MAX_DEADBAND_GBPS}]"
            )
        self.ssd_cap_gbps = ssd_cap_gbps
        self.npu_cap_gbps = npu_cap_gbps
        if materialized_ssu_mask is None:
            self._materialized_ssu_mask: tuple[bool, ...] | None = None
        else:
            configured_mask = tuple(materialized_ssu_mask)
            if not configured_mask or any(
                not isinstance(value, bool) for value in configured_mask
            ):
                raise ValueError(
                    "materialized_ssu_mask must contain one boolean per SSU"
                )
            # Copy caller-owned sequences once.  The controller never mutates
            # or re-resolves a configured mask after construction.
            self._materialized_ssu_mask = configured_mask

        self.selected_request_by_npu: dict[int, int] = {}
        self.last_plan: PFOControllerPlan | None = None
        self.evaluations = 0
        self.decisions = 0
        self.planned_register_writes = 0
        self.safety_forced_writes = 0
        # This counter is deliberately immutable in behavior: PFO never calls
        # the pressure-reporting interface.
        self.path_pressure_reads = 0

    @property
    def materialized_ssu_mask(self) -> tuple[bool, ...] | None:
        """Return the immutable configured mask; ``None`` means all SSUs."""

        return self._materialized_ssu_mask

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

    def __call__(self, snapshot: CIRControlSnapshot) -> CIRControlDecision:
        self.evaluations += 1
        if len(self.path_by_npu) != snapshot.num_npu:
            raise ValueError("dedicated Path mapping does not cover the fleet")
        if snapshot.num_ssu <= 0:
            raise ValueError("PFO-SB requires at least one SSU")
        materialized_ssu_mask = (
            (True,) * snapshot.num_ssu
            if self.materialized_ssu_mask is None
            else self.materialized_ssu_mask
        )
        if len(materialized_ssu_mask) != snapshot.num_ssu:
            raise ValueError(
                "materialized_ssu_mask must contain exactly one entry per SSU"
            )

        current, path_count = _rectangular_table(
            snapshot.current_path_cirs_by_ssu,
            row_count=snapshot.num_ssu,
            column_count=None,
            name="snapshot.current_path_cirs_by_ssu",
        )
        if any(path_id >= path_count for path_id in self.path_by_npu):
            raise ValueError("dedicated Path mapping is outside the snapshot table")
        _reject_non_dedicated_cirs(
            np.asarray(current, dtype=float),
            self.path_by_npu,
            name="snapshot.current_path_cirs_by_ssu",
        )

        work = np.zeros((snapshot.num_npu, snapshot.num_ssu), dtype=float)
        compute_s = np.zeros(snapshot.num_npu, dtype=float)
        request_by_npu: dict[int, int] = {}
        requests: dict[int, ControlRequestView] = {}
        for request in snapshot.active_requests:
            remaining_work, compute_ms = self._remaining_manifest(request)
            if len(remaining_work) != snapshot.num_ssu:
                raise ValueError("active request manifest does not cover every SSU")
            if (
                any(value < 0.0 or not math.isfinite(value) for value in remaining_work)
                or not math.isfinite(compute_ms)
                or compute_ms < 0.0
            ):
                raise ValueError(
                    "active request manifest must be finite and non-negative"
                )
            if compute_ms <= 0.0 or not any(value > _EPS for value in remaining_work):
                continue
            if request.npu_id < 0 or request.npu_id >= snapshot.num_npu:
                raise ValueError("active request references an invalid NPU")
            if request.npu_id in request_by_npu:
                raise ValueError("PFO-SB requires batch-1 active coflows")
            request_by_npu[request.npu_id] = int(request.request_id)
            requests[request.npu_id] = request
            compute_s[request.npu_id] = compute_ms / 1000.0
            work[request.npu_id] = remaining_work

        demand = np.divide(
            work,
            compute_s[:, None],
            out=np.zeros_like(work),
            where=compute_s[:, None] > 0.0,
        )
        pinned = tuple(
            npu_id
            for npu_id, request_id in self.selected_request_by_npu.items()
            if request_by_npu.get(npu_id) == request_id
        )
        allocation = allocate_protected_floor_grants(
            _matrix_tuple(demand),
            target_ratio=self.target_ratio,
            required_ratio=self.required_ratio,
            background_reserve_fraction=self.background_reserve_fraction,
            pinned_npu_ids=pinned,
            ssd_caps=self.ssd_cap_gbps,
            npu_caps=self.npu_cap_gbps,
        )
        next_selected_request_by_npu = {
            npu_id: request_by_npu[npu_id]
            for npu_id in allocation.selected_npu_ids
            if npu_id in request_by_npu
        }

        ideal = np.zeros((snapshot.num_ssu, path_count), dtype=float)
        required = np.zeros_like(ideal)
        materialized = np.asarray(materialized_ssu_mask, dtype=bool)
        materialize_all = all(materialized_ssu_mask)
        for npu_id, path_id in enumerate(self.path_by_npu):
            protected_values = np.asarray(
                allocation.protected_floor_grants_gbps[npu_id], dtype=float
            )
            required_values = np.asarray(
                allocation.required_floor_grants_gbps[npu_id], dtype=float
            )
            if materialize_all:
                # Preserve the original all-SSU assignment path exactly.
                ideal[:, path_id] = protected_values
                required[:, path_id] = required_values
            else:
                ideal[materialized, path_id] = protected_values[materialized]
                required[materialized, path_id] = required_values[materialized]

        reconciliation = reconcile_protected_floor_cirs(
            current,
            _matrix_tuple(ideal),
            _matrix_tuple(required),
            path_by_npu=self.path_by_npu,
            deadband_gbps=self.deadband_gbps,
            ssd_caps=self.ssd_cap_gbps,
            npu_caps=self.npu_cap_gbps,
        )
        selected = set(allocation.selected_npu_ids)
        request_rows = tuple(
            PFORequestPlan(
                request_id=int(request_by_npu[npu_id]),
                npu_id=npu_id,
                raw_demand_gbps=math.fsum(float(value) for value in demand[npu_id]),
                protected_floor_gbps=float(
                    math.fsum(allocation.protected_floor_grants_gbps[npu_id])
                ),
                required_floor_gbps=float(
                    math.fsum(allocation.required_floor_grants_gbps[npu_id])
                ),
                was_pinned=npu_id in pinned,
                selected=npu_id in selected,
            )
            for npu_id in sorted(requests)
        )
        plan = PFOControllerPlan(
            time_ms=float(snapshot.time_ms),
            evaluation=int(snapshot.evaluation),
            materialized_ssu_mask=materialized_ssu_mask,
            requests=request_rows,
            allocation=allocation,
            reconciliation=reconciliation,
        )
        self.selected_request_by_npu = next_selected_request_by_npu
        self.last_plan = plan
        self.decisions += 1
        self.planned_register_writes += len(reconciliation.ordered_changes)
        self.safety_forced_writes += sum(
            change.safety_forced for change in reconciliation.ordered_changes
        )
        if self.path_pressure_reads != 0:
            raise AssertionError("PFO-SB must not read Path pressure")
        return CIRControlDecision(reconciliation.install_path_cirs_by_ssu)


__all__ = (
    "CIRDeadbandHold",
    "CIRRegisterChange",
    "DEFAULT_DEADBAND_GBPS",
    "DEFAULT_NPU_CAP_GBPS",
    "DEFAULT_SSD_CAP_GBPS",
    "MAX_DEADBAND_GBPS",
    "PFOControllerPlan",
    "PFOReconciliation",
    "PFORequestPlan",
    "ProtectedFloorAllocation",
    "ProtectedFloorSchemeBController",
    "REQUIRED_DOWNSTREAM_DEADBAND_GBPS",
    "allocate_protected_floor_grants",
    "reconcile_protected_floor_cirs",
)
