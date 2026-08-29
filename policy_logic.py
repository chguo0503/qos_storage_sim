"""Pure client/control-plane policies for the retained QoS experiments.

This module deliberately has no dependency on :mod:`sim`, an event queue, or
simulation time.  A real client or another simulator supplies immutable
snapshots and consumes the returned Path IDs / CIR table.

Public policy boundary
----------------------
* baseline: :func:`baseline_path_ids`
* layer-once: :func:`layer_once_path_ids`
* refresh8: :func:`refresh8_path_ids`
* Scheme B: :func:`plan_scheme_b` and :func:`plan_causal_scheme_b`
* feasible oracle comparator: :func:`oracle_priority_key`

The SSD/NPU data plane remains outside this module.  In particular, these
functions do not enqueue I/O, advance a clock, or predict an event completion.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence

from continuous_batch_control import GrantMatrix, allocate_grants


PATH_COUNT = 256
GROUP_COUNT = 8
PATHS_PER_GROUP = PATH_COUNT // GROUP_COUNT
MAX_NPU = 128
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0
REFRESH8_WINDOW_IO = 8
ORACLE_DEMAND_EXPONENT = 0.25
_EPS = 1e-12

__all__ = (
    "REFRESH8_WINDOW_IO",
    "QoSHardwareView",
    "PathPressureSnapshot",
    "ManifestDemand",
    "SchemeBPlan",
    "CausalLayerObservation",
    "OracleFlowView",
    "hardware_view",
    "pressure_snapshot",
    "category_path_ids",
    "baseline_path_ids",
    "pressure_aware_path_ids",
    "layer_once_path_ids",
    "refresh8_path_ids",
    "dedicated_path_id",
    "cold_start_hybrid_path_id",
    "plan_scheme_b",
    "plan_causal_scheme_b",
    "oracle_priority_key",
    "validate_scheme_b_plan",
)


@dataclass(frozen=True)
class QoSHardwareView:
    """Read-only QoS registers needed by client-side Path selection."""

    path_cirs: tuple[float, ...]
    path_pirs: tuple[float, ...]
    path_weights: tuple[float, ...]
    group_weights: tuple[float, ...]
    category_paths_per_group: tuple[int, ...]
    category_labels: tuple[str, ...] = ("SS", "SL", "LS", "LL")

    def __post_init__(self):
        path_count = len(self.path_cirs)
        if not (
            path_count
            == len(self.path_pirs)
            == len(self.path_weights)
        ):
            raise ValueError("Path CIR/PIR/weight vectors must have equal length")
        if not self.group_weights or path_count % len(self.group_weights):
            raise ValueError("Path count must divide evenly into QoS groups")
        if sum(self.category_paths_per_group) != self.paths_per_group:
            raise ValueError("routing categories must fill one QoS group")
        if len(self.category_paths_per_group) != len(self.category_labels):
            raise ValueError("category labels and widths must have equal length")

    @property
    def path_count(self) -> int:
        return len(self.path_cirs)

    @property
    def group_count(self) -> int:
        return len(self.group_weights)

    @property
    def paths_per_group(self) -> int:
        return self.path_count // self.group_count


@dataclass(frozen=True)
class PathPressureSnapshot:
    """Count-only telemetry returned by one SSU pressure-table read."""

    counts: tuple[int, ...]
    group_io_counts: tuple[int, ...]
    active_paths_per_group: tuple[int, ...]
    active_path_weights: tuple[float, ...]
    active_group_weight_sum: float
    active_cir_sum: float


@dataclass(frozen=True)
class PathEstimate:
    path_id: int
    finish_time_s: float
    effective_path_rate_gbps: float
    near_term_rate_gbps: float
    estimated_old_backlog_gb: float


@dataclass(frozen=True)
class ManifestDemand:
    """One active request/microbatch contribution visible to Scheme B."""

    request_id: int
    npu_id: int
    compute_budget_s: float
    work_by_ssu_gb: tuple[float, ...]


@dataclass(frozen=True)
class SchemeBPlan:
    """Deployment-neutral output of one Scheme B logical control epoch."""

    active_request_ids: tuple[int, ...]
    demands_gbps: GrantMatrix
    grants_gbps: GrantMatrix
    path_by_npu: tuple[int, ...]
    path_cirs_by_ssu: tuple[tuple[float, ...], ...]
    target_hash: str


@dataclass(frozen=True)
class CausalLayerObservation:
    """Previous-layer information available without future knowledge."""

    request_id: int
    npu_id: int
    observed_layer: int
    compute_budget_ms: float
    observed_work_gb_by_ssu: tuple[float, ...]


@dataclass(frozen=True)
class OracleFlowView:
    """Released-I/O metadata consumed by the feasible oracle comparator."""

    demand_gbps: float
    layer_work_gb: float
    deadline_time: float
    enqueue_order: float
    request_id: int
    layer: int
    block_idx: int
    ssu_id: int


def hardware_view(config) -> QoSHardwareView:
    """Copy any structurally compatible QoS register object into a pure view."""
    return QoSHardwareView(
        path_cirs=tuple(map(float, config.path_cirs)),
        path_pirs=tuple(map(float, config.path_pirs)),
        path_weights=tuple(map(float, config.path_weights)),
        group_weights=tuple(map(float, config.group_weights)),
        category_paths_per_group=tuple(config.category_paths_per_group),
        category_labels=tuple(config.category_labels),
    )


def pressure_snapshot(snapshot) -> PathPressureSnapshot:
    """Copy a hardware/simulator pressure report into the public policy ABI."""
    return PathPressureSnapshot(
        counts=tuple(snapshot.counts),
        group_io_counts=tuple(snapshot.group_io_counts),
        active_paths_per_group=tuple(snapshot.active_paths_per_group),
        active_path_weights=tuple(map(float, snapshot.active_path_weights)),
        active_group_weight_sum=float(snapshot.active_group_weight_sum),
        active_cir_sum=float(snapshot.active_cir_sum),
    )


def category_path_ids(category: str, qos: QoSHardwareView) -> tuple[int, ...]:
    """Return category-legal Path IDs in the legacy local-offset order."""
    category_index = qos.category_labels.index(category)
    category_offset = sum(qos.category_paths_per_group[:category_index])
    category_count = qos.category_paths_per_group[category_index]
    return tuple(
        group_id * qos.paths_per_group + category_offset + local_offset
        for local_offset in range(category_count)
        for group_id in range(qos.group_count)
    )


def baseline_path_ids(io_count: int, path_id: int = 0) -> tuple[int, ...]:
    """Final baseline: every I/O uses one fixed Path and reads no pressure."""
    return (int(path_id),) * int(io_count)


def _projection_choices(
    *,
    block_size_gb: float,
    representative_block_gb: float,
    allowed_path_ids: Sequence[int],
    qos: QoSHardwareView,
    disk_bw_gbps: float,
    counts: Sequence[int],
    group_io_counts: Sequence[int],
    active_paths_per_group: Sequence[int],
    active_path_weights: Sequence[float],
    active_group_weight_sum: float,
    active_cir_sum: float,
    projected_work_gb: Sequence[float] | None = None,
):
    allowed = tuple(allowed_path_ids)
    best_primary = None
    choices = []
    equivalence = {}
    for allowed_index, path_id in enumerate(allowed):
        group_id = path_id // qos.paths_per_group
        was_empty = counts[path_id] == 0
        old_work = (
            projected_work_gb[path_id]
            if projected_work_gb is not None
            else counts[path_id] * representative_block_gb
        )
        rate_class = (
            group_id,
            was_empty,
            qos.path_cirs[path_id],
            qos.path_pirs[path_id],
            qos.path_weights[path_id],
        )
        dominance_rank = (old_work, counts[path_id])
        record = (allowed_index, path_id, old_work, counts[path_id])
        current = equivalence.get(rate_class)
        if current is None or dominance_rank < current[0]:
            equivalence[rate_class] = (dominance_rank, [record])
        elif dominance_rank == current[0]:
            current[1].append(record)

    for rate_class, (_, equivalent_paths) in equivalence.items():
        group_id, was_empty, path_cir, path_pir, path_weight = rate_class
        group_weight = qos.group_weights[group_id]
        base_rate = min(path_cir, path_pir)
        cir_sum = active_cir_sum + (base_rate if was_empty else 0.0)
        group_weight_sum = active_group_weight_sum
        path_weight_sum = active_path_weights[group_id]
        if was_empty:
            path_weight_sum += path_weight
            if active_paths_per_group[group_id] == 0:
                group_weight_sum += group_weight
        if group_weight_sum <= _EPS or path_weight_sum <= _EPS:
            continue
        remaining = max(0.0, disk_bw_gbps - cir_sum)
        group_extra = remaining * group_weight / group_weight_sum
        path_extra = group_extra * path_weight / path_weight_sum
        rate = min(path_pir, base_rate + path_extra)
        if rate <= _EPS:
            continue
        _, _, old_work, path_count = equivalent_paths[0]
        finish_s = (old_work + block_size_gb) / rate
        primary = (finish_s, path_count, group_io_counts[group_id])
        equivalent_choices = [
            (index, candidate_path, finish_s, rate, candidate_old_work)
            for index, candidate_path, candidate_old_work, _ in equivalent_paths
        ]
        if best_primary is None or primary < best_primary:
            best_primary = primary
            choices = equivalent_choices
        elif primary == best_primary:
            choices.extend(equivalent_choices)

    if not choices:
        best_fallback = None
        for allowed_index, path_id in enumerate(allowed):
            group_id = path_id // qos.paths_per_group
            old_work = (
                projected_work_gb[path_id]
                if projected_work_gb is not None
                else counts[path_id] * representative_block_gb
            )
            primary = (old_work, counts[path_id], group_io_counts[group_id])
            record = (allowed_index, path_id, float("inf"), 0.0, old_work)
            if best_fallback is None or primary < best_fallback:
                best_fallback = primary
                choices = [record]
            elif primary == best_fallback:
                choices.append(record)
    choices.sort(key=lambda choice: choice[0])
    return tuple(choice[0] for choice in choices), tuple(choices)


def _select_projection_choice(
    choices,
    *,
    allowed_count: int,
    start_offset: int,
    block_size_gb: float,
    disk_bw_gbps: float,
) -> PathEstimate:
    offset = start_offset % allowed_count
    allowed_indices, records = choices
    position = bisect.bisect_left(allowed_indices, offset)
    if position == len(records):
        position = 0
    selected = records[position]
    return PathEstimate(
        path_id=selected[1],
        finish_time_s=selected[2],
        effective_path_rate_gbps=selected[3],
        near_term_rate_gbps=min(
            disk_bw_gbps,
            block_size_gb / max(selected[2], _EPS),
        ),
        estimated_old_backlog_gb=selected[4],
    )


class _PlanningShadow:
    def __init__(
        self,
        snapshot: PathPressureSnapshot,
        representative_block_gb: float,
    ):
        self.counts = list(snapshot.counts)
        self.group_io_counts = list(snapshot.group_io_counts)
        self.active_paths_per_group = list(snapshot.active_paths_per_group)
        self.active_path_weights = list(snapshot.active_path_weights)
        self.active_group_weight_sum = snapshot.active_group_weight_sum
        self.active_cir_sum = snapshot.active_cir_sum
        self.projected_work_gb = [
            count * representative_block_gb for count in snapshot.counts
        ]

    def estimate(
        self,
        block_size_gb: float,
        representative_block_gb: float,
        allowed_path_ids: Sequence[int],
        qos: QoSHardwareView,
        disk_bw_gbps: float,
        start_offset: int,
    ) -> PathEstimate:
        choices = _projection_choices(
            block_size_gb=block_size_gb,
            representative_block_gb=representative_block_gb,
            allowed_path_ids=allowed_path_ids,
            qos=qos,
            disk_bw_gbps=disk_bw_gbps,
            counts=self.counts,
            group_io_counts=self.group_io_counts,
            active_paths_per_group=self.active_paths_per_group,
            active_path_weights=self.active_path_weights,
            active_group_weight_sum=self.active_group_weight_sum,
            active_cir_sum=self.active_cir_sum,
            projected_work_gb=self.projected_work_gb,
        )
        return _select_projection_choice(
            choices,
            allowed_count=len(tuple(allowed_path_ids)),
            start_offset=start_offset,
            block_size_gb=block_size_gb,
            disk_bw_gbps=disk_bw_gbps,
        )

    def add(self, path_id: int, block_size_gb: float, qos: QoSHardwareView):
        group_id = path_id // qos.paths_per_group
        if self.counts[path_id] == 0:
            if self.active_paths_per_group[group_id] == 0:
                self.active_group_weight_sum += qos.group_weights[group_id]
            self.active_paths_per_group[group_id] += 1
            self.active_path_weights[group_id] += qos.path_weights[path_id]
            self.active_cir_sum += min(
                qos.path_cirs[path_id], qos.path_pirs[path_id]
            )
        self.counts[path_id] += 1
        self.group_io_counts[group_id] += 1
        self.projected_work_gb[path_id] += block_size_gb


def pressure_aware_path_ids(
    block_sizes_gb: Sequence[float],
    snapshot: PathPressureSnapshot,
    allowed_path_ids: Sequence[int],
    qos: QoSHardwareView,
    *,
    disk_bw_gbps: float = SSD_CAP_GBPS,
    start_offset: int = 0,
) -> tuple[int, ...]:
    """Route one pressure-refresh window using only its immutable snapshot."""
    sizes = tuple(map(float, block_sizes_gb))
    if not sizes:
        return ()
    allowed = tuple(allowed_path_ids)
    representative_gb = sorted(sizes)[len(sizes) // 2]
    if len(sizes) == 1:
        choices = _projection_choices(
            block_size_gb=sizes[0],
            representative_block_gb=representative_gb,
            allowed_path_ids=allowed,
            qos=qos,
            disk_bw_gbps=disk_bw_gbps,
            counts=snapshot.counts,
            group_io_counts=snapshot.group_io_counts,
            active_paths_per_group=snapshot.active_paths_per_group,
            active_path_weights=snapshot.active_path_weights,
            active_group_weight_sum=snapshot.active_group_weight_sum,
            active_cir_sum=snapshot.active_cir_sum,
        )
        return (
            _select_projection_choice(
                choices,
                allowed_count=len(allowed),
                start_offset=start_offset,
                block_size_gb=sizes[0],
                disk_bw_gbps=disk_bw_gbps,
            ).path_id,
        )

    shadow = _PlanningShadow(snapshot, representative_gb)
    selected = [0] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda item: (-sizes[item], item)):
        estimate = shadow.estimate(
            sizes[index],
            representative_gb,
            allowed,
            qos,
            disk_bw_gbps,
            start_offset,
        )
        selected[index] = estimate.path_id
        shadow.add(estimate.path_id, sizes[index], qos)
    return tuple(selected)


def layer_once_path_ids(
    block_sizes_gb: Sequence[float],
    snapshot: PathPressureSnapshot,
    allowed_path_ids: Sequence[int],
    qos: QoSHardwareView,
    *,
    disk_bw_gbps: float = SSD_CAP_GBPS,
    start_offset: int = 0,
) -> tuple[int, ...]:
    """Plan all I/Os of one request-layer-SSU from one pressure snapshot."""
    return pressure_aware_path_ids(
        block_sizes_gb,
        snapshot,
        allowed_path_ids,
        qos,
        disk_bw_gbps=disk_bw_gbps,
        start_offset=start_offset,
    )


def refresh8_path_ids(
    block_sizes_gb: Sequence[float],
    snapshot: PathPressureSnapshot,
    allowed_path_ids: Sequence[int],
    qos: QoSHardwareView,
    *,
    disk_bw_gbps: float = SSD_CAP_GBPS,
    start_offset: int = 0,
) -> tuple[int, ...]:
    """Plan one refresh8 window; the caller must pass at most eight I/Os.

    Hardware integration reads the SSU pressure table, passes that snapshot
    here with the next <=8 blocks, submits the returned Path IDs, then repeats.
    """
    sizes = tuple(block_sizes_gb)
    if len(sizes) > REFRESH8_WINDOW_IO:
        raise ValueError("refresh8_path_ids accepts at most eight I/Os per snapshot")
    return pressure_aware_path_ids(
        sizes,
        snapshot,
        allowed_path_ids,
        qos,
        disk_bw_gbps=disk_bw_gbps,
        start_offset=start_offset,
    )


def dedicated_path_id(
    npu_id: int,
    *,
    path_count: int = PATH_COUNT,
    group_count: int = GROUP_COUNT,
) -> int:
    """Spread up to 128 NPU-dedicated Paths evenly across eight groups."""
    paths_per_group = path_count // group_count
    return (int(npu_id) % group_count) * paths_per_group + int(npu_id) // group_count


def cold_start_hybrid_path_id(
    npu_id: int,
    *,
    path_count: int = PATH_COUNT,
    group_count: int = GROUP_COUNT,
    max_npu: int = MAX_NPU,
) -> int:
    """Use the unused half of each group while reserving Path 0 for cold I/O."""
    return dedicated_path_id(
        npu_id, path_count=path_count, group_count=group_count
    ) + max_npu // group_count


def _target_hash(active_ids, demands, grants) -> str:
    encoded = json.dumps(
        {
            "active_request_ids": active_ids,
            "demands_gbps": demands,
            "grants_gbps": grants,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_cirs_from_grants(
    grants: GrantMatrix,
    path_by_npu: Sequence[int],
    *,
    num_ssu: int,
    path_count: int,
    cold_path_id: int | None = None,
    cold_path_cir_gbps: float = 0.0,
) -> tuple[tuple[float, ...], ...]:
    tables = []
    for ssu_id in range(num_ssu):
        cirs = [0.0] * path_count
        for row_id, path_id in enumerate(path_by_npu):
            cirs[path_id] = grants[row_id][ssu_id]
        if cold_path_id is not None:
            cirs[cold_path_id] = cold_path_cir_gbps
        tables.append(tuple(cirs))
    return tuple(tables)


def plan_scheme_b(
    manifests: Iterable[ManifestDemand],
    *,
    num_npu: int,
    num_ssu: int,
    ssd_cap_gbps: float = SSD_CAP_GBPS,
    npu_cap_gbps: float = NPU_CAP_GBPS,
    path_count: int = PATH_COUNT,
    group_count: int = GROUP_COUNT,
    path_by_npu: Sequence[int] | None = None,
) -> SchemeBPlan:
    """Allocate demand-capped max-min grants from the visible active manifest."""
    rows = tuple(sorted(manifests, key=lambda row: row.request_id))
    work = [[0.0] * num_ssu for _ in range(num_npu)]
    compute_s = [0.0] * num_npu
    for row in rows:
        if len(row.work_by_ssu_gb) != num_ssu:
            raise ValueError("manifest work vector has the wrong SSU count")
        compute_s[row.npu_id] += row.compute_budget_s
        for ssu_id, amount in enumerate(row.work_by_ssu_gb):
            work[row.npu_id][ssu_id] += amount
    demands = tuple(
        tuple(
            amount / compute_s[npu_id] if compute_s[npu_id] > 0.0 else 0.0
            for amount in work[npu_id]
        )
        for npu_id in range(num_npu)
    )
    grants = allocate_grants(
        demands,
        ssd_caps=ssd_cap_gbps,
        npu_caps=npu_cap_gbps,
    )
    path_by_npu = (
        tuple(map(int, path_by_npu))
        if path_by_npu is not None
        else tuple(
            dedicated_path_id(
                npu_id,
                path_count=path_count,
                group_count=group_count,
            )
            for npu_id in range(num_npu)
        )
    )
    if len(path_by_npu) != num_npu or len(set(path_by_npu)) != num_npu:
        raise ValueError("Scheme B requires one unique Path per NPU")
    return SchemeBPlan(
        active_request_ids=tuple(row.request_id for row in rows),
        demands_gbps=demands,
        grants_gbps=grants,
        path_by_npu=path_by_npu,
        path_cirs_by_ssu=_path_cirs_from_grants(
            grants,
            path_by_npu,
            num_ssu=num_ssu,
            path_count=path_count,
        ),
        target_hash=_target_hash(
            tuple(row.request_id for row in rows), demands, grants
        ),
    )


def plan_causal_scheme_b(
    observations: Iterable[CausalLayerObservation],
    *,
    num_npu: int,
    num_ssu: int,
    cold_path_id: int = 0,
    cold_path_cir_gbps: float = 0.0,
    path_count: int = PATH_COUNT,
    group_count: int = GROUP_COUNT,
    ssd_cap_gbps: float = SSD_CAP_GBPS,
    npu_cap_gbps: float = NPU_CAP_GBPS,
    path_by_npu: Sequence[int] | None = None,
) -> SchemeBPlan:
    """Plan from completed previous layers, reserving one shared cold Path.

    ``observed_layer < 0`` means that the NPU has no prior-layer information.
    Those cold requests are excluded from max-min demand and activate the fixed
    reserve.  No timestamp or future completion estimate is accepted.
    """
    rows = tuple(sorted(observations, key=lambda row: row.request_id))
    warm = tuple(row for row in rows if row.observed_layer >= 0)
    cold = tuple(row for row in rows if row.observed_layer < 0)
    reserve = float(cold_path_cir_gbps) if cold else 0.0
    manifests = tuple(
        ManifestDemand(
            request_id=row.request_id,
            npu_id=row.npu_id,
            compute_budget_s=row.compute_budget_ms / 1000.0,
            work_by_ssu_gb=row.observed_work_gb_by_ssu,
        )
        for row in warm
    )
    selected_paths = (
        tuple(map(int, path_by_npu))
        if path_by_npu is not None
        else tuple(
            cold_start_hybrid_path_id(
                npu_id,
                path_count=path_count,
                group_count=group_count,
            )
            for npu_id in range(num_npu)
        )
    )
    if cold and cold_path_id in selected_paths:
        raise ValueError("the shared cold Path must not overlap a dedicated Path")
    compact = plan_scheme_b(
        manifests,
        num_npu=num_npu,
        num_ssu=num_ssu,
        ssd_cap_gbps=ssd_cap_gbps - reserve,
        npu_cap_gbps=npu_cap_gbps,
        path_count=path_count,
        group_count=group_count,
        path_by_npu=selected_paths,
    )
    return SchemeBPlan(
        active_request_ids=tuple(row.request_id for row in rows),
        demands_gbps=compact.demands_gbps,
        grants_gbps=compact.grants_gbps,
        path_by_npu=compact.path_by_npu,
        path_cirs_by_ssu=_path_cirs_from_grants(
            compact.grants_gbps,
            compact.path_by_npu,
            num_ssu=num_ssu,
            path_count=path_count,
            cold_path_id=(cold_path_id if cold else None),
            cold_path_cir_gbps=reserve,
        ),
        target_hash=_target_hash(
            tuple(row.request_id for row in rows),
            compact.demands_gbps,
            compact.grants_gbps,
        ),
    )


def oracle_priority_key(
    flow: OracleFlowView,
    *,
    demand_exponent: float = ORACLE_DEMAND_EXPONENT,
) -> tuple[float, ...]:
    """Priority of the capacity-preserving released-I/O oracle candidate."""
    demand = max(flow.demand_gbps, _EPS)
    weighted_work = flow.layer_work_gb / demand**demand_exponent
    return (
        weighted_work,
        flow.deadline_time,
        flow.layer_work_gb,
        flow.enqueue_order,
        flow.request_id,
        flow.layer,
        flow.block_idx,
        flow.ssu_id,
    )


def validate_scheme_b_plan(
    plan: SchemeBPlan,
    *,
    ssd_cap_gbps: float = SSD_CAP_GBPS,
    npu_cap_gbps: float = NPU_CAP_GBPS,
) -> bool:
    """Check the deployable max-min/CIR invariants without simulator state."""
    num_npu = len(plan.grants_gbps)
    num_ssu = len(plan.path_cirs_by_ssu)
    return (
        all(
            0.0 <= plan.grants_gbps[npu][ssu]
            <= plan.demands_gbps[npu][ssu] + 1e-9
            for npu in range(num_npu)
            for ssu in range(num_ssu)
        )
        and all(sum(row) <= npu_cap_gbps + 1e-9 for row in plan.grants_gbps)
        and all(
            sum(plan.grants_gbps[npu][ssu] for npu in range(num_npu))
            <= ssd_cap_gbps + 1e-9
            for ssu in range(num_ssu)
        )
        and all(sum(cirs) <= ssd_cap_gbps + 1e-9 for cirs in plan.path_cirs_by_ssu)
    )
