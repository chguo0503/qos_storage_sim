"""Full-prefill microbatches on the shared SSD/NPU data plane.

Each NPU freezes up to ``batch_size`` requests into one microbatch.  All
members wait at every layer's I/O barrier, execute one joint batch-layer
compute event, and finish the final layer together.  The default batch compute
model conserves the available singleton work by summing member layer times;
it does not assume an unmeasured batching speedup.  SSD commands remain
non-preemptive at 40 GB/s and each NPU owns one FCFS 50 GB/s receive link.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import heapq
import math
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

import sim
from policy_logic import (
    CausalLayerObservation as PolicyCausalLayerObservation,
    ManifestDemand,
    baseline_path_ids as policy_baseline_path_ids,
    plan_causal_scheme_b,
    plan_scheme_b,
    plan_slo_aware_scheme_b,
)


REQUEST_ARRIVAL = -1
BATCH_DISPATCH = -0.5
STEADY_MEASUREMENT_START = -2.0
STEADY_STATIONARITY_SAMPLE = -1.75
STEADY_MEASUREMENT_END = -1.5
CIR_CONTROL = 2.5
COMPUTE_SCHEDULE = 2.75
CAUSAL_LAYER_CONTROL = 2.875
_EPS = sim._EPS
STEADY_ACCOUNTING_TOLERANCE_MS = 1e-8

EXECUTION_MODEL = "full_prefill_layer_synchronous_microbatch_v1"
BATCH_COMPUTE_MODEL = "sum_member_singleton_layer_ms"
PARTIAL_BATCH_POLICY = "wait_for_full_then_drain_final_partial"
PREFETCH_POLICY = "next_layer_at_batch_compute_start"


def _steady_resource_overlap_within_bounds(busy_ms, duration_ms):
    """Accept only finite resource accounting within a tiny absolute tolerance."""
    busy_ms = float(busy_ms)
    duration_ms = float(duration_ms)
    return (
        math.isfinite(busy_ms)
        and math.isfinite(duration_ms)
        and duration_ms >= 0.0
        and -STEADY_ACCOUNTING_TOLERANCE_MS
        <= busy_ms
        <= duration_ms + STEADY_ACCOUNTING_TOLERANCE_MS
    )


def _steady_accounting_numeric_contract():
    """Self-test the tolerance used for cumulative steady-state accounting."""
    duration_ms = 8000.0
    two_ulp_above = math.nextafter(
        math.nextafter(duration_ms, math.inf), math.inf
    )
    checks = {
        "two_ulp_residual_accepted": _steady_resource_overlap_within_bounds(
            two_ulp_above, duration_ms
        ),
        "positive_boundary_accepted": _steady_resource_overlap_within_bounds(
            duration_ms + STEADY_ACCOUNTING_TOLERANCE_MS, duration_ms
        ),
        "negative_boundary_accepted": _steady_resource_overlap_within_bounds(
            -STEADY_ACCOUNTING_TOLERANCE_MS, duration_ms
        ),
        "positive_material_excess_rejected": not _steady_resource_overlap_within_bounds(
            duration_ms + 2.0 * STEADY_ACCOUNTING_TOLERANCE_MS, duration_ms
        ),
        "negative_material_excess_rejected": not _steady_resource_overlap_within_bounds(
            -2.0 * STEADY_ACCOUNTING_TOLERANCE_MS, duration_ms
        ),
        "nan_rejected": not _steady_resource_overlap_within_bounds(
            math.nan, duration_ms
        ),
        "infinity_rejected": not _steady_resource_overlap_within_bounds(
            math.inf, duration_ms
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(
            "steady accounting numeric contract failed: " + ", ".join(failed)
        )
    return {
        "tolerance_ms": STEADY_ACCOUNTING_TOLERANCE_MS,
        "tested_duration_ms": duration_ms,
        "two_ulp_residual_ms": two_ulp_above - duration_ms,
        "checks": checks,
        "passed": True,
    }


@dataclass(frozen=True)
class ContinuousBatchRequest:
    """One immutable prefill request assigned to one NPU.

    ``placement[layer][block]`` is ``(ssu_id, size_gb)``.  Request IDs must be
    globally unique because they are also the stable block-placement identity.
    """

    request_id: int
    npu_id: int
    arrival_time_ms: float
    load: Mapping[str, object]
    placement: tuple[tuple[tuple[int, float], ...], ...]

    def __post_init__(self):
        placement = tuple(
            tuple((int(ssu), float(size_gb)) for ssu, size_gb in layer)
            for layer in self.placement
        )
        object.__setattr__(self, "placement", placement)

    @classmethod
    def from_normalized(cls, request_id, npu_id, arrival_time_ms, load, placement):
        """Build from the workload generator's immutable normalized tuples."""
        request = object.__new__(cls)
        object.__setattr__(request, "request_id", request_id)
        object.__setattr__(request, "npu_id", npu_id)
        object.__setattr__(request, "arrival_time_ms", arrival_time_ms)
        object.__setattr__(request, "load", load)
        object.__setattr__(request, "placement", placement)
        return request


def request_from_prepared(
    load: Mapping[str, object],
    placement_by_layer: Mapping[int, Sequence[tuple[int, float]]],
    *,
    npu_id: Optional[int] = None,
    arrival_time_ms: Optional[float] = None,
) -> ContinuousBatchRequest:
    """Convert the repository's prepared request representation."""
    request_id = int(load["request_id"])
    layers = tuple(
        tuple(placement_by_layer[layer]) for layer in sorted(placement_by_layer)
    )
    return ContinuousBatchRequest(
        request_id=request_id,
        npu_id=int(load["npu_id"] if npu_id is None else npu_id),
        arrival_time_ms=float(
            load["arrival_time"] if arrival_time_ms is None else arrival_time_ms
        ),
        load=load,
        placement=layers,
    )


def requests_from_continuous_prefill_workload(workload):
    """Adapt ``continuous_prefill_workload.ContinuousPrefillWorkload``."""
    return tuple(
        ContinuousBatchRequest.from_normalized(
            request.request_id,
            request.npu_id,
            request.arrival_ms,
            request.as_dict(),
            (workload.placement_by_request[request.request_id],),
        )
        for request in workload.requests
    )


@dataclass(frozen=True)
class ControlRequestView:
    request_id: int
    npu_id: int
    category: str
    per_layer_compute_ms: float
    compute_done_up_to: int
    remaining_layers: int
    next_layer_work_gb_by_ssu: tuple[float, ...]
    waiting_for_io: bool
    remaining_work_gb_by_ssu: tuple[float, ...] = ()
    remaining_compute_budget_ms: float = 0.0
    prefetch_only: bool = False


@dataclass(frozen=True)
class CIRControlSnapshot:
    time_ms: float
    evaluation: int
    layer_jobs_since_previous: int
    num_npu: int
    num_ssu: int
    active_requests: tuple[ControlRequestView, ...]
    current_path_cirs_by_ssu: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class CIRControlDecision:
    path_cirs_by_ssu: tuple[tuple[float, ...], ...]


CIRControlCallback = Callable[[CIRControlSnapshot], Optional[CIRControlDecision]]


@dataclass(frozen=True, init=False)
class CIRControlConfig:
    """Evaluate at batch membership changes and optionally on a period.

    One layer-equivalent is one completed batch-layer weighted by its member
    count.  Wall-clock ticks are anchored at the first evaluation.  A causal
    evaluation also runs whenever an active microbatch is admitted or released
    unless ``on_batch_boundary`` is disabled.

    The custom initializer keeps the original positional form
    ``CIRControlConfig(every_layers, callback)`` compatible while also allowing
    ``CIRControlConfig(callback=callback, interval_ms=10.0)``.
    """

    every_layers: Optional[int]
    callback: CIRControlCallback
    interval_ms: Optional[float]
    on_batch_boundary: bool
    min_interval_ms: float

    def __init__(
        self,
        every_layers=None,
        callback=None,
        *,
        interval_ms=None,
        on_batch_boundary=True,
        min_interval_ms=0.0,
    ):
        if callback is None:
            raise ValueError("a CIR control callback is required")
        layer_mode = every_layers is not None
        interval_mode = interval_ms is not None
        if layer_mode and interval_mode:
            raise ValueError("set at most one of every_layers and interval_ms")
        if not layer_mode and not interval_mode and not on_batch_boundary:
            raise ValueError("the CIR controller has no evaluation trigger")
        if layer_mode and int(every_layers) <= 0:
            raise ValueError("every_layers must be positive")
        if interval_mode and float(interval_ms) <= 0.0:
            raise ValueError("interval_ms must be positive")
        if float(min_interval_ms) < 0.0:
            raise ValueError("min_interval_ms cannot be negative")
        object.__setattr__(
            self, "every_layers", int(every_layers) if layer_mode else None
        )
        object.__setattr__(self, "callback", callback)
        object.__setattr__(
            self, "interval_ms", float(interval_ms) if interval_mode else None
        )
        object.__setattr__(self, "on_batch_boundary", bool(on_batch_boundary))
        object.__setattr__(self, "min_interval_ms", float(min_interval_ms))


@dataclass(frozen=True)
class CausalLayerObservation:
    """Only facts available after one microbatch layer has completed I/O."""

    batch_id: int
    npu_id: int
    observed_layer: int
    observed_work_gb_by_ssu: tuple[float, ...]
    compute_budget_ms: float
    manifest_layer0: bool = False


@dataclass(frozen=True)
class CausalLayerSnapshot:
    """Deployment-visible input; deliberately contains no simulator clock."""

    num_npu: int
    num_ssu: int
    active_batches: tuple[CausalLayerObservation, ...]


CausalLayerControlCallback = Callable[
    [CausalLayerSnapshot], Optional[CIRControlDecision]
]


@dataclass(frozen=True)
class CausalLayerControlConfig:
    callback: CausalLayerControlCallback


@dataclass(frozen=True)
class SteadyStateConfig:
    """Measure one common full-load window after a finite warm-up."""

    warmup_requests_per_npu: int = 4
    settle_ms: float = 500.0
    measurement_ms: float = 2_000.0
    slo_alpha: float = 2.0
    block_ms: float = 500.0


class SteadyStateInvariantError(AssertionError):
    """A scientifically invalid steady window, distinct from a program bug."""

    def __init__(self, invariants, diagnostics):
        self.invariants = dict(invariants)
        self.diagnostics = dict(diagnostics)
        super().__init__(
            f"steady-state invariant failed: {self.invariants}; "
            f"diagnostics: {self.diagnostics}"
        )

    def __reduce__(self):
        return type(self), (self.invariants, self.diagnostics)


class MaxMinSchemeBController:
    """Manifest-based dynamic Scheme B for NPU-dedicated Paths."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        horizon_layers: int = 1,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.horizon_layers = int(horizon_layers)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)

    def __call__(self, snapshot: CIRControlSnapshot):
        manifests = []
        for request in snapshot.active_requests:
            horizon = min(request.remaining_layers, self.horizon_layers)
            if horizon <= 0:
                continue
            manifests.append(
                ManifestDemand(
                    request_id=request.request_id,
                    npu_id=request.npu_id,
                    compute_budget_s=(horizon * request.per_layer_compute_ms / 1000.0),
                    work_by_ssu_gb=tuple(
                        horizon * work_gb
                        for work_gb in request.next_layer_work_gb_by_ssu
                    ),
                )
            )
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        plan = plan_scheme_b(
            manifests,
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_count=path_count,
            path_by_npu=self.path_by_npu,
        )
        return CIRControlDecision(plan.path_cirs_by_ssu)


class SLOAwareSchemeBController:
    """Full-manifest Scheme B with request/coflow-synchronized progress.

    The client-visible snapshot contains all not-yet-ready layers of every
    active request.  A single scalar progress ratio is allocated to the whole
    NPU x SSU demand vector, first toward the warm TTFT target and then toward
    full I/O hiding.  This prevents bandwidth on a request's small SSU flow
    from masking starvation of its barrier-critical flow.
    """

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        slo_alpha: float = 2.0,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        if slo_alpha <= 0.0:
            raise ValueError("slo_alpha must be positive")
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.slo_alpha = float(slo_alpha)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self.last_plan = None

    def __call__(self, snapshot: CIRControlSnapshot):
        manifests = []
        for request in snapshot.active_requests:
            work = request.remaining_work_gb_by_ssu
            compute_ms = request.remaining_compute_budget_ms
            # Compatibility for callers that construct the older, next-layer
            # only ControlRequestView directly.
            if not work and request.remaining_layers > 0:
                work = tuple(
                    request.remaining_layers * amount
                    for amount in request.next_layer_work_gb_by_ssu
                )
                compute_ms = request.remaining_layers * request.per_layer_compute_ms
            if compute_ms <= 0.0 or not any(amount > 0.0 for amount in work):
                continue
            manifests.append(
                ManifestDemand(
                    request_id=request.request_id,
                    npu_id=request.npu_id,
                    compute_budget_s=compute_ms / 1000.0,
                    work_by_ssu_gb=tuple(work),
                )
            )
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        plan = plan_slo_aware_scheme_b(
            manifests,
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            slo_alpha=self.slo_alpha,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_count=path_count,
            path_by_npu=self.path_by_npu,
        )
        self.last_plan = plan
        return CIRControlDecision(plan.path_cirs_by_ssu)


class CausalMaxMinSchemeBController:
    """Max-min Scheme B driven only by completed previous-layer bytes."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        cold_path_id: int,
        cold_path_cir_gbps: float,
        path_count: int,
        ssd_cap_gbps: float = sim.DISK_BW,
        npu_cap_gbps: float = sim.NPU_BW_LIMIT,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.cold_path_id = int(cold_path_id)
        self.cold_path_cir_gbps = float(cold_path_cir_gbps)
        self.path_count = int(path_count)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self._previous_signature = None

    def __call__(self, snapshot: CausalLayerSnapshot):
        warm = tuple(
            observation
            for observation in snapshot.active_batches
            if observation.observed_layer >= 0
        )
        cold_count = len(snapshot.active_batches) - len(warm)
        signature = (
            bool(cold_count),
            tuple(
                (
                    observation.npu_id,
                    observation.observed_work_gb_by_ssu,
                    observation.compute_budget_ms,
                )
                for observation in warm
            ),
        )
        if signature == self._previous_signature:
            return None
        self._previous_signature = signature

        plan = plan_causal_scheme_b(
            tuple(
                PolicyCausalLayerObservation(
                    request_id=observation.batch_id,
                    npu_id=observation.npu_id,
                    observed_layer=observation.observed_layer,
                    compute_budget_ms=observation.compute_budget_ms,
                    observed_work_gb_by_ssu=observation.observed_work_gb_by_ssu,
                )
                for observation in snapshot.active_batches
            ),
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            cold_path_id=self.cold_path_id,
            cold_path_cir_gbps=self.cold_path_cir_gbps,
            path_count=self.path_count,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_by_npu=self.path_by_npu,
        )
        return CIRControlDecision(plan.path_cirs_by_ssu)


@dataclass
class _RequestState:
    manifest: ContinuousBatchRequest
    category: str
    per_layer_compute_ms: float
    placement_groups: Optional[
        tuple[tuple[tuple[int, tuple[tuple[int, float], ...], float], ...], ...]
    ] = None
    arrived: bool = False
    admitted: bool = False
    completed: bool = False
    admission_time_ms: float = math.nan
    completion_time_ms: float = math.nan
    batch_id: int = -1
    batch_size: int = 0
    io_started: bytearray = field(default_factory=bytearray)
    io_ready: bytearray = field(default_factory=bytearray)
    io_start_time_ms: list[float] = field(default_factory=list)
    pending_blocks: list[int] = field(default_factory=list)
    io_ready_time_ms: list[float] = field(default_factory=list)
    completed_gb_by_layer_ssu: list[list[float]] = field(default_factory=list)
    layer0_cross_request_prefetched: bool = False
    layer0_manifest_controlled: bool = False
    layer0_path_cirs_at_submit: dict[tuple[int, int], float] = field(
        default_factory=dict
    )
    ssd_queue_wait_ms: float = 0.0
    link_queue_wait_ms: float = 0.0
    io_latency_ms: float = 0.0
    io_count: int = 0


@dataclass
class _MicrobatchLayerMetric:
    layer: int
    io_start_time_ms: float = math.nan
    io_ready_time_ms: float = math.nan
    compute_start_ms: float = math.nan
    compute_end_ms: float = math.nan
    compute_duration_ms: float = 0.0
    io_barrier_wait_ms: float = 0.0


@dataclass
class _MicrobatchState:
    batch_id: int
    npu_id: int
    member_request_ids: tuple[int, ...]
    admission_time_ms: float
    previous_compute_end_ms: float
    compute_done_up_to: int = -1
    compute_active_layer: int = -1
    completion_time_ms: float = math.nan
    compute_busy_ms: float = 0.0
    io_barrier_wait_ms: float = 0.0
    observed_layer: int = -1
    observed_work_gb_by_ssu: tuple[float, ...] = ()
    layer_metrics: list[_MicrobatchLayerMetric] = field(default_factory=list)


@dataclass
class _NPUState:
    npu_id: int
    admission_queue: deque[int] = field(default_factory=deque)
    future_arrivals: int = 0
    active_batch: Optional[_MicrobatchState] = None
    layer0_prefetch_request_id: Optional[int] = None
    batch_dispatch_pending: bool = False
    compute_active: Optional[tuple[int, int]] = None
    compute_generation: int = 0
    compute_busy_ms: float = 0.0
    compute_dispatch_pending: bool = False
    max_batch_size: int = 0
    first_admission_ms: float = math.inf
    last_completion_ms: float = 0.0
    link_pending: deque[sim.BlockIOFlow] = field(default_factory=deque)
    link_pending_gb: float = 0.0
    link_active_flow: Optional[sim.BlockIOFlow] = None
    link_generation: int = 0
    link_completed_gb: float = 0.0
    link_busy_ms: float = 0.0
    link_accounted_busy_ms: float = 0.0
    link_last_account_ms: float = 0.0
    link_served_gb_by_ssu: dict[int, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    max_link_outstanding: int = 0


@dataclass
class _SubmissionState:
    state_id: int
    npu_id: int
    request_id: int
    category: str
    layer: int
    disk_id: int
    blocks: tuple[tuple[int, float], ...]
    ready_time_ms: float
    start_offset: int
    allowed_path_ids: tuple[int, ...]
    demand_gbps: float
    deadline_time_ms: float
    layer_work_gb: float
    manifest_dedicated_path: bool = False
    cursor: int = 0
    planned_path_ids: list[int] = field(default_factory=list)


def _group_placement_layer(layer):
    blocks_by_ssu = defaultdict(list)
    for block_index, (ssu_id, size_gb) in enumerate(layer):
        blocks_by_ssu[ssu_id].append((block_index, size_gb))
    return tuple(
        (
            ssu_id,
            tuple(blocks_by_ssu[ssu_id]),
            sum(size_gb for _, size_gb in blocks_by_ssu[ssu_id]),
        )
        for ssu_id in sorted(blocks_by_ssu)
    )


def _manifest_layer(request: ContinuousBatchRequest, layer: int):
    return request.placement[0 if len(request.placement) == 1 else layer]


def _state_placement_groups(request: _RequestState, layer: int):
    if request.placement_groups is None:
        raise AssertionError("active request placement was not materialized")
    return request.placement_groups[0 if len(request.placement_groups) == 1 else layer]


def _materialize_request(context: _Context, request: _RequestState):
    if request.placement_groups is None:
        request.placement_groups = tuple(
            _group_placement_layer(layer) for layer in request.manifest.placement
        )
    if not request.completed_gb_by_layer_ssu:
        request.completed_gb_by_layer_ssu = [
            [0.0] * context.num_ssu for _ in range(context.n_layers)
        ]


def _layer_work_by_ssu(
    context: _Context,
    request: _RequestState,
    layer: int,
):
    _materialize_request(context, request)
    work = [0.0] * context.num_ssu
    for ssu_id, _, work_gb in _state_placement_groups(request, layer):
        work[ssu_id] = work_gb
    return tuple(work)


class _Context:
    def __init__(
        self,
        *,
        requests: Sequence[ContinuousBatchRequest],
        num_npu: int,
        num_ssu: int,
        n_layers: int,
        batch_size: int,
        policy: str,
        qos_configs_by_ssu: tuple[sim.StaticQoSConfig, ...],
        npu_dedicated_paths: Optional[tuple[int, ...]],
        layer0_path_id: Optional[int],
        cross_request_layer0_prefetch: bool,
        client_io_config: sim.ClientIOConfig,
        control: Optional[CIRControlConfig],
        causal_control: Optional[CausalLayerControlConfig],
        steady_state: Optional[SteadyStateConfig],
        disk_bw_gbps: float,
        npu_bw_gbps: float,
        pressure_ttl_ms: float,
        cir_write_threshold_gbps: float,
        submit_order_seed: int,
        oracle_priority_key,
    ):
        self.num_npu = num_npu
        self.num_ssu = num_ssu
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.policy = policy
        self.qos_configs_by_ssu = qos_configs_by_ssu
        self.npu_dedicated_paths = npu_dedicated_paths
        self.layer0_path_id = layer0_path_id
        self.cross_request_layer0_prefetch = cross_request_layer0_prefetch
        self.client_io_config = client_io_config
        self.control = control
        self.causal_control = causal_control
        self.steady_state = steady_state
        self.disk_bw_gbps = disk_bw_gbps
        self.npu_bw_gbps = npu_bw_gbps
        self.pressure_ttl_ms = pressure_ttl_ms
        self.cir_write_threshold_gbps = cir_write_threshold_gbps
        self.submit_rng = np.random.RandomState(int(submit_order_seed))

        self.event_heap: list[tuple[float, float, int, int, int]] = []
        self.event_payloads: dict[int, object] = {}
        self.next_event_sequence = 0
        self.event_counts = defaultdict(int)
        self.stale_events = 0
        self.current_time_ms = 0.0

        self.requests = {}
        for request in requests:
            self.requests[request.request_id] = _RequestState(
                manifest=request,
                category=str(request.load["category"]),
                per_layer_compute_ms=float(request.load["per_layer_us"]) / 1000.0,
                io_started=bytearray(n_layers),
                io_ready=bytearray(n_layers),
                io_start_time_ms=[math.nan] * n_layers,
                pending_blocks=[0] * n_layers,
                io_ready_time_ms=[math.nan] * n_layers,
            )
        self.npus = [_NPUState(npu_id) for npu_id in range(num_npu)]
        self.disks = [sim.DiskState(ssu_id) for ssu_id in range(num_ssu)]
        for disk in self.disks:
            qos = (
                qos_configs_by_ssu[disk.disk_id]
                if policy == sim.POLICY_QOS_STATIC_CIR
                else None
            )
            sim.DiskIOScheduler(
                disk,
                policy,
                disk_bw_gbps,
                qos,
                oracle_priority_key,
                pressure_ttl_ms=pressure_ttl_ms,
                cir_write_threshold_gbps=cir_write_threshold_gbps,
            )

        self.client_path_pools_by_ssu = (
            tuple(
                {
                    category: sim.client_category_paths(category, qos)
                    for category in sim.QOS_ROUTING_CATEGORIES
                }
                for qos in qos_configs_by_ssu
            )
            if policy == sim.POLICY_QOS_STATIC_CIR and npu_dedicated_paths is None
            else ()
        )
        self.submission_states: dict[int, _SubmissionState] = {}
        self.submission_queues: dict[int, deque[int]] = defaultdict(deque)
        self.next_submission_id = 0
        self.client_next_issue_ms = [0.0] * num_npu
        self.client_event_generation = 0
        self.pending_client_event_ms: Optional[float] = None
        self.pending_link_start_ids: set[int] = set()
        self.completed_requests = 0
        self.completed_batch_layers = 0
        self.completed_request_layer_jobs = 0
        self.layer_jobs_since_control = 0
        self.control_threshold_jobs = 0
        self.control_reasons_by_time: dict[float, set[str]] = defaultdict(set)
        self.causal_control_pending = False
        self.pending_causal_prefetches: dict[int, tuple[int, int, float]] = {}
        self.pending_causal_request_prefetches: dict[int, tuple[int, float]] = {}
        self.causal_observations = 0
        self.cross_request_layer0_prefetches = 0
        self.manifest_layer0_prefetches = 0
        self.next_wall_control_ms: Optional[float] = None
        self.last_control_ms: Optional[float] = None
        self.control_evaluations = 0
        self.cir_commits = 0
        self.cir_path_writes = 0
        # ``cir_commits`` is the legacy fleet-wide epoch count.  The two
        # vectors below retain the hardware-facing cost: one transaction per
        # SSU whose table changed, and the number of register entries written.
        self.cir_write_transactions_by_ssu = [0] * num_ssu
        self.cir_path_writes_by_ssu = [0] * num_ssu
        self.max_cir_sum_gbps = 0.0
        initial_cir_sums = [
            sum(path.cir for path in disk.scheduler.paths.values())
            for disk in self.disks
        ]
        self.max_actual_cir_sum_gbps_by_ssu = list(initial_cir_sums)
        self.max_actual_npu_cir_sum_gbps_by_npu = (
            [
                sum(disk.scheduler.paths[path_id].cir for disk in self.disks)
                for path_id in npu_dedicated_paths
            ]
            if npu_dedicated_paths is not None
            else None
        )
        self.completed_by_npu = [0] * num_npu
        self.steady_warmup_reached_ms: Optional[float] = None
        self.measurement_start_ms: Optional[float] = None
        self.measurement_end_ms: Optional[float] = None
        self.measurement_open = False
        self.measurement_ended = False
        self.measurement_request_ids: set[int] = set()
        self.measurement_completed_ids: set[int] = set()
        self.measurement_disk_busy_start_ms: Optional[tuple[float, ...]] = None
        self.measurement_disk_busy_end_ms: Optional[tuple[float, ...]] = None
        self.measurement_npu_ssu_served_start_gb: Optional[
            tuple[tuple[float, ...], ...]
        ] = None
        self.measurement_npu_ssu_served_end_gb: Optional[
            tuple[tuple[float, ...], ...]
        ] = None
        self.measurement_link_busy_start_ms: Optional[tuple[float, ...]] = None
        self.measurement_link_busy_end_ms: Optional[tuple[float, ...]] = None
        self.measurement_npu_ssu_link_start_gb: Optional[
            tuple[tuple[float, ...], ...]
        ] = None
        self.measurement_npu_ssu_link_end_gb: Optional[
            tuple[tuple[float, ...], ...]
        ] = None
        self.measurement_disk_outstanding_start: Optional[tuple[int, ...]] = None
        self.measurement_disk_outstanding_end: Optional[tuple[int, ...]] = None
        self.measurement_control_counters_start: Optional[dict] = None
        self.measurement_control_counters_end: Optional[dict] = None
        self.measurement_actual_cir_start: Optional[dict] = None
        self.measurement_actual_cir_end: Optional[dict] = None
        self.measurement_stationarity_snapshots: list[dict] = []
        self.steady_backlog_exhausted = False

        self.microbatches: list[_MicrobatchState] = []
        self.next_batch_id = 0
        for request in requests:
            self.npus[request.npu_id].future_arrivals += 1

        self.block_offsets: dict[tuple[int, int], int] = {}
        self.expected_blocks = 0
        self.expected_read_gb = 0.0
        for request in requests:
            if len(request.placement) == 1:
                blocks = request.placement[0]
                block_count = len(blocks)
                per_layer_read_gb = sum(size_gb for _, size_gb in blocks)
                request_base = self.expected_blocks
                for layer in range(n_layers):
                    self.block_offsets[(request.request_id, layer)] = (
                        request_base + layer * block_count
                    )
                    self.expected_blocks += block_count
                    self.expected_read_gb += per_layer_read_gb
            else:
                for layer in range(n_layers):
                    blocks = request.placement[layer]
                    self.block_offsets[
                        (request.request_id, layer)
                    ] = self.expected_blocks
                    self.expected_blocks += len(blocks)
                    self.expected_read_gb += sum(size_gb for _, size_gb in blocks)
        self.block_states = bytearray(self.expected_blocks)
        self.submitted_blocks = 0
        self.completed_blocks = 0

    def push_event(self, time_ms, event_type, resource_id, payload=None, generation=0):
        sequence = self.next_event_sequence
        self.next_event_sequence += 1
        if payload is not None:
            self.event_payloads[sequence] = payload
        heapq.heappush(
            self.event_heap,
            (float(time_ms), event_type, int(resource_id), sequence, int(generation)),
        )
        return sequence

    def pop_payload(self, sequence):
        return self.event_payloads.pop(sequence, None)


def _actual_cir_state(context: _Context):
    """Return the CIRs that are installed in schedulers, not controller targets."""
    per_ssu = tuple(
        float(sum(path.cir for path in disk.scheduler.paths.values()))
        for disk in context.disks
    )
    per_npu = None
    if context.npu_dedicated_paths is not None:
        per_npu = tuple(
            float(sum(disk.scheduler.paths[path_id].cir for disk in context.disks))
            for path_id in context.npu_dedicated_paths
        )
    return {
        "per_ssu_gbps": per_ssu,
        "per_npu_gbps": per_npu,
    }


def _record_actual_cir_state(context: _Context):
    """Retain post-control-epoch high-water marks of installed CIR tables."""
    state = _actual_cir_state(context)
    for ssu_id, value in enumerate(state["per_ssu_gbps"]):
        context.max_actual_cir_sum_gbps_by_ssu[ssu_id] = max(
            context.max_actual_cir_sum_gbps_by_ssu[ssu_id], value
        )
    if state["per_npu_gbps"] is not None:
        for npu_id, value in enumerate(state["per_npu_gbps"]):
            context.max_actual_npu_cir_sum_gbps_by_npu[npu_id] = max(
                context.max_actual_npu_cir_sum_gbps_by_npu[npu_id], value
            )
    return state


def _control_counter_snapshot(context: _Context):
    """Capture cumulative control-plane counters at one exact wall-time edge."""
    pressure_reports_by_ssu = tuple(
        int(disk.scheduler.pressure_reports) for disk in context.disks
    )
    pressure_cache_hits_by_ssu = tuple(
        int(getattr(disk.scheduler, "pressure_cache_hits", 0)) for disk in context.disks
    )
    return {
        "pressure_reports_by_ssu": pressure_reports_by_ssu,
        "pressure_cache_hits_by_ssu": pressure_cache_hits_by_ssu,
        "control_evaluations": int(context.control_evaluations),
        "cir_commits": int(context.cir_commits),
        "cir_write_transactions_by_ssu": tuple(
            map(int, context.cir_write_transactions_by_ssu)
        ),
        "cir_path_writes_by_ssu": tuple(map(int, context.cir_path_writes_by_ssu)),
        "cir_path_writes": int(context.cir_path_writes),
    }


def _counter_delta(end, start, name):
    delta = int(end) - int(start)
    if delta < 0:
        raise AssertionError(f"control counter moved backwards: {name}")
    return delta


def _counter_vector_delta(end, start, name):
    if len(end) != len(start):
        raise AssertionError(f"control counter shape changed: {name}")
    return tuple(
        _counter_delta(end_value, start_value, f"{name}[{index}]")
        for index, (end_value, start_value) in enumerate(zip(end, start))
    )


def _measurement_control_metrics(context: _Context, duration_ms: float):
    """Build exact half-open measurement-window control costs and rates."""
    start = context.measurement_control_counters_start
    end = context.measurement_control_counters_end
    if start is None or end is None:
        raise AssertionError("steady-state control counter snapshots are missing")
    if duration_ms <= 0.0:
        raise AssertionError("measurement duration must be positive")

    pressure_by_ssu = _counter_vector_delta(
        end["pressure_reports_by_ssu"],
        start["pressure_reports_by_ssu"],
        "pressure_reports_by_ssu",
    )
    cache_hits_by_ssu = _counter_vector_delta(
        end["pressure_cache_hits_by_ssu"],
        start["pressure_cache_hits_by_ssu"],
        "pressure_cache_hits_by_ssu",
    )
    transactions_by_ssu = _counter_vector_delta(
        end["cir_write_transactions_by_ssu"],
        start["cir_write_transactions_by_ssu"],
        "cir_write_transactions_by_ssu",
    )
    path_writes_by_ssu = _counter_vector_delta(
        end["cir_path_writes_by_ssu"],
        start["cir_path_writes_by_ssu"],
        "cir_path_writes_by_ssu",
    )
    pressure_reports = sum(pressure_by_ssu)
    cache_hits = sum(cache_hits_by_ssu)
    transactions = sum(transactions_by_ssu)
    path_writes = _counter_delta(
        end["cir_path_writes"], start["cir_path_writes"], "cir_path_writes"
    )
    if path_writes != sum(path_writes_by_ssu):
        raise AssertionError("fleet and per-SSU CIR entry-write counters diverged")
    duration_s = duration_ms / 1000.0

    def rates(values):
        return [float(value) / duration_s for value in values]

    pressure_requests_by_ssu = tuple(
        reports + hits for reports, hits in zip(pressure_by_ssu, cache_hits_by_ssu)
    )
    pressure_requests = pressure_reports + cache_hits
    control_evaluations = _counter_delta(
        end["control_evaluations"],
        start["control_evaluations"],
        "control_evaluations",
    )
    cir_commits = _counter_delta(
        end["cir_commits"], start["cir_commits"], "cir_commits"
    )
    return {
        "measurement_pressure_reports": pressure_reports,
        "measurement_pressure_reports_by_ssu": list(pressure_by_ssu),
        "measurement_pressure_report_rate_hz": pressure_reports / duration_s,
        "measurement_pressure_report_rate_hz_by_ssu": rates(pressure_by_ssu),
        "measurement_pressure_cache_hits": cache_hits,
        "measurement_pressure_cache_hits_by_ssu": list(cache_hits_by_ssu),
        "measurement_pressure_requests": pressure_requests,
        "measurement_pressure_requests_by_ssu": list(pressure_requests_by_ssu),
        "measurement_pressure_cache_hit_fraction": (
            cache_hits / pressure_requests if pressure_requests else 0.0
        ),
        "measurement_pressure_cache_hit_fraction_by_ssu": [
            hits / requests if requests else 0.0
            for hits, requests in zip(cache_hits_by_ssu, pressure_requests_by_ssu)
        ],
        "measurement_control_evaluations": control_evaluations,
        "measurement_control_evaluation_rate_hz": control_evaluations / duration_s,
        "measurement_cir_commits": cir_commits,
        "measurement_cir_commit_rate_hz": cir_commits / duration_s,
        "measurement_cir_write_transactions": transactions,
        "measurement_cir_write_transactions_by_ssu": list(transactions_by_ssu),
        "measurement_cir_write_transaction_rate_hz": transactions / duration_s,
        "measurement_cir_write_transaction_rate_hz_by_ssu": rates(transactions_by_ssu),
        "measurement_cir_path_writes": path_writes,
        "measurement_cir_path_writes_by_ssu": list(path_writes_by_ssu),
        "measurement_cir_path_write_rate_hz": path_writes / duration_s,
        "measurement_cir_path_write_rate_hz_by_ssu": rates(path_writes_by_ssu),
        "measurement_cir_entries_per_transaction": (
            path_writes / transactions if transactions else 0.0
        ),
        "measurement_cir_entries_per_transaction_by_ssu": [
            entries / transaction_count if transaction_count else 0.0
            for entries, transaction_count in zip(
                path_writes_by_ssu, transactions_by_ssu
            )
        ],
    }


def _project_ssd_active_remaining_gb(flow, current_time_ms: float):
    """Project one active SSD command without mutating its progress fields."""
    if flow is None:
        return 0.0
    elapsed_ms = max(0.0, current_time_ms - flow.start_time)
    served_gb = max(0.0, flow.bw) * elapsed_ms / 1000.0
    return max(0.0, float(flow.remaining_gb) - served_gb)


def _project_ssd_snapshot(context: _Context, current_time_ms: float):
    """Return read-only cumulative service and instantaneous SSD queue state."""
    cumulative_busy_ms = []
    cumulative_served_gb = []
    outstanding_blocks = []
    outstanding_gb = []
    for disk in context.disks:
        scheduler = disk.scheduler
        active_flow = disk.active_flows[0] if disk.active_flows else None
        active_duration_ms = 0.0
        if active_flow is not None:
            service_end_ms = min(current_time_ms, active_flow.end_time)
            active_duration_ms = max(0.0, service_end_ms - disk.last_event_time)
        projected_busy_ms = float(disk.busy_time) + active_duration_ms
        cumulative_busy_ms.append(projected_busy_ms)
        cumulative_served_gb.append(projected_busy_ms * context.disk_bw_gbps / 1000.0)

        if scheduler.policy == sim.POLICY_QOS_STATIC_CIR:
            queued_gb = math.fsum(path.pending_gb for path in scheduler.paths.values())
        else:
            queued_gb = math.fsum(flow.total_gb for _, flow in scheduler.oracle_heap)
        active_remaining_gb = _project_ssd_active_remaining_gb(
            active_flow, current_time_ms
        )
        outstanding_blocks.append(int(scheduler.outstanding_blocks))
        outstanding_gb.append(max(0.0, queued_gb + active_remaining_gb))
    return {
        "cumulative_busy_ms_by_ssu": tuple(cumulative_busy_ms),
        "cumulative_served_gb_by_ssu": tuple(cumulative_served_gb),
        "outstanding_blocks_by_ssu": tuple(outstanding_blocks),
        "outstanding_gb_by_ssu": tuple(outstanding_gb),
    }


def _project_link_active_remaining_gb(
    context: _Context,
    flow,
    current_time_ms: float,
):
    """Project one active NPU-link transfer from its immutable start edge."""
    if flow is None:
        return 0.0
    service_end_ms = min(current_time_ms, flow.link_end_time)
    elapsed_ms = max(0.0, service_end_ms - flow.link_start_time)
    served_gb = context.npu_bw_gbps * elapsed_ms / 1000.0
    return max(0.0, float(flow.total_gb) - served_gb)


def _project_npu_snapshot(context: _Context, current_time_ms: float):
    """Return read-only compute/link projections for every NPU."""
    compute_busy_ms = []
    link_busy_ms = []
    link_served_gb = []
    link_outstanding_blocks = []
    link_outstanding_gb = []
    for npu in context.npus:
        projected_compute_ms = float(npu.compute_busy_ms)
        if npu.compute_active is not None:
            batch_id, layer = npu.compute_active
            batch = npu.active_batch
            if batch is None or batch.batch_id != batch_id:
                raise AssertionError("active NPU compute has no matching microbatch")
            metric = batch.layer_metrics[layer]
            compute_end_ms = min(current_time_ms, metric.compute_end_ms)
            projected_compute_ms += max(0.0, compute_end_ms - metric.compute_start_ms)
        compute_busy_ms.append(projected_compute_ms)

        active_flow = npu.link_active_flow
        active_duration_ms = 0.0
        if active_flow is not None:
            service_end_ms = min(current_time_ms, active_flow.link_end_time)
            active_duration_ms = max(0.0, service_end_ms - npu.link_last_account_ms)
        projected_link_busy_ms = float(npu.link_accounted_busy_ms) + active_duration_ms
        link_busy_ms.append(projected_link_busy_ms)
        link_served_gb.append(projected_link_busy_ms * context.npu_bw_gbps / 1000.0)
        link_outstanding_blocks.append(
            sum(flow.block_count for flow in npu.link_pending)
            + (active_flow.block_count if active_flow is not None else 0)
        )
        link_outstanding_gb.append(
            max(
                0.0,
                float(npu.link_pending_gb)
                + _project_link_active_remaining_gb(
                    context, active_flow, current_time_ms
                ),
            )
        )
    return {
        "compute_cumulative_busy_ms_by_npu": tuple(compute_busy_ms),
        "link_cumulative_busy_ms_by_npu": tuple(link_busy_ms),
        "link_cumulative_served_gb_by_npu": tuple(link_served_gb),
        "link_outstanding_blocks_by_npu": tuple(link_outstanding_blocks),
        "link_outstanding_gb_by_npu": tuple(link_outstanding_gb),
    }


def _capture_stationarity_snapshot(
    context: _Context,
    boundary: int,
    current_time_ms: float,
):
    """Capture a left-limit boundary without settling any simulated resource."""
    ssd = _project_ssd_snapshot(context, current_time_ms)
    npu = _project_npu_snapshot(context, current_time_ms)
    return {
        "boundary": int(boundary),
        "time_ms": float(current_time_ms),
        "ssd_cumulative_busy_ms_by_ssu": ssd["cumulative_busy_ms_by_ssu"],
        "ssd_cumulative_served_gb_by_ssu": ssd["cumulative_served_gb_by_ssu"],
        "ssd_outstanding_blocks_by_ssu": ssd["outstanding_blocks_by_ssu"],
        "ssd_outstanding_gb_by_ssu": ssd["outstanding_gb_by_ssu"],
        "npu_compute_cumulative_busy_ms_by_npu": npu[
            "compute_cumulative_busy_ms_by_npu"
        ],
        "npu_link_cumulative_busy_ms_by_npu": npu["link_cumulative_busy_ms_by_npu"],
        "npu_link_cumulative_served_gb_by_npu": npu["link_cumulative_served_gb_by_npu"],
        "npu_link_outstanding_blocks_by_npu": npu["link_outstanding_blocks_by_npu"],
        "npu_link_outstanding_gb_by_npu": npu["link_outstanding_gb_by_npu"],
    }


def _register_submit(context: _Context, flow: sim.BlockIOFlow):
    request_state = context.requests[flow.request_id]
    request = request_state.manifest
    expected_ssu, expected_gb = _manifest_layer(request, flow.layer)[flow.block_idx]
    if flow.disk_id != expected_ssu or not math.isclose(
        flow.total_gb, expected_gb, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("submitted block differs from prepared placement")
    block_id = context.block_offsets[(flow.request_id, flow.layer)] + flow.block_idx
    if context.block_states[block_id] != 0:
        raise AssertionError("block submitted more than once")
    context.block_states[block_id] = 1
    context.submitted_blocks += 1
    if flow.layer == 0 and flow.queue_id >= 0:
        request_state.layer0_path_cirs_at_submit[(flow.disk_id, flow.queue_id)] = (
            context.disks[flow.disk_id].scheduler.paths[flow.queue_id].cir
        )


def _register_complete(context: _Context, flow: sim.BlockIOFlow):
    block_id = context.block_offsets[(flow.request_id, flow.layer)] + flow.block_idx
    if context.block_states[block_id] != 1:
        raise AssertionError("block completed before submit or more than once")
    context.block_states[block_id] = 2
    context.completed_blocks += 1


def _batch_compute_duration_ms(context: _Context, batch: _MicrobatchState):
    return sum(
        context.requests[request_id].per_layer_compute_ms
        for request_id in batch.member_request_ids
    )


def _schedule_compute_dispatch(context: _Context, npu: _NPUState, current_time_ms):
    batch = npu.active_batch
    if batch is None or npu.compute_active is not None:
        return
    layer = batch.compute_done_up_to + 1
    if layer >= context.n_layers or batch.compute_active_layer >= 0:
        return
    if not all(
        context.requests[request_id].io_ready[layer]
        for request_id in batch.member_request_ids
    ):
        return
    metric = batch.layer_metrics[layer]
    metric.io_ready_time_ms = max(
        context.requests[request_id].io_ready_time_ms[layer]
        for request_id in batch.member_request_ids
    )
    if npu.compute_dispatch_pending:
        return
    npu.compute_dispatch_pending = True
    context.push_event(current_time_ms, COMPUTE_SCHEDULE, npu.npu_id)


def _mark_layer_io_ready(
    context: _Context,
    request: _RequestState,
    layer: int,
    current_time_ms: float,
):
    request.io_ready[layer] = 1
    request.io_ready_time_ms[layer] = current_time_ms
    if not request.admitted:
        return
    npu = context.npus[request.manifest.npu_id]
    batch = npu.active_batch
    if batch is None or request.batch_id != batch.batch_id:
        raise AssertionError("I/O completed outside the request's active microbatch")
    _schedule_compute_dispatch(context, npu, current_time_ms)


def _schedule_client_event(context: _Context):
    if not context.submission_states:
        return
    next_times = [
        max(
            context.submission_states[state_ids[0]].ready_time_ms,
            context.client_next_issue_ms[npu_id],
        )
        for npu_id, state_ids in context.submission_queues.items()
        if state_ids
    ]
    if not next_times:
        return
    next_time = min(next_times)
    pending = context.pending_client_event_ms
    if pending is not None and pending <= next_time + _EPS:
        return
    context.client_event_generation += 1
    context.pending_client_event_ms = next_time
    context.push_event(
        next_time,
        sim.CLIENT_SUBMISSION,
        0,
        generation=context.client_event_generation,
    )


def _start_layer_io(
    context: _Context,
    request: _RequestState,
    layer: int,
    current_time_ms: float,
    deadline_time_ms: float,
    demand_window_ms: float,
    *,
    manifest_dedicated_path: bool = False,
):
    if layer < 0 or layer >= context.n_layers or request.io_started[layer]:
        return
    request.io_started[layer] = 1
    request.io_start_time_ms[layer] = current_time_ms
    placement_groups = _state_placement_groups(request, layer)
    request.pending_blocks[layer] = sum(
        len(blocks) for _, blocks, _ in placement_groups
    )
    if not placement_groups:
        _mark_layer_io_ready(context, request, layer, current_time_ms)
        return
    work_by_ssu = {ssu_id: work_gb for ssu_id, _, work_gb in placement_groups}
    demand_by_ssu = (
        sim.capped_proportional_demands(
            context.npu_bw_gbps,
            demand_window_ms / 1000.0,
            work_by_ssu,
        )
        if context.policy == sim.POLICY_PER_SSD_FULL_VISIBLE_EDF
        else {}
    )
    npu_id = request.manifest.npu_id
    for ssu_id, blocks, work_gb in placement_groups:
        allowed_paths = (
            (context.npu_dedicated_paths[npu_id],)
            if manifest_dedicated_path
            else (context.layer0_path_id,)
            if layer == 0 and context.layer0_path_id is not None
            else (context.npu_dedicated_paths[npu_id],)
            if context.npu_dedicated_paths is not None
            else context.client_path_pools_by_ssu[ssu_id][request.category]
            if context.policy == sim.POLICY_QOS_STATIC_CIR
            else ()
        )
        state_id = context.next_submission_id
        context.next_submission_id += 1
        context.submission_states[state_id] = _SubmissionState(
            state_id=state_id,
            npu_id=npu_id,
            request_id=request.manifest.request_id,
            category=request.category,
            layer=layer,
            disk_id=ssu_id,
            blocks=blocks,
            ready_time_ms=current_time_ms,
            start_offset=(request.manifest.request_id + layer * 13 + ssu_id * 29)
            % len(allowed_paths)
            if allowed_paths
            else 0,
            allowed_path_ids=allowed_paths,
            demand_gbps=demand_by_ssu.get(ssu_id, 0.0),
            deadline_time_ms=deadline_time_ms,
            layer_work_gb=work_gb,
            manifest_dedicated_path=manifest_dedicated_path,
        )
        context.submission_queues[npu_id].append(state_id)
    _schedule_client_event(context)


def _start_batch_layer_io(
    context: _Context,
    batch: _MicrobatchState,
    layer: int,
    current_time_ms: float,
    deadline_time_ms: float,
):
    metric = batch.layer_metrics[layer]
    if np.isfinite(metric.io_start_time_ms):
        return
    metric.io_start_time_ms = current_time_ms
    demand_window_ms = _batch_compute_duration_ms(context, batch)
    for request_id in batch.member_request_ids:
        _start_layer_io(
            context,
            context.requests[request_id],
            layer,
            current_time_ms,
            deadline_time_ms,
            demand_window_ms,
        )


def _plan_paths(context: _Context, state: _SubmissionState, current_time_ms):
    window_start = len(state.planned_path_ids)
    if (
        state.layer == 0
        and context.layer0_path_id is not None
        and not state.manifest_dedicated_path
    ):
        state.planned_path_ids.extend(
            context.layer0_path_id for _ in range(window_start, len(state.blocks))
        )
        return
    if context.npu_dedicated_paths is not None:
        state.planned_path_ids.extend(
            state.allowed_path_ids[0] for _ in range(window_start, len(state.blocks))
        )
        return
    mode = context.client_io_config.path_selection_mode
    if mode == sim.PATH_SELECTION_FIXED_PATH_ZERO:
        state.planned_path_ids.extend(
            policy_baseline_path_ids(len(state.blocks) - window_start)
        )
        return

    pressure_window = context.client_io_config.pressure_window_io
    window_end = (
        len(state.blocks)
        if pressure_window is None
        else min(len(state.blocks), window_start + pressure_window)
    )
    scheduler = context.disks[state.disk_id].scheduler
    sizes = tuple(size_gb for _, size_gb in state.blocks[window_start:window_end])
    pressure = scheduler.report_path_pressure_analysis(current_time_ms)
    state.planned_path_ids.extend(
        sim._select_qos_paths_from_analysis(
            sizes,
            pressure,
            state.allowed_path_ids,
            sim.ClientRoutingConfig(
                qos_config=context.qos_configs_by_ssu[state.disk_id],
                disk_bw=scheduler.disk_bw,
                start_offset=state.start_offset,
            ),
        )
    )


def _submit_one_client_batch(
    context: _Context,
    state: _SubmissionState,
    current_time_ms: float,
):
    scheduler = context.disks[state.disk_id].scheduler
    submit_end = min(
        len(state.blocks),
        state.cursor + context.client_io_config.submit_batch_size,
    )
    while state.cursor < submit_end:
        if context.policy == sim.POLICY_QOS_STATIC_CIR:
            if state.cursor == len(state.planned_path_ids):
                _plan_paths(context, state, current_time_ms)
            chunk_end = min(submit_end, len(state.planned_path_ids))
            path_ids = state.planned_path_ids[state.cursor : chunk_end]
        else:
            chunk_end = submit_end
            path_ids = [-1] * (chunk_end - state.cursor)
        flows = []
        for (block_index, size_gb), path_id in zip(
            state.blocks[state.cursor : chunk_end], path_ids
        ):
            flow = sim.BlockIOFlow(
                npu_id=state.npu_id,
                layer=state.layer,
                block_idx=block_index,
                disk_id=state.disk_id,
                total_gb=size_gb,
                queue_id=path_id,
                block_count=1,
                enqueue_time=current_time_ms,
                request_id=state.request_id,
                demand_gbps=state.demand_gbps,
                deadline_time=state.deadline_time_ms,
                layer_work_gb=state.layer_work_gb,
            )
            _register_submit(context, flow)
            flows.append(flow)
        scheduler.enqueue_many(flows, current_time_ms)
        state.cursor = chunk_end
    scheduler.request_dispatch(current_time_ms, context.event_heap)
    return state.cursor == len(state.blocks)


def _handle_client_submission(context: _Context, generation, current_time_ms):
    if generation != context.client_event_generation:
        context.stale_events += 1
        return
    context.pending_client_event_ms = None
    ready_npus = []
    for npu_id, state_ids in context.submission_queues.items():
        if not state_ids:
            continue
        state = context.submission_states[state_ids[0]]
        if (
            state.ready_time_ms <= current_time_ms + _EPS
            and context.client_next_issue_ms[npu_id] <= current_time_ms + _EPS
        ):
            ready_npus.append(npu_id)
    ready_npus.sort()
    context.submit_rng.shuffle(ready_npus)
    issue_round = []
    for npu_id in ready_npus:
        state_ids = context.submission_queues[npu_id]
        state_id = state_ids.popleft()
        state = context.submission_states[state_id]
        issue_round.append((npu_id, state_ids, state_id, state))

    # All NPUs issuing at the same timestamp read the same pre-submit Path
    # state.  Commands are still linearized below, and the next 0.1-us issue
    # tick observes everything inserted by this round.
    if context.policy == sim.POLICY_QOS_STATIC_CIR:
        for _, _, _, state in issue_round:
            if state.cursor == len(state.planned_path_ids):
                _plan_paths(context, state, current_time_ms)

    for npu_id, state_ids, state_id, state in issue_round:
        issued = min(
            context.client_io_config.submit_batch_size,
            len(state.blocks) - state.cursor,
        )
        finished = _submit_one_client_batch(context, state, current_time_ms)
        context.client_next_issue_ms[npu_id] = current_time_ms + (
            issued * context.client_io_config.issue_interval_us / 1000.0
        )
        if finished:
            del context.submission_states[state_id]
        else:
            state_ids.append(state_id)
        if not state_ids:
            del context.submission_queues[npu_id]
    _schedule_client_event(context)


def _queue_control_event(
    context: _Context,
    current_time_ms: float,
    reason: str,
):
    if context.control is None:
        return
    if reason == "batch_boundary" and not context.control.on_batch_boundary:
        return
    scheduled_time_ms = float(current_time_ms)
    if (
        reason == "batch_boundary"
        and context.last_control_ms is not None
        and context.control.min_interval_ms > 0.0
    ):
        scheduled_time_ms = max(
            scheduled_time_ms,
            context.last_control_ms + context.control.min_interval_ms,
        )
    reasons = context.control_reasons_by_time[scheduled_time_ms]
    if not reasons:
        context.push_event(scheduled_time_ms, CIR_CONTROL, 0)
    reasons.add(reason)


def _queue_causal_control_event(context: _Context, current_time_ms: float):
    if context.causal_control is None or context.causal_control_pending:
        return
    context.causal_control_pending = True
    context.push_event(current_time_ms, CAUSAL_LAYER_CONTROL, 0)


def _schedule_batch_dispatch(
    context: _Context,
    npu: _NPUState,
    current_time_ms: float,
):
    if npu.active_batch is not None or npu.batch_dispatch_pending:
        return
    if not npu.admission_queue:
        return
    if len(npu.admission_queue) < context.batch_size and npu.future_arrivals > 0:
        return
    npu.batch_dispatch_pending = True
    context.push_event(current_time_ms, BATCH_DISPATCH, npu.npu_id)


def _handle_batch_dispatch(
    context: _Context,
    npu_id: int,
    current_time_ms: float,
):
    npu = context.npus[npu_id]
    npu.batch_dispatch_pending = False
    if npu.active_batch is not None or not npu.admission_queue:
        return
    if len(npu.admission_queue) < context.batch_size and npu.future_arrivals > 0:
        return
    member_ids = tuple(
        npu.admission_queue.popleft()
        for _ in range(min(context.batch_size, len(npu.admission_queue)))
    )
    batch = _MicrobatchState(
        batch_id=context.next_batch_id,
        npu_id=npu_id,
        member_request_ids=member_ids,
        admission_time_ms=current_time_ms,
        previous_compute_end_ms=current_time_ms,
        layer_metrics=[
            _MicrobatchLayerMetric(layer) for layer in range(context.n_layers)
        ],
    )
    context.next_batch_id += 1
    context.microbatches.append(batch)
    npu.active_batch = batch
    npu.max_batch_size = max(npu.max_batch_size, len(member_ids))
    npu.first_admission_ms = min(npu.first_admission_ms, current_time_ms)
    for request_id in member_ids:
        request = context.requests[request_id]
        _materialize_request(context, request)
        request.admitted = True
        request.admission_time_ms = current_time_ms
        request.batch_id = batch.batch_id
        request.batch_size = len(member_ids)
        if context.measurement_open and current_time_ms < context.measurement_end_ms:
            context.measurement_request_ids.add(request_id)
    if len(member_ids) == 1:
        request = context.requests[member_ids[0]]
        if request.io_started[0]:
            batch.layer_metrics[0].io_start_time_ms = request.io_start_time_ms[0]
    if context.control is not None:
        request.layer0_manifest_controlled = True
        context.manifest_layer0_prefetches += 1
        _queue_control_event(context, current_time_ms, "batch_boundary")
    _queue_causal_control_event(context, current_time_ms)
    _start_batch_layer_io(
        context,
        batch,
        0,
        current_time_ms,
        current_time_ms,
    )
    _schedule_compute_dispatch(context, npu, current_time_ms)


def _handle_arrival(context: _Context, request_id: int, current_time_ms: float):
    request = context.requests[request_id]
    request.arrived = True
    npu = context.npus[request.manifest.npu_id]
    npu.future_arrivals -= 1
    npu.admission_queue.append(request_id)
    _schedule_batch_dispatch(context, npu, current_time_ms)


def _start_cross_request_layer0_prefetch(
    context: _Context,
    npu: _NPUState,
    current_time_ms: float,
    deadline_time_ms: float,
):
    if (
        not context.cross_request_layer0_prefetch
        or context.batch_size != 1
        or not npu.admission_queue
    ):
        return
    request = context.requests[npu.admission_queue[0]]
    if request.io_started[0] or npu.layer0_prefetch_request_id is not None:
        return
    _materialize_request(context, request)
    request.layer0_cross_request_prefetched = True
    npu.layer0_prefetch_request_id = request.manifest.request_id
    context.cross_request_layer0_prefetches += 1
    if context.causal_control is not None:
        request.layer0_manifest_controlled = True
        context.manifest_layer0_prefetches += 1
        context.pending_causal_request_prefetches[npu.npu_id] = (
            request.manifest.request_id,
            deadline_time_ms,
        )
        _queue_causal_control_event(context, current_time_ms)
        return
    # A manifest-aware controller may retarget this NPU's dedicated Path for
    # the queued request before admission.  The event is coalesced/rate-limited
    # by CIRControlConfig, while the I/O itself may be submitted immediately.
    _queue_control_event(context, current_time_ms, "batch_boundary")
    _start_layer_io(
        context,
        request,
        0,
        current_time_ms,
        deadline_time_ms,
        request.per_layer_compute_ms,
    )


def _handle_compute_schedule(context: _Context, npu_id: int, current_time_ms: float):
    npu = context.npus[npu_id]
    npu.compute_dispatch_pending = False
    batch = npu.active_batch
    if batch is None or npu.compute_active is not None:
        return
    layer = batch.compute_done_up_to + 1
    if layer >= context.n_layers or not all(
        context.requests[request_id].io_ready[layer]
        for request_id in batch.member_request_ids
    ):
        return
    duration_ms = _batch_compute_duration_ms(context, batch)
    observed_work = [0.0] * context.num_ssu
    for request_id in batch.member_request_ids:
        completed = context.requests[request_id].completed_gb_by_layer_ssu[layer]
        for ssu_id, work_gb in enumerate(completed):
            observed_work[ssu_id] += work_gb
    batch.observed_layer = layer
    batch.observed_work_gb_by_ssu = tuple(observed_work)
    if layer == 0 and npu.layer0_prefetch_request_id in batch.member_request_ids:
        npu.layer0_prefetch_request_id = None
    context.causal_observations += int(context.causal_control is not None)
    metric = batch.layer_metrics[layer]
    metric.compute_start_ms = current_time_ms
    metric.compute_duration_ms = duration_ms
    metric.compute_end_ms = current_time_ms + duration_ms
    metric.io_barrier_wait_ms = max(
        0.0, current_time_ms - batch.previous_compute_end_ms
    )
    batch.io_barrier_wait_ms += metric.io_barrier_wait_ms
    batch.compute_active_layer = layer
    npu.compute_active = (batch.batch_id, layer)
    npu.compute_generation += 1
    end_time_ms = metric.compute_end_ms
    if layer + 1 < context.n_layers:
        if context.causal_control is None:
            _start_batch_layer_io(
                context,
                batch,
                layer + 1,
                current_time_ms,
                end_time_ms,
            )
        else:
            context.pending_causal_prefetches[npu_id] = (
                batch.batch_id,
                layer + 1,
                end_time_ms,
            )
            _queue_causal_control_event(context, current_time_ms)
    else:
        _start_cross_request_layer0_prefetch(
            context,
            npu,
            current_time_ms,
            end_time_ms,
        )
    context.push_event(
        end_time_ms,
        sim.COMPUTE_DONE,
        npu_id,
        payload=(batch.batch_id, layer),
        generation=npu.compute_generation,
    )


def _schedule_control_if_due(context: _Context, current_time_ms: float):
    if context.control is None or context.control.every_layers is None:
        return
    if (
        context.control_threshold_jobs > 0
        and context.layer_jobs_since_control >= context.control_threshold_jobs
        and context.completed_requests < len(context.requests)
    ):
        _queue_control_event(context, current_time_ms, "fleet_layer")


def _handle_steady_measurement_start(context: _Context, current_time_ms: float):
    # Account partial commands exactly at the common window boundary.  A
    # completed-byte counter would miss a command that straddles either edge;
    # busy-time overlap is exact because every active SSD command runs at the
    # physical ``disk_bw_gbps`` rate.
    for disk in context.disks:
        disk.scheduler.settle(current_time_ms)
    context.measurement_disk_busy_start_ms = tuple(
        disk.busy_time for disk in context.disks
    )
    context.measurement_npu_ssu_served_start_gb = tuple(
        tuple(disk.served_gb_by_npu.get(npu_id, 0.0) for disk in context.disks)
        for npu_id in range(context.num_npu)
    )
    for npu in context.npus:
        _settle_npu_link(context, npu, current_time_ms)
    context.measurement_link_busy_start_ms = tuple(
        npu.link_accounted_busy_ms for npu in context.npus
    )
    context.measurement_npu_ssu_link_start_gb = tuple(
        tuple(
            npu.link_served_gb_by_ssu.get(ssu_id, 0.0)
            for ssu_id in range(context.num_ssu)
        )
        for npu in context.npus
    )
    context.measurement_disk_outstanding_start = tuple(
        disk.scheduler.outstanding_blocks for disk in context.disks
    )
    context.measurement_control_counters_start = _control_counter_snapshot(context)
    context.measurement_actual_cir_start = _actual_cir_state(context)
    context.measurement_start_ms = current_time_ms
    context.measurement_end_ms = current_time_ms + context.steady_state.measurement_ms
    context.measurement_open = True
    block_bounds = _steady_block_bounds(
        context.measurement_start_ms,
        context.measurement_end_ms,
        context.steady_state.measurement_ms,
        context.steady_state.block_ms,
    )
    boundary_times_ms = (context.measurement_start_ms,) + tuple(
        block_end_ms for _, block_end_ms, _ in block_bounds
    )
    for boundary, boundary_time_ms in enumerate(boundary_times_ms):
        context.push_event(
            boundary_time_ms,
            STEADY_STATIONARITY_SAMPLE,
            boundary,
        )
    context.measurement_request_ids.update(
        request.manifest.request_id
        for request in context.requests.values()
        if request.admitted and abs(request.admission_time_ms - current_time_ms) <= _EPS
    )
    if any(
        npu.active_batch is None
        and not npu.batch_dispatch_pending
        and not npu.admission_queue
        for npu in context.npus
    ):
        context.steady_backlog_exhausted = True
    context.push_event(
        context.measurement_end_ms,
        STEADY_MEASUREMENT_END,
        0,
    )


def _handle_steady_stationarity_sample(
    context: _Context,
    boundary: int,
    current_time_ms: float,
):
    """Append one read-only, pre-business-event measurement boundary."""
    if boundary != len(context.measurement_stationarity_snapshots):
        raise AssertionError("steady-state stationarity boundaries are out of order")
    context.measurement_stationarity_snapshots.append(
        _capture_stationarity_snapshot(context, boundary, current_time_ms)
    )


def _handle_steady_measurement_end(context: _Context, current_time_ms: float):
    for disk in context.disks:
        disk.scheduler.settle(current_time_ms)
    context.measurement_disk_busy_end_ms = tuple(
        disk.busy_time for disk in context.disks
    )
    context.measurement_npu_ssu_served_end_gb = tuple(
        tuple(disk.served_gb_by_npu.get(npu_id, 0.0) for disk in context.disks)
        for npu_id in range(context.num_npu)
    )
    for npu in context.npus:
        _settle_npu_link(context, npu, current_time_ms)
    context.measurement_link_busy_end_ms = tuple(
        npu.link_accounted_busy_ms for npu in context.npus
    )
    context.measurement_npu_ssu_link_end_gb = tuple(
        tuple(
            npu.link_served_gb_by_ssu.get(ssu_id, 0.0)
            for ssu_id in range(context.num_ssu)
        )
        for npu in context.npus
    )
    context.measurement_disk_outstanding_end = tuple(
        disk.scheduler.outstanding_blocks for disk in context.disks
    )
    context.measurement_control_counters_end = _control_counter_snapshot(context)
    context.measurement_actual_cir_end = _actual_cir_state(context)
    context.measurement_open = False
    context.measurement_ended = True


def _complete_microbatch(
    context: _Context,
    npu: _NPUState,
    batch: _MicrobatchState,
    current_time_ms: float,
):
    batch.completion_time_ms = current_time_ms
    for request_id in batch.member_request_ids:
        request = context.requests[request_id]
        request.completed = True
        request.completion_time_ms = current_time_ms
        request.placement_groups = None
        request.completed_gb_by_layer_ssu = []
    context.completed_requests += len(batch.member_request_ids)
    context.completed_by_npu[npu.npu_id] += len(batch.member_request_ids)
    context.measurement_completed_ids.update(
        request_id
        for request_id in batch.member_request_ids
        if request_id in context.measurement_request_ids
    )
    npu.last_completion_ms = current_time_ms
    npu.active_batch = None
    if (
        context.steady_state is not None
        and context.steady_warmup_reached_ms is None
        and min(context.completed_by_npu)
        >= context.steady_state.warmup_requests_per_npu
    ):
        context.steady_warmup_reached_ms = current_time_ms
        context.push_event(
            current_time_ms + context.steady_state.settle_ms,
            STEADY_MEASUREMENT_START,
            0,
        )
    _queue_control_event(context, current_time_ms, "batch_boundary")
    _queue_causal_control_event(context, current_time_ms)
    _schedule_batch_dispatch(context, npu, current_time_ms)
    if context.steady_warmup_reached_ms is not None and not npu.admission_queue:
        context.steady_backlog_exhausted = True


def _handle_compute_done(
    context: _Context,
    npu_id: int,
    payload,
    generation: int,
    current_time_ms: float,
):
    npu = context.npus[npu_id]
    if generation != npu.compute_generation or npu.compute_active != payload:
        context.stale_events += 1
        return
    batch_id, layer = payload
    batch = npu.active_batch
    if batch is None or batch.batch_id != batch_id:
        context.stale_events += 1
        return
    npu.compute_active = None
    batch.compute_active_layer = -1
    batch.compute_done_up_to = layer
    batch.previous_compute_end_ms = current_time_ms
    duration_ms = batch.layer_metrics[layer].compute_duration_ms
    batch.compute_busy_ms += duration_ms
    npu.compute_busy_ms += duration_ms
    member_count = len(batch.member_request_ids)
    context.completed_batch_layers += 1
    context.completed_request_layer_jobs += member_count
    context.layer_jobs_since_control += member_count
    if layer + 1 == context.n_layers:
        _complete_microbatch(context, npu, batch, current_time_ms)
    else:
        _schedule_compute_dispatch(context, npu, current_time_ms)
    _schedule_compute_dispatch(context, npu, current_time_ms)
    _schedule_control_if_due(context, current_time_ms)


def _control_request_view(context: _Context, request: _RequestState):
    batch = context.microbatches[request.batch_id] if request.admitted else None
    compute_done_up_to = batch.compute_done_up_to if batch is not None else -1
    next_layer = min(compute_done_up_to + 1, context.n_layers - 1)
    work = [0.0] * context.num_ssu
    for ssu_id, _, work_gb in _state_placement_groups(request, next_layer):
        work[ssu_id] = work_gb
    remaining_work = [0.0] * context.num_ssu
    remaining_io_layers = 0
    for layer in range(max(0, compute_done_up_to + 1), context.n_layers):
        # Ready layers no longer consume SSD bandwidth.  Future layers and an
        # in-progress barrier retain their full manifest work so the policy
        # does not depend on simulator-private partial-command progress.
        if request.io_ready[layer]:
            continue
        remaining_io_layers += 1
        for ssu_id, _, work_gb in _state_placement_groups(request, layer):
            remaining_work[ssu_id] += work_gb
    waiting_for_io = batch is None or (
        batch.compute_active_layer < 0
        and compute_done_up_to + 1 < context.n_layers
        and not request.io_ready[compute_done_up_to + 1]
    )
    return ControlRequestView(
        request_id=request.manifest.request_id,
        npu_id=request.manifest.npu_id,
        category=request.category,
        per_layer_compute_ms=request.per_layer_compute_ms,
        compute_done_up_to=compute_done_up_to,
        remaining_layers=context.n_layers - compute_done_up_to - 1,
        next_layer_work_gb_by_ssu=tuple(work),
        waiting_for_io=waiting_for_io,
        remaining_work_gb_by_ssu=tuple(remaining_work),
        remaining_compute_budget_ms=(
            remaining_io_layers * request.per_layer_compute_ms
        ),
        prefetch_only=batch is None,
    )


def _apply_cir_decision(
    context: _Context,
    decision: Optional[CIRControlDecision],
    current_time_ms: float,
):
    if decision is None:
        return
    if len(decision.path_cirs_by_ssu) != context.num_ssu:
        raise ValueError("controller must provide one CIR table per SSU")
    normalized_cirs = tuple(
        tuple(float(cir) for cir in cirs) for cirs in decision.path_cirs_by_ssu
    )
    for ssu_id, cirs in enumerate(normalized_cirs):
        scheduler = context.disks[ssu_id].scheduler
        if (
            len(cirs) != len(scheduler.paths)
            or any(cir < 0.0 for cir in cirs)
            or sum(cirs) > scheduler.disk_bw + _EPS
            or any(
                cir > scheduler.paths[path_id].pir + _EPS
                for path_id, cir in enumerate(cirs)
            )
        ):
            raise ValueError("controller returned an invalid CIR table")
    forced_path_ids_by_ssu = [set() for _ in range(context.num_ssu)]
    if context.npu_dedicated_paths is not None:
        cap_tolerance = 1e-9
        for npu_id, path_id in enumerate(context.npu_dedicated_paths):
            target_sum = math.fsum(
                normalized_cirs[ssu_id][path_id]
                for ssu_id in range(context.num_ssu)
            )
            if target_sum > context.npu_bw_gbps + cap_tolerance:
                raise ValueError(
                    f"controller target exceeds NPU {npu_id} CIR capacity"
                )

            projected_cirs = []
            suppressed_decreases = []
            for ssu_id, cirs in enumerate(normalized_cirs):
                scheduler = context.disks[ssu_id].scheduler
                old_cir = scheduler.paths[path_id].cir
                target_cir = cirs[path_id]
                delta = target_cir - old_cir
                if scheduler.cir_write_threshold_gbps <= 0.0:
                    selected = abs(delta) > _EPS
                else:
                    threshold = max(_EPS, scheduler.cir_write_threshold_gbps)
                    active_set_change = (old_cir > 0.0) != (target_cir > 0.0)
                    selected = active_set_change or abs(delta) > threshold
                projected_cirs.append(target_cir if selected else old_cir)
                if not selected and delta < 0.0:
                    suppressed_decreases.append((delta, ssu_id))

            projected_sum = math.fsum(projected_cirs)
            if projected_sum > context.npu_bw_gbps + cap_tolerance:
                # Largest suppressed releases first minimizes the number of
                # cross-SSU compensating register writes for this NPU.
                suppressed_decreases.sort(key=lambda item: (item[0], item[1]))
                for _, ssu_id in suppressed_decreases:
                    forced_path_ids_by_ssu[ssu_id].add(path_id)
                    projected_cirs[ssu_id] = normalized_cirs[ssu_id][path_id]
                    projected_sum = math.fsum(projected_cirs)
                    if projected_sum <= context.npu_bw_gbps + cap_tolerance:
                        break
            if projected_sum > context.npu_bw_gbps + cap_tolerance:
                raise AssertionError(
                    f"sparse CIR plan cannot preserve NPU {npu_id} capacity"
                )
    changed = 0
    # The DES exposes the fleet update atomically at one timestamp: no I/O
    # arbitration observes an intermediate SSU in this loop.  A real deployment
    # without an atomic fleet commit must issue every cross-SSU decrease before
    # any increase so that the NPU-link cap also holds during register writes.
    for ssu_id, cirs in enumerate(normalized_cirs):
        ssu_changed = context.disks[ssu_id].scheduler.update_path_cirs(
            cirs,
            current_time_ms,
            forced_path_ids=tuple(sorted(forced_path_ids_by_ssu[ssu_id])),
        )
        if ssu_changed:
            context.cir_write_transactions_by_ssu[ssu_id] += 1
            context.cir_path_writes_by_ssu[ssu_id] += ssu_changed
        changed += ssu_changed
        context.max_cir_sum_gbps = max(context.max_cir_sum_gbps, sum(cirs))
    # Sample only after the fleet decision has been applied.  This is the
    # coherent installed table seen by the next arbitration on every SSU.
    actual_cirs = _record_actual_cir_state(context)
    if any(
        value > context.disk_bw_gbps + 1e-9
        for value in actual_cirs["per_ssu_gbps"]
    ):
        raise AssertionError("installed CIR table exceeds an SSU capacity")
    if actual_cirs["per_npu_gbps"] is not None and any(
        value > context.npu_bw_gbps + 1e-9
        for value in actual_cirs["per_npu_gbps"]
    ):
        raise AssertionError("installed cross-SSU CIRs exceed an NPU capacity")
    if changed:
        context.cir_commits += 1
        context.cir_path_writes += changed


def _handle_control(context: _Context, current_time_ms: float):
    reasons = context.control_reasons_by_time.pop(float(current_time_ms), set())
    context.control_evaluations += 1
    active_request_ids = sorted(
        {
            request_id
            for npu in context.npus
            for request_id in (
                (
                    ()
                    if npu.active_batch is None
                    else npu.active_batch.member_request_ids
                )
                + (
                    ()
                    if npu.layer0_prefetch_request_id is None
                    else (npu.layer0_prefetch_request_id,)
                )
            )
        }
    )
    active_requests = tuple(
        _control_request_view(context, context.requests[request_id])
        for request_id in active_request_ids
    )
    current_cirs = tuple(
        tuple(path.cir for path in disk.scheduler.paths.values())
        for disk in context.disks
    )
    snapshot = CIRControlSnapshot(
        time_ms=current_time_ms,
        evaluation=context.control_evaluations,
        layer_jobs_since_previous=context.layer_jobs_since_control,
        num_npu=context.num_npu,
        num_ssu=context.num_ssu,
        active_requests=active_requests,
        current_path_cirs_by_ssu=current_cirs,
    )
    decision = context.control.callback(snapshot)
    _apply_cir_decision(context, decision, current_time_ms)
    context.last_control_ms = current_time_ms
    context.layer_jobs_since_control = 0
    active_request_count = sum(
        len(npu.active_batch.member_request_ids)
        for npu in context.npus
        if npu.active_batch is not None
    )
    if context.control.every_layers is not None:
        context.control_threshold_jobs = (
            context.control.every_layers * active_request_count
            if active_request_count
            else 0
        )
    elif "wall_clock" in reasons:
        context.control_threshold_jobs = 0
        context.next_wall_control_ms = current_time_ms + context.control.interval_ms
        if context.completed_requests < len(context.requests):
            _queue_control_event(
                context,
                context.next_wall_control_ms,
                "wall_clock",
            )


def _handle_causal_layer_control(context: _Context, current_time_ms: float):
    context.causal_control_pending = False
    context.control_evaluations += 1
    observations = []
    for npu in context.npus:
        prefetch_request_id = npu.layer0_prefetch_request_id
        if prefetch_request_id is not None:
            request = context.requests[prefetch_request_id]
            observations.append(
                CausalLayerObservation(
                    batch_id=-prefetch_request_id - 1,
                    npu_id=npu.npu_id,
                    observed_layer=0,
                    observed_work_gb_by_ssu=_layer_work_by_ssu(context, request, 0),
                    compute_budget_ms=request.per_layer_compute_ms,
                    manifest_layer0=True,
                )
            )
            continue
        batch = npu.active_batch
        if batch is None:
            continue
        observations.append(
            CausalLayerObservation(
                batch_id=batch.batch_id,
                npu_id=npu.npu_id,
                observed_layer=batch.observed_layer,
                observed_work_gb_by_ssu=batch.observed_work_gb_by_ssu,
                compute_budget_ms=_batch_compute_duration_ms(context, batch),
            )
        )
    snapshot = CausalLayerSnapshot(
        num_npu=context.num_npu,
        num_ssu=context.num_ssu,
        active_batches=tuple(observations),
    )
    decision = context.causal_control.callback(snapshot)
    _apply_cir_decision(context, decision, current_time_ms)

    pending = tuple(sorted(context.pending_causal_prefetches.items()))
    context.pending_causal_prefetches.clear()
    for npu_id, (batch_id, layer, deadline_time_ms) in pending:
        batch = context.npus[npu_id].active_batch
        if batch is not None and batch.batch_id == batch_id:
            _start_batch_layer_io(
                context,
                batch,
                layer,
                current_time_ms,
                deadline_time_ms,
            )
    request_prefetches = tuple(
        sorted(context.pending_causal_request_prefetches.items())
    )
    context.pending_causal_request_prefetches.clear()
    for npu_id, (request_id, deadline_time_ms) in request_prefetches:
        request = context.requests[request_id]
        if (
            context.npus[npu_id].layer0_prefetch_request_id == request_id
            and not request.io_started[0]
        ):
            _start_layer_io(
                context,
                request,
                0,
                current_time_ms,
                deadline_time_ms,
                request.per_layer_compute_ms,
                manifest_dedicated_path=True,
            )


def _settle_npu_link(
    context: _Context,
    npu: _NPUState,
    current_time_ms: float,
):
    """Attribute exact NPU-link service up to an external time boundary."""
    if current_time_ms <= npu.link_last_account_ms + _EPS:
        return
    flow = npu.link_active_flow
    if flow is not None:
        service_end_ms = min(current_time_ms, flow.link_end_time)
        duration_ms = max(0.0, service_end_ms - npu.link_last_account_ms)
        if duration_ms > 0.0:
            npu.link_accounted_busy_ms += duration_ms
            npu.link_served_gb_by_ssu[flow.disk_id] += (
                context.npu_bw_gbps * duration_ms / 1000.0
            )
    npu.link_last_account_ms = current_time_ms


def _start_next_link_io(context: _Context, npu: _NPUState, current_time_ms: float):
    if npu.link_active_flow is not None or not npu.link_pending:
        return
    _settle_npu_link(context, npu, current_time_ms)
    flow = npu.link_pending.popleft()
    npu.link_pending_gb -= flow.total_gb
    npu.link_active_flow = flow
    flow.link_start_time = current_time_ms
    service_ms = sim.npu_link_service_time_ms(flow.total_gb, context.npu_bw_gbps)
    flow.link_end_time = current_time_ms + service_ms
    npu.link_busy_ms += service_ms
    npu.link_generation += 1
    context.push_event(
        max(current_time_ms + _EPS, flow.link_end_time),
        sim.NPU_LINK_COMPLETION,
        npu.npu_id,
        generation=npu.link_generation,
    )


def _enqueue_link_io(context: _Context, flow: sim.BlockIOFlow, current_time_ms: float):
    npu = context.npus[flow.npu_id]
    flow.link_enqueue_time = current_time_ms
    npu.link_pending.append(flow)
    npu.link_pending_gb += flow.total_gb
    npu.max_link_outstanding = max(
        npu.max_link_outstanding,
        len(npu.link_pending) + int(npu.link_active_flow is not None),
    )
    context.pending_link_start_ids.add(npu.npu_id)


def _flush_link_starts(context: _Context, current_time_ms: float):
    npu_ids = sorted(context.pending_link_start_ids)
    context.pending_link_start_ids.clear()
    for npu_id in npu_ids:
        npu = context.npus[npu_id]
        simultaneous = []
        while (
            npu.link_pending
            and abs(npu.link_pending[-1].link_enqueue_time - current_time_ms) <= _EPS
        ):
            simultaneous.append(npu.link_pending.pop())
        simultaneous.sort(
            key=lambda flow: (
                flow.request_id,
                flow.layer,
                flow.block_idx,
                flow.disk_id,
            )
        )
        npu.link_pending.extend(simultaneous)
        _start_next_link_io(context, npu, current_time_ms)


def _handle_disk_completion(
    context: _Context,
    ssu_id: int,
    generation: int,
    current_time_ms: float,
):
    disk = context.disks[ssu_id]
    if generation != disk.generation:
        context.stale_events += 1
        return
    scheduler = disk.scheduler
    completed = scheduler.complete_ready_flows(current_time_ms)
    if not completed:
        raise AssertionError("SSD completion event had no completed command")
    for flow in completed:
        _enqueue_link_io(context, flow, current_time_ms)
    scheduler.request_dispatch(current_time_ms, context.event_heap)


def _handle_link_completion(
    context: _Context,
    npu_id: int,
    generation: int,
    current_time_ms: float,
):
    npu = context.npus[npu_id]
    flow = npu.link_active_flow
    if generation != npu.link_generation or flow is None:
        context.stale_events += 1
        return
    if current_time_ms + _EPS < flow.link_end_time:
        context.stale_events += 1
        return
    _settle_npu_link(context, npu, current_time_ms)
    npu.link_active_flow = None
    npu.link_completed_gb += flow.total_gb
    request = context.requests[flow.request_id]
    request.io_count += 1
    request.completed_gb_by_layer_ssu[flow.layer][flow.disk_id] += flow.total_gb
    request.ssd_queue_wait_ms += flow.ssd_queue_wait_ms
    request.link_queue_wait_ms += flow.link_start_time - flow.link_enqueue_time
    request.io_latency_ms += current_time_ms - flow.enqueue_time
    _register_complete(context, flow)
    request.pending_blocks[flow.layer] -= flow.block_count
    if request.pending_blocks[flow.layer] == 0:
        _mark_layer_io_ready(context, request, flow.layer, current_time_ms)
    _start_next_link_io(context, npu, current_time_ms)


def _input_fingerprint(requests: Sequence[ContinuousBatchRequest]):
    digest = hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    for request in sorted(requests, key=lambda item: item.request_id):
        digest.update(
            repr(
                (
                    request.request_id,
                    request.npu_id,
                    request.arrival_time_ms,
                    request.load["category"],
                    request.load["per_layer_us"],
                    request.placement,
                )
            ).encode()
        )
    return digest.hexdigest()


def continuous_batch_input_fingerprint(
    requests: Sequence[ContinuousBatchRequest],
) -> str:
    """Return the strategy-independent fingerprint used by result pairing."""
    return _input_fingerprint(tuple(requests))


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else 0.0


def _request_profile_metadata(load: Mapping[str, object]):
    """Normalize real-data profile identity already carried by a request."""
    raw_key = load.get("profile_key")
    if raw_key is None and "seq_len_k" in load and "nql" in load:
        raw_key = (load["seq_len_k"], load["nql"])
    profile_key = None
    if raw_key is not None:
        profile_key = [int(value) for value in raw_key]
    canonical_id = (
        ",".join(str(value) for value in profile_key)
        if profile_key is not None
        else None
    )
    profile_id = load.get("profile_id", canonical_id)
    profile_name = load.get(
        "profile_name",
        (
            f"seq_len_k={profile_key[0]},nql={profile_key[1]}"
            if profile_key is not None and len(profile_key) == 2
            else canonical_id
        ),
    )
    raw_demand = load.get(
        "raw_demand_gbps",
        load.get("required_bw_input_gbps"),
    )
    return {
        "profile_id": profile_id,
        "profile_key": profile_key,
        "profile_name": profile_name,
        "raw_demand_gbps": (float(raw_demand) if raw_demand is not None else None),
    }


def _window_overlap_ms(start_ms, end_ms, window_start_ms, window_end_ms):
    return max(
        0.0,
        min(end_ms, window_end_ms) - max(start_ms, window_start_ms),
    )


def _steady_block_bounds(window_start_ms, window_end_ms, measurement_ms, block_ms):
    """Build positive blocks from configured durations, not float subtraction."""
    block_count = int(math.ceil(measurement_ms / block_ms))
    bounds = []
    for block in range(block_count):
        offset_ms = block * block_ms
        duration_ms = min(block_ms, measurement_ms - offset_ms)
        block_start_ms = window_start_ms + offset_ms
        block_end_ms = (
            window_end_ms if block + 1 == block_count else block_start_ms + duration_ms
        )
        bounds.append((block_start_ms, block_end_ms, duration_ms))
    return tuple(bounds)


def _vectors_close(left, right, *, abs_tol=1e-8):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=abs_tol) for a, b in zip(left, right)
    )


def _nonnegative_delta(end, start):
    delta = float(end) - float(start)
    return max(0.0, delta) if delta >= -1e-8 else delta


def _build_stationarity_metrics(
    context: _Context,
    block_bounds,
    *,
    compute_ms_by_npu,
    block_compute_ms_by_npu,
    disk_busy_ms_by_ssu,
    disk_served_gb_by_ssu,
    npu_ssu_served_gb,
    link_busy_ms_by_npu,
    npu_ssu_link_served_gb,
):
    """Reconstruct every measurement block from read-only boundary samples."""
    snapshots = context.measurement_stationarity_snapshots
    expected_times_ms = (context.measurement_start_ms,) + tuple(
        block_end_ms for _, block_end_ms, _ in block_bounds
    )
    boundary_count_ok = len(snapshots) == len(block_bounds) + 1
    if not boundary_count_ok:
        raise AssertionError(
            "steady-state stationarity boundary count does not match block count"
        )

    per_ssu_fields = (
        "ssd_cumulative_busy_ms_by_ssu",
        "ssd_cumulative_served_gb_by_ssu",
        "ssd_outstanding_blocks_by_ssu",
        "ssd_outstanding_gb_by_ssu",
    )
    per_npu_fields = (
        "npu_compute_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_served_gb_by_npu",
        "npu_link_outstanding_blocks_by_npu",
        "npu_link_outstanding_gb_by_npu",
    )
    shape_ok = all(
        len(snapshot[field]) == context.num_ssu
        for snapshot in snapshots
        for field in per_ssu_fields
    ) and all(
        len(snapshot[field]) == context.num_npu
        for snapshot in snapshots
        for field in per_npu_fields
    )
    if not shape_ok:
        raise AssertionError("steady-state stationarity snapshot shape changed")

    boundary_times_exact = all(
        snapshot["boundary"] == boundary
        and math.isclose(
            snapshot["time_ms"],
            expected_time_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for boundary, (snapshot, expected_time_ms) in enumerate(
            zip(snapshots, expected_times_ms)
        )
    )
    all_values_nonnegative = all(
        math.isfinite(float(value)) and float(value) >= -1e-8
        for snapshot in snapshots
        for field in per_ssu_fields + per_npu_fields
        for value in snapshot[field]
    )
    cumulative_fields = (
        "ssd_cumulative_busy_ms_by_ssu",
        "ssd_cumulative_served_gb_by_ssu",
        "npu_compute_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_busy_ms_by_npu",
        "npu_link_cumulative_served_gb_by_npu",
    )
    cumulative_monotonic = all(
        float(end) + 1e-8 >= float(start)
        for previous, current in zip(snapshots, snapshots[1:])
        for field in cumulative_fields
        for start, end in zip(previous[field], current[field])
    )
    cumulative_service_consistency = all(
        math.isclose(
            served_gb,
            busy_ms * context.disk_bw_gbps / 1000.0,
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        for snapshot in snapshots
        for busy_ms, served_gb in zip(
            snapshot["ssd_cumulative_busy_ms_by_ssu"],
            snapshot["ssd_cumulative_served_gb_by_ssu"],
        )
    ) and all(
        math.isclose(
            served_gb,
            busy_ms * context.npu_bw_gbps / 1000.0,
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        for snapshot in snapshots
        for busy_ms, served_gb in zip(
            snapshot["npu_link_cumulative_busy_ms_by_npu"],
            snapshot["npu_link_cumulative_served_gb_by_npu"],
        )
    )

    first = snapshots[0]
    last = snapshots[-1]
    whole_ssd_busy_ms = [
        _nonnegative_delta(end, start)
        for start, end in zip(
            first["ssd_cumulative_busy_ms_by_ssu"],
            last["ssd_cumulative_busy_ms_by_ssu"],
        )
    ]
    whole_ssd_served_gb = [
        _nonnegative_delta(end, start)
        for start, end in zip(
            first["ssd_cumulative_served_gb_by_ssu"],
            last["ssd_cumulative_served_gb_by_ssu"],
        )
    ]
    whole_compute_ms = [
        _nonnegative_delta(end, start)
        for start, end in zip(
            first["npu_compute_cumulative_busy_ms_by_npu"],
            last["npu_compute_cumulative_busy_ms_by_npu"],
        )
    ]
    whole_link_busy_ms = [
        _nonnegative_delta(end, start)
        for start, end in zip(
            first["npu_link_cumulative_busy_ms_by_npu"],
            last["npu_link_cumulative_busy_ms_by_npu"],
        )
    ]
    whole_link_served_gb = [
        _nonnegative_delta(end, start)
        for start, end in zip(
            first["npu_link_cumulative_served_gb_by_npu"],
            last["npu_link_cumulative_served_gb_by_npu"],
        )
    ]

    resource_rows = []
    for block, (start, end, bounds) in enumerate(
        zip(snapshots, snapshots[1:], block_bounds)
    ):
        _, _, duration_ms = bounds
        ssd_busy_ms = [
            _nonnegative_delta(end_value, start_value)
            for start_value, end_value in zip(
                start["ssd_cumulative_busy_ms_by_ssu"],
                end["ssd_cumulative_busy_ms_by_ssu"],
            )
        ]
        ssd_served_gb = [
            _nonnegative_delta(end_value, start_value)
            for start_value, end_value in zip(
                start["ssd_cumulative_served_gb_by_ssu"],
                end["ssd_cumulative_served_gb_by_ssu"],
            )
        ]
        npu_compute_ms = [
            _nonnegative_delta(end_value, start_value)
            for start_value, end_value in zip(
                start["npu_compute_cumulative_busy_ms_by_npu"],
                end["npu_compute_cumulative_busy_ms_by_npu"],
            )
        ]
        link_busy_ms = [
            _nonnegative_delta(end_value, start_value)
            for start_value, end_value in zip(
                start["npu_link_cumulative_busy_ms_by_npu"],
                end["npu_link_cumulative_busy_ms_by_npu"],
            )
        ]
        link_served_gb = [
            _nonnegative_delta(end_value, start_value)
            for start_value, end_value in zip(
                start["npu_link_cumulative_served_gb_by_npu"],
                end["npu_link_cumulative_served_gb_by_npu"],
            )
        ]
        resource_rows.append(
            {
                "ssd_busy_ms_by_ssu": ssd_busy_ms,
                "ssd_served_gb_by_ssu": ssd_served_gb,
                "ssd_utilizations": [value / duration_ms for value in ssd_busy_ms],
                "ssd_mean_utilization": float(
                    np.mean([value / duration_ms for value in ssd_busy_ms])
                ),
                "ssd_outstanding_blocks_at_start": list(
                    start["ssd_outstanding_blocks_by_ssu"]
                ),
                "ssd_outstanding_blocks_at_end": list(
                    end["ssd_outstanding_blocks_by_ssu"]
                ),
                "ssd_outstanding_blocks_delta": [
                    int(end_value) - int(start_value)
                    for start_value, end_value in zip(
                        start["ssd_outstanding_blocks_by_ssu"],
                        end["ssd_outstanding_blocks_by_ssu"],
                    )
                ],
                "ssd_outstanding_gb_at_start": list(start["ssd_outstanding_gb_by_ssu"]),
                "ssd_outstanding_gb_at_end": list(end["ssd_outstanding_gb_by_ssu"]),
                "ssd_outstanding_gb_delta": [
                    float(end_value) - float(start_value)
                    for start_value, end_value in zip(
                        start["ssd_outstanding_gb_by_ssu"],
                        end["ssd_outstanding_gb_by_ssu"],
                    )
                ],
                "compute_ms_by_npu": npu_compute_ms,
                "npu_utilizations": [value / duration_ms for value in npu_compute_ms],
                "npu_link_busy_ms_by_npu": link_busy_ms,
                "npu_link_served_gb_by_npu": link_served_gb,
                "npu_link_utilizations": [
                    value / duration_ms for value in link_busy_ms
                ],
                "npu_link_mean_utilization": float(
                    np.mean([value / duration_ms for value in link_busy_ms])
                ),
                "npu_link_outstanding_blocks_at_start": list(
                    start["npu_link_outstanding_blocks_by_npu"]
                ),
                "npu_link_outstanding_blocks_at_end": list(
                    end["npu_link_outstanding_blocks_by_npu"]
                ),
                "npu_link_outstanding_blocks_delta": [
                    int(end_value) - int(start_value)
                    for start_value, end_value in zip(
                        start["npu_link_outstanding_blocks_by_npu"],
                        end["npu_link_outstanding_blocks_by_npu"],
                    )
                ],
                "npu_link_outstanding_gb_at_start": list(
                    start["npu_link_outstanding_gb_by_npu"]
                ),
                "npu_link_outstanding_gb_at_end": list(
                    end["npu_link_outstanding_gb_by_npu"]
                ),
                "npu_link_outstanding_gb_delta": [
                    float(end_value) - float(start_value)
                    for start_value, end_value in zip(
                        start["npu_link_outstanding_gb_by_npu"],
                        end["npu_link_outstanding_gb_by_npu"],
                    )
                ],
            }
        )

    block_sums_match_whole = all(
        _vectors_close(
            [sum(row[field][index] for row in resource_rows) for index in range(size)],
            whole,
        )
        for field, size, whole in (
            ("ssd_busy_ms_by_ssu", context.num_ssu, whole_ssd_busy_ms),
            ("ssd_served_gb_by_ssu", context.num_ssu, whole_ssd_served_gb),
            ("compute_ms_by_npu", context.num_npu, whole_compute_ms),
            ("npu_link_busy_ms_by_npu", context.num_npu, whole_link_busy_ms),
            (
                "npu_link_served_gb_by_npu",
                context.num_npu,
                whole_link_served_gb,
            ),
        )
    )
    independent_compute_match = all(
        _vectors_close(row["compute_ms_by_npu"], independent)
        for row, independent in zip(resource_rows, block_compute_ms_by_npu)
    )
    legacy_edges_match = (
        _vectors_close(
            first["ssd_cumulative_busy_ms_by_ssu"],
            context.measurement_disk_busy_start_ms,
        )
        and _vectors_close(
            last["ssd_cumulative_busy_ms_by_ssu"],
            context.measurement_disk_busy_end_ms,
        )
        and _vectors_close(
            first["npu_link_cumulative_busy_ms_by_npu"],
            context.measurement_link_busy_start_ms,
        )
        and _vectors_close(
            last["npu_link_cumulative_busy_ms_by_npu"],
            context.measurement_link_busy_end_ms,
        )
        and _vectors_close(
            first["ssd_cumulative_served_gb_by_ssu"],
            [
                sum(row[ssu_id] for row in context.measurement_npu_ssu_served_start_gb)
                for ssu_id in range(context.num_ssu)
            ],
        )
        and _vectors_close(
            last["ssd_cumulative_served_gb_by_ssu"],
            [
                sum(row[ssu_id] for row in context.measurement_npu_ssu_served_end_gb)
                for ssu_id in range(context.num_ssu)
            ],
        )
        and _vectors_close(
            first["npu_link_cumulative_served_gb_by_npu"],
            [sum(row) for row in context.measurement_npu_ssu_link_start_gb],
        )
        and _vectors_close(
            last["npu_link_cumulative_served_gb_by_npu"],
            [sum(row) for row in context.measurement_npu_ssu_link_end_gb],
        )
        and list(first["ssd_outstanding_blocks_by_ssu"])
        == list(context.measurement_disk_outstanding_start)
        and list(last["ssd_outstanding_blocks_by_ssu"])
        == list(context.measurement_disk_outstanding_end)
    )
    whole_window_match = (
        _vectors_close(whole_ssd_busy_ms, disk_busy_ms_by_ssu)
        and _vectors_close(whole_ssd_served_gb, disk_served_gb_by_ssu)
        and _vectors_close(whole_compute_ms, compute_ms_by_npu)
        and _vectors_close(whole_link_busy_ms, link_busy_ms_by_npu)
        and _vectors_close(
            whole_link_served_gb,
            [sum(row) for row in npu_ssu_link_served_gb],
        )
        and _vectors_close(
            whole_ssd_served_gb,
            [
                sum(row[ssu_id] for row in npu_ssu_served_gb)
                for ssu_id in range(context.num_ssu)
            ],
        )
    )
    block_resource_bounds = all(
        _steady_resource_overlap_within_bounds(value, duration_ms)
        for row, (_, _, duration_ms) in zip(resource_rows, block_bounds)
        for field in (
            "ssd_busy_ms_by_ssu",
            "compute_ms_by_npu",
            "npu_link_busy_ms_by_npu",
        )
        for value in row[field]
    )
    invariants = {
        "stationarity_boundary_count": boundary_count_ok,
        "stationarity_boundary_times_exact": boundary_times_exact,
        "stationarity_snapshot_shapes": shape_ok,
        "stationarity_values_nonnegative": all_values_nonnegative,
        "stationarity_cumulative_monotonic": cumulative_monotonic,
        "stationarity_cumulative_service_consistency": (cumulative_service_consistency),
        "stationarity_block_sums_match_whole": block_sums_match_whole,
        "stationarity_independent_compute_match": independent_compute_match,
        "stationarity_legacy_edges_match": legacy_edges_match,
        "stationarity_whole_window_match": whole_window_match,
        "stationarity_block_resource_bounds": block_resource_bounds,
    }
    return {
        "snapshots": snapshots,
        "block_resources": resource_rows,
        "invariants": invariants,
        "whole_ssd_busy_ms": whole_ssd_busy_ms,
        "whole_ssd_served_gb": whole_ssd_served_gb,
        "whole_compute_ms": whole_compute_ms,
        "whole_link_busy_ms": whole_link_busy_ms,
        "whole_link_served_gb": whole_link_served_gb,
    }


def _build_steady_state_summary(context: _Context, current_time_ms, events_processed):
    config = context.steady_state
    window_start_ms = context.measurement_start_ms
    window_end_ms = context.measurement_end_ms
    duration_ms = config.measurement_ms
    if (
        context.measurement_disk_busy_start_ms is None
        or context.measurement_disk_busy_end_ms is None
        or context.measurement_npu_ssu_served_start_gb is None
        or context.measurement_npu_ssu_served_end_gb is None
        or context.measurement_link_busy_start_ms is None
        or context.measurement_link_busy_end_ms is None
        or context.measurement_npu_ssu_link_start_gb is None
        or context.measurement_npu_ssu_link_end_gb is None
        or context.measurement_disk_outstanding_start is None
        or context.measurement_disk_outstanding_end is None
        or context.measurement_control_counters_start is None
        or context.measurement_control_counters_end is None
        or context.measurement_actual_cir_start is None
        or context.measurement_actual_cir_end is None
    ):
        raise AssertionError("steady-state SSD window snapshots are missing")
    measurement_control_metrics = _measurement_control_metrics(context, duration_ms)
    actual_cir_at_stop = _record_actual_cir_state(context)
    max_actual_ssu_cirs = list(context.max_actual_cir_sum_gbps_by_ssu)
    max_actual_npu_cirs = (
        None
        if context.max_actual_npu_cir_sum_gbps_by_npu is None
        else list(context.max_actual_npu_cir_sum_gbps_by_npu)
    )
    disk_busy_ms_by_ssu = [
        max(0.0, end_ms - start_ms)
        for start_ms, end_ms in zip(
            context.measurement_disk_busy_start_ms,
            context.measurement_disk_busy_end_ms,
        )
    ]
    disk_utilizations = [busy_ms / duration_ms for busy_ms in disk_busy_ms_by_ssu]
    disk_served_gb_by_ssu = [
        busy_ms * context.disk_bw_gbps / 1000.0 for busy_ms in disk_busy_ms_by_ssu
    ]
    disk_served_gbps_by_ssu = [
        served_gb * 1000.0 / duration_ms for served_gb in disk_served_gb_by_ssu
    ]
    npu_ssu_served_gb = [
        [max(0.0, end_gb - start_gb) for start_gb, end_gb in zip(start_row, end_row)]
        for start_row, end_row in zip(
            context.measurement_npu_ssu_served_start_gb,
            context.measurement_npu_ssu_served_end_gb,
        )
    ]
    npu_ssu_served_gbps = [
        [served_gb * 1000.0 / duration_ms for served_gb in row]
        for row in npu_ssu_served_gb
    ]
    link_busy_ms_by_npu = [
        max(0.0, end_ms - start_ms)
        for start_ms, end_ms in zip(
            context.measurement_link_busy_start_ms,
            context.measurement_link_busy_end_ms,
        )
    ]
    npu_ssu_link_served_gb = [
        [max(0.0, end_gb - start_gb) for start_gb, end_gb in zip(start_row, end_row)]
        for start_row, end_row in zip(
            context.measurement_npu_ssu_link_start_gb,
            context.measurement_npu_ssu_link_end_gb,
        )
    ]
    npu_ssu_link_served_gbps = [
        [delivered_gb * 1000.0 / duration_ms for delivered_gb in row]
        for row in npu_ssu_link_served_gb
    ]
    link_utilizations = [busy_ms / duration_ms for busy_ms in link_busy_ms_by_npu]
    compute_ms_by_npu = [0.0] * context.num_npu
    block_bounds = _steady_block_bounds(
        window_start_ms,
        window_end_ms,
        config.measurement_ms,
        config.block_ms,
    )
    block_compute_ms_by_npu = [
        [0.0] * context.num_npu for _ in range(len(block_bounds))
    ]
    for batch in context.microbatches:
        for metric in batch.layer_metrics:
            if not (
                math.isfinite(metric.compute_start_ms)
                and math.isfinite(metric.compute_end_ms)
            ):
                continue
            overlap_ms = _window_overlap_ms(
                metric.compute_start_ms,
                metric.compute_end_ms,
                window_start_ms,
                window_end_ms,
            )
            compute_ms_by_npu[batch.npu_id] += overlap_ms
            for block, (block_start_ms, block_end_ms, _) in enumerate(block_bounds):
                block_compute_ms_by_npu[block][batch.npu_id] += _window_overlap_ms(
                    metric.compute_start_ms,
                    metric.compute_end_ms,
                    block_start_ms,
                    block_end_ms,
                )

    block_compute_ms = [sum(row) for row in block_compute_ms_by_npu]

    request_rows = []
    outcomes_by_npu = [[] for _ in range(context.num_npu)]
    for request_id in sorted(context.measurement_request_ids):
        request = context.requests[request_id]
        ideal_ttft_ms = context.n_layers * request.per_layer_compute_ms
        ttft_ms = request.completion_time_ms - request.admission_time_ms
        slo_met = ttft_ms <= config.slo_alpha * ideal_ttft_ms + _EPS
        outcomes_by_npu[request.manifest.npu_id].append(slo_met)
        request_rows.append(
            {
                "request_id": request_id,
                "npu_id": request.manifest.npu_id,
                "sequence": int(request.manifest.load["stream_id"]),
                "category": request.category,
                **_request_profile_metadata(request.manifest.load),
                "admission_time_ms": request.admission_time_ms,
                "completion_time_ms": request.completion_time_ms,
                "ttft_ms": ttft_ms,
                "ideal_ttft_ms": ideal_ttft_ms,
                "slo_met": bool(slo_met),
            }
        )

    request_counts_by_npu = [len(rows) for rows in outcomes_by_npu]
    all_npus_sampled = all(request_counts_by_npu)
    exact_measurement_request_ids = {
        request.manifest.request_id
        for request in context.requests.values()
        if request.admitted
        and window_start_ms <= request.admission_time_ms < window_end_ms
    }
    per_npu_slo = [float(np.mean(rows)) for rows in outcomes_by_npu if rows]
    npu_utilizations = [busy_ms / duration_ms for busy_ms in compute_ms_by_npu]
    stationarity = _build_stationarity_metrics(
        context,
        block_bounds,
        compute_ms_by_npu=compute_ms_by_npu,
        block_compute_ms_by_npu=block_compute_ms_by_npu,
        disk_busy_ms_by_ssu=disk_busy_ms_by_ssu,
        disk_served_gb_by_ssu=disk_served_gb_by_ssu,
        npu_ssu_served_gb=npu_ssu_served_gb,
        link_busy_ms_by_npu=link_busy_ms_by_npu,
        npu_ssu_link_served_gb=npu_ssu_link_served_gb,
    )
    block_rows = []
    for block, (compute_ms, bounds, resources) in enumerate(
        zip(block_compute_ms, block_bounds, stationarity["block_resources"])
    ):
        block_start_ms, block_end_ms, block_duration_ms = bounds
        admitted = [
            row
            for row in request_rows
            if block_start_ms <= row["admission_time_ms"] < block_end_ms
        ]
        block_rows.append(
            {
                "block": block,
                "start_ms": block_start_ms,
                "end_ms": block_end_ms,
                "duration_ms": block_duration_ms,
                "npu_utilization": compute_ms / (context.num_npu * block_duration_ms),
                "request_count": len(admitted),
                "request_weighted_slo_attainment": float(
                    np.mean([row["slo_met"] for row in admitted])
                )
                if admitted
                else None,
                **resources,
            }
        )

    invariants = {
        "warmup_reached_all_npus": min(context.completed_by_npu)
        >= config.warmup_requests_per_npu,
        "measurement_window_closed": context.measurement_ended,
        "measurement_duration_exact": math.isclose(
            window_end_ms - window_start_ms,
            config.measurement_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "all_npus_sampled_for_slo": all_npus_sampled,
        "all_tagged_requests_completed": (
            context.measurement_completed_ids == context.measurement_request_ids
        ),
        "tagged_admissions_inside_window": all(
            window_start_ms <= row["admission_time_ms"] < window_end_ms
            for row in request_rows
        ),
        "all_window_admissions_tagged": context.measurement_request_ids
        == exact_measurement_request_ids,
        "no_backlog_exhaustion": not context.steady_backlog_exhausted,
        "compute_overlap_bounds": all(
            _steady_resource_overlap_within_bounds(busy_ms, duration_ms)
            for busy_ms in compute_ms_by_npu
        ),
        "ssd_overlap_bounds": all(
            _steady_resource_overlap_within_bounds(busy_ms, duration_ms)
            for busy_ms in disk_busy_ms_by_ssu
        ),
        "ssd_service_attribution": all(
            math.isclose(
                sum(row[ssu_id] for row in npu_ssu_served_gb),
                disk_served_gb_by_ssu[ssu_id],
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            for ssu_id in range(context.num_ssu)
        ),
        "npu_link_overlap_bounds": all(
            _steady_resource_overlap_within_bounds(busy_ms, duration_ms)
            for busy_ms in link_busy_ms_by_npu
        ),
        "npu_link_service_attribution": all(
            math.isclose(
                sum(npu_ssu_link_served_gb[npu_id]),
                link_busy_ms_by_npu[npu_id] * context.npu_bw_gbps / 1000.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            for npu_id in range(context.num_npu)
        ),
        "one_command_per_ssd": all(
            disk.scheduler.max_backend_active_io <= 1 for disk in context.disks
        ),
        "cir_capacity": context.max_cir_sum_gbps <= context.disk_bw_gbps + 1e-9,
        "actual_cir_per_ssu_capacity": all(
            value <= context.disk_bw_gbps + 1e-9 for value in max_actual_ssu_cirs
        ),
        "actual_cir_per_npu_capacity": (
            max_actual_npu_cirs is None
            or all(value <= context.npu_bw_gbps + 1e-9 for value in max_actual_npu_cirs)
        ),
        "cir_entry_write_counter_consistency": (
            context.cir_path_writes == sum(context.cir_path_writes_by_ssu)
        ),
        **stationarity["invariants"],
    }
    if not all(invariants.values()):
        diagnostics = {
            "measurement_start_ms": window_start_ms,
            "measurement_end_ms": window_end_ms,
            "measurement_duration_ms": duration_ms,
            "drain_stop_ms": current_time_ms,
            "measurement_ssd_busy_ms_by_ssu": disk_busy_ms_by_ssu,
            "measurement_ssd_busy_excess_ms_by_ssu": [
                busy_ms - duration_ms for busy_ms in disk_busy_ms_by_ssu
            ],
            "measurement_ssd_cumulative_busy_start_ms_by_ssu": list(
                context.measurement_disk_busy_start_ms
            ),
            "measurement_ssd_cumulative_busy_end_ms_by_ssu": list(
                context.measurement_disk_busy_end_ms
            ),
            "stationarity_whole_ssd_busy_ms_by_ssu": stationarity[
                "whole_ssd_busy_ms"
            ],
            "ssd_overlap_tolerance_ms": STEADY_ACCOUNTING_TOLERANCE_MS,
            "ssd_overlap_offending_ssu_ids": [
                ssu_id
                for ssu_id, busy_ms in enumerate(disk_busy_ms_by_ssu)
                if not _steady_resource_overlap_within_bounds(busy_ms, duration_ms)
            ],
            "completed_by_npu_at_stop": list(context.completed_by_npu),
            "request_counts_by_npu": request_counts_by_npu,
            "actual_cir_sum_gbps_by_ssu_at_stop": list(
                actual_cir_at_stop["per_ssu_gbps"]
            ),
            "max_actual_cir_sum_gbps_by_ssu": max_actual_ssu_cirs,
            "actual_npu_cir_sum_gbps_at_stop": (
                None
                if actual_cir_at_stop["per_npu_gbps"] is None
                else list(actual_cir_at_stop["per_npu_gbps"])
            ),
            "max_actual_npu_cir_sum_gbps_by_npu": max_actual_npu_cirs,
            "stationarity_boundary_count": len(
                context.measurement_stationarity_snapshots
            ),
            "stationarity_boundary_times_ms": [
                snapshot["time_ms"]
                for snapshot in context.measurement_stationarity_snapshots
            ],
        }
        failed = {name for name, holds in invariants.items() if not holds}
        collectible = {
            "all_npus_sampled_for_slo",
            "no_backlog_exhaustion",
        }
        if failed <= collectible:
            raise SteadyStateInvariantError(invariants, diagnostics)
        raise AssertionError(
            f"steady-state correctness invariant failed: {invariants}; "
            f"diagnostics: {diagnostics}"
        )

    ttfts = [row["ttft_ms"] for row in request_rows]
    return {
        "schema_version": 2,
        "mode": "steady_state_full_load",
        "num_npu": context.num_npu,
        "num_ssu": context.num_ssu,
        "n_layers": context.n_layers,
        "batch_size": context.batch_size,
        "warmup_requests_per_npu": config.warmup_requests_per_npu,
        "warmup_reached_ms": context.steady_warmup_reached_ms,
        "settle_ms": config.settle_ms,
        "measurement_start_ms": window_start_ms,
        "measurement_end_ms": window_end_ms,
        "measurement_duration_ms": duration_ms,
        "stationarity_boundary_semantics": (
            "read-only left-limit snapshot before workload events at the same time"
        ),
        "measurement_stationarity_boundary_count": len(stationarity["snapshots"]),
        "measurement_stationarity_boundaries": stationarity["snapshots"],
        "measurement_control_counter_window": (
            "half-open [measurement_start_ms, measurement_end_ms)"
        ),
        "legacy_control_counter_scope": (
            "cumulative from simulation start through tagged-request drain"
        ),
        "pressure_ttl_ms": context.pressure_ttl_ms,
        "cir_write_threshold_gbps": context.cir_write_threshold_gbps,
        "drain_stop_ms": current_time_ms,
        "tail_drain_ms": current_time_ms - window_end_ms,
        "slo_alpha": config.slo_alpha,
        "mean_npu_utilization": float(np.mean(npu_utilizations)),
        "npu_utilizations": npu_utilizations,
        "compute_ms_by_npu": compute_ms_by_npu,
        "measurement_ssd_mean_utilization": float(np.mean(disk_utilizations)),
        "measurement_ssd_utilizations": disk_utilizations,
        "measurement_ssd_busy_ms_by_ssu": disk_busy_ms_by_ssu,
        "measurement_ssd_served_gb_by_ssu": disk_served_gb_by_ssu,
        "measurement_ssd_served_gbps_by_ssu": disk_served_gbps_by_ssu,
        "measurement_npu_ssu_ssd_served_gb": npu_ssu_served_gb,
        "measurement_npu_ssu_ssd_served_gbps": npu_ssu_served_gbps,
        "measurement_npu_link_mean_utilization": float(np.mean(link_utilizations)),
        "measurement_npu_link_utilizations": link_utilizations,
        "measurement_npu_link_busy_ms_by_npu": link_busy_ms_by_npu,
        "measurement_npu_ssu_link_served_gb": npu_ssu_link_served_gb,
        "measurement_npu_ssu_link_served_gbps": (npu_ssu_link_served_gbps),
        "measurement_ssd_outstanding_blocks_at_start": list(
            context.measurement_disk_outstanding_start
        ),
        "measurement_ssd_outstanding_blocks_at_end": list(
            context.measurement_disk_outstanding_end
        ),
        "measurement_ssd_outstanding_blocks_drift": [
            int(end) - int(start)
            for start, end in zip(
                context.measurement_disk_outstanding_start,
                context.measurement_disk_outstanding_end,
            )
        ],
        "measurement_ssd_outstanding_gb_at_start": list(
            stationarity["snapshots"][0]["ssd_outstanding_gb_by_ssu"]
        ),
        "measurement_ssd_outstanding_gb_at_end": list(
            stationarity["snapshots"][-1]["ssd_outstanding_gb_by_ssu"]
        ),
        "measurement_ssd_outstanding_gb_drift": [
            float(end) - float(start)
            for start, end in zip(
                stationarity["snapshots"][0]["ssd_outstanding_gb_by_ssu"],
                stationarity["snapshots"][-1]["ssd_outstanding_gb_by_ssu"],
            )
        ],
        "measurement_npu_link_outstanding_blocks_at_start": list(
            stationarity["snapshots"][0]["npu_link_outstanding_blocks_by_npu"]
        ),
        "measurement_npu_link_outstanding_blocks_at_end": list(
            stationarity["snapshots"][-1]["npu_link_outstanding_blocks_by_npu"]
        ),
        "measurement_npu_link_outstanding_blocks_drift": [
            int(end) - int(start)
            for start, end in zip(
                stationarity["snapshots"][0]["npu_link_outstanding_blocks_by_npu"],
                stationarity["snapshots"][-1]["npu_link_outstanding_blocks_by_npu"],
            )
        ],
        "measurement_npu_link_outstanding_gb_at_start": list(
            stationarity["snapshots"][0]["npu_link_outstanding_gb_by_npu"]
        ),
        "measurement_npu_link_outstanding_gb_at_end": list(
            stationarity["snapshots"][-1]["npu_link_outstanding_gb_by_npu"]
        ),
        "measurement_npu_link_outstanding_gb_drift": [
            float(end) - float(start)
            for start, end in zip(
                stationarity["snapshots"][0]["npu_link_outstanding_gb_by_npu"],
                stationarity["snapshots"][-1]["npu_link_outstanding_gb_by_npu"],
            )
        ],
        "measurement_actual_cir_sum_gbps_by_ssu_at_start": list(
            context.measurement_actual_cir_start["per_ssu_gbps"]
        ),
        "measurement_actual_cir_sum_gbps_by_ssu_at_end": list(
            context.measurement_actual_cir_end["per_ssu_gbps"]
        ),
        "measurement_actual_npu_cir_sum_gbps_at_start": (
            None
            if context.measurement_actual_cir_start["per_npu_gbps"] is None
            else list(context.measurement_actual_cir_start["per_npu_gbps"])
        ),
        "measurement_actual_npu_cir_sum_gbps_at_end": (
            None
            if context.measurement_actual_cir_end["per_npu_gbps"] is None
            else list(context.measurement_actual_cir_end["per_npu_gbps"])
        ),
        **measurement_control_metrics,
        "ttft_slo_attainment": float(np.mean(per_npu_slo)),
        "request_weighted_slo_attainment": float(
            np.mean([row["slo_met"] for row in request_rows])
        ),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p99_ttft_ms": _percentile(ttfts, 99),
        "measurement_request_count": len(request_rows),
        "request_counts_by_npu": request_counts_by_npu,
        "request_rows": request_rows,
        "measurement_blocks": block_rows,
        "completed_by_npu_at_stop": context.completed_by_npu,
        "all_input_requests_completed": context.completed_requests
        == len(context.requests),
        "input_fingerprint": _input_fingerprint(
            tuple(request.manifest for request in context.requests.values())
        ),
        "events_processed": events_processed,
        "pressure_reports": sum(
            disk.scheduler.pressure_reports for disk in context.disks
        ),
        "pressure_reports_by_ssu": [
            int(disk.scheduler.pressure_reports) for disk in context.disks
        ],
        "pressure_cache_hits": sum(
            int(getattr(disk.scheduler, "pressure_cache_hits", 0))
            for disk in context.disks
        ),
        "pressure_cache_hits_by_ssu": [
            int(getattr(disk.scheduler, "pressure_cache_hits", 0))
            for disk in context.disks
        ],
        "control_evaluations": context.control_evaluations,
        "control_min_interval_ms": (
            context.control.min_interval_ms if context.control is not None else None
        ),
        "cir_commits": context.cir_commits,
        "cir_write_transactions": sum(context.cir_write_transactions_by_ssu),
        "cir_write_transactions_by_ssu": list(context.cir_write_transactions_by_ssu),
        "cir_path_writes": context.cir_path_writes,
        "cir_path_writes_by_ssu": list(context.cir_path_writes_by_ssu),
        "actual_cir_sum_gbps_by_ssu_at_stop": list(actual_cir_at_stop["per_ssu_gbps"]),
        "max_actual_cir_sum_gbps_by_ssu": max_actual_ssu_cirs,
        "actual_npu_cir_sum_gbps_at_stop": (
            None
            if actual_cir_at_stop["per_npu_gbps"] is None
            else list(actual_cir_at_stop["per_npu_gbps"])
        ),
        "max_actual_npu_cir_sum_gbps_by_npu": max_actual_npu_cirs,
        "actual_cir_per_npu_capacity_applicable": (
            context.npu_dedicated_paths is not None
        ),
        "invariants": invariants,
    }


def _build_summary(context: _Context, requests, current_time_ms, events_processed):
    microbatch_metrics = []
    for batch in context.microbatches:
        processing_ms = batch.completion_time_ms - batch.admission_time_ms
        layer_metrics = [
            {
                "layer": metric.layer,
                "io_start_time_ms": metric.io_start_time_ms,
                "io_ready_time_ms": metric.io_ready_time_ms,
                "compute_start_ms": metric.compute_start_ms,
                "compute_end_ms": metric.compute_end_ms,
                "compute_duration_ms": metric.compute_duration_ms,
                "io_barrier_wait_ms": metric.io_barrier_wait_ms,
            }
            for metric in batch.layer_metrics
        ]
        microbatch_metrics.append(
            {
                "batch_id": batch.batch_id,
                "npu_id": batch.npu_id,
                "member_request_ids": list(batch.member_request_ids),
                "batch_size": len(batch.member_request_ids),
                "admission_time_ms": batch.admission_time_ms,
                "completion_time_ms": batch.completion_time_ms,
                "processing_latency_ms": processing_ms,
                "compute_busy_ms": batch.compute_busy_ms,
                "io_barrier_wait_ms": batch.io_barrier_wait_ms,
                "latency_accounting_error_ms": processing_ms
                - batch.compute_busy_ms
                - batch.io_barrier_wait_ms,
                "layer_metrics": layer_metrics,
            }
        )

    request_metrics = []
    for request in sorted(
        context.requests.values(), key=lambda item: item.manifest.request_id
    ):
        batch = context.microbatches[request.batch_id]
        own_compute_ms = context.n_layers * request.per_layer_compute_ms
        admission_wait_ms = request.admission_time_ms - request.manifest.arrival_time_ms
        processing_ms = request.completion_time_ms - request.admission_time_ms
        latency_ms = request.completion_time_ms - request.manifest.arrival_time_ms
        batch_compute_ms = batch.compute_busy_ms
        barrier_ms = batch.io_barrier_wait_ms
        request_metrics.append(
            {
                "request_id": request.manifest.request_id,
                "npu_id": request.manifest.npu_id,
                "category": request.category,
                "initial": bool(request.manifest.load.get("initial", True)),
                "batch_id": request.batch_id,
                "batch_size": request.batch_size,
                "arrival_time_ms": request.manifest.arrival_time_ms,
                "admission_time_ms": request.admission_time_ms,
                "completion_time_ms": request.completion_time_ms,
                "layer0_cross_request_prefetched": (
                    request.layer0_cross_request_prefetched
                ),
                "layer0_manifest_controlled": request.layer0_manifest_controlled,
                "layer0_io_start_time_ms": request.io_start_time_ms[0],
                "layer0_path_cirs_at_submit": [
                    {
                        "ssu_id": ssu_id,
                        "path_id": path_id,
                        "cir_gbps": cir_gbps,
                    }
                    for (ssu_id, path_id), cir_gbps in sorted(
                        request.layer0_path_cirs_at_submit.items()
                    )
                ],
                "admission_wait_ms": admission_wait_ms,
                "processing_latency_ms": processing_ms,
                "latency_ms": latency_ms,
                "own_compute_ms": own_compute_ms,
                "peer_compute_ms": batch_compute_ms - own_compute_ms,
                "batch_compute_ms": batch_compute_ms,
                "batch_io_barrier_wait_ms": barrier_ms,
                "io_stall_ms": barrier_ms,
                "compute_queue_wait_ms": 0.0,
                "latency_accounting_error_ms": processing_ms
                - batch_compute_ms
                - barrier_ms,
                "request_compute_fraction": own_compute_ms / processing_ms,
                "own_compute_contribution_fraction": own_compute_ms / processing_ms,
                "batch_compute_fraction": batch_compute_ms / processing_ms,
                "request_io_efficiency": batch_compute_ms
                / (batch_compute_ms + barrier_ms),
                "io_count": request.io_count,
                "avg_ssd_queue_wait_ms": request.ssd_queue_wait_ms / request.io_count,
                "avg_npu_link_queue_wait_ms": request.link_queue_wait_ms
                / request.io_count,
                "avg_end_to_end_io_latency_ms": request.io_latency_ms
                / request.io_count,
            }
        )

    total_compute_ms = sum(npu.compute_busy_ms for npu in context.npus)
    active_window_ms = sum(
        max(0.0, npu.last_completion_ms - npu.first_admission_ms)
        for npu in context.npus
        if np.isfinite(npu.first_admission_ms)
    )
    total_read_gb = sum(npu.link_completed_gb for npu in context.npus)
    disk_read_gb = sum(disk.completed_bytes_gb for disk in context.disks)
    disk_stats = []
    for disk in context.disks:
        scheduler = disk.scheduler
        scheduler.settle(current_time_ms)
        disk_stats.append(
            {
                "ssu_id": disk.disk_id,
                "utilization": disk.busy_time / current_time_ms
                if current_time_ms
                else 0.0,
                "completed_gb": disk.completed_bytes_gb,
                "max_backend_active_io": scheduler.max_backend_active_io,
                "max_outstanding_blocks": scheduler.max_outstanding_blocks,
                "pressure_reports": scheduler.pressure_reports,
                "pressure_cache_hits": int(
                    getattr(scheduler, "pressure_cache_hits", 0)
                ),
                "backend_dispatches": scheduler.backend_dispatches,
            }
        )

    actual_cir_at_stop = _record_actual_cir_state(context)
    max_actual_ssu_cirs = list(context.max_actual_cir_sum_gbps_by_ssu)
    max_actual_npu_cirs = (
        None
        if context.max_actual_npu_cir_sum_gbps_by_npu is None
        else list(context.max_actual_npu_cir_sum_gbps_by_npu)
    )
    invariants = {
        "all_requests_completed": context.completed_requests == len(requests),
        "microbatch_capacity": all(
            npu.max_batch_size <= context.batch_size for npu in context.npus
        ),
        "one_compute_job_per_npu": all(
            npu.compute_active is None for npu in context.npus
        ),
        "one_link_io_per_npu": all(
            npu.link_active_flow is None and not npu.link_pending
            for npu in context.npus
        ),
        "one_command_per_ssd": all(
            disk.scheduler.max_backend_active_io <= 1 for disk in context.disks
        ),
        "block_conservation": (
            context.submitted_blocks
            == context.completed_blocks
            == context.expected_blocks
            and all(state == 2 for state in context.block_states)
        ),
        "ssd_byte_conservation": math.isclose(
            disk_read_gb,
            context.expected_read_gb,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ),
        "npu_link_byte_conservation": math.isclose(
            total_read_gb,
            context.expected_read_gb,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ),
        "cir_capacity": context.max_cir_sum_gbps <= context.disk_bw_gbps + 1e-9,
        "actual_cir_per_ssu_capacity": all(
            value <= context.disk_bw_gbps + 1e-9 for value in max_actual_ssu_cirs
        ),
        "actual_cir_per_npu_capacity": (
            max_actual_npu_cirs is None
            or all(value <= context.npu_bw_gbps + 1e-9 for value in max_actual_npu_cirs)
        ),
        "cir_entry_write_counter_consistency": (
            context.cir_path_writes == sum(context.cir_path_writes_by_ssu)
        ),
        "causal_control_drained": (
            not context.pending_causal_prefetches
            and not context.pending_causal_request_prefetches
        ),
        "batch_latency_decomposition": all(
            abs(row["latency_accounting_error_ms"]) <= 1e-7
            for row in microbatch_metrics
        ),
        "request_latency_decomposition": all(
            abs(row["latency_accounting_error_ms"]) <= 1e-7 for row in request_metrics
        ),
        "compute_work_conservation": math.isclose(
            total_compute_ms,
            sum(
                context.n_layers * request.per_layer_compute_ms
                for request in context.requests.values()
            ),
            rel_tol=1e-10,
            abs_tol=1e-7,
        ),
        "microbatch_membership": sorted(
            request_id
            for batch in context.microbatches
            for request_id in batch.member_request_ids
        )
        == sorted(context.requests),
        "fixed_batch_layer_barrier": all(
            metric.compute_start_ms + _EPS >= metric.io_ready_time_ms
            and metric.compute_start_ms + _EPS
            >= (
                batch.admission_time_ms
                if metric.layer == 0
                else batch.layer_metrics[metric.layer - 1].compute_end_ms
            )
            for batch in context.microbatches
            for metric in batch.layer_metrics
        ),
        "next_layer_prefetch_at_compute_start": all(
            math.isclose(
                batch.layer_metrics[layer].io_start_time_ms,
                batch.layer_metrics[layer - 1].compute_start_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for batch in context.microbatches
            for layer in range(1, context.n_layers)
        ),
        "all_microbatches_released": all(
            npu.active_batch is None and not npu.admission_queue for npu in context.npus
        ),
    }
    if not all(invariants.values()):
        raise AssertionError(
            f"continuous-batch invariant failed: {invariants}; "
            "actual CIR diagnostics: "
            f"per_ssu_at_stop={actual_cir_at_stop['per_ssu_gbps']}, "
            f"max_per_ssu={max_actual_ssu_cirs}, "
            f"per_npu_at_stop={actual_cir_at_stop['per_npu_gbps']}, "
            f"max_per_npu={max_actual_npu_cirs}"
        )

    latencies = [row["latency_ms"] for row in request_metrics]
    processing = [row["processing_latency_ms"] for row in request_metrics]
    request_compute = [row["request_compute_fraction"] for row in request_metrics]
    batch_compute = [row["batch_compute_fraction"] for row in request_metrics]
    io_efficiency = [row["request_io_efficiency"] for row in request_metrics]
    membership_digest = hashlib.sha256(b"full-prefill-microbatch-members-v1\0")
    for batch in context.microbatches:
        membership_digest.update(
            repr((batch.batch_id, batch.npu_id, batch.member_request_ids)).encode()
        )
    return {
        "schema_version": 3,
        "execution_model": EXECUTION_MODEL,
        "batch_compute_model": BATCH_COMPUTE_MODEL,
        "batch_compute_calibration": "singleton layer times added; no batch speedup",
        "partial_batch_policy": PARTIAL_BATCH_POLICY,
        "prefetch_policy": PREFETCH_POLICY,
        "layer0_path_id": context.layer0_path_id,
        "cross_request_layer0_prefetch": context.cross_request_layer0_prefetch,
        "cross_request_layer0_prefetches": (context.cross_request_layer0_prefetches),
        "manifest_layer0_prefetches": context.manifest_layer0_prefetches,
        "routing_mode": (
            "layer0_path0_then_causal_npu_dedicated"
            if context.layer0_path_id is not None
            else "unchanged"
        ),
        "workload_size_field_unit": "GiB despite legacy _gb field names",
        "bandwidth_unit_assumption": "configured numeric rates applied directly; no GiB-to-GB conversion",
        "policy": context.policy,
        "num_npu": context.num_npu,
        "num_ssu": context.num_ssu,
        "n_layers": context.n_layers,
        "batch_size": context.batch_size,
        "pressure_ttl_ms": context.pressure_ttl_ms,
        "cir_write_threshold_gbps": context.cir_write_threshold_gbps,
        "client_submit_batch_size": context.client_io_config.submit_batch_size,
        "client_issue_interval_us": context.client_io_config.issue_interval_us,
        "oracle_priority": (
            context.disks[0].scheduler.oracle_priority.__name__
            if context.policy == sim.POLICY_PER_SSD_FULL_VISIBLE_EDF
            else None
        ),
        "causal_layer_observations": context.causal_observations,
        "request_count": len(requests),
        "input_fingerprint": _input_fingerprint(requests),
        "microbatch_membership_fingerprint": membership_digest.hexdigest(),
        "microbatch_count": len(context.microbatches),
        "full_microbatch_count": sum(
            len(batch.member_request_ids) == context.batch_size
            for batch in context.microbatches
        ),
        "partial_microbatch_count": sum(
            len(batch.member_request_ids) < context.batch_size
            for batch in context.microbatches
        ),
        "makespan_ms": current_time_ms,
        "throughput_requests_per_s": len(requests) / current_time_ms * 1000.0,
        "fleet_npu_compute_utilization": total_compute_ms
        / (context.num_npu * current_time_ms),
        "active_window_npu_compute_utilization": total_compute_ms / active_window_ms,
        "avg_request_compute_fraction": float(np.mean(request_compute)),
        "avg_own_compute_contribution_fraction": float(np.mean(request_compute)),
        "avg_batch_compute_fraction": float(np.mean(batch_compute)),
        "avg_request_io_efficiency": float(np.mean(io_efficiency)),
        "avg_request_latency_ms": float(np.mean(latencies)),
        "p50_request_latency_ms": _percentile(latencies, 50),
        "p95_request_latency_ms": _percentile(latencies, 95),
        "p99_request_latency_ms": _percentile(latencies, 99),
        "avg_processing_latency_ms": float(np.mean(processing)),
        "avg_admission_wait_ms": float(
            np.mean([row["admission_wait_ms"] for row in request_metrics])
        ),
        "avg_io_stall_ms": float(
            np.mean([row["batch_io_barrier_wait_ms"] for row in request_metrics])
        ),
        "avg_batch_io_barrier_wait_ms": float(
            np.mean([row["batch_io_barrier_wait_ms"] for row in request_metrics])
        ),
        "avg_compute_queue_wait_ms": 0.0,
        "ssd_mean_utilization": float(
            np.mean([row["utilization"] for row in disk_stats])
        ),
        "npu_link_mean_utilization": sum(npu.link_busy_ms for npu in context.npus)
        / (context.num_npu * current_time_ms),
        "pressure_reports": sum(
            disk.scheduler.pressure_reports for disk in context.disks
        ),
        "pressure_reports_by_ssu": [
            int(disk.scheduler.pressure_reports) for disk in context.disks
        ],
        "pressure_cache_hits": sum(
            int(getattr(disk.scheduler, "pressure_cache_hits", 0))
            for disk in context.disks
        ),
        "pressure_cache_hits_by_ssu": [
            int(getattr(disk.scheduler, "pressure_cache_hits", 0))
            for disk in context.disks
        ],
        "control_evaluations": context.control_evaluations,
        "control_trigger": (
            "none"
            if context.control is None
            else "batch_boundary"
            if context.control.every_layers is None
            and context.control.interval_ms is None
            else "batch_boundary_and_fleet_layer_equivalent"
            if context.control.every_layers is not None
            else "batch_boundary_and_wall_clock"
        ),
        "control_every_layers": (
            context.control.every_layers if context.control is not None else None
        ),
        "control_interval_ms": (
            context.control.interval_ms if context.control is not None else None
        ),
        "control_min_interval_ms": (
            context.control.min_interval_ms if context.control is not None else None
        ),
        "cir_commits": context.cir_commits,
        "cir_write_transactions": sum(context.cir_write_transactions_by_ssu),
        "cir_write_transactions_by_ssu": list(context.cir_write_transactions_by_ssu),
        "cir_path_writes": context.cir_path_writes,
        "cir_path_writes_by_ssu": list(context.cir_path_writes_by_ssu),
        "actual_cir_sum_gbps_by_ssu_at_stop": list(actual_cir_at_stop["per_ssu_gbps"]),
        "max_actual_cir_sum_gbps_by_ssu": max_actual_ssu_cirs,
        "actual_npu_cir_sum_gbps_at_stop": (
            None
            if actual_cir_at_stop["per_npu_gbps"] is None
            else list(actual_cir_at_stop["per_npu_gbps"])
        ),
        "max_actual_npu_cir_sum_gbps_by_npu": max_actual_npu_cirs,
        "actual_cir_per_npu_capacity_applicable": (
            context.npu_dedicated_paths is not None
        ),
        "completed_batch_layers": context.completed_batch_layers,
        "completed_request_layer_jobs": context.completed_request_layer_jobs,
        "completed_layer_jobs": context.completed_request_layer_jobs,
        "submitted_blocks": context.submitted_blocks,
        "completed_blocks": context.completed_blocks,
        "expected_read_gb": context.expected_read_gb,
        "completed_read_gb": total_read_gb,
        "events_processed": events_processed,
        "stale_events": context.stale_events,
        "event_counts": {
            str(key): value for key, value in context.event_counts.items()
        },
        "disk_stats": disk_stats,
        "microbatch_metrics": microbatch_metrics,
        "request_metrics": request_metrics,
        "invariants": invariants,
    }


def simulate_continuous_batch(
    requests: Sequence[ContinuousBatchRequest],
    *,
    num_npu: int,
    num_ssu: int,
    n_layers: int = 16,
    batch_size: int = 8,
    policy: str = sim.POLICY_QOS_STATIC_CIR,
    qos_config: Optional[sim.StaticQoSConfig] = None,
    qos_configs_by_ssu: Optional[Sequence[sim.StaticQoSConfig]] = None,
    npu_dedicated_paths: Optional[Sequence[int]] = None,
    layer0_path_id: Optional[int] = None,
    cross_request_layer0_prefetch: bool = False,
    client_io_config: sim.ClientIOConfig = sim.DEFAULT_CLIENT_IO_CONFIG,
    control: Optional[CIRControlConfig] = None,
    causal_control: Optional[CausalLayerControlConfig] = None,
    steady_state: Optional[SteadyStateConfig] = None,
    disk_bw_gbps: float = sim.DISK_BW,
    npu_bw_gbps: float = sim.NPU_BW_LIMIT,
    pressure_ttl_ms: float = 0.0,
    cir_write_threshold_gbps: float = 0.0,
    submit_order_seed: int = 42,
    oracle_priority_key=None,
):
    """Run a finite Full-prefill microbatch trace and return JSON-safe metrics.

    Static routing strategies pass the same QoS configuration and different
    ``ClientIOConfig`` values.  Dynamic Scheme B additionally passes one
    dedicated Path per NPU and a periodic ``CIRControlConfig``.  With batch
    size one, ``cross_request_layer0_prefetch`` starts an already-arrived next
    request's Layer-0 I/O at the current request's final-layer compute start;
    admission still waits for current-request completion.  The physical oracle
    uses ``POLICY_PER_SSD_FULL_VISIBLE_EDF`` and no QoS configuration; an
    explicit released-I/O priority callback selects its retained candidate.
    """
    requests = tuple(requests)
    if not requests:
        raise ValueError("continuous-batch trace must contain requests")
    if num_npu <= 0 or num_ssu <= 0 or n_layers <= 0 or batch_size <= 0:
        raise ValueError("simulation dimensions must be positive")
    pressure_ttl_ms = float(pressure_ttl_ms)
    cir_write_threshold_gbps = float(cir_write_threshold_gbps)
    if (
        not math.isfinite(pressure_ttl_ms)
        or pressure_ttl_ms < 0.0
        or not math.isfinite(cir_write_threshold_gbps)
        or cir_write_threshold_gbps < 0.0
    ):
        raise ValueError(
            "pressure TTL and CIR write threshold must be finite and nonnegative"
        )
    if cross_request_layer0_prefetch and batch_size != 1:
        raise ValueError("cross-request Layer-0 prefetch requires batch_size=1")
    if steady_state is not None and (
        batch_size != 1
        or steady_state.warmup_requests_per_npu <= 0
        or steady_state.settle_ms < 0.0
        or steady_state.measurement_ms <= 0.0
        or steady_state.slo_alpha <= 0.0
        or steady_state.block_ms <= 0.0
    ):
        raise ValueError("steady-state measurement configuration is invalid")
    if policy not in sim.SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    if any(
        request.npu_id < 0
        or request.npu_id >= num_npu
        or request.arrival_time_ms < 0.0
        or len(request.placement) not in (1, n_layers)
        for request in requests
    ):
        raise ValueError("request NPU, arrival, or layer count is invalid")
    if any(
        ssu_id < 0 or ssu_id >= num_ssu or size_gb <= 0.0
        for request in requests
        for layer in request.placement
        for ssu_id, size_gb in layer
    ):
        raise ValueError("request placement contains an invalid block")

    disk_qos_configs: tuple[sim.StaticQoSConfig, ...] = ()
    dedicated_paths = None
    if policy == sim.POLICY_QOS_STATIC_CIR:
        if qos_configs_by_ssu is None:
            if qos_config is None:
                raise ValueError("QoS policy requires a Path configuration")
            disk_qos_configs = (qos_config,) * num_ssu
        else:
            disk_qos_configs = tuple(qos_configs_by_ssu)
            if len(disk_qos_configs) != num_ssu:
                raise ValueError("qos_configs_by_ssu must cover every SSU")
        if npu_dedicated_paths is not None:
            dedicated_paths = tuple(int(path) for path in npu_dedicated_paths)
            if (
                len(dedicated_paths) != num_npu
                or len(set(dedicated_paths)) != num_npu
                or any(
                    path < 0 or path >= qos.path_count
                    for path in dedicated_paths
                    for qos in disk_qos_configs
                )
            ):
                raise ValueError("dedicated Path mapping is invalid")
        elif any(
            qos.category_labels != sim.QOS_ROUTING_CATEGORIES
            for qos in disk_qos_configs
        ):
            raise ValueError("static routing requires SS/SL/LS/LL Path classes")
    elif qos_config is not None or qos_configs_by_ssu is not None:
        raise ValueError("the physical EDF policy does not use QoS tables")
    if oracle_priority_key is not None and (
        policy != sim.POLICY_PER_SSD_FULL_VISIBLE_EDF
    ):
        raise ValueError("oracle priority requires the physical full-info policy")

    if control is not None and (
        policy != sim.POLICY_QOS_STATIC_CIR or dedicated_paths is None
    ):
        raise ValueError("runtime CIR control requires NPU-dedicated QoS Paths")
    if causal_control is not None and (
        control is not None
        or policy != sim.POLICY_QOS_STATIC_CIR
        or dedicated_paths is None
        or layer0_path_id is None
    ):
        raise ValueError("causal control requires Layer-0 and dedicated QoS Paths")
    if layer0_path_id is not None and (
        policy != sim.POLICY_QOS_STATIC_CIR
        or dedicated_paths is None
        or layer0_path_id in dedicated_paths
        or any(
            layer0_path_id < 0 or layer0_path_id >= qos.path_count
            for qos in disk_qos_configs
        )
    ):
        raise ValueError("Layer-0 Path must be distinct from dedicated Paths")

    context = _Context(
        requests=requests,
        num_npu=num_npu,
        num_ssu=num_ssu,
        n_layers=n_layers,
        batch_size=batch_size,
        policy=policy,
        qos_configs_by_ssu=disk_qos_configs,
        npu_dedicated_paths=dedicated_paths,
        layer0_path_id=(int(layer0_path_id) if layer0_path_id is not None else None),
        cross_request_layer0_prefetch=bool(cross_request_layer0_prefetch),
        client_io_config=client_io_config,
        control=control,
        causal_control=causal_control,
        steady_state=steady_state,
        disk_bw_gbps=float(disk_bw_gbps),
        npu_bw_gbps=float(npu_bw_gbps),
        pressure_ttl_ms=pressure_ttl_ms,
        cir_write_threshold_gbps=cir_write_threshold_gbps,
        submit_order_seed=int(submit_order_seed),
        oracle_priority_key=oracle_priority_key,
    )
    if disk_qos_configs:
        context.max_cir_sum_gbps = max(sum(qos.path_cirs) for qos in disk_qos_configs)
    for request in sorted(
        requests,
        key=lambda item: (item.arrival_time_ms, item.npu_id, item.request_id),
    ):
        context.push_event(
            request.arrival_time_ms,
            REQUEST_ARRIVAL,
            request.npu_id,
            payload=request.request_id,
        )
    if control is not None:
        _queue_control_event(
            context,
            min(r.arrival_time_ms for r in requests),
            "wall_clock" if control.interval_ms is not None else "initial",
        )

    current_time_ms = 0.0
    events_processed = 0
    while context.completed_requests < len(requests):
        if not context.event_heap:
            raise RuntimeError("event queue drained before all requests completed")
        event_time, event_type, resource_id, value, generation = heapq.heappop(
            context.event_heap
        )
        current_time_ms = max(current_time_ms, event_time)
        context.current_time_ms = current_time_ms
        # Measurement sampling is instrumentation, not a simulated workload
        # event.  Excluding it preserves legacy event-count comparability.
        if event_type != STEADY_STATIONARITY_SAMPLE:
            context.event_counts[event_type] += 1
            events_processed += 1
        if event_type == STEADY_MEASUREMENT_START:
            _handle_steady_measurement_start(context, current_time_ms)
        elif event_type == STEADY_STATIONARITY_SAMPLE:
            _handle_steady_stationarity_sample(context, resource_id, current_time_ms)
        elif event_type == STEADY_MEASUREMENT_END:
            _handle_steady_measurement_end(context, current_time_ms)
        elif event_type == REQUEST_ARRIVAL:
            _handle_arrival(context, context.pop_payload(value), current_time_ms)
        elif event_type == BATCH_DISPATCH:
            _handle_batch_dispatch(context, resource_id, current_time_ms)
        elif event_type == sim.COMPUTE_DONE:
            _handle_compute_done(
                context,
                resource_id,
                context.pop_payload(value),
                generation,
                current_time_ms,
            )
        elif event_type == sim.DISK_COMPLETION:
            _handle_disk_completion(context, resource_id, generation, current_time_ms)
            next_is_disk_completion = (
                bool(context.event_heap)
                and abs(context.event_heap[0][0] - current_time_ms) <= _EPS
                and context.event_heap[0][1] == sim.DISK_COMPLETION
            )
            if not next_is_disk_completion:
                _flush_link_starts(context, current_time_ms)
        elif event_type == sim.NPU_LINK_COMPLETION:
            _handle_link_completion(context, resource_id, generation, current_time_ms)
        elif event_type == CIR_CONTROL:
            _handle_control(context, current_time_ms)
        elif event_type == COMPUTE_SCHEDULE:
            _handle_compute_schedule(context, resource_id, current_time_ms)
        elif event_type == CAUSAL_LAYER_CONTROL:
            _handle_causal_layer_control(context, current_time_ms)
        elif event_type == sim.CLIENT_SUBMISSION:
            _handle_client_submission(context, generation, current_time_ms)
        elif event_type == sim.DISK_SCHEDULE:
            scheduler = context.disks[resource_id].scheduler
            if generation != scheduler.dispatch_generation:
                context.stale_events += 1
            else:
                scheduler.dispatch(current_time_ms, context.event_heap)
        else:
            raise RuntimeError(f"unknown event type: {event_type}")

        if (
            steady_state is not None
            and context.measurement_ended
            and context.measurement_completed_ids == context.measurement_request_ids
        ):
            break

    if steady_state is not None:
        if (
            context.measurement_start_ms is None
            or context.measurement_end_ms is None
            or not context.measurement_ended
        ):
            raise SteadyStateInvariantError(
                {"input_prefix_covers_measurement_window": False},
                {
                    "current_time_ms": current_time_ms,
                    "measurement_start_ms": context.measurement_start_ms,
                    "measurement_end_ms": context.measurement_end_ms,
                    "completed_by_npu_at_stop": list(context.completed_by_npu),
                },
            )
        return _build_steady_state_summary(context, current_time_ms, events_processed)
    return _build_summary(context, requests, current_time_ms, events_processed)
