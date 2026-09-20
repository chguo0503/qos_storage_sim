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
    duration_ms = 16000.0
    two_ulp_above = math.nextafter(math.nextafter(duration_ms, math.inf), math.inf)
    historical_duration_ms = 8000.0
    historical_two_ulp_above = math.nextafter(
        math.nextafter(historical_duration_ms, math.inf), math.inf
    )
    block_duration_ms = 500.0
    block_two_ulp_above = math.nextafter(
        math.nextafter(block_duration_ms, math.inf), math.inf
    )
    checks = {
        "two_ulp_residual_accepted": _steady_resource_overlap_within_bounds(
            two_ulp_above, duration_ms
        ),
        "historical_two_ulp_residual_accepted": (
            _steady_resource_overlap_within_bounds(
                historical_two_ulp_above, historical_duration_ms
            )
        ),
        "stationarity_block_two_ulp_residual_accepted": (
            _steady_resource_overlap_within_bounds(
                block_two_ulp_above, block_duration_ms
            )
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
        "historical_duration_ms": historical_duration_ms,
        "historical_two_ulp_residual_ms": (
            historical_two_ulp_above - historical_duration_ms
        ),
        "stationarity_block_duration_ms": block_duration_ms,
        "stationarity_block_two_ulp_residual_ms": (
            block_two_ulp_above - block_duration_ms
        ),
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
    # ``arrival_time_ms`` is an observed input timestamp.  The other two
    # fields are deliberately absent for a cross-request Layer-0 prefetch:
    # admission has not happened yet, so assigning it an admission time or an
    # absolute deadline would expose a fictional future fact.  The deadline is
    # specifically for this simulator's admission-to-completion TTFT; callers
    # must use arrival_time_ms separately for an arrival-to-completion SLO.
    admission_time_ms: Optional[float] = None
    hard_deadline_time_ms: Optional[float] = None
    arrival_time_ms: Optional[float] = None


@dataclass(frozen=True)
class CIRControlSnapshot:
    time_ms: float
    evaluation: int
    layer_jobs_since_previous: int
    num_npu: int
    num_ssu: int
    active_requests: tuple[ControlRequestView, ...]
    current_path_cirs_by_ssu: tuple[tuple[float, ...], ...]
    trigger_reasons: tuple[str, ...] = ()
    # One immutable row per SSU and one count per hardware QoS Path.  The
    # scheduler's public pressure API may return a TTL-cached row; the three
    # cumulative counters make that observation cost explicit.
    path_outstanding_io_counts_by_ssu: tuple[tuple[int, ...], ...] = ()
    pressure_queries_cumulative_by_ssu: tuple[int, ...] = ()
    pressure_reads_cumulative_by_ssu: tuple[int, ...] = ()
    pressure_cache_hits_cumulative_by_ssu: tuple[int, ...] = ()


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

    When explicitly set, ``hard_ttft_ideal_multiplier`` configures the absolute
    deadline exported for admitted requests as
    ``admission + multiplier * n_layers * C``.  It defaults to ``None`` so an
    old controller is not silently assigned a new SLO.  The resulting deadline
    is an admission-to-completion budget, matching this simulator's current
    TTFT metric, and is not an arrival-to-completion production latency promise.
    """

    every_layers: Optional[int]
    callback: CIRControlCallback
    interval_ms: Optional[float]
    on_batch_boundary: bool
    min_interval_ms: float
    hard_ttft_ideal_multiplier: Optional[float]

    def __init__(
        self,
        every_layers=None,
        callback=None,
        *,
        interval_ms=None,
        on_batch_boundary=True,
        min_interval_ms=0.0,
        hard_ttft_ideal_multiplier=None,
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
        if hard_ttft_ideal_multiplier is not None:
            hard_ttft_ideal_multiplier = float(hard_ttft_ideal_multiplier)
            if (
                not math.isfinite(hard_ttft_ideal_multiplier)
                or hard_ttft_ideal_multiplier <= 0.0
            ):
                raise ValueError(
                    "hard_ttft_ideal_multiplier must be positive and finite"
                )
        object.__setattr__(
            self, "every_layers", int(every_layers) if layer_mode else None
        )
        object.__setattr__(self, "callback", callback)
        object.__setattr__(
            self, "interval_ms", float(interval_ms) if interval_mode else None
        )
        object.__setattr__(self, "on_batch_boundary", bool(on_batch_boundary))
        object.__setattr__(self, "min_interval_ms", float(min_interval_ms))
        object.__setattr__(
            self,
            "hard_ttft_ideal_multiplier",
            hard_ttft_ideal_multiplier,
        )


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
    timeline_diagnostics: bool = False
    timeline_dispatch_probe_ms: float = 50.0
    timeline_dispatch_probe_limit: int = 10_000
    # Optional, read-only profile-cycle trend probe.  ``generation`` is
    # supplied by each request's load metadata; it is deliberately not
    # inferred from a request-ID layout because the simulator is otherwise
    # agnostic to how callers allocate IDs.  A zero period plus an empty NPU
    # tuple disables the probe and preserves the legacy output byte-for-byte.
    profile_cycle_probe_npu_ids: tuple[int, ...] = ()
    profile_cycle_period: int = 0


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
        # Controller pressure telemetry is kept separate from client-routing
        # pressure reads.  A query can be served by the scheduler's TTL cache,
        # so queries, fresh reads, and cache hits are all counted explicitly.
        self.cir_control_pressure_queries_by_ssu = [0] * num_ssu
        self.cir_control_pressure_reads_by_ssu = [0] * num_ssu
        self.cir_control_pressure_cache_hits_by_ssu = [0] * num_ssu
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
        self.measurement_fragmented_npu_ssu_served_start_gb: Optional[
            tuple[tuple[float, ...], ...]
        ] = None
        self.measurement_fragmented_npu_ssu_served_end_gb: Optional[
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

        # The profile-cycle frontier observer is independent of the dense
        # timeline diagnostics.  It records only at request-completion
        # frontiers, performs no scheduler calls, consumes no RNG state, and
        # inserts no event.  Input validation has already established that all
        # probe NPUs carry the same consecutive generation sequence.
        self.profile_cycle_probe_npu_ids = (
            tuple(steady_state.profile_cycle_probe_npu_ids)
            if steady_state is not None
            else ()
        )
        self.profile_cycle_period = (
            int(steady_state.profile_cycle_period)
            if steady_state is not None
            else 0
        )
        self.profile_cycle_probe_enabled = bool(
            self.profile_cycle_probe_npu_ids
        )
        self.profile_generation_by_request_id = (
            {
                request.request_id: int(request.load["generation"])
                for request in requests
                if "generation" in request.load
            }
            if self.profile_cycle_probe_enabled
            else {}
        )
        self.profile_request_id_by_npu_generation = (
            {
                (request.npu_id, int(request.load["generation"])): request.request_id
                for request in requests
                if request.npu_id in self.profile_cycle_probe_npu_ids
            }
            if self.profile_cycle_probe_enabled
            else {}
        )
        self.profile_cycle_last_completed_generation_by_npu: list[Optional[int]] = [
            None
        ] * num_npu
        self.profile_cycle_next_frontier_generation: Optional[int] = None
        self.profile_cycle_frontier_snapshots: list[dict] = []
        self.profile_cycle_pending_frontiers: list[dict] = []
        self.profile_cycle_ssd_submitted_gb = (
            [[0.0] * num_ssu for _ in range(num_npu)]
            if self.profile_cycle_probe_enabled
            else []
        )
        self.profile_cycle_ssd_submitted_compensation_gb = (
            [[0.0] * num_ssu for _ in range(num_npu)]
            if self.profile_cycle_probe_enabled
            else []
        )

        # The timeline observer is deliberately fixed-size and disabled by
        # default.  It never calls a scheduler API, consumes RNG state, or
        # inserts an event.  Boundary snapshots below only read these counters,
        # so enabling it cannot change the simulated decision order.
        self.timeline_diagnostics = bool(
            steady_state is not None and steady_state.timeline_diagnostics
        )
        self.timeline_ssd_enqueued_gb = [
            [0.0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_ssd_enqueued_compensation_gb = [
            [0.0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_ssd_enqueued_blocks = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_ssd_completed_blocks = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_link_enqueued_gb = [
            [0.0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_link_enqueued_blocks = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_link_completed_blocks = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_activated_compute_ms = [0.0] * num_npu
        self.timeline_activated_io_gb = [
            [0.0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_route_plans = [[0] * num_ssu for _ in range(num_npu)]
        self.timeline_route_pressure_fresh = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        self.timeline_route_pressure_cache = [
            [0] * num_ssu for _ in range(num_npu)
        ]
        timeline_group_count = (
            len(qos_configs_by_ssu[0].group_weights) if qos_configs_by_ssu else 0
        )
        self.timeline_route_blocks_by_group = [
            [[0] * timeline_group_count for _ in range(num_ssu)]
            for _ in range(num_npu)
        ]
        self.timeline_route_probe_records: list[dict] = []
        self.timeline_control_triggers: list[dict] = []
        self.timeline_dispatch_probe_records: list[dict] = []

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


def _cir_control_pressure_metrics(context: _Context):
    """Return controller-only telemetry lookup costs for result provenance."""
    queries_by_ssu = tuple(map(int, context.cir_control_pressure_queries_by_ssu))
    reads_by_ssu = tuple(map(int, context.cir_control_pressure_reads_by_ssu))
    cache_hits_by_ssu = tuple(
        map(int, context.cir_control_pressure_cache_hits_by_ssu)
    )
    if any(
        queries != reads + cache_hits
        for queries, reads, cache_hits in zip(
            queries_by_ssu, reads_by_ssu, cache_hits_by_ssu
        )
    ):
        raise AssertionError("CIR-control pressure counters diverged")
    return {
        "cir_control_pressure_queries": sum(queries_by_ssu),
        "cir_control_pressure_queries_by_ssu": list(queries_by_ssu),
        "cir_control_pressure_reads": sum(reads_by_ssu),
        "cir_control_pressure_reads_by_ssu": list(reads_by_ssu),
        "cir_control_pressure_cache_hits": sum(cache_hits_by_ssu),
        "cir_control_pressure_cache_hits_by_ssu": list(cache_hits_by_ssu),
    }


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
        "cir_control_pressure_queries_by_ssu": tuple(
            map(int, context.cir_control_pressure_queries_by_ssu)
        ),
        "cir_control_pressure_reads_by_ssu": tuple(
            map(int, context.cir_control_pressure_reads_by_ssu)
        ),
        "cir_control_pressure_cache_hits_by_ssu": tuple(
            map(int, context.cir_control_pressure_cache_hits_by_ssu)
        ),
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
    control_pressure_queries_by_ssu = _counter_vector_delta(
        end["cir_control_pressure_queries_by_ssu"],
        start["cir_control_pressure_queries_by_ssu"],
        "cir_control_pressure_queries_by_ssu",
    )
    control_pressure_reads_by_ssu = _counter_vector_delta(
        end["cir_control_pressure_reads_by_ssu"],
        start["cir_control_pressure_reads_by_ssu"],
        "cir_control_pressure_reads_by_ssu",
    )
    control_pressure_cache_hits_by_ssu = _counter_vector_delta(
        end["cir_control_pressure_cache_hits_by_ssu"],
        start["cir_control_pressure_cache_hits_by_ssu"],
        "cir_control_pressure_cache_hits_by_ssu",
    )
    control_pressure_queries = sum(control_pressure_queries_by_ssu)
    control_pressure_reads = sum(control_pressure_reads_by_ssu)
    control_pressure_cache_hits = sum(control_pressure_cache_hits_by_ssu)
    if control_pressure_queries != (
        control_pressure_reads + control_pressure_cache_hits
    ):
        raise AssertionError("CIR-control pressure query accounting diverged")
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
        "measurement_cir_control_pressure_queries": control_pressure_queries,
        "measurement_cir_control_pressure_queries_by_ssu": list(
            control_pressure_queries_by_ssu
        ),
        "measurement_cir_control_pressure_reads": control_pressure_reads,
        "measurement_cir_control_pressure_reads_by_ssu": list(
            control_pressure_reads_by_ssu
        ),
        "measurement_cir_control_pressure_cache_hits": (
            control_pressure_cache_hits
        ),
        "measurement_cir_control_pressure_cache_hits_by_ssu": list(
            control_pressure_cache_hits_by_ssu
        ),
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


def _project_ssd_active_served_gb(flow, current_time_ms: float):
    """Project active service from the command's immutable activation edge."""
    if flow is None:
        return 0.0
    activation_time_ms = float(flow.ssd_activation_time)
    if current_time_ms >= flow.end_time:
        return float(flow.total_gb)
    elapsed_ms = max(0.0, float(current_time_ms) - activation_time_ms)
    served_gb = max(0.0, float(flow.bw)) * elapsed_ms / 1000.0
    return min(float(flow.total_gb), max(0.0, served_gb))


def _project_ssd_active_remaining_gb(flow, current_time_ms: float):
    """Project one active SSD command without mutable settle fragmentation."""
    if flow is None:
        return 0.0
    return max(
        0.0,
        float(flow.total_gb)
        - _project_ssd_active_served_gb(flow, current_time_ms),
    )


def _project_ssd_service_by_npu(context: _Context, current_time_ms: float):
    """Stable cumulative SSD service for every NPU x SSU cell.

    Completed bytes are added once per non-preemptive command with compensated
    summation.  At a boundary at most one active command exists per SSD, and
    its serviced prefix is reconstructed from its immutable activation edge.
    The result is therefore invariant to the number of intervening ``settle``
    calls.
    """
    values = [
        [
            float(disk.completed_gb_by_npu.get(npu_id, 0.0))
            for disk in context.disks
        ]
        for npu_id in range(context.num_npu)
    ]
    for ssu_id, disk in enumerate(context.disks):
        if not disk.active_flows:
            continue
        flow = disk.active_flows[0]
        values[flow.npu_id][ssu_id] = math.fsum(
            (
                values[flow.npu_id][ssu_id],
                _project_ssd_active_served_gb(flow, current_time_ms),
            )
        )
    return values


def _project_fragmented_ssd_service_by_npu(
    context: _Context,
    current_time_ms: float,
):
    """Retain the old settle-fragmented projection for residual diagnostics."""
    values = [
        [
            float(disk.served_gb_by_npu.get(npu_id, 0.0))
            for disk in context.disks
        ]
        for npu_id in range(context.num_npu)
    ]
    for ssu_id, disk in enumerate(context.disks):
        if not disk.active_flows:
            continue
        flow = disk.active_flows[0]
        service_end_ms = min(current_time_ms, flow.end_time)
        duration_ms = max(0.0, service_end_ms - disk.last_event_time)
        values[flow.npu_id][ssu_id] += flow.bw * duration_ms / 1000.0
    return values


def _project_physical_ssd_outstanding_by_npu(
    context: _Context,
    current_time_ms: float,
):
    """Enumerate the physical pending/active SSD queues by NPU and SSU."""
    terms = [
        [[] for _ in range(context.num_ssu)]
        for _ in range(context.num_npu)
    ]
    blocks = [
        [0] * context.num_ssu for _ in range(context.num_npu)
    ]
    for ssu_id, disk in enumerate(context.disks):
        scheduler = disk.scheduler
        if scheduler.policy == sim.POLICY_QOS_STATIC_CIR:
            pending_flows = (
                flow
                for path in scheduler.paths.values()
                for flow in path.pending
            )
        else:
            pending_flows = (flow for _, flow in scheduler.oracle_heap)
        for flow in pending_flows:
            terms[flow.npu_id][ssu_id].append(float(flow.total_gb))
            blocks[flow.npu_id][ssu_id] += int(flow.block_count)
        if disk.active_flows:
            flow = disk.active_flows[0]
            terms[flow.npu_id][ssu_id].append(
                _project_ssd_active_remaining_gb(flow, current_time_ms)
            )
            blocks[flow.npu_id][ssu_id] += int(flow.block_count)
    outstanding_gb = [
        [math.fsum(cell_terms) for cell_terms in row]
        for row in terms
    ]
    return outstanding_gb, blocks


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


def _project_physical_link_outstanding_by_npu_ssu(
    context: _Context,
    current_time_ms: float,
):
    """Enumerate queued/active NPU-link bytes without observer counters."""
    terms = [
        [[] for _ in range(context.num_ssu)]
        for _ in range(context.num_npu)
    ]
    blocks = [
        [0] * context.num_ssu for _ in range(context.num_npu)
    ]
    for npu in context.npus:
        for flow in npu.link_pending:
            terms[npu.npu_id][flow.disk_id].append(float(flow.total_gb))
            blocks[npu.npu_id][flow.disk_id] += int(flow.block_count)
        flow = npu.link_active_flow
        if flow is not None:
            terms[npu.npu_id][flow.disk_id].append(
                _project_link_active_remaining_gb(context, flow, current_time_ms)
            )
            blocks[npu.npu_id][flow.disk_id] += int(flow.block_count)
    return (
        [[math.fsum(cell) for cell in row] for row in terms],
        blocks,
    )


def _project_client_unissued_by_npu_ssu(context: _Context):
    """Enumerate bytes materialized for issue but not submitted to an SSD."""
    terms = [
        [[] for _ in range(context.num_ssu)]
        for _ in range(context.num_npu)
    ]
    blocks = [
        [0] * context.num_ssu for _ in range(context.num_npu)
    ]
    for state in context.submission_states.values():
        remaining = state.blocks[state.cursor :]
        terms[state.npu_id][state.disk_id].extend(
            float(size_gb) for _, size_gb in remaining
        )
        blocks[state.npu_id][state.disk_id] += len(remaining)
    return (
        [[math.fsum(cell) for cell in row] for row in terms],
        blocks,
    )


def _project_ssd_served_awaiting_link_by_npu_ssu(
    context: _Context,
    current_time_ms: float,
):
    """Bytes in a non-streaming active SSD command not yet link-visible."""
    values = [
        [0.0] * context.num_ssu for _ in range(context.num_npu)
    ]
    for ssu_id, disk in enumerate(context.disks):
        if disk.active_flows:
            flow = disk.active_flows[0]
            values[flow.npu_id][ssu_id] = _project_ssd_active_served_gb(
                flow, current_time_ms
            )
    return values


def _profile_generation_for_request(context: _Context, request_id: int):
    return context.profile_generation_by_request_id.get(request_id)


def _profile_cycle_npu_row(context: _Context, npu: _NPUState):
    batch = npu.active_batch
    active_ids = () if batch is None else batch.member_request_ids
    current_layer = None
    next_layer = None
    if npu.compute_active is not None:
        _, current_layer = npu.compute_active
        if current_layer + 1 < context.n_layers:
            next_layer = current_layer + 1
    elif batch is not None:
        candidate = batch.compute_done_up_to + 1
        if candidate < context.n_layers:
            next_layer = candidate
    prefetch_id = npu.layer0_prefetch_request_id
    prefetch = None
    if prefetch_id is not None:
        request = context.requests[prefetch_id]
        prefetch = {
            "request_id": int(prefetch_id),
            "generation": _profile_generation_for_request(context, prefetch_id),
            "layer": 0,
            "io_started": bool(request.io_started[0]),
            "io_ready": bool(request.io_ready[0]),
            "pending_blocks": int(request.pending_blocks[0]),
        }
    queue_head_id = npu.admission_queue[0] if npu.admission_queue else None
    return {
        "npu_id": int(npu.npu_id),
        "last_completed_generation": (
            context.profile_cycle_last_completed_generation_by_npu[npu.npu_id]
        ),
        "pipeline_state": _timeline_pipeline_state(context, npu),
        "active_request_ids": [int(request_id) for request_id in active_ids],
        "active_generations": [
            _profile_generation_for_request(context, request_id)
            for request_id in active_ids
        ],
        "current_compute_layer": current_layer,
        "next_compute_layer": next_layer,
        "compute_done_up_to": None if batch is None else batch.compute_done_up_to,
        "prefetched_layer0": prefetch,
        "admission_queue_head_request_id": (
            None if queue_head_id is None else int(queue_head_id)
        ),
        "admission_queue_head_generation": (
            None
            if queue_head_id is None
            else _profile_generation_for_request(context, queue_head_id)
        ),
    }


def _matrix_totals(values, num_npu: int, num_ssu: int):
    return {
        "by_npu": [math.fsum(row) for row in values],
        "by_ssu": [
            math.fsum(values[npu_id][ssu_id] for npu_id in range(num_npu))
            for ssu_id in range(num_ssu)
        ],
        "fleet": math.fsum(math.fsum(row) for row in values),
    }


def _request_profile_and_placement_forcing_sha256(request: _RequestState):
    """Fingerprint request inputs that determine compute and placed I/O work."""
    digest = hashlib.sha256(b"profile-placement-forcing-v1\0")
    digest.update(
        repr(
            (
                request.category,
                request.per_layer_compute_ms,
                request.manifest.placement,
            )
        ).encode()
    )
    return digest.hexdigest()


def _profile_cycle_forcing_identity(
    context: _Context,
    frontier_generation: int,
):
    """Expose adjacent exact profile+placement vectors for trend diagnosis."""
    period = context.profile_cycle_period

    def cycle(first_generation):
        last_generation = first_generation + period - 1
        request_ids = {}
        request_hashes = {}
        digest = hashlib.sha256(b"profile-cycle-forcing-vector-v1\0")
        for npu_id in context.profile_cycle_probe_npu_ids:
            ids = []
            hashes = []
            for slot, generation in enumerate(
                range(first_generation, last_generation + 1)
            ):
                request_id = context.profile_request_id_by_npu_generation.get(
                    (npu_id, generation)
                )
                if request_id is None:
                    return None
                forcing_hash = _request_profile_and_placement_forcing_sha256(
                    context.requests[request_id]
                )
                ids.append(int(request_id))
                hashes.append(forcing_hash)
                digest.update(repr((npu_id, slot, forcing_hash)).encode())
            request_ids[str(npu_id)] = ids
            request_hashes[str(npu_id)] = hashes
        return {
            "first_generation": int(first_generation),
            "last_generation": int(last_generation),
            "request_ids_by_probe_npu": request_ids,
            "request_forcing_sha256_by_probe_npu": request_hashes,
            "fleet_profile_and_placement_forcing_sha256": digest.hexdigest(),
        }

    current_first = frontier_generation - period + 1
    previous = cycle(current_first - period)
    current = cycle(current_first)
    following = cycle(current_first + period)
    previous_matches = (
        None
        if previous is None or current is None
        else previous["fleet_profile_and_placement_forcing_sha256"]
        == current["fleet_profile_and_placement_forcing_sha256"]
    )
    following_matches = (
        None
        if following is None or current is None
        else following["fleet_profile_and_placement_forcing_sha256"]
        == current["fleet_profile_and_placement_forcing_sha256"]
    )
    return {
        "current_cycle": current,
        "previous_cycle": previous,
        "following_cycle": following,
        "previous_cycle_matches_current": previous_matches,
        "following_cycle_matches_current": following_matches,
        "profile_period_also_repeats_exact_placement_forcing": (
            previous_matches is True and following_matches is True
        ),
        "interpretation": (
            "frontiers are long-term trend/drift samples; matching these "
            "fingerprints alone does not establish a repeating simulator state"
        ),
    }


def _capture_profile_cycle_frontier(
    context: _Context,
    *,
    frontier_generation: int,
    trigger_npu_id: int,
    trigger_request_id: int,
    current_time_ms: float,
):
    """Capture one pure-read post-completion profile-cycle frontier."""
    ssd_gb, ssd_blocks = _project_physical_ssd_outstanding_by_npu(
        context, current_time_ms
    )
    link_gb, link_blocks = _project_physical_link_outstanding_by_npu_ssu(
        context, current_time_ms
    )
    client_gb, client_blocks = _project_client_unissued_by_npu_ssu(context)
    ssd_to_link_gb = _project_ssd_served_awaiting_link_by_npu_ssu(
        context, current_time_ms
    )
    pipeline_gb = [
        [
            math.fsum(
                (
                    client_gb[npu_id][ssu_id],
                    ssd_gb[npu_id][ssu_id],
                    ssd_to_link_gb[npu_id][ssu_id],
                    link_gb[npu_id][ssu_id],
                )
            )
            for ssu_id in range(context.num_ssu)
        ]
        for npu_id in range(context.num_npu)
    ]
    npu_projection = _project_npu_snapshot(context, current_time_ms)
    probe_generations = [
        context.profile_cycle_last_completed_generation_by_npu[npu_id]
        for npu_id in context.profile_cycle_probe_npu_ids
    ]
    if any(generation is None for generation in probe_generations):
        raise AssertionError("profile-cycle frontier reached with an incomplete probe")
    phase_slots = [
        (int(generation) + 1) % context.profile_cycle_period
        for generation in probe_generations
    ]
    relative_generations = [
        int(generation) - int(frontier_generation)
        for generation in probe_generations
    ]
    return {
        "schema": "steady_profile_cycle_frontier_v1",
        "frontier_generation": int(frontier_generation),
        "cycle_index": (
            (frontier_generation + 1) // context.profile_cycle_period - 1
        ),
        "time_ms": float(current_time_ms),
        "elapsed_measurement_ms": float(
            current_time_ms - context.measurement_start_ms
        ),
        "trigger_npu_id": int(trigger_npu_id),
        "trigger_request_id": int(trigger_request_id),
        "last_completed_generation_by_probe_npu": {
            str(npu_id): int(generation)
            for npu_id, generation in zip(
                context.profile_cycle_probe_npu_ids, probe_generations
            )
        },
        "completed_phase_slot_by_probe_npu": {
            str(npu_id): int(slot)
            for npu_id, slot in zip(
                context.profile_cycle_probe_npu_ids, phase_slots
            )
        },
        "relative_generation_by_probe_npu": {
            str(npu_id): relative_generation
            for npu_id, relative_generation in zip(
                context.profile_cycle_probe_npu_ids, relative_generations
            )
        },
        "generation_phase_spread": int(
            max(probe_generations) - min(probe_generations)
        ),
        "profile_and_placement_forcing": _profile_cycle_forcing_identity(
            context, frontier_generation
        ),
        "npu_compute_cumulative_busy_ms_by_npu": list(
            npu_projection["compute_cumulative_busy_ms_by_npu"]
        ),
        "completed_requests_cumulative_by_npu": list(context.completed_by_npu),
        "npu_state": [
            _profile_cycle_npu_row(context, npu) for npu in context.npus
        ],
        "inventory": {
            "ssd_submitted_cumulative_gb_by_npu_ssu": [
                list(row) for row in context.profile_cycle_ssd_submitted_gb
            ],
            "ssd_submitted_cumulative_gb_totals": _matrix_totals(
                context.profile_cycle_ssd_submitted_gb,
                context.num_npu,
                context.num_ssu,
            ),
            "ssd_outstanding_gb_by_npu_ssu": ssd_gb,
            "ssd_outstanding_blocks_by_npu_ssu": ssd_blocks,
            "ssd_outstanding_gb_totals": _matrix_totals(
                ssd_gb, context.num_npu, context.num_ssu
            ),
            "npu_link_outstanding_gb_by_npu_ssu": link_gb,
            "npu_link_outstanding_blocks_by_npu_ssu": link_blocks,
            "npu_link_outstanding_gb_totals": _matrix_totals(
                link_gb, context.num_npu, context.num_ssu
            ),
            "client_unissued_gb_by_npu_ssu": client_gb,
            "client_unissued_blocks_by_npu_ssu": client_blocks,
            "client_unissued_gb_totals": _matrix_totals(
                client_gb, context.num_npu, context.num_ssu
            ),
            "ssd_served_awaiting_link_enqueue_gb_by_npu_ssu": ssd_to_link_gb,
            "ssd_served_awaiting_link_enqueue_gb_totals": _matrix_totals(
                ssd_to_link_gb, context.num_npu, context.num_ssu
            ),
            "total_physical_io_outstanding_gb_by_npu_ssu": pipeline_gb,
            "total_physical_io_outstanding_gb_totals": _matrix_totals(
                pipeline_gb, context.num_npu, context.num_ssu
            ),
        },
    }


def _project_timeline_ssd_service(context: _Context, current_time_ms: float):
    """Project exact cumulative SSD bytes for every NPU x SSU cell."""
    return _project_ssd_service_by_npu(context, current_time_ms)


def _project_timeline_link_service(context: _Context, current_time_ms: float):
    """Project exact cumulative NPU-link bytes for every NPU x SSU cell."""
    values = [
        [
            float(npu.link_served_gb_by_ssu.get(ssu_id, 0.0))
            for ssu_id in range(context.num_ssu)
        ]
        for npu in context.npus
    ]
    for npu in context.npus:
        flow = npu.link_active_flow
        if flow is None:
            continue
        service_end_ms = min(current_time_ms, flow.link_end_time)
        duration_ms = max(0.0, service_end_ms - npu.link_last_account_ms)
        values[npu.npu_id][flow.disk_id] += (
            context.npu_bw_gbps * duration_ms / 1000.0
        )
    return values


def _timeline_compute_inventory_ms(
    context: _Context,
    npu: _NPUState,
    current_time_ms: float,
):
    """Activated, unfinished compute Q(t); queued future requests are excluded."""
    batch = npu.active_batch
    if batch is None:
        return 0.0
    per_layer_ms = _batch_compute_duration_ms(context, batch)
    if npu.compute_active is not None:
        batch_id, layer = npu.compute_active
        if batch_id != batch.batch_id:
            raise AssertionError("timeline observed a mismatched compute batch")
        remaining_current_ms = max(
            0.0, batch.layer_metrics[layer].compute_end_ms - current_time_ms
        )
        return remaining_current_ms + (context.n_layers - layer - 1) * per_layer_ms
    next_layer = batch.compute_done_up_to + 1
    return max(0, context.n_layers - next_layer) * per_layer_ms


def _timeline_pipeline_state(context: _Context, npu: _NPUState):
    if npu.compute_active is not None:
        return "compute"
    batch = npu.active_batch
    if batch is not None:
        layer = batch.compute_done_up_to + 1
        if layer < context.n_layers and all(
            context.requests[request_id].io_ready[layer]
            for request_id in batch.member_request_ids
        ):
            return "ready_not_running"
        return "io_barrier"
    if npu.batch_dispatch_pending or npu.admission_queue:
        return "between_batches"
    if npu.future_arrivals > 0:
        return "waiting_arrival"
    return "drained"


def _timeline_active_request_ids(npu: _NPUState):
    active = (
        () if npu.active_batch is None else npu.active_batch.member_request_ids
    )
    prefetch = (
        ()
        if npu.layer0_prefetch_request_id is None
        else (npu.layer0_prefetch_request_id,)
    )
    return tuple(dict.fromkeys(active + prefetch))


def _timeline_request_physical_remaining(
    context: _Context,
    request: _RequestState,
    current_time_ms: float,
):
    """Bytes not yet delivered on the NPU link, including partial transfers."""
    remaining = [0.0] * context.num_ssu
    for layer in range(context.n_layers):
        if request.io_ready[layer]:
            continue
        for ssu_id, _, work_gb in _state_placement_groups(request, layer):
            remaining[ssu_id] += work_gb
        if request.completed_gb_by_layer_ssu:
            for ssu_id, delivered_gb in enumerate(
                request.completed_gb_by_layer_ssu[layer]
            ):
                remaining[ssu_id] -= delivered_gb
    npu = context.npus[request.manifest.npu_id]
    flow = npu.link_active_flow
    if flow is not None and flow.request_id == request.manifest.request_id:
        service_end_ms = min(current_time_ms, flow.link_end_time)
        elapsed_ms = max(0.0, service_end_ms - flow.link_start_time)
        partial_gb = min(
            flow.total_gb, context.npu_bw_gbps * elapsed_ms / 1000.0
        )
        remaining[flow.disk_id] -= partial_gb
    return [max(0.0, value) for value in remaining]


def _timeline_path_state(context: _Context, current_time_ms: float):
    rows = []
    active_commands = []
    pressure_state = []
    for ssu_id, disk in enumerate(context.disks):
        scheduler = disk.scheduler
        rates = {}
        if scheduler.policy == sim.POLICY_QOS_STATIC_CIR:
            backlogged = [path for path in scheduler.paths.values() if path.pending]
            rates = sim._static_qos_service_rates(
                backlogged, scheduler.disk_bw, scheduler.group_weights
            )
            for path in scheduler.paths.values():
                if path.io_count() <= 0:
                    continue
                head = path.peek()
                active = path.active_flow
                rows.append(
                    {
                        "ssu_id": ssu_id,
                        "path_id": path.path_id,
                        "group_id": path.group_id,
                        "cir_gbps": float(path.cir),
                        "path_weight": float(path.path_weight),
                        "virtual_finish": float(path.virtual_finish),
                        "estimated_next_arbitration_rate_gbps": float(
                            rates.get(path, 0.0)
                        ),
                        "pending_blocks": int(path.pending_io_count),
                        "pending_gb": float(path.pending_gb),
                        "active_remaining_gb": _project_ssd_active_remaining_gb(
                            active, current_time_ms
                        ),
                        "head_wait_age_ms": (
                            max(0.0, current_time_ms - head.enqueue_time)
                            if head is not None
                            else None
                        ),
                        "head_npu_id": head.npu_id if head is not None else None,
                        "head_request_id": (
                            head.request_id if head is not None else None
                        ),
                        "head_layer": head.layer if head is not None else None,
                    }
                )
        active = disk.active_flows[0] if disk.active_flows else None
        active_commands.append(
            None
            if active is None
            else {
                "ssu_id": ssu_id,
                "npu_id": active.npu_id,
                "request_id": active.request_id,
                "layer": active.layer,
                "block_idx": active.block_idx,
                "path_id": active.queue_id,
                "remaining_gb": _project_ssd_active_remaining_gb(
                    active, current_time_ms
                ),
                # ``flow.start_time`` remains the most recent scheduler-settle
                # edge for compatibility; this field is the immutable physical
                # command activation edge used for progress reconstruction.
                "command_start_time_ms": float(active.ssd_activation_time),
                "command_age_ms": max(
                    0.0,
                    current_time_ms - active.ssd_activation_time,
                ),
                "physical_service_gbps": float(active.bw),
                "non_preemptive": True,
            }
        )
        cache_time = scheduler._pressure_cache_time
        pressure_state.append(
            {
                "ssu_id": ssu_id,
                "reports_cumulative": int(scheduler.pressure_reports),
                "cache_hits_cumulative": int(scheduler.pressure_cache_hits),
                "cache_time_ms": cache_time,
                "cache_age_ms": (
                    None
                    if cache_time is None
                    else max(0.0, current_time_ms - cache_time)
                ),
                "ttl_ms": float(scheduler.pressure_ttl_ms),
            }
        )
    return {
        "sparse_ssu_path_rows": rows,
        "active_command_by_ssu": active_commands,
        "pressure_state_by_ssu": pressure_state,
    }


def _capture_timeline_snapshot(
    context: _Context,
    current_time_ms: float,
    *,
    ssd_snapshot=None,
):
    ssd_served = _project_timeline_ssd_service(context, current_time_ms)
    fragmented_ssd_served = _project_fragmented_ssd_service_by_npu(
        context, current_time_ms
    )
    link_served = _project_timeline_link_service(context, current_time_ms)
    (
        ssd_outstanding_gb,
        ssd_outstanding_blocks,
    ) = _project_physical_ssd_outstanding_by_npu(
        context, current_time_ms
    )
    stable_counter_ssd_outstanding_gb = [
        [
            max(0.0, context.timeline_ssd_enqueued_gb[npu][ssu] - ssd_served[npu][ssu])
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    fragmented_counter_ssd_outstanding_gb = [
        [
            max(
                0.0,
                context.timeline_ssd_enqueued_gb[npu][ssu]
                - fragmented_ssd_served[npu][ssu],
            )
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    link_outstanding_gb = [
        [
            max(
                0.0,
                context.timeline_link_enqueued_gb[npu][ssu]
                - link_served[npu][ssu],
            )
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    # SSD commands are non-streaming: partially serviced bytes remain inside
    # the active SSD command and are enqueued to the NPU link only when the
    # whole command completes.  Keep this real intermediate stage explicit.
    ssd_served_awaiting_link_enqueue_gb = [
        [
            ssd_served[npu][ssu]
            - context.timeline_link_enqueued_gb[npu][ssu]
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    counter_ssd_outstanding_blocks = [
        [
            context.timeline_ssd_enqueued_blocks[npu][ssu]
            - context.timeline_ssd_completed_blocks[npu][ssu]
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    if ssd_snapshot is None:
        ssd_snapshot = _project_ssd_snapshot(context, current_time_ms)
    ssd_accounting_residuals = []
    for ssu in range(context.num_ssu):
        enqueued_total = math.fsum(
            context.timeline_ssd_enqueued_gb[npu][ssu]
            for npu in range(context.num_npu)
        )
        stable_service_total = math.fsum(
            ssd_served[npu][ssu] for npu in range(context.num_npu)
        )
        fragmented_service_total = math.fsum(
            fragmented_ssd_served[npu][ssu]
            for npu in range(context.num_npu)
        )
        physical_queue_total = math.fsum(
            ssd_outstanding_gb[npu][ssu]
            for npu in range(context.num_npu)
        )
        stable_counter_queue_total = math.fsum(
            stable_counter_ssd_outstanding_gb[npu][ssu]
            for npu in range(context.num_npu)
        )
        fragmented_counter_queue_total = math.fsum(
            fragmented_counter_ssd_outstanding_gb[npu][ssu]
            for npu in range(context.num_npu)
        )
        per_npu_queue_identity = [
            math.fsum(
                (
                    context.timeline_ssd_enqueued_gb[npu][ssu],
                    -ssd_served[npu][ssu],
                    -ssd_outstanding_gb[npu][ssu],
                )
            )
            for npu in range(context.num_npu)
        ]
        max_identity_npu = max(
            range(context.num_npu),
            key=lambda npu: abs(per_npu_queue_identity[npu]),
        )
        ssd_accounting_residuals.append(
            {
                "ssu_id": ssu,
                "stable_service_minus_busy_counter_gb": (
                    stable_service_total
                    - ssd_snapshot["cumulative_served_gb_by_ssu"][ssu]
                ),
                "fragmented_service_minus_stable_gb": (
                    fragmented_service_total - stable_service_total
                ),
                "physical_queue_minus_scheduler_gb": (
                    physical_queue_total
                    - ssd_snapshot["outstanding_gb_by_ssu"][ssu]
                ),
                "enqueue_minus_service_minus_physical_queue_gb": math.fsum(
                    (enqueued_total, -stable_service_total, -physical_queue_total)
                ),
                "counter_queue_minus_physical_queue_gb": (
                    stable_counter_queue_total - physical_queue_total
                ),
                "fragmented_counter_queue_minus_physical_queue_gb": (
                    fragmented_counter_queue_total - physical_queue_total
                ),
                "maximum_abs_npu_queue_identity_residual_gb": abs(
                    per_npu_queue_identity[max_identity_npu]
                ),
                "maximum_abs_npu_queue_identity_residual_npu_id": (
                    max_identity_npu
                ),
                "physical_queue_block_minus_scheduler_blocks": (
                    sum(
                        ssd_outstanding_blocks[npu][ssu]
                        for npu in range(context.num_npu)
                    )
                    - ssd_snapshot["outstanding_blocks_by_ssu"][ssu]
                ),
                "counter_queue_block_minus_physical_blocks": (
                    sum(
                        counter_ssd_outstanding_blocks[npu][ssu]
                        for npu in range(context.num_npu)
                    )
                    - sum(
                        ssd_outstanding_blocks[npu][ssu]
                        for npu in range(context.num_npu)
                    )
                ),
            }
        )
    link_outstanding_blocks = [
        [
            context.timeline_link_enqueued_blocks[npu][ssu]
            - context.timeline_link_completed_blocks[npu][ssu]
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    client_unissued_gb = [
        [0.0] * context.num_ssu for _ in range(context.num_npu)
    ]
    for state in context.submission_states.values():
        client_unissued_gb[state.npu_id][state.disk_id] += math.fsum(
            size_gb for _, size_gb in state.blocks[state.cursor :]
        )

    controller_remaining = [
        [0.0] * context.num_ssu for _ in range(context.num_npu)
    ]
    physical_remaining = [
        [0.0] * context.num_ssu for _ in range(context.num_npu)
    ]
    controller_compute_ms = [0.0] * context.num_npu
    controller_request_ids = [None] * context.num_npu
    controller_prefetch_only = [None] * context.num_npu
    npu_rows = []
    for npu in context.npus:
        request_ids = _timeline_active_request_ids(npu)
        for request_id in request_ids:
            request = context.requests[request_id]
            view = _control_request_view(context, request)
            remaining_work, compute_ms = view.remaining_work_gb_by_ssu, (
                view.remaining_compute_budget_ms
            )
            if compute_ms > 0.0 and any(value > 0.0 for value in remaining_work):
                if controller_request_ids[npu.npu_id] is not None:
                    raise AssertionError(
                        "timeline observed two controller demands on one NPU"
                    )
                controller_request_ids[npu.npu_id] = request_id
                controller_prefetch_only[npu.npu_id] = bool(view.prefetch_only)
                controller_compute_ms[npu.npu_id] = float(compute_ms)
                controller_remaining[npu.npu_id] = list(map(float, remaining_work))
            request_physical = _timeline_request_physical_remaining(
                context, request, current_time_ms
            )
            for ssu_id, value in enumerate(request_physical):
                physical_remaining[npu.npu_id][ssu_id] += value

        batch = npu.active_batch
        active_request_id = (
            None if batch is None else int(batch.member_request_ids[0])
        )
        active_request = (
            None if active_request_id is None else context.requests[active_request_id]
        )
        current_compute_layer = None
        next_compute_layer = None
        compute_start_ms = None
        compute_end_ms = None
        if npu.compute_active is not None:
            _, current_compute_layer = npu.compute_active
            metric = batch.layer_metrics[current_compute_layer]
            compute_start_ms = metric.compute_start_ms
            compute_end_ms = metric.compute_end_ms
            candidate = current_compute_layer + 1
            if candidate < context.n_layers:
                next_compute_layer = candidate
        elif batch is not None:
            candidate = batch.compute_done_up_to + 1
            if candidate < context.n_layers:
                next_compute_layer = candidate
        profile = (
            {}
            if active_request is None
            else _request_profile_metadata(active_request.manifest.load)
        )
        ideal_ttft_ms = (
            None
            if active_request is None
            else context.n_layers * active_request.per_layer_compute_ms
        )
        elapsed_ttft_ms = (
            None
            if active_request is None
            else max(0.0, current_time_ms - active_request.admission_time_ms)
        )
        npu_rows.append(
            {
                "npu_id": npu.npu_id,
                "pipeline_state": _timeline_pipeline_state(context, npu),
                "compute_inventory_q_ms": _timeline_compute_inventory_ms(
                    context, npu, current_time_ms
                ),
                "activated_compute_cumulative_ms": float(
                    context.timeline_activated_compute_ms[npu.npu_id]
                ),
                "active_request_id": active_request_id,
                "active_batch_id": None if batch is None else batch.batch_id,
                "current_compute_layer": current_compute_layer,
                "next_compute_layer": next_compute_layer,
                "compute_done_up_to": (
                    None if batch is None else batch.compute_done_up_to
                ),
                "compute_start_ms": compute_start_ms,
                "compute_end_ms": compute_end_ms,
                "next_compute_layer_io_ready": (
                    None
                    if active_request is None or next_compute_layer is None
                    else bool(active_request.io_ready[next_compute_layer])
                ),
                "waiting_on_io_layer": (
                    next_compute_layer
                    if _timeline_pipeline_state(context, npu) == "io_barrier"
                    else None
                ),
                "prefetch_request_id": npu.layer0_prefetch_request_id,
                "controller_request_id": controller_request_ids[npu.npu_id],
                "controller_prefetch_only": controller_prefetch_only[npu.npu_id],
                "admission_time_ms": (
                    None if active_request is None else active_request.admission_time_ms
                ),
                "elapsed_ttft_ms": elapsed_ttft_ms,
                "ideal_ttft_ms": ideal_ttft_ms,
                "slo_alpha1p5_slack_ms": (
                    None
                    if ideal_ttft_ms is None
                    else 1.5 * ideal_ttft_ms - elapsed_ttft_ms
                ),
                "slo_alpha2_slack_ms": (
                    None
                    if ideal_ttft_ms is None
                    else 2.0 * ideal_ttft_ms - elapsed_ttft_ms
                ),
                "category": (
                    None if active_request is None else active_request.category
                ),
                "sequence": (
                    None
                    if active_request is None
                    else int(active_request.manifest.load["stream_id"])
                ),
                **profile,
            }
        )

    controller_demand = [
        [
            (
                controller_remaining[npu][ssu]
                / (controller_compute_ms[npu] / 1000.0)
                if controller_compute_ms[npu] > 0.0
                else 0.0
            )
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    physical_demand = [
        [
            (
                physical_remaining[npu][ssu]
                / (controller_compute_ms[npu] / 1000.0)
                if controller_compute_ms[npu] > 0.0
                else 0.0
            )
            for ssu in range(context.num_ssu)
        ]
        for npu in range(context.num_npu)
    ]
    installed_cir = None
    if context.npu_dedicated_paths is not None:
        installed_cir = [
            [
                float(
                    context.disks[ssu].scheduler.paths[
                        context.npu_dedicated_paths[npu]
                    ].cir
                )
                for ssu in range(context.num_ssu)
            ]
            for npu in range(context.num_npu)
        ]
    return {
        "schema": "steady_timeline_boundary_v3",
        "ssd_accounting_residuals_by_ssu": ssd_accounting_residuals,
        "npu_rows": npu_rows,
        "npu_ssu": {
            "ssd_enqueued_cumulative_gb": [
                list(row) for row in context.timeline_ssd_enqueued_gb
            ],
            "ssd_served_cumulative_gb": ssd_served,
            "ssd_served_fragmented_diagnostic_cumulative_gb": (
                fragmented_ssd_served
            ),
            "ssd_outstanding_gb": ssd_outstanding_gb,
            "ssd_outstanding_blocks": ssd_outstanding_blocks,
            "link_enqueued_cumulative_gb": [
                list(row) for row in context.timeline_link_enqueued_gb
            ],
            "link_served_cumulative_gb": link_served,
            "link_outstanding_gb": link_outstanding_gb,
            "link_outstanding_blocks": link_outstanding_blocks,
            "ssd_served_awaiting_link_enqueue_gb": (
                ssd_served_awaiting_link_enqueue_gb
            ),
            "client_unissued_gb": client_unissued_gb,
            "activated_io_cumulative_gb": [
                list(row) for row in context.timeline_activated_io_gb
            ],
            "physical_remaining_gb": physical_remaining,
            "controller_declared_remaining_gb": controller_remaining,
            "controller_remaining_compute_ms": list(controller_compute_ms),
            "physical_demand_gbps": physical_demand,
            "controller_demand_gbps": controller_demand,
            "installed_dedicated_path_cir_gbps": installed_cir,
            "route_plans_cumulative": [
                list(row) for row in context.timeline_route_plans
            ],
            "route_pressure_fresh_cumulative": [
                list(row) for row in context.timeline_route_pressure_fresh
            ],
            "route_pressure_cache_cumulative": [
                list(row) for row in context.timeline_route_pressure_cache
            ],
            "route_blocks_by_group_cumulative": [
                [[int(value) for value in groups] for groups in row]
                for row in context.timeline_route_blocks_by_group
            ],
        },
        **_timeline_path_state(context, current_time_ms),
    }


def _prepare_timeline_dispatch_probe(
    context: _Context,
    scheduler: sim.DiskIOScheduler,
    current_time_ms: float,
):
    """Predict one QoS winner read-only inside a bounded measurement probe."""
    config = context.steady_state
    if (
        not context.timeline_diagnostics
        or not context.measurement_open
        or context.measurement_start_ms is None
        or current_time_ms
        >= context.measurement_start_ms + config.timeline_dispatch_probe_ms
        or len(context.timeline_dispatch_probe_records)
        >= config.timeline_dispatch_probe_limit
        or scheduler.policy != sim.POLICY_QOS_STATIC_CIR
        or scheduler.state.active_flows
    ):
        return None
    backlogged = [path for path in scheduler.paths.values() if path.pending]
    if not backlogged:
        return None
    rates = sim._static_qos_service_rates(
        backlogged, scheduler.disk_bw, scheduler.group_weights
    )
    candidates = []
    for path in backlogged:
        head = path.peek()
        rate = float(rates.get(path, 0.0))
        if head is None or rate <= _EPS or path.pir <= _EPS:
            continue
        finish = float(path.virtual_finish + head.total_gb / rate)
        candidates.append((path, head, rate, finish))
    if not candidates:
        return None
    minimum_finish = min(candidate[3] for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if candidate[3] <= minimum_finish + _EPS
    ]
    path_count = len(scheduler.paths)
    winner = min(
        tied,
        key=lambda candidate: (
            (candidate[0].path_id - scheduler.qos_rr_cursor) % path_count,
            candidate[0].path_id,
            candidate[3],
        ),
    )
    return {
        "ssu_id": scheduler.state.disk_id,
        "time_ms": float(current_time_ms),
        "rr_cursor_before": int(scheduler.qos_rr_cursor),
        "candidate_path_count": len(candidates),
        "minimum_finish_tag": float(minimum_finish),
        "finish_tie_count": len(tied),
        "expected_path_id": int(winner[0].path_id),
        "winner_finish_tag_before": float(winner[3]),
        "winner_estimated_arbitration_rate_gbps": float(winner[2]),
        "winner_cir_gbps": float(winner[0].cir),
        "winner_group_id": int(winner[0].group_id),
        "winner_path_weight": float(winner[0].path_weight),
        "winner_pending_blocks_before": int(winner[0].pending_io_count),
        "winner_pending_gb_before": float(winner[0].pending_gb),
        "selection_rule": "minimum_virtual_finish_then_round_robin",
    }


def _finish_timeline_dispatch_probe(
    context: _Context,
    scheduler: sim.DiskIOScheduler,
    probe,
    flow,
):
    if probe is None:
        return
    if flow is None or flow.queue_id != probe["expected_path_id"]:
        raise AssertionError(
            "read-only timeline arbitration replay disagreed with scheduler"
        )
    path = scheduler.paths[flow.queue_id]
    probe.update(
        {
            "actual_path_id": int(flow.queue_id),
            "winner_npu_id": int(flow.npu_id),
            "winner_request_id": int(flow.request_id),
            "winner_layer": int(flow.layer),
            "winner_block_idx": int(flow.block_idx),
            "winner_queue_wait_ms": float(flow.ssd_queue_wait_ms),
            "winner_command_gb": float(flow.total_gb),
            "winner_virtual_finish_after": float(path.virtual_finish),
            "physical_command_service_gbps": float(flow.bw),
            "physical_command_non_preemptive": True,
            "prediction_matches_actual": True,
        }
    )
    context.timeline_dispatch_probe_records.append(probe)


def _capture_stationarity_snapshot(
    context: _Context,
    boundary: int,
    current_time_ms: float,
):
    """Capture a left-limit boundary without settling any simulated resource."""
    ssd = _project_ssd_snapshot(context, current_time_ms)
    npu = _project_npu_snapshot(context, current_time_ms)
    snapshot = {
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
    if context.timeline_diagnostics:
        snapshot["timeline"] = _capture_timeline_snapshot(
            context,
            current_time_ms,
            ssd_snapshot=ssd,
        )
    return snapshot


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
    if context.profile_cycle_probe_enabled:
        total, compensation = sim.DiskState._kahan_add(
            context.profile_cycle_ssd_submitted_gb[flow.npu_id][flow.disk_id],
            context.profile_cycle_ssd_submitted_compensation_gb[flow.npu_id][
                flow.disk_id
            ],
            flow.total_gb,
        )
        context.profile_cycle_ssd_submitted_gb[flow.npu_id][flow.disk_id] = total
        context.profile_cycle_ssd_submitted_compensation_gb[flow.npu_id][
            flow.disk_id
        ] = compensation
    if context.timeline_diagnostics:
        total, compensation = sim.DiskState._kahan_add(
            context.timeline_ssd_enqueued_gb[flow.npu_id][flow.disk_id],
            context.timeline_ssd_enqueued_compensation_gb[flow.npu_id][
                flow.disk_id
            ],
            flow.total_gb,
        )
        context.timeline_ssd_enqueued_gb[flow.npu_id][flow.disk_id] = total
        context.timeline_ssd_enqueued_compensation_gb[flow.npu_id][
            flow.disk_id
        ] = compensation
        context.timeline_ssd_enqueued_blocks[flow.npu_id][flow.disk_id] += (
            flow.block_count
        )
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
    if context.timeline_diagnostics:
        for ssu_id, _, work_gb in placement_groups:
            context.timeline_activated_io_gb[request.manifest.npu_id][
                ssu_id
            ] += work_gb
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


def _record_timeline_route_plan(
    context: _Context,
    state: _SubmissionState,
    window_start: int,
    current_time_ms: float,
    rule: str,
    pressure_source: Optional[str],
    pressure=None,
):
    """Record cumulative routing facts plus a bounded causal input sample."""
    if not context.timeline_diagnostics:
        return
    npu_id = state.npu_id
    ssu_id = state.disk_id
    context.timeline_route_plans[npu_id][ssu_id] += 1
    if pressure_source == "fresh":
        context.timeline_route_pressure_fresh[npu_id][ssu_id] += 1
    elif pressure_source == "cache":
        context.timeline_route_pressure_cache[npu_id][ssu_id] += 1
    groups = context.timeline_route_blocks_by_group[npu_id][ssu_id]
    if not groups:
        return
    scheduler = context.disks[ssu_id].scheduler
    selected_path_ids = tuple(state.planned_path_ids[window_start:])
    for path_id in selected_path_ids:
        groups[scheduler.paths[path_id].group_id] += 1

    config = context.steady_state
    if (
        not context.measurement_open
        or context.measurement_start_ms is None
        or current_time_ms
        >= context.measurement_start_ms + config.timeline_dispatch_probe_ms
        or len(context.timeline_route_probe_records)
        >= config.timeline_dispatch_probe_limit
    ):
        return
    block_slice = state.blocks[window_start : window_start + len(selected_path_ids)]
    allowed = tuple(int(path_id) for path_id in state.allowed_path_ids)
    cache_time = scheduler._pressure_cache_time
    context.timeline_route_probe_records.append(
        {
            "time_ms": float(current_time_ms),
            "rule": str(rule),
            "npu_id": int(npu_id),
            "ssu_id": int(ssu_id),
            "request_id": int(state.request_id),
            "layer": int(state.layer),
            "category": str(state.category),
            "start_offset": int(state.start_offset),
            "pressure_source": pressure_source,
            "pressure_snapshot_time_ms": (
                None
                if pressure is None
                else float(current_time_ms if cache_time is None else cache_time)
            ),
            "pressure_age_ms": (
                None
                if pressure is None or cache_time is None
                else max(0.0, float(current_time_ms - cache_time))
            ),
            "pressure_ttl_ms": float(scheduler.pressure_ttl_ms),
            "allowed_path_ids": allowed,
            "allowed_path_pressure_counts": (
                None
                if pressure is None
                else tuple(int(pressure.counts[path_id]) for path_id in allowed)
            ),
            "allowed_path_cir_gbps": tuple(
                float(scheduler.paths[path_id].cir) for path_id in allowed
            ),
            "allowed_path_pir_gbps_or_null": tuple(
                (
                    float(scheduler.paths[path_id].pir)
                    if math.isfinite(scheduler.paths[path_id].pir)
                    else None
                )
                for path_id in allowed
            ),
            "allowed_path_weights": tuple(
                float(scheduler.paths[path_id].path_weight) for path_id in allowed
            ),
            "allowed_path_group_ids": tuple(
                int(scheduler.paths[path_id].group_id) for path_id in allowed
            ),
            "disk_bw_gbps": float(scheduler.disk_bw),
            "path_count": len(scheduler.paths),
            "paths_per_group": (
                len(scheduler.paths) // len(scheduler.group_weights)
            ),
            "group_weights": tuple(float(value) for value in scheduler.group_weights),
            "group_io_counts": (
                None
                if pressure is None
                else tuple(int(value) for value in pressure.group_io_counts)
            ),
            "active_paths_per_group": (
                None
                if pressure is None
                else tuple(
                    int(value) for value in pressure.active_paths_per_group
                )
            ),
            "active_path_weights": (
                None
                if pressure is None
                else tuple(float(value) for value in pressure.active_path_weights)
            ),
            "active_group_weight_sum": (
                None if pressure is None else float(pressure.active_group_weight_sum)
            ),
            "active_cir_sum_gbps": (
                None if pressure is None else float(pressure.active_cir_sum)
            ),
            "block_indices": tuple(int(index) for index, _ in block_slice),
            "block_sizes_gb": tuple(float(size_gb) for _, size_gb in block_slice),
            "selected_path_ids": selected_path_ids,
            "selected_group_ids": tuple(
                int(scheduler.paths[path_id].group_id)
                for path_id in selected_path_ids
            ),
        }
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
        _record_timeline_route_plan(
            context,
            state,
            window_start,
            current_time_ms,
            "reserved_layer0_path",
            None,
        )
        return
    if context.npu_dedicated_paths is not None:
        state.planned_path_ids.extend(
            state.allowed_path_ids[0] for _ in range(window_start, len(state.blocks))
        )
        _record_timeline_route_plan(
            context,
            state,
            window_start,
            current_time_ms,
            "npu_dedicated_path",
            None,
        )
        return
    mode = context.client_io_config.path_selection_mode
    if mode == sim.PATH_SELECTION_FIXED_PATH_ZERO:
        state.planned_path_ids.extend(
            policy_baseline_path_ids(len(state.blocks) - window_start)
        )
        _record_timeline_route_plan(
            context,
            state,
            window_start,
            current_time_ms,
            "fixed_path_zero",
            None,
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
    cache_hit = bool(
        scheduler.pressure_ttl_ms > 0.0
        and scheduler._pressure_cache is not None
        and current_time_ms
        < scheduler._pressure_cache_time + scheduler.pressure_ttl_ms
    )
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
    _record_timeline_route_plan(
        context,
        state,
        window_start,
        current_time_ms,
        "pressure_aware_once_per_layer",
        "cache" if cache_hit else "fresh",
        pressure,
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
    coalesced = bool(reasons)
    if not coalesced:
        context.push_event(scheduled_time_ms, CIR_CONTROL, 0)
    reasons.add(reason)
    if context.timeline_diagnostics:
        context.timeline_control_triggers.append(
            {
                "raw_time_ms": float(current_time_ms),
                "effective_time_ms": float(scheduled_time_ms),
                "reason": str(reason),
                "rate_limited": scheduled_time_ms > current_time_ms + _EPS,
                "coalesced": coalesced,
                "min_interval_ms": float(context.control.min_interval_ms),
            }
        )


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
    if context.timeline_diagnostics:
        context.timeline_activated_compute_ms[npu_id] += (
            context.n_layers * _batch_compute_duration_ms(context, batch)
        )
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
    # Settle the physical busy counter at the common window boundary.  Stable
    # NPU attribution below combines once-per-command completed bytes with the
    # immutable active-command prefix, so a straddling command is neither lost
    # nor fragmented by unrelated scheduler observations.
    for disk in context.disks:
        disk.scheduler.settle(current_time_ms)
    context.measurement_disk_busy_start_ms = tuple(
        disk.busy_time for disk in context.disks
    )
    stable_ssd_service = _project_ssd_service_by_npu(context, current_time_ms)
    context.measurement_npu_ssu_served_start_gb = tuple(
        tuple(row) for row in stable_ssd_service
    )
    fragmented_ssd_service = _project_fragmented_ssd_service_by_npu(
        context, current_time_ms
    )
    context.measurement_fragmented_npu_ssu_served_start_gb = tuple(
        tuple(row) for row in fragmented_ssd_service
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
    if context.profile_cycle_probe_enabled:
        completed = [
            context.profile_cycle_last_completed_generation_by_npu[npu_id]
            for npu_id in context.profile_cycle_probe_npu_ids
        ]
        if any(generation is None for generation in completed):
            raise AssertionError(
                "steady warm-up opened before every profile-cycle probe completed"
            )
        minimum_generation = min(completed)
        period = context.profile_cycle_period
        context.profile_cycle_next_frontier_generation = (
            ((minimum_generation + 1) // period + 1) * period - 1
        )
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


def _observe_profile_request_completion(
    context: _Context,
    npu: _NPUState,
    batch: _MicrobatchState,
    current_time_ms: float,
):
    """Advance and, if due, sample the optional profile-cycle frontier."""
    if (
        not context.profile_cycle_probe_enabled
        or npu.npu_id not in context.profile_cycle_probe_npu_ids
    ):
        return
    if len(batch.member_request_ids) != 1:
        raise AssertionError("profile-cycle probe requires singleton batches")
    request_id = batch.member_request_ids[0]
    generation = context.profile_generation_by_request_id[request_id]
    previous = context.profile_cycle_last_completed_generation_by_npu[npu.npu_id]
    if previous is not None and generation != previous + 1:
        raise AssertionError(
            "profile-cycle request generations did not complete consecutively"
        )
    context.profile_cycle_last_completed_generation_by_npu[npu.npu_id] = generation
    if not context.measurement_open:
        return
    frontier = context.profile_cycle_next_frontier_generation
    if frontier is None:
        raise AssertionError("profile-cycle frontier was not initialized")
    minimum_generation = min(
        context.profile_cycle_last_completed_generation_by_npu[probe_npu_id]
        for probe_npu_id in context.profile_cycle_probe_npu_ids
    )
    while minimum_generation >= frontier:
        context.profile_cycle_pending_frontiers.append(
            {
                "frontier_generation": int(frontier),
                "trigger_npu_id": int(npu.npu_id),
                "trigger_request_id": int(request_id),
                "time_ms": float(current_time_ms),
            }
        )
        frontier += context.profile_cycle_period
    context.profile_cycle_next_frontier_generation = frontier


def _flush_profile_cycle_frontiers(
    context: _Context,
    current_time_ms: float,
):
    """Capture pending frontiers at the normalized post-timestamp cut."""
    pending = context.profile_cycle_pending_frontiers
    if not pending:
        return
    if any(
        not math.isclose(
            row["time_ms"], current_time_ms, rel_tol=0.0, abs_tol=_EPS
        )
        for row in pending
    ):
        raise AssertionError("profile-cycle frontier crossed two unflushed timestamps")
    context.profile_cycle_frontier_snapshots.extend(
        _capture_profile_cycle_frontier(
            context,
            frontier_generation=row["frontier_generation"],
            trigger_npu_id=row["trigger_npu_id"],
            trigger_request_id=row["trigger_request_id"],
            current_time_ms=current_time_ms,
        )
        for row in pending
    )
    pending.clear()


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
    stable_ssd_service = _project_ssd_service_by_npu(context, current_time_ms)
    context.measurement_npu_ssu_served_end_gb = tuple(
        tuple(row) for row in stable_ssd_service
    )
    fragmented_ssd_service = _project_fragmented_ssd_service_by_npu(
        context, current_time_ms
    )
    context.measurement_fragmented_npu_ssu_served_end_gb = tuple(
        tuple(row) for row in fragmented_ssd_service
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
    # Mark the frontier after successor dispatch is queued.  The event loop
    # captures it only after every business event at this timestamp drains.
    _observe_profile_request_completion(context, npu, batch, current_time_ms)
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
    admission_time_ms = (
        float(request.admission_time_ms) if request.admitted else None
    )
    hard_deadline_time_ms = (
        None
        if admission_time_ms is None
        or context.control.hard_ttft_ideal_multiplier is None
        else admission_time_ms
        + context.control.hard_ttft_ideal_multiplier
        * context.n_layers
        * request.per_layer_compute_ms
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
        admission_time_ms=admission_time_ms,
        hard_deadline_time_ms=hard_deadline_time_ms,
        arrival_time_ms=float(request.manifest.arrival_time_ms),
    )


def _control_path_pressure_snapshot(
    context: _Context,
    current_time_ms: float,
):
    """Read count-only QoS telemetry without consulting future event state.

    ``report_path_pressure_analysis`` exposes only outstanding logical-I/O
    counts and aggregate QoS weights.  In particular, this projection never
    reads a flow's simulator-private completion time.  The scheduler may serve
    a query from its configured TTL cache, so fresh reads and cache hits are
    distinguished instead of treating every controller evaluation as a
    hardware pressure-table read.
    """
    rows = []
    for ssu_id, disk in enumerate(context.disks):
        scheduler = disk.scheduler
        reads_before = int(scheduler.pressure_reports)
        cache_hits_before = int(getattr(scheduler, "pressure_cache_hits", 0))
        pressure = scheduler.report_path_pressure_analysis(current_time_ms)
        reads_after = int(scheduler.pressure_reports)
        cache_hits_after = int(getattr(scheduler, "pressure_cache_hits", 0))
        fresh_reads = reads_after - reads_before
        cache_hits = cache_hits_after - cache_hits_before
        if fresh_reads < 0 or cache_hits < 0 or fresh_reads + cache_hits != 1:
            raise AssertionError("one pressure query must produce one observation")
        context.cir_control_pressure_queries_by_ssu[ssu_id] += 1
        context.cir_control_pressure_reads_by_ssu[ssu_id] += fresh_reads
        context.cir_control_pressure_cache_hits_by_ssu[ssu_id] += cache_hits
        row = tuple(int(value) for value in pressure.counts)
        if len(row) != len(scheduler.paths) or any(value < 0 for value in row):
            raise AssertionError("CIR-control Path pressure shape is invalid")
        rows.append(row)
    return tuple(rows)


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
    path_pressure = _control_path_pressure_snapshot(context, current_time_ms)
    snapshot = CIRControlSnapshot(
        time_ms=current_time_ms,
        evaluation=context.control_evaluations,
        layer_jobs_since_previous=context.layer_jobs_since_control,
        num_npu=context.num_npu,
        num_ssu=context.num_ssu,
        active_requests=active_requests,
        current_path_cirs_by_ssu=current_cirs,
        trigger_reasons=tuple(sorted(reasons)),
        path_outstanding_io_counts_by_ssu=path_pressure,
        pressure_queries_cumulative_by_ssu=tuple(
            context.cir_control_pressure_queries_by_ssu
        ),
        pressure_reads_cumulative_by_ssu=tuple(
            context.cir_control_pressure_reads_by_ssu
        ),
        pressure_cache_hits_cumulative_by_ssu=tuple(
            context.cir_control_pressure_cache_hits_by_ssu
        ),
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
    if context.timeline_diagnostics:
        context.timeline_ssd_completed_blocks[flow.npu_id][flow.disk_id] += (
            flow.block_count
        )
        context.timeline_link_enqueued_gb[flow.npu_id][flow.disk_id] += flow.total_gb
        context.timeline_link_enqueued_blocks[flow.npu_id][flow.disk_id] += (
            flow.block_count
        )
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
    if context.timeline_diagnostics:
        context.timeline_link_completed_blocks[flow.npu_id][flow.disk_id] += (
            flow.block_count
        )
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


def _timeline_stationarity_invariants(context: _Context, snapshots):
    if not context.timeline_diagnostics:
        return {}
    shape_ok = all(
        len(snapshot.get("timeline", {}).get("npu_rows", ())) == context.num_npu
        and all(
            len(row) == context.num_ssu
            for field in (
                "ssd_enqueued_cumulative_gb",
                "ssd_served_cumulative_gb",
                "ssd_served_fragmented_diagnostic_cumulative_gb",
                "ssd_outstanding_gb",
                "ssd_outstanding_blocks",
                "link_enqueued_cumulative_gb",
                "link_served_cumulative_gb",
                "link_outstanding_gb",
                "link_outstanding_blocks",
                "ssd_served_awaiting_link_enqueue_gb",
                "client_unissued_gb",
                "activated_io_cumulative_gb",
                "controller_declared_remaining_gb",
                "physical_remaining_gb",
                "controller_demand_gbps",
                "physical_demand_gbps",
            )
            for row in snapshot["timeline"]["npu_ssu"][field]
        )
        for snapshot in snapshots
    )
    if not shape_ok:
        return {"timeline_snapshot_shapes": False}

    nonnegative = all(
        math.isfinite(float(value)) and float(value) >= -1e-8
        for snapshot in snapshots
        for field in (
            "ssd_enqueued_cumulative_gb",
            "ssd_served_cumulative_gb",
            "ssd_served_fragmented_diagnostic_cumulative_gb",
            "ssd_outstanding_gb",
            "ssd_outstanding_blocks",
            "link_enqueued_cumulative_gb",
            "link_served_cumulative_gb",
            "link_outstanding_gb",
            "link_outstanding_blocks",
            "ssd_served_awaiting_link_enqueue_gb",
            "client_unissued_gb",
            "activated_io_cumulative_gb",
            "controller_declared_remaining_gb",
            "physical_remaining_gb",
            "controller_demand_gbps",
            "physical_demand_gbps",
        )
        for row in snapshot["timeline"]["npu_ssu"][field]
        for value in row
    ) and all(
        row["compute_inventory_q_ms"] >= -1e-8
        and row["activated_compute_cumulative_ms"] >= -1e-8
        for snapshot in snapshots
        for row in snapshot["timeline"]["npu_rows"]
    )

    service_attribution = all(
        math.isclose(
            math.fsum(
                snapshot["timeline"]["npu_ssu"][
                    "ssd_served_cumulative_gb"
                ][npu_id][ssu_id]
                for npu_id in range(context.num_npu)
            ),
            snapshot["ssd_cumulative_served_gb_by_ssu"][ssu_id],
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        for snapshot in snapshots
        for ssu_id in range(context.num_ssu)
    ) and all(
        math.isclose(
            math.fsum(
                snapshot["timeline"]["npu_ssu"][
                    "link_served_cumulative_gb"
                ][npu_id]
            ),
            snapshot["npu_link_cumulative_served_gb_by_npu"][npu_id],
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        for snapshot in snapshots
        for npu_id in range(context.num_npu)
    )

    # Close the observer's NPU-by-SSU queues against independently maintained
    # aggregate scheduler/NPU state.  Unlike the temporal identity below,
    # these comparisons do not reuse the observer's enqueued counters.
    independent_queue_attribution = all(
        math.isclose(
            math.fsum(
                snapshot["timeline"]["npu_ssu"]["ssd_outstanding_gb"][npu][
                    ssu
                ]
                for npu in range(context.num_npu)
            ),
            snapshot["ssd_outstanding_gb_by_ssu"][ssu],
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        and sum(
            snapshot["timeline"]["npu_ssu"]["ssd_outstanding_blocks"][npu][
                ssu
            ]
            for npu in range(context.num_npu)
        )
        == snapshot["ssd_outstanding_blocks_by_ssu"][ssu]
        for snapshot in snapshots
        for ssu in range(context.num_ssu)
    ) and all(
        math.isclose(
            math.fsum(
                snapshot["timeline"]["npu_ssu"]["link_outstanding_gb"][npu]
            ),
            snapshot["npu_link_outstanding_gb_by_npu"][npu],
            rel_tol=1e-10,
            abs_tol=1e-8,
        )
        and sum(
            snapshot["timeline"]["npu_ssu"]["link_outstanding_blocks"][npu]
        )
        == snapshot["npu_link_outstanding_blocks_by_npu"][npu]
        for snapshot in snapshots
        for npu in range(context.num_npu)
    )

    # Every activated byte is in exactly one physical stage: not yet issued by
    # the client, unserved inside the SSD command/queue, already serviced but
    # awaiting non-streaming command completion, inside the NPU link, or
    # already delivered.
    io_stage_conservation = True
    remaining_work_bounds = True
    for snapshot in snapshots:
        matrix = snapshot["timeline"]["npu_ssu"]
        for npu in range(context.num_npu):
            for ssu in range(context.num_ssu):
                undelivered_activated = (
                    matrix["client_unissued_gb"][npu][ssu]
                    + matrix["ssd_outstanding_gb"][npu][ssu]
                    + matrix["ssd_served_awaiting_link_enqueue_gb"][npu][ssu]
                    + matrix["link_outstanding_gb"][npu][ssu]
                )
                io_stage_conservation &= math.isclose(
                    matrix["activated_io_cumulative_gb"][npu][ssu],
                    undelivered_activated
                    + matrix["link_served_cumulative_gb"][npu][ssu],
                    rel_tol=1e-10,
                    abs_tol=1e-8,
                )
                physical = matrix["physical_remaining_gb"][npu][ssu]
                declared = matrix["controller_declared_remaining_gb"][npu][ssu]
                remaining_work_bounds &= (
                    physical + 1e-8 >= undelivered_activated
                    and declared + 1e-8 >= physical
                )

    queue_conservation = True
    compute_inventory_conservation = True
    cumulative_monotonic = True
    for start, end in zip(snapshots, snapshots[1:]):
        start_matrix = start["timeline"]["npu_ssu"]
        end_matrix = end["timeline"]["npu_ssu"]
        for npu_id in range(context.num_npu):
            start_npu = start["timeline"]["npu_rows"][npu_id]
            end_npu = end["timeline"]["npu_rows"][npu_id]
            busy_delta = (
                end["npu_compute_cumulative_busy_ms_by_npu"][npu_id]
                - start["npu_compute_cumulative_busy_ms_by_npu"][npu_id]
            )
            activated_delta = (
                end_npu["activated_compute_cumulative_ms"]
                - start_npu["activated_compute_cumulative_ms"]
            )
            expected_busy = (
                activated_delta
                + start_npu["compute_inventory_q_ms"]
                - end_npu["compute_inventory_q_ms"]
            )
            compute_inventory_conservation &= math.isclose(
                busy_delta,
                expected_busy,
                rel_tol=1e-10,
                abs_tol=1e-7,
            )
            for ssu_id in range(context.num_ssu):
                for prefix in ("ssd", "link"):
                    enqueued = f"{prefix}_enqueued_cumulative_gb"
                    served = f"{prefix}_served_cumulative_gb"
                    outstanding = f"{prefix}_outstanding_gb"
                    enqueue_delta = (
                        end_matrix[enqueued][npu_id][ssu_id]
                        - start_matrix[enqueued][npu_id][ssu_id]
                    )
                    served_delta = (
                        end_matrix[served][npu_id][ssu_id]
                        - start_matrix[served][npu_id][ssu_id]
                    )
                    expected_outstanding = (
                        start_matrix[outstanding][npu_id][ssu_id]
                        + enqueue_delta
                        - served_delta
                    )
                    queue_conservation &= math.isclose(
                        end_matrix[outstanding][npu_id][ssu_id],
                        expected_outstanding,
                        rel_tol=1e-10,
                        abs_tol=1e-8,
                    )
                    cumulative_monotonic &= (
                        enqueue_delta >= -1e-8 and served_delta >= -1e-8
                    )
                fragmented_service_delta = (
                    end_matrix[
                        "ssd_served_fragmented_diagnostic_cumulative_gb"
                    ][npu_id][ssu_id]
                    - start_matrix[
                        "ssd_served_fragmented_diagnostic_cumulative_gb"
                    ][npu_id][ssu_id]
                )
                cumulative_monotonic &= fragmented_service_delta >= -1e-8
    dispatch_replay = all(
        record.get("prediction_matches_actual") is True
        and record.get("expected_path_id") == record.get("actual_path_id")
        for record in context.timeline_dispatch_probe_records
    )
    return {
        "timeline_snapshot_shapes": shape_ok,
        "timeline_values_nonnegative": nonnegative,
        "timeline_service_attribution": service_attribution,
        "timeline_independent_queue_attribution": (
            independent_queue_attribution
        ),
        "timeline_io_stage_conservation": io_stage_conservation,
        "timeline_remaining_work_bounds": remaining_work_bounds,
        "timeline_cumulative_monotonic": cumulative_monotonic,
        "timeline_ssd_link_queue_conservation": queue_conservation,
        "timeline_compute_inventory_conservation": (
            compute_inventory_conservation
        ),
        "timeline_dispatch_replay_exact": dispatch_replay,
    }


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
                math.fsum(
                    row[ssu_id]
                    for row in context.measurement_npu_ssu_served_start_gb
                )
                for ssu_id in range(context.num_ssu)
            ],
        )
        and _vectors_close(
            last["ssd_cumulative_served_gb_by_ssu"],
            [
                math.fsum(
                    row[ssu_id]
                    for row in context.measurement_npu_ssu_served_end_gb
                )
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
                math.fsum(row[ssu_id] for row in npu_ssu_served_gb)
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
        **_timeline_stationarity_invariants(context, snapshots),
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


def _timeline_state_durations(
    context: _Context,
    window_start_ms: float,
    window_end_ms: float,
    block_bounds,
):
    """Exact per-NPU compute/barrier/complement durations, including carry-in."""

    duration_ms = window_end_ms - window_start_ms
    compute_ms = [0.0] * context.num_npu
    barrier_ms = [0.0] * context.num_npu
    block_compute_ms = [
        [0.0] * context.num_npu for _ in range(len(block_bounds))
    ]
    block_barrier_ms = [
        [0.0] * context.num_npu for _ in range(len(block_bounds))
    ]
    intervals_complete = True
    for batch in context.microbatches:
        barrier_start_ms = batch.admission_time_ms
        for metric in batch.layer_metrics:
            if not (
                math.isfinite(barrier_start_ms)
                and math.isfinite(metric.compute_start_ms)
                and math.isfinite(metric.compute_end_ms)
            ):
                # An incomplete interval wholly outside the window is harmless;
                # any potentially overlapping interval invalidates attribution.
                if (
                    math.isfinite(barrier_start_ms)
                    and barrier_start_ms < window_end_ms
                ):
                    intervals_complete = False
                break
            barrier_ms[batch.npu_id] += _window_overlap_ms(
                barrier_start_ms,
                metric.compute_start_ms,
                window_start_ms,
                window_end_ms,
            )
            compute_ms[batch.npu_id] += _window_overlap_ms(
                metric.compute_start_ms,
                metric.compute_end_ms,
                window_start_ms,
                window_end_ms,
            )
            for block, (block_start_ms, block_end_ms, _) in enumerate(
                block_bounds
            ):
                block_barrier_ms[block][batch.npu_id] += _window_overlap_ms(
                    barrier_start_ms,
                    metric.compute_start_ms,
                    block_start_ms,
                    block_end_ms,
                )
                block_compute_ms[block][batch.npu_id] += _window_overlap_ms(
                    metric.compute_start_ms,
                    metric.compute_end_ms,
                    block_start_ms,
                    block_end_ms,
                )
            barrier_start_ms = metric.compute_end_ms

    rows = []
    partition_ok = intervals_complete
    for npu_id in range(context.num_npu):
        classified_ms = compute_ms[npu_id] + barrier_ms[npu_id]
        partition_ok &= classified_ms <= duration_ms + 1e-7
        other_ms = max(0.0, duration_ms - classified_ms)
        rows.append(
            {
                "npu_id": npu_id,
                "compute_ms": compute_ms[npu_id],
                "io_barrier_ms": barrier_ms[npu_id],
                "other_ms": other_ms,
                "measurement_ms": duration_ms,
                "compute_fraction": compute_ms[npu_id] / duration_ms,
                "io_barrier_fraction": barrier_ms[npu_id] / duration_ms,
                "other_fraction": other_ms / duration_ms,
            }
        )
    block_rows = []
    for block, (start_ms, end_ms, block_duration_ms) in enumerate(block_bounds):
        other_ms = []
        for npu_id in range(context.num_npu):
            classified_ms = (
                block_compute_ms[block][npu_id]
                + block_barrier_ms[block][npu_id]
            )
            partition_ok &= classified_ms <= block_duration_ms + 1e-7
            other_ms.append(max(0.0, block_duration_ms - classified_ms))
        block_rows.append(
            {
                "block": block,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": block_duration_ms,
                "compute_ms_by_npu": block_compute_ms[block],
                "io_barrier_ms_by_npu": block_barrier_ms[block],
                "other_ms_by_npu": other_ms,
            }
        )
    return rows, block_rows, partition_ok


def _timeline_carry_in_batch_rows(
    context: _Context,
    window_start_ms: float,
):
    """Export only batches active at the left-limit measurement boundary.

    Full layer intervals for admissions inside the measurement window already
    live in ``request_rows``.  A NPU executes batches serially, so at most one
    earlier-admitted batch can be active at that boundary.  A batch whose
    completion event is exactly at the boundary is included because snapshots
    are captured before same-time workload events; its half-open interval
    contribution is still zero.  Exporting just that carry-in batch avoids
    duplicating all interval history while making the state-duration totals
    independently reproducible.
    """

    batches = [
        batch
        for batch in context.microbatches
        if batch.admission_time_ms < window_start_ms <= batch.completion_time_ms
    ]
    rows = [
        {
            "batch_id": int(batch.batch_id),
            "request_ids": [int(value) for value in batch.member_request_ids],
            "npu_id": int(batch.npu_id),
            "admission_time_ms": float(batch.admission_time_ms),
            "completion_time_ms": float(batch.completion_time_ms),
            "layer_count": len(batch.layer_metrics),
            "per_layer_compute_ms": (
                float(context.requests[batch.member_request_ids[0]].per_layer_compute_ms)
                if len(batch.member_request_ids) == 1
                and batch.member_request_ids[0] in context.requests
                else None
            ),
            "ideal_compute_ms": (
                float(
                    context.n_layers
                    * context.requests[
                        batch.member_request_ids[0]
                    ].per_layer_compute_ms
                )
                if len(batch.member_request_ids) == 1
                and batch.member_request_ids[0] in context.requests
                else None
            ),
            "io_ready_time_ms": [
                float(metric.io_ready_time_ms) for metric in batch.layer_metrics
            ],
            "compute_start_ms": [
                float(metric.compute_start_ms) for metric in batch.layer_metrics
            ],
            "compute_end_ms": [
                float(metric.compute_end_ms) for metric in batch.layer_metrics
            ],
            "compute_duration_ms": [
                float(metric.compute_duration_ms) for metric in batch.layer_metrics
            ],
            "io_barrier_wait_ms": [
                float(metric.io_barrier_wait_ms) for metric in batch.layer_metrics
            ],
        }
        for batch in batches
    ]

    expected_ids = [batch.batch_id for batch in batches]
    exported_ids = [row["batch_id"] for row in rows]
    definition_exact = exported_ids == expected_ids and all(
        row["admission_time_ms"] < window_start_ms <= row["completion_time_ms"]
        for row in rows
    )
    npu_ids = [row["npu_id"] for row in rows]
    unique_per_npu = len(npu_ids) == len(set(npu_ids))
    batch_size_one = all(len(batch.member_request_ids) == 1 for batch in batches)
    layer_shape_exact = all(
        row["layer_count"] == context.n_layers
        and all(
            len(row[field]) == context.n_layers
            for field in (
                "io_ready_time_ms",
                "compute_start_ms",
                "compute_end_ms",
                "compute_duration_ms",
                "io_barrier_wait_ms",
            )
        )
        for row in rows
    )
    request_identity_exact = all(
        tuple(row["request_ids"]) == batch.member_request_ids
        and all(
            request_id in context.requests
            and context.requests[request_id].batch_id == batch.batch_id
            and context.requests[request_id].manifest.npu_id == batch.npu_id
            and math.isclose(
                context.requests[request_id].admission_time_ms,
                batch.admission_time_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                context.requests[request_id].completion_time_ms,
                batch.completion_time_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for request_id in batch.member_request_ids
        )
        for row, batch in zip(rows, batches)
    )
    interval_closure = True
    compute_budget_exact = True
    for batch in batches:
        per_layer_compute_ms = (
            float(context.requests[batch.member_request_ids[0]].per_layer_compute_ms)
            if len(batch.member_request_ids) == 1
            and batch.member_request_ids[0] in context.requests
            else math.nan
        )
        barrier_start_ms = float(batch.admission_time_ms)
        for expected_layer, metric in enumerate(batch.layer_metrics):
            values = (
                metric.io_ready_time_ms,
                metric.compute_start_ms,
                metric.compute_end_ms,
                metric.compute_duration_ms,
                metric.io_barrier_wait_ms,
            )
            interval_closure &= metric.layer == expected_layer and all(
                math.isfinite(value) for value in values
            )
            if not all(math.isfinite(value) for value in values):
                break
            expected_compute_start_ms = max(
                float(metric.io_ready_time_ms),
                barrier_start_ms,
            )
            interval_closure &= (
                metric.compute_start_ms == expected_compute_start_ms
                and metric.compute_end_ms >= metric.compute_start_ms
                and metric.io_barrier_wait_ms
                == max(0.0, metric.compute_start_ms - barrier_start_ms)
                and math.isclose(
                    metric.compute_end_ms - metric.compute_start_ms,
                    metric.compute_duration_ms,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            compute_budget_exact &= metric.compute_duration_ms == per_layer_compute_ms
            barrier_start_ms = metric.compute_end_ms
        interval_closure &= barrier_start_ms == batch.completion_time_ms
        compute_budget_exact &= (
            math.isfinite(per_layer_compute_ms)
            and math.isclose(
                math.fsum(
                    metric.compute_duration_ms for metric in batch.layer_metrics
                ),
                context.n_layers * per_layer_compute_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return rows, {
        "timeline_carry_in_definition_exact": definition_exact,
        "timeline_carry_in_unique_per_npu": unique_per_npu,
        "timeline_carry_in_batch_size_one": batch_size_one,
        "timeline_carry_in_layer_shape_exact": layer_shape_exact,
        "timeline_carry_in_request_identity_exact": request_identity_exact,
        "timeline_carry_in_compute_budget_exact": compute_budget_exact,
        "timeline_carry_in_interval_closure": interval_closure,
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
        or context.measurement_fragmented_npu_ssu_served_start_gb is None
        or context.measurement_fragmented_npu_ssu_served_end_gb is None
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
    fragmented_npu_ssu_served_gb = [
        [
            max(0.0, end_gb - start_gb)
            for start_gb, end_gb in zip(start_row, end_row)
        ]
        for start_row, end_row in zip(
            context.measurement_fragmented_npu_ssu_served_start_gb,
            context.measurement_fragmented_npu_ssu_served_end_gb,
        )
    ]
    ssd_service_attribution_residual_gb_by_ssu = [
        math.fsum(row[ssu_id] for row in npu_ssu_served_gb)
        - disk_served_gb_by_ssu[ssu_id]
        for ssu_id in range(context.num_ssu)
    ]
    fragmented_ssd_service_minus_stable_gb_by_ssu = [
        math.fsum(row[ssu_id] for row in fragmented_npu_ssu_served_gb)
        - math.fsum(row[ssu_id] for row in npu_ssu_served_gb)
        for ssu_id in range(context.num_ssu)
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
    (
        timeline_state_durations,
        timeline_block_state_durations,
        timeline_state_partition_ok,
    ) = (
        _timeline_state_durations(
            context, window_start_ms, window_end_ms, block_bounds
        )
        if context.timeline_diagnostics
        else ([], [], True)
    )
    (
        timeline_carry_in_batches,
        timeline_carry_in_invariants,
    ) = (
        _timeline_carry_in_batch_rows(context, window_start_ms)
        if context.timeline_diagnostics
        else ([], {})
    )

    request_rows = []
    outcomes_by_npu = [[] for _ in range(context.num_npu)]
    for request_id in sorted(context.measurement_request_ids):
        request = context.requests[request_id]
        batch = context.microbatches[request.batch_id]
        ideal_ttft_ms = context.n_layers * request.per_layer_compute_ms
        ttft_ms = request.completion_time_ms - request.admission_time_ms
        slo_met = ttft_ms <= config.slo_alpha * ideal_ttft_ms + _EPS
        outcomes_by_npu[request.manifest.npu_id].append(slo_met)
        row = {
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
        if context.timeline_diagnostics:
            row["io_barrier_ms"] = batch.io_barrier_wait_ms
            row["ttft_accounting_error_ms"] = (
                ttft_ms - ideal_ttft_ms - batch.io_barrier_wait_ms
            )
            per_layer_work = [0.0] * context.num_ssu
            for ssu_id, size_gb in _manifest_layer(request.manifest, 0):
                per_layer_work[ssu_id] += size_gb
            row["timeline_layers"] = {
                "io_start_time_ms": [
                    metric.io_start_time_ms for metric in batch.layer_metrics
                ],
                "io_ready_time_ms": [
                    metric.io_ready_time_ms for metric in batch.layer_metrics
                ],
                "compute_start_ms": [
                    metric.compute_start_ms for metric in batch.layer_metrics
                ],
                "compute_end_ms": [
                    metric.compute_end_ms for metric in batch.layer_metrics
                ],
                "io_barrier_wait_ms": [
                    metric.io_barrier_wait_ms for metric in batch.layer_metrics
                ],
                "per_layer_work_gb_by_ssu": per_layer_work,
            }
        request_rows.append(row)

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
    timeline_ssd_accounting_maxima = {}
    if context.timeline_diagnostics:
        residual_fields = (
            "stable_service_minus_busy_counter_gb",
            "fragmented_service_minus_stable_gb",
            "physical_queue_minus_scheduler_gb",
            "enqueue_minus_service_minus_physical_queue_gb",
            "counter_queue_minus_physical_queue_gb",
            "fragmented_counter_queue_minus_physical_queue_gb",
            "maximum_abs_npu_queue_identity_residual_gb",
        )
        for field in residual_fields:
            candidates = [
                (
                    snapshot,
                    row,
                    float(row[field]),
                )
                for snapshot in stationarity["snapshots"]
                for row in snapshot["timeline"][
                    "ssd_accounting_residuals_by_ssu"
                ]
            ]
            snapshot, row, residual = max(
                candidates,
                key=lambda candidate: abs(candidate[2]),
            )
            timeline_ssd_accounting_maxima[field] = {
                "signed_gb": residual,
                "signed_decimal_bytes": residual * 1e9,
                "absolute_gb": abs(residual),
                "absolute_decimal_bytes": abs(residual) * 1e9,
                "boundary": int(snapshot["boundary"]),
                "elapsed_ms": float(snapshot["time_ms"] - window_start_ms),
                "ssu_id": int(row["ssu_id"]),
            }
            if field == "maximum_abs_npu_queue_identity_residual_gb":
                timeline_ssd_accounting_maxima[field]["npu_id"] = int(
                    row["maximum_abs_npu_queue_identity_residual_npu_id"]
                )
        block_residual_fields = (
            "physical_queue_block_minus_scheduler_blocks",
            "counter_queue_block_minus_physical_blocks",
        )
        for field in block_residual_fields:
            candidates = [
                (snapshot, row, int(row[field]))
                for snapshot in stationarity["snapshots"]
                for row in snapshot["timeline"][
                    "ssd_accounting_residuals_by_ssu"
                ]
            ]
            snapshot, row, residual = max(
                candidates,
                key=lambda candidate: abs(candidate[2]),
            )
            timeline_ssd_accounting_maxima[field] = {
                "signed_blocks": residual,
                "absolute_blocks": abs(residual),
                "boundary": int(snapshot["boundary"]),
                "elapsed_ms": float(snapshot["time_ms"] - window_start_ms),
                "ssu_id": int(row["ssu_id"]),
            }
    measurement_ssd_accounting_maxima = {}
    for name, residuals in (
        (
            "stable_service_minus_busy_counter",
            ssd_service_attribution_residual_gb_by_ssu,
        ),
        (
            "fragmented_service_minus_stable",
            fragmented_ssd_service_minus_stable_gb_by_ssu,
        ),
    ):
        ssu_id = max(
            range(context.num_ssu),
            key=lambda index: abs(residuals[index]),
        )
        residual = float(residuals[ssu_id])
        measurement_ssd_accounting_maxima[name] = {
            "signed_gb": residual,
            "signed_decimal_bytes": residual * 1e9,
            "absolute_gb": abs(residual),
            "absolute_decimal_bytes": abs(residual) * 1e9,
            "ssu_id": int(ssu_id),
        }
    measurement_ssd_accounting_residuals = {
        "schema": "steady_ssd_accounting_residuals_v1",
        "service_absolute_tolerance_gb": 1e-8,
        "block_tolerance": 0,
        "decimal_bytes_per_gb": 1e9,
        "stable_service_minus_busy_counter_gb_by_ssu": list(
            ssd_service_attribution_residual_gb_by_ssu
        ),
        "stable_service_minus_busy_counter_decimal_bytes_by_ssu": [
            residual * 1e9
            for residual in ssd_service_attribution_residual_gb_by_ssu
        ],
        "fragmented_service_minus_stable_gb_by_ssu": list(
            fragmented_ssd_service_minus_stable_gb_by_ssu
        ),
        "fragmented_service_minus_stable_decimal_bytes_by_ssu": [
            residual * 1e9
            for residual in fragmented_ssd_service_minus_stable_gb_by_ssu
        ],
        "busy_time_compensation_ms_by_ssu_at_stop": [
            float(disk.busy_time_compensation) for disk in context.disks
        ],
        "measurement_maxima": measurement_ssd_accounting_maxima,
        "timeline_maxima": timeline_ssd_accounting_maxima,
    }
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
            math.isclose(residual, 0.0, rel_tol=0.0, abs_tol=1e-8)
            for residual in ssd_service_attribution_residual_gb_by_ssu
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
        "timeline_dispatch_probe_nonempty": (
            not context.timeline_diagnostics
            or config.timeline_dispatch_probe_ms == 0.0
            or config.timeline_dispatch_probe_limit == 0
            or bool(context.timeline_dispatch_probe_records)
        ),
        "timeline_route_probe_nonempty": (
            not context.timeline_diagnostics
            or config.timeline_dispatch_probe_ms == 0.0
            or config.timeline_dispatch_probe_limit == 0
            or bool(context.timeline_route_probe_records)
        ),
        **stationarity["invariants"],
    }
    if context.timeline_diagnostics:
        invariants.update(
            {
                "request_ttft_compute_barrier_decomposition": all(
                    abs(row["ttft_accounting_error_ms"]) <= 1e-7
                    for row in request_rows
                ),
                "timeline_state_duration_partition": (
                    timeline_state_partition_ok
                    and all(
                        math.isclose(
                            row["compute_ms"]
                            + row["io_barrier_ms"]
                            + row["other_ms"],
                            duration_ms,
                            rel_tol=0.0,
                            abs_tol=1e-7,
                        )
                        for row in timeline_state_durations
                    )
                ),
                "timeline_state_compute_matches_utilization": all(
                    math.isclose(
                        row["compute_ms"],
                        compute_ms_by_npu[row["npu_id"]],
                        rel_tol=0.0,
                        abs_tol=1e-7,
                    )
                    for row in timeline_state_durations
                ),
                "timeline_block_state_duration_partition": all(
                    math.isclose(
                        compute_ms + barrier_ms + other_ms,
                        block["duration_ms"],
                        rel_tol=0.0,
                        abs_tol=1e-7,
                    )
                    for block in timeline_block_state_durations
                    for compute_ms, barrier_ms, other_ms in zip(
                        block["compute_ms_by_npu"],
                        block["io_barrier_ms_by_npu"],
                        block["other_ms_by_npu"],
                    )
                ),
                "timeline_block_state_compute_matches_stationarity": all(
                    math.isclose(
                        compute_ms,
                        block_compute_ms_by_npu[block["block"]][npu_id],
                        rel_tol=0.0,
                        abs_tol=1e-7,
                    )
                    for block in timeline_block_state_durations
                    for npu_id, compute_ms in enumerate(
                        block["compute_ms_by_npu"]
                    )
                ),
                **timeline_carry_in_invariants,
            }
        )
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
            "measurement_ssd_accounting_residuals": (
                measurement_ssd_accounting_residuals
            ),
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
        "timeline_diagnostics_enabled": context.timeline_diagnostics,
        "timeline_demand_semantics": {
            "controller_demand": (
                "remaining full-manifest bytes for every not-ready layer divided "
                "by remaining not-ready-layer compute; this is the Adaptive input"
            ),
            "physical_demand": (
                "bytes not yet delivered by the NPU link divided by the same "
                "remaining compute; diagnostic only and never a controller input"
            ),
            "installed_cir": (
                "dedicated-Path arbitration guarantee, not realized service; "
                "null for baseline and layer-once"
            ),
            "realized_service": (
                "difference of settle-fragmentation-independent cumulative "
                "NPU-by-SSU service bytes between two left-limit boundaries, "
                "divided by interval duration"
            ),
            "ssd_served_awaiting_link_enqueue": (
                "bytes already serviced within the active non-streaming SSD "
                "command but not visible to the NPU link until command completion"
            ),
        },
        "timeline_ssd_accounting_semantics": {
            "schema": "steady_timeline_boundary_v3",
            "ssd_served_cumulative_gb": (
                "compensated whole-command completion totals per NPU and SSU, "
                "plus the at-most-one active command prefix reconstructed from "
                "its immutable activation time and total_gb"
            ),
            "ssd_outstanding_gb": (
                "direct math.fsum enumeration of every physical pending command "
                "total_gb plus the active command's immutable projected remainder; "
                "never derived by subtracting cumulative counters"
            ),
            "ssd_outstanding_blocks": (
                "exact integer enumeration of every pending and active command's "
                "block_count"
            ),
            "busy_service_reference": (
                "independent compensated SSD busy-time counter multiplied by the "
                "physical SSD bandwidth"
            ),
            "fragmented_service_diagnostic": (
                "historical observer-fragmentation-dependent per-settle service "
                "accumulation, exposed as "
                "ssd_served_fragmented_diagnostic_cumulative_gb and retained "
                "only for accounting residuals; it is not a scientific output "
                "or actual-service/conservation input, and is checked only for "
                "shape, finiteness, nonnegativity, and monotonicity"
            ),
            "queue_identity": (
                "compensated enqueued_gb minus stable cumulative service minus "
                "direct physical outstanding_gb"
            ),
        },
        "timeline_adaptive_deadline_input": bool(
            context.control is not None
            and context.control.hard_ttft_ideal_multiplier is not None
        ),
        "timeline_adaptive_deadline_note": (
            "this flag records whether the public CIRControlSnapshot carries "
            "an explicitly configured admission-based hard deadline; it does "
            "not prove that a particular callback consumes that field, and "
            "the alpha1.5/alpha2 timeline slack remains diagnostic only"
        ),
        "timeline_state_durations_ms_by_npu": (
            timeline_state_durations if context.timeline_diagnostics else []
        ),
        "timeline_block_state_durations_ms": (
            timeline_block_state_durations
            if context.timeline_diagnostics
            else []
        ),
        "timeline_carry_in_batches_schema": "steady_timeline_carry_in_batch_v1",
        "timeline_carry_in_batches": (
            timeline_carry_in_batches if context.timeline_diagnostics else []
        ),
        "timeline_carry_in_batch_semantics": (
            "exact batches satisfying admission_time_ms < measurement_start_ms "
            "<= completion_time_ms under the before-same-time-events left-limit "
            "snapshot order; equality contributes zero elapsed time; at most "
            "one per NPU; half-open interval intersections use "
            "max(0, min(end, window_end) - "
            "max(start, window_start)); admissions inside the measurement "
            "window remain exclusively in request_rows"
        ),
        "timeline_state_duration_semantics": (
            "exact intersections of every microbatch layer compute interval and "
            "its preceding I/O-barrier interval with the measurement window; "
            "includes carry-in, and other is the exact window complement"
        ),
        "timeline_control_trigger_records": (
            context.timeline_control_triggers
            if context.timeline_diagnostics
            else []
        ),
        "timeline_route_probe_records": (
            context.timeline_route_probe_records
            if context.timeline_diagnostics
            else []
        ),
        "timeline_dispatch_probe_ms": (
            config.timeline_dispatch_probe_ms
            if context.timeline_diagnostics
            else 0.0
        ),
        "timeline_dispatch_probe_limit": (
            config.timeline_dispatch_probe_limit
            if context.timeline_diagnostics
            else 0
        ),
        "timeline_dispatch_probe_truncated": (
            context.timeline_diagnostics
            and len(context.timeline_dispatch_probe_records)
            >= config.timeline_dispatch_probe_limit
            and config.timeline_dispatch_probe_limit > 0
        ),
        "timeline_route_probe_truncated": (
            context.timeline_diagnostics
            and len(context.timeline_route_probe_records)
            >= config.timeline_dispatch_probe_limit
            and config.timeline_dispatch_probe_limit > 0
        ),
        "timeline_dispatch_probe_records": (
            context.timeline_dispatch_probe_records
            if context.timeline_diagnostics
            else []
        ),
        "measurement_stationarity_boundary_count": len(stationarity["snapshots"]),
        "measurement_stationarity_boundaries": stationarity["snapshots"],
        **(
            {
                "profile_cycle_frontier_trend": {
                    "schema": "steady_profile_cycle_frontier_series_v1",
                    "probe_npu_ids": list(
                        context.profile_cycle_probe_npu_ids
                    ),
                    "period_generations": context.profile_cycle_period,
                    "generation_source": "request.load['generation']",
                    "request_id_layout_assumed": False,
                    "boundary_definition": (
                        "generation g closes a configured profile cycle exactly when "
                        "(g + 1) % period_generations == 0; a frontier is "
                        "recorded after every probe NPU has completed at "
                        "least g"
                    ),
                    "measurement_membership": (
                        "frontier-triggering completion is processed while the "
                        "half-open measurement window is open"
                    ),
                    "same_timestamp_semantics": (
                        "the triggering completion queues a marker only; the "
                        "pure-read snapshot is captured after the complete "
                        "transitive closure of business events at that timestamp "
                        "has drained, immediately before simulated time advances"
                    ),
                    "physical_io_outstanding_semantics": (
                        "client-unissued plus physical SSD remaining plus the "
                        "serviced prefix of each active non-streaming SSD "
                        "command awaiting link enqueue plus physical NPU-link "
                        "remaining; a fully delivered/ready prefetched Layer-0 "
                        "has zero outstanding bytes and remains visible in "
                        "npu_state.prefetched_layer0 or, after dispatch consumes "
                        "that descriptor, in npu_state.active_generations"
                    ),
                    "ssd_submitted_cumulative_semantics": (
                        "compensated cumulative GB submitted since simulation "
                        "start while the probe is enabled; adjacent frontier "
                        "differences expose actual per-NPU/per-SSU placement work"
                    ),
                    "interpretation": (
                        "profile-cycle frontiers are long-term trend/drift "
                        "samples, not proof of a repeating simulator state; "
                        "request-ID-dependent placement is exposed and checked "
                        "separately in every snapshot"
                    ),
                    "snapshot_count": len(
                        context.profile_cycle_frontier_snapshots
                    ),
                    "snapshots": context.profile_cycle_frontier_snapshots,
                    "all_snapshots_profile_period_repeats_exact_placement_forcing": (
                        bool(context.profile_cycle_frontier_snapshots)
                        and all(
                            snapshot["profile_and_placement_forcing"][
                                "profile_period_also_repeats_exact_placement_forcing"
                            ]
                            for snapshot in context.profile_cycle_frontier_snapshots
                        )
                    ),
                }
            }
            if context.profile_cycle_probe_enabled
            else {}
        ),
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
        "measurement_fragmented_npu_ssu_ssd_served_gb": (
            fragmented_npu_ssu_served_gb
        ),
        "measurement_ssd_accounting_residuals": (
            measurement_ssd_accounting_residuals
        ),
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
        **_cir_control_pressure_metrics(context),
        "control_evaluations": context.control_evaluations,
        "control_min_interval_ms": (
            context.control.min_interval_ms if context.control is not None else None
        ),
        "control_hard_ttft_ideal_multiplier": (
            context.control.hard_ttft_ideal_multiplier
            if context.control is not None
            else None
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
        **_cir_control_pressure_metrics(context),
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
        "control_hard_ttft_ideal_multiplier": (
            context.control.hard_ttft_ideal_multiplier
            if context.control is not None
            else None
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
    if steady_state is not None:
        profile_cycle_probe_ids = tuple(
            steady_state.profile_cycle_probe_npu_ids
        )
        profile_cycle_period_value = steady_state.profile_cycle_period
        profile_cycle_period_is_int = (
            not isinstance(profile_cycle_period_value, bool)
            and isinstance(profile_cycle_period_value, (int, np.integer))
        )
        profile_cycle_probe_pair_valid = (
            (not profile_cycle_probe_ids and profile_cycle_period_value == 0)
            or (
                bool(profile_cycle_probe_ids)
                and profile_cycle_period_is_int
                and profile_cycle_period_value > 0
            )
        )
        if (
            batch_size != 1
            or steady_state.warmup_requests_per_npu <= 0
            or steady_state.settle_ms < 0.0
            or steady_state.measurement_ms <= 0.0
            or steady_state.slo_alpha <= 0.0
            or steady_state.block_ms <= 0.0
            or type(steady_state.timeline_diagnostics) is not bool
            or not math.isfinite(steady_state.timeline_dispatch_probe_ms)
            or steady_state.timeline_dispatch_probe_ms < 0.0
            or isinstance(steady_state.timeline_dispatch_probe_limit, bool)
            or steady_state.timeline_dispatch_probe_limit < 0
            or not profile_cycle_period_is_int
            or not profile_cycle_probe_pair_valid
            or len(set(profile_cycle_probe_ids)) != len(profile_cycle_probe_ids)
            or any(
                isinstance(npu_id, bool)
                or not isinstance(npu_id, (int, np.integer))
                or npu_id < 0
                or npu_id >= num_npu
                for npu_id in profile_cycle_probe_ids
            )
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
    if steady_state is not None and steady_state.profile_cycle_probe_npu_ids:
        generation_sequences = []
        for probe_npu_id in steady_state.profile_cycle_probe_npu_ids:
            ordered = sorted(
                (
                    request
                    for request in requests
                    if request.npu_id == probe_npu_id
                ),
                key=lambda request: (
                    request.arrival_time_ms,
                    request.request_id,
                ),
            )
            if not ordered:
                raise ValueError("profile-cycle trend probe NPU has no requests")
            generations = []
            for request in ordered:
                generation = request.load.get("generation")
                if (
                    isinstance(generation, bool)
                    or not isinstance(generation, (int, np.integer))
                    or generation < 0
                ):
                    raise ValueError(
                        "profile-cycle trend probe requests require nonnegative "
                        "integer load['generation'] metadata"
                    )
                generations.append(int(generation))
            if any(
                right != left + 1
                for left, right in zip(generations, generations[1:])
            ):
                raise ValueError(
                    "profile-cycle trend probe generations must be consecutive "
                    "in per-NPU arrival order"
                )
            generation_sequences.append(tuple(generations))
        if len(set(generation_sequences)) != 1:
            raise ValueError(
                "profile-cycle trend probe NPUs must share one generation sequence"
            )

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
                timeline_probe = _prepare_timeline_dispatch_probe(
                    context, scheduler, current_time_ms
                )
                dispatched = scheduler.dispatch(current_time_ms, context.event_heap)
                _finish_timeline_dispatch_probe(
                    context, scheduler, timeline_probe, dispatched
                )
        else:
            raise RuntimeError(f"unknown event type: {event_type}")

        # Frontier observation is not an event.  Drain the entire timestamp's
        # business-event closure first so equal-time heap tie-break order
        # cannot manufacture a per-NPU phase asymmetry in the snapshot.
        if (
            context.profile_cycle_pending_frontiers
            and (
                not context.event_heap
                or abs(context.event_heap[0][0] - current_time_ms) > _EPS
            )
        ):
            _flush_profile_cycle_frontiers(context, current_time_ms)

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
