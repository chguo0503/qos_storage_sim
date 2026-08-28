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
from continuous_batch_control import allocate_grants


REQUEST_ARRIVAL = -1
BATCH_DISPATCH = -0.5
CIR_CONTROL = 2.5
COMPUTE_SCHEDULE = 2.75
CAUSAL_LAYER_CONTROL = 2.875
_EPS = sim._EPS

EXECUTION_MODEL = "full_prefill_layer_synchronous_microbatch_v1"
BATCH_COMPUTE_MODEL = "sum_member_singleton_layer_ms"
PARTIAL_BATCH_POLICY = "wait_for_full_then_drain_final_partial"
PREFETCH_POLICY = "next_layer_at_batch_compute_start"


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


CIRControlCallback = Callable[
    [CIRControlSnapshot], Optional[CIRControlDecision]
]


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

    def __init__(
        self,
        every_layers=None,
        callback=None,
        *,
        interval_ms=None,
        on_batch_boundary=True,
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
        object.__setattr__(
            self, "every_layers", int(every_layers) if layer_mode else None
        )
        object.__setattr__(self, "callback", callback)
        object.__setattr__(
            self, "interval_ms", float(interval_ms) if interval_mode else None
        )
        object.__setattr__(self, "on_batch_boundary", bool(on_batch_boundary))


@dataclass(frozen=True)
class CausalLayerObservation:
    """Only facts available after one microbatch layer has completed I/O."""

    batch_id: int
    npu_id: int
    observed_layer: int
    observed_work_gb_by_ssu: tuple[float, ...]
    compute_budget_ms: float


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
        work = np.zeros((snapshot.num_npu, snapshot.num_ssu), dtype=float)
        compute_s = np.zeros(snapshot.num_npu, dtype=float)
        for request in snapshot.active_requests:
            horizon = min(request.remaining_layers, self.horizon_layers)
            if horizon <= 0:
                continue
            work[request.npu_id] += horizon * np.asarray(
                request.next_layer_work_gb_by_ssu, dtype=float
            )
            compute_s[request.npu_id] += (
                horizon * request.per_layer_compute_ms / 1000.0
            )

        demand = np.zeros_like(work)
        active = compute_s > 0.0
        demand[active] = work[active] / compute_s[active, None]
        grants = allocate_grants(
            demand,
            ssd_caps=self.ssd_cap_gbps,
            npu_caps=self.npu_cap_gbps,
        )
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        cirs_by_ssu = []
        for ssu_id in range(snapshot.num_ssu):
            cirs = [0.0] * path_count
            for npu_id, path_id in enumerate(self.path_by_npu):
                cirs[path_id] = grants[npu_id][ssu_id]
            cirs_by_ssu.append(tuple(cirs))
        return CIRControlDecision(tuple(cirs_by_ssu))


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

        demand_rows = []
        for observation in warm:
            compute_s = observation.compute_budget_ms / 1000.0
            demand = tuple(
                work_gb / compute_s
                for work_gb in observation.observed_work_gb_by_ssu
            )
            demand_rows.append(demand)
        cold_reserve = self.cold_path_cir_gbps if cold_count else 0.0
        grants = allocate_grants(
            demand_rows,
            ssd_caps=self.ssd_cap_gbps - cold_reserve,
            npu_caps=self.npu_cap_gbps,
        )
        cirs_by_ssu = []
        for ssu_id in range(snapshot.num_ssu):
            cirs = [0.0] * self.path_count
            for row_id, observation in enumerate(warm):
                cirs[self.path_by_npu[observation.npu_id]] = grants[row_id][ssu_id]
            if cold_count:
                cirs[self.cold_path_id] = cold_reserve
            cirs_by_ssu.append(tuple(cirs))
        return CIRControlDecision(tuple(cirs_by_ssu))


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
    pending_blocks: list[int] = field(default_factory=list)
    io_ready_time_ms: list[float] = field(default_factory=list)
    completed_gb_by_layer_ssu: list[list[float]] = field(default_factory=list)
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
    return request.placement_groups[
        0 if len(request.placement_groups) == 1 else layer
    ]


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
        client_io_config: sim.ClientIOConfig,
        control: Optional[CIRControlConfig],
        causal_control: Optional[CausalLayerControlConfig],
        disk_bw_gbps: float,
        npu_bw_gbps: float,
        submit_order_seed: int,
    ):
        self.num_npu = num_npu
        self.num_ssu = num_ssu
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.policy = policy
        self.qos_configs_by_ssu = qos_configs_by_ssu
        self.npu_dedicated_paths = npu_dedicated_paths
        self.layer0_path_id = layer0_path_id
        self.client_io_config = client_io_config
        self.control = control
        self.causal_control = causal_control
        self.disk_bw_gbps = disk_bw_gbps
        self.npu_bw_gbps = npu_bw_gbps
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
            sim.DiskIOScheduler(disk, policy, disk_bw_gbps, qos)

        self.client_path_pools_by_ssu = (
            tuple(
                {
                    category: sim.client_category_paths(category, qos)
                    for category in sim.QOS_ROUTING_CATEGORIES
                }
                for qos in qos_configs_by_ssu
            )
            if policy == sim.POLICY_QOS_STATIC_CIR
            and npu_dedicated_paths is None
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
        self.causal_observations = 0
        self.next_wall_control_ms: Optional[float] = None
        self.control_evaluations = 0
        self.cir_commits = 0
        self.cir_path_writes = 0
        self.max_cir_sum_gbps = 0.0

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
                    self.block_offsets[(request.request_id, layer)] = (
                        self.expected_blocks
                    )
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


def _register_submit(context: _Context, flow: sim.BlockIOFlow):
    request = context.requests[flow.request_id].manifest
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
    if not all(context.requests[request_id].io_ready[layer]
               for request_id in batch.member_request_ids):
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
):
    if layer < 0 or layer >= context.n_layers or request.io_started[layer]:
        return
    request.io_started[layer] = 1
    placement_groups = _state_placement_groups(request, layer)
    request.pending_blocks[layer] = sum(
        len(blocks) for _, blocks, _ in placement_groups
    )
    if not placement_groups:
        _mark_layer_io_ready(context, request, layer, current_time_ms)
        return
    work_by_ssu = {
        ssu_id: work_gb for ssu_id, _, work_gb in placement_groups
    }
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
            (context.layer0_path_id,)
            if layer == 0 and context.layer0_path_id is not None
            else
            (context.npu_dedicated_paths[npu_id],)
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
            start_offset=(
                request.manifest.request_id + layer * 13 + ssu_id * 29
            )
            % len(allowed_paths)
            if allowed_paths
            else 0,
            allowed_path_ids=allowed_paths,
            demand_gbps=demand_by_ssu.get(ssu_id, 0.0),
            deadline_time_ms=deadline_time_ms,
            layer_work_gb=work_gb,
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
    if state.layer == 0 and context.layer0_path_id is not None:
        state.planned_path_ids.extend(
            context.layer0_path_id
            for _ in range(window_start, len(state.blocks))
        )
        return
    if context.npu_dedicated_paths is not None:
        state.planned_path_ids.extend(
            state.allowed_path_ids[0]
            for _ in range(window_start, len(state.blocks))
        )
        return
    mode = context.client_io_config.path_selection_mode
    if mode == sim.PATH_SELECTION_FIXED_PATH_ZERO:
        state.planned_path_ids.extend(
            0 for _ in range(window_start, len(state.blocks))
        )
        return
    if mode == sim.PATH_SELECTION_STATELESS_RR:
        allowed = state.allowed_path_ids
        state.planned_path_ids.extend(
            allowed[(state.start_offset + index) % len(allowed)]
            for index in range(window_start, len(state.blocks))
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
            path_ids = state.planned_path_ids[state.cursor:chunk_end]
        else:
            chunk_end = submit_end
            path_ids = [-1] * (chunk_end - state.cursor)
        flows = []
        for (block_index, size_gb), path_id in zip(
            state.blocks[state.cursor:chunk_end], path_ids
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
    reasons = context.control_reasons_by_time[float(current_time_ms)]
    if not reasons:
        context.push_event(current_time_ms, CIR_CONTROL, 0)
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
        request.placement_groups = tuple(
            _group_placement_layer(layer) for layer in request.manifest.placement
        )
        request.completed_gb_by_layer_ssu = [
            [0.0] * context.num_ssu for _ in range(context.n_layers)
        ]
        request.admitted = True
        request.admission_time_ms = current_time_ms
        request.batch_id = batch.batch_id
        request.batch_size = len(member_ids)
    _queue_control_event(context, current_time_ms, "batch_boundary")
    _queue_causal_control_event(context, current_time_ms)
    _start_batch_layer_io(
        context,
        batch,
        0,
        current_time_ms,
        current_time_ms,
    )


def _handle_arrival(context: _Context, request_id: int, current_time_ms: float):
    request = context.requests[request_id]
    request.arrived = True
    npu = context.npus[request.manifest.npu_id]
    npu.future_arrivals -= 1
    npu.admission_queue.append(request_id)
    _schedule_batch_dispatch(context, npu, current_time_ms)


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
    context.push_event(
        end_time_ms,
        sim.COMPUTE_DONE,
        npu_id,
        payload=(batch.batch_id, layer),
        generation=npu.compute_generation,
    )


def _schedule_control_if_due(context: _Context, current_time_ms: float):
    if (
        context.control is None
        or context.control.every_layers is None
    ):
        return
    if (
        context.control_threshold_jobs > 0
        and context.layer_jobs_since_control >= context.control_threshold_jobs
        and context.completed_requests < len(context.requests)
    ):
        _queue_control_event(context, current_time_ms, "fleet_layer")


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
    npu.last_completion_ms = current_time_ms
    npu.active_batch = None
    _queue_control_event(context, current_time_ms, "batch_boundary")
    _queue_causal_control_event(context, current_time_ms)
    _schedule_batch_dispatch(context, npu, current_time_ms)


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
    batch = context.microbatches[request.batch_id]
    next_layer = min(batch.compute_done_up_to + 1, context.n_layers - 1)
    work = [0.0] * context.num_ssu
    for ssu_id, _, work_gb in _state_placement_groups(request, next_layer):
        work[ssu_id] = work_gb
    waiting_for_io = (
        batch.compute_active_layer < 0
        and batch.compute_done_up_to + 1 < context.n_layers
        and not request.io_ready[batch.compute_done_up_to + 1]
    )
    return ControlRequestView(
        request_id=request.manifest.request_id,
        npu_id=request.manifest.npu_id,
        category=request.category,
        per_layer_compute_ms=request.per_layer_compute_ms,
        compute_done_up_to=batch.compute_done_up_to,
        remaining_layers=context.n_layers - batch.compute_done_up_to - 1,
        next_layer_work_gb_by_ssu=tuple(work),
        waiting_for_io=waiting_for_io,
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
        tuple(float(cir) for cir in cirs)
        for cirs in decision.path_cirs_by_ssu
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
    changed = 0
    for ssu_id, cirs in enumerate(normalized_cirs):
        changed += context.disks[ssu_id].scheduler.update_path_cirs(
            cirs, current_time_ms
        )
        context.max_cir_sum_gbps = max(context.max_cir_sum_gbps, sum(cirs))
    if changed:
        context.cir_commits += 1
        context.cir_path_writes += changed


def _handle_control(context: _Context, current_time_ms: float):
    reasons = context.control_reasons_by_time.pop(float(current_time_ms), set())
    context.control_evaluations += 1
    active_request_ids = sorted(
        request_id
        for npu in context.npus
        if npu.active_batch is not None
        for request_id in npu.active_batch.member_request_ids
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


def _start_next_link_io(context: _Context, npu: _NPUState, current_time_ms: float):
    if npu.link_active_flow is not None or not npu.link_pending:
        return
    flow = npu.link_pending.popleft()
    npu.link_pending_gb -= flow.total_gb
    npu.link_active_flow = flow
    flow.link_start_time = current_time_ms
    service_ms = sim.npu_link_service_time_ms(
        flow.total_gb, context.npu_bw_gbps
    )
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


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else 0.0


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
                "avg_ssd_queue_wait_ms": request.ssd_queue_wait_ms
                / request.io_count,
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
                "backend_dispatches": scheduler.backend_dispatches,
            }
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
        "cir_capacity": context.max_cir_sum_gbps
        <= context.disk_bw_gbps + 1e-9,
        "causal_control_drained": not context.pending_causal_prefetches,
        "batch_latency_decomposition": all(
            abs(row["latency_accounting_error_ms"]) <= 1e-7
            for row in microbatch_metrics
        ),
        "request_latency_decomposition": all(
            abs(row["latency_accounting_error_ms"]) <= 1e-7
            for row in request_metrics
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
        ) == sorted(context.requests),
        "fixed_batch_layer_barrier": all(
            metric.compute_start_ms + _EPS >= metric.io_ready_time_ms
            and metric.compute_start_ms + _EPS >= (
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
            npu.active_batch is None and not npu.admission_queue
            for npu in context.npus
        ),
    }
    if not all(invariants.values()):
        raise AssertionError(f"continuous-batch invariant failed: {invariants}")

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
        "schema_version": 2,
        "execution_model": EXECUTION_MODEL,
        "batch_compute_model": BATCH_COMPUTE_MODEL,
        "batch_compute_calibration": "singleton layer times added; no batch speedup",
        "partial_batch_policy": PARTIAL_BATCH_POLICY,
        "prefetch_policy": PREFETCH_POLICY,
        "layer0_path_id": context.layer0_path_id,
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
        "active_window_npu_compute_utilization": total_compute_ms
        / active_window_ms,
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
        "cir_commits": context.cir_commits,
        "cir_path_writes": context.cir_path_writes,
        "completed_batch_layers": context.completed_batch_layers,
        "completed_request_layer_jobs": context.completed_request_layer_jobs,
        "completed_layer_jobs": context.completed_request_layer_jobs,
        "submitted_blocks": context.submitted_blocks,
        "completed_blocks": context.completed_blocks,
        "expected_read_gb": context.expected_read_gb,
        "completed_read_gb": total_read_gb,
        "events_processed": events_processed,
        "stale_events": context.stale_events,
        "event_counts": {str(key): value for key, value in context.event_counts.items()},
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
    client_io_config: sim.ClientIOConfig = sim.DEFAULT_CLIENT_IO_CONFIG,
    control: Optional[CIRControlConfig] = None,
    causal_control: Optional[CausalLayerControlConfig] = None,
    disk_bw_gbps: float = sim.DISK_BW,
    npu_bw_gbps: float = sim.NPU_BW_LIMIT,
    submit_order_seed: int = 42,
):
    """Run a finite Full-prefill microbatch trace and return JSON-safe metrics.

    Static routing strategies pass the same QoS configuration and different
    ``ClientIOConfig`` values.  Dynamic Scheme B additionally passes one
    dedicated Path per NPU and a periodic ``CIRControlConfig``.  The physical
    oracle uses ``POLICY_PER_SSD_FULL_VISIBLE_EDF`` and no QoS configuration.
    """
    requests = tuple(requests)
    if not requests:
        raise ValueError("continuous-batch trace must contain requests")
    if num_npu <= 0 or num_ssu <= 0 or n_layers <= 0 or batch_size <= 0:
        raise ValueError("simulation dimensions must be positive")
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
        layer0_path_id=(
            int(layer0_path_id) if layer0_path_id is not None else None
        ),
        client_io_config=client_io_config,
        control=control,
        causal_control=causal_control,
        disk_bw_gbps=float(disk_bw_gbps),
        npu_bw_gbps=float(npu_bw_gbps),
        submit_order_seed=int(submit_order_seed),
    )
    if disk_qos_configs:
        context.max_cir_sum_gbps = max(
            sum(qos.path_cirs) for qos in disk_qos_configs
        )
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
        context.event_counts[event_type] += 1
        events_processed += 1
        if event_type == REQUEST_ARRIVAL:
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

    return _build_summary(context, requests, current_time_ms, events_processed)
