"""Causal multi-SSU CIR planning for dedicated per-NPU QoS Paths.

Payload remains SSD -> HBM.  A coflow is one layer's reads across all SSUs;
its barrier waits for every component.  Plans obey both SSD-column and
NPU-link-row budgets.  CIR is a preference/reservation in the native virtual
finish scheduler, NOT EDF, a hard peak-rate limit, or a no-stall guarantee.
In particular, zero-CIR Paths can borrow and create a real HBM receive queue.

The pure planner needs client-known unfinished bytes, compute deadline, and
topology.  The optional native adapter observes only arrived request manifests,
actual compute starts, and HBM completion notifications.  It never reads SSD
FIFO order, SSD active-command remainder, or unknown future arrivals.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from unittest.mock import patch

from simulator.policies.allocation import allocate_coflow_grants
from simulator.policies.policy_logic import dedicated_path_id

# Re-export the previous public API without retaining decisions in this adapter.
from simulator.policies.multi_ssu_rates import (
    IO_GIB, _EPS, PendingCoflow, coflow_solo_ms, serial_stall_objective,
    stall_interchange_order, choose_coflow_rates,
)


def snapshot_pending_coflows(context, now_ms):
    """Read client-side progress; unfinished work includes link-only bytes.

    completed_gb_by_layer_ssu is updated at HBM completion, not SSD service.
    This conservative work estimate cannot distinguish queued disk work from
    bytes already crossing the link. No retroactive late-arrival prefetch is
    invented. A computing predecessor's end is the cross-request L0 deadline.
    """
    jobs = []
    for npu in context.npus:
        batch = npu.active_batch
        if batch is None:
            continue
        if len(batch.member_request_ids) != 1:
            raise ValueError("the experimental controller requires batch_size=1")
        rid = batch.member_request_ids[0]
        if npu.compute_active is not None:
            current_layer = npu.compute_active[1]
            metric = batch.layer_metrics[current_layer]
            deadline = metric.compute_end_ms
            compute_ms = metric.compute_duration_ms
            layer = current_layer + 1
            if layer == context.n_layers:
                rid = npu.layer0_prefetch_request_id
                layer = 0
                if rid is None:
                    continue
        else:
            layer = batch.compute_done_up_to + 1
            deadline = batch.previous_compute_end_ms
            compute_ms = context.requests[rid].per_layer_compute_ms
        request = context.requests[rid]
        if (layer >= context.n_layers or not request.io_started[layer]
                or request.io_ready[layer]):
            continue
        # placement_groups is only a cached grouping of the already-arrived
        # manifest. It contains no current scheduler or future-event state.
        groups = request.placement_groups[0 if len(request.placement_groups) == 1
                                          else layer]
        work = [0.0] * context.num_ssu
        for ssu_id, _blocks, total_gib in groups:
            work[ssu_id] = max(0.0, total_gib -
                              request.completed_gb_by_layer_ssu[layer][ssu_id])
        jobs.append(PendingCoflow(npu.npu_id, rid, layer, tuple(work),
                                 deadline, compute_ms))
    return tuple(jobs)


class MultiSSUController:
    def __init__(self, mode="deadline", *, ssd_cap_gib_s=40.0,
                 npu_cap_gib_s=50.0, log_limit=64, on_ssu_done=True):
        self.mode = mode
        self.ssd_cap_gib_s = ssd_cap_gib_s
        self.npu_cap_gib_s = npu_cap_gib_s
        self.log_limit = log_limit
        self.on_ssu_done = on_ssu_done
        self.context = None
        self.decisions = []
        self.evaluations = 0
        self.total_decision_wall_us = 0.0
        self.max_decision_wall_us = 0.0
        self.ssu_component_done_events = 0

    def __call__(self, snapshot):
        from simulator.core.continuous_batch_sim import CIRControlDecision
        started = perf_counter()
        jobs = snapshot_pending_coflows(self.context, snapshot.time_ms)
        rates = choose_coflow_rates(
            jobs, snapshot.time_ms, num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu, mode=self.mode,
            ssd_cap_gib_s=self.ssd_cap_gib_s,
            npu_cap_gib_s=self.npu_cap_gib_s)
        tables = [[0.0] * 256 for _ in range(snapshot.num_ssu)]
        for npu_id, row in enumerate(rates):
            path = dedicated_path_id(npu_id)
            for ssu_id, rate in enumerate(row):
                tables[ssu_id][path] = rate
        decision = CIRControlDecision(tuple(tuple(table) for table in tables))
        elapsed = (perf_counter() - started) * 1e6
        self.evaluations += 1
        self.total_decision_wall_us += elapsed
        self.max_decision_wall_us = max(self.max_decision_wall_us, elapsed)
        if len(self.decisions) < self.log_limit:
            self.decisions.append({"time_ms": snapshot.time_ms,
                                   "jobs": [asdict(job) for job in jobs],
                                   "grants_gib_s": rates,
                                   "decision_wall_us": elapsed})
        return decision

    def statistics(self):
        return {
            "mode": self.mode,
            "evaluations": self.evaluations,
            "total_decision_wall_us": self.total_decision_wall_us,
            "mean_decision_wall_us": (self.total_decision_wall_us / self.evaluations
                                      if self.evaluations else 0.0),
            "max_decision_wall_us": self.max_decision_wall_us,
            "logged_decisions": len(self.decisions),
            "log_limit": self.log_limit,
            "ssu_component_done_events": self.ssu_component_done_events,
            "on_ssu_done": self.on_ssu_done,
            "hardware_control_latency_modeled": False,
            "planner_row_cap_is_not_a_PIR_or_instantaneous_emission_cap": True,
        }


@contextmanager
def multi_ssu_control_events(controller):
    """Attach layer-start/ready and optional per-SSU HBM-completion triggers.

    The component-complete trigger occurs at most once per SSU per layer,
    rather than on every 176 KiB completion. It releases a non-bottleneck SSU's
    obsolete reservation. Native CIRControlConfig.min_interval_ms coalesces
    updates if desired. Native counters charge observations/register writes,
    but neither communication latency nor controller CPU time delays I/O.
    Temporary hooks are process-local and restored on exit; use processes,
    not threads, for independent concurrent experiments.
    """
    from simulator.core import continuous_batch_sim as native
    original_control = native._handle_control
    original_compute = native._handle_compute_schedule
    original_ready = native._mark_layer_io_ready
    original_link = native._handle_link_completion
    original_queue_control = native._queue_control_event

    def queue_control(context, now, reason):
        # Native min_interval_ms only applies to "batch_boundary". Extend the
        # same causal gate to our event reasons, without changing the core or
        # delaying any data-plane event. Pending notifications then coalesce
        # in the native reasons-by-time map at the same eligible control time.
        effective = now
        if (reason.startswith("coflow_") and context.control is not None
                and context.last_control_ms is not None):
            effective = max(now, context.last_control_ms +
                            context.control.min_interval_ms)
        original_queue_control(context, effective, reason)
        if (context.timeline_diagnostics and reason.startswith("coflow_")
                and context.control is not None):
            context.timeline_control_triggers[-1]["raw_time_ms"] = float(now)
            context.timeline_control_triggers[-1]["rate_limited"] = effective > now + _EPS

    def control(context, now):
        controller.context = context
        original_control(context, now)

    def compute(context, npu_id, now):
        previous = context.npus[npu_id].compute_active
        original_compute(context, npu_id, now)
        if previous != context.npus[npu_id].compute_active:
            native._queue_control_event(context, now, "coflow_layer_start")

    def ready(context, request, layer, now):
        original_ready(context, request, layer, now)
        native._queue_control_event(context, now, "coflow_layer_ready")

    def link(context, npu_id, generation, now):
        # Read only the identity/amount reported by this actual host completion.
        # No SSD active-flow state or predicted link finish time is inspected.
        npu = context.npus[npu_id]
        flow = npu.link_active_flow
        if not controller.on_ssu_done or flow is None:
            original_link(context, npu_id, generation, now)
            return
        request = context.requests[flow.request_id]
        previous = request.completed_gb_by_layer_ssu[flow.layer][flow.disk_id]
        original_link(context, npu_id, generation, now)
        completed = request.completed_gb_by_layer_ssu[flow.layer][flow.disk_id]
        if completed <= previous:
            return
        groups = request.placement_groups[0 if len(request.placement_groups) == 1
                                          else flow.layer]
        total = next(amount for ssu_id, _blocks, amount in groups
                     if ssu_id == flow.disk_id)
        if total - completed <= _EPS:
            controller.ssu_component_done_events += 1
            native._queue_control_event(context, now, "coflow_ssu_component_ready")

    with patch.object(native, "_handle_control", control), \
         patch.object(native, "_handle_compute_schedule", compute), \
         patch.object(native, "_mark_layer_io_ready", ready), \
         patch.object(native, "_handle_link_completion", link), \
         patch.object(native, "_queue_control_event", queue_control):
        yield controller
