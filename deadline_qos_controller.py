"""Per-layer CIR steering using observed deadlines and unfinished read counts.

Experimental adapter for four dedicated Paths on the existing SSD scheduler.
It does not change NPU assignment, submission timing, compute duration, or the
SSD arbitration algorithm. A CIR preference is NOT an exact EDF scheduler.
Only NPU-side observed work/progress is read, never the SSD FIFO or events.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter
from unittest.mock import patch

from npu_stall_predictor import io_service_ms, required_rate_gib_s


@dataclass(frozen=True)
class PendingLayer:
    npu_id: int
    request_id: int
    layer: int
    remaining_blocks: int
    deadline_ms: float
    compute_ms: float


def choose_deadline_rates(jobs, now_ms, capacity=40.0, mode="edf"):
    """Plan 4 CIRs from current observable layer jobs, without FIFO order.

    EDF gives full CIR to the earliest data deadline. 'least_slack' subtracts
    solo service from the deadline. I/O commands remain non-preemptive and
    virtual-finish history remains native, so neither implements exact EDF.
    Zero-CIR queues can borrow when the selected Path has no submitted I/O.
    """
    rates = [0.0] * 4
    pending = [j for j in jobs if j.remaining_blocks]
    if not pending:
        return tuple(rates)
    def key(job):
        remaining_service = job.remaining_blocks * io_service_ms(capacity)
        deadline = job.deadline_ms
        if mode == "least_slack":
            deadline -= remaining_service
        return deadline, remaining_service, job.npu_id
    chosen = min(pending, key=key)
    rates[chosen.npu_id] = capacity
    return tuple(rates)


def snapshot_pending_layers(context, now_ms):
    """Client-visible layer progress; pending counts include the HBM tail.

    Counting the possibly link-only final command is conservative work, not
    hidden SSD remainder telemetry. Layer-ready callbacks promptly drop it.
    A late arrival does not get retroactive cross-request prefetch here.
    """
    jobs = []
    for npu in context.npus:
        batch = npu.active_batch
        if batch is None:
            continue
        rid = batch.member_request_ids[0]
        if npu.compute_active is not None:
            current_layer = npu.compute_active[1]
            deadline = batch.layer_metrics[current_layer].compute_end_ms
            layer = current_layer + 1
            if layer == context.n_layers:
                rid = npu.layer0_prefetch_request_id
                layer = 0
                if rid is None:
                    continue
        else:
            layer = batch.compute_done_up_to + 1
            deadline = batch.previous_compute_end_ms
        request = context.requests[rid]
        if layer >= context.n_layers or not request.io_started[layer] or request.io_ready[layer]:
            continue
        jobs.append(PendingLayer(npu.npu_id, rid, layer,
                                 request.pending_blocks[layer], deadline,
                                 request.per_layer_compute_ms))
    return tuple(jobs)


class DeadlineController:
    def __init__(self, mode="edf"):
        self.mode = mode
        self.context = None
        self.decisions = []

    def __call__(self, snapshot):
        from continuous_batch_sim import CIRControlDecision
        started = perf_counter()
        jobs = snapshot_pending_layers(self.context, snapshot.time_ms)
        rates = choose_deadline_rates(jobs, snapshot.time_ms, mode=self.mode)
        cirs = [0.0] * 256
        for path, rate in zip((0, 32, 64, 96), rates):
            cirs[path] = rate
        self.decisions.append({
            "time_ms": snapshot.time_ms, "cirs": rates,
            "jobs": [asdict(j) for j in jobs],
            "fluid_required_gib_s": [required_rate_gib_s(
                j.remaining_blocks, j.deadline_ms - snapshot.time_ms) for j in jobs],
            "decision_wall_us": (perf_counter() - started) * 1e6,
        })
        return CIRControlDecision((tuple(cirs),))


@contextmanager
def deadline_control_events(controller):
    """Evaluate at actual layer start/ready events plus native batch changes.

    Temporary hooks are restored on exit. Run parallel experiments in separate
    processes. These extra evaluations and CIR writes are charged in native
    counters, but hardware read/write latency is not modeled by the simulator.
    """
    import continuous_batch_sim as native
    original_control = native._handle_control
    original_compute = native._handle_compute_schedule
    original_ready = native._mark_layer_io_ready

    def control(context, now):
        controller.context = context
        original_control(context, now)

    def compute(context, npu_id, now):
        previous = context.npus[npu_id].compute_active
        original_compute(context, npu_id, now)
        if previous != context.npus[npu_id].compute_active:
            native._queue_control_event(context, now, "deadline_layer_start")

    def ready(context, request, layer, now):
        original_ready(context, request, layer, now)
        native._queue_control_event(context, now, "deadline_layer_ready")

    with patch.object(native, "_handle_control", control), \
         patch.object(native, "_handle_compute_schedule", compute), \
         patch.object(native, "_mark_layer_io_ready", ready):
        yield controller
