"""Thin process-local adapters for shared-Path client policies.

The native SSD/FCFS receive-link service and input arrivals remain unchanged.
Only the collector may read SSU pressure, on a 5-ms clock. New clients also
share their own routing reservations and HBM acknowledgements. Such logical
outstanding work is NOT an exact measurement of SSD queue occupancy.

Strategy2 additionally permits same-Path pending-I/O reordering; its SSD-side
adapter sees only already enqueued commands and their submitted metadata.
All five policies use the SAME static CIR table, with no runtime CIR writes.
Use separate processes (not threads) for concurrent simulations.
"""

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

import continuous_batch_sim as native
import sim
from policy_logic import hardware_view, pressure_snapshot
from shared_ssu_state import PeriodicSSUState

IO_GIB = 176 * 1024 / 2**30


class SharedPathAdapter:
    def __init__(self, strategy, collector_interval_ms=5.0,
                 cir_min_interval_ms=100.0):
        if strategy not in ("baseline", "once", "new_once", "strategy1", "strategy2"):
            raise ValueError("unknown shared-Path policy")
        self.strategy = strategy
        self.collector_interval_ms = collector_interval_ms
        self.cir_min_interval_ms = cir_min_interval_ms
        self.context = None
        self.collector = None
        self.pending = []
        self.assignment_log = []
        self.layer_counts_cache = {}
        self.routing_calls = 0
        self.routing_wall_us = 0.0
        self.assignment_wall_us = 0.0
        self.collector_wall_us = 0.0
        self.ack_ledger_wall_us = 0.0
        self.reorder_calls = 0
        self.reorder_changed = 0
        self.reorder_wall_us = 0.0
        self.reserved_blocks = 0
        self.acknowledged_blocks = 0
        self.min_ledger_count = 0
        self.max_ledger_count = 0
        self.route_examples = []

    def layer_counts(self, request):
        """Only call for a request whose arrival is being handled or is past."""
        rid = request.manifest.request_id
        if rid not in self.layer_counts_cache:
            rows = []
            for placement in request.manifest.placement:
                counts = [0] * self.context.num_ssu
                for ssu_id, size in placement:
                    counts[ssu_id] += round(size / IO_GIB)
                rows.append(tuple(counts))
            self.layer_counts_cache[rid] = tuple(rows)
        return self.layer_counts_cache[rid]

    def backlogs(self, now_ms):
        from shared_path_strategy1 import KnownNPU
        context = self.context
        result = []
        for npu in context.npus:
            ids = list(npu.admission_queue)
            compute = sum(context.n_layers * context.requests[r].per_layer_compute_ms
                          for r in ids)
            batch = npu.active_batch
            if batch is not None:
                rid = batch.member_request_ids[0]
                ids.append(rid)
                c = context.requests[rid].per_layer_compute_ms
                if npu.compute_active is not None:
                    layer = npu.compute_active[1]
                    compute += max(0, batch.layer_metrics[layer].compute_end_ms - now_ms)
                    compute += (context.n_layers - layer - 1) * c
                else:
                    compute += (context.n_layers - batch.compute_done_up_to - 1) * c
            unfinished = [0] * context.num_ssu
            for rid in ids:
                request = context.requests[rid]
                rows = self.layer_counts(request)
                for layer in range(context.n_layers):
                    if request.io_ready[layer]:
                        continue
                    row = rows[0 if len(rows) == 1 else layer]
                    for s, count in enumerate(row):
                        done = (round(request.completed_gb_by_layer_ssu[layer][s] / IO_GIB)
                                if request.completed_gb_by_layer_ssu else 0)
                        unfinished[s] += count - done
            result.append(KnownNPU(npu.npu_id, compute, tuple(unfinished), len(ids)))
        return tuple(result)

    def statistics(self):
        return {
            **self.collector.statistics(),
            "strategy": self.strategy,
            "routing_calls": self.routing_calls,
            "routing_wall_us": self.routing_wall_us,
            "assignment_wall_us": self.assignment_wall_us,
            "collector_wall_us": self.collector_wall_us,
            "ack_ledger_wall_us": self.ack_ledger_wall_us,
            "assignment_count": len(self.assignment_log),
            "reorder_calls": self.reorder_calls,
            "reorder_changed": self.reorder_changed,
            "reorder_wall_us": self.reorder_wall_us,
            "reserved_blocks": self.reserved_blocks,
            "acknowledged_blocks": self.acknowledged_blocks,
            "ledger_end_counts_by_ssu": [sum(row) for row in self.pending],
            "min_ledger_count": self.min_ledger_count,
            "max_ledger_count": self.max_ledger_count,
            "route_examples": self.route_examples,
            "native_fresh_reads_by_ssu": [d.scheduler.pressure_reports
                                          for d in self.context.disks],
            "modeled_control_communication_latency_ms": 0,
            "modeled_control_cpu_latency_ms": 0,
            "global_client_ledger": "all managed IO: planned/unissued + sent/unacked at HBM",
            "runtime_cir_policy": "static; zero runtime writes",
        }


@contextmanager
def shared_path_adapter(strategy="once", collector_interval_ms=5.0,
                        cir_min_interval_ms=100.0):
    from shared_path_baseline import baseline_path_ids
    from shared_path_once import once_path_ids
    from shared_path_new_once import new_once_path_ids
    from shared_path_strategy1 import choose_npu
    from shared_path_strategy2 import PendingIO, reorder_pending

    adapter = SharedPathAdapter(strategy, collector_interval_ms, cir_min_interval_ms)
    original_init = native._Context.__init__
    original_pressure = sim.DiskIOScheduler.report_path_pressure_analysis
    original_complete = native._register_complete
    original_arrival = native._handle_arrival
    original_enqueue = sim.PathQueue.enqueue
    original_activate = sim.PathQueue.activate_next

    def collect(context, now_ms):
        started = perf_counter()
        def read(ssu_id):
            scheduler = context.disks[ssu_id].scheduler
            # Force precisely the periodic device read; clients cannot call it.
            scheduler._pressure_cache = None
            snapshot = original_pressure(scheduler, now_ms)
            return snapshot, tuple(path.cir for path in scheduler.paths.values())
        adapter.collector.collect(now_ms, read)
        context.push_event(adapter.collector.next_collection_ms, native.CIR_CONTROL, 0)
        adapter.collector_wall_us += (perf_counter() - started) * 1e6

    def initialize(context, *args, **kwargs):
        # Post-arbitration same-Path reordering is safe here because every
        # command has the same size; a different head cannot change its tag.
        if any(size != IO_GIB for request in kwargs["requests"]
               for layer in request.placement for _ssu, size in layer):
            raise ValueError("shared-Path experiments require equal 176 KiB I/O")
        original_init(context, *args, **kwargs)
        if context.control is not None or context.npu_dedicated_paths is not None:
            raise ValueError("shared-Path experiments do not use dedicated CIR control")
        adapter.context = context
        adapter.pending = [[0] * 256 for _ in range(context.num_ssu)]
        adapter.qos = tuple(hardware_view(q) for q in context.qos_configs_by_ssu)
        adapter.collector = PeriodicSSUState(context.num_ssu, collector_interval_ms,
                                            cir_min_interval_ms)
        collect(context, 0.0)

    def cached_pressure(scheduler, now_ms):
        scheduler.pressure_cache_hits += 1
        return adapter.collector.get(scheduler.state.disk_id, now_ms)

    def plan(context, state, now_ms):
        started = perf_counter()
        first = len(state.planned_path_ids)
        count = len(state.blocks) - first
        s = state.disk_id
        if strategy == "baseline":
            ids = baseline_path_ids(count)
        else:
            observed = pressure_snapshot(adapter.collector.get(s, now_ms))
            args = (count, observed)
            kwargs = dict(allowed_path_ids=state.allowed_path_ids, qos=adapter.qos[s],
                          start_offset=state.start_offset,
                          disk_bw_gib_s=context.disk_bw_gbps)
            ids = (once_path_ids(*args, **kwargs) if strategy == "once" else
                   new_once_path_ids(*args, client_pending_counts=tuple(adapter.pending[s]),
                                     **kwargs))
        state.planned_path_ids.extend(ids)
        # Atomic global reservations include unissued suffixes, preventing
        # simultaneous planners from repeatedly choosing the same empty Path.
        for path_id, number in Counter(ids).items():
            adapter.pending[s][path_id] += number
            adapter.max_ledger_count = max(adapter.max_ledger_count,
                                           adapter.pending[s][path_id])
        adapter.reserved_blocks += len(ids)
        adapter.routing_calls += 1
        adapter.routing_wall_us += (perf_counter() - started) * 1e6
        if len(adapter.route_examples) < 12:
            adapter.route_examples.append({"time_ms": now_ms,
                "snapshot_time_ms": adapter.collector.sampled_at_ms,
                "request_id": state.request_id, "layer": state.layer,
                "ssu_id": s, "io_count": count,
                "path_counts": dict(Counter(ids))})

    def completed(context, flow):
        original_complete(context, flow)
        started = perf_counter()
        adapter.pending[flow.disk_id][flow.queue_id] -= flow.block_count
        adapter.acknowledged_blocks += flow.block_count
        value = adapter.pending[flow.disk_id][flow.queue_id]
        adapter.min_ledger_count = min(adapter.min_ledger_count, value)
        if value < 0:
            raise AssertionError("HBM acknowledgement without a routing reservation")
        adapter.ack_ledger_wall_us += (perf_counter() - started) * 1e6

    def arrival(context, rid, now_ms):
        if strategy in ("strategy1", "strategy2"):
            started = perf_counter()
            request = context.requests[rid]
            old = request.manifest
            rows = adapter.layer_counts(request)
            if len(rows) != 1:
                raise ValueError("this simple admission policy expects one repeated layer layout")
            backlogs = adapter.backlogs(now_ms)
            snapshots = tuple(adapter.collector.get(s, now_ms).counts
                              for s in range(context.num_ssu))
            choice = choose_npu(now_ms, backlogs, rows[0], request.per_layer_compute_ms,
                                n_layers=context.n_layers,
                                snapshot_counts_by_ssu=snapshots,
                                disk_bw_gib_s=context.disk_bw_gbps,
                                link_bw_gib_s=context.npu_bw_gbps)
            if choice.npu_id != old.npu_id:
                context.npus[old.npu_id].future_arrivals -= 1
                context.npus[choice.npu_id].future_arrivals += 1
                request.manifest = replace(old, npu_id=choice.npu_id,
                                           load={**old.load, "npu_id": choice.npu_id})
            elapsed = (perf_counter() - started) * 1e6
            adapter.assignment_wall_us += elapsed
            adapter.assignment_log.append({"request_id": rid, "arrival_time_ms": now_ms,
                "original_npu_id": old.npu_id, "assigned_npu_id": choice.npu_id,
                "snapshot_time_ms": adapter.collector.sampled_at_ms,
                "scores_ms": choice.scores_ms, "decision_wall_us": elapsed})
        original_arrival(context, rid, now_ms)

    def enqueue(path, flow):
        path._shared_reorder_dirty = True
        return original_enqueue(path, flow)

    def activate(path):
        if (strategy == "strategy2" and len(path.pending) > 1 and
                getattr(path, "_shared_reorder_dirty", False)):
            started = perf_counter()
            commands = list(path.pending)
            metadata = [PendingIO(io_id=i, npu_id=f.npu_id, request_id=f.request_id,
                layer=f.layer, path_id=f.queue_id, ssu_id=f.disk_id,
                arrival_ms=f.enqueue_time, deadline_ms=f.deadline_time,
                layer_io_count=max(1, round(f.layer_work_gb / IO_GIB)))
                for i, f in enumerate(commands)]
            order = reorder_pending(metadata, adapter.context.current_time_ms)
            if tuple(order) != tuple(range(len(commands))):
                path.pending = deque(commands[i] for i in order)
                adapter.reorder_changed += 1
            adapter.reorder_calls += 1
            adapter.reorder_wall_us += (perf_counter() - started) * 1e6
            path._shared_reorder_dirty = False
        return original_activate(path)

    with patch.object(native._Context, "__init__", initialize), \
         patch.object(sim.DiskIOScheduler, "report_path_pressure_analysis", cached_pressure), \
         patch.object(native, "_handle_control", collect), \
         patch.object(native, "_plan_paths", plan), \
         patch.object(native, "_register_complete", completed), \
         patch.object(native, "_handle_arrival", arrival), \
         patch.object(sim.PathQueue, "enqueue", enqueue), \
         patch.object(sim.PathQueue, "activate_next", activate):
        yield adapter
