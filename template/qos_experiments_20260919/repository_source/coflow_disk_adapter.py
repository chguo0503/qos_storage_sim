"""Non-preemptive, equal-I/O QoS tie-break hooks for new S2/S3 experiments.

S2 chooses across native minimum-virtual-finish-equivalent Paths by local
deadline and may move a pending command to its own Path's head. S3 replaces
the first priority component with a caller-supplied joint-coflow key. It never
reads another disk. Neither strategy may choose a higher-tag non-equivalent
Path, change CIR/PIR/rates, migrate an I/O, or preempt an active command.

Native sim.py rejects finite PIR: these experiments retain PIR=infinity, not
a newly implemented token bucket or a claimed instantaneous bandwidth cap.
Use inside shared_path_adapter('new_once'), not its old strategy2 sorter.
Concurrent experiments require separate processes, not threads.
"""

from collections import Counter
from contextlib import contextmanager
import heapq
from time import perf_counter
from unittest.mock import patch

import sim
from coflow_disk_policy import PathCandidate, choose_path, local_priority
from shared_path_common import IO_GIB


class DiskPolicyAudit:
    def __init__(self, mode, global_priority):
        self.mode = mode
        self.global_priority = global_priority
        self.schedulers = {}
        self.initial_qos = {}
        self.now_by_disk = {}
        self.serial = 0
        self.selection_calls = 0
        self.multi_path_opportunities = 0
        self.cross_path_changed = 0
        self.same_path_head_changed = 0
        self.candidate_set_sizes = Counter()
        self.max_selected_tag_gap = 0.0
        self.active_guard_calls = 0
        self.global_priority_calls = 0
        self.examined_coflow_heads = 0
        self.enqueue_commands = 0
        self.enqueue_wall_us = 0.0
        self.decision_wall_us = 0.0
        self.examples = []

    def remember(self, scheduler):
        disk = scheduler.state.disk_id
        if disk not in self.schedulers:
            self.schedulers[disk] = scheduler
            self.initial_qos[disk] = tuple((p.cir, p.pir) for p in scheduler.paths.values())

    def register(self, path, flow):
        """Keep local per-coflow heaps; do not scan all pending I/O per choice.

        The global callback must be constant across commands of one
        (request_id, layer) coflow at a given decision time. Only its local
        earliest pending command need then represent that coflow on this Path.
        """
        if not hasattr(path, "_coflow_groups"):
            path._coflow_groups = {}
        flow._coflow_pending = True
        key = (flow.request_id, flow.layer)
        group = path._coflow_groups.setdefault(key, [])
        heapq.heappush(group, (local_priority(flow, flow.enqueue_time), self.serial, flow))
        self.serial += 1
        self.enqueue_commands += 1

    def best_pending(self, path, now, global_keys):
        options = []
        for coflow, group in tuple(path._coflow_groups.items()):
            while group and not group[0][2]._coflow_pending:
                heapq.heappop(group)
            if not group:
                del path._coflow_groups[coflow]
                continue
            local_key, serial, flow = group[0]
            self.examined_coflow_heads += 1
            if self.mode == "strategy3":
                if coflow not in global_keys:
                    global_keys[coflow] = tuple(self.global_priority(flow, now))
                    self.global_priority_calls += 1
                key = global_keys[coflow] + local_key
            else:
                key = local_key
            options.append((key, serial, flow))
        return min(options, key=lambda item: (item[0], item[1]))

    def statistics(self):
        unchanged = all(tuple((p.cir, p.pir) for p in scheduler.paths.values())
                        == self.initial_qos[disk]
                        for disk, scheduler in self.schedulers.items())
        return {"mode": self.mode,
                "scope": "cross-Path choice only within native min-finish+EPS equivalence set",
                "selection_calls": self.selection_calls,
                "multi_path_opportunities": self.multi_path_opportunities,
                "cross_path_changed_vs_native_rr": self.cross_path_changed,
                "same_path_pending_head_changed": self.same_path_head_changed,
                "candidate_set_size_histogram": dict(sorted(self.candidate_set_sizes.items())),
                "max_selected_tag_gap": self.max_selected_tag_gap,
                "native_tag_epsilon": sim._EPS,
                "active_guard_calls": self.active_guard_calls,
                "global_priority_calls": self.global_priority_calls,
                "examined_coflow_heads": self.examined_coflow_heads,
                "enqueue_commands": self.enqueue_commands,
                "enqueue_wall_us": self.enqueue_wall_us,
                "decision_wall_us": self.decision_wall_us,
                "cir_pir_unchanged": unchanged,
                "finite_pir_supported": False,
                "modeled_control_latency_ms": 0.0,
                "decision_examples": self.examples}


@contextmanager
def coflow_disk_adapter(mode="strategy2", get_global_priority=None):
    """Yield audit counters; get_global_priority(flow, now_ms) returns a tuple.

    The supplied callback must use a precomputed caller-owned progress map,
    not hidden SSD inspection. It returns the same key for all commands of a
    given (request_id, layer). S2 requires no global callback. Submitted
    command placement and equal 176-KiB size are preserved exactly.
    """
    if mode not in ("strategy2", "strategy3"):
        raise ValueError("disk mode must be strategy2 or strategy3")
    if mode == "strategy3" and get_global_priority is None:
        raise ValueError("strategy3 needs caller-provided coflow progress")
    audit = DiskPolicyAudit(mode, get_global_priority)
    original_enqueue_many = sim.DiskIOScheduler.enqueue_many
    original_enqueue = sim.PathQueue.enqueue
    original_activate = sim.PathQueue.activate_next
    original_dispatch_one = sim.DiskIOScheduler._dispatch_one

    def enqueue_many(scheduler, flows, now):
        flows = tuple(flows)
        if any(flow.total_gb != IO_GIB or flow.block_count != 1 for flow in flows):
            raise ValueError("coflow disk policies require one equal 176 KiB I/O per command")
        audit.remember(scheduler)
        return original_enqueue_many(scheduler, flows, now)

    def enqueue(path, flow):
        result = original_enqueue(path, flow)
        started = perf_counter()
        audit.register(path, flow)
        audit.enqueue_wall_us += (perf_counter() - started) * 1e6
        return result

    def activate(path):
        flow = original_activate(path)
        if flow is not None:
            flow._coflow_pending = False
        return flow

    def dispatch_one(scheduler, now):
        audit.remember(scheduler)
        audit.now_by_disk[scheduler.state.disk_id] = now
        active = scheduler.state.active_flows
        before = ((id(active[0]), active[0].ssd_activation_time, active[0].end_time)
                  if active else None)
        result = original_dispatch_one(scheduler, now)
        if before is not None:
            audit.active_guard_calls += 1
            assert (id(result), result.ssd_activation_time, result.end_time) == before
        return result

    def select(scheduler):
        started = perf_counter()
        if scheduler._qos_finish_heap is None:
            scheduler._build_qos_arbitration_cache()
        if not scheduler._qos_backlogged_paths_cache:
            return None
        minimum = heapq.heappop(scheduler._qos_finish_heap)
        tied_tags = [minimum]
        while scheduler._qos_finish_heap and scheduler._qos_finish_heap[0] <= minimum + sim._EPS:
            tied_tags.append(heapq.heappop(scheduler._qos_finish_heap))
        now = audit.now_by_disk[scheduler.state.disk_id]
        candidates, best_flows, global_keys = [], {}, {}
        for tag in tied_tags:
            for path_id in scheduler._qos_finish_buckets[tag]:
                key, _serial, flow = audit.best_pending(scheduler.paths[path_id], now, global_keys)
                candidates.append(PathCandidate(path_id, tag, key))
                best_flows[path_id] = flow
        native_rr = min(candidates, key=lambda item: (
            (item.path_id - scheduler.qos_rr_cursor) % len(scheduler.paths),
            item.path_id, item.finish_tag))
        chosen = choose_path(candidates, scheduler.qos_rr_cursor, len(scheduler.paths), sim._EPS)
        gap = chosen.finish_tag - minimum
        assert chosen.finish_tag <= minimum + sim._EPS
        audit.max_selected_tag_gap = max(audit.max_selected_tag_gap, gap)
        audit.selection_calls += 1
        audit.candidate_set_sizes[len(candidates)] += 1
        audit.multi_path_opportunities += len(candidates) > 1
        changed = chosen.path_id != native_rr.path_id
        audit.cross_path_changed += changed
        scheduler._qos_finish_buckets[chosen.finish_tag].remove(chosen.path_id)
        for tag in tied_tags:
            if scheduler._qos_finish_buckets[tag]:
                heapq.heappush(scheduler._qos_finish_heap, tag)
            else:
                del scheduler._qos_finish_buckets[tag]
        path = scheduler.paths[chosen.path_id]
        path.virtual_finish = chosen.finish_tag
        scheduler._publish_qos_floor(path)
        scheduler.qos_rr_cursor = (path.path_id + 1) % len(scheduler.paths)
        flow = best_flows[chosen.path_id]
        moved = path.pending[0] is not flow
        if moved:
            path.pending.remove(flow)
            path.pending.appendleft(flow)
            audit.same_path_head_changed += 1
        if len(audit.examples) < 64 and (changed or moved):
            audit.examples.append({"ssu_id": scheduler.state.disk_id, "time_ms": now,
                "candidate_count": len(candidates), "native_rr_path": native_rr.path_id,
                "selected_path": path.path_id, "minimum_tag": minimum,
                "selected_tag": chosen.finish_tag, "request_id": flow.request_id,
                "layer": flow.layer, "block_idx": flow.block_idx,
                "same_path_head_changed": moved})
        audit.decision_wall_us += (perf_counter() - started) * 1e6
        return path

    with patch.object(sim.DiskIOScheduler, "enqueue_many", enqueue_many), \
         patch.object(sim.PathQueue, "enqueue", enqueue), \
         patch.object(sim.PathQueue, "activate_next", activate), \
         patch.object(sim.DiskIOScheduler, "_dispatch_one", dispatch_one), \
         patch.object(sim.DiskIOScheduler, "_select_qos_path", select):
        yield audit
