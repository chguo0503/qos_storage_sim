"""Global client grants plus optional within-disk / across-disk coflow order.

The shared 5-ms collector, original Once engine and native data plane are
reused unchanged. Only client-known arrived/activated work enters decisions.
Use separate processes for simulations because adapters are process-local.
"""

from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from time import perf_counter
from unittest.mock import patch

import continuous_batch_sim as native
import sim
from coflow_client_policy import ClientRead, choose_npu, plan_global_batch
from shared_path_sim_adapter import shared_path_adapter


class GlobalClient:
    def __init__(self, shared, policy, queue_window_ms, assignment, joint_rule):
        self.shared = shared
        self.policy = policy
        self.queue_window_ms = queue_window_ms
        self.assignment = assignment
        self.joint_rule = joint_rule
        self.assignment_log = []
        self.reads = {}
        self.states = {}
        self.grants = deque()
        self.next_issue_ms = 0.0
        self.blocked = False
        self.sent_by_ssu = []
        self.sent_by_npu = []
        self.backend = None
        self.plans = 0
        self.empty_plans = 0
        self.issued_commands = 0
        self.max_sent_by_ssu = []
        self.max_sent_by_npu = []
        self.activation_count = 0
        self.delayed_activation_count = 0
        self.first_issue_examples = []
        self.plan_wall_us = 0.0
        self.assignment_wall_us = 0.0
        self.ack_wall_us = 0.0

    @property
    def context(self):
        return self.shared.context

    def statistics(self):
        stats = self.shared.statistics()
        stats.update(strategy=self.policy, assignment_count=len(self.assignment_log),
                     assignment_wall_us=self.assignment_wall_us)
        stats["global_client"] = {
            "queue_window_ms": self.queue_window_ms, "grant_batch_size": 64,
            "assignment": self.assignment, "plans": self.plans,
            "empty_plans": self.empty_plans, "issued_commands": self.issued_commands,
            "activation_count": self.activation_count,
            "delayed_activation_count": self.delayed_activation_count,
            "first_issue_examples": self.first_issue_examples,
            "max_sent_unacked_by_ssu": self.max_sent_by_ssu,
            "max_sent_unacked_by_npu": self.max_sent_by_npu,
            "final_sent_unacked_by_ssu": self.sent_by_ssu,
            "final_sent_unacked_by_npu": self.sent_by_npu,
            "plan_wall_us": self.plan_wall_us, "ack_wall_us": self.ack_wall_us,
            "grant_execution_interval_us": self.shared.context.client_io_config.issue_interval_us,
            "counter_source": "actual client sends minus HBM ACKs; not SSD queue reads",
            "queued_io_migration": False, "future_requests_visible": False,
        }
        stats["global_coflow"] = {
            "enabled": self.policy == "strategy3",
            "priority_rule": self.joint_rule,
            "progress_source": "activated layer counts minus client HBM ACKs",
            "remaining_coflows_at_end": len(self.reads),
            "instant_other_ssu_queue_access": False,
            "coordination_latency_modeled_ms": 0,
        }
        stats["backend_arbitration"] = self.backend.statistics() if self.backend else {}
        return stats

    def global_priority(self, flow, now_ms):
        from coflow_joint_policy import coflow_priority
        read = self.reads[(flow.request_id, flow.layer)]
        return coflow_priority(read["deadline"], tuple(read["remaining"]), now_ms,
                               rule=self.joint_rule)


@contextmanager
def coflow_adapter(policy="strategy1", collector_interval_ms=5.0,
                   cir_min_interval_ms=100.0, queue_window_ms=0.25,
                   assignment="compute", joint_rule="least_slack"):
    if policy not in ("strategy1", "strategy2", "strategy3"):
        raise ValueError("expected strategy1/strategy2/strategy3")
    with shared_path_adapter("new_once", collector_interval_ms, cir_min_interval_ms) as shared:
        client = GlobalClient(shared, policy, queue_window_ms, assignment, joint_rule)
        old_init = native._Context.__init__
        old_arrival = native._handle_arrival
        old_start = native._start_layer_io
        old_submit = native._register_submit
        old_ack = native._register_complete

        def initialize(context, *args, **kwargs):
            old_init(context, *args, **kwargs)
            if context.batch_size != 1 or context.client_io_config.submit_batch_size != 1:
                raise ValueError("global grants require batch=1 and one 176-KiB I/O per grant")
            client.sent_by_ssu = [0] * context.num_ssu
            client.sent_by_npu = [0] * context.num_npu
            client.max_sent_by_ssu = [0] * context.num_ssu
            client.max_sent_by_npu = [0] * context.num_npu

        def arrival(context, rid, now_ms):
            started = perf_counter()
            request = context.requests[rid]
            original = request.manifest
            backlogs = shared.backlogs(now_ms)
            assigned = (original.npu_id if assignment == "fixed" else choose_npu(
                tuple(n.remaining_compute_ms for n in backlogs), original.npu_id,
                tuple(n.unfinished_request_count for n in backlogs)))
            if assigned != original.npu_id:
                context.npus[original.npu_id].future_arrivals -= 1
                context.npus[assigned].future_arrivals += 1
                request.manifest = replace(original, npu_id=assigned,
                    load={**original.load, "npu_id": assigned})
            elapsed = (perf_counter() - started) * 1e6
            client.assignment_wall_us += elapsed
            client.assignment_log.append({"request_id": rid, "arrival_time_ms": now_ms,
                "original_npu_id": original.npu_id, "assigned_npu_id": assigned,
                "snapshot_time_ms": shared.collector.sampled_at_ms,
                "scores_ms": tuple(n.remaining_compute_ms for n in backlogs),
                "decision_wall_us": elapsed})
            old_arrival(context, rid, now_ms)

        def schedule(context):
            if not context.submission_states or client.blocked:
                return
            # This adapter creates states at activation (no per-NPU JIT),
            # and its serial issue clock dominates every lane issue clock.
            # Thus max(now, global clock) already bounds all eligible states;
            # scanning every SSU component on every send/ACK is unnecessary.
            when = max(context.current_time_ms, client.next_issue_ms)
            pending = context.pending_client_event_ms
            if pending is not None and pending <= when + 1e-12:
                return
            context.client_event_generation += 1
            context.pending_client_event_ms = when
            context.push_event(when, sim.CLIENT_SUBMISSION, 0,
                               generation=context.client_event_generation)

        def start(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs):
            if not 0 <= layer < context.n_layers or request.io_started[layer]:
                return old_start(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs)
            before = context.next_submission_id
            result = old_start(context, request, layer, now_ms, deadline_ms, demand_window_ms, **kwargs)
            states = [context.submission_states[s] for s in range(before, context.next_submission_id)]
            if states:
                counts = [0] * context.num_ssu
                rid = request.manifest.request_id
                for state in states:
                    counts[state.disk_id] = len(state.blocks)
                    client.states[(rid, layer, state.disk_id)] = state.state_id
                client.reads[(rid, layer)] = {"npu": request.manifest.npu_id,
                    "deadline": deadline_ms, "remaining": counts.copy(), "unsent": counts.copy(),
                    "activation": now_ms, "first_issue": None}
                client.activation_count += 1
                # A newly activated urgent coflow can displace unissued grants.
                client.grants.clear()
                client.blocked = False
                schedule(context)
            return result

        def register_submit(context, flow):
            read = client.reads[(flow.request_id, flow.layer)]
            s, n = flow.disk_id, flow.npu_id
            read["unsent"][s] -= 1
            client.sent_by_ssu[s] += 1
            client.sent_by_npu[n] += 1
            client.max_sent_by_ssu[s] = max(client.max_sent_by_ssu[s], client.sent_by_ssu[s])
            client.max_sent_by_npu[n] = max(client.max_sent_by_npu[n], client.sent_by_npu[n])
            if read["first_issue"] is None:
                read["first_issue"] = flow.enqueue_time
                delay = flow.enqueue_time - read["activation"]
                client.delayed_activation_count += delay > 1e-9
                if len(client.first_issue_examples) < 128:
                    client.first_issue_examples.append({"request_id": flow.request_id,
                        "layer": flow.layer, "npu_id": n, "activation_ms": read["activation"],
                        "deadline_ms": read["deadline"], "first_issue_ms": flow.enqueue_time,
                        "delay_ms": delay})
            client.issued_commands += 1
            old_submit(context, flow)

        def acknowledge(context, flow):
            old_ack(context, flow)
            started = perf_counter()
            key = (flow.request_id, flow.layer)
            client.reads[key]["remaining"][flow.disk_id] -= 1
            client.sent_by_ssu[flow.disk_id] -= 1
            client.sent_by_npu[flow.npu_id] -= 1
            assert min(client.sent_by_ssu[flow.disk_id], client.sent_by_npu[flow.npu_id]) >= 0
            if not any(client.reads[key]["remaining"]):
                del client.reads[key]
            client.blocked = False
            schedule(context)
            client.ack_wall_us += (perf_counter() - started) * 1e6

        def submit_event(context, generation, now_ms):
            if generation != context.client_event_generation:
                context.stale_events += 1
                return
            context.pending_client_event_ms = None
            if not client.grants:
                started = perf_counter()
                views = [ClientRead(rid, layer, r["npu"], r["deadline"],
                                    tuple(r["unsent"]), tuple(r["remaining"]))
                         for (rid, layer), r in client.reads.items() if any(r["unsent"])]
                client.grants.extend(plan_global_batch(views, client.sent_by_ssu,
                    client.sent_by_npu, now_ms, queue_window_ms=queue_window_ms,
                    disk_bw_gib_s=context.disk_bw_gbps, link_bw_gib_s=context.npu_bw_gbps))
                client.plans += 1
                client.plan_wall_us += (perf_counter() - started) * 1e6
            if not client.grants:
                client.empty_plans += 1
                client.blocked = True
                return
            rid, layer, n, s = client.grants[0]
            state_id = client.states[(rid, layer, s)]
            state = context.submission_states[state_id]
            eligible = max(state.ready_time_ms, context.client_next_issue_ms[n], client.next_issue_ms)
            if now_ms + 1e-12 < eligible:
                client.next_issue_ms = eligible
                schedule(context)
                return
            client.grants.popleft()
            finished = native._submit_one_client_batch(context, state, now_ms)
            interval = context.client_io_config.issue_interval_us / 1000
            client.next_issue_ms = now_ms + interval
            context.client_next_issue_ms[n] = now_ms + interval
            if finished:
                del context.submission_states[state_id]
                context.submission_queues[n].remove(state_id)
                del client.states[(rid, layer, s)]
                if not context.submission_queues[n]:
                    del context.submission_queues[n]
            schedule(context)

        if policy == "strategy1":
            backend_context = nullcontext(None)
        else:
            from coflow_disk_adapter import coflow_disk_adapter
            backend_context = coflow_disk_adapter(mode=policy,
                get_global_priority=client.global_priority if policy == "strategy3" else None)
        with patch.object(native._Context, "__init__", initialize), \
             patch.object(native, "_handle_arrival", arrival), \
             patch.object(native, "_schedule_client_event", schedule), \
             patch.object(native, "_start_layer_io", start), \
             patch.object(native, "_register_submit", register_submit), \
             patch.object(native, "_register_complete", acknowledge), \
             patch.object(native, "_handle_client_submission", submit_event), \
             backend_context as backend:
            client.backend = backend
            yield client
