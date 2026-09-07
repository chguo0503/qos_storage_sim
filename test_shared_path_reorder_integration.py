"""Exercise actual same-Path reordering on the native command scheduler.

A tiny full Prefill run constructs a real simulator context. Afterwards this
test injects a FIXED backend arrival/Path trace into its drained native SSD to
isolate the allowed queue capability: the client planner and layer-release
feedback are deliberately bypassed for these four extra commands. This is not
a utilization experiment or a client-ledger test. SSD commands themselves use
the real native enqueue, arbitration, dispatch, settle and completion methods.
"""

import pytest

import sim
from continuous_batch_sim import ContinuousBatchRequest, simulate_continuous_batch
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from shared_path_sim_adapter import IO_GIB, shared_path_adapter


def _prime_context(io_size=IO_GIB):
    request = ContinuousBatchRequest(
        request_id=1, npu_id=0, arrival_time_ms=0,
        load={"category": "SS", "per_layer_us": 10.0},
        placement=(((0, io_size),),))
    return simulate_continuous_batch(
        (request,), num_npu=4, num_ssu=1, n_layers=8, batch_size=1,
        policy=sim.POLICY_QOS_STATIC_CIR, qos_config=static_qos_config(),
        cross_request_layer0_prefetch=True,
        client_io_config=routing_strategy_specs()[1].client_config(),
        disk_bw_gbps=40, npu_bw_gbps=50, pressure_ttl_ms=5,
        submit_order_seed=17)


def _fixed_backend_trace(strategy):
    with shared_path_adapter(strategy) as adapter:
        summary = _prime_context()
        assert all(summary["invariants"].values())
        context = adapter.context
        scheduler = context.disks[0].scheduler
        assert not scheduler.state.active_flows
        assert scheduler.outstanding_blocks == 0
        before_bytes = scheduler.state.completed_bytes_gb
        before_enqueued = scheduler.blocks_enqueued
        now = context.current_time_ms
        unit_ms = 1000 * IO_GIB / 40
        cirs_before = tuple(path.cir for path in scheduler.paths.values())

        def command(rid, npu, path, arrival, deadline):
            return sim.BlockIOFlow(
                npu_id=npu, layer=1, block_idx=0, disk_id=0,
                total_gb=IO_GIB, queue_id=path, block_count=1,
                enqueue_time=arrival, request_id=rid,
                deadline_time=deadline, layer_work_gb=IO_GIB)

        # First command has already started before urgent commands arrive.
        active = command(1000, 0, 0, now, now + 100 * unit_ms)
        scheduler.enqueue_many((active,), now)
        events = []
        assert scheduler.dispatch(now, events) is active
        active_end = active.end_time
        arrival = now + unit_ms / 2
        queued = (
            command(1001, 1, 0, arrival, now + 80 * unit_ms),
            command(1002, 2, 0, arrival, now + .6 * unit_ms),
            # Even more urgent, but on a different Path: the original CIR
            # arbiter, not this deadline sorter, decides when Path32 is chosen.
            command(1003, 3, 32, arrival, now + .55 * unit_ms),
        )
        context.current_time_ms = arrival
        scheduler.enqueue_many(queued, arrival)
        event_count = len(events)
        scheduler.request_dispatch(arrival, events)
        assert len(events) == event_count  # busy SSD did not schedule preemption
        assert scheduler.complete_ready_flows(arrival) == []
        assert scheduler.state.active_flows == [active]
        assert active.end_time == active_end
        assert active.ssd_activation_time == now
        assert active.remaining_gb == pytest.approx(IO_GIB / 2, abs=1e-14)

        trace = [(active.request_id, active.queue_id, active.ssd_activation_time,
                  active.end_time)]
        context.current_time_ms = active_end
        assert scheduler.complete_ready_flows(active_end) == [active]
        timestamp = active_end
        while scheduler.outstanding_blocks:
            context.current_time_ms = timestamp
            flow = scheduler.dispatch(timestamp, events)
            assert scheduler.state.active_flows == [flow]
            assert flow.end_time - flow.ssd_activation_time == pytest.approx(unit_ms)
            trace.append((flow.request_id, flow.queue_id, flow.ssd_activation_time,
                          flow.end_time))
            timestamp = flow.end_time
            context.current_time_ms = timestamp
            assert scheduler.complete_ready_flows(timestamp) == [flow]

        assert scheduler.state.completed_bytes_gb - before_bytes == 4 * IO_GIB
        assert scheduler.blocks_enqueued - before_enqueued == 4
        assert scheduler.max_backend_active_io == 1
        assert not scheduler.state.active_flows
        assert all(path.pending_io_count == 0 and not path.pending
                   for path in scheduler.paths.values())
        assert tuple(path.cir for path in scheduler.paths.values()) == cirs_before
        for flow, expected_path in zip((active,) + queued, (0, 0, 0, 32)):
            assert flow.queue_id == expected_path
            assert flow.queue.path_id == expected_path
            assert flow.disk_id == 0
            assert flow.total_gb == IO_GIB
            assert flow.remaining_gb == 0
        return {"trace": tuple(trace), "changed": adapter.reorder_changed,
                "calls": adapter.reorder_calls, "active_end": active_end,
                "cir_writes": adapter.collector.cir_write_events}


def test_native_same_path_service_changes_without_preemption_or_path_migration():
    ordinary = _fixed_backend_trace("strategy1")
    reordered = _fixed_backend_trace("strategy2")
    old_path0 = [row[0] for row in ordinary["trace"] if row[1] == 0]
    new_path0 = [row[0] for row in reordered["trace"] if row[1] == 0]
    assert old_path0 == [1000, 1001, 1002]
    assert new_path0 == [1000, 1002, 1001]
    assert reordered["changed"] > 0
    assert reordered["calls"] > 0
    assert ordinary["calls"] == 0
    assert ordinary["active_end"] == reordered["active_end"]
    old_finish = {rid: end for rid, _path, _start, end in ordinary["trace"]}
    new_finish = {rid: end for rid, _path, _start, end in reordered["trace"]}
    assert new_finish[1002] < old_finish[1002]
    assert new_finish[1001] > old_finish[1001]
    assert new_finish[1000] == old_finish[1000]


def test_fixed_equal_size_trace_keeps_native_cross_path_arbitration_sequence():
    ordinary = _fixed_backend_trace("strategy1")
    reordered = _fixed_backend_trace("strategy2")
    # In this fixed-arrival backend trace the Path schedule really is equal.
    # Do not generalize that equality to closed-loop inference, where changed
    # completion times legitimately change later-layer arrivals and Path loads.
    assert tuple((p, start, end) for _, p, start, end in ordinary["trace"]) == \
           tuple((p, start, end) for _, p, start, end in reordered["trace"])
    assert ordinary["cir_writes"] == reordered["cir_writes"] == []


@pytest.mark.parametrize("io_size", (IO_GIB / 2, IO_GIB * 2))
def test_shared_adapter_rejects_non_176kib_io_before_native_arbitration(io_size):
    with shared_path_adapter("strategy2"):
        with pytest.raises(ValueError, match="equal 176 KiB"):
            _prime_context(io_size)
