from dataclasses import replace

import pytest

import sim
from coflow_disk_adapter import coflow_disk_adapter
from continuous_batch_sim import ContinuousBatchRequest, simulate_continuous_batch
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from shared_path_common import IO_GIB
from shared_path_sim_adapter import shared_path_adapter


def scheduler(ssu=0, qos=None):
    return sim.DiskIOScheduler(sim.DiskState(ssu), sim.POLICY_QOS_STATIC_CIR,
                               40, qos or static_qos_config())


def flow(rid, path, deadline, *, ssu=0, block=0, size=IO_GIB):
    return sim.BlockIOFlow(npu_id=rid % 4, layer=1, block_idx=block,
        disk_id=ssu, total_gb=size, queue_id=path, block_count=1,
        enqueue_time=0.0, request_id=rid, deadline_time=deadline,
        layer_work_gb=4 * IO_GIB)


def drain(disk, now=0):
    trace = []
    events = []
    while disk.outstanding_blocks:
        command = disk.dispatch(now, events)
        trace.append((command.request_id, command.block_idx, command.queue_id,
                      command.ssd_activation_time, command.end_time))
        now = command.end_time
        assert disk.complete_ready_flows(now) == [command]
    return trace


def test_real_cross_path_choice_changes_native_rr_without_changing_qos_tag():
    with coflow_disk_adapter() as audit:
        disk = scheduler()
        commands = (flow(1, 0, 20), flow(2, 32, 1))
        disk.enqueue_many(commands, 0)
        trace = drain(disk)
        stats = audit.statistics()
    assert trace[0][0] == 2  # Native equal-tag RR would choose Path0/request1.
    assert stats["cross_path_changed_vs_native_rr"] >= 1
    assert stats["multi_path_opportunities"] >= 1
    assert stats["candidate_set_size_histogram"][2] >= 1
    assert stats["max_selected_tag_gap"] <= sim._EPS
    assert stats["cir_pir_unchanged"]
    assert disk.state.completed_bytes_gb == 2 * IO_GIB
    assert all(command.queue_id == path for command, path in zip(commands, (0, 32)))


def test_same_path_pending_can_reorder_even_without_cross_path_tie():
    with coflow_disk_adapter() as audit:
        disk = scheduler()
        disk.enqueue_many((flow(1, 0, 20), flow(2, 0, 1)), 0)
        trace = drain(disk)
        stats = audit.statistics()
    assert [row[0] for row in trace] == [2, 1]
    assert stats["same_path_pending_head_changed"] == 1
    assert stats["cross_path_changed_vs_native_rr"] == 0


def test_higher_virtual_tag_cannot_jump_ahead_even_if_deadline_is_earlier():
    with coflow_disk_adapter() as audit:
        disk = scheduler()
        disk.enqueue_many((flow(1, 0, 20), flow(2, 32, 0)), 0)
        disk.paths[32].virtual_finish = 1.0
        trace = drain(disk)
    assert trace[0][0] == 1
    assert audit.statistics()["max_selected_tag_gap"] <= sim._EPS


def test_active_command_is_not_preempted_by_new_urgent_work():
    with coflow_disk_adapter() as audit:
        disk = scheduler()
        active = flow(1, 0, 20)
        disk.enqueue_many((active,), 0)
        events = []
        assert disk.dispatch(0, events) is active
        end = active.end_time
        urgent = flow(2, 32, 0)
        urgent.enqueue_time = end / 2
        disk.enqueue_many((urgent,), end / 2)
        assert disk.dispatch(end / 2, events) is active
        assert active.end_time == end and active.ssd_activation_time == 0
        assert disk.complete_ready_flows(end) == [active]
        trace = drain(disk, end)
    assert trace[0][0] == 2
    assert audit.statistics()["active_guard_calls"] == 1
    assert disk.max_backend_active_io == 1


def test_equal_priorities_reproduce_native_rr_trace_and_virtual_accounting():
    def run():
        disk = scheduler()
        disk.enqueue_many(tuple(flow(100 * p + b, p, 10, block=b)
                                for p in (0, 32, 64, 96) for b in range(3)), 0)
        trace = drain(disk)
        return trace, tuple(p.virtual_finish for p in disk.paths.values())
    expected = run()
    with coflow_disk_adapter() as audit:
        actual = run()
    assert actual == expected
    assert audit.statistics()["cross_path_changed_vs_native_rr"] == 0
    assert audit.statistics()["same_path_pending_head_changed"] == 0


def test_s3_coordinates_both_disks_using_only_supplied_coflow_map():
    priorities = {(1, 1): (0,), (2, 1): (1,)}
    def priority(command, now):
        return priorities[command.request_id, command.layer]
    with coflow_disk_adapter("strategy3", priority) as audit:
        disks = (scheduler(0), scheduler(1))
        disks[0].enqueue_many((flow(2, 0, 1, ssu=0), flow(1, 32, 20, ssu=0)), 0)
        disks[1].enqueue_many((flow(1, 0, 20, ssu=1), flow(2, 32, 1, ssu=1)), 0)
        traces = [drain(disk) for disk in disks]
    assert [trace[0][0] for trace in traces] == [1, 1]
    stats = audit.statistics()
    assert stats["global_priority_calls"] > 0
    assert stats["cir_pir_unchanged"]
    assert stats["max_selected_tag_gap"] <= sim._EPS


def test_changed_caller_progress_key_is_seen_at_next_decision():
    priorities = {1: (0,), 2: (1,)}
    with coflow_disk_adapter("strategy3", lambda command, now: priorities[command.request_id]):
        disk = scheduler()
        disk.enqueue_many((flow(1, 0, 10, block=0), flow(1, 0, 10, block=1),
                           flow(2, 0, 10)), 0)
        first = disk.dispatch(0, [])
        assert first.request_id == 1
        assert disk.complete_ready_flows(first.end_time) == [first]
        priorities[2] = (-1,)
        remaining = drain(disk, first.end_time)
    assert [row[0] for row in remaining] == [2, 1]


def test_illegal_4k_io_is_rejected_before_any_queue_mutation():
    with coflow_disk_adapter():
        disk = scheduler()
        with pytest.raises(ValueError, match="176 KiB"):
            disk.enqueue_many((flow(1, 0, 1), flow(2, 32, 0, size=4096 / 2**30)), 0)
        assert disk.outstanding_blocks == 0
        assert not any(path.pending for path in disk.paths.values())


def test_native_finite_pir_limitation_is_not_silently_bypassed():
    qos = static_qos_config()
    qos = replace(qos, path_pirs=(1.0,) + qos.path_pirs[1:])
    with coflow_disk_adapter():
        with pytest.raises(ValueError, match="PIR"):
            scheduler(qos=qos)


@pytest.mark.parametrize("mode,ssus,npus", (("strategy2", 6, 4), ("strategy3", 7, 4),
                                          ("strategy2", 6, 32), ("strategy3", 7, 32)))
def test_native_client_composition_conserves_io_and_periodic_reads(mode, ssus, npus):
    requests = tuple(ContinuousBatchRequest(
        request_id=npu * 100 + generation, npu_id=npu,
        arrival_time_ms=generation * 5,
        load={"category": "SS", "per_layer_us": 2000.0, "seq_len_k": 32, "nql": 128},
        placement=(tuple((block % ssus, IO_GIB) for block in range(32)),))
        for npu in range(npus) for generation in range(2))
    # This callback is deliberately just a known-metadata map substitute;
    # it does not read another SSD or claim a workload performance benefit.
    key = lambda command, now: (command.deadline_time, command.request_id)
    with shared_path_adapter("new_once") as client, coflow_disk_adapter(mode, key) as disk:
        result = simulate_continuous_batch(requests, num_npu=npus, num_ssu=ssus,
            n_layers=8, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=static_qos_config(), cross_request_layer0_prefetch=True,
            client_io_config=routing_strategy_specs()[1].client_config(),
            disk_bw_gbps=40, npu_bw_gbps=50, pressure_ttl_ms=5, submit_order_seed=17)
        host_stats = client.statistics()
        disk_stats = disk.statistics()
    assert all(result["invariants"].values())
    assert host_stats["reserved_blocks"] == host_stats["acknowledged_blocks"] == npus * 2 * 8 * 32
    assert host_stats["ledger_end_counts_by_ssu"] == [0] * ssus
    assert host_stats["fresh_reads_by_ssu"] == host_stats["native_fresh_reads_by_ssu"]
    assert host_stats["cir_write_events"] == []
    assert disk_stats["enqueue_commands"] == npus * 2 * 8 * 32
    assert disk_stats["cir_pir_unchanged"]
    assert disk_stats["selection_calls"] == npus * 2 * 8 * 32
    assert disk_stats["max_selected_tag_gap"] <= sim._EPS
