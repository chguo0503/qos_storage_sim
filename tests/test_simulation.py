import numpy as np
import pytest

from sim import (
    BATCH_DISPATCH,
    BW_TIER_CIR,
    BlockIOFlow,
    DiskIOScheduler,
    DiskState,
    IOSchedulingConfig,
    QOS_LAYOUT_EIGHT_GROUP,
    build_block_placement,
    calculate_token_partition,
    load_bw_table_cache,
    simulate_continuous,
)


SYNTHETIC_BW_TABLE = {
    (32, 64): (
        20.0,  # required bandwidth: maps to the middle QoS tier
        1_000.0,  # one millisecond of compute per layer
        10.0,
        0.02,  # half a millisecond when the only active queue borrows 40 GB/s
    )
}


def _run_small_simulation(**overrides):
    arguments = {
        "bw_table": SYNTHETIC_BW_TABLE,
        "num_npu": 1,
        "num_disk": 1,
        "n_layers": 2,
        "placement_mode": "roundrobin",
        "rng": np.random.RandomState(7),
    }
    arguments.update(overrides)
    return simulate_continuous(**arguments)


def test_bandwidth_loader_falls_back_to_project_data(tmp_path):
    table = load_bw_table_cache(results_dir=tmp_path, num_npu=999)

    assert len(table) == 84
    assert all(isinstance(key, tuple) and len(key) == 2 for key in table)
    assert all(len(values) == 4 for values in table.values())


def test_nql_tokens_are_excluded_from_ssu_block_placement():
    load = {
        "npu_id": 0,
        "seq_len_k": 1,
        "nql": 300,
        "per_layer_kv_gb": 0.000724,
    }

    assert calculate_token_partition(1, 300) == (1024, 300, 724)
    placement = build_block_placement(
        [load],
        np.random.RandomState(1),
        mode="roundrobin",
        n_layers=1,
        num_disk=2,
    )[0][0]

    assert len(placement) == 6
    assert [block["disk"] for block in placement] == [0, 1, 0, 1, 0, 1]
    assert [block["gb"] for block in placement[:5]] == pytest.approx([0.000128] * 5)
    assert placement[-1]["gb"] == pytest.approx(0.000084)
    assert sum(block["gb"] for block in placement) == pytest.approx(0.000724)


def test_nql_cannot_exceed_total_request_tokens():
    with pytest.raises(ValueError, match="nql cannot exceed"):
        calculate_token_partition(seq_len_k=1, nql=1025)


def test_request_with_all_tokens_on_npu_skips_ssu_io():
    table = {(1, 1024): (0.0, 1_000.0, 2.0, 0.0)}

    npus, summary = simulate_continuous(
        table,
        num_npu=1,
        num_disk=1,
        n_layers=2,
        rng=np.random.RandomState(3),
    )

    assert summary["makespan_ms"] == pytest.approx(2.0)
    assert summary["disk_stats"][0]["flows_enqueued"] == 0
    assert npus[0].ssu_read_tokens == 0
    assert npus[0].npu_compute_tokens == 1024
    assert npus[0].current_request_kv_actual_dur == {0: 0.0, 1: 0.0}


def test_qos_queues_borrow_surplus_by_wrr_weight():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(disk, policy="queue_wrr", disk_bw=40.0)
    event_heap = []
    flows = []

    for npu_id, demand_bw, queue_id in (
        (0, 200.0, 0),
        (0, 20.0, 8),
        (0, 5.0, 16),
    ):
        flow = BlockIOFlow(
            npu_id=npu_id,
            layer=0,
            block_idx=queue_id,
            disk_id=0,
            total_gb=1.0,
            bw=0.0,
            start_time=0.0,
            demand_bw=demand_bw,
            queue_id=queue_id,
        )
        scheduler.enqueue_many([flow], current_time=0.0)
        flows.append(flow)

    scheduler.redistribute(current_time=0.0, event_heap=event_heap)

    high, mid, low = flows
    assert high.bw == pytest.approx(23.3714285714)
    assert mid.bw == pytest.approx(11.1857142857)
    assert low.bw == pytest.approx(5.4428571429)
    assert low.bw > low.demand_bw
    assert (high.bw - BW_TIER_CIR[0]) / (mid.bw - BW_TIER_CIR[1]) == (
        pytest.approx(2.0)
    )
    assert sum(flow.bw for flow in flows) == pytest.approx(40.0)
    for flow in flows:
        queue = scheduler.queues[flow.queue_id]
        assert flow.bw >= queue.cir
        assert queue.assigned_bw == pytest.approx(flow.bw)


def test_single_qos_queue_can_borrow_the_full_disk_bandwidth():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(disk, policy="queue_wrr", disk_bw=40.0)
    flow = BlockIOFlow(
        npu_id=0,
        layer=0,
        block_idx=0,
        disk_id=0,
        total_gb=1.0,
        bw=0.0,
        start_time=0.0,
        demand_bw=200.0,
        queue_id=0,
    )

    scheduler.enqueue_many([flow], current_time=0.0)
    scheduler.redistribute(current_time=0.0, event_heap=[])

    assert scheduler.queues[0].pir == float("inf")
    assert scheduler.queues[0].assigned_bw == pytest.approx(40.0)
    assert flow.bw == pytest.approx(40.0)


def test_256_queues_use_eight_equal_weight_groups():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(
        disk,
        policy="queue_wrr",
        disk_bw=40.0,
        io_sched=IOSchedulingConfig(
            qos_queue_count=256,
            qos_layout=QOS_LAYOUT_EIGHT_GROUP,
        ),
    )

    assert len(scheduler.queues) == 256
    assert scheduler.group_queue_counts == (32,) * 8
    assert scheduler.group_weights == (1.0,) * 8
    assert scheduler.configured_queue_bandwidth == pytest.approx(40.0)
    assert {queue.group_id for queue in scheduler.queues.values()} == set(range(8))
    assert all(
        queue.cir == pytest.approx(0.15625) for queue in scheduler.queues.values()
    )

    mapping_flows = [
        BlockIOFlow(
            npu_id=npu_id,
            layer=layer,
            block_idx=0,
            disk_id=0,
            total_gb=1.0,
            bw=0.0,
            start_time=0.0,
            demand_bw=20.0,
        )
        for npu_id, layer in ((0, 0), (1, 0), (8, 0), (0, 1))
    ]
    assert [scheduler.queue_id_for_flow(flow) for flow in mapping_flows] == [
        0,
        32,
        1,
        1,
    ]

    active_flows = []
    for queue_id in (0, 32, 33):
        flow = BlockIOFlow(
            npu_id=queue_id,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=1.0,
            bw=0.0,
            start_time=0.0,
            demand_bw=1.0,
            queue_id=queue_id,
        )
        scheduler.enqueue_many([flow], current_time=0.0)
        active_flows.append(flow)

    scheduler.redistribute(current_time=0.0, event_heap=[])

    assert [flow.bw for flow in active_flows] == pytest.approx(
        [19.921875, 10.0390625, 10.0390625]
    )
    assert sum(flow.bw for flow in active_flows) == pytest.approx(40.0)


def test_fair_uses_one_path_for_all_competing_flows():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(disk, policy="fair", disk_bw=40.0)
    event_heap = []
    flows = [
        BlockIOFlow(
            npu_id=npu_id,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=1.0,
            bw=0.0,
            start_time=0.0,
            demand_bw=20.0,
            queue_id=0,
        )
        for npu_id in range(6)
    ]

    scheduler.enqueue_many(flows, current_time=0.0)
    scheduler.redistribute(current_time=0.0, event_heap=event_heap)

    assert set(scheduler.queues) == {0}
    queue = scheduler.queues[0]
    assert queue.max_depth == 4
    assert queue.active_flows == flows[:4]
    assert list(queue.pending) == flows[4:]
    assert queue.max_active_flows_observed == 4
    assert [flow.bw for flow in flows[:4]] == pytest.approx([10.0] * 4)
    assert sum(flow.bw for flow in flows[:4]) == pytest.approx(40.0)

    flows[0].remaining_gb = 0.0
    assert scheduler.complete_ready_flows(current_time=0.0) == [flows[0]]
    assert queue.active_flows == flows[1:5]
    assert list(queue.pending) == [flows[5]]


def test_qos_queue_activates_four_flows_then_keeps_fcfs_waiters():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(disk, policy="queue_wrr", disk_bw=40.0)
    flows = [
        BlockIOFlow(
            npu_id=npu_id,
            layer=0,
            block_idx=0,
            disk_id=0,
            total_gb=1.0,
            bw=0.0,
            start_time=0.0,
            demand_bw=200.0,
            queue_id=0,
        )
        for npu_id in range(5)
    ]

    scheduler.enqueue_many(flows, current_time=0.0)
    scheduler.redistribute(current_time=0.0, event_heap=[])

    queue = scheduler.queues[0]
    assert queue.max_depth == 4
    assert queue.active_flows == flows[:4]
    assert list(queue.pending) == [flows[4]]
    assert queue.max_active_flows_observed == 4
    assert [flow.bw for flow in flows[:4]] == pytest.approx([10.0] * 4)

    flows[0].remaining_gb = 0.0
    assert scheduler.complete_ready_flows(current_time=0.0) == [flows[0]]
    assert queue.active_flows == flows[1:]
    assert not queue.pending
    assert scheduler.outstanding_blocks == 4


def test_fair_aggregated_flows_preserve_virtual_flow_competition():
    disk = DiskState(disk_id=0, disk_bw=40.0)
    scheduler = DiskIOScheduler(disk, policy="fair", disk_bw=40.0)
    event_heap = []
    three_blocks = BlockIOFlow(
        npu_id=0,
        layer=0,
        block_idx=0,
        disk_id=0,
        total_gb=3.0,
        bw=0.0,
        start_time=0.0,
        demand_bw=20.0,
        queue_id=0,
        block_count=3,
    )
    one_block = BlockIOFlow(
        npu_id=1,
        layer=0,
        block_idx=0,
        disk_id=0,
        total_gb=1.0,
        bw=0.0,
        start_time=0.0,
        demand_bw=20.0,
        queue_id=0,
        block_count=1,
    )

    scheduler.enqueue_many([three_blocks, one_block], current_time=0.0)
    scheduler.redistribute(current_time=0.0, event_heap=event_heap)

    assert three_blocks.bw == pytest.approx(30.0)
    assert one_block.bw == pytest.approx(10.0)
    assert scheduler.n_flows_enqueued == 2
    assert scheduler.n_blocks_enqueued == 4


def test_fixed_queue_defaults_must_fit_the_physical_disk():
    disk = DiskState(disk_id=0, disk_bw=20.0)

    with pytest.raises(ValueError, match="exceeds physical disk bandwidth"):
        DiskIOScheduler(disk, policy="queue_wrr", disk_bw=20.0)


def test_prefetch_timing_matches_a_hand_calculated_example():
    npus, summary = _run_small_simulation(io_mode="prefetch")

    assert summary["completed_requests"] == 1
    assert summary["makespan_ms"] == pytest.approx(2.5)
    assert summary["avg_ttft_ms"] == pytest.approx(2.5)
    assert summary["fixed_qos_queue_bandwidth"] is False
    assert summary["qos_surplus_borrowing"] is True
    assert summary["qos_queue_pir_uncapped"] is True
    assert npus[0].current_request_kv_actual_dur == pytest.approx({0: 0.5, 1: 0.5})
    assert npus[0].current_request_io_waits == pytest.approx({0: 0.5, 1: 0.0})

    active_queues = {
        queue_id: queue
        for queue_id, queue in summary["disk_stats"][0]["queues"].items()
        if queue["activations"] > 0
    }
    assert set(active_queues) == {8, 9}
    assert all(
        queue["max_observed_bw_gbps"] == pytest.approx(40.0)
        for queue in active_queues.values()
    )
    assert summary["disk_stats"][0]["flows_enqueued"] == 4
    assert summary["disk_stats"][0]["blocks_enqueued"] == 512


def test_sequential_mode_waits_for_each_layer_io():
    _, prefetch = _run_small_simulation(io_mode="prefetch")
    _, sequential = _run_small_simulation(io_mode="sequential")

    assert prefetch["makespan_ms"] == pytest.approx(2.5)
    assert sequential["makespan_ms"] == pytest.approx(3.0)


def test_waiting_request_is_dispatched_after_the_npu_becomes_idle():
    npus, summary = _run_small_simulation(
        num_requests_per_npu=2,
        n_layers=1,
    )

    assert summary["completed_requests"] == 2
    assert summary["makespan_ms"] == pytest.approx(3.0)
    assert summary["avg_ttft_ms"] == pytest.approx(2.25)
    assert summary["avg_processing_ttft_ms"] == pytest.approx(1.5)
    metrics = sorted(summary["request_metrics"], key=lambda item: item["request_id"])
    assert [item["request_id"] for item in metrics] == [0, 1]
    assert [item["request_start_time"] for item in metrics] == pytest.approx([0.0, 1.5])
    assert [item["queueing_delay_ms"] for item in metrics] == pytest.approx([0.0, 1.5])
    assert npus[0].request_count == 2
    assert npus[0].ttft_list == pytest.approx([1.5, 3.0])


@pytest.mark.parametrize("dispatch_mode", ["batched", "traffic_aware_batched"])
def test_batched_dispatch_modes_finish_all_blocks(dispatch_mode):
    npus, summary = _run_small_simulation(
        trace=True,
        io_sched=IOSchedulingConfig(
            io_dispatch_mode=dispatch_mode,
            batch_size=8,
        ),
    )

    assert summary["completed_requests"] == 1
    assert summary["batch_dispatch_interval_us"] == pytest.approx(200.0)
    assert summary["event_counts"][BATCH_DISPATCH] == 62
    assert npus[0]._batches_dispatched == 64
    assert all(count == 0 for count in npus[0].pending_blocks.values())
    for layer in range(2):
        dispatch_times = npus[0].layer_trace[layer]["batch_dispatch_ms"]
        assert len(dispatch_times) == 32
        assert np.diff(dispatch_times) == pytest.approx([0.2] * 31)


@pytest.mark.parametrize("dispatch_mode", ["batched", "traffic_aware_batched"])
def test_fixed_batch_interval_allows_overlapping_batches(dispatch_mode):
    table = {(2, 0): (20.0, 1_000.0, 2.0, 0.032)}

    npus, summary = simulate_continuous(
        table,
        num_npu=1,
        num_disk=1,
        n_layers=1,
        rng=np.random.RandomState(11),
        trace=True,
        io_sched=IOSchedulingConfig(
            io_dispatch_mode=dispatch_mode,
            batch_size=8,
            batch_dispatch_interval_us=200.0,
        ),
    )

    assert summary["makespan_ms"] == pytest.approx(1.8)
    assert summary["event_counts"][BATCH_DISPATCH] == 1
    assert npus[0].layer_trace[0]["batch_dispatch_ms"] == pytest.approx([0.0, 0.2])
    queue_stats = summary["disk_stats"][0]["queues"][8]
    assert queue_stats["max_active_flows"] == 4
    assert queue_stats["max_active_flows_observed"] == 2
    assert summary["disk_stats"][0]["max_outstanding_blocks"] == 16


@pytest.mark.parametrize("dispatch_mode", ["batched", "traffic_aware_batched"])
def test_demand_aware_batch_interval_uses_payload_and_required_bw(dispatch_mode):
    table = {(2, 0): (20.0, 1_000.0, 2.0, 0.032)}

    npus, summary = simulate_continuous(
        table,
        num_npu=1,
        num_disk=1,
        n_layers=1,
        rng=np.random.RandomState(11),
        trace=True,
        io_sched=IOSchedulingConfig(
            io_dispatch_mode=dispatch_mode,
            batch_size=8,
            batch_interval_mode="demand_aware",
            batch_dispatch_headroom=1.1,
        ),
    )

    expected_interval_us = 0.016 / (20.0 * 1.1) * 1e6
    expected_second_dispatch_ms = expected_interval_us / 1000.0
    assert summary["batch_interval_mode"] == "demand_aware"
    assert summary["batch_dispatch_headroom"] == pytest.approx(1.1)
    assert npus[0].layer_trace[0]["batch_interval_us"] == pytest.approx(
        [expected_interval_us]
    )
    assert npus[0].layer_trace[0]["batch_dispatch_ms"] == pytest.approx(
        [0.0, expected_second_dispatch_ms]
    )
    assert summary["makespan_ms"] == pytest.approx(expected_second_dispatch_ms + 1.4)
