"""Controlled 10/30 GB/s calibration using the real command schedulers."""

from collections import Counter

from sim import (
    DISK_BW,
    POLICY_BASELINE_BYPASS,
    POLICY_QOS_DEMAND_MAXMIN,
    BlockIOFlow,
    DiskIOScheduler,
    DiskState,
)


IO_SIZE_GB = 0.001
COMMANDS_MEASURED = 40
BACKLOG_PER_NPU = 40


def _flow(npu_id, block_idx, demand_gbps):
    return BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=block_idx,
        disk_id=0,
        total_gb=IO_SIZE_GB,
        queue_id=-1,
        block_count=1,
        enqueue_time=0.0,
        demand_gbps=demand_gbps,
        deadline_time=0.0,
        layer_work_gb=BACKLOG_PER_NPU * IO_SIZE_GB,
    )


def _run_policy(policy):
    scheduler = DiskIOScheduler(DiskState(0), policy, DISK_BW)
    flows = [
        _flow(npu_id, npu_id * BACKLOG_PER_NPU + block_idx, demand)
        for npu_id, demand in enumerate((10.0, 30.0))
        for block_idx in range(BACKLOG_PER_NPU)
    ]
    scheduler.enqueue_many(flows, 0.0)
    selected = []
    current_time = 0.0
    for _ in range(COMMANDS_MEASURED):
        active = scheduler.dispatch(current_time, [], schedule_completion=False)
        selected.append(active.npu_id)
        current_time = active.end_time
        scheduler.complete_ready_flows(current_time)
    command_counts = Counter(selected)
    elapsed_s = current_time / 1000.0
    return {
        "policy": policy,
        "command_counts": [command_counts[0], command_counts[1]],
        "achieved_ssd_service_gbps": [
            command_counts[npu_id] * IO_SIZE_GB / elapsed_s
            for npu_id in (0, 1)
        ],
        "elapsed_ms": current_time,
        "command_trace": selected,
    }

def run_two_npu_calibration():
    return {
        "ssd_capacity_gbps": DISK_BW,
        "npu_demands_gbps": [10.0, 30.0],
        "io_size_gb": IO_SIZE_GB,
        "commands_measured": COMMANDS_MEASURED,
        "baseline": _run_policy(POLICY_BASELINE_BYPASS),
        "demand_maxmin": _run_policy(POLICY_QOS_DEMAND_MAXMIN),
    }
