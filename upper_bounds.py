"""Optimistic no-contention bounds for the shared SSD→NPU data plane."""

from __future__ import annotations

from collections import defaultdict

from sim import npu_link_service_time_ms, ssd_command_service_time_ms


def isolated_layer_io_time_ms(blocks, disk_bw_gbps=40.0, npu_bw_gbps=50.0):
    """Layer completion time when one request owns every SSD and its NPU link."""
    disk_free = defaultdict(float)
    ssd_completions = []
    for block_idx, (disk_id, size_gb) in enumerate(blocks):
        disk_free[disk_id] += ssd_command_service_time_ms(size_gb, disk_bw_gbps)
        ssd_completions.append((disk_free[disk_id], block_idx, disk_id, size_gb))

    link_free = 0.0
    for ssd_end, block_idx, disk_id, size_gb in sorted(ssd_completions):
        link_free = max(link_free, ssd_end)
        link_free += npu_link_service_time_ms(size_gb, npu_bw_gbps)
    return link_free


def fluid_layer_io_lower_bound_ms(
    blocks, disk_bw_gbps=40.0, npu_bw_gbps=50.0
):
    """Optimistic layer time after removing command and cross-NPU contention.

    Each SSD may stream its assigned bytes immediately and the NPU link may consume
    bytes from time zero.  The larger of the busiest SSD work and aggregate NPU-link
    work is therefore a lower bound on every realizable command schedule.
    """
    bytes_by_disk = defaultdict(float)
    total_gb = 0.0
    for disk_id, size_gb in blocks:
        bytes_by_disk[disk_id] += size_gb
        total_gb += size_gb
    busiest_ssd_ms = max(bytes_by_disk.values()) / disk_bw_gbps * 1000.0
    npu_link_ms = total_gb / npu_bw_gbps * 1000.0
    return max(busiest_ssd_ms, npu_link_ms)


def isolated_no_contention_bound(prepared_inputs):
    """Compute a fluid upper bound while preserving placement and both capacities."""
    rows = []
    loads = {load["request_id"]: load for load in prepared_inputs.request_loads}
    for request_id, layers in prepared_inputs.placement_by_request.items():
        load = loads[request_id]
        compute_ms = load["per_layer_us"] / 1000.0
        previous_compute_end = float(load["arrival_time"])
        total_wait_ms = 0.0
        for layer in range(prepared_inputs.n_layers):
            io_start = (
                float(load["arrival_time"])
                if layer == 0
                else previous_compute_end - compute_ms
            )
            ready = io_start + fluid_layer_io_lower_bound_ms(layers[layer])
            compute_start = max(previous_compute_end, ready)
            total_wait_ms += compute_start - previous_compute_end
            previous_compute_end = compute_start + compute_ms
        total_compute_ms = prepared_inputs.n_layers * compute_ms
        rows.append(
            {
                "request_id": request_id,
                "category": load["category"],
                "compute_fraction_upper_bound": total_compute_ms
                / (total_compute_ms + total_wait_ms),
                "io_wait_lower_bound_ms": total_wait_ms,
            }
        )
    return {
        "name": "fluid_no_inter_npu_contention_upper_bound",
        "relaxations": [
            "remove_inter_npu_ssd_contention",
            "allow_fluid_ssd_to_npu_streaming_from_time_zero",
            "ignore_nonpreemptive_command_boundaries",
        ],
        "avg_request_compute_fraction_upper_bound": sum(
            row["compute_fraction_upper_bound"] for row in rows
        )
        / len(rows),
        "request_bounds": rows,
    }
