"""Shared result and plotting helpers for the final experiment."""

from __future__ import annotations

import math

import sim


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def compact_summary(full):
    """Keep the metrics needed by validation and the final report."""
    disks = full["disk_stats"]
    links = full["npu_link_stats"]
    blocks = sum(disk["blocks_enqueued"] for disk in disks)
    queue_wait = sum(disk["total_queue_wait_ms"] for disk in disks)
    link_dispatches = sum(link["dispatches"] for link in links)
    link_queue_wait = sum(
        link["avg_queue_wait_ms"] * link["dispatches"] for link in links
    )
    conservation = full["block_conservation"]
    invariants = {
        "requests_completed": full["completed_requests"]
        == full["total_requests"],
        "blocks_conserved": conservation["expected"]
        == conservation["submitted"]
        == conservation["completed"],
        "placement_preserved": conservation["placement_targets_preserved"],
        "bytes_conserved": math.isclose(
            conservation["expected_read_gb"],
            conservation["ssd_completed_read_gb"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        )
        and math.isclose(
            conservation["expected_read_gb"],
            conservation["completed_read_gb"],
            rel_tol=1e-10,
            abs_tol=1e-9,
        ),
        "npu_cap_respected": full["npu_link_peak_effective_bw_gbps"]
        <= sim.NPU_BW_LIMIT + 1e-9,
        "single_backend_io": max(
            (disk["max_backend_active_io"] for disk in disks), default=0
        )
        <= 1,
        "single_npu_link_io": max(
            (link["max_active_io"] for link in links), default=0
        )
        <= 1,
        "queues_drained": all(
            disk["outstanding_blocks"] == 0 for disk in disks
        )
        and all(link["outstanding_io"] == 0 for link in links),
    }
    return {
        "policy": full["policy"],
        "backend_model": full["backend_model"],
        "data_plane_stages": full["data_plane_stages"],
        "backend_capacity_gbps": full["backend_capacity_gbps"],
        "npu_bw_limit_gbps": full["npu_bw_limit_gbps"],
        "avg_request_compute_fraction": full[
            "avg_request_compute_fraction"
        ],
        "fleet_npu_compute_utilization": full[
            "fleet_npu_compute_utilization"
        ],
        "request_compute_fraction_jain": full[
            "request_compute_fraction_jain"
        ],
        "makespan_ms": full["makespan_ms"],
        "throughput_requests_per_s": full["throughput_requests_per_s"],
        "category_metrics": full["category_metrics"],
        "npu_link_utilization": full["npu_link_utilization"],
        "npu_link_peak_raw_bw_gbps": full["npu_link_peak_raw_bw_gbps"],
        "npu_link_peak_effective_bw_gbps": full[
            "npu_link_peak_effective_bw_gbps"
        ],
        "npu_link_cap_hit_fraction": full["npu_link_cap_hit_fraction"],
        "npu_link_busy_fraction": full["npu_link_busy_fraction"],
        "avg_npu_link_queue_wait_ms": (
            link_queue_wait / link_dispatches if link_dispatches else 0.0
        ),
        "max_npu_link_queue_wait_ms": max(
            (link["max_queue_wait_ms"] for link in links), default=0.0
        ),
        "max_npu_link_outstanding_io": max(
            (link["max_outstanding_io"] for link in links), default=0
        ),
        "ssu_active_time_utilization": _mean(
            [disk["active_time_utilization"] for disk in disks]
        ),
        "ssu_effective_bandwidth_utilization": _mean(
            [disk["effective_bandwidth_utilization"] for disk in disks]
        ),
        "avg_queue_wait_ms_per_block": (
            queue_wait / blocks if blocks else 0.0
        ),
        "max_queue_wait_ms": max(
            (disk["max_queue_wait_ms"] for disk in disks), default=0.0
        ),
        "max_path_outstanding_io": max(
            (disk["max_path_outstanding_io"] for disk in disks), default=0
        ),
        "enqueued_path_ids": sorted(
            {
                path_id
                for disk in disks
                for path_id in disk.get("enqueued_path_ids", ())
            }
        ),
        "pressure_reports": sum(
            disk["pressure_reports"] for disk in disks
        ),
        "pressure_telemetry_mb": sum(
            disk["pressure_reports"] for disk in disks
        )
        * 256
        * 4
        / 1_000_000,
        "backend_dispatches": sum(
            disk["backend_dispatches"] for disk in disks
        ),
        "blocks_enqueued": blocks,
        "workload_fingerprint": full["workload_fingerprint"],
        "placement_hash": full["placement_hash"],
        "block_conservation": conservation,
        "invariants": invariants,
        "request_metrics": full["request_metrics"],
    }


def plot_axis(axis, data):
    """Plot SSU series on one axis with the original experiment style."""
    ssus = data["ssus"]
    for series in data["series"]:
        plot_kwargs = dict(series.get("plot_kwargs", {}))
        axis.plot(
            ssus,
            series["values"],
            series.get("style", "o-"),
            linewidth=2.2,
            label=series["label"],
            **plot_kwargs,
        )
    axis.set(
        xlabel="Number of SSUs",
        ylabel=data.get("ylabel", "Average NPU Utilization (%)"),
        xticks=ssus,
        ylim=(0, 100),
        title=data.get(
            "title", "16-layer routing on shared SSD→NPU data plane"
        ),
    )
    axis.grid(alpha=0.3)
    legend = dict(data.get("legend", {}))
    if legend.pop("row_major", False):
        handles, labels = axis.get_legend_handles_labels()
        columns = legend.get("ncol", 1)
        rows = math.ceil(len(handles) / columns)
        order = [
            row * columns + column
            for column in range(columns)
            for row in range(rows)
            if row * columns + column < len(handles)
        ]
        handles = [handles[index] for index in order]
        labels = [labels[index] for index in order]
        axis.legend(handles, labels, **legend)
    else:
        axis.legend(**legend)


def plot_results(data, output_path):
    """Plot one or more SSU utilization series with the original style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    plot_axis(axis, data)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
