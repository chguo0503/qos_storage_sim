"""Two-NPU 10/30 calibration using the real SSD command schedulers."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim


IO_SIZE_GB = 0.001
COMMANDS_MEASURED = 40
BACKLOG_PER_NPU = 40
NPU_DEMANDS_GBPS = (10.0, 30.0)
DEFAULT_OUTPUT_DIR = Path("results/joint_dynamic_cir")


def _flow(npu_id, block_idx, path_id):
    return sim.BlockIOFlow(
        npu_id=npu_id,
        request_id=npu_id,
        layer=0,
        block_idx=block_idx,
        disk_id=0,
        total_gb=IO_SIZE_GB,
        queue_id=path_id,
        block_count=1,
        enqueue_time=0.0,
    )


def _measure_scheduler(scheduler, controller=None):
    trace = []
    current_time = 0.0
    for command_index in range(COMMANDS_MEASURED):
        active = scheduler.dispatch(
            current_time,
            [],
            schedule_completion=False,
        )
        trace.append(
            {
                "command_index": command_index,
                "npu_id": active.npu_id,
                "start_ms": current_time,
                "end_ms": active.end_time,
                "command_service_gbps": active.bw,
                "ssd_active_io": len(scheduler.state.active_flows),
            }
        )
        current_time = active.end_time
        completed = scheduler.complete_ready_flows(current_time)
        if controller is not None:
            controller.complete_ssd(completed[0])

    command_counts = Counter(item["npu_id"] for item in trace)
    elapsed_s = current_time / 1000.0
    command_rates = sorted(
        {item["command_service_gbps"] for item in trace}
    )
    active_counts = sorted({item["ssd_active_io"] for item in trace})
    return {
        "policy": scheduler.policy,
        "command_counts": [command_counts[npu_id] for npu_id in (0, 1)],
        "measured_ssd_service_gbps": [
            round(command_counts[npu_id] * IO_SIZE_GB / elapsed_s, 12)
            for npu_id in (0, 1)
        ],
        "elapsed_ms": current_time,
        "command_service_gbps_unique": command_rates,
        "ssd_active_io_unique": active_counts,
        "max_ssd_active_io": scheduler.max_backend_active_io,
        "command_trace": trace,
    }


def _run_baseline():
    scheduler = sim.DiskIOScheduler(
        sim.DiskState(0),
        sim.POLICY_BASELINE_BYPASS,
        sim.DISK_BW,
    )
    scheduler.enqueue_many(
        [
            _flow(
                npu_id,
                npu_id * BACKLOG_PER_NPU + block_idx,
                -1,
            )
            for npu_id in (0, 1)
            for block_idx in range(BACKLOG_PER_NPU)
        ],
        0.0,
    )
    return _measure_scheduler(scheduler)


def _run_joint_dynamic():
    npus = [sim.NPUState(npu_id) for npu_id in range(2)]
    config = sim.DynamicCIRPolicyConfig(
        mode=sim.DYNAMIC_CIR_DEMAND_PROPORTIONAL
    )
    controller = sim.DynamicCIRController(npus, config)
    pools = sim.build_dynamic_npu_path_pools(2, 1)
    owners = {
        path_id: npu_id
        for (npu_id, _), path_ids in pools.items()
        for path_id in path_ids
    }
    scheduler = sim.DiskIOScheduler(
        sim.DiskState(0),
        sim.POLICY_QOS_DYNAMIC_JOINT_CIR,
        sim.DISK_BW,
        dynamic_cir_controller=controller,
        dynamic_path_owners=owners,
        dynamic_cir_config=config,
    )

    path_io_counts = [0] * 256
    flows = []
    for npu_id, demand_gbps in enumerate(NPU_DEMANDS_GBPS):
        controller.register_layer(
            npu_id=npu_id,
            layer=0,
            input_demand_gbps=demand_gbps,
            deadline_ms=0.0,
            work_by_disk={0: BACKLOG_PER_NPU * IO_SIZE_GB},
        )
        selected_paths = sim.client_select_dynamic_owned_paths(
            block_sizes_gb=[IO_SIZE_GB] * BACKLOG_PER_NPU,
            path_io_counts=path_io_counts,
            allowed_path_ids=pools[(npu_id, 0)],
        )
        for block_idx, path_id in enumerate(selected_paths):
            path_io_counts[path_id] += 1
            flows.append(
                _flow(
                    npu_id,
                    npu_id * BACKLOG_PER_NPU + block_idx,
                    path_id,
                )
            )
    scheduler.enqueue_many(flows, 0.0)
    initial_grants = scheduler.apply_dynamic_cir_epoch(0.0)
    result = _measure_scheduler(scheduler, controller)
    result.update(
        {
            "control_mode": config.mode,
            "paths_per_npu": config.paths_per_npu,
            "initial_cir_grants_gbps": [
                initial_grants[npu_id] for npu_id in (0, 1)
            ],
            "cir_epochs": scheduler.dynamic_cir_epochs,
            "max_total_cir_gbps": scheduler.dynamic_cir_max_total_gbps,
        }
    )
    return result


def _rates_equal(actual, expected):
    return len(actual) == len(expected) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(actual, expected)
    )


def run_calibration():
    """Run equal-block sustained-backlog calibration for both schedulers."""
    baseline = _run_baseline()
    joint_dynamic = _run_joint_dynamic()
    all_command_rates = (
        baseline["command_service_gbps_unique"]
        + joint_dynamic["command_service_gbps_unique"]
    )
    return {
        "experiment": {
            "description": "two NPU, one SSD, sustained equal-size blocks",
            "ssd_capacity_gbps": sim.DISK_BW,
            "npu_demands_gbps": list(NPU_DEMANDS_GBPS),
            "io_size_gb": IO_SIZE_GB,
            "backlog_per_npu": BACKLOG_PER_NPU,
            "commands_measured": COMMANDS_MEASURED,
            "data_plane": "single nonpreemptive SSD command",
        },
        "baseline": baseline,
        "joint_dynamic": joint_dynamic,
        "checks": {
            "baseline_is_20_20": _rates_equal(
                baseline["measured_ssd_service_gbps"],
                (20.0, 20.0),
            ),
            "joint_dynamic_is_10_30": _rates_equal(
                joint_dynamic["measured_ssd_service_gbps"],
                NPU_DEMANDS_GBPS,
            ),
            "every_command_is_40_gbps": _rates_equal(
                all_command_rates,
                (sim.DISK_BW, sim.DISK_BW),
            ),
            "single_active_ssd_command": (
                baseline["ssd_active_io_unique"] == [1]
                and joint_dynamic["ssd_active_io_unique"] == [1]
                and baseline["max_ssd_active_io"] == 1
                and joint_dynamic["max_ssd_active_io"] == 1
            ),
            "joint_dynamic_total_cir_at_most_40": (
                joint_dynamic["max_total_cir_gbps"] <= sim.DISK_BW
            ),
        },
    }


def _annotate_bars(axis, bars):
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.7,
            f"{bar.get_height():.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def render_calibration(result, output_path):
    """Render measured scheduler shares and data-plane invariants."""
    figure, (rates_axis, checks_axis) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        gridspec_kw={"width_ratios": (1.7, 1.0)},
    )
    positions = (0.0, 1.0)
    width = 0.24
    target_bars = rates_axis.bar(
        [position - width for position in positions],
        result["experiment"]["npu_demands_gbps"],
        width,
        label="Input demand",
        color="white",
        edgecolor="#555555",
        hatch="///",
    )
    baseline_bars = rates_axis.bar(
        positions,
        result["baseline"]["measured_ssd_service_gbps"],
        width,
        label="Baseline NPU-RR",
        color="#4C78A8",
    )
    dynamic_bars = rates_axis.bar(
        [position + width for position in positions],
        result["joint_dynamic"]["measured_ssd_service_gbps"],
        width,
        label="Joint dynamic CIR",
        color="#F58518",
    )
    for bars in (target_bars, baseline_bars, dynamic_bars):
        _annotate_bars(rates_axis, bars)
    rates_axis.set_xticks(positions)
    rates_axis.set_xticklabels(("NPU 0", "NPU 1"))
    rates_axis.set_ylabel("Measured SSD service (GB/s)")
    rates_axis.set_ylim(0.0, 42.0)
    rates_axis.grid(axis="y", alpha=0.25)
    rates_axis.legend(loc="upper left")
    rates_axis.set_title("Measured 20/20 baseline vs 10/30 dynamic split")

    checks_axis.axis("off")
    checks_axis.set_title("Observed data-plane invariants", pad=14)
    baseline = result["baseline"]
    dynamic = result["joint_dynamic"]
    lines = (
        "Real DiskIOScheduler execution",
        "",
        f"Command size: {result['experiment']['io_size_gb']:.3f} GB",
        f"Commands measured: {result['experiment']['commands_measured']}",
        "",
        "Baseline",
        f"  command rate: {baseline['command_service_gbps_unique'][0]:.0f} GB/s",
        f"  max active SSD I/O: {baseline['max_ssd_active_io']}",
        "",
        "Joint dynamic CIR",
        f"  command rate: {dynamic['command_service_gbps_unique'][0]:.0f} GB/s",
        f"  max active SSD I/O: {dynamic['max_ssd_active_io']}",
        f"  max total CIR: {dynamic['max_total_cir_gbps']:.0f} GB/s",
        "",
        "CIR changes arbitration only;",
        "SSD service remains one 40 GB/s command.",
    )
    checks_axis.text(
        0.02,
        0.94,
        "\n".join(lines),
        transform=checks_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "#F7F7F7",
            "edgecolor": "#CCCCCC",
        },
    )
    figure.suptitle(
        "10/30 demand calibration on one 40 GB/s SSD",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_calibration(output_dir=DEFAULT_OUTPUT_DIR):
    """Write JSON evidence and its PNG summary into one directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_calibration()
    assert all(result["checks"].values())
    json_path = output_dir / "calibration.json"
    png_path = output_dir / "calibration.png"
    json_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    render_calibration(result, png_path)
    return result, json_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate baseline RR against joint dynamic CIR."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    result, _, _ = write_calibration(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
