"""Validate and plot the four-strategy focused formal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


SSUS = (40, 56, 80)
STRATEGIES = (
    "baseline",
    "current_refresh8",
    "tune__low_protect_cir_20_6_8_6_current_paths",
    "isolated_no_contention_bound",
)
LABELS = {
    "baseline": "Baseline (NPU-RR)",
    "current_refresh8": "Static CIR before (20/4/12/4)",
    "tune__low_protect_cir_20_6_8_6_current_paths": (
        "Static CIR after (20/6/8/6)"
    ),
    "isolated_no_contention_bound": "Ideal fluid bound (no contention)",
}
COLORS = {
    "baseline": "#4c78a8",
    "current_refresh8": "#f58518",
    "tune__low_protect_cir_20_6_8_6_current_paths": "#54a24b",
    "isolated_no_contention_bound": "#b8b51b",
}


def validate(data):
    if not data["complete"]:
        raise AssertionError("focused matrix is incomplete")
    runtime = data["experiment"]["runtime"]
    if runtime["num_npu"] != 128 or runtime["n_layers"] != 16:
        raise AssertionError("formal NPU/layer contract mismatch")
    if runtime["ssu_list"] != list(SSUS):
        raise AssertionError("formal SSU contract mismatch")
    if runtime["arrival_delay_ms"] != [0.0, 5.0]:
        raise AssertionError("arrival-delay contract mismatch")
    if tuple(data["selected_strategies"]) != STRATEGIES:
        raise AssertionError("focused strategy set mismatch")
    rows = data["results"]
    if len(rows) != len(SSUS) * len(STRATEGIES):
        raise AssertionError("focused row count mismatch")
    index = {(row["num_ssu"], row["strategy"]): row for row in rows}
    if set(index) != {(ssu, name) for ssu in SSUS for name in STRATEGIES}:
        raise AssertionError("focused matrix key mismatch")
    for ssu in SSUS:
        paired = [index[(ssu, name)] for name in STRATEGIES]
        if len({row["workload_fingerprint"] for row in paired}) != 1:
            raise AssertionError("unpaired workload")
        if len({row["placement_hash"] for row in paired}) != 1:
            raise AssertionError("unpaired placement")
        for row in paired:
            if row["kind"] == "simulation" and not all(
                row["summary"]["invariants"].values()
            ):
                raise AssertionError("data-plane invariant failure")
    return index


def request_fraction(row):
    if row["kind"] == "upper_bound":
        return row["summary"]["avg_request_compute_fraction_upper_bound"]
    return row["summary"]["avg_request_compute_fraction"]


def plot(index, output_path):
    figure, axis = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    markers = ("o", "s", "^", "<")
    for name, marker in zip(STRATEGIES, markers):
        axis.plot(
            SSUS,
            [100.0 * request_fraction(index[(ssu, name)]) for ssu in SSUS],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            linestyle="--" if name == "isolated_no_contention_bound" else "-",
            color=COLORS[name],
            label=LABELS[name],
        )
    axis.set_title("Mean per-request NPU compute fraction")
    axis.set_ylabel("Compute / (compute + exposed I/O stall)  (%)")
    axis.set_xlabel("Number of SSUs")
    axis.set_xticks(SSUS)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=True, fontsize=9)
    figure.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_report(index, output_dir):
    lines = [
        "# 四策略正式对比",
        "",
        "固定合同：128 NPU、16 层、SSU=40/56/80、每 NPU 0–5 ms 随机到达延迟；三个可运行策略共享 workload、block placement 和 SSD40→NPU50 数据面。",
        "",
        "## 四条曲线分别是什么",
        "",
        "1. **Baseline (NPU-RR)**：每块 SSD 在 NPU 队列之间轮转，不使用 QoS Path/CIR。",
        "2. **Static CIR before (20/4/12/4)**：当前 `qos_static_cir`，SS/SL/LS/LL 的每盘 CIR 总预算为 20/4/12/4 GB/s，组内 Path 数为 12/4/12/4，Path 压力每 8 条读取一次。",
        "3. **Static CIR after (20/6/8/6)**：调优后的固定配置；Path 数仍为 12/4/12/4，只把类别 CIR 调成 20/6/8/6 GB/s。",
        "4. **Ideal fluid bound**：删除所有跨 NPU SSD 竞争，并允许 SSD 与 NPU fluid cut-through 的不可运行 request 指标上界；它不是硬件策略，没有联合 makespan，所以没有 fleet 数值。",
        "",
        "## 结果",
        "",
        "每格为 `request compute fraction / fleet compute utilization`。",
        "",
        "| SSU | Baseline | Static before | Static after | Ideal fluid bound |",
        "|---:|---:|---:|---:|---:|",
    ]
    for ssu in SSUS:
        values = []
        for name in STRATEGIES:
            row = index[(ssu, name)]
            request = 100.0 * request_fraction(row)
            if row["kind"] == "upper_bound":
                values.append(f"{request:.3f}% / N/A")
            else:
                fleet = 100.0 * row["summary"]["fleet_npu_compute_utilization"]
                values.append(f"{request:.3f}% / {fleet:.3f}%")
        lines.append(f"| {ssu} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "图片只画用户指定的四条 request compute-fraction 曲线。它不是 fleet 指标；fluid bound 没有合法的 fleet/makespan。",
            "",
            "![Four strategy overview](01_strategy_overview.png)",
            "",
            "无竞争上界的完整定义见 [IDEAL_NO_CONTENTION_BOUND_CN.md](../../doc/IDEAL_NO_CONTENTION_BOUND_CN.md)。",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def analyze(input_path, output_dir):
    data = json.loads(input_path.read_text())
    index = validate(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot(index, output_dir / "01_strategy_overview.png")
    write_report(index, output_dir)
    summary = {
        "strategies": list(STRATEGIES),
        "rows": {
            f"{ssu}:{name}": {
                "request_compute_fraction": request_fraction(index[(ssu, name)]),
                "fleet_npu_compute_utilization": (
                    index[(ssu, name)]["summary"].get(
                        "fleet_npu_compute_utilization"
                    )
                ),
            }
            for ssu in SSUS
            for name in STRATEGIES
        },
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/four_strategy_focus/results.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/four_strategy_focus")
    )
    args = parser.parse_args()
    result = analyze(args.input, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
