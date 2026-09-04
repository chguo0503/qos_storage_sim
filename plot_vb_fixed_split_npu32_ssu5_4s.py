#!/usr/bin/env python3
"""Create four auditable 500-ms timelines for the 28-high/4-low run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "results" / "vb_fixed_split_npu32_ssu5_28high_4low_4s"
ORDER = (
    "baseline",
    "route_only",
    "vb_catalog",
    "vb_duration_aware",
    "vb_ll36",
    "vb_split_aware",
)
LABELS = {
    "baseline": "Baseline (Path 0)",
    "route_only": "Route-only",
    "vb_catalog": "V/B catalog (24.57/15.43)",
    "vb_duration_aware": "V/B duration-aware (22.00/18.00)",
    "vb_ll36": "V/B 36/4",
    "vb_split_aware": "V/B request-boundary (37.62/2.38)",
}
COLORS = {
    "baseline": "#4C566A",
    "route_only": "#5E81AC",
    "vb_catalog": "#2E8B57",
    "vb_duration_aware": "#D95F02",
    "vb_ll36": "#B48EAD",
    "vb_split_aware": "#BF616A",
}


def _load(result_dir):
    rows = {}
    for case in ORDER:
        path = result_dir / f"{case}.json"
        if path.exists():
            rows[case] = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit(f"no case JSON files in {result_dir}")
    return rows


def _style():
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


def _blocks(document):
    return document["steady_summary"]["measurement_blocks"]


def _x(blocks, document):
    start = float(document["steady_summary"]["measurement_start_ms"])
    return [
        (0.5 * (float(row["start_ms"]) + float(row["end_ms"])) - start) / 1000.0
        for row in blocks
    ]


def _save(fig, result_dir, stem):
    fig.tight_layout()
    fig.savefig(result_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(result_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot01(rows, result_dir):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for case, document in rows.items():
        blocks = _blocks(document)
        mean = 100.0 * document["metrics"]["mean_npu_utilization"]
        ax.plot(
            _x(blocks, document),
            [100.0 * float(row["npu_utilization"]) for row in blocks],
            marker="o",
            markersize=3.5,
            linewidth=1.55,
            color=COLORS[case],
            label=f"{LABELS[case]} ({mean:.2f}%)",
        )
    ax.set(xlabel="Time in 4 s measurement window (s)", ylabel="Fleet NPU utilization (%)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(88.0, 100.2)
    ax.grid(color="#D8DEE9", linewidth=0.7, alpha=0.8)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    ax.set_title("32 NPU / 5 SSU: compute-area utilization in 500 ms blocks")
    _save(fig, result_dir, "01_npu_utilization_timeline")


def plot02(rows, result_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    for case, document in rows.items():
        blocks = _blocks(document)
        xs = _x(blocks, document)
        high = [100.0 * statistics.fmean(map(float, row["npu_utilizations"][4:32])) for row in blocks]
        low = [100.0 * statistics.fmean(map(float, row["npu_utilizations"][0:4])) for row in blocks]
        axes[0].plot(xs, high, marker="o", markersize=3, linewidth=1.4, color=COLORS[case], label=LABELS[case])
        axes[1].plot(xs, low, marker="o", markersize=3, linewidth=1.4, color=COLORS[case], label=LABELS[case])
    axes[0].set_title("28 high-V lanes")
    axes[1].set_title("4 low-V lanes")
    axes[0].set_ylim(94.0, 100.2)
    axes[1].set_ylim(25.0, 65.0)
    for ax in axes:
        ax.set_ylabel("NPU utilization (%)")
        ax.set_xlim(0.0, 4.0)
        ax.grid(color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axes[1].set_xlabel("Time in 4 s measurement window (s)")
    axes[1].legend(ncol=2, frameon=False, loc="lower right")
    fig.suptitle("V/B benefit and cost by fixed NPU cohort", fontsize=13)
    _save(fig, result_dir, "02_high_low_npu_utilization_timeline")


def plot03(rows, result_dir):
    columns = 2
    nrows = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(11.2, 3.3 * nrows), sharex=True, sharey=True)
    axes = list(getattr(axes, "flat", [axes]))
    ssu_colors = ("#5E81AC", "#A3BE8C", "#EBCB8B", "#D08770", "#B48EAD")
    for ax, (case, document) in zip(axes, rows.items()):
        blocks = _blocks(document)
        xs = _x(blocks, document)
        for ssu_id, color in enumerate(ssu_colors):
            ax.plot(xs, [100.0 * float(row["ssd_utilizations"][ssu_id]) for row in blocks], color=color, linewidth=1.15, label=f"SSU {ssu_id}")
        ax.plot(xs, [100.0 * float(row["ssd_mean_utilization"]) for row in blocks], color="#2E3440", linewidth=2.0, linestyle="--", label="mean")
        ax.set_title(f"{LABELS[case]} — mean {100.0 * document['metrics']['mean_ssd_utilization']:.2f}%")
        ax.grid(color="#D8DEE9", linewidth=0.65, alpha=0.75)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(65.0, 100.5)
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    for index, ax in enumerate(axes[: len(rows)]):
        if index % columns == 0:
            ax.set_ylabel("SSD busy time (%)")
        if index >= (nrows - 1) * columns:
            ax.set_xlabel("Time in measurement window (s)")
    axes[0].legend(ncol=3, frameon=False, loc="lower right")
    fig.suptitle("Per-SSU service utilization in 500 ms blocks", fontsize=13)
    _save(fig, result_dir, "03_ssu_utilization_timeline")


def plot04(rows, result_dir):
    columns = 2
    nrows = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(11.2, 3.25 * nrows), sharex=True, sharey=True)
    axes = list(getattr(axes, "flat", [axes]))
    for ax, (case, document) in zip(axes, rows.items()):
        summary = document["steady_summary"]
        start = float(summary["measurement_start_ms"])
        high_x, high_y, low_x, low_y = [], [], [], []
        for row in summary["request_rows"]:
            x = (float(row["admission_time_ms"]) - start) / 1000.0
            ratio = float(row["ttft_ms"]) / float(row["ideal_ttft_ms"])
            target_x, target_y = (high_x, high_y) if int(row["npu_id"]) >= 4 else (low_x, low_y)
            target_x.append(x)
            target_y.append(ratio)
        ax.scatter(high_x, high_y, s=9, alpha=0.55, color="#5E81AC", label="high-V lane")
        ax.scatter(low_x, low_y, s=13, alpha=0.8, color="#D95F02", label="low-V lane")
        ax.axhline(8.0, color="#BF616A", linewidth=1.4, linestyle="--", label="guard = 8")
        metric = document["metrics"]
        ax.set_title(f"{LABELS[case]} — max {metric['ttft_over_ideal_max']:.2f}, >8: {metric['ttft_over_ideal_gt8_count']}")
        ax.grid(color="#D8DEE9", linewidth=0.65, alpha=0.75)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, max(13.5, 1.04 * max(high_y + low_y)))
    for ax in axes[len(rows):]:
        ax.set_visible(False)
    for index, ax in enumerate(axes[: len(rows)]):
        if index % columns == 0:
            ax.set_ylabel("TTFT / ideal TTFT")
        if index >= (nrows - 1) * columns:
            ax.set_xlabel("Admission time in measurement window (s)")
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    fig.suptitle("Request latency guard over the same 4 s windows", fontsize=13)
    _save(fig, result_dir, "04_ttft_over_ideal_timeline")


def write_csv(rows, result_dir):
    fields = (
        "case", "npu_utilization_pct", "delta_vs_baseline_pp",
        "high_v_utilization_pct", "low_v_utilization_pct", "ssd_utilization_pct",
        "compute_area_npu_ms", "ttft_over_ideal_max", "ttft_over_ideal_gt8_count",
        "request_count", "all_invariants_pass", "no_backlog_exhaustion", "elapsed_wall_s",
    )
    baseline = rows["baseline"]["metrics"]["mean_npu_utilization"]
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case, document in rows.items():
            m = document["metrics"]
            writer.writerow(
                {
                    "case": case,
                    "npu_utilization_pct": 100.0 * m["mean_npu_utilization"],
                    "delta_vs_baseline_pp": 100.0 * (m["mean_npu_utilization"] - baseline),
                    "high_v_utilization_pct": 100.0 * m["high_v_28_mean_npu_utilization"],
                    "low_v_utilization_pct": 100.0 * m["low_v_4_mean_npu_utilization"],
                    "ssd_utilization_pct": 100.0 * m["mean_ssd_utilization"],
                    "compute_area_npu_ms": m["compute_area_npu_ms"],
                    "ttft_over_ideal_max": m["ttft_over_ideal_max"],
                    "ttft_over_ideal_gt8_count": m["ttft_over_ideal_gt8_count"],
                    "request_count": m["request_count"],
                    "all_invariants_pass": m["all_invariants_pass"],
                    "no_backlog_exhaustion": m["no_backlog_exhaustion"],
                    "elapsed_wall_s": document["elapsed_wall_s"],
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    rows = _load(args.result_dir)
    _style()
    plot01(rows, args.result_dir)
    plot02(rows, args.result_dir)
    plot03(rows, args.result_dir)
    plot04(rows, args.result_dir)
    write_csv(rows, args.result_dir)
    print(f"wrote 01-04 PNG/PDF and summary.csv to {args.result_dir}")


if __name__ == "__main__":
    main()
