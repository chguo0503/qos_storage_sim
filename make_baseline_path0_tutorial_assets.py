#!/usr/bin/env python3
"""Build explanatory figures for the Baseline Path0 low-utilization tutorial.

The script reads only the already-generated, authoritative result artifacts.
It does not rerun or mutate the simulator result.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from baseline_path0_layout import DATA, FIGURES, RESULT, ROOT

COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
COMPUTE = "#63b7af"
STALL = "#e45756"
RELEASE = "#235789"
IDLE = "#e8e8e8"
READ_PENDING = "#2679b2"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Noto Sans CJK JP",
                "DengXian",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.2,
            "figure.dpi": 120,
        }
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _clip_interval(start: float, end: float, lower: float, upper: float):
    start = max(start, lower)
    end = min(end, upper)
    if end <= start:
        return None
    return start, end - start


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fig.savefig(
        temporary,
        format="png",
        dpi=220,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)
    temporary.replace(path)


def _save_vector_and_png(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    vector = FIGURES / f"{stem}.pdf"
    temporary = vector.with_name(vector.name + ".tmp")
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(temporary, format="pdf", bbox_inches="tight", facecolor="white")
    temporary.replace(vector)
    _save(fig, FIGURES / f"{stem}.png")


def _profile_legend(fig, result: dict, *, top=0.954, size=9.8) -> None:
    """Repeat the required bytes and compute budget wherever a NPU is shown."""
    handles = []
    for p in result["input"]["profiles"]:
        npu = p["npu_id"]
        label = (f"NPU{npu}：V={p['per_layer_kv_gib'] * 1024:.3f} MiB，"
                 f"C={p['per_layer_compute_us'] / 1000:.3f} ms")
        handles.append(Patch(facecolor=COLORS[npu], label=label))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, top),
               ncol=2, fontsize=size, frameon=False, columnspacing=1.4)


def _relative_layer_timeline(result: dict, layer_rows: list[dict]) -> list[dict]:
    """Keep exact event endpoints, including carry-in/out of the 1-s window."""
    start = float(result["baseline"]["measurement_start_ms"])
    fields = {
        "release": "io_release_time_ms",
        "ready": "io_ready_time_ms",
        "deadline": "comparison_deadline_ms",
        "compute_start": "compute_start_time_ms",
        "compute_end": "compute_end_time_ms",
    }
    rows = []
    for raw in layer_rows:
        row = {name: float(raw[field]) - start for name, field in fields.items()}
        row.update({name: int(raw[name]) for name in ("npu_id", "request_id", "layer")})
        if not all(math.isfinite(row[name]) for name in fields):
            raise AssertionError("Timeline endpoints must be finite")
        if not (row["release"] <= row["ready"] + 1e-8
                and row["ready"] <= row["compute_start"] + 1e-8
                and row["deadline"] <= row["compute_start"] + 1e-8
                and row["compute_start"] < row["compute_end"]):
            raise AssertionError("Invalid read/compute ordering in saved layer trace")
        rows.append(row)
    return rows


def _prefetch_gap_example(rows: list[dict], npu_id=1, request_id=100020) -> dict:
    """Derive and verify the illustrated gap; never infer idle from a blank row."""
    selected = {r["layer"]: r for r in rows
                if r["npu_id"] == npu_id and r["request_id"] == request_id}
    if not {2, 3, 4, 5, 6}.issubset(selected):
        raise AssertionError("The illustrated request needs layers 2 through 6")
    previous, prefetched, following = (selected[n] for n in (3, 4, 5))
    equalities = (
        (previous["ready"], previous["compute_start"]),
        (previous["compute_start"], prefetched["release"]),
        (previous["compute_end"], prefetched["compute_start"]),
        (prefetched["compute_start"], following["release"]),
    )
    if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-7) for a, b in equalities):
        raise AssertionError("Example no longer follows the illustrated prefetch triggers")
    gap_start, gap_end = prefetched["ready"], following["release"]
    if not (previous["compute_start"] <= gap_start < gap_end <= previous["compute_end"] + 1e-7):
        raise AssertionError("The read gap must be covered by the previous layer's compute")
    for row in rows:
        if row["npu_id"] == npu_id and min(row["ready"], gap_end) - max(row["release"], gap_start) > 1e-7:
            raise AssertionError("Another unfinished layer read intersects the annotated NPU gap")
    return {"layers": selected, "gap_start": gap_start, "gap_end": gap_end,
            "gap_ms": gap_end - gap_start, "handoff": previous["ready"]}


def _make_npu_timeline_readable(result: dict, layer_rows: list[dict]):
    """One-second overview plus a layer-resolved, explicitly annotated NPU1 zoom.

    Blue means the logical read has not completed, not continuous SSD service.
    State and read lanes are separate so early-ready reads cannot imply NPU idle.
    """
    rows = _relative_layer_timeline(result, layer_rows)
    example = _prefetch_gap_example(rows)
    layers = example["layers"]
    gap_start, gap_end = example["gap_start"], example["gap_end"]
    baseline = result["baseline"]
    duration = float(baseline["measurement_end_ms"]) - float(baseline["measurement_start_ms"])
    lower, upper = layers[3]["release"], 65.0
    fig = plt.figure(figsize=(8.8, 10.2))
    overview = fig.add_axes((0.20, 0.595, 0.76, 0.225))
    detail = fig.add_axes((0.20, 0.235, 0.76, 0.235))
    fig.suptitle("图 1｜蓝线是“读取还没完成”，不是 SSD 持续传输", fontsize=15, y=0.990)
    fig.text(0.055, 0.944, "蓝条从“发起读取”到“该层 KV 全部到达 HBM”，包含排队、服务与接收。",
             fontsize=11.5)
    fig.text(0.055, 0.916, "同一 NPU 的读取行全部空白＝没有未完成的读取；不代表 NPU 或 SSD 空闲。",
             fontsize=11.5)
    fig.legend(handles=[Patch(facecolor=READ_PENDING, label="蓝：层读取未完成"),
                        Patch(facecolor=COMPUTE, label="青：NPU 计算"),
                        Patch(facecolor=STALL, label="红：NPU 等 I/O")],
               loc="upper left", bbox_to_anchor=(0.04, 0.897), ncol=3,
               fontsize=11.2, frameon=False, columnspacing=1.4)

    overview.set_title(f"A  完整 1 秒总览 · 四卡平均计算利用率 {100 * baseline['mean_npu_utilization']:.4f}%",
                       fontsize=12, loc="left", pad=12)
    for npu in range(4):
        base = (3 - npu) * 2.0
        npu_rows = [r for r in rows if r["npu_id"] == npu]
        for left_field, right_field, y, color in (
            ("release", "ready", base + 0.36, READ_PENDING),
            ("deadline", "compute_start", base - 0.26, STALL),
            ("compute_start", "compute_end", base - 0.26, COMPUTE),
        ):
            intervals = [_clip_interval(r[left_field], r[right_field], 0, duration)
                         for r in npu_rows]
            intervals = [interval for interval in intervals if interval is not None]
            if color == COMPUTE:
                total_compute = sum(width for _, width in intervals)
                if not math.isclose(total_compute, baseline["compute_ms_by_npu"][npu], abs_tol=1e-7):
                    raise AssertionError("Figure 1 compute coverage differs from the original result")
            if intervals:
                # Exact boundaries; no white edges that would invent small gaps.
                overview.broken_barh(intervals, (y - 0.17, 0.34), facecolors=color, linewidths=0)
        overview.text(-0.11, base + 0.05, f"NPU{npu}\n{100 * baseline['npu_utilizations'][npu]:.2f}%",
                      transform=overview.get_yaxis_transform(), ha="right", va="center",
                      fontsize=11, fontweight="bold", linespacing=1.35)
        for label, y in (("读取", base + 0.36), ("算/等", base - 0.26)):
            overview.text(-0.014, y, label, transform=overview.get_yaxis_transform(),
                          ha="right", va="center", fontsize=9.5, color="#444444")
        if npu in (2, 3):
            overview.text(550, base - 0.26, "持续计算", ha="center", va="center", fontsize=9.8)
        if npu > 0:
            overview.axhline(base + 1.0, color="#e0e0e0", linewidth=0.7)
    overview.add_patch(plt.Rectangle((lower, 3.38), upper - lower, 1.32,
                                    fill=False, edgecolor="#222222", linestyle="--", linewidth=1.0))
    overview.annotate("B 图放大区域", xy=(upper, 4.03), xytext=(170, 4.8),
                      fontsize=10, arrowprops={"arrowstyle": "->", "color": "#333333"},
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    overview.set(xlim=(0, duration), ylim=(-0.85, 6.95), yticks=[])
    overview.set_xticks([0, 200, 400, 600, 800, 1000])
    overview.set_xlabel("中间 1 秒窗口内的相对时间（ms）", fontsize=10.5, labelpad=4)
    overview.tick_params(labelsize=10)
    overview.grid(axis="x", color="#dddddd", linewidth=0.6)
    overview.set_axisbelow(True)

    detail.set_title("B  NPU1 放大：分开看每一层的读取，再对照 NPU 状态",
                     fontsize=12, loc="left", pad=32)
    detail.text(0, 1.08, "请求 100020；层号从 L0 起计。空心圆＝发起，实心方块＝HBM 到齐。",
                transform=detail.transAxes, fontsize=10.1)
    # The highlighted column applies to the complete set of NPU1 reads, not
    # just one layer. _prefetch_gap_example checks all requests for this NPU.
    detail.add_patch(plt.Rectangle((gap_start, 0.65), gap_end - gap_start, 4.45,
                                  facecolor="#f1f3f5", edgecolor="#888888",
                                  linestyle=":", linewidth=0.8, zorder=0))
    for layer, y in ((3, 4), (4, 3), (5, 2), (6, 1)):
        row = layers[layer]
        interval = _clip_interval(row["release"], row["ready"], lower, upper)
        detail.broken_barh([interval], (y - 0.22, 0.44), facecolors=READ_PENDING, linewidths=0)
        detail.plot(row["release"], y, marker="o", markersize=6, markerfacecolor="white",
                    markeredgecolor=READ_PENDING, markeredgewidth=1.4, zorder=5, clip_on=False)
        detail.plot(row["ready"], y, marker="s", markersize=5, color=READ_PENDING, zorder=5)
        detail.text((row["release"] + row["ready"]) / 2, y, f"L{layer}",
                    color="white", ha="center", va="center", fontsize=11, fontweight="bold")
    for row in rows:
        if row["npu_id"] != 1:
            continue
        for first, last, color, label in (
            ("deadline", "compute_start", STALL, f"等 L{row['layer']}"),
            ("compute_start", "compute_end", COMPUTE, f"算 L{row['layer']}"),
        ):
            interval = _clip_interval(row[first], row[last], lower, upper)
            if interval is None:
                continue
            left, width = interval
            detail.broken_barh([interval], (-0.30, 0.60), facecolors=color,
                               edgecolors=("#89342e" if color == STALL else "#427e77"),
                               linewidths=0.6, hatch=("///" if color == STALL else None))
            if width > 4.0:
                detail.text(left + width / 2, 0, label, ha="center", va="center",
                            fontsize=9.4, color="#111111")
    for x in (example["handoff"], gap_start, gap_end):
        detail.vlines(x, -0.36, 4.42, color="#777777", linestyle="--", linewidth=0.7, zorder=0)
    detail.annotate("", xy=(gap_end, 4.72), xytext=(gap_start, 4.72),
                    arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "#333333"})
    detail.text((gap_start + gap_end) / 2, 4.94, f"空白 {example['gap_ms']:.3f} ms",
                ha="center", va="bottom", fontsize=10.5)
    for key, x in (("a", example["handoff"]), ("b", gap_start), ("c", gap_end)):
        detail.text(x, -0.59, key, ha="center", va="center", fontsize=10.5, fontweight="bold",
                    bbox={"boxstyle": "circle,pad=0.12", "facecolor": "white",
                          "edgecolor": "#555555", "linewidth": 0.7})
    detail.set_xlim(lower, upper)
    detail.set_ylim(-0.97, 5.4)
    detail.set_yticks([4, 3, 2, 1, 0], ["L3 读取", "L4 读取", "L5 读取", "L6 读取", "NPU1 状态"])
    detail.set_xticks([10, 20, 30, 40, 50, 60, 65])
    detail.tick_params(labelsize=10.5)
    detail.set_xlabel("与 A 图相同的时间原点（ms）；所有边界均来自原始 trace", fontsize=10.3, labelpad=5)
    detail.grid(axis="x", color="#dddddd", linewidth=0.6)
    detail.set_axisbelow(True)
    for ax in (overview, detail):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    note_lines = [
        (f"a  {example['handoff']:.3f} ms", "L3 读齐、红色等待结束；开始算 L3，同时发起 L4 读取。"),
        (f"b  {gap_start:.3f} ms", "L4 已经到齐；NPU1 仍在算 L3，尚未发起 L5 读取。"),
        (f"c  {gap_end:.3f} ms", "L3 计算结束；开始算 L4，同时发起 L5 读取。"),
    ]
    for index, (label, description) in enumerate(note_lines):
        y = 0.151 - index * 0.030
        fig.text(0.055, y, label, fontsize=11.2, fontweight="bold")
        fig.text(0.272, y, description, fontsize=10.8)
    fig.text(0.055, 0.041, "读图关键：b→c 是预取已完成的空白，不是 I/O 卡住；NPU 状态条仍为青色计算。",
             fontsize=11.1, color="#183b50")
    fig.text(0.055, 0.017, "当前模型：开始计算 L 层才预取 L+1；跨请求预取已开启，但不是无限层深预取。",
             fontsize=10.5, color="#444444")
    return fig, example


def plot_npu_timeline_readable(result: dict, layer_rows: list[dict]) -> None:
    fig, _ = _make_npu_timeline_readable(result, layer_rows)
    _save_vector_and_png(fig, "01_npu_io_compute_timeline_readable")


def plot_zoom(result: dict, layer_rows: list[dict], block_rows: list[dict]) -> None:
    window_start = float(result["baseline"]["measurement_start_ms"])
    zoom_start = 0.0
    zoom_end = 100.0
    period = float(result["input"]["profiles"][2]["per_layer_compute_us"]) / 1000.0

    fig = plt.figure(figsize=(13.0, 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(3.7, 1.25), hspace=0.15)
    ax = fig.add_subplot(grid[0])
    service_ax = fig.add_subplot(grid[1], sharex=ax)

    lane_y = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}
    for raw in layer_rows:
        npu = int(raw["npu_id"])
        y = lane_y[npu]
        compute_start = float(raw["compute_start_time_ms"]) - window_start
        compute_end = float(raw["compute_end_time_ms"]) - window_start
        deadline = float(raw["comparison_deadline_ms"]) - window_start
        io_release = float(raw["io_release_time_ms"]) - window_start

        interval = _clip_interval(deadline, compute_start, zoom_start, zoom_end)
        if interval is not None:
            ax.broken_barh([interval], (y - 0.25, 0.50), color=STALL, linewidth=0)
        interval = _clip_interval(compute_start, compute_end, zoom_start, zoom_end)
        if interval is not None:
            ax.broken_barh([interval], (y - 0.25, 0.50), color=COMPUTE, linewidth=0)
        if zoom_start <= io_release <= zoom_end:
            ax.vlines(io_release, y + 0.28, y + 0.43, color=RELEASE, linewidth=0.7)

    pacer_releases = sorted(
        float(row["io_release_time_ms"]) - window_start
        for row in layer_rows
        if int(row["npu_id"]) == 3
        and zoom_start <= float(row["io_release_time_ms"]) - window_start <= zoom_end
    )
    for release in pacer_releases:
        ax.axvline(release, color="#555555", linestyle="--", linewidth=0.8, alpha=0.55)
        service_ax.axvline(release, color="#555555", linestyle="--", linewidth=0.8, alpha=0.55)
    if len(pacer_releases) >= 2:
        x0, x1 = pacer_releases[0], pacer_releases[1]
        ax.annotate(
            "",
            xy=(x1, 3.55),
            xytext=(x0, 3.55),
            arrowprops={"arrowstyle": "<->", "color": "#333333", "linewidth": 1.0},
        )
        ax.text(
            (x0 + x1) / 2,
            3.62,
            f"重复周期 {period:.4f} ms",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.set_ylim(-0.55, 3.92)
    ax.set_yticks([3, 2, 1, 0], ["NPU 0（短流）", "NPU 1（大/快）", "NPU 2（节拍器）", "NPU 3（节拍器）"])
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title("前 100 ms 放大：错位发生了，但同一等待图样每 29.8566 ms 返回")
    ax.legend(
        handles=[
            Patch(facecolor=COMPUTE, label="NPU 计算"),
            Patch(facecolor=STALL, label="I/O barrier 等待"),
            plt.Line2D([0], [0], color=RELEASE, linewidth=1.2, label="层 I/O 释放时刻（短竖线）"),
        ],
        loc="upper right",
        ncol=3,
        frameon=True,
    )

    service_ax.hlines(0, zoom_start, zoom_end, color=IDLE, linewidth=22, zorder=0)
    segments = []
    segment_colors = []
    for raw in block_rows:
        start = float(raw["ssd_start_time_ms"]) - window_start
        end = float(raw["ssd_end_time_ms"]) - window_start
        interval = _clip_interval(start, end, zoom_start, zoom_end)
        if interval is None:
            continue
        clipped_start, width = interval
        segments.append(((clipped_start, 0), (clipped_start + width, 0)))
        segment_colors.append(COLORS[int(raw["npu_id"])])
    service_ax.add_collection(
        LineCollection(segments, colors=segment_colors, linewidths=22, capstyle="butt", zorder=1)
    )
    service_ax.set_ylim(-0.65, 0.65)
    service_ax.set_yticks([0], ["SSU 0 / Path0"])
    service_ax.set_xlim(zoom_start, zoom_end)
    service_ax.set_xlabel("1 秒测量窗口内的相对时间（ms）")
    service_ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    service_ax.legend(
        handles=[Patch(facecolor=COLORS[npu], label=f"服务 NPU {npu}") for npu in range(4)]
        + [Patch(facecolor=IDLE, label="SSD 空闲")],
        loc="upper right",
        ncol=5,
        bbox_to_anchor=(1.0, 1.55),
        frameon=True,
    )

    fig.suptitle("Baseline 单 Path0 的计算、等待和物理服务", fontsize=15, y=0.995)
    _save(fig, FIGURES / "05_phase_lock_zoom.png")


def plot_phase_columns(result: dict, layer_rows: list[dict]) -> None:
    window_start = float(result["baseline"]["measurement_start_ms"])
    window_end = float(result["baseline"]["measurement_end_ms"])
    period = float(result["input"]["profiles"][2]["per_layer_compute_us"]) / 1000.0
    npu3_releases = sorted(
        float(row["io_release_time_ms"])
        for row in layer_rows
        if int(row["npu_id"]) == 3
        and window_start <= float(row["io_release_time_ms"]) < window_end
    )
    anchor = npu3_releases[0]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for npu, ax in enumerate(axes):
        phases = []
        cycles = []
        for raw in layer_rows:
            if int(raw["npu_id"]) != npu:
                continue
            release = float(raw["io_release_time_ms"])
            if not (anchor <= release < window_end):
                continue
            cycle = int(math.floor((release - anchor) / period + 1e-8))
            phase = release - anchor - cycle * period
            if phase < -1e-7:
                continue
            if phase > period - 1e-6:
                phase = 0.0
                cycle += 1
            phases.append(phase)
            cycles.append(cycle)
        ax.scatter(phases, cycles, s=11 if npu == 0 else 20, color=COLORS[npu], alpha=0.76, edgecolors="none")
        ax.set_title(f"NPU {npu}: 每层 I/O 释放相位")
        ax.grid(color="#dddddd", linewidth=0.65, alpha=0.8)
        ax.set_xlim(-0.5, period + 0.5)
        ax.set_ylim(-1, max(cycles) + 1)
        if npu >= 2:
            ax.set_xlabel("在 29.8566 ms 周期内的位置（ms）")
        if npu % 2 == 0:
            ax.set_ylabel("周期编号")

    fig.suptitle(
        "相位没有持续漂移：散点形成竖列，说明释放模式被锁在固定相位\n"
        "（若等待能不断把流打散，竖列会逐周期向左右移动或扩散）",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, FIGURES / "06_release_phase_columns.png")


def _bin_ssd_service(block_rows, window_start, lower, upper, bin_ms=0.1):
    """Integrate exact, nonoverlapping SSD service into display bins.

    Returns bin edges, per-NPU service milliseconds, and idle milliseconds.
    This is time-weighted, not command-count-weighted. Original events and
    all simulator metrics remain unchanged.
    """
    if bin_ms <= 0 or upper <= lower:
        raise ValueError("A positive bin size and nonempty interval are required")
    count = math.ceil((upper - lower) / bin_ms)
    edges = [min(upper, lower + i * bin_ms) for i in range(count + 1)]
    service = [[0.0] * count for _ in range(4)]
    intervals = []
    for row in block_rows:
        start = float(row["ssd_start_time_ms"]) - window_start
        end = float(row["ssd_end_time_ms"]) - window_start
        clipped = _clip_interval(start, end, lower, upper)
        if clipped is not None:
            start, width = clipped
            intervals.append((start, start + width, int(row["npu_id"])))
    previous_end = lower
    for start, end, npu in sorted(intervals):
        if start < previous_end - 1e-8:
            raise AssertionError("The single SSD must not serve overlapping commands")
        previous_end = max(previous_end, end)
        first = max(0, int(math.floor((start - lower) / bin_ms)))
        last = min(count - 1, int(math.floor((end - lower) / bin_ms)))
        for index in range(first, last + 1):
            service[npu][index] += max(
                0.0, min(end, edges[index + 1]) - max(start, edges[index]))
    idle = []
    for index in range(count):
        width = edges[index + 1] - edges[index]
        used = sum(lane[index] for lane in service)
        if used > width + 1e-8:
            raise AssertionError("Binned service exceeds the available SSD time")
        idle.append(max(0.0, width - used))
    return edges, service, idle


def _npu2_read_example(result: dict, layer_rows: list[dict], block_rows: list[dict]) -> dict:
    """Audit the complete NPU2/L2 lifecycle, not just its service envelope."""
    rows = _relative_layer_timeline(result, layer_rows)
    request = {r["layer"]: r for r in rows if r["request_id"] == 200010 and r["npu_id"] == 2}
    previous, current, following = (request[layer] for layer in (1, 2, 3))
    for a, b in ((previous["compute_start"], current["release"]),
                 (previous["compute_end"], current["deadline"]),
                 (current["deadline"], current["compute_start"]),
                 (current["compute_start"], following["release"])):
        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-7):
            raise AssertionError("NPU2 read example no longer has the illustrated compute/IO overlap")
    start = float(result["baseline"]["measurement_start_ms"])
    commands = sorted((float(r["ssd_start_time_ms"]) - start,
                       float(r["ssd_end_time_ms"]) - start,
                       float(r["size_gib"])) for r in block_rows
                      if int(r["request_id"]) == 200010 and int(r["layer"]) == 2
                      and int(r["npu_id"]) == 2)
    volume = sum(size for _, _, size in commands)
    service = sum(b - a for a, b, _ in commands)
    expected = result["input"]["profiles"][2]["per_layer_kv_gib"]
    if not math.isclose(volume, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError("NPU2 example needs the full layer's physical service trace")
    if not math.isclose(service, volume / 40 * 1000, rel_tol=1e-10, abs_tol=1e-7):
        raise AssertionError("NPU2 service time must match the original 40-GiB/s byte model")
    if not current["release"] <= commands[0][0] < commands[-1][1] <= current["ready"] < current["deadline"]:
        raise AssertionError("Invalid NPU2 release/service/ready/deadline order")
    return {"request_id": 200010, "layer": 2, "npu_id": 2,
            "release_ms": current["release"], "ready_ms": current["ready"],
            "next_release_ms": following["release"],
            "latency_ms": current["ready"] - current["release"],
            "period_ms": following["release"] - current["release"],
            "ready_slack_ms": current["deadline"] - current["ready"],
            "first_service_ms": commands[0][0], "last_service_ms": commands[-1][1],
            "service_span_ms": commands[-1][1] - commands[0][0],
            "service_ms": service, "volume_gib": volume, "command_count": len(commands)}


def _draw_service_fractions(ax, edges, service, idle) -> None:
    """Band thickness, not its cumulative top, is each NPU's time fraction."""
    bottom = [0.0] * (len(edges) - 1)
    for lane, color in zip(service + [idle], list(COLORS) + [IDLE]):
        fraction = [100 * value / (edges[i + 1] - edges[i]) for i, value in enumerate(lane)]
        top = [a + b for a, b in zip(bottom, fraction)]
        ax.fill_between(edges, bottom + [bottom[-1]], top + [top[-1]],
                        step="post", facecolor=color, linewidth=0)
        bottom = top
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax.set_ylabel("SSD 时间占比", fontsize=11)
    ax.grid(axis="y", color="white", linewidth=0.5, alpha=0.55)


def _make_zoom_annotated(result: dict, layer_rows: list[dict], block_rows: list[dict]):
    """Figure 4: one identity palette, explicit wait state, truthful multiscale view.

    Legacy images remain unchanged. The new figure uses
    time bins for the coarse SSD view so subpixel commands do not disappear
    or blend into misleading colors; the third panel shows the exact order.
    """
    window_start = float(result["baseline"]["measurement_start_ms"])
    example = _npu2_read_example(result, layer_rows, block_rows)
    lower, upper = 0.0, 100.0
    period = float(result["input"]["profiles"][2]["per_layer_compute_us"]) / 1000
    edges, service, idle = _bin_ssd_service(block_rows, window_start, lower, upper)
    wait_face, wait_edge = "#ececec", "#555555"
    fig = plt.figure(figsize=(8.8, 11.0))
    grid = fig.add_gridspec(
        3, 1, height_ratios=(4.35, 1.8, 1.0), hspace=0.68,
        left=0.15, right=0.985, bottom=0.12, top=0.815)
    ax = fig.add_subplot(grid[0])
    service_ax = fig.add_subplot(grid[1], sharex=ax)
    micro_ax = fig.add_subplot(grid[2])
    fig.suptitle("图 4｜分清计算时间、读取延迟与 SSD 服务时间", fontsize=15, y=0.993)
    _profile_legend(fig, result)
    fig.text(0.15, 0.873, "V＝每层读取量；C＝每层计算时间。彩色横条是计算，非读取。", fontsize=10)
    lane_y = {n: 3.0 - n for n in range(4)}
    compute_ms = [0.0] * 4
    for row in layer_rows:
        npu = int(row["npu_id"])
        y = lane_y[npu]
        start = float(row["compute_start_time_ms"]) - window_start
        end = float(row["compute_end_time_ms"]) - window_start
        deadline = float(row["comparison_deadline_ms"]) - window_start
        release = float(row["io_release_time_ms"]) - window_start
        waiting = _clip_interval(deadline, start, lower, upper)
        computing = _clip_interval(start, end, lower, upper)
        if waiting is not None:
            ax.broken_barh([waiting], (y - 0.24, 0.48),
                           facecolors=wait_face, edgecolors=wait_edge,
                           linewidths=0.35, hatch="////")
        if computing is not None:
            compute_ms[npu] += computing[1]
            ax.broken_barh([computing], (y - 0.24, 0.48),
                           facecolors=COLORS[npu], linewidths=0)
        if lower <= release <= upper:
            ax.vlines(release, y + 0.28, y + 0.43, color="#111111", linewidth=0.9)
    for npu in (2, 3):
        if math.isclose(compute_ms[npu], upper - lower, abs_tol=1e-7):
            ax.text(53, lane_y[npu], "持续计算（0–100 ms）", ha="center",
                    va="center", fontsize=11.5, color="black",
                    bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2})
    pacer_releases = sorted(float(row["io_release_time_ms"]) - window_start
                            for row in layer_rows if int(row["npu_id"]) == 3
                            and lower <= float(row["io_release_time_ms"]) - window_start <= upper)
    for release in pacer_releases:
        for panel in (ax, service_ax):
            panel.axvline(release, color="#333333", linestyle="--", linewidth=0.8, alpha=0.6)
    if len(pacer_releases) >= 2:
        x0, x1 = pacer_releases[:2]
        ax.annotate("", xy=(x1, 3.63), xytext=(x0, 3.63),
                    arrowprops={"arrowstyle": "<->", "color": "#333333"})
        ax.text((x0 + x1) / 2, 3.76, f"两次发起间隔 {period:.4f} ms", ha="center", fontsize=10)
        ax.text(43, 3.63, f"NPU3：{x0:.6f} → {x1:.6f} ms", fontsize=9.3, va="center")
    ax.set_title("A  NPU 计算与等待：精确事件时间线", loc="left", pad=34, fontsize=12.5)
    ax.legend(
        handles=[tuple(Patch(facecolor=c) for c in COLORS),
                 Patch(facecolor=wait_face, edgecolor=wait_edge, hatch="////", label="灰斜纹：等待 I/O"),
                 plt.Line2D([0], [0], color="#111111", marker="|", linestyle="none",
                            markersize=10, label="短竖线：发起读取")],
        labels=["彩色实心：计算", "灰斜纹：等待 I/O", "短竖线：发起读取（非结束）"],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        loc="lower left", bbox_to_anchor=(-0.02, 1.015), ncol=3,
        fontsize=9.7, frameon=False, columnspacing=0.8, handletextpad=0.4)
    # A separate read lane prevents an IO-latency bracket from looking like
    # compute duration. Its endpoints and the next release come from the trace.
    y = -1.02
    release, ready, next_release = (example[key] for key in ("release_ms", "ready_ms", "next_release_ms"))
    ax.axhline(-0.48, color="#dddddd", linewidth=0.7)
    ax.plot([release, ready], [y, y], color="#196b25", linewidth=2.3, zorder=5)
    ax.plot(release, y, marker="o", markerfacecolor="white", markeredgecolor="#196b25",
            markeredgewidth=1.4, markersize=6, zorder=6)
    ax.plot(ready, y, marker="D", color="#196b25", markersize=5, zorder=6)
    ax.plot([ready, next_release], [y, y], color="#777777", linestyle=":", linewidth=1.1)
    ax.plot(next_release, y, marker="|", markersize=11, color="black")
    ax.text((release + ready) / 2, y + 0.27, f"L2：{example['latency_ms']:.3f} ms",
            ha="center", fontsize=10, color="#155b20")
    for x in (release, ready, next_release):
        ax.text(x, y - 0.28, f"{x:.3f}", ha="center", va="top", fontsize=9.3)
    ax.text(44, y + 0.14, "○ 发起 → ◆ HBM 到齐；含排队与接收", fontsize=9.5, va="center")
    ax.text(44, y - 0.21, f"到齐后仍算 L1；{next_release:.3f} ms 才发起 L3", fontsize=9.5, va="center")
    ax.set_ylim(-1.68, 4.04)
    ax.set_xlim(lower, upper)
    ax.set_yticks([3, 2, 1, 0, y], ["NPU0", "NPU1", "NPU2", "NPU3", "NPU2\nL2 读取"])
    ax.tick_params(labelsize=11)
    for tick, color in zip(ax.get_yticklabels(), COLORS):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("测量窗口内的相对时间（ms）", fontsize=11)

    _draw_service_fractions(service_ax, edges, service, idle)
    service_ax.set_title("B  同一个 SSD 的时间分配（每 0.1 ms 分桶）",
                         loc="left", fontsize=12.5, pad=29)
    service_ax.text(0, 1.06, "色带厚度＝该卡服务时长 ÷ 0.1 ms × 100%；灰色＝空闲",
                    transform=service_ax.transAxes, fontsize=10.1)
    service_ax.set_ylim(0, 100)
    service_ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    service_ax.tick_params(labelsize=10.5)
    service_ax.set_ylabel("SSD 时间占比", fontsize=11)
    service_ax.set_xlabel("测量窗口内的相对时间（ms）", fontsize=11)
    service_ax.grid(axis="y", color="white", linewidth=0.5, alpha=0.55)

    micro_lower, micro_upper = 8.0, 8.1
    for row in block_rows:
        start = float(row["ssd_start_time_ms"]) - window_start
        end = float(row["ssd_end_time_ms"]) - window_start
        interval = _clip_interval(start, end, micro_lower, micro_upper)
        if interval is None:
            continue
        left, width = interval
        left, width = (left - micro_lower) * 1000, width * 1000
        npu = int(row["npu_id"])
        micro_ax.broken_barh([(left, width)], (0.15, 0.7),
                             facecolors=COLORS[npu], edgecolors="white", linewidths=0.7)
        if width >= 2.5:
            micro_ax.text(left + width / 2, 0.5, str(npu), ha="center", va="center",
                           color=("black" if npu == 2 else "white"), fontsize=10.5, fontweight="bold")
    micro_ax.set_xlim(0, 100)
    micro_ax.set_ylim(0, 1)
    micro_ax.set_yticks([0.5], ["SSD 0"])
    micro_ax.set_xticks([0, 20, 40, 60, 80, 100])
    micro_ax.tick_params(labelsize=10.5)
    micro_ax.set_title("C  微秒放大：8.000–8.100 ms 的真实命令顺序",
                       loc="left", fontsize=12.5, pad=28)
    micro_ax.text(0, 1.1, "块内数字为 NPU 编号；绿/紫交替；一条满命令约 4.196 µs",
                  transform=micro_ax.transAxes, fontsize=10.5)
    micro_ax.set_xlabel("从 8.000 ms 起计的相对时间（µs）", fontsize=11)
    fig.text(0.15, 0.037, "B 中蓝细条是短服务片段，不是细线符号；排队时间不计入 SSD 服务。", fontsize=10.3)
    fig.text(0.15, 0.015, "长虚线仅为周期参考线；不是读取结束。B 的上下堆叠不表示并行服务。", fontsize=10.3)
    return fig, example


def plot_zoom_annotated(result: dict, layer_rows: list[dict], block_rows: list[dict]) -> None:
    fig, _ = _make_zoom_annotated(result, layer_rows, block_rows)
    _save_vector_and_png(fig, "05_phase_lock_zoom_annotated")


def _make_npu_0_10ms_zoom(result: dict, layer_rows: list[dict], block_rows: list[dict]):
    """Exact four-NPU state zoom with the same 0.1-ms SSD bins as Figure 4."""
    rows = _relative_layer_timeline(result, layer_rows)
    origin = float(result["baseline"]["measurement_start_ms"])
    lower, upper = 0.0, 10.0
    edges, service, idle = _bin_ssd_service(block_rows, origin, lower, upper)
    fig = plt.figure(figsize=(8.8, 8.4))
    ax = fig.add_axes((0.15, 0.415, 0.83, 0.37))
    service_ax = fig.add_axes((0.15, 0.13, 0.83, 0.155), sharex=ax)
    fig.suptitle("图 5｜0–10 ms 放大：四卡状态与 SSD 时间占比", fontsize=15, y=0.99)
    _profile_legend(fig, result, size=9.6)
    fig.text(0.08, 0.862, "V＝每层读取量；C＝计算时间。条内 Lx＝计算层；黑竖线/“读Lx”＝发起读取。", fontsize=10)
    ax.set_title("A  图 4A 的精确事件放大（不是新的利用率统计窗口）", loc="left", fontsize=12, pad=14)
    summary = []
    for npu in range(4):
        y = 3.0 - npu
        compute_ms = wait_ms = 0.0
        for row in rows:
            if row["npu_id"] != npu:
                continue
            waiting = _clip_interval(row["deadline"], row["compute_start"], lower, upper)
            computing = _clip_interval(row["compute_start"], row["compute_end"], lower, upper)
            if waiting:
                wait_ms += waiting[1]
                ax.broken_barh([waiting], (y - 0.23, 0.46), facecolors="#ececec",
                               edgecolors="#555555", linewidths=0.4, hatch="////")
                if waiting[1] > 0.7:
                    suffix = " →" if row["compute_start"] > upper else ""
                    label = f"等 L{row['layer']}{suffix}"
                    if row["deadline"] >= lower:
                        label += f"\n{row['deadline']:.3f} 开始"
                    ax.text(waiting[0] + waiting[1] / 2, y, label,
                            ha="center", va="center", fontsize=10.5,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1})
            if computing:
                compute_ms += computing[1]
                ax.broken_barh([computing], (y - 0.23, 0.46), facecolors=COLORS[npu],
                               edgecolors="white", linewidths=0.7)
                suffix = " →" if row["compute_end"] > upper else ""
                ax.text(computing[0] + computing[1] / 2, y, f"L{row['layer']}{suffix}",
                        ha="center", va="center", fontsize=9.8,
                        color="black" if npu == 2 else "white", fontweight="bold")
            if lower <= row["release"] <= upper:
                # Lower rows have an occupied NPU lane directly above them.
                # Keep their two-line release labels wholly in the inter-lane gap.
                tick_start, tick_end, label_offset = ((0.27, 0.43, 0.47) if npu == 0
                                                       else (0.25, 0.34, 0.36))
                ax.vlines(row["release"], y + tick_start, y + tick_end, color="black", linewidth=1.0)
                ax.text(row["release"], y + label_offset, f"读L{row['layer']}\n{row['release']:.3f}",
                        ha="center", va="bottom", fontsize=8.5, linespacing=1.1)
        if not math.isclose(compute_ms + wait_ms, upper - lower, abs_tol=1e-7):
            raise AssertionError("The 0-10-ms state zoom must cover all time, including carry-in")
        summary.append({"npu_id": npu, "compute_ms": compute_ms, "wait_ms": wait_ms})
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.48, 3.88)
    ax.set_yticks([3, 2, 1, 0], ["NPU0", "NPU1", "NPU2", "NPU3"])
    for tick, color in zip(ax.get_yticklabels(), COLORS):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_xticks(list(range(11)))
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel("测量窗口内的相对时间（ms）", fontsize=10.5, labelpad=5)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.add_patch(plt.Rectangle((5.2, -0.26), 0.1, 3.52, fill=False,
                              edgecolor="#555555", linewidth=0.7, linestyle=":"))

    _draw_service_fractions(service_ax, edges, service, idle)
    service_ax.set_title("B  同期 SSD 服务：仍按 0.1 ms 分桶，蓝条宽度放大 10 倍",
                         loc="left", fontsize=11.8, pad=30)
    sample = 52  # [5.2, 5.3) ms, also used for the worked arithmetic in the text.
    blue_pct = service[0][sample] / (edges[sample + 1] - edges[sample]) * 100
    idle_pct = idle[sample] / (edges[sample + 1] - edges[sample]) * 100
    service_ax.text(0, 1.08, f"5.2–5.3 ms：蓝色 {blue_pct:.3f}% + 灰色空闲 {idle_pct:.3f}% = 100%",
                    transform=service_ax.transAxes, fontsize=10.5)
    service_ax.add_patch(plt.Rectangle((edges[sample], 0), edges[sample + 1] - edges[sample], 100,
                                      fill=False, edgecolor="black", linewidth=0.9))
    service_ax.tick_params(labelsize=10.5)
    service_ax.set_xlabel("与 A 图相同的时间原点（ms）", fontsize=10.5, labelpad=5)
    for panel in (ax, service_ax):
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
    fig.text(0.08, 0.06, "NPU0 的 L7→L0 是请求 110→111 的切换；L0 读取已由前一请求提前发起。", fontsize=10.3)
    fig.text(0.08, 0.032, "→ 表示该计算/等待延伸到右边界之外；10 ms 只是视窗边界，不是完成时刻。", fontsize=10.3)
    return fig, {"npu_state_ms": summary, "bin_edges_ms": edges, "service_ms": service,
                 "idle_ms": idle, "worked_bin_ms": [edges[sample], edges[sample + 1]],
                 "worked_blue_percent": blue_pct, "worked_idle_percent": idle_pct}


def plot_npu_0_10ms_zoom(result: dict, layer_rows: list[dict], block_rows: list[dict]) -> None:
    fig, _ = _make_npu_0_10ms_zoom(result, layer_rows, block_rows)
    _save_vector_and_png(fig, "10_npu_0_10ms_zoom")


def plot_search(search_rows: list[dict[str, str]]) -> None:
    x = [int(row["victim_nql"]) for row in search_rows]
    mean = [100.0 * float(row["mean_npu_utilization"]) for row in search_rows]
    victim = [100.0 * float(row["npu0_utilization"]) for row in search_rows]
    bandwidth = [float(row["total_bandwidth_gibps"]) for row in search_rows]
    winner_index = min(range(len(mean)), key=mean.__getitem__)

    fig, (mean_ax, victim_ax) = plt.subplots(
        2,
        1,
        figsize=(11.8, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.15), "hspace": 0.12},
        layout="constrained",
    )
    bandwidth_ax = victim_ax.twinx()
    mean_ax.plot(
        x,
        mean,
        marker="o",
        markersize=3.7,
        linewidth=1.7,
        color="#234e70",
        label="4 NPU 平均利用率",
    )
    victim_ax.plot(
        x,
        victim,
        marker=".",
        markersize=3.2,
        linewidth=1.2,
        color="#e45756",
        label="NPU 0 利用率",
    )
    bandwidth_ax.plot(
        x,
        bandwidth,
        linestyle="--",
        linewidth=1.3,
        color="#777777",
        label="名义带宽和",
    )
    mean_ax.scatter(
        [x[winner_index]],
        [mean[winner_index]],
        s=85,
        color="#f2c14e",
        edgecolor="#333333",
        zorder=5,
    )
    victim_ax.scatter(
        [x[winner_index]],
        [victim[winner_index]],
        s=65,
        color="#f2c14e",
        edgecolor="#333333",
        zorder=5,
    )
    mean_ax.annotate(
        f"有限网格最低点：NQL={x[winner_index]}，平均={mean[winner_index]:.4f}%",
        xy=(x[winner_index], mean[winner_index]),
        xytext=(x[winner_index] + 2.7, mean[winner_index] - 0.13),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9.5,
    )
    mean_ax.set_ylim(min(mean) - 0.22, max(mean) + 0.18)
    victim_ax.set_ylim(min(victim) - 0.35, max(victim) + 0.35)
    victim_ax.set_xlabel("NPU 0 的整数 NQL（其余三条流固定）")
    mean_ax.set_ylabel("4 NPU 平均利用率（%）")
    victim_ax.set_ylabel("NPU 0 利用率（%）")
    bandwidth_ax.set_ylabel("四条流名义带宽和（GiB/s）")
    mean_ax.grid(color="#dddddd", linewidth=0.7, alpha=0.8)
    victim_ax.grid(color="#dddddd", linewidth=0.7, alpha=0.8)
    mean_ax.set_title("最后一步：统一 1 秒口径逐点复跑 NQL=153…192")
    mean_ax.legend(loc="upper right", frameon=True)
    handles1, labels1 = victim_ax.get_legend_handles_labels()
    handles2, labels2 = bandwidth_ax.get_legend_handles_labels()
    victim_ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)
    _save(fig, FIGURES / "07_bounded_search_curve.png")


def plot_feedback_loop() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, "多条流的层 I/O\n在相近时刻释放"),
        (0.265, "同一 Path0 FCFS\n大流块形成队列"),
        (0.51, "目标层数据晚到\nNPU 进入 barrier"),
        (0.755, "compute start 后移\n下一层预取也后移"),
    ]
    width = 0.205
    height = 0.30
    y = 0.49
    for index, (x, label) in enumerate(boxes):
        color = ("#d9edf7", "#fde2c5", "#f8d7da", "#e2e3f3")[index]
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#404040",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
        if index < len(boxes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.006, y + height / 2),
                    (boxes[index + 1][0] - 0.006, y + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.4,
                    color="#404040",
                )
            )
    ax.add_patch(
        FancyArrowPatch(
            (0.86, y - 0.02),
            (0.14, y - 0.02),
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color="#7b2cbf",
        )
    )
    ax.text(
        0.50,
        0.08,
        "反馈：等待不是独立噪声；它同时改写后续释放时刻。确定性节拍器会把系统带回相同相位。",
        ha="center",
        va="center",
        color="#5a189a",
        fontsize=11,
    )
    ax.set_title("为什么“等待导致错位”不等于“以后不会再撞车”", fontsize=14, pad=8)
    fig.tight_layout()
    _save(fig, FIGURES / "08_closed_loop_feedback.png")


def plot_queue_causality(audit: dict) -> None:
    """Show queue delay before the exposed barrier, with times derived from trace."""
    row = audit["npu1_causal_example"]
    waiting = row["queue_before_first_service_ms"]
    service = row["own_service_ms"]
    latency = row["ready_latency_ms"]
    budget = row["budget_ms"]
    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.broken_barh([(0, waiting)], (1.0, 0.42), color="#f2c14e")
    ax.broken_barh([(waiting, service)], (1.0, 0.42), color="#5b8db8")
    ax.broken_barh([(waiting + service, row["link_tail_ms"])], (1.0, 0.42), color="#555555")
    ax.broken_barh([(0, budget)], (0.20, 0.42), color=COMPUTE)
    ax.broken_barh([(budget, latency - budget)], (0.20, 0.42), color=STALL)
    for x, y, text_value in [
        (waiting / 2, 1.21, f"首次服务前等待 {waiting:.3f} ms"),
        (waiting + service / 2, 1.21, f"自身 SSD 服务 {service:.3f} ms"),
        (budget / 2, 0.41, f"前一层计算 / 预取预算 {budget:.3f} ms"),
        ((budget + latency) / 2, 0.41, f"barrier {latency - budget:.3f} ms"),
    ]:
        ax.text(x, y, text_value, ha="center", va="center", fontsize=10.5)
    ax.axvline(budget, color="#a52a2a", linestyle="--", linewidth=1.5)
    ax.text(budget, 1.64, "deadline", ha="center", color="#a52a2a", fontsize=11)
    ax.annotate(
        f"HBM ready: {latency:.3f} ms\n（含 {row['link_tail_ms'] * 1000:.3f} µs 接收尾部）",
        xy=(latency, 1.40), xytext=(latency + 0.5, 1.8),
        ha="right", fontsize=9.5,
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set_xlim(-0.2, latency + 1.0)
    ax.set_ylim(0, 2.05)
    ax.set_yticks([1.21, 0.41], ["该层读取", "NPU1 时间预算"])
    ax.set_xlabel("从该层 I/O 释放开始计时（ms）")
    ax.grid(axis="x", color="#dddddd", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title("请求 100020 / 第 3 层：先排队 11.030 ms，随后才暴露为计算停顿", pad=12)
    fig.tight_layout()
    _save(fig, FIGURES / "09_npu1_queue_causality.png")


def main() -> None:
    _configure_matplotlib()
    with (DATA / "result.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    layer_rows = _read_rows(DATA / "request_layer_timeline.csv")
    block_rows = _read_rows(DATA / "physical_block_trace.csv")
    search_rows = _read_rows(DATA / "search_summary.csv")
    plot_npu_timeline_readable(result, layer_rows)
    plot_zoom_annotated(result, layer_rows, block_rows)
    plot_npu_0_10ms_zoom(result, layer_rows, block_rows)
    plot_phase_columns(result, layer_rows)
    plot_search(search_rows)
    plot_feedback_loop()
    audit_path = DATA / "tutorial_numeric_audit.json"
    if audit_path.exists():
        with audit_path.open(encoding="utf-8") as handle:
            plot_queue_causality(json.load(handle))
    print("generated readable 01, annotated 05, 0-10-ms zoom 10 (PDF/PNG), figures 06--08, and 09 when audited")


if __name__ == "__main__":
    main()
