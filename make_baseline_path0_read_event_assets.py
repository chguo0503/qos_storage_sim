#!/usr/bin/env python3
"""Draw Figure 4A's SSU-side read events, without altering the trace."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from baseline_path0_layout import DATA
from make_baseline_path0_tutorial_assets import (
    COLORS, _bin_ssd_service, _clip_interval, _configure_matplotlib,
    _draw_service_fractions, _profile_legend, _read_rows,
    _relative_layer_timeline, _save_vector_and_png,
)


SELECTED = {
    "0a": (0, 111, 6), "0b": (0, 111, 7),
    "1a": (1, 100020, 3), "1b": (1, 100020, 4),
    "2a": (2, 200010, 2), "2b": (2, 200010, 3),
    "3a": (3, 300010, 2), "3b": (3, 300010, 3),
}


def layer_read_events(result, layers, blocks):
    """Join logical reads to final SSD completions; preserve carry-in/out.

    The saved physical trace is clipped to the measurement second. For a
    carry-in layer its observed bytes may omit an earlier prefix, but its
    final SSD completion remains present. Only complete layers can be used
    for the eight numerical teaching examples.
    """
    origin = float(result["baseline"]["measurement_start_ms"])
    logical = _relative_layer_timeline(result, layers)
    visible = {(r["npu_id"], r["request_id"], r["layer"]): r for r in logical
               if r["release"] < 100 and r["ready"] > 0}
    physical = defaultdict(list)
    for row in blocks:
        key = tuple(int(row[name]) for name in ("npu_id", "request_id", "layer"))
        if key in visible:
            physical[key].append(row)
    events = []
    for key, row in visible.items():
        commands = physical[key]
        if not commands:
            raise AssertionError("Visible read is missing its physical SSD trace")
        first = min(float(c["ssd_start_time_ms"]) for c in commands) - origin
        last = max(float(c["ssd_end_time_ms"]) for c in commands) - origin
        volume = sum(float(c["size_gib"]) for c in commands)
        service = sum(float(c["ssd_end_time_ms"]) - float(c["ssd_start_time_ms"]) for c in commands)
        expected = result["input"]["profiles"][key[0]]["per_layer_kv_gib"]
        complete = math.isclose(volume, expected, rel_tol=1e-10, abs_tol=1e-12)
        if not row["release"] <= first < last <= row["ready"] + 1e-8:
            raise AssertionError("SSD completion must follow release and precede HBM ready")
        if not math.isclose(service, volume / 40 * 1000, rel_tol=1e-9, abs_tol=1e-7):
            raise AssertionError("Physical service no longer matches 40 GiB/s")
        events.append({"npu_id": key[0], "request_id": key[1], "layer": key[2],
                       "release_ms": row["release"], "last_ssd_ms": last,
                       "ready_ms": row["ready"], "volume_gib": volume,
                       "service_ms": service, "complete_trace": complete,
                       "command_count": len(commands)})
    return sorted(events, key=lambda e: (e["npu_id"], e["release_ms"]))


def selected_read_events(events):
    lookup = {(e["npu_id"], e["request_id"], e["layer"]): e for e in events}
    selected = {}
    for tag, key in SELECTED.items():
        if key not in lookup or not lookup[key]["complete_trace"]:
            raise AssertionError("Selected example requires the full layer trace")
        event = lookup[key]
        if not event["release_ms"] < event["last_ssd_ms"] < event["ready_ms"]:
            raise AssertionError("Selected example must distinguish SSD completion and HBM ready")
        selected[tag] = event
    for npu in range(4):
        first, second = (selected[f"{npu}{letter}"] for letter in "ab")
        between = [e for e in events if e["npu_id"] == npu
                   and first["release_ms"] < e["release_ms"] < second["release_ms"]]
        if between:
            raise AssertionError("The illustrated release endpoints must be adjacent on the same NPU")
    return selected


def make_read_event_figure(result, layers, blocks):
    events = layer_read_events(result, layers, blocks)
    selected = selected_read_events(events)
    tags = {SELECTED[tag]: tag for tag in selected}
    intervals = {n: selected[f"{n}b"]["release_ms"] - selected[f"{n}a"]["release_ms"]
                 for n in range(4)}
    fig = plt.figure(figsize=(8.8, 10.7))
    fig.suptitle("图 4A｜聚焦 SSU 侧：读取发起 → 最后 SSD 完成", fontsize=15, y=0.994)
    _profile_legend(fig, result, top=0.966, size=9.2)
    fig.text(0.04, 0.894, "V＝每层读取量；C＝每层计算时间。读取线包含排队，不表示连续独占 SSD。", fontsize=10)
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", markerfacecolor="white", color="black", linestyle="none", label="发起读取"),
        plt.Line2D([], [], marker="s", color="black", linestyle="none", label="最后 SSD 完成"),
        Patch(facecolor="#eeeeee", edgecolor="#666666", hatch="////", label="NPU 等 I/O"),
    ], loc="upper center", bbox_to_anchor=(0.51, 0.883), ncol=3,
        fontsize=9.3, frameon=False, handletextpad=0.4, columnspacing=0.9)
    ax = fig.add_axes((0.15, 0.365, 0.825, 0.467))
    positions = {n: (10.35 - 3*n, 9.35 - 3*n) for n in range(4)}
    for row in _relative_layer_timeline(result, layers):
        npu = row["npu_id"]
        compute_y, _ = positions[npu]
        for start, end, waiting in ((row["deadline"], row["compute_start"], True),
                                    (row["compute_start"], row["compute_end"], False)):
            interval = _clip_interval(start, end, 0, 100)
            if interval:
                ax.broken_barh([interval], (compute_y - 0.23, 0.46),
                               facecolors="#eeeeee" if waiting else COLORS[npu],
                               edgecolors="#666666" if waiting else "none",
                               linewidth=0.3, hatch="////" if waiting else None)
    for event in events:
        npu = event["npu_id"]
        _, y = positions[npu]
        start, last = (event[k] for k in ("release_ms", "last_ssd_ms"))
        tag = tags.get((npu, event["request_id"], event["layer"]))
        alpha = 1 if tag else 0.36
        color = COLORS[npu]
        interval = _clip_interval(start, last, 0, 100)
        if interval:
            ax.plot([interval[0], sum(interval)], [y, y], color=color,
                    alpha=alpha, linewidth=2.0 if tag else 1.2, solid_capstyle="butt")
        for time, marker, marker_y, face in ((start, "o", y, "white"),
                                            (last, "s", y, color)):
            if 0 <= time <= 100:
                ax.plot(time, marker_y, marker=marker, linestyle="none", color=color,
                        markerfacecolor=face, markersize=4.6 if tag else 3.0,
                        alpha=alpha, markeredgewidth=0.9, zorder=5)
        if tag:
            label = f"2a：○→■ {last-start:.6f} ms" if tag == "2a" else tag
            ax.text((start + last)/2, y - 0.43, label, ha="center", va="top",
                    fontsize=9.2, color=color, fontweight="bold")
    for npu, (compute_y, read_y) in positions.items():
        ax.axhline(read_y - 0.87, color="#dddddd", linewidth=0.6)
        if npu in (2, 3):
            ax.text(72, compute_y, "持续计算", fontsize=10, ha="center", va="center",
                    color="white", fontweight="bold")
            left = selected[f"{npu}a"]["release_ms"]
            right = selected[f"{npu}b"]["release_ms"]
            arrow_y = read_y + 0.34
            ax.vlines([left, right], read_y, arrow_y, color=COLORS[npu], linewidth=0.8)
            ax.annotate("", xy=(right, arrow_y), xytext=(left, arrow_y),
                        arrowprops={"arrowstyle": "<->", "color": COLORS[npu], "linewidth": 1.1})
            ax.text((left + right)/2, arrow_y + 0.07,
                    f"{npu}a 的○ → {npu}b 的○：{intervals[npu]:.6f} ms",
                    ha="center", va="bottom", fontsize=8.4, color=COLORS[npu])
            # This is a gap in this NPU's reads, not in shared SSD activity.
            done = selected[f"{npu}a"]["last_ssd_ms"]
            ax.plot([done, right], [read_y, read_y], color="#999999", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 11.0)
    tick_y, tick_text = [], []
    for npu, (compute_y, read_y) in positions.items():
        tick_y.extend([compute_y, read_y])
        tick_text.extend([f"NPU{npu}\n计算/等待", f"NPU{npu}\n层读取"])
    ax.set_yticks(tick_y, tick_text)
    ax.tick_params(axis="y", labelsize=8.9, length=0, pad=6)
    for index, tick in enumerate(ax.get_yticklabels()):
        tick.set_color(COLORS[index//2])
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("测量窗口内的相对时间（ms）；图下表给出标记的精确时刻", fontsize=10)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.04, 0.307, "同页查时刻：a/b 是各卡两次相邻读取；深色示例对应下表，浅色为其余读取。", fontsize=10)
    table_ax = fig.add_axes((0.04, 0.110, 0.935, 0.180))
    table_ax.axis("off")
    cells = [[tag, f"R{e['request_id']}/L{e['layer']}",
              f"{e['release_ms']:.6f}", f"{e['last_ssd_ms']:.6f}"]
             for tag, e in selected.items()]
    table = table_ax.table(cellText=cells,
                          colLabels=["标记", "请求 / 层", "○ 发起 (ms)", "■ SSD 完成 (ms)"],
                          colWidths=[0.08, 0.30, 0.31, 0.31],
                          cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#edf2f5")
            cell.get_text().set_fontweight("bold")
        else:
            if col == 0:
                cell.get_text().set_color(COLORS[(row-1)//2])
                cell.get_text().set_fontweight("bold")
            if ((row-1)//2) % 2 == 0:
                cell.set_facecolor("#fafafa")
    fig.text(0.04, 0.075, "读法：发起间隔量的是○到下一次○，不是○到■。虚点线不是 SSD 空闲。", fontsize=10)
    fig.text(0.04, 0.048, "○→■ 为发起至 SSU/SSD 服务完成的延迟，含排队；不含后续接收链路尾部。", fontsize=9.6)
    fig.text(0.04, 0.022, "图边界不是读取结束；跨边界的读取照实截断。共享 SSD 实际服务请看下一页 B/C。", fontsize=9.6)
    return fig, {"display_endpoint": "last_ssd_ms", "selected_events": selected, "release_intervals_ms": intervals,
                 "all_events": events, "all_checks_pass": True}


def make_ssd_panels(result, blocks):
    """Keep B/C's original 100-ms scope and exact 0.1-ms bins on a new page."""
    origin = result["baseline"]["measurement_start_ms"]
    edges, service, idle = _bin_ssd_service(blocks, origin, 0, 100)
    fig = plt.figure(figsize=(8.8, 5.5))
    fig.suptitle("图 4（续）｜B/C 只看 SSD 真正服务的时间", fontsize=15, y=0.99)
    _profile_legend(fig, result, top=0.945, size=9.1)
    ax = fig.add_axes((0.13, 0.46, 0.85, 0.27))
    micro = fig.add_axes((0.13, 0.135, 0.85, 0.115))
    _draw_service_fractions(ax, edges, service, idle)
    ax.set_xlim(0, 100)
    ax.set_title("B  每 0.1 ms 分桶：色带厚度＝该卡服务时长 ÷ 0.1 ms × 100%", loc="left", fontsize=10.5, pad=12)
    ax.set_xlabel("与 A 相同的窗口内相对时间（ms）；灰色为空闲", fontsize=10)
    for row in blocks:
        start, end = (float(row[k]) - origin for k in ("ssd_start_time_ms", "ssd_end_time_ms"))
        interval = _clip_interval(start, end, 8, 8.1)
        if not interval:
            continue
        x, width = (interval[0]-8)*1000, interval[1]*1000
        npu = int(row["npu_id"])
        micro.broken_barh([(x, width)], (0.15, 0.7), facecolors=COLORS[npu], edgecolors="white", linewidth=0.7)
        if width >= 2.5:
            micro.text(x+width/2, 0.5, str(npu), ha="center", va="center",
                       color="black" if npu == 2 else "white", fontsize=10, fontweight="bold")
    micro.set_xlim(0, 100)
    micro.set_ylim(0, 1)
    micro.set_yticks([0.5], ["SSD 0"])
    micro.set_title("C  8.000–8.100 ms 的命令顺序：块内数字是 NPU 编号", loc="left", fontsize=10.5, pad=12)
    micro.set_xlabel("从 8.000 ms 起计的相对时间（µs）；满命令约 4.196 µs", fontsize=10)
    fig.text(0.04, 0.016, "B 的上下堆叠不表示并行服务；蓝条很窄是因为 NPU0 每层累计 SSD 服务仅 28.029 µs。", fontsize=9.7)
    return fig


def main():
    _configure_matplotlib()
    result = json.loads((DATA / "result.json").read_text())
    layers = _read_rows(DATA / "request_layer_timeline.csv")
    blocks = _read_rows(DATA / "physical_block_trace.csv")
    fig, report = make_read_event_figure(result, layers, blocks)
    _save_vector_and_png(fig, "05a_per_npu_ssu_events")
    _save_vector_and_png(make_ssd_panels(result, blocks), "05bc_ssd_service")
    with (DATA / "tutorial_read_event_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (DATA / "figure4_read_events_0_100ms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["all_events"][0]))
        writer.writeheader()
        writer.writerows(report["all_events"])
    print("Built Figure 4A per-NPU read events, unchanged B/C service, and event audit/CSV")


if __name__ == "__main__":
    main()
