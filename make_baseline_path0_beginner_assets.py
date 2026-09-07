#!/usr/bin/env python3
"""Trace-derived teaching diagrams with event labels and recurring V/C reminders."""

from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from baseline_path0_layout import DATA
from make_baseline_path0_tutorial_assets import (
    COLORS, _configure_matplotlib, _read_rows, _relative_layer_timeline,
    _clip_interval, _save_vector_and_png, _profile_legend,
    _npu2_read_example, _bin_ssd_service,
)
from validate_baseline_path0_tutorial import coverage


def cycle_report(result, raw_rows):
    rows = _relative_layer_timeline(result, raw_rows)
    releases = sorted(r["release"] for r in rows if r["npu_id"] == 3 and 0 <= r["release"] < 100)
    start, end = releases[:2]
    period = end - start
    assert math.isclose(period, result["input"]["profiles"][3]["per_layer_compute_us"] / 1000, abs_tol=1e-7)
    lanes = []
    for p in result["input"]["profiles"]:
        rr = [r for r in rows if r["npu_id"] == p["npu_id"]]
        compute = coverage([(r["compute_start"], r["compute_end"]) for r in rr], start, end)
        wait = coverage([(r["deadline"], r["compute_start"]) for r in rr], start, end)
        assert math.isclose(compute + wait, period, abs_tol=1e-7)
        count = compute / (p["per_layer_compute_us"] / 1000)
        assert math.isclose(count, (11, 2, 1, 1)[p["npu_id"]], abs_tol=1e-7)
        lanes.append({"npu_id": p["npu_id"], "compute_ms": compute, "wait_ms": wait,
                      "equivalent_layers": round(count), "compute_per_layer_ms": p["per_layer_compute_us"] / 1000})
    return {"start_ms": start, "end_ms": end, "period_ms": period, "lanes": lanes}


def make_cycle_ledger(result, rows):
    report = cycle_report(result, rows)
    p = report["period_ms"]
    fig = plt.figure(figsize=(8.8, 7.9))
    fig.suptitle("周期账本｜先确定 P，再相加计算与等待", fontsize=16, y=0.99)
    _profile_legend(fig, result)
    fig.text(0.08, 0.85, "V＝每层读取量；C＝每层计算时间。P 是重复间隔，不是请求耗时或读取延迟。", fontsize=10.5)
    fig.text(0.08, 0.79, f"① 选 NPU3 两次发起：{report['start_ms']:.6f} → {report['end_ms']:.6f} ms", fontsize=12)
    fig.text(0.08, 0.745, f"② 相减得 P = {report['end_ms']:.6f} − {report['start_ms']:.6f} = {p:.6f} ms", fontsize=12)
    ax = fig.add_axes((0.18, 0.285, 0.77, 0.35))
    for lane in report["lanes"]:
        y = 3 - lane["npu_id"]
        c, wait = lane["compute_ms"], lane["wait_ms"]
        ax.broken_barh([(0, c)], (y - 0.22, 0.44), facecolors=COLORS[lane["npu_id"]])
        if wait:
            ax.broken_barh([(c, wait)], (y - 0.22, 0.44), facecolors="#eeeeee",
                           edgecolors="#777777", hatch="////", linewidth=0.4)
        ax.text(0.4, y + 0.30, f"计算 {c:.6f} + 等待 {wait:.6f} ≈ {p:.6f} ms", fontsize=10)
    ax.set(xlim=(0, p), ylim=(-0.55, 3.83), yticks=[3, 2, 1, 0],
           yticklabels=["NPU0", "NPU1", "NPU2", "NPU3"])
    ax.set_xticks([0, p], ["0", f"P={p:.6f}"])
    ax.set_xlabel("累计时长（ms）：先画计算总量，再画等待总量；不表示真实先后顺序", fontsize=10)
    ax.set_title("③ 在同一个周期内累计；每行相加都是 P", loc="left", fontsize=12, pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    explanations = [
        "NPU2/3：1 × 29.856563 + 0 ≈ P；读取提前到齐，因此没有暴露等待。",
        "NPU1：2 × 12.390165 + 5.076233 ≈ P；算得更快，但中途等了数据。",
        "NPU0：11 × 0.582149 + 23.452918 ≈ P；显示值舍入，核算使用原始精度。",
        "1、2、11 是累计计算折合的层数；周期边界可切到半层，不是完整请求个数。",
    ]
    for i, line in enumerate(explanations):
        fig.text(0.08, 0.185 - i * 0.038, line, fontsize=10.5)
    return fig, report


def _event_table(ax, events, widths=(0.08, 0.25, 0.67)):
    ax.axis("off")
    table = ax.table(cellText=events, colLabels=["标记", "时刻（ms）", "事件含义"],
                     cellLoc="left", colLoc="left", colWidths=widths, bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#edf2f5")
            cell.set_text_props(weight="bold")
    return table


def make_npu1_causal_story(result, audit):
    r = audit["npu1_causal_example"]
    a, b, c, d, e = (r[k] for k in ("release_relative_ms", "first_service_relative_ms",
                                      "deadline_relative_ms", "last_ssd_completion_relative_ms", "ready_relative_ms"))
    fig = plt.figure(figsize=(8.8, 9.0))
    fig.suptitle("NPU1｜R100020 / L3 为什么晚到？", fontsize=16, y=0.99)
    _profile_legend(fig, result)
    fig.text(0.075, 0.864, "本例：读 263.506 MiB；计算预算 12.390165 ms；自身 SSD 服务只需 6.433249 ms。", fontsize=10.2)
    ax = fig.add_axes((0.17, 0.53, 0.78, 0.23))
    ax.broken_barh([(a, b-a)], (1.0, 0.4), color="#f0d264")
    ax.broken_barh([(b, d-b)], (1.0, 0.4), color=COLORS[1])
    ax.broken_barh([(a, c-a)], (0.1, 0.4), color=COLORS[1])
    ax.broken_barh([(c, e-c)], (0.1, 0.4), facecolor="#eeeeee", edgecolor="#555555", hatch="////")
    ax.text((a+b)/2, 1.2, f"首次服务前等待\n{b-a:.6f} ms", ha="center", va="center", fontsize=10.2)
    ax.text((b+d)/2, 1.2, f"SSD 服务\n{d-b:.6f} ms", ha="center", va="center", color="white", fontsize=10.2)
    ax.text((a+c)/2, 0.3, "计算前一层 L2（掩盖读取的预算）", ha="center", va="center", fontsize=10, color="white")
    ax.text((c+e)/2, 0.3, f"等 L3\n{e-c:.6f} ms", ha="center", va="center", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1})
    for x in (a, b, c, e):
        ax.axvline(x, color="#666666", linestyle="--", linewidth=0.7)
    ax.set_xticks([a, b, c, e], [f"a\n{a:.6f}", f"b\n{b:.6f}", f"c\n{c:.6f}", "d/e\n见尾部放大"])
    ax.tick_params(axis="x", labelsize=9.2)
    ax.get_xticklabels()[1].set_ha("right")
    ax.get_xticklabels()[2].set_ha("left")
    ax.get_xticklabels()[3].set_ha("right")
    ax.set(xlim=(a-0.25, e+0.25), ylim=(-0.08, 1.65), yticks=[1.2, 0.3], yticklabels=["L3 读取", "NPU1 状态"])
    ax.set_title("A  同一时间原点：窗口内的相对时刻（不是从 a 重新计时）", loc="left", fontsize=11.3, pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    tail = fig.add_axes((0.25, 0.325, 0.60, 0.075))
    tail_us = (e-d)*1000
    tail.plot([0, tail_us], [0.5, 0.5], color="#333333", linewidth=3, marker="o")
    tail.set(xlim=(-0.15, tail_us+0.15), ylim=(0, 1), yticks=[])
    tail.set_xticks([0, tail_us], [f"d：{d:.6f} ms", f"e：{e:.6f} ms"])
    tail.set_title(f"B  接收尾部放大：d→e = {tail_us:.6f} µs（不是 ms）", fontsize=11.3, pad=13)
    tail.spines[["top", "right", "left"]].set_visible(False)
    table_ax = fig.add_axes((0.075, 0.07, 0.86, 0.19))
    _event_table(table_ax, [
        ["a", f"{a:.6f}", "开始算 L2，同时发起 L3 读取"],
        ["b", f"{b:.6f}", "L3 的第一条命令终于开始 SSD 服务"],
        ["c", f"{c:.6f}", "L2 算完；L3 尚未到齐，开始等待"],
        ["d", f"{d:.6f}", "L3 的最后一条命令完成 SSD 服务"],
        ["e", f"{e:.6f}", "L3 到齐 HBM，等待结束，开始算 L3"],
    ])
    fig.text(0.075, 0.025, "结论：b−a 的排队耗掉提前量；从 a 到 e 共 17.466398 ms，超过预算 5.076233 ms。", fontsize=10.1)
    return fig


def make_npu0_causal_story(result, raw_rows):
    rows = _relative_layer_timeline(result, raw_rows)
    selected = {r["layer"]: r for r in rows if r["npu_id"] == 0 and r["request_id"] == 111}
    l6, l7 = selected[6], selected[7]
    a, b, c, d, e = l6["release"], l6["deadline"], l6["ready"], l7["deadline"], l7["ready"]
    assert math.isclose(c, l7["release"], abs_tol=1e-7)
    fig = plt.figure(figsize=(8.8, 7.6))
    fig.suptitle("NPU0｜只读 1.148 MiB，为什么仍长时间停顿？", fontsize=15, y=0.99)
    _profile_legend(fig, result)
    fig.text(0.075, 0.849, "本例固定为 R111 的 L6、L7：每层计算 0.582149 ms；自身 SSD 服务 0.028029 ms。", fontsize=10.3)
    ax = fig.add_axes((0.17, 0.44, 0.78, 0.31))
    for layer, y in ((6, 2), (7, 1)):
        r = selected[layer]
        ax.plot([r["release"], r["ready"]], [y, y], color=COLORS[0], linewidth=5)
        ax.plot(r["release"], y, marker="o", markerfacecolor="white", markeredgecolor=COLORS[0], markersize=6)
        ax.plot(r["ready"], y, marker="D", color=COLORS[0], markersize=5)
        text_x = r["release"] if layer == 6 else (r["release"]+r["ready"])/2
        ax.text(text_x, y+0.26, f"发起→到齐 {r['ready']-r['release']:.6f} ms",
                ha="left" if layer == 6 else "center", fontsize=10)
    for start, end, color in ((a, b, COLORS[0]), (b, c, "#eeeeee"),
                               (c, d, COLORS[0]), (d, e, "#eeeeee")):
        ax.broken_barh([(start, end-start)], (-0.18, 0.36), facecolor=color,
                       edgecolor="#666666", linewidth=0.5, hatch="////" if color == "#eeeeee" else None)
    ax.text((d+e)/2, 0, f"等待 L7：{e-d:.6f} ms", ha="center", va="center", fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    for key, x in zip("abcde", (a, b, c, d, e)):
        ax.axvline(x, color="#888888", linewidth=0.7, linestyle="--")
        ax.text(x, -0.38, key, ha="center", va="top", fontweight="bold", fontsize=11)
    ax.set(xlim=(a-0.3, e+0.3), ylim=(-0.65, 2.65), yticks=[2, 1, 0],
           yticklabels=["R111/L6 读取", "R111/L7 读取", "NPU0 算/等"])
    ax.set_xticks([a, c, e], [f"{a:.6f}", f"{c:.6f}", f"{e:.6f}"])
    ax.set_xlabel("窗口内相对时刻（ms）；近邻事件 b/d 的精确时刻列在下表", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    table_ax = fig.add_axes((0.075, 0.12, 0.86, 0.235))
    _event_table(table_ax, [
        ["a", f"{a:.6f}", "开始算 L5，同时发起 L6 读取"],
        ["b", f"{b:.6f}", "L5 算完，开始等待 L6"],
        ["c", f"{c:.6f}", "L6 到齐并开始计算，同时发起 L7 读取"],
        ["d", f"{d:.6f}", "L6 算完，开始等待 L7"],
        ["e", f"{e:.6f}", "L7 到齐，开始计算 L7；本图在此结束"],
    ])
    fig.text(0.075, 0.065, f"等 L6：c−b = {c-b:.6f} ms；等 L7：e−d = {e-d:.6f} ms。", fontsize=11)
    fig.text(0.075, 0.03, "空心圆为发起，实心菱形为到齐；蓝读线包含排队，灰斜纹是 NPU 已无法继续计算。", fontsize=10.2)
    return fig, {"a": a, "b": b, "c": c, "d": d, "e": e, "wait_l6_ms": c-b, "wait_l7_ms": e-d}


def make_estimator_examples(result):
    fig = plt.figure(figsize=(8.8, 6.7))
    fig.suptitle("估算函数｜这两例是预测，不是已采集的 trace", fontsize=15, y=0.99)
    fig.text(0.07, 0.928, "共同约定：现在记为 t=0；只预测 SSD 阶段；不含最终传入 HBM 的接收尾部。", fontsize=10.5)
    top = fig.add_axes((0.17, 0.635, 0.74, 0.16))
    top.barh([0], [1.1], height=0.45, color="#98b9d1")
    top.axvline(1.0, color="#b02d2d", linestyle="--")
    top.set(xlim=(0, 1.15), ylim=(-0.5, 0.5), yticks=[])
    top.set_xticks([0, 1.0, 1.1], ["现在 0", "截止\n1.000", "完成\n1.100"])
    top.tick_params(labelsize=9)
    top.get_xticklabels()[1].set_ha("right")
    top.get_xticklabels()[2].set_ha("left")
    top.set_title("例 A：100 条排前方 + 自己 10 条；4 KiB/条；上限 100,000 IOPS", loc="left", fontsize=11.2, pad=16)
    fig.text(0.09, 0.555, "110 ÷ 100,000 = 1.1 ms；允许 1.0 ms，因此欠缺 0.1 ms。此例不是 NPU0 的输入画像。", fontsize=10.2)
    ax = fig.add_axes((0.17, 0.30, 0.74, 0.15))
    own = result["input"]["profiles"][0]["per_layer_kv_gib"] / 40 * 1000
    budget = result["input"]["profiles"][0]["per_layer_compute_us"] / 1000
    total = 6.25 + own
    ax.barh([0], [6.25], height=0.45, color="#f0d264")
    ax.barh([0], [own], left=[6.25], height=0.45, color=COLORS[0])
    ax.axvline(budget, color="#b02d2d", linestyle="--")
    ax.set(xlim=(0, total+0.35), ylim=(-0.5, 0.5), yticks=[])
    ax.set_xticks([0, budget, total], ["现在 0", f"截止\n{budget:.6f}", f"完成\n{total:.6f}"])
    ax.tick_params(labelsize=9)
    ax.get_xticklabels()[0].set_ha("right")
    ax.get_xticklabels()[1].set_ha("left")
    ax.set_title("例 B：前方 Q=256 MiB；NPU0 自己 R=1.148 MiB；SSD=40 GiB/s", loc="left", fontsize=11.2, pad=16)
    ax.text(3.0, 0, "先服务前方字节：6.250000 ms", ha="center", va="center", fontsize=10.5)
    fig.text(0.09, 0.17, f"自己仅需 {own:.6f} ms（末端蓝段）；累计 {total:.6f} ms，欠缺 {total-budget:.6f} ms。", fontsize=10.5)
    fig.text(0.09, 0.115, "Q 是确定排在目标前方的剩余字节，不是整个 Path 的全部字节；不要重复计入目标 R。", fontsize=10.2)
    fig.text(0.09, 0.07, "图内时间单位均为 ms。例 B 的短蓝段未人为加宽，其 0.028029 ms 时长在文字中直接标出。", fontsize=10.2)
    for panel in (top, ax):
        panel.spines[["top", "right", "left"]].set_visible(False)
    return fig


def make_prefetch_budget(result):
    profile = result["input"]["profiles"][0]
    c = profile["per_layer_compute_us"] / 1000
    v = profile["per_layer_kv_gib"] * 1024
    counts = [2, 29, 30]
    fig = plt.figure(figsize=(8.8, 5.7))
    fig.suptitle("深预取预算｜NPU0 的“一层提前量”只有 0.582149 ms", fontsize=15, y=0.98)
    fig.text(0.08, 0.905, "每份 NPU0 KV=1.148071 MiB；以下仅为固定延迟下的预算估算，未实现或验证深预取。", fontsize=10.4)
    ax = fig.add_axes((0.19, 0.35, 0.75, 0.40))
    for y, n in zip([2, 1, 0], counts):
        ax.barh([y], [n*c], height=0.45, color=COLORS[0], alpha=0.8)
        ax.text(0.1, y+0.29, f"{n}C = {n*c:.6f} ms；对应 KV {n*v:.3f} MiB", fontsize=10.1)
    ax.axvline(16.7, color="#b02d2d", linestyle="--")
    ax.set(xlim=(0, 19), ylim=(-0.55, 2.8), yticks=[2, 1, 0], yticklabels=["提前 2 份", "提前 29 份", "提前 30 份"])
    ax.set_xticks([0, 16.7], ["0", "坏排队参照：16.7 ms"])
    ax.set_xlabel("可提供的累计计算窗口（ms），不表示这些层已经能够全部提前发起", fontsize=10.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.235, "2 份只能提供 1.164299 ms；29 份才超过 16.7 ms；严格覆盖 17 ms 需要 30 份。", fontsize=10.5)
    fig.text(0.08, 0.175, "NPU2/3 每份 KV=262.797 MiB，同样缓存 29 份要约 7.442 GiB，远大于 NPU0。", fontsize=10.5)
    fig.text(0.08, 0.11, "每请求仅 8 层，29 份意味着更深跨请求/跨层机制；还要求未来地址已知、HBM 容量够。", fontsize=10.3)
    fig.text(0.08, 0.055, "更多预取也会改变排队，所以“29”不是收益承诺，更不是当前仿真已经开启的层深。", fontsize=10.3)
    return fig


def main():
    _configure_matplotlib()
    with (DATA / "result.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    with (DATA / "tutorial_numeric_audit.json").open(encoding="utf-8") as handle:
        audit = json.load(handle)
    rows = _read_rows(DATA / "request_layer_timeline.csv")
    blocks = _read_rows(DATA / "physical_block_trace.csv")
    fig, cycle = make_cycle_ledger(result, rows)
    _save_vector_and_png(fig, "11_cycle_budget_explained")
    _save_vector_and_png(make_npu1_causal_story(result, audit), "09_npu1_queue_causality_annotated")
    fig, short = make_npu0_causal_story(result, rows)
    _save_vector_and_png(fig, "12_npu0_two_layer_story")
    _save_vector_and_png(make_estimator_examples(result), "13_fifo_budget_examples")
    _save_vector_and_png(make_prefetch_budget(result), "14_prefetch_budget")
    origin = result["baseline"]["measurement_start_ms"]
    edges, service, idle = _bin_ssd_service(blocks, origin, 0, 10)
    examples = []
    for k in (52, 80):
        width = edges[k+1] - edges[k]
        examples.append({"bin_ms": edges[k:k+2], "npu_service_us": [v[k]*1000 for v in service],
                         "npu_percent": [v[k]/width*100 for v in service], "idle_percent": idle[k]/width*100})
    report = {"cycle": cycle, "npu0_events": short, "all_checks_pass": True,
              "npu2_read": _npu2_read_example(result, rows, blocks), "ssd_time_bin_examples": examples}
    (DATA / "tutorial_beginner_event_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Built event-labeled beginner assets: cycle, NPU1, NPU0, deadline examples, prefetch budgets")


if __name__ == "__main__":
    main()
