#!/usr/bin/env python3
"""Plot two existing Baseline traces; no simulation and no parent-file edits."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

import plot_case
from plot_separate import configure_font
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

START, END = 2000.0, 4000.0
OUT = HERE / "figures"
CASES = (
    ("raw_feasible_q2_seed7", "raw_random_baseline_seed7", "Random 随机顺序",
     "每张卡独立打乱完整请求清单；同一批原始画像、同一物理 placement"),
    ("raw_feasible_q2_seed7__exact_cohort4_p1", "raw_ordered_baseline_seed7", "Ordered 四组重排",
     "exact_cohort4_p1：四组采用不同起点，同组八张卡采用相同画像顺序"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(case, stem, name, description):
    fig, ax = plt.subplots(figsize=(15, 10.4))
    fig.subplots_adjust(left=.075, right=.985, top=.773, bottom=.180)
    fig.suptitle(f"原始四画像 · {name}的 Baseline 时间线", fontsize=23, y=.977)
    fig.text(.5, .929,
             f"32 NPU / 6 SSU · 每盘 40 GiB/s · [2, 4) 秒 · 设备 U = {100*case['util']:.4f}%",
             ha="center", va="top", fontsize=16.5)
    fig.text(.5, .884, description, ha="center", va="top", fontsize=14, color="#454545")
    fig.text(.5, .848,
             "原始 data：32K/1024、48K/1024、64K/1024（SL）；160K/1024（LL）",
             ha="center", va="top", fontsize=13.5)
    fig.legend(handles=[
        Patch(color=plot_case.COLORS["short"], label="短画像计算（SL）"),
        Patch(color=plot_case.COLORS["long"], label="长画像计算（LL）"),
        Patch(color=plot_case.COLORS["stall"], label="接纳后 I/O 等待"),
        Patch(color="#eceff1", label="空闲（本窗为 0）"),
    ], loc="upper center", bbox_to_anchor=(.52, .820), ncol=4,
        frameon=False, fontsize=13, columnspacing=1.5)
    for npu, spans in enumerate(case["spans"]):
        ax.broken_barh([(START, END-START)], (npu-.41, .82),
                       facecolors="#eceff1", edgecolors="none")
        for role in ("short", "long", "stall"):
            if spans[role]:
                ax.broken_barh(spans[role], (npu-.41, .82),
                               facecolors=plot_case.COLORS[role], edgecolors="none",
                               linewidth=0, antialiased=False, rasterized=True)
    ax.set_ylim(31.8, -.8)
    ax.set_yticks(range(32))
    ax.tick_params(axis="y", labelsize=11, length=3)
    ax.set_ylabel("计算卡编号（NPU）", fontsize=16, labelpad=12)
    ax.set_xlim(START, END)
    ax.set_xticks(range(2000, 4001, 250))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}"))
    ax.set_xlabel("仿真时间（秒）", fontsize=16, labelpad=8)
    for border in (7.5, 15.5, 23.5):
        ax.axhline(border, color="#c9cfd2", linewidth=.7, zorder=0)
    fig.text(.075, .095,
             "蓝 / 绿：真实逐层计算；橙：接纳后实际 I/O barrier 等待，不包含接纳前排队。",
             fontsize=12.5)
    fig.text(.075, .064,
             "32 卡整窗 active；区间未移动、拼接或加宽。极短等待可能不足一个像素，精确总量见审计 JSON。",
             fontsize=12)
    fig.text(.075, .033,
             f"种子 7 | 输入 {case['audit']['input_fingerprint'][:16]} | 原始 C/V 未缩放或填充 | 仿真结果",
             fontsize=11.5, color="#5c6164")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            outside.append({"text": text.get_text(), "bounds": list(box.extents)})
    assert not outside, outside
    metadata = {"script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
                "audit": case["audit"], "all_visible_text_inside_canvas": True}
    description_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, metadata={"Description": description_json})
    fig.savefig(stem.with_suffix(".pdf"), dpi=240,
                metadata={"Title": f"原始四画像 {name} Baseline 时间线",
                          "Subject": description_json, "Creator": "plot_raw_timelines.py / matplotlib"})
    plt.close(fig)
    return {"png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf')),
            "png_sha256": sha(stem.with_suffix('.png')), "pdf_sha256": sha(stem.with_suffix('.pdf')),
            "all_visible_text_inside_canvas": True}


def main():
    font = configure_font()
    # Process-local path redirection only; the frozen parent loader is unchanged.
    plot_case.BASE = HERE
    cases = [plot_case.load_case(label, START, END) for label, *_ in CASES]
    assert cases[0]["population"] == cases[1]["population"]
    OUT.mkdir(exist_ok=True)
    audits = []
    for case, (label, filename, name, description) in zip(cases, CASES):
        a = case["audit"]
        compute = math.fsum(math.fsum(width for _, width in s[r])
                            for s in case["spans"] for r in ("short", "long"))
        stall = math.fsum(math.fsum(width for _, width in s["stall"]) for s in case["spans"])
        assert math.isclose(compute, math.fsum(a["window_compute_ms_by_npu"]), abs_tol=1e-7)
        assert math.isclose(stall, math.fsum(a["window_stall_ms_by_npu"]), abs_tol=1e-7)
        assert math.isclose(compute+stall, 32*(END-START), abs_tol=1e-7)
        a.update(manifest_sha256=sha(a["manifest"]), total_compute_npu_ms=compute,
                 total_io_stall_npu_ms=stall, total_idle_npu_ms=32*(END-START)-compute-stall,
                 plotted_spans_match_verified_layer_and_io_barrier_intervals=True)
        audits.append({**draw(case, OUT/filename, name, description), "source_audit": a})
    audit = {"no_simulation": True, "window_ms": [START, END],
             "population_and_actual_placement_matched": True,
             "script_sha256": sha(__file__),
             "reused_sources": {str(p): sha(p) for p in (PARENT/'plot_case.py', PARENT/'plot_separate.py')},
             "font": str(font), "font_sha256": sha(font), "figures": audits,
             "definitions": {"U": "clipped actual compute / (32*2000ms)",
                 "stall": "admission/previous layer compute_end to current compute_start; checked against io_barrier_wait_ms",
                 "profile_short_label": "relative analysis role; all three raw short profiles are SL policy category",
                 "tiny_spans": "true widths retained, never enlarged to a minimum visible pixel"}}
    (OUT/'timeline_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({"figures": [{"label": x['source_audit']['label'],
          "U_percent": 100*x['source_audit']['fleet_compute_utilization'],
          "compute_npu_ms": x['source_audit']['total_compute_npu_ms'],
          "stall_npu_ms": x['source_audit']['total_io_stall_npu_ms'],
          "png": x['png']} for x in audits]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
