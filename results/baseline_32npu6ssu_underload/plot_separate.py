#!/usr/bin/env python3
"""Create four Chinese single-plot figures from frozen results.

This script performs no simulation. It uses plot_case.load_case to check actual
layer intervals, post-admission I/O waits, fixed NPU placement, and exact
admission/completion event demand. Original comparison figures are untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from plot_case import BASE, COLORS, NPU, SSU, load_case


LEFT_MS, RIGHT_MS = 2000.0, 4000.0
OUT = BASE / "figures" / "separate"
CASES = (
    ("concurrency_l768_seed7", "random_baseline", "随机顺序", "每张卡分别打乱自己的任务清单"),
    ("concurrency_l768_seed7__exact_cohort4_p1", "ordered_baseline", "重排顺序", "四组卡采用不同起点，同组八张卡采用相同任务类型顺序"),
)


def configure_font():
    candidates = (
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError("A Chinese font is required; no supported local CJK font found")
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({
        "font.family": name, "font.size": 14, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "svg.hashsalt": "underload-separate-panorama-v1",
    })
    return path


def draw(case, filename, name, order_description, font_path, kind):
    audit = case["audit"]
    full_peak = max(audit["full_run_current_profile_peak_by_ssu_gib_s"])
    window_peak = max(audit["window_current_profile_peak_by_ssu_gib_s"])
    is_timeline = kind == "timeline"
    fig, ax = plt.subplots(figsize=(15, 10.4 if is_timeline else 7.6))
    fig.subplots_adjust(left=.075, right=.985, top=.778 if is_timeline else .740,
                        bottom=.180 if is_timeline else .285)
    subject = "32 卡计算与等待时间线" if is_timeline else "六盘名义需求"
    fig.suptitle(f"{name}的 Baseline：{subject}", fontsize=23, y=.976)
    fig.text(.5, .923 if is_timeline else .908,
             f"32 张 NPU / 6 块 SSU  ·  [2, 4) 秒  ·  设备计算利用率 U = {case['util'] * 100:.4f}%",
             ha="center", va="top", fontsize=17)
    fig.text(.5, .879 if is_timeline else .854,
             f"名义需求最热盘峰值：全程 {full_peak:.4f} GiB/s  /  本窗口 {window_peak:.4f} GiB/s",
             ha="center", va="top", fontsize=14.5)
    fig.text(.5, .843 if is_timeline else .803, order_description,
             ha="center", va="top", fontsize=14, color="#454545")
    if is_timeline:
        legend = [Patch(color=COLORS["short"], label="短任务计算"),
                  Patch(color=COLORS["long"], label="长任务计算"),
                  Patch(color=COLORS["stall"], label="I/O 等待（任务已接纳）"),
                  Patch(color="#eceff1", label="空闲（本窗口为 0）")]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.52, .820),
                   ncol=4, frameon=False, fontsize=13, columnspacing=1.5)
        for npu, spans in enumerate(case["spans"]):
            ax.broken_barh([(LEFT_MS, RIGHT_MS - LEFT_MS)], (npu-.41, .82),
                          facecolors="#eceff1", edgecolors="none")
            for role in ("short", "long", "stall"):
                if spans[role]:
                    ax.broken_barh(
                        spans[role], (npu-.41, .82), facecolors=COLORS[role],
                        edgecolors="none", linewidth=0, antialiased=False, rasterized=True,
                    )
        ax.set_ylim(NPU-.2, -.8)
        ax.set_yticks(range(NPU))
        ax.tick_params(axis="y", labelsize=11, length=3)
        ax.set_ylabel("计算卡编号（NPU）", fontsize=16, labelpad=12)
        for border in (7.5, 15.5, 23.5):
            ax.axhline(border, color="#c9cfd2", linewidth=.7, zorder=0)
        fig.text(.075, .087,
                 "蓝色 / 绿色：短 / 长任务的真实计算区间；橙色：接纳后的 I/O 等待，不包含接纳前排队。",
                 fontsize=12.5)
        fig.text(.075, .055,
                 "32 张卡整窗有任务，区间未移动或拼接。标题中的名义需求为当前画像 V/C，不是 SSD 实际吞吐。",
                 fontsize=12.5)
    else:
        palette = plt.get_cmap("tab10").colors
        for ssu in range(SSU):
            ax.step(case["times"], case["rates"][:, ssu], where="post",
                    color=palette[ssu], linewidth=1.2, alpha=.85, label=f"盘 {ssu}")
        ax.axhline(40, color="#aa2424", linestyle="--", linewidth=1.7,
                   label="每盘容量 40 GiB/s")
        ax.set_ylim(0, 43)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel("每盘名义需求（GiB/s）", fontsize=16, labelpad=12)
        ax.grid(axis="y", color="#d9dee1", linewidth=.6)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .164),
                   ncol=7, frameon=False, fontsize=12, columnspacing=1.3)
        fig.text(.075, .126,
                 "名义需求 = 当前已接纳任务的每层盘上读取量 ÷ 每层计算时间；六条阶梯线不是 SSD 实际吞吐。",
                 fontsize=12.5)
        fig.text(.075, .083,
                 "按全部接纳 / 完成事件更新，不额外叠加下一请求首层预取。低于容量线不保证所有突发及时完成。",
                 fontsize=12.5)
    ax.set_xlim(LEFT_MS, RIGHT_MS)
    ax.set_xticks([2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value/1000:g}"))
    ax.set_xlabel("仿真时间（秒）", fontsize=16, labelpad=8)
    fig.text(.075, .023,
             f"种子 7  |  输入指纹 {audit['input_fingerprint'][:16]}  |  仿真结果，非硬件实测",
             fontsize=11, color="#5c6164")

    metadata = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_loader_sha256": hashlib.sha256((BASE / "plot_case.py").read_bytes()).hexdigest(),
        "font_path": str(font_path), "figure_scope": "One case, one plot; actual [2000,4000) ms",
        "plot_kind": kind,
        "audit": audit,
    }
    description = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / filename
    fig.savefig(stem.with_suffix(".png"), dpi=280, metadata={"Description": description})
    fig.savefig(stem.with_suffix(".svg"), dpi=280, metadata={"Description": description})
    fig.savefig(stem.with_suffix(".pdf"), dpi=280,
                metadata={"Title": f"{name}的 Baseline：{subject}", "Subject": description,
                          "Creator": "plot_separate.py / matplotlib"})
    plt.close(fig)
    return {"output_stem": str(stem), "input_fingerprint": audit["input_fingerprint"],
            "device_U": case["util"], "full_run_peak_gib_s": full_peak,
            "window_peak_gib_s": window_peak, "all_32_active": audit["all_32_active"]}


def main():
    font_path = configure_font()
    cases = [load_case(label, LEFT_MS, RIGHT_MS) for label, *_ in CASES]
    assert cases[0]["population"] == cases[1]["population"], "Unmatched request populations"
    outputs = []
    for case, (_, filename, name, explanation) in zip(cases, CASES):
        outputs.append(draw(case, filename, name, explanation, font_path, "timeline"))
        outputs.append(draw(case, filename.replace("_baseline", "_demand"), name,
                            explanation, font_path, "demand"))
    print(json.dumps({"population_and_placement_matched": True, "figures": outputs},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
