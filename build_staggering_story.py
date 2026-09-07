#!/usr/bin/env python3
"""Build an illustrated, deliberately fictional FIFO/pre-fetch teaching PDF.

Only writes results/path0_staggering_story. No real simulator data is changed.
Requires matplotlib, pandoc and xelatex for publishing, not for the toy model.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from toy_staggering_model import get_story_data


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/path0_staggering_story"
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 11,
    "axes.unicode_minus": False, "pdf.fonttype": 3,
})
SMALL, LARGE = "#1976A3", "#C69C6D"
WORK, STALL, COLD, INK = "#C7E6DA", "#F7CBC3", "#E5E7EB", "#243447"
NAMES = {"A": "小林", "B": "老周"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bar(ax, left, right, y, text, color, *, hatch=None, height=.52, fontsize=11):
    ax.add_patch(Rectangle((left, y-height/2), right-left, height,
                           facecolor=color, edgecolor=INK, linewidth=.65, hatch=hatch))
    if text:
        ax.text((left+right)/2, y, text, ha="center", va="center", fontsize=fontsize,
                color="white" if color == SMALL else INK)


def axes(ax, *, end, ticks, labels):
    ax.set_xlim(-.15, end+.15)
    ax.set_ylim(-.65, len(labels)-.2)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("同一只时钟：第几秒（刻度选取图中的事件时刻）", fontsize=10)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def one_order(stem, release, compute, headline):
    """One next-dish order only; later orders intentionally not depicted."""
    ready, first = max(6, release)+1, max(6, release)
    deadline = release+compute
    assert release in (1, 3) and ready == 7
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    fig.subplots_adjust(left=.14, right=.98, top=.77, bottom=.22)
    axes(ax, end=10, ticks=sorted({0, release, deadline, 6, 7, 10}),
         labels=["小林的订单", "小林在做什么", "窗口实际装袋"])
    for i in range(6):
        bar(ax, i, i+1, 2, "周", LARGE, fontsize=11)
    bar(ax, 6, 7, 2, "林", SMALL, fontsize=11)
    ax.text(8.5, 2, "后续订单未画出", ha="center", color="#6B7280", fontsize=10)
    bar(ax, release, deadline, 1, f"做当前菜 {compute} 秒", WORK)
    if ready > deadline:
        bar(ax, deadline, ready, 1, f"停工 {ready-deadline} 秒", STALL, hatch="///")
    ax.text(max(ready, deadline), 1.43, f"{max(ready, deadline)} 秒：下一菜开工",
            ha="center", fontsize=10)
    ax.plot([release, first], [0, 0], color=INK, lw=2, linestyle="--")
    ax.plot([first, ready], [0, 0], color=SMALL, lw=5, solid_capstyle="butt")
    ax.scatter([release], [0], s=45, color=INK, marker="o", zorder=5)
    ax.scatter([ready], [0], s=45, color=INK, marker="s", zorder=5)
    ax.text((release+first)/2, -.28, f"订单排队 {first-release} 秒", ha="center", fontsize=10)
    ax.text(release, .22, f"{release} 秒下单", ha="center", fontsize=10)
    ax.text(ready, .22, "7 秒到齐", ha="center", fontsize=10)
    ax.axvline(deadline, color="#A33B2B", linestyle=":", linewidth=1.6)
    fig.suptitle(headline, x=.53, y=.98, fontsize=14, fontweight="bold")
    fig.text(.53, .855, f"老周：6袋，0秒已排入队列；小林：1袋，{release}秒下单，必须在{deadline}秒前拿到",
             ha="center", fontsize=10.5)
    save(fig, stem)
    return {"release_s": release, "compute_s": compute, "deadline_s": deadline,
            "ready_s": ready, "queue_s": first-release, "own_service_s": 1,
            "stall_s": max(0, ready-deadline)}


def chronological(stem, case, *, origin, end, headline, ticks):
    """Real chronological tracks, never packed compute/wait totals."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    fig.subplots_adjust(left=.13, right=.985, top=.76, bottom=.20)
    axes(ax, end=end, ticks=ticks,
         labels=["老周：做菜 / 停工", "小林：做菜 / 停工", "窗口：实际装袋"])
    row_y = {"A": 1, "B": 0}
    for io in case["io_services"]:
        left, right = max(0, io["start_s"]-origin), min(end, io["end_s"]-origin)
        if right > left:
            bar(ax, left, right, 2, "林" if io["name"] == "A" else "周",
                SMALL if io["name"] == "A" else LARGE, fontsize=9)
    # Actual idle gaps, independently derived from the service intervals.
    occupied = sorted((max(0, x["start_s"]-origin), min(end, x["end_s"]-origin))
                      for x in case["io_services"]
                      if x["start_s"] < origin+end and x["end_s"] > origin)
    cursor = 0
    for left, right in occupied+[(end, end)]:
        if left > cursor:
            ax.text((cursor+left)/2, 2, "空闲", ha="center", va="center", fontsize=9)
        cursor = max(cursor, right)
    for name in row_y:
        if origin == 0:
            first = min(x["start_s"] for x in case["computes"] if x["name"] == name)
            bar(ax, 0, first, row_y[name], "首料", COLD, fontsize=9)
    for typ, color, hatch in (("computes", WORK, None), ("stalls", STALL, "///")):
        for item in case[typ]:
            left = max(0, item["start_s"]-origin)
            right = min(end, item["end_s"]-origin)
            if right <= left:
                continue
            label = ("做菜" if typ == "computes" else "停工")
            if item["end_s"] > origin+end:
                label = "做到图外" if right-left >= 2 else "续"
            elif right-left >= 2:
                label += f"{right-left:g}秒"
            else:
                label = "尾段" if item["start_s"] < origin else label
            bar(ax, left, right, row_y[item["name"]], label, color,
                hatch=hatch, fontsize=10)
    for order in case["orders"]:
        time = order["release_s"]-origin
        if not order["cold_start"] and 0 <= time <= end:
            y = row_y[order["name"]]
            ax.scatter([time], [y+.37], color=INK, s=22, zorder=5)
            ax.text(time, y+.50, f"{time:g}下单", ha="center", fontsize=9)
    fig.suptitle(headline, x=.54, y=.98, fontsize=14, fontweight="bold")
    fig.text(.54, .865, "每个窗口小格 = 1袋 / 1秒；黑点 = 开始一道菜，同时下单下一菜的材料",
             ha="center", fontsize=10.5)
    fig.legend(handles=[Patch(facecolor=WORK, edgecolor=INK, label="正在做菜"),
                        Patch(facecolor=STALL, edgecolor=INK, hatch="///", label="料未到，真正停工"),
                        Patch(facecolor=LARGE, edgecolor=INK, label="窗口给老周装袋"),
                        Patch(facecolor=SMALL, edgecolor=INK, label="窗口给小林装袋")],
               loc="lower center", bbox_to_anchor=(.54, -.015), ncol=4, frameon=False, fontsize=10)
    save(fig, stem)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = get_story_data()
    data["single_order_examples"] = [
        one_order("01_offset_not_enough", 1, 4, "已经错开 1 秒，为什么仍停工？"),
        one_order("02_offset_enough", 3, 4, "换一个开始时刻：错开 3 秒，这一次刚好赶上"),
        one_order("03_wait_is_hidden", 1, 8, "还是错开 1 秒，但手头工作更长：不必停工"),
    ]
    chronological("04_natural_staggering_works", data["success"], origin=0, end=11,
                  headline="两人都只取 1 袋：首次取料后，自然错开且能持续做菜",
                  ticks=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])
    chronological("05_staggered_but_repeats", data["failure"], origin=27, end=21,
                  headline="换成一大一小：已错开，停工仍能每 10 秒重演",
                  ticks=[0, 1, 5, 6, 7, 8, 10, 11, 15, 16, 17, 18, 20, 21])
    (OUT / "toy_story_audit.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    source = OUT / "why_staggering_sometimes_works.md"
    output = OUT / "why_staggering_sometimes_works.pdf"
    temporary = OUT / "why_staggering_sometimes_works.building.pdf"
    subprocess.run(["pandoc", str(source), "--standalone", "--number-sections",
                    "--resource-path", str(OUT), "--pdf-engine=xelatex",
                    "--output", str(temporary)], check=True, cwd=ROOT)
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
