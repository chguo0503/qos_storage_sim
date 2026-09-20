#!/usr/bin/env python3
"""Draw per-NPU demand and complete-layer-cycle SSD service, PNG only.

Reads input.json prepared from frozen simulation results and exact SSD service
observations.  Does not run simulations or change existing study figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
REGIMES = ("full", "under", "semi")
REGIME_LABEL = {"full": "持续过载输入组", "under": "持续欠载输入组", "semi": "局部欠载输入组"}
POLICIES = ("od_baseline", "once")
POLICY_LABEL = {"od_baseline": "OD Baseline", "once": "Once per layer（流量分配策略）"}
POLICY_FILE = {"od_baseline": "od", "once": "once"}
DEMAND_COLOR = "#D55E00"
SUPPLY_COLOR = "#0072B2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_font():
    path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=path).get_name(),
                         "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.spines.bottom": False})


def validate(data):
    cases = {(case["regime"], case["policy"]): case for case in data["cases"]}
    assert len(cases) == len(data["cases"]) == 6
    assert set(cases) == {(regime, policy) for regime in REGIMES for policy in POLICIES}
    checks = []
    for key, case in sorted(cases.items()):
        assert case["seed"] == 7 and case["window_ms"] == [2000, 4000]
        cards = case["per_npu"]
        assert len(cards) == 32 and {card["npu_id"] for card in cards} == set(range(32))
        assert abs(math.fsum(card["U_percent"] for card in cards) / 32 - case["fleet_U_percent"]) < 1e-7
        for card in cards:
            assert 0 <= card["U_percent"] <= 100 + 1e-7
            assert card["mean_demand_GiB_s"] >= 0 and card["mean_supply_GiB_s"] >= 0
            averages, maxima = {}, {}
            for kind in ("demand", "supply"):
                rows = np.asarray(card[kind + "_segments"], dtype=float)
                assert rows.ndim == 2 and rows.shape[1] == 3 and len(rows)
                assert np.all(np.isfinite(rows)) and np.all(rows[:, 2] >= -1e-9)
                assert rows[0, 0] == 2000 and rows[-1, 1] == 4000
                assert np.all(rows[:, 1] > rows[:, 0]) and np.all(rows[1:, 0] == rows[:-1, 1])
                averages[kind] = float(np.dot(rows[:, 1] - rows[:, 0], rows[:, 2]) / 2000)
                maxima[kind] = float(rows[:, 2].max())
            assert abs(averages["demand"] - card["mean_demand_GiB_s"]) < 1e-7
            checks.append(dict(regime=key[0], policy=key[1], npu_id=card["npu_id"],
                               U_percent=card["U_percent"],
                               physical_mean_demand_GiB_s=card["mean_demand_GiB_s"],
                               physical_mean_supply_GiB_s=card["mean_supply_GiB_s"],
                               projected_supply_curve_mean_GiB_s=averages["supply"],
                               supply_edge_difference_GiB_s=averages["supply"] - card["mean_supply_GiB_s"],
                               demand_max_GiB_s=maxima["demand"], supply_max_GiB_s=maxima["supply"],
                               demand_area_verified=True))
    return cases, checks


def nice_upper(value):
    target = max(value * 1.075, .1)
    scale = 10 ** math.floor(math.log10(target))
    for multiplier in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if multiplier * scale >= target:
            return multiplier * scale
    raise AssertionError("unreachable")


def regime_ymax(cases, regime):
    return nice_upper(max(row[2] for policy in POLICIES for card in cases[regime, policy]["per_npu"]
                          for kind in ("demand_segments", "supply_segments") for row in card[kind]))


def draw_card(ax, card, ymax, *, last=False, comparison=False):
    for start, end in card.get("stall_segments", []):
        assert end >= start
        if end > 2000 and start < 4000:
            ax.axvspan(max(start, 2000) / 1000, min(end, 4000) / 1000,
                       color="#858B92", alpha=.18, linewidth=0, zorder=0)
    for kind, color, width in (("demand", DEMAND_COLOR, 2.6), ("supply", SUPPLY_COLOR, 1.25)):
        rows = np.asarray(card[kind + "_segments"], dtype=float)
        edges = np.r_[rows[:, 0], rows[-1, 1]] / 1000
        ax.stairs(rows[:, 2], edges, baseline=None, color=color, linewidth=width,
                  linestyle="--" if kind == "demand" else "-", zorder=3 if kind == "demand" else 4)
    ax.axhline(0, color="#ADB3BA", linewidth=.65, zorder=1)
    ax.set(xlim=(2, 4), ylim=(-ymax * .055, ymax), yticks=[0, ymax])
    ax.tick_params(axis="y", labelsize=6.7, pad=2, length=0)
    ax.tick_params(axis="x", labelsize=8.6, length=0, pad=2, labelbottom=last)
    ax.set_xticks(np.arange(2, 4.001, .25))
    ax.grid(axis="x", color="#BBBBBB", alpha=.26, linewidth=.5)
    if card["npu_id"] % 2:
        ax.set_facecolor("#FAFBFC")
    ax.text(-.038, .47, f"NPU {card['npu_id']:02d}\nU={card['U_percent']:.2f}%",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.4 if comparison else 9.2,
            linespacing=1.2, color="#333333")
    ax.text(1.012, .47, f"需求 {card['mean_demand_GiB_s']:.3f}\n供给 {card['mean_supply_GiB_s']:.3f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=8.1 if comparison else 9,
            linespacing=1.25, color="#333333")
    if last:
        ax.set_xlabel("时间（秒）", fontsize=11, labelpad=5)


def legend(fig, y):
    handles = [Line2D([], [], color=DEMAND_COLOR, linewidth=2.6, linestyle="--", label="需求 B（当前请求的参考需求）"),
               Line2D([], [], color=SUPPLY_COLOR, linewidth=1.6, label="供给 b（完整层周期平均实际供给）"),
               Patch(facecolor="#858B92", alpha=.18, label="IO 等待")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y),
               ncol=3, frameon=False, fontsize=10.5, handlelength=3, columnspacing=2)


def footer(fig, *, comparison=False):
    size = 9 if comparison else 9.1
    fig.text(.065, .027,
             "每行一张卡。橙线随当前请求切换；蓝线按本层开始计算→下一层开始计算（含等待）取完整周期均值，非瞬时速率；灰色为 IO 等待。",
             fontsize=size, color="#555555")
    fig.text(.065, .014,
             "跨请求周期中，蓝线读取下一请求，不能直接与橙线作比。窗边只裁显示，平均仍包含窗外完整周期；右侧是真实 warm 均值，并非蓝线可见面积。",
             fontsize=size, color="#555555")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)
    return name


def single_figure(case, ymax):
    fig, axes = plt.subplots(32, 1, figsize=(18.5, 28), sharex=True)
    fig.subplots_adjust(left=.13, right=.865, top=.922, bottom=.057, hspace=.31)
    for ax, card in zip(axes, sorted(case["per_npu"], key=lambda c: c["npu_id"])):
        draw_card(ax, card, ymax, last=card["npu_id"] == 31)
    axes[0].text(1.012, 1.45, "真实warm均值\nGiB/s", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=9.5, fontweight="bold")
    fig.suptitle(f"{REGIME_LABEL[case['regime']]} · {POLICY_LABEL[case['policy']]}：32 张 NPU 的需求与实际供给",
                 fontsize=20, y=.989)
    fig.text(.5, .970,
             f"32 NPU / 3 SSU × 40 GiB/s · Ring hash · Random · seed 7 · warm [2,4) 秒 · 整机 U={case['fleet_U_percent']:.2f}%",
             ha="center", fontsize=11.2, color="#555555")
    legend(fig, .958)
    fig.text(.13, .938, f"纵轴单位 GiB/s；全部32卡与本组另一策略共用纵轴 0～{ymax:g}；输入组名沿用 OD 的逐盘负载分类。",
             fontsize=10, color="#555555")
    footer(fig)
    return save(fig, f"{case['regime']}_{POLICY_FILE[case['policy']]}_32npu_bandwidth.png")


def comparison_figure(cases, regime, ymax):
    fig, axes = plt.subplots(32, 2, figsize=(27.5, 28), sharex=True)
    fig.subplots_adjust(left=.073, right=.931, top=.904, bottom=.057, hspace=.31, wspace=.43)
    for column, policy in enumerate(POLICIES):
        case = cases[regime, policy]
        for ax, card in zip(axes[:, column], sorted(case["per_npu"], key=lambda c: c["npu_id"])):
            draw_card(ax, card, ymax, last=card["npu_id"] == 31, comparison=True)
        axes[0, column].set_title(f"{POLICY_LABEL[policy]} · 整机 U={case['fleet_U_percent']:.2f}%",
                                  fontsize=14, loc="center", pad=23, fontweight="bold")
        axes[0, column].text(1.012, 1.40, "真实warm均值\nGiB/s", transform=axes[0, column].transAxes,
                             ha="left", va="bottom", fontsize=9, fontweight="bold")
    fig.suptitle(f"{REGIME_LABEL[regime]}：OD 与 Once 的逐卡带宽对照", fontsize=22, y=.989)
    fig.text(.5, .970, "32 NPU / 3 SSU × 40 GiB/s · Ring hash · Random · seed 7 · warm [2,4) 秒",
             ha="center", fontsize=12, color="#555555")
    legend(fig, .958)
    fig.text(.073, .934, f"对应行是同一张 NPU；两列共用纵轴 0～{ymax:g} GiB/s。输入组名沿用 OD 的负载分类；策略改变执行进度，同一时刻的在运行请求可能不同。",
             fontsize=10.2, color="#555555")
    footer(fig, comparison=True)
    return save(fig, f"{regime}_od_vs_once_32npu_bandwidth.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "input.json")
    args = parser.parse_args()
    source = args.input.resolve()
    source_sha = sha(source)
    old_figures = [p for p in HERE.parent.rglob("*.png") if HERE not in p.parents]
    protected_sha = {str(p): sha(p) for p in old_figures}
    data = json.loads(source.read_text())
    cases, checks = validate(data)
    setup_font()
    outputs, limits = [], {}
    for regime in REGIMES:
        limits[regime] = regime_ymax(cases, regime)
        for policy in POLICIES:
            outputs.append(single_figure(cases[regime, policy], limits[regime]))
        outputs.append(comparison_figure(cases, regime, limits[regime]))
    assert sha(source) == source_sha
    assert protected_sha == {str(p): sha(p) for p in old_figures}
    audit = dict(all_checks_passed=True, source_sha256=source_sha, renderer_sha256=sha(__file__),
                 source_config=data.get("config", {}), seed=7, window_ms=[2000, 4000],
                 case_count=6, per_npu_checked_count=len(checks),
                 demand_definition="current request per-layer read volume / own per-layer compute; held through compute and stall",
                 supply_definition="real SSD service / complete compute_start(k) to compute_start(k+1) interval; display only clipped",
                 right_hand_means="true physical warm-window means from input, never projected supply-curve area",
                 IO_stall_background="exact supplied stall intervals clipped for display; grey background",
                 curve_style="orange dashed demand, blue solid supply with uniform linewidth",
                 regime_labels="input group labels follow OD per-disk demand classification; not a claim of identical Once instantaneous load",
                 cross_request_warning="supply can serve the next request while demand describes the current request; direct b/B comparison invalid",
                 shared_ymax_GiB_s=limits, all_peaks_visible=True,
                 no_simulation_run_by_renderer=True, old_figures_unchanged=True,
                 old_figure_sha256=protected_sha, checks=checks,
                 outputs=["figures/" + name for name in outputs],
                 output_sha256={"figures/" + name: sha(FIGURES / name) for name in outputs},
                 visual_review="pending")
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(png_count=len(outputs), figures_dir=str(FIGURES)), ensure_ascii=False))


if __name__ == "__main__":
    main()
