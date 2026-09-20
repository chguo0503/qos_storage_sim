#!/usr/bin/env python3
"""Plot three existing workloads from the frozen comparison data, PNG only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SEEDS = (7, 19, 43)
POLICIES = ("asu_baseline", "od_baseline")
LABELS = {"asu_baseline": "ASU Baseline", "od_baseline": "OD Baseline"}
COLORS = {"asu_baseline": "#454545", "od_baseline": "#009E73"}
DEMAND_COLOR, SUPPLY_COLOR = "#D55E00", "#1768B4"
OUTPUTS = ("overview.png", "ttft_normalized_cdf.png", "ttft_normalized_cdf_zoom.png",
           "npu_utilization.png", "total_bandwidth.png", "asu_per_ssu_bandwidth.png", "od_per_ssu_bandwidth.png")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_font():
    path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=path).get_name(),
                         "font.size": 10.5, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left"})


def case_index(condition):
    return {(case["policy"], case["seed"]): case for case in condition["cases"]}


def cdf(condition, policy, xx):
    ix = case_index(condition)
    return np.mean([np.searchsorted(np.asarray(ix[policy, seed]["ratios"]), xx, side="right") /
                    len(ix[policy, seed]["ratios"]) for seed in SEEDS], axis=0)


def mean_slo(condition, policy):
    ix = case_index(condition)
    return math.fsum(ix[policy, seed]["slo_percent"] for seed in SEEDS) / len(SEEDS)


def cdf_panel(ax, condition, zoom=False, compact=False):
    ix = case_index(condition)
    xmax = 3.0 if zoom else max(1.8, math.ceil((max(case["ratios"][-1] for case in condition["cases"]) + .05) * 2) / 2)
    for policy in POLICIES:
        knots = np.unique(np.concatenate([ix[policy, seed]["ratios"] for seed in SEEDS]))
        xx = np.r_[.9, knots[(knots > .9) & (knots < xmax)], xmax]
        yy = cdf(condition, policy, xx)
        assert yy[0] == 0 and np.all(np.diff(yy) >= -1e-12)
        if not zoom:
            assert yy[-1] == 1
        # A wider dashed line remains visible beneath an exactly coincident line.
        ax.step(xx, yy, where="post", color=COLORS[policy],
                linestyle="--" if policy == "asu_baseline" else "-",
                linewidth=2.9 if policy == "asu_baseline" else 1.8,
                label=f"{LABELS[policy]} · {mean_slo(condition, policy):.2f}%")
        ax.plot(1.5, cdf(condition, policy, [1.5])[0],
                marker="s" if policy == "asu_baseline" else "o", color=COLORS[policy],
                markersize=6 if policy == "asu_baseline" else 4,
                markeredgecolor="white", markeredgewidth=.6, zorder=5)
    ax.axvline(1.5, color="#777777", linestyle=":", linewidth=1)
    ax.text(1.5, 1.044, "1.5×", ha="center", va="bottom", fontsize=9, color="#555555")
    ax.set(xlim=(.9, xmax), ylim=(0, 1.035), xlabel="归一化耗时（倍）")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower right", fontsize=8.7 if compact else 9.5,
              framealpha=.95, edgecolor="#DDDDDD", handlelength=3)
    if condition["id"] == "under":
        ax.text(.50, .62, "两条 CDF 完全重合", transform=ax.transAxes,
                ha="center", fontsize=9 if compact else 10, color="#555555")


def utility_panel(ax, condition, compact=False):
    ix = case_index(condition)
    for j, policy in enumerate(POLICIES):
        values = np.asarray([ix[policy, seed]["U_percent"] for seed in SEEDS])
        average = float(values.mean())
        ax.bar(j, average, color=COLORS[policy], alpha=.22, edgecolor=COLORS[policy], width=.56)
        ax.errorbar(j, average, yerr=[[average - float(values.min())], [float(values.max()) - average]],
                    color=COLORS[policy], linewidth=1.5, capsize=5, fmt="none", zorder=4)
        for value, xshift, marker in zip(values, (-.11, 0, .11), ("o", "s", "^")):
            ax.scatter(j + xshift, value, color=COLORS[policy], marker=marker,
                       s=28 if compact else 38, edgecolors="white", linewidths=.5, zorder=5)
        ax.text(j, max(float(values.max()), average) + 3.0, f"{average:.2f}%", ha="center",
                fontsize=10 if compact else 12, fontweight="bold", color=COLORS[policy])
    ax.set_xticks([0, 1], ["ASU", "OD"])
    ax.set(xlim=(-.65, 1.65), ylim=(0, 109))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=.18)


def bandwidth_data(case, disk=None):
    segments = np.asarray(case["demand_segments"], dtype=float)
    demand = segments[:, 2:].sum(axis=1) if disk is None else segments[:, disk + 2]
    supply = np.asarray(case["supply_10ms"], dtype=float)
    actual = supply.sum(axis=0) if disk is None else supply[disk]
    edges = np.r_[segments[:, 0], segments[-1, 1]] / 1000
    overload = np.any(segments[:, 2:] > 40, axis=1) if disk is None else segments[:, disk + 2] > 40
    demand_mean = float(np.dot(demand, segments[:, 1] - segments[:, 0]) / 2000)
    return segments, edges, demand, actual, overload, demand_mean


def bandwidth_panel(ax, case, disk=None, *, ymax=None, compact=False):
    segments, edges, demand, actual, overload, demand_mean = bandwidth_data(case, disk)
    for (a, z, *_), over in zip(segments, overload):
        if over:
            ax.axvspan(a / 1000, z / 1000, color="#D88080", alpha=.075, linewidth=0)
    ax.stairs(demand, edges, baseline=None, color=DEMAND_COLOR, linewidth=1.35 if compact else 1.6)
    ax.stairs(actual, np.linspace(2, 4, 201), baseline=None, color=SUPPLY_COLOR, linewidth=1.25 if compact else 1.5)
    capacity = 120 if disk is None else 40
    ax.axhline(capacity, color="#4D4D4D", linestyle="--", linewidth=1)
    if ymax is None:
        ymax = max(capacity * 1.15, float(demand.max()) * 1.08)
    ax.set(xlim=(2, 4), ylim=(0, ymax), xlabel="时间（秒）")
    ax.grid(alpha=.15)
    ax.text(.03, .94, f"窗口均值：需求 {demand_mean:.1f} / 供给 {actual.mean():.1f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5 if compact else 9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1.4))


def bandwidth_legend(fig, y, *, per_disk=False):
    capacity = 40 if per_disk else 120
    handles = [Line2D([], [], color=DEMAND_COLOR, linewidth=2, label="当前请求参考需求 Σ(V/C)"),
               Line2D([], [], color=SUPPLY_COLOR, linewidth=2, label="SSD 实际供给（10ms 平均）"),
               Line2D([], [], color="#4D4D4D", linestyle="--", label=f"容量 {capacity} GiB/s"),
               Patch(facecolor="#D88080", alpha=.16, label="本盘 D>40 时段" if per_disk else "至少一盘 D>40 时段")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, y), ncol=4,
               frameon=False, fontsize=9.5, handlelength=3, columnspacing=2.3)


def footnote(fig, *, cdf_only=False, y=.025):
    text = ("CDF：(接纳至 prefill 完成) / 自身8层纯计算；不含接纳前排队，不是真实首token。" if cdf_only else
            "CDF / NPU 利用率：seed 7、19、43 等权；带宽：seed 7，需求逐事件、实际供给10ms平均。")
    fig.text(.055, y + .023, text, fontsize=9.3, color="#555555")
    fig.text(.055, y, "负载分类仅指 warm [2,4) 秒，三组输入配比不同；按逐盘需求分类，不以总需求是否超过120代替。",
             fontsize=9.3, color="#555555")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, facecolor="white")
    plt.close(fig)


def overview(conditions):
    fig, axes = plt.subplots(4, 3, figsize=(20.5, 17.5),
                             gridspec_kw=dict(height_ratios=[1.08, .76, 1, 1]))
    fig.subplots_adjust(left=.065, right=.98, top=.915, bottom=.14, hspace=.55, wspace=.25)
    for column, condition in enumerate(conditions):
        ymax = max(140, max(float(bandwidth_data(case_index(condition)[p, 7])[2].max()) for p in POLICIES) * 1.12)
        axes[0, column].set_title(condition["label"], fontsize=15, fontweight="bold", pad=27, loc="center")
        cdf_panel(axes[0, column], condition, zoom=True, compact=True)
        utility_panel(axes[1, column], condition, compact=True)
        for row, policy in enumerate(POLICIES, start=2):
            bandwidth_panel(axes[row, column], case_index(condition)[policy, 7], ymax=ymax, compact=True)
    axes[0, 0].set_ylabel("归一化耗时 CDF\n累计请求比例")
    axes[1, 0].set_ylabel("NPU 平均利用率（%）")
    axes[2, 0].set_ylabel("ASU · 整机总带宽\nGiB/s")
    axes[3, 0].set_ylabel("OD · 整机总带宽\nGiB/s")
    fig.suptitle("三种负载：ASU 与 OD 的请求耗时、NPU 利用率和存储带宽", fontsize=21, y=.986)
    fig.text(.5, .95, "32 NPU / 3 SSU × 40 GiB/s · Ring hash · Random · warm [2,4) 秒",
             ha="center", fontsize=12, color="#555555")
    bandwidth_legend(fig, .095)
    fig.text(.065, .055, "归一化耗时=(接纳至prefill完成)/本请求8层纯计算；CDF图例百分比=SLO×1.5达标率，不含接纳前排队。",
             fontsize=9.5, color="#555555")
    fig.text(.065, .035, "CDF仅放大0.9～3倍；完整长尾见单独CDF图。U柱为种子均值，点为各seed，须线为最小～最大。",
             fontsize=9.5, color="#555555")
    fig.text(.065, .015, "分类仅指[2,4)窗、配比不同。带宽每列独立纵轴，同列两策略一致；CDF/利用率三种子等权，带宽seed7。",
             fontsize=9.5, color="#555555")
    save(fig, "overview.png")


def cdf_figure(conditions, zoom):
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.2))
    fig.subplots_adjust(left=.065, right=.98, top=.76, bottom=.22, wspace=.25)
    for ax, condition in zip(axes, conditions):
        cdf_panel(ax, condition, zoom)
        ax.set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=25)
    axes[0].set_ylabel("累计请求比例")
    fig.suptitle("ASU / OD 归一化耗时 CDF" + ("：统一放大0.9～3倍" if zoom else "：保留全部长尾"),
                 fontsize=21, y=.975)
    fig.text(.5, .905, "32 NPU / 3 SSU · Random · warm [2,4) 秒 · seed 7、19、43 等权 · 图例百分比 = SLO×1.5 达标率",
             ha="center", fontsize=11.2, color="#555555")
    footnote(fig, cdf_only=True, y=.045)
    if not zoom:
        fig.text(.055, .12, "三个面板分别采用适合其完整长尾的横轴范围；纵轴口径一致。", fontsize=9.3, color="#555555")
    save(fig, "ttft_normalized_cdf_zoom.png" if zoom else "ttft_normalized_cdf.png")


def utilization_figure(conditions):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.8), sharey=True)
    fig.subplots_adjust(left=.075, right=.98, top=.77, bottom=.22, wspace=.25)
    for ax, condition in zip(axes, conditions):
        utility_panel(ax, condition)
        ax.set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=15)
    axes[0].set_ylabel("NPU 平均利用率（%）")
    markers = [Line2D([], [], linestyle="none", marker=m, color="#555555", label=f"seed {s}")
               for s, m in zip(SEEDS, ("o", "s", "^"))]
    fig.legend(handles=markers, loc="upper center", bbox_to_anchor=(.54, .89), ncol=3, frameon=False)
    fig.suptitle("NPU 平均利用率：种子均值与实际离散程度", fontsize=20, y=.976)
    fig.text(.075, .14, "柱高为三个种子的等权均值；点是各次实测；须线是最小～最大范围，并非置信区间。", fontsize=10)
    fig.text(.075, .075, "32 NPU / 3 SSU × 40 GiB/s · Ring hash · Random；分类仅指 warm [2,4) 秒，三个负载输入配比不同。",
             fontsize=10, color="#555555")
    save(fig, "npu_utilization.png")


def total_bandwidth_figure(conditions):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5), sharex=True)
    fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.12, hspace=.30, wspace=.19)
    for row, policy in enumerate(POLICIES):
        for column, condition in enumerate(conditions):
            ymax = max(140, max(float(bandwidth_data(case_index(condition)[p, 7])[2].max()) for p in POLICIES) * 1.12)
            bandwidth_panel(axes[row, column], case_index(condition)[policy, 7], ymax=ymax)
            if row == 0:
                axes[row, column].set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=14)
        axes[row, 0].set_ylabel(f"{LABELS[policy]}\n整机总带宽（GiB/s）")
    fig.suptitle("整机总带宽：参考需求与 SSD 实际供给", fontsize=21, y=.983)
    fig.text(.5, .932, "32 NPU / 3 SSU · Ring hash · Random · seed 7 · warm [2,4) 秒 · 每列独立纵轴，同列 ASU / OD 一致",
             ha="center", fontsize=11, color="#555555")
    bandwidth_legend(fig, .899)
    footnote(fig, y=.025)
    save(fig, "total_bandwidth.png")


def per_disk_figure(conditions, policy):
    fig, axes = plt.subplots(3, 3, figsize=(20, 12.6), sharex=True)
    fig.subplots_adjust(left=.075, right=.98, top=.82, bottom=.10, hspace=.29, wspace=.18)
    for disk in range(3):
        for column, condition in enumerate(conditions):
            ymax = max(48, max(float(bandwidth_data(case_index(condition)[p, 7], d)[2].max())
                              for p in POLICIES for d in range(3)) * 1.12)
            ax = axes[disk, column]
            case = case_index(condition)[policy, 7]
            bandwidth_panel(ax, case, disk, ymax=ymax)
            if disk == 0:
                ax.set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=16)
            ax.text(.03, .83, f"D>40 时间：{case['per_disk_overload_percent'][disk]:.2f}%",
                    transform=ax.transAxes, va="top", fontsize=8.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.78, pad=1.4))
        axes[disk, 0].set_ylabel(f"SSU {disk} 带宽（GiB/s）")
    fig.suptitle(f"{LABELS[policy]}：逐盘带宽需求与实际供给", fontsize=21, y=.984)
    fig.text(.5, .939, "32 NPU / 3 SSU · Ring hash · Random · seed 7 · warm [2,4) 秒 · 每列独立纵轴，同列跨盘与策略一致",
             ha="center", fontsize=11, color="#555555")
    bandwidth_legend(fig, .907, per_disk=True)
    fig.text(.055, .040, "需求按真实落盘量求和，实际供给包含跨请求预取；分类仅指[2,4)窗，三个负载输入配比不同。",
             fontsize=9.5, color="#555555")
    save(fig, "asu_per_ssu_bandwidth.png" if policy == "asu_baseline" else "od_per_ssu_bandwidth.png")


def main():
    source = HERE / "plot_data.json"
    initial_sha = sha(source)
    data = json.loads(source.read_text())
    conditions = data["conditions"]
    assert [condition["id"] for condition in conditions] == ["full", "under", "semi"]
    checks = []
    for condition in conditions:
        ix = case_index(condition)
        assert len(condition["cases"]) == len(ix) == 6
        assert set(ix) == {(policy, seed) for policy in POLICIES for seed in SEEDS}
        for (policy, seed), case in ix.items():
            ratios = np.asarray(case["ratios"], dtype=float)
            assert len(ratios) and np.all(np.isfinite(ratios)) and ratios.min() > .9
            assert np.all(np.diff(ratios) >= 0)
            observed = 100 * np.count_nonzero(ratios <= 1.5) / len(ratios)
            assert abs(observed - case["slo_percent"]) < 1e-8
            supply = np.asarray(case["supply_10ms"], dtype=float)
            assert supply.shape == (3, 200)
            assert np.all(supply >= -1e-8) and np.all(supply <= 40 + 1e-7)
            assert np.max(np.abs(supply.mean(axis=1) - case["SSD_GiB_s"])) < 1e-7
            segments = np.asarray(case["demand_segments"], dtype=float)
            assert segments.shape[1] == 5 and segments[0, 0] == 2000 and segments[-1, 1] == 4000
            assert np.all(segments[:, 1] >= segments[:, 0]) and np.all(segments[1:, 0] == segments[:-1, 1])
            assert np.all(segments[:, 2:] >= -1e-7)
            checks.append(dict(condition=condition["id"], policy=policy, seed=seed,
                               sample_count=len(ratios), cdf_1p5_percent=observed,
                               U_percent=case["U_percent"], bandwidth_integral_verified=True))
        for policy in POLICIES:
            assert abs(float(cdf(condition, policy, [1.5])[0]) * 100 - mean_slo(condition, policy)) < 1e-8
    under = next(condition for condition in conditions if condition["id"] == "under")
    under_knots = np.unique(np.concatenate([case["ratios"] for case in under["cases"]]))
    # The display note is permitted only when the supplied ECDFs truly agree.
    assert np.max(np.abs(cdf(under, POLICIES[0], under_knots) - cdf(under, POLICIES[1], under_knots))) < 1e-10
    setup_font()
    overview(conditions)
    cdf_figure(conditions, False)
    cdf_figure(conditions, True)
    utilization_figure(conditions)
    total_bandwidth_figure(conditions)
    for policy in POLICIES:
        per_disk_figure(conditions, policy)
    assert sha(source) == initial_sha, "Plot input changed during rendering"
    audit = dict(all_checks_passed=True, source_plot_data_sha256=initial_sha,
                 input_config=data.get("config", {}),
                 renderer_sha256=sha(Path(__file__)), source_unchanged=True,
                 policies=list(POLICIES), condition_order=[condition["id"] for condition in conditions],
                 CDF_aggregation="equal mean of seed7/19/43 ECDFs, no request pooling",
                 NPU_aggregation="equal seed mean, min/max range and original seed points",
                 bandwidth_seed=7, bandwidth_time_average_ms=10,
                 underload_CDFs_verified_identical=True, full_tails_preserved=True,
                 zoom_range=[.9, 3], zoom_keeps_original_denominator=True,
                 old_experiment_files_not_modified=True, no_new_simulation=True,
                 bandwidth_column_axes="per-condition range; same range across ASU/OD and all disks within a condition",
                 load_regime_classification_window_ms=[2000, 4000],
                 checks=checks, outputs=["figures/" + name for name in OUTPUTS],
                 output_sha256={"figures/" + name: sha(FIGURES / name) for name in OUTPUTS}, visual_review="pending")
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(png_count=len(OUTPUTS), output_dir=str(HERE), source_sha256=initial_sha), ensure_ascii=False))


if __name__ == "__main__":
    main()
