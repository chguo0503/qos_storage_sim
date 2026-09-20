#!/usr/bin/env python3
"""Render independently averaged layer-cycle bandwidth without changing old figures.

The canonical plotting input contains six cases, each with ``segments`` rows
``[start_ms, end_ms, demand0, demand1, demand2, supply0, supply1, supply2]``.
These must be sums of separately averaged complete per-NPU cycles.  Window-edge
cycles are clipped only for display; their denominators remain complete cycle
durations.  Exact clipped integrals are audited separately.  This file does not infer SSD
service from request volumes, nor from the old aggregate 10 ms bandwidth.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
FIGURES = HERE / "figures"
POLICIES = ("asu_baseline", "od_baseline")
OUTPUTS = ("overview.png", "total_bandwidth.png",
           "asu_per_ssu_bandwidth.png", "od_per_ssu_bandwidth.png")
DEMAND_COLOR, SUPPLY_COLOR = "#D55E00", "#1768B4"

spec = importlib.util.spec_from_file_location("original_load_regime_renderer", PARENT / "render_figures.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def validate_cycle_source(service):
    """Audit clipped conservation; reconstruct complete-cycle averages for display."""
    assert service["seed"] == 7 and service["window_ms"] == [2000, 4000]
    timelines = service["per_npu_cycles"]
    assert len(timelines) == 32
    total_service = np.zeros((3, 32))
    total_demand = np.zeros(3)
    events = {}
    full_events = {}
    boundary_delta = np.zeros(6)
    clipped_count, crossing_count, cycle_count = 0, 0, 0
    for npu, cycles in enumerate(timelines):
        assert cycles and cycles[0]["clipped_start_ms"] == 2000 and cycles[-1]["clipped_end_ms"] == 4000
        previous_end = 2000
        for cycle in cycles:
            a, z = cycle["clipped_start_ms"], cycle["clipped_end_ms"]
            assert a == previous_end and z > a
            assert cycle["start_ms"] <= a < z <= cycle["end_ms"]
            previous_end = z
            seconds = (z - a) / 1000
            supplied = np.asarray(cycle["clipped_service_GiB_by_ssu"], dtype=float)
            demanded = np.asarray(cycle["clipped_demand_integral_GiB_by_ssu"], dtype=float)
            mean_supply = np.asarray(cycle["clipped_mean_supply_GiB_s_by_ssu"], dtype=float)
            mean_demand = np.asarray(cycle["clipped_mean_demand_GiB_s_by_ssu"], dtype=float)
            assert supplied.shape == demanded.shape == mean_supply.shape == mean_demand.shape == (3,)
            assert np.max(np.abs(mean_supply * seconds - supplied)) < 1e-9
            assert np.max(np.abs(mean_demand * seconds - demanded)) < 1e-9
            assert np.all(supplied >= -1e-9) and np.all(demanded >= -1e-9)
            full_seconds = (cycle["end_ms"] - cycle["start_ms"]) / 1000
            full_supplied = np.asarray(cycle["full_service_GiB_by_ssu"], dtype=float)
            full_demanded = np.asarray(cycle["full_demand_integral_GiB_by_ssu"], dtype=float)
            full_mean_supply = np.asarray(cycle["full_mean_supply_GiB_s_by_ssu"], dtype=float)
            full_mean_demand = np.asarray(cycle["full_mean_demand_GiB_s_by_ssu"], dtype=float)
            assert np.max(np.abs(full_mean_supply * full_seconds - full_supplied)) < 1e-9
            assert np.max(np.abs(full_mean_demand * full_seconds - full_demanded)) < 1e-9
            assert np.max(np.abs(full_supplied - cycle["expected_next_layer_GiB_by_ssu"])) < 1e-7
            total_service[:, npu] += supplied
            total_demand += demanded
            values = np.r_[mean_demand, mean_supply]
            events[a] = events.get(a, np.zeros(6)) + values
            events[z] = events.get(z, np.zeros(6)) - values
            full_values = np.r_[full_mean_demand, full_mean_supply]
            full_events[a] = full_events.get(a, np.zeros(6)) + full_values
            full_events[z] = full_events.get(z, np.zeros(6)) - full_values
            if cycle["boundary_clipped"]:
                boundary_delta += full_values * seconds - np.r_[demanded, supplied]
            else:
                assert np.max(np.abs(full_values - values)) < 1e-8
            clipped_count += bool(cycle["boundary_clipped"])
            crossing_count += not bool(cycle["same_request"])
            cycle_count += 1
    assert np.max(np.abs(total_service - np.asarray(service["warm_actual_service_GiB_by_ssu_npu"]))) < 1e-8
    assert np.max(np.abs(total_service.sum(axis=1) - service["warm_actual_service_GiB_by_ssu"])) < 1e-8
    assert np.max(np.abs(total_demand - service["warm_reference_demand_integral_GiB_by_ssu"])) < 1e-8
    rows = np.asarray(service["fleet_segments"], dtype=float)
    times, running = sorted(events), np.zeros(6)
    for a, z in zip(times[:-1], times[1:]):
        running += events[a]
        midpoint = (a + z) / 2
        row_index = np.searchsorted(rows[:, 0], midpoint, side="right") - 1
        assert 0 <= row_index < len(rows)
        assert rows[row_index, 0] <= a and z <= rows[row_index, 1]
        assert np.max(np.abs(running - rows[row_index, 2:8])) < 1e-7
    full_rows, full_running = [], np.zeros(6)
    for a, z in zip(times[:-1], times[1:]):
        full_running += full_events[a]
        full_rows.append([a, z, *full_running.tolist()])
    full_rows = np.asarray(full_rows)
    full_area = (full_rows[:, 1] - full_rows[:, 0]) @ full_rows[:, 2:8] / 1000
    physical_area = np.r_[total_demand, total_service.sum(axis=1)]
    assert np.max(np.abs(full_area - physical_area - boundary_delta)) < 1e-7
    audit = dict(npu_count=32, cycle_count=cycle_count, boundary_clipped_count=clipped_count,
                cross_request_count=crossing_count, independently_reconstructed_fleet_values=True,
                per_npu_service_integrals_verified=True,
                full_mean_equals_full_integral_divided_by_complete_duration=True,
                full_cycle_service_equals_next_layer_bytes=True,
                display_area_difference_only_from_boundary_cycles=True,
                displayed_minus_physical_demand_GiB_by_ssu=boundary_delta[:3].tolist(),
                displayed_minus_physical_supply_GiB_by_ssu=boundary_delta[3:].tolist(),
                physical_demand_GiB_by_ssu=total_demand.tolist(),
                physical_service_GiB_by_ssu=total_service.sum(axis=1).tolist())
    return audit, full_rows.tolist()


def load_services(paths):
    cases, cycle_checks = [], []
    for path in paths:
        data = read_json(path)
        command = read_json(path.parent / "command.json")
        assert command["status"] == "complete" and command["completed_simulation"] is True
        assert command["all_protected_files_unchanged"] is True
        assert command["services_sha256"] == sha(path)
        assert data["checks"] and all(value is True for value in data["checks"].values())
        assert command["checks"] == data["checks"]
        audit, full_segments = validate_cycle_source(data)
        regime = data["regime"]
        if regime == "random":
            regime = "under"
        case = dict(condition=regime, policy=data["policy"], seed=data["seed"], segments=full_segments,
                    clipped_segments=data["fleet_segments"], boundary_audit=audit,
                    window_statistics=data["window_statistics"])
        cases.append(case)
        cycle_checks.append(dict(source=str(path), command_sha256=sha(path.parent / "command.json"),
                                 timing_comparison=data["timing_comparison"],
                                 runner_checks=data["checks"], condition=regime, policy=data["policy"], **audit))
    return dict(config=dict(input_type="replayed exact SSD services averaged over complete per-NPU layer cycles",
                            visible_window_only_clips_display_not_average_denominator=True,
                            window_ms=[2000, 4000], bandwidth_seed=7), cases=cases), cycle_checks


def bandwidth_data(case, disk=None):
    values = np.asarray(case["segments"], dtype=float)
    demand = values[:, 2:5].sum(axis=1) if disk is None else values[:, 2 + disk]
    supply = values[:, 5:8].sum(axis=1) if disk is None else values[:, 5 + disk]
    duration = values[:, 1] - values[:, 0]
    edges = np.r_[values[:, 0], values[-1, 1]] / 1000
    return edges, demand, supply, np.dot(duration, demand) / 2000, np.dot(duration, supply) / 2000


def column_ymax(layer_cases, condition_id, *, per_disk=False):
    maxima = []
    for policy in POLICIES:
        for disk in range(3) if per_disk else [None]:
            _, demand, supply, _, _ = bandwidth_data(layer_cases[condition_id, policy], disk)
            maxima.extend((float(demand.max()), float(supply.max())))
    return max((40 if per_disk else 120) * 1.15, max(maxima) * 1.12)


def bandwidth_panel(ax, case, exact_case, disk=None, *, ymax, compact=False):
    edges, demand, supply, demand_mean, supply_mean = bandwidth_data(case, disk)
    exact = np.asarray(exact_case["demand_segments"], dtype=float)
    overload = np.any(exact[:, 2:5] > 40, axis=1) if disk is None else exact[:, 2 + disk] > 40
    for row, over in zip(exact, overload):
        if over:
            ax.axvspan(row[0] / 1000, row[1] / 1000, color="#D88080", alpha=.075, linewidth=0)
    ax.stairs(demand, edges, baseline=None, color=DEMAND_COLOR, linewidth=1.45 if compact else 1.65)
    ax.stairs(supply, edges, baseline=None, color=SUPPLY_COLOR, linewidth=1.35 if compact else 1.6)
    ax.axhline(120 if disk is None else 40, color="#4D4D4D", linestyle="--", linewidth=1)
    ax.set(xlim=(2, 4), ylim=(0, ymax), xlabel="时间（秒）")
    ax.grid(alpha=.15)
    physical_demand_mean = (exact[:, 1] - exact[:, 0]) @ exact[:, 2:5] / 2000
    physical_demand_mean = physical_demand_mean.sum() if disk is None else physical_demand_mean[disk]
    physical_supply_mean = sum(exact_case["SSD_GiB_s"]) if disk is None else exact_case["SSD_GiB_s"][disk]
    annotation = (f"物理warm均值：需求 {physical_demand_mean:.1f} / 供给 {physical_supply_mean:.1f}\n"
                  f"完整周期曲线均值：需求 {demand_mean:.1f} / 供给 {supply_mean:.1f}")
    ax.text(.03, .96, annotation,
            transform=ax.transAxes, ha="left", va="top", fontsize=8.0 if compact else 8.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=1.4))


def bandwidth_legend(fig, y, *, per_disk=False, compact=False):
    capacity = 40 if per_disk else 120
    handles = [Line2D([], [], color=DEMAND_COLOR, linewidth=2, label="各卡完整层周期平均需求之和"),
               Line2D([], [], color=SUPPLY_COLOR, linewidth=2, label="各卡完整层周期平均实际供给之和"),
               Line2D([], [], color="#4D4D4D", linestyle="--",
                      label=f"物理容量 {capacity} GiB/s（非异步均值叠加线的点值上限）"),
               Patch(facecolor="#D88080", alpha=.16,
                     label="原始逐事件需求：本盘>40时段" if per_disk else "原始逐事件需求：至少一盘>40时段")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, y), ncol=2,
               frameon=False, fontsize=9 if compact else 10, handlelength=3,
               columnspacing=2.0, labelspacing=.7)


def bandwidth_footnotes(fig, *, y=.075, fontsize=9.5):
    lines = [
        "每卡完整周期=本层开始计算→下一层开始计算（含等待、跨请求）；供给=完整周期真实 SSD 服务量/时长，需求=同期 V/C 的时间平均。",
        "窗口边缘仅裁显示，平均仍使用包含窗外时段的完整周期，因此曲线均值可与物理warm均值略有不同。32卡异步均值叠加可超容量，不代表 SSD 超速。",
        "粉色背景按原始逐盘需求判定，不能用平滑后的橙/蓝线分类。负载分类仅指 warm [2,4)；三个负载输入配比不同；带宽为 seed 7。"
    ]
    for index, line in enumerate(lines):
        fig.text(.06, y - index * .024, line, fontsize=fontsize, color="#555555")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=180, facecolor="white")
    plt.close(fig)


def overview(conditions, layer_cases):
    fig, axes = plt.subplots(4, 3, figsize=(21.5, 19),
                             gridspec_kw=dict(height_ratios=[1.08, .76, 1, 1]))
    fig.subplots_adjust(left=.065, right=.98, top=.915, bottom=.19, hspace=.55, wspace=.25)
    for column, condition in enumerate(conditions):
        ymax = column_ymax(layer_cases, condition["id"])
        axes[0, column].set_title(condition["label"], fontsize=15, fontweight="bold", pad=27, loc="center")
        original.cdf_panel(axes[0, column], condition, zoom=True, compact=True)
        original.utility_panel(axes[1, column], condition, compact=True)
        for row, policy in enumerate(POLICIES, start=2):
            bandwidth_panel(axes[row, column], layer_cases[condition["id"], policy],
                            original.case_index(condition)[policy, 7], ymax=ymax, compact=True)
    axes[0, 0].set_ylabel("归一化耗时 CDF\n累计请求比例")
    axes[1, 0].set_ylabel("NPU 平均利用率（%）")
    axes[2, 0].set_ylabel("ASU · 各卡周期均值叠加\nGiB/s")
    axes[3, 0].set_ylabel("OD · 各卡周期均值叠加\nGiB/s")
    fig.suptitle("三种负载：请求耗时、NPU 利用率与按层周期统计的带宽", fontsize=21, y=.986)
    fig.text(.5, .95, "32 NPU / 3 SSU × 40 GiB/s · Ring hash · Random · warm [2,4) 秒",
             ha="center", fontsize=12, color="#555555")
    bandwidth_legend(fig, .151, compact=True)
    fig.text(.06, .10,
             "归一化耗时=(接纳至prefill完成)/本请求8层纯计算；CDF图例为SLO×1.5达标率，不含接纳前排队；CDF/U 三种子等权，数值与原图一致。",
             fontsize=9.5, color="#555555")
    bandwidth_footnotes(fig, y=.077)
    save(fig, "overview.png")


def total_bandwidth_figure(conditions, layer_cases):
    fig, axes = plt.subplots(2, 3, figsize=(21, 11.5), sharex=True)
    fig.subplots_adjust(left=.07, right=.98, top=.75, bottom=.18, hspace=.31, wspace=.19)
    for row, policy in enumerate(POLICIES):
        for column, condition in enumerate(conditions):
            bandwidth_panel(axes[row, column], layer_cases[condition["id"], policy],
                            original.case_index(condition)[policy, 7],
                            ymax=column_ymax(layer_cases, condition["id"]))
            if row == 0:
                axes[row, column].set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=14)
        axes[row, 0].set_ylabel(f"{original.LABELS[policy]}\n各卡周期均值叠加（GiB/s）")
    fig.suptitle("整机带宽：32张卡各自层周期的平均需求与实际供给之和", fontsize=21, y=.983)
    fig.text(.5, .934, "32 NPU / 3 SSU · Ring hash · Random · seed 7 · warm [2,4) 秒 · 每列独立纵轴，同列 ASU / OD 一致",
             ha="center", fontsize=11, color="#555555")
    bandwidth_legend(fig, .898)
    bandwidth_footnotes(fig, y=.11)
    save(fig, "total_bandwidth.png")


def per_disk_figure(conditions, layer_cases, policy):
    fig, axes = plt.subplots(3, 3, figsize=(21, 14), sharex=True)
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.14, hspace=.29, wspace=.18)
    for disk in range(3):
        for column, condition in enumerate(conditions):
            ax = axes[disk, column]
            exact_case = original.case_index(condition)[policy, 7]
            bandwidth_panel(ax, layer_cases[condition["id"], policy], exact_case, disk,
                            ymax=column_ymax(layer_cases, condition["id"], per_disk=True))
            if disk == 0:
                ax.set_title(condition["label"], fontsize=14, fontweight="bold", loc="center", pad=16)
            ax.text(.03, .79, f"原始 D>40 时间：{exact_case['per_disk_overload_percent'][disk]:.2f}%",
                    transform=ax.transAxes, va="top", fontsize=8.8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=1.4))
        axes[disk, 0].set_ylabel(f"SSU {disk}\n各卡周期均值叠加（GiB/s）")
    fig.suptitle(f"{original.LABELS[policy]}：逐盘需求与实际供给，按各卡层周期平均后叠加", fontsize=21, y=.985)
    fig.text(.5, .944, "32 NPU / 3 SSU · Ring hash · Random · seed 7 · warm [2,4) 秒 · 每列独立纵轴，同列跨盘与策略一致",
             ha="center", fontsize=11, color="#555555")
    bandwidth_legend(fig, .911, per_disk=True)
    bandwidth_footnotes(fig, y=.091)
    save(fig, "asu_per_ssu_bandwidth.png" if policy == "asu_baseline" else "od_per_ssu_bandwidth.png")


def validate(parent_data, layer_data):
    """Validate clipped conservation and complete-cycle display edge differences."""
    cases = {(case["condition"], case["policy"]): case for case in layer_data["cases"]}
    assert len(cases) == len(layer_data["cases"]) == 6
    assert set(cases) == {(condition, policy) for condition in ("full", "under", "semi") for policy in POLICIES}
    checks = []
    for condition in parent_data["conditions"]:
        for policy in POLICIES:
            case = cases[condition["id"], policy]
            assert case["seed"] == 7
            rows = np.asarray(case["segments"], dtype=float)
            assert rows.ndim == 2 and rows.shape[1] == 8 and len(rows)
            assert np.all(np.isfinite(rows)) and np.all(rows[:, 2:] >= -1e-7)
            assert rows[0, 0] == 2000 and rows[-1, 1] == 4000
            assert np.all(rows[:, 1] > rows[:, 0])
            assert np.all(rows[1:, 0] == rows[:-1, 1])
            exact = original.case_index(condition)[policy, 7]
            if "window_statistics" in case:
                assert abs(case["window_statistics"]["U_percent"] - exact["U_percent"]) < 1e-7
                assert abs(case["window_statistics"]["slo"]["percent"] - exact["slo_percent"]) < 1e-7
            raw_demand = np.asarray(exact["demand_segments"], dtype=float)
            old_demand_mean = (raw_demand[:, 1] - raw_demand[:, 0]) @ raw_demand[:, 2:5] / 2000
            new_demand_mean = (rows[:, 1] - rows[:, 0]) @ rows[:, 2:5] / 2000
            new_supply_mean = (rows[:, 1] - rows[:, 0]) @ rows[:, 5:8] / 2000
            clipped_rows = np.asarray(case["clipped_segments"], dtype=float)
            clipped_demand_mean = (clipped_rows[:, 1] - clipped_rows[:, 0]) @ clipped_rows[:, 2:5] / 2000
            clipped_supply_mean = (clipped_rows[:, 1] - clipped_rows[:, 0]) @ clipped_rows[:, 5:8] / 2000
            assert np.max(np.abs(old_demand_mean - clipped_demand_mean)) < 1e-7
            assert np.max(np.abs(clipped_supply_mean - np.asarray(exact["SSD_GiB_s"]))) < 1e-7
            boundary = case["boundary_audit"]
            assert np.max(np.abs(new_demand_mean - clipped_demand_mean - np.asarray(boundary["displayed_minus_physical_demand_GiB_by_ssu"]) / 2)) < 1e-7
            assert np.max(np.abs(new_supply_mean - clipped_supply_mean - np.asarray(boundary["displayed_minus_physical_supply_GiB_by_ssu"]) / 2)) < 1e-7
            checks.append(dict(condition=condition["id"], policy=policy, seed=7,
                               segment_count=len(rows), displayed_demand_mean_GiB_s=new_demand_mean.tolist(),
                               displayed_supply_mean_GiB_s=new_supply_mean.tolist(),
                               physical_demand_mean_GiB_s=clipped_demand_mean.tolist(),
                               physical_supply_mean_GiB_s=clipped_supply_mean.tolist(),
                               supply_peak_GiB_s=rows[:, 5:8].max(axis=0).tolist(),
                               total_supply_peak_GiB_s=float(rows[:, 5:8].sum(axis=1).max()),
                               clipped_window_demand_and_service_areas_preserved=True,
                               displayed_area_difference_explained_by_complete_boundary_cycles=True))
    return cases, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data", type=Path, help="Optional already canonicalized plotting JSON")
    group.add_argument("--services", type=Path, nargs="+", help="Six exact SSD service replay JSON.gz files")
    args = parser.parse_args()
    auto_services = sorted((HERE / "runs").glob("*/services.json.gz")) if not args.data and not args.services else []
    service_paths = args.services or auto_services
    sources = [path.resolve() for path in service_paths] if service_paths else [(args.data or HERE / "layer_plot_data.json").resolve()]
    if not all(source.exists() for source in sources):
        parser.error("Layer-cycle integrals are not ready; no figures generated")
    protected = [PARENT / "plot_data.json", PARENT / "render_figures.py", *sorted((PARENT / "figures").glob("*.png"))]
    protected_sha = {str(path.relative_to(PARENT)): sha(path) for path in protected}
    source_sha = {str(path): sha(path) for path in sources}
    parent_data = read_json(PARENT / "plot_data.json")
    if service_paths:
        layer_data, cycle_checks = load_services(sources)
    else:
        layer_data, cycle_checks = read_json(sources[0]), []
    layer_cases, checks = validate(parent_data, layer_data)
    original.setup_font()
    overview(parent_data["conditions"], layer_cases)
    total_bandwidth_figure(parent_data["conditions"], layer_cases)
    for policy in POLICIES:
        per_disk_figure(parent_data["conditions"], layer_cases, policy)
    assert source_sha == {str(path): sha(path) for path in sources}
    assert protected_sha == {str(path.relative_to(PARENT)): sha(path) for path in protected}
    audit = dict(all_checks_passed=True, source_sha256=source_sha, renderer_sha256=sha(__file__),
                 original_artifacts_sha256=protected_sha, original_artifacts_unchanged=True,
                 source_config=layer_data.get("config", {}),
                 bandwidth_cycle="each NPU: compute_start(k) to compute_start(k+1), including wait and request transitions",
                 boundary_rule="main curves use full-cycle means; only the visible span is clipped; clipped true integrals audited separately",
                 aggregate="sum of independently averaged asynchronous NPU cycles; point values may exceed physical capacity",
                 overload_shading="original exact per-SSU request demand, never smoothed supply",
                 CDF_and_NPU_statistics="unchanged original plot_data, equal mean across seeds7/19/43",
                 checks=checks, cycle_checks=cycle_checks, outputs=["figures/" + name for name in OUTPUTS],
                 output_sha256={"figures/" + name: sha(FIGURES / name) for name in OUTPUTS},
                 visual_review="pending")
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(png_count=len(OUTPUTS), output_dir=str(FIGURES)), ensure_ascii=False))


if __name__ == "__main__":
    main()
