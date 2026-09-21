#!/usr/bin/env python3
"""Audit completed runs, export equal-seed tables and PNG-only CDFs.

    No simulation is invoked. Incomplete inputs are labelled pending; use
    --require-complete for the final report (54 main runs plus optional control).
    --output-dir supports a
temporary preview without touching the final README or figures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL_GROUPS = ("XY12_32", "XY12_24", "XY12_20", "XY12_16", "X16")
CANDIDATE = "R32_A200_B10"
CONTROL = "XY12_16_S4_control"
MAIN_GROUPS = ORIGINAL_GROUPS + (CANDIDATE,)
GROUPS = MAIN_GROUPS + (CONTROL,)
SEEDS = (7, 19, 43)
POLICIES = ("asu_baseline", "od_baseline", "once")
LABELS = {"asu_baseline": "ASU", "od_baseline": "OD", "once": "Once（流量分配）"}
COLORS = {"asu_baseline": "#4b5563", "od_baseline": "#d45e00", "once": "#2474b7"}
STYLES = {"asu_baseline": "--", "od_baseline": "-", "once": "-."}
EPS = 1e-9


def group_key(config):
    return CONTROL if config["id"] == "XY12_16" and config["ssu"] == 4 else config["id"]


def read(path):
    path = Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=1e-7):
    assert math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance), (a, b)


def overlap(a, b, start, end):
    return max(0.0, min(b, end) - max(a, start))


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_ratio(elapsed, pure):
    if abs(elapsed - pure) <= EPS:
        return 1.0
    if 1.5 * pure < elapsed <= 1.5 * pure + EPS:
        return 1.5
    return elapsed / pure


def demand_check(events, stats, start, end, disks):
    peaks = np.zeros(disks)
    integral = np.zeros(disks)
    over = np.zeros(disks)
    any_over = 0.0
    covered = 0.0
    previous = None
    for event in events:
        a, b = event["start_ms"], event["end_ms"]
        assert b >= a
        if previous is not None:
            close(a, previous)
        previous = b
        dt = overlap(a, b, start, end)
        if not dt:
            continue
        rates = np.array(event["demand_decimal_gb_s_by_ssu"])
        assert len(rates) == disks and float(rates.min()) >= -1e-7
        peaks = np.maximum(peaks, rates)
        integral += rates * dt
        flags = rates > 40.0 + EPS
        over += flags * dt
        any_over += bool(flags.any()) * dt
        covered += dt
    close(covered, end - start)
    for disk in range(disks):
        close(peaks[disk], stats["peak_gb_s_by_ssu"][disk])
        close(integral[disk] / covered, stats["mean_gb_s_by_ssu"][disk])
        close(over[disk], stats["overload_ms_by_ssu"][disk])
    close(any_over, stats["any_ssu_overload_ms"])
    assert bool((peaks < 40.0 - EPS).all()) == stats["strictly_under_capacity"]


def verify_run(config_path, config, policy, source_cache):
    directory = HERE / "runs" / config["name"] / policy
    names = ("metrics.json", "native_summary.json.gz", "cdf_samples.json.gz",
             "nominal_demand_events.json.gz", "manifest.json.gz", "config.json", "command.json")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        return None, {"config": config["name"], "strategy": policy,
                      "status": "pending", "missing": missing}
    metrics = read(directory / "metrics.json")
    assert metrics["name"] == config["name"] and metrics["strategy"] == policy
    assert metrics["seed"] == config["seed"] and metrics["npu_count"] == 32
    assert metrics["ssu_count"] == config["ssu"] and metrics["n_layers"] == 8
    assert metrics["source_unchanged"] and all(metrics["invariants"].values())
    assert metrics["config_sha256"] == sha(config_path)
    normalized_config = dict(config)
    for k, v in dict(num_npu=32, mode="random", window_ms=[2000, 4000], disk_gb_s=40.0, npu_gb_s=50.0).items():
        normalized_config.setdefault(k, v)
    assert read(directory / "config.json") == normalized_config
    assert metrics["manifest_sha256"] == sha(directory / "manifest.json.gz")
    for path, expected in metrics["source_sha256"].items():
        if path not in source_cache:
            source_cache[path] = sha(REPO / path)
        assert source_cache[path] == expected, f"Source changed: {path}"
    manifest = read(directory / "manifest.json.gz")
    assert manifest["input_fingerprint"] == metrics["input_fingerprint"]
    metadata = {r["request_id"]: r for r in manifest["requests"]}
    native = read(directory / "native_summary.json.gz")
    samples = read(directory / "cdf_samples.json.gz")
    raw = {r["request_id"]: r for r in native["request_metrics"]}
    assert len(raw) == len(samples) == len(metadata) == metrics["n_requests"]
    assert set(raw) == set(metadata) == {r["request_id"] for r in samples}
    assert native["request_count"] == len(raw)
    assert all(native["invariants"].values())
    close(native["makespan_ms"], metrics["makespan_ms"])
    close(native["completed_read_gb"], metrics["completed_read_gib"])
    assert native["completed_blocks"] == native["submitted_blocks"] == metrics["completed_blocks"]
    for sample in samples:
        row = raw[sample["request_id"]]
        inp = metadata[sample["request_id"]]
        assert sample["npu_id"] == row["npu_id"] == inp["npu_id"]
        assert sample["group"] == inp["load"]["profile_group"]
        close(row["own_compute_ms"], 8 * inp["load"]["per_layer_us"] / 1000)
        close(sample["admission_ms"], row["admission_time_ms"])
        close(sample["completion_ms"], row["completion_time_ms"])
        elapsed = row["completion_time_ms"] - row["admission_time_ms"]
        close(sample["compute_ms"], row["own_compute_ms"])
        close(sample["ttft_ms"], elapsed)
        close(sample["ttft_ratio"], elapsed / row["own_compute_ms"])
        sample["plotted_ratio"] = normalized_ratio(elapsed, row["own_compute_ms"])
    for window in ("warm", "full"):
        stat = metrics[window]
        a, b = stat["start_ms"], stat["end_ms"]
        if window == "warm":
            assert [a, b] == [2000, 4000]
        compute = np.zeros(32)
        active = np.zeros(32)
        group_compute = {g: np.zeros(32) for g in ("A", "B")}
        for batch in native["microbatch_metrics"]:
            assert len(batch["member_request_ids"]) == 1 and len(batch["layer_metrics"]) == 8
            rid = batch["member_request_ids"][0]
            npu = batch["npu_id"]
            group = metadata[rid]["load"]["profile_group"]
            active[npu] += overlap(batch["admission_time_ms"], batch["completion_time_ms"], a, b)
            for layer in batch["layer_metrics"]:
                dt = overlap(layer["compute_start_ms"], layer["compute_end_ms"], a, b)
                compute[npu] += dt
                group_compute[group][npu] += dt
        close(float(compute.sum()) / (32 * (b - a)), stat["fleet_utilization"])
        assert bool((np.abs(active - (b - a)) < 1e-6).all()) == stat["all_npus_active_whole_window"]
        assert bool(((group_compute["A"] > EPS) & (group_compute["B"] > EPS)).all()) == stat["all_npus_compute_both_groups"]
        for npu, row in enumerate(stat["per_npu"]):
            assert row["npu_id"] == npu
            close(compute[npu], row["compute_card_ms"])
        population = samples if window == "full" else [s for s in samples if a <= s["admission_ms"] < b]
        slo = metrics["slo_all_requests"] if window == "full" else stat["slo_admitted"]
        for group in ("all", "A", "B"):
            part = population if group == "all" else [s for s in population if s["group"] == group]
            saved = slo if group == "all" else slo["by_group"][group]
            passed = sum(s["ttft_ms"] <= 1.5 * s["compute_ms"] + EPS for s in part)
            assert saved["count"] == len(part) and saved["passed"] == passed
            if part:
                close(saved["rate"], passed / len(part))
        close(metrics[f"{window}_U_percent"], 100 * stat["fleet_utilization"])
        if population:
            close(metrics[f"{window}_SLO_1p5_percent"], 100 * slo["rate"])
    event_data = read(directory / "nominal_demand_events.json.gz")
    for window in ("warm", "full"):
        stat = metrics[window]
        demand_check(event_data["intervals"], stat["ordinary_demand"],
                     stat["start_ms"], stat["end_ms"], config["ssu"])
    case = dict(config=config, metrics=metrics, samples=samples,
                path=str(directory.relative_to(HERE)))
    evidence = dict(config=config["name"], strategy=policy, status="complete",
                    input_fingerprint=metrics["input_fingerprint"],
                    manifest_sha256=metrics["manifest_sha256"],
                    checked_raw_layers=True, checked_raw_request_slo=True,
                    checked_saved_cdf_samples=True, checked_nominal_event_integrals=True,
                    checked_sources=True,
                    artifact_sha256={name: sha(directory / name) for name in names})
    return case, evidence


def per_seed_row(case):
    c, m = case["config"], case["metrics"]
    w, f = m["warm"], m["full"]
    group = group_key(c)
    row = dict(group=group, scenario_kind="capacity_control" if group == CONTROL else "original_five" if group in ORIGINAL_GROUPS else "new_candidate",
               strategy=m["strategy"], seed=c["seed"], npu=32, ssu=c["ssu"],
               warm_U_percent=m["warm_U_percent"], warm_SLO_1p5_percent=m["warm_SLO_1p5_percent"],
               warm_slo_passed=w["slo_admitted"]["passed"], warm_slo_count=w["slo_admitted"]["count"],
               full_U_percent=m["full_U_percent"], full_SLO_1p5_percent=m["full_SLO_1p5_percent"],
               full_slo_passed=m["slo_all_requests"]["passed"], full_slo_count=m["slo_all_requests"]["count"],
               warm_all32_active=w["all_npus_active_whole_window"],
               warm_mixed_active_npu_count=sum(set(p["roles"]) == {"A", "B"} for p in w["per_npu"]),
               warm_mixed_compute_npu_count=sum(all(p["by_group"][g]["compute_card_ms"] > EPS for g in ("A", "B")) for p in w["per_npu"]),
               makespan_ms=m["makespan_ms"], input_fingerprint=m["input_fingerprint"],
               manifest_sha256=m["manifest_sha256"], full_ratio_max=max(s["plotted_ratio"] for s in case["samples"]))
    for role in ("A", "B"):
        row[f"warm_missing_compute_{role}_npu_ids"] = ",".join(str(p["npu_id"]) for p in w["per_npu"] if p["by_group"][role]["compute_card_ms"] <= EPS)
    row["warm_not_continuously_active_npu_ids"] = ",".join(str(p["npu_id"]) for p in w["per_npu"] if abs(p["active_card_ms"] - 2000) >= 1e-6)
    for window, stat in (("warm", w), ("full", f)):
        for kind in ("ordinary_demand", "ordinary_pending_demand"):
            demand = stat[kind]
            prefix = f"{window}_{'nominal' if kind == 'ordinary_demand' else 'internal_pending'}"
            row[prefix + "_strict_under40"] = demand["strictly_under_capacity"]
            row[prefix + "_max_disk_GB_s"] = demand["max_single_ssu_gb_s"]
            row[prefix + "_any_disk_over40_percent"] = demand["any_ssu_overload_fraction"] * 100
        for group in ("A", "B"):
            slo = (m["slo_all_requests"] if window == "full" else stat["slo_admitted"])["by_group"][group]
            row[f"{window}_{group}_SLO_1p5_percent"] = None if slo["rate"] is None else 100 * slo["rate"]
            row[f"{window}_{group}_slo_passed"] = slo["passed"]
            row[f"{window}_{group}_slo_count"] = slo["count"]
    return row


def macros(per_seed, active_groups):
    rows = []
    average = ("warm_U_percent", "warm_SLO_1p5_percent", "full_U_percent", "full_SLO_1p5_percent",
               "warm_A_SLO_1p5_percent", "warm_B_SLO_1p5_percent", "full_A_SLO_1p5_percent", "full_B_SLO_1p5_percent",
               "warm_nominal_any_disk_over40_percent", "full_nominal_any_disk_over40_percent")
    for group in active_groups:
        for policy in POLICIES:
            part = [r for r in per_seed if r["group"] == group and r["strategy"] == policy]
            row = dict(group=group, strategy=policy, seed_count=len(part),
                       seeds=",".join(str(r["seed"]) for r in part),
                       status="complete" if len(part) == 3 else "pending",
                       ssu=4 if group == CONTROL else 6 if group in ORIGINAL_GROUPS else 3)
            for key in average:
                values = [p[key] for p in part if p[key] is not None]
                row[key] = statistics.mean(values) if values else None
                row[key + "_min"] = min(values) if values else None
                row[key + "_max"] = max(values) if values else None
            for window in ("warm", "full"):
                row[f"{window}_slo_total_count"] = sum(p[f"{window}_slo_count"] for p in part)
                row[f"{window}_slo_total_passed"] = sum(p[f"{window}_slo_passed"] for p in part)
                row[f"{window}_nominal_underload_seed_count"] = sum(p[f"{window}_nominal_strict_under40"] for p in part)
                row[f"{window}_nominal_max_disk_GB_s"] = max((p[f"{window}_nominal_max_disk_GB_s"] for p in part), default=None)
                row[f"{window}_internal_pending_underload_seed_count"] = sum(p[f"{window}_internal_pending_strict_under40"] for p in part)
            row["warm_all32_active_seed_count"] = sum(p["warm_all32_active"] for p in part)
            row["warm_all32_compute_AB_seed_count"] = sum(p["warm_mixed_compute_npu_count"] == 32 for p in part)
            row["warm_min_mixed_compute_npu_count"] = min((p["warm_mixed_compute_npu_count"] for p in part), default=None)
            rows.append(row)
    return rows


def old_comparison(macro):
    path = REPO / "results/template_od_baseline_20260919/formula_sensitivity/per_seed_metrics.csv"
    with path.open(encoding="utf-8-sig") as f:
        old = list(csv.DictReader(f))
    result = []
    for row in macro:
        if row["group"] not in ORIGINAL_GROUPS:
            continue
        previous = [p for p in old if p["group"] == row["group"] and p["strategy"] == row["strategy"]]
        assert sorted(int(p["seed"]) for p in previous) == list(SEEDS)
        old_u = statistics.mean(float(p["U_warm_percent"]) for p in previous)
        old_slo = statistics.mean(float(p["SLO_warm_1p5_percent"]) for p in previous)
        ready = row["status"] == "complete"
        result.append(dict(group=row["group"], strategy=row["strategy"], status=row["status"],
                           old_npu=8, old_ssu=1, new_npu=32, new_ssu=6,
                           old_warm_U_percent=old_u, new_warm_U_percent=row["warm_U_percent"] if ready else None,
                           U_difference_pp=row["warm_U_percent"] - old_u if ready else None,
                           old_warm_SLO_1p5_percent=old_slo,
                           new_warm_SLO_1p5_percent=row["warm_SLO_1p5_percent"] if ready else None,
                           SLO_difference_pp=row["warm_SLO_1p5_percent"] - old_slo if ready else None))
    return result, {str(path.relative_to(REPO)): sha(path)}


def configure_font():
    candidates = (Path.home() / ".fonts/msyh.ttc", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})


def render_group(group, cases, output):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    panels = (("all", "全部请求"), ("A", "A：大读取、长计算"), ("B", "B：小读取、短计算"))
    checks = []
    for ax, (role, title) in zip(axes, panels):
        arrays = {}
        expected = {}
        for policy in POLICIES:
            samples = [[s for s in cases[(group, seed, policy)]["samples"] if role == "all" or s["group"] == role]
                       for seed in SEEDS]
            assert len({len(s) for s in samples}) == 1 and samples[0]
            arrays[policy] = [np.sort([s["plotted_ratio"] for s in sample]) for sample in samples]
            expected[policy] = statistics.mean(100 * sum(s["ttft_ms"] <= 1.5 * s["compute_ms"] + EPS for s in sample) / len(sample)
                                                for sample in samples)
        sample_max = max(float(a[-1]) for aa in arrays.values() for a in aa)
        upper = max(1.8, sample_max * 1.03)
        x = np.unique(np.concatenate([np.array([0.9, 1, 1.5, upper])] + [a for aa in arrays.values() for a in aa]))
        curves = {}
        for policy in POLICIES:
            y = np.mean([np.searchsorted(a, x, side="right") / len(a) * 100 for a in arrays[policy]], axis=0)
            pooled = np.sort(np.concatenate(arrays[policy]))
            assert np.allclose(y, np.searchsorted(pooled, x, side="right") / len(pooled) * 100, atol=1e-10, rtol=0)
            close(y[np.where(x == 1.5)[0][0]], expected[policy])
            curves[policy] = y
            ax.step(x, y, where="post", color=COLORS[policy], linestyle=STYLES[policy], linewidth=2.2,
                    label=f"{LABELS[policy]}  {expected[policy]:.2f}%")
            checks.append(dict(group=group, request_group=role, strategy=policy, n_per_seed=len(arrays[policy][0]),
                               SLO_1p5_percent=expected[policy], max_ratio=float(pooled[-1]),
                               equal_seed_ecdf_equals_pooled=True))
        ax.axvline(1.5, color="#778899", linewidth=1.1, linestyle=":")
        ax.set_xlim(0.9, upper)
        if sample_max > 12:
            ax.set_xscale("log")
            ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500) if t <= upper]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_xlabel("归一化耗时（对数横轴）")
        else:
            ax.set_xlabel("归一化耗时")
        ax.set_ylim(0, 103)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=9, framealpha=.95)
        overlaps = [(a, b) for i, a in enumerate(POLICIES) for b in POLICIES[i + 1:]
                    if np.allclose(curves[a], curves[b], atol=1e-10, rtol=0)]
        if overlaps:
            note = "三条曲线重合" if len(overlaps) == 3 else "；".join(f"{LABELS[a].split('（')[0]}/{LABELS[b].split('（')[0]}重合" for a, b in overlaps)
            ax.text(.03, .15, note, transform=ax.transAxes, fontsize=9, color="#5b6470")
    axes[0].set_ylabel("累计请求比例（%）")
    disks = 4 if group == CONTROL else 6 if group in ORIGINAL_GROUPS else 3
    label = "同每卡总容量控制，欠载需审计" if group == CONTROL else "原五组画像" if group in ORIGINAL_GROUPS else "新增候选，不能预设为欠载"
    fig.suptitle(f"{group} · 全输入 TTFT 归一化 CDF", fontsize=16, y=.98)
    fig.text(.5, .905, f"{label} · 32 NPU / {disks} SSU × 40 GB/s（十进制） · seed 7/19/43 等权 · 图例为全输入 SLO×1.5 达标率",
             ha="center", fontsize=10)
    fig.text(.5, .065, "归一化耗时 =（prefill 完成 − 上卡）/ 自身 8 层纯计算；包括启动，不含上卡前排队，不是真实端到端首 token 时延。",
             ha="center", fontsize=9)
    fig.text(.5, .025, "每种子同类请求数相同，已核验等权 CDF 与合并样本 CDF 一致；warm [2,4) 指标另见表格。虚线阈值为 1.5。",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.055, right=.985, bottom=.20, top=.81, wspace=.12)
    path = output / "figures" / f"{group}_ttft_normalized_cdf.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, checks


def percent(value):
    return "—" if value is None else f"{value:.2f}%"


def report_text(macro, comparison, per_seed, completed, pending, figures, expected):
    complete = not pending
    lines = ["# 32 NPU 随机 A/B 输入：ASU、OD、Once 对比", "",
             (f"**正式结果：{expected}/{expected} 个运行全部完整排空并通过汇总核验。**" if complete else
              f"**临时报告：仅 {completed}/{expected} 个运行完整排空；还有 {len(pending)} 个待完成。未齐三种子的行不作为最终均值。**"), "",
             (("本次共完成63次运行：原五组45次、新增R32候选9次、第4组的4盘容量控制9次。三部分分别列出。" if complete and expected == 63 else
               "主计划为原五组加新增候选，共54次；额外4盘容量控制单独列出。") +
              "原五组沿用旧画像和随机输入规则，统一为32 NPU、6 SSU；新增R32候选使用3 SSU。每盘40十进制GB/s、每卡接收上限50十进制GB/s；8层、batch=1、A:B请求数量比1:12、seed 7/19/43、全部arrival=0、RingHash和精确尾块。" +
              ("所有输入均已完整排空，" if complete else "所有输入需完整排空，") + "4500 ms仅用于生成请求数量。"), "",
             "ASU 使用一个共享 FIFO Path；OD 每盘为各 NPU 分配独占 Path、均分 CIR 并允许借用空闲带宽；Once 使用原按层选路与原 QoS 配置。这不是只改变一个选路函数的单变量对照。本轮 OD 未启用固定队深上限。", "",
             "主表：利用率按 warm [2,4) 的真实计算区间积分；SLO 按此窗口上卡的人口、跟踪到最终完成。SLO×1.5 = 上卡至 prefill 完成耗时 ≤ 自身 8 层纯计算的 1.5 倍，不含上卡前排队。三个种子分别计算后等权平均。", "",
             "## 原五组：warm [2,4)", "",
             "| 画像组 | 策略 | 完成种子 | NPU 利用率 | SLO×1.5 | 逐盘严格欠载 | 全32卡持续活跃 | 每卡实际计算 A+B |",
             "|---|---|---:|---:|---:|---|---|---|"]
    if complete:
        original = [r for r in macro if r["group"] in ORIGINAL_GROUPS]
        lead = [
            f"本次原五组在6盘下，三策略warm利用率为{min(r['warm_U_percent'] for r in original):.2f}%–{max(r['warm_U_percent'] for r in original):.2f}%，warm及全人口SLO×1.5均为100%，**没有复现明显低利用率**。实际逐盘普通需求全程严格欠载，但部分固定窗口不满足每卡都计算A/B的覆盖要求，详见卡号表。",
            "",
        ]
        control = {r["strategy"]: r for r in macro if r["group"] == CONTROL}
        if control:
            lead += [f"保持每卡总容量的4盘控制，ASU/OD/Once的warm利用率分别为{control['asu_baseline']['warm_U_percent']:.2f}% / {control['od_baseline']['warm_U_percent']:.2f}% / {control['once']['warm_U_percent']:.2f}%。它和R32候选都包含实际逐盘过载的种子，不能将整组结果称为严格欠载反例。", ""]
        lines[4:4] = lead
    def add_row(row):
        ready = row["status"] == "complete"
        n = row["seed_count"]
        return (f"| {row['group']} | {LABELS[row['strategy']]} | {n}/3 | "
                f"{percent(row['warm_U_percent']) if ready else '待齐'} | "
                f"{percent(row['warm_SLO_1p5_percent']) if ready else '待齐'} | "
                f"{row['warm_nominal_underload_seed_count']}/{n} | {row['warm_all32_active_seed_count']}/{n} | "
                f"{row['warm_all32_compute_AB_seed_count']}/{n}（最少{row['warm_min_mixed_compute_npu_count'] if n else '—'}卡） |")
    lines.extend(add_row(row) for row in macro if row["group"] in ORIGINAL_GROUPS)
    lines += ["", "后三列是满足条件的种子数/已完成种子数；‘每卡 A+B’要求窗口内每张卡确实计算过两类请求。未达 3/3 时不应将该组宣传为满足全部窗口条件。", "",
              "## 新增候选 R32：独立列出", "",
              "本候选改变了画像且只有 3 盘。无等待的平均工作量估计不能证明逐时逐盘欠载；启动及 warm 是否过载，以以下完成轨迹的逐事件审计为准。", "",
              "| 画像组 | 策略 | 完成种子 | NPU 利用率 | SLO×1.5 | 逐盘严格欠载 | 全32卡持续活跃 | 每卡实际计算 A+B |",
              "|---|---|---:|---:|---:|---|---|---|"]
    lines.extend(add_row(row) for row in macro if row["group"] == CANDIDATE)
    lines += ["", "| 候选策略 | warm 最坏盘峰值 GB/s | warm 任一盘过载时间占比 | 全程最坏盘峰值 GB/s | 全程任一盘过载时间占比 | 判定 |",
              "|---|---:|---:|---:|---:|---|"]
    for row in macro:
        if row["group"] != CANDIDATE:
            continue
        if row["status"] != "complete":
            lines.append(f"| {LABELS[row['strategy']]} | — | — | — | — | 待齐三种子 |")
            continue
        verdict = ("warm也发生过载，不能算该窗口欠载反例" if row["warm_nominal_any_disk_over40_percent"] > 0 else
                   "warm触及容量边界，不满足严格欠载" if row["warm_nominal_underload_seed_count"] < 3 else
                   "warm欠载，但全程不是严格欠载" if row["full_nominal_underload_seed_count"] < 3 else "warm与全程均严格欠载")
        lines.append(f"| {LABELS[row['strategy']]} | {row['warm_nominal_max_disk_GB_s']:.4f} | {percent(row['warm_nominal_any_disk_over40_percent'])} | {row['full_nominal_max_disk_GB_s']:.4f} | {percent(row['full_nominal_any_disk_over40_percent'])} | {verdict} |")
    lines += ["", "峰值取三种子中最坏盘；时间占比为三个种子的等权均值。普通需求是当前已上卡请求逐盘 V/C 之和，包括其 I/O 等待；不额外加入下一请求 L0 预取突发，但这些突发造成的所有等待仍计入利用率和 SLO。‘严格欠载’要求每盘所有事件区间均低于 40；过载时间使用大于 40 的区间。逐盘及内部层待处理包络见 per_ssu_audit.csv。", "",
              "## 全输入人口与 CDF（包含启动）", "",
              "下表和图使用全部输入请求，与上面的 warm 上卡人口不同。归一化耗时 =（完成−上卡）/自身 8 层纯计算。每种子同类请求数相同，因此已验证三个种子等权 CDF 与合并样本 CDF 等价；warm 人口数一般不同，warm 仍必须按种子等权。", "",
              "| 画像组 | 策略 | 全输入 SLO×1.5 | A 达标率 | B 达标率 | 请求总数 | 全程 NPU 利用率 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in macro:
        if row["status"] == "complete":
            lines.append(f"| {row['group']} | {LABELS[row['strategy']]} | {percent(row['full_SLO_1p5_percent'])} | {percent(row['full_A_SLO_1p5_percent'])} | {percent(row['full_B_SLO_1p5_percent'])} | {row['full_slo_total_count']} | {percent(row['full_U_percent'])} |")
        else:
            lines.append(f"| {row['group']} | {LABELS[row['strategy']]} | 待齐 | — | — | — | — |")
    if any(row["group"] == CONTROL for row in macro):
        lines += ["", "## 4盘容量控制：保持每卡总容量5 GB/s", "",
                  "8卡1盘时，每卡平均总容量为40/8=5 GB/s；32卡6盘增至240/32=7.5 GB/s，多了50%。因此6盘结果的提高可能包含容量因素，不能归因于卡数增加。第4组另跑32卡4盘，保持160/32=5 GB/s；画像、随机规则和每卡请求数保留。4盘静态上界不保证欠载，必须按真实事件判断。它仍改变了多盘分布和共享竞争，不能把所有差异都解释为容量。", "",
                  "| 画像组 | 策略 | 完成种子 | NPU 利用率 | SLO×1.5 | 逐盘严格欠载 | 全32卡持续活跃 | 每卡实际计算 A+B |",
                  "|---|---|---:|---:|---:|---|---|---|"]
        lines.extend(add_row(row) for row in macro if row["group"] == CONTROL)
        lines += ["", "| 控制策略 | warm 最坏盘峰值 GB/s | warm 任一盘过载时间占比 | 全程严格欠载种子 |",
                  "|---|---:|---:|---:|"]
        for row in macro:
            if row["group"] == CONTROL:
                ready = row["status"] == "complete"
                peak = f"{row['warm_nominal_max_disk_GB_s']:.4f}" if ready else "待齐"
                lines.append(f"| {LABELS[row['strategy']]} | {peak} | {percent(row['warm_nominal_any_disk_over40_percent']) if ready else '待齐'} | {row['full_nominal_underload_seed_count']}/{row['seed_count']} |")
    lines += ["", "## 固定窗口内的混合覆盖限制", "",
              "每卡输入含A/B，不保证固定2秒窗口内每张卡都计算过A和B。下面保留所有未满足覆盖条件的种子和卡号；不剔除种子、不移动窗口。相关利用率仍是真实值，但不能称为满足‘warm内每卡都运行两类’的结果。", "",
              "| 场景 | 策略 | seed | 未计算A的卡号 | 未计算B的卡号 | 未持续活跃的卡号 |",
              "|---|---|---:|---|---|---|"]
    missing_rows = [row for row in per_seed if row["warm_missing_compute_A_npu_ids"] or row["warm_missing_compute_B_npu_ids"] or row["warm_not_continuously_active_npu_ids"]]
    for row in missing_rows:
        lines.append(f"| {row['group']} | {LABELS[row['strategy']]} | {row['seed']} | {row['warm_missing_compute_A_npu_ids'] or '无'} | {row['warm_missing_compute_B_npu_ids'] or '无'} | {row['warm_not_continuously_active_npu_ids'] or '无'} |")
    if not missing_rows:
        lines.append("| 已完成运行 | 全部 | — | 无 | 无 | 无 |" if completed else "| 待结果 | — | — | — | — | — |")
    lines += ["", "全程利用率包含启动和最后排空，不能当作长期稳态或替代warm利用率。CDF保留全部长尾；必要时横轴使用对数刻度并明确标注。完全重合的曲线会在图内标明。", ""]
    lines.extend(f"- [{g} 全输入 CDF](figures/{g}_ttft_normalized_cdf.png)" for g in GROUPS if g in figures)
    lines += ["", "## 原 8 卡结果如何变化", "",
              "旧对照是8 NPU / 1 SSU，新原五组是32 NPU / 6 SSU；每卡总容量从5增至7.5 GB/s。输入画像和每卡数量规则保留，但容量、Path竞争和RingHash分盘都变了。因此差异不能单独归因于NPU数量，应结合上面的4盘控制。此处旧、新warm SLO均从逐种子数据等权计算。", "",
              "| 画像组 | 策略 | 旧8卡 U | 新32卡 U | U 变化（百分点） | 旧 warm SLO | 新 warm SLO |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        change = "—" if row["U_difference_pp"] is None else f"{row['U_difference_pp']:+.2f}"
        lines.append(f"| {row['group']} | {LABELS[row['strategy']]} | {percent(row['old_warm_U_percent'])} | {percent(row['new_warm_U_percent'])} | {change} | {percent(row['old_warm_SLO_1p5_percent'])} | {percent(row['new_warm_SLO_1p5_percent'])} |")
    lines += ["", "## 数学、来源与复现", "",
              "- [原五组扩到32卡的容量与公式说明](math_notes.md)。低于32K的 B 保留原长度外推，不能称为直接实测画像。",
              "- [新增候选的构造与预先推导](configs/candidate_math.md)。平均负载估计不是逐时欠载证明。",
              "- [逐种子指标](per_seed_metrics.csv)、[三种子汇总](macro_summary.csv)、[逐盘审计](per_ssu_audit.csv)、[逐卡 warm 指标](per_npu_warm.csv)、[8到32卡对照](8to32_comparison.csv)。",
              "- [汇总检查与原始文件 SHA](summary_checks.json)；每运行原始输入、层时序、请求样本和事件需求位于 runs/。", "",
              "- [独立审计说明](AUDIT.md)、[独立审计程序](audit_results.py)、[审计结果](verification.json)。其完成状态以verification.json为准，不能仅凭本汇总替代。",
              "- [运行入口](runner.py)、[配置目录](configs/)、[汇总与绘图入口](summarize.py)。", "",
              "汇总核验包括请求/层计算积分、SLO计数、CDF边界容差、逐盘事件积分、各策略同输入 manifest 指纹及运行源码 SHA。浮点误差只在 1 和 1.5 的理论边界按 1e-9 ms 规范化绘图副本，原始样本不改。", "",
              "单场复现使用新的空输出目录，避免覆盖正式结果；将strategy改为asu_baseline或once可做配对：", "",
              "```bash", "python results/formula_ab_32npu_20260921/runner.py \\",
              "  --config results/formula_ab_32npu_20260921/configs/XY12_16_n32_s6_seed7.json \\",
              "  --strategy od_baseline --output /tmp/formula_ab32_replay_xy12_16_od_seed7", "```", "",
              "```bash", "python results/formula_ab_32npu_20260921/summarize.py --require-complete", "```", ""]
    if pending:
        lines += ["待完成运行：", ""] + [f"- {p['config']} / {p['strategy']}" for p in pending] + [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    configs = [(p, read(p)) for p in sorted((HERE / "configs").glob("*.json"))]
    plan_keys = {(group_key(c), c["seed"]) for _, c in configs}
    main_keys = {(g, s) for g in MAIN_GROUPS for s in SEEDS}
    assert main_keys <= plan_keys
    control_keys = {(CONTROL, s) for s in SEEDS}
    assert plan_keys in (main_keys, main_keys | control_keys)
    assert len(configs) == len(plan_keys)
    active_groups = tuple(g for g in GROUPS if any(key[0] == g for key in plan_keys))
    expected = len(configs) * len(POLICIES)
    cases, pending, evidence, source_cache = {}, [], [], {}
    for path, config in configs:
        for policy in POLICIES:
            case, check = verify_run(path, config, policy, source_cache)
            if case is None:
                pending.append(check)
            else:
                cases[(group_key(config), config["seed"], policy)] = case
                evidence.append(check)
    for group in GROUPS:
        for seed in SEEDS:
            same = [c["metrics"] for (g, s, _), c in cases.items() if g == group and s == seed]
            assert len({m["input_fingerprint"] for m in same}) <= 1
            assert len({m["manifest_sha256"] for m in same}) <= 1
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    per_seed = [per_seed_row(cases[key]) for key in sorted(cases, key=lambda k: (GROUPS.index(k[0]), POLICIES.index(k[2]), k[1]))]
    macro = macros(per_seed, active_groups)
    comparison, reference_sha = old_comparison(macro)
    disk_rows, npu_rows = [], []
    for key, case in cases.items():
        group, seed, policy = key
        for window in ("warm", "full"):
            for kind in ("ordinary_demand", "ordinary_pending_demand"):
                stat = case["metrics"][window][kind]
                for disk, peak in enumerate(stat["peak_gb_s_by_ssu"]):
                    disk_rows.append(dict(group=group, strategy=policy, seed=seed, window=window, demand_kind=kind,
                                          ssu=disk, peak_GB_s=peak, mean_GB_s=stat["mean_gb_s_by_ssu"][disk],
                                          over40_percent=stat["overload_fraction_by_ssu"][disk] * 100,
                                          strictly_under40=peak < 40 - EPS))
        for npu in case["metrics"]["warm"]["per_npu"]:
            npu_rows.append(dict(group=group, strategy=policy, seed=seed, npu=npu["npu_id"], U_percent=npu["utilization"] * 100,
                                 active_ms=npu["active_card_ms"], A_compute_ms=npu["by_group"]["A"]["compute_card_ms"],
                                 B_compute_ms=npu["by_group"]["B"]["compute_card_ms"]))
    for name, rows in (("per_seed_metrics", per_seed), ("macro_summary", macro), ("per_ssu_audit", disk_rows),
                       ("per_npu_warm", npu_rows), ("8to32_comparison", comparison)):
        write_csv(output / f"{name}.csv", rows)
    plots, cdf_checks = {}, []
    if not args.no_plots:
        configure_font()
        for group in GROUPS:
            if all((group, seed, policy) in cases for seed in SEEDS for policy in POLICIES):
                path, checked = render_group(group, cases, output)
                plots[group] = sha(path)
                cdf_checks.extend(checked)
    text = report_text(macro, comparison, per_seed, len(cases), pending, plots, expected)
    (output / "README.md").write_text(text, encoding="utf-8")
    checks = dict(status="complete" if not pending else "pending", expected_runs=expected, main_plan_expected_runs=54,
                  capacity_control_expected_runs=expected - 54, completed_runs=len(cases),
                  pending=pending, cases=evidence, reference_sha256=reference_sha,
                  same_input_per_group_seed=True, cdf_checks=cdf_checks,
                  figures_sha256=plots, visual_review="pending_manual_review",
                  report_script_sha256=sha(Path(__file__)),
                  macro_definition="Equal mean of independently calculated seed7/19/43 percentages; partial rows labelled pending",
                  full_cdf_definition="All completed input requests including startup; equal-seed ECDF; checked equal seed counts",
                  metrics_window_ms=[2000, 4000])
    (output / "summary_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=checks["status"], complete=len(cases), pending=len(pending), png_count=len(plots), output=str(output))))
    if args.require_complete and pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
