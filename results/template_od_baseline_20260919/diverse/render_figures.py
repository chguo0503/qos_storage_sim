#!/usr/bin/env python3
"""Extend the template's exact frozen-input comparisons with completed OD runs.

This renderer does not run simulations or invoke archived report generators.
Only the seven explicitly allowed PNGs below are produced. Archived Baseline
and Once CDFs are independently reconstructed and checked point by point.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TEMPLATE = ROOT / "template/qos_experiments_20260919"
DIVERSE = TEMPLATE / "repository_source/results/diverse_data_ssu3_l3_20260916"
NEAR = TEMPLATE / "03_continuous_underload/near35"
FIGURES = HERE / "figures"
SEEDS = (7, 19, 43)
REGIMES = ("full", "semi", "near35")
LABELS = {"full": "持续过载输入 · full 24画像", "semi": "局部过载输入 · semi 24画像",
          "near35": "平均欠载、允许局部过载 · near35 24画像"}
POLICIES = ("baseline", "once", "od_baseline")
NAMES = {"baseline": "原 Baseline（ASU）", "once": "流量分配（原始 Once）", "od_baseline": "OD Baseline"}
STYLES = {"baseline": ("#4b5563", "--"), "once": ("#2474b7", "-"), "od_baseline": ("#d45e00", "-.")}
ALIASES = {"Baseline Random": "baseline", "流量分配 Random": "once", "Once Random": "once"}
ALLOWED_FIGURES = {f"{regime}_random_ttft_ratio_cdf_three_strategies.png" for regime in REGIMES}
ALLOWED_FIGURES |= {f"{regime}_od_baseline_per_ssu_seed7.png" for regime in REGIMES}
ALLOWED_FIGURES.add("semi_od_baseline_32npu_timeline_seed7.png")
SOURCES: dict[str, str] = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracked(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return data


def read(path):
    path = Path(path)
    data = tracked(path)
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def read_csv(path):
    return list(csv.DictReader(tracked(path).decode("utf-8-sig").splitlines()))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_path(regime, seed):
    if regime == "near35":
        return NEAR / f"source/results/diverse_near35_20260916/inputs/near35_seed{seed}.json.gz"
    return DIVERSE / f"inputs/{regime}_seed{seed}.json.gz"


def curve(arrays, x):
    return np.mean([np.searchsorted(arrays[seed], x, side="right") / len(arrays[seed])
                    for seed in SEEDS], axis=0)


def warm(raw):
    return next(a for a in raw["analysis"] if a["start_ms"] == 2000 and a["end_ms"] == 4000)


def layer_time_accounting(summary):
    compute = np.zeros(32)
    active = np.zeros(32)
    wait = np.zeros(32)
    idle = np.zeros(32)
    segments = defaultdict(lambda: {"compute": [], "wait": [], "active": []})
    def clip(a, b):
        a, b = max(2000.0, a), min(4000.0, b)
        return a, max(a, b)
    for batch in summary["microbatch_metrics"]:
        assert batch["batch_size"] == 1
        npu = batch["npu_id"]
        a, b = clip(batch["admission_time_ms"], batch["completion_time_ms"])
        if b > a:
            active[npu] += b - a
            segments[npu]["active"].append((a, b))
        previous_end = batch["admission_time_ms"]
        for layer in batch["layer_metrics"]:
            start, end = layer["compute_start_ms"], layer["compute_end_ms"]
            a, b = clip(previous_end, start)
            if b > a:
                wait[npu] += b - a
                segments[npu]["wait"].append((a, b))
            a, b = clip(start, end)
            if b > a:
                compute[npu] += b - a
                segments[npu]["compute"].append((a, b))
            previous_end = end
    idle = 2000.0 - active
    assert np.allclose(compute + wait, active, rtol=0, atol=1e-7)
    assert np.min(idle) >= -1e-7
    return dict(compute_ms=compute, active_ms=active, wait_ms=wait, idle_ms=idle,
                per_npu_U_percent=compute / 20, U_percent=float(compute.sum() / 640),
                segments=segments)


def check_sample(row, request, *, normalize_one):
    ideal = 8 * request["load"]["per_layer_us"] / 1000
    elapsed = row["completion_ms"] - row["admission_ms"]
    assert 2000 <= row["admission_ms"] < 4000
    assert math.isfinite(elapsed) and elapsed >= ideal - 1e-8
    assert math.isclose(row["ideal_ms"], ideal, rel_tol=0, abs_tol=1e-8)
    raw_ratio = elapsed / ideal
    expected = raw_ratio
    if normalize_one and abs(elapsed - ideal) <= 1e-9:
        expected = 1.0
    if elapsed > 1.5 * ideal and elapsed <= 1.5 * ideal + 1e-9:
        expected = 1.5
    assert math.isclose(row["ratio"], expected, rel_tol=0, abs_tol=1e-11)
    assert (row["ratio"] <= 1.5) == (elapsed <= 1.5 * ideal + 1e-9)
    row.update(passed=elapsed <= 1.5 * ideal + 1e-9,
               completed_after_window=row["completion_ms"] > 4000,
               raw_ratio=raw_ratio)


def load_original():
    manifests = {(r, s): read(manifest_path(r, s)) for r in REGIMES for s in SEEDS}
    requests = {key: {q["request_id"]: q for q in m["requests"]} for key, m in manifests.items()}
    for (regime, seed), m in manifests.items():
        meta = m["metadata"]
        assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"], meta["seed"]) == (32, 3, 8, seed)
        assert len(m["requests"]) == (960 if regime == "semi" else 1344)
    samples = []
    for row in read_csv(DIVERSE / "figures/ttft_cdf_three_strategies/request_samples.csv"):
        if row["policy"] not in ("baseline", "once"):
            continue
        item = dict(regime=row["scenario"], policy=row["policy"], seed=int(row["seed"]),
                    request_id=int(row["request_id"]), npu_id=int(row["npu_id"]),
                    admission_ms=float(row["admission_ms"]), completion_ms=float(row["completion_ms"]),
                    ideal_ms=float(row["pure_compute_ms"]), ratio=float(row["latency_ratio"]))
        check_sample(item, requests[item["regime"], item["seed"]][item["request_id"]], normalize_one=False)
        samples.append(item)
    for row in read_csv(NEAR / "near35_random_results/data/request_samples.csv"):
        item = dict(regime="near35", policy=ALIASES[row["policy"]], seed=int(row["seed"]),
                    request_id=int(row["request_id"]), npu_id=int(row["npu"]),
                    admission_ms=float(row["admission_ms"]), completion_ms=float(row["completion_ms"]),
                    ideal_ms=float(row["SLO_base_ms"]), ratio=float(row["SLO_multiple"]))
        check_sample(item, requests["near35", item["seed"]][item["request_id"]], normalize_one=True)
        samples.append(item)
    arrays = {(r, p, s): np.sort([q["ratio"] for q in samples
                                if (q["regime"], q["policy"], q["seed"]) == (r, p, s)])
              for r in REGIMES for p in ("baseline", "once") for s in SEEDS}
    assert all(len(a) for a in arrays.values())
    refs = {}
    # The template archived the macro table and individual CDF samples, but
    # omitted these per-seed tables. Read the original repository tables only
    # after checking that all six corresponding frozen inputs are identical.
    original_tables = ROOT / "results/diverse_data_ssu3_l3_20260916"
    for regime in ("full", "semi"):
        for seed in SEEDS:
            assert read(original_tables / f"inputs/{regime}_seed{seed}.json.gz") == manifests[regime, seed]
    for row in read_csv(original_tables / "comparison.csv") + read_csv(original_tables / "once_control_comparison.csv"):
        if row["window"] != "warm_2_4s" or row["policy"] not in ("baseline", "once"):
            continue
        refs[row["scenario"], row["policy"], int(row["seed"])] = dict(
            U_percent=float(row["U_percent"]), count=int(row["slo_count"]), passed=int(row["slo_passed"]))
    for row in read_csv(NEAR / "near35_random_results/data/per_seed_metrics.csv"):
        if row["window"] != "warm_2_4s":
            continue
        refs["near35", ALIASES[row["policy"]], int(row["seed"])] = dict(
            U_percent=float(row["NPU_utilization_percent"]), count=int(row["requests"]),
            passed=round(int(row["requests"]) * float(row["SLO1_5_percent"]) / 100))
    for key, a in arrays.items():
        assert (len(a), int(np.count_nonzero(a <= 1.5))) == (refs[key]["count"], refs[key]["passed"])
    for row in read_csv(DIVERSE / "threeway_macro_summary.csv"):
        if row["window"] != "warm_2_4s" or row["policy"] not in ("baseline", "once"):
            continue
        rows = [refs[row["scenario"], row["policy"], seed] for seed in SEEDS]
        assert abs(np.mean([r["U_percent"] for r in rows]) - float(row["mean_U_percent"])) < 1e-10
        assert abs(np.mean([100*r["passed"]/r["count"] for r in rows]) - float(row["mean_slo_percent"])) < 1e-10
    checks = {}
    for regime in REGIMES:
        if regime == "near35":
            archived = read_csv(NEAR / "near35_random_results/data/cdf_points.csv")
        else:
            area = "01_continuous_overload/full24" if regime == "full" else "02_partial_overload/semi24"
            archived = read_csv(TEMPLATE / area / "cdf/cdf_points_two_strategies.csv")
        for policy in ("baseline", "once"):
            rows = [q for q in archived if ALIASES.get(q["policy"], q["policy"]) == policy]
            x = np.array([float(q["SLO_multiple" if regime == "near35" else "latency_ratio"]) for q in rows])
            y = np.array([float(q["CDF_percent" if regime == "near35" else "macro_cdf"]) for q in rows])
            if regime == "near35":
                y /= 100
            actual = curve({s: arrays[regime, policy, s] for s in SEEDS}, x)
            error = float(np.max(np.abs(y - actual)))
            assert error < 1e-12, (regime, policy, error)
            checks[f"{regime}_{policy}"] = dict(archived_points=len(x), maximum_cdf_error=error,
                slo_1p5_percent=float(curve({s: arrays[regime, policy, s] for s in SEEDS}, np.array([1.5]))[0] * 100))
    return manifests, samples, arrays, refs, checks


def load_od(manifests, samples, arrays, refs, *, regimes=REGIMES):
    raws, audits = {}, {}
    for regime in regimes:
        parity_folder = HERE / f"runs/{regime}_asu_baseline_seed7"
        parity = read(parity_folder / "parity.json")
        parity_command = read(parity_folder / "command.json")
        assert parity["passed"] and parity["same_input"]
        assert parity_command["status"] == "complete" and parity_command["source_unchanged"]
        for seed in SEEDS:
            folder = HERE / f"runs/{regime}_od_baseline_seed{seed}"
            command = read(folder / "command.json")
            assert command["status"] == "complete", folder
            assert command["policy"] == "od_baseline" and not command.get("smoke"), folder
            assert command["source_unchanged"], folder
            if "checks" in command:
                assert all(command["checks"].values()), folder
            raw = read(folder / "result.json.gz")
            manifest = read(folder / "manifest.json.gz")
            assert manifest == manifests[regime, seed], ("input changed", regime, seed)
            for field, filename in (("result_sha256", "result.json.gz"), ("manifest_sha256", "manifest.json.gz")):
                if field in command:
                    assert command[field] == sha(folder / filename)
            assert all(raw["summary"]["invariants"].values())
            assert raw["summary"]["n_layers"] == 8
            assert raw["summary"]["num_npu"] == 32 and raw["summary"]["num_ssu"] == 3
            assert raw["input_fingerprint"] == manifest["input_fingerprint"]
            assert raw.get("strategy", "od_baseline") == "od_baseline"
            byid = {q["request_id"]: q for q in manifest["requests"]}
            records = raw["summary"]["request_metrics"]
            assert len(records) == len(byid) == len({q["request_id"] for q in records})
            assert {q["request_id"] for q in records} == set(byid)
            cohort = []
            for q in records:
                if not 2000 <= q["admission_time_ms"] < 4000:
                    continue
                ideal = 8 * byid[q["request_id"]]["load"]["per_layer_us"] / 1000
                elapsed = q["completion_time_ms"] - q["admission_time_ms"]
                ratio = elapsed / ideal
                if abs(elapsed - ideal) <= 1e-9:
                    ratio = 1.0
                if 1.5 * ideal < elapsed <= 1.5 * ideal + 1e-9:
                    ratio = 1.5
                row = dict(regime=regime, policy="od_baseline", seed=seed,
                           request_id=q["request_id"], npu_id=q["npu_id"],
                           admission_ms=q["admission_time_ms"], completion_ms=q["completion_time_ms"],
                           ideal_ms=ideal, ratio=ratio)
                check_sample(row, byid[q["request_id"]], normalize_one=True)
                cohort.append(row)
            a = warm(raw)
            count, passed = len(cohort), sum(q["passed"] for q in cohort)
            assert (count, passed) == (a["slo"]["count"], a["slo"]["passed"])
            assert {q["request_id"] for q in cohort} == set(a["cohort_request_ids"])
            accounting = layer_time_accounting(raw["summary"])
            assert math.isclose(accounting["U_percent"], a["U_percent"], abs_tol=1e-8, rel_tol=0)
            assert np.allclose(accounting["per_npu_U_percent"], a["per_npu_U_percent"], atol=1e-8, rtol=0)
            assert np.allclose(accounting["active_ms"], 2000, atol=1e-7, rtol=0)
            key = (regime, "od_baseline", seed)
            arrays[key] = np.sort([q["ratio"] for q in cohort])
            refs[key] = dict(U_percent=a["U_percent"], count=count, passed=passed)
            samples.extend(cohort)
            if seed == 7:
                raws[regime] = raw
            audits[f"{regime}_seed{seed}"] = dict(status="complete", input_fingerprint=manifest["input_fingerprint"],
                same_decoded_manifest=True, asu_parity_seed7_passed=True,
                U_independent_recalculation=True, cohort_count=count, slo_passed=passed,
                result_path=(folder / "result.json.gz").relative_to(ROOT).as_posix())
    return raws, audits


def setup_plotting():
    font = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 11, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    assert name in ALLOWED_FIGURES, name
    path = FIGURES / name
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def render_cdf(regime, arrays):
    maximum = max(float(arrays[regime, policy, seed][-1]) for policy in POLICIES for seed in SEEDS)
    original_right = {"full": 17.0, "semi": 3.0, "near35": 2.0}[regime]
    right = max(original_right, math.ceil(maximum * 4) / 4)
    left = 0.5 if regime == "near35" else 0.9
    fig, ax = plt.subplots(figsize=(12.7, 7.7))
    fig.subplots_adjust(left=.09, right=.97, top=.79, bottom=.23)
    points = []
    for policy in POLICIES:
        aa = {seed: arrays[regime, policy, seed] for seed in SEEDS}
        x = np.unique(np.concatenate(([left, 1.5, right], *aa.values())))
        y = curve(aa, x)
        slo = float(curve(aa, np.array([1.5]))[0])
        color, style = STYLES[policy]
        ax.step(x, y, where="post", color=color, ls=style, lw=2.35,
                label=f"{NAMES[policy]}  |  SLO×1.5={100 * slo:.2f}%")
        ax.plot(1.5, slo, "o", color=color, ms=6, mec="white", mew=.7, zorder=6)
        points.extend(dict(regime=regime, policy=policy, ratio=float(xx), macro_cdf=float(yy)) for xx, yy in zip(x, y))
    ax.axvline(1.5, color="#737373", ls=":", lw=1.2)
    ax.text(1.5, 1.048, "SLO×1.5", ha="center", fontsize=10)
    ax.set(xlim=(left, right), ylim=(0, 1.035), ylabel="累计请求比例（三个种子的 CDF 等权平均）",
           xlabel="归一化耗时 = 接纳至 Prefill 完成耗时 / 本请求 8 层纯计算时间")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=.23)
    if regime == "full":
        ax.set_xticks(sorted(set([1, 1.5] + list(np.arange(3, right + .01, 2)) + [right])))
    else:
        ax.set_xticks(sorted(set(list(np.arange(math.ceil(left * 2) / 2, right + .01, .5)) + [1.5, right])))
    ax.legend(loc="lower right", framealpha=.97, fontsize=10.5, labelspacing=.8)
    fig.text(.09, .955, f"{LABELS[regime]}：三策略 CDF", fontsize=18, weight="bold", va="top")
    fig.text(.09, .885, "32 NPU / 3 SSU × 40 GiB/s · Random · 原冻结条带输入 · warm [2,4)秒接纳", fontsize=11, color="#526171")
    fig.text(.09, .12, "同种子共用完整输入；seed 7、19、43 等权平均。原 Baseline / Once 曲线逐点核对保留。", fontsize=10)
    fig.text(.09, .075, "所有入选请求跟踪至完成，保留完整长尾；不含接纳前排队，不代表真实首 token 时延。", fontsize=10, color="#526171")
    fig.text(.09, .035, "策略会改变接纳时刻，因此 warm 入选请求集合可以不同。", fontsize=10, color="#526171")
    return save(fig, f"{regime}_random_ttft_ratio_cdf_three_strategies.png"), points


def bandwidth(raw):
    a = warm(raw)
    seg = np.asarray(a["demand"]["segments"], dtype=float)
    assert seg.shape[1] == 5
    assert np.allclose(seg[1:, 0], seg[:-1, 1], rtol=0, atol=1e-8)
    assert abs(seg[0, 0] - 2000) < 1e-8 and abs(seg[-1, 1] - 4000) < 1e-8
    durations = seg[:, 1] - seg[:, 0]
    assert np.min(durations) > 0
    if "warm_ssd_10ms_GiB_s" in raw:
        supply = np.asarray(raw["warm_ssd_10ms_GiB_s"], dtype=float)
    else:
        fine = np.asarray(raw["bandwidth_2ms"]["ssd_GiB_s"], dtype=float)
        assert fine.shape == (3, 1000)
        supply = fine.reshape(3, 200, 5).mean(axis=2)
    assert supply.shape == (3, 200) and supply.min() >= -1e-8 and supply.max() <= 40 + 1e-8
    actual_mean = supply.mean(axis=1)
    demand_mean = np.sum(seg[:, 2:] * durations[:, None], axis=0) / 2000
    overload = np.sum((seg[:, 2:] > 40) * durations[:, None], axis=0) / 20
    assert np.allclose(actual_mean, a["SSD_GiB_s"], rtol=0, atol=1e-7)
    assert np.allclose(demand_mean, a["demand"]["per_disk_mean_GiB_s"], rtol=0, atol=1e-7)
    assert np.allclose(overload, a["demand"]["per_disk_overload_percent"], rtol=0, atol=1e-6)
    return seg, supply, demand_mean, actual_mean, overload


def render_disks(regime, raw):
    seg, supply, demand_mean, actual_mean, overload = bandwidth(raw)
    edges = np.r_[seg[:, 0], seg[-1, 1]] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.975, top=.805, bottom=.12, hspace=.32)
    maximum = max(60.0, math.ceil(float(seg[:, 2:].max()) / 10) * 10)
    for disk, ax in enumerate(axes):
        ax.stairs(seg[:, disk + 2], edges, baseline=None, color="#D76D00", lw=1.8, label="当前已接纳请求参考需求 V/C")
        ax.stairs(supply[disk], np.linspace(2, 4, 201), baseline=None, color="#1767AC", lw=1.6,
                  label="实际 SSD 供给（10 ms 平均）")
        ax.axhline(40, color="#555555", ls="--", lw=1.0, label="单盘容量 40 GiB/s")
        ax.set(xlim=(2, 4), ylim=(0, maximum * 1.06), ylabel=f"SSU {disk}\n带宽（GiB/s）")
        ax.set_title(f"平均需求 {demand_mean[disk]:.2f} · 平均供给 {actual_mean[disk]:.2f} GiB/s"
                     f" · 需求>40 的时间 {overload[disk]:.2f}%", loc="left", fontsize=10.5)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle(f"{LABELS[regime]}\nOD Baseline：逐盘需求与真实 SSD 供给", fontsize=16, y=.974)
    fig.text(.085, .893, "32 NPU / 3 SSU × 40 GiB/s · 原冻结条带输入 · seed 7 · warm [2,4)秒", fontsize=11, color="#526171")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.53, .872), frameon=False, fontsize=10)
    fig.text(.085, .054, "需求按当前请求逐盘 V/C 逐事件统计，计算与等待期间均保留；供给由本次 OD 物理读取独立积分。", fontsize=10, color="#526171")
    fig.text(.085, .024, "跨请求预取不重复叠加为参考需求，其真实读取计入供给；两线之比不等于 NPU 瞬时利用率。", fontsize=10, color="#526171")
    return save(fig, f"{regime}_od_baseline_per_ssu_seed7.png"), dict(
        demand_mean_GiB_s=demand_mean.tolist(), actual_mean_GiB_s=actual_mean.tolist(),
        overload_percent=overload.tolist(), physical_supply_max_GiB_s=float(supply.max()))


def render_timeline(raw):
    accounting = layer_time_accounting(raw["summary"])
    seg, supply, _, _, _ = bandwidth(raw)
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 3.05), left=.075, right=.82,
                         top=.88, bottom=.10, hspace=.30)
    top, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    top.stairs(seg[:, 2:].sum(axis=1), np.r_[seg[:, 0], seg[-1, 1]] / 1000,
               baseline=None, color="#D76D00", lw=1.7, label="整机参考需求")
    top.stairs(supply.sum(axis=0), np.linspace(2, 4, 201), baseline=None,
               color="#1767AC", lw=1.6, label="实际 SSD 供给（10 ms 平均）")
    top.axhline(120, color="#667481", ls="--", lw=1.1, label="总容量 120 GiB/s")
    top.set(xlim=(2, 4), ylim=(0, max(130, float(seg[:, 2:].sum(axis=1).max()) * 1.1)), ylabel="整机带宽（GiB/s）")
    top.grid(alpha=.18)
    top.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), ncol=3, frameon=False, fontsize=10)
    for npu in range(32):
        row = accounting["segments"][npu]
        for kind, color in (("wait", "#E28B16"), ("compute", "#168574")):
            spans = [(a / 1000, (b - a) / 1000) for a, b in row[kind]]
            if spans:
                ax.broken_barh(spans, (npu - .37, .74), facecolors=color, edgecolors="none")
        ax.text(1.045, npu, f"{accounting['wait_ms'][npu]:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9.2, color="#526171")
        ax.text(1.175, npu, f"{accounting['per_npu_U_percent'][npu]:.2f}%", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9.2, color="#168574")
    ax.text(1.045, 1.04, "等待 ms", transform=ax.transAxes, ha="right", fontsize=10)
    ax.text(1.175, 1.04, "利用率", transform=ax.transAxes, ha="right", fontsize=10)
    ax.set(xlim=(2, 4), ylim=(31.7, -.7), yticks=range(32), ylabel="NPU ID", xlabel="时间（秒）")
    ax.grid(axis="x", alpha=.17)
    ax.legend(handles=[Patch(color="#168574", label="计算"), Patch(color="#E28B16", label="数据未到齐而等待")],
              loc="upper left", bbox_to_anchor=(0, 1.065), frameon=False, ncol=2, fontsize=10)
    fig.suptitle("semi 24画像 · OD Baseline · 全部 32 NPU 计算时序", fontsize=18, y=.977)
    fig.text(.075, .94, "同一份 seed 7 冻结条带输入 · 3 SSU × 40 GiB/s · warm [2,4)秒", fontsize=12, color="#526171")
    fig.text(.075, .06, f"计算 {accounting['compute_ms'].sum()/1000:.3f} 卡·秒  ·  "
                      f"等待 {accounting['wait_ms'].sum()/1000:.3f} 卡·秒  ·  整机 U={accounting['U_percent']:.2f}%", fontsize=14, color="#168574")
    fig.text(.075, .025, "橙色表示计算已可继续但下一层数据未到齐，包含盘/接收链路影响；未把整段等待归因于盘内排队。", fontsize=10, color="#526171")
    return save(fig, "semi_od_baseline_32npu_timeline_seed7.png"), dict(
        U_percent=accounting["U_percent"], compute_card_ms=float(accounting["compute_ms"].sum()),
        wait_card_ms=float(accounting["wait_ms"].sum()), all_npus_active=True)


def write_figure_readme(summary_rows):
    lines = ["# 模板原输入：新增 OD 比较图", "",
             "新增 OD 使用模板原冻结输入，不重新生成 Ring hash 落盘。原 Baseline（ASU）和原始 Once 的统计与曲线保留；三组 seed 7 的 ASU 重放均通过原结果时序一致性检查。", "",
             "配置：32 NPU、3 SSU × 40 GiB/s、8 层、Random；warm 统计为 [2,4) 秒，种子 7、19、43 等权平均。", "",
             "OD在每盘启用32条独占Path，每张NPU绑定一条，静态CIR为40/32=1.25 GiB/s。空闲容量按原两级调度借用；等CIR不代表任意时刻各卡实际供给都相等。相较原单Path Baseline，这同时改变了隔离方式和带宽分配配置。", "",
             "| 输入组 | 策略 | NPU平均利用率 | SLO×1.5达标率 | 三种子入选请求总数 |",
             "|---|---|---:|---:|---:|"]
    for row in summary_rows:
        policy = "baseline" if row["strategy"] == "asu_baseline" else row["strategy"]
        lines.append(f"| {row['scenario']} | {NAMES[policy]} | {row['U_percent']:.4f}% | "
                     f"{row['slo_1p5_percent']:.4f}% | {row['sample_count']} |")
    lines += ["", "SLO分母是本请求8层纯计算时间。CDF取窗口内接纳请求，跟踪至完整运行结束；不含接纳前排队，不是实际首token时延。请求总数仅用于说明样本量，CDF和达标率仍先逐种子计算再等权平均。策略改变接纳时刻，所以窗口入选集合可能不同。", "",
              "near35的含义是平均欠载、允许局部过载，不能解释成逐盘全时欠载。各组负载标签沿用原输入组，不将原Baseline的过载时间比例直接套给OD。", "",
              "## 新增图", ""]
    for regime in REGIMES:
        lines += [f"### {LABELS[regime]}", "",
                  f"![三策略CDF](figures/{regime}_random_ttft_ratio_cdf_three_strategies.png)", "",
                  f"![OD逐盘带宽](figures/{regime}_od_baseline_per_ssu_seed7.png)", ""]
    lines += ["### semi：OD全部32卡计算时序", "",
              "![OD计算时序](figures/semi_od_baseline_32npu_timeline_seed7.png)", "",
              "逐盘图和时序图来自OD自己的seed7运行：参考需求为逐事件当前请求的V/C之和，实际供给为物理SSD服务的10ms均值。跨请求预取的读取计入供给；不重复计入参考需求。时序中的等待是数据未到齐造成的暴露等待，不等同于纯SSD排队。", "",
              "原NEW画像混排图显示的是各策略共用的输入顺序，继续保留即可。本目录不生成本轮排除的8卡、NQL、固定并发、等待分解或sensitivity20k带宽图。", "",
              "## 来源与复核", "",
              "- [统一页面汇总](summary.csv)、[JSON汇总](summary.json)",
              "- [逐种子指标](plot_per_seed_metrics.csv)、[逐请求样本](plot_request_samples.csv)、[CDF点](plot_cdf_points.csv)",
              "- [原曲线独立审计](original_source_audit.json)、[新图校验](render_checks.json)、[图表范围审查](SCOPE_REVIEW.md)",
              "- [重绘脚本](render_figures.py)：只读取完整结果，不运行仿真、不改写模板原图。", ""]
    (HERE / "FIGURES.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="audit archived curves without requiring OD runs")
    args = parser.parse_args()
    manifests, samples, arrays, refs, original_checks = load_original()
    if args.prepare_only:
        audit = dict(status="original_inputs_and_cdfs_verified", source_sha256=SOURCES,
                     original_cdf_checks=original_checks, no_simulation_started=True,
                     requires_completed_od_runs=9, allowed_figures=sorted(ALLOWED_FIGURES))
        (HERE / "original_source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(audit["original_cdf_checks"], ensure_ascii=False, indent=2))
        return
    raws, od_checks = load_od(manifests, samples, arrays, refs)
    setup_plotting()
    FIGURES.mkdir(exist_ok=True)
    outputs, cdf_points, bandwidth_checks = [], [], {}
    for regime in REGIMES:
        p, points = render_cdf(regime, arrays)
        outputs.append(p)
        cdf_points.extend(points)
        p, checks = render_disks(regime, raws[regime])
        outputs.append(p)
        bandwidth_checks[regime] = checks
    p, timeline_checks = render_timeline(raws["semi"])
    outputs.append(p)
    assert {p.name for p in outputs} == ALLOWED_FIGURES
    per_seed = []
    for key, ref in sorted(refs.items()):
        regime, policy, seed = key
        per_seed.append(dict(regime=regime, policy=policy, seed=seed, **ref,
                             slo_percent=100 * ref["passed"] / ref["count"], maximum_ratio=float(arrays[key][-1])))
    macro = []
    for regime in REGIMES:
        for policy in POLICIES:
            rows = [r for r in per_seed if r["regime"] == regime and r["policy"] == policy]
            assert len(rows) == 3
            macro.append(dict(regime=regime, policy=policy, seed_count=3,
                              mean_U_percent=float(np.mean([r["U_percent"] for r in rows])),
                              mean_slo_percent=float(np.mean([r["slo_percent"] for r in rows])),
                              sample_count=sum(r["count"] for r in rows),
                              maximum_ratio=max(r["maximum_ratio"] for r in rows)))
    for name, rows in (("plot_request_samples.csv", samples), ("plot_per_seed_metrics.csv", per_seed),
                       ("plot_macro_metrics.csv", macro), ("plot_cdf_points.csv", cdf_points)):
        write_csv(HERE / name, rows)
    summary_rows = [dict(scenario=r["regime"], strategy="asu_baseline" if r["policy"] == "baseline" else r["policy"],
                         U_percent=r["mean_U_percent"], slo_1p5_percent=r["mean_slo_percent"],
                         sample_count=r["sample_count"], seeds=json.dumps(list(SEEDS)),
                         cohort="admitted in warm [2000,4000) ms; followed through completion",
                         aggregation="equal mean of per-seed U and SLO", placement="original frozen stripe",
                         maximum_ratio=r["maximum_ratio"]) for r in macro]
    write_csv(HERE / "summary.csv", summary_rows)
    (HERE / "summary.json").write_text(json.dumps(dict(
        num_npu=32, num_ssu=3, ssu_capacity_GiB_s=40, n_layers=8, window_ms=[2000,4000],
        seeds=list(SEEDS), normalized_x="(prefill_completion-admission)/(8*per_layer_compute)",
        admission_clock_excludes_queue_before_admission=True,
        rows=[{**r, "seeds": list(SEEDS)} for r in summary_rows]), ensure_ascii=False, indent=2) + "\n")
    write_figure_readme(summary_rows)
    assert all(sha(ROOT / p) == digest for p, digest in SOURCES.items()), "read-only source changed"
    audit = dict(status="complete", no_simulation_started_by_renderer=True, source_files_unchanged=True,
                 source_sha256=SOURCES, original_cdf_checks=original_checks, completed_od_checks=od_checks,
                 bandwidth_checks=bandwidth_checks, timeline_checks=timeline_checks,
                 normalized_x="(prefill_completion-admission)/(8*per_layer_compute)", window_ms=[2000, 4000],
                 seeds=list(SEEDS), aggregation="equal mean of three seed ECDFs", full_tails_preserved=True,
                 old_curves_preserved=True, placement="exact original frozen stripe placements; no ring regeneration",
                 generated_png_sha256={p.name: sha(p) for p in outputs}, builder_sha256=sha(Path(__file__)),
                 excluded_figure_types_generated=False, visual_review="pending")
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(macro, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
