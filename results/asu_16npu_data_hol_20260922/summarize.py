#!/usr/bin/env python3
"""Summarize completed native runs only; optionally draw selected PNGs.

Examples (from any working directory):
  python /path/to/this/summarize.py
  python /path/to/this/summarize.py --cases I1_s1_r18_two_cohorts_64s_seed7 \
      --figures-dir /tmp/asu16_preview

Only runs/*/{asu_baseline,od_baseline,once} is read. Approximate phase_search/
and interp_search/ outputs are never used. No simulator or runner is imported.
Incomplete runs are pending, and unavailable long windows are never clipped.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fnmatch
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICIES = ("asu_baseline", "od_baseline", "once")
LABELS = {"asu_baseline": "ASU", "od_baseline": "OD", "once": "Once（流量分配）"}
POLICY_COLORS = {"asu_baseline": "#4b5563", "od_baseline": "#b95c15", "once": "#1478a9"}
ROLE_COLORS = {"A": "#2474b7", "B": "#ed951c", "stall": "#d63e45"}
EPS = 1e-9
FILES = ("metrics.json", "native_summary.json.gz", "manifest.json.gz", "config.json", "command.json")
STANDARD_WINDOWS = {"warm_2_4": (2000., 4000.), "steady_2_10": (2000., 10000.),
                    "late_4_10": (4000., 10000.), "long_2_60": (2000., 60000.),
                    "late_20_40": (20000., 40000.), "late_20_60": (20000., 60000.),
                    "late_40_60": (40000., 60000.)}


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tolerance=1e-7):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), (a, b)


def durations(starts, ends, start, end):
    return np.maximum(0., np.minimum(ends, end) - np.maximum(starts, start))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


class NativeRun:
    """One complete raw run in memory; arrays support many exact windows."""

    def __init__(self, path, current_source_cache):
        self.path = path
        self.metrics, native, manifest, self.config, command = (read(path / name) for name in FILES)
        self.case = path.parent.name
        self.policy = path.name
        self.n = int(native["num_npu"])
        self.disks = int(native["num_ssu"])
        self.end = float(native["makespan_ms"])
        assert self.n == 16 and self.config["num_npu"] == 16
        assert self.policy in POLICIES and native["n_layers"] == 8
        assert self.metrics["source_unchanged"] and all(native["invariants"].values())
        assert command["strategy"] == self.metrics["strategy"] == self.policy
        assert command["source_sha256"] == self.metrics["source_sha256"]
        self.fingerprint = manifest["input_fingerprint"]
        assert self.fingerprint == native["input_fingerprint"] == self.metrics["input_fingerprint"]
        self.manifest_sha = sha(path / "manifest.json.gz")
        assert self.manifest_sha == command["manifest_sha256"] == self.metrics["manifest_sha256"]
        # config_sha256 belongs to the ORIGINAL CLI input file. The archived
        # run/config.json is validated/expanded (e.g. cyclic counts), so hashing
        # that file or canonicalizing it would check the wrong artifact.
        args = command["argv"]
        configured_path = Path(args[args.index("--config") + 1])
        candidates = (ROOT / configured_path, HERE / "configs" / configured_path.name)
        source_config = next((p for p in candidates if p.is_file()), None)
        assert source_config is not None, f"Missing original config: {configured_path}"
        assert sha(source_config) == command["config_sha256"] == self.metrics["config_sha256"], str(source_config)
        self.source_config = source_config
        self.source_hashes = command["source_sha256"]
        self.source_mismatches = []
        for source, expected in self.source_hashes.items():
            if source not in current_source_cache:
                file = ROOT / source
                current_source_cache[source] = sha(file) if file.is_file() else None
            if current_source_cache[source] != expected:
                self.source_mismatches.append(source)
        self.metadata = {int(row["request_id"]): row["load"] for row in manifest["requests"]}
        self.requests = native["request_metrics"]
        assert len(self.requests) == len(self.metadata) == native["request_count"]
        assert {r["request_id"] for r in self.requests} == set(self.metadata)
        assert len(native["microbatch_metrics"]) == len(self.requests)
        expected_blocks = 8 * sum(len(manifest["placements"][r["placement_index"]][0]) for r in manifest["requests"])
        assert expected_blocks == native["submitted_blocks"] == native["completed_blocks"]
        self.first_finish = math.inf
        finish = np.zeros(self.n)
        self.req_start = np.array([r["admission_time_ms"] for r in self.requests])
        self.req_end = np.array([r["completion_time_ms"] for r in self.requests])
        self.req_npu = np.array([r["npu_id"] for r in self.requests], dtype=int)
        self.req_group = np.array([self.metadata[r["request_id"]]["profile_group"] for r in self.requests])
        self.req_pure = np.array([r["own_compute_ms"] for r in self.requests])
        self.req_elapsed = self.req_end - self.req_start
        self.req_pass = self.req_elapsed <= 1.5 * self.req_pure + EPS
        events = defaultdict(lambda: np.zeros(self.disks))
        events[0.]
        events[self.end]
        for r in self.requests:
            meta = self.metadata[r["request_id"]]
            close(r["own_compute_ms"], 8 * meta["per_layer_us"] / 1000)
            finish[r["npu_id"]] = max(finish[r["npu_id"]], r["completion_time_ms"])
            rate = np.array(meta["disk_gib"]) * 2**30 / 1e9 / (meta["per_layer_us"] / 1e6)
            assert len(rate) == self.disks and float(rate.min()) >= 0
            events[r["admission_time_ms"]] += rate
            events[r["completion_time_ms"]] -= rate
        self.finish = finish
        self.first_finish = float(finish.min())
        for npu in range(self.n):
            indices = np.where(self.req_npu == npu)[0]
            indices = indices[np.argsort(self.req_start[indices])]
            assert len(indices)
            close(self.req_start[indices[0]], 0.)
            assert np.allclose(self.req_end[indices[:-1]], self.req_start[indices[1:]], atol=1e-7, rtol=0)
        self.layers = []
        for batch in native["microbatch_metrics"]:
            assert len(batch["member_request_ids"]) == 1 and len(batch["layer_metrics"]) == 8
            rid = batch["member_request_ids"][0]
            meta = self.metadata[rid]
            previous = batch["admission_time_ms"]
            for layer in batch["layer_metrics"]:
                a, b = layer["compute_start_ms"], layer["compute_end_ms"]
                wait = layer["io_barrier_wait_ms"]
                close(b - a, meta["per_layer_us"] / 1000)
                close(a - previous, wait)
                self.layers.append((a, b, int(batch["npu_id"]), meta["profile_group"], max(0., wait), int(layer["layer"]), rid))
                previous = b
            close(previous, batch["completion_time_ms"])
        self.cstart = np.array([r[0] for r in self.layers])
        self.cend = np.array([r[1] for r in self.layers])
        self.cnpu = np.array([r[2] for r in self.layers], dtype=int)
        self.cgroup = np.array([r[3] for r in self.layers])
        self.wait_start = self.cstart - np.array([r[4] for r in self.layers])
        self.layer_index = np.array([r[5] for r in self.layers])
        points = sorted(events)
        self.dstart = np.array(points[:-1])
        self.dend = np.array(points[1:])
        rate = np.zeros(self.disks)
        values = []
        for t in points[:-1]:
            rate = rate + events[t]
            rate[np.abs(rate) < 1e-10] = 0.
            assert float(rate.min()) > -1e-7
            values.append(rate.copy())
        self.demand = np.array(values)
        self.artifact_sha = {name: sha(path / name) for name in FILES}
        self.window_cache = {}
        self.extended = self.metrics.get("extended_windows")
        if self.extended:
            close(self.extended["first_npu_finishes_ms"], self.first_finish)
            assert np.allclose(self.extended["per_npu_final_completion_ms"], finish, atol=1e-7, rtol=0)
        self.check_reported()

    def window(self, start, end):
        key = (float(start), float(end))
        if key in self.window_cache:
            return self.window_cache[key]
        assert start >= 0 and end > start
        row = dict(start_ms=start, end_ms=end, available=end <= self.end + EPS,
                   first_npu_finishes_ms=self.first_finish,
                   ends_before_first_npu_finishes=end <= self.first_finish + EPS)
        if not row["available"]:
            row["reason"] = "requested window extends beyond complete finite trajectory; not clipped"
            self.window_cache[key] = row
            return row
        duration = end - start
        cd = durations(self.cstart, self.cend, start, end)
        wd = durations(self.wait_start, self.cstart, start, end)
        ad = durations(self.req_start, self.req_end, start, end)
        compute = np.bincount(self.cnpu, weights=cd, minlength=self.n)
        active = np.bincount(self.req_npu, weights=ad, minlength=self.n)
        wait = np.bincount(self.cnpu, weights=wd, minlength=self.n)
        assert np.allclose(compute + wait, active, atol=1e-5, rtol=0)
        group_compute = {g: np.bincount(self.cnpu[self.cgroup == g], weights=cd[self.cgroup == g], minlength=self.n) for g in ("A", "B")}
        mixed = (group_compute["A"] > EPS) & (group_compute["B"] > EPS)
        cohort = (self.req_start >= start) & (self.req_start < end)
        count = int(cohort.sum())
        passed = int((cohort & self.req_pass).sum())
        dt = durations(self.dstart, self.dend, start, end)
        covered = dt > 0
        close(float(dt.sum()), duration)
        peaks = self.demand[covered].max(axis=0)
        over = self.demand > 40 + EPS
        at_cap = self.demand >= 40 - EPS
        row.update(duration_ms=duration, U_percent=100 * float(compute.sum()) / (self.n * duration),
                   SLO_1p5_percent=100 * passed / count if count else None,
                   admitted_count=count, passed_count=passed,
                   completed_after_window_count=int((cohort & (self.req_end > end)).sum()),
                   all_npus_active=bool((np.abs(active - duration) < 1e-6).all()),
                   mixed_npu_count=int(mixed.sum()), all_npus_compute_AB=bool(mixed.all()),
                   missing_A_npu_ids=",".join(str(i) for i in np.where(group_compute["A"] <= EPS)[0]),
                   missing_B_npu_ids=",".join(str(i) for i in np.where(group_compute["B"] <= EPS)[0]),
                   inactive_npu_ids=",".join(str(i) for i in np.where(np.abs(active - duration) >= 1e-6)[0]),
                   nominal_strict_under40=bool((peaks < 40 - EPS).all()),
                   nominal_peak_GB_s=float(peaks.max()),
                   any_ssu_over40_percent=100 * float(dt @ over.any(axis=1)) / duration,
                   any_ssu_at_or_above40_percent=100 * float(dt @ at_cap.any(axis=1)) / duration,
                   internal_stall_card_ms=float(wd[self.layer_index >= 1].sum()),
                   layer0_stall_card_ms=float(wd[self.layer_index == 0].sum()),
                   per_npu_U_percent=(100 * compute / duration).tolist())
        row["valid_all_active_mixed_underload_window"] = bool(row["ends_before_first_npu_finishes"] and row["all_npus_active"] and row["all_npus_compute_AB"] and row["nominal_strict_under40"])
        for group in ("A", "B"):
            mask = cohort & (self.req_group == group)
            n = int(mask.sum())
            p = int((mask & self.req_pass).sum())
            group_active_ms = float(ad[self.req_group == group].sum())
            group_compute_ms = float(group_compute[group].sum())
            group_wait_ms = float(wd[self.cgroup == group].sum())
            close(group_compute_ms + group_wait_ms, group_active_ms, tolerance=1e-5)
            row[f"{group}_compute_card_ms"] = group_compute_ms
            row[f"{group}_active_card_ms"] = group_active_ms
            row[f"{group}_wait_card_ms"] = group_wait_ms
            row[f"{group}_active_U_percent"] = 100 * group_compute_ms / group_active_ms if group_active_ms else None
            row[f"{group}_admitted_count"] = n
            row[f"{group}_passed_count"] = p
            row[f"{group}_SLO_1p5_percent"] = 100 * p / n if n else None
        for disk in range(self.disks):
            row[f"SSU{disk}_nominal_peak_GB_s"] = float(peaks[disk])
            row[f"SSU{disk}_nominal_mean_GB_s"] = float(dt @ self.demand[:, disk]) / duration
            row[f"SSU{disk}_over40_percent"] = 100 * float(dt @ over[:, disk]) / duration
        self.window_cache[key] = row
        return row

    def check_one_reported(self, saved):
        if "start_ms" not in saved or "end_ms" not in saved:
            return
        own = self.window(saved["start_ms"], saved["end_ms"])
        assert own["available"] == saved.get("available", True)
        if not own["available"]:
            return
        close(own["U_percent"], 100 * saved["fleet_utilization"])
        assert own["admitted_count"] == saved["slo_admitted"]["count"]
        assert own["passed_count"] == saved["slo_admitted"]["passed"]
        assert own["all_npus_active"] == saved["all_npus_active_whole_window"]
        if "all_npus_compute_both_groups" in saved:
            assert own["all_npus_compute_AB"] == saved["all_npus_compute_both_groups"]
        if "ends_before_first_npu_finishes" in saved:
            assert own["ends_before_first_npu_finishes"] == saved["ends_before_first_npu_finishes"]
        demand = saved["ordinary_demand"]
        close(own["nominal_peak_GB_s"], demand["max_single_ssu_gb_s"])
        close(own["any_ssu_over40_percent"], 100 * demand["any_ssu_overload_fraction"])
        assert own["nominal_strict_under40"] == demand["strictly_under_capacity"]

    def check_reported(self):
        self.check_one_reported(self.metrics["warm"])
        self.check_one_reported(self.metrics["full"])
        if self.extended:
            for saved in list(self.extended["windows"].values()) + self.extended["consecutive_2s"]:
                self.check_one_reported(saved)

    def base_row(self):
        return dict(case=self.case, strategy=self.policy, seed=self.config["seed"],
                    order_mode=self.config["mode"], npu=self.n, ssu=self.disks,
                    complete_native_run=True, has_extended_windows=bool(self.extended),
                    request_count=len(self.requests), makespan_ms=self.end,
                    input_fingerprint=self.fingerprint, manifest_sha256=self.manifest_sha,
                    source_unchanged_during_run=True,
                    current_source_files_match=not self.source_mismatches)

    def export(self, extra_windows=()):
        windows = dict(STANDARD_WINDOWS)
        horizon = float(self.config.get("horizon_ms", 0))
        if horizon > 2000 and (2000., horizon) not in windows.values():
            windows["planned_2_horizon"] = (2000., horizon)
        windows["full"] = (0., self.end)
        windows["before_first_finish"] = (0., self.first_finish)
        if self.first_finish > 2000:
            windows["post_start_before_first_finish"] = (2000., self.first_finish)
        for start, end in extra_windows:
            windows[f"extra_{start/1000:g}_{end/1000:g}"] = (start, end)
        if self.extended:
            for name, value in self.extended["windows"].items():
                windows.setdefault(name, (value["start_ms"], value["end_ms"]))
        rows = []
        for name, (a, b) in windows.items():
            rows.append(dict(self.base_row(), window=name, **self.window(a, b)))
        bins = []
        for start in range(0, math.ceil(self.end), 2000):
            end = min(start + 2000., self.end)
            if end > start:
                bins.append(dict(self.base_row(), window="consecutive_2s", full_2s_bin=end == start + 2000,
                                 **self.window(float(start), end)))
        full = self.window(0., self.end)
        close(sum(r["U_percent"] / 100 * self.n * r["duration_ms"] for r in bins),
              full["U_percent"] / 100 * self.n * self.end, tolerance=1e-5)
        assert sum(r["admitted_count"] for r in bins) == full["admitted_count"] == len(self.requests)
        assert sum(r["passed_count"] for r in bins) == full["passed_count"]
        evidence = dict(case=self.case, strategy=self.policy, status="passed_native_metrics",
                        artifact_sha256=self.artifact_sha, input_fingerprint=self.fingerprint,
                        manifest_sha256=self.manifest_sha,
                        original_config_path=str(self.source_config), original_config_sha256=sha(self.source_config),
                        source_sha256=self.source_hashes,
                        current_source_mismatches=self.source_mismatches,
                        first_npu_finishes_ms=self.first_finish,
                        checked_windows=len(self.window_cache), raw_intervals_and_slo_match_reported=True,
                        all_bins_compute_and_population_conserved=True)
        return rows, bins, evidence


def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in (Path.home() / ".fonts/msyh.ttc", Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if path.is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def utilization_plot(case, runs, rows, output, plt):
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    qualified = {}
    for run in runs:
        bins = [r for r in rows if r["case"] == case and r["strategy"] == run.policy
                and r["full_2s_bin"] and r["ends_before_first_npu_finishes"] and r["all_npus_active"]]
        assert bins
        qualified[run.policy] = [r["start_ms"] for r in bins]
        x = [(r["start_ms"] + r["end_ms"]) / 2000 for r in bins]
        y = [r["U_percent"] for r in bins]
        ax.plot(x, y, "o-", color=POLICY_COLORS[run.policy], ms=3, lw=1.8,
                label=f"{LABELS[run.policy]}：连续完整2秒窗")
        late = run.window(20000., 60000.)
        if late["available"] and late["ends_before_first_npu_finishes"] and late["all_npus_active"]:
            ax.hlines(late["U_percent"], 20, 60, color=POLICY_COLORS[run.policy], ls="--", lw=1.3,
                      label=f"{LABELS[run.policy]} [20,60)：{late['U_percent']:.2f}%")
    ax.set_ylim(0, 102)
    ax.set_xlabel("时间（秒；点位是2秒窗口中心）")
    ax.set_ylabel("NPU平均利用率（%）")
    ax.grid(alpha=.22)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("完整原生轨迹：随时间观察利用率变化", fontsize=15)
    fig.text(.5, .90, case, ha="center", fontsize=10)
    fig.text(.5, .055, "只绘首张卡排空前、全部16卡持续活跃的完整2秒窗；曲线停止表示输入开始排空，不是利用率变零。", ha="center", fontsize=9)
    fig.text(.5, .022, "是否长期低需同时看后期宽窗与变化趋势，不能只挑早期低点或把含排空的全程平均当稳态。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.19)
    path = output / f"{case}_utilization_2s.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, {"valid_bin_start_ms_by_strategy": qualified}


def timeline_plot(run, start, end, output, plt):
    from matplotlib.patches import Patch
    window = run.window(start, end)
    if not window["available"] or not window["ends_before_first_npu_finishes"]:
        raise ValueError(f"Timeline crosses first-card drain for {run.case}/{run.policy}; choose an earlier --timeline-ms")
    assert window["all_npus_active"]
    fig, ax = plt.subplots(figsize=(15, 8))
    for npu in range(run.n):
        for group in ("A", "B"):
            mask = (run.cnpu == npu) & (run.cgroup == group) & (run.cend > start) & (run.cstart < end)
            bars = [(max(a, start) / 1000, (min(b, end) - max(a, start)) / 1000)
                    for a, b in zip(run.cstart[mask], run.cend[mask])]
            ax.broken_barh(bars, (npu - .34, .68), facecolors=ROLE_COLORS[group], edgecolors="none")
        mask = (run.cnpu == npu) & (run.cstart > start) & (run.wait_start < end) & (run.cstart - run.wait_start > EPS)
        waits = [(max(a, start) / 1000, (min(b, end) - max(a, start)) / 1000)
                 for a, b in zip(run.wait_start[mask], run.cstart[mask]) if min(b, end) > max(a, start)]
        ax.broken_barh(waits, (npu - .34, .68), facecolors=ROLE_COLORS["stall"], edgecolors="none")
    ax.set_yticks(range(run.n), [f"NPU {i:02d}  {window['per_npu_U_percent'][i]:.1f}%" for i in range(run.n)], fontsize=9)
    ax.set_ylim(run.n - .5, -.5)
    ax.set_xlim(start / 1000, end / 1000)
    ax.set_xlabel("时间（秒）")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=ROLE_COLORS[g], label=l) for g, l in (("A", "长请求 A（计算）"), ("B", "短请求 B（计算）"), ("stall", "I/O等待（含首层）"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .90), ncol=3, frameon=False)
    zoom = "（局部放大）" if end - start < 1000 else "（首卡排空前全段）" if start == 0 and abs(end - run.first_finish) < EPS else ""
    fig.suptitle(f"{LABELS[run.policy]}：16卡计算与I/O等待{zoom}", fontsize=16, y=.99)
    fig.text(.56, .945, f"{run.case}  ·  [{start/1000:g},{end/1000:g})秒 U={window['U_percent']:.2f}%  ·  A+B实际计算覆盖={window['mixed_npu_count']}/16卡", ha="center", fontsize=10)
    fig.text(.56, .035, "红色是实际计算前的I/O阻塞区间，不是完整读取服务时间；利用率由所有计算块真实面积积分。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.13, right=.985, top=.84, bottom=.135)
    path = output / f"{run.case}_{run.policy}_timeline_{start/1000:g}_{end/1000:g}s.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, {"window_ms": [start, end], "U_percent": window["U_percent"], "all16active": True,
                  "mixed_cards": window["mixed_npu_count"], "ends_before_first_npu_finishes": True}


def cdf_plot(case, runs, start, end, full, output, plt):
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    data, statistics = {}, {}
    for run in runs:
        mask = np.ones(len(run.requests), dtype=bool) if full else (run.req_start >= start) & (run.req_start < end)
        if not full and end > run.end + EPS:
            raise ValueError(f"CDF cohort window unavailable: {run.case}/{run.policy}")
        elapsed, pure = run.req_elapsed[mask], run.req_pure[mask]
        if not len(elapsed):
            raise ValueError(f"Empty admission cohort: {run.case}/{run.policy}")
        ratio = elapsed / pure
        ratio[np.abs(elapsed - pure) <= EPS] = 1.
        ratio[(elapsed > 1.5 * pure) & (elapsed <= 1.5 * pure + EPS)] = 1.5
        ratio.sort()
        passed = int(np.sum(elapsed <= 1.5 * pure + EPS))
        close(np.searchsorted(ratio, 1.5, side="right"), passed)
        data[run.policy] = ratio
        statistics[run.policy] = dict(count=len(ratio), passed=passed, SLO_1p5_percent=100 * passed / len(ratio), max_ratio=float(ratio[-1]))
    xmax = max(float(v[-1]) for v in data.values())
    upper = max(1.8, xmax * 1.04)
    x = np.unique(np.concatenate([np.array([.9, 1., 1.5, upper])] + list(data.values())))
    curves = {}
    styles = ("--", "-", "-.")
    for run in runs:
        ratio = data[run.policy]
        y = np.searchsorted(ratio, x, side="right") / len(ratio) * 100
        curves[run.policy] = y
        row = statistics[run.policy]
        ax.step(x, y, where="post", color=POLICY_COLORS[run.policy], ls=styles[POLICIES.index(run.policy)], lw=2,
                label=f"{LABELS[run.policy]}：{row['SLO_1p5_percent']:.2f}%（{row['passed']}/{row['count']}）")
    if len(curves) > 1 and all(np.allclose(v, next(iter(curves.values())), atol=1e-10, rtol=0) for v in curves.values()):
        ax.text(.05, .18, "曲线重合", transform=ax.transAxes)
    ax.axvline(1.5, color="#718096", ls=":", lw=1.2)
    ax.set_xlim(.9, upper)
    ax.set_ylim(0, 102)
    ax.set_xlabel("归一化耗时 =（prefill完成 − 上卡）/ 自身8层纯计算")
    if xmax > 12:
        ax.set_xscale("log")
        ticks = [t for t in (1, 1.5, 2, 3, 5, 10, 20, 50, 100, 200, 500) if t <= upper]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_xlabel("归一化耗时（对数横轴）")
    else:
        ticks = sorted({float(t) for t in ax.get_xticks() if .9 <= t <= upper} | {1.5})
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.set_ylabel("累计请求比例（%）")
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", fontsize=10)
    cohort = "全输入，包括启动" if full else f"[{start/1000:g},{end/1000:g})秒上卡人口，跟踪至完成"
    ax.set_title("原生完整结果：上卡至prefill完成的归一化CDF", fontsize=14)
    fig.text(.52, .90, f"{case} · {cohort}", ha="center", fontsize=10)
    fig.text(.52, .055, "图例为SLO×1.5达标率及人数；不含上卡前排队，不是真实端到端首token时延。", ha="center", fontsize=9)
    fig.text(.52, .022, "相同输入下，策略推进速度不同，固定窗口内上卡的人口也可能不同；没有剔除窗口外完成的请求。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.19)
    tag = "full" if full else f"admitted_{start/1000:g}_{end/1000:g}s"
    path = output / f"{case}_ttft_cdf_{tag}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, {"cohort": cohort, "statistics": statistics, "normalized_boundary_tolerance_ms": EPS}


def demand_plot(case, runs, start, end, output, plt):
    """Exact current-request V/C events; this is not an SSD supply estimate."""
    assert len({r.disks for r in runs}) == 1
    disks = runs[0].disks
    fig, axes = plt.subplots(disks, 1, figsize=(13, max(5.4, 2.6 * disks)), squeeze=False)
    statistics = {}
    for run in runs:
        window = run.window(start, end)
        if not window["available"] or not window["ends_before_first_npu_finishes"]:
            raise ValueError(f"Demand window crosses first-card drain: {run.case}/{run.policy}")
        assert window["all_npus_active"]
        mask = (run.dend > start) & (run.dstart < end)
        edges = np.concatenate(([max(start, run.dstart[mask][0])], np.minimum(run.dend[mask], end))) / 1000
        per_disk = []
        for disk, ax in enumerate(axes[:, 0]):
            values = run.demand[mask, disk]
            peak = window[f"SSU{disk}_nominal_peak_GB_s"]
            average = window[f"SSU{disk}_nominal_mean_GB_s"]
            ax.stairs(values, edges, baseline=None, color=POLICY_COLORS[run.policy],
                      linestyle="--" if run.policy == "asu_baseline" else "-", linewidth=1.7,
                      label=f"{LABELS[run.policy]}：均值 {average:.3f}，峰值 {peak:.6f}")
            per_disk.append({"disk": disk, "mean_GB_s": average, "peak_GB_s": peak,
                             "over40_percent": window[f"SSU{disk}_over40_percent"]})
        statistics[run.policy] = {"per_disk": per_disk, "nominal_strict_under40": window["nominal_strict_under40"],
                                  "all16active": window["all_npus_active"]}
    for disk, ax in enumerate(axes[:, 0]):
        ax.axhline(40, color="#b5303b", linewidth=1.4, linestyle=":", label="SSU物理容量：40 GB/s")
        ax.set_xlim(start / 1000, end / 1000)
        max_rate = max(run.window(start, end)[f"SSU{disk}_nominal_peak_GB_s"] for run in runs)
        ax.set_ylim(0, max(42, max_rate * 1.06))
        ax.set_ylabel(f"SSU {disk} 名义需求（GB/s）")
        ax.grid(alpha=.18)
        ax.legend(loc="lower right", fontsize=9)
    axes[-1, 0].set_xlabel("时间（秒）")
    fig.suptitle("逐事件带宽核验：16卡当前请求在各盘的 V/C 总和", fontsize=14, y=.975)
    fig.text(.53, .905, f"{case}  ·  [{start/1000:g},{end/1000:g})秒  ·  十进制 GB/s", ha="center", fontsize=10)
    fig.text(.53, .055, "计算和等待中的当前请求均计入；按本实验约定不另加跨请求首层预取需求。此线不是实际SSD吞吐。", ha="center", fontsize=9)
    fig.text(.53, .022, "同一冻结输入的推进速度可以不同，因此两种策略的当前请求需求曲线也可以不同。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.985, top=.82, bottom=.19, hspace=.25)
    path = output / f"{case}_nominal_demand_{start/1000:g}_{end/1000:g}s.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, {"window_ms": [start, end], "units": "decimal GB/s", "statistics": statistics,
                  "exact_event_segments": True, "physical_supply_plotted": False}


def extract_hol_examples(run, start, end):
    """Prove a FIFO predecessor-work lower bound without inventing SSD traces.

    This intentionally only supports the audited one-disk, unlimited-depth,
    single-path ASU setup. Submission endpoints follow the native 0.1-us issue
    rule, after checking that each card finishes issuing before its next I/O
    activation. io_ready is HBM ready, never labelled as SSD completion.
    """
    assert run.policy == "asu_baseline" and run.disks == 1
    assert run.metrics["queue_depth_limit"] is None
    assert {r["path"] for r in run.metrics["observed_paths"]} == {0}
    native = read(run.path / "native_summary.json.gz")
    manifest = read(run.path / "manifest.json.gz")
    assert native["client_submit_batch_size"] == 1
    interval_ms = native["client_issue_interval_us"] / 1000
    close(interval_ms, .0001)
    close(run.metrics["disk_bw_gib_s"] * 2**30 / 1e9, 40.)
    metadata = {r["request_id"]: r for r in manifest["requests"]}
    io = []
    for batch in native["microbatch_metrics"]:
        rid = batch["member_request_ids"][0]
        request = metadata[rid]
        blocks = manifest["placements"][request["placement_index"]][0]
        assert all(disk == 0 for disk, _ in blocks)
        volume_mb = sum(v for _, v in blocks) * 2**30 / 1e6
        close(volume_mb, request["load"]["disk_gib"][0] * 2**30 / 1e6)
        for layer in batch["layer_metrics"]:
            row = dict(npu_id=batch["npu_id"], request_id=rid,
                       group=request["load"]["profile_group"], V_MB=volume_mb,
                       block_count=len(blocks), **layer)
            row["last_enqueue_derived_ms"] = layer["io_start_time_ms"] + (len(blocks) - 1) * interval_ms
            io.append(row)
    issue_slack = math.inf
    for npu in range(run.n):
        ordered = sorted((x for x in io if x["npu_id"] == npu), key=lambda x: x["io_start_time_ms"])
        for previous, following in zip(ordered, ordered[1:]):
            slack = following["io_start_time_ms"] - previous["last_enqueue_derived_ms"] - interval_ms
            issue_slack = min(issue_slack, slack)
            assert slack > 1e-6, "Cannot derive enqueue times when per-card submission streams overlap"
    a_layers = [x for x in io if x["group"] == "A" and x["layer"] >= 1]
    b_layers = [x for x in io if x["group"] == "B" and x["layer"] >= 1 and
                start <= x["io_start_time_ms"] and x["compute_start_ms"] < end and
                x["io_barrier_wait_ms"] > EPS]
    candidates = []
    for b in b_layers:
        t = b["io_start_time_ms"]
        prior = [a for a in a_layers if a["last_enqueue_derived_ms"] < t - 1e-6 and a["io_ready_time_ms"] > t]
        if not prior:
            continue
        earliest = min(a["io_start_time_ms"] for a in prior)
        total_mb = sum(a["V_MB"] for a in prior)
        maximum_served_mb = 40 * (t - earliest)  # 40 GB/s = 40 MB/ms.
        residual_lower_mb = max(0., total_mb - maximum_served_mb)
        predecessor_lower_ms = residual_lower_mb / 40
        own_lower_ms = b["V_MB"] / 40
        c = b["compute_duration_ms"]
        ready = b["io_ready_time_ms"]
        r = ready - t
        period = b["compute_start_ms"] - t
        stall_lower_ms = max(0., predecessor_lower_ms + own_lower_ms - c)
        if stall_lower_ms <= EPS:
            continue
        close(period, c + b["io_barrier_wait_ms"])
        close(b["io_barrier_wait_ms"], max(0., r - c))
        assert r + 1e-6 >= predecessor_lower_ms + own_lower_ms
        candidates.append(dict(
            npu_id=b["npu_id"], request_id=b["request_id"], receiving_layer=b["layer"],
            previous_layer=b["layer"] - 1, V_MB=b["V_MB"], C_ms=c,
            io_start_ms=t, previous_compute_end_ms=t + c, io_ready_HBM_ms=ready,
            next_compute_start_ms=b["compute_start_ms"], next_compute_end_ms=b["compute_end_ms"],
            R_to_HBM_ms=r, exposed_stall_ms=b["io_barrier_wait_ms"], full_period_ms=period,
            B_nominal_GB_s=b["V_MB"] / c,
            b_layer_cycle_average_GB_s=b["V_MB"] / period,
            b_over_B=c / period, cycle_U_percent=100 * c / period,
            preceding_A_layers=sorted(prior, key=lambda a: a["io_start_time_ms"]),
            preceding_A_count=len(prior), earliest_A_issue_ms=earliest,
            latest_A_last_enqueue_ms=max(a["last_enqueue_derived_ms"] for a in prior),
            A_total_read_MB=total_mb, before_B_elapsed_ms=t - earliest,
            maximum_total_SSD_service_before_B_MB=maximum_served_mb,
            residual_A_bytes_lower_bound_MB=residual_lower_mb,
            predecessor_service_lower_bound_ms=predecessor_lower_ms,
            B_own_service_lower_bound_ms=own_lower_ms,
            unavoidable_stall_lower_bound_ms=stall_lower_ms,
            all_predecessors_internal_layers=True))
    candidates.sort(key=lambda x: x["unavoidable_stall_lower_bound_ms"], reverse=True)
    return dict(case=run.case, strategy=run.policy, search_window_ms=[start, end],
                status="proved_examples" if candidates else "no_positive_lower_bound_found",
                source_artifact_sha256=run.artifact_sha,
                min_per_card_issue_clearance_ms=issue_slack,
                client_issue_interval_ms=interval_ms, examples=candidates[:5],
                assumptions={"one_SSD": True, "capacity_decimal_GB_s": 40,
                             "one_FIFO_path0": True, "unlimited_submission_depth": True,
                             "no_per_card_submission_overlap": True,
                             "all_A_predecessors_are_internal_layers": True},
                definitions={"R": "Layer activation to all data HBM ready, including queueing and link; not SSD service time",
                             "last_enqueue": "Derived from activation plus (block_count-1)*native issue interval; floating error tolerance 1e-6ms",
                             "proof": "Earlier fully enqueued A bytes minus maximum SSD capacity since their earliest activation; all residual A bytes precede every B block in FIFO",
                             "lower_bound": "Conservative capacity/FIFO deduction, not a measured per-block service trace",
                             "b": "Next layer V divided by one same-request compute-start to compute-start period; not instantaneous throughput"})


def hol_plot(evidence, output, plt):
    if not evidence["examples"]:
        return None, None
    x = evidence["examples"][0]
    c, r, stall = x["C_ms"], x["R_to_HBM_ms"], x["exposed_stall_ms"]
    fig, ax = plt.subplots(figsize=(13.5, 7.3))
    ax.broken_barh([(0, c), (r, c)], (.32, .40), facecolors=ROLE_COLORS["B"])
    ax.broken_barh([(c, stall)], (.32, .40), facecolors=ROLE_COLORS["stall"])
    ax.text(c / 2, .52, f"前层计算\nC={c:.3f} ms", ha="center", va="center", fontsize=11)
    ax.text(c + stall / 2, .52, f"I/O等待\nI={stall:.3f} ms", ha="center", va="center", fontsize=11, color="white")
    ax.text(r + c / 2, .52, "本层开始计算", ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(r, .11), xytext=(0, .11), arrowprops={"arrowstyle": "<->", "color": "#3d4550", "lw": 1.5})
    ax.text(r / 2, .15, f"读取发起 → HBM数据就绪：R={r:.3f} ms（包含排队）", ha="center", fontsize=10)
    ax.axvline(c, color="#666", linestyle=":", linewidth=1)
    ax.axvline(r, color="#666", linestyle=":", linewidth=1)
    ax.set_xlim(-.5, r + c + 1)
    ax.set_ylim(0, .9)
    ax.set_yticks([])
    ax.set_xlabel("相对此层读取发起的时间（毫秒）")
    ax.grid(axis="x", alpha=.15)
    fig.suptitle("一个短B内部层：名义欠载，为什么仍然要等？", fontsize=17, y=.97)
    fig.text(.52, .905, f"{evidence['case']} · ASU / NPU {x['npu_id']} / 请求 {x['request_id']} / 层{x['previous_layer']}→{x['receiving_layer']}（从0编号）", ha="center", fontsize=10)
    fig.text(.52, .863, f"原生读取发起时间 {x['io_start_ms']:.6f} ms；同一请求内的完整层周期，无跨请求首层示例。", ha="center", fontsize=10)
    proof = (
        f"它前面已有 {x['preceding_A_count']} 个长A的内部层：全部块比B更早进入同一个FIFO。\n"
        f"总读取 {x['A_total_read_MB']:.3f} MB；B发起前仅过 {x['before_B_elapsed_ms']:.3f} ms，盘至多读掉 {x['maximum_total_SSD_service_before_B_MB']:.3f} MB。\n"
        f"因此前方至少剩 {x['residual_A_bytes_lower_bound_MB']:.3f} MB，先服务它们至少需 {x['predecessor_service_lower_bound_ms']:.3f} ms > B可遮住读取的 {c:.3f} ms。\n"
        f"再计B自身读取，必需等待下界 {x['unavoidable_stall_lower_bound_ms']:.3f} ms；实测等待 {stall:.3f} ms。")
    fig.text(.08, .255, proof, ha="left", va="top", fontsize=10, linespacing=1.7)
    fig.text(.08, .087, f"B=V/C={x['B_nominal_GB_s']:.3f} GB/s；b=V/(C+I)={x['b_layer_cycle_average_GB_s']:.3f} GB/s；b/B=C/(C+I)={x['cycle_U_percent']:.2f}%。", fontsize=10)
    fig.text(.08, .038, "上述排前读取量是保守下界，不是伪造的SSD服务条；R的终点是HBM就绪，不能称为SSD完成。带宽采用十进制GB/s。", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.77, bottom=.39)
    path = output / f"{evidence['case']}_asu_B_internal_HOL.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, {k: x[k] for k in ("npu_id", "request_id", "receiving_layer", "C_ms", "R_to_HBM_ms", "exposed_stall_ms", "unavoidable_stall_lower_bound_ms", "cycle_U_percent")}


def final_study_tables(rows, output):
    """Explicit final-pool comparison; earlier similarly named pools excluded."""
    main = "full_input_aligned_A10B45"
    random_cases = [f"full_input_pool_random_inputseed{seed}" for seed in (7, 17, 27)]
    sensitive = "full_input_aligned_submitseed17"
    source = read(HERE / "configs" / f"{main}.json")
    original = source["per_npu_sequences"]
    pool_checks = []
    for case in random_cases:
        cfg = read(HERE / "configs" / f"{case}.json")
        assert cfg["profiles_catalog"] == source["profiles_catalog"]
        assert cfg["seed"] == source["seed"] == 7
        assert all(Counter(a) == Counter(b) for a, b in zip(original, cfg["per_npu_sequences"]))
        assert len(cfg["per_npu_sequences"]) == len(original) == 16
        pool_checks.append(dict(case=case, same_profiles_and_per_card_multiset=True,
                                simulator_submit_seed=7, config_sha256=sha(HERE / "configs" / f"{case}.json")))
    cfg = read(HERE / "configs" / f"{sensitive}.json")
    assert cfg["per_npu_sequences"] == original and cfg["profiles_catalog"] == source["profiles_catalog"]
    assert cfg["seed"] == 17
    assert {k: v for k, v in cfg.items() if k not in ("name", "seed")} == {k: v for k, v in source.items() if k not in ("name", "seed")}
    full_population = {case: [r for r in rows if r["case"] == case and r["window"] == "full"]
                       for case in [main, sensitive]}
    if full_population[main] and full_population[sensitive]:
        assert len({r["input_fingerprint"] for v in full_population.values() for r in v}) == 1
    windows = sorted({r["window"] for r in rows if r["case"] in [main, sensitive, *random_cases]})
    comparisons = []
    for group, cases, policies in [("aligned", [main], ("asu_baseline", "once")),
                                   ("same_pool_random", random_cases, ("asu_baseline", "once")),
                                   ("submitseed17_sensitivity", [sensitive], ("asu_baseline",))]:
        for policy in policies:
            for window in windows:
                selected = [r for r in rows if r["case"] in cases and r["strategy"] == policy and r["window"] == window]
                complete = len(selected) == len(cases)
                available = complete and all(r["available"] for r in selected)
                out = dict(group=group, strategy=policy, window=window,
                           expected_case_count=len(cases), completed_case_count=len(selected),
                           group_complete=complete, all_windows_available=available,
                           cases=json.dumps(cases),
                           weighting="equal mean over input shuffle seeds" if group == "same_pool_random" else "one native run")
                if available:
                    out.update(start_ms=selected[0]["start_ms"],
                               end_ms_min=min(r["end_ms"] for r in selected),
                               end_ms_max=max(r["end_ms"] for r in selected),
                               U_percent_mean=float(np.mean([r["U_percent"] for r in selected])),
                               U_percent_min=min(r["U_percent"] for r in selected),
                               U_percent_max=max(r["U_percent"] for r in selected),
                               SLO_1p5_percent_mean=float(np.mean([r["SLO_1p5_percent"] for r in selected])) if all(r["SLO_1p5_percent"] is not None for r in selected) else None,
                               admitted_count_total=sum(r["admitted_count"] for r in selected),
                               passed_count_total=sum(r["passed_count"] for r in selected),
                               every_case_all16active=all(r["all_npus_active"] for r in selected),
                               every_case_all16computeAB=all(r["all_npus_compute_AB"] for r in selected),
                               every_case_before_first_finish=all(r["ends_before_first_npu_finishes"] for r in selected),
                               every_case_nominal_under40=all(r["nominal_strict_under40"] for r in selected),
                               nominal_peak_GB_s_max=max(r["nominal_peak_GB_s"] for r in selected))
                comparisons.append(out)
    evidence = dict(main_case=main, source_config_sha256=sha(HERE / "configs" / f"{main}.json"),
                    random_pool_checks=pool_checks,
                    sensitivity=dict(case=sensitive, same_order_and_profile_parameters=True,
                                     only_simulator_submit_seed_changes=True, simulator_submit_seed=17),
                    approximate_search_used=False, rows=comparisons,
                    status="complete" if comparisons and all(r["group_complete"] for r in comparisons) else "pending_native_runs",
                    notes=["Unavailable windows are not filled with zero or clipped.",
                           "Random SLO is the equal seed mean, not the ratio of pooled passing counts.",
                           "Full and before-first-finish intervals use strategy-specific endpoints, explicitly reported."])
    write_csv(output / "final_comparison.csv", comparisons)
    (output / "final_comparison.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    return {"status": evidence["status"], "csv_sha256": sha(output / "final_comparison.csv"),
            "json_sha256": sha(output / "final_comparison.json"), "rows": len(comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=[], help="Case names or shell-style patterns to plot; no plots by default")
    parser.add_argument("--strategies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--timeline-ms", nargs=2, type=float, default=[2000., 4000.])
    parser.add_argument("--full-timeline", action="store_true", help="Also draw all16 cards from time zero to each strategy's own first-card completion")
    parser.add_argument("--zoom-ms", nargs=2, type=float, help="Optional additional local timeline in milliseconds")
    parser.add_argument("--demand-ms", nargs=2, type=float, help="Optional exact nominal-demand window in milliseconds")
    parser.add_argument("--cdf-window-ms", nargs=2, type=float, default=[2000., 4000.])
    parser.add_argument("--additional-cdf-window-ms", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--hol-window-ms", nargs=2, type=float, help="Derive one-disk ASU FIFO HOL examples from this window; save JSON and one PNG")
    parser.add_argument("--cdf-full", action="store_true")
    parser.add_argument("--include-full-cdf", action="store_true", help="Also draw the full admitted population without replacing the selected window CDF")
    parser.add_argument("--require-complete", action="store_true", help="Return 2 if any discovered run directory is pending")
    parser.add_argument("--extra-window-ms", nargs=2, type=float, action="append", default=[], help="Additional exact statistics windows; unavailable trajectories are not clipped")
    parser.add_argument("--final-study", action="store_true", help="Also summarize only the frozen full-input case, its exact-pool random controls and submit-seed sensitivity")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, bins, evidence, pending, failed = [], [], [], [], []
    paths, source_cache = {}, {}
    for path in sorted((HERE / "runs").glob("*/*")):
        if not path.is_dir() or path.name not in POLICIES:
            continue
        missing = [name for name in FILES if not (path / name).is_file()]
        if missing:
            pending.append(dict(case=path.parent.name, strategy=path.name, missing=missing))
            continue
        try:
            run = NativeRun(path, source_cache)
            a, b, check = run.export(args.extra_window_ms)
            rows.extend(a)
            bins.extend(b)
            evidence.append(check)
            paths[(run.case, run.policy)] = path
            del run
        except Exception as error:
            failed.append(dict(case=path.parent.name, strategy=path.name,
                               error_type=type(error).__name__, detail=str(error)))
    pairs = []
    for case in sorted({key[0] for key in paths}):
        part = [e for e in evidence if e["case"] == case]
        fingerprints = {e["input_fingerprint"] for e in part}
        manifests = {e["manifest_sha256"] for e in part}
        # Frozen event-engine snapshots must match; historical runner-only
        # analysis extensions are reported separately, never silently ignored.
        cores = [{p: h for p, h in e["source_sha256"].items() if "/frozen_source/" in p or p.endswith("od_policy_snapshot.py")} for e in part]
        okay = len(fingerprints) == len(manifests) == 1 and all(c == cores[0] for c in cores)
        pairs.append(dict(case=case, completed_strategies=[e["strategy"] for e in part],
                          paired_comparison_available=len(part) > 1,
                          identical_input_manifest=okay, frozen_core_compatible=all(c == cores[0] for c in cores)))
        if not okay:
            failed.append(dict(case=case, error_type="PairedInputOrCoreMismatch"))
    # Lists are JSON strings inside CSV rather than implementation reprs.
    for collection in (rows, bins):
        for row in collection:
            if "per_npu_U_percent" in row:
                row["per_npu_U_percent"] = json.dumps(row["per_npu_U_percent"])
    write_csv(output / "all_native_runs.csv", rows)
    write_csv(output / "all_native_2s_windows.csv", bins)
    final_tables = final_study_tables(rows, output) if args.final_study and not failed else None
    selected = sorted({case for case, policy in paths if policy in args.strategies and any(fnmatch.fnmatchcase(case, pattern) for pattern in args.cases)})
    if args.cases and not selected:
        failed.append(dict(error_type="NoCompletedSelectedCases", patterns=args.cases))
    figures = []
    if selected and not failed:
        plt = setup_plotting()
        folder = (args.figures_dir or output / "figures").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        for case in selected:
            runs = [NativeRun(paths[(case, policy)], source_cache) for policy in args.strategies if (case, policy) in paths]
            path, info = utilization_plot(case, runs, bins, folder, plt)
            figures.append(dict(path=str(path), sha256=sha(path), kind="consecutive_2s_U", details=info))
            for run in runs:
                path, info = timeline_plot(run, *args.timeline_ms, folder, plt)
                figures.append(dict(path=str(path), sha256=sha(path), kind="compute_stall_timeline", details=info))
                if args.zoom_ms:
                    path, info = timeline_plot(run, *args.zoom_ms, folder, plt)
                    figures.append(dict(path=str(path), sha256=sha(path), kind="local_compute_stall_timeline", details=info))
                if args.full_timeline:
                    path, info = timeline_plot(run, 0., run.first_finish, folder, plt)
                    figures.append(dict(path=str(path), sha256=sha(path), kind="full_active_compute_stall_timeline", details=info))
            path, info = cdf_plot(case, runs, *args.cdf_window_ms, args.cdf_full, folder, plt)
            figures.append(dict(path=str(path), sha256=sha(path), kind="normalized_TTFT_CDF", details=info))
            for window in args.additional_cdf_window_ms:
                path, info = cdf_plot(case, runs, *window, False, folder, plt)
                figures.append(dict(path=str(path), sha256=sha(path), kind="normalized_TTFT_CDF", details=info))
            if args.include_full_cdf and not args.cdf_full:
                path, info = cdf_plot(case, runs, 0., 0., True, folder, plt)
                figures.append(dict(path=str(path), sha256=sha(path), kind="full_population_normalized_TTFT_CDF", details=info))
            if args.demand_ms:
                path, info = demand_plot(case, runs, *args.demand_ms, folder, plt)
                figures.append(dict(path=str(path), sha256=sha(path), kind="exact_nominal_demand", details=info))
            if args.hol_window_ms:
                asu = next((r for r in runs if r.policy == "asu_baseline"), None)
                if asu is None:
                    raise ValueError("--hol-window-ms requires a completed selected ASU run")
                hol = extract_hol_examples(asu, *args.hol_window_ms)
                hol_path = output / f"{case}_hol_examples.json"
                hol_path.write_text(json.dumps(hol, ensure_ascii=False, indent=2) + "\n")
                path, info = hol_plot(hol, folder, plt)
                if path:
                    figures.append(dict(path=str(path), sha256=sha(path), kind="internal_B_FIFO_proof", details=info,
                                        evidence_json=str(hol_path), evidence_sha256=sha(hol_path)))
            del runs
    result = dict(status="failed" if failed else "completed_native_snapshot_with_pending" if pending else "completed_native_snapshot",
                  native_runs=len(evidence), pending=pending, failures=failed, pairs=pairs,
                  approximate_search_outputs_used=False, cases=evidence, figures=figures,
                  final_study_tables=final_tables,
                  visual_review="pending" if figures else "not_requested", script_sha256=sha(Path(__file__)),
                  definitions=dict(U="Exact compute overlap divided by NPU count and wall time",
                    SLO="Admission cohort followed to final completion; elapsed<=1.5*own_compute+1e-9ms",
                    primary_window="[2,4) seconds; additional long windows never clipped",
                    tail="A window after first NPU finish is explicitly flagged, never called sustained all-card work",
                    nominal_demand="Current admitted request per-disk V/C, decimal GB/s; extra next-request L0 excluded",
                    source_history="Current source mismatch is recorded separately from source_unchanged during the archived run",
                    scope="Only native runs/*/*; phase_search and interp_search excluded; no final scientific conclusion inferred"))
    (output / "summary_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], native_runs=len(evidence), pending=len(pending), failures=len(failed), window_rows=len(rows), figures=len(figures))))
    if failed:
        print(json.dumps(failed, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.require_complete and pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
