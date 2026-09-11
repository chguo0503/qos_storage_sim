#!/usr/bin/env python3
"""Independently audit frozen direct-data inputs and Baseline/Once outputs.

Writes only this directory. --partial writes preview_* and no five-seed means.
Nominal overload, old warmup and old mixed-role failures are diagnostics, never
seed exclusion criteria. Raw inputs and simulator sources are read-only.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(ROOT), str(STUDY)]
SEEDS = (7, 19, 43, 67, 101)
ARMS = ("raw84", "underload33")
STRATEGIES = ("baseline", "once")
START, END = 2000.0, 4000.0
IO_GIB = 176 * 1024 / 2**30


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def catalog():
    """Recompute selection and C/V without importing construction logic."""
    source = ast.literal_eval((ROOT / "data").read_text())
    rows = []
    for (seq, nql), (bw, c_us, ttft, raw_v) in sorted(source.items()):
        blocks = math.ceil((seq * 1024 - nql) / 128)
        physical = blocks * IO_GIB
        c = c_us / 1000
        worst = 32 * math.ceil(blocks / 6) * IO_GIB * 1000 / c
        rows.append({"seq_len_k": seq, "nql": nql, "source_bw_gib_s": bw,
            "source_C_us": c_us, "layer_C_ms": c, "request_8C_ms": 8*c,
            "source_ttft_78layer_ms": ttft, "raw_layer_V_gib": raw_v,
            "physical_layer_V_gib": physical, "padding_gib_per_layer": physical-raw_v,
            "blocks_per_layer": blocks, "physical_V_over_C_gib_s": physical*1000/c,
            "max_32_same_profile_ssu_demand_gib_s": worst, "underload33_eligible": worst < 40,
            "category": ("S" if seq <= 80 else "L") + ("S" if nql < 512 else "L"),
            "role": "short" if seq <= 80 else "long",
            "post_admission_link_lower_bound_ms": c + 7*physical/50*1000,
            "slo_1_5_threshold_ms": 12*c,
            "link_alone_forced_slo_failure": c + 7*physical/50*1000 > 12*c + 1e-9})
    assert len(rows) == 84 and sum(r["underload33_eligible"] for r in rows) == 33
    return rows


def audit_input(item, plan, all_rows, auditor):
    path = HERE / "inputs" / (item["label"] + ".json.gz")
    requests, meta = auditor.load_manifest(path)
    rows = [r for r in all_rows if item["arm"] == "raw84" or r["underload33_eligible"]]
    lookup = {(r["seq_len_k"], r["nql"]): r for r in rows}
    budget = 4000 + 2*max(r["request_8C_ms"] for r in all_rows)
    checks = {
        "manifest_file_hash_matches_plan": auditor.sha(path) == item["manifest_sha256"],
        "metadata_label_seed_arm": (meta["label"], meta["seed"], meta["arm"]) == (item["label"], item["seed"], item["arm"]),
        "metadata_declares_original_C": meta["compute_scale_actual"] == 1.0,
        "request_count_matches_plan": len(requests) == item["request_count"],
        "source_data_hash_matches_plan": meta["source_data_sha256"] == plan["data_sha256"] == auditor.sha(ROOT / "data"),
        "construction_hash_matches_plan": meta["construction_runner_sha256"] == plan["runner_sha256"] == auditor.sha(HERE / "run_raw.py"),
        "budget_recomputed": meta["minimum_pure_compute_budget_ms"] == plan["minimum_lane_pure_compute_budget_ms"] == budget,
        "iid_prefix_and_stopping_rule": True,
        "identity_arrival_category_role_C_rawV_physicalV_padding": True,
        "eight_exact_176kib_striped_layers": True,
        "per_lane_metadata_counts_and_indices": True,
    }
    placement_cache = {}
    for n in range(32):
        lane = sorted((r for r in requests if r.npu_id == n), key=lambda r: r.request_id)
        rng = random.Random(item["seed"] + 100003*n)
        expected_indices = [rng.randrange(len(rows)) for _ in lane]
        cs = []
        for i, (request, index) in enumerate(zip(lane, expected_indices)):
            p, load = rows[index], request.load
            k = (load["seq_len_k"], load["nql"])
            checks["iid_prefix_and_stopping_rule"] &= k == (p["seq_len_k"], p["nql"])
            if k not in lookup:
                raise ValueError("profile is not in predeclared arm catalog")
            p = lookup[k]
            checks["identity_arrival_category_role_C_rawV_physicalV_padding"] &= (
                request.request_id == n*1000000+i and request.arrival_time_ms == 0
                and load["request_id"] == request.request_id and load["npu_id"] == n
                and load["generation"] == i and load["arrival_ms"] == load["arrival_time"] == 0
                and load["category"] == p["category"] and load["role"] == p["role"]
                and load["constructed_profile"] is False
                and load["per_layer_us"] == load["original_compute_us"] == p["source_C_us"]
                and load["per_layer_kv_gb"] == load["source_per_layer_kv_gib"] == p["raw_layer_V_gib"]
                and load["physical_per_layer_kv_gib"] == p["physical_layer_V_gib"]
                and math.isclose(load["padding_gib_per_layer"], p["padding_gib_per_layer"], abs_tol=1e-12)
                and load["source_ttft_ms"] == p["source_ttft_78layer_ms"]
                and load["required_bw_input_gbps"] == p["source_bw_gib_s"])
            placement_key = (n//4, p["blocks_per_layer"])
            if placement_key not in placement_cache:
                placement_cache[placement_key] = tuple(((b+n//4) % 6, IO_GIB) for b in range(p["blocks_per_layer"]))
            expected = placement_cache[placement_key]
            checks["eight_exact_176kib_striped_layers"] &= len(request.placement) == 8 and all(layer == expected for layer in request.placement)
            cs.append(p["request_8C_ms"])
        checks["iid_prefix_and_stopping_rule"] &= math.fsum(cs) >= budget and math.fsum(cs[:-1]) < budget
        lm = meta["per_npu"][n]
        checks["per_lane_metadata_counts_and_indices"] &= (lm["npu_id"] == n and lm["request_count"] == len(lane)
            and lm["rng_seed"] == item["seed"]+100003*n and lm["catalog_indices"] == expected_indices
            and math.isclose(lm["pure_compute_ms"], math.fsum(cs), abs_tol=1e-9)
            and math.isclose(lm["pure_compute_before_last_request_ms"], math.fsum(cs[:-1]), abs_tol=1e-9))
    # Reuse the already loaded immutable manifest, avoiding a second decompression.
    original_loader = auditor.load_manifest
    auditor.load_manifest = lambda _: (requests, meta)
    try:
        info = auditor.load_input(path, HERE)
    finally:
        auditor.load_manifest = original_loader
    checks["input_fingerprint_matches_manifest_and_plan"] = info["input_fingerprint"] == meta["input_fingerprint"] == item["input_fingerprint"]
    info["independent_input_audit"] = {"passed": all(checks.values()), "checks": checks}
    initial = [info["_requests"][n*1000000] for n in range(32)]
    initial_rates = [math.fsum(r["rate_by_ssu_gib_s"][s] for r in initial) for s in range(6)]
    info["first_requests_nominal_demand"] = {"per_ssu_gib_s": initial_rates, "max_ssu_gib_s": max(initial_rates),
        "definition": "Simultaneously admitted first request on each NPU, actual striped physical V/C. No future L0 demand is added."}
    pure = math.fsum(8*r["layer_compute_ms"] for r in info["_requests"].values())
    info["complete_input_composition"] = {cat: {
        "request_count": sum(r["category"] == cat for r in info["_requests"].values()),
        "request_fraction": sum(r["category"] == cat for r in info["_requests"].values()) / info["request_count"],
        "pure_compute_ms": math.fsum(8*r["layer_compute_ms"] for r in info["_requests"].values() if r["category"] == cat),
        "pure_compute_fraction": math.fsum(8*r["layer_compute_ms"] for r in info["_requests"].values() if r["category"] == cat) / pure,
    } for cat in ("SS", "SL", "LS", "LL")}
    return info


def link_lower_bound(raw, info, profile_map):
    rows = raw["summary"]["request_metrics"]
    out = {"definition": "completion-admission >= 7*physical_layer_V/50GiBps + one final layer C; L0 may already be prefetched before admission.",
           "all_requests_obey_bound": True, "minimum_slack_ms": float("inf"), "cohorts": {}}
    for r in rows:
        inp = info["_requests"][r["request_id"]]
        p = profile_map[(inp["seq_len_k"], inp["nql"])]
        slack = r["completion_time_ms"] - r["admission_time_ms"] - p["post_admission_link_lower_bound_ms"]
        out["minimum_slack_ms"] = min(out["minimum_slack_ms"], slack)
        out["all_requests_obey_bound"] &= slack >= -1e-7
    for name, sample in (("all_requests", rows), ("window_admissions", [r for r in rows if START <= r["admission_time_ms"] < END])):
        forced = [r for r in sample if profile_map[(info["_requests"][r["request_id"]]["seq_len_k"], info["_requests"][r["request_id"]]["nql"])]["link_alone_forced_slo_failure"]]
        passed = sum(r["completion_time_ms"]-r["admission_time_ms"] <= 12*info["_requests"][r["request_id"]]["layer_compute_ms"]+1e-9 for r in forced)
        out["cohorts"][name] = {"population_count": len(sample), "intrinsically_infeasible_count": len(forced),
            "forced_failure_fraction": len(forced)/len(sample) if sample else None,
            "intrinsically_infeasible_slo_pass_count": passed,
            "best_possible_slo_upper_bound": 1-len(forced)/len(sample) if sample else None}
    return out


def analyze_one(path, info, plan, auditor, paired, profile_map):
    parent = auditor.analyze_result(path, info, HERE)
    raw = auditor.read_json(path)
    paired.EXPECTED_REQUESTS = len(info["_requests"])
    metrics, metric_checks = paired.independent_metrics(raw, info, auditor)
    metrics["window_admission_request_ids"] = sorted(r["request_id"] for r in raw["summary"]["request_metrics"] if START <= r["admission_time_ms"] < END)
    metric_checks["request_metric_population_matches_manifest"] = metric_checks.pop("request_metric_population_19456")
    expected_blocks = sum(8*profile_map[(r["seq_len_k"], r["nql"])]["blocks_per_layer"] for r in info["_requests"].values())
    metric_checks["all_eight_layers_physical_blocks_submitted_and_completed"] = raw["summary"]["submitted_blocks"] == raw["summary"]["completed_blocks"] == expected_blocks
    policy = paired.policy_checks(raw, info)
    source = {"frozen_core_hashes_complete": set(raw.get("core_and_policy_sha256", {})) == set(plan["source_sha256"]) - {"run_baseline_npu32_stress.py"},
              "frozen_core_hashes_match": bool(raw.get("core_and_policy_sha256")) and all(plan["source_sha256"].get(k) == v for k, v in raw.get("core_and_policy_sha256", {}).items()),
              "frozen_stress_runner_hash_matches": raw.get("stress_runner_sha256") == plan["source_sha256"]["run_baseline_npu32_stress.py"]}
    command_path = path.parent / "command.json"
    command = auditor.read_json(command_path) if command_path.exists() else {}
    execution = {"child_completed_with_rc0": command.get("status") == "complete" and command.get("returncode") == 0,
        "command_label_seed_strategy": (command.get("label"), command.get("seed"), command.get("strategy")) == (info["label"], info["metadata"]["seed"], raw["strategy"])}
    bound = link_lower_bound(raw, info, profile_map)
    checks = {"input_audit_passed": info["independent_input_audit"]["passed"], "parent_audit_passed": parent["audit"]["passed"],
        "independent_metric_checks_passed": all(metric_checks.values()), "policy_checks_passed": all(policy.values()),
        "frozen_sources_passed": all(source.values()), "link_lower_bound_observed": bound["all_requests_obey_bound"],
        "execution_record_passed": all(execution.values()),
        "intrinsically_infeasible_never_pass": all(v["intrinsically_infeasible_slo_pass_count"] == 0 for v in bound["cohorts"].values())}
    first = [min(r["completion_time_ms"] for r in raw["summary"]["request_metrics"] if r["npu_id"] == n) for n in range(32)]
    w = parent["windows"][0]
    nominal = parent["nominal_demand_scan"]["full_run"]
    passed = all(checks.values())
    return {"status": "complete" if passed else "audit_failed", "label": info["label"], "arm": info["metadata"]["arm"], "seed": info["metadata"]["seed"],
        "strategy": raw["strategy"], "path": str(path.relative_to(HERE)), "file_sha256": auditor.sha(path),
        "input_fingerprint": info["input_fingerprint"], "submit_seed": raw["submit_seed"], "core_source_hashes": raw["core_and_policy_sha256"],
        "static_path_cirs": raw["static_path_cirs_gib_s"], "audit_passed": passed, "audit_checks": checks,
        "metric_checks": metric_checks, "policy_checks": policy, "frozen_source_checks": source,
        "execution_checks": execution, "command_record": command,
        "command_file_sha256": auditor.sha(command_path) if command_path.exists() else None,
        "primary_valid": passed and w["all_npus_active"],
        "warm_diagnostics": {"first_completion_by_npu_ms": first, "all_first_completions_by_1500": max(first) <= 1500+1e-7,
            "first_by_1500_count": sum(v <= 1500+1e-7 for v in first), "all_fourth_completions_by_1500": parent["warmup"]["all_fourth_completions_by_1500"],
            "all_32_active": w["all_npus_active"], "each_npu_mixed_roles": w["each_npu_short_and_long_positive_compute"], "old_main_window_valid": parent["main_window_valid"]},
        "full_run_nominal_capacity_satisfied": nominal["all_ssu_within_capacity"] and nominal["all_npu_links_within_capacity"],
        "metrics": metrics, "link_lower_bound_audit": bound, "parent_audit": paired.public_audit(parent)}


def aggregate(records, seeds, paired):
    out = []
    def maybe_stat(values):
        return paired.stat(values) if all(v is not None for v in values) else {"n":len(values), "mean":None, "sample_sd":None, "min":None, "max":None, "reason":"Empty cohort in at least one seed; no seed removed."}
    for arm in ARMS:
        for strategy in STRATEGIES:
            selected = [r for r in records if r["arm"] == arm and r["strategy"] == strategy and r["seed"] in seeds]
            assert len(selected) == len(seeds) and all("metrics" in r for r in selected)
            row = {"arm":arm, "strategy":strategy, "seeds":list(seeds),
                "audit_passed_count":sum(r["audit_passed"] for r in selected), "primary_valid_count":sum(r["primary_valid"] for r in selected),
                "full_run_nominal_capacity_satisfied_count":sum(r["full_run_nominal_capacity_satisfied"] for r in selected),
                "device_utilization":paired.stat(r["metrics"]["device_utilization"] for r in selected),
                "makespan_ms":paired.stat(r["metrics"]["makespan_ms"] for r in selected), "cohorts":{}, "by_category":{}, "by_role":{}}
            for name in ("window_admissions", "all_requests", "window_arrivals"):
                row["cohorts"][name] = {"count":paired.stat(r["metrics"]["cohorts"][name]["count"] for r in selected)}
                for clock in ("admission", "arrival"):
                    row["cohorts"][name][clock] = maybe_stat([r["metrics"]["cohorts"][name][clock]["rate"] for r in selected])
            for grouping, labels in (("by_category", ("SS","SL","LS","LL")), ("by_role", ("short","long"))):
                for label in labels:
                    row[grouping][label] = {}
                    for name in ("window_admissions", "all_requests"):
                        row[grouping][label][name] = {
                            "count":paired.stat(r["metrics"][grouping][label][name]["count"] for r in selected),
                            "admission":maybe_stat([r["metrics"][grouping][label][name]["admission"]["rate"] for r in selected])}
            out.append(row)
    return out


def flatten(r):
    out = {k:r.get(k) for k in ("arm","seed","strategy","label","status","path","file_sha256","input_fingerprint","error","audit_passed","primary_valid","full_run_nominal_capacity_satisfied")}
    if "metrics" not in r:
        return out
    m = r["metrics"]
    out.update(device_utilization_percent=100*m["device_utilization"], makespan_ms=m["makespan_ms"], **r["warm_diagnostics"])
    out.pop("first_completion_by_npu_ms")
    out["max_first_completion_ms"] = max(r["warm_diagnostics"]["first_completion_by_npu_ms"])
    out["max_fourth_completion_ms"] = max(r["parent_audit"]["warmup"]["fourth_completion_by_npu_ms"])
    for name, prefix in (("window_admissions","warm"),("all_requests","full"),("window_arrivals","warm_arrivals")):
        c = m["cohorts"][name]
        out[prefix+"_count"] = c["count"]
        out[prefix+"_completion_after_4000ms_count"] = c["completion_after_window_end_count"]
        for clock in ("admission","arrival"):
            out[prefix+"_"+clock+"_passed"] = c[clock]["passed"]
            out[prefix+"_"+clock+"_slo_percent"] = None if c[clock]["rate"] is None else 100*c[clock]["rate"]
        for group, labels in (("by_category",("SS","SL","LS","LL")),("by_role",("short","long"))):
            for label in labels:
                cc = m[group][label][name]
                out[f"{prefix}_{label}_count"] = cc["count"]
                out[f"{prefix}_{label}_admission_passed"] = cc["admission"]["passed"]
                out[f"{prefix}_{label}_admission_slo_percent"] = None if cc["admission"]["rate"] is None else 100*cc["admission"]["rate"]
    for name, scan in r["parent_audit"]["nominal_demand_scan"].items():
        for key in ("max_ssu_gib_s","any_ssu_over_capacity_ms","max_npu_link_gib_s","all_ssu_within_capacity","all_npu_links_within_capacity"):
            out[f"nominal_{name}_{key}"] = scan[key]
    for name, bound in r["link_lower_bound_audit"]["cohorts"].items():
        for key,value in bound.items():
            out[f"link_bound_{name}_{key}"] = value
    return out


def csv_text(rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, keys);writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def markdown(report, paired):
    lines = ["# 原始 data 均匀抽样：32 NPU / 6 SSU，Baseline 与 Once per layer", "",
        f"固定 seeds：{list(SEEDS)}；解析 {report['counts']['parsed']}/20，pending={report['counts']['pending']}，错误={report['counts']['errors']}，重复={report['counts']['duplicates']}。",
        "两臂分别是原始 84 行全表均匀有放回抽样，以及事先按逐盘最坏界筛定的 33 行子集。每卡独立 Random(seed+100003×NPU)，从排序目录逐条抽至纯八层计算总和至少 6261.372612 ms。没有固定暖机画像、失败 seed 重抽、C 缩放或循环模板。所有请求在 t=0 到达，固定每卡顺序与物理布局；同 seed 两策略输入完全相同。", "",
        "每个 seed 同时确定输入抽样序列与模拟器提交随机种子；因此五次运行同时覆盖这两种随机变化。同 seed 的 Baseline/Once 使用相同输入和相同提交种子，不将跨 seed 方差解释为单独某一种随机来源。", "",
        "每条请求 8 层，计算时间 C 严格来自原始 data；原始 V 保留。每层按 176 KiB 物理命令分块，6 盘条带为 (block_index+NPU//4)%6。仅 12 个 NQL=64 画像将最后 88 KiB 分配成 176 KiB，每层额外 88 KiB；统计容量用实际物理 V。NQL 是画像参数，不能把 V/C 当成实际时刻的已发 IO。", "",
        r"主窗固定 $[2000,4000)$ ms：$U=\sum_n C_n^{window}/(32\times2000)$；主表 $SLO=\#\{r:a_r\in[2000,4000),f_r-a_r\le1.5\times8C_r\}/\#\{r:a_r\in[2000,4000)\}$。",
        "这是接纳后八层 TTFT 代理，排除接纳前排队，并非真实硬件首 token 测量。窗口接纳请求跟到完成，不在 4 秒截断；策略之间暖接纳人口不同。所有到达均为 0，因此暖窗到达人口为空（N/A）；暖接纳人口的到达→完成 SLO 为 0，也不能改叫真实 TTFT。8C 是纯计算下界，不使用 data 原有 78 层 TTFT。", "",
        "均值先按各 seed 独立求率，再对五个 seed 等权平均；± 为样本标准差，[] 为 min/max。不会池化不同规模暖窗人口。仅源码/输入/指标审计通过且 32 卡全窗 active 作为本组有效条件；首请求/第四请求在 1500 ms 前完成、每卡窗口长短混合、名义欠载另列，任何不满足均保留。", ""]
    if report["means_published"]:
        lines += ["| 抽样臂 | 策略 | 暖窗 NPU U：均值±SD [min,max] | 暖接纳 SLO×1.5：均值±SD [min,max] | 主条件通过 | 全程名义欠载通过 |", "|---|---|---:|---:|---:|---:|"]
        for row in report["aggregates_all_five_seeds"]:
            lines.append(f"| {row['arm']} | {row['strategy']} | {paired.fmt_stat(row['device_utilization'])} | {paired.fmt_stat(row['cohorts']['window_admissions']['admission'])} | {row['primary_valid_count']}/5 | {row['full_run_nominal_capacity_satisfied_count']}/5 |")
    else:
        lines += ["尚未完整验收五个 seed；这里只发布逐运行记录，不发布不完整均值。"]
    lines += ["", "## 逐 seed 与暖机/容量诊断", "", "| 臂 | seed | 策略 | U % | 暖接纳 SLO (通过/总数) | 首/第四完成≤1.5s | 全卡 active / 每卡混合 | 全程名义盘峰值 GiB/s / 超40时长ms |", "|---|---:|---|---:|---:|---|---|---:|"]
    for r in report["runs"]:
        if "metrics" not in r:
            lines.append(f"| {r['arm']} | {r['seed']} | {r['strategy']} | {r['status']} | — | — | — | — |")
            continue
        d=r["warm_diagnostics"];scan=r["parent_audit"]["nominal_demand_scan"]["full_run"]
        lines.append(f"| {r['arm']} | {r['seed']} | {r['strategy']} | {100*r['metrics']['device_utilization']:.6f} | {paired.fmt_cohort(r['metrics']['cohorts']['window_admissions'])} | {d['all_first_completions_by_1500']}/{d['all_fourth_completions_by_1500']} | {d['all_32_active']}/{d['each_npu_mixed_roles']} | {scan['max_ssu_gib_s']:.6f} / {scan['any_ssu_over_capacity_ms']:.6f} |")
    lines += ["", "名义需求扫描以每卡当前已接纳但未完成请求的实际逐盘 V/C 求和，同刻完成和接纳批量处理，只考察正持续时间区间。它不包含未来请求额外 L0 预取，也不是 SSD 物理服务速率或无 stall 证明；40/50 GiB/s 是设备服务上限，表中的超限值表示名义需求。", "",
        "## 目录与容量口径", "", "| 目录 | 行数 | 平均单层 C ms | 平均原V / 物理V MiB | E[Vphysical]/E[C] GiB/s | E[Vphysical/C] GiB/s | V/C>50行数 | 类别行数 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm,s in report["catalog_statistics"].items():
        lines.append(f"| {arm} | {s['row_count']} | {s['mean_layer_compute_ms']:.9f} | {s['mean_raw_V_mib']:.9f} / {s['mean_physical_V_mib']:.9f} | {s['ratio_mean_physical_V_over_mean_C_gib_s']:.9f} | {s['request_equal_mean_V_over_C_gib_s']:.9f} | {s['rows_nominal_link_above_50']} | {s['category_counts']} |")
    lines += ["", "全表请求等概率与计算时间等权不是一回事：长计算画像占用更多运行时间，因此 E[V]/E[C] 和 E[V/C] 不能互换，也不能据较小的长期比例断言任意瞬间欠载。33 行是预先筛选后重新均匀抽样，仅包含 SL/LL，不能代表原 84 行分布。S/L 第一位按 seq≤80K 划分，第二位按 NQL<512 划分；role short/long 仅指 seq≤80K/＞80K，不是实测运行时间。", "",
        r"单卡链路还给出独立下界：$f-a\ge7V_{physical}/50+C$（单位统一为秒），因为只有 L0 可以在接纳前预取，而 L1–7 都需在接纳后通过本卡链路，最后一层还要计算。若 $V/C>50\times11/7=78.5714286$ GiB/s，则 SLO×1.5 即使无 SSD 排队也不可能达标。全目录有 9 行满足该条件；具体暖窗上界必须用该窗实际接纳的这类请求数计算，不能直接套目录 9/84。", "",
        "## 分类别与完整人口附录", ""]
    if report["means_published"]:
        lines += ["| 臂 / 策略 | 类别 | 暖接纳人数均值 [min,max] | 暖接纳 SLO：均值±SD [min,max] | 完整人口 SLO：均值±SD [min,max] |", "|---|---|---:|---:|---:|"]
        for row in report["aggregates_all_five_seeds"]:
            for cat,v in row["by_category"].items():
                c=v["window_admissions"]["count"]
                lines.append(f"| {row['arm']} / {row['strategy']} | {cat} | {c['mean']:.2f} [{c['min']},{c['max']}] | {paired.fmt_stat(v['window_admissions']['admission'])} | {paired.fmt_stat(v['all_requests']['admission'])} |")
        lines += ["", "| 臂 / 策略 | 完整输入人数均值 | 完整人口接纳后 SLO | 完整人口到达后 SLO | 完工时间 ms 均值±SD |", "|---|---:|---:|---:|---:|"]
        for row in report["aggregates_all_five_seeds"]:
            m=row["makespan_ms"];c=row["cohorts"]["all_requests"]
            lines.append(f"| {row['arm']} / {row['strategy']} | {c['count']['mean']:.1f} | {paired.fmt_stat(c['admission'])} | {paired.fmt_stat(c['arrival'])} | {m['mean']:.6f} ± {m['sample_sd']:.6f} |")
    lines += ["", "逐画像、长短 role、窗口请求等权利用率、L0/L1–7 stall 分解、逐卡计算占用、逐时刻需求证据和逐项源码/公平性审计保存在 analysis.json；per_seed.csv 保留所有运行和人口计数。删除 seed7 的四 seed 敏感性结果只作附录，保存在同一 JSON，主表始终使用预定五个 seed。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--partial",action="store_true");args=parser.parse_args()
    auditor=module(STUDY/"analyze.py","raw_parent_auditor");auditor.BASE=HERE;auditor.WINDOWS=((START,END),)
    paired=module(STUDY/"paired_once_5seeds"/"analyze_paired.py","raw_paired_metrics")
    plan=auditor.read_json(HERE/"plan.json")
    sources={str(p.relative_to(ROOT)):auditor.sha(p) for p in (Path(__file__), HERE/"run_raw.py", HERE/"plan.json", STUDY/"analyze.py", STUDY/"paired_once_5seeds"/"analyze_paired.py",ROOT/"data")}
    frozen={name:auditor.sha(ROOT/name)==expected and auditor.sha(HERE/"sources"/name)==expected for name,expected in plan["source_sha256"].items()}
    all_rows=catalog();profile_map={(r["seq_len_k"],r["nql"]):r for r in all_rows}
    infos=[];records=[]
    for item in plan["inputs"]:
        info=audit_input(item,plan,all_rows,auditor);infos.append(info)
        for strategy in STRATEGIES:
            files=sorted((HERE/"runs"/item["label"]/strategy).glob("*.json.gz"))
            r={"label":item["label"],"arm":item["arm"],"seed":item["seed"],"strategy":strategy,"status":"pending"}
            if len(files)>1:r.update(status="duplicate",paths=[str(p.relative_to(HERE)) for p in files])
            elif files:
                try:r=analyze_one(files[0],info,plan,auditor,paired,profile_map)
                except Exception as exc:r.update(status="error",error=f"{type(exc).__name__}: {exc}")
            records.append(r)
            print(json.dumps({k:r[k] for k in ("label","strategy","status")},ensure_ascii=False),flush=True)
    pairs=paired.policy_pairs(infos,records)
    for pair in pairs:
        matched = {r["strategy"]: r for r in records if r["label"] == pair["label"] and "metrics" in r}
        if set(matched) == set(STRATEGIES):
            b = set(matched["baseline"]["metrics"]["window_admission_request_ids"])
            o = set(matched["once"]["metrics"]["window_admission_request_ids"])
            pair["warm_population_overlap"] = {"intersection_count": len(b & o), "baseline_only_count": len(b-o),
                "once_only_count": len(o-b), "identical_request_ids": b == o}
    counts={"planned":20,"parsed":sum("metrics" in r for r in records), "pending":sum(r["status"]=="pending" for r in records),
            "errors":sum(r["status"]=="error" for r in records),"duplicates":sum(r["status"]=="duplicate" for r in records),
            "audit_passed":sum(r.get("audit_passed",False) for r in records),"primary_valid":sum(r.get("primary_valid",False) for r in records)}
    unchanged=all(auditor.sha(ROOT/k)==v for k,v in sources.items()) and all(auditor.sha(ROOT/name)==expected for name,expected in plan["source_sha256"].items())
    complete=counts["parsed"]==20 and not any(counts[k] for k in ("pending","errors","duplicates"))
    publish=complete and not args.partial
    report={"generated_utc":datetime.now(timezone.utc).isoformat(),"partial_preview":args.partial,"counts":counts,"means_published":publish,
        "source_paths_relative_to":str(ROOT),"analysis_source_sha256":sources,"frozen_source_checks":frozen,"sources_unchanged_during_analysis":unchanged,
        "all_audits_passed":complete and counts["audit_passed"]==20 and all(frozen.values()) and unchanged and all(p["status"]=="passed" for p in pairs),
        "primary_validity_definition":"Input/source/metric integrity and all32 NPUs active for the complete [2000,4000)ms. Old warmup, mixed roles and nominal capacity are diagnostics and never used to exclude a seed.",
        "catalog_statistics":plan["catalog_stats"],"catalog_independently_recomputed":all_rows,
        "inputs":[paired.public_input(i) for i in infos],"policy_pairs":pairs,"runs":records,
        "aggregates_all_five_seeds":aggregate(records,SEEDS,paired) if publish else [],
        "sensitivity_excluding_seed7":aggregate(records,SEEDS[1:],paired) if publish else []}
    prefix="preview_" if args.partial else ""
    paired.atomic_text(HERE/(prefix+"analysis.json"),json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    compact = {k: report[k] for k in ("generated_utc", "partial_preview", "counts", "means_published", "all_audits_passed",
        "primary_validity_definition", "analysis_source_sha256", "source_paths_relative_to", "sources_unchanged_during_analysis",
        "catalog_statistics", "policy_pairs", "sensitivity_excluding_seed7")}
    compact["groups"] = report["aggregates_all_five_seeds"]
    compact["detail_artifact"] = prefix+"analysis.json"
    paired.atomic_text(HERE/(prefix+"summary.json"),json.dumps(compact,ensure_ascii=False,indent=2)+"\n")
    paired.atomic_text(HERE/(prefix+"per_seed.csv"),csv_text([flatten(r) for r in records]))
    paired.atomic_text(HERE/(prefix+"catalog_profiles.csv"),csv_text(all_rows))
    paired.atomic_text(HERE/(prefix+"comparison.md"),markdown(report,paired))
    print(json.dumps({"counts":counts,"all_audits_passed":report["all_audits_passed"],"means_published":publish},ensure_ascii=False),flush=True)
    return 0 if args.partial or report["all_audits_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
