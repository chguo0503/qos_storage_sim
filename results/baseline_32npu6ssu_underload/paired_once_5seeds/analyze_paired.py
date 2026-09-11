#!/usr/bin/env python3
"""Audit the fixed 5-seed Random/Ordered x Baseline/Once comparison.

Read-only with respect to manifests, raw runs, core code and the parent study.
Writes only this directory's per_seed.csv, summary.json and comparison.md.
Use --partial for a progress preview: no across-seed means are published.
All JSON rates are fractions. CSV rate columns explicitly use percent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SEEDS = (7, 19, 43, 67, 101)
ORDERS = ("random", "ordered")
STRATEGIES = ("baseline", "once")
START, END, ALPHA = 2000.0, 4000.0, 1.5
EXPECTED_REQUESTS = 19456
PROFILE_COUNTS_PER_NPU = {(1.0, 128.0): 200, (1.0, 256.0): 200,
                          (1.0, 384.0): 200, (192.0, 768.0): 8}


def import_auditor(base):
    source = base / "analyze.py"
    spec = importlib.util.spec_from_file_location("paired_once_parent_auditor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Change this imported module's in-memory window list only. Never call main.
    module.BASE = base
    module.WINDOWS = ((START, END),)
    return module


def label_for(seed, order):
    return f"concurrency_l768_seed{seed}" + ("__exact_cohort4_p1" if order == "ordered" else "")


def stat(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        stream.write(value)
        tmp = Path(stream.name)
    tmp.replace(path)


def cohort(rows, inputs):
    result = {"count": len(rows), "completion_after_window_end_count":
              sum(r["completion_time_ms"] > END for r in rows)}
    for name, field in (("arrival", "arrival_time_ms"), ("admission", "admission_time_ms")):
        passed = sum(r["completion_time_ms"] - r[field]
                     <= ALPHA * 8 * inputs[r["request_id"]]["layer_compute_ms"] + 1e-9
                     for r in rows)
        result[name] = {"count": len(rows), "passed": passed,
                        "rate": passed / len(rows) if rows else None}
    return result


def independent_metrics(raw, info, auditor):
    summary = raw["summary"]
    rows = summary["request_metrics"]
    req = info["_requests"]
    ids = [r["request_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(req):
        raise ValueError("request_metrics does not contain each input request exactly once")
    if not all(math.isfinite(r["completion_time_ms"]) for r in rows):
        raise ValueError("unfinished request; cannot report uncensored SLO")
    batches = summary["microbatch_metrics"]
    compute = math.fsum(auditor.overlap(l["compute_start_ms"], l["compute_end_ms"], START, END)
                        for b in batches for l in b["layer_metrics"])
    warm = [r for r in rows if START <= r["admission_time_ms"] < END]
    arrivals = [r for r in rows if START <= r["arrival_time_ms"] < END]
    populations = {"all_requests": rows, "window_admissions": warm, "window_arrivals": arrivals}
    out = {"device_utilization": compute / (32 * (END - START)),
           "window_compute_ms": compute, "makespan_ms": summary["makespan_ms"],
           "cohorts": {name: cohort(sample, req) for name, sample in populations.items()},
           "by_role": {}, "by_category": {}, "by_profile": []}
    for field, labels in (("role", ("short", "long")), ("category", ("SS", "SL", "LS", "LL"))):
        out["by_" + field] = {
            name: {key: cohort([r for r in sample if req[r["request_id"]][field] == name], req)
                   for key, sample in populations.items()}
            for name in labels}
    for seq, nql in sorted({(r["seq_len_k"], r["nql"]) for r in req.values()}):
        out["by_profile"].append({"seq_len_k": seq, "nql": nql,
            **{name: cohort([r for r in sample if (req[r["request_id"]]["seq_len_k"],
                                                  req[r["request_id"]]["nql"]) == (seq, nql)], req)
               for name, sample in populations.items()}})
    checks = {"own_compute_matches_eight_input_layers": all(
        auditor.close(r["own_compute_ms"], 8 * req[r["request_id"]]["layer_compute_ms"]) for r in rows),
        "request_metric_population_19456": len(rows) == EXPECTED_REQUESTS,
        "independent_all_input_arrivals_zero": all(r["arrival_time_ms"] == 0 for r in rows)}
    stored = raw.get("slo", {})
    checks["stored_slo_alpha_1_5"] = stored.get("alpha") == ALPHA
    checks["stored_slo_window_2_4s"] = (stored.get("window_start_ms"), stored.get("window_end_ms")) == (START, END)
    checks["stored_slo_excludes_78layer_source_ttft"] = stored.get("threshold_excludes_source_data_78layer_ttft") is True
    checks["stored_slo_ideal_8C"] = stored.get("ideal") == "n_layers * per_layer_compute_ms (own_compute_ms)"

    def equal_cohort(a, b):
        for clock in ("arrival", "admission"):
            aa, bb = a[clock], b.get(clock, {})
            if aa["count"] != bb.get("count") or aa["passed"] != bb.get("passed"):
                return False
            if aa["rate"] is None:
                if bb.get("rate") is not None:
                    return False
            elif bb.get("rate") is None or not math.isclose(aa["rate"], bb["rate"], rel_tol=1e-12, abs_tol=1e-12):
                return False
        return a["completion_after_window_end_count"] == b.get("completion_after_window_end_count")

    for name, sample in populations.items():
        saved = stored.get(name, {})
        checks[f"stored_{name}_slo_recomputed"] = equal_cohort(out["cohorts"][name], saved)
        checks[f"stored_{name}_exact_request_ids"] = (
            len(saved.get("request_ids", [])) == len(sample)
            and set(saved.get("request_ids", [])) == {r["request_id"] for r in sample})
    profiles = {(p["seq_len_k"], p["nql"]): p for p in stored.get("per_profile", [])}
    checks["stored_per_profile_slo_recomputed"] = len(profiles) == len(out["by_profile"]) and all(
        equal_cohort(p["all_requests"], profiles.get((p["seq_len_k"], p["nql"]), {}))
        for p in out["by_profile"])
    stored_windows = [w for w in raw.get("windows", [])
                      if (w.get("start_ms"), w.get("end_ms")) == (START, END)]
    checks["stored_main_window_unique"] = len(stored_windows) == 1
    checks["stored_main_U_recomputed"] = len(stored_windows) == 1 and auditor.close(
        stored_windows[0]["mean_npu_utilization"], out["device_utilization"])
    checks["warm_arrival_cohort_empty"] = len(arrivals) == 0
    return out, checks


def policy_checks(raw, info):
    adapter = raw.get("adapter_statistics", {})
    summary = raw["summary"]
    times = adapter.get("collector_times_ms", [])
    from strategy_profiles import FINAL_STATIC
    expected_cirs = list(FINAL_STATIC.path_cirs())
    return {
        "strategy_baseline_or_once": raw.get("strategy") in STRATEGIES,
        "adapter_strategy_matches": adapter.get("strategy") == raw.get("strategy"),
        "shared_collector_5ms": raw.get("collector_interval_ms") == adapter.get("collector_interval_ms") == 5.0,
        "actual_collector_5ms_ticks": bool(times) and all(abs(t - 5*i) < 1e-7 for i, t in enumerate(times)),
        "all_ssu_collector_counts": adapter.get("fresh_reads_by_ssu") == [len(times)] * 6,
        "static_final_path_cirs": raw.get("static_path_cirs_gib_s") == expected_cirs,
        "no_runtime_cir_writes": adapter.get("cir_write_events") == [],
        "no_npu_reassignment": adapter.get("assignment_count") == 0 and raw.get("assignment_log") == [],
        "no_path_internal_reorder": adapter.get("reorder_calls") == 0 and adapter.get("reorder_changed") == 0,
        "reservation_ack_conservation": adapter.get("reserved_blocks") == adapter.get("acknowledged_blocks") == summary.get("completed_blocks"),
        "ledger_finally_empty": adapter.get("ledger_end_counts_by_ssu") == [0]*6,
        "zero_modeled_control_cost": adapter.get("modeled_control_communication_latency_ms") == 0 and adapter.get("modeled_control_cpu_latency_ms") == 0,
        "expected_seed": raw.get("submit_seed") == info["metadata"].get("seed"),
        "placement_fingerprints_unchanged": (bool(raw.get("input_placement_fingerprint"))
            and raw.get("execution_placement_fingerprint") == raw.get("input_placement_fingerprint")),
    }


def public_input(info):
    return {key: value for key, value in info.items() if not key.startswith("_")}


def public_audit(result):
    value = {key: val for key, val in result.items() if key != "original_slo"}
    value["windows"] = [{k: v for k, v in w.items() if k != "requests"} for w in result["windows"]]
    # The eight 250-ms subwindows were deliberately not requested in this audit.
    value.pop("all_subwindows_active_and_mixed", None)
    return value


def flatten(record):
    flat = {key: record.get(key) for key in ("seed", "order", "strategy", "label", "status", "path", "file_sha256")}
    flat["error"] = record.get("error", "")
    flat["audit_passed"] = record.get("audit_passed")
    flat["main_window_valid"] = record.get("main_window_valid")
    flat["full_run_nominal_capacity_satisfied"] = record.get("full_run_nominal_capacity_satisfied")
    flat["all_study_conditions_met"] = record.get("all_study_conditions_met")
    flat["input_fingerprint"] = record.get("input_fingerprint")
    m = record.get("metrics")
    if not m:
        return flat
    flat["device_utilization_percent"] = 100*m["device_utilization"]
    flat["makespan_ms"] = m["makespan_ms"]
    for cname, prefix in (("window_admissions", "warm"), ("window_arrivals", "warm_arrivals"), ("all_requests", "full")):
        c = m["cohorts"][cname]
        flat[prefix + "_count"] = c["count"]
        flat[prefix + "_completion_after_4000ms_count"] = c["completion_after_window_end_count"]
        for clock in ("admission", "arrival"):
            flat[f"{prefix}_{clock}_passed"] = c[clock]["passed"]
            flat[f"{prefix}_{clock}_slo_percent"] = None if c[clock]["rate"] is None else 100*c[clock]["rate"]
        for role in ("short", "long"):
            rc = m["by_role"][role][cname]
            flat[f"{prefix}_{role}_count"] = rc["count"]
            for clock in ("admission", "arrival"):
                flat[f"{prefix}_{role}_{clock}_passed"] = rc[clock]["passed"]
                flat[f"{prefix}_{role}_{clock}_slo_percent"] = None if rc[clock]["rate"] is None else 100*rc[clock]["rate"]
    pa = record.get("parent_audit")
    if pa:
        scan = pa["nominal_demand_scan"]["full_run"]
        w = pa["windows"][0]
        flat.update(nominal_full_max_ssu_gib_s=scan["max_ssu_gib_s"],
                    nominal_full_any_ssu_over_capacity_ms=scan["any_ssu_over_capacity_ms"],
                    nominal_full_max_npu_link_gib_s=scan["max_npu_link_gib_s"],
                    max_fourth_completion_ms=max(pa["warmup"]["fourth_completion_by_npu_ms"]),
                    all_32_active_whole_window=w["all_npus_active"],
                    each_npu_has_short_and_long_compute=w["each_npu_short_and_long_positive_compute"],
                    min_per_npu_short_compute_ms=min(n["by_role"]["short"]["compute_ms"] for n in w["per_npu"]),
                    min_per_npu_long_compute_ms=min(n["by_role"]["long"]["compute_ms"] for n in w["per_npu"]))
    return flat


def aggregate(records, seeds):
    result = []
    for order in ORDERS:
        for strategy in STRATEGIES:
            selected = [r for r in records if r["order"] == order and r["strategy"] == strategy and r["seed"] in seeds]
            assert len(selected) == len(seeds) and all("metrics" in r for r in selected)
            row = {"order": order, "strategy": strategy, "seeds": list(seeds),
                   "n_seeds": len(seeds), "audit_passed_count": sum(r["audit_passed"] for r in selected),
                   "study_conditions_met_count": sum(r["all_study_conditions_met"] for r in selected)}
            row["device_utilization"] = stat(r["metrics"]["device_utilization"] for r in selected)
            row["makespan_ms"] = stat(r["metrics"]["makespan_ms"] for r in selected)
            row["cohorts"] = {}
            for name in ("window_admissions", "window_arrivals", "all_requests"):
                row["cohorts"][name] = {"request_count": stat(r["metrics"]["cohorts"][name]["count"] for r in selected)}
                for clock in ("admission", "arrival"):
                    values = [r["metrics"]["cohorts"][name][clock]["rate"] for r in selected]
                    row["cohorts"][name][clock] = stat(values) if all(v is not None for v in values) else {
                        "n": len(values), "mean": None, "sample_sd": None, "min": None, "max": None,
                        "reason": "At least one seed has an empty cohort; no seed was dropped."}
            row["by_role"] = {}
            for role in ("short", "long"):
                row["by_role"][role] = {}
                for name in ("window_admissions", "all_requests"):
                    values = [r["metrics"]["by_role"][role][name]["admission"]["rate"] for r in selected]
                    row["by_role"][role][name] = stat(values) if all(v is not None for v in values) else {
                        "n": len(values), "mean": None, "sample_sd": None, "min": None, "max": None,
                        "reason": "At least one seed has an empty role cohort; no seed was dropped."}
            result.append(row)
    return result


def policy_pairs(inputs, records):
    out = []
    for info in inputs:
        groups = {s: [r for r in records if r["label"] == info["label"] and r["strategy"] == s] for s in STRATEGIES}
        pair = {"label": info["label"], "input_fingerprint": info["input_fingerprint"], "status": "pending"}
        if any(len(g) > 1 for g in groups.values()):
            pair["status"] = "duplicate"
        elif all(len(g) == 1 and "metrics" in g[0] for g in groups.values()):
            b, o = groups["baseline"][0], groups["once"][0]
            checks = {"same_manifest_and_input_fingerprint": b["input_fingerprint"] == o["input_fingerprint"] == info["input_fingerprint"],
                      "same_submit_seed": b["submit_seed"] == o["submit_seed"] == info["metadata"]["seed"],
                      "same_core_source_hashes": b.get("core_source_hashes") == o.get("core_source_hashes"),
                      "same_static_path_cirs": b.get("static_path_cirs") == o.get("static_path_cirs"),
                      "both_run_audits_passed": b["audit_passed"] and o["audit_passed"]}
            pair.update(status="passed" if all(checks.values()) else "audit_failed", checks=checks)
            pair["warm_cohort_counts"] = {s: groups[s][0]["metrics"]["cohorts"]["window_admissions"]["count"] for s in STRATEGIES}
            pair["note"] = "Identical complete input population, but warm admission cohorts are policy-dependent."
        out.append(pair)
    return out


def fmt_pct(value):
    return "N/A" if value is None else f"{100*value:.4f}%"


def fmt_stat(value):
    if value["mean"] is None:
        return "N/A"
    sd = "N/A" if value["sample_sd"] is None else f"{100*value['sample_sd']:.4f}"
    return f"{100*value['mean']:.4f} ± {sd}% [{100*value['min']:.4f}, {100*value['max']:.4f}]"


def fmt_cohort(value, clock="admission"):
    m = value[clock]
    return f"{m['passed']}/{m['count']} ({fmt_pct(m['rate'])})"


def markdown(report):
    counts = report["counts"]
    lines = ["# Random / Ordered × Baseline / Once per layer：五种子配对比较", "",
        f"固定种子：{', '.join(map(str, SEEDS))}。已解析 {counts['parsed_slots']}/20 格；pending {counts['pending_slots']}，重复 {counts['duplicate_slots']}，错误 {counts['error_slots']}。",
        "主窗固定 [2000,4000) ms，32 NPU、6 SSU、每请求 8 层。主表 SLO 人口为该窗内接纳的请求，每条跟踪至真正完成，不在 4 秒截断。", "",
        r"\[ U=\frac{\sum_{n=1}^{32}|\text{计算区间}_n\cap[2000,4000)|}{32\times2000},\qquad "
        r"SLO_{warm}=\frac{\#\{r:2000\le a_r<4000,\ f_r-a_r\le1.5\times8C_r\}}{\#\{r:2000\le a_r<4000\}}.\]", "",
        "其中 a 为接纳时刻、f 为八层完成时刻、C 为该请求单层纯计算时间。这里称为“接纳后 TTFT 代理”，不含接纳前等待，也不是实际硬件首 token 测量；8C 是纯计算下界，不是孤立运行实测延迟或原始数据中的 78 层 TTFT。",
        "所有输入在 t=0 到达，暖窗到达人口为空（N/A）。暖窗接纳人口的到达→完成达标率单独保留；不得把到达时刻重置成接纳时刻来称作用户端 TTFT。", "",
        "每个单元格先对 seed 独立计算，再对预定 seed 等权平均；± 为样本标准差（分母 n−1），方括号为 min/max。窗口接纳人数随策略和 seed 改变，不能直接池化窗口请求来替代种子等权均值。",
        "任何源码、暖机、每卡混合或容量条件不通过的运行均保留，不静默删除；条件列明确列出合格数量。", ""]
    if report["means_published"]:
        lines += ["| 输入顺序 | 策略 | 暖窗 NPU U：均值 ± SD [min,max] | 暖接纳 SLO×1.5：均值 ± SD [min,max] | 条件通过 |",
                  "|---|---|---:|---:|---:|"]
        for row in report["aggregates_all_five_seeds"]:
            lines.append(f"| {row['order'].title()} | {'Baseline' if row['strategy']=='baseline' else 'Once per layer（5 ms共享采样）'} | {fmt_stat(row['device_utilization'])} | {fmt_stat(row['cohorts']['window_admissions']['admission'])} | {row['study_conditions_met_count']}/5 |")
        lines += ["", "删除探索时使用的 seed 7 后的敏感性结果（19、43、67、101；仍逐 seed 等权）：", "",
                  "| 输入顺序 | 策略 | 暖窗 NPU U：均值 ± SD [min,max] | 暖接纳 SLO×1.5：均值 ± SD [min,max] | 条件通过 |",
                  "|---|---|---:|---:|---:|"]
        for row in report["aggregates_excluding_seed7"]:
            lines.append(f"| {row['order'].title()} | {row['strategy']} | {fmt_stat(row['device_utilization'])} | {fmt_stat(row['cohorts']['window_admissions']['admission'])} | {row['study_conditions_met_count']}/4 |")
    else:
        lines += ["**当前是未完成或 --partial 预览，不发布部分种子的均值冒充五种子结果。**"]
    lines += ["", "## 逐 seed 主指标与短／长窗口人口", "",
              "| seed | 顺序 | 策略 | 状态 | NPU U | 暖接纳 SLO | 短类暖接纳 SLO | 长类暖接纳 SLO | 4秒后完成数 |",
              "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for slot in report["slots"]:
        if len(slot["runs"]) != 1 or "metrics" not in slot["runs"][0]:
            lines.append(f"| {slot['seed']} | {slot['order']} | {slot['strategy']} | {slot['status']} | — | — | — | — | — |")
            continue
        r=slot["runs"][0];m=r["metrics"];c=m["cohorts"]["window_admissions"]
        lines.append(f"| {r['seed']} | {r['order']} | {r['strategy']} | {r['status']} | {fmt_pct(m['device_utilization'])} | {fmt_cohort(c)} | {fmt_cohort(m['by_role']['short']['window_admissions'])} | {fmt_cohort(m['by_role']['long']['window_admissions'])} | {c['completion_after_window_end_count']} |")
    if report["means_published"]:
        lines += ["", "## 短／长窗口 SLO：逐 seed 等权汇总", "",
                  "| 顺序 | 策略 | 短类：均值 ± SD [min,max] | 长类：均值 ± SD [min,max] |", "|---|---|---:|---:|"]
        for row in report["aggregates_all_five_seeds"]:
            lines.append(f"| {row['order']} | {row['strategy']} | {fmt_stat(row['by_role']['short']['window_admissions'])} | {fmt_stat(row['by_role']['long']['window_admissions'])} |")
    lines += ["", "## 附录：完整同人口的 SLO 与暖窗到达口径", "",
              "完整人口为每 seed 的同一 19,456 请求，其中短类 19,200、长类 256；顺序重编码通过 original_request_id 恢复身份。以下仍跟踪全部请求至完成。",
              "", "| seed | 顺序 | 策略 | 完整接纳 SLO | 完整到达 SLO | 完整短接纳 SLO | 完整长接纳 SLO | 暖接纳人口的到达 SLO | 暖到达人口 |", "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["runs"]:
        if "metrics" not in r:
            continue
        m=r["metrics"];c=m["cohorts"]
        lines.append(f"| {r['seed']} | {r['order']} | {r['strategy']} | {fmt_cohort(c['all_requests'])} | {fmt_cohort(c['all_requests'],'arrival')} | {fmt_cohort(m['by_role']['short']['all_requests'])} | {fmt_cohort(m['by_role']['long']['all_requests'])} | {fmt_cohort(c['window_admissions'],'arrival')} | {fmt_cohort(c['window_arrivals'],'arrival')} |")
    lines += ["", "## 有效性与可复算来源", "",
              "| seed | 顺序 | 策略 | 源码/输入/统计核对 | 暖机且32卡全程active并每卡有长短计算 | 全程逐盘名义峰值 GiB/s | 超40时间 ms | 全程名义容量通过 |",
              "|---:|---|---|---|---|---:|---:|---|"]
    for r in report["runs"]:
        pa=r.get("parent_audit")
        if not pa: continue
        scan=pa["nominal_demand_scan"]["full_run"]
        lines.append(f"| {r['seed']} | {r['order']} | {r['strategy']} | {r['audit_passed']} | {r['main_window_valid']} | {scan['max_ssu_gib_s']:.9f} | {scan['any_ssu_over_capacity_ms']:.9f} | {r['full_run_nominal_capacity_satisfied']} |")
    lines += ["", "暖机要求：各卡第4条完成≤1500 ms；主窗每卡active=2000 ms且短／长各有正计算时间。名义需求按当前接纳请求的实际逐盘 V/C，对 admission/completion 同刻事件批处理后扫描全程；它不额外叠加跨请求 L0，也不是物理队列无突发的证明。",
              "", "两个随机源同时变化：各卡随机输入序列和同刻客户端提交的随机顺序。Ordered 使用固定的 exact_cohort4_p1 模板，跨 seed 可具有相同模拟器 input_fingerprint；metadata/original_request_id 映射与 submit_seed 不同，必须保留不同 label 的实际运行，不能称为五个独立坏顺序。",
              "", "同 seed 两策略核对同一输入指纹、提交 seed、静态 CIR 和核心源码；Random/Ordered 按 original_request_id 双射核对每卡身份、计算、到达及实际 placement。顺序编码 request_id 也参与选路平局，因此这是保留请求内容的顺序敏感性比较。",
              "", f"配对审计：输入顺序配对 {counts['order_input_pairs_passed']}/5；同输入策略配对 {counts['policy_pairs_passed']}/10。运行审计通过 {counts['run_audits_passed']}/{counts['parsed_runs']}；完整条件通过 {counts['study_conditions_met']}/{counts['parsed_runs']}。",
              "", "完整运行路径、SHA256、逐卡时间、逐事件容量峰值见 [summary.json](summary.json)；数值长表见 [per_seed.csv](per_seed.csv)。此脚本不运行仿真，不修改父 study 的 analysis.json/summary.csv。"]
    problems=[r for r in report["runs"] if not r.get("audit_passed",False)]
    if problems or report["errors"]:
        lines += ["", "审计失败与解析错误（保留）：", "", "```json", json.dumps({"errors":report["errors"], "runs":[{"label":r["label"],"strategy":r["strategy"],"status":r["status"],"error":r.get("error"),"failed_checks":r.get("failed_checks",[])} for r in problems]},ensure_ascii=False,indent=2), "```"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--partial", action="store_true", help="preview per-seed rows without publishing across-seed means")
    args = parser.parse_args()
    base = args.base.resolve()
    auditor = import_auditor(base)
    source_before = {str(Path(__file__).resolve()):auditor.sha(Path(__file__)),
                     str(base/"analyze.py"):auditor.sha(base/"analyze.py")}
    inputs, records, slots, errors = [], [], [], []
    for seed in SEEDS:
        for order in ORDERS:
            label = label_for(seed,order)
            manifest = base/"inputs"/(label+".json.gz")
            info = None
            if manifest.is_file():
                try:
                    info=auditor.load_input(manifest,base)
                    population_checks={"label_matches_expected":info["label"]==label,
                        "metadata_seed_matches_expected":info["metadata"].get("seed")==seed,
                        "request_population_19456":info["request_count"]==EXPECTED_REQUESTS,
                        "each_npu_expected_profile_quota":all(Counter((r["seq_len_k"],r["nql"]) for r in info["_requests"].values() if r["npu_id"]==n)==PROFILE_COUNTS_PER_NPU for n in range(32))}
                    info["paired_population_audit"]={"passed":all(population_checks.values()),"checks":population_checks}
                    inputs.append(info)
                except Exception as exc:
                    errors.append({"path":str(manifest),"kind":"input_error","error":f"{type(exc).__name__}: {exc}"})
            for strategy in STRATEGIES:
                paths=sorted((base/"runs"/label/strategy).glob("*.json.gz"))
                slot={"seed":seed,"order":order,"strategy":strategy,"label":label,"runs":[],"status":"pending"}
                if len(paths)>1: slot["status"]="duplicate"
                for path in paths:
                    r={"seed":seed,"order":order,"strategy":strategy,"label":label,
                       "path":auditor.relative(path,base),"file_sha256":auditor.sha(path),"status":"error"}
                    try:
                        if info is None: raise ValueError("missing or invalid matching input manifest")
                        raw=auditor.read_json(path)
                        r.update(input_fingerprint=raw.get("input_fingerprint"),submit_seed=raw.get("submit_seed"),
                                 core_source_hashes=raw.get("core_and_policy_sha256"),static_path_cirs=raw.get("static_path_cirs_gib_s"))
                        metrics,checks=independent_metrics(raw,info,auditor)
                        r["metrics"]=metrics
                        checks.update(policy_checks(raw,info))
                        checks.update({"input_"+k:v for k,v in info["paired_population_audit"]["checks"].items()})
                        r["independent_checks"]=checks
                        parent=auditor.analyze_result(path,info,base)
                        checks["window_U_agrees_with_parent_independent_auditor"]=auditor.close(metrics["device_utilization"],parent["windows"][0]["device_utilization"])
                        r["parent_audit"]=public_audit(parent)
                        r["audit_passed"]=parent["audit"]["passed"] and all(checks.values())
                        r["main_window_valid"]=parent["main_window_valid"]
                        scan=parent["nominal_demand_scan"]["full_run"]
                        r["full_run_nominal_capacity_satisfied"]=scan["all_ssu_within_capacity"] and scan["all_npu_links_within_capacity"]
                        r["all_study_conditions_met"]=r["audit_passed"] and r["main_window_valid"] and r["full_run_nominal_capacity_satisfied"]
                        r["failed_checks"]=[k for k,v in checks.items() if not v]+["parent:"+k for k,v in parent["audit"]["checks"].items() if not v]
                        r["status"]="complete" if r["all_study_conditions_met"] else "audit_failed" if not r["audit_passed"] else "condition_failed"
                    except Exception as exc:
                        r.update(error=f"{type(exc).__name__}: {exc}",audit_passed=False,
                                 all_study_conditions_met=False,status="error")
                        errors.append({"path":r["path"],"kind":"result_error","error":r["error"]})
                    records.append(r);slot["runs"].append(r)
                    print(json.dumps({"seed":seed,"order":order,"strategy":strategy,"status":r["status"],"completed":len(records)},ensure_ascii=False),flush=True)
                if len(paths)==1:slot["status"]=slot["runs"][0]["status"]
                slots.append(slot)
    parent_runs=[]
    for r in records:
        if "parent_audit" not in r:continue
        p={**r["parent_audit"],"all_subwindows_active_and_mixed":None}
        parent_runs.append(p)
    order_pairs=auditor.order_pair_audit(inputs,parent_runs,STRATEGIES)
    for pair in order_pairs:
        for p in pair["strategies"]:
            p.pop("both_all_subwindows_active_and_mixed",None)
    ppairs=policy_pairs(inputs,records)
    aliases=defaultdict(list)
    for info in inputs:
        aliases[info["input_fingerprint"]].append({"label":info["label"],"seed":info["metadata"].get("seed"),"manifest_sha256":info["file_sha256"]})
    unique_complete=all(len(s["runs"])==1 and "metrics" in s["runs"][0] and s["runs"][0]["status"]!="error" for s in slots)
    publish=unique_complete and not args.partial
    counts={"expected_slots":20,"parsed_slots":sum(len(s["runs"])==1 and "metrics" in s["runs"][0] for s in slots),
        "pending_slots":sum(not s["runs"] for s in slots),"duplicate_slots":sum(len(s["runs"])>1 for s in slots),
        "error_slots":sum(any(r["status"]=="error" for r in s["runs"]) for s in slots),
        "parsed_runs":sum("metrics" in r for r in records),"run_audits_passed":sum(r.get("audit_passed",False) for r in records),
        "study_conditions_met":sum(r.get("all_study_conditions_met",False) for r in records),
        "order_input_pairs_passed":sum(p["input_audit"]["passed"] for p in order_pairs),
        "policy_pairs_passed":sum(p["status"]=="passed" for p in ppairs)}
    sources={str(Path(k).relative_to(base)) if Path(k).is_relative_to(base) else k:v for k,v in source_before.items()}
    for name in ("plan.json","preregistration.json","measurement_amendment.json"):
        path=HERE/name
        if path.is_file():sources[auditor.relative(path,base)]=auditor.sha(path)
    report={"schema_version":1,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "seeds":list(SEEDS),"window_ms":[START,END],"slo_alpha":ALPHA,"counts":counts,
        "partial_requested":args.partial,"complete_grid":unique_complete,"means_published":publish,
        "all_study_conditions_met":unique_complete and counts["study_conditions_met"]==20 and counts["order_input_pairs_passed"]==5 and counts["policy_pairs_passed"]==10,
        "definitions":{"primary_slo":"admission-to-completion <= 1.5 * 8 * input per-layer pure compute, for admissions in [2000,4000); follow every request to completion",
            "primary_utilization":"sum exact compute interval overlap with [2000,4000) / (32*2000)",
            "seed_aggregation":"unweighted arithmetic mean of all prespecified seed rates; sample SD with n-1; min/max; never pool unequal warm cohorts",
            "invalid_run_policy":"retain numeric results of invalid runs, mark audit/condition failures, never silently discard; no aggregate from incomplete/duplicate grid",
            "ttft_scope":"8-layer prefill admission-time proxy, not arrival-based user TTFT and not a measured first-token event",
            "all_arrival_zero":True,"empty_cohort_rate":None,
            "random_sources":"seed changes per-NPU full-list random input shuffle and same-timestamp client submission permutation; ordered profile/placement template is fixed across seeds",
            "nominal_capacity":"current admitted request V_s/C per original NPU, exact same-time event batches; no extra cross-request L0 term; not released queue or deadline feasibility"},
        "source_paths_relative_to":"parent study directory (base)",
        "source_sha256":sources,"inputs":[public_input(i) for i in inputs],"slots":slots,"runs":records,
        "input_fingerprint_aliases":[{"input_fingerprint":fp,"manifests":values} for fp,values in aliases.items() if len(values)>1],
        "order_pair_audits":order_pairs,"policy_pair_audits":ppairs,"errors":errors,
        "aggregates_all_five_seeds":aggregate(records,SEEDS) if publish else None,
        "aggregates_excluding_seed7":aggregate(records,SEEDS[1:]) if publish else None}
    for path,digest in source_before.items():
        if auditor.sha(Path(path))!=digest:raise RuntimeError(f"analysis source changed during execution: {path}")
    flattened=[flatten(r) for r in records]+[flatten({**s,"status":"pending"}) for s in slots if not s["runs"]]
    flattened.sort(key=lambda r:(r["seed"],ORDERS.index(r["order"]),STRATEGIES.index(r["strategy"]),r.get("path") or ""))
    fields=list(dict.fromkeys(k for r in flattened for k in r))
    import io
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(flattened)
    atomic_text(HERE/"per_seed.csv",stream.getvalue())
    atomic_text(HERE/"summary.json",json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    atomic_text(HERE/"comparison.md",markdown(report))
    print(json.dumps({"counts":counts,"means_published":publish,"all_study_conditions_met":report["all_study_conditions_met"],"output":str(HERE)},ensure_ascii=False),flush=True)
    return 0 if args.partial or report["all_study_conditions_met"] else 2


if __name__=="__main__":
    raise SystemExit(main())
