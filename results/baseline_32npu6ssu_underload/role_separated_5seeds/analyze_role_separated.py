#!/usr/bin/env python3
"""Analyze only the frozen role-separated study; never alter parent results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
sys.path[:0] = [str(ROOT), str(PARENT)]


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auditor = import_path("role_parent_auditor", PARENT / "analyze.py")
auditor.WINDOWS = ((2000.0, 4000.0),)
paired = import_path("role_paired_metrics", PARENT / "paired_once_5seeds" / "analyze_paired.py")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stat(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def analyze_one(item, strategy, path, plan):
    cache = HERE / "analysis_cache" / item["label"] / f"{strategy}.json"
    signature = dict(result_sha256=auditor.sha(path), input_sha256=auditor.sha(item["manifest"]),
                     analyzer_sha256=auditor.sha(__file__), parent_analyzer_sha256=auditor.sha(PARENT / "analyze.py"),
                     paired_metrics_sha256=auditor.sha(PARENT / "paired_once_5seeds/analyze_paired.py"))
    if cache.exists():
        previous = auditor.read_json(cache)
        if previous.get("analysis_sources") == signature:
            return previous
    info = auditor.load_input(Path(item["manifest"]), HERE)
    technical = auditor.analyze_result(path, info, HERE)
    raw = auditor.read_json(path)
    metrics, checks = paired.independent_metrics(raw, info, auditor)
    checks.update(paired.policy_checks(raw, info))
    roles = info["metadata"]["assigned_role_by_npu"]
    req = info["_requests"]
    checks.update(
        plan_manifest_sha_matches=signature["input_sha256"] == item["manifest_sha256"],
        plan_input_fp_matches=info["input_fingerprint"] == item["input_fingerprint"],
        same_global_request_count=len(req) == 19456,
        profile_role_assignment=all(r["role"] == roles[r["npu_id"]] for r in req.values()),
        runtime_role_assignment=all(req[b["member_request_ids"][0]]["role"] == roles[b["npu_id"]]
                                    for b in raw["summary"]["microbatch_metrics"]),
        pure_compute_horizon=all(n["ideal_compute_ms"] > 4000 for n in info["per_npu"]),
        static_per_ssu_underload=info["static_max_proof"]["all_ssu_within_capacity"],
        static_link_underload=info["static_max_proof"]["all_npu_links_within_capacity"],
        frozen_construction_sources=all(auditor.sha(ROOT / name) == digest for name,digest in plan["source_sha256"].items()),
    )
    window = technical["windows"][0]
    metrics["card_groups"] = {}
    for role in ("short", "long"):
        cards = [n for n in window["per_npu"] if roles[n["npu_id"]] == role]
        metrics["card_groups"][role] = dict(
            card_count=len(cards),
            device_utilization=math.fsum(n["compute_ms"] for n in cards)/(len(cards)*2000),
            all_cards_active=all(auditor.close(n["active_ms"],2000) for n in cards),
            l0_exposed_stall_ms=math.fsum(n["l0_exposed_stall_ms"] for n in cards),
            l1_7_exposed_stall_ms=math.fsum(n["l1_7_exposed_stall_ms"] for n in cards),
        )
    scan = technical["nominal_demand_scan"]["full_run"]
    capacity = scan["all_ssu_within_capacity"] and scan["all_npu_links_within_capacity"]
    warm_valid = technical["warmup"]["all_fourth_completions_by_1500"] and window["all_npus_active"]
    audit_ok = technical["audit"]["passed"] and all(checks.values())
    metrics["window_request_equal_utilization"] = window["request_equal_utilization"]
    metrics["full_run_device_utilization"] = technical["full_run"]["device_utilization"]
    metrics["full_run_request_equal_admission_efficiency"] = technical["full_run"]["request_equal_admission_efficiency"]
    ideal_total = math.fsum(n["ideal_compute_ms"] for n in info["per_npu"])
    ideal_makespan_bound = max(n["ideal_compute_ms"] for n in info["per_npu"])
    metrics["full_population_ideal_max_card_compute_ms"] = ideal_makespan_bound
    metrics["full_population_device_utilization_upper_bound_from_binding"] = ideal_total/(32*ideal_makespan_bound)
    # The imported original main_window_valid requires mixed roles on each card.
    # Preserve that value only as an explicitly inapplicable diagnostic.
    original_mixed_flag = technical.pop("main_window_valid")
    technical.pop("all_subwindows_active_and_mixed", None)
    for w in technical["windows"]:
        w.pop("requests", None)
    record = dict(seed=item["seed"], long_npu_count=item["long_npu_count"], short_npu_count=item["short_npu_count"],
                  order=item["order"], strategy=strategy, label=item["label"], status="complete",
                  path=str(path.relative_to(HERE)), input_fingerprint=info["input_fingerprint"],
                  audit_passed=audit_ok, checks=checks, metrics=metrics,
                  main_window_valid=warm_valid and checks["runtime_role_assignment"],
                  full_run_nominal_capacity_satisfied=capacity,
                  all_study_conditions_met=audit_ok and warm_valid and capacity,
                  original_mixed_window_flag_inapplicable=original_mixed_flag,
                  validity_definition="four completions by 1500 ms, all 32 cards active in [2000,4000), assigned role only; no per-card mixing requirement",
                  static_max_proof=info["static_max_proof"], parent_technical_audit=technical,
                  analysis_sources=signature)
    write_json(cache, record)
    return record


def flatten(record):
    row = {k:record.get(k) for k in ("seed","long_npu_count","short_npu_count","order","strategy","label","status",
                                    "input_fingerprint","path","audit_passed","main_window_valid",
                                    "full_run_nominal_capacity_satisfied","all_study_conditions_met")}
    if "metrics" not in record:
        row["error"] = record.get("error", "")
        return row
    m = record["metrics"]
    row.update(device_utilization_percent=100*m["device_utilization"],
               short_card_utilization_percent=100*m["card_groups"]["short"]["device_utilization"],
               long_card_utilization_percent=100*m["card_groups"]["long"]["device_utilization"],
               window_request_equal_utilization_percent=100*m["window_request_equal_utilization"],
               makespan_ms=m["makespan_ms"], full_run_device_utilization_percent=100*m["full_run_device_utilization"],
               full_population_binding_U_upper_bound_percent=100*m["full_population_device_utilization_upper_bound_from_binding"])
    for cohort_name, prefix in (("window_admissions","warm"),("window_arrivals","warm_arrivals"),("all_requests","full")):
        c = m["cohorts"][cohort_name]
        row[f"{prefix}_count"] = c["count"]
        row[f"{prefix}_completion_after_4000_count"] = c["completion_after_window_end_count"]
        for clock in ("admission","arrival"):
            row[f"{prefix}_{clock}_passed"] = c[clock]["passed"]
            row[f"{prefix}_{clock}_slo_percent"] = None if c[clock]["rate"] is None else 100*c[clock]["rate"]
        for role in ("short","long"):
            rc = m["by_role"][role][cohort_name]
            row[f"{prefix}_{role}_count"] = rc["count"]
            row[f"{prefix}_{role}_admission_passed"] = rc["admission"]["passed"]
            row[f"{prefix}_{role}_admission_slo_percent"] = None if rc["admission"]["rate"] is None else 100*rc["admission"]["rate"]
    for role in ("short","long"):
        for kind in ("l0_exposed_stall_ms","l1_7_exposed_stall_ms"):
            row[f"{role}_{kind}"] = m["card_groups"][role][kind]
    t = record["parent_technical_audit"]
    row.update(max_fourth_completion_ms=max(t["warmup"]["fourth_completion_by_npu_ms"]),
               full_run_max_ssu_gib_s=t["nominal_demand_scan"]["full_run"]["max_ssu_gib_s"],
               full_run_any_ssu_over_capacity_ms=t["nominal_demand_scan"]["full_run"]["any_ssu_over_capacity_ms"],
               static_max_ssu_gib_s=record["static_max_proof"]["max_ssu_gib_s"])
    return row


MEAN_FIELDS = (
    "device_utilization_percent", "short_card_utilization_percent", "long_card_utilization_percent",
    "warm_admission_slo_percent", "warm_short_admission_slo_percent", "warm_long_admission_slo_percent",
    "warm_count", "warm_short_count", "warm_long_count", "warm_completion_after_4000_count",
    "full_admission_slo_percent", "full_short_admission_slo_percent", "full_long_admission_slo_percent",
    "full_arrival_slo_percent", "makespan_ms", "full_run_device_utilization_percent",
    "full_population_binding_U_upper_bound_percent", "window_request_equal_utilization_percent",
    "short_l0_exposed_stall_ms", "short_l1_7_exposed_stall_ms",
    "long_l0_exposed_stall_ms", "long_l1_7_exposed_stall_ms",
)


def aggregate(rows, plan):
    result = []
    for n_long in plan["long_npu_counts"]:
        for order in plan["orders"]:
            for strategy in plan["strategies"]:
                members = [r for r in rows if (r["long_npu_count"],r["order"],r["strategy"]) == (n_long,order,strategy)]
                assert len(members)==5 and {r["seed"] for r in members} == set(plan["seeds"])
                summary = dict(long_npu_count=n_long, short_npu_count=32-n_long, order=order, strategy=strategy,
                               seeds=plan["seeds"], n_seeds=5, all_conditions_count=sum(r["all_study_conditions_met"] for r in members))
                for field in MEAN_FIELDS:
                    values = [r[field] for r in members]
                    summary[field] = stat(values) if all(x is not None for x in values) else None
                result.append(summary)
    return result


def paired_contrasts(rows, plan):
    by_key={(r["seed"],r["long_npu_count"],r["order"],r["strategy"]):r for r in rows}
    differences=[]
    fields=MEAN_FIELDS[:6]
    for n_long in plan["long_npu_counts"]:
        for seed in plan["seeds"]:
            pairs=[]
            for strategy in plan["strategies"]:
                pairs.append(("round_robin_minus_random", strategy,
                              by_key[(seed,n_long,"random",strategy)],
                              by_key[(seed,n_long,"round_robin",strategy)]))
            for order in plan["orders"]:
                pairs.append(("once_minus_baseline", order,
                              by_key[(seed,n_long,order,"baseline")],
                              by_key[(seed,n_long,order,"once")]))
            for kind,context,left,right in pairs:
                d=dict(seed=seed,long_npu_count=n_long,contrast=kind,context=context,
                       both_valid=left["all_study_conditions_met"] and right["all_study_conditions_met"])
                for field in fields:
                    d[field+"_delta_pp"]=right[field]-left[field] if left[field] is not None and right[field] is not None else None
                differences.append(d)
    summaries=[]
    for key in sorted({(r["long_npu_count"],r["contrast"],r["context"]) for r in differences}):
        members=[r for r in differences if (r["long_npu_count"],r["contrast"],r["context"])==key]
        s=dict(long_npu_count=key[0],contrast=key[1],context=key[2],n_seeds=5,
               valid_pairs=sum(r["both_valid"] for r in members))
        for field in fields:
            values=[r[field+"_delta_pp"] for r in members]
            s[field+"_delta_pp"]=stat(values) if all(x is not None for x in values) else None
        summaries.append(s)
    return differences,summaries


def write_report(rows, groups, payload):
    complete = sum(r["status"] == "complete" for r in rows)
    text = ["# 原人口固定长短卡的追加实验", "",
            f"完成 {complete}/40 格。主指标为固定 [2000,4000) ms 的真实设备计算利用率；主 SLO 为该窗口内 admission 的请求，completion−admission ≤ 1.5×8C，每条跟到完成，即使完成晚于4000 ms。",
            "所有真实 arrival 均为0；暖窗 arrival cohort 为空，admission TTFT 是排除入卡前排队的处理时间代理。两种内部顺序的暖窗 admission 集合可能不同；完整19456请求配对结果另列。", "",
            "原人口保持256条192K/768长请求与三种1K短画像各6400条。长画像为原data在NQL512/1024间插值，短画像为原data的1K外推；本追加没有修改任何请求C/V。重新绑定会改变原混合实验的每卡人口；只在相同比例与seed内部比较random和round_robin时，每新卡人口保持严格相同。原请求placement逐块保留，未按新卡重新striping。", "",
            "11长/21短与6长/26短的任意内部顺序逐盘current V/C上界分别为31.735895和26.559836 GiB/s，均低于40。该证书约束当前请求名义速率，不保证每次突发I/O的短deadline可行。最少每卡纯C分别为4712.671292与4181.230813 ms，可排除4秒前人口耗尽。", "",
            "有效性要求：源码/输入/策略审计通过，每卡1500 ms前完成至少4请求，暖窗32卡全active且角色绑定正确，完整运行逐事件盘/链路名义需求欠载。此干预明确不要求每卡同时有长短画像。", ""]
    if groups:
        text += ["五个seed等权平均 ± 样本标准差；没有把请求池化计算跨seed均值。U和SLO均为%。", "",
                 "|长/短卡|短卡顺序|策略|设备U|短卡U|长卡U|暖窗SLO|短SLO|长SLO|有效格|",
                 "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        def fmt(x):
            return "NA" if x is None else f"{x['mean']:.3f} ± {x['sample_sd']:.3f}"
        for g in groups:
            values=[fmt(g[f]) for f in MEAN_FIELDS[:6]]
            text.append(f"|{g['long_npu_count']}/{g['short_npu_count']}|{g['order']}|{g['strategy']}|"+"|".join(values)+f"|{g['all_conditions_count']}/5|")
        text += ["", "配对差值先在同seed内相减，再汇总五seed；单位为百分点。", "",
                 "|长卡数|差值方向|固定条件|设备U差值|暖窗SLO差值|有效配对|",
                 "|---:|---|---|---:|---:|---:|"]
        for c in payload["paired_contrasts"]:
            text.append(f"|{c['long_npu_count']}|{c['contrast']}|{c['context']}|{fmt(c['device_utilization_percent_delta_pp'])}|{fmt(c['warm_admission_slo_percent_delta_pp'])}|{c['valid_pairs']}/5|")
        text += ["", "完整人口附录：同一19456条请求，无暖窗截断。6长卡下最长纯C卡有43条长请求，即8810.646328 ms；即使没有任何I/O等待，绑定不均衡也限制完整运行设备U，因此这里的低完整U不能直接解释为FIFO损失。", "",
                 "|长/短卡|短卡顺序|策略|完整设备U|完整admission SLO|完整arrival SLO|makespan ms|绑定给出的完整U上界|",
                 "|---|---|---|---:|---:|---:|---:|---:|"]
        for g in groups:
            values=[fmt(g[f]) for f in ("full_run_device_utilization_percent","full_admission_slo_percent","full_arrival_slo_percent","makespan_ms","full_population_binding_U_upper_bound_percent")]
            text.append(f"|{g['long_npu_count']}/{g['short_npu_count']}|{g['order']}|{g['strategy']}|"+"|".join(values)+"|")
    else:
        text += ["尚未收齐40格；不发布五seed均值。下列逐格仅为进度，不能据此替代最终组统计。"]
    text += ["", "|seed|长/短卡|顺序|策略|状态|设备U %|暖admission SLO %|短卡U %|长卡U %|有效|",
             "|---:|---|---|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        def val(k):
            x=r.get(k)
            return "NA" if x is None else f"{x:.4f}"
        text.append(f"|{r['seed']}|{r['long_npu_count']}/{r['short_npu_count']}|{r['order']}|{r['strategy']}|{r['status']}|{val('device_utilization_percent')}|{val('warm_admission_slo_percent')}|{val('short_card_utilization_percent')}|{val('long_card_utilization_percent')}|{r.get('all_study_conditions_met','')}|")
    text += ["", "原始记录：plan.json、inputs/、runs/*/*/command.json；逐seed数据：per_seed.csv；聚合：aggregate.csv、summary.json。独立输入与结果复核见audit/。", "",
             "只测试random与round_robin两种短卡顺序。即使二者都低，也不能推出任意内部排列都低；固定角色后长卡画像相同，顺序变动主要影响短卡内部相位。"]
    (HERE / "comparison.md").write_text("\n".join(text)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args=parser.parse_args()
    plan=auditor.read_json(HERE / "plan.json")
    assert all(auditor.sha(ROOT / name)==digest for name,digest in plan["source_sha256"].items()), "frozen simulation/construction source changed"
    inputs={r["label"]:r for r in plan["inputs"]}
    records=[]
    for job in plan["jobs"]:
        item=inputs[job["label"]]
        strategy=job["strategy"]
        directory=HERE / "runs" / item["label"] / strategy
        paths=list(directory.glob("*.json.gz"))
        if not paths:
            record={**{k:item[k] for k in ("seed","long_npu_count","short_npu_count","order","label")},
                    "strategy":strategy,"status":"pending"}
            command=directory / "command.json"
            if command.exists():
                status=auditor.read_json(command).get("status")
                if status in ("failed","timeout"):
                    record["status"]=status
        else:
            try:
                assert len(paths)==1, "duplicate raw result"
                record=analyze_one(item,strategy,paths[0],plan)
            except Exception as exc:
                record={**{k:item[k] for k in ("seed","long_npu_count","short_npu_count","order","label")},
                        "strategy":strategy,"status":"analysis_error","error":repr(exc)}
        records.append(record)
        if record["status"] != "pending":
            print(json.dumps(dict(label=item["label"],strategy=strategy,status=record["status"],
                                  valid=record.get("all_study_conditions_met"),
                                  U=record.get("metrics",{}).get("device_utilization"))),flush=True)
    rows=[flatten(r) for r in records]
    complete=all(r["status"]=="complete" for r in records)
    groups=aggregate(rows,plan) if complete else []
    differences,contrasts=paired_contrasts(rows,plan) if complete else ([],[])
    payload=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=auditor.sha(HERE/"plan.json"),
                 expected_count=40,complete_count=sum(r["status"]=="complete" for r in records),
                 all_complete=complete,all_study_conditions_met=complete and all(r["all_study_conditions_met"] for r in records),
                 seed_weighting="Arithmetic mean of the five per-seed metrics; sample SD uses n-1. No pooling of warm-admission counts.",
                 records=records,groups=groups,paired_contrasts=contrasts)
    write_json(HERE / "summary.json",payload)
    csv_write(HERE / "per_seed.csv",rows)
    if groups:
        csv_write(HERE / "contrasts_per_seed.csv",differences)
        flatgroups=[]
        for g in groups:
            f={k:v for k,v in g.items() if k not in MEAN_FIELDS}
            for metric in MEAN_FIELDS:
                if g[metric] is not None:
                    for name,value in g[metric].items():f[f"{metric}_{name}"]=value
            flatgroups.append(f)
        csv_write(HERE / "aggregate.csv",flatgroups)
    write_report(rows,groups,payload)
    assert all(auditor.sha(ROOT / name)==digest for name,digest in plan["source_sha256"].items()), "frozen source changed during analysis"
    print(json.dumps({k:payload[k] for k in ("expected_count","complete_count","all_complete","all_study_conditions_met")}))
    if not args.partial and not complete:
        raise SystemExit("Incomplete matrix; use --partial for progress output")
    if not args.partial and not payload["all_study_conditions_met"]:
        raise SystemExit("Complete matrix contains failed study conditions; retained in report")


if __name__ == "__main__":
    main()
