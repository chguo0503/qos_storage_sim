#!/usr/bin/env python3
"""Audit frozen raw quartets and analyze their planned 22 results; no simulations.

Writes only comparison.md, summary.json, per_seed.csv, and audit.json here.
--partial retains all statuses and suppresses across-seed statistics. A complete
run keeps scientifically invalid warm windows in the means and labels them.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(BASE)]
SEEDS = (7, 19, 43, 67, 101)
START, END = 2000.0, 4000.0
IO_GIB = 176 * 1024 / 2**30
SPECS = {
    "raw_feasible_q2": (((32, 1024), (48, 1024), (64, 1024), (160, 1024)), (2, 2, 2, 1)),
    "raw_nearest_q25": (((32, 128), (32, 256), (32, 512), (192, 1024)), (25, 25, 25, 1)),
}


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def atomic(path, text):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    temporary.replace(path)


def write_json(path, value):
    atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def science(row, placements):
    return {"npu_id": row["npu_id"], "arrival_time_ms": row["arrival_time_ms"],
            "load": {k: v for k, v in row["load"].items() if k not in ("request_id", "generation", "original_request_id")},
            "placement": placements[row["placement_index"]]}


def audit_input(item, table, auditor):
    path = Path(item["manifest"])
    raw = read(path)
    meta = raw["metadata"]
    require(item["order"] in ("random", "ordered"), "Unplanned input order")
    require(item["seed"] in (SEEDS if item["family"] == "raw_feasible_q2" else (7,)), "Unplanned input seed")
    keys, quotas = SPECS[item["family"]]
    costs = [table[key][1]/1000 for key in keys]
    packet_c = 8*math.fsum(q*c for q, c in zip(quotas, costs))
    needed = 4000+16*max(costs)
    cycles = math.ceil(needed/packet_c)
    counts = [q*cycles for q in quotas]
    require(sha(path) == item["manifest_sha256"], "Manifest differs from frozen bytes")
    require((meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (32, 6, 8), "Wrong topology/layers")
    require(meta["label"] == item["label"] and meta["seed"] == item["seed"] and meta["quartet_family"] == item["family"], "Wrong manifest identity")
    require(meta["data_sha256"] == sha(ROOT/"data"), "Data source changed")
    require(meta["profile_quotas_per_unit"] == list(quotas) and meta["profile_counts_per_npu"] == counts, "Changed full quotas")
    require(meta["quota_units_per_npu"] == cycles and meta["requests_per_npu"] == sum(counts), "Wrong finite population rule")
    require(len(raw["requests"]) == 32*sum(counts) == item["request_count"], "Wrong total request count")
    require(meta["compute_scale_actual"] == 1.0, "Compute scaling applied")
    key_to_index = {key: i for i, key in enumerate(keys)}
    lanes = {n: [] for n in range(32)}
    science_by_id = {}
    for r in raw["requests"]:
        load = r["load"]
        n, rid = r["npu_id"], r["request_id"]
        require(n in lanes and rid not in science_by_id, "Foreign NPU or duplicate request ID")
        key = (load["seq_len_k"], load["nql"])
        require(key in key_to_index, "Foreign profile")
        bw, compute, ttft, volume = table[key]
        require(load["per_layer_us"] == compute and load["original_compute_us"] == compute, "C not the original raw row")
        require(load["per_layer_kv_gb"] == volume and load["required_bw_input_gbps"] == bw and load["source_ttft_ms"] == ttft, "Raw V/bandwidth/source TTFT changed")
        require(load["constructed_profile"] is False and load["profile_construction"]["method"] == "direct_data_row", "Constructed profile marked as raw")
        require(load["padding_gib_per_layer"] == 0, "Padded volume")
        category = ("S" if key[0] <= 80 else "L")+("S" if key[1] < 512 else "L")
        require(load["category"] == category and load["role"] == ("short" if key_to_index[key] < 3 else "long"), "Wrong policy category or analysis role")
        require(r["arrival_time_ms"] == 0 and load["npu_id"] == n and load["request_id"] == rid, "Arrival/NPU/ID mismatch")
        blocks = (key[0]*1024-key[1])//128
        require((key[0]*1024-key[1]) % 128 == 0, "Nonaligned source prefix")
        expected = [[(j+n//4)%6, IO_GIB] for j in range(blocks)]
        placement = raw["placements"][r["placement_index"]]
        require(len(placement) in (1, 8) and all(layer == expected for layer in placement), "Original physical 176KiB stripe changed")
        require(math.isclose(blocks*IO_GIB, volume, rel_tol=1e-12, abs_tol=1e-12), "Exact block volume disagrees with raw data")
        science_by_id[rid] = digest(science(r, raw["placements"]))
        lanes[n].append((rid, key_to_index[key], load["generation"]))
    phases = []
    for n, lane in lanes.items():
        lane.sort()
        require([x[0] for x in lane] == [n*1_000_000+i for i in range(sum(counts))], "Queue position IDs differ")
        require([x[2] for x in lane] == list(range(sum(counts))), "Generation differs from position")
        require(Counter(x[1] for x in lane) == dict(enumerate(counts)), "Per-card population differs")
        if item["order"] == "random":
            deck = [i for i, q in enumerate(quotas) for _ in range(q)]*cycles
            random.Random(item["seed"]+n*100003).shuffle(deck)
        else:
            packet = [3]+[i for turn in range(max(quotas[:3])) for i in range(3) if turn < quotas[i]]
            boundaries = [0.0]
            for i in packet:
                boundaries.append(boundaries[-1]+costs[i])
            target = (n//8)/4*boundaries[-1]
            cut = min(range(len(boundaries)), key=lambda k: (abs(boundaries[k]-target), k))
            deck = packet*cycles
            deck = deck[cut:]+deck[:cut]
            recorded = meta["per_npu_order_design"][n]
            require(recorded["phase_boundary_request_count"] == cut and recorded["period_packets"] == 1 and recorded["mode"] == "exact_cohort4", "Recorded cohort construction differs")
            phases.append({"npu_id": n, "group": n//8, "cut": cut, "pure_compute_phase_ms": 8*boundaries[cut]})
        require([x[1] for x in lane] == deck, "Random deck or exact cohort profile order differs")
    info = auditor.load_input(path, HERE)
    require(info["input_fingerprint"] == item["input_fingerprint"] == raw["input_fingerprint"], "Recomputed fingerprint differs")
    proof = info["static_max_proof"]
    require(all(math.isclose(x, y, rel_tol=0, abs_tol=1e-10) for x,y in zip(proof["per_ssu_gib_s"], meta["active_profile_rate_certificate"]["per_ssu_upper_bound_gib_s"])), "Static certificate differs")
    short = math.fsum(q*c for q,c in zip(quotas[:3],costs[:3]))/(packet_c/8)
    require(math.isclose(short, meta["short_pure_compute_share"], abs_tol=1e-12), "Wrong short pure-C share")
    require(min(n["ideal_compute_ms"] for n in info["per_npu"]) > 4000, "Input can exhaust before warm window ends")
    summary = {**item, "status": "passed", "profile_counts_per_npu": counts, "cycles": cycles,
               "packet_compute_ms": packet_c, "required_compute_ms": needed, "per_npu_compute_ms": cycles*packet_c,
               "short_pure_compute_share": short, "static_max_proof": proof, "phase_design": phases,
               "raw_profiles": [{"key": key, "C_ms": table[key][1]/1000, "V_MiB": table[key][3]*1024,
                   "rate_gib_s": table[key][0], "category": ("S" if key[0]<=80 else "L")+("S" if key[1]<512 else "L")} for key in keys],
               "minimum_nominal_total_when_32_active_gib_s": 32*min(table[key][0] for key in keys)}
    return info, summary, raw, science_by_id


def audit_order_pairs(inputs, payloads):
    rows = []
    for item in inputs:
        if item["order"] != "ordered":
            continue
        ordered, ohashes = payloads[item["label"]]
        reference_label = ordered["metadata"]["random_reference_label"]
        reference, rhashes = payloads[reference_label]
        mapping = [r["load"]["original_request_id"] for r in ordered["requests"]]
        checks = {"complete_identity_bijection": len(mapping) == len(set(mapping)) == len(rhashes) and set(mapping) == set(rhashes),
                  "every_scientific_field_NPU_and_block_placement_preserved": all(ohashes[r["request_id"]] == rhashes.get(r["load"]["original_request_id"]) for r in ordered["requests"]),
                  "reference_fingerprint_matches": ordered["metadata"]["random_reference_input_fingerprint"] == reference["input_fingerprint"],
                  "reference_bytes_match": ordered["metadata"]["random_reference_manifest_sha256"] == sha(HERE/"inputs"/(reference_label+".json.gz"))}
        rows.append({"family": item["family"], "seed": item["seed"], "random_label": reference_label,
                     "ordered_label": item["label"], "status": "passed" if all(checks.values()) else "audit_failed", "checks": checks})
    return rows


def analyze_run(job, info, plan, auditor, paired):
    item, strategy = job["input"], job["strategy"]
    record = {**{k:item[k] for k in ("family", "seed", "order", "label", "input_fingerprint")}, "strategy": strategy}
    directory = HERE/"runs"/item["label"]/strategy
    outputs = [p for p in directory.glob("*.json.gz") if ".failure." not in p.name]
    command = read(directory/"command.json") if (directory/"command.json").exists() else {}
    if len(outputs) != 1 or command.get("status") != "complete":
        record.update(status="duplicate_results" if len(outputs)>1 else command.get("status", "running" if command else "pending"),
                      result_files=[str(p) for p in outputs], failure_artifacts=[str(p) for p in directory.glob("*.failure*")])
        return record
    path = outputs[0]
    raw = read(path)
    parent = auditor.analyze_result(path, info, HERE)
    # The generic metric implementation is independent of population size; its
    # original-study constant is explicitly replaced for this new population.
    paired.EXPECTED_REQUESTS = len(info["_requests"])
    metrics, checks = paired.independent_metrics(raw, info, auditor)
    checks["request_metric_population_matches_new_manifest"] = checks.pop("request_metric_population_19456")
    checks.update(parent["audit"]["checks"])
    checks.update(paired.policy_checks(raw, info))
    from run_coflow_experiments import source_files
    checks["complete_core29_keyset"] = set(raw.get("core_and_policy_sha256", {})) == set(source_files()) and len(source_files()) == 29
    checks["frozen_core_values_match_plan"] = all(plan["source_sha256"].get(k) == v for k,v in raw.get("core_and_policy_sha256", {}).items())
    checks["fixed_policy_arguments"] = raw.get("policy_config") == {"assignment":"fixed", "joint_rule":"urgent_short", "queue_window_ms":1.0}
    checks["cir_interval_100ms"] = raw.get("cir_min_interval_ms") == 100.0
    checks["stress_runner_hash_frozen"] = raw.get("stress_runner_sha256") == plan["source_sha256"]["run_baseline_npu32_stress.py"]
    argv = command.get("command", [])
    options = {argv[i]:argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}
    checks["completed_matching_command"] = (command.get("returncode") == 0 and command.get("label") == item["label"] and command.get("strategy") == strategy and command.get("input_fingerprint") == item["input_fingerprint"])
    checks["command_manifest_policy_window"] = (Path(options.get("--manifest", "")).resolve() == Path(item["manifest"]).resolve() and options.get("--strategy") == strategy and options.get("--assignment") == "fixed" and options.get("--window") == "2000:4000")
    window = parent["windows"][0]
    mixed_cards = [n["npu_id"] for n in window["per_npu"] if all(n["by_role"][r]["compute_ms"] > 0 for r in ("short","long"))]
    scan = parent["nominal_demand_scan"]["full_run"]
    capacity = scan["max_ssu_gib_s"] < 40 and scan["max_npu_link_gib_s"] < 50
    warm = parent["warmup"]["all_fourth_completions_by_1500"] and window["all_npus_active"] and len(mixed_cards) == 32
    technical = all(checks.values())
    # Unmet scientific conditions are retained as complete results, not errors.
    record.update(status="complete", path=str(path.relative_to(HERE)), file_sha256=sha(path),
                  audit_passed=technical, checks=checks, metrics=metrics, parent_audit=paired.public_audit(parent),
                  main_window_valid=warm, warm_mixed_card_count=len(mixed_cards), warm_nonmixed_npu_ids=sorted(set(range(32))-set(mixed_cards)),
                  full_run_nominal_capacity_satisfied=capacity, all_study_conditions_met=technical and warm and capacity,
                  core_source_hashes=raw["core_and_policy_sha256"], static_path_cirs=raw["static_path_cirs_gib_s"], submit_seed=raw["submit_seed"])
    return record


def contrasts(records):
    by_key = {(r["seed"], r["order"], r["strategy"]):r for r in records}
    rows = []
    for seed in SEEDS:
        for strategy in ("baseline", "once"):
            a,b = [by_key[(seed,order,strategy)] for order in ("random","ordered")]
            u,v = a["metrics"]["device_utilization"], b["metrics"]["device_utilization"]
            sa = a["metrics"]["cohorts"]["window_admissions"]["admission"]["rate"]
            sb = b["metrics"]["cohorts"]["window_admissions"]["admission"]["rate"]
            rows.append({"seed":seed, "strategy":strategy, "ordered_minus_random_U_pp":100*(v-u),
                         "relative_U_reduction_vs_random_percent":100*(u-v)/u,
                         "ordered_minus_random_warm_SLO_pp":None if sa is None or sb is None else 100*(sb-sa),
                         "both_meet_study_conditions":a["all_study_conditions_met"] and b["all_study_conditions_met"]})
    return rows


def report_text(summary, input_audit, paired):
    counts = summary["counts"]
    text = ["# 原始四画像替换：独立输入与结果审计", "",
        f"已完成 {counts['complete']}/22 格，技术审计通过 {counts['technical_passed']} 格。主组 raw_feasible_q2 为预定五 seed × random/ordered × Baseline/Once，共20格；raw_nearest_q25 仅 seed7 ordered 两策略，是过载诊断。其 random manifest 只作排列基准，未计划运行，不计为缺失实验。", "",
        "32 NPU、6 SSU、每请求8层；固定暖窗[2000,4000) ms。设备U=真实计算区间与暖窗交集总时长/(32×2000)，请求等权U另列。active包括计算与接纳后的I/O等待，接纳前排队不是I/O stall。",
        "主SLO人口为暖窗内接纳的请求，逐条跟到完成；completion−admission ≤1.5×8C。这里是接纳后处理时间的TTFT代理，不含入卡前排队，也不使用原始数据的78层TTFT。所有arrival=0，因此暖窗arrival人口为空；完整输入人口结果另存CSV/JSON。不同顺序/策略的暖窗接纳人口可能不同。", "",
        "主组每卡[14,14,14,7]共49请求、全局1568，纯C4949.368761 ms；packet=L+(S1 S2 S3)×2，纯C707.052680 ms、7 cycles。旧组每卡[200,200,200,8]共608请求：本组把每卡long数从8改为7，并改变short配额与总人口，不能声称两组是同一请求集合。短类纯C份额67.490889%，旧构造为67.467124%，差0.023766个百分点。原始C/V均未缩放、未填充；只在新组的random/ordered之间逐块保留原NPU的176KiB striping。",
        "主组任意当前画像组合的逐盘名义V/C上界为39.630697580 GiB/s <40。诊断组四画像最小V/C=7.520933953，因此32卡active时总量至少240.669886488>240；任何排序都不可能满足原六盘欠载约束。这些是当前已接纳画像的名义需求，不是实际吞吐、突发I/O到达包络或deadline保证；跨请求L0不另叠加到该名义定义中。诊断组画像依次为32K/128、32K/256、32K/512、192K/1024，类别是SS/SS/SL/LL：第三短画像也已跨入SL，不能暗示最近原始键保留了原SS/LL分类。", "",
        "本组是原始画像下的外部有效性对照，不是只移除外推的消融：短C增至旧值13.75–14.26倍，短V增至35.43–100.8倍；long/short C比由28.48–48.52降至2.276–3.959。长类占字节比例由77.27%降至36.05%，纯C加权rho6由0.562846升至0.893343。旧三短为SS，新三短为SL，Once合法Path池由96缩至32；LL仍32，Baseline仍Path0。5ms快照相对短C也从旧5.56–9.47倍变成0.40–0.69倍。", "",
        "exact_cohort4_p1沿用同一构造规则，新实际切点[0,1,3,5]，纯C相位[0,229.856539,367.447084,526.512462] ms。组内8卡使用人工同步模板，不代表真实业务到达分布；五个ordered manifest的模拟输入指纹相同，不能称为五种独立坏顺序。随机组为五种完整人口独立shuffle。新短类别仍叫short仅为分析角色，其实际单层C为7–13ms。", "",
        "有效性逐格报告：技术通过、第四请求在1500ms前完成、暖窗32卡全active、每卡都有short/long计算、全程逐事件盘/链路名义欠载。未满足warm mixed的seed也保留；不重采样、不按有效性筛选均值。每卡纯C>4s能排除提前跑空，不能保证暖窗出现long。", ""]
    if summary["means_published"]:
        by_group={(g["order"],g["strategy"]):g for g in summary["aggregates_all_five_seeds"]}
        rb=by_group[("random","baseline")]["device_utilization"]["mean"]
        ob=by_group[("ordered","baseline")]["device_utilization"]["mean"]
        reductions=[r["relative_U_reduction_vs_random_percent"] for r in summary["paired_order_contrasts"] if r["strategy"]=="baseline"]
        text[4:4]=[f"主组结果：Baseline五seed平均设备U由random的{100*rb:.4f}%变为ordered的{100*ob:.4f}%；ordered−random为{100*(ob-rb):.4f}个百分点，逐seed相对下降的均值为{statistics.mean(reductions):.4f}%。本组通过全部研究条件的运行是{summary['main_valid_count']}/20，全部预定seed均纳入统计，不按有效性筛选。", ""]
        text += ["五seed等权均值 ± 样本标准差（n−1），包含全部预定seed，条件不通过也不剔除。", "",
                 "|顺序|策略|设备U %|暖接纳SLO %|全约束通过|", "|---|---|---:|---:|---:|"]
        def percent_stat(x):
            return "NA" if x["mean"] is None else f"{100*x['mean']:.4f} ± {100*x['sample_sd']:.4f}"
        for g in summary["aggregates_all_five_seeds"]:
            text.append(f"|{g['order']}|{g['strategy']}|{percent_stat(g['device_utilization'])}|{percent_stat(g['cohorts']['window_admissions']['admission'])}|{g['study_conditions_met_count']}/5|")
    else:
        text += ["当前为partial或结果未齐，不发布部分seed的均值替代五seed统计。"]
    text += ["", "逐格结果（诊断组不与主组混合）：", "",
             "|组|seed|顺序|策略|状态|设备U %|暖接纳SLO %|暖混合卡|全active|第四完成≤1500|全程欠载|",
             "|---|---:|---|---|---|---:|---:|---:|---|---|---|"]
    for r in summary["records"]:
        if "metrics" not in r:
            text.append(f"|{r['family']}|{r['seed']}|{r['order']}|{r['strategy']}|{r['status']}|—|—|—|—|—|—|")
            continue
        p=r["parent_audit"]; w=p["windows"][0]; c=r["metrics"]["cohorts"]["window_admissions"]["admission"]
        slo="NA" if c["rate"] is None else f"{100*c['rate']:.4f} ({c['passed']}/{c['count']})"
        text.append(f"|{r['family']}|{r['seed']}|{r['order']}|{r['strategy']}|{'complete' if r['audit_passed'] else 'AUDIT FAILED'}|{100*r['metrics']['device_utilization']:.4f}|{slo}|{r['warm_mixed_card_count']}/32|{w['all_npus_active']}|{p['warmup']['all_fourth_completions_by_1500']}|{r['full_run_nominal_capacity_satisfied']}|")
    text += ["", "所有输入与身份置换证据见audit.json；逐请求/层的来源位于runs，主表及全部人口、角色、类别SLO见summary.json与per_seed.csv。mean/SD用fraction保存于JSON，用明确percent字段保存于CSV。名义欠载不证明FIFO等待的唯一原因；任何机制判断仍需结合真实层时序及队列证据。", ""]
    return "\n".join(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args=parser.parse_args()
    require(not(args.partial and args.require_complete), "Choose partial or require-complete")
    auditor=module("quartet_parent_auditor", BASE/"analyze.py")
    auditor.WINDOWS=((START,END),)
    paired=module("quartet_paired_metrics", BASE/"paired_once_5seeds/analyze_paired.py")
    plan=read(HERE/"plan.json")
    table=ast.literal_eval((ROOT/"data").read_text())
    require(plan["seeds_for_feasible"]==list(SEEDS) and plan["window_ms"]==[2000,4000], "Plan seeds/window differ")
    require(plan["strategies"]==["baseline","once"] and len(plan["inputs"])==12 and len(plan["jobs"])==22, "Plan grid differs")
    require(plan["runner_sha256"]==sha(HERE/"run_quartet.py"), "Frozen constructor changed")
    require(all(sha(ROOT/name)==expected for name,expected in plan["source_sha256"].items()), "Frozen source changed")
    expected={(i["label"],s) for i in plan["inputs"] if i["family"]=="raw_feasible_q2" or i["order"]=="ordered" for s in ("baseline","once")}
    require(len(expected)==22 and len({(j["input"]["label"],j["strategy"]) for j in plan["jobs"]})==22 and expected=={(j["input"]["label"],j["strategy"]) for j in plan["jobs"]}, "Unplanned or duplicate slots")
    by_label={i["label"]:i for i in plan["inputs"]}
    require(all(j["input"] == by_label[j["input"]["label"]] for j in plan["jobs"]), "Job input differs from frozen input list")
    infos, inputs, payloads, failures = {}, [], {}, []
    for item in plan["inputs"]:
        try:
            info,row,raw,hashes=audit_input(item,table,auditor)
            infos[item["label"]]=info; inputs.append(row); payloads[item["label"]]=(raw,hashes)
            print(json.dumps({"input":item["label"],"status":"passed"}),flush=True)
        except Exception as error:
            failures.append({"label":item["label"],"error":str(error),"traceback":traceback.format_exc()})
    order_pairs=audit_order_pairs(plan["inputs"],payloads) if not failures else []
    del payloads
    records=[]
    for job in plan["jobs"]:
        try:
            record=analyze_run(job,infos[job["input"]["label"]],plan,auditor,paired)
        except Exception as error:
            record={**{k:job["input"][k] for k in ("family","seed","order","label")},"strategy":job["strategy"],"status":"analysis_error","error":str(error),"traceback":traceback.format_exc()}
        records.append(record)
        print(json.dumps({k:record.get(k) for k in ("label","strategy","status","audit_passed","warm_mixed_card_count","all_study_conditions_met")}),flush=True)
    complete=all(r["status"]=="complete" for r in records)
    policy_pairs=paired.policy_pairs([infos[i["label"]] for i in plan["inputs"] if (i["label"],"baseline") in expected and i["label"] in infos],records)
    technical=(not failures and len(order_pairs)==6 and all(p["status"]=="passed" for p in order_pairs)
               and all(r.get("audit_passed") is True for r in records)
               and len(policy_pairs)==11 and all(p["status"]=="passed" for p in policy_pairs))
    publish=not args.partial and complete and technical
    mains=[r for r in records if r["family"]=="raw_feasible_q2"]
    groups=paired.aggregate(mains,SEEDS) if publish else []
    same_fp={order:len({i["input_fingerprint"] for i in inputs if i["family"]=="raw_feasible_q2" and i["order"]==order}) for order in ("random","ordered")}
    audit={"created_utc":datetime.now(timezone.utc).isoformat(),"analyzer_sha256":sha(__file__),"plan_sha256":sha(HERE/"plan.json"),"data_sha256":sha(ROOT/"data"),
           "source_sha256":plan["source_sha256"],"analysis_dependencies_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (BASE/"analyze.py",BASE/"paired_once_5seeds/analyze_paired.py")},
           "inputs":inputs,"input_failures":failures,"order_pairs":order_pairs,"policy_pairs":policy_pairs,
           "expected_policy_pairs":11,"policy_pairs_passed":sum(p["status"]=="passed" for p in policy_pairs),
           "all_policy_pairs_passed":len(policy_pairs)==11 and all(p["status"]=="passed" for p in policy_pairs),
           "unique_feasible_fingerprints_by_order":same_fp,
           "result_checks":[{k:r.get(k) for k in ("label","strategy","status","path","file_sha256","audit_passed","checks","warm_mixed_card_count","warm_nonmixed_npu_ids","main_window_valid","full_run_nominal_capacity_satisfied","all_study_conditions_met","error")} for r in records],
           "all_complete":complete,"all_technical_audits_passed":technical}
    summary={"created_utc":audit["created_utc"],"plan_sha256":audit["plan_sha256"],"analyzer_sha256":audit["analyzer_sha256"],
             "counts":{"planned":22,"complete":sum(r["status"]=="complete" for r in records),"technical_passed":sum(r.get("audit_passed") is True for r in records),"statuses":dict(Counter(r["status"] for r in records))},
             "means_published":publish,"all_complete":complete,"all_technical_audits_passed":technical,
             "main_valid_count":sum(r.get("all_study_conditions_met") is True for r in mains),"main_expected_count":20,
             "main_condition_counts":{
                 "technical_passed":sum(r.get("audit_passed") is True for r in mains),
                 "all_32_active":sum(r.get("parent_audit",{}).get("windows",[{}])[0].get("all_npus_active") is True for r in mains),
                 "all_32_warm_mixed":sum(r.get("warm_mixed_card_count")==32 for r in mains),
                 "fourth_completions_by_1500":sum(r.get("parent_audit",{}).get("warmup",{}).get("all_fourth_completions_by_1500") is True for r in mains),
                 "full_run_nominal_underload":sum(r.get("full_run_nominal_capacity_satisfied") is True for r in mains)},
             "policy_pairs_passed":audit["policy_pairs_passed"],"all_policy_pairs_passed":audit["all_policy_pairs_passed"],
             "seed_weighting":"Equal mean of all five prespecified seeds; sample SD n-1; never drop invalid warm windows",
             "unique_feasible_fingerprints_by_order":same_fp,"aggregates_all_five_seeds":groups,"paired_order_contrasts":contrasts(mains) if publish else [],"records":records}
    rows=[]
    for r in records:
        row=paired.flatten(r);row["family"]=r["family"]
        row.update(warm_mixed_card_count=r.get("warm_mixed_card_count"),warm_nonmixed_npu_ids=json.dumps(r.get("warm_nonmixed_npu_ids")))
        if "parent_audit" in r:
            p=r["parent_audit"];w=p["windows"][0]
            row.update(window_request_equal_utilization_percent=100*w["request_equal_utilization"],
                       full_run_device_utilization_percent=100*p["full_run"]["device_utilization"],
                       fourth_completions_by_1500=p["warmup"]["all_fourth_completions_by_1500"],
                       window_nominal_peak_ssu_gib_s=p["nominal_demand_scan"]["window_2000_4000"]["max_ssu_gib_s"],
                       window_any_ssu_over_capacity_ms=p["nominal_demand_scan"]["window_2000_4000"]["any_ssu_over_capacity_ms"])
        rows.append(row)
    buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
    write_json(HERE/"audit.json",audit);write_json(HERE/"summary.json",summary)
    atomic(HERE/"per_seed.csv",buffer.getvalue());atomic(HERE/"comparison.md",report_text(summary,inputs,paired))
    require(all(sha(ROOT/name)==expected for name,expected in plan["source_sha256"].items()), "Source changed during audit")
    print(json.dumps({"counts":summary["counts"],"means_published":publish,"main_valid_count":summary["main_valid_count"]}),flush=True)
    if failures or any(r["status"] in ("analysis_error","duplicate_results") or (r["status"]=="complete" and not r["audit_passed"]) for r in records):
        raise SystemExit(1)
    if not args.partial and not(complete and technical):
        raise SystemExit("Not all planned slots completed and passed technical audit; outputs retained")


if __name__=="__main__":
    main()
