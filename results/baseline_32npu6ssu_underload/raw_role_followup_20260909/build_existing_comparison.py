#!/usr/bin/env python3
"""Read frozen experiments; independently rebuild warm U/SLO comparison tables.

No simulator execution. Only writes existing_comparison.{md,csv,json} here.
Every U is recomputed from raw layer intervals; every SLO from raw completion
and admission times and the corresponding frozen input's eight-layer C.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
SEEDS=(7,19,43,67,101)
START,END,ALPHA=2000.0,4000.0,1.5
STUDIES={"paired_once_5seeds":20,"role_separated_5seeds":40,
         "raw_data_random_5seeds":20,"raw_quartet_replacement":22}
KINDS={
    "mixed_random":"每卡长短混合 / Random",
    "mixed_ordered":"每卡长短混合 / Ordered",
    "fixed_11_random":"11 长卡 / 21 短卡 / Random",
    "fixed_11_round_robin":"11 长卡 / 21 短卡 / 轮转",
    "fixed_6_random":"6 长卡 / 26 短卡 / Random",
    "fixed_6_round_robin":"6 长卡 / 26 短卡 / 轮转",
    "raw84":"原始完整 84 行 / 独立均匀抽样",
    "underload33":"原始欠载 33 行子集 / 独立均匀抽样",
    "raw_quartet_random":"原始欠载四画像 / Random",
    "raw_quartet_ordered":"原始欠载四画像 / Ordered",
    "raw_nearest_ordered":"最接近原图的原始四画像 / Ordered（过载诊断）",
}


def read(path):
    with (gzip.open(path,"rt") if str(path).endswith(".gz") else open(path)) as stream:
        return json.load(stream)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def close(x,y):
    return math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-8)


def overlap(a,b):
    return max(0.0,min(b,END)-max(a,START))


def stat(values):
    values=list(values)
    if any(x is None for x in values):
        return {"n":len(values),"mean":None,"sample_sd":None,"min":None,"max":None,"reason":"Empty cohort; no seed dropped."}
    return {"n":len(values),"mean":statistics.mean(values),"sample_sd":statistics.stdev(values) if len(values)>1 else None,
            "min":min(values),"max":max(values)}


def kind_for(study,row):
    if study=="paired_once_5seeds":return "mixed_"+row["order"]
    if study=="role_separated_5seeds":return f"fixed_{row['long_npu_count']}_{row['order']}"
    if study=="raw_data_random_5seeds":return row["arm"]
    return "raw_quartet_"+row["order"] if row["family"]=="raw_feasible_q2" else "raw_nearest_ordered"


@lru_cache(maxsize=None)
def input_info(path):
    data=read(Path(path));rows=data["requests"]
    inputs={r["request_id"]:{"npu_id":r["npu_id"],"arrival_ms":r["arrival_time_ms"],
        "own_C_ms":8*r["load"]["per_layer_us"]/1000,"role":r["load"]["role"],
        "category":r["load"]["category"]} for r in rows}
    assert len(inputs)==len(rows) and all(r["arrival_ms"]==0 for r in inputs.values())
    return {"sha256":sha(path),"input_fingerprint":data["input_fingerprint"],"metadata":data["metadata"],"requests":inputs}


def cohort(rows,inputs):
    out={"count":len(rows),"completion_after_4000_count":sum(r["completion_time_ms"]>END for r in rows)}
    for clock,field in (("admission","admission_time_ms"),("arrival","arrival_time_ms")):
        passed=sum(r["completion_time_ms"]-r[field]<=ALPHA*inputs[r["request_id"]]["own_C_ms"]+1e-9 for r in rows)
        out[clock]={"passed":passed,"count":len(rows),"rate_percent":100*passed/len(rows) if rows else None}
    return out


def source_aggregate(study,summary,row):
    if row.get("family")=="raw_nearest_q25":return None
    groups=summary.get("aggregates_all_five_seeds",summary.get("groups"))
    selected=[g for g in groups if g["strategy"]==row["strategy"]]
    if study=="raw_data_random_5seeds":selected=[g for g in selected if g["arm"]==row["arm"]]
    else:selected=[g for g in selected if g["order"]==row["order"]]
    if study=="role_separated_5seeds":selected=[g for g in selected if g["long_npu_count"]==int(row["long_npu_count"])]
    assert len(selected)==1
    g=selected[0]
    if study=="role_separated_5seeds":return {"u":g["device_utilization_percent"],"slo":g["warm_admission_slo_percent"]}
    def percent(x):return {k:(100*v if k!="n" and isinstance(v,(float,int)) else v) for k,v in x.items()}
    return {"u":percent(g["device_utilization"]),"slo":percent(g["cohorts"]["window_admissions"]["admission"])}


def audit_run(study,row,record,plan,items,source_current):
    base=BASE if study=="paired_once_5seeds" else BASE/study
    path=base/row["path"];raw=read(path);s=raw["summary"]
    item=items[row["label"]];manifest=Path(item["manifest"]);info=input_info(str(manifest));inputs=info["requests"]
    previous_hash=record.get("file_sha256",record.get("analysis_sources",{}).get("result_sha256"))
    actual_hash=sha(path)
    checks={"source_csv_and_existing_audit_complete":row["status"]=="complete" and row["audit_passed"]=="True" and record["audit_passed"],
        "raw_result_sha_matches_existing_audit":actual_hash==previous_hash,
        "manifest_sha_matches_frozen_plan":info["sha256"]==item["manifest_sha256"],
        "input_fingerprint_pairing":raw["input_fingerprint"]==s["input_fingerprint"]==info["input_fingerprint"]==item["input_fingerprint"]==row["input_fingerprint"],
        "seed_and_strategy":raw["submit_seed"]==int(row["seed"])==item["seed"] and raw["strategy"]==row["strategy"] in ("baseline","once"),
        "dimensions_32_6_8_batch1":(s["num_npu"],s["num_ssu"],s["n_layers"],s["batch_size"])==(32,6,8,1),
        "all_simulator_invariants":bool(s["invariants"]) and all(s["invariants"].values()),
        "core_hashes_frozen_and_current":bool(raw["core_and_policy_sha256"]) and all(plan["source_sha256"].get(name)==value==source_current[name] for name,value in raw["core_and_policy_sha256"].items()),
        "stress_runner_hash_frozen_and_current":raw["stress_runner_sha256"]==plan["source_sha256"]["run_baseline_npu32_stress.py"]==source_current["run_baseline_npu32_stress.py"],
        "all_layers_match_input_C":True,"runtime_original_binding":True,"singleton_batches":True}
    c=[[] for _ in range(32)];active=[[] for _ in range(32)];batch_ids=[]
    for b in s["microbatch_metrics"]:
        checks["singleton_batches"] &= len(b["member_request_ids"])==1
        rid=b["member_request_ids"][0];batch_ids.append(rid);n=b["npu_id"]
        checks["runtime_original_binding"] &= n==inputs[rid]["npu_id"]
        checks["all_layers_match_input_C"] &= sorted(l["layer"] for l in b["layer_metrics"])==list(range(8))
        active[n].append(overlap(b["admission_time_ms"],b["completion_time_ms"]))
        for layer in b["layer_metrics"]:
            cs,ce=layer["compute_start_ms"],layer["compute_end_ms"]
            checks["all_layers_match_input_C"] &= close(8*(ce-cs),inputs[rid]["own_C_ms"])
            c[n].append(overlap(cs,ce))
    rows=s["request_metrics"];ids=[r["request_id"] for r in rows]
    checks["complete_unique_populations"]=(len(set(ids))==len(ids)==len(inputs) and set(ids)==set(inputs)
        and len(set(batch_ids))==len(batch_ids)==len(inputs) and set(batch_ids)==set(inputs))
    checks["request_C_arrival_npu_match_manifest"] = all(close(r["own_compute_ms"],inputs[r["request_id"]]["own_C_ms"])
        and r["arrival_time_ms"]==inputs[r["request_id"]]["arrival_ms"]==0 and r["npu_id"]==inputs[r["request_id"]]["npu_id"] for r in rows)
    checks["every_request_followed_to_completion"]=all(math.isfinite(r["completion_time_ms"]) and r["completion_time_ms"]>=r["admission_time_ms"] for r in rows)
    warm=[r for r in rows if START<=r["admission_time_ms"]<END]
    arrivals=[r for r in rows if START<=r["arrival_time_ms"]<END]
    u=100*math.fsum(math.fsum(x) for x in c)/(32*(END-START))
    co={name:cohort(sample,inputs) for name,sample in (("window_admissions",warm),("window_arrivals",arrivals),("all_requests",rows))}
    stored=raw["slo"]
    checks["stored_slo_window_alpha_and_ideal"] = (stored["window_start_ms"],stored["window_end_ms"],stored["alpha"])==(START,END,ALPHA) and stored["ideal"]=="n_layers * per_layer_compute_ms (own_compute_ms)"
    for name in co:
        for clock in ("admission","arrival"):
            a,b=co[name][clock],stored[name][clock]
            checks[f"raw_stored_{name}_{clock}_slo_matches"] = a["count"]==b["count"] and a["passed"]==b["passed"] and (b["rate"] is None if a["rate_percent"] is None else close(a["rate_percent"],100*b["rate"]))
    checks["CSV_warm_count_and_pass_count_match"]=len(warm)==int(row["warm_count"]) and co["window_admissions"]["admission"]["passed"]==int(row["warm_admission_passed"])
    checks["CSV_U_and_warm_SLO_match"]=close(u,float(row["device_utilization_percent"])) and close(co["window_admissions"]["admission"]["rate_percent"],float(row["warm_admission_slo_percent"]))
    windows=[w for w in raw["windows"] if (w["start_ms"],w["end_ms"])==(START,END)]
    checks["stored_window_U_matches"]=len(windows)==1 and close(u,100*windows[0]["mean_npu_utilization"])
    adapter=raw["adapter_statistics"]
    times=adapter["collector_times_ms"]
    checks["once_or_baseline_5ms_collector"]=adapter["strategy"]==raw["strategy"] and raw["collector_interval_ms"]==adapter["collector_interval_ms"]==5 and all(close(t,5*i) for i,t in enumerate(times)) and adapter["fresh_reads_by_ssu"]==[len(times)]*6
    checks["fixed_binding_no_migration"]=adapter["assignment_count"]==0 and raw["assignment_log"]==[]
    checks["static_CIR_and_no_internal_queue_reorder"]=adapter["cir_write_events"]==[] and adapter["reorder_calls"]==adapter["reorder_changed"]==0
    checks["modeled_control_cost_zero"]=adapter["modeled_control_cpu_latency_ms"]==adapter["modeled_control_communication_latency_ms"]==0
    checks["all_32_active_warm"]=all(close(math.fsum(x),END-START) for x in active)
    checks["warm_arrivals_empty"]=not arrivals
    checks["warm_admissions_arrival_origin_slo_zero"]=co["window_admissions"]["arrival"]["passed"]==0
    assert all(checks.values()),(study,row["label"],row["strategy"],[k for k,v in checks.items() if not v])
    roles={role:cohort([r for r in warm if inputs[r["request_id"]]["role"]==role],inputs) for role in ("short","long")}
    categories={cat:cohort([r for r in warm if inputs[r["request_id"]]["category"]==cat],inputs) for cat in ("SS","SL","LS","LL")}
    return {"study":study,"condition":kind_for(study,row),"condition_name":KINDS[kind_for(study,row)],"seed":int(row["seed"]),"order":row.get("order","random"),
        "strategy":row["strategy"],"label":row["label"],"input_fingerprint":raw["input_fingerprint"],"source_result":str(path.relative_to(ROOT)),
        "source_result_sha256":actual_hash,"source_manifest":str(manifest.relative_to(ROOT)),"source_manifest_sha256":info["sha256"],
        "device_utilization_percent":u,"cohorts":co,"warm_by_role":roles,"warm_by_category":categories,
        "warm_request_ids":sorted(r["request_id"] for r in warm),"warm_npu_compute_ms":[math.fsum(x) for x in c],
        "existing_nominal_capacity_passed":row["full_run_nominal_capacity_satisfied"]=="True",
        "existing_study_conditions_met":row.get("all_study_conditions_met"),"existing_primary_valid":row.get("primary_valid"),
        "audit_passed":True,"checks":checks,"core_source_sha256":raw["core_and_policy_sha256"],"static_path_cirs":raw["static_path_cirs_gib_s"]}


def fmt(x):
    if x["mean"] is None:return "N/A"
    return f"{x['mean']:.4f} ± {x['sample_sd']:.4f}" if x["sample_sd"] is not None else f"{x['mean']:.4f}（单次）"


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    monitored={str(Path(__file__).relative_to(ROOT)):sha(__file__)};sources={};runs=[];expected_groups={}
    for study,expected in STUDIES.items():
        directory=BASE/study;summary=read(directory/"summary.json");plan=read(directory/"plan.json")
        rows=list(csv.DictReader((directory/"per_seed.csv").open()));assert len(rows)==expected
        if study=="raw_data_random_5seeds":detail=read(directory/"analysis.json");records=detail["runs"]
        else:records=summary.get("runs",summary.get("records"))
        record_map={(r["label"],r["strategy"]):r for r in records};assert len(record_map)==expected
        items={i["label"]:i for i in plan["inputs"]}
        for name in ("summary.json","per_seed.csv","plan.json"):
            p=directory/name;monitored[str(p.relative_to(ROOT))]=sha(p)
        if study=="raw_data_random_5seeds":monitored[str((directory/"analysis.json").relative_to(ROOT))]=sha(directory/"analysis.json")
        current={name:sha(ROOT/name) for name in plan["source_sha256"]}
        assert current==plan["source_sha256"],"Frozen source changed"
        monitored.update(current)
        sources[study]={"source_files":{str((directory/name).relative_to(ROOT)):monitored[str((directory/name).relative_to(ROOT))] for name in ("summary.json","per_seed.csv","plan.json")},
            "raw_case_count":expected,"frozen_core_and_construction_sources_match":True}
        for row in rows:
            key=(kind_for(study,row),row["strategy"])
            expected_groups[key]=source_aggregate(study,summary,row)
            r=audit_run(study,row,record_map[(row["label"],row["strategy"])],plan,items,current);runs.append(r)
            print(json.dumps({"study":study,"label":r["label"],"strategy":r["strategy"],"passed":r["audit_passed"]}),flush=True)
    assert len(runs)==102
    groups=[]
    for condition in KINDS:
        for strategy in ("baseline","once"):
            selected=[r for r in runs if r["condition"]==condition and r["strategy"]==strategy]
            expected_seeds=(7,) if condition=="raw_nearest_ordered" else SEEDS
            assert sorted(r["seed"] for r in selected)==list(expected_seeds)
            g={"condition":condition,"condition_name":KINDS[condition],"strategy":strategy,"seeds":list(expected_seeds),"n_seeds":len(selected),
                "device_utilization_percent":stat(r["device_utilization_percent"] for r in selected),
                "warm_admission_slo_percent":stat(r["cohorts"]["window_admissions"]["admission"]["rate_percent"] for r in selected),
                "warm_count":stat(r["cohorts"]["window_admissions"]["count"] for r in selected),
                "warm_arrival_origin_slo_percent":stat(r["cohorts"]["window_admissions"]["arrival"]["rate_percent"] for r in selected),
                "full_admission_slo_percent":stat(r["cohorts"]["all_requests"]["admission"]["rate_percent"] for r in selected),
                "all_recomputed_checks_passed":all(r["audit_passed"] for r in selected),
                "nominal_capacity_passed_count":sum(r["existing_nominal_capacity_passed"] for r in selected),
                "warm_by_role":{},"warm_by_category":{}}
            for field,labels in (("warm_by_role",("short","long")),("warm_by_category",("SS","SL","LS","LL"))):
                for label in labels:
                    g[field][label]={"count":stat(r[field][label]["count"] for r in selected),
                        "admission_slo_percent":stat(r[field][label]["admission"]["rate_percent"] for r in selected)}
            old=expected_groups[(condition,strategy)]
            if old:
                for actual,wanted in ((g["device_utilization_percent"],old["u"]),(g["warm_admission_slo_percent"],old["slo"])):
                    assert actual["n"]==wanted["n"] and all(close(actual[k],wanted[k]) for k in ("mean","sample_sd","min","max")),(condition,strategy,actual,wanted)
                g["prior_five_seed_aggregate_matches"]=True
            else:g["prior_five_seed_aggregate_matches"]=None
            groups.append(g)
    pairs=[]
    for study in STUDIES:
        for label in sorted({r["label"] for r in runs if r["study"]==study}):
            a,b=sorted((r for r in runs if r["study"]==study and r["label"]==label),key=lambda r:r["strategy"])
            assert a["strategy"]=="baseline" and b["strategy"]=="once"
            checks={"identical_input_fingerprint":a["input_fingerprint"]==b["input_fingerprint"],"identical_manifest_sha":a["source_manifest_sha256"]==b["source_manifest_sha256"],
                "same_seed":a["seed"]==b["seed"],"same_core_sources":a["core_source_sha256"]==b["core_source_sha256"],"same_static_CIR":a["static_path_cirs"]==b["static_path_cirs"]}
            assert all(checks.values())
            x,y=set(a["warm_request_ids"]),set(b["warm_request_ids"])
            pairs.append({"study":study,"label":label,"passed":True,"checks":checks,"baseline_warm_count":len(x),"once_warm_count":len(y),
                "intersection_count":len(x&y),"baseline_only_count":len(x-y),"once_only_count":len(y-x)})
    unchanged=all(sha(ROOT/name)==value for name,value in monitored.items());assert unchanged
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"rate_units":"percent (0..100), sample SD in percentage points", "window_ms":[START,END],"alpha":ALPHA,
        "seeds":list(SEEDS),"case_count":len(runs),"audit_passed_count":sum(r["audit_passed"] for r in runs),"pair_count":len(pairs),
        "all_audits_passed":unchanged and all(r["audit_passed"] for r in runs) and all(p["passed"] for p in pairs),
        "definitions":{"device_U":"Sum of clipped raw layer compute intervals / (32 * 2000ms)",
            "primary_SLO":"Admissions in [2000,4000)ms; completion-admission <=1.5 * eight-layer pure C from corresponding frozen manifest; follow completion past4000ms.",
            "TTFT_scope":"Post-admission 8-layer processing proxy, not user arrival TTFT or real token/hardware timing. No decode/first-token event is measured.",
            "arrival":"All arrivals zero. Warm arrival cohort is empty/N/A. Arrival-origin SLO on warm-admission cohort is0%.",
            "seed_weight":"Arithmetic mean of each seed's rate; sample SD uses n-1. No pooling of unequal warm cohorts.",
            "policy":"Baseline vs shared-path once/Once per layer, collector5ms, identical staticCIR, no runtimeCIR writes, assignment fixed, no NPU migration or internal path reorder, modeled control costs0; no NewOnce.",
            "capacity":"Copied from the already audited full-run event scan; this integration independently recomputes U/SLO and verifies raw result/input/source hashes, not the event-scan capacity proof.",
            "main_intervention":"All studies already bind requests to fixed NPUs. Role-separated experiments change per-NPU population and its IO phases, not fixed versus dynamic dispatch."},
        "source_paths_relative_to":str(ROOT),"source_files_sha256":monitored,"study_sources":sources,"sources_unchanged_during_analysis":unchanged,
        "groups":groups,"policy_pairs":pairs,"runs":runs}
    (HERE/"existing_comparison.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    flat=[]
    for g in groups:
        r={k:g[k] for k in ("condition","condition_name","strategy","n_seeds","all_recomputed_checks_passed","nominal_capacity_passed_count")}
        r["seeds"]=";".join(map(str,g["seeds"]))
        for metric in ("device_utilization_percent","warm_admission_slo_percent","warm_count","full_admission_slo_percent"):
            for key,value in g[metric].items():r[metric+"_"+key]=value
        flat.append(r)
    with (HERE/"existing_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,list(flat[0]));writer.writeheader();writer.writerows(flat)
    lookup={(g["condition"],g["strategy"]):g for g in groups}
    lines=["# 已有实验：每卡混合与长短角色专用分卡", "",
        "本页仅整理已完成的冻结实验，没有重跑。已直接重读 102 份压缩结果，从逐层区间独立重算 U，从请求完成时间和对应 manifest 的 C 独立重算 SLO；逐次结果与既有 CSV 一致，五种子均值、样本标准差及 min/max 与原汇总一致。102 份均通过本次检查，51 组 Baseline/Once 同输入配对通过。", "",
        "统一 32 NPU、6 SSU（每盘40 GiB/s、每卡链路50 GiB/s），每请求8层，主窗 [2000,4000) ms。五个预定种子为7、19、43、67、101；先对每个seed分别求率，再等权求均值，±为样本标准差。下面均为百分数；标准差反映百分点波动，不池化暖窗请求。", "",
        r"$U=\frac{\sum_{n=0}^{31}|\text{计算区间}_n\cap[2000,4000)|}{32\times2000}$；$SLO=\frac{\#\{r:2000\le a_r<4000,\ f_r-a_r\le1.5\times8C_r\}}{\#\{r:2000\le a_r<4000\}}$。", "",
        "**表中的 TTFT SLO 是接纳后的八层处理时间代理，计时起点是 admission，不是 arrival。** a为接纳、f为完成、C为原输入单层纯计算时间。暖窗内接纳的请求都跟到完成，即使完成晚于4秒。所有arrival均为0，所以暖窗到达人口为空，arrival队列SLO应记N/A；若对暖接纳人口按真实arrival计时，达标率全部为0%。这里没有测真实硬件首token或decode事件。", "",
        "Once是共享路径实现的once/Once per layer，5ms采集快照、静态CIR、不更新运行时CIR；不是NewOnce。两策略使用相同输入、提交种子与CIR。所有运行均固定NPU绑定，无跨卡迁移或path内部重排，控制计算与通信额外延迟在模型中为0。", "",
        "## 同一批构造请求：重点比较", "",
        "这六种配置都使用同一全局19,456条构造请求：S1/S2/S3=(1K,128/256/384)，每种6,400条；L=(192K,768)，256条。短画像外推到1K，长画像在NQL512/1024之间插值，均不能说是原data四行。角色分卡保留原请求C/V、arrival、逐块落盘位置，改变每卡分配人口；原混合实验本来也固定绑定，不能归因于‘fixed比dynamic差’。", "",
        "| 输入分配与队列 | Baseline 暖窗 U % | Once 暖窗 U % | Baseline 暖接纳 SLO×1.5 % | Once 暖接纳 SLO×1.5 % |", "|---|---:|---:|---:|---:|"]
    for condition in list(KINDS)[:6]:
        a,b=lookup[condition,"baseline"],lookup[condition,"once"]
        lines.append(f"| {KINDS[condition]} | {fmt(a['device_utilization_percent'])} | {fmt(b['device_utilization_percent'])} | {fmt(a['warm_admission_slo_percent'])} | {fmt(b['warm_admission_slo_percent'])} |")
    lines += ["", "固定11/21与6/26的长卡分别为NPU0–10与0–5，其余卡只运行三种短画像。同一比例、同一种子内random与轮转保持每卡原请求身份和落盘位置相同；长卡两种模式的身份次序也相同。更换比例或从混合改为专用分卡会改变每卡人口及读取相位。上述60格均通过原各自约定的暖机、全32卡active及全程逐盘/链路名义欠载检查；专用分卡有意不要求每卡长短混合。", "",
        "这些结果支持这批构造输入在专用分卡后更容易暴露Baseline短卡等待，不能推出所有原始输入或所有内部排列都差。U主表仅统计暖窗；6长卡方案的完整批次另有绑定不均衡导致的尾部空卡，不能把该尾部归为FIFO损失。", "",
        "Random种子同时影响卡内随机序列和同刻提交次序。混合Ordered跨seed使用相同画像模板；固定角色round_robin是S1→S2→S3轮转，并非混合Ordered的相位构造。", "",
        "## 原始 data：不同输入，单独列出", "", "| 输入 | Baseline 暖窗 U % | Once 暖窗 U % | Baseline 暖接纳 SLO×1.5 % | Once 暖接纳 SLO×1.5 % |", "|---|---:|---:|---:|---:|"]
    for condition in list(KINDS)[6:10]:
        a,b=lookup[condition,"baseline"],lookup[condition,"once"]
        lines.append(f"| {KINDS[condition]} | {fmt(a['device_utilization_percent'])} | {fmt(b['device_utilization_percent'])} | {fmt(a['warm_admission_slo_percent'])} | {fmt(b['warm_admission_slo_percent'])} |")
    lines += ["", "原始84/33臂按每卡独立有放回抽样，保留原C，NQL64的88KiB尾块补为176KiB物理命令。完整84行这批运行不满足原逐盘逐时刻名义欠载条件，且9种画像受单卡链路下界限制，SLO×1.5即使无磁盘排队也不可行。33行子集事先筛定以保证任意组合名义欠载，只含SL/LL；其U/SLO100%不能代表完整原表。两臂没有为满足旧的第四完成/单卡混合条件而重抽seed。", "",
        "原始欠载四画像为32K/1024、48K/1024、64K/1024、160K/1024；每卡前三类各14条、第四类7条，完整人口1,568条。它同时改变计算时间比、读取量、路由类别和配额，不能叫只去掉外推的单因素消融，也不能由这一组合未复现下降推断整个原表不存在坏输入。", "",
        "原表Once总体SLO改善，也不等于各类都受益：暖LL SLO从Baseline100%降为Once五seed均值98.8510%；原表seed67和101的Once设备U还略低于Baseline。完整同人口结果及各暖类别计数保留在JSON，以检查政策间暖接纳集合不同的影响。", "",
        "### 单seed过载诊断（没有五seed标准差）", "", "| seed7 Ordered 输入 | Baseline U % | Once U % | Baseline 暖SLO % | Once 暖SLO % |", "|---|---:|---:|---:|---:|"]
    a,b=lookup["raw_nearest_ordered","baseline"],lookup["raw_nearest_ordered","once"]
    lines += [f"| 最近原始四画像 | {fmt(a['device_utilization_percent'])} | {fmt(b['device_utilization_percent'])} | {fmt(a['warm_admission_slo_percent'])} | {fmt(b['warm_admission_slo_percent'])} |", "",
        "该诊断用32K/128、32K/256、32K/512和192K/1024，不能满足原欠载约束；低U不能作为欠载调度反例。只有已完成的seed7 Ordered两策略，未伪造其他种子、均值或标准差。", "",
        "## 审核与可复算来源", "",
        "JSON按运行保存原始结果与manifest路径/哈希、完整检查项、暖接纳ID、人数/通过数、分类SLO、每卡裁剪计算时间；按策略配对保留暖人口交集，不能把相同人数当成相同请求。容量判定沿用先前已完成的逐事件独立审计，并与原始结果/输入哈希绑定，本次没有重新扫描容量或运行仿真。", "",
        "- [本次汇总CSV](existing_comparison.csv)：22个策略行，100格五seed实验加2格单seed诊断。",
        "- [本次完整审核JSON](existing_comparison.json)：102格，51个同输入策略对及所有来源SHA256。",
        "- 复算：`python -B build_existing_comparison.py`。", ""]
    for study,entry in sources.items():
        lines.append(f"- [{study} 的既有逐seed表](../{study}/per_seed.csv)；[原说明](../{study}/comparison.md)；CSV SHA256 `{entry['source_files']['results/baseline_32npu6ssu_underload/'+study+'/per_seed.csv']}`。")
    (HERE/"existing_comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"cases":len(runs),"groups":len(groups),"pairs":len(pairs),"all_audits_passed":report["all_audits_passed"]}),flush=True)


if __name__=="__main__":main()
