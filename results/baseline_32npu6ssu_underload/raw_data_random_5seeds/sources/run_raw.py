#!/usr/bin/env python3
"""Freeze direct-data IID inputs, then run paired Baseline/Once without core edits."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(ROOT), str(STUDY)]
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, read_json, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_coflow_experiments import source_files

SEEDS = (7, 19, 43, 67, 101)
ARMS = ("raw84", "underload33")
STRATEGIES = ("baseline", "once")
IO_GIB = 176 * 1024 / 2**30
NPU, SSU, LAYERS = 32, 6, 8
WINDOW = (2000.0, 4000.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog():
    data = ast.literal_eval((ROOT / "data").read_text())
    assert len(data) == 84
    rows = []
    for (seq, nql), (bw, c_us, source_ttft, raw_v) in sorted(data.items()):
        tokens = seq * 1024 - nql
        count = (tokens + 127) // 128
        physical_v = count * IO_GIB
        padding = physical_v - raw_v
        assert padding >= -1e-12
        if nql == 64:
            assert math.isclose(padding, IO_GIB/2, abs_tol=1e-12)
        else:
            assert math.isclose(padding, 0, abs_tol=1e-12)
        c_ms = c_us / 1000
        b = 1000 * physical_v / c_ms
        conservative_disk = 32 * ((count + 5)//6) * IO_GIB * 1000 / c_ms
        lower = c_ms + 7 * physical_v / 50 * 1000
        rows.append({"seq_len_k":seq,"nql":nql,"per_layer_us":c_us,
            "source_required_bw_gib_s":bw,"source_ttft_78layer_ms":source_ttft,
            "source_per_layer_kv_gib":raw_v,"physical_per_layer_kv_gib":physical_v,
            "padding_gib_per_layer":max(0,padding),"block_count_per_layer":count,
            "category":("S" if seq<=80 else "L")+("S" if nql<512 else "L"),
            "physical_V_over_C_gib_s":b,"max_concurrent_32_per_ssu_gib_s":conservative_disk,
            "eligible_underload33":conservative_disk<40,
            "post_admission_link_lower_bound_ms":lower,
            "slo_threshold_ms":1.5*8*c_ms,"link_alone_forces_slo_failure":lower>1.5*8*c_ms+1e-9})
    assert sum(r["eligible_underload33"] for r in rows)==33
    assert sum(r["link_alone_forces_slo_failure"] for r in rows)==9
    return rows


def catalog_stats(rows):
    c=math.fsum(r["per_layer_us"]/1000 for r in rows)/len(rows)
    raw=math.fsum(r["source_per_layer_kv_gib"] for r in rows)/len(rows)
    physical=math.fsum(r["physical_per_layer_kv_gib"] for r in rows)/len(rows)
    return {"row_count":len(rows),"mean_layer_compute_ms":c,"mean_raw_V_mib":1024*raw,
        "mean_physical_V_mib":1024*physical,"ratio_mean_physical_V_over_mean_C_gib_s":1000*physical/c,
        "request_equal_mean_V_over_C_gib_s":math.fsum(r["physical_V_over_C_gib_s"] for r in rows)/len(rows),
        "ideal_compute_time_weighted_average_ssu_gib_s":32/6*1000*physical/c,
        "rows_nominal_link_above_50":sum(r["physical_V_over_C_gib_s"]>50 for r in rows),
        "rows_link_alone_forces_slo_failure":sum(r["link_alone_forces_slo_failure"] for r in rows),
        "category_counts":dict(Counter(r["category"] for r in rows)),
        "category_compute_fractions":{cat:math.fsum(r["per_layer_us"] for r in rows if r["category"]==cat)/math.fsum(r["per_layer_us"] for r in rows) for cat in ("SS","SL","LS","LL")}}


def make_input(arm, seed, all_rows, budget):
    rows=[r for r in all_rows if arm=="raw84" or r["eligible_underload33"]]
    requests=[];lanes=[];placement_cache={}
    for n in range(NPU):
        rng=random.Random(seed+100003*n)
        compute=[];selected=[];order=[]
        while math.fsum(compute)<budget:
            k=rng.randrange(len(rows));p=rows[k];idx=len(compute);rid=n*1000000+idx
            key=(n//4,p["seq_len_k"],p["nql"])
            if key not in placement_cache:
                one=tuple(((b+n//4)%SSU,IO_GIB) for b in range(p["block_count_per_layer"]))
                placement_cache[key]=(one,)*LAYERS
            load={"request_id":rid,"npu_id":n,"generation":idx,"initial":True,
                "arrival_time":0.0,"arrival_ms":0.0,"seq_len_k":p["seq_len_k"],"nql":p["nql"],
                "category":p["category"],"role":"short" if p["seq_len_k"]<=80 else "long",
                "per_layer_us":p["per_layer_us"],"original_compute_us":p["per_layer_us"],
                "per_layer_kv_gb":p["source_per_layer_kv_gib"],
                "source_per_layer_kv_gib":p["source_per_layer_kv_gib"],
                "physical_per_layer_kv_gib":p["physical_per_layer_kv_gib"],
                "padding_gib_per_layer":p["padding_gib_per_layer"],
                "required_bw_input_gbps":p["source_required_bw_gib_s"],
                "source_ttft_ms":p["source_ttft_78layer_ms"],"constructed_profile":False}
            requests.append(ContinuousBatchRequest.from_normalized(rid,n,0.0,load,placement_cache[key]))
            compute.append(LAYERS*p["per_layer_us"]/1000);selected.append(p);order.append(k)
        lanes.append({"npu_id":n,"rng_seed":seed+100003*n,"request_count":len(compute),
            "pure_compute_ms":math.fsum(compute),"pure_compute_before_last_request_ms":math.fsum(compute[:-1]),
            "catalog_indices":order,"profile_counts":{f"{s}:{q}":count for (s,q),count in sorted(Counter((p["seq_len_k"],p["nql"]) for p in selected).items())},
            "category_counts":dict(Counter(p["category"] for p in selected)),
            "nominal_link_infeasible_request_count":sum(p["physical_V_over_C_gib_s"]>50 for p in selected),
            "link_alone_forced_slo_failure_count":sum(p["link_alone_forces_slo_failure"] for p in selected)})
    requests=tuple(requests);label=f"{arm}_uniform_seed{seed}";fp=continuous_batch_input_fingerprint(requests)
    meta={"experiment":"direct_raw_data_uniform_iid_5seeds_v1","label":label,"case_id":label+"_"+fp[:12],
        "arm":arm,"seed":seed,"family":"raw","num_npu":NPU,"num_ssu":SSU,"n_layers":LAYERS,
        "request_count":len(requests),"input_fingerprint":fp,"logical_input_fingerprint":logical_input_fingerprint(requests),
        "equal_176kib_blocks":True,"blocks":"176KiB_physical_allocation","layout":"stripe",
        "stripe_rule":"ssu=(block_index+npu_id//4)%6; same exact placement under both strategies",
        "compute_scale_actual":1.0,"all_arrival_zero":True,"measurement_window_ms":list(WINDOW),
        "horizon_ms":WINDOW[1],"minimum_pure_compute_budget_ms":budget,"per_npu":lanes,
        "source_data_sha256":sha(ROOT/"data"),"construction_runner_sha256":sha(__file__),
        "sampling":"Each NPU uses random.Random(seed+100003*npu), sorted catalog, independent uniform randrange with replacement, stopping only at declared pure-compute budget.",
        "sampling_caveat":"No fixed warmup profiles, seed retries, compute scaling or repeating decks. Stop-dependent terminal request lies after warm window even under pure-compute execution.",
        "role_definition":"short means seq_len_k<=80; long means seq_len_k>80. These are sequence-length labels, not measured compute duration; inspect all four categories.",
        "physical_volume_rule":"Original data C and V retained in load; each raw64 tail of 88KiB occupies a full 176KiB physical command, so 88KiB extra per layer for NQL64 only. Actual placement defines physical traffic.",
        "catalog_rows":rows,"catalog_stats":catalog_stats(rows),
        "old_validity_caveat":"Do not reject/retry runs for failure of original fourth-completion warmup, per-card mixed roles, per-SSD nominal capacity or individual link feasibility. Report these independently."}
    return requests,meta


def prepare():
    HERE.mkdir(parents=True,exist_ok=True)
    rows=catalog();maximum=max(8*r["per_layer_us"]/1000 for r in rows);budget=4000+2*maximum
    names=sorted(set(source_files())|{"run_baseline_npu32_stress.py","data"})
    hashes={name:sha(ROOT/name) for name in names}
    inputs=[]
    for seed in SEEDS:
        for arm in ARMS:
            requests,meta=make_input(arm,seed,rows,budget)
            path=HERE/"inputs"/(meta["label"]+".json.gz")
            save_manifest(path,requests,meta)
            existing,stored=load_manifest(path)
            assert stored==meta and continuous_batch_input_fingerprint(existing)==meta["input_fingerprint"]
            inputs.append({"label":meta["label"],"arm":arm,"seed":seed,"manifest":str(path),
                "manifest_sha256":sha(path),"input_fingerprint":meta["input_fingerprint"],"request_count":len(requests),
                "min_per_npu_pure_compute_ms":min(n["pure_compute_ms"] for n in meta["per_npu"]),
                "min_per_npu_pure_compute_before_last_ms":min(n["pure_compute_before_last_request_ms"] for n in meta["per_npu"])})
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"seeds":list(SEEDS),"arms":list(ARMS),
        "strategies":list(STRATEGIES),"planned_cases":20,"window_ms":list(WINDOW),
        "num_npu":NPU,"num_ssu":SSU,"n_layers":LAYERS,"assignment":"fixed","collector_interval_ms":5,
        "source_sha256":hashes,"runner_sha256":sha(__file__),"data_sha256":sha(ROOT/"data"),
        "catalog":rows,"catalog_stats":{arm:catalog_stats([r for r in rows if arm=="raw84" or r["eligible_underload33"]]) for arm in ARMS},
        "underload_catalog_rule":"Before any seed draw: retain iff 32*ceil(physical_blocks/6)*176KiB/C <40GiB/s. This independent per-profile worst-SSU certificate gives 33 rows, only SL/LL; this is not uniform sampling from all84.",
        "maximum_request_pure_compute_ms":maximum,"minimum_lane_pure_compute_budget_ms":budget,
        "slo":"Warm admissions [2000,4000), completion-admission <=1.5*8C, follow all requests through completion. This is post-admission 8-layer TTFT proxy, not user arrival TTFT.",
        "seed_mean":"Equal weight per seed rate; sample SD/min/max; never pool unequal warm populations.",
        "primary_validity":"Source/input/metric integrity and all32 NPUs active in [2,4)s. First-completion<=1500, fourth-completion<=1500, per-card mixed roles, full-run per-SSD and individual-link nominal capacity are separately reported, not used to select seeds.",
        "link_lower_bound":"L1-L7 cannot prefetch before admission; completion-admission >=7*physical_V/50GiBps + C, including final compute. Nine full-catalog rows make 1.5*8C intrinsically infeasible for both strategies.",
        "inputs":inputs,"python_version":sys.version}
    destination=HERE/"plan.json"
    if destination.exists():
        old=read_json(destination)
        for key in ("inputs","source_sha256","runner_sha256","arms","seeds","catalog"):
            assert old[key]==plan[key],f"frozen plan changed: {key}"
        return old
    for name in names:
        p=HERE/"sources"/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,p)
    shutil.copyfile(__file__,HERE/"sources"/"run_raw.py")
    write_json(destination,plan)
    print(json.dumps({"prepared":inputs,"catalog_counts":{k:v["row_count"] for k,v in plan["catalog_stats"].items()},"budget_ms":budget},ensure_ascii=False),flush=True)
    return plan


def validate_result(path,item,strategy,plan):
    result=read_json(path)
    assert result["input_fingerprint"]==item["input_fingerprint"]
    assert result["strategy"]==result["adapter_statistics"]["strategy"]==strategy
    assert result["submit_seed"]==item["seed"]
    assert result["collector_interval_ms"]==5
    assert all(result["summary"]["invariants"].values())
    assert result["summary"]["request_count"]==item["request_count"]
    assert all(plan["source_sha256"].get(name)==value for name,value in result["core_and_policy_sha256"].items())
    assert result["stress_runner_sha256"]==plan["source_sha256"]["run_baseline_npu32_stress.py"]
    return result


def run_job(item,strategy,plan,timeout):
    out=HERE/"runs"/item["label"]/strategy;out.mkdir(parents=True,exist_ok=True)
    files=list(out.glob("*.json.gz"))
    if files:
        assert len(files)==1
        validate_result(files[0],item,strategy,plan)
        return {"label":item["label"],"arm":item["arm"],"seed":item["seed"],"strategy":strategy,"status":"existing","output":str(files[0])}
    argv=[sys.executable,"-B",str(ROOT/"run_baseline_npu32_stress.py"),"--manifest",item["manifest"],
          "--strategy",strategy,"--assignment","fixed","--window","2000:4000","--output",str(out)]
    record={"label":item["label"],"arm":item["arm"],"seed":item["seed"],"strategy":strategy,
            "argv":argv,"status":"running","start_utc":datetime.now(timezone.utc).isoformat()}
    start=time.perf_counter()
    with (out/"stdout.log").open("w") as stream:
        proc=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        record["pid"]=proc.pid;write_json(out/"command.json",record)
        try:
            code=proc.wait(timeout=timeout)
            record.update(returncode=code,status="complete" if code==0 else "failed")
            if code==0:
                files=list(out.glob("*.json.gz"));assert len(files)==1
                validate_result(files[0],item,strategy,plan);record["output"]=str(files[0])
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            record.update(status="timeout",returncode=proc.returncode)
        except Exception as exc:
            record.update(status="audit_failed",error=f"{type(exc).__name__}: {exc}")
    record.update(wall_seconds=time.perf_counter()-start,end_utc=datetime.now(timezone.utc).isoformat())
    write_json(out/"command.json",record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true");parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--timeout",type=float,default=7200)
    args=parser.parse_args();plan=prepare()
    if not args.run:return 0
    assert all(sha(ROOT/name)==value for name,value in plan["source_sha256"].items())
    rows=[];started=time.perf_counter();jobs=[(i,s) for i in plan["inputs"] for s in STRATEGIES]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_job,i,s,plan,args.timeout):(i,s) for i,s in jobs}
        while pending:
            done,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                item,strategy=pending.pop(future)
                try:row=future.result()
                except Exception as exc:row={"label":item["label"],"arm":item["arm"],"seed":item["seed"],"strategy":strategy,"status":"orchestrator_error","error":f"{type(exc).__name__}: {exc}"}
                rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            status={"rows":rows,"finished":len(rows),"planned":len(jobs),"pending":len(pending),
                    "successful":sum(r["status"] in ("complete","existing") for r in rows),
                    "failed":sum(r["status"] not in ("complete","existing") for r in rows),
                    "elapsed_seconds":time.perf_counter()-started,"all_complete":not pending}
            write_json(HERE/"status.json",status)
            if not done:print(json.dumps({k:v for k,v in status.items() if k!="rows"}),flush=True)
    assert all(sha(ROOT/name)==value for name,value in plan["source_sha256"].items()),"core source changed during run"
    assert sha(__file__)==plan["runner_sha256"],"runner changed during run"
    return 0 if all(r["status"] in ("complete","existing") for r in rows) else 1


if __name__=="__main__":
    raise SystemExit(main())
