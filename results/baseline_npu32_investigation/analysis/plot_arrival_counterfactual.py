#!/usr/bin/env python3
"""Audit the micro-arrival intervention and plot real admitted-request lanes.

Colors encode the data-source SSU of an admitted request, including computation
and waiting. They are not disk activity measurements. Reads saved results only.
"""

import argparse
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
spec = importlib.util.spec_from_file_location("stress_analysis", BASE / "analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def one(pattern):
    paths = sorted(BASE.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one source result, found {len(paths)}: {pattern}")
    return paths[0]


def audit_inputs(old_path, new_path):
    old, new = analysis.read_json(old_path), analysis.read_json(new_path)
    before = {r["request_id"]: r for r in old["requests"]}
    after = {r["request_id"]: r for r in new["requests"]}
    checks = {"same_unique_request_ids": len(before)==len(old["requests"])==len(after)==len(new["requests"])
              and set(before)==set(after), "same_placement_catalog": old["placements"]==new["placements"],
              "same_complete_input_demand": old["metadata"]["input_demand"]==new["metadata"]["input_demand"]}
    changes, role_home, nonarrival_before, nonarrival_after = [], {}, [], []
    fields = Counter()
    checks["only_arrival_fields_changed"] = True
    for rid, a in sorted(before.items()):
        b = after[rid]
        for key in set(a)|set(b):
            if a.get(key)!=b.get(key):
                fields[key] += 1
                if key not in ("arrival_time_ms","load"):
                    checks["only_arrival_fields_changed"] = False
        for key in set(a["load"])|set(b["load"]):
            if a["load"].get(key)!=b["load"].get(key):
                fields["load."+key] += 1
                if key not in ("arrival_ms","arrival_time"):
                    checks["only_arrival_fields_changed"] = False
        for r, target in ((a,nonarrival_before),(b,nonarrival_after)):
            stripped = {k:v for k,v in r.items() if k not in ("arrival_time_ms","load")}
            stripped["load"] = {k:v for k,v in r["load"].items() if k not in ("arrival_ms","arrival_time")}
            target.append(stripped)
        home = {ssu for layer in old["placements"][a["placement_index"]] for ssu,_ in layer}
        if len(home)!=1:
            raise AssertionError("This display requires exactly one source SSU per request")
        role_home[rid] = next(iter(home))
        changes.append({"request_id":rid,"original_npu_id":a["npu_id"],"generation":a["load"]["generation"],
            "home_ssu":role_home[rid],"old_arrival_ms":a["arrival_time_ms"],"new_arrival_ms":b["arrival_time_ms"],
            "delta_arrival_us":1000*(b["arrival_time_ms"]-a["arrival_time_ms"])})
    ranked = sorted(after.values(),key=lambda r:(r["load"]["generation"],r["npu_id"]))
    checks["exact_rank_times_1ns"] = all(r["arrival_time_ms"]==i*1e-6 for i,r in enumerate(ranked))
    checks["load_arrival_fields_match"] = all(r["load"]["arrival_ms"]==r["load"]["arrival_time"]==r["arrival_time_ms"] for r in ranked)
    checks["old_all_zero"] = all(r["arrival_time_ms"]==0 for r in before.values())
    checks["per_lane_order_preserved"] = all(
        [r["request_id"] for r in sorted((r for r in before.values() if r["npu_id"]==n),key=lambda r:r["request_id"])]
        == [r["request_id"] for r in sorted((r for r in after.values() if r["npu_id"]==n),key=lambda r:r["arrival_time_ms"])]
        for n in range(old["metadata"]["num_npu"]))
    original_nonarrival = analysis.value_hash({"placements":old["placements"],"requests":nonarrival_before})
    changed_nonarrival = analysis.value_hash({"placements":new["placements"],"requests":nonarrival_after})
    checks["same_nonarrival_fingerprint"] = original_nonarrival==changed_nonarrival
    return {"checks":checks,"all_checks_pass":all(checks.values()),"request_count":len(before),
        "changed_fields_and_counts":dict(fields),"old_fingerprint":old["input_fingerprint"],
        "new_fingerprint":new["input_fingerprint"],"nonarrival_fingerprint":original_nonarrival,
        "maximum_arrival_ms":max(r["arrival_time_ms"] for r in ranked),
        "maximum_arrival_us":1000*max(r["arrival_time_ms"] for r in ranked),
        "spacing_ns":1,"single_176kib_ssd_command_service_us":176*1024/2**30/40*1e6,
        "old_manifest_sha256":analysis.file_hash(old_path),"new_manifest_sha256":analysis.file_hash(new_path)},changes,role_home


def intervals_and_utilization(result, label, home, left=0.0, right=3000.0):
    records, windows = [], []
    batches = result["summary"]["microbatch_metrics"]
    n = result["metadata"]["num_npu"]
    for batch in batches:
        rid = batch["member_request_ids"][0]
        a,b = batch["admission_time_ms"],batch["completion_time_ms"]
        if analysis.overlap(a,b,left,right)>0:
            records.append({"panel":label,"strategy":result["strategy"],"input_fingerprint":result["input_fingerprint"],
                "npu_id":batch["npu_id"],"request_id":rid,"home_ssu":home[rid],
                "admission_time_ms":a,"completion_time_ms":b,"clipped_start_ms":max(left,a),"clipped_end_ms":min(right,b)})
    for start in range(int(left),int(right),100):
        end = min(start+100,right)
        compute, active, by_home = [0.0]*n,[0.0]*n,[0.0]*8
        for batch in batches:
            i,rid = batch["npu_id"],batch["member_request_ids"][0]
            a = analysis.overlap(batch["admission_time_ms"],batch["completion_time_ms"],start,end)
            c = sum(analysis.overlap(layer["compute_start_ms"],layer["compute_end_ms"],start,end) for layer in batch["layer_metrics"])
            compute[i] += c;active[i] += a;by_home[home[rid]] += a
        windows.append({"panel":label,"strategy":result["strategy"],"input_fingerprint":result["input_fingerprint"],
            "start_ms":start,"end_ms":end,"compute_npu_ms":sum(compute),"active_npu_ms":sum(active),
            "stall_npu_ms":sum(active)-sum(compute),"idle_npu_ms":n*(end-start)-sum(active),
            "fleet_utilization":sum(compute)/(n*(end-start)),"active_npu_ms_by_home_ssu":by_home})
    checks = {}
    for i in range(n):
        own = sorted((r for r in records if r["npu_id"]==i),key=lambda r:r["clipped_start_ms"])
        checks[f"npu{i}_admitted_intervals_do_not_overlap"] = all(a["clipped_end_ms"]<=b["clipped_start_ms"]+1e-8 for a,b in zip(own,own[1:]))
    checks["active_interval_total_equals_100ms_accounting"] = abs(sum(r["clipped_end_ms"]-r["clipped_start_ms"] for r in records)
        -sum(w["active_npu_ms"] for w in windows))<1e-6
    for saved in result["windows"]:
        chunks = [w for w in windows if saved["start_ms"]<=w["start_ms"] and w["end_ms"]<=saved["end_ms"]]
        if sum(w["end_ms"]-w["start_ms"] for w in chunks)==saved["end_ms"]-saved["start_ms"]:
            actual = sum(w["compute_npu_ms"] for w in chunks)/(n*(saved["end_ms"]-saved["start_ms"]))
            checks[f"U_{saved['start_ms']:g}_{saved['end_ms']:g}_matches_saved"] = abs(actual-saved["mean_npu_utilization"])<1e-9
    return records,windows,checks


def plot(panels, records, windows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
    colors = ["#4477AA","#EE7733","#228833","#CCBB44","#AA3377","#66CCEE","#882255","#44AA99"]
    fig = plt.figure(figsize=(15,8.2))
    grid = fig.add_gridspec(2,3,height_ratios=(4.5,1.35),hspace=.16,wspace=.10)
    for j,(key,title,result) in enumerate(panels):
        ax = fig.add_subplot(grid[0,j])
        ax.set_facecolor("#EEEEEE")
        for n in range(32):
            for s in range(8):
                spans = [(r["clipped_start_ms"],r["clipped_end_ms"]-r["clipped_start_ms"])
                         for r in records if r["panel"]==key and r["npu_id"]==n and r["home_ssu"]==s]
                if spans:
                    ax.broken_barh(spans,(n-.43,.86),facecolors=colors[s],linewidth=0)
        ax.set_xlim(0,3000);ax.set_ylim(31.65,-.65)
        ax.set_yticks(range(32),[str(i) for i in range(32)] if j==0 else [""]*32)
        ax.tick_params(axis="y",labelsize=7,length=2)
        ax.tick_params(axis="x",labelbottom=False)
        ax.set_title(title,fontsize=11,pad=9)
        if j==0:ax.set_ylabel("Actual execution NPU ID")
        for t in (1000,2000):ax.axvline(t,color="black",linestyle="--",linewidth=.7,alpha=.5)
        ax.spines[["top","right"]].set_visible(False)
        lower = fig.add_subplot(grid[1,j],sharex=ax)
        own = [w for w in windows if w["panel"]==key]
        lower.stairs([100*w["fleet_utilization"] for w in own],[w["start_ms"] for w in own]+[own[-1]["end_ms"]],
                     color="#333333",linewidth=1.5,baseline=None)
        lower.set_ylim(0,103);lower.set_yticks((0,25,50,75,100))
        lower.set_xticks((0,1000,2000,3000))
        if j==0:lower.set_ylabel("Compute U (%)")
        else:lower.tick_params(axis="y",labelleft=False)
        lower.set_xlabel("Absolute simulation time (ms)")
        lower.grid(axis="y",alpha=.2)
        lower.spines[["top","right"]].set_visible(False)
        for t in (1000,2000):lower.axvline(t,color="black",linestyle="--",linewidth=.7,alpha=.5)
    fig.suptitle("Data-source SSU of each admitted request: simultaneous arrivals can align all 32 queue fronts",fontsize=13,y=.99)
    handles = [Patch(facecolor=c,label=f"SSU {s}") for s,c in enumerate(colors)]+[Patch(facecolor="#EEEEEE",label="No admitted request")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.953),ncol=9,fontsize=8,frameon=False)
    fig.text(.5,.055,"Colored spans are request admission-to-completion, including computation and waiting; they are not SSD activity. Lower plots recompute each real 100 ms window.",ha="center",fontsize=8)
    fig.text(.5,.025,"Only arrival fields change: rank(generation, original NPU) x 1 ns, maximum 2.735 us. C, V, SSD placement, request IDs and original NPU binding are preserved.",ha="center",fontsize=8)
    fig.subplots_adjust(top=.865,bottom=.14,left=.055,right=.985)
    analysis.save_figure(fig,output,"04_arrival_counterfactual_home_ssu")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    old_manifest=BASE/"screen/inputs/raw32_local.json.gz"
    new_manifest=BASE/"native_assignment/inputs/raw32_local_interleaved_arrivals.json.gz"
    preservation,arrival_rows,home=audit_inputs(old_manifest,new_manifest)
    paths={"baseline_original":one("native_assignment/runs/raw32_local/baseline_native/*.json.gz"),
           "fluid_original":one("native_assignment/runs/raw32_local/deadline_fluid/*.json.gz"),
           "baseline_interleaved":one("native_assignment/runs/raw32_local_interleaved_arrivals/baseline_native/*.json.gz"),
           "fluid_interleaved":one("native_assignment/runs/raw32_local_interleaved_arrivals/deadline_fluid/*.json.gz")}
    results={key:analysis.read_json(path) for key,path in paths.items()}
    cache={};audits={key:analysis.audit_result(paths[key],result,cache) for key,result in results.items()}
    checks={"input_preservation":preservation["all_checks_pass"],"all_four_result_audits":all(a["all_checks_pass"] for a in audits.values()),
        "original_pair_fingerprint":results["baseline_original"]["input_fingerprint"]==results["fluid_original"]["input_fingerprint"]==preservation["old_fingerprint"],
        "interleaved_pair_fingerprint":results["baseline_interleaved"]["input_fingerprint"]==results["fluid_interleaved"]["input_fingerprint"]==preservation["new_fingerprint"],
        "same_submit_seed":{r["submit_seed"] for r in results.values()}=={42}}
    common_core={name for r in results.values() for name in r["core_and_policy_sha256"]}
    checks["same_common_recorded_sources"]=all(len({r["core_and_policy_sha256"][name] for r in results.values() if name in r["core_and_policy_sha256"]})==1 for name in common_core)
    panels=[("baseline_original","Baseline | original t = 0 input",results["baseline_original"]),
            ("fluid_original","B (deadline + fluid) | original t = 0",results["fluid_original"]),
            ("fluid_interleaved","B (deadline + fluid) | interleaved",results["fluid_interleaved"])]
    intervals,windows,interval_checks=[],[],{}
    # Export the fourth cell as data too; the requested visual displays 3 cells.
    for key,result in results.items():
        ir,wr,cr=intervals_and_utilization(result,key,home)
        intervals.extend(ir);windows.extend(wr);interval_checks[key]=cr
    checks["all_chronological_interval_checks"]=all(all(c.values()) for c in interval_checks.values())
    summary_path=BASE/"native_assignment/arrival_tie_comparison.json"
    independent_summary=analysis.read_json(summary_path)
    summary_checks={}
    for key,result in results.items():
        matches=[entry for entry in independent_summary["entries"]
                 if entry["input_fingerprint"]==result["input_fingerprint"] and entry["strategy"]==result["strategy"]]
        summary_checks[key+"_unique_entry"]=len(matches)==1
        if len(matches)!=1:
            continue
        entry=matches[0]
        summary_checks[key+"_makespan"]=abs(entry["makespan_ms"]-result["summary"]["makespan_ms"])<1e-9
        summary_checks[key+"_full_utilization"]=abs(entry["full_run_mean_npu_utilization"]-result["summary"]["fleet_npu_compute_utilization"])<1e-9
        for w in entry["windows"]:
            left,right=w["window_ms"]
            chunks=[row for row in windows if row["panel"]==key and left<=row["start_ms"] and row["end_ms"]<=right]
            compute=sum(row["compute_npu_ms"] for row in chunks)
            active=[sum(row["active_npu_ms_by_home_ssu"][s] for row in chunks) for s in range(8)]
            summary_checks[f"{key}_{left:g}_{right:g}_compute_utilization"]=abs(compute/(32*(right-left))-w["mean_npu_utilization"])<1e-9
            summary_checks[f"{key}_{left:g}_{right:g}_active_by_home"]=all(abs(a-b)<1e-6 for a,b in zip(active,w["active_npu_ms_by_ssu"]))
    checks["independent_summary_agreement"]=all(summary_checks.values())
    grid=[]
    for key,result in results.items():
        for w in result["windows"]:
            grid.append({"cell":key,"strategy":result["strategy"],"input_fingerprint":result["input_fingerprint"],
                "result":str(paths[key]),"start_ms":w["start_ms"],"end_ms":w["end_ms"],
                "utilization":w["mean_npu_utilization"],"active_npu_ms":sum(w["active_ms_by_npu"]),
                "stall_npu_ms":sum(w["io_stall_ms_by_npu"]),"idle_npu_ms":sum(w["idle_ms_by_npu"]),
                "makespan_ms":result["summary"]["makespan_ms"],"full_run_utilization":result["summary"]["fleet_npu_compute_utilization"]})
    analysis.write_csv(args.output/"04_arrival_changes.csv",arrival_rows)
    analysis.write_csv(args.output/"04_four_cell_comparison.csv",grid)
    analysis.write_csv(args.output/"04_admitted_request_intervals.csv",intervals)
    analysis.write_csv(args.output/"04_actual_100ms_utilization.csv",windows)
    artifact={"checks":checks,"all_checks_pass":all(checks.values()),"preservation":preservation,
        "analysis_script_sha256":analysis.file_hash(Path(__file__)),"source_results":{key:{"path":str(paths[key]),"sha256":analysis.file_hash(paths[key])} for key in paths},
        "four_result_audits":audits,"chronological_interval_checks":interval_checks,
        "independent_summary_crosscheck":{"path":str(summary_path),"sha256":analysis.file_hash(summary_path),"checks":summary_checks},
        "figure_panels":[key for key,_,_ in panels],"display_window_ms":[0,3000],"figure_npu_order":list(range(32)),
        "baseline_wrapper_scope":"Both baseline cells use the same native wrapper on the same local Python environment; both fluid cells use the same native deadline+fluid wrapper. An additional same-input raw32 bridge finds exact request/layer timelines between the shared and native baseline wrappers; see native_assignment/raw32_baseline_wrapper_bridge.json.",
        "causal_scope":"Two different arrival inputs. Pair strategies within each fingerprint; compare the order intervention while holding all nonarrival request fields and hardware constant. This demonstrates sensitivity to initial request ordering and its feedback, not universal robustness or production frequency.",
        "display_semantics":"Each colored rectangle is an actual admitted request's lifetime on its executed NPU, colored by immutable home SSU. It includes compute and I/O stalls and does not measure disk busy time, actual concurrent I/O, or SSD throughput."}
    analysis.write_json(args.output/"04_arrival_counterfactual_audit.json",artifact)
    if not all(checks.values()):
        raise AssertionError(checks)
    plot(panels,intervals,windows,args.output)
    print({"all_checks_pass":True,"four_result_count":4,"requests":preservation["request_count"],
           "interval_rows":len(intervals),"100ms_rows":len(windows),"output":str(args.output)})


if __name__=="__main__":
    main()
