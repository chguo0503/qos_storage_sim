#!/usr/bin/env python3
"""Independent fixed/pipeline extension audit; keep it outside the 36-case main study."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import aggregate_mixed_profiles as agg


def result_paths(study, label, variant):
    return sorted(list((study / "runs" / label / variant).glob("*.json.gz")) +
                  list(study.glob(f"*/runs/{label}/{variant}/*.json.gz")))


def plot(output, profiles):
    if not profiles:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline":"#6b7280", "once":"#2563eb", "new_once":"#059669",
              "strategy1_pipeline":"#d97706", "strategy2_pipeline":"#dc2626",
              "strategy3_pipeline":"#7c3aed", "strategy3_fixed":"#0891b2"}
    for label in sorted({r["label"] for r in profiles}):
        rows = [r for r in profiles if r["label"] == label]
        keys = sorted({(r["seq_len_k"],r["nql"]) for r in rows})
        strategies = [s for s in colors if any(r["strategy"] == s for r in rows)]
        pending = [s for s in colors if s not in strategies]
        fig, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        width = .8 / len(strategies)
        for i, strategy in enumerate(strategies):
            p = [next(r for r in rows if r["strategy"] == strategy and (r["seq_len_k"],r["nql"]) == key) for key in keys]
            axis.bar([j-.4+width*(i+.5) for j in range(len(keys))], [100*r["aggregate_compute_fraction"] for r in p],
                     width, label=strategy, color=colors[strategy])
        labels = []
        for key in keys:
            r = next(r for r in rows if (r["seq_len_k"],r["nql"]) == key)
            labels.append(f"{key[0]}K/{key[1]}\n{r['sim_category']}; {100*r['compute_share_of_full_input']:.1f}% of C")
        axis.set_xticks(range(len(keys)),labels)
        axis.set_ylim(0,103)
        axis.set_ylabel("Same complete request cohort: compute / active (%)")
        axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
        axis.legend(loc="upper center",bbox_to_anchor=(.5,-.19),ncol=4,fontsize=8)
        axis.set_title(label + "\nPipeline changes assignment; frozen request population and disk placement stay the same" +
                       ("\nPending: " + ", ".join(pending) if pending else ""),fontsize=10)
        for suffix in ("png","svg","pdf"):
            fig.savefig(output/f"{label}_all_policy_profiles.{suffix}",dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=agg.BASE/"mixed_policy")
    parser.add_argument("--main-study",type=Path,default=agg.BASE/"mixed_varied")
    parser.add_argument("--output",type=Path,default=agg.BASE/"notes/mixed_policy_audit")
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    plan = agg.read(args.study/"plan.json")
    references = ("baseline","once","new_once")
    audits, results, profiles, windows, weighted, cards = [], [], [], [], [], []
    for label,item in plan["inputs"].items():
        source = agg.BASE/item["source_relative_to_study"]
        copied = args.study/"inputs"/f"{label}.json.gz"
        original, replica = agg.read(source), agg.read(copied)
        assert original["requests"] == replica["requests"] and original["placements"] == replica["placements"]
        m, lookup, audit = agg.audit_manifest(copied)
        assert m["input_fingerprint"] == item["frozen_input_fingerprint"] == original["input_fingerprint"]
        audits.append(audit)
        jobs = [{"strategy":s,"variant":s,"input":label,"config":{"assignment":"fixed"},"source_group":"main_reference"}
                for s in references]
        jobs += [{**j,"source_group":"policy_extension"} for j in plan["jobs"] if j["input"] == label]
        signature = None
        for job in jobs:
            group = job["source_group"]
            root = args.main_study if group == "main_reference" else args.study
            paths = result_paths(root,label,job["variant"])
            if not paths:
                results.append({"label":label,"strategy":job["variant"],"policy_strategy":job["strategy"],
                                "assignment":job["config"]["assignment"],"source_group":group,
                                "input_fingerprint":m["input_fingerprint"],"status":"pending",
                                "note":"No complete result locally; queued/running/not synced is not a zero outcome."})
                continue
            assert len(paths) == 1,(label,job["variant"],paths)
            result,pr,wr,wp,cr = agg.aggregate_result(paths[0],job["strategy"],m,lookup,
                                                     assignment=job["config"]["assignment"],variant=job["variant"])
            current = (result["core_and_policy_sha256"],result["stress_runner_sha256"],result["submit_seed"])
            assert signature is None or signature == current,"Do not compare unequal source versions or submission seeds"
            signature = current
            result["source_group"] = group
            results.append(result);profiles.extend(pr);windows.extend(wr);weighted.extend(wp);cards.extend(cr)
    run_index = {(r["label"],r["strategy"]):r for r in results}
    window_index = {(r["label"],r["strategy"],r["window_start_ms"]):r for r in windows}
    profile_index = {(r["label"],r["strategy"],r["seq_len_k"],r["nql"]):r for r in profiles}
    run_rows,run_pairs,profile_pairs = [],[],[]
    for r in results:
        row = {k:r[k] for k in ("label","strategy","policy_strategy","assignment","source_group","input_fingerprint","status")}
        if r["status"] == "complete":
            w1,w2 = [window_index[r["label"],r["strategy"],start] for start in (1000,2000)]
            row.update(U1=w1["utilization"],U2=w2["utilization"],all_active_1=w1["all_active"],all_active_2=w2["all_active"],
                       makespan_ms=r["makespan_ms"],full_run_U=r["fleet_full_run_utilization"],
                       short_cohort_U=r["short"]["aggregate_compute_fraction"],short_SLO=r["short"]["admission_slo_pass_rate"],
                       all_SLO=r["all_requests"]["admission_slo_pass_rate"],reassigned_requests=r["reassigned_request_count"],
                       assigned_mean_hottest_ssu_rho=r["assigned_mean_hottest_ssu_rho"],
                       minimum_assigned_compute_ms=min(r["assigned_ideal_compute_ms_by_npu"]),
                       maximum_assigned_compute_ms=max(r["assigned_ideal_compute_ms_by_npu"]),result_path=r["path"],result_sha256=r["sha256"])
        run_rows.append(row)
        if r["status"] != "complete" or r["source_group"] != "policy_extension":
            continue
        for reference in references:
            baseline = run_index[r["label"],reference]
            assert baseline["status"] == "complete"
            pair = {"label":r["label"],"candidate":r["strategy"],"candidate_assignment":r["assignment"],
                    "reference":reference,"input_fingerprint":r["input_fingerprint"],
                    "delta_U1_pp":100*(w1["utilization"]-window_index[r["label"],reference,1000]["utilization"]),
                    "delta_U2_pp":100*(w2["utilization"]-window_index[r["label"],reference,2000]["utilization"]),
                    "delta_makespan_percent":100*(r["makespan_ms"]/baseline["makespan_ms"]-1),
                    "delta_short_cohort_U_pp":100*(r["short"]["aggregate_compute_fraction"]-baseline["short"]["aggregate_compute_fraction"]),
                    "all_active_both_windows":w1["all_active"] and w2["all_active"]}
            run_pairs.append(pair)
            for p in [p for p in profiles if p["label"] == r["label"] and p["strategy"] == r["strategy"]]:
                b = profile_index[r["label"],reference,p["seq_len_k"],p["nql"]]
                assert p["cohort_id_sha256"] == b["cohort_id_sha256"]
                profile_pairs.append({**{k:pair[k] for k in ("label","candidate","candidate_assignment","reference","input_fingerprint")},
                                      "seq_len_k":p["seq_len_k"],"nql":p["nql"],"role":p["role"],"sim_category":p["sim_category"],
                                      "request_count":p["request_count"],"cohort_id_sha256":p["cohort_id_sha256"],
                                      "delta_cohort_U_pp":100*(p["aggregate_compute_fraction"]-b["aggregate_compute_fraction"]),
                                      "delta_admission_SLO_pp":100*(p["admission_slo_pass_rate"]-b["admission_slo_pass_rate"]),
                                      "delta_mean_stall_ms":p["mean_io_stall_ms"]-b["mean_io_stall_ms"]})
    for name,rows in (("runs",run_rows),("run_pairs",run_pairs),("profiles",profiles),("profile_pairs",profile_pairs),
                      ("windows",windows),("window_profiles",weighted),("cards",cards)):
        agg.csv_dump(args.output/f"{name}.csv",rows)
    counts = {status:sum(r["source_group"] == "policy_extension" and r["status"] == status for r in results)
              for status in ("complete","pending")}
    artifact = {"created_utc":datetime.now(timezone.utc).isoformat(),"policy_extension_status":counts,
                "source_plan":str(args.study/"plan.json"),"source_plan_sha256":agg.sha(args.study/"plan.json"),
                "audit_script_sha256":agg.sha(Path(__file__)),"aggregation_library_sha256":agg.sha(Path(agg.__file__)),
                "method":"Separate extension: compare same complete IDs to all three fixed references; pipeline changes assignment, not request population or SSD blocks.",
                "inputs":audits,"results":results,"runs":run_rows,"run_pairs":run_pairs,"profiles":profiles,"profile_pairs":profile_pairs,
                "windows":windows,"window_profiles":weighted,"cards":cards}
    agg.dump(args.output/"audit.json",artifact)
    lines=["# Mixed policy 独立配对审核","",f"新增8作业：complete={counts['complete']}，pending={counts['pending']}；另列主实验6份固定绑定参照。", "",
           "所有比较使用相同 frozen input fingerprint、完整请求ID cohort、原始C与逐块SSD placement；S1/S2/S3 pipeline额外改变选卡，S3fixed保留原选卡。窗口按实际执行NPU独立积分，cohort按原manifest画像分组。逐项核心源码hash与提交种子一致，不能把pipeline收益或损失说成仅IO排序效果。", "",
           "Assigned mean rho是按实际分卡后完整画像配额重新求和，仍只是平均容量必要条件；不是瞬时截止期或局部窗口的容量保证。", "",
           "|输入|策略/模式|状态|U1|U2|all-active|makespan ms|短组U|短组SLO|重分卡请求数|", "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in run_rows:
        if r["status"] != "complete":
            lines.append(f"|{r['label']}|{r['strategy']}|pending|—|—|—|—|—|—|—|")
        else:
            lines.append(f"|{r['label']}|{r['strategy']}|complete|{100*r['U1']:.3f}%|{100*r['U2']:.3f}%|{r['all_active_1'] and r['all_active_2']}|{r['makespan_ms']:.3f}|{100*r['short_cohort_U']:.3f}%|{100*r['short_SLO']:.3f}%|{r['reassigned_requests']}|")
    lines += ["", "## 相对各固定参照的负例", "", "完整保留：任一窗口U较低或full makespan更长。正Δmakespan表示更慢。", "",
              "|输入|策略|参照|ΔU1 pp|ΔU2 pp|Δmakespan %|", "|---|---|---|---:|---:|---:|"]
    for r in run_pairs:
        if r["delta_U1_pp"] < -1e-9 or r["delta_U2_pp"] < -1e-9 or r["delta_makespan_percent"] > 1e-9:
            lines.append(f"|{r['label']}|{r['candidate']}|{r['reference']}|{r['delta_U1_pp']:+.3f}|{r['delta_U2_pp']:+.3f}|{r['delta_makespan_percent']:+.3f}|")
    lines += ["", "## 全画像完整cohort", "", "|输入|策略|seqK/NQL|模拟器类别|请求数|cohort U|接纳SLO|平均stall ms|L0/warm ms|", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in profiles:
        lines.append(f"|{r['label']}|{r['strategy']}|{r['seq_len_k']}/{r['nql']}|{r['sim_category']}|{r['request_count']}|{100*r['aggregate_compute_fraction']:.3f}%|{r['admission_slo_passed']}/{r['request_count']}|{r['mean_io_stall_ms']:.3f}|{r['mean_layer0_stall_ms']:.3f}/{r['mean_warm_layers_stall_ms']:.3f}|")
    (args.output/"summary.md").write_text("\n".join(lines)+"\n")
    if not args.no_plots:
        plot(args.output,profiles)
    print({"extension":counts,"reference_complete":sum(r["source_group"] == "main_reference" and r["status"] == "complete" for r in results),"output":str(args.output)})


if __name__ == "__main__":
    main()
