#!/usr/bin/env python3
"""Rebuild scientific figures/CSVs from completed native experiment outputs.

Usage: python build_plots.py [--root EXPERIMENT_DIRECTORY]
Only completed matched, strictly underloaded pairs enter scientific plots.
CSV exports retain completed excluded runs and explain the selection status.
No simulator modules or inferred/fabricated experiment values are used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


STRATEGIES = ("baseline", "once")
LABELS = {"baseline": "Baseline FIFO", "once": "Once (5 ms)"}
COLORS = {"baseline": "#4D6C91", "once": "#D88435"}
GIB_TO_GB = 2**30 / 1e9
EPS = 1e-9


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def natural(value):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value))]


def ratios(config):
    a, b = config["profile_A"], config["profile_B"]
    return float(a["read_gib"])/float(b["read_gib"]), float(a["compute_us"])/float(b["compute_us"])


def load_runs(root):
    runs, warnings = [], []
    for analysis_path in sorted((root/"results").glob("*/*/analysis.json")):
        directory = analysis_path.parent
        if directory.name not in STRATEGIES:
            continue
        try:
            analysis, config = read_json(analysis_path), read_json(directory/"config.json")
            if not (directory/"native_summary.json.gz").exists():
                warnings.append(f"Missing native summary: {directory}")
                continue
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"Incomplete file skipped: {directory}: {error}")
            continue
        full = analysis["full"]
        ordinary = full["ordinary_demand"]["max_single_ssu_gb_s"]
        pending = full["ordinary_pending_demand"]["max_single_ssu_gb_s"]
        strict_pass = ordinary < 40-EPS and pending < 40-EPS
        runs.append({"directory": directory, "analysis": analysis, "config": config,
                     "strategy": directory.name, "strict_underload": strict_pass})
    return runs, warnings


def metric(run, name):
    analysis = run["analysis"]
    if name == "fleet":
        return 100*analysis["warm"]["fleet_utilization"]
    if name in ("A", "B"):
        value = analysis["warm"]["by_group"][name]["compute_active_utilization"]
        return None if value is None else 100*value
    if name == "slo_all":
        value = analysis["slo_all_requests"]["rate"]
        return None if value is None else 100*value
    if name == "slo_warm":
        value = analysis["warm"]["slo_admitted"]["rate"]
        return None if value is None else 100*value
    raise KeyError(name)


def pair_runs(runs):
    by_name = defaultdict(dict)
    for run in runs:
        by_name[run["config"]["name"]][run["strategy"]] = run
    pairs = []
    for name, records in by_name.items():
        if set(records) != set(STRATEGIES):
            continue
        configs_equal = records["baseline"]["config"] == records["once"]["config"]
        fp_equal = records["baseline"]["analysis"].get("input_fingerprint") == records["once"]["analysis"].get("input_fingerprint")
        pairs.append({"name": name, "runs": records,
                      "config": records["baseline"]["config"],
                      "matched": configs_equal and fp_equal,
                      "eligible": configs_equal and fp_equal and all(r["strict_underload"] for r in records.values())
                      and all(r["analysis"]["warm"]["all_npus_active_whole_window"] for r in records.values())})
    return pairs, by_name


def write_csv(path, rows):
    fields = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_columns(config):
    x, y = ratios(config)
    row = {"id": config["id"], "mode": config["mode"], "ssu": config["ssu"],
           "x": x, "y": y, "n_A": config.get("n_A"), "n_B": config.get("n_B"),
           "input_counts_A_B": ":".join(map(str, config.get("input_counts", [])))}
    for group in ("A", "B"):
        p = config["profile_"+group]
        row.update({group+"_total_length_k": p["total_length_k"], group+"_nql": p["nql"],
                    group+"_V_MB": p["read_gib"]*2**30/1e6,
                    group+"_C_ms": p["compute_us"]/1000,
                    group+"_B_GB_s": p["read_gib"]*GIB_TO_GB/(p["compute_us"]/1e6),
                    group+"_constructed": p.get("constructed_profile"),
                    group+"_extrapolated": p.get("profile_construction", {}).get("extrapolated")})
    return row


def export_all(runs, pairs, by_name, output):
    paired = {p["name"]: p for p in pairs}
    rows = []
    for run in sorted(runs, key=lambda r: (natural(r["config"]["name"]), r["strategy"])):
        a, c = run["analysis"], run["config"]
        pair = paired.get(c["name"])
        row = {"case": c["name"], **profile_columns(c), "seed": c["seed"], "strategy": run["strategy"],
               "warm_start_ms": a["warm"]["start_ms"], "warm_end_ms": a["warm"]["end_ms"],
               "full_makespan_ms": a["full"]["end_ms"],
               "pair_complete": pair is not None, "pair_matched": pair["matched"] if pair else False,
               "pair_plot_eligible": pair["eligible"] if pair else False,
               "strict_underload_full_both_audits": run["strict_underload"],
               "warm_all_npus_active": a["warm"]["all_npus_active_whole_window"],
               "native_invariants_all_pass": all(a.get("invariants", {}).values()),
               "request_count": a.get("n_requests", a["input_audit"]["request_count"])}
        for key in ("fleet", "A", "B", "slo_all", "slo_warm"):
            row[key+"_pct"] = metric(run, key)
        for window in ("full", "warm"):
            w = a[window]
            row[window+"_fleet_utilization_pct"] = 100*w["fleet_utilization"]
            for kind in ("internal", "layer0"):
                row[window+"_"+kind+"_stall_card_ms"] = w["stall"][kind]["card_ms"]
                row[window+"_"+kind+"_stall_count"] = w["stall"][kind]["count"]
            for audit in ("ordinary_demand", "ordinary_pending_demand"):
                d = w[audit]
                prefix = window+"_"+audit
                row[prefix+"_max_ssu_GB_s"] = d["max_single_ssu_gb_s"]
                row[prefix+"_peak_total_GB_s"] = d["peak_total_gb_s"]
                row[prefix+"_any_overload_ms"] = d["any_ssu_overload_ms"]
                row[prefix+"_any_ge36_ms"] = d["any_ssu_at_or_above_near_ms"]
                row[prefix+"_any_ge36_pct"] = 100*d["any_ssu_at_or_above_near_fraction"]
        row["slo_all_count"] = a["slo_all_requests"]["count"]
        row["slo_warm_admitted_count"] = a["warm"]["slo_admitted"]["count"]
        row["source_directory"] = str(run["directory"])
        rows.append(row)
    write_csv(output/"all_runs.csv", rows)


def grouped_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        c = pair["config"]
        groups[(c["mode"], c["id"], int(c["ssu"]))].append(pair)
    for group in groups.values():
        group.sort(key=lambda p: p["config"]["seed"])
    return groups


def export_paired(groups, output):
    rows = []
    for key in sorted(groups, key=lambda k: (k[0], natural(k[1]), k[2])):
        group = groups[key]
        expected = 3 if key[0] == "random" else 1
        row = {**profile_columns(group[0]["config"]), "paired_seed_count": len(group),
               "seeds": ",".join(str(p["config"]["seed"]) for p in group),
               "expected_seed_count": expected,
               "all_pairs_matched": all(p["matched"] for p in group),
               "all_pairs_underloaded_and_active": all(p["eligible"] for p in group),
               "series_point_complete_and_eligible": len(group) == expected and all(p["eligible"] for p in group)}
        for strategy in STRATEGIES:
            for key_metric in ("fleet", "A", "B", "slo_all", "slo_warm"):
                values = [metric(p["runs"][strategy], key_metric) for p in group]
                values = [v for v in values if v is not None]
                for stat, fn in (("mean", fmean), ("min", min), ("max", max)):
                    row[f"{strategy}_{key_metric}_{stat}_pct"] = fn(values) if values else None
        for key_metric in ("fleet", "A", "B", "slo_all", "slo_warm"):
            deltas = [(metric(p["runs"]["once"], key_metric), metric(p["runs"]["baseline"], key_metric)) for p in group]
            values = [once-base for once, base in deltas if once is not None and base is not None]
            row["delta_"+key_metric+"_mean_pp"] = fmean(values) if values else None
        row["worst_full_ordinary_peak_GB_s"] = max(p["runs"][s]["analysis"]["full"]["ordinary_demand"]["max_single_ssu_gb_s"] for p in group for s in STRATEGIES)
        row["worst_full_pending_peak_GB_s"] = max(p["runs"][s]["analysis"]["full"]["ordinary_pending_demand"]["max_single_ssu_gb_s"] for p in group for s in STRATEGIES)
        rows.append(row)
    write_csv(output/"paired_summary.csv", rows)


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DFE4EA", linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(fig, name, output):
    for suffix in ("png", "pdf"):
        fig.savefig(output/(name+"."+suffix), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fixed_plot(pairs, output):
    choices = {}
    for pair in pairs:
        c = pair["config"]
        if c["mode"] != "fixed" or not pair["eligible"]:
            continue
        previous = choices.get(c["id"])
        if previous is None or (c["ssu"], -c["seed"]) > (previous["config"]["ssu"], -previous["config"]["seed"]):
            choices[c["id"]] = pair
    selected = [choices[k] for k in sorted(choices, key=natural)]
    if not selected:
        return {"status": "no eligible complete fixed pairs"}
    fig, axes = plt.subplots(2, 1, figsize=(max(10, .95*len(selected)), 7.8), sharex=True)
    xs = list(range(len(selected)))
    labels = []
    for pair in selected:
        c = pair["config"]; x, y = ratios(c)
        labels.append(f'{c["id"]} | {c["n_A"]}A:{c["n_B"]}B | S={c["ssu"]}\nx={x:.2f}, y={y:.2f}')
    for ax, key_metric, title in zip(axes, ("fleet", "B"), ("Fleet NPU utilization", "B compute / B active time")):
        for index, strategy in enumerate(STRATEGIES):
            positions = [x+(-.19 if index == 0 else .19) for x in xs]
            values = [metric(p["runs"][strategy], key_metric) for p in selected]
            bars = ax.bar(positions, values, width=.36, color=COLORS[strategy], label=LABELS[strategy])
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=8, padding=2)
        ax.set_ylim(0, 110); ax.set_yticks([0,20,40,60,80,100]); ax.set_ylabel("Utilization (%)")
        ax.set_title(title, loc="left", fontsize=11); style_axis(ax)
    axes[0].legend(loc="lower right", ncol=2, frameon=False)
    axes[1].set_xticks(xs, labels, fontsize=8)
    fig.suptitle("Fixed concurrent A/B roles | 8 NPUs | warm 2-4 s", fontsize=14)
    fig.text(.5,.01,"Each point uses the largest eligible S for its ID; both full-run demand audits are below 40 GB/s per SSU.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96))
    save_figure(fig,"fixed_concurrent_utilization",output)
    return {"selected_cases": [p["name"] for p in selected]}


def series_plot(groups, series, output):
    selected = []
    skipped = []
    for (mode, identity, ssu), group in groups.items():
        include = mode == "random" and (identity.startswith("XY12_") if series == "x" else (re.fullmatch(r"Y\d+",identity) is not None or identity == "XY12_20"))
        if not include:
            continue
        ready = ssu == 1 and {int(p["config"]["seed"]) for p in group} == {7,19,43} and all(p["eligible"] for p in group)
        if ready:
            selected.append(group)
        else:
            skipped.append({"id":identity,"ssu":ssu,"paired_seeds":[p["config"]["seed"] for p in group],"all_pairs_eligible":all(p["eligible"] for p in group)})
    index = 0 if series == "x" else 1
    selected.sort(key=lambda group: ratios(group[0]["config"])[index])
    if not selected:
        return {"status":"waiting for eligible three-seed points","skipped":skipped}
    fig, axes = plt.subplots(2,1,figsize=(8.6,7.5),sharex=True)
    xvalues = [ratios(g[0]["config"])[index] for g in selected]
    for ax, key_metric, title in zip(axes,("fleet","B"),("Fleet NPU utilization","B compute / B active time")):
        all_seed_values = [metric(p["runs"][strategy],key_metric) for group in selected for p in group for strategy in STRATEGIES]
        local_lower = max(0.0, math.floor((min(all_seed_values)-2.5)*2)/2)
        for strategy in STRATEGIES:
            values = [[metric(p["runs"][strategy],key_metric) for p in group] for group in selected]
            means = [fmean(v) for v in values]
            error = [[mean-min(v) for mean,v in zip(means,values)], [max(v)-mean for mean,v in zip(means,values)]]
            ax.errorbar(xvalues,means,yerr=error,marker="o",capsize=4,linewidth=1.6,color=COLORS[strategy],label=LABELS[strategy])
        ax.set_ylim(local_lower,101); ax.yaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,2.5,5,10]))
        ax.set_ylabel("Utilization (%)"); style_axis(ax)
        ax.set_title(title,loc="left",fontsize=11)
        ax.text(.01,.035,f"Displayed y range: {local_lower:g}-101%",transform=ax.transAxes,fontsize=8,color="#606872")
    axes[0].legend(frameon=False,loc="lower right")
    axes[1].set_xticks(xvalues,[f"{x:.2f}" for x in xvalues])
    axes[1].set_xlabel("x = V_A / V_B" if series=="x" else "y = C_A / C_B")
    fixed = "y approximately 12" if series=="x" else "x approximately 10.5"
    fig.suptitle(f"Random A/B on every NPU | {fixed} | S=1",fontsize=13)
    fig.text(.5,.01,"8 NPUs; input A:B=1:12; warm 2-4 s. Points: 3-seed mean; whiskers: minimum/maximum.\nLocal y-axis ranges. Only matched seeds 7, 19, 43 passing both full-run underload audits are shown.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96))
    save_figure(fig,f"random_{series}_utilization",output)
    return {"selected_ids":[g[0]["config"]["id"] for g in selected],"skipped":skipped,"y_center_includes_XY12_20":series=="y",
            "X16_control":"CSV only; excluded from y-approximately-12 line"}


def stall_plot(pairs, output):
    targets=(("E1","fixed",7),("C1","fixed",7),("XY12_16","random",7))
    selected=[]; missing=[]
    for identity, mode, seed in targets:
        options=[p for p in pairs if p["config"]["id"]==identity and p["config"]["mode"]==mode and p["config"]["seed"]==seed and p["eligible"]]
        if options:
            selected.append(max(options,key=lambda p:p["config"]["ssu"]))
        else:
            missing.append(f"{identity}/{mode}/seed{seed}")
    if not selected:
        return {"status":"waiting for target pairs","missing":missing}
    fig,axes=plt.subplots(1,len(selected),figsize=(4.2*len(selected),4.5),squeeze=False)
    for ax,pair in zip(axes[0],selected):
        internal=[pair["runs"][s]["analysis"]["warm"]["stall"]["internal"]["card_ms"] for s in STRATEGIES]
        initial=[pair["runs"][s]["analysis"]["warm"]["stall"]["layer0"]["card_ms"] for s in STRATEGIES]
        ax.bar([0,1],internal,width=.6,color="#4D6C91",label="Internal layers")
        ax.bar([0,1],initial,bottom=internal,width=.6,color="#D88435",label="Layer 0")
        for j,total in enumerate([a+b for a,b in zip(internal,initial)]):
            ax.annotate(f"{total:.1f}",(j,total),xytext=(0,4),textcoords="offset points",ha="center",fontsize=9)
        ax.set_xticks([0,1],["Baseline FIFO","Once"]); ax.set_ylabel("Accumulated stall (card-ms)"); style_axis(ax)
        ax.set_ylim(0,max(1.,max(a+b for a,b in zip(internal,initial)))*1.22)
        c=pair["config"];ax.set_title(f'{c["id"]} | {c["mode"]} | S={c["ssu"]}',fontsize=11)
    axes[0][0].legend(frameon=False,loc="upper right",fontsize=9)
    fig.suptitle("Warm 2-4 s | all stall retained, including request boundaries",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));save_figure(fig,"stall_decomposition",output)
    return {"selected_cases":[p["name"] for p in selected],"missing":missing}


def demand_events(run):
    with gzip.open(run["directory"]/"native_summary.json.gz","rt",encoding="utf-8") as stream:
        summary=json.load(stream)
    metadata={int(r["request_id"]):r for r in read_json(run["directory"]/"requests.json")}
    ssu=run["config"]["ssu"]; events=defaultdict(lambda:[0.]*ssu)
    events[0.];events[float(summary["makespan_ms"])]
    for batch in summary["microbatch_metrics"]:
        rid=batch["member_request_ids"][0]; m=metadata[rid]
        rates=[v*GIB_TO_GB/(m["per_layer_us"]/1e6) for v in m["disk_gib"]]
        for s,rate in enumerate(rates):
            events[float(batch["admission_time_ms"])][s]+=rate
            events[float(batch["completion_time_ms"])][s]-=rate
    times=sorted(events); demand=[0.]*ssu; curves=[[] for _ in range(ssu)]
    for t in times:
        for s in range(ssu):
            demand[s]+=events[t][s]
            if abs(demand[s])<1e-9:demand[s]=0.
            curves[s].append(demand[s])
    # Cross-check the independently reconstructed profile-demand plot.
    expected=run["analysis"]["full"]["ordinary_demand"]["peak_gb_s_by_ssu"]
    for s in range(ssu):
        if not math.isclose(max(curves[s][:-1]),expected[s],rel_tol=1e-9,abs_tol=1e-7):
            raise AssertionError("Reconstructed plot peak differs from saved audit")
    return [t/1000 for t in times],curves


def c1_demand_plot(pairs,output):
    options=[p for p in pairs if p["config"]["id"]=="C1" and p["config"]["mode"]=="fixed" and p["config"]["ssu"]==2 and p["eligible"]]
    if not options:return {"status":"waiting for eligible C1 two-SSU pair"}
    pair=min(options,key=lambda p:p["config"]["seed"])
    fig,axes=plt.subplots(2,1,figsize=(10,6.2),sharex=True,sharey=True)
    colors=("#278083","#805B98"); max_time=0.
    for ax,strategy in zip(axes,STRATEGIES):
        times,curves=demand_events(pair["runs"][strategy]);max_time=max(max_time,times[-1])
        for s,values in enumerate(curves):ax.step(times,values,where="post",label=f"SSU {s}",color=colors[s],linewidth=1.25)
        ax.axhline(40,color="#BD4B45",linestyle="--",linewidth=1.2,label="Capacity 40 GB/s")
        ax.axhline(36,color="#868D95",linestyle=":",linewidth=1.,label="90% capacity")
        ax.axvspan(2,4,color="#DCE4EC",alpha=.25,zorder=0)
        ax.set_title(LABELS[strategy],loc="left",fontsize=11);ax.set_ylabel("Nominal demand (GB/s)");style_axis(ax)
        ax.set_ylim(0,44)
    axes[0].legend(loc="lower left",ncol=4,frameon=False,fontsize=8.5)
    axes[-1].set_xlim(0,max_time);axes[-1].set_xlabel("Time (s)")
    fig.suptitle("C1 | 8 NPUs, 2 SSUs, ring hash | full trajectory",fontsize=13)
    fig.text(.5,.01,"Nominal current-request demand: sum of per-SSU V/C. Extra cross-request L0 burst excluded; all time retained.\nShaded interval: warm 2-4 s. Curves are demand, not actual service bandwidth.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.96));save_figure(fig,"C1_full_nominal_demand",output)
    return {"selected_case":pair["name"],"independent_peak_check_passed":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();root=args.root.resolve();output=root/"outputs";output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"ps.fonttype":42})
    runs,warnings=load_runs(root);pairs,by_name=pair_runs(runs);groups=grouped_pairs(pairs)
    export_all(runs,pairs,by_name,output);export_paired(groups,output)
    manifest={"completed_run_count":len(runs),"matched_pair_count":sum(p["matched"] for p in pairs),
              "eligible_pair_count":sum(p["eligible"] for p in pairs),"warnings":warnings,
              "selection":"Both strategies completed, identical input/config, full ordinary and internal pending peaks <40 GB/s, all NPUs active throughout warm window",
              "fixed_concurrent_utilization":fixed_plot(pairs,output),
              "random_x_utilization":series_plot(groups,"x",output),
              "random_y_utilization":series_plot(groups,"y",output),
              "stall_decomposition":stall_plot(pairs,output),
              "C1_full_nominal_demand":c1_demand_plot(pairs,output)}
    (output/"plot_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
