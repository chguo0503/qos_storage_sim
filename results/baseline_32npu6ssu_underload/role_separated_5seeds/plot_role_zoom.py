#!/usr/bin/env python3
"""Plot one explicitly selected role-separated timing example; no simulation.

Only existing request/microbatch layer logs are used. Read lifetimes include
queueing, issue and transfer time, and are never interpreted as SSD service.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OUT = HERE / "figures" / "separate"
LABEL = "role_l11_s21_seed7_random"
FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
# Match the full-window figures' shared plot_case.COLORS exactly.
SHORT, LONG, STALL, READ = "#26749b", "#72b5a5", "#efa640", "#DACDEA"


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b, lo, hi):
    return max(0.0, min(b, hi)-max(a, lo))


def selected_record(batch, previous, layer, requests):
    rid = batch["member_request_ids"][0]
    request = requests[rid]
    c = previous["compute_end_ms"]-previous["compute_start_ms"]
    stall = layer["compute_start_ms"]-previous["compute_end_ms"]
    lifetime = layer["io_ready_time_ms"]-layer["io_start_time_ms"]
    assert math.isclose(c, previous["compute_duration_ms"], abs_tol=1e-8)
    assert math.isclose(layer["io_start_time_ms"], previous["compute_start_ms"], abs_tol=1e-8)
    assert math.isclose(stall, layer["io_barrier_wait_ms"], abs_tol=1e-8)
    assert layer["io_ready_time_ms"] <= layer["compute_start_ms"]+1e-8
    assert previous["compute_start_ms"] >= batch["admission_time_ms"]-1e-8
    return {"npu_id":batch["npu_id"], "request_id":rid, "batch_id":batch["batch_id"],
            "admission_time_ms":batch["admission_time_ms"], "completion_time_ms":batch["completion_time_ms"],
            "request_own_compute_ms":request["own_compute_ms"],
            "previous_layer":previous, "selected_read_layer":layer,
            "previous_layer_compute_ms":c, "exposed_stall_ms":stall,
            "read_lifetime_ms":lifetime, "stall_over_one_layer_compute":stall/c,
            "read_ready_slack_before_previous_compute_ends_ms":previous["compute_end_ms"]-layer["io_ready_time_ms"]}


def select_case(raw):
    batches = raw["summary"]["microbatch_metrics"]
    requests = {r["request_id"]:r for r in raw["summary"]["request_metrics"]}
    candidates = []
    for batch in batches:
        if batch["npu_id"] < 11:
            continue
        layers = sorted(batch["layer_metrics"], key=lambda l:l["layer"])
        for k, layer in enumerate(layers[1:], 1):
            previous = layers[k-1]
            a,b = previous["compute_end_ms"], layer["compute_start_ms"]
            if 2000 <= a < b <= 4000:
                candidates.append((b-a, -batch["npu_id"], -batch["member_request_ids"][0], -layer["layer"], batch, previous, layer))
    assert candidates
    winner = max(candidates, key=lambda x:x[:4])
    short = selected_record(*winner[-3:], requests)
    lo, hi = short["previous_layer"]["compute_end_ms"], short["selected_read_layer"]["compute_start_ms"]
    references = []
    for batch in batches:
        if batch["npu_id"] >= 11:
            continue
        layers = sorted(batch["layer_metrics"], key=lambda l:l["layer"])
        for k, layer in enumerate(layers[1:], 1):
            previous = layers[k-1]
            shared = overlap(layer["io_start_time_ms"], layer["io_ready_time_ms"], lo, hi)
            if shared > 0 and layer["io_ready_time_ms"] <= previous["compute_end_ms"]+1e-8:
                references.append((shared, -batch["npu_id"], -batch["member_request_ids"][0], -layer["layer"], batch, previous, layer))
    assert references
    match = max(references, key=lambda x:x[:4])
    long = selected_record(*match[-3:], requests)
    assert short["exposed_stall_ms"] > 0 and abs(long["exposed_stall_ms"]) < 1e-8
    # Include the entire long compute interval and both selected read lifetimes,
    # plus at least 0.5 ms on either side; snap endpoints outward to integer ms.
    first = min(short["previous_layer"]["compute_start_ms"], long["previous_layer"]["compute_start_ms"])
    last = max(short["selected_read_layer"]["compute_end_ms"], long["previous_layer"]["compute_end_ms"])
    window = (math.floor(first-0.5), math.ceil(last+0.5))
    assert 2000 <= window[0] < window[1] <= 4000
    return short, long, window, {"eligible_positive_short_stall_count":len(candidates),
        "eligible_zero_stall_long_reference_count":len(references), "selected_read_overlap_with_short_stall_ms":match[0]}


def lane_events(raw, npu, lo, hi):
    events = []
    for batch in raw["summary"]["microbatch_metrics"]:
        if batch["npu_id"] != npu:
            continue
        previous = batch["admission_time_ms"]
        rid = batch["member_request_ids"][0]
        for layer in sorted(batch["layer_metrics"], key=lambda l:l["layer"]):
            for kind,a,b in (("io_stall",previous,layer["compute_start_ms"]),
                             ("compute",layer["compute_start_ms"],layer["compute_end_ms"])):
                if overlap(a,b,lo,hi)>1e-9:
                    events.append({"kind":kind,"npu_id":npu,"request_id":rid,"layer_zero_based":layer["layer"],
                                   "actual_start_ms":a,"actual_end_ms":b,"plot_start_ms":max(a,lo),"plot_end_ms":min(b,hi)})
            previous = layer["compute_end_ms"]
    events.sort(key=lambda e:(e["plot_start_ms"],e["kind"]))
    assert math.isclose(math.fsum(e["plot_end_ms"]-e["plot_start_ms"] for e in events),hi-lo,abs_tol=1e-7)
    return events


def main():
    results = list((HERE/"runs"/LABEL/"baseline").glob("*.json.gz"))
    assert len(results)==1
    result_path=results[0]
    raw=read(result_path)
    manifest_path=HERE/"inputs"/(LABEL+".json.gz")
    manifest=read(manifest_path)
    command=read(result_path.parent/"command.json")
    from run_coflow_experiments import source_files
    checks={"successful_completed_run":command.get("status")=="complete" and command.get("returncode")==0,
            "correct_strategy_seed":raw["strategy"]=="baseline" and raw["submit_seed"]==7,
            "input_fingerprint_matches":raw["input_fingerprint"]==manifest["input_fingerprint"]==command["input_fingerprint"],
            "dimensions":tuple(raw["summary"][k] for k in ("num_npu","num_ssu","n_layers","batch_size"))==(32,6,8,1),
            "all_original_invariants":all(raw["summary"]["invariants"].values()),
            "complete_core29_hashset":set(raw["core_and_policy_sha256"])==set(source_files()) and len(source_files())==29,
            "core_source_hashes_current":all(sha(ROOT/p)==value for p,value in raw["core_and_policy_sha256"].items()),
            "stress_runner_hash_current":raw["stress_runner_sha256"]==sha(ROOT/"run_baseline_npu32_stress.py"),
            "fixed_11_long_21_short_roles":manifest["metadata"]["assigned_role_by_npu"]==["long"]*11+["short"]*21}
    assert all(checks.values()),checks
    short,long,(lo,hi),selection_counts=select_case(raw)
    short_events=lane_events(raw,short["npu_id"],lo,hi)
    long_events=lane_events(raw,long["npu_id"],lo,hi)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=str(FONT)).get_name(),
                         "font.size":42,"axes.unicode_minus":False,"pdf.fonttype":42,"ps.fonttype":42,
                         "svg.fonttype":"path","savefig.facecolor":"white"})
    fig=plt.figure(figsize=(42,24),facecolor="white")
    ax=fig.add_axes((.205,.30,.775,.535))
    ys={"short":7.6,"short_read":5.65,"long":3.35,"long_read":1.30}
    height=1.15
    def bar(a,b,y,color,edge="white",hatch=None,zorder=3):
        a,b=max(a,lo),min(b,hi)
        if b>a:
            patch=Rectangle((a,y-height/2),b-a,height,facecolor=color,edgecolor=edge,linewidth=1.2,hatch=hatch,zorder=zorder)
            ax.add_patch(patch)
            return patch
    for name,events,color in (("short",short_events,SHORT),("long",long_events,LONG)):
        for event in events:
            p=bar(event["plot_start_ms"],event["plot_end_ms"],ys[name],color if event["kind"]=="compute" else STALL)
            p.set_gid(f"npu{event['npu_id']}_request{event['request_id']}_L{event['layer_zero_based']}_{event['kind']}")
    sp,sl=short["previous_layer"],short["selected_read_layer"]
    lp,ll=long["previous_layer"],long["selected_read_layer"]
    for name,layer in (("short_read",sl),("long_read",ll)):
        bar(layer["io_start_time_ms"],layer["io_ready_time_ms"],ys[name],READ,edge="#89749C",hatch="//")
    bar(sp["compute_start_ms"],sp["compute_end_ms"],ys["short"],SHORT,edge="#164D68",zorder=4)
    bar(lp["compute_start_ms"],lp["compute_end_ms"],ys["long"],LONG,edge="#3B7063",zorder=4)
    ax.text((sp["compute_end_ms"]+sl["compute_start_ms"])/2,ys["short"],
            f"I/O stall：{short['exposed_stall_ms']:.3f} ms\n约为自身单层 C 的 {short['stall_over_one_layer_compute']:.2f} 倍",
            ha="center",va="center",fontsize=42,linespacing=1.05,zorder=6)
    ax.text((lp["compute_start_ms"]+lp["compute_end_ms"])/2,ys["long"],
            f"第4层计算 (L3)：{long['previous_layer_compute_ms']:.3f} ms",
            ha="center",va="center",fontsize=42,color="#142F28",weight="bold",zorder=6)
    for name,record,human,layer in (("short_read",short,6,sl),("long_read",long,5,ll)):
        ax.text((layer["io_start_time_ms"]+layer["io_ready_time_ms"])/2,ys[name],
                f"第{human}层读取：{record['read_lifetime_ms']:.3f} ms\n{layer['io_start_time_ms']:.3f} → {layer['io_ready_time_ms']:.3f} ms",
                ha="center",va="center",fontsize=42,linespacing=1.05,zorder=6)
    ax.annotate(f"第5层计算 (L4)\nC = {short['previous_layer_compute_ms']:.3f} ms",
                xy=((sp["compute_start_ms"]+sp["compute_end_ms"])/2,ys["short"]+height/2),
                xytext=(lo+3.9,9.35),ha="center",va="center",fontsize=42,
                arrowprops={"arrowstyle":"-|>","color":SHORT,"lw":2.5},zorder=8)
    ax.annotate("第6层开始计算 (L5)",xy=(sl["compute_start_ms"],ys["short"]+height/2),
                xytext=(lo+18.1,9.35),ha="center",va="center",fontsize=42,
                arrowprops={"arrowstyle":"-|>","color":SHORT,"lw":2.5},zorder=8)
    ax.vlines([sp["compute_end_ms"],sl["io_ready_time_ms"]],ys["short_read"]-.65,ys["short"]+.65,
              colors="#805F30",linestyles="dashed",linewidth=2,zorder=2)
    ax.vlines([ll["io_ready_time_ms"],lp["compute_end_ms"]],.45,ys["long"]+.65,
              colors="#54718A",linestyles="dashed",linewidth=2,zorder=2)
    midpoint=(ll["io_ready_time_ms"]+lp["compute_end_ms"])/2
    ax.text(midpoint,ys["long_read"]+.06,f"提前 {long['read_ready_slack_before_previous_compute_ends_ms']:.3f} ms 就绪\n第5层开始时无需等待",
            ha="center",va="center",fontsize=42,linespacing=1.05,zorder=6)
    ax.annotate("",xy=(ll["io_ready_time_ms"],.45),xytext=(lp["compute_end_ms"],.45),
                arrowprops={"arrowstyle":"<->","lw":2.4,"color":"#3E607D"},zorder=5)
    ax.set_xlim(lo,hi);ax.set_ylim(0,10.5)
    ax.set_xticks([x for x in range(lo,hi+1) if x%4==0]+([] if hi%4==0 else [hi]))
    ax.set_yticks([ys[k] for k in ("short","short_read","long","long_read")])
    ax.set_yticklabels([f"短卡 NPU{short['npu_id']}\n计算 / 等待","短卡第6层 (L5)\n读取生命周期",
                        f"长卡 NPU{long['npu_id']}\n计算 / 等待","长卡第5层 (L4)\n读取生命周期"],fontsize=42)
    ax.tick_params(axis="y",length=0,pad=22);ax.tick_params(axis="x",labelsize=42,pad=14)
    ax.set_xlabel("仿真时间（毫秒）",fontsize=44,labelpad=15)
    ax.grid(axis="x",color="#D8DEE4",alpha=.7,linewidth=1.2,zorder=0)
    for name in ("top","right","left"):
        ax.spines[name].set_visible(False)
    fig.suptitle("固定角色分卡：短卡在等数据，长卡仍能计算",x=.5,y=.976,fontsize=50,weight="bold")
    fig.text(.205,.922,f"11长 / 21短 · Baseline · seed 7 · 同一时间窗 [{lo}, {hi}) ms",fontsize=42)
    fig.text(.205,.872,f"短卡请求 {short['request_id']}（1K/128）；长卡请求 {long['request_id']}（192K/768）",fontsize=42)
    handles=[Patch(facecolor=SHORT,label="短卡计算"),Patch(facecolor=LONG,label="长卡计算"),
             Patch(facecolor=STALL,label="I/O stall"),Patch(facecolor=READ,edgecolor="#89749C",hatch="//",label="读取生命周期")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.53,.205),ncol=4,frameon=False,fontsize=42,columnspacing=1.8)
    notes=["紫色斜线＝I/O释放至数据就绪（含排队与传输），不代表SSD连续服务。",
           "长卡仅为同时段参照；没有逐块队列轨迹，不能认定它就是队头阻塞者。",
           "选样：暖窗[2,4)s内最大内部层短stall；这是局部示例，不代表平均表现。",
           "计算行保留窗口内全部实际事件；读取行仅显示所选层。第1层对应日志L0。"]
    for y,note in zip((.143,.107,.071,.035),notes):
        fig.text(.025,y,note,fontsize=42,color="#34404B")
    OUT.mkdir(parents=True,exist_ok=True)
    files=[]
    for extension in ("png","pdf","svg"):
        path=OUT/("l11_s21_random_zoom."+extension)
        fig.savefig(path,dpi=160,bbox_inches="tight",pad_inches=.25)
        files.append({"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size})
    plt.close(fig)
    evidence={"created_utc":datetime.now(timezone.utc).isoformat(),"script_path":str(Path(__file__).resolve()),"script_sha256":sha(__file__),
              "source_result":str(result_path),"source_result_sha256":sha(result_path),"manifest":str(manifest_path),"manifest_sha256":sha(manifest_path),
              "input_fingerprint":raw["input_fingerprint"],"core_source_sha256":raw["core_and_policy_sha256"],"checks":checks,
              "window_ms":[lo,hi],"warm_selection_window_ms":[2000,4000],
              "selection_rule":"Choose largest positive L1-L7 short exposed stall wholly contained in [2000,4000); break ties by lowest NPU, request ID, layer. This is an extreme local example, not an average.",
              "long_reference_rule":"Choose a long internal read fully hidden by its previous compute with greatest temporal overlap with selected short stall; tie break by lowest NPU, request ID, layer. This does not identify a physical FIFO blocker.",
              "plot_window_rule":"Cover both selected reads and complete preceding long computation, expand each side by at least 0.5 ms, round outward to integer ms.",
              "selection_counts":selection_counts,"short":short,"long_reference":long,
              "all_plot_compute_and_stall_intervals":{"short":short_events,"long":long_events},
              "semantic_limits":["Read spans are layer issue-to-data-ready lifetimes across the I/O path, including queueing and transfer; not SSD busy/service intervals.",
                  "No per-block physical trace exists for this case; the selected long request is a simultaneous reference, not an identified head-of-line blocker.",
                  "Stall starts only after admission or the previous layer compute ends; no pre-admission queueing is plotted as stall.",
                  "Matplotlib uses original absolute event times and clips intervals to the common window; bars are never packed or shifted.",
                  "Human-readable layers are one-based; parenthesized L indices use the original zero-based log numbering."],
              "font":{"path":str(FONT),"sha256":sha(FONT),"minimum_chinese_font_size_pt":42},
              "colors":{"short":SHORT,"long":LONG,"stall":STALL,"read_lifetime":READ},"artifacts":files}
    (OUT/"zoom_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"window_ms":[lo,hi],"short_stall_ms":short["exposed_stall_ms"],"short_stall_over_C":short["stall_over_one_layer_compute"],
                      "long_read_ms":long["read_lifetime_ms"],"long_compute_ms":long["previous_layer_compute_ms"],"artifacts":files},ensure_ascii=False))


if __name__=="__main__":
    main()
