"""Read-only independent raw-trace audit and PNGs; never invokes a simulator.

Only a completed result with a completed command is treated as evidence.
Running/cancelled cases are listed separately, without invented 2-second bins.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import gc

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
RUNS = STUDY / "local_overload/runs"
FIGURES = STUDY / "figures"
PRIMARY = "a128_group4_8_prefix2_60s_od_baseline_full"
LONG_CASES = {
    PRIMARY: ("128K：4A+8B，前缀2B", "#d45e00"),
    "a128_group2_4_prefix0_60s_od_baseline_full": ("128K：2A+4B", "#2474b7"),
    "a200_b200_r12_60s_od_baseline_full": ("200K：ABB", "#65727c"),
}
CONTROLS = {
    "short_ABB": (RUNS / "a128_b128_r12_od_baseline_full", "128K 原ABB：局部超载", "#777777"),
    "short_shuffled": (STUDY / "order_control/runs/shuffled_seed7_od_baseline_full", "同批128K：每卡随机顺序", "#2474b7"),
    "strict_replay": (STUDY / "strict_replay", "旧合成画像：严格名义欠载", "#228866"),
}
FIXED = [(0.,20000.),(2000.,4000.),(4000.,8000.),(8000.,12000.),
         (12000.,20000.),(20000.,40000.),(40000.,60000.),(20000.,60000.)]


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8") as stream:
        return json.load(stream)


def write(path,payload):
    with path.open("w",encoding="utf-8") as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a,z,left,right):
    return max(0.,min(z,right)-max(a,left))


def demand_trace(requests,profiles,left,right):
    changes=defaultdict(list)
    changes[left];changes[right]
    for row in requests:
        a=max(left,row["admission_time_ms"])
        z=min(right,row["completion_time_ms"])
        if z<=a:
            continue
        changes[a].append((row["request_id"],1))
        changes[z].append((row["request_id"],-1))
    active=set();segments=[]
    times=sorted(changes)
    for a,z in zip(times,times[1:]):
        for rid,sign in changes[a]:
            if sign<0:active.remove(rid)
        for rid,sign in changes[a]:
            if sign>0:active.add(rid)
        assert len({profiles[rid]["npu"] for rid in active})==len(active)
        ordered=sorted(active)
        rates=[math.fsum(profiles[rid]["rates"][d] for rid in ordered) for d in range(3)]
        roles=Counter(profiles[rid]["role"] for rid in ordered)
        segments.append([a,z,*rates,len(active),roles.get("A",0),roles.get("B",0)])
    return segments


def window_stats(summary,profiles,segments,left,right):
    dt=right-left
    role_compute=[defaultdict(float) for _ in range(32)]
    role_active=[defaultdict(float) for _ in range(32)]
    wait=[defaultdict(float) for _ in range(32)]
    for row in summary["request_metrics"]:
        p=profiles[row["request_id"]]
        role_active[p["npu"]][p["role"]]+=overlap(row["admission_time_ms"],row["completion_time_ms"],left,right)
    for batch in summary["microbatch_metrics"]:
        assert len(batch["member_request_ids"])==1
        p=profiles[batch["member_request_ids"][0]]
        for layer in batch["layer_metrics"]:
            a,z=layer["compute_start_ms"],layer["compute_end_ms"]
            assert math.isfinite(a) and math.isfinite(z)
            role_compute[p["npu"]][p["role"]]+=overlap(a,z,left,right)
            kind="internal_layer" if layer["layer"] else "layer_zero"
            wait[p["npu"]][kind]+=overlap(a-layer["io_barrier_wait_ms"],a,left,right)
    computes=[math.fsum(x.values()) for x in role_compute]
    active=[math.fsum(x.values()) for x in role_active]
    stall=[math.fsum(x.values()) for x in wait]
    errors=[active[n]-computes[n]-stall[n] for n in range(32)]
    assert max(map(abs,errors))<1e-6
    integral=[0.]*3;minimum=[math.inf]*3;peak=[0.]*3;over=[0.]*3;ge=[0.]*3
    all_over=any_over=coverage=0.
    for row in segments:
        elapsed=overlap(row[0],row[1],left,right)
        if not elapsed:continue
        coverage+=elapsed
        flags=[v>40 for v in row[2:5]]
        all_over+=elapsed*all(flags);any_over+=elapsed*any(flags)
        for d,v in enumerate(row[2:5]):
            integral[d]+=elapsed*v
            minimum[d]=min(minimum[d],v);peak[d]=max(peak[d],v)
            over[d]+=elapsed*(v>40);ge[d]+=elapsed*(v>=40)
    assert abs(coverage-dt)<1e-6
    cohort=[r for r in summary["request_metrics"] if left<=r["admission_time_ms"]<right]
    passes=sum(r["completion_time_ms"]-r["admission_time_ms"]<=
               1.5*8*profiles[r["request_id"]]["C_ms"]+1e-9 for r in cohort)
    mixed=[role_compute[n].get("A",0)>0 and role_compute[n].get("B",0)>0 for n in range(32)]
    return dict(start_ms=left,end_ms=right,U_percent=100*math.fsum(computes)/(32*dt),
                per_npu_U_percent=[100*x/dt for x in computes],
                per_npu_active_ms=active,all_npus_active=all(abs(x-dt)<1e-6 for x in active),
                per_npu_role_compute_ms=[dict(x) for x in role_compute],
                per_npu_role_active_ms=[dict(x) for x in role_active],
                mixed_npus=sum(mixed),per_npu_mixed=mixed,
                io_stall_card_ms=math.fsum(stall),
                internal_layer_stall_card_ms=math.fsum(x.get("internal_layer",0) for x in wait),
                layer_zero_stall_card_ms=math.fsum(x.get("layer_zero",0) for x in wait),
                compute_stall_active_max_error_ms=max(map(abs,errors)),
                per_disk_mean_GiB_s=[x/dt for x in integral],per_disk_min_GiB_s=minimum,
                per_disk_peak_GiB_s=peak,per_disk_overload_percent=[100*x/dt for x in over],
                strict_nominal_underload=all(x==0 for x in ge),
                all_disks_overload_percent=100*all_over/dt,any_disk_overload_percent=100*any_over/dt,
                slo_1p5_passed=passes,slo_1p5_count=len(cohort),
                slo_1p5_percent=100*passes/len(cohort) if cohort else None)


def load_compact_manifest(path,verify_data=False):
    raw=read(path)
    table=ast.literal_eval((STUDY/"local_overload/runtime/data").read_text()) if verify_data else None
    profiles={};role_counts=[Counter() for _ in range(32)]
    role_sequences=[[] for _ in range(32)]
    blocks=0;physical_ids=[];data_checks=0
    for r in sorted(raw["requests"],key=lambda r:(r["npu_id"],r["request_id"])):
        load=r["load"];placement=raw["placements"][r["placement_index"]]
        assert len(placement)==1
        assert r["arrival_time_ms"]==0
        values=[math.fsum(v for d,v in placement[0] if d==disk) for disk in range(3)]
        C_ms=load["per_layer_us"]/1000
        if verify_data:
            data=table[(load["seq_len_k"],load["nql"])]
            assert load["per_layer_us"]==data[1]
            assert math.isclose(math.fsum(values),data[3],rel_tol=0,abs_tol=1e-13)
            assert not load.get("constructed_profile",False)
            data_checks+=1
        profiles[r["request_id"]]=dict(npu=r["npu_id"],role=load["role"],C_ms=C_ms,
            rates=[v*1000/C_ms for v in values],volume_GiB_by_ssu=values,
            original_request_id=load.get("original_request_id",r["request_id"]))
        physical_ids.append(load.get("original_request_id",r["request_id"]))
        role_counts[r["npu_id"]][load["role"]]+=1
        role_sequences[r["npu_id"]].append(load["role"])
        blocks+=len(placement[0])*8
    assert len(profiles)==len(raw["requests"])
    assert len(set(physical_ids))==len(physical_ids)
    details=dict(metadata=raw["metadata"],request_count=len(profiles),expected_blocks=blocks,
        role_counts_per_npu=[dict(x) for x in role_counts],
        role_sequences_per_npu=role_sequences,
        pure_compute_ms_per_npu=[8*math.fsum(p["C_ms"] for p in profiles.values() if p["npu"]==n) for n in range(32)],
        direct_data_profiles_verified=data_checks,
        manifest_sha256=sha(path),input_fingerprint=raw["input_fingerprint"])
    return profiles,details


def complete_fifo_cycles(summary,profiles):
    """Auxiliary per-card complete 4A+8B cycles; not synchronous fleet U."""
    rows=[];by_cycle=[]
    requests={r["request_id"]:r for r in summary["request_metrics"]}
    first_npu_finish=min(max(r["completion_time_ms"] for r in requests.values() if r["npu_id"]==n) for n in range(32))
    for npu in range(32):
        ids=sorted(rid for rid,p in profiles.items() if p["npu"]==npu)
        assert [profiles[rid]["role"] for rid in ids[:2]]==["B","B"]
        for cycle,start in enumerate(range(2,len(ids)-11,12)):
            selected=ids[start:start+12]
            roles=[profiles[rid]["role"] for rid in selected]
            assert roles==["A"]*4+["B"]*8
            left=requests[selected[0]]["admission_time_ms"]
            right=requests[selected[-1]]["completion_time_ms"]
            pure=8*math.fsum(profiles[rid]["C_ms"] for rid in selected)
            rows.append(dict(npu_id=npu,cycle_index=cycle,start_ms=left,end_ms=right,
                duration_ms=right-left,pure_compute_ms=pure,U_percent=100*pure/(right-left),
                ends_before_first_npu_finishes=right<=first_npu_finish))
    for cycle in sorted({r["cycle_index"] for r in rows}):
        group=[r for r in rows if r["cycle_index"]==cycle]
        assert len(group)==32
        by_cycle.append(dict(cycle_index=cycle,
            earliest_start_ms=min(r["start_ms"] for r in group),latest_start_ms=max(r["start_ms"] for r in group),
            earliest_end_ms=min(r["end_ms"] for r in group),latest_end_ms=max(r["end_ms"] for r in group),
            all_cycles_before_any_npu_finishes=all(r["ends_before_first_npu_finishes"] for r in group),
            mean_duration_ms=math.fsum(r["duration_ms"] for r in group)/32,
            asynchronous_duration_weighted_U_percent=100*math.fsum(r["pure_compute_ms"] for r in group)/math.fsum(r["duration_ms"] for r in group)))
    return dict(definition="Each card's full 4A+8B FIFO segment after 2B prefix; asynchronous start/end; weighted cycle occupancy is NOT common-window fleet U; flag excludes fleet drain tail",
                first_npu_final_completion_ms=first_npu_finish,
                per_npu_cycles=rows,by_cycle=by_cycle)


def audit_case(path,label,*,long=False):
    command=read(path/"command.json")
    result_path=path/"result.json.gz"
    if command.get("status")!="complete" or not result_path.exists():
        preview_file=path/"window_previews.json"
        previews=[]
        if preview_file.exists():
            for row in read(preview_file).values():
                previews.append({k:row.get(k) for k in ("start_ms","end_ms","U_percent","all_npus_active","mixed_cards","observed_at_ms")})
        return dict(label=label,path=str(path),complete=False,status=command.get("status"),
                    scientific_status="Not fully drained; previews are runner observations, not independent final audit",
                    previews=sorted(previews,key=lambda r:r["start_ms"])),None
    raw=read(result_path);summary=raw["summary"]
    assert all(summary["invariants"].values())
    assert command.get("source_unchanged",command.get("all_protected_files_unchanged",False))
    profiles,input_info=load_compact_manifest(path/"manifest.json.gz",verify_data=long)
    assert summary["completed_blocks"]==input_info["expected_blocks"]
    assert len(summary["request_metrics"])==len(profiles)
    assert raw["input_fingerprint"]==input_info["input_fingerprint"]
    qd=summary["ssd_queue_depth"]
    assert max(map(max,qd["peak_outstanding_blocks_by_npu_ssu"]))<=256
    assert qd["host_deferred_blocks_at_stop"]==qd["ssd_outstanding_blocks_at_stop"]==qd["link_outstanding_blocks_at_stop"]==0
    finishes=[max(r["completion_time_ms"] for r in summary["request_metrics"] if r["npu_id"]==n) for n in range(32)]
    earliest_finish=min(finishes)
    horizon=60000. if long else min(16000.,2000*math.floor(earliest_finish/2000))
    assert summary["makespan_ms"]>=horizon
    segments=demand_trace(summary["request_metrics"],profiles,0.,horizon)
    bins=[window_stats(summary,profiles,segments,float(a),float(a+2000)) for a in range(0,int(horizon),2000)]
    fixed=[window_stats(summary,profiles,segments,a,z) for a,z in FIXED if z<=horizon]
    assert all(w["all_npus_active"] for w in bins)
    check_errors=[];physical_means=[]
    for reported in raw.get("analysis",[]):
        if "start_ms" not in reported:continue
        a,z=reported["start_ms"],reported["end_ms"]
        if z>horizon:continue
        recalculated=next((w for w in fixed if w["start_ms"]==a and w["end_ms"]==z),None)
        if recalculated is None:
            recalculated=window_stats(summary,profiles,segments,a,z)
        error=abs(recalculated["U_percent"]-reported["U_percent"])
        assert error<1e-8
        if "demand" in reported:
            for ours,theirs in zip(recalculated["per_disk_mean_GiB_s"],reported["demand"]["per_disk_mean_GiB_s"]):
                assert abs(ours-theirs)<1e-7
        check_errors.append(error)
        supply=reported.get("SSD_GiB_s")
        if supply is not None:
            assert len(supply)==3 and all(-1e-8<=v<=40+1e-8 for v in supply)
            physical_means.append(dict(start_ms=a,end_ms=z,SSD_GiB_s=supply,
                definition="physical SSD service average over entire labeled interval; not instantaneous"))
    if long:
        assert earliest_finish>60000
        assert all(w["mixed_npus"]==32 for w in fixed if w["start_ms"] in (20000.,40000.))
        for name,digest in command["source_sha256"].items():
            assert sha(STUDY/"local_overload"/name)==digest,(label,name)
    output=dict(label=label,path=str(path),complete=True,status="complete",passed=True,
        result_sha256=sha(result_path),command_sha256=sha(path/"command.json"),input=input_info,
        completed_blocks=summary["completed_blocks"],makespan_ms=summary["makespan_ms"],
        earliest_npu_final_completion_ms=earliest_finish,last_completion_ms_by_npu=finishes,
        analysis_horizon_ms=horizon,all_2s_bins_npus_active=all(w["all_npus_active"] for w in bins),
        two_second_bins=bins,fixed_windows=fixed,
        per_disk_demand_segments=segments,segment_columns=["start_ms","end_ms","D0","D1","D2","active_cards","A_cards","B_cards"],
        physical_ssd_window_means=physical_means,
        reported_U_max_error=max(check_errors,default=0.),
        queue_depth_256_verified=True,raw_compute_stall_accounting_verified=True)
    if label==PRIMARY:
        output["complete_fifo_cycles"]=complete_fifo_cycles(summary,profiles)
    return output,summary


def font_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names={f.name for f in font_manager.fontManager.ttflist}
    font=next(f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","DejaVu Sans") if f in names)
    plt.rcParams.update({"font.family":font,"axes.unicode_minus":False,"font.size":11})
    return plt


def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#dfe4e8",linewidth=.65)
    ax.set_axisbelow(True)


def render(cases,controls):
    plt=font_setup();outputs=[]
    main=cases[PRIMARY]
    if not main["complete"]:
        return outputs
    fig,ax=plt.subplots(figsize=(15.5,6.8))
    fig.subplots_adjust(left=.07,right=.98,bottom=.21,top=.80)
    for name,(label,color) in LONG_CASES.items():
        result=cases[name]
        if not result["complete"]:continue
        rows=result["two_second_bins"]
        ax.plot([(r["start_ms"]+r["end_ms"])/2000 for r in rows],
                [r["U_percent"] for r in rows],label=label,color=color,
                marker="o" if name==PRIMARY else None,markersize=3.5,
                linewidth=2.4 if name==PRIMARY else 1.6,alpha=1 if name==PRIMARY else .75)
    for i,(left,right) in enumerate(((20000.,40000.),(40000.,60000.))):
        row=next(r for r in main["fixed_windows"] if r["start_ms"]==left and r["end_ms"]==right)
        ax.axvspan(left/1000,right/1000,color="#d45e00",alpha=.035 if i==0 else .07)
        ax.hlines(row["U_percent"],left/1000,right/1000,color="#982d00",linestyle="--",linewidth=2.2)
        ax.text((left+right)/2000,row["U_percent"]-3.1,
                f"{left/1000:g}–{right/1000:g} 秒均值 {row['U_percent']:.2f}%",ha="center",color="#982d00",fontsize=12,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=2))
    ax.axhline(90,color="#bbbbbb",linestyle=":",linewidth=1)
    ax.set(xlim=(0,60),ylim=(0,103),xlabel="仿真时间（秒）",ylabel="NPU 平均利用率（%）")
    ax.set_xticks(range(0,61,5));style(ax)
    fig.suptitle("OD 长时间平均利用率：保留每一个连续 2 秒窗口",fontsize=19,fontweight="bold",y=.965)
    fig.text(.07,.895,"32 NPU / 3 SSU × 40 GiB/s · seed 7 · 每盘固定深度 8192 · 橙色为 4A+8B 主候选",fontsize=11)
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.53,.866),ncol=3,frameon=False)
    fig.text(.07,.105,"实线点 = 对应 2 秒内的实际计算时间占比；橙色虚线 = 预先固定的 20 秒区间均值。允许周期内高低交替。",fontsize=10.5)
    fig.text(.07,.067,"所有显示窗口均为 32 卡持续活跃；主候选两个后期 20 秒窗口内，每卡都计算过 A 与 B。未包含末端排空。",fontsize=10.5)
    fig.text(.07,.029,"本图候选有局部逐盘过载；无等待人口的平均需求低于容量，不等于执行中逐时欠载。仅有 seed 7，不能概括所有随机输入。",fontsize=10.5,color="#555555")
    path=FIGURES/"long_term_npu_utilization.png";fig.savefig(path,dpi=170);plt.close(fig);outputs.append(path)

    fig,axes=plt.subplots(3,1,figsize=(16,9),sharex=True)
    fig.subplots_adjust(left=.07,right=.98,bottom=.17,top=.82,hspace=.30)
    segments=main["per_disk_demand_segments"]
    times=[r[0]/1000 for r in segments]+[segments[-1][1]/1000]
    ymax=max(v for r in segments for v in r[2:5])*1.12
    for disk,ax in enumerate(axes):
        demand=[r[disk+2] for r in segments]+[segments[-1][disk+2]]
        ax.step(times,demand,where="post",color="#d45e00",linewidth=1.15,label="名义需求：当前各卡 V/C 之和")
        ax.axhline(40,color="#555555",linestyle="--",linewidth=1.2,label="物理容量 40 GiB/s")
        for j,row in enumerate(main["physical_ssd_window_means"]):
            ax.hlines(row["SSD_GiB_s"][disk],row["start_ms"]/1000,row["end_ms"]/1000,
                      color="#176db0",linewidth=2.1,label="物理服务：标示窗口的平均值" if j==0 else None)
            if row["start_ms"]>=20000:
                ax.text((row["start_ms"]+row["end_ms"])/2000,row["SSD_GiB_s"][disk]+22,
                        f"{row['SSD_GiB_s'][disk]:.2f}",ha="center",fontsize=9,color="#176db0",
                        bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=1))
        ax.set(ylim=(0,ymax),xlim=(0,60),ylabel=f"SSU {disk}\nGiB/s")
        style(ax)
    axes[-1].set_xlabel("仿真时间（秒）");axes[-1].set_xticks(range(0,61,5))
    fig.suptitle("4A+8B：读取需求成批出现，逐盘存在过载时段",fontsize=19,fontweight="bold",y=.965)
    fig.text(.07,.905,"A：128K / miss 256，单层计算 6.024 ms；B：128K / miss 4096，单层计算 93.017 ms。两者均直接取自 data。",fontsize=11)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="upper center",bbox_to_anchor=(.53,.875),ncol=3,frameon=False)
    fig.text(.07,.099,"橙线按每次请求接纳/完成事件精确重算；等待 I/O 的卡也计入。下一请求首层预取不额外叠加第二份画像需求。",fontsize=10.5)
    fig.text(.07,.058,"蓝线是物理 SSD 服务的整段均值（2–4、4–8、8–12、12–20、20–40、40–60 秒），不是瞬时速率或层周期叠加。",fontsize=10.5)
    fig.text(.07,.020,"不能用低的全程平均需求或蓝线，证明橙线逐时低于容量；本候选属于局部过载，不属于严格逐盘欠载。",fontsize=10.5,color="#555555")
    path=FIGURES/"long_term_per_ssu_demand.png";fig.savefig(path,dpi=170);plt.close(fig);outputs.append(path)

    ready=[(name,r) for name,r in controls.items() if r["complete"]]
    if ready:
        fig,ax=plt.subplots(figsize=(13.5,6.3));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.80)
        for name,row in ready:
            _,label,color=CONTROLS[name]
            bins=row["two_second_bins"]
            ax.plot([(r["start_ms"]+r["end_ms"])/2000 for r in bins],[r["U_percent"] for r in bins],
                    label=label,color=color,marker="o",markersize=4,linewidth=2)
        ax.set(xlim=(0,16),ylim=(60,101),xlabel="仿真时间（秒）",ylabel="NPU 平均利用率（%）")
        ax.set_xticks(range(0,17,2));style(ax)
        fig.suptitle("恢复对照：短期低值不能替代长期平均",fontsize=18,fontweight="bold",y=.96)
        fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.53,.885),ncol=3,frameon=False,fontsize=10)
        fig.text(.08,.115,"每点为连续 2 秒窗口；显示范围内全卡活跃，没有加入末端排空。灰/蓝使用同一批原始 128K 请求，仅卡内顺序不同。",fontsize=10)
        fig.text(.08,.070,"绿线是另一批合成画像及对抗性选址的严格名义欠载输入，用于展示恢复，不是同输入策略比较。",fontsize=10)
        fig.text(.08,.025,"随机顺序并非每个 2 秒窗口都每卡混合；各窗实际覆盖卡数见独立审计表。",fontsize=10,color="#555555")
        path=FIGURES/"recovery_controls.png";fig.savefig(path,dpi=170);plt.close(fig);outputs.append(path)
    return outputs


def output_tables(cases,controls):
    rows=[]
    for label,result in {**cases,**controls}.items():
        if not result["complete"]:continue
        for kind,key in (("continuous_2s","two_second_bins"),("predefined_interval","fixed_windows")):
            for r in result[key]:
                rows.append(dict(case=label,window_kind=kind,start_s=r["start_ms"]/1000,end_s=r["end_ms"]/1000,
                    U_percent=r["U_percent"],all_npus_active=r["all_npus_active"],mixed_npus=r["mixed_npus"],
                    strict_nominal_underload=r["strict_nominal_underload"],
                    **{f"disk{d}_{metric}":r[key][d] for metric,key in
                       (("mean_GiB_s","per_disk_mean_GiB_s"),("peak_GiB_s","per_disk_peak_GiB_s"),("overload_percent","per_disk_overload_percent")) for d in range(3)},
                    internal_layer_stall_card_ms=r["internal_layer_stall_card_ms"],
                    layer_zero_stall_card_ms=r["layer_zero_stall_card_ms"],
                    SLO_1p5_percent=r["slo_1p5_percent"],SLO_count=r["slo_1p5_count"]))
    if rows:
        with (HERE/"long_term_windows.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if cases[PRIMARY]["complete"]:
        cycle_rows=cases[PRIMARY]["complete_fifo_cycles"]["per_npu_cycles"]
        with (HERE/"complete_fifo_cycles.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(cycle_rows[0]));writer.writeheader();writer.writerows(cycle_rows)
    lines=["# OD 长时间平均：独立原始轨迹审计","",
           "不调用仿真器或 runner 的指标函数。利用率由原始层计算起止区间积分；全部连续 2 秒窗均保留。预先固定 [20,40)、[40,60) 及 [20,60) 秒用于长期均值，不挑低谷。", "",
           "|案例|状态|", "|---|---|"]
    for name,r in cases.items():lines.append(f"|{name}|{'完整排空，已独立复算' if r['complete'] else r['status']+'；非最终结果'}|")
    lines += ["", "|案例|区间 秒|U|活跃卡|实际计算过A/B的卡|最早单卡结束 秒|逐盘超40比例|", "|---|---|---:|---:|---:|---:|---|"]
    for name,result in cases.items():
        if not result["complete"]:continue
        for r in result["fixed_windows"]:
            if r["start_ms"] not in (20000.,40000.):continue
            lines.append(f"|{LONG_CASES[name][0]}|[{r['start_ms']/1000:g},{r['end_ms']/1000:g})|{r['U_percent']:.4f}%|{32 if r['all_npus_active'] else '<32'}|{r['mixed_npus']}|{result['earliest_npu_final_completion_ms']/1000:.3f}|"+" / ".join(f"{v:.2f}%" for v in r["per_disk_overload_percent"])+"|")
    primary=cases[PRIMARY]
    if primary["complete"]:
        early=next(r for r in primary["fixed_windows"] if (r["start_ms"],r["end_ms"])==(20000.,40000.))
        late=next(r for r in primary["fixed_windows"] if (r["start_ms"],r["end_ms"])==(40000.,60000.))
        all_late=next(r for r in primary["fixed_windows"] if (r["start_ms"],r["end_ms"])==(20000.,60000.))
        cycles=[r for r in primary["complete_fifo_cycles"]["by_cycle"] if r["all_cycles_before_any_npu_finishes"]]
        lines += ["",f"主候选未形成持续的八十几平台：[20,40) 为 {early['U_percent']:.4f}%，[40,60) 已升至 {late['U_percent']:.4f}%；两个区间合计 [20,60) 为 {all_late['U_percent']:.4f}%。不是通过只取周期低谷得出判断。",
                  "",f"完整周期补证也显示恢复：不涉及任何卡排空的第 {cycles[0]['cycle_index']+1} 至第 {cycles[-1]['cycle_index']+1} 个周期，异步逐卡按时长加权占比从 {cycles[0]['asynchronous_duration_weighted_U_percent']:.4f}% 升至 {cycles[-1]['asynchronous_duration_weighted_U_percent']:.4f}%。这排除了仅仅20秒窗口落在不同相位的替代解释，但该周期占比仍不是同步墙钟整机U。"]
    lines += ["", "主候选是每卡静态队列：先 2B，再以 4A+8B 成批排列同一批请求，末尾不足完整周期部分照常保留。每卡 42A+84B，纯计算 64.532 秒；A/B 均为 data 原始 128K 画像，miss 分别 256/4096，没有缩放计算或读取量。32 卡按相同角色顺序运行，物理地址逐请求不同，Ring hash 不变。", "",
              "每盘需求 D_s 是当前已接纳请求实际 V_i,s/C_i 之和，等待期间仍计入；下一请求 L0 不额外叠加第二份画像需求。严格欠载要求每个事件区间每盘均小于 40 GiB/s。当前成批输入属于局部过载，不能因为平均值低就改称严格欠载。", "",
              "物理服务只读取已存的窗口服务积分均值，不伪造长轨迹瞬时供给。SLO 是窗口内接纳请求完整排空后的耗时/自身八层纯计算时间≤1.5（容差1e-9ms），不包括接纳前排队。", "",
              "每个 2 秒窗未必包含完整 4A+8B 周期，也未必每卡都计算到两种画像；这些覆盖数逐窗公开。后期两个固定 20 秒窗必须验证每卡均运行两类且始终活跃；否则不能作为本任务的长期证据。有限运行的长期平均仍不能证明无限期稳定，也不能以完整生命周期均值的尾部代替它。", "",
              "补充 complete_fifo_cycles.csv 按每卡前缀2B后的完整12请求段统计，避免只凭固定时间窗的周期相位推测恢复；对应周期的卡间起止时刻不同，这种逐卡周期加权占比不是共同时间窗口的整机U，仅作机制证据。末尾周期若延伸至第一张卡结束之后，明确标记为含排空影响，不拿它证明长期恢复。每周期画像重复，但 Ring hash 的实际逐盘块数不保证完全相同。", "",
              "文件：long_term_audit.json 包含所有窗口逐卡积分、逐盘逐事件曲线与输入/结果 SHA；long_term_windows.csv 为便于阅读的平表。PNG 在 ../figures/。运行脚本仅输出已完成 raw 的正式曲线；未完成项只记录明确标注的状态/runner预览。"]
    (HERE/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--require-primary-complete",action="store_true")
    args=parser.parse_args();HERE.mkdir(exist_ok=True);FIGURES.mkdir(exist_ok=True)
    cases={};controls={}
    for name in LONG_CASES:
        cases[name],_=audit_case(RUNS/name,name,long=True)
        gc.collect()
    for name,(path,label,color) in CONTROLS.items():
        controls[name],_=audit_case(path,name)
        gc.collect()
    output_tables(cases,controls)
    payload=dict(primary=PRIMARY,primary_complete=cases[PRIMARY]["complete"],
        all_long_cases_complete=all(r["complete"] for r in cases.values()),
        cases=cases,controls=controls,method="Independent raw JSON overlap integrals; no runner/metrics imports",
        preselected_late_windows_ms=[[20000,40000],[40000,60000],[20000,60000]],
        script_sha256=sha(Path(__file__)))
    write(HERE/"long_term_audit.json",payload)
    if args.require_primary_complete and not cases[PRIMARY]["complete"]:
        raise SystemExit("Primary run has not fully drained; final plots not generated")
    outputs=render(cases,controls)
    checks=dict(primary_complete=cases[PRIMARY]["complete"],
        source_audit_sha256=sha(HERE/"long_term_audit.json"),
        generated_png_sha256={str(p.relative_to(STUDY)):sha(p) for p in outputs},
        visual_review="pending" if outputs else "not applicable: primary incomplete",
        figures_do_not_use_preview_data=True,
        physical_supply_note="Only measured original interval means; no instantaneous reconstruction")
    write(HERE/"long_term_render_checks.json",checks)
    print(json.dumps(dict(primary_complete=cases[PRIMARY]["complete"],outputs=[str(p) for p in outputs],
        statuses={k:v["status"] for k,v in cases.items()})))


if __name__=="__main__":main()
