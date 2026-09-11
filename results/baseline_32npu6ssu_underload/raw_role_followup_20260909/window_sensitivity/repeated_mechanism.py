#!/usr/bin/env python3
"""Read completed tripled ordered Baseline logs; no simulation or block trace."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
OUT = HERE / "repeated_queues"
LABEL = "raw176_ordered_repeat3_seed7"
WINDOWS = ((2000,4000),(4000,12000),(2000,12000))


def read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix==".gz" else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip(a,b,start,end):
    return max(0.0,min(b,end)-max(a,start))


def distribution(values):
    a=sorted(values)
    if not a:return dict(count=0)
    def q(p):
        z=(len(a)-1)*p/100;lo=math.floor(z);hi=math.ceil(z)
        return a[lo]*(hi-z)+a[hi]*(z-lo) if hi!=lo else a[lo]
    return dict(count=len(a),mean=statistics.mean(a),p0=a[0],p50=q(50),p95=q(95),p99=q(99),p100=a[-1])


def phase(values,period):
    a=sorted(v%period for v in values)
    gaps=[y-x for x,y in zip(a,a[1:])]+[a[0]+period-a[-1]]
    return dict(period_ms=period,linear_spread_ms=max(values)-min(values),
                circular_minimum_covering_arc_ms=period-max(gaps),circular_maximum_gap_ms=max(gaps),values_ms=values,
                phase_modulo_period_ms=[v%period for v in values])


def window(summary,requests,start,end):
    roles={role:dict(active_ms=0.,compute_ms=0.,stall_ms=0.) for role in ("long","short")}
    groups={g:{role:dict(active_ms=0.,compute_ms=0.,stall_ms=0.) for role in roles}
            for g in ("front20","rear12")}
    profiles={p:dict(active_ms=0.,compute_ms=0.,stall_ms=0.,latencies=[],budgets=[],ratios=[],exposed=[]) for p in range(4)}
    npu_active=[0.]*32
    long_changes={start:0,end:0}
    for batch in summary["microbatch_metrics"]:
        rid=batch["member_request_ids"][0];r=requests[rid];npu=r["npu_id"]
        role=r["load"]["role"];p=r["load"]["profile_index"];g="front20" if npu<20 else "rear12"
        active=clip(batch["admission_time_ms"],batch["completion_time_ms"],start,end)
        if role=="long" and active>0:
            lo=max(start,batch["admission_time_ms"]);hi=min(end,batch["completion_time_ms"])
            long_changes[lo]=long_changes.get(lo,0)+1
            long_changes[hi]=long_changes.get(hi,0)-1
        compute=math.fsum(clip(x["compute_start_ms"],x["compute_end_ms"],start,end) for x in batch["layer_metrics"])
        stall=active-compute
        assert stall>=-1e-7
        for target in (roles[role],groups[g][role],profiles[p]):
            target["active_ms"]+=active;target["compute_ms"]+=compute;target["stall_ms"]+=stall
        npu_active[npu]+=active
        for j,x in enumerate(batch["layer_metrics"]):
            if j and start<=x["io_start_time_ms"]<end:
                latency=x["io_ready_time_ms"]-x["io_start_time_ms"]
                budget=batch["layer_metrics"][j-1]["compute_duration_ms"]
                profiles[p]["latencies"].append(latency);profiles[p]["budgets"].append(budget)
                profiles[p]["ratios"].append(latency/budget);profiles[p]["exposed"].append(x["io_barrier_wait_ms"])
    for p,x in profiles.items():
        x["internal_read_ms"]=distribution(x.pop("latencies"))
        x["prefetch_compute_budget_ms"]=distribution(x.pop("budgets"))
        ratios=x.pop("ratios");x["internal_read_to_budget_ratio"]=distribution(ratios)
        x["read_within_budget_fraction"]=sum(v<=1+1e-9 for v in ratios)/len(ratios) if ratios else None
        x["released_layer_exposed_stall_ms"]=distribution(x.pop("exposed"))
        x["compute_fraction_of_device_compute"]=x["compute_ms"]/math.fsum(y["compute_ms"] for y in profiles.values())
        x["time_average_active_npu_count"]=x["active_ms"]/(end-start)
    for target in (roles,*groups.values()):
        for x in target.values():
            x["time_average_active_npu_count"]=x["active_ms"]/(end-start)
            x["conditional_compute_fraction"]=x["compute_ms"]/x["active_ms"] if x["active_ms"] else None
    c=math.fsum(x["compute_ms"] for x in roles.values());a=math.fsum(npu_active)
    assert all(math.isclose(x,end-start,abs_tol=1e-6) for x in npu_active)
    assert math.isclose(a,32*(end-start),abs_tol=1e-6)
    times=sorted(long_changes);current=0;duration_by_count={}
    for lo,hi in zip(times,times[1:]):
        current+=long_changes[lo]
        if hi>lo:duration_by_count[current]=duration_by_count.get(current,0.)+hi-lo
    assert math.isclose(math.fsum(k*v for k,v in duration_by_count.items()),roles['long']['active_ms'],abs_tol=1e-6)
    return dict(start_ms=start,end_ms=end,device_U_percent=100*c/(32*(end-start)),
                exposed_stall_ms=a-c,exposed_stall_percent=100*(a-c)/(32*(end-start)),
                global_long_concurrency=dict(min=min(duration_by_count),max=max(duration_by_count),
                    mean=roles['long']['active_ms']/(end-start),duration_ms_by_count=duration_by_count,
                    definition="current admitted Long requests; simultaneous admission/completion timestamp deltas grouped before positive-duration intervals"),
                all_32_active=True,roles=roles,groups=groups,profiles=profiles)


def fixed_comparison(period):
    label="raw176_fixed_repeat3_seed7"
    manifest_path=OUT/"inputs"/(label+".json.gz")
    directory=OUT/"runs"/label/"baseline"
    command=read(directory/"command.json")
    assert command["status"]=="complete" and command["returncode"]==0
    path=Path(command["output"]);raw=read(path);manifest=read(manifest_path)
    assert raw["strategy"]=="baseline" and raw["submit_seed"]==7
    assert raw["input_fingerprint"]==manifest["input_fingerprint"]
    requests={r["request_id"]:r for r in manifest["requests"]}
    batches={b["member_request_ids"][0]:b for b in raw["summary"]["microbatch_metrics"]}
    sequences=[]
    for npu in range(20):
        lane=sorted((r for r in requests.values() if r["npu_id"]==npu),key=lambda r:r["request_id"])
        assert all(r["load"]["role"]=="long" for r in lane)
        sequences.append([layer for r in lane for layer in batches[r["request_id"]]["layer_metrics"]])
    waves=[]
    for ordinal in range(min(map(len,sequences))):
        if ordinal%8==0:continue
        layers=[lane[ordinal] for lane in sequences]
        releases=[x["io_start_time_ms"] for x in layers];compute=[x["compute_start_ms"] for x in layers]
        if min(releases)<2000 or max(releases)>=12000:continue
        waves.append(dict(logical_long_layer=ordinal,io_start_min_ms=min(releases),io_start_max_ms=max(releases),
            io_start_spread_ms=max(releases)-min(releases),compute_start_spread_ms=max(compute)-min(compute),
            io_circular_arc_ms=phase(releases,period)["circular_minimum_covering_arc_ms"],
            io_circular_maximum_gap_ms=phase(releases,period)["circular_maximum_gap_ms"]))
    windows=[window(raw["summary"],requests,a,b) for a,b in WINDOWS]
    longest=windows[-1]["profiles"][3]["internal_read_ms"]["p100"]
    return dict(result_path=str(path),result_sha256=sha(path),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),
        windows=windows,internal_wave_scope_ms=[2000,12000],internal_waves=waves,
        internal_io_start_spread_ms=distribution([w["io_start_spread_ms"] for w in waves]),
        internal_compute_start_spread_ms=distribution([w["compute_start_spread_ms"] for w in waves]),
        internal_io_circular_arc_ms=distribution([w["io_circular_arc_ms"] for w in waves]),
        longest_internal_long_read_ms=longest,long_layer_C_ms=period,
        all_observed_warm_long_internal_reads_within_C=longest<=period+1e-8)


def main():
    manifest_path=OUT/"inputs"/(LABEL+".json.gz")
    directory=OUT/"runs"/LABEL/"baseline"
    command=read(directory/"command.json");assert command["status"]=="complete" and command["returncode"]==0
    result_path=Path(command["output"]);raw=read(result_path);manifest=read(manifest_path)
    assert raw["strategy"]=="baseline" and raw["submit_seed"]==7
    assert raw["input_fingerprint"]==manifest["input_fingerprint"]
    assert all(raw["summary"]["invariants"].values())
    requests={r["request_id"]:r for r in manifest["requests"]}
    batches={b["member_request_ids"][0]:b for b in raw["summary"]["microbatch_metrics"]}
    period=next(r["load"]["per_layer_us"]/1000 for r in requests.values() if r["load"]["role"]=="long")
    cycles=[]
    for cycle in (0,1):
        rows=[];long_sequences=[];long_intervals=[]
        for npu in range(20):
            lane=sorted((r for r in requests.values() if r["npu_id"]==npu and r["load"]["source_cycle"]==cycle and r["load"]["role"]=="long"),key=lambda r:r["request_id"])
            first=lane[0];batch=batches[first["request_id"]];layer=batch["layer_metrics"][0]
            rows.append(dict(npu=npu,request_id=first["request_id"],base_request_id=first["load"]["base_request_id"],
                admission_ms=batch["admission_time_ms"],io_start_ms=layer["io_start_time_ms"],
                compute_start_ms=layer["compute_start_ms"],layer0_prefetched=layer["io_start_time_ms"]<batch["admission_time_ms"]))
            long_sequences.append([layer for r in lane for layer in batches[r["request_id"]]["layer_metrics"]])
            long_intervals.append((batch["admission_time_ms"],batches[lane[-1]["request_id"]]["completion_time_ms"]))
            # All these Long requests are a contiguous block in this ordered lane.
            assert all(math.isclose(batches[x["request_id"]]["completion_time_ms"],batches[y["request_id"]]["admission_time_ms"],abs_tol=1e-8) for x,y in zip(lane,lane[1:]))
        overlap_start=max(x[0] for x in long_intervals);overlap_end=min(x[1] for x in long_intervals)
        assert overlap_start<overlap_end
        waves=[]
        for logical in range(min(map(len,long_sequences))):
            layers=[seq[logical] for seq in long_sequences]
            release=[x["io_start_time_ms"] for x in layers];compute=[x["compute_start_ms"] for x in layers]
            waves.append(dict(logical_long_layer=logical,io_start_min_ms=min(release),io_start_max_ms=max(release),
                io_start_spread_ms=max(release)-min(release),compute_start_spread_ms=max(compute)-min(compute),
                compute_circular_arc_ms=phase(compute,period)["circular_minimum_covering_arc_ms"],
                io_circular_arc_ms=phase(release,period)["circular_minimum_covering_arc_ms"],
                io_circular_maximum_gap_ms=phase(release,period)["circular_maximum_gap_ms"]))
        relevant=[w for w in waves if (2000 if cycle==0 else 4000)<=w["io_start_min_ms"] and w["io_start_max_ms"]<(4000 if cycle==0 else 12000)]
        internal_overlap=[w for w in relevant if w["logical_long_layer"]%8!=0 and
                          overlap_start<=w["io_start_min_ms"] and w["io_start_max_ms"]<overlap_end]
        cycles.append(dict(cycle=cycle,first_long_per_front20=rows,
            first_long_io_start=phase([x["io_start_ms"] for x in rows],period),
            first_long_compute_start=phase([x["compute_start_ms"] for x in rows],period),
            common_logical_long_waves=waves,
            all_front20_continuously_long_interval_ms=[overlap_start,overlap_end],
            actual_roles_during_front20_overlap=window(raw["summary"],requests,overlap_start,overlap_end),
            internal_waves_all_releases_inside_front20_overlap=internal_overlap,
            internal_overlap_io_circular_arc_ms=distribution([w["io_circular_arc_ms"] for w in internal_overlap]),
            internal_overlap_io_circular_maximum_gap_ms=distribution([w["io_circular_maximum_gap_ms"] for w in internal_overlap]),
            relevant_wave_scope_ms=[2000,4000] if cycle==0 else [4000,12000],
            relevant_io_spread_ms=distribution([w["io_start_spread_ms"] for w in relevant]),
            relevant_compute_circular_arc_ms=distribution([w["compute_circular_arc_ms"] for w in relevant])))
    windows=[window(raw["summary"],requests,a,b) for a,b in WINDOWS]
    original_stall_share=windows[0]["exposed_stall_ms"]/windows[2]["exposed_stall_ms"]
    fixed=fixed_comparison(period)
    output=dict(script_sha256=sha(__file__),result_path=str(result_path),result_sha256=sha(result_path),
        manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),seed=7,strategy="baseline",
        definitions=dict(group="front20=NPU0..19; rear12=NPU20..31; request roles change by queue position",
            nominal_role="current admitted request role; next-request L0 prefetch is separate and can precede admission",
            phase="compare front20's same logical Long index inside each whole-lane cycle; circular spread uses physical Long layer C, not request count",
            occupancy="actual [admission,completion) clipped to window, divided by window duration; reports mean concurrent role cards",
            io_cohort="internal layers L1..7 whose io_start is inside the window; follow complete read/stall even if after window",
            stall="actual clipped occupied minus actual clipped compute; never sum IO read lifetimes as exposed device stall",
            limit="A descriptive single-seed joint change in phase, active role count and short-profile composition, not an intervention separating their causal effects. No individual FIFO head blocker is identified."),
        original_2_4_window_fraction_of_total_2_12_stall=original_stall_share,
        fixed_baseline_comparison=fixed,cycles=cycles,windows=windows)
    target=OUT/"repeated_mechanism.json";target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    lines=["# 三倍定序队列的后续变化：seed7 Baseline 层日志核查","",
        "仅读取现有完整结果，不新增仿真。前 20 卡指 NPU0–19，后 12 卡指 NPU20–31；进入后续轮次后两组都会切换长短请求。", "",
        "## 前 20 卡的 Long 波次", "",
        "| 全队列轮次 | 首个 Long L0 释放跨度(ms) | 首个 Long 计算起点跨度(ms) | 计算起点模 Long 层 C 的最小圆弧(ms) |",
        "|---|---:|---:|---:|"]
    for x in cycles:
        lines.append(f"| {x['cycle']+1} | {x['first_long_io_start']['linear_spread_ms']:.6f} | {x['first_long_compute_start']['linear_spread_ms']:.6f} | {x['first_long_compute_start']['circular_minimum_covering_arc_ms']:.6f} |")
    lines += ["",f"Long 每层纯 C={period:.9f} ms。JSON 保留每卡原请求 ID、实际 L0 释放/接纳/计算起点，以及每个共同逻辑 Long 层的跨度；L0 可能在接纳前跨请求预取。", "",
        "绝对开始跨度超过层 C 不能单独证明相位分散。下面仅取前 20 卡实际同时处于连续 Long 段、且各卡该内部层 L1–L7 的释放全部落在共同区间的波次，对释放时刻模 Long 层 C 计算最小覆盖圆弧。第一轮另限原暖窗 [2,4)，第二轮限 [4,12)。", "",
        "| 轮次 | 前20卡同时Long的实际区间(ms) | 纳入内部波次数 | IO释放圆弧 p50(ms) | p95(ms) | 最大空隙 p50(ms) | 全32卡Long数范围 / 时间均值 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for x in cycles:
        lo,hi=x['all_front20_continuously_long_interval_ms'];arc=x['internal_overlap_io_circular_arc_ms'];gap=x['internal_overlap_io_circular_maximum_gap_ms']
        count=x['actual_roles_during_front20_overlap']['global_long_concurrency']
        lines.append(f"| {x['cycle']+1} | [{lo:.6f},{hi:.6f}) | {arc['count']} | {arc['p50']:.6f} | {arc['p95']:.6f} | {gap['p50']:.6f} | {count['min']}–{count['max']} / {count['mean']:.6f} |")
    lines += ["",
        "“前20卡共同Long”只描述被对齐的卡群，不代表整机只有20张Long。全局Long数来自所有32卡的真实接纳/完成区间，对同刻事件先合并再计算有正长度的区间；下一请求L0预取未改写当前请求角色。", "",
        "## 卡时间与角色占用", "",
        "| 窗口(s) | 设备U% | 暴露stall% | 平均Long卡数 | 平均Short卡数 | 后12平均Long卡数 | 后12平均Short卡数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for w in windows:
        roles=w['roles'];rear=w['groups']['rear12']
        lines.append(f"| [{w['start_ms']/1000:g},{w['end_ms']/1000:g}) | {w['device_U_percent']:.6f} | {w['exposed_stall_percent']:.6f} | {roles['long']['time_average_active_npu_count']:.6f} | {roles['short']['time_average_active_npu_count']:.6f} | {rear['long']['time_average_active_npu_count']:.6f} | {rear['short']['time_average_active_npu_count']:.6f} |")
    lines += ["", "全部窗口均 32 卡全程 active；等待按真实计算区间的补集计算，未把重叠的 I/O 生命周期相加。",
        f"Short 自身的计算/占用比例从 [2,4) 的 {100*windows[0]['roles']['short']['conditional_compute_fraction']:.6f}% 升到 [4,12) 的 {100*windows[1]['roles']['short']['conditional_compute_fraction']:.6f}%；因此恢复不只是 Long 占比增大对整机平均数的遮掩。原 [2,4) 的 stall 占整个 [2,12) stall 的 {100*original_stall_share:.6f}%。", "",
        "## 短画像组成及预取预算", "",
        "读取/预算统计选取 io_start 落窗的内部层并随访到 ready 和完整层 stall，可能越过窗尾；表中暴露 stall 列则是实际裁剪到窗口的卡时间账，两者不是同一裁剪口径。", "",
        "| 窗口(s) | 画像 | 平均活跃卡数 | 暴露stall(ms) | 内层读取/预算 p50 | p95 | 内层读取不超过预算比例 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    keys={0:'32K/1024',1:'48K/1024',2:'64K/1024'}
    for w in windows[:2]:
        for p in range(3):
            x=w['profiles'][p];ratio=x['internal_read_to_budget_ratio']
            lines.append(f"| [{w['start_ms']/1000:g},{w['end_ms']/1000:g}) | {keys[p]} | {x['time_average_active_npu_count']:.6f} | {x['stall_ms']:.6f} | {ratio['p50']:.6f} | {ratio['p95']:.6f} | {100*x['read_within_budget_fraction']:.6f}% |")
    fixed_w=fixed['windows'][-1]
    lines += ["", "三个短画像的 simulator 类别均为 SL，长画像为 LL；这里 short 是相对于 Long 计算时间而言。", "",
        "## 固定角色的对照", "",
        f"同一全局三倍人口的 fixed Baseline 在 [2,12) U={fixed_w['device_U_percent']:.6f}%，Short 自身计算/占用={100*fixed_w['roles']['short']['conditional_compute_fraction']:.6f}%。20 张长卡全程只处理同一 Long 画像，12 张短卡保持 Short；这与混合队列的角色切换条件不同。",
        f"前 20 卡纳入 {len(fixed['internal_waves'])} 个全部释放落在 [2,12) 的共同内部层波次，IO 释放跨度最大 {fixed['internal_io_start_spread_ms']['p100']:.6f} ms，计算起点跨度最大 {fixed['internal_compute_start_spread_ms']['p100']:.6f} ms。Long 内部读生命周期最大 {fixed['longest_internal_long_read_ms']:.6f} ms < Long C {period:.6f} ms，持续被计算遮住；同一期间 Short 仍有显著暴露等待。这里的读生命周期用于预算比较，不等于盘 busy 时间。", "",
        "后续低 stall 有日志支持：Long 起点相位、实际并发 Long/Short 数量及 Short 内部画像组成都改变了。以上是这些共同变化的描述，不能把恢复幅度全部归给自然错相，或单独归给 Long 数增加；没有做隔离三者的新增反事实。读日志也不能识别某个同时发生的 Long 就是特定 Short 的物理 FIFO 队头。", "",
        f"定序来源：`{result_path.relative_to(HERE.parent)}`，SHA256 `{sha(result_path)}`。固定来源：`{Path(fixed['result_path']).relative_to(HERE.parent)}`，SHA256 `{fixed['result_sha256']}`。详细可复算值：`repeated_mechanism.json`。", ""]
    (OUT/"repeated_mechanism.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(target),windows=[{k:w[k] for k in ('start_ms','end_ms','device_U_percent','exposed_stall_percent')} for w in windows]),ensure_ascii=False))


if __name__=="__main__":
    main()
