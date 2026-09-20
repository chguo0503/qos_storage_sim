"""Independent input/trace audit for frozen, offline OD-tailored workloads.

No simulator or experiment runner is imported.  Only complete clean replays
are measured.  Supply per complete NPU layer cycle is reconstructed only
after checking native I/O release and readiness endpoints.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, ast, bisect, csv, gzip, hashlib, json, math, re, struct
from audit_long_term import demand_trace, window_stats, overlap

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[1]
SOURCE=STUDY/"scheduled_input"
IO_GIB=176*1024/2**30
POLICIES=("od_baseline","once")


def read(p):
    with (gzip.open if str(p).endswith(".gz") else open)(p,"rt",encoding="utf-8") as f:return json.load(f)


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def interpolation(table,hit,miss):
    length=(hit+miss)/1024
    lo,hi=(96,128) if length<128 else (128,160)
    x=(length-lo)/(hi-lo);y=(miss-2048)/2048
    assert 0<=x<=1 and 0<=y<=1
    weights=[(1-x)*(1-y),(1-x)*y,x*(1-y),x*y]
    assert min(weights)>=0 and abs(math.fsum(weights)-1)<1e-14
    corners=[(lo,2048),(lo,4096),(hi,2048),(hi,4096)]
    return math.fsum(w*table[key][1] for w,key in zip(weights,corners)),list(zip(corners,weights))


def inspect_input(name):
    path=SOURCE/"inputs"/(name+".json.gz")
    manifest=read(path);meta=manifest["metadata"]
    table=ast.literal_eval((ROOT/"data").read_text())
    assert sha(ROOT/"data")==meta["data_sha256"]
    hit=meta["fixed_B_hit_tokens"]
    grid=[interpolation(table,hit,m)[0] for m in range(2048,4097)]
    assert all(a<b for a,b in zip(grid,grid[1:]))
    maxstep=max(b-a for a,b in zip(grid,grid[1:]))
    bound=4*maxstep/1000
    assert abs(bound-meta["quantization_end_bound_ms"])<1e-9
    # Independent copy of the documented hash protocol, not a core import.
    source=(ROOT/"simulator/core/sim.py").read_text()
    vnode_count=int(re.search(r"^BLOCK_RING_VIRTUAL_NODES\s*=\s*(\d+)",source,re.M)[1])
    def position(namespace,first,second):
        return int.from_bytes(hashlib.sha256(namespace+struct.pack("!QQ",first,second)).digest(),"big")
    ring=sorted((position(b"qos_storage_sim:block_ring_hash:vnode:v1\0",d,v),d)
                for d in range(3) for v in range(vnode_count))
    positions=[p for p,d in ring]
    profiles={};loads={};deltas=[];physical=[];checked=set();counts=[Counter() for _ in range(32)]
    bytes_by_npu=[[0.]*3 for _ in range(32)];pure=[0.]*32;expected_blocks=0
    fingerprint=hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    for row in sorted(manifest["requests"],key=lambda r:r["request_id"]):
        rid=row["request_id"];n=row["npu_id"];load=row["load"]
        placement=tuple(tuple((int(d),float(v)) for d,v in layer) for layer in manifest["placements"][row["placement_index"]])
        assert len(placement)==1 and row["arrival_time_ms"]==0
        fingerprint.update(repr((rid,n,row["arrival_time_ms"],load["category"],load["per_layer_us"],placement)).encode())
        assert isinstance(load["nql"],int)
        assert load["total_tokens"]==load["seq_len_k"]*1024
        assert load["total_tokens"]-load["nql"]==load["fixed_hit_tokens"]
        assert load["category"]==("LS" if load["nql"]<512 else "LL")
        prefix=int(load.get("physical_prefix_id",load["original_request_id"]))
        assert load["original_request_id"]==prefix
        physical.append(prefix)
        volume=[math.fsum(v for d,v in placement[0] if d==disk) for disk in range(3)]
        assert all(v==IO_GIB and d in (0,1,2) for d,v in placement[0])
        assert len(placement[0])==load["fixed_hit_tokens"]//128
        assert abs(math.fsum(volume)-load["per_layer_kv_gb"])<1e-14
        assert abs(load["required_bw_input_gbps"]-math.fsum(volume)*1e6/load["per_layer_us"])<1e-10
        identity=(prefix,row["placement_index"])
        if identity not in checked:
            for block,(disk,value) in enumerate(placement[0]):
                p=position(b"qos_storage_sim:block_ring_hash:block:v1\0",prefix,block)
                assert disk==ring[bisect.bisect_left(positions,p)%len(ring)][1]
            checked.add(identity)
        if load["role"]=="A":
            assert (load["total_tokens"],load["nql"])==(128*1024,256)
            assert load["per_layer_us"]==table[128,256][1]
        else:
            assert load["fixed_hit_tokens"]==hit and 2048<=load["nql"]<=4096
            c,anchors=interpolation(table,hit,load["nql"])
            assert abs(c-load["per_layer_us"])<1e-8
            recorded=load["compute_interpolation_anchors"]
            assert len(recorded)==4
            for item,(key,weight) in zip(recorded,anchors):
                assert (item["length_k"],item["miss"])==key
                assert abs(item["weight"]-weight)<1e-14 and item["C_us"]==table[key][1]
            # The same bilinear grid gives exactly the fixed-hit read volume.
            assert abs(math.fsum(weight*table[key][3] for key,weight in anchors)-math.fsum(volume))<1e-14
            if load["position"]==2:
                target=load["target_unquantized_C_us"]
                assert grid[0]<=target<=grid[-1]
                best=min(range(len(grid)),key=lambda j:abs(grid[j]-target))
                assert best+2048==load["nql"]
                assert 8*abs(c-target)/1000<=bound+1e-9
                deltas.append(8*(c/1000-meta["original_B_C_ms"]))
        profiles[rid]=dict(npu=n,role=load["role"],C_ms=load["per_layer_us"]/1000,
            rates=[v*1e6/load["per_layer_us"] for v in volume],volume_GiB_by_ssu=volume)
        loads[rid]=load;counts[n][load["role"]]+=1
        pure[n]+=8*load["per_layer_us"]/1000
        for d,v in enumerate(volume):bytes_by_npu[n][d]+=8*v
        expected_blocks+=len(placement[0])*8
    assert fingerprint.hexdigest()==manifest["input_fingerprint"]
    assert len(profiles)==len(manifest["requests"])
    if not meta.get("physical_prefix_reuse",True):assert len(set(physical))==len(physical)
    for row in counts:assert row==Counter(A=meta["cycles"],B=2*meta["cycles"])
    tails=[l for l in loads.values() if l["position"]==2]
    details=dict(name=name,manifest_sha256=sha(path),input_fingerprint=fingerprint.hexdigest(),
        metadata=meta,request_count=len(profiles),expected_blocks=expected_blocks,
        unique_physical_ids=len(set(physical)),physical_prefix_reuse=len(set(physical))<len(physical),
        all_placements_independently_ring_verified=True,ring_placements_checked=len(checked),
        all_profiles_and_nearest_integer_misses_verified=True,
        role_counts_per_npu=[dict(c) for c in counts],pure_compute_ms_per_npu=pure,
        total_read_GiB_by_ssu=[math.fsum(v[d] for v in bytes_by_npu) for d in range(3)],
        pure_compute_reference_rate_GiB_s_by_ssu=[math.fsum(bytes_by_npu[n][d]*1000/pure[n] for n in range(32)) for d in range(3)],
        reference_rate_definition="sum over NPUs of own total bytes on disk / own total pure compute seconds; NOT time-average active-profile demand or arrival rate",
        tail_miss_min=min(l["nql"] for l in tails),tail_miss_max=max(l["nql"] for l in tails),
        tail_C_min_ms=min(l["per_layer_us"] for l in tails)/1000,tail_C_max_ms=max(l["per_layer_us"] for l in tails)/1000,
        quantization_individual_end_bound_ms=bound,quantization_pair_spread_bound_ms=2*bound,
        signed_compute_change_card_ms=math.fsum(deltas),positive_compute_change_card_ms=math.fsum(max(v,0) for v in deltas),
        negative_compute_change_card_ms=math.fsum(min(v,0) for v in deltas),absolute_compute_change_card_ms=math.fsum(abs(v) for v in deltas),
        reduced_tail_request_count=sum(v<0 for v in deltas))
    details["profile_groups"]={}
    for position,label in ((0,"A"),(1,"B1"),(2,"B2")):
        group=[l for l in loads.values() if l["position"]==position]
        details["profile_groups"][label]=dict(count=len(group),
            total_tokens_min=min(l["total_tokens"] for l in group),total_tokens_max=max(l["total_tokens"] for l in group),
            miss_min=min(l["nql"] for l in group),miss_max=max(l["nql"] for l in group),
            per_layer_compute_ms_min=min(l["per_layer_us"] for l in group)/1000,
            per_layer_compute_ms_max=max(l["per_layer_us"] for l in group)/1000,
            per_layer_read_MiB_min=min(l["per_layer_kv_gb"] for l in group)*1024,
            per_layer_read_MiB_max=max(l["per_layer_kv_gb"] for l in group)*1024)
    compute_floor=max(pure)
    disk_floor=max(details["total_read_GiB_by_ssu"])*1000/40
    floor=max(compute_floor,disk_floor)
    details["finite_input_capacity_bound"]=dict(max_per_npu_pure_compute_ms=compute_floor,
        bottleneck_disk_service_ms=disk_floor,makespan_lower_bound_ms=floor,
        full_U_upper_bound_percent=min(100.,100*math.fsum(pure)/(32*floor)),
        caveat="Necessary workload-only lower bound, not an attainable scheduler optimum and not a bound on a selected warm window.")
    designed_ms=meta["cycles"]*meta["period_ms"]
    details["frozen_input_arithmetic"]=dict(designed_end_ms=designed_ms,
        mean_pure_compute_ms_per_npu=math.fsum(pure)/32,
        mean_pure_compute_ms_per_npu_per_cycle=math.fsum(pure)/(32*meta["cycles"]),
        U_percent_if_all_finish_at_designed_end=100*math.fsum(pure)/(32*designed_ms),
        caveat="Calculated after offline planning from the frozen input. This is a conditional arithmetic identity, not an independent pre-planning prediction or a measured replay result.")
    return profiles,loads,details


def verify_sources(name,policy,command,details):
    plan_dir=SOURCE/"planning"/name;formal=SOURCE/"formal"/(name+"_"+policy)
    plan=read(plan_dir/"command.json")
    assert plan["status"]=="complete" and plan["source_unchanged"] and plan["hook_restored"]
    assert plan["manifest_sha256"]==details["manifest_sha256"]==command["manifest_sha256"]
    assert command["source_and_artifacts_unchanged"] and command["runtime_hooks"] is False
    resolutions={}
    for key,digest in command["source_and_artifact_sha256"].items():
        path=ROOT/key
        if path.name=="clean_replay.py":path=formal/"clean_replay_source.py"
        assert sha(path)==digest,("formal source mismatch",key)
        resolutions[key]=str(path.relative_to(ROOT))
    for key,digest in plan["source_sha256"].items():assert sha(ROOT/key)==digest
    artifacts=plan.get("planning_artifact_sha256")
    if artifacts:
        assert plan["planning_artifacts_unchanged"]
        assert artifacts==details["metadata"]["planning_artifact_sha256"]
        for key,digest in artifacts.items():
            path=plan_dir/"generator_source.py" if Path(key).name.startswith("run_interpolated") else ROOT/key
            assert sha(path)==digest,("planning source mismatch",key)
            assert command["source_and_artifact_sha256"][str(path.relative_to(ROOT))]==digest
    if details["metadata"]["cycles"]>=50:assert artifacts,"Long planning must freeze all source artifacts before execution"
    return dict(formal_files_verified=len(resolutions),planning_artifact_chain_present=bool(artifacts),
                planning_files_verified=len(plan["source_sha256"]),source_resolutions=resolutions)


def population_slo(rows,profiles):
    passed=0;ratios=[]
    for r in rows:
        own=8*profiles[r["request_id"]]["C_ms"]
        elapsed=r["completion_time_ms"]-r["admission_time_ms"]
        passed+=elapsed<=1.5*own+1e-9;ratios.append(elapsed/own)
    ratios.sort()
    return dict(passed=passed,count=len(rows),percent=100*passed/len(rows),
                mean_normalized=math.fsum(ratios)/len(rows),p95_normalized=ratios[math.ceil(.95*len(rows))-1],
                definition="same entire finite request population; admission to prefill completion <= 1.5*own 8C + 1e-9ms; excludes admission queue")


def add_role_metrics(row):
    row["by_role"]={}
    dt=row["end_ms"]-row["start_ms"]
    for role in ("A","B"):
        compute=math.fsum(p.get(role,0.) for p in row["per_npu_role_compute_ms"])
        active=math.fsum(p.get(role,0.) for p in row["per_npu_role_active_ms"])
        row["by_role"][role]=dict(compute_card_ms=compute,active_card_ms=active,stall_card_ms=active-compute,
            active_U_percent=100*compute/active if active else None,
            share_of_window_card_time_percent=100*active/(32*dt),
            contribution_to_fleet_U_percentage_points=100*compute/(32*dt))
    return row


def layer_cycle_supply(summary,profiles):
    per_npu=[[] for _ in range(32)]
    for batch in summary["microbatch_metrics"]:
        rid=batch["member_request_ids"][0]
        for l in batch["layer_metrics"]:per_npu[batch["npu_id"]].append((rid,l))
    cycles=[[] for _ in range(32)];endpoint_error=0.;counts=0
    for n,layers in enumerate(per_npu):
        layers.sort(key=lambda z:z[1]["compute_start_ms"])
        for (rid,current),(nrid,nxt) in zip(layers,layers[1:]):
            a,z=current["compute_start_ms"],nxt["compute_start_ms"]
            error=abs(nxt["io_start_time_ms"]-a);endpoint_error=max(endpoint_error,error)
            assert error<1e-8 and a<=nxt["io_ready_time_ms"]<=z+1e-8 and z>a
            assert abs(z-current["compute_end_ms"]-nxt["io_barrier_wait_ms"])<1e-8
            V=profiles[nrid]["volume_GiB_by_ssu"]
            cycles[n].append(dict(start_ms=a,end_ms=z,request_id=rid,next_request_id=nrid,
                layer=current["layer"],next_layer=nxt["layer"],same_request=rid==nrid,
                next_io_ready_ms=nxt["io_ready_time_ms"],actual_cycle_read_GiB_by_ssu=V,
                mean_supply_GiB_s_by_ssu=[v*1000/(z-a) for v in V]))
            counts+=1
    return dict(per_npu_cycles=cycles,count=counts,maximum_release_endpoint_error_ms=endpoint_error,
        method="At each compute start, the only new read is next layer or next request L0; all of that read reaches NPU before next compute start. Therefore exact SSD bytes in this full per-NPU cycle = next-layer manifest V. Initial cold reads and final compute without prefetch are excluded.",
        caveat="Asynchronous cycle averages are not instantaneous SSD rates; their sum may exceed physical disk capacity. Display clipping must retain full-cycle denominator. No exact physical service integral at arbitrary 20s window boundaries is claimed.")


def audit_policy(name,policy,profiles,loads,details):
    path=SOURCE/"formal"/(name+"_"+policy)
    if not (path/"command.json").exists():return dict(complete=False,status="completed_raw_not_yet_available_locally",policy=policy)
    command=read(path/"command.json")
    if command["status"]!="complete" or not (path/"result.json.gz").exists():
        return dict(complete=False,status=command["status"],policy=policy)
    sources=verify_sources(name,policy,command,details)
    result=read(path/"result.json.gz");summary=result["summary"]
    assert result["input_fingerprint"]==details["input_fingerprint"]
    assert all(summary["invariants"].values()) and summary["completed_blocks"]==details["expected_blocks"]
    assert len(summary["request_metrics"])==len(profiles)
    assert summary["num_npu"]==32 and summary["num_ssu"]==3 and summary["n_layers"]==8 and summary["batch_size"]==1
    assert summary["cross_request_layer0_prefetch"]
    for batch in summary["microbatch_metrics"]:
        assert len(batch["member_request_ids"])==1 and len(batch["layer_metrics"])==8
        rid=batch["member_request_ids"][0]
        for layer in batch["layer_metrics"]:
            assert layer["compute_duration_ms"]==profiles[rid]["C_ms"]
            assert abs(layer["compute_end_ms"]-layer["compute_start_ms"]-profiles[rid]["C_ms"])<1e-8
    first_finish=min(max(r["completion_time_ms"] for r in summary["request_metrics"] if r["npu_id"]==n) for n in range(32))
    horizon=min(80000.,2000.*math.floor(first_finish/2000.))
    if details["metadata"]["cycles"]>=50:assert first_finish>60000
    segments=demand_trace(summary["request_metrics"],profiles,0.,first_finish)
    bins=[add_role_metrics(window_stats(summary,profiles,segments,a,a+2000)) for a in range(0,int(horizon),2000)]
    fixed=[add_role_metrics(window_stats(summary,profiles,segments,a,z)) for a,z in
           ((2000.,4000.),(20000.,40000.),(40000.,60000.),(20000.,60000.),(60000.,78000.)) if z<=horizon]
    assert all(w["all_npus_active"] for w in bins)
    assert all(w["mixed_npus"]==32 for w in fixed if w["start_ms"]>=20000)
    cohort_ids={f"{int(w['start_ms'])}:{int(w['end_ms'])}":[r["request_id"] for r in summary["request_metrics"] if w["start_ms"]<=r["admission_time_ms"]<w["end_ms"]] for w in fixed}
    full_supply=[]
    for disk in summary["disk_stats"]:
        d=disk["ssu_id"];assert abs(disk["completed_gb"]-details["total_read_GiB_by_ssu"][d])<1e-7
        mean=disk["completed_gb"]*1000/summary["makespan_ms"]
        assert abs(mean-40*disk["utilization"])<1e-7
        full_supply.append(dict(ssu_id=d,mean_GiB_s=mean,start_ms=0,end_ms=summary["makespan_ms"],
                                definition="exact entire simulation physical SSD bytes / makespan; includes startup and drain, not a 20s-window rate"))
    if policy=="od_baseline":
        q=summary["ssd_queue_depth"]
        assert max(map(max,q["peak_outstanding_blocks_by_npu_ssu"]))<=256
        assert q["host_deferred_blocks_at_stop"]==q["ssd_outstanding_blocks_at_stop"]==q["link_outstanding_blocks_at_stop"]==0
        planning=read(SOURCE/"planning"/name/"PLANNING_ONLY_result.json.gz")["summary"]
        assert planning["microbatch_metrics"]==summary["microbatch_metrics"]
        assert planning["request_metrics"]==summary["request_metrics"]
    native=read(path/"analysis.json")
    for saved in native["windows"]:
        if saved["end_ms"]>horizon:continue
        row=window_stats(summary,profiles,segments,saved["start_ms"],saved["end_ms"])
        assert abs(row["U_percent"]-saved["U_percent"])<1e-8
        assert row["slo_1p5_passed"]==saved["slo"]["passed"] and row["slo_1p5_count"]==saved["slo"]["count"]
    per_cycle=[];group={(b["npu_id"],loads[b["member_request_ids"][0]]["cycle"],loads[b["member_request_ids"][0]]["position"]):b for b in summary["microbatch_metrics"]}
    requests={r["request_id"]:r for r in summary["request_metrics"]}
    for k in range(details["metadata"]["cycles"]):
        tails=[group[n,k,2] for n in range(32)]
        ends=[b["completion_time_ms"] for b in tails];starts=[b["layer_metrics"][0]["compute_start_ms"] for b in tails]
        waits=[l["io_barrier_wait_ms"] for n in range(32) for p in (1,2) for l in group[n,k,p]["layer_metrics"][1:]]
        E=(k+1)*details["metadata"]["period_ms"]
        row=dict(cycle=k,earliest_tail_start_ms=min(starts),latest_tail_start_ms=max(starts),tail_end_spread_ms=max(ends)-min(ends),
            maximum_target_error_ms=max(abs(e-E) for e in ends),B_internal_stall_max_ms=max(waits),ends_before_first_card_finishes=max(ends)<=first_finish)
        cycle_compute=cycle_stall=cycle_elapsed=0.;A_compute=A_stall=B_compute=B_stall=0.
        A_starts=[]
        for n in range(32):
            first=group[n,k,0];first_id=first["member_request_ids"][0]
            elapsed=ends[n]-requests[first_id]["admission_time_ms"]
            A_starts.append(first["layer_metrics"][0]["compute_start_ms"])
            computed=stalled=0.
            for position in (0,1,2):
                layers=group[n,k,position]["layer_metrics"]
                c=math.fsum(l["compute_duration_ms"] for l in layers)
                w=math.fsum(l["io_barrier_wait_ms"] for l in layers)
                computed+=c;stalled+=w
                if position==0:A_compute+=c;A_stall+=w
                else:B_compute+=c;B_stall+=w
            assert abs(elapsed-computed-stalled)<1e-6
            cycle_compute+=computed;cycle_stall+=stalled;cycle_elapsed+=elapsed
        row.update(complete_cycle_card_time_weighted_U_percent=100*cycle_compute/cycle_elapsed,
            complete_cycle_card_time_ms=cycle_elapsed,complete_cycle_compute_card_ms=cycle_compute,
            complete_cycle_stall_card_ms=cycle_stall,A_compute_card_ms=A_compute,A_stall_card_ms=A_stall,
            B_compute_card_ms=B_compute,B_stall_card_ms=B_stall,A_compute_start_spread_ms=max(A_starts)-min(A_starts),
            earliest_A_compute_start_ms=min(A_starts),latest_A_compute_start_ms=max(A_starts),
            latest_tail_end_ms=max(ends),
            cycle_U_definition="sum compute over per-card complete ABB cycles / sum per-card ABB durations; asynchronous card-time weighted statistic, NOT a common wall-clock window fleet U")
        if k+1<details["metadata"]["cycles"]:
            row["next_A_prefetch_min_slack_ms"]=min(ends[n]-group[n,k+1,0]["layer_metrics"][0]["io_ready_time_ms"] for n in range(32))
        if policy=="od_baseline":
            assert row["maximum_target_error_ms"]<=details["quantization_individual_end_bound_ms"]+1e-7
            assert max(waits)<1e-7 and row.get("next_A_prefetch_min_slack_ms",0)>=-1e-7
        per_cycle.append(row)
    # Compact true compute/stall timeline for preselected cycles 25 and 26.
    a,z=24*details["metadata"]["period_ms"],26*details["metadata"]["period_ms"]
    if details["metadata"]["cycles"]<26:
        a,z=0.,min(2*details["metadata"]["period_ms"],first_finish)
    timeline=[]
    if z<=first_finish:
        for batch in summary["microbatch_metrics"]:
            rid=batch["member_request_ids"][0];load=loads[rid]
            for layer in batch["layer_metrics"]:
                s,e=layer["compute_start_ms"],layer["compute_end_ms"]
                if overlap(s,e,a,z)>0:timeline.append(dict(npu_id=batch["npu_id"],start_ms=max(s,a),end_ms=min(e,z),kind="A" if load["position"]==0 else "B1" if load["position"]==1 else "B2",request_id=rid,layer=layer["layer"]))
                if overlap(s-layer["io_barrier_wait_ms"],s,a,z)>0:timeline.append(dict(npu_id=batch["npu_id"],start_ms=max(a,s-layer["io_barrier_wait_ms"]),end_ms=min(s,z),kind="stall",request_id=rid,layer=layer["layer"]))
    supply=layer_cycle_supply(summary,profiles)
    examples=[]
    for cycle in supply["per_npu_cycles"][0]:
        profile=profiles[cycle["request_id"]]
        if not (cycle["same_request"] and profile["role"]=="A" and a<=cycle["start_ms"] and cycle["end_ms"]<=z):continue
        C=profile["C_ms"];T=cycle["end_ms"]-cycle["start_ms"]
        V=math.fsum(profile["volume_GiB_by_ssu"])
        B=V*1000/C;b=V*1000/T
        assert T>=C-1e-8 and abs(b/B-C/T)<1e-12
        examples.append(dict(npu_id=0,request_id=cycle["request_id"],layer=cycle["layer"],
            start_ms=cycle["start_ms"],end_ms=cycle["end_ms"],C_ms=C,T_ms=T,I_ms=max(0,T-C),
            next_io_ready_ms=cycle["next_io_ready_ms"],per_layer_read_MiB=V*1024,
            B_GiB_s=B,b_GiB_s=b,cycle_U_percent=100*C/T,b_over_B=b/B,
            selection="NPU 0, first three complete same-request A layer cycles in the preselected displayed time interval"))
        if len(examples)==3:break
    return dict(complete=True,status="complete",policy=policy,passed=True,source_checks=sources,
        result_sha256=sha(path/"result.json.gz"),command_sha256=sha(path/"command.json"),manifest_sha256=details["manifest_sha256"],
        makespan_ms=summary["makespan_ms"],first_card_drains_ms=first_finish,horizon_ms=horizon,
        two_second_bins=bins,fixed_windows=fixed,cohort_request_ids=cohort_ids,
        same_population_slo_1p5=population_slo(summary["request_metrics"],profiles),
        same_population_slo_1p5_by_role={role:population_slo(
            [r for r in summary["request_metrics"] if profiles[r["request_id"]]["role"]==role],profiles) for role in ("A","B")},
        full_physical_SSD_supply=full_supply,per_disk_demand_segments=segments,
        layer_cycle_supply=supply,cycle_checks=per_cycle,representative_internal_A_cycles=examples,
        selected_two_cycles=dict(start_ms=a,end_ms=z,timeline=timeline),
        original_once_queue_depth_unlimited=(policy=="once"),
        full_lifetime_U_percent=100*summary["fleet_npu_compute_utilization"])


def export(name,audit):
    output=HERE/("scheduled_"+name)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for policy,result in audit.get("policies",{}).items():
        if not result["complete"]:continue
        for kind,key in (("2s","two_second_bins"),("fixed","fixed_windows")):
            for r in result[key]:
                rows.append(dict(policy=policy,window_type=kind,start_s=r["start_ms"]/1000,end_s=r["end_ms"]/1000,
                    U_percent=r["U_percent"],SLO_1p5_percent=r["slo_1p5_percent"],cohort_count=r["slo_1p5_count"],
                    all_npus_active=r["all_npus_active"],mixed_npus=r["mixed_npus"],
                    strict_nominal_underload=r["strict_nominal_underload"],
                    **{f"SSU{d}_{label}":r[field][d] for label,field in (("mean_demand_GiB_s","per_disk_mean_GiB_s"),("peak_demand_GiB_s","per_disk_peak_GiB_s"),("overload_percent","per_disk_overload_percent")) for d in range(3)}))
    if rows:
        with (output/"windows.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cycle_rows=[];population_rows=[];supply_rows=[]
    for policy,result in audit.get("policies",{}).items():
        if not result["complete"]:continue
        cycle_rows.extend(dict(policy=policy,**r) for r in result["cycle_checks"])
        for role,row in {"all":result["same_population_slo_1p5"],**result["same_population_slo_1p5_by_role"]}.items():
            population_rows.append(dict(policy=policy,role=role,**{k:v for k,v in row.items() if k!="definition"}))
        for n,cycles in enumerate(result["layer_cycle_supply"]["per_npu_cycles"]):
            for row in cycles:
                supply_rows.append(dict(policy=policy,npu_id=n,start_ms=row["start_ms"],end_ms=row["end_ms"],
                    request_id=row["request_id"],next_request_id=row["next_request_id"],layer=row["layer"],next_layer=row["next_layer"],
                    same_request=row["same_request"],next_io_ready_ms=row["next_io_ready_ms"],
                    **{f"SSU{d}_actual_cycle_GiB":v for d,v in enumerate(row["actual_cycle_read_GiB_by_ssu"])},
                    **{f"SSU{d}_mean_supply_GiB_s":v for d,v in enumerate(row["mean_supply_GiB_s_by_ssu"])}))
    for filename,table in (("cycles.csv",cycle_rows),("population_slo.csv",population_rows),("layer_cycle_supply.csv",supply_rows)):
        if table:
            fields=list(dict.fromkeys(k for r in table for k in r))
            with (output/filename).open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    # The exact 76k full-cycle rows already live in CSV; do not duplicate them
    # into a ~75 MB pretty-printed report JSON.
    compact=dict(audit,policies={})
    for policy,result in audit.get("policies",{}).items():
        compact["policies"][policy]=dict(result)
        if result["complete"]:
            compact["policies"][policy]["layer_cycle_supply"]={k:v for k,v in result["layer_cycle_supply"].items() if k!="per_npu_cycles"}
            compact["policies"][policy]["layer_cycle_supply"]["complete_cycle_rows_csv"]="layer_cycle_supply.csv"
    compact["derived_csv_sha256"]={p.name:sha(p) for p in output.glob("*.csv")}
    write(output/"audit.json",compact)
    return output


def audit_depth_control(name,details):
    directory=SOURCE/"formal"/(name+"_od_depth_none")
    base=SOURCE/"formal"/(name+"_od_baseline")
    if not all((p/"command.json").exists() and (p/"result.json.gz").exists() for p in (directory,base)):
        return dict(complete=False,status="both completed depth variants not yet available locally")
    commands=[read(p/"command.json") for p in (base,directory)]
    if not all(c["status"]=="complete" for c in commands):return dict(complete=False,status="not complete")
    assert all(c["manifest_sha256"]==details["manifest_sha256"] for c in commands)
    assert all(c["source_and_artifacts_unchanged"] and not c["runtime_hooks"] for c in commands)
    for key in set(commands[0]["source_and_artifact_sha256"])&set(commands[1]["source_and_artifact_sha256"]):
        assert commands[0]["source_and_artifact_sha256"][key]==commands[1]["source_and_artifact_sha256"][key]
    for key,digest in commands[1]["source_and_artifact_sha256"].items():
        assert sha(ROOT/key)==digest
    raws=[read(p/"result.json.gz") for p in (base,directory)]
    assert all(r["input_fingerprint"]==details["input_fingerprint"] for r in raws)
    one,two=[r["summary"] for r in raws]
    assert all(two["invariants"].values()) and two["completed_blocks"]==details["expected_blocks"]
    assert "ssd_queue_depth" not in two
    assert one["microbatch_metrics"]==two["microbatch_metrics"]
    req1={r["request_id"]:r for r in one["request_metrics"]};req2={r["request_id"]:r for r in two["request_metrics"]}
    assert set(req1)==set(req2)
    changed=Counter()
    for rid,row in req1.items():
        for key,value in row.items():
            if value!=req2[rid][key]:changed[key]+=1
    assert set(changed)<=set(("avg_end_to_end_io_latency_ms","avg_ssd_queue_wait_ms"))
    assert one["makespan_ms"]==two["makespan_ms"]
    return dict(complete=True,passed=True,same_frozen_input_and_common_source_hashes=True,
        all_request_timing_fields_equal=True,all_microbatch_and_layer_records_equal=True,
        request_count=len(req1),layer_count=8*len(req1),changed_diagnostic_fields=dict(changed),
        result_sha256={"depth256":sha(base/"result.json.gz"),"unlimited":sha(directory/"result.json.gz")},
        interpretation="For this exact input, changing only OD queue depth to unlimited changes no request/layer timing. Enqueue-based latency diagnostics change because waiting moves to the host; this is not a speedup. Does not generalize to other inputs.")


def audit_seed_control(name,details):
    directory=SOURCE/"sensitivity"/(name+"_od_seed19")
    base=SOURCE/"formal"/(name+"_od_baseline")
    if not all((p/"command.json").exists() and (p/"result.json.gz").exists() for p in (directory,base)):
        return dict(complete=False,status="both completed seed variants not yet available locally")
    command=read(directory/"command.json")
    if command["status"]!="complete":return dict(complete=False,status=command["status"])
    assert command["seed"]==19 and command["planning_seed"]==7
    assert command["manifest_sha256"]==details["manifest_sha256"] and command["input_fingerprint"]==details["input_fingerprint"]
    assert command["input_bytes_unchanged"] and command["source_unchanged"] and command["planning_command_unchanged"]
    assert not command["observer_hooks"] and not command["runtime_planning_hooks"]
    assert command["source_sha256_before"]==command["source_sha256_after"]
    for key,digest in command["source_sha256_before"].items():assert sha(ROOT/key)==digest
    raws=[read(p/"result.json.gz") for p in (base,directory)]
    one,two=[r["summary"] for r in raws]
    assert one["request_metrics"]==two["request_metrics"]
    assert one["microbatch_metrics"]==two["microbatch_metrics"]
    assert all(two["invariants"].values()) and two["completed_blocks"]==details["expected_blocks"]
    return dict(complete=True,passed=True,seed7_seed19_request_and_layer_records_exact=True,
        seed19_result_sha256=sha(directory/"result.json.gz"),source_and_input_hashes_verified=True,
        interpretation="Only equal-timestamp client submission order changes. In this run it caused no effective OD timing perturbation; it is not compute/service noise robustness or a second random workload sample.")


def main():
    p=argparse.ArgumentParser();p.add_argument("--name",default="abb_interp_unique_50");p.add_argument("--require-both",action="store_true");a=p.parse_args()
    path=SOURCE/"inputs"/(a.name+".json.gz")
    if not path.exists():
        print(json.dumps(dict(complete=False,name=a.name,status="frozen manifest not yet available")));return
    profiles,loads,details=inspect_input(a.name)
    policies={policy:audit_policy(a.name,policy,profiles,loads,details) for policy in POLICIES}
    all_complete=all(p["complete"] for p in policies.values())
    overlap_counts={}
    if all_complete:
        assert policies["od_baseline"]["manifest_sha256"]==policies["once"]["manifest_sha256"]
        for key,ids in policies["od_baseline"]["cohort_request_ids"].items():
            if key in policies["once"]["cohort_request_ids"]:
                other=policies["once"]["cohort_request_ids"][key]
                overlap_counts[key]=dict(OD_count=len(ids),Once_count=len(other),shared_count=len(set(ids)&set(other)))
    audit=dict(name=a.name,all_policies_complete=all_complete,input=details,policies=policies,
        depth_control=audit_depth_control(a.name,details),
        seed_control=audit_seed_control(a.name,details),
        window_cohort_overlap=overlap_counts,
        independent_method="raw JSON input model, RingHash, full layer durations, compute/stall integrals, exact admitted-profile demand events; no simulator/runner imported",
        source_sha256={p.name:sha(p) for p in (Path(__file__),HERE/"audit_long_term.py")})
    out=export(a.name,audit)
    print(json.dumps(dict(name=a.name,all_complete=all_complete,output=str(out),statuses={k:v["status"] for k,v in policies.items()})))
    if a.require_both and not all_complete:raise SystemExit("Both clean replays must complete before paired final reporting")


if __name__=="__main__":main()
