#!/usr/bin/env python3
"""Exact clipped physical service and per-layer deadline evidence from passive traces."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
LEFT,RIGHT,DT=3200.,4000.,.5
EDGES=np.arange(LEFT,RIGHT+DT/2,DT)
N=len(EDGES)-1
EXAMPLES={'same_time_short':(22000046,4),'largest_short':(26000038,5),'long_l0':(16,0)}

def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,d):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'wt') as f:
        json.dump(d,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def add_interval(row,a,z,value):
    a,z=max(LEFT,a),min(RIGHT,z)
    if z<=a:return
    i=min(N-1,int((a-LEFT)//DT));j=min(N-1,int((z-LEFT)//DT))
    row[i]+=value*(min(z,EDGES[i+1])-a)/DT
    if j>i:
        row[i+1:j]+=value
        row[j]+=value*(z-EDGES[j])/DT

def stage_bins(arr,start,end,resource,resources,rate):
    """Integrate each immutable, constant-rate physical service interval."""
    a=np.maximum(LEFT,arr[:,start]);z=np.minimum(RIGHT,arr[:,end])
    keep=z>a;a,z=a[keep],z[keep];r=arr[keep,resource].astype(int)
    assert np.all(z-a<=DT+1e-8),'This vectorization expects individual block service shorter than one bin.'
    first=np.minimum(N-1,((a-LEFT)//DT).astype(int));last=np.minimum(N-1,((z-LEFT)//DT).astype(int))
    values=np.zeros((resources,N))
    np.add.at(values,(r,first),rate*(np.minimum(z,EDGES[first+1])-a)/DT)
    more=last>first
    np.add.at(values,(r[more],last[more]),rate*(z[more]-EDGES[last[more]])/DT)
    direct=np.bincount(r,weights=(z-a)*rate/1000,minlength=resources)
    recovered=values.sum(axis=1)*DT/1000
    assert np.allclose(direct,recovered,rtol=1e-10,atol=1e-9)
    return values,direct

def validate_intervals(arr,start,end,resource,resources,rate):
    errors=[]
    for r in range(resources):
        a=arr[arr[:,resource]==r];a=a[np.argsort(a[:,start],kind='stable')]
        if len(a)>1 and np.any(a[1:,start]<a[:-1,end]-1e-8):errors.append(r)
    assert not errors,('Overlapping physical service',resource,errors)
    assert np.allclose(arr[:,end]-arr[:,start],arr[:,6]*1000/rate,rtol=1e-9,atol=1e-8)

def progress(blocks,t,start,end):
    """Physical bytes transferred, including fractional service of an active block."""
    return blocks[:,6]*1024*np.clip((t-blocks[:,start])/(blocks[:,end]-blocks[:,start]),0.,1.)

def cumulative_curves(blocks,release,deadline,ready):
    times=np.unique(np.r_[blocks[:,9],blocks[:,10],blocks[:,11],blocks[:,12],release,deadline,ready])
    curves={'time_ms':times.tolist()}
    for name,start,end in [('ssd_cumulative_MiB',9,10),('link_cumulative_MiB',11,12)]:
        changes=np.zeros(len(times))
        rates=blocks[:,6]*1024/(blocks[:,end]-blocks[:,start])
        np.add.at(changes,np.searchsorted(times,blocks[:,start]),rates)
        np.add.at(changes,np.searchsorted(times,blocks[:,end]),-rates)
        slopes=np.cumsum(changes)
        values=np.r_[0.,np.cumsum(slopes[:-1]*np.diff(times))]
        assert np.min(np.diff(values))>=-1e-6
        assert math.isclose(values[-1],blocks[:,6].sum()*1024,rel_tol=1e-9,abs_tol=1e-6)
        curves[name]=np.clip(values,0,blocks[:,6].sum()*1024).tolist()
    return curves

def analyze(policy):
    directory=HERE/'traces'/policy;command=read(directory/'command.json')
    assert command['status']=='prefix_complete'
    trace_path=directory/'trace.json.gz';trace=read(trace_path)
    assert trace['audit_passed'] and trace['audit']['passed']
    source=trace['source'];manifest=Path(source['manifest']);result=Path(source['reference_result'])
    assert sha(manifest)==source['manifest_sha256'] and sha(result)==source['reference_sha256']
    if command.get('trace_sha256'):assert sha(trace_path)==command['trace_sha256']
    man,raw=read(manifest),read(result)
    arr=np.asarray(trace['rows'],dtype=float);columns=trace['columns']
    assert columns==['request_id','npu_id','layer','block_idx','ssu_id','path_id','size_gib','block_count','enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms']
    assert arr.shape[1]==13 and len(arr)>0
    assert np.all(arr[:,8]<=arr[:,9]+1e-8) and np.all(arr[:,10]<=arr[:,11]+1e-8)
    validate_intervals(arr,9,10,4,6,40.)
    validate_intervals(arr,11,12,1,32,50.)
    ssd,ssd_gib=stage_bins(arr,9,10,1,32,40.)
    link,link_gib=stage_bins(arr,11,12,1,32,50.)
    disk,disk_gib=stage_bins(arr,9,10,4,6,40.)
    assert disk.max()<=40+1e-6 and link.max()<=50+1e-6
    assert math.isclose(ssd_gib.sum(),disk_gib.sum(),rel_tol=1e-10,abs_tol=1e-8)
    groups=defaultdict(list)
    for i in range(len(arr)):groups[(int(arr[i,0]),int(arr[i,2]))].append(i)
    requests={r['request_id']:r for r in man['requests']}
    batches=raw['summary']['microbatch_metrics'];by_request={b['member_request_ids'][0]:b for b in batches}
    lanes={n:sorted([b for b in batches if b['npu_id']==n],key=lambda b:b['admission_time_ms']) for n in range(32)}
    predecessor={}
    for lane in lanes.values():
        for i,b in enumerate(lane):
            rid=b['member_request_ids'][0]
            for k,l in enumerate(b['layer_metrics']):
                previous=b['layer_metrics'][k-1] if k else lane[i-1]['layer_metrics'][-1] if i else None
                predecessor[(rid,k)]=previous
    demand=np.zeros((32,N));stall_bins=np.zeros((32,N));compute=np.zeros(32)
    stalls=[[] for _ in range(32)];roles=[[] for _ in range(32)]
    for b in batches:
        n=b['npu_id'];rid=b['member_request_ids'][0];q=requests[rid]['load']
        if b['admission_time_ms']<RIGHT and b['completion_time_ms']>LEFT:
            add_interval(demand[n],b['admission_time_ms'],b['completion_time_ms'],q['per_layer_kv_gb']/(q['per_layer_us']/1e6))
            role='long' if q['role']=='long' else 'bridge' if q['nql']==4096 else 'short'
            roles[n].append([max(LEFT,b['admission_time_ms']),min(RIGHT,b['completion_time_ms']),role,rid])
        previous=b['admission_time_ms']
        for l in b['layer_metrics']:
            cs,ce=l['compute_start_ms'],l['compute_end_ms']
            a,z=max(LEFT,previous),min(RIGHT,cs)
            if z>a:stalls[n].append([a,z]);add_interval(stall_bins[n],a,z,1.)
            compute[n]+=max(0.,min(RIGHT,ce)-max(LEFT,cs));previous=ce
    stall_ms=stall_bins.sum(axis=1)*DT
    assert np.allclose(compute+stall_ms,RIGHT-LEFT,atol=1e-6)
    layer_rows=[];examples={}
    for key,indices in sorted(groups.items()):
        blocks=arr[indices];rid,k=key;b=by_request[rid];l=b['layer_metrics'][k];q=requests[rid]['load'];prev=predecessor[key]
        assert prev is not None,'Initialization is outside the selected warm layers.'
        release=l['io_start_time_ms'];deadline=prev['compute_end_ms'];ready=l['io_ready_time_ms'];budget=prev['compute_duration_ms']
        assert math.isclose(release,prev['compute_start_ms'],abs_tol=1e-7)
        assert math.isclose(deadline-release,budget,abs_tol=1e-7)
        assert math.isclose(l['compute_start_ms'],max(deadline,ready),abs_tol=1e-7)
        assert math.isclose(blocks[:,12].max(),ready,abs_tol=1e-8)
        placement=man['placements'][requests[rid]['placement_index']][0] if 'placement_index' in requests[rid] else requests[rid]['placement'][0]
        target_gib=math.fsum(v for _,v in placement)
        assert len(blocks)==len(placement) and math.isclose(blocks[:,6].sum(),target_gib,abs_tol=1e-10)
        target_mib=target_gib*1024;budget_rate=target_gib*1000/budget
        ssd_d=progress(blocks,deadline,9,10);link_d=progress(blocks,deadline,11,12)
        stall=max(0.,ready-deadline)
        assert math.isclose(stall,l['io_barrier_wait_ms'],abs_tol=1e-7)
        ssd_missing=max(0.,target_mib-ssd_d.sum());link_missing=max(0.,target_mib-link_d.sum())
        assert (stall>1e-7)==(link_missing>1e-5)
        last=blocks[np.argmax(blocks[:,12])]
        row=dict(request_id=rid,layer=k,npu=b['npu_id'],role='long' if q['role']=='long' else 'bridge' if q['nql']==4096 else 'short',
            profile=f'{q["seq_len_k"]}K/{q["nql"]}',release_ms=release,deadline_ms=deadline,ready_ms=ready,
            compute_start_ms=l['compute_start_ms'],payload_MiB=target_mib,budget_C_ms=budget,budget_gib_s=budget_rate,stall_ms=stall,
            own_D_over_C_gib_s=q['per_layer_kv_gb']/(q['per_layer_us']/1e6),
            display_cohort=release<RIGHT and ready>=LEFT,deadline_inside_display=LEFT<=deadline<RIGHT,
            ssd_before_deadline_MiB=float(ssd_d.sum()),link_before_deadline_MiB=float(link_d.sum()),
            ssd_deficit_at_deadline_MiB=ssd_missing,link_deficit_at_deadline_MiB=link_missing,
            ssd_mean_during_budget_gib_s=float(ssd_d.sum())/1024*1000/budget,
            link_mean_during_budget_gib_s=float(link_d.sum())/1024*1000/budget,
            ssd_mean_during_stall_gib_s=ssd_missing/1024*1000/stall if stall else None,
            link_mean_during_stall_gib_s=link_missing/1024*1000/stall if stall else None,
            per_ssu_payload_MiB=np.bincount(blocks[:,4].astype(int),weights=blocks[:,6]*1024,minlength=6).tolist(),
            per_ssu_ssd_at_deadline_MiB=np.bincount(blocks[:,4].astype(int),weights=ssd_d,minlength=6).tolist(),
            per_ssu_link_at_deadline_MiB=np.bincount(blocks[:,4].astype(int),weights=link_d,minlength=6).tolist(),
            last_block={name:int(v) if i in (0,1,2,3,4,5,7) else float(v) for i,(name,v) in enumerate(zip(columns,last))})
        layer_rows.append(row)
        for name,wanted in EXAMPLES.items():
            if key==wanted:examples[name]=dict(row,curves=cumulative_curves(blocks,release,deadline,ready))
    assert set(examples)==set(EXAMPLES)
    summaries=[]
    for n in range(32):
        layers=[r for r in layer_rows if r['npu']==n and r['deadline_inside_display']]
        late=[r for r in layers if r['stall_ms']>1e-7]
        high=[r for r in late if r['ssd_mean_during_stall_gib_s']>=r['budget_gib_s']]
        payload=sum(r['payload_MiB'] for r in layers)
        summaries.append(dict(npu=n,U_percent=100*compute[n]/(RIGHT-LEFT),stall_ms=float(stall_ms[n]),
            demand_mean_gib_s=float(demand[n].mean()),ssd_mean_gib_s=float(ssd[n].mean()),link_mean_gib_s=float(link[n].mean()),
            ssd_MiB=float(ssd_gib[n]*1024),link_MiB=float(link_gib[n]*1024),
            deadline_layer_count=len(layers),late_layer_count=len(late),
            stalled_layers_with_ssd_stall_mean_above_budget=len(high),
            deadline_payload_MiB=payload,ssd_deficit_at_deadlines_MiB=sum(r['ssd_deficit_at_deadline_MiB'] for r in layers),
            link_deficit_at_deadlines_MiB=sum(r['link_deficit_at_deadline_MiB'] for r in layers)))
    high=[r for r in layer_rows if r['deadline_inside_display'] and r['role']=='short' and r['stall_ms']>1e-7 and r['ssd_mean_during_stall_gib_s']>r['budget_gib_s']]
    if high:
        row=max(high,key=lambda r:r['stall_ms']);blocks=arr[groups[(row['request_id'],row['layer'])]]
        examples['high_bandwidth_during_stall']=dict(row,curves=cumulative_curves(blocks,row['release_ms'],row['deadline_ms'],row['ready_ms']))
    checks=dict(source_manifest_unchanged=True,source_result_unchanged=True,passive_replay_audit_passed=True,
        ssd_service_nonoverlap=True,npu_link_service_nonoverlap=True,per_block_service_durations=True,
        all_captured_layers_complete=True,all_target_deadline_stall_relations=True,
        all_window_bytes_binned_with_conservation=True,all_cards_compute_plus_stall_equals_window=True,
        each_ssu_binned_capacity_40=True,each_npu_link_binned_capacity_50=True)
    return dict(source=dict(manifest=str(manifest),manifest_sha256=sha(manifest),result=str(result),result_sha256=sha(result),
        trace=str(trace_path),trace_sha256=sha(trace_path),replay_audit_sha256=sha(directory/'audit.json')),
        bins=dict(demand_gib_s=demand.tolist(),ssd_gib_s=ssd.tolist(),link_gib_s=link.tolist(),stall_fraction=stall_bins.tolist()),
        ssu_service_bins_gib_s=disk.tolist(),stall_intervals_ms=stalls,request_intervals_ms=roles,
        npu_summary=summaries,window_U_percent=100*float(compute.sum())/(32*(RIGHT-LEFT)),
        layers=layer_rows,examples=examples,checks=checks)

def main():
    policies={p:analyze(p) for p in ('baseline','once')}
    out=dict(schema_version=1,window_ms=[LEFT,RIGHT],bin_ms=DT,bin_edges_ms=EDGES.tolist(),policies=policies,
        analysis_source_sha256=sha(__file__),definitions={
            'demand':'Current admitted request own per-layer D/C, exact bin average. Reference rate only; integrating it through stalls is not an actual byte-arrival workload.',
            'ssd':'Sum of actual per-block SSD service interval overlaps across all six SSDs divided by bin width. Physical service, not layer-read lifetime.',
            'link':'Actual per-block NPU link service interval overlaps divided by bin width; entering NPU, not SSD completion instants.',
            'deadline':'For target layer, predecessor compute end; cross-request L0 uses previous request final C and next actual payload. Budget accumulation stops at full payload.',
            'stall':'Exact compute-barrier intervals from original complete result, verified against target last-byte deadline; no pre-admission waiting.',
            'cohorts':'Fixed absolute display window contains different request IDs under the policies. Named examples pair same request ID/layer but have different absolute releases.',
            'diagnostic_scope':'Passive prefix replay of original complete 60-second run. Only [3.2,4.0)s service is plotted; long-window device U remains in the main report.'})
    write(HERE/'analysis.json',out)
    summary=[dict(policy=p,**row) for p,d in policies.items() for row in d['npu_summary']]
    with (HERE/'per_npu_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    flat=[dict(policy=p,**{k:v for k,v in r.items() if not isinstance(v,(dict,list))}) for p,d in policies.items() for r in d['layers']]
    with (HERE/'per_layer_deadlines.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with gzip.open(HERE/'per_npu_bandwidth_bins.csv.gz','wt',newline='') as f:
        fields=['policy','npu','start_ms','end_ms','demand_gib_s','ssd_gib_s','link_gib_s','stall_fraction']
        w=csv.writer(f);w.writerow(fields)
        for p,d in policies.items():
            for n in range(32):
                for k in range(N):w.writerow([p,n,EDGES[k],EDGES[k+1],*[d['bins'][v][n][k] for v in fields[4:]]])
    print(json.dumps({p:dict(U=d['window_U_percent'],captured_layers=len(d['layers']),checks_passed=all(d['checks'].values()),
        examples={k:{z:v[z] for z in ['request_id','layer','stall_ms','ssd_deficit_at_deadline_MiB','link_deficit_at_deadline_MiB','ssd_mean_during_stall_gib_s','budget_gib_s']} for k,v in d['examples'].items()}) for p,d in policies.items()},ensure_ascii=False))

if __name__=='__main__':main()
