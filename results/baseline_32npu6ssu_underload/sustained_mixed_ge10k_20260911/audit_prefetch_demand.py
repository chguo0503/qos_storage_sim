#!/usr/bin/env python3
"""Additional deadline-budget demand proxy; no simulation, no acceptance changes.

Replace the current final layer's nominal D/C by actual NEXT-request D/C_current.
Two variants: computation budget interval only, or extend through exposed wait.
This is not actual throughput or a proof of all read deadlines being feasible.
"""
from pathlib import Path
from collections import defaultdict
import argparse,gzip,hashlib,json,math

def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(manifest_path,result_path,start,end):
    m,r=read(manifest_path),read(result_path)
    assert m['input_fingerprint']==r['input_fingerprint']
    requests={v['request_id']:v for v in m['requests']};lanes=[[] for _ in range(32)]
    for b in r['summary']['microbatch_metrics']:lanes[b['npu_id']].append(b)
    intervals=[];handoffs=[]
    def vector(req,layer):
        p=m['placements'][req['placement_index']];v=p[layer%len(p)]
        return [math.fsum(x for s,x in v if s==d) for d in range(6)]
    for n,lane in enumerate(lanes):
        lane.sort(key=lambda b:b['admission_time_ms'])
        for j,b in enumerate(lane):
            source=requests[b['member_request_ids'][0]];ll=b['layer_metrics']
            for k,layer in enumerate(ll):
                if k<7:target=source;target_layer=ll[k+1]
                elif j+1<len(lane):target=requests[lane[j+1]['member_request_ids'][0]];target_layer=lane[j+1]['layer_metrics'][0]
                else:continue
                cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
                assert math.isclose(target_layer['io_start_time_ms'],cs,abs_tol=1e-6)
                rates=[v/(source['load']['per_layer_us']/1e6) for v in vector(target,target_layer['layer'])]
                row=dict(npu=n,source=source['request_id'],target=target['request_id'],source_profile=[source['load']['seq_len_k'],source['load']['nql']],
                    target_profile=[target['load']['seq_len_k'],target['load']['nql']],layer=k,start_ms=cs,budget_end_ms=ce,
                    next_compute_ms=target_layer['compute_start_ms'],rates=rates,cross_request=k==7)
                intervals.append(row)
                if k==7 and start<=cs<end:handoffs.append(row)
    variants={}
    for name,bound in [('compute_budget_only','budget_end_ms'),('through_exposed_wait','next_compute_ms')]:
        events=defaultdict(lambda:[[],[]])
        for i,v in enumerate(intervals):
            a=max(start,v['start_ms']);z=min(end,v[bound])
            if z>a:events[a][1].append(i);events[z][0].append(i)
        times=sorted(events);active=set();peak=[0.]*6;over=[0.]*6;witness=[None]*6;any_over=0.
        for i,t in enumerate(times[:-1]):
            active.difference_update(events[t][0]);active.update(events[t][1]);z=times[i+1]
            rates=[math.fsum(intervals[x]['rates'][s] for x in active) for s in range(6)]
            for s,v in enumerate(rates):
                if v>peak[s]:
                    peak[s]=v;witness[s]=dict(start_ms=t,end_ms=z,rate=v,active=[intervals[x] for x in sorted(active)])
                if v>40:over[s]+=z-t
            if max(rates)>40:any_over+=z-t
        variants[name]=dict(per_ssu_peak_gib_s=peak,max_ssu_gib_s=max(peak),per_ssu_over40_ms=over,
            any_ssu_over40_ms=any_over,peak_witnesses=witness)
    return dict(manifest=str(Path(manifest_path).resolve()),result=str(Path(result_path).resolve()),
        manifest_sha256=sha(manifest_path),result_sha256=sha(result_path),script_sha256=sha(__file__),window_ms=[start,end],
        variants=variants,warm_handoff_count=len(handoffs),
        definitions=dict(rate='Real next-layer per-SSD bytes / current-layer original compute budget; at handoff, next request L0 replaces current request nominal bytes',
            compute_budget_only='Rate active on [current compute start,current compute end)',
            through_exposed_wait='Rate active until next layer begins computing, including exposed wait; next request admission may be later than prefetch completion',
            exclusions='No prior compute budget exists for each NPU initial L0; final request last layer has no successor. Fixed warm start avoids initial stage.',
            status='Additional descriptive proxy; not actual SSD throughput, a service schedule, all-deadline feasibility proof, or replacement for agreed current-request D/C acceptance.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--result',required=True);p.add_argument('--out',required=True)
    p.add_argument('--start-ms',type=float,default=2000);p.add_argument('--end-ms',type=float,default=12000);a=p.parse_args()
    r=audit(a.manifest,a.result,a.start_ms,a.end_ms);Path(a.out).write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='peak_witnesses'} for k,v in r['variants'].items()},indent=2))
if __name__=='__main__':main()
