#!/usr/bin/env python3
"""Independently audit a completed sustained raw-input result, without simulation.

python audit_sustained.py --manifest INPUT.json.gz --result RESULT.json.gz
                         --out AUDIT.json [--end-ms 30000]
"""
import argparse,ast,gzip,hashlib,json,math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def read(p):
    p=Path(p);b=p.read_bytes();return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7)
def overlap(a,b,start,end):return max(0.,min(b,end)-max(a,start))

def manifest_info(path,table):
    d=read(path);rows=d['requests'];N=32;S=6
    ps=[tuple(tuple((int(s),float(v)) for s,v in layer) for layer in p) for p in d['placements']]
    checks=dict(unique_request_ids=True,raw_C_exact=True,raw_D_equal=True,total_input_ge10K=True,
        role_is_short_or_long=True,placement_dimensions=True,all_layer_disk_vectors_identical=True,
        physical_D_matches_raw=True,load_identifiers_match=True,all_arrival_zero=True,category_consistent=True)
    fp=hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0');inputs={};maxima=[[0.]*S for _ in range(N)]
    counts=Counter();profiles={}
    for r in sorted(rows,key=lambda x:x['request_id']):
        rid,n,load=r['request_id'],r['npu_id'],r['load'];p=ps[r['placement_index']]
        checks['unique_request_ids'] &= rid not in inputs
        assert 0<=n<N and len(p) in (1,8)
        key=(int(load['seq_len_k']),int(load['nql']));assert key in table,('missing raw data row',key)
        c=float(load['per_layer_us'])/1000.;ref=table[key];role=load['role']
        checks['raw_C_exact'] &= load['per_layer_us']==ref[1]
        checks['raw_D_equal'] &= abs(load['per_layer_kv_gb']-ref[3])<1e-13
        checks['total_input_ge10K'] &= key[0]*1024>=10*1024
        checks['role_is_short_or_long'] &= role in ('short','long')
        checks['load_identifiers_match'] &= load.get('request_id',rid)==rid and load.get('npu_id',n)==n
        checks['all_arrival_zero'] &= r['arrival_time_ms']==0
        cat=('S' if key[0]<=80 else 'L')+('S' if key[1]<512 else 'L')
        checks['category_consistent'] &= load.get('category',cat)==cat
        vectors=[]
        for layer in p:
            checks['placement_dimensions'] &= all(0<=s<S and v>=0 for s,v in layer)
            checks['physical_D_matches_raw'] &= abs(math.fsum(v for s,v in layer)-ref[3])<1e-13
            vector=[math.fsum(v for s,v in layer if s==disk) for disk in range(S)];vectors.append(vector)
            maxima[n]=[max(m,1000*v/c) for m,v in zip(maxima[n],vector)]
        identical=all(all(close(a,b) for a,b in zip(vector,vectors[0])) for vector in vectors)
        checks['all_layer_disk_vectors_identical'] &= identical
        rates=[math.fsum(v[s] for v in vectors)/len(vectors)/c*1000 for s in range(S)]
        fp.update(repr((rid,n,float(r['arrival_time_ms']),load['category'],load['per_layer_us'],p)).encode())
        info=dict(npu=n,role=role,profile=f'{key[0]}:{key[1]}',seq_len_k=key[0],nql=key[1],category=cat,C_ms=c,own_C_ms=8*c,
            arrival_ms=r['arrival_time_ms'],rates=rates,link_rate=math.fsum(rates),vectors_identical=identical)
        inputs[rid]=info;counts[(n,role)]+=1
        profiles[str(key)]=dict(seq_len_k=key[0],total_tokens=key[0]*1024,nql=key[1],role=role,category=cat,
            raw_C_ms=c,raw_D_gib=ref[3],actual_D_MiB=math.fsum(vectors[0])*1024)
    checks['fp_recomputed']=fp.hexdigest()==d['input_fingerprint']
    proof=[math.fsum(v[s] for v in maxima) for s in range(S)]
    return d,inputs,dict(checks=checks,passed=all(checks.values()),fingerprint=fp.hexdigest(),
        request_count=len(inputs),profiles=list(profiles.values()),static_per_ssu_upper_bound_gib_s=proof,
        static_max_gib_s=max(proof),static_sufficient_certificate_pass=max(proof)<40,
        population_by_npu=[dict(npu=n,short=counts[(n,'short')],long=counts[(n,'long')],
            pure_C_ms=math.fsum(r['own_C_ms'] for r in inputs.values() if r['npu']==n)) for n in range(N)])

def scan_nominal(rows,inputs):
    events=defaultdict(lambda:dict(add=[],remove=[]))
    for r in rows:
        events[r['admission_time_ms']]['add'].append(r['request_id'])
        events[r['completion_time_ms']]['remove'].append(r['request_id'])
    times=sorted(events);active={};peaks=[0.]*6;over=[0.]*6;equal=[0.]*6;witness=[None]*6
    link_peak=0.;any_over=0.;link_over=0.;max_long=0
    for i,t in enumerate(times):
        for rid in events[t]['remove']:
            n=inputs[rid]['npu'];assert active.pop(n)==rid
        for rid in events[t]['add']:
            n=inputs[rid]['npu'];assert n not in active;active[n]=rid
        if i+1==len(times):break
        end=times[i+1];dt=end-t;assert dt>0
        rates=[math.fsum(inputs[r]['rates'][s] for r in active.values()) for s in range(6)]
        links=max((inputs[r]['link_rate'] for r in active.values()),default=0.)
        max_long=max(max_long,sum(inputs[r]['role']=='long' for r in active.values()))
        for s,v in enumerate(rates):
            if v>peaks[s]:
                peaks[s]=v;witness[s]=dict(start_ms=t,end_ms=end,value_gib_s=v,active_requests={str(n):r for n,r in sorted(active.items())})
            if v>40:over[s]+=dt
            if v==40:equal[s]+=dt
        if max(rates)>40:any_over+=dt
        link_peak=max(link_peak,links)
        if links>50:link_over+=dt
    assert not active
    return dict(event_times=len(times),start_ms=times[0],end_ms=times[-1],per_ssu_peak_gib_s=peaks,max_ssu_gib_s=max(peaks),
        per_ssu_over40_ms=over,per_ssu_equal40_ms=equal,any_ssu_over40_ms=any_over,
        max_link_gib_s=link_peak,any_link_over50_ms=link_over,maximum_long_active_cards=max_long,
        strict_underload=max(peaks)<40 and link_peak<50,peak_witnesses=witness,
        nominal_definition='Current admitted request actual per-layer V_s/C over [admission,completion), no next-L0 addition; ties fully batched; fresh sums over every positive event interval',
        layer_vectors='Mean across physical layer vectors; all-layer-vector equality is separately required for a passing primary audit.')

def timing(raw,inputs):
    s=raw['summary'];rows=s['request_metrics'];byid={r['request_id']:r for r in rows}
    checks=dict(dimensions=(s['num_npu'],s['num_ssu'],s['n_layers'],s['batch_size'])==(32,6,8,1),
        complete_population=len(rows)==len(byid)==len(inputs) and set(byid)==set(inputs),
        singleton_batches=True,same_request_npu=True,C_matches_raw=True,timestamps_ordered=True,full_C_account=True)
    intervals=[[] for _ in range(32)];layers=[[] for _ in range(32)];seen=[]
    for b in s['microbatch_metrics']:
        checks['singleton_batches'] &= b['batch_size']==len(b['member_request_ids'])==1
        rid=b['member_request_ids'][0];r=inputs[rid];n=b['npu_id'];seen.append(rid)
        a,f=b['admission_time_ms'],b['completion_time_ms'];q=byid[rid]
        checks['same_request_npu'] &= n==r['npu']==q['npu_id'] and close(a,q['admission_time_ms']) and close(f,q['completion_time_ms']) and close(q['own_compute_ms'],r['own_C_ms'])
        checks['timestamps_ordered'] &= math.isfinite(a) and math.isfinite(f) and f>=a>=q['arrival_time_ms']==r['arrival_ms']
        intervals[n].append((a,f,rid,r['role'],r['profile']));previous=a;total=[]
        ll=sorted(b['layer_metrics'],key=lambda x:x['layer']);checks['full_C_account'] &= [x['layer'] for x in ll]==list(range(8))
        for l in ll:
            cs,ce=l['compute_start_ms'],l['compute_end_ms'];checks['timestamps_ordered'] &= previous<=cs+1e-8 and cs<ce<=f+1e-8
            checks['C_matches_raw'] &= close(ce-cs,r['C_ms']) and close(cs-previous,l['io_barrier_wait_ms'])
            layers[n].append((cs,ce,previous,l['layer'],r['role'],r['profile'],rid));total.append(ce-cs);previous=ce
        checks['full_C_account'] &= close(previous,f) and close(math.fsum(total),r['own_C_ms'])
    checks['complete_batch_population']=len(seen)==len(set(seen))==len(inputs) and set(seen)==set(inputs)
    for lane in intervals:
        lane.sort();checks['timestamps_ordered'] &= bool(lane) and all(x[1]<=y[0]+1e-8 for x,y in zip(lane,lane[1:]))
    return rows,intervals,layers,checks

def slo(rows,inputs):
    good=sum(r['completion_time_ms']-r['admission_time_ms']<=1.5*inputs[r['request_id']]['own_C_ms']+1e-9 for r in rows)
    agood=sum(r['completion_time_ms']-r['arrival_time_ms']<=1.5*inputs[r['request_id']]['own_C_ms']+1e-9 for r in rows)
    return dict(count=len(rows),passed=good,rate_percent=100*good/len(rows) if rows else None,
        arrival_clock_passed=agood,arrival_clock_rate_percent=100*agood/len(rows) if rows else None,
        request_ids=sorted(r['request_id'] for r in rows))

def window(rows,inputs,intervals,layers,start,end,min_switches,min_share):
    cards=[]
    for n,(lane,ll) in enumerate(zip(intervals,layers)):
        c=math.fsum(overlap(a,b,start,end) for a,b,prev,l,role,p,rid in ll)
        active=math.fsum(overlap(a,b,start,end) for a,b,rid,role,p in lane)
        l0=math.fsum(overlap(prev,a,start,end) for a,b,prev,l,role,p,rid in ll if l==0)
        inner=math.fsum(overlap(prev,a,start,end) for a,b,prev,l,role,p,rid in ll if l!=0)
        assert close(c+l0+inner,active)
        switches=[dict(time_ms=y[0],from_role=x[3],to_role=y[3],request_id=y[2]) for x,y in zip(lane,lane[1:]) if start<=y[0]<end and x[3]!=y[3]]
        roles={}
        for role in ('short','long'):
            cc=math.fsum(overlap(a,b,start,end) for a,b,prev,l,rr,p,rid in ll if rr==role)
            aa=math.fsum(overlap(a,b,start,end) for a,b,rid,rr,p in lane if rr==role)
            roles[role]=dict(compute_ms=cc,active_ms=aa,pure_compute_share=cc/c if c else None,
                conditional_U_percent=100*cc/aa if aa else None,
                admissions_count=sum(start<=a<end and rr==role for a,b,rid,rr,p in lane),
                overlapping_requests_count=sum(max(a,start)<min(b,end) and rr==role for a,b,rid,rr,p in lane))
        fast=math.fsum(overlap(a,b,start,end) for a,b,prev,l,rr,p,rid in ll if rr=='short' and inputs[rid]['nql']==1024)
        bridge=math.fsum(overlap(a,b,start,end) for a,b,prev,l,rr,p,rid in ll if rr=='short' and inputs[rid]['nql']==4096)
        cards.append(dict(npu=n,compute_ms=c,active_ms=active,device_U_percent=100*c/(end-start),
            idle_ms=end-start-active,tail_idle_ms=max(0.,end-max(start,lane[-1][1])),
            l0_stall_ms=l0,internal_stall_ms=inner,role_switch_count=len(switches),role_switches=switches,roles=roles,
            fast_short_C_ms=fast,fast_short_fraction=fast/c if c else None,
            bridge_C_ms=bridge,bridge_fraction=bridge/c if c else None,long_fraction=roles['long']['pure_compute_share']))
    c=math.fsum(x['compute_ms'] for x in cards);a=math.fsum(x['active_ms'] for x in cards);den=32*(end-start)
    admitted=[r for r in rows if start<=r['admission_time_ms']<end]
    # Preserve exact profiles: a relatively short role may include a bridge
    # with nearly Long compute time. Never merge its contribution invisibly.
    by_profile={}
    for p in sorted({r['profile'] for r in inputs.values()},key=lambda p:tuple(map(int,p.split(':')))):
        ref=next(r for r in inputs.values() if r['profile']==p)
        cc=math.fsum(overlap(a,b,start,end) for ll in layers for a,b,prev,l,rr,pp,rid in ll if pp==p)
        aa=math.fsum(overlap(a,b,start,end) for lane in intervals for a,b,rid,rr,pp in lane if pp==p)
        l0=math.fsum(overlap(prev,a,start,end) for ll in layers for a,b,prev,l,rr,pp,rid in ll if pp==p and l==0)
        inner=math.fsum(overlap(prev,a,start,end) for ll in layers for a,b,prev,l,rr,pp,rid in ll if pp==p and l!=0)
        role_C=math.fsum(x['roles'][ref['role']]['compute_ms'] for x in cards)
        assert close(cc+l0+inner,aa)
        cohort=[r for r in admitted if inputs[r['request_id']]['profile']==p]
        by_profile[p]=dict(role=ref['role'],category=ref['category'],seq_len_k=ref['seq_len_k'],nql=ref['nql'],
            per_layer_C_ms=ref['C_ms'],per_layer_actual_D_MiB=ref['link_rate']*ref['C_ms']/1000*1024,
            compute_ms=cc,active_ms=aa,mean_active_cards=aa/(end-start),L0_stall_ms=l0,internal_stall_ms=inner,
            conditional_U_percent=100*cc/aa if aa else None,fleet_compute_share=cc/c if c else None,
            within_role_compute_share=cc/role_C if role_C else None,device_U_contribution_pp=100*cc/den,
            warm_admission_SLO=slo(cohort,inputs),
            overlapping_requests_count=sum(max(a,start)<min(b,end) and pp==p for lane in intervals for a,b,rid,rr,pp in lane),
            per_npu_compute_ms=[math.fsum(overlap(a,b,start,end) for a,b,prev,l,rr,pp,rid in ll if pp==p) for ll in layers])
    assert close(math.fsum(z['compute_ms'] for z in by_profile.values()),c)
    by_role_nql={}
    for role,nql in sorted({(z['role'],z['nql']) for z in by_profile.values()}):
        selected={p:z for p,z in by_profile.items() if (z['role'],z['nql'])==(role,nql)}
        cc=math.fsum(z['compute_ms'] for z in selected.values());aa=math.fsum(z['active_ms'] for z in selected.values())
        role_C=math.fsum(x['roles'][role]['compute_ms'] for x in cards)
        by_role_nql[f'{role}:nql{nql}']=dict(role=role,nql=nql,profiles=list(selected),compute_ms=cc,active_ms=aa,
            fleet_compute_share=cc/c if c else None,within_role_compute_share=cc/role_C if role_C else None,
            conditional_U_percent=100*cc/aa if aa else None,device_U_contribution_pp=100*cc/den,
            L0_stall_ms=math.fsum(z['L0_stall_ms'] for z in selected.values()),internal_stall_ms=math.fsum(z['internal_stall_ms'] for z in selected.values()))
    w=dict(start_ms=start,end_ms=end,duration_ms=end-start,U_percent=100*c/den,compute_ms=c,active_ms=a,idle_ms=den-a,
        l0_stall_ms=math.fsum(x['l0_stall_ms'] for x in cards),internal_stall_ms=math.fsum(x['internal_stall_ms'] for x in cards),
        tail_idle_ms=math.fsum(x['tail_idle_ms'] for x in cards),all_32_active=all(close(x['active_ms'],end-start) for x in cards),
        mixed_card_count=sum(all(x['roles'][role]['compute_ms']>0 for role in ('short','long')) for x in cards),
        min_role_C_ms=min(x['roles'][role]['compute_ms'] for x in cards for role in ('short','long')),
        minimum_per_card_role_compute_share=min(x['roles'][role]['pure_compute_share'] or 0 for x in cards for role in ('short','long')),
        minimum_per_card_role_switches=min(x['role_switch_count'] for x in cards),
        min_fast_short_C_ms=min(x['fast_short_C_ms'] for x in cards),
        min_fast_short_fraction=min(x['fast_short_fraction'] or 0 for x in cards),
        cards_fast_and_long_ge5percent=sum((x['fast_short_fraction'] or 0)>=.05 and (x['long_fraction'] or 0)>=.05 for x in cards),
        cards_at_least_min_switches=sum(x['role_switch_count']>=min_switches for x in cards),
        cards_both_roles_at_least_min_share=sum(all((x['roles'][role]['pure_compute_share'] or 0)>=min_share for role in ('short','long')) for x in cards),
        warm_admission_SLO=slo(admitted,inputs),completion_after_window_count=sum(r['completion_time_ms']>end for r in admitted),
        warm_SLO_by_role={role:slo([r for r in admitted if inputs[r['request_id']]['role']==role],inputs) for role in ('short','long')},
        warm_SLO_by_category={cat:slo([r for r in admitted if inputs[r['request_id']]['category']==cat],inputs) for cat in ('SS','SL','LS','LL')},
        by_profile=by_profile,by_role_and_nql=by_role_nql,
        per_npu=cards)
    w['sustained_mixed_initial_screen']=w['all_32_active'] and w['mixed_card_count']==32 and w['cards_at_least_min_switches']==32 and w['cards_both_roles_at_least_min_share']==32
    return w

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--manifest',required=True);ap.add_argument('--result',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--end-ms',type=float,default=12000);ap.add_argument('--min-switches',type=int,default=4);ap.add_argument('--min-role-share',type=float,default=.05)
    args=ap.parse_args();assert args.end_ms>4000 and args.min_switches>=0 and 0<=args.min_role_share<=.5
    raw=read(args.result);table=ast.literal_eval((ROOT/'data').read_text());m,inputs,im=manifest_info(args.manifest,table)
    source_paths={str(Path(__file__).resolve()),str((ROOT/'data').resolve()),str(Path(args.manifest).resolve()),str(Path(args.result).resolve())}
    checks=dict(input_fingerprint=raw['input_fingerprint']==raw['summary']['input_fingerprint']==im['fingerprint'],
        core_invariants=bool(raw['summary']['invariants']) and all(raw['summary']['invariants'].values()),
        core_sources_present=bool(raw['core_and_policy_sha256']),core_sources=True,
        execution_placement=raw['execution_placement_fingerprint']==raw['input_placement_fingerprint'],
        fixed_binding=raw['policy_config']['assignment']=='fixed' and raw['assignment_log']==[])
    for name,digest in raw['core_and_policy_sha256'].items():
        p=(ROOT/name).resolve();source_paths.add(str(p));checks['core_sources'] &= p.exists() and sha(p)==digest
    checks['stress_source']=sha(ROOT/'run_baseline_npu32_stress.py')==raw['stress_runner_sha256'];source_paths.add(str(ROOT/'run_baseline_npu32_stress.py'))
    if 'source_data_sha256' in m['metadata']:checks['metadata_data_sha']=m['metadata']['source_data_sha256']==sha(ROOT/'data')
    command_path=Path(args.result).parent/'command.json'
    if command_path.exists():
        cmd=read(command_path);source_paths.add(str(command_path));checks['command_complete']=cmd.get('status')=='complete' and cmd.get('returncode',0)==0
        if 'input_sha256' in cmd:checks['command_input_sha']=cmd['input_sha256']==sha(args.manifest)
        if 'manifest_sha256' in cmd:checks['command_manifest_sha']=cmd['manifest_sha256']==sha(args.manifest)
        if 'output_sha256' in cmd:checks['command_output_sha']=cmd['output_sha256']==sha(args.result)
    adapter=raw.get('adapter_statistics',{})
    checks['no_runtime_migration']=adapter.get('assignment_count')==0
    checks['policy_literal_and_5ms']=raw['strategy']==adapter.get('strategy') and raw['collector_interval_ms']==adapter.get('collector_interval_ms')==5
    checks['zero_modeled_control_cost']=adapter.get('modeled_control_cpu_latency_ms')==adapter.get('modeled_control_communication_latency_ms')==0
    source_before={p:sha(p) for p in sorted(source_paths)}
    rows,intervals,layers,timechecks=timing(raw,inputs);checks.update(timechecks)
    ends=args.end_ms;spans=[(2000.,ends),(4000.,ends)]+[(float(s),min(float(s+2000),ends)) for s in range(2000,math.ceil(ends),2000) if s<ends]
    spans=list(dict.fromkeys(spans));windows=[window(rows,inputs,intervals,layers,a,b,args.min_switches,args.min_role_share) for a,b in spans]
    scan=scan_nominal(rows,inputs)
    first=[min(r['completion_time_ms'] for r in rows if r['npu_id']==n) for n in range(32)]
    fourth=[sorted(r['completion_time_ms'] for r in rows if r['npu_id']==n)[3] for n in range(32)]
    checks['sources_unchanged']=all(sha(p)==digest for p,digest in source_before.items())
    technical=im['passed'] and all(checks.values())
    primary=windows[0];valid=technical and scan['strict_underload'] and primary['sustained_mixed_initial_screen']
    out=dict(schema_version=3,created_utc=datetime.now(timezone.utc).isoformat(),manifest=str(Path(args.manifest).resolve()),result=str(Path(args.result).resolve()),
        strategy=raw['strategy'],submit_seed=raw['submit_seed'],technical_audit_passed=technical,primary_conditions_passed=valid,
        primary_U_around89_descriptive=abs(primary['U_percent']-89)<=1,
        checks=checks,input_audit=im,nominal_full_run=scan,windows=windows,primary_window=primary,
        first_card_final_completion_ms=min(max(r['completion_time_ms'] for r in rows if r['npu_id']==n) for n in range(32)),
        makespan_ms=max(r['completion_time_ms'] for r in rows),fourth_request_by_1500=all(t<=1500 for t in fourth),maximum_fourth_completion_ms=max(fourth),
        first_request_completion_ms_by_npu=first,maximum_first_request_completion_ms=max(first),initial_request_finished_before_primary_window=all(t<=primary['start_ms'] for t in first),
        source_sha256=source_before,
        definitions=dict(U='Actual clipped layer compute/(32*window_duration), always retaining all32 cards in denominator',
            role_share='Clipped role compute / all clipped compute on the same NPU; no active-time or fleet-time weighting in this per-card fraction',
            profile_share='Each exact seqK:NQL profile retains its own C, D, fleet-compute share, within-role-compute share and SLO; role+NQL groups separately expose a longer-compute bridge inside the short role.',
            fast_short='role=short and NQL=1024; bridge is role=short and NQL=4096. Per-card fractions use actual clipped card compute as denominator; secondary cards_fast_and_long_ge5percent does not alter the original role-based primary acceptance.',
            role_switch='Different roles on consecutive requests on a card, with new admission time in the half-open window; same-role request changes do not count',
            initial_screen=dict(minimum_role_switches_per_card=args.min_switches,minimum_each_role_compute_share=args.min_role_share,applies_to_primary_window=True),
            SLO='admission in [start,end), follow complete completion; completion-admission <=1.5*8*original data-row per-layer C; not arrival-clock TTFT',
            capacity='Current admitted request actual per-layer V_s/C only; full-run event scan; next-request L0 not counted additionally; not SSD service throughput',
            steady_state='Meeting these finite-window filters is not a proof of stationary or infinite-horizon behavior.',
            metadata='Scientific input checks derive from actual request fields, placement bytes and data rows, not metadata pass flags'))
    target=Path(args.out);target=target if target.suffix=='.json' else target/'audit.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(out=str(target),technical_passed=technical,conditions_passed=valid,U=primary['U_percent'],
        minimum_switches=primary['minimum_per_card_role_switches'],minimum_role_share=primary['minimum_per_card_role_compute_share'],
        all_active=primary['all_32_active'],max_disk=scan['max_ssu_gib_s'],over_ms=scan['any_ssu_over40_ms']),ensure_ascii=False))
    # Scientific failures remain readable; a nonzero exit denotes audit failure,
    # not merely an unsuccessful workload candidate.
    if not technical:raise SystemExit(1)

if __name__=='__main__':main()
