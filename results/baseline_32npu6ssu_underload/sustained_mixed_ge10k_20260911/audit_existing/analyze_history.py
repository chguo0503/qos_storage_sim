#!/usr/bin/env python3
"""Read-only rewindowing of existing raw Baseline runs; no simulator imports."""
import csv,gzip,hashlib,json,math,statistics
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
F=STUDY/'raw_role_followup_20260909'
ROOT=STUDY.parents[1]
FIXED_WINDOWS=((2000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,12000),(2000,12000),(4000,12000))

def read(p):
    p=Path(p);x=p.read_bytes();return json.loads(gzip.decompress(x) if p.suffix=='.gz' else x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clipped(a,b,x,y):return max(0.0,min(b,y)-max(a,x))

def candidates():
    items=[];sources={}
    for p in (F/'followup_results.json',F/'mixed_rebinding/results.json'):
        d=read(p);sources[str(p)]=sha(p)
        for r in d['rows']:
            if r['strategy']=='baseline':items.append(dict(label=r['label'],family=r['spec_name'],mode=r['mode'],seed=r['seed'],
                path=r['result_path'],sha256=r['result_sha256'],inherited_capacity_pass=r['full_run_capacity_pass'],source=str(p)))
    p=STUDY/'raw_quartet_replacement/summary.json';d=read(p);sources[str(p)]=sha(p)
    for r in d['records']:
        if r['strategy']=='baseline' and r['status']=='complete':items.append(dict(label=r['label'],family=r['family'],mode=r['order'],seed=r['seed'],
            path=str(p.parent/r['path']),sha256=r['file_sha256'],inherited_capacity_pass=r['full_run_nominal_capacity_satisfied'],source=str(p)))
    p=F/'window_sensitivity/audit_repeated.json';d=read(p);sources[str(p)]=sha(p)
    for r in d['runs']:
        if r['strategy']=='baseline' and r['status']=='complete':items.append(dict(label=r['label'],family='raw176_repeat3',mode=r['mode'],seed=7,
            path=r['result_path'],sha256=r['result_sha256'],inherited_capacity_pass=r['scientific_full_run_nominal_capacity_passed'],source=str(p)))
    assert len({x['path'] for x in items})==len(items)
    return items,sources

def inspect(item):
    path=Path(item['path']);assert sha(path)==item['sha256'];raw=read(path);s=raw['summary']
    assert raw['strategy']=='baseline' and s['num_npu']==32 and s['n_layers']==8 and s['batch_size']==1
    assert all(s['invariants'].values())
    manifest=read(raw['manifest_path']);req={r['request_id']:r for r in manifest['requests']}
    assert len(req)==s['request_count'] and manifest['input_fingerprint']==raw['input_fingerprint']
    batches=s['microbatch_metrics'];lanes=[[] for _ in range(32)];layer_rows=[]
    for b in batches:
        assert b['batch_size']==len(b['member_request_ids'])==1
        rid=b['member_request_ids'][0];n=b['npu_id'];r=req[rid];assert r['npu_id']==n
        role=r['load']['role'];key=f"{r['load']['seq_len_k']}:{r['load']['nql']}"
        a,f=b['admission_time_ms'],b['completion_time_ms'];lanes[n].append((a,f,role,rid,key))
        prev=a
        for l in sorted(b['layer_metrics'],key=lambda l:l['layer']):
            cs,ce=l['compute_start_ms'],l['compute_end_ms'];assert prev<=cs+1e-8 and cs<ce<=f+1e-8
            layer_rows.append((n,role,key,cs,ce,prev,l['layer'],rid));prev=ce
    for lane in lanes:
        lane.sort();assert all(a[1]<=b[0]+1e-8 for a,b in zip(lane,lane[1:]))
    finishes=[max(r[1] for r in lane) for lane in lanes];first=min(finishes);last=max(finishes)
    def window(a,b):
        C=[[0.,0.] for _ in range(32)];A=[[0.,0.] for _ in range(32)];l0=0.;inner=0.
        for n,lane in enumerate(lanes):
            for x,y,role,rid,key in lane:A[n][int(role=='long')]+=clipped(a,b,x,y)
        for n,role,key,cs,ce,prev,l,rid in layer_rows:
            C[n][int(role=='long')]+=clipped(a,b,cs,ce)
            if l==0:l0+=clipped(a,b,prev,cs)
            else:inner+=clipped(a,b,prev,cs)
        c=math.fsum(v for row in C for v in row);active=math.fsum(v for row in A for v in row);den=32*(b-a)
        assert math.isclose(c+l0+inner,active,abs_tol=1e-6)
        by_role={role:dict(C_ms=math.fsum(row[i] for row in C),active_ms=math.fsum(row[i] for row in A)) for i,role in enumerate(('short','long'))}
        for v in by_role.values():
            v['conditional_U_percent']=100*v['C_ms']/v['active_ms'] if v['active_ms'] else None
            v['mean_cards']=v['active_ms']/(b-a)
        return dict(start_ms=a,end_ms=b,duration_ms=b-a,U_percent=100*c/den,active_percent=100*active/den,
            all_32_active=all(math.isclose(sum(row),b-a,abs_tol=1e-6) for row in A),
            mixed_cards=sum(all(v>1e-8 for v in row) for row in C),min_role_C_ms=min(v for row in C for v in row),
            l0_stall_ms=l0,internal_stall_ms=inner,tail_idle_ms=math.fsum(max(0.,b-max(a,t)) for t in finishes),
            by_role=by_role)
    windows=[window(a,b) for a,b in FIXED_WINDOWS]
    cutoff=math.floor((first-1e-6)/50)*50
    for a in (2000,4000):
        if cutoff>a:windows.append(dict(**window(a,cutoff),posthoc_common_active_endpoint=True))
    sliding=[window(a,a+2000) for a in range(2000,max(2000,int(first)-1999),250) if a+2000<first]
    after4=math.fsum(clipped(4000,last,prev,cs) for n,role,key,cs,ce,prev,l,rid in layer_rows)
    return dict(**item,manifest_path=raw['manifest_path'],manifest_sha256=sha(raw['manifest_path']),
        request_count=len(req),minimum_total_tokens=min(r['load']['seq_len_k']*1024 for r in req.values()),
        all_direct_data=all(not r['load'].get('constructed_profile',True) and r['load'].get('profile_construction')=={'method':'direct_data_row'} for r in req.values()),
        first_card_done_ms=first,makespan_ms=last,all_run_stall_after4_ms=after4,
        windows=windows,sliding_2s=sliding,
        qualifying_8s_or_more_windows=[w for w in windows if w['duration_ms']>=8000 and w['all_32_active'] and w['mixed_cards']==32 and w['U_percent']<=90.5 and item['inherited_capacity_pass']])

def main():
    HERE.mkdir(parents=True,exist_ok=True);items,sources=candidates();sources[str(Path(__file__))]=sha(__file__)
    runs=[inspect(item) for item in items]
    summary=[]
    for r in runs:
        main=next(w for w in r['windows'] if (w['start_ms'],w['end_ms'])==(2000,4000))
        tail=next((w for w in r['windows'] if w.get('posthoc_common_active_endpoint') and w['start_ms']==4000),None)
        later=[w for w in r['sliding_2s'] if w['start_ms']>=4000 and w['all_32_active']]
        summary.append(dict(label=r['label'],family=r['family'],mode=r['mode'],seed=r['seed'],
            request_count=r['request_count'],first_card_done_ms=r['first_card_done_ms'],makespan_ms=r['makespan_ms'],
            U_2_4=main['U_percent'],mixed_cards_2_4=main['mixed_cards'],
            U_4_to_first_drain=tail['U_percent'] if tail else None,
            active_tail_duration_ms=tail['duration_ms'] if tail else 0,
            tail_mixed_cards=tail['mixed_cards'] if tail else None,
            worst_later_full_active_2s_U=min((w['U_percent'] for w in later),default=None),
            all_run_stall_after4_ms=r['all_run_stall_after4_ms'],
            direct_raw=r['all_direct_data'],full_run_capacity_pass=r['inherited_capacity_pass'],
            qualifying_8s_or_more_windows=len(r['qualifying_8s_or_more_windows']),path=r['path'],sha256=r['sha256']))
    assert all(sha(p)==v for p,v in sources.items())
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,runs=runs,
        scope='Existing Baseline raw followup, raw mixed rebinding, raw quartet replacement, and three completed repeat3 runs only; not all repository simulations.',
        definitions=dict(U='Actual clipped compute/(32*window duration), no denominator normalization after drain',
            capacity='Inherited prior exact-time scan, bound by result SHA; this timing search does not re-run capacity scanner',
            posthoc='Endpoints based on observed earliest queue exhaustion and 250ms sliding grid are exploratory, not preregistered validation',
            qualifier='Descriptive screen: at least8s after2000ms, all32active, all32positive short+long C, U<=90.5%, inherited capacity passed',
            raw_rule='direct_data_row plus constructed_profile false in each frozen request load; C/V authenticity is inherited from prior audit'),
        run_count=len(runs),qualifying_count=sum(len(r['qualifying_8s_or_more_windows']) for r in runs),summary=summary)
    (HERE/'history.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'history.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(dict(runs=len(runs),qualifying_count=output['qualifying_count'],
        longest_low_examples=[s for s in summary if s['U_2_4']<93 and s['direct_raw']],),ensure_ascii=False))

if __name__=='__main__':main()
