#!/usr/bin/env python3
"""Read-only simulation-result audit; writes only this pilot's analysis files."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import ast
import csv
import gzip
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(a,z,left,right):return max(0.0,min(z,right)-max(a,left))
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-7)


def window(ledger,left,right):
    cards=[]
    for n in range(32):
        batches=[x for x in ledger if x['npu']==n]
        c={r:math.fsum(overlap(a,z,left,right) for x in batches if x['role']==r for a,z in x['compute']) for r in ['long','short']}
        s={r:math.fsum(overlap(a,z,left,right) for x in batches if x['role']==r for a,z in x['stall']) for r in ['long','short']}
        a={r:math.fsum(overlap(x['admission'],x['completion'],left,right) for x in batches if x['role']==r) for r in ['long','short']}
        assert all(close(c[r]+s[r],a[r]) for r in c)
        active=math.fsum(a.values());compute=math.fsum(c.values());stall=math.fsum(s.values())
        cards.append({'npu':n,'compute_ms':compute,'stall_ms':stall,'active_ms':active,
                      'idle_ms':max(0.0,right-left-active),'role_compute_ms':c,'role_active_ms':a,
                      'role_stall_ms':s,'all_active':close(active,right-left),
                      'both_roles_positive_compute':all(z>1e-7 for z in c.values())})
    warm=[x for x in ledger if left<=x['admission']<right]
    passed=sum(x['completion']-x['admission']<=1.5*8*x['C']+1e-8 for x in warm)
    compute=math.fsum(x['compute_ms'] for x in cards);stall=math.fsum(x['stall_ms'] for x in cards);active=math.fsum(x['active_ms'] for x in cards)
    roleC={r:math.fsum(x['role_compute_ms'][r] for x in cards) for r in ['long','short']}
    roleA={r:math.fsum(x['role_active_ms'][r] for x in cards) for r in ['long','short']}
    assert close(compute+stall,active)
    return {'window_ms':[left,right],'device_U_percent':100*compute/(32*(right-left)),
            'stall_percent':100*stall/(32*(right-left)),'idle_percent':100*(32*(right-left)-active)/(32*(right-left)),
            'compute_ms':compute,'stall_ms':stall,'active_ms':active,
            'all32active':all(x['all_active'] for x in cards),
            'warm_mixed_card_count':sum(x['both_roles_positive_compute'] for x in cards),
            'warm_admissions':len(warm),'warm_slo_passed':passed,'warm_slo_percent':100*passed/len(warm) if warm else None,
            'mean_long_active_cards':roleA['long']/(right-left),'mean_short_active_cards':roleA['short']/(right-left),
            'role_compute_ms':roleC,'role_active_ms':roleA,
            'role_conditional_U_percent':{r:100*roleC[r]/roleA[r] if roleA[r] else None for r in roleC},'per_npu':cards}


def nominal(ledger):
    changes=defaultdict(lambda:{'add':[],'remove':[]})
    for x in ledger:
        changes[x['admission']]['add'].append(x);changes[x['completion']]['remove'].append(x)
    times=sorted(changes);active={};peaks=[0.]*6;duration=[0.]*6;any_over=0.;link_peak=0.
    for i,t in enumerate(times[:-1]):
        for x in changes[t]['remove']:assert active.pop(x['npu'])['rid']==x['rid']
        for x in changes[t]['add']:
            assert x['npu'] not in active;active[x['npu']]=x
        dt=times[i+1]-t;current=[math.fsum(x['rates'][s] for x in active.values()) for s in range(6)]
        for s,v in enumerate(current):
            peaks[s]=max(peaks[s],v)
            if v>40+1e-8:duration[s]+=dt
        if any(v>40+1e-8 for v in current):any_over+=dt
        link_peak=max(link_peak,max((math.fsum(x['rates']) for x in active.values()),default=0.))
    return {'per_ssu_peak_gib_s':peaks,'per_ssu_over40_ms':duration,'any_ssu_over40_ms':any_over,
            'max_ssu_gib_s':max(peaks),'max_npu_link_gib_s':link_peak,
            'passed':any_over==0 and link_peak<50,
            'definition':'Current admitted requests per-SSD physical V / pure layer C, ties remove then add atomically; no additional cross-request L0 term.'}


def audit(item,plan):
    command_path=HERE/'runs'/item['label']/'baseline'/'command.json'
    if not command_path.exists():return {'label':item['label'],'status':'pending'}
    cmd=read(command_path)
    if cmd['status']!='complete':return {'label':item['label'],'status':cmd['status'],'command':cmd}
    assert cmd['returncode']==0 and cmd['input_sha256']==sha(item['manifest'])==item['manifest_sha256']
    assert cmd['source_sha256']==plan['source_sha256']
    assert all(sha(ROOT/k)==v for k,v in plan['source_sha256'].items())
    paths=list(command_path.parent.glob('*.json.gz'));assert len(paths)==1
    raw=read(paths[0]);m=read(item['manifest']);q={x['request_id']:x for x in m['requests']};data=ast.literal_eval((ROOT/'data').read_text())
    assert raw['input_fingerprint']==m['input_fingerprint']==item['input_fingerprint']
    assert raw['strategy']=='baseline' and raw['submit_seed']==7
    assert raw['collector_interval_ms']==5 and raw['policy_config']['assignment']=='fixed' and raw['assignment_log']==[]
    assert raw['execution_placement_fingerprint']==raw['input_placement_fingerprint']
    assert len(raw['core_and_policy_sha256'])==29 and all(raw['core_and_policy_sha256'][k]==plan['source_sha256'][k] for k in raw['core_and_policy_sha256'])
    summary=raw['summary'];assert all(summary['invariants'].values()) and len(q)==len(summary['request_metrics'])==4096
    ledger=[];seen=set()
    for b in summary['microbatch_metrics']:
        assert b['batch_size']==1;rid=b['member_request_ids'][0];assert rid not in seen;seen.add(rid);r=q[rid];load=r['load'];key=(load['seq_len_k'],load['nql'])
        assert load['per_layer_us']==data[key][1] and load['per_layer_kv_gb']==data[key][3] and key[0]>10
        assert r['arrival_time_ms']==0 and b['npu_id']==r['npu_id']
        p=m['placements'][r['placement_index']];assert len(p)==8 and all(l==p[0] for l in p)
        assert all(v==176*1024/2**30 and s==(j+r['npu_id']//4)%6 for j,(s,v) in enumerate(p[0]))
        C=load['per_layer_us']/1000;assert close(math.fsum(v for s,v in p[0]),load['per_layer_kv_gb'])
        cs=[];st=[];prev=b['admission_time_ms']
        for i,l in enumerate(sorted(b['layer_metrics'],key=lambda x:x['layer'])):
            assert l['layer']==i and close(l['compute_end_ms']-l['compute_start_ms'],C)
            assert close(l['compute_start_ms']-prev,l['io_barrier_wait_ms']) and close(max(prev,l['io_ready_time_ms']),l['compute_start_ms'])
            cs.append((l['compute_start_ms'],l['compute_end_ms']));st.append((prev,l['compute_start_ms']));prev=l['compute_end_ms']
        assert len(cs)==8 and close(prev,b['completion_time_ms'])
        ledger.append({'rid':rid,'npu':b['npu_id'],'role':load['role'],'C':C,
                       'admission':b['admission_time_ms'],'completion':b['completion_time_ms'],'compute':cs,'stall':st,
                       'rates':[math.fsum(v for s,v in p[0] if s==disk)*1000/C for disk in range(6)]})
    assert seen==q.keys()
    segments=[];transitions=[]
    for n in range(32):
        lane=sorted((x for x in ledger if x['npu']==n),key=lambda x:x['admission'])
        assert [x['rid'] for x in lane]==sorted(x['rid'] for x in lane)
        assert all(close(a['completion'],b['admission']) for a,b in zip(lane,lane[1:]))
        for x in lane:
            if segments and segments[-1]['npu']==n and segments[-1]['role']==x['role']:
                segments[-1]['completion']=x['completion'];segments[-1]['request_count']+=1
            else:segments.append({'npu':n,'role':x['role'],'admission':x['admission'],'completion':x['completion'],'request_count':1})
        ns=[x for x in segments if x['npu']==n]
        transitions.append({'npu':n,'role_segments_total':len(ns),'boundaries_2_12_ms':[x['admission'] for x in ns[1:] if 2000<=x['admission']<12000],
                            'both_role_segments_overlapping_2_12':dict(Counter(x['role'] for x in ns if overlap(x['admission'],x['completion'],2000,12000)>0))})
    cohort_spread=[]
    for cohort in [0,1]:
        grouped=[[x for x in segments if x['npu']==n] for n in range(cohort*16,(cohort+1)*16)]
        assert len({len(x) for x in grouped})==1
        for i in range(len(grouped[0])):
            cells=[x[i] for x in grouped];assert len({x['role'] for x in cells})==1
            starts=[x['admission'] for x in cells];ends=[x['completion'] for x in cells]
            cohort_spread.append({'cohort':cohort,'segment_index':i,'role':cells[0]['role'],
                'start_min_ms':min(starts),'start_max_ms':max(starts),'start_spread_ms':max(starts)-min(starts),
                'end_min_ms':min(ends),'end_max_ms':max(ends),'end_spread_ms':max(ends)-min(ends),
                'mean_active_duration_ms':math.fsum(z-a for a,z in zip(starts,ends))/16})
    windows=[window(ledger,*w) for w in plan['windows_ms']]
    for w in windows:
        saved=next(x for x in raw['windows'] if [x['start_ms'],x['end_ms']]==w['window_ms'])
        assert close(w['device_U_percent']/100,saved['mean_npu_utilization'])
    scan=nominal(ledger);assert max(scan['per_ssu_peak_gib_s'])<=max(item['capacity_certificate']['per_ssu_upper_gib_s'])+1e-7
    return {'label':item['label'],'status':'complete','audit_passed':True,'result':str(paths[0]),'result_sha256':sha(paths[0]),
            'input_fingerprint':raw['input_fingerprint'],'makespan_ms':summary['makespan_ms'],'full_run_nominal':scan,
            'windows':windows,'per_npu_role_transitions':transitions,'role_segments':segments,
            'cohort_role_segment_phase_spread':cohort_spread,
            'all_7_windows_active32':all(x['all32active'] for x in windows),
            'all_7_windows_mixed32':all(x['warm_mixed_card_count']==32 for x in windows),
            'max_two_second_bin_U_percent':max(x['device_U_percent'] for x in windows if x['window_ms'][1]-x['window_ms'][0]==2000)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    plan=read(HERE/'plan.json');results=[];errors=[]
    for item in plan['inputs']:
        try:results.append(audit(item,plan))
        except Exception as e:errors.append({'label':item['label'],'error':repr(e)})
    out={'planned':2,'complete':sum(r['status']=='complete' for r in results),'errors':errors,'results':results,
         'analysis_sha256':sha(__file__),'plan_sha256':sha(HERE/'plan.json'),
         'pilot_scope':'Single Short profile mechanism probe; no exclusion of weak/invalid results.'}
    (HERE/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    fields=['label','start_ms','end_ms','device_U_percent','stall_percent','idle_percent','mean_long_active_cards',
            'warm_mixed_card_count','all32active','warm_slo_percent','warm_admissions','full_run_ssu_peak_gib_s','full_run_nominal_pass']
    with (HERE/'windows.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader()
        for r in results:
            if r['status']!='complete':continue
            for w in r['windows']:
                row={k:w[k] for k in fields if k in w};row.update(label=r['label'],start_ms=w['window_ms'][0],end_ms=w['window_ms'][1],
                    full_run_ssu_peak_gib_s=r['full_run_nominal']['max_ssu_gib_s'],full_run_nominal_pass=r['full_run_nominal']['passed']);wr.writerow(row)
    print(json.dumps({'complete':out['complete'],'errors':errors,'results':[{'label':r['label'],'status':r['status'],
                     'main_U':r.get('windows',[{}])[0].get('device_U_percent')} for r in results]}))
    if errors or (args.require_complete and out['complete']!=2):raise SystemExit(1)


if __name__=='__main__':main()
