#!/usr/bin/env python3
"""Offline audit for the single calibration bridge control; never runs a simulator."""
from pathlib import Path
from collections import Counter
import argparse
import ast
import csv
import importlib.util
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from run_baseline_npu32_stress import load_manifest
import sim

spec = importlib.util.spec_from_file_location('rotation_audit_functions', HERE.parent/'pilot_rotation/audit_pilots.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
read, sha, close, window, nominal = shared.read, shared.sha, shared.close, shared.window, shared.nominal


def audit_input(item):
    m = read(item['manifest'])
    meta = m['metadata']
    requests, _ = load_manifest(item['manifest'])
    cfg = read(HERE/'spec.json')[0]
    data = ast.literal_eval((ROOT/'data').read_text())
    assert sha(item['manifest']) == item['manifest_sha256']
    assert m['input_fingerprint'] == meta['input_fingerprint'] == item['input_fingerprint']
    assert meta['construction_spec'] == cfg
    assert meta['source_data_sha256'] == sha(ROOT/'data')
    assert meta['compute_scale_actual'] == 1 and meta['num_npu'] == 32 and meta['num_ssu'] == 6
    assert len(requests) == len(m['requests']) == item['request_count'] == meta['request_count']
    maxima = [[0.]*6 for _ in range(32)]
    pure = [0.]*32
    volume_roundoff = {}
    q = {r['request_id']: r for r in m['requests']}
    assert len(q) == len(requests)
    for r in m['requests']:
        load = r['load']; key = (load['seq_len_k'], load['nql']); n = r['npu_id']
        assert key[0] > 10 and load['per_layer_us'] == data[key][1]
        # The data's decimal arithmetic can be one ULP off exact physical block bytes.
        volume_delta = load['per_layer_kv_gb']-data[key][3]
        assert abs(volume_delta) <= math.ulp(data[key][3])
        volume_roundoff[str(key)] = volume_delta
        assert not load['constructed_profile'] and load['padding_gib_per_layer'] == 0
        assert load['category'] == sim.classify_request(*key)
        assert load['role'] == ('long' if key == (176,1024) else 'short')
        assert r['arrival_time_ms'] == load['arrival_ms'] == load['arrival_time'] == 0
        assert r['request_id'] == n*1000000+load['generation'] == load['original_request_id']
        assert load['request_id'] == r['request_id'] and load['npu_id'] == n
        p = m['placements'][r['placement_index']]
        assert len(p) == 1
        assert all(v == 176*1024/2**30 and s == (j+n)%6 for j,(s,v) in enumerate(p[0]))
        assert close(math.fsum(v for s,v in p[0]), data[key][3])
        rates = [math.fsum(v for s,v in p[0] if s==d)*1e6/load['per_layer_us'] for d in range(6)]
        maxima[n] = [max(a,b) for a,b in zip(maxima[n], rates)]
        pure[n] += 8*load['per_layer_us']/1000
    assignments = []
    for n in range(32):
        cycle = list(cfg['long_cycle'] if n<16 else cfg['short_cycle'])
        offset = 0 if n<16 else (n-16)*len(cycle)//16
        cycle = cycle[offset:]+cycle[:offset]
        rows = sorted((r for r in q.values() if r['npu_id']==n),key=lambda r:r['request_id'])
        actual = [r['load']['profile_index'] for r in rows]
        assert len(actual)%len(cycle)==0 and actual == cycle*(len(actual)//len(cycle))
        assert pure[n] >= 14000
        assignments.append(dict(npu=n,cycle=cycle,initial_cycle_offset=offset,requests=len(rows),
                                profile_counts=dict(Counter(actual)),pure_compute_ms=pure[n]))
    bounds = [math.fsum(x[d] for x in maxima) for d in range(6)]
    assert all(close(a,b) for a,b in zip(bounds,meta['active_profile_rate_certificate']['per_ssu_upper_bound_gib_s']))
    assert max(bounds)<40 and max(map(math.fsum,maxima))<50
    return m,q,dict(passed=True,request_count=len(q),per_ssu_any_combination_upper_gib_s=bounds,
                   physical_volume_minus_data_gib=volume_roundoff,
                   min_per_npu_pure_compute_ms=min(pure),per_npu=assignments,
                   qualification='Long group Short is S32/NQL4096, C close to Long. This is a phase-preservation mechanism control, not a fast-Short final example.')


def audit_result(item, q, m):
    path = HERE/'runs'/item['label']/'baseline'/'command.json'
    cmd = read(path)
    if cmd['status'] != 'complete': return dict(status=cmd['status'],pid=cmd.get('pid'))
    assert cmd['returncode'] == 0
    assert cmd['manifest_sha256'] == sha(item['manifest'])
    assert cmd['runner_sha256'] == read(HERE/'runner_snapshot.json')['sha256'] == sha(HERE/'run_candidates_frozen.py.txt')
    assert len(cmd['core_source_sha256'])==29
    assert all(sha(ROOT/k)==v for k,v in cmd['core_source_sha256'].items())
    raw = read(cmd['output'])
    assert sha(cmd['output']) == cmd['output_sha256']
    assert raw['input_fingerprint'] == item['input_fingerprint']
    assert raw['core_and_policy_sha256'] == cmd['core_source_sha256']
    assert raw['strategy']=='baseline' and raw['submit_seed']==7
    assert raw['collector_interval_ms']==5 and raw['policy_config']['assignment']=='fixed'
    assert raw['assignment_log']==[] and raw['execution_placement_fingerprint']==raw['input_placement_fingerprint']
    summary = raw['summary']
    assert all(summary['invariants'].values())
    assert len(summary['request_metrics'])==len(q)
    ledger=[]; seen=set()
    for b in summary['microbatch_metrics']:
        assert b['batch_size']==1
        rid=b['member_request_ids'][0]; assert rid not in seen; seen.add(rid)
        r=q[rid]; load=r['load']; C=load['per_layer_us']/1000
        assert b['npu_id']==r['npu_id']
        previous=b['admission_time_ms']; cs=[]; stalls=[]
        layers=sorted(b['layer_metrics'],key=lambda l:l['layer'])
        assert len(layers)==8
        for i,l in enumerate(layers):
            assert l['layer']==i and close(l['compute_end_ms']-l['compute_start_ms'],C)
            assert close(max(previous,l['io_ready_time_ms']),l['compute_start_ms'])
            assert close(l['compute_start_ms']-previous,l['io_barrier_wait_ms'])
            cs.append((l['compute_start_ms'],l['compute_end_ms']))
            stalls.append((previous,l['compute_start_ms']));previous=l['compute_end_ms']
        assert close(previous,b['completion_time_ms'])
        p=m['placements'][r['placement_index']][0]
        ledger.append(dict(rid=rid,npu=r['npu_id'],role=load['role'],C=C,profile_index=load['profile_index'],
            admission=b['admission_time_ms'],completion=b['completion_time_ms'],compute=cs,stall=stalls,
            layers=layers,rates=[math.fsum(v for s,v in p if s==d)*1000/C for d in range(6)]))
    assert seen==q.keys()
    for n in range(32):
        lane=sorted((x for x in ledger if x['npu']==n),key=lambda x:x['admission'])
        assert [x['rid'] for x in lane]==sorted(x['rid'] for x in lane)
        assert all(close(a['completion'],b['admission']) for a,b in zip(lane,lane[1:]))
    planned=read(HERE/'plan.json')['jobs'][0]['windows']
    windows=[window(ledger,*w) for w in planned]
    for w in windows:
        s=next(x for x in raw['windows'] if [x['start_ms'],x['end_ms']]==w['window_ms'])
        assert close(w['device_U_percent']/100,s['mean_npu_utilization'])
        left,right=w['window_ms']
        profile_c={i:math.fsum(shared.overlap(a,z,left,right) for x in ledger if x['profile_index']==i for a,z in x['compute']) for i in range(5)}
        profile_s={i:math.fsum(shared.overlap(a,z,left,right) for x in ledger if x['profile_index']==i for a,z in x['stall']) for i in range(5)}
        w['profile_compute_ms']=profile_c
        w['profile_stall_ms']=profile_s
        w['profile_conditional_U_percent']={i:100*profile_c[i]/(profile_c[i]+profile_s[i]) if profile_c[i]+profile_s[i]>0 else None for i in range(5)}
        w['cards_with_fast_short_positive_compute']=sum(any(x['npu']==n and x['profile_index'] in (0,1,2) and any(shared.overlap(a,z,left,right)>1e-7 for a,z in x['compute']) for x in ledger) for n in range(32))
        w['original_long_group_device_U_percent']=100*math.fsum(x['compute_ms'] for x in w['per_npu'] if x['npu']<16)/(16*(right-left))
        w['original_short_group_device_U_percent']=100*math.fsum(x['compute_ms'] for x in w['per_npu'] if x['npu']>=16)/(16*(right-left))
    extra=[window(ledger,*w) for w in [[100,1000],[1000,2000],[4000,12000]]]
    scan=nominal(ledger)
    long_group_nonstartup=[l['io_barrier_wait_ms'] for x in ledger if x['npu']<16 for l in x['layers']
                          if not (x['rid']%1000000==0 and l['layer']==0)]
    group=[]
    lanes={n:sorted((x for x in ledger if x['npu']==n),key=lambda x:x['rid']) for n in range(16)}
    for i in range(len(lanes[0])):
        if i%9 not in (0,7,8):continue
        xs=[lanes[n][i] for n in range(16)]
        def spread(values):return dict(min_ms=min(values),max_ms=max(values),spread_ms=max(values)-min(values))
        group.append(dict(cycle=i//9,position_in_cycle=i%9,profile_index=xs[0]['profile_index'],
            admission=spread([x['admission'] for x in xs]),completion=spread([x['completion'] for x in xs]),
            first_layer_ready=spread([x['layers'][0]['io_ready_time_ms'] for x in xs]),
            first_layer_compute=spread([x['layers'][0]['compute_start_ms'] for x in xs]),
            max_first_layer_stall_ms=max(x['layers'][0]['io_barrier_wait_ms'] for x in xs),
            max_request_stall_ms=max(math.fsum(z-a for a,z in x['stall']) for x in xs)))
    return dict(status='complete',audit_passed=True,result=cmd['output'],result_sha256=cmd['output_sha256'],
                makespan_ms=summary['makespan_ms'],wall_seconds=cmd['wall_seconds'],full_run_nominal=scan,
                long_group_nonstartup_layer_stall=dict(layers=len(long_group_nonstartup),
                    positive_count=sum(x>1e-9 for x in long_group_nonstartup),max_ms=max(long_group_nonstartup)),
                windows=windows,posthoc_diagnostic_windows=extra,long_group_phase=group)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    plan=read(HERE/'plan.json');assert len(plan['jobs'])==1
    assert plan['spec_sha256']==sha(HERE/'spec.json')
    item=plan['jobs'][0]['input'];m,q,input_audit=audit_input(item)
    result=audit_result(item,q,m)
    output=dict(planned=1,complete=int(result['status']=='complete'),input_audit=input_audit,result=result,
                auditor_sha256=sha(__file__),shared_audit_sha256=sha(shared.__file__),plan_sha256=sha(HERE/'plan.json'))
    (HERE/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    if result['status']=='complete':
        fields=['start_ms','end_ms','device_U_percent','stall_percent','idle_percent','all32active',
                'warm_mixed_card_count','mean_long_active_cards','warm_slo_percent','warm_admissions']
        with (HERE/'windows.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for w in result['windows']:
                writer.writerow(dict(start_ms=w['window_ms'][0],end_ms=w['window_ms'][1],**{k:w[k] for k in fields[2:]}))
    print(json.dumps(dict(input_passed=input_audit['passed'],request_count=len(q),result_status=result['status'],
                         max_static_ssu=max(input_audit['per_ssu_any_combination_upper_gib_s']),
                         main_U=result.get('windows',[{}])[0].get('device_U_percent'))))
    if args.require_complete and result['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
