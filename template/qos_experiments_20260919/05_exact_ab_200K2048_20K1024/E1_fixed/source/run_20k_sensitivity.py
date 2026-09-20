#!/usr/bin/env python3
"""NATIVE SIMULATOR WITH EXTRAPOLATED COMPUTE: uncalibrated20K sensitivity.

Derived from run_random_multitype.py. Original SSD/NPU/policy model retained.
20K compute uses existing run_fixed128_32.profile extrapolation with actual
32K/48K length anchors, weights1.75/-0.75. This is not measured-compute evidence.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

import sim
import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from explore_fifo_underload import demand_audit
from fixed_total_compat import run_case
from run_baseline_npu32_stress import save_manifest, write_json
from run_fixed128_32 import profile, window_lanes, PhysicalReceiptObserver
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914'
LAYERS=8

def build(spec, seed=7, horizon_ms=4500.):
    table=ast.literal_eval((ROOT/'data').read_text())
    N=8;S=int(spec.get('ssu',1));groups=spec['groups']
    assert S>=1 and len(groups)>=2
    assert len({g['id'] for g in groups})==len(groups)
    assert {g['role'] for g in groups}=={'L','S'}
    for g in groups:
        assert 20<=g['total_k']<=200 and g['role'] in ('L','S')
        assert int(g['weight'])==g['weight'] and g['weight']>0
        assert 64<=g['nql']<=4096
    profiles={}
    def get(g,y):
        key=(int(round(g['total_k']*1024)),y)
        if key not in profiles: profiles[key]=profile(table,key[0]-y,y)
        return profiles[key]
    cycle_ms=LAYERS*sum(g['weight']*get(g,g['nql'])['compute_us']/1000 for g in groups)
    cycles=math.ceil(horizon_ms/cycle_ms)+1
    # Pools have a small integer-NQL spread and never extrapolate. At NQL=4096
    # the nearest legal candidates are below the boundary; at 64 they are above.
    while True:
        pools={};used=set()
        for g in groups:
            count=cycles*g['weight'];size=math.ceil(count*1.15)+4
            candidates=sorted(range(64,4097),key=lambda y:(abs(y-g['nql']),y))
            ys=[]
            for y in candidates:
                key=(int(round(g['total_k']*1024)),y)
                # A lone request must be serviceable by both total storage and
                # its NPU link. Aggregate burst overload is audited separately.
                if key not in used and get(g,y)['B_gib_s']<=min(40*S,50)*(1-1e-9):
                    ys.append(y);used.add(key)
                if len(ys)==size:break
            if len(ys)<size:raise ValueError('Not enough unique profiles')
            pools[g['id']]=[get(g,y) for y in ys]
        lower_ms=LAYERS*sum(cycles*g['weight']*min(p['compute_us']/1000 for p in pools[g['id']]) for g in groups)
        if lower_ms>horizon_ms:break
        cycles+=1
    requests=[];vectors={};lane_checks=[];maxima=[[0.]*S for _ in range(N)]
    for n in range(N):
        rng=random.Random(seed+100003*n)
        deck=[g['id'] for g in groups for _ in range(cycles*g['weight'])]
        rng.shuffle(deck)
        picks={g['id']:rng.sample(pools[g['id']],cycles*g['weight']) for g in groups}
        positions=Counter();seen=set();ideal_ms=0.;gmap={g['id']:g for g in groups}
        for generation,gid in enumerate(deck):
            g=gmap[gid];p=picks[gid][positions[gid]];positions[gid]+=1
            key=(p['total_tokens'],p['nql']);assert key not in seen;seen.add(key)
            rid=n*1000000+generation;prefix=p['ssd_prefix_tokens']
            layer=tuple((sim.block_ring_hash_disk_id(rid,j,S),min(128,prefix-128*j)*1408/2**30)
                        for j in range(math.ceil(prefix/128)))
            assert math.isclose(math.fsum(v for _,v in layer),p['read_gib'],abs_tol=1e-12)
            load=dict(request_id=rid,npu_id=n,generation=generation,original_request_id=rid,
                total_tokens=p['total_tokens'],seq_len_k=p['total_length_k'],nql=p['nql'],
                role=g['role'],profile_group=gid,ssd_prefix_tokens=prefix,
                category=sim.classify_request(p['total_length_k'],p['nql']),
                per_layer_us=p['compute_us'],per_layer_kv_gb=p['read_gib'],required_bw_input_gbps=p['B_gib_s'],
                arrival_time=0.,arrival_ms=0.,initial=True,constructed_profile=p['constructed_profile'],
                profile_construction=p['profile_construction'],original_compute_us=p['compute_us'],padding_gib_per_layer=0.)
            q=ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,))
            assert all(native._manifest_layer(q,l) is layer for l in range(LAYERS))
            requests.append(q);ideal_ms+=LAYERS*p['compute_us']/1000
            vector=[math.fsum(v for disk,v in layer if disk==s)/(p['compute_us']/1e6) for s in range(S)]
            vectors[rid]=vector;maxima[n]=[max(a,b) for a,b in zip(maxima[n],vector)]
        assert ideal_ms>horizon_ms
        lane_checks.append(dict(npu_id=n,request_count=len(deck),unique_length_nql_count=len(seen),
            profile_group_counts=dict(Counter(deck)),profile_group_order=deck,
            request_counts=dict(Counter(gmap[gid]['role'] for gid in deck)),ideal_compute_ms=ideal_ms,
            shuffle_seed=seed+100003*n))
    requests=tuple(requests);static=[sum(row[s] for row in maxima) for s in range(S)]
    specs={}
    for g in groups:
        rows=[q.load for q in requests if q.load['profile_group']==g['id']]
        specs[g['id']]=dict(**g,requests=len(rows),
            nql_range=[min(p['nql'] for p in rows),max(p['nql'] for p in rows)],
            compute_ms_range=[min(p['per_layer_us'] for p in rows)/1000,max(p['per_layer_us'] for p in rows)/1000],
            read_gib_range=[min(p['per_layer_kv_gb'] for p in rows),max(p['per_layer_kv_gb'] for p in rows)],
            B_gib_s_range=[min(p['required_bw_input_gbps'] for p in rows),max(p['required_bw_input_gbps'] for p in rows)],
            direct_data_rows=sum(not p['constructed_profile'] for p in rows))
    counts=Counter()
    for g in groups:counts[g['role']]+=g['weight']
    meta=dict(experiment='lower_fifo_followup_20k_extrapolated_20260914',
        evidence_scope='Native simulator with uncalibrated extrapolated20K compute; not measured-compute native evidence.',
        compute_calibrated=all(not q.load['profile_construction']['extrapolated'] for q in requests),
        contains_extrapolated_compute=any(q.load['profile_construction']['extrapolated'] for q in requests),case=dict(name=spec['name'],num_npu=N,ssu=S,
        mode='all_npu_mixed',order='random',phase='sync',short_per_long=counts['S']/counts['L'],groups=groups),
        specification=spec,profiles=specs,num_npu=N,num_ssu=S,n_layers=LAYERS,seed=seed,horizon_ms=horizon_ms,
        order='random',phase='sync',lane_checks=lane_checks,layout='block_ring_hash',placement_virtual_nodes_per_ssu=256,
        placement_key='(request_id, block_index); layer excluded; all layers reuse exact mapping',
        input_fingerprint=continuous_batch_input_fingerprint(requests),logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests,N,S),per_ssu_static_upper_bound_gib_s=static,
        static_underload_all_request_combinations=max(static)<=40+1e-8,
        static_upper_definition='Per SSU: sum each NPU maximum D/C over its complete frozen mixed deck',
        all_length_nql_unique_within_each_npu=True,all_npus_have_both_roles_in_input=True,
        all_compute_profiles_within_measured_grid=all(not q.load['profile_construction']['extrapolated'] for q in requests),variable_tail_io_supported=True,
        equal_176kib_blocks=all(v==176*1024/2**30 for q in requests for _,v in q.placement[0]),
        data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
        workload_scope='Each NPU has the same group counts and an independent random permutation. '
        'All requests arrive at t=0; fixed NPU assignment; batch size 1; exact KV prefix bytes, no padding. '
        'NQLs are sampled without replacement near each base value; total lengths are fixed per group. '
        'NQL interpolation stays in the measured grid; lengths below32K use explicitly uncalibrated extrapolation. No compute scaling. Negative anchor weights are preserved in every load profile.')
    return requests,meta,vectors

def by_group(summary,requests,left,right):
    loads={q.request_id:q.load for q in requests};rows={}
    clip=lambda a,b:max(0.,min(b,right)-max(a,left))
    for b in summary['microbatch_metrics']:
        q=loads[b['member_request_ids'][0]];gid=q['profile_group']
        row=rows.setdefault(gid,dict(group=gid,role=q['role'],active_ms=0.,compute_ms=0.,internal_stall_ms=0.,l0_stall_ms=0.,warm_admitted=0,computed=0))
        active=clip(b['admission_time_ms'],b['completion_time_ms'])
        compute=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for l in b['layer_metrics'])
        row['active_ms']+=active;row['compute_ms']+=compute;row['computed']+=compute>0
        row['warm_admitted']+=left<=b['admission_time_ms']<right
        last=b['admission_time_ms']
        for l in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            row['l0_stall_ms' if l['layer']==0 else 'internal_stall_ms']+=clip(last,l['compute_start_ms'])
            last=l['compute_end_ms']
    for row in rows.values():
        row['U_percent']=100*row['compute_ms']/row['active_ms'] if row['active_ms'] else None
        row['stall_ms']=row['active_ms']-row['compute_ms']
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--case')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--policy',choices=['fifo','once','short_first'],default='fifo')
    p.add_argument('--stage',default='screen');p.add_argument('--horizon-ms',type=float,default=4500.)
    p.add_argument('--window',type=float,nargs=2,default=[2000.,4000.]);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();assert 0<=a.window[0]<a.window[1]<=a.horizon_ms
    spec=json.loads(a.spec.read_text())
    if isinstance(spec,list):spec=next(s for s in spec if s['name']==a.case)
    started=time.perf_counter();requests,meta,vectors=build(spec,a.seed,a.horizon_ms)
    meta['probe_policy']=a.policy;meta['queue_order']='FIFO path0' if a.policy=='fifo' else a.policy
    dest=OUT/a.stage/f"{spec['name']}_seed{a.seed}_{a.policy}"
    if (dest/'result.json.gz').exists():raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    save_manifest(dest/'manifest.json.gz',requests,meta);write_json(dest/'metadata.json',meta)
    print(json.dumps(dict(event='prepared' if a.prepare_only else 'start',case=spec['name'],seed=a.seed,policy=a.policy,
        requests=len(requests),input_mean_gib_s=meta['input_demand']['total_gib_s'],static_upper=meta['per_ssu_static_upper_bound_gib_s'],
        destination=str(dest))),flush=True)
    if a.prepare_only:return
    observer=PhysicalReceiptObserver(requests,8,meta['num_ssu'],a.window,policy=a.policy)
    with contextlib.ExitStack() as stack:
        if a.policy=='short_first':
            from run_fifo_proof import SizePriorityPending
            volumes={q.request_id:q.load['per_layer_kv_gb'] for q in requests};original=sim.PathQueue.__init__
            def init(queue,*args,**kwargs):
                original(queue,*args,**kwargs)
                if queue.path_id==0:queue.pending=SizePriorityPending(volumes)
            stack.enter_context(patch.object(sim.PathQueue,'__init__',init))
        stack.enter_context(patch.object(native,'_register_complete',observer))
        stack.enter_context(patch.object(native,'_enqueue_link_io',observer.observe_ssd))
        result=run_case(requests,meta,strategy='once' if a.policy=='once' else 'baseline',assignment='fixed',windows=[tuple(a.window)])
    whole=demand_audit(result['summary'],vectors,meta['num_ssu'],0.,result['summary']['makespan_ms'])
    audit=demand_audit(result['summary'],vectors,meta['num_ssu'],*a.window)
    lanes=window_lanes(result['summary'],requests,8,*a.window)
    result.update(evidence_scope=meta['evidence_scope'],compute_calibrated=meta['compute_calibrated'],probe_policy=a.policy,demand_audit=audit,whole_run_demand_audit=whole,mixed_window_by_npu=lanes,exploration_case=meta['case'])
    write_json(dest/'result.json.gz',result);write_json(dest/'receipts.json',observer.payload(result,dest))
    w=result['windows'][0];slo=result['slo']['window_admissions']['admission']
    util=lambda role:100*w['by_role'][role]['active_compute_fraction'] if w['by_role'].get(role,{}).get('active_compute_fraction') is not None else None
    row=dict(evidence_scope=meta['evidence_scope'],compute_calibrated=meta['compute_calibrated'],name=spec['name'],num_npu=8,num_ssu=meta['num_ssu'],seed=a.seed,policy=a.policy,window_ms=a.window,
        U_percent=100*w['mean_npu_utilization'],short_U_percent=util('S'),long_U_percent=util('L'),
        slo_1p5_percent=100*slo['rate'] if slo['rate'] is not None else None,slo_passed=slo['passed'],slo_count=slo['count'],
        input_average_gib_s=meta['input_demand']['per_ssu_gib_s'],static_upper_gib_s=meta['per_ssu_static_upper_bound_gib_s'],
        static_underload_all_request_combinations=meta['static_underload_all_request_combinations'],
        all_npus_both_roles_computed=all(r['both_roles_computed'] for r in lanes),all_active=w['all_npus_active_whole_window'],
        all_invariants_passed=all(result['summary']['invariants'].values()),all_length_nql_unique_within_each_npu=True,
        demand_audit=audit,whole_run_demand_audit=whole,window_by_npu=lanes,
        by_profile_group=by_group(result['summary'],requests,*a.window),wall_seconds=time.perf_counter()-started)
    write_json(dest/'metrics.json',row)
    print(json.dumps({k:v for k,v in row.items() if k not in ('window_by_npu','by_profile_group','demand_audit','whole_run_demand_audit')}),flush=True)

if __name__=='__main__':main()
