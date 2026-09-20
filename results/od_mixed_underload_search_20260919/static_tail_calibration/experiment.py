#!/usr/bin/env python3
"""Frozen, synthetic tail-compute calibration. Never changes inputs at runtime."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(ROOT)]
from simulator.core import continuous_batch_sim as native
from inputs.runners.run_baseline_npu32_stress import load_manifest,save_manifest,run_case
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from inputs.runners.run_coflow_experiments import source_files
from metrics import live_summary,exact_demand,overlap
BASE=HERE.parent/'runs/native_phase_lock_a102_od_local'

class Stop(Exception):pass

def freeze(gain):
    requests,meta=load_manifest(BASE/'manifest.json.gz');raw=json.load(gzip.open(BASE/'result.json.gz','rt'))['summary']
    byid={q.request_id:q for q in requests};CB=next(q.load['per_layer_us']/1000 for q in requests if q.load['role']=='B')
    anchor=max(b['layer_metrics'][0]['compute_start_ms']for b in raw['microbatch_metrics']if b['member_request_ids'][0]%1000000==0)
    corrections=[]
    for n in range(8,16):
        rows=[b for b in raw['microbatch_metrics']if b['npu_id']==n];i=next(j for j,b in enumerate(rows)if byid[b['member_request_ids'][0]].load['role']=='A');j=i
        while byid[rows[j]['member_request_ids'][0]].load['role']=='A':j+=1
        actual=rows[j]['layer_metrics'][0]['compute_start_ms'];target=anchor+(i+1)*8*CB
        assert target>actual
        corrections.append(dict(template_npu=n,target_B_start_ms=target,source_B_start_ms=actual,
                                advance_ms=target-actual,per_layer_add_ms=gain*(target-actual)/7))
    out=[];calibrated=[]
    for n in range(32):
        qs=[q for q in requests if q.npu_id==n]
        for j,q in enumerate(qs):
            load=dict(q.load);load['input_kind']=load['role']
            if load['role']=='A'and j+1<len(qs)and qs[j+1].load['role']=='B':
                old=load['per_layer_us']/1000;new=old+corrections[n%8]['per_layer_add_ms'];load.update(per_layer_us=new*1000,
                    required_bw_input_gbps=load['per_layer_kv_gb']*1000/new,input_kind='P',constructed_profile=True,
                    profile_construction=dict(method='offline frozen synthetic tail calibration',gain=gain,original_C_ms=old,
                       template_npu=8+n%8,delta_C_ms=new-old,formula='gain * earliest-cohort advance / 7; reused for all A blocks'))
                calibrated.append(dict(request_id=q.request_id,npu=n,original_C_ms=old,C_ms=new))
            out.append(native.ContinuousBatchRequest.from_normalized(q.request_id,q.npu_id,q.arrival_time_ms,load,q.placement))
    out=tuple(out);label=f'tail_gain_{gain:g}';meta=dict(meta,label=label,case_id=label,order='static_frozen_tail_calibration',
        constructed_profile=True,input_fingerprint=native.continuous_batch_input_fingerprint(out),logical_input_fingerprint=logical_input_fingerprint(out),
        calibration=dict(gain=gain,source_trace=str(BASE/'result.json.gz'),source_trace_sha256=hashlib.sha256((BASE/'result.json.gz').read_bytes()).hexdigest(),
            source_manifest_sha256=hashlib.sha256((BASE/'manifest.json.gz').read_bytes()).hexdigest(),target_anchor_ms=anchor,C_B_ms=CB,
            superperiod_ms=32*CB,cohort_offset_ms=8*CB,corrections=corrections,calibrated_requests=calibrated,
            runtime_control=False,formula_denominator=7,caveat='Third synthetic request kind P, not pure queue reordering and not a data measurement'))
    path=HERE/'inputs'/f'{label}.json.gz';save_manifest(path,out,meta);return path

def measure(summary,requests,lo,hi):
    byid={q.request_id:q for q in requests};cards=[dict(A=0.,B=0.,P=0.)for _ in range(32)];active=[0.]*32
    for r in summary['request_metrics']:active[r['npu_id']]+=overlap(r['admission_time_ms'],r['completion_time_ms'],lo,hi)
    for b in summary['microbatch_metrics']:
        kind=byid[b['member_request_ids'][0]].load['input_kind']
        for l in b['layer_metrics']:cards[b['npu_id']][kind]+=overlap(l['compute_start_ms'],l['compute_end_ms'],lo,hi)
    return dict(start_ms=lo,end_ms=hi,U_percent=100*sum(sum(c.values())for c in cards)/(32*(hi-lo)),
        per_npu_compute_ms=cards,mixed_cards=sum(c['A']+c['P']>1e-7 and c['B']>1e-7 for c in cards),
        ordinary_A_and_B_cards=sum(c['A']>1e-7 and c['B']>1e-7 for c in cards),all_active=all(abs(v-(hi-lo))<1e-7 for v in active),
        demand=exact_demand(summary['request_metrics'],byid,lo,hi))

def run(path,cpu):
    os.sched_setaffinity(0,{cpu});requests,meta=load_manifest(path);target=HERE/'runs'/meta['label'];target.mkdir(exist_ok=False)
    sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in source_files()};start=time.time();old=native._register_complete;summary=None
    def observe(context,flow):
        nonlocal summary
        ret=old(context,flow)
        if context.current_time_ms>8000.000001:summary=live_summary(context);raise Stop()
        return ret
    record=dict(status='running',manifest=str(path),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu=cpu,source_sha256=sources,
                calibration=meta['calibration'],stop_ms=8000,pilot_only=True,SLO=None,completed_simulation=False)
    (target/'command.json').write_text(json.dumps(record,indent=2))
    try:
        with patch.object(native,'_register_complete',observe):run_case(requests,meta,strategy='od_baseline',assignment='fixed',windows=((2000.,4000.),(4000.,8000.)))
    except Stop:pass
    assert summary is not None
    after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in source_files()};assert after==sources
    with gzip.open(target/'live_summary.json.gz','wt')as f:json.dump(summary,f)
    record.update(status='complete_pilot',source_unchanged=True,wall_s=time.time()-start,
       windows=[measure(summary,requests,*w)for w in [(2000.,4000.),(4000.,8000.),(0.,8000.)]])
    (target/'command.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(label=meta['label'],windows=[dict(U=w['U_percent'],under=w['demand']['strict_underload_all_disks'],mixed=w['mixed_cards'],ordinary_mix=w['ordinary_A_and_B_cards'])for w in record['windows']],wall_s=record['wall_s'])),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gain',type=float,required=True);ap.add_argument('--cpu',type=int,required=True);args=ap.parse_args()
    run(freeze(args.gain),args.cpu)

if __name__=='__main__':main()
