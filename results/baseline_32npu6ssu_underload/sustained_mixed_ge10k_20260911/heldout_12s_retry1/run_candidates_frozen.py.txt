#!/usr/bin/env python3
"""Frozen periodic mixed raw-data queues. Uses unchanged full simulator."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import sim
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import profiles_for, save_manifest, read_json, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_multi_ssu_stall_experiments import input_demand
from run_coflow_experiments import source_files

BLOCK=176*1024/2**30

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def certificate(requests):
    maxima=[[0.]*6 for _ in range(32)]
    receiver=[0.]*32
    for r in requests:
        rate=[math.fsum(v for s,v in r.placement[0] if s==d)/(r.load['per_layer_us']/1e6) for d in range(6)]
        maxima[r.npu_id]=[max(a,b) for a,b in zip(maxima[r.npu_id],rate)]
        receiver[r.npu_id]=max(receiver[r.npu_id],sum(rate))
    disks=[math.fsum(m[d] for m in maxima) for d in range(6)]
    return dict(passes=max(disks)<40 and max(receiver)<50,per_ssu_upper_bound_gib_s=disks,
                per_npu_receive_upper_bound_gib_s=receiver,
                definition='Sum over NPUs of each-card maximum current-request per-disk D/C. Next-request L0 not separately added; all I/O simulated.')

def build(spec,mode,seed,out):
    keys=spec.get('keys','32:1024,48:1024,64:1024,176:1024')
    profiles,provenance=profiles_for('raw',keys)
    target=spec.get('horizon_ms',14000.)
    requests=[];assign=[]
    for n in range(32):
        cycle=list(spec['long_cycle'] if n<spec['long_cards'] else spec['short_cycle'])
        if n>=spec['long_cards'] and spec.get('short_phase_stagger'):
            offset=(n-spec['long_cards'])*len(cycle)//(32-spec['long_cards'])
            cycle=cycle[offset:]+cycle[:offset]
        cycles=math.ceil(target/(8*sum(profiles[i]['per_layer_compute_us']/1000 for i in cycle)))
        order=cycle*cycles
        # Different request identities; random control permutes exact per-card population.
        items=list(enumerate(order))
        if mode=='random':random.Random(seed*1000003+n*100003+71923).shuffle(items)
        role_C=Counter()
        for pos,(original,index) in enumerate(items):
            p=profiles[index];rid=n*1000000+pos
            assert p['ssd_prefix_tokens']%128==0
            layer=tuple(((j+n)%6,BLOCK) for j in range(p['ssd_prefix_tokens']//128))
            v=math.fsum(x for _,x in layer)
            assert math.isclose(v,p['per_layer_kv_gib'],abs_tol=1e-12)
            role='long' if index==len(profiles)-1 else 'short'
            role_C[role]+=8*p['per_layer_compute_us']/1000
            load=dict(request_id=rid,npu_id=n,generation=pos,seq_len_k=p['seq_len_k'],nql=p['nql'],
                category=sim.classify_request(p['seq_len_k'],p['nql']),per_layer_us=p['per_layer_compute_us'],
                per_layer_kv_gb=v,required_bw_input_gbps=v*1e6/p['per_layer_compute_us'],
                source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],arrival_time=0.,arrival_ms=0.,initial=True,
                role=role,profile_index=index,original_compute_us=p['per_layer_compute_us'],constructed_profile=False,
                profile_construction=p['construction'],padding_gib_per_layer=0.,original_request_id=n*1000000+original,
                original_cycle=original//len(cycle),original_cycle_position=original%len(cycle))
            requests.append(ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,)))
        assign.append(dict(npu_id=n,cycle=cycle,repetitions=cycles,requests=len(items),role_pure_compute_ms=dict(role_C),
                           pure_compute_ms=sum(role_C.values()),profile_counts=dict(Counter(order))))
    requests=tuple(requests);proof=certificate(requests)
    assert proof['passes']
    fp=continuous_batch_input_fingerprint(requests)
    label=f"{spec['name']}_{mode}_seed{seed}"
    meta=dict(experiment='sustained_mixed_ge10k_v1',label=label,case_id=label+'_'+fp[:12],
        num_npu=32,num_ssu=6,n_layers=8,seed=seed,family='raw',profiles=profiles,source=provenance,
        equal_176kib_blocks=True,blocks='exact',layout='stripe_npu_mod6',order=mode,
        assignment_mode='mixed_periodic',long_cards=None,short_cards=None,profile_keys=keys,
        input_fingerprint=fp,logical_input_fingerprint=logical_input_fingerprint(requests),
        active_profile_rate_certificate=proof,input_demand=input_demand(requests,32,6),
        load_within_disk_and_link_capacity=True,compute_scale_actual=1.,request_count=len(requests),
        per_npu_assignment=assign,measurement_window_ms=spec.get('windows',[[2000,12000]])[0],last_arrival_ms=0,
        source_data_sha256=sha(ROOT/'data'),construction_spec=spec,
        placement_rule='All layers of a request use exact176KiB blocks striped as (block_index+npu_id)%6; same original request retains placement under random shuffle.',
        population_rule='Deterministic mixed cycles repeated to cover horizon by pure compute; all arrivals0, per-card progress only, no sleeps or barriers.',
        caveat='Artificial correlated workload, not measured production traffic. Baseline low utility must survive long/subwindow auditing.')
    manifest=out/'inputs'/f'{label}.json.gz'
    save_manifest(manifest,requests,meta)
    return dict(label=label,mode=mode,seed=seed,manifest=str(manifest),manifest_sha256=sha(manifest),
                input_fingerprint=fp,request_count=len(requests),certificate=proof)

def run_job(item,strategy,out,windows):
    directory=out/'runs'/item['label']/strategy
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'command.json'
    if path.exists():
        prior=read_json(path)
        assert prior['status']=='complete','Inspect prior incomplete run before resuming'
        assert prior['manifest_sha256']==item['manifest_sha256']
        return prior
    cmd=[sys.executable,'-B',str(ROOT/'run_baseline_npu32_stress.py'),'--manifest',item['manifest'],
         '--strategy',strategy,'--assignment','fixed','--output',str(directory)]
    for a,b in windows:cmd+=['--window',f'{a}:{b}']
    source={f:sha(ROOT/f) for f in source_files()}
    rec=dict(**item,strategy=strategy,status='starting',command=cmd,cwd=str(ROOT),
             core_source_sha256=source,runner_sha256=sha(__file__),started_unix=time.time())
    t=time.perf_counter()
    with (directory/'stdout.log').open('w') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        rec.update(status='running',pid=proc.pid);write_json(path,rec)
        print(json.dumps(dict(started=item['label'],strategy=strategy,pid=proc.pid)),flush=True)
        try:code=proc.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            rec['status']='timeout';code=proc.returncode
    rec.update(returncode=code,wall_seconds=time.perf_counter()-t,ended_unix=time.time())
    if code==0:
        paths=list(directory.glob('*.json.gz'));assert len(paths)==1
        raw=read_json(paths[0]);assert raw['input_fingerprint']==item['input_fingerprint']
        assert all(raw['summary']['invariants'].values())
        assert all(sha(ROOT/f)==s for f,s in source.items())
        rec.update(status='complete',output=str(paths[0]),output_sha256=sha(paths[0]),
          windows=[dict(start_ms=w['start_ms'],end_ms=w['end_ms'],U=w['mean_npu_utilization'],
                        all_active=w['all_npus_active_whole_window']) for w in raw['windows']])
    elif rec['status']!='timeout':rec['status']='failed'
    write_json(path,rec)
    print(json.dumps(dict(finished=item['label'],strategy=strategy,status=rec['status'],
                         wall_seconds=rec['wall_seconds'],windows=rec.get('windows'))),flush=True)
    return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    specs=read_json(args.spec);jobs=[]
    for spec in specs:
        windows=spec.get('windows',[[2000,12000]]+[[a,a+2000] for a in range(2000,12000,2000)])
        for seed in spec.get('seeds',[7]):
            for mode in spec.get('modes',['ordered']):
                item=build(spec,mode,seed,args.out.resolve())
                for strategy in spec.get('strategies',['baseline']):jobs.append((item,strategy,windows))
    write_json(args.out/'plan.json',dict(spec=str(args.spec.resolve()),spec_sha256=sha(args.spec),
                                       jobs=[dict(input=i,strategy=s,windows=w) for i,s,w in jobs]))
    print(json.dumps(dict(prepared=len(jobs),requests=[i['request_count'] for i,_,_ in jobs])),flush=True)
    if args.run:
        rows=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fs=[pool.submit(run_job,i,s,args.out.resolve(),w) for i,s,w in jobs]
            for f in as_completed(fs):
                rows.append(f.result());write_json(args.out/'status.json',rows)
        assert all(r['status']=='complete' for r in rows)

if __name__=='__main__':main()
