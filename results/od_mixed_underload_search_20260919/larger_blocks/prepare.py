#!/usr/bin/env python3
"""Larger-read synthetic profiles and explicitly adversarial, legal hash IDs."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import sys
import time
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(STUDY),str(ROOT)]
from simulator.core import sim
from simulator.core.continuous_batch_sim import ContinuousBatchRequest,continuous_batch_input_fingerprint
from inputs.runners.run_baseline_npu32_stress import load_manifest,save_manifest
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from prepare_candidate import make_profiles
CORES=(8,10,12)
SPEC={'A':dict(blocks=180,disk1=62,wanted=2688),'B':dict(blocks=1039,disk1=360,wanted=656)}
BLOCK=176*1024/2**30

def worker(i):
    os.sched_setaffinity(0,{CORES[i]});pool={'A':[],'B':[]};quota={r:SPEC[r]['wanted']//3+(i<SPEC[r]['wanted']%3)for r in SPEC};j=0;calls=0;start=time.time()
    while any(len(pool[r])<quota[r]for r in SPEC):
        rid=200000000+i+3*j;j+=1;needb=len(pool['B'])<quota['B'];limit=1039 if needb else 180;counts=[0,0,0];aa=None
        for k in range(limit):
            counts[sim.block_ring_hash_disk_id(rid,k,3)]+=1
            if k==179:aa=counts.copy()
        calls+=limit
        if needb and counts[1]==360 and max(counts)<=360:pool['B'].append(dict(request_id=rid,counts=counts))
        elif len(pool['A'])<quota['A']and aa[1]==62 and max(aa)<=62:pool['A'].append(dict(request_id=rid,counts=aa))
    return dict(cpu=CORES[i],pools=pool,tested=j,hash_calls=calls,wall_s=time.time()-start)

def main():
    start=time.time();before=hashlib.sha256((ROOT/'simulator/core/sim.py').read_bytes()).hexdigest();sim._block_hash_ring(3)
    path=HERE/'address_pool.json'
    if path.exists():pool=json.loads(path.read_text())['pools']
    else:
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('fork'))as ex:workers=list(ex.map(worker,range(3)))
        pool={r:sorted([z for w in workers for z in w['pools'][r]],key=lambda z:z['request_id'])for r in SPEC}
        ids=[z['request_id']for r in SPEC for z in pool[r]];assert len(ids)==len(set(ids))
        assert hashlib.sha256((ROOT/'simulator/core/sim.py').read_bytes()).hexdigest()==before
        path.write_text(json.dumps(dict(classification='Synthetic profiles + adversarial valid physical-ID selection; not ordinary random input',
            specs=SPEC,pools=pool,workers=[{k:v for k,v in w.items()if k!='pools'}for w in workers],
            source_sim_sha256=before,placement_function_modified=False,all_ids_unique=True,wall_s=time.time()-start),indent=2))
    base_spec=dict(label='larger180_1039_base',seed=7,profiles={'A':dict(block_count=180,C_ms=4.153639478141864),'B':dict(block_count=1039,C_ms=74.98957484197959)})
    profiles=make_profiles(base_spec);requests=[];cursor={'A':0,'B':0};prefix=[2,4,1,3]
    _,meta=load_manifest(STUDY/'runs/fixedssu1_safe38_od_local/manifest.json.gz')
    for n in range(32):
        q=['B']*prefix[n//8]+(['A']*14+['B']*3)*6
        for pos,role in enumerate(q):
            row=pool[role][cursor[role]];cursor[role]+=1;p=profiles[role];rid=n*1000000+pos
            placement=tuple((sim.block_ring_hash_disk_id(row['request_id'],k,3),BLOCK)for k in range(p['block_count']))
            actual=Counter(d for d,b in placement);assert [actual[s]for s in range(3)]==row['counts']
            load=dict(request_id=rid,npu_id=n,generation=pos,original_request_id=row['request_id'],role=role,
              seq_len_k=p['seq_len_k'],total_tokens=p['total_tokens'],nql=p['nql'],ssd_prefix_tokens=p['ssd_prefix_tokens'],category=p['category'],
              per_layer_us=p['per_layer_compute_us'],per_layer_kv_gb=p['per_layer_kv_gib'],required_bw_input_gbps=p['required_bandwidth_gibps'],
              source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],original_compute_us=p['original_compute_us'],constructed_profile=True,
              profile_construction=p['construction'],padding_gib_per_layer=0.,arrival_time=0.,arrival_ms=0.,initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid,n,0.,load,(placement,)))
    requests=tuple(requests);assert cursor=={r:SPEC[r]['wanted']for r in SPEC}
    per_npu=[]
    for n in range(32):
        qs=[q for q in requests if q.npu_id==n]
        rates=[[sum(v for d,v in q.placement[0]if d==s)*1e6/q.load['per_layer_us']for s in range(3)]for q in qs]
        per_npu.append(dict(npu_id=n,role_counts=dict(Counter(q.load['role']for q in qs)),request_count=len(qs),
            pure_compute_ms=sum(8*q.load['per_layer_us']/1000 for q in qs),per_ssu_rate_max=[max(r[s]for r in rates)for s in range(3)],queue=''.join(q.load['role']for q in qs)))
    expected=sum(len(q.placement[0])*8 for q in requests)
    meta=dict(meta,label=base_spec['label'],case_id=base_spec['label'],order='larger_block_adversarial_static_queue',
       candidate_spec=base_spec,address_selection=dict(method='SSU1 fixed A62/B360; other disks no larger; unique legal IDs >=200000000',
         is_random_address_population=False,pool_file='larger_blocks/address_pool.json',pool_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),counts_used=cursor,placement_function_unchanged=True),
       constructed_profile=True,request_count=len(requests),input_fingerprint=continuous_batch_input_fingerprint(requests),
       logical_input_fingerprint=logical_input_fingerprint(requests),profile_source='synthetic_equal_176KiB_blocks',
       placement_identity_key='original_request_id',equal_176kib_blocks=True,layout=sim.PLACEMENT_BLOCK_RING_HASH,
       profiles=profiles,per_npu_assignment=per_npu,blocks=expected,expected_blocks=expected,
       identity_rule='Unique adversarial hash IDs from the independent larger_blocks/address_pool.json; runtime ID is npu*1e6+queue_position',
       static_per_ssu_upper_bound_GiB_s=[sum(q['per_ssu_rate_max'][s]for q in per_npu)for s in range(3)],
       caveat='Synthetic computation and adversarial native Ring Hash addresses; no claim that data contains these exact compute values')
    save_manifest(HERE/'inputs/larger180_1039_base.json.gz',requests,meta)
    grid=[(12,1.03,.98),(12,1.05,.98),(12,1.07,.98),(13,.99,1.02),(13,1.01,1.02),(13,1.03,1.02),(14,.97,1.06),(14,1.,1.06),(12,1.03,1.),(13,1.,1.)]
    plan=[]
    for j,(m,sa,sb)in enumerate(grid):
        spec=dict(label=f'larger180_grid_{j:02}',a_run=m,b_run=3,C_A_ms=base_spec['profiles']['A']['C_ms']*sa,C_B_ms=base_spec['profiles']['B']['C_ms']*sb,
                  construction='Synthetic C and adversarial fixed bottleneck disk counts; no runtime barrier',scale_A=sa,scale_B=sb)
        (HERE/'specs'/f'{spec["label"]}.json').write_text(json.dumps(spec,indent=2));plan.append(spec)
    (HERE/'plan.json').write_text(json.dumps(dict(base_manifest='inputs/larger180_1039_base.json.gz',cores=list(CORES),candidates=plan,wall_s=time.time()-start),indent=2))
    print(json.dumps(dict(prepared=True,requests=len(requests),address_pool={r:len(pool[r])for r in SPEC},wall_s=time.time()-start)),flush=True)

if __name__=='__main__':main()
