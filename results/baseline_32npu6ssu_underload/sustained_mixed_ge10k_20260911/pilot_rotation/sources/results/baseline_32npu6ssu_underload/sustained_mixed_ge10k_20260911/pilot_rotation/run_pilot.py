#!/usr/bin/env python3
"""Two prespecified Baseline probes of repeated, unsynchronised role exchange.

Only the input queue is constructed. No simulation core or runtime state is
changed. The two cohorts start at different positions in a fixed periodic
queue; there is no simulated clock barrier, sleep, reset, or adaptive order.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import save_manifest, read_json
from run_shared_path_experiments import logical_input_fingerprint
from run_coflow_experiments import source_files

BLOCK = 176*1024/2**30
PROFILES = {'L':(160,1024),'S':(32,1024)}
CONFIGS = [{'label':'rotation16_L4_S12_seed7','long_per_segment':4,'short_per_segment':12,'cycles':8},
           {'label':'rotation16_L8_S24_seed7','long_per_segment':8,'short_per_segment':24,'cycles':4}]
WINDOWS = [(2000,12000),(2000,4000),(4000,6000),(6000,8000),(8000,10000),(10000,12000),(4000,12000)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def build(spec):
    data = ast.literal_eval((ROOT/'data').read_text())
    profiles = {}
    for role,key in PROFILES.items():
        bw,c,ttft,v = data[key]
        count = round(v/BLOCK)
        assert key[0]>10 and math.isclose(count*BLOCK,v,abs_tol=1e-12)
        profiles[role] = {'key':list(key),'C_ms':c/1000,'V_gib':v,'blocks':count,
                          'source_required_bw':bw,'source_ttft_78layer_ms':ttft}
    requests, lanes, identities = [], [], {}
    for npu in range(32):
        L=['L']*spec['long_per_segment'];S=['S']*spec['short_per_segment']
        template=(L+S) if npu<16 else (S+L)
        roles=template*spec['cycles'];seen=Counter();lane=[]
        for pos,role in enumerate(roles):
            p=profiles[role];rid=npu*1_000_000+pos
            original_id=npu*1_000_000+(0 if role=='L' else 32)+seen[role];seen[role]+=1
            one=tuple(((j+npu//4)%6,BLOCK) for j in range(p['blocks']))
            placement=(one,)*8
            load={'request_id':rid,'npu_id':npu,'generation':pos,'original_request_id':original_id,
                  'original_compute_us':p['C_ms']*1000,'per_layer_us':p['C_ms']*1000,
                  'per_layer_kv_gb':p['V_gib'],'source_per_layer_kv_gib':p['V_gib'],
                  'physical_per_layer_kv_gib':p['blocks']*BLOCK,'padding_gib_per_layer':0.0,
                  'seq_len_k':p['key'][0],'nql':p['key'][1],
                  'category':'LL' if role=='L' else 'SL','role':'long' if role=='L' else 'short',
                  'required_bw_input_gbps':p['source_required_bw'],'source_ttft_ms':p['source_ttft_78layer_ms'],
                  'constructed_profile':False,'initial':True,'arrival_ms':0.0,'arrival_time':0.0}
            request=ContinuousBatchRequest.from_normalized(rid,npu,0.0,load,placement)
            requests.append(request);lane.append(request)
            identities[original_id]={'npu':npu,'role':role,'C_ms':p['C_ms'],'V_gib':p['V_gib'],'placement':placement,'arrival':0.0}
        pure=math.fsum(8*r.load['per_layer_us']/1000 for r in lane)
        assert seen=={'L':32,'S':96} and pure>12000+2*8*profiles['L']['C_ms']
        lanes.append({'npu':npu,'cohort':0 if npu<16 else 1,'template':''.join(template),
                      'counts':dict(seen),'cycles':spec['cycles'],'pure_compute_ms':pure})
    requests=tuple(requests)
    # Universal sufficient certificate: independently maximize each NPU's
    # actual stored per-SSD V/C over its possible profiles, then sum the 32 maxima.
    per_ssu=[]
    for disk in range(6):
        per_ssu.append(math.fsum(max(sum(BLOCK for j in range(p['blocks']) if (j+n//4)%6==disk)*1000/p['C_ms']
                                    for p in profiles.values()) for n in range(32)))
    max_link=max(p['V_gib']*1000/p['C_ms'] for p in profiles.values())
    assert max(per_ssu)<40 and max_link<50
    fingerprint=continuous_batch_input_fingerprint(requests)
    meta={'experiment':'sustained_mixed_ge10k_role_rotation_pilot_v1','family':'raw',
          'label':spec['label'],'case_id':spec['label']+'_'+fingerprint[:12],
          'num_npu':32,'num_ssu':6,'n_layers':8,'seed':7,'equal_176kib_blocks':True,
          'blocks':'exact','layout':'stripe','profile_keys':[list(PROFILES['L']),list(PROFILES['S'])],
          'profiles':profiles,'order':'periodic_fixed_cohort_queues','assignment_mode':'fixed_original_npu',
          'input_fingerprint':fingerprint,'logical_input_fingerprint':logical_input_fingerprint(requests),
          'source_data_sha256':sha(ROOT/'data'),'construction_spec':spec,'per_npu':lanes,
          'static_capacity_certificate':{'per_ssu_upper_gib_s':per_ssu,'max_npu_link_gib_s':max_link,'passed':True,
              'scope':'Every possible active profile combination under these stored per-NPU placements; excludes extra next-request L0 by study definition.'},
          'horizon_ms':12000,'measurement_windows_ms':WINDOWS,'request_count':len(requests),
          'placement_rule':'(block_index+npu//4)%6; same placement per original request across both segment-length pilots.',
          'population_rule':'Each NPU has 32 raw L160K1024 and 96 raw S32K1024; only fixed queue order changes.',
          'timing_rule':'All arrivals zero; no runtime barriers, clock-based role assignment, sleeps, or resets.',
          'pilot_limitation':'Single Short profile mechanism probe. Does not yet satisfy the requested varied Short robustness study.',
          'period_prediction':'3 Short layers per Long C approximately matches observed fixed16 Short active utilization; entrainment is a hypothesis, not a runtime constraint.',
          'slo_rule':'Each window selects admissions then follows to completion; completion-admission <=1.5*8*raw layer C.'}
    return requests,meta,identities


def prepare():
    path=HERE/'plan.json'
    if path.exists():
        plan=read_json(path)
        assert plan['source_sha256'][str(Path(__file__).relative_to(ROOT))]==sha(__file__)
        return plan
    hashes={name:sha(ROOT/name) for name in source_files()}
    for p in [ROOT/'data',ROOT/'run_baseline_npu32_stress.py',Path(__file__)]:hashes[str(p.relative_to(ROOT))]=sha(p)
    for name,digest in hashes.items():
        dest=HERE/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        assert sha(dest)==digest
    inputs=[];population=None
    for spec in CONFIGS:
        requests,meta,identity=build(spec)
        if population is None:population=identity
        else:assert identity==population
        mp=HERE/'inputs'/f'{spec["label"]}.json.gz';save_manifest(mp,requests,meta)
        item={'label':spec['label'],'manifest':str(mp),'manifest_sha256':sha(mp),
              'input_fingerprint':meta['input_fingerprint'],'spec':spec,'request_count':len(requests),
              'pure_compute_ms_per_npu':meta['per_npu'][0]['pure_compute_ms'],
              'capacity_certificate':meta['static_capacity_certificate']}
        inputs.append(item);print(json.dumps({'prepared':item}),flush=True)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':7,'windows_ms':WINDOWS,
          'workers':2,'inputs':inputs,'jobs':[{'input':i,'strategy':'baseline'} for i in inputs],
          'same_per_npu_complete_population_and_placement':True,'source_sha256':hashes,
          'max_simulations':2,'python':sys.version,'executable':sys.executable}
    write(path,plan);return plan


def run_job(job,plan):
    i=job['input'];out=HERE/'runs'/i['label']/'baseline';out.mkdir(parents=True,exist_ok=True)
    cp=out/'command.json'
    if cp.exists():raise FileExistsError('Preserve existing run; do not retry silently: '+str(cp))
    assert sha(i['manifest'])==i['manifest_sha256']
    assert all(sha(ROOT/k)==v for k,v in plan['source_sha256'].items())
    cmd=[sys.executable,'-B',str(ROOT/'run_baseline_npu32_stress.py'),'--manifest',i['manifest'],
         '--strategy','baseline','--assignment','fixed']
    for a,z in WINDOWS:cmd+=['--window',f'{a}:{z}']
    cmd+=['--output',str(out)]
    record={'label':i['label'],'strategy':'baseline','status':'running','command':cmd,'cwd':str(ROOT),
            'source_sha256':plan['source_sha256'],'input_sha256':i['manifest_sha256'],
            'started_utc':datetime.now(timezone.utc).isoformat()};started=time.perf_counter()
    with (out/'stdout.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record['pid']=p.pid;write(cp,record);print(json.dumps({'started':i['label'],'pid':p.pid}),flush=True)
        rc=p.wait()
    record.update(status='complete' if rc==0 else 'failed',returncode=rc,
                  wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(cp,record);print(json.dumps({k:record[k] for k in ['label','status','returncode','wall_seconds']}),flush=True)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');args=ap.parse_args()
    plan=prepare()
    if args.run:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda job:run_job(job,plan),plan['jobs']))
        write(HERE/'status.json',{'complete':sum(x['status']=='complete' for x in results),'runs':results})


if __name__=='__main__':main()
