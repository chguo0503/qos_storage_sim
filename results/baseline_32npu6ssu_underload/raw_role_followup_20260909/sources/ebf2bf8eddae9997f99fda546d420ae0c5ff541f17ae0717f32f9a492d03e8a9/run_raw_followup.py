#!/usr/bin/env python3
"""Frozen direct-data role/mixed probes; never modify simulation or policy code."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(ROOT), str(STUDY)]
import sim
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import profiles_for, save_manifest, read_json, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_multi_ssu_stall_experiments import input_demand
from run_coflow_experiments import source_files
from run_study import certificate

BLOCK = 176 * 1024 / 2**30


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(spec, seed, mode):
    keys = ','.join(f'{a}:{b}' for a,b in spec['short_keys']+[spec['long_key']])
    profiles, provenance = profiles_for('raw', keys)
    short = profiles[:-1]
    long = profiles[-1]
    nlong = spec['long_cards']
    target = spec.get('pure_horizon_ms', 5200)
    short_repeats = math.ceil(target / (8*sum(p['per_layer_compute_us']/1000 for p in short)))
    long_repeats = math.ceil(target / (8*long['per_layer_compute_us']/1000))
    population = []
    # The fixed-role workload defines the scientific population and placement.
    # Mixed reassigns this exact population; C/V/arrival/blocks do not change.
    for npu in range(32):
        deck = ([len(short)] * long_repeats if npu < nlong else
                [i for i in range(len(short)) for _ in range(short_repeats)])
        for pos,index in enumerate(deck):
            p = profiles[index]
            tokens = p['ssd_prefix_tokens']
            assert tokens % 128 == 0, 'Exact 176KiB raw alignment required; no padding'
            layer = tuple(((j+npu//4)%6, BLOCK) for j in range(tokens//128))
            volume = math.fsum(v for _,v in layer)
            assert math.isclose(volume, p['per_layer_kv_gib'], abs_tol=1e-12)
            rid=npu*1_000_000+pos
            load=dict(request_id=rid, npu_id=npu, generation=pos,
                seq_len_k=p['seq_len_k'], nql=p['nql'],
                category=sim.classify_request(p['seq_len_k'],p['nql']),
                per_layer_us=p['per_layer_compute_us'], per_layer_kv_gb=volume,
                required_bw_input_gbps=volume*1e6/p['per_layer_compute_us'],
                source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                arrival_time=0.0, arrival_ms=0.0, initial=True,
                role='long' if index==len(short) else 'short', profile_index=index,
                original_compute_us=p['per_layer_compute_us'], constructed_profile=False,
                profile_construction=p['construction'], padding_gib_per_layer=0.0,
                original_request_id=rid, source_original_npu_id=npu)
            population.append(ContinuousBatchRequest.from_normalized(rid,npu,0.0,load,(layer,)))
    lanes=[[] for _ in range(32)]
    if mode=='fixed':
        for r in population: lanes[r.npu_id].append(r)
    else:
        for index in range(len(profiles)):
            pool=sorted((r for r in population if r.load['profile_index']==index),key=lambda r:r.request_id)
            rng=random.Random(seed*1_000_003+index*10_007+283)
            rng.shuffle(pool)
            # Rotated targets keep remainder profiles from always favoring NPU0.
            targets=list(range(32))
            rng.shuffle(targets)
            for k,r in enumerate(pool): lanes[targets[k%32]].append(r)
    requests=[]
    assignments=[]
    for npu,lane in enumerate(lanes):
        lane.sort(key=lambda r:r.request_id)
        if not (mode=='fixed' and npu<nlong):
            random.Random(seed*1_000_003+npu*100_003+71923).shuffle(lane)
        pure=8*math.fsum(r.load['per_layer_us']/1000 for r in lane)
        assert pure>4000, (mode,npu,pure)
        assignments.append(dict(npu_id=npu,request_count=len(lane),pure_compute_ms=pure,
            profile_counts=[sum(r.load['profile_index']==i for r in lane) for i in range(len(profiles))]))
        for pos,old in enumerate(lane):
            rid=npu*1_000_000+pos
            load=dict(old.load,request_id=rid,npu_id=npu,generation=pos)
            requests.append(ContinuousBatchRequest.from_normalized(rid,npu,0.0,load,old.placement))
    requests=tuple(requests)
    assert Counter(r.load['original_request_id'] for r in requests)==Counter(r.request_id for r in population)
    proof=certificate(requests)
    if mode=='fixed': assert proof['passes'] and max(proof['per_ssu_upper_bound_gib_s'])<40
    label=f"{spec['name']}_{mode}_seed{seed}"
    fp=continuous_batch_input_fingerprint(requests)
    metadata=dict(experiment='raw_role_followup_20260909_v1',label=label,case_id=f'{label}_{fp[:12]}',
        num_npu=32,num_ssu=6,n_layers=8,seed=seed,family='raw',profiles=profiles,source=provenance,
        equal_176kib_blocks=True,blocks='exact',layout='preserved_source_stripe',order='independent_full_queue_shuffle',
        assignment_mode=mode,long_cards=nlong,short_cards=32-nlong,profile_keys=keys,
        input_fingerprint=fp,logical_input_fingerprint=logical_input_fingerprint(requests),
        active_profile_rate_certificate=proof,input_demand=input_demand(requests,32,6),
        load_within_disk_and_link_capacity=proof['passes'],compute_scale_actual=1.0,
        request_count=len(requests),source_profile_counts=[sum(r.load['profile_index']==i for r in population) for i in range(len(profiles))],
        per_npu_assignment=assignments,measurement_window_ms=[2000,4000],last_arrival_ms=0,
        source_data_sha256=sha(ROOT/'data'),construction_spec=spec,
        placement_rule='Each original fixed-role NPU uses (block_index+source_npu//4)%6; mixed preserves every original block verbatim.',
        population_rule='Fixed and mixed share the same finite population; per-role fixed queues have at least 5200ms pure compute, no replication of shuffled short decks.',
        slo_rule='Warm admissions [2000,4000), follow all to completion, completion-admission <=1.5*8*C; excludes arrival-to-admission queue.',
        caveat='Conditional mechanism probe: identical long C, all arrivals at zero, fixed long-role streams may remain synchronous. Not a production arrival sample.')
    return requests,metadata


def prepare(spec_path):
    specs=json.loads(spec_path.read_text())
    plan_path=HERE/'plans'/f'{spec_path.stem}.json'
    if plan_path.exists():
        plan=read_json(plan_path)
        assert plan['spec_sha256']==sha(spec_path)
        return plan
    paths={ROOT/name for name in source_files()}
    paths.update((ROOT/'data',ROOT/'run_baseline_npu32_stress.py',Path(__file__),STUDY/'run_study.py'))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    for path in paths:
        dest=HERE/'sources'/hashes[str(path.relative_to(ROOT))]/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copyfile(path,dest)
        assert sha(dest)==sha(path)
    items=[]
    for spec in specs:
        for seed in spec.get('seeds',[7]):
            for mode in spec.get('modes',['fixed','mixed']):
                requests,meta=build(spec,seed,mode)
                path=HERE/'inputs'/f"{meta['label']}.json.gz"
                save_manifest(path,requests,meta)
                item=dict(label=meta['label'],spec_name=spec['name'],seed=seed,mode=mode,
                    manifest=str(path),manifest_sha256=sha(path),input_fingerprint=meta['input_fingerprint'],
                    strategies=spec.get('strategies',['baseline','once']),
                    static_max_disk=max(meta['active_profile_rate_certificate']['per_ssu_upper_bound_gib_s']))
                items.append(item)
                print(json.dumps(dict(prepared=item['label'],requests=len(requests),static_max_disk=item['static_max_disk'])),flush=True)
    jobs=[dict(input=i,strategy=s) for i in items for s in i['strategies']]
    plan=dict(spec_file=str(spec_path),spec_sha256=sha(spec_path),inputs=items,jobs=jobs,source_sha256=hashes,
        created_utc=datetime.now(timezone.utc).isoformat(),seeds_are_predeclared=True,window_ms=[2000,4000])
    write_json(plan_path,plan)
    return plan


def run_job(job,plan):
    item,s=job['input'],job['strategy']
    directory=HERE/'runs'/item['label']/s
    directory.mkdir(parents=True,exist_ok=True)
    existing=list(directory.glob('*.json.gz'))
    if existing:
        assert len(existing)==1
        raw=read_json(existing[0])
        assert raw['input_fingerprint']==item['input_fingerprint'] and raw['strategy']==s
        assert all(raw['summary']['invariants'].values())
        return dict(label=item['label'],strategy=s,status='existing')
    assert sha(item['manifest'])==item['manifest_sha256']
    command=[sys.executable,'-B',str(ROOT/'run_baseline_npu32_stress.py'),'--manifest',item['manifest'],
        '--strategy',s,'--assignment','fixed','--window','2000:4000','--output',str(directory)]
    record=dict(label=item['label'],strategy=s,command=command,cwd=str(ROOT),
        input_sha256=item['manifest_sha256'],source_sha256=plan['source_sha256'],
        started_utc=datetime.now(timezone.utc).isoformat())
    start=time.perf_counter()
    with (directory/'stdout.log').open('w') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        record['pid']=p.pid
        write_json(directory/'command.json',record)
        try: code=p.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            code=p.returncode
    record.update(returncode=code,status='complete' if code==0 else 'failed',wall_seconds=time.perf_counter()-start)
    write_json(directory/'command.json',record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    plan=prepare(args.spec.resolve())
    if not args.run:return
    # Dispatch baseline first so numerical diagnostics arrive promptly.
    jobs=sorted(plan['jobs'],key=lambda j:j['strategy']!='baseline')
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,j,plan) for j in jobs]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            write_json(HERE/'status'/f'{args.spec.stem}.json',dict(finished=len(rows),total=len(jobs),rows=rows))
            print(json.dumps({k:row[k] for k in ('label','strategy','status','wall_seconds') if k in row}),flush=True)
    assert all(r['status'] in ('complete','existing') for r in rows)
    assert all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items())


if __name__=='__main__':main()
