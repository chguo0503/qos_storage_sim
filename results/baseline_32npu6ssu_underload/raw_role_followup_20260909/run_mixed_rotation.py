#!/usr/bin/env python3
"""Reorder the frozen mixed per-NPU population; preserve C/V/arrivals/placement."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

FOLLOWUP=Path(__file__).resolve().parent
HERE=FOLLOWUP/'mixed_rotation'
ROOT=FOLLOWUP.parent.parents[1]
sys.path[:0]=[str(ROOT),str(FOLLOWUP),str(FOLLOWUP.parent)]
import run_raw_followup as runner
from run_baseline_npu32_stress import load_manifest,save_manifest,read_json,write_json
from continuous_batch_sim import ContinuousBatchRequest,continuous_batch_input_fingerprint
from run_shared_path_experiments import logical_input_fingerprint
from run_coflow_experiments import source_files


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(spec_path):
    plan_path=HERE/'plans'/f'{spec_path.stem}.json'
    if plan_path.exists():
        p=read_json(plan_path);assert p['spec_sha256']==sha(spec_path);return p
    import mixed_design as design
    items=[]
    for spec in read_json(spec_path):
        for seed in spec.get('seeds',[7]):
            source=FOLLOWUP/'inputs'/f"{spec.get('source_name','raw200_three_l20')}_mixed_seed{seed}.json.gz"
            original,meta=load_manifest(source)
            new=[];descriptions=[]
            for npu in range(32):
                lane=sorted((r for r in original if r.npu_id==npu),key=lambda r:r.request_id)
                ordered,desc=design.order_lane(lane,meta,npu,seed,spec['variant'])
                assert sorted(r.request_id for r in lane)==sorted(r.request_id for r in ordered)
                descriptions.append(dict(npu_id=npu,**desc))
                for pos,old in enumerate(ordered):
                    rid=npu*1_000_000+pos
                    load=dict(old.load,request_id=rid,npu_id=npu,generation=pos,
                        source_mixed_request_id=old.request_id)
                    new.append(ContinuousBatchRequest.from_normalized(rid,npu,old.arrival_time_ms,load,old.placement))
            new=tuple(new)
            fp=continuous_batch_input_fingerprint(new)
            label=f"{spec.get('source_name','raw200_three_l20')}_{spec['variant']}_seed{seed}"
            metadata=dict(meta,label=label,case_id=f'{label}_{fp[:12]}',input_fingerprint=fp,
                logical_input_fingerprint=logical_input_fingerprint(new),assignment_mode='mixed_rotation',
                order=spec['variant'],source_mixed_manifest=str(source),source_mixed_manifest_sha256=sha(source),
                source_mixed_input_fingerprint=meta['input_fingerprint'],per_npu_order_design=descriptions,
                order_design_source_sha256=sha(design.__file__),
                ordering_scope='Only permutes each existing mixed NPU queue; same per-card/global population, placement, C/V, arrivals. No sleeps, delays, or admission control.')
            path=HERE/'inputs'/f'{label}.json.gz'
            save_manifest(path,new,metadata)
            items.append(dict(label=label,spec_name=spec['variant'],seed=seed,mode='mixed_rotation',
                manifest=str(path),manifest_sha256=sha(path),input_fingerprint=fp,
                strategies=spec.get('strategies',['baseline']),source_manifest=str(source),source_sha256=sha(source)))
            print(json.dumps(dict(prepared=label,requests=len(new))),flush=True)
    paths={ROOT/name for name in source_files()}
    paths.update((ROOT/'data',ROOT/'run_baseline_npu32_stress.py',Path(__file__),Path(design.__file__),Path(runner.__file__)))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for p in paths:
        target=HERE/'sources'/hashes[str(p.relative_to(ROOT))]/p.name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(p,target)
        assert sha(target)==sha(p)
    p=dict(spec_file=str(spec_path),spec_sha256=sha(spec_path),source_sha256=hashes,inputs=items,
        jobs=[dict(input=i,strategy=s) for i in items for s in i['strategies']],
        created_utc=datetime.now(timezone.utc).isoformat(),window_ms=[2000,4000],
        hypothesis='Reproduce concentrated Long I/O with rotating roles; target mean active Long/Short cards near20/12 and each NPU warm mixed. Targets are measured, not enforced by fake wall-clock delays.')
    write_json(plan_path,p);return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--run',action='store_true')
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    plan=prepare(args.spec.resolve())
    if not args.run:return
    runner.HERE=HERE
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(runner.run_job,j,plan) for j in plan['jobs']]
        for f in as_completed(fs):
            row=f.result();rows.append(row)
            write_json(HERE/'status'/f'{args.spec.stem}.json',dict(finished=len(rows),total=len(plan['jobs']),rows=rows))
            print(json.dumps({k:row[k] for k in ('label','strategy','status','wall_seconds') if k in row}),flush=True)
    assert all(r['status'] in ('existing','complete') for r in rows)


if __name__=='__main__':main()
