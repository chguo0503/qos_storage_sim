#!/usr/bin/env python3
"""Execute a saved list of independent native experiment jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/random_multitype_search_20260914'
def run(job):
    name=job['name'];seed=job.get('seed',7);policy=job.get('policy','fifo');stage=job.get('stage','validation')
    dest=OUT/stage/f'{name}_seed{seed}_{policy}'
    metric=dest/'metrics.json'
    if metric.exists():return json.loads(metric.read_text())
    cmd=[sys.executable,'-u','run_random_multitype.py','--spec',job['spec'],'--case',name,
         '--seed',str(seed),'--policy',policy,'--stage',stage]
    if 'horizon_ms' in job:cmd.extend(['--horizon-ms',str(job['horizon_ms'])])
    if 'window_ms' in job:cmd.extend(['--window',*[str(v) for v in job['window_ms']]])
    logs=OUT/'batch_logs';logs.mkdir(exist_ok=True,parents=True)
    tag=f'{stage}_{name}_seed{seed}_{policy}'
    (logs/f'{tag}_command.json').write_text(json.dumps(cmd))
    with (logs/f'{tag}.log').open('w') as f:
        result=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{tag} failed with {result.returncode}; see log')
    return json.loads(metric.read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    jobs=json.loads(a.jobs.read_text())
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(run,j):j for j in jobs}
        for f in as_completed(fs):
            m=f.result()
            print(json.dumps({k:m[k] for k in ['name','seed','policy','U_percent','short_U_percent','long_U_percent','slo_1p5_percent','all_active','all_npus_both_roles_computed','wall_seconds']}),flush=True)
