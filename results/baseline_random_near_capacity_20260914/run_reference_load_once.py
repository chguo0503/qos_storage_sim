#!/usr/bin/env python3
"""Paired Once sensitivity on two raw manifests, declared before pilot outcomes."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = HERE/'reference_load_once_plan.json'
    assert not output.exists()
    jobs=[]
    for name in ('reference_load105', 'reference_load110'):
        m=HERE/'inputs'/f'{name}_ssu3_h22000_seed7.json.gz'
        assert m.exists()
        jobs.append(dict(name=name, manifest=str(m), sha256=sha(m),
            argv=[sys.executable,str(HERE/'experiment.py'),'--manifest',str(m),'--strategy','once']))
    plan=dict(created_utc=datetime.now(timezone.utc).isoformat(), max_parallel=2, jobs=jobs,
        reason='Same input baseline/Once comparison separates general capacity limits from strategy-sensitive losses; selected before Baseline pilots finish.',
        source_plan_sha256=sha(HERE/'reference_load_local_plan.json'),
        runner_sha256=sha(HERE/'experiment.py'),queue_sha256=sha(Path(__file__)))
    output.write_text(json.dumps(plan,indent=2)+'\n')

    def run(job):
        assert sha(Path(job['manifest']))==job['sha256']
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        started=datetime.now(timezone.utc).isoformat()
        with (HERE/'execution_logs'/(job['name']+'_seed7_once.local.log')).open('w') as f:
            p=subprocess.run(job['argv'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        r=dict(name=job['name'],returncode=p.returncode,started_utc=started,
               ended_utc=datetime.now(timezone.utc).isoformat())
        print(json.dumps(r),flush=True);return r

    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,jobs))
    (HERE/'reference_load_once_results.json').write_text(json.dumps(results,indent=2)+'\n')
    assert all(r['returncode']==0 for r in results)


if __name__=='__main__':main()
