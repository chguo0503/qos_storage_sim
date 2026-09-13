#!/usr/bin/env python3
"""Two-worker local queue for four predeclared constructed pilot inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def main():
    names = ['fit6_m2048_s10m128', 'fit8_m2048_s10m128',
             'fit6_m4096_s10m128', 'fit8_m4096_s10m128']
    jobs = []
    for name in names:
        disks = 6 if name.startswith('fit6') else 8
        manifest = HERE/'inputs'/f'{name}_ssu{disks}_h22000_seed7.json.gz'
        assert manifest.is_file()
        argv = [sys.executable, str(HERE/'experiment.py'), '--manifest', str(manifest), '--strategy', 'baseline']
        # Keep one representative trace per long profile; the 8-disk cases
        # remain cheap screening runs and can be replayed later if needed.
        if disks == 6:
            argv.append('--trace')
        jobs.append(dict(name=name, manifest=str(manifest), manifest_sha256=sha(manifest), argv=argv))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), max_parallel=2, seed=7,
        family='data_affine_extrapolation', no_ordered=True,
        runner_sha256=sha(HERE/'experiment.py'), builder_sha256=sha(HERE/'prepare_constructed.py'),
        candidate_plan_sha256=sha(HERE/'constructed_candidates.json'), jobs=jobs)
    plan_path = HERE/'constructed_local_queue_plan.json'
    assert not plan_path.exists()
    write(plan_path, plan)

    def run(job):
        assert sha(Path(job['manifest']))==job['manifest_sha256']
        assert sha(HERE/'experiment.py')==plan['runner_sha256']
        assert sha(HERE/'prepare_constructed.py')==plan['builder_sha256']
        log = HERE/'execution_logs'/(job['name']+'_seed7.local.log')
        status = log.with_suffix('.status.json')
        state = dict(name=job['name'], started_utc=datetime.now(timezone.utc).isoformat(), argv=job['argv'])
        write(status, state)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('w') as stream:
            process = subprocess.Popen(job['argv'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            state.update(pid=process.pid, status='running'); write(status, state)
            code = process.wait()
        state.update(returncode=code, status='complete' if code==0 else 'failed', ended_utc=datetime.now(timezone.utc).isoformat())
        write(status, state)
        print(json.dumps(state), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, jobs))
    write(HERE/'constructed_local_queue_results.json', results)
    assert all(r['returncode']==0 for r in results)


if __name__=='__main__':
    main()
