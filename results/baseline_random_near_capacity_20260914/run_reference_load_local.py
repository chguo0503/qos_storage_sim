#!/usr/bin/env python3
"""Predeclared raw-data load sensitivity; independent full Random queues."""
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


def write(p, d):
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=2) + '\n')
    t.replace(p)


def main():
    definitions = [('reference_load100', '7,15'), ('reference_load105', '32,63'),
                   ('reference_load110', '17,31')]
    plan_path = HERE / 'reference_load_local_plan.json'
    assert not plan_path.exists()
    jobs = []
    for name, counts in definitions:
        argv = [sys.executable, str(HERE/'experiment.py'), '--name', name,
                '--profiles', '128:256,32:4096', '--counts', counts,
                '--roles', 'A,B', '--num-ssu', '3', '--seed', '7', '--horizon-ms', '22000']
        prepared = subprocess.run(argv + ['--prepare-only'], cwd=ROOT, capture_output=True, text=True)
        assert prepared.returncode == 0, prepared.stderr
        manifest = HERE/'inputs'/f'{name}_ssu3_h22000_seed7.json.gz'
        run = [sys.executable, str(HERE/'experiment.py'), '--manifest', str(manifest), '--strategy', 'baseline']
        if name == 'reference_load110':
            run.append('--trace')
        jobs.append(dict(name=name, counts=counts, prepare_argv=argv, manifest=str(manifest),
                         manifest_sha256=sha(manifest), argv=run))
    original = json.loads((HERE/'study_plan.json').read_text())
    source_sha = original['core_hashes']
    for name, digest in source_sha.items():
        path = ROOT/name
        assert path.exists() and sha(path) == digest, name
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        motivation='User clarified sustained Random U in the 80s is sufficient; local sensitivity around raw Saturday profiles.',
        interpretation='rho approximately 1.00, 1.05, 1.10; last is explicitly beyond initial 0.95-1.05 band. Capacity limitation is not FIFO-specific loss.',
        no_ordered=True, seed=7, max_parallel=3, jobs=jobs,
        runner_sha256=sha(HERE/'experiment.py'), queue_sha256=sha(Path(__file__)), core_sha256=source_sha)
    write(plan_path, plan)

    def run(job):
        assert sha(Path(job['manifest'])) == job['manifest_sha256']
        log = HERE/'execution_logs'/(job['name']+'_seed7.local.log')
        state = dict(name=job['name'], started_utc=datetime.now(timezone.utc).isoformat(), argv=job['argv'])
        status = log.with_suffix('.status.json')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with log.open('w') as f:
            p = subprocess.Popen(job['argv'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
            state.update(status='running', pid=p.pid); write(status, state)
            code = p.wait()
        state.update(status='complete' if code == 0 else 'failed', returncode=code,
                     ended_utc=datetime.now(timezone.utc).isoformat())
        write(status, state); print(json.dumps(state), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, jobs))
    write(HERE/'reference_load_local_results.json', results)
    assert all(r['returncode'] == 0 for r in results)
    for name, digest in source_sha.items():
        assert sha(ROOT/name) == digest, name


if __name__ == '__main__':
    main()
