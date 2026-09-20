#!/usr/bin/env python3
"""Launch exactly three original-Once controls on CPU 0, 2 and 4."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    inputs = ROOT/'results/continuous_underload_asu_od_20260918/inputs'
    status_path = HERE/'under_once_queue_status.json'
    plan_path = HERE/'under_once_plan.json'
    assert not status_path.exists() and not plan_path.exists()
    watched = [HERE/'run_trial.py', HERE/'metrics.py', Path(__file__)]
    watched += [inputs/f'random_seed{s}_ring_hash.json.gz' for s in (7,19,43)]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in watched}
    jobs = []
    for seed,cpu in zip((7,19,43),(0,2,4)):
        label = f'under_once_seed{seed}_local'
        argv = ['taskset','-c',str(cpu),sys.executable,'-u',str((HERE/'run_trial.py').relative_to(ROOT)),
                '--manifest',str((inputs/f'random_seed{seed}_ring_hash.json.gz').relative_to(ROOT)),
                '--policy','once','--label',label]
        jobs.append(dict(label=label,seed=seed,cpu_id=cpu,argv=argv))
    plan = dict(created_utc=now(),source_sha256=sources,jobs=jobs)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    status = dict(started_utc=now(),queue_pid=os.getpid(),complete=False,
                  all_successful=False,source_sha256=sources,jobs={})
    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
    running = {}
    logs = HERE/'execution_logs';logs.mkdir(exist_ok=True)
    for job in jobs:
        logfile = logs/(job['label']+'.log')
        stream = logfile.open('x')
        process = subprocess.Popen(job['argv'],cwd=ROOT,stdin=subprocess.DEVNULL,
                                   stdout=stream,stderr=subprocess.STDOUT)
        running[job['label']] = (process,stream,time.perf_counter())
        status['jobs'][job['label']] = dict(**job,pid=process.pid,status='running',
                                            started_utc=now(),log=str(logfile.relative_to(ROOT)))
    save();print(json.dumps(status),flush=True)
    last_print = time.perf_counter()
    while running:
        for label,(process,stream,started) in tuple(running.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                status['jobs'][label].update(status='complete' if code==0 else 'failed',returncode=code,
                                            ended_utc=now(),wall_seconds=time.perf_counter()-started)
                del running[label];save()
        if time.perf_counter()-last_print>=15:
            progress = {}
            for label in status['jobs']:
                p = HERE/'runs'/label/'progress.json'
                progress[label] = json.loads(p.read_text()) if p.exists() else dict(status=status['jobs'][label]['status'])
            print(json.dumps(dict(updated_utc=now(),progress=progress)),flush=True)
            status['updated_utc']=now();save();last_print=time.perf_counter()
        if running:time.sleep(2)
    unchanged = all(sha(ROOT/name)==digest for name,digest in sources.items())
    status.update(complete=True,all_successful=all(j['returncode']==0 for j in status['jobs'].values()),
                  sources_verified_after=unchanged,ended_utc=now())
    save();print(json.dumps(dict(complete=True,all_successful=status['all_successful'],sources_verified_after=unchanged)),flush=True)
    assert status['all_successful'] and unchanged


if __name__=='__main__':main()
