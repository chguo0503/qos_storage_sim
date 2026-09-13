#!/usr/bin/env python3
"""Run only the explicitly frozen remote Baseline Random queue, six jobs maximum."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LOCK = threading.Lock()
PLAN = HERE/'audit_remote_queue_plan.json'
STATE = HERE/'audit_remote_queue_status.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def verify_sources(plan):
    for name,expected in plan['frozen_source_sha256'].items():
        assert sha(ROOT/name)==expected, name


def run_one(job, plan, state):
    label=job['label'];manifest=ROOT/job['manifest']
    assert sha(manifest)==job['manifest_sha256']
    output=HERE/'runs'/label/'baseline'
    assert not output.exists(), f'Preserving existing remote run: {output}'
    log=HERE/'execution_logs'/(label+'.remote.log')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter()
    with log.open('w') as stream:
        p=subprocess.Popen([sys.executable,str(HERE/'experiment.py'),'--manifest',str(manifest),
                            '--strategy','baseline'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        with LOCK:
            state[label].update(status='running',pid=p.pid,started_utc=utc(),log=str(log))
        code=p.wait()
    command=output/'command.json'
    record=json.loads(command.read_text()) if command.exists() else {}
    if code==0:
        assert record['status']=='complete' and record['returncode']==0
        assert record['manifest_sha256']==job['manifest_sha256']
        assert record['input_fingerprint']==job['input_fingerprint']
        assert sha(output/'result.json.gz')==record['output_sha256']
        assert sha(output/'physical_service.json')==record['physical_service_sha256']
    with LOCK:
        state[label].update(status='complete' if code==0 else 'failed',returncode=code,
                            ended_utc=utc(),wall_seconds=time.perf_counter()-start,
                            windows=record.get('windows'),command_sha256=sha(command) if command.exists() else None)
    return label


def main():
    plan=json.loads(PLAN.read_text())
    assert plan['max_parallel']==6 and len(plan['jobs'])==21
    assert len({j['label'] for j in plan['jobs']})==21
    assert all(j['strategy']=='baseline' and j['order']=='random' and not j['trace'] for j in plan['jobs'])
    verify_sources(plan)
    (HERE/'execution_logs').mkdir(exist_ok=True)
    state={j['label']:{'status':'pending','priority':i} for i,j in enumerate(plan['jobs'])}
    meta={'host':platform.node(),'python':sys.version,'pid':os.getpid(),'started_utc':utc(),
          'plan_sha256':sha(PLAN),'queue_runner_sha256':sha(Path(__file__)),
          'max_parallel':6,'job_count':21,'source_hashes_verified_before':True}
    def publish():
        with LOCK:
            snapshot={k:dict(v) for k,v in state.items()}
        write(STATE,dict(meta,updated_utc=utc(),jobs=snapshot,
                        complete=sum(v['status']=='complete' for v in snapshot.values()),
                        failed=sum(v['status']=='failed' for v in snapshot.values()),
                        running=sum(v['status']=='running' for v in snapshot.values())))
    publish()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run_one,job,plan,state):job['label'] for job in plan['jobs']}
        pending=set(futures)
        while pending:
            done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                label=futures[future]
                try:
                    future.result()
                except BaseException as exc:
                    with LOCK:
                        state[label].update(status='failed',error=repr(exc),ended_utc=utc())
                print(json.dumps(dict(label=label,**state[label]),ensure_ascii=False),flush=True)
            publish()
    verify_sources(plan)
    meta.update(ended_utc=utc(),source_hashes_verified_after=True)
    publish()
    if any(v['status']!='complete' for v in state.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
