#!/usr/bin/env python3
"""Two separately selected Once pairs, after the frozen Baseline queue has room."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_sources(plan):
    assert all(sha(ROOT / name) == digest for name, digest in plan['frozen_source_sha256'].items())


def main():
    plan_path = HERE / 'audit_remote_once_plan.json'
    plan = read(plan_path)
    assert plan['max_once_parallel'] == 2 and plan['remote_simulation_limit'] == 6
    assert len(plan['jobs']) == 2 and len({job['label'] for job in plan['jobs']}) == 2
    assert all(job['strategy'] == 'once' and job['order'] == 'random' and not job['trace'] for job in plan['jobs'])
    verify_sources(plan)
    for job in plan['jobs']:
        assert sha(ROOT / job['manifest']) == job['manifest_sha256']
        assert not (HERE / 'runs' / job['label'] / 'once').exists()
    state = {job['label']: {'status': 'pending'} for job in plan['jobs']}
    pending = list(plan['jobs'])
    active = {}
    meta = {'host': platform.node(), 'python_version': sys.version, 'pid': os.getpid(),
            'started_utc': utc(), 'plan_sha256': sha(plan_path),
            'queue_runner_sha256': sha(Path(__file__)), 'job_count': 2,
            'max_once_parallel': 2, 'remote_simulation_limit': 6,
            'source_hashes_verified_before': True,
            'selection': plan['selection']}

    def publish():
        write(HERE / 'audit_remote_once_status.json', dict(meta, updated_utc=utc(), jobs=state,
              complete=sum(v['status'] == 'complete' for v in state.values()),
              failed=sum(v['status'] == 'failed' for v in state.values()),
              running=len(active)))

    publish()
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (HERE / 'execution_logs').mkdir(exist_ok=True)
    while pending or active:
        for label, (process, stream, started, job) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            record_path = HERE / 'runs' / label / 'once' / 'command.json'
            record = read(record_path) if record_path.exists() else {}
            error = None
            try:
                assert code == 0 and record['status'] == 'complete' and record['returncode'] == 0
                assert record['completed_simulation'] and record['strategy'] == 'once'
                assert record['manifest_sha256'] == job['manifest_sha256']
                assert record['input_fingerprint'] == job['input_fingerprint']
                assert sha(record_path.parent / 'result.json.gz') == record['output_sha256']
                assert sha(record_path.parent / 'physical_service.json') == record['physical_service_sha256']
            except BaseException as exc:
                error = repr(exc)
            state[label].update(status='complete' if error is None else 'failed', returncode=code,
                                ended_utc=utc(), wall_seconds=time.perf_counter() - started,
                                command_sha256=sha(record_path) if record_path.exists() else None,
                                windows=record.get('windows'), error=error)
            del active[label]
            print(json.dumps(dict(label=label, **state[label]), ensure_ascii=False), flush=True)
        baseline = read(HERE / 'audit_remote_queue_status.json')
        assert baseline['plan_sha256'] == plan['baseline_queue_plan_sha256']
        # Once there are no pending Baseline jobs, its running count can only
        # decrease. A stale published count thus overestimates occupied slots.
        old_pending = sum(v['status'] == 'pending' for v in baseline['jobs'].values())
        occupied = baseline['running']
        while pending and old_pending == 0 and len(active) < 2 and occupied + len(active) < 6:
            job = pending.pop(0)
            label = job['label']
            log = HERE / 'execution_logs' / (label + '.once.remote.log')
            stream = log.open('w')
            started = time.perf_counter()
            process = subprocess.Popen([sys.executable, str(HERE / 'experiment.py'), '--manifest',
                                        str(ROOT / job['manifest']), '--strategy', 'once'],
                                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[label] = (process, stream, started, job)
            state[label].update(status='running', pid=process.pid, started_utc=utc(), log=str(log),
                                baseline_running_at_start=occupied, once_running_after_start=len(active),
                                total_owned_simulations_upper_bound_at_start=occupied + len(active))
        publish()
        if pending or active:
            time.sleep(10)
    verify_sources(plan)
    meta.update(ended_utc=utc(), source_hashes_verified_after=True)
    publish()
    if any(value['status'] != 'complete' for value in state.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
