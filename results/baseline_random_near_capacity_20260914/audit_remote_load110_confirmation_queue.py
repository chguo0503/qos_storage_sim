#!/usr/bin/env python3
"""Four predeclared raw-data load110 Baseline confirmation seeds, within the shared six-process limit."""
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
    plan_path = HERE / 'audit_remote_load110_confirmation_plan.json'
    plan = read(plan_path)
    launch = read(HERE / 'audit_remote_load110_confirmation_launch.json')
    assert launch['plan_sha256'] == sha(plan_path) and launch['input_audit_passed'] is True
    assert plan['max_load110_confirmation_parallel'] == 4 and plan['remote_simulation_limit'] == 6
    assert len(plan['jobs']) == 4 and len({job['label'] for job in plan['jobs']}) == 4
    assert all(job['strategy'] == 'baseline' and job['order'] == 'random' and not job['trace'] for job in plan['jobs'])
    assert plan['input_profiles_are_raw_data'] is True
    verify_sources(plan)
    for job in plan['jobs']:
        assert sha(ROOT / job['manifest']) == job['manifest_sha256']
        assert not (HERE / 'runs' / job['label'] / 'baseline').exists()
    state = {job['label']: {'status': 'pending'} for job in plan['jobs']}
    pending = list(plan['jobs'])
    active = {}
    meta = {'host': platform.node(), 'python_version': sys.version, 'pid': os.getpid(),
            'started_utc': utc(), 'plan_sha256': sha(plan_path),
            'queue_runner_sha256': sha(Path(__file__)), 'job_count': 4,
            'launch_record_sha256': sha(HERE / 'audit_remote_load110_confirmation_launch.json'),
            'max_load110_confirmation_parallel': 4, 'remote_simulation_limit': 6,
            'source_hashes_verified_before': True, 'selection': plan['selection']}

    def publish():
        write(HERE / 'audit_remote_load110_confirmation_status.json', dict(meta, updated_utc=utc(), jobs=state,
              complete=sum(v['status'] == 'complete' for v in state.values()),
              failed=sum(v['status'] == 'failed' for v in state.values()), running=len(active)))

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
            record_path = HERE / 'runs' / label / 'baseline' / 'command.json'
            record = read(record_path) if record_path.exists() else {}
            error = None
            try:
                assert code == 0 and record['status'] == 'complete' and record['returncode'] == 0
                assert record['completed_simulation'] and record['strategy'] == 'baseline'
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
        occupied = 0
        predecessor_pending = 0
        predecessor_counts = {}
        for filename, expected in plan['predecessor_status_plan_sha256'].items():
            status = read(HERE / filename)
            assert status['plan_sha256'] == expected
            predecessor_pending += sum(v['status'] == 'pending' for v in status['jobs'].values())
            occupied += status['running']
            predecessor_counts[filename] = status['running']
        # Every predecessor must have dispatched every job first. Their stale
        # running counts then only overestimate occupied simulation slots.
        while pending and predecessor_pending == 0 and len(active) < 4 and occupied + len(active) < 6:
            job = pending.pop(0)
            label = job['label']
            log = HERE / 'execution_logs' / (label + '.load110_confirmation.remote.log')
            stream = log.open('w')
            started = time.perf_counter()
            process = subprocess.Popen([sys.executable, str(HERE / 'experiment.py'), '--manifest',
                                        str(ROOT / job['manifest']), '--strategy', 'baseline'],
                                       cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            active[label] = (process, stream, started, job)
            state[label].update(status='running', pid=process.pid, started_utc=utc(), log=str(log),
                                predecessors_running_at_start=dict(predecessor_counts),
                                load110_confirmation_running_after_start=len(active),
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
