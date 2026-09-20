#!/usr/bin/env python3
"""Run an explicit frozen experiment plan on owned remote CPU processes."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    args = ap.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan['root']).resolve()
    status_path = plan_path.with_name(plan_path.stem + '_status.json')
    assert not status_path.exists(), 'Preserve the status of any previous attempt'
    max_parallel = int(plan.get('max_parallel', 6))
    cores = plan.get('physical_cpu_ids', [0, 2, 4, 6, 8, 10, 12, 14])
    assert 1 <= max_parallel <= len(cores)
    labels = [j['label'] for j in plan['jobs']]
    assert len(labels) == len(set(labels))
    status = dict(started_utc=now(), queue_pid=os.getpid(), python=sys.version,
                  plan_sha256=sha(plan_path), queue_source_sha256=sha(Path(__file__)),
                  jobs={}, complete=False, max_parallel=max_parallel)

    def verify():
        for name, digest in plan['source_sha256'].items():
            assert sha(root/name) == digest, name

    def save():
        status['updated_utc'] = now()
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(status_path)

    verify()
    status['sources_verified_before'] = True
    save()
    waiting = iter(plan['jobs'])
    running = {}
    available = cores[:max_parallel].copy()
    more = True
    while more or running:
        while more and available:
            try:
                job = next(waiting)
            except StopIteration:
                more = False
                break
            label = job['label']
            assert Path(label).name == label and label not in ('.', '..')
            logfile = root/plan['log_dir']/(label + '.log')
            logfile.parent.mkdir(parents=True, exist_ok=True)
            stream = logfile.open('x')
            cpu = available.pop(0)
            process = subprocess.Popen(job['argv'], cwd=root, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       preexec_fn=lambda cpu=cpu: os.sched_setaffinity(0, {cpu}))
            running[label] = (process, stream, time.perf_counter(), cpu)
            status['jobs'][label] = dict(status='running', pid=process.pid, cpu_id=cpu,
                                         started_utc=now(), argv=job['argv'], log=str(logfile))
            save()
        for label, (process, stream, started, cpu) in tuple(running.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                status['jobs'][label].update(status='complete' if code == 0 else 'failed',
                                             returncode=code, ended_utc=now(),
                                             wall_seconds=time.perf_counter()-started)
                available.append(cpu)
                del running[label]
                save()
        if more or running:
            time.sleep(2)
    verify()
    status.update(complete=True, sources_verified_after=True, ended_utc=now())
    save()


if __name__ == '__main__':
    main()
