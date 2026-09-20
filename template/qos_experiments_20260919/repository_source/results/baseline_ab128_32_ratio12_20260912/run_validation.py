#!/usr/bin/env python3
"""Long-window validation of the simple ABB ordering and predeclared random seeds."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args(); out = HERE/'validation20s'; out.mkdir(parents=True, exist_ok=True)
    choices = [(d, mode, 7) for d in (3,4,6) for mode in ('random','ordered')]
    choices += [(d,'random',seed) for seed in (19,43) for d in (3,4,6)]
    jobs = []
    for disks, mode, seed in choices:
        label = f'ssu{disks}_{mode}_k1_sync_seed{seed}'
        cmd = [sys.executable, '-B', str(HERE/'experiment.py'), '--num-ssu', str(disks),
               '--mode', mode, '--block', '1', '--phase', 'sync', '--a-count', '40',
               '--seed', str(seed), '--horizon-ms', '20000', '--window', '2000:4000',
               '--window', '2000:20000', '--output', str(out), '--label', label, '--run']
        for start in range(4000,20000,2000):
            cmd += ['--window', f'{start}:{start+2000}']
        jobs.append(dict(label=label, cmd=cmd, num_ssu=disks, mode=mode, seed=seed))
    (out/'validation_plan.json').write_text(json.dumps(dict(jobs=jobs, workers=args.workers,
        purpose='Validate sustained behavior and independent random seeds; simple ABB ordering chosen before k4 screen finishes',
        population='40 A and 80 B per card; 20227.102572912 ms pure compute per NPU',
        caveat='Different queue lengths change full-shuffle prefixes; validation is a separate population from screen. Ordered queue-type prefixes match.',
        primary_window_ms=[2000,4000], long_window_ms=[2000,20000]), indent=2)+'\n')
    def run(job):
        t = time.perf_counter()
        print(json.dumps(dict(event='start', label=job['label'])), flush=True)
        with (out/(job['label']+'.log')).open('w') as log:
            proc = subprocess.run(job['cmd'], stdout=log, stderr=subprocess.STDOUT)
        row = dict(**job, returncode=proc.returncode, wall_seconds=time.perf_counter()-t)
        print(json.dumps(dict(event='finish', label=job['label'], returncode=proc.returncode,
                              wall_seconds=row['wall_seconds'])), flush=True)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            tmp = out/'validation_status.tmp'; tmp.write_text(json.dumps(rows, indent=2)+'\n')
            tmp.replace(out/'validation_status.json')
    if any(row['returncode'] for row in rows):
        raise SystemExit('Validation case failed; retained logs require inspection.')


if __name__ == '__main__':
    main()
