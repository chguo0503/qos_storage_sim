#!/usr/bin/env python3
"""Predeclared first-pass matrix; keep every success and failure."""
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
    p.add_argument('--output', type=Path, default=HERE/'screen')
    args = p.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    orders = [('random', 1, 'sync'), ('ordered', 1, 'sync'), ('ordered', 4, 'sync'),
              ('ordered', 4, 'split'), ('ordered', 4, 'stagger')]
    jobs = []
    # First start the closest-capacity pair, then both underloaded references.
    for mode, block, phase in orders:
        for disks in (3, 4, 6):
            label = f'ssu{disks}_{mode}_k{block}_{phase}_seed7'
            cmd = [sys.executable, '-B', str(HERE/'experiment.py'), '--num-ssu', str(disks),
                   '--mode', mode, '--block', str(block), '--phase', phase, '--a-count', '12',
                   '--seed', '7', '--horizon-ms', '4500', '--window', '2000:4000',
                   '--window', '2000:6000', '--window', '4000:6000',
                   '--output', str(out), '--label', label, '--run']
            jobs.append(dict(label=label, cmd=cmd, num_ssu=disks, mode=mode, block=block, phase=phase))
    (out/'matrix_plan.json').write_text(json.dumps(dict(jobs=jobs, workers=args.workers,
        population='Each card 12A+24B, direct data, same original identities across orders',
        primary_window_ms=[2000,4000], supplementary_windows_ms=[[2000,6000],[4000,6000]],
        constraints='3 disks population mean overload; all orders audited for per-instant nominal overload. No seed/case omitted based on outcome.'), indent=2)+'\n')
    def run(job):
        log = out / (job['label']+'.log')
        t = time.perf_counter()
        print(json.dumps(dict(event='start', label=job['label'])), flush=True)
        with log.open('w') as stream:
            process = subprocess.run(job['cmd'], stdout=stream, stderr=subprocess.STDOUT)
        row = dict(**job, returncode=process.returncode, wall_seconds=time.perf_counter()-t, log=str(log))
        print(json.dumps(dict(event='finish', label=job['label'], returncode=process.returncode,
                              wall_seconds=row['wall_seconds'])), flush=True)
        return row
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            tmp = out/'matrix_status.tmp'
            tmp.write_text(json.dumps(rows, indent=2)+'\n'); tmp.replace(out/'matrix_status.json')
    if any(row['returncode'] for row in rows):
        raise SystemExit('Some cases failed; inspect preserved logs.')


if __name__ == '__main__':
    main()
