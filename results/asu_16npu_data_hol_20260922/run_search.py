"""Execute explicitly selected configurations in independent native processes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='+', required=True, type=Path)
    p.add_argument('--strategies', nargs='+', default=['asu_baseline'])
    p.add_argument('--runner', type=Path, default=PROJECT / 'results/formula_ab_32npu_20260921/runner.py')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--label', required=True)
    p.add_argument('--run-root', type=Path, default=HERE / 'runs')
    args = p.parse_args()
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    jobs = [(c, s) for c in args.configs for s in args.strategies]
    started = time.monotonic()
    results = []

    def run(job):
        c, strategy = job
        out = args.run_root / c.stem / strategy
        if (out / 'metrics.json').exists():
            return dict(case=c.stem, strategy=strategy, exit_code=0, cached=True)
        command = [sys.executable, str(args.runner), '--config', str(c),
                   '--strategy', strategy, '--output', str(out)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        begin = time.monotonic()
        with (logs / f'{c.stem}__{strategy}.log').open('w') as log:
            r = subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return dict(case=c.stem, strategy=strategy, exit_code=r.returncode,
                    wall_seconds=time.monotonic()-begin, cached=False)

    print(json.dumps(dict(event='start', label=args.label, jobs=len(jobs))), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            row = f.result()
            results.append(row)
            payload = dict(label=args.label, total=len(jobs), finished=len(results),
                           results=results, wall_seconds=time.monotonic()-started)
            target = HERE / f'batch_{args.label}.json'
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, indent=2)+'\n')
            tmp.replace(target)
            print(json.dumps(dict(event='finished', **row, completed=len(results), total=len(jobs))), flush=True)
    if any(r['exit_code'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
