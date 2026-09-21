"""Run independent frozen-input comparisons with bounded process concurrency."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--label', required=True)
    parser.add_argument('--groups', nargs='+', default=['XY12_16', 'R32_A200_B10',
        'XY12_32', 'XY12_24', 'XY12_20', 'X16'])
    parser.add_argument('--ssu', type=int, help='Select only this disk count')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    configs = sorted(HERE.joinpath('configs').glob('*.json'))
    jobs = []
    for group in args.groups:
        for config in configs:
            values = json.loads(config.read_text())
            if (values['id'] == group and values['seed'] in args.seeds
                    and (args.ssu is None or values['ssu'] == args.ssu)):
                for strategy in ('od_baseline', 'asu_baseline', 'once'):
                    jobs.append((config, strategy))
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    progress = HERE / f'batch_{args.label}.json'
    started = time.time()
    results = []

    def run(job):
        config, strategy = job
        output = HERE / 'runs' / config.stem / strategy
        log = logs / f'{config.stem}__{strategy}.log'
        begin = time.time()
        if (output / 'metrics.json').exists():
            return dict(case=config.stem, strategy=strategy, status='cached',
                        exit_code=0, wall_seconds=0.0)
        command = [sys.executable, str(HERE / 'runner.py'), '--config', str(config),
                   '--strategy', strategy, '--output', str(output)]
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=PROJECT, stdout=stream,
                                    stderr=subprocess.STDOUT)
        return dict(case=config.stem, strategy=strategy,
                    status='completed' if result.returncode == 0 else 'failed',
                    exit_code=result.returncode, wall_seconds=time.time() - begin,
                    output=str(output.relative_to(HERE)), log=str(log.relative_to(HERE)))

    print(json.dumps(dict(event='batch_start', label=args.label, jobs=len(jobs),
                          workers=args.workers)), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            payload = dict(label=args.label, seeds=args.seeds, total=len(jobs),
                           finished=len(results), wall_seconds=time.time() - started,
                           results=results)
            temporary = progress.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(progress)
            print(json.dumps(dict(event='job_finished', finished=len(results),
                                  total=len(jobs), **row)), flush=True)
    if any(r['exit_code'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
