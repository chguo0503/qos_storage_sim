#!/usr/bin/env python3
"""Predeclared paired Baseline/Once experiment; preserve existing frozen results."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(BASE), str(ROOT / 'results/baseline_npu32_investigation')]
import run_mixed_sustained_probe as helper
from run_baseline_npu32_stress import load_manifest, save_manifest, write_json
from run_coflow_experiments import source_files
from run_study import certificate, run_job, sha

SEEDS = [7, 19, 43, 67, 101]
STRATEGIES = ['baseline', 'once']


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    _, ref = load_manifest(BASE / 'inputs/concurrency_l768_seed7.json.gz')
    helper.NUM_SSU = 6
    need_order = []
    for seed in SEEDS:
        label = f'concurrency_l768_seed{seed}'
        path = BASE / 'inputs' / f'{label}.json.gz'
        if not path.exists():
            requests, meta = helper.build_input(
                family=ref['family'], seed=seed, profile_keys=ref['profile_keys'],
                quotas=ref['profile_quotas_per_unit'],
                short_profile_count=ref['short_profile_count_per_npu'], label='concurrency_l768')
            assert meta['label'] == label
            meta.update(
                experiment='32npu6ssu_paired_baseline_once_5seeds',
                stripe_group_rule='group=original_npu_id//4; ssu=(block_index+group)%6',
                active_profile_rate_certificate=certificate(requests),
                measurement_window_ms=[2000, 4000],
                replication_reference_label='concurrency_l768_seed7',
                preregistered_order={'runner': 'run_exact_orders.py', 'mode': 'exact_cohort4', 'packets': 1},
                construction_runner_sha256=sha(__file__),
                replication_scope='Same per-card population/placement, fixed ordered template. Seed changes random input order and runtime submission order; no tuning to new results.')
            save_manifest(path, requests, meta)
            write_json(path.with_name(path.stem + '.description.json'), meta)
        if not (BASE / 'inputs' / f'{label}__exact_cohort4_p1.json.gz').exists():
            need_order.append(label)
    if need_order:
        command = [sys.executable, '-B', str(BASE / 'run_exact_orders.py')]
        for label in need_order:
            command += ['--source-label', label]
        command += ['--mode', 'exact_cohort4', '--packets', '1']
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    inputs = []
    for seed in SEEDS:
        for order in ['random', 'ordered']:
            label = f'concurrency_l768_seed{seed}' + ('__exact_cohort4_p1' if order == 'ordered' else '')
            path = BASE / 'inputs' / f'{label}.json.gz'
            requests, meta = load_manifest(path)
            assert meta['seed'] == seed and (meta['num_npu'], meta['num_ssu']) == (32, 6)
            assert len(requests) == 19456
            assert all(r.arrival_time_ms == 0 for r in requests)
            inputs.append({'label': label, 'seed': seed, 'order': order, 'manifest': str(path),
                           'manifest_sha256': sha(path), 'input_fingerprint': meta['input_fingerprint'],
                           'request_count': len(requests)})
    names = set(source_files()) | {'run_baseline_npu32_stress.py', 'data'}
    hashes = {name: sha(ROOT / name) for name in sorted(names)}
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seeds': SEEDS,
            'strategies': STRATEGIES, 'window_ms': [2000, 4000], 'num_npu': 32, 'num_ssu': 6,
            'inputs': inputs, 'source_sha256': hashes, 'runner_sha256': sha(__file__),
            'planned_cases': 20, 'assignment': 'fixed', 'collector_interval_ms': 5,
            'slo_alpha': 1.5, 'slo_threshold': '1.5 * own_compute_ms = 1.5 * 8 * per_layer_compute_ms',
            'slo_population': 'All 19456 requests, followed to completion; separately report arrival and admission origins.',
            'averaging': 'Equal weight per seed; sample standard deviation, ranges, and sensitivity excluding original selected seed 7.',
            'validity': 'Audit [2,4)s active/mixed on every NPU, four completions per NPU by 1500ms, all-time per-SSD nominal current-profile V/C <40 GiB/s. Retain failures without cherry-picking.',
            'reuse': 'Existing Baseline seeds 7,19,43 are reused after source/input audits. NewOnce is excluded from this comparison.',
            'randomness': 'Same seed paired across policies and orders. Random manifests change order and submit seed together; ordered profile template is fixed across seeds.',
            'python_version': sys.version}
    path = OUT / 'plan.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ['seeds', 'strategies', 'source_sha256', 'inputs', 'runner_sha256']:
            assert existing[key] == plan[key], f'Frozen plan changed: {key}'
        return existing
    write_json(path, plan)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    plan = prepare()
    print(json.dumps({'prepared': 20, 'seeds': SEEDS, 'strategies': STRATEGIES}), flush=True)
    if not args.run:
        return
    jobs = [(item, strategy) for item in plan['inputs'] for strategy in STRATEGIES]
    # Start missing Baseline seeds early; existing results are quick cache reads.
    jobs.sort(key=lambda x: (0 if x[1] == 'baseline' else 1, x[0]['seed'], x[0]['order']))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, item, strategy, 3600) for item, strategy in jobs}
        while pending:
            done, pending = wait(pending, timeout=25, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(OUT / 'status.json', {'total': 20, 'finished': len(rows), 'rows': rows})
                print(json.dumps({k: row[k] for k in ['label', 'strategy', 'status', 'wall_seconds'] if k in row}), flush=True)
            if not done:
                print(json.dumps({'finished': len(rows), 'total': 20, 'pending_including_queued': len(pending)}), flush=True)
    assert all(sha(ROOT / name) == expected for name, expected in plan['source_sha256'].items())
    assert all(sha(item['manifest']) == item['manifest_sha256'] for item in plan['inputs'])
    assert all(row['status'] in ['complete', 'existing'] for row in rows), 'Inspect status.json for failure; do not drop cases.'


if __name__ == '__main__':
    main()
