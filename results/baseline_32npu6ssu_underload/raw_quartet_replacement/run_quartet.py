#!/usr/bin/env python3
"""Frozen raw-data quartet replacements; no edits to simulation or policy code."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(ROOT), str(STUDY), str(ROOT / 'results/baseline_npu32_investigation')]
import run_mixed_sustained_probe as helper
import run_study
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_coflow_experiments import source_files
from exact_order_design import exact_order_indices, describe_exact_order

SEEDS = [7, 19, 43, 67, 101]
STRATEGIES = ['baseline', 'once']
SPECS = {
    'raw_feasible_q2': {'keys': '32:1024,48:1024,64:1024,160:1024', 'quotas': [2, 2, 2, 1], 'seeds': SEEDS},
    'raw_nearest_q25': {'keys': '32:128,32:256,32:512,192:1024', 'quotas': [25, 25, 25, 1], 'seeds': [7]},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reordered(requests, metadata):
    profiles = [dict(p, analysis_role='short' if i < 3 else 'long') for i, p in enumerate(metadata['profiles'])]
    pmap = {(p['seq_len_k'], p['nql']): i for i, p in enumerate(profiles)}
    new, descriptions = [], []
    for npu in range(32):
        lane = sorted((r for r in requests if r.npu_id == npu), key=lambda r: (r.arrival_time_ms, r.request_id))
        indices = [pmap[(r.load['seq_len_k'], r.load['nql'])] for r in lane]
        kwargs = {'npu': npu, 'seed': metadata['seed'], 'mode': 'exact_cohort4', 'period_packets': 1, 'phase_jitter': 0.0}
        order = exact_order_indices(indices, profiles, **kwargs)
        descriptions.append(dict(npu_id=npu, **describe_exact_order(indices, profiles, **kwargs)))
        assert sorted(order) == list(range(len(lane)))
        for pos, old_index in enumerate(order):
            old = lane[old_index]
            rid = npu * 1_000_000 + pos
            load = dict(old.load, request_id=rid, generation=pos, original_request_id=old.request_id)
            new.append(ContinuousBatchRequest.from_normalized(rid, npu, old.arrival_time_ms, load, old.placement))
    new = tuple(new)
    fp = continuous_batch_input_fingerprint(new)
    label = metadata['label'] + '__exact_cohort4_p1'
    meta = dict(metadata, label=label, case_id=f'{label}_{fp[:12]}', input_fingerprint=fp,
                logical_input_fingerprint=logical_input_fingerprint(new),
                random_reference_label=metadata['label'], random_reference_input_fingerprint=metadata['input_fingerprint'],
                random_reference_manifest_sha256=sha(HERE / 'inputs' / (metadata['label'] + '.json.gz')),
                per_npu_order_design=descriptions, order_mode='exact_cohort4', order_period_packets=1,
                phase_jitter=0.0, order='exact_cohort4',
                id_rule='new queue position ID; original_request_id maps to the frozen random reference; original NPU and every block placement retained')
    return new, meta


def prepare():
    HERE.mkdir(parents=True, exist_ok=True)
    helper.NUM_SSU = 6
    table = ast.literal_eval((ROOT / 'data').read_text())
    inputs, jobs, descriptions = [], [], []
    for family, spec in SPECS.items():
        for seed in spec['seeds']:
            label = f'{family}_seed{seed}'
            path = HERE / 'inputs' / f'{label}.json.gz'
            if path.exists():
                requests, meta = load_manifest(path)
            else:
                requests, meta = helper.build_input(family='raw', seed=seed, profile_keys=spec['keys'],
                                                   quotas=spec['quotas'], short_profile_count=3, label=family)
                proof = run_study.certificate(requests)
                cs = [p['per_layer_compute_us'] / 1000 for p in meta['profiles']]
                packet_ms = 8 * sum(q*c for q, c in zip(spec['quotas'], cs))
                raw_checks = []
                for r in requests:
                    key = (r.load['seq_len_k'], r.load['nql'])
                    bw, c, _, v = table[key]
                    assert r.load['per_layer_us'] == c
                    assert math.isclose(sum(volume for _, volume in r.placement[0]), v, rel_tol=1e-12, abs_tol=1e-12)
                    assert r.load['constructed_profile'] is False
                meta.update(
                    experiment='raw_quartet_replacement_v1', quartet_family=family,
                    measurement_window_ms=[2000, 4000], data_sha256=sha(ROOT / 'data'),
                    stripe_group_rule='ssu=(block_index+original_npu_id//4)%6; all selected rows exactly align to 176KiB',
                    active_profile_rate_certificate=proof,
                    pure_packet_compute_ms=packet_ms,
                    short_pure_compute_share=sum(q*c for q, c in zip(spec['quotas'][:3], cs[:3])) / sum(q*c for q, c in zip(spec['quotas'], cs)),
                    pure_horizon_rule='ceil((4000 + 2*maximum 8-layer request compute)/packet compute) complete quota packets per card',
                    construction_runner_sha256=sha(__file__),
                    replacement_scope=('Raw feasible external-validity control: profiles and short quotas both change; short compute share nearly matches original, but I/O volumes, C scales, class pools and phase cuts differ.'
                                       if family == 'raw_feasible_q2' else
                                       'Nearest grid keys with ties to upper NQL and distinct short keys; retain 25 short triplets per long. Mathematically impossible to satisfy current-profile underload with all 32 cards active.'))
                save_manifest(path, requests, meta)
                write_json(path.with_name(path.stem + '.description.json'), meta)
            ordered_label = label + '__exact_cohort4_p1'
            ordered_path = HERE / 'inputs' / f'{ordered_label}.json.gz'
            if not ordered_path.exists():
                ore, ometa = reordered(requests, meta)
                save_manifest(ordered_path, ore, ometa)
                write_json(ordered_path.with_name(ordered_path.stem + '.description.json'), ometa)
            for order, mpath in [('random', path), ('ordered', ordered_path)]:
                rs, md = load_manifest(mpath)
                item = {'family': family, 'seed': seed, 'order': order, 'label': md['label'],
                        'manifest': str(mpath), 'manifest_sha256': sha(mpath), 'input_fingerprint': md['input_fingerprint'],
                        'request_count': len(rs)}
                inputs.append(item)
                # The nearest replacement is an infeasible diagnostic, not a five-seed search.
                if family == 'raw_feasible_q2' or order == 'ordered':
                    for strategy in STRATEGIES:
                        jobs.append({'input': item, 'strategy': strategy})
            if seed == spec['seeds'][0]:
                descriptions.append({k: meta[k] for k in ['quartet_family', 'profile_keys', 'profile_quotas_per_unit',
                    'profile_counts_per_npu', 'requests_per_npu', 'quota_units_per_npu', 'pure_packet_compute_ms',
                    'short_pure_compute_share', 'active_profile_rate_certificate', 'replacement_scope']})
    source_names = set(source_files()) | {'run_baseline_npu32_stress.py', 'data'}
    source_hashes = {name: sha(ROOT / name) for name in sorted(source_names)}
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'num_npu': 32, 'num_ssu': 6,
            'window_ms': [2000, 4000], 'strategies': STRATEGIES, 'seeds_for_feasible': SEEDS,
            'collector_interval_ms': 5, 'assignment': 'fixed', 'inputs': inputs, 'jobs': jobs,
            'source_sha256': source_hashes, 'runner_sha256': sha(__file__), 'descriptions': descriptions,
            'slo': 'Admission in [2000,4000), followed to completion; completion-admission <=1.5*8C; mean per-seed rates equally.',
            'validity': 'All checks and every run retained. Feasible family must be audited for full-time nominal underload, all 32 active, each card warm mixed, and fourth completion <=1500. No resampling failed cases.',
            'nearest_diagnostic': 'Only seed7 Ordered, both strategies. It is not a valid underload counterexample and will never be pooled with the feasible family.',
            'scientific_caveats': ['Feasible raw short policy classes become SL rather than SS.',
                'Exact-cohort algorithm is unchanged, but physical phase cuts and packet length change.',
                'Finite request count is recomputed from the same minimum-compute horizon rule.',
                'No measured data values or compute scale are fitted to simulation results.']}
    p = HERE / 'plan.json'
    if p.exists():
        old = json.loads(p.read_text())
        for key in ['inputs', 'jobs', 'source_sha256', 'runner_sha256']:
            assert old[key] == plan[key], f'Frozen plan mismatch: {key}'
        return old
    write_json(p, plan)
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    plan = prepare()
    print(json.dumps({'prepared': len(plan['jobs']), 'families': [x['quartet_family'] for x in plan['descriptions']]}), flush=True)
    if not args.run:
        return
    # The imported orchestrator launches independent subprocesses. Only its output root changes here.
    run_study.HERE = HERE
    jobs = sorted(plan['jobs'], key=lambda x: (x['strategy'] != 'baseline',
                  x['input']['family'] != 'raw_feasible_q2', x['input']['seed'], x['input']['order']))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_study.run_job, job['input'], job['strategy'], 7200) for job in jobs}
        while pending:
            done, pending = wait(pending, timeout=25, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                records.append(row)
                write_json(HERE / 'status.json', {'finished': len(records), 'total': len(jobs), 'rows': records})
                print(json.dumps({k: row[k] for k in ['label', 'strategy', 'status', 'wall_seconds'] if k in row}), flush=True)
            if not done:
                print(json.dumps({'finished': len(records), 'total': len(jobs), 'pending_including_queued': len(pending)}), flush=True)
    assert all(sha(ROOT / name) == value for name, value in plan['source_sha256'].items())
    assert all(sha(i['manifest']) == i['manifest_sha256'] for i in plan['inputs'])
    assert all(r['status'] in ['complete', 'existing'] for r in records)


if __name__ == '__main__':
    main()
