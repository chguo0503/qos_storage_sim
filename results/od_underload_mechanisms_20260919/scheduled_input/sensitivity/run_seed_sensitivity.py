"""Unhooked OD replay of immutable OD-tailored input with a different seed.

The input generator is never imported. No target-E assertions apply here:
loss of synchronization is a valid sensitivity result, not a failed run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SCHEDULED = HERE.parent
ROOT = SCHEDULED.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]
import metrics
from inputs.manifest import load_manifest, write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = sorted((ROOT / 'simulator').rglob('*.py'))
    paths += sorted((ROOT / 'inputs').rglob('*.py'))
    paths += [ROOT / 'data', Path(__file__), HERE / 'metrics.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def validate(name, expected_cycles):
    manifest = SCHEDULED / 'inputs' / f'{name}.json.gz'
    planning = SCHEDULED / 'planning' / name / 'command.json'
    command = json.loads(planning.read_text())
    assert command['status'] == 'complete' and command['conditions_passed']
    assert command['source_unchanged'] and command['hook_restored']
    assert sha(manifest) == command['manifest_sha256']
    requests, meta = load_manifest(manifest)
    assert meta['cycles'] == expected_cycles and meta['OD_tailored']
    assert not meta['physical_prefix_reuse']
    assert (meta['num_npu'], meta['num_ssu'], meta['n_layers']) == (32, 3, 8)
    assert sha(ROOT / 'data') == meta['data_sha256']
    assert all(sha(ROOT / p) == value for p, value in meta['planning_source_sha256'].items())
    assert all(sha(ROOT / p) == value for p, value in meta['planning_artifact_sha256'].items())
    assert len(requests) == 32 * 3 * expected_cycles
    assert len({r.request_id for r in requests}) == len(requests)
    assert len({r.load['original_request_id'] for r in requests}) == len(requests)
    assert all(r.arrival_time_ms == 0 and len(r.placement) == 1 for r in requests)
    assert all(math.isfinite(r.load['per_layer_us']) and r.load['per_layer_us'] > 0 for r in requests)
    pure = [sum(8 * r.load['per_layer_us'] / 1000 for r in requests if r.npu_id == n) for n in range(32)]
    if expected_cycles == 50:
        assert min(pure) > 60000
    return requests, meta, manifest, planning, pure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='abb_interp_unique_50')
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--expected-cycles', type=int, default=50)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    requests, meta, manifest, planning, pure = validate(args.name, args.expected_cycles)
    if args.validate_only:
        print(json.dumps(dict(validated=True, name=args.name, requests=len(requests),
                              manifest_sha256=sha(manifest), pure_compute_ms_by_npu=pure)), flush=True)
        return
    out = HERE / f'{args.name}_od_seed{args.seed}'
    out.mkdir(exist_ok=False)
    (out / 'manifest.json.gz').write_bytes(manifest.read_bytes())
    (out / 'planning_command.json').write_bytes(planning.read_bytes())
    before = source_hashes()
    manifest_sha = sha(manifest)
    planning_sha = sha(planning)
    fingerprint = core.continuous_batch_input_fingerprint(requests)
    start = time.perf_counter()
    record = dict(status='running', name=args.name, pid=os.getpid(), seed=args.seed,
                  planning_seed=7, strategy='od_baseline', od_queue_depth_per_ssu=8192,
                  runtime_planning_hooks=False, observer_hooks=False,
                  source_sha256_before=before, manifest_sha256=manifest_sha,
                  planning_command_sha256=planning_sha, input_fingerprint=fingerprint,
                  pure_compute_ms_by_npu=pure, argv=sys.argv,
                  sole_configuration_change='submit_order_seed 7 -> requested seed; manifest byte-identical',
                  expected_blocks=sum(8 * len(r.placement[0]) for r in requests))
    write_json(out / 'command.json', record)
    print(json.dumps(dict(status='running', name=args.name, seed=args.seed,
                          expected_blocks=record['expected_blocks'], input_fingerprint=fingerprint)), flush=True)
    try:
        result = run_simulation(requests, strategy='od_baseline', num_npu=32, num_ssu=3,
                                n_layers=8, seed=args.seed, disk_bw_gib_s=40,
                                npu_bw_gib_s=50, collector_interval_ms=5,
                                cross_request_layer0_prefetch=True,
                                od_queue_depth_per_ssu=8192)
        raw = result['summary']
        assert raw['completed_blocks'] == record['expected_blocks']
        assert all(raw['invariants'].values())
        assert result['input_fingerprint'] == fingerprint
        assert core.continuous_batch_input_fingerprint(requests) == fingerprint
        qdepth = raw['ssd_queue_depth']
        assert qdepth['per_npu_per_ssu_slots'] == 256
        assert max(map(max, qdepth['peak_outstanding_blocks_by_npu_ssu'])) <= 256
        assert qdepth['host_deferred_blocks_at_stop'] == qdepth['ssd_outstanding_blocks_at_stop'] == qdepth['link_outstanding_blocks_at_stop'] == 0
        write_json(out / 'result.json.gz', result)
        byid = {r.request_id: r for r in requests}
        rows = raw['request_metrics']
        first_drain = min(max(r['completion_time_ms'] for r in rows if r['npu_id'] == n) for n in range(32))
        main_windows = [(2000, 4000), (20000, 40000), (40000, 60000), (20000, 60000)]
        contiguous = [(a, a + 2000) for a in range(0, int(raw['makespan_ms'] // 2000) * 2000, 2000)]
        statistics = {}
        for left, right in sorted(set(main_windows + contiguous)):
            w = metrics.summarize(raw, requests, left, right)
            w['measurement_definition']['own_compute'] = '8 times frozen per-layer C; includes OD-tailored interpolated B compute'
            # Independent layer overlap and complete admission-cohort checks.
            compute = sum(max(0., min(right, l['compute_end_ms']) - max(left, l['compute_start_ms']))
                          for b in raw['microbatch_metrics'] for l in b['layer_metrics'])
            assert math.isclose(w['U_percent'], 100 * compute / (32 * (right - left)), abs_tol=1e-8)
            statistics[left, right] = w
        phases = []
        indexed = {(r['npu_id'], byid[r['request_id']].load['cycle'], byid[r['request_id']].load['position']): r for r in rows}
        for k in range(meta['cycles']):
            ar = [indexed[n, k, 0] for n in range(32)]
            tails = [indexed[n, k, 2] for n in range(32)]
            starts = [r['admission_time_ms'] for r in ar]
            ends = [r['completion_time_ms'] for r in tails]
            compute = sum(indexed[n, k, pos]['own_compute_ms'] for n in range(32) for pos in range(3))
            phases.append(dict(cycle=k, A_admission_range_ms=max(starts) - min(starts),
                               tail_completion_range_ms=max(ends) - min(ends),
                               maximum_deviation_from_seed7_E_ms=max(abs(e - (k + 1) * meta['period_ms']) for e in ends),
                               duration_weighted_cycle_U_percent=100 * compute / sum(z - a for a, z in zip(starts, ends)),
                               end_latest_ms=max(ends), ends_before_first_card_drains=max(ends) < first_drain))
        analysis = dict(first_card_drains_ms=first_drain, makespan_ms=raw['makespan_ms'],
                        all_native_invariants=True, no_seed7_target_assertions=True,
                        main_windows=[statistics[w] for w in main_windows],
                        contiguous_2s_windows=[statistics[w] for w in contiguous], cycles=phases)
        write_json(out / 'analysis.json.gz', analysis)
        brief = dict(seed=args.seed, input_fingerprint=fingerprint, manifest_sha256=manifest_sha,
                     first_card_drains_ms=first_drain, makespan_ms=raw['makespan_ms'],
                     windows=[dict(start_ms=w['start_ms'], end_ms=w['end_ms'], U_percent=w['U_percent'],
                                   SLO15=w['slo'], all_npus_active=w['all_npus_active'],
                                   mixed_cards=w['role_and_stall']['npus_with_A_and_B_compute'],
                                   overload_percent=w['demand']['per_disk_overload_percent'])
                              for w in analysis['main_windows']])
        write_json(out / 'brief.json', brief)
        record.update(status='complete', completed_blocks=raw['completed_blocks'], brief=brief)
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        after = source_hashes()
        record.update(wall_seconds=time.perf_counter() - start, source_sha256_after=after,
                      source_unchanged=after == before,
                      input_bytes_unchanged=sha(manifest) == sha(out / 'manifest.json.gz') == manifest_sha,
                      planning_command_unchanged=sha(planning) == planning_sha)
        write_json(out / 'command.json', record)
        assert record['source_unchanged'] and record['input_bytes_unchanged'] and record['planning_command_unchanged']
    print(json.dumps(record['brief']), flush=True)


if __name__ == '__main__':
    main()
