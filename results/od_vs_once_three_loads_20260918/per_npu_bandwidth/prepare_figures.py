#!/usr/bin/env python3
"""Recover complete-cycle SSD averages from frozen, completed seed-7 runs.

Read-only with respect to all simulation inputs/results. No new simulation.
Complete cycles are checked against prefetch timing and OD service observers.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from inputs.runners.run_baseline_npu32_stress import load_manifest, read_json

LEFT, RIGHT = 2000.0, 4000.0
SOURCES = {
    ('full', 'od_baseline'): 'od_baseline_diverse_ssu3_20260918/runs/full_od_baseline_seed7_remote',
    ('full', 'once'): 'od_baseline_diverse_ssu3_20260918/runs/full_once_seed7_remote',
    ('under', 'od_baseline'): 'continuous_underload_asu_od_20260918/runs/random_od_baseline_seed7_remote',
    ('under', 'once'): 'od_vs_once_three_loads_20260918/runs/under_once_seed7_local',
    ('semi', 'od_baseline'): 'od_baseline_diverse_ssu3_20260918/runs/semi_od_baseline_seed7_remote',
    ('semi', 'once'): 'od_baseline_diverse_ssu3_20260918/runs/semi_once_seed7_remote',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a, z, left=LEFT, right=RIGHT):
    return max(0.0, min(z, right)-max(a, left))


def close(a, b, tol=1e-7):
    assert abs(a-b) <= tol, (a, b, tol)


def coverage(segments):
    close(segments[0][0], LEFT)
    close(segments[-1][1], RIGHT)
    for a, b in zip(segments, segments[1:]):
        close(a[1], b[0])
    close(math.fsum(z-a for a, z, *_ in segments), RIGHT-LEFT)


def build_case(regime, policy, relative):
    source = ROOT/'results'/relative
    paths = [source/name for name in ('command.json', 'result.json.gz', 'manifest.json.gz')]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    command, result = read_json(paths[0]), read_json(paths[1])
    requests, meta = load_manifest(paths[2])
    assert command['completed_simulation'] and command['status'] == 'complete'
    assert command['policy'] == policy
    assert sha(paths[1]) == command['result_sha256']
    assert sha(paths[2]) == command['manifest_sha256']
    assert meta['seed'] == 7 and meta['order'] == 'random'
    assert (meta['num_npu'], meta['num_ssu'], meta['n_layers']) == (32, 3, 8)
    warm = next(a for a in result['analysis'] if a['start_ms'] == LEFT and a['end_ms'] == RIGHT)
    assert warm['all_npus_active']
    volume = {r.request_id: [math.fsum(v for d, v in r.placement[0] if d == disk)
                            for disk in range(3)] for r in requests}
    demand = {r.request_id: math.fsum(volume[r.request_id])*1e6/r.load['per_layer_us']
              for r in requests}
    batches = [[] for _ in range(32)]
    for batch in result['summary']['microbatch_metrics']:
        assert batch['batch_size'] == 1 and len(batch['member_request_ids']) == 1
        rid = batch['member_request_ids'][0]
        for layer in batch['layer_metrics']:
            batches[batch['npu_id']].append((layer['compute_start_ms'], rid, layer))
    admissions = [[] for _ in range(32)]
    for request in result['summary']['request_metrics']:
        a, z = request['admission_time_ms'], request['completion_time_ms']
        if overlap(a, z):
            admissions[request['npu_id']].append([max(a, LEFT), min(z, RIGHT), demand[request['request_id']]])

    observer = None
    if policy == 'od_baseline':
        obs_path = ROOT/'results/load_regimes_asu_od_comparison_20260918/layer_average/runs'/f'{regime}_od_baseline_seed7/services.json.gz'
        observer = read_json(obs_path)
        assert observer['source_case'] == str(source.relative_to(ROOT))
        hashes[str(obs_path.relative_to(ROOT))] = sha(obs_path)
    max_service_error = max_timing_error = 0.0
    all_cycles = warm_cycles = cross_cycles = 0
    cards = []
    for npu, states in enumerate(batches):
        states.sort()
        curve, records, stalls, crosses = [], [], [], []
        for current, following in zip(states, states[1:]):
            all_cycles += 1
            a, rid, layer = current
            z, next_rid, following_layer = following
            assert z > a
            error = max(abs(following_layer['io_start_time_ms']-a),
                        abs(z-max(layer['compute_end_ms'], following_layer['io_ready_time_ms'])))
            close(error, 0.0)
            max_timing_error = max(max_timing_error, error)
            if not overlap(a, z):
                continue
            warm_cycles += 1
            total = math.fsum(volume[next_rid])
            rate = total/((z-a)/1000)
            curve.append([max(a, LEFT), min(z, RIGHT), rate])
            record = dict(start_ms=a, end_ms=z, request_id=rid, layer=layer['layer'],
                          next_request_id=next_rid, next_layer=following_layer['layer'],
                          same_request=rid == next_rid, next_layer_GiB_by_ssu=volume[next_rid],
                          complete_cycle_supply_GiB_s=rate,
                          compute_end_ms=layer['compute_end_ms'])
            records.append(record)
            if rid != next_rid:
                cross_cycles += 1
                crosses.append([max(a, LEFT), min(z, RIGHT)])
            stall_start = layer['compute_end_ms']
            if overlap(stall_start, z) > 1e-9:
                stalls.append([max(stall_start, LEFT), min(z, RIGHT)])
        coverage(curve)
        request_segments = sorted(admissions[npu])
        coverage(request_segments)
        if observer is not None:
            measured = observer['per_npu_cycles'][npu]
            assert len(measured) == len(records)
            for record, cycle in zip(records, measured):
                close(record['start_ms'], cycle['start_ms'])
                close(record['end_ms'], cycle['end_ms'])
                assert (record['request_id'], record['next_request_id']) == (cycle['request_id'], cycle['next_request_id'])
                for expected, actual in zip(record['next_layer_GiB_by_ssu'], cycle['full_service_GiB_by_ssu']):
                    close(expected, actual)
                    max_service_error = max(max_service_error, abs(expected-actual))
        compute_ms = math.fsum(overlap(l['compute_start_ms'], l['compute_end_ms']) for _, _, l in states)
        utilization = 100*compute_ms/(RIGHT-LEFT)
        close(utilization, warm['per_npu_U_percent'][npu])
        close(compute_ms+math.fsum(z-a for a, z in stalls), RIGHT-LEFT)
        mean_demand = math.fsum((z-a)*b for a, z, b in request_segments)/(RIGHT-LEFT)
        mean_supply = math.fsum(result['warm_ssd_GiB_s_by_ssu_npu'][disk][npu] for disk in range(3))
        cards.append(dict(npu_id=npu, U_percent=utilization,
                          mean_demand_GiB_s=mean_demand, mean_supply_GiB_s=mean_supply,
                          demand_segments=request_segments, supply_segments=curve,
                          stall_segments=stalls, cross_request_cycles=crosses,
                          source_cycle_count=len(records), cycles=records))
    close(math.fsum(c['U_percent'] for c in cards)/32, warm['U_percent'])
    close(math.fsum(c['mean_supply_GiB_s'] for c in cards), math.fsum(warm['SSD_GiB_s']))
    reference_demand = math.fsum((s[1]-s[0])*math.fsum(s[2:]) for s in warm['demand']['segments'])/(RIGHT-LEFT)
    close(math.fsum(c['mean_demand_GiB_s'] for c in cards), reference_demand)
    for name, digest in hashes.items():
        assert sha(ROOT/name) == digest
    return dict(regime=regime, policy=policy, seed=7, window_ms=[LEFT, RIGHT],
                source_case=str(source.relative_to(ROOT)), source_sha256=hashes,
                manifest_sha256=command['manifest_sha256'], fleet_U_percent=warm['U_percent'],
                per_npu=cards,
                checks=dict(all_input_cycles_checked=all_cycles, warm_cycles=warm_cycles,
                            warm_cross_request_cycles=cross_cycles,
                            max_prefetch_timing_error_ms=max_timing_error,
                            measured_od_max_service_volume_error_GiB=max_service_error if observer else None,
                            all_32_npus_active=True, coverage_and_metrics_verified=True,
                            source_files_unchanged=True))


def main():
    cases = [build_case(regime, policy, relative) for (regime, policy), relative in SOURCES.items()]
    for regime in ('full', 'semi', 'under'):
        pair = [c for c in cases if c['regime'] == regime]
        assert pair[0]['manifest_sha256'] == pair[1]['manifest_sha256']
    payload = dict(schema_version=1, units='GiB/s',
                   demand_definition='Current admitted request per-layer V / pure compute C; unchanged during stall.',
                   supply_definition='Complete-cycle next-layer SSD GiB / (next compute start - current compute start). Clip display only.',
                   warm_mean_supply_definition='Recorded physical SSD service overlapping [2000,4000), divided by 2 seconds.',
                   cases=cases)
    (HERE/'input.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    fields = ['regime', 'policy', 'seed', 'npu_id', 'U_percent', 'mean_demand_GiB_s', 'mean_supply_GiB_s', 'source_cycle_count']
    with (HERE/'per_npu_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for card in case['per_npu']:
                row = {k: case[k] if k in case else card[k] for k in fields}
                writer.writerow(row)
    checks = [dict(regime=c['regime'], policy=c['policy'], **c['checks']) for c in cases]
    (HERE/'measurement_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps([dict(regime=c['regime'], policy=c['policy'], U_percent=c['fleet_U_percent'], **c['checks']) for c in cases], indent=2))


if __name__ == '__main__':
    main()
