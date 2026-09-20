#!/usr/bin/env python3
"""Export the matched mixed-8-NPU experiment; read-only, no simulation."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'results/fifo_mixed_unique_20260914'
OUT = ROOT / 'results/mixed8_complete_figures_20260914/data'
LEFT, RIGHT = 2000.0, 4000.0

def read(p):
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        return json.load(f)

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), (a, b)

def clip(a, b):
    return max(0.0, min(RIGHT, b) - max(LEFT, a))

def write(name, rows):
    (OUT / (name + '.json')).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    with (OUT / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {p: next((INPUT / ('formal_random_s1_once' if p == 'once' else 'formal_random_s1')).glob('*_' + p))
             for p in ('fifo', 'once', 'short_first')}
    manifests = {p: read(d / 'manifest.json.gz') for p, d in cases.items()}
    base = manifests['fifo']
    for p, m in manifests.items():
        assert m['requests'] == base['requests'], p
        assert m['placements'] == base['placements'], p
    n_layers = base['metadata']['n_layers']
    assert n_layers == 8 and len(base['requests']) == 576
    profiles = []
    for q in sorted(base['requests'], key=lambda x: (x['npu_id'], x['load']['generation'])):
        x = q['load']
        c = x['profile_construction']
        v, cm = x['per_layer_kv_gb'], x['per_layer_us'] / 1000
        close(x['total_tokens'], x['ssd_prefix_tokens'] + x['nql'])
        close(v * 1000 / cm, x['required_bw_input_gbps'])
        profiles.append(dict(
            request_id=q['request_id'], npu_id=q['npu_id'], generation=x['generation'],
            input_order_1based=x['generation'] + 1, role=x['role'], qos_category=x['category'],
            total_tokens=x['total_tokens'], total_length_k=x['seq_len_k'], nql_tokens=x['nql'],
            hit_prefix_tokens=x['ssd_prefix_tokens'], per_layer_read_gib=v, per_layer_read_mib=v * 1024,
            per_layer_compute_ms=cm, bandwidth_demand_gib_s=v * 1000 / cm,
            n_layers=n_layers, ideal_compute_8layers_ms=n_layers * cm,
            total_read_8layers_gib=n_layers * v, placement='ring_hash', ssu_count=1,
            constructed_profile=x['constructed_profile'], compute_method=c['method'],
            compute_extrapolated=c['extrapolated'], compute_source=c['source'],
            compute_scale=c['compute_scale'], padding_gib_per_layer=x['padding_gib_per_layer'],
            construction_anchors_json=json.dumps(c['anchors'], ensure_ascii=False, separators=(',', ':')),
        ))
    for n in range(8):
        a = [x for x in profiles if x['npu_id'] == n]
        assert len(a) == 72 and len({(x['total_tokens'], x['nql_tokens']) for x in a}) == 72
        assert Counter(x['role'] for x in a) == {'L': 6, 'S': 66}
    byid = {x['request_id']: x for x in profiles}
    executions, summary, source_hashes = [], [], {}
    for policy, folder in cases.items():
        result, metric = read(folder / 'result.json.gz'), read(folder / 'metrics.json')
        batches = {b['member_request_ids'][0]: b for b in result['summary']['microbatch_metrics']}
        rm = {r['request_id']: r for r in result['summary']['request_metrics']}
        assert set(batches) == set(rm) == set(byid)
        current = []
        for rid, p in byid.items():
            b, r = batches[rid], rm[rid]
            assert b['npu_id'] == p['npu_id'] and b['batch_size'] == 1
            a, z = b['admission_time_ms'], b['completion_time_ms']
            ttft, ideal = z - a, p['ideal_compute_8layers_ms']
            close(ttft, r['processing_latency_ms'])
            close(ideal, b['compute_busy_ms'])
            active = clip(a, z)
            compute = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms']) for l in b['layer_metrics'])
            stall = active - compute
            if abs(stall) < 1e-8:
                stall = 0.0
            l0 = math.fsum(clip(l['compute_start_ms'] - l['io_barrier_wait_ms'], l['compute_start_ms'])
                           for l in b['layer_metrics'] if l['layer'] == 0)
            li = math.fsum(clip(l['compute_start_ms'] - l['io_barrier_wait_ms'], l['compute_start_ms'])
                           for l in b['layer_metrics'] if l['layer'] > 0)
            close(l0 + li, stall)
            current.append(dict(
                policy=policy, request_id=rid, npu_id=p['npu_id'], generation=p['generation'],
                input_order_1based=p['input_order_1based'], role=p['role'], qos_category=p['qos_category'],
                total_tokens=p['total_tokens'], total_length_k=p['total_length_k'], nql_tokens=p['nql_tokens'],
                per_layer_read_gib=p['per_layer_read_gib'], per_layer_compute_ms=p['per_layer_compute_ms'],
                bandwidth_demand_gib_s=p['bandwidth_demand_gib_s'], arrival_time_ms=r['arrival_time_ms'],
                admission_time_ms=a, completion_time_ms=z, ttft_admission_ms=ttft,
                ideal_compute_8layers_ms=ideal, slo_1p5_threshold_ms=1.5 * ideal,
                ttft_over_ideal=ttft / ideal, slo_1p5_passed=ttft <= 1.5 * ideal + 1e-9,
                window_admitted=LEFT <= a < RIGHT, window_active=active > 0,
                window_computed=compute > 0, window_completed=LEFT <= z < RIGHT,
                window_active_ms=active, window_compute_ms=compute, window_stall_ms=stall,
                window_l0_stall_ms=l0, window_l1_l7_stall_ms=li,
                window_active_utilization=compute / active if active else None,
                all_run_stall_ms=b['io_barrier_wait_ms'],
                request_active_utilization=ideal / ttft, ttft_arrival_ms=z - r['arrival_time_ms'],
                compute_extrapolated=p['compute_extrapolated'],
            ))
        executions.extend(current)
        warm = [r for r in current if r['window_admitted']]
        assert len(warm) == metric['slo_count']
        assert sum(x['slo_1p5_passed'] for x in warm) == metric['slo_passed']
        close(sum(r['window_compute_ms'] for r in current) / 16000 * 100, metric['U_percent'])
        for cohort, rows in [('window_admissions', warm), ('all_requests', current)]:
            for role in ('all', 'L', 'S'):
                chosen = [r for r in rows if role == 'all' or r['role'] == role]
                passed = sum(r['slo_1p5_passed'] for r in chosen)
                summary.append(dict(policy=policy, cohort=cohort, role=role, request_count=len(chosen),
                                    passed_count=passed, pass_rate=passed / len(chosen)))
        source_hashes[policy] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                 for name in ('manifest.json.gz', 'result.json.gz')}
    write('request_profiles', profiles)
    write('request_execution', executions)
    write('slo_1p5_summary_from_details', summary)
    definitions = dict(
        scope='8 NPU, 1 SSU. Three policies share the identical 576 requests, order and ring-hash placement. All input requests run to completion.',
        units={'K': '1024 tokens', 'GiB': '2**30 bytes', 'MiB': '2**20 bytes', 'time': 'milliseconds'},
        population={'input_requests': 576, 'execution_records': 1728, 'per_npu': '72 unique (total_tokens,nql_tokens) pairs: 6 L + 66 S'},
        window_ms=[LEFT, RIGHT],
        formulas={'bandwidth_demand_gib_s': 'per_layer_read_gib / (per_layer_compute_ms / 1000)',
                  'ttft_admission_ms': 'completion_time_ms - admission_time_ms',
                  'ideal_compute_8layers_ms': '8 * per_layer_compute_ms',
                  'slo_1p5_threshold_ms': '1.5 * ideal_compute_8layers_ms',
                  'ttft_over_ideal': 'ttft_admission_ms / ideal_compute_8layers_ms; pass if <= 1.5',
                  'window_stall_ms': 'window_active_ms - window_compute_ms; exact interval clipping',
                  'window_active_utilization': 'window_compute_ms/window_active_ms; null when no active overlap'},
        field_notes={'generation': 'zero-based serial position on its NPU after shuffling',
                     'input_order_1based': 'one-based serial position on its NPU after shuffling',
                     'role': 'experiment class L=about200K long read+compute, S=about20K short read+compute',
                     'qos_category': 'original simulator category, independently retained; not the experiment L/S class',
                     'nql_tokens': 'new-query / uncached tokens requiring computation',
                     'hit_prefix_tokens': '128-token-aligned SSD hit prefix; total_tokens = hit_prefix_tokens + nql_tokens',
                     'window_admitted': '2000 <= admission_time_ms < 4000; primary SLO cohort',
                     'window_active': 'admission-to-completion interval has positive overlap with [2000,4000)',
                     'window_computed': 'at least one actual compute interval overlaps [2000,4000)',
                     'window_completed': '2000 <= completion_time_ms < 4000',
                     'all_run_stall_ms': 'sum exposed IO barrier waits over all eight layers of this request',
                     'ttft_arrival_ms': 'also includes the per-NPU admission queue; not the primary TTFT SLO metric',
                     'construction_anchors_json': 'source data compute_us, seq_len_k, nql and interpolation weights; negative weights indicate extrapolation'},
        caveats=['All requests arrive at t=0. Input is an independently shuffled frozen deck per NPU, not a Poisson arrival trace.',
                 'Window admission cohorts differ by policy. The all_requests cohort is the identical 576-request population.',
                 'All 528 short-request compute profiles extrapolate the length dimension below32K; long profiles interpolate. No new physical measurements.',
                 'No compute scaling or extra KV padding. Fixed aligned hit prefix means equal per-layer bytes within each L/S class.',
                 'One SSU makes native ring hash placement trivial: every block maps to the same physical disk.',
                 'short_first is a diagnostic nonpreemptive request-read-size priority, not the Once strategy.'],
        request_and_placement_equality=True, all_per_npu_profiles_unique=True, source_sha256=source_hashes,
    )
    (OUT / 'request_field_definitions.json').write_text(json.dumps(definitions, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'profiles': len(profiles), 'executions': len(executions), 'slo_summary': summary, 'output': str(OUT)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
