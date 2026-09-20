#!/usr/bin/env python3
"""Independent, read-only audit of the 8-NPU AB + one C/D per-card experiment.

Recomputes TTFT and compute occupancy from native timelines. Reports both
the original S -> L exemption and the latest all-cross-request L0 exemption.
Does not modify manifests, native results, or simulator code.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from run_fixed128_32 import profile

ROOT = Path(__file__).resolve().parent


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def clip(a, b, left, right):
    return max(0., min(b, right) - max(a, left))


def group(load):
    return {'long': 'A', 'short': 'B'}.get(load['profile_group'], load['profile_group'])


def same(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)
    else:
        assert a == b, (a, b)


def sweep(intervals, left, right, capacity=40.):
    """One-SSU event integration; simultaneous events apply as one update."""
    events = defaultdict(list)
    events[left]
    events[right]
    for a, b, rate in intervals:
        a, b = max(a, left), min(b, right)
        if b > a:
            events[a].append(rate)
            events[b].append(-rate)
    rate = peak = integral = overload = near = 0.
    times = sorted(events)
    for a, b in zip(times, times[1:]):
        rate = math.fsum([rate, *events[a]])
        assert rate >= -1e-7, rate
        rate = max(rate, 0.)
        dt = b-a
        peak = max(peak, rate)
        integral += dt*rate
        overload += dt if rate > capacity+1e-8 else 0.
        near += dt if rate >= .9*capacity-1e-8 else 0.
    duration = right-left
    return dict(peak_gib_s=peak, mean_gib_s=integral/duration,
                overload_ms=overload, overload_percent=100*overload/duration,
                near90_ms=near, near90_percent=100*near/duration,
                within_capacity=overload == 0.)


def summarize(rows):
    out = {}
    for gid in ['all', 'A', 'B', 'C', 'D']:
        selected = [x for x in rows if gid == 'all' or x['group'] == gid]
        if not selected:
            out[gid] = dict(count=0, passed=0, slo15_percent=None, mean_ttft_ms=None)
            continue
        passed = sum(x['slo15_met'] for x in selected)
        ordered = sorted(x['ttft_ms'] for x in selected)
        out[gid] = dict(count=len(selected), passed=passed,
                        slo15_percent=100*passed/len(selected),
                        mean_ttft_ms=math.fsum(ordered)/len(ordered),
                        p99_ttft_ms=ordered[max(0, math.ceil(.99*len(ordered))-1)],
                        mean_ttft_over_ideal=math.fsum(x['ttft_ms']/x['ideal_ms'] for x in selected)/len(selected))
    return out


def validate_input(man, base):
    assert man['metadata']['num_npu'] == 8 and man['metadata']['num_ssu'] == 1
    assert man['metadata']['n_layers'] == 8
    assert len(man['requests']) == 408
    table = ast.literal_eval((ROOT/'data').read_text())
    original = {q['request_id']: q for q in base['requests']}
    matched = set()
    for q in man['requests']:
        load, rid = q['load'], q['request_id']
        assert load['request_id'] == rid and load['npu_id'] == q['npu_id']
        rebuilt = profile(table, load['ssd_prefix_tokens'], load['nql'])
        for actual, expected in [('total_tokens', 'total_tokens'), ('seq_len_k', 'total_length_k'),
                                 ('per_layer_us', 'compute_us'), ('per_layer_kv_gb', 'read_gib'),
                                 ('required_bw_input_gbps', 'B_gib_s'),
                                 ('profile_construction', 'profile_construction')]:
            same(load[actual], rebuilt[expected])
        same(load['original_compute_us'], load['per_layer_us'])
        layers = man['placements'][q['placement_index']]
        assert len(layers) in (1, 8) and all(layer == layers[0] for layer in layers)
        assert len(layers[0]) == math.ceil(load['ssd_prefix_tokens']/128)
        for idx, (ssu, volume) in enumerate(layers[0]):
            assert ssu == 0
            same(volume, min(128, load['ssd_prefix_tokens']-idx*128)*1408/2**30)
        gid = group(load)
        assert gid in ('A', 'B', 'C', 'D')
        if gid in ('A', 'B'):
            oid = load['original_request_id']
            assert oid in original and oid not in matched
            matched.add(oid)
            old = original[oid]
            assert q['npu_id'] == old['npu_id'] and group(old['load']) == gid
            # Execution IDs, generation, and initial marker may change after insertion.
            for field in old['load']:
                if field not in {'request_id', 'generation', 'initial', 'profile_group'}:
                    same(old['load'][field], load[field])
            same(base['placements'][old['placement_index']], layers)
        else:
            total, nql = (200*1024, 1024) if gid == 'C' else (20*1024, 2048)
            assert load['total_tokens'] == total and load['nql'] == nql
            assert load['role'] == ('L' if gid == 'C' else 'S')
    assert matched == set(original)
    checks = []
    for n in range(8):
        deck = sorted((q for q in man['requests'] if q['npu_id'] == n), key=lambda q:(q['arrival_time_ms'], q['request_id']))
        old_deck = sorted((q for q in base['requests'] if q['npu_id'] == n), key=lambda q:(q['arrival_time_ms'], q['request_id']))
        assert Counter(group(q['load']) for q in deck) == {'A': 7, 'B': 42, 'C': 1, 'D': 1}
        assert [q['load']['original_request_id'] for q in deck if group(q['load']) in ('A', 'B')] == [q['request_id'] for q in old_deck]
        assert [q['request_id'] for q in deck] == [n*1000000+i for i in range(51)]
        checks.append(dict(npu_id=n, counts=dict(Counter(group(q['load']) for q in deck)),
                           C_position=next(i for i,q in enumerate(deck) if group(q['load']) == 'C'),
                           D_position=next(i for i,q in enumerate(deck) if group(q['load']) == 'D')))
    return checks


def audit_case(case):
    man = read(case/'manifest.json.gz')
    result = read(case/'result.json.gz')
    metrics = read(case/'metrics.json')
    requests = {q['request_id']: q for q in man['requests']}
    batches = result['summary']['microbatch_metrics']
    assert len(batches) == 408 and result['summary']['request_count'] == 408
    assert all(result['summary']['invariants'].values())
    current, continuing, boundary, total_prefetch = [], [], [], []
    same_request, all_cross_request = [], []
    records, timelines = [], []
    first_last = []
    for n in range(8):
        lane = sorted((b for b in batches if b['npu_id'] == n), key=lambda b:b['admission_time_ms'])
        assert len(lane) == 51
        expected_ids = sorted(q['request_id'] for q in requests.values() if q['npu_id'] == n)
        assert [b['member_request_ids'][0] for b in lane] == expected_ids
        first_last.append([lane[0]['admission_time_ms'], lane[-1]['completion_time_ms']])
        flat = []
        for prev, b in zip(lane, lane[1:]):
            same(prev['completion_time_ms'], b['admission_time_ms'])
        for b in lane:
            assert len(b['member_request_ids']) == 1
            rid = b['member_request_ids'][0]
            q = requests[rid]['load']
            layers = sorted(b['layer_metrics'], key=lambda l:l['layer'])
            assert [l['layer'] for l in layers] == list(range(8))
            ideal = 8*q['per_layer_us']/1000
            same(b['compute_busy_ms'], ideal)
            same(math.fsum(l['compute_end_ms']-l['compute_start_ms'] for l in layers), ideal)
            ttft = b['completion_time_ms']-b['admission_time_ms']
            records.append(dict(request_id=rid, group=group(q), npu_id=n,
                                admission_ms=b['admission_time_ms'], completion_ms=b['completion_time_ms'],
                                ttft_ms=ttft, ideal_ms=ideal, slo15_met=ttft <= 1.5*ideal+1e-8))
            timelines.append(dict(record=records[-1], layers=layers))
            current.append((b['admission_time_ms'], b['completion_time_ms'], q['per_layer_kv_gb']/(q['per_layer_us']/1e6)))
            flat.extend(dict(request_id=rid, **l) for l in layers)
        for previous, following in zip(flat, flat[1:]):
            old = requests[previous['request_id']]['load']
            new = requests[following['request_id']]['load']
            a, b = previous['compute_start_ms'], previous['compute_end_ms']
            same(a, following['io_start_time_ms'])
            entry = (a, b, new['per_layer_kv_gb']*1000/(b-a))
            is_exempt = previous['request_id'] != following['request_id'] and old['role'] == 'S' and new['role'] == 'L'
            assert not is_exempt or following['layer'] == 0
            (boundary if is_exempt else continuing).append(entry)
            (same_request if previous['request_id'] == following['request_id'] else all_cross_request).append(entry)
            total_prefetch.append(entry)
    windows = {}
    for label, left, right in [('full_run', 0., result['summary']['makespan_ms']), ('warm_2000_4000ms', 2000., 4000.)]:
        per_card = [dict(active_card_ms=0., compute_card_ms=0.) for _ in range(8)]
        by_group = {g:dict(active_card_ms=0., compute_card_ms=0., npu_ids=set()) for g in ('A', 'B', 'C', 'D')}
        for t in timelines:
            r = t['record']
            active = clip(r['admission_ms'], r['completion_ms'], left, right)
            busy = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms'], left, right) for l in t['layers'])
            per_card[r['npu_id']]['active_card_ms'] += active
            per_card[r['npu_id']]['compute_card_ms'] += busy
            by_group[r['group']]['active_card_ms'] += active
            by_group[r['group']]['compute_card_ms'] += busy
            if busy:
                by_group[r['group']]['npu_ids'].add(r['npu_id'])
        for g, row in by_group.items():
            row['npu_ids'] = sorted(row['npu_ids'])
            row['U_percent'] = 100*row['compute_card_ms']/row['active_card_ms'] if row['active_card_ms'] else None
        for row in per_card:
            row['U_percent'] = 100*row['compute_card_ms']/(right-left)
            row['active_U_percent'] = 100*row['compute_card_ms']/row['active_card_ms'] if row['active_card_ms'] else None
        U = 100*math.fsum(r['compute_card_ms'] for r in per_card)/(8*(right-left))
        demand = {key:sweep(values, left, right) for key,values in [('current',current), ('continuing',continuing), ('boundary',boundary), ('total_prefetch',total_prefetch), ('same_request',same_request), ('all_cross_request',all_cross_request)]}
        same(demand['total_prefetch']['mean_gib_s'], demand['continuing']['mean_gib_s']+demand['boundary']['mean_gib_s'])
        same(demand['total_prefetch']['mean_gib_s'], demand['same_request']['mean_gib_s']+demand['all_cross_request']['mean_gib_s'])
        windows[label] = dict(window_ms=[left,right], U_percent=U, by_npu=per_card, by_group=by_group,
                              all_npus_active_entire_window=all(math.isclose(r['active_card_ms'],right-left, abs_tol=1e-7) for r in per_card),
                              demand=demand,
                              all_cross_request_exempt_under_capacity=all(demand[k]['within_capacity'] for k in ['current','same_request']),
                              only_S_to_L_exempt_under_capacity=all(demand[k]['within_capacity'] for k in ['current','continuing']),
                              normal_demand_within_capacity=all(demand[k]['within_capacity'] for k in ['current','same_request']))
    same(windows['warm_2000_4000ms']['U_percent'], metrics['U_percent'])
    assert windows['warm_2000_4000ms']['all_npus_active_entire_window']
    physical = read(case/'receipts.json')
    assert all(physical['observer_checks'].values())
    edges = physical['bin_edges_ms']
    peaks = [max(1000*v/(b-a) for v,a,b in zip(row,edges,edges[1:])) for row in physical['ssu_bin_read_gib']]
    assert max(peaks) <= 40.+1e-6
    return dict(case=str(case), input_fingerprint=man['input_fingerprint'],
                source_sha256={f:hashlib.sha256((case/f).read_bytes()).hexdigest() for f in ('manifest.json.gz','result.json.gz')},
                all_native_invariants_passed=True, physical_observer_checks_passed=True,
                physical_ssu_peak_gib_s=peaks, first_admission_last_completion_by_npu=first_last,
                slo15_all=summarize(records), slo15_warm_admitted=summarize([r for r in records if 2000<=r['admission_ms']<4000]),
                windows=windows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--fifo', type=Path, required=True)
    p.add_argument('--once', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base = read(a.base/'manifest.json.gz')
    fifo = read(a.fifo/'manifest.json.gz')
    once = read(a.once/'manifest.json.gz')
    same(fifo['requests'], once['requests'])
    same(fifo['placements'], once['placements'])
    assert fifo['input_fingerprint'] == once['input_fingerprint']
    inputs = validate_input(fifo, base)
    result = dict(input_checks=inputs, same_input_across_policies=True,
                  original_AB_profiles_placements_order_preserved=True,
                  execution_request_ids_renumbered=True,
                  definitions=dict(ttft='completion minus admission; pre-admission queue time excluded',
                                   slo15='TTFT <= 1.5 * 8 * own per-layer compute time',
                                   U='actual compute intersection with [2000,4000), divided by 8*2000 card-ms',
                                   current='sum current admitted request own V/C, including stalled spans',
                                   continuing='Vnext/Ccurrent during current compute; only S-to-L cross-request L0 removed',
                                   boundary='only S-to-L cross-request L0 contribution; other simultaneous demand retained',
                                   same_request='internal-layer Vnext/Ccurrent only; all cross-request L0 contributions removed',
                                   all_cross_request='all cross-request L0 contributions, in both directions and including same-role boundaries',
                                   latest_underload_criterion='current and same_request independently stay <=40 GiB/s; entire time intervals never removed',
                                   demand_warning='current and continuing are distinct reference curves, not added'),
                  fifo=audit_case(a.fifo), once=audit_case(a.once))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({p:{'U_percent':result[p]['windows']['warm_2000_4000ms']['U_percent'],
                        'slo15_all':result[p]['slo15_all']['all'],
                        'slo15_warm_admitted':result[p]['slo15_warm_admitted']['all'],
                        'full_run_normal_under_capacity':result[p]['windows']['full_run']['normal_demand_within_capacity']}
                      for p in ('fifo','once')}, indent=2))


if __name__ == '__main__':
    main()
