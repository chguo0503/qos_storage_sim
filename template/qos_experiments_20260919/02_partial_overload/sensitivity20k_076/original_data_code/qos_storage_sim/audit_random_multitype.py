#!/usr/bin/env python3
"""Audit completed random multi-profile runs; never run or alter the simulator.

Current demand sums each active request's own V/C, including its exposed stall.
Prefetch reference instead spreads the next layer's bytes across the current
layer's compute interval. Neither is actual SSD traffic or a feasibility proof.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/random_multitype_search_20260914'
CAP = 40.0
LINK = 50.0
EPS = 1e-8


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def sweep(intervals, ssu_count, left, right):
    events = defaultdict(lambda: [[] for _ in range(ssu_count)])
    events[left]; events[right]
    for start, end, rates in intervals:
        x, y = max(start, left), min(end, right)
        if y <= x:
            continue
        for s, rate in enumerate(rates):
            events[x][s].append(rate)
            events[y][s].append(-rate)
    rates = [0.] * ssu_count
    peaks = [0.] * ssu_count
    over = [0.] * ssu_count
    integral = [0.] * ssu_count
    any_over = 0.
    times = sorted(events)
    for a, b in zip(times, times[1:]):
        rates = [math.fsum([rates[s], *events[a][s]]) for s in range(ssu_count)]
        for s in range(ssu_count):
            peaks[s] = max(peaks[s], rates[s])
            over[s] += b - a if rates[s] > CAP + EPS else 0.
            integral[s] += (b - a) * rates[s]
        if max(rates) > CAP + EPS:
            any_over += b - a
    return dict(peak_per_ssu_gib_s=peaks, mean_per_ssu_gib_s=[x/(right-left) for x in integral],
                over40_ms_per_ssu=over, any_ssu_over40_ms=any_over,
                under40_entire_window=max(peaks) <= CAP + EPS)


def audit(case):
    man = read(case / 'manifest.json.gz')
    result = read(case / 'result.json.gz')
    meta = read(case / 'metadata.json')
    metrics = read(case / 'metrics.json')
    qs = {q['request_id']: q for q in man['requests']}
    S, N = meta['num_ssu'], meta['num_npu']
    layers = meta['n_layers']
    left, right = metrics['window_ms']
    clip = lambda a, b: max(0., min(b, right) - max(a, left))
    volumes, original_rates, minima, input_checks = {}, {}, [], []
    group_inputs = defaultdict(list)
    own_impossible = []
    for rid, q in qs.items():
        load = q['load']
        placement = man['placements'][q['placement_index']]
        layer = placement[0]
        assert len(placement) in (1, layers)
        assert all(x == layer for x in placement)
        assert 32*1024 <= load['total_tokens'] <= 200*1024
        assert load['ssd_prefix_tokens'] + load['nql'] == load['total_tokens']
        assert 64 <= load['nql'] <= 4096
        assert not load['profile_construction'].get('extrapolated', False)
        assert load['profile_construction'].get('compute_scale', 1) == 1
        C = load['per_layer_us']/1e6
        volumes[rid] = [math.fsum(float(v) for disk, v in layer if int(disk) == s) for s in range(S)]
        assert math.isclose(math.fsum(volumes[rid]), load['per_layer_kv_gb'], abs_tol=1e-12)
        assert math.isclose(math.fsum(volumes[rid]), load['ssd_prefix_tokens']*1408/2**30, abs_tol=1e-12)
        assert math.isclose(load['original_compute_us'], load['per_layer_us'], abs_tol=1e-9)
        original_rates[rid] = [v/C for v in volumes[rid]]
        solo = max(max(volumes[rid])/CAP, math.fsum(volumes[rid])/LINK)
        group_inputs[load['profile_group']].append(rid)
        if solo > C + EPS/1000:
            own_impossible.append(dict(request_id=rid, npu_id=q['npu_id'], group=load['profile_group'],
                compute_ms=C*1000, exclusive_storage_and_link_lower_bound_ms=solo*1000,
                unavoidable_internal_stall_lower_bound_ms=(solo-C)*1000,
                required_per_ssu_gib_s=original_rates[rid], total_required_gib_s=sum(original_rates[rid])))
    nominal_upper = [0.] * S
    strict_upper = [0.] * S
    for n in range(N):
        deck = [q for q in qs.values() if q['npu_id'] == n]
        assert deck
        combinations = [(q['load']['total_tokens'], q['load']['nql']) for q in deck]
        assert len(set(combinations)) == len(combinations)
        assert {q['load']['role'] for q in deck} == {'L', 'S'}
        min_C = min(q['load']['per_layer_us']/1e6 for q in deck)
        minima.append(min_C)
        for s in range(S):
            nominal_upper[s] += max(original_rates[q['request_id']][s] for q in deck)
            strict_upper[s] += max(volumes[q['request_id']][s] for q in deck)/min_C
        input_checks.append(dict(npu_id=n, unique_profiles=len(set(combinations)), request_count=len(deck),
            group_counts=dict(Counter(q['load']['profile_group'] for q in deck))))
    assert all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(nominal_upper, meta['per_ssu_static_upper_bound_gib_s']))

    def fresh():
        return dict(active_card_ms=0., compute_card_ms=0., internal_stall_card_ms=0., l0_stall_card_ms=0.,
                    warm_admitted=0, computed_requests=0, computed_npus=set(),
                    stall_by_layer_card_ms={str(l):0. for l in range(layers)})
    groups = defaultdict(fresh)
    roles = defaultdict(fresh)
    per_npu = defaultdict(lambda: defaultdict(float))
    nominal_intervals, prefetch_intervals, transitions, impossible_jobs = [], [], [], []
    for n in range(N):
        batches = sorted((b for b in result['summary']['microbatch_metrics'] if b['npu_id'] == n),
                         key=lambda b:b['admission_time_ms'])
        flat = []
        for batch in batches:
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            q = qs[rid]['load']
            gid, role = q['profile_group'], q['role']
            active = clip(batch['admission_time_ms'], batch['completion_time_ms'])
            computation = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms']) for l in batch['layer_metrics'])
            for row in (groups[gid], roles[role]):
                row['active_card_ms'] += active
                row['compute_card_ms'] += computation
                row['warm_admitted'] += left <= batch['admission_time_ms'] < right
                row['computed_requests'] += computation > 0
                if computation > 0:
                    row['computed_npus'].add(n)
            per_npu[n][role] += computation
            nominal_intervals.append((batch['admission_time_ms'], batch['completion_time_ms'], original_rates[rid]))
            previous = batch['admission_time_ms']
            for layer in sorted(batch['layer_metrics'], key=lambda l:l['layer']):
                flat.append(dict(request_id=rid, **layer))
                stall = clip(previous, layer['compute_start_ms'])
                for row in (groups[gid], roles[role]):
                    row['l0_stall_card_ms' if layer['layer'] == 0 else 'internal_stall_card_ms'] += stall
                    row['stall_by_layer_card_ms'][str(layer['layer'])] += stall
                previous = layer['compute_end_ms']
        for current, following in zip(flat, flat[1:]):
            start, end = current['compute_start_ms'], current['compute_end_ms']
            C = (end-start)/1000
            assert math.isclose(start, following['io_start_time_ms'], abs_tol=1e-7)
            if min(end, right) <= max(start, left):
                continue
            rid = following['request_id']
            rates = [v/C for v in volumes[rid]]
            prefetch_intervals.append((start, end, rates))
            v = volumes[rid]
            lower = max(max(v)/CAP, math.fsum(v)/LINK)*1000
            job = dict(npu_id=n, from_request_id=current['request_id'], to_request_id=rid,
                from_group=qs[current['request_id']]['load']['profile_group'], to_group=qs[rid]['load']['profile_group'],
                from_role=qs[current['request_id']]['load']['role'], to_role=qs[rid]['load']['role'],
                layer=following['layer'], start_ms=start, deadline_ms=end,
                next_compute_start_ms=following['compute_start_ms'], prefetch_rates_gib_s=rates,
                complete_compute_window=left <= start < end <= right,
                exclusive_storage_and_link_lower_bound_ms=lower,
                unavoidable_stall_lower_bound_ms=max(0., lower-(end-start)),
                actual_stall_ms=max(0., following['compute_start_ms']-end),
                clipped_actual_stall_card_ms=clip(end, following['compute_start_ms']))
            if current['request_id'] != rid:
                transitions.append(job)
            if job['unavoidable_stall_lower_bound_ms'] > EPS:
                impossible_jobs.append(job)
    for mapping in (groups, roles):
        for row in mapping.values():
            row['computed_npus'] = sorted(row['computed_npus'])
            row['total_stall_card_ms'] = row['l0_stall_card_ms'] + row['internal_stall_card_ms']
            row['U_percent'] = 100*row['compute_card_ms']/row['active_card_ms'] if row['active_card_ms'] else None
            row['internal_fraction_of_stall_percent'] = (100*row['internal_stall_card_ms']/row['total_stall_card_ms']
                                                          if row['total_stall_card_ms'] else 0.)
            assert math.isclose(row['compute_card_ms']+row['total_stall_card_ms'], row['active_card_ms'], abs_tol=1e-6)
    fleet_compute = math.fsum(row['compute_card_ms'] for row in groups.values())
    assert math.isclose(fleet_compute/(N*(right-left))*100, metrics['U_percent'], abs_tol=1e-7)
    current = sweep(nominal_intervals, S, left, right)
    prefetch = sweep(prefetch_intervals, S, left, right)
    physical = None
    receipt_path = case/'receipts.json'
    if receipt_path.exists():
        rec = read(receipt_path)
        widths = [b-a for a, b in zip(rec['bin_edges_ms'], rec['bin_edges_ms'][1:])]
        peak = [max(x/(dt/1000) for x, dt in zip(row, widths)) for row in rec['ssu_bin_read_gib']]
        assert max(peak) <= CAP + 1e-6
        assert all(rec['observer_checks'].values())
        physical = dict(peak_2ms_per_ssu_gib_s=peak, mean_per_ssu_gib_s=rec['window_ssu_read_bandwidth_gib_s'],
                        observer_checks=rec['observer_checks'], exact_io_service_no_capacity_breach=True)
    complete_impossible = [j for j in impossible_jobs if j['complete_compute_window']]
    group_specs = {}
    for gid, rids in group_inputs.items():
        loads = [qs[rid]['load'] for rid in rids]
        group_specs[gid] = dict(role=loads[0]['role'], input_requests=len(loads),
            total_k_range=[min(x['total_tokens']/1024 for x in loads), max(x['total_tokens']/1024 for x in loads)],
            nql_range=[min(x['nql'] for x in loads), max(x['nql'] for x in loads)],
            compute_ms_range=[min(x['per_layer_us']/1000 for x in loads),max(x['per_layer_us']/1000 for x in loads)],
            direct_data_rows=sum(not x['constructed_profile'] for x in loads),
            max_own_per_ssu_demand_gib_s=max(max(original_rates[rid]) for rid in rids))
    out = dict(schema_version=1, case=str(case.relative_to(ROOT)), name=metrics['name'], policy=metrics['policy'],
        seed=meta['seed'], num_ssu=S, num_npu=N, window_ms=[left,right],
        source_sha256={name:hashlib.sha256((case/name).read_bytes()).hexdigest()
                       for name in ('manifest.json.gz','result.json.gz','metadata.json','metrics.json')},
        nominal_static_upper_gib_s=nominal_upper, nominal_static_underload=max(nominal_upper)<=CAP+EPS,
        strict_any_prefetch_combination_upper_gib_s=strict_upper,
        strict_any_prefetch_combination_underload=max(strict_upper)<=CAP+EPS,
        realized_current_request_demand=current, realized_prefetch_deadline_reference=prefetch,
        original_single_request_infeasible_count=len(own_impossible), original_single_request_infeasible=own_impossible,
        max_original_single_request_per_ssu_demand_gib_s=[max(rates[s] for rates in original_rates.values()) for s in range(S)],
        max_original_single_request_npu_demand_gib_s=max(math.fsum(rates) for rates in original_rates.values()),
        complete_individually_impossible_prefetch_count=len(complete_impossible),
        sum_exclusive_unavoidable_stall_lower_bound_card_ms=sum(j['unavoidable_stall_lower_bound_ms'] for j in complete_impossible),
        individually_impossible_prefetch_jobs=impossible_jobs, transitions=transitions,
        warm_by_group=dict(groups), warm_by_role=dict(roles), input_groups=group_specs, input_lane_checks=input_checks,
        all_npus_both_roles_computed=all(row['L']>0 and row['S']>0 for row in per_npu.values()) and len(per_npu)==N,
        all_npus_all_profile_groups_computed=all(row['computed_npus']==list(range(N)) for row in groups.values()) and set(groups)==set(group_inputs),
        all_npus_active_entire_window=metrics['all_active'], all_inputs_unique_per_npu=True,
        all_profiles_within_measured_grid=True, all_exact_total_lengths=True, physical_ssd=physical,
        definitions={
            'current_demand':'During each admission-to-completion span, sum its own per-SSU V / own original C; includes stalled spans.',
            'prefetch_reference':'During each actual compute span only, sum next layer per-SSU V / current C; includes next-request L0.',
            'strict_upper':'Per SSU sum over NPU: maximum any next V divided by minimum any current C in its frozen deck.',
            'unavoidable_lower_bound':'max(max_s V_s/40, total V/50) minus available C; storage/link overlap allowed; ignores small packet latency.',
            'limits':'A demand reference above 40 is not actual service above 40 and alone does not prove deadline infeasibility; internal stall alone does not prove FIFO causation.',
            'workload':'All t=0, fixed NPU, serial batch=1; independent random per-NPU permutation; seed and selected-candidate scope matter.'})
    write(case/'deadline_audit.json',out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--case',type=Path,action='append')
    args = parser.parse_args()
    rows = []
    cases = args.case or [p.parent for p in sorted(args.root.glob('*/*/metrics.json'))]
    for case in cases:
        if not all((case/name).exists() for name in ('result.json.gz','manifest.json.gz','metadata.json','metrics.json')):
            continue
        d = audit(case)
        m = read(case/'metrics.json')
        r = {k:m[k] for k in ('name','policy','seed','num_ssu','U_percent','short_U_percent','long_U_percent','slo_1p5_percent','all_active','all_npus_both_roles_computed')}
        r.update(case=str(case.relative_to(ROOT)),
            input_mean_total_gib_s=sum(m['input_average_gib_s']),
            nominal_static_upper=max(d['nominal_static_upper_gib_s']),
            strict_prefetch_upper=max(d['strict_any_prefetch_combination_upper_gib_s']),
            current_peak=max(d['realized_current_request_demand']['peak_per_ssu_gib_s']),
            current_over_ms=d['realized_current_request_demand']['any_ssu_over40_ms'],
            prefetch_peak=max(d['realized_prefetch_deadline_reference']['peak_per_ssu_gib_s']),
            prefetch_over_ms=d['realized_prefetch_deadline_reference']['any_ssu_over40_ms'],
            original_single_infeasible=d['original_single_request_infeasible_count'],
            impossible_prefetch_jobs=d['complete_individually_impossible_prefetch_count'],
            impossible_stall_lower_bound_card_ms=d['sum_exclusive_unavoidable_stall_lower_bound_card_ms'],
            short_internal_stall_card_ms=d['warm_by_role']['S']['internal_stall_card_ms'],
            short_l0_stall_card_ms=d['warm_by_role']['S']['l0_stall_card_ms'],
            short_internal_stall_fraction_percent=d['warm_by_role']['S']['internal_fraction_of_stall_percent'],
            long_internal_stall_card_ms=d['warm_by_role']['L']['internal_stall_card_ms'],
            long_l0_stall_card_ms=d['warm_by_role']['L']['l0_stall_card_ms'],
            all_npus_all_profile_groups_computed=d['all_npus_all_profile_groups_computed'])
        rows.append(r)
    args.root.mkdir(parents=True, exist_ok=True)
    write(args.root/'native_summary.json',rows)
    with (args.root/'native_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['case'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False))


if __name__ == '__main__':
    main()
