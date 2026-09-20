#!/usr/bin/env python3
"""Event-exact, full-run demand audit for completed strict-underload simulations.

Run after native simulation; never alter or execute the simulator.  Demand
references are not physical service rates or a proof of scheduling feasibility.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_random_multitype import audit as audit_common, read

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/strict_random_underload_20260914'
CAPACITY = 40.0
LINK_CAPACITY = 50.0
EPS = 1e-8


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def weighted_quantile(pairs, probability):
    """Left-continuous inverse time-weighted CDF, including idle zero rates."""
    if not pairs:
        return 0.0
    target = probability * math.fsum(dt for _, dt in pairs)
    cumulative = 0.0
    for value, dt in sorted(pairs):
        cumulative += dt
        if cumulative >= target - 1e-10:
            return value
    return max(value for value, _ in pairs)


def event_sweep(intervals, count, left, right, capacity, near_fraction=0.9):
    """Simultaneous starts/ends aggregate before the following positive interval."""
    if right <= left:
        raise ValueError(f'Empty audit window: {left}, {right}')
    events = defaultdict(lambda: [[] for _ in range(count)])
    events[left]
    events[right]
    for start, end, vector in intervals:
        assert len(vector) == count
        a, b = max(left, start), min(right, end)
        if b <= a:
            continue
        for s, rate in enumerate(vector):
            if rate:
                events[a][s].append(rate)
                events[b][s].append(-rate)
    rates = [0.] * count
    peaks = [0.] * count
    integrals = [0.] * count
    over = [0.] * count
    near = [0.] * count
    pairs = [[] for _ in range(count)]
    fleet_pairs = []
    any_over = any_near = fleet_over = fleet_near = 0.
    fleet_peak = 0.
    times = sorted(events)
    for a, b in zip(times, times[1:]):
        dt = b-a
        rates = [math.fsum([rates[s], *events[a][s]]) for s in range(count)]
        rates = [0. if abs(rate) < EPS else rate for rate in rates]
        assert min(rates) >= -EPS, (a, rates)
        total = math.fsum(rates)
        fleet_peak = max(fleet_peak, total)
        fleet_pairs.append((total, dt))
        fleet_over += dt if total > capacity*count+EPS else 0.
        fleet_near += dt if total >= capacity*count*near_fraction-EPS else 0.
        any_over += dt if max(rates) > capacity+EPS else 0.
        any_near += dt if max(rates) >= capacity*near_fraction-EPS else 0.
        for s, rate in enumerate(rates):
            peaks[s] = max(peaks[s], rate)
            integrals[s] += rate*dt
            over[s] += dt if rate > capacity+EPS else 0.
            near[s] += dt if rate >= capacity*near_fraction-EPS else 0.
            pairs[s].append((rate, dt))
    duration = right-left
    return dict(window_ms=[left, right], duration_ms=duration,
        capacity_per_unit_gib_s=capacity, fleet_capacity_gib_s=count*capacity,
        near_threshold_per_unit_gib_s=capacity*near_fraction,
        near_threshold_fleet_gib_s=capacity*count*near_fraction,
        peak_per_unit_gib_s=peaks, mean_per_unit_gib_s=[x/duration for x in integrals],
        p95_per_unit_gib_s=[weighted_quantile(x, .95) for x in pairs],
        p99_per_unit_gib_s=[weighted_quantile(x, .99) for x in pairs],
        over_capacity_ms_per_unit=over, near_capacity_ms_per_unit=near,
        over_capacity_percent_per_unit=[100*x/duration for x in over],
        near_capacity_percent_per_unit=[100*x/duration for x in near],
        any_unit_over_capacity_ms=any_over, any_unit_over_capacity_percent=100*any_over/duration,
        any_unit_near_capacity_ms=any_near, any_unit_near_capacity_percent=100*any_near/duration,
        fleet_peak_gib_s=fleet_peak, fleet_mean_gib_s=math.fsum(integrals)/duration,
        fleet_p95_gib_s=weighted_quantile(fleet_pairs, .95),
        fleet_p99_gib_s=weighted_quantile(fleet_pairs, .99),
        fleet_over_capacity_ms=fleet_over, fleet_over_capacity_percent=100*fleet_over/duration,
        fleet_near_capacity_ms=fleet_near, fleet_near_capacity_percent=100*fleet_near/duration,
        every_unit_within_capacity=max(peaks) <= capacity+EPS)


def overlapping(intervals, left, right):
    return [(max(a,left), min(b,right)) for a,b in intervals if min(b,right)>max(a,left)]


def interval_counts(intervals, left, right):
    clipped = overlapping(intervals, left, right)
    if not clipped:
        return dict(jobs_overlapping=0, card_ms=0., any_job_ms=0., max_concurrent_jobs=0)
    events = defaultdict(int)
    for a,b in clipped:
        events[a] += 1
        events[b] -= 1
    active = peak = 0
    union = 0.
    times = sorted(events)
    for a,b in zip(times, times[1:]):
        active += events[a]
        peak = max(active, peak)
        union += b-a if active else 0.
    return dict(jobs_overlapping=len(clipped), card_ms=math.fsum(b-a for a,b in clipped),
                any_job_ms=union, max_concurrent_jobs=peak)


def build_full_intervals(manifest, result, metadata):
    qs = {q['request_id']:q for q in manifest['requests']}
    S, N = metadata['num_ssu'], metadata['num_npu']
    volumes = {}
    for rid, q in qs.items():
        placement = manifest['placements'][q['placement_index']][0]
        volumes[rid] = [math.fsum(float(v) for disk,v in placement if int(disk)==s) for s in range(S)]
    current_intervals, prefetch_intervals, link_intervals = [], [], []
    cold_jobs, missed_jobs, impossible_jobs = [], [], []
    all_job_count = 0
    per_npu_prefetch_max = [0.]*N
    expected = 0
    for npu in range(N):
        batches = sorted((b for b in result['summary']['microbatch_metrics'] if b['npu_id']==npu),
                         key=lambda b:b['admission_time_ms'])
        assert batches
        flat = []
        for batch in batches:
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            own_c = qs[rid]['load']['per_layer_us']/1e6
            current_intervals.append((batch['admission_time_ms'], batch['completion_time_ms'],
                                      [v/own_c for v in volumes[rid]]))
            ordered = sorted(batch['layer_metrics'], key=lambda layer:layer['layer'])
            assert len(ordered)==metadata['n_layers']
            flat.extend(dict(request_id=rid, **layer) for layer in ordered)
        expected += len(flat)-1
        first = flat[0]
        cold_jobs.append(dict(npu_id=npu, request_id=first['request_id'], layer=first['layer'],
            io_start_time_ms=first['io_start_time_ms'], io_ready_time_ms=first['io_ready_time_ms'],
            compute_start_ms=first['compute_start_ms'], total_read_gib=math.fsum(volumes[first['request_id']]),
            note='The first L0 has no preceding compute window: no Vnext/Ccurrent reference is defined.'))
        for previous, following in zip(flat, flat[1:]):
            start, deadline = previous['compute_start_ms'], previous['compute_end_ms']
            assert math.isclose(start, following['io_start_time_ms'], abs_tol=1e-7)
            c_seconds = (deadline-start)/1000
            assert c_seconds > 0
            rid = following['request_id']
            rates = [v/c_seconds for v in volumes[rid]]
            npu_rate = math.fsum(rates)
            prefetch_intervals.append((start, deadline, rates))
            link_vector = [0.]*N
            link_vector[npu] = npu_rate
            link_intervals.append((start, deadline, link_vector))
            per_npu_prefetch_max[npu] = max(per_npu_prefetch_max[npu], npu_rate)
            all_job_count += 1
            ready = following['io_ready_time_ms']
            compute_start = following['compute_start_ms']
            lower_ms = max(max(volumes[rid])/CAPACITY, math.fsum(volumes[rid])/LINK_CAPACITY)*1000
            job = dict(npu_id=npu, from_request_id=previous['request_id'], to_request_id=rid,
                from_group=qs[previous['request_id']]['load']['profile_group'],
                to_group=qs[rid]['load']['profile_group'], to_layer=following['layer'],
                cross_request=previous['request_id']!=rid, release_ms=start, deadline_ms=deadline,
                io_ready_ms=ready, next_compute_start_ms=compute_start,
                available_compute_ms=deadline-start, next_layer_gib=math.fsum(volumes[rid]),
                required_per_ssu_gib_s=rates, required_npu_link_gib_s=npu_rate,
                io_deadline_lateness_ms=max(0.,ready-deadline),
                actual_stall_ms=max(0.,compute_start-deadline),
                exclusive_transfer_lower_bound_ms=lower_ms,
                unavoidable_transfer_stall_lower_bound_ms=max(0.,lower_ms-(deadline-start)))
            if ready > deadline+EPS:
                missed_jobs.append(job)
            if lower_ms > deadline-start+EPS:
                impossible_jobs.append(job)
    assert all_job_count == expected
    return dict(current=current_intervals, prefetch=prefetch_intervals, link=link_intervals,
                cold_jobs=cold_jobs, missed_jobs=missed_jobs, impossible_jobs=impossible_jobs,
                all_prefetch_job_count=all_job_count, max_prefetch_rate_per_npu_gib_s=per_npu_prefetch_max)


def audit(case, near_fraction=.9, max_near_percent=5.):
    base = audit_common(case)
    man, result, meta, metrics = (read(case/name) for name in
        ('manifest.json.gz','result.json.gz','metadata.json','metrics.json'))
    S, N = meta['num_ssu'], meta['num_npu']
    makespan = result['summary']['makespan_ms']
    intervals = build_full_intervals(man, result, meta)
    cold_spans = [(j['io_start_time_ms'],j['io_ready_time_ms']) for j in intervals['cold_jobs']]
    debt_spans = [(j['deadline_ms'],j['io_ready_ms']) for j in intervals['missed_jobs']]
    windows = {}
    requested_windows = [('full_run',0.,makespan),('first_4500ms',0.,4500.),
                         ('broad_500_4000ms',500.,4000.),('warm_2000_4000ms',2000.,4000.)]
    for label,left,requested_right in requested_windows:
        right = min(requested_right,makespan)
        if right <= left:
            windows[label] = dict(requested_window_ms=[left,requested_right], unavailable=True)
            continue
        row = dict(requested_window_ms=[left,requested_right], observed_window_ms=[left,right],
                   requested_end_after_makespan=requested_right>makespan,
                   current=event_sweep(intervals['current'],S,left,right,CAPACITY,near_fraction),
                   prefetch=event_sweep(intervals['prefetch'],S,left,right,CAPACITY,near_fraction),
                   npu_prefetch_link=event_sweep(intervals['link'],N,left,right,LINK_CAPACITY,near_fraction),
                   cold_initial_io=interval_counts(cold_spans,left,right),
                   overdue_io_debt=interval_counts(debt_spans,left,right))
        row['storage_references_within_capacity'] = all(row[name]['every_unit_within_capacity'] for name in ('current','prefetch'))
        row['storage_near_capacity_time_within_limit'] = all(row[name]['any_unit_near_capacity_percent']<=max_near_percent+EPS for name in ('current','prefetch'))
        row['npu_prefetch_links_within_capacity'] = row['npu_prefetch_link']['every_unit_within_capacity']
        windows[label] = row
    full = windows['full_run']
    native_invariants = result['summary']['invariants']
    assert native_invariants and all(value is True for value in native_invariants.values())
    physical = base['physical_ssd']
    assert physical and physical['observer_checks']
    assert all(value is True for value in physical['observer_checks'].values())
    all_windows_available = all(not row.get('unavailable',False) for row in windows.values())
    all_window_near_max = max(row[name]['any_unit_near_capacity_percent']
        for row in windows.values() if not row.get('unavailable',False) for name in ('current','prefetch'))
    passes = dict(storage_references_within_capacity_full_run=full['storage_references_within_capacity'],
        storage_near_capacity_time_within_limit_full_run=full['storage_near_capacity_time_within_limit'],
        all_four_audit_windows_available=all_windows_available,
        storage_near_capacity_time_within_limit_all_windows=all_windows_available and all_window_near_max<=max_near_percent+EPS,
        npu_prefetch_links_within_capacity_full_run=full['npu_prefetch_links_within_capacity'],
        no_exclusive_transfer_infeasible_prefetch_jobs=not intervals['impossible_jobs'],
        all_native_invariants_passed=all(value is True for value in native_invariants.values()),
        all_physical_observer_checks_passed=all(value is True for value in physical['observer_checks'].values()),
        all_npus_all_profile_groups_computed_in_warm_window=base['all_npus_all_profile_groups_computed'],
        all_npus_active_entire_warm_window=base['all_npus_active_entire_window'])
    # Capacity/near-time failures outside the selected warm interval reject the case.
    out = dict(schema_version=1, case=str(case.relative_to(ROOT)), name=metrics['name'], policy=metrics['policy'],
        seed=meta['seed'], num_npu=N, num_ssu=S, makespan_ms=makespan,
        threshold=dict(ssu_gib_s=CAPACITY,npu_link_gib_s=LINK_CAPACITY,near_capacity_fraction=near_fraction,
                       max_any_ssu_near_capacity_time_percent=max_near_percent,tolerance_gib_s=EPS),
        strict_candidate_pass=all(passes.values()), pass_checks=passes, windows=windows,
        all_windows_max_any_ssu_near_capacity_percent=all_window_near_max,
        native_invariants=native_invariants,
        source_sha256={name:hashlib.sha256((case/name).read_bytes()).hexdigest() for name in
                       ('manifest.json.gz','result.json.gz','metadata.json','metrics.json')},
        common_invariant_audit='deadline_audit.json',
        all_npus_all_profile_groups_computed=base['all_npus_all_profile_groups_computed'],
        warm_by_group=base['warm_by_group'], input_groups=base['input_groups'],
        input_lane_checks=base['input_lane_checks'], physical_ssd=base['physical_ssd'],
        nominal_static_upper_gib_s=base['nominal_static_upper_gib_s'],
        strict_any_prefetch_combination_upper_gib_s=base['strict_any_prefetch_combination_upper_gib_s'],
        U_percent=metrics['U_percent'], slo_1p5_percent=metrics['slo_1p5_percent'],
        all_prefetch_job_count=intervals['all_prefetch_job_count'],
        max_prefetch_rate_per_npu_gib_s=intervals['max_prefetch_rate_per_npu_gib_s'],
        initial_cold_l0_jobs=intervals['cold_jobs'],
        missed_io_deadline_job_count=len(intervals['missed_jobs']),
        missed_io_deadline_jobs=intervals['missed_jobs'],
        individually_infeasible_prefetch_job_count=len(intervals['impossible_jobs']),
        individually_infeasible_prefetch_jobs=intervals['impossible_jobs'],
        definitions={
            'current':'During each request admission-to-completion span, own per-SSU V/own original C, including stalled spans.',
            'prefetch':'During each current compute span, next-layer per-SSU V/current C, including next-request L0; built for the entire completed run.',
            'npu_prefetch_link':'During each current compute span, total next-layer V/current C on that NPU; exclusive link capacity is 50 GiB/s.',
            'time_statistics':'Exact event intervals, half-open [start,end); weighted quantiles include zero demand. Near means >= 90% capacity by default; any-unit times are union time, never summed across SSUs.',
            'strict_screen':'Both storage references must remain within capacity on every SSU over the complete run. For each reference, the any-SSU near-capacity fraction must satisfy threshold.max_any_ssu_near_capacity_time_percent in each of the four audit windows; near threshold is capacity multiplied by threshold.near_capacity_fraction. Warm occupancy/group coverage, per-NPU link feasibility, and native/physical invariants also required.',
            'physical_service':'Physical SSD service runs at up to 40 GiB/s while serving IO and can touch capacity while these demand references remain under capacity.',
            'cold_l0_limit':'The initial L0 on each NPU has no preceding compute window and therefore no prefetch reference. It remains included in current-request demand, physical receipts, and cold-IO spans.',
            'overdue_debt_limit':'At a missed prefetch deadline, Vnext/Ccurrent stops although residual IO can remain. Every missed IO job and its deadline-to-IO-ready interval is recorded. Neither reference includes a remaining-byte urgency rate after a deadline; the audit is not an all-interval deadline-feasibility proof.',
            'causality_limit':'IO stalls in a passing case do not alone establish FIFO causation; comparison with another scheduler or a trace-based HOL proof is needed.'})
    write_json(case/'strict_audit.json',out)
    return out


def summary_row(audit_result):
    d = audit_result
    full = d['windows']['full_run']
    row = {key:d[key] for key in ('name','case','policy','seed','num_ssu','num_npu','U_percent','slo_1p5_percent','strict_candidate_pass')}
    row.update(d['pass_checks'])
    row.update(makespan_ms=d['makespan_ms'],
        full_current_peak_per_ssu_gib_s=max(full['current']['peak_per_unit_gib_s']),
        full_prefetch_peak_per_ssu_gib_s=max(full['prefetch']['peak_per_unit_gib_s']),
        full_current_any_ssu_over_capacity_ms=full['current']['any_unit_over_capacity_ms'],
        full_prefetch_any_ssu_over_capacity_ms=full['prefetch']['any_unit_over_capacity_ms'],
        full_current_any_ssu_near_capacity_percent=full['current']['any_unit_near_capacity_percent'],
        full_prefetch_any_ssu_near_capacity_percent=full['prefetch']['any_unit_near_capacity_percent'],
        all_windows_max_any_ssu_near_capacity_percent=d['all_windows_max_any_ssu_near_capacity_percent'],
        full_current_fleet_peak_gib_s=full['current']['fleet_peak_gib_s'],
        full_prefetch_fleet_peak_gib_s=full['prefetch']['fleet_peak_gib_s'],
        max_prefetch_npu_gib_s=max(d['max_prefetch_rate_per_npu_gib_s']),
        individually_infeasible_prefetch_job_count=d['individually_infeasible_prefetch_job_count'],
        missed_io_deadline_job_count=d['missed_io_deadline_job_count'],
        full_overdue_io_card_ms=full['overdue_io_debt']['card_ms'])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--case',type=Path,action='append')
    parser.add_argument('--near-fraction',type=float,default=.9)
    parser.add_argument('--max-near-percent',type=float,default=5.)
    args = parser.parse_args()
    if not 0 < args.near_fraction <= 1 or not 0 <= args.max_near_percent <= 100:
        parser.error('near-fraction must be (0,1], max-near-percent must be [0,100]')
    cases = args.case or [p.parent for p in sorted((args.root/'native').glob('*/metrics.json'))]
    rows = []
    for case in cases:
        case = case.resolve()
        if not all((case/name).exists() for name in ('manifest.json.gz','result.json.gz','metadata.json','metrics.json')):
            continue
        rows.append(summary_row(audit(case,args.near_fraction,args.max_near_percent)))
    if rows:
        args.root.mkdir(parents=True,exist_ok=True)
        write_json(args.root/'native_strict_summary.json',rows)
        with (args.root/'native_strict_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False))


if __name__ == '__main__':
    main()
