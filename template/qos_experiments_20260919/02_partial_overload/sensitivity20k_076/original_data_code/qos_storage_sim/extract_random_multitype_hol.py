#!/usr/bin/env python3
"""Find a verified, locally avoidable FIFO short-layer stall in saved receipts.

This is offline arithmetic over an actual run, not a rerun or a new policy.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'results/random_multitype_search_20260914/screen/fixed_three_seed7_fifo'


def read(path):
    with (gzip.open(path,'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=DEFAULT)
    parser.add_argument('--short-group',default='g0')
    parser.add_argument('--long-group',default='g2')
    parser.add_argument('--minimum-short-slack-ms',type=float,default=.05)
    args = parser.parse_args()
    case = args.case
    result, manifest, receipts, metrics = [read(case/name) for name in
        ('result.json.gz','manifest.json.gz','receipts.json','metrics.json')]
    assert receipts['num_ssu'] == 1
    assert receipts['probe_policy'] == 'fifo'
    assert all(receipts['observer_checks'].values())
    assert receipts['observer_counters']['nonzero_path_io'] == 0
    left, right = metrics['window_ms']
    summaries = {}
    loads = {q['request_id']:q['load'] for q in manifest['requests']}
    for batch in result['summary']['microbatch_metrics']:
        deadline = batch['admission_time_ms']
        rid = batch['member_request_ids'][0]
        for layer in sorted(batch['layer_metrics'],key=lambda x:x['layer']):
            summaries[rid,layer['layer']] = {**layer,'previous_compute_end_ms':deadline}
            deadline = layer['compute_end_ms']
    rows = []
    issue_ms = result['summary']['client_issue_interval_us']/1000
    cap = receipts['ssu_capacity_gib_s']
    link_cap = receipts['npu_link_capacity_gib_s']
    max_io_gib = 176*1024/2**30
    link_tail_ms = max_io_gib/link_cap*1000
    for receipt in receipts['layers']:
        rid, layer = receipt['request_id'],receipt['layer']
        load, measured = loads[rid],summaries[rid,layer]
        row = dict(**receipt, role=load['role'], profile_group=load['profile_group'],
            total_tokens=load['total_tokens'], total_length_k=load['total_tokens']/1024,
            nql=load['nql'], per_layer_compute_ms=load['per_layer_us']/1000,
            io_release_ms=measured['io_start_time_ms'],
            compute_deadline_ms=measured['previous_compute_end_ms'],
            layer_stall_ms=measured['compute_start_ms']-measured['previous_compute_end_ms'],
            minimum_ssd_service_ms=receipt['bytes_gib']/cap*1000,
            bytes_mib=receipt['bytes_gib']*1024,
            all_io_submitted_by_upper_bound_ms=measured['io_start_time_ms']+receipt['completed_blocks']*issue_ms)
        row['span_equals_bytes_over_capacity'] = math.isclose(row['last_ssd_end_ms']-row['first_ssd_start_ms'],
                                                            row['minimum_ssd_service_ms'],abs_tol=1e-7)
        rows.append(row)
    rows.sort(key=lambda r:r['first_ssd_start_ms'])
    candidates = []
    for index, short in enumerate(rows):
        if (short['profile_group'] != args.short_group or short['layer'] == 0 or
            short['layer_stall_ms'] <= 1e-8 or not short['span_equals_bytes_over_capacity'] or
            not left < short['io_release_ms'] < short['last_link_end_ms'] < right):
            continue
        longs = []
        boundary = short['first_ssd_start_ms']
        for j in range(index-1,-1,-1):
            previous = rows[j]
            if (previous['profile_group'] != args.long_group or previous['layer'] == 0 or
                not previous['span_equals_bytes_over_capacity'] or
                not math.isclose(previous['last_ssd_end_ms'],boundary,abs_tol=1e-7)):
                break
            longs.insert(0,previous)
            boundary = previous['first_ssd_start_ms']
            if short['all_io_submitted_by_upper_bound_ms'] > boundary:
                continue
            cursor = boundary+short['minimum_ssd_service_ms']
            slack = short['compute_deadline_ms']-cursor-link_tail_ms
            if slack < args.minimum_short_slack_ms:
                continue
            reordered = []
            okay = True
            for long in longs:
                okay &= long['all_io_submitted_by_upper_bound_ms'] <= cursor
                start = cursor
                cursor += long['minimum_ssd_service_ms']
                okay &= cursor+link_tail_ms <= long['compute_deadline_ms']
                reordered.append(dict(request_id=long['request_id'],npu_id=long['npu_id'],layer=long['layer'],
                    start_ms=start,ssd_end_ms=cursor,link_end_upper_bound_ms=cursor+link_tail_ms,
                    deadline_ms=long['compute_deadline_ms'],deadline_slack_lower_bound_ms=long['compute_deadline_ms']-cursor-link_tail_ms))
            if okay:
                candidates.append(dict(short=short,longs=list(longs),boundary=boundary,short_slack=slack,reordered=reordered))
    if not candidates:
        print(json.dumps(dict(found=False,case=str(case),reason='No qualifying contiguous, locally feasible witness; no evidence written.')))
        return
    # Prefer one whole preceding long and comfortable deadline slack; keep the
    # deterministic selection and eligibility criteria explicit in the evidence.
    chosen = min(candidates,key=lambda c:(len(c['longs']),-c['short_slack'],-c['short']['layer_stall_ms']))
    short,longs,start = chosen['short'],chosen['longs'],chosen['boundary']
    short_end = start+short['minimum_ssd_service_ms']
    all_end = chosen['reordered'][-1]['ssd_end_ms']
    long_volume = math.fsum(x['bytes_gib'] for x in longs)
    checks = dict(
        physical_service_no_overlap_and_correct_duration=receipts['observer_checks']['no_ssd_overlap_or_duration_error'],
        physical_capacity_check=receipts['observer_checks']['no_physical_ssd_capacity_breach'],
        measured_all_io_path0=receipts['observer_counters']['nonzero_path_io']==0,
        all_blocking_layers_are_internal=all(x['layer']>0 for x in longs),
        short_layer_is_internal=short['layer']>0,
        all_layers_have_contiguous_full_rate_ssd_service=all(x['span_equals_bytes_over_capacity'] for x in [*longs,short]),
        long_intervals_contiguously_end_at_short_service=all(math.isclose(x['last_ssd_end_ms'],y['first_ssd_start_ms'],abs_tol=1e-7)
                                                           for x,y in zip(longs,[*longs[1:],short])),
        whole_interval_volume_equals_capacity_times_duration=math.isclose((long_volume+short['bytes_gib'])/cap*1000,
                                                               short['last_ssd_end_ms']-start,abs_tol=1e-7),
        all_blocking_releases_precede_short_release=all(x['io_release_ms']<short['io_release_ms'] for x in longs),
        short_fully_issued_before_real_swap_boundary=short['all_io_submitted_by_upper_bound_ms']<=start,
        all_blocking_layers_fully_issued_before_real_swap_boundary=all(x['all_io_submitted_by_upper_bound_ms']<=start for x in longs),
        short_stall_matches_ready_minus_deadline=math.isclose(short['layer_stall_ms'],short['last_link_end_ms']-short['compute_deadline_ms'],abs_tol=1e-7),
        short_priority_local_schedule_meets_short_deadline=short_end+link_tail_ms<=short['compute_deadline_ms'],
        short_priority_local_schedule_meets_all_long_deadlines=all(x['link_end_upper_bound_ms']<=x['deadline_ms'] for x in chosen['reordered']),
        local_schedule_keeps_total_ssd_finish_unchanged=math.isclose(all_end,short['last_ssd_end_ms'],abs_tol=1e-7),
        selected_short_deadline_has_explicit_slack=chosen['short_slack']>=args.minimum_short_slack_ms)
    assert all(checks.values()),checks
    audit = read(case/'deadline_audit.json')
    evidence = dict(schema_version=1,case=case.name,window_ms=[left,right],
        source_sha256={name:hashlib.sha256((case/name).read_bytes()).hexdigest()
                       for name in ('manifest.json.gz','result.json.gz','receipts.json','metrics.json')},
        selection=dict(method='Retrospective contiguous whole-layer witness; prefer fewer preceding long layers, then larger short deadline slack.',
            required_short_deadline_slack_ms=args.minimum_short_slack_ms,qualifying_candidate_count=len(candidates),
            note='This restrictive witness count is not the frequency of all FIFO-induced stalls.'),
        short_layer=short,preceding_long_layers=longs,
        time_accounting=dict(short_release_ms=short['io_release_ms'],short_compute_deadline_ms=short['compute_deadline_ms'],
            short_io_ready_ms=short['last_link_end_ms'],short_layer_read_latency_ms=short['last_link_end_ms']-short['io_release_ms'],
            short_compute_ms=short['per_layer_compute_ms'],short_exposed_stall_ms=short['layer_stall_ms'],
            short_layer_cycle_utilization=short['per_layer_compute_ms']/(short['per_layer_compute_ms']+short['layer_stall_ms']),
            short_wait_until_first_ssd_service_ms=short['first_ssd_start_ms']-short['io_release_ms'],
            preceding_long_layers_total_gib=long_volume,preceding_long_layers_physical_service_ms=long_volume/cap*1000,
            contiguous_interval_ms=[start,short['last_ssd_end_ms']],contiguous_interval_total_gib=long_volume+short['bytes_gib']),
        local_feasibility_witness=dict(type='Offline arithmetic local counterfactual, not a simulator run',
            rule='At this actual IO boundary, read the already-submitted short layer first, then preserve the preceding long-layer service order.',
            start_ms=start,short_ssd_end_ms=short_end,short_link_end_upper_bound_ms=short_end+link_tail_ms,
            short_deadline_ms=short['compute_deadline_ms'],short_deadline_slack_lower_bound_ms=chosen['short_slack'],
            all_jobs_ssd_end_ms=all_end,all_long_link_end_upper_bound_ms=all_end+link_tail_ms,
            earliest_long_deadline_ms=min(x['compute_deadline_ms'] for x in longs),
            max_one_io_link_tail_ms=link_tail_ms,reordered_long_intervals=chosen['reordered'],
            client_issue_interval_us=result['summary']['client_issue_interval_us'],
            issuance_bound_note='One SSU, batch=1 and one next-layer producer per NPU; blocks*issue_interval is a conservative full issuance duration.'),
        window_stall_breakdown_by_group=audit['warm_by_group'],warm_metrics={k:metrics[k] for k in
            ('U_percent','short_U_percent','long_U_percent','slo_1p5_percent','input_average_gib_s')},
        validation_checks=checks,
        evidence_limits=[
            'First/last receipts alone do not prove per-IO order in general. Here each span equals bytes/40 and the global per-IO audit forbids overlap, certifying uninterrupted full-rate service.',
            'The local swap proves avoidable FIFO waiting for this measured pending task set at unchanged capacity; it is not the utilization result of a full priority strategy.',
            'Future newly released layers are not replayed; the paired full Once run is necessary for policy-wide performance.',
            'The global workload has input-average underload but transient current-demand overload. This local feasible witness does not establish global deadline feasibility.',
            'The illustrative layer is selected retrospectively and does not estimate how often one-long-only blocking occurs.'])
    (case/'fifo_hol_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(found=True,path=str(case/'fifo_hol_evidence.json'),short=short,longs=longs,
                         local=evidence['local_feasibility_witness'],checks=checks),ensure_ascii=False))


if __name__ == '__main__':
    main()
