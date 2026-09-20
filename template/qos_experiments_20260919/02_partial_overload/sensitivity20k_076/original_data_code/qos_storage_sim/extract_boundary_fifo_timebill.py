#!/usr/bin/env python3
"""Offline proof and time bill for the maximum warm short internal FIFO stall."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_boundary_underload import build, event_sweep, overlap, read, write_json

ROOT=Path(__file__).resolve().parent
DEFAULT_CASE=ROOT/'results/random_multitype_search_20260914/screen/strict_extended_seed7_fifo'
DEFAULT_OUTPUT=ROOT/'results/boundary_exempt_underload_20260914/local_fifo_timebill.json'


def extract(case,output):
    man,res,meta,rec,metrics=[read(case/f) for f in ('manifest.json.gz','result.json.gz','metadata.json','receipts.json','metrics.json')]
    assert rec['num_ssu']==1 and rec['probe_policy']=='fifo'
    assert all(rec['observer_checks'].values()) and rec['observer_counters']['nonzero_path_io']==0
    d=build(man,res,meta)
    loads={q['request_id']:q['load'] for q in man['requests']}
    timings={}
    for b in res['summary']['microbatch_metrics']:
        previous=None
        for layer in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            timings[b['member_request_ids'][0],layer['layer']]=dict(layer,previous=previous)
            previous=layer
    issue_ms=res['summary']['client_issue_interval_us']/1000
    cap=rec['ssu_capacity_gib_s'];linkcap=rec['npu_link_capacity_gib_s']
    tail=176*1024/2**30/linkcap*1000
    rows=[]
    for receipt in rec['layers']:
        key=(receipt['request_id'],receipt['layer']);t=timings[key];q=loads[key[0]]
        if t['previous'] is None:
            continue
        p=t['previous']
        row=dict(receipt,role=q['role'],profile_group=q['profile_group'],total_k=q['total_tokens']/1024,
            nql=q['nql'],compute_ms=q['per_layer_us']/1000,io_release_ms=t['io_start_time_ms'],
            io_ready_ms=t['io_ready_time_ms'],current_compute_layer=p['layer'],current_compute_start_ms=p['compute_start_ms'],
            current_compute_end_ms=p['compute_end_ms'],next_compute_start_ms=t['compute_start_ms'],
            compute_deadline_ms=p['compute_end_ms'],stall_ms=max(0.,t['compute_start_ms']-p['compute_end_ms']),
            minimum_ssd_service_ms=receipt['bytes_gib']/cap*1000,
            all_io_submitted_by_upper_bound_ms=t['io_start_time_ms']+receipt['completed_blocks']*issue_ms)
        row['ssd_span_ms']=row['last_ssd_end_ms']-row['first_ssd_start_ms']
        row['contiguous_full_rate_ssd_service_proven']=math.isclose(row['ssd_span_ms'],row['minimum_ssd_service_ms'],abs_tol=1e-7)
        rows.append(row)
    left,right=metrics['window_ms']
    shorts=[x for x in rows if x['role']=='S' and overlap(x['compute_deadline_ms'],x['next_compute_start_ms'],left,right)>0]
    target=max(shorts,key=lambda x:x['stall_ms'])
    assert left<target['io_release_ms']<target['last_link_end_ms']<right
    assert target['contiguous_full_rate_ssd_service_proven']
    # Reconstruct only spans whose measured bytes occupy the entire first/last
    # interval. No inference of contiguity is made for interleaved long layers.
    chain=[target]
    cursor=target['first_ssd_start_ms']
    while True:
        previous=[x for x in rows if x['contiguous_full_rate_ssd_service_proven'] and
            math.isclose(x['last_ssd_end_ms'],cursor,abs_tol=1e-7)]
        assert len(previous)<=1
        if not previous or previous[0]['first_ssd_start_ms']<target['io_release_ms']:
            break
        chain.insert(0,previous[0]);cursor=previous[0]['first_ssd_start_ms']
    longs=[x for x in chain if x['role']=='L']
    assert len(longs)==2
    assert [x['role'] for x in chain]==['L','L','S','S','S']
    assert all(x['io_release_ms']<target['io_release_ms'] for x in chain[:-1])
    start=chain[0]['first_ssd_start_ms'];end=chain[-1]['last_ssd_end_ms']
    assert all(x['all_io_submitted_by_upper_bound_ms']<=start for x in chain)
    assert all(math.isclose(x['last_ssd_end_ms'],y['first_ssd_start_ms'],abs_tol=1e-7) for x,y in zip(chain,chain[1:]))
    assert math.isclose(sum(x['bytes_gib'] for x in chain)/cap*1000,end-start,abs_tol=1e-7)
    # Same pending five layer reads; put the three short reads before the two
    # long reads, keeping order within each group. This is not a native rerun.
    reordered=[];cursor=start
    for job in [x for x in chain if x['role']=='S']+longs:
        service_start=cursor;cursor+=job['minimum_ssd_service_ms']
        reordered.append(dict(request_id=job['request_id'],npu_id=job['npu_id'],layer=job['layer'],role=job['role'],
            start_ms=service_start,ssd_end_ms=cursor,link_end_upper_bound_ms=cursor+tail,
            deadline_ms=job['compute_deadline_ms'],deadline_slack_lower_bound_ms=job['compute_deadline_ms']-cursor-tail,
            full_io_issued_by_ms=job['all_io_submitted_by_upper_bound_ms']))
    assert all(x['deadline_slack_lower_bound_ms']>0 for x in reordered)
    assert math.isclose(cursor,end,abs_tol=1e-7)
    target_cf=next(x for x in reordered if x['request_id']==target['request_id'] and x['layer']==target['layer'])
    local={k:event_sweep(d['intervals'][k],1,target['io_release_ms'],target['io_ready_ms'],cap) for k in ('current','continuing','boundary','total_prefetch')}
    active_boundaries=[j for j in d['jobs'] if j['kind']=='cross_role_boundary' and
        overlap(j['release_ms'],max(j['deadline_ms'],j['io_ready_ms']),target['io_release_ms'],target['io_ready_ms'])>0]
    bill=dict(io_release_ms=target['io_release_ms'],compute_window_end_ms=target['current_compute_end_ms'],
        first_ssd_start_ms=target['first_ssd_start_ms'],last_ssd_end_ms=target['last_ssd_end_ms'],io_ready_ms=target['io_ready_ms'],
        waiting_before_first_ssd_ms=target['first_ssd_start_ms']-target['io_release_ms'],
        continuous_own_ssd_service_ms=target['minimum_ssd_service_ms'],
        final_link_tail_ms=target['io_ready_ms']-target['last_ssd_end_ms'],
        total_read_latency_ms=target['io_ready_ms']-target['io_release_ms'],
        hidden_by_current_compute_ms=target['compute_ms'],exposed_stall_ms=target['stall_ms'],
        two_contiguous_long_layers_service_ms=sum(x['minimum_ssd_service_ms'] for x in longs),
        intervening_two_short_layers_service_ms=sum(x['minimum_ssd_service_ms'] for x in chain if x['role']=='S' and x is not target),
        residual_previous_service_after_release_ms=start-target['io_release_ms'],
        single_layer_cycle_utilization_percent=100*target['compute_ms']/(target['compute_ms']+target['stall_ms']))
    assert math.isclose(bill['waiting_before_first_ssd_ms']+bill['continuous_own_ssd_service_ms']+bill['final_link_tail_ms'],bill['total_read_latency_ms'],abs_tol=1e-7)
    assert math.isclose(bill['total_read_latency_ms']-bill['hidden_by_current_compute_ms'],bill['exposed_stall_ms'],abs_tol=1e-7)
    assert math.isclose(bill['residual_previous_service_after_release_ms']+bill['two_contiguous_long_layers_service_ms']+bill['intervening_two_short_layers_service_ms'],bill['waiting_before_first_ssd_ms'],abs_tol=1e-7)
    payload=dict(schema_version=1,case=str(case),selection='Maximum actual exposed stall among short-role internal layers intersecting native warm window; selected read and compute window are wholly inside warm.',
        window_ms=[left,right],short_internal_stalled_layer_count=len(shorts),target_layer=target,
        time_accounting=bill,verified_contiguous_ssd_sequence=chain,preceding_long_layers=longs,
        local_demand_references=local,cross_role_boundaries_active_in_target_read_window=active_boundaries,
        local_feasibility_witness=dict(type='Offline local arithmetic, no native simulation',
            pending_jobs=len(chain),all_jobs_fully_issued_before_start=True,start_ms=start,
            original_ssd_finish_ms=end,reordered_ssd_finish_ms=cursor,
            max_one_io_link_tail_ms=tail,reordered_jobs=reordered,
            target_io_ready_upper_bound_ms=target_cf['link_end_upper_bound_ms'],
            target_deadline_slack_lower_bound_ms=target_cf['deadline_slack_lower_bound_ms'],
            all_five_deadlines_met=True,total_ssd_finish_unchanged=True),
        validation_checks=dict(path0_only=True,physical_no_ssd_overlap=True,physical_no_capacity_breach=True,
            each_selected_span_equals_its_bytes_divided_by_40=True,selected_layers_are_contiguous=True,
            target_maximum_warm_short_internal_stall=True,all_selected_jobs_fully_submitted_before_reorder_boundary=True,
            short_read_time_bill_conserves=True,local_reorder_meets_all_five_deadlines=True),
        source_sha256={f:hashlib.sha256((case/f).read_bytes()).hexdigest() for f in ('manifest.json.gz','result.json.gz','receipts.json','metadata.json','metrics.json')},
        limitations=[
            'Receipts contain first/last timestamps per layer, not a full per-IO trace. Selected spans equal exactly bytes/40 and the per-IO observer forbids overlap, certifying no idle or other service inside those spans. Other interleaved long spans are not treated as continuous.',
            'Measured large layers released and were fully submitted before the target, and were serviced first on sole FIFO path0. The arithmetic witness establishes avoidable waiting for these already pending reads at unchanged capacity.',
            'The local reorder does not replay future layer arrivals, so it cannot predict whole-run utilization or every other request deadline. Use the paired native Once run for policy-wide performance.',
            'Demand reference is an amortized byte/compute-window rate; being under capacity does not imply FIFO meets every deadline. In this selected window there is no active cross-role boundary reference or its overdue IO, yet FIFO still creates an internal short-layer deadline miss.',
            'The target is selected retrospectively as maximum warm stall; this one interval does not represent average utilization or the frequency of all head-of-line blocking events.' ])
    write_json(output,payload)
    print(json.dumps(dict(output=str(output),target={k:target[k] for k in ('request_id','npu_id','layer','nql','stall_ms')},bill=bill,
        local_current_peak=local['current']['fleet_peak_gib_s'],local_total_prefetch_peak=local['total_prefetch']['fleet_peak_gib_s'],
        local_boundary_count=len(active_boundaries),reordered=reordered),ensure_ascii=False))
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=DEFAULT_CASE)
    p.add_argument('--out',type=Path,default=DEFAULT_OUTPUT);a=p.parse_args();extract(a.case,a.out)


if __name__=='__main__':main()
