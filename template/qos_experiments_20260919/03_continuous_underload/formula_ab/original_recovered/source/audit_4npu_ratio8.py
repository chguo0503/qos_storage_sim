#!/usr/bin/env python3
"""Read-only four-NPU ratio-eight audit, including labelled 20K extrapolation.

Derived from audit_4npu_exact_data; bandwidth, placement and stall gates retained.
Only a short S -> long L cross-request Layer-0 contribution is exempted.
Long L -> short S Layer-0 remains in continuing demand and its capacity gates.

S and L are workload role labels, not the native SS/SL/LS/LL QoS categories.
Outputs are written under --out, never into the native source case directory.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from audit_random_multitype import read
from audit_strict_random import event_sweep, interval_counts
from run_fixed128_32 import profile as reconstruct_profile

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'results/ratio8_4npu_20260914'
CAP, LINK, EPS = 40., 50., 1e-8
KINDS = ('same_request', 'same_role_cross_request', 'long_to_short_cross_request',
         'cross_role_boundary')


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')


def overlap(a, b, left, right):
    return max(0., min(b, right)-max(a, left))


def union_spans(spans):
    out = []
    for a,b in sorted((a,b) for a,b in spans if b>a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a,b))
    return out


def complement(spans, left, right):
    cursor, out = left, []
    for a,b in union_spans(spans):
        a,b = max(a,left), min(b,right)
        if b<=a:
            continue
        if a>cursor:
            out.append((cursor,a))
        cursor = max(cursor,b)
    if cursor<right:
        out.append((cursor,right))
    return out


def intersection_size(a,b,spans,left,right):
    return math.fsum(overlap(a,b,max(x,left),min(y,right)) for x,y in spans)


def masked_sweep(intervals, masks, count, left, right, capacity, near_fraction):
    """Peak valid only on mask intervals; fractions use mask duration explicitly."""
    clipped = [(max(a,x),min(b,y),v) for a,b,v in intervals for x,y in masks
               if min(b,y)>max(a,x)]
    row = event_sweep(clipped,count,left,right,capacity,near_fraction)
    selected = math.fsum(b-a for a,b in masks)
    row['selected_duration_ms'] = selected
    row['selected_fraction_of_window_percent'] = selected/(right-left)*100
    row['selected_any_unit_near_capacity_percent'] = (100*row['any_unit_near_capacity_ms']/selected if selected else None)
    row['selected_any_unit_over_capacity_percent'] = (100*row['any_unit_over_capacity_ms']/selected if selected else None)
    row['note'] = 'Unselected times contribute zeros to generic mean/quantile fields; selected percentages have their own explicit denominator.'
    return row


def assert_profile_equal(actual, expected, label):
    """Compare every construction field, rejecting omitted or invented metadata."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), label
        for key, value in expected.items():
            assert_profile_equal(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), label
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert_profile_equal(a, e, f'{label}[{index}]')
    elif isinstance(expected, bool):
        assert actual is expected, label
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), label
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label
    else:
        assert actual == expected, label


def validate_profile(table, load):
    """Rebuild direct/interpolated/extrapolated profiles from their exact data anchors."""
    rid = load['request_id']
    for field in ('total_tokens','ssd_prefix_tokens','nql'):
        value = load[field]
        assert isinstance(value, (int,float)) and not isinstance(value, bool), (rid,field)
        assert math.isfinite(value) and value > 0 and value == int(value), (rid,field)
    assert 20*1024 <= load['total_tokens'] <= 200*1024 and 64 <= load['nql'] <= 4096, rid
    assert load['total_tokens'] == load['ssd_prefix_tokens'] + load['nql'], rid
    rebuilt = reconstruct_profile(table, load['ssd_prefix_tokens'], load['nql'])
    mapping = dict(total_tokens='total_tokens', seq_len_k='total_length_k', nql='nql',
        ssd_prefix_tokens='ssd_prefix_tokens', per_layer_us='compute_us',
        per_layer_kv_gb='read_gib', required_bw_input_gbps='B_gib_s',
        original_compute_us='compute_us', constructed_profile='constructed_profile',
        profile_construction='profile_construction')
    for field, expected_field in mapping.items():
        assert_profile_equal(load[field], rebuilt[expected_field], f'request {rid}.{field}')
    assert_profile_equal(load['padding_gib_per_layer'], 0., f'request {rid}.padding_gib_per_layer')
    anchors = load['profile_construction']['anchors']
    for anchor in anchors:
        raw = table[(anchor['seq_len_k'], anchor['nql'])]
        assert_profile_equal(anchor['compute_us'], raw[1], f'request {rid}.data_anchor_compute')
    assert_profile_equal(math.fsum(a['weight'] for a in anchors), 1., f'request {rid}.anchor_weight_sum')
    assert_profile_equal(math.fsum(a['weight']*a['compute_us'] for a in anchors),
                         load['per_layer_us'], f'request {rid}.anchor_compute_sum')
    if rebuilt['profile_construction']['extrapolated']:
        assert any(a['weight'] < 0 for a in anchors), rid
    return rebuilt


def build(man, result, meta):
    data_path = ROOT/'data'
    table = ast.literal_eval(data_path.read_text())
    assert meta['data_sha256'] == hashlib.sha256(data_path.read_bytes()).hexdigest(), 'data_sha256'
    assert_profile_equal(man['metadata'], meta, 'manifest.metadata')
    qs = {q['request_id']:q for q in man['requests']}
    assert len(qs) == len(man['requests']), 'duplicate request IDs'
    S,N = meta['num_ssu'],meta['num_npu']
    assert N == 4 and S >= 1
    volumes,ownrates,groups = {},{},defaultdict(list)
    profile_checks = dict(same_blocks_on_same_ssu_all_layers=True,exact_prefix_volume=True,
                          profiles_in_measured_grid=True,unique_profiles_per_npu=meta.get("all_length_nql_unique_within_each_npu",True),
                          raw_profile_repetition_authorized=bool(meta.get("exact_profile_repetition_intentional",False)),
                          profile_repetition_authorized=bool(meta.get("profile_repetition_intentional",
                              meta.get("exact_profile_repetition_intentional",False))),
                          compute_reconstruction_verified=True,all_construction_metadata_verified=True,
                          source_anchors_match_data=True,data_source_sha256_verified=True,
                          contains_constructed_profiles=False,contains_extrapolated_compute=False,
                          both_roles_on_all_npus=True)
    rebuilt_by_role = defaultdict(dict)
    for rid,q in qs.items():
        load = q['load']
        rebuilt = validate_profile(table, load)
        assert load['request_id'] == rid
        assert load['npu_id'] == q['npu_id'] and 0 <= q['npu_id'] < N
        assert load['role'] in ('L','S')
        rebuilt_by_role[load['role']][(load['total_tokens'],load['nql'])] = rebuilt
        profile_checks['contains_constructed_profiles'] |= rebuilt['constructed_profile']
        profile_checks['contains_extrapolated_compute'] |= rebuilt['profile_construction']['extrapolated']
        profile_checks['profiles_in_measured_grid'] &= not rebuilt['profile_construction']['extrapolated']
        layers = man['placements'][q['placement_index']]
        assert len(layers) in (1,meta['n_layers'])
        assert all(layer==layers[0] for layer in layers)
        prefix = load['ssd_prefix_tokens']
        assert len(layers[0]) == math.ceil(prefix/128), rid
        for block, (disk, volume) in enumerate(layers[0]):
            assert disk == int(disk) and 0 <= disk < S, (rid,block)
            assert_profile_equal(volume, min(128,prefix-128*block)*1408/2**30,
                                 f'request {rid}.block {block}.volume')
        volumes[rid] = [math.fsum(float(v) for disk,v in layers[0] if int(disk)==s) for s in range(S)]
        assert math.isclose(math.fsum(volumes[rid]),load['per_layer_kv_gb'],abs_tol=1e-12)
        assert math.isclose(math.fsum(volumes[rid]),load['ssd_prefix_tokens']*1408/2**30,abs_tol=1e-12)
        ownrates[rid] = [v/(load['per_layer_us']/1e6) for v in volumes[rid]]
        groups[load['profile_group']].append(rid)
    expected_flags = dict(
        all_compute_profiles_within_measured_grid=profile_checks['profiles_in_measured_grid'],
        compute_calibrated=not profile_checks['contains_extrapolated_compute'],
        contains_extrapolated_compute=profile_checks['contains_extrapolated_compute'],
        contains_constructed_profiles=profile_checks['contains_constructed_profiles'],
        profiles_exactly_from_data=not profile_checks['contains_constructed_profiles'])
    for key, expected in expected_flags.items():
        if key in meta:
            assert_profile_equal(meta[key], expected, f'metadata.{key}')
    profile_checks['raw_profile_repetition_authorized'] &= not profile_checks['contains_constructed_profiles']
    assert max(p['total_tokens'] for p in rebuilt_by_role['S'].values()) < min(
        p['total_tokens'] for p in rebuilt_by_role['L'].values()), 'S/L length roles'
    for role, field in (('L','long'),('S','short')):
        role_profiles = rebuilt_by_role[role]
        assert len(role_profiles) == 1, f'{role}: expected one repeated profile'
        rebuilt = next(iter(role_profiles.values()))
        assert_profile_equal(meta['case'][field], [rebuilt['total_length_k'],rebuilt['nql']],
                             f'metadata.case.{field}')
        if 'profiles' in meta:
            assert_profile_equal(meta['profiles'][role], rebuilt, f'metadata.profiles.{role}')
    role_profiles = {role:next(iter(values.values())) for role,values in rebuilt_by_role.items()}
    if 'raw_profile_anchor_rows' in meta:
        expected_rows = {role:[dict(total_k=a['seq_len_k'],nql=a['nql'],
            raw_data_row=list(table[(a['seq_len_k'],a['nql'])]))
            for a in p['profile_construction']['anchors']] for role,p in role_profiles.items()}
        assert_profile_equal(meta['raw_profile_anchor_rows'], expected_rows, 'metadata.raw_profile_anchor_rows')
        profile_checks['raw_profile_anchor_rows_verified'] = True
    if 'ratio_comparison' in meta:
        pa,pb = role_profiles['L'],role_profiles['S']
        vr,cr = pa['read_gib']/pb['read_gib'],pa['compute_us']/pb['compute_us']
        expected_ratio = dict(VA_over_VB=vr,CA_over_CB=cr,
            VA_gt_8VB=pa['read_gib']>8*pb['read_gib'],CA_gt_8CB=pa['compute_us']>8*pb['compute_us'],
            both_gt_8=vr>8 and cr>8,
            ideal_A_read_over_CB=pa['read_gib']/(40*S)/(pb['compute_us']/1e6),
            pure_four_npu_peak_gib_s=N*max(p['B_gib_s'] for p in role_profiles.values()))
        assert_profile_equal(meta['ratio_comparison'], expected_ratio, 'metadata.ratio_comparison')
        profile_checks['ratio_comparison_verified'] = True
    lane_checks=[]
    for n in range(N):
        deck = [q for q in qs.values() if q['npu_id']==n]
        combos=[(q['load']['total_tokens'],q['load']['nql']) for q in deck]
        if meta.get("all_length_nql_unique_within_each_npu",True):
            assert len(combos)==len(set(combos))
        else:
            assert profile_checks['profile_repetition_authorized']
            assert len(set(combos))==2
        assert {q['load']['role'] for q in deck}=={'L','S'}
        lane_checks.append(dict(npu_id=n,request_count=len(deck),unique_profiles=len(set(combos)),
            group_counts=dict(Counter(q['load']['profile_group'] for q in deck))))
    intervals={key:[] for key in ('current','own_link','total_prefetch','continuing','boundary',
                                 'total_link','continuing_link','boundary_link')}
    jobs,cold,batch_records=[],[],[]
    for n in range(N):
        batches=sorted((b for b in result['summary']['microbatch_metrics'] if b['npu_id']==n),
                       key=lambda b:b['admission_time_ms'])
        flat=[]
        for batch in batches:
            assert len(batch['member_request_ids'])==1
            rid=batch['member_request_ids'][0]
            layers=sorted(batch['layer_metrics'],key=lambda l:l['layer'])
            assert len(layers)==meta['n_layers']
            flat.extend(dict(request_id=rid,**layer) for layer in layers)
            intervals['current'].append((batch['admission_time_ms'],batch['completion_time_ms'],ownrates[rid]))
            own_link=[0.]*N;own_link[n]=math.fsum(ownrates[rid])
            intervals['own_link'].append((batch['admission_time_ms'],batch['completion_time_ms'],own_link))
            batch_records.append(dict(npu_id=n,request_id=rid,group=qs[rid]['load']['profile_group'],
                role=qs[rid]['load']['role'],admission_ms=batch['admission_time_ms'],
                completion_ms=batch['completion_time_ms'],layers=layers))
        first=flat[0]
        cold.append(dict(npu_id=n,request_id=first['request_id'],release_ms=first['io_start_time_ms'],
            admission_ms=batches[0]['admission_time_ms'],ready_ms=first['io_ready_time_ms'],
            compute_start_ms=first['compute_start_ms']))
        for previous,following in zip(flat,flat[1:]):
            old,rid=previous['request_id'],following['request_id']
            oldrole,role=qs[old]['load']['role'],qs[rid]['load']['role']
            if old == rid:
                kind = 'same_request'
            else:
                assert following['layer'] == 0 and previous['layer'] == meta['n_layers']-1
                kind = ('same_role_cross_request' if oldrole == role else
                        'cross_role_boundary' if (oldrole,role) == ('S','L') else
                        'long_to_short_cross_request')
            key='boundary' if kind=='cross_role_boundary' else 'continuing'
            start,deadline=previous['compute_start_ms'],previous['compute_end_ms']
            assert math.isclose(start,following['io_start_time_ms'],abs_tol=1e-7)
            c=deadline-start
            rates=[v/c*1000 for v in volumes[rid]]
            link=[0.]*N;link[n]=math.fsum(rates)
            for k in (key,'total_prefetch'):
                intervals[k].append((start,deadline,rates))
            for k in (key+'_link','total_link'):
                intervals[k].append((start,deadline,link))
            lower=max(max(volumes[rid])/CAP,math.fsum(volumes[rid])/LINK)*1000
            stall=max(0.,following['compute_start_ms']-deadline)
            assert stall+1e-6>=max(0.,lower-c)
            jobs.append(dict(npu_id=n,kind=kind,from_request_id=old,to_request_id=rid,
                from_role=oldrole,to_role=role,from_group=qs[old]['load']['profile_group'],
                to_group=qs[rid]['load']['profile_group'],to_layer=following['layer'],
                release_ms=start,deadline_ms=deadline,io_ready_ms=following['io_ready_time_ms'],
                next_compute_start_ms=following['compute_start_ms'],available_compute_ms=c,
                next_layer_gib=math.fsum(volumes[rid]),required_per_ssu_gib_s=rates,
                required_npu_link_gib_s=math.fsum(rates),actual_stall_ms=stall,
                io_deadline_lateness_ms=max(0.,following['io_ready_time_ms']-deadline),
                exclusive_storage_lower_bound_ms=max(volumes[rid])/CAP*1000,
                exclusive_link_lower_bound_ms=math.fsum(volumes[rid])/LINK*1000,
                exclusive_transfer_lower_bound_ms=lower,
                unavoidable_transfer_stall_lower_bound_ms=max(0.,lower-c),
                stall_above_exclusive_lower_bound_ms=max(0.,stall-max(0.,lower-c))))
    return dict(qs=qs,volumes=volumes,ownrates=ownrates,groups=groups,profile_checks=profile_checks,
                lane_checks=lane_checks,intervals=intervals,jobs=jobs,cold=cold,batches=batch_records)


def job_stats(jobs,left,right):
    affected=[j for j in jobs if j['release_ms']<right and max(j['deadline_ms'],j['next_compute_start_ms'])>left]
    stalled=[j for j in affected if overlap(j['deadline_ms'],j['next_compute_start_ms'],left,right)>EPS]
    impossible=[j for j in affected if j['unavoidable_transfer_stall_lower_bound_ms']>EPS]
    return dict(jobs_with_reference_window_overlapping=sum(overlap(j['release_ms'],j['deadline_ms'],left,right)>0 for j in jobs),
        jobs_released=sum(left<=j['release_ms']<right for j in jobs),
        release_frequency_per_second=sum(left<=j['release_ms']<right for j in jobs)/(right-left)*1000,
        stalled_job_count=len(stalled),
        actual_stall_card_ms=math.fsum(overlap(j['deadline_ms'],j['next_compute_start_ms'],left,right) for j in stalled),
        individually_infeasible_job_count=len(impossible),
        unavoidable_transfer_stall_lower_bound_card_ms=math.fsum(overlap(j['deadline_ms'],j['deadline_ms']+j['unavoidable_transfer_stall_lower_bound_ms'],left,right) for j in impossible),
        max_actual_stall_ms=max((j['actual_stall_ms'] for j in affected),default=0.),
        max_npu_reference_gib_s=max((j['required_npu_link_gib_s'] for j in affected),default=0.),
        reference_window_union=interval_counts([(j['release_ms'],j['deadline_ms']) for j in jobs],left,right),
        io_lifetime_union=interval_counts([(j['release_ms'],j['io_ready_ms']) for j in jobs],left,right),
        reference_plus_overdue_io_union=interval_counts([(j['release_ms'],max(j['deadline_ms'],j['io_ready_ms'])) for j in jobs],left,right),
        overdue_io_union=interval_counts([(j['deadline_ms'],j['io_ready_ms']) for j in jobs if j['io_ready_ms']>j['deadline_ms']],left,right))


def utilization(batches,cold,jobs,N,left,right):
    by_group=defaultdict(lambda:dict(active_card_ms=0.,compute_card_ms=0.,internal_stall_card_ms=0.,l0_stall_card_ms=0.,npu_ids=set()))
    active=[0.]*N
    for b in batches:
        group=by_group[b['group']]
        a=overlap(b['admission_ms'],b['completion_ms'],left,right)
        c=math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],left,right) for l in b['layers'])
        group['active_card_ms']+=a;group['compute_card_ms']+=c;active[b['npu_id']]+=a
        if c>0:group['npu_ids'].add(b['npu_id'])
        cursor=b['admission_ms']
        for l in b['layers']:
            group['l0_stall_card_ms' if l['layer']==0 else 'internal_stall_card_ms']+=overlap(cursor,l['compute_start_ms'],left,right)
            cursor=l['compute_end_ms']
    for row in by_group.values():
        row['npu_ids']=sorted(row['npu_ids'])
        row['U_percent']=100*row['compute_card_ms']/row['active_card_ms'] if row['active_card_ms'] else None
        assert math.isclose(row['compute_card_ms']+row['internal_stall_card_ms']+row['l0_stall_card_ms'],row['active_card_ms'],abs_tol=1e-6)
    comp=math.fsum(r['compute_card_ms'] for r in by_group.values())
    return dict(U_percent=100*comp/(N*(right-left)),active_U_percent=100*comp/math.fsum(active),
        compute_card_ms=comp,active_card_ms=math.fsum(active),
        all_npus_active_entire_window=all(math.isclose(a,right-left,abs_tol=1e-7) for a in active),
        all_npus_all_groups_computed=all(r['npu_ids']==list(range(N)) for r in by_group.values()),
        internal_stall_card_ms=math.fsum(r['internal_stall_card_ms'] for r in by_group.values()),
        l0_stall_card_ms=math.fsum(r['l0_stall_card_ms'] for r in by_group.values()),
        cold_initial_stall_card_ms=math.fsum(overlap(j['admission_ms'],j['compute_start_ms'],left,right) for j in cold),
        by_group=dict(by_group))


def audit(case,out=DEFAULT,near_fraction=.9,max_near_percent=5.):
    case=case.resolve()
    man,result,meta,metrics=(read(case/f) for f in ('manifest.json.gz','result.json.gz','metadata.json','metrics.json'))
    d=build(man,result,meta)
    S,N=meta['num_ssu'],meta['num_npu']
    jobs=d['jobs']; intervals=d['intervals']; makespan=result['summary']['makespan_ms']
    kinds={k:[j for j in jobs if j['kind']==k] for k in KINDS}
    boundary=kinds['cross_role_boundary']
    continuing=[j for j in jobs if j['kind']!='cross_role_boundary']
    assert all(j['from_role']=='S' and j['to_role']=='L' and j['to_layer']==0 for j in boundary)
    boundary_reference_spans=union_spans([(j['release_ms'],j['deadline_ms']) for j in boundary])
    boundary_io_spans=union_spans([(j['release_ms'],max(j['deadline_ms'],j['io_ready_ms'])) for j in boundary])
    boundary_debt_spans=union_spans([(j['deadline_ms'],j['io_ready_ms']) for j in boundary if j['io_ready_ms']>j['deadline_ms']])
    windows={}
    for label,left,wanted_right in [('full_run',0.,makespan),('first_4500ms',0.,4500.),('broad_500_4000ms',500.,4000.),('warm_2000_4000ms',2000.,4000.)]:
        right=min(wanted_right,makespan)
        if right<=left:
            windows[label]=dict(unavailable=True);continue
        row=dict(window_ms=[left,right],requested_window_ms=[left,wanted_right])
        for key in ('current','total_prefetch','continuing','boundary'):
            row[key]=event_sweep(intervals[key],S,left,right,CAP,near_fraction)
        for key in ('own_link','total_link','continuing_link','boundary_link'):
            row[key]=event_sweep(intervals[key],N,left,right,LINK,near_fraction)
        row['total_when_no_boundary_reference']=masked_sweep(intervals['total_prefetch'],complement(boundary_reference_spans,left,right),S,left,right,CAP,near_fraction)
        row['total_when_no_boundary_reference_or_overdue_io']=masked_sweep(intervals['total_prefetch'],complement(boundary_io_spans,left,right),S,left,right,CAP,near_fraction)
        row['job_stats']={k:job_stats(v,left,right) for k,v in kinds.items()}
        row['job_stats']['continuing']=job_stats(continuing,left,right)
        row['utilization']=utilization(d['batches'],d['cold'],jobs,N,left,right)
        # Boundary contributions are removed by job identity, not by dropping
        # whole time intervals. Integral conservation checks this partition.
        assert math.isclose(row['total_prefetch']['fleet_mean_gib_s'],
            row['continuing']['fleet_mean_gib_s']+row['boundary']['fleet_mean_gib_s'],abs_tol=1e-8)
        total_job_stall=math.fsum(row['job_stats'][k]['actual_stall_card_ms'] for k in KINDS)
        assert math.isclose(total_job_stall+row['utilization']['cold_initial_stall_card_ms'],
            row['utilization']['internal_stall_card_ms']+row['utilization']['l0_stall_card_ms'],abs_tol=1e-6)
        internal=kinds['same_request']
        row['internal_stall_temporal_overlap_with_boundary_reference_card_ms']=math.fsum(intersection_size(j['deadline_ms'],j['next_compute_start_ms'],boundary_reference_spans,left,right) for j in internal)
        row['internal_stall_temporal_overlap_with_boundary_reference_or_overdue_io_card_ms']=math.fsum(intersection_size(j['deadline_ms'],j['next_compute_start_ms'],boundary_io_spans,left,right) for j in internal)
        row['internal_stall_temporal_overlap_with_boundary_overdue_io_card_ms']=math.fsum(intersection_size(j['deadline_ms'],j['next_compute_start_ms'],boundary_debt_spans,left,right) for j in internal)
        exposed_internal=[j for j in internal if overlap(j['deadline_ms'],j['next_compute_start_ms'],left,right)>EPS]
        overlapping_boundary_reads=[j for j in exposed_internal if intersection_size(j['release_ms'],j['io_ready_ms'],boundary_io_spans,0.,makespan)>EPS]
        row['internal_stalled_jobs_whose_read_lifetime_overlaps_boundary_reference_or_overdue_io_count']=len(overlapping_boundary_reads)
        row['internal_stall_from_jobs_whose_read_lifetime_overlaps_boundary_reference_or_overdue_io_card_ms']=math.fsum(overlap(j['deadline_ms'],j['next_compute_start_ms'],left,right) for j in overlapping_boundary_reads)
        row['storage_non_boundary_references_under_capacity']=all(row[k]['every_unit_within_capacity'] for k in ('current','continuing'))
        row['storage_non_boundary_near_time_within_limit']=all(row[k]['fleet_near_capacity_percent']<=max_near_percent+EPS for k in ('current','continuing'))
        row['own_and_continuing_npu_links_under_capacity']=all(row[k]['every_unit_within_capacity'] for k in ('own_link','continuing_link'))
        windows[label]=row
    invariants=result['summary']['invariants'];assert invariants and all(invariants.values())
    physical={}
    if (case/'receipts.json').exists():
        rec=read(case/'receipts.json')
        widths=[b-a for a,b in zip(rec['bin_edges_ms'],rec['bin_edges_ms'][1:])]
        physical=dict(peak_per_ssu_gib_s=[max(v/(dt/1000) for v,dt in zip(row,widths)) for row in rec['ssu_bin_read_gib']],observer_checks=rec['observer_checks'])
        assert max(physical['peak_per_ssu_gib_s'])<=CAP+1e-6
        assert all(physical['observer_checks'].values())
    full=windows['full_run'];warm=windows['warm_2000_4000ms']
    if metrics['window_ms']==[2000.,4000.] and not warm.get('unavailable'):
        assert math.isclose(metrics['U_percent'],warm['utilization']['U_percent'],abs_tol=1e-7)
    checks=dict(non_boundary_storage_capacity_full_run=full['storage_non_boundary_references_under_capacity'],
        own_and_continuing_link_capacity_full_run=full['own_and_continuing_npu_links_under_capacity'],
        no_non_boundary_individually_infeasible_jobs=not any(j['unavoidable_transfer_stall_lower_bound_ms']>EPS for j in continuing),
        near_time_limit_all_windows=all(not w.get('unavailable') and w['storage_non_boundary_near_time_within_limit'] for w in windows.values()),
        warm_all_npus_active=not warm.get('unavailable') and warm['utilization']['all_npus_active_entire_window'],
        warm_all_npus_all_groups_computed=not warm.get('unavailable') and warm['utilization']['all_npus_all_groups_computed'],
        all_native_invariants=all(invariants.values()),physical_receipt_checks=bool(physical) and all(physical['observer_checks'].values()))
    basic_checks={k:v for k,v in checks.items() if k!='near_time_limit_all_windows'}
    group_profiles={gid:dict(role=d['qs'][rids[0]]['load']['role'],request_count=len(rids),
        nql_range=[min(d['qs'][r]['load']['nql'] for r in rids),max(d['qs'][r]['load']['nql'] for r in rids)],
        total_k_range=[min(d['qs'][r]['load']['total_tokens']/1024 for r in rids),max(d['qs'][r]['load']['total_tokens']/1024 for r in rids)],
        compute_ms_range=[min(d['qs'][r]['load']['per_layer_us']/1000 for r in rids),max(d['qs'][r]['load']['per_layer_us']/1000 for r in rids)],
        max_own_npu_gib_s=max(math.fsum(d['ownrates'][r]) for r in rids),
        constructed_profile_count=sum(d['qs'][r]['load']['constructed_profile'] for r in rids),
        extrapolated_compute_count=sum(d['qs'][r]['load']['profile_construction']['extrapolated'] for r in rids))
        for gid,rids in d['groups'].items()}
    result_audit=dict(schema_version=1,case=str(case),name=metrics['name'],policy=metrics['policy'],seed=meta['seed'],
        num_npu=N,num_ssu=S,makespan_ms=makespan,U_percent=metrics['U_percent'],slo_1p5_percent=metrics['slo_1p5_percent'],
        boundary_exempt_candidate_pass=all(checks.values()),capacity_only_candidate_pass=all(basic_checks.values()),
        pass_checks=checks,threshold=dict(ssu_gib_s=CAP,npu_link_gib_s=LINK,near_fraction=near_fraction,max_near_percent=max_near_percent),
        windows=windows,input_groups=group_profiles,input_lane_checks=d['lane_checks'],profile_checks=d['profile_checks'],physical_ssd=physical,
        source_sha256={f:hashlib.sha256((case/f).read_bytes()).hexdigest() for f in ('manifest.json.gz','result.json.gz','metadata.json','metrics.json')},
        native_invariants=invariants,all_job_count=len(jobs),job_counts_by_kind=dict(Counter(j['kind'] for j in jobs)),
        boundary_jobs=boundary,missed_jobs=[j for j in jobs if j['io_deadline_lateness_ms']>EPS],
        individually_infeasible_jobs=[j for j in jobs if j['unavoidable_transfer_stall_lower_bound_ms']>EPS],
        definitions=dict(current='Each admitted current request own per-SSU V/C over admission-to-completion, including stalled spans; never exempted.',
            continuing='Vnext/Ccurrent for same-request internal layers, same-role next-request L0, and long L-to-short S next-request L0, over actual current compute window.',
            boundary='The cross_role_boundary schema key contains only short S-to-long L cross-request next-Layer-0 Vnext/Ccurrent contributions. Long L-to-short S is not exempt. S/L are input read-length role labels, not native QoS SS/LL classes.',
            exception='Only short S-to-long L cross-request Layer-0 contributions are exempt. Continuing demand, including every long L-to-short S contribution, must stay within capacity. No time interval is removed from current-demand check.',
            near='Fleet time at >=90% of total capacity, limited to 5% in each audit window by default; applies to current and continuing, not exempt boundary contributions. Per-SSU near-capacity time is reported but is not a hard rejection criterion; per-SSU peak <=40 remains mandatory.',
            lower_bound='max(max_s V_s/40,total V/50)-C, floored at zero, is an exclusive-transfer unavoidable stall lower bound; it omits small per-IO/control latencies.',
            causality='Stall above the exclusive lower bound is not proven FIFO loss. Temporal overlap of internal stalls and boundary IO/debt is correlation; same-input scheduler comparison or detailed IO trace is needed for causation.',
            debt='Vnext/Ccurrent ends at the deadline even if IO remains. Boundary overdue IO spans are retained separately; no reference is an all-interval scheduling feasibility proof.',
            full_run_utilization='Fixed fleet wall-time utilization includes draining tail idle; active_U excludes no-request idle. Warm all-NPU occupancy is checked explicitly.',
            cold='Initial L0 has no preceding compute window; included in current, physical traffic and utilization but no synthetic prefetch reference.',
            profiles='Every profile is rebuilt with run_fixed128_32.profile from the SHA256-verified data file. Full construction metadata and raw compute anchors are checked. Lengths below 32K, down to 20K, use explicitly labelled uncalibrated extrapolation; they are not measured data rows.'))
    ident=case.name+'_'+hashlib.sha256(str(case).encode()).hexdigest()[:8]
    result_audit['audit_output_path']=str(out/'audits'/f'{ident}_audit.json')
    write_json(Path(result_audit['audit_output_path']),result_audit)
    return result_audit


def summary_row(d):
    f,w=d['windows']['full_run'],d['windows']['warm_2000_4000ms']
    row={k:d[k] for k in ('name','case','policy','seed','num_ssu','num_npu','U_percent','slo_1p5_percent','boundary_exempt_candidate_pass','capacity_only_candidate_pass','audit_output_path')}
    row.update(full_current_peak=max(f['current']['peak_per_unit_gib_s']),full_current_near_percent=f['current']['any_unit_near_capacity_percent'],
        full_continuing_peak=max(f['continuing']['peak_per_unit_gib_s']),full_continuing_near_percent=f['continuing']['any_unit_near_capacity_percent'],
        full_total_prefetch_peak=max(f['total_prefetch']['peak_per_unit_gib_s']),full_total_prefetch_over_ms=f['total_prefetch']['any_unit_over_capacity_ms'],
        full_no_boundary_total_peak=max(f['total_when_no_boundary_reference']['peak_per_unit_gib_s']),
        all_window_max_current_near_percent=max(x['current']['any_unit_near_capacity_percent'] for x in d['windows'].values() if not x.get('unavailable')),
        all_window_max_continuing_near_percent=max(x['continuing']['any_unit_near_capacity_percent'] for x in d['windows'].values() if not x.get('unavailable')),
        all_window_max_current_fleet_near_percent=max(x['current']['fleet_near_capacity_percent'] for x in d['windows'].values() if not x.get('unavailable')),
        all_window_max_continuing_fleet_near_percent=max(x['continuing']['fleet_near_capacity_percent'] for x in d['windows'].values() if not x.get('unavailable')))
    if not w.get('unavailable'):
        long_to_short = w['job_stats']['long_to_short_cross_request']
        row.update(warm_internal_stall_card_ms=w['utilization']['internal_stall_card_ms'],warm_l0_stall_card_ms=w['utilization']['l0_stall_card_ms'],
            warm_boundary_stall_card_ms=w['job_stats']['cross_role_boundary']['actual_stall_card_ms'],
            warm_boundary_unavoidable_lower_bound_card_ms=w['job_stats']['cross_role_boundary']['unavoidable_transfer_stall_lower_bound_card_ms'],
            warm_boundary_reference_union_ms=w['job_stats']['cross_role_boundary']['reference_window_union']['any_job_ms'],
            warm_boundary_io_and_reference_union_ms=w['job_stats']['cross_role_boundary']['reference_plus_overdue_io_union']['any_job_ms'],
            warm_L_to_S_stall_card_ms=long_to_short['actual_stall_card_ms'],
            warm_L_to_S_unavoidable_lower_bound_card_ms=long_to_short['unavoidable_transfer_stall_lower_bound_card_ms'],
            warm_internal_stall_overlapping_boundary_io_card_ms=w['internal_stall_temporal_overlap_with_boundary_reference_or_overdue_io_card_ms'])
        for suffix in ('stall_card_ms','unavoidable_lower_bound_card_ms',
                       'reference_union_ms','io_and_reference_union_ms'):
            row['warm_S_to_L_'+suffix] = row['warm_boundary_'+suffix]
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path)
    p.add_argument('--case',action='append',type=Path)
    p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--near-fraction',type=float,default=.9)
    p.add_argument('--max-near-percent',type=float,default=5.)
    p.add_argument('--summary-prefix',default='boundary_native_summary')
    a=p.parse_args()
    cases=a.case or [f.parent for f in sorted((a.root or DEFAULT).rglob('metrics.json'))]
    rows=[]
    for case in cases:
        if all((case/f).exists() for f in ('manifest.json.gz','result.json.gz','metadata.json','metrics.json')):
            row=summary_row(audit(case,a.out,a.near_fraction,a.max_near_percent));rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(a.out/(a.summary_prefix+'.json'),rows)
    with (a.out/(a.summary_prefix+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['case']);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    main()
