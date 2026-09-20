#!/usr/bin/env python3
"""Read-only audit for fixed per-NPU 7A+42B+1C+42D, 8 NPUs and 1 SSU.

The time accounting below is copied from audit_abcd_8npu.audit_case with only
request-count expectations changed from 408/51 to 736/92. It does not change
any simulator inputs, scheduling decisions, or native outputs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import sim
from audit_abcd_8npu import ROOT, read, same, group, clip, sweep, summarize
from audit_abcd_fixed_8npu import PROFILES, reference_profile

COUNTS = {'A': 7, 'B': 42, 'C': 1, 'D': 42}


def audit_case(case):
    man = read(case/'manifest.json.gz')
    result = read(case/'result.json.gz')
    metrics = read(case/'metrics.json')
    requests = {q['request_id']: q for q in man['requests']}
    batches = result['summary']['microbatch_metrics']
    assert len(batches) == 736 and result['summary']['request_count'] == 736
    assert all(result['summary']['invariants'].values())
    current, continuing, boundary, total_prefetch = [], [], [], []
    same_request, all_cross_request = [], []
    records, timelines = [], []
    first_last = []
    for n in range(8):
        lane = sorted((b for b in batches if b['npu_id'] == n), key=lambda b:b['admission_time_ms'])
        assert len(lane) == 92
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


def validate_input(man, previous):
    metadata = man['metadata']
    assert (metadata['num_npu'], metadata['num_ssu'], metadata['n_layers']) == (8, 1, 8)
    assert metadata['layout'] == 'block_ring_hash'
    assert metadata['placement_virtual_nodes_per_ssu'] == sim.BLOCK_RING_VIRTUAL_NODES == 256
    assert len(man['requests']) == 736 and len(previous['requests']) == 408
    table = ast.literal_eval((ROOT / 'data').read_text())
    data_sha = hashlib.sha256((ROOT / 'data').read_bytes()).hexdigest()
    assert metadata['data_sha256'] == data_sha
    expected = {g: reference_profile(table, g) for g in PROFILES}
    old_by_identity = {q['load']['original_request_id']: q for q in previous['requests']}
    assert len(old_by_identity) == 408
    matched, new_ids, execution_ids = set(), set(), set()
    unique_values = {g: set() for g in PROFILES}
    block_count = 0
    for q in man['requests']:
        rid, npu, load = q['request_id'], q['npu_id'], q['load']
        oid = load['original_request_id']
        assert rid not in execution_ids
        execution_ids.add(rid)
        assert load['request_id'] == rid and load['npu_id'] == npu
        assert load['generation'] == rid - npu * 1000000
        assert q['arrival_time_ms'] == 0.
        gid = group(load)
        if oid in old_by_identity:
            assert oid not in matched
            matched.add(oid)
            old = old_by_identity[oid]
            assert q['npu_id'] == old['npu_id'] and gid == group(old['load'])
            assert load['previous_request_id'] == old['request_id']
            assert load['added_in_bd_heavy_revision'] is False
            same(q['arrival_time_ms'], old['arrival_time_ms'])
            for name in ('seq_len_k', 'nql', 'per_layer_us', 'per_layer_kv_gb',
                         'required_bw_input_gbps', 'profile_construction'):
                same(load[name], old['load'][name])
        else:
            assert oid not in new_ids and gid == 'D'
            assert load['added_in_bd_heavy_revision'] is True
            new_ids.add(oid)
        p = expected[gid]
        assert load['role'] == ('L' if gid in ('A', 'C') else 'S')
        assert load['total_tokens'] == p['total_k'] * 1024
        assert load['seq_len_k'] == p['total_k']
        assert load['nql'] == p['nql']
        assert load['ssd_prefix_tokens'] == p['prefix_tokens']
        same(load['per_layer_us'], p['compute_us'])
        same(load['original_compute_us'], p['compute_us'])
        same(load['per_layer_kv_gb'], p['read_gib'])
        same(load['required_bw_input_gbps'], p['demand_gib_s'])
        assert load['category'] == sim.classify_request(p['total_k'], p['nql'])
        assert load['constructed_profile'] is (not p['direct_data'])
        assert load['padding_gib_per_layer'] == 0.
        construction = load['profile_construction']
        assert construction['source'] == 'data'
        assert construction['compute_scale'] == 1.
        assert construction['extrapolated'] is (gid in ('B', 'D'))
        assert construction['method'] == ('direct_data_row' if p['direct_data'] else
                                          'bilinear_length_extrapolation_nql_interpolation')
        same(construction['anchors'], p['anchors'])
        layers = man['placements'][q['placement_index']]
        assert len(layers) in (1, 8) and all(layer == layers[0] for layer in layers)
        assert p['prefix_tokens'] % 128 == 0
        assert len(layers[0]) == p['prefix_tokens'] // 128
        for index, (ssu, volume) in enumerate(layers[0]):
            assert ssu == sim.block_ring_hash_disk_id(rid, index, 1)
            same(volume, 128 * 1408 / 2**30)
        same(math.fsum(v for _, v in layers[0]), p['read_gib'])
        block_count += len(layers[0])
        unique_values[gid].add((load['total_tokens'], load['nql'], load['per_layer_us'],
                                load['per_layer_kv_gb'], load['required_bw_input_gbps']))
    assert matched == set(old_by_identity) and len(new_ids) == 328
    assert all(len(values) == 1 for values in unique_values.values())
    lanes = []
    for npu in range(8):
        lane = sorted((q for q in man['requests'] if q['npu_id'] == npu), key=lambda q:q['request_id'])
        old = sorted((q for q in previous['requests'] if q['npu_id'] == npu), key=lambda q:q['request_id'])
        assert Counter(group(q['load']) for q in lane) == COUNTS
        assert [q['request_id'] for q in lane] == [npu * 1000000 + i for i in range(92)]
        kept = [q for q in lane if q['load']['original_request_id'] in old_by_identity]
        added = [q for q in lane if q['load']['original_request_id'] not in old_by_identity]
        assert len(kept) == 51 and len(added) == 41
        assert all(group(q['load']) == 'D' for q in added)
        assert sorted(q['load']['generation'] for q in added) == sorted(
            random.Random(7 + 100003 * npu).sample(range(92), 41))
        assert [q['load']['original_request_id'] for q in added] == [
            npu * 1000000 + 100 + index for index in range(41)]
        assert [group(q['load']) for q in kept] == [group(q['load']) for q in old]
        assert [q['load']['original_request_id'] for q in kept] == [q['load']['original_request_id'] for q in old]
        lanes.append(dict(npu_id=npu, counts=dict(COUNTS),
                          order=[group(q['load']) for q in lane],
                          added_D_positions=[q['load']['generation'] for q in added],
                          old_relative_order_preserved=True))
    return dict(all_four_profiles_exactly_fixed=True,
                all_408_original_identities_and_relative_order_preserved=True,
                new_requests_are_exactly_41_D_per_npu=True,
                all_execution_ids_renumbered_by_new_position=True,
                all_eight_layers_reuse_identical_block_placement=True,
                all_blocks_match_ring_hash_key=True,
                verified_block_mappings=block_count,
                full_blocks_only=True, data_sha256=data_sha,
                profiles=expected, per_npu=lanes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True,
                   help='Previous no-jitter 7A42B1C1D native case directory')
    p.add_argument('--fifo', type=Path, required=True)
    p.add_argument('--once', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    previous = read(a.previous / 'manifest.json.gz')
    fifo = read(a.fifo / 'manifest.json.gz')
    once = read(a.once / 'manifest.json.gz')
    same(fifo['requests'], once['requests'])
    same(fifo['placements'], once['placements'])
    assert fifo['input_fingerprint'] == once['input_fingerprint']
    result = dict(input_checks=validate_input(fifo, previous),
                  same_input_across_policies=True,
                  source_previous_manifest=str(a.previous / 'manifest.json.gz'),
                  source_previous_manifest_sha256=hashlib.sha256((a.previous / 'manifest.json.gz').read_bytes()).hexdigest(),
                  definitions=dict(
                      ttft='completion minus admission; pre-admission queue time excluded',
                      slo15='TTFT <= 1.5 * 8 * own per-layer compute time; all 736 requests',
                      warm_admitted_slo15='same threshold for requests admitted in [2000,4000)',
                      U='actual compute intersection with [2000,4000), divided by 8*2000 card-ms',
                      class_U='class compute card-ms divided by class active card-ms in [2000,4000)',
                      current='sum current admitted request own V/C, including stalled spans',
                      same_request='internal-layer Vnext/Ccurrent; all cross-request L0 contributions removed individually',
                      latest_underload_criterion='current and same_request separately stay <=40 GiB/s in full run; never delete entire time intervals',
                      demand_warning='current and same_request are separate reference curves and are not added',
                      stall_warning='all actual stalls, including cross-request stalls, remain in U and TTFT'),
                  fifo=audit_case(a.fifo), once=audit_case(a.once))
    result['full_run_normal_under_capacity_both_policies'] = all(
        result[policy]['windows']['full_run']['normal_demand_within_capacity']
        for policy in ('fifo', 'once'))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({policy: dict(
        U_percent=result[policy]['windows']['warm_2000_4000ms']['U_percent'],
        slo15_all=result[policy]['slo15_all']['all'],
        slo15_warm_admitted=result[policy]['slo15_warm_admitted']['all'],
        normal_full_run=result[policy]['windows']['full_run']['demand']['current'],
        internal_full_run=result[policy]['windows']['full_run']['demand']['same_request'],
        full_run_normal_under_capacity=result[policy]['windows']['full_run']['normal_demand_within_capacity'])
        for policy in ('fifo', 'once')}, indent=2))


if __name__ == '__main__':
    main()
