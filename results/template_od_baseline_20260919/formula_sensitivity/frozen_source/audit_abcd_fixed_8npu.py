#!/usr/bin/env python3
"""Read-only audit of fixed A/B/C/D profiles and their native FIFO/Once runs.

The existing audit's timeline integration is reused. Input checks are specific
to the no-jitter workload and compare all 408 identities and per-NPU positions
against the previous ABCD manifest. No simulator input/result is modified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import sim
from audit_abcd_8npu import ROOT, audit_case, group, read, same


PROFILES = {'A': (200, 2048), 'B': (20, 1152),
            'C': (200, 1024), 'D': (20, 2048)}
COUNTS = {'A': 7, 'B': 42, 'C': 1, 'D': 1}


def reference_profile(table, gid):
    """Reconstruct the four profiles directly from their explicit data anchors."""
    total_k, nql = PROFILES[gid]
    if gid in ('A', 'C'):
        anchors = [(total_k, nql, 1.)]
    elif gid == 'B':
        # NQL1152 is 1/8 of the interval [1024, 2048]. Length20K uses
        # extrapolation weights 1.75 for 32K and -0.75 for 48K.
        anchors = [(32, 1024, 1.75 * .875), (32, 2048, 1.75 * .125),
                   (48, 1024, -.75 * .875), (48, 2048, -.75 * .125)]
    else:
        anchors = [(32, 2048, 1.75), (48, 2048, -.75)]
    compute_us = math.fsum(table[k, q][1] * w for k, q, w in anchors)
    prefix = total_k * 1024 - nql
    read_gib = prefix * 1408 / 2**30
    return dict(total_k=total_k, nql=nql, prefix_tokens=prefix,
                read_mib=read_gib * 1024, read_gib=read_gib,
                compute_us=compute_us, compute_ms=compute_us / 1000,
                demand_gib_s=read_gib / (compute_us / 1e6),
                direct_data=gid in ('A', 'C'),
                anchors=[dict(seq_len_k=k, nql=q, weight=w,
                              compute_us=table[k, q][1]) for k, q, w in anchors])


def validate_input(man, previous):
    metadata = man['metadata']
    assert (metadata['num_npu'], metadata['num_ssu'], metadata['n_layers']) == (8, 1, 8)
    assert metadata['layout'] == 'block_ring_hash'
    assert metadata['placement_virtual_nodes_per_ssu'] == sim.BLOCK_RING_VIRTUAL_NODES == 256
    assert len(man['requests']) == len(previous['requests']) == 408
    table = ast.literal_eval((ROOT / 'data').read_text())
    data_sha = hashlib.sha256((ROOT / 'data').read_bytes()).hexdigest()
    assert metadata['data_sha256'] == data_sha
    expected = {g: reference_profile(table, g) for g in PROFILES}
    old_by_id = {q['request_id']: q for q in previous['requests']}
    assert len(old_by_id) == len(man['requests'])
    matched = set()
    unique_values = {g: set() for g in PROFILES}
    block_count = 0
    for q in man['requests']:
        rid, npu, load = q['request_id'], q['npu_id'], q['load']
        assert rid not in matched and rid in old_by_id
        matched.add(rid)
        old = old_by_id[rid]
        for field in ('request_id', 'npu_id', 'arrival_time_ms'):
            same(q[field], old[field])
        for field in ('request_id', 'original_request_id', 'npu_id', 'generation', 'profile_group'):
            same(load[field], old['load'][field])
        assert load['request_id'] == rid and load['npu_id'] == npu
        assert q['arrival_time_ms'] == 0.
        gid = group(load)
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
    assert matched == set(old_by_id)
    assert all(len(values) == 1 for values in unique_values.values())
    lanes = []
    for npu in range(8):
        lane = sorted((q for q in man['requests'] if q['npu_id'] == npu), key=lambda q:q['request_id'])
        old = sorted((q for q in previous['requests'] if q['npu_id'] == npu), key=lambda q:q['request_id'])
        assert Counter(group(q['load']) for q in lane) == COUNTS
        assert [q['request_id'] for q in lane] == [npu * 1000000 + i for i in range(51)]
        assert [group(q['load']) for q in lane] == [group(q['load']) for q in old]
        assert [q['load']['original_request_id'] for q in lane] == [q['load']['original_request_id'] for q in old]
        lanes.append(dict(npu_id=npu, counts=dict(COUNTS),
                          order=[group(q['load']) for q in lane],
                          C_position=next(i for i,q in enumerate(lane) if group(q['load']) == 'C'),
                          D_position=next(i for i,q in enumerate(lane) if group(q['load']) == 'D')))
    return dict(all_four_profiles_exactly_fixed=True,
                all_408_request_ids_original_ids_npu_positions_preserved=True,
                all_eight_layers_reuse_identical_block_placement=True,
                all_blocks_match_ring_hash_key=True,
                verified_block_mappings=block_count,
                full_blocks_only=True, data_sha256=data_sha,
                profiles=expected, per_npu=lanes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True,
                   help='Previous jittered ABCD native case directory')
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
                      slo15='TTFT <= 1.5 * 8 * own per-layer compute time; all 408 requests',
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
