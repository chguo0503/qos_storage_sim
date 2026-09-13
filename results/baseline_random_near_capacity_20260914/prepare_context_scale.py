#!/usr/bin/env python3
"""Prepare explicit context-scale model stress inputs, without any simulation.

Raw 200K/miss1024 is preserved. Longer contexts and 10K/miss128 are
transparently extrapolated from the frozen plan, never called measured data.
Generation is intentionally separate from executing the frozen runner.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import json
import math
import random

import experiment as base

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE/'long_scale_math.json'


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-8), (a, b)


def extrapolated_profile(role, total_k, miss, compute_us, model, provenance, plan_hash):
    total = int(total_k)*1024
    assert total >= 10240 and 0 <= miss < total and (total-miss) % 128 == 0
    assert total_k < 32 or total_k > 200, 'This study extrapolates outside the raw [32,200]K range.'
    direction = 'below_raw_minimum' if total_k < 32 else 'above_raw_maximum'
    method = 'affine_extrapolation_'+direction
    assert model['method'] == method and model['fixed_miss'] == miss
    predicted_us = (float(model['intercept_ms'])+float(model['slope_ms_per_K'])*total_k)*1000
    close(compute_us, predicted_us)
    assert math.isfinite(compute_us) and compute_us > 0
    volume = (total-miss)*1408/2**30
    return dict(npu_id=0 if role == 'L' else 1, role=role, name=role,
                seq_len_k=int(total_k), total_tokens=total, nql=int(miss), ssd_prefix_tokens=total-miss,
                per_layer_kv_gib=volume, per_layer_compute_us=float(compute_us),
                required_bandwidth_gibps=volume*1e6/compute_us,
                source_equivalent_ttft_78_layers_ms=None,
                extrapolated_78_layer_pure_compute_ms=78*compute_us/1000,
                construction=dict(method=method, measured_data_row=False,
                    raw_data_min_total_k=32, raw_data_max_total_k=200,
                    compute_model=copy.deepcopy(model), model_provenance=copy.deepcopy(provenance),
                    candidate_plan_path=str(PLAN_PATH), candidate_plan_sha256=plan_hash,
                    compute_model_scale=1.,
                    limitation='Affine extrapolated prefill compute, not measured at this context length. Exact KV-prefix block volume is part of a simulator model stress test, not a real-device claim.'))


def prepare(spec, plan, seed, horizon=22000.):
    horizon = float(horizon)
    assert math.isfinite(horizon) and horizon >= 22000
    assert isinstance(seed, int)
    assert int(spec['num_ssu']) == 8
    assert int(spec['long_total_k']) in (200, 256, 384, 512) and int(spec['long_nql']) == 1024
    assert int(spec['short_total_k']) == 10 and int(spec['short_nql']) == 128
    assert isinstance(spec['long_is_raw'], bool)
    assert spec['long_is_raw'] == (int(spec['long_total_k']) == 200)
    # Identity and source metadata are tied to the exact frozen plan on disk.
    plan_hash, builder_hash = base.sha(PLAN_PATH), base.sha(__file__)
    assert base.read_json(PLAN_PATH) == plan
    assert any(candidate == spec for candidate in plan['selected_candidates'])
    data_hash = base.sha(base.ROOT/'data')
    provenance = copy.deepcopy(spec['model_provenance'])
    assert provenance['source_data_sha256'] == provenance['data_sha256'] == data_hash
    long_model, short_model = provenance['long_compute_model'], provenance['short_compute_model']
    assert short_model['fixed_miss'] == 128 and long_model['fixed_miss'] == 1024
    short_source = HERE/short_model['frozen_source']
    assert short_source.resolve().is_relative_to(HERE)
    assert base.sha(short_source) == short_model['frozen_source_sha256']
    core_before, experiment_before = base.source_hashes(), base.sha(base.__file__)
    anchors, source = base.profiles_for('raw', '200:1024')
    if spec['long_is_raw']:
        long = anchors[0]
        long.update(role='L', name='L')
        assert long_model['method'] == 'direct_data_row' and long_model['raw_key'] == [200, 1024]
        close(long['per_layer_compute_us'], spec['long_compute_us'])
    else:
        long = extrapolated_profile('L', int(spec['long_total_k']), int(spec['long_nql']),
                                    float(spec['long_compute_us']), long_model, provenance, plan_hash)
    short = extrapolated_profile('S', int(spec['short_total_k']), int(spec['short_nql']),
                                 float(spec['short_compute_us']), short_model, provenance, plan_hash)
    profiles, roles = [long, short], ('L', 'S')
    flags = [p['construction']['method'] != 'direct_data_row' for p in profiles]
    assert flags == [not spec['long_is_raw'], True]
    counts = tuple(int(x) for x in spec['counts'])
    assert len(counts) == 2 and min(counts) > 0
    disks = int(spec['num_ssu'])
    for p, constructed in zip(profiles, flags):
        assert p['total_tokens'] >= 10240 and p['ssd_prefix_tokens'] % 128 == 0
        assert p['per_layer_compute_us'] > 0 and p['required_bandwidth_gibps'] < base.NPU_BW
        assert (p['source_equivalent_ttft_78_layers_ms'] is None) == constructed
    costs = [base.LAYERS*p['per_layer_compute_us']/1000 for p in profiles]
    cycle_ms = math.fsum(n*c for n, c in zip(counts, costs))
    repeats = math.ceil(horizon/cycle_ms)
    canonical = [i for i, n in enumerate(counts) for _ in range(n*repeats)]
    assert len(canonical) < 1000000, 'Queue positions must not overlap another card request-ID namespace.'
    requests, assignments, disk_means, disk_maxima = [], [], [], []
    for npu in range(base.NPU):
        identities = list(range(len(canonical)))
        random.Random(seed+100003*npu).shuffle(identities)
        placements, rates = [], []
        for p in profiles:
            layer = tuple(((j+npu)%disks, base.BLOCK_GIB) for j in range(p['ssd_prefix_tokens']//128))
            assert math.isclose(math.fsum(v for _, v in layer), p['per_layer_kv_gib'], abs_tol=1e-12)
            placements.append((layer,))
            rates.append([math.fsum(v for d, v in layer if d == s)*1e6/p['per_layer_compute_us'] for s in range(disks)])
        disk_means.append([math.fsum(rates[i][s]*costs[i]*counts[i] for i in range(2))/cycle_ms for s in range(disks)])
        disk_maxima.append([max(r[s] for r in rates) for s in range(disks)])
        for position, original in enumerate(identities):
            idx = canonical[original]
            p, rid = profiles[idx], npu*1000000+position
            load = dict(request_id=rid, npu_id=npu, generation=position,
                original_request_id=npu*1000000+original, profile_index=idx,
                role=p['role'], seq_len_k=p['seq_len_k'], nql=p['nql'],
                total_tokens=p['total_tokens'], ssd_prefix_tokens=p['ssd_prefix_tokens'],
                category=base.sim.classify_request(p['seq_len_k'], p['nql']),
                per_layer_us=p['per_layer_compute_us'], per_layer_kv_gb=p['per_layer_kv_gib'],
                required_bw_input_gbps=p['required_bandwidth_gibps'],
                source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                original_compute_us=p['per_layer_compute_us'], constructed_profile=flags[idx],
                profile_construction=p['construction'], padding_gib_per_layer=0.,
                arrival_time=0., arrival_ms=0., initial=True)
            requests.append(base.ContinuousBatchRequest.from_normalized(rid, npu, 0., load, placements[idx]))
        assert Counter(canonical[i] for i in identities) == {i: n*repeats for i, n in enumerate(counts)}
        assignments.append(dict(npu_id=npu, requests=len(canonical), pure_compute_ms=repeats*cycle_ms,
            role_counts={r: n*repeats for r, n in zip(roles, counts)},
            role_pure_compute_ms={r: c*n*repeats for r, c, n in zip(roles, costs, counts)},
            shuffle_seed=seed+100003*npu, first_32_roles=[roles[canonical[i]] for i in identities[:32]]))
    requests = tuple(requests)
    fp = base.continuous_batch_input_fingerprint(requests)
    label = f"{spec['name']}_ssu{disks}_h{horizon:g}_seed{seed}"
    averages = [math.fsum(r[s] for r in disk_means) for s in range(disks)]
    maxima = [math.fsum(r[s] for r in disk_maxima) for s in range(disks)]
    metadata = dict(experiment='baseline_random_near_capacity_20260914', label=label, case_id=label+'_'+fp[:12],
        candidate=spec['name'], num_npu=base.NPU, num_ssu=disks, n_layers=base.LAYERS, seed=seed,
        disk_bw_gib_s=base.DISK_BW, npu_bw_gib_s=base.NPU_BW, family='data_affine_context_scale',
        profiles=profiles, source=source, model_provenance=provenance,
        profile_keys=f"{long['seq_len_k']}:{long['nql']},{short['seq_len_k']}:{short['nql']}",
        role_names=list(roles), count_ratio=list(counts), equal_176kib_blocks=True, blocks='exact', layout='stripe_npu_mod_ssu',
        input_fingerprint=fp, logical_input_fingerprint=base.logical_input_fingerprint(requests),
        source_data_sha256=data_hash, compute_scale_actual=1., constructed_profile=any(flags),
        constructed_long_profile=flags[0], constructed_short_profile=flags[1],
        profile_is_constructed_by_role=dict(zip(roles, flags)),
        constructed_builder_sha256=builder_hash, constructed_plan_sha256=plan_hash,
        constructed_plan_path=str(PLAN_PATH), core_source_sha256=core_before, frozen_experiment_sha256=experiment_before,
        last_arrival_ms=0., request_count=len(requests), order='random', order_mode='random',
        regime='saturated_finite_backlog', horizon_pure_compute_ms=horizon, quota_cycles=repeats,
        per_npu_assignment=assignments, measurement_window_ms=[2000., 4000.],
        placement_rule='(block_index+npu_id)%num_ssu; exact 176KiB; one placement reused for 8 layers',
        random_rule='Independent full-population shuffle Random(seed+100003*npu); no repeated deck, synchronized barriers or seed rejection',
        population_rule='Same integer count ratio on every card; minimum repetitions covering pure-compute horizon before shuffle',
        horizon_change_rule='Changing the pure-compute horizon changes quotas and the entire shuffled population; it is not a continuation or shared prefix of the 22s population.',
        static_per_ssu_upper_bound_gib_s=maxima, static_upper_bound_passes=all(v <= base.DISK_BW for v in maxima),
        time_weighted_per_ssu_nominal_gib_s=averages, time_weighted_fleet_nominal_gib_s=math.fsum(averages),
        ideal_load_ratio=math.fsum(averages)/(disks*base.DISK_BW),
        nominal_definition='Current admitted per-disk V/C; next-request L0 not added again; every physical read simulated',
        purpose='Model sensitivity to absolute long-read burst scale at approximately similar long V/C and near-capacity ideal aggregate load; FIFO burst costs are absent from a fluid bandwidth-only model.',
        caveat='200K long profile is an unchanged data row; larger long C and all 10K short C are explicit extrapolations. This is not real 256/384/512K hardware or original-data performance. Near-capacity is ideal mean, not per-instant disk underload; real warm mixed coverage requires post-run audit.')
    path = HERE/'inputs'/(label+'.json.gz')
    if path.exists():
        previous = base.read_json(path)
        assert previous['input_fingerprint'] == fp and previous['metadata'] == metadata
    assert base.source_hashes() == core_before and base.sha(base.__file__) == experiment_before
    assert base.sha(PLAN_PATH) == plan_hash and base.sha(__file__) == builder_hash
    base.save_manifest(path, requests, metadata)
    restored, restored_metadata = base.load_manifest(path)
    assert base.continuous_batch_input_fingerprint(restored) == fp and restored_metadata == metadata
    return dict(manifest=str(path), label=label, requests=len(requests),
                ideal_load_ratio=metadata['ideal_load_ratio'], horizon_pure_compute_ms=horizon,
                constructed_long_profile=flags[0], constructed_short_profile=flags[1],
                expected_blocks=sum(len(r.placement[0])*base.LAYERS for r in requests), input_fingerprint=fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', action='append')
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--horizon-ms', type=float, default=22000.)
    args = parser.parse_args()
    if not math.isfinite(args.horizon_ms) or args.horizon_ms < 22000:
        parser.error('--horizon-ms must be finite and >= 22000')
    plan = base.read_json(PLAN_PATH)
    names = {spec['name'] for spec in plan['selected_candidates']}
    if args.name and not set(args.name) <= names:
        parser.error('Unknown --name: '+', '.join(sorted(set(args.name)-names)))
    for spec in plan['selected_candidates']:
        if args.name and spec['name'] not in args.name:
            continue
        for seed in args.seed or [7]:
            print(json.dumps(prepare(spec, plan, seed, horizon=args.horizon_ms), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
