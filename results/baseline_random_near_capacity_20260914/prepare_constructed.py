#!/usr/bin/env python3
"""Honest affine-data-extrapolation inputs; frozen simulator/runner unchanged."""
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import random

import experiment as base

HERE = Path(__file__).resolve().parent


def prepare(spec, plan, seed, horizon=22000.):
    profiles, provenance = base.profiles_for('raw', f"{spec['long_total_k']}:{spec['long_nql']}")
    long = profiles[0]
    long.update(role='L', name='L')
    total = int(spec['short_total_k']) * 1024
    miss = int(spec['short_nql'])
    volume = (total - miss) * 1408 / 2**30
    compute = float(spec['short_compute_us'])
    short = dict(npu_id=1, role='S', name='S', seq_len_k=int(spec['short_total_k']),
        total_tokens=total, nql=miss, ssd_prefix_tokens=total-miss,
        per_layer_kv_gib=volume, per_layer_compute_us=compute,
        required_bandwidth_gibps=volume*1e6/compute,
        source_equivalent_ttft_78_layers_ms=None,
        extrapolated_78_layer_pure_compute_ms=78*compute/1000,
        construction=dict(method='affine_extrapolation_below_raw_minimum',
            measured_data_row=False, raw_data_min_total_k=32,
            model_provenance=spec.get('model_provenance', plan.get('model_provenance')),
            candidate_plan_sha256=base.sha(HERE/'constructed_candidates.json'),
            compute_model_scale=spec.get('compute_scale',1.),
            limitation='Extrapolated prefill compute, not measured at this input length; V uses exact retained-prefix token count.'))
    profiles.append(short)
    counts = tuple(int(x) for x in spec['counts'])
    roles = ('L','S')
    disks = int(spec['num_ssu'])
    assert len(counts)==2 and min(counts)>0 and disks>0
    for p in profiles:
        assert p['total_tokens']>=10240 and p['ssd_prefix_tokens']%128==0
        assert p['per_layer_compute_us']>0
        assert p['required_bandwidth_gibps']<base.NPU_BW
    costs = [base.LAYERS*p['per_layer_compute_us']/1000 for p in profiles]
    cycle_ms = math.fsum(n*c for n,c in zip(counts,costs))
    repeats = math.ceil(horizon/cycle_ms)
    canonical = [i for i,n in enumerate(counts) for _ in range(n*repeats)]
    requests, assignments, disk_means, disk_maxima = [],[],[],[]
    for npu in range(base.NPU):
        identities = list(range(len(canonical)))
        random.Random(seed+100003*npu).shuffle(identities)
        placements,rates=[],[]
        for p in profiles:
            layer = tuple(((j+npu)%disks,base.BLOCK_GIB) for j in range(p['ssd_prefix_tokens']//128))
            assert math.isclose(math.fsum(v for _,v in layer),p['per_layer_kv_gib'],abs_tol=1e-12)
            placements.append((layer,))
            rates.append([math.fsum(v for d,v in layer if d==s)*1e6/p['per_layer_compute_us'] for s in range(disks)])
        disk_means.append([math.fsum(rates[i][s]*costs[i]*counts[i] for i in range(2))/cycle_ms for s in range(disks)])
        disk_maxima.append([max(r[s] for r in rates) for s in range(disks)])
        for position,original in enumerate(identities):
            idx=canonical[original]; p=profiles[idx]; rid=npu*1000000+position
            load=dict(request_id=rid,npu_id=npu,generation=position,original_request_id=npu*1000000+original,
                profile_index=idx,role=p['role'],seq_len_k=p['seq_len_k'],nql=p['nql'],
                total_tokens=p['total_tokens'],ssd_prefix_tokens=p['ssd_prefix_tokens'],
                category=base.sim.classify_request(p['seq_len_k'],p['nql']),
                per_layer_us=p['per_layer_compute_us'],per_layer_kv_gb=p['per_layer_kv_gib'],
                required_bw_input_gbps=p['required_bandwidth_gibps'],
                source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                original_compute_us=p['per_layer_compute_us'],constructed_profile=idx==1,
                profile_construction=p['construction'],padding_gib_per_layer=0.,
                arrival_time=0.,arrival_ms=0.,initial=True)
            requests.append(base.ContinuousBatchRequest.from_normalized(rid,npu,0.,load,placements[idx]))
        assert Counter(canonical[i] for i in identities)=={i:n*repeats for i,n in enumerate(counts)}
        assignments.append(dict(npu_id=npu,requests=len(canonical),pure_compute_ms=repeats*cycle_ms,
            role_counts={r:n*repeats for r,n in zip(roles,counts)},
            role_pure_compute_ms={r:c*n*repeats for r,c,n in zip(roles,costs,counts)},
            shuffle_seed=seed+100003*npu,first_32_roles=[roles[canonical[i]] for i in identities[:32]]))
    requests=tuple(requests)
    fp=base.continuous_batch_input_fingerprint(requests)
    label=f"{spec['name']}_ssu{disks}_h{horizon:g}_seed{seed}"
    averages=[math.fsum(r[s] for r in disk_means) for s in range(disks)]
    maxima=[math.fsum(r[s] for r in disk_maxima) for s in range(disks)]
    metadata=dict(experiment='baseline_random_near_capacity_20260914',label=label,case_id=label+'_'+fp[:12],
        candidate=spec['name'],num_npu=base.NPU,num_ssu=disks,n_layers=base.LAYERS,seed=seed,
        disk_bw_gib_s=base.DISK_BW,npu_bw_gib_s=base.NPU_BW,family='data_affine_extrapolation',
        profiles=profiles,source=provenance,
        profile_keys=f"{long['seq_len_k']}:{long['nql']},{short['seq_len_k']}:{short['nql']}",
        role_names=list(roles),count_ratio=list(counts),equal_176kib_blocks=True,blocks='exact',layout='stripe_npu_mod_ssu',
        input_fingerprint=fp,logical_input_fingerprint=base.logical_input_fingerprint(requests),
        source_data_sha256=base.sha(base.ROOT/'data'),compute_scale_actual=1.,constructed_profile=True,
        constructed_short_profile=True,constructed_builder_sha256=base.sha(__file__),
        constructed_plan_sha256=base.sha(HERE/'constructed_candidates.json'),
        last_arrival_ms=0.,request_count=len(requests),order='random',order_mode='random',
        regime='saturated_finite_backlog',horizon_pure_compute_ms=horizon,quota_cycles=repeats,
        per_npu_assignment=assignments,measurement_window_ms=[2000.,4000.],
        placement_rule='(block_index+npu_id)%num_ssu; exact 176KiB; one placement reused for 8 layers',
        random_rule='Independent full-population shuffle Random(seed+100003*npu); no repeated deck or seed rejection',
        population_rule='Same integer count ratio on every card; minimum repetitions covering pure-compute horizon before shuffle',
        static_per_ssu_upper_bound_gib_s=maxima,static_upper_bound_passes=all(v<=base.DISK_BW for v in maxima),
        time_weighted_per_ssu_nominal_gib_s=averages,time_weighted_fleet_nominal_gib_s=math.fsum(averages),
        ideal_load_ratio=math.fsum(averages)/(disks*base.DISK_BW),
        nominal_definition='Current admitted per-disk V/C; next-request L0 not added again; every physical read simulated',
        caveat='Short C extrapolated below raw minimum, not a measured data row. Near-capacity is ideal mean, not instantaneous per-disk underload.')
    path=HERE/'inputs'/(label+'.json.gz')
    if path.exists():
        previous=base.read_json(path)
        assert previous['input_fingerprint']==fp and previous['metadata']==metadata
    base.save_manifest(path,requests,metadata)
    restored,m=base.load_manifest(path)
    assert base.continuous_batch_input_fingerprint(restored)==fp and m==metadata
    return dict(manifest=str(path),label=label,requests=len(requests),ideal_load_ratio=metadata['ideal_load_ratio'],
                expected_blocks=sum(len(r.placement[0])*base.LAYERS for r in requests),input_fingerprint=fp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',action='append')
    parser.add_argument('--seed',type=int,action='append')
    args=parser.parse_args()
    plan=base.read_json(HERE/'constructed_candidates.json')
    for spec in plan['selected_candidates']:
        if args.name and spec['name'] not in args.name:continue
        for seed in args.seed or [7]:
            print(json.dumps(prepare(spec,plan,seed),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
