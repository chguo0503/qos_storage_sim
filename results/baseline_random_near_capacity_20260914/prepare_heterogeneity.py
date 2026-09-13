#!/usr/bin/env python3
"""Prepare paired within-class shape controls, never run a simulation.

The complete coarse L/S identity shuffle is fixed before deterministic subtype
assignment. All C values are explicit affine extrapolations from the plan.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import json
import math
import random

import experiment as base

HERE=Path(__file__).resolve().parent
PLAN_PATH=HERE/'heterogeneity_math.json'


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-8),(a,b)


def profile(spec,index,plan_hash):
    K,miss=int(spec['total_k']),int(spec['nql'])
    total=K*1024;hit=total-miss
    assert K>=10 and (K<32 or K>200) and hit>0 and hit%128==0
    assert spec['is_raw'] is False and spec['source_ttft_ms'] is None
    provenance=copy.deepcopy(spec['model_provenance']);model=provenance['compute_model']
    data_hash=base.sha(base.ROOT/'data')
    assert provenance['source_data_sha256']==provenance['data_sha256']==data_hash
    assert model['fixed_miss']==miss
    method='affine_extrapolation_'+('below_raw_minimum' if K<32 else 'above_raw_maximum')
    assert model['method']==method
    frozen_source=(HERE/model['frozen_source']).resolve()
    assert frozen_source.is_relative_to(HERE)
    assert base.sha(frozen_source)==model['frozen_source_sha256']
    C=float(spec['compute_us']);V=hit//128*base.BLOCK_GIB
    close(C,(model['intercept_ms']+model['slope_ms_per_K']*K)*1000)
    assert math.isfinite(C) and C>0 and V==spec['V_GiB']
    assert hit//128==spec['blocks_per_layer']
    B=V*1e6/C;close(B,spec['required_B_GiB_s']);assert B<base.NPU_BW
    assert base.sim.classify_request(K,miss)==spec['category']
    return dict(npu_id=index,role=spec['role'],name=spec['role'],coarse_role=spec['coarse_role'],
        seq_len_k=K,total_tokens=total,nql=miss,ssd_prefix_tokens=hit,
        per_layer_compute_us=C,per_layer_kv_gib=V,required_bandwidth_gibps=B,
        source_equivalent_ttft_78_layers_ms=None,extrapolated_78_layer_pure_compute_ms=78*C/1000,
        fraction_of_coarse_role=spec['fraction_of_coarse_role'],
        construction=dict(method=method,measured_data_row=False,raw_data_min_total_k=32,
            raw_data_max_total_k=200,compute_model=copy.deepcopy(model),model_provenance=provenance,
            candidate_plan_path=str(PLAN_PATH),candidate_plan_sha256=plan_hash,compute_model_scale=1.,
            limitation='Both short and long compute are affine extrapolations, not measured hardware. Within-class context variation is a controlled simulator assumption; exact cached-token bytes contain no padding.'))


def prepare(spec,plan,seed,horizon=22000.):
    assert math.isfinite(horizon) and horizon>=22000 and isinstance(seed,int)
    assert base.read_json(PLAN_PATH)==plan and spec in plan['selected_candidates']
    assert spec['num_npu']==base.NPU==32 and spec['num_ssu']==8 and spec['n_layers']==base.LAYERS==8
    plan_hash,builder_hash=base.sha(PLAN_PATH),base.sha(__file__)
    core_before,experiment_before=base.source_hashes(),base.sha(base.__file__)
    assert plan['source_data_sha256']==base.sha(base.ROOT/'data')
    profiles=[profile(p,i,plan_hash) for i,p in enumerate(spec['profiles'])]
    roles=[p['role'] for p in profiles]
    coarse_roles=['L','S'];coarse_ratio=spec['coarse_counts']
    assert spec['coarse_roles']==coarse_roles and coarse_ratio[0]==1
    by_coarse={r:[i for i,p in enumerate(profiles) if p['coarse_role']==r] for r in coarse_roles}
    assert set(roles)==set(spec['short_roles'])|set(spec['long_roles'])
    mean_C={r:math.fsum(profiles[i]['per_layer_compute_us']/len(indices) for i in indices)/1000
            for r,indices in by_coarse.items()}
    cycle=base.LAYERS*math.fsum(n*mean_C[r] for n,r in zip(coarse_ratio,coarse_roles))
    multiple=spec['quota_repeat_multiple']
    repeats=math.ceil(math.ceil(horizon/cycle)/multiple)*multiple
    coarse_counts=dict(zip(coarse_roles,(n*repeats for n in coarse_ratio)))
    counts=[]
    for p in profiles:
        fraction=p['fraction_of_coarse_role'];indices=by_coarse[p['coarse_role']]
        assert fraction==[1,len(indices)]
        assert coarse_counts[p['coarse_role']]%len(indices)==0
        counts.append(coarse_counts[p['coarse_role']]//len(indices))
    total_C=math.fsum(n*p['per_layer_compute_us'] for n,p in zip(counts,profiles))
    total_V=math.fsum(n*p['per_layer_kv_gib'] for n,p in zip(counts,profiles))
    pure_ms=base.LAYERS*total_C/1000;assert pure_ms>=horizon
    N=sum(counts);assert N<1000000
    canonical=[]
    for original in range(N):
        coarse='L' if original<coarse_counts['L'] else 'S'
        indices=by_coarse[coarse]
        offset=original if coarse=='L' else original-coarse_counts['L']
        canonical.append(indices[offset%len(indices)])
    assert Counter(canonical)==dict(enumerate(counts))
    if horizon==22000:
        expected=spec['minimum_22s_population']
        assert repeats==expected['repeats'] and N==expected['requests_per_card']
        assert counts==[p['count_per_npu'] for p in spec['profiles']]
        assert total_C==expected['per_layer_C_work_us_per_card']
        assert total_V==expected['per_layer_V_work_GiB_per_card']
    requests=[];assignments=[];streams=[];disk_means=[];disk_maxima=[]
    for npu in range(base.NPU):
        identities=list(range(N));random.Random(seed+100003*npu).shuffle(identities)
        streams.append([[npu*1000000+position,npu*1000000+original,
                         profiles[canonical[original]]['coarse_role']]
                        for position,original in enumerate(identities)])
        placements=[];rates=[];volumes=[]
        for p in profiles:
            layer=tuple(((j+npu)%8,base.BLOCK_GIB) for j in range(p['ssd_prefix_tokens']//128))
            assert math.fsum(v for _,v in layer)==p['per_layer_kv_gib']
            placements.append((layer,))
            per_disk=[math.fsum(v for d,v in layer if d==s) for s in range(8)]
            volumes.append(per_disk);rates.append([v*1e6/p['per_layer_compute_us'] for v in per_disk])
        disk_means.append([math.fsum(n*volumes[i][s] for i,n in enumerate(counts))*1e6/total_C for s in range(8)])
        disk_maxima.append([max(row[s] for row in rates) for s in range(8)])
        for position,original in enumerate(identities):
            index=canonical[original];p=profiles[index];rid=npu*1000000+position
            load=dict(request_id=rid,npu_id=npu,generation=position,
                original_request_id=npu*1000000+original,profile_index=index,
                role=p['role'],coarse_role=p['coarse_role'],seq_len_k=p['seq_len_k'],nql=p['nql'],
                total_tokens=p['total_tokens'],ssd_prefix_tokens=p['ssd_prefix_tokens'],
                category=base.sim.classify_request(p['seq_len_k'],p['nql']),
                per_layer_us=p['per_layer_compute_us'],per_layer_kv_gb=p['per_layer_kv_gib'],
                required_bw_input_gbps=p['required_bandwidth_gibps'],source_ttft_ms=None,
                original_compute_us=p['per_layer_compute_us'],constructed_profile=True,
                profile_construction=p['construction'],padding_gib_per_layer=0.,
                arrival_time=0.,arrival_ms=0.,initial=True)
            requests.append(base.ContinuousBatchRequest.from_normalized(rid,npu,0.,load,placements[index]))
        assignments.append(dict(npu_id=npu,requests=N,pure_compute_ms=pure_ms,
            role_counts=dict(zip(roles,counts)),coarse_role_counts=coarse_counts,
            role_pure_compute_ms={p['role']:n*base.LAYERS*p['per_layer_compute_us']/1000 for p,n in zip(profiles,counts)},
            shuffle_seed=seed+100003*npu,first_32_roles=[profiles[canonical[i]]['role'] for i in identities[:32]],
            first_32_coarse_roles=[profiles[canonical[i]]['coarse_role'] for i in identities[:32]]))
    coarse_sha=hashlib.sha256(json.dumps(streams,separators=(',',':')).encode()).hexdigest()
    if horizon==22000 and str(seed) in spec['coarse_input_sha256_by_seed']:
        assert coarse_sha==spec['coarse_input_sha256_by_seed'][str(seed)]
    old_reference_verified=False
    if spec['name']=='hetero_L352_384_416_S10' and seed==7 and horizon==22000:
        old_path=base.ROOT/plan['original_context384_manifest']
        assert base.sha(old_path)==plan['original_context384_manifest_sha256']
        old=base.read_json(old_path);old_streams=[[] for _ in range(base.NPU)]
        for q in old['requests']:
            old_streams[q['npu_id']].append([q['request_id'],q['load']['original_request_id'],q['load']['role']])
        for stream in old_streams:stream.sort()
        assert old_streams==streams
        original_counts=old['metadata']['per_npu_assignment'][0]['role_counts']
        assert total_C==math.fsum(original_counts[p['role']]*p['per_layer_compute_us'] for p in old['metadata']['profiles'])
        assert total_V==math.fsum(original_counts[p['role']]*p['per_layer_kv_gib'] for p in old['metadata']['profiles'])
        old_reference_verified=True
    requests=tuple(requests);fp=base.continuous_batch_input_fingerprint(requests)
    label=f"{spec['name']}_ssu8_h{horizon:g}_seed{seed}"
    averages=[math.fsum(row[s] for row in disk_means) for s in range(8)]
    maxima=[math.fsum(row[s] for row in disk_maxima) for s in range(8)]
    divisor=math.gcd(*counts);ratio=[n//divisor for n in counts]
    _,source=base.profiles_for('raw','200:1024')
    metadata=dict(experiment='baseline_random_near_capacity_20260914',label=label,case_id=label+'_'+fp[:12],
        candidate=spec['name'],num_npu=32,num_ssu=8,n_layers=8,seed=seed,
        disk_bw_gib_s=base.DISK_BW,npu_bw_gib_s=base.NPU_BW,family='data_affine_heterogeneity',
        plan_family=spec['family'],profiles=profiles,source=source,
        profile_keys=','.join(f"{p['seq_len_k']}:{p['nql']}" for p in profiles),
        role_names=roles,coarse_roles=coarse_roles,short_roles=spec['short_roles'],long_roles=spec['long_roles'],
        coarse_role_by_role={p['role']:p['coarse_role'] for p in profiles},
        count_ratio=ratio,quota_cycles=divisor,coarse_count_ratio=coarse_ratio,coarse_quota_cycles=repeats,
        quota_repeat_multiple=multiple,equal_176kib_blocks=True,blocks='exact',layout='stripe_npu_mod_ssu',
        input_fingerprint=fp,logical_input_fingerprint=base.logical_input_fingerprint(requests),
        coarse_input_sha256=coarse_sha,coarse_input_hash_definition='SHA256 of JSON nested 32 lanes of [request_id,original_request_id,coarse_role], separators comma/colon.',
        per_card_per_layer_total_compute_us=total_C,per_card_per_layer_total_read_GiB=total_V,
        source_data_sha256=base.sha(base.ROOT/'data'),compute_scale_actual=1.,constructed_profile=True,
        constructed_long_profile=True,constructed_short_profile=True,
        profile_is_constructed_by_role={r:True for r in roles},
        constructed_builder_sha256=builder_hash,constructed_plan_sha256=plan_hash,
        constructed_plan_path=str(PLAN_PATH),core_source_sha256=core_before,frozen_experiment_sha256=experiment_before,
        last_arrival_ms=0.,request_count=len(requests),order='random',order_mode='random',
        regime='saturated_finite_backlog',horizon_pure_compute_ms=horizon,
        per_npu_assignment=assignments,measurement_window_ms=[2000.,4000.],
        placement_rule='(block_index+npu_id)%8; exact176KiB; one placement reused for8 layers',
        random_rule=spec['canonical_rule'],subtype_rule=spec['subtype_rule'],
        population_rule='Round minimum coarse repeats up to the plan multiple, preserving exact subtype fractions before one full identity shuffle.',
        horizon_change_rule='A changed horizon is a new full shuffle, not an old prefix; long subtypes require coarse repeats multiple of3. Only the audited22s/seed7 population is declared identical to the saved original context384 coarse stream.',
        static_per_ssu_upper_bound_gib_s=maxima,static_upper_bound_passes=all(v<=base.DISK_BW for v in maxima),
        time_weighted_per_ssu_nominal_gib_s=averages,time_weighted_fleet_nominal_gib_s=math.fsum(averages),
        ideal_load_ratio=math.fsum(averages)/(8*base.DISK_BW),
        nominal_definition='Current admitted per-disk V/C; next-request L0 not added again; every physical read simulated',
        matched_control_name=spec['matched_control_name'],comparison_scope=spec['comparison_scope'],
        original_context384_coarse_ID_order_and_totals_verified=old_reference_verified,
        purpose='Within-class shape robustness at matched per-card totalC/totalV/request count/coarseL-S input order; not another search for lower utilization.',
        caveat='All long and short C values are explicit affine extrapolations, with source TTFT None. Same queued coarse identities do not guarantee same admission times or window populations. Warm requires actual L/S coverage, not coverage of every subtype. Near-capacity ideal mean is not instantaneous per-disk underload.')
    close(metadata['ideal_load_ratio'],spec['rho_ideal'])
    path=HERE/'inputs'/(label+'.json.gz')
    if path.exists():
        previous=base.read_json(path)
        assert previous['input_fingerprint']==fp and previous['metadata']==metadata
    else:
        base.save_manifest(path,requests,metadata)
    restored,restored_meta=base.load_manifest(path)
    assert base.continuous_batch_input_fingerprint(restored)==fp and restored_meta==metadata
    assert base.sha(PLAN_PATH)==plan_hash and base.sha(__file__)==builder_hash
    assert base.source_hashes()==core_before and base.sha(base.__file__)==experiment_before
    expected_blocks=sum(n*p['ssd_prefix_tokens']//128 for n,p in zip(counts,profiles))*32*8
    return dict(manifest=str(path),manifest_sha256=base.sha(path),label=label,requests=len(requests),
        input_fingerprint=fp,coarse_input_sha256=coarse_sha,role_counts=dict(zip(roles,counts)),
        per_card_pure_compute_ms=pure_ms,per_card_per_layer_total_compute_us=total_C,
        per_card_per_layer_total_read_GiB=total_V,ideal_load_ratio=metadata['ideal_load_ratio'],
        expected_blocks=expected_blocks,horizon_pure_compute_ms=horizon,
        original_context384_coarse_ID_order_and_totals_verified=old_reference_verified)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',action='append');parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--horizon-ms',type=float,default=22000.)
    args=parser.parse_args()
    if not math.isfinite(args.horizon_ms) or args.horizon_ms<22000:
        parser.error('--horizon-ms must be finite and >=22000')
    plan=base.read_json(PLAN_PATH);names={s['name'] for s in plan['selected_candidates']}
    if args.name and not set(args.name)<=names:parser.error('Unknown --name')
    for spec in plan['selected_candidates']:
        if args.name and spec['name'] not in args.name:continue
        for seed in args.seed or [7]:
            print(json.dumps(prepare(spec,plan,seed,args.horizon_ms),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
