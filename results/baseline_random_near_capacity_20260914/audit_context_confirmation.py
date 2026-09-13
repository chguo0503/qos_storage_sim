#!/usr/bin/env python3
"""Independently verify context confirmation manifests; do not import any builder."""
import argparse
import ast
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BLOCK_GIB=176/1048576


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fit_ms(raw,miss):
    points=sorted((k[0],v[1]/1000) for k,v in raw.items() if k[1]==miss)
    x=statistics.mean(x for x,y in points);y=statistics.mean(y for x,y in points)
    slope=math.fsum((a-x)*(b-y) for a,b in points)/math.fsum((a-x)**2 for a,b in points)
    return y-slope*x,slope


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9),(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=HERE/'context384_confirmation_preparation.json')
    parser.add_argument('--name',action='append')
    parser.add_argument('--seed',type=int,action='append')
    parser.add_argument('--horizon-ms',type=float,default=22000.)
    parser.add_argument('--output',type=Path,default=HERE/'context_confirmation_input_audit.json')
    args=parser.parse_args()
    names=args.name or ['context8_L384m1024_S10m128']
    seeds=args.seed or [19,43,67,101]
    horizon=args.horizon_ms
    assert math.isfinite(horizon) and horizon>=22000
    expected_cases={(name,seed) for name in names for seed in seeds}
    plan_path=HERE/'long_scale_math.json'
    protected=[plan_path,HERE/'long_scale_math.py',HERE/'prepare_context_scale.py',
        HERE/'experiment.py',HERE/'constructed_candidates.json',HERE/'math_followup_constructed.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw=ast.literal_eval((ROOT/'data').read_text());data_sha=sha(ROOT/'data')
    plan=json.loads(plan_path.read_text());plan_sha=sha(plan_path)
    prepared_path=args.prepared
    text=prepared_path.read_text()
    payload=[json.loads(line) for line in text.splitlines() if line.strip()] if prepared_path.suffix=='.jsonl' else json.loads(text)
    def gather_records(value):
        if isinstance(value,list):return [r for child in value for r in gather_records(child)]
        if isinstance(value,dict):
            if isinstance(value.get('manifest'),str):return [value]
            return [r for child in value.values() for r in gather_records(child)]
        return []
    records=gather_records(payload)
    assert len(records)==len(expected_cases) and len(plan['selected_candidates'])==4
    assert len({r['manifest'] for r in records})==len(records)
    observed_cases=set()
    by_name={c['name']:c for c in plan['selected_candidates']}
    fits={m:fit_ms(raw,m) for m in (128,1024)}
    audits=[];source_hash_cache={}
    for record in records:
        path=Path(record['manifest']);manifest_sha=sha(path)
        if 'manifest_sha256' in record:assert record['manifest_sha256']==manifest_sha
        with gzip.open(path,'rt') as f:man=json.load(f)
        meta=man['metadata'];candidate=by_name[meta['candidate']]
        name=candidate['name'];seed=meta['seed'];label=f'{name}_ssu8_h{horizon:g}_seed{seed}'
        assert (name,seed) in expected_cases and (name,seed) not in observed_cases
        observed_cases.add((name,seed))
        assert meta['label']==label and path.name==label+'.json.gz'
        if 'label' in record:assert record['label']==label
        assert meta['constructed_plan_sha256']==plan_sha
        assert Path(meta['constructed_plan_path']).resolve()==plan_path.resolve()
        assert meta['constructed_builder_sha256']==sha(HERE/'prepare_context_scale.py')
        assert meta['frozen_experiment_sha256']==sha(HERE/'experiment.py')
        assert meta['source_data_sha256']==data_sha and meta['source']['source_sha256']==data_sha
        assert meta['num_npu']==32 and meta['num_ssu']==8 and meta['n_layers']==8 and meta['seed']==seed
        assert meta['disk_bw_gib_s']==40 and meta['npu_bw_gib_s']==50
        assert meta['order']==meta['order_mode']=='random'
        assert meta['family']=='data_affine_context_scale' and meta['constructed_profile']
        assert meta['constructed_long_profile']==(not candidate['long_is_raw'])
        assert meta['constructed_short_profile'] and meta['compute_scale_actual']==1.
        assert meta['horizon_pure_compute_ms']==horizon and meta['measurement_window_ms']==[2000,4000]
        assert meta['last_arrival_ms']==0 and meta['role_names']==['L','S']
        assert meta['count_ratio']==candidate['counts'] and meta['equal_176kib_blocks']
        assert meta['blocks']=='exact' and meta['layout']=='stripe_npu_mod_ssu'
        for filename,digest in meta['core_source_sha256'].items():
            if filename not in source_hash_cache:source_hash_cache[filename]=sha(ROOT/filename)
            assert source_hash_cache[filename]==digest,filename

        profiles=[]
        for idx,role in enumerate(('L','S')):
            K=candidate['long_total_k'] if idx==0 else 10
            miss=1024 if idx==0 else 128
            is_raw=idx==0 and K==200
            a,b=fits[miss];c_us=raw[K,miss][1] if is_raw else (a+b*K)*1000
            hit=K*1024-miss;assert hit%128==0
            v=hit//128*BLOCK_GIB;B=v/(c_us/1e6)
            assert B<50 and K>=10
            expected_method='direct_data_row' if is_raw else (
                'affine_extrapolation_above_raw_maximum' if idx==0 else 'affine_extrapolation_below_raw_minimum')
            source=meta['profiles'][idx]
            assert source['role']==role and source['seq_len_k']==K and source['nql']==miss
            near(source['per_layer_compute_us'],c_us);near(source['per_layer_kv_gib'],v)
            near(source['required_bandwidth_gibps'],B)
            assert source['construction']['method']==expected_method
            if is_raw:
                assert source['per_layer_compute_us']==raw[K,miss][1]
                assert source['per_layer_kv_gib']==raw[K,miss][3]
                assert source['source_equivalent_ttft_78_layers_ms']==raw[K,miss][2]
            else:
                assert source['source_equivalent_ttft_78_layers_ms'] is None
                assert source['construction']['measured_data_row'] is False
                assert source['construction']['candidate_plan_sha256']==plan_sha
                model=source['construction']['compute_model']
                near(model['intercept_ms'],a);near(model['slope_ms_per_K'],b)
                assert model['fixed_miss']==miss and model['method']==expected_method
                if idx==1:assert model['frozen_source_sha256']==sha(HERE/'constructed_candidates.json')
                near(source['extrapolated_78_layer_pure_compute_ms'],78*c_us/1000)
            profiles.append(dict(role=role,total_K=K,miss=miss,C_us=c_us,V_GiB=v,B_GiB_s=B,
                raw=is_raw,method=expected_method,blocks=hit//128,source=source))
        counts=candidate['counts'];cycle=math.fsum(8*p['C_us']/1000*n for p,n in zip(profiles,counts))
        repeats=math.ceil(horizon/cycle);population=[n*repeats for n in counts]
        assert meta['quota_cycles']==repeats
        canonical=[0]*population[0]+[1]*population[1]
        assert len(man['requests'])==len(canonical)*32==meta['request_count']
        if 'requests' in record:assert len(man['requests'])==record['requests']
        groups=defaultdict(list)
        for r in man['requests']:groups[r['npu_id']].append(r)
        assert set(groups)==set(range(32))
        cached_placement={i:tuple(tuple((int(ssu),float(v)) for ssu,v in layer) for layer in item)
                          for i,item in enumerate(man['placements'])}
        checked_placement=set();role_sequences=set();disk_work=[0.]*8;full_compute=[]
        per_npu=[];fp=hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
        placement_repr={i:repr(x) for i,x in cached_placement.items()}
        all_sorted=[];raw_count=constructed_count=0;blocks_total=0
        for npu in range(32):
            requests=sorted(groups[npu],key=lambda r:r['request_id'])
            assert len(requests)==len(canonical)
            expected_order=list(range(len(canonical)))
            random.Random(seed+100003*npu).shuffle(expected_order)
            seen_roles=[];card_compute=[];card_work=[0.]*8
            for position,(r,original) in enumerate(zip(requests,expected_order)):
                load=r['load'];idx=canonical[original];p=profiles[idx]
                rid=npu*1000000+position
                assert r['request_id']==load['request_id']==rid and load['npu_id']==npu
                assert load['original_request_id']==npu*1000000+original
                assert load['profile_index']==idx and load['role']==p['role']
                assert load['generation']==position and load['initial'] is True
                assert r['arrival_time_ms']==load['arrival_time']==load['arrival_ms']==0.
                assert load['seq_len_k']==p['total_K'] and load['nql']==p['miss']
                assert load['total_tokens']==p['total_K']*1024
                assert load['ssd_prefix_tokens']==p['total_K']*1024-p['miss']
                assert load['constructed_profile']==(not p['raw'])
                assert load['profile_construction']==p['source']['construction']
                assert load['padding_gib_per_layer']==0.
                assert load['source_ttft_ms']==(raw[p['total_K'],p['miss']][2] if p['raw'] else None)
                near(load['per_layer_us'],p['C_us']);near(load['original_compute_us'],p['C_us'])
                near(load['per_layer_kv_gb'],p['V_GiB']);near(load['required_bw_input_gbps'],p['B_GiB_s'])
                assert load['category']==('LL' if idx==0 else 'SS')
                pi=r['placement_index'];placement=cached_placement[pi]
                key=(pi,npu%8,idx)
                if key not in checked_placement:
                    expected=(tuple(((j+npu)%8,BLOCK_GIB) for j in range(p['blocks'])),)
                    assert placement==expected
                    checked_placement.add(key)
                fp.update(f"({rid}, {npu}, {float(r['arrival_time_ms'])!r}, {load['category']!r}, {load['per_layer_us']!r}, {placement_repr[pi]})".encode())
                seen_roles.append(p['role']);card_compute.append(8*p['C_us']/1000)
                raw_count+=int(p['raw']);constructed_count+=int(not p['raw']);blocks_total+=8*p['blocks']
                all_sorted.append(r)
            role_sequences.add(tuple(seen_roles))
            total_compute=math.fsum(card_compute)
            assert total_compute>=horizon;near(total_compute,repeats*cycle)
            for p,n in zip(profiles,population):
                for j in range(p['blocks']):card_work[(j+npu)%8]+=n*8*BLOCK_GIB
            for disk in range(8):disk_work[disk]+=card_work[disk]/(total_compute/1000)
            full_compute.append(total_compute)
            assignment=meta['per_npu_assignment'][npu]
            assert assignment['npu_id']==npu and assignment['shuffle_seed']==seed+100003*npu
            assert assignment['role_counts']==dict(zip(('L','S'),population))==Counter(seen_roles)
            assert assignment['first_32_roles']==seen_roles[:32]
            near(assignment['pure_compute_ms'],total_compute)
            per_npu.append(dict(npu=npu,requests=len(requests),role_counts=dict(Counter(seen_roles)),pure_compute_ms=total_compute))
        fingerprint=fp.hexdigest()
        assert fingerprint==man['input_fingerprint']==meta['input_fingerprint']
        if 'input_fingerprint' in record:assert fingerprint==record['input_fingerprint']
        logical_rows=[dict(request_id=r['request_id'],npu_id=r['npu_id'],arrival_time_ms=r['arrival_time_ms'],load=r['load']) for r in all_sorted]
        logical_hash=hashlib.sha256(json.dumps(logical_rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert logical_hash==meta['logical_input_fingerprint']
        assert len(role_sequences)==32
        if 'expected_blocks' in record:assert blocks_total==record['expected_blocks']
        rho=math.fsum(disk_work)/320
        near(rho,meta['ideal_load_ratio']);near(rho,candidate['rho_ideal'])
        if 'ideal_load_ratio' in record:near(rho,record['ideal_load_ratio'])
        assert .95<=rho<=1.05
        for a,b in zip(disk_work,meta['time_weighted_per_ssu_nominal_gib_s']):near(a,b)
        near(math.fsum(disk_work),meta['time_weighted_fleet_nominal_gib_s'])
        audits.append(dict(candidate=name,seed=seed,horizon_pure_compute_ms=horizon,manifest=str(path.relative_to(ROOT)),manifest_sha256=manifest_sha,
            input_fingerprint=fingerprint,logical_input_fingerprint=logical_hash,
            audited_requests=len(all_sorted),raw_request_count=raw_count,constructed_request_count=constructed_count,
            unique_full_role_queues=32,per_npu=per_npu,profiles=[{k:v for k,v in p.items() if k!='source'} for p in profiles],
            expected_physical_block_commands=blocks_total,rho_ideal=rho,
            per_ssu_ideal_GiB_s=disk_work,pure_compute_ms_min=min(full_compute),pure_compute_ms_max=max(full_compute),
            checks=dict(raw_source_unchanged=True,constructed_C_independently_fit=True,constructed_source_TTFT_null=True,
                no_padding=True,exact_176KiB_stripes=True,all_full_shuffles_reconstructed=True,
                all_fixed_NPU_assignments_verified=True,all_NPU_pure_compute_at_least_requested_horizon=True,
                full_v2_and_logical_fingerprints_rebuilt=True,all_declared_source_hashes_match=True)))
        assert sha(path)==manifest_sha
        print(json.dumps(dict(candidate=name,seed=seed,requests=len(all_sorted),rho=rho,pure_ms=min(full_compute),passed=True)),flush=True)
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert observed_cases==expected_cases
    assert len({a['input_fingerprint'] for a in audits})==len(audits)
    out=dict(schema_version='independent-context-confirmation-input-audit-v1',all_checks_passed=True,
        no_builder_imported=True,no_simulations_run=True,
        scope=f'All requested context confirmation inputs: names={names}, seeds={seeds}, pure-compute horizon={horizon:g}ms',
        fit_recomputed_from_raw={str(k):dict(intercept_ms=a,slope_ms_per_K=b) for k,(a,b) in fits.items()},
        total_audited_requests=sum(a['audited_requests'] for a in audits),cases=audits,
        plan_sha256=plan_sha,prepared_jsonl_sha256=sha(prepared_path),protected_files_sha256=before,
        core_files_verified_sha256=source_hash_cache,generator_sha256=sha(Path(__file__)),
        caveat='Input coverage and>=22s pure work do not prove actual warm mixing or per-instant per-disk underload; those require final execution statistics.')
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(all_checks_passed=True,cases=len(audits),requests=out['total_audited_requests'])),flush=True)


if __name__=='__main__':
    main()
