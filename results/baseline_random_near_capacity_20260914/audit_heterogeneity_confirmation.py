#!/usr/bin/env python3
"""Independently audit selected heterogeneity confirmation inputs, never simulate."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLOCK = 176 / 1048576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), (a, b)


def fit(raw, miss):
    points = sorted((k[0], v[1] / 1000) for k, v in raw.items() if k[1] == miss)
    x = statistics.mean(p[0] for p in points)
    y = statistics.mean(p[1] for p in points)
    slope = math.fsum((a-x)*(b-y) for a, b in points) / math.fsum((a-x)**2 for a, _ in points)
    return y - slope*x, slope


def audit(record, spec, seed, horizon, plan_path, plan_hash, fits, core_hashes):
    path = Path(record['manifest'])
    if not path.is_absolute():
        path = ROOT/path
    manifest_hash = sha(path)
    assert manifest_hash == record['manifest_sha256']
    manifest = read(path)
    m = manifest['metadata']
    name = spec['name']
    label = f'{name}_ssu8_h{horizon:g}_seed{seed}'
    assert m['label'] == record['label'] == label and path.name == label+'.json.gz'
    assert m['candidate'] == name and m['seed'] == seed
    assert (m['num_npu'], m['num_ssu'], m['n_layers']) == (32, 8, 8)
    assert (m['disk_bw_gib_s'], m['npu_bw_gib_s']) == (40, 50)
    assert m['family'] == 'data_affine_heterogeneity'
    assert m['plan_family'] == spec['family'] == 'data_affine_heterogeneity_control'
    # The runtime records its imported-source subset; the study plan additionally
    # covers historical tests and report generators. Verify both scopes, without
    # requiring those intentionally different key sets to be identical.
    assert {'sim.py','continuous_batch_sim.py','shared_path_once.py','shared_path_sim_adapter.py','data'} <= set(m['core_source_sha256'])
    assert all(core_hashes[name] == h for name,h in m['core_source_sha256'].items())
    assert m['source_data_sha256'] == core_hashes['data']
    assert m['source']['source_sha256'] == core_hashes['data']
    assert m['constructed_plan_sha256'] == plan_hash
    assert Path(m['constructed_plan_path']).resolve() == plan_path.resolve()
    assert m['constructed_builder_sha256'] == sha(HERE/'prepare_heterogeneity.py')
    assert m['frozen_experiment_sha256'] == sha(HERE/'experiment.py')
    assert m['order'] == m['order_mode'] == 'random'
    assert m['horizon_pure_compute_ms'] == record['horizon_pure_compute_ms'] == horizon
    assert m['measurement_window_ms'] == [2000, 4000] and m['last_arrival_ms'] == 0
    assert m['blocks'] == 'exact' and m['layout'] == 'stripe_npu_mod_ssu' and m['equal_176kib_blocks']
    assert m['short_roles'] == spec['short_roles'] and m['long_roles'] == spec['long_roles']
    assert m['constructed_profile'] and m['constructed_long_profile'] and m['constructed_short_profile']
    assert m['compute_scale_actual'] == 1
    assert len(m['profiles']) == len(spec['profiles'])
    profiles = []
    for p, z in zip(spec['profiles'], m['profiles']):
        a, b = fits[p['nql']]
        C = (a + b*p['total_k'])*1000
        hit = p['total_k']*1024-p['nql']
        assert hit % 128 == 0 and p['total_k'] >= 10
        count = hit//128
        V = count*BLOCK
        B = V*1e6/C
        near(C, p['compute_us']); near(V, p['V_GiB']); near(B, p['required_B_GiB_s'])
        near(C, z['per_layer_compute_us']); near(V, z['per_layer_kv_gib']); near(B, z['required_bandwidth_gibps'])
        assert z['role'] == p['role'] and z['coarse_role'] == p['coarse_role']
        assert z['seq_len_k'] == p['total_k'] and z['nql'] == p['nql'] and B < 50
        assert z['source_equivalent_ttft_78_layers_ms'] is None and not z['construction']['measured_data_row']
        method = 'affine_extrapolation_below_raw_minimum' if p['total_k'] < 32 else 'affine_extrapolation_above_raw_maximum'
        construction = z['construction']
        assert construction['method'] == method and construction['candidate_plan_sha256'] == plan_hash
        model = construction['compute_model']
        assert model == p['model_provenance']['compute_model']
        assert model['frozen_source_sha256'] == sha(HERE/model['frozen_source'])
        near(model['intercept_ms'], a); near(model['slope_ms_per_K'], b)
        near(z['extrapolated_78_layer_pure_compute_ms'], 78*C/1000)
        profiles.append(dict(role=p['role'], coarse=p['coarse_role'], K=p['total_k'], miss=p['nql'], C=C, V=V, B=B,
            blocks=count, source=z, fraction=p['fraction_of_coarse_role']))
    by_coarse = {r:[i for i,p in enumerate(profiles) if p['coarse']==r] for r in ['L','S']}
    mean_C = {r:math.fsum(profiles[i]['C'] for i in indices)/len(indices) for r,indices in by_coarse.items()}
    cycle_ms = 8*math.fsum(n*mean_C[r] for n,r in zip(spec['coarse_counts'], ['L','S']))/1000
    multiple = spec['quota_repeat_multiple']
    repeats = math.ceil(math.ceil(horizon/cycle_ms)/multiple)*multiple
    NL, NS = (n*repeats for n in spec['coarse_counts'])
    coarse_counts = dict(L=NL, S=NS)
    counts = []
    for p in profiles:
        subtype_count = len(by_coarse[p['coarse']])
        assert p['fraction'] == [1, subtype_count] and coarse_counts[p['coarse']] % subtype_count == 0
        counts.append(coarse_counts[p['coarse']]//subtype_count)
    total_C = math.fsum(n*p['C'] for n,p in zip(counts,profiles))
    total_V = math.fsum(n*p['V'] for n,p in zip(counts,profiles))
    expected_blocks = 32*8*sum(n*p['blocks'] for n,p in zip(counts,profiles))
    if horizon == 22000:
        pop = spec['minimum_22s_population']
        assert counts == [p['count_per_npu'] for p in spec['profiles']]
        assert repeats == pop['repeats'] and (NL,NS) == (pop['L'],pop['S'])
        near(total_C,pop['per_layer_C_work_us_per_card']); near(total_V,pop['per_layer_V_work_GiB_per_card'])
        assert expected_blocks == pop['expected_blocks_all_32']
    divisor = math.gcd(*counts)
    role_counts = dict(zip([p['role'] for p in profiles],counts))
    assert m['role_names'] == [p['role'] for p in profiles]
    assert m['count_ratio'] == [n//divisor for n in counts] and m['quota_cycles'] == divisor
    assert m['coarse_count_ratio'] == spec['coarse_counts'] and m['coarse_quota_cycles'] == repeats
    assert len(manifest['requests']) == m['request_count'] == record['requests'] == (NL+NS)*32
    assert record['role_counts'] == role_counts
    lanes = defaultdict(list)
    for r in manifest['requests']:
        lanes[r['npu_id']].append(r)
    assert set(lanes) == set(range(32)) and len(m['per_npu_assignment']) == 32
    assert [r['request_id'] for r in manifest['requests']] == sorted(r['request_id'] for r in manifest['requests'])
    places = {i:tuple(tuple((int(s),float(v)) for s,v in layer) for layer in item) for i,item in enumerate(manifest['placements'])}
    representations = {i:repr(p) for i,p in places.items()}
    checked = set(); coarse = []; logical = []; totals = []; blocks = 0
    fingerprint = hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
    for npu in range(32):
        ordered = sorted(lanes[npu],key=lambda r:r['request_id'])
        identities = list(range(NL+NS)); random.Random(seed+100003*npu).shuffle(identities)
        assert len(ordered) == NL+NS
        lane_coarse = []; observed = Counter(); C_values = []; V_values = []; fine = []
        for position,(r,original) in enumerate(zip(ordered,identities)):
            cr = 'L' if original < NL else 'S'
            offset = original if cr == 'L' else original-NL
            idx = by_coarse[cr][offset % len(by_coarse[cr])]
            p = profiles[idx]; l = r['load']; rid = npu*1000000+position; orig = npu*1000000+original
            assert r['request_id'] == l['request_id'] == rid and r['npu_id'] == l['npu_id'] == npu
            assert l['original_request_id'] == orig and l['profile_index'] == idx
            assert l['coarse_role'] == cr and l['role'] == p['role']
            assert l['generation'] == position and l['initial']
            assert r['arrival_time_ms'] == l['arrival_time'] == l['arrival_ms'] == 0
            assert l['seq_len_k'] == p['K'] and l['nql'] == p['miss'] and l['total_tokens'] == p['K']*1024
            assert l['ssd_prefix_tokens'] == p['K']*1024-p['miss']
            assert l['constructed_profile'] and l['source_ttft_ms'] is None and l['padding_gib_per_layer'] == 0
            assert l['profile_construction'] == p['source']['construction']
            assert l['category'] == ('LL' if cr == 'L' else 'SS')
            near(l['per_layer_us'],p['C']); near(l['original_compute_us'],p['C'])
            near(l['per_layer_kv_gb'],p['V']); near(l['required_bw_input_gbps'],p['B'])
            pi = r['placement_index']; cache_key = pi,npu%8,idx
            if cache_key not in checked:
                assert places[pi] == (tuple(((j+npu)%8,BLOCK) for j in range(p['blocks'])),)
                checked.add(cache_key)
            fingerprint.update(f"({rid}, {npu}, {float(r['arrival_time_ms'])!r}, {l['category']!r}, {l['per_layer_us']!r}, {representations[pi]})".encode())
            logical.append(dict(request_id=rid,npu_id=npu,arrival_time_ms=r['arrival_time_ms'],load=l))
            lane_coarse.append([rid,orig,cr]); observed[p['role']] += 1
            C_values.append(l['per_layer_us']); V_values.append(l['per_layer_kv_gb']); fine.append(p['role']); blocks += 8*p['blocks']
        assignment = m['per_npu_assignment'][npu]
        assert assignment['npu_id'] == npu and assignment['shuffle_seed'] == seed+100003*npu
        assert dict(observed) == assignment['role_counts'] == role_counts
        assert assignment['coarse_role_counts'] == coarse_counts and assignment['requests'] == NL+NS
        assert assignment['first_32_roles'] == fine[:32]
        assert assignment['first_32_coarse_roles'] == [r[2] for r in lane_coarse[:32]]
        Csum = math.fsum(C_values); Vsum = math.fsum(V_values)
        near(Csum,total_C); near(Vsum,total_V); near(8*Csum/1000,assignment['pure_compute_ms'])
        assert 8*Csum/1000 >= horizon
        totals.append([Csum,Vsum]); coarse.append(lane_coarse)
    ch = hashlib.sha256(json.dumps(coarse,separators=(',',':')).encode()).hexdigest()
    assert ch == m['coarse_input_sha256'] == record['coarse_input_sha256']
    if horizon == 22000 and str(seed) in spec['coarse_input_sha256_by_seed']:
        assert ch == spec['coarse_input_sha256_by_seed'][str(seed)]
    fp = fingerprint.hexdigest()
    assert fp == manifest['input_fingerprint'] == m['input_fingerprint'] == record['input_fingerprint']
    lf = hashlib.sha256(json.dumps(logical,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert lf == m['logical_input_fingerprint']
    assert blocks == record['expected_blocks'] == expected_blocks
    for Csum,Vsum in totals:
        near(Csum,m['per_card_per_layer_total_compute_us']); near(Vsum,m['per_card_per_layer_total_read_GiB'])
    near(total_C,record['per_card_per_layer_total_compute_us']); near(total_V,record['per_card_per_layer_total_read_GiB'])
    rho = 32*total_V*1e6/(320*total_C)
    near(rho,m['ideal_load_ratio']); near(rho,spec['rho_ideal']); near(rho,record['ideal_load_ratio'])
    near(math.fsum(m['time_weighted_per_ssu_nominal_gib_s']),rho*320)
    assert sha(path) == manifest_hash
    return dict(candidate=name,seed=seed,horizon_ms=horizon,manifest=str(path.relative_to(ROOT)),manifest_sha256=manifest_hash,
        requests=len(logical),input_fingerprint=fp,logical_input_fingerprint=lf,coarse_input_sha256=ch,
        role_counts=role_counts,all_32_per_card_totals=totals,pure_compute_ms=8*total_C/1000,
        rho=rho,expected_blocks=blocks,checks=dict(all_source_hashes=True,all_C_V_independently_recomputed=True,
            all_full_identity_shuffles=True,all_subtypes_mapped_by_original_identity=True,
            all_fingerprints=True,all_TTFT_none=True,all_exact_stripes=True,
            preexisting_coarse_seed_hash_verified=horizon==22000 and str(seed) in spec['coarse_input_sha256_by_seed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='hetero_S10_12_14_L384')
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--horizon-ms', type=float, default=22000)
    parser.add_argument('--prepared', type=Path, default=HERE/'heterogeneity_S10_12_14_prepared_4seeds.jsonl')
    parser.add_argument('--output', type=Path, default=HERE/'heterogeneity_confirmation_input_audit.json')
    args = parser.parse_args(); seeds = args.seed or [19,43,67,101]
    assert len(seeds)==len(set(seeds)) and math.isfinite(args.horizon_ms) and args.horizon_ms>=22000
    plan_path = HERE/'heterogeneity_math.json'; plan = read(plan_path); plan_hash = sha(plan_path)
    spec = next(p for p in plan['selected_candidates'] if p['name']==args.name)
    protected = [plan_path,HERE/'heterogeneity_math.py',HERE/'prepare_heterogeneity.py',HERE/'audit_heterogeneity_inputs.py',
        HERE/'experiment.py',HERE/'long_scale_math.json',HERE/'constructed_candidates.json',args.prepared]
    before = {str(p.resolve().relative_to(ROOT)):sha(p) for p in protected}
    core_hashes = read(HERE/'study_plan.json')['core_hashes']
    assert all(sha(ROOT/name)==h for name,h in core_hashes.items())
    raw = ast.literal_eval((ROOT/'data').read_text()); fits = {m:fit(raw,m) for m in (128,1024)}
    records = [json.loads(line) for line in args.prepared.read_text().splitlines() if line.strip()]
    by_label = {r['label']:r for r in records}; assert len(by_label)==len(records)
    expected = [f'{args.name}_ssu8_h{args.horizon_ms:g}_seed{seed}' for seed in seeds]
    assert set(expected)<=set(by_label)
    cases = []
    for seed,label in zip(seeds,expected):
        cases.append(audit(by_label[label],spec,seed,args.horizon_ms,plan_path,plan_hash,fits,core_hashes))
        print(json.dumps(dict(seed=seed,requests=cases[-1]['requests'],passed=True)),flush=True)
    assert all(case['all_32_per_card_totals']==cases[0]['all_32_per_card_totals'] for case in cases)
    assert before=={str(p.resolve().relative_to(ROOT)):sha(p) for p in protected}
    assert all(sha(ROOT/name)==h for name,h in core_hashes.items())
    out = dict(all_checks_passed=True,no_builder_imported=True,no_simulation_run=True,cases=cases,
        selected_candidate=args.name,selected_seeds=seeds,total_requests=sum(c['requests'] for c in cases),
        all_selected_seeds_same_C_V_counts=True,plan_sha256=plan_hash,prepared_jsonl_sha256=sha(args.prepared),
        protected_files_sha256=before,source_core_sha256=core_hashes,generator_sha256=sha(Path(__file__)),
        paired_control_scope='No new S12 control simulation or input is required. Frozen per-seed coarse hashes and independently derived original identities define the same matched coarse order; all C/V totals are recomputed.',
        caveat='Input balance and full shuffle do not guarantee every card computes both classes in each actual window. Actual coverage remains a post-run scientific condition; no seed is excluded here.')
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(all_checks_passed=True,cases=len(cases),requests=out['total_requests'],output=str(args.output))),flush=True)


if __name__ == '__main__':
    main()
