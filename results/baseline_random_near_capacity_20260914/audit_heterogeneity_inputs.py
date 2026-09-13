#!/usr/bin/env python3
"""Independently audit paired heterogeneity inputs; never import their builder."""
import ast
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
from audit_context_inputs import fit_ms,near,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BLOCK=176/1048576


def main():
    plan_path=HERE/'heterogeneity_math.json';plan=json.loads(plan_path.read_text());ph=sha(plan_path)
    records_path=HERE/'heterogeneity_prepared_seed7.jsonl'
    records=[json.loads(s) for s in records_path.read_text().splitlines() if s.strip()]
    assert len(records)==3
    candidates={c['name']:c for c in plan['selected_candidates']}
    protected=[plan_path,HERE/'heterogeneity_math.py',HERE/'prepare_heterogeneity.py',HERE/'experiment.py',HERE/'long_scale_math.json',HERE/'constructed_candidates.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw=ast.literal_eval((ROOT/'data').read_text());fits={m:fit_ms(raw,m) for m in (128,1024)};dh=sha(ROOT/'data')
    audits=[];coarse_by_name={};source_hashes={}
    for record in records:
        path=Path(record['manifest']);mh=sha(path);assert mh==record['manifest_sha256']
        with gzip.open(path,'rt') as stream:man=json.load(stream)
        m=man['metadata'];name=m['candidate'];spec=candidates[name]
        label=f'{name}_ssu8_h22000_seed7'
        assert m['label']==record['label']==label and path.name==label+'.json.gz'
        assert m['num_npu']==32 and m['num_ssu']==8 and m['n_layers']==8 and m['seed']==7
        assert m['disk_bw_gib_s']==40 and m['npu_bw_gib_s']==50
        assert m['family']=='data_affine_heterogeneity' and m['plan_family']==spec['family']=='data_affine_heterogeneity_control'
        assert m['constructed_plan_sha256']==ph and m['constructed_builder_sha256']==sha(HERE/'prepare_heterogeneity.py')
        assert m['frozen_experiment_sha256']==sha(HERE/'experiment.py') and m['source_data_sha256']==dh
        assert m['order']==m['order_mode']=='random' and m['horizon_pure_compute_ms']==22000
        assert m['measurement_window_ms']==[2000,4000] and m['last_arrival_ms']==0
        assert m['blocks']=='exact' and m['layout']=='stripe_npu_mod_ssu' and m['equal_176kib_blocks']
        assert m['short_roles']==spec['short_roles'] and m['long_roles']==spec['long_roles']
        assert m['constructed_profile'] and m['constructed_long_profile'] and m['constructed_short_profile']
        for file,h in m['core_source_sha256'].items():
            source_hashes.setdefault(file,sha(ROOT/file));assert source_hashes[file]==h,file
        profiles=[]
        for i,(p,z) in enumerate(zip(spec['profiles'],m['profiles'])):
            a,b=fits[p['nql']];C=(a+b*p['total_k'])*1000
            count=(p['total_k']*1024-p['nql'])//128;V=count*BLOCK;B=V*1e6/C
            near(C,p['compute_us']);near(V,p['V_GiB']);near(C,z['per_layer_compute_us']);near(V,z['per_layer_kv_gib']);near(B,z['required_bandwidth_gibps'])
            assert z['role']==p['role'] and z['coarse_role']==p['coarse_role']
            assert z['seq_len_k']==p['total_k'] and z['nql']==p['nql'] and B<50
            assert z['source_equivalent_ttft_78_layers_ms'] is None and z['construction']['measured_data_row'] is False
            method='affine_extrapolation_below_raw_minimum' if p['total_k']<32 else 'affine_extrapolation_above_raw_maximum'
            assert z['construction']['method']==method and z['construction']['candidate_plan_sha256']==ph
            model=z['construction']['compute_model'];assert model==p['model_provenance']['compute_model']
            assert model['frozen_source_sha256']==sha(HERE/model['frozen_source'])
            near(z['extrapolated_78_layer_pure_compute_ms'],78*C/1000)
            profiles.append(dict(role=p['role'],coarse=p['coarse_role'],K=p['total_k'],miss=p['nql'],C=C,V=V,B=B,blocks=count,quota=p['count_per_npu'],source=z))
        assert len(profiles)==len(spec['profiles'])==len(m['profiles'])
        counts=[p['quota'] for p in profiles];gcd=math.gcd(*counts)
        assert m['role_names']==[p['role'] for p in profiles] and m['count_ratio']==[n//gcd for n in counts] and m['quota_cycles']==gcd
        assert m['coarse_count_ratio']==spec['coarse_counts'] and m['coarse_quota_cycles']==spec['minimum_22s_population']['repeats']
        pop=spec['minimum_22s_population'];NL,NS=pop['L'],pop['S'];N=NL+NS
        assert len(man['requests'])==m['request_count']==record['requests']==N*32
        by_coarse={r:[i for i,p in enumerate(profiles) if p['coarse']==r] for r in ['L','S']}
        lanes=defaultdict(list)
        for r in man['requests']:lanes[r['npu_id']].append(r)
        assert set(lanes)==set(range(32))
        places={i:tuple(tuple((int(s),float(v)) for s,v in layer) for layer in item) for i,item in enumerate(man['placements'])}
        reprs={i:repr(p) for i,p in places.items()};checked=set();coarse=[];logical=[];totals=[];blocks=0
        fingerprint=hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
        for npu in range(32):
            ordered=sorted(lanes[npu],key=lambda r:r['request_id']);ids=list(range(N));random.Random(7+100003*npu).shuffle(ids)
            assert len(ordered)==N
            lane_coarse=[];seen=Counter();seenC=[];seenV=[];fine=[]
            for position,(r,original) in enumerate(zip(ordered,ids)):
                cr='L' if original<NL else 'S';sub=(original if cr=='L' else original-NL)%len(by_coarse[cr]);idx=by_coarse[cr][sub];p=profiles[idx]
                l=r['load'];rid=npu*1000000+position;orig=npu*1000000+original
                assert r['request_id']==l['request_id']==rid and r['npu_id']==l['npu_id']==npu
                assert l['original_request_id']==orig and l['profile_index']==idx and l['coarse_role']==cr and l['role']==p['role']
                assert l['generation']==position and l['initial'] and r['arrival_time_ms']==l['arrival_time']==l['arrival_ms']==0.
                assert l['seq_len_k']==p['K'] and l['nql']==p['miss'] and l['total_tokens']==p['K']*1024
                assert l['ssd_prefix_tokens']==p['K']*1024-p['miss']
                assert l['constructed_profile'] and l['source_ttft_ms'] is None and l['padding_gib_per_layer']==0
                assert l['profile_construction']==p['source']['construction']
                assert l['category']==('LL' if cr=='L' else 'SS')
                near(l['per_layer_us'],p['C']);near(l['original_compute_us'],p['C']);near(l['per_layer_kv_gb'],p['V']);near(l['required_bw_input_gbps'],p['B'])
                pi=r['placement_index'];key=pi,npu%8,idx
                if key not in checked:
                    assert places[pi]==(tuple(((j+npu)%8,BLOCK) for j in range(p['blocks'])),);checked.add(key)
                fingerprint.update(f"({rid}, {npu}, {float(r['arrival_time_ms'])!r}, {l['category']!r}, {l['per_layer_us']!r}, {reprs[pi]})".encode())
                logical.append(dict(request_id=rid,npu_id=npu,arrival_time_ms=r['arrival_time_ms'],load=l))
                lane_coarse.append([rid,orig,cr]);seen[p['role']]+=1;seenC.append(p['C']);seenV.append(p['V']);fine.append(p['role']);blocks+=8*p['blocks']
            assignment=m['per_npu_assignment'][npu];assert assignment['shuffle_seed']==7+100003*npu
            assert dict(seen)==dict(zip([p['role'] for p in profiles],counts))==assignment['role_counts']
            assert assignment['coarse_role_counts']=={'L':NL,'S':NS}
            assert assignment['first_32_roles']==fine[:32] and assignment['first_32_coarse_roles']==[r[2] for r in lane_coarse[:32]]
            Csum=math.fsum(seenC);Vsum=math.fsum(seenV)
            near(Csum,pop['per_layer_C_work_us_per_card']);near(Vsum,pop['per_layer_V_work_GiB_per_card']);near(8*Csum/1000,assignment['pure_compute_ms'])
            assert 8*Csum/1000>=22000
            totals.append((Csum,Vsum));coarse.append(lane_coarse)
        ch=hashlib.sha256(json.dumps(coarse,separators=(',',':')).encode()).hexdigest()
        assert ch==m['coarse_input_sha256']==record['coarse_input_sha256']==spec['coarse_input_sha256_by_seed']['7']
        fp=fingerprint.hexdigest();assert fp==man['input_fingerprint']==m['input_fingerprint']==record['input_fingerprint']
        lf=hashlib.sha256(json.dumps(logical,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert lf==m['logical_input_fingerprint']
        assert blocks==record['expected_blocks']==pop['expected_blocks_all_32']
        rho=32*totals[0][1]*1e6/(320*totals[0][0]);near(rho,m['ideal_load_ratio']);near(rho,spec['rho_ideal'])
        near(sum(m['time_weighted_per_ssu_nominal_gib_s']),rho*320)
        coarse_by_name[name]=coarse
        audits.append(dict(candidate=name,manifest=str(path.relative_to(ROOT)),manifest_sha256=mh,requests=len(logical),
            input_fingerprint=fp,logical_input_fingerprint=lf,coarse_input_sha256=ch,role_counts=record['role_counts'],
            all_32_per_card_totals=totals,pure_compute_ms=8*totals[0][0]/1000,rho=rho,expected_blocks=blocks,
            checks=dict(all_source_hashes=True,all_C_V_independently_recomputed=True,all_full_identity_shuffles=True,
                all_subtypes_mapped_by_original_identity=True,all_fingerprints=True,all_TTFT_none=True,all_exact_stripes=True)))
        assert sha(path)==mh
        print(json.dumps(dict(candidate=name,requests=len(logical),passed=True)),flush=True)
    s0,s1,l0=[a['candidate'] for a in audits]
    assert set([s0,s1,l0])==set(candidates)
    control='hetero_S12_control_L384';hetero='hetero_S10_12_14_L384';longname='hetero_L352_384_416_S10'
    assert coarse_by_name[control]==coarse_by_name[hetero]
    byname={a['candidate']:a for a in audits};assert byname[control]['all_32_per_card_totals']==byname[hetero]['all_32_per_card_totals']
    original=ROOT/plan['original_context384_manifest'];assert sha(original)==plan['original_context384_manifest_sha256']
    with gzip.open(original,'rt') as stream:old=json.load(stream)
    old_coarse=[[] for _ in range(32)];old_totals=[[] for _ in range(32)]
    for r in old['requests']:
        npu=r['npu_id'];l=r['load'];old_coarse[npu].append([r['request_id'],l['original_request_id'],l['role']]);old_totals[npu].append((l['per_layer_us'],l['per_layer_kv_gb']))
    for lane in old_coarse:lane.sort()
    assert old_coarse==coarse_by_name[longname]
    old_sums=[(math.fsum(v[0] for v in lane),math.fsum(v[1] for v in lane)) for lane in old_totals]
    assert old_sums==byname[longname]['all_32_per_card_totals']
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    out=dict(all_checks_passed=True,no_builder_imported=True,no_simulation_run=True,cases=audits,
        total_requests=sum(a['requests'] for a in audits),paired_coarse_streams_and_C_V_exactly_equal=True,
        original_context384_long_pair_equal=True,plan_sha256=ph,prepared_jsonl_sha256=sha(records_path),
        protected_files_sha256=before,source_core_sha256=source_hashes,generator_sha256=sha(Path(__file__)),
        caveat='Whole-input equality does not imply equal execution timing or warm population. Actual32-cardL/S coverage remains a post-run check.')
    (HERE/'heterogeneity_input_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(all_checks_passed=True,requests=out['total_requests'],pairs_equal=True)),flush=True)


if __name__=='__main__':main()
