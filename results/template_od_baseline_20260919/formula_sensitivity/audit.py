"""Independent post-run checks from raw intervals, bytes and saved populations."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.name.endswith('.gz') else json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    provenance=read(ROOT/'source_provenance.json')
    original_unchanged={p:sha(PROJECT/p)==v for p,v in provenance['archived_file_sha256'].items()}
    assert all(original_unchanged.values())
    source_unchanged={p:sha(ROOT/'frozen_source'/p)==v for p,v in provenance['frozen_source_sha256'].items()}
    assert all(source_unchanged.values())
    with gzip.open(ROOT/'cdf_samples.csv.gz','rt',encoding='utf-8-sig') as f:samples=list(csv.DictReader(f))
    cohorts=defaultdict(dict)
    for r in samples:cohorts[r['case_group']].setdefault(r['strategy'],set()).add((int(r['seed']),int(r['npu_id']),int(r['request_id'])))
    assert all(set(by)=={'asu_baseline','od_baseline','once'} and len({len(x) for x in by.values()})==1 and by['asu_baseline']==by['od_baseline']==by['once'] for by in cohorts.values())
    # Compare the original CDF sources, preserving each archive's arithmetic.
    template=PROJECT/'template/qos_experiments_20260919'
    formula_csv=template/'03_continuous_underload/formula_ab/ttft_slo_cdf_bundle/cdf_requests.csv.gz'
    with gzip.open(formula_csv,'rt',encoding='utf-8-sig') as f:original_formula=list(csv.DictReader(f))
    lookup={(r['case_group'],r['seed'],r['request_id'],r['strategy']):r for r in samples}
    formula_errors={k:0. for k in ['ttft_ms','compute_ms','ttft_ratio']}
    for old in original_formula:
        new=lookup[old['id'],old['seed'],old['request_id'],'asu_baseline' if old['strategy']=='baseline' else 'once']
        for k in formula_errors:formula_errors[k]=max(formula_errors[k],abs(float(old[k])-float(new[k])))
    assert max(formula_errors.values())==0.
    sensitivity_points=template/'02_partial_overload/sensitivity20k_076/original_data_code/qos_storage_sim/results/lower_fifo_followup_20260914/figures/data/ttft_cdf_points.csv'
    with sensitivity_points.open(encoding='utf-8-sig') as f:original_sensitivity=list(csv.DictReader(f))
    sensitivity_errors=[]
    for policy in ['fifo','once']:
        for role in ['all','L','S']:
            values=[float(r['ttft_ratio']) for r in samples if r['case_group']=='sensitivity20k_076'
                    and r['strategy']==('asu_baseline' if policy=='fifo' else 'once')
                    and (role=='all' or r['group']==('A' if role=='L' else 'B'))]
            unique=sorted(Counter(values).items());old=[r for r in original_sensitivity if r['policy']==policy and r['role']==role]
            assert len(unique)==len(old)
            cumulative=0;xerror=yerror=0.
            for (x,count),p in zip(unique,old):
                cumulative+=count
                xerror=max(xerror,abs(x-float(p['ratio'])))
                yerror=max(yerror,abs(cumulative*100/len(values)-float(p['cdf_percent'])))
            assert xerror==0. and yerror==0.
            sensitivity_errors.append(dict(policy=policy,role=role,points=len(old),max_x_error=xerror,max_y_error=yerror))
    threshold_consistency=all((float(r['ttft_ratio'])<=1.5)==(float(r['ttft_ratio'])<=1.5+1e-10) for r in samples)
    assert threshold_consistency
    cases=[]
    for job in read(ROOT/'jobs.json'):
        manifest=ROOT/'inputs'/f"{job['case']}.json.gz";m=read(manifest)
        assert sha(manifest)==job['manifest_sha256']
        dest=ROOT/'runs'/f"{job['case']}_od_baseline";raw=read(dest/'native_summary.json.gz');reported=read(dest/'metrics.json')
        qs={q['request_id']:q for q in m['requests']}
        assert raw['input_fingerprint']==job['input_fingerprint']==m['input_fingerprint']
        assert {r['request_id'] for r in raw['request_metrics']}==set(qs)
        expected_blocks=Counter();expected_gib=0.
        for q in qs.values():
            layer=m['placements'][q['placement_index']][0]
            expected_blocks[q['npu_id']]+=8*len(layer)
            expected_gib+=8*math.fsum(v for disk,v in layer)
        observed=Counter()
        for r in reported['observed_paths']:
            assert r['ssu']==0 and r['path']==r['npu']*32
            observed[r['npu']]+=r['blocks']
        assert observed==expected_blocks
        assert {r['npu'] for r in reported['observed_layer0_paths']}==set(range(8))
        assert raw['completed_blocks']==sum(expected_blocks.values())
        assert abs(raw['completed_read_gb']-expected_gib)<1e-7
        assert math.isclose(reported['cir_sum'],reported['disk_bw_gib_s'],abs_tol=1e-10)
        assert len(reported['active_path_cirs'])==8
        assert all(math.isclose(v,reported['disk_bw_gib_s']/8,abs_tol=1e-12) for v in reported['active_path_cirs'].values())
        assert reported['pir_unlimited'] and all(raw['invariants'].values())
        compute=math.fsum(max(0.,min(l['compute_end_ms'],4000.)-max(l['compute_start_ms'],2000.))
                         for b in raw['microbatch_metrics'] for l in b['layer_metrics'])
        u=100*compute/(8*2000)
        assert abs(u-reported['warm_U_percent'])<1e-10
        allcount=len(raw['request_metrics']);passed=0;warmcount=0;warmpassed=0
        for r in raw['request_metrics']:
            good=r['completion_time_ms']-r['admission_time_ms']<=1.5*r['own_compute_ms']+1e-9
            passed+=good
            if 2000<=r['admission_time_ms']<4000:warmcount+=1;warmpassed+=good
        assert (allcount,passed)==(reported['cohorts']['all_requests']['all']['count'],reported['cohorts']['all_requests']['all']['slo1.5_passed'])
        assert (warmcount,warmpassed)==(reported['cohorts']['warm_admissions']['all']['count'],reported['cohorts']['warm_admissions']['all']['slo1.5_passed'])
        cases.append(dict(case=job['case'],requests=allcount,blocks=raw['completed_blocks'],
                          U_warm_percent=u,U_warm_error=abs(u-reported['warm_U_percent']),
                          SLO_full_passed=passed,SLO_warm_passed=warmpassed,SLO_warm_count=warmcount,
                          completed_volume_error_gib=raw['completed_read_gb']-expected_gib,
                          path_ownership_all_blocks_verified=True,layer0_ownership_verified=True,
                          source_and_input_unchanged=True))
    parity={}
    for p in (ROOT/'runs').glob('*_asu_baseline/parity.json'):
        checks=read(p);assert all(checks.values());parity[p.parent.name]=checks
    assert len(parity)==2 and len(cases)==16
    out=dict(passed=True,case_count=16,source_file_count=len(source_unchanged),
             original_file_count=len(original_unchanged),original_files_unchanged=True,
             exact_asu_parity=parity,cases=cases,
             original_cdf_comparison={'formula_sample_count':len(original_formula),'formula_max_errors':formula_errors,
                                      'sensitivity_all_input_count':392,'sensitivity_points':sensitivity_errors,
                                      'x_1p5_cdf_equals_reported_SLO_no_epsilon_edge_cases':threshold_consistency},
             matching_cdf_populations={g:len(s['od_baseline']) for g,s in cohorts.items()},
             completed_figures={p.name:sha(p) for p in (ROOT/'figures').glob('*.png')})
    (ROOT/'verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['cases','completed_figures']}))

if __name__=='__main__':main()
