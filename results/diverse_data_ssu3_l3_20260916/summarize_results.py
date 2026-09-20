#!/usr/bin/env python3
"""Verify complete cases and keep exact warm previews explicitly separate."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from inputs.runners.run_baseline_npu32_stress import load_manifest
from metrics import summarize


def read(p):
    b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def flatten(c,w,window):
    r=dict(scenario=c['scenario'],seed=c['seed'],policy=c['policy'],window=window,
        U_percent=w['U_percent'],slo_percent=w['slo']['percent'],
        slo_count=w['slo']['count'],slo_passed=w['slo']['passed'],
        arrival_slo_percent=w['arrival_slo']['percent'],
        timely_completions_per_second=w['timely_completions_per_second'],
        total_completions_per_second=w['total_completions_per_second'],
        all_npus_active=w['all_npus_active'],
        all_disks_overload_percent=w['demand']['all_disks_overload_percent'])
    for d in range(3):
        r[f'SSU{d}_mean_demand_GiB_s']=w['demand']['per_disk_mean_GiB_s'][d]
        r[f'SSU{d}_overload_percent']=w['demand']['per_disk_overload_percent'][d]
        r[f'SSU{d}_actual_GiB_s']=w.get('SSD_GiB_s',[None]*3)[d]
    for cat,s in w['slo_by_category'].items():
        r[f'{cat}_slo_percent']=s['percent'];r[f'{cat}_slo_count']=s['count'];r[f'{cat}_slo_passed']=s['passed']
        r[f'{cat}_active_U_percent']=w['category_active_U_percent'].get(cat)
    return r


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def main():
    plan=read(HERE/'formal_plan.json')
    complete=[];previews=[];profiles=[];sources={};cohorts={}
    for job in plan['jobs']:
        p=HERE/'runs'/job['label'];cp=p/'command.json'
        if not cp.exists():continue
        c=read(cp)
        assert not c['pilot'] and not c['smoke']
        manifest=p/'manifest.json.gz'
        assert sha(manifest)==c['manifest_sha256']==plan['inputs'][
            f'results/diverse_data_ssu3_l3_20260916/inputs/{c["scenario"]}_seed{c["seed"]}.json.gz']
        if c['status']=='complete':
            assert c['core_unchanged'] and c['extension_unchanged'] and all(c['checks'].values())
            assert c['observed_blocks']==c['expected_blocks']
            for name,digest in c['extension_source_sha256'].items():assert sha(HERE/name)==digest
            for name,digest in c['core_source_sha256'].items():assert sha(ROOT/name)==digest
            rp=p/'result.json.gz';assert sha(rp)==c['result_sha256']
            raw=read(rp);requests,meta=load_manifest(manifest)
            assert len(raw['summary']['request_metrics'])==len(requests)==c['completed_requests']
            assert all(raw['summary']['invariants'].values())
            assert raw['adapter_statistics']['reorder_calls']==0 and raw['adapter_statistics']['cir_write_events']==[]
            for index,w in enumerate(raw['analysis']):
                name=['warm_2_4s','long_2_6s','full_population'][index]
                # Recalculate from raw timings; fail instead of trusting tables.
                measured=summarize(raw['summary'],requests,w['start_ms'],w['end_ms'],full=index==2)
                assert abs(w['U_percent']-measured['U_percent'])<1e-7
                assert w['slo']==measured['slo'] and w['demand']==measured['demand']
                r=flatten(c,w,name);r.update(makespan_ms=raw['summary']['makespan_ms'],
                    manifest_sha256=c['manifest_sha256'],result_sha256=c['result_sha256'])
                complete.append(r)
                cohorts[c['scenario'],c['seed'],c['policy'],name]=w['cohort_request_ids']
                for profile,s in w['slo_by_profile'].items():
                    profiles.append(dict(scenario=c['scenario'],seed=c['seed'],policy=c['policy'],
                        window=name,profile=profile,**s))
            sources[str(rp.relative_to(ROOT))]=sha(rp)
        elif c['status']=='running' and (p/'warm_preview.json').exists():
            previews.append(flatten(c,read(p/'warm_preview.json'),'warm_2_4s'))
        elif c['status'] not in ('running',):
            raise AssertionError((job['label'],c['status'],c.get('error')))
    groups=defaultdict(list)
    for r in complete:groups[r['scenario'],r['policy'],r['window']].append(r)
    macros=[]
    for (scenario,policy,window),part in sorted(groups.items()):
        m=dict(scenario=scenario,policy=policy,window=window,seed_count=len(part),seeds=[r['seed'] for r in part])
        for field in part[0]:
            if field.endswith(('_percent','_per_second','_GiB_s')) or field=='makespan_ms':
                vals=[r[field] for r in part if r[field] is not None]
                m['mean_'+field]=math.fsum(vals)/len(vals) if vals else None
                m['min_'+field]=min(vals) if vals else None;m['max_'+field]=max(vals) if vals else None
        m['pooled_slo_count']=sum(r['slo_count'] for r in part)
        m['pooled_slo_passed']=sum(r['slo_passed'] for r in part)
        macros.append(m)
    write_csv('comparison.csv',complete);write_csv('warm_previews.csv',previews)
    write_csv('macro_summary.csv',macros);write_csv('profile_slo.csv',profiles)
    out=dict(complete_cases=len(complete)//3,planned_cases=len(plan['jobs']),
        exact_warm_previews=len(previews),rows=complete,sources=sources)
    (HERE/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['complete_cases','planned_cases','exact_warm_previews']}))


if __name__=='__main__':main()
