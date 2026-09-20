#!/usr/bin/env python3
"""Summarize completed native jobs; approximate rankings are never results."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
E=ROOT/'results/random_multitype_search_20260914'

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def summarize():
    rows=[];group_rows=[];cases=[]
    for p in sorted(E.glob('*/*/metrics.json')):
        m=json.loads(p.read_text());meta=json.loads((p.parent/'metadata.json').read_text())
        if not (p.parent/'result.json.gz').exists():continue
        raw=json.load(gzip.open(p.parent/'result.json.gz','rt'))
        qs={q['request_id']:q['load'] for q in json.load(gzip.open(p.parent/'manifest.json.gz','rt'))['requests']}
        full={};left,right=m['window_ms']
        for b in raw['summary']['microbatch_metrics']:
            q=qs[b['member_request_ids'][0]];g=q['profile_group']
            z=full.setdefault(g,dict(count=0,passed=0,warm_count=0,warm_passed=0))
            ttft=b['completion_time_ms']-b['admission_time_ms'];ideal=q['per_layer_us']/1000*8
            passed=ttft<=1.5*ideal+1e-9;z['count']+=1;z['passed']+=passed
            if left<=b['admission_time_ms']<right:z['warm_count']+=1;z['warm_passed']+=passed
        audit_path=p.parent/'deadline_audit.json'
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        row={k:m[k] for k in ['name','seed','policy','num_ssu','U_percent','short_U_percent','long_U_percent','slo_1p5_percent','slo_passed','slo_count','all_active','all_npus_both_roles_computed']}
        row.update(stage=p.parent.parent.name,window_left_ms=left,window_right_ms=right,
            input_load_ratio=meta['input_demand']['aggregate_load_ratio'],input_mean_gib_s=meta['input_demand']['total_gib_s'],
            hottest_input_mean_gib_s=max(meta['input_demand']['per_ssu_gib_s']),
            max_nominal_static_per_ssu_gib_s=max(meta['per_ssu_static_upper_bound_gib_s']),
            nominal_static_underload=meta['static_underload_all_request_combinations'],
            all_request_slo_1p5_percent=100*sum(v['passed'] for v in full.values())/sum(v['count'] for v in full.values()),
            requests=len(qs),input_fingerprint=meta['input_fingerprint'],case=str(p.parent.relative_to(ROOT)))
        rows.append(row)
        for gid,v in m['by_profile_group'].items():
            f=full[gid];pr=meta['profiles'][gid]
            group_rows.append(dict(case=meta['case']['name'],stage=row['stage'],seed=m['seed'],policy=m['policy'],
                window_left_ms=left,window_right_ms=right,profile_group=gid,role=v['role'],
                total_k=pr['total_k'],base_nql=pr['nql'],weight=pr['weight'],nql_min=pr['nql_range'][0],nql_max=pr['nql_range'][1],
                U_percent=v['U_percent'],active_ms=v['active_ms'],compute_ms=v['compute_ms'],
                internal_stall_ms=v['internal_stall_ms'],l0_stall_ms=v['l0_stall_ms'],
                warm_slo_passed=f['warm_passed'],warm_slo_count=f['warm_count'],
                warm_slo_percent=100*f['warm_passed']/f['warm_count'] if f['warm_count'] else None,
                all_slo_passed=f['passed'],all_slo_count=f['count'],all_slo_percent=100*f['passed']/f['count']))
        cases.append(dict(summary=row,metadata=meta,by_group=m['by_profile_group'],deadline_audit=audit))
    group={}
    for r in rows:
        key=(r['name'],r['policy'],r['window_left_ms'],r['window_right_ms'])
        group.setdefault(key,[]).append(r)
    aggregate=[]
    for (name,policy,left,right),rs in sorted(group.items()):
        assert len({r['seed'] for r in rs})==len(rs)
        us=[r['U_percent'] for r in rs]
        aggregate.append(dict(name=name,policy=policy,window_left_ms=left,window_right_ms=right,n_seeds=len(rs),
            seeds=','.join(str(r['seed']) for r in sorted(rs,key=lambda r:r['seed'])),
            U_mean_percent=statistics.mean(us),U_min_percent=min(us),U_max_percent=max(us),
            input_load_mean=statistics.mean(r['input_load_ratio'] for r in rs),
            nominal_static_underload_all_seeds=all(r['nominal_static_underload'] for r in rs),
            all_npus_active_all_seeds=all(r['all_active'] for r in rs),
            all_npus_both_roles_computed_all_seeds=all(r['all_npus_both_roles_computed'] for r in rs)))
    for name,data in [('native_results',rows),('native_profile_results',group_rows),('native_seed_summary',aggregate)]:
        (E/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        csv_write(E/f'{name}.csv',data)
    (E/'native_full_evidence.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
    print(json.dumps(aggregate,ensure_ascii=False))
    return rows,aggregate

if __name__=='__main__':summarize()
