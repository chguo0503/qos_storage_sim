#!/usr/bin/env python3
"""Summarize completed four-second pilots and freeze reviewable candidates."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE.parent),str(ROOT)]
import run_pilot
from inputs.runners.run_baseline_npu32_stress import load_manifest,save_manifest
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint

def metadata_for(requests,meta,label,spec):
    profiles={}
    for role in ('A','B'):
        q=next(q for q in requests if q.load['role']==role);p=dict(meta['profiles'][role])
        p.update(per_layer_compute_us=q.load['per_layer_us'],required_bandwidth_gibps=q.load['required_bw_input_gbps'],
                 source_equivalent_ttft_78_layers_ms=78*q.load['per_layer_us']/1000,
                 construction=q.load['profile_construction'])
        profiles[role]=p
    cards=[]
    for n in range(32):
        qs=[q for q in requests if q.npu_id==n]
        rates=[[sum(v for d,v in q.placement[0]if d==s)*1e6/q.load['per_layer_us']for s in range(3)]for q in qs]
        cards.append(dict(npu_id=n,role_counts=dict(Counter(q.load['role']for q in qs)),request_count=len(qs),
            pure_compute_ms=sum(8*q.load['per_layer_us']/1000 for q in qs),per_ssu_rate_max=[max(r[s]for r in rates)for s in range(3)],queue=''.join(q.load['role']for q in qs)))
    blocks=sum(8*len(q.placement[0])for q in requests)
    return dict(meta,label=label,case_id=label,candidate_spec=spec,profiles=profiles,per_npu_assignment=cards,
       request_count=len(requests),blocks=blocks,expected_blocks=blocks,
       input_fingerprint=continuous_batch_input_fingerprint(requests),logical_input_fingerprint=logical_input_fingerprint(requests),
       static_per_ssu_upper_bound_GiB_s=[sum(c['per_ssu_rate_max'][s]for c in cards)for s in range(3)])

def main():
    plan=json.loads((HERE/'plan.json').read_text());rows=[]
    for spec in plan['candidates']:
        f=HERE/'pilots'/spec['label']/'command.json';r=json.loads(f.read_text());assert r['status']=='complete_pilot';m=r['measurement']
        valid=m['demand']['strict_underload_all_disks']and m['initial_4s_demand']['strict_underload_all_disks']and m['mixed_cards']==32 and m['all_npus_active']
        rows.append(dict(label=spec['label'],a_run=spec['a_run'],C_A_ms=spec['C_A_ms'],C_B_ms=spec['C_B_ms'],U_percent=m['U_percent'],
          warm_under=m['demand']['strict_underload_all_disks'],initial_4s_under=m['initial_4s_demand']['strict_underload_all_disks'],
          mixed_cards=m['mixed_cards'],all_active=m['all_npus_active'],valid=valid,
          max_initial_demand=max(m['initial_4s_demand']['per_disk_max_GiB_s']),
          min_A_compute_ms=min(c['A']for c in m['per_npu_role_compute_ms']),min_B_compute_ms=min(c['B']for c in m['per_npu_role_compute_ms']),
          wall_seconds=r['wall_seconds'],source_unchanged=r['source_unchanged']))
    rows.sort(key=lambda r:r['U_percent'])
    with (HERE/'pilot_summary.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    valid=[r for r in rows if r['valid']];best=valid[0];spec=next(s for s in plan['candidates']if s['label']==best['label'])
    base,meta=load_manifest(HERE/plan['base_manifest']);requests,meta=run_pilot.prepare(base,meta,spec)
    exact_label='frozen_exact_'+spec['label'];exact_path=HERE/'inputs'/f'{exact_label}.json.gz'
    save_manifest(exact_path,requests,metadata_for(requests,meta,exact_label,spec))
    trimmed=[];removed=[]
    for n in range(32):
        qs=[q for q in requests if q.npu_id==n];prefix=0
        while qs[prefix].load['role']=='B':prefix+=1
        limit=prefix+6*(spec['a_run']+3);trimmed.extend(qs[:limit]);removed.extend(qs[limit:])
        assert all(q.load['role']=='A'for q in qs[limit:])
    trimmed=tuple(trimmed);trim_label='frozen_6cycles_'+spec['label'];trim_path=HERE/'inputs'/f'{trim_label}.json.gz'
    trim_meta=metadata_for(trimmed,meta,trim_label,spec)
    trim_meta['tail_adjustment']=dict(removed_count=len(removed),rule='Remove only A tail beyond six complete A^m B^3 cycles; prefix through complete six cycles is unchanged',
        exact_pilot_manifest_sha256=hashlib.sha256(exact_path.read_bytes()).hexdigest(),same_population_as_pilot=not removed,
        caveat='If removals exist, this full population has not yet been simulated; its unchanged prefix covers the measured pilot.')
    save_manifest(trim_path,trimmed,trim_meta)
    result=dict(pilot_only=True,window_ms=[2000,4000],stop_ms=4000,TTFT_SLO=None,
        caveat='Four-second original-simulator screening; no long-run or drained-population result is claimed',
        best=best,all_candidates=rows,manifest_exact=str(exact_path.relative_to(HERE.parent)),manifest_6cycles=str(trim_path.relative_to(HERE.parent)),tail_removed=len(removed))
    (HERE/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['best']),flush=True)

if __name__=='__main__':main()
