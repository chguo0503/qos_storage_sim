#!/usr/bin/env python3
"""Correct top-level metadata without changing any screened request or placement."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE),str(ROOT)]
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint
from inputs.runners.run_baseline_npu32_stress import load_manifest, read_json, save_manifest, write_json
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot',default='native_grid1_01')
    ap.add_argument('--label',default='native_phase_lock_a102')
    args=ap.parse_args()
    label=args.label
    assert all(Path(v).name==v and v not in ('.','..') for v in (label,args.pilot))
    source = HERE/'inputs'/f'{args.pilot}.json.gz'
    target = HERE/'inputs'/f'{label}.json.gz'
    assert not target.exists()
    requests, old = load_manifest(source)
    pilot = read_json(HERE/'pilots'/args.pilot/'command.json')
    assert pilot['status']=='complete_pilot' and pilot['source_unchanged']
    fingerprint = continuous_batch_input_fingerprint(requests)
    assert fingerprint == pilot['input_fingerprint'] == old['input_fingerprint']
    spec = dict(pilot['spec'],label=label,promoted_from_pilot=args.pilot,
                preparation='run_pilot.py --freeze-only, followed by top-level metadata correction',
                base_manifest='results/od_mixed_underload_search_20260919/inputs/fixedssu1_safe38.json.gz',
                base_manifest_sha256=pilot['base_sha256'])
    profiles = {}
    for role in ('A','B'):
        group = [q for q in requests if q.load['role']==role]
        computes = {q.load['per_layer_us'] for q in group}
        volumes = {q.load['per_layer_kv_gb'] for q in group}
        assert len(computes)==len(volumes)==1
        C, V = next(iter(computes)), next(iter(volumes))
        assert abs(C/1000-pilot['spec'][f'C_{role}_ms'])<1e-12
        row = dict(old['profiles'][role])
        row.update(per_layer_compute_us=C,per_layer_kv_gib=V,
                   required_bandwidth_gibps=V*1e6/C,
                   source_equivalent_ttft_78_layers_ms=78*C/1000,
                   ideal_ttft_8_layers_ms=8*C/1000,
                   base_profile_compute_us=row['per_layer_compute_us'],
                   constructed_profile=True,
                   construction=dict(method='original_simulator_parameter_search',
                       C_ms=C/1000,parameters=spec,
                       note='Synthetic compute; 8-layer ideal is authoritative. Historical load.source_ttft_ms remains the base-profile reference.'))
        profiles[role]=row
    assignment=[]
    for n in range(32):
        group=[q for q in requests if q.npu_id==n]
        rates=[[math.fsum(v for d,v in q.placement[0] if d==s)*1e6/q.load['per_layer_us'] for s in range(3)] for q in group]
        assignment.append(dict(npu_id=n,role_counts=dict(Counter(q.load['role'] for q in group)),
            request_count=len(group),pure_compute_ms=math.fsum(8*q.load['per_layer_us']/1000 for q in group),
            per_ssu_rate_max=[max(row[s] for row in rates) for s in range(3)],
            queue=''.join(q.load['role'] for q in group)))
    meta=dict(old,label=label,case_id=label,candidate_spec=spec,profiles=profiles,
        per_npu_assignment=assignment,
        static_per_ssu_upper_bound_GiB_s=[math.fsum(a['per_ssu_rate_max'][s] for a in assignment) for s in range(3)],
        request_count=len(requests),expected_blocks=sum(8*len(q.placement[0]) for q in requests),
        input_fingerprint=fingerprint,logical_input_fingerprint=logical_input_fingerprint(requests),
        metadata_correction=dict(source_manifest=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            request_load_and_placement_unchanged=True,
            request_load_source_ttft_ms='Preserved base-profile reference; not this candidate ideal or SLO denominator',
            corrected_fields=['profiles','per_npu_assignment','candidate_spec','static_per_ssu_upper_bound_GiB_s']))
    save_manifest(target,requests,meta)
    restored,actual=load_manifest(target)
    assert actual==meta and restored==requests
    assert continuous_batch_input_fingerprint(restored)==pilot['input_fingerprint']
    audit=dict(manifest=str(target.relative_to(ROOT)),manifest_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        pilot_input_fingerprint=pilot['input_fingerprint'],formal_input_fingerprint=fingerprint,
        request_inputs_identical_to_pilot=True,metadata_matches_actual_loads=True,
        pure_compute_ms_by_npu=[a['pure_compute_ms'] for a in assignment],
        requests=len(requests),blocks=meta['expected_blocks'],
        role_compute_ms={role:row['per_layer_compute_us']/1000 for role,row in profiles.items()})
    write_json(HERE/f'{label}_metadata_audit.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    main()
