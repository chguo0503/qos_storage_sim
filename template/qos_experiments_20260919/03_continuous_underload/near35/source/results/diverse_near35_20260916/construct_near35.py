#!/usr/bin/env python3
"""Freeze raw-data near-35 GiB/s inputs using unchanged original constructor logic."""
from pathlib import Path
import sys, json, hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ORIGINAL=ROOT/'results/diverse_data_ssu3_l3_20260916'
sys.path[:0]=[str(ORIGINAL),str(ROOT)]
import construct_manifest as source

def main():
    # Only choose a new raw-data catalog and equal request counts in memory.
    # The on-disk original constructor and simulator stay byte-identical.
    source.LENGTHS=(32,64,80,128,160,200)
    source.MISSES=(256,1024,2048,4096)
    outputs=[]
    for seed in (7,19,43):
        requests,meta=source.build_workload('semi',seed,6500.0,weights=(1,1,3,2))
        peaks=meta['static_per_ssu_upper_bound_gib_s']
        assert all(p['construction']['method']=='direct_data_row' for p in meta['profiles'])
        assert meta['compute_scale_actual']==1.0
        meta.update(experiment='diverse_near35_20260916',scenario_candidate='near35',
            label=f'near35_seed{seed}',case_id=f"near35_seed{seed}_{meta['input_fingerprint'][:12]}",
            static_upper_bound_passes=all(v<40.0 for v in peaks),
            nominal_capacity_status='Average reference demand target near35 GiB/s per disk; instantaneous reference demand may exceed40; request-boundary prefetch is excluded from demand',
            selection_rule='Raw lengths32/64/80/128/160/200K x misses256/1024/2048/4096; per-length weights1/1/3/2; no scaling; pure-compute-weighted reference demand34.99491263 GiB/s per disk',
            source_constructor_sha256=hashlib.sha256((ORIGINAL/'construct_manifest.py').read_bytes()).hexdigest(),
            extension_constructor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        out=HERE/'inputs'/f'near35_seed{seed}.json.gz'
        if out.exists():raise FileExistsError(out)
        source.save_manifest(out,requests,meta)
        restored,again=source.load_manifest(out)
        assert again==meta and source.continuous_batch_input_fingerprint(restored)==meta['input_fingerprint']
        outputs.append(dict(seed=seed,manifest=str(out),request_count=len(requests),
            pure_compute_ms=meta['actual_pure_compute_ms_per_npu'],
            per_disk_upper_bound_GiB_s=peaks,ideal_fleet_GiB_s=meta['time_weighted_fleet_nominal_gib_s'],
            expected_blocks=sum(len(q.placement[0])*8 for q in requests)))
    (HERE/'input_audit.json').write_text(json.dumps(outputs,indent=2))
    print(json.dumps(outputs,indent=2))

if __name__=='__main__':main()
