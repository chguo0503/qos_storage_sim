#!/usr/bin/env python3
"""Freeze six Ring-hash inputs with the previous diverse study's exact queues."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'results/diverse_data_ssu3_l3_20260916'
sys.path.insert(0, str(ROOT))
from inputs.runners.run_baseline_npu32_stress import load_manifest, save_manifest
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from simulator.core import sim


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = importlib.util.spec_from_file_location('od_diverse_input_builder', OLD/'construct_manifest.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    (HERE/'inputs').mkdir(parents=True, exist_ok=True)
    audit = dict(placement=sim.PLACEMENT_BLOCK_RING_HASH, seeds=[7,19,43],
                 scenarios=['semi','full'], source_data_sha256=sha(ROOT/'data'),
                 builder_sha256=sha(OLD/'construct_manifest.py'), cases=[])
    for scenario in ('semi','full'):
        for seed in (7,19,43):
            requests, meta = builder.build_workload(scenario, seed, horizon_ms=6500.)
            historical_path = OLD/'inputs'/f'{scenario}_seed{seed}.json.gz'
            original_sha = sha(historical_path)
            original, original_meta = load_manifest(historical_path)
            logical = logical_input_fingerprint(requests)
            assert logical == logical_input_fingerprint(original) == original_meta['logical_input_fingerprint']
            assert meta['layout'] == sim.PLACEMENT_BLOCK_RING_HASH
            assert len(requests) == (960 if scenario == 'semi' else 1344)
            for request in requests:
                assert len(request.placement) == 1
                assert all(d == sim.block_ring_hash_disk_id(request.load['original_request_id'], j, 3)
                           for j, (d, _) in enumerate(request.placement[0]))
            target = HERE/'inputs'/f'{scenario}_seed{seed}_ring_hash.json.gz'
            if target.exists():
                _, previous_meta = load_manifest(target)
                assert previous_meta == meta, 'Preserve different existing input'
            save_manifest(target, requests, meta)
            assert sha(historical_path) == original_sha
            case = dict(scenario=scenario, seed=seed, input=str(target.relative_to(ROOT)),
                        manifest_sha256=sha(target), request_count=len(requests),
                        blocks=len(requests) and sum(len(q.placement[0])*8 for q in requests),
                        input_fingerprint=meta['input_fingerprint'], logical_input_fingerprint=logical,
                        same_requests_compute_bytes_and_order_as_previous=True,
                        previous_manifest_sha256=original_sha, previous_input_unchanged=True,
                        per_ssu_ideal_mean_GiB_s=meta['time_weighted_per_ssu_nominal_gib_s'],
                        fleet_ideal_mean_GiB_s=meta['time_weighted_fleet_nominal_gib_s'])
            audit['cases'].append(case)
            print(json.dumps(case), flush=True)
    (HERE/'input_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
