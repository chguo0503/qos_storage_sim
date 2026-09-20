#!/usr/bin/env python3
"""Replay a previously frozen strict-underload candidate with the new OD quota.

No input, service policy, placement, or arrival is changed. Statistics are
computed after full drainage; the later windows test whether warm loss lasts.
"""
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from inputs.manifest import load_manifest, write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as native

OLD = ROOT / 'results/od_mixed_underload_search_20260919'
spec = importlib.util.spec_from_file_location('underload_measurements', OLD / 'metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def main():
    target = HERE / 'strict_replay'
    target.mkdir(parents=True, exist_ok=False)
    source = OLD / 'inputs/native_phase_lock_a101_b0995.json.gz'
    requests, metadata = load_manifest(source)
    (target / 'manifest.json.gz').write_bytes(source.read_bytes())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for base in ('simulator', 'inputs')
              for p in sorted((ROOT / base).rglob('*.py'))}
    windows = [(2000., 4000.), (4000., 8000.), (8000., 12000.), (12000., 16000.)]
    busy = [[0.] * 3 for _ in windows]
    observed = 0
    started = last = time.perf_counter()
    callback = native._register_complete

    def observe(ctx, flow):
        nonlocal observed, last
        observed += 1
        for i, (a, z) in enumerate(windows):
            busy[i][flow.disk_id] += metrics.overlap(
                flow.ssd_activation_time, flow.link_enqueue_time, a, z)
        result = callback(ctx, flow)
        if observed % 50000 == 0 and time.perf_counter() - last > 20:
            progress = dict(blocks=observed, simulation_ms=ctx.current_time_ms,
                            wall_seconds=time.perf_counter() - started)
            write_json(target / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            last = time.perf_counter()
        return result

    write_json(target / 'command.json', dict(
        status='running', manifest=str(source.relative_to(ROOT)), source_sha256=hashes,
        od_queue_depth_per_ssu=8192, seed=7, windows_ms=windows,
        interpretation='previous synthetic adversarial-address candidate; unchanged input'))
    with patch.object(native, '_register_complete', observe):
        result = run_simulation(requests, strategy='od_baseline', num_npu=32,
                                num_ssu=3, n_layers=8, seed=7,
                                od_queue_depth_per_ssu=8192)
    summary = result['summary']
    expected = sum(8 * len(q.placement[0]) for q in requests)
    assert observed == expected == summary['completed_blocks']
    assert all(summary['invariants'].values())
    analysis = []
    for index, (a, z) in enumerate(windows):
        row = metrics.summarize(summary, requests, a, z)
        row['physical_ssd_GiB_s'] = [x * 40 / (z - a) for x in busy[index]]
        analysis.append(row)
    result.update(analysis=analysis, metadata=metadata, source_sha256=hashes,
                  od_queue_depth_per_ssu=8192, wall_seconds=time.perf_counter() - started)
    write_json(target / 'result.json.gz', result)
    concise = [dict(window_ms=[r['start_ms'], r['end_ms']], U_percent=r['U_percent'],
                    slo=r['slo'], all_npus_active=r['all_npus_active'],
                    role_and_stall=r['role_and_stall'],
                    strict_underload=r['demand']['strict_underload_all_disks'],
                    per_disk_peak=r['demand']['per_disk_max_GiB_s'],
                    physical_ssd_GiB_s=r['physical_ssd_GiB_s']) for r in analysis]
    write_json(target / 'summary.json', concise)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
               for p, digest in hashes.items())
    write_json(target / 'command.json', dict(status='complete', expected_blocks=expected,
        source_sha256=hashes, source_unchanged=True, od_queue_depth_per_ssu=8192,
        seed=7, windows_ms=windows, wall_seconds=time.perf_counter() - started,
        input_fingerprint=result['input_fingerprint']))
    print(json.dumps([{k: v for k, v in row.items() if k != 'role_and_stall'}
                      for row in concise]), flush=True)


if __name__ == '__main__':
    main()
