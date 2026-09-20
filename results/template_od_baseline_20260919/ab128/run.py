#!/usr/bin/env python3
"""Replay archived A/B manifests without regenerating order or placement."""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from inputs.manifest import load_manifest, read_json, write_json
from inputs.provenance import source_files
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as native
from simulator.policies.od_baseline import od_npu_path_ids
from inputs.runners.run_baseline_npu32_stress import window_evidence

ARCHIVE = ROOT / 'template/qos_experiments_20260919/repository_source/results/baseline_ab128_32_ratio12_20260912'
ORIGINAL = ROOT / 'results/baseline_ab128_32_ratio12_20260912'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--order', choices=('random', 'ordered'), required=True)
    ap.add_argument('--strategy', choices=('asu_baseline', 'od_baseline'), required=True)
    args = ap.parse_args()
    relative = Path('validation20s/runs') / f'ssu3_{args.order}_k1_sync_seed7/baseline'
    source = ARCHIVE / relative / 'manifest.json.gz'
    directory = HERE / 'runs' / f'{args.order}_{args.strategy}'
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, directory / 'manifest.json.gz')
    requests, meta = load_manifest(source)
    assert len(requests) == 3840
    assert (meta['num_npu'], meta['num_ssu'], meta['n_layers'], meta['seed']) == (32, 3, 8, 7)
    assert meta['layout'] == 'stripe_npu_mod_ssu'
    source_hashes = {p: sha(ROOT / p) for p in source_files()}
    record = dict(status='running', pid=os.getpid(), strategy=args.strategy, order=args.order,
                  source_manifest=str(source.relative_to(ROOT)), manifest_sha256=sha(source),
                  input_fingerprint=meta['input_fingerprint'], source_hashes=source_hashes,
                  runner_sha256=sha(Path(__file__)), original_placement_preserved=True,
                  windows_ms=[[2000, 4000], [2000, 20000]], completed_simulation=False)
    write_json(directory / 'command.json', record)
    expected = sum(len(q.placement[0]) * 8 for q in requests)
    observed = 0
    per_layer = {}
    received = [0.] * 32
    ssd_served = [0.] * 3
    paths = od_npu_path_ids(32)
    started = last_progress = time.perf_counter()
    callback = native._register_complete

    def observe(context, flow):
        nonlocal observed, last_progress
        observed += 1
        if args.strategy == 'od_baseline':
            assert flow.queue_id == paths[flow.npu_id]
            # All completed flows are counted; no sampling of byte volume.
            key = (flow.request_id, flow.layer)
            row = per_layer.get(key)
            if row is None:
                row = [0, 0., flow.link_start_time, flow.link_end_time]
                per_layer[key] = row
            row[0] += 1
            row[1] += flow.total_gb
            row[2] = min(row[2], flow.link_start_time)
            row[3] = max(row[3], flow.link_end_time)
            received[flow.npu_id] += max(0., min(4000., flow.link_end_time) - max(2000., flow.link_start_time)) * 50 / 1000
            ssd_served[flow.disk_id] += max(0., min(4000., flow.link_enqueue_time) - max(2000., flow.ssd_activation_time)) * 40 / 1000
        if observed % 20000 == 0 and time.perf_counter() - last_progress > 30:
            progress = dict(simulation_ms=context.current_time_ms, completed_blocks=observed,
                            expected_blocks=expected, completed_requests=context.completed_requests,
                            wall_seconds=time.perf_counter() - started)
            write_json(directory / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            last_progress = time.perf_counter()
        return callback(context, flow)

    try:
        with patch.object(native, '_register_complete', observe):
            result = run_simulation(requests, strategy=args.strategy, num_npu=32, num_ssu=3,
                                    n_layers=8, seed=7, disk_bw_gib_s=40, npu_bw_gib_s=50)
        assert observed == expected
        assert result['input_fingerprint'] == meta['input_fingerprint']
        assert source_hashes == {p: sha(ROOT / p) for p in source_hashes}
        assert sha(source) == sha(directory / 'manifest.json.gz') == record['manifest_sha256']
        result['windows'] = [window_evidence(result['summary'], requests, *w) for w in record['windows_ms']]
        result['common_window'] = result['windows'][0]
        write_json(directory / 'result.json.gz', result)
        if args.strategy == 'asu_baseline':
            refpath = ORIGINAL / relative / 'result.json.gz'
            original_command = read_json(ARCHIVE / relative / 'command.json')
            assert sha(refpath) == original_command['output_sha256']
            old = read_json(refpath)['summary']
            new = result['summary']
            keys = ('admission_time_ms', 'completion_time_ms', 'own_compute_ms')
            by_id = lambda rows: {r['request_id']: r for r in rows}
            a, b = by_id(old['request_metrics']), by_id(new['request_metrics'])
            assert set(a) == set(b)
            request_error = max(abs(a[r][k] - b[r][k]) for r in a for k in keys)
            layer_keys = ('compute_start_ms', 'compute_end_ms', 'io_ready_time_ms', 'io_start_time_ms', 'io_barrier_wait_ms')
            batchmap = lambda s: {tuple(b['member_request_ids']): b for b in s['microbatch_metrics']}
            a, b = batchmap(old), batchmap(new)
            assert set(a) == set(b)
            layer_error = max(abs(x[k]-y[k]) for r in a for x,y in zip(a[r]['layer_metrics'],b[r]['layer_metrics']) for k in layer_keys)
            assert request_error == layer_error == 0
            record['parity'] = dict(reference_result=str(refpath.relative_to(ROOT)), reference_sha256=sha(refpath),
                                    all_3840_requests_identical=True, all_30720_layers_identical=True,
                                    max_request_time_error_ms=request_error, max_layer_time_error_ms=layer_error)
        else:
            for q in requests:
                for layer in range(8):
                    row = per_layer[(q.request_id, layer)]
                    assert row[0] == len(q.placement[0])
                    assert math.isclose(row[1], math.fsum(v for _, v in q.placement[0]), abs_tol=1e-12)
            audit = dict(columns=['request_id', 'layer', 'block_count', 'received_GiB', 'first_link_start_ms', 'last_link_end_ms'],
                         rows=[[rid, layer, *row] for (rid, layer), row in sorted(per_layer.items())],
                         warm_received_GiB_by_npu=received, warm_ssd_served_GiB_by_ssu=ssd_served,
                         all_flows_on_execution_npu_exclusive_path=True,
                         expected_completed_blocks=expected, observed_completed_blocks=observed)
            write_json(directory / 'io_audit.json.gz', audit)
        record.update(status='complete', completed_simulation=True, output_sha256=sha(directory / 'result.json.gz'),
                      U_percent=100*result['common_window']['mean_npu_utilization'],
                      invariants_passed=all(result['summary']['invariants'].values()), observed_completed_blocks=observed)
    except BaseException as exc:
        record.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['wall_seconds'] = time.perf_counter() - started
        write_json(directory / 'command.json', record)
    print(json.dumps({k:v for k,v in record.items() if k != 'source_hashes'}), flush=True)


if __name__ == '__main__':
    main()
