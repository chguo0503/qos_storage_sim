#!/usr/bin/env python3
"""Re-run the frozen A:B=1:2 random workload with native block ring hash."""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import argparse
import copy
import hashlib
import json
import math
import sys
import time

import sim
import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, run_case, write_json
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/ring_hash_baseline_random_20260914'
REFERENCE = ROOT / 'reference_manifest.json.gz'
REFERENCE_SHA256 = 'ec6337cc9939bcfa56279166502db06df11cc7471707fa33a28076cd8f36355b'
SOURCE_COMMIT = '75e10b8a84d4054921cd3149af40507444efb3fb'
WINDOWS = ((2000., 4000.), (2000., 20000.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(num_ssu):
    assert sha(REFERENCE) == REFERENCE_SHA256
    original, reference_metadata = load_manifest(REFERENCE)
    assert len(original) == 3840
    assert reference_metadata['seed'] == 7
    assert reference_metadata['order'] == 'random'
    assert reference_metadata['n_layers'] == 8
    assert all(q.arrival_time_ms == 0.0 for q in original)
    assert Counter(q.npu_id for q in original) == {i: 120 for i in range(32)}
    assert all(Counter(q.load['role'] for q in original if q.npu_id == n) == {'A': 40, 'B': 80}
               for n in range(32))
    requests = []
    counts = [0] * num_ssu
    moved = old_blocks = 0
    for q in original:
        assert len(q.placement) == 1
        # Physical identity precedes random shuffling; queue position must not move data.
        identity = int(q.load['original_request_id'])
        layer = tuple((sim.block_ring_hash_disk_id(identity, j, num_ssu), size)
                      for j, (_, size) in enumerate(q.placement[0]))
        assert math.isclose(math.fsum(size for _, size in layer),
                            q.load['per_layer_kv_gb'], rel_tol=0, abs_tol=1e-12)
        for j, (ssu, size) in enumerate(layer):
            assert size == 176 * 1024 / 2**30
            counts[ssu] += 1
            old_blocks += 1
            moved += ssu != q.placement[0][j][0]
        new = ContinuousBatchRequest.from_normalized(
            q.request_id, q.npu_id, q.arrival_time_ms, dict(q.load), (layer,))
        assert new.request_id == q.request_id and new.npu_id == q.npu_id
        assert dict(new.load) == dict(q.load)
        assert all(native._manifest_layer(new, k) is layer for k in range(8))
        requests.append(new)
    requests = tuple(requests)
    logical = logical_input_fingerprint(requests)
    assert logical == logical_input_fingerprint(original)
    metadata = copy.deepcopy(reference_metadata)
    # These old fields describe the former stripe placement; replace with actual demand.
    for key in ('static_per_ssu_upper_bound_gib_s', 'static_upper_bound_passes',
                'time_weighted_fleet_nominal_gib_s', 'time_weighted_per_ssu_nominal_gib_s',
                'nominal_capacity_status', 'nominal_definition'):
        metadata.pop(key, None)
    fingerprint = continuous_batch_input_fingerprint(requests)
    metadata.update(
        experiment='ring_hash_baseline_random_20260914', num_ssu=num_ssu,
        label=f'ring_hash_baseline_random_ssu{num_ssu}_seed7',
        case_id=f'ring_hash_baseline_random_ssu{num_ssu}_{fingerprint[:12]}',
        layout='block_ring_hash', placement_rule='Native sim.block_ring_hash_disk_id(original_request_id, block_index, num_ssu); all 8 layers reuse the same mapping',
        placement_key_fields=['original_request_id', 'block_index'],
        placement_layer_in_key=False, placement_virtual_nodes_per_ssu=sim.BLOCK_RING_VIRTUAL_NODES,
        input_fingerprint=fingerprint, logical_input_fingerprint=logical,
        reference_manifest_sha256=REFERENCE_SHA256, source_commit=SOURCE_COMMIT,
        input_demand=input_demand(requests, 32, num_ssu),
        reference_layout='stripe_npu_mod_ssu',
        placement_only_change=True,
    )
    audit = dict(all_nonplacement_request_fields_unchanged=True,
                 same_logical_input_fingerprint=True, all_layers_same_ssu_per_block=True,
                 all_blocks_exact_176kib=True, requests=3840, layers=8,
                 blocks_per_layer_population=old_blocks, expected_completed_io_blocks=old_blocks * 8,
                 blocks_per_layer_population_by_ssu=counts,
                 ssu_population_shares=[c / old_blocks for c in counts],
                 changed_block_destinations_from_reference=moved,
                 physical_identity='original_request_id', virtual_nodes_per_ssu=256)
    return requests, metadata, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-ssu', type=int, choices=(3, 4), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    directory = OUT / f'ssu{args.num_ssu}'
    directory.mkdir(parents=True, exist_ok=True)
    requests, metadata, audit = prepare(args.num_ssu)
    manifest = directory / 'manifest.json.gz'
    save_manifest(manifest, requests, metadata)
    write_json(directory / 'placement_audit.json', audit)
    print(json.dumps({'event': 'prepared', 'ssu': args.num_ssu, 'placement': audit}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    result_path = directory / 'result.json.gz'
    if result_path.exists():
        raise FileExistsError(f'Preserving completed result: {result_path}')
    started = time.perf_counter()
    expected = audit['expected_completed_io_blocks']
    observed = 0
    nonzero_paths = 0
    last_progress = started
    native_complete = native._register_complete

    def observe(context, flow):
        nonlocal observed, nonzero_paths, last_progress
        observed += 1
        nonzero_paths += int(flow.queue_id != 0)
        if observed % 100000 == 0 and time.perf_counter() - last_progress >= 20:
            status = dict(event='progress', ssu=args.num_ssu,
                          simulation_ms=context.current_time_ms,
                          completed_requests=context.completed_requests,
                          completed_io_blocks=observed, expected_io_blocks=expected,
                          wall_seconds=time.perf_counter() - started)
            write_json(directory / 'progress.json', status)
            print(json.dumps(status), flush=True)
            last_progress = time.perf_counter()
        return native_complete(context, flow)

    record = dict(status='running', strategy='baseline', order='random', assignment='fixed',
                  num_npu=32, num_ssu=args.num_ssu, seed=7,
                  argv=sys.argv, windows_ms=WINDOWS, source_commit=SOURCE_COMMIT,
                  reference_manifest_sha256=REFERENCE_SHA256,
                  manifest_sha256=sha(manifest), runner_sha256=sha(__file__),
                  modeled_observer_latency_ms=0, retains_full_io_trace=False)
    write_json(directory / 'command.json', record)
    try:
        with patch.object(native, '_register_complete', observe):
            result = run_case(requests, metadata, strategy='baseline',
                              assignment='fixed', windows=WINDOWS)
        assert observed == expected
        assert nonzero_paths == 0
        assert all(result['summary']['invariants'].values())
        assert result['summary']['invariants']['all_requests_completed']
        assert len(result['summary']['request_metrics']) == len(requests)
        assert sha(manifest) == record['manifest_sha256']
        assert sha(__file__) == record['runner_sha256']
        audit.update(all_completed_ios_on_path0=True, observed_completed_io_blocks=observed,
                     all_expected_io_blocks_completed=True)
        result['placement_audit'] = audit
        result['experiment'] = metadata['experiment']
        result['runner_sha256'] = record['runner_sha256']
        write_json(result_path, result)
        write_json(directory / 'placement_audit.json', audit)
        record.update(status='complete', result_sha256=sha(result_path),
                      completed_io_blocks=observed, all_completed_ios_on_path0=True,
                      wall_seconds=time.perf_counter() - started)
        print(json.dumps({'event': 'complete', 'ssu': args.num_ssu,
                          'windows': [{k: w[k] for k in ('start_ms', 'end_ms', 'mean_npu_utilization')}
                                      for w in result['windows']],
                          'main_window_slo': result['slo']['window_admissions']['admission'],
                          'wall_seconds': record['wall_seconds']}), flush=True)
    except BaseException as exc:
        record.update(status='failed', error=repr(exc), wall_seconds=time.perf_counter() - started)
        raise
    finally:
        write_json(directory / 'command.json', record)


if __name__ == '__main__':
    main()
