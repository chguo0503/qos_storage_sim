#!/usr/bin/env python3
"""Frozen raw-data, independent-Random experiments; unchanged simulator core."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import sim
import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import profiles_for, save_manifest, load_manifest, read_json, write_json, run_case
from run_shared_path_experiments import logical_input_fingerprint
from run_coflow_experiments import source_files

NPU, LAYERS, DISK_BW, NPU_BW = 32, 8, 40.0, 50.0
BLOCK_GIB = 176 * 1024 / 2**30
TRACE_COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
                 'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
                 'link_start_ms', 'link_end_ms']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha(ROOT / name) for name in source_files()}


def prepare(args):
    profiles, provenance = profiles_for('raw', args.profiles)
    counts = tuple(map(int, args.counts.split(',')))
    roles = args.roles.split(',')
    assert len(profiles) == len(counts) == len(roles)
    assert min(counts) >= 1 and len(set(roles)) == len(roles)
    costs = []
    for p, role in zip(profiles, roles):
        p.update(role=role, name=role)
        assert p['construction']['method'] == 'direct_data_row'
        assert p['total_tokens'] >= 10240 and p['ssd_prefix_tokens'] % 128 == 0
        costs.append(LAYERS * p['per_layer_compute_us'] / 1000)
        assert p['per_layer_kv_gib'] / (p['per_layer_compute_us'] / 1e6) < NPU_BW
    cycle_ms = math.fsum(c * n for c, n in zip(costs, counts))
    repeats = math.ceil(args.horizon_ms / cycle_ms)
    canonical = [i for i, count in enumerate(counts) for _ in range(count * repeats)]
    requests, assignment, avg_disks, max_disks = [], [], [], []
    for npu in range(NPU):
        identities = list(range(len(canonical)))
        random.Random(args.seed + 100003 * npu).shuffle(identities)
        placements, rates = [], []
        for p in profiles:
            layer = tuple(((j + npu) % args.num_ssu, BLOCK_GIB)
                          for j in range(p['ssd_prefix_tokens'] // 128))
            assert math.isclose(math.fsum(v for _, v in layer), p['per_layer_kv_gib'], abs_tol=1e-12)
            placements.append((layer,))
            rates.append([math.fsum(v for d, v in layer if d == s) / (p['per_layer_compute_us'] / 1e6)
                          for s in range(args.num_ssu)])
        avg_disks.append([math.fsum(rates[i][s] * costs[i] * counts[i] for i in range(len(profiles))) / cycle_ms
                          for s in range(args.num_ssu)])
        max_disks.append([max(r[s] for r in rates) for s in range(args.num_ssu)])
        for position, original in enumerate(identities):
            idx = canonical[original]
            p = profiles[idx]
            rid = npu * 1000000 + position
            load = dict(request_id=rid, npu_id=npu, generation=position,
                        original_request_id=npu * 1000000 + original, profile_index=idx,
                        role=p['role'], seq_len_k=p['seq_len_k'], nql=p['nql'],
                        total_tokens=p['total_tokens'], ssd_prefix_tokens=p['ssd_prefix_tokens'],
                        category=sim.classify_request(p['seq_len_k'], p['nql']),
                        per_layer_us=p['per_layer_compute_us'], per_layer_kv_gb=p['per_layer_kv_gib'],
                        required_bw_input_gbps=p['required_bandwidth_gibps'],
                        source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                        original_compute_us=p['per_layer_compute_us'], constructed_profile=False,
                        profile_construction=p['construction'], padding_gib_per_layer=0.0,
                        arrival_time=0.0, arrival_ms=0.0, initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, placements[idx]))
        actual = Counter(canonical[i] for i in identities)
        assert actual == {i: n * repeats for i, n in enumerate(counts)}
        assignment.append(dict(npu_id=npu, requests=len(canonical), pure_compute_ms=repeats * cycle_ms,
                               role_counts={roles[i]: n * repeats for i, n in enumerate(counts)},
                               role_pure_compute_ms={roles[i]: costs[i] * n * repeats for i, n in enumerate(counts)},
                               shuffle_seed=args.seed + 100003 * npu,
                               first_32_roles=[roles[canonical[i]] for i in identities[:32]]))
    requests = tuple(requests)
    fp = continuous_batch_input_fingerprint(requests)
    label = f'{args.name}_ssu{args.num_ssu}_h{args.horizon_ms:g}_seed{args.seed}'
    averages = [math.fsum(r[s] for r in avg_disks) for s in range(args.num_ssu)]
    maxima = [math.fsum(r[s] for r in max_disks) for s in range(args.num_ssu)]
    metadata = dict(experiment='baseline_random_near_capacity_20260914', label=label,
                    case_id=label + '_' + fp[:12], candidate=args.name,
                    num_npu=NPU, num_ssu=args.num_ssu, n_layers=LAYERS, seed=args.seed,
                    disk_bw_gib_s=DISK_BW, npu_bw_gib_s=NPU_BW,
                    family='raw', profiles=profiles, source=provenance,
                    profile_keys=args.profiles, role_names=roles, count_ratio=list(counts),
                    equal_176kib_blocks=True, blocks='exact', layout='stripe_npu_mod_ssu',
                    input_fingerprint=fp, logical_input_fingerprint=logical_input_fingerprint(requests),
                    source_data_sha256=sha(ROOT / 'data'), compute_scale_actual=1.0,
                    constructed_profile=False, last_arrival_ms=0.0, request_count=len(requests),
                    order='random', order_mode='random', regime='saturated_finite_backlog',
                    horizon_pure_compute_ms=args.horizon_ms, quota_cycles=repeats,
                    per_npu_assignment=assignment, measurement_window_ms=[2000., 4000.],
                    placement_rule='(block_index + npu_id) % num_ssu; exact 176KiB; one placement reused for all 8 layers',
                    random_rule='Independent full-population shuffle Random(seed+100003*npu); no repeated deck, phase alignment, barriers or seed rejection',
                    population_rule='Same integer profile-count ratio on every card; minimum whole repetitions covering requested pure-compute horizon, before randomization',
                    identity_rule='request_id=npu*1000000+queue_position; original_request_id preserves pre-shuffle identity',
                    static_per_ssu_upper_bound_gib_s=maxima,
                    static_upper_bound_passes=all(v <= DISK_BW for v in maxima),
                    time_weighted_per_ssu_nominal_gib_s=averages,
                    time_weighted_fleet_nominal_gib_s=math.fsum(averages),
                    ideal_load_ratio=math.fsum(averages) / (DISK_BW * args.num_ssu),
                    nominal_definition='Current admitted request per-SSU V/C; next-request L0 not added again; every physical read is simulated',
                    caveat='Near-capacity refers to ideal whole-population mean, not instantaneous/per-disk underload. Every-card mixed warm activity requires post-run validation.')
    path = HERE / 'inputs' / (label + '.json.gz')
    if path.exists():
        previous = read_json(path)
        assert previous['input_fingerprint'] == fp and previous['metadata'] == metadata
    save_manifest(path, requests, metadata)
    return requests, metadata, path


def execute(args, requests, metadata, manifest):
    before = source_hashes()
    directory = HERE / 'runs' / metadata['label'] / args.strategy
    directory.mkdir(parents=True, exist_ok=False)
    local_manifest = directory / 'manifest.json.gz'
    local_manifest.write_bytes(manifest.read_bytes())
    assert sha(local_manifest) == sha(manifest)
    windows = ((2000., 4000.), (2000., 20000.))
    expected = sum(len(r.placement[0]) * LAYERS for r in requests)
    record = dict(status='running', pid=os.getpid(), strategy=args.strategy, order='random',
                  assignment='fixed', label=metadata['label'], argv=sys.argv, started_utc=utc(),
                  host=platform.node(), python_version=sys.version,
                  manifest_sha256=sha(local_manifest), input_fingerprint=metadata['input_fingerprint'],
                  core_source_sha256=before, runner_sha256=sha(__file__),
                  source_data_sha256=sha(ROOT/'data'), expected_blocks=expected,
                  trace_window_ms=[1800., 4200.] if args.trace else None,
                  windows=list(windows), completed_simulation=False)
    write_json(directory / 'command.json', record)
    started = last_progress = time.perf_counter()
    rows, observed, bad_paths = [], 0, 0
    # Exact physical service integrals on fixed 2-second bins. Completion callbacks
    # run through the full drain, so late completions retain earlier service too.
    nbins = 10
    ssd_ms = [[0.0] * nbins for _ in range(metadata['num_ssu'])]
    link_ms = [[0.0] * nbins for _ in range(NPU)]
    original_callback = native._register_complete

    def add_interval(target, start, end):
        if start >= 20000. or end <= 0.:
            return
        start, end = max(0., start), min(20000., end)
        first, last = int(start // 2000.), min(nbins - 1, int(end // 2000.))
        if first == last:
            target[first] += end - start
        else:
            for k in range(first, last + 1):
                target[k] += max(0., min(end, (k + 1) * 2000.) - max(start, k * 2000.))

    def observe(context, flow):
        nonlocal observed, last_progress, bad_paths
        add_interval(ssd_ms[flow.disk_id], flow.ssd_activation_time, flow.link_enqueue_time)
        add_interval(link_ms[flow.npu_id], flow.link_start_time, flow.link_end_time)
        if args.trace and ((flow.ssd_activation_time < 4200. and flow.link_enqueue_time > 1800.) or
                           (flow.link_start_time < 4200. and flow.link_end_time > 1800.)):
            rows.append([flow.request_id, flow.npu_id, flow.layer, flow.block_idx,
                         flow.disk_id, flow.queue_id, flow.total_gb, flow.block_count,
                         flow.enqueue_time, flow.ssd_activation_time, flow.link_enqueue_time,
                         flow.link_start_time, flow.link_end_time])
        if args.strategy == 'baseline' and flow.queue_id != 0:
            bad_paths += 1
        observed += 1
        if observed % 20000 == 0 and time.perf_counter() - last_progress >= 25:
            progress = dict(label=metadata['label'], strategy=args.strategy,
                            simulation_ms=context.current_time_ms, completed_requests=context.completed_requests,
                            completed_blocks=observed, expected_blocks=expected,
                            wall_seconds=time.perf_counter() - started)
            write_json(directory / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            last_progress = time.perf_counter()
        return original_callback(context, flow)

    print(json.dumps(dict(event='start', label=metadata['label'], strategy=args.strategy,
                          expected_blocks=expected, ideal_load_ratio=metadata['ideal_load_ratio'])), flush=True)
    try:
        with patch.object(native, '_register_complete', observe):
            result = run_case(requests, metadata, strategy=args.strategy, assignment='fixed', windows=windows)
        assert source_hashes() == before == result['core_and_policy_sha256']
        assert sha(__file__) == record['runner_sha256']
        assert sha(local_manifest) == sha(manifest) == record['manifest_sha256']
        assert result['input_fingerprint'] == metadata['input_fingerprint']
        assert all(result['summary']['invariants'].values()) and observed == expected and bad_paths == 0
        assert all(t <= 2000. + 1e-5 for bins in ssd_ms + link_ms for t in bins)
        result['manifest_path'] = str(local_manifest)
        result['experiment_runner_sha256'] = record['runner_sha256']
        write_json(directory/'result.json.gz', result)
        physical = dict(bin_edges_ms=list(range(0, 20001, 2000)), ssd_busy_ms=ssd_ms, npu_link_busy_ms=link_ms,
                        ssd_gib_per_second=DISK_BW, npu_link_gib_per_second=NPU_BW,
                        method='Exact overlap of immutable physical SSD/link service intervals with fixed 2-second bins; all block completions observed',
                        observed_completed_blocks=observed, expected_blocks=expected, baseline_nonzero_paths=bad_paths,
                        core_unchanged=True, input_fingerprint=metadata['input_fingerprint'])
        write_json(directory/'physical_service.json', physical)
        if args.trace:
            write_json(directory/'trace.json.gz', dict(schema_version=1, columns=TRACE_COLUMNS, rows=rows,
                       window_ms=[1800., 4200.], strategy=args.strategy, completed_simulation=True,
                       observed_completed_blocks=observed, expected_completed_blocks=expected,
                       retained_blocks=len(rows), source=dict(manifest_sha256=sha(local_manifest),
                       reference_sha256=sha(directory/'result.json.gz'), core_source_sha256=before,
                       observer_source_sha256=record['runner_sha256']),
                       checks=dict(all_expected_blocks_observed=observed == expected, core_unchanged=True,
                       all_baseline_path0=bad_paths == 0)))
        record.update(status='complete', returncode=0, completed_simulation=True,
                      output_sha256=sha(directory/'result.json.gz'),
                      physical_service_sha256=sha(directory/'physical_service.json'),
                      observed_completed_blocks=observed, retained_blocks=len(rows),
                      windows=[dict(start_ms=w['start_ms'], end_ms=w['end_ms'], U=w['mean_npu_utilization'],
                                    all_active=w['all_npus_active_whole_window']) for w in result['windows']])
        if args.trace:
            record['trace_sha256'] = sha(directory/'trace.json.gz')
    except BaseException as exc:
        record.update(status='failed', returncode=1, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(ended_utc=utc(), wall_seconds=time.perf_counter() - started)
        write_json(directory/'command.json', record)
    write_json(directory/'progress.json', dict(status='complete', completed_blocks=observed,
               expected_blocks=expected, completed_requests=len(requests), wall_seconds=record['wall_seconds']))
    print(json.dumps({k: record[k] for k in ('status', 'label', 'strategy', 'windows', 'wall_seconds')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='main512')
    p.add_argument('--profiles', default='192:4096,32:512')
    p.add_argument('--counts', default='1,9')
    p.add_argument('--roles', default='L,S')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num-ssu', type=int, default=3)
    p.add_argument('--horizon-ms', type=float, default=22000.)
    p.add_argument('--strategy', choices=('baseline', 'once'), default='baseline')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    assert args.horizon_ms >= 22000. and args.num_ssu >= 1
    if args.manifest:
        requests, metadata = load_manifest(args.manifest)
        manifest = args.manifest
        assert metadata['order'] == 'random' and metadata['num_npu'] == NPU and metadata['n_layers'] == LAYERS
    else:
        requests, metadata, manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(manifest=str(manifest), label=metadata['label'], requests=len(requests),
                             ideal_load_ratio=metadata['ideal_load_ratio']), ensure_ascii=False))
    else:
        execute(args, requests, metadata, manifest)


if __name__ == '__main__':
    main()
