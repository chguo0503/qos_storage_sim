#!/usr/bin/env python3
"""Freeze exact per-card A:B=1:2 queues; optionally run unchanged Baseline.

Default invocation only prepares input.  --run executes the complete finite
population, observing immutable completed block fields without new events.
"""
from pathlib import Path
from collections import Counter, deque
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
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

NUM_NPU, LAYERS, DISK_BW, NPU_BW = 32, 8, 40.0, 50.0
BLOCK_GIB = 176 * 1024 / 2**30
TRACE_COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
                 'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
                 'link_start_ms', 'link_end_ms']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def window(text):
    try:
        a, z = map(float, text.split(':'))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Use START_MS:END_MS') from exc
    if not (math.isfinite(a) and math.isfinite(z) and 0 <= a < z):
        raise argparse.ArgumentTypeError('Window must satisfy 0 <= START < END')
    return a, z


def source_hashes():
    return {name: sha(ROOT / name) for name in source_files()}


def build(args):
    profiles, provenance = profiles_for('raw', '128:256,32:4096')
    for p, role in zip(profiles, ('A', 'B')):
        p['role'] = role
        p['name'] = role
        assert p['construction']['method'] == 'direct_data_row'
        assert p['ssd_prefix_tokens'] % 128 == 0
    costs = [LAYERS * p['per_layer_compute_us'] / 1000 for p in profiles]
    cycle_ms = costs[0] + 2 * costs[1]
    minimum_quota = math.ceil(args.horizon_ms / cycle_ms)
    quota = args.a_count if args.a_count is not None else 4 * math.ceil(minimum_quota / 4)
    assert quota >= 1 and quota * cycle_ms >= args.horizon_ms
    canonical = [i for _ in range(quota) for i in (0, 1, 1)]
    requests, assignment = [], []
    per_card_maxima, per_card_average = [], []
    for npu in range(NUM_NPU):
        # Original identities come from ABB repeated, independent of tested order.
        pool = {i: deque(j for j, p in enumerate(canonical) if p == i) for i in (0, 1)}
        ordered_ids = []
        left = quota
        while left:
            k = min(args.block, left)
            ordered_ids.extend(pool[0].popleft() for _ in range(k))
            ordered_ids.extend(pool[1].popleft() for _ in range(2 * k))
            left -= k
        fraction = (3 * npu // NUM_NPU) / 3 if args.phase == '3groups' else 0.0
        period_ms = min(args.block, quota) * cycle_ms
        target_ms = fraction * period_ms
        cumulative = [0.0]
        for identity in ordered_ids:
            cumulative.append(cumulative[-1] + costs[canonical[identity]])
        # Nearest whole-request offset within the first block period; no retiming.
        offset = min(range(min(3 * args.block, len(ordered_ids))), key=lambda i: abs(cumulative[i] - target_ms))
        if args.phase == 'split':
            offset = 0 if npu < 16 else (3 * min(args.block, quota) - 1)
        elif args.phase == 'stagger':
            k0 = min(args.block, quota)
            offset = (0, k0, k0 + k0 // 2, 2 * k0)[npu // 8]
        if args.phase in ('split', 'stagger'):
            target_ms = cumulative[offset]
            fraction = target_ms / period_ms
        if args.mode == 'ordered':
            identities = ordered_ids[offset:] + ordered_ids[:offset]
            rng_seed = None
        else:
            identities = list(range(len(canonical)))
            rng_seed = args.seed + 100003 * npu
            random.Random(rng_seed).shuffle(identities)
        placements = []
        rate_profiles = []
        for p in profiles:
            layer = tuple(((j + npu) % args.num_ssu, BLOCK_GIB)
                          for j in range(p['ssd_prefix_tokens'] // 128))
            assert math.isclose(math.fsum(v for _, v in layer), p['per_layer_kv_gib'], abs_tol=1e-12)
            placements.append((layer,))
            rate_profiles.append([math.fsum(v for d, v in layer if d == s) / (p['per_layer_compute_us'] / 1e6)
                                  for s in range(args.num_ssu)])
        per_card_maxima.append([max(r[s] for r in rate_profiles) for s in range(args.num_ssu)])
        per_card_average.append([sum(rate_profiles[i][s] * costs[i] * (1 if i == 0 else 2) for i in (0, 1)) / cycle_ms
                                 for s in range(args.num_ssu)])
        role_C = Counter()
        for position, original in enumerate(identities):
            profile_index = canonical[original]
            p = profiles[profile_index]
            rid = npu * 1000000 + position
            original_id = npu * 1000000 + original
            load = dict(request_id=rid, npu_id=npu, generation=position,
                        original_request_id=original_id, profile_index=profile_index,
                        role=p['role'], seq_len_k=p['seq_len_k'], nql=p['nql'],
                        total_tokens=p['total_tokens'], ssd_prefix_tokens=p['ssd_prefix_tokens'],
                        category=sim.classify_request(p['seq_len_k'], p['nql']),
                        per_layer_us=p['per_layer_compute_us'], per_layer_kv_gb=p['per_layer_kv_gib'],
                        required_bw_input_gbps=p['required_bandwidth_gibps'],
                        source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                        original_compute_us=p['per_layer_compute_us'], constructed_profile=False,
                        profile_construction=p['construction'], padding_gib_per_layer=0.0,
                        arrival_time=0.0, arrival_ms=0.0, initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, placements[profile_index]))
            role_C[p['role']] += costs[profile_index]
        assert Counter(canonical[i] for i in identities) == {0: quota, 1: 2 * quota}
        assignment.append(dict(npu_id=npu, A_requests=quota, B_requests=2 * quota,
                               requests=3 * quota, pure_compute_ms=quota * cycle_ms,
                               role_pure_compute_ms=dict(role_C),
                               shuffle_seed=rng_seed,
                               ordered_phase_fraction=fraction if args.mode == 'ordered' else None,
                               ordered_phase_target_ms=target_ms if args.mode == 'ordered' else None,
                               ordered_phase_offset_requests=offset if args.mode == 'ordered' else None,
                               ordered_phase_actual_ms=cumulative[offset] if args.mode == 'ordered' else None,
                               first_24_roles=[profiles[canonical[i]]['role'] for i in identities[:24]]))
    requests = tuple(requests)
    fp = continuous_batch_input_fingerprint(requests)
    label = args.label or f'ab12_ssu{args.num_ssu}_{args.mode}_k{args.block}_{args.phase}_a{quota}_h{args.horizon_ms:g}_seed{args.seed}'
    peaks = [math.fsum(r[s] for r in per_card_maxima) for s in range(args.num_ssu)]
    averages = [math.fsum(r[s] for r in per_card_average) for s in range(args.num_ssu)]
    metadata = dict(experiment='ab128_32_ratio12_20260912', label=label, case_id=label + '_' + fp[:12],
                    num_npu=NUM_NPU, num_ssu=args.num_ssu, n_layers=LAYERS, seed=args.seed,
                    disk_bw_gib_s=DISK_BW, npu_bw_gib_s=NPU_BW,
                    family='raw', profiles=profiles, source=provenance,
                    profile_keys='128:256,32:4096', equal_176kib_blocks=True, blocks='exact',
                    input_fingerprint=fp, logical_input_fingerprint=logical_input_fingerprint(requests),
                    source_data_sha256=sha(ROOT / 'data'), compute_scale_actual=1.0,
                    constructed_profile=False, last_arrival_ms=0.0, request_count=len(requests),
                    order=args.mode, order_mode=args.mode, block_A_requests=args.block, phase=args.phase,
                    horizon_pure_compute_ms=args.horizon_ms, quota_cycles=quota,
                    minimum_A_count_for_horizon=minimum_quota, explicit_A_count=args.a_count,
                    per_npu_assignment=assignment, measurement_window_ms=list(args.window[0]),
                    layout='stripe_npu_mod_ssu', placement_rule='(block_index + npu_id) % num_ssu, exact 176KiB per block; physical placement stays with original identity',
                    population_rule='Every NPU has q A and 2q B; q is explicit --a-count or the minimum horizon-covering A count rounded upward to a multiple of 4, independent of k; all arrive at zero, fixed NPU, no barriers',
                    ordered_rule='A^k B^(2k) blocks; final partial block preserves 1:2. Optional per-card whole-request cyclic rotation nearest target pure-compute phase',
                    phase_rule='sync offset0; split cards0-15 offset0 and cards16-31 offset3k-1; stagger four contiguous groups of8 offset0,k,k+floor(k/2),2k (k4 gives0,4,6,8); 3groups three contiguous groups rotate nearest pure-compute fractions0,1/3,2/3 of one block period',
                    random_rule='Independent full-population shuffle with Random(seed+100003*npu), no repeated random deck',
                    identity_rule='request_id=npu*1000000+queue_position encodes admission order; original_request_id preserves identity across order. All other load fields and placement unchanged',
                    input_length_semantics='seq_len_k is total input including new query; A=128K total with NQL256, B=32K total with NQL4096. Not hit-prefix length plus NQL',
                    static_per_ssu_upper_bound_gib_s=peaks,
                    static_upper_bound_passes=all(v < DISK_BW for v in peaks),
                    time_weighted_per_ssu_nominal_gib_s=averages,
                    time_weighted_fleet_nominal_gib_s=math.fsum(averages),
                    nominal_capacity_status='PENDING actual admission/completion event scan; static maxima or population mean are not the observed peak',
                    nominal_definition='Current admitted request actual per-SSU layer D/C; next-request L0 not added again; all physical I/O simulated',
                    caveat='Artificial ordering/population, original data C/D. No claim that low U or any capacity condition will hold')
    path = args.output / 'inputs' / (label + '.json.gz')
    if path.exists():
        existing = read_json(path)
        assert existing['input_fingerprint'] == fp and existing['metadata'] == metadata, 'Preserve different existing manifest/metadata'
    save_manifest(path, requests, metadata)
    return requests, metadata, path


def execute(args, requests, metadata, manifest, before):
    directory = args.output / 'runs' / metadata['label'] / 'baseline'
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive directory claim prevents accidental overwrite or duplicate execution.
    directory.mkdir(exist_ok=False)
    # A self-contained per-case input copy is byte-identical to the frozen catalog.
    local_manifest = directory / 'manifest.json.gz'
    local_manifest.write_bytes(manifest.read_bytes())
    assert sha(local_manifest) == sha(manifest)
    raw_path, trace_path = directory / 'result.json.gz', directory / 'trace.json.gz'
    record = dict(status='running', pid=os.getpid(), strategy='baseline', assignment='fixed',
                  label=metadata['label'], input_fingerprint=metadata['input_fingerprint'],
                  argv=sys.argv, cwd=str(ROOT), started_utc=utc(),
                  manifest=str(manifest), manifest_sha256=sha(manifest), output=str(raw_path),
                  case_manifest=str(local_manifest), case_manifest_sha256=sha(local_manifest),
                  runner_sha256=sha(__file__), core_source_sha256=before,
                  source_data_sha256=sha(ROOT / 'data'),
                  stress_runner_sha256=sha(ROOT / 'run_baseline_npu32_stress.py'),
                  windows=[list(w) for w in args.window],
                  trace_window_ms=None if args.no_trace else list(args.trace_window),
                  completed_simulation=False)
    write_json(directory / 'command.json', record)
    started, rows, completed_count = time.perf_counter(), [], 0
    original = native._register_complete
    a, z = args.trace_window

    def observe(context, flow):
        nonlocal completed_count
        if ((flow.ssd_activation_time < z and flow.link_enqueue_time > a) or
                (flow.link_start_time < z and flow.link_end_time > a)):
            rows.append([flow.request_id, flow.npu_id, flow.layer, flow.block_idx,
                         flow.disk_id, flow.queue_id, flow.total_gb, flow.block_count,
                         flow.enqueue_time, flow.ssd_activation_time, flow.link_enqueue_time,
                         flow.link_start_time, flow.link_end_time])
        completed_count += 1
        return original(context, flow)

    try:
        if args.no_trace:
            result = run_case(requests, metadata, strategy='baseline', assignment='fixed', windows=tuple(args.window))
        else:
            # Adapter enters inside run_case and captures this observer as its original callback.
            with patch.object(native, '_register_complete', observe):
                result = run_case(requests, metadata, strategy='baseline', assignment='fixed', windows=tuple(args.window))
        assert source_hashes() == before
        assert sha(__file__) == record['runner_sha256']
        assert sha(ROOT / 'data') == record['source_data_sha256']
        assert sha(ROOT / 'run_baseline_npu32_stress.py') == record['stress_runner_sha256']
        assert sha(manifest) == record['manifest_sha256']
        assert result['core_and_policy_sha256'] == before
        assert result['input_fingerprint'] == metadata['input_fingerprint']
        assert all(result['summary']['invariants'].values())
        result['manifest_path'] = str(manifest)
        result['experiment_runner_sha256'] = record['runner_sha256']
        write_json(raw_path, result)
        if not args.no_trace:
            expected = sum(len(r.placement[0]) * LAYERS for r in requests)
            assert completed_count == expected, (completed_count, expected)
            trace = dict(schema_version=1, columns=TRACE_COLUMNS, rows=rows,
                         window_ms=[a, z], completed_simulation=True, strategy='baseline',
                         source=dict(manifest=str(manifest), manifest_sha256=record['manifest_sha256'],
                                     reference_result=str(raw_path), reference_sha256=sha(raw_path),
                                     input_fingerprint=metadata['input_fingerprint'],
                                     core_source_sha256=before, observer_source_sha256=record['runner_sha256']),
                         capture_rule='Record complete immutable block fields iff physical SSD service or NPU-link service overlaps [left,right); observe every completion until the finite run fully drains',
                         completeness_scope='All physical service in capture window is represented. A layer may contain only some blocks; require exact manifest block count/identity before drawing its full cumulative curve',
                         checks=dict(all_expected_blocks_observed=completed_count == expected,
                                     all_baseline_path0=all(r[5] == 0 for r in rows),
                                     core_unchanged=True, original_invariants_passed=True),
                         observed_completed_blocks=completed_count, expected_completed_blocks=expected,
                         retained_blocks=len(rows))
            assert all(trace['checks'].values())
            write_json(trace_path, trace)
            record.update(trace=str(trace_path), trace_sha256=sha(trace_path), retained_blocks=len(rows),
                          observed_completed_blocks=completed_count)
        record.update(status='complete', returncode=0, completed_simulation=True,
                      output_sha256=sha(raw_path),
                      windows=[dict(start_ms=w['start_ms'], end_ms=w['end_ms'],
                                    U=w['mean_npu_utilization'], all_active=w['all_npus_active_whole_window'])
                               for w in result['windows']])
    except BaseException as exc:
        record.update(status='failed', returncode=1, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(wall_seconds=time.perf_counter() - started, ended_utc=utc())
        write_json(directory / 'command.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-ssu', type=int, choices=(3, 4, 6), default=6)
    parser.add_argument('--mode', choices=('random', 'ordered'), default='ordered')
    parser.add_argument('--block', type=int, default=1, help='Ordered block A^k B^(2k)')
    parser.add_argument('--phase', choices=('sync', 'split', 'stagger', '3groups'), default='sync')
    parser.add_argument('--a-count', type=int, help='Exact A count per card; B count is twice this; must cover horizon')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--horizon-ms', type=float, default=4500, help='Minimum pure compute per NPU, not simulation stop time')
    parser.add_argument('--window', type=window, action='append', help='Measured window in ms; default 2000:4000')
    parser.add_argument('--trace-window', type=window, default=(1800.0, 4200.0))
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--label')
    parser.add_argument('--manifest', type=Path, help='Run an already frozen manifest; queue construction arguments are ignored')
    parser.add_argument('--run', action='store_true', help='Execute a complete simulation; otherwise only freeze input')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.window = args.window or [(2000.0, 4000.0)]
    if args.block <= 0 or (args.a_count is not None and args.a_count <= 0) or not math.isfinite(args.horizon_ms) or args.horizon_ms <= 0:
        parser.error('block and horizon must be positive')
    if args.label and (Path(args.label).name != args.label or args.label in ('.', '..')):
        parser.error('label must be one path component')
    args.output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    if args.manifest:
        manifest = args.manifest.resolve()
        requests, metadata = load_manifest(manifest)
        assert metadata['experiment'] == 'ab128_32_ratio12_20260912'
        assert metadata['source_data_sha256'] == sha(ROOT / 'data')
    else:
        requests, metadata, manifest = build(args)
    snapshot = args.output / 'source_snapshots' / sha(__file__) / 'experiment.py'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    assert sha(snapshot) == sha(__file__)
    plan = dict(status='prepared', created_utc=utc(), label=metadata['label'],
                manifest=str(manifest), manifest_sha256=sha(manifest),
                input_fingerprint=metadata['input_fingerprint'], request_count=len(requests),
                core_source_sha256=before, runner_sha256=sha(__file__),
                source_data_sha256=sha(ROOT / 'data'),
                expected_strategy='baseline', expected_assignment='fixed',
                windows=[list(w) for w in args.window],
                trace_window_ms=None if args.no_trace else list(args.trace_window))
    plan_path = args.output / 'plans' / (metadata['label'] + '.json')
    if not plan_path.exists():
        write_json(plan_path, plan)
    else:
        prior = read_json(plan_path)
        assert all(prior[k] == plan[k] for k in ('manifest_sha256', 'input_fingerprint', 'core_source_sha256', 'runner_sha256', 'source_data_sha256')), 'Frozen plan source/input mismatch'
    print(json.dumps(dict(status='prepared', label=metadata['label'], manifest=str(manifest),
                         requests=len(requests), quota_cycles=metadata['quota_cycles'],
                         pure_compute_ms_per_npu=metadata['per_npu_assignment'][0]['pure_compute_ms'],
                         static_capacity_is_only_bound=metadata['static_per_ssu_upper_bound_gib_s'],
                         time_weighted_nominal_gib_s=metadata['time_weighted_fleet_nominal_gib_s'],
                         will_run=args.run), ensure_ascii=False), flush=True)
    if args.run:
        execute(args, requests, metadata, manifest, before)


if __name__ == '__main__':
    main()

