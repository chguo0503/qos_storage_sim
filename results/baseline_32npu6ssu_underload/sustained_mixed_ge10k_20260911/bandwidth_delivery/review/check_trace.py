#!/usr/bin/env python3
"""Independent offline trace validation; never imports or executes the simulator."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
           'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
           'link_start_ms', 'link_end_ms']
EXPLICIT_EXTRA_KEYS = {(22000046, 4), (26000038, 5), (16, 0)}
EXAMPLE_KEYS = EXPLICIT_EXTRA_KEYS | {(24000046, 2)}


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def inspect_policy(policy, bin_ms):
    directory = BASE / 'traces' / policy
    command_path = directory / 'command.json'
    if not command_path.exists():
        return dict(policy=policy, status='pending', reason='No replay command yet.')
    command = read(command_path)
    if command['status'] not in ('prefix_complete', 'audit_failed', 'failed'):
        return dict(policy=policy, status='pending', replay_status=command['status'])
    if command['status'] != 'prefix_complete':
        return dict(policy=policy, status='audit_failed', reason='Replay did not pass.', replay_status=command['status'])
    trace_path = directory / 'trace.json.gz'
    audit_path = directory / 'audit.json'
    trace, audit = read(trace_path), read(audit_path)
    source = trace['source']
    checks, details = {}, {}

    def check(name, ok, detail=None):
        checks[name] = bool(ok)
        if not ok and detail is not None:
            details[name] = detail

    check('replay_audit_passed', audit['passed'] and not audit['failed_checks'] and
          audit['status'] == 'prefix_complete' and trace['audit_passed'] and trace['audit']['passed'])
    check('trace_sha256', sha(trace_path) == command['trace_sha256'])
    check('audit_sha256', sha(audit_path) == command['audit_sha256'])
    check('columns_exact', trace['columns'] == COLUMNS)
    check('marked_partial_simulation', trace['completed_simulation'] is False and audit['completed_simulation'] is False)
    check('policy_identity', trace['strategy'] == policy == source['strategy'] == audit['strategy'])
    check('collector5ms_fixed_assignment', source['collector_interval_ms'] == 5.0 and
          source['reference_policy_config']['assignment'] == 'fixed')
    for key, hashkey in [('manifest', 'manifest_sha256'), ('reference_result', 'reference_sha256'),
                         ('reference_command', 'reference_command_sha256')]:
        check(key + '_hash', sha(source[key]) == source[hashkey])
    check('source_keyset29', len(source['core_source_sha256']) == 29)
    check('current_core_hashes', all(sha(ROOT / k) == v for k, v in source['core_source_sha256'].items()))
    check('observer_hash', sha(BASE / 'replay_trace.py') == source['observer_source_sha256'])
    check('data_hash', sha(ROOT / 'data') == source['data_sha256'])
    manifest = read(source['manifest'])
    reference = read(source['reference_result'])
    check('reference_policy_seed_input', reference['strategy'] == policy and reference['submit_seed'] == 7 and
          reference['input_fingerprint'] == source['input_fingerprint'] == manifest['input_fingerprint'])
    rows = trace['rows']
    left, right = map(float, trace['display_window_ms'])
    bins = round((right-left) / bin_ms)
    assert bins > 0 and math.isclose(bins * bin_ms, right-left, abs_tol=1e-9)
    requests = {r['request_id']: r for r in manifest['requests']}
    expected, all_layers = {}, {}
    for b in reference['summary']['microbatch_metrics']:
        assert len(b['member_request_ids']) == 1
        rid = b['member_request_ids'][0]
        for layer in b['layer_metrics']:
            all_layers[(rid, layer['layer'])] = layer
            if layer['io_start_time_ms'] < right and layer['io_ready_time_ms'] >= left:
                expected[(rid, layer['layer'])] = layer
    display_keys = set(expected)
    check('explicit_extra_keys_declared', {tuple(x) for x in source['extra_target_keys']} == EXPLICIT_EXTRA_KEYS)
    for key in EXPLICIT_EXTRA_KEYS:
        expected[key] = all_layers[key]
    grouped = defaultdict(list)
    physical_ok = identity_ok = stages_ok = baseline_path_ok = True
    max_ssd_duration_error = max_link_duration_error = 0.0
    for r in rows:
        grouped[(r[0], r[2])].append(r)
        physical_ok &= len(r) == 13 and r[6] == 176 * 1024 / 2**30 and r[7] == 1
        identity_ok &= r[0] in requests and r[1] == requests[r[0]]['npu_id'] and 0 <= r[4] < 6
        stages_ok &= r[8] <= r[9] + 1e-8 and r[9] < r[10] <= r[11] + 1e-8 and r[11] < r[12]
        baseline_path_ok &= policy != 'baseline' or r[5] == 0
        max_ssd_duration_error = max(max_ssd_duration_error, abs((r[10]-r[9])-r[6]*1000/40))
        max_link_duration_error = max(max_link_duration_error, abs((r[12]-r[11])-r[6]*1000/50))
    check('nonempty_rows', bool(rows))
    check('actual176KiB_blocks', physical_ok)
    check('request_and_resource_identity', identity_ok)
    check('chronological_stages', stages_ok)
    check('baseline_path0', baseline_path_ok)
    check('ssd_duration_40', max_ssd_duration_error < 1e-7, max_ssd_duration_error)
    check('link_duration_50', max_link_duration_error < 1e-7, max_link_duration_error)
    check('all_and_only_display_and_declared_example_layers', set(grouped) == set(expected))
    layer_errors = []
    for key, metric in expected.items():
        r = requests[key[0]]
        placement = manifest['placements'][r['placement_index']] if 'placement_index' in r else r['placement']
        placement = placement[0 if len(placement) == 1 else key[1]]
        blocks = sorted(grouped.get(key, []), key=lambda x: x[3])
        good = len(blocks) == len(placement)
        good &= [x[3] for x in blocks] == list(range(len(placement)))
        if good:
            good &= all(b[4] == p[0] and b[6] == p[1] for b, p in zip(blocks, placement))
            good &= max(b[12] for b in blocks) == metric['io_ready_time_ms']
        if not good:
            layer_errors.append(list(key))
    check('every_target_block_placement_and_last_hbm', not layer_errors, layer_errors[:10])
    check('target_ready_strictly_before_stop', max(x['io_ready_time_ms'] for x in expected.values()) < trace['stop_time_ms'])
    predecessors = {}
    for n in range(32):
        previous = None
        lane = sorted((b for b in reference['summary']['microbatch_metrics'] if b['npu_id'] == n),
                      key=lambda b: b['admission_time_ms'])
        for batch in lane:
            rid = batch['member_request_ids'][0]
            for layer in batch['layer_metrics']:
                predecessors[(rid, layer['layer'])] = previous
                previous = (rid, layer)
    examples = {}
    for key in sorted(EXAMPLE_KEYS):
        if key not in grouped:
            examples[str(key)] = dict(status='not_captured', explanation='Outside display and declared extra keys; no inferred trace.')
            continue
        prior_rid, prior = predecessors[key]
        metric, blocks = all_layers[key], grouped[key]
        release, deadline, ready = metric['io_start_time_ms'], prior['compute_end_ms'], metric['io_ready_time_ms']
        duration = prior['compute_duration_ms']
        payload = math.fsum(b[6] for b in blocks)
        def transferred(t, start, end):
            return math.fsum(b[6]*min(1.,max(0.,(t-b[start])/(b[end]-b[start]))) for b in blocks)
        ssd_before, hbm_before = transferred(deadline,9,10), transferred(deadline,11,12)
        stall = max(0.,ready-deadline)
        target = requests[key[0]]['load']; predecessor_load = requests[prior_rid]['load']
        check('example_deadline_stall_' + str(key), math.isclose(release,prior['compute_start_ms'],abs_tol=1e-7)
              and math.isclose(stall,metric['io_barrier_wait_ms'],abs_tol=1e-7)
              and math.isclose(metric['compute_start_ms'],max(deadline,ready),abs_tol=1e-7))
        examples[str(key)] = dict(status='complete', request_id=key[0],layer=key[1],
            release_ms=release,deadline_ms=deadline,ready_ms=ready,stall_ms=stall,
            payload_MiB=payload*1024,budget_C_ms=duration,budget_gib_s=payload*1000/duration,
            own_nominal_gib_s=target['per_layer_kv_gb']/(target['per_layer_us']/1e6),
            predecessor_own_nominal_gib_s=predecessor_load['per_layer_kv_gb']/(predecessor_load['per_layer_us']/1e6),
            ssd_before_deadline_MiB=ssd_before*1024,hbm_before_deadline_MiB=hbm_before*1024,
            ssd_deficit_MiB=max(0.,payload-ssd_before)*1024,hbm_deficit_MiB=max(0.,payload-hbm_before)*1024,
            actual_target_ssd_mean_during_stall_gib_s=(transferred(ready,9,10)-ssd_before)*1000/stall if stall else None,
            actual_target_link_mean_during_stall_gib_s=(transferred(ready,11,12)-hbm_before)*1000/stall if stall else None)
    parent_path = BASE / 'analysis.json'
    parent = read(parent_path) if parent_path.exists() else None
    parent_sha = sha(parent_path) if parent is not None else None
    if parent is not None:
        check('parent_bin_scope_matches', parent['window_ms'] == [left,right] and parent['bin_ms'] == bin_ms)
    stage_results = {}
    # Three independently tallied views: disk allocation to NPU, physical disks,
    # and NPU/HBM links. Generic overlap loops do not assume short block lengths.
    for name, start, end, resource, resource_count, rate in [
            ('ssd_by_npu', 9, 10, 1, 32, 40.),
            ('ssd_by_disk', 9, 10, 4, 6, 40.),
            ('link_by_npu', 11, 12, 1, 32, 50.)]:
        intervals = [[] for _ in range(resource_count)]
        direct = [[] for _ in range(resource_count)]
        bin_bytes = [[0.] * bins for _ in range(resource_count)]
        for r in rows:
            resource_id = int(r[resource])
            intervals[resource_id].append((r[start], r[end]))
            a, z = max(left, r[start]), min(right, r[end])
            if z <= a:
                continue
            direct[resource_id].append((z-a) * rate / 1000)
            lo = max(0, int(math.floor((a-left)/bin_ms)))
            hi = min(bins, int(math.ceil((z-left)/bin_ms)))
            for k in range(lo, hi):
                overlap = max(0., min(z, left+(k+1)*bin_ms)-max(a, left+k*bin_ms))
                bin_bytes[resource_id][k] += overlap * rate / 1000
        physical_resource = name != 'ssd_by_npu'
        max_overlap = 0.
        overlaps = 0
        if physical_resource:
            for resource_intervals in intervals:
                last_end = -math.inf
                for a, z in sorted(resource_intervals):
                    overlap = max(0., last_end-a)
                    max_overlap = max(max_overlap, overlap)
                    overlaps += int(overlap > 1e-8)
                    last_end = max(last_end, z)
            check(name + '_nonoverlap', overlaps == 0, dict(count=overlaps, max_overlap_ms=max_overlap))
        direct_totals = [math.fsum(x) for x in direct]
        binned_totals = [math.fsum(x) for x in bin_bytes]
        errors = [abs(a-b) for a,b in zip(direct_totals,binned_totals)]
        check(name + '_bin_conservation', all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9)
              for a,b in zip(direct_totals,binned_totals)), errors)
        peak = max(max(x) for x in bin_bytes) * 1000 / bin_ms
        if physical_resource:
            check(name + '_binned_capacity', peak <= rate + 1e-6, peak)
        stage_results[name] = dict(resource_count=resource_count, rate_during_one_block_gib_s=rate,
            direct_clipped_gib_by_resource=direct_totals, binned_gib_by_resource=binned_totals,
            total_clipped_gib=math.fsum(direct_totals), max_bin_conservation_error_gib=max(errors),
            max_binned_gib_s=peak, physical_resource_nonoverlap_checked=physical_resource,
            positive_overlap_count=overlaps if physical_resource else None,
            max_overlap_ms=max_overlap if physical_resource else None)
        if parent is not None:
            comparison = parent['policies'][policy]['ssu_service_bins_gib_s'] if name == 'ssd_by_disk' else (
                parent['policies'][policy]['bins']['ssd_gib_s' if name == 'ssd_by_npu' else 'link_gib_s'])
            difference = max(abs(bin_bytes[r][k]*1000/bin_ms-comparison[r][k])
                             for r in range(resource_count) for k in range(bins))
            check(name + '_parent_bins_match', difference <= 1e-8, difference)
            stage_results[name]['max_parent_bin_difference_gib_s'] = difference
        del intervals, direct, bin_bytes
    check('ssd_disk_npu_total_conservation', math.isclose(stage_results['ssd_by_npu']['total_clipped_gib'],
          stage_results['ssd_by_disk']['total_clipped_gib'], rel_tol=1e-12, abs_tol=1e-9))
    check('source_files_unchanged_at_end', sha(trace_path) == command['trace_sha256'] and
          sha(source['manifest']) == source['manifest_sha256'] and sha(source['reference_result']) == source['reference_sha256'])
    if parent is not None:
        check('parent_analysis_unchanged', sha(parent_path) == parent_sha)
    return dict(policy=policy, status='complete' if all(checks.values()) else 'audit_failed',
        passed=all(checks.values()), checks=checks, failed_details=details, captured_blocks=len(rows),
        independently_selected_layers=len(expected), display_layers=len(display_keys),
        extra_layers=len(set(expected)-display_keys),examples=examples,window_ms=[left,right], bin_ms=bin_ms,
        max_ssd_duration_error_ms=max_ssd_duration_error, max_link_duration_error_ms=max_link_duration_error,
        stages=stage_results, source=dict(trace_sha256=sha(trace_path), audit_sha256=sha(audit_path),
        reference_sha256=source['reference_sha256'], manifest_sha256=source['manifest_sha256'],parent_analysis_sha256=parent_sha),
        scope='Independent physical interval/byte/payload audit from raw trace and frozen manifest/reference. Replay prefix state equality is separately inherited from hashed replay audit; no new simulation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policies', nargs='+', choices=['baseline','once'], default=['baseline','once'])
    parser.add_argument('--bin-ms', type=float, default=.5)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    before = sha(__file__)
    reports = []
    for policy in args.policies:
        try:
            report = inspect_policy(policy, args.bin_ms)
        except Exception as exc:
            report = dict(policy=policy, status='audit_failed', error_type=type(exc).__name__, error=str(exc))
        reports.append(report)
        save(HERE / (policy + '_trace_check.json'), report)
        print(json.dumps({k:report.get(k) for k in ('policy','status','passed','captured_blocks')},ensure_ascii=False), flush=True)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), checker_source_sha256=before,
        source_unchanged=before==sha(__file__), independent_no_simulator_import=True,
        policies=reports, complete=all(r['status']=='complete' for r in reports),
        pending=[r['policy'] for r in reports if r['status']=='pending'],
        failed=[r['policy'] for r in reports if r['status']=='audit_failed'])
    save(HERE / 'trace_checks.json', result)
    if result['failed'] or not result['source_unchanged'] or (args.require_complete and not result['complete']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
