"""Independent raw-result comparison; no simulation or experiment metrics import."""
from pathlib import Path
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
SCHEDULED = HERE.parent
ROOT = SCHEDULED.parents[2]
NAME = 'abb_interp_unique_50'


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(left, right, start, end):
    return max(0., min(right, end) - max(left, start))


def main():
    manifest_path = SCHEDULED / 'inputs' / f'{NAME}.json.gz'
    manifest = read(manifest_path)
    source_requests = {r['request_id']: r for r in manifest['requests']}
    own_compute = {rid: 8 * r['load']['per_layer_us'] / 1000 for rid, r in source_requests.items()}
    paths = {7: SCHEDULED / 'formal' / f'{NAME}_od_baseline',
             19: HERE / f'{NAME}_od_seed19'}
    results = {}
    audit_rows = []
    windows = [(2000, 4000), (20000, 40000), (40000, 60000), (20000, 60000)]
    for seed, path in paths.items():
        command = read(path / 'command.json')
        assert command['status'] == 'complete'
        assert command['manifest_sha256'] == sha(manifest_path)
        if seed == 19:
            assert command['seed'] == 19 and command['od_queue_depth_per_ssu'] == 8192
            assert command['source_sha256_before'] == command['source_sha256_after']
            assert command['source_unchanged'] and command['input_bytes_unchanged']
            assert sha(path / 'manifest.json.gz') == sha(manifest_path)
            expected_sources = command['source_sha256_before']
            analysis = read(path / 'analysis.json.gz')['main_windows']
        else:
            assert command['depth_per_npu_per_ssu'] == 256
            assert command['source_and_artifacts_unchanged']
            expected_sources = command['source_and_artifact_sha256']
            analysis = read(path / 'analysis.json')['windows']
        assert all(sha(ROOT / p) == value for p, value in expected_sources.items())
        result = read(path / 'result.json.gz')
        assert result['input_fingerprint'] == manifest['input_fingerprint']
        raw = result['summary']
        assert all(raw['invariants'].values())
        assert raw['completed_blocks'] == 38630400 and len(raw['request_metrics']) == 4800
        assert raw['ssd_queue_depth']['per_npu_per_ssu_slots'] == 256
        requests = {r['request_id']: r for r in raw['request_metrics']}
        assert set(requests) == set(source_requests)
        assert all(math.isclose(r['own_compute_ms'], own_compute[rid], abs_tol=1e-8) for rid, r in requests.items())
        first_drain = min(max(r['completion_time_ms'] for r in requests.values() if r['npu_id'] == n) for n in range(32))
        measures = []
        for left, right in windows:
            compute = [0.] * 32
            active = [0.] * 32
            roles = [set() for _ in range(32)]
            for r in requests.values():
                active[r['npu_id']] += overlap(left, right, r['admission_time_ms'], r['completion_time_ms'])
            for b in raw['microbatch_metrics']:
                q = source_requests[b['member_request_ids'][0]]
                for layer in b['layer_metrics']:
                    amount = overlap(left, right, layer['compute_start_ms'], layer['compute_end_ms'])
                    compute[b['npu_id']] += amount
                    if amount > 0:
                        roles[b['npu_id']].add(q['load']['role'])
            cohort = [r for r in requests.values() if left <= r['admission_time_ms'] < right]
            passed = sum(r['completion_time_ms'] - r['admission_time_ms'] <= 1.5 * own_compute[r['request_id']] + 1e-9 for r in cohort)
            measured = dict(start_ms=left, end_ms=right, U_percent=100 * math.fsum(compute) / (32 * (right-left)),
                            SLO15=dict(count=len(cohort), passed=passed, percent=100 * passed / len(cohort)),
                            all_npus_active=all(abs(t - (right-left)) < 1e-7 for t in active),
                            mixed_cards=sum(r == {'A', 'B'} for r in roles))
            matching = [w for w in analysis if w['start_ms'] == left and w['end_ms'] == right]
            if matching:
                w = matching[0]
                assert abs(measured['U_percent'] - w['U_percent']) < 1e-8
                assert measured['SLO15'] == w['slo']
                assert measured['all_npus_active'] == w['all_npus_active']
                assert measured['mixed_cards'] == w['role_and_stall']['npus_with_A_and_B_compute']
            measures.append(measured)
        phases = []
        for k in range(50):
            tails = [requests[rid]['completion_time_ms'] for rid, q in source_requests.items()
                     if q['load']['cycle'] == k and q['load']['position'] == 2]
            phases.append(dict(cycle=k, end_spread_ms=max(tails)-min(tails),
                               max_E_error_ms=max(abs(t - (k+1)*1650) for t in tails)))
        audit_rows.append(dict(seed=seed, first_card_drains_ms=first_drain, makespan_ms=raw['makespan_ms'],
                               windows=measures, cycles=phases, result_sha256=sha(path / 'result.json.gz')))
        results[seed] = raw
    fields = ('compute_start_ms', 'compute_end_ms', 'compute_duration_ms',
              'io_start_time_ms', 'io_ready_time_ms', 'io_barrier_wait_ms')
    layer_maps = {seed: {(b['npu_id'], tuple(b['member_request_ids']), l['layer']): l
                        for b in raw['microbatch_metrics'] for l in b['layer_metrics']}
                  for seed, raw in results.items()}
    assert set(layer_maps[7]) == set(layer_maps[19])
    differences = {field: max(abs(layer_maps[7][key][field]-layer_maps[19][key][field])
                              for key in layer_maps[7]) for field in fields}
    comparison = dict(status='passed', requests_per_seed=4800, layers_per_seed=38400,
                      blocks_per_seed=38630400, manifest_sha256=sha(manifest_path),
                      identical_manifest_bytes=True,
                      all_request_metrics_exact=results[7]['request_metrics'] == results[19]['request_metrics'],
                      all_microbatch_layer_metrics_exact=results[7]['microbatch_metrics'] == results[19]['microbatch_metrics'],
                      max_layer_field_difference_ms=differences, results=audit_rows,
                      interpretation='Seed changes only same-timestamp client submission order. Exact OD timelines would show no effective timing perturbation in this test; this is not evidence of robustness to compute or SSD service noise.')
    (HERE / 'independent_audit.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps({k:v for k,v in comparison.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
