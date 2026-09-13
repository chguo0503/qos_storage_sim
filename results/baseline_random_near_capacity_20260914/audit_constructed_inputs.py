#!/usr/bin/env python3
"""Independent read-only audit of final fit* input manifests; never simulates."""
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
import math
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from authenticated_workload_inputs import load_authenticated_bw_table


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-8):
    assert math.isclose(float(a), float(b), abs_tol=tol, rel_tol=1e-11), (a, b)


def main():
    plan_path, builder = HERE/'constructed_candidates.json', HERE/'prepare_constructed.py'
    plan, plan_hash, builder_hash = read(plan_path), sha(plan_path), sha(builder)
    prepared_path = HERE/'constructed_prepared_seed7.jsonl'
    prepared = [json.loads(line) for line in prepared_path.read_text().splitlines() if line.strip()]
    assert len(prepared) == len(plan['selected_candidates']) == 8
    with redirect_stdout(io.StringIO()):
        table, provenance = load_authenticated_bw_table(32)
    points = sorted((k[0], v[1]/1000) for k, v in table.items() if k[1] == 128)
    xmean, ymean = statistics.mean(x for x, y in points), statistics.mean(y for x, y in points)
    slope = math.fsum((x-xmean)*(y-ymean) for x, y in points)/math.fsum((x-xmean)**2 for x, y in points)
    intercept = ymean-slope*xmean
    close(intercept, plan['fit']['intercept_ms'], 1e-12)
    close(slope, plan['fit']['slope_ms_per_K'], 1e-12)
    assert points == [(p['total_k'], p['C_ms']) for p in plan['fit']['source_points']]
    assert sha(ROOT/'data') == plan['fit']['source_data_sha256']
    preserved = dict(plan['preserved_files_sha256'])
    reference = read(HERE/'runs/main512_ssu3_h22000_seed7/baseline/command.json')
    preserved.update(reference['core_source_sha256'])
    preserved[str((HERE/'experiment.py').relative_to(ROOT))] = reference['runner_sha256']
    for path, digest in preserved.items():
        assert sha(ROOT/path) == digest, path
    report = dict(all_checks_passed=False, no_simulation=True,
                  methodology='Independent authenticated-data OLS and raw JSON row audit; builder never imported or executed.',
                  source_sha256={str(plan_path): plan_hash, str(builder): builder_hash,
                                 str(prepared_path): sha(prepared_path), str(Path(__file__)): sha(__file__)},
                  preserved_source_sha256=preserved,
                  fitted_intercept_ms=intercept, fitted_slope_ms_per_K=slope,
                  limitations=['Extrapolation below raw minimum 32K is not a 10/12/16K hardware measurement.',
                               'Whole-queue mixture and >=22s pure compute do not guarantee every card computes both roles during warm [2,4).',
                               'Ideal whole-population load near capacity does not imply instantaneous per-disk underload.'], cases=[])
    specs = {s['name']: s for s in plan['selected_candidates']}
    actual_paths = {p.resolve() for p in (HERE/'inputs').glob('fit*.json.gz')}
    assert actual_paths == {Path(p['manifest']).resolve() for p in prepared}
    for prepared_row in prepared:
        path = Path(prepared_row['manifest'])
        manifest_hash_before = sha(path)
        data = read(path)
        meta = data['metadata']
        spec = specs[meta['candidate']]
        assert meta['seed'] == 7 and meta['num_npu'] == 32 and meta['n_layers'] == 8
        assert meta['horizon_pure_compute_ms'] == 22000
        disks = meta['num_ssu']
        assert disks == spec['num_ssu'] and meta['disk_bw_gib_s'] == 40 and meta['npu_bw_gib_s'] == 50
        assert meta['constructed_builder_sha256'] == builder_hash and meta['constructed_plan_sha256'] == plan_hash
        assert meta['source_data_sha256'] == sha(ROOT/'data')
        assert meta['family'] == 'data_affine_extrapolation' and meta['constructed_profile'] and meta['constructed_short_profile']
        assert meta['order'] == meta['order_mode'] == 'random' and meta['last_arrival_ms'] == 0
        assert meta['compute_scale_actual'] == 1 and meta['measurement_window_ms'] == [2000., 4000.]
        profiles = {p['role']: p for p in meta['profiles']}
        assert set(profiles) == {'L', 'S'}
        long_raw = table[(spec['long_total_k'], spec['long_nql'])]
        assert (spec['short_total_k'], spec['short_nql']) not in table and spec['short_nql'] == 128
        expected_C = {'L': long_raw[1], 'S': (intercept+slope*spec['short_total_k'])*1000}
        for role, p in profiles.items():
            close(p['per_layer_compute_us'], expected_C[role])
            assert p['total_tokens'] == p['seq_len_k']*1024 >= 10240
            assert p['ssd_prefix_tokens'] == p['total_tokens']-p['nql'] and p['ssd_prefix_tokens'] % 128 == 0
            close(p['per_layer_kv_gib'], p['ssd_prefix_tokens']*1408/2**30, 1e-12)
            close(p['required_bandwidth_gibps'], p['per_layer_kv_gib']*1e6/expected_C[role])
            assert p['required_bandwidth_gibps'] < 50
        assert profiles['L']['construction']['method'] == 'direct_data_row'
        close(profiles['L']['source_equivalent_ttft_78_layers_ms'], long_raw[2])
        p = profiles['S']
        assert p['source_equivalent_ttft_78_layers_ms'] is None
        close(p['extrapolated_78_layer_pure_compute_ms'], 78*expected_C['S']/1000)
        assert p['construction']['method'] == 'affine_extrapolation_below_raw_minimum'
        assert p['construction']['measured_data_row'] is False
        assert p['construction']['candidate_plan_sha256'] == plan_hash
        assert p['construction']['model_provenance']['source_ttft_for_short'] is None
        assert p['construction']['compute_model_scale'] == 1
        counts = dict(zip(('L', 'S'), spec['counts']))
        cycle_C_ms = 8*math.fsum(counts[r]*expected_C[r] for r in ('L', 'S'))/1000
        repeats = math.ceil(22000/cycle_C_ms)
        assert repeats == meta['quota_cycles'] == spec['per_card_population']['repeats']
        assert meta['count_ratio'] == spec['counts']
        quota = {r: n*repeats for r, n in counts.items()}
        canonical_roles = ['L']*quota['L']+['S']*quota['S']
        by_npu = defaultdict(list)
        for row in data['requests']:
            by_npu[row['npu_id']].append(row)
        assert set(by_npu) == set(range(32))
        placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p) for p in data['placements']]
        placement_reprs = [repr(p) for p in placements]
        validated_placements = set()
        fingerprint = hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
        logical = hashlib.sha256(); logical.update(b'[')
        first_logical = True
        sorted_requests = sorted(data['requests'], key=lambda r: r['request_id'])
        assert len({r['request_id'] for r in sorted_requests}) == len(sorted_requests)
        # Matches the repository's repr-tuple fingerprint, caching immutable placement text only.
        for row in sorted_requests:
            q = row['load']
            prefix = repr((row['request_id'], row['npu_id'], row['arrival_time_ms'], q['category'], q['per_layer_us']))[:-1]
            fingerprint.update((prefix+', '+placement_reprs[row['placement_index']]+')').encode())
        for row in data['requests']:
            if not first_logical: logical.update(b',')
            first_logical = False
            value = {k: row[k] for k in ('request_id', 'npu_id', 'arrival_time_ms', 'load')}
            logical.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        logical.update(b']')
        assert fingerprint.hexdigest() == data['input_fingerprint'] == meta['input_fingerprint'] == prepared_row['input_fingerprint']
        assert logical.hexdigest() == meta['logical_input_fingerprint']
        per_card = []; disk_means = []; disk_maxima = []; role_sequence_hashes = []
        expected_blocks = 0
        for npu in range(32):
            rows = sorted(by_npu[npu], key=lambda r: r['request_id'])
            identities = list(range(len(canonical_roles)))
            random.Random(7+100003*npu).shuffle(identities)
            assert len(rows) == len(identities)
            role_seq = []
            for pos, (row, original) in enumerate(zip(rows, identities)):
                q = row['load']; role = canonical_roles[original]; p = profiles[role]
                rid = npu*1000000+pos
                assert row['request_id'] == q['request_id'] == rid
                assert row['npu_id'] == q['npu_id'] == npu
                assert q['generation'] == pos and q['original_request_id'] == npu*1000000+original
                assert q['role'] == role and q['profile_index'] == (0 if role == 'L' else 1)
                assert row['arrival_time_ms'] == q['arrival_time'] == q['arrival_ms'] == 0
                assert q['constructed_profile'] == (role == 'S') and q['profile_construction'] == p['construction']
                assert q['source_ttft_ms'] == p['source_equivalent_ttft_78_layers_ms']
                close(q['per_layer_us'], expected_C[role]); close(q['original_compute_us'], expected_C[role])
                close(q['per_layer_kv_gb'], p['per_layer_kv_gib'], 1e-12)
                assert q['padding_gib_per_layer'] == 0 and q['total_tokens'] == p['total_tokens']
                assert q['seq_len_k'] == p['seq_len_k'] and q['nql'] == p['nql'] and q['ssd_prefix_tokens'] == p['ssd_prefix_tokens']
                close(q['required_bw_input_gbps'], p['required_bandwidth_gibps'])
                index = row['placement_index']; placement = placements[index]
                key = (index, npu, role)
                if key not in validated_placements:
                    expected = tuple(((j+npu)%disks, 176*1024/2**30) for j in range(q['ssd_prefix_tokens']//128))
                    assert placement == (expected,)
                    validated_placements.add(key)
                expected_blocks += len(placement[0])*8
                role_seq.append(role)
            assert Counter(role_seq) == quota
            C_ms = math.fsum(q['load']['per_layer_us']*8/1000 for q in rows)
            assert C_ms >= 22000
            close(C_ms, repeats*cycle_C_ms)
            recorded = meta['per_npu_assignment'][npu]
            assert recorded['npu_id'] == npu and recorded['shuffle_seed'] == 7+100003*npu
            assert recorded['role_counts'] == quota and recorded['first_32_roles'] == role_seq[:32]
            close(recorded['pure_compute_ms'], C_ms)
            for role in ('L', 'S'): close(recorded['role_pure_compute_ms'][role], expected_C[role]*8*quota[role]/1000)
            rates = {r: [sum(1 for j in range(profiles[r]['ssd_prefix_tokens']//128) if (j+npu)%disks == s)*176*1024/2**30*1e6/expected_C[r] for s in range(disks)] for r in ('L', 'S')}
            disk_means.append([math.fsum(rates[r][s]*quota[r]*8*expected_C[r]/1000 for r in ('L', 'S'))/C_ms for s in range(disks)])
            disk_maxima.append([max(rates[r][s] for r in ('L', 'S')) for s in range(disks)])
            sequence_hash = hashlib.sha256(''.join(role_seq).encode()).hexdigest()
            role_sequence_hashes.append(sequence_hash)
            per_card.append(dict(npu=npu, requests=len(rows), role_counts=quota, pure_compute_ms=C_ms, independent_seed=7+100003*npu, role_sequence_sha256=sequence_hash))
        means = [math.fsum(v[s] for v in disk_means) for s in range(disks)]
        maxima = [math.fsum(v[s] for v in disk_maxima) for s in range(disks)]
        for a, b in zip(means, meta['time_weighted_per_ssu_nominal_gib_s']): close(a, b)
        for a, b in zip(maxima, meta['static_per_ssu_upper_bound_gib_s']): close(a, b)
        rho = math.fsum(means)/(40*disks)
        close(rho, meta['ideal_load_ratio']); close(rho, spec['rho_ideal'])
        assert expected_blocks == prepared_row['expected_blocks'] == spec['per_card_population']['total_blocks_all_32_npu']
        assert len(data['requests']) == meta['request_count'] == prepared_row['requests']
        assert len(set(role_sequence_hashes)) == 32
        assert meta['static_upper_bound_passes'] == all(v <= 40 for v in maxima)
        assert sha(path) == manifest_hash_before, 'Manifest changed during its read-only audit.'
        row = dict(candidate=spec['name'], manifest=str(path), manifest_sha256=manifest_hash_before, input_fingerprint=fingerprint.hexdigest(),
                   num_requests=len(data['requests']), expected_blocks=expected_blocks, num_ssu=disks,
                   short_C_ms=expected_C['S']/1000, short_V_MiB=profiles['S']['per_layer_kv_gib']*1024,
                   short_B_GiB_s=profiles['S']['required_bandwidth_gibps'], ideal_load_ratio=rho,
                   mean_nominal_per_disk_GiB_s=means, static_worst_case_per_disk_GiB_s=maxima,
                   distinct_role_sequences=32, all_card_pure_compute_ms_at_least_22000=True,
                   all_synthetic_source_ttft_null=True, all_rows_and_placements_checked=True, per_card=per_card)
        report['cases'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('per_card', 'static_worst_case_per_disk_GiB_s')}, ensure_ascii=False), flush=True)
    assert sha(builder) == builder_hash and sha(plan_path) == plan_hash
    for path, digest in preserved.items(): assert sha(ROOT/path) == digest
    report['all_checks_passed'] = True
    (HERE/'constructed_input_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('All eight final manifests passed independent input and source-integrity audit.', flush=True)


if __name__ == '__main__':
    main()
