#!/usr/bin/env python3
"""Independent raw-log crosscheck; does not import the study analyzer."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
SEEDS = [7, 19, 43, 67, 101]
WINDOWS = [(2000., 4000.), (2000., 20000.)]
TOL = 1e-7


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
        return json.load(f)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(a, b):
    assert math.isclose(a, b, abs_tol=TOL, rel_tol=1e-10), (a, b)


def clipped_sum(intervals, left, right):
    clipped = sorted((max(left, a), min(right, z)) for a, z in intervals
                     if z > left and a < right)
    for (a, z), (b, y) in zip(clipped, clipped[1:]):
        assert z <= b + TOL, 'Intervals overlap.'
    return math.fsum(z - a for a, z in clipped)


def summarize(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values),
                minimum=min(values), maximum=max(values), seed_values=values)


def main():
    output = dict(method='Independent clipping of raw compute intervals; admission-to-completion SLO proxy.',
                  analyzer_imported=False, no_simulations=True,
                  warning='Five-seed means retain every preselected seed. Do not discard failed warm mixture seeds to improve results.',
                  seed_order=SEEDS, cases=[], groups={}, source_sha256={str(Path(__file__).resolve()): digest(Path(__file__))})
    for candidate in ('main512', 'main1024'):
        cases = []
        for seed in SEEDS:
            case = HERE / 'runs' / f'{candidate}_ssu3_h22000_seed{seed}' / 'baseline'
            files = [case / n for n in ('manifest.json.gz', 'result.json.gz', 'command.json', 'analysis.json')]
            man, raw, command, expected = map(read, files)
            assert command['status'] == 'complete'
            assert command['manifest_sha256'] == digest(files[0])
            assert command['output_sha256'] == digest(files[1])
            output['source_sha256'].update({str(f): digest(f) for f in files})
            req = {r['request_id']: r for r in man['requests']}
            compute = {n: {role: [] for role in ('L', 'S')} for n in range(32)}
            active = {n: [] for n in range(32)}
            for batch in raw['summary']['microbatch_metrics']:
                assert len(batch['member_request_ids']) == 1
                request = req[batch['member_request_ids'][0]]
                npu, role = batch['npu_id'], request['load']['role']
                assert npu == request['npu_id']
                active[npu].append((batch['admission_time_ms'], batch['completion_time_ms']))
                compute[npu][role].extend((l['compute_start_ms'], l['compute_end_ms']) for l in batch['layer_metrics'])
            rows = []
            for left, right in WINDOWS:
                ref = next(w for w in expected['windows'] if w['start_ms'] == left and w['end_ms'] == right)
                per_card = []
                for npu in range(32):
                    role_ms = {r: clipped_sum(v, left, right) for r, v in compute[npu].items()}
                    total = clipped_sum(compute[npu]['L'] + compute[npu]['S'], left, right)
                    active_ms = clipped_sum(active[npu], left, right)
                    equal(total, sum(role_ms.values()))
                    equal(total, ref['per_npu'][npu]['compute_ms'])
                    equal(active_ms, ref['per_npu'][npu]['active_ms'])
                    for role, ms in role_ms.items():
                        equal(ms, ref['per_npu'][npu]['role_compute_ms'][role])
                    per_card.append(dict(npu=npu, role_compute_ms=role_ms, compute_ms=total,
                                         active_ms=active_ms, mixed=all(x > TOL for x in role_ms.values())))
                U = math.fsum(c['compute_ms'] for c in per_card) * 100 / (32 * (right - left))
                equal(U, ref['U_percent'])
                mixed = sum(c['mixed'] for c in per_card)
                assert mixed == ref['long_short_mixed_card_count'] == ref['mixed_card_count']
                all_active = all(abs(c['active_ms'] - (right - left)) < TOL for c in per_card)
                assert all_active == ref['all_32_active']
                cohort = [r for r in raw['summary']['request_metrics'] if left <= r['admission_time_ms'] < right]
                slos = {}
                for alpha in (1.5, 2.):
                    stats = {}
                    for role in ('all', 'L', 'S'):
                        sample = cohort if role == 'all' else [r for r in cohort if req[r['request_id']]['load']['role'] == role]
                        passes = sum(r['completion_time_ms'] - r['admission_time_ms'] <= alpha * r['own_compute_ms'] + 1e-9 for r in sample)
                        rate = passes * 100 / len(sample) if sample else None
                        match = ref['slo']['alphas'][str(alpha)]['admission'] if role == 'all' else ref['slo']['alphas'][str(alpha)]['per_role'][role]
                        assert (len(sample), passes) == (match['count'], match['passed'])
                        if rate is not None: equal(rate, match['percent'])
                        stats[role] = dict(count=len(sample), passed=passes, percent=rate)
                    slos[str(alpha)] = stats
                rows.append(dict(start_ms=left, end_ms=right, U_percent=U, all_32_active=all_active,
                                 long_short_mixed_card_count=mixed, strict_warm_pass=all_active and mixed == 32,
                                 missing_roles_by_npu={str(c['npu']): [r for r, ms in c['role_compute_ms'].items() if ms <= TOL] for c in per_card if not c['mixed']},
                                 minimum_role_compute_ms={r: min(c['role_compute_ms'][r] for c in per_card) for r in ('L', 'S')},
                                 admission_prefill_proxy_slo=slos, per_npu=per_card))
            cases.append(dict(candidate=candidate, seed=seed, case=str(case), windows=rows))
        output['cases'].extend(cases)
        aggregate = []
        for index, (left, right) in enumerate(WINDOWS):
            windows = [c['windows'][index] for c in cases]
            aggregate.append(dict(start_ms=left, end_ms=right,
                                  U_percent=summarize([w['U_percent'] for w in windows]),
                                  strict_pass_seeds=[s for s, w in zip(SEEDS, windows) if w['strict_warm_pass']],
                                  mixed_card_counts=[w['long_short_mixed_card_count'] for w in windows],
                                  all_active_seed_count=sum(w['all_32_active'] for w in windows),
                                  admission_prefill_proxy_slo={str(a): {r: summarize([w['admission_prefill_proxy_slo'][str(a)][r]['percent'] for w in windows]) for r in ('all', 'L', 'S')} for a in (1.5, 2.)}))
        output['groups'][candidate] = aggregate
    output['all_checks_passed'] = True
    path = HERE / 'analysis_five_seed_crosscheck.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(all_checks_passed=True, groups=output['groups']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
