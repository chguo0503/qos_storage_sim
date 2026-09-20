#!/usr/bin/env python3
"""Read complete original-Once controls and verify their paired OD inputs."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'results/continuous_underload_asu_od_20260918'


def read(path):
    with gzip.open(path,'rt') if str(path).endswith('.gz') else path.open() as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a,z,left,right):
    return max(0.0,min(z,right)-max(a,left))


def check_metrics(raw, analysis):
    left,right = analysis['start_ms'],analysis['end_ms']
    rows = raw['summary']['request_metrics']
    cohort = rows if left == 0 else [r for r in rows if left <= r['admission_time_ms'] < right]
    passed = sum(r['completion_time_ms']-r['admission_time_ms'] <= 1.5*r['own_compute_ms']+1e-9 for r in cohort)
    compute = math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],left,right)
                        for b in raw['summary']['microbatch_metrics'] for l in b['layer_metrics'])
    utilization = 100*compute/(32*(right-left))
    assert passed == analysis['slo']['passed'] and len(cohort) == analysis['slo']['count']
    assert abs(utilization-analysis['U_percent']) < 1e-7


def main():
    status = read(HERE/'under_once_queue_status.json')
    assert status['complete'] and status['all_successful'] and status['sources_verified_after']
    assert sha(HERE/'metrics.py') == sha(OLD/'metrics.py')
    od_cases = {}
    for path in (OLD/'runs').glob('*/command.json'):
        c = read(path)
        if not c.get('smoke') and c['policy'] == 'od_baseline' and c['order'] == 'random':
            assert c['seed'] not in od_cases
            od_cases[c['seed']] = path.parent
    assert set(od_cases) == {7,19,43}
    rows = []; cases = []; protected = {}
    for seed in (7,19,43):
        case = HERE/'runs'/f'under_once_seed{seed}_local'
        command = read(case/'command.json');raw = read(case/'result.json.gz')
        od_path = od_cases[seed];od_command = read(od_path/'command.json');od_raw = read(od_path/'result.json.gz')
        manifest = OLD/'inputs'/f'random_seed{seed}_ring_hash.json.gz'
        for p in (case/'command.json',case/'result.json.gz',case/'manifest.json.gz',
                  od_path/'command.json',od_path/'result.json.gz',od_path/'manifest.json.gz',manifest):
            protected[str(p.relative_to(ROOT))] = sha(p)
        assert command['status'] == od_command['status'] == 'complete'
        assert command['completed_simulation'] and command['core_unchanged'] and command['extension_unchanged']
        assert command['once_control']['all_checks_passed']
        assert command['once_control']['candidate_pool_extension_installed'] is False
        assert command['once_control']['candidate_pool'] == 'unchanged full category-legal pool'
        assert sha(case/'result.json.gz') == command['result_sha256']
        assert sha(od_path/'result.json.gz') == od_command['result_sha256']
        assert sha(case/'manifest.json.gz') == sha(od_path/'manifest.json.gz') == sha(manifest) == command['manifest_sha256']
        assert raw['input_fingerprint'] == od_raw['input_fingerprint'] == command['input_fingerprint']
        assert command['observed_blocks'] == command['expected_blocks'] == 3678208
        assert command['completed_requests'] == 640
        assert all(raw['summary']['invariants'].values()) and all(command['checks'].values())
        assert raw['prefetch_audit']['observed_cross_request_prefetches'] == 608
        assert command['core_source_sha256'] == od_command['core_source_sha256']
        for policy,result in (('once',raw),('od_baseline',od_raw)):
            for a in result['analysis']:check_metrics(result,a)
        for once,od in zip(raw['analysis'],od_raw['analysis']):
            name = 'full_population' if once['start_ms'] == 0 else ('warm_2_4s' if once['end_ms'] == 4000 else 'long_2_6s')
            assert once['start_ms'] == od['start_ms']
            if name != 'full_population':assert once['end_ms'] == od['end_ms']
            row = dict(seed=seed,window=name,once_U_percent=once['U_percent'],
                       once_slo_1p5_percent=once['slo']['percent'],once_slo_count=once['slo']['count'],once_slo_passed=once['slo']['passed'],
                       OD_U_percent=od['U_percent'],OD_slo_1p5_percent=od['slo']['percent'],
                       OD_slo_count=od['slo']['count'],OD_slo_passed=od['slo']['passed'],
                       delta_U_pp=once['U_percent']-od['U_percent'],delta_slo_pp=once['slo']['percent']-od['slo']['percent'],
                       once_all_npus_active=once['all_npus_active'],OD_all_npus_active=od['all_npus_active'],
                       once_strict_underload=once['demand']['strict_underload_all_disks'],OD_strict_underload=od['demand']['strict_underload_all_disks'],
                       manifest_sha256=command['manifest_sha256'],once_case=case.name,OD_case=od_path.name)
            for d in range(3):
                row.update({f'once_SSU{d}_mean_demand_GiB_s':once['demand']['per_disk_mean_GiB_s'][d],
                            f'once_SSU{d}_max_demand_GiB_s':once['demand']['per_disk_max_GiB_s'][d],
                            f'once_SSU{d}_actual_GiB_s':once['SSD_GiB_s'][d],
                            f'once_SSU{d}_at_or_above_capacity_percent':once['demand']['per_disk_at_or_above_capacity_percent'][d],
                            f'OD_SSU{d}_mean_demand_GiB_s':od['demand']['per_disk_mean_GiB_s'][d],
                            f'OD_SSU{d}_max_demand_GiB_s':od['demand']['per_disk_max_GiB_s'][d],
                            f'OD_SSU{d}_actual_GiB_s':od['SSD_GiB_s'][d]})
            rows.append(row)
        cases.append(dict(seed=seed,case=case.name,paired_OD_case=od_path.name,
                          manifest_sha256=command['manifest_sha256'],input_fingerprint=command['input_fingerprint'],
                          byte_identical_paired_input=True,original_once_verified=True,
                          completed_requests=640,completed_blocks=3678208,source_checks_passed=True))
    macro = []
    for window in ('warm_2_4s','long_2_6s','full_population'):
        selected = [r for r in rows if r['window'] == window]
        row = dict(window=window,seeds='7,19,43',seed_count=3)
        for key in ('once_U_percent','once_slo_1p5_percent','OD_U_percent','OD_slo_1p5_percent','delta_U_pp','delta_slo_pp'):
            row['mean_'+key] = math.fsum(r[key] for r in selected)/3
        macro.append(row)
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    for name,content in (('under_once_metrics.csv',rows),('under_once_macro.csv',macro)):
        with (HERE/name).open('x',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(content[0]));writer.writeheader();writer.writerows(content)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=True,
                  metric_clock='prefill completion minus request admission; SLO <= 1.5 * 8 * raw data layer compute',
                  total_completed_requests=1920,total_completed_blocks=11034624,
                  same_metric_source_as_previous_OD=True,original_once_full_pool=True,
                  source_sha256=protected,cases=cases,per_seed_windows=rows,macro=macro)
    with (HERE/'under_once_results.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(macro,indent=2))


if __name__ == '__main__':main()
