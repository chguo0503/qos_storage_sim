#!/usr/bin/env python3
"""Compare the same frozen Baseline run across local and isolated remote Python."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
LOCAL = HERE / 'runs/main512_ssu3_h22000_seed7/baseline'
REMOTE = HERE / 'runtime_validation/remote_main512'

# Explicit observational fields only. Modeled CPU/control latency is NOT excluded.
HOST_FIELDS = {
    'wall_seconds', 'wall_seconds_total',
    'adapter_statistics.ack_ledger_wall_us',
    'adapter_statistics.assignment_wall_us',
    'adapter_statistics.collector_wall_us',
    'adapter_statistics.reorder_wall_us',
    'adapter_statistics.routing_wall_us',
    'adapter_statistics.jit_prefetch.decision_wall_us',
}
ENV_FIELDS = {'python_version', 'manifest_path'}


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b, path='', differences=None, exempt=None):
    differences = differences if differences is not None else []
    exempt = exempt if exempt is not None else []
    if path in HOST_FIELDS | ENV_FIELDS:
        if a != b:
            exempt.append({'field': path, 'local': a, 'remote': b,
                           'kind': 'observed_wall_time' if path in HOST_FIELDS else 'explicit_host_environment'})
        return differences, exempt
    if type(a) is not type(b):
        differences.append({'field': path, 'local': a, 'remote': b, 'kind': 'type'})
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            differences.append({'field': path, 'kind': 'keys',
                                'local_only': sorted(a.keys()-b.keys()),
                                'remote_only': sorted(b.keys()-a.keys())})
        for key in a.keys() & b.keys():
            compare(a[key], b[key], path+'.'+key if path else key, differences, exempt)
    elif isinstance(a, list):
        if len(a) != len(b):
            differences.append({'field': path, 'kind': 'length', 'local': len(a), 'remote': len(b)})
        for index, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f'{path}[{index}]', differences, exempt)
    elif a != b:
        differences.append({'field': path, 'local': a, 'remote': b, 'kind': 'value'})
    return differences, exempt


def main():
    setup = read(HERE / 'audit_remote_setup.json')
    local, remote = read(LOCAL / 'result.json.gz'), read(REMOTE / 'result.json.gz')
    lc, rc = read(LOCAL / 'command.json'), read(REMOTE / 'command.json')
    assert lc['status'] == rc['status'] == 'complete'
    assert sha(LOCAL/'manifest.json.gz') == sha(REMOTE/'manifest.json.gz')
    assert lc['manifest_sha256'] == rc['manifest_sha256'] == sha(LOCAL/'manifest.json.gz')
    assert lc['input_fingerprint'] == rc['input_fingerprint'] == local['input_fingerprint'] == remote['input_fingerprint']
    assert lc['core_source_sha256'] == rc['core_source_sha256'] == local['core_and_policy_sha256'] == remote['core_and_policy_sha256']
    assert lc['runner_sha256'] == rc['runner_sha256'] == local['experiment_runner_sha256'] == remote['experiment_runner_sha256']
    assert sha(LOCAL/'result.json.gz') == lc['output_sha256']
    assert sha(REMOTE/'result.json.gz') == rc['output_sha256']
    assert sha(LOCAL/'physical_service.json') == lc['physical_service_sha256']
    assert sha(REMOTE/'physical_service.json') == rc['physical_service_sha256']
    assert remote['manifest_path'] == setup['remote_dir'] + '/' + str(LOCAL.relative_to(HERE.parents[1])) + '/manifest.json.gz'
    differences, exemptions = compare(local, remote)
    physical_differences, physical_exemptions = compare(read(LOCAL/'physical_service.json'), read(REMOTE/'physical_service.json'))
    assert not physical_exemptions
    report = {
        'same_manifest_bytes': True, 'same_input_fingerprint': True,
        'same_core_and_runner_hashes': True,
        'result_scientific_differences': differences,
        'physical_service_differences': physical_differences,
        'explicit_runtime_environment_differences': exemptions,
        'float_comparison': 'Exact Python value equality, no numeric tolerance; dict key order ignored.',
        'all_scientific_fields_equal': not differences and not physical_differences,
        'local_command_sha256': sha(LOCAL/'command.json'),
        'remote_command_sha256': sha(REMOTE/'command.json'),
        'local_result_sha256': sha(LOCAL/'result.json.gz'),
        'remote_result_sha256': sha(REMOTE/'result.json.gz'),
        'local_python': local['python_version'], 'remote_python': remote['python_version'],
        'local_wall_seconds': lc['wall_seconds'], 'remote_wall_seconds': rc['wall_seconds'],
        'windows': lc['windows'],
        'note': 'All serialized scientific fields match exactly, including event counts, every request/layer metric and fixed-bin physical service. No full block trace was retained, so this does not assert equality of an unrecorded per-block event stream. Matching one workload supports reuse for this study, not all future workloads or runtimes.',
    }
    target = HERE/'runtime_validation/main512_cross_host_checks.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('all_scientific_fields_equal', 'local_wall_seconds', 'remote_wall_seconds', 'windows')}, ensure_ascii=False))
    if differences or physical_differences:
        print(json.dumps({'result_difference_count': len(differences), 'physical_difference_count': len(physical_differences),
                          'first_differences': differences[:12]+physical_differences[:12]}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
