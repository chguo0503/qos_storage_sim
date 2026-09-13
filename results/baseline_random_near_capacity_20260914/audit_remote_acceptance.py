#!/usr/bin/env python3
"""Independently recheck the imported remote queue without rerunning simulations."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Frozen source root; after cleanup, use an isolated extraction of audit_remote_sources.tar.gz.')
    source_root = parser.parse_args().source_root.resolve()
    plan = read(HERE / 'audit_remote_queue_plan.json')
    queue = read(HERE / 'audit_remote_queue_status.json')
    setup = read(HERE / 'audit_remote_setup.json')
    imported = read(HERE / 'audit_remote_sync_record.json')['imported']
    labels = {job['label'] for job in plan['jobs']}
    assert len(labels) == len(plan['jobs']) == queue['job_count'] == 21
    assert queue['plan_sha256'] == sha(HERE / 'audit_remote_queue_plan.json')
    assert queue['queue_runner_sha256'] == sha(HERE / 'audit_remote_queue.py')
    assert queue['complete'] == 21 and queue['failed'] == queue['running'] == 0
    assert queue['ended_utc'] and queue['source_hashes_verified_before']
    assert queue['source_hashes_verified_after'] and set(imported) == labels
    assert plan['max_parallel'] == queue['max_parallel'] == 6
    assert all(sha(source_root / path) == digest for path, digest in plan['frozen_source_sha256'].items())
    checked = []
    for job in plan['jobs']:
        label = job['label']
        directory = HERE / 'runs' / label / 'baseline'
        command = read(directory / 'command.json')
        result = read(directory / 'result.json.gz')
        physical = read(directory / 'physical_service.json')
        assert queue['jobs'][label]['status'] == command['status'] == 'complete'
        assert queue['jobs'][label]['returncode'] == command['returncode'] == 0
        assert command['completed_simulation']
        assert command['label'] == label and command['strategy'] == 'baseline'
        assert command['order'] == 'random' and command['trace_window_ms'] is None
        assert command['retained_blocks'] == 0
        assert command['host'] == queue['host'] == imported[label]['host']
        assert sha(directory / 'command.json') == queue['jobs'][label]['command_sha256'] == imported[label]['command_sha256']
        assert sha(directory / 'manifest.json.gz') == sha(ROOT / job['manifest']) == job['manifest_sha256'] == command['manifest_sha256']
        assert result['input_fingerprint'] == command['input_fingerprint'] == job['input_fingerprint']
        assert result['core_and_policy_sha256'] == command['core_source_sha256'] == setup['environment']['core_source_sha256']
        assert command['runner_sha256'] == plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/experiment.py']
        assert command['source_data_sha256'] == plan['frozen_source_sha256']['data']
        assert sha(directory / 'result.json.gz') == command['output_sha256'] == imported[label]['result_sha256']
        assert sha(directory / 'physical_service.json') == command['physical_service_sha256'] == imported[label]['physical_service_sha256']
        assert physical['observed_completed_blocks'] == physical['expected_blocks'] == command['observed_completed_blocks'] == command['expected_blocks']
        assert physical['baseline_nonzero_paths'] == 0
        assert result['summary']['invariants'] and all(value is True for value in result['summary']['invariants'].values())
        assert [(window['start_ms'], window['end_ms']) for window in command['windows']] == [(2000, 4000), (2000, 20000)]
        checked.append({'label': label, 'all_hashes_and_invariants_passed': True,
                        'observed_blocks': physical['observed_completed_blocks'],
                        'windows_all_active': [window['all_active'] for window in command['windows']],
                        'host': command['host'], 'python_version': command['python_version'],
                        'result_sha256': command['output_sha256']})
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'passed': True, 'case_count': len(checked),
              'source_file_count': len(plan['frozen_source_sha256']),
              'checked_source_root': str(source_root),
              'queue_plan_sha256': sha(HERE / 'audit_remote_queue_plan.json'),
              'ssh_target': setup['host'], 'runtime_hostname': queue['host'],
              'remote_sources_verified_before_and_after': True,
              'local_frozen_sources_verified_now': True,
              'note': 'No simulation rerun and no unretained per-block event comparison; checks cover saved artifacts, observed block conservation, and source/input identity.',
              'cases': checked}
    output = HERE / 'audit_remote_acceptance.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'case_count': len(checked), 'report': str(output)}))


if __name__ == '__main__':
    main()
