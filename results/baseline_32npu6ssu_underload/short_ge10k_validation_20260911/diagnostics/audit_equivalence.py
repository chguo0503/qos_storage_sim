#!/usr/bin/env python3
"""Audit the saved replay; never execute a simulation or alter strict evidence."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [{'field': path, 'reference': a, 'replay': b}]
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys() ^ b.keys())
        return [d for k in a for d in differences(a[k], b[k], f'{path}.{k}' if path else k)]
    if isinstance(a, list):
        assert len(a) == len(b), path
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f'{path}[{i}]')]
    return [] if a == b else [{'field': path, 'reference': a, 'replay': b}]


def main():
    strict_path = HERE / 'ordered_block_trace_audit.json'
    strict = read(strict_path)
    reference = read(strict['reference_result'])
    replay_path = HERE / 'ordered_trace_replay.json.gz'
    replay = read(replay_path)
    diffs = differences(reference, replay)
    allowed = {'wall_seconds', 'wall_seconds_total',
               'adapter_statistics.ack_ledger_wall_us',
               'adapter_statistics.collector_wall_us',
               'adapter_statistics.routing_wall_us'}
    found = {d['field'] for d in diffs}
    core = reference['core_and_policy_sha256']
    checks = {
        'only_five_elapsed_wall_clock_fields_differ': found == allowed,
        'entire_summary_exactly_equal': reference['summary'] == replay['summary'],
        'all_windows_exactly_equal': reference['windows'] == replay['windows'],
        'all_other_strict_checks_pass': all(v for k,v in strict['checks'].items()
                                          if k != 'adapter_statistics_exactly_equal'),
        'core29_hashes_match_original_and_current': len(core) == 29 and
            core == replay['core_and_policy_sha256'] and
            all(sha(ROOT / k) == v for k,v in core.items()),
        'strict_failure_preserved': strict['passed'] is False and
            strict['checks']['adapter_statistics_exactly_equal'] is False,
    }
    report = {'passed': all(checks.values()), 'checks': checks,
              'strict_full_record_equality': False,
              'physical_and_control_results_exactly_equal': all(checks.values()),
              'differences': diffs,
              'elapsed_timer_source': {
                  'adapter_statistics.collector_wall_us': 'shared_path_sim_adapter.py:155-164; perf_counter elapsed microseconds',
                  'adapter_statistics.routing_wall_us': 'shared_path_sim_adapter.py:187-211; perf_counter elapsed microseconds',
                  'adapter_statistics.ack_ledger_wall_us': 'shared_path_sim_adapter.py:221-228; perf_counter elapsed microseconds',
                  'wall_seconds': 'run_coflow_experiments.py:176-203; perf_counter elapsed seconds',
                  'wall_seconds_total': 'run_baseline_npu32_stress.py:292-315; perf_counter elapsed seconds'},
              'interpretation': 'Only host execution durations differ. Every simulated timestamp, request, layer, event/operation count, SSD statistic, configuration and source hash is identical. No second replay was run.',
              'strict_audit_sha256': sha(strict_path),
              'reference_result': strict['reference_result'],
              'reference_sha256': sha(strict['reference_result']),
              'replay_sha256': sha(replay_path),
              'trace_sha256': sha(HERE / 'ordered_block_trace.json.gz'),
              'script_sha256': sha(__file__)}
    (HERE / 'ordered_physical_equivalence_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'changed_fields': sorted(found)}))
    assert report['passed']


if __name__ == '__main__':
    main()
