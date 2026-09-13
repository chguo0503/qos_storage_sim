#!/usr/bin/env python3
"""Run the established 5-ms Once policy on byte-identical Baseline inputs."""
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import argparse
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(ROOT), str(STUDY)]
import experiment as original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', choices=('random', 'ordered'), required=True)
    args = parser.parse_args()
    baseline = STUDY/'validation20s/runs'/f'ssu4_{args.order}_k1_sync_seed7'/'baseline'
    reference = original.read_json(baseline/'command.json')
    frozen = baseline/'manifest.json.gz'
    before = original.source_hashes()
    assert before == reference['core_source_sha256'], 'Core source differs from Baseline'
    assert original.sha(frozen) == reference['manifest_sha256']
    requests, metadata = original.load_manifest(frozen)
    assert metadata['num_npu'] == 32 and metadata['num_ssu'] == 4 and metadata['seed'] == 7
    assert metadata['order'] == args.order and metadata['n_layers'] == 8
    assert len(requests) == 3840 and all(q.arrival_time_ms == 0 for q in requests)
    assert Counter(q.npu_id for q in requests) == {n: 120 for n in range(32)}
    target = HERE/'runs'/args.order/'once'
    target.mkdir(parents=True, exist_ok=False)
    manifest = target/'manifest.json.gz'
    manifest.write_bytes(frozen.read_bytes())
    assert original.sha(manifest) == original.sha(frozen)
    result_path, trace_path = target/'result.json.gz', target/'trace.json.gz'
    windows = ((2000., 4000.), (2000., 20000.))
    record = dict(status='running', strategy='once', strategy_label='Once per layer',
                  assignment='fixed', order=args.order, pid=os.getpid(), argv=sys.argv,
                  started_utc=original.utc(), manifest=str(manifest),
                  python_executable=sys.executable, python_version=sys.version,
                  num_npu=32, num_ssu=4, seed=7,
                  manifest_sha256=original.sha(manifest),
                  case_manifest_sha256=original.sha(manifest),
                  baseline_command=str(baseline/'command.json'),
                  baseline_command_sha256=original.sha(baseline/'command.json'),
                  input_fingerprint=metadata['input_fingerprint'],
                  core_source_sha256=before, runner_sha256=original.sha(__file__),
                  source_data_sha256=original.sha(ROOT/'data'),
                  collector_interval_ms=5.0, trace_window_ms=[1800., 4200.],
                  windows=list(windows), completed_simulation=False)
    original.write_json(target/'command.json', record)
    start = last_progress = time.perf_counter()
    rows, observed = [], 0
    expected = sum(len(q.placement[0])*8 for q in requests)
    callback = original.native._register_complete

    def observe(context, flow):
        nonlocal observed, last_progress
        if ((flow.ssd_activation_time < 4200. and flow.link_enqueue_time > 1800.) or
                (flow.link_start_time < 4200. and flow.link_end_time > 1800.)):
            rows.append([flow.request_id, flow.npu_id, flow.layer, flow.block_idx,
                         flow.disk_id, flow.queue_id, flow.total_gb, flow.block_count,
                         flow.enqueue_time, flow.ssd_activation_time, flow.link_enqueue_time,
                         flow.link_start_time, flow.link_end_time])
        observed += 1
        if observed % 10000 == 0 and time.perf_counter()-last_progress >= 20:
            progress = dict(order=args.order, simulation_ms=context.current_time_ms,
                            completed_requests=context.completed_requests,
                            completed_blocks=observed, expected_blocks=expected,
                            wall_seconds=time.perf_counter()-start)
            original.write_json(target/'progress.json', progress)
            print(json.dumps(progress), flush=True)
            last_progress = time.perf_counter()
        return callback(context, flow)

    print(json.dumps(dict(event='start', order=args.order, strategy='once',
                          input_sha256=record['manifest_sha256'], expected_blocks=expected)), flush=True)
    try:
        with patch.object(original.native, '_register_complete', observe):
            result = original.run_case(requests, metadata, strategy='once',
                                       assignment='fixed', windows=windows)
        assert original.source_hashes() == before
        assert original.sha(__file__) == record['runner_sha256']
        assert original.sha(manifest) == record['manifest_sha256']
        assert result['core_and_policy_sha256'] == before
        assert result['input_fingerprint'] == metadata['input_fingerprint']
        assert all(result['summary']['invariants'].values())
        assert result['summary']['invariants']['all_requests_completed']
        assert observed == expected
        result['manifest_path'] = str(manifest)
        result['experiment_runner_sha256'] = record['runner_sha256']
        result['paired_baseline_command'] = str(baseline/'command.json')
        original.write_json(result_path, result)
        trace = dict(schema_version=1, columns=original.TRACE_COLUMNS, rows=rows,
                     window_ms=[1800., 4200.], strategy='once', completed_simulation=True,
                     source=dict(manifest=str(manifest), manifest_sha256=record['manifest_sha256'],
                                 reference_result=str(result_path), reference_sha256=original.sha(result_path),
                                 input_fingerprint=metadata['input_fingerprint'],
                                 core_source_sha256=before, observer_source_sha256=record['runner_sha256']),
                     capture_rule='Unmodified completed block fields if SSD or NPU-link service overlaps [1800,4200) ms; observe all block completions until complete finite drain',
                     checks=dict(all_expected_blocks_observed=observed == expected,
                                 original_invariants_passed=True, core_unchanged=True,
                                 identical_manifest_to_baseline=True,
                                 all_path_ids_valid=all(0 <= r[5] < 256 for r in rows)),
                     observed_completed_blocks=observed, expected_completed_blocks=expected,
                     retained_blocks=len(rows))
        assert all(trace['checks'].values())
        original.write_json(trace_path, trace)
        record.update(status='complete', returncode=0, completed_simulation=True,
                      output_sha256=original.sha(result_path), trace_sha256=original.sha(trace_path),
                      observed_completed_blocks=observed, retained_blocks=len(rows),
                      windows=[dict(start_ms=w['start_ms'], end_ms=w['end_ms'],
                                    U=w['mean_npu_utilization'],
                                    all_active=w['all_npus_active_whole_window']) for w in result['windows']])
    except BaseException as exc:
        record.update(status='failed', returncode=1, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(ended_utc=original.utc(), wall_seconds=time.perf_counter()-start)
        original.write_json(target/'command.json', record)
    print(json.dumps({k:record[k] for k in ('status','order','windows','wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
