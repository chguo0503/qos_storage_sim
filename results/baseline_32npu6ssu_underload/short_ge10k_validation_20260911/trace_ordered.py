#!/usr/bin/env python3
"""One frozen Baseline replay with a passive block-completion observer.

The observer only copies already-recorded flow fields into a Python list. It
does not change simulation objects, RNG, queues, events, or timestamps.
"""
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
SOURCE = STUDY / 'raw_role_followup_20260909' / 'mixed_rebinding'
LABEL = 'raw176_extendedhot_ordered_seed7'
MANIFEST = SOURCE / 'inputs' / f'{LABEL}.json.gz'
OUT = HERE / 'diagnostics'
LEFT, RIGHT = 1980.0, 2030.0
COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
           'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
           'link_start_ms', 'link_end_ms']
sys.path.insert(0, str(ROOT))
import continuous_batch_sim as native
from run_baseline_npu32_stress import load_manifest, run_case


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'wt') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=None if str(path).endswith('.gz') else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.parse_args()
    references = list((SOURCE / 'runs' / LABEL / 'baseline').glob('*.json.gz'))
    assert len(references) == 1
    reference_path = references[0]
    reference = read(reference_path)
    assert reference['strategy'] == 'baseline' and reference['submit_seed'] == 7
    expected_sources = reference['core_and_policy_sha256']
    assert len(expected_sources) == 29
    assert all(sha(ROOT / name) == value for name, value in expected_sources.items())
    assert sha(ROOT / 'run_baseline_npu32_stress.py') == reference['stress_runner_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    command_path = OUT / 'replay_command.json'
    if command_path.exists():
        raise FileExistsError('One-shot replay record already exists; do not silently rerun.')
    requests, metadata = load_manifest(MANIFEST)
    rows, calls = [], 0
    original = native._register_complete

    def observe(context, flow):
        nonlocal calls
        if flow.enqueue_time < RIGHT and flow.link_end_time >= LEFT:
            rows.append([flow.request_id, flow.npu_id, flow.layer, flow.block_idx,
                         flow.disk_id, flow.queue_id, flow.total_gb, flow.block_count,
                         flow.enqueue_time, flow.ssd_activation_time,
                         flow.link_enqueue_time, flow.link_start_time, flow.link_end_time])
        calls += 1
        return original(context, flow)

    command = {'status': 'running', 'pid': os.getpid(), 'argv': sys.argv,
               'cwd': os.getcwd(), 'python': sys.version, 'executable': sys.executable,
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'observer_source_sha256': sha(__file__), 'manifest': str(MANIFEST),
               'manifest_sha256': sha(MANIFEST), 'reference_result': str(reference_path),
               'reference_sha256': sha(reference_path), 'core_source_sha256': expected_sources,
               'stress_runner_sha256': reference['stress_runner_sha256'],
               'capture_ms': [LEFT, RIGHT], 'authorized_scope': 'Exactly one Baseline replay; seed 7; frozen input.'}
    write(command_path, command)
    print(json.dumps({'status': 'running', 'pid': os.getpid(), 'capture_ms': [LEFT, RIGHT]}), flush=True)
    started = time.perf_counter()
    try:
        with patch.object(native, '_register_complete', observe):
            result = run_case(requests, metadata, strategy='baseline', assignment='fixed',
                              windows=((2000.0, 4000.0),))
        result['manifest_path'] = str(MANIFEST)
        checks = {'entire_summary_exactly_equal': result['summary'] == reference['summary']}
        checks.update({f'{key}_exactly_equal': result.get(key) == reference.get(key)
                       for key in reference if key not in ('summary', 'wall_seconds', 'wall_seconds_total')})
        checks.update({
            'all_29_sources_unchanged': all(sha(ROOT / k) == v for k, v in expected_sources.items()),
            'stress_runner_unchanged': sha(ROOT / 'run_baseline_npu32_stress.py') == reference['stress_runner_sha256'],
            'manifest_unchanged': sha(MANIFEST) == command['manifest_sha256'],
            'reference_unchanged': sha(reference_path) == command['reference_sha256'],
            'observer_called_for_every_block': calls == result['summary']['completed_blocks'],
            'all_captured_blocks_path0': all(r[5] == 0 for r in rows),
            'one176KiB_per_block': all(r[7] == 1 and r[6] == 176 * 1024 / 2**30 for r in rows),
            'physical_ssd_durations': all(abs(r[10] - r[9] - 1000*r[6]/40) < 1e-7 for r in rows),
            'physical_link_durations': all(abs(r[12] - r[11] - 1000*r[6]/50) < 1e-7 for r in rows),
            'physical_stage_order': all(r[8] <= r[9]+1e-8 and r[9] < r[10] and
                                        r[10] <= r[11]+1e-8 and r[11] < r[12] for r in rows),
            'nonempty_capture': len(rows) > 0,
        })
        audit = {'passed': all(checks.values()), 'checks': checks, **command,
                 'status': 'complete' if all(checks.values()) else 'audit_failed',
                 'wall_seconds': time.perf_counter()-started, 'captured_blocks': len(rows),
                 'total_completed_blocks': calls,
                 'observer_semantics': 'Copies completed flow fields only; original register callback called once; no added simulation events.',
                 'capture_rule': 'enqueue < 2030ms and HBM completion >= 1980ms; includes crossing intervals.',
                 'ssd_service_definition': 'ssd_activation_time -> link_enqueue_time (SSD completion)',
                 'link_service_definition': 'link_start_time -> link_end_time',
                 'equality_exclusions': ['wall_seconds', 'wall_seconds_total']}
        write(OUT / 'ordered_trace_replay.json.gz', result)
        write(OUT / 'ordered_block_trace.json.gz', {'audit': audit, 'columns': COLUMNS, 'rows': rows})
        write(OUT / 'ordered_block_trace_audit.json', audit)
        command.update(status=audit['status'], returncode=0 if audit['passed'] else 1,
                       wall_seconds=audit['wall_seconds'], finished_utc=datetime.now(timezone.utc).isoformat())
        write(command_path, command)
        print(json.dumps({'status': audit['status'], 'wall_seconds': audit['wall_seconds'],
                          'blocks': len(rows), 'failed_checks': [k for k,v in checks.items() if not v]}), flush=True)
        if not audit['passed']:
            raise AssertionError('Replay differs: ' + ', '.join(k for k,v in checks.items() if not v))
    except BaseException as exc:
        command.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                       returncode=1, wall_seconds=time.perf_counter()-started)
        write(command_path, command)
        raise


if __name__ == '__main__':
    main()
