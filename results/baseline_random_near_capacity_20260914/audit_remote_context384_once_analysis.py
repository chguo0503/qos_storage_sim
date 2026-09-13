#!/usr/bin/env python3
"""Analyze only the four accepted remote context384 Once confirmations.

No simulation, no manifest/source mutation, and no writes to other case families.
The analyzer and selection plan hashes are pinned before any result is available.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN_SHA256 = 'dadd4dccc042e57d143da9016661260ddc70e2624fc1b993c33d5ea3433c5bb1'
ANALYZER_SHA256 = '3cc2c8459ea705b524a6f202f04a7524347056fd269d0dde8904117c755c7947'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan_path = HERE / 'audit_remote_context384_once_confirmation_plan.json'
    analyzer = HERE / 'analyze.py'
    assert sha(plan_path) == PLAN_SHA256
    assert sha(analyzer) == ANALYZER_SHA256
    plan = read(plan_path)
    assert sorted(job['seed'] for job in plan['jobs']) == [19, 43, 67, 101]
    assert all(job['strategy'] == 'once' for job in plan['jobs'])
    imported_path = HERE / 'audit_remote_context384_once_confirmation_sync_record.json'
    imported = read(imported_path)['imported'] if imported_path.exists() else {}
    record_path = HERE / 'audit_remote_context384_once_analysis.json'
    record = read(record_path) if record_path.exists() else dict(
        scope='Only four remote context384 Once confirmations; original Baseline and all other cases are untouched.',
        selection_plan_sha256=PLAN_SHA256, analyzer_sha256=ANALYZER_SHA256,
        verifier_sha256=sha(__file__), cases={})
    assert record['selection_plan_sha256'] == PLAN_SHA256
    assert record['analyzer_sha256'] == ANALYZER_SHA256
    assert record['verifier_sha256'] == sha(__file__)
    newly_analyzed = []
    for job in plan['jobs']:
        label = job['label']
        if label not in imported:
            continue
        case = HERE / 'runs' / label / 'once'
        output = case / 'analysis.json'
        source_names = ('manifest.json.gz', 'command.json', 'result.json.gz', 'physical_service.json')
        before = {name: sha(case / name) for name in source_names}
        assert before['manifest.json.gz'] == job['manifest_sha256'] == imported[label]['manifest_sha256']
        assert before['result.json.gz'] == imported[label]['result_sha256']
        assert before['command.json'] == imported[label]['command_sha256']
        assert before['physical_service.json'] == imported[label]['physical_service_sha256']
        argv = [sys.executable, str(analyzer), '--case', str(case), '--output', str(output), '--short-role', 'S']
        execution_mode = 'previously_recorded_and_verified'
        if label not in record['cases'] and not output.exists():
            completed = subprocess.run(argv, cwd=ROOT, check=True, capture_output=True, text=True)
            execution = json.loads(completed.stdout)
            assert execution['technical_passed'] is True
            newly_analyzed.append(label)
            execution_mode = 'executed_frozen_analyzer'
        elif label not in record['cases']:
            # Recover a valid analyzer artifact if a later wrapper check failed.
            # Its exact source map and analyzer SHA are verified below; never
            # overwrite the existing analysis or infer that it was freshly run.
            execution_mode = 'reused_existing_verified_analysis'
        else:
            assert sha(output) == record['cases'][label]['analysis_sha256']
        analysis = read(output)
        command = read(case / 'command.json')
        assert analysis['all_technical_checks_passed'] is True and analysis['no_new_simulation'] is True
        assert analysis['strategy'] == 'once' and analysis['seed'] == job['seed']
        assert analysis['builder_sha256'] == ANALYZER_SHA256
        assert analysis['sources'] == {str(case / name): digest for name, digest in before.items()}
        assert analysis['input_fingerprint'] == job['input_fingerprint'] == command['input_fingerprint']
        assert analysis['short_roles'] == ['S'] and analysis['num_npu'] == 32 and analysis['num_ssu'] == 8
        windows = []
        for reference in command['windows']:
            matches = [row for row in analysis['windows']
                       if (row['start_ms'], row['end_ms']) == (reference['start_ms'], reference['end_ms'])]
            assert len(matches) == 1
            row = matches[0]
            assert math.isclose(row['U_percent'], 100 * reference['U'], rel_tol=0, abs_tol=1e-7)
            windows.append({key: row[key] for key in
                            ('start_ms', 'end_ms', 'U_percent', 'all_32_active', 'long_short_mixed_card_count')})
        assert before == {name: sha(case / name) for name in source_names}
        assert sha(analyzer) == ANALYZER_SHA256 and sha(plan_path) == PLAN_SHA256
        if label not in record['cases']:
            record['cases'][label] = dict(argv=argv, verified_utc=datetime.now(timezone.utc).isoformat(),
                execution_mode=execution_mode,
                analysis_sha256=sha(output), scientific_source_sha256=before, windows=windows,
                technical_checks_passed=True, sources_unchanged=True)
        record['all_four_analyzed'] = len(record['cases']) == 4
        temporary = record_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(record_path)
    print(json.dumps(dict(newly_analyzed=newly_analyzed, analyzed_count=len(record['cases']),
                          all_four_analyzed=len(record['cases']) == 4), ensure_ascii=False))


if __name__ == '__main__':
    main()
