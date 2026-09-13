#!/usr/bin/env python3
"""Final immutable-result/source audit, to run before authorized cleanup."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path,'rt') as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    plan = read(HERE/'study_plan.json')
    original = plan['core_hashes']
    for name, expected in original.items():
        assert sha(ROOT/name) == expected, f'Original source changed: {name}'
    snapshot = HERE/'audit_remote_sources.tar.gz'
    with tarfile.open(snapshot,'r:gz') as archive:
        for name, expected in original.items():
            stream = archive.extractfile(name)
            assert stream and hashlib.sha256(stream.read()).hexdigest() == expected, name
    acceptances = {}
    for suffix in ['', '_once', '_constructed', '_context_scale',
                   '_load110_confirmation', '_load105_confirmation',
                   '_context384_confirmation', '_low_b_short', '_heterogeneity',
                   '_context384_once_confirmation']:
        file = HERE/f'audit_remote{suffix}_acceptance.json'
        assert file.exists() and read(file)['passed'], f'Remote queue not fully accepted: {file}'
        acceptances[file.name] = sha(file)
    supporting_audits = {}
    success_fields = {
        'audit_remote_context384_once_analysis.json':'all_four_analyzed',
        'audit_remote_concurrency_check.json':'passed',
        'audit_remote_final_sources.json':'passed',
        'heterogeneity_confirmation_local_acceptance.json':'all_checks_passed',
        'large_artifact_roundtrip_audit.json':'passed',
        'final_main_refresh.json':'passed',
    }
    for name, success_field in success_fields.items():
        file = HERE/name
        assert file.exists() and read(file).get(success_field) is True, f'Incomplete supporting audit: {name}'
        supporting_audits[name] = sha(file)
    records = []
    commands = sorted((HERE/'runs').glob('*/*/command.json'))
    commands += sorted((HERE/'long_horizon'/'context384'/'runs').glob('*/*/command.json'))
    expected_path = HERE/'expected_scientific_cases.json'
    expected_registry = read(expected_path)
    assert {str(p.parent.relative_to(HERE)) for p in commands} == set(expected_registry['cases'])
    for name, digest in expected_registry['remote_plan_sha256'].items():
        assert sha(HERE/name) == digest
    for key in ['additional_input_audit_sha256','additional_plan_sha256']:
        for name, digest in expected_registry.get(key,{}).items():
            assert sha(HERE/name) == digest
    long_cases = set()
    for name in ['reference_long65_plan.json','long_horizon/context384/plan.json']:
        long_plan = read(HERE/name)
        label = read(Path(long_plan['manifest']))['metadata']['label']
        base = Path(long_plan.get('runs_base', HERE/'runs'))
        long_cases.update(base/label/strategy for strategy in long_plan['strategies'])
    for file in commands:
        case = file.parent
        c = read(file)
        assert c['status']=='complete' and c['returncode']==0 and c['completed_simulation']
        assert c['strategy'] in ['baseline','once'] and c['order']=='random'
        assert c['observed_completed_blocks']==c['expected_blocks']
        assert sha(case/'result.json.gz')==c['output_sha256']
        assert sha(case/'physical_service.json')==c['physical_service_sha256']
        assert sha(case/'manifest.json.gz')==c['manifest_sha256']
        assert c['source_data_sha256']==original['data']
        assert c['runner_sha256']==sha(HERE/'experiment.py')
        for name, expected_sha in c['core_source_sha256'].items():
            assert sha(ROOT/name)==expected_sha
        manifest = read(case/'manifest.json.gz')
        assert manifest['metadata']['input_fingerprint']==c['input_fingerprint']
        analysis_path = case/('extended_analysis.json' if case in long_cases else 'analysis.json')
        a = read(analysis_path)
        assert a['all_technical_checks_passed'] and a['builder_sha256']==sha(HERE/'analyze.py')
        assert a['input_fingerprint']==c['input_fingerprint']
        assert a['num_npu']==32 and a['n_layers']==8 and a['order']=='random'
        for name, digest in a['sources'].items():
            assert Path(name).resolve().parent == case.resolve()
            assert sha(Path(name)) == digest, f'Stale analysis source: {name}'
        if case in long_cases:
            bins = [w for w in a['windows'] if w['end_ms']-w['start_ms']==2000]
            assert sorted(w['start_ms'] for w in bins) == list(range(2000,60000,2000))
            full = next(w for w in a['windows'] if (w['start_ms'],w['end_ms'])==(2000,60000))
            assert abs(sum(w['U_percent'] for w in bins)/29-full['U_percent']) < 1e-8
        records.append({'case':str(case.relative_to(HERE)), 'strategy':c['strategy'],
                        'seed':a['seed'], 'num_ssu':a['num_ssu'],
                        'input_fingerprint':c['input_fingerprint'],
                        'completed_requests':a['completed_request_count'],
                        'completed_blocks':c['observed_completed_blocks'],
                        'command_sha256':sha(file),'analysis_sha256':sha(analysis_path),
                        'analysis_file':analysis_path.name,
                        'all_32_active_all_saved_windows':all(w['all_32_active'] for w in a['windows']),
                        'activity_failure_windows':[{'start_ms':w['start_ms'],'end_ms':w['end_ms']}
                                                    for w in a['windows'] if not w['all_32_active']],
                        'coverage_failure_windows':[{'start_ms':w['start_ms'],'end_ms':w['end_ms'],
                                                     'mixed_cards':w['long_short_mixed_card_count']}
                                                    for w in a['windows'] if w['long_short_mixed_card_count']<32]})
    # Every same-label policy pair must use the identical whole input, not just
    # the same profile names or random seed.
    grouped = {}
    for row in records:
        grouped.setdefault(str(Path(row['case']).parent),[]).append(row)
    pairs = 0
    for rows in grouped.values():
        if len(rows)>1:
            assert len({r['input_fingerprint'] for r in rows})==1
            pairs += 1
    replay_checks = {}
    for file in (HERE/'replays').glob('*/replay_checks.json'):
        assert read(file)['passed']
        replay_checks[str(file.relative_to(HERE))] = sha(file)
    for file in [HERE/'long_horizon_comparison.json', HERE/'long_horizon'/'context384'/'long_horizon_comparison.json']:
        long_summary = read(file)
        assert not long_summary['pending']
        assert len(long_summary['rows']) == 2*len(read(HERE/long_summary['rows'][0]['source'])['windows'])
        for row in long_summary['rows']:
            assert sha(HERE/row['source']) == row['source_sha256']
    comparison = read(HERE/'comparison.json')
    assert not comparison['pending_or_unanalyzed']
    assert {r['case'] for r in comparison['rows']} == {name for name in expected_registry['cases'] if name.startswith('runs/')}
    for row in comparison['rows']:
        assert sha(HERE/row['case']/'analysis.json') == row['analysis_sha256']
    key_results = read(HERE/'key_results.json')
    assert not key_results['pending_window_rows']
    assert key_results['completed_window_rows']==key_results['expected_window_rows']==72
    assert key_results['generator_sha256']==sha(HERE/'build_key_results.py')
    tests = read(HERE/'retained_tests_check.json')
    assert tests['returncode']==0
    assert all(sha(ROOT/name)==expected for name,expected in tests['source_sha256'].items())
    audit = {'passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),
             'scope':'Completed scientific cases only; exact replay and interrupted attempt are separate evidence, not new seeds.',
             'original_sources_equal_count':len(original),'source_snapshot_sha256':sha(snapshot),
             'completed_scientific_cases':len(records),'paired_identical_inputs':pairs,
             'expected_case_registry_sha256':sha(expected_path),
             'comparison_sha256':sha(HERE/'comparison.json'),
             'key_results_sha256':sha(HERE/'key_results.json'),
             'all_random_no_new_ordered':True,'all_coverage_failures_retained':True,
             'remote_acceptance_sha256':acceptances,'exact_replay_checks':replay_checks,
             'supporting_audit_sha256':supporting_audits,
             'retained_test_file_hashes_still_equal':True,'records':records,
             'builder_sha256':sha(Path(__file__)),
             'limit':'Technical validity is not proof of hardware realism, strict instantaneous underload, or all-card mixed coverage in every short window.'}
    (HERE/'final_research_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['passed','completed_scientific_cases','paired_identical_inputs','original_sources_equal_count']}))


if __name__=='__main__':
    main()
