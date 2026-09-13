#!/usr/bin/env python3
"""Five authorized no-trace jobs; audited inputs, at most three local workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'heterogeneity_confirmation_local_plan.json'
AUDIT = HERE / 'heterogeneity_confirmation_input_audit.json'
PREPARED = HERE / 'heterogeneity_S10_12_14_prepared_4seeds.jsonl'
NAME = 'hetero_S10_12_14_L384'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prepare_plan():
    assert not PLAN.exists()
    audited = read(AUDIT)
    assert audited['all_checks_passed'] and audited['no_builder_imported']
    assert audited['prepared_jsonl_sha256'] == sha(PREPARED)
    assert audited['selected_seeds'] == [19, 43, 67, 101]
    seed7_audit = HERE / 'heterogeneity_input_audit.json'
    old_audit = read(seed7_audit)
    assert old_audit['all_checks_passed']
    rows = [json.loads(line) for line in PREPARED.read_text().splitlines() if line.strip()]
    records = {int(r['label'].rsplit('_seed', 1)[1]): r for r in rows}
    assert set(records) == {19, 43, 67, 101}
    for c in audited['cases']:
        r = records[c['seed']]
        assert (ROOT / c['manifest']).resolve() == Path(r['manifest']).resolve()
        assert c['manifest_sha256'] == r['manifest_sha256'] == sha(r['manifest'])
        assert c['input_fingerprint'] == r['input_fingerprint']
        assert r['role_counts'] == {'L': 35, 'S1': 210, 'S2': 210, 'S3': 210}
    seed7_case = HERE / 'runs' / f'{NAME}_ssu8_h22000_seed7' / 'baseline'
    seed7_command = read(seed7_case / 'command.json')
    assert seed7_command['status'] == 'complete'
    assert sha(seed7_case / 'manifest.json.gz') == seed7_command['manifest_sha256']
    assert any(c['manifest_sha256'] == seed7_command['manifest_sha256'] for c in old_audit['cases'])
    protected = {str(HERE / n): sha(HERE / n) for n in
        ('experiment.py', 'analyze.py', 'prepare_heterogeneity.py', 'heterogeneity_math.json',
         'audit_heterogeneity_confirmation.py', 'audit_heterogeneity_inputs.py')}
    protected[str(Path(__file__))] = sha(Path(__file__))
    jobs = []
    # Start the slow Once job first, followed by the four predetermined seeds.
    for strategy, seed in [('once', 7)] + [('baseline', s) for s in (19, 43, 67, 101)]:
        manifest = seed7_case / 'manifest.json.gz' if seed == 7 else Path(records[seed]['manifest'])
        label = f'{NAME}_ssu8_h22000_seed{seed}'
        output = HERE / 'runs' / label / strategy
        assert not output.exists(), f'Never overwrite a prior case: {output}'
        job_id = f'hetero_short_seed{seed}_{strategy}'
        argv = [sys.executable, str(HERE / 'experiment.py'), '--manifest', str(manifest), '--strategy', strategy]
        assert '--trace' not in argv
        jobs.append(dict(job_id=job_id, seed=seed, strategy=strategy, label=label,
            manifest=str(manifest), manifest_sha256=sha(manifest), output=str(output), argv=argv,
            log=str(HERE / 'execution_logs' / (job_id + '.confirmation.log')),
            status=str(HERE / 'execution_logs' / (job_id + '.confirmation.status.json'))))
    plan = dict(created_utc=utc(), max_parallel=3, candidate=NAME, no_trace=True, no_ordered=True,
        only_authorized_jobs=True, additional_jobs_forbidden=True, no_retry_or_overwrite=True,
        description='Four additional predetermined Baseline seeds and the existing seed7 identical-manifest Once; no new seed7 Baseline.',
        target_completion_utc='2026-09-13T20:10:00+00:00', target_is_schedule_not_result=True,
        horizon_pure_compute_ms=22000, short_roles=['S1', 'S2', 'S3'],
        independent_audit=str(AUDIT), independent_audit_sha256=sha(AUDIT),
        prior_seed7_audit=str(seed7_audit), prior_seed7_audit_sha256=sha(seed7_audit),
        prepared_jsonl=str(PREPARED), prepared_jsonl_sha256=sha(PREPARED),
        protected_sources_sha256=protected, core_sources_sha256=audited['source_core_sha256'],
        prior_seed7_baseline_files_sha256={str(seed7_case/n):sha(seed7_case/n) for n in
            ('manifest.json.gz', 'result.json.gz', 'command.json', 'analysis.json')}, jobs=jobs)
    for name, digest in plan['core_sources_sha256'].items():
        assert sha(ROOT / name) == digest
    write(PLAN, plan)
    print(json.dumps(dict(plan=str(PLAN), plan_sha256=sha(PLAN), jobs=len(jobs)), ensure_ascii=False), flush=True)


def run_plan():
    plan = read(PLAN)
    plan_hash = sha(PLAN)
    assert plan['max_parallel'] == 3 and len(plan['jobs']) == 5
    assert sha(AUDIT) == plan['independent_audit_sha256'] and read(AUDIT)['all_checks_passed']
    assert sha(PREPARED) == plan['prepared_jsonl_sha256']

    def verify():
        assert sha(PLAN) == plan_hash
        for p, digest in plan['protected_sources_sha256'].items():
            assert sha(p) == digest
        for p, digest in plan['core_sources_sha256'].items():
            assert sha(ROOT / p) == digest
        for p, digest in plan['prior_seed7_baseline_files_sha256'].items():
            assert sha(p) == digest

    verify()
    for job in plan['jobs']:
        assert not Path(job['output']).exists()
        assert not Path(job['log']).exists() and not Path(job['status']).exists()
        assert sha(job['manifest']) == job['manifest_sha256']
    launch = HERE / 'heterogeneity_confirmation_local_launch.json'
    with launch.open('x') as f:
        json.dump(dict(status='running', pid=os.getpid(), started_utc=utc(), plan=str(PLAN),
                       plan_sha256=plan_hash, max_parallel=3), f, indent=2)
        f.write('\n')
    (HERE / 'execution_logs').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(job):
        state = dict(job_id=job['job_id'], seed=job['seed'], strategy=job['strategy'],
            plan_sha256=plan_hash, manifest_sha256=job['manifest_sha256'],
            runner_sha256=plan['protected_sources_sha256'][str(HERE/'experiment.py')],
            output=job['output'], started_utc=utc(), argv=job['argv'], status='starting')
        status = Path(job['status'])
        try:
            verify()
            assert not Path(job['output']).exists()
            assert sha(job['manifest']) == job['manifest_sha256']
            with Path(job['log']).open('x') as stream:
                process = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                state.update(pid=process.pid, status='running'); write(status, state)
                print(json.dumps(state), flush=True)
                code = process.wait()
                state.update(returncode=code, simulation_finished_utc=utc())
                assert code == 0, f'Simulation exit {code}; no automatic retry.'
                case = Path(job['output']); command = read(case / 'command.json')
                assert command['status'] == 'complete' and command['completed_simulation']
                assert command['observed_completed_blocks'] == command['expected_blocks'] == 42775040
                assert command['manifest_sha256'] == job['manifest_sha256'] == sha(case / 'manifest.json.gz')
                assert command['core_source_sha256'] == plan['core_sources_sha256']
                assert command['runner_sha256'] == state['runner_sha256']
                assert sha(case / 'result.json.gz') == command['output_sha256']
                assert sha(case / 'physical_service.json') == command['physical_service_sha256']
                assert command['trace_window_ms'] is None and not (case / 'trace.json.gz').exists()
                analysis_path = case / 'analysis.json'; assert not analysis_path.exists()
                argv = [sys.executable, str(HERE / 'analyze.py'), '--case', str(case), '--output', str(analysis_path)]
                for role in plan['short_roles']:
                    argv.extend(['--short-role', role])
                state.update(status='analyzing', analysis_argv=argv); write(status, state)
                analysis_code = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
                assert analysis_code == 0
            result = read(analysis_path)
            assert result['all_technical_checks_passed']
            selected = [w for w in result['windows'] if w['start_ms'] == 2000 and w['end_ms'] in (4000, 20000)]
            state.update(status='complete', analysis_sha256=sha(analysis_path), output_sha256=command['output_sha256'],
                command_sha256=sha(case / 'command.json'), physical_service_sha256=command['physical_service_sha256'],
                windows=[{k:w[k] for k in ('start_ms','end_ms','U_percent','all_32_active','mixed_card_count',
                    'long_short_mixed_card_count','long_short_mixed_pass')} for w in selected])
            verify()
        except BaseException as exc:
            state.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        state['ended_utc'] = utc(); write(status, state)
        print(json.dumps(state), flush=True)
        return state

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, plan['jobs']))
    verify()
    passed = all(r['status'] == 'complete' for r in results)
    summary = dict(all_completed=passed, plan_sha256=plan_hash, max_parallel=3, ended_utc=utc(), jobs=results)
    write(HERE / 'heterogeneity_confirmation_local_results.json', summary)
    state = read(launch); state.update(status='complete' if passed else 'failed', ended_utc=utc()); write(launch, state)
    assert passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-plan', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    assert args.prepare_plan != args.run
    prepare_plan() if args.prepare_plan else run_plan()
