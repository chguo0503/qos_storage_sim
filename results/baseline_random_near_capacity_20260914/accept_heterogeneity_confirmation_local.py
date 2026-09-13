#!/usr/bin/env python3
"""Read-only final acceptance for the five predeclared local confirmation jobs.

Does not import the input builder, runner, or analyzer and never launches jobs.
Recomputes each window's actual compute utilization from original layer logs.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import gzip
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / 'heterogeneity_confirmation_local_plan.json'
OUT = HERE / 'heterogeneity_confirmation_local_acceptance.json'
WINDOWS = ((2000., 4000.), (2000., 20000.))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
        return json.load(f)


def near(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), (a, b)


def clip(a, z, left, right):
    return max(0., min(z, right)-max(a, left))


def case_audit(case, expected_manifest_sha, plan, new_case):
    paths = {name:case/name for name in ('manifest.json.gz','result.json.gz','command.json',
                                       'analysis.json','physical_service.json')}
    hashes = {str(p):sha(p) for p in paths.values()}
    man, raw, command, analysis = [read(paths[n]) for n in ('manifest.json.gz','result.json.gz','command.json','analysis.json')]
    assert command['status'] == 'complete' and command['completed_simulation']
    assert command['manifest_sha256'] == expected_manifest_sha == hashes[str(paths['manifest.json.gz'])]
    assert command['output_sha256'] == hashes[str(paths['result.json.gz'])]
    assert command['physical_service_sha256'] == hashes[str(paths['physical_service.json'])]
    assert command['core_source_sha256'] == plan['execution_core_sources_sha256']
    assert command['runner_sha256'] == plan['protected_sources_sha256'][str(HERE/'experiment.py')]
    assert command['observed_completed_blocks'] == command['expected_blocks'] == 42775040
    if new_case:
        assert command['trace_window_ms'] is None and command['retained_blocks'] == 0
        assert not (case/'trace.json.gz').exists() and not (case/'trace.json.gz.tmp').exists()
    assert analysis['all_technical_checks_passed']
    assert analysis['builder_sha256'] == plan['protected_sources_sha256'][str(HERE/'analyze.py')]
    for p, digest in analysis['sources'].items():
        assert sha(p) == digest
    assert man['input_fingerprint'] == raw['input_fingerprint'] == analysis['input_fingerprint'] == command['input_fingerprint']
    assert raw['strategy'] == command['strategy'] == analysis['strategy']
    meta, summary = man['metadata'], raw['summary']
    assert meta['candidate'] == 'hetero_S10_12_14_L384'
    assert (summary['num_npu'],summary['num_ssu'],summary['n_layers'],summary['batch_size']) == (32,8,8,1)
    assert meta['horizon_pure_compute_ms'] == 22000
    assert meta['short_roles'] == ['S1','S2','S3'] == analysis['short_roles']
    assert all(p['construction']['method'].startswith('affine_extrapolation') for p in meta['profiles'])
    requests = {r['request_id']:r for r in man['requests']}
    assert len(requests) == len(man['requests']) == len(summary['microbatch_metrics']) == 21280
    lanes = defaultdict(list); seen = set()
    for batch in summary['microbatch_metrics']:
        assert len(batch['member_request_ids']) == 1
        rid = batch['member_request_ids'][0]; assert rid not in seen; seen.add(rid)
        assert batch['npu_id'] == requests[rid]['npu_id']
        q = requests[rid]['load']
        assert q['coarse_role'] == ('L' if q['role'] == 'L' else 'S')
        assert q['source_ttft_ms'] is None
        assert [l['layer'] for l in batch['layer_metrics']] == list(range(8))
        for layer in batch['layer_metrics']:
            near(layer['compute_end_ms']-layer['compute_start_ms'],q['per_layer_us']/1000)
        lanes[batch['npu_id']].append(batch)
    assert seen == set(requests) and set(lanes) == set(range(32))
    for lane in lanes.values():
        lane.sort(key=lambda b:b['admission_time_ms'])
        assert all(a['completion_time_ms'] <= b['admission_time_ms']+1e-7 for a,b in zip(lane,lane[1:]))
    windows = []
    for left,right in WINDOWS:
        per_npu = []
        for npu in range(32):
            compute, active, by_role = [], [], defaultdict(list)
            for batch in lanes[npu]:
                rid = batch['member_request_ids'][0]; role = requests[rid]['load']['role']
                active.append(clip(batch['admission_time_ms'],batch['completion_time_ms'],left,right))
                for layer in batch['layer_metrics']:
                    dt = clip(layer['compute_start_ms'],layer['compute_end_ms'],left,right)
                    compute.append(dt); by_role[role].append(dt)
            compute_ms, active_ms = math.fsum(compute), math.fsum(active)
            role_compute = {role:math.fsum(values) for role,values in by_role.items()}
            roles = sorted(role for role,dt in role_compute.items() if dt>1e-7)
            mixed = role_compute.get('L',0)>1e-7 and math.fsum(role_compute.get(r,0) for r in ('S1','S2','S3'))>1e-7
            per_npu.append(dict(npu=npu,compute_ms=compute_ms,active_ms=active_ms,U_percent=100*compute_ms/(right-left),
                actual_roles_computed=roles,long_short_mixed=mixed,role_compute_ms=role_compute))
        U = 100*math.fsum(p['compute_ms'] for p in per_npu)/(32*(right-left))
        active = all(abs(p['active_ms']-(right-left))<=1e-6 for p in per_npu)
        mixed_count = sum(p['long_short_mixed'] for p in per_npu)
        old = next(w for w in analysis['windows'] if (w['start_ms'],w['end_ms'])==(left,right))
        near(U,old['U_percent'])
        assert active == old['all_32_active']
        assert mixed_count == old['long_short_mixed_card_count']
        assert (mixed_count==32) == old['long_short_mixed_pass']
        for p in per_npu:
            saved = old['per_npu'][p['npu']]
            near(p['U_percent'],saved['U_percent'])
            near(p['compute_ms'],saved['compute_ms'])
            near(p['active_ms'],saved['active_ms'])
        windows.append(dict(start_ms=left,end_ms=right,U_percent=U,all_32_active=active,
            long_short_mixed_card_count=mixed_count,long_short_mixed_pass=mixed_count==32,
            all_fine_roles_mixed_card_count=sum(len(p['actual_roles_computed'])==4 for p in per_npu),per_npu=per_npu))
    return dict(all_technical_checks_passed=True,case=str(case),new_case=new_case,seed=meta['seed'],strategy=raw['strategy'],
        command_pid=command['pid'],command_host=command['host'],started_utc=command['started_utc'],ended_utc=command['ended_utc'],
        expected_blocks=command['expected_blocks'],observed_completed_blocks=command['observed_completed_blocks'],
        input_fingerprint=man['input_fingerprint'],manifest_sha256=expected_manifest_sha,
        sources_sha256=hashes,source_core_sha256=command['core_source_sha256'],windows=windows,
        no_trace=new_case or not (case/'trace.json.gz').exists())


def main():
    plan = read(PLAN); plan_hash = sha(PLAN)
    fix_path = HERE/'heterogeneity_confirmation_supervision_fix.json'
    fix = read(fix_path)
    assert fix['original_plan_sha256'] == plan_hash
    assert len(fix['execution_core_sources_sha256']) == 29
    assert all(plan['core_sources_sha256'][p]==d for p,d in fix['execution_core_sources_sha256'].items())
    assert sha(fix['replacement_supervisor']) == fix['replacement_supervisor_sha256']
    assert sha(HERE/'run_heterogeneity_confirmation_local.py') == fix['immutable_original_queue_source_sha256']
    plan = dict(plan,execution_core_sources_sha256=fix['execution_core_sources_sha256'])
    run_results_path = HERE/'heterogeneity_confirmation_takeover_results.json'
    run_results = read(run_results_path)
    assert not run_results['all_completed'] and run_results['plan_sha256'] == plan_hash
    assert run_results['fix_sha256'] == sha(fix_path)
    assert run_results['no_jobs_added']
    # The initial takeover checks passed, but its three attached attempts later
    # disappeared. Do not accept its historical preservation flag as a claim
    # that those attempts completed; bind the failed records and recovery.
    recovery_plan_path=HERE/'heterogeneity_confirmation_recovery_plan.json'
    recovery_plan=read(recovery_plan_path)
    recovery_results_path=HERE/'heterogeneity_confirmation_recovery_results.json'
    recovery_results=read(recovery_results_path)
    assert recovery_plan['original_plan_sha256']==plan_hash
    assert recovery_plan['previous_supervision_fix_sha256']==sha(fix_path)
    assert recovery_plan['root_authorized_recovery'] and recovery_plan['full_rerun_only']
    assert not recovery_plan['new_seeds_or_experiments_added']
    assert recovery_plan['max_total_heterogeneity_simulations']==5
    assert recovery_results['all_completed'] and recovery_results['interrupted_attempts_preserved']
    assert recovery_results['recovery_plan_sha256']==sha(recovery_plan_path)
    assert recovery_results['original_plan_sha256']==plan_hash and not recovery_results['new_seeds_or_experiments_added']
    assert sha(HERE/'recover_heterogeneity_interrupted_attempts.py')==recovery_plan['recovery_supervisor_sha256']
    assert plan['max_parallel'] == run_results['max_parallel'] == 3
    assert len(plan['jobs']) == len(run_results['jobs']) == 5
    assert {(j['strategy'],j['seed']) for j in plan['jobs']} == {('once',7),('baseline',19),('baseline',43),('baseline',67),('baseline',101)}
    registry_path=HERE/'expected_scientific_cases.json';registry=read(registry_path)
    assert registry['case_count']==len(registry['cases'])==len(set(registry['cases']))==90
    assert registry['additional_plan_sha256'][PLAN.name]==plan_hash
    assert not any('failed_attempts' in p or 'replays/' in p for p in registry['cases'])
    assert all(str(Path(j['output']).relative_to(HERE)) in registry['cases'] for j in plan['jobs'])
    assert sha(plan['independent_audit']) == plan['independent_audit_sha256']
    audit = read(plan['independent_audit']); assert audit['all_checks_passed']
    assert sha(HERE/'audit_heterogeneity_confirmation.py') == audit['generator_sha256']
    assert sha(plan['prior_seed7_audit']) == plan['prior_seed7_audit_sha256']
    assert sha(plan['prepared_jsonl']) == plan['prepared_jsonl_sha256'] == audit['prepared_jsonl_sha256']
    for p,digest in plan['protected_sources_sha256'].items():assert sha(p)==digest
    for p,digest in plan['core_sources_sha256'].items():assert sha(ROOT/p)==digest
    for p,digest in plan['prior_seed7_baseline_files_sha256'].items():assert sha(p)==digest
    original_states={j['job_id']:j for j in run_results['jobs']}
    recovered={j['job_id']:j for j in recovery_results['jobs']}
    recovery_jobs={j['job_id']:j for j in recovery_plan['jobs']}
    assert set(recovered)==set(recovery_jobs)=={j['job_id'] for j in plan['jobs'][:3]}
    assert all(original_states[key]['status']=='failed' for key in recovered)
    assert all(original_states[j['job_id']]['status']=='complete' for j in plan['jobs'][3:])
    jobs_by_id={**original_states,**recovered}
    cases = []
    failed_attempts=[]
    for job in plan['jobs']:
        recorded = jobs_by_id[job['job_id']]
        state_path = Path(recorded['status_path'])
        state = read(state_path)
        assert state == recorded and state['status']=='complete' and state['returncode']==0
        assert state['manifest_sha256'] == job['manifest_sha256']
        if job['job_id'] in recovered:
            assert state['original_plan_sha256']==plan_hash and state['recovery_plan_sha256']==sha(recovery_plan_path)
            assert state['full_restart_same_input'] and not state['new_seed']
            archive=Path(state['failed_attempt'])
            assert archive.is_relative_to(HERE/'failed_attempts/heterogeneity_supervisor_cleanup')
            assert not archive.is_relative_to(HERE/'runs')
            evidence_path=archive/'attempt_interrupted.json';evidence=read(evidence_path)
            assert sha(evidence_path)==recovery_jobs[job['job_id']]['failed_attempt_evidence_sha256']
            assert evidence['interrupted_attempt'] and evidence['not_a_completed_result']
            assert evidence['excluded_from_canonical_case_registry']
            for relative,digest in evidence['original_files_sha256'].items():assert sha(archive/relative)==digest
            assert sha(archive/'original_execution.log')==evidence['original_execution_log_sha256']
            interrupted=read(archive/'command.json')
            assert interrupted['status']=='running' and not interrupted['completed_simulation']
            assert interrupted['manifest_sha256']==job['manifest_sha256']
            assert interrupted['pid']==original_states[job['job_id']]['pid']
            failed_attempts.append(dict(job_id=job['job_id'],archive=str(archive),evidence_sha256=sha(evidence_path),
                original_files_sha256=evidence['original_files_sha256'],original_pid=interrupted['pid'],
                new_pid=state['pid'],original_started_utc=interrupted['started_utc'],observed_loss_utc=evidence['observed_loss_utc'],
                suspected_cause=evidence['suspected_cause'],complete_same_manifest_rerun=True,not_an_additional_seed=True))
            execution_log=recovery_jobs[job['job_id']]['log']
        else:
            assert state['plan_sha256']==plan_hash and state['fix_sha256']==sha(fix_path)
            execution_log=job['log']
        assert sha(job['manifest']) == job['manifest_sha256']
        case = case_audit(Path(job['output']),job['manifest_sha256'],plan,True)
        assert case['seed']==job['seed'] and case['strategy']==job['strategy']
        assert case['command_pid'] == state['pid']
        assert case['sources_sha256'][str(Path(job['output'])/'analysis.json')] == state['analysis_sha256']
        case.update(job_id=job['job_id'],status_path=str(state_path),status_sha256=sha(state_path),
                    execution_log_path=execution_log,execution_log_sha256=sha(execution_log),
                    recovered_after_interruption=job['job_id'] in recovered,
                    returncode_observation=state.get('returncode_source','Direct child waitpid during authorized recovery'))
        cases.append(case)
    events=[]
    for case in cases:
        events.extend([(datetime.fromisoformat(case['started_utc']),1),(datetime.fromisoformat(case['ended_utc']),-1)])
    for failed in failed_attempts:
        events.extend([(datetime.fromisoformat(failed['original_started_utc']),1),(datetime.fromisoformat(failed['observed_loss_utc']),-1)])
    active=maximum=0
    for _,delta in sorted(events):
        active+=delta;maximum=max(maximum,active)
    assert active==0 and maximum<=5
    takeover_launch_path=HERE/'heterogeneity_confirmation_takeover_launch.json'
    takeover_launch=read(takeover_launch_path)
    assert takeover_launch['status']=='failed' and takeover_launch['all_original_sim_pids_and_log_fds_preserved']
    recovery_launch_path=HERE/'heterogeneity_confirmation_recovery_launch.json'
    recovery_launch=read(recovery_launch_path);assert recovery_launch['status']=='complete'
    original = Path(plan['jobs'][0]['manifest']).parent
    seed7 = case_audit(original,plan['jobs'][0]['manifest_sha256'],plan,False)
    baseline = sorted([seed7]+[c for c in cases if c['strategy']=='baseline'],key=lambda c:c['seed'])
    assert [c['seed'] for c in baseline] == [7,19,43,67,101]
    once = next(c for c in cases if c['strategy']=='once')
    assert once['manifest_sha256'] == seed7['manifest_sha256'] and once['input_fingerprint'] == seed7['input_fingerprint']
    aggregates = []
    for index,(left,right) in enumerate(WINDOWS):
        def stats(group):
            values=[c['windows'][index]['U_percent'] for c in group]
            return dict(seeds=[c['seed'] for c in group],count=len(values),mean=statistics.mean(values),
                        sample_stddev=statistics.stdev(values),min=min(values),max=max(values),values=values)
        aggregates.append(dict(start_ms=left,end_ms=right,baseline_all_five=stats(baseline),
            baseline_four_confirmation=stats([c for c in baseline if c['seed']!=7]),
            paired_seed7=dict(baseline_U=seed7['windows'][index]['U_percent'],once_U=once['windows'][index]['U_percent'],
                             improvement_pp=once['windows'][index]['U_percent']-seed7['windows'][index]['U_percent'])))
    all_cases = cases+[seed7]
    record = dict(all_checks_passed=True,no_new_simulation=True,no_runner_builder_or_analyzer_imported=True,
        exact_five_authorized_new_cases=True,no_extra_seed7_baseline=True,no_completed_case_overwritten=True,
        prior_seed7_baseline_unchanged=True,max_local_concurrency=5,max_simultaneous_simulations_observed=maximum,
        concurrency_interpretation='Failed attempts end at the first recorded observation of their disappearance. This is a conservative overlap bound from recorded intervals, not an exact OS exit timestamp.',
        all_five_canonical_execution_records_complete=True,
        expected_scientific_case_registry=str(registry_path),expected_scientific_case_registry_sha256=sha(registry_path),
        expected_scientific_case_count=90,all_five_cases_in_registry=True,failed_attempts_excluded_from_registry=True,
        all_cases_all_32_active=all(w['all_32_active'] for c in all_cases for w in c['windows']),
        all_cases_long_short_mixed=all(w['long_short_mixed_pass'] for c in all_cases for w in c['windows']),
        plan=str(PLAN),plan_sha256=plan_hash,plan_source_sha256=plan['protected_sources_sha256'],
        supervision_fix=str(fix_path),supervision_fix_sha256=sha(fix_path),supervision_correction_disclosed=True,
        execution_core_sources_sha256=fix['execution_core_sources_sha256'],
        takeover_launch_sha256=sha(takeover_launch_path),initial_takeover_preserved_original_pids=True,
        three_attempts_interrupted_after_initial_takeover=True,interrupted_attempts_not_accepted=True,
        same_input_full_recovery_authorized=True,failed_attempts=failed_attempts,
        recovery_plan=str(recovery_plan_path),recovery_plan_sha256=sha(recovery_plan_path),
        recovery_results_sha256=sha(recovery_results_path),recovery_launch_sha256=sha(recovery_launch_path),
        independent_input_audit=str(plan['independent_audit']),independent_input_audit_sha256=plan['independent_audit_sha256'],
        queue_results_sha256=sha(run_results_path),new_cases=cases,existing_seed7_baseline=seed7,aggregates=aggregates,
        prior_seed7_baseline_files_sha256=plan['prior_seed7_baseline_files_sha256'],
        builder_sha256=sha(Path(__file__)),
        caveats=['No seed is discarded for low utilization or incomplete warm class coverage.',
                 'The five-seed Baseline pool includes the original exploratory seed7; the four newly fixed seeds are also reported separately.',
                 'Once has one paired seed7 run; do not report a five-seed Once mean or compare its sample count as if matched.',
                 'Both long and short C are explicit data-affine extrapolations; ideal mean load about100.203% is not per-disk instantaneous underload.'])
    for p,digest in plan['prior_seed7_baseline_files_sha256'].items():assert sha(p)==digest
    assert sha(PLAN)==plan_hash
    assert not OUT.exists(), 'Never replace a completed acceptance artifact.'
    temporary=OUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(OUT)
    print(json.dumps(dict(all_checks_passed=True,output=str(OUT),all_cases_all_32_active=record['all_cases_all_32_active'],
                          all_cases_long_short_mixed=record['all_cases_long_short_mixed'],aggregates=aggregates),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
