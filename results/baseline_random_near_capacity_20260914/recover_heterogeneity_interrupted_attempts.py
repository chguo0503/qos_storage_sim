#!/usr/bin/env python3
"""Archive three interrupted attempts and rerun their unchanged full inputs.

The two already running pending seeds are untouched. This creates no new seed,
profile, strategy, or logical experiment. Original incomplete records persist.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'heterogeneity_confirmation_recovery_plan.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def prepare():
    assert not PLAN.exists()
    original_path=HERE/'heterogeneity_confirmation_local_plan.json';original=read(original_path)
    fix_path=HERE/'heterogeneity_confirmation_supervision_fix.json';fix=read(fix_path)
    core=fix['execution_core_sources_sha256'];assert len(core)==29
    for p,d in core.items():assert sha(ROOT/p)==d==original['core_sources_sha256'][p]
    for p,d in original['prior_seed7_baseline_files_sha256'].items():assert sha(p)==d
    existing=[]
    for job in original['jobs'][3:]:
        c=read(Path(job['output'])/'command.json')
        assert c['status']=='running' and Path('/proc',str(c['pid'])).exists()
        assert c['core_source_sha256']==core
        existing.append(dict(job_id=job['job_id'],pid=c['pid'],command_sha256=sha(Path(job['output'])/'command.json')))
    archive_root=HERE/'failed_attempts/heterogeneity_supervisor_cleanup'
    jobs=[]
    for job in original['jobs'][:3]:
        case=Path(job['output']);command=read(case/'command.json')
        assert command['status']=='running' and not command['completed_simulation']
        assert not Path('/proc',str(command['pid'])).exists(),'Never archive a live simulation.'
        assert sha(job['manifest'])==job['manifest_sha256']
        archived=archive_root/job['label']/job['strategy'];assert not archived.exists()
        before={str(p.relative_to(case)):sha(p) for p in case.rglob('*') if p.is_file()}
        archived.parent.mkdir(parents=True,exist_ok=True)
        case.rename(archived)
        assert not case.exists()
        assert before=={str(p.relative_to(archived)):sha(p) for p in archived.rglob('*') if p.is_file()}
        log=Path(job['log']);old_log_hash=sha(log)
        shutil.copyfile(log,archived/'original_execution.log')
        assert sha(archived/'original_execution.log')==old_log_hash
        takeover_status=HERE/'execution_logs'/(job['job_id']+'.takeover.status.json')
        assert read(takeover_status)['status']=='failed'
        shutil.copyfile(takeover_status,archived/'original_takeover_status.json')
        evidence=dict(interrupted_attempt=True,not_a_completed_result=True,excluded_from_canonical_case_registry=True,
            original_canonical_path=str(case),archived_path=str(archived),original_pid=command['pid'],archived_utc=utc(),
            observed_loss_utc=read(takeover_status)['ended_utc'],original_files_sha256=before,
            original_execution_log_sha256=old_log_hash,
            observation='Three attached processes disappeared around19:06:57UTC while the previous terminated exec session was polled/reaped. All three command records remained running and no final output was accepted.',
            suspected_cause='Execution-tool session descendant cleanup; inferred from timing, not confirmed by OS-level evidence.',
            supervisor_was_single_PID_SIGTERM=True,child_regular_file_fds_were_verified_preserved_at_initial_takeover=True,
            recovery='Rerun the complete original manifest from the beginning, never resume a partial state; same logical seed/strategy only.')
        write(archived/'attempt_interrupted.json',evidence)
        jobs.append(dict(job,failed_attempt=str(archived),failed_attempt_evidence_sha256=sha(archived/'attempt_interrupted.json'),
            log=str(HERE/'execution_logs'/(job['job_id']+'.recovery.log')),
            status=str(HERE/'execution_logs'/(job['job_id']+'.recovery.status.json'))))
    plan=dict(created_utc=utc(),original_plan=str(original_path),original_plan_sha256=sha(original_path),
        previous_supervision_fix=str(fix_path),previous_supervision_fix_sha256=sha(fix_path),
        root_authorized_recovery=True,full_rerun_only=True,new_seeds_or_experiments_added=False,
        max_recovery_workers=3,max_total_heterogeneity_simulations=5,two_live_simulations_untouched=existing,
        independent_input_audit=original['independent_audit'],independent_input_audit_sha256=original['independent_audit_sha256'],
        execution_core_sources_sha256=core,protected_sources_sha256=original['protected_sources_sha256'],
        prior_seed7_baseline_files_sha256=original['prior_seed7_baseline_files_sha256'],
        recovery_supervisor_sha256=sha(__file__),jobs=jobs)
    write(PLAN,plan);print(json.dumps(dict(plan=str(PLAN),sha256=sha(PLAN),archived=3,untouched_running=existing)),flush=True)


def run():
    plan=read(PLAN);plan_hash=sha(PLAN)
    assert sha(__file__)==plan['recovery_supervisor_sha256']
    def verify():
        assert sha(PLAN)==plan_hash and sha(__file__)==plan['recovery_supervisor_sha256']
        assert sha(plan['original_plan'])==plan['original_plan_sha256']
        assert sha(plan['previous_supervision_fix'])==plan['previous_supervision_fix_sha256']
        assert sha(plan['independent_input_audit'])==plan['independent_input_audit_sha256']
        assert read(plan['independent_input_audit'])['all_checks_passed']
        for p,d in plan['execution_core_sources_sha256'].items():assert sha(ROOT/p)==d
        for p,d in plan['protected_sources_sha256'].items():assert sha(p)==d
        for p,d in plan['prior_seed7_baseline_files_sha256'].items():assert sha(p)==d
    verify()
    launch_path=HERE/'heterogeneity_confirmation_recovery_launch.json';assert not launch_path.exists()
    launch=dict(status='running',pid=os.getpid(),started_utc=utc(),plan_sha256=plan_hash,
                max_recovery_workers=3,max_total_heterogeneity_simulations=5)
    write(launch_path,launch)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def execute(job):
        status=Path(job['status']);assert not status.exists()
        state=dict(job_id=job['job_id'],seed=job['seed'],strategy=job['strategy'],output=job['output'],
            recovery_plan_sha256=plan_hash,original_plan_sha256=plan['original_plan_sha256'],
            manifest_sha256=job['manifest_sha256'],failed_attempt=job['failed_attempt'],
            started_utc=utc(),new_seed=False,full_restart_same_input=True)
        try:
            verify();case=Path(job['output']);assert not case.exists()
            assert sha(job['manifest'])==job['manifest_sha256']
            with Path(job['log']).open('x') as stream:
                process=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state.update(pid=process.pid,status='running',argv=job['argv']);write(status,state)
                print(json.dumps(state),flush=True);code=process.wait()
                state['returncode']=code;assert code==0
                command=read(case/'command.json')
                assert command['status']=='complete' and command['completed_simulation']
                assert command['observed_completed_blocks']==command['expected_blocks']==42775040
                assert command['core_source_sha256']==plan['execution_core_sources_sha256']
                assert command['runner_sha256']==plan['protected_sources_sha256'][str(HERE/'experiment.py')]
                assert command['manifest_sha256']==job['manifest_sha256']==sha(case/'manifest.json.gz')
                assert command['output_sha256']==sha(case/'result.json.gz')
                assert command['physical_service_sha256']==sha(case/'physical_service.json')
                assert command['trace_window_ms'] is None and command['retained_blocks']==0 and not (case/'trace.json.gz').exists()
                output=case/'analysis.json';assert not output.exists()
                argv=[sys.executable,str(HERE/'analyze.py'),'--case',str(case),'--output',str(output)]
                for role in ('S1','S2','S3'):argv.extend(['--short-role',role])
                state.update(status='analyzing',analysis_argv=argv);write(status,state)
                code=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                assert code==0
            analysis=read(output);assert analysis['all_technical_checks_passed']
            state.update(status='complete',analysis_sha256=sha(output),output_sha256=command['output_sha256'],
                command_sha256=sha(case/'command.json'),physical_service_sha256=command['physical_service_sha256'],
                windows=[{k:w[k] for k in ('start_ms','end_ms','U_percent','all_32_active','mixed_card_count',
                    'long_short_mixed_card_count','long_short_mixed_pass')} for w in analysis['windows']
                    if w['start_ms']==2000 and w['end_ms'] in (4000,20000)])
            verify()
        except BaseException as exc:
            state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        state['ended_utc']=utc();state['status_path']=str(status);write(status,state)
        print(json.dumps(state),flush=True);return state
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(execute,plan['jobs']))
    verify();passed=all(r['status']=='complete' for r in results)
    write(HERE/'heterogeneity_confirmation_recovery_results.json',dict(all_completed=passed,
        recovery_plan_sha256=plan_hash,original_plan_sha256=plan['original_plan_sha256'],ended_utc=utc(),
        jobs=results,interrupted_attempts_preserved=True,new_seeds_or_experiments_added=False))
    launch.update(status='complete' if passed else 'failed',ended_utc=utc());write(launch_path,launch)
    assert passed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-recovery',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args();assert a.prepare_recovery!=a.run
    prepare() if a.prepare_recovery else run()
