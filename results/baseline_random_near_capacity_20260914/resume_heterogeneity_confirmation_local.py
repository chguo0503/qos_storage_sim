#!/usr/bin/env python3
"""Transparent supervisor correction; preserve the three existing simulations.

The original frozen queue confused a 106-file audit snapshot with the runner's
29-file execution-core inventory. This adapter does not change either record,
the runner, inputs, active simulations, or the original five-job scope.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE/'heterogeneity_confirmation_local_plan.json'
FIX = HERE/'heterogeneity_confirmation_supervision_fix.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def utc():return datetime.now(timezone.utc).isoformat()


def write(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def process_info(pid):
    proc=Path('/proc')/str(pid)
    if not proc.exists():return None
    raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,state=fields[0],start_ticks=fields[19],
                argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))


def output_fd(pid):
    p=Path('/proc')/str(pid)/'fd/1';s=p.stat()
    assert stat.S_ISREG(s.st_mode),'Do not detach a child whose stdout is a pipe.'
    return dict(target=os.readlink(p),device=s.st_dev,inode=s.st_ino,regular_file=True)


def prepare():
    assert not FIX.exists()
    plan=read(PLAN);launch_path=HERE/'heterogeneity_confirmation_local_launch.json';launch=read(launch_path)
    assert launch['status']=='running' and launch['plan_sha256']==sha(PLAN)
    baseline=HERE/'runs/hetero_S10_12_14_L384_ssu8_h22000_seed7/baseline/command.json'
    core=read(baseline)['core_source_sha256']
    assert len(core)==29 and len(plan['core_sources_sha256'])==106
    assert set(core)<set(plan['core_sources_sha256'])
    assert all(core[n]==plan['core_sources_sha256'][n] for n in core)
    old=process_info(launch['pid']);assert old and 'run_heterogeneity_confirmation_local.py --run' in old['argv']
    attached=[];pending=[]
    for job in plan['jobs']:
        output=Path(job['output'])
        if output.exists():
            command=read(output/'command.json');state=read(job['status'])
            assert command['status']=='running' and state['status']=='running'
            assert command['pid']==state['pid'] and command['core_source_sha256']==core
            assert command['manifest_sha256']==job['manifest_sha256']
            info=process_info(command['pid']);assert info and 'experiment.py' in info['argv']
            fd=output_fd(info['pid']);assert fd['target']==job['log']
            attached.append(dict(job_id=job['job_id'],process=info,stdout_fd=fd,
                old_status_sha256=sha(job['status']),command_sha256_at_takeover=sha(output/'command.json')))
        else:
            assert not Path(job['log']).exists() and not Path(job['status']).exists()
            pending.append(job['job_id'])
    assert len(attached)==3 and set(pending)=={'hetero_short_seed67_baseline','hetero_short_seed101_baseline'}
    fix=dict(created_utc=utc(),reason='Supervisor-only correction: execution core contains29 files and is an exact matching subset of the106-file historical audit snapshot. Full dictionary equality was incorrectly required in the original supervisor.',
        no_simulation_error_detected=True,no_input_or_simulator_change=True,no_experiments_added=True,
        original_plan=str(PLAN),original_plan_sha256=sha(PLAN),original_supervisor=old,
        original_launch_sha256=sha(launch_path),execution_core_sources_sha256=core,
        legacy_audit_snapshot_count=106,execution_core_count=29,all_shared_hashes_match=True,
        attached_simulations=attached,pending_original_jobs=pending,max_parallel=3,
        action_order=['verify original hashes and regular-file child stdout','SIGTERM only original supervisor PID',
                      'verify all three child identities and stdout inodes unchanged','attach original simulations',
                      'finish their analysis and run only the two original pending jobs'],
        replacement_supervisor=str(Path(__file__).resolve()),replacement_supervisor_sha256=sha(__file__),
        immutable_original_queue_source_sha256=sha(HERE/'run_heterogeneity_confirmation_local.py'))
    write(FIX,fix);print(json.dumps(dict(fix=str(FIX),sha256=sha(FIX),attached=3,pending=2)),flush=True)


def run():
    plan,fix=read(PLAN),read(FIX);fix_hash=sha(FIX)
    assert sha(PLAN)==fix['original_plan_sha256'] and sha(__file__)==fix['replacement_supervisor_sha256']
    core=fix['execution_core_sources_sha256']
    def verify():
        assert sha(FIX)==fix_hash and sha(PLAN)==fix['original_plan_sha256']
        assert sha(__file__)==fix['replacement_supervisor_sha256']
        for p,d in plan['protected_sources_sha256'].items():assert sha(p)==d
        for p,d in plan['core_sources_sha256'].items():assert sha(ROOT/p)==d
        for p,d in plan['prior_seed7_baseline_files_sha256'].items():assert sha(p)==d
        assert sha(plan['independent_audit'])==plan['independent_audit_sha256']
        assert read(plan['independent_audit'])['all_checks_passed']
    verify()
    launch_path=HERE/'heterogeneity_confirmation_takeover_launch.json';assert not launch_path.exists()
    launch=dict(status='starting',pid=os.getpid(),started_utc=utc(),fix_sha256=fix_hash,
                original_plan_sha256=fix['original_plan_sha256'],max_parallel=3)
    write(launch_path,launch)
    original=process_info(fix['original_supervisor']['pid'])
    assert original and original['start_ticks']==fix['original_supervisor']['start_ticks']
    for entry in fix['attached_simulations']:
        current=process_info(entry['process']['pid'])
        assert current and current['start_ticks']==entry['process']['start_ticks']
        assert output_fd(current['pid'])==entry['stdout_fd']
    # A single PID is signalled, never the process group. Child stdout is a
    # separately inherited regular file descriptor, so no pipe is closed.
    os.kill(original['pid'],signal.SIGTERM)
    for _ in range(100):
        state=process_info(original['pid'])
        if state is None or state['state']=='Z':break
        time.sleep(.05)
    else:raise RuntimeError('Original supervisor did not exit; no pending job was launched.')
    for entry in fix['attached_simulations']:
        current=process_info(entry['process']['pid'])
        assert current and current['start_ticks']==entry['process']['start_ticks']
        assert output_fd(current['pid'])==entry['stdout_fd']
    launch.update(status='running',original_supervisor_stopped_utc=utc(),
                  original_signal='SIGTERM to supervisor PID only',all_original_sim_pids_and_log_fds_preserved=True)
    write(launch_path,launch)
    print(json.dumps(launch),flush=True)
    attached={a['job_id']:a for a in fix['attached_simulations']}
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def job_run(job):
        status=HERE/'execution_logs'/(job['job_id']+'.takeover.status.json');assert not status.exists()
        state=dict(job_id=job['job_id'],seed=job['seed'],strategy=job['strategy'],output=job['output'],
            started_monitoring_utc=utc(),fix_sha256=fix_hash,plan_sha256=fix['original_plan_sha256'],
            manifest_sha256=job['manifest_sha256'],attached_existing_simulation=job['job_id'] in attached)
        try:
            verify();assert sha(job['manifest'])==job['manifest_sha256']
            case=Path(job['output'])
            if state['attached_existing_simulation']:
                entry=attached[job['job_id']];pid=entry['process']['pid']
                state.update(pid=pid,status='attached',initial_process_start_ticks=entry['process']['start_ticks']);write(status,state)
                while True:
                    command=read(case/'command.json')
                    assert command['pid']==pid
                    if command['status'] in ('complete','failed'):break
                    current=process_info(pid)
                    assert current and current['state']!='Z' and current['start_ticks']==entry['process']['start_ticks'],'Attached simulation exited without a final command record.'
                    time.sleep(2)
                assert command['status']=='complete' and command['returncode']==0
                for _ in range(30):
                    current=process_info(pid)
                    if current is None or current['state']=='Z':break
                    time.sleep(1)
                else:raise RuntimeError('Completed attached simulation did not exit.')
                state.update(returncode=0,returncode_source='Completed runner command; orphan process not directly waitable by replacement supervisor')
            else:
                assert not case.exists() and not Path(job['log']).exists()
                with Path(job['log']).open('x') as stream:
                    process=subprocess.Popen(job['argv'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    state.update(pid=process.pid,status='running',argv=job['argv']);write(status,state)
                    print(json.dumps(state),flush=True);code=process.wait()
                state.update(returncode=code,returncode_source='Direct child waitpid')
                assert code==0;command=read(case/'command.json')
            assert command['status']=='complete' and command['completed_simulation']
            assert command['observed_completed_blocks']==command['expected_blocks']==42775040
            assert command['core_source_sha256']==core
            assert command['runner_sha256']==plan['protected_sources_sha256'][str(HERE/'experiment.py')]
            assert command['manifest_sha256']==job['manifest_sha256']==sha(case/'manifest.json.gz')
            assert command['output_sha256']==sha(case/'result.json.gz')
            assert command['physical_service_sha256']==sha(case/'physical_service.json')
            assert command['trace_window_ms'] is None and command['retained_blocks']==0 and not (case/'trace.json.gz').exists()
            output=case/'analysis.json';assert not output.exists()
            argv=[sys.executable,str(HERE/'analyze.py'),'--case',str(case),'--output',str(output)]
            for role in plan['short_roles']:argv.extend(['--short-role',role])
            state.update(status='analyzing',analysis_argv=argv);write(status,state)
            with Path(job['log']).open('a') as stream:
                code=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            assert code==0
            result=read(output);assert result['all_technical_checks_passed']
            state.update(status='complete',analysis_sha256=sha(output),output_sha256=command['output_sha256'],
                command_sha256=sha(case/'command.json'),physical_service_sha256=command['physical_service_sha256'],
                windows=[{k:w[k] for k in ('start_ms','end_ms','U_percent','all_32_active','mixed_card_count',
                    'long_short_mixed_card_count','long_short_mixed_pass')} for w in result['windows']
                    if w['start_ms']==2000 and w['end_ms'] in (4000,20000)])
            verify()
        except BaseException as exc:
            state.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        state['ended_utc']=utc();state['status_path']=str(status);write(status,state)
        print(json.dumps(state),flush=True);return state

    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(job_run,plan['jobs']))
    verify();passed=all(r['status']=='complete' for r in results)
    write(HERE/'heterogeneity_confirmation_takeover_results.json',dict(all_completed=passed,max_parallel=3,
        plan_sha256=fix['original_plan_sha256'],fix_sha256=fix_hash,ended_utc=utc(),jobs=results,
        original_three_simulations_preserved=True,no_simulations_repeated=True,no_jobs_added=True))
    launch.update(status='complete' if passed else 'failed',ended_utc=utc());write(launch_path,launch)
    assert passed


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare-fix',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args();assert a.prepare_fix!=a.run
    prepare() if a.prepare_fix else run()
