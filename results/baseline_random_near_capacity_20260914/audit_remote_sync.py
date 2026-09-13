#!/usr/bin/env python3
"""Poll the owned remote queue and atomically import hash-verified completed runs.

Credentials are passed in memory to sync(password); they are never stored here.
"""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
import shlex
import shutil
import tarfile
import pexpect

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:
        return json.load(stream)


def transport(program, args, password, timeout=90):
    child=pexpect.spawn(program,args,encoding='utf-8',timeout=timeout)
    choice=child.expect(['[Pp]assword:',pexpect.EOF])
    if choice==0:
        child.sendline(password)
        child.expect(pexpect.EOF)
    output=child.before
    child.close()
    assert child.exitstatus==0, f'{program} failed: {output}'
    return output


def remote_json(code,password):
    args=['-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
          '-o','NumberOfPasswordPrompts=1','192.168.31.126','python3 -c '+shlex.quote(code)]
    output=transport('ssh',args,password)
    return json.loads(output.split('AUDIT_JSON:',1)[1].strip())


def sync(password):
    setup=read(HERE/'audit_remote_setup.json')
    plan=read(HERE/'audit_remote_queue_plan.json')
    state_path=HERE/'audit_remote_sync_record.json'
    state=read(state_path) if state_path.exists() else {'imported':{}}
    remaining=[j['label'] for j in plan['jobs'] if j['label'] not in state['imported']]
    code='''import pathlib,json,tarfile,hashlib,time
D=pathlib.Path(__DIR__);H=D/'results/baseline_random_near_capacity_20260914'
Q=json.loads((H/'audit_remote_queue_status.json').read_text());remaining=__REMAINING__
done=[label for label in remaining if Q['jobs'][label]['status']=='complete']
details={}
for label,v in Q['jobs'].items():
 if v['status']=='running':
  p=H/'runs'/label/'baseline/progress.json'
  if p.exists():details[label]=json.loads(p.read_text())
out={'queue':Q,'new_complete':done,'running_progress':details}
if done:
 A=D/('audit_sync_'+str(time.time_ns())+'.tar.gz')
 with tarfile.open(A,'w:gz') as t:
  for label in done:
   P=H/'runs'/label/'baseline'
   for name in ('manifest.json.gz','command.json','result.json.gz','physical_service.json','progress.json'):
    t.add(P/name,arcname=label+'/baseline/'+name,recursive=False)
   t.add(H/'execution_logs'/(label+'.remote.log'),arcname='execution_logs/'+label+'.remote.log',recursive=False)
 out.update(archive=str(A),archive_sha256=hashlib.sha256(A.read_bytes()).hexdigest())
print('AUDIT_JSON:'+json.dumps(out))
'''.replace('__DIR__',repr(setup['remote_dir'])).replace('__REMAINING__',repr(remaining))
    response=remote_json(code,password)
    assert response['queue']['plan_sha256']==sha(HERE/'audit_remote_queue_plan.json')
    (HERE/'audit_remote_queue_status.json').write_text(json.dumps(response['queue'],ensure_ascii=False,indent=2)+'\n')
    new=response['new_complete']
    if new:
        incoming=HERE/'runtime_validation'/'remote_incoming'
        incoming.mkdir(parents=True,exist_ok=True)
        archive=incoming/Path(response['archive']).name
        transport('scp',['-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5','-o','NumberOfPasswordPrompts=1',
                  'chguo@192.168.31.126:'+response['archive'],str(archive)],password)
        assert sha(archive)==response['archive_sha256']
        batch=incoming/archive.stem.replace('.tar','')
        batch.mkdir(exist_ok=False)
        expected={label+'/baseline/'+name for label in new for name in
                  ('manifest.json.gz','command.json','result.json.gz','physical_service.json','progress.json')}
        expected.update('execution_logs/'+label+'.remote.log' for label in new)
        with tarfile.open(archive) as t:
            assert {member.name for member in t.getmembers()}==expected
            assert all(member.isfile() for member in t.getmembers())
            t.extractall(batch)
        by_label={j['label']:j for j in plan['jobs']}
        for label in new:
            job=by_label[label];src=batch/label/'baseline';cmd=read(src/'command.json')
            assert cmd['status']=='complete' and cmd['returncode']==0 and cmd['completed_simulation']
            assert cmd['strategy']=='baseline' and cmd['order']=='random' and cmd['label']==label
            assert cmd['trace_window_ms'] is None
            assert sha(src/'manifest.json.gz')==sha(ROOT/job['manifest'])==job['manifest_sha256']==cmd['manifest_sha256']
            assert cmd['input_fingerprint']==job['input_fingerprint']
            assert cmd['core_source_sha256']==setup['environment']['core_source_sha256']
            assert cmd['runner_sha256']==plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/experiment.py']
            assert sha(src/'result.json.gz')==cmd['output_sha256']
            assert sha(src/'physical_service.json')==cmd['physical_service_sha256']
            result=read(src/'result.json.gz');physical=read(src/'physical_service.json')
            assert result['input_fingerprint']==job['input_fingerprint'] and result['core_and_policy_sha256']==cmd['core_source_sha256']
            assert all(result['summary']['invariants'].values())
            assert physical['observed_completed_blocks']==physical['expected_blocks']==cmd['expected_blocks']
            assert physical['baseline_nonzero_paths']==0
            target=HERE/'runs'/label
            if target.exists():
                raise FileExistsError(f'Preserving local result already present: {target}')
            (HERE/'runs').mkdir(exist_ok=True)
            os.replace(batch/label,target)
            (HERE/'execution_logs').mkdir(exist_ok=True)
            shutil.copyfile(batch/'execution_logs'/(label+'.remote.log'),HERE/'execution_logs'/(label+'.remote.log'))
            state['imported'][label]={'command_sha256':sha(target/'baseline/command.json'),
                                      'manifest_sha256':cmd['manifest_sha256'],'result_sha256':cmd['output_sha256'],
                                      'physical_service_sha256':cmd['physical_service_sha256'],
                                      'host':cmd['host'],'python_version':cmd['python_version'],
                                      'source_input_result_and_physical_hashes_verified':True,
                                      'windows':cmd['windows'],'imported_utc':datetime.now(timezone.utc).isoformat()}
            state['updated_utc']=datetime.now(timezone.utc).isoformat()
            state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        state['last_archive_sha256']=response['archive_sha256']
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report={'complete_remote':response['queue']['complete'],'running_remote':response['queue']['running'],
            'failed_remote':response['queue']['failed'],'imported_local':len(state['imported']),
            'newly_imported':new,'running_progress':response['running_progress'],
            'all_finished':response['queue']['complete']==len(plan['jobs']) and response['queue'].get('source_hashes_verified_after',False),
            'queue_terminal':bool(response['queue'].get('ended_utc'))}
    print(json.dumps(report,ensure_ascii=False))
    return report


if __name__=='__main__':
    import getpass
    sync(getpass.getpass('SSH password (not stored): '))
