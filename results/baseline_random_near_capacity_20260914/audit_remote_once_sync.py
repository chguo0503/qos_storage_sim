#!/usr/bin/env python3
"""Import and validate only the two exploratory Once pairs; no credentials stored."""
from datetime import datetime, timezone
from pathlib import Path
import gzip
import json
import os
import shutil
import tarfile

from audit_remote_sync import HERE, ROOT, read, sha, remote_json, transport


def verify_one(job, directory, plan, setup, queue):
    command = read(directory / 'command.json')
    result = read(directory / 'result.json.gz')
    physical = read(directory / 'physical_service.json')
    label = job['label']
    baseline = ROOT / job['paired_baseline']
    assert command['status'] == queue['jobs'][label]['status'] == 'complete'
    assert command['returncode'] == 0 and command['completed_simulation']
    assert command['strategy'] == 'once' and command['order'] == 'random'
    assert command['label'] == label and command['trace_window_ms'] is None and command['retained_blocks'] == 0
    assert sha(directory / 'command.json') == queue['jobs'][label]['command_sha256']
    assert sha(directory / 'manifest.json.gz') == sha(ROOT / job['manifest']) == job['manifest_sha256'] == command['manifest_sha256']
    assert sha(baseline / 'manifest.json.gz') == job['manifest_sha256']
    assert command['input_fingerprint'] == job['input_fingerprint'] == result['input_fingerprint']
    assert read(baseline / 'command.json')['input_fingerprint'] == command['input_fingerprint']
    assert command['core_source_sha256'] == result['core_and_policy_sha256'] == setup['environment']['core_source_sha256']
    assert command['runner_sha256'] == plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/experiment.py']
    assert command['source_data_sha256'] == plan['frozen_source_sha256']['data']
    assert sha(directory / 'result.json.gz') == command['output_sha256']
    assert sha(directory / 'physical_service.json') == command['physical_service_sha256']
    assert physical['observed_completed_blocks'] == physical['expected_blocks'] == command['observed_completed_blocks'] == command['expected_blocks']
    assert result['summary']['invariants'] and all(value is True for value in result['summary']['invariants'].values())
    assert [(w['start_ms'], w['end_ms']) for w in command['windows']] == [(2000, 4000), (2000, 20000)]
    assert command['host'] == queue['host']
    assert queue['jobs'][label]['total_owned_simulations_upper_bound_at_start'] <= 6
    return dict(command_sha256=sha(directory / 'command.json'), manifest_sha256=command['manifest_sha256'],
                result_sha256=command['output_sha256'], physical_service_sha256=command['physical_service_sha256'],
                host=command['host'], python_version=command['python_version'], windows=command['windows'],
                all_hashes_blocks_and_invariants_verified=True, paired_input_bytes_identical=True)


def sync(password):
    plan = read(HERE / 'audit_remote_once_plan.json')
    setup = read(HERE / 'audit_remote_setup.json')
    record_path = HERE / 'audit_remote_once_sync_record.json'
    record = read(record_path) if record_path.exists() else {'selection': plan['selection'], 'imported': {}}
    remaining = [job['label'] for job in plan['jobs'] if job['label'] not in record['imported']]
    code = '''from pathlib import Path
import json,tarfile,hashlib,time
D=Path(__DIR__);H=D/'results/baseline_random_near_capacity_20260914'
Q=json.loads((H/'audit_remote_once_status.json').read_text());remaining=__REMAINING__
done=[label for label in remaining if Q['jobs'][label]['status']=='complete'];progress={}
for label,v in Q['jobs'].items():
 if v['status']=='running':
  p=H/'runs'/label/'once/progress.json'
  if p.exists():progress[label]=json.loads(p.read_text())
out={'queue':Q,'new_complete':done,'running_progress':progress}
if done:
 A=D/('audit_once_sync_'+str(time.time_ns())+'.tar.gz')
 with tarfile.open(A,'w:gz') as t:
  for label in done:
   P=H/'runs'/label/'once'
   for name in ('manifest.json.gz','command.json','result.json.gz','physical_service.json','progress.json'):
    t.add(P/name,arcname=label+'/once/'+name,recursive=False)
   t.add(H/'execution_logs'/(label+'.once.remote.log'),arcname='execution_logs/'+label+'.once.remote.log',recursive=False)
 out.update(archive=str(A),archive_sha256=hashlib.sha256(A.read_bytes()).hexdigest())
print('AUDIT_JSON:'+json.dumps(out))
'''.replace('__DIR__', repr(setup['remote_dir'])).replace('__REMAINING__', repr(remaining))
    response = remote_json(code, password)
    queue = response['queue']
    assert queue['plan_sha256'] == sha(HERE / 'audit_remote_once_plan.json')
    assert queue['queue_runner_sha256'] == sha(HERE / 'audit_remote_once_queue.py')
    (HERE / 'audit_remote_once_status.json').write_text(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
    new = response['new_complete']
    if new:
        incoming = HERE / 'runtime_validation/remote_once_incoming'
        incoming.mkdir(parents=True, exist_ok=True)
        archive = incoming / Path(response['archive']).name
        transport('scp', ['-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5',
                         '-o', 'NumberOfPasswordPrompts=1',
                         'chguo@192.168.31.126:' + response['archive'], str(archive)], password)
        assert sha(archive) == response['archive_sha256']
        batch = incoming / archive.stem.replace('.tar', '')
        batch.mkdir(exist_ok=False)
        expected = {label + '/once/' + name for label in new for name in
                    ('manifest.json.gz', 'command.json', 'result.json.gz', 'physical_service.json', 'progress.json')}
        expected.update('execution_logs/' + label + '.once.remote.log' for label in new)
        with tarfile.open(archive) as stream:
            assert {member.name for member in stream.getmembers()} == expected
            assert all(member.isfile() for member in stream.getmembers())
            stream.extractall(batch)
        by_label = {job['label']: job for job in plan['jobs']}
        for label in new:
            source = batch / label / 'once'
            checked = verify_one(by_label[label], source, plan, setup, queue)
            target = HERE / 'runs' / label / 'once'
            assert not target.exists(), f'Preserving existing Once result: {target}'
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            shutil.copyfile(batch / 'execution_logs' / (label + '.once.remote.log'),
                            HERE / 'execution_logs' / (label + '.once.remote.log'))
            record['imported'][label] = dict(checked, imported_utc=datetime.now(timezone.utc).isoformat())
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    finished = queue['complete'] == 2 and bool(queue.get('source_hashes_verified_after'))
    if finished:
        assert queue['running'] == queue['failed'] == 0 and len(record['imported']) == 2
        assert queue['source_hashes_verified_before'] and queue['ended_utc']
        assert all(sha(ROOT / name) == digest for name, digest in plan['frozen_source_sha256'].items())
        checks = {job['label']: verify_one(job, HERE / 'runs' / job['label'] / 'once', plan, setup, queue)
                  for job in plan['jobs']}
        acceptance = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                          scope='Two exploratory Once pairs; excluded from Baseline seed counts.',
                          selection=plan['selection'], cases=checks,
                          source_file_count=len(plan['frozen_source_sha256']),
                          sources_verified_before_after_and_locally=True,
                          note='The frozen observer checks Path0 only for Baseline. Its baseline_nonzero_paths=0 does not assert that Once used Path0.')
        (HERE / 'audit_remote_once_acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    report = dict(complete_remote=queue['complete'], running_remote=queue['running'], failed_remote=queue['failed'],
                  newly_imported=new, imported_local=len(record['imported']), running_progress=response['running_progress'],
                  all_finished=finished, queue_terminal=bool(queue.get('ended_utc')))
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    import getpass
    sync(getpass.getpass('SSH password (not stored): '))
