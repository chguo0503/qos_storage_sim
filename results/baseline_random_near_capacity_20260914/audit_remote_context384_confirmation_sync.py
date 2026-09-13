#!/usr/bin/env python3
"""Import four context-scale Baseline cases, with four predeclared seeds and no trace."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shlex
import shutil
import tarfile

from audit_remote_sync import HERE, ROOT, read, sha, transport


def remote_json(code, password):
    arguments = ['-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5',
                 '-o', 'NumberOfPasswordPrompts=1', '192.168.31.126',
                 'python3 -c ' + shlex.quote(code)]
    output = transport('ssh', arguments, password, timeout=360)
    return json.loads(output.split('AUDIT_JSON:', 1)[1].strip())


def verify_one(job, directory, plan, setup, queue):
    command = read(directory / 'command.json')
    result = read(directory / 'result.json.gz')
    physical = read(directory / 'physical_service.json')
    manifest = read(directory / 'manifest.json.gz')
    metadata = manifest['metadata']
    label = job['label']
    assert command['status'] == queue['jobs'][label]['status'] == 'complete'
    assert command['returncode'] == 0 and command['completed_simulation']
    assert command['strategy'] == 'baseline' and command['order'] == 'random' and command['label'] == label
    assert command['trace_window_ms'] == job['trace_window_ms']
    assert sha(directory / 'command.json') == queue['jobs'][label]['command_sha256']
    assert sha(directory / 'manifest.json.gz') == sha(ROOT / job['manifest']) == job['manifest_sha256'] == command['manifest_sha256']
    assert command['input_fingerprint'] == job['input_fingerprint'] == result['input_fingerprint'] == manifest['input_fingerprint']
    assert command['core_source_sha256'] == result['core_and_policy_sha256'] == metadata['core_source_sha256'] == setup['environment']['core_source_sha256']
    assert command['runner_sha256'] == metadata['frozen_experiment_sha256'] == plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/experiment.py']
    assert command['source_data_sha256'] == plan['frozen_source_sha256']['data']
    assert sha(directory / 'result.json.gz') == command['output_sha256']
    assert sha(directory / 'physical_service.json') == command['physical_service_sha256']
    assert physical['observed_completed_blocks'] == physical['expected_blocks'] == command['observed_completed_blocks'] == command['expected_blocks'] == job['expected_blocks']
    assert physical['baseline_nonzero_paths'] == 0
    assert len(physical['ssd_busy_ms']) == 8 and len(physical['npu_link_busy_ms']) == 32
    assert result['summary']['invariants'] and all(value is True for value in result['summary']['invariants'].values())
    assert [(w['start_ms'], w['end_ms']) for w in command['windows']] == [(2000, 4000), (2000, 20000)]
    assert command['host'] == queue['host']
    assert queue['jobs'][label]['total_owned_simulations_upper_bound_at_start'] <= 6
    assert metadata['family'] == 'data_affine_context_scale' and metadata['constructed_profile'] is True
    assert metadata['num_npu'] == 32 and metadata['n_layers'] == 8 and metadata['num_ssu'] == 8
    assert metadata['constructed_long_profile'] == job['constructed_long_profile']
    assert metadata['constructed_short_profile'] is True
    assert result['metadata']['profile_is_constructed_by_role'] == metadata['profile_is_constructed_by_role']
    assert metadata['constructed_builder_sha256'] == job['constructed_builder_sha256'] == plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/prepare_context_scale.py']
    assert metadata['constructed_plan_sha256'] == job['constructed_plan_sha256'] == plan['frozen_source_sha256']['results/baseline_random_near_capacity_20260914/long_scale_math.json']
    for row in manifest['requests']:
        load = row['load']
        if load['role'] == 'S':
            assert load['constructed_profile'] is True and load['source_ttft_ms'] is None
            assert load['total_tokens'] == 10 * 1024 and load['nql'] == 128
            assert load['profile_construction']['method'] == 'affine_extrapolation_below_raw_minimum'
        else:
            assert load['role'] == 'L' and load['constructed_profile'] == job['constructed_long_profile']
            assert load['total_tokens'] == job['long_total_k'] * 1024 and load['nql'] == 1024
            assert (load['source_ttft_ms'] is None) == job['constructed_long_profile']
            assert load['profile_construction']['method'] == ('affine_extrapolation_above_raw_maximum' if job['constructed_long_profile'] else 'direct_data_row')
    checked = dict(command_sha256=sha(directory / 'command.json'), manifest_sha256=command['manifest_sha256'],
                   result_sha256=command['output_sha256'], physical_service_sha256=command['physical_service_sha256'],
                   host=command['host'], python_version=command['python_version'], windows=command['windows'],
                   all_hashes_blocks_and_invariants_verified=True, per_role_raw_or_extrapolated_provenance_verified=True)
    if job['trace']:
        validation = read(directory / 'trace_validation.json')
        assert validation['passed'] and all(value is True for value in validation['checks'].values())
        assert validation['verifier_sha256'] == setup['trace_validator']['sha256'] == sha(HERE / 'audit_remote_trace_check.py')
        assert sha(directory / 'trace.json.gz') == validation['trace_sha256'] == command['trace_sha256']
        assert validation['window_ms'] == job['trace_window_ms']
        assert validation['retained_blocks'] == command['retained_blocks'] > 0
        assert validation['observed_completed_blocks'] == validation['expected_completed_blocks'] == command['expected_blocks']
        assert validation['source']['manifest_sha256'] == command['manifest_sha256']
        assert validation['source']['reference_sha256'] == command['output_sha256']
        assert validation['source']['observer_source_sha256'] == command['runner_sha256']
        assert validation['source']['core_source_sha256'] == command['core_source_sha256']
        checked.update(trace_sha256=command['trace_sha256'], trace_validation_sha256=sha(directory / 'trace_validation.json'),
                       retained_blocks=command['retained_blocks'], remote_all_retained_rows_verified=True,
                       local_trace_bytes_match_remote_verified_hash=True)
    else:
        assert command['retained_blocks'] == 0
    return checked


def sync(password):
    plan = read(HERE / 'audit_remote_context384_confirmation_plan.json')
    setup = read(HERE / 'audit_remote_setup.json')
    record_path = HERE / 'audit_remote_context384_confirmation_sync_record.json'
    record = read(record_path) if record_path.exists() else {'selection': plan['selection'], 'imported': {}}
    remaining = [job['label'] for job in plan['jobs'] if job['label'] not in record['imported']]
    code = '''from pathlib import Path
import json,tarfile,hashlib,time,subprocess
D=Path(__DIR__);H=D/'results/baseline_random_near_capacity_20260914'
Q=json.loads((H/'audit_remote_context384_confirmation_status.json').read_text());remaining=__REMAINING__
plan=json.loads((H/'audit_remote_context384_confirmation_plan.json').read_text());jobs={j['label']:j for j in plan['jobs']}
done=[label for label in remaining if Q['jobs'][label]['status']=='complete'];progress={}
for label,v in Q['jobs'].items():
 if v['status']=='running':
  p=H/'runs'/label/'baseline/progress.json'
  if p.exists():progress[label]=json.loads(p.read_text())
out={'queue':Q,'new_complete':done,'running_progress':progress}
if done:
 checker=H/'audit_remote_trace_check.py'
 assert hashlib.sha256(checker.read_bytes()).hexdigest()==__CHECKER_SHA__
 for label in done:
  if jobs[label]['trace']:
   process=subprocess.run([str(D/'venv/bin/python'),str(checker),str(H/'runs'/label/'baseline')],capture_output=True,text=True,timeout=300)
   assert process.returncode==0,process.stderr+process.stdout
 A=D/('audit_context384_confirmation_sync_'+str(time.time_ns())+'.tar')
 with tarfile.open(A,'w') as t:
  for label in done:
   P=H/'runs'/label/'baseline'
   names=['manifest.json.gz','command.json','result.json.gz','physical_service.json','progress.json']
   if jobs[label]['trace']:names+=['trace.json.gz','trace_validation.json']
   for name in names:t.add(P/name,arcname=label+'/baseline/'+name,recursive=False)
   t.add(H/'execution_logs'/(label+'.context384_confirmation.remote.log'),arcname='execution_logs/'+label+'.context384_confirmation.remote.log',recursive=False)
 out.update(archive=str(A),archive_sha256=hashlib.sha256(A.read_bytes()).hexdigest())
print('AUDIT_JSON:'+json.dumps(out))
'''.replace('__DIR__', repr(setup['remote_dir'])).replace('__REMAINING__', repr(remaining)).replace('__CHECKER_SHA__', repr(setup['trace_validator']['sha256']))
    response = remote_json(code, password)
    queue = response['queue']
    assert queue['plan_sha256'] == sha(HERE / 'audit_remote_context384_confirmation_plan.json')
    assert queue['queue_runner_sha256'] == sha(HERE / 'audit_remote_context384_confirmation_queue.py')
    assert queue['launch_record_sha256'] == sha(HERE / 'audit_remote_context384_confirmation_launch.json')
    launch = read(HERE / 'audit_remote_context384_confirmation_launch.json')
    assert launch['construction_audit_passed'] and sha(HERE / launch['audit_report']) == launch['audit_report_sha256']
    (HERE / 'audit_remote_context384_confirmation_status.json').write_text(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
    new = response['new_complete']
    if new:
        incoming = HERE / 'runtime_validation/remote_context384_confirmation_incoming'
        incoming.mkdir(parents=True, exist_ok=True)
        archive = incoming / Path(response['archive']).name
        transport('scp', ['-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-o', 'NumberOfPasswordPrompts=1',
                         'chguo@192.168.31.126:' + response['archive'], str(archive)], password, timeout=180)
        assert sha(archive) == response['archive_sha256']
        batch = incoming / archive.stem
        batch.mkdir(exist_ok=False)
        by_label = {job['label']: job for job in plan['jobs']}
        expected = {label + '/baseline/' + name for label in new for name in
                    ('manifest.json.gz', 'command.json', 'result.json.gz', 'physical_service.json', 'progress.json')}
        expected.update(label + '/baseline/' + name for label in new if by_label[label]['trace']
                        for name in ('trace.json.gz', 'trace_validation.json'))
        expected.update('execution_logs/' + label + '.context384_confirmation.remote.log' for label in new)
        with tarfile.open(archive) as stream:
            assert {member.name for member in stream.getmembers()} == expected
            assert all(member.isfile() for member in stream.getmembers())
            stream.extractall(batch)
        for label in new:
            checked = verify_one(by_label[label], batch / label / 'baseline', plan, setup, queue)
            target = HERE / 'runs' / label
            assert not target.exists(), f'Preserving existing context-scale result: {target}'
            os.replace(batch / label, target)
            shutil.copyfile(batch / 'execution_logs' / (label + '.context384_confirmation.remote.log'),
                            HERE / 'execution_logs' / (label + '.context384_confirmation.remote.log'))
            record['imported'][label] = dict(checked, imported_utc=datetime.now(timezone.utc).isoformat())
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    finished = queue['complete'] == 4 and bool(queue.get('source_hashes_verified_after'))
    if finished:
        assert queue['running'] == queue['failed'] == 0 and len(record['imported']) == 4
        assert queue['source_hashes_verified_before'] and queue['ended_utc']
        assert all(sha(ROOT / name) == digest for name, digest in plan['frozen_source_sha256'].items())
        checks = {job['label']: verify_one(job, HERE / 'runs' / job['label'] / 'baseline', plan, setup, queue)
                  for job in plan['jobs']}
        acceptance = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                          scope='Four predeclared context384 Baseline confirmation seeds; no trace. Extrapolated contexts are not hardware measurements.',
                          selection=plan['selection'], cases=checks, source_file_count=len(plan['frozen_source_sha256']),
                          sources_verified_before_after_and_locally=True,
                          trace_validation='No trace requested in these four confirmation runs. The separate seed7 trace replay is handled locally by root.')
        (HERE / 'audit_remote_context384_confirmation_acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    report = dict(complete_remote=queue['complete'], running_remote=queue['running'], failed_remote=queue['failed'],
                  newly_imported=new, imported_local=len(record['imported']), running_progress=response['running_progress'],
                  all_finished=finished, queue_terminal=bool(queue.get('ended_utc')))
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    import getpass
    sync(getpass.getpass('SSH password (not stored): '))
