#!/usr/bin/env python3
"""Guarded cleanup of the single explicitly authorized remote experiment directory."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import tarfile

from audit_remote_sync import HERE, ROOT, read, remote_json

AUTHORIZED_DIRECTORY = '/tmp/qos_random_near_capacity_20260914_z34z48vi'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(password, execute=False):
    output = HERE / 'remote_cleanup.json'
    assert not output.exists(), 'Preserve existing cleanup evidence; inspect it before any retry.'
    setup = read(HERE / 'audit_remote_setup.json')
    assert setup['remote_dir'] == AUTHORIZED_DIRECTORY
    final = read(HERE / 'audit_remote_final_sources.json')
    concurrency = read(HERE / 'audit_remote_concurrency_check.json')
    assert final['passed'] and final['unique_queue_jobs'] == 54 and final['source_file_count'] == 144
    assert concurrency['passed'] and concurrency['job_count'] == 54 and concurrency['peak_parallel_jobs'] <= 6
    assert sha(HERE / 'audit_remote_concurrency_check.json') == final['concurrency_check_sha256']
    protected = {str(HERE / name): sha(HERE / name) for name in
                 ('audit_remote_final_sources.json', 'audit_remote_concurrency_check.json')}
    local_artifacts, remote_artifacts, identities = {}, {}, set()

    def retain(path, expected=None, remote_relative=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        local_artifacts[str(path)] = digest
        if remote_relative is not None:
            assert remote_relative not in remote_artifacts or remote_artifacts[remote_relative] == digest
            remote_artifacts[remote_relative] = digest
        return digest

    def validate_case(directory, expected_manifest, remote_case):
        command = read(directory / 'command.json')
        assert command['status'] == 'complete' and command['returncode'] == 0 and command['completed_simulation']
        assert command['observed_completed_blocks'] == command['expected_blocks']
        names = {'command.json': None, 'manifest.json.gz': expected_manifest,
                 'result.json.gz': command['output_sha256'], 'physical_service.json': command['physical_service_sha256']}
        assert command['manifest_sha256'] == expected_manifest
        if command.get('trace_sha256'):
            names['trace.json.gz'] = command['trace_sha256']
            names['trace_validation.json'] = None
            validation = read(directory / 'trace_validation.json')
            assert validation.get('all_checks_passed', validation.get('passed', False))
        for name, digest in names.items():
            retain(directory / name, digest, remote_case + '/' + name)
        return command

    for evidence in final['evidence']:
        for key in ('plan', 'status', 'acceptance'):
            path = HERE / evidence[key]
            assert sha(path) == evidence[key + '_sha256']
            protected[str(path)] = sha(path)
        plan = read(HERE / evidence['plan']); status = read(HERE / evidence['status'])
        accepted = read(HERE / evidence['acceptance'])
        assert accepted['passed'] and status['ended_utc']
        assert status['running'] == status['failed'] == 0 and status['complete'] == len(plan['jobs'])
        assert status['source_hashes_verified_before'] and status['source_hashes_verified_after']
        for name in (evidence['plan'], evidence['status']):
            retain(HERE / name, remote_relative='results/baseline_random_near_capacity_20260914/' + name)
        for job in plan['jobs']:
            identity = job['label'], job['strategy']
            assert identity not in identities
            identities.add(identity)
            assert status['jobs'][job['label']]['status'] == 'complete'
            directory = HERE / 'runs' / job['label'] / job['strategy']
            relative = str(directory.relative_to(ROOT))
            command = validate_case(directory, job['manifest_sha256'], relative)
            assert sha(directory / 'command.json') == status['jobs'][job['label']]['command_sha256']
            assert command['strategy'] == job['strategy'] and command['input_fingerprint'] == job['input_fingerprint']
            analysis = read(directory / 'analysis.json')
            assert analysis['all_technical_checks_passed'] and analysis['input_fingerprint'] == job['input_fingerprint']
            retain(directory / 'analysis.json')
    assert len(identities) == 54

    replay = read(HERE / 'runtime_validation/main512_cross_host_checks.json')
    assert replay['all_scientific_fields_equal'] and replay['same_manifest_bytes'] and replay['same_core_and_runner_hashes']
    retain(HERE / 'runtime_validation/main512_cross_host_checks.json')
    replay_dir = HERE / 'runtime_validation/remote_main512'
    remote_replay = 'results/baseline_random_near_capacity_20260914/runs/main512_ssu3_h22000_seed7/baseline'
    validate_case(replay_dir, setup['remote_main512_validation']['manifest_sha256'], remote_replay)
    retain(replay_dir / 'command.json', replay['remote_command_sha256'])
    retain(replay_dir / 'result.json.gz', replay['remote_result_sha256'])
    local_reference = HERE / 'runs/main512_ssu3_h22000_seed7/baseline'
    retain(local_reference / 'command.json', replay['local_command_sha256'])
    retain(local_reference / 'result.json.gz', replay['local_result_sha256'])

    archive = HERE / 'audit_remote_sources.tar.gz'
    retain(archive, setup['bundle_sha256'])
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        assert len(members) == len(setup['frozen_source_sha256']) == 106
        assert {m.name for m in members} == set(setup['frozen_source_sha256'])
        assert all(m.isfile() for m in members)
        for member in members:
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == setup['frozen_source_sha256'][member.name]

    report = dict(status='local_validation_complete', checked_utc=datetime.now(timezone.utc).isoformat(),
        authorized_scope=AUTHORIZED_DIRECTORY, host=setup['host'], user=setup['user'], execute_requested=execute,
        local_checks=dict(all_passed=True, unique_queue_jobs=54, separate_cross_host_replay=1,
            completed_queue_count=len(final['evidence']), frozen_source_union_count=144,
            source_archive_sha256=sha(archive), source_archive_members_verified=106,
            local_artifact_sha256=local_artifacts, frozen_audit_sha256=protected,
            original_retained_traces_verified=sum(p.endswith('/trace.json.gz') for p in local_artifacts)),
        builder_sha256=sha(__file__))
    code = '''from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
D=Path(__DIRECTORY__);expected=__EXPECTED__;execute=__EXECUTE__
assert str(D)=='/tmp/qos_random_near_capacity_20260914_z34z48vi'
assert D.parent==Path('/tmp') and D.exists() and not D.is_symlink() and D.resolve()==D
assert D.stat().st_uid==os.getuid()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
 return h.hexdigest()
for name,value in expected.items():
 p=D/name
 assert p.is_relative_to(D) and p.is_file() and digest(p)==value,name
ancestors=set();pid=os.getpid()
while pid>0 and pid not in ancestors:
 ancestors.add(pid)
 try:pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
 except FileNotFoundError:break
matches=[];uncertain=[]
def inside(value):return value==str(D) or value.startswith(str(D)+'/')
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) in ancestors:continue
 try:
  if proc.stat().st_uid!=os.getuid():continue
  reasons=[]
  for name in ('cwd','exe'):
   try:
    if inside(os.readlink(proc/name)):reasons.append(name)
   except FileNotFoundError:pass
  args=(proc/'cmdline').read_bytes().split(b'\\0')
  if any(str(D).encode() in arg for arg in args):reasons.append('command_argument')
  for fd in (proc/'fd').iterdir():
   try:
    if inside(os.readlink(fd)):reasons.append('open_file');break
   except FileNotFoundError:pass
  if reasons:matches.append(dict(pid=int(proc.name),reasons=sorted(set(reasons))))
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError:uncertain.append(int(proc.name))
assert not matches and not uncertain,{'active_processes':matches,'uninspectable_owned_processes':uncertain}
files=0;bytes_total=0;directories=0
for parent,dirs,names in os.walk(D,followlinks=False):
 directories+=1
 for name in names:
  p=Path(parent)/name;files+=1;bytes_total+=p.lstat().st_size
before=datetime.now(timezone.utc).isoformat()
if execute:shutil.rmtree(D)
out=dict(remote_checks_passed=True,exact_directory=str(D),owned_by_uid=os.getuid(),
 compared_artifact_count=len(expected),active_owned_directory_processes=matches,
 uninspectable_owned_processes=uncertain,pre_delete_file_count=files,
 pre_delete_directory_count=directories,pre_delete_logical_bytes=bytes_total,
 removed=execute and not D.exists(),exists_after=D.exists(),
 deletion_started_utc=before,finished_utc=datetime.now(timezone.utc).isoformat(),
 deletion_method='shutil.rmtree on exact nonsymlink owned directory; no other path modified')
assert not execute or out['removed']
print('AUDIT_JSON:'+json.dumps(out))
'''.replace('__DIRECTORY__', repr(AUTHORIZED_DIRECTORY)).replace('__EXPECTED__', repr(remote_artifacts)).replace('__EXECUTE__', repr(execute))
    try:
        report['remote'] = remote_json(code, password)
        assert all(sha(Path(name)) == value for name, value in protected.items())
        assert all(sha(Path(name)) == value for name, value in local_artifacts.items())
        report['local_artifacts_and_frozen_audits_unchanged_after'] = True
        report['status'] = 'removed_authorized_remote_directory' if execute else 'checked_without_deletion'
        report['all_checks_passed'] = True
    except Exception as exc:
        report['status'] = 'not_confirmed_removed_inspect_before_retry'
        report['all_checks_passed'] = False
        report['error'] = repr(exc)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'all_checks_passed', 'remote')}, ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    import getpass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Delete only after every local and remote check passes.')
    args = parser.parse_args()
    main(getpass.getpass('SSH password (not stored): '), execute=args.execute)
