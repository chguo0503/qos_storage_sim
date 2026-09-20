#!/usr/bin/env python3
"""Hash-verify and atomically mirror only the new original-Once controls."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = str(HERE.relative_to(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', default='/tmp/qos_diverse_once_20260916_mux')
    args = parser.parse_args()
    plan = read(HERE/'once_control_remote_plan.json')
    code = r'''
from pathlib import Path
import io,json,tarfile,hashlib,sys
H=Path(REMOTE)/REL
plan=json.loads((H/'once_control_remote_plan.json').read_text());files={};states={}
for job in plan['jobs']:
 p=H/'runs'/job['label'];c=p/'command.json'
 if not c.exists():continue
 data=c.read_bytes();record=json.loads(data);states[job['label']]=record['status']
 names=['command.json','progress.json','warm_preview.json','manifest.json.gz']
 if record['status']=='complete':names.append('result.json.gz')
 for name in names:
  f=p/name
  if f.exists():
   blob=data if name=='command.json' else f.read_bytes()
   if name.endswith('.json'):json.loads(blob)
   files[str(f.relative_to(H))]=blob
 if record['status']=='complete':
  log=H/'once_control_remote_logs'/(job['label']+'.log')
  if log.exists():files[str(log.relative_to(H))]=log.read_bytes()
for name in ('once_control_remote_plan_status.json','once_control_remote_launch.json','once_control_remote_queue.log'):
 f=H/name
 if f.exists():files[name]=f.read_bytes()
audit={'states':states,'files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}
files['transport_manifest.json']=(json.dumps(audit,indent=2)+chr(10)).encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
 for name,blob in files.items():
  info=tarfile.TarInfo(name);info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
sys.stdout.buffer.write(buf.getvalue())
'''
    code = 'REMOTE='+repr(plan['root'])+'\nREL='+repr(REL)+'\n'+code
    response = subprocess.run(['ssh', '-S', args.socket, '-o', 'BatchMode=yes',
                               'chguo@192.168.31.126', shlex.join(['python3', '-c', code])],
                              capture_output=True)
    assert response.returncode == 0, response.stderr.decode()
    files = {}
    with tarfile.open(fileobj=io.BytesIO(response.stdout), mode='r:gz') as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            files[str(name)] = tar.extractfile(member).read()
    audit = json.loads(files.pop('transport_manifest.json'))
    assert set(files) == set(audit['files'])
    assert all(hashlib.sha256(files[name]).hexdigest() == digest
               for name, digest in audit['files'].items())
    jobs = {job['label']: job for job in plan['jobs']}
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    for label, status in audit['states'].items():
        prefix = 'runs/'+label+'/'
        command = json.loads(files[prefix+'command.json'])
        assert command['policy'] == 'once'
        job = jobs[label]
        source_manifest = ROOT/job['argv'][job['argv'].index('--manifest')+1]
        assert hashlib.sha256(files[prefix+'manifest.json.gz']).hexdigest() == command['manifest_sha256'] == sha(source_manifest)
        control = command['once_control']
        assert control['runner_sha256'] == sha(HERE/'run_once_control.py')
        assert not control['new_candidate_pool_extension_installed']
        if status == 'complete':
            assert command['core_unchanged'] and command['extension_unchanged']
            assert all(command['checks'].values()) and control['checks_passed']
            assert command['observed_blocks'] == command['expected_blocks']
            assert hashlib.sha256(files[prefix+'result.json.gz']).hexdigest() == command['result_sha256']
            result = json.loads(gzip.decompress(files[prefix+'result.json.gz']))
            assert all(result['summary']['invariants'].values())
            assert result['routing_statistics']['calls'] == 0
    changed = 0
    # Publish complete commands last, after their referenced results exist.
    for name, blob in sorted(files.items(), key=lambda item: (Path(item[0]).name == 'command.json', item[0])):
        dest = HERE/name
        if dest.exists():
            previous = dest.read_bytes()
            if previous == blob:
                continue
            assert dest.name not in ('manifest.json.gz', 'warm_preview.json', 'result.json.gz'), name
            if dest.name == 'command.json':
                assert json.loads(previous)['status'] != 'complete', name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name+'.sync_tmp')
        tmp.write_bytes(blob)
        tmp.replace(dest)
        changed += 1
    audit.update(archive_sha256=hashlib.sha256(response.stdout).hexdigest(),
                 source_input_result_hashes_verified=True)
    (HERE/'once_control_remote_sync.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(dict(updated_files=changed, states=audit['states'])))


if __name__ == '__main__':
    main()
