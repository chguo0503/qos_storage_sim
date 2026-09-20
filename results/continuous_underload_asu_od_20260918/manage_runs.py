#!/usr/bin/env python3
"""Launch and collect only the 12 immutable document-input baseline runs."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = str(HERE.relative_to(ROOT))
SSH = ['ssh','-S','/tmp/qos_od_baseline_20260918_mux','-o','BatchMode=yes','chguo@192.168.31.126']
REMOTE_PYTHON = '/tmp/qos_random_near_capacity_20260914_z34z48vi/venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def remote(code, data=None):
    return subprocess.run(SSH+[shlex.join(['python3','-c',code])], input=data,
                          capture_output=True, check=True).stdout


def launch():
    assert not (HERE/'execution.json').exists(), 'Preserve previous execution'
    audit = json.loads((HERE/'input_audit.json').read_text())
    inputs = {(x['order'], x['seed']):x for x in audit['cases']}
    files = list(ROOT.glob('*.py'))+[ROOT/'data', ROOT/'docs/continuous_underload_reproduction.md']
    files += [HERE/name for name in ('run_trial.py','metrics.py','run_queue.py','prepare_inputs.py','input_audit.json')]
    files += [ROOT/x[key] for x in audit['cases'] for key in ('manifest','csv')]
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    remote_root = f'/home/chguo/work/qos_continuous_underload_20260918_{time.time_ns()}'
    jobs=[]
    for seed in (7,19,43):
        host='local' if seed==43 else 'remote'
        for order in ('random','ordered'):
            for policy in ('asu_baseline','od_baseline'):
                label=f'{order}_{policy}_seed{seed}_{host}'
                jobs.append(dict(label=label, order=order, seed=seed, policy=policy, host=host,
                    manifest_sha256=inputs[order,seed]['manifest_sha256'],
                    argv=[sys.executable if host=='local' else REMOTE_PYTHON,
                          f'{REL}/run_trial.py','--manifest',inputs[order,seed]['manifest'],
                          '--policy',policy,'--label',label]))
    formal=dict(seeds=[7,19,43], orders=['random','ordered'], policies=['asu_baseline','od_baseline'],
                num_npu=32,num_ssu=3,disk_GiB_s=40.,layers=8,requests=640,
                placement='block_ring_hash', all_arrivals_at_zero=True,
                preserve_document_profiles_and_queues=True, underload_is_measured_not_assumed=True,
                windows_ms=[[2000,4000],[2000,6000]], od_CIR_GiB_s=1.25, od_PIR='unlimited',
                source_sha256=hashes, jobs=jobs)
    write(HERE/'formal_plan.json',formal)
    for host,root,cpus in [('local',str(ROOT),[0,2,4,6]),('remote',remote_root,[0,2,4,6,8,10,12,14])]:
        write(HERE/f'{host}_plan.json',dict(root=root,max_parallel=len(cpus),physical_cpu_ids=cpus,
              source_sha256=hashes,log_dir=f'{REL}/execution_logs',jobs=[j for j in jobs if j['host']==host]))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for p in files+[HERE/'formal_plan.json',HERE/'remote_plan.json']:
            tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    payload=buf.getvalue()
    (HERE/'source_and_inputs.tar.gz').write_bytes(payload)
    deploy=f'''
from pathlib import Path
import io,sys,tarfile
p=Path({remote_root!r});p.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t.getmembers():
  n=Path(m.name)
  assert m.isfile() and not n.is_absolute() and '..' not in n.parts
  q=p/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
'''
    remote(deploy,payload)
    start=f'''
from pathlib import Path
import json,subprocess
p=Path({remote_root!r});h=p/{REL!r}
with (h/'remote_queue.log').open('x') as f:
 q=subprocess.Popen([{REMOTE_PYTHON!r},str(h/'run_queue.py'),'--plan',str(h/'remote_plan.json')],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=q.pid)))
'''
    remote_pid=json.loads(remote(start))['pid']
    with (HERE/'local_queue.log').open('x') as f:
        local=subprocess.Popen([sys.executable,str(HERE/'run_queue.py'),'--plan',str(HERE/'local_plan.json')],
              cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(remote_root=remote_root,remote_python=REMOTE_PYTHON,remote_queue_pid=remote_pid,
               local_queue_pid=local.pid,archive_sha256=hashlib.sha256(payload).hexdigest(),jobs=12)
    write(HERE/'execution.json',state)
    print(json.dumps(state))


def sync():
    state=json.loads((HERE/'execution.json').read_text())
    code=f'''
from pathlib import Path
import io,json,sys,tarfile
p=Path({state['remote_root']!r})/{REL!r}
plan=json.loads((p/'remote_plan.json').read_text());files=[]
for j in plan['jobs']:
 q=p/'runs'/j['label'];c=q/'command.json'
 if not c.exists():continue
 status=json.loads(c.read_text())['status']
 names=['command.json','progress.json','warm_preview.json','manifest.json.gz']
 if status=='complete':names.append('result.json.gz')
 files.extend(q/n for n in names if (q/n).exists())
 if status.startswith('failed'):
  log=p/'execution_logs'/(j['label']+'.log')
  if log.exists():files.append(log)
files.extend(p/n for n in ['remote_plan_status.json','remote_queue.log'] if (p/n).exists())
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
sys.stdout.buffer.write(buf.getvalue())
'''
    data=remote(code);updated=0
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
        for item in tar.getmembers():
            name=Path(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts
            path=HERE/name;payload=tar.extractfile(item).read()
            if path.exists() and path.read_bytes()==payload:continue
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(payload);tmp.replace(path);updated+=1
    counts={};warm=[]
    for job in json.loads((HERE/'formal_plan.json').read_text())['jobs']:
        p=HERE/'runs'/job['label'];c=p/'command.json'
        status=json.loads(c.read_text())['status'] if c.exists() else 'queued'
        counts[status]=counts.get(status,0)+1
        if (p/'warm_preview.json').exists():
            x=json.loads((p/'warm_preview.json').read_text())
            warm.append(dict(label=job['label'],status=status,U=x['U_percent'],SLO15=x['slo']['percent'],
                        demand_max=x['demand']['per_disk_max_GiB_s'],strict_underload=x['demand']['strict_underload_all_disks']))
    print(json.dumps(dict(states=counts,updated_files=updated,warm=warm),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['launch','sync'])
    args=parser.parse_args();launch() if args.action=='launch' else sync()
