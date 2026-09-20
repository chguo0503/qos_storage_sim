#!/usr/bin/env python3
"""Launch/sync this study only, using an already authenticated SSH master."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
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
POLICIES = ('asu_baseline','od_baseline','once','static','mild','aggressive')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def remote_python(code, payload=None):
    return subprocess.run(SSH+[shlex.join(['python3','-c',code])], input=payload,
                          capture_output=True, check=True).stdout


def launch():
    state = HERE/'execution.json'
    assert not state.exists(), 'Preserve previous launch record'
    audit = json.loads((HERE/'input_audit.json').read_text())
    inputs = {(r['scenario'],r['seed']):r for r in audit['cases']}
    remote_root = f'/home/chguo/work/qos_od_baseline_diverse_20260918_{time.time_ns()}'
    required = list(ROOT.glob('*.py'))+[ROOT/'data',
        ROOT/'results/diverse_data_ssu3_l3_20260916/policy.py',
        ROOT/'results/diverse_data_ssu3_l3_20260916/construct_manifest.py']
    required += [HERE/n for n in ('run_trial.py','metrics.py','run_queue.py','prepare_inputs.py','input_audit.json')]
    required += [ROOT/r['input'] for r in audit['cases']]
    source_hashes = {str(p.relative_to(ROOT)):sha(p) for p in required}
    jobs = []
    for seed in (7,19,43):
        for scenario in ('full','semi'):
            for policy in POLICIES:
                host = 'local' if seed == 43 else 'remote'
                python = sys.executable if host == 'local' else REMOTE_PYTHON
                label = f'{scenario}_{policy}_seed{seed}_{host}'
                jobs.append(dict(label=label, scenario=scenario, seed=seed, policy=policy,host=host,
                    manifest_sha256=inputs[scenario,seed]['manifest_sha256'],
                    argv=[python,f'{REL}/run_trial.py','--manifest',inputs[scenario,seed]['input'],
                          '--policy',policy,'--label',label]))
    formal = dict(seeds=[7,19,43], policies=list(POLICIES), scenarios=['semi','full'],
        num_npu=32,num_ssu=3,disk_GiB_s=40.,placement='block_ring_hash',
        od_CIR_GiB_s=1.25,od_PIR='unlimited',idle_capacity_borrowing=True,
        requested_equal_share_interpretation='equal static CIR; native two-level WRR redistributes unused capacity',
        windows_ms=[[2000,4000],[2000,6000]],input_sha256={r['input']:r['manifest_sha256'] for r in audit['cases']},
        source_sha256=source_hashes,jobs=jobs)
    write(HERE/'formal_plan.json',formal)
    for host, root, parallel, cpus in [('local',str(ROOT),4,[0,2,4,6]),
                                     ('remote',remote_root,8,[0,2,4,6,8,10,12,14])]:
        write(HERE/f'{host}_plan.json',dict(root=root,max_parallel=parallel,physical_cpu_ids=cpus,
            source_sha256=source_hashes,log_dir=f'{REL}/execution_logs',jobs=[j for j in jobs if j['host']==host]))
    package=io.BytesIO()
    with tarfile.open(fileobj=package,mode='w:gz') as tar:
        for path in required+[HERE/'formal_plan.json',HERE/'remote_plan.json']:
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    payload=package.getvalue()
    deploy=f'''
from pathlib import Path
import io,json,sys,tarfile
p=Path({remote_root!r});p.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t.getmembers():
  n=Path(m.name)
  assert m.isfile() and not n.is_absolute() and '..' not in n.parts
  q=p/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(t.extractfile(m).read())
print(json.dumps(dict(remote_root=str(p),deployed=True)))
'''
    print(remote_python(deploy,payload).decode(),end='')
    remote_start=f'''
from pathlib import Path
import json,subprocess
p=Path({remote_root!r});h=p/{REL!r}
with (h/'remote_queue.log').open('x') as f:
 q=subprocess.Popen([{REMOTE_PYTHON!r},str(h/'run_queue.py'),'--plan',str(h/'remote_plan.json')],cwd=p,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(queue_pid=q.pid)))
'''
    remote_launch=json.loads(remote_python(remote_start))
    with (HERE/'local_queue.log').open('x') as stream:
        local=subprocess.Popen([sys.executable,str(HERE/'run_queue.py'),'--plan',str(HERE/'local_plan.json')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(remote_root=remote_root,remote_python=REMOTE_PYTHON,remote_queue_pid=remote_launch['queue_pid'],
                local_queue_pid=local.pid,archive_sha256=hashlib.sha256(payload).hexdigest(),
                jobs=len(jobs),local_jobs=12,remote_jobs=24)
    write(state,record)
    print(json.dumps(record))


def sync():
    record=json.loads((HERE/'execution.json').read_text())
    code=f'''
from pathlib import Path
import io,json,sys,tarfile
p=Path({record['remote_root']!r})/{REL!r}
plan=json.loads((p/'remote_plan.json').read_text());files=[]
for job in plan['jobs']:
 q=p/'runs'/job['label'];c=q/'command.json'
 if not c.exists():continue
 state=json.loads(c.read_text())
 names=['command.json','progress.json','warm_preview.json','manifest.json.gz']
 if state['status']=='complete':names.append('result.json.gz')
 files.extend(q/n for n in names if (q/n).exists())
 if state['status'] in ('failed','pilot_complete'):
  log=p/'execution_logs'/(job['label']+'.log')
  if log.exists():files.append(log)
files.extend(p/n for n in ['remote_plan_status.json','remote_queue.log'] if (p/n).exists())
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
sys.stdout.buffer.write(buf.getvalue())
'''
    data=remote_python(code)
    updated=0
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
        for item in tar.getmembers():
            name=Path(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts
            target=HERE/name;payload=tar.extractfile(item).read()
            if target.exists() and target.read_bytes()==payload:continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.tmp');temporary.write_bytes(payload);temporary.replace(target);updated+=1
    states={}
    previews=[]
    for job in json.loads((HERE/'formal_plan.json').read_text())['jobs']:
        folder=HERE/'runs'/job['label'];cmd=folder/'command.json'
        state=json.loads(cmd.read_text())['status'] if cmd.exists() else 'queued'
        states[state]=states.get(state,0)+1
        if (folder/'warm_preview.json').exists():
            p=json.loads((folder/'warm_preview.json').read_text())
            previews.append(dict(label=job['label'],state=state,U=p['U_percent'],slo=p['slo']['percent']))
    print(json.dumps(dict(updated_files=updated,states=states,warm=previews),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','sync'])
    args=parser.parse_args()
    launch() if args.action=='launch' else sync()
