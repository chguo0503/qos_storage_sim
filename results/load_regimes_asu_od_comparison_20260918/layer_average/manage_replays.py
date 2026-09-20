#!/usr/bin/env python3
"""Launch four local and two remote measurement replays without changing inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REL = HERE.relative_to(ROOT)
SSH = ['ssh', '-S', '/tmp/qos_od_baseline_20260918_mux', '-o', 'BatchMode=yes', 'chguo@192.168.31.126']
REMOTE_PYTHON = '/tmp/qos_random_near_capacity_20260914_z34z48vi/venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def start():
    destination = HERE/'execution.json'
    if destination.exists():
        raise SystemExit('execution.json already exists; use status/sync, not start')
    runner = HERE/'replay_services.py'
    assert runner.exists()
    data = json.loads((HERE.parent/'plot_data.json').read_text())
    cases = []
    local_cpu = iter([0, 2, 4, 6])
    remote_cpu = iter([0, 2])
    for condition in data['conditions']:
        for case in condition['cases']:
            if case['seed'] != 7:
                continue
            location = 'remote' if condition['id']=='full' else 'local'
            cases.append(dict(label=f"{condition['id']}_{case['policy']}_seed7", location=location,
                              source_case=str(Path(condition['source'])/'runs'/case['source_case']),
                              cpu=next(remote_cpu if location=='remote' else local_cpu)))
    remote_root = '/home/chguo/work/qos_layer_average_' + str(time.time_ns())
    paths = set(ROOT.glob('*.py')) | {ROOT/'data', runner}
    for c in cases:
        paths.update((ROOT/c['source_case']).glob('*.json*'))
    archive = HERE/'replay_sources.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(paths):
            tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    command = 'mkdir -p '+shlex.quote(remote_root)+' && tar -xzf - -C '+shlex.quote(remote_root)
    with archive.open('rb') as stream:
        subprocess.run(SSH+[command],stdin=stream,check=True)
    log_dir = HERE/'logs'
    log_dir.mkdir(exist_ok=True)
    execution = dict(remote_root=remote_root, python_local=sys.executable,
                     python_remote=REMOTE_PYTHON, runner_sha256=sha(runner),
                     archive_sha256=sha(archive), cases=cases)
    destination.write_text(json.dumps(execution,indent=2)+'\n')
    launch(execution)


def launch(execution):
    destination=HERE/'execution.json'
    remote_root=execution['remote_root']
    for c in execution['cases']:
        command = ['taskset','-c',str(c['cpu']),
                   REMOTE_PYTHON if c['location']=='remote' else sys.executable,
                   str(REL/'replay_services.py'),'--source-case',c['source_case'],'--label',c['label']]
        c['argv'] = command
        log = REL/'logs'/(c['label']+'.log')
        if c['location']=='local':
            record=HERE/'runs'/c['label']/'command.json'
            if record.exists():
                c['pid']=json.loads(record.read_text())['pid']
                continue
            with (ROOT/log).open('w') as stream:
                process = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL,start_new_session=True,
                                           env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            c['pid'] = process.pid
        else:
            code=("import subprocess,pathlib,json,os; root=pathlib.Path("+repr(remote_root)+
                  "); record=root/"+repr(str(REL/'runs'/c['label']/'command.json'))+
                  "; log=root/"+repr(str(log))+"; log.parent.mkdir(parents=True,exist_ok=True); "
                  "pid=json.loads(record.read_text())['pid'] if record.exists() else "
                  "subprocess.Popen("+repr(command)+",cwd=root,stdin=subprocess.DEVNULL,"
                  "stdout=log.open('w'),stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,"
                  "env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')).pid; print(pid)")
            shell=shlex.join([REMOTE_PYTHON,'-c',code])
            c['pid'] = int(run(SSH+[shell],capture_output=True).stdout.strip())
        destination.write_text(json.dumps(execution,indent=2)+'\n')
    print(json.dumps(execution,indent=2))


def resume_launch():
    launch(json.loads((HERE/'execution.json').read_text()))


def sync():
    execution = json.loads((HERE/'execution.json').read_text())
    (HERE/'runs').mkdir(exist_ok=True)
    for folder in ['runs','logs']:
        run(['rsync','-a','-e',shlex.join(SSH[:-1]),
             SSH[-1]+':'+execution['remote_root']+'/'+str(REL/folder)+'/',str(HERE/folder)+'/'])


def status():
    execution = json.loads((HERE/'execution.json').read_text())
    for c in execution['cases']:
        target = HERE/'runs'/c['label']
        files = sorted(p.name for p in target.glob('*')) if target.exists() else []
        log = HERE/'logs'/(c['label']+'.log')
        tail = log.read_text().splitlines()[-2:] if log.exists() else []
        print(json.dumps(dict(label=c['label'],location=c['location'],files=files,log_tail=tail),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['start','resume_launch','sync','status'])
    args=parser.parse_args()
    globals()[args.action]()
