#!/usr/bin/env python3
"""Atomically mirror only this study's remote artifacts over an SSH master."""
from pathlib import Path
import io
import json
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
REMOTE=json.loads((HERE/'remote_execution.json').read_text())['remote_dir']
SCRIPT=r'''
from pathlib import Path
import io,json,sys,tarfile
p=Path(REMOTE)/'results/diverse_data_ssu3_l3_20260916'
plan=json.loads((p/'formal_remote_plan.json').read_text())
files=[]
for job in plan['jobs']:
 q=p/'runs'/job['label'];c=q/'command.json'
 if not c.exists():continue
 state=json.loads(c.read_text())
 names=['command.json','progress.json','warm_preview.json','manifest.json.gz']
 if state['status']=='complete':names.append('result.json.gz')
 files.extend(q/n for n in names if (q/n).exists())
 log=p/'execution_logs'/(job['label']+'.log')
 if log.exists():files.append(log)
files.extend(p/n for n in ['formal_remote_plan_status.json','formal_remote_launch.json','formal_remote_queue.log'] if (p/n).exists())
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
sys.stdout.buffer.write(buf.getvalue())
'''


def main():
    code='REMOTE='+repr(REMOTE)+'\n'+SCRIPT
    r=subprocess.run(['ssh','-S','/tmp/qos_diverse_20260916_mux',
                      '-o','BatchMode=yes','chguo@192.168.31.126',
                      shlex.join(['python3','-c',code])],capture_output=True,check=True)
    count=0
    with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:
        for m in t.getmembers():
            name=Path(m.name)
            assert m.isfile() and not name.is_absolute() and '..' not in name.parts
            dest=HERE/name;dest.parent.mkdir(parents=True,exist_ok=True)
            payload=t.extractfile(m).read()
            if dest.exists() and dest.read_bytes()==payload:continue
            tmp=dest.with_name(dest.name+'.sync_tmp');tmp.write_bytes(payload);tmp.replace(dest);count+=1
    states={}
    for f in HERE.glob('runs/*_remote/command.json'):
        states[f.parent.name]=json.loads(f.read_text())['status']
    print(json.dumps(dict(updated_files=count,states=states)))


if __name__=='__main__':main()
