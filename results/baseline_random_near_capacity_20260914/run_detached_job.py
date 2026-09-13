#!/usr/bin/env python3
"""Monitor one declared local command and persist its exit status."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess


def main():
    p=ArgumentParser();p.add_argument('job',type=Path);a=p.parse_args()
    job=json.loads(a.job.read_text());status=Path(job['status']);log=Path(job['log'])
    assert not status.exists()
    state=dict(started_utc=datetime.now(timezone.utc).isoformat(),status='running',
        monitor_pid=os.getpid(),argv=job['argv'],job_sha256=hashlib.sha256(a.job.read_bytes()).hexdigest())
    def save():
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
    with log.open('w') as f:
        process=subprocess.Popen(job['argv'],cwd=job['cwd'],stdout=f,stderr=subprocess.STDOUT)
        state['pid']=process.pid;save();code=process.wait()
    state.update(returncode=code,status='complete' if code==0 else 'failed',ended_utc=datetime.now(timezone.utc).isoformat());save()


if __name__=='__main__':main()
