#!/usr/bin/env python3
"""Bounded local/remote queue for original-simulator 4-second pilot screens."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--host',choices=('local','remote'),required=True)
    args=ap.parse_args()
    plan=json.loads(args.plan.read_text())
    jobs=queue.Queue()
    for row in plan['jobs']:
        if row['host']==args.host:jobs.put(row)
    status=dict(host=args.host,pid=os.getpid(),started=time.time(),jobs={})
    lock=threading.Lock()
    path=HERE/f"pilot_queue_{plan['label']}_{args.host}.json"
    def publish():
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(path)
    def work(cpu):
        while True:
            try:job=jobs.get_nowait()
            except queue.Empty:return
            label=job['label'];target=HERE/'pilots'/label/'command.json'
            if target.exists():
                old=json.loads(target.read_text())
                assert old['status']=='complete_pilot',('Do not duplicate in-progress run',label)
                code=0
            else:
                log=HERE/'execution_logs'/f'{label}.log'
                with lock:
                    status['jobs'][label]=dict(status='running',cpu=cpu);publish()
                with log.open('x')as out:
                    result=subprocess.run(['taskset','-c',str(cpu),sys.executable,'-u',str(HERE/'run_pilot.py'),
                        '--base',str(ROOT/plan['base_manifest']),'--spec',str(HERE/'pilot_specs'/f'{label}.json')],
                        cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
                code=result.returncode
            with lock:
                status['jobs'][label]=dict(status='complete' if code==0 else 'failed',exit_code=code,cpu=cpu)
                if code==0:
                    r=json.loads(target.read_text())['measurement']
                    status['jobs'][label].update(U=r['U_percent'],under=r['demand']['strict_underload_all_disks'],
                        under_initial_4s=r['initial_4s_demand']['strict_underload_all_disks'],mixed=r['mixed_cards'])
                publish();print(json.dumps({label:status['jobs'][label]}),flush=True)
            jobs.task_done()
    with ThreadPoolExecutor(max_workers=len(plan['cpus'][args.host]))as pool:
        list(pool.map(work,plan['cpus'][args.host]))
    status['complete']=True;status['ended']=time.time();publish()


if __name__=='__main__':main()
