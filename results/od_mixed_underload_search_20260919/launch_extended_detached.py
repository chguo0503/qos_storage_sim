#!/usr/bin/env python3
"""Detach one pre-authorized run, closing every SSH descriptor in the child."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--policy', choices=('asu_baseline','od_baseline','once'), required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--cpu', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    assert Path(args.label).name == args.label and args.label not in ('.','..')
    manifest = args.manifest.resolve()
    assert manifest.is_file() and not (HERE/'runs'/args.label).exists()
    logs = HERE/'execution_logs'; logs.mkdir(exist_ok=True)
    log = logs/(args.label+'.log'); status = logs/(args.label+'.launch.json')
    assert not status.exists()
    argv = ['taskset','-c',str(args.cpu),sys.executable,'-u',str(HERE/'run_extended_trial.py'),
            '--manifest',str(manifest),'--policy',args.policy,'--label',args.label]
    if args.smoke:
        argv.append('--smoke')
    with log.open('x') as stream:
        child = subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    record = dict(pid=child.pid,argv=argv,log=str(log),cwd=str(ROOT),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  detached=True,all_standard_fds_redirected=True)
    with status.open('x') as out:
        json.dump(record,out,indent=2);out.write('\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
