#!/usr/bin/env python3
"""Offline report finalization; never launches or changes a simulation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def main():
    started=time.time()
    destination=HERE/'report_finalization_status.json'
    while time.time()-started<7200:
        records=read(HERE/'audit_existing/long_runtime_audit_status.json').get('records',{})
        render=read(HERE/'figures/separate/render_status.json')
        complete=len(records)==5 and all(r.get('status')=='audited' for r in records.values())
        figures=render.get('complete')==4 and (HERE/'figures/separate/README.md').exists()
        state=dict(status='waiting',audits_complete=complete,figures_complete=figures,
                   updated_unix=time.time(),no_simulation_run=True)
        if complete and figures:
            command=[sys.executable,'-B',str(HERE/'build_report.py')]
            p=subprocess.run(command,cwd=HERE,capture_output=True,text=True)
            (HERE/'report_build_stdout.log').write_text(p.stdout)
            (HERE/'report_build_stderr.log').write_text(p.stderr)
            state.update(status='complete' if p.returncode==0 else 'report_build_error',
                         returncode=p.returncode,command=command)
            if p.returncode==0:
                state['report_sha256']=hashlib.sha256((HERE/'report.md').read_bytes()).hexdigest()
            destination.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(state,ensure_ascii=False),flush=True)
            return p.returncode
        destination.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
        time.sleep(30)
    destination.write_text(json.dumps(dict(status='watch_timeout',no_simulation_run=True),indent=2)+'\n')
    return 1

if __name__=='__main__':
    raise SystemExit(main())
