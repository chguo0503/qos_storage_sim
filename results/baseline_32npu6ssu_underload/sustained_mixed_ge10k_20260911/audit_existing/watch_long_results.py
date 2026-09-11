#!/usr/bin/env python3
"""Watch only the five authorized long jobs and run offline audits on completion.

This launches no simulator and writes only this audit directory. Restarting
validates audit/result/script hashes before using previously completed work.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import audit_sustained as audit

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'long_validation_7200s'

def main():
    p=argparse.ArgumentParser();p.add_argument('--max-wait-seconds',type=int,default=7200);p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
    assert 30<=a.poll_seconds<=60 and a.max_wait_seconds>0
    jobs=audit.read(STUDY/'plan.json')['jobs'];assert len(jobs)==5
    started=time.time();outpath=HERE/'long_runtime_audit_status.json';records={};laststatus={}
    if outpath.exists():records=audit.read(outpath).get('records',{})
    while time.time()-started<a.max_wait_seconds:
        for job in jobs:
            label=job['input']['label'];strategy=job['strategy'];key=label+'/'+strategy;command=STUDY/'runs'/label/strategy/'command.json'
            if not command.exists():status='pending';cmd=None
            else:cmd=audit.read(command);status=cmd['status']
            if status!=laststatus.get(key):print(json.dumps(dict(job=key,simulation_status=status),ensure_ascii=False),flush=True);laststatus[key]=status
            if status!='complete':
                records[key]=dict(status='simulation_'+status,command=str(command));continue
            result=Path(cmd['output']);manifest=Path(cmd['manifest']);assert audit.sha(result)==cmd['output_sha256'] and audit.sha(manifest)==job['input']['manifest_sha256']
            short='main_random' if '_6l_' in label and '_random_' in label else 'main_ordered' if '_6l_' in label else 'backup_ordered'
            target=HERE/f'long_{short}_{strategy}_validation.json';phase=HERE/f'long_{short}_{strategy}_phase.json';prefetch=HERE/f'long_{short}_{strategy}_prefetch_demand.json'
            prefetch_script=HERE.parent/'audit_prefetch_demand.py'
            fingerprint=dict(result_sha256=audit.sha(result),manifest_sha256=audit.sha(manifest),auditor_sha256=audit.sha(HERE/'audit_sustained.py'),phase_script_sha256=audit.sha(HERE/'analyze_mostly_phase.py'),prefetch_script_sha256=audit.sha(prefetch_script))
            old=records.get(key,{})
            if old.get('status') in ('audited','audit_error','phase_error','prefetch_error') and old.get('fingerprint')==fingerprint and target.exists() and old.get('audit_sha256')==audit.sha(target):continue
            commandline=[sys.executable,'-B',str(HERE/'audit_sustained.py'),'--manifest',str(manifest),'--result',str(result),'--out',str(target),'--end-ms','60000']
            proc=subprocess.run(commandline,capture_output=True,text=True)
            record=dict(status='audit_error' if proc.returncode else 'audited',fingerprint=fingerprint,command=str(command),audit_command=commandline,audit_returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr)
            if target.exists():
                d=audit.read(target);w=d['primary_window'];record.update(audit_path=str(target),audit_sha256=audit.sha(target),technical=d['technical_audit_passed'],conditions=d['primary_conditions_passed'],U=w['U_percent'],minimum_switches=w['minimum_per_card_role_switches'],minimum_role_share=w['minimum_per_card_role_compute_share'])
            if proc.returncode==0 and '_ordered_' in label:
                phasecmd=[sys.executable,'-B',str(HERE/'analyze_mostly_phase.py'),'--manifest',str(manifest),'--result',str(result),'--out',str(phase),'--end-ms','60000']
                q=subprocess.run(phasecmd,capture_output=True,text=True);record.update(phase_command=phasecmd,phase_returncode=q.returncode,phase_stdout=q.stdout,phase_stderr=q.stderr)
                if q.returncode:record['status']='phase_error'
                elif phase.exists():record.update(phase_path=str(phase),phase_sha256=audit.sha(phase))
            if proc.returncode==0:
                prefetchcmd=[sys.executable,'-B',str(prefetch_script),'--manifest',str(manifest),'--result',str(result),'--out',str(prefetch),'--end-ms','60000']
                q=subprocess.run(prefetchcmd,capture_output=True,text=True);record.update(prefetch_command=prefetchcmd,prefetch_returncode=q.returncode,prefetch_stdout=q.stdout,prefetch_stderr=q.stderr)
                if q.returncode:record['status']='prefetch_error'
                elif prefetch.exists():record.update(prefetch_path=str(prefetch),prefetch_sha256=audit.sha(prefetch))
            records[key]=record;print(json.dumps(dict(completed_offline_audit=key,**{k:record.get(k) for k in ('status','technical','conditions','U','minimum_switches','minimum_role_share')}),ensure_ascii=False),flush=True)
            outpath.write_text(json.dumps(dict(records=records,updated_unix=time.time(),elapsed_watch_seconds=time.time()-started),ensure_ascii=False,indent=2)+'\n')
        outpath.write_text(json.dumps(dict(records=records,updated_unix=time.time(),elapsed_watch_seconds=time.time()-started),ensure_ascii=False,indent=2)+'\n')
        complete=sum(r['status']=='audited' for r in records.values());failed=[k for k,r in records.items() if r['status'] in ('simulation_failed','simulation_timeout','simulation_cancelled','audit_error','phase_error','prefetch_error')]
        if complete+len(failed)==len(jobs):print(json.dumps(dict(watch_finished=True,audited=complete,failed=failed)),flush=True);return
        print(json.dumps(dict(watch_waiting=True,audited=complete,pending=len(jobs)-complete-len(failed),elapsed_seconds=round(time.time()-started))),flush=True)
        time.sleep(a.poll_seconds)
    print(json.dumps(dict(watch_timeout=True,status_file=str(outpath))),flush=True)

if __name__=='__main__':main()
