"""Bounded one-process-per-core queue; each case runs through complete drain."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--queue',required=True);a=ap.parse_args()
 plans=json.loads((HERE/a.plan).read_text());q=next(q for q in plans if q['name']==a.queue)
 os.sched_setaffinity(0,{q['cpu']});logs=HERE/'execution_logs';logs.mkdir(exist_ok=True)
 status=dict(queue=q['name'],cpu=q['cpu'],pid=os.getpid(),state='running',jobs=[])
 target=logs/(q['name']+'_status.json')
 def save():
  tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(target)
 save()
 for j in q['jobs']:
  label=f"{j['scenario']}_{j['policy']}_seed{j['seed']}"
  argv=[sys.executable,str(HERE/'run_trial.py'),'--scenario',j['scenario'],'--seed',str(j['seed']),'--policy',j['policy'],'--label',label]
  record=dict(**j,label=label,argv=argv,started=time.time(),state='running');status['jobs'].append(record);save()
  with (logs/(label+'.log')).open('xb') as f:
   process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True)
   record['pid']=process.pid;save();rc=process.wait()
  record.update(ended=time.time(),returncode=rc,state='complete' if rc==0 else 'failed');save()
  if rc:
   status['state']='failed';save();raise SystemExit(rc)
 status['state']='complete';save()
if __name__=='__main__':main()
