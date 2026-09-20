"""Detach four bounded runs after the approved runtime is copied to this host."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('--seed',type=int);ap.add_argument('--depth-per-npu');ap.add_argument('--cpu',type=int);args=ap.parse_args()
logs=HERE/'execution_logs';logs.mkdir(exist_ok=True)
if args.worker:
 os.sched_setaffinity(0,{args.cpu})
 variant='od_unlimited' if args.depth_per_npu=='none' else 'od_depth256'
 label=f'full_{variant}_seed{args.seed}'
 argv=[sys.executable,str(HERE/'run_trial.py'),'--seed',str(args.seed),'--depth-per-npu',args.depth_per_npu,'--label',label]
 record=dict(label=label,cpu=args.cpu,worker_pid=os.getpid(),argv=argv,status='running',started=time.time())
 path=logs/(label+'_status.json');path.write_text(json.dumps(record,indent=2)+'\n')
 with (logs/(label+'.log')).open('xb') as output:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,close_fds=True)
  record['pid']=child.pid;path.write_text(json.dumps(record,indent=2)+'\n');code=child.wait()
 record.update(status='complete' if code==0 else 'failed',returncode=code,ended=time.time())
 path.write_text(json.dumps(record,indent=2)+'\n');raise SystemExit(code)
else:
 plan=[dict(seed=7,depth='none',cpu=0),dict(seed=7,depth='256',cpu=2),dict(seed=19,depth='256',cpu=4),dict(seed=43,depth='256',cpu=6)]
 (HERE/'execution_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 for job in plan:
  argv=[sys.executable,str(Path(__file__).resolve()),'--worker','--seed',str(job['seed']),'--depth-per-npu',job['depth'],'--cpu',str(job['cpu'])]
  with (logs/(f"worker_depth{job['depth']}_seed{job['seed']}.log")).open('xb') as output:
   child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
  print(json.dumps(dict(**job,worker_pid=child.pid)),flush=True)
