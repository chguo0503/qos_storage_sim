from pathlib import Path
import argparse,json,subprocess,sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--cases',nargs='+');p.add_argument('--strategies',nargs='+',default=['od_baseline']);a=p.parse_args()
cases=a.cases or [r['label'] for r in json.loads((HERE/'candidate_plan.json').read_text())]
(HERE/'logs').mkdir(exist_ok=True);jobs=[(case,s) for case in cases for s in a.strategies]
assert len(jobs)<=6
for cpu,(case,s) in zip((8,10,12,14,16,18),jobs):
 label=case+'_'+s+('_full' if a.full else '_pilot')
 argv=['taskset','-c',str(cpu),sys.executable,str(HERE/'run_trial.py'),'--case',case,'--strategy',s]+([] if a.full else ['--pilot'])
 with (HERE/'logs'/f'{label}.log').open('xb') as f:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
 launch=dict(label=label,argv=argv,pid=child.pid,cpu=cpu)
 (HERE/'logs'/f'{label}_launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
