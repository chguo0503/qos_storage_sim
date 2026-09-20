"""One offline planning process, then two independent formal API processes."""
from pathlib import Path
import json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from inputs.manifest import load_manifest,write_json
NAME='abb_interp_unique_50'

def main():
 status={'name':NAME,'stage':'planning','planning_cpu':0,'formal_cpus':{'od_baseline':2,'once':4}}
 write_json(HERE/'long_queue_status.json',status)
 plan=[sys.executable,str(HERE/'run_interpolated.py'),'--name',NAME,'--cycles','50','--mode','plan','--unique-prefix']
 with (HERE/'long_plan.log').open('w') as stream:
  p=subprocess.Popen(['taskset','-c','0',*plan],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  status['planning_pid']=p.pid;status['planning_command']=plan;write_json(HERE/'long_queue_status.json',status)
  ret=p.wait()
 if ret:
  status.update(stage='planning_failed',returncode=ret);write_json(HERE/'long_queue_status.json',status);raise SystemExit(ret)
 reqs,meta=load_manifest(HERE/'inputs'/f'{NAME}.json.gz')
 pure=[sum(8*q.load['per_layer_us']/1000 for q in reqs if q.npu_id==n) for n in range(32)]
 assert min(pure)>60000,'Every NPU must retain >=60 seconds of pure compute work'
 status.update(stage='formal',pure_compute_ms_by_npu=pure,formal={})
 jobs={};streams={}
 for policy,cpu in [('od_baseline',2),('once',4)]:
  command=['taskset','-c',str(cpu),sys.executable,str(HERE/'clean_replay.py'),'--name',NAME,'--strategy',policy]
  streams[policy]=(HERE/f'long_{policy}.log').open('w')
  p=subprocess.Popen(command,cwd=ROOT,stdout=streams[policy],stderr=subprocess.STDOUT);jobs[policy]=p
  status['formal'][policy]={'pid':p.pid,'command':command,'status':'running'}
 write_json(HERE/'long_queue_status.json',status)
 while any(p.poll() is None for p in jobs.values()):
  for policy,p in jobs.items():
   status['formal'][policy]['returncode']=p.poll()
   if p.poll() is not None:status['formal'][policy]['status']='complete' if p.returncode==0 else 'failed'
  write_json(HERE/'long_queue_status.json',status);time.sleep(5)
 for policy,p in jobs.items():
  streams[policy].close();status['formal'][policy].update(returncode=p.returncode,status='complete' if p.returncode==0 else 'failed')
 status['stage']='complete' if all(p.returncode==0 for p in jobs.values()) else 'formal_failed'
 write_json(HERE/'long_queue_status.json',status)
 if status['stage']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
