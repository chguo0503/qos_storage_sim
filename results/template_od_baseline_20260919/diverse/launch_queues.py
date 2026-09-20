from pathlib import Path
import argparse,json,subprocess,sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args()
logs=HERE/'execution_logs';logs.mkdir(exist_ok=True)
for q in json.loads((HERE/a.plan).read_text()):
 argv=[sys.executable,str(HERE/'queue.py'),'--plan',a.plan,'--queue',q['name']]
 with (logs/(q['name']+'_queue.log')).open('xb') as f:
  child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
 print(json.dumps(dict(queue=q['name'],pid=child.pid,cpu=q['cpu'],argv=argv)),flush=True)
