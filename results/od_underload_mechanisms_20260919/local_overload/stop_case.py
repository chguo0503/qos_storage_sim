"""Preserve an explicitly cancelled negative long-run without claiming full SLO."""
from pathlib import Path
import argparse,json,os,signal,time
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);ap.add_argument('--reason',required=True);a=ap.parse_args()
p=HERE/'runs'/a.label/'command.json';d=json.loads(p.read_text());assert d['status']=='running'
launch=json.loads((HERE/'logs'/(a.label+'_launch.json')).read_text());pid=launch['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes();assert b'run_long.py' in cmd and d['case'].encode() in cmd
os.kill(pid,signal.SIGINT)
for _ in range(50):
 time.sleep(.1)
 if not Path(f'/proc/{pid}').exists():break
else:raise RuntimeError('Process did not finish after SIGINT')
d=json.loads(p.read_text());d.update(status='cancelled_after_recovery',completed_simulation=False,cancellation_reason=a.reason,cancelled_pid=pid,no_full_population_SLO=True)
p.write_text(json.dumps(d,indent=2)+'\n');print(d['status'],a.label)
