from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/fixed128_32_underload_20260914'
def run(spec):
    folder=OUT/spec['stage']
    found=list(folder.glob('*/metrics.json'))
    if found:return json.loads(found[0].read_text())
    cmd=[sys.executable,'-u','run_fixed128_32.py','--policy','fifo']
    for k in ['ssu','long_nql','short_nql','short_per_long','stage']:
        cmd.extend(['--'+k.replace('_','-'),str(spec[k])])
    (OUT/(spec['stage']+'_command.json')).write_text(json.dumps(cmd))
    started=time.perf_counter()
    with (OUT/(spec['stage']+'.log')).open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"Failed {spec['stage']}; see log")
    row=json.loads(next(folder.glob('*/metrics.json')).read_text())
    return {k:row[k] for k in ['name','U_percent','short_U_percent','long_U_percent','slo_1p5_percent','all_npus_both_roles_computed','all_active','wall_seconds']}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='screen_plan.json');parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    specs=json.loads((OUT/args.plan).read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(run,s):s['index'] for s in specs}
        for task in as_completed(tasks):print(json.dumps({'index':tasks[task],**task.result()}),flush=True)
