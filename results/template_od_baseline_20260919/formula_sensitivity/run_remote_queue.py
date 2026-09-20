from pathlib import Path
import concurrent.futures, json, queue, subprocess, sys
ROOT=Path(__file__).resolve().parent
cpus=queue.Queue()
for cpu in [0,2,4,6]:cpus.put(cpu)
def run(job):
    cpu=cpus.get()
    try:
        dest=ROOT/'logs';dest.mkdir(exist_ok=True)
        with (dest/f"{job['case']}.log").open('w') as log:
            result=subprocess.run(['taskset','-c',str(cpu),sys.executable,str(ROOT/'run_case.py'),
                                   '--case',job['case'],'--policy','od_baseline'],stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps({'case':job['case'],'returncode':result.returncode,'cpu':cpu}),flush=True)
        if result.returncode:raise RuntimeError(job['case'])
    finally:cpus.put(cpu)
if __name__=='__main__':
    jobs=json.loads((ROOT/'jobs.json').read_text())
    # Start sensitivity first so its timeline can be reviewed while others finish.
    jobs=sorted(jobs,key=lambda j:j['family']!='sensitivity')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(run,jobs):pass
