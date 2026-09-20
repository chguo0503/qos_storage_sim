#!/usr/bin/env python3
import argparse, concurrent.futures, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('batch'); ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args(); jobs=json.loads(Path(args.batch).read_text()); (ROOT/'logs').mkdir(exist_ok=True)
    def run(job):
        config,strategy=job; name=Path(config).stem
        if (ROOT/'results'/name/strategy/'analysis.json').exists(): return dict(name=name,strategy=strategy,cached=True)
        start=time.time()
        with (ROOT/'logs'/f'{name}_{strategy}.log').open('w') as out:
            p=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),str(ROOT/config),'--strategy',strategy],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        return dict(name=name,strategy=strategy,exit_code=p.returncode,wall_seconds=round(time.time()-start,1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures): print(json.dumps(f.result()),flush=True)
if __name__=='__main__': main()
