#!/usr/bin/env python3
"""Static-queue search in a balanced, fluid OD proxy (not formal simulator)."""
import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = {"A": {"C_ms": 1.0, "V_per_disk_GiB": .002},
            "B": {"C_ms": 100.0, "V_per_disk_GiB": .076}}


def run(queues, profiles=PROFILES, horizon=8000.0, link=True):
    inp = "{} {} {} {} {} {}\n".format(
        profiles['A']['C_ms'], profiles['A']['V_per_disk_GiB'],
        profiles['B']['C_ms'], profiles['B']['V_per_disk_GiB'], horizon, int(link))
    inp += "\n".join(str(len(q)) + " " + " ".join(str(int(r == 'B')) for r in q) for q in queues)
    p = subprocess.run([str(HERE/'fluid_proxy')], input=inp, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)


def grouped(m, k, sizes=(12,12,8), prefixes=(0,1,2), repeats=16, interleave=False):
    groups = sum(([g]*size for g,size in enumerate(sizes)), [])
    if interleave:
        groups = sorted(range(32), key=lambda i: (i % len(sizes), i))
        mapping = [0]*32
        for g,ids in enumerate((groups[:sizes[0]],groups[sizes[0]:sizes[0]+sizes[1]],groups[sizes[0]+sizes[1]:])):
            for i in ids: mapping[i]=g
        groups = mapping
    return [['B']*prefixes[g]+(['A']*m+['B']*k)*repeats for g in groups]


def one(spec):
    q = grouped(**spec)
    result = run(q)
    return dict(parameters=spec, **result)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    specs=[]
    for sizes in ((12,12,8),(11,11,10),(8,12,12),(10,12,10)):
        for k in (2,3,4):
            for m in (15,20,25,30,35,40,45,50,55,60):
                for prefixes in ((0,1,2),(1,2,0),(2,0,1)):
                    specs.append(dict(m=m,k=k,sizes=sizes,prefixes=prefixes))
    start=time.time(); results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for j,r in enumerate(pool.map(one,specs)):
            results.append(r)
            w=r['windows']; valid=all(x['mixed_cards']==32 and x['under_fraction']>1-1e-10 and x['min_active']==32 for x in w)
            if valid and r['max_A_all']<=12:
                print(json.dumps({"valid":True,"parameters":r['parameters'],"U":[x['U'] for x in w],"max_A_all":r['max_A_all']}),flush=True)
            if (j+1)%60==0: print(json.dumps({"progress":j+1,"elapsed_s":time.time()-start}),flush=True)
    (HERE/'search_results.json').write_text(json.dumps(results,indent=2))
    valid=[r for r in results if r['max_A_all']<=12 and all(x['mixed_cards']==32 and x['min_active']==32 and x['under_fraction']>1-1e-10 for x in r['windows'])]
    valid.sort(key=lambda r: sum(w['U'] for w in r['windows'])/3)
    (HERE/'valid_results.json').write_text(json.dumps(valid,indent=2))
    for index,r in enumerate(valid[:8]):
        payload=dict(profiles=PROFILES,n_layers=8,queues=grouped(**r['parameters']),parameters=r['parameters'],proxy_result=r,
                     caveat='Balanced per-SSD continuous service proxy; not the Ring Hash discrete-command simulator.')
        (HERE/f'candidate_{index:02d}.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(done=len(results),valid=len(valid),elapsed_s=time.time()-start)),flush=True)


if __name__=='__main__': main()
