#!/usr/bin/env python3
"""Phase search using native-calibrated block issuance model, still not native."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from block_issue_probe import simulate,HERE

def work(task):
    name,pats=task;r=simulate(pats,right=12000.);r['name']=name;return r

def main():
    jobs={}
    def add(name,pats):jobs[tuple(tuple(p) for p in pats)]=(name,pats)
    for q in [6,7,8,9,10]:
        base=[0]+[1]*q
        for cohort in [6,7,8,9,10]:
            for off in sorted(set([1,max(1,q//2-1),q//2,q//2+1,q])):
                add(f'q{q}_cohort{cohort}_offset{off}',[base[off:]+base[:off] if n<cohort else base for n in range(16)])
    for lengths in [(7,8),(8,9),(7,9),(6,8),(8,10)]:
        base=[]
        for length in lengths:base += [0]+[1]*length
        for cohort in [7,8,9]:
            for off in sorted(set([1,3,4,5,len(base)//2-1,len(base)//2,len(base)//2+1,len(base)-1])):
                add(f'q{lengths}_cohort{cohort}_offset{off}',[base[off:]+base[:off] if n<cohort else base for n in range(16)])
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,t) for t in jobs.values()]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%20==0:
                rows.sort(key=lambda r:r['U']);(HERE/'block_phase_partial.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),best=rows[:8]),indent=2)+'\n')
                print(len(rows),len(jobs),rows[0]['name'],rows[0]['U'],flush=True)
    rows.sort(key=lambda r:r['U'])
    (HERE/'block_phase_results.json').write_text(json.dumps(dict(approximate_only=True,block_issue_model=True,total=len(rows),rows=rows),indent=2)+'\n')
    long=[]
    for r in [r for r in rows if r['all_warm_AB'] and r['overload_ms']==0][:4]:
        z=simulate([[int(c=='B') for c in pat] for pat in r['patterns']],right=60000.);z['name']=r['name'];long.append(z);print('long',z['name'],z['U'],flush=True)
    (HERE/'block_phase_shortlist_60s.json').write_text(json.dumps(long,indent=2)+'\n')

if __name__=='__main__':main()
