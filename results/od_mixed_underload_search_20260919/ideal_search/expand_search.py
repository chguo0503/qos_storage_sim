#!/usr/bin/env python3
"""Explore static group offsets and profile scales, keeping exact proxy audits."""
import concurrent.futures
import json
import random
import time
from pathlib import Path
from search import HERE, run


def make(spec):
    groups=sum(([g]*s for g,s in enumerate(spec['sizes'])),[])
    if spec.get('permute'):
        random.Random(spec['permute']).shuffle(groups)
    return [['B']*spec['prefixes'][g]+(['A']*spec['m'][g]+['B']*spec['k'][g])*spec.get('repeats',24) for g in groups]


def one(spec):
    result=run(make(spec),spec['profiles'])
    return dict(parameters=spec,**result)


def valid(r):
    return r['over_ms_all']<1e-7 and r['min_active_all']==32 and all(w['mixed_cards']==32 for w in r['windows'])


def main():
    rng=random.Random(19);specs=[]
    for j in range(6000):
        cb=rng.choice((25,40,50,60,80,100));ca=rng.choice((.5,1,1.5,2))
        sizes=rng.choice(((12,12,8),(11,11,10),(8,8,8,8),(10,10,10,2)))
        ba=rng.choice((1.75,2.,2.2,2.4,2.6))
        # Exactly 39.2 at largest cohort; profiles may have smaller actual peaks.
        bb=(39.2-max(sizes)*ba)/(32-max(sizes))
        if bb<=0:continue
        k=rng.choice((len(sizes)-1,len(sizes),len(sizes)+1))
        frac=rng.uniform(.28,.82)
        m=max(1,round(frac*cb/ca))
        mm=[max(1,m+rng.randint(-4,4)) for _ in sizes] if rng.random()<.35 else [m]*len(sizes)
        kk=[k]*len(sizes)
        prefixes=list(range(len(sizes)));rng.shuffle(prefixes)
        specs.append(dict(profiles={'A':dict(C_ms=ca,V_per_disk_GiB=ba*ca/1000),'B':dict(C_ms=cb,V_per_disk_GiB=bb*cb/1000)},
                          sizes=sizes,m=mm,k=kk,prefixes=prefixes,permute=rng.randrange(10000) if rng.random()<.35 else 0))
    results=[];best=1.;start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for j,r in enumerate(pool.map(one,specs)):
            results.append(r)
            if valid(r):
                score=sum(w['U'] for w in r['windows'])/3
                if score<best:
                    best=score
                    payload=dict(profiles=r['parameters']['profiles'],n_layers=8,queues=make(r['parameters']),parameters=r['parameters'],proxy_result=r,
                                 caveat='Balanced reference SSD fluid proxy only; actual Ring Hash simulation required.')
                    (HERE/'best_expanded.json').write_text(json.dumps(payload,indent=2))
                    print(json.dumps(dict(index=j,best=score,parameters=r['parameters'],U=[w['U'] for w in r['windows']])),flush=True)
            if (j+1)%500==0:print(json.dumps(dict(progress=j+1,elapsed=time.time()-start,best=best)),flush=True)
    passed=sorted((r for r in results if valid(r)),key=lambda r:sum(w['U'] for w in r['windows'])/3)
    (HERE/'expanded_results.json').write_text(json.dumps(results,indent=2))
    (HERE/'expanded_valid.json').write_text(json.dumps(passed,indent=2))
    for i,r in enumerate(passed[:12]):
        (HERE/f'expanded_candidate_{i:02}.json').write_text(json.dumps(dict(profiles=r['parameters']['profiles'],n_layers=8,queues=make(r['parameters']),parameters=r['parameters'],proxy_result=r),indent=2))
    print(json.dumps(dict(done=len(results),valid=len(passed),elapsed=time.time()-start)),flush=True)


if __name__=='__main__':main()
