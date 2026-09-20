#!/usr/bin/env python3
"""Tune static phase recurrence with >=80 total 176-KiB blocks in A."""
import concurrent.futures
import json
import random
import time
from expand_search import HERE, make, run, valid
BLOCK=176*1024/2**30

def one(s):return dict(parameters=s,**run(make(s),s['profiles']))
def score(r):return max(w['U']for w in r['windows'])

def main():
 rng=random.Random(631);specs=[]
 for j in range(18000):
  sizes=rng.choice(((8,8,8,8),(12,12,8),(10,10,10,2)))
  blocks=rng.choice((80,96,120,160,180,248))
  ba=rng.uniform(1.7,2.8)
  va=blocks*BLOCK/3;ca=1000*va/ba
  target=rng.choice((38.0,38.8,39.2,39.6))
  bb=(target-max(sizes)*ba)/(32-max(sizes))
  if bb<.3:continue
  cb=rng.uniform(25,115);vb=cb*bb/1000
  k=len(sizes)-1
  m=max(1,round(rng.uniform(.36,.76)*cb/ca))
  mm=[m]*len(sizes)
  if rng.random()<.20:mm=[max(1,m+rng.randrange(-1,2))for _ in sizes]
  prefix=list(range(rng.choice((0,1)),len(sizes)))
  if len(prefix)!=len(sizes):prefix=list(range(1,len(sizes)+1))
  rng.shuffle(prefix)
  specs.append(dict(profiles={'A':dict(C_ms=ca,V_per_disk_GiB=va),'B':dict(C_ms=cb,V_per_disk_GiB=vb)},
                    sizes=sizes,m=mm,k=[k]*len(sizes),prefixes=prefix,permute=0,A_total_blocks=blocks,target_demand=target))
 results=[];best=1.;start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  for j,r in enumerate(pool.map(one,specs)):
   results.append(r)
   if valid(r)and score(r)<best:
    best=score(r)
    (HERE/'resonance_best.json').write_text(json.dumps(dict(profiles=r['parameters']['profiles'],n_layers=8,queues=make(r['parameters']),parameters=r['parameters'],proxy_result=r),indent=2))
    print(json.dumps(dict(index=j,best=best,parameters=r['parameters'],U=[w['U']for w in r['windows']])),flush=True)
   if (j+1)%1000==0:print(json.dumps(dict(progress=j+1,best=best,elapsed=time.time()-start)),flush=True)
 passed=sorted((r for r in results if valid(r)),key=score)
 (HERE/'resonance_valid.json').write_text(json.dumps(passed,indent=2))
 for i,r in enumerate(passed[:20]):
  (HERE/f'resonance_candidate_{i:02d}.json').write_text(json.dumps(dict(profiles=r['parameters']['profiles'],n_layers=8,queues=make(r['parameters']),parameters=r['parameters'],proxy_result=r),indent=2))
 print(json.dumps(dict(done=len(results),valid=len(passed),elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()
