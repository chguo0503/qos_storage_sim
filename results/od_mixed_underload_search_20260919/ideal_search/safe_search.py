#!/usr/bin/env python3
"""Find phase-stable candidates with room for fixed Ring Hash disk imbalance."""
import concurrent.futures
import json
import random
import time
from expand_search import HERE, make, run, valid
BLOCK=176*1024/2**30

def one(s):return dict(parameters=s,**run(make(s),s['profiles']))
def score(r):return max(w['U']for w in r['windows'])

def main():
 rng=random.Random(872);specs=[]
 for j in range(24000):
  blocks=rng.choice((80,96,120,160,180,248))
  ba=rng.uniform(1.9,2.9);va=blocks*BLOCK/3;ca=1000*va/ba
  target=rng.choice((35.5,36.,36.5,37.,37.5))
  bb=(target-8*ba)/24
  cb=rng.uniform(35,100)
  # Quantize B to full original command blocks, distributed in expectation.
  bblocks=max(1,round(cb*bb/1000*3/BLOCK));vb=bblocks*BLOCK/3
  m=max(1,round(rng.uniform(.55,.8)*cb/ca))
  prefix=[1,2,3,4];rng.shuffle(prefix)
  specs.append(dict(profiles={'A':dict(C_ms=ca,V_per_disk_GiB=va),'B':dict(C_ms=cb,V_per_disk_GiB=vb)},
                    sizes=[8]*4,m=[m]*4,k=[3]*4,prefixes=prefix,permute=0,A_total_blocks=blocks,B_total_blocks=bblocks,target_demand=target))
 results=[];best=1.;start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  for j,r in enumerate(pool.map(one,specs)):
   results.append(r)
   if valid(r)and score(r)<best:
    best=score(r)
    print(json.dumps(dict(index=j,best=best,parameters=r['parameters'],U=[w['U']for w in r['windows']])),flush=True)
   if (j+1)%2000==0:print(json.dumps(dict(progress=j+1,best=best,elapsed=time.time()-start)),flush=True)
 passed=sorted((r for r in results if valid(r)),key=score)
 (HERE/'safe_valid.json').write_text(json.dumps(passed,indent=2))
 for i,r in enumerate(passed[:20]):
  p=dict(r['parameters'],repeats=6)
  extended=run(make(dict(p,repeats=64)),p['profiles'],horizon=64000)
  (HERE/f'formal_candidate_{i:02d}.json').write_text(json.dumps(dict(profiles=p['profiles'],n_layers=8,queues=make(p),parameters=p,proxy_result=r,proxy_long_result=extended,
    caveat='Synthetic >=10K-compatible total block counts. Equal three-disk fluid proxy; actual Ring Hash and discrete SSD simulation required.'),indent=2))
 print(json.dumps(dict(done=len(results),valid=len(passed),elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()
