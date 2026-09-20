#!/usr/bin/env python3
"""Search supplied unmodified data profiles with balanced per-disk placement."""
import concurrent.futures
import json
import random
import time
from expand_search import HERE, make, run, valid

PROFILES={
 'A32_B128':{'A':dict(C_ms=7.257231579,V_per_disk_GiB=.0416259765625/3),
             'B':dict(C_ms=93.017325202,V_per_disk_GiB=.16650390625/3)},
 'A32_B200':{'A':dict(C_ms=7.257231579,V_per_disk_GiB=.0416259765625/3),
             'B':dict(C_ms=141.335788279,V_per_disk_GiB=.26318359375/3)},
 'A32_B32':{'A':dict(C_ms=7.257231579,V_per_disk_GiB=.0416259765625/3),
             'B':dict(C_ms=28.592842105,V_per_disk_GiB=.03759765625/3)},
}

def one(s):return dict(parameters=s,**run(make(s),s['profiles']))

def main():
 rng=random.Random(74);specs=[]
 for name,p in PROFILES.items():
  for sizes in ((12,12,8),(11,11,10),(8,8,8,8),(4,4,4,4,4,4,4,4)):
   for k in (1,2,3):
    for m in range(1,17):
     for variant in range(5):
      prefix=list(range(len(sizes)))
      if variant:rng.shuffle(prefix)
      if variant==4:prefix=[x%3 for x in prefix]
      mm=[m]*len(sizes)
      if variant in (2,3):mm=[max(1,m+rng.randint(-2,2)) for _ in sizes]
      specs.append(dict(name=name,profiles=p,sizes=sizes,m=mm,k=[k]*len(sizes),prefixes=prefix,
                        permute=rng.randrange(10000) if variant==3 else 0))
 results=[];best={name:1 for name in PROFILES};start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for j,r in enumerate(pool.map(one,specs)):
   results.append(r)
   if valid(r):
    name=r['parameters']['name'];score=sum(w['U'] for w in r['windows'])/3
    if score<best[name]:
     best[name]=score
     print(json.dumps(dict(index=j,name=name,score=score,parameters=r['parameters'],U=[w['U']for w in r['windows']])),flush=True)
   if (j+1)%500==0:print(json.dumps(dict(progress=j+1,best=best,elapsed=time.time()-start)),flush=True)
 (HERE/'data_results.json').write_text(json.dumps(results,indent=2))
 for name in PROFILES:
  passed=sorted((r for r in results if r['parameters']['name']==name and valid(r)),key=lambda r:sum(w['U']for w in r['windows'])/3)
  (HERE/f'{name}_valid.json').write_text(json.dumps(passed,indent=2))
  for i,r in enumerate(passed[:5]):
   (HERE/f'{name}_candidate_{i:02d}.json').write_text(json.dumps(dict(profiles=r['parameters']['profiles'],n_layers=8,queues=make(r['parameters']),parameters=r['parameters'],proxy_result=r,
      caveat='Profile values from parent-supplied data; uniform reference placement is a proxy only.'),indent=2))
 print(json.dumps(dict(done=len(results),best=best,elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()
