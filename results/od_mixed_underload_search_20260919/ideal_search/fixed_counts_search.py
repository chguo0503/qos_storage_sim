#!/usr/bin/env python3
"""Adversarial address hypothesis: fixed legal per-profile disk counts."""
import concurrent.futures
import json
import random
import time
from ring_search import HERE,run,valid
BLOCK=176*1024/2**30
CASES={
 'A80B1210':dict(block_counts={'A':[25,28,27],'B':[382,419,409]},C={'A':1.8407390890701447,'B':90.02253536560275},prefixes=[3,1,2,4],m_range=range(30,40)),
 'A180B1039':dict(block_counts={'A':[57,62,61],'B':[328,360,351]},C={'A':4.153639478141864,'B':74.98957484197959},prefixes=[2,4,1,3],m_range=range(10,17)),
}

def payload(case,m,repeats=6,jitter=0,seed=7):
 p=CASES[case];rng=random.Random(seed);queues=[['B']*p['prefixes'][i//8]+(['A']*m+['B']*3)*repeats for i in range(32)]
 lines=[]
 for q in queues:
  lines.append(str(len(q)))
  for r in q:
   counts=list(p['block_counts'][r])
   if jitter:
    for _ in range(jitter):
     a,b=rng.sample(range(3),2);counts[a]+=1;counts[b]-=1
   lines.append(' '.join(map(str,(int(r=='B'),p['C'][r],*[n*BLOCK for n in counts]))))
 return '\n'.join(lines),queues

def main():
 texts={};specs=[]
 for case,p in CASES.items():
  for m in p['m_range']:
   texts[case,m]=payload(case,m)[0]
   for a in range(-5,11):
    for b in range(-5,11):specs.append((case,m,1+a*.01,1+b*.01))
 def one(s):
  case,m,ca,cb=s
  return dict(case=case,m=m,scale_A=ca,scale_B=cb,**run(texts[case,m],(ca,cb)))
 results=[];best=1.;start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  for j,r in enumerate(pool.map(one,specs)):
   results.append(r);score=max(w['U']for w in r['windows'])
   if valid(r)and score<best:
    best=score
    print(json.dumps(dict(index=j,case=r['case'],m=r['m'],scale_A=r['scale_A'],scale_B=r['scale_B'],maxd=r['max_demand_by_ssu'],U=[w['U']for w in r['windows']])),flush=True)
   if (j+1)%500==0:print(json.dumps(dict(progress=j+1,best=best,elapsed=time.time()-start)),flush=True)
 passed=sorted((r for r in results if valid(r)),key=lambda r:max(w['U']for w in r['windows']))
 (HERE/'fixed_counts_results.json').write_text(json.dumps(results,indent=2))
 (HERE/'fixed_counts_valid.json').write_text(json.dumps(passed,indent=2))
 for i,r in enumerate(passed[:10]):
  p=CASES[r['case']];text,queues=payload(r['case'],r['m'])
  jitters=[]
  for jitter in (1,2):
   for seed in (7,19,43):
    text,_=payload(r['case'],r['m'],jitter=jitter,seed=seed)
    jitters.append(dict(jitter_moves=jitter,seed=seed,result=run(text,(r['scale_A'],r['scale_B']))))
  text,_=payload(r['case'],r['m'],repeats=64)
  extended=run(text,(r['scale_A'],r['scale_B']),horizon=64000)
  (HERE/f'fixed_counts_candidate_{i:02d}.json').write_text(json.dumps(dict(queues=queues,profiles={role:dict(block_count=sum(p['block_counts'][role]),counts_by_ssu=p['block_counts'][role],C_ms=p['C'][role]*r['scale_'+role])for role in ('A','B')},proxy_result=r,long_result=extended,jitter_results=jitters,
   caveat='Adversarial address selection hypothesis. Every request volume per disk is fixed here; actual IDs must remain valid Ring Hash.'),indent=2))
 print(json.dumps(dict(done=len(results),valid=len(passed),elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()
