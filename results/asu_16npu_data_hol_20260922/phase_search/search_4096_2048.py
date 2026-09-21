#!/usr/bin/env python3
"""Approximate search of raw A200/4096 and B32/2048, one 40 GB/s disk."""
import json,random
from approx_phase_search import simulate,HERE

rng=random.Random(93077);rows=[]
def add(name,pats):
    r=simulate(pats,m=4096,m_b=2048,ssu=1);r['name']=name;rows.append(r)
    return r
for ratio in [1,2,3,4,5,6,8]:
    base=[0]+[1]*ratio
    for count_b in range(8):
        for off in range(1,len(base)):
            add(f'two_cohort_ratio{ratio}_initial_B{count_b}_offset{off}',[base[off:]+base[:off] if n<count_b else base for n in range(16)])
for trial in range(10000):
    pats=[]
    ratio=rng.choice([1,2,3,4,5]);count=rng.choice([1,1,2,3])
    # Ensure start has at most seven B; future overlap remains audited.
    initial_B=set(rng.sample(range(16),rng.randrange(8)))
    for n in range(16):
        own_ratio=ratio if trial%3 else rng.choice([1,2,3,4,5])
        p=[0]*count+[1]*(count*own_ratio);rng.shuffle(p)
        first=int(n in initial_B);j=p.index(first);p[0],p[j]=p[j],p[0]
        pats.append(p)
    add(f'periodic_draw{trial}',pats)
    if trial%1000==0:print(trial,len([r for r in rows if r['overload_ms']==0]),flush=True)
rows.sort(key=lambda r:r['U'])
strict=[r for r in rows if r['overload_ms']==0 and r['every_card_AB_2_4']]
(HERE/'4096_2048_approx_all.json').write_text(json.dumps(dict(approximate_only=True,rows=rows),indent=2)+'\n')
(HERE/'4096_2048_approx_shortlist.json').write_text(json.dumps(dict(approximate_only=True,total=len(rows),strict_mixed=len(strict),best=strict[:10],unconstrained_best=rows[:3]),indent=2)+'\n')
print(json.dumps(strict[:4],indent=2))
