#!/usr/bin/env python3
"""Near-capacity mixed periodic phase selector; grid interpolation, not raw rows."""
import json,random
from approx_phase_search import simulate,HERE

rng=random.Random(730132);rows=[]
def add(name,pats):
    r=simulate(pats,m=3302,m_b=2448,ssu=1);r['name']=name;rows.append(r)
    return r
for ratio in [1,2,3,4,5,6,8,10,12,16]:
    base=[0]+[1]*ratio
    for cohort in [0,2,4,6,8,10,12,14,16]:
        for off in range(1,len(base)):
            add(f'two_cohort_ratio{ratio}_initial_B{cohort}_offset{off}',[base[off:]+base[:off] if n<cohort else base for n in range(16)])
for trial in range(5000):
    pats=[];ratio=rng.choice([1,2,3,4,5,6,8,10,12]);count=rng.choice([1,1,2,3])
    for n in range(16):
        own_ratio=ratio if trial%3 else rng.choice([1,2,3,4,5,6,8,10,12])
        p=[0]*count+[1]*(count*own_ratio);rng.shuffle(p);pats.append(p)
    add(f'interpolated_periodic_draw{trial}',pats)
    if trial%1000==0:print(trial,flush=True)
rows.sort(key=lambda r:r['U'])
strict=[r for r in rows if r['overload_ms']==0 and r['every_card_AB_2_4']]
(HERE/'interpolated_approx_all.json').write_text(json.dumps(dict(approximate_only=True,grid_interpolation=True,rows=rows),indent=2)+'\n')
(HERE/'interpolated_approx_shortlist.json').write_text(json.dumps(dict(approximate_only=True,total=len(rows),strict_mixed=len(strict),best=strict[:10],unconstrained_best=rows[:3]),indent=2)+'\n')
print(json.dumps(strict[:4],indent=2))
