#!/usr/bin/env python3
"""Approximate two-cohort screen with measured-grid interpolation and demand audit."""
import json
from approx_phase_search import simulate,HERE,profile

rows=[]
for ma in [3000,3100,3200,3289,3302,3400,3500,3600,3800,4096]:
    for mb in [2048,2120,2160,2200,2240,2280,2320,2360,2400,2438,2448,2500,2600,2800,3000]:
        ca,va=profile(200,ma);cb,vb=profile(32,mb)
        # Preserve a potentially feasible 8A/8B coexistence region.
        if 8*(va/ca+vb/cb)>40:continue
        for ratio in [3,4,5,6,7,8,9,10]:
            base=[0]+[1]*ratio
            for cohort in [6,7,8,9,10]:
                for off in [1,max(1,ratio//2),ratio]:
                    pats=[base[off:]+base[:off] if n<cohort else base for n in range(16)]
                    r=simulate(pats,m=ma,m_b=mb,ssu=1);r['name']=f'A{ma}_B{mb}_q{ratio}_cohort{cohort}_offset{off}'
                    r['A_gb_s']=va/ca;r['B_gb_s']=vb/cb;rows.append(r)
    print(ma,len(rows),flush=True)
rows.sort(key=lambda r:r['U'])
strict=[r for r in rows if r['overload_ms']==0 and r['every_card_AB_2_4']]
long=[]
for r in strict[:40]:
    z=simulate([[int(c=='B') for c in pat] for pat in r['patterns']],m=r['miss'],m_b=r['miss_B'],right=60000.)
    z['name']=r['name'];z['A_gb_s']=r['A_gb_s'];z['B_gb_s']=r['B_gb_s'];long.append(z)
long.sort(key=lambda r:r['U'])
(HERE/'constrained_pairs_approx.json').write_text(json.dumps(dict(approximate_only=True,total=len(rows),strict12=len(strict),best12=strict[:20],best60=long,unconstrained=rows[:4]),indent=2)+'\n')
print(json.dumps(long[:8],indent=2))
