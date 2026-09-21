#!/usr/bin/env python3
"""Approximate rational A:B period and cohort offsets, below per-card 2.5 GB/s."""
import argparse,itertools,json,math
from approx_phase_search import simulate,HERE,profile

parser=argparse.ArgumentParser();parser.add_argument('--focus',choices=['low','high'],default='low');args=parser.parse_args()
rows=[]
templates=[]
families=[(4,5),(5,6),(4,4,5,5),(4,5,4,5),(4,5,5,5),(5,5,5,6),(5,6,6,6),(5,5,6,6),(4,6),(3,7),(5,), (4,), (6,)] if args.focus=='low' else [(7,8),(8,9),(7,9),(7,7,8,8),(8,8,9,9),(7,8,8,8),(8,8,8,9),(7,),(8,),(9,)]
for lengths in families:
    base=[]
    for length in lengths:base += [0]+[1]*length
    for cohort in range(6,11):
        for off in range(1,len(base)):
            pats=[base[off:]+base[:off] if n<cohort else base for n in range(16)]
            name=f'q{lengths}_cohort{cohort}_offset{off}'
            r=simulate(pats,m=3302,m_b=2448,ssu=1);r['name']=name;rows.append(r)
rows.sort(key=lambda r:r['U'])
# Tune five strongest period templates over nearby independently underloaded profiles.
for ma in [3289,3300,3302,3330,3370,3400,3500,3600]:
    for mb in [2438,2440,2448,2470,2500,2600,2700]:
        ca,va=profile(200,ma);cb,vb=profile(32,mb)
        if max(va/ca,vb/cb)>=2.5:continue
        for source in list(rows[:5]):
            pats=[[int(c=='B') for c in pat] for pat in source['patterns']]
            r=simulate(pats,m=ma,m_b=mb,ssu=1);r['name']=f'A{ma}_B{mb}_'+source['name'];rows.append(r)
rows.sort(key=lambda r:r['U'])
strict=[r for r in rows if r['overload_ms']==0 and r['every_card_AB_2_4']]
long=[]
for r in strict[:15]:
    z=simulate([[int(c=='B') for c in pat] for pat in r['patterns']],m=r['miss'],m_b=r['miss_B'],right=180000.)
    z['name']=r['name'];long.append(z)
long.sort(key=lambda r:r['U'])
(HERE/('rational_phase_approx.json' if args.focus=='low' else 'rational_high_phase_approx.json')).write_text(json.dumps(dict(approximate_only=True,total=len(rows),strict_mixed=len(strict),best12=strict[:30],best180=long),indent=2)+'\n')
print(json.dumps(long[:5],indent=2))
