#!/usr/bin/env python3
"""APPROXIMATE selector only: whole-layer FIFO and balanced disks.

Uses only raw data rows, decimal 40 GB/s disks and 50 GB/s NPU link.
The native simulator remains the sole source of reportable outcomes.
"""
from __future__ import annotations
import ast, heapq, itertools, json, math, random
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ast.literal_eval((ROOT/'data').read_text())

def profile(k,m):
    if (k,m) in DATA:return DATA[k,m][1]/1000, DATA[k,m][3]*2**30/1e6
    measured=sorted(q for t,q in DATA if t==k)
    lower=max(q for q in measured if q<m);upper=min(q for q in measured if q>m)
    weight=(m-lower)/(upper-lower)
    compute=(1-weight)*DATA[k,lower][1]+weight*DATA[k,upper][1]
    return compute/1000,(k*1024-m)*1408/1e6

def simulate(patterns,m=4096,ssu=1,left=2000.,right=12000.,m_b=None):
    ps=[profile(200,m),profile(32,m if m_b is None else m_b)]
    cap=ssu*40.
    free=0.;ev=[];counter=itertools.count();busy=[0.]*16
    wins=[[0.]*16 for _ in range(5)]; coverage=[set() for _ in range(16)]
    first=0.;last=0.;stall=[0.,0.];role_events=[]
    def read(now,role):
        nonlocal free
        c,v=ps[role]
        begin=max(now,free);free=begin+v/cap
        return begin+max(v/cap,v/50.)+.00361
    for n,pat in enumerate(patterns):
        role_events.append((0.,int(pat[0]==1)))
        heapq.heappush(ev,(read(0,pat[0]),next(counter),n,0,0,0.))
    while ev:
        start,_,n,r,l,barrier=heapq.heappop(ev)
        if start>=right: continue
        role=patterns[n][r%len(patterns[n])];c,v=ps[role];end=start+c
        clipped=max(0.,min(end,right)-max(start,left));busy[n]+=clipped
        stall[role]+=max(0.,min(start,right)-max(barrier,left))
        if min(end,4000.)>max(start,2000.):coverage[n].add(role)
        for w in range(5):wins[w][n]+=max(0.,min(end,4000.+w*2000)-max(start,2000.+w*2000))
        nr,nl=(r,l+1) if l<7 else (r+1,0)
        nxt=patterns[n][nr%len(patterns[n])]
        if l==7 and end<right:role_events.append((end,int(nxt==1)-int(role==1)))
        ready=read(start,nxt)
        heapq.heappush(ev,(max(end,ready),next(counter),n,nr,nl,end))
    count=0;max_b=0;over_ms=0.;previous=0.;demand=0.;peak=0.
    role_events.sort()
    for now,group in itertools.groupby(role_events,key=lambda e:e[0]):
        if demand>cap+1e-9:over_ms+=now-previous
        count+=sum(delta for _,delta in group)
        demand=(16-count)*ps[0][1]/ps[0][0]+count*ps[1][1]/ps[1][0]
        peak=max(peak,demand);max_b=max(max_b,count);previous=now
    if demand>cap+1e-9:over_ms+=right-previous
    return dict(U=sum(busy)/16/(right-left)*100,windows=[sum(w)/16/2000*100 for w in wins],
                every_card_AB_2_4=all(len(c)==2 for c in coverage),stall_ms=stall,
                patterns=[''.join('AB'[v] for v in p) for p in patterns],
                approximate_only=True,miss=m,miss_B=m if m_b is None else m_b,ssu=ssu,
                max_B_cards=max_b,peak_nominal_gb_s=peak,overload_ms=over_ms)

def main():
    rng=random.Random(71829);rows=[]
    base=[0,1,1,1]
    def add(name,pats,m=4096,ssu=1):
        r=simulate(pats,m,ssu);r['name']=name;rows.append(r);return r
    for m,s in [(4096,1),(2048,2),(1024,4)]:
        for ratio in [1,2,3,4,5,6,8]:
            base=[0]+[1]*ratio
            for step in range(len(base)):
                add(f'period_r1{ratio}_offset_step{step}',[base[(n*step)%len(base):]+base[:(n*step)%len(base)] for n in range(16)],m,s)
            for cohort in [4,6,8,10,12]:
                for offset in range(1,len(base)):
                    add(f'period_r1{ratio}_cohort{cohort}_offset{offset}',[base if n<cohort else base[offset:]+base[:offset] for n in range(16)],m,s)
        for chunk in [2,3,4]:
            for ratio in [2,3,4]:
                base=[0]*chunk+[1]*(chunk*ratio)
                for trial in range(15):
                    offsets=[rng.randrange(len(base)) for n in range(16)]
                    add(f'chunk{chunk}_ratio{ratio}_trial{trial}',[base[o:]+base[:o] for o in offsets],m,s)
    rows.sort(key=lambda r:r['U'])
    # Explore periodic independent per-card decks with identical A:B counts.
    for m,s in [(4096,1),(2048,2),(1024,4)]:
        for trial in range(1000):
            ratio=rng.choice([2,3,4]);count=rng.choice([1,2,3])
            pats=[]
            for n in range(16):
                p=[0]*count+[1]*(count*ratio);rng.shuffle(p);pats.append(p)
            add(f'independent_periodic_trial{trial}',pats,m,s)
    rows.sort(key=lambda r:r['U'])
    (HERE/'approx_results.json').write_text(json.dumps(dict(approximate_only=True,assumptions=['whole layer atomically queued','perfectly balanced SSUs','infinite periodic decks','no native routing or block simulation'],rows=rows),indent=2)+'\n')
    best=[]
    for m in [4096,2048,1024]:
        candidates=[r for r in rows if r['miss']==m and r['every_card_AB_2_4']]
        best.extend(candidates[:4])
    (HERE/'approx_shortlist.json').write_text(json.dumps(best,indent=2)+'\n')
    print(json.dumps(best,indent=2))

if __name__=='__main__':main()
