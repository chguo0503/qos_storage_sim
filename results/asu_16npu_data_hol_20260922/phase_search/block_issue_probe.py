#!/usr/bin/env python3
"""Single disk FIFO selector with exact 128-token IO size and 0.1us issue spacing.

Still independent/approximate: excludes native QoS event state and client counters.
This repairs the atomic whole-layer assumption specifically for phase screening.
"""
import heapq,itertools,json
from approx_phase_search import profile,HERE

def simulate(patterns,ma=3302,mb=2448,left=2000.,right=12000.):
    ps=[profile(200,ma),profile(32,mb)]; prefixes=[200*1024-ma,32*1024-mb]
    blocks=[]
    for prefix in prefixes:
        full,tail=divmod(prefix,128);blocks.append([128*1408/1e6]*full+([tail*1408/1e6] if tail else []))
    counter=itertools.count();events=[];free=0.;busy=[0.]*16;windows=[0.]*5;seen=[set() for _ in range(16)]
    role_events=[]; stall=[0.,0.]
    # Kind 0 = issue one block, kind 1 = compute start. Stable serial tie order.
    def read(now,n,r,l,barrier):
        role=patterns[n][r%len(patterns[n])]
        heapq.heappush(events,(now,next(counter),0,n,r,l,barrier,0))
    for n,p in enumerate(patterns):
        role_events.append((0,int(p[0]==1)));read(0,n,0,0,0.)
    while events:
        now,_,kind,n,r,l,barrier,k=heapq.heappop(events)
        if now>=right:continue
        role=patterns[n][r%len(patterns[n])]
        if kind==0:
            v=blocks[role][k];free=max(free,now)+v/40.
            if k+1<len(blocks[role]):heapq.heappush(events,(now+.0001,next(counter),0,n,r,l,barrier,k+1))
            else:heapq.heappush(events,(max(barrier,free+v/50.),next(counter),1,n,r,l,barrier,0))
            continue
        c,_=ps[role];end=now+c
        busy[n]+=max(0.,min(end,right)-max(now,left))
        stall[role]+=max(0.,min(now,right)-max(barrier,left))
        for w in range(5):windows[w]+=max(0.,min(end,right,4000+w*2000)-max(now,2000+w*2000))
        if min(end,4000)>max(now,2000):seen[n].add(role)
        nr,nl=(r,l+1) if l<7 else (r+1,0)
        nxt=patterns[n][nr%len(patterns[n])]
        if l==7 and end<right:role_events.append((end,int(nxt==1)-int(role==1)))
        read(now,n,nr,nl,end)
    count=0;peak=0.;over=0.;previous=0.;demand=0.;max_b=0
    for now,group in itertools.groupby(sorted(role_events),key=lambda e:e[0]):
        if demand>40+1e-9:over+=now-previous
        count+=sum(delta for _,delta in group);demand=(16-count)*ps[0][1]/ps[0][0]+count*ps[1][1]/ps[1][0]
        peak=max(peak,demand);max_b=max(max_b,count);previous=now
    if demand>40+1e-9:over+=right-previous
    return dict(approximate_only=True,block_issue_model=True,ma=ma,mb=mb,U=sum(busy)/16/(right-left)*100,
                windows=[w/32000*100 for w in windows],all_warm_AB=all(len(s)==2 for s in seen),
                peak_demand_gb_s=peak,max_B_cards=max_b,overload_ms=over,stall_ms=stall,
                patterns=[''.join('AB'[v] for v in p) for p in patterns])

if __name__=='__main__':
    rows=[]
    for q,offset in [(5,1),(8,4)]:
        base=[0]+[1]*q;pats=[base[offset:]+base[:offset]]*8+[base]*8
        r=simulate(pats,right=10000.);r['name']=f'I1_q{q}_offset{offset}';rows.append(r);print(r,flush=True)
    (HERE/'block_issue_native_calibration.json').write_text(json.dumps(rows,indent=2)+'\n')
