#!/usr/bin/env python3
"""Single-SSU FIFO selector with native submission RNG and per-NPU link FIFO.

Independent model, not the authoritative simulator. Calibrate full layer traces.
"""
import heapq,itertools,math
import numpy as np
from align_probe import HERE

def simulate(catalog,seqs,right=60000.,left=2000.,seed=7):
    pre={}
    for key,p in catalog.items():
        full,tail=divmod(p['ssd_prefix_tokens'],128)
        pre[key]=(p['compute_us']/1000,[128*1408/2**30]*full+([tail*1408/2**30] if tail else []),p['family'])
    disk_bw=40*1e9/2**30;link_bw=50*1e9/2**30;EPS=1e-12
    rng=np.random.RandomState(int(seed));serial=itertools.count();events=[]
    free=0.;link_free=[0.]*16;next_issue=[0.]*16;states={};busy=0.
    starts={};ends={};stalls={};phases={};seen=[set() for _ in range(16)]
    windows=[0.]*math.ceil(max(0.,right-2000)/2000);demand_events=[];rounds=0
    def rate(key):
        p=catalog[key];return p['read_gib']*2**30/p['compute_us']/1000
    def read(now,n,r,l,barrier):
        if r<len(seqs[n]):
            if n in states:raise AssertionError('Unexpected overlapping S1 submissions')
            states[n]=[r,l,barrier,0,now]
    for n,s in enumerate(seqs):demand_events.append((0.,rate(s[0])));read(0.,n,0,0,0.)
    while events or states:
        issue=min((max(st[4],next_issue[n]) for n,st in states.items()),default=math.inf)
        compute=events[0][0] if events else math.inf
        if min(issue,compute)>=right:break
        if compute<=issue:
            now,n,_,r,l,barrier=heapq.heappop(events);key=seqs[n][r];c,blocks,family=pre[key]
            if l==0:starts[n,r]=now
            end=now+c;busy+=max(0.,min(end,right)-max(now,left));stalls[n,r]=stalls.get((n,r),0)+now-barrier;phases[n,r,l]=now
            for w in range(len(windows)):windows[w]+=max(0.,min(end,right,4000+w*2000)-max(now,2000+w*2000))
            if min(end,4000)>max(now,2000):seen[n].add(family)
            nr,nl=(r,l+1) if l<7 else (r+1,0)
            if l==7:
                ends[n,r]=end
                if end<right:demand_events.append((end,(rate(seqs[n][nr]) if nr<len(seqs[n]) else 0.)-rate(key)))
            read(now,n,nr,nl,end)
            continue
        now=issue;ready=sorted(n for n,st in states.items() if st[4]<=now+EPS and next_issue[n]<=now+EPS)
        rng.shuffle(ready);rounds+=1
        for n in ready:
            r,l,barrier,k,release=states[n];key=seqs[n][r];v=pre[key][1][k]
            free=max(free,now)+v/disk_bw*1000.
            link_free[n]=max(link_free[n],free)+v/link_bw*1000.
            next_issue[n]=now+.1/1000.
            if k+1<len(pre[key][1]):states[n][3]+=1
            else:
                del states[n];heapq.heappush(events,(max(barrier,link_free[n]),n,next(serial),r,l,barrier))
    d=0.;peak=0.;over=0.;prev=0.
    for now,g in itertools.groupby(sorted(demand_events),key=lambda e:e[0]):
        if d>40+1e-8:over+=now-prev
        d+=sum(x for _,x in g);peak=max(peak,d);prev=now
    if d>40+1e-8:over+=right-prev
    metrics=dict(approximate_only=True,block_issue_model=True,native_submission_rng=True,exact_npu_link_fifo=True,
                 submit_seed=seed,submit_rounds=rounds,U=busy/16/(right-left)*100,
                 windows=[w/32000*100 for w in windows],all_warm_AB=all(len(s)==2 for s in seen),
                 warm_families_by_npu=[sorted(s) for s in seen],peak_nominal_gb_s=peak,overload_ms=over,right_ms=right)
    return metrics,dict(starts=starts,ends=ends,stalls=stalls,phases=phases)
