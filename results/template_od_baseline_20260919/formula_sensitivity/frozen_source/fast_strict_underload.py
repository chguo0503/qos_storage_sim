"""Bounded approximate FIFO search with event-exact demand checks.

This is a whole-layer queue surrogate, never native simulation evidence.
All demand changes are swept at their exact event times (no bin smoothing).
Nominal demand remains active during stalls. Prefetch reference demand is
V_next/C_current, throughout the current compute window, including next L0.
"""
from __future__ import annotations
import heapq,itertools,json,random,time
from pathlib import Path
from fast_multitype_probe import profile,native_matched_decks

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/strict_random_underload_20260914'

def audit(intervals,ssu,left,right):
    events=[]
    for a,b,vec in intervals:
        a=max(left,a);b=min(right,b)
        if b<=a:continue
        events.append((a,vec));events.append((b,[-v for v in vec]))
    events.sort(key=lambda x:x[0]);rates=[0.]*ssu;peak=[0.]*ssu
    over=[0.]*ssu;near=[0.]*ssu;anynear=0.;anyover=0.;last=left
    for now,delta in events:
        dt=now-last
        if dt>1e-9:
            peak=[max(a,b) for a,b in zip(peak,rates)]
            for s,b in enumerate(rates):
                over[s]+=dt*(b>40+1e-8);near[s]+=dt*(b>=36-1e-8)
            anynear+=dt*(max(rates)>=36-1e-8)
            anyover+=dt*(max(rates)>40+1e-8)
        rates=[a+b for a,b in zip(rates,delta)];last=now
    return dict(window_ms=[left,right],peak_gib_s=peak,over40_ms=over,
                near36_percent=[100*x/(right-left) for x in near],
                any_near36_percent=100*anynear/(right-left),any_over40_ms=anyover)

def simulate(spec,seed=7,vectors=None):
    lanes=native_matched_decks(spec,seed);ssu=spec['ssu'];diskfree=[0.]*ssu
    vec=lambda n,r: vectors[n,r] if vectors is not None else [lanes[n][r][2]/ssu]*ssu
    serial=itertools.count();events=[];busy=[0.]*8;seen=[set() for _ in range(8)]
    current=[];prefetch=[];linkrefs=[[] for _ in range(8)]
    clip=lambda a,b:max(0.,min(b,4000.)-max(a,2000.))
    def read_at(release,n,r):
        changes=[]
        for s,v in enumerate(vec(n,r)):
            a=max(release,diskfree[s]);b=a+v/40*1000;diskfree[s]=b
            changes.extend(((a,40.),(b,-40.)))
        changes.sort();last=changes[0][0];backlog=0.;rate=0.
        for now,delta in changes:
            backlog=max(0.,backlog+(rate-50)*(now-last)/1000);rate+=delta;last=now
        # Outstanding layer reads never overlap on one lane. Fluid receipt
        # is conservative only to one final max-sized link IO tail.
        return last+backlog/50*1000+176/1024**2/50*1000
    for n in range(8):heapq.heappush(events,(read_at(0,n,0),next(serial),n,0,0,0.))
    while events:
        start,_,n,r,l,barrier=heapq.heappop(events)
        role,c,v,k,q=lanes[n][r];end=start+c
        current.append((barrier,end,[x/c*1000 for x in vec(n,r)]))
        got=clip(start,end);busy[n]+=got
        if got:seen[n].add(role)
        if start>=4500:continue
        nr,nl=(r,l+1) if l<7 else (r+1,0)
        if nr<len(lanes[n]):
            prefetch.append((start,end,[x/c*1000 for x in vec(n,nr)]))
            linkrefs[n].append((start,end,sum(vec(n,nr))/c*1000,r,l,nr,nl))
            ready=read_at(start,n,nr)
            heapq.heappush(events,(max(end,ready),next(serial),n,nr,nl,end))
    checks={}
    for name,(a,b) in dict(warm=(2000,4000),steady=(500,4000),startup_to_end=(0,4500)).items():
        checks[name]={kind:audit(rows,ssu,a,b) for kind,rows in [('current',current),('prefetch',prefetch)]}
    linkchecks={}
    for name,(a,b) in dict(warm=(2000,4000),steady=(500,4000),startup_to_end=(0,4500)).items():
        peaks=[];overs=[]
        for lane in linkrefs:
            selected=[row for row in lane if min(b,row[1])-max(a,row[0])>1e-9]
            peaks.append(max((row[2] for row in selected),default=0.))
            overs.append(sum(min(b,row[1])-max(a,row[0]) for row in selected if row[2]>50+1e-8))
        linkchecks[name]=dict(window_ms=[a,b],per_npu_peak_gib_s=peaks,per_npu_over50_ms=overs,
            passed=all(x<=50+1e-8 for x in peaks))
    wholedeck=[]
    for n,lane in enumerate(lanes):
        refs=[]
        for r,p in enumerate(lane):
            refs.append(dict(source_request_index=r,target_request_index=r,kind='internal',reference_gib_s=p[2]/p[1]*1000))
            if r+1<len(lane):refs.append(dict(source_request_index=r,target_request_index=r+1,kind='next_L0',reference_gib_s=lane[r+1][2]/p[1]*1000))
        wholedeck.append(max(refs,key=lambda x:x['reference_gib_s']))
    wholedeck_check=dict(per_npu_max_transition=wholedeck,passed=all(x['reference_gib_s']<=50+1e-8 for x in wholedeck))
    passed=all(x['any_over40_ms']<1e-8 and x['any_near36_percent']<=5. for w in checks.values() for x in w.values())
    return dict(spec=spec,seed=seed,approximate_only=True,actual_ring=vectors is not None,
        U_percent=sum(busy)/16000*100,lane_U_percent=[x/2000*100 for x in busy],
        all_groups_every_npu_warm=all(len(x)==len(spec['groups']) for x in seen),
        warm_groups=[sorted(x) for x in seen],checks=checks,passed=passed,
        link_reference_checks=linkchecks,whole_deck_link_transition_check=wholedeck_check,
        disk_and_link_passed=passed and all(x['passed'] for x in linkchecks.values()) and wholedeck_check['passed'])

def make_spec(i,ssu,tuples):
    maxlen=max(x[0] for x in tuples)
    groups=[dict(id=f'g{j}',role='L' if k==maxlen and j else 'S',total_k=k,nql=n,weight=w)
            for j,(k,n,w) in enumerate(tuples)]
    groups[0]['role']='S';groups[-1]['role']='L'
    return dict(name=f'strict_probe_{i:03}',ssu=ssu,groups=groups)

def main():
    begin=time.perf_counter();rng=random.Random(918271);rows=[];seen=set();candidates=[]
    # Target a short compute window facing much larger volumes, with some
    # long-compute / small-read jobs supplying true headroom.
    for s in (1,2,3,4):
      for sk,sn,lk,ln,sw,lw in [(32,1280,128,2048,1,1),(32,1024,200,2048,1,1),
          (32,512,200,2048,2,1),(32,768,200,1536,3,1),(32,256,128,2048,1,1),
          (32,384,200,2048,1,2),(32,512,128,1536,1,1),(32,256,200,3072,1,1),
          (32,768,200,3072,4,1),(32,512,200,1024,4,1),(32,1024,128,1536,1,1)]:
        candidates.append((s,[(sk,sn,sw),(lk,ln,lw)]))
    while len(candidates)<260:
        s=rng.choice((1,2,2,3,3,4));sk=rng.choice((32,32,48,64))
        sn=rng.choice((128,256,384,512,640,768,1024,1280,1536))
        lk=rng.choice((96,128,160,200,200));ln=rng.choice((768,1024,1536,2048,3072,4096))
        tuples=[(sk,sn,rng.choice((1,2,3,4,6,8))),(lk,ln,rng.choice((1,1,2,3)))]
        if rng.random()<.65:tuples.insert(1,(32,rng.choice((1536,2048,3072,4096)),rng.choice((1,2,4,8))))
        signature=(s,tuple(tuples))
        if signature in seen:continue
        seen.add(signature);candidates.append((s,tuples))
    for i,(s,tuples) in enumerate(candidates):
        spec=make_spec(i,s,tuples)
        try:row=simulate(spec)
        except ValueError:continue
        rows.append(row)
    rows.sort(key=lambda r:r['U_percent']);passed=[r for r in rows if r['passed'] and r['all_groups_every_npu_warm']]
    OUT.mkdir(parents=True,exist_ok=True)
    payload=dict(description=__doc__,elapsed_seconds=time.perf_counter()-begin,evaluated=len(rows),passed_count=len(passed),
        selection_seed=918271,workload_seed=7,rows=rows)
    (OUT/'strict_proxy_screen.json').write_text(json.dumps(payload,indent=2))
    (OUT/'strict_proxy_candidates.json').write_text(json.dumps([r['spec'] for r in passed[:6]],indent=2))
    print(json.dumps(dict(elapsed=payload['elapsed_seconds'],evaluated=len(rows),passed=len(passed),best=passed[:6]),indent=2))

if __name__=='__main__':main()
