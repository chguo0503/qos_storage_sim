#!/usr/bin/env python3
"""Offline data-grid phase alignment selector; final claims require native replay."""
from __future__ import annotations
import ast,heapq,itertools,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=ast.literal_eval((ROOT/'data').read_text())

def interp_compute_us(total_tokens,miss):
    k=total_tokens/1024
    ks=sorted(set(a for a,b in DATA));ms=sorted(set(b for a,b in DATA))
    lo=max(x for x in ks if x<=k);hi=min(x for x in ks if x>=k)
    ml=max(x for x in ms if x<=miss);mh=min(x for x in ms if x>=miss)
    kw=0 if hi==lo else (k-lo)/(hi-lo);mw=0 if mh==ml else (miss-ml)/(mh-ml)
    terms={}
    for x,wx in [(lo,1-kw),(hi,kw)]:
        for m,wm in [(ml,1-mw),(mh,mw)]:terms[x,m]=terms.get((x,m),0)+wx*wm
    anchors=[dict(seq_len_k=x,nql=m,weight=w,compute_us=DATA[x,m][1],original_data_row=list(DATA[x,m])) for (x,m),w in terms.items() if w]
    return sum(a['weight']*a['compute_us'] for a in anchors),anchors

def make_profile(total_tokens,miss,family):
    c,anchors=interp_compute_us(total_tokens,miss);v=(total_tokens-miss)*1408/2**30
    return dict(family=family,total_tokens=total_tokens,total_length_k=total_tokens/1024,nql=miss,
                ssd_prefix_tokens=total_tokens-miss,compute_us=c,read_gib=v,B_gib_s=v*1e6/c,
                constructed_profile=True,profile_construction=dict(method='bilinear_interpolation_total_tokens_and_miss',source='data',
                extrapolated=False,total_length_interpolated=total_tokens%1024!=0 or total_tokens/1024 not in [k for k,m in DATA],
                compute_scale=1.,kv_formula='exact hit prefix tokens * 1408 bytes; no block padding',anchors=anchors))

def nearest_profile(target_ms):
    target=target_ms*1000;best=None
    # Every miss is integral; choose the nearest integral total-token count.
    for m in range(3200,4097):
        weight=(m-2048)/2048
        low=DATA[192,2048][1]*(1-weight)+DATA[192,4096][1]*weight
        high=DATA[200,2048][1]*(1-weight)+DATA[200,4096][1]*weight
        if not low<=target<=high:continue
        token=round(192*1024+(target-low)/(high-low)*8*1024)
        if not 192*1024<=token<=200*1024:continue
        got=low+(high-low)*(token-192*1024)/(8*1024);error=abs(got-target)
        if best is None or error<best[0]:best=(error,token,m)
    if best is None:raise ValueError(f'No in-grid long profile for C={target_ms}ms')
    return make_profile(best[1],best[2],'A'),best[0]

def simulate(catalog,seqs,right=60000.,left=2000.):
    pre={}
    for key,p in catalog.items():
        full,tail=divmod(p['ssd_prefix_tokens'],128)
        pre[key]=(p['compute_us']/1000,[128*1408/1e6]*full+([tail*1408/1e6] if tail else []),p['family'])
    events=[];serial=itertools.count();free=0.;busy=0.;seen=[set() for _ in range(16)];windows=[0.]*math.ceil((right-2000)/2000)
    starts={};ends={};stalls={};phases={};demand_events=[]
    def read(now,n,r,l,barrier):
        if r<len(seqs[n]):heapq.heappush(events,(now,next(serial),0,n,r,l,barrier,0))
    for n,s in enumerate(seqs):
        p=catalog[s[0]];demand_events.append((0.,p['read_gib']*2**30/p['compute_us']/1000));read(0,n,0,0,0.)
    while events:
        now,_,kind,n,r,l,barrier,k=heapq.heappop(events)
        if now>=right:continue
        key=seqs[n][r];c,blocks,family=pre[key]
        if kind==0:
            v=blocks[k];free=max(free,now)+v/40
            if k+1<len(blocks):heapq.heappush(events,(now+.0001,next(serial),0,n,r,l,barrier,k+1))
            else:heapq.heappush(events,(max(barrier,free+v/50),next(serial),1,n,r,l,barrier,0))
            continue
        if l==0:starts[n,r]=now
        end=now+c;busy+=max(0.,min(end,right)-max(now,left));stalls[n,r]=stalls.get((n,r),0)+now-barrier
        phases[n,r,l]=now
        for w in range(len(windows)):windows[w]+=max(0.,min(end,right,4000+w*2000)-max(now,2000+w*2000))
        if min(end,4000)>max(now,2000):seen[n].add(family)
        nr,nl=(r,l+1) if l<7 else (r+1,0)
        if l==7:
            ends[n,r]=end
            old=catalog[key]['read_gib']*2**30/catalog[key]['compute_us']/1000
            new=0. if nr>=len(seqs[n]) else catalog[seqs[n][nr]]['read_gib']*2**30/catalog[seqs[n][nr]]['compute_us']/1000
            if end<right:demand_events.append((end,new-old))
        read(now,n,nr,nl,end)
    d=0.;peak=0.;over=0.;prev=0.
    for now,g in itertools.groupby(sorted(demand_events),key=lambda e:e[0]):
        if d>40+1e-8:over+=now-prev
        d+=sum(x for _,x in g);peak=max(peak,d);prev=now
    if d>40+1e-8:over+=right-prev
    metrics=dict(approximate_only=True,block_issue_model=True,U=busy/16/(right-left)*100,
                 windows=[w/32000*100 for w in windows],all_warm_AB=all(len(s)==2 for s in seen),
                 warm_families_by_npu=[sorted(s) for s in seen],
                 peak_nominal_gb_s=peak,overload_ms=over,right_ms=right)
    return metrics,dict(starts=starts,ends=ends,stalls=stalls,phases=phases)

def workload(cycles=5,na=10,nb=45):
    catalog={'A':make_profile(200*1024,3302,'A'),'B':make_profile(32*1024,2448,'B')}
    seqs=[];groups=[]
    for n in range(16):
        seq=(['A']*3+['B']*15 if n<8 else ['B']*15+['A']*3)
        seq+=(['A']*na+['B']*nb if n<8 else ['B']*nb+['A']*na)*cycles
        seqs.append(seq)
    for cohort in [0,1]:
        n=cohort*8
        for r,key in enumerate(seqs[n]):
            if key=='A' and r>0 and seqs[n][r-1]=='B':
                members=[]
                for i in range(n,n+8):
                    custom=f'align_g{cohort}_r{r}_n{i}';catalog[custom]=dict(catalog['A']);seqs[i][r]=custom;members.append((i,r,custom))
                groups.append(members)
    return catalog,seqs,groups

if __name__=='__main__':
    cat,seqs,groups=workload();history=[]
    for iteration in range(12):
        met,trace=simulate(cat,seqs);history.append(dict(iteration=iteration,metrics=met));print(iteration,met,flush=True)
        errors=[];skipped=[]
        for members in groups:
            if any((n,r) not in trace['starts'] for n,r,key in members):continue
            times=[trace['starts'][n,r] for n,r,key in members]
            target=max(times)+8*cat['A']['compute_us']/1000+.01
            proposed=[]
            try:
                for n,r,key in members:
                    p,error=nearest_profile((target-trace['starts'][n,r])/8);proposed.append((key,p,error))
            except ValueError as exc:
                skipped.append(dict(group=members[0][2],reason=str(exc)));continue
            for key,p,error in proposed:cat[key]=p;errors.append(error)
        history[-1]['max_compute_match_error_us']=max(errors,default=0.)
        history[-1]['skipped_groups']=skipped
        (HERE/'alignment_partial.json').write_text(json.dumps(dict(history=history),indent=2)+'\n')
    met,trace=simulate(cat,seqs);history.append(dict(iteration='final',metrics=met));print('final',met,flush=True)
    phase=[]
    for members in groups:
        if all((n,r+1) in trace['starts'] for n,r,key in members):
            times=[trace['starts'][n,r+1] for n,r,key in members]
            phase.append(dict(group=members[0][2],second_A_start_spread_ms=max(times)-min(times),second_A_start_min_ms=min(times)))
    (HERE/'alignment_history.json').write_text(json.dumps(dict(history=history,phase=phase),indent=2)+'\n')
    (HERE/'aligned_workload.json').write_text(json.dumps(dict(profiles_catalog=cat,per_npu_sequences=seqs,
        construction={'method':'offline_iteration_first_A_compute_alignment','iterations':12,'common_target_margin_ms':.01,
                      'num_npu':16,'ssu':1,'prefix_A_count':3,'prefix_B_count':15,'block_A':10,'block_B':45,'cycles':5,
                      'interpolation_bounds_total_tokens':[192*1024,200*1024],'data_source':'data'},
        approximate_metrics=met),indent=2)+'\n')
