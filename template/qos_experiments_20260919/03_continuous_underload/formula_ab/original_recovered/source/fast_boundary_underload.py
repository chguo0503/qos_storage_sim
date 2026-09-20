"""Approximate FIFO search; cross-role next-L0 demand is separately exempt.

Native-matched frozen random decks, complete deck drainage, layer-at-once
FIFO surrogate, pipelined 50 GiB/s receipt, exact event demand sweeps.
Current V/C and all non-cross-role prefetch contributions remain audited
throughout every boundary. This is candidate screening, not native evidence.
"""
from __future__ import annotations
import argparse,heapq,itertools,json,random,time
from pathlib import Path
from fast_multitype_probe import native_matched_decks,profile
from fast_strict_underload import audit as disk_audit
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/boundary_exempt_underload_20260914'

def audit(intervals,ssu,left,right):
    result=disk_audit(intervals,ssu,left,right)
    fleet=disk_audit([(a,b,[sum(vec)/ssu]) for a,b,vec in intervals],1,left,right)
    result['fleet_peak_gib_s']=fleet['peak_gib_s'][0]*ssu
    result['fleet_near90_percent']=fleet['any_near36_percent']
    result['fleet_over_capacity_ms']=fleet['any_over40_ms']
    return result

def simulate(spec,seed=7,vectors=None):
    lanes=native_matched_decks(spec,seed);ssu=spec['ssu'];diskfree=[0.]*ssu
    group_role=[g['role'] for g in spec['groups']]
    vec=lambda n,r:vectors[n,r] if vectors is not None else [lanes[n][r][2]/ssu]*ssu
    serial=itertools.count();events=[];current=[];stable=[];boundary=[];allpref=[]
    computes=[];stalls=[];bounds=[];done=[0.]*8;seen=[set() for _ in range(8)]
    def read_at(release,n,r):
        changes=[]
        for s,v in enumerate(vec(n,r)):
            a=max(release,diskfree[s]);b=a+v/40*1000;diskfree[s]=b
            changes.extend(((a,40.),(b,-40.)))
        changes.sort();last=changes[0][0];backlog=0.;rate=0.
        for now,delta in changes:
            backlog=max(0.,backlog+(rate-50)*(now-last)/1000);rate+=delta;last=now
        return last+backlog/50*1000+176/1024**2/50*1000
    for n in range(8):heapq.heappush(events,(read_at(0,n,0),next(serial),n,0,0,0.))
    while events:
        start,_,n,r,l,barrier=heapq.heappop(events)
        gid,c,v,k,q=lanes[n][r];end=start+c
        current.append((barrier,end,[x/c*1000 for x in vec(n,r)]))
        computes.append((start,end,n,gid));stalls.append((barrier,start,n,gid,l,r))
        if max(start,2000)<min(end,4000):seen[n].add(gid)
        nr,nl=(r,l+1) if l<7 else (r+1,0)
        if nr<len(lanes[n]):
            cross=l==7 and group_role[gid]!=group_role[lanes[n][nr][0]]
            iv=(start,end,[x/c*1000 for x in vec(n,nr)])
            (boundary if cross else stable).append(iv);allpref.append(iv)
            solo=max(max(vec(n,nr))/40,sum(vec(n,nr))/50)*1000
            hard=max(0.,solo-c)
            if hard>1e-9:bounds.append((end,end+hard,n,lanes[n][nr][0],cross,r,nr,nl))
            ready=read_at(start,n,nr)
            heapq.heappush(events,(max(end,ready),next(serial),n,nr,nl,end))
        else:done[n]=end
    makespan=max(done)
    windows=dict(warm=(2000,4000),first4500=(0,4500),full=(0,makespan))
    checks={};metrics={}
    for key,(a,b) in windows.items():
        checks[key]={kind:audit(rows,ssu,a,b) for kind,rows in [('current',current),('non_cross_role_prefetch',stable),('cross_role_L0_prefetch',boundary),('total_prefetch',allpref)]}
        clip=lambda x,y:max(0.,min(y,b)-max(x,a))
        bygroup=[dict(group=g['id'],role=g['role'],compute_ms=0.,internal_stall_ms=0.,L0_stall_ms=0.,cold_start_ms=0.,boundary_hard_lower_ms=0.,non_boundary_hard_lower_ms=0.) for g in spec['groups']]
        for x,y,n,gid in computes:bygroup[gid]['compute_ms']+=clip(x,y)
        for x,y,n,gid,l,r in stalls:bygroup[gid]['cold_start_ms' if r==0 and l==0 else ('L0_stall_ms' if l==0 else 'internal_stall_ms')]+=clip(x,y)
        for x,y,n,gid,cross,*_ in bounds:bygroup[gid]['boundary_hard_lower_ms' if cross else 'non_boundary_hard_lower_ms']+=clip(x,y)
        for row in bygroup:
            row['U_percent']=100*row['compute_ms']/sum(row[x] for x in ('compute_ms','internal_stall_ms','L0_stall_ms','cold_start_ms')) if row['compute_ms'] else None
        csum=sum(r['compute_ms'] for r in bygroup)
        metrics[key]=dict(U_percent=100*csum/(8*(b-a)),bygroup=bygroup,
            internal_stall_ms=sum(r['internal_stall_ms'] for r in bygroup),
            L0_stall_ms=sum(r['L0_stall_ms'] for r in bygroup),
            boundary_hard_lower_ms=sum(r['boundary_hard_lower_ms'] for r in bygroup),
            non_boundary_hard_lower_ms=sum(r['non_boundary_hard_lower_ms'] for r in bygroup))
    current_pass=all(c['current']['any_over40_ms']<1e-8 and c['current']['fleet_near90_percent']<=5 for c in checks.values())
    stable_pass=all(c['non_cross_role_prefetch']['any_over40_ms']<1e-8 and c['non_cross_role_prefetch']['fleet_near90_percent']<=5 for c in checks.values())
    return dict(spec=spec,seed=seed,approximate_only=True,actual_ring=vectors is not None,
        U_percent=metrics['warm']['U_percent'],checks=checks,metrics=metrics,makespan_ms=makespan,
        full_deck_completed=True,per_npu_completion_ms=done,
        current_pass=current_pass,non_boundary_prefetch_pass=stable_pass,passed=current_pass and stable_pass,
        all_groups_every_npu_warm=all(len(x)==len(spec['groups']) for x in seen),
        all_active_warm=min(done)>=4000,counts_per_npu=[sum(p[0]==i for p in lanes[0]) for i in range(len(spec['groups']))])

def make_spec(i,s,triples):
    return dict(name=f'boundary_probe_{i:03}',ssu=s,groups=[dict(id=f'g{j}',role='L' if j==len(triples)-1 else 'S',total_k=k,nql=q,weight=w) for j,(k,q,w) in enumerate(triples)])

def candidates(count):
    rows=[];seen=set()
    def add(spec):
        key=(spec['ssu'],tuple((g['total_k'],g['nql'],g['weight'],g['role']) for g in spec['groups']))
        if key not in seen:seen.add(key);rows.append(spec)
    for dirname in ('strict_random_underload_20260914','random_multitype_search_20260914'):
        for p in (ROOT/'results'/dirname).glob('**/metadata.json'):
            m=json.loads(p.read_text());spec=m.get('specification')
            if spec:add(spec)
    for s in (1,2,3,4,6,8,10,12):
        for sk,sn,lk,ln,sw,lw in ((32,128,200,256,3,1),(32,128,200,512,6,1),(32,128,200,2048,12,1),
            (32,256,200,512,4,1),(32,256,200,2048,6,1),(32,512,128,512,4,1),(32,512,200,512,4,1),
            (32,512,200,2048,8,1),(32,768,200,2048,4,1),(32,1024,200,512,4,1),
            (32,128,128,256,3,1),(32,256,128,512,3,1),(32,512,128,1024,6,1)):
            add(make_spec(len(rows),s,[(sk,sn,sw),(lk,ln,lw)]))
    rng=random.Random(741939)
    while len(rows)<count:
        s=rng.choice((1,2,2,3,3,4,4,5,6,8,10,12))
        sk=rng.choice((32,32,32,48,64));sq=rng.choice((128,256,384,512,640,768,1024,1280))
        lk=rng.choice((96,128,128,160,200,200));lq=rng.choice((128,256,512,768,1024,1536,2048,3072,4096))
        triples=[(sk,sq,rng.choice((1,2,3,4,6,8,12,16))),(lk,lq,rng.choice((1,1,1,2,3)))]
        if rng.random()<.4:triples.insert(1,(32,rng.choice((1024,1536,2048,3072,4096)),rng.choice((1,2,4,8))))
        add(make_spec(len(rows),s,triples))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=540);p.add_argument('--ring-top',type=int,default=12);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();rows=[]
    for i,spec in enumerate(candidates(a.count)):
        try:row=simulate(spec)
        except ValueError:continue
        rows.append(row)
        if i%50==0:print(json.dumps(dict(progress=i,elapsed=time.perf_counter()-begin,passed=sum(r['passed'] for r in rows))),flush=True)
    rows.sort(key=lambda r:r['U_percent'])
    accepted=[r for r in rows if r['passed'] and r['all_groups_every_npu_warm'] and r['all_active_warm']]
    (OUT/'boundary_proxy_screen.json').write_text(json.dumps(dict(description=__doc__,count=len(rows),elapsed_seconds=time.perf_counter()-begin,rows=rows),indent=2))
    (OUT/'boundary_proxy_candidates.json').write_text(json.dumps([r['spec'] for r in accepted[:6]],indent=2))
    print(json.dumps(dict(event='equal_split_done',passed=len(accepted),best=[dict(spec=r['spec'],U=r['U_percent'],metrics=r['metrics']['warm'],checks=r['checks']['full']) for r in accepted[:6]])),flush=True)
    if a.ring_top:
        from run_random_multitype import build
        real=[]
        selected=accepted[:max(1,a.ring_top//2)]
        causal=[r for r in accepted if r['metrics']['warm']['boundary_hard_lower_ms']<1e-8]
        for row in causal+accepted:
            if row not in selected:selected.append(row)
            if len(selected)>=a.ring_top:break
        for row in selected:
            reqs,meta,bw=build(row['spec'],row['seed']);vectors={}
            for q in reqs:
                load=q.load;vectors[q.npu_id,load['generation']]=[x*load['per_layer_us']/1e6 for x in bw[q.request_id]]
            r=simulate(row['spec'],row['seed'],vectors);real.append(r)
            print(json.dumps(dict(event='real_ring',spec=r['spec'],U=r['U_percent'],passed=r['passed'],metrics=r['metrics']['warm'],checks=r['checks']['full'])),flush=True)
        real.sort(key=lambda r:r['U_percent']);(OUT/'boundary_proxy_real_ring.json').write_text(json.dumps(real,indent=2))
        good=[r for r in real if r['passed'] and r['all_groups_every_npu_warm']]
        (OUT/'boundary_proxy_candidates.json').write_text(json.dumps([r['spec'] for r in good[:6]],indent=2))
    print(json.dumps(dict(event='done',elapsed_seconds=time.perf_counter()-begin)),flush=True)
if __name__=='__main__':main()
