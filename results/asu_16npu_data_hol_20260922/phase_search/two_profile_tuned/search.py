#!/usr/bin/env python3
"""Bounded two-profile offline search; all outputs are predictions, not native results."""
import concurrent.futures as cf
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
ROOT=STUDY.parents[1]
sys.path.insert(0,str(STUDY/'phase_alignment'))
sys.path.insert(0,str(STUDY))
from native_like_probe import simulate
from variant_runner import make_profile, validate_config

SETTINGS=[(3302,2448),(3310,2448),(3330,2448),(3370,2448),(3430,2448),
          (3478,2448),(3540,2448),(3302,2464),(3302,2496),(3302,2560)]
CPUS=[8,10,12,14]

def source_hashes():
    paths=[Path(__file__),STUDY/'phase_alignment/native_like_probe.py',STUDY/'phase_alignment/align_probe.py',STUDY/'variant_runner.py',STUDY/'runner.py',ROOT/'data']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def config(ma,mb,q,kind='periodic'):
    profiles={'A':make_profile(200*1024,ma,'A'),'B':make_profile(32*1024,mb,'B')}
    if kind=='fixed_roles':
        seqs=[['A']*100 if n<8 else ['B']*650 for n in range(16)]
    else:
        seqs=[(['A']*3+['B']*15+(['A']*10+['B']*q)*6) if n<8 else
              (['B']*15+['A']*3+(['B']*q+['A']*10)*6) for n in range(16)]
    return dict(name=f'A{ma}_B{mb}_q{q}_{kind}',num_npu=16,ssu=1,seed=7,horizon_ms=60000.,
        window_ms=[2000,4000],mode='variant_explicit',profiles_catalog=profiles,per_npu_sequences=seqs,
        design_note='Exactly two shared physical profiles; no per-card correction, no inserted idle. Opposite ordered cohorts, not random.')

def work(spec):
    ma,mb,q,kind,right=spec
    # CPU affinity affects wall time only; four independent, single-threaded jobs.
    identity=os.getpid()%len(CPUS)
    # Parent initializer fixes an actual CPU; do not change it per candidate.
    cfg=config(ma,mb,q,kind); cat=cfg['profiles_catalog']; seqs=cfg['per_npu_sequences']
    if kind=='periodic':validate_config(cfg)
    maximum=16*max(p['B_gib_s']*2**30/1e9 for p in cat.values())
    assert maximum<40
    started=time.monotonic(); metrics,trace=simulate(cat,seqs,right=right,seed=7)
    def measure(left,stop):
        per=[{'A':0.,'B':0.} for _ in range(16)]
        for (n,r,l),a in trace['phases'].items():
            key=seqs[n][r]; b=a+cat[key]['compute_us']/1000
            per[n][key]+=max(0.,min(stop,b)-max(left,a))
        total=math.fsum(math.fsum(x.values()) for x in per)
        return dict(start_ms=left,end_ms=stop,U_percent=total/16/(stop-left)*100,
            per_card_U=[math.fsum(x.values())/(stop-left)*100 for x in per],
            all_cards_compute_both=all(x['A']>0 and x['B']>0 for x in per),
            mixed_card_count=sum(x['A']>0 and x['B']>0 for x in per),
            group_A_compute_card_ms=math.fsum(x['A'] for x in per),
            group_B_compute_card_ms=math.fsum(x['B'] for x in per))
    ranges=[(2000.,right),(2000.,4000.),(4000.,min(right,10000.)),(8000.,min(right,12000.))]
    if right>=60000:ranges += [(20000.,40000.),(40000.,60000.),(20000.,60000.)]
    windows={f'{int(a/1000)}_{int(b/1000)}':measure(a,b) for a,b in ranges}
    phase=[]
    if kind=='periodic':
        for g in range(2):
            members=range(8*g,8*g+8)
            for cycle in range(6):
                for family,r in ([('A',18+cycle*(10+q)),('B',28+cycle*(10+q))] if g==0 else [('B',18+cycle*(10+q)),('A',18+q+cycle*(10+q))]):
                    starts=[trace['starts'].get((n,r)) for n in members]
                    if all(x is not None for x in starts):phase.append(dict(group=g,cycle=cycle,family=family,start_min_ms=min(starts),start_max_ms=max(starts),spread_ms=max(starts)-min(starts)))
    out=dict(name=cfg['name'],kind=kind,A_miss=ma,B_miss=mb,q=q,right_ms=right,
        approximate_only=True,native_run=False,source_hashes=source_hashes(),wall_seconds=time.monotonic()-started,
        profile_metrics={k:dict(total_tokens=p['total_tokens'],miss=p['nql'],C_ms=p['compute_us']/1000,V_MB=p['read_gib']*2**30/1e6,B_GB_s=p['B_gib_s']*2**30/1e9) for k,p in cat.items()},
        strict_any_mix_D_upper_GB_s=maximum,profile_C_ratio=cat['A']['compute_us']/cat['B']['compute_us'],
        original_model_metrics=metrics,windows=windows,phase_segments=phase,
        pure_work_ms_by_card=[8*math.fsum(cat[k]['compute_us'] for k in s)/1000 for s in seqs])
    if kind=='fixed_roles':
        m=measure(4000.,right);ua=sum(m['per_card_U'][:8])/8;ub=sum(m['per_card_U'][8:])/8
        out['fixed_role_estimate']=dict(U_A=ua,U_B=ub,q_for_10_A=(cat['A']['compute_us']/cat['B']['compute_us'])*ub/ua*10,
            caveat='Finite-window throughput estimate, not a proof of matching after roles swap.')
    return out

def initializer(queue):
    os.sched_setaffinity(0,{queue.get()})

def main():
    import multiprocessing
    initial=source_hashes(); jobs=[(a,b,q,'periodic',12000.) for a,b in SETTINGS for q in [44,45,46,47,48]]
    jobs += [(a,b,0,'fixed_roles',12000.) for a,b in SETTINGS]
    (HERE/'plan.json').write_text(json.dumps(dict(source_hashes=initial,short_jobs=jobs,workers=4,cpus=CPUS,
        selection_rule='Best four 2-12s U, best two 8-12s U, and original I1 q45; unique, max eight; extend all to60s.'),indent=2)+'\n')
    context=multiprocessing.get_context('fork'); queue=context.Queue()
    for cpu in CPUS:queue.put(cpu)
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=4,mp_context=context,initializer=initializer,initargs=(queue,)) as pool:
        future={pool.submit(work,job):job for job in jobs}
        for f in cf.as_completed(future):
            row=f.result();rows.append(row)
            (HERE/'short_results.json').write_text(json.dumps(sorted(rows,key=lambda x:x['name']),indent=2)+'\n')
            print('short',len(rows),len(jobs),row['name'],row['windows']['2_12']['U_percent'],flush=True)
        periodic=[r for r in rows if r['kind']=='periodic']
        selected=[]
        for r in sorted(periodic,key=lambda x:x['windows']['2_12']['U_percent'])[:4]+sorted(periodic,key=lambda x:x['windows']['8_12']['U_percent'])[:2]+[r for r in periodic if r['A_miss']==3302 and r['B_miss']==2448 and r['q']==45]:
            if r['name'] not in [x['name'] for x in selected]:selected.append(r)
        (HERE/'selection.json').write_text(json.dumps([dict(name=r['name'],A_miss=r['A_miss'],B_miss=r['B_miss'],q=r['q']) for r in selected],indent=2)+'\n')
        full=[]
        for r in selected:
            cfg=config(r['A_miss'],r['B_miss'],r['q']);(HERE/(r['name']+'_config.json')).write_text(json.dumps(cfg,indent=2)+'\n')
        future={pool.submit(work,(r['A_miss'],r['B_miss'],r['q'],'periodic',60000.)):r for r in selected}
        for f in cf.as_completed(future):
            row=f.result();full.append(row)
            (HERE/'long_results.json').write_text(json.dumps(sorted(full,key=lambda x:x['name']),indent=2)+'\n')
            print('long',len(full),len(selected),row['name'], {k:v['U_percent'] for k,v in row['windows'].items()},flush=True)
    assert initial==source_hashes(),'Source changed during search'
    summary=dict(status='complete',approximate_only=True,short_cases=len(rows),long_cases=len(full),source_hashes=initial,
                 all_sources_unchanged=True,lowest_20_60=min(full,key=lambda r:r['windows']['20_60']['U_percent'])['name'])
    (HERE/'checks.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
