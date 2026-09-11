#!/usr/bin/env python3
"""Rewindow complete saved layer logs; keep compute, I/O stall, and idle separate."""
import argparse,csv,gzip,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
F=HERE.parent
ROOT=F.parent.parents[1]
sys.path[:0]=[str(ROOT),str(F),str(F.parent)]
from run_baseline_npu32_stress import load_manifest,read_json,write_json

WINDOWS=[(0,4000),(1000,4000),(2000,4000),(2000,4250),(2000,4500),(2000,5000),(2000,6000),(2000,8000)]
EXPANDED=[(2000,4000),(2000,6000),(2000,8000),(2000,10000),(2000,12000),(4000,12000),(6000,12000)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def overlap(a,b,x,y):return np.maximum(0,np.minimum(y,b)-np.maximum(x,a))

def load_run(item):
    raw=read_json(item['result_path']);requests,meta=load_manifest(item['manifest'])
    assert raw['input_fingerprint']==meta['input_fingerprint']
    assert all(raw['summary']['invariants'].values())
    byid={r.request_id:r for r in requests}
    layer=[];batch=[];seen=[]
    for b in raw['summary']['microbatch_metrics']:
        assert b['batch_size']==1
        rid=b['member_request_ids'][0];seen.append(rid);r=byid[rid];n=b['npu_id'];role=int(r.load['role']=='long')
        assert n==r.npu_id
        prev=b['admission_time_ms'];end=b['completion_time_ms']
        batch.append((n,role,prev,end,8*r.load['per_layer_us']/1000))
        for x in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            cs,ce=x['compute_start_ms'],x['compute_end_ms']
            assert math.isclose(cs-prev,x['io_barrier_wait_ms'],abs_tol=1e-6)
            assert math.isclose(ce-cs,r.load['per_layer_us']/1000,abs_tol=1e-6)
            layer.append((n,role,cs,ce,prev,cs));prev=ce
        assert math.isclose(prev,end,abs_tol=1e-6)
    assert sorted(seen)==sorted(byid)
    la=np.array(layer);ba=np.array(batch)
    finish=np.array([max(ba[ba[:,0]==n,3]) for n in range(32)])
    starts=np.array([min(ba[ba[:,0]==n,2]) for n in range(32)])
    summary=dict(item,requests=len(byid),first_card_done_ms=float(min(finish)),last_card_done_ms=float(max(finish)),
        last_initial_admission_ms=float(max(starts)),per_npu_last_completion_ms=finish.tolist(),
        result_sha256=sha(item['result_path']),manifest_sha256=sha(item['manifest']))
    return summary,la,ba

def window(summary,la,ba,a,b):
    duration=b-a
    comp=overlap(a,b,la[:,2],la[:,3]);stall=overlap(a,b,la[:,4],la[:,5]);active=overlap(a,b,ba[:,2],ba[:,3])
    cc=np.bincount(la[:,0].astype(int),weights=comp,minlength=32)
    ss=np.bincount(la[:,0].astype(int),weights=stall,minlength=32)
    aa=np.bincount(ba[:,0].astype(int),weights=active,minlength=32)
    idle=duration-aa
    assert np.allclose(cc+ss,aa,rtol=0,atol=1e-5)
    assert np.min(idle)>-1e-5
    ctot,stot,atot=map(float,[sum(cc),sum(ss),sum(aa)]);den=32*duration
    rolec=[np.bincount(la[:,0].astype(int),weights=comp*(la[:,1]==role),minlength=32) for role in (0,1)]
    admitted=(ba[:,2]>=a)&(ba[:,2]<b);passed=(ba[:,3]-ba[:,2]<=1.5*ba[:,4]+1e-8)
    return dict(label=summary['label'],mode=summary['mode'],strategy=summary['strategy'],seed=summary['seed'],
        start_ms=a,end_ms=b,width_ms=duration,U_percent=100*ctot/den,
        stall_percent=100*stot/den,idle_percent=100*(den-atot)/den,
        processing_only_U_percent=100*ctot/atot if atot else None,
        compute_npu_ms=ctot,stall_npu_ms=stot,idle_npu_ms=den-atot,
        all_32_active=bool(np.max(idle)<1e-5),cards_with_exhaustion=sum(t<b-1e-7 for t in summary['per_npu_last_completion_ms']),
        warm_mixed_cards=int(sum((rolec[0]>1e-8)&(rolec[1]>1e-8))),
        mean_long_cards=float(sum(active*(ba[:,1]==1))/duration),
        mean_short_cards=float(sum(active*(ba[:,1]==0))/duration),
        min_card_role_compute_ms=float(min(np.min(rolec[0]),np.min(rolec[1]))),
        first_card_done_ms=summary['first_card_done_ms'],last_card_done_ms=summary['last_card_done_ms'],
        admission_slo_count=int(sum(admitted)),admission_slo_passed=int(sum(admitted&passed)))

def old_items():
    items=[]
    for mode in ['fixed','random','ordered']:
      for seed in [7,19,43,67,101]:
        parent=F if mode=='fixed' else F/'mixed_rebinding'
        label=f'raw176_three_l20_fixed_seed{seed}' if mode=='fixed' else f'raw176_extendedhot_{mode}_seed{seed}'
        for policy in ['baseline','once']:
          result=list((parent/'runs'/label/policy).glob('*.json.gz'));assert len(result)==1
          items.append(dict(label=label,mode=mode,strategy=policy,seed=seed,manifest=str(parent/'inputs'/f'{label}.json.gz'),result_path=str(result[0])))
    return items

def aggregate(rows):
    result=[]
    for key in sorted({(r['mode'],r['strategy'],r['start_ms'],r['end_ms']) for r in rows}):
        rs=[r for r in rows if (r['mode'],r['strategy'],r['start_ms'],r['end_ms'])==key]
        out=dict(zip(['mode','strategy','start_ms','end_ms'],key),seeds=len(rs),all_active_count=sum(r['all_32_active'] for r in rs))
        for field in ['U_percent','stall_percent','idle_percent','processing_only_U_percent','mean_long_cards','mean_short_cards']:
            v=[r[field] for r in rs if r[field] is not None]
            out[field]=dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v) if len(v)>1 else None) if v else None
        result.append(out)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repeated',action='store_true');args=ap.parse_args()
    if args.repeated:
        parent=HERE/'repeated_queues';plan=read_json(parent/'plan.json');items=[]
        for job in plan['jobs']:
            item=job['input'];policy=job['strategy'];result=list((parent/'runs'/item['label']/policy).glob('*.json.gz'));assert len(result)==1
            items.append(dict(label=item['label'],mode=item['mode'],strategy=policy,seed=item['seed'],manifest=item['manifest'],result_path=str(result[0])))
        windows=EXPANDED;prefix='repeated'
    else:items=old_items();windows=WINDOWS;prefix='existing'
    data=[load_run(i) for i in items]
    if not args.repeated:
        common=math.floor((min(s['first_card_done_ms'] for s,_,_ in data)-1e-6)/50)*50
        if (2000,common) not in windows:windows=windows+[(2000,common)]
    rows=[window(s,l,b,a,z) for s,l,b in data for a,z in windows]
    # Account for any core summary window in the new files; old [2,4) also checked.
    for s,l,b in data:
        raw=read_json(s['result_path'])
        for w in raw['windows']:
            own=window(s,l,b,w['start_ms'],w['end_ms'])
            assert math.isclose(own['U_percent'],100*w['mean_npu_utilization'],abs_tol=1e-7)
    sliding=[]
    stop=10000 if args.repeated else 6000
    for s,l,b in data:
        for a in range(0,stop+1,50):sliding.append(window(s,l,b,a,a+2000))
    out=dict(runs=[s for s,_,_ in data],rows=rows,groups=aggregate(rows),sliding_groups=aggregate(sliding),
        windows_ms=windows,analyzer_sha256=sha(__file__),definition='U = compute/(32*T); processing_only_U = compute/active, a different metric; idle is measured independently from admissions/completions.')
    write_json(HERE/f'{prefix}_windows.json',out)
    with (HERE/f'{prefix}_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(prefix=prefix,runs=len(data),common_first_done_ms=min(s['first_card_done_ms'] for s,_,_ in data),groups=out['groups']),ensure_ascii=False))

if __name__=='__main__':main()
