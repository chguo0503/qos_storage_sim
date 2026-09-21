#!/usr/bin/env python3
"""Screen composition-dependent underload; no arbitrary-mixture guarantee.

Uses the independent block-issue selector, not the native simulator. The full
trajectory must pass, and any selected case must later be checked separately
for ASU and Once because their current request mixtures differ.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'phase_search/block_issue_probe.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('conditional_readonly_block',SOURCE)
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


def run(args):
    ma,mb,q,off,right=args
    base=[0]+[1]*q
    pats=[base[off:]+base[:off] for _ in range(8)]+[base[:] for _ in range(8)]
    r=model.simulate(pats,ma=ma,mb=mb,right=right)
    ca,va=model.profile(200,ma);cb,vb=model.profile(32,mb)
    ba,bb=va/ca,vb/cb
    r.update(q=q,offset=off,right_ms=right,C_A_ms=ca,C_B_ms=cb,
             B_A_GB_s=ba,B_B_GB_s=bb,nominal_8A8B_GB_s=8*(ba+bb),
             static_any_mix_peak_GB_s=16*max(ba,bb),
             allowed_max_A=math.ceil((40-16*bb)/(ba-bb))-1 if ba>bb else 16,
             U_2_10=sum(r['windows'][:4])/4 if right>=10000 else None)
    return r


def main():
    sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    profiles=[]
    for ma in (2500,2600,2700,2800,2900,3000,3100,3200):
        for mb in (2800,3000,3200,3600,4096):
            ca,va=model.profile(200,ma);cb,vb=model.profile(32,mb)
            if 8*(va/ca+vb/cb)<40:profiles.append((ma,mb))
    plan=[(ma,mb,q,off,12000.) for ma,mb in profiles for q in (3,4,5)
          for off in sorted({1,math.ceil(q/2)})]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for index,future in enumerate(as_completed([pool.submit(run,args) for args in plan]),1):
            r=future.result();rows.append(r)
            if r['overload_ms']==0 and r['all_warm_AB']:
                print('PASS',r['ma'],r['mb'],r['q'],r['offset'],r['U'],r['peak_demand_gb_s'],flush=True)
            elif index%20==0:print('progress',index,'/',len(plan),flush=True)
    rows.sort(key=lambda r:r['U'])
    (HERE/'conditional_grid_all.json').write_text(json.dumps(dict(approximate_only=True,
        composition_dependent_underload=True,source_sha256=sha,raw_profile_grid=40,
        startup_8A8B_underload_profiles=len(profiles),rows=rows),indent=2)+'\n')
    selected=[];seen=set()
    for r in rows:
        if r['overload_ms']>0 or not r['all_warm_AB']:continue
        key=r['ma'],r['mb']
        if key in seen:continue
        seen.add(key);selected.append(r)
        if len(selected)==5:break
    with ProcessPoolExecutor(max_workers=3) as pool:
        longer=list(pool.map(run,[(r['ma'],r['mb'],r['q'],r['offset'],60000.) for r in selected]))
    for r,long in zip(selected,longer):r['long_60s']=long
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha
    (HERE/'conditional_grid_shortlist.json').write_text(json.dumps(dict(approximate_only=True,
        composition_dependent_underload=True,source_sha256=sha,rows=selected),indent=2)+'\n')
    for r in selected:print('LONG',r['ma'],r['mb'],r['q'],r['offset'],r['long_60s']['U'],r['long_60s']['overload_ms'],flush=True)


if __name__=='__main__':main()
