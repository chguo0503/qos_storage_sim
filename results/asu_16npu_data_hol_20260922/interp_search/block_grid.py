#!/usr/bin/env python3
"""Independent near-capacity profile grid using the calibrated block selector.

Still a selector, not a native ASU/Once benchmark. Fixed 1:8 two-cohort input.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'phase_search/block_issue_probe.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('readonly_block_selector',SOURCE)
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)


def run(args):
    ma,mb,right=args
    base=[0]+[1]*8
    patterns=[base[4:]+base[:4] for _ in range(8)]+[base[:] for _ in range(8)]
    r=model.simulate(patterns,ma=ma,mb=mb,right=right)
    ca,va=model.profile(200,ma);cb,vb=model.profile(32,mb)
    r.update(C_A_ms=ca,C_B_ms=cb,V_A_MB=va,V_B_MB=vb,B_A_GB_s=va/ca,B_B_GB_s=vb/cb,
             static_any_mix_peak_GB_s=16*max(va/ca,vb/cb),right_ms=right,
             U_2_10=sum(r['windows'][:4])/4 if right>=10000 else None)
    assert r['static_any_mix_peak_GB_s']<40
    return r


def main():
    sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    plan=[(ma,mb,12000.) for ma in (3302,3328,3360,3400,3500) for mb in (2448,2500,2600,2700)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,args) for args in plan]):
            r=future.result();rows.append(r)
            print(r['ma'],r['mb'],'U_2_10',r['U_2_10'],'U_2_12',r['U'],flush=True)
    rows.sort(key=lambda r:r['U'])
    (HERE/'block_grid_all.json').write_text(json.dumps(dict(approximate_only=True,
        source_sha256=sha,rows=rows),indent=2)+'\n')
    # Longer periodic runs can reverse the short-window ranking.
    top=[r for r in rows if r['all_warm_AB']][:3]
    with ProcessPoolExecutor(max_workers=2) as pool:
        longer=list(pool.map(run,[(r['ma'],r['mb'],60000.) for r in top]))
    for r in top:
        r['long_60s']=next(v for v in longer if v['ma']==r['ma'] and v['mb']==r['mb'])
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==sha
    (HERE/'block_grid_shortlist.json').write_text(json.dumps(dict(approximate_only=True,
        source_sha256=sha,rows=top),indent=2)+'\n')
    for r in top:print('LONG',r['ma'],r['mb'],r['long_60s']['U'],flush=True)


if __name__=='__main__':main()
