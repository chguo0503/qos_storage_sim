#!/usr/bin/env python3
"""Read frozen actual Ring Hash inputs; continuous service remains approximate."""
import concurrent.futures
import gzip
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def payload(path):
    data=json.load(gzip.open(path,'rt'));cards=[[]for _ in range(32)]
    for r in data['requests']:
        v=[0.,0.,0.]
        for s,b in data['placements'][r['placement_index']][0]:v[s]+=b
        cards[r['npu_id']].append((int(r['load']['role']=='B'),r['load']['per_layer_us']/1000,*v))
    return '\n'.join(str(len(q))+'\n'+'\n'.join(' '.join(map(str,r))for r in q)for q in cards)

def run(text,scales=(1.,1.),horizon=8000):
    inp=f'{horizon} 1 {scales[0]} {scales[1]}\n'+text
    p=subprocess.run([str(HERE/'ring_fluid_proxy')],input=inp,capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return json.loads(p.stdout)

def valid(r):
    return r['over_ms_all']<1e-7 and r['min_active_all']==32 and all(w['mixed_cards']==32 and w['min_active']==32 for w in r['windows'])

def main():
    names=['synth_phase80_1210','synth_phase80_1210_c103','synth_phase180_1039','synth_phase180_1039_c103']
    texts={};initial=[]
    for name in names:
        path=HERE.parent/'inputs'/f'{name}.json.gz';texts[name]=payload(path)
        r=run(texts[name]);initial.append(dict(name=name,manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),result=r))
        print(json.dumps(dict(name=name,maxd=r['max_demand_by_ssu'],over_ms=r['over_ms_all'],U=[w['U']for w in r['windows']],mix=[w['mixed_cards']for w in r['windows']])),flush=True)
    (HERE/'ring_initial.json').write_text(json.dumps(initial,indent=2))
    specs=[]
    for name in ('synth_phase80_1210','synth_phase180_1039'):
        for ai in range(-10,21):
            for bi in range(-10,21):
                specs.append((name,1+ai*.005,1+bi*.005))
    def one(spec):
        name,sa,sb=spec
        return dict(name=name,scale_A=sa,scale_B=sb,**run(texts[name],(sa,sb)))
    start=time.time();results=[];best=1.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
        for i,r in enumerate(pool.map(one,specs)):
            results.append(r)
            score=max(w['U']for w in r['windows'])
            if valid(r)and score<best:
                best=score
                print(json.dumps(dict(index=i,name=r['name'],scale_A=r['scale_A'],scale_B=r['scale_B'],maxd=r['max_demand_by_ssu'],U=[w['U']for w in r['windows']])),flush=True)
            if (i+1)%300==0:print(json.dumps(dict(progress=i+1,best=best,elapsed=time.time()-start)),flush=True)
    passed=sorted((r for r in results if valid(r)),key=lambda r:max(w['U']for w in r['windows']))
    (HERE/'ring_results.json').write_text(json.dumps(results,indent=2))
    (HERE/'ring_valid.json').write_text(json.dumps(passed,indent=2))
    print(json.dumps(dict(done=len(results),valid=len(passed),elapsed=time.time()-start)),flush=True)

if __name__=='__main__':main()
