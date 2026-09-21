#!/usr/bin/env python3
"""Independent controls on the frozen 60-second candidate; inputs never modified."""
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,random,sys
from align_probe import HERE,simulate
sys.path.insert(0,str(HERE.parent))
from variant_runner import make_profile

def work(mode):
    raw=(HERE/'aligned_sequential_60000ms_config.json').read_bytes();base=json.loads(raw)
    rng=random.Random(789);cat={};standard=base['profiles_catalog']['A'];changes=0
    for k,v in base['profiles_catalog'].items():
        t,m=v['total_tokens'],v['nql']
        corrected=v['family']=='A' and (t,m)!=(standard['total_tokens'],standard['nql'])
        if corrected:
            if mode=='total_token_jitter_1':t+=rng.choice([-1,1])
            elif mode=='miss_jitter_1':m+=rng.choice([-1,1])
            elif mode=='total_round_1k':t=round(t/1024)*1024
            elif mode=='miss_round_64':m=round(m/64)*64
        if (t,m)!=(v['total_tokens'],v['nql']):changes+=1
        cat[k]=make_profile(t,m,v['family'])
    seqs=[s.copy() for s in base['per_npu_sequences']]
    if mode.startswith('shuffle_seed'):
        seed=int(mode.removeprefix('shuffle_seed'))
        for n,s in enumerate(seqs):random.Random(seed+100003*n).shuffle(s)
    metrics,_=simulate(cat,seqs,right=60000.)
    return dict(mode=mode,changed_profile_count=changes,source_config_sha256=hashlib.sha256(raw).hexdigest(),
                queues_sha256=hashlib.sha256(json.dumps(seqs,separators=(',',':')).encode()).hexdigest(),metrics=metrics)

if __name__=='__main__':
    modes=['unaltered','total_token_jitter_1','miss_jitter_1','total_round_1k','miss_round_64','shuffle_seed7','shuffle_seed19']
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(work,modes))
    (HERE/'sensitivity_and_shuffle_60s.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:print(row['mode'],row['metrics']['U'],row['metrics']['windows'][0],row['metrics']['all_warm_AB'],flush=True)
