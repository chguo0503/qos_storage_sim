#!/usr/bin/env python3
"""Frozen corrected candidate sensitivity using the calibrated single-disk selector."""
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json,random,sys
from native_like_probe import HERE,simulate
sys.path.insert(0,str(HERE.parent))
from variant_runner import make_profile

SOURCE=HERE/'aligned_sequential_60000ms_native_like_config.json'
def work(mode):
    raw=SOURCE.read_bytes();base=json.loads(raw);rng=random.Random(789)
    standard=base['profiles_catalog']['A'];cat={};changes=0
    for key,p in base['profiles_catalog'].items():
        t,m=p['total_tokens'],p['nql']
        corrected=p['family']=='A' and (t,m)!=(standard['total_tokens'],standard['nql'])
        if corrected:
            if mode=='total_token_jitter_1':t+=rng.choice([-1,1])
            elif mode=='miss_jitter_1':m+=rng.choice([-1,1])
            elif mode=='total_round_1k':t=round(t/1024)*1024
            elif mode=='miss_round_64':m=round(m/64)*64
        if (t,m)!=(p['total_tokens'],p['nql']):changes+=1
        cat[key]=make_profile(t,m,p['family'])
    seqs=[s.copy() for s in base['per_npu_sequences']]
    if mode.startswith('shuffle_seed'):
        order_seed=int(mode.removeprefix('shuffle_seed'))
        for n,s in enumerate(seqs):random.Random(order_seed+100003*n).shuffle(s)
    submit_seed=int(mode.removeprefix('submit_seed')) if mode.startswith('submit_seed') else 7
    metrics,_=simulate(cat,seqs,right=60000.,seed=submit_seed)
    return dict(mode=mode,changed_profile_count=changes,source_config_sha256=hashlib.sha256(raw).hexdigest(),
                submit_seed=submit_seed,queues_sha256=hashlib.sha256(json.dumps(seqs,separators=(',',':')).encode()).hexdigest(),metrics=metrics)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-input',action='store_true')
    args=parser.parse_args()
    if args.full_input:
        SOURCE=HERE/'aligned_sequential_60000ms_full_input_native_like_config.json'
        modes=['submit_seed17','submit_seed27','total_token_jitter_1','miss_jitter_1']
        prefix='native_like_full_input_controls'
    else:
        modes=['unaltered','submit_seed17','submit_seed27','total_token_jitter_1','miss_jitter_1','total_round_1k','miss_round_64','shuffle_seed7','shuffle_seed19']
        prefix='native_like_controls'
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for r in pool.map(work,modes):
            rows.append(r);print(r['mode'],r['metrics']['U'],r['metrics']['windows'][0],r['metrics']['all_warm_AB'],flush=True)
            (HERE/f'{prefix}_partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/f'{prefix}_60s.json').write_text(json.dumps(rows,indent=2)+'\n')
