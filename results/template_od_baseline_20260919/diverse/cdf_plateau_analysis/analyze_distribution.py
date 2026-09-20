"""Read-only decomposition of the full/OD warm-admission CDF plateau."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent

def csvout(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    path=STUDY/'plot_request_samples.csv'
    samples=[r for r in csv.DictReader(path.open(encoding='utf-8-sig'))
             if r['regime']=='full' and r['policy']=='od_baseline']
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    seeds=(7,19,43)
    inputs={}
    for seed in seeds:
        p=STUDY/'inputs'/f'full_seed{seed}.json.gz'
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with gzip.open(p,'rt') as f:m=json.load(f)
        inputs[seed]={q['request_id']:q for q in m['requests']}
    for r in samples:
        r['seed']=int(r['seed']);r['request_id']=int(r['request_id'])
        q=inputs[r['seed']][r['request_id']]['load']
        r.update(total_K=q['seq_len_k'],miss=q['nql'],C_layer_ms=q['per_layer_us']/1000,
                 V_layer_GiB=q['per_layer_kv_gb'],B_GiB_s=q['per_layer_kv_gb']*1e6/q['per_layer_us'])
        for k in ('ratio','admission_ms','completion_ms','ideal_ms'):r[k]=float(r[k])
        r['ttft_ms']=r['completion_ms']-r['admission_ms']
        assert 2000<=r['admission_ms']<4000
        assert abs(r['ratio']-r['ttft_ms']/r['ideal_ms'])<1e-8
    def describe(part):
        x=[r['ratio'] for r in part]
        return dict(request_count=len(part),ratio_min=min(x),ratio_median=float(np.median(x)),ratio_max=max(x),
                    B_min_GiB_s=min(r['B_GiB_s'] for r in part),B_max_GiB_s=max(r['B_GiB_s'] for r in part),
                    count_le_1p5=sum(t<=1.5 for t in x),count_1p5_to_4=sum(1.5<t<=4 for t in x),count_gt_4=sum(t>4 for t in x))
    by_miss=[dict(miss=miss,**describe([r for r in samples if r['miss']==miss])) for miss in (256,1024,2048,4096)]
    by_profile=[]
    for length,miss in sorted({(r['total_K'],r['miss']) for r in samples}):
        part=[r for r in samples if (r['total_K'],r['miss'])==(length,miss)]
        by_profile.append(dict(total_K=length,miss=miss,C_layer_ms=part[0]['C_layer_ms'],V_layer_GiB=part[0]['V_layer_GiB'],**describe(part)))
    points=[]
    for x in (1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,6.,7.):
        yy=[sum(r['ratio']<=x for r in samples if r['seed']==s)/sum(r['seed']==s for r in samples) for s in seeds]
        points.append(dict(ratio_threshold=x,CDF_seed_equal_mean_percent=float(np.mean(yy)*100)))
    low=[r['ratio'] for r in samples if r['miss']>=1024]
    high=[r['ratio'] for r in samples if r['miss']==256]
    gap=(max(low),min(high))
    assert gap[0]<gap[1]
    assert not any(gap[0]<r['ratio']<gap[1] for r in samples)
    examples=[]
    for miss in (256,1024):
        part=[r for r in samples if r['total_K']==128 and r['miss']==miss]
        target=float(np.median([r['ratio'] for r in part]))
        r=min(part,key=lambda r:abs(r['ratio']-target))
        examples.append({k:r[k] for k in ('seed','request_id','npu_id','total_K','miss','C_layer_ms','V_layer_GiB','B_GiB_s','ideal_ms','ttft_ms','ratio')})
    output=dict(scenario='full',policy='od_baseline',cohort='admission [2,4)s, uncensored completion',
                seeds=list(seeds),request_count=len(samples),seed_counts={s:sum(r['seed']==s for r in samples) for s in seeds},
                cdf_aggregation='Equal mean of three per-seed ECDFs; profile counts are pooled counts.',
                plateau_open_interval=list(gap),plateau_request_count=0,by_miss=by_miss,by_profile=by_profile,
                cdf_thresholds=points,examples_128K=examples,
                sources_sha256=sources,no_new_simulation=True)
    (HERE/'distribution.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    csvout('by_miss.csv',by_miss);csvout('by_profile.csv',by_profile);csvout('cdf_thresholds.csv',points)
    print(json.dumps({k:output[k] for k in ('request_count','plateau_open_interval','by_miss','cdf_thresholds','examples_128K')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
