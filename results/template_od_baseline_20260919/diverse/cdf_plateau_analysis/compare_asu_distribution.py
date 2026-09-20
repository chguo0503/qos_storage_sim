"""Compare ASU and OD output distributions on the frozen full workload."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
PROJECT=HERE.parents[3]
GAP=(1.8142302265732846,3.588658678248402)

def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)

def main():
    audit=read(STUDY/'input_audit.json')['cases']
    records=[];all_rows={};sources={}
    for seed in (7,19,43):
        manifest=read(STUDY/'inputs'/f'full_seed{seed}.json.gz')
        profiles={q['request_id']:q['load'] for q in manifest['requests']}
        paths={'asu_baseline':PROJECT/next(c['reference_asu'] for c in audit if c['scenario']=='full' and c['seed']==seed),
               'od_baseline':STUDY/'runs'/f'full_od_baseline_seed{seed}/result.json.gz'}
        for policy,path in paths.items():
            raw=read(path);s=raw['summary'];assert raw['input_fingerprint']==manifest['input_fingerprint']
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            batches={b['member_request_ids'][0]:b for b in s['microbatch_metrics']}
            rows={}
            for q in s['request_metrics']:
                p=profiles[q['request_id']];ls=batches[q['request_id']]['layer_metrics']
                r=dict(policy=policy,seed=seed,request_id=q['request_id'],npu=q['npu_id'],
                       total_K=p['seq_len_k'],miss=p['nql'],V_layer_MiB=1024*p['per_layer_kv_gb'],
                       C_layer_ms=p['per_layer_us']/1000,
                       admission_ms=q['admission_time_ms'],completion_ms=q['completion_time_ms'],
                       ratio=(q['completion_time_ms']-q['admission_time_ms'])/q['own_compute_ms'],
                       pure_compute_ms=q['own_compute_ms'],TTFT_ms=q['completion_time_ms']-q['admission_time_ms'],
                       internal_read_latency_ms=float(np.mean([l['io_ready_time_ms']-l['io_start_time_ms'] for l in ls[1:]])),
                       internal_stall_ms=sum(l['io_barrier_wait_ms'] for l in ls[1:]),
                       mean_block_ssd_queue_wait_ms=q['avg_ssd_queue_wait_ms'],
                       mean_block_link_queue_wait_ms=q['avg_npu_link_queue_wait_ms'])
                rows[r['request_id']]=r
                if 2000<=r['admission_ms']<4000:records.append(r)
            all_rows[seed,policy]=rows
    aggregate=[]
    for policy in ('asu_baseline','od_baseline'):
        for miss in (256,1024,2048,4096):
            pp=[r for r in records if r['policy']==policy and r['miss']==miss]
            aggregate.append(dict(policy=policy,miss=miss,count=len(pp),ratio_min=min(r['ratio'] for r in pp),
                                  ratio_max=max(r['ratio'] for r in pp),gap_count=sum(GAP[0]<r['ratio']<GAP[1] for r in pp)))
    perprofile=[]
    for length,miss in sorted({(r['total_K'],r['miss']) for r in records}):
        for policy in ('asu_baseline','od_baseline'):
            pp=[r for r in records if (r['total_K'],r['miss'],r['policy'])==(length,miss,policy)]
            perprofile.append(dict(total_K=length,miss=miss,policy=policy,count=len(pp),
                                   C_layer_ms=pp[0]['C_layer_ms'],V_layer_MiB=pp[0]['V_layer_MiB'],
                                   ratio_median=float(np.median([r['ratio'] for r in pp])),
                                   internal_read_latency_mean_ms=float(np.mean([r['internal_read_latency_ms'] for r in pp])),
                                   gap_count=sum(GAP[0]<r['ratio']<GAP[1] for r in pp)))
    intersections=[]
    for seed in (7,19,43):
        a=all_rows[seed,'asu_baseline'];o=all_rows[seed,'od_baseline']
        assert a.keys()==o.keys()
        common=[rid for rid in a if 2000<=a[rid]['admission_ms']<4000 and 2000<=o[rid]['admission_ms']<4000]
        intersections.append(dict(seed=seed,common_warm_ids=len(common),
                                  asu_gap_count=sum(GAP[0]<a[rid]['ratio']<GAP[1] for rid in common),
                                  od_gap_count=sum(GAP[0]<o[rid]['ratio']<GAP[1] for rid in common)))
    paired_examples=[all_rows[seed,policy][rid] for seed,rid in ((43,16000007),(43,12000008))
                     for policy in ('asu_baseline','od_baseline')]
    d=dict(cohort='warm admissions [2,4)s; completion followed without truncation',
           gap_open_interval=list(GAP),by_miss=aggregate,by_profile=perprofile,
           matched_warm_id_intersections=intersections,paired_examples=paired_examples,
           source_sha256=sources,no_simulation_run=True,
           note='Profile medians and counts pool seeds; original CDF uses equal mean of per-seed ECDFs. Input IDs match globally, warm selected IDs need not match.')
    (HERE/'asu_od_distribution.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'asu_od_by_profile.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(perprofile[0]));w.writeheader();w.writerows(perprofile)
    print(json.dumps({k:d[k] for k in ('by_miss','matched_warm_id_intersections','paired_examples')},indent=2))

if __name__=='__main__':main()
