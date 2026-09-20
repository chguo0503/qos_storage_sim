#!/usr/bin/env python3
"""Independently recompute six-run window checks from saved final outputs."""
from pathlib import Path
import gzip,hashlib,json,math

HERE=Path(__file__).resolve().parent
def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def overlap(a,z,l,r):return max(0.,min(z,r)-max(a,l))

def main():
    records=[]
    for seed in (7,19,43):
        digest=sha(HERE/'inputs'/f'near35_seed{seed}.json.gz')
        m=read(HERE/'inputs'/f'near35_seed{seed}.json.gz')['metadata']
        assert m['scenario_candidate']=='near35' and m['static_upper_bound_passes'] is False
        assert m['per_length_miss_counts']=={'256':1,'1024':1,'2048':3,'4096':2}
        assert len(m['profiles'])==24 and m['compute_scale_actual']==1.0
        assert abs(m['time_weighted_fleet_nominal_gib_s']/3-35)<0.01
        for strategy in ('baseline','once'):
            directory=HERE/'runs'/f'{strategy}_seed{seed}'
            cmd=read(directory/'command.json');r=read(directory/'result.json.gz')
            assert cmd['status']=='complete' and cmd['completed_simulation']
            assert cmd['core_unchanged'] and cmd['extension_unchanged'] and all(cmd['checks'].values())
            assert digest==sha(directory/'manifest.json.gz')==cmd['manifest_sha256']
            assert sha(directory/'result.json.gz')==cmd['result_sha256']
            assert cmd['observed_blocks']==cmd['expected_blocks']==9331712
            assert len(r['summary']['request_metrics'])==1344
            bw=r['bandwidth_2ms'];step=bw['step_ms']
            assert step==2 and (bw['start_ms'],bw['end_ms'])==(2000,4000)
            ssd_peak=max(max(row) for row in bw['ssd_GiB_s'])
            hbm_peak=max(max(row) for row in bw['npu_hbm_GiB_s'])
            assert ssd_peak<=40+1e-7 and hbm_peak<=50+1e-7
            for b in range(1000):
                assert abs(sum(row[b] for row in bw['ssd_GiB_s'])-sum(row[b] for row in bw['npu_ssd_GiB_s']))<1e-6
            windows=[]
            for i,(left,right) in enumerate(((2000,4000),(2000,6000))):
                cohort=[q for q in r['summary']['request_metrics'] if left<=q['admission_time_ms']<right]
                assert cohort and all(math.isfinite(q['completion_time_ms']) for q in cohort)
                ratios=[(q['completion_time_ms']-q['admission_time_ms'])/q['own_compute_ms'] for q in cohort]
                stalls=[q['io_stall_ms'] for q in cohort]
                busy=sum(overlap(layer['compute_start_ms'],layer['compute_end_ms'],left,right)
                    for batch in r['summary']['microbatch_metrics'] for layer in batch['layer_metrics'])
                util=100*busy/(32*(right-left));a=r['analysis'][i]
                assert abs(util-a['U_percent'])<1e-8 and a['all_npus_active']
                for disk in range(3):
                    dm=a['demand'];segs=dm['segments'];duration=right-left
                    mean=sum((z-x)*rates[disk] for x,z,*rates in segs)/duration
                    over=sum((z-x)*(rates[disk]>40+1e-8) for x,z,*rates in segs)*100/duration
                    assert abs(mean-dm['per_disk_mean_GiB_s'][disk])<1e-7
                    assert abs(over-dm['per_disk_overload_percent'][disk])<1e-7
                windows.append(dict(start_ms=left,end_ms=right,U_percent=util,
                    cohort_count=len(cohort),completed_after_window=sum(q['completion_time_ms']>right for q in cohort),
                    min_TTFT_ratio_raw=min(ratios),max_TTFT_ratio_raw=max(ratios),
                    max_io_stall_ms=max(stalls),requests_with_stall_over_1e_9_ms=sum(s>1e-9 for s in stalls),
                    SLO1_percent=100*sum(q['completion_time_ms']-q['admission_time_ms']<=q['own_compute_ms']+1e-9 for q in cohort)/len(cohort),
                    SLO1p5_percent=100*sum(x<=1.5+1e-12 for x in ratios)/len(ratios),
                    per_ssu_demand_mean_GiB_s=a['demand']['per_disk_mean_GiB_s'],
                    per_ssu_overload_percent=a['demand']['per_disk_overload_percent'],
                    per_ssu_demand_max_GiB_s=a['demand']['per_disk_max_GiB_s'],
                    per_ssu_actual_mean_GiB_s=a['SSD_GiB_s']))
            records.append(dict(seed=seed,strategy=strategy,manifest_sha256=digest,
                result_sha256=sha(directory/'result.json.gz'),completed_requests=1344,completed_blocks=cmd['observed_blocks'],
                full_U_percent=r['analysis'][2]['U_percent'],full_SLO1p5_percent=r['analysis'][2]['slo']['percent'],
                physical_ssd_peak_2ms_GiB_s=ssd_peak,physical_hbm_peak_2ms_GiB_s=hbm_peak,
                checks=cmd['checks'],windows=windows))
    report=dict(status='passed',run_count=len(records),
        static_per_ssu_upper_bound_GiB_s=m['static_per_ssu_upper_bound_gib_s'],
        raw_data_no_scaling=True,full_population_each_run=1344,seed_pair_manifests_identical=True,
        original_sources_unchanged_during_execution=True,all_runs_complete=True,records=records)
    (HERE/'six_run_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
