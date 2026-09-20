"""Export completed OD results using the template's warm admission cohort."""
from pathlib import Path
import csv,gzip,hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
SCENARIOS=('full','semi','near35');SEEDS=(7,19,43)
def read(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return json.load(f)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 out=HERE/'data';out.mkdir(exist_ok=True)
 perseed=[];samples=[];pernpu=[];checks=[]
 for scenario in SCENARIOS:
  parity=read(HERE/'runs'/f'{scenario}_asu_baseline_seed7'/'parity.json')
  assert parity['passed'] and parity['same_input'],scenario
  for seed in SEEDS:
   run=HERE/'runs'/f'{scenario}_od_baseline_seed{seed}'
   command=read(run/'command.json');assert command['status']=='complete'
   assert command['source_unchanged'] and all(command['checks'].values())
   assert sha(run/'result.json.gz')==command['result_sha256']
   result=read(run/'result.json.gz');manifest=read(run/'manifest.json.gz')
   assert result['input_fingerprint']==manifest['input_fingerprint']==command['input_fingerprint']
   assert sha(run/'manifest.json.gz')==sha(HERE/'inputs'/f'{scenario}_seed{seed}.json.gz')
   byid={q['request_id']:q for q in manifest['requests']}
   summary=result['summary'];assert all(summary['invariants'].values())
   for i,a in enumerate(result['analysis']):
    window=('warm_2_4s','warm_2_6s','full_population')[i]
    cohort=summary['request_metrics'] if i==2 else [q for q in summary['request_metrics'] if a['start_ms']<=q['admission_time_ms']<a['end_ms']]
    ttft=np.array([q['completion_time_ms']-q['admission_time_ms'] for q in cohort]);base=np.array([q['own_compute_ms'] for q in cohort])
    ratios=ttft/base;ratios[np.abs(ttft-base)<=1e-9]=1.;ratios[(ttft>1.5*base)&(ttft<=1.5*base+1e-9)]=1.5
    slo=float(np.mean(ratios<=1.5)*100);assert abs(slo-a['slo']['percent'])<1e-8
    row=dict(scenario=scenario,policy='od_baseline',seed=seed,window=window,NPU_utilization_percent=a['U_percent'],SLO1_percent=float(np.mean(ratios<=1)*100),SLO1_5_percent=slo,requests=len(cohort),TTFT_mean_ms=float(np.mean(ttft)),TTFT_p99_ms=float(np.percentile(ttft,99)),ratio_max=float(ratios.max()),completed_after_window=a['completed_after_window'],all_npus_active=a['all_npus_active'],any_disk_over40_percent=a['demand']['any_disk_overload_percent'])
    for d in range(3):
     row[f'SSU{d}_mean_demand_GiB_s']=a['demand']['per_disk_mean_GiB_s'][d];row[f'SSU{d}_over40_percent']=a['demand']['per_disk_overload_percent'][d];row[f'SSU{d}_max_demand_GiB_s']=a['demand']['per_disk_max_GiB_s'][d];row[f'SSU{d}_mean_supply_GiB_s']=a['SSD_GiB_s'][d]
    perseed.append(row)
    if i==0:
     for q,t,x in zip(cohort,ttft,ratios):
      inp=byid[q['request_id']]
      samples.append(dict(scenario=scenario,policy='od_baseline',seed=seed,request_id=q['request_id'],npu=inp['npu_id'],total_K=inp['load']['seq_len_k'],miss=inp['load']['nql'],category=inp['load']['category'],admission_ms=q['admission_time_ms'],completion_ms=q['completion_time_ms'],TTFT_ms=float(t),SLO_base_ms=q['own_compute_ms'],SLO_multiple=float(x)))
     pernpu.extend(dict(scenario=scenario,policy='od_baseline',seed=seed,npu=n,U_percent=u,SSD_supply_GiB_s=sum(result['warm_ssd_GiB_s_by_ssu_npu'][d][n] for d in range(3))) for n,u in enumerate(a['per_npu_U_percent']))
   checks.append(dict(scenario=scenario,seed=seed,input_fingerprint=result['input_fingerprint'],manifest_sha256=sha(run/'manifest.json.gz'),result_sha256=sha(run/'result.json.gz'),blocks=summary['completed_blocks'],requests=summary['request_count'],parity_reference_seed7_passed=True,source_unchanged=True,all_checks=True))
 macro=[];cdf=[]
 for scenario in SCENARIOS:
  for window in ('warm_2_4s','warm_2_6s','full_population'):
   part=[r for r in perseed if r['scenario']==scenario and r['window']==window]
   row=dict(scenario=scenario,policy='od_baseline',window=window,seed_count=3,requests=sum(r['requests'] for r in part))
   for key in ('NPU_utilization_percent','SLO1_percent','SLO1_5_percent','TTFT_mean_ms','any_disk_over40_percent',*(f'SSU{d}_{m}' for d in range(3) for m in ('mean_demand_GiB_s','mean_supply_GiB_s','over40_percent'))):row[key]=float(np.mean([r[key] for r in part]))
   macro.append(row)
  pp=[r for r in samples if r['scenario']==scenario];x=np.unique([r['SLO_multiple'] for r in pp]+[0.,.5,1.,1.5,2.])
  y=np.mean([np.searchsorted(np.sort([r['SLO_multiple'] for r in pp if r['seed']==s]),x,side='right')/sum(r['seed']==s for r in pp) for s in SEEDS],axis=0)*100
  expected=next(r['SLO1_5_percent'] for r in macro if r['scenario']==scenario and r['window']=='warm_2_4s')
  assert abs(y[np.where(x==1.5)[0][0]]-expected)<1e-8
  cdf.extend(dict(scenario=scenario,policy='od_baseline',SLO_multiple=float(xx),CDF_percent=float(yy)) for xx,yy in zip(x,y))
 for name,rows in [('od_per_seed_metrics',perseed),('od_summary',macro),('od_request_samples',samples),('od_cdf_points',cdf),('od_per_npu',pernpu)]:write_csv(out/(name+'.csv'),rows)
 (out/'od_metrics.json').write_text(json.dumps(dict(summary=macro,per_seed=perseed,checks=checks,cohort='admission [2,4)s; uncensored completion; seed7/19/43 equal-weight CDF'),indent=2)+'\n')
 (HERE/'validation.json').write_text(json.dumps(dict(status='complete',case_count=9,parity_case_count=3,checks=checks),indent=2)+'\n')
 print(json.dumps([r for r in macro if r['window']=='warm_2_4s'],indent=2))
if __name__=='__main__':main()
