"""Postprocess only: compare two completed policies on one immutable input."""
from pathlib import Path
import csv,gzip,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'transition'))
import metrics
from inputs.manifest import load_manifest,write_json
NAME='abb_cal2_bootsteady12'

def main():
 reqs,meta=load_manifest(HERE/'inputs'/f'{NAME}.json.gz');out={'name':NAME,'metadata':meta,'policies':{}};rows=[]
 for policy in ('od_baseline','once'):
  directory=HERE/'runs'/(NAME if policy=='od_baseline' else NAME+'_'+policy)
  raw=json.load(gzip.open(directory/'result.json.gz','rt'))['summary']
  data={'makespan_ms':raw['makespan_ms'],'windows':[]}
  data['first_card_drains_ms']=min(max(r['completion_time_ms'] for r in raw['request_metrics'] if r['npu_id']==n) for n in range(32))
  for a,b in [(2000.,4000.),(4000.,8000.),(8000.,12000.),(12000.,16000.),(16000.,18000.),(2000.,18000.),(12000.,20000.),(0.,raw['makespan_ms'])]:
   if b>raw['makespan_ms']:continue
   m=metrics.summarize(raw,reqs,a,b,full=a==0);m['measurement_definition']['own_compute']='8 times frozen C; calibrated tail B adds actual synthetic compute'
   data['windows'].append(m)
   rows.append(dict(policy=policy,window='full' if a==0 else f'[{a/1000:g},{b/1000:g})',U_percent=m['U_percent'],
    SLO_1p5_percent=m['slo']['percent'],cohort_count=m['slo']['count'],all32active=m['all_npus_active'],mixed_cards=m['role_and_stall']['npus_with_A_and_B_compute'],
    strict_underload=m['demand']['strict_underload_all_disks'],any_disk_overload_percent=m['demand']['any_disk_overload_percent'],
    max_disk_demand_GiB_s=max(m['demand']['per_disk_max_GiB_s'])))
  out['policies'][policy]=data
 write_json(HERE/'comparison.json',out)
 with (HERE/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows))
if __name__=='__main__':main()
