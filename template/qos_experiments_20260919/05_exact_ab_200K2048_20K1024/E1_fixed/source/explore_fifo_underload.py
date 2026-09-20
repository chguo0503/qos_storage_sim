#!/usr/bin/env python3
"""Bounded raw-data search for FIFO HOL with fixed roles and mixed request decks."""
import argparse, ast, copy, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path
import sim
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import run_case, save_manifest, write_json
from run_shared_path_experiments import input_demand, logical_input_fingerprint, summarize_slo

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/fifo_underload_exploration_20260914'
IO=176*1024/2**30
CASES=[
 dict(name='s3_L19_S13',ssu=3,long=[200,2048],short=[32,2048],nlong=19,mode='separated'),
 dict(name='s3_L16_S16',ssu=3,long=[200,2048],short=[32,2048],nlong=16,mode='separated'),
 dict(name='s4_L20_S12',ssu=4,long=[200,2048],short=[32,1024],nlong=20,mode='separated'),
 dict(name='s4_L24_S8',ssu=4,long=[200,2048],short=[32,1024],nlong=24,mode='separated'),
 dict(name='s4_L18_S14',ssu=4,long=[200,2048],short=[32,1024],nlong=18,mode='separated'),
 dict(name='s4_L128_20_S12',ssu=4,long=[128,2048],short=[32,1024],nlong=20,mode='separated'),
 dict(name='s3_mix_1to4',ssu=3,long=[200,2048],short=[32,2048],short_per_long=4,mode='mixed'),
 dict(name='s3_mix_1to8',ssu=3,long=[200,2048],short=[32,2048],short_per_long=8,mode='mixed'),
 dict(name='s4_mix_1to2',ssu=4,long=[200,2048],short=[32,1024],short_per_long=2,mode='mixed'),
 dict(name='s4_mix_1to4',ssu=4,long=[200,2048],short=[32,1024],short_per_long=4,mode='mixed'),
 dict(name='s4_L15_S17_lowB',ssu=4,long=[200,4096],short=[32,1024],nlong=15,mode='separated'),
 dict(name='s3_L24_S8_lowB',ssu=3,long=[200,4096],short=[32,1024],nlong=24,mode='separated'),
]

def build(case,horizon,seed):
 table=ast.literal_eval((ROOT/'data').read_text())
 requests=[]; specifications={}
 for role,key in [('L',case['long']),('S',case['short'])]:
  b,c,ttft,d=table[tuple(key)]
  specifications[role]=dict(total_length_k=key[0],nql=key[1],compute_ms=c/1000,read_gib=d,B_gib_s=b)
 for n in range(32):
  rng=random.Random(seed+n*100003)
  roles=[]
  if case['mode']=='separated':
   role='L' if n<case['nlong'] else 'S'
   count=math.ceil(horizon/(8*specifications[role]['compute_ms']))+1
   roles=[role]*count
  else:
   deck=['L']+['S']*case['short_per_long']
   cycles=math.ceil(horizon/sum(8*specifications[x]['compute_ms'] for x in deck))+1
   roles=deck*cycles; rng.shuffle(roles)
  for g,role in enumerate(roles):
   p=specifications[role];rid=n*1000000+g
   blocks=(p['total_length_k']*1024-p['nql'])//128
   layer=tuple((sim.block_ring_hash_disk_id(rid,j,case['ssu']),IO) for j in range(blocks))
   assert math.isclose(blocks*IO,p['read_gib'],rel_tol=0,abs_tol=1e-12)
   load=dict(request_id=rid,npu_id=n,generation=g,original_request_id=rid,
      seq_len_k=p['total_length_k'],nql=p['nql'],role=role,
      category=sim.classify_request(p['total_length_k'],p['nql']),
      per_layer_us=p['compute_ms']*1000,per_layer_kv_gb=blocks*IO,
      required_bw_input_gbps=p['B_gib_s'],arrival_time=0.,arrival_ms=0.,initial=True,
      constructed_profile=False,profile_construction={'method':'direct_data_row'},
      original_compute_us=p['compute_ms']*1000,padding_gib_per_layer=0.)
   requests.append(ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,)))
 requests=tuple(requests)
 vectors={}; maxima=[[0.]*case['ssu'] for _ in range(32)]
 for r in requests:
  counts=Counter(s for s,_ in r.placement[0]);c=r.load['per_layer_us']/1e6
  v=[counts[s]*IO/c for s in range(case['ssu'])];vectors[r.request_id]=v
  maxima[r.npu_id]=[max(a,b) for a,b in zip(maxima[r.npu_id],v)]
 static=[sum(v[s] for v in maxima) for s in range(case['ssu'])]
 meta=dict(experiment='fifo_underload_exploration_20260914',case=case,profiles=specifications,
  num_npu=32,num_ssu=case['ssu'],n_layers=8,seed=seed,order=case['mode'],
  equal_176kib_blocks=True,layout='block_ring_hash',placement_virtual_nodes_per_ssu=256,
  input_fingerprint=continuous_batch_input_fingerprint(requests),
  logical_input_fingerprint=logical_input_fingerprint(requests),input_demand=input_demand(requests,32,case['ssu']),
  per_ssu_static_upper_bound_gib_s=static,static_underload_all_request_combinations=max(static)<=40,
  workload_scope='Controlled repeated raw profiles for screening; unique request IDs but repeated length/NQL combinations')
 return requests,meta,vectors

def demand_audit(summary,vectors,num_ssu,left,right):
 events=defaultdict(lambda:[0.]*num_ssu)
 for q in summary['request_metrics']:
  a,b=q['admission_time_ms'],q['completion_time_ms'];v=vectors[q['request_id']]
  for s in range(num_ssu):events[a][s]+=v[s];events[b][s]-=v[s]
 cur=[0.]*num_ssu;peak=[0.]*num_ssu;area=[0.]*num_ssu;over=0.;prev=0.
 for t,d in sorted(events.items()):
  dur=max(0.,min(t,right)-max(prev,left))
  if dur:
   for s in range(num_ssu):peak[s]=max(peak[s],cur[s]);area[s]+=dur*cur[s]
   if max(cur)>40+1e-8:over+=dur
  cur=[a+b for a,b in zip(cur,d)];prev=t
 return dict(peak_nominal_gib_s_by_ssu=peak,mean_nominal_gib_s_by_ssu=[v/(right-left) for v in area],
             any_ssu_overload_ms=over,any_ssu_overload_fraction=over/(right-left),
             underloaded_every_admission_interval=max(peak)<=40+1e-8,
             definition='sum current admitted requests per-layer bytes on SSU / own layer compute time; layer0 prefetch not counted twice')

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True)
 p.add_argument('--seed',type=int,default=7);p.add_argument('--horizon-ms',type=float,default=1000.)
 p.add_argument('--window',type=float,nargs=2,default=[200.,800.]);p.add_argument('--strategy',choices=['baseline','once'],default='baseline')
 p.add_argument('--stage',default='screen');a=p.parse_args();case=CASES[a.case]
 dest=OUT/a.stage/f"{case['name']}_seed{a.seed}_{a.strategy}";dest.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json.gz').exists():raise FileExistsError(dest)
 started=time.perf_counter();requests,meta,vectors=build(case,a.horizon_ms,a.seed)
 save_manifest(dest/'manifest.json.gz',requests,meta);write_json(dest/'metadata.json',meta)
 print(json.dumps({'event':'start','name':case['name'],'strategy':a.strategy,'request_count':len(requests),
    'static_max':meta['per_ssu_static_upper_bound_gib_s'],'average_ssu':meta['input_demand']['per_ssu_gib_s']}),flush=True)
 result=run_case(requests,meta,strategy=a.strategy,assignment='fixed',windows=[tuple(a.window)])
 audit=demand_audit(result['summary'],vectors,case['ssu'],*a.window)
 result['demand_audit']=audit;result['exploration_case']=case
 write_json(dest/'result.json.gz',result)
 w=result['windows'][0];slo=result['slo']['window_admissions']['admission']
 row=dict(case_index=a.case,name=case['name'],seed=a.seed,strategy=a.strategy,mode=case['mode'],num_ssu=case['ssu'],
  window_ms=a.window,U_percent=100*w['mean_npu_utilization'],
  short_U_percent=100*w['by_role']['S']['active_compute_fraction'],long_U_percent=100*w['by_role']['L']['active_compute_fraction'],
  short_active_share=w['by_role']['S']['active_ms']/(32*(a.window[1]-a.window[0])),
  short_stall_card_ms=w['by_role']['S']['exposed_stall_ms'],
  slo_1p5_percent=100*slo['rate'],slo_count=slo['count'],slo_passed=slo['passed'],
  static_upper_gib_s=meta['per_ssu_static_upper_bound_gib_s'],
  input_average_gib_s=meta['input_demand']['per_ssu_gib_s'],demand_audit=audit,
  all_active=w['all_npus_active_whole_window'],all_invariants_passed=all(result['summary']['invariants'].values()),
  wall_seconds=time.perf_counter()-started)
 write_json(dest/'metrics.json',row);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
