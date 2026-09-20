"""Formal replay: imports no planning module and installs no runtime hooks."""
from pathlib import Path
import argparse,ast,bisect,gzip,hashlib,json,math,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'transition'))
import metrics
from inputs.manifest import load_manifest,write_json
from simulator.api import run_simulation

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--strategy',choices=['od_baseline','once'],default='od_baseline');a=ap.parse_args()
 manifest=HERE/'inputs'/f'{a.name}.json.gz';reqs,meta=load_manifest(manifest);out=HERE/'formal'/f'{a.name}_{a.strategy}';out.mkdir(parents=True,exist_ok=False)
 sources=list((ROOT/'simulator').rglob('*.py'))+[ROOT/'data',ROOT/'inputs/manifest.py',HERE.parent/'transition/metrics.py',Path(__file__),
  HERE/'planning'/a.name/'generator_source.py',HERE/'planning'/a.name/'planning_records.json',
  HERE/'planning'/a.name/'command.json',HERE/'planning'/a.name/'PLANNING_ONLY_result.json.gz',manifest]
 before={str(p.relative_to(ROOT)):digest(p) for p in sources}
 plan_command=json.loads((HERE/'planning'/a.name/'command.json').read_text())
 assert plan_command['status']=='complete' and plan_command['manifest_sha256']==digest(manifest)
 assert digest(ROOT/'data')==meta['data_sha256']
 assert all(before[k]==v for k,v in meta['planning_source_sha256'].items())
 assert math.isfinite(meta['period_ms']) and meta['period_ms']>0
 assert all(math.isfinite(q.load['per_layer_us']) and q.load['per_layer_us']>0 for q in reqs)
 if meta.get('input_mode')=='interpolated':
  table=ast.literal_eval((ROOT/'data').read_text());hit=meta['fixed_B_hit_tokens']
  def interpolate(m):
   length=(hit+m)/1024;lo,hi=(96,128) if length<128 else (128,160);x=(length-lo)/(hi-lo);y=(m-2048)/2048
   assert 0<=x<=1 and 0<=y<=1
   return sum(w*table[L,M][1] for L,M,w in ((lo,2048,(1-x)*(1-y)),(lo,4096,(1-x)*y),(hi,2048,x*(1-y)),(hi,4096,x*y)))
  costs=[interpolate(m) for m in range(2048,4097)]
  for q in reqs:
   l=q.load
   if l['role']=='B':
    assert l['total_tokens']-l['nql']==hit and isinstance(l['nql'],int)
    assert l['per_layer_us']==interpolate(l['nql'])
    assert len(q.placement[0])==hit//128 and all(v==176*1024/2**30 for d,v in q.placement[0])
    if l['position']==2:
     target=l['target_unquantized_C_us'];j=bisect.bisect_left(costs,target)
     best=min(range(max(0,j-1),min(len(costs),j+1)),key=lambda z:abs(costs[z]-target))
     assert l['nql']==best+2048
  write_json(out/'interpolation_audit.json',dict(all_profiles_exact_data_bilinear=True,all_integer_misses_nearest_target=True,fixed_hit_and_read_bytes=True,data_sha256=digest(ROOT/'data'))) 
 command=dict(status='running',name=a.name,strategy=a.strategy,runtime_hooks=False,source_and_artifact_sha256=before,
  manifest_sha256=digest(manifest),depth_per_npu_per_ssu=256 if a.strategy=='od_baseline' else None)
 write_json(out/'command.json',command);start=time.perf_counter()
 result=run_simulation(reqs,strategy=a.strategy,num_npu=32,num_ssu=3,n_layers=8,seed=7,od_queue_depth_per_ssu=8192 if a.strategy=='od_baseline' else None)
 assert before=={str(p.relative_to(ROOT)):digest(p) for p in sources}
 write_json(out/'result.json.gz',result);raw=result['summary'];byid={q.request_id:q for q in reqs}
 mb={(m['npu_id'],byid[m['member_request_ids'][0]].load['cycle'],byid[m['member_request_ids'][0]].load['position']):m for m in raw['microbatch_metrics']}
 cycles=[]
 for k in range(meta['cycles']):
  tails=[mb[n,k,2] for n in range(32)];ss=[m['layer_metrics'][0]['compute_start_ms'] for m in tails];ee=[m['completion_time_ms'] for m in tails]
  cc=[m['layer_metrics'][0]['compute_duration_ms'] for m in tails]
  for n,m in enumerate(tails):
   q=byid[m['member_request_ids'][0]];assert all(l['compute_duration_ms']==q.load['per_layer_us']/1000 for l in m['layer_metrics'])
  waits=[l['io_barrier_wait_ms'] for n in range(32) for p in (1,2) for l in mb[n,k,p]['layer_metrics'][1:]]
  row=dict(cycle=k,target_E_ms=(k+1)*meta['period_ms'],C_min_ms=min(cc),C_max_ms=max(cc),C_sum_ms=sum(cc),
   base_C_ms=meta['original_B_C_ms'],added_compute_card_ms=8*sum(c-meta['original_B_C_ms'] for c in cc),
   signed_compute_change_card_ms=8*sum(c-meta['original_B_C_ms'] for c in cc),
   positive_compute_change_card_ms=8*sum(max(0.,c-meta['original_B_C_ms']) for c in cc),
   negative_compute_change_card_ms=8*sum(min(0.,c-meta['original_B_C_ms']) for c in cc),
   absolute_compute_adjustment_card_ms=8*sum(abs(c-meta['original_B_C_ms']) for c in cc),
   reduced_compute_request_count=sum(c<meta['original_B_C_ms'] for c in cc),
   all_C_at_least_base=min(cc)>=meta['original_B_C_ms'],start_spread_ms=max(ss)-min(ss),causal_delta_lt_minC=max(ss)-min(ss)<min(cc),
   end_spread_ms=max(ee)-min(ee),max_E_error_ms=max(abs(e-(k+1)*meta['period_ms']) for e in ee),
   B_internal_stall_card_ms=sum(waits),B_internal_stall_max_ms=max(waits))
  if k+1<meta['cycles']:
   margins=[ee[n]-mb[n,k+1,0]['layer_metrics'][0]['io_ready_time_ms'] for n in range(32)];row['next_A0_min_prefetch_margin_ms']=min(margins)
  cycles.append(row)
  if a.strategy=='od_baseline':
   assert row['causal_delta_lt_minC'] and row['B_internal_stall_max_ms']<1e-7
   assert row['max_E_error_ms']<=meta.get('quantization_end_bound_ms',0)+1e-7
   assert row.get('next_A0_min_prefetch_margin_ms',0)>=-1e-7
 analysis=dict(cycles=cycles,windows=[],makespan_ms=raw['makespan_ms'],
  first_card_drains_ms=min(max(m['completion_time_ms'] for m in raw['request_metrics'] if m['npu_id']==n) for n in range(32)),
  all_native_invariants=all(raw['invariants'].values()),synthetic_OD_tailored_input=True)
 for left,right in [(2000.,4000.),(8000.,12000.),(12000.,20000.),(20000.,40000.),(40000.,60000.),(60000.,78000.)]:
  if right<=raw['makespan_ms']+1e-7:
   m=metrics.summarize(raw,reqs,left,right);m['measurement_definition']['own_compute']='8 times frozen request C, including all calibrated tail B computation'
   analysis['windows'].append(m)
 if a.strategy=='od_baseline':
  planned=json.load(gzip.open(HERE/'planning'/a.name/'PLANNING_ONLY_result.json.gz','rt'))['summary']
  parity=dict(all_request_metrics_exact=raw['request_metrics']==planned['request_metrics'],
   all_microbatch_layer_metrics_exact=raw['microbatch_metrics']==planned['microbatch_metrics'],source_identical=True,
   independent_clean_process=True,no_runtime_hook=True,manifest_sha256=digest(manifest),max_layer_error_ms={})
  for key in ('compute_start_ms','compute_end_ms','compute_duration_ms','io_start_time_ms','io_ready_time_ms','io_barrier_wait_ms'):
   parity['max_layer_error_ms'][key]=max(abs(l[key]-pl[key]) for b,pb in zip(raw['microbatch_metrics'],planned['microbatch_metrics']) for l,pl in zip(b['layer_metrics'],pb['layer_metrics']))
  write_json(out/'parity.json',parity);assert parity['all_request_metrics_exact'] and parity['all_microbatch_layer_metrics_exact']
 write_json(out/'analysis.json',analysis)
 (out/'clean_replay_source.py').write_bytes(Path(__file__).read_bytes())
 command.update(status='complete',source_and_artifacts_unchanged=True,wall_seconds=time.perf_counter()-start)
 write_json(out/'command.json',command)
 print(json.dumps(dict(name=a.name,strategy=a.strategy,cycles=cycles,
  windows=[dict(start_ms=w['start_ms'],end_ms=w['end_ms'],U_percent=w['U_percent'],SLO=w['slo']['percent'],all32active=w['all_npus_active'],mixed_cards=w['role_and_stall']['npus_with_A_and_B_compute']) for w in analysis['windows']])),flush=True)
if __name__=='__main__':main()
