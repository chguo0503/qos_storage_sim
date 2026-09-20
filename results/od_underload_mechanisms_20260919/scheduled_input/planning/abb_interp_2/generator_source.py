"""OFFLINE OD-tailored input generator plus separate unhooked API replay.

Planning outcomes are never the experiment's measurements. Only immutable
manifest replays run with the unmodified native event handlers count.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,ast,bisect,gzip,hashlib,json,math,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'transition'))
import metrics
from inputs.manifest import load_manifest,save_manifest,write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core,sim
IO=176*1024/2**30

def hashes():
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'simulator').rglob('*.py'))}

TABLE=ast.literal_eval((ROOT/'data').read_text())
HIT=127744
BASE_MISS=3328

def interpolate(m):
 total=HIT+m;length=total/1024
 lo,hi=(96,128) if length<128 else (128,160)
 x=(length-lo)/(hi-lo);y=(m-2048)/2048
 assert 0<=x<=1 and 0<=y<=1
 anchors=[dict(length_k=L,miss=M,weight=w,C_us=TABLE[L,M][1]) for L,M,w in
          ((lo,2048,(1-x)*(1-y)),(lo,4096,(1-x)*y),(hi,2048,x*(1-y)),(hi,4096,x*y))]
 us=sum(a['weight']*a['C_us'] for a in anchors)
 return us,anchors

COST_US=[interpolate(m)[0] for m in range(2048,4097)]
assert all(a<b for a,b in zip(COST_US,COST_US[1:]))
QUANTIZATION_END_BOUND_MS=4*max(b-a for a,b in zip(COST_US,COST_US[1:]))/1000

def base_input(cycles):
 requests=[]
 for n in range(32):
  for k in range(cycles):
   for pos,miss in enumerate((256,BASE_MISS,BASE_MISS)):
    rid=n*1000000+k*3+pos;prefix=n*1000000+pos
    total=128*1024;blocks=(total-miss)//128;v=blocks*IO
    if pos==0:us=TABLE[128,256][1];anchors=[]
    else:us,anchors=interpolate(miss)
    placement=(tuple((sim.block_ring_hash_disk_id(prefix,b,3),IO) for b in range(blocks)),)
    load=dict(request_id=rid,npu_id=n,original_request_id=prefix,physical_prefix_id=prefix,
     cycle=k,position=pos,generation=k*3+pos,role='A' if pos==0 else 'B',total_tokens=total,
     seq_len_k=total/1024,nql=miss,category=sim.classify_request(total/1024,miss),per_layer_us=us,
     per_layer_kv_gb=v,required_bw_input_gbps=v*1e6/us,arrival_time=0.,arrival_ms=0.,initial=True,
     compute_interpolation_anchors=anchors,constructed_profile=pos!=0,physical_prefix_reused=True,
     timing_provenance='raw data A row' if pos==0 else 'data bilinear interpolation, not directly measured row',
     fixed_hit_tokens=total-miss)
    requests.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,placement))
 return tuple(requests)

def with_compute(q,target_us,E,start):
 assert math.isfinite(target_us) and COST_US[0]<=target_us<=COST_US[-1], 'target outside interpolation grid'
 idx=bisect.bisect_left(COST_US,target_us)
 best=min(range(max(0,idx-1),min(len(COST_US),idx+1)),key=lambda i:abs(COST_US[i]-target_us))
 m=2048+best;stored_us,anchors=interpolate(m);total=HIT+m
 load=dict(q.load);load.update(per_layer_us=stored_us,required_bw_input_gbps=load['per_layer_kv_gb']*1e6/stored_us,
  nql=m,total_tokens=total,seq_len_k=total/1024,category=sim.classify_request(total/1024,m),
  constructed_profile=True,timing_provenance='integer miss; total=fixed hit+miss; data bilinear C; offline OD-tailored',
  compute_interpolation_anchors=anchors,fixed_hit_tokens=HIT,
  target_unquantized_C_us=target_us,planned_completion_target_ms=E,planner_observed_B0_start_ms=start)
 assert load['category']==q.load['category'], 'changing category could change pre-L0 routing semantics'
 return core.ContinuousBatchRequest.from_normalized(q.request_id,q.npu_id,q.arrival_time_ms,load,q.placement)

def validate_cycles(raw,reqs,period):
 byid={q.request_id:q for q in reqs};mb={(m['npu_id'],byid[m['member_request_ids'][0]].load['cycle'],byid[m['member_request_ids'][0]].load['position']):m for m in raw['microbatch_metrics']}
 cycles=max(q.load['cycle'] for q in reqs)+1;rows=[]
 for k in range(cycles):
  tails=[mb[n,k,2] for n in range(32)];ss=[m['layer_metrics'][0]['compute_start_ms'] for m in tails];ee=[m['completion_time_ms'] for m in tails]
  cs=[m['layer_metrics'][0]['compute_duration_ms'] for m in tails]
  stall=[l['io_barrier_wait_ms'] for n in range(32) for pos in (1,2) for l in mb[n,k,pos]['layer_metrics'][1:]]
  row=dict(cycle=k,target_E_ms=(k+1)*period,min_C_ms=min(cs),max_C_ms=max(cs),start_spread_ms=max(ss)-min(ss),
   causal_delta_lt_minC=max(ss)-min(ss)<min(cs),max_target_error_ms=max(abs(e-(k+1)*period) for e in ee),
   tail_end_spread_ms=max(ee)-min(ee),B_internal_stall_card_ms=sum(stall),B_internal_stall_max_ms=max(stall),
   per_npu_C_ms=cs,per_npu_start_ms=ss,per_npu_end_ms=ee)
  if k+1<cycles:
   margins=[ee[n]-mb[n,k+1,0]['layer_metrics'][0]['io_ready_time_ms'] for n in range(32)]
   aa=[mb[n,k+1,0]['layer_metrics'][0]['compute_start_ms'] for n in range(32)]
   row.update(next_A_prefetch_min_margin_ms=min(margins),next_A_start_spread_ms=max(aa)-min(aa),next_A_all_prefetched=min(margins)>=-1e-7)
  rows.append(row)
 return rows

def plan(name,cycles,period):
 directory=HERE/'planning'/name;directory.mkdir(parents=True,exist_ok=False)
 assert math.isfinite(period) and period>0
 before=hashes();start=time.perf_counter();requests=base_input(cycles);frozen={};records=[]
 original=core._handle_compute_schedule
 def choose(ctx,n,t):
  npu=ctx.npus[n];batch=npu.active_batch
  if batch is not None and npu.compute_active is None and batch.compute_done_up_to==-1 and all(ctx.requests[r].io_ready[0] for r in batch.member_request_ids):
   assert len(batch.member_request_ids)==1
   state=ctx.requests[batch.member_request_ids[0]];q=state.manifest
   if q.load['position']==2:
    assert q.request_id not in frozen
    E=(q.load['cycle']+1)*period;stored_us=((E-t)/8)*1000
    assert math.isfinite(stored_us) and stored_us>0
    new=with_compute(q,stored_us,E,t)
    # Runtime state and its own copied manifest agree; miss and total change
    # together, keeping hit bytes and native physical placement fixed. Caller requests remain
    # immutable; API input-fingerprint protection is neither disabled nor patched.
    state.manifest=new;state.per_layer_compute_ms=new.load['per_layer_us']/1000
    frozen[q.request_id]=new
    records.append(dict(request_id=q.request_id,npu_id=n,cycle=q.load['cycle'],B0_start_ms=t,
      target_E_ms=E,target_unquantized_us=stored_us,stored_per_layer_us=new.load['per_layer_us'],actual_C_ms=state.per_layer_compute_ms,miss=new.load['nql'],total_tokens=new.load['total_tokens']))
  return original(ctx,n,t)
 write_json(directory/'command.json',dict(status='running',kind='PLANNING_ONLY_NOT_EXPERIMENT',name=name,cycles=cycles,period_ms=period,source_sha256=before))
 with patch.object(core,'_handle_compute_schedule',choose):
  result=run_simulation(requests,strategy='od_baseline',num_npu=32,num_ssu=3,n_layers=8,seed=7,od_queue_depth_per_ssu=8192)
 assert core._handle_compute_schedule is original and before==hashes()
 assert len(frozen)==32*cycles
 final=tuple(frozen.get(q.request_id,q) for q in requests)
 cycle_checks=validate_cycles(result['summary'],final,period)
 valid=all(r['causal_delta_lt_minC'] and r['B_internal_stall_max_ms']<1e-7 and r['max_target_error_ms']<=QUANTIZATION_END_BOUND_MS+1e-7 and r.get('next_A_all_prefetched',True) for r in cycle_checks)
 (directory/'generator_source.py').write_bytes(Path(__file__).read_bytes())
 write_json(directory/'PLANNING_ONLY_result.json.gz',result)
 write_json(directory/'planning_records.json',records)
 write_json(directory/'cycle_checks.json',cycle_checks)
 assert valid,'Planning construction failed; do not promote'
 meta=dict(name=name,cycles=cycles,num_npu=32,num_ssu=3,n_layers=8,sequence='ABB repeated',all_arrivals_ms=0,
  period_ms=period,E_rule='E_k=(k+1)*period fixed before planning',
  tail_C_rule='nearest integer miss on fixed-hit data bilinear C grid to (E_k-s_i,k)/8; replay stored_us/1000',
  generator='one offline OD planning pass, never used as the measured experiment',
  synthetic_compute=True,input_mode='interpolated',OD_tailored=True,fixed_B_hit_tokens=HIT,base_B_miss=BASE_MISS,quantization_end_bound_ms=QUANTIZATION_END_BOUND_MS,physical_prefix_reuse=True,
  physical_prefix_rule='npu*1000000+position; original native RingHash; no ID search',
  runtime_logical_id_rule='npu*1000000+cycle*3+position; unique',
  od_queue_depth_per_ssu=8192,planning_source_sha256=before,
  data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
  original_B_C_ms=interpolate(BASE_MISS)[0]/1000)
 save_manifest(HERE/'inputs'/f'{name}.json.gz',final,meta)
 write_json(directory/'command.json',dict(status='complete',kind='PLANNING_ONLY_NOT_EXPERIMENT',name=name,cycles=cycles,period_ms=period,
  source_sha256=before,source_unchanged=True,hook_restored=True,wall_seconds=time.perf_counter()-start,conditions_passed=valid,
  manifest_sha256=hashlib.sha256((HERE/'inputs'/f'{name}.json.gz').read_bytes()).hexdigest()))
 print(json.dumps(dict(planning_complete=name,conditions_passed=valid,cycles=cycles,wall_seconds=time.perf_counter()-start)),flush=True)

def replay(name,strategy):
 requests,meta=load_manifest(HERE/'inputs'/f'{name}.json.gz');directory=HERE/'runs'/f'{name}_{strategy}';directory.mkdir(parents=True,exist_ok=False)
 before=hashes();start=time.perf_counter();assert before==meta['planning_source_sha256']
 command=dict(status='running',name=name,strategy=strategy,source_sha256=before,runtime_hooks=False,
  manifest_sha256=hashlib.sha256((HERE/'inputs'/f'{name}.json.gz').read_bytes()).hexdigest())
 write_json(directory/'command.json',command)
 result=run_simulation(requests,strategy=strategy,num_npu=32,num_ssu=3,n_layers=8,seed=7,od_queue_depth_per_ssu=8192 if strategy=='od_baseline' else None)
 assert before==hashes();write_json(directory/'result.json.gz',result)
 raw=result['summary'];analysis=dict(makespan_ms=raw['makespan_ms'],cycles=validate_cycles(raw,requests,meta['period_ms']),windows=[])
 for a,b in [(2000.,4000.),(8000.,12000.),(12000.,20000.),(20000.,40000.),(40000.,60000.),(60000.,78000.)]:
  if b<=raw['makespan_ms']+1e-7:
   row=metrics.summarize(raw,requests,a,b);row['measurement_definition']['own_compute']='8 times frozen request C; tail B synthetic and OD-tailored'
   analysis['windows'].append(row)
 if strategy=='od_baseline':
  planned=json.load(gzip.open(HERE/'planning'/name/'PLANNING_ONLY_result.json.gz','rt'))['summary']
  reqequal=planned['request_metrics']==raw['request_metrics'];layerequal=planned['microbatch_metrics']==raw['microbatch_metrics']
  errors={key:0. for key in ('compute_start_ms','compute_end_ms','compute_duration_ms','io_start_time_ms','io_ready_time_ms','io_barrier_wait_ms')}
  for a,b in zip(planned['microbatch_metrics'],raw['microbatch_metrics']):
   assert a['member_request_ids']==b['member_request_ids']
   for al,bl in zip(a['layer_metrics'],b['layer_metrics']):
    for key in errors:errors[key]=max(errors[key],abs(al[key]-bl[key]))
  parity=dict(all_request_metrics_exact=reqequal,all_microbatch_layer_metrics_exact=layerequal,max_layer_field_error_ms=errors,
   completed_blocks_equal=planned['completed_blocks']==raw['completed_blocks'],source_sha256_identical=True)
  write_json(directory/'planning_replay_parity.json',parity)
  assert reqequal and layerequal,'Planning is not exact replay: stop promotion'
 write_json(directory/'analysis.json',analysis)
 command.update(status='complete',source_unchanged=True,wall_seconds=time.perf_counter()-start);write_json(directory/'command.json',command)
 brief=dict(name=name,strategy=strategy,makespan_ms=raw['makespan_ms'],
  max_tail_end_spread_ms=max(c['tail_end_spread_ms'] for c in analysis['cycles']),
  windows=[dict(start_ms=w['start_ms'],end_ms=w['end_ms'],U_percent=w['U_percent'],SLO_percent=w['slo']['percent'],mixed_cards=w['role_and_stall']['npus_with_A_and_B_compute'],all32active=w['all_npus_active']) for w in analysis['windows']])
 write_json(directory/'brief.json',brief);print(json.dumps(brief),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--mode',choices=['plan','replay','both'],default='both')
 p.add_argument('--cycles',type=int,default=2);p.add_argument('--period-ms',type=float,default=1650.);p.add_argument('--strategy',choices=['od_baseline','once'],default='od_baseline');a=p.parse_args()
 if a.mode in ('plan','both'):plan(a.name,a.cycles,a.period_ms)
 if a.mode in ('replay','both'):replay(a.name,a.strategy)
if __name__=='__main__':main()
