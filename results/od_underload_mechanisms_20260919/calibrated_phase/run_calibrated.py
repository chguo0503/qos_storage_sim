"""Offline fixed-input phase calibration. No simulator hooks or runtime changes."""
from pathlib import Path
import argparse,ast,gzip,hashlib,json,math,statistics,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE.parent/'transition'))
import metrics
from inputs.manifest import save_manifest,load_manifest,write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core,sim
IO=176*1024/2**30

def hashes():
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'simulator').rglob('*.py'))}

def prepare(name,cycles,calibrate=None,calibration_cycle=0,bootstrap_from=None):
 table=ast.literal_eval((ROOT/'data').read_text());base=table[128,4096][1]/1000
 cs=[base]*32;calibration=None
 if calibrate:
  old=json.load(gzip.open(HERE/'runs'/calibrate/'result.json.gz','rt'))['summary']
  oldreq,oldmeta=load_manifest(HERE/'inputs'/f'{calibrate}.json.gz');lookup={q.request_id:q for q in oldreq}
  tails={m['npu_id']:m for m in old['microbatch_metrics'] if lookup[m['member_request_ids'][0]].load['cycle']==calibration_cycle and lookup[m['member_request_ids'][0]].load['position']==2}
  common=max(m['completion_time_ms'] for m in tails.values())
  cs=[lookup[tails[n]['member_request_ids'][0]].load['per_layer_us']/1000+(common-tails[n]['completion_time_ms'])/8 for n in range(32)]
  calibration={'source':calibrate,'cycle':calibration_cycle,'rule':'new_C_i=old_C_i+(max_j(end_j)-end_i)/8; frozen per NPU, same value every cycle',
   'source_ends_ms':[tails[n]['completion_time_ms'] for n in range(32)],'common_end_ms':common,
   'source_starts_ms':[tails[n]['layer_metrics'][0]['compute_start_ms'] for n in range(32)]}
 boots=cs
 if bootstrap_from:
  _,bootmeta=load_manifest(HERE/'inputs'/f'{bootstrap_from}.json.gz');boots=bootmeta.get('bootstrap_tail_B_C_ms_by_npu',bootmeta['tail_B_C_ms_by_npu'])
 reqs=[]
 for n in range(32):
  for cycle in range(cycles):
   for position,miss in enumerate((256,4096,4096)):
    role='A' if position==0 else 'B';physical=n*1000000+position;rid=n*1000000+cycle*3+position
    row=table[128,miss];blocks=(128*1024-miss)//128;v=blocks*IO;c=(boots[n] if cycle==0 else cs[n]) if position==2 else row[1]/1000
    layer=tuple((sim.block_ring_hash_disk_id(physical,b,3),IO) for b in range(blocks))
    load=dict(request_id=rid,original_request_id=physical,physical_prefix_id=physical,npu_id=n,
     cycle=cycle,position=position,generation=cycle*3+position,role=role,total_tokens=128*1024,
     seq_len_k=128,nql=miss,category=sim.classify_request(128,miss),per_layer_us=c*1000,
     per_layer_kv_gb=v,required_bw_input_gbps=v/(c/1000),arrival_time=0.,arrival_ms=0.,initial=True,
     constructed_profile=bool(calibrate and position==2),original_data_row=list(row),
     timing_provenance='offline synthetic C; V and physical prefix unchanged' if calibrate and position==2 else 'exact original data row',
     physical_placement_provenance='same per-NPU/per-position prefix repeatedly read; ordinary RingHash; no ID search')
    reqs.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,)))
 meta=dict(name=name,num_npu=32,num_ssu=3,n_layers=8,cycles=cycles,sequence='ABB repeated',tail_B_C_ms_by_npu=cs,bootstrap_tail_B_C_ms_by_npu=boots,
  original_B_C_ms=base,calibration=calibration,all_arrivals_ms=0,od_queue_depth_per_ssu=8192,
  per_npu_per_ssu_slots=256,physical_prefix_reuse=True,placement='native RingHash(physical_prefix_id, block, 3)',
  logical_request_ids_unique=True,offline_only=True,no_idle=True,no_runtime_hooks=True,
  synthetic_compute=bool(calibrate),data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
  units='ms/GiB; disks40GiB/s; NPU50GiB/s',
  added_compute_per_cycle_card_ms=sum(8*(c-base) for c in cs),
  original_compute_per_cycle_card_ms=32*8*(table[128,256][1]/1000+2*base))
 save_manifest(HERE/'inputs'/f'{name}.json.gz',tuple(reqs),meta)
 return reqs,meta

def analyze(name,strategy="od_baseline"):
 reqs,meta=load_manifest(HERE/'inputs'/f'{name}.json.gz');q={r.request_id:r for r in reqs}
 directory=HERE/'runs'/(name if strategy=='od_baseline' else name+'_'+strategy)
 raw=json.load(gzip.open(directory/'result.json.gz','rt'))['summary']
 mb={(m['npu_id'],q[m['member_request_ids'][0]].load['cycle'],q[m['member_request_ids'][0]].load['position']):m for m in raw['microbatch_metrics']}
 cycles=[]
 for k in range(meta['cycles']):
  aa=[mb[n,k,0] for n in range(32)];tt=[mb[n,k,2] for n in range(32)]
  starts=[m['layer_metrics'][0]['compute_start_ms'] for m in tt];ends=[m['completion_time_ms'] for m in tt]
  gapmin=min(m['layer_metrics'][0]['compute_duration_ms'] for m in tt)
  b_layers=[l for n in range(32) for p in (1,2) for l in mb[n,k,p]['layer_metrics'][1:]]
  internals=[l['io_barrier_wait_ms'] for l in b_layers]
  rr=dict(cycle=k,A_start_spread_ms=max(m['layer_metrics'][0]['compute_start_ms'] for m in aa)-min(m['layer_metrics'][0]['compute_start_ms'] for m in aa),
   tail_B_start_spread_ms=max(starts)-min(starts),tail_B_end_spread_ms=max(ends)-min(ends),
   tail_B_start_ms=starts,tail_B_end_ms=ends,min_tail_B_C_ms=gapmin,
   causality_delta_less_than_C=max(starts)-min(starts)<gapmin,
   B_internal_stall_card_ms=sum(internals),B_internal_stall_max_ms=max(internals),
   B_internal_all_hidden=max(internals)<1e-7,
   cycle_wall_start_ms=min(m['admission_time_ms'] for m in aa),cycle_wall_end_ms=max(ends))
  if k+1<meta['cycles']:
   margins=[ends[n]-mb[n,k+1,0]['layer_metrics'][0]['io_ready_time_ms'] for n in range(32)]
   rr.update(next_A_prefetch_min_slack_ms=min(margins),next_A_all_prefetched=min(margins)>=-1e-7,
    next_A_start_spread_ms=max(mb[n,k+1,0]['layer_metrics'][0]['compute_start_ms'] for n in range(32))-min(mb[n,k+1,0]['layer_metrics'][0]['compute_start_ms'] for n in range(32)))
  cycles.append(rr)
 windows=[]
 for a,b in [(2000.,4000.),(4000.,8000.),(8000.,12000.),(12000.,20000.)]:
  if b<raw['makespan_ms']:
   mm=metrics.summarize(raw,reqs,a,b);mm['measurement_definition']['own_compute']='8 times frozen request C; calibrated tail B has synthetic C'
   windows.append(mm)
 first_drain=min(max(m['completion_time_ms'] for m in raw['request_metrics'] if m['npu_id']==n) for n in range(32))
 result=dict(metadata=meta,cycles=cycles,windows=windows,makespan_ms=raw['makespan_ms'],first_card_drains_ms=first_drain,
  all_native_invariants=all(raw['invariants'].values()),full_U_percent=100*raw['fleet_npu_compute_utilization'],
  expected_blocks=raw['completed_blocks'],first3_end_spreads_ms=[c['tail_B_end_spread_ms'] for c in cycles[:3]],
  last3_end_spreads_ms=[c['tail_B_end_spread_ms'] for c in cycles[-3:]],
  full=metrics.summarize(raw,reqs,0.,raw['makespan_ms'],full=True))
 result['full']['measurement_definition']['own_compute']='8 times frozen request C; calibrated tail B has synthetic C'
 write_json(directory/'analysis.json',result)
 compact=dict(name=name,makespan_ms=result['makespan_ms'],first_card_drains_ms=first_drain,
  end_spreads_ms=[c['tail_B_end_spread_ms'] for c in cycles],A_start_spreads_ms=[c['A_start_spread_ms'] for c in cycles],
  B_internal_stall_ms=[c['B_internal_stall_card_ms'] for c in cycles],
  windows=[dict(start_ms=w['start_ms'],end_ms=w['end_ms'],U_percent=w['U_percent'],mixed_cards=w['role_and_stall']['npus_with_A_and_B_compute'],
   active=w['all_npus_active'],any_overload_percent=w['demand']['any_disk_overload_percent'],SLO=w['slo']['percent']) for w in windows])
 write_json(directory/'brief.json',compact);print(json.dumps(compact),flush=True)
 return result

def run(name,strategy="od_baseline"):
 reqs,meta=load_manifest(HERE/'inputs'/f'{name}.json.gz');out=HERE/'runs'/(name if strategy=='od_baseline' else name+'_'+strategy);out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter();before=hashes();command=dict(name=name,strategy=strategy,status='running',source_sha256=before,
  manifest_sha256=hashlib.sha256((HERE/'inputs'/f'{name}.json.gz').read_bytes()).hexdigest(),runtime_hooks=False)
 write_json(out/'command.json',command);print(json.dumps({'start':name,'requests':len(reqs),'cycles':meta['cycles']}),flush=True)
 raw=run_simulation(reqs,strategy=strategy,num_npu=32,num_ssu=3,n_layers=8,seed=7,od_queue_depth_per_ssu=8192 if strategy=='od_baseline' else None)
 assert before==hashes();write_json(out/'result.json.gz',raw)
 command.update(status='complete',source_unchanged=True,wall_seconds=time.perf_counter()-start);write_json(out/'command.json',command)
 analyze(name,strategy)

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cycles',type=int,default=2)
 p.add_argument('--calibrate');p.add_argument('--calibration-cycle',type=int,default=0);p.add_argument('--bootstrap-from')
 p.add_argument('--strategy',choices=['od_baseline','once'],default='od_baseline');p.add_argument('--replay-existing',action='store_true')
 p.add_argument('--analyze',action='store_true');a=p.parse_args()
 if a.analyze:analyze(a.name,a.strategy);return
 if not a.replay_existing:prepare(a.name,a.cycles,a.calibrate,a.calibration_cycle,a.bootstrap_from)
 run(a.name,a.strategy)
if __name__=='__main__':main()
