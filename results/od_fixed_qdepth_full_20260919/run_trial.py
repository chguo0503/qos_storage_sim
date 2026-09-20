#!/usr/bin/env python3
"""OD fixed SSD slots: frozen full24 input, full drain, optional original-depth parity.

--depth-per-npu 256 maps to API od_queue_depth_per_ssu=8192 for32NPUs.
Depth None preserves the original OD policy; no slots are borrowed when capped.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,math,os,platform,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
from inputs.manifest import load_manifest,read_json,write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as native
from simulator.policies.baselines import baseline_configuration,baseline_qos_config,od_npu_path_ids
from metrics import summarize,live_summary,overlap

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def source_hashes():
 return {str(p.relative_to(HERE)):sha(p) for p in sorted((HERE/'runtime').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}|{p.name:sha(p) for p in (Path(__file__),HERE/'metrics.py')}

def rows_signature(summary):
 req=sorted(summary['request_metrics'],key=lambda r:r['request_id'])
 layer=sorted([[b['npu_id'],b['member_request_ids'],m] for b in summary['microbatch_metrics'] for m in b['layer_metrics']],key=lambda r:(r[0],r[1],r[2]['layer']))
 return dict(request=stable(req),layer=stable(layer))

def numeric_difference(a,b,path='',differences=None):
 if differences is None:differences=[]
 if isinstance(a,dict) and isinstance(b,dict):
  for key in sorted(a.keys()|b.keys()):
   if key not in a or key not in b:differences.append(dict(path=path+'.'+key,error='missing key'))
   else:numeric_difference(a[key],b[key],path+'.'+key,differences)
 elif isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):differences.append(dict(path=path,error='length',actual=len(a),reference=len(b)))
  else:
   for i,(x,y) in enumerate(zip(a,b)):numeric_difference(x,y,f'{path}[{i}]',differences)
 elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
  if not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-7):differences.append(dict(path=path,actual=a,reference=b,absolute_error=abs(a-b)))
 elif a!=b:differences.append(dict(path=path,actual=a,reference=b))
 return differences

def project_like(actual,reference):
 if isinstance(reference,dict):return {k:project_like(actual[k],v) for k,v in reference.items()}
 if isinstance(reference,list):return [project_like(a,b) for a,b in zip(actual,reference)] if len(actual)==len(reference) else actual
 return actual

def compare_reference(result,path):
 old=read_json(path);now=result['summary'];then=old['summary']
 req=lambda s:sorted(s['request_metrics'],key=lambda r:r['request_id'])
 layers=lambda s:sorted(s['microbatch_metrics'],key=lambda b:b['batch_id'])
 rr=project_like(req(now),req(then));bb=project_like(layers(now),layers(then))
 fields=('request_id','npu_id','admission_time_ms','completion_time_ms','own_compute_ms','processing_latency_ms','io_stall_ms','compute_queue_wait_ms')
 ar=[{k:r[k] for k in fields} for r in rr];br=[{k:r[k] for k in fields} for r in req(then)]
 timing_errors=numeric_difference(ar,br,'request_timing')+numeric_difference(bb,layers(then),'microbatch_metrics')
 request_errors=numeric_difference(rr,req(then),'original_request_metrics')
 window_rows=[]
 for n,o in zip(result['analysis'],old['analysis']):
  window_rows.append(dict(window_ms=[n['start_ms'],n['end_ms']],U_error_pp=n['U_percent']-o['U_percent'],SLO15_error_pp=n['slo']['percent']-o['slo']['percent'],same_cohort=n['cohort_request_ids']==o['cohort_request_ids']))
 max_bin_error=max(abs(x-y) for a,b in zip(result['warm_ssd_10ms_GiB_s'],old['warm_ssd_10ms_GiB_s']) for x,y in zip(a,b))
 return dict(reference_sha256=sha(path),same_input=old['input_fingerprint']==result['input_fingerprint'],timing_equal=not timing_errors,original_request_metrics_equal=not request_errors,request_timing_sha256=dict(actual=stable(ar),reference=stable(br)),layer_metrics_sha256=dict(actual=stable(bb),reference=stable(layers(then))),timing_error_count=len(timing_errors),timing_errors=timing_errors[:20],original_request_metric_error_count=len(request_errors),original_request_metric_differences=request_errors[:20],max_numeric_timing_error_ms=max([e.get('absolute_error',0) for e in timing_errors],default=0),windows=window_rows,max_SSD_10ms_error_GiB_s=max_bin_error)

def check_depth(summary,depth,expected,*,smoke=False):
 if depth is None:
  assert 'ssd_queue_depth' not in summary
  return dict(passed=True,enabled=False,default_summary_preserved=True)
 d=summary['ssd_queue_depth']
 assert d['enabled'] and d['per_npu_per_ssu_slots']==depth
 assert d['total_reserved_slots_per_ssu']==depth*32
 assert not d['queue_slot_borrowing'] and not d['bandwidth_policy_changed']
 assert d['matrix_order']=='[npu_id][ssu_id]'
 names=('peak_outstanding_blocks_by_npu_ssu','end_outstanding_blocks_by_npu_ssu','end_host_deferred_blocks_by_npu_ssu','blocked_state_episodes_by_npu_ssu','host_blocked_state_ms_by_npu_ssu')
 for name in names:assert len(d[name])==32 and all(len(row)==3 for row in d[name])
 peak=d['peak_outstanding_blocks_by_npu_ssu'];maximum=max(map(max,peak))
 assert 0<=maximum<=depth
 upper_bound=[sum(row[s] for row in peak) for s in range(3)]
 assert all(v<=depth*32 for v in upper_bound)
 for name in ('end_outstanding_blocks_by_npu_ssu','end_host_deferred_blocks_by_npu_ssu'):
  assert all(x==0 for row in d[name] for x in row)
 for name in ('activated_blocks','issued_blocks','ssd_completed_blocks','hbm_completed_blocks'):
  assert d[name]==expected
 for name in ('host_deferred_blocks_at_stop','ssd_outstanding_blocks_at_stop','link_outstanding_blocks_at_stop','input_not_yet_activated_blocks'):
  assert d[name]==0
 invariants={k:v for k,v in summary['invariants'].items() if k.startswith('ssd_depth_')}
 assert len(invariants)>=6 and all(invariants.values())
 blocked=sum(map(sum,d['blocked_state_episodes_by_npu_ssu']))
 host_blocked=sum(map(sum,d['host_blocked_state_ms_by_npu_ssu']))
 assert math.isclose(d['host_activation_to_enqueue_block_ms'],d['completed_host_activation_to_enqueue_block_ms'],rel_tol=1e-10,abs_tol=1e-8)
 assert math.isclose(d['completed_host_activation_to_enqueue_block_ms']+d['completed_enqueue_to_hbm_block_ms'],d['completed_activation_to_hbm_block_ms'],rel_tol=1e-10,abs_tol=1e-8)
 if not smoke:assert maximum==depth and blocked>0 and host_blocked>0
 return dict(passed=True,enabled=True,per_npu_per_ssu=depth,peak_observed_per_npu_per_ssu=maximum,sum_individual_peaks_by_ssu_upper_bound=upper_bound,upper_bound_not_simultaneous_peak=True,blocked_state_episodes=blocked,host_blocked_submission_state_ms=host_blocked,activation_to_enqueue_block_mean_ms=d['completed_host_activation_to_enqueue_block_ms']/expected,enqueue_to_hbm_block_mean_ms=d['completed_enqueue_to_hbm_block_ms']/expected,activation_to_hbm_block_mean_ms=d['completed_activation_to_hbm_block_ms']/expected,activation_to_enqueue_includes_original_issue_pacing=True,completed_blocks=expected,invariants=invariants)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--scenario',choices=('full',),default='full')
 ap.add_argument('--seed',type=int,choices=(7,19,43),required=True)
 ap.add_argument('--policy',choices=('od_baseline',),default='od_baseline')
 ap.add_argument('--depth-per-npu',choices=('none','256'),required=True)
 ap.add_argument('--label')
 ap.add_argument('--smoke',action='store_true')
 args=ap.parse_args()
 depth=None if args.depth_per_npu=='none' else int(args.depth_per_npu)
 variant='od_unlimited' if depth is None else f'od_depth{depth}'
 label=args.label or f'full_{variant}_seed{args.seed}'
 assert Path(label).name==label
 target=HERE/'runs'/label;target.mkdir(parents=True,exist_ok=False)
 manifest=HERE/'inputs'/f'{args.scenario}_seed{args.seed}.json.gz'
 requests,meta=load_manifest(manifest)
 audit=read_json(HERE/'input_audit.json')
 frozen=next(r for r in audit['cases'] if r['seed']==args.seed)
 assert sha(manifest)==frozen['manifest_sha256']
 assert native.continuous_batch_input_fingerprint(requests)==frozen['input_fingerprint']
 assert (meta['num_npu'],meta['num_ssu'],meta['n_layers'],meta['seed'])==(32,3,8,args.seed)
 (target/'manifest.json.gz').write_bytes(manifest.read_bytes())
 if args.smoke:
  first={}
  for q in requests:first.setdefault(q.npu_id,q)
  requests=tuple(first.values())
 assert 'runtime_sha256' in audit, 'Freeze the tested runtime before execution'
 assert all(sha(HERE/name)==digest for name,digest in audit['runtime_sha256'].items())
 before=source_hashes();started=last=time.perf_counter()
 expected=sum(8*len(q.placement[0]) for q in requests)
 observed=0;busy=[[0.]*3 for _ in range(2)];fullbusy=[0.]*3
 per_npu=[[0.]*32 for _ in range(3)];bins=[[0.]*1000 for _ in range(3)]
 windows=((2000.,4000.),(2000.,6000.));warm_saved=False
 configuration=dict(audit['configuration'],slot_limit_per_NPU_per_SSU=depth,
                    slot_limit_total_per_SSU=None if depth is None else depth*32,
                    fixed_slot_borrowing=False if depth is not None else None)
 record=dict(status='running',scenario=args.scenario,seed=args.seed,policy=args.policy,variant=variant,queue_depth_per_npu_per_ssu=depth,queue_depth_per_ssu=None if depth is None else depth*32,smoke=args.smoke,argv=sys.argv,pid=os.getpid(),host=platform.node(),python=sys.version,started_utc=utc(),source_sha256=before,manifest_sha256=sha(manifest),input_fingerprint=frozen['input_fingerprint'],expected_blocks=expected,configuration=configuration,original_input_audit=frozen)
 write_json(target/'command.json',record)
 callback=native._register_complete
 def preview(ctx):
  nonlocal warm_saved
  if args.smoke or warm_saved or ctx.current_time_ms<4100:return
  cohort=[q for q in ctx.requests.values() if q.admitted and 2000<=q.admission_time_ms<4000]
  if not cohort or not all(q.completed for q in cohort):return
  for npu in ctx.npus:
   if npu.link_active_flow is not None and npu.link_active_flow.ssd_activation_time<4000:return
   if any(f.ssd_activation_time<4000 for f in npu.link_pending):return
  row=summarize(live_summary(ctx),requests,2000.,4000.)
  row.update(policy=args.policy,scenario=args.scenario,seed=args.seed,SSD_GiB_s=[x*40/2000 for x in busy[0]],observed_at_ms=ctx.current_time_ms)
  write_json(target/'warm_preview.json',row);warm_saved=True
  print(json.dumps(dict(warm=True,scenario=args.scenario,policy=args.policy,seed=args.seed,U=row['U_percent'],slo=row['slo'])),flush=True)
 def observe(ctx,flow):
  nonlocal observed,last
  observed+=1;a,z=flow.ssd_activation_time,flow.link_enqueue_time;d=flow.disk_id
  fullbusy[d]+=z-a
  for i,(left,right) in enumerate(windows):busy[i][d]+=overlap(a,z,left,right)
  if a<4000 and z>2000:
   a1,z1=max(a,2000.),min(z,4000.)
   per_npu[d][flow.npu_id]+=z1-a1
   first,lastbin=int((a1-2000)//2),min(999,int((z1-2000)//2))
   for j in range(first,lastbin+1):bins[d][j]+=overlap(a1,z1,2000+2*j,2002+2*j)
  ret=callback(ctx,flow)
  if observed%50000==0:
   preview(ctx);now=time.perf_counter()
   if now-last>=15:
    progress=dict(completed_blocks=observed,expected_blocks=expected,simulation_ms=ctx.current_time_ms,completed_requests=ctx.completed_requests,wall_seconds=now-started)
    write_json(target/'progress.json',progress);print(json.dumps(progress),flush=True);last=now
  return ret
 try:
  with patch.object(native,'_register_complete',observe):
   result=run_simulation(requests,strategy=args.policy,num_npu=32,num_ssu=3,n_layers=8,seed=args.seed,disk_bw_gib_s=40,npu_bw_gib_s=50,collector_interval_ms=5,cross_request_layer0_prefetch=True,od_queue_depth_per_ssu=None if depth is None else depth*32)
  summary=result['summary'];stats=result['adapter_statistics']
  assert observed==expected==summary['completed_blocks']==summary['submitted_blocks']
  assert all(summary['invariants'].values()) and summary['request_count']==len(requests)
  assert stats['assignment_count']==stats['reorder_calls']==0 and not stats['cir_write_events']
  byid={q.request_id:q for q in requests}
  for n in range(32):
   actual=[r['request_id'] for r in sorted(summary['request_metrics'],key=lambda r:r['admission_time_ms']) if byid[r['request_id']].npu_id==n]
   assert actual==[q.request_id for q in requests if q.npu_id==n]
  analyses=[]
  if not args.smoke:
   for i,(a,z) in enumerate(windows):
    row=summarize(summary,requests,a,z);row['SSD_GiB_s']=[x*40/(z-a) for x in busy[i]];analyses.append(row)
   full=summarize(summary,requests,0.,summary['makespan_ms'],full=True)
   full['SSD_GiB_s']=[x*40/summary['makespan_ms'] for x in fullbusy];analyses.append(full)
   assert abs(full['U_percent']-100*summary['fleet_npu_compute_utilization'])<1e-7
   assert all(row['all_npus_active'] for row in analyses[:2])
   assert all(abs(sum(bins[d])-busy[0][d])<1e-7 for d in range(3))
  qos=baseline_qos_config(args.policy,32,40)
  ownership=True
  if args.policy=='od_baseline':
   paths=od_npu_path_ids(32);assert len(set(paths))==32
   assert all(qos.path_cirs[p]==1.25 for p in paths)
   assert all(row['path_id']==paths[row['npu_id']] for row in stats['routed_blocks_by_ssu_npu_path'])
   assert sum(row['blocks'] for row in stats['routed_blocks_by_ssu_npu_path'])==observed
  result.update(metadata=meta,scenario=args.scenario,seed=args.seed,variant=variant,queue_depth_per_npu_per_ssu=depth,queue_depth_per_ssu=None if depth is None else depth*32,queue_depth_diagnostics={k:v for k,v in summary.items() if any(token in k for token in ('queue_depth','backpressure','host_wait','host_pending','outstanding'))},analysis=analyses,baseline_configuration=baseline_configuration(args.policy,32,40),static_path_cirs_gib_s=list(qos.path_cirs),warm_ssd_10ms_GiB_s=[[sum(d[i:i+5])*4 for i in range(0,1000,5)] for d in bins],bandwidth_2ms=dict(start_ms=2000.,end_ms=4000.,interval_ms=2,ssd_GiB_s=[[x*20 for x in d] for d in bins]),warm_ssd_GiB_s_by_ssu_npu=[[x*40/2000 for x in d] for d in per_npu],input_placement_preserved=True,source_sha256=before,wall_seconds_total=time.perf_counter()-started)
  write_json(target/'result.json.gz',result)
  if not args.smoke:
   ref=HERE/'references'/f'full_od_unlimited_seed{args.seed}.json.gz'
   comparison=compare_reference(result,ref);write_json(target/'reference_comparison.json',comparison)
   assert comparison['same_input']
   if depth is None:
    assert comparison['timing_equal'] and comparison['original_request_metrics_equal'],f'Unlimited OD parity failed: {comparison}'
  # The depth checks are filled from the approved core diagnostics below.
  depth_checks=check_depth(summary,depth,expected,smoke=args.smoke)
  write_json(target/'queue_depth_checks.json',depth_checks)
  assert source_hashes()==before
  record.update(status='complete',completed_requests=len(summary['request_metrics']),observed_blocks=observed,result_sha256=sha(target/'result.json.gz'),source_unchanged=True,checks=dict(all_invariants=True,FIFO_preserved=True,static_QoS=True,unchanged_input=True,unchanged_L1_L2=True,OD_owner_and_CIR=ownership,queue_depth_limit=depth_checks['passed']),metrics=[dict(start_ms=r['start_ms'],end_ms=r['end_ms'],U_percent=r['U_percent'],slo=r['slo']) for r in analyses])
 except Exception as error:
  record.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
 finally:
  record.update(ended_utc=utc(),wall_seconds=time.perf_counter()-started);write_json(target/'command.json',record)
 print(json.dumps(dict(complete=True,scenario=args.scenario,policy=args.policy,seed=args.seed,wall_seconds=record['wall_seconds'],metrics=record['metrics'])),flush=True)
if __name__=='__main__':main()
