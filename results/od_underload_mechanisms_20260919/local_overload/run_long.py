"""Bounded OD screening and full-drain OD/Once verification; no core mutation."""
from pathlib import Path
from unittest.mock import patch
import argparse,hashlib,json,math,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'runtime'),str(HERE)]
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core
from inputs.manifest import load_manifest,write_json
from metrics import summarize,exact_demand,live_summary,overlap
WINDOWS=[(2000.,4000.),(4000.,8000.),(8000.,12000.),(12000.,20000.),(20000.,40000.),(40000.,60000.)]
PREVIEW_WINDOWS=sorted(set(WINDOWS+[(float(t),float(t+4000)) for t in range(12000,60000,4000)]))
def hashes():return {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted((HERE/'runtime').rglob('*.py')),HERE/'runtime/data',Path(__file__),HERE/'metrics.py']}
class StopPilot(Exception):pass

def pilot_metrics(ctx,requests,left=2000.,right=4000.):
 s=live_summary(ctx);roles=[dict(A=0.,B=0.) for _ in range(32)];active=[0.]*32;byid={q.request_id:q for q in requests}
 for r in s['request_metrics']:active[r['npu_id']]+=overlap(r['admission_time_ms'],r['completion_time_ms'],left,right)
 for b in s['microbatch_metrics']:
  role=byid[b['member_request_ids'][0]].load['role']
  for l in b['layer_metrics']:roles[b['npu_id']][role]+=overlap(l['compute_start_ms'],l['compute_end_ms'],left,right)
 return dict(U_percent=100*sum(sum(r.values()) for r in roles)/(32*(right-left)),all_npus_active=all(abs(x-(right-left))<1e-7 for x in active),per_npu_role_compute_ms=roles,mixed_cards=sum(min(r.values())>0 for r in roles),demand=exact_demand(s['request_metrics'],byid,left,right),observed_at_ms=ctx.current_time_ms,SLO_not_computed=True,scientific_status='window_preview_before_full_drain',start_ms=left,end_ms=right)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--strategy',choices=('od_baseline','once'),default='od_baseline');ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
 label=a.case+'_'+a.strategy+('_pilot' if a.pilot else '_full');target=HERE/('pilots' if a.pilot else 'runs')/label;target.mkdir(parents=True,exist_ok=False)
 manifest=HERE/'inputs'/f'{a.case}.json.gz';requests,meta=load_manifest(manifest);(target/'manifest.json.gz').write_bytes(manifest.read_bytes());before=hashes();start=last=time.perf_counter();observed=0;measured=None
 busy=[[0.]*3 for _ in WINDOWS];warm_bins=[[0.]*200 for _ in range(3)];per_npu=[[0.]*32 for _ in range(3)];saved_preview=False;window_previews={}
 record=dict(status='running',argv=sys.argv,pid=os.getpid(),pilot=a.pilot,strategy=a.strategy,case=a.case,metadata=meta,source_sha256=before,input_fingerprint=core.continuous_batch_input_fingerprint(requests),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),OD_queue_depth_per_ssu=8192 if a.strategy=='od_baseline' else None,Once_depth_note='original Once API retains unbounded submission depth; disclosed comparison difference' if a.strategy=='once' else None)
 write_json(target/'command.json',record);callback=core._register_complete
 def observe(ctx,flow):
  nonlocal observed,last,measured,saved_preview
  observed+=1;d=flow.disk_id;x,z=flow.ssd_activation_time,flow.link_enqueue_time
  for k,(left,right) in enumerate(WINDOWS):busy[k][d]+=overlap(x,z,left,right)
  if x<4000 and z>2000:
   lo,hi=max(x,2000),min(z,4000);per_npu[d][flow.npu_id]+=hi-lo
   for j in range(max(0,int((lo-2000)//10)),min(199,int((hi-2000)//10))+1):warm_bins[d][j]+=overlap(lo,hi,2000+j*10,2010+j*10)
  ret=callback(ctx,flow)
  if ctx.current_time_ms>=4010 and not saved_preview:
   pending=[f for n in ctx.npus for f in ([n.link_active_flow] if n.link_active_flow else [])+list(n.link_pending)]
   if not any(f.ssd_activation_time<4000 for f in pending):
    measured=pilot_metrics(ctx,requests);measured['SSD_GiB_s']=[v*40/2000 for v in busy[0]];write_json(target/'warm_preview.json',measured);saved_preview=True
    print(json.dumps(dict(warm=True,label=label,U=measured['U_percent'],over=measured['demand']['per_disk_overload_percent'],meanD=measured['demand']['per_disk_mean_GiB_s'],mixed=measured['mixed_cards'])),flush=True)
    if a.pilot:raise StopPilot()
  if observed%25000==0:
   for k,(left,right) in enumerate(PREVIEW_WINDOWS):
    if str(k) in window_previews or ctx.current_time_ms<right+10:continue
    pending=[f for n in ctx.npus for f in ([n.link_active_flow] if n.link_active_flow else [])+list(n.link_pending)]
    if any(f.ssd_activation_time<right for f in pending):continue
    row=pilot_metrics(ctx,requests,left,right)
    row['SSD_GiB_s']=([v*40/(right-left) for v in busy[WINDOWS.index((left,right))]] if (left,right) in WINDOWS else None)
    window_previews[str(k)]=row;write_json(target/'window_previews.json',window_previews)
    print(json.dumps(dict(window=[left,right],U=row['U_percent'],over=row['demand']['per_disk_overload_percent'],mixed=row['mixed_cards'])),flush=True)
  if observed%25000==0 and time.perf_counter()-last>=15:
   p=dict(simulation_ms=ctx.current_time_ms,completed_blocks=observed,expected_blocks=meta['expected_blocks'],wall_seconds=time.perf_counter()-start);write_json(target/'progress.json',p);print(json.dumps(p),flush=True);last=time.perf_counter()
  return ret
 try:
  with patch.object(core,'_register_complete',observe):
   result=run_simulation(requests,strategy=a.strategy,num_npu=32,num_ssu=3,n_layers=8,seed=7,disk_bw_gib_s=40,npu_bw_gib_s=50,collector_interval_ms=5,cross_request_layer0_prefetch=True,od_queue_depth_per_ssu=8192 if a.strategy=='od_baseline' else None)
  assert not a.pilot,'Input drained before4s'
  summary=result['summary'];assert observed==meta['expected_blocks']==summary['completed_blocks'] and all(summary['invariants'].values())
  analyses=[]
  for k,(left,right) in enumerate(WINDOWS):
   row=summarize(summary,requests,left,right);row['SSD_GiB_s']=[v*40/(right-left) for v in busy[k]];analyses.append(row)
  assert all(r['all_npus_active'] for r in analyses)
  if a.strategy=='od_baseline':
   q=summary['ssd_queue_depth'];assert max(map(max,q['peak_outstanding_blocks_by_npu_ssu']))<=256 and q['host_deferred_blocks_at_stop']==q['ssd_outstanding_blocks_at_stop']==q['link_outstanding_blocks_at_stop']==0
  result.update(metadata=meta,analysis=analyses,source_sha256=before,warm_ssd_10ms_GiB_s=[[v*4 for v in disk] for disk in warm_bins],warm_ssd_GiB_s_by_ssu_npu=[[v*40/2000 for v in disk] for disk in per_npu])
  write_json(target/'result.json.gz',result);record.update(status='complete',analysis=analyses,completed_blocks=observed)
 except StopPilot:record.update(status='complete_pilot',measurement=measured,completed_blocks=observed,completed_simulation=False)
 except Exception as e:record.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
 finally:
  record.update(wall_seconds=time.perf_counter()-start,source_unchanged=hashes()==before);assert record['source_unchanged'];write_json(target/'command.json',record)
 print(json.dumps(dict(label=label,status=record['status'],wall_seconds=record['wall_seconds'])),flush=True)
if __name__=='__main__':main()
