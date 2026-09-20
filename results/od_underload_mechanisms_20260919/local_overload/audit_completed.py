"""Independent read-only checks for completed runs, excluding pilots/cancellations."""
from pathlib import Path
import gzip,hashlib,json,math
HERE=Path(__file__).resolve().parent

def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];checks=[]
 for path in sorted((HERE/'runs').glob('*/command.json')):
  c=read(path)
  if c['status']!='complete':continue
  raw=read(path.with_name('result.json.gz'));s=raw['summary'];stats=raw['adapter_statistics'];manifest=read(path.with_name('manifest.json.gz'))
  assert c['source_unchanged'] and all(sha(HERE/p)==v for p,v in c['source_sha256'].items())
  assert sha(path.with_name('manifest.json.gz'))==c['manifest_sha256']
  assert c['input_fingerprint']==raw['input_fingerprint']==s['input_fingerprint']==manifest['input_fingerprint']
  assert all(s['invariants'].values()) and s['completed_blocks']==c['metadata']['expected_blocks']
  assert len(s['request_metrics'])==len(manifest['requests'])
  assert stats['assignment_count']==stats['reorder_calls']==0 and not stats['cir_write_events']
  if c['strategy']=='od_baseline':
   q=s['ssd_queue_depth'];assert q['per_npu_per_ssu_slots']==256 and q['total_reserved_slots_per_ssu']==8192
   assert not q['queue_slot_borrowing'] and not q['bandwidth_policy_changed']
   assert max(map(max,q['peak_outstanding_blocks_by_npu_ssu']))<=256
   assert q['host_deferred_blocks_at_stop']==q['ssd_outstanding_blocks_at_stop']==q['link_outstanding_blocks_at_stop']==0
   conf=stats['baseline_configuration'];assert conf['usable_path_count_per_ssu']==32 and conf['per_npu_cir_gib_s']==1.25 and conf['idle_capacity_borrowing'] and conf['path_pir']=='unlimited'
   assert all(row['path_id']==conf['npu_path_ids'][row['npu_id']] for row in stats['routed_blocks_by_ssu_npu_path'])
  for w in raw['analysis']:
   left,right=w['start_ms'],w['end_ms'];duration=right-left;compute=[0.]*32
   for b in s['microbatch_metrics']:
    for layer in b['layer_metrics']:compute[b['npu_id']]+=max(0,min(right,layer['compute_end_ms'])-max(left,layer['compute_start_ms']))
   u=100*sum(compute)/(32*duration);assert math.isclose(u,w['U_percent'],abs_tol=1e-8)
   assert all(math.isclose(100*x/duration,y,abs_tol=1e-8) for x,y in zip(compute,w['per_npu_U_percent']))
   # Bounded near-capacity runs intentionally retain their original population.
   # Their last 12--16s window may include draining; audit the flag, never hide it.
   active=[0.]*32
   for request in s['request_metrics']:
    active[request['npu_id']]+=max(0,min(right,request['completion_time_ms'])-max(left,request['admission_time_ms']))
   independently_all_active=all(math.isclose(t,duration,abs_tol=1e-7) for t in active)
   assert independently_all_active==w['all_npus_active']
   req=[r for r in s['request_metrics'] if left<=r['admission_time_ms']<right];hits=sum(r['processing_latency_ms']<=1.5*r['own_compute_ms']+1e-9 for r in req)
   assert hits==w['slo']['passed'] and len(req)==w['slo']['count']
   assert w['role_and_stall']['io_stall']['maximum_accounting_error_ms']<1e-6
   rows.append(dict(label=path.parent.name,window_ms=[left,right],U_percent=u,SLO15=w['slo'],all_npus_active=independently_all_active,eligible_for_all_active_claim=independently_all_active,mixed_cards=w['role_and_stall']['npus_with_A_and_B_compute'],demand_mean_GiB_s=w['demand']['per_disk_mean_GiB_s'],demand_min_GiB_s=w['demand']['per_disk_min_GiB_s'],demand_max_GiB_s=w['demand']['per_disk_max_GiB_s'],per_disk_overload_percent=w['demand']['per_disk_overload_percent'],SSD_GiB_s=w['SSD_GiB_s'],stall_card_ms=w['role_and_stall']['io_stall']['by_kind_card_ms']))
  checks.append(dict(label=path.parent.name,passed=True,source_unchanged=True,input_fingerprint=c['input_fingerprint'],result_sha256=sha(path.with_name('result.json.gz')),blocks=s['completed_blocks'],requests=len(s['request_metrics'])))
 result=dict(status='partial' if any(read(p)['status']=='running' for p in (HERE/'runs').glob('*/command.json')) else 'complete',completed_runs=checks,rows=rows)
 (HERE/'completed_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],runs=len(checks),rows=len(rows))))
if __name__=='__main__':main()
