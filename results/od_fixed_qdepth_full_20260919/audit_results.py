"""Read completed experiments and independently validate frozen inputs/queue limits."""
from pathlib import Path
import csv,gzip,hashlib,json,math,statistics
HERE=Path(__file__).resolve().parent

def read(p):
 if p.suffix=='.gz':
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 audit=read(HERE/'input_audit.json');rows=[];checks=[]
 for seed,depth in ((7,None),(7,256),(19,256),(43,256)):
  name=f"full_od_{'unlimited' if depth is None else 'depth256'}_seed{seed}"
  case=HERE/'runs'/name;c=read(case/'command.json');r=read(case/'result.json.gz')
  ref=read(case/'reference_comparison.json');q=read(case/'queue_depth_checks.json')
  original=next(x for x in audit['cases'] if x['seed']==seed)
  assert c['status']=='complete' and not c['smoke'] and all(c['checks'].values())
  assert c['result_sha256']==sha(case/'result.json.gz')
  assert c['manifest_sha256']==sha(case/'manifest.json.gz')==original['manifest_sha256']
  assert c['input_fingerprint']==r['input_fingerprint']==original['input_fingerprint']
  assert all(sha(HERE/path)==digest for path,digest in c['source_sha256'].items())
  assert q['passed'] and c['source_unchanged']
  s=r['summary'];assert all(s['invariants'].values())
  assert c['observed_blocks']==s['completed_blocks']==s['submitted_blocks']==original['blocks']
  assert c['completed_requests']==len(s['request_metrics'])==original['requests']
  assert ref['same_input']
  if depth is None:
   assert ref['timing_equal'] and ref['original_request_metrics_equal']
   assert s==read(HERE/'references'/f'full_od_unlimited_seed{seed}.json.gz')['summary']
  else:
   d=s['ssd_queue_depth'];assert d['per_npu_per_ssu_slots']==256
   assert d['total_reserved_slots_per_ssu']==8192 and not d['queue_slot_borrowing']
   peaks=d['peak_outstanding_blocks_by_npu_ssu']
   assert max(map(max,peaks))==256 and min(map(min,peaks))>=0
   assert all(sum(row[k] for row in peaks)<=8192 for k in range(3))
   assert sum(map(sum,d['blocked_state_episodes_by_npu_ssu']))>0
   assert all(x['max_outstanding_blocks']<=8192 for x in s['disk_stats'])
   assert d['host_deferred_blocks_at_stop']==d['ssd_outstanding_blocks_at_stop']==d['link_outstanding_blocks_at_stop']==0
  w=r['analysis'][0];assert (w['start_ms'],w['end_ms'])==(2000.,4000.)
  assert w['all_npus_active'] and len(w['per_npu_U_percent'])==32
  # Independently integrate compute intervals, rather than trusting the runner U.
  compute=[0.]*32
  for batch in s['microbatch_metrics']:
   for layer in batch['layer_metrics']:
    compute[batch['npu_id']]+=max(0,min(4000.,layer['compute_end_ms'])-max(2000.,layer['compute_start_ms']))
  u=100*sum(compute)/64000.
  assert math.isclose(u,w['U_percent'],abs_tol=1e-8)
  for actual,expected in zip(w['per_npu_U_percent'],[100*x/2000 for x in compute]):assert math.isclose(actual,expected,abs_tol=1e-8)
  cohort=[x for x in s['request_metrics'] if 2000<=x['admission_time_ms']<4000]
  ratio=[x['processing_latency_ms']/x['own_compute_ms'] for x in cohort]
  hits=sum(x['completion_time_ms']-x['admission_time_ms']<=1.5*x['own_compute_ms']+1e-9 for x in cohort);slo=100*hits/len(cohort)
  assert math.isclose(slo,w['slo']['percent'],abs_tol=1e-8)
  assert set(w['cohort_request_ids'])=={x['request_id'] for x in cohort}
  for k in range(3):
   assert math.isclose(statistics.mean(r['warm_ssd_10ms_GiB_s'][k]),w['SSD_GiB_s'][k],abs_tol=1e-8)
   assert math.isclose(sum(r['warm_ssd_GiB_s_by_ssu_npu'][k]),w['SSD_GiB_s'][k],abs_tol=1e-8)
  rows.append(dict(seed=seed,depth_per_npu=depth,variant=r['variant'],warm_U_percent=u,warm_SLO15_percent=slo,warm_admitted_requests=len(cohort),warm_SLO15_hits=hits,warm_SSD0_GiB_s=w['SSD_GiB_s'][0],warm_SSD1_GiB_s=w['SSD_GiB_s'][1],warm_SSD2_GiB_s=w['SSD_GiB_s'][2],full_U_percent=r['analysis'][-1]['U_percent'],full_SLO15_percent=r['analysis'][-1]['slo']['percent'],makespan_ms=s['makespan_ms'],SSD0_peak_outstanding=s['disk_stats'][0]['max_outstanding_blocks'],SSD1_peak_outstanding=s['disk_stats'][1]['max_outstanding_blocks'],SSD2_peak_outstanding=s['disk_stats'][2]['max_outstanding_blocks'],wall_seconds=c['wall_seconds'],timing_equal_to_old=ref['timing_equal'],request_metrics_equal_to_old=ref['original_request_metrics_equal'],maximum_timing_error_ms=ref['max_numeric_timing_error_ms'],blocked_state_episodes=q.get('blocked_state_episodes',0),host_blocked_submission_state_ms=q.get('host_blocked_submission_state_ms',0),activation_to_enqueue_block_mean_ms=q.get('activation_to_enqueue_block_mean_ms'),enqueue_to_hbm_block_mean_ms=q.get('enqueue_to_hbm_block_mean_ms'),activation_to_hbm_block_mean_ms=q.get('activation_to_hbm_block_mean_ms')))
  checks.append(dict(label=name,passed=True,result_sha256=c['result_sha256'],input_fingerprint=r['input_fingerprint'],source_unchanged=True,requests=len(cohort),depth_checks=q,reference_comparison=ref,entire_native_summary_equal_to_reference=True if depth is None else None))
 out=HERE/'data';out.mkdir(exist_ok=True)
 with (out/'per_seed_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 final=dict(status='complete',passed=True,cases=checks,rows=rows,macro_depth256={k:statistics.mean(r[k] for r in rows if r['depth_per_npu']==256) for k in ('warm_U_percent','warm_SLO15_percent','full_U_percent','full_SLO15_percent')},total_requests=sum(x['requests'] for x in audit['cases'])+audit['cases'][0]['requests'],total_blocks=sum(x['blocks'] for x in audit['cases'])+audit['cases'][0]['blocks'])
 (out/'independent_audit.json').write_text(json.dumps(final,indent=2)+'\n')
 print(json.dumps(dict(status=final['status'],rows=rows,macro_depth256=final['macro_depth256']),indent=2))
if __name__=='__main__':main()
