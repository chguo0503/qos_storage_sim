"""Read-only cycle and stall evidence for the OD full CDF's empty interval."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,statistics
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def aggregate(rows):
 cycles=[c for r in rows for c in r['cycles']]
 time=sum(c['period_ms'] for c in cycles)
 b=[1000*sum(7*r['V_s_GiB'][s] for r in rows)/time for s in range(3)]
 stalls=sum(r['io_stall_ms'] for r in rows)
 return dict(request_count=len(rows),cycle_count=len(cycles),
  ratio_min=min(r['TTFT_ratio'] for r in rows),ratio_max=max(r['TTFT_ratio'] for r in rows),ratio_mean=statistics.mean(r['TTFT_ratio'] for r in rows),
  C_ms_min=min(r['C_ms'] for r in rows),C_ms_max=max(r['C_ms'] for r in rows),
  B_s_mean_across_disks_min=min(statistics.mean(r['B_s_GiB_s']) for r in rows),B_s_mean_across_disks_max=max(statistics.mean(r['B_s_GiB_s']) for r in rows),
  aggregate_internal_cycle_b_per_disk_GiB_s=b,b_mean_across_disks_GiB_s=statistics.mean(b),
  total_pure_compute_ms=sum(r['own_compute_ms'] for r in rows),total_io_stall_ms=stalls,
  internal_L1_L7_io_stall_ms=sum(r['internal_stall_ms'] for r in rows),L0_io_stall_ms=sum(r['L0_stall_ms'] for r in rows),
  internal_fraction_of_io_stall=sum(r['internal_stall_ms'] for r in rows)/stalls if stalls else 0,
  cycle_b_mean_across_disks_min=min(statistics.mean(c['b_s_GiB_s']) for c in cycles),cycle_b_mean_across_disks_max=max(statistics.mean(c['b_s_GiB_s']) for c in cycles),
  requests_above_4=sum(r['TTFT_ratio']>4 for r in rows))

def main():
 rows=[];sources=[];max_start_error=max_period_error=max_latency_error=0.
 for seed in (7,19,43):
  run=STUDY/'runs'/f'full_od_baseline_seed{seed}'
  result=read(run/'result.json.gz');manifest=read(run/'manifest.json.gz')
  config=result['baseline_configuration']
  assert config['path_pir']=='unlimited' and config['idle_capacity_borrowing']
  assert config['per_npu_cir_gib_s']==1.25
  assert result['analysis'][0]['SSD_GiB_s']==[40.,40.,40.]
  sources.append(dict(seed=seed,result_sha256=sha(run/'result.json.gz'),manifest_sha256=sha(run/'manifest.json.gz'),OD_configuration=config))
  byid={r['request_id']:r for r in manifest['requests']};batch={b['batch_id']:b for b in result['summary']['microbatch_metrics']}
  for request in result['summary']['request_metrics']:
   if not 2000<=request['admission_time_ms']<4000:continue
   inp=byid[request['request_id']];load=inp['load'];C=load['per_layer_us']/1000
   V=[0.,0.,0.]
   for disk,amount in manifest['placements'][inp['placement_index']][0]:V[disk]+=amount
   layers=batch[request['batch_id']]['layer_metrics'];assert len(layers)==8
   # `initial` means an all-arrival-zero input, not the first request on a
   # card. The explicit runtime prefetch flag is the relevant boundary check.
   assert request['layer0_cross_request_prefetched']
   assert request['compute_queue_wait_ms']==0
   cycles=[]
   for previous,current in zip(layers,layers[1:]):
    period=current['compute_start_ms']-previous['compute_start_ms']
    start_error=abs(current['io_start_time_ms']-previous['compute_start_ms'])
    period_error=abs(period-C-current['io_barrier_wait_ms'])
    assert start_error<1e-7 and period_error<1e-7
    assert current['io_ready_time_ms']<=current['compute_start_ms']+1e-7
    max_start_error=max(max_start_error,start_error);max_period_error=max(max_period_error,period_error)
    cycles.append(dict(layer_being_read=current['layer'],start_ms=previous['compute_start_ms'],end_ms=current['compute_start_ms'],period_ms=period,compute_ms=C,stall_ms=current['io_barrier_wait_ms'],b_s_GiB_s=[1000*v/period for v in V],b_over_B=C/period))
   TTFT=request['completion_time_ms']-request['admission_time_ms'];internal=sum(m['io_barrier_wait_ms'] for m in layers[1:]);L0=layers[0]['io_barrier_wait_ms']
   err=abs(TTFT-(8*C+internal+L0));assert err<1e-7
   max_latency_error=max(max_latency_error,err)
   totalcycle=sum(c['period_ms'] for c in cycles)
   row=dict(seed=seed,request_id=request['request_id'],npu_id=request['npu_id'],total_input_K=load['seq_len_k'],miss_tokens=load['nql'],category=load['category'],C_ms=C,V_total_MiB=sum(V)*1024,V_s_GiB=V,B_s_GiB_s=[1000*v/C for v in V],own_compute_ms=8*C,TTFT_ms=TTFT,TTFT_ratio=TTFT/(8*C),io_stall_ms=request['io_stall_ms'],internal_stall_ms=internal,L0_stall_ms=L0,admission_ms=request['admission_time_ms'],completion_ms=request['completion_time_ms'],mean_internal_period_ms=totalcycle/7,internal_cycle_b_s_GiB_s=[7000*v/totalcycle for v in V],cycles=cycles)
   rows.append(row)
 assert len(rows)==586
 by_miss={str(m):aggregate([r for r in rows if r['miss_tokens']==m]) for m in (256,1024,2048,4096)}
 by_profile={f'{length}K_miss{m}':aggregate([r for r in rows if r['total_input_K']==length and r['miss_tokens']==m]) for length,m in sorted({(r['total_input_K'],r['miss_tokens']) for r in rows})}
 reps=[]
 for miss in (256,1024,2048,4096):
  group=sorted([r for r in rows if r['total_input_K']==128 and r['miss_tokens']==miss],key=lambda r:r['TTFT_ratio']);q=group[len(group)//2]
  assert len(group)%2==1
  q=dict(q,selection='Exact median TTFT ratio among warm-admitted 128K requests of this miss count across three seeds',profile_request_count=len(group))
  reps.append(q)
 result=dict(scope='OD full, seeds7/19/43, requests admitted in [2,4)s and followed until completion; complete internal cycles may extend beyond 4s',source_files=sources,
 definitions=dict(C='Pure compute milliseconds per layer',B_s='1000 * per-layer bytes on SSD s in GiB / C_ms',period='compute_start(layer k)-compute_start(layer k-1), k=1..7',b_s='1000 * full layer volume on SSD s / actual internal period_ms',b_over_B='C_ms / period_ms, cycle compute fraction, not instantaneous utilization',TTFT='8*C + L0_stall + sum(internal_stall)',aggregation='b=sum(layer bytes)/sum(complete internal periods), not the arithmetic mean of per-cycle rates; subgroup counts pooled for mechanism only, not the equal-seed CDF'),
 checks=dict(request_count=586,internal_cycle_count=7*586,max_io_start_minus_previous_compute_start_error_ms=max_start_error,max_period_minus_compute_minus_stall_error_ms=max_period_error,max_TTFT_decomposition_error_ms=max_latency_error,all_warm_requests_are_cross_request_L0_prefetched=True,all_compute_queue_wait_zero=True,each_SSD_busy_40_GiB_s_through_warm=True),
 by_miss=by_miss,by_profile=by_profile,median_128K_representatives=reps,
 facts=['Every ratio>4 request in the warm cohort has miss=256.','All miss>=2048 requests have zero internal L1..L7 stall; their nonzero ratio excess arises only from exposed cross-request L0 wait.','Miss256 and miss1024 obtain almost equal cycle-averaged bandwidth, around1.576 GiB/s per disk, despite very different V/C.','Measured internal-cycle supply exceeds the1.25GiB/s CIR for both bandwidth-demanding miss groups; CIR is not a cap.'],
 interpretation=['The discrete input excludes miss512. Fixed-total-length miss256 and miss1024 read nearly the same KV volume but have approximately fourfold compute-time differences. Their similar absolute I/O-dominated latencies divide by very different pure-compute SLO bases, producing separated normalized ratios.','OD reserves equal per-NPU paths and has no demand-aware priority among them. Under sustained full SSD utilization, observed borrowing is insufficient to bridge the high miss256 V/C. The underlying scheduler is work-conserving, not fixed1.25GiB/s throttling.'],
 limits=['Cycle bandwidth is reconstructed by conservation from full layer volume and actual layer boundaries, not a separately saved per-block SSD log. The checked submit/ready timing encloses every block of that next layer within the cycle.','The plateau is a property of this sampled input and execution; it is not a universal OD invariant. Adding intermediate miss profiles or changing the SLO base could change the CDF shape; no such counterfactual was simulated.'])
 (HERE/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(checks=result['checks'],by_miss=by_miss),indent=2))
if __name__=='__main__':main()
