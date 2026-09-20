"""Compare archived ASU and paired OD layer timings without rerunning."""
from pathlib import Path
from collections import defaultdict
import gzip,hashlib,json,statistics
import numpy as np
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[2]
ARCHIVE=ROOT/'template/qos_experiments_20260919/repository_source/results/diverse_data_ssu3_l3_20260916'
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quantiles(a):return dict(zip(('min','p10','p50','p90','max'),map(float,np.percentile(a,[0,10,50,90,100]))))
def collect(result,manifest,seed,policy):
 byid={q['request_id']:q for q in manifest['requests']};batches={b['batch_id']:b for b in result['summary']['microbatch_metrics']};rows=[]
 for q in result['summary']['request_metrics']:
  inp=byid[q['request_id']];load=inp['load'];C=load['per_layer_us']/1000;V=[0.]*3
  layout=manifest['placements'][inp['placement_index']][0]
  for s,v in layout:V[s]+=v
  assert q['io_count']==8*len(layout)
  assert q['compute_queue_wait_ms']==0
  ls=batches[q['batch_id']]['layer_metrics'];cs=[]
  for prev,cur in zip(ls,ls[1:]):
   P=cur['compute_start_ms']-prev['compute_start_ms']
   assert abs(cur['io_start_time_ms']-prev['compute_start_ms'])<1e-7
   assert abs(P-C-cur['io_barrier_wait_ms'])<1e-7
   cs.append(dict(layer=cur['layer'],start_ms=prev['compute_start_ms'],end_ms=cur['compute_start_ms'],period_ms=P,io_read_latency_ms=cur['io_ready_time_ms']-cur['io_start_time_ms'],stall_ms=cur['io_barrier_wait_ms'],b_s_GiB_s=[1000*v/P for v in V]))
  rows.append(dict(seed=seed,policy=policy,request_id=q['request_id'],npu_id=q['npu_id'],total_input_K=load['seq_len_k'],miss_tokens=load['nql'],C_ms=C,V_MiB=sum(V)*1024,V_s_GiB=V,TTFT_ms=q['processing_latency_ms'],ratio=q['processing_latency_ms']/q['own_compute_ms'],admission_ms=q['admission_time_ms'],in_warm=2000<=q['admission_time_ms']<4000,own_compute_ms=q['own_compute_ms'],stall_ms=q['io_stall_ms'],internal_stall_ms=sum(c['stall_ms'] for c in cs),L0_stall_ms=ls[0]['io_barrier_wait_ms'],avg_ssd_queue_wait_ms=q['avg_ssd_queue_wait_ms'],avg_npu_link_queue_wait_ms=q['avg_npu_link_queue_wait_ms'],cycles=cs))
 return rows

def aggregate(rows):
 cs=[c for r in rows for c in r['cycles']];period=sum(c['period_ms'] for c in cs)
 return dict(request_count=len(rows),ratio=quantiles([r['ratio'] for r in rows]),io_duration_ms=quantiles([c['io_read_latency_ms'] for c in cs]),mean_io_duration_ms=statistics.mean(c['io_read_latency_ms'] for c in cs),mean_period_ms=period/len(cs),mean_layer_stall_ms=statistics.mean(c['stall_ms'] for c in cs),requests_with_internal_stall=sum(r['internal_stall_ms']>1e-8 for r in rows),aggregate_b_s_GiB_s=[7000*sum(r['V_s_GiB'][s] for r in rows)/period for s in range(3)],internal_stall_total_ms=sum(r['internal_stall_ms'] for r in rows),mean_request_average_ssd_queue_wait_ms=statistics.mean(r['avg_ssd_queue_wait_ms'] for r in rows),mean_request_average_npu_link_queue_wait_ms=statistics.mean(r['avg_npu_link_queue_wait_ms'] for r in rows))

def main():
 rows=[];sources=[]
 for seed in (7,19,43):
  mp=STUDY/'inputs'/f'full_seed{seed}.json.gz';manifest=read(mp)
  ap=next((ARCHIVE/'runs').glob(f'full_baseline_seed{seed}_*/result.json.gz'));op=STUDY/'runs'/f'full_od_baseline_seed{seed}'/'result.json.gz'
  a=read(ap);o=read(op)
  assert a['input_fingerprint']==o['input_fingerprint']==manifest['input_fingerprint']
  assert all(v==40 for v in a['analysis'][0]['SSD_GiB_s'])
  assert all(v==40 for v in o['analysis'][0]['SSD_GiB_s'])
  for policy,result in (('asu_baseline',a),('od_baseline',o)):rows.extend(collect(result,manifest,seed,policy))
  sources.append(dict(seed=seed,asu_path=str(ap.relative_to(ROOT)),asu_sha256=sha(ap),od_path=str(op.relative_to(ROOT)),od_sha256=sha(op),manifest_sha256=sha(mp),exact_same_input_fingerprint=manifest['input_fingerprint']))
 keyed={(r['policy'],r['seed'],r['request_id']):r for r in rows}
 warm=[r for r in rows if r['in_warm']]
 shared={p:{(r['seed'],r['request_id']) for r in warm if r['policy']==p} for p in ('asu_baseline','od_baseline')}
 intersection=shared['asu_baseline']&shared['od_baseline']
 groups={f'{p}_miss{m}':aggregate([r for r in warm if r['policy']==p and r['miss_tokens']==m]) for p in shared for m in (256,1024,2048,4096)}
 profiles={};matched_profiles={}
 for L,M in sorted({(r['total_input_K'],r['miss_tokens']) for r in warm}):
  key=f'{L}K_miss{M}';profiles[key]={};matched_profiles[key]={}
  for p in shared:
   profiles[key][p]=aggregate([r for r in warm if r['policy']==p and (r['total_input_K'],r['miss_tokens'])==(L,M)])
   rr=[r for r in warm if r['policy']==p and (r['total_input_K'],r['miss_tokens'])==(L,M) and (r['seed'],r['request_id']) in intersection]
   matched_profiles[key][p]=aggregate(rr) if rr else None
 pairs=[]
 for seed,rid,reason in ((7,18000011,'previous OD median128K miss256 sample'),(19,20000009,'previous OD median128K miss1024 sample'),(19,19000008,'middle ASU ratio among64K/miss1024 shared-warm request IDs'),(43,16000007,'middle ASU ratio among32K/miss2048 shared-warm request IDs'),(43,12000008,'middle ASU ratio among200K/miss256 shared-warm request IDs')):
  a=keyed['asu_baseline',seed,rid];o=keyed['od_baseline',seed,rid]
  assert a['in_warm'] and o['in_warm']
  pairs.append(dict(seed=seed,request_id=rid,selection=reason,same_load_and_placement=True,ASU=a,OD=o))
 out=dict(status='complete',sources=sources,cohorts=dict(asu_warm_requests=len(shared['asu_baseline']),od_warm_requests=len(shared['od_baseline']),both_warm_same_request_ids=len(intersection),ASU_only=len(shared['asu_baseline']-intersection),OD_only=len(shared['od_baseline']-intersection)),
  definitions=dict(scope='Policy-specific warm admission cohorts are descriptive only. Matched examples and intersection tables control the exact seed/request ID, frozen load and placement; execution admission times and competing traffic can still shift as a consequence of policy.',io_duration='io_ready_time_ms-io_start_time_ms: includes SSD queue, own SSD service and NPU-link time',period='max(compute C, next-layer IO duration) for internal next-layer prefetch; verified by actual layer metric identities',cycle_b='layer bytes / full actual layer cycle; interval may extend beyond warm endpoint',queue_wait='request metric averages over all blocks; not summable into critical-path TTFT'),
  per_miss_policy_warm=groups,per_profile_policy_warm=profiles,per_profile_same_id_warm_intersection=matched_profiles,matched_request_examples=pairs,
  facts=['ASU supplies only one FIFO Path0 per SSD; OD has32 independent per-NPU paths with equal1.25GiB/s CIR and unlimited PIR/borrowing.','Every I/O command here is176KiB; a large layer is many equal-size commands, not one large nonpreemptive command.','ASU small-read internal I/O duration is approximately the same30-35ms as large-read I/O duration, whereas OD duration changes strongly with own volume.','Some same-ID small-volume requests have large internal stalls underASU and zero internal stalls underOD; some large-volume requests improve underASU instead.'],
  interpretation=['ASU common FIFO exposes requests to a shared backlog, so a small request cannot bypass already queued commands from other cards. The per-layer enqueue burst is far shorter than the observed queue wait. This makes layer completion time much less sensitive to its own volume and yields unequal achieved bandwidth by profile.','Mixing six total lengths then fills the gap: smaller-read miss1024/2048 profiles move right, while large-read miss256 profiles move left. This is more specific than sayingASU simply adds random noise.','OD isolation prevents otherNPUs from filling this request dedicated queue, but all paths still compete for the same disks. Equal-CIR service makes layer duration more dependent on ownV, preserving the miss-group separation in this input.'],
  limitations=['No block-order trace was saved. Layer timings and per-request block queue averages cannot identify the exact predecessor request causing each stall. FIFO source semantics plus queue statistics support the mechanism, but the historical data cannot assign stall milliseconds to a named competing request.','ASU NPU-link queue times are also nonzero, especially for large reads; the analysis must not attribute every io_start-to-ready millisecond toSSD FIFO.','Even with same input IDs, policy changes phase and competing traffic. No counterfactual was simulated with an identical instantaneous backlog held fixed.'])
 (HERE/'asu_mechanism.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out['cohorts'],indent=2))
 for k in ('32K_miss2048','64K_miss1024','200K_miss256'):
  print(k,json.dumps({p:dict(count=v['request_count'],io_ms=v['mean_io_duration_ms'],median_ratio=v['ratio']['p50']) for p,v in matched_profiles[k].items()},indent=2))
if __name__=='__main__':main()
