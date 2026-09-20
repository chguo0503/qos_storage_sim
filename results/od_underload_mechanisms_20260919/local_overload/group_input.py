"""Only rearrange the frozen per-card requests; preserve their physical IDs."""
from pathlib import Path
from collections import Counter
import argparse,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'runtime'))
from inputs.manifest import load_manifest,save_manifest
from simulator.core import continuous_batch_sim as core
ap=argparse.ArgumentParser();ap.add_argument('--a-run',type=int,required=True);ap.add_argument('--b-prefix',type=int,default=0);a=ap.parse_args()
base,meta=load_manifest(HERE/'inputs/a128_b128_r12_60s.json.gz');out=[];queues=[]
for n in range(32):
 pools={r:[q for q in base if q.npu_id==n and q.load['role']==r] for r in ('A','B')};seen=Counter();roles=['B']*a.b_prefix
 remaining={r:len(pool) for r,pool in pools.items()};remaining['B']-=a.b_prefix
 while sum(remaining.values()):
  for role in ['A']*a.a_run+['B']*(2*a.a_run):
   if remaining[role]:roles.append(role);remaining[role]-=1
 for j,role in enumerate(roles):
  q=pools[role][seen[role]];seen[role]+=1;rid=n*1000000+j;load=dict(q.load,request_id=rid)
  out.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,q.placement))
 queues.append(''.join(roles))
assert sorted((q.npu_id,q.load['original_request_id'],q.load['role'],q.placement) for q in out)==sorted((q.npu_id,q.load['original_request_id'],q.load['role'],q.placement) for q in base)
label=f'a128_group{a.a_run}_{2*a.a_run}_prefix{a.b_prefix}_60s'
meta=dict(meta,label=label,order='synchronized_grouped_static_FIFO',reordered_from='a128_b128_r12_60s',all_request_profiles_and_original_placements_preserved=True,initial_B_prefix=a.b_prefix,cycle=['A']*a.a_run+['B']*(2*a.a_run),queue_per_card=list(queues[0]),input_role_sequence=queues[0],existing_prefix_preserved=False)
save_manifest(HERE/'inputs'/f'{label}.json.gz',out,meta);print(label,len(out),meta['pure_compute_ms_per_card'])
