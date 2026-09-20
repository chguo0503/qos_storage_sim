"""Constrained random ordering of unchanged data/placement populations."""
from pathlib import Path
import argparse,random,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'runtime'))
from inputs.manifest import load_manifest,save_manifest
from simulator.core import continuous_batch_sim as core
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);a=ap.parse_args();base,meta=load_manifest(HERE/'inputs'/f'{a.base}.json.gz');out=[];sequences=[]
for n in range(32):
 pools={role:[q for q in base if q.npu_id==n and q.load['role']==role] for role in ('A','B')};assert len(pools['A'])==len(pools['B']);rng=random.Random(7+n);roles=[]
 for _ in range(len(pools['A'])):roles.extend(['A','B'] if rng.random()<.5 else ['B','A'])
 seen=dict(A=0,B=0)
 for j,role in enumerate(roles):
  q=pools[role][seen[role]];seen[role]+=1;rid=n*1000000+j
  out.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,dict(q.load,request_id=rid),q.placement))
 sequences.append(''.join(roles))
assert sorted((q.npu_id,q.load['original_request_id'],q.load['role'],q.placement) for q in out)==sorted((q.npu_id,q.load['original_request_id'],q.load['role'],q.placement) for q in base)
label=a.base+'_pair_random';meta=dict(meta,label=label,order='constrained_random_independent_AB_BA_pairs',randomization='Each NPU uses Random(7+npu_id); each pair independently chooses AB or BA with equal probability; role occurrence consumes its original identity/placement pool',reordered_from=a.base,all_request_profiles_and_original_placements_preserved=True,maximum_same_role_run=2,per_npu_role_sequences=sequences,queue_per_card=None)
save_manifest(HERE/'inputs'/f'{label}.json.gz',out,meta);print(label,len(out),meta['pure_compute_ms_per_card'])
