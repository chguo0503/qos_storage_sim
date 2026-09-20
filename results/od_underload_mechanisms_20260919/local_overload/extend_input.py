"""Extend the same immutable alternating population; unchanged existing prefix."""
from pathlib import Path
import argparse,math,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runtime'))
from inputs.manifest import load_manifest,save_manifest
from simulator.core import sim,continuous_batch_sim as core
BLOCK=176*1024/2**30
ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--pure-ms',type=float,default=62000);a=ap.parse_args()
base,meta=load_manifest(HERE/'inputs'/f'{a.base}.json.gz');profiles=meta['profiles'];cycle=meta['cycle'];repeat=math.ceil(a.pure_ms/(8*sum(profiles[r]['per_layer_us']/1000 for r in cycle)))+1;queue=cycle*repeat
old={q.request_id:q for q in base};templates={q.load['role']:q for q in base};requests=[]
for n in range(32):
 for j,role in enumerate(queue):
  rid=n*1000000+j
  if rid in old:
   q=old[rid];assert q.load['role']==role
  else:
   p=profiles[role];load=dict(templates[role].load,request_id=rid,npu_id=n,original_request_id=rid)
   placement=tuple((sim.block_ring_hash_disk_id(rid,b,3),BLOCK) for b in range(p['blocks']))
   q=core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,(placement,))
  requests.append(q)
label=a.base+'_60s'
meta=dict(meta,label=label,extended_from=a.base,existing_prefix_preserved=True,queue_per_card=queue,requests_per_card=len(queue),pure_compute_ms_per_card=8*sum(profiles[r]['per_layer_us']/1000 for r in queue),expected_blocks=sum(8*len(q.placement[0]) for q in requests))
save_manifest(HERE/'inputs'/f'{label}.json.gz',requests,meta)
print(label,meta['requests_per_card'],meta['pure_compute_ms_per_card'],meta['expected_blocks'])
