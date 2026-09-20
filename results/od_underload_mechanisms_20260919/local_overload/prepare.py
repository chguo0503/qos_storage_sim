"""Six unchanged data-row profiles; unique native RingHash placement per request."""
from pathlib import Path
import ast,hashlib,math,sys,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from inputs.manifest import save_manifest,write_json
from simulator.core import continuous_batch_sim as core,sim
BLOCK=176*1024/2**30
CASES=[('a128_b128_r11',128,128,1),('a128_b128_r12',128,128,2),('a200_b200_r11',200,200,1),('a200_b200_r12',200,200,2),('a200_b32_r12',200,32,2),('a200_b32_r14',200,32,4)]
def main():
 data=ast.literal_eval((ROOT/'data').read_text());plans=[]
 for name,alen,blen,nb in CASES:
  profiles={}
  for role,key in (('A',(alen,256)),('B',(blen,4096))):
   bw,c,ttft,v=data[key];blocks=(key[0]*1024-key[1])//128
   assert math.isclose(v,blocks*BLOCK,abs_tol=1e-12)
   profiles[role]=dict(data_key=key,category=sim.classify_request(*key),per_layer_us=c,per_layer_GiB=v,blocks=blocks,B_GiB_s=v*1e6/c,raw_data_row=data[key])
  cycle=['A']+['B']*nb
  repeat=math.ceil(14000/(8*sum(profiles[r]['per_layer_us']/1000 for r in cycle)))+1
  queue=cycle*repeat;requests=[]
  for n in range(32):
   for j,role in enumerate(queue):
    p=profiles[role];rid=n*1000000+j;seq,miss=p['data_key']
    placement=tuple((sim.block_ring_hash_disk_id(rid,b,3),BLOCK) for b in range(p['blocks']))
    load=dict(request_id=rid,npu_id=n,original_request_id=rid,role=role,category=p['category'],seq_len_k=seq,nql=miss,total_tokens=seq*1024,ssd_prefix_tokens=seq*1024-miss,per_layer_us=p['per_layer_us'],per_layer_kv_gb=p['per_layer_GiB'],required_bw_input_gbps=p['B_GiB_s'],source_ttft_ms=p['raw_data_row'][2],constructed_profile=False,profile_construction=dict(method='direct_data_row',data_key=p['data_key']),arrival_time=0.,arrival_ms=0.,initial=True)
    requests.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,(placement,)))
  pure_per_card=8*sum(profiles[r]['per_layer_us']/1000 for r in queue)
  ideal_fleet=32*sum(profiles[r]['per_layer_GiB'] for r in cycle)/sum(profiles[r]['per_layer_us']/1e6 for r in cycle)
  meta=dict(label=name,seed=7,order='synchronized_cycle',num_npu=32,num_ssu=3,n_layers=8,profiles=profiles,cycle=cycle,queue_per_card=queue,requests_per_card=len(queue),pure_compute_ms_per_card=pure_per_card,input_constructed=False,arrival_ms=0.,placement='native RingHash(original_request_id,block_index,3)',block_KiB=176,expected_blocks=sum(8*len(q.placement[0]) for q in requests),source_data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),ideal_no_stall_fleet_bytes_per_compute_second_GiB_s=ideal_fleet,ideal_no_stall_mean_per_disk_GiB_s=ideal_fleet/3,ideal_no_stall_under_capacity=ideal_fleet<120,nominal_demand_definition='sum admitted request V_is/C_i; actual time weighted mean is measured separately, includes stalls')
  save_manifest(HERE/'inputs'/f'{name}.json.gz',requests,meta);plans.append(meta)
 runtime=HERE/'runtime';runtime.mkdir(exist_ok=False)
 for package in ('simulator','inputs'):shutil.copytree(ROOT/package,runtime/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(ROOT/'data',runtime/'data')
 write_json(HERE/'candidate_plan.json',plans)
 print([(r['label'],len(r['queue_per_card']),round(r['ideal_no_stall_mean_per_disk_GiB_s'],3)) for r in plans])
if __name__=='__main__':main()
