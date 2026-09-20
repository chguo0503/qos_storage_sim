"""Bounded independent followup; unchanged matched builder / FIFO surrogate."""
from __future__ import annotations
import argparse, json, random, time, math
from pathlib import Path
from fast_multitype_probe import simulate as warm_sim, profile
from fast_boundary_underload import simulate as full_sim
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914';OUT.mkdir(exist_ok=True)

def make(i,s,triples):
 return dict(name=f'lower_agent_{i:05d}',ssu=s,groups=[dict(id=f'g{j}',role='S' if k<=64 else 'L',total_k=k,nql=q,weight=w)for j,(k,q,w)in enumerate(triples)])

def candidates():
 rng=random.Random(947713); seen=set();i=0
 # Single disk: rare short-compute reads among large low-demand jobs.
 for ss in (1,2,3,4,6,8):
  for sq in (128,192,256,384,512,640,768,896,1024,1152,1280):
   for lq in (768,1024,1536,2048,3072,4096):
    for sw in (1,2,3,4,6,8,12,16,24,32):
     tri=[(32,sq,sw),(200,lq,1)]; i+=1
     yield make(i,ss,tri)
 # Three/four groups genuine length-based S/L groups; no labels for exemptions.
 for n in range(7000):
  ss=rng.choice((1,1,1,2,2,3,4,6,8,10))
  tri=[(rng.choice((32,32,32,48,64)),rng.choice((128,256,384,512,640,768,896,1024,1152,1280,1536)),rng.choice((1,2,4,6,8,12,16,24))),
       (rng.choice((96,128,160,200,200)),rng.choice((128,256,512,768,1024,1536,2048,2560,3072,4096)),rng.choice((1,1,1,2,3)))]
  if n%5!=0:tri.insert(1,(32,rng.choice((128,256,512,1024,2048,3072,4096)),rng.choice((1,2,4,8))))
  if n%4==0:tri.append((rng.choice((96,128,160,200)),rng.choice((512,1024,2048,4096)),rng.choice((1,2,4))))
  key=(ss,tuple(tri))
  if key in seen:continue
  seen.add(key);i+=1;yield make(i,ss,tri)

def main():
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=280);args=p.parse_args()
 begin=time.perf_counter();rows=[];full=[];considered=0
 # Reserve approximately90secs for complete drainage audits and multi-seed confirmation.
 for spec in candidates():
  if time.perf_counter()-begin>args.seconds*.58:break
  gs=spec['groups']; ss=spec['ssu'];C=sum(g['weight']*profile(g['total_k'],g['nql'])[0]for g in gs);V=sum(g['weight']*profile(g['total_k'],g['nql'])[1]for g in gs)
  mean=8*V/C*1000/(40*ss)
  if not .2<mean<.92:continue
  triples=[(g['total_k'],g['nql'],g['weight'])for g in gs]
  try:r=warm_sim(triples,7,n_ssu=ss,matched_spec=spec,enforce_npu_link=True)
  except ValueError:continue
  considered+=1
  # Upper bound prevents gross impossible cases but permits absent combinations.
  if r['utilization_pct']<99 and r['all_lanes_all_roles_warm'] and r['nominal_max_current_load']<2.0:
   rows.append(dict(spec=spec,seed=7,U=r['utilization_pct'],warm=r))
 rows.sort(key=lambda r:r['U']);(OUT/'search_agent_warm.json').write_text(json.dumps(dict(count=considered,rows=rows),indent=2))
 print(json.dumps(dict(event='warm_done',count=considered,candidates=len(rows),elapsed=time.perf_counter()-begin,best=rows[:2])),flush=True)
 # Audit diverse specs: pick each disk/count regime plus lowU top rows.
 selected=[]; buckets={}
 for row in rows:
  spec=row['spec'];key=(spec['ssu'],len(spec['groups']))
  buckets.setdefault(key,[]).append(row)
 for rank in range(30):
  for key,rs in buckets.items():
   if rank<len(rs):selected.append(rs[rank])
 for row in selected:
  if time.perf_counter()-begin>args.seconds*.87:break
  for seed in (7,23):
   try:r=full_sim(row['spec'],seed)
   except ValueError:continue
   full.append(r)
   if r['passed'] and r['all_active_warm'] and r['all_groups_every_npu_warm']:
    print(json.dumps(dict(event='pass',spec=r['spec'],seed=seed,U=r['U_percent'],internal=r['metrics']['warm']['internal_stall_ms'],boundary_hard=r['metrics']['warm']['boundary_hard_lower_ms'])),flush=True)
 good=[r for r in full if r['passed'] and r['all_active_warm'] and r['all_groups_every_npu_warm']]
 good.sort(key=lambda r:r['U_percent'])
 zero=sorted(good,key=lambda r:(r['metrics']['warm']['boundary_hard_lower_ms']>1e-8,r['U_percent']))
 # Multiple fixed independently shuffled seeds for promising valid candidates.
 specs=[]
 for r in good[:4]+zero[:4]:
  if r['spec']not in specs:specs.append(r['spec'])
 for spec in specs:
  for seed in (1,3,11,19,31,41,59):
   if time.perf_counter()-begin>args.seconds:break
   try:full.append(full_sim(spec,seed))
   except ValueError:continue
 good=[r for r in full if r['passed'] and r['all_active_warm'] and r['all_groups_every_npu_warm']];good.sort(key=lambda r:r['U_percent'])
 payload=dict(description=__doc__,seconds=time.perf_counter()-begin,warm_count=considered,full_count=len(full),valid_count=len(good),constraint_windows=['full','0-4500ms','2000-4000ms'],additional_native_gate='Also audit500-4000ms window and actual ring placement; layer-at-once surrogate only.',good=good,rows=full)
 (OUT/'search_agent_full.json').write_text(json.dumps(payload,indent=2))
 print(json.dumps(dict(event='done',seconds=payload['seconds'],warm_count=considered,full_count=len(full),valid_count=len(good),best=[dict(spec=r['spec'],seed=r['seed'],U=r['U_percent'],metrics=r['metrics']['warm'])for r in good[:8]])),flush=True)
if __name__=='__main__':main()
