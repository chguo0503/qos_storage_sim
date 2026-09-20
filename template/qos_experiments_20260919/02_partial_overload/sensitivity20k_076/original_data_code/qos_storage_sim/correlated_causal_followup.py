"""Prespecified extra correlated grid with zero expected boundary solo floor."""
import copy,itertools,json,time
from pathlib import Path
import fast_boundary_underload as fb
from correlated_fifo_probe import correlated_decks,OUT
fb.native_matched_decks=correlated_decks
begin=time.monotonic();rows=[]
# Alternate parameters across low-correlation shuffle and positive/negative
# persistence; fixed seed, fixed windows, entire frozen deck drainage.
for i,(q,lq,w,mode) in enumerate(itertools.product((1344,1408),(2048,2304,3072,4096),(4,8,16),(dict(kind='common_shuffle',swap_fraction=0),dict(kind='common_markov',persistence=4,swap_fraction=0),dict(kind='common_markov',persistence=.1,swap_fraction=0)))):
    if time.monotonic()-begin>65:break
    spec=fb.make_spec(i,1,[(32,q,w),(200,lq,1)]);spec['name']='correlated_causal_'+str(i);spec['joint_distribution']=mode
    r=fb.simulate(spec,7);r['eligible']=r['passed'] and r['all_groups_every_npu_warm'] and r['all_active_warm'];rows.append(r)
    if len(rows)%10==0:print(json.dumps(dict(count=len(rows),elapsed=time.monotonic()-begin,best=min((x['U_percent'] for x in rows if x['eligible']),default=None))),flush=True)
rows.sort(key=lambda x:(not x['eligible'],x['U_percent']))
(OUT/'correlated_causal_screen.json').write_text(json.dumps(dict(description=__doc__,elapsed_seconds=time.monotonic()-begin,rows=rows),indent=2))
best=[r for r in rows if r['eligible']][:6]
(OUT/'correlated_causal_frozen_decks.json').write_text(json.dumps([dict(spec=r['spec'],seed=r['seed'],U_percent=r['U_percent'],decks=correlated_decks(r['spec'],7)) for r in best],indent=2))
print(json.dumps(dict(evaluated=len(rows),eligible=len([r for r in rows if r['eligible']]),best=[dict(spec=r['spec'],U=r['U_percent'],warm=r['metrics']['warm']) for r in best])),flush=True)
