"""Distinct joint random input distribution: common random role order across cards.

Unique per-card profiles use untouched native_matched_decks profile pools/draws.
Only the type ordering is changed. All cards reuse a common count-preserving
random group sequence, optionally with random local swaps or persistence.
No idle insertion, truncation, or favorable-window search. Whole-deck audits.
Approximate whole-layer FIFO only; native confirmation is required.
"""
from __future__ import annotations
import argparse,collections,copy,json,random,time
from pathlib import Path
import fast_boundary_underload as fb
from fast_multitype_probe import native_matched_decks as original_decks
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914'
DECK_CACHE={}

def correlated_decks(spec,seed=7,horizon_ms=4500.):
    mode=spec['joint_distribution'];base=copy.deepcopy(spec);base.pop('joint_distribution');key=json.dumps([base,seed,horizon_ms],sort_keys=True)
    if key not in DECK_CACHE:DECK_CACHE[key]=original_decks(base,seed,horizon_ms)
    original=DECK_CACHE[key]
    counts=collections.Counter(p[0] for p in original[0])
    rng=random.Random(seed+812739)
    if mode['kind']=='common_shuffle':
        roles=[i for i,c in counts.items() for _ in range(c)];rng.shuffle(roles)
    else:
        remaining=dict(counts);roles=[];last=None
        while sum(remaining.values()):
            options=[i for i,c in remaining.items() if c]
            weights=[remaining[i]*(mode['persistence'] if i==last else 1.) for i in options]
            last=rng.choices(options,weights=weights,k=1)[0];remaining[last]-=1;roles.append(last)
    lanes=[]
    for n,lane in enumerate(original):
        local=list(roles);lrng=random.Random(seed+191113*n+41423)
        for j in range(round(mode.get('swap_fraction',0)*len(local))):
            a=lrng.randrange(len(local));b=lrng.randrange(len(local));local[a],local[b]=local[b],local[a]
        bygroup={i:iter([p for p in lane if p[0]==i]) for i in counts}
        lanes.append([next(bygroup[i]) for i in local])
    return lanes

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=float,default=160);a=ap.parse_args()
    fb.native_matched_decks=correlated_decks
    old=json.loads((ROOT/'results/boundary_exempt_underload_20260914/boundary_proxy_screen.json').read_text())['rows']
    good=sorted([x for x in old if x['passed'] and x['all_groups_every_npu_warm']],key=lambda x:x['U_percent'])
    causal=sorted([x for x in old if x['metrics']['warm']['boundary_hard_lower_ms']<1e-8],key=lambda x:(not x['passed'],x['U_percent']))
    selected=[];seen=set()
    for row in good[:12]+causal[:18]+sorted(old,key=lambda x:x['U_percent'])[:18]:
        key=json.dumps(row['spec'],sort_keys=True)
        if key not in seen:selected.append(row['spec']);seen.add(key)
    # Add nearby SSU counts for strong previously feasible choices.
    for row in good[:6]:
        for s in [max(1,row['spec']['ssu']-1),row['spec']['ssu']+1]:
            spec=copy.deepcopy(row['spec']);spec['ssu']=s;spec['name']+='_'+str(s);selected.append(spec)
    modes=[dict(kind='common_shuffle',swap_fraction=0),dict(kind='common_shuffle',swap_fraction=.1),dict(kind='common_markov',persistence=4,swap_fraction=0),dict(kind='common_markov',persistence=.1,swap_fraction=0)]
    rows=[];begin=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    for mode in modes:
      for base in selected:
        if time.monotonic()-begin>a.seconds:break
        spec=copy.deepcopy(base);spec['joint_distribution']=mode
        try:row=fb.simulate(spec,7)
        except ValueError:continue
        # Transition definition audits remain the same. Current profiles already
        # satisfy <=50 GiB/s. Same-role next transitions are checked explicitly.
        lanes=correlated_decks(spec,7);roles=[g['role'] for g in spec['groups']]
        same=[]
        for lane in lanes:
            for p,q in zip(lane,lane[1:]):
                if roles[p[0]]==roles[q[0]]:same.append(q[2]/p[1]*1000)
        row['non_boundary_npu_reference_peak_gib_s']=max(same,default=0)
        row['passed']=row['passed'] and row['non_boundary_npu_reference_peak_gib_s']<=50+1e-8
        row['eligible']=row['passed'] and row['all_groups_every_npu_warm'] and row['all_active_warm']
        rows.append(row)
        if len(rows)%10==0:print(json.dumps(dict(evaluated=len(rows),elapsed=time.monotonic()-begin,best=min([r['U_percent'] for r in rows if r['eligible']],default=None))),flush=True)
      if time.monotonic()-begin>a.seconds:break
    rows.sort(key=lambda x:(not x['eligible'],x['U_percent']))
    payload=dict(description=__doc__,joint_distribution_notice='Different cross-card joint distribution from independent random decks; no same-input comparison is implied.',evaluated=len(rows),elapsed_seconds=time.monotonic()-begin,rows=rows)
    (OUT/'correlated_proxy_screen.json').write_text(json.dumps(payload,indent=2))
    best=[r for r in rows if r['eligible']][:6]
    exact=[]
    for r in best:
        exact.append(dict(spec=r['spec'],seed=r['seed'],U_percent=r['U_percent'],boundary_hard_lower_ms=r['metrics']['warm']['boundary_hard_lower_ms'],decks=correlated_decks(r['spec'],r['seed'])))
    (OUT/'correlated_frozen_decks.json').write_text(json.dumps(exact,indent=2))
    print(json.dumps(dict(evaluated=len(rows),eligible=sum(x['eligible'] for x in rows),best=[dict(spec=r['spec'],U=r['U_percent'],warm=r['metrics']['warm'],full=r['checks']['full']) for r in best])),flush=True)
if __name__=='__main__':main()
