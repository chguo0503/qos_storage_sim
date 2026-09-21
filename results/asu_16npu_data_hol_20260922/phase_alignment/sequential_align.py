#!/usr/bin/env python3
"""Freeze each earlier alignment before solving later role transitions."""
import argparse,json,sys
from pathlib import Path
from align_probe import HERE,DATA,workload,simulate as old_simulate,make_profile
from native_like_probe import simulate as native_like_simulate
sys.path.insert(0,str(HERE.parent))
from variant_runner import make_profile as official_profile,validate_config

def stable_profile(target_ms,previous):
    target=target_ms*1000;options=[]
    for miss in range(3200,4097):
        w=(miss-2048)/2048
        lo=DATA[192,2048][1]*(1-w)+DATA[192,4096][1]*w
        hi=DATA[200,2048][1]*(1-w)+DATA[200,4096][1]*w
        # Small interior margin stabilizes bytes as timing estimates move.
        if lo+50<=target<=hi-50:
            tok=round(192*1024+(target-lo)/(hi-lo)*8192)
            got=lo+(hi-lo)*(tok-192*1024)/8192
            options.append((abs(miss-previous['nql']),abs(got-target),tok,miss))
    if not options:raise ValueError(f'No interpolation candidate for {target_ms}ms')
    options.sort();_,error,tok,miss=options[0]
    return make_profile(tok,miss,'A'),error

def main():
    p=argparse.ArgumentParser();p.add_argument('--right-ms',type=float,default=60000.);p.add_argument('--rounds',type=int,default=8);p.add_argument('--native-like',action='store_true');p.add_argument('--cycles',type=int,default=5);p.add_argument('--full-input',action='store_true');p.add_argument('--initial-config',type=Path);a=p.parse_args()
    simulate=native_like_simulate if a.native_like else old_simulate
    suffix=('_full_input' if a.full_input else '')+('_native_like' if a.native_like else '')
    cat,seqs,groups=workload(cycles=a.cycles);done=set();history=[]
    if a.initial_config:
        initial=json.loads(a.initial_config.read_text());cat=initial['profiles_catalog'];seqs=initial['per_npu_sequences']
    probe_right=200000. if a.full_input else a.right_ms
    def observe():
        met,tr=simulate(cat,seqs,right=probe_right)
        if a.full_input:
            finish=[tr['ends'].get((n,len(s)-1)) for n,s in enumerate(seqs)]
            if any(v is None for v in finish):raise AssertionError('Full-input probe did not drain')
            def util(lo,hi):
                work=sum(max(0.,min(t+cat[seqs[n][r]]['compute_us']/1000,hi)-max(t,lo)) for (n,r,l),t in tr['phases'].items())
                return work/16/(hi-lo)*100
            first,last=min(finish),max(finish)
            met.update(full_input_drained=True,first_npu_finished_ms=first,last_npu_finished_ms=last,
                       U_all_busy_2s_to_first_idle=util(2000,first),U_finite_population_0_to_drain=util(0,last),
                       U_2s_to_60s=util(2000,60000),U_after_60s_to_first_idle=util(60000,first) if first>60000 else None)
            met['U']=met['U_all_busy_2s_to_first_idle']
        return met,tr
    if a.initial_config:
        initial_metrics,initial_trace=observe()
        for j,members in enumerate(groups):
            values=[initial_trace['starts'].get((n,r+1)) for n,r,k in members]
            if all(v is not None for v in values) and max(values)-min(values)<.05:done.add(j)
        (HERE/f'aligned_sequential_{int(a.right_ms)}ms{suffix}_before.json').write_text(json.dumps(initial_metrics,indent=2)+'\n')
    while True:
        met,trace=observe()
        choices=[]
        for j,members in enumerate(groups):
            if j not in done and all((n,r) in trace['starts'] for n,r,k in members):choices.append((min(trace['starts'][n,r] for n,r,k in members),j,members))
        if not choices:break
        _,j,members=min(choices);status=[]
        for iteration in range(a.rounds):
            target=max(trace['starts'][n,r] for n,r,k in members)+8*cat['A']['compute_us']/1000+.01
            updates=[]
            try:
                for n,r,k in members:
                    z,err=stable_profile((target-trace['starts'][n,r])/8,cat[k]);updates.append((k,z,err))
            except ValueError as exc:
                status.append({'iteration':iteration,'skipped':str(exc)});break
            for k,z,err in updates:cat[k]=z
            met,trace=observe()
            starts=[trace['starts'][n,r+1] for n,r,k in members if (n,r+1) in trace['starts']]
            spread=max(starts)-min(starts) if len(starts)==8 else None
            status.append({'iteration':iteration,'U':met['U'],'second_A_start_spread_ms':spread,'max_match_error_us':max(err for k,z,err in updates)})
            print('align',j,iteration,met['U'],spread,flush=True)
            if spread is None or spread<.05:break
        done.add(j);history.append(dict(group=j,members=members,steps=status))
        cfg={'name':f'aligned_sequential_{int(a.right_ms)}ms{suffix}','num_npu':16,'ssu':1,'seed':7,'horizon_ms':a.right_ms,
             'window_ms':[2000,4000],'mode':'variant_explicit',
             'profiles_catalog':{k:official_profile(v['total_tokens'],v['nql'],v['family']) for k,v in cat.items()},'per_npu_sequences':seqs}
        validate_config(cfg)
        (HERE/f'aligned_sequential_{int(a.right_ms)}ms{suffix}_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (HERE/f'aligned_sequential_{int(a.right_ms)}ms{suffix}_history.json').write_text(json.dumps(dict(history=history,metrics=met),indent=2)+'\n')
    print('FINAL',met,flush=True)

if __name__=='__main__':main()
