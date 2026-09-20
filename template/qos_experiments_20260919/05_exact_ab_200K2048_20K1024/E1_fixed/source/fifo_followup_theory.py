"""Read-only analytical follow-up to strict_extended; not a native search.

This reports measured accounting and explicitly conditional toy calculations.
No universal utilization bound or phase-causality claim is made.
"""
from pathlib import Path
import gzip,json,math,statistics
from fast_multitype_probe import profile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914'
CASE=ROOT/'results/random_multitype_search_20260914/screen/strict_extended_seed7_fifo'
def read(path):
    return json.load(gzip.open(path,'rt') if str(path).endswith('.gz') else open(path))
def quantile(xs,p):
    xs=sorted(xs)
    if not xs:return None
    i=(len(xs)-1)*p;a=math.floor(i);b=math.ceil(i)
    return xs[a]*(b-i)+xs[b]*(i-a) if a!=b else xs[a]
def stats(xs):
    return dict(n=len(xs),mean=statistics.mean(xs) if xs else None,
                p50=quantile(xs,.5),p90=quantile(xs,.9),p99=quantile(xs,.99),max=max(xs) if xs else None)
def main():
    manifest=read(CASE/'manifest.json.gz');r=read(CASE/'result.json.gz');metrics=read(CASE/'metrics.json')
    loads={q['request_id']:q['load'] for q in manifest['requests']}
    groups=metrics['by_profile_group'];cs=groups['g0']['compute_ms'];cl=groups['g1']['compute_ms'];f=cs/(cs+cl)
    ul=groups['g1']['U_percent']/100
    out=dict(source_case=str(CASE),scope='Numerical accounting plus conditional explanatory calculations; not an impossibility proof.')
    out['accounting']=dict(short_request_count_fraction=8/9,short_nominal_compute_fraction=8*profile(32,1280)[0]/(8*profile(32,1280)[0]+profile(200,2048)[0]),
        short_warm_compute_fraction=f,short_warm_active_fraction=groups['g0']['active_ms']/16000,
        warm_stall_card_ms=16000-cs-cl,stall_for_90_percent_card_ms=1600,
        stall_increase_factor_for_90=1600/(16000-cs-cl),
        required_short_U_for_90_percent_with_current_compute_mix_and_long_U=100*f/(1/.9-(1-f)/ul))
    out['profiles']=[]
    for k,q in [(32,1280),(200,2048),(32,1024),(32,768),(32,128),(200,256)]:
        c,v=profile(k,q)
        out['profiles'].append(dict(length_k=k,nql=q,compute_ms=c,read_gib=v,demand_gib_s=v/c*1000,
            one_ssu_service_ms=v/40*1000,npu_link_service_ms=v/50*1000))
    sc,sv=profile(32,1280);lc,lv=profile(200,2048)
    out['conditional_storage_scaling']=[dict(ssu=s,equal_stripe_long_disk_occupation_ms=lv/(40*s)*1000,
        long_receiver_lower_ms=lv/50*1000,long_solo_lower_ms=max(lv/(40*s),lv/50)*1000,
        maximum_whole_long_jobs_ahead_plus_short_equal_stripe_service_ms=(7*lv+sv)/(40*s)*1000,
        note='Equal stripes, whole-layer FIFO, at most one read outstanding per other NPU; not an exact ring/native bound.') for s in (1,2,4,8)]
    out['layer_stats']={};long_releases=[]
    for role in ('S','L'):
        for kind in ('internal','l0'):
            elapsed=[];waits=[];slacks=[]
            for b in r['summary']['microbatch_metrics']:
                load=loads[b['member_request_ids'][0]]
                if load['role']!=role:continue
                for l in b['layer_metrics']:
                    if (l['layer']==0)!=(kind=='l0') or not 2000<=l['compute_start_ms']<4000:continue
                    elapsed.append(l['io_ready_time_ms']-l['io_start_time_ms']);waits.append(l['io_barrier_wait_ms'])
                    if kind=='internal':
                        slacks.append(l['compute_duration_ms']-elapsed[-1])
                        if role=='L':long_releases.append(l['io_start_time_ms'])
            out['layer_stats'][role+'_'+kind]=dict(read_elapsed_ms=stats(elapsed),stall_ms=stats(waits),
                stalled_layers=sum(x>1e-7 for x in waits),deadline_slack_ms=stats(slacks))
    long_releases.sort()
    out['long_internal_release_interarrival_ms']=stats([b-a for a,b in zip(long_releases,long_releases[1:])])
    # Normalized long-compute phases measure spreading only, not causality.
    phase_events=[]
    for b in r['summary']['microbatch_metrics']:
        if loads[b['member_request_ids'][0]]['role']!='L':continue
        for l in b['layer_metrics']:phase_events.append((l['compute_start_ms'],l['compute_end_ms']))
    circular=[];active_counts=[]
    for t in range(2000,4000):
        phases=[2*math.pi*(t-a)/(b-a) for a,b in phase_events if a<=t<b]
        if len(phases)>=3:
            circular.append(abs(sum(complex(math.cos(x),math.sin(x)) for x in phases))/len(phases));active_counts.append(len(phases))
    out['long_compute_phase_diagnostic']=dict(sample_step_ms=1,only_samples_with_at_least_3_long_computes=True,
        circular_resultant_length=stats(circular),number_long_computing=stats(active_counts),
        interpretation='Resultant 1 means aligned normalized compute phase; near 0 means spread. Observational and not a causal ablation.')
    transitions=[];at_l7=0
    for b in r['summary']['microbatch_metrics']:
        if loads[b['member_request_ids'][0]]['role']!='S':continue
        layers=sorted(b['layer_metrics'],key=lambda x:x['layer'])
        for i,l in enumerate(layers):
            if l['layer']==0 or not 2000<=l['compute_start_ms']<4000 or l['io_barrier_wait_ms']<=1e-7:continue
            if i+1==len(layers):at_l7+=1;continue
            z=layers[i+1]
            transitions.append(dict(npu=b['npu_id'],rid=b['member_request_ids'][0],layer=l['layer'],stall_ms=l['io_barrier_wait_ms'],
                next_stall_ms=z['io_barrier_wait_ms'],next_compute_start_ms=z['compute_start_ms']))
    recovered=sum(x['next_stall_ms']<=1e-7 for x in transitions)
    recovery=dict(scope='Anchor is short internal layer compute start in [2000,4000). Next layer must be same request, may start outside warm; L7 excluded from successor count. Native trace; descriptive, not causal.',
        counts=dict(warm_internal_stalled=len(transitions)+at_l7,at_L7_no_next_same_request_layer=at_l7,
            with_next_same_request_layer=len(transitions),next_recovered=recovered,next_stalled=len(transitions)-recovered),transitions=transitions)
    out['next_layer_recovery']=dict(scope=recovery['scope'],counts=recovery['counts'])
    screen=read(ROOT/'results/boundary_exempt_underload_20260914/boundary_proxy_screen.json')['rows']
    accepted=[x for x in screen if x['passed'] and x['all_active_warm'] and x['all_groups_every_npu_warm']]
    causal=[x for x in accepted if x['metrics']['warm']['boundary_hard_lower_ms']<1e-8]
    out['prior_finite_screen']=dict(evaluated=len(screen),accepted=len(accepted),accepted_zero_warm_boundary_solo_lower=len(causal),
        lowest_zero_warm_boundary_solo_lower_proxy_U=min(x['U_percent'] for x in causal),
        warning='Proxy-only, finite candidate set, no proof that lower native U cannot exist.')
    out['metric_caveats']=[
        'Current V/C is charged during request stalls, so it does not simply set demand to zero for a stalled card. It still depends on the current request and policy-dependent admission/completion times.',
        'Continuing prefetch Vnext/Ccurrent is charged only on the preceding compute interval. After its deadline, unfinished read debt is absent from that reference, although physical service continues. Low continuing reference alone is not low queue pressure.',
        'Current, continuing prefetch, cross-role L0 reference, outstanding overdue bytes, and actual service answer different questions; do not add current and continuing.',
        'The measured compute table starts at32K. User-allowed20K to below32K needs explicitly labeled extrapolation sensitivity or new calibration. Keeping prior searches inside measured lengths was an implementation choice, not a user constraint.',
        'Average demand below capacity is not a deadline-schedulability proof. FIFO can miss a short deadline behind multiple long reads even if all reference demand snapshots pass.',
        'Increasing long count increases potential long blockers but reduces the compute-time share of short requests, diluting their contribution to fleet utilization loss.',
        'Multiple SSUs reduce per-disk long blocking approximately as1/S under balanced ring placement, while the per-NPU50GiB/s receive floor does not shrink. Reducing short compute below long receive time creates boundary loss that reordering alone cannot remove.'
    ]
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'theory_accounting.json';path.write_text(json.dumps(out,indent=2))
    (OUT/'theory_recovery.json').write_text(json.dumps(recovery,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
