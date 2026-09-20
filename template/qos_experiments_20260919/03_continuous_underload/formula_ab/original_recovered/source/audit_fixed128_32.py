"""Read-only input and deadline-capacity audit for exact 128K/32K experiments."""
import argparse,csv,gzip,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def audit(case):
    man=read(case/'manifest.json.gz');r=read(case/'result.json.gz');m=read(case/'metadata.json');metric=read(case/'metrics.json')
    qs={q['request_id']:q for q in man['requests']};S=m['num_ssu'];N=m['num_npu'];left,right=metric['window_ms']
    volumes={}
    for rid,q in qs.items():
        placement=man['placements'][q['placement_index']][0]
        volumes[rid]=[math.fsum(float(size) for disk,size in placement if int(disk)==s) for s in range(S)]
        assert q['load']['total_tokens'] in [32*1024,128*1024]
        assert q['load']['ssd_prefix_tokens']+q['load']['nql']==q['load']['total_tokens']
        assert math.isclose(sum(volumes[rid]),q['load']['per_layer_kv_gb'],abs_tol=1e-12)
    minima=[min(q['load']['per_layer_us']/1e6 for q in qs.values() if q['npu_id']==n) for n in range(N)]
    upper=[sum(max(volumes[rid][s] for rid,q in qs.items() if q['npu_id']==n)/minima[n] for n in range(N)) for s in range(S)]
    events={left:[[] for _ in range(S)],right:[[] for _ in range(S)]};transitions=[]
    role_layer_stalls={role:{str(l):0. for l in range(8)} for role in ['S','L']}
    for n in range(N):
        batches=sorted((b for b in r['summary']['microbatch_metrics'] if b['npu_id']==n),key=lambda b:b['admission_time_ms'])
        flat=[]
        for b in batches:
            rid=b['member_request_ids'][0];old=b['admission_time_ms']
            for l in sorted(b['layer_metrics'],key=lambda l:l['layer']):
                flat.append(dict(request_id=rid,**l))
                role_layer_stalls[qs[rid]['load']['role']][str(l['layer'])]+=max(0.,min(l['compute_start_ms'],right)-max(old,left))
                old=l['compute_end_ms']
        for a,b in zip(flat,flat[1:]):
            start,end=a['compute_start_ms'],a['compute_end_ms'];C=(end-start)/1000
            assert math.isclose(start,b['io_start_time_ms'],abs_tol=1e-7)
            if min(end,right)<=max(start,left):continue
            rate=[v/C for v in volumes[b['request_id']]]
            x,y=max(start,left),min(end,right)
            for s in range(S):
                events.setdefault(x,[[] for _ in range(S)])[s].append(rate[s]);events.setdefault(y,[[] for _ in range(S)])[s].append(-rate[s])
            if a['request_id']!=b['request_id']:
                transitions.append(dict(npu_id=n,from_request_id=a['request_id'],to_request_id=b['request_id'],
                    from_role=qs[a['request_id']]['load']['role'],to_role=qs[b['request_id']]['load']['role'],
                    start_ms=start,deadline_ms=end,next_compute_start_ms=b['compute_start_ms'],
                    prefetch_rates_gib_s=rate,complete_window=left<=start<end<=right,
                    exclusive_ssu_stall_lower_bound_ms=max(0.,max(volumes[b['request_id']])/40*1000-(end-start)),
                    actual_stall_ms=max(0.,b['compute_start_ms']-end)))
    rates=[0.]*S;peaks=[0.]*S;over=[0.]*S;any_over=0.;integral=[0.]*S
    times=sorted(events)
    for a,b in zip(times,times[1:]):
        rates=[math.fsum([rates[s],*events[a][s]]) for s in range(S)]
        for s in range(S):
            peaks[s]=max(peaks[s],rates[s]);over[s]+=(b-a) if rates[s]>40+1e-8 else 0.;integral[s]+=(b-a)*rates[s]/1000
        if max(rates)>40+1e-8:any_over+=b-a
    hard=[x for x in transitions if x['complete_window'] and x['exclusive_ssu_stall_lower_bound_ms']>1e-8]
    out=dict(case=str(case.relative_to(ROOT)),policy=metric['policy'],seed=m['seed'],num_ssu=S,window_ms=[left,right],
        nominal_static_upper_gib_s=m['per_ssu_static_upper_bound_gib_s'],nominal_static_underload=bool(m['static_underload_all_request_combinations']),
        strict_any_prefetch_combination_upper_gib_s=upper,strict_any_prefetch_combination_underload=max(upper)<40,
        realized_deadline_reference_peak_per_ssu_gib_s=peaks,realized_deadline_reference_over40_ms_per_ssu=over,
        realized_deadline_reference_any_ssu_over40_ms=any_over,realized_deadline_reference_under40_entire_window=max(peaks)<=40+1e-8,
        realized_deadline_reference_mean_per_ssu_gib_s=[v*1000/(right-left) for v in integral],
        complete_individually_impossible_prefetch_count=len(hard),sum_exclusive_ssu_unavoidable_stall_card_ms=sum(x['exclusive_ssu_stall_lower_bound_ms'] for x in hard),
        warm_stall_by_role_and_layer_card_ms=role_layer_stalls,transitions=transitions,
        all_exact_total_lengths=True,all_unique_within_npu=m['all_length_nql_unique_within_each_npu'],
        all_npus_both_roles_computed=metric['all_npus_both_roles_computed'],all_npus_active_entire_window=metric['all_active'],
        all_profiles_within_measured_grid=m['all_compute_profiles_within_measured_grid'])
    (case/'deadline_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/fixed128_32_underload_20260914');a=p.parse_args()
    rows=[]
    for file in sorted(a.root.glob('*/*/metrics.json')):
        case=file.parent;m=read(file);d=audit(case)
        row={k:m[k] for k in ['policy','seed','num_ssu','U_percent','short_U_percent','long_U_percent','slo_1p5_percent','all_active','all_npus_both_roles_computed']}
        row.update(case=str(case.relative_to(ROOT)),nominal_static_upper=max(d['nominal_static_upper_gib_s']),
            strict_prefetch_upper=max(d['strict_any_prefetch_combination_upper_gib_s']),
            realized_deadline_peak=max(d['realized_deadline_reference_peak_per_ssu_gib_s']),
            deadline_over_ms=d['realized_deadline_reference_any_ssu_over40_ms'],individually_impossible=d['complete_individually_impossible_prefetch_count'])
        rows.append(row)
    (a.root/'all_completed_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    if rows:
        with (a.root/'all_completed_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,ensure_ascii=False))
if __name__=='__main__':main()
