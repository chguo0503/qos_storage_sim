#!/usr/bin/env python3
"""Independent offline audit of the frozen raw176_extendedhot 20-case study.

Standard library only. No simulator or previous metric functions are imported.
Only this script's six named artifacts are written; no simulation is executed.
"""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F = HERE.parent / 'raw_role_followup_20260909'
OLD = F / 'mixed_rebinding'
SEEDS = (7, 19, 43, 67, 101)
MODES = ('random', 'ordered')
POLICIES = ('baseline', 'once')
START, END, ALPHA = 2000.0, 4000.0, 1.5
BLOCK = 176 / 1024**2
PROFILE_KEYS = ((32,1024),(48,1024),(64,1024),(176,1024))


def read(path):
    path = Path(path)
    value = path.read_bytes()
    return json.loads(gzip.decompress(value) if path.suffix == '.gz' else value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def close(a,b):
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7)


def clip(a,b):
    return max(0.0,min(END,b)-max(START,a))


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values),sample_sd=statistics.stdev(values),
                min=min(values),max=max(values))


def csv_write(name,rows):
    assert rows
    with (HERE/name).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_input(item,plan,table):
    path=Path(item['manifest']);data=read(path);meta=data['metadata'];rows=data['requests']
    placements=[tuple(tuple((int(s),float(v)) for s,v in layer) for layer in p) for p in data['placements']]
    checks=dict(manifest_sha_matches_frozen_plan=sha(path)==item['manifest_sha256'],
        dimensions=(meta['num_npu'],meta['num_ssu'],meta['n_layers'])==(32,6,8),
        population=len(rows)==meta['request_count']==1212,
        mode_seed_label=(meta['seed'],meta['label'])==(item['seed'],item['label']),
        no_C_scaling=meta['compute_scale_actual']==1.0,
        source_data_sha=meta['source_data_sha256']==plan['source_sha256']['data']==sha(ROOT/'data'),
        direct_data_row=True,raw_C_exact=True,raw_V_physical_equal=True,no_padding=True,
        all_total_input_at_least_10K=True,category=True,all_arrival_zero=True,
        order_encoded_by_ID=True,original_source_stripe=True,equal_176KiB_blocks=True)
    by_id={};by_original={};lanes=[[] for _ in range(32)];profiles={}
    fp=hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
    for row in sorted(rows,key=lambda x:x['request_id']):
        rid,npu,load=row['request_id'],row['npu_id'],row['load']
        assert rid not in by_id and 0<=npu<32
        original=load['original_request_id'];assert original not in by_original
        p=placements[row['placement_index']];assert len(p)==1
        key=(load['seq_len_k'],load['nql']);assert key in PROFILE_KEYS
        reference=table[key];c=load['per_layer_us']/1000;v=math.fsum(x for _,x in p[0])
        rates=[math.fsum(x for s,x in p[0] if s==disk)/c*1000 for disk in range(6)]
        category=('S' if key[0]<=80 else 'L')+('L' if key[1]>=512 else 'S')
        checks['direct_data_row'] &= load['constructed_profile'] is False and load['profile_construction']=={'method':'direct_data_row'}
        checks['raw_C_exact'] &= load['per_layer_us']==load['original_compute_us']==reference[1]
        checks['raw_V_physical_equal'] &= abs(v-reference[3])<1e-14 and abs(load['per_layer_kv_gb']-reference[3])<1e-14
        checks['no_padding'] &= load['padding_gib_per_layer']==0
        checks['all_total_input_at_least_10K'] &= key[0]*1024>=10*1024
        checks['category'] &= load['category']==category and load['role']==('long' if key==(176,1024) else 'short')
        checks['all_arrival_zero'] &= row['arrival_time_ms']==load['arrival_time']==load['arrival_ms']==0
        checks['order_encoded_by_ID'] &= load['request_id']==rid and load['npu_id']==npu and rid==npu*1000000+load['generation']
        checks['original_source_stripe'] &= all(s==(j+load['source_original_npu_id']//4)%6 for j,(s,_) in enumerate(p[0]))
        checks['equal_176KiB_blocks'] &= all(x==BLOCK and 0<=s<6 for layer in p for s,x in layer)
        fp.update(repr((rid,npu,float(row['arrival_time_ms']),load['category'],load['per_layer_us'],p)).encode())
        # Current NPU is preserved for the order comparison, unlike the older
        # fixed-to-mixed binding comparison. Only ID/generation encode order.
        scientific_load={k:x for k,x in load.items() if k not in ('request_id','generation')}
        identity=dict(original_request_id=original,npu_id=npu,arrival_time_ms=row['arrival_time_ms'],load=scientific_load,placement=p)
        info=dict(request_id=rid,original_request_id=original,npu=npu,generation=load['generation'],
            profile=f'{key[0]}:{key[1]}',profile_index=PROFILE_KEYS.index(key),role=load['role'],category=category,
            C_ms=c,eight_layer_C_ms=8*c,V_gib=v,rates=rates,link_rate=v/c*1000,
            identity_sha256=stable(identity),placement_sha256=stable(p),total_tokens=key[0]*1024,nql=key[1])
        by_id[rid]=info;by_original[original]=info;lanes[npu].append(info)
        if key not in profiles:
            profiles[key]=dict(seq_len_k=key[0],total_tokens=key[0]*1024,nql=key[1],ssd_prefix_tokens=key[0]*1024-key[1],
                role=load['role'],category=category,per_layer_C_ms=c,eight_layer_C_ms=8*c,
                raw_per_layer_V_gib=reference[3],physical_per_layer_V_gib=v,physical_per_layer_V_MiB=v*1024,
                physical_blocks_per_layer=len(p[0]),padding_gib_per_layer=0,direct_data_row=True,global_count=0)
        profiles[key]['global_count']+=1
    checks['input_fingerprint_recomputed']=fp.hexdigest()==data['input_fingerprint']==meta['input_fingerprint']==item['input_fingerprint']
    checks['global_profile_quota']=[profiles[k]['global_count'] for k in PROFILE_KEYS]==[264,264,264,420]
    maximum=[[max(r['rates'][s] for r in lane) for s in range(6)] for lane in lanes]
    certificate=[math.fsum(x[s] for x in maximum) for s in range(6)]
    per_npu=[]
    for n,lane in enumerate(lanes):
        lane.sort(key=lambda r:r['request_id'])
        per_npu.append(dict(npu=n,request_count=len(lane),profile_counts=[sum(r['profile_index']==p for r in lane) for p in range(4)],
            pure_C_ms=math.fsum(r['eight_layer_C_ms'] for r in lane),
            population_sha256=stable(sorted((r['original_request_id'],r['identity_sha256']) for r in lane))))
    total_C=math.fsum(r['eight_layer_C_ms'] for r in by_id.values())
    for p in profiles.values():
        p['request_fraction']=p['global_count']/1212
        p['pure_compute_fraction']=p['global_count']*p['eight_layer_C_ms']/total_C
    checks['static_certificate_matches_metadata']=all(close(a,b) for a,b in zip(certificate,meta['active_profile_rate_certificate']['per_ssu_upper_bound_gib_s']))
    return dict(label=item['label'],mode=item['mode'],seed=item['seed'],manifest=str(path),manifest_sha256=sha(path),
        input_fingerprint=fp.hexdigest(),checks=checks,audit_passed=all(checks.values()),request_count=len(by_id),
        total_pure_C_ms=total_C,minimum_per_card_pure_C_ms=min(x['pure_C_ms'] for x in per_npu),
        static_per_ssu_upper_bound_gib_s=certificate,static_max_gib_s=max(certificate),
        static_certificate_pass=max(certificate)<40,
        static_certificate_note='A >40 sufficient upper bound is inconclusive; actual event demand is independently scanned for every result.',
        per_npu=per_npu,profiles=[profiles[k] for k in PROFILE_KEYS],_by_id=by_id,_by_original=by_original)


def nominal_scan(requests,inputs):
    events=defaultdict(lambda:dict(remove=[],add=[]))
    for row in requests:
        events[row['admission_time_ms']]['add'].append(row['request_id'])
        events[row['completion_time_ms']]['remove'].append(row['request_id'])
    times=sorted(events);active={};disk_peaks=[0.0]*6;disk_over=[0.0]*6;witness=[None]*6
    any_over=0.0;link_peak=0.0;link_over=0.0;maximum_long=0;warm_peak=0.0;accounted=0.0
    for i,t in enumerate(times):
        # All removals and additions at a tied timestamp form one transition.
        # Only the resulting state over the next positive interval is scored.
        for rid in events[t]['remove']:
            n=inputs[rid]['npu'];assert active.pop(n)==rid
        for rid in events[t]['add']:
            n=inputs[rid]['npu'];assert n not in active;active[n]=rid
        if i==len(times)-1:break
        end=times[i+1];assert end>t
        values=[math.fsum(inputs[rid]['rates'][s] for rid in active.values()) for s in range(6)]
        link=max((inputs[rid]['link_rate'] for rid in active.values()),default=0.0)
        maximum_long=max(maximum_long,sum(inputs[rid]['role']=='long' for rid in active.values()))
        dt=end-t;accounted+=dt
        for s,v in enumerate(values):
            if v>disk_peaks[s]:
                disk_peaks[s]=v;witness[s]=dict(start_ms=t,end_ms=end,value_gib_s=v,
                    active_request_ids_by_npu={str(n):rid for n,rid in sorted(active.items())})
            if v>40:disk_over[s]+=dt
        if max(values)>40:any_over+=dt
        link_peak=max(link_peak,link)
        if link>50:link_over+=dt
        if max(t,START)<min(end,END):warm_peak=max(warm_peak,max(values))
    assert not active and close(accounted,times[-1]-times[0])
    return dict(method='fresh math.fsum across active requests at every exact tied event transition; no fixed-step sampling',
        definition='sum of current admitted request actual per-layer V_s/C; active interval [admission,completion); next-request L0 not added',
        event_time_count=len(times),first_event_ms=times[0],last_event_ms=times[-1],
        per_ssu_peak_gib_s=disk_peaks,max_ssu_gib_s=max(disk_peaks),per_ssu_over_40_ms=disk_over,
        any_ssu_over_40_ms=any_over,max_link_gib_s=link_peak,any_link_over_50_ms=link_over,
        warm_max_ssu_gib_s=warm_peak,max_simultaneous_long_requests=maximum_long,
        strictly_underload=max(disk_peaks)<40 and link_peak<50,peak_witnesses=witness)


def cohort(rows,inputs):
    passed=sum(r['completion_time_ms']-r['admission_time_ms']<=ALPHA*inputs[r['request_id']]['eight_layer_C_ms']+1e-9 for r in rows)
    arrival_passed=sum(r['completion_time_ms']-r['arrival_time_ms']<=ALPHA*inputs[r['request_id']]['eight_layer_C_ms']+1e-9 for r in rows)
    return dict(count=len(rows),passed=passed,rate_percent=100*passed/len(rows) if rows else None,
        arrival_clock_passed=arrival_passed,arrival_clock_rate_percent=100*arrival_passed/len(rows) if rows else None,
        completion_after_window_count=sum(r['completion_time_ms']>END for r in rows),
        request_ids=sorted(r['request_id'] for r in rows))


def audit_run(info,policy,plan,prior):
    directory=OLD/'runs'/info['label']/policy
    command=read(directory/'command.json');paths=list(directory.glob('*.json.gz'));assert len(paths)==1
    path=paths[0];raw=read(path);summary=raw['summary'];inputs=info['_by_id'];requests=summary['request_metrics']
    checks=dict(input_audit=info['audit_passed'],command_complete=command['status']=='complete' and command['returncode']==0,
        frozen_manifest=command['input_sha256']==info['manifest_sha256'],
        seed_strategy=raw['submit_seed']==info['seed'] and raw['strategy']==policy,
        fp=raw['input_fingerprint']==summary['input_fingerprint']==info['input_fingerprint'],
        dimensions=(summary['num_npu'],summary['num_ssu'],summary['n_layers'],summary['batch_size'])==(32,6,8,1),
        core_invariants=all(summary['invariants'].values()),
        preserved_execution_placement=raw['input_placement_fingerprint']==raw['execution_placement_fingerprint'],
        command_frozen_sources=command['source_sha256']==plan['source_sha256'],
        raw_frozen_core_sources=all(plan['source_sha256'][k]==v for k,v in raw['core_and_policy_sha256'].items()),
        stress_runner_source=raw['stress_runner_sha256']==plan['source_sha256']['run_baseline_npu32_stress.py'],
        single_request_batches=True,layer_times_match_raw_C=True,request_batch_identity=True,
        layer_order=True,request_time_order=True)
    extras={'run_baseline_npu32_stress.py',str((F/'mixed_design_v8.py').relative_to(ROOT)),
        str((F/'run_mixed_extendedhot.py').relative_to(ROOT)),str((F/'run_raw_followup.py').relative_to(ROOT))}
    checks['complete_core_source_keyset']=set(raw['core_and_policy_sha256'])==set(plan['source_sha256'])-extras
    by_request={r['request_id']:r for r in requests}
    checks['full_population']=len(requests)==len(by_request)==len(inputs)==1212 and set(by_request)==set(inputs)
    lanes=[dict(npu=n,compute_ms=0.,active_ms=0.,l0_stall_ms=0.,l1_7_stall_ms=0.,
        short_compute_ms=0.,short_active_ms=0.,long_compute_ms=0.,long_active_ms=0.) for n in range(32)]
    batches_seen=[];full_C=[];per_npu_requests=[[] for _ in range(32)]
    for r in requests:
        info_r=inputs[r['request_id']]
        checks['request_time_order'] &= math.isfinite(r['completion_time_ms']) and r['completion_time_ms']>=r['admission_time_ms']>=r['arrival_time_ms']==0
        checks['request_batch_identity'] &= r['npu_id']==info_r['npu'] and close(r['own_compute_ms'],info_r['eight_layer_C_ms'])
        per_npu_requests[r['npu_id']].append(r)
    for batch in summary['microbatch_metrics']:
        checks['single_request_batches'] &= batch['batch_size']==len(batch['member_request_ids'])==1
        rid=batch['member_request_ids'][0];batches_seen.append(rid);r=inputs[rid];n=batch['npu_id'];z=lanes[n]
        a,f=batch['admission_time_ms'],batch['completion_time_ms'];active=clip(a,f);role=r['role']
        checks['request_batch_identity'] &= n==r['npu'] and close(a,by_request[rid]['admission_time_ms']) and close(f,by_request[rid]['completion_time_ms'])
        z['active_ms']+=active;z[role+'_active_ms']+=active
        previous=a;layers=sorted(batch['layer_metrics'],key=lambda x:x['layer'])
        checks['layer_order'] &= [x['layer'] for x in layers]==list(range(8))
        for layer in layers:
            cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
            checks['layer_times_match_raw_C'] &= previous<=cs+1e-8 and cs<ce<=f+1e-8 and close(ce-cs,r['C_ms']) and close(cs-previous,layer['io_barrier_wait_ms'])
            value=clip(cs,ce);z['compute_ms']+=value;z[role+'_compute_ms']+=value
            z['l0_stall_ms' if layer['layer']==0 else 'l1_7_stall_ms']+=clip(previous,cs)
            full_C.append(ce-cs);previous=ce
        checks['layer_times_match_raw_C'] &= close(previous,f)
    checks['full_unique_batch_population']=len(batches_seen)==len(set(batches_seen))==1212 and set(batches_seen)==set(inputs)
    checks['full_pure_C']=close(math.fsum(full_C),info['total_pure_C_ms'])
    fourth=[];finishes=[]
    for z,rr in zip(lanes,per_npu_requests):
        rr.sort(key=lambda r:r['admission_time_ms'])
        checks['request_time_order'] &= all(a['completion_time_ms']<=b['admission_time_ms']+1e-8 for a,b in zip(rr,rr[1:]))
        fourth.append(sorted(r['completion_time_ms'] for r in rr)[3]);finishes.append(max(r['completion_time_ms'] for r in rr))
        z['fourth_completion_ms']=fourth[-1];z['final_completion_ms']=finishes[-1]
        z['device_U_percent']=100*z['compute_ms']/(END-START)
        z['idle_ms']=(END-START)-z['active_ms']
        for role in ('short','long'):
            z[role+'_pooled_U_percent']=100*z[role+'_compute_ms']/z[role+'_active_ms'] if z[role+'_active_ms'] else None
    checks['warm_time_account']=all(close(z['compute_ms']+z['l0_stall_ms']+z['l1_7_stall_ms'],z['active_ms']) for z in lanes)
    warm=[r for r in requests if START<=r['admission_time_ms']<END];warm_score=cohort(warm,inputs);whole=cohort(requests,inputs)
    by_role={role:cohort([r for r in warm if inputs[r['request_id']]['role']==role],inputs) for role in ('short','long')}
    by_category={cat:cohort([r for r in warm if inputs[r['request_id']]['category']==cat],inputs) for cat in ('SS','SL','LS','LL')}
    stored=raw['slo'];checks['stored_slo_definition']=(stored['window_start_ms'],stored['window_end_ms'],stored['alpha'])==(START,END,ALPHA)
    for name,score in [('window_admissions',warm_score),('all_requests',whole)]:
        target=stored[name]
        checks[name+'_SLO_crosscheck']=target['admission']['count']==score['count'] and target['admission']['passed']==score['passed'] and close(100*target['admission']['rate'],score['rate_percent']) and target['arrival']['passed']==score['arrival_clock_passed'] and sorted(target['request_ids'])==score['request_ids']
    u=100*math.fsum(z['compute_ms'] for z in lanes)/(32*(END-START))
    w=next(w for w in raw['windows'] if (w['start_ms'],w['end_ms'])==(START,END))
    checks['raw_window_U_crosscheck']=close(u,100*w['mean_npu_utilization']) and all(close(z['compute_ms'],c) for z,c in zip(lanes,w['compute_ms_by_npu']))
    adapter=raw['adapter_statistics'];ticks=adapter['collector_times_ms']
    checks['once_is_literal_once_5ms']=raw['strategy']==adapter['strategy']==policy and raw['collector_interval_ms']==adapter['collector_interval_ms']==5 and all(close(t,5*i) for i,t in enumerate(ticks))
    checks['fixed_no_runtime_migration']=raw['policy_config']['assignment']=='fixed' and raw['assignment_log']==[] and adapter['assignment_count']==0
    checks['static_CIR_and_no_reorder']=adapter['cir_write_events']==[] and adapter['reorder_calls']==adapter['reorder_changed']==0
    checks['zero_modeled_control_cost']=adapter['modeled_control_cpu_latency_ms']==adapter['modeled_control_communication_latency_ms']==0
    scan=nominal_scan(requests,inputs)
    metrics=dict(device_U_percent=u,warm_slo_percent=warm_score['rate_percent'],warm_slo_count=warm_score['count'],warm_slo_passed=warm_score['passed'],
        all_32_active=all(close(z['active_ms'],END-START) for z in lanes),
        mixed_card_count=sum(z['short_compute_ms']>0 and z['long_compute_ms']>0 for z in lanes),
        minimum_short_C_ms=min(z['short_compute_ms'] for z in lanes),minimum_long_C_ms=min(z['long_compute_ms'] for z in lanes),
        minimum_role_C_ms=min(min(z['short_compute_ms'],z['long_compute_ms']) for z in lanes),
        fourth_by_1500=all(t<=1500 for t in fourth),maximum_fourth_completion_ms=max(fourth),
        max_ssu_gib_s=scan['max_ssu_gib_s'],any_ssu_over_40_ms=scan['any_ssu_over_40_ms'],max_link_gib_s=scan['max_link_gib_s'],
        full_run_strict_underload=scan['strictly_underload'],static_max_ssu_gib_s=info['static_max_gib_s'],
        l0_stall_ms=math.fsum(z['l0_stall_ms'] for z in lanes),l1_7_stall_ms=math.fsum(z['l1_7_stall_ms'] for z in lanes),
        first_card_done_ms=min(finishes),makespan_ms=max(finishes))
    for role in ('short','long'):
        metrics[role+'_warm_count']=by_role[role]['count'];metrics[role+'_warm_SLO_percent']=by_role[role]['rate_percent']
        metrics[role+'_pooled_U_percent']=100*math.fsum(z[role+'_compute_ms'] for z in lanes)/math.fsum(z[role+'_active_ms'] for z in lanes)
    previous=prior[(info['label'],policy)]
    checks['previous_raw_and_input_SHA']=previous['result_sha256']==sha(path) and previous['input_sha256']==info['manifest_sha256']
    checks['previous_independent_numbers_match']=all(close(a,b) for a,b in [(u,previous['device_utilization_percent']),
        (metrics['warm_slo_percent'],previous['warm_slo_percent']),(scan['max_ssu_gib_s'],previous['max_ssu_nominal_gib_s']),
        (scan['any_ssu_over_40_ms'],previous['any_ssu_over_capacity_ms'])]) and warm_score['count']==previous['warm_admissions'] and warm_score['passed']==previous['warm_slo_passed']
    technical=all(checks.values());valid=technical and metrics['all_32_active'] and metrics['mixed_card_count']==32 and metrics['full_run_strict_underload'] and metrics['fourth_by_1500']
    return dict(label=info['label'],mode=info['mode'],seed=info['seed'],strategy=policy,audit_passed=technical,conditions_passed=valid,
        checks=checks,result_path=str(path),result_sha256=sha(path),command_sha256=sha(directory/'command.json'),
        manifest_sha256=info['manifest_sha256'],input_fingerprint=info['input_fingerprint'],core_source_sha256=raw['core_and_policy_sha256'],
        metrics=metrics,per_npu=lanes,warm_admissions=warm_score,warm_by_role=by_role,warm_by_category=by_category,
        whole_population=whole,window_arrivals=cohort([r for r in requests if START<=r['arrival_time_ms']<END],inputs),nominal_event_scan=scan)


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    table=ast.literal_eval((ROOT/'data').read_text())
    plans={};jobs={};source_before={str(Path(__file__)):sha(__file__)};source_checks={}
    for name in ('extendedhot_pilot_specs.json','extendedhot_confirmation_specs.json'):
        path=OLD/'plans'/name;plan=read(path);source_before[str(path)]=sha(path)
        source_checks[name+'::spec_hash']=sha(plan['spec_file'])==plan['spec_sha256']
        source_before[plan['spec_file']]=sha(plan['spec_file'])
        for name2,digest in plan['source_sha256'].items():
            p=ROOT/name2;source_before[str(p)]=sha(p);source_checks[name+'::'+name2+'::current']=sha(p)==digest
            archives=[OLD/'sources'/digest/p.name,F/'sources'/digest/p.name]
            source_checks[name+'::'+name2+'::archive']=any(a.exists() and sha(a)==digest for a in archives)
        for item in plan['inputs']:
            assert item['label'] not in jobs
            jobs[item['label']]=item;plans[item['label']]=plan
    expected={f'raw176_extendedhot_{mode}_seed{seed}' for mode in MODES for seed in SEEDS}
    assert set(jobs)==expected
    prior_path=OLD/'results.json';prior_data=read(prior_path);source_before[str(prior_path)]=sha(prior_path)
    prior={(r['label'],r['strategy']):r for r in prior_data['rows'] if r['spec_name']=='raw176_extendedhot'}
    assert len(prior)==20
    inputs={label:audit_input(item,plans[label],table) for label,item in sorted(jobs.items())}
    order_pairs=[]
    for seed in SEEDS:
        a,b=[inputs[f'raw176_extendedhot_{mode}_seed{seed}'] for mode in MODES]
        checks=dict(both_inputs_audited=a['audit_passed'] and b['audit_passed'],same_original_IDs=set(a['_by_original'])==set(b['_by_original']),
            per_request_same_NPU_C_V_arrival_full_load_placement=all(a['_by_original'][rid]['identity_sha256']==b['_by_original'][rid]['identity_sha256'] for rid in a['_by_original']),
            per_npu_scientific_population=all(x['population_sha256']==y['population_sha256'] for x,y in zip(a['per_npu'],b['per_npu'])))
        order_pairs.append(dict(seed=seed,checks=checks,passed=all(checks.values()),random_fingerprint=a['input_fingerprint'],ordered_fingerprint=b['input_fingerprint']))
    runs=[]
    for seed in SEEDS:
        for mode in MODES:
            label=f'raw176_extendedhot_{mode}_seed{seed}'
            for policy in POLICIES:runs.append(audit_run(inputs[label],policy,plans[label],prior))
    policy_pairs=[]
    for label in sorted(inputs):
        rr=[r for r in runs if r['label']==label]
        policy_pairs.append(dict(label=label,passed=len(rr)==2 and len({r['manifest_sha256'] for r in rr})==1 and len({r['input_fingerprint'] for r in rr})==1))
    groups=[]
    for mode in MODES:
        for policy in POLICIES:
            rr=[r for r in runs if r['mode']==mode and r['strategy']==policy]
            assert sorted(r['seed'] for r in rr)==sorted(SEEDS)
            groups.append(dict(mode=mode,strategy=policy,seeds=list(SEEDS),n=5,passed=sum(r['conditions_passed'] for r in rr),
                device_U_percent=stats([r['metrics']['device_U_percent'] for r in rr]),warm_slo_percent=stats([r['metrics']['warm_slo_percent'] for r in rr]),
                minimum_role_C_ms=min(r['metrics']['minimum_role_C_ms'] for r in rr),max_ssu_gib_s=max(r['metrics']['max_ssu_gib_s'] for r in rr)))
    drops={}
    for seed in SEEDS:
        rr={r['mode']:r for r in runs if r['seed']==seed and r['strategy']=='baseline'}
        drops[str(seed)]=rr['random']['metrics']['device_U_percent']-rr['ordered']['metrics']['device_U_percent']
    source_checks['sources_unchanged_during_audit']=all(sha(p)==digest for p,digest in source_before.items())
    passed=all(source_checks.values()) and all(x['audit_passed'] for x in inputs.values()) and all(r['conditions_passed'] for r in runs) and all(p['passed'] for p in order_pairs+policy_pairs)
    clean_inputs=[{k:v for k,v in x.items() if not k.startswith('_')} for x in inputs.values()]
    out=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),audit_passed=passed,
        planned_runs=20,completed_runs=len(runs),technical_passed=sum(r['audit_passed'] for r in runs),conditions_passed=sum(r['conditions_passed'] for r in runs),
        definitions=dict(U='actual layer compute intersect [2000,4000) / (32*2000ms)',
            warm_SLO='admission in [2000,4000), completion-admission <=1.5*8*raw per-layer C; follow completion beyond4000',
            TTFT_scope='admission-clock eight-layer prefill proxy, not arrival-clock or measured hardware TTFT; all arrivals0, window-arrival cohort empty',
            population='1212 total requests, raw profiles32/48/64/176K, NQL1024;10 manifests and20 raw completed results',
            per_card_condition='each card active for full2000ms and both short/long have strictly positive clipped compute; not a100ms threshold',
            nominal='current admitted request actual per-layer V_s/C over [admission,completion); no additional nextL0 term; full-run exact-time scan',
            static='sum over cards of per-card max profile actual-placement V_s/C; failed sufficient bound does not imply actual overload',
            fairness='same original request identity+current NPU+all scientific load fields+actual placement across random/ordered; ID/generation encode order',
            uncertainty='designed finite all-at-zero queues; five seeds do not establish sustained steady-state degradation; wider-window recovery documented in previous study',
            statistics='five seeds equally weighted; sampleSD uses n-1; no filtering of scientific-condition failures'),
        source_sha256=source_before,source_checks=source_checks,inputs=clean_inputs,runs=runs,order_pairs=order_pairs,policy_pairs=policy_pairs,
        groups=groups,paired_random_minus_ordered_pp=dict(per_seed=drops,statistics=stats(list(drops.values()))),
        minimum_role_C_ms=min(r['metrics']['minimum_role_C_ms'] for r in runs),
        minimum_ordered_baseline_role_C_ms=min(r['metrics']['minimum_role_C_ms'] for r in runs if r['mode']=='ordered' and r['strategy']=='baseline'),
        max_full_run_ssu_gib_s=max(r['metrics']['max_ssu_gib_s'] for r in runs))
    (HERE/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    csv_write('per_seed.csv',[dict(label=r['label'],seed=r['seed'],mode=r['mode'],strategy=r['strategy'],audit_passed=r['audit_passed'],conditions_passed=r['conditions_passed'],
        **r['metrics'],input_fingerprint=r['input_fingerprint'],result_path=r['result_path'],result_sha256=r['result_sha256']) for r in runs])
    csv_write('per_npu_seed7.csv',[dict(label=r['label'],mode=r['mode'],strategy=r['strategy'],**z) for r in runs if r['seed']==7 for z in r['per_npu']])
    csv_write('input_profiles.csv',inputs['raw176_extendedhot_ordered_seed7']['profiles'])
    assignment=[]
    for mode in MODES:
        info=inputs[f'raw176_extendedhot_{mode}_seed7']
        for r in sorted(info['_by_id'].values(),key=lambda x:x['request_id']):
            assignment.append(dict(mode=mode,seed=7,npu=r['npu'],queue_position=r['generation'],request_id=r['request_id'],original_request_id=r['original_request_id'],
                profile=r['profile'],profile_index=r['profile_index'],role=r['role'],category=r['category'],total_tokens=r['total_tokens'],nql=r['nql'],arrival_ms=0,
                per_layer_C_ms=r['C_ms'],eight_layer_C_ms=r['eight_layer_C_ms'],per_layer_V_MiB=r['V_gib']*1024,
                placement_sha256=r['placement_sha256'],scientific_identity_sha256=r['identity_sha256']))
    csv_write('input_assignment_seed7.csv',assignment)
    print(json.dumps(dict(audit_passed=passed,runs=len(runs),inputs=len(inputs),minimum_role_C_ms=out['minimum_role_C_ms'],groups=groups,
        failed_source_checks=[k for k,v in source_checks.items() if not v],failed_inputs=[x['label'] for x in clean_inputs if not x['audit_passed']],
        failed_runs=[dict(label=r['label'],strategy=r['strategy'],checks={k:v for k,v in r['checks'].items() if not v}) for r in runs if not r['conditions_passed']]),ensure_ascii=False))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
