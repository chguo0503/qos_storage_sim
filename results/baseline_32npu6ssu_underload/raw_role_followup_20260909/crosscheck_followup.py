#!/usr/bin/env python3
"""Independent, read-only audit of 40 main raw fixed/mixed runs (no simulator imports).

Writes only crosscheck_followup.json. Recomputes C/V, identities/placement,
[2000,4000) device U, admission-clock SLO, and an independent V_s/C event scan.
Incomplete cells stay pending; no five-seed mean is published for partial groups.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
START, END, ALPHA = 2000.0, 4000.0, 1.5
SEEDS = (7, 19, 43, 67, 101)
SPECS = ('raw160_three_l20', 'raw200_three_l20')
MODES, POLICIES = ('fixed', 'mixed'), ('baseline', 'once')
BLOCK = 176 * 1024 / 2**30


def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else Path(path).open()) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def clip(a, b, start=START, end=END):
    return max(0.0, min(b, end) - max(a, start))


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                min=min(values), max=max(values))


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def inspect_input(item, plan, table):
    path = Path(item['manifest'])
    data = read(path)
    meta = data['metadata']
    placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p) for p in data['placements']]
    rows = data['requests']
    checks = dict(manifest_sha_matches_plan=sha(path) == item['manifest_sha256'],
                  dimensions=(meta['num_npu'], meta['num_ssu'], meta['n_layers']) == (32, 6, 8),
                  metadata_label_seed_mode=(meta['label'], meta['seed'], meta['assignment_mode']) ==
                  (item['label'], item['seed'], item['mode']), no_scale=meta['compute_scale_actual'] == 1.0,
                  no_padding=True, raw_compute_exact=True, raw_volume_matches=True,
                  original_identity_unique=True, request_identity_unique=True, raw_category=True,
                  original_source_stripe_exact=True, all_arrival_zero=True, load_identifiers_match=True,
                  all_single_layer_templates=True, exact_176kib_commands=True)
    by_id, by_original, profile_rows = {}, {}, {}
    lane_rows = [[] for _ in range(32)]
    fp = hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
    for r in sorted(rows, key=lambda x: x['request_id']):
        rid, npu, load = r['request_id'], r['npu_id'], r['load']
        original = load['original_request_id']
        p = placements[r['placement_index']]
        key = (load['seq_len_k'], load['nql'])
        raw = table[key]
        c = load['per_layer_us'] / 1000
        volume = math.fsum(v for _, v in p[0])
        vector = [math.fsum(v for s, v in p[0] if s == disk) / c * 1000 for disk in range(6)]
        category = ('S' if key[0] <= 80 else 'L') + ('L' if key[1] >= 512 else 'S')
        checks['raw_category'] &= load['category'] == category
        checks['raw_compute_exact'] &= load['per_layer_us'] == raw[1] == load['original_compute_us']
        checks['raw_volume_matches'] &= close(volume, raw[3]) and close(load['per_layer_kv_gb'], raw[3])
        checks['no_padding'] &= load['padding_gib_per_layer'] == 0 and load['constructed_profile'] is False and load['profile_construction'] == {'method': 'direct_data_row'}
        checks['original_identity_unique'] &= original not in by_original
        checks['request_identity_unique'] &= rid not in by_id
        checks['all_arrival_zero'] &= r['arrival_time_ms'] == load['arrival_time'] == load['arrival_ms'] == 0
        checks['load_identifiers_match'] &= load['npu_id'] == npu and load['request_id'] == rid and rid == npu * 1000000 + load['generation']
        checks['all_single_layer_templates'] &= len(p) == 1
        checks['exact_176kib_commands'] &= all(v == BLOCK and 0 <= s < 6 for layer in p for s, v in layer)
        checks['original_source_stripe_exact'] &= all(s == (j + load['source_original_npu_id'] // 4) % 6 for j, (s, v) in enumerate(p[0]))
        # The fingerprint deliberately reproduces the documented input schema, not U/SLO code.
        fp.update(repr((rid, npu, float(r['arrival_time_ms']), load['category'], load['per_layer_us'], p)).encode())
        scientific_load = {k: v for k, v in load.items() if k not in ('request_id', 'npu_id', 'generation')}
        identity = dict(original_request_id=original, arrival_time_ms=r['arrival_time_ms'],
                        scientific_load=scientific_load, placement=p)
        info = dict(request_id=rid, original_request_id=original, npu=npu, C_ms=c, own_C_ms=8*c,
                    V_gib=volume, rate_by_ssu_gib_s=vector, link_rate_gib_s=volume/c*1000,
                    role=load['role'], category=category, profile=f'{key[0]}:{key[1]}', identity_hash=stable_hash(identity))
        by_id[rid], by_original[original] = info, info
        lane_rows[npu].append(info)
        if key not in profile_rows:
            profile_rows[key] = dict(seq_len_k=key[0], nql=key[1], category=category, role=load['role'],
                C_ms=c, eight_layer_C_ms=8*c, raw_V_gib=raw[3], physical_V_gib=volume,
                physical_V_MiB=volume*1024, physical_blocks_per_layer=len(p[0]), count=0)
        profile_rows[key]['count'] += 1
    checks['input_fingerprint_independently_recomputed'] = fp.hexdigest() == data['input_fingerprint'] == meta['input_fingerprint'] == item['input_fingerprint']
    checks['population_count_matches_metadata'] = len(rows) == len(by_id) == meta['request_count']
    checks['source_data_frozen_hash'] = meta['source_data_sha256'] == plan['source_sha256']['data'] == sha(ROOT/'data')
    per_npu = []
    maxima = [[max((r['rate_by_ssu_gib_s'][s] for r in lane), default=0) for s in range(6)] for lane in lane_rows]
    certificate = [math.fsum(v[s] for v in maxima) for s in range(6)]
    for n, lane in enumerate(lane_rows):
        roles = sorted({r['role'] for r in lane})
        count = Counter(r['profile'] for r in lane)
        per_npu.append(dict(npu_id=n, request_count=len(lane), profile_counts=dict(sorted(count.items())),
            roles=roles, pure_compute_ms=math.fsum(r['own_C_ms'] for r in lane),
            max_rate_by_ssu_gib_s=maxima[n]))
    checks['fixed_role_assignment_or_all_mixed'] = all(
        p['roles'] == (['long'] if p['npu_id'] < 20 else ['short']) if item['mode']=='fixed' else p['roles']==['long','short']
        for p in per_npu)
    expected_short_repeats = math.ceil(5200 / (8*sum(table[k][1]/1000 for k in ((32,1024),(48,1024),(64,1024)))))
    expected_long_repeats = math.ceil(5200 / (8*table[(160 if item['spec_name'].startswith('raw160') else 200,1024)][1]/1000))
    expected_counts = {f'{x}:1024':12*expected_short_repeats for x in (32,48,64)}
    expected_counts[f"{160 if item['spec_name'].startswith('raw160') else 200}:1024"] = 20*expected_long_repeats
    checks['global_profile_counts_from_5200ms_rule'] = Counter(r['profile'] for r in by_id.values()) == Counter(expected_counts)
    checks['every_lane_pure_compute_exceeds_window_end'] = all(p['pure_compute_ms'] > END for p in per_npu)
    checks['static_proof_matches_frozen_metadata'] = all(close(a,b) for a,b in zip(certificate,meta['active_profile_rate_certificate']['per_ssu_upper_bound_gib_s']))
    return dict(label=item['label'], spec_name=item['spec_name'], seed=item['seed'], mode=item['mode'],
        path=str(path.relative_to(ROOT)), sha256=sha(path), fingerprint=fp.hexdigest(),
        audit_passed=all(checks.values()), checks=checks, request_count=len(rows),
        profiles=list(profile_rows.values()), per_npu=per_npu,
        scientific_population_sha256=stable_hash([(rid, by_original[rid]['identity_hash']) for rid in sorted(by_original)]),
        static_proof=dict(per_ssu_gib_s=certificate, max_ssu_gib_s=max(certificate),
                          max_link_gib_s=max(r['link_rate_gib_s'] for r in by_id.values()),
                          all_combinations_underload=max(certificate)<40 and max(r['link_rate_gib_s'] for r in by_id.values())<50),
        _requests=by_id, _original=by_original)


def score(rows, inputs):
    out = dict(count=len(rows), request_ids=sorted(r['request_id'] for r in rows),
               original_request_ids=sorted(inputs[r['request_id']]['original_request_id'] for r in rows),
               completion_after_4000_count=sum(r['completion_time_ms'] > END for r in rows))
    for clock in ('admission', 'arrival'):
        passed = sum(r['completion_time_ms'] - r[clock+'_time_ms'] <= ALPHA*inputs[r['request_id']]['own_C_ms'] + 1e-9 for r in rows)
        out[clock] = dict(count=len(rows), passed=passed, rate_percent=100*passed/len(rows) if rows else None)
    return out


def demand_scan(rows, inputs):
    """Half-open active-request V_s/C; batch exact-time ends/starts before interval evaluation."""
    events = defaultdict(lambda: {'remove': [], 'add': []})
    for r in rows:
        events[r['admission_time_ms']]['add'].append(r['request_id'])
        events[r['completion_time_ms']]['remove'].append(r['request_id'])
    times = sorted(events)
    active = {}
    out = {name:dict(per_ssu_peak_gib_s=[0.0]*6, max_link_gib_s=0.0,
            any_ssu_over_capacity_ms=0.0, all_32_active_ms=0.0, max_concurrent_requests=0,
            peak_interval_ms=None, peak_current_requests_by_npu=None) for name in ('full_run','main_window')}
    no_overlap=True
    for i,t in enumerate(times):
        for rid in events[t]['remove']:
            n=inputs[rid]['npu']
            if active.get(n)!=rid: no_overlap=False
            active.pop(n,None)
        for rid in events[t]['add']:
            n=inputs[rid]['npu']
            if n in active:no_overlap=False
            active[n]=rid
        if i+1==len(times):break
        t1=times[i+1]
        if t1<=t:continue
        vector=[math.fsum(inputs[rid]['rate_by_ssu_gib_s'][s] for rid in active.values()) for s in range(6)]
        link=max((inputs[rid]['link_rate_gib_s'] for rid in active.values()),default=0.0)
        for name,duration in (('full_run',t1-t),('main_window',clip(t,t1))):
            if duration<=0:continue
            z=out[name]
            if max(vector)>max(z['per_ssu_peak_gib_s']):
                z['peak_interval_ms']=[t,t1]
                z['peak_current_requests_by_npu']={str(n):rid for n,rid in sorted(active.items())}
            z['per_ssu_peak_gib_s']=[max(a,b) for a,b in zip(vector,z['per_ssu_peak_gib_s'])]
            z['max_link_gib_s']=max(z['max_link_gib_s'],link)
            z['any_ssu_over_capacity_ms'] += duration if max(vector)>40+1e-9 else 0
            z['all_32_active_ms'] += duration if len(active)==32 else 0
            z['max_concurrent_requests']=max(z['max_concurrent_requests'],len(active))
    for z in out.values():
        z['max_ssu_gib_s']=max(z['per_ssu_peak_gib_s'])
        z['strict_underload']=z['max_ssu_gib_s']<40 and z['max_link_gib_s']<50
    out['one_active_request_per_npu_and_drained'] = no_overlap and not active
    out['event_time_count']=len(times)
    return out


def inspect_result(path, item, plan, info, prior):
    raw,command=read(path),read(path.parent/'command.json')
    s=raw['summary']; inputs=info['_requests']; rows=s['request_metrics']
    checks=dict(input_audit=info['audit_passed'], command_completed=command.get('status')=='complete' and command.get('returncode')==0,
        frozen_command_input=command['input_sha256']==info['sha256'],
        matching_label_seed_strategy=raw['submit_seed']==item['seed'] and raw['strategy']==path.parent.name,
        matching_fingerprint=raw['input_fingerprint']==s['input_fingerprint']==info['fingerprint'],
        execution_placement_unchanged=raw['execution_placement_fingerprint']==raw['input_placement_fingerprint'],
        all_invariants=bool(s['invariants']) and all(s['invariants'].values()),
        dimensions=(s['num_npu'],s['num_ssu'],s['n_layers'],s['batch_size'])==(32,6,8,1),
        raw_core_sources_match_plan_command=bool(raw['core_and_policy_sha256']) and all(
            plan['source_sha256'].get(k)==command['source_sha256'].get(k)==v for k,v in raw['core_and_policy_sha256'].items()),
        stress_hash=raw['stress_runner_sha256']==plan['source_sha256']['run_baseline_npu32_stress.py'],
        singleton_batches=True, layer_C_matches_manifest=True, interval_order=True,
        request_batch_times_match=True,
        original_npu_binding=True, full_C_account=True)
    extra={'run_baseline_npu32_stress.py',str((HERE/'run_raw_followup.py').relative_to(ROOT)),
           str((HERE.parent/'run_study.py').relative_to(ROOT))}
    checks['complete_core_source_set']=set(raw['core_and_policy_sha256'])==set(plan['source_sha256'])-extra
    req_ids=[r['request_id'] for r in rows]
    batches=s['microbatch_metrics'];batch_ids=[]
    request_by_id={r['request_id']:r for r in rows}
    checks['complete_unique_request_population']=len(req_ids)==len(set(req_ids))==len(inputs) and set(req_ids)==set(inputs)
    by_npu=[dict(compute_ms=0.0,active_ms=0.0,l0_stall_ms=0.0,l1_7_stall_ms=0.0,
                 roles={r:dict(compute_ms=0.0,active_ms=0.0) for r in ('short','long')}) for _ in range(32)]
    for b in batches:
        checks['singleton_batches'] &= len(b['member_request_ids'])==b['batch_size']==1
        rid=b['member_request_ids'][0];batch_ids.append(rid);r=inputs[rid];n=b['npu_id'];z=by_npu[n]
        checks['original_npu_binding'] &= n==r['npu']
        checks['layer_C_matches_manifest'] &= sorted(x['layer'] for x in b['layer_metrics'])==list(range(8))
        a,f=b['admission_time_ms'],b['completion_time_ms'];active=clip(a,f)
        checks['request_batch_times_match'] &= close(a,request_by_id[rid]['admission_time_ms']) and close(f,request_by_id[rid]['completion_time_ms'])
        z['active_ms']+=active;z['roles'][r['role']]['active_ms']+=active
        prev=a; csum=0.0
        for l in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            cs,ce=l['compute_start_ms'],l['compute_end_ms']
            checks['interval_order'] &= finite(cs) and finite(ce) and prev<=cs+1e-8 and cs<=ce and ce<=f+1e-8
            checks['layer_C_matches_manifest'] &= close(ce-cs,r['C_ms'])
            value=clip(cs,ce);z['compute_ms']+=value;z['roles'][r['role']]['compute_ms']+=value
            z['l0_stall_ms' if l['layer']==0 else 'l1_7_stall_ms']+=clip(prev,cs)
            csum+=ce-cs;prev=ce
        checks['full_C_account'] &= close(csum,r['own_C_ms']) and close(prev,f)
    checks['complete_unique_batch_population']=len(batch_ids)==len(set(batch_ids))==len(inputs) and set(batch_ids)==set(inputs)
    checks['request_times_C_NPU_match']=all(
        finite(r['completion_time_ms']) and r['completion_time_ms']>=r['admission_time_ms']>=r['arrival_time_ms']==0
        and r['npu_id']==inputs[r['request_id']]['npu']
        and close(r['own_compute_ms'],inputs[r['request_id']]['own_C_ms']) for r in rows)
    checks['window_time_account']=all(close(z['compute_ms']+z['l0_stall_ms']+z['l1_7_stall_ms'],z['active_ms']) for z in by_npu)
    warm=[r for r in rows if START<=r['admission_time_ms']<END]
    cohort={name:score(sample,inputs) for name,sample in (('window_admissions',warm),
             ('window_arrivals',[r for r in rows if START<=r['arrival_time_ms']<END]),('all_requests',rows))}
    stored=raw['slo']
    checks['stored_slo_definition']=(stored['window_start_ms'],stored['window_end_ms'],stored['alpha'])==(START,END,ALPHA) and stored['ideal']=='n_layers * per_layer_compute_ms (own_compute_ms)'
    for name,co in cohort.items():
        for clock in ('admission','arrival'):
            a,b=co[clock],stored[name][clock]
            checks[f'stored_{name}_{clock}_SLO']=a['count']==b['count'] and a['passed']==b['passed'] and (b['rate'] is None if a['rate_percent'] is None else close(a['rate_percent'],100*b['rate']))
        checks[f'stored_{name}_cohort_identity']=co['request_ids']==sorted(stored[name]['request_ids'])
    u=math.fsum(z['compute_ms'] for z in by_npu)/(32*(END-START))*100
    w=[w for w in raw['windows'] if (w['start_ms'],w['end_ms'])==(START,END)]
    checks['stored_window_U']=len(w)==1 and close(u,100*w[0]['mean_npu_utilization'])
    checks['stored_per_npu_U']=len(w)==1 and all(close(z['compute_ms'],v) for z,v in zip(by_npu,w[0]['compute_ms_by_npu']))
    ad=raw['adapter_statistics'];times=ad['collector_times_ms']
    checks['literal_policy_once_or_baseline']=raw['strategy']==ad['strategy'] in POLICIES
    checks['shared_5ms_collection']=raw['collector_interval_ms']==ad['collector_interval_ms']==5 and bool(times) and all(close(v,5*i) for i,v in enumerate(times)) and ad['fresh_reads_by_ssu']==[len(times)]*6
    checks['fixed_no_migration']=raw['policy_config']['assignment']=='fixed' and ad['assignment_count']==0 and raw['assignment_log']==[]
    checks['static_CIR_no_reorder']=ad['cir_write_events']==[] and ad['reorder_calls']==ad['reorder_changed']==0
    checks['zero_modeled_control_cost']=ad['modeled_control_cpu_latency_ms']==ad['modeled_control_communication_latency_ms']==0
    scan=demand_scan(rows,inputs)
    checks['one_current_request_per_npu']=scan['one_active_request_per_npu_and_drained']
    fourth=[]
    for n in range(32):
        completions=sorted(r['completion_time_ms'] for r in rows if r['npu_id']==n)
        fourth.append(completions[3] if len(completions)>=4 else None)
    metrics=dict(device_utilization_percent=u,warm_slo_percent=cohort['window_admissions']['admission']['rate_percent'],
        warm_admissions=len(warm),warm_slo_passed=cohort['window_admissions']['admission']['passed'],
        completion_after_window_count=cohort['window_admissions']['completion_after_4000_count'],
        all_npus_active=all(close(z['active_ms'],END-START) for z in by_npu),
        warm_mixed_card_count=sum(all(z['roles'][role]['compute_ms']>0 for role in ('short','long')) for z in by_npu),
        fourth_request_by_1500=all(t is not None and t<=1500 for t in fourth), max_fourth_completion_ms=max(t for t in fourth if t is not None),
        max_ssu_nominal_gib_s=scan['full_run']['max_ssu_gib_s'],any_ssu_over_capacity_ms=scan['full_run']['any_ssu_over_capacity_ms'],
        max_npu_nominal_gib_s=scan['full_run']['max_link_gib_s'],full_run_capacity_pass=scan['full_run']['strict_underload'],
        makespan_ms=max(r['completion_time_ms'] for r in rows))
    for role in ('short','long'):
        comp=math.fsum(z['roles'][role]['compute_ms'] for z in by_npu);act=math.fsum(z['roles'][role]['active_ms'] for z in by_npu)
        metrics[role+'_pooled_U_percent']=100*comp/act if act else None
    previous=prior.get((item['label'],raw['strategy']))
    if previous:
        checks['parent_result_hash_binding']=previous['result_sha256']==sha(path) and previous['input_sha256']==info['sha256']
        checks['parent_U_SLO_counts_and_nominal_scan_match']=all(
            (metrics[k]==previous[k] if isinstance(metrics[k],(bool,int)) else close(metrics[k],previous[k])) for k in metrics if k in previous)
    return dict(label=item['label'],spec_name=item['spec_name'],seed=item['seed'],mode=item['mode'],strategy=raw['strategy'],
        status='complete' if all(checks.values()) else 'audit_failed', audit_passed=all(checks.values()),checks=checks,
        input_fingerprint=info['fingerprint'],manifest_sha256=info['sha256'],result_path=str(path.relative_to(ROOT)),result_sha256=sha(path),
        command_sha256=sha(path.parent/'command.json'),core_source_sha256=raw['core_and_policy_sha256'],static_path_cirs_gib_s=raw['static_path_cirs_gib_s'],
        metrics=metrics,cohorts=cohort,warm_by_role={role:score([r for r in warm if inputs[r['request_id']]['role']==role],inputs) for role in ('short','long')},
        warm_by_category={cat:score([r for r in warm if inputs[r['request_id']]['category']==cat],inputs) for cat in ('SS','SL','LS','LL')},
        per_npu=by_npu,fourth_completion_ms=fourth,independent_nominal_scan=scan,
        parent_numeric_crosscheck='matched' if previous else 'pending_parent_aggregation',
        active_underload_valid=all(checks.values()) and metrics['all_npus_active'] and metrics['full_run_capacity_pass'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args()
    source_before={str(Path(__file__).relative_to(ROOT)):sha(__file__),'data':sha(ROOT/'data')}
    table=ast.literal_eval((ROOT/'data').read_text())
    jobs={};plans={};global_checks={};plans_info={}
    for file in sorted((HERE/'plans').glob('*.json')):
        plan=read(file)
        relevant=[j for j in plan['jobs'] if j['input']['spec_name'] in SPECS]
        if not relevant:continue
        name=str(file.relative_to(ROOT));source_before[name]=sha(file)
        plans_info[name]=dict(sha256=sha(file),spec_file=plan['spec_file'],spec_sha256=plan['spec_sha256'])
        global_checks[name+'_spec_frozen']=sha(plan['spec_file'])==plan['spec_sha256']
        source_before[str(Path(plan['spec_file']).relative_to(ROOT))]=sha(plan['spec_file'])
        for k,v in plan['source_sha256'].items():
            source_before[k]=sha(ROOT/k)
            global_checks[name+'::'+k+'_frozen']=source_before[k]==v
            frozen=HERE/'sources'/v/Path(k).name
            global_checks[name+'::'+k+'_archive']=frozen.exists() and sha(frozen)==v
        for job in relevant:
            item=job['input'];key=(item['label'],job['strategy'])
            if key in jobs and jobs[key]!=job:raise ValueError(f'Conflicting frozen job {key}')
            jobs[key],plans[item['label']]=job,plan
    parent_file=HERE/'followup_results.json'
    parent_snapshot=None;prior={}
    if parent_file.exists():
        data_bytes=parent_file.read_bytes();parent_data=json.loads(data_bytes)
        parent_snapshot=dict(path=str(parent_file.relative_to(ROOT)),sha256=hashlib.sha256(data_bytes).hexdigest())
        prior={(r['label'],r['strategy']):r for r in parent_data['rows']}
    inputs={};input_errors=[]
    for label in sorted(plans):
        item=next(j['input'] for key,j in jobs.items() if key[0]==label)
        try:inputs[label]=inspect_input(item,plans[label],table)
        except Exception as error:input_errors.append(dict(label=label,error=f'{type(error).__name__}: {error}'))
    assignments=[]
    for spec in SPECS:
        for seed in SEEDS:
            a,b=[inputs.get(f'{spec}_{m}_seed{seed}') for m in MODES]
            if a is None or b is None:
                assignments.append(dict(spec_name=spec,seed=seed,status='pending'));continue
            checks=dict(both_inputs_pass=a['audit_passed'] and b['audit_passed'],
                original_ids_identical=set(a['_original'])==set(b['_original']),
                scientific_population_and_actual_placement_identical=a['scientific_population_sha256']==b['scientific_population_sha256'])
            assignments.append(dict(spec_name=spec,seed=seed,status='complete' if all(checks.values()) else 'audit_failed',checks=checks,
                fixed_label=a['label'],mixed_label=b['label'],population_sha256=a['scientific_population_sha256'],
                population_count=a['request_count'],allowed_changes=['npu_id','load.npu_id','request_id','load.request_id','load.generation','list_order'],
                moved_npu_count=sum(a['_original'][rid]['npu']!=b['_original'][rid]['npu'] for rid in a['_original'])))
    results=[]
    for spec in SPECS:
        for seed in SEEDS:
            for mode in MODES:
                label=f'{spec}_{mode}_seed{seed}'
                for strategy in POLICIES:
                    key=(label,strategy);base=dict(label=label,spec_name=spec,seed=seed,mode=mode,strategy=strategy)
                    if key not in jobs or label not in inputs:
                        failed_input=next((x for x in input_errors if x['label']==label),None)
                        results.append(dict(**base,status='error' if failed_input else 'pending',
                            reason=failed_input['error'] if failed_input else 'frozen_plan_or_input_missing'));continue
                    files=sorted((HERE/'runs'/label/strategy).glob('*.json.gz'))
                    command=HERE/'runs'/label/strategy/'command.json'
                    if command.exists() and read(command).get('status')=='failed':
                        results.append(dict(**base,status='error',error='runner_command_failed',command_sha256=sha(command)));continue
                    if not files or not command.exists() or read(command).get('status')!='complete':
                        results.append(dict(**base,status='pending',reason='result_or_completed_command_not_ready'));continue
                    try:
                        if len(files)!=1:raise ValueError(f'{len(files)} raw results; ambiguous')
                        result=inspect_result(files[0],jobs[key]['input'],plans[label],inputs[label],prior)
                        results.append(result)
                        print(json.dumps(dict(label=label,strategy=strategy,status=result['status'],U=result['metrics']['device_utilization_percent'],SLO=result['metrics']['warm_slo_percent'])),flush=True)
                    except Exception as error:
                        results.append(dict(**base,status='error',error=f'{type(error).__name__}: {error}'))
    policy_pairs=[]
    by_key={(r['label'],r['strategy']):r for r in results}
    for label in sorted(inputs):
        a,b=[by_key.get((label,s)) for s in POLICIES]
        if any(r is None or r['status']=='pending' for r in (a,b)):
            policy_pairs.append(dict(label=label,status='pending'));continue
        if any(r['status']!='complete' for r in (a,b)):
            policy_pairs.append(dict(label=label,status='audit_failed'));continue
        checks={k:a[k]==b[k] for k in ('input_fingerprint','manifest_sha256','seed','core_source_sha256','static_path_cirs_gib_s')}
        ids=[set(r['cohorts']['window_admissions']['request_ids']) for r in (a,b)]
        policy_pairs.append(dict(label=label,status='complete' if all(checks.values()) else 'audit_failed',checks=checks,
            warm_baseline_count=len(ids[0]),warm_once_count=len(ids[1]),warm_common_count=len(ids[0]&ids[1]),
            baseline_only_ids=sorted(ids[0]-ids[1]),once_only_ids=sorted(ids[1]-ids[0])))
    groups=[]
    for spec in SPECS:
        for mode in MODES:
            for strategy in POLICIES:
                rows=[r for r in results if (r['spec_name'],r['mode'],r['strategy'])==(spec,mode,strategy)]
                complete=len(rows)==5 and sorted(r['seed'] for r in rows)==list(SEEDS) and all(r['status']=='complete' for r in rows)
                group=dict(spec_name=spec,mode=mode,strategy=strategy,planned=5,
                    complete=sum(r['status']=='complete' for r in rows),status='complete' if complete else 'pending_or_failed',
                    seeds=SEEDS,aggregate=None)
                if complete:
                    fields=('device_utilization_percent','warm_slo_percent','warm_admissions','warm_slo_passed','short_pooled_U_percent','long_pooled_U_percent')
                    group['aggregate']={field:stats([r['metrics'][field] for r in rows]) for field in fields}
                    group['active_underload_valid_count']=sum(r['active_underload_valid'] for r in rows)
                groups.append(group)
    global_checks['sources_unchanged_during_analysis']=all(sha(ROOT/k)==v for k,v in source_before.items())
    counts=Counter(r['status'] for r in results)
    all_closed=(counts['complete']==40 and not input_errors and all(global_checks.values()) and all(x['status']=='complete' for x in assignments+policy_pairs))
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),planned=40,counts=dict(counts),all_40_closed_and_passed=all_closed,
        definitions=dict(window_ms=[START,END],npu_utilization='sum clipped layer compute / (32 * 2000ms)',
            warm_SLO='admission in [2000,4000), completion-admission <= 1.5 * manifest 8C; follow to completion without truncating at 4000',
            arrival_clock='all arrivals are zero; window-arrival cohort empty (N/A); warm admission cohort arrival-clock SLO separately retained',
            alpha=ALPHA,not_hardware_TTFT=True,source_of_U_SLO='independent standard-library calculation; no imported runner/analyzer metric functions',
            nominal_capacity='independently re-scanned half-open admission-completion current-request physical V_s/C; ties batched; no additional L0 term; not instantaneous physical SSD throughput',
            group_statistics='five seeds equally weighted; sample SD; no partial five-seed aggregates; condition failures retained',
            input_pairing='original_request_id; full remaining load, arrival and actual block placement invariant, permitting only NPU/order-encoding fields to change',
            exclusions='four seed7 pilot configurations excluded from this 40-cell main comparison'),
        source_sha256=source_before,plans=plans_info,parent_comparison_snapshot=parent_snapshot,
        global_checks=global_checks,input_errors=input_errors,
        inputs=[{k:v for k,v in x.items() if not k.startswith('_')} for x in inputs.values()],
        assignment_pairs=assignments,policy_pairs=policy_pairs,groups=groups,runs=results)
    dest=HERE/'crosscheck_followup.json'
    dest.write_text(json.dumps(output,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(planned=40,counts=dict(counts),assignment_pairs=len(assignments),all_closed=all_closed,
        failed_checks={r['label']+'/'+r['strategy']:[k for k,v in r.get('checks',{}).items() if not v] for r in results if r['status'] not in ('complete','pending')},input_errors=input_errors)),flush=True)
    if args.require_complete and not all_closed:raise SystemExit(1)


if __name__=='__main__':main()
