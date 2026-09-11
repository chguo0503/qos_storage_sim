#!/usr/bin/env python3
"""Read frozen inputs/results; independently audit fixed, predeclared windows.

No simulations or input/result mutations. Outputs analysis.json.gz and summary.csv.
All utilization values in JSON/CSV are fractions, and all times are milliseconds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from continuous_batch_sim import continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest

NPU, SSU, LAYERS = 32, 6, 8
DISK_RATE, LINK_RATE = 40.0, 50.0
MAIN_WINDOW = (2000.0, 4000.0)
WINDOWS = (MAIN_WINDOW,) + tuple((2000.0 + 250*i, 2250.0 + 250*i) for i in range(8))
CATEGORIES = ("SS", "SL", "LS", "LL")
TIME_TOL, RATE_TOL = 1e-7, 1e-9


def read_json(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt', encoding='utf-8-sig') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def overlap(a, b, lo, hi):
    return max(0.0, min(b, hi) - max(a, lo))


def mean(xs):
    return math.fsum(xs) / len(xs) if xs else None


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=TIME_TOL)


def relative(path, base):
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def load_input(path, base):
    requests, metadata = load_manifest(path)
    fp = continuous_batch_input_fingerprint(requests)
    by_id, rows = {}, []
    maxima = [[0.0]*SSU for _ in range(NPU)]
    max_links = [0.0]*NPU
    witnesses = [[None]*SSU for _ in range(NPU)]
    identical_layers = True
    for r in requests:
        assert r.request_id not in by_id, 'duplicate request id'
        assert 0 <= r.npu_id < NPU, 'input NPU outside 0..31'
        c = float(r.load['per_layer_us']) / 1000.0
        assert c > 0
        assert len(r.placement) in (1, LAYERS), 'unsupported placement layer count'
        layer_rates, layer_volumes = [], []
        for layer, placement in enumerate(r.placement):
            assert all(0 <= s < SSU and v >= 0 for s, v in placement)
            vols = [math.fsum(v for s, v in placement if s == k) for k in range(SSU)]
            rates = [1000*v/c for v in vols]
            layer_volumes.append(vols)
            layer_rates.append(rates)
            max_links[r.npu_id] = max(max_links[r.npu_id], math.fsum(rates))
            for s, value in enumerate(rates):
                if value > maxima[r.npu_id][s]:
                    maxima[r.npu_id][s] = value
                    witnesses[r.npu_id][s] = {'request_id': r.request_id, 'placement_layer': layer}
        identical = all(all(close(x,y) for x,y in zip(v, layer_volumes[0])) for v in layer_volumes)
        identical_layers &= identical
        # The studied manifests repeat placement in all eight layers. Retain an
        # explicit flag if a future input uses different layer byte vectors.
        rates = [mean([v[s] for v in layer_rates]) for s in range(SSU)]
        seq, nql = float(r.load['seq_len_k']), float(r.load['nql'])
        category = ('S' if seq <= 80 else 'L') + ('S' if nql < 512 else 'L')
        assert r.load.get('category', category) == category
        row = {'request_id': r.request_id, 'npu_id': r.npu_id,
               'arrival_ms': r.arrival_time_ms, 'category': category,
               'role': str(r.load.get('role', 'unknown')),
               'seq_len_k': seq, 'nql': nql, 'layer_compute_ms': c,
               'rate_by_ssu_gib_s': rates, 'layer_vectors_identical': identical}
        # Simulator request_id also determines equal-arrival admission order.
        # Ordered manifests keep scientific identity in original_request_id.
        content_load = {k:v for k,v in r.load.items()
                        if k not in ('request_id','generation','original_request_id')}
        row['original_request_id'] = int(r.load.get('original_request_id',r.request_id))
        row['has_original_request_id'] = 'original_request_id' in r.load
        row['content_and_placement_sha256'] = canonical_sha({
            'npu_id':r.npu_id,'arrival_time_ms':r.arrival_time_ms,
            'load':content_load,'placement':r.placement})
        by_id[r.request_id] = row
        rows.append(row)
    assert {r['npu_id'] for r in rows} == set(range(NPU)), 'input lacks an NPU'
    proof = [math.fsum(maxima[n][s] for n in range(NPU)) for s in range(SSU)]
    lane = []
    for n in range(NPU):
        g = [r for r in rows if r['npu_id'] == n]
        lane.append({'npu_id': n, 'requests': len(g),
                     'role_counts': dict(Counter(r['role'] for r in g)),
                     'category_counts': dict(Counter(r['category'] for r in g)),
                     'distinct_profiles': len({(r['seq_len_k'],r['nql']) for r in g}),
                     'ideal_compute_ms': LAYERS*math.fsum(r['layer_compute_ms'] for r in g),
                     'max_profile_rate_by_ssu_gib_s': maxima[n],
                     'max_profile_rate_by_ssu_witness': witnesses[n],
                     'max_profile_link_rate_gib_s': max_links[n]})
    return {'path': relative(path, base), 'file_sha256': sha(path),
            'label': metadata.get('label', path.name.removesuffix('.json.gz')),
            'input_fingerprint': fp, 'metadata': metadata, 'request_count': len(rows),
            'all_arrival_zero': all(r['arrival_ms'] == 0 for r in rows),
            'layer_vectors_identical': identical_layers,
            'per_npu': lane,
            'static_max_proof': {
                'definition': 'For each SSU sum over NPUs the maximum 1000*actual_layer_V/C among every request and layer on that original NPU.',
                'scope': 'Upper bound on nominal current admitted-request V/C; not a released-I/O or FIFO deadline guarantee; assumes original NPU assignment.',
                'per_ssu_gib_s': proof, 'max_ssu_gib_s': max(proof),
                'total_of_per_npu_max_link_rates_gib_s': math.fsum(max_links),
                'all_ssu_within_capacity': max(proof) <= DISK_RATE+RATE_TOL,
                'max_npu_link_gib_s': max(max_links),
                'all_npu_links_within_capacity': max(max_links) <= LINK_RATE+RATE_TOL},
            '_requests': by_id}


def group_metrics(rows):
    if not rows:
        return {'request_count': 0, 'request_equal_utilization': None,
                'pooled_utilization': None, 'mean_per_npu_conditional_utilization': None,
                'active_npus': 0, 'compute_ms': 0.0, 'active_ms': 0.0,
                'stall_ms': 0.0, 'l0_exposed_stall_ms': 0.0, 'l1_7_exposed_stall_ms': 0.0}
    comp = math.fsum(r['compute_ms'] for r in rows)
    active = math.fsum(r['active_ms'] for r in rows)
    cards = defaultdict(list)
    for r in rows:
        cards[r['npu_id']].append(r)
    return {'request_count': len(rows),
            'request_equal_utilization': mean([r['compute_ms']/r['active_ms'] for r in rows]),
            'pooled_utilization': comp/active,
            'mean_per_npu_conditional_utilization': mean([
                math.fsum(r['compute_ms'] for r in g)/math.fsum(r['active_ms'] for r in g)
                for g in cards.values()]),
            'active_npus': len(cards), 'compute_ms': comp, 'active_ms': active,
            'stall_ms': active-comp,
            'l0_exposed_stall_ms': math.fsum(r['l0_exposed_stall_ms'] for r in rows),
            'l1_7_exposed_stall_ms': math.fsum(r['l1_7_exposed_stall_ms'] for r in rows)}


def analyze_window(batches, input_info, start, end):
    requests = input_info['_requests']
    rows = []
    for b in batches:
        a, f = b['admission_time_ms'], b['completion_time_ms']
        active = overlap(a, f, start, end)
        if active <= 0:
            continue
        rid = b['member_request_ids'][0]
        original = requests[rid]
        comp, l0, warm = [], [], []
        previous = a
        for layer in sorted(b['layer_metrics'], key=lambda x: x['layer']):
            cs, ce = layer['compute_start_ms'], layer['compute_end_ms']
            comp.append(overlap(cs, ce, start, end))
            (l0 if layer['layer'] == 0 else warm).append(overlap(previous, cs, start, end))
            previous = ce
        c, s0, sw = math.fsum(comp), math.fsum(l0), math.fsum(warm)
        assert close(c+s0+sw, active), 'window compute/stall identity failed'
        rows.append({'request_id': rid, 'npu_id': b['npu_id'],
                     'category': original['category'], 'role': original['role'],
                     'active_ms': active, 'compute_ms': c, 'stall_ms': active-c,
                     'l0_exposed_stall_ms': s0, 'l1_7_exposed_stall_ms': sw,
                     'left_clipped': a < start, 'right_clipped': f > end,
                     'completed_in_window': start <= f < end})
    by_role = {g: group_metrics([r for r in rows if r['role']==g])
               for g in sorted({'short','long'} | {r['role'] for r in requests.values()})}
    by_category = {g: group_metrics([r for r in rows if r['category']==g]) for g in CATEGORIES}
    cards = []
    for n in range(NPU):
        g = [r for r in rows if r['npu_id'] == n]
        x = group_metrics(g)
        x.update(npu_id=n, device_utilization=x['compute_ms']/(end-start),
                 idle_ms=(end-start)-x['active_ms'],
                 by_role={role: group_metrics([r for r in g if r['role']==role]) for role in by_role})
        cards.append(x)
    totals = group_metrics(rows)
    totals.update(start_ms=start, end_ms=end, duration_ms=end-start,
                  device_utilization=totals['compute_ms']/(NPU*(end-start)),
                  all_npus_active=all(close(c['active_ms'],end-start) for c in cards),
                  each_npu_short_and_long_positive_compute=all(
                      c['by_role']['short']['compute_ms'] > 0 and c['by_role']['long']['compute_ms'] > 0
                      for c in cards),
                  overlapping_requests=len(rows), completed_in_window=sum(r['completed_in_window'] for r in rows),
                  left_clipped_requests=sum(r['left_clipped'] for r in rows),
                  right_clipped_requests=sum(r['right_clipped'] for r in rows),
                  by_role=by_role, by_category=by_category, per_npu=cards, requests=rows)
    return totals


def nominal_scan(batches, input_info, windows):
    """Exact constant-interval scan, processing all same-time changes together."""
    req = input_info['_requests']
    events = defaultdict(lambda: {'remove': [], 'add': []})
    for b in batches:
        rid = b['member_request_ids'][0]
        events[float(b['admission_time_ms'])]['add'].append((b['npu_id'],rid))
        events[float(b['completion_time_ms'])]['remove'].append((b['npu_id'],rid))
    times = sorted(events)
    specs = [('full_run',times[0],times[-1])] + [
        (f'window_{int(a)}_{int(b)}',a,b) for a,b in windows]
    out = {}
    for key,a,b in specs:
        out[key] = {'start_ms':a, 'end_ms':b, 'segments':0,
                    'max_per_ssu_gib_s':[0.0]*SSU, 'witness_by_ssu':[None]*SSU,
                    'over_capacity_ms_by_ssu':[0.0]*SSU,
                    'any_ssu_over_capacity_ms':0.0, 'max_total_gib_s':0.0,
                    'max_npu_link_gib_s':0.0, 'min_active_npus':NPU,
                    'all_32_active_ms':0.0, 'covered_ms':0.0}
    active = {}
    for i,t in enumerate(times):
        # Removal before addition is an implementation detail within a tie.
        # No state inside a zero-duration tie is included in any maximum.
        for n,rid in events[t]['remove']:
            assert active.get(n)==rid, 'overlap or unmatched completion'
            del active[n]
        for n,rid in events[t]['add']:
            assert n not in active, 'more than one admitted request per NPU'
            active[n]=rid
        if i+1 == len(times):
            continue
        stop = times[i+1]
        rates = [math.fsum(req[r]['rate_by_ssu_gib_s'][s] for r in active.values()) for s in range(SSU)]
        link = max((math.fsum(req[r]['rate_by_ssu_gib_s']) for r in active.values()), default=0.0)
        for key,a,b in specs:
            duration = overlap(t,stop,a,b)
            if duration <= 0:
                continue
            q = out[key]
            q['segments'] += 1
            q['covered_ms'] += duration
            q['min_active_npus'] = min(q['min_active_npus'],len(active))
            if len(active)==NPU:
                q['all_32_active_ms'] += duration
            q['max_total_gib_s']=max(q['max_total_gib_s'],math.fsum(rates))
            q['max_npu_link_gib_s']=max(q['max_npu_link_gib_s'],link)
            exceeded = False
            for s,rate in enumerate(rates):
                if rate > q['max_per_ssu_gib_s'][s]:
                    q['max_per_ssu_gib_s'][s]=rate
                    q['witness_by_ssu'][s]={'time_ms':max(t,a), 'interval_end_ms':min(stop,b),
                        'active_request_ids_by_npu':{str(n):r for n,r in sorted(active.items())},
                        'all_ssu_gib_s':rates}
                if rate > DISK_RATE+RATE_TOL:
                    q['over_capacity_ms_by_ssu'][s] += duration
                    exceeded = True
            if exceeded:
                q['any_ssu_over_capacity_ms'] += duration
    assert not active
    proof = input_info['static_max_proof']['per_ssu_gib_s']
    for q in out.values():
        q['max_ssu_gib_s']=max(q['max_per_ssu_gib_s'])
        q['all_ssu_within_capacity']=q['max_ssu_gib_s']<=DISK_RATE+RATE_TOL
        q['all_npu_links_within_capacity']=q['max_npu_link_gib_s']<=LINK_RATE+RATE_TOL
        q['bounded_by_input_static_max']=all(x<=y+RATE_TOL for x,y in zip(q['max_per_ssu_gib_s'],proof))
        q['whole_requested_interval_covered']=close(q['covered_ms'],q['end_ms']-q['start_ms'])
    return out


def analyze_result(path, input_info, base):
    result = read_json(path)
    summary = result['summary']
    req = input_info['_requests']
    checks = {}
    checks['input_fingerprint_matches_manifest'] = result.get('input_fingerprint')==input_info['input_fingerprint']
    checks['result_label_matches_manifest'] = path.parent.parent.name==input_info['label'] and result.get('metadata',{}).get('label',input_info['label'])==input_info['label']
    checks['submission_seed_matches_manifest'] = result.get('submit_seed')==input_info['metadata'].get('seed')
    checks['summary_input_fingerprint_matches'] = summary.get('input_fingerprint')==input_info['input_fingerprint']
    checks['dimensions_32_6_8_batch1'] = (summary.get('num_npu'),summary.get('num_ssu'),summary.get('n_layers'),summary.get('batch_size'))==(NPU,SSU,LAYERS,1)
    checks['all_input_arrivals_at_zero'] = input_info['all_arrival_zero']
    checks['retained_prefill_and_submission_semantics'] = (
        summary.get('execution_model')=='full_prefill_layer_synchronous_microbatch_v1'
        and summary.get('prefetch_policy')=='next_layer_at_batch_compute_start'
        and summary.get('cross_request_layer0_prefetch') is True
        and summary.get('client_submit_batch_size')==1
        and close(summary.get('client_issue_interval_us',-1),0.1))
    checks['simulator_invariants'] = bool(summary.get('invariants')) and all(summary['invariants'].values())
    batches = summary['microbatch_metrics']
    checks['singleton_batches'] = all(len(b['member_request_ids'])==1 for b in batches)
    assert checks['singleton_batches'], 'analysis requires singleton batches'
    ids = [b['member_request_ids'][0] for b in batches]
    checks['complete_unique_request_population'] = len(ids)==len(set(ids))==len(req) and set(ids)==set(req)
    assert checks['complete_unique_request_population'], 'partial/foreign request population'
    checks['original_npu_assignment'] = all(b['npu_id']==req[b['member_request_ids'][0]]['npu_id'] for b in batches)
    runtime = {r['request_id']:r for r in summary.get('request_metrics',[])}
    checks['runtime_request_identity_and_arrival'] = set(runtime)==set(req) and all(
        runtime[rid]['npu_id']==r['npu_id'] and close(runtime[rid]['arrival_time_ms'],r['arrival_ms'])
        and runtime[rid]['category']==r['category'] for rid,r in req.items())
    checks['all_eight_layers_recorded'] = all(sorted(l['layer'] for l in b['layer_metrics'])==list(range(LAYERS)) for b in batches)
    checks['layer_compute_matches_input'] = True
    checks['stored_stall_matches_interval_decomposition'] = True
    for b in batches:
        previous = b['admission_time_ms']
        c = req[b['member_request_ids'][0]]['layer_compute_ms']
        assert b['completion_time_ms'] > previous
        for l in sorted(b['layer_metrics'],key=lambda x:x['layer']):
            cs,ce = l['compute_start_ms'],l['compute_end_ms']
            checks['layer_compute_matches_input'] &= close(ce-cs,c)
            assert cs >= previous-TIME_TOL, 'compute layer intervals overlap'
            if 'io_barrier_wait_ms' in l:
                checks['stored_stall_matches_interval_decomposition'] &= close(l['io_barrier_wait_ms'],max(0.0,cs-previous))
            previous=ce
        assert close(previous,b['completion_time_ms'])
    source_results = {}
    for name,expected in result.get('core_and_policy_sha256',{}).items():
        p = ROOT/name
        actual = sha(p) if p.is_file() else None
        source_results[name]={'expected':expected,'current':actual,'matches':expected==actual}
    checks['core_hashes_present_and_current'] = bool(source_results) and all(v['matches'] for v in source_results.values())
    if 'stress_runner_sha256' in result:
        checks['stress_runner_hash_current'] = result['stress_runner_sha256']==sha(ROOT/'run_baseline_npu32_stress.py')
    checks['layer_volume_vectors_constant'] = input_info['layer_vectors_identical']
    expected_strategy=path.parent.name
    checks['strategy_matches_directory'] = result.get('strategy')==expected_strategy
    fourth=[]
    for n in range(NPU):
        finishes=sorted(b['completion_time_ms'] for b in batches if b['npu_id']==n)
        fourth.append(finishes[3] if len(finishes)>=4 else None)
    warm_ok=all(x is not None and x<=1500.0+TIME_TOL for x in fourth)
    windows=[analyze_window(batches,input_info,a,b) for a,b in WINDOWS]
    scan=nominal_scan(batches,input_info,WINDOWS)
    checks['event_scan_within_static_max_proof'] = all(x['bounded_by_input_static_max'] for x in scan.values())
    # Results from changing NPU assignment are retained but fail this proof's
    # premise; they are never silently paired against the original lane bound.
    full_processing=[(b['completion_time_ms']-b['admission_time_ms']) for b in batches]
    full_compute=[LAYERS*req[b['member_request_ids'][0]]['layer_compute_ms'] for b in batches]
    full_arrival=[b['completion_time_ms']-req[b['member_request_ids'][0]]['arrival_ms'] for b in batches]
    return {'path':relative(path,base), 'file_sha256':sha(path),
            'strategy':result.get('strategy'), 'input_fingerprint':result.get('input_fingerprint'),
            'submit_seed':result.get('submit_seed'),
            'label':path.parent.parent.name,
            'audit':{'passed':all(checks.values()),'checks':checks,'source_hashes':source_results},
            'warmup':{'fourth_completion_by_npu_ms':fourth, 'deadline_ms':1500.0,
                      'all_fourth_completions_by_1500':warm_ok,
                      'settle_before_main_window_ms':2000.0-max(x for x in fourth if x is not None)},
            'main_window_valid':warm_ok and windows[0]['all_npus_active'] and windows[0]['each_npu_short_and_long_positive_compute'],
            'all_subwindows_active_and_mixed':all(w['all_npus_active'] and w['each_npu_short_and_long_positive_compute'] for w in windows[1:]),
            'full_run':{'makespan_ms':summary['makespan_ms'],
                        'device_utilization':summary['fleet_npu_compute_utilization'],
                        'request_count':len(batches),
                        'request_equal_admission_efficiency':mean([c/a for c,a in zip(full_compute,full_processing)]),
                        'request_equal_arrival_efficiency':mean([c/a for c,a in zip(full_compute,full_arrival)])},
            'windows':windows, 'nominal_demand_scan':scan,
            'original_slo':{k:v for k,v in result.get('slo',{}).items() if k!='per_profile'}}


def metric_change(random,ordered):
    return {'random':random,'ordered':ordered,
            'delta_ordered_minus_random_percentage_points':None if random is None or ordered is None else 100*(ordered-random),
            'relative_reduction_vs_random':None if random is None or ordered is None or random==0 else (random-ordered)/random}


def order_pair_audit(inputs,runs,strategies):
    """Different order fingerprints; pair preserved identities/content exactly."""
    by_label={x['label']:x for x in inputs}
    comparisons=[]
    for ordered in inputs:
        meta=ordered['metadata']
        reference=meta.get('comparison_random_label') or meta.get('random_reference_label')
        if not reference:
            continue
        item={'ordered_label':ordered['label'],'random_label':reference,
              'ordered_input_fingerprint':ordered['input_fingerprint'],
              'order_mode':meta.get('order_mode')}
        if reference not in by_label:
            item.update(status='missing_reference_input',input_audit={'passed':False})
            comparisons.append(item)
            continue
        original=by_label[reference]
        item['random_input_fingerprint']=original['input_fingerprint']
        a,b=original['_requests'],ordered['_requests']
        ids=[r['original_request_id'] for r in b.values()]
        checks={'ordered_rows_have_original_request_id':all(r['has_original_request_id'] for r in b.values()),
                'identity_bijection':len(ids)==len(set(ids))==len(a) and set(ids)==set(a),
                'both_all_arrival_zero':original['all_arrival_zero'] and ordered['all_arrival_zero'],
                'reference_fingerprint_metadata_matches':meta.get('random_reference_input_fingerprint',original['input_fingerprint'])==original['input_fingerprint'],
                'reference_manifest_sha_metadata_matches':meta.get('random_reference_manifest_sha256',original['file_sha256'])==original['file_sha256']}
        order_source_checks={name: {'expected':value,'current':sha(BASE/name) if (BASE/name).is_file() else None}
                             for name,value in meta.get('order_source_sha256',{}).items()}
        checks['order_source_hashes_current']=all(x['expected']==x['current'] for x in order_source_checks.values())
        checks['identity_load_placement_and_npu_exact']=checks['identity_bijection'] and all(
            r['content_and_placement_sha256']==a[r['original_request_id']]['content_and_placement_sha256']
            for r in b.values())
        per_npu=[]
        for n in range(NPU):
            ar=[r for r in a.values() if r['npu_id']==n]
            br=[r for r in b.values() if r['npu_id']==n]
            equal=Counter(r['content_and_placement_sha256'] for r in ar)==Counter(r['content_and_placement_sha256'] for r in br)
            seq_a=[r['request_id'] for r in sorted(ar,key=lambda r:r['request_id'])]
            seq_b=[r['original_request_id'] for r in sorted(br,key=lambda r:r['request_id'])]
            per_npu.append({'npu_id':n,'multiset_exact':equal,
                            'request_count_random':len(ar),'request_count_ordered':len(br),
                            'admission_order_changed':seq_a!=seq_b,
                            'random_order_identity_sha256':canonical_sha(seq_a),
                            'ordered_order_identity_sha256':canonical_sha(seq_b)})
        checks['each_npu_multiset_exact']=all(x['multiset_exact'] for x in per_npu)
        checks['static_max_certificate_identical']=all(close(x,y) for x,y in zip(
            original['static_max_proof']['per_ssu_gib_s'],ordered['static_max_proof']['per_ssu_gib_s']))
        item['input_audit']={'passed':all(checks.values()),'checks':checks,'per_npu':per_npu,
                            'order_source_hashes':order_source_checks,
                            'npus_with_changed_admission_order':sum(x['admission_order_changed'] for x in per_npu),
                            'load_fields_ignored_only':['request_id','generation','original_request_id'],
                            'note':'Physical placement compared exactly, never rehashed from new simulator IDs. Simulator IDs encode equal-arrival admission order and also occur in routing/tie-breaking code; this is preserved-input-content order sensitivity, not identical input fingerprint.'}
        item['strategies']=[]
        for strategy in sorted(strategies):
            rr=[r for r in runs if r['label']==original['label'] and r['input_fingerprint']==original['input_fingerprint'] and r['strategy']==strategy]
            ro=[r for r in runs if r['label']==ordered['label'] and r['input_fingerprint']==ordered['input_fingerprint'] and r['strategy']==strategy]
            pair={'strategy':strategy,'status':'pending','random_result_paths':[r['path'] for r in rr],
                  'ordered_result_paths':[r['path'] for r in ro]}
            if len(rr)>1 or len(ro)>1:
                pair['status']='duplicate'
            elif rr and ro:
                r,o=rr[0],ro[0]
                same_sources={k:v['expected'] for k,v in r['audit']['source_hashes'].items()}=={
                    k:v['expected'] for k,v in o['audit']['source_hashes'].items()}
                pair.update(status='complete' if item['input_audit']['passed'] and r['audit']['passed'] and o['audit']['passed'] and same_sources else 'audit_failed',
                            core_source_hashes_identical=same_sources,
                            both_main_windows_valid=r['main_window_valid'] and o['main_window_valid'],
                            both_all_subwindows_active_and_mixed=r['all_subwindows_active_and_mixed'] and o['all_subwindows_active_and_mixed'])
                pair['windows']=[]
                for rw,ow in zip(r['windows'],o['windows']):
                    assert (rw['start_ms'],rw['end_ms'])==(ow['start_ms'],ow['end_ms'])
                    w={'start_ms':rw['start_ms'],'end_ms':rw['end_ms'],
                       'device_utilization':metric_change(rw['device_utilization'],ow['device_utilization']),
                       'request_equal_utilization':metric_change(rw['request_equal_utilization'],ow['request_equal_utilization']),
                       'both_active_and_mixed':all(x['all_npus_active'] and x['each_npu_short_and_long_positive_compute'] for x in (rw,ow)),
                       'cohort_note':'Each window includes its own overlapping request segments; identities are matched over the full finite batch, not necessarily within this window.'}
                    for key in ('by_category','by_role'):
                        w[key]={g:{metric:metric_change(rw[key][g][metric],ow[key][g][metric])
                                   for metric in ('request_equal_utilization','pooled_utilization')}
                                for g in rw[key]}
                    pair['windows'].append(w)
                pair['full_run']={metric:metric_change(r['full_run'][metric],o['full_run'][metric])
                                  for metric in ('device_utilization','request_equal_admission_efficiency','request_equal_arrival_efficiency')}
                pair['makespan_ms']={'random':r['full_run']['makespan_ms'],'ordered':o['full_run']['makespan_ms'],
                                     'ordered_minus_random':o['full_run']['makespan_ms']-r['full_run']['makespan_ms']}
                reduction=pair['windows'][0]['device_utilization']['relative_reduction_vs_random']
                pair['main_device_utilization_relative_reduction_at_least_10_percent']=reduction is not None and reduction>=0.10
                pair['reduction_flag_scope']='The preceding flag is numeric-only. Use valid_main_device_target_met to enforce input/source audit, main-window validity, and all-run nominal capacity.'
                pair['both_full_run_nominal_capacity_satisfied']=all(
                    x['nominal_demand_scan']['full_run']['all_ssu_within_capacity']
                    and x['nominal_demand_scan']['full_run']['all_npu_links_within_capacity']
                    for x in (r,o))
                pair['both_static_max_certificates_within_capacity']=all(
                    x['static_max_proof']['all_ssu_within_capacity']
                    and x['static_max_proof']['all_npu_links_within_capacity']
                    for x in (original,ordered))
                pair['valid_main_device_target_met']=(
                    pair['status']=='complete' and pair['both_main_windows_valid']
                    and pair['both_full_run_nominal_capacity_satisfied']
                    and pair['main_device_utilization_relative_reduction_at_least_10_percent'])
            item['strategies'].append(pair)
        item['status']='input_audit_passed' if item['input_audit']['passed'] else 'input_audit_failed'
        comparisons.append(item)
    return comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--strategies',nargs='+',help='Expected strategies; otherwise plan.json or discovered directories.')
    args=parser.parse_args()
    base=args.base.resolve()
    inputs,errors=[],[]
    for path in sorted((base/'inputs').glob('*.json.gz')):
        try:
            inputs.append(load_input(path,base))
        except Exception as exc:
            errors.append({'path':relative(path,base),'kind':'input_error','error':f'{type(exc).__name__}: {exc}'})
    by_label={x['label']:x for x in inputs}
    assert len(by_label)==len(inputs), 'multiple manifest files claim the same label'
    by_fp=defaultdict(list)
    for info in inputs:
        by_fp[info['input_fingerprint']].append(info)
    aliases=[{'input_fingerprint':fp,
              'manifests':[{'label':i['label'],'file_sha256':i['file_sha256'],
                            'seed':i['metadata'].get('seed'),'path':i['path']} for i in group],
              'explanation':'Same simulator input fingerprint can arise from a deterministic profile/placement order under several metadata seeds. Metadata, original-identity mappings, and submission seeds can differ; retain distinct label+fingerprint+strategy runs. These are not independent bad input orders.'}
             for fp,group in by_fp.items() if len(group)>1]
    expected=set(args.strategies or [])
    if not expected and (base/'plan.json').is_file():
        plan=read_json(base/'plan.json')
        if isinstance(plan,dict):
            expected.update(x for x in plan.get('strategies',[]) if isinstance(x,str))
            for item in plan.get('jobs',[]):
                if isinstance(item,dict) and isinstance(item.get('strategy'),str):expected.add(item['strategy'])
    paths=sorted((base/'runs').glob('*/*/*.json.gz'))
    if not expected:
        expected={'baseline'}
    runs=[]
    for path in paths:
        try:
            payload=read_json(path)
            if 'summary' not in payload:
                errors.append({'path':relative(path,base),'kind':'non_result','error':'missing summary'})
                continue
            fp=payload.get('input_fingerprint')
            label=path.parent.parent.name
            assert label in by_label, 'result directory label has no frozen input'
            assert fp==by_label[label]['input_fingerprint'], 'result label/fingerprint does not match its frozen input'
            runs.append(analyze_result(path,by_label[label],base))
        except Exception as exc:
            errors.append({'path':relative(path,base),'kind':'result_error','error':f'{type(exc).__name__}: {exc}'})
    execution=[]
    for path in sorted((base/'runs').glob('*/*/command.json')):
        try:
            record=read_json(path)
            execution.append({'label':path.parent.parent.name,'strategy':path.parent.name,
                              'status':record.get('status','running_or_queued'),
                              'returncode':record.get('returncode'),
                              'input_fingerprint':record.get('input_fingerprint'),
                              'path':relative(path,base)})
        except Exception as exc:
            errors.append({'path':relative(path,base),'kind':'command_read_error','error':f'{type(exc).__name__}: {exc}'})
    pairings=[]
    for info in inputs:
        matched=[r for r in runs if r['label']==info['label'] and r['input_fingerprint']==info['input_fingerprint']]
        # Explicit expected strategies apply to every input. Additional observed
        # strategies apply only to the label where they were actually executed.
        input_strategies=expected | {p.parent.name for p in paths if p.parent.parent.name==info['label']}
        for strategy in sorted(input_strategies):
            group=[r for r in matched if r['strategy']==strategy]
            status='pending' if not group else 'duplicate' if len(group)>1 else 'complete' if group[0]['audit']['passed'] else 'audit_failed'
            matching_execution=[e for e in execution if e['label']==info['label'] and e['strategy']==strategy]
            if not group and any(e['status'] in ('failed','timeout') or e['returncode'] not in (None,0) for e in matching_execution):
                status='execution_failed'
            if not group and any(e['kind']=='result_error' and Path(e['path']).parent.parent.name==info['label'] and Path(e['path']).parent.name==strategy for e in errors):
                status='result_error'
            pairings.append({'label':info['label'],'input_fingerprint':info['input_fingerprint'],
                             'strategy':strategy,'status':status,
                             'result_paths':[r['path'] for r in group]})
    output={'schema_version':1, 'analysis_source_sha256':sha(Path(__file__)),
            'definitions':{
                'windows_ms':WINDOWS,'warm_condition':'Each NPU fourth completion <=1500 ms; main window starts at2000 ms, allowing at least500 ms settle.',
                'window_validity':'Main-window validity is separate from all eight subwindows being active and containing positive compute from both long and short on every card.',
                'device_U':'sum clipped layer compute / (32*window duration)',
                'request_equal_U':'arithmetic mean of clipped compute/clipped(admission,completion) over every positive-overlap request; not complete-request or arrival efficiency',
                'exposed_stall':'L0 admission->compute_start; L1-7 previous_compute_end->compute_start; clip intervals to window; excludes useful preadmission L0 prefetch.',
                'nominal_demand':'sum over currently admitted requests of1000*per-layer actual SSD bytes/per-layer C_ms; equal-time completions/admissions processed atomically; not actual released backlog or a FIFO no-stall guarantee.',
                'units':'times ms, volume GiB, rates GiB/s, utilizations fractions',
                'capacity_gib_s':{'each_ssu':DISK_RATE,'each_npu_link':LINK_RATE},
                'source_verification':'Result core hashes compared individually against current files; input fingerprints recomputed from frozen manifests; simulator/core semantics are not inferred from other historical runs.'},
            'inputs':[{k:v for k,v in x.items() if not k.startswith('_')} for x in inputs],
            'runs':runs,'pairings':pairings,'input_fingerprint_aliases':aliases,
            'order_comparisons':order_pair_audit(inputs,runs,expected),
            'execution_records':execution,'errors':errors,
            'counts':{'inputs':len(inputs),'expected_strategies':sorted(expected),'parsed_runs':len(runs),
                      'audit_passed':sum(r['audit']['passed'] for r in runs),
                      'main_window_valid':sum(r['main_window_valid'] for r in runs),
                      'pairing_status':dict(Counter(x['status'] for x in pairings))}}
    fields=['label','strategy','submit_seed','status','input_fingerprint','result_path','audit_passed','warmup_passed',
            'start_ms','end_ms','all_npus_active','each_npu_short_and_long_positive_compute','main_window_valid',
            'device_utilization','request_equal_utilization','pooled_utilization','l0_exposed_stall_ms','l1_7_exposed_stall_ms',
            'short_device_compute_contribution','short_request_equal_utilization','short_pooled_utilization',
            'long_device_compute_contribution','long_request_equal_utilization','long_pooled_utilization',
            'nominal_max_ssu_gib_s','nominal_max_npu_link_gib_s','nominal_any_ssu_over_capacity_ms',
            'full_run_nominal_max_ssu_gib_s','static_max_proof_gib_s','makespan_ms','full_run_device_utilization']
    fields += [f'{category}_{metric}' for category in CATEGORIES
               for metric in ('request_count','request_equal_utilization','pooled_utilization')]
    fields += ['random_reference_label','order_mode','pair_input_content_valid','paired_random_device_utilization',
               'paired_device_delta_pp','paired_device_relative_reduction',
               'paired_request_equal_delta_pp','paired_request_equal_relative_reduction']
    order_comparison_by_label={x['ordered_label']:x for x in output['order_comparisons']}
    base.mkdir(parents=True,exist_ok=True)
    tmp=base/'analysis.json.gz.tmp'
    with tmp.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as f:
                json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.write('\n')
    tmp.replace(base/'analysis.json.gz')
    tmp=base/'summary.csv.tmp'
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for pair in pairings:
            group=[r for r in runs if r['label']==pair['label'] and r['input_fingerprint']==pair['input_fingerprint'] and r['strategy']==pair['strategy']]
            if not group:
                writer.writerow({k:pair[k] for k in ('label','strategy','status','input_fingerprint')})
                continue
            for r in group:
                for w in r['windows']:
                    scan=r['nominal_demand_scan'][f"window_{int(w['start_ms'])}_{int(w['end_ms'])}"]
                    row={k:pair[k] for k in ('label','strategy','status','input_fingerprint')}
                    row.update(result_path=r['path'],submit_seed=r['submit_seed'],audit_passed=r['audit']['passed'],warmup_passed=r['warmup']['all_fourth_completions_by_1500'],main_window_valid=r['main_window_valid'])
                    for key in ('start_ms','end_ms','all_npus_active','each_npu_short_and_long_positive_compute','device_utilization','request_equal_utilization','pooled_utilization','l0_exposed_stall_ms','l1_7_exposed_stall_ms'):row[key]=w[key]
                    for role in ('short','long'):
                        g=w['by_role'][role]
                        row[role+'_device_compute_contribution']=g['compute_ms']/(NPU*w['duration_ms'])
                        for key in ('request_equal_utilization','pooled_utilization'):row[role+'_'+key]=g[key]
                    for category in CATEGORIES:
                        for key in ('request_count','request_equal_utilization','pooled_utilization'):
                            row[category+'_'+key]=w['by_category'][category][key]
                    row.update(nominal_max_ssu_gib_s=scan['max_ssu_gib_s'],nominal_max_npu_link_gib_s=scan['max_npu_link_gib_s'],nominal_any_ssu_over_capacity_ms=scan['any_ssu_over_capacity_ms'],
                               full_run_nominal_max_ssu_gib_s=r['nominal_demand_scan']['full_run']['max_ssu_gib_s'],static_max_proof_gib_s=by_label[r['label']]['static_max_proof']['max_ssu_gib_s'],
                               makespan_ms=r['full_run']['makespan_ms'],full_run_device_utilization=r['full_run']['device_utilization'])
                    intervention=order_comparison_by_label.get(pair['label'])
                    if intervention:
                        row.update(random_reference_label=intervention['random_label'],order_mode=intervention.get('order_mode'),
                                   pair_input_content_valid=intervention['input_audit']['passed'])
                        matched=[x for x in intervention.get('strategies',[]) if x['strategy']==pair['strategy']]
                        for x in matched:
                            for paired_window in x.get('windows',[]):
                                if (paired_window['start_ms'],paired_window['end_ms'])==(w['start_ms'],w['end_ms']):
                                    dc=paired_window['device_utilization'];rc=paired_window['request_equal_utilization']
                                    row.update(paired_random_device_utilization=dc['random'],
                                               paired_device_delta_pp=dc['delta_ordered_minus_random_percentage_points'],
                                               paired_device_relative_reduction=dc['relative_reduction_vs_random'],
                                               paired_request_equal_delta_pp=rc['delta_ordered_minus_random_percentage_points'],
                                               paired_request_equal_relative_reduction=rc['relative_reduction_vs_random'])
                    writer.writerow(row)
    tmp.replace(base/'summary.csv')
    print(json.dumps({'counts':output['counts'],'errors':errors},ensure_ascii=False))


if __name__=='__main__':
    main()
