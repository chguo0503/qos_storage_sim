#!/usr/bin/env python3
"""Offline audit of raw A=(128,256), B=(32,4096), per-card request ratio 1:2.

Reads existing complete simulator summaries only.  Scientific failures (idle,
nonmixed windows, nominal overload) are retained separately from technical
failures.  No simulation, core edits, role-based fast/bridge assumptions, or
conversion of read lifetime to physical SSD throughput.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from run_baseline_npu32_stress import load_manifest
from continuous_batch_sim import continuous_batch_input_fingerprint

KEYS = {(128, 256): 'A', (32, 4096): 'B'}
BLOCK_GIB = 176 * 1024 / 2**30


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 2**20), b''):
            digest.update(part)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def overlap(a, b, left, right):
    return max(0., min(b, right) - max(a, left))


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)


def cohort(rows, info, layers, alpha, right):
    result = dict(count=len(rows), completion_after_window_count=sum(r['completion_time_ms'] > right for r in rows),
                  request_ids=sorted(r['request_id'] for r in rows))
    for clock in ('arrival', 'admission'):
        passed = sum(r['completion_time_ms'] - r[clock + '_time_ms'] <=
                     alpha * layers * info[r['request_id']]['C_ms'] + 1e-9 for r in rows)
        result[clock + '_clock'] = dict(passed=passed, count=len(rows),
            slo_percent=100 * passed / len(rows) if rows else None)
    return result


def scan_segments(request_rows, info, npu_count, ssu_count):
    events = defaultdict(lambda: dict(add=[], remove=[]))
    for r in request_rows:
        events[r['admission_time_ms']]['add'].append(r['request_id'])
        events[r['completion_time_ms']]['remove'].append(r['request_id'])
    times = sorted(events)
    active = {}
    segments = []
    for i, t in enumerate(times):
        for rid in events[t]['remove']:
            assert active.pop(info[rid]['npu']) == rid, 'nonmatching completion/removal'
        for rid in events[t]['add']:
            npu = info[rid]['npu']
            assert npu not in active, 'overlapping active requests on one NPU'
            active[npu] = rid
        if i + 1 == len(times):
            break
        end = times[i + 1]
        assert end > t
        segments.append(dict(start_ms=t, end_ms=end,
            per_ssu_gib_s=[math.fsum(info[r]['rates'][s] for r in active.values()) for s in range(ssu_count)],
            link_max_gib_s=max((info[r]['rate_total'] for r in active.values()), default=0.),
            K_A=sum(info[r]['class'] == 'A' for r in active.values()), K_active=len(active),
            request_ids=tuple(active.values())))
    assert not active
    return segments


def nominal_window(segments, left, right, ssu_count, npu_count, disk_cap, link_cap):
    peak = [0.] * ssu_count
    integral = [0.] * ssu_count
    over = [0.] * ssu_count
    equal = [0.] * ssu_count
    witness = [None] * ssu_count
    k_hist = defaultdict(float)
    a_hist = defaultdict(float)
    covered = any_over = link_over = ka_integral = active_integral = 0.
    link_peak = 0.
    for seg in segments:
        dt = overlap(seg['start_ms'], seg['end_ms'], left, right)
        if not dt:
            continue
        covered += dt
        ka_integral += dt * seg['K_A']
        active_integral += dt * seg['K_active']
        k_hist[seg['K_A']] += dt
        a_hist[seg['K_active']] += dt
        for s, value in enumerate(seg['per_ssu_gib_s']):
            integral[s] += value * dt / 1000
            over[s] += dt if value > disk_cap else 0.
            equal[s] += dt if value == disk_cap else 0.
            if value > peak[s]:
                peak[s] = value
                witness[s] = dict(start_ms=max(left, seg['start_ms']), end_ms=min(right, seg['end_ms']),
                    rate_gib_s=value, K_A=seg['K_A'], active_request_ids=list(seg['request_ids']))
        any_over += dt if max(seg['per_ssu_gib_s']) > disk_cap else 0.
        link_peak = max(link_peak, seg['link_max_gib_s'])
        link_over += dt if seg['link_max_gib_s'] > link_cap else 0.
    missing = max(0., right - left - covered)
    if missing > 1e-7:
        k_hist[0] += missing
        a_hist[0] += missing
    duration = right - left
    return dict(per_ssu_peak_gib_s=peak, max_ssu_gib_s=max(peak),
        per_ssu_nominal_integral_gib=integral,
        per_ssu_mean_gib_s=[x * 1000 / duration for x in integral],
        per_ssu_over_capacity_ms=over, per_ssu_equal_capacity_ms=equal,
        any_ssu_over_capacity_ms=any_over, per_ssu_peak_witness=witness,
        max_npu_link_nominal_gib_s=link_peak, any_npu_link_over_capacity_ms=link_over,
        strict_underload=max(peak) < disk_cap and link_peak < link_cap,
        within_capacity=max(peak) <= disk_cap and link_peak <= link_cap,
        K_A_peak=max(k_hist, default=0), K_A_mean=ka_integral / duration,
        K_A_card_ms=ka_integral, K_A_duration_ms={str(k): v for k, v in sorted(k_hist.items())},
        K_active_mean=active_integral / duration, active_card_ms=active_integral,
        K_active_duration_ms={str(k): v for k, v in sorted(a_hist.items())},
        all_32_active=close(active_integral, npu_count * duration))


def analyze(args):
    man, raw = read(args.manifest), read(args.result)
    summary = raw['summary']
    num_npu, num_ssu, n_layers = (int(summary[k]) for k in ('num_npu', 'num_ssu', 'n_layers'))
    makespan = float(summary['makespan_ms'])
    assert makespan > 0 and num_npu == 32 and num_ssu > 0 and n_layers == 8
    assert summary['batch_size'] == 1
    table = ast.literal_eval((ROOT / 'data').read_text())
    requests, metadata = load_manifest(args.manifest)
    failures = []
    checks = {}

    def check(name, condition, detail=None):
        checks[name] = bool(condition)
        if not condition:
            failures.append(dict(check=name, detail=detail))

    check('dimensions_match_manifest', metadata['num_npu'] == num_npu and metadata['num_ssu'] == num_ssu and metadata['n_layers'] == n_layers)
    check('capacity_thresholds_match_recorded_hardware',
          args.disk_capacity_gib_s == metadata['disk_bw_gib_s'] == 40. and
          args.link_capacity_gib_s == metadata['npu_bw_gib_s'] == 50.)
    check('baseline_fixed_assignment', raw.get('strategy') == 'baseline' and
          raw.get('policy_config', {}).get('assignment') == 'fixed')
    check('authenticated_data_hash', metadata.get('source_data_sha256') == sha(ROOT / 'data'))
    check('required_provenance_present', bool(raw.get('core_and_policy_sha256')) and
          bool(raw.get('stress_runner_sha256')) and 'submit_seed' in raw)
    check('source_input_fingerprint', raw.get('input_fingerprint') == man['input_fingerprint'] == continuous_batch_input_fingerprint(requests))
    check('source_invariants_all_true', bool(summary.get('invariants')) and all(summary['invariants'].values()))
    check('seed_matches', raw.get('submit_seed') == metadata['seed'])
    if 'core_and_policy_sha256' in raw:
        sources = raw['core_and_policy_sha256']
        check('recorded_core_sources_unchanged', bool(sources) and all((ROOT / p).is_file() and sha(ROOT / p) == value for p, value in sources.items()))
    if 'stress_runner_sha256' in raw:
        check('stress_runner_unchanged', sha(ROOT / 'run_baseline_npu32_stress.py') == raw['stress_runner_sha256'])
    if 'execution_placement_fingerprint' in raw:
        check('execution_placement_unchanged', raw['execution_placement_fingerprint'] == raw['input_placement_fingerprint'])
    info = {}
    counts = [Counter() for _ in range(num_npu)]
    original_ids = []
    for r in requests:
        key = (int(r.load['seq_len_k']), int(r.load['nql']))
        assert key in KEYS, ('unexpected profile', key)
        assert r.request_id not in info and 0 <= r.npu_id < num_npu
        c = float(r.load['per_layer_us']) / 1000
        assert c > 0
        vectors = [[math.fsum(v for disk, v in layer if disk == s) for s in range(num_ssu)] for layer in r.placement]
        assert len(vectors) in (1, n_layers)
        assert all(0 <= disk < num_ssu and value == BLOCK_GIB for layer in r.placement for disk, value in layer)
        assert all(vector == vectors[0] for vector in vectors), 'D/C sweep requires unchanged per-layer disk vector'
        assert r.load['per_layer_us'] == table[key][1]
        assert all(abs(math.fsum(vector) - table[key][3]) < 1e-12 for vector in vectors)
        assert close(float(r.load['per_layer_kv_gb']), math.fsum(vectors[0]))
        assert all(len(layer) == (key[0] * 1024 - key[1]) // 128 for layer in r.placement)
        expected_category = ('S' if key[0] <= 80 else 'L') + ('S' if key[1] < 512 else 'L')
        assert r.load['category'] == expected_category
        counts[r.npu_id][KEYS[key]] += 1
        original = r.load.get('original_request_id', r.request_id)
        original_ids.append(original)
        info[r.request_id] = dict(npu=r.npu_id, **{'class': KEYS[key]}, profile=list(key), C_ms=c,
            D_gib=math.fsum(vectors[0]), rates=[v * 1000 / c for v in vectors[0]],
            rate_total=math.fsum(vectors[0]) * 1000 / c, original_request_id=original,
            arrival_ms=float(r.arrival_time_ms))
    check('every_card_request_ratio_A1_B2', all(x['A'] > 0 and x['B'] == 2 * x['A'] for x in counts))
    check('original_identity_unique', len(set(original_ids)) == len(original_ids))
    check('all_arrivals_zero', all(x['arrival_ms'] == 0 for x in info.values()))
    request_rows = summary['request_metrics']
    byid = {r['request_id']: r for r in request_rows}
    assert len(byid) == len(request_rows) == len(info) and set(byid) == set(info)
    assert all(math.isfinite(r['completion_time_ms']) for r in request_rows)
    lanes = [[] for _ in range(num_npu)]
    layer_rows = [[] for _ in range(num_npu)]
    batch_ids = []
    timing_good = True
    for batch in summary['microbatch_metrics']:
        assert batch['batch_size'] == len(batch['member_request_ids']) == 1
        rid = batch['member_request_ids'][0]
        batch_ids.append(rid)
        inp, req = info[rid], byid[rid]
        npu = inp['npu']
        admit, complete = batch['admission_time_ms'], batch['completion_time_ms']
        timing_good &= batch['npu_id'] == req['npu_id'] == npu and close(admit, req['admission_time_ms']) and close(complete, req['completion_time_ms'])
        timing_good &= req['arrival_time_ms'] == inp['arrival_ms'] and req['arrival_time_ms'] <= admit <= complete
        timing_good &= close(req['own_compute_ms'], n_layers * inp['C_ms'])
        previous = admit
        metrics = sorted(batch['layer_metrics'], key=lambda l: l['layer'])
        assert [l['layer'] for l in metrics] == list(range(n_layers))
        for layer in metrics:
            start, end = layer['compute_start_ms'], layer['compute_end_ms']
            timing_good &= previous <= start + 1e-7 and start < end <= complete + 1e-7
            timing_good &= close(end - start, inp['C_ms']) and close(start - previous, layer['io_barrier_wait_ms'])
            timing_good &= layer['io_start_time_ms'] <= layer['io_ready_time_ms'] + 1e-7 <= start + 2e-7
            layer_rows[npu].append(dict(request_id=rid, npu=npu, **{'class': inp['class']}, layer=layer['layer'],
                compute_start_ms=start, compute_end_ms=end, wait_start_ms=previous,
                io_release_ms=layer['io_start_time_ms'], io_ready_ms=layer['io_ready_time_ms']))
            previous = end
        timing_good &= close(previous, complete)
        lanes[npu].append(dict(request_id=rid, **{'class': inp['class']}, start_ms=admit, end_ms=complete,
            final_compute_start_ms=metrics[-1]['compute_start_ms'], final_compute_end_ms=metrics[-1]['compute_end_ms']))
    check('complete_batch_population', len(batch_ids) == len(set(batch_ids)) == len(info) and set(batch_ids) == set(info))
    check('all_timing_and_compute_fields', timing_good)
    previous_by_id = {}
    for lane in lanes:
        lane.sort(key=lambda r: r['start_ms'])
        assert all(x['end_ms'] <= y['start_ms'] + 1e-7 for x, y in zip(lane, lane[1:]))
        for i, item in enumerate(lane):
            previous_by_id[item['request_id']] = lane[i - 1] if i else None
    for items in layer_rows:
        for row in items:
            previous = previous_by_id[row['request_id']]
            row['predecessor_class'] = previous['class'] if previous else 'initial'
    check('full_compute_matches_input', close(math.fsum(l['compute_end_ms'] - l['compute_start_ms'] for items in layer_rows for l in items),
        math.fsum(n_layers * r['C_ms'] for r in info.values())))
    check('makespan_matches_last_completion', close(makespan, max(r['completion_time_ms'] for r in request_rows)))
    segments = scan_segments(request_rows, info, num_npu, num_ssu)
    windows = [(0., makespan)]
    omitted = []
    desired = [(2000., 4000.), (2000., 12000.)]
    if args.window:
        desired += [tuple(map(float, w.split(':'))) for w in args.window]
    desired += [(float(x), float(x + 2000)) for x in range(4000, 12000, 2000)]
    for left, right in desired:
        assert 0 <= left < right
        if (left, right) in windows:
            continue
        if right <= makespan + 1e-7:
            windows.append((left, right))
        else:
            omitted.append(dict(start_ms=left, end_ms=right, reason='result makespan does not cover requested end; not padded with artificial idle'))
    output_windows = []
    transition_csv = []
    for left, right in windows:
        duration = right - left
        cards = []
        for npu, (lane, ll) in enumerate(zip(lanes, layer_rows)):
            active = math.fsum(overlap(r['start_ms'], r['end_ms'], left, right) for r in lane)
            c = math.fsum(overlap(l['compute_start_ms'], l['compute_end_ms'], left, right) for l in ll)
            parts = {}
            for cls in ('A', 'B'):
                c0 = math.fsum(overlap(l['compute_start_ms'], l['compute_end_ms'], left, right) for l in ll if l['class'] == cls)
                a0 = math.fsum(overlap(r['start_ms'], r['end_ms'], left, right) for r in lane if r['class'] == cls)
                l0 = math.fsum(overlap(l['wait_start_ms'], l['compute_start_ms'], left, right) for l in ll if l['class'] == cls and l['layer'] == 0)
                internal = math.fsum(overlap(l['wait_start_ms'], l['compute_start_ms'], left, right) for l in ll if l['class'] == cls and l['layer'] > 0)
                assert close(c0 + l0 + internal, a0), ('class time accounting', npu, cls, left, right)
                parts[cls] = dict(compute_ms=c0, active_ms=a0, L0_stall_ms=l0, internal_stall_ms=internal,
                    conditional_utilization_percent=100 * c0 / a0 if a0 else None,
                    compute_share_of_card=c0 / c if c else None,
                    admissions_count=sum(left <= r['start_ms'] < right and r['class'] == cls for r in lane))
            assert close(c + sum(p['L0_stall_ms'] + p['internal_stall_ms'] for p in parts.values()), active)
            cards.append(dict(npu=npu, compute_ms=c, active_ms=active, idle_ms=max(0., duration - active),
                device_utilization_percent=100 * c / duration, both_A_and_B_positive_compute=all(parts[k]['compute_ms'] > 0 for k in parts),
                classes=parts))
        c = math.fsum(x['compute_ms'] for x in cards)
        active = math.fsum(x['active_ms'] for x in cards)
        cap = nominal_window(segments, left, right, num_ssu, num_npu, args.disk_capacity_gib_s, args.link_capacity_gib_s)
        assert close(active, cap['active_card_ms'])
        assert close(math.fsum(x['classes']['A']['active_ms'] for x in cards), cap['K_A_card_ms'])
        admitted = [r for r in request_rows if left <= r['admission_time_ms'] < right]
        arrived = [r for r in request_rows if left <= r['arrival_time_ms'] < right]
        cls_out = {}
        for cls in ('A', 'B'):
            cc = math.fsum(x['classes'][cls]['compute_ms'] for x in cards)
            aa = math.fsum(x['classes'][cls]['active_ms'] for x in cards)
            cls_out[cls] = dict(compute_ms=cc, active_ms=aa, mean_active_cards=aa / duration,
                conditional_utilization_percent=100 * cc / aa if aa else None, device_utilization_contribution_pp=100 * cc / (num_npu * duration),
                compute_share_of_fleet=cc / c if c else None,
                L0_stall_ms=math.fsum(x['classes'][cls]['L0_stall_ms'] for x in cards),
                internal_stall_ms=math.fsum(x['classes'][cls]['internal_stall_ms'] for x in cards),
                warm_admission_SLO=cohort([r for r in admitted if info[r['request_id']]['class'] == cls], info, n_layers, args.slo_alpha, right))
        transitions = {}
        for source in ('initial', 'A', 'B'):
            for dest in ('A', 'B'):
                selected = [l for ll in layer_rows for l in ll if l['layer'] == 0 and l['class'] == dest and l['predecessor_class'] == source]
                wait = math.fsum(overlap(l['wait_start_ms'], l['compute_start_ms'], left, right) for l in selected)
                transitions[source + '_to_' + dest] = dict(L0_stall_ms=wait,
                    L0_compute_starts_inside=sum(left <= l['compute_start_ms'] < right for l in selected),
                    stalled_L0_intersecting_window=sum(overlap(l['wait_start_ms'], l['compute_start_ms'], left, right) > 0 for l in selected))
                transition_csv.append(dict(start_ms=left, end_ms=right, predecessor=source, target_class=dest, **transitions[source + '_to_' + dest]))
        assert close(sum(v['L0_stall_ms'] for v in transitions.values()), sum(x['L0_stall_ms'] for x in cls_out.values()))
        output_windows.append(dict(start_ms=left, end_ms=right, full_run=(left == 0 and right == makespan),
            compute_ms=c, active_ms=active, idle_ms=num_npu * duration - active,
            device_utilization_percent=100 * c / (num_npu * duration),
            io_stall_ms=active - c, all_32_active=all(close(x['active_ms'], duration) for x in cards),
            mixed_card_count=sum(x['both_A_and_B_positive_compute'] for x in cards),
            classes=cls_out, nominal=cap, per_npu=cards, L0_transitions=transitions,
            cohorts=dict(window_admissions=cohort(admitted, info, n_layers, args.slo_alpha, right),
                         window_arrivals=cohort(arrived, info, n_layers, args.slo_alpha, right))))
    profile_rows = []
    for key, cls in KEYS.items():
        c = table[key][1] / 1000
        profile_rows.append(dict(name=cls, total_length_K=key[0], new_tokens=key[1], hit_tokens=key[0] * 1024 - key[1],
            layer_C_ms=c, layer_D_MiB=table[key][3] * 1024, nominal_D_over_C_gib_s=table[key][3] * 1000 / c))
    main = next((w for w in output_windows if w['start_ms'] == 2000 and w['end_ms'] == 4000), None)
    audit = dict(status='complete', technical_passed=not failures, checks=checks, failed_checks=failures,
        source=dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest), result=str(args.result.resolve()),
            result_sha256=sha(args.result), data_sha256=sha(ROOT / 'data'), analyzer_sha256=sha(__file__)),
        dimensions=dict(num_npu=num_npu, num_ssu=num_ssu, n_layers=n_layers, batch_size=1,
            disk_capacity_gib_s=args.disk_capacity_gib_s, npu_link_capacity_gib_s=args.link_capacity_gib_s),
        strategy=raw.get('strategy'), seed=metadata.get('seed'), order=metadata.get('order', metadata.get('mode')),
        label=metadata.get('label', metadata.get('case_id')), makespan_ms=makespan,
        input_fingerprint=man['input_fingerprint'], request_count=len(info), profiles=profile_rows,
        input_per_npu=[dict(npu=n, A_count=x['A'], B_count=x['B'], ideal_compute_ms=math.fsum(n_layers * v['C_ms'] for v in info.values() if v['npu'] == n)) for n, x in enumerate(counts)],
        windows=output_windows, omitted_windows=omitted,
        primary_2_4s_conditions=None if main is None else dict(all_32_active=main['all_32_active'], all_32_mixed=main['mixed_card_count'] == 32,
            nominal_strict_underload=main['nominal']['strict_underload']),
        full_population_SLO=cohort(request_rows, info, n_layers, args.slo_alpha, makespan),
        definitions=dict(classification='A and B identified by exact data keys, independent of legacy short/long/bridge fields. A is LS; B is SL.',
            device_U='Sum of clipped actual layer compute / (32 * wall-clock window length); idle remains in denominator.',
            class_U='Class actual compute / class admitted-active time; distinct from its contribution to device U.',
            stalls='L0: admission to first compute start. Internal: preceding layer compute end to current compute start. Clip these intervals to window; never filter by IO release or count pre-admission delay as NPU stall.',
            L0_transitions='Class of previous completed request on same NPU. First request uses initial. B after A is separate from B after B. Sums are clipped actual waits, not entire read lifetimes.',
            nominal='Exact current admitted request actual per-layer per-SSU D/C on every positive event interval; same-time completions removed and admissions added as one batch. Does not additionally count next-request L0. Integrated nominal D/C is not actual transferred bytes.',
            physical_bandwidth='Not inferred from request/layer lifetime. Physical SSD and link bandwidth require block service timestamps and nonoverlap/byte-conservation checks in a separate trace analysis.',
            SLO=f'For each stated cohort, complete-admission or complete-arrival <= {args.slo_alpha} * {n_layers} * original layer C. Window admissions are followed to final completion; strategies/orders may admit different cohorts. Arrival-clock results include queueing. Neither is a measured first-token event.',
            capacity_status='Scientific condition reported for full run and each window; overload/nonmixed/idle is never silently filtered or treated as a technical simulation failure.',
            input_identity='Random/ordered queue positions may have different request_id; original_request_id identifies the matching original request across orderings.'),
        created_utc=datetime.now(timezone.utc).isoformat(), no_simulation_run=True)
    flat = []
    per_npu = []
    for w in output_windows:
        row = dict(label=audit['label'], order=audit['order'], strategy=audit['strategy'], seed=audit['seed'], num_ssu=num_ssu,
            start_ms=w['start_ms'], end_ms=w['end_ms'], full_run=w['full_run'], technical_passed=audit['technical_passed'],
            device_utilization_percent=w['device_utilization_percent'], compute_ms=w['compute_ms'], io_stall_ms=w['io_stall_ms'], idle_ms=w['idle_ms'],
            all_32_active=w['all_32_active'], mixed_card_count=w['mixed_card_count'], nominal_strict_underload=w['nominal']['strict_underload'],
            nominal_peak_gib_s=w['nominal']['max_ssu_gib_s'], nominal_any_ssu_over_capacity_ms=w['nominal']['any_ssu_over_capacity_ms'],
            K_A_peak=w['nominal']['K_A_peak'], K_A_mean=w['nominal']['K_A_mean'],
            warm_count=w['cohorts']['window_admissions']['count'],
            warm_admission_slo_percent=w['cohorts']['window_admissions']['admission_clock']['slo_percent'],
            warm_arrival_clock_slo_percent=w['cohorts']['window_admissions']['arrival_clock']['slo_percent'])
        for cls in ('A', 'B'):
            for key in ('compute_ms', 'active_ms', 'conditional_utilization_percent', 'L0_stall_ms', 'internal_stall_ms', 'mean_active_cards'):
                row[cls + '_' + key] = w['classes'][cls][key]
            row[cls + '_warm_admission_slo_percent'] = w['classes'][cls]['warm_admission_SLO']['admission_clock']['slo_percent']
        for key in ('A_to_B', 'B_to_B', 'initial_to_B'):
            row[key + '_L0_stall_ms'] = w['L0_transitions'][key]['L0_stall_ms']
        flat.append(row)
        for card in w['per_npu']:
            cardrow = dict(start_ms=w['start_ms'], end_ms=w['end_ms'], **{k: v for k, v in card.items() if k != 'classes'})
            for cls, values in card['classes'].items():
                for key, value in values.items():
                    cardrow[cls + '_' + key] = value
            per_npu.append(cardrow)
    return audit, flat, per_npu, transition_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', action='append', help='Additional START:END in ms')
    parser.add_argument('--disk-capacity-gib-s', type=float, default=40.)
    parser.add_argument('--link-capacity-gib-s', type=float, default=50.)
    parser.add_argument('--slo-alpha', type=float, default=1.5)
    args = parser.parse_args()
    assert min(args.disk_capacity_gib_s, args.link_capacity_gib_s, args.slo_alpha) > 0
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        audit, rows, per_npu, transitions = analyze(args)
        write(args.out / 'audit.json', audit)
        write_csv(args.out / 'summary.csv', rows)
        write_csv(args.out / 'per_npu.csv', per_npu)
        write_csv(args.out / 'transition_stalls.csv', transitions)
        print(json.dumps(dict(out=str(args.out), technical_passed=audit['technical_passed'],
            num_ssu=audit['dimensions']['num_ssu'], primary=audit['primary_2_4s_conditions'],
            windows=[{k: row[k] for k in ('start_ms','end_ms','device_utilization_percent','B_L0_stall_ms','B_internal_stall_ms','nominal_strict_underload','K_A_mean')} for row in rows]), ensure_ascii=False))
        if not audit['technical_passed']:
            raise SystemExit(1)
    except Exception as exc:
        write(args.out / 'audit_error.json', dict(status='audit_error', technical_passed=False,
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
            manifest=str(args.manifest), result=str(args.result), no_simulation_run=True))
        raise


if __name__ == '__main__':
    main()
