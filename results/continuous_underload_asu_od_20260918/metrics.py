"""Exact admitted-image demand and uncensored finite-request measurements.

Reference demand is V_is/C_i for the currently admitted request. Cross-request
L0 bytes are physically measured but are not counted as a second active image.
Latency references are this simulation's eight layers, not a data-file TTFT.
"""
from collections import defaultdict
import math


def overlap(a, z, left, right):
    return max(0.0, min(z, right) - max(a, left))


def live_summary(context):
    rows = [dict(request_id=q.manifest.request_id, npu_id=q.manifest.npu_id,
                 admission_time_ms=q.admission_time_ms, arrival_time_ms=q.manifest.arrival_time_ms,
                 completion_time_ms=q.completion_time_ms if q.completed else math.inf,
                 own_compute_ms=context.n_layers * q.per_layer_compute_ms)
            for q in context.requests.values() if q.admitted]
    batches = [dict(npu_id=b.npu_id, member_request_ids=list(b.member_request_ids),
                    layer_metrics=[dict(compute_start_ms=m.compute_start_ms,
                                        compute_end_ms=m.compute_start_ms + m.compute_duration_ms)
                                   for m in b.layer_metrics if math.isfinite(m.compute_start_ms)])
               for b in context.microbatches]
    return dict(request_metrics=rows, microbatch_metrics=batches)


def latency_summary(values):
    ordered = sorted(values)
    if not ordered:
        return dict(mean=None, p50=None, p95=None, minimum=None, maximum=None)
    return dict(mean=math.fsum(ordered) / len(ordered),
                p50=ordered[math.ceil(0.50 * len(ordered)) - 1],
                p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
                minimum=ordered[0], maximum=ordered[-1])


def request_stats(rows):
    result = dict(count=len(rows))
    for name, clock in (('admission', 'admission_time_ms'), ('arrival', 'arrival_time_ms')):
        values = [r['completion_time_ms'] - r[clock] for r in rows]
        ratios = [v / r['own_compute_ms'] for v, r in zip(values, rows)]
        slos = {}
        for factor in (1.0, 1.5):
            passed = sum(v <= factor * r['own_compute_ms'] + 1e-9 for v, r in zip(values, rows))
            slos[f'{factor:g}'] = dict(count=len(rows), passed=passed,
                                      percent=100 * passed / len(rows) if rows else None)
        result[name] = dict(latency_ms=latency_summary(values),
                            normalized_latency=latency_summary(ratios), slo=slos)
    return result


def exact_demand(rows, byid, left, right, num_ssu=3, capacity=40.0):
    """Recompute each right-continuous event interval from active identities.

    Comparing >= capacity is literal, without an overload tolerance. Demand
    rates use math.fsum over a deterministic identity ordering to avoid drift
    from repeatedly adding/subtracting floating-point rates across events.
    """
    changes = defaultdict(lambda: {'start': [], 'end': []})
    changes[left]; changes[right]
    rates = {}
    for r in rows:
        a, z = max(left, r['admission_time_ms']), min(right, r['completion_time_ms'])
        if z <= a:
            continue
        rid = r['request_id']; q = byid[rid]
        assert len(q.placement) == 1
        rates[rid] = tuple(math.fsum(v for d, v in q.placement[0] if d == disk)
                           * 1e6 / q.load['per_layer_us'] for disk in range(num_ssu))
        changes[a]['start'].append(rid); changes[z]['end'].append(rid)
    active = set(); segments = []; active_segments = []
    ge_intervals = [[] for _ in range(num_ssu)]
    over = [0.0] * num_ssu; ge = [0.0] * num_ssu; integrals = [0.0] * num_ssu
    minima = [math.inf] * num_ssu; maxima = [0.0] * num_ssu
    all_over = any_over = any_ge = 0.0
    times = sorted(changes)
    for a, z in zip(times, times[1:]):
        active.difference_update(changes[a]['end']); active.update(changes[a]['start'])
        values = [math.fsum(rates[rid][d] for rid in sorted(active)) for d in range(num_ssu)]
        assert len({byid[rid].npu_id for rid in active}) == len(active), 'Multiple active requests per NPU'
        flags = [v > capacity for v in values]; ge_flags = [v >= capacity for v in values]
        dt = z - a
        all_over += dt * all(flags); any_over += dt * any(flags); any_ge += dt * any(ge_flags)
        for d, v in enumerate(values):
            over[d] += dt * flags[d]; ge[d] += dt * ge_flags[d]; integrals[d] += dt * v
            minima[d] = min(minima[d], v); maxima[d] = max(maxima[d], v)
            if ge_flags[d]:
                ge_intervals[d].append(dict(start_ms=a, end_ms=z, demand_GiB_s=v))
        segments.append([a, z, *values]); active_segments.append([a, z, len(active)])
    duration = right - left
    return dict(definition='sum current admitted request actual per-SSU layer GiB / pure layer seconds; no double-counted next-request L0',
                capacity_GiB_s=capacity, comparison='strict underload requires D_s < capacity; >= intervals include equality',
                per_disk_mean_GiB_s=[v / duration for v in integrals],
                per_disk_min_GiB_s=minima, per_disk_max_GiB_s=maxima,
                per_disk_overload_percent=[100 * v / duration for v in over],
                per_disk_at_or_above_capacity_percent=[100 * v / duration for v in ge],
                all_disks_overload_percent=100 * all_over / duration,
                any_disk_overload_percent=100 * any_over / duration,
                any_disk_at_or_above_capacity_percent=100 * any_ge / duration,
                strict_underload_all_disks=all(v == 0 for v in ge),
                minimum_capacity_margin_GiB_s=[capacity - v for v in maxima],
                segments=segments, segment_columns=['start_ms', 'end_ms', 'SSU0_GiB_s', 'SSU1_GiB_s', 'SSU2_GiB_s'],
                at_or_above_capacity_intervals_by_ssu=ge_intervals,
                active_npu_count_segments=active_segments)


def summarize(summary, manifests, left, right, *, full=False):
    assert right > left
    byid = {q.request_id: q for q in manifests}
    rows = summary['request_metrics']; batches = summary['microbatch_metrics']
    cohort = rows if full else [r for r in rows if left <= r['admission_time_ms'] < right]
    assert all(math.isfinite(r['completion_time_ms']) for r in cohort), 'SLO cohort must drain fully'
    duration = right - left
    compute = [0.0] * 32; active = [0.0] * 32
    req_compute = defaultdict(float); req_active = defaultdict(float)
    for r in rows:
        q = byid[r['request_id']]
        t = overlap(r['admission_time_ms'], r['completion_time_ms'], left, right)
        active[q.npu_id] += t; req_active[q.request_id] += t
    for b in batches:
        assert len(b['member_request_ids']) == 1
        q = byid[b['member_request_ids'][0]]
        for layer in b['layer_metrics']:
            if math.isfinite(layer['compute_start_ms']):
                t = overlap(layer['compute_start_ms'], layer['compute_end_ms'], left, right)
                compute[q.npu_id] += t; req_compute[q.request_id] += t
    finished = [r for r in rows if left <= r['completion_time_ms'] < right
                or (full and r['completion_time_ms'] == right)]
    stats = request_stats(cohort)
    group_specs = dict(length=lambda q: str(q.load['seq_len_k']),
                       category=lambda q: q.load['category'],
                       profile=lambda q: f"{q.load['seq_len_k']}:{q.load['nql']}")
    grouped = {}
    for dimension, key in group_specs.items():
        entries = {}
        for value in sorted({key(q) for q in manifests}):
            ids = {q.request_id for q in manifests if key(q) == value}
            entry = request_stats([r for r in cohort if r['request_id'] in ids])
            c = math.fsum(req_compute[rid] for rid in ids)
            a = math.fsum(req_active[rid] for rid in ids)
            entry.update(compute_ms=c, active_ms=a, active_U_percent=100*c/a if a else None)
            entries[value] = entry
        grouped[dimension] = entries
    return dict(start_ms=left, end_ms=right, warm_start_s=None if full else left/1000,
                warm_end_s=None if full else right/1000, full_population=full,
                U_percent=100 * math.fsum(compute) / (32 * duration),
                per_npu_U_percent=[100*x/duration for x in compute],
                per_npu_active_ms=active,
                all_npus_active=all(abs(x-duration)<1e-7 for x in active),
                admitted_in_window=sum(left <= r['admission_time_ms'] < right for r in rows),
                completed_in_window=len(finished), cohort_request_ids=sorted(r['request_id'] for r in cohort),
                completed_after_window=sum(r['completion_time_ms'] > right for r in cohort),
                request_statistics=stats, slo=stats['admission']['slo']['1.5'],
                arrival_slo=stats['arrival']['slo']['1.5'],
                slo_by_category={k:v['admission']['slo']['1.5'] for k,v in grouped['category'].items()},
                slo_by_profile={k:v['admission']['slo']['1.5'] for k,v in grouped['profile'].items()},
                category_active_U_percent={k:v['active_U_percent'] for k,v in grouped['category'].items()},
                by_length=grouped['length'], by_category=grouped['category'], by_profile=grouped['profile'],
                timely_completions_per_second=request_stats(finished)['admission']['slo']['1.5']['passed']/(duration/1000),
                total_completions_per_second=len(finished)/(duration/1000),
                measurement_definition=dict(latency='completion minus admission or original arrival; no first-token event simulated',
                    own_compute='8 × original data per-layer computation; no data TTFT/layer-count inferred',
                    quantiles='nearest rank ceil(p*n), no interpolation',
                    slo_tolerance_ms=1e-9, cohorts='window admissions, followed to completion; full population when full=true'),
                demand=exact_demand(rows, byid, left, right))
