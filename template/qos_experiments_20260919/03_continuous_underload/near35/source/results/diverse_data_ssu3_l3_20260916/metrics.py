"""Request-owned reference demand and uncensored admission-clock SLO."""
from collections import defaultdict
import math


def overlap(a, z, left, right):
    return max(0., min(z, right) - max(a, left))


def live_summary(context):
    rows = [dict(request_id=q.manifest.request_id, admission_time_ms=q.admission_time_ms,
                 arrival_time_ms=q.manifest.arrival_time_ms,
                 completion_time_ms=q.completion_time_ms if q.completed else math.inf,
                 own_compute_ms=context.n_layers*q.per_layer_compute_ms)
            for q in context.requests.values() if q.admitted]
    batches = [dict(npu_id=b.npu_id, member_request_ids=list(b.member_request_ids),
                    admission_time_ms=b.admission_time_ms,
                    completion_time_ms=b.completion_time_ms,
                    layer_metrics=[dict(compute_start_ms=m.compute_start_ms,
                        compute_end_ms=m.compute_start_ms+m.compute_duration_ms)
                        for m in b.layer_metrics if math.isfinite(m.compute_start_ms)])
               for b in context.microbatches]
    return dict(request_metrics=rows, microbatch_metrics=batches)


def summarize(summary, manifests, left, right, *, full=False):
    byid = {q.request_id:q for q in manifests}
    rows = summary['request_metrics']
    cohort = rows if full else [r for r in rows if left <= r['admission_time_ms'] < right]
    assert cohort and all(math.isfinite(r['completion_time_ms']) for r in cohort)
    def stats(part, clock='admission_time_ms'):
        passed = sum(r['completion_time_ms']-r[clock] <= 1.5*r['own_compute_ms']+1e-9 for r in part)
        return dict(count=len(part), passed=passed, percent=100*passed/len(part) if part else None)
    compute = [0.]*32
    active = [0.]*32
    category_work = defaultdict(lambda:[0.,0.])
    for r in rows:
        q = byid[r['request_id']]
        t = overlap(r['admission_time_ms'],r['completion_time_ms'],left,right)
        active[q.npu_id] += t
        category_work[q.load['category']][1] += t
    for b in summary['microbatch_metrics']:
        q = byid[b['member_request_ids'][0]]
        for m in b['layer_metrics']:
            if math.isfinite(m['compute_start_ms']):
                t=overlap(m['compute_start_ms'],m['compute_end_ms'],left,right)
                compute[q.npu_id] += t
                category_work[q.load['category']][0] += t
    # One active request image per card, including its exposed stalls. The
    # next request's pre-admission prefetch is not counted twice as demand.
    changes = defaultdict(lambda:[0.]*3)
    changes[left]; changes[right]
    placement_rates = {}
    for r in rows:
        a,z=max(left,r['admission_time_ms']),min(right,r['completion_time_ms'])
        if z<=a: continue
        q=byid[r['request_id']]
        key=(q.placement,q.load['per_layer_us'])
        if key not in placement_rates:
            assert len(q.placement)==1, 'This demand reference assumes the same placement every layer'
            rates=[0.]*3
            for disk,amount in q.placement[0]:rates[disk]+=amount*1e6/q.load['per_layer_us']
            placement_rates[key]=rates
        for d,v in enumerate(placement_rates[key]):
            changes[a][d]+=v;changes[z][d]-=v
    edges=sorted(changes);rates=[0.]*3;segments=[]
    over=[0.]*3;means=[0.]*3;all_over=any_over=0.
    min_rates=[math.inf]*3;max_rates=[0.]*3
    for a,z in zip(edges,edges[1:]):
        rates=[rates[d]+changes[a][d] for d in range(3)]
        flags=[v>40+1e-8 for v in rates];dt=z-a
        all_over+=dt*all(flags);any_over+=dt*any(flags)
        for d in range(3):
            over[d]+=dt*flags[d];means[d]+=dt*rates[d]
            min_rates[d]=min(min_rates[d],rates[d]);max_rates[d]=max(max_rates[d],rates[d])
        segments.append([a,z,*rates])
    duration=right-left
    finished=[r for r in rows if left<=r['completion_time_ms']<right or (full and r['completion_time_ms']==right)]
    classes=sorted({q.load['category'] for q in manifests})
    profile_keys=sorted({(q.load['seq_len_k'],q.load['nql']) for q in manifests})
    bycategory={cat:stats([r for r in cohort if byid[r['request_id']].load['category']==cat]) for cat in classes}
    byprofile={f'{seq}:{miss}':stats([r for r in cohort if
        (byid[r['request_id']].load['seq_len_k'],byid[r['request_id']].load['nql'])==(seq,miss)])
        for seq,miss in profile_keys}
    return dict(start_ms=left,end_ms=right,U_percent=100*sum(compute)/(32*duration),
        per_npu_U_percent=[100*x/duration for x in compute],
        all_npus_active=all(abs(x-duration)<1e-7 for x in active),
        slo=stats(cohort),arrival_slo=stats(cohort,'arrival_time_ms'),
        slo_by_category=bycategory,slo_by_profile=byprofile,
        category_active_U_percent={c:100*x[0]/x[1] if x[1] else None for c,x in category_work.items()},
        cohort_request_ids=sorted(r['request_id'] for r in cohort),
        completed_after_window=sum(r['completion_time_ms']>right for r in cohort),
        timely_completions_per_second=stats(finished)['passed']/(duration/1000),
        total_completions_per_second=len(finished)/(duration/1000),
        demand=dict(per_disk_mean_GiB_s=[x/duration for x in means],
            per_disk_overload_percent=[100*x/duration for x in over],
            all_disks_overload_percent=100*all_over/duration,
            any_disk_overload_percent=100*any_over/duration,
            per_disk_min_GiB_s=min_rates,per_disk_max_GiB_s=max_rates,
            segments=segments))

