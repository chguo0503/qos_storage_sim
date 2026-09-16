#!/usr/bin/env python3
"""Static-packing ablation, snapshot of run_trial.py with fixed initial budgets.

Only changes versus the dynamic driver: policy labels, initial budget instead
of elapsed/progress budget, and metadata describing this ablation. The original
FIFO, Once congestion routing, 5ms sampling, all statistics, and source/input
checks are unchanged. Unknown pre-admission L0 retains exact Once. For the
current A/B profiles, all admitted A use full pools and B use fixed prefixes.
"""
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from unittest.mock import patch
import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(STUDY)]
import experiment as original
import continuous_batch_sim as native
import shared_path_sim_adapter as shared
import shared_path_once
from slo_router import RequestBudget, RouterConfig, slo_path_ids


def upper_budget(context, state, now_ms):
    """Fixed initial request budget; independent of elapsed time and progress."""
    req = context.requests[state.request_id]
    if not req.admitted:
        # Cross-request prefetch has no admission-clock deadline yet.
        return None
    assert req.batch_size == 1
    c = req.per_layer_compute_ms
    # Freeze the *initial* budget: no elapsed time, current compute state,
    # layer progress, or already-consumed waiting budget changes the pool.
    initial_compute_ms = context.n_layers*c
    layout = req.manifest.placement[0 if len(req.manifest.placement)==1 else state.layer]
    work = [0.] * context.num_ssu
    for s, size in layout:
        work[s] += size
    return RequestBudget(remaining_slo_ms=1.5*initial_compute_ms,
                         remaining_compute_ms=initial_compute_ms,
                         remaining_layers=context.n_layers,
                         layer_work_gib_by_ssu=tuple(work))


@contextmanager
def routing_adapter(policy, routing_stats, **kwargs):
    original_manager = routing_adapter.original_manager
    assert policy in ('static_mild', 'static_aggressive')
    config = RouterConfig(slack_budget_threshold_ms=6.0,
                          slack_paths_per_group=2 if policy=='static_mild' else 1)
    budget = None
    def route(count, snapshot, allowed_path_ids, qos, start_offset=0,
              disk_bw_gib_s=40.):
        ids = slo_path_ids(count, snapshot, allowed_path_ids, qos,
                           budget=budget, config=config,
                           start_offset=start_offset, disk_bw_gib_s=disk_bw_gib_s)
        assert len(ids)==count and set(ids).issubset(allowed_path_ids)
        routing_stats['unique_paths_sum'] += len(set(ids))
        return ids
    with patch.object(shared_path_once, 'once_path_ids', route):
        with original_manager(**kwargs) as adapter:
            old_plan = native._plan_paths
            def plan(context, state, now_ms):
                nonlocal budget
                budget = upper_budget(context, state, now_ms)
                routing_stats['calls'] += 1
                if budget is None:
                    group = 'unadmitted_exact_once'
                    slack = None
                else:
                    slack = (budget.remaining_slo_ms-budget.remaining_compute_ms)/max(1,budget.remaining_layers)
                    group = 'slack' if slack > config.slack_budget_threshold_ms else 'urgent_or_expired'
                routing_stats[group] += 1
                if len(routing_stats['examples'])<32:
                    routing_stats['examples'].append(dict(time_ms=now_ms,
                        request_id=state.request_id, npu_id=state.npu_id,
                        layer=state.layer, ssu_id=state.disk_id,
                        remaining_wait_per_read_layer_ms=slack, decision_group=group))
                return old_plan(context, state, now_ms)
            with patch.object(native, '_plan_paths', plan):
                yield adapter


def cohort_stats(rows, manifests):
    def summarize(part):
        passed = sum(r['completion_time_ms']-r['admission_time_ms'] <=
                     1.5*8*manifests[r['request_id']].load['per_layer_us']/1000 + 1e-9 for r in part)
        return dict(count=len(part), passed=passed,
                    percent=100*passed/len(part) if part else None)
    return dict(overall=summarize(rows), by_role={role:summarize([
        r for r in rows if manifests[r['request_id']].load['role']==role]) for role in ('A','B')})


def warm_statistics(context, byid):
    """Read-only early statistic; live microbatches are a list, requests a dict."""
    cohort = [q for q in context.requests.values() if q.admitted and 2000.<=q.admission_time_ms<4000.]
    if not cohort or not all(q.completed for q in cohort):
        return None
    compute = math.fsum(max(0.,min(m.compute_start_ms+m.compute_duration_ms,4000.)-
                max(m.compute_start_ms,2000.)) for b in context.microbatches
                for m in b.layer_metrics if math.isfinite(m.compute_start_ms))
    rows = [dict(request_id=q.manifest.request_id, admission_time_ms=q.admission_time_ms,
                 completion_time_ms=q.completion_time_ms) for q in cohort]
    return dict(U_percent=100*compute/64000., slo=cohort_stats(rows,byid),
                request_ids=sorted(q['request_id'] for q in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', choices=('static_mild','static_aggressive'), required=True)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--label')
    args = ap.parse_args()
    label = args.label or f'{args.policy}_seed{args.seed}' + ('_smoke' if args.smoke else '')
    assert Path(label).name == label and label not in ('.','..')
    case = STUDY/'validation20s/runs'/f'ssu3_random_k1_sync_seed{args.seed}'/'baseline'
    frozen = case/'manifest.json.gz'
    ref = original.read_json(case/'command.json')
    before = original.source_hashes()
    assert before==ref['core_source_sha256']
    assert original.sha(frozen)==ref['manifest_sha256']
    requests, metadata = original.load_manifest(frozen)
    assert metadata['num_ssu']==3 and metadata['num_npu']==32 and metadata['n_layers']==8
    assert metadata['seed']==args.seed and metadata['order']=='random'
    assert all(q.arrival_time_ms==0 for q in requests)
    target = HERE/'runs'/label
    target.mkdir(parents=True, exist_ok=False)
    manifest = target/'manifest.json.gz'
    if args.smoke:
        requests = [q for q in requests if q.load['generation'] < 3]
        metadata = dict(metadata, input_fingerprint=original.continuous_batch_input_fingerprint(requests))
        original.save_manifest(manifest, requests, metadata)
    else:
        manifest.write_bytes(frozen.read_bytes())
    byid = {q.request_id:q for q in requests}
    windows = ((0.,50.),) if args.smoke else ((2000.,4000.),(2000.,20000.))
    extension_hashes = {p.name:original.sha(p) for p in (Path(__file__), HERE/'slo_router.py')}
    stats = dict(calls=0, unique_paths_sum=0, slack=0, urgent_or_expired=0,
                 unadmitted_exact_once=0, examples=[])
    record = dict(status='running', strategy=args.policy, seed=args.seed,
        assignment='fixed', order='random', smoke=args.smoke, pid=os.getpid(),
        argv=sys.argv, started_utc=original.utc(), windows=windows,
        manifest_sha256=original.sha(manifest), baseline_manifest_sha256=original.sha(frozen),
        core_source_sha256=before, extension_source_sha256=extension_hashes,
        physical_policy='unchanged per-path FIFO; static CIR; unlimited PIR',
        information='fixed initial own-request compute/SLO profile + original 5ms count-only disk snapshot; no elapsed/progress urgency',
        ablation='initial budget frozen for every admitted layer; same unadmitted L0 exact Once fallback',
        slo_clock='admission; unadmitted cross-request prefetch uses exact Once',
        parameters=dict(slack_budget_threshold_ms=6.,slack_paths_per_group=2 if args.policy=='static_mild' else 1),
        control_latency_ms=0, matched_to_reference_control_latency=True)
    original.write_json(target/'command.json', record)
    started = last_progress = time.perf_counter()
    expected = sum(sum(len(layer) for layer in q.placement)*(8 if len(q.placement)==1 else 1) for q in requests)
    observed = 0
    ssd_busy_warm = [0.,0.,0.]
    warm_saved = False
    callback = native._register_complete

    def warm_preview(context):
        nonlocal warm_saved
        if args.smoke or warm_saved or context.current_time_ms < 4500.:
            return
        measured = warm_statistics(context, byid)
        if measured is None:
            return
        preview = dict(status='exact_warm_preview_full_run_continues', policy=args.policy,
            seed=args.seed, window_ms=[2000.,4000.], **measured,
            observed_at_ms=context.current_time_ms, wall_seconds=time.perf_counter()-started,
            SSD_GiB_s=sum(ssd_busy_warm)*40/2000.)
        original.write_json(target/'warm_preview.json', preview)
        print(json.dumps({k:v for k,v in preview.items() if k!='request_ids'}),flush=True)
        warm_saved = True

    def observe(context, flow):
        nonlocal observed,last_progress
        observed += 1
        ssd_busy_warm[flow.disk_id] += max(0.,min(flow.link_enqueue_time,4000.)-max(flow.ssd_activation_time,2000.))
        ret = callback(context,flow)
        if observed % 10000 == 0:
            warm_preview(context)
            if time.perf_counter()-last_progress>=15:
                p = dict(policy=args.policy, seed=args.seed, simulation_ms=context.current_time_ms,
                    completed_requests=context.completed_requests, completed_blocks=observed,
                    expected_blocks=expected, wall_seconds=time.perf_counter()-started)
                original.write_json(target/'progress.json',p)
                print(json.dumps(p),flush=True)
                last_progress=time.perf_counter()
        return ret

    routing_adapter.original_manager = shared.shared_path_adapter
    def manager(**kwargs):
        return routing_adapter(args.policy,stats,**kwargs)
    try:
        with patch.object(native,'_register_complete',observe), patch.object(shared,'shared_path_adapter',manager):
            result = original.run_case(requests,metadata,
                strategy='once',
                assignment='fixed',windows=windows)
        assert observed==expected
        assert all(result['summary']['invariants'].values())
        assert original.source_hashes()==before
        assert {p.name:original.sha(p) for p in (Path(__file__),HERE/'slo_router.py')}==extension_hashes
        assert result['adapter_statistics']['reorder_calls']==0
        assert result['adapter_statistics']['cir_write_events']==[]
        assert result['adapter_statistics']['assignment_count']==0
        assert result['input_fingerprint']==metadata['input_fingerprint']
        # Freeze actual per-NPU request order, not just assignment.
        observed_order = {}
        for r in sorted(result['summary']['request_metrics'],key=lambda q:q['admission_time_ms']):
            observed_order.setdefault(byid[r['request_id']].npu_id,[]).append(r['request_id'])
        assert all(ids==[q.request_id for q in requests if q.npu_id==n] for n,ids in observed_order.items())
        result['strategy']=args.policy
        result['routing_extension']=dict(source_sha256=extension_hashes,stats=stats)
        result['slo_all_windows']=[]
        for a,z in windows:
            rows=[q for q in result['summary']['request_metrics'] if a<=q['admission_time_ms']<z]
            result['slo_all_windows'].append(dict(start_ms=a,end_ms=z,
                **cohort_stats(rows,byid),request_ids=sorted(q['request_id'] for q in rows)))
        result['full_population_slo']=cohort_stats(result['summary']['request_metrics'],byid)
        result['warm_SSD_GiB_s']=sum(ssd_busy_warm)*40/2000.
        original.write_json(target/'result.json.gz',result)
        record.update(status='complete', completed_simulation=True,
            observed_completed_blocks=observed,expected_completed_blocks=expected,
            output_sha256=original.sha(target/'result.json.gz'),routing_statistics=stats,
            checks=dict(core_unchanged=True,extension_unchanged=True,all_invariants=True,
                fifo_preserved=True,static_qos_preserved=True,L1_L2_unchanged=True),
            metrics=[dict(start_ms=w['start_ms'],end_ms=w['end_ms'],
                U_percent=100*w['mean_npu_utilization'], all_active=w['all_npus_active_whole_window'],
                slo=result['slo_all_windows'][i]['overall'],
                by_role=result['slo_all_windows'][i]['by_role']) for i,w in enumerate(result['windows'])])
        if (target/'warm_preview.json').exists():
            p=original.read_json(target/'warm_preview.json');w=record['metrics'][0]
            assert abs(p['U_percent']-w['U_percent'])<1e-7 and p['slo']['overall']==w['slo']
    except BaseException as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(ended_utc=original.utc(),wall_seconds=time.perf_counter()-started)
        original.write_json(target/'command.json',record)
    print(json.dumps({k:record[k] for k in ('status','strategy','seed','metrics','wall_seconds')}),flush=True)


if __name__=='__main__':
    main()
