#!/usr/bin/env python3
"""Run original Baseline/Once and observe immutable block completion timestamps.

All flow service is integrated over its true SSD/HBM interval; 2ms binning is
measurement only. Original simulation, path policies, and QoS are not changed.
"""
from pathlib import Path
from unittest.mock import patch
import argparse, hashlib, json, math, os, sys, time, traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ORIGINAL=ROOT/'results/diverse_data_ssu3_l3_20260916'
OLD=ROOT/'results/baseline_ab128_32_ratio12_20260912'
sys.path[:0]=[str(ORIGINAL),str(ROOT),str(OLD)]
import experiment as original
import continuous_batch_sim as native
from metrics import live_summary,summarize,overlap

WINDOWS=((2000.,4000.),(2000.,6000.))
LEFT,RIGHT,STEP=2000.,4000.,2.
NBINS=int((RIGHT-LEFT)/STEP)

def add_interval(row,a,z):
    if a>=RIGHT or z<=LEFT:return
    a,z=max(a,LEFT),min(z,RIGHT)
    first=max(0,int((a-LEFT)//STEP));last=min(NBINS-1,int((z-LEFT)//STEP))
    for j in range(first,last+1):
        row[j]+=overlap(a,z,LEFT+j*STEP,LEFT+(j+1)*STEP)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed',type=int,choices=(7,19,43),required=True)
    ap.add_argument('--strategy',choices=('baseline','once'),required=True)
    args=ap.parse_args()
    manifest=HERE/'inputs'/f'near35_seed{args.seed}.json.gz'
    requests,meta=original.load_manifest(manifest)
    assert (meta['num_npu'],meta['num_ssu'],meta['n_layers'])==(32,3,8)
    assert meta['scenario_candidate']=='near35'
    assert meta['source_data_sha256']==original.sha(ROOT/'data')
    target=HERE/'runs'/f'{args.strategy}_seed{args.seed}'
    target.mkdir(parents=True,exist_ok=False)
    (target/'manifest.json.gz').write_bytes(manifest.read_bytes())
    before=original.source_hashes()
    ext={str(p.relative_to(ROOT)):original.sha(p) for p in (Path(__file__),HERE/'construct_near35.py',ORIGINAL/'metrics.py',ORIGINAL/'construct_manifest.py')}
    expected=sum(len(q.placement[0])*8 for q in requests)
    observed=0;warm_saved=False;last_progress=started=time.perf_counter()
    ssd_busy=[[0.]*3 for _ in WINDOWS]
    hbm_busy=[[0.]*32 for _ in WINDOWS]
    ssd_bins=[[0.]*NBINS for _ in range(3)]
    npu_ssd_bins=[[0.]*NBINS for _ in range(32)]
    hbm_bins=[[0.]*NBINS for _ in range(32)]
    max_ssd_error=max_hbm_error=0.
    record=dict(status='running',strategy=args.strategy,seed=args.seed,scenario='near35',
        pid=os.getpid(),started_utc=original.utc(),argv=sys.argv,
        manifest_sha256=original.sha(manifest),source_data_sha256=original.sha(ROOT/'data'),
        core_source_sha256=before,extension_source_sha256=ext,expected_blocks=expected,
        windows_ms=WINDOWS,bin_ms=STEP,collector_interval_ms=5,modeled_control_latency_ms=0,
        placement='stripe_npu_mod_ssu',candidate_pool='original complete category-legal pool',
        actual_ssd_definition='integral of 40GiB/s on [ssd_activation_time,link_enqueue_time]',
        actual_hbm_definition='integral of 50GiB/s on [link_start_time,link_end_time]')
    original.write_json(target/'command.json',record)
    callback=native._register_complete

    def preview(context):
        nonlocal warm_saved
        if warm_saved or context.current_time_ms<4100:return
        cohort=[q for q in context.requests.values() if q.admitted and 2000<=q.admission_time_ms<4000]
        if not cohort or not all(q.completed for q in cohort):return
        for npu in context.npus:
            if npu.link_active_flow is not None and npu.link_active_flow.ssd_activation_time<4000:return
            if any(f.ssd_activation_time<4000 for f in npu.link_pending):return
        row=summarize(live_summary(context),requests,2000.,4000.)
        row.update(strategy=args.strategy,seed=args.seed,scenario='near35',
            SSD_GiB_s=[v*40/2000 for v in ssd_busy[0]],
            observed_at_ms=context.current_time_ms,warm_exact_full_run_continues=True)
        original.write_json(target/'warm_preview.json',row)
        warm_saved=True
        print(json.dumps(dict(warm=True,strategy=args.strategy,seed=args.seed,U=row['U_percent'],
            slo=row['slo'],mean_demand_GiB_s=row['demand']['per_disk_mean_GiB_s'],
            overload_percent=row['demand']['per_disk_overload_percent'])),flush=True)

    def observe(context,flow):
        nonlocal observed,last_progress,max_ssd_error,max_hbm_error
        observed+=1
        a,z=flow.ssd_activation_time,flow.link_enqueue_time
        h,j=flow.link_start_time,flow.link_end_time
        max_ssd_error=max(max_ssd_error,abs((z-a)*40/1000-flow.total_gb))
        max_hbm_error=max(max_hbm_error,abs((j-h)*50/1000-flow.total_gb))
        for i,(left,right) in enumerate(WINDOWS):
            ssd_busy[i][flow.disk_id]+=overlap(a,z,left,right)
            hbm_busy[i][flow.npu_id]+=overlap(h,j,left,right)
        add_interval(ssd_bins[flow.disk_id],a,z)
        add_interval(npu_ssd_bins[flow.npu_id],a,z)
        add_interval(hbm_bins[flow.npu_id],h,j)
        ret=callback(context,flow)
        if observed%10000==0:
            preview(context)
            now=time.perf_counter()
            if now-last_progress>=15:
                progress=dict(completed_blocks=observed,expected_blocks=expected,
                    simulation_ms=context.current_time_ms,wall_seconds=now-started,
                    completed_requests=context.completed_requests)
                original.write_json(target/'progress.json',progress)
                print(json.dumps(progress),flush=True);last_progress=now
        return ret
    try:
        with patch.object(native,'_register_complete',observe):
            result=original.run_case(requests,meta,strategy=args.strategy,assignment='fixed',windows=WINDOWS)
        assert observed==expected
        assert max_ssd_error<1e-9 and max_hbm_error<1e-9
        assert all(abs(sum(ssd_bins[d])-ssd_busy[0][d])<1e-6 for d in range(3))
        assert all(abs(sum(hbm_bins[n])-hbm_busy[0][n])<1e-6 for n in range(32))
        assert all(abs(sum(npu_ssd_bins[n][j] for n in range(32))-sum(ssd_bins[d][j] for d in range(3)))<1e-6 for j in range(NBINS))
        assert all(v<=STEP+1e-7 for row in ssd_bins+hbm_bins for v in row)
        assert all(result['summary']['invariants'].values())
        adapter=result['adapter_statistics']
        assert adapter['reorder_calls']==0 and adapter['cir_write_events']==[] and adapter['assignment_count']==0
        assert adapter['collector_interval_ms']==5
        assert result['input_fingerprint']==meta['input_fingerprint']
        rows=result['summary']['request_metrics'];byid={q.request_id:q for q in requests}
        assert len(rows)==len(requests)
        for n in range(32):
            actual=[r['request_id'] for r in sorted(rows,key=lambda x:x['admission_time_ms']) if byid[r['request_id']].npu_id==n]
            assert actual==[q.request_id for q in requests if q.npu_id==n]
        analyses=[]
        for i,(a,z) in enumerate(WINDOWS):
            row=summarize(result['summary'],requests,a,z)
            assert row['all_npus_active']
            assert max(row['demand']['per_disk_max_GiB_s'])<=max(meta['static_per_ssu_upper_bound_gib_s'])+1e-7
            assert abs(row['U_percent']-100*result['windows'][i]['mean_npu_utilization'])<1e-7
            row['SSD_GiB_s']=[v*40/(z-a) for v in ssd_busy[i]]
            row['HBM_GiB_s']=[v*50/(z-a) for v in hbm_busy[i]]
            analyses.append(row)
        full=summarize(result['summary'],requests,0.,result['summary']['makespan_ms'],full=True)
        analyses.append(full)
        if (target/'warm_preview.json').exists():
            preview_row=original.read_json(target/'warm_preview.json')
            assert preview_row['slo']==analyses[0]['slo']
            assert abs(preview_row['U_percent']-analyses[0]['U_percent'])<1e-7
            assert preview_row['demand']==analyses[0]['demand']
        result.update(strategy=args.strategy,scenario='near35',seed=args.seed,analysis=analyses,
            bandwidth_2ms=dict(start_ms=LEFT,end_ms=RIGHT,step_ms=STEP,
                ssd_GiB_s=[[v*40/STEP for v in row] for row in ssd_bins],
                npu_ssd_GiB_s=[[v*40/STEP for v in row] for row in npu_ssd_bins],
                npu_hbm_GiB_s=[[v*50/STEP for v in row] for row in hbm_bins]))
        original.write_json(target/'result.json.gz',result)
        record.update(status='complete',completed_simulation=True,
            result_sha256=original.sha(target/'result.json.gz'),completed_requests=len(rows),observed_blocks=observed,
            checks=dict(all_invariants=True,FIFO_preserved=True,static_QoS=True,unchanged_L1_L2=True,same_input=True,
                reference_demand_audited=True,all_32_active_in_both_windows=True,uncensored_window_cohorts=True,
                actual_ssd_capacity_pass=True,actual_hbm_capacity_pass=True,byte_conservation=True),
            max_ssd_flow_bytes_error_GiB=max_ssd_error,max_hbm_flow_bytes_error_GiB=max_hbm_error,
            metrics=[dict(start_ms=r['start_ms'],end_ms=r['end_ms'],U=r['U_percent'],slo=r['slo'],
                overload_percent=r['demand']['per_disk_overload_percent']) for r in analyses])
    except BaseException as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        core_ok=original.source_hashes()==before
        ext_ok={p:original.sha(ROOT/p) for p in ext}==ext
        record.update(ended_utc=original.utc(),wall_seconds=time.perf_counter()-started,
            core_unchanged=core_ok,extension_unchanged=ext_ok)
        if not core_ok or not ext_ok:record['status']='failed_source_changed'
        original.write_json(target/'command.json',record)
        assert core_ok and ext_ok
    print(json.dumps({k:record[k] for k in ('status','strategy','seed','wall_seconds','metrics')}),flush=True)

if __name__=='__main__':main()
