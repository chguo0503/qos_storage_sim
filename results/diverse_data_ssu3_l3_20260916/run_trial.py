#!/usr/bin/env python3
"""Frozen diverse-input comparison; pilots stop only after warm cohort closes."""
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'results/baseline_ab128_32_ratio12_20260912'
sys.path[:0]=[str(HERE),str(ROOT),str(OLD)]
import experiment as original
from simulator.core import continuous_batch_sim as native
from policy import install_policy,make_stats,simulator_strategy,POLICIES
from metrics import live_summary,summarize,overlap


class PilotFinished(Exception):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--policy',choices=POLICIES,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    assert Path(args.label).name==args.label and args.label not in ('.','..')
    requests,meta=original.load_manifest(args.manifest)
    scenario=meta['scenario_candidate']
    assert (meta['num_npu'],meta['num_ssu'],meta['n_layers'])==(32,3,8)
    assert meta['order']=='random' and meta['source_data_sha256']==original.sha(ROOT/'data')
    target=HERE/'runs'/args.label;target.mkdir(parents=True,exist_ok=False)
    if args.smoke:
        first={}
        for q in requests:first.setdefault(q.npu_id,q)
        requests=tuple(first.values())
        meta=dict(meta,input_fingerprint=original.continuous_batch_input_fingerprint(requests))
        original.save_manifest(target/'manifest.json.gz',requests,meta)
    else:(target/'manifest.json.gz').write_bytes(args.manifest.read_bytes())
    windows=((2000.,4000.),(2000.,6000.))
    before=original.source_hashes()
    sources={p.name:original.sha(p) for p in (Path(__file__),HERE/'metrics.py',HERE/'policy.py')}
    expected=sum(len(q.placement[0])*8 for q in requests)
    stats=make_stats();observed=0;warm_saved=False
    busy=[[0.]*3 for _ in windows]
    # Actual SSD service averaged into common 10-ms bins for clear plotting.
    bins=[[0.]*200 for _ in range(3)]
    started=last_progress=time.perf_counter()
    record=dict(status='running',policy=args.policy,seed=meta['seed'],scenario=scenario,
        pilot=args.pilot,smoke=args.smoke,pid=os.getpid(),started_utc=original.utc(),argv=sys.argv,
        manifest_sha256=original.sha(target/'manifest.json.gz'),source_data_sha256=original.sha(ROOT/'data'),
        core_source_sha256=before,extension_source_sha256=sources,expected_blocks=expected,
        windows_ms=windows,collector_interval_ms=5,modeled_control_latency_ms=0)
    original.write_json(target/'command.json',record)
    callback=native._register_complete

    def preview(context):
        nonlocal warm_saved
        if args.smoke or warm_saved or context.current_time_ms<4100:return
        cohort=[q for q in context.requests.values() if q.admitted and 2000<=q.admission_time_ms<4000]
        if not cohort or not all(q.completed for q in cohort):return
        # Measurement-only guard: SSD service is accumulated on final HBM
        # completion. Do not publish its warm total while old bytes are in flight.
        for npu in context.npus:
            if npu.link_active_flow is not None and npu.link_active_flow.ssd_activation_time<4000:return
            if any(f.ssd_activation_time<4000 for f in npu.link_pending):return
        row=summarize(live_summary(context),requests,2000.,4000.)
        row.update(policy=args.policy,seed=meta['seed'],scenario=scenario,
                   SSD_GiB_s=[x*40/2000 for x in busy[0]],observed_at_ms=context.current_time_ms,
                   warm_exact_full_run_continues=not args.pilot)
        original.write_json(target/'warm_preview.json',row)
        warm_saved=True
        print(json.dumps(dict(warm=True,policy=args.policy,scenario=scenario,
            seed=meta['seed'],U=row['U_percent'],slo=row['slo'],
            overload_percent=row['demand']['per_disk_overload_percent'])),flush=True)
        if args.pilot:raise PilotFinished()

    def observe(context,flow):
        nonlocal observed,last_progress
        observed+=1
        a,z=flow.ssd_activation_time,flow.link_enqueue_time
        for i,(left,right) in enumerate(windows):busy[i][flow.disk_id]+=overlap(a,z,left,right)
        if a<4000 and z>2000:
            a1,z1=max(a,2000.),min(z,4000.)
            first,last=int((a1-2000)//10),min(199,int((z1-2000)//10))
            for j in range(first,last+1):bins[flow.disk_id][j]+=overlap(a1,z1,2000+j*10,2010+j*10)
        ret=callback(context,flow)
        if observed%10000==0:
            preview(context)
            now=time.perf_counter()
            if now-last_progress>=15:
                p=dict(completed_blocks=observed,expected_blocks=expected,
                    simulation_ms=context.current_time_ms,wall_seconds=now-started,
                    completed_requests=context.completed_requests)
                original.write_json(target/'progress.json',p);print(json.dumps(p),flush=True)
                last_progress=now
        return ret
    try:
        with patch.object(native,'_register_complete',observe),install_policy(args.policy,stats):
            result=original.run_case(requests,meta,strategy=simulator_strategy(args.policy),
                                     assignment='fixed',windows=windows)
        assert observed==expected
        assert all(abs(sum(bins[d])-busy[0][d])<1e-7 for d in range(3))
        assert all(v<=z-a+1e-7 for i,(a,z) in enumerate(windows) for v in busy[i])
        assert all(result['summary']['invariants'].values())
        assert result['adapter_statistics']['reorder_calls']==0
        assert result['adapter_statistics']['cir_write_events']==[]
        assert result['adapter_statistics']['assignment_count']==0
        assert result['input_fingerprint']==meta['input_fingerprint']
        byid={q.request_id:q for q in requests}
        rows=result['summary']['request_metrics']
        assert len(rows)==len(requests)
        for n in range(32):
            actual=[r['request_id'] for r in sorted(rows,key=lambda x:x['admission_time_ms']) if byid[r['request_id']].npu_id==n]
            assert actual==[q.request_id for q in requests if q.npu_id==n]
        analyses=[]
        if not args.smoke:
            for i,(a,z) in enumerate(windows):
                row=summarize(result['summary'],requests,a,z)
                assert row['all_npus_active']
                assert abs(row['U_percent']-100*result['windows'][i]['mean_npu_utilization'])<1e-7
                row['SSD_GiB_s']=[x*40/(z-a) for x in busy[i]]
                analyses.append(row)
            full=summarize(result['summary'],requests,0.,result['summary']['makespan_ms'],full=True)
            assert abs(full['U_percent']-100*result['summary']['fleet_npu_compute_utilization'])<1e-7
            analyses.append(full)
            if (target/'warm_preview.json').exists():
                p=original.read_json(target/'warm_preview.json')
                assert p['slo']==analyses[0]['slo'] and abs(p['U_percent']-analyses[0]['U_percent'])<1e-7
                assert p['demand']==analyses[0]['demand']
        result.update(strategy=args.policy,analysis=analyses,routing_statistics=stats,
                      warm_ssd_10ms_GiB_s=[[v*4 for v in disk] for disk in bins])
        original.write_json(target/'result.json.gz',result)
        record.update(status='complete',result_sha256=original.sha(target/'result.json.gz'),
                      completed_requests=len(rows),observed_blocks=observed,
                      checks=dict(all_invariants=True,FIFO_preserved=True,static_QoS=True,
                                  unchanged_L1_L2=True,same_input=True),
                      metrics=[dict(start_ms=r['start_ms'],end_ms=r['end_ms'],U=r['U_percent'],
                                    slo=r['slo'],overload_percent=r['demand']['per_disk_overload_percent']) for r in analyses])
    except PilotFinished:
        record.update(status='pilot_complete',completed_simulation=False,
                      observed_blocks=observed,warm_exact=True)
    except BaseException as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        core_ok=original.source_hashes()==before
        ext_ok={p.name:original.sha(p) for p in (Path(__file__),HERE/'metrics.py',HERE/'policy.py')}==sources
        record.update(ended_utc=original.utc(),wall_seconds=time.perf_counter()-started,
                      core_unchanged=core_ok,extension_unchanged=ext_ok,routing_statistics=stats)
        if not core_ok or not ext_ok:record['status']='failed_source_changed'
        original.write_json(target/'command.json',record)
        assert core_ok and ext_ok
    print(json.dumps({k:record[k] for k in ('status','policy','scenario','seed','wall_seconds')}),flush=True)


if __name__=='__main__':main()
