#!/usr/bin/env python3
"""Replay a completed seed-7 ASU/OD case with a read-only SSD observer.

No original source, manifest, or result is changed. Frozen original layer
boundaries are used only by this measurement observer, never by the policy.
Complete drainage verifies the original request/layer timing and outcome.
"""
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from simulator.core import continuous_batch_sim as native
from inputs.runners.run_baseline_npu32_stress import load_manifest, read_json, write_json, run_case
from inputs.runners.run_coflow_experiments import source_files

LEFT, RIGHT, DISK_BW = 2000.0, 4000.0, 40.0
TIME_TOLERANCE_MS = 1e-7
VOLUME_TOLERANCE_GIB = 1e-7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def overlap(a, z, left, right):
    return max(0.0, min(z, right) - max(a, left))


def close(a, b, tolerance=VOLUME_TOLERANCE_GIB):
    assert abs(a-b) <= tolerance, (a, b, tolerance)


def timing_view(summary):
    """Exclude wall-clock instrumentation and retain actual simulation events."""
    request_fields = ('request_id', 'npu_id', 'arrival_time_ms', 'admission_time_ms',
                      'completion_time_ms', 'own_compute_ms', 'io_stall_ms',
                      'layer0_io_start_time_ms', 'layer0_cross_request_prefetched')
    requests = [{k:r[k] for k in request_fields} for r in summary['request_metrics']]
    requests.sort(key=lambda r:r['request_id'])
    batches = [dict(npu_id=b['npu_id'], member_request_ids=b['member_request_ids'],
                    admission_time_ms=b['admission_time_ms'], completion_time_ms=b['completion_time_ms'],
                    layer_metrics=b['layer_metrics']) for b in summary['microbatch_metrics']]
    batches.sort(key=lambda b:(b['npu_id'], b['admission_time_ms']))
    return dict(requests=requests, batches=batches, makespan_ms=summary['makespan_ms'])


def compare_timing(expected, actual):
    """Permit tiny cross-Python float drift, recording its maximum explicitly."""
    maximum = 0.0; differences = 0
    def walk(a, b, name):
        nonlocal maximum, differences
        if isinstance(a, dict):
            assert set(a) == set(b), name
            for key in a:walk(a[key], b[key], f'{name}.{key}')
        elif isinstance(a, list):
            assert len(a) == len(b), name
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{name}[{i}]')
        elif isinstance(a, float):
            error = abs(a-b)
            assert math.isfinite(error) and error <= TIME_TOLERANCE_MS, (name,a,b,error)
            maximum = max(maximum,error);differences += int(a != b)
        else:
            assert a == b, (name,a,b)
    walk(expected, actual, 'timing')
    return dict(max_absolute_numeric_error_ms=maximum, nonidentical_float_count=differences,
                allowed_error_ms=TIME_TOLERANCE_MS, exact_float_match=differences == 0)


def prepare_cycles(reference, requests):
    byid = {q.request_id:q for q in requests}
    volume = {q.request_id:[math.fsum(v for disk,v in q.placement[0] if disk == d)
                           for d in range(3)] for q in requests}
    demand = {rid:[v*1e6/byid[rid].load['per_layer_us'] for v in values]
              for rid,values in volume.items()}
    by_npu = [[] for _ in range(32)]
    for b in reference['summary']['microbatch_metrics']:
        assert len(b['member_request_ids']) == 1
        rid = b['member_request_ids'][0]
        for layer in b['layer_metrics']:
            by_npu[b['npu_id']].append((layer['compute_start_ms'], rid, layer))
    admissions = [[] for _ in range(32)]
    for r in reference['summary']['request_metrics']:
        admissions[r['npu_id']].append((r['admission_time_ms'],r['completion_time_ms'],r['request_id']))
    cycles = []
    for npu, states in enumerate(by_npu):
        states.sort(); lane = []
        for index, (current, following) in enumerate(zip(states,states[1:])):
            start,rid,layer = current; end,next_rid,next_layer = following
            if end <= LEFT or start >= RIGHT:continue
            assert end > start and layer['compute_end_ms'] <= end + TIME_TOLERANCE_MS
            clipped_start,clipped_end = max(start,LEFT),min(end,RIGHT)
            full_demand = [0.0]*3; clipped_demand = [0.0]*3
            for a,z,active_rid in admissions[npu]:
                full_dt = overlap(a,z,start,end)/1000
                clip_dt = overlap(a,z,clipped_start,clipped_end)/1000
                for d in range(3):
                    full_demand[d] += demand[active_rid][d]*full_dt
                    clipped_demand[d] += demand[active_rid][d]*clip_dt
            lane.append(dict(npu_id=npu, cycle_index_in_npu=index, request_id=rid,
                layer=layer['layer'], next_request_id=next_rid, next_layer=next_layer['layer'],
                same_request=rid == next_rid, boundary_clipped=start < LEFT or end > RIGHT,
                start_ms=start, end_ms=end, clipped_start_ms=clipped_start, clipped_end_ms=clipped_end,
                compute_end_ms=layer['compute_end_ms'],
                compute_ms=layer['compute_end_ms']-start,
                stall_ms=max(0.0,end-layer['compute_end_ms']),
                expected_next_layer_GiB_by_ssu=volume[next_rid],
                full_demand_integral_GiB_by_ssu=full_demand,
                clipped_demand_integral_GiB_by_ssu=clipped_demand,
                full_service_GiB_by_ssu=[0.0]*3,
                clipped_service_GiB_by_ssu=[0.0]*3))
        assert lane, npu
        close(lane[0]['clipped_start_ms'],LEFT,TIME_TOLERANCE_MS)
        close(lane[-1]['clipped_end_ms'],RIGHT,TIME_TOLERANCE_MS)
        for a,b in zip(lane,lane[1:]):close(a['end_ms'],b['start_ms'],TIME_TOLERANCE_MS)
        close(math.fsum(c['clipped_end_ms']-c['clipped_start_ms'] for c in lane),RIGHT-LEFT,TIME_TOLERANCE_MS)
        cycles.append(lane)
    return cycles


def integrate_service(lane, starts, disk, a, z):
    """Integrate physical SSD service, including fragments crossing boundaries."""
    if z <= lane[0]['start_ms'] or a >= lane[-1]['end_ms']:return
    index = max(0,bisect_right(starts,a)-1)
    while index < len(lane) and lane[index]['start_ms'] < z:
        c = lane[index]
        c['full_service_GiB_by_ssu'][disk] += DISK_BW*overlap(a,z,c['start_ms'],c['end_ms'])/1000
        c['clipped_service_GiB_by_ssu'][disk] += DISK_BW*overlap(a,z,c['clipped_start_ms'],c['clipped_end_ms'])/1000
        index += 1


def fleet_curve(cycles):
    changes = defaultdict(lambda:{'start':[], 'end':[]})
    for lane in cycles:
        for c in lane:
            seconds = (c['clipped_end_ms']-c['clipped_start_ms'])/1000
            full_seconds = (c['end_ms']-c['start_ms'])/1000
            for scope,duration in (('full',full_seconds),('clipped',seconds)):
                c[f'{scope}_mean_demand_GiB_s_by_ssu'] = [v/duration for v in c[f'{scope}_demand_integral_GiB_by_ssu']]
                c[f'{scope}_mean_supply_GiB_s_by_ssu'] = [v/duration for v in c[f'{scope}_service_GiB_by_ssu']]
            rates = c['clipped_mean_demand_GiB_s_by_ssu'] + c['clipped_mean_supply_GiB_s_by_ssu']
            changes[c['clipped_start_ms']]['start'].append((c['npu_id'],rates))
            changes[c['clipped_end_ms']]['end'].append(c['npu_id'])
    times = sorted(changes); active = {}; segments = []
    for a,z in zip(times,times[1:]):
        for npu in changes[a]['end']:del active[npu]
        for npu,rates in changes[a]['start']:
            assert npu not in active
            active[npu] = rates
        assert len(active) == 32
        segments.append([a,z,*[math.fsum(active[n][d] for n in sorted(active)) for d in range(6)]])
    return segments


def warm_statistics(summary):
    cohort = [r for r in summary['request_metrics'] if LEFT <= r['admission_time_ms'] < RIGHT]
    passed = sum(r['completion_time_ms']-r['admission_time_ms'] <= 1.5*r['own_compute_ms']+1e-9 for r in cohort)
    compute = math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],LEFT,RIGHT)
                        for b in summary['microbatch_metrics'] for l in b['layer_metrics'])
    return dict(U_percent=100*compute/(32*(RIGHT-LEFT)),
                slo=dict(count=len(cohort),passed=passed,percent=100*passed/len(cohort)),
                cohort_request_ids=sorted(r['request_id'] for r in cohort))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-case',type=Path,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--prepare-only','--inspect-only',action='store_true',help='Validate inputs/cycle coverage without running or writing outputs')
    args = ap.parse_args()
    assert Path(args.label).name == args.label and args.label not in ('.','..')
    source = args.source_case.resolve()
    source.relative_to(ROOT)
    source_paths = [source/name for name in ('command.json','manifest.json.gz','result.json.gz')]
    protected = {str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    original_command,reference = read_json(source_paths[0]),read_json(source_paths[2])
    assert original_command['status'] == 'complete' and original_command['completed_simulation']
    assert not original_command.get('smoke') and not original_command.get('pilot')
    assert sha(source_paths[1]) == original_command['manifest_sha256']
    assert sha(source_paths[2]) == original_command['result_sha256']
    requests,meta = load_manifest(source_paths[1])
    policy = original_command['policy']
    assert policy in ('asu_baseline','od_baseline')
    assert meta['seed'] == 7 and meta['order'] == 'random'
    assert (meta['num_npu'],meta['num_ssu'],meta['n_layers']) == (32,3,8)
    regime = {'document':'under','semi':'semi','full':'full'}[meta['scenario_candidate']]
    for name,digest in original_command['core_source_sha256'].items():
        assert sha(ROOT/name) == digest, ('Original simulator source changed',name)
    assert sha(ROOT/'inputs/runners/run_baseline_npu32_stress.py') == reference['stress_runner_sha256']
    assert sha(ROOT/'data') == meta['source_data_sha256']
    current_source = {name:sha(ROOT/name) for name in source_files()}
    current_source['run_baseline_npu32_stress.py'] = sha(ROOT/'inputs/runners/run_baseline_npu32_stress.py')
    current_source[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    current_source['data'] = sha(ROOT/'data')
    protected.update(current_source)
    cycles = prepare_cycles(reference,requests)
    starts = [[c['start_ms'] for c in lane] for lane in cycles]
    expected = sum(len(q.placement[0])*8 for q in requests)
    summary = dict(regime=regime,policy=policy,seed=7,order='random',expected_blocks=expected,
                   cycle_count=sum(map(len,cycles)),window_ms=[LEFT,RIGHT],
                   clipped_cycle_count=sum(c['boundary_clipped'] for lane in cycles for c in lane))
    if args.prepare_only:
        print(json.dumps(summary),flush=True);return
    target = HERE/'runs'/args.label
    target.mkdir(parents=True,exist_ok=False)
    record = dict(status='running',completed_simulation=False,started_utc=utc(),pid=os.getpid(),
                  argv=sys.argv,source_case=str(source.relative_to(ROOT)),source_sha256=protected,
                  **summary)
    write_json(target/'command.json',record)
    callback = native._register_complete
    observed = 0; retained = 0; started = last_progress = time.perf_counter()
    warm_gib = [[0.0]*32 for _ in range(3)]
    warm_bins_ms = [[0.0]*200 for _ in range(3)]
    qos_checked = False

    def observe(context,flow):
        nonlocal observed,retained,last_progress,qos_checked
        observed += 1
        if not qos_checked:
            for q,original in zip(context.qos_configs_by_ssu,reference['actual_qos_by_ssu']):
                assert list(q.path_cirs) == original['path_cirs_gib_s']
                assert ['unlimited' if math.isinf(v) else v for v in q.path_pirs] == original['path_pirs_gib_s']
                assert list(q.path_weights) == original['path_weights']
                assert list(q.group_weights) == original['group_weights']
            qos_checked = True
        n,d = flow.npu_id,flow.disk_id
        a,z = flow.ssd_activation_time,flow.link_enqueue_time
        lane = cycles[n]
        if z > lane[0]['start_ms'] and a < lane[-1]['end_ms']:
            retained += 1
            close((z-a)*DISK_BW/1000,flow.total_gb,1e-9)
            integrate_service(lane,starts[n],d,a,z)
        if z > LEFT and a < RIGHT:
            warm_gib[d][n] += DISK_BW*overlap(a,z,LEFT,RIGHT)/1000
            first = max(0,int((max(a,LEFT)-LEFT)//10))
            last = min(199,int((min(z,RIGHT)-LEFT)//10))
            for j in range(first,last+1):
                warm_bins_ms[d][j] += overlap(a,z,LEFT+10*j,LEFT+10*(j+1))
        ret = callback(context,flow)
        if observed % 10000 == 0:
            now = time.perf_counter()
            if now-last_progress >= 15:
                progress = dict(completed_blocks=observed,expected_blocks=expected,
                                simulation_ms=context.current_time_ms,wall_seconds=now-started,
                                completed_requests=context.completed_requests)
                write_json(target/'progress.json',progress);print(json.dumps(progress),flush=True)
                last_progress = now
        return ret

    try:
        with patch.object(native,'_register_complete',observe):
            replay = run_case(requests,meta,strategy=policy,assignment='fixed',windows=((LEFT,RIGHT),))
        assert observed == expected == replay['summary']['completed_blocks']
        assert len(replay['summary']['request_metrics']) == len(requests)
        assert replay['input_fingerprint'] == reference['input_fingerprint'] == meta['input_fingerprint']
        assert replay['input_placement_fingerprint'] == replay['execution_placement_fingerprint'] == reference['execution_placement_fingerprint']
        assert replay['baseline_configuration'] == reference['baseline_configuration']
        assert replay['static_path_cirs_gib_s'] == reference['static_path_cirs_gib_s']
        assert replay['adapter_statistics']['routed_blocks_by_ssu_npu_path'] == reference['adapter_statistics']['routed_blocks_by_ssu_npu_path']
        assert replay['summary']['cross_request_layer0_prefetches'] == reference['summary']['cross_request_layer0_prefetches']
        assert all(replay['summary']['invariants'].values())
        for field in ('reorder_calls','assignment_count'):assert replay['adapter_statistics'][field] == 0
        assert replay['adapter_statistics']['cir_write_events'] == []
        timing = compare_timing(timing_view(reference['summary']),timing_view(replay['summary']))
        actual_warm = warm_statistics(replay['summary']);old_warm = warm_statistics(reference['summary'])
        assert actual_warm['slo'] == old_warm['slo'] and actual_warm['cohort_request_ids'] == old_warm['cohort_request_ids']
        close(actual_warm['U_percent'],old_warm['U_percent'],1e-7)
        max_cycle_volume_error = max_warm_volume_error = 0.0
        for n,lane in enumerate(cycles):
            for c in lane:
                for d in range(3):
                    error = abs(c['full_service_GiB_by_ssu'][d]-c['expected_next_layer_GiB_by_ssu'][d])
                    max_cycle_volume_error = max(max_cycle_volume_error,error)
                    assert error <= VOLUME_TOLERANCE_GIB, (n,c['request_id'],c['layer'],d,error)
            for d in range(3):
                integrated = math.fsum(c['clipped_service_GiB_by_ssu'][d] for c in lane)
                close(integrated,warm_gib[d][n])
                old_gib = reference['warm_ssd_GiB_s_by_ssu_npu'][d][n]*(RIGHT-LEFT)/1000
                max_warm_volume_error = max(max_warm_volume_error,abs(integrated-old_gib))
                close(integrated,old_gib)
        bins = [[value*DISK_BW/10 for value in disk] for disk in warm_bins_ms]
        for d in range(3):
            for actual,old in zip(bins[d],reference['warm_ssd_10ms_GiB_s'][d]):close(actual,old,1e-6)
        segments = fleet_curve(cycles)
        physical = [math.fsum(row) for row in warm_gib]
        demand_integral = [math.fsum(c['clipped_demand_integral_GiB_by_ssu'][d] for lane in cycles for c in lane) for d in range(3)]
        reference_window = next(a for a in reference['analysis'] if a['start_ms'] == LEFT and a['end_ms'] == RIGHT)
        for d in range(3):
            close(math.fsum((s[1]-s[0])*s[5+d]/1000 for s in segments),physical[d])
            close(math.fsum((s[1]-s[0])*s[2+d]/1000 for s in segments),demand_integral[d])
            close(demand_integral[d],reference_window['demand']['per_disk_mean_GiB_s'][d]*(RIGHT-LEFT)/1000)
            close(physical[d],reference_window['SSD_GiB_s'][d]*(RIGHT-LEFT)/1000)
        checks = dict(input_identical=True,placement_identical=True,all_requests_and_blocks_complete=True,
                      original_timing_matches=True,warm_U_and_SLO_match=True,actual_QoS_identical=True,
                      FIFO_and_NPU_binding_preserved=True,full_cycles_equal_next_layer_volume=True,
                      clipped_cycles_match_original_per_npu_disk_bytes=True,original_10ms_service_bins_match=True,
                      demand_and_service_warm_areas_preserved=True)
        payload = dict(schema_version=1,**summary,source_case=str(source.relative_to(ROOT)),
            source_sha256=protected,per_npu_cycles=cycles,fleet_segments=segments,
            fleet_segment_columns=['start_ms','end_ms','demand_ssu0_GiB_s','demand_ssu1_GiB_s','demand_ssu2_GiB_s',
                                   'supply_ssu0_GiB_s','supply_ssu1_GiB_s','supply_ssu2_GiB_s'],
            warm_actual_service_GiB_by_ssu=physical,warm_actual_service_GiB_by_ssu_npu=warm_gib,
            warm_reference_demand_integral_GiB_by_ssu=demand_integral,warm_ssd_10ms_GiB_s=bins,
            window_statistics=actual_warm,timing_comparison=timing,checks=checks,
            max_full_cycle_volume_error_GiB=max_cycle_volume_error,
            max_warm_per_npu_disk_volume_error_GiB=max_warm_volume_error,
            definitions=dict(cycle='compute_start(k) through next compute_start, including exposed wait; may cross requests',
                supply='actual SSD service during [ssd_activation_time,link_enqueue_time), includes every physical read for that NPU',
                demand='integral of active admitted request per-disk V/C; changes at admission even inside a boundary cycle',
                clipping='first/last window fragments use their own exact demand/service integrals divided by clipped duration',
                full_volume_check='each complete cycle carries following request/layer data; cross-request cycles carry next-request L0',
                caution='sum of asynchronous per-card cycle averages is not instantaneous throughput and may exceed physical capacity',
                observer='old boundaries used only for measurement; full replay checks actual layer/request events afterward'))
        write_json(target/'services.json.gz',payload)
        record.update(status='complete',completed_simulation=True,checks=checks,timing_comparison=timing,
                      completed_requests=len(requests),observed_blocks=observed,measured_cycle_blocks=retained,
                      services_sha256=sha(target/'services.json.gz'),window_statistics=actual_warm,
                      max_full_cycle_volume_error_GiB=max_cycle_volume_error,
                      max_warm_per_npu_disk_volume_error_GiB=max_warm_volume_error)
    except BaseException as exc:
        record.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        unchanged = all(sha(ROOT/name) == digest for name,digest in protected.items())
        record.update(ended_utc=utc(),wall_seconds=time.perf_counter()-started,all_protected_files_unchanged=unchanged)
        if not unchanged:record['status'] = 'failed_source_changed'
        write_json(target/'command.json',record)
        assert unchanged
    print(json.dumps({k:record[k] for k in ('status','regime','policy','observed_blocks','wall_seconds')}),flush=True)


if __name__ == '__main__':main()
