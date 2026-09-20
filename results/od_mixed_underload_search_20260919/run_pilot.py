#!/usr/bin/env python3
"""Original simulator parameter screening; intentionally stop after 4 s.

Only exact clipped utilization, active roles and reference demand are reported.
No SLO is reported because admitted requests have not all drained. A successful
screen must be replayed to full drainage with run_trial.py before publication.
"""
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from simulator.core import continuous_batch_sim as native
from metrics import exact_demand, live_summary, overlap
from inputs.runners.run_baseline_npu32_stress import load_manifest, run_case, save_manifest, write_json
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from inputs.runners.run_coflow_experiments import source_files


class StopPilot(Exception):
    pass


def prepare(base, meta, spec):
    by_npu = [[] for _ in range(32)]
    for q in base:
        by_npu[q.npu_id].append(q)
    out = []
    for n, original in enumerate(by_npu):
        original.sort(key=lambda r: r.request_id)
        original_roles = [q.load['role'] for q in original]
        prefix = 0
        while prefix < len(original_roles) and original_roles[prefix] == 'B':
            prefix += 1
        pools = {r: [q for q in original if q.load['role'] == r] for r in ('A', 'B')}
        if 'a_run' in spec:
            cycle = ['A']*spec['a_run']+['B']*spec.get('b_run', 3)
            roles = ['B']*min(prefix, len(pools['B']))
            remaining = dict(A=len(pools['A']), B=len(pools['B'])-len(roles))
            while sum(remaining.values()):
                for role in cycle:
                    if remaining[role]:
                        roles.append(role)
                        remaining[role] -= 1
        else:
            roles = original_roles
        cursor = dict(A=0, B=0)
        for position, role in enumerate(roles):
            q = pools[role][cursor[role]]
            cursor[role] += 1
            load = dict(q.load)
            compute_us = spec.get('C_'+role+'_ms', load['per_layer_us']/1000)*1000
            load.update(request_id=n*1000000+position, generation=position,
                        per_layer_us=compute_us,
                        required_bw_input_gbps=load['per_layer_kv_gb']*1e6/compute_us,
                        constructed_profile=True,
                        profile_construction=dict(method='original_simulator_parameter_search', parameters=spec))
            out.append(native.ContinuousBatchRequest.from_normalized(n*1000000+position, n, 0., load, q.placement))
    out = tuple(out)
    meta = dict(meta, label=spec['label'], case_id=spec['label'], candidate_spec=spec,
                constructed_profile=True, order='parameter_search_static_queue',
                input_fingerprint=native.continuous_batch_input_fingerprint(out),
                logical_input_fingerprint=logical_input_fingerprint(out))
    return out, meta


def measure(context, requests):
    summary = live_summary(context)
    byid = {q.request_id: q for q in requests}
    cards = [dict(A=0., B=0.) for _ in range(32)]
    active = [0.]*32
    for r in summary['request_metrics']:
        active[r['npu_id']] += overlap(r['admission_time_ms'], r['completion_time_ms'], 2000., 4000.)
    for b in summary['microbatch_metrics']:
        role = byid[b['member_request_ids'][0]].load['role']
        for layer in b['layer_metrics']:
            cards[b['npu_id']][role] += overlap(layer['compute_start_ms'], layer['compute_end_ms'], 2000., 4000.)
    warm = exact_demand(summary['request_metrics'], byid, 2000., 4000.)
    all_demand = exact_demand(summary['request_metrics'], byid, 0., 4000.)
    return dict(U_percent=100*sum(sum(c.values()) for c in cards)/64000,
                per_npu_role_compute_ms=cards,
                mixed_cards=sum(min(c.values()) > 1e-7 for c in cards),
                all_npus_active=all(abs(v-2000)<1e-7 for v in active),
                demand=warm, initial_4s_demand=all_demand,
                scientific_status='pilot_terminated_after_4s_not_drained_SLO_not_computed',
                observed_at_ms=context.current_time_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--freeze-only', action='store_true')
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    assert Path(spec['label']).name == spec['label']
    base, meta = load_manifest(args.base)
    requests, meta = prepare(base, meta, spec)
    if args.freeze_only:
        destination = HERE/'inputs'/f"{spec['label']}.json.gz"
        save_manifest(destination, requests, meta)
        print(destination)
        return
    target = HERE/'pilots'/spec['label']
    target.mkdir(parents=True, exist_ok=False)
    sources = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files()}
    started = last = time.perf_counter()
    record = dict(status='running', pilot=True, spec=spec, base_manifest=str(args.base),
                  pid=os.getpid(), base_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
                  input_fingerprint=meta['input_fingerprint'], core_sha256=sources)
    write_json(target/'command.json', record)
    callback = native._register_complete
    observations = 0
    measured = None
    def observe(context, flow):
        nonlocal observations, last, measured
        ret = callback(context, flow)
        observations += 1
        if context.current_time_ms > 4000.000001:
            measured = measure(context, requests)
            raise StopPilot()
        if observations % 20000 == 0 and time.perf_counter()-last > 20:
            write_json(target/'progress.json',dict(simulation_ms=context.current_time_ms,
                       completed_blocks=observations,wall_seconds=time.perf_counter()-started))
            last = time.perf_counter()
        return ret
    try:
        with patch.object(native, '_register_complete', observe):
            run_case(requests, meta, strategy='od_baseline', assignment='fixed', windows=((2000.,4000.),))
        raise AssertionError('Population drained before pilot stop')
    except StopPilot:
        record.update(status='complete_pilot', completed_simulation=False,
                      measurement=measured, wall_seconds=time.perf_counter()-started,
                      completed_blocks=observations)
    finally:
        after = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files()}
        assert after == sources
        record['source_unchanged'] = True
        write_json(target/'command.json', record)
    print(json.dumps(dict(label=spec['label'],U=measured['U_percent'],
                         under=measured['demand']['strict_underload_all_disks'],
                         all4_under=measured['initial_4s_demand']['strict_underload_all_disks'],
                         mixed=measured['mixed_cards'],wall_seconds=record['wall_seconds'])),flush=True)


if __name__ == '__main__':
    main()
