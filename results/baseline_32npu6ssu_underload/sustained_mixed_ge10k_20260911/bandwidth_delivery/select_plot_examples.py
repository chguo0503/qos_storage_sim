#!/usr/bin/env python3
"""Select real layer examples from existing logs. No simulator import/run."""
from pathlib import Path
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
LABEL = 'sustained16_l176_6l_s64_bridge_48s1l_ordered_seed7'
LEFT, RIGHT = 3200.0, 4000.0


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    return max(0.0, min(b, RIGHT) - max(a, LEFT))


def main():
    manifest = STUDY / 'long_validation_7200s/inputs' / (LABEL + '.json.gz')
    man = read(manifest)
    requests = {r['request_id']: r for r in man['requests']}
    sources, raws, batches = {}, {}, {}
    for policy in ('baseline', 'once'):
        command_path = STUDY / 'long_validation_7200s/runs' / LABEL / policy / 'command.json'
        command = read(command_path)
        assert command['status'] == 'complete' and command['returncode'] == 0
        result = Path(command['output'])
        assert sha(result) == command['output_sha256'] and sha(manifest) == command['manifest_sha256']
        raw = read(result)
        assert raw['input_fingerprint'] == man['input_fingerprint'] and all(raw['summary']['invariants'].values())
        raws[policy] = raw
        batches[policy] = {b['member_request_ids'][0]: b for b in raw['summary']['microbatch_metrics']}
        sources[policy] = dict(result=str(result), result_sha256=sha(result), command_sha256=sha(command_path))

    def physical_volumes(rid):
        request = requests[rid]
        placement = request.get('placement', man['placements'][request['placement_index']])
        layer = placement[0]
        return [math.fsum(gib for disk, gib in layer if disk == ssu) for ssu in range(6)]

    def case(policy, rid, layer_number):
        b = batches[policy][rid]
        layer = next(l for l in b['layer_metrics'] if l['layer'] == layer_number)
        deadline = layer['compute_start_ms'] - layer['io_barrier_wait_ms']
        release = layer['io_start_time_ms']
        budget = deadline - release
        volume = math.fsum(physical_volumes(rid))
        return dict(policy=policy, npu=b['npu_id'], request_id=rid, layer=layer_number,
                    profile=requests[rid]['load'], admission_ms=b['admission_time_ms'],
                    raw_layer=layer, release_ms=release, no_stall_deadline_ms=deadline,
                    io_ready_ms=layer['io_ready_time_ms'], read_lifetime_ms=layer['io_ready_time_ms']-release,
                    predecessor_compute_budget_ms=budget, exposed_stall_ms=layer['io_barrier_wait_ms'],
                    physical_volume_gib=volume, physical_volume_mib=volume*1024,
                    physical_volume_gib_by_ssu=physical_volumes(rid),
                    layer_budget_reference_gib_s=volume*1000/budget if budget > 0 else None,
                    current_own_request_nominal_gib_s=volume*1e6/requests[rid]['load']['per_layer_us'],
                    within_default_window=(release >= LEFT and layer['io_ready_time_ms'] <= RIGHT))

    def short_internal_candidates(policy):
        result=[]
        for rid,b in batches[policy].items():
            load=requests[rid]['load']
            if load['role']!='short' or load['nql']!=1024:
                continue
            for l in b['layer_metrics'][1:]:
                deadline=l['compute_start_ms']-l['io_barrier_wait_ms']
                if LEFT <= l['io_start_time_ms'] and deadline >= LEFT and l['compute_start_ms'] <= RIGHT:
                    result.append(case(policy,rid,l['layer']))
        return result

    rank=lambda c:(c['exposed_stall_ms'],-c['npu'],-c['request_id'],-c['layer'])
    maximum={p:max(short_internal_candidates(p),key=rank) for p in batches}
    front_bridge={p:case(p,15,7) for p in batches}
    next_long={p:case(p,16,0) for p in batches}
    for p in batches:
        assert next_long[p]['release_ms'] == front_bridge[p]['raw_layer']['compute_start_ms']
        assert next_long[p]['no_stall_deadline_ms'] == front_bridge[p]['raw_layer']['compute_end_ms']
    ref=next_long['baseline']
    simultaneous=[c for c in short_internal_candidates('baseline')
                  if c['release_ms'] < ref['io_ready_ms'] and c['io_ready_ms'] > ref['release_ms']]
    coincident=max(simultaneous,key=rank)
    selected_same_ids={name:{p:case(p,c['request_id'],c['layer']) for p in batches}
                       for name,c in [('window_maximum',maximum['baseline']),('long_L0_overlap',coincident)]}
    summaries={}
    for policy,bs in batches.items():
        compute=[0.0]*32;stall=[0.0]*32;active=[0.0]*32
        for b in bs.values():
            if b['completion_time_ms'] <= LEFT or b['admission_time_ms'] >= RIGHT:
                continue
            n=b['npu_id'];active[n]+=overlap(b['admission_time_ms'],b['completion_time_ms'])
            previous=b['admission_time_ms']
            for layer in b['layer_metrics']:
                cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
                compute[n]+=overlap(cs,ce);stall[n]+=overlap(previous,cs);previous=ce
        assert all(math.isclose(c+w,RIGHT-LEFT,abs_tol=1e-7) for c,w in zip(compute,stall))
        assert all(math.isclose(a,RIGHT-LEFT,abs_tol=1e-7) for a in active)
        summaries[policy]=dict(compute_ms_by_npu=compute,stall_ms_by_npu=stall,active_ms_by_npu=active,
                               mean_device_utilization=math.fsum(compute)/(32*(RIGHT-LEFT)))
    output=dict(no_simulation_run=True,service_trace_not_yet_observed=True,
        manifest=str(manifest),manifest_sha256=sha(manifest),input_fingerprint=man['input_fingerprint'],
        script_sha256=sha(__file__),sources=sources,window_ms=[LEFT,RIGHT],window_summary=summaries,
        largest_short_internal_stall=maximum,front_bridge_last_layer=front_bridge,next_long_L0=next_long,
        baseline_short_stall_overlapping_long_L0=coincident,selected_layers_same_identity_in_both_policies=selected_same_ids,
        proposed_cumulative_window_ms=[3660,3696],
        selection_rule='Maximum actual exposed wait among NQL1024 Short internal layers fully released/ready in [3200,4000)ms; overlap candidate additionally intersects NPU0 rid16 L0 IO lifetime. Ties break by npu,rid,layer ascending.',
        limitations=['Existing layer milestones do not identify SSD/link service fragments or bytes missing at deadline.',
                    'The default window compares the same NPU and absolute time, not the same currently active request in both policies.',
                    'A simultaneously active Long is not asserted to be the physical FIFO predecessor.',
                    'The primary current-request D/C reference cannot be integrated over admission lifetime as the actual required bytes.'])
    (HERE/'layer_selection.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(window_U={p:x['mean_device_utilization'] for p,x in summaries.items()},
                         maximum_ids={p:[c['npu'],c['request_id'],c['layer'],c['exposed_stall_ms']] for p,c in maximum.items()},
                         overlap=[coincident['npu'],coincident['request_id'],coincident['layer'],coincident['exposed_stall_ms']])))


if __name__ == '__main__':
    main()
