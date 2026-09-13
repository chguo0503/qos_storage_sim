#!/usr/bin/env python3
"""Independent offline checks of local A handoffs and the B first-layer example.

No simulator is imported or executed. Physical service is integrated directly
from immutable block timestamps; full-window quantities come from raw layers.
"""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
COLUMNS = ['request_id', 'npu_id', 'layer', 'block_idx', 'ssu_id', 'path_id',
           'size_gib', 'block_count', 'enqueue_ms', 'ssd_start_ms', 'ssd_end_ms',
           'link_start_ms', 'link_end_ms']


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def clip(a, z, left, right):
    return max(0., min(z, right) - max(a, left))


def close(a, b, tolerance=1e-7):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', type=Path, default=STUDY / 'validation20s/runs/ssu4_ordered_k1_sync_seed7/baseline')
    ap.add_argument('--examples', type=Path, default=STUDY / 'figures/ssu4/deadline_ordered.json')
    ap.add_argument('--out', type=Path, default=HERE / 'handoff_checks.json')
    args = ap.parse_args()
    paths = {k: args.case / v for k, v in [('manifest', 'manifest.json.gz'), ('result', 'result.json.gz'),
             ('trace', 'trace.json.gz'), ('command', 'command.json')]}
    paths.update(examples=args.examples, data=ROOT / 'data', analyzer=Path(__file__))
    hashes = {k: sha(p) for k, p in paths.items()}
    man, raw, trace, command, examples = [read(paths[k]) for k in ('manifest', 'result', 'trace', 'command', 'examples')]
    data = ast.literal_eval(paths['data'].read_text())
    checks = {}

    def check(key, value):
        checks[key] = bool(value)
        assert value, key

    check('complete_run', command['status'] == 'complete' and command['returncode'] == 0 and trace['completed_simulation'])
    check('manifest_hash', hashes['manifest'] == command['manifest_sha256'] == trace['source']['manifest_sha256'])
    check('result_hash', hashes['result'] == command['output_sha256'] == trace['source']['reference_sha256'])
    check('trace_hash', hashes['trace'] == command['trace_sha256'])
    check('input_fingerprint', man['input_fingerprint'] == raw['input_fingerprint'] == trace['source']['input_fingerprint'])
    check('frozen_core_hashes', raw['core_and_policy_sha256'] == command['core_source_sha256'] == trace['source']['core_source_sha256']
          and all(sha(ROOT / name) == h for name, h in raw['core_and_policy_sha256'].items()))
    check('data_hash', man['metadata']['source_data_sha256'] == hashes['data'])
    check('original_invariants', bool(raw['summary']['invariants']) and all(raw['summary']['invariants'].values()))
    check('trace_completion_checks', bool(trace['checks']) and all(trace['checks'].values()))
    check('scope', raw['strategy'] == 'baseline' and raw['submit_seed'] == 7 and man['metadata']['num_npu'] == 32
          and man['metadata']['num_ssu'] == 4 and raw['summary']['n_layers'] == 8)
    check('column_schema', trace['columns'] == COLUMNS)
    arr = np.asarray(trace.pop('rows'), dtype=float)
    check('finite_rows', arr.ndim == 2 and arr.shape[1] == 13 and np.all(np.isfinite(arr)))
    check('block_path0', np.all(arr[:, 5] == 0) and np.all(arr[:, 7] == 1))
    for name, start, end, resource, count, rate in [('ssd', 9, 10, 4, 4, 40.), ('link', 11, 12, 1, 32, 50.)]:
        check(name + '_service_duration', np.allclose(arr[:, end] - arr[:, start], arr[:, 6] * 1000 / rate, atol=1e-8, rtol=1e-9))
        for n in range(count):
            lane = arr[arr[:, resource] == n]
            lane = lane[np.argsort(lane[:, start], kind='stable')]
            check(f'{name}_{n}_no_overlap', np.all(lane[1:, start] >= lane[:-1, end] - 1e-8))
    reqs = {r['request_id']: r for r in man['requests']}
    batches = {b['member_request_ids'][0]: b for b in raw['summary']['microbatch_metrics']}
    for req in reqs.values():
        key = (req['load']['seq_len_k'], req['load']['nql'])
        check('raw_data_' + str(key), req['load']['per_layer_us'] == data[key][1] and close(req['load']['per_layer_kv_gb'], data[key][3]))
    CA, VA = data[(128, 256)][1] / 1000., data[(128, 256)][3]
    predicted_R = 32 * VA / (4 * 40.) * 1000
    a_rows = []
    for n in range(32):
        rid = n * 1000000 + 9
        b = batches[rid]; layer, previous = b['layer_metrics'][2], b['layer_metrics'][1]
        R = layer['io_ready_time_ms'] - layer['io_start_time_ms']
        T = layer['compute_start_ms'] - previous['compute_start_ms']
        S = layer['compute_start_ms'] - previous['compute_end_ms']
        check(f'A{n}_exact_handoff', close(R, predicted_R) and close(T, max(CA, R)) and close(S, max(0., R-CA))
              and close(S, layer['io_barrier_wait_ms']) and close(layer['io_start_time_ms'], previous['compute_start_ms']))
        a_rows.append(dict(npu=n, request_id=rid, layer=2, release_ms=layer['io_start_time_ms'],
                           ready_ms=layer['io_ready_time_ms'], deadline_ms=previous['compute_end_ms'],
                           R_ms=R, S_ms=S, T_ms=T, C_ms=CA, local_cycle_U_percent=100*CA/T))
    left, right = a_rows[0]['release_ms'], a_rows[0]['ready_ms']
    check('capture_covers_A_interval', trace['window_ms'][0] <= left < right <= trace['window_ms'][1])
    common_C, common_A_active, common_other_active = [0.]*32, [0.]*32, [0.]*32
    for b in batches.values():
        n = b['npu_id']; q = reqs[b['member_request_ids'][0]]['load']
        occupied = clip(b['admission_time_ms'], b['completion_time_ms'], left, right)
        destination = common_A_active if (q['seq_len_k'], q['nql']) == (128,256) else common_other_active
        destination[n] += occupied
        common_C[n] += math.fsum(clip(l['compute_start_ms'], l['compute_end_ms'], left, right) for l in b['layer_metrics'])
    check('all32_A_active_entire_common_window', all(close(v,right-left) for v in common_A_active) and all(v == 0. for v in common_other_active))
    actual_common_U = 100*math.fsum(common_C)/(32*(right-left))
    check('actual_common_window_U_equals_local_model', close(actual_common_U,100*CA/predicted_R))

    def integrate(left, right, start, end, resource, count, rate):
        duration = np.maximum(0., np.minimum(arr[:, end], right) - np.maximum(arr[:, start], left))
        gib = np.bincount(arr[:, resource].astype(int), weights=duration * rate / 1000, minlength=count)
        return dict(bytes_GiB=gib.tolist(), mean_GiB_s=(gib*1000/(right-left)).tolist())

    disks = integrate(left, right, 9, 10, 4, 4, 40.)
    ssd = integrate(left, right, 9, 10, 1, 32, 40.)
    link = integrate(left, right, 11, 12, 1, 32, 50.)
    check('A_service_conservation', close(math.fsum(disks['bytes_GiB']), math.fsum(ssd['bytes_GiB'])))
    idle = [(right-left) - v*1000/40. for v in disks['bytes_GiB']]
    # Boundary effects are measurements, not assertions of an ideal allocation.
    saturated = all(abs(x) < 1e-7 for x in idle)
    per_npu = [dict(npu=n, SSD_GiB=ssd['bytes_GiB'][n], SSD_mean_GiB_s=ssd['mean_GiB_s'][n],
                    link_GiB=link['bytes_GiB'][n], link_mean_GiB_s=link['mean_GiB_s'][n]) for n in range(32)]

    target = examples['B']; rid, layer_index = target['request_id'], target['layer']
    check('B_requested_target', rid == 15000010 and layer_index == 0)
    b = batches[rid]; layer = b['layer_metrics'][0]
    lane = sorted((x for x in batches.values() if x['npu_id'] == b['npu_id']), key=lambda x: x['admission_time_ms'])
    index = next(i for i, x in enumerate(lane) if x['member_request_ids'][0] == rid)
    check('B_has_predecessor', index > 0)
    prev_batch = lane[index-1]; prev_rid = prev_batch['member_request_ids'][0]; prev = prev_batch['layer_metrics'][-1]
    check('B_predecessor_A', (reqs[prev_rid]['load']['seq_len_k'], reqs[prev_rid]['load']['nql']) == (128, 256))
    blocks = arr[(arr[:, 0] == rid) & (arr[:, 2] == 0)]
    blocks = blocks[np.argsort(blocks[:, 3], kind='stable')]
    placement = man['placements'][reqs[rid]['placement_index']][0]
    check('B_complete_unique_blocks', len(blocks) == len(placement) and np.array_equal(blocks[:, 3], np.arange(len(placement))))
    check('B_complete_placement', all(int(r[1]) == b['npu_id'] and int(r[4]) == disk and r[6] == size for r, (disk, size) in zip(blocks, placement)))
    release, deadline, ready = layer['io_start_time_ms'], prev['compute_end_ms'], layer['io_ready_time_ms']
    payload = math.fsum(size for _, size in placement)*1024
    check('B_release_and_budget', close(release, prev['compute_start_ms']) and close(deadline-release, CA) and close(deadline, b['admission_time_ms']))
    check('B_last_block_ready', close(float(blocks[:, 12].max()), ready))
    check('B_figure_matches', all(close(v, target[k]) for k, v in [('release_ms', release), ('deadline_ms', deadline), ('ready_ms', ready), ('payload_MiB', payload)]))
    delivered = {}
    for name, start, end in [('SSD', 9, 10), ('HBM', 11, 12)]:
        arrived = blocks[:, 6]*1024*np.clip((deadline-blocks[:, start])/(blocks[:, end]-blocks[:, start]), 0., 1.)
        delivered[name] = float(arrived.sum())
    stall = max(0., ready-deadline)
    check('B_deadline_received_zero', delivered['HBM'] == 0.)
    check('B_formula12', close(stall, layer['io_barrier_wait_ms']) and close(stall, target['stall_ms']))

    # Demonstrate why averaging nominal and delivered bandwidth is not formula 8.
    wl, wr, npu = 2000., 4000., 15
    nominal_integral, compute, active = 0., 0., 0.
    for b in batches.values():
        if b['npu_id'] != npu:
            continue
        q = reqs[b['member_request_ids'][0]]['load']
        overlap = clip(b['admission_time_ms'], b['completion_time_ms'], wl, wr)
        active += overlap
        nominal_integral += overlap*q['per_layer_kv_gb']*1e6/q['per_layer_us']
        compute += math.fsum(clip(l['compute_start_ms'], l['compute_end_ms'], wl, wr) for l in b['layer_metrics'])
    mean_nominal = nominal_integral/(wr-wl)
    mean_ssd = integrate(wl, wr, 9, 10, 1, 32, 40.)['mean_GiB_s'][npu]
    mean_link = integrate(wl, wr, 11, 12, 1, 32, 50.)['mean_GiB_s'][npu]
    check('NPU15_full_window_active', close(active, wr-wl))
    check('sources_unchanged', all(sha(path) == hashes[k] for k, path in paths.items()))
    out = dict(technical_passed=all(checks.values()), checks=checks, no_simulation_run=True,
               sources={k: dict(path=str(p.resolve()), sha256=hashes[k]) for k,p in paths.items()},
               A_local=dict(profile=[128,256], C_ms=CA, layer_V_MiB=VA*1024,
                  prediction_assumptions='A saturated rotation with 32 equally sized active consumers sharing four 40GiB/s SSDs; all layers must be released and sustained backlog maintained. These scheduling assumptions are checked locally after observing this run.',
                  model_R_ms=predicted_R, effective_cycle_bandwidth_GiB_s=4*40/32,
                  model_cycle_U_percent=100*CA/predicted_R, per_npu_handoffs=a_rows,
                  actual_common_window_ms=[left,right], actual_per_ssu=disks,
                  actual_common_window_compute_ms_by_npu=common_C,
                  actual_common_window_total_compute_ms=math.fsum(common_C),
                  actual_common_window_fleet_U_percent=actual_common_U,
                  actual_A_occupied_ms_by_npu=common_A_active,
                  common_window_note='This is also a direct shared-window fleet-U measurement, independently clipped from actual compute intervals; each card remains in A throughout. Equality with local C/R is observed here, not presumed for other windows.',
                  per_ssu_idle_ms=idle, all_four_ssds_busy_whole_interval=saturated,
                  actual_per_npu=per_npu,
                  all_npu_SSD_means_equal_five=all(abs(v-5.)<1e-7 for v in ssd['mean_GiB_s']),
                  note='R is release-to-last-HBM latency, not an SSD-only interval. Common-window resource shares can contain other layer ordinals and boundary effects. Mean=5 would not mean constant service; queued block service is discrete.'),
               B_first_layer=dict(request_id=rid, npu=batches[rid]['npu_id'], layer=0, preceding_A_request_id=prev_rid,
                  captured_blocks=len(blocks), expected_blocks=len(placement), release_ms=release,
                  deadline_ms=deadline, ready_ms=ready, budget_C_ms=deadline-release,
                  payload_MiB=payload, before_deadline_MiB=delivered, HBM_deficit_MiB=payload-delivered['HBM'],
                  stall_ms=stall, actual_R_ms=ready-release, formula11_passed=delivered['HBM']>=payload,
                  formula12_verified=True, budget_GiB_s=payload/1024*1000/(deadline-release)),
               invalid_full_window_extension=dict(npu=npu, window_ms=[wl,wr], mean_nominal_GiB_s=mean_nominal,
                  actual_mean_SSD_GiB_s=mean_ssd, actual_mean_HBM_GiB_s=mean_link,
                  invalid_SSD_ratio_U_percent=100*min(1.,mean_ssd/mean_nominal),
                  invalid_HBM_ratio_U_percent=100*min(1.,mean_link/mean_nominal),
                  actual_compute_U_percent=100*compute/(wr-wl),
                  reason='Current profile and delivery timing change; nominal rate remains displayed during stall, and first-layer prefetch can serve the next request. Ratio of window averages is not the constant-flow model.'),
               inference_limit='A local fluid-cycle equality is supported here, conditional on the observed saturated cyclic regime. It is not an a-priori prediction from A:B ratio alone, a pointwise bandwidth allocation, a whole-window U formula, or proof that another order behaves identically.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.out),technical_passed=out['technical_passed'],R_ms=predicted_R,
         cycle_U=out['A_local']['model_cycle_U_percent'],SSD_means=disks['mean_GiB_s'],
         all_npu_means_five=out['A_local']['all_npu_SSD_means_equal_five'],
         npu_SSD_minmax=[min(ssd['mean_GiB_s']),max(ssd['mean_GiB_s'])],B=out['B_first_layer'],
         invalid=out['invalid_full_window_extension']),ensure_ascii=False))


if __name__ == '__main__':
    main()
