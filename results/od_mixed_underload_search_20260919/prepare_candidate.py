#!/usr/bin/env python3
"""Freeze an explicit, per-NPU mixed queue without runtime admission control."""
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulator.core.continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from inputs.runners.run_baseline_npu32_stress import profiles_for, save_manifest, load_manifest
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from simulator.core import sim

BLOCK = 176*1024/2**30


def make_profiles(spec):
    profiles = {}
    for role in ('A', 'B'):
        cfg = spec['profiles'][role]
        if 'data_key' in cfg:
            key = cfg['data_key']
            result, provenance = profiles_for('raw', f'{key[0]}:{key[1]}')
            p = dict(result[0])
            p['block_count'] = p['ssd_prefix_tokens']//128
            assert p['ssd_prefix_tokens'] % 128 == 0
            p['provenance'] = provenance
        else:
            count = cfg['block_count']
            assert isinstance(count, int) and count > 0
            # Synthetic requests retain internally consistent token/byte metadata.
            miss = cfg.get('miss_tokens', 256 if role == 'A' else 4096)
            hit = count*128
            total = hit+miss
            p = dict(seq_len_k=total/1024, total_tokens=total, nql=miss,
                     ssd_prefix_tokens=hit, block_count=count,
                     per_layer_kv_gib=count*BLOCK,
                     per_layer_compute_us=cfg['C_ms']*1000,
                     source_equivalent_ttft_78_layers_ms=78*cfg['C_ms'],
                     construction=dict(method='synthetic_equal_176KiB_blocks',
                                       note='C is constructed, not a data measurement',
                                       requested_C_ms=cfg['C_ms'], block_count=count))
        scale = cfg.get('compute_scale', 1.0)
        p['original_compute_us'] = p['per_layer_compute_us']
        p['per_layer_compute_us'] *= scale
        if scale != 1:
            p['construction'] = dict(method='scaled_data_compute',
                                      original=p['construction'], compute_scale=scale)
        p['required_bandwidth_gibps'] = p['per_layer_kv_gib']*1e6/p['per_layer_compute_us']
        p['role'] = role
        p['category'] = sim.classify_request(p['seq_len_k'], p['nql'])
        profiles[role] = p
    return profiles


def make_queues(spec, profiles):
    if 'queues' in spec:
        queues = [list(q) for q in spec['queues']]
        assert len(queues) == 32
        assert all(set(q) == {'A', 'B'} for q in queues)
        return queues
    scheme = spec['schedule']
    na, nb = scheme.get('a_run', 4), scheme.get('b_run', 1)
    cycle = ['A']*na+['B']*nb
    horizon = spec.get('horizon_pure_compute_ms', 8500)
    cost = {k: 8*p['per_layer_compute_us']/1000 for k, p in profiles.items()}
    repeats = math.ceil(horizon/(na*cost['A']+nb*cost['B']))+1
    base = cycle*repeats
    queues = []
    mode = scheme.get('mode', 'phase')
    group_count = scheme.get('groups', 4)
    group_sizes = scheme.get('group_sizes')
    if group_sizes:
        assert sum(group_sizes) == 32
        group_ids = [g for g, count in enumerate(group_sizes) for _ in range(count)]
        group_count = len(group_sizes)
    else:
        group_ids = [n % group_count for n in range(32)]
    predicted_cost = dict(cost)
    predicted_cost['A'] /= scheme.get('A_utilization_guess', .85)
    predicted_cost['B'] /= scheme.get('B_utilization_guess', 1.0)
    phase_costs = [0.0]
    for role in cycle:
        phase_costs.append(phase_costs[-1]+predicted_cost[role])
    for npu in range(32):
        if mode == 'random':
            queue = list(base)
            random.Random(spec.get('seed', 7)+npu*100003).shuffle(queue)
        elif mode == 'phase':
            target = phase_costs[-1]*group_ids[npu]/group_count
            offset = min(range(len(cycle)), key=lambda i: abs(phase_costs[i]-target))
            queue = base[offset:]+base[:offset]
        elif mode == 'prefix':
            prefixes = scheme.get('b_prefix_by_group', list(range(group_count)))
            queue = ['B']*prefixes[group_ids[npu]]+base
        elif mode == 'synchronized':
            queue = list(base)
        elif mode == 'balanced_roles':
            # Rotate a long logical period at evenly spaced request boundaries.
            target = sum(predicted_cost[r] for r in base)*npu/32
            acc, offset = 0.0, 0
            while offset+1 < len(base) and acc < target:
                acc += predicted_cost[base[offset]]
                offset += 1
            queue = base[offset:]+base[:offset]
        else:
            raise ValueError(mode)
        queues.append(queue)
    return queues


def build(spec):
    profiles = make_profiles(spec)
    queues = make_queues(spec, profiles)
    seed = spec.get('seed', 7)
    requests = []
    per_npu = []
    identity_offset = spec.get('identity_offset', 0)
    selected_addresses = None
    pool_cursors = Counter()
    if spec.get('address_pool'):
        pool_path = HERE/spec['address_pool']
        selected_addresses = json.loads(pool_path.read_text())
        assert selected_addresses['status'] == 'complete'
        pool_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    for npu, queue in enumerate(queues):
        counts = Counter(queue)
        # Same per-card role population has the same immutable physical IDs
        # across ordering treatments. Runtime IDs identify queue positions only.
        offsets = dict(A=0, B=counts['A'])
        seen = Counter()
        for position, role in enumerate(queue):
            p = profiles[role]
            original = identity_offset+npu*1000000+offsets[role]+seen[role]
            seen[role] += 1
            if selected_addresses is not None:
                entry = selected_addresses['pools'][role][pool_cursors[role]]
                pool_cursors[role] += 1
                original = entry['request_id']
            rid = npu*1000000+position
            layer = tuple((sim.block_ring_hash_disk_id(original, k, 3), BLOCK)
                          for k in range(p['block_count']))
            if selected_addresses is not None:
                assert [sum(d == disk for d, _ in layer) for disk in range(3)] == entry['counts_by_ssu']
            volume = math.fsum(v for _, v in layer)
            assert abs(volume-p['per_layer_kv_gib']) < 1e-12
            load = dict(request_id=rid, npu_id=npu, generation=position,
                        original_request_id=original, role=role,
                        seq_len_k=p['seq_len_k'], total_tokens=p['total_tokens'], nql=p['nql'],
                        ssd_prefix_tokens=p['ssd_prefix_tokens'], category=p['category'],
                        per_layer_us=p['per_layer_compute_us'], per_layer_kv_gb=volume,
                        required_bw_input_gbps=p['required_bandwidth_gibps'],
                        source_ttft_ms=p['source_equivalent_ttft_78_layers_ms'],
                        original_compute_us=p['original_compute_us'],
                        constructed_profile=p['construction']['method'] != 'direct_data_row',
                        profile_construction=p['construction'], padding_gib_per_layer=0.0,
                        arrival_time=0.0, arrival_ms=0.0, initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid, npu, 0., load, (layer,)))
        rates = [[math.fsum(v for d, v in r.placement[0] if d == disk)*1e6/r.load['per_layer_us']
                  for disk in range(3)] for r in requests if r.npu_id == npu]
        per_npu.append(dict(npu_id=npu, role_counts=dict(counts), request_count=len(queue),
                            pure_compute_ms=sum(8*profiles[r]['per_layer_compute_us']/1000 for r in queue),
                            per_ssu_rate_max=[max(r[d] for r in rates) for d in range(3)],
                            queue=''.join(queue)))
    requests = tuple(requests)
    assert len({r.load['original_request_id'] for r in requests}) == len(requests)
    meta = dict(experiment='od_mixed_underload_search_20260919', label=spec['label'],
                case_id=spec['label'], num_npu=32, num_ssu=3, n_layers=8, seed=seed,
                order=spec.get('schedule', {}).get('mode', 'explicit'),
                scenario_candidate='mixed_underload_candidate',
                source_data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
                profiles=profiles, candidate_spec=spec, per_npu_assignment=per_npu,
                constructed_profile=any(p['construction']['method'] != 'direct_data_row' for p in profiles.values()),
                layout=sim.PLACEMENT_BLOCK_RING_HASH, equal_176kib_blocks=True, blocks='exact',
                request_count=len(requests), last_arrival_ms=0., disk_bw_gib_s=40., npu_bw_gib_s=50.,
                placement_identity_key='original_request_id',
                placement_rule='sim.block_ring_hash_disk_id(original_request_id,block_index,3)',
                identity_rule='runtime ID follows queue order; physical ID follows role occurrence in per-NPU canonical population',
                expected_blocks=sum(len(r.placement[0])*8 for r in requests),
                input_fingerprint=continuous_batch_input_fingerprint(requests),
                logical_input_fingerprint=logical_input_fingerprint(requests),
                static_per_ssu_upper_bound_GiB_s=[sum(r['per_ssu_rate_max'][d] for r in per_npu) for d in range(3)],
                nominal_definition='Current admitted request V_is/C_i, including stalls; no double-counting next L0.',
                no_runtime_gating=True, no_artificial_idle=True,
                caveat='Candidate: strict underload and both roles actually computing in warm must be verified from execution.')
    if selected_addresses is not None:
        meta.update(address_selection=dict(pool_file=spec['address_pool'], pool_sha256=pool_sha,
                                           method='adversarial selection of distinct request IDs by their native ring-hash per-disk block counts',
                                           counts_used=dict(pool_cursors),
                                           is_random_address_population=False,
                                           placement_function_unchanged=True),
                    identity_rule='Distinct native Ring Hash address IDs from the disclosed selected pool; runtime IDs still follow queue order.')
    return requests, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    assert Path(spec['label']).name == spec['label']
    requests, meta = build(spec)
    target = HERE/'inputs'/f"{spec['label']}.json.gz"
    save_manifest(target, requests, meta)
    restored, stored = load_manifest(target)
    assert len(restored) == len(requests) and stored == meta
    print(json.dumps(dict(manifest=str(target), label=spec['label'],
                         requests=len(requests), blocks=meta['expected_blocks'],
                         constructed=meta['constructed_profile'],
                         pure_compute_ms_range=[min(r['pure_compute_ms'] for r in meta['per_npu_assignment']),
                                                max(r['pure_compute_ms'] for r in meta['per_npu_assignment'])],
                         static_upper=meta['static_per_ssu_upper_bound_GiB_s'])))


if __name__ == '__main__':
    main()
