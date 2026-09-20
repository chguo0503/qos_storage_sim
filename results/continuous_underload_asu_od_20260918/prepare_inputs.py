#!/usr/bin/env python3
"""Freeze the document's exact queues with immutable request hash identities."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
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

LENGTHS = (32, 64, 80, 128, 160)
MISSES = (2048, 4096)
SEEDS = (7, 19, 43)
LAYERS, NPUS, DISKS = 8, 32, 3
BLOCK_GIB = 176 * 1024 / 2**30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(order, seed):
    keys = ','.join(f'{length}:{miss}' for length in LENGTHS for miss in MISSES)
    profiles, provenance = profiles_for('raw', keys)
    canonical = [index for index in range(10) for _ in range(2)]
    pure_ms = math.fsum(LAYERS * profiles[index]['per_layer_compute_us'] / 1000 for index in canonical)
    requests, csv_rows, per_npu, all_rates = [], [], [], []
    identity_rows = []
    for index, profile in enumerate(profiles):
        profile['role'] = profile['name'] = f"L{profile['seq_len_k']}_M{profile['nql']}"
        profile['category'] = sim.classify_request(profile['seq_len_k'], profile['nql'])
        assert profile['construction']['method'] == 'direct_data_row'
        assert profile['ssd_prefix_tokens'] % 128 == 0
        assert math.isclose(profile['source_equivalent_ttft_78_layers_ms'],
                            78 * profile['per_layer_compute_us'] / 1000, abs_tol=1e-9)
    for npu in range(NPUS):
        identities = list(range(20))
        if order == 'random':
            random.Random(seed + npu).shuffle(identities)
        elif order != 'ordered':
            raise ValueError(order)
        lane_rates = []
        for position, original in enumerate(identities):
            index = canonical[original]
            profile = profiles[index]
            rid = npu * 20 + position
            original_id = npu * 20 + original
            layer = tuple((sim.block_ring_hash_disk_id(original_id, block, DISKS), BLOCK_GIB)
                          for block in range(profile['ssd_prefix_tokens'] // 128))
            volume = math.fsum(size for _, size in layer)
            assert math.isclose(volume, profile['per_layer_kv_gib'], abs_tol=1e-12)
            disk_bytes = [math.fsum(size for d, size in layer if d == disk) for disk in range(DISKS)]
            compute_ms = profile['per_layer_compute_us'] / 1000
            rates = [v * 1000 / compute_ms for v in disk_bytes]
            lane_rates.append((rates, LAYERS * compute_ms))
            load = dict(request_id=rid, npu_id=npu, generation=position,
                        original_request_id=original_id, original_sequence=original,
                        profile_index=index, role=profile['role'], seq_len_k=profile['seq_len_k'],
                        nql=profile['nql'], total_tokens=profile['total_tokens'],
                        ssd_prefix_tokens=profile['ssd_prefix_tokens'], category=profile['category'],
                        per_layer_us=profile['per_layer_compute_us'], per_layer_kv_gb=volume,
                        required_bw_input_gbps=profile['required_bandwidth_gibps'],
                        source_ttft_ms=profile['source_equivalent_ttft_78_layers_ms'],
                        original_compute_us=profile['per_layer_compute_us'], constructed_profile=False,
                        profile_construction=profile['construction'], padding_gib_per_layer=0.0,
                        arrival_time=0.0, arrival_ms=0.0, initial=True)
            requests.append(ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, (layer,)))
            csv_rows.append(dict(request_id=original_id, runtime_request_id=rid, npu_id=npu,
                                 sequence=position, original_sequence=original,
                                 total_tokens_k=profile['seq_len_k'], total_tokens=profile['total_tokens'],
                                 nql=profile['nql'], hit_tokens=profile['ssd_prefix_tokens'], arrival_ms=0.0,
                                 per_layer_compute_ms=compute_ms, per_layer_read_GiB=volume,
                                 **{f'per_layer_SSU{d}_read_GiB': v for d, v in enumerate(disk_bytes)}))
            identity_rows.append((original_id, npu, profile['seq_len_k'], profile['nql'],
                                  profile['per_layer_compute_us'], layer))
        all_rates.append(lane_rates)
        per_npu.append(dict(npu_id=npu, requests=20, pure_compute_ms=pure_ms,
                            shuffle_seed=seed + npu if order == 'random' else None,
                            first_20_profiles=[profiles[canonical[i]]['role'] for i in identities],
                            profile_counts=dict(Counter(profiles[i]['role'] for i in canonical))))
    requests = tuple(requests)
    identity_bytes = json.dumps(sorted(identity_rows), separators=(',', ':')).encode()
    static_max = [math.fsum(max(rate[d] for rate, _ in lane) for lane in all_rates) for d in range(DISKS)]
    static_min = [math.fsum(min(rate[d] for rate, _ in lane) for lane in all_rates) for d in range(DISKS)]
    mean = [math.fsum(rate[d] * cost / pure_ms for lane in all_rates for rate, cost in lane) for d in range(DISKS)]
    fp = continuous_batch_input_fingerprint(requests)
    label = f'document_{order}_seed{seed}_ring_hash'
    meta = dict(experiment='continuous_underload_asu_od_20260918', label=label, case_id=label+'_'+fp[:12],
                scenario_candidate='document', num_npu=NPUS, num_ssu=DISKS, n_layers=LAYERS,
                disk_bw_gib_s=40.0, npu_bw_gib_s=50.0, seed=seed, order=order, order_mode=order,
                family='raw', profiles=profiles, profile_keys=keys, source=provenance,
                source_data_sha256=sha(ROOT/'data'), source_document_sha256=sha(ROOT/'docs/continuous_underload_reproduction.md'),
                equal_176kib_blocks=True, blocks='exact', constructed_profile=False, compute_scale_actual=1.0,
                input_fingerprint=fp, logical_input_fingerprint=logical_input_fingerprint(requests),
                canonical_population_and_placement_sha256=hashlib.sha256(identity_bytes).hexdigest(),
                request_count=640, requests_per_npu=20, count_per_profile_per_npu=2,
                actual_pure_compute_ms_per_npu=pure_ms, last_arrival_ms=0.0, quota_repeats=2,
                per_npu_assignment=per_npu, per_length_miss_counts={'2048': 2, '4096': 2},
                total_input_Ki_tokens=59392, measurement_window_ms=[2000,4000],
                additional_measurement_windows_ms=[[2000,6000]], layout=sim.PLACEMENT_BLOCK_RING_HASH,
                placement_identity_key='original_request_id',
                placement_rule='sim.block_ring_hash_disk_id(original_request_id,block_index,3); immutable across layers and order',
                identity_rule='original_request_id=npu*20+canonical_ordinal; runtime request_id=npu*20+queue_position',
                random_rule='Random(seed+npu_id).shuffle(20 individual requests); exactly the document profile sequence',
                ordered_rule='ascending total length then miss, each profile twice, same canonical sequence on all cards',
                arrival_rule='all640 requests arrive at0; fixed NPU queues; no staggering, external rate limit or compute scaling',
                source_ttft_rule='source data TTFT equals78*C; experiment uses8 layers and own_compute=8*C',
                nominal_definition='sum of current admitted request per-disk layer volume / its layer compute time; includes stall residence; excludes queued and pre-admission next-request L0 duplication',
                nominal_capacity_status='TO_BE_MEASURED; record overload without changing input',
                static_per_ssu_lower_bound_gib_s=static_min, static_per_ssu_upper_bound_gib_s=static_max,
                static_underload_guarantee=all(v<40 for v in static_max),
                time_weighted_per_ssu_nominal_gib_s=mean, time_weighted_fleet_nominal_gib_s=math.fsum(mean),
                fluid_rho=math.fsum(mean)/120)
    assert sum(row['total_tokens'] for row in csv_rows) == 59392 * 1024
    return requests, meta, csv_rows


def main():
    inputs = HERE/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    audit = dict(source_document_sha256=sha(ROOT/'docs/continuous_underload_reproduction.md'),
                 source_data_sha256=sha(ROOT/'data'), builder_sha256=sha(__file__),
                 no_input_changes_to_enforce_underload=True, seeds=list(SEEDS), orders=['random','ordered'], cases=[])
    populations = set()
    for order in ('random','ordered'):
        for seed in SEEDS:
            requests, meta, rows = build(order, seed)
            path = inputs/f'{order}_seed{seed}_ring_hash.json.gz'
            save_manifest(path, requests, meta)
            restored, saved = load_manifest(path)
            assert saved == meta and continuous_batch_input_fingerprint(restored) == meta['input_fingerprint']
            csv_path = path.with_name(f'{order}_seed{seed}.csv')
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)
            populations.add(meta['canonical_population_and_placement_sha256'])
            case = dict(order=order, seed=seed, manifest=str(path.relative_to(ROOT)),
                        csv=str(csv_path.relative_to(ROOT)), manifest_sha256=sha(path), csv_sha256=sha(csv_path),
                        requests=len(requests), blocks=sum(len(q.placement[0])*LAYERS for q in requests),
                        pure_compute_ms_per_npu=meta['actual_pure_compute_ms_per_npu'],
                        canonical_population_and_placement_sha256=meta['canonical_population_and_placement_sha256'],
                        static_per_ssu_upper_bound_gib_s=meta['static_per_ssu_upper_bound_gib_s'],
                        input_fingerprint=meta['input_fingerprint'])
            audit['cases'].append(case)
            print(json.dumps(case), flush=True)
    assert len(populations)==1
    audit['all_orders_and_seeds_share_same_requests_and_placement']=True
    (HERE/'input_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')


if __name__=='__main__':
    main()
