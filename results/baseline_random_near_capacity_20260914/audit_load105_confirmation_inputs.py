#!/usr/bin/env python3
"""Independently validate all four raw-data confirmation manifests before launch."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import random

from audit_remote_sync import HERE, ROOT, read, sha


def main():
    plan_path = HERE / 'audit_remote_load105_confirmation_plan.json'
    plan = read(plan_path)
    raw = ast.literal_eval((ROOT / 'data').read_text())
    assert [job['seed'] for job in plan['jobs']] == [19, 43, 67, 101]
    assert all(sha(ROOT / name) == digest for name, digest in plan['frozen_source_sha256'].items())
    profiles = [('A',128,256,'LS'), ('B',32,4096,'SL')]
    counts = [32,63]
    block = 176 / 1048576
    audits = {}
    for job in plan['jobs']:
        path = ROOT / job['manifest']; m = read(path); meta = m['metadata']
        assert sha(path) == job['manifest_sha256']
        assert meta['candidate'] == 'reference_load105' and meta['seed'] == job['seed']
        assert meta['family'] == 'raw' and meta['constructed_profile'] is False
        assert meta['num_npu'] == 32 and meta['num_ssu'] == 3 and meta['n_layers'] == 8
        assert meta['count_ratio'] == counts and meta['role_names'] == ['A','B']
        assert meta['profile_keys'] == '128:256,32:4096'
        assert meta['order'] == meta['order_mode'] == 'random' and meta['horizon_pure_compute_ms'] == 22000
        assert meta['source_data_sha256'] == sha(ROOT / 'data')
        assert meta['disk_bw_gib_s'] == 40 and meta['npu_bw_gib_s'] == 50
        C = [raw[k,miss][1] for role,k,miss,category in profiles]
        V = [raw[k,miss][3] for role,k,miss,category in profiles]
        cycle = math.fsum(n * 8 * c / 1000 for n,c in zip(counts,C))
        repeat = math.ceil(22000 / cycle)
        canonical = [i for i,n in enumerate(counts) for _ in range(n * repeat)]
        assert meta['quota_cycles'] == repeat
        assert len(m['requests']) == len(canonical) * 32 == meta['request_count']
        placements = {i:tuple(tuple((int(d),float(v)) for d,v in layer) for layer in value)
                      for i,value in enumerate(m['placements'])}
        placement_repr = {i:repr(value) for i,value in placements.items()}
        groups = defaultdict(list)
        for request in m['requests']:
            groups[request['npu_id']].append(request)
        assert set(groups) == set(range(32))
        fp = hashlib.sha256(b'full-prefill-microbatch-des-input-v2\0')
        all_sorted = []; sequences = set(); verified_placements = set(); blocks_total = 0
        for npu in range(32):
            rows = sorted(groups[npu],key=lambda r:r['request_id'])
            assert len(rows) == len(canonical)
            order = list(range(len(canonical)))
            random.Random(job['seed'] + 100003 * npu).shuffle(order)
            roles = []
            for position,(request,original) in enumerate(zip(rows,order)):
                idx = canonical[original]; role,k,miss,category = profiles[idx]
                load = request['load']; rid = npu * 1000000 + position
                assert request['request_id'] == load['request_id'] == rid and load['npu_id'] == npu
                assert load['original_request_id'] == npu * 1000000 + original
                assert load['profile_index'] == idx and load['role'] == role and load['category'] == category
                assert load['generation'] == position and load['initial'] is True
                assert request['arrival_time_ms'] == load['arrival_time'] == load['arrival_ms'] == 0
                assert load['constructed_profile'] is False and load['profile_construction'] == {'method':'direct_data_row'}
                assert (load['seq_len_k'],load['nql'],load['total_tokens'],load['ssd_prefix_tokens']) == (k,miss,k*1024,k*1024-miss)
                assert load['per_layer_us'] == load['original_compute_us'] == C[idx]
                assert load['per_layer_kv_gb'] == V[idx] and load['source_ttft_ms'] == raw[k,miss][2]
                assert load['padding_gib_per_layer'] == 0
                assert math.isclose(load['required_bw_input_gbps'],V[idx] / (C[idx]/1e6),rel_tol=1e-12)
                nblocks = (k*1024-miss)//128
                assert nblocks * block == V[idx]
                pi = request['placement_index']; key = (pi,npu%3,idx)
                if key not in verified_placements:
                    assert placements[pi] == (tuple(((j+npu)%3,block) for j in range(nblocks)),)
                    verified_placements.add(key)
                fp.update(f"({rid}, {npu}, {float(request['arrival_time_ms'])!r}, {category!r}, {load['per_layer_us']!r}, {placement_repr[pi]})".encode())
                roles.append(role); all_sorted.append(request); blocks_total += 8*nblocks
            assignment = meta['per_npu_assignment'][npu]
            assert assignment['role_counts'] == dict(Counter(roles)) == {'A':32*repeat,'B':63*repeat}
            assert assignment['shuffle_seed'] == job['seed']+100003*npu
            assert assignment['first_32_roles'] == roles[:32]
            assert math.isclose(assignment['pure_compute_ms'],repeat*cycle,rel_tol=1e-12)
            assert assignment['pure_compute_ms'] >= 22000
            sequences.add(tuple(roles))
        assert len(sequences) == 32
        assert fp.hexdigest() == m['input_fingerprint'] == meta['input_fingerprint'] == job['input_fingerprint']
        logical_rows = [dict(request_id=r['request_id'],npu_id=r['npu_id'],arrival_time_ms=r['arrival_time_ms'],load=r['load']) for r in all_sorted]
        logical = hashlib.sha256(json.dumps(logical_rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert logical == meta['logical_input_fingerprint']
        nominal = 32*math.fsum(n*8*v for n,v in zip(counts,V))/(cycle/1000)
        assert math.isclose(nominal/120,meta['ideal_load_ratio'],rel_tol=1e-12)
        audits[job['label']] = dict(manifest_sha256=sha(path),input_fingerprint=fp.hexdigest(),
            logical_input_fingerprint=logical,request_count=len(all_sorted),expected_blocks=blocks_total,
            unique_role_queues=32,per_card_counts={'A':32*repeat,'B':63*repeat},
            per_card_pure_compute_ms=repeat*cycle,ideal_load_ratio=nominal/120,
            every_request_raw_C_V_source_TTFT_checked=True,every_placement_checked=True,full_shuffle_reconstructed=True)
    report = dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(plan_path),auditor_sha256=sha(HERE/'audit_load105_confirmation_inputs.py'),
        seed_selection='19,43,67,101 fixed before these confirmation results; pilot seed7 selected load105 after load sensitivity results.',
        caveat='Ideal population mean demand is about 105% of 3-disk capacity. This is an overload confirmation, not a strict-underload FIFO counterexample.',
        cases=audits)
    (HERE/'audit_load105_confirmation_inputs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(audits),audited_requests=sum(c['request_count'] for c in audits.values()))))


if __name__ == '__main__':
    main()
