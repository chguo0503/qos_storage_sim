#!/usr/bin/env python3
"""Screen all original data rows for 16-NPU AB head-of-line experiments.

No simulation, interpolation, or extrapolation. All bandwidths are decimal
GB/s. Burst calculations are hypotheses about a specific enqueue order, not
predictions of random-trace utilization. RingHash checks certify only the
finite generated input queues and the specified nominal-demand definition.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import struct

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
N = 16
CAPACITY = 40.0
SEEDS = (7, 19, 43)


def profiles():
    raw = ast.literal_eval((REPO / 'data').read_text())
    answer = []
    for (length, miss), values in sorted(raw.items()):
        prefix = length * 1024 - miss
        volume = prefix * 1408
        compute_ms = values[1] / 1000
        assert math.isclose(volume / 2**30, values[3], abs_tol=1e-14)
        assert math.isclose(values[3] / (compute_ms / 1000), values[0], rel_tol=1e-12)
        answer.append(dict(length_k=length, miss=miss, hit_tokens=prefix,
                           C_ms=compute_ms, V_bytes=volume, V_MB=volume / 1e6,
                           B_GB_s=volume / (compute_ms / 1000) / 1e9,
                           source_key=[length, miss], source='data',
                           original_row=list(values)))
    assert len(answer) == 84
    return answer


def pair_screen(a, b):
    x, y = a['V_MB'] / b['V_MB'], a['C_ms'] / b['C_ms']
    s = math.floor(N * max(a['B_GB_s'], b['B_GB_s']) / CAPACITY) + 1
    capacity = CAPACITY * s
    choices = []
    for na in range(1, N):
        nb = N - na
        # A synchronized batch of A reads is placed fully ahead of B reads.
        # If repeated once per C_A, this has the following dimensional scale.
        # It is deliberately NOT called an upper/lower bound for the simulator.
        waiting_ms = max(0.0, na * a['V_MB'] / capacity - b['C_ms'])
        scale = nb / N * waiting_ms / a['C_ms']
        choices.append((scale, na, nb, y * nb / na, waiting_ms))
    scale, na, nb, ratio, wait = max(choices)
    return dict(A_key=a['source_key'], B_key=b['source_key'],
                A_C_ms=a['C_ms'], B_C_ms=b['C_ms'], A_V_MB=a['V_MB'], B_V_MB=b['V_MB'],
                A_B_GB_s=a['B_GB_s'], B_B_GB_s=b['B_GB_s'], x=x, y=y,
                ideal_min_S_any_mix=s, ideal_total_peak_GB_s=N * max(a['B_GB_s'], b['B_GB_s']),
                single_A_service_over_C_B=a['V_MB'] / capacity / b['C_ms'],
                A_count_needed_ahead_of_B=math.floor(capacity * b['C_ms'] / a['V_MB']) + 1,
                heuristic_nA=na, heuristic_nB=nb,
                no_stall_count_ratio_B_per_A=ratio,
                hypothetical_B_wait_ms=wait,
                hypothetical_fleet_wait_scale=scale,
                note='Synchronized bulk enqueue heuristic, not a utilization bound or forecast; ideal equal striping.')


def hpos(namespace, first, second):
    return int.from_bytes(hashlib.sha256(namespace + struct.pack('!QQ', first, second)).digest(), 'big')


@lru_cache(maxsize=None)
def ring(s):
    rows = sorted((hpos(b'qos_storage_sim:block_ring_hash:vnode:v1\0', disk, v), disk)
                  for disk in range(s) for v in range(256))
    return tuple(x for x, _ in rows), tuple(d for _, d in rows)


@lru_cache(maxsize=12000)
def block_counts(s, rid, prefix):
    if s == 1:
        return (prefix * 1408,)
    positions, disks = ring(s)
    answer = [0] * s
    for block in range(math.ceil(prefix / 128)):
        position = hpos(b'qos_storage_sim:block_ring_hash:block:v1\0', rid, block)
        disk = disks[bisect.bisect_left(positions, position) % len(positions)]
        answer[disk] += min(128, prefix - block * 128) * 1408
    assert sum(answer) == prefix * 1408
    return tuple(answer)


def static_ring_screen(a, b, s, q, seed, horizon_ms):
    cycles = math.ceil(horizon_ms / (8 * (a['C_ms'] + q * b['C_ms']))) + 1
    worst = [[0.0] * s for _ in range(N)]
    max_a_bytes = 0
    for npu in range(N):
        deck = ['A'] * cycles + ['B'] * (q * cycles)
        random.Random(seed + 100003 * npu).shuffle(deck)
        for generation, role in enumerate(deck):
            p = a if role == 'A' else b
            sizes = block_counts(s, npu * 1000000 + generation, p['hit_tokens'])
            if role == 'A':
                max_a_bytes = max(max_a_bytes, max(sizes))
            for disk, size in enumerate(sizes):
                worst[npu][disk] = max(worst[npu][disk], size / (p['C_ms'] / 1000) / 1e9)
    per_disk = [sum(row[disk] for row in worst) for disk in range(s)]
    return dict(seed=seed, horizon_ms=horizon_ms, S=s, request_count_ratio=[1, q],
                requests_per_npu=(q + 1) * cycles, per_disk_upper_GB_s=per_disk,
                peak_upper_GB_s=max(per_disk),
                any_current_combination_under40=max(per_disk) < CAPACITY,
                max_A_single_disk_MB=max_a_bytes / 1e6,
                max_A_single_disk_service_ms=max_a_bytes / 1e6 / CAPACITY)


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--horizon-ms', type=float, default=12000)
    ap.add_argument('--skip-ring', action='store_true')
    args = ap.parse_args()
    pp = profiles()
    by_key = {tuple(p['source_key']): p for p in pp}
    all_pairs = [pair_screen(a, b) for a in pp for b in pp
                 if a['V_bytes'] > b['V_bytes'] and a['C_ms'] > b['C_ms']]
    all_pairs.sort(key=lambda r: r['hypothetical_fleet_wait_scale'], reverse=True)
    # Diversified shortlist, with one deliberately near-capacity candidate
    # whose finite RingHash proof is expected to fail at its ideal minimum S.
    shortlist = [
        ((200, 4096), (32, 4096), 1, 3),
        ((200, 2048), (32, 2048), 2, 3),
        ((200, 1024), (32, 1024), 4, 3),
        ((192, 4096), (32, 4096), 1, 3),
        ((176, 4096), (32, 4096), 1, 3),
        ((200, 4096), (48, 4096), 1, 2),
        ((200, 2048), (32, 1024), 3, 6),
        ((200, 1024), (32, 512), 5, 7),
        ((200, 4096), (32, 2048), 2, 6),
        ((200, 4096), (32, 1024), 3, 12),
    ]
    selected = []
    for index, (ak, bk, s, q) in enumerate(shortlist, 1):
        a, b = by_key[ak], by_key[bk]
        row = pair_screen(a, b)
        row.update(candidate=index, A=a, B=b, suggested_S=s, suggested_ratio=[1, q])
        if not args.skip_ring:
            row['static_ring_checks'] = [static_ring_screen(a, b, s, q, seed, args.horizon_ms)
                                         for seed in SEEDS]
            row['all_three_seeds_certified'] = all(r['any_current_combination_under40']
                                                    for r in row['static_ring_checks'])
        selected.append(row)
        print(f"candidate {index}: A{ak} B{bk}, S={s}, 1:{q}, "
              f"ring_certified={row.get('all_three_seeds_certified')}", flush=True)
    maximum_x = max(a['V_bytes'] / b['V_bytes'] for a in pp for b in pp)
    assert maximum_x == 7.140625 < N
    report = dict(data_sha256=hashlib.sha256((REPO / 'data').read_bytes()).hexdigest(),
                  num_profiles=len(pp), num_eligible_pairs=len(all_pairs), num_npu=N,
                  units='decimal GB/s, MB, ms', maximum_raw_volume_ratio=maximum_x,
                  single_A_any_16_mix_underload_possible=False,
                  proof='D > 16*max(BA,BB) >= 16*BB, while single-A blocking requires D < VA/CB = x*BB; x <= 7.140625 < 16.',
                  warning='One-A proof is for ideal shared capacity, not arbitrary placement, multiple A, partial compute slack, or finite FIFO interactions.',
                  shortlist=selected)
    HERE.mkdir(parents=True, exist_ok=True)
    save_csv(HERE / 'screen_raw_profiles.csv', pp)
    save_csv(HERE / 'screen_all_pairs.csv', all_pairs)
    (HERE / 'screen_candidates.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Screened {len(pp)} original profiles and {len(all_pairs)} ordered A/B pairs.")


if __name__ == '__main__':
    main()
