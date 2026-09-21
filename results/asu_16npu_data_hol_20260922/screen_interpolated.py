#!/usr/bin/env python3
"""Four near-capacity 16-NPU/1-SSU pairs, interpolating only miss in data.

This file is separate from the original-row screen. Interpolated compute
times are model estimates, not independently measured data rows. All lengths
are existing 32K or 200K data lengths and all miss values are within the grid.
No simulator, historical result, input config, or runner is modified.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / 'data'


def make_profile(raw, length_k, miss, data_sha):
    misses = sorted(m for length, m in raw if length == length_k)
    assert misses[0] <= miss <= misses[-1]
    lo = max(m for m in misses if m <= miss)
    hi = min(m for m in misses if m >= miss)
    anchors = [(lo, 1.0)] if lo == hi else [(lo, (hi - miss) / (hi - lo)),
                                          (hi, (miss - lo) / (hi - lo))]
    compute_us = sum(raw[length_k, m][1] * weight for m, weight in anchors)
    total = length_k * 1024
    hit = total - miss
    volume_bytes = hit * 1408
    constructed = lo != hi
    return dict(total_tokens=total, total_length_k=length_k, nql=miss,
                ssd_prefix_tokens=hit, compute_us=compute_us,
                read_gib=volume_bytes / 2**30,
                B_gib_s=volume_bytes / 2**30 / (compute_us / 1e6),
                constructed_profile=constructed,
                profile_construction=dict(
                    method='linear_interpolation_miss_same_length' if constructed else 'original_data_row',
                    source='data', data_sha256=data_sha, extrapolated=False,
                    total_length_interpolated=False, compute_scale=1.0,
                    kv_formula='exact hit prefix tokens * 1408 bytes; no block padding',
                    interpretation='C is an interpolation estimate; V follows the unchanged simulator KV byte model.',
                    anchors=[dict(seq_len_k=length_k, nql=m, weight=weight,
                                  compute_us=raw[length_k, m][1],
                                  original_data_row=list(raw[length_k, m]))
                             for m, weight in anchors]),
                display=dict(C_ms=compute_us / 1000, V_bytes=volume_bytes,
                             V_MB=volume_bytes / 1e6,
                             B_GB_s=volume_bytes / (compute_us / 1e6) / 1e9))


def main():
    raw = ast.literal_eval(DATA.read_text())
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    # I1 allows a slightly smaller B miss than the initial approximate range
    # to put BOTH profiles just below the requested 2.49 GB/s threshold.
    specifications = [
        ('I1_both_near_cap', 3302, 2448, 4, 7),
        ('I2_small_margin', 3328, 2500, 4, 7),
        ('I3_more_margin', 3400, 2800, 4, 6),
        ('I4_raw_A_interpolated_B', 4096, 2500, 5, 8),
    ]
    rows = []
    for name, miss_a, miss_b, q10, q8 in specifications:
        a, b = make_profile(raw, 200, miss_a, sha), make_profile(raw, 32, miss_b, sha)
        ad, bd = a['display'], b['display']
        assert max(ad['B_GB_s'], bd['B_GB_s']) <= 2.49
        assert 16 * max(ad['B_GB_s'], bd['B_GB_s']) < 40
        service3 = 3 * ad['V_MB'] / 40
        assert service3 > bd['C_ms']
        mixes = []
        for q, target in ((q10, 'about 10 A / 6 B cards without stalls'),
                          (q8, 'about 8 A / 8 B cards without stalls')):
            n_a = 16 * ad['C_ms'] / (ad['C_ms'] + q * bd['C_ms'])
            n_b = 16 - n_a
            demand = n_a * ad['B_GB_s'] + n_b * bd['B_GB_s']
            mixes.append(dict(request_count_ratio=[1, q], target=target,
                              no_wait_mean_A_cards=n_a, no_wait_mean_B_cards=n_b,
                              no_wait_mean_demand_GB_s=demand,
                              caution='Time mix ignores waiting; actual concurrency is policy dependent.'))
        rows.append(dict(
            name=name, num_npu=16, ssu=1, disk_GB_s=40,
            profile_A=a, profile_B=b,
            x=ad['V_MB'] / bd['V_MB'], y=ad['C_ms'] / bd['C_ms'],
            all_combinations_demand_upper_GB_s=16 * max(ad['B_GB_s'], bd['B_GB_s']),
            all_combinations_headroom_GB_s=40 - 16 * max(ad['B_GB_s'], bd['B_GB_s']),
            all_combinations_underload=True,
            one_A_disk_service_ms=ad['V_MB'] / 40,
            three_A_disk_service_ms=service3,
            three_A_excess_over_B_compute_ms=service3 - bd['C_ms'],
            three_full_A_reads_ahead_sufficient_to_exceed_full_B_compute=True,
            B_8_layer_compute_ms=8 * bd['C_ms'],
            B_slo_1p5_extra_wait_budget_ms=4 * bd['C_ms'],
            suggested_mixes=mixes,
            warning='Three full A reads must really precede B. This is a sufficient queue configuration, not proof that random inputs sustain it.',
        ))
    report = dict(
        data_sha256=sha, num_original_data_rows=len(raw), units='decimal GB/s, MB, ms',
        nominal_underload_definition='Sum of current admitted requests V/C; extra next-request L0 prefetch excluded by historical convention.',
        construction='Only miss-axis interpolation between existing data rows at the same total length; no extrapolation and no C scaling.',
        scope='This is an additional modeled-profile experiment, distinct from the all-original-row experiment.',
        shortlist=rows,
    )
    (HERE / 'screen_interpolated.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# 16 NPU、1 SSU：只在 data 网格内插值 miss 的候选', '',
        '这四组与直接使用原始行的实验分开记录。总长度固定为 data 已有的 200K、32K；只在相同长度的 miss=2048 和 4096 两行之间，按权重线性估计计算时间。读取量仍按 `(总token−miss)×1408` 精确计算。没有长度外推、计算缩放或读取补块。', '',
        '**这些 C 是插值估计，不是 data 中独立测量的原始行。** I4 的 A 是原始 200K/4096 行，只有 B 插值。全部锚点、权重和原始行保存在 JSON。', '',
        '| 候选 | A miss | A C(ms) | A V(MB) | A B(GB/s) | B miss | B C(ms) | B V(MB) | B B(GB/s) | 16卡任意组合需求上界 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        a, b = r['profile_A'], r['profile_B']
        x, y = a['display'], b['display']
        lines.append(f"| {r['name']} | {a['nql']} | {x['C_ms']:.6f} | {x['V_MB']:.6f} | {x['B_GB_s']:.6f} | {b['nql']} | {y['C_ms']:.6f} | {y['V_MB']:.6f} | {y['B_GB_s']:.6f} | {r['all_combinations_demand_upper_GB_s']:.6f} |")
    lines.extend(['', '每盘容量为 40 GB/s；这里只有一个盘，所以任意 16 张卡的 A/B 组合都欠载，不需要平均分盘假设，也不依赖队列长度、随机种子或策略的推进顺序。I1 的 B miss=2448 略低于最初约 2500 的搜索起点，目的是让 A、B 两者都接近但不超过 2.49 GB/s。', '',
                  '| 候选 | 3个A的盘服务(ms) | 超过B计算(ms) | 约10A/6B的请求数比 | 实际无等待近似A卡数 | 约8A/8B的请求数比 | B的SLO额外等待预算(ms) |',
                  '|---|---:|---:|---|---:|---|---:|'])
    for r in rows:
        m10, m8 = r['suggested_mixes']
        lines.append(f"| {r['name']} | {r['three_A_disk_service_ms']:.6f} | {r['three_A_excess_over_B_compute_ms']:.6f} | 1:{m10['request_count_ratio'][1]} | {m10['no_wait_mean_A_cards']:.3f} | 1:{m8['request_count_ratio'][1]} | {r['B_slo_1p5_extra_wait_budget_ms']:.6f} |")
    lines.extend(['',
                  '为什么可以比原始 miss=4096 的 B 更容易等待：B 的每层计算由约 28.6ms 缩到约 17.1–19.6ms。原来要约5个A的读取排在B前面才超过这段计算，现在3个A就够了。同时每张卡最高需求仍不到2.49GB/s，16张卡合计仍低于40GB/s。', '',
                  '但“3个A就够”仍以其完整读取确实排在B前面为前提。随机输入是否长期聚集出这种队列、自然错开是否消除它、Once是否改善，都必须看仿真。这里不预测利用率，也不保证TTFT违约。', '',
                  '比例来自 `A时间占比=C_A/(C_A+q*C_B)`，其中请求数 A:B=1:q。出现等待后，实际时间占比会改变。SLO×1.5的额外等待预算为 `0.5*8*C_B`，仍需累计请求各层等待进行判断。', '',
                  '欠载沿用普通名义需求口径：当前请求从上卡到完成计入V/C，跨请求额外预取下个L0不另加需求。额外预取产生的实际等待没有从利用率或TTFT中删除。', '',
                  f'原始 data SHA256：`{sha}`。', '',
                  '复算：`python results/asu_16npu_data_hol_20260922/screen_interpolated.py`。', '',
                  '[完整候选和锚点](screen_interpolated.json) · [程序](screen_interpolated.py)', ''])
    (HERE / 'screen_interpolated.md').write_text('\n'.join(lines))
    for r in rows:
        print(json.dumps({k: r[k] for k in ('name', 'all_combinations_demand_upper_GB_s',
                                           'three_A_disk_service_ms', 'three_A_excess_over_B_compute_ms')}))


if __name__ == '__main__':
    main()
