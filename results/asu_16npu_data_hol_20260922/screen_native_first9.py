#!/usr/bin/env python3
"""Independent standard-library audit of the first nine raw-data ASU runs.

Reads native intervals and the frozen manifest. Does not import the runner,
simulator, archived audit, or recorded demand-event output.
"""
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def overlap(a, b, lo, hi):
    return max(0.0, min(b, hi) - max(a, lo))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)


def audit_run(path):
    metrics = read(path / 'metrics.json')
    native = read(path / 'native_summary.json.gz')
    manifest = read(path / 'manifest.json.gz')
    config = read(path / 'config.json')
    loads = {r['request_id']: r['load'] for r in manifest['requests']}
    assert native['num_npu'] == config['num_npu'] == 16
    assert native['request_count'] == len(loads) == len(native['microbatch_metrics'])
    assert all(native['invariants'].values())
    raw = ast.literal_eval((REPO / 'data').read_text())
    for role in 'AB':
        p = config['profile_' + role]
        source = raw[p['total_length_k'], p['nql']]
        close(p['compute_us'], source[1])
        close(p['read_gib'], source[3])
        assert p['constructed_profile'] is False
    row = dict(name=config['name'], strategy=metrics['strategy'], seed=config['seed'],
               S=config['ssu'], B_per_A=config['input_counts'][1],
               A_miss=config['profile_A']['nql'], B_miss=config['profile_B']['nql'])
    per_npu = []
    for tag, lo, hi in [('warm', 2000.0, 4000.0), ('long', 2000.0, 10000.0),
                        ('full', 0.0, native['makespan_ms'])]:
        comp = [[0.0, 0.0] for _ in range(16)]
        active = [0.0] * 16
        admitted = []
        for batch in native['microbatch_metrics']:
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            load = loads[rid]
            npu = batch['npu_id']
            role = load['profile_group']
            start, finish = batch['admission_time_ms'], batch['completion_time_ms']
            active[npu] += overlap(start, finish, lo, hi)
            busy = 0.0
            for layer in batch['layer_metrics']:
                a, b = layer['compute_start_ms'], layer['compute_end_ms']
                close(b - a, load['per_layer_us'] / 1000)
                busy += b - a
                comp[npu]['AB'.index(role)] += overlap(a, b, lo, hi)
            close(busy, 8 * load['per_layer_us'] / 1000)
            if lo <= start < hi:
                admitted.append(finish - start <= 1.5 * busy + 1e-8)
        fleet = sum(sum(v) for v in comp) / (16 * (hi - lo)) * 100
        missing = [i for i, c in enumerate(comp) if min(c) <= 1e-8]
        row.update({f'{tag}_U_percent': fleet,
                    f'{tag}_SLO1p5_percent': 100 * sum(admitted) / len(admitted),
                    f'{tag}_mixed_cards': 16 - len(missing),
                    f'{tag}_missing_AB_cards': ','.join(map(str, missing)),
                    f'{tag}_all_cards_active': all(abs(v - (hi - lo)) < 1e-6 for v in active)})
        if tag == 'warm':
            close(fleet, metrics['warm_U_percent'])
            close(row[f'{tag}_SLO1p5_percent'], metrics['warm_SLO_1p5_percent'])
            assert (not missing) == metrics['warm']['all_npus_compute_both_groups']
        if tag == 'full':
            close(fleet, metrics['full_U_percent'])
            close(row[f'{tag}_SLO1p5_percent'], metrics['full_SLO_1p5_percent'])
        for i in range(16):
            per_npu.append(dict(name=config['name'], window=tag, npu=i,
                                A_compute_ms=comp[i][0], B_compute_ms=comp[i][1],
                                U_percent=sum(comp[i]) / (hi-lo) * 100,
                                active_ms=active[i], both_AB=min(comp[i]) > 1e-8))
    events = defaultdict(lambda: [0.0] * config['ssu'])
    for batch in native['microbatch_metrics']:
        load = loads[batch['member_request_ids'][0]]
        rate = [v * 2**30 / (load['per_layer_us'] / 1e6) / 1e9 for v in load['disk_gib']]
        close(sum(load['disk_gib']) * 2**30, load['ssd_prefix_tokens'] * 1408)
        for t, sign in ((batch['admission_time_ms'], 1), (batch['completion_time_ms'], -1)):
            for d, r in enumerate(rate):
                events[t][d] += sign * r
    points = sorted(events)
    current = [0.0] * config['ssu']
    peaks = [0.0] * config['ssu']
    total_area = [0.0] * config['ssu']
    over_ms = 0.0
    for index, t in enumerate(points):
        for d, delta in enumerate(events[t]):
            current[d] += delta
        if index + 1 == len(points):
            break
        duration = points[index + 1] - t
        for d, value in enumerate(current):
            peaks[d] = max(peaks[d], value)
            total_area[d] += value * duration
        if max(current) > 40 + 1e-8:
            over_ms += duration
    assert max(abs(x) for x in current) < 1e-7
    close(max(peaks), metrics['full']['ordinary_demand']['max_single_ssu_gb_s'])
    close(over_ms, metrics['full']['ordinary_demand']['any_ssu_overload_ms'])
    for actual, recorded in zip(peaks, metrics['full']['ordinary_demand']['peak_gb_s_by_ssu']):
        close(actual, recorded)
    row.update(full_peak_single_ssu_GB_s=max(peaks), full_any_ssu_overload_ms=over_ms,
               full_per_disk_peak_GB_s=json.dumps(peaks),
               full_per_disk_mean_GB_s=json.dumps([v/native['makespan_ms'] for v in total_area]),
               checks_pass=True)
    return row, per_npu


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_interpolated():
    raw = ast.literal_eval((REPO/'data').read_text())
    sha = hashlib.sha256((REPO/'data').read_bytes()).hexdigest()
    data = read(HERE/'screen_interpolated.json')
    assert data['data_sha256'] == sha
    cases = [('screen:' + row['name'], row) for row in data['shortlist']]
    for path in sorted((HERE/'configs').glob('*.json')):
        config = read(path)
        if any(config['profile_' + r].get('constructed_profile') for r in 'AB'):
            cases.append(('config:' + path.name, config))
    result = []
    for name, config in cases:
        for role in 'AB':
            p = config['profile_' + role]
            c = p['profile_construction']
            anchors = c.get('anchors')
            if anchors is None:
                close(p['compute_us'], raw[p['total_length_k'],p['nql']][1])
                continue
            assert c['data_sha256'] == sha
            close(sum(a['weight'] for a in anchors), 1.0)
            for a in anchors:
                assert a['seq_len_k'] == p['total_length_k']
                assert 0 <= a['weight'] <= 1
                close(a['compute_us'], raw[a['seq_len_k'],a['nql']][1])
            close(sum(a['weight']*a['nql'] for a in anchors), p['nql'])
            close(sum(a['weight']*a['compute_us'] for a in anchors), p['compute_us'])
            close(p['read_gib']*2**30, (p['total_tokens']-p['nql'])*1408)
            close(p['B_gib_s'], p['read_gib']/(p['compute_us']/1e6))
            result.append(dict(case=name, role=role, passed=True))
    return result


def main():
    rows, per_card = [], []
    # Fixed first-nine cohort: avoid silently including later research runs.
    paths = []
    for miss, disks in ((1024,4),(2048,2),(4096,1)):
        for q in (2,3,4):
            name = f'A200m{miss}_B32m{miss}_s{disks}_r1{q}_random_seed7'
            paths.append(HERE/'runs'/name/'asu_baseline')
    for path in paths:
        row, cards = audit_run(path)
        rows.append(row); per_card.extend(cards)
        print(row['name'], row['warm_U_percent'], row['long_U_percent'], flush=True)
    save_csv(HERE/'screen_native_first9.csv',rows)
    save_csv(HERE/'screen_native_first9_per_npu.csv',per_card)
    checks = verify_interpolated()
    (HERE/'screen_native_first9_checks.json').write_text(json.dumps(dict(
        raw_run_count=len(rows), all_native_checks_passed=True, interpolation_checks=checks),indent=2)+'\n')
    lines=['# 第一批九场原始画像 ASU：独立核验','',
           '只读取原生层计算区间、请求上卡/完成时间和冻结输入，使用标准库独立重算；未调用 runner、仿真器或原审计程序，未使用预先生成的需求事件。', '',
           '| miss(A/B相同) | SSU | 请求数A:B | [2,4) U | [2,10) U | [2,4)每卡AB | [2,10)每卡AB | 全程逐盘峰值GB/s | 全程过载ms |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['A_miss']} | {r['S']} | 1:{r['B_per_A']} | {r['warm_U_percent']:.4f}% | {r['long_U_percent']:.4f}% | {r['warm_mixed_cards']}/16 | {r['long_mixed_cards']}/16 | {r['full_peak_single_ssu_GB_s']:.6f} | {r['full_any_ssu_overload_ms']:.6f} |")
    lines += ['', '九场全部 seed=7、原始 200K A 与 32K B、16卡、8层；只改变共同 miss、盘数、请求数量比。它们是候选试验，不是跨种子均值。', '',
              'warm 利用率、full 利用率、warm/full SLO×1.5，以及全程逐盘需求峰值和过载时长，均与已存 metrics 一致。较长的 [2,10) 统计为本次独立追加计算。每卡AB表示该窗口中两个类别都具有正的真实计算时间。', '',
              '普通名义需求在当前请求上卡至完成区间计入 V_disk/C；额外跨请求 L0 预取不重复添加。这里的欠载仍不是无排队保证。', '',
              f'插值候选/当前插值配置共核验 {len(checks)} 个画像：权重非负且和为1，miss与C均由同总长原始锚点线性组合得到，V与精确命中token一致；所有检查通过。', '',
              '[汇总CSV](screen_native_first9.csv) · [逐卡CSV](screen_native_first9_per_npu.csv) · [检查记录](screen_native_first9_checks.json) · [复算程序](screen_native_first9.py)', '']
    (HERE/'screen_native_first9_notes.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
