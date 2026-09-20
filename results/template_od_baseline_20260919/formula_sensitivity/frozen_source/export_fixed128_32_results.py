#!/usr/bin/env python3
"""Export frozen 8-NPU fixed-128K/32K cases; no simulation or replay.

Example: python export_fixed128_32_results.py --case CASE_FIFO --case CASE_ONCE
         --out OUTPUT --font /absolute/path/chinese.ttf

The experiment's warm window is [2000,4000) ms. Per-NPU blue curves retain
the complete internal-layer-cycle definition. Physical SSD curves exclusively
use common 2-ms service-byte bins. CDF comparisons are grouped by exact input
fingerprint, never across different input configurations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

import plot_fifo_mixed8 as old

LEFT, RIGHT = 2000., 4000.
INK, MUTED = '#17324d', '#596c80'
BLUE, PURPLE, ORANGE = '#086bd9', '#a32b91', '#d97b12'
POLICY = {'fifo': 'Baseline FIFO', 'once': 'Once per layer (5 ms)',
          'short_first': '短读取优先（诊断）'}
COLORS = {'fifo': '#3977b4', 'once': '#cf7820', 'short_first': '#278953'}


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def clipped(start, end):
    return max(0., min(end, RIGHT)-max(start, LEFT))


def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-7), (x, y)


def add_to_bins(row, edges, start, end, rate):
    start, end = max(start, edges[0]), min(end, edges[-1])
    if start >= end:
        return
    i = min(len(row)-1, int(np.searchsorted(edges, start, side='right')-1))
    while start < end and i < len(row):
        stop = min(end, edges[i+1])
        row[i] += (stop-start)*rate/(edges[i+1]-edges[i])
        start = stop
        i += 1


def volume_vector(manifest, request, layer, num_ssu):
    placement = manifest['placements'][request['placement_index']]
    pairs = placement[0 if len(placement) == 1 else layer]
    vector = [0.]*num_ssu
    for disk, size in pairs:
        vector[int(disk)] += float(size)
    close(sum(vector), request['load']['per_layer_kv_gb'])
    return vector


def request_rows(case):
    rows = []
    requests = {q['request_id']: q for q in case['manifest']['requests']}
    for b in case['raw']['summary']['microbatch_metrics']:
        rid = b['member_request_ids'][0]
        q = requests[rid]['load']
        layers = sorted(b['layer_metrics'], key=lambda x: x['layer'])
        admission, completion = b['admission_time_ms'], b['completion_time_ms']
        ttft = completion-admission
        ideal = sum(l['compute_end_ms']-l['compute_start_ms'] for l in layers)
        close(ideal, len(layers)*q['per_layer_us']/1000)
        active = clipped(admission, completion)
        compute = sum(clipped(l['compute_start_ms'], l['compute_end_ms']) for l in layers)
        row = dict(configuration=case['group'], policy=case['policy'], npu_id=b['npu_id'],
            request_id=rid, input_order_1based=q['generation']+1, role=q['role'],
            qos_category=q['category'], total_tokens=q['total_tokens'],
            total_length_k=q['total_tokens']/1024, nql_tokens=q['nql'],
            hit_prefix_tokens=q['ssd_prefix_tokens'], per_layer_read_gib=q['per_layer_kv_gb'],
            per_layer_read_mib=q['per_layer_kv_gb']*1024,
            per_layer_compute_ms=q['per_layer_us']/1000,
            bandwidth_demand_gib_s=q['per_layer_kv_gb']*1e6/q['per_layer_us'],
            n_layers=len(layers), num_ssu=case['num_ssu'],
            arrival_ms=q.get('arrival_ms', 0.), admission_ms=admission, completion_ms=completion,
            ttft_admission_ms=ttft, ttft_arrival_ms=completion-q.get('arrival_ms', 0.),
            ideal_compute_ms=ideal, slo_1p5_ms=1.5*ideal,
            ttft_over_ideal=ttft/ideal, slo_1p5_passed=ttft <= 1.5*ideal+1e-9,
            admitted_in_warm=LEFT <= admission < RIGHT,
            active_in_warm=active > 0, computed_in_warm=compute > 0,
            full_stall_ms=max(0., ttft-ideal), full_active_utilization_percent=100*ideal/ttft,
            warm_active_ms=active, warm_compute_ms=compute,
            warm_stall_ms=max(0., active-compute),
            warm_active_utilization_percent=100*compute/active if active else None,
            compute_method=q.get('profile_construction', {}).get('method', 'unknown'),
            compute_extrapolated=q.get('profile_construction', {}).get('extrapolated', False),
            compute_scale=1., input_fingerprint=case['fingerprint'])
        rows.append(row)
    return sorted(rows, key=lambda x: (x['npu_id'], x['input_order_1based']))


def aggregate_rows(case, rows):
    summaries = []
    for npu in [None, *range(8)]:
        for role in ['all', 'S', 'L']:
            pool = [r for r in rows if (npu is None or r['npu_id'] == npu) and (role == 'all' or r['role'] == role)]
            for cohort in ['warm_admissions', 'all_requests']:
                sample = [r for r in pool if cohort == 'all_requests' or r['admitted_in_warm']]
                passed = sum(r['slo_1p5_passed'] for r in sample)
                active = sum(r['warm_active_ms'] if cohort == 'warm_admissions' else r['ttft_admission_ms'] for r in pool)
                compute = sum(r['warm_compute_ms'] if cohort == 'warm_admissions' else r['ideal_compute_ms'] for r in pool)
                summaries.append(dict(configuration=case['group'], policy=case['policy'],
                    npu_id='all' if npu is None else npu, role=role, cohort=cohort,
                    slo_passed=passed, slo_count=len(sample),
                    slo_1p5_percent=100*passed/len(sample) if sample else None,
                    utilization_scope='all_active_intervals_clipped_to_warm' if cohort == 'warm_admissions' else 'all_request_active_intervals',
                    active_ms=active, compute_ms=compute, stall_ms=max(0., active-compute),
                    active_time_utilization_percent=100*compute/active if active else None))
    return summaries


def physical_data(case):
    receipt, manifest = case['receipt'], case['manifest']
    assert all(receipt['observer_checks'].values())
    edges = np.asarray(receipt['bin_edges_ms'], dtype=float)
    assert [edges[0], edges[-1]] == [LEFT, RIGHT]
    widths = np.diff(edges)
    assert np.allclose(widths, 2., atol=1e-10)
    num_ssu, bins = case['num_ssu'], len(widths)
    physical = np.asarray(receipt['ssu_bin_read_gib'])*1000/widths
    assert physical.shape == (num_ssu, bins)
    assert np.max(physical) <= 40+1e-7
    nominal, deadline = np.zeros_like(physical), np.zeros_like(physical)
    stalled = np.zeros(bins)
    req = {q['request_id']: q for q in manifest['requests']}
    jobs = []
    for npu in range(8):
        batches = sorted((b for b in case['raw']['summary']['microbatch_metrics'] if b['npu_id'] == npu), key=lambda b: b['admission_time_ms'])
        flat = []
        for b in batches:
            rid = b['member_request_ids'][0]
            q = req[rid]['load']
            vector = volume_vector(manifest, req[rid], 0, num_ssu)
            for disk, v in enumerate(vector):
                add_to_bins(nominal[disk], edges, b['admission_time_ms'], b['completion_time_ms'], v*1e6/q['per_layer_us'])
            previous = b['admission_time_ms']
            for layer in sorted(b['layer_metrics'], key=lambda l: l['layer']):
                flat.append(dict(request_id=rid, **layer))
                add_to_bins(stalled, edges, previous, layer['compute_start_ms'], 1.)
                previous = layer['compute_end_ms']
        for current, following in zip(flat, flat[1:]):
            start, end = current['compute_start_ms'], current['compute_end_ms']
            close(following['io_start_time_ms'], start)
            C = end-start
            vector = volume_vector(manifest, req[following['request_id']], following['layer'], num_ssu)
            for disk, v in enumerate(vector):
                add_to_bins(deadline[disk], edges, start, end, v*1000/C)
                if clipped(start, end):
                    jobs.append(dict(npu_id=npu, ssu_id=disk, current_request_id=current['request_id'],
                        current_layer=current['layer'], next_request_id=following['request_id'],
                        next_layer=following['layer'], cross_request=current['request_id'] != following['request_id'],
                        release_ms=start, deadline_ms=end, available_compute_ms=C, next_read_gib=v,
                        deadline_reference_gib_s=v*1000/C,
                        actual_next_io_ready_ms=following['io_ready_time_ms'],
                        actual_stall_ms=max(0., following['compute_start_ms']-end),
                        impossible_even_on_exclusive_ssu=v > 40*C/1000+1e-12))
    close(sum(stalled*widths), (100-case['metrics']['U_percent'])/100*8*(RIGHT-LEFT))
    rows = []
    for i in range(bins):
        row = dict(start_ms=float(edges[i]), end_ms=float(edges[i+1]),
            fleet_actual_ssd_gib_s=float(sum(physical[:, i])),
            fleet_nominal_demand_gib_s=float(sum(nominal[:, i])),
            fleet_deadline_reference_gib_s=float(sum(deadline[:, i])),
            mean_io_stalled_npus=float(stalled[i]))
        for disk in range(num_ssu):
            row.update({f'ssu{disk}_actual_gib_s': float(physical[disk, i]),
                f'ssu{disk}_nominal_gib_s': float(nominal[disk, i]),
                f'ssu{disk}_deadline_reference_gib_s': float(deadline[disk, i])})
        rows.append(row)
    audit = dict(physical_ssu_mean_gib_s=[float(sum(x*widths)/(RIGHT-LEFT)) for x in physical],
        physical_ssu_max_gib_s=[float(max(x)) for x in physical],
        physical_fleet_mean_gib_s=float(sum(np.sum(physical, axis=0)*widths)/(RIGHT-LEFT)),
        physical_fleet_max_gib_s=float(max(np.sum(physical, axis=0))),
        physical_capacity_violation_bins=int(np.sum(physical > 40+1e-7)),
        bin_ms=2., no_rate_clipping=True,
        nominal_demand_static_upper_gib_s=case['metadata']['per_ssu_static_upper_bound_gib_s'],
        deadline_reference_is_not_a_strict_feasibility_test=True)
    return dict(edges=edges, physical=physical, nominal=nominal, deadline=deadline,
        stalled=stalled, jobs=jobs, rows=rows, audit=audit)


def draw_npu_bandwidth(data, path, ymax):
    fig, axes = plt.subplots(8, 1, figsize=(18, 12.5), dpi=150, sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.81, top=.765, bottom=.155, hspace=.39)
    old.add_header(fig, data, '每层平均接收带宽与需求（全部8张卡）')
    handles = [Line2D([], [], color=old.PURPLE, lw=2.3, ls='--', label='需求：当前请求每层 V / C'),
        Line2D([], [], color=old.BLUE, lw=2.2, marker='o', markerfacecolor='white', label='层周期平均接收：V / (C + stall)'),
        Patch(facecolor=old.GRAY, label='灰区：跨请求 / 窗口截断')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.031, .855), frameon=False, ncol=3, fontsize=11.5)
    fig.text(.035, .794, f'完整同请求周期的蓝线不超过紫线；所有图纵轴统一 0-{ymax:g} GiB/s。', fontsize=12, color=MUTED)
    fig.text(.83, .779, '整窗均值（GiB/s）', fontsize=11, color=INK)
    for npu, ax in enumerate(axes):
        lane = data['lanes'][npu]
        old.bandwidth(ax, lane, ymax)
        ax.set_ylabel(f'NPU {npu}\nU={lane["U_percent"]:.2f}%', fontsize=11, rotation=0, ha='right', va='center', labelpad=14, color=INK)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=10, labelbottom=npu == 7)
        ax.text(1.028, .76, f'需求 {lane["mean_demand_GiB_s"]:.3f}', transform=ax.transAxes, fontsize=11, color=PURPLE)
        ax.text(1.028, .37, f'实际接收 {lane["mean_supply_GiB_s"]:.3f}', transform=ax.transAxes, fontsize=11, color=BLUE)
        ax.text(1.028, -.01, '（含预取）', transform=ax.transAxes, fontsize=9, color=MUTED)
    axes[-1].set_xlabel('仿真时间（秒）；带宽单位 GiB/s', fontsize=13, labelpad=9)
    fig.text(.035, .092, '层周期 = 当前层开始计算 → 下一层开始计算；蓝线只表示该周期接收的下一层数据量 / 周期时长。', fontsize=11.5, color=INK)
    fig.text(.035, .066, '右侧实际接收统计全部2秒字节，包含灰区和跨边界预取；它可能高于需求均值，两者相除不等于利用率。', fontsize=11.5, color=MUTED)
    fig.text(.035, .040, old.underload_note(data), fontsize=11.5, color=MUTED)
    fig.text(.035, .016, '实际盘吞吐请查看 physical_ssd_bandwidth 图；不要把各NPU不同层周期平均值相加当作物理吞吐。', fontsize=11, color=MUTED)
    return old.save_checked(fig, path)


def draw_physical(case, values, path):
    num_ssu = case['num_ssu']
    count = 3 if num_ssu == 1 else num_ssu+3
    fig, axes = plt.subplots(count, 1, figsize=(17, 3.05*count+2.8), sharex=True)
    fig.subplots_adjust(left=.085, right=.96, top=.88, bottom=.115, hspace=.66)
    fig.text(.06, .96, f'{POLICY[case["policy"]]}：真实盘吞吐、常规需求与预取需求', fontsize=23, color=INK)
    fig.text(.06, .927, f'8 NPU / {num_ssu} SSU × 40 GiB/s · 固定128K/32K · 每卡随机混合 · 统一2 ms窗口 · U={case["metrics"]["U_percent"]:.2f}%', fontsize=13, color=MUTED)
    x = values['edges']/1000
    def supply_panel(ax, actual, nominal, capacity, title):
        ax.stairs(actual, x, color=BLUE, lw=1.3, label='实际盘服务字节 / 2 ms')
        ax.stairs(nominal, x, color=PURPLE, lw=1.6, label='当前请求常规 V/C（同窗均值）')
        ax.axhline(capacity, color='#333', lw=1.1, ls='--', label=f'容量 {capacity:g} GiB/s')
        ax.set_ylim(0, max(capacity*1.13, float(max(nominal))*1.13))
        ax.set_title(title, loc='left', fontsize=14, pad=10)
        ax.legend(loc='upper left', ncol=3, fontsize=10, frameon=True, facecolor='white', framealpha=.95)
        ax.set_ylabel('GiB/s')
    supply_panel(axes[0], np.sum(values['physical'], axis=0), np.sum(values['nominal'], axis=0), 40*num_ssu,
        f'整机物理带宽：均值 {values["audit"]["physical_fleet_mean_gib_s"]:.3f}；峰值 {values["audit"]["physical_fleet_max_gib_s"]:.3f} GiB/s')
    index = 1
    if num_ssu > 1:
        for disk in range(num_ssu):
            supply_panel(axes[index], values['physical'][disk], values['nominal'][disk], 40,
                f'SSU {disk}：真实盘吞吐与分配到该盘的常规需求')
            index += 1
    ax = axes[index]
    disk_colors = ['#d97b12', '#bc354e', '#743c9f', '#488577']
    for disk in range(num_ssu):
        ax.stairs(values['deadline'][disk], x, color=disk_colors[disk % len(disk_colors)], lw=1.2,
            label=f'SSU {disk}：下一层读取量 / 当前层计算窗口')
    ax.axhline(40, color='#333', lw=1.1, ls='--', label='单盘容量 40 GiB/s')
    ax.set_ylim(0, max(45., float(np.max(values['deadline']))*1.2))
    ax.set_title('实际执行顺序对应的预取截止参考需求：包括短→长首层；仅在当前层计算窗口累计', loc='left', fontsize=14, pad=10)
    ax.set_ylabel('GiB/s')
    ax.legend(loc='upper left', ncol=min(3, num_ssu+1), fontsize=10,
        frameon=True, facecolor='white', framealpha=.97)
    ax = axes[-1]
    ax.stairs(values['stalled'], x, color='#b87500', fill=True, alpha=.65)
    ax.set_ylim(0, 8.4)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_ylabel('NPU 数')
    ax.set_title('实际 IO Stall：每个2 ms内等待IO的平均NPU数', loc='left', fontsize=14, pad=10)
    ax.set_xlabel('仿真时间（秒）')
    for ax in axes:
        ax.set_xlim(2, 4)
        ax.set_xticks(np.arange(2, 4.01, .25))
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.06, .065, '蓝线来自所有IO共同时间窗内的实际SSD服务量，未做限幅；每盘始终不超过40 GiB/s。', fontsize=12, color=INK)
    fig.text(.06, .034, '预取需求超过40表示匀速参考超载，单凭曲线越线不能证明截止时间不可满足；当前请求常规V/C欠载也不保证没有预取争用。', fontsize=11.5, color=MUTED)
    return old.save_checked(fig, path)


def draw_cdf(group, cases, cohort, path, point_rows):
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150)
    fig.subplots_adjust(left=.10, right=.96, bottom=.20, top=.78)
    label = 'warm [2,4)秒内接纳请求' if cohort == 'warm_admissions' else '相同的全部输入请求'
    fig.text(.065, .93, f'TTFT SLO × 1.5：{label}', fontsize=23, color=INK)
    fig.text(.065, .88, f'8 NPU / {cases[0]["num_ssu"]} SSU · '+old.short_description(cases[0]['analysed']), fontsize=10.6, color=MUTED)
    all_values = [r['ttft_over_ideal'] for c in cases for r in c['rows']
        if cohort == 'all_requests' or r['admitted_in_warm']]
    xmax = max(1.65, max(all_values)*1.025)
    for case in cases:
        rows = [r for r in case['rows'] if cohort == 'all_requests' or r['admitted_in_warm']]
        values = sorted(r['ttft_over_ideal'] for r in rows)
        assert values
        unique, counts = np.unique(values, return_counts=True)
        pct = np.cumsum(counts)*100/len(values)
        passed = sum(r['slo_1p5_passed'] for r in rows)
        ax.step(np.r_[.98, unique, xmax], np.r_[0., pct, 100.], where='post', color=COLORS[case['policy']], lw=2.,
            label=f'{POLICY[case["policy"]]}：{passed}/{len(values)} = {100*passed/len(values):.2f}%')
        for v, y in zip(unique, pct):
            point_rows.append(dict(configuration=group, policy=case['policy'], cohort=cohort,
                ttft_over_ideal=float(v), cumulative_percent=float(y), request_count=len(values)))
    ax.axvline(1.5, color='#555', ls='--', lw=1.5, label='SLO阈值 1.5')
    tick_step = .1 if xmax < 2 else .25 if xmax < 3 else .5
    ax.set(xlim=(.98, xmax), ylim=(0, 102), xticks=np.arange(1., xmax+1e-9, tick_step),
        xlabel='TTFT / 纯计算时间（8层计算时间之和）', ylabel='累计请求比例（%）')
    ax.grid(alpha=.17)
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.065, .115, 'TTFT = 完成时间 - NPU接纳时间；不含接纳前的输入队列等待。达标条件：TTFT ≤ 1.5 × 纯计算时间。', fontsize=11.5, color=INK)
    fig.text(.065, .072, 'warm内接纳集合可能随策略变化；全部输入图使用完全相同请求ID，便于对齐比较。', fontsize=11.5, color=MUTED)
    return old.save_checked(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    for directory in ['images', 'data']:
        (args.out/directory).mkdir(parents=True, exist_ok=True)
    fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': FontProperties(fname=str(args.font)).get_name(),
        'axes.unicode_minus': False, 'font.size': 12, 'figure.facecolor': 'white'})
    cases, groups = [], defaultdict(list)
    for folder in args.case:
        folder = folder.resolve()
        analysed = old.analyse(folder)
        manifest, metadata = read(folder/'manifest.json.gz'), read(folder/'metadata.json')
        fingerprint = metadata['input_fingerprint']
        group = f's{metadata["num_ssu"]}_{fingerprint[:12]}'
        case = dict(folder=folder, analysed=analysed, manifest=manifest, metadata=metadata,
            metrics=read(folder/'metrics.json'), receipt=read(folder/'receipts.json'),
            raw=read(folder/'result.json.gz'), group=group, fingerprint=fingerprint,
            num_ssu=metadata['num_ssu'])
        case['policy'] = case['metrics']['policy']
        case['stem'] = f'{group}_{case["policy"]}'
        case['rows'] = request_rows(case)
        by_class = aggregate_rows(case, case['rows'])
        class_u = {(r['npu_id'], r['role'], r['cohort']): r['active_time_utilization_percent'] for r in by_class}
        for row in case['rows']:
            row['warm_class_utilization_percent'] = class_u['all', row['role'], 'warm_admissions']
            row['full_class_utilization_percent'] = class_u['all', row['role'], 'all_requests']
            row['warm_npu_class_utilization_percent'] = class_u[row['npu_id'], row['role'], 'warm_admissions']
        assert {r['total_tokens'] for r in case['rows']} == {128*1024, 32*1024}
        cases.append(case)
        groups[group].append(case)
    assert len({c['stem'] for c in cases}) == len(cases), 'Duplicate policy/input case'
    ymax = float(math.ceil(max(r['bandwidth_demand_gib_s'] for c in cases for r in c['rows'])*1.05))
    all_rows, class_rows, figure_audits, scenarios, points = [], [], [], [], []
    for case in cases:
        stem = case['stem']
        values = physical_data(case)
        figure_audits.extend([draw_npu_bandwidth(case['analysed'], args.out/'images'/f'{stem}_8npu_bandwidth.png', ymax),
            old.draw_timeline(case['analysed'], args.out/'images'/f'{stem}_8npu_timeline.png'),
            draw_physical(case, values, args.out/'images'/f'{stem}_physical_ssd_bandwidth.png')])
        summaries = aggregate_rows(case, case['rows'])
        write_csv(args.out/'data'/f'{stem}_requests.csv', case['rows'])
        write_csv(args.out/'data'/f'{stem}_layer_cycles.csv', case['analysed']['cycles'])
        write_csv(args.out/'data'/f'{stem}_per_npu_class.csv', case['analysed']['class_rows'])
        write_csv(args.out/'data'/f'{stem}_physical_2ms.csv', values['rows'])
        write_csv(args.out/'data'/f'{stem}_prefetch_deadline_jobs.csv', values['jobs'])
        write_json(args.out/'data'/f'{stem}_physical_audit.json', values['audit'])
        all_rows.extend(case['rows'])
        class_rows.extend(summaries)
        full = next(r for r in summaries if r['npu_id'] == 'all' and r['role'] == 'all' and r['cohort'] == 'all_requests')
        scenarios.append(dict(configuration=case['group'], policy=case['policy'], case_directory=str(case['folder']),
            num_ssu=case['num_ssu'], short_per_long=case['metadata']['case']['short_per_long'],
            input_fingerprint=case['fingerprint'], U_percent=case['metrics']['U_percent'],
            short_U_percent=case['metrics']['short_U_percent'], long_U_percent=case['metrics']['long_U_percent'],
            warm_slo_percent=case['metrics']['slo_1p5_percent'],
            full_slo_percent=full['slo_1p5_percent'], request_count=len(case['rows']),
            source_hashes=case['analysed']['source_hashes'], **values['audit']))
    for group, items in groups.items():
        ids = [{r['request_id'] for r in c['rows']} for c in items]
        assert all(x == ids[0] for x in ids), 'Full CDF request sets differ'
        for cohort in ['warm_admissions', 'all_requests']:
            figure_audits.append(draw_cdf(group, items, cohort,
                args.out/'images'/f'{group}_ttft_slo15_{cohort}.png', points))
        profile_fields = ['configuration', 'npu_id', 'request_id', 'input_order_1based', 'role', 'qos_category',
            'total_tokens', 'total_length_k', 'nql_tokens', 'hit_prefix_tokens', 'per_layer_read_gib',
            'per_layer_read_mib', 'per_layer_compute_ms', 'bandwidth_demand_gib_s', 'compute_method', 'compute_extrapolated']
        write_csv(args.out/'data'/f'{group}_input_profiles.csv', [{k: row[k] for k in profile_fields} for row in items[0]['rows']])
    write_csv(args.out/'data'/'all_request_execution.csv', all_rows)
    write_csv(args.out/'data'/'slo_and_class_utilization.csv', class_rows)
    write_csv(args.out/'data'/'ttft_cdf_points.csv', points)
    write_json(args.out/'data'/'scenario_summary.json', scenarios)
    write_csv(args.out/'data'/'scenario_summary.csv', [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in scenarios])
    for info in figure_audits:
        image_path = args.out/'images'/info['file']
        with Image.open(image_path) as png:
            png.load()
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == info['sha256']
    write_json(args.out/'data'/'figure_audit.json', dict(figures=figure_audits, all_checks_passed=True,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_groups=list(groups), request_execution_rows=len(all_rows), no_simulation_executed=True))
    notes = ('# 固定128K/32K实验导出\n\n'
        '每卡混合长短请求，8 NPU；全部请求总长度精确等于128×1024或32×1024。\n\n'
        '- `*_requests.csv` / `all_request_execution.csv`：逐请求画像、输入顺序、时间、TTFT、SLO与等待。\n'
        '- `slo_and_class_utilization.csv`：全卡及每卡、所有/短/长类别的SLO与按active时间加权利用率。warm的SLO仅取窗口内接纳请求；warm利用率取所有与窗口相交的active区间，两者不是同一队列集合。\n'
        '- `*_physical_2ms.csv`：共同2ms时间窗的真实SSD服务量、按block实际盘映射计算的需求；包含不足176KiB的尾块。蓝线从不按NPU各自层周期相加，也没有限幅。\n'
        '- 逐NPU图蓝线为完整同请求层周期接收均值；右侧实际接收均值包含预取，可超过当前请求需求均值。\n'
        '- 预取截止参考需求是下一层数据量/当前层计算时间，只在计算窗口累计；它包含跨请求L0。曲线越容量不是不可调度的充分证明。\n'
        '- TTFT从NPU接纳计时，阈值为8层纯计算总时间×1.5；CDF按完全相同input_fingerprint分组。全部输入集合相同，warm接纳集合可不同。\n'
        '- 各策略不改变输入顺序，读取量按命中tokens×1408字节精确计算，计算时间仅使用data网格内插值。\n')
    (args.out/'README.md').write_text(notes, encoding='utf-8')
    print(json.dumps(dict(output=str(args.out.resolve()), cases=len(cases), input_groups=len(groups),
        figures=len(figure_audits), request_execution_rows=len(all_rows), all_checks_passed=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
