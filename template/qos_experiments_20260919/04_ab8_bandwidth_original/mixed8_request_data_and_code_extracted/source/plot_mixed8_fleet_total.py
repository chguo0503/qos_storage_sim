#!/usr/bin/env python3
"""Exact frozen-result fleet bandwidth curves, including request handoffs.

No simulator execution. Each NPU's physical next-layer receipts are averaged
over compute-start -> next-compute-start periods. At the analysis-window edges,
both receipt bytes and period duration are clipped to the window. Summing these
per-NPU functions preserves the measured total received-byte area exactly.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'mixed8_fleet_total_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results/fifo_mixed_unique_20260914'
OUT = ROOT / 'results/mixed8_complete_figures_20260914'
LEFT, RIGHT, N = 2000.0, 4000.0, 8
PURPLE, BLUE, INK, MUTED = '#a32b91', '#0068d9', '#172d45', '#546980'
LABELS = {'fifo': 'Baseline Random（FIFO）', 'once': 'Once per layer Random（TTL=5 ms）',
          'short_first': '短读取优先 Random（诊断对照）'}


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as handle:
        return json.load(handle)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, message):
    assert math.isclose(a, b, abs_tol=1e-7, rel_tol=1e-9), (message, a, b)


def duration(a, b):
    return max(0.0, min(b, RIGHT) - max(a, LEFT))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyse(folder):
    raw, manifest, metrics, physical = [read(folder/name) for name in
        ('result.json.gz', 'manifest.json.gz', 'metrics.json', 'receipts.json')]
    assert metrics['num_npu'] == physical['num_npu'] == N
    assert metrics['num_ssu'] == physical['num_ssu'] == 1
    assert metrics['window_ms'] == physical['window_ms'] == [LEFT, RIGHT]
    assert all(physical['observer_checks'].values())
    assert physical['source_result_sha256'] == sha(folder/'result.json.gz')
    assert physical['source_manifest_sha256'] == sha(folder/'manifest.json.gz')
    assert all(raw['summary']['invariants'].values())
    requests = {row['request_id']: row for row in manifest['requests']}
    receipts = {(row['request_id'], row['layer']): row for row in physical['layers']}
    periods, demands, lane_audit = [], [], []
    used_receipts = set()
    for npu in range(N):
        batches = sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id'] == npu),
                         key=lambda b: b['admission_time_ms'])
        layers, lane_periods, lane_demands = [], [], []
        for batch in batches:
            assert len(batch['member_request_ids']) == 1
            rid = batch['member_request_ids'][0]
            load = requests[rid]['load']
            layers.extend(dict(request_id=rid, **row) for row in
                          sorted(batch['layer_metrics'], key=lambda x: x['layer']))
            start, end = batch['admission_time_ms'], batch['completion_time_ms']
            if duration(start, end) > 0:
                lane_demands.append(dict(npu_id=npu, request_id=rid, role=load['role'],
                    start_ms=max(LEFT, start), end_ms=min(RIGHT, end),
                    demand_GiB_s=load['per_layer_kv_gb']*1e6/load['per_layer_us']))
        for current, following in zip(layers, layers[1:]):
            start, end = current['compute_start_ms'], following['compute_start_ms']
            if duration(start, end) <= 0:
                continue
            key = (following['request_id'], following['layer'])
            receipt = receipts[key]
            load_next = requests[following['request_id']]['load']
            close(start, following['io_start_time_ms'], 'next read begins at current compute start')
            close(end, max(current['compute_end_ms'], following['io_ready_time_ms']), 'read/compute barrier')
            assert receipt['first_link_start_ms'] >= start - 1e-7, ('receipt begins before its cycle', key)
            assert receipt['last_link_end_ms'] <= end + 1e-7, ('receipt ends after its cycle', key)
            close(receipt['last_link_end_ms'], following['io_ready_time_ms'], 'last physical byte is IO ready')
            close(receipt['bytes_gib'], load_next['per_layer_kv_gb'], 'full next-layer receipt volume')
            placement = manifest['placements'][requests[following['request_id']]['placement_index']]
            expected_blocks = len(placement[0 if len(placement) == 1 else following['layer']])
            assert receipt['completed_blocks'] == expected_blocks
            clipped_start, clipped_end = max(LEFT, start), min(RIGHT, end)
            received = receipt['window_received_gib']
            if start >= LEFT and end <= RIGHT:
                close(received, receipt['bytes_gib'], 'interior cycle receives whole layer')
            assert key not in used_receipts
            used_receipts.add(key)
            lane_periods.append(dict(npu_id=npu,
                current_request_id=current['request_id'], current_layer=current['layer'],
                next_request_id=following['request_id'], next_layer=following['layer'],
                current_role=requests[current['request_id']]['load']['role'], next_role=load_next['role'],
                cross_request=current['request_id'] != following['request_id'],
                full_start_ms=start, full_end_ms=end, clipped_start_ms=clipped_start, clipped_end_ms=clipped_end,
                clipped_at_window_edge=start < LEFT or end > RIGHT,
                clipped_duration_ms=clipped_end-clipped_start,
                full_next_layer_received_GiB=receipt['bytes_gib'],
                actual_window_received_GiB=received,
                next_layer_first_physical_receive_ms=receipt['first_link_start_ms'],
                next_layer_last_physical_receive_ms=receipt['last_link_end_ms'],
                period_mean_supply_GiB_s=received*1000/(clipped_end-clipped_start)))
        for rows, start_name, end_name in ((lane_periods, 'clipped_start_ms', 'clipped_end_ms'),
                                           (lane_demands, 'start_ms', 'end_ms')):
            assert rows
            cursor = LEFT
            for row in rows:
                close(row[start_name], cursor, 'continuous exact lane partition')
                cursor = row[end_name]
            close(cursor, RIGHT, 'full warm-window lane coverage')
        sum_received = math.fsum(p['actual_window_received_GiB'] for p in lane_periods)
        close(sum_received, physical['window_npu_received_gib'][npu], 'per-NPU window receipts conserved')
        lane_audit.append(dict(npu_id=npu, periods=len(lane_periods),
            cross_request_periods=sum(p['cross_request'] for p in lane_periods),
            clipped_periods=sum(p['clipped_at_window_edge'] for p in lane_periods),
            covered_ms=math.fsum(p['clipped_duration_ms'] for p in lane_periods),
            period_integral_received_GiB=sum_received,
            physical_window_received_GiB=physical['window_npu_received_gib'][npu],
            mean_demand_GiB_s=math.fsum((d['end_ms']-d['start_ms'])*d['demand_GiB_s'] for d in lane_demands)/(RIGHT-LEFT)))
        periods.extend(lane_periods)
        demands.extend(lane_demands)
    unallocated = [key for key, r in receipts.items() if r['window_received_gib'] > 1e-12 and key not in used_receipts]
    assert not unallocated, ('unallocated physical receipt bytes', unallocated)

    events = defaultdict(lambda: [[], []])
    for d in demands:
        events[d['start_ms']][0].append(d['demand_GiB_s'])
        events[d['end_ms']][0].append(-d['demand_GiB_s'])
    for p in periods:
        events[p['clipped_start_ms']][1].append(p['period_mean_supply_GiB_s'])
        events[p['clipped_end_ms']][1].append(-p['period_mean_supply_GiB_s'])
    edges = sorted(events)
    close(edges[0], LEFT, 'fleet starts at window start')
    close(edges[-1], RIGHT, 'fleet ends at window end')
    values = [0.0, 0.0]
    segments = []
    for left, right in zip(edges, edges[1:]):
        values = [math.fsum([values[i], *events[left][i]]) for i in range(2)]
        assert values[0] > 0 and values[1] >= -1e-8
        segments.append(dict(start_ms=left, end_ms=right, start_s=left/1000, end_s=right/1000,
            total_demand_GiB_s=values[0], total_period_mean_supply_GiB_s=values[1],
            demand_integral_GiB=values[0]*(right-left)/1000,
            actual_received_integral_GiB=values[1]*(right-left)/1000))
    residuals = [math.fsum([values[i], *events[RIGHT][i]]) for i in range(2)]
    for value in residuals:
        close(value, 0, 'sweep closes to zero')
    demand_area = math.fsum(s['demand_integral_GiB'] for s in segments)
    received_area = math.fsum(s['actual_received_integral_GiB'] for s in segments)
    mean_demand, mean_supply = demand_area*1000/(RIGHT-LEFT), received_area*1000/(RIGHT-LEFT)
    close(mean_demand, math.fsum(metrics['demand_audit']['mean_nominal_gib_s_by_ssu']), 'matches demand audit mean')
    close(max(s['total_demand_GiB_s'] for s in segments), metrics['demand_audit']['peak_nominal_gib_s_by_ssu'][0], 'matches demand audit peak')
    close(received_area, math.fsum(physical['window_npu_received_gib']), 'fleet curve area equals exact physical bytes')
    policy = metrics['policy']
    audit = dict(policy=policy, source_case=folder.name, num_npu=N, num_ssu=metrics['num_ssu'], window_ms=[LEFT, RIGHT],
        demand_definition='Sum over NPUs of currently admitted request per-layer bytes / own layer compute time; no division by NPU count.',
        supply_definition='Per-NPU physical next-layer received bytes / compute-start to next-compute-start period, including request handoffs. Edge periods use actual in-window bytes / clipped period duration. Sum the resulting eight time functions.',
        instantaneous_throughput=False, utilization_is_supply_demand_ratio=False,
        mean_total_demand_GiB_s=mean_demand, mean_total_received_supply_GiB_s=mean_supply,
        peak_total_demand_GiB_s=max(s['total_demand_GiB_s'] for s in segments),
        peak_total_period_mean_supply_GiB_s=max(s['total_period_mean_supply_GiB_s'] for s in segments),
        total_supply_curve_area_GiB=received_area, total_physical_received_GiB=math.fsum(physical['window_npu_received_gib']),
        U_percent=metrics['U_percent'], SLO_1p5_percent=metrics['slo_1p5_percent'],
        period_count=len(periods), segment_count=len(segments), per_npu=lane_audit,
        checks=dict(same_run_observer_checks=True, original_result_hash_matches=True,
            original_manifest_hash_matches=True, physical_receipts_supported_inside_periods=True,
            all_eight_lanes_cover_entire_window=True, every_in_window_receipt_allocated_once=True,
            per_npu_supply_area_equals_received_bytes=True, fleet_supply_area_equals_received_bytes=True,
            mean_and_peak_demand_equal_original_audit=True),
        source_sha256={name: sha(folder/name) for name in ('result.json.gz', 'manifest.json.gz', 'metrics.json', 'receipts.json')})
    return dict(policy=policy, audit=audit, segments=segments, periods=periods, demands=demands)


def render(data, output, ymax):
    a, rows = data['audit'], data['segments']
    fig = plt.figure(figsize=(14.4, 6.6), dpi=160, facecolor='white')
    ax = fig.add_axes([.085, .235, .88, .53])
    fig.text(.065, .94, LABELS[data['policy']] + '：8 张 NPU 的总带宽需求与供给', fontsize=19, color=INK)
    fig.text(.065, .89, '8 NPU / 1 SSU × 40 GiB/s · ring hash · 每卡长短随机混合 · seed 7 · warm [2,4) 秒',
             fontsize=12.5, color=MUTED)
    fig.text(.065, .847, f'整个窗口的时间平均：需求 {a["mean_total_demand_GiB_s"]:.3f} GiB/s；'
             f'实际供给 {a["mean_total_received_supply_GiB_s"]:.3f} GiB/s', fontsize=12.5, color=INK)
    x = [r['start_s'] for r in rows] + [RIGHT/1000]
    d = [r['total_demand_GiB_s'] for r in rows]
    s = [r['total_period_mean_supply_GiB_s'] for r in rows]
    ax.step(x, d + [d[-1]], where='post', color=PURPLE, linestyle='--', linewidth=1.9,
            label='总需求：8 卡当前 B_i 相加', zorder=3)
    ax.step(x, s + [s[-1]], where='post', color=BLUE, linewidth=1.7,
            label='总供给：各卡按层周期平均后，再对 8 卡相加', zorder=2)
    ax.set_xlim(LEFT/1000, RIGHT/1000)
    ax.set_ylim(0, ymax)
    ax.set_xlabel('仿真时间（秒）', fontsize=12)
    ax.set_ylabel('总带宽（GiB/s）', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=.19, color='#99a8b5', linewidth=.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=11.4,
              handlelength=2.4, columnspacing=2)
    footnotes = [
        '供给先按每张卡各自的层周期取平均（包含等待），再对 8 卡相加；蓝线不是瞬时物理传输速率。',
        '跨请求周期也计入；窗口两端只用窗内实际收到的字节和片段时长，因此曲线面积与整窗收到量一致。',
        '纵轴为 8 卡总量，不除以 8。两条曲线之比不能作为 NPU 利用率；三张图使用相同纵轴。',
    ]
    for y, text in zip((.126, .084, .042), footnotes):
        fig.text(.065, y, text, fontsize=11.2, color=MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            bbox = artist.get_window_extent(renderer)
            assert bbox.x0 >= -1 and bbox.x1 <= width+1 and bbox.y0 >= -1 and bbox.y1 <= height+1, ('text overflow', artist.get_text())
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--font', type=Path, default=Path('/workspace/scratch/b8c9d6c799a1/fifo_figure_reference/fonts/KaiXinSong-Charts.ttf'))
    args = parser.parse_args()
    if args.font.exists():
        fontManager.addfont(str(args.font))
        plt.rcParams['font.family'] = FontProperties(fname=args.font).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    folders = {p: next(DATA.glob(f'formal_random_s1*/*_{p}/metrics.json')).parent for p in ('fifo', 'once', 'short_first')}
    analyses = [analyse(folders[p]) for p in ('fifo', 'once', 'short_first')]
    ymax = max(40, math.ceil(max(max(a['audit']['peak_total_demand_GiB_s'],
                                  a['audit']['peak_total_period_mean_supply_GiB_s']) for a in analyses)/10)*10)
    image_dir, data_dir = args.output_dir/'images/fleet_total_bandwidth', args.output_dir/'data'
    image_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    audits = []
    for data in analyses:
        policy = data['policy']
        data['audit']['common_y_axis_GiB_s'] = [0, ymax]
        render(data, image_dir/f'{policy}_total_demand_supply.png', ymax)
        write_csv(data_dir/f'{policy}_fleet_total_bandwidth_segments.csv', data['segments'])
        write_csv(data_dir/f'{policy}_fleet_total_layer_periods.csv', data['periods'])
        write_csv(data_dir/f'{policy}_fleet_total_request_demand_intervals.csv', data['demands'])
        write_json(data_dir/f'{policy}_fleet_total_bandwidth_audit.json', data['audit'])
        audits.append(data['audit'])
    write_json(data_dir/'fleet_total_bandwidth_audit.json', dict(all_checks_passed=True, scenarios=audits))
    print(json.dumps([dict(policy=a['policy'], mean_demand=a['mean_total_demand_GiB_s'],
                          mean_supply=a['mean_total_received_supply_GiB_s'],
                          supply_peak=a['peak_total_period_mean_supply_GiB_s'], periods=a['period_count']) for a in audits], indent=2))


if __name__ == '__main__':
    main()
