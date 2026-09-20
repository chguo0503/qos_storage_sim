#!/usr/bin/env python3
"""Render SSU3/SSU4 Once fleet totals from complete immutable event logs."""
from pathlib import Path
from unittest.mock import patch
import argparse
import csv
import json
import math

from once_per_layer_ssu4_seed7 import bandwidth_source as source
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEFT, RIGHT = 2000., 4000.
source.plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def combined(pieces, edges):
    values = np.asarray(pieces, dtype=float)
    changes = np.zeros(len(edges))
    np.add.at(changes, np.searchsorted(edges, values[:, 0]), values[:, 2])
    np.add.at(changes, np.searchsorted(edges, values[:, 1]), -values[:, 2])
    rates = np.cumsum(changes)
    source.close(rates[-1], 0.)
    assert rates[:-1].min() > -1e-7
    return np.maximum(0., rates[:-1])


def build(directory, num_ssu, order):
    # Reuse the existing physical-byte audit with the selected topology. These
    # in-memory settings are restored after each case; no source file is edited.
    with patch.object(source, 'HERE', directory), patch.object(source, 'DISKS', num_ssu):
        data = source.analyse(order, strategy='once')
    assert data['num_ssu'] == num_ssu and data['strategy'] == 'once'
    demands, supplies, periods = [], [], []
    for lane in data['lanes']:
        for a, z, rate in lane['demands']:
            a, z = max(LEFT, a), min(RIGHT, z)
            if z > a:
                demands.append((a, z, rate))
        for cycle in lane['cycles']:
            a, z = cycle['clipped_start_ms'], cycle['clipped_end_ms']
            rate = cycle['total_curve_mean_supply_GiB_s']
            supplies.append((a, z, rate))
            periods.append(dict(npu=lane['npu'], start_ms=a, end_ms=z,
                                mean_supply_GiB_s=rate,
                                received_GiB=cycle['total_curve_received_GiB'],
                                source_group=cycle['group']))
    edges = np.unique([t for pieces in (demands, supplies) for a, z, _ in pieces for t in (a, z)])
    B, b = combined(demands, edges), combined(supplies, edges)
    mean_B = float(np.dot(B, np.diff(edges)) / (RIGHT-LEFT))
    mean_b = float(np.dot(b, np.diff(edges)) / (RIGHT-LEFT))
    received = math.fsum(lane['received_GiB'] for lane in data['lanes'])
    source.close(mean_B, data['totals']['sum_mean_demand_GiB_s'])
    source.close(mean_b, data['totals']['sum_mean_supply_GiB_s'])
    source.close(mean_b * (RIGHT-LEFT) / 1000, received)
    source.close(math.fsum(p['received_GiB'] for p in periods), received)
    comparison_path = directory / 'comparison.json'
    previous = source.read(comparison_path, data['sources'])
    reference = next(r for r in previous['results'] if r['strategy'] == 'once' and r['order'] == order
                     and r['start_ms'] == LEFT and r['end_ms'] == RIGHT)
    source.close(data['totals']['U_percent'], reference['U_percent'])
    out = directory / 'figures/fleet_total_bandwidth_curves'
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f'{order}_curves.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_ms', 'end_ms', 'total_demand_GiB_s', 'total_supply_GiB_s'])
        writer.writerows(zip(edges[:-1], edges[1:], B, b))

    fig = source.plt.figure(figsize=(15, 6.7), dpi=160, facecolor='white')
    fig.text(.065, .945, f'Once per layer {order.title()}：32 张 NPU 的总带宽需求与供给',
             fontsize=22, color=source.INK)
    fig.text(.065, .894, f'32 NPU / {num_ssu} SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · '
             f'NPU 平均利用率 {data["totals"]["U_percent"]:.2f}%', fontsize=13, color=source.MUTED)
    fig.text(.065, .850, f'整个窗口的时间平均：需求 {mean_B:.3f} GiB/s；实际供给 {mean_b:.3f} GiB/s',
             fontsize=13, color=source.INK)
    ax = fig.add_axes([.085, .225, .87, .555])
    ax.stairs(B, edges/1000, baseline=None, color=source.PURPLE, lw=2.3, ls='--',
              label='总需求：32 卡当前 B_i 相加')
    ax.stairs(b, edges/1000, baseline=None, color=source.BLUE, lw=2.0,
              label='总供给：各卡按层周期平均后，再对 32 卡相加')
    ax.set(xlim=(2, 4), ylim=(0, 1000), xticks=np.arange(2, 4.01, .25),
           yticks=[0, 200, 400, 600, 800, 1000], xlabel='仿真时间（秒）', ylabel='总带宽（GiB/s）')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=2, frameon=False,
              fontsize=10.5, borderaxespad=0)
    fig.text(.065, .115, '供给先按每张卡各自的层周期取平均（包含等待），再对 32 卡相加。',
             fontsize=12, color=source.MUTED)
    fig.text(.065, .073, '跨请求周期也计入；窗口两端只用窗内实际收到的字节和片段时长，因此曲线面积与整窗收到量一致。',
             fontsize=11.5, color=source.MUTED)
    fig.text(.065, .032, '蓝线表示周期平均供给；总需求是 32 卡 V/C 参考值之和。两条线之比不能作为 NPU 利用率。',
             fontsize=11.5, color=source.MUTED)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for artist in fig.findobj(source.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -2 and box.y0 >= -2 and box.x1 <= width+2 and box.y1 <= height+2, (artist.get_text(), box.bounds)
    artifacts = {str(csv_path.relative_to(ROOT)): source.sha(csv_path)}
    for ext in ('png', 'pdf'):
        path = out / f'{order}_total_demand_supply.{ext}'
        fig.savefig(path, dpi=160)
        artifacts[str(path.relative_to(ROOT))] = source.sha(path)
    source.plt.close(fig)
    result = dict(order=order, strategy='once', num_npu=32, num_ssu=num_ssu, seed=7,
                  window_ms=[LEFT, RIGHT], U_percent=data['totals']['U_percent'],
                  time_mean_total_demand_GiB_s=mean_B, time_mean_total_supply_GiB_s=mean_b,
                  received_GiB=received, curve_segments=len(B), periods=periods,
                  per_npu=data['per_npu'], sources=data['sources'], artifacts=artifacts,
                  pixels=[width, height], exactly_two_curves=True,
                  area_matches_whole_window=True, visible_labels_inside_canvas=True,
                  physically_verified_internal_cycles=data['physically_verified_internal_cycles'])
    print(json.dumps({key: result[key] for key in ('num_ssu', 'order', 'U_percent',
          'time_mean_total_demand_GiB_s', 'time_mean_total_supply_GiB_s')}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-ssu', nargs='+', type=int, choices=(3, 4), default=[3, 4])
    args = parser.parse_args()
    for num_ssu in args.num_ssu:
        directory = HERE / f'once_per_layer_ssu{num_ssu}_seed7'
        results = [build(directory, num_ssu, order) for order in ('ordered', 'random')]
        out = directory / 'figures/fleet_total_bandwidth_curves'
        write_json(out / 'checks.json', dict(all_checks_passed=True, no_new_simulation=True,
            num_npu=32, num_ssu=num_ssu, seed=7, window_ms=[LEFT, RIGHT], results=results,
            builders={str(p.relative_to(ROOT)): source.sha(p) for p in (Path(__file__), Path(source.__file__))}))
        lines = ['# Once per layer：32 卡总带宽需求与供给', '',
                 f'32 NPU、{num_ssu} SSU × 40 GiB/s、seed 7，warm [2,4) 秒。使用完整 Once 仿真的原始日志。', '',
                 '| 顺序 | NPU 利用率 | 总需求均值 GiB/s | 总供给均值 GiB/s | 图 |',
                 '|---|---:|---:|---:|---|']
        for result in results:
            order = result['order']
            lines.append(f'| {order.title()} | {result["U_percent"]:.4f}% | '
                         f'{result["time_mean_total_demand_GiB_s"]:.6f} | '
                         f'{result["time_mean_total_supply_GiB_s"]:.6f} | '
                         f'[PNG]({order}_total_demand_supply.png) · [PDF]({order}_total_demand_supply.pdf) |')
        lines += ['', '紫色虚线：同一时刻 32 张卡当前请求的 Bi=V/C 之和。蓝线：先把每张卡实际收到的数据量按自己的层周期平均，再在同一时刻对 32 张卡求和。', '',
                  '层周期从当前层开始计算到下一层开始计算，包含等待。跨请求周期计入；窗口边界按窗内收到量和片段时长计算。曲线面积已与整窗实际收到量核对一致，完整内部周期另外核对读取字节及计算时间。', '',
                  '两条线之比不能作为 NPU 利用率。蓝线为各卡周期平均后的总量，各卡周期并不对齐，局部峰值不能用来判断磁盘是否超速。四张图均采用 0–1000 GiB/s 纵轴，与参考图一致。', '',
                  '[Ordered 曲线 CSV](ordered_curves.csv) · [Random 曲线 CSV](random_curves.csv) · [来源与校验](checks.json) · [绘图脚本](../../../render_once_fleet_total_bandwidth_curves.py)', '']
        (out / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
