#!/usr/bin/env python3
"""Plot the measured screen_12 FIFO HOL witness; no simulation is run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'results/fixed128_32_underload_20260914'
DEFAULT_EVIDENCE = EXPERIMENT / 'screen_12/n8_s1_L128n1493_S32n1173_r1_random_b1_sync_jnarrow_seed7_fifo/fifo_hol_evidence.json'
DEFAULT_OUTPUT = EXPERIMENT / 'figures_selected/images/s1_174eba4438c9_fifo_hol_zoom.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--font', type=Path, default=Path('/workspace/scratch/b8c9d6c799a1/fifo_figure_reference/fonts/KaiXinSong-Charts.ttf'))
    args = parser.parse_args()
    e = json.loads(args.evidence.read_text())
    assert all(e['validation_checks'].values())
    fontManager.addfont(str(args.font))
    plt.rcParams.update({'font.family': FontProperties(fname=str(args.font)).get_name(),
                         'font.size': 12, 'axes.unicode_minus': False,
                         'savefig.facecolor': 'white'})
    ink, muted = '#20344c', '#617286'
    green, amber, blue, red = '#339667', '#f2cc69', '#297ab9', '#ba4141'
    long_colors = ['#d2e5f5', '#bed7ed', '#d2e5f5', '#bed7ed', '#d2e5f5', '#bed7ed']
    short = e['short_layer']
    longs = e['preceding_long_layers']
    t = e['time_accounting']
    witness = e['local_feasibility_witness']
    deadline = t['short_compute_deadline_ms']
    fig = plt.figure(figsize=(19, 12.8), facecolor='white')
    fig.text(.055, .953, '128K / 32K：FIFO 让短请求等待前方六个长读取', fontsize=25, color=ink)
    fig.text(.055, .919,
             '8 NPU / 1 SSU × 40 GiB/s · 每卡长短随机混合 · 常规 V/C 总需求上界 39.976 GiB/s · 真实内部层片段',
             fontsize=14.2, color=muted)
    fig.text(.055, .883,
             'NPU4 的短请求 4000021：计算 L2 仅 8.362 ms，却为下一层 L3 暴露等待 19.328 ms。',
             fontsize=16, color=ink)

    ax = fig.add_axes([.115, .47, .845, .34])
    ax.set_xlim(3862, 3894)
    ax.set_ylim(-.47, 2.72)
    ax.set_yticks([2, 1, 0], ['实际 NPU4\n计算 L2 → 等待 L3',
                             '实际 FIFO\n盘侧服务',
                             '局部可行次序\n离线推演 · 非重跑'])
    ax.tick_params(axis='y', length=0, labelsize=12)
    ax.set_xticks(range(3862, 3895, 4))
    ax.set_xlabel('绝对仿真时间（ms）', labelpad=10, fontsize=13)
    ax.grid(axis='x', color='#e9eef4', zorder=0)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#b9c7d4')
    height = .42
    def bar(start, end, y, color, label=None, labelcolor=ink):
        ax.barh(y, end-start, left=start, height=height, color=color,
                edgecolor='white', linewidth=1.0, zorder=3)
        if label:
            ax.text((start+end)/2, y, label, ha='center', va='center',
                    color=labelcolor, fontsize=11.5, zorder=4)
    bar(t['short_release_ms'], deadline, 2, green, '计算 L2\n8.362 ms', 'white')
    bar(deadline, t['short_io_ready_ms'], 2, amber, 'IO Stall：19.328 ms')
    ax.scatter([t['short_io_ready_ms']], [2], marker='D', s=53, color=green, zorder=6)
    ax.annotate('3890.614\nL3 开始计算', (t['short_io_ready_ms'], 2.22),
                xytext=(3891.0, 2.38), ha='center', va='bottom', fontsize=10.5, color=green,
                arrowprops={'arrowstyle': '-', 'color': green})

    for i, row in enumerate(longs):
        bar(row['first_ssd_start_ms'], row['last_ssd_end_ms'], 1, long_colors[i],
            f'NPU{row["npu_id"]}\nL{row["layer"]}')
    bar(short['first_ssd_start_ms'], short['last_ssd_end_ms'], 1, blue)
    ax.annotate('短 L3\n1.035 ms', ((short['first_ssd_start_ms']+short['last_ssd_end_ms'])/2, 1.23),
                xytext=(3891.3, 1.44), ha='center', fontsize=10.8, color=blue,
                arrowprops={'arrowstyle': '-', 'color': blue})
    ax.text((longs[0]['first_ssd_start_ms']+short['first_ssd_start_ms'])/2, 1.35,
            '六个长层：1.019402 GiB ÷ 40 GiB/s = 25.485 ms',
            ha='center', fontsize=12.5, color=ink)

    cursor = witness['start_ms']
    end = cursor + short['minimum_ssd_service_ms']
    bar(cursor, end, 0, blue)
    ax.annotate('先读短 L3\n最晚 3865.130 到齐', ((cursor+end)/2, .22),
                xytext=(3865.1, .39), va='bottom', ha='center', fontsize=10.4, color=blue,
                arrowprops={'arrowstyle': '-', 'color': blue})
    cursor = end
    for i, row in enumerate(longs):
        end = cursor + row['minimum_ssd_service_ms']
        bar(cursor, end, 0, long_colors[i], f'NPU{row["npu_id"]}\nL{row["layer"]}')
        cursor = end
    assert abs(cursor-short['last_ssd_end_ms']) < 1e-8
    ax.text((witness['start_ms']+cursor)/2, .34,
            '同样读取量、同样 40 GiB/s；这七个已释放任务可全部按时到齐',
            ha='center', fontsize=12.3, color=ink)

    ax.axvline(deadline, color=red, linestyle='--', linewidth=1.8, zorder=5)
    ax.text(deadline+.14, 2.58, '短层截止 3871.286', color=red, fontsize=11.7)
    earliest = witness['earliest_long_deadline_ms']
    ax.axvline(earliest, color='#6d7e91', linestyle=':', linewidth=1.8, zorder=2)
    ax.text(earliest-.1, .56, '最早长层截止\n3892.478', color=muted,
            fontsize=10.5, ha='right', va='center')

    fig.text(.115, .417, '对应请求的每层读取量 V、计算时间 C 和常规带宽需求 V/C', fontsize=15, color=ink)
    table_ax = fig.add_axes([.115, .16, .845, .235])
    table_ax.axis('off')
    columns = ['任务', '总长度 / NQL', 'V（MiB）', 'C（ms）', 'V/C（GiB/s）', '实际到齐（ms）', '截止（ms）']
    data = []
    for row in [*longs, short]:
        data.append([f'NPU{row["npu_id"]} · L{row["layer"]}',
                     f'{row["total_length_k"]:.0f}K / {row["nql"]}',
                     f'{row["bytes_mib"]:.3f}', f'{row["per_layer_compute_ms"]:.3f}',
                     f'{row["bytes_gib"]*1000/row["per_layer_compute_ms"]:.3f}',
                     f'{row["last_link_end_ms"]:.3f}', f'{row["compute_deadline_ms"]:.3f}'])
    table = table_ax.table(cellText=data, colLabels=columns, cellLoc='center',
                           loc='center', bbox=[0, 0, 1, 1],
                           colWidths=[.14, .18, .12, .12, .14, .16, .14])
    table.auto_set_font_size(False)
    table.set_fontsize(11.6)
    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1.2)
        if ri == 0:
            cell.set_facecolor('#294765'); cell.get_text().set_color('white')
        elif ri == 7:
            cell.set_facecolor('#e1eff9'); cell.get_text().set_color('#155a8c')
        else:
            cell.set_facecolor('#f0f4f8' if ri % 2 else '#f8fafc'); cell.get_text().set_color(ink)

    fig.text(.055, .108,
             '本图选取严重的局部片段：该层周期利用率 30.197%；整个 [2,4) s 窗口的短请求利用率为 86.296%。',
             fontsize=12.8, color=ink)
    fig.text(.055, .074,
             '下方“局部可行次序”是基于实测释放时刻、字节量、截止时间的离线算术；未重跑后续新层，不能当作完整策略结果。',
             fontsize=12.2, color=muted)
    fig.text(.055, .043,
             '各层均为 L1–L7 内部读取，短请求的等待与自身跨请求 L0 无关。盘实际服务始终 ≤ 40 GiB/s，图中没有突破容量。',
             fontsize=12.2, color=muted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
