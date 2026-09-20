#!/usr/bin/env python3
"""Replot archived, common 2-ms measurements. No simulator or smoothing."""
from pathlib import Path
import argparse
import csv
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
LABELS = {'fifo': 'Baseline FIFO', 'once': 'Once per layer (5 ms)',
          'short_first': '短读取优先（诊断）'}
BLUE, ORANGE, GRAY = '#2166AD', '#D87513', '#62676F'

def load(policy, kind):
    with (ROOT/'data'/f'{policy}_{kind}_2ms.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

def style(ax, npu=False):
    ax.set_xlim(2, 4)
    ax.set_xticks(np.arange(2, 4.001, .25))
    ax.set_ylim(0, 55 if npu else 130)
    ax.set_yticks([0, 25, 50] if npu else [0, 40, 80, 120])
    ax.grid(axis='y', color='#E6E9EE', linewidth=.65)
    ax.tick_params(axis='both', length=3, color='#AEB6BF', labelcolor='#343B44')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color('#AFB8C3')

def legend(fig, npu=False, y=.93):
    handles = [Line2D([], [], color=ORANGE, lw=1.6, label='预取窗口需求（含 L0）'),
               Line2D([], [], color=BLUE, lw=1.6,
                      label='实际接收带宽（2 ms）' if npu else '盘实际带宽（2 ms）')]
    if not npu:
        handles.append(Line2D([], [], color=GRAY, lw=1.2, ls='--', label='容量 40 GiB/s'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, y),
               ncol=len(handles), frameon=False, fontsize=11, handlelength=2.2,
               columnspacing=2.2)

def total(ax, d):
    edges = np.r_[d['start_ms'], d['end_ms'][-1]] / 1000
    ax.stairs(d['ssd_actual_gib_s'], edges, color=BLUE, lw=.95, zorder=3)
    ax.stairs(d['prefetch_deadline_reference_gib_s'], edges, color=ORANGE,
              lw=1.1, zorder=4)
    ax.axhline(40, color=GRAY, lw=1.1, ls=(0, (5, 4)), zorder=5)
    style(ax)
    ax.set_ylabel('带宽（GiB/s）', labelpad=10)

def save(fig, name):
    fig.savefig(ROOT/'images'/f'{name}.png', dpi=200, facecolor='white')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', required=True, help='Chinese TTF/OTF font file')
    args = parser.parse_args()
    fontManager.addfont(args.font)
    plt.rcParams.update({'font.family': ['DejaVu Sans', FontProperties(fname=args.font).get_name()],
                         'font.size': 11, 'axes.unicode_minus': False,
                         'figure.facecolor': 'white', 'axes.linewidth': .7})
    (ROOT/'images').mkdir(exist_ok=True)
    for p, title in LABELS.items():
        d = load(p, 'total')
        fig, ax = plt.subplots(figsize=(13, 4.4))
        fig.subplots_adjust(left=.078, right=.985, bottom=.16, top=.75)
        fig.suptitle(title, x=.53, y=.968, fontsize=17)
        legend(fig, y=.895)
        total(ax, d)
        ax.set_xlabel('时间（s）', labelpad=8)
        save(fig, f'{p}_total_bandwidth_simple')

        d = load(p, 'npu')
        fig, axes = plt.subplots(8, 1, figsize=(13, 13.2), sharex=True)
        fig.subplots_adjust(left=.09, right=.985, bottom=.058, top=.895, hspace=.20)
        fig.suptitle(title, x=.53, y=.981, fontsize=18)
        legend(fig, npu=True, y=.954)
        fig.supylabel('带宽（GiB/s）', x=.018, fontsize=13)
        for n, ax in enumerate(axes):
            sel = d['npu_id'] == n
            edges = np.r_[d['start_ms'][sel], d['end_ms'][sel][-1]] / 1000
            ax.stairs(d['actual_received_gib_s'][sel], edges, color=BLUE, lw=.85)
            ax.stairs(d['prefetch_demand_gib_s'][sel], edges, color=ORANGE, lw=1.05)
            style(ax, npu=True)
            ax.text(.009, .78, f'NPU {n}', transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.90, pad=1.5))
        axes[-1].set_xlabel('时间（s）', labelpad=8)
        save(fig, f'{p}_8npu_bandwidth_simple')

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.1), sharex=True)
    fig.subplots_adjust(left=.078, right=.985, bottom=.095, top=.80, hspace=.25)
    fig.suptitle('8 NPU · 1 SSU', x=.53, y=.975, fontsize=19)
    legend(fig, y=.923)
    for ax, p in zip(axes, ['fifo', 'once']):
        total(ax, load(p, 'total'))
        ax.text(.012, .86, LABELS[p], transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    axes[-1].set_xlabel('时间（s）', labelpad=8)
    save(fig, 'fifo_once_bandwidth_comparison')

    with zipfile.ZipFile(ROOT/'mixed8_bandwidth_simple_images.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT/'images').glob('*.png')):
            z.write(path, path.name)
    with zipfile.ZipFile(ROOT/'mixed8_bandwidth_plot_data.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in [ROOT/'plot_bandwidth.py', ROOT/'README.md', *sorted((ROOT/'data').glob('*'))]:
            z.write(path, path.relative_to(ROOT))
    print(json.dumps({'images': [p.name for p in sorted((ROOT/'images').glob('*.png'))]}, ensure_ascii=False))

if __name__ == '__main__':
    main()
