#!/usr/bin/env python3
"""Rebuild exact A/B request TTFT ECDF and SLO x1/x1.5 tables.

Uses saved request-level samples from completed runs; no simulation is rerun.
"""
from pathlib import Path
import csv
import json
import math
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

HERE = Path(__file__).resolve().parent
EPS_MS = 1e-9
POLICIES = [
    ('baseline', 'Baseline', '#4B5563', '--'),
    ('once', '流量分配', '#2474B7', '-'),
]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (HERE / 'source_samples.csv').open(newline='') as f:
        samples = list(csv.DictReader(f))
    refs = {r['policy']:r for r in csv.DictReader((HERE / 'source_summary.csv').open())}
    for row in samples:
        for key in ('admission_ms', 'completion_ms', 'latency_ms', 'pure_compute_ms', 'latency_ratio'):
            row[key] = float(row[key])
        for key in ('request_id', 'npu_id'):
            row[key] = int(row[key])
        assert 2000 <= row['admission_ms'] < 4000
        assert math.isclose(row['completion_ms'] - row['admission_ms'], row['latency_ms'], abs_tol=1e-9)
        assert row['latency_ms'] >= row['pure_compute_ms'] - EPS_MS
        assert math.isclose(row['latency_ratio'], row['latency_ms']/row['pure_compute_ms'], abs_tol=1e-12)
        row['slo_1_pass'] = row['latency_ms'] <= row['pure_compute_ms'] + EPS_MS
        row['slo_1p5_pass_recomputed'] = row['latency_ms'] <= 1.5*row['pure_compute_ms'] + EPS_MS
        assert row['slo_1p5_pass_recomputed'] == (row['slo_1p5_pass'] == 'True')
        # Collapse only floating-point roundoff at the decision boundaries.
        row['plot_ratio'] = row['latency_ratio']
        for threshold in (1.0, 1.5):
            if abs(row['latency_ms'] - threshold*row['pure_compute_ms']) <= EPS_MS:
                row['plot_ratio'] = threshold

    summaries = []
    for policy, label, _, _ in POLICIES:
        group = [r for r in samples if r['policy'] == policy]
        assert len({r['request_id'] for r in group}) == len(group)
        for role in ('all', 'A', 'B'):
            rr = [r for r in group if role == 'all' or r['role'] == role]
            n = len(rr)
            n1 = sum(r['slo_1_pass'] for r in rr)
            n15 = sum(r['slo_1p5_pass_recomputed'] for r in rr)
            assert sum(r['plot_ratio'] <= 1 for r in rr) == n1
            assert sum(r['plot_ratio'] <= 1.5 for r in rr) == n15
            summary = dict(policy=policy, label=label, role=role, request_count=n,
                           slo_1_passed=n1, slo_1_percent=100*n1/n,
                           slo_1p5_passed=n15, slo_1p5_percent=100*n15/n,
                           after_window_count=sum(r['completion_ms'] > 4000 for r in rr),
                           mean_ttft_ms=float(np.mean([r['latency_ms'] for r in rr])),
                           max_ratio=max(r['plot_ratio'] for r in rr))
            summaries.append(summary)
            if role == 'all':
                assert n == int(refs[policy]['n'])
                assert n15 == int(refs[policy]['slo_passed'])
                assert abs(summary['slo_1p5_percent'] - float(refs[policy]['slo_percent'])) < 1e-10
                assert summary['after_window_count'] == 32

    write_csv(HERE/'request_samples_with_slo.csv', samples)
    write_csv(HERE/'slo_summary.csv', summaries)

    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(HERE/'fonts/NotoSansCJKsc-Subset.otf'))
    plt.rcParams.update({'font.family':'Noto Sans CJK SC', 'font.size':11.5,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(10.6, 6.55))
    fig.subplots_adjust(left=.096, right=.968, top=.79, bottom=.17)
    xmin, xmax = .88, 4.1
    for policy, label, color, style in POLICIES:
        rr = [r for r in samples if r['policy'] == policy]
        values, counts = np.unique([r['plot_ratio'] for r in rr], return_counts=True)
        fraction = np.cumsum(counts)/len(rr)
        assert fraction[-1] == 1 and xmax > values[-1]
        ax.step(np.r_[xmin,values,xmax], np.r_[0,fraction,1], where='post',
                label=label, color=color, linestyle=style, linewidth=2.45, zorder=3)
        for threshold in (1,1.5):
            rate = sum(r['plot_ratio'] <= threshold for r in rr)/len(rr)
            ax.scatter([threshold],[rate],s=46,color=color,edgecolor='white',linewidth=.8,zorder=5)
    for threshold,label in [(1,'SLO × 1'),(1.5,'SLO × 1.5')]:
        ax.axvline(threshold,color='#8B929A',linestyle=':',linewidth=1.3,zorder=1)
        ax.text(threshold,1.04,label,ha='center',va='bottom',fontsize=10.5,color='#515963')
    ax.set(xlim=(xmin,xmax),ylim=(0,1.025),xlabel='TTFT / SLO  (SLO = 8-layer pure compute time)',
           ylabel='Cumulative request fraction (CDF)')
    ax.xaxis.set_major_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis='y',color='#D9DFE5',linewidth=.7,alpha=.8,zorder=0)
    ax.tick_params(colors='#404650',length=3.5)
    ax.spines['left'].set_color('#9198A1')
    ax.spines['bottom'].set_color('#9198A1')
    ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='#E0E4E8',
              framealpha=.98,borderpad=.8,labelspacing=.75,fontsize=11.2)
    fig.text(.096,.945,'A/B request TTFT CDF',fontsize=21,fontweight='bold',color='#1B2634')
    fig.text(.096,.886,'A:B = 1:2  |  32 NPU / 3 SSU  |  Random, seed 7  |  Warm [2,4) s',
             fontsize=11.3,color='#5A6573')
    fig.text(.096,.055,'A: 128K / NQL 256    B: 32K / NQL 4096    |    TTFT: admission to prefill completion',
             fontsize=10.2,color='#66717E')
    for ext in ('png','svg'):
        fig.savefig(HERE/f'AB_random_TTFT_CDF_SLO_1_1p5.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
    checks = dict(no_new_simulation=True,num_npu=32,num_ssu=3,n_layers=8,seed=7,
                  order='random',request_ratio='1:2',window_ms=[2000,4000],
                  cohorts='Admission in window; followed to completion; equal weight per request',
                  slo_definition='8 * request per-layer pure compute time',epsilon_ms=EPS_MS,
                  ecdf_1_and_1p5_match_counts=True,source_slo_1p5_matches=True,
                  complete_tail_shown=True,source_csv_sha256=hashlib.sha256((HERE/'source_samples.csv').read_bytes()).hexdigest(),
                  all_samples_finite=True)
    assert all(math.isfinite(r['latency_ratio']) for r in samples)
    (HERE/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps([r for r in summaries if r['role']=='all'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
