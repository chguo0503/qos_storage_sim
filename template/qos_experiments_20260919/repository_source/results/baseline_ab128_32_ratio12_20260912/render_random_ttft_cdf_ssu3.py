#!/usr/bin/env python3
"""Plot exact request ECDFs for the legacy Random/SSU3 matched seed7 cases.

Reads complete existing logs; never runs or changes a simulation. The warm
cohort contains requests admitted in [2000,4000) ms, followed to completion.
Ratio and physical milliseconds are separate PNGs, not combined panels.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
L3 = HERE/'slo_routing_ssu3_20260915'
OUT = HERE/'figures'/'ssu3'/'ttft_cdf_random_seed7'
CASES = (
    ('baseline', 'Baseline', 'baseline_seed7_remote_v2', '#454545', '--'),
    ('once', '原始 Once', 'once_seed7_remote_v2', '#1768B4', '-'),
    ('static_aggressive', '固定候选池 + Once', 'static_aggressive_seed7_remote', '#D55E00', '-'),
)
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    payload = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def load_samples():
    reference = list(csv.DictReader((L3/'comparison.csv').open()))
    SOURCES[str((L3/'comparison.csv').relative_to(ROOT))] = sha(L3/'comparison.csv')
    samples, summaries, manifests = [], [], set()
    for policy, label, folder, color, style in CASES:
        case = L3/'runs'/folder
        command, manifest, raw = [read(case/n) for n in
                                  ('command.json', 'manifest.json.gz', 'result.json.gz')]
        assert command['status'] == 'complete' and command['completed_simulation']
        assert command['strategy'] == policy and command['assignment'] == 'fixed'
        assert sha(case/'manifest.json.gz') == command['manifest_sha256']
        assert sha(case/'result.json.gz') == command['output_sha256']
        manifests.add(command['manifest_sha256'])
        meta = manifest['metadata']
        assert (meta['num_npu'], meta['num_ssu'], meta['seed'], meta['n_layers'], meta['order']) == (32,3,7,8,'random')
        assert raw['input_fingerprint'] == manifest['input_fingerprint']
        assert all(raw['summary']['invariants'].values())
        reqs = {q['request_id']:q for q in manifest['requests']}
        complete = raw['summary']['request_metrics']
        assert len(complete) == len(reqs) == 3840
        assert {q['request_id'] for q in complete} == set(reqs)
        cohort = []
        for q in complete:
            if not 2000 <= q['admission_time_ms'] < 4000:
                continue
            request = reqs[q['request_id']]
            ideal = 8*request['load']['per_layer_us']/1000
            latency = q['completion_time_ms']-q['admission_time_ms']
            assert math.isfinite(latency) and latency >= ideal-1e-8
            assert math.isclose(q['own_compute_ms'], ideal, rel_tol=0, abs_tol=1e-8)
            row = dict(policy=policy, request_id=q['request_id'], npu_id=request['npu_id'],
                       role=request['load']['role'], admission_ms=q['admission_time_ms'],
                       completion_ms=q['completion_time_ms'], latency_ms=latency,
                       pure_compute_ms=ideal, latency_ratio=latency/ideal,
                       slo_1p5_pass=(latency <= 1.5*ideal+1e-9),
                       completed_after_window=(q['completion_time_ms'] > 4000))
            cohort.append(row)
        ref = next(r for r in reference if r['policy']==policy and int(r['seed'])==7
                   and r['window']=='warm_2_4s')
        n = len(cohort)
        passed = sum(q['slo_1p5_pass'] for q in cohort)
        assert (n, passed) == (int(ref['slo_count']), int(ref['slo_passed']))
        # Verify the plotted ECDF(1.5) exactly gives the published SLO rate;
        # the old absolute-time epsilon does not change classification here.
        assert sum(q['latency_ratio'] <= 1.5 for q in cohort) == passed
        summary = dict(policy=policy, label=label, seed=7, start_ms=2000, end_ms=4000,
                       n=n, A_count=sum(q['role']=='A' for q in cohort),
                       B_count=sum(q['role']=='B' for q in cohort),
                       after_window_count=sum(q['completed_after_window'] for q in cohort),
                       slo_passed=passed, slo_percent=100*passed/n,
                       U_percent=float(ref['U_percent']))
        assert abs(summary['slo_percent']-float(ref['slo_percent'])) < 1e-8
        for field in ('latency_ratio', 'latency_ms'):
            values = np.array([q[field] for q in cohort])
            for quantile in (50,90,95,99):
                summary[f'{field}_p{quantile}'] = float(np.percentile(values, quantile, method='linear'))
            summary[f'{field}_max'] = float(values.max())
        samples.extend(cohort)
        summaries.append(summary)
    assert len(manifests) == 1
    return samples, summaries, manifests.pop()


def configure_font():
    path = subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'font.family':family, 'axes.unicode_minus':False,
                         'font.size':12, 'axes.labelsize':13,
                         'axes.spines.top':False, 'axes.spines.right':False})


def ecdf(values, xmin, xmax):
    unique, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    fraction = np.cumsum(counts)/sum(counts)
    assert np.all(np.diff(unique) > 0) and np.all(np.diff(fraction) > 0)
    assert fraction[-1] == 1 and xmin < unique[0] <= unique[-1] < xmax
    return np.r_[xmin, unique, xmax], np.r_[0, fraction, 1]


def plot(samples, summaries, metric):
    ratio = metric == 'ratio'
    field = 'latency_ratio' if ratio else 'latency_ms'
    xmin = .90 if ratio else 0
    max_value = max(q[field] for q in samples)
    xmax = math.ceil((max_value+.08)*10)/10 if ratio else math.ceil((max_value+5)/20)*20
    fig, ax = plt.subplots(figsize=(12.7,7.7))
    fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.205)
    for policy, name, folder, color, style in CASES:
        s = next(s for s in summaries if s['policy']==policy)
        vals = [q[field] for q in samples if q['policy']==policy]
        xx, yy = ecdf(vals, xmin, xmax)
        legend = (f"{name}  |  n={s['n']}  |  SLO={s['slo_percent']:.2f}%" if ratio else
                  f"{name}  |  n={s['n']}  |  P95={s['latency_ms_p95']:.2f} ms")
        ax.step(xx, yy, where='post', label=legend, color=color, linestyle=style,
                linewidth=2.3, zorder=3)
        if ratio:
            ax.plot(1.5, s['slo_percent']/100, marker='o', color=color,
                    markersize=5, markeredgecolor='white', markeredgewidth=.65, zorder=5)
    ax.set(xlim=(xmin,xmax), ylim=(0,1.035), ylabel='累计请求比例（CDF）')
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis='y', color='#CDD2D6', linewidth=.7, alpha=.65, zorder=0)
    ax.tick_params(length=4, colors='#333333')
    ax.spines['left'].set_color('#777777')
    ax.spines['bottom'].set_color('#777777')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=.97,
              edgecolor='#DDDDDD', fontsize=11.3, handlelength=2.7, labelspacing=.85,
              borderpad=.9)
    if ratio:
        ax.axvline(1.5, color='#666666', linestyle=':', linewidth=1.5, zorder=2)
        ax.text(1.5,1.047,'SLO × 1.5',ha='center',va='bottom',fontsize=11.5,color='#555555')
        ax.xaxis.set_major_locator(MultipleLocator(.5))
        ax.set_xlabel('归一化耗时 = 接纳至prefill完成耗时 / 本请求纯计算时间', labelpad=12)
        title = 'Random：三种策略的归一化耗时 CDF'
        explanation = '读图：横轴1.5表示耗时为纯计算时间的1.5倍；此处纵轴就是 SLO×1.5 达标率。'
        filename = 'random_ttft_ratio_cdf.png'
    else:
        limits = {role:1.5*next(q['pure_compute_ms'] for q in samples if q['role']==role)
                  for role in ('A','B')}
        for role, threshold in limits.items():
            ax.axvline(threshold,color='#777777',linestyle=':',linewidth=1.2,zorder=2)
            ax.text(threshold,1.047,f'{role}类SLO {threshold:.2f} ms',ha='center',va='bottom',
                    fontsize=11.0,color='#555555')
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.set_xlabel('接纳至prefill完成耗时（毫秒）', labelpad=12)
        title = 'Random：三种策略的实际耗时 CDF'
        explanation = '两类请求的SLO阈值不同，混合毫秒CDF不能直接读总体达标率；固定池的毫秒长尾更长。'
        filename = 'random_ttft_ms_cdf.png'
    fig.text(.095,.950,title,fontsize=20,fontweight='bold',ha='left',va='top')
    fig.text(.095,.895,'32 NPU / 3 SSU × 40 GiB/s  ·  seed 7  ·  warm [2,4)秒内接纳的请求',
             fontsize=12.2,color='#555555',ha='left')
    fig.text(.095,.095,explanation,fontsize=11.5,color='#333333',ha='left')
    fig.text(.095,.050,'每个请求等权，均跟踪到最终完成；不含接纳前排队。三策略的窗口入选请求集合可能不同。',
             fontsize=10.7,color='#666666',ha='left')
    fig.savefig(OUT/filename,dpi=190,facecolor='white')
    plt.close(fig)
    return filename


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('ratio','ms','both'), default='both')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    samples, summaries, manifest_sha = load_samples()
    configure_font()
    outputs = [plot(samples,summaries,m) for m in (('ratio','ms') if args.metric=='both' else (args.metric,))]
    write_csv(OUT/'request_samples.csv', samples)
    write_csv(OUT/'summary.csv', summaries)
    table = ['| 策略 | 样本数 | A / B | CDF(1.5) | 归一化P95 | 毫秒P95 |',
             '|---|---:|---|---:|---:|---:|']
    for s in summaries:
        table.append(f"| {s['label']} | {s['n']} | {s['A_count']} / {s['B_count']} | {s['slo_percent']:.2f}% | {s['latency_ratio_p95']:.4f} | {s['latency_ms_p95']:.2f} ms |")
    note = f'''**SSU3 Random：Baseline / 原始Once / 固定候选池的请求耗时CDF**

本次从已完成的原始日志画图，没有重跑仿真。32 NPU、3 SSU×40 GiB/s，seed7，每卡40A＋80B；A总长128K/miss256，B总长32K/miss4096，8层、batch=1。

样本：2000≤接纳时刻<4000ms的全部请求，跟踪至最终完成；三策略各有32个请求在窗口结束后完成，均保留。延迟=完成−接纳，不包含接纳前队列时间，也不是真实首token事件。

- [归一化耗时CDF](random_ttft_ratio_cdf.png)：横轴=延迟/本请求8层纯计算时间。CDF(x)是耗时倍数≤x的请求比例；因此CDF(1.5)正好等于之前表格的SLO×1.5达标率。此图不是NPU利用率的CDF。
- [实际毫秒耗时CDF](random_ttft_ms_cdf.png)：横轴=延迟毫秒值。A的SLO阈值约72.29ms，B约343.11ms；两个不同阈值不能用混合曲线的一个纵坐标替代总体达标率。

{chr(10).join(table)}

两图都为精确经验CDF（阶梯曲线），每条请求等权，不平滑、不拟合、不按计算时间加权；横轴包含所有样本，包括最大值，不截掉长尾。分位数表使用线性插值，与ECDF最近秩分位数并非完全相同。

固定池的归一化P95更低，而实际毫秒P95更高，两者并不矛盾：该策略改变A/B的等待分布，相对于各自计算时间改善紧预算请求的同时，会让部分B请求等得更久。曲线也会交叉，不能把SLO某个阈值处更高解释成所有分位数都更快。

三策略的完整3840请求输入逐字节一致；推进速度改变窗口接纳集合，所以窗口样本数分别为341、343、324，A/B构成也略有不同。这里只画与之前表格一致的窗口人群，不宣称三曲线对应相同请求ID集合。

输出：[逐请求样本](request_samples.csv)、[分位数与SLO汇总](summary.csv)、[数据源SHA与校验](checks.json)。图中数据与slo_routing_ssu3_20260915/comparison.csv的seed7 warm行逐项核对一致。
'''
    (OUT/'README.md').write_text(note)
    assert all(sha(ROOT/p) == digest for p,digest in SOURCES.items())
    checks = dict(no_new_simulation=True, source_files_unchanged=True,
                  same_full_input_sha256=manifest_sha, source_sha256=SOURCES,
                  builder_sha256=sha(Path(__file__)), num_npu=32, num_ssu=3, seed=7,
                  window_ms=[2000,4000], cohort='admitted in window, followed to completion',
                  cdf_definition='number of samples <= x / sample count',
                  no_smoothing=True, entire_tail_shown=True,
                  slo_from_cdf_matches_reference=True, outputs=outputs,
                  summary=summaries, visual_review='pending')
    (OUT/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(OUT),png=outputs,
                         statistics=[{k:s[k] for k in ('policy','n','slo_percent','latency_ratio_p95','latency_ms_p95')} for s in summaries]),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
