#!/usr/bin/env python3
"""Read existing complete runs and render equal-seed normalized latency CDFs."""
from __future__ import annotations

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
OUT = HERE/'figures'/'ttft_cdf_three_strategies'
SEEDS = (7, 19, 43)
SCENARIOS = {'semi': '间歇过载', 'full': '持续过载'}
POLICIES = {
    'baseline': ('Baseline', '#454545', '--'),
    'once': ('原始 Once per layer', '#1768B4', '-'),
    'static': ('固定候选池 + Once', '#D55E00', '-.'),
}
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def read_csv(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def load_samples():
    reference = read_csv(HERE/'comparison.csv') + read_csv(HERE/'once_control_comparison.csv')
    ref = {(r['scenario'], r['policy'], int(r['seed'])): r for r in reference
           if r['window'] == 'warm_2_4s' and r['policy'] in POLICIES}
    macro = {(r['scenario'], r['policy']): r
             for r in read_csv(HERE/'threeway_macro_summary.csv') if r['window'] == 'warm_2_4s'}
    assert len(ref) == 18 and len(macro) == 6
    samples, runs, manifests, arrays = [], [], {}, {}
    for path in sorted((HERE/'runs').glob('*/command.json')):
        command = json.loads(path.read_bytes())
        if command.get('status') != 'complete' or command.get('pilot') or command.get('smoke'):
            continue
        key = (command.get('scenario'), command.get('policy'), command.get('seed'))
        if key not in ref:
            continue
        assert key not in arrays, ('duplicate case', key)
        command = read(path)
        folder = path.parent
        manifest, raw = [read(folder/f) for f in ('manifest.json.gz', 'result.json.gz')]
        scenario, policy, seed = key
        assert all(command['checks'].values()) and all(raw['summary']['invariants'].values())
        assert sha(folder/'manifest.json.gz') == command['manifest_sha256'] == ref[key]['manifest_sha256']
        assert sha(folder/'result.json.gz') == command['result_sha256'] == ref[key]['result_sha256']
        meta = manifest['metadata']
        assert (meta['num_npu'], meta['num_ssu'], meta['n_layers'], meta['seed'], meta['order']) == (32, 3, 8, seed, 'random')
        assert raw['input_fingerprint'] == manifest['input_fingerprint']
        if policy == 'once':
            assert command['once_control']['checks_passed']
            assert command['once_control']['new_candidate_pool_extension_installed'] is False
        identity = (scenario, seed)
        manifests.setdefault(identity, set()).add(command['manifest_sha256'])
        requests = {r['request_id']: r for r in manifest['requests']}
        metrics = raw['summary']['request_metrics']
        expected = 960 if scenario == 'semi' else 1344
        assert len(metrics) == len(requests) == command['completed_requests'] == expected
        assert {r['request_id'] for r in metrics} == set(requests)
        cohort = []
        for row in metrics:
            if not 2000 <= row['admission_time_ms'] < 4000:
                continue
            request = requests[row['request_id']]
            ideal = 8*request['load']['per_layer_us']/1000
            latency = row['completion_time_ms']-row['admission_time_ms']
            assert math.isfinite(latency) and latency >= ideal-1e-8
            assert math.isclose(row['own_compute_ms'], ideal, abs_tol=1e-8, rel_tol=0)
            ratio = latency/ideal
            cohort.append(dict(scenario=scenario, policy=policy, seed=seed,
                               request_id=row['request_id'], npu_id=request['npu_id'],
                               profile=request['load']['role'], category=request['load']['category'],
                               admission_ms=row['admission_time_ms'], completion_ms=row['completion_time_ms'],
                               latency_ms=latency, pure_compute_ms=ideal, latency_ratio=ratio,
                               slo_1p5_pass=ratio <= 1.5,
                               completed_after_window=row['completion_time_ms'] > 4000))
            assert (ratio <= 1.5) == (latency <= 1.5*ideal+1e-9)
        count = len(cohort)
        passed = sum(r['slo_1p5_pass'] for r in cohort)
        assert (count, passed) == (int(ref[key]['slo_count']), int(ref[key]['slo_passed']))
        values = np.sort([r['latency_ratio'] for r in cohort])
        arrays[key] = values
        runs.append(dict(scenario=scenario, policy=policy, seed=seed, count=count, passed=passed,
                         slo_percent=100*passed/count, U_percent=float(ref[key]['U_percent']),
                         after_window_count=sum(r['completed_after_window'] for r in cohort),
                         maximum_ratio=float(values[-1]), case=folder.name))
        for row in cohort:
            row['equal_seed_weight'] = 1/(len(SEEDS)*count)
        samples.extend(cohort)
    assert len(arrays) == 18 and all(len(hashes) == 1 for hashes in manifests.values())
    summary = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            rows = [r for r in runs if (r['scenario'], r['policy']) == (scenario, policy)]
            assert sorted(r['seed'] for r in rows) == list(SEEDS)
            slo = float(np.mean([r['slo_percent'] for r in rows]))
            assert abs(slo-float(macro[scenario, policy]['mean_slo_percent'])) < 1e-8
            summary.append(dict(scenario=scenario, policy=policy, seed_count=3,
                                total_samples=sum(r['count'] for r in rows),
                                mean_slo_percent=slo,
                                mean_U_percent=float(macro[scenario, policy]['mean_U_percent']),
                                maximum_ratio=max(r['maximum_ratio'] for r in rows)))
    return samples, runs, arrays, summary, manifests


def macro_cdf(arrays, scenario, policy, xx):
    return np.mean([np.searchsorted(arrays[scenario, policy, seed], xx, side='right') /
                    len(arrays[scenario, policy, seed]) for seed in SEEDS], axis=0)


def configure_font():
    path = subprocess.check_output(['fc-match', '-f', '%{file}', 'Noto Sans CJK SC'], text=True)
    font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'font.size': 12, 'axes.labelsize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False})


def plot(arrays, summary, scenario, zoom=False):
    xmin = .9
    maximum = max(r['maximum_ratio'] for r in summary if r['scenario'] == scenario)
    xmax = 3 if zoom else math.ceil(maximum+.15)
    fig, ax = plt.subplots(figsize=(12.7, 7.7))
    fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.205)
    for policy, (label, color, style) in POLICIES.items():
        s = next(r for r in summary if (r['scenario'], r['policy']) == (scenario, policy))
        knots = np.unique(np.concatenate([arrays[scenario, policy, seed] for seed in SEEDS]))
        xx = np.r_[xmin, knots[(knots > xmin) & (knots < xmax)], xmax]
        yy = macro_cdf(arrays, scenario, policy, xx)
        assert np.all(np.diff(yy) >= -1e-12) and yy[0] == 0
        if not zoom:
            assert yy[-1] == 1
        at_slo = float(macro_cdf(arrays, scenario, policy, np.array([1.5]))[0])
        assert abs(at_slo*100-s['mean_slo_percent']) < 1e-8
        ax.step(xx, yy, where='post', color=color, linestyle=style, linewidth=2.3,
                label=f"{label}  |  SLO={s['mean_slo_percent']:.2f}%", zorder=3)
        ax.plot(1.5, at_slo, 'o', color=color, markersize=5,
                markeredgecolor='white', markeredgewidth=.65, zorder=5)
    ax.set(xlim=(xmin, xmax), ylim=(0, 1.035), ylabel='累计请求比例（各种子 CDF 的均值）',
           xlabel='归一化耗时 = 接纳至prefill完成耗时 / 本请求纯计算时间')
    ax.xaxis.labelpad = 12
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if xmax <= 4:
        ax.xaxis.set_major_locator(MultipleLocator(.25 if zoom else .5))
    else:
        ax.set_xticks([1, 1.5, 3, 5, 7, 9, 11, 13, 15, 17])
    ax.grid(axis='y', color='#CDD2D6', linewidth=.7, alpha=.65, zorder=0)
    ax.tick_params(length=4, colors='#333333')
    ax.spines['left'].set_color('#777777')
    ax.spines['bottom'].set_color('#777777')
    ax.axvline(1.5, color='#666666', linestyle=':', linewidth=1.4, zorder=2)
    ax.text(1.5, 1.047, 'SLO × 1.5', ha='center', va='bottom', fontsize=11.5, color='#555555')
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=.97,
              edgecolor='#DDDDDD', fontsize=11.3, handlelength=3, labelspacing=.85, borderpad=.9)
    suffix = '（阈值附近放大）' if zoom else ''
    fig.text(.095, .950, f'{SCENARIOS[scenario]}：三种策略的归一化耗时 CDF{suffix}',
             fontsize=19 if zoom else 20, fontweight='bold', ha='left', va='top')
    fig.text(.095, .895, '32 NPU / 3 SSU × 40 GiB/s  ·  Random  ·  warm [2,4)秒内接纳的请求',
             fontsize=12.2, color='#555555', ha='left')
    note = ('仅放大横轴0.9～3倍，纵轴仍按全部样本计算；完整长尾请看全范围图。' if zoom else
            '读图：x=1.5处的纵轴就是 SLO×1.5 达标率；保留所有样本及完整长尾。')
    fig.text(.095, .095, note, fontsize=11.5, color='#333333', ha='left')
    fig.text(.095, .050, 'seed 7、19、43等权平均；窗内接纳后跟踪至完成，不含接纳前排队。各策略入选请求可能不同。',
             fontsize=10.3, color='#666666', ha='left')
    name = f'{scenario}_random_ttft_ratio_cdf' + ('_zoom' if zoom else '') + '.png'
    fig.savefig(OUT/name, dpi=190, facecolor='white')
    plt.close(fig)
    return name


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    samples, runs, arrays, summary, manifests = load_samples()
    OUT.mkdir(parents=True, exist_ok=True)
    configure_font()
    outputs = [plot(arrays, summary, scenario) for scenario in SCENARIOS]
    outputs.append(plot(arrays, summary, 'full', zoom=True))
    write_csv(OUT/'request_samples.csv', samples)
    write_csv(OUT/'per_seed_summary.csv', runs)
    write_csv(OUT/'summary.csv', summary)
    curve_rows = []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            knots = np.unique(np.r_[1.5, np.concatenate([arrays[scenario, policy, seed] for seed in SEEDS])])
            for x, y in zip(knots, macro_cdf(arrays, scenario, policy, knots)):
                curve_rows.append(dict(scenario=scenario, policy=policy, latency_ratio=float(x), macro_cdf=float(y)))
    write_csv(OUT/'cdf_points.csv', curve_rows)
    table = ['| 场景 | 策略 | 三种子总样本数 | CDF(1.5) | 最大耗时倍数 |',
             '|---|---|---:|---:|---:|']
    for row in summary:
        table.append(f"| {SCENARIOS[row['scenario']]} | {POLICIES[row['policy']][0]} | {row['total_samples']} | {row['mean_slo_percent']:.2f}% | {row['maximum_ratio']:.4f} |")
    (OUT/'README.md').write_text('''**多样 data 输入：三种策略的归一化耗时 CDF**

使用本目录已完成的18份原始日志，不重跑仿真、不修改原图。两种场景分别比较 Baseline、原始 Once per layer、固定候选池 + Once（static）。

- [间歇过载](semi_random_ttft_ratio_cdf.png)
- [持续过载：完整长尾](full_random_ttft_ratio_cdf.png)
- [持续过载：阈值附近放大](full_random_ttft_ratio_cdf_zoom.png)

横轴 =（prefill完成时刻 − 接纳时刻）/ 本请求8层纯计算时间。因此横轴1.5对应既有SLO×1.5阈值。这是接纳后prefill耗时，不含接纳前排队，不是真实首token事件。

32 NPU、3 SSU×40 GiB/s、24种data画像、Random输入。每个种子取2000≤接纳时刻<4000ms的全部请求，跟踪至最终完成，保留窗后完成及超时请求。相同场景、相同种子的三策略完整输入SHA一致；窗口中入选的请求ID集合和数量仍可能因策略不同而不同。

每个种子先计算精确经验CDF，再将seed 7、19、43的CDF等权平均：

```text
CDF_seed(x) = 该种子耗时倍数 <= x 的请求数 / 该种子窗内接纳请求数
CDF_mean(x) = (CDF_7(x) + CDF_19(x) + CDF_43(x)) / 3
```

同一种子内请求等权，不按计算时间加权；不同种子各占1/3。不能直接混合全部请求再算比例，否则会改变之前三种子等权表格的口径。归一化没有改变各请求自身的纯计算时间分母，也没有除以样本最大值。

''' + '\n'.join(table) + '''

两张主图均不截断长尾、不平滑、不拟合。持续过载放大图仅改变显示范围，保留原分母；三条曲线可能交叉，不能把某个阈值的改善解释为所有请求都变快。间歇组中Once与固定池的曲线接近，因此用不同颜色和线型区分。

CDF(1.5)逐种子与comparison.csv及once_control_comparison.csv核对，并与threeway_macro_summary.csv的mean_slo_percent核对。输出附逐请求样本、逐种子汇总、CDF数值及来源SHA，便于复查。
''')
    assert all(sha(ROOT/path) == digest for path, digest in SOURCES.items())
    checks = dict(no_new_simulation=True, source_files_unchanged=True,
                  source_sha256=SOURCES, builder_sha256=sha(Path(__file__)),
                  same_input_by_scenario_seed={f'{s}_seed{k}': next(iter(v)) for (s, k), v in manifests.items()},
                  full_runs_checked=18, normalized_x='(completion-admission)/own_compute',
                  window_ms=[2000, 4000], seeds=list(SEEDS), aggregation='equal mean of three seed ECDFs',
                  cdf_1p5_matches_existing_per_seed_and_macro_tables=True,
                  no_smoothing=True, full_tail_in_main_figures=True,
                  zoom_preserves_full_population_denominator=True,
                  outputs=outputs, summary=summary, visual_review='pending')
    (OUT/'checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(OUT), png=outputs, statistics=summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

