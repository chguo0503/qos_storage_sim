"""Render the original intermittent-overload CDF with two requested strategy labels."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FONT = HERE / 'DroidSansFallback.ttf'
font_manager.fontManager.addfont(str(FONT))
family = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                     'font.size': 12, 'axes.labelsize': 13,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'path'})
with (HERE / 'cdf_points_two_strategies.csv').open() as f:
    rows = list(csv.DictReader(f))

policies = {'baseline': ('Baseline', '#454545', '--'),
            'once': ('流量分配', '#1768B4', '-')}
fig, ax = plt.subplots(figsize=(12.7, 7.7))
fig.subplots_adjust(left=.095, right=.97, top=.80, bottom=.205)
checks = {}
for policy, (label, color, style) in policies.items():
    data = sorted((r for r in rows if r['scenario'] == 'semi' and r['policy'] == policy),
                  key=lambda r: float(r['latency_ratio']))
    x = np.array([float(r['latency_ratio']) for r in data])
    y = np.array([float(r['macro_cdf']) for r in data])
    assert len(x) > 0 and np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= -1e-12)
    assert y[-1] == 1 and x[0] > .9 and x[-1] < 3
    at_slo = float(y[np.flatnonzero(x == 1.5)[0]])
    expected = .9627054975026562 if policy == 'baseline' else .998095238095238
    assert abs(at_slo - expected) < 1e-12
    xx, yy = np.r_[.9, x, 3.], np.r_[0., y, 1.]
    ax.step(xx, yy, where='post', color=color, linestyle=style, linewidth=2.3,
            label=f'{label}  |  SLO={at_slo*100:.2f}%', zorder=3)
    ax.plot(1.5, at_slo, 'o', color=color, markersize=5,
            markeredgecolor='white', markeredgewidth=.65, zorder=5)
    checks[policy] = {'label': label, 'source_points': len(x),
                      'cdf_at_1p5': at_slo, 'maximum_ratio': float(x[-1])}
ax.set(xlim=(.9, 3), ylim=(0, 1.035), ylabel='累计请求比例（各种子 CDF 的均值）',
       xlabel='归一化耗时 = 接纳至prefill完成耗时 / 本请求纯计算时间')
ax.xaxis.labelpad = 12
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.xaxis.set_major_locator(MultipleLocator(.5))
ax.grid(axis='y', color='#CDD2D6', linewidth=.7, alpha=.65, zorder=0)
ax.tick_params(length=4, colors='#333333')
ax.spines['left'].set_color('#777777')
ax.spines['bottom'].set_color('#777777')
ax.axvline(1.5, color='#666666', linestyle=':', linewidth=1.4, zorder=2)
ax.text(1.5, 1.047, 'SLO × 1.5', ha='center', va='bottom', fontsize=11.5, color='#555555')
ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=.97,
          edgecolor='#DDDDDD', fontsize=11.3, handlelength=3, labelspacing=.85, borderpad=.9)
fig.text(.095, .950, '间歇过载：两种策略的归一化耗时 CDF',
         fontsize=20, fontweight='bold', ha='left', va='top')
fig.text(.095, .895, '32 NPU / 3 SSU × 40 GiB/s  ·  Random  ·  warm [2,4)秒内接纳的请求',
         fontsize=12.2, color='#555555', ha='left')
fig.text(.095, .095, '读图：x=1.5处的纵轴就是 SLO×1.5 达标率；保留所有样本及完整长尾。',
         fontsize=11.5, color='#333333', ha='left')
fig.text(.095, .050, 'seed 7、19、43等权平均；窗内接纳后跟踪至完成，不含接纳前排队。各策略入选请求可能不同。',
         fontsize=10.3, color='#666666', ha='left')
stem = HERE / 'semi_random_ttft_ratio_cdf_two_strategies'
fig.savefig(stem.with_suffix('.png'), dpi=190, facecolor='white')
fig.savefig(stem.with_suffix('.svg'), facecolor='white')
plt.close(fig)
(HERE / 'checks.json').write_text(json.dumps({'no_new_simulation': True,
    'source': 'https://github.com/chguo0503/qos_storage_sim/blob/main/results/diverse_data_ssu3_l3_20260916/figures/ttft_cdf_three_strategies/cdf_points.csv',
    'source_git_blob_sha': 'fe2b17f6c64179d0dd70f9fe1dd3d002f30c6046',
    'curves': checks, 'curve_count': 2, 'static_removed': True,
    'cdf_values_preserved': True, 'axis_range': [.9, 3]}, ensure_ascii=False, indent=2))
print(json.dumps(checks, ensure_ascii=False))
