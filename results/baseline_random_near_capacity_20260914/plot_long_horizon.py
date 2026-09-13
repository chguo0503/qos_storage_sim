#!/usr/bin/env python3
"""Plot every saved consecutive 2s bin; does not run or change simulations."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qos_long_plot_mpl'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FONT = Path('/home/chguo/.fonts/msyh.ttc')
if not FONT.exists():
    FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(FONT)).get_name(),
                     'axes.unicode_minus': False, 'font.size': 11})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=HERE/'long_horizon_comparison.json')
    parser.add_argument('--title', required=True)
    parser.add_argument('--subtitle', required=True)
    args = parser.parse_args()
    source = args.comparison.resolve()
    data = json.loads(source.read_text())
    assert not data['pending'], 'Wait for both planned strategies before final plotting.'
    out = source.parent/'figures'/'long_horizon'
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=.085, right=.97, top=.79, bottom=.245)
    fig.text(.045, .945, args.title, fontsize=20, weight='bold', color='#17354D')
    fig.text(.045, .89, args.subtitle, fontsize=11, color='#667684')
    ax.axhspan(80, 90, color='#F3F7FB', label='目标区间：80%≤U<90%')
    ax.axvspan(2, 20, color='#F5EADA', alpha=.42)
    ax.axvline(20, color='#A4AFB9', linestyle=':', linewidth=1.2)
    notes = []
    bins_by_strategy = {}
    for strategy, name, color in [('baseline', 'Baseline', '#135DCF'),
                                   ('once', '流量分配策略', '#D27100')]:
        rows = [r for r in data['rows'] if r['strategy'] == strategy]
        bins = sorted([r for r in rows if r['end_s']-r['start_s'] == 2], key=lambda r:r['start_s'])
        assert len(bins) == 29
        assert all(r['start_s'] == 2+2*i for i,r in enumerate(bins))
        bins_by_strategy[strategy] = bins
        full = next(r for r in rows if (r['start_s'], r['end_s']) == (2,60))
        late = next(r for r in rows if (r['start_s'], r['end_s']) == (20,60))
        assert abs(sum(r['U_percent'] for r in bins)/29-full['U_percent']) < 1e-8
        xs = [r['start_s'] for r in bins] + [60]
        ys = [r['U_percent'] for r in bins]
        ax.step(xs, ys+[ys[-1]], where='post', color=color, linewidth=2.2,
                label=f'{name}：2–60秒平均 {full["U_percent"]:.2f}%')
        coverage = sum(r['mixed_cards'] == 32 for r in bins)
        active = sum(r['all_active'] for r in bins)
        notes.append(f'{name}：20–60秒平均 {late["U_percent"]:.2f}%；29个分窗中全卡有任务 {active}/29，全卡长短都有计算 {coverage}/29。')
    ax.set_xlim(2,60)
    ax.set_ylim(65,100)
    ax.set_xticks([2,10,20,30,40,50,60])
    ax.set_xlabel('时间（秒）', labelpad=10)
    ax.set_ylabel('每个完整2秒窗口的实际平均 NPU 利用率（%）', labelpad=10)
    ax.grid(axis='y', alpha=.3)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=10)
    fig.text(.045,.15,notes[0],fontsize=10.5,color='#135DCF')
    fig.text(.045,.105,notes[1],fontsize=10.5,color='#D27100')
    fig.text(.045,.055,'同一批≥65秒纯计算的有限输入；独立Random，seed7。保留全部连续分窗，不代表无限稳态；此图不推算SSD带宽。',fontsize=10,color='#667684')
    png = out/'random_utilization_all_2s_windows.png'
    fig.savefig(png,dpi=180,facecolor='white')
    plt.close(fig)
    provenance = {'source':str(source.relative_to(HERE)), 'source_sha256':sha(source),
                  'generator_sha256':sha(Path(__file__)), 'png_sha256':sha(png),
                  'title':args.title, 'subtitle':args.subtitle, 'all_29_bins':bins_by_strategy,
                  'meaning':'Every step is exact compute overlap/(32*2s), not an instantaneous utilization estimate.'}
    (out/'plot_data.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (out/'README.md').write_text('# 扩大统计窗口后的 Random 利用率\n\n'
        '每条台阶是一个完整2秒窗口的真实计算利用率，全部29个窗口均保留。图例给出2–60秒平均，图下给出20–60秒平均及逐卡长短覆盖。\n\n'
        '该65秒工作量重新对整个请求人口独立打乱，与22秒人口不是相同前缀。只有一个种子，不能当成多种子或无限稳态结论。\n\n'
        '![连续2秒利用率](random_utilization_all_2s_windows.png)\n',encoding='utf-8')
    print(png)


if __name__ == '__main__':
    main()
