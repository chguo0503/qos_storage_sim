#!/usr/bin/env python3
"""Two standalone PNGs from exactly ten canonical analysis.json files; no simulation."""
from pathlib import Path
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STUDY = HERE.parents[1]
SEEDS = (7, 19, 43, 67, 101)
STRATEGIES = ('baseline', 'once')
WINDOWS = (('warm_2_4s', 2000., 4000.), ('long_2_20s', 2000., 20000.))
BLUE, ORANGE, INK, MUTED = '#1976b9', '#e77c21', '#243648', '#5c6b77'
NAMES = {'baseline': 'Baseline', 'once': '流量分配策略'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(values):
    assert len(values) == 5
    return dict(n=5, mean_percent=statistics.mean(values), sample_sd_pp=statistics.stdev(values))


def main():
    font = Path('/home/chguo/.fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        family = 'Noto Sans CJK JP'
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'font.size': 13, 'savefig.facecolor': 'white'})
    source_sha, analyses, checked_windows = {}, {}, []
    source_profiles = None
    for seed in SEEDS:
        for strategy in STRATEGIES:
            path = STUDY / 'runs' / f'context8_L384m1024_S10m128_ssu8_h22000_seed{seed}' / strategy / 'analysis.json'
            relative = str(path.relative_to(STUDY))
            source_sha[relative] = sha(path)
            a = json.loads(path.read_text())
            assert a['all_technical_checks_passed'] and a['no_new_simulation']
            assert (a['seed'], a['strategy'], a['order'], a['num_npu'], a['num_ssu'], a['n_layers']) == (seed, strategy, 'random', 32, 8, 8)
            assert set(a['roles']) == {'L', 'S'} and a['short_roles'] == ['S']
            profiles = {p['role']: {key: p[key] for key in ('seq_len_k', 'miss_tokens', 'C_ms', 'V_GiB', 'requests')}
                        for p in a['profiles']}
            assert len(profiles) == 2
            assert profiles['L']['seq_len_k'] == 384 and profiles['L']['miss_tokens'] == 1024
            assert profiles['S']['seq_len_k'] == 10 and profiles['S']['miss_tokens'] == 128
            assert profiles['S']['requests'] == profiles['L']['requests'] * 24
            assert all(p['constructed_profile'] and p['source_ttft_ms'] is None and
                       p['construction']['method'].startswith('affine_extrapolation') for p in a['profiles'])
            if source_profiles is None:
                source_profiles = profiles
            assert profiles == source_profiles
            for key, left, right in WINDOWS:
                matches = [w for w in a['windows'] if (w['start_ms'], w['end_ms']) == (left, right)]
                assert len(matches) == 1
                w = matches[0]
                assert w['all_32_active'] and w['long_short_mixed_pass'] and w['long_short_mixed_card_count'] == 32
                assert len(w['per_npu']) == 32
                assert w['nominal']['per_ssu_capacity_GiB_s'] == [40.] * 8
                recomputed = 100 * math.fsum(n['compute_ms'] for n in w['per_npu']) / (32 * (right - left))
                assert math.isclose(recomputed, w['U_percent'], abs_tol=1e-7, rel_tol=0)
                checked_windows.append(dict(seed=seed, strategy=strategy, window=key,
                    U_percent=w['U_percent'], recomputed_from_per_card_compute_percent=recomputed,
                    all_32_active=True, long_short_mixed_card_count=32))
            analyses[seed, strategy] = a
        assert analyses[seed, 'baseline']['input_fingerprint'] == analyses[seed, 'once']['input_fingerprint']
        def manifest_digest(a):
            matches = [digest for name, digest in a['sources'].items() if name.endswith('/manifest.json.gz')]
            assert len(matches) == 1
            return matches[0]
        assert manifest_digest(analyses[seed, 'baseline']) == manifest_digest(analyses[seed, 'once'])

    data = dict(schema_version=1, source_analysis_sha256=source_sha,
        builder_sha256=sha(__file__), seeds=list(SEEDS), strategy_display_names=NAMES,
        configuration=dict(num_npu=32, num_ssu=8, disk_capacity_GiB_s=40, n_layers=8,
                           request_count_ratio_long_to_short='1:24', profiles=source_profiles),
        definitions=dict(U='100 * exact compute overlap / (32 * common window duration).',
            sample_sd='Sample standard deviation across the five seed-level U values; denominator n-1 = 4; unit percentage points.',
            paired_gain='For each identical-input seed, allocation U minus Baseline U, then unweighted arithmetic mean of the five differences.',
            selection='Seed7 is the exploratory pilot; seeds19/43/67/101 were fixed for confirmation. The Once confirmation phase began after observing the separate seed7 policy gain.',
            input_protocol='Complete, independently shuffled per-card Random queues from the frozen experiment. This chart reads audited canonical analysis files only; it does not re-parse/re-audit queue permutations.',
            shaded_band='80–90% is the requested region of interest, not an SLO or confidence interval.',
            limitation='Both context lengths use affine extrapolated C from data, not measured prefill C at384K/10K. The near-capacity population does not imply per-disk instantaneous strict underload.'),
        windows={})
    outputs = {}
    for key, left, right in WINDOWS:
        rows = []
        for seed in SEEDS:
            pair = {s: next(w['U_percent'] for w in analyses[seed, s]['windows']
                           if (w['start_ms'], w['end_ms']) == (left, right)) for s in STRATEGIES}
            rows.append(dict(seed=seed, baseline_U_percent=pair['baseline'], once_U_percent=pair['once'],
                             paired_gain_pp=pair['once'] - pair['baseline']))
        summaries = {s: stats([r[s + '_U_percent'] for r in rows]) for s in STRATEGIES}
        gain = statistics.mean(r['paired_gain_pp'] for r in rows)
        data['windows'][key] = dict(start_ms=left, end_ms=right, rows=rows, summaries=summaries,
                                   mean_paired_gain_pp=gain, all_10_runs_active_and_mixed=True)

        fig = plt.figure(figsize=(16, 10), dpi=180, facecolor='white')
        fig.text(.07, .946, f'五个随机种子的 NPU 利用率｜{left / 1000:g}–{right / 1000:g} 秒',
                 fontsize=28, fontweight='bold', color=INK)
        fig.text(.07, .898, '32 NPU · 8 SSU × 40 GiB/s · 每卡完整独立 Random · 同种子使用同一批输入',
                 fontsize=16, color=MUTED)
        for x, s, color in ((.07, 'baseline', BLUE), (.405, 'once', ORANGE)):
            st = summaries[s]
            fig.text(x, .844, f'{NAMES[s]}：五种子均值 ± 样本 SD', fontsize=14, color=color)
            fig.text(x, .797, f'{st["mean_percent"]:.2f}% ± {st["sample_sd_pp"]:.2f} pp',
                     fontsize=24, fontweight='bold', color=color)
        fig.text(.76, .844, '同种子配对差的平均值', fontsize=14, color=MUTED)
        fig.text(.76, .797, f'+{gain:.2f} 个百分点', fontsize=24, fontweight='bold', color=INK)

        ax = fig.add_axes([.09, .267, .86, .47])
        ax.axhspan(80, 90, facecolor='#eaf2e6', alpha=.78, zorder=0)
        ax.text(-.49, 80.5, '关注区间 80–90%', fontsize=12, color='#6c7f66', va='bottom')
        for index, row in enumerate(rows):
            ax.plot([index - .13, index + .13], [row['baseline_U_percent'], row['once_U_percent']],
                    color='#bec8d1', linewidth=1.8, zorder=2)
            for s, offset, color, marker in (('baseline', -.13, BLUE, 'o'), ('once', .13, ORANGE, 's')):
                y = row[s + '_U_percent']
                ax.scatter(index + offset, y, color=color, marker=marker, s=115,
                           edgecolors='white', linewidths=1.5, zorder=4)
                ax.annotate(f'{y:.2f}%', (index + offset, y), xytext=(0, 11),
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=16, fontweight='bold', color=color)
        ax.set(xlim=(-.6, 4.6), ylim=(78, 100), xticks=list(range(5)),
               xticklabels=[f'seed {seed}' for seed in SEEDS], yticks=list(range(80, 101, 5)))
        ax.set_ylabel('实际 NPU 平均利用率（%）', fontsize=15, labelpad=15)
        ax.tick_params(axis='both', labelsize=14, length=0, pad=9)
        ax.grid(axis='y', color='#d7dee5', linewidth=.8, zorder=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#aebbc6')
        ax.legend(handles=[Line2D([], [], marker='o', linestyle='None', color=BLUE, markersize=9, label='Baseline'),
                           Line2D([], [], marker='s', linestyle='None', color=ORANGE, markersize=9, label='流量分配策略')],
                  loc='upper left', bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=13)
        footer = [
            '两策略全部 10 个运行：该窗口内 32 卡全程活跃，且每张卡都计算过长请求和短请求。',
            '输入：长 384K / miss 1024，短 10K / miss 128；请求数 1:24。长、短请求 C 均由 data 拟合外推，非对应长度的硬件实测。',
            '阴影 80–90% 为关注区间；纵轴放大为 78–100%。SD 表示五种子间波动，单位为百分点（pp），不是置信区间。',
            'seed 7 为探索案例；19/43/67/101 为固定确认种子。配对差 = 同 seed 的流量分配策略利用率 − Baseline 利用率。'
        ]
        for y, text in zip((.184, .139, .094, .049), footer):
            fig.text(.07, y, text, fontsize=12.1, color=MUTED)
        output = HERE / f'{key}_npu_utilization.png'
        fig.savefig(output, dpi=180, facecolor='white')
        plt.close(fig)
        outputs[output.name] = dict(sha256=sha(output), pixels=[2880, 1800], window_ms=[left, right])

    assert all(sha(STUDY / name) == digest for name, digest in source_sha.items())
    write(HERE / 'plot_data.json', data)
    checks = dict(all_checks_passed=True, no_simulation=True, source_analysis_count=10,
        source_analysis_sha256=source_sha, sources_unchanged_after_render=True,
        no_key_results_dependency=True, same_manifest_per_seed=True, checked_windows=checked_windows,
        all_20_run_windows_active_and_mixed=True, png_only=True, output_files=outputs,
        plot_data_sha256=sha(HERE / 'plot_data.json'), builder_sha256=sha(__file__), font_family=family,
        validation='Seed-level U independently recomputed from each analysis per-card compute overlap; all 20 selected run/window combinations have32 active and mixed cards. No simulation or canonical analysis was modified.')
    write(HERE / 'checks.json', checks)
    lines = ['# 五种子策略对比', '',
        '- [2–4 秒 warm 窗口](warm_2_4s_npu_utilization.png)',
        '- [2–20 秒长期窗口](long_2_20s_npu_utilization.png)', '',
        '两张独立 PNG，横轴为 seed 7/19/43/67/101。蓝色圆点为 Baseline，橙色方点为流量分配策略（内部目录名 once）。每个点标明实际利用率；灰色线只连接同一 seed 的两个策略。', '',
        '32 NPU，8 SSU，每盘 40 GiB/s，8 层。长请求总长 384K、miss 1024；短请求总长 10K、miss 128；请求数之比 1:24。每卡使用完整独立 Random 队列，两策略使用相同 manifest。长、短请求的 C 均由 data 线性拟合外推，属于模型构造，不是这些输入长度的硬件实测。', '',
        '利用率 = 该窗口全部 NPU 的实际计算时间 /（32 × 窗口长度）。两个窗口的全部10个运行均满足32卡全程活跃、每卡都计算过长短请求。', '',
        '图上统计为5个种子等权均值、样本标准差（分母 n−1=4），以及同种子策略差的均值。百分点和百分比不同：例如87%→94%为增加7个百分点。阴影80–90%是本研究关注区间，不是SLO或置信区间；纵轴放大为78–100%。', '',
        'seed7为探索案例，19/43/67/101为事先固定的确认种子；Once确认是在看到单独seed7收益后启动的阶段。五种子汇总保留全部种子，不声称整个搜索过程从未适应性选候选。近容量理想平均负载不等于逐盘逐时严格欠载。', '',
        '| 窗口 | Baseline均值 ± 样本SD | 流量分配策略均值 ± 样本SD | 同seed差均值 |',
        '|---|---:|---:|---:|']
    for key, left, right in WINDOWS:
        w = data['windows'][key];b = w['summaries']['baseline'];o = w['summaries']['once']
        lines.append(f'| [{left/1000:g}, {right/1000:g}) 秒 | {b["mean_percent"]:.4f}% ± {b["sample_sd_pp"]:.4f} pp | {o["mean_percent"]:.4f}% ± {o["sample_sd_pp"]:.4f} pp | +{w["mean_paired_gain_pp"]:.4f} pp |')
    lines.extend(['', '重建：`python results/baseline_random_near_capacity_20260914/figures/five_seed_strategy_comparison/render_five_seed.py`。', '',
        '绘图脚本只读取10份canonical `analysis.json`，在 [plot_data.json](plot_data.json) 和 [checks.json](checks.json) 绑定各文件SHA，并验证绘图前后不变。未读取或绑定会继续更新的key_results.json。脚本从每卡计算时间独立复核所有20个运行/窗口利用率，也检查成对manifest摘要相同。独立洗牌属于已冻结实验输入协议；本脚本不再次解析请求队列。', '',
        '来源：'])
    for name, digest in source_sha.items():
        lines.append(f'- [{name}](../../{name}) — SHA256 `{digest}`')
    (HERE / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(all_checks_passed=True, outputs=list(outputs),
                         summaries={key: {'strategies': value['summaries'], 'mean_paired_gain_pp': value['mean_paired_gain_pp']}
                                    for key, value in data['windows'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
