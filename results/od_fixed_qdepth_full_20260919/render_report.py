#!/usr/bin/env python3
"""Read completed fixed-depth OD runs and compare them with frozen full24 OD.

This file does not run the simulator. All displayed metrics are recalculated
from request/layer records. Output is restricted to this study's README and
figures directory; archived results and the template are never written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'results/template_od_baseline_20260919/diverse'
SEEDS = (7, 19, 43)
CATEGORIES = ('SS', 'LS', 'SL', 'LL')
POLICIES = ('old_od', 'fixed256_od')
LABELS = {'old_od': '原 OD（不限队列深度）', 'fixed256_od': 'OD（每卡每盘 256）'}
STYLES = {'old_od': ('#475569', '-', 3.8), 'fixed256_od': ('#d45e00', '--', 2.1)}
WARM = (2000.0, 4000.0)
SOURCES: dict[str, str] = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def dump_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams.update({'font.family': next(f for f in preferred if f in available),
                         'axes.unicode_minus': False, 'font.size': 11,
                         'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})


def category(load):
    return ('S' if load['seq_len_k'] <= 80 else 'L') + ('S' if load['nql'] < 512 else 'L')


def warm_analysis(raw):
    return next(a for a in raw['analysis']
                if a['start_ms'] == WARM[0] and a['end_ms'] == WARM[1])


def extract(raw, manifest, policy, seed):
    """Independent fixed-window compute and uncensored admission-cohort metrics."""
    summary = raw['summary']
    assert summary['num_npu'] == 32 and summary['num_ssu'] == 3
    assert summary['n_layers'] == 8 and summary['batch_size'] == 1
    assert summary['completed_blocks'] == summary['submitted_blocks']
    loads = {r['request_id']: r['load'] for r in manifest['requests']}
    assert summary['request_count'] == len(loads) == len(summary['request_metrics'])
    assert raw['input_fingerprint'] == manifest['input_fingerprint']
    busy = np.zeros(32)
    active = np.zeros(32)
    for batch in summary['microbatch_metrics']:
        npu = batch['npu_id']
        active[npu] += max(0, min(WARM[1], batch['completion_time_ms']) -
                           max(WARM[0], batch['admission_time_ms']))
        for layer in batch['layer_metrics']:
            busy[npu] += max(0, min(WARM[1], layer['compute_end_ms']) -
                             max(WARM[0], layer['compute_start_ms']))
    assert np.max(np.abs(active - (WARM[1] - WARM[0]))) < 1e-7
    utilization = float(np.mean(busy / (WARM[1] - WARM[0])) * 100)
    samples = []
    for row in summary['request_metrics']:
        if not WARM[0] <= row['admission_time_ms'] < WARM[1]:
            continue
        load = loads[row['request_id']]
        cat = category(load)
        assert cat == load['category'] == row['category']
        ideal = float(row['own_compute_ms'])
        assert math.isclose(ideal, 8 * load['per_layer_us'] / 1000, abs_tol=1e-8)
        elapsed = float(row['completion_time_ms'] - row['admission_time_ms'])
        raw_ratio = elapsed / ideal
        ratio = raw_ratio
        # Normalize only the existing 1e-9 ms acceptance tolerance, not the raw data.
        for boundary in (1.0, 1.5):
            if abs(elapsed - boundary * ideal) <= 1e-9:
                ratio = boundary
        passed = elapsed <= 1.5 * ideal + 1e-9
        assert (ratio <= 1.5) == passed
        samples.append(dict(policy=policy, seed=seed, request_id=row['request_id'],
                            npu_id=row['npu_id'], category=cat, total_K=load['seq_len_k'],
                            miss=load['nql'], admission_ms=row['admission_time_ms'],
                            completion_ms=row['completion_time_ms'], ideal_ms=ideal,
                            elapsed_ms=elapsed, noncompute_ms=elapsed-ideal,
                            ratio=ratio, raw_ratio=raw_ratio, passed=passed))
    assert samples
    analysis = warm_analysis(raw)
    assert math.isclose(utilization, analysis['U_percent'], abs_tol=1e-8)
    count, passed = len(samples), sum(r['passed'] for r in samples)
    assert count == analysis['slo']['count'] and passed == analysis['slo']['passed']
    metrics = dict(policy=policy, seed=seed, U_percent=utilization, count=count,
                   passed=passed, slo_percent=100*passed/count,
                   SSD_total_GiB_s=sum(analysis['SSD_GiB_s']),
                   full_U_percent=summary['fleet_npu_compute_utilization']*100)
    # Guard the legacy fraction convention rather than guessing units.
    assert 0 <= metrics['full_U_percent'] <= 100 + 1e-7
    per_npu = [dict(policy=policy, seed=seed, npu_id=i,
                    U_percent=float(busy[i]/(WARM[1]-WARM[0])*100)) for i in range(32)]
    return metrics, samples, per_npu


def ecdf(samples, policy, cat='ALL'):
    arrays = {seed: np.sort([r['ratio'] for r in samples
                            if r['policy'] == policy and r['seed'] == seed and
                            (cat == 'ALL' or r['category'] == cat)]) for seed in SEEDS}
    assert all(len(a) for a in arrays.values()), (policy, cat)
    return arrays


def curve(arrays, x):
    return np.mean([np.searchsorted(arrays[s], x, side='right')/len(arrays[s])
                    for s in SEEDS], axis=0)


def axis_cdf(ax, arrays, title, xmax=None):
    right = xmax or max(float(a[-1]) for d in arrays.values() for a in d.values()) * 1.04
    x = np.unique(np.concatenate([np.array([0.9, 1.0, 1.5, right])] +
                                 [a for d in arrays.values() for a in d.values()]))
    for policy in POLICIES:
        color, linestyle, width = STYLES[policy]
        rate = float(curve(arrays[policy], np.array([1.5]))[0]) * 100
        ax.step(x, curve(arrays[policy], x), where='post', color=color,
                linestyle=linestyle, linewidth=width,
                label=f'{LABELS[policy]}：{rate:.2f}%')
    error = float(np.max(np.abs(curve(arrays['old_od'], x) - curve(arrays['fixed256_od'], x))))
    ax.axvline(1.5, color='#b91c1c', linestyle=':', linewidth=1.5)
    ax.set_xlim(0.9, right); ax.set_ylim(0, 1.035)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title, loc='left', fontweight='bold')
    ax.grid(axis='y', color='#e2e8f0')
    ax.set_xlabel('归一化耗时（接纳至 prefill 完成 / 8 层纯计算）')
    ax.set_ylabel('请求累计比例')
    if error < 1e-12:
        ax.text(.98, .08, '两条曲线重合；橙色虚线覆盖灰色实线',
                transform=ax.transAxes, ha='right', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=.95))
    ax.legend(loc='lower right', bbox_to_anchor=(1, .14), fontsize=9)
    return error


def render_cdf(figdir, samples, timing):
    arrays = {p: ecdf(samples, p) for p in POLICIES}
    fig, ax = plt.subplots(figsize=(13, 7))
    error = axis_cdf(ax, arrays, '完整尾部；红线为 SLO × 1.5')
    max_timing_error = max(t['max_difference_ms'] for t in timing.values())
    if max_timing_error <= 1e-9 and error >= 1e-12:
        ax.text(.98, .08, '原始时序差不超过 1e-9 ms；曲线在数值精度内重合',
                transform=ax.transAxes, ha='right', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=.95))
    fig.suptitle('固定队列深度前后：OD 的请求耗时', fontsize=19, fontweight='bold', y=.97)
    fig.text(.075, .035, 'full24 · 32 NPU / 3 SSU × 40 GiB/s · warm [2,4) 秒接纳并跟踪至完成 · seed 7/19/43 等权\n'
             '百分比为 SLO × 1.5 达标率；不含接纳前排队。每卡每盘独占 256 个 SSD 在途槽，带宽配置不变。', fontsize=10)
    fig.subplots_adjust(left=.085, right=.97, top=.82, bottom=.21)
    fig.savefig(figdir/'overall_ttft_cdf.png', dpi=180); plt.close(fig)
    return dict(max_pointwise_ECDF_difference=error,
                note='An ECDF jump at nearly equal floating endpoints is not a latency effect.',
                maximum_request_layer_timing_difference_ms=max_timing_error)


def category_metrics(samples):
    rows = []
    for policy in POLICIES:
        for cat in CATEGORIES:
            groups = [[r for r in samples if r['policy']==policy and r['category']==cat and r['seed']==s]
                      for s in SEEDS]
            rates = [100*sum(r['passed'] for r in g)/len(g) for g in groups]
            rows.append(dict(policy=policy, category=cat, count=sum(map(len, groups)),
                             count_seed7=len(groups[0]), count_seed19=len(groups[1]), count_seed43=len(groups[2]),
                             slo_percent=float(np.mean(rates)), min_seed_percent=min(rates), max_seed_percent=max(rates)))
    return rows


def render_host_wait(figdir, wait_ms, peak_by_npu_ssu, disk_peaks):
    wait = np.asarray(wait_ms, dtype=float)
    peak = np.asarray(peak_by_npu_ssu, dtype=int)
    assert wait.shape == peak.shape == (32, 3)
    assert np.all(wait >= 0) and np.all(peak <= 256)
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(32)
    for disk, color in enumerate(('#0072b2', '#e69f00', '#009e73')):
        values = wait[:, disk] / 1000
        ax.bar(np.arange(32), values, bottom=bottom, width=.8,
               label=f'SSU {disk}', color=color)
        bottom += values
    ax.set_xticks(np.arange(32)); ax.tick_params(axis='x', labelsize=9)
    ax.set_xlim(-.8,31.8); ax.set_ylim(bottom=0)
    ax.set_xlabel('NPU 编号')
    ax.set_ylabel('配额满导致的主机阻塞时间之和（秒）')
    ax.grid(axis='y', alpha=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,1.17), ncol=3, frameon=False)
    affected = int(np.sum(np.sum(wait, axis=1) > 0))
    fig.suptitle(f'限深确实生效：{affected} 张卡在主机侧等待过槽位',
                 fontsize=18, fontweight='bold', y=.97)
    peak_text = ' / '.join(str(v) for v in disk_peaks)
    fig.text(.075,.84, f'seed 7 · 完整运行 · 卡/盘峰值 {int(peak.max())} / 256 块；'
             f'SSU 0/1/2 峰值：{peak_text}；每盘上限 8192 块', fontsize=11)
    fig.text(.075,.035, '每盘 8192 槽，32 卡各独占 256；SSD 完成即释放，不等到 HBM 接收完成。\n'
             '柱按三盘阻塞区间累计后叠加；并行等待可重叠，因此不是请求 TTFT、NPU stall 或整机墙钟时间。',fontsize=10)
    fig.subplots_adjust(left=.08,right=.98,top=.73,bottom=.21)
    fig.savefig(figdir/'host_queue_backpressure.png',dpi=180);plt.close(fig)


def compare_timing(old, new):
    """Do not compare queue-wait fields: delaying enqueue changes their clock."""
    previous = {r['request_id']: r for r in old['summary']['request_metrics']}
    current = {r['request_id']: r for r in new['summary']['request_metrics']}
    assert previous.keys() == current.keys()
    fields = ('admission_time_ms', 'completion_time_ms', 'own_compute_ms', 'io_stall_ms')
    differences = {field: max(abs(float(previous[r][field])-float(current[r][field]))
                              for r in previous) for field in fields}
    old_batches = {b['member_request_ids'][0]: b for b in old['summary']['microbatch_metrics']}
    new_batches = {b['member_request_ids'][0]: b for b in new['summary']['microbatch_metrics']}
    assert old_batches.keys() == new_batches.keys()
    for field in ('compute_start_ms', 'compute_end_ms', 'io_start_time_ms',
                  'io_ready_time_ms', 'io_barrier_wait_ms'):
        differences['layer_'+field] = max(
            abs(float(a[field])-float(b[field]))
            for rid in old_batches
            for a, b in zip(old_batches[rid]['layer_metrics'],
                            new_batches[rid]['layer_metrics']))
    warm_old = {rid for rid, r in previous.items() if WARM[0] <= r['admission_time_ms'] < WARM[1]}
    warm_new = {rid for rid, r in current.items() if WARM[0] <= r['admission_time_ms'] < WARM[1]}
    return dict(max_absolute_timing_difference_ms=differences,
                max_difference_ms=max(differences.values()),
                exact_timing_match=all(v == 0 for v in differences.values()),
                shared_warm_request_count=len(warm_old & warm_new),
                old_only_warm_request_count=len(warm_old - warm_new),
                new_only_warm_request_count=len(warm_new - warm_old))


def profile_block_counts(manifest):
    profiles = defaultdict(list)
    for request in manifest['requests']:
        load = request['load']
        placements = manifest['placements'][request['placement_index']]
        assert len(placements) == 1
        counts = np.zeros(3, dtype=int)
        for disk, size in placements[0]:
            assert size == 176 / 1048576
            counts[disk] += 1
        profiles[(load['seq_len_k'], load['nql'])].append(counts)
    return [dict(total_K=length, miss=miss, requests_per_seed=len(counts),
                 blocks_per_disk_min=int(np.min(counts)),
                 blocks_per_disk_max=int(np.max(counts)),
                 exceeds_256=bool(np.max(counts) > 256))
            for (length, miss), counts in sorted(profiles.items())]


def macro_metrics(metrics):
    output = []
    for policy in POLICIES:
        rows = [r for r in metrics if r['policy'] == policy]
        assert sorted(r['seed'] for r in rows) == list(SEEDS)
        row = dict(policy=policy, seeds='7,19,43', count_total=sum(r['count'] for r in rows))
        for key in ('U_percent', 'slo_percent', 'SSD_total_GiB_s', 'full_U_percent'):
            values = [r[key] for r in rows]
            row[key] = float(np.mean(values))
            row[key+'_min'] = min(values)
            row[key+'_max'] = max(values)
        output.append(row)
    return output


def validate_category_reconstruction(samples):
    errors = {}
    for policy in POLICIES:
        overall = ecdf(samples, policy)
        separated = {cat: ecdf(samples, policy, cat) for cat in CATEGORIES}
        x = np.unique(np.concatenate(list(overall.values())))
        recomposed = np.mean([
            sum(len(separated[cat][seed]) / len(overall[seed]) *
                (np.searchsorted(separated[cat][seed], x, side='right') /
                 len(separated[cat][seed])) for cat in CATEGORIES)
            for seed in SEEDS], axis=0)
        errors[policy] = float(np.max(np.abs(curve(overall, x)-recomposed)))
        assert errors[policy] < 1e-12
    return errors


def make_readme(macros, category_rows, timing, queue_section):
    by_policy = {r['policy']: r for r in macros}
    old, new = (by_policy[p] for p in POLICIES)
    delta_u = new['U_percent'] - old['U_percent']
    delta_slo = new['slo_percent'] - old['slo_percent']
    unchanged = abs(delta_u) < 1e-8 and abs(delta_slo) < 1e-8
    conclusion = ('本次限深后的 NPU 利用率和 SLO 达标率与原 OD 相同。'
                  if unchanged else
                  f'限深后 NPU 利用率变化 {delta_u:+.6f} 个百分点，SLO 达标率变化 {delta_slo:+.6f} 个百分点。')
    if unchanged and all(t['exact_timing_match'] for t in timing.values()):
        conclusion = '本次三个种子的所有请求与层完成时序完全相同，NPU 利用率和 SLO 达标率不变。'
    lines = ['# OD：固定队列深度与原配置对比', '', conclusion, '',
             '使用原 full24 的冻结输入：32 NPU、3 SSU × 40 GiB/s、8 层、batch=1、Random、seed 7/19/43。'
             '请求、卡内顺序、计算时间和原有 stripe 放置逐字节保持一致。这里只改变 OD 的 SSD 提交深度；'
             '没有修改旧结果或重新生成输入。', '',
             '每盘总深度 8192，32 张卡各独占 256 个槽，不借用其他卡的槽。'
             '槽计数包含已入盘队列与正在 SSD 服务的块，在 SSD 完成时释放；'
             '已经离开 SSD、尚在 NPU 接收链路上的块不占槽。盘侧 CIR 仍为每卡每盘 1.25 GiB/s，'
             'PIR 不限，仍按原 OD 机制借用空闲带宽。**不借队列槽，不等于不借带宽。**', '',
             '## 结果', '',
             '下表使用固定 warm `[2,4)` 秒。U 按 32 张卡的实际计算时间统计；'
             'SLO 取该窗口接纳的请求，跟踪至完整排空，三种子各自计算后等权平均。', '',
             '| 配置 | NPU 平均利用率 | TTFT SLO × 1.5 | SSU 实际总供给 | 样本数（3 种子合计） |',
             '|---|---:|---:|---:|---:|']
    for policy in POLICIES:
        r = by_policy[policy]
        lines.append(f"| {LABELS[policy]} | {r['U_percent']:.6f}% | {r['slo_percent']:.6f}% | "
                     f"{r['SSD_total_GiB_s']:.6f} GiB/s | {r['count_total']} |")
    lines.extend(['', f'利用率差为 **{delta_u:+.9f} 个百分点**，SLO 差为 **{delta_slo:+.9f} 个百分点**。', '',
                  f"全程（从输入开始至最后请求完成）的 NPU 利用率种子均值为：原 OD **{old['full_U_percent']:.6f}%**，"
                  f"限深 OD **{new['full_U_percent']:.6f}%**。全程包含启动及排空，与 warm 窗口不是同一统计范围。", '',
                  'TTFT 在这里是“接纳至 prefill 完成”的代理时延，**不含接纳前排队**，不是实际首个输出 token 时延。'
                  '归一化耗时为该代理时延除以请求自身 8 层纯计算时间；SLO × 1.5 为归一化耗时不超过 1.5。'
                  '绘图在 1 倍和 1.5 倍边界沿用原统计的 1e-9 ms 浮点容差；CSV 的 raw_ratio 保留归一前值。'
                  '这只处理浮点误差，没有改变仿真时序。', '',
                  '| 类别 | 原 OD 达标率 | 限深 OD 达标率 |', '|---|---:|---:|'])
    category_map = {(r['policy'], r['category']): r for r in category_rows}
    for cat in CATEGORIES:
        lines.append(f"| {cat} | {category_map['old_od',cat]['slo_percent']:.6f}% | "
                     f"{category_map['fixed256_od',cat]['slo_percent']:.6f}% |")
    lines.extend(['', '类别第 1 位按总输入长度：S ≤ 80K，L > 80K；第 2 位按 miss 数：S < 512，L ≥ 512。'
                  '各类别达标率以本类别样本为分母，不能把四类别达标率直接等权平均成整体达标率。', '',
                  '## 限深实际改变了什么', '', queue_section, '',
                  '每块 176 KiB，256 块等于每卡每盘 **44 MiB**。本输入的 32/64/80K 画像每盘只有 74–213 块；'
                  '128/160/200K 画像每盘有 330–533 块，会超过 256。每个种子共有 672/1344 个请求属于后一组。', '',
                  '队列容量限制了提前放入盘队列的数据量，没有直接降低服务带宽。盘持续有待服务块时，'
                  '完成一块即可补发一块；队尾变短不一定改变盘的队首及服务次序。'
                  '当前每块 SSD 服务约 4.196 μs，每卡发令间隔为 0.1 μs。'
                  '没有额外建模真实硬件的 doorbell、completion polling 或软件调度开销，因此本结果不能外推为所有设备限深均无代价。', '',
                  '## 等待的计时边界', '',
                  '盘内排队从**实际入盘队列**计时。限深会使部分块在主机侧等待提交；'
                  '只看盘内排队变短，不能据此宣称请求变快。需要保留层最初启动时刻，'
                  '待全部块到达 HBM 后才结束读取。各块等待相互重叠，其等待时间之和也不能直接加到请求 TTFT。', '',
                  'NPU 利用率、接纳/完成时刻及层等待由原始请求与层记录独立复算。'
                  '下表明确核对 warm 选入集合；时序检查则覆盖所有相同请求 ID，不只检查 warm 样本。', '',
                  '| seed | 请求/层最大时序差（ms） | 两边共有 warm 请求数 | 原 OD 独有 / 限深独有 |',
                  '|---:|---:|---:|---:|'])
    for seed in SEEDS:
        t = timing[seed]
        lines.append(f"| {seed} | {t['max_difference_ms']:.12g} | {t['shared_warm_request_count']} | "
                     f"{t['old_only_warm_request_count']} / {t['new_only_warm_request_count']} |")
    lines.extend(['', '## 图与数据', '',
                  '- [整体 CDF：完整尾部](figures/overall_ttft_cdf.png)',
                  '- [主机侧限深阻塞证据](figures/host_queue_backpressure.png)',
                  '- [逐种子指标](figures/per_seed_metrics.csv)',
                  '- [种子等权汇总](figures/summary.csv)',
                  '- [分类指标](figures/category_summary.csv)',
                  '- [全程队列与等待指标](figures/queue_metrics.csv)',
                  '- [请求样本](figures/request_samples.csv)',
                  '- [逐画像的盘上块数](figures/input_profile_blocks.csv)',
                  '- [报告核验与来源哈希](figures/render_checks.json)', '',
                  '图中若两条曲线重合，会明确标注，并使用较宽灰色实线与较窄橙色虚线叠画。'
                  '本报告仅读取完整正式结果，不启动仿真。', '',
                  '实现验证：项目回归 `python -m pytest -q tests` 已通过 69 项测试，包含固定深度行为验证；'
                  '不启用限深的 seed 7 对照复跑还须与旧 OD 请求/层时序 SHA 完全相同，报告生成前会强制检查。'
                  '各正式 run 的 `queue_depth_checks.json` 保留深度边界、块守恒和延迟分解的核验结果。', ''])
    return '\n'.join(lines)


def queue_metrics(raw, policy, seed):
    summary = raw['summary']
    requests = summary['request_metrics']
    blocks = sum(r['io_count'] for r in requests)
    assert blocks == summary['completed_blocks']
    result = dict(policy=policy, seed=seed, scope='full finite input; block weighted',
                  completed_blocks=blocks,
                  mean_SSD_queue_wait_ms=sum(r['avg_ssd_queue_wait_ms']*r['io_count'] for r in requests)/blocks,
                  mean_enqueue_to_HBM_ms=sum(r['avg_end_to_end_io_latency_ms']*r['io_count'] for r in requests)/blocks,
                  host_full_quota_state_seconds=None,
                  mean_host_activation_to_enqueue_ms=None,
                  mean_activation_to_HBM_ms=None,
                  blocked_episodes=None,
                  peak_path_outstanding=None,
                  peak_disk_outstanding=max(d['max_outstanding_blocks'] for d in summary['disk_stats']))
    for disk in summary['disk_stats']:
        result[f"peak_outstanding_ssu{disk['ssu_id']}"] = disk['max_outstanding_blocks']
    if policy == 'old_od':
        return result
    q = summary['ssd_queue_depth']
    assert q['enabled'] is True and q['per_npu_per_ssu_slots'] == 256
    assert q['total_reserved_slots_per_ssu'] == 8192
    assert q['queue_slot_borrowing'] is False and q['bandwidth_policy_changed'] is False
    assert q['matrix_order'] == '[npu_id][ssu_id]'
    peak = np.asarray(q['peak_outstanding_blocks_by_npu_ssu'])
    waits = np.asarray(q['host_blocked_state_ms_by_npu_ssu'])
    assert peak.shape == waits.shape == (32, 3)
    assert np.all(peak <= 256) and np.all(peak >= 0) and peak.max() == 256
    assert result['peak_disk_outstanding'] <= 8192
    assert waits.sum() > 0
    for key in ('activated_blocks', 'issued_blocks', 'ssd_completed_blocks', 'hbm_completed_blocks'):
        assert q[key] == blocks
    for key in ('host_deferred_blocks_at_stop', 'ssd_outstanding_blocks_at_stop',
                'link_outstanding_blocks_at_stop', 'input_not_yet_activated_blocks'):
        assert q[key] == 0
    assert all(summary['invariants'].values())
    host = q['completed_host_activation_to_enqueue_block_ms']/blocks
    after_enqueue = q['completed_enqueue_to_hbm_block_ms']/blocks
    total = q['completed_activation_to_hbm_block_ms']/blocks
    assert math.isclose(after_enqueue, result['mean_enqueue_to_HBM_ms'], rel_tol=1e-9, abs_tol=1e-8)
    assert math.isclose(host+after_enqueue, total, rel_tol=1e-9, abs_tol=1e-8)
    result.update(host_full_quota_state_seconds=float(waits.sum()/1000),
                  mean_host_activation_to_enqueue_ms=host,
                  mean_activation_to_HBM_ms=total,
                  blocked_episodes=int(np.asarray(q['blocked_state_episodes_by_npu_ssu']).sum()),
                  peak_path_outstanding=int(peak.max()))
    return result


def describe_queues(rows):
    result = ['下列诊断覆盖**完整有限输入**，不是 warm 两秒；每种策略对三个种子等权平均。'
              '盘内等待均按所有已完成块加权，主机阻塞区间之和则是另一种计数。', '',
              '| 指标 | 原 OD | 限深 OD |', '|---|---:|---:|']
    old = [r for r in rows if r['policy'] == 'old_od']
    new = [r for r in rows if r['policy'] == 'fixed256_od']
    for key, name, unit in [('mean_SSD_queue_wait_ms', '每块平均盘内排队', ' ms'),
                            ('mean_enqueue_to_HBM_ms', '每块平均实际入队至 HBM 到齐', ' ms')]:
        result.append(f'| {name} | {np.mean([r[key] for r in old]):.6f}{unit} | '
                      f'{np.mean([r[key] for r in new]):.6f}{unit} |')
    for key, name, unit in [('mean_host_activation_to_enqueue_ms', '每块平均主机提交等待', ' ms'),
                            ('mean_activation_to_HBM_ms', '每块平均层读取启动至 HBM 到齐', ' ms'),
                            ('host_full_quota_state_seconds', '配额满时各提交状态阻塞区间之和', ' 秒')]:
        result.append(f'| {name} | 原结果未记录该独立计数 | {np.mean([r[key] for r in new]):.6f}{unit} |')
    result.extend(['', '**主机提交等待**从层读取启动算到实际入盘队列，包含原有发令节拍；'
                   '**配额满阻塞区间**专门记录槽满造成的停止提交，两者不是同一指标。'
                   '所有限深 run 均观测到真实配额阻塞和 256 块峰值，同时满足每盘总深度不超过 8192。'
                   '原结果没有独立主机等待计数，因此表中没有将其错误填成零。', '',
                   '对限深结果逐块计时的核对关系是：主机提交等待 + 入队至 HBM 耗时 = 层读取启动至 HBM 耗时。'
                   '它是逐块累计量的恒等式，不是请求墙钟时间的可加分解。'])
    result.extend(['', '盘侧真实同时在途峰值（块，SSU 0 / 1 / 2）：', '',
                   '| seed | 原 OD | 限深 OD |', '|---:|---:|---:|'])
    for seed in SEEDS:
        pair = [next(r for r in rows if r['policy'] == policy and r['seed'] == seed)
                for policy in POLICIES]
        peaks = [' / '.join(str(r[f'peak_outstanding_ssu{disk}']) for disk in range(3)) for r in pair]
        result.append(f'| {seed} | {peaks[0]} | {peaks[1]} |')
    result.extend(['', '这里读取原始盘侧的同时在途峰值，未将各卡在不同时刻出现的峰值相加。'])
    return '\n'.join(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run-pattern',default='runs/full_od_depth256_seed{seed}',
                        help='relative to study, includes {seed}')
    parser.add_argument('--require-complete',action='store_true',
                        help='explicit reminder; three complete seeds are always required')
    args=parser.parse_args()
    # Require every complete run and the unlimited parity control before writing.
    paths={s:HERE/args.new_run_pattern.format(seed=s) for s in SEEDS}
    commands = {}
    for s, path in paths.items():
        command=read(path/'command.json')
        if command.get('status')!='complete' or command.get('smoke',False):
            raise SystemExit(f'Seed {s} is not a complete formal run; no report generated.')
        assert command['seed'] == s
        assert command['queue_depth_per_npu_per_ssu'] == 256
        assert command['queue_depth_per_ssu'] == 8192
        assert command['source_unchanged'] is True and all(command['checks'].values())
        assert command['result_sha256'] == sha(path/'result.json.gz')
        depth_check = read(path/'queue_depth_checks.json')
        assert depth_check['passed'] is True
        commands[s] = command
    control_dir = HERE/'runs/full_od_unlimited_seed7'
    control = read(control_dir/'command.json')
    parity = read(control_dir/'reference_comparison.json')
    assert control['status'] == 'complete' and control['source_unchanged'] is True
    assert parity['same_input'] and parity['timing_equal'] and parity['original_request_metrics_equal']
    assert parity['request_timing_sha256']['actual'] == parity['request_timing_sha256']['reference']
    assert parity['layer_metrics_sha256']['actual'] == parity['layer_metrics_sha256']['reference']
    metrics, samples, per_npu, queues = [], [], [], []
    timings, new_raw = {}, {}
    for seed in SEEDS:
        old_raw = read(OLD/'runs'/f'full_od_baseline_seed{seed}'/'result.json.gz')
        new_raw[seed] = read(paths[seed]/'result.json.gz')
        old_manifest_path = OLD/'inputs'/f'full_seed{seed}.json.gz'
        new_manifest_path = paths[seed]/'manifest.json.gz'
        old_manifest, new_manifest = read(old_manifest_path), read(new_manifest_path)
        assert sha(old_manifest_path) == sha(new_manifest_path) == commands[seed]['manifest_sha256']
        assert new_raw[seed]['queue_depth_per_npu_per_ssu'] == 256
        for policy, raw in [('old_od', old_raw), ('fixed256_od', new_raw[seed])]:
            m, s, n = extract(raw, old_manifest, policy, seed)
            metrics.append(m); samples.extend(s); per_npu.extend(n)
            queues.append(queue_metrics(raw, policy, seed))
        timings[seed] = compare_timing(old_raw, new_raw[seed])
    reconstruction = validate_category_reconstruction(samples)
    macros = macro_metrics(metrics)
    categories = category_metrics(samples)
    profiles = profile_block_counts(new_manifest)
    figdir = HERE/'figures'
    figdir.mkdir(exist_ok=True)
    setup_font()
    cdf_check = render_cdf(figdir, samples, timings)
    q = new_raw[7]['summary']['ssd_queue_depth']
    peaks = [d['max_outstanding_blocks'] for d in new_raw[7]['summary']['disk_stats']]
    render_host_wait(figdir, q['host_blocked_state_ms_by_npu_ssu'],
                     q['peak_outstanding_blocks_by_npu_ssu'], peaks)
    for filename, rows in [('per_seed_metrics.csv', metrics), ('summary.csv', macros),
                           ('category_summary.csv', categories), ('request_samples.csv', samples),
                           ('per_npu.csv', per_npu), ('queue_metrics.csv', queues),
                           ('input_profile_blocks.csv', profiles)]:
        dump_csv(figdir/filename, rows)
    readme = make_readme(macros, categories, timings, describe_queues(queues))
    (HERE/'README.md').write_text(readme, encoding='utf-8')
    assert all(sha(ROOT/path) == digest for path, digest in SOURCES.items())
    checks = dict(status='complete', seeds=list(SEEDS), no_simulation_run=True,
                  source_sha256=SOURCES, all_source_files_unchanged=True,
                  fixed_window_ms=list(WARM), same_input_manifest_bytes=True,
                  cohort='admitted in warm; follow to full completion',
                  macro='equal mean of per-seed statistics and ECDFs',
                  category_reconstruction_max_error=reconstruction,
                  cdf_comparison=cdf_check, request_layer_timing_comparison=timings,
                  plotted_ratio_boundary_normalizations=sum(r['ratio'] != r['raw_ratio'] for r in samples),
                  queue_depth_enabled_and_triggered=True,
                  queue_depth_and_block_conservation_passed=True,
                  unlimited_reference_parity=parity,
                  generated_png_sha256={p.name:sha(p) for p in sorted(figdir.glob('*.png'))},
                  generator_sha256=sha(__file__),
                  visual_review={'status':'pending'})
    (figdir/'render_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='complete', summary=macros, figures=list(checks['generated_png_sha256'])), ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
