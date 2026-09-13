#!/usr/bin/env python3
"""Recompute SSU4 utilization/SLO and render the established fleet curves.

Run individual cases after their simulations finish, then --assemble. All
statistics come from complete raw logs; no simulation is started here.
"""
from pathlib import Path
import argparse
import csv
import json
import math

import bandwidth_source as source
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE, ROOT = source.HERE, source.ROOT
STUDY = HERE.parent
FIGURES = STUDY / 'figures/ssu4'
ALPHA = 1.5
WINDOWS = ((2000., 4000.), (2000., 20000.))
CASES = ('baseline_random', 'baseline_ordered', 'once_random', 'once_ordered')
LABELS = {'baseline': 'Baseline', 'once': 'Once per layer'}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def case_path(strategy, order):
    if strategy == 'once':
        return HERE / 'runs' / order / 'once'
    return STUDY / 'validation20s/runs' / f'ssu4_{order}_k1_sync_seed7' / 'baseline'


def metrics(strategy, order, sources):
    case = case_path(strategy, order)
    command, man, raw = [source.read(case / name, sources) for name in
                         ('command.json', 'manifest.json.gz', 'result.json.gz')]
    baseline = case_path('baseline', order)
    reference = source.read(baseline / 'command.json', sources)
    assert command['status'] == 'complete' and command['completed_simulation']
    assert command['strategy'] == strategy and command['assignment'] == 'fixed'
    assert command['core_source_sha256'] == reference['core_source_sha256']
    assert raw['core_and_policy_sha256'] == command['core_source_sha256']
    assert source.sha(case / 'manifest.json.gz') == command['manifest_sha256']
    assert command['manifest_sha256'] == source.sha(baseline / 'manifest.json.gz')
    assert source.sha(case / 'result.json.gz') == command['output_sha256']
    assert raw['input_fingerprint'] == man['input_fingerprint']
    assert all(raw['summary']['invariants'].values())
    assert raw['summary']['num_npu'] == 32 and raw['summary']['num_ssu'] == 4
    assert raw['summary']['n_layers'] == 8 and raw['summary']['batch_size'] == 1
    assert raw['summary']['cross_request_layer0_prefetch']
    assert raw['modeled_control_latency_ms'] == 0
    if strategy == 'once':
        assert raw['collector_interval_ms'] == 5.0
    requests = {q['request_id']: q for q in man['requests']}
    completed = raw['summary']['request_metrics']
    assert len(completed) == len(requests) == 3840
    assert {q['request_id'] for q in completed} == set(requests)
    thresholds = {}
    for q in completed:
        request = requests[q['request_id']]
        ideal = 8 * request['load']['per_layer_us'] / 1000
        source.close(q['own_compute_ms'], ideal)
        source.close(q['arrival_time_ms'], 0.)
        assert math.isfinite(q['completion_time_ms'])
        thresholds[request['load']['role']] = ALPHA * ideal

    def slo(rows, clock='admission'):
        passed = sum(q['completion_time_ms'] - q[clock + '_time_ms'] <=
                     thresholds[requests[q['request_id']]['load']['role']] + 1e-9
                     for q in rows)
        return dict(passed=passed, count=len(rows),
                    percent=100 * passed / len(rows) if rows else None)

    results = []
    for left, right in WINDOWS:
        cohort = [q for q in completed if left <= q['admission_time_ms'] < right]
        admission = slo(cohort)
        by_role = {role: slo([q for q in cohort if requests[q['request_id']]['load']['role'] == role])
                   for role in ('A', 'B')}
        assert sum(r['count'] for r in by_role.values()) == admission['count']
        assert sum(r['passed'] for r in by_role.values()) == admission['passed']
        arrival = slo(cohort, 'arrival')
        per_npu = []
        classes = {role: dict(compute_ms=0., active_ms=0.) for role in ('A', 'B')}
        for npu in range(32):
            batches = [b for b in raw['summary']['microbatch_metrics'] if b['npu_id'] == npu]
            compute = active = 0.
            roles = set()
            for batch in batches:
                assert len(batch['member_request_ids']) == 1
                req = requests[batch['member_request_ids'][0]]
                assert req['npu_id'] == npu
                role = req['load']['role']
                busy = source.clip(batch['admission_time_ms'], batch['completion_time_ms'], left, right)
                work = math.fsum(source.clip(layer['compute_start_ms'], layer['compute_end_ms'], left, right)
                                 for layer in batch['layer_metrics'])
                compute += work
                active += busy
                classes[role]['compute_ms'] += work
                classes[role]['active_ms'] += busy
                if work > 0:
                    roles.add(role)
            assert 0 <= compute <= active + 1e-7 <= right - left + 2e-7
            per_npu.append(dict(npu=npu, compute_ms=compute, active_ms=active,
                                U_percent=100 * compute / (right - left), computed_roles=sorted(roles)))
        U = math.fsum(p['U_percent'] for p in per_npu) / 32
        reference_window = next(w for w in raw['windows'] if w['start_ms'] == left and w['end_ms'] == right)
        source.close(U, 100 * reference_window['mean_npu_utilization'])
        for values in classes.values():
            values['conditional_U_percent'] = 100 * values['compute_ms'] / values['active_ms']
            values['window_card_time_share_percent'] = 100 * values['active_ms'] / (32 * (right - left))
        if right == 4000.:
            ref_slo = raw['slo']['window_admissions']
            assert raw['slo']['alpha'] == ALPHA
            assert set(ref_slo['request_ids']) == {q['request_id'] for q in cohort}
            for clock, actual in (('admission', admission), ('arrival', arrival)):
                assert actual['passed'] == ref_slo[clock]['passed']
                assert actual['count'] == ref_slo[clock]['count']
        results.append(dict(strategy=strategy, order=order, start_ms=left, end_ms=right,
                            U_percent=U, admission_clock=admission, per_role=by_role,
                            arrival_clock_same_cohort=arrival,
                            window_arrivals=slo([q for q in completed if left <= q['arrival_time_ms'] < right], 'arrival'),
                            completed_after_window=sum(q['completion_time_ms'] > right for q in cohort),
                            request_ids=sorted(q['request_id'] for q in cohort), per_npu=per_npu, classes=classes,
                            all_cards_active=all(math.isclose(p['active_ms'], right-left, abs_tol=1e-7) for p in per_npu),
                            mixed_card_count=sum(p['computed_roles'] == ['A', 'B'] for p in per_npu)))
    return results, thresholds


def combined(pieces, edges):
    values = np.asarray(pieces)
    changes = np.zeros(len(edges))
    np.add.at(changes, np.searchsorted(edges, values[:, 0]), values[:, 2])
    np.add.at(changes, np.searchsorted(edges, values[:, 1]), -values[:, 2])
    rates = np.cumsum(changes)
    source.close(rates[-1], 0.)
    assert rates[:-1].min() > -1e-7
    return np.maximum(0., rates[:-1])


def build_case(name):
    strategy, order = name.split('_')
    data = source.analyse(order, strategy)
    results, thresholds = metrics(strategy, order, data['sources'])
    source.close(data['totals']['U_percent'], results[0]['U_percent'])
    demand, supply = [], []
    for lane in data['lanes']:
        for a, z, rate in lane['demands']:
            a, z = max(2000., a), min(4000., z)
            if z > a:
                demand.append((a, z, rate))
        for cycle in lane['cycles']:
            supply.append((cycle['clipped_start_ms'], cycle['clipped_end_ms'],
                           cycle['total_curve_mean_supply_GiB_s']))
    edges = np.unique([t for pieces in (demand, supply) for a, z, _ in pieces for t in (a, z)])
    B, b = combined(demand, edges), combined(supply, edges)
    mean_B = float(np.dot(B, np.diff(edges)) / 2000.)
    mean_b = float(np.dot(b, np.diff(edges)) / 2000.)
    source.close(mean_B, data['totals']['sum_mean_demand_GiB_s'])
    source.close(mean_b, data['totals']['sum_mean_supply_GiB_s'])
    source.close(mean_b * 2, math.fsum(lane['received_GiB'] for lane in data['lanes']))
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    curve_path = out / f'{name}_curves.csv'
    with curve_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_ms', 'end_ms', 'total_demand_GiB_s', 'total_supply_GiB_s'])
        writer.writerows(zip(edges[:-1], edges[1:], B, b))
    audit = dict(all_checks_passed=True, num_npu=32, num_ssu=4, seed=7, strategy=strategy, order=order,
                 window_ms=[2000., 4000.], alpha=ALPHA, thresholds_ms=thresholds,
                 sources=data['sources'], builders={str(p.relative_to(ROOT)): source.sha(p)
                     for p in (Path(__file__), Path(source.__file__))},
                 metrics=results, totals=data['totals'], per_npu_bandwidth=data['per_npu'],
                 curves=str(curve_path.relative_to(ROOT)), curves_sha256=source.sha(curve_path),
                 curve_segments=len(B), cycles=data['cycles'],
                 area_matches_whole_window=True, physically_verified_internal_cycles=True,
                 trace_resource_nonoverlap_verified=True)
    write_json(out / f'{name}.json', audit)
    print(json.dumps(dict(case=name, U_percent=results[0]['U_percent'],
                         slo=results[0]['admission_clock'], mean_total_demand_GiB_s=mean_B,
                         mean_total_supply_GiB_s=mean_b), ensure_ascii=False), flush=True)


def load_case(name):
    result = json.loads((HERE / 'analysis' / f'{name}.json').read_text())
    for path, digest in result['sources'].items():
        assert source.sha(ROOT / path) == digest, path
    for path, digest in result['builders'].items():
        assert source.sha(ROOT / path) == digest, path
    curve_path = ROOT / result['curves']
    assert source.sha(curve_path) == result['curves_sha256']
    curves = np.loadtxt(curve_path, delimiter=',', skiprows=1, ndmin=2)
    assert np.array_equal(curves[:-1, 1], curves[1:, 0])
    source.close(curves[0, 0], 2000.)
    source.close(curves[-1, 1], 4000.)
    return result, curves


def plot_curves(ax, curves):
    edges = np.r_[curves[:, 0], curves[-1, 1]] / 1000
    ax.stairs(curves[:, 2], edges, baseline=None, color=source.PURPLE, lw=2.0, ls='--')
    ax.stairs(curves[:, 3], edges, baseline=None, color=source.BLUE, lw=1.7)
    ax.set(xlim=(2, 4), ylim=(0, 1000), xticks=np.arange(2, 4.01, .5),
           yticks=[0, 200, 400, 600, 800, 1000], xlabel='仿真时间（秒）', ylabel='总带宽（GiB/s）')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.15)


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for artist in fig.findobj(source.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -2 and box.y0 >= -2 and box.x1 <= width+2 and box.y1 <= height+2, (artist.get_text(), box.bounds)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return dict(path=str(path.relative_to(ROOT)), sha256=source.sha(path),
                pixels=[width, height], visible_labels_inside_canvas=True)


def legend():
    return [Line2D([], [], color=source.PURPLE, lw=2, ls='--', label='总需求：32 卡当前 B_i = V/C 相加'),
            Line2D([], [], color=source.BLUE, lw=2, label='总供给：各卡按层周期平均，再对 32 卡相加')]


def draw_individual(audit, curves):
    strategy, order = audit['strategy'], audit['order']
    totals, metric = audit['totals'], audit['metrics'][0]
    fig = plt.figure(figsize=(15, 6.7), dpi=160, facecolor='white')
    fig.text(.065, .945, f'{LABELS[strategy]} {order.title()}：32 张 NPU 的总带宽需求与供给', fontsize=22, color=source.INK)
    fig.text(.065, .895, f'32 NPU / 4 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · U={metric["U_percent"]:.2f}% · SLO×1.5={metric["admission_clock"]["percent"]:.2f}%', fontsize=13, color=source.MUTED)
    fig.text(.065, .851, f'整个窗口的时间平均：需求 {totals["sum_mean_demand_GiB_s"]:.3f} GiB/s；实际供给 {totals["sum_mean_supply_GiB_s"]:.3f} GiB/s', fontsize=13, color=source.INK)
    ax = fig.add_axes([.085, .225, .87, .555])
    plot_curves(ax, curves)
    ax.set_xticks(np.arange(2, 4.01, .25))
    ax.legend(handles=legend(), loc='lower left', bbox_to_anchor=(0, 1.015), ncol=2, frameon=False, fontsize=10.5, borderaxespad=0)
    fig.text(.065, .115, '供给先按每张卡各自的层周期取平均（包含等待），再对 32 卡相加；蓝线表示周期平均供给。', fontsize=12, color=source.MUTED)
    fig.text(.065, .073, '跨请求周期计入；窗口两端按窗内实际收到量与片段时长计算，曲线面积等于整窗收到量。', fontsize=11.5, color=source.MUTED)
    fig.text(.065, .032, 'U 由真实计算时间统计；两条线之比不能作为 U。SLO 使用接纳到 8 层完成的延迟。', fontsize=11.5, color=source.MUTED)
    directory = FIGURES / ('once_per_layer' if strategy == 'once' else 'fleet_total_bandwidth_curves')
    return save_figure(fig, directory / f'{order}_total_demand_supply.png')


def draw_comparison(cases):
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), dpi=160, facecolor='white')
    fig.subplots_adjust(left=.065, right=.98, top=.795, bottom=.145, hspace=.58, wspace=.17)
    fig.text(.05, .955, '4 SSU：Baseline 与 Once per layer 的整机总带宽', fontsize=24, color=source.INK)
    fig.text(.05, .91, '32 NPU · 4 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 相同顺序的输入逐字节一致', fontsize=14, color=source.MUTED)
    fig.legend(handles=legend(), loc='upper left', bbox_to_anchor=(.047, .883), ncol=2, frameon=False, fontsize=12)
    for row, order in enumerate(('random', 'ordered')):
        for col, strategy in enumerate(('baseline', 'once')):
            audit, curves = cases[f'{strategy}_{order}']
            metric, totals = audit['metrics'][0], audit['totals']
            ax = axes[row, col]
            plot_curves(ax, curves)
            ax.set_title(f'{order.title()} / {LABELS[strategy]}    U={metric["U_percent"]:.2f}%    SLO×1.5={metric["admission_clock"]["percent"]:.2f}%\n整窗平均需求 {totals["sum_mean_demand_GiB_s"]:.3f}；供给 {totals["sum_mean_supply_GiB_s"]:.3f} GiB/s', fontsize=12, loc='left', color=source.INK, pad=10)
            ax.tick_params(labelsize=10)
    fig.text(.05, .070, '需求为当前请求 V/C 之和；供给按每卡层周期（含等待）平均后相加，包含跨请求及窗口截断段。', fontsize=12, color=source.MUTED)
    fig.text(.05, .032, '曲线面积核对实际收到量；两线之比不能作为 U。SLO：窗口内接纳请求，完成延迟 ≤ 1.5 × 8 层纯计算时间。', fontsize=12, color=source.MUTED)
    return save_figure(fig, FIGURES / 'fleet_total_bandwidth_curves.png')


def assemble():
    cases = {name: load_case(name) for name in CASES}
    prior = json.loads((HERE / 'prior_figures_sha256.json').read_text())
    for path, digest in prior.items():
        assert source.sha(ROOT / path) == digest, path
    images = [draw_individual(*cases[name]) for name in CASES]
    images.append(draw_comparison(cases))
    results = [row for name in CASES for row in cases[name][0]['metrics']]
    flat = []
    for row in results:
        a, A, B = row['admission_clock'], row['per_role']['A'], row['per_role']['B']
        flat.append({key: row[key] for key in ('order', 'strategy', 'start_ms', 'end_ms', 'U_percent')} |
                    dict(slo_percent=a['percent'], passed=a['passed'], count=a['count'],
                         A_slo_percent=A['percent'], A_passed=A['passed'], A_count=A['count'],
                         B_slo_percent=B['percent'], B_passed=B['passed'], B_count=B['count'],
                         completed_after_window=row['completed_after_window']))
    with (HERE / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    per_npu = [{key: row[key] for key in ('order', 'strategy', 'start_ms', 'end_ms')} |
               {key: lane[key] for key in ('npu', 'U_percent', 'compute_ms', 'active_ms')} |
               dict(computed_roles=','.join(lane['computed_roles']))
               for row in results for lane in row['per_npu']]
    with (HERE / 'per_npu_utilization.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_npu[0]))
        writer.writeheader()
        writer.writerows(per_npu)
    thresholds = cases['once_random'][0]['thresholds_ms']
    audit = dict(all_checks_passed=True, num_npu=32, num_ssu=4, seed=7, alpha=ALPHA,
                 thresholds_ms=thresholds, original_figures_preserved=len(prior), images=images,
                 case_checks={name: str((HERE/'analysis'/f'{name}.json').relative_to(ROOT)) for name in CASES},
                 builder_sha256=source.sha(Path(__file__)), results=results,
                 metric='8-layer prefill completion minus admission; TTFT proxy, not a measured first-token event',
                 cohort='Admitted within the half-open window; followed through final completion')
    write_json(HERE / 'comparison.json', audit)
    report = ['# SSU4：Once per layer 利用率、TTFT SLO×1.5 和整机带宽', '',
              '配置：32 NPU、4 SSU × 40 GiB/s、每卡接收链路 50 GiB/s、seed 7。每卡 40A＋80B、8 层、batch=1；全部请求在 t=0 到达。Random 为各卡独立洗牌，Ordered 为 ABB 重复 40 轮。', '',
              '本次新增 Random、Ordered 两次完整 Once 仿真，每次完成 3840 个请求、15052800 个 I/O 块。每组直接复制对应 SSU4 Baseline 的冻结输入，输入 SHA 和核心代码 SHA 均一致，固定 NPU 绑定、落盘与请求顺序。所有数值为 seed 7。', '',
              "Once per layer 沿用 `strategy='once'`：最近一次 5 ms 共享采样快照，每请求/层/SSU 一次规划全部 I/O 块路径。沿用类别允许路径、静态 CIR 和跨请求首层预取。控制通信和规划 CPU 的仿真延迟均为 0；Python 运行时间不计入 NPU 时序。", '',
              '利用率 = 窗口内真实计算卡时间 / (32 × 窗口时长)。TTFT 沿用项目的接纳计时口径：8 层 prefill 完成时间 − 接纳时间，是首 token 延迟的代理，不含接纳前排队。', '',
              f"SLO×1.5 达标条件：延迟 ≤ 1.5 × 8 × 原始每层计算时间。A 阈值为 {thresholds['A']:.6f} ms；B 阈值为 {thresholds['B']:.6f} ms。样本为窗口内接纳的请求，每请求等权，跨窗完成的请求跟踪到最终完成并保留在分母中。", '']
    for end, title in ((4000., '主窗口 [2,4) 秒'), (20000., '补充窗口 [2,20) 秒')):
        report += [f'## {title}', '', '| 顺序 | 策略 | NPU 平均利用率 | TTFT SLO×1.5 达标率 | A 类达标率 | B 类达标率 |',
                   '|---|---|---:|---:|---:|---:|']
        for order in ('random', 'ordered'):
            for strategy in ('baseline', 'once'):
                r = next(x for x in results if x['order']==order and x['strategy']==strategy and x['end_ms']==end)
                def cell(s):
                    return f"{s['percent']:.2f}%（{s['passed']}/{s['count']}）"
                report.append(f"| {order.title()} | {LABELS[strategy]} | {r['U_percent']:.4f}% | {cell(r['admission_clock'])} | {cell(r['per_role']['A'])} | {cell(r['per_role']['B'])} |")
        report.append('')
    report += ['## 带宽图', '',
               '- [四种情况总览 PNG](../figures/ssu4/fleet_total_bandwidth_curves.png)',
               '- [Once Random PNG](../figures/ssu4/once_per_layer/random_total_demand_supply.png)',
               '- [Once Ordered PNG](../figures/ssu4/once_per_layer/ordered_total_demand_supply.png)',
               '- [Baseline Random PNG](../figures/ssu4/fleet_total_bandwidth_curves/random_total_demand_supply.png)',
               '- [Baseline Ordered PNG](../figures/ssu4/fleet_total_bandwidth_curves/ordered_total_demand_supply.png)', '',
               '图形沿用 SSU3 的定义与 0–1000 GiB/s 纵轴。紫色虚线为当前请求 Bi=V/C 对 32 卡求和；蓝线先按每卡层周期平均实际收到的数据量，再对 32 卡求和。周期从本层开始计算到下一层开始计算，包含 I/O 等待。跨请求周期计入，窗边片段按窗内收到量计算。', '',
               '蓝线为周期平均供给，各卡周期不对齐。其局部峰值不能用于判断磁盘是否超速，两线之比也不能用作 NPU 利用率。已核对曲线面积与整窗实际收到字节量、逐盘/逐卡物理服务不重叠、完整内部周期字节和计算时间。', '',
               '## 统计口径与复查', '',
               '相同完整输入不代表窗口内接纳到同一批请求；应连同类别达标率与分母一起比较。每卡实际计算、活跃时长及窗口内计算过的类别保存在逐卡 CSV 中。', '',
               '若对同一窗口内接纳样本改用“完成 − 外部到达”，达标率均为 0%，因为所有请求 t=0 到达。若选取窗口内新到达请求，则样本数为 0，达标率为 N/A。', '',
               '[精确 CSV](comparison.csv) · [逐卡利用率 CSV](per_npu_utilization.csv) · [统计与校验 JSON](comparison.json) · [原始运行](runs/) · [复算脚本](analyze.py)', '',
               '复现：先运行 `python run_once.py --order random` 和 `python run_once.py --order ordered`（运行目录已存在时拒绝覆盖）；再运行 `python analyze.py --cases baseline_random baseline_ordered once_random once_ordered --assemble`。', '']
    assert all(r['arrival_clock_same_cohort']['passed'] == 0 and r['window_arrivals']['count'] == 0 for r in results)
    (HERE / 'README.md').write_text('\n'.join(report))
    (FIGURES / 'once_per_layer/README.md').write_text('# Once per layer · 32 NPU / 4 SSU\n\n[利用率、SLO 与完整说明](../../../once_per_layer_ssu4_seed7/README.md)\n\n[Random](random_total_demand_supply.png) · [Ordered](ordered_total_demand_supply.png) · [Baseline/Once 总览](../fleet_total_bandwidth_curves.png)\n')
    print(json.dumps(dict(results=flat, images=images, original_figures_preserved=len(prior)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='*', choices=CASES, default=[])
    parser.add_argument('--assemble', action='store_true')
    args = parser.parse_args()
    for name in args.cases:
        build_case(name)
    if args.assemble:
        assemble()


if __name__ == '__main__':
    main()
