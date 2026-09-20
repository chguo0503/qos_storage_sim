#!/usr/bin/env python3
"""Build comparable plotting data from completed runs; never run a simulation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POLICIES = ('asu_baseline', 'od_baseline')
SEEDS = (7, 19, 43)
START, END = 2000.0, 4000.0
SPECS = (
    ('full', '持续过载', 'od_baseline_diverse_ssu3_20260918', 'scenario', 'full',
     '24种画像；每种长度的 miss 256/1024/2048/4096 数量为3/2/1/1；每卡42条'),
    ('under', '持续欠载', 'continuous_underload_asu_od_20260918', 'order', 'random',
     '10种画像；32/64/80/128/160K × miss 2048/4096，各2条；每卡20条'),
    ('semi', '局部欠载', 'od_baseline_diverse_ssu3_20260918', 'scenario', 'semi',
     '间歇过载配比：24种画像；每种长度的 miss 256/1024/2048/4096 数量为1/1/1/2；每卡30条'),
)


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(macros):
    names = {'asu_baseline': 'ASU Baseline', 'od_baseline': 'OD Baseline'}
    table = ['| 负载 | 策略 | NPU利用率 | SLO×1.5达标率 | 平均总带宽需求 GiB/s | 平均实际供给 GiB/s |',
             '|---|---|---:|---:|---:|---:|']
    status = ['| 负载 | 策略 | 全部盘同时欠载的时间 | 至少一盘过载的时间 | 全部盘同时过载的时间 |',
              '|---|---|---:|---:|---:|']
    for r in macros:
        table.append(f"| {r['label']} | {names[r['policy']]} | {r['mean_U_percent']:.2f}% | "
                     f"{r['mean_slo_1p5_percent']:.2f}% | {r['mean_total_mean_demand_GiB_s']:.2f} | "
                     f"{r['mean_total_actual_supply_GiB_s']:.2f} |")
        status.append(f"| {r['label']} | {names[r['policy']]} | {r['mean_all_disks_underload_percent']:.2f}% | "
                      f"{r['mean_any_disk_overload_percent']:.2f}% | {r['mean_all_disks_overload_percent']:.2f}% |")
    text = f'''# 三类负载：ASU 与 OD 对比

复用18次已完成的正式运行，未运行新仿真、未修改输入或原有图像。这里只比较 ASU Baseline 和 OD Baseline。

共同配置：**32 NPU、3 SSU×40 GiB/s、Ring hash、Random、8层、batch=1，warm `[2,4)` 秒**。
CDF及指标对seed 7、19、43等权平均；带宽曲线使用seed 7，保留实际时序，不把多个seed的曲线混合平滑。

## 对比图

![三类负载总览](figures/overview.png)

- [总览：三类负载 × CDF、利用率、ASU与OD带宽](figures/overview.png)
- [TTFT代理归一化CDF：完整长尾](figures/ttft_normalized_cdf.png)
- [TTFT代理归一化CDF：SLO阈值附近放大](figures/ttft_normalized_cdf_zoom.png)
- [NPU平均利用率：均值与三个种子的分布](figures/npu_utilization.png)
- [整机总带宽：需求与实际供给](figures/total_bandwidth.png)
- [ASU：三种负载的逐盘带宽](figures/asu_per_ssu_bandwidth.png)
- [OD：三种负载的逐盘带宽](figures/od_per_ssu_bandwidth.png)

## 数字对照

{chr(10).join(table)}

表中带宽也是三个种子各自窗口均值的等权平均，和只显示seed 7的时序图标注可能不同。

持续过载时，两策略平均利用率约60%，但OD的SLO×1.5达标率低于ASU；隔离路径不保证某个截止阈值下的达标率提高。持续欠载时，两策略的读取都被计算覆盖，CDF在归一化耗时1处达到100%，曲线完全重合。局部欠载时，整体利用率仍高于98%，但有部分请求超出1.5倍门槛。

## 三种负载如何定义

**这里的“持续”限定为所比较的 `[2,4)` 秒窗口**，不把有限请求全部结束后的排空阶段称为持续过载。

```text
SSU s 的参考需求 D_s(t) = 所有当前已接纳请求的 sum(每层落在盘s的读取量 / 每层纯计算时间)
持续过载：窗口内每一时刻，每张盘 D_s > 40 GiB/s
持续欠载：窗口内每一时刻，每张盘 D_s < 40 GiB/s
局部欠载：窗口内既有所有盘都欠载的阶段，也有至少一张盘过载的阶段
```

本次“局部欠载”对应旧实验的“间歇过载”配比，指时间上交替出现两种状态。不是特指固定某一块盘始终欠载。实际选中的事件区间没有恰好等于40的边界情况。

{chr(10).join(status)}

逐盘每个请求切换区间均已核对，分类不是依据目录名或10ms曲线抽样判断。仅看到整机总需求低于120，也不能推出各盘都低于40；因此另外提供两张逐盘图。

## 曲线怎么读

```text
CDF横轴 = (prefill完成时间 - 请求接纳时间) / 本请求8层纯计算时间
CDF纵轴 = 不超过横轴倍数的请求比例（三个seed的经验CDF等权平均）
横轴1.5处的CDF值 = 表中的SLO×1.5达标率
NPU利用率 = 窗内全部卡实际计算时间 / (32 × 2秒)
```

窗口内接纳的请求全部跟踪至完成，保留窗后完成及超时请求。CDF局部放大不删除长尾样本、不改变分母；完整图保留全部长尾。每个seed的CDF先独立计算，再等权平均，并非把全部seed的请求直接合并。

SLO沿用前次实验的**接纳后prefill完成代理**，不是从原始到达起算的端到端TTFT，也未模拟真实首token事件。全部输入在t=0到达，接纳前排队没有计入主图。不同策略会改变接纳时间，所以同一窗口选中的具体请求集合可能不同。

用户所说的“内存诉求”在本图解释为**SSU读取带宽需求**，单位为GiB/s，不是内存容量（GiB）。橙线是当前请求的参考需求，蓝线是SSD实际服务字节除以10ms。需求按请求切换事件绘制；供给使用10ms均值。两者都不是NPU瞬时利用率。

当前请求在I/O等待期间仍保留V/C参考需求；下一请求尚未接纳的首层预取不重复叠加为第二个活跃请求，但真实预取字节计入实际供给。参考需求可大于供给，也可在单个时段小于供给；不能把两线的瞬时比值直接当成U。

## 来源与比较边界

| 负载 | 原始结果 | 输入 |
|---|---|---|
| 持续过载 | [OD多样输入实验 full](../od_baseline_diverse_ssu3_20260918/README.md) | 24画像；每种长度的miss 256/1024/2048/4096配比3/2/1/1，每卡42条，共1344条 |
| 持续欠载 | [文档原样输入 random](../continuous_underload_asu_od_20260918/README.md) | 10画像；长度32/64/80/128/160K × miss2048/4096，各2条，每卡20条，共640条 |
| 局部欠载 | [OD多样输入实验 semi](../od_baseline_diverse_ssu3_20260918/README.md) | 24画像；每种长度的miss 256/1024/2048/4096配比1/1/1/2，每卡30条，共960条 |

24画像使用长度32/64/80/128/160/200K；各画像的读取量、计算时间都来自data。三组的核心模拟器与策略源码一致。同一负载、同一seed中的ASU和OD共享完全相同的冻结输入。

**跨负载的画像配比、请求数量、请求ID布局及随机排列规则不同。** 因此这是一组现有负载的对照展示，不能把跨组三个点解释成“只改变负载率、其余全部不变”的因果实验。策略优劣应在每组相同输入内比较。

ASU每盘共享一条Path；OD每盘32条NPU独占Path，CIR均分、PIR不限，空闲带宽可借用。它同时改变隔离与份额分配，不能把差值只归因于Path数量。

## 数据与复现

- [三种子均值表](macro_summary.csv)、[18次运行逐项表](comparison.csv)、[CDF逐请求样本](request_samples.csv)
- [统一绘图数据](plot_data.json)、[输入与数值校验](data_checks.json)
- [逐盘逐事件独立审计](audit_load_regimes.json)、[绘图审计](render_checks.json)

```bash
python results/load_regimes_asu_od_comparison_20260918/prepare_comparison.py
python results/load_regimes_asu_od_comparison_20260918/render_figures.py
```

以上命令只读取已完成结果和生成本目录图表，不启动仿真。原始文件哈希保存在校验记录中。
'''
    (HERE/'README.md').write_text(text)


def main():
    output = {'config': {
        'num_npu': 32, 'num_ssu': 3, 'ssu_capacity_GiB_s': 40,
        'n_layers': 8, 'order': 'Random', 'placement': 'ring_hash',
        'start_ms': START, 'end_ms': END, 'seeds': list(SEEDS),
        'policies': list(POLICIES), 'bandwidth_plot_seed': 7,
        'cdf_normalization': '(prefill_completion-admission)/(8*layer_compute)',
        'slo_factor': 1.5, 'slo_numeric_tolerance_ms': 1e-9,
        'aggregation': 'equal mean of three seed statistics / CDFs',
        'classification_scope': 'warm [2,4) seconds, each physical SSU at every demand event interval',
        'different_input_populations_across_regimes': True,
        'no_simulation_started': True,
    }, 'conditions': []}
    sources, case_rows, request_rows = {}, [], []
    snaps = 0
    for cid, label, study, selector, value, description in SPECS:
        source = ROOT / 'results' / study
        check_path = source / 'summary_checks.json'
        check = read(check_path)
        assert check['status'] == 'complete'
        sources[str(check_path.relative_to(ROOT))] = sha(check_path)
        condition = {'id': cid, 'label': label, 'source': str(source.relative_to(ROOT)),
                     'description': description, 'cases': []}
        selected = [case for case in check['cases']
                    if case[selector] == value and case['policy'] in POLICIES]
        assert {(c['policy'], c['seed']) for c in selected} == {(p, s) for p in POLICIES for s in SEEDS}
        for seed in SEEDS:
            pair = [c for c in selected if c['seed'] == seed]
            assert pair[0]['manifest_sha256'] == pair[1]['manifest_sha256']
        for meta in sorted(selected, key=lambda c: (POLICIES.index(c['policy']), c['seed'])):
            run_dir = source / 'runs' / meta['case']
            raw_path, manifest_path = run_dir / 'result.json.gz', run_dir / 'manifest.json.gz'
            for path, expected in ((raw_path, meta['result_sha256']), (manifest_path, meta['manifest_sha256'])):
                assert sha(path) == expected, str(path)
                sources[str(path.relative_to(ROOT))] = expected
            raw = read(raw_path)
            summary = raw['summary']
            assert (summary['num_npu'], summary['num_ssu'], summary['n_layers'], summary['batch_size']) == (32, 3, 8, 1)
            window = next(a for a in raw['analysis'] if a['start_ms'] == START and a['end_ms'] == END)
            assert window['all_npus_active']
            compute = math.fsum(max(0.0, min(END, layer['compute_end_ms']) - max(START, layer['compute_start_ms']))
                                for batch in summary['microbatch_metrics'] for layer in batch['layer_metrics'])
            utilization = 100 * compute / (32 * (END-START))
            assert abs(utilization-window['U_percent']) < 1e-8
            samples, passed = [], 0
            cohort = [r for r in summary['request_metrics'] if START <= r['admission_time_ms'] < END]
            assert cohort
            for r in cohort:
                latency = r['completion_time_ms'] - r['admission_time_ms']
                ideal = r['own_compute_ms']
                ratio = latency / ideal
                plot_ratio = ratio
                for factor in (1.0, 1.5):
                    if abs(latency-factor*ideal) <= 1e-9:
                        snaps += int(plot_ratio != factor)
                        plot_ratio = factor
                ok = latency <= 1.5 * ideal + 1e-9
                assert (plot_ratio <= 1.5) == ok
                passed += ok
                samples.append(plot_ratio)
                request_rows.append(dict(condition=cid, policy=meta['policy'], seed=meta['seed'],
                                         request_id=r['request_id'], admission_ms=r['admission_time_ms'],
                                         completion_ms=r['completion_time_ms'], own_compute_ms=ideal,
                                         admission_latency_ms=latency, raw_ratio=ratio, plotted_ratio=plot_ratio,
                                         slo_1p5_pass=ok))
            slo = 100 * passed / len(cohort)
            assert (len(cohort), passed) == (window['slo']['count'], window['slo']['passed'])
            assert abs(slo-window['slo']['percent']) < 1e-8
            segments = window['demand']['segments']
            assert segments[0][0] == START and segments[-1][1] == END
            assert all(abs(a[1]-b[0]) < 1e-9 for a,b in zip(segments,segments[1:]))
            def fraction(predicate):
                return 100 * math.fsum(b-a for a,b,*d in segments if predicate(d)) / (END-START)
            all_under = fraction(lambda d: all(v < 40 for v in d))
            all_over = fraction(lambda d: all(v > 40 for v in d))
            any_over = fraction(lambda d: any(v > 40 for v in d))
            per_over = [fraction(lambda d: d[s] > 40) for s in range(3)]
            mean_demand = [math.fsum((b-a)*d[s] for a,b,*d in segments)/(END-START) for s in range(3)]
            if cid == 'full':
                assert all(min(row[2:]) > 40 for row in segments)
            elif cid == 'under':
                assert all(max(row[2:]) < 40 for row in segments)
            else:
                assert 0 < all_under < 100 and 0 < any_over < 100
            supply = raw['warm_ssd_10ms_GiB_s']
            assert len(supply)==3 and all(len(v)==200 for v in supply)
            assert all(-1e-8 <= v <= 40+1e-8 for disk in supply for v in disk)
            for s in range(3):
                assert abs(statistics.mean(supply[s])-window['SSD_GiB_s'][s]) < 1e-7
                assert abs(per_over[s]-window['demand']['per_disk_overload_percent'][s]) < 1e-7
            case = dict(policy=meta['policy'], seed=meta['seed'], U_percent=utilization, slo_percent=slo,
                        cohort_count=len(cohort), ratios=sorted(samples), demand_segments=segments,
                        supply_10ms=supply, SSD_GiB_s=window['SSD_GiB_s'],
                        per_disk_overload_percent=per_over, per_disk_mean_demand_GiB_s=mean_demand,
                        all_disks_underload_percent=all_under, all_disks_overload_percent=all_over,
                        any_disk_overload_percent=any_over, source_case=meta['case'])
            condition['cases'].append(case)
            case_rows.append(dict(condition=cid, label=label, policy=meta['policy'], seed=meta['seed'],
                                  U_percent=utilization, slo_1p5_percent=slo, cohort_count=len(cohort),
                                  all_disks_underload_percent=all_under, all_disks_overload_percent=all_over,
                                  any_disk_overload_percent=any_over,
                                  total_mean_demand_GiB_s=sum(mean_demand),
                                  total_actual_supply_GiB_s=sum(window['SSD_GiB_s']), source_case=meta['case']))
        output['conditions'].append(condition)
    macro_rows = []
    for cid,label,*_ in SPECS:
        for policy in POLICIES:
            group=[r for r in case_rows if r['condition']==cid and r['policy']==policy]
            row=dict(condition=cid,label=label,policy=policy,seeds='7,19,43')
            for field in ('U_percent','slo_1p5_percent','all_disks_underload_percent',
                          'all_disks_overload_percent','any_disk_overload_percent',
                          'total_mean_demand_GiB_s','total_actual_supply_GiB_s'):
                row['mean_'+field]=statistics.mean(r[field] for r in group)
                if field=='U_percent':
                    row['min_'+field]=min(r[field] for r in group)
                    row['max_'+field]=max(r[field] for r in group)
            macro_rows.append(row)
    output['config']['cdf_boundary_normalization_count']=snaps
    output['source_sha256']=sources
    (HERE/'plot_data.json').write_text(json.dumps(output,ensure_ascii=False,separators=(',',':'))+'\n')
    write_csv('comparison.csv',case_rows)
    write_csv('macro_summary.csv',macro_rows)
    write_csv('request_samples.csv',request_rows)
    write_report(macro_rows)
    assert all(sha(ROOT/p)==h for p,h in sources.items())
    audit=dict(status='passed',cases=18,source_files_unchanged=True,no_simulation_started=True,
               raw_compute_intervals_recalculated=True,cdf_matches_slo_per_seed=True,
               each_regime_verified_every_disk_every_event=True,
               same_frozen_input_within_each_policy_pair=True,
               all_32_npus_active_in_window=True,source_sha256=sources,
               script_sha256=sha(Path(__file__)),plot_data_sha256=sha(HERE/'plot_data.json'),
               cdf_boundary_normalization_count=snaps)
    (HERE/'data_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(macro_rows,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
