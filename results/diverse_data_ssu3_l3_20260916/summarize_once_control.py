#!/usr/bin/env python3
"""Validate six original-Once controls and publish a separate three-way report."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math

from summarize_results import read, sha, flatten
from inputs.runners.run_baseline_npu32_stress import load_manifest
from metrics import summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENARIOS = {'semi': '间歇过载', 'full': '持续过载'}
POLICIES = {'baseline': 'Baseline Random', 'once': '原始 Once per layer',
            'static': '固定候选池 + Once'}
WINDOWS = ['warm_2_4s', 'long_2_6s', 'full_population']


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    plan = read(HERE/'once_control_plan.json')
    complete = []
    profile_rows = []
    cases = []
    for job in plan['jobs']:
        folder = HERE/'runs'/job['label']
        c = read(folder/'command.json')
        assert c['status'] == 'complete', (job['label'], c['status'])
        assert c['policy'] == 'once' and not c['pilot'] and not c['smoke']
        assert c['once_control']['checks_passed']
        assert c['once_control']['new_candidate_pool_extension_installed'] is False
        assert c['once_control']['runner_sha256'] == plan['wrapper_sha256'] == sha(HERE/'run_once_control.py')
        assert c['core_unchanged'] and c['extension_unchanged'] and all(c['checks'].values())
        assert c['expected_blocks'] == c['observed_blocks']
        for name, digest in c['core_source_sha256'].items():
            assert sha(ROOT/name) == digest
        for name, digest in c['extension_source_sha256'].items():
            assert sha(HERE/name) == digest
        manifest = folder/'manifest.json.gz'
        key = str((HERE/'inputs'/f"{c['scenario']}_seed{c['seed']}.json.gz").relative_to(ROOT))
        assert sha(manifest) == c['manifest_sha256'] == plan['inputs'][key]
        requests, meta = load_manifest(manifest)
        raw_path = folder/'result.json.gz'
        assert sha(raw_path) == c['result_sha256']
        raw = read(raw_path)
        assert raw['strategy'] == 'once'
        assert raw['input_fingerprint'] == meta['input_fingerprint']
        assert len(raw['summary']['request_metrics']) == len(requests) == c['completed_requests']
        assert all(raw['summary']['invariants'].values())
        adapter = raw['adapter_statistics']
        assert adapter['strategy'] == 'once' and adapter['routing_calls'] > 0
        assert adapter['collector_interval_ms'] == 5 and adapter['max_snapshot_age_ms'] <= 5+1e-7
        assert adapter['reorder_calls'] == adapter['assignment_count'] == 0
        assert adapter['cir_write_events'] == []
        assert raw['routing_statistics']['calls'] == raw['routing_statistics']['io_blocks'] == 0
        for i, w in enumerate(raw['analysis']):
            measured = summarize(raw['summary'], requests, w['start_ms'], w['end_ms'], full=i == 2)
            assert abs(measured['U_percent']-w['U_percent']) < 1e-7
            assert measured['slo'] == w['slo'] and measured['demand'] == w['demand']
            assert measured['slo_by_category'] == w['slo_by_category']
            if i < 2:
                assert measured['all_npus_active']
            r = flatten(c, w, WINDOWS[i])
            r.update(makespan_ms=raw['summary']['makespan_ms'],
                     manifest_sha256=c['manifest_sha256'], result_sha256=c['result_sha256'])
            complete.append(r)
            for profile, result in w['slo_by_profile'].items():
                profile_rows.append(dict(scenario=c['scenario'], policy='once', seed=c['seed'],
                                         window=WINDOWS[i], profile=profile, **result))
        cases.append(dict(label=job['label'], manifest_sha256=c['manifest_sha256'],
                          result_sha256=c['result_sha256'], requests=c['completed_requests'],
                          blocks=c['observed_blocks'], routing_calls=adapter['routing_calls'],
                          max_snapshot_age_ms=adapter['max_snapshot_age_ms'],
                          wrapper_checks=True, core_unchanged=True))

    assert len(cases) == 6 and len(complete) == 18
    groups = defaultdict(list)
    for r in complete:
        groups[r['scenario'], r['policy'], r['window']].append(r)
    macros = []
    for (scenario, policy, window), rows in sorted(groups.items()):
        assert sorted(r['seed'] for r in rows) == [7, 19, 43]
        m = dict(scenario=scenario, policy=policy, window=window, seed_count=3,
                 seeds=[7, 19, 43])
        for field in rows[0]:
            if field.endswith(('_percent', '_per_second', '_GiB_s')) or field == 'makespan_ms':
                values = [r[field] for r in rows if r[field] is not None]
                m['mean_'+field] = math.fsum(values)/len(values) if values else None
                m['min_'+field] = min(values) if values else None
                m['max_'+field] = max(values) if values else None
        m['pooled_slo_count'] = sum(r['slo_count'] for r in rows)
        m['pooled_slo_passed'] = sum(r['slo_passed'] for r in rows)
        macros.append(m)
    csv_write(HERE/'once_control_comparison.csv', complete)
    csv_write(HERE/'once_control_macro_summary.csv', macros)
    csv_write(HERE/'once_control_profile_slo.csv', profile_rows)

    previous = list(csv.DictReader((HERE/'macro_summary.csv').open()))
    together = [r for r in previous if r['policy'] in ('baseline', 'static')] + macros
    index = {(r['scenario'], r['policy'], r['window']): r for r in together}
    assert len(index) == 18
    csv_write(HERE/'threeway_macro_summary.csv', sorted(together,
              key=lambda r: (r['scenario'], r['window'], list(POLICIES).index(r['policy']))))

    def value(s, p, w, field):
        return float(index[s, p, w]['mean_'+field])

    def table(window, categories=False):
        header = '| 场景 | 策略 | NPU平均利用率 | SLO×1.5达标率 |'
        separator = '|---|---|---:|---:|'
        if categories:
            header += ' SS达标率 | SL达标率 | LS达标率 | LL达标率 |'
            separator += '---:|---:|---:|---:|'
        lines = [header, separator]
        for s in SCENARIOS:
            for p in POLICIES:
                row = (f'| {SCENARIOS[s]} | {POLICIES[p]} | '
                       f'{value(s,p,window,"U_percent"):.2f}% | '
                       f'{value(s,p,window,"slo_percent"):.2f}% |')
                if categories:
                    row += ' ' + ' | '.join(f'{value(s,p,window,c+"_slo_percent"):.2f}%'
                                           for c in ('SS','SL','LS','LL')) + ' |'
                lines.append(row)
        return '\n'.join(lines)

    deltas = []
    for s in SCENARIOS:
        du = value(s,'static','warm_2_4s','U_percent')-value(s,'once','warm_2_4s','U_percent')
        ds = value(s,'static','warm_2_4s','slo_percent')-value(s,'once','warm_2_4s','slo_percent')
        deltas.append(f'{SCENARIOS[s]}：在原始Once之上增加固定候选池，warm NPU利用率变化{du:+.2f}个百分点，SLO达标率变化{ds:+.2f}个百分点。')
    ll = [value('full',p,'warm_2_4s','LL_slo_percent') for p in POLICIES]
    original_rows = list(csv.DictReader((HERE/'comparison.csv').open()))
    paired = []
    for seed in (7, 19, 43):
        old = next(r for r in original_rows if r['scenario'] == 'full'
                   and r['policy'] == 'static' and r['window'] == 'warm_2_4s'
                   and int(r['seed']) == seed)
        new = next(r for r in complete if r['scenario'] == 'full'
                   and r['window'] == 'warm_2_4s' and r['seed'] == seed)
        paired.append(f"seed{seed}：{float(old['slo_percent'])-new['slo_percent']:+.2f}个百分点")
    report = f'''**Baseline、原始Once、固定候选池：相同24画像输入的独立对照**

补跑6次原始Once完整仿真（两种负载×seed7/19/43）。之前的24次结果保持不变。本表的Once不是旧两画像实验的数字，也不是mild或aggressive改名。

配置：32 NPU、3 SSU×40 GiB/s、NPU链路50 GiB/s、8层、batch=1。原六份manifest字节与输入顺序不变，所有策略均为Random输入。每个种子先算比例，再三个种子等权平均。

原始Once使用该类别完整合法Path池，在每层规划时根据每5ms更新的拥塞快照逐块选路。固定候选池策略在其前面增加候选范围限制。两者的FIFO、CIR/PIR、NPU绑定及卡内请求顺序相同。

**主结果：warm [2,4)秒**

{table('warm_2_4s')}

SLO沿用实验口径：`prefill完成时刻 - 接纳时刻 <= 1.5 × 8 × 本请求每层纯计算时间`。窗内接纳的全部请求跟踪至完成，包括窗后完成和超时请求。它不包含接纳前等待，也不是真实首token计时。NPU利用率为窗口实际计算卡时间除以`32×2秒`，32张卡均持续有请求执行。

{chr(10).join(deltas)}

持续过载中，固定候选池相对原始Once的逐种子SLO变化为：{'；'.join(paired)}。均值差不代表每个种子均有改善；三个种子也不足以据此证明普遍或稳定收益。

**分类SLO：收益与代价**

{table('warm_2_4s', True)}

持续过载时，LL达标率在Baseline / 原始Once / 固定候选池下依次为{ll[0]:.2f}% / {ll[1]:.2f}% / {ll[2]:.2f}%。整体SLO提高不能解释为每个类别都改善。

**扩大窗口：[2,6)秒**

{table('long_2_6s')}

**整个相同有限请求集合：含开始和排空**

{table('full_population')}

完整人口每种子间歇组960请求、持续组1344请求；全部完成。各策略warm窗口内接纳的人口会不同，因此补充全请求集合对照。全程NPU利用率包含排空，不能与warm利用率混为同一指标。

来源：[Once逐种子结果](once_control_comparison.csv)、[Once三种子汇总](once_control_macro_summary.csv)、[三策略汇总](threeway_macro_summary.csv)、[原Baseline/固定池汇总](macro_summary.csv)、[补跑校验](audit/once_control_validation.json)。
'''
    (HERE/'ONCE_CONTROL_COMPARISON.md').write_text(report)
    validation = dict(complete_cases=6, completed_requests=sum(c['requests'] for c in cases),
                      completed_blocks=sum(c['blocks'] for c in cases),
                      raw_metric_recalculation=True, same_six_inputs=True,
                      original_once_without_pool_extension=True, cases=cases)
    for name, digest in plan['previous_results_sha256'].items():
        assert sha(HERE/name) == digest, ('previous_result_changed', name)
    validation['previous_results_unchanged'] = True
    (HERE/'audit'/'once_control_validation.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2)+'\n')
    print(table('warm_2_4s'))


if __name__ == '__main__':
    main()
