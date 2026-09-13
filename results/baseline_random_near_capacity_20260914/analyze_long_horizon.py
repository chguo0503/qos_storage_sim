#!/usr/bin/env python3
"""Read completed >=65s-pure-work cases; never simulate or extend saved IO bins."""
from pathlib import Path
from argparse import ArgumentParser
import json
import math
import statistics
import analyze

HERE=Path(__file__).resolve().parent


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=HERE/'reference_long65_plan.json')
    args=parser.parse_args()
    plan_path=args.plan.resolve()
    plan=analyze.read(plan_path)
    destination=Path(plan.get('comparison_output',HERE))
    runs_base=Path(plan.get('runs_base',HERE/'runs'))
    manifest=Path(plan['manifest'])
    assert analyze.sha(manifest)==plan['manifest_sha256']
    label=analyze.read(manifest)['metadata']['label']
    windows=[tuple(map(float,w)) for w in plan['extra_windows_ms']]
    # Regular two-second diagnostic bins are exhaustive, not selected minima.
    for left in range(2000,60000,2000):
        pair=(float(left),float(left+2000))
        if pair not in windows:windows.append(pair)
    records=[];pending=[]
    for strategy in plan['strategies']:
        case=runs_base/label/strategy
        if not (case/'command.json').exists() or analyze.read(case/'command.json')['status']!='complete':
            pending.append(strategy);continue
        result=analyze.analyze_case(case,windows=windows,short_roles=plan.get('short_roles',['A']))
        assert result['all_technical_checks_passed']
        for w in result['windows']:
            if w['end_ms']>20000:assert not w['physical']['available']
        output=case/'extended_analysis.json'
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        for w in result['windows']:
            records.append(dict(strategy=strategy,start_s=w['start_ms']/1000,end_s=w['end_ms']/1000,
                U_percent=w['U_percent'],all_active=w['all_32_active'],
                mixed_cards=w['long_short_mixed_card_count'],
                slo_1p5_percent=w['slo']['alphas']['1.5']['admission']['percent'],
                slo_2_percent=w['slo']['alphas']['2.0']['admission']['percent'],
                slo_cohort=w['slo']['admission_cohort_count'],
                physical_available=w['physical']['available'],
                nominal_over_capacity_percent=w['nominal']['any_ssu_over_capacity_percent'],
                source=str(output.relative_to(HERE)),source_sha256=analyze.sha(output)))
        # 29 disjoint bins must exactly reconstruct [2,60), independent of the
        # per-request admission cohorts used for SLO.
        bins=[w for w in result['windows'] if w['end_ms']-w['start_ms']==2000]
        aggregate=next(w for w in result['windows'] if (w['start_ms'],w['end_ms'])==(2000,60000))
        assert len(bins)==29
        assert math.isclose(statistics.mean(w['U_percent'] for w in bins),aggregate['U_percent'],abs_tol=1e-9)
    value=dict(plan_sha256=analyze.sha(plan_path),
        builder_sha256=analyze.sha(Path(__file__)),analyzer_sha256=analyze.sha(HERE/'analyze.py'),
        pending=pending,rows=records,all_two_second_bins_preserved=True,
        caveat='New full population shuffle at >=65s pure horizon; not a continuation of h22000 input. Beyond20s there are no recorded physical service bins; U/SLO/coverage are measured from complete request/layer logs.')
    (destination/'long_horizon_comparison.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    lines=['# 65秒纯计算人口：扩大窗口后的同输入策略对照','',
        '每卡预先准备至少65秒纯计算工作，再完整独立Random；它不是22秒队列的同一前缀。没有提前结束仿真，也没有丢弃窗口末未完成的SLO样本。', '',
        '| 策略 | 观察窗 | NPU平均U | SLO×1.5 | SLO×2 | 接纳样本 | 长短覆盖卡数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        if (r['start_s']*1000,r['end_s']*1000) not in [tuple(w) for w in plan['extra_windows_ms']]:continue
        name='Baseline' if r['strategy']=='baseline' else '流量分配策略'
        lines.append(f"| {name} | {r['start_s']:g}–{r['end_s']:g}s | {r['U_percent']:.2f}% | {r['slo_1p5_percent']:.2f}% | {r['slo_2_percent']:.2f}% | {r['slo_cohort']} | {r['mixed_cards']}/32 |")
    lines+=['','SLO沿用接纳后prefill完成代理。20秒以后的实际SSD/链路带宽没有被原观测器保存，因此不填推测值；NPU计算与等待时序来自完整日志。全部29个连续2秒分窗保存在JSON，并核验其计算利用率平均等于2–60秒结果。']
    if pending:lines+=['',f'仍待完成的策略：{", ".join(pending)}。']
    (destination/'long_horizon_comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed_strategies=len(plan['strategies'])-len(pending),pending=pending)))


if __name__=='__main__':main()
