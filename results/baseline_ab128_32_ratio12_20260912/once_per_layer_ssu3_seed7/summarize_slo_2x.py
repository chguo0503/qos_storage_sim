#!/usr/bin/env python3
"""Change only the SLO multiplier; retain each original admission cohort."""
from pathlib import Path
import csv
import json
import math
import summarize as util

HERE, STUDY, ROOT = util.HERE, util.STUDY, util.ROOT


def main():
    previous = util.read(HERE/'ttft_slo.json')
    assert previous['all_checks_passed'] and previous['alpha'] == 1.5
    sources = dict(previous['sources'])
    for name,digest in sources.items():
        assert util.sha(ROOT/name) == digest, name
    sources[str((HERE/'ttft_slo.json').relative_to(ROOT))] = util.sha(HERE/'ttft_slo.json')
    cases = {}
    results = []
    thresholds = {}
    for old in previous['results']:
        order,strategy = old['order'],old['strategy']
        if (order,strategy) not in cases:
            case = (STUDY/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
                    if strategy=='baseline' else HERE/'runs'/order/'once')
            raw,man = util.read(case/'result.json.gz'),util.read(case/'manifest.json.gz')
            cases[order,strategy] = raw, {q['request_id']:q for q in man['requests']}
        raw,reqs = cases[order,strategy]
        left,right = old['start_ms'],old['end_ms']
        sample = [q for q in raw['summary']['request_metrics'] if left <= q['admission_time_ms'] < right]
        assert {q['request_id'] for q in sample} == set(old['request_ids'])
        def stats(rows,alpha,clock='admission'):
            passed=0
            for q in rows:
                load=reqs[q['request_id']]['load']
                ideal=8*load['per_layer_us']/1000
                assert math.isclose(q['own_compute_ms'],ideal,abs_tol=1e-8)
                thresholds[load['role']]=2*ideal
                passed += q['completion_time_ms']-q[clock+'_time_ms'] <= alpha*ideal+1e-9
            return dict(passed=passed,count=len(rows),percent=100*passed/len(rows) if rows else None)
        original=stats(sample,1.5)
        assert original == old['admission_clock']
        new=stats(sample,2.)
        assert new['passed'] >= original['passed']
        roles={role:stats([q for q in sample if reqs[q['request_id']]['load']['role']==role],2.) for role in ('A','B')}
        assert sum(v['passed'] for v in roles.values()) == new['passed']
        assert sum(v['count'] for v in roles.values()) == new['count']
        results.append(dict(order=order,strategy=strategy,start_ms=left,end_ms=right,
            U_percent=old['U_percent'],slo_1p5=original,slo_2=new,per_role=roles,
            arrival_clock_same_cohort=stats(sample,2.,'arrival'),
            window_arrivals_count=sum(left <= q['arrival_time_ms'] < right for q in raw['summary']['request_metrics']),
            completed_after_window=sum(q['completion_time_ms'] > right for q in sample),request_ids=old['request_ids']))
    report=dict(all_checks_passed=True,no_new_simulation=True,alpha=2.,num_npu=32,num_ssu=3,seed=7,
        thresholds_ms=thresholds,cohorts_identical_to_1p5=True,sources=sources,
        builder_sha256=util.sha(Path(__file__)),results=results)
    (HERE/'ttft_slo_2x.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for r in results:
        flat.append({k:r[k] for k in ('order','strategy','start_ms','end_ms','U_percent')} |
            dict(slo_1p5_percent=r['slo_1p5']['percent'],slo_2_percent=r['slo_2']['percent'],
                 passed=r['slo_2']['passed'],count=r['slo_2']['count'],
                 A_slo_2_percent=r['per_role']['A']['percent'],B_slo_2_percent=r['per_role']['B']['percent']))
    with (HERE/'ttft_slo_2x.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    parts=[]
    for end,label in ((4000.,'主窗口 [2,4) 秒'),(20000.,'补充窗口 [2,20) 秒')):
        lines=[]
        for r in results:
            if r['end_ms'] != end:continue
            name='Baseline' if r['strategy']=='baseline' else '流量分配策略'
            s=r['slo_2'];a=r['per_role']['A'];b=r['per_role']['B']
            lines.append(f"| {r['order'].title()} | {name} | {r['slo_1p5']['percent']:.2f}% | {s['percent']:.2f}%（{s['passed']}/{s['count']}） | {a['percent']:.2f}%（{a['passed']}/{a['count']}） | {b['percent']:.2f}%（{b['passed']}/{b['count']}） |")
        parts.append(f'## {label}\n\n| 输入 | 策略 | SLO×1.5 | SLO×2 | A类 SLO×2 | B类 SLO×2 |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(lines))
    text='''# TTFT SLO×2 对比

32 NPU、3 SSU、seed 7。只把阈值倍率从1.5改为2，输入、仿真轨迹和各组入选请求完全不变，NPU利用率也不变。“流量分配策略”是原Once per layer的展示名称，算法实现仍为`strategy=once`。

选取接纳时间落在指定窗口内的请求，跟踪到最终完成，包含窗末后才完成的32个请求。每条请求权重相同。

```text
达标条件：最终完成时间 − 接纳时间 <= 2 × 8 × 原始每层计算时间
```

'''+f"A阈值为{thresholds['A']:.6f} ms，B阈值为{thresholds['B']:.6f} ms。"+'''
这是不含接纳前排队的8层prefill完成延迟代理，不是真实首token事件。若对同批请求改用外部到达计时，四组两个窗口的达标率仍都为0%；窗口内新到达请求数为0，该空样本的达标率为N/A。

'''+ '\n\n'.join(parts)+'''

各策略的窗口接纳样本集合可能不同，应连同A/B分项与人数一起比较。倍率变大只代表判定标准放宽，不代表运行更快。

[精确CSV](ttft_slo_2x.csv) · [来源与入选请求ID](ttft_slo_2x.json) · [复算脚本](summarize_slo_2x.py) · [原×1.5报告](TTFT_SLO对比.md)
'''
    (HERE/'TTFT_SLO乘2对比.md').write_text(text)
    print(json.dumps(flat,ensure_ascii=False))


if __name__=='__main__':
    main()
