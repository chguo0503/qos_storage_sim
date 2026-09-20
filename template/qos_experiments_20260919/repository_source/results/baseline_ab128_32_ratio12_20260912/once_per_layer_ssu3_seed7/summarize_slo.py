#!/usr/bin/env python3
"""Recompute admission-cohort SLOs from completed requests, without simulation."""
from pathlib import Path
import csv
import json
import math
import summarize as util

HERE, STUDY, ROOT = util.HERE, util.STUDY, util.ROOT


def main():
    sources = {}
    results = []
    for order in ('random', 'ordered'):
        for strategy in ('baseline', 'once'):
            case = (STUDY/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
                    if strategy == 'baseline' else HERE/'runs'/order/'once')
            command, man, raw = [util.read(case/name) for name in
                                ('command.json', 'manifest.json.gz', 'result.json.gz')]
            assert command['status'] == 'complete' and command['strategy'] == strategy
            assert util.sha(case/'manifest.json.gz') == command['manifest_sha256']
            assert util.sha(case/'result.json.gz') == command['output_sha256']
            assert all(raw['summary']['invariants'].values())
            assert raw['input_fingerprint'] == man['input_fingerprint']
            assert man['metadata']['n_layers'] == 8 and man['metadata']['seed'] == 7
            for name in ('command.json', 'manifest.json.gz', 'result.json.gz'):
                sources[str((case/name).relative_to(ROOT))] = util.sha(case/name)
            reqs = {q['request_id']: q for q in man['requests']}
            complete = raw['summary']['request_metrics']
            assert len(complete) == len(reqs) == 3840
            assert {q['request_id'] for q in complete} == set(reqs)
            for q in complete:
                ideal = 8 * reqs[q['request_id']]['load']['per_layer_us']/1000
                assert math.isclose(q['own_compute_ms'], ideal, abs_tol=1e-8)
                assert math.isfinite(q['completion_time_ms'])
            for left, right in ((2000., 4000.), (2000., 20000.)):
                sample = [q for q in complete if left <= q['admission_time_ms'] < right]
                def stats(rows, clock):
                    passed = sum(q['completion_time_ms']-q[clock+'_time_ms'] <=
                                 1.5*8*reqs[q['request_id']]['load']['per_layer_us']/1000+1e-9
                                 for q in rows)
                    return dict(passed=passed, count=len(rows),
                                percent=100*passed/len(rows) if rows else None)
                by_role = {role: stats([q for q in sample if reqs[q['request_id']]['load']['role'] == role], 'admission')
                           for role in ('A', 'B')}
                admission, arrival = stats(sample,'admission'), stats(sample,'arrival')
                arrivals = [q for q in complete if left <= q['arrival_time_ms'] < right]
                assert sum(g['count'] for g in by_role.values()) == admission['count']
                assert sum(g['passed'] for g in by_role.values()) == admission['passed']
                if right == 4000.:
                    reference = raw['slo']['window_admissions']
                    assert raw['slo']['alpha'] == 1.5
                    assert set(reference['request_ids']) == {q['request_id'] for q in sample}
                    for clock, values in (('admission',admission),('arrival',arrival)):
                        assert values['passed'] == reference[clock]['passed']
                        assert values['count'] == reference[clock]['count']
                compute = math.fsum(util.clip(l['compute_start_ms'],l['compute_end_ms'],left,right)
                                    for b in raw['summary']['microbatch_metrics'] for l in b['layer_metrics'])
                results.append(dict(order=order, strategy=strategy, start_ms=left, end_ms=right,
                    U_percent=100*compute/(32*(right-left)), admission_clock=admission,
                    arrival_clock_same_cohort=arrival, per_role=by_role,
                    completed_after_window=sum(q['completion_time_ms'] > right for q in sample),
                    window_arrivals=stats(arrivals,'arrival'), request_ids=sorted(q['request_id'] for q in sample)))
    thresholds = {q['load']['role']:1.5*8*q['load']['per_layer_us']/1000 for q in reqs.values()}
    audit = dict(all_checks_passed=True, no_new_simulation=True, num_npu=32,num_ssu=3,seed=7,
                 alpha=1.5, thresholds_ms=thresholds, sources=sources,
                 builder_sha256=util.sha(Path(__file__)),
                 cohort='Requests admitted inside the stated half-open window, followed to final completion',
                 metric='8-layer prefill completion minus admission; proxy, not measured first-token event',
                 results=results)
    (HERE/'ttft_slo.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for row in results:
        flat.append({k:row[k] for k in ('order','strategy','start_ms','end_ms','U_percent')} |
                    dict(passed=row['admission_clock']['passed'],count=row['admission_clock']['count'],
                         slo_percent=row['admission_clock']['percent'],
                         A_slo_percent=row['per_role']['A']['percent'], B_slo_percent=row['per_role']['B']['percent'],
                         completion_after_window_count=row['completed_after_window']))
    with (HERE/'ttft_slo.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    tables=[]
    for right,label in ((4000.,'主窗口 [2,4) 秒'),(20000.,'补充窗口 [2,20) 秒')):
        lines=[]
        for row in results:
            if row['end_ms'] != right: continue
            a=row['admission_clock'];A=row['per_role']['A'];B=row['per_role']['B']
            label_strategy='Baseline' if row['strategy']=='baseline' else 'Once per layer'
            lines.append(f"| {row['order'].title()} | {label_strategy} | {row['U_percent']:.2f}% | {a['percent']:.2f}%（{a['passed']}/{a['count']}） | {A['percent']:.2f}%（{A['passed']}/{A['count']}） | {B['percent']:.2f}%（{B['passed']}/{B['count']}） |")
        tables.append(f'## {label}\n\n| 输入顺序 | 策略 | NPU平均利用率 | 总体 SLO×1.5 达标率 | A 类达标率 | B 类达标率 |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(lines))
    text='''# 四种情况的 TTFT SLO×1.5

32 NPU、3 SSU、seed 7，沿用同一批40A＋80B/卡的输入和现有完整日志；没有重跑仿真。

**本表沿用接纳计时口径，是8层prefill完成延迟的TTFT代理，不是真实首token测量，也不包含接纳前的客户端排队时间。**

```text
选取请求：接纳时间落在 [窗口起点, 窗口终点) 内
延迟：最终完成时间 − 接纳时间
达标条件：延迟 <= 1.5 × 8 × 该请求原始每层计算时间
达标率：达标请求数量 / 入选请求数量
```

A 阈值约72.288 ms，B阈值约343.114 ms；不是直接把data中的78层TTFT乘1.5。每条请求权重相同，没有按计算时间加权。窗口末尾仍未完成的请求跟踪到最终完成；这8组样本各有32条在窗末后完成，均保留在分母中。

'''+ '\n\n'.join(tables)+'''

## 两个不能混淆的统计

- 对上表相同的“窗口内接纳”请求，若改为“最终完成时间−外部到达时间”，四种情况在两个窗口内的达标率都为0%。全部请求在t=0到达，这些请求在2秒以后才被接纳，已经超过各自阈值。
- 若选取“窗口内新到达”的请求，则两个窗口都没有样本，达标率为N/A，不能写成0%。

## 比较时的限制

相同完整输入不代表窗口内接纳的请求集合相同，调度会改变推进速度与A/B样本比例。Ordered warm中，Baseline样本B占160/224，Once样本B占210/319；尽管A达标率由0%提高到25.69%，总体达标率还受类别比例变化影响。应同时查看总率、类别率及分母，不能仅凭总率判断某算法普遍更优。

[精确CSV](ttft_slo.csv) · [源文件hash与入选请求ID](ttft_slo.json) · [复算脚本](summarize_slo.py) · [利用率与图表报告](README.md)
'''
    (HERE/'TTFT_SLO对比.md').write_text(text)
    print(json.dumps(flat,ensure_ascii=False))


if __name__=='__main__':
    main()
