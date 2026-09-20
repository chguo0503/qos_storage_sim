#!/usr/bin/env python3
"""Paired fixed-window utilization, computed directly from complete layer logs."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip(a, z, left, right):
    return max(0., min(z, right)-max(a, left))


def main():
    results = []
    sources = {}
    for order in ('random', 'ordered'):
        baseline = STUDY/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
        frozen_sha = sha(baseline/'manifest.json.gz')
        baseline_command = read(baseline/'command.json')
        for strategy, case in (('baseline', baseline), ('once', HERE/'runs'/order/'once')):
            command = read(case/'command.json')
            assert command['status'] == 'complete' and command['completed_simulation']
            assert command['strategy'] == strategy and command['assignment'] == 'fixed'
            assert command['core_source_sha256'] == baseline_command['core_source_sha256']
            assert sha(case/'manifest.json.gz') == command['manifest_sha256'] == frozen_sha
            assert sha(case/'result.json.gz') == command['output_sha256']
            man = read(case/'manifest.json.gz')
            raw = read(case/'result.json.gz')
            assert all(raw['summary']['invariants'].values())
            assert raw['input_fingerprint'] == man['input_fingerprint']
            reqs = {q['request_id']: q for q in man['requests']}
            for name in ('command.json', 'manifest.json.gz', 'result.json.gz'):
                path = case/name
                sources[str(path.relative_to(ROOT))] = sha(path)
            for left, right in ((2000., 4000.), (2000., 20000.)):
                per_npu = []
                classes = {role: dict(compute_ms=0., active_ms=0.) for role in ('A', 'B')}
                for npu in range(32):
                    batches = sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id'] == npu),
                                     key=lambda b: b['admission_time_ms'])
                    compute = active = 0.
                    roles = set()
                    for batch in batches:
                        assert len(batch['member_request_ids']) == 1
                        request = reqs[batch['member_request_ids'][0]]
                        role = request['load']['role']
                        assert request['npu_id'] == npu
                        duration = clip(batch['admission_time_ms'], batch['completion_time_ms'], left, right)
                        work = math.fsum(clip(l['compute_start_ms'], l['compute_end_ms'], left, right)
                                         for l in batch['layer_metrics'])
                        compute += work
                        active += duration
                        classes[role]['compute_ms'] += work
                        classes[role]['active_ms'] += duration
                        if work > 0:
                            roles.add(role)
                    assert 0 <= compute <= active+1e-7 <= right-left+2e-7
                    per_npu.append(dict(npu=npu, compute_ms=compute, active_ms=active,
                                        U_percent=100*compute/(right-left), computed_roles=sorted(roles)))
                U = math.fsum(p['U_percent'] for p in per_npu)/32
                reference = next(w for w in raw['windows'] if w['start_ms'] == left and w['end_ms'] == right)
                assert math.isclose(U, 100*reference['mean_npu_utilization'], abs_tol=1e-8)
                for values in classes.values():
                    values['conditional_U_percent'] = 100*values['compute_ms']/values['active_ms']
                    values['window_card_time_share_percent'] = 100*values['active_ms']/(32*(right-left))
                    values['U_contribution_pp'] = 100*values['compute_ms']/(32*(right-left))
                results.append(dict(order=order, strategy=strategy, start_ms=left, end_ms=right,
                                    U_percent=U, per_npu=per_npu, classes=classes,
                                    all_cards_active=all(math.isclose(p['active_ms'], right-left, abs_tol=1e-7) for p in per_npu),
                                    mixed_card_count=sum(p['computed_roles'] == ['A','B'] for p in per_npu)))
    previous = read(HERE/'prior_figures_sha256.json')
    for path, digest in previous.items():
        assert sha(ROOT/path) == digest, path
    audit = dict(all_checks_passed=True, identical_input_per_order=True,
                 original_figures_preserved=len(previous), num_npu=32, num_ssu=3, seed=7,
                 sources=sources, builder_sha256=sha(Path(__file__)), results=results)
    (HERE/'comparison.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    fields = ('order','strategy','start_ms','end_ms','U_percent','all_cards_active','mixed_card_count')
    with (HERE/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    def lookup(order, strategy, end):
        return next(r for r in results if r['order'] == order and r['strategy'] == strategy and r['end_ms'] == end)
    sections = []
    for end, label in ((4000., '主统计窗口 [2,4) 秒'), (20000., '补充长窗口 [2,20) 秒')):
        rows = []
        for order in ('random', 'ordered'):
            a, b = lookup(order,'baseline',end), lookup(order,'once',end)
            rows.append(f"| {order.title()} | {a['U_percent']:.4f}% | {b['U_percent']:.4f}% | {b['U_percent']-a['U_percent']:+.4f} |")
        sections.append(f"## {label}\n\n| 请求顺序 | Baseline U | Once per layer U | Once − Baseline，百分点 |\n|---|---:|---:|---:|\n"+'\n'.join(rows))
    conditions=[]
    for order in ('random','ordered'):
        r=lookup(order,'once',4000.)
        conditions.append(f"- {order.title()} Once：全32卡整窗有任务={r['all_cards_active']}；窗口内计算过A和B的卡数={r['mixed_card_count']}/32。")
    details = []
    for order in ('random','ordered'):
        for strategy in ('baseline','once'):
            row=lookup(order,strategy,4000.)
            for role,values in row['classes'].items():
                details.append(f"| {order.title()} | {strategy} | {role} | {values['conditional_U_percent']:.4f}% | {values['window_card_time_share_percent']:.4f}% |")
    report = '''# 相同输入下的 Baseline / Once per layer 对照

配置：32 NPU、3 SSU × 40 GiB/s、每卡接收链路50 GiB/s、seed 7。每卡40A＋80B，8层，batch=1，全部请求t=0到达，固定NPU绑定和数据落盘。Random各卡独立洗牌；Ordered各卡按ABB重复40轮。沿用跨请求首层预取。

Once per layer对应现有`strategy='once'`：使用最近一次5ms共享采样快照，每请求/层/SSU一次规划全部I/O块路径，沿用类别允许路径及静态CIR。不使用`once_native`或`new_once`；不改变L1分卡或L2请求顺序。

两种策略均未计入控制通信和规划CPU的仿真延迟（设为0）；Python运行耗时不计入NPU计算时序。静态CIR不是不能借用的带宽上限，空闲服务份额可分配给有积压的Path；Once没有额外物理带宽。

每种顺序的Once直接复制原Baseline的manifest，字节级SHA相同；核心源码也相同。两次完整有限请求仿真均执行到全部块完成，主图只显示预定warm窗口[2,4)秒。所有数字均为seed 7，不能当作多种子均值。

'''+ '\n\n'.join(sections) + '''

利用率统一按窗口内真实计算卡时间 / (32 × 窗口时长)，包含I/O等待与窗口边界。差值使用百分点，不是相对百分比。

## 两张32卡带宽图

- [Random — Once per layer](figures/random_all_32npu_layer_average.png)
- [Ordered — Once per layer](figures/ordered_all_32npu_layer_average.png)
- [原Random — Baseline](../figures/ssu3/random_all_32npu_layer_average.png)
- [原Ordered — Baseline](../figures/ssu3/ordered_all_32npu_layer_average.png)

每行一张卡。紫虚线为当前请求的Bi=V/C；蓝线为同一请求内部完整层周期内实际收到的下一层数据量/周期时长，周期包含等待。跨请求和窗口截断段标灰、不填蓝线，不代表供给为零。左右的利用率与带宽数值完整统计[2,4)秒；右侧供给包含灰区真实收到的字节。两个整窗带宽均值相除不等于利用率。

'''+ '\n'.join(conditions)+'''

## 主窗口的请求类别统计

此处类别利用率=该类实际计算时间/该类已接纳占用卡时间，包括首层交接等待；不是图中的完整内部周期比值。

| 顺序 | 策略 | 类别 | 类别利用率 | 窗口卡时间占比 |
|---|---|---|---:|---:|
'''+ '\n'.join(details)+'''

输入保持一致不代表窗口内执行到同一批请求：调度改变推进速度和A/B驻留比例。此实验存在名义需求超限，不能称为逐盘逐时刻欠载对照；结果不能推广为任意输入下某策略更优。

[精确数值CSV](comparison.csv) · [来源和校验JSON](comparison.json) · [运行脚本](run_once.py) · [绘图脚本](render_fleet.py)
'''
    (HERE/'README.md').write_text(report)
    print(json.dumps([{k:r[k] for k in fields} for r in results],ensure_ascii=False))


if __name__ == '__main__':
    main()
