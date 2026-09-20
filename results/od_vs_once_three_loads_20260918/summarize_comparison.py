#!/usr/bin/env python3
"""Combine existing matched OD/Once runs and the new underload Once control."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'od_baseline_diverse_ssu3_20260918'
WINDOWS=('warm_2_4s','long_2_6s','full_population')
LABELS={'full':'持续过载组','under':'持续欠载组','semi':'局部欠载组'}


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    assert json.loads((HERE/'existing_evidence.json').read_text())['status']=='passed'
    assert json.loads((HERE/'under_once_results.json').read_text())['all_checks_passed']
    sources=[OLD/'macro_summary.csv',HERE/'under_once_macro.csv',HERE/'existing_evidence.json',HERE/'under_once_results.json']
    index={(r['scenario'],r['policy'],r['window']):r for r in read_csv(sources[0])}
    under={r['window']:r for r in read_csv(sources[1])}
    rows=[]
    for window in WINDOWS:
        for scenario in LABELS:
            if scenario=='under':
                r=under[window]
                values=[float(r[f'mean_{policy}_{field}']) for policy in ('OD','once')
                        for field in ('U_percent','slo_1p5_percent')]
            else:
                values=[float(index[scenario,policy,window][f'mean_{field}']) for policy in ('od_baseline','once')
                        for field in ('U_percent','slo_percent')]
            od_u,od_s,once_u,once_s=values
            rows.append(dict(scenario=scenario,window=window,seeds='7,19,43',OD_U_percent=od_u,
                             Once_U_percent=once_u,OD_SLO1p5_percent=od_s,Once_SLO1p5_percent=once_s,
                             delta_U_pp=once_u-od_u,delta_SLO_pp=once_s-od_s))
    with (HERE/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    def table(window):
        out=['| 输入组 | OD NPU利用率 | Once NPU利用率 | OD SLO×1.5 | Once SLO×1.5 |',
             '|---|---:|---:|---:|---:|']
        for r in rows:
            if r['window']==window:
                out.append(f"| {LABELS[r['scenario']]} | {r['OD_U_percent']:.2f}% | {r['Once_U_percent']:.2f}% | "
                           f"{r['OD_SLO1p5_percent']:.2f}% | {r['Once_SLO1p5_percent']:.2f}% |")
        return '\n'.join(out)
    report=f'''# 哪些输入使OD较差，而原始Once较好

**已有明确的SLO反例，但没有证明Once能把这三组的整机利用率提高10个百分点。** 这里的Once是原始完整候选池Once，不是固定候选池或动态候选池策略。

32 NPU、3 SSU×40 GiB/s、Ring hash、8层、batch=1、每卡Random，seed7/19/43等权。同一组同一seed的两策略共享逐字节相同manifest。原full/semi结果已完成，新增under Once三个完整运行；未修改原输入或旧图。

## 主窗口 `[2,4)` 秒

{table('warm_2_4s')}

持续过载组的Once相对OD：NPU利用率提高2.2031个百分点，SLO提高36.9938个百分点。局部欠载组：利用率提高0.5994个百分点，SLO提高5.4578个百分点。当前持续欠载输入两者均为100%，不是OD较差的输入。

“持续/局部”沿用对OD在主窗口的逐盘需求分类。调度会改变请求推进和当前画像组合，所以同输入Once的轨迹不一定仍每盘全时过载；并未换输入。

## 能复现明显SLO差距的输入

全部直接取data，不放大读取、不缩短计算：

- 总长度32/64/80/128/160/200K。
- 每种总长度分别取miss256/1024/2048/4096，数量3/2/1/1。
- 每卡42条，共1344条；每卡包含全部24种画像，各卡独立随机排列。

miss256画像每层参考需求约21.34～29.66 GiB/s，而miss4096约1.31～1.86 GiB/s；它们的读取期限和带宽需求差别很大。`B=V/C`是为了把预取藏进计算所需的平均速率，不能由总输入长度一个字段代替。

OD每盘每卡CIR为1.25 GiB/s；当所有32条路径持续积压时，均分服务不区分不同请求的计算窗口。计算很短的请求可能在下层需要开始时尚未读完。CIR不是PIR硬上限：空闲份额仍可借用，不能解释成OD永远只能获得1.25。

原始Once沿用SS/SL/LS/LL每盘20/6/8/6 GiB/s的静态类别CIR及96/32/96/32条类别合法路径，再按5ms快照的拥塞与服务速率估计，为每层各块选择路径。它不是EDF，也没有直接按所有请求的B排序。本例短计算类别得到更及时服务，但不同类别的收益不一致。

**因此OD与Once的差异同时包含类别QoS配置和选路机制，不能把所有收益都归因于“每层选一次路”。**

## 收益与代价

持续过载组主窗口的类别SLO：

| 类别 | OD | Once |
|---|---:|---:|
| SS | 0.00% | 99.10% |
| SL | 71.48% | 100.00% |
| LS | 0.00% | 74.01% |
| LL | 49.40% | 25.40% |

分类按总输入长度与miss：总长度≤80K为首字母S，miss<512为第二字母S。本批SS/LS均为miss256，不能把SS标签直接读成“小读取”。Once救回许多短计算请求，同时牺牲部分LL请求。

## 扩大窗口与完整请求集合

`[2,6)` 秒：

{table('long_2_6s')}

完整请求集合：

{table('full_population')}

完整人口结果仍支持SLO改善，但持续过载组Once的全程利用率反而从OD的61.68%降至60.83%。这组输入受到容量约束，不能称Once全指标占优。完整人口的归一化P95改善，但绝对毫秒P95从约780.10升至1027.62；不同指标回答不同问题。

同一批full请求、三个seed合计：Once新增1592个达标请求，也使338个原本达标的请求变为失败，其中316个属于LL。具体画像和配对记录见[独立复核](existing_evidence.json)。

## 口径与证据

```text
NPU利用率 = 窗内实际计算卡时间 / (32 × 窗口长度)
SLO×1.5 = 接纳至prefill完成耗时 <= 1.5 × 本请求8层纯计算时间
```

SLO是接纳后prefill完成的代理指标，不包含接纳前排队、未模拟真实首token。窗口内接纳的请求全部跟踪到完成；不同策略的窗口请求集合可能不同，完整人口对照使用相同的全请求集。

对于“另找持续欠载但OD明显差、Once好”的输入，目前尚未完成这样的搜索，不能从本例100%外推OD在所有欠载输入都好。候选应具备不同读取期限、明显不同V/C，以及竞争时均分带宽不足以赶上部分请求的下一层；还需逐盘逐事件验证欠载并测实际Once收益。

- [三组逐窗口结果CSV](comparison.csv)
- [新增欠载Once：逐seed结果](under_once_metrics.csv)、[均值](under_once_macro.csv)、[输入与执行核验](under_once_results.json)
- [已有full/semi独立复算](existing_evidence.json)
- [原多样输入与数学说明](../od_baseline_diverse_ssu3_20260918/input_math.md)
- [原持续欠载输入报告](../continuous_underload_asu_od_20260918/README.md)
'''
    (HERE/'README.md').write_text(report)
    audit=dict(all_checks_passed=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)
    (HERE/'comparison_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(table('warm_2_4s'))


if __name__=='__main__':main()
