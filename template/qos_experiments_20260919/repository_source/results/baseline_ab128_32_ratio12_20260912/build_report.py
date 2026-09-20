#!/usr/bin/env python3
"""Summarize every preserved result; distinguish capacity and timing effects."""
from pathlib import Path
import csv
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
import render

HERE=Path(__file__).resolve().parent


def window(a,left,right):
    return next(w for w in a['windows'] if w['start_ms']==left and w['end_ms']==right)


def load(stage):
    out={}
    for path in sorted((HERE/stage/'runs').glob('*/baseline/audit/audit.json')):
        a=json.loads(path.read_text()); assert a['technical_passed']
        out[a['label']]=a
    return out


def pct(value):return f'{value:.2f}%'


def main():
    screen,validation=load('screen'),load('validation20s')
    assert len(screen)==15 and len(validation)==12,(len(screen),len(validation))
    audit_rows=[]
    for stage,cases in [('screen',screen),('validation20s',validation)]:
        for a in cases.values():
            for w in a['windows']:
                row=dict(stage=stage,label=a['label'],num_ssu=a['dimensions']['num_ssu'],seed=a['seed'],order=a['order'],
                    start_ms=w['start_ms'],end_ms=w['end_ms'],U_percent=w['device_utilization_percent'],
                    A_U_percent=w['classes']['A']['conditional_utilization_percent'],B_U_percent=w['classes']['B']['conditional_utilization_percent'],
                    A_stall_card_ms=w['classes']['A']['L0_stall_ms']+w['classes']['A']['internal_stall_ms'],
                    B_L0_stall_card_ms=w['classes']['B']['L0_stall_ms'],B_internal_stall_card_ms=w['classes']['B']['internal_stall_ms'],
                    all_active=w['all_32_active'],mixed_cards=w['mixed_card_count'],strict_underload=w['nominal']['strict_underload'],
                    nominal_peak_per_disk_gib_s=w['nominal']['max_ssu_gib_s'],any_disk_over_ms=w['nominal']['any_ssu_over_capacity_ms'],
                    K_A_mean=w['nominal']['K_A_mean'],K_A_peak=w['nominal']['K_A_peak'])
                audit_rows.append(row)
    with (HERE/'all_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audit_rows[0]));writer.writeheader();writer.writerows(audit_rows)
    def cases(disks,mode):
        return [validation[f'ssu{disks}_{mode}_k1_sync_seed{s}'] for s in ((7,19,43) if mode=='random' else (7,))]
    rows=[]
    for disks in (3,4,6):
        r,o=cases(disks,'random'),cases(disks,'ordered')[0]
        rv=[[window(a,2000,end)['device_utilization_percent'] for a in r] for end in (4000,20000)]
        rows.append(dict(num_ssu=disks,capacity=40*disks,ideal_load_percent=124.90975862175765/(40*disks)*100,
            random_warm_mean=statistics.mean(rv[0]),random_warm_min=min(rv[0]),random_warm_max=max(rv[0]),
            ordered_warm=window(o,2000,4000)['device_utilization_percent'],
            random_long_mean=statistics.mean(rv[1]),random_long_min=min(rv[1]),random_long_max=max(rv[1]),
            ordered_long=window(o,2000,20000)['device_utilization_percent']))
    (HERE/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    figure_dir=HERE/'figures';figure_dir.mkdir(exist_ok=True)
    for disks in (3,4,6):
        r,o=cases(disks,'random'),cases(disks,'ordered')[0]
        starts=list(range(2000,20000,2000)); mid=np.array(starts)/1000+1
        values=np.array([[window(a,x,x+2000)['device_utilization_percent'] for x in starts] for a in r])
        ov=[window(o,x,x+2000)['device_utilization_percent'] for x in starts]
        fig,ax=plt.subplots(figsize=(11,5.6));fig.subplots_adjust(top=.8,bottom=.18)
        fig.suptitle(f'{disks} SSU × 40 GiB/s：扩大统计窗口后是否仍然低',fontsize=18,y=.97)
        fig.text(.5,.89,'每张卡 40A＋80B；Random 为 3 个种子，Ordered 为 ABB 循环、seed 7',ha='center',fontsize=10)
        ax.fill_between(mid,values.min(axis=0),values.max(axis=0),color='#4b7da7',alpha=.2,label='Random 三种子范围')
        ax.plot(mid,values.mean(axis=0),'o-',color='#366e9c',label='Random 三种子均值')
        ax.plot(mid,ov,'s-',color='#bf6b31',label='Ordered')
        ax.set(xlabel='2 秒统计窗的中心时刻（秒）',ylabel='NPU 平均利用率（%）',ylim=(40,101),xticks=mid)
        ax.grid(alpha=.2);ax.legend(loc='lower right')
        row=next(row for row in rows if row['num_ssu']==disks)
        fig.text(.125,.055,f'[2,20) 秒：Random {row["random_long_mean"]:.2f}%；Ordered {row["ordered_long"]:.2f}%。每个点使用完整日志裁剪计算时间。',fontsize=10)
        render.save(fig,figure_dir/f'window_history_ssu{disks}')
    lines=['# A 总长 128K / 新增 256，B 总长 32K / 新增 4K：Baseline 顺序对照','',
        '结果来自未修改的原始模拟器。每卡请求数量 A:B=1:2；主统计窗为 `[2,4)` 秒，另验证 `[2,20)` 秒。', '',
        '**3 盘的长窗结果：Ordered 64.22%，Random 三种子平均 91.41%；主要损失来自 A。所有已测输入都存在逐时刻名义需求超限，因此本实验不是此前“逐盘逐时刻欠载”的反例。**','',
        '## 主要结果','',
        '下表使用长验证人口：每卡 40A＋80B，共 3840 个请求。Random 为 seed 7、19、43 的利用率算术平均；Ordered 为 seed 7 的 `ABB` 循环。括号给出 Random 三个种子的最小—最大值。','',
        '| SSU 数量 | 总容量 GiB/s | 理想平均需求/容量 | Random U，[2,4) | Ordered U，[2,4) | Random U，[2,20) | Ordered U，[2,20) |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["num_ssu"]} | {row["capacity"]} | {row["ideal_load_percent"]:.2f}% | {row["random_warm_mean"]:.2f}% ({row["random_warm_min"]:.2f}–{row["random_warm_max"]:.2f}) | {row["ordered_warm"]:.2f}% | {row["random_long_mean"]:.2f}% ({row["random_long_min"]:.2f}–{row["random_long_max"]:.2f}) | {row["ordered_long"]:.2f}% |')
    lines += ['', '理想平均需求约 124.91 GiB/s，计算式为 `32 × (V_A + 2V_B) / (C_A + 2C_B)`，C 使用秒、V 使用 GiB。这里假设 32 卡持续计算且按完整配额执行。3 盘容量 120 GiB/s，理想平均已过载 4.09%；4、6 盘通过这个平均容量条件，但仍可能出现瞬时集中读取。它不能替代实际逐时刻检查。', '',
        '长验证与最初筛选使用不同队列长度。完整随机洗牌的前缀随人口长度变化，因此前期截图数字与此表的 Random 数字可能不同；最终图统一使用长验证 seed 7，截图本身不代表三种子平均。','',
        '所有长验证在 `[2,20)` 内每卡均计算过 A 和 B，所有预定 2 秒窗均 32 卡持续有任务。但 Random 的部分 2 秒子窗并非每卡都包含两类；不能把长窗逐卡混合当成每个短窗都逐卡混合。每格 audit 的 mixed_card_count 保留具体数量。','',
        '## 输入怎样分给 32 张卡','',
        '| 请求 | 总输入 | 新增 | 命中 | 每层读取 | 每层计算 | 每层 V/C |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| A | 128K | 256 token | 127.75K | 175.65625 MiB | 6.024012 ms | 28.475926 GiB/s |',
        '| B | 32K | 4096 token | 28K | 38.5 MiB | 28.592842 ms | 1.314932 GiB/s |','',
        '这两行直接来自根目录 `data` 的 `(128,256)` 和 `(32,4096)`；没有插值、缩放计算时间或补读取量。每个请求 8 层、batch=1。','',
        '- **Ordered 主对照**：NPU 0–31 都按 `A→B→B→A→B→B→…` 执行，重复 40 轮。所有请求 t=0 到达，各卡独立按完成进度推进，没有运行时同步屏障。类型序列和初始起点相同，是刻意同步的压力输入。',
        '- **Random**：每张卡独立打乱自己的同一批 40A＋80B，随机种子为 `seed+100003×NPU编号`；不是反复播放同一小段随机序列。提交同时事件的顺序也由该次 seed 控制。',
        '- **身份与落盘**：`original_request_id` 保留原请求身份；每个 KV 块是 128 token、176 KiB，落在 `(块编号+NPU编号) mod SSU数量`。同一拓扑的 Random/Ordered 保留同一原请求的 NPU、读取量、计算时间和落盘。',
        '- **硬件**：32 NPU；3/4/6 SSU，每盘 40 GiB/s；每卡接收链路 50 GiB/s。Baseline 的每盘所有 IO 都在 Path0 FIFO。',
        '- **人口与真实流量的区别**：这是两个原始画像组成的人工排队压力实验，不是生产请求分布；全部 t=0 到达也不能代表真实开放到达流。','',
        '## 是 B 在等，还是 A 在等','',
        '以下是 `[2,20)` 秒的类别计算利用率。类别 U=该类别计算时间÷该类别已接纳占用卡时间；它与该类别对整机 U 的贡献不同。Random 列仍为三种子类别 U 的算术平均。','',
        '| SSU | Random A 类 U | Random B 类 U | Ordered A 类 U | Ordered B 类 U | Ordered B 首层等待（卡·ms） | Ordered B 内部等待（卡·ms） |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for disks in (3,4,6):
        r=[window(a,2000,20000) for a in cases(disks,'random')];o=window(cases(disks,'ordered')[0],2000,20000)
        lines.append(f'| {disks} | {statistics.mean(w["classes"]["A"]["conditional_utilization_percent"] for w in r):.2f}% | {statistics.mean(w["classes"]["B"]["conditional_utilization_percent"] for w in r):.2f}% | {o["classes"]["A"]["conditional_utilization_percent"]:.2f}% | {o["classes"]["B"]["conditional_utilization_percent"]:.2f}% | {o["classes"]["B"]["L0_stall_ms"]:.3f} | {o["classes"]["B"]["internal_stall_ms"]:.3f} |')
    lines += ['', 'A 每层只能用约 6 ms 隐藏 175.7 MiB 的读取。很多卡一起进入 A 时，FIFO 中形成大读取波，各卡反复错过下一层数据截止时间。B 每层约算 28.6 ms，只读 38.5 MiB，后续层通常能藏住等待。', '',
        '**A→B 首层是例外**：B 首层在前一个 A 最后一层计算时预取，预算仍只有 A 的约 6 ms，而不是 B 自己的 28.6 ms。B 可以在此等待，但不能把这个等待外推为其后续七层都在堵。', '',
        '本次没有证明“多加 B 就能让 B 大量等待”。增加 B 同时增加可隐藏读取的计算时间、降低理想平均需求；效果需要实测。当前 Baseline 没有 A/B 各自不能互借的限额，因此不能用“B 配额用不完、A 拿不到余量”解释这些结果。', '',
        '**为什么 6 盘后期恢复，而 3/4 盘没有同样恢复？** 原日志中，6 盘同一轮 A 在 32 张卡上的接纳时间跨度，从第 2 轮的 7.989 ms 扩大到第 13 轮的 500.210 ms；读取逐渐分散。3/4 盘在这些轮次的跨度分别仍约 10.963/8.929 ms，集中读取波持续。第 1 轮所有卡同时接纳，跨度都是 0。见 [逐轮相位证据](phase_evidence.json)。', '',
        '32 卡全在 A 时，名义总需求约 911.23 GiB/s；全在 B 时仅约 42.08 GiB/s。B 阶段闲置的磁盘服务时间不能存到以后使用，有限预取也不会提前释放队列中所有未来 A 的读取。这是时间分布与读取依赖的问题，不是某类请求保留了不可借用的带宽份额。', '',
        '## 欠载条件是否满足','',
        '以下列出长验证 seed 7 的主窗 `[2,4)`。名义需求取每个当前已接纳请求的逐盘 V/C，按所有接纳/完成事件精确扫描；不另加下一请求首层，但它的实际读取完整执行。','',
        '| SSU | 顺序 | 单盘名义峰值 GiB/s | 任一盘超过 40 的时长 ms / 2000 | 当前 A 平均卡数 | 当前 A 峰值卡数 | 全卡有任务 / 两类都有计算 |',
        '|---:|---|---:|---:|---:|---:|---|']
    for disks in (3,4,6):
        for mode in ('random','ordered'):
            w=window(cases(disks,mode)[0],2000,4000);c=w['nominal']
            lines.append(f'| {disks} | {mode.title()} | {c["max_ssu_gib_s"]:.3f} | {c["any_ssu_over_capacity_ms"]:.3f} | {c["K_A_mean"]:.3f} | {c["K_A_peak"]} | {w["all_32_active"]} / {w["mixed_card_count"]}/32 |')
    lines += ['', '名义 V/C 超限不等于磁盘超速，也不等价于一定发生 stall。实际每盘仍按 40 GiB/s 服务；真正没赶上计算截止的读取才造成暴露等待。例如 6 盘 Random seed 7 在本窗 U 为 100%，名义需求仍短时超限 0.714 ms。', '',
        'A 发生等待后，其驻留时间会延长，使当前 A 卡数增加，所以不能拿无等待时的平均 A 卡数证明欠载。名义需求、实际字节服务和计算截止分别统计。', '',
        '3 盘的全批次字节守恒给出理想整机 U 上限约 96.07%，即使完美调度也不能达到 100%。更大的实际损失还包含输入的集中波动和流水依赖。本次只测试 Baseline，无法把全部损失量化为相对其他调度器的差距。','',
        '## 看图','',
        '所有图都是实际日志，没有移动或拼接时间。每卡带宽按 0.5 ms 的物理服务区间重叠积分，并核验字节守恒、每盘不重叠及接收链路不超速。','',
        '| SSU | Random 时序 | Ordered 时序 | Ordered 首层局部图 | 32 卡带宽目录 | 长窗变化 |',
        '|---:|---|---|---|---|---|']
    for disks in (3,4,6):
        base=f'figures/ssu{disks}'
        lines.append(f'| {disks} | [2秒]({base}/baseline_random.png) / [18秒]({base}/long/baseline_random.png) | [2秒]({base}/baseline_ordered.png) / [18秒]({base}/long/baseline_ordered.png) | [PNG]({base}/ordered_B_zoom.png) | [逐卡链接](figures/index.md#ssu{disks}) | [PNG](figures/window_history_ssu{disks}.png) |')
    lines += ['', '图中需求线是当前请求 V/C 的参考值，等待期间仍保留；它的积分不是新到达的字节量。SSD 实际服务与 NPU 实际收到分别绘制：多盘可暂时合计超过某卡 50 GiB/s，但数据随后经该卡 50 GiB/s 链路排队接收。', '',
        '**I/O 等待不要求等待期间的瞬时带宽一直低于需求线。** 真正条件是截止时数据还未收齐；随后以高带宽补读也不能追回已经暴露的等待。累计字节图给出了截止和缺口。','',
        '## 完整筛选与可复现性','',
        '首先预定 15 格：3 种盘数 × Random、ABB、A⁴B⁸、两组错开 A⁴B⁸、四组错开 A⁴B⁸。每卡 12A＋24B；所有结果都保留。下面的短窗低点不作为持续结果。','',
        '| SSU | 顺序 | U，[2,4) | U，[4,6) | B 类 U，[2,4) | B 内部等待（卡·ms），[2,4) |',
        '|---:|---|---:|---:|---:|---:|']
    for name,a in sorted(screen.items()):
        w=window(a,2000,4000);later=window(a,4000,6000)
        lines.append(f'| {a["dimensions"]["num_ssu"]} | {name.split("_",1)[1].removesuffix("_seed7")} | {w["device_utilization_percent"]:.2f}% | {later["device_utilization_percent"]:.2f}% | {w["classes"]["B"]["conditional_utilization_percent"]:.2f}% | {w["classes"]["B"]["internal_stall_ms"]:.3f} |')
    lines += ['', '两组错开 k4：卡 0–15 为 `A⁴B⁸`，卡 16–31 为 `BA⁴B⁷`；四组错开：每组 8 卡，以长度 12 环的索引 0、4、6、8 开始。都只旋转同一批请求，不改变画像配额。', '',
        '随后用简单 ABB 主对照延长到每卡 40A＋80B，验证 3 个盘数、Random 三种子和 Ordered seed 7，共 12 格；没有挑选最差随机种子。最终 27 次完整仿真全部技术核验通过。', '',
        '- [所有结果 CSV](all_results.csv)、[主对比 JSON](comparison.json)。',
        '- [输入和运行脚本](experiment.py)、[筛选矩阵](run_matrix.py)、[长验证矩阵](run_validation.py)。',
        '- [逐时刻审计](audit_results.py)、[带宽与时序绘图](render.py)、[局部图](zoom.py)。',
        '- `screen/runs/*/baseline/` 和 `validation20s/runs/*/baseline/` 保存完整 input/result、command、audit，以及 `[1.8,4.2)` 秒物理服务采样 trace；SHA 绑定原数据、核心源码、输入与结果。所有块完成后才结束仿真，trace 只保留采样窗相交块。',
        '- 每格审计还保存接纳时钟/到达时钟的 SLO×1.5；全部到达时间为零，暖窗没有新到达请求。接纳后的完成时间比率不能冒称真实到达计时 TTFT。','']
    (HERE/'report.md').write_text('\n'.join(lines))
    idx=['# 时序与逐卡带宽图','', '最终图使用长验证人口、seed 7、[2,4) 秒；同名 PDF 可用于放大。','']
    for disks in (3,4,6):
        idx += [f'<a id="ssu{disks}"></a>',f'## {disks} SSU × 40 GiB/s','',
            f'[Random 2秒时序](ssu{disks}/baseline_random.png) · [Ordered 2秒时序](ssu{disks}/baseline_ordered.png) · [Ordered 局部](ssu{disks}/ordered_B_zoom.png)', '',
            f'[Random 18秒时序](ssu{disks}/long/baseline_random.png) · [Ordered 18秒时序](ssu{disks}/long/baseline_ordered.png)', '',
            f'[Random 带宽总览](ssu{disks}/bandwidth_overview_random.png) · [Ordered 带宽总览](ssu{disks}/bandwidth_overview_ordered.png)', '',
            '| NPU | 带宽对照 PNG | PDF |','|---:|---|---|']
        for n in range(32):
            base=f'ssu{disks}/per_npu/npu_{n:02d}'
            idx.append(f'| {n} | [PNG]({base}.png) | [PDF]({base}.pdf) |')
        idx += ['',f'[Random 原始带宽分箱](ssu{disks}/bandwidth_random.csv.gz) · [Ordered 原始带宽分箱](ssu{disks}/bandwidth_ordered.csv.gz)','']
    (figure_dir/'index.md').write_text('\n'.join(idx))
    print(json.dumps(dict(screen_cases=len(screen),validation_cases=len(validation),comparison=rows),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
