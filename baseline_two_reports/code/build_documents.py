#!/usr/bin/env python3
"""Build the two Chinese Markdown reports from recomputed event accounts."""
from pathlib import Path
import csv
import gzip
import json
import math

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8') as f:return json.load(f)

def pct(x):return '—' if x is None else f'{100*x:.4f}%'
def f(x):return f'{x:.6f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def image(path,caption):return f'![{caption}]({path})\n\n{caption}\n'

DEFINITIONS=r'''## 统计口径：把“逐卡平均”和“按时间合并”分开

所有下列利用率都只使用指定的同一个 1 秒窗口 `[a,b)`，边界事件按实际重叠时长裁剪，不只统计窗口内完成的请求。

- 整机利用率：每卡在窗口内的计算时间除以 1000 ms，再对 32 卡等权平均。等待 I/O 计入分母。
- 逐卡、逐类条件利用率：令 `Cᵢ,g` 为卡 i 处理类别 g 的计算时长，`Aᵢ,g` 为其处理该类的占用时长（计算＋等待），则 `Uᵢ,g=Cᵢ,g/Aᵢ,g`。先逐卡算，再对窗口内确实出现该类的卡等权平均。
- 合并时间条件利用率：`U_g=ΣCᵢ,g/ΣAᵢ,g`。这是按该类占用时长加权的结果，不等同于上项逐卡等权平均。
- 表中 `—` 表示该卡在这一秒没有运行该类，条件利用率无分母；不能填成 0，也不意味着完整输入中没有该类。

\[
\bar U=\frac1{32}\sum_i\frac{C_i}{1000}
=\sum_g\underbrace{\frac{\sum_iA_{i,g}}{32000}}_{该类占用的卡时间份额}
\underbrace{\frac{\sum_iC_{i,g}}{\sum_iA_{i,g}}}_{合并时间条件利用率}.
\]

上式的类别求和只包括 `ΣᵢAᵢ,g>0` 的类别；没有占用时间的类别不参与。

所有卡在观测窗口全程有已接纳请求，故这几组的 `compute+stall=1000 ms/卡`、idle=0。“已接纳”与“外部已到达”是两个时刻；跨请求 L0 预取可早于接纳。
'''

CLASSIFICATION='''类别使用原仿真器的两个维度，不是按本次请求是否等待来分类：

| 类别 | 总序列长度 | NQL |
|---|---|---|
| SS | ≤80K | <512 |
| SL | ≤80K | ≥512 |
| LS | >80K | <512 |
| LL | >80K | ≥512 |

第一位表示序列长短，第二位表示 NQL 大小。`1K=1024 token`。类别只是画像分类；Baseline 在每个 SSU 均走本盘的 Path0 FIFO，不按这四类分配四条路径。
'''

def group_table(stats,key):
    rows=[]
    source=stats[key]
    for label in (['SS','SL','LS','LL'] if key=='category_summary' else [r['label'] for r in source]):
        r=next((x for x in source if x['label']==label),None)
        if r is None:rows.append([label,0,0,'—','—','0.0000%']);continue
        rows.append([label,r['input_count'],r['warm_active_npus'],pct(r['mean_per_npu_conditional_utilization']),pct(r['pooled_conditional_utilization']),pct(r['occupancy_share'])])
    return table(['类别/画像','完整输入条数','该秒出现卡数','先逐卡算再平均','合并时间计算比','占全部卡时间'],rows)

def npu_category_table(stats):
    lookup={(r['npu_id'],r['label']):r['conditional_utilization'] for r in stats['per_npu_category']}
    return table(['NPU','整秒 U','SS 条件 U','SL 条件 U','LS 条件 U','LL 条件 U'],[[r['npu_id'],pct(r['utilization'])]+[pct(lookup.get((r['npu_id'],c))) for c in ('SS','SL','LS','LL')] for r in stats['per_npu']])

def detail_table(case,selection):
    r=next(r for r in case['requests'] if r['request_id']==selection['request_id'])
    rows=[]
    for l in r['layers']:
        prior=r['admission_ms'] if l['layer']==0 else r['layers'][l['layer']-1]['compute_end_ms']
        cs=l['compute_start_ms'];ce=l['compute_end_ms']
        rows.append([l['layer'],f(l['io_start_time_ms']),f(l['io_ready_time_ms']),f(cs),f(ce),f(max(0,cs-prior))])
    text=table(['层','发出 I/O ms','I/O 就绪 ms','计算开始 ms','计算结束 ms','该层 NPU 等待 ms'],rows)
    compute=math.fsum(l['compute_end_ms']-l['compute_start_ms'] for l in r['layers'])
    wall=r['completion_ms']-r['admission_ms']
    text+=f'\n此请求接纳到完成 {wall:.6f} ms，8 层纯计算 {compute:.6f} ms，I/O 等待 {wall-compute:.6f} ms，计算比 {pct(compute/wall)}。这是单个明确标出的请求，不代替全体平均。\n'
    return text

def high_report():
    key='high';stats=read(ROOT/f'data/{key}/statistics.json');case=read(ROOT/f'data/{key}/normalized.json.gz')
    audit=read(ROOT/'audits/high_source_audit.json');detail=read(ROOT/f'data/{key}/detail_selection.json')
    assert all(r['request_id']==r['npu_id']*1000000+r['input_order'] for r in case['inputs'])
    with (ROOT/f'data/{key}/profile_catalog.csv').open(encoding='utf-8-sig',newline='') as f_:catalog=list(csv.DictReader(f_))
    profile_rows=[[r['profile_id'],f"{r['seq_len_k']}K / {r['nql']}",r['category'],r['quota'],int(r['quota'])*32,f"{float(r['compute_ms']):.6f}",f"{float(r['layer_kv_gib'])*1024:.6f}",f"{float(r['required_bw_gib_s']):.6f}"] for r in catalog]
    input_total_compute=math.fsum(r['layer_compute_ms']*8 for r in case['inputs'])
    short_compute=math.fsum(r['layer_compute_ms']*8 for r in case['inputs'] if r['category']=='SS')
    ss=next(r for r in stats['category_summary'] if r['label']=='SS');p7=next(r for r in stats['profile_summary'] if r['label']=='P7')
    text=f'''# 原 Baseline 高利用率不等于短请求服务良好：输入与 1 秒时序反证

日期：2026-09-07。本文复核 `baseline_npu32_investigation` 所分析的原 coflow 六盘案例，不重新构造输入、不重跑仿真、不改变请求顺序。

## 要反驳的结论

**“Baseline 整机利用率很高，因此各类请求都没有明显 I/O 等待”不成立。** 本案例的 {pct(stats['fleet_utilization'])} 是正确的设备时间统计；错误在于把它推广成短请求的服务结论。也不能因为短类计算比低，就反过来把整机实际高利用率说成计算错误。

这里长短请求确实在**每张 NPU 的同一串行请求队列中混排**，并非若干卡专跑短流。只把整机均值改为“先逐卡算，再平均”不会解决掩盖，因为原代码本来就这样计算。

## 1. 精确输入：每卡 22 条，32 卡共 704 条

案例：`{case['case_id']}`；32 NPU、6 SSU、每盘40 GiB/s、每卡接收链路50 GiB/s；每请求8层、batch=1，单层及跨请求L0预取；Baseline-only，seed=20260906。

NPU0–31 **每一张卡**都具有下表相同配额，卡内顺序各自独立打乱。P1:P2:P3:P4:P5:P6:P7 = **4:2:2:2:4:3:5**。

'''
    text+=table(['画像','序列/NQL','类别','每卡请求数','32卡请求数','每层计算 ms','每层读取 MiB','V/C GiB/s'],profile_rows)
    text+='\n'+CLASSIFICATION
    text+='\n每卡按模拟器类别计数为 **SS:SL:LS:LL = 6:2:0:14**；全体为 **192:64:0:448**。如果把32K当短、192K当长，则还有64K与96K两种中间画像，不能将七画像偷偷合成一个没有定义的二分类。\n'
    text+='\n**历史输入存在同卡重复画像**，例如每卡有5条192K/NQL2048。本文保留该历史事实；新低利用率文档使用的是每卡画像不重复的另一组输入，两者不能混称。\n'
    text+=f'\n请求归属精确规则：`request_id = npu_id × 1,000,000 + input_order`，`input_order=0..21`。例如 request 17000004 是 NPU17 的第5条输入。完整704条、参数与所有时间见 [all_inputs_and_times.csv.gz](data/high/all_inputs_and_times.csv.gz)。\n\n每卡前4条的 arrival=0，共128条；其余576条按原 coflow 到达时钟入队，最后一条 arrival={audit["last_arrival_ms"]:.6f} ms。arrival、admission、first_layer_io_start、completion 均分别保存，未套用新实验的全0到达时钟。\n'
    text+=image('images/high_input_order.png','图1：全部32卡的完整22请求顺序。每格等宽表示一条输入的位置，不是执行时长；P1–P7与上表一一对应。')
    text+='\n## 2. 为什么时间均值高：计算预算与占用份额都偏向长计算请求\n'
    text+=f'\nSS占完整输入请求数27.2727%，但只占完整输入纯计算时间的 **{pct(short_compute/input_total_compute)}**；这是完整704请求的统计，不是warm分母。原near配额为了满足平均盘容量条件，增加了NQL2048长计算请求，因此这组人工输入的难度与固定短卡坏例不同，不能称作生产trace。\n'
    text+=f'\n在本秒实际时间账中，SS只占 **{pct(ss["occupancy_share"])}** 的NPU占用时间；P7单独占 **{pct(p7["occupancy_share"])}**，其中 **{pct(p7["pooled_conditional_utilization"])}** 都在计算。这才是“高均值掩盖短类等待”的直接窗口证据，不能把完整输入纯计算占比拿来代替窗口占用占比。\n'
    text+='\n'+DEFINITIONS
    text+='\n## 3. 原始事件绘制的 1 秒时序图\n'
    text+=f'\n固定窗口 **[{stats["window_start_ms"]:.6f}, {stats["window_end_ms"]:.6f}) ms**。共有{stats["warm_overlapping_requests"]}条请求与窗口重叠，{stats["warm_completed_count"]}条在窗口完成；这两个数都不等于完整输入704条。\n'
    text+=image('images/high_timeline_1s.png','图2：全部32张NPU的同一秒。横轴是相对窗口起点的0–1000ms；绝对时间=1000ms+横轴。彩色条为对应P画像的计算，红色为等待I/O，黑色短竖线为请求完成；右侧为逐卡整秒利用率。')
    text+='\n长块标注 `P7 / #6` 的含义是P7画像、该卡第7条输入（序号从0起）；较窄请求不堆叠文字，全部完成时间和层边界在CSV中。横条上反复出现P7/P6而短请求成为较窄的红段，说明掩盖已经发生在每张卡内部。图并未省略任何与窗口重叠的计算事件。\n'
    text+=f'\n可手算：所有卡计算合计 **{stats["compute_ms"]:.6f} NPU-ms**，等待合计 **{stats["stall_ms"]:.6f} NPU-ms**，两者之和32000；整机U=`{stats["compute_ms"]:.6f}/32000`={pct(stats["fleet_utilization"])}。\n'
    text+='\n## 4. 同一秒的逐类别、逐画像利用率\n\n下表“先逐卡算再平均”对应你提出的口径；“合并时间计算比”用于还原整机的时间账。两者都不是“每请求等权平均”。\n\n'+group_table(stats,'category_summary')
    text+='\nSS只有24张卡在该秒出现，SL只有11张，LL有32张；因此分别使用24、11、32作逐卡条件平均的卡数，不能一律除32或给缺失格填0。\n'
    text+='\n'+group_table(stats,'profile_summary')
    text+=image('images/high_profile_utilization.png','图3：每张NPU × 七种画像的条件利用率，所有格子都来自图2同一秒。灰格“--”表示该秒没有对应画像，不是0%。颜色刻度0–100%，格内标出数值。')
    text+='\n### 每张 NPU 的整秒与分类利用率\n\n'+npu_category_table(stats)
    text+='\n## 5. 层间错位不能保证下一层不堵\n'
    text+=f'\n按预先说明的规则，选择“本秒内SS请求等待重叠时间最大”的一条作为放大例：NPU{detail["npu_id"]}、request `{detail["request_id"]}`，输入序号{detail["input_order"]}。这是最坏请求示例，不能把它的比例当总体均值。\n'
    text+=image('images/high_request_detail.png','图4：该真实请求的8层事件。灰色是I/O发出至就绪的总历时（包含排队、服务和链路），不是SSD独占服务；蓝色是计算，红色是NPU实际等待。图使用绝对毫秒，文字标出计算起止。')
    text+='\n'+detail_table(case,detail)
    text+='\n这条请求L1–L7连续存在等待，直接反驳“第N层堵过后，N+1层必然因错位而清空”。相位移动会改变后续竞争，有时缓解、有时复发；这张图不能单独估算错位对整机U的因果贡献。\n'
    text+='\n## 6. 全部32卡的输入序列（可与图1和请求ID逐项对应）\n\n'
    text+=table(['NPU','输入序号0→21的画像顺序'],[[r['npu_id'],' '.join(r['profile_order'])] for r in stats['allocation']])
    text+='\n## 数据与复核范围\n\n本次没有新仿真。原native结果、精确manifest与冻结源码分别校验哈希，704请求身份和全部卡内顺序匹配；两套独立重聚合均得到相同逐卡整秒U。完整记录见 [源数据审计](audits/high_source_audit.json)、[统计结果](data/high/statistics.json)、[逐请求窗口账](data/high/warm_requests.csv)、[逐层时刻](data/high/warm_layers.csv.gz)、[逐卡逐类别表](data/high/per_npu_category.csv)。所有CSV的utilization字段以0–1保存，本文百分比乘100。\n'
    text+='\n本例支持“高设备利用率不代表短请求服务好”，不支持“Baseline的设备利用率实际很低”。该一秒也不证明所有窗口或业务分布相同。\n\n来源：\n\n'
    text+='- [调查报告](https://github.com/chguo0503/qos_storage_sim/blob/main/results/baseline_npu32_investigation/docs/report.md)\n'
    for name,url in zip(['原Baseline结果','原精确输入manifest','冻结分类源码','冻结输入配额代码','冻结coflow到达时钟代码'],case['source_urls']):text+=f'- [{name}]({url})\n'
    (ROOT/'01_baseline_high_utilization_rebuttal.md').write_text(text,encoding='utf-8')

def low_report():
    cases={n:read(ROOT/f'data/low{n}/normalized.json.gz') for n in (4,5,6)}
    stats={n:read(ROOT/f'data/low{n}/statistics.json') for n in (4,5,6)}
    metrics={n:read(ROOT/f'sources/low{n}/metrics.json') for n in (4,5,6)}
    text='''# 什么输入使 Baseline 利用率低：精确绑定、数量、类别与 1 秒时序

日期：2026-09-07。本报告重聚合此前已完成的三个32-NPU Baseline边界案例，不新增或重跑仿真。

## 结论与适用范围

少数长流卡持续发出大层读取，多数短流卡的计算窗口又很短时，短流等待可以进入整机均值。本组固定画像族、固定卡绑定与seed下，4/5/6 SSU分别在 **6长:26短、9长:23短、11长:21短** 得到64.0552%、60.2169%、67.3949%的整机利用率；长流仍100%。

这些点通过“完整输入理想平均需求不超过最热盘40GiB/s”的筛选。它排除了持续平均容量超额，不证明每个瞬时deadline可满足，也不证明全部损失都能由另一调度策略消除。比例不是适用于任意画像的常数。

与第一份文档相比，本例改变了输入画像和固定分离/混排方式；它们是不同输入的机制实例，不能当作同输入算法优劣对照。

## 1. 请求类别与精确输入构造

'''+CLASSIFICATION
    sample=cases[4]['inputs'];rows=[]
    for pid,name in [('S','短族，约1K/NQL128'),('L','长族，约192K/NQL512')]:
        group=[r for r in sample if r['profile_id']==pid]
        rows.append([pid,name,f"{min(r['total_tokens'] for r in group)}–{max(r['total_tokens'] for r in group)}",f"{min(r['nql'] for r in group)}–{max(r['nql'] for r in group)}",'/'.join(sorted({r['category'] for r in group})),f"{min(r['layer_compute_ms'] for r in group):.6f}–{max(r['layer_compute_ms'] for r in group):.6f}",f"{min(r['layer_kv_gib']*1024 for r in group):.6f}–{max(r['layer_kv_gib']*1024 for r in group):.6f}"])
    text+=table(['族名','含义','实际总token范围¹','实际NQL范围¹','模拟器类别','每层计算ms¹','每层读取MiB¹'],rows)
    text+='\n¹表中数值范围从4-SSU案例实际输入读取；各卡逐请求精确值均在对应输入CSV。生成候选池短族为token960–1088/NQL112–144，长族为token196480–196608/NQL496–528。长族跨NQL512边界，因而同时包含LS和LL，不能把全部长族直接标为LL。\n\n**1K短族的计算时间由原data的32K/48K向外推，长族邻近值使用插值；这是一组合成机制输入，不是硬件实测1K画像，也不是生产trace。** 每请求8层、batch=1；每SSU40GiB/s，每NPU接收链路50GiB/s；Baseline每盘独立Path0 FIFO，一层预取并启用跨请求L0预取，seed=20260907。\n'
    text+='\n每卡准备约5秒纯计算工作量的有限有序backlog，故短卡有约1182条、长卡37条，不能把卡数比写成请求条数比。每卡内部 `(total_tokens,NQL)` 不重复；各卡输入顺序固定，跨不同SSU数时同角色同卡序列保持不变。跨卡仍可出现相同参数。\n\n所有 `arrival_ms=0`；实际接纳、L0 I/O与完成时间另记。未运行到的未来请求保留空时刻，不把它们当作已经完成。\n'
    text+='\n### 三个案例的精确卡绑定与完整请求人口\n\n'
    rows=[]
    for n in (4,5,6):
        s=stats[n];m=metrics[n];cats={r['label']:r['input_count'] for r in s['category_summary']}
        rows.append([n,f"0–{m['n_short']-1}",f"{m['n_short']}–31",f"{m['n_long']}:{m['n_short']}",f"{cats.get('LS',0)+cats.get('LL',0)}:{cats.get('SS',0)}",s['input_count'],cats.get('SS',0),cats.get('LS',0),cats.get('LL',0),f"{m['max_ideal_ssu_gibps']:.6f}"])
    text+=table(['SSU','短族NPU','长族NPU','长:短卡数','长:短请求条数','输入总条数','SS条数','LS条数','LL条数','最热盘理想GiB/s'],rows)
    text+='\n三例均无SL。请求归属精确规则：`request_id=npu_id×100000+input_order`，每卡序号从0连续递增。各卡的ID区间与请求数在后文全部列出；区间内每一个整数都对应该卡的一条请求，具体参数和时间见完整CSV。\n'
    text+='\n'+DEFINITIONS
    text+='\n## 2. 相同统计口径下的结果\n\n'
    rows=[]
    for n in (4,5,6):
        s=stats[n];profiles={r['label']:r for r in s['profile_summary']}
        rows.append([n,f"[{s['window_start_ms']:.6f}, {s['window_end_ms']:.6f})",pct(s['fleet_utilization']),pct(profiles['S']['mean_per_npu_conditional_utilization']),pct(profiles['L']['mean_per_npu_conditional_utilization']),s['observed_admitted_count'],s['warm_completed_count']])
    text+=table(['SSU','绝对warm窗口ms','整机U','短族逐卡平均U','长族逐卡平均U','捕获时已接纳','warm内完成'],rows)
    text+='\n每卡至少完成4个请求达到暖机门槛，再继续运行1000ms才进入观测窗口，随后连续积分1000ms；不是启动第1秒，也不是整个有限manifest的几何中心。一个warm窗口不能单独证明统计稳态。\n'
    for n in (4,5,6):
        s=stats[n];c=cases[n];m=metrics[n];key=f'low{n}';det=read(ROOT/f'data/{key}/detail_selection.json')
        text+=f'\n## {n-1}. {n} SSU：{m["n_long"]}张长卡＋{m["n_short"]}张短卡\n\n'
        text+=image(f'images/{key}_timeline_1s.png',f'图：{n} SSU的全部32卡同一秒。横轴0对应绝对{s["window_start_ms"]:.6f}ms，右端对应{s["window_end_ms"]:.6f}ms。S蓝色与L灰色表示画像族的计算，红色为I/O等待，黑色短竖线是请求完成点；S/L族名不要与SS/LS/LL类别混淆。')
        text+=f'\n这一秒总计算 **{s["compute_ms"]:.6f} NPU-ms**、等待 **{s["stall_ms"]:.6f} NPU-ms**；二者合计32000。共有{s["warm_overlapping_requests"]}条已接纳请求与窗口重叠，{s["warm_completed_count"]}条在窗口完成。\n'
        text+='\n### 类别统计与每卡条件利用率\n\n'+group_table(s,'category_summary')
        text+=image(f'images/{key}_category_utilization.png','每卡×模拟器类别的条件利用率。SS只出现在短族卡，LS/LL只出现在长族卡；无该类占用的格子标“--”。本例各活跃卡的LS、LL条件利用率均100%。')
        text+='\n### 全部32卡的输入分配与整秒利用率\n\n'
        rows=[]
        for alloc in s['allocation']:
            ni=alloc['npu_id'];cats=alloc['category_counts'];role='S短族' if ni<m['n_short'] else 'L长族'
            rows.append([ni,role,alloc['input_count'],f"{cats.get('SS',0)}/{cats.get('LS',0)}/{cats.get('LL',0)}",f"{alloc['request_id_first']}–{alloc['request_id_last']}",pct(s['per_npu'][ni]['utilization'])])
        text+=table(['NPU','固定画像族','输入条数','SS/LS/LL条数','完整request_id区间','整秒U'],rows)
        text+=f'\n[完整输入与四类时刻](data/{key}/all_inputs_and_times.csv.gz) · [逐请求窗口账](data/{key}/warm_requests.csv) · [逐层事件](data/{key}/warm_layers.csv.gz) · [每卡分类数值](data/{key}/per_npu_category.csv)。CSV保留全部输入顺序，数据中空观察时刻表示未执行到。\n'
        text+='\n### 具体输入样例：短卡NPU0与第一张长卡\n\n'
        sample_rows=[]
        for ni in (0,m['n_short']):
            for r in [r for r in c['inputs'] if r['npu_id']==ni][:6]:
                sample_rows.append([ni,r['input_order'],r['request_id'],r['total_tokens'],r['nql'],r['category'],f(r['arrival_ms'])])
        text+=table(['NPU','卡内序号','request_id','总token','NQL','类别','arrival ms'],sample_rows)
        text+='\n上表只用于阅读示例；完整请求不抽样，全部保存在前述CSV。\n'
        text+=f'\n### 层级放大：request {det["request_id"]}\n\n选择规则仍是本秒内SS请求等待重叠时间最大的一条。\n'
        text+=image(f'images/{key}_request_detail.png','灰条是I/O发出到就绪的总历时，不代表盘一直独占服务；蓝条计算，红条是暴露给NPU的等待，文字为绝对计算起止时间。')
        text+='\n'+detail_table(c,det)
    text+='\n## 6. 为什么这种输入能压低整机U\n\n短卡持续处理短族，不会在同一卡后续穿插长计算请求来稀释短类等待。长族每层读取约263MiB，短族只有约1.2MiB，但短族计算窗口约0.53ms，无法容忍共享队列前方的大量工作；长族约17ms的计算窗口则仍把自己的I/O隐藏住。两者共享各盘Path0时，出现“长卡满算、短卡反复等待”的结果。\n\n这支持“短deadline与共享FIFO排队相冲突”的解释；单张NPU事件图不能给出每条物理IO前面是谁，也不能据此宣称任何其他策略都能100%。所有层使用同一请求的原ring placement，层号变化不会自动把数据换到另一组盘。\n'
    text+='\n例如4SSU：`(6×100%+26×55.7602498%)/32=64.0552030%`。这里26张短卡占整机81.25%的权重；与原混排案例SS只占6.0010%窗口卡时间的情况不同。这解释了同样“短请求受伤”，为什么一个整机仍高、另一个整机明显下降。\n'
    text+='\n## 7. 固定长流后，什么短流更容易使U下降\n\n此前同seed、4SSU、固定4张约192K/NQL512长卡＋28张短卡的已完成对照如下；此表是对应各case自身的1秒warm统计，不与上面的6长:26短图混用：\n\n'
    rows=[]
    for family,label in [('1k_nql64','约1K/NQL64'),('1k_nql128','约1K/NQL128'),('1k_nql256','约1K/NQL256'),('1k_nql512','约1K/NQL512'),('32k_nql2048','约32K/NQL2048')]:
        base=ROOT/'sources/fixed_long_sensitivity'/family
        m=read(base/'metrics.json');meta=read(base/'input_metadata.json')
        lanes=[r for r in meta['lanes'] if r['role']=='short']
        mean_c=math.fsum(r['ideal_compute_ms'] for r in lanes)/(8*sum(r['request_count'] for r in lanes))
        rows.append([label,f'{mean_c:.3f}',pct(m['mean_utilization']),pct(m['short_utilization']),pct(m['long_utilization'])])
    text+=table(['短族','输入请求平均每层计算ms','整机U','短族U','长族U'],rows)
    text+='\n同一批长卡不变时，真正关键的是短流计算窗口/容忍排队的余量；序列更短并不是唯一条件。这些变化同时改变V与C，不能从这几行把C的独立因果贡献与V的贡献完全拆开。\n'
    text+='\n## 数据与复核范围\n\n此次对三份完整原生事件档案重新求交，逐卡计算区间、活跃区间、类别、卡内顺序及输入唯一性均核验；结果与前次原始metrics一致。没有用完成请求数量估计利用率，没有改变输入、到达或请求分卡。输入采用此前每卡唯一整数画像构造；其哈希与配置保存在sources目录。\n\n- [4SSU独立审计](audits/low4_source_audit.json)、[5SSU独立审计](audits/low5_source_audit.json)、[6SSU独立审计](audits/low6_source_audit.json)。\n- 每个 `data/low*/statistics.json` 含全部32卡计数、类别分解与窗口时间账。\n- `sources/low*/metrics.json`、`input_metadata.json` 保留原配置；`sources/fixed_long_sensitivity/` 保留上述固定4长对照的原指标与输入元数据。\n- 本组原生仿真器基线为 [qos_storage_sim / f18ac089](https://github.com/chguo0503/qos_storage_sim/tree/f18ac08991c6449aa7700052e3125cc162d2bca3)；本文数据属于本次对话中已运行的新输入，不伪装为GitHub既有结果。\n'
    (ROOT/'02_baseline_low_utilization_inputs.md').write_text(text,encoding='utf-8')

if __name__=='__main__':
    high_report();low_report()
    print('Wrote both Markdown reports from recomputed statistics.')
