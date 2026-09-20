#!/usr/bin/env python3
"""Write the human report only after all four promoted runs have drained."""
from pathlib import Path
import gzip
import json

HERE = Path(__file__).resolve().parent


def load_case(name):
    path = HERE/'runs'/name
    command = json.loads((path/'command.json').read_text())
    assert command['status'] == 'complete' and command['completed_simulation']
    with gzip.open(path/'result.json.gz', 'rt') as stream:
        return json.load(stream)


def main():
    index = json.loads((HERE/'search_index.json').read_text())
    assert not index['pending'], index['pending']
    names = ['native_phase_lock_a101_b0995_od_local', 'native_phase_lock_a101_b0995_once_remote',
             'fixedssu1_safe38_od_local', 'fixedssu1_safe38_once_remote']
    od, once, safe_od, safe_once = [load_case(name) for name in names]
    assert od['input_fingerprint'] == once['input_fingerprint']
    assert safe_od['input_fingerprint'] == safe_once['input_fingerprint']
    a, b = od['analysis'][0], once['analysis'][0]
    good = lambda x: '通过' if x else '**未通过**'
    warm_rows = []
    for label, run in [('较强候选 / OD',od),('较强候选 / Once',once),
                       ('保守候选 / OD',safe_od),('保守候选 / Once',safe_once)]:
        r=run['analysis'][0]
        warm_rows.append(f'| {label} | {r["U_percent"]:.4f}% | {r["slo"]["percent"]:.4f}% ({r["slo"]["passed"]}/{r["slo"]["count"]}) | '
                         f'{max(r["demand"]["per_disk_max_GiB_s"]):.4f} | {good(r["demand"]["strict_underload_all_disks"])} |')
    long_rows=[]
    for left,right in [(2000,4000),(2000,6000),(4000,8000),(8000,12000),(12000,16000)]:
        ro=next(r for r in od['analysis'] if r['start_ms']==left and r['end_ms']==right)
        rn=next(r for r in once['analysis'] if r['start_ms']==left and r['end_ms']==right)
        long_rows.append(f'| [{left/1000:g},{right/1000:g}) | {ro["U_percent"]:.4f}% | {rn["U_percent"]:.4f}% | '
                         f'{good(ro["demand"]["strict_underload_all_disks"])} | {good(rn["demand"]["strict_underload_all_disks"])} |')
    role_rows=[]
    for role in ('A','B'):
        ro=a['by_role'][role];rn=b['by_role'][role]
        role_rows.append(f'| {role} | {ro["active_U_percent"]:.4f}% | {rn["active_U_percent"]:.4f}% | '
                         f'{ro["admission"]["slo"]["1.5"]["percent"]:.4f}% | {rn["admission"]["slo"]["1.5"]["percent"]:.4f}% |')
    fo,fn=od['analysis'][-1],once['analysis'][-1]
    raw=[r for r in index['formal'] if r['policy']=='od_baseline' and not r['constructed_profile'] and r['warm_under']]
    once_full = ('Once 的完整运行也逐盘欠载。' if fn['demand']['strict_underload_all_disks'] else
                 f'**Once 虽然主窗口欠载，完整运行仍出现超限**，最忙盘需求峰值 {max(fn["demand"]["per_disk_max_GiB_s"]):.4f} GiB/s。不能把该配对说成“两策略在全程都欠载”；若要求这一点，请使用上表的保守候选。')
    text=f'''# 每卡都混合长短请求，OD 在欠载时能有多差？

本轮本机与远端并行探索已完成。**找到了逐盘欠载时的 warm 退化，但没有找到长期保持八十几、或相差 10 个百分点的可靠案例。**

较强的完整实验中，OD 在 `[2,4)` 秒的平均 NPU 利用率为 **{a['U_percent']:.2f}%**，Once 为 **{b['U_percent']:.2f}%**；SLO×1.5 分别为 **{a['slo']['percent']:.2f}% / {b['slo']['percent']:.2f}%**。同一份输入，warm 两策略均逐盘欠载，32 张卡一直活跃、每张卡都真实计算过 A 和 B。但 OD 到 `[8,12)` 秒已恢复到 **99.38%**，因此不能把前面的低值叫作长期状态。

## 1. 先看结果

共同配置：32 NPU、3 SSU × 40 GiB/s、Ring hash、每请求 8 层、每卡 batch=1、下一层和下一请求首层预取。所有请求在 0 时刻到达，固定每卡队列，运行时没有重排、搬卡、额外等待或 CIR 更新。

| 输入与策略 | warm NPU U | SLO×1.5 达标率 | warm 最忙盘需求峰值 GiB/s | warm 严格欠载 |
|---|---:|---:|---:|---|
{chr(10).join(warm_rows)}

表内每个 OD/Once 配对均使用相同冻结 manifest。较强候选提升 **{b['U_percent']-a['U_percent']:.4f} 个百分点的 U**、**{b['slo']['percent']-a['slo']['percent']:.4f} 个百分点的 SLO 达标率**。它和保守候选是不同计算参数，不能交叉比较来表示策略收益。

“欠载”严格按当前已接纳请求的实际逐盘 `V/C` 扫描全部事件区间，等 I/O 的请求也计入。跨请求 L0 预取不额外叠加第二份画像需求，但物理 I/O 和等待都照常计入；所以这不等于任何短时间内所有 I/O 截止时间都能满足。

## 2. 扩大观察窗口后，低 U 没有保持

以下均为较强候选。每个表内窗口，OD/Once 都是 32 卡活跃且每卡计算过 A/B；“欠载”另列，不能从活跃或平均带宽推断。

| 统计窗口（秒） | OD U | Once U | OD 逐盘欠载 | Once 逐盘欠载 |
|---|---:|---:|---|---|
{chr(10).join(long_rows)}

OD 整个完整运行都逐盘欠载。{once_full}不同策略改变完成进度，会改变同一时刻正在运行的请求组合，因此逐盘欠载必须分别核验，不能只查一次输入配比。

OD 从启动到最后排空的 U 为 {fo['U_percent']:.4f}%，其中包含部分卡提前完成后的空闲；**这个较低的全程平均不能用来声称稳态很差**。完整相同请求人口的 SLO 为 OD {fo['slo']['percent']:.4f}%、Once {fn['slo']['percent']:.4f}%；它与 warm 接纳请求子集的 SLO 是不同统计。

## 3. 输入具体是什么样的

这里“长短”按计算时间命名。较强候选只有两种画像；各卡复用这两种画像，但请求 ID 和数据块地址不同。它是有意构造的压力输入，不是从真实到达轨迹随机抽样。

| 项目 | A：短计算请求 | B：长计算请求 |
|---|---:|---:|
| 总输入长度 | 10.25K token | 155.25K token |
| hit / miss | 10K / 256 | 151.25K / 4096 |
| 每层读取 | 13.75 MiB | 207.96875 MiB |
| 每层计算 C | 1.914921 ms | 93.155320 ms |
| 总参考带宽 V/C | 7.012161 GiB/s | 2.180171 GiB/s |
| 瓶颈盘 SSU1 的参考带宽 | 2.454257 GiB/s | 0.754952 GiB/s |

这两种计算时间是合成参数，不是 data 原始测量。另有未修改原始 data 的正式测试；其中满足 warm 欠载的 OD U 在 {min(r['U'] for r in raw):.2f}%–{max(r['U'] for r in raw):.2f}% 之间，也未得到长期很低的 U。

每张卡都连续运行 `35 个 A → 3 个 B`，重复 6 轮。四组各 8 张卡，在队首分别增加 3、1、2、4 个 B，令它们轮流进入 A 段。这不是固定长卡、短卡：每一张卡都轮换角色。

物理地址保持原 Ring hash，但事先筛选不同 ID，使每个 A 在 SSU1 有 28 个 176 KiB 块、每个 B 有 419 块，其他盘不更多。**这是对抗性选址**，用来减少盘间随机差异对同步竞争的打散；不是更换 placement 策略。普通地址和 Random 对照也已保留，部分出现短暂超限，不能算有效欠载反例。

## 4. 为什么它会等，但又会恢复

当 8 张卡运行 A、24 张运行 B，最忙盘的需求约为：

```text
8 × 2.454257 + 24 × 0.754952 ≈ 37.7529 GiB/s < 40
```

OD 每卡每盘保证份额是 `40/32=1.25 GiB/s`。大家同时有读取时，A 想赶在约 1.915 ms 的计算结束前拿齐数据，B 却有约 93.155 ms 的计算可以隐藏读取。均分服务机会不等于按各自的紧迫程度分配，A 因而可能等下一层。

但 **1.25 不是硬限速**。B 读完后，A 能借用空闲份额；各卡的实际完成时间也逐渐不同，后续大读取会错开。真实层时刻的独立审计发现：A 内部预取来不及的比例从 warm 的 43.07% 降到 `[8,12)` 的 5.14%；B 的读取起点也从高度集中变得分散。A 被阻塞更少、驻留更短，平均 A 卡数又从 7.70 变为 6.20。相位和驻留比例都在变化，不能假设永远固定为 8/24。

这个恢复并非把 A 从输入里删掉：重复段中，无等待时 A 的时间份额是 `35*C_A/(35*C_A+3*C_B)≈19.34%`，对应 32 卡中的约 6.19 张；后期实测约 6.20 张正接近它。warm 的 A 份额更高，是因为等待延长了 A 的驻留。A 自身的角色利用率也从 76.98% 恢复到约 99.49%。

连续服务模型曾预测约 92%–93%，但没有充分表达逐块仲裁及短路径重新入队的细节。正式原仿真中，每层细小的提前会累积并改变后续竞争。这说明模型能帮忙找输入，却不能代替实际验证；详见 [模型偏差核对](ideal_search/proxy_native_gap.md) 和 [恢复证据](mechanism_audit/native_phase_lock_a101_b0995_od_local_phase_recovery/README.md)。

## 5. 平均 U 还不错，不代表短请求体验好

较强候选 warm 的分角色统计：

| 角色 | OD 角色利用率 | Once 角色利用率 | OD SLO×1.5 | Once SLO×1.5 |
|---|---:|---:|---:|---:|
{chr(10).join(role_rows)}

角色利用率的分母是该角色占用 NPU 的时间（计算加等待），不是整个 32 卡窗口。OD 中 A 占约 24.07% 的卡时间，B 占约 75.93%，所以 A 约 76.98% 的利用率被 B 的高利用率部分掩盖。

本报告 SLO 沿用实验口径：`接纳后到 prefill 完成 <= 1.5 × 8 × 该请求每层纯计算时间`。A 纯计算约 15.32 ms，只容许再等约 7.66 ms；B 纯计算约 745.24 ms，可容许约 372.62 ms 的额外等待。它是接纳后的 prefill 时延代理，不包含最初到达后的请求排队，也不是实测首 token 时间。

窗口内接纳的请求全部跟到完成。不同策略的 warm 请求子集和数量会变化，表中保留了分子、分母；完整人口结果另列，不能混用。

## 6. 图在哪里，看哪张

- [较强候选 OD / Once 32 卡计算时序](findings/native_phase_lock_a101_b0995/figures/od_vs_once_32npu_timeline.png)：直接看灰色 I/O 等待。
- [较强候选逐盘需求和实际物理供给](findings/native_phase_lock_a101_b0995/figures/od_once_per_ssu_physical_bandwidth.png)：需求是逐事件 V/C；实际供给是统一 10 ms 的物理服务平均，不能把两者当同一个东西。
- [较强候选 OD 每卡完整层周期带宽](findings/native_phase_lock_a101_b0995/figures/od_32npu_layer_average_bandwidth.png)：蓝线按完整层周期平均，窗口边缘只裁显示。
- [利用率和 A 驻留份额随窗口的变化](findings/native_phase_lock_a101_b0995/figures/utilization_recovery_windows.png)：看前期差距如何逐渐消失。
- [保守候选：同一短请求内部的局部放大](findings/fixed_bottleneck_safe38/figures/od_internal_cycle_zoom.png)：每层约算 1.90 ms、等 1.66 ms，完整周期 `b/B` 约 53%；这是局部周期，不是整机 U。
- [保守候选全部图与定义](findings/fixed_bottleneck_safe38/README_figures.md)。

同一请求内部、完整周期的 `b=V/(C+I)`、`B=V/C`，所以 `b/B=C/(C+I)`。跨请求时前后画像不同，不能直接套这个比值；整机 U 始终用真实计算时间积分。

## 7. 搜索范围、限制和复现

本轮完整排空 **{index['formal_count']} 次**；原仿真 4 秒筛选 **{index['pilot_screens']} 次、{index['distinct_pilot_screen_inputs']} 种输入**，另有 1 次截断测量校验；连续模型仅用于候选筛选，没有把模型的低值冒充正式结果。

另外做了 3 次运行至 8 秒的静态尾请求补偿：把少数 A 的计算时间预先增加一点，试图让后续大读取重新对齐。最强的一次 warm U 约 **93.59%**，但 `[4,8)` 又升到约 **97.35%**。它改变了请求画像，是第三种合成请求，不是纯粹重排；未完整排空，不提供 SLO。冻结输入和失败后的恢复结果均保留在补偿子目录。

这不是证明“OD 在所有欠载输入下都表现好”。它支持更有限的结论：**混合顺序可以造成局部及前期退化，SLO 可能比整机 U 更敏感；本轮找到的重复混合模式会逐渐恢复，尚未构成长期严重退化反例。**这里的 warm 只是统计窗口名称，不是“已经达到稳态”的证明。最优候选经过自适应调参和地址选择，不能视为普通随机负载的平均表现。

Once 使用原始类别 CIR 和逐层选路，改善是这套完整 L3 配置的结果，不能仅归因于选路；本轮未加入动态候选池、L2 重排或修改核心仿真。

- [输入和公式的通俗说明](experiment_design.md)
- [全部正式结果与筛选表](search_log.md) · [机器可读索引](search_index.json) · [正式结果 CSV](formal_results.csv)
- [晋级候选五个窗口与完整人口对照](native_phase_lock_comparison.md)
- [原始 data 画像分析](profile_analysis.md)
- [较大读取块数探索](larger_blocks/README.md)
- [静态尾请求校准探索](static_tail_calibration/README.md)

原始输入在 `inputs/`，完整原始输出在 `runs/<case>/result.json.gz`，每次命令、输入/输出/源码 SHA、FIFO、静态 QoS、块守恒、独占 Path 归属校验保存在同目录的 `command.json`。分析脚本不改变策略；未通过欠载约束的实验同样保留。

复现较强候选，使用新的输出标签，防止覆盖已有结果：

```bash
python results/od_mixed_underload_search_20260919/run_extended_trial.py \\
  --manifest results/od_mixed_underload_search_20260919/inputs/native_phase_lock_a101_b0995.json.gz \\
  --policy od_baseline --label replay_phase_od

python results/od_mixed_underload_search_20260919/run_extended_trial.py \\
  --manifest results/od_mixed_underload_search_20260919/inputs/native_phase_lock_a101_b0995.json.gz \\
  --policy once --label replay_phase_once
```
'''
    (HERE/'README.md').write_text(text)
    print('Wrote', HERE/'README.md')


if __name__ == '__main__':
    main()
