#!/usr/bin/env python3
"""Build final Markdown only after all five 60-second runs are independently audited."""
from pathlib import Path
import csv,gzip,json,math

HERE=Path(__file__).resolve().parent
AUD=HERE/'audit_existing'

def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt') as f:return json.load(f)

def aggregate(a,start,end):
    rows=[w for w in a['windows'] if w['duration_ms']==2000 and w['start_ms']>=start and w['end_ms']<=end]
    assert math.isclose(sum(w['duration_ms'] for w in rows),end-start)
    count=sum(w['warm_admission_SLO']['count'] for w in rows)
    passed=sum(w['warm_admission_SLO']['passed'] for w in rows)
    return dict(U=100*sum(w['compute_ms'] for w in rows)/32/(end-start),SLO=100*passed/count,
                all_active=all(w['all_32_active'] for w in rows),count=count,passed=passed)

def fast_long_cards(window):
    return sum(c['fast_short_C_ms']>0 and c['roles']['long']['compute_ms']>0 for c in window['per_npu'])

def main():
    cases={(mode,policy):read(AUD/f'long_main_{mode}_{policy}_validation.json')
           for mode in ['ordered','random'] for policy in ['baseline','once']}
    backup=read(AUD/'long_backup_ordered_baseline_validation.json')
    assert all(a['technical_audit_passed'] for a in [*cases.values(),backup])
    ordered=cases['ordered','baseline'];w=ordered['primary_window'];p=ordered['input_audit']['profiles']
    names={('random','baseline'):'Random Baseline',('ordered','baseline'):'Ordered Baseline',
           ('random','once'):'Random Once per layer',('ordered','once'):'Ordered Once per layer'}
    rows=[]
    for key in [('random','baseline'),('ordered','baseline'),('random','once'),('ordered','once')]:
        a=cases[key];m=a['primary_window'];parts=[v['U_percent'] for v in a['windows'] if v['duration_ms']==2000]
        rows.append(f'| {names[key]} | {m["U_percent"]:.4f}% | {aggregate(a,20000,60000)["U"]:.4f}% | {min(parts):.4f}–{max(parts):.4f}% | {m["warm_admission_SLO"]["rate_percent"]:.4f}% |')
    window_rows=[]
    for a,b in [(2000,10000),*[(k,k+10000) for k in range(10000,60000,10000)]]:
        vals=[aggregate(cases[key],a,b)['U'] for key in [('random','baseline'),('ordered','baseline'),('random','once'),('ordered','once')]]
        window_rows.append(f'| [{a/1000:g}, {b/1000:g}) 秒 | '+' | '.join(f'{v:.4f}%' for v in vals)+' |')
    prof_rows=[]
    for q in sorted(p,key=lambda q:(q['nql'],q['seq_len_k'])):
        role='长请求' if q['role']=='long' else '桥接请求（真实计算）' if q['nql']==4096 else '短请求'
        prof_rows.append(f'| {role} | {q["seq_len_k"]}K | {q["nql"]:,} | {q["raw_C_ms"]:.6f} | {q["actual_D_MiB"]:.3f} | {q["raw_D_gib"]/(q["raw_C_ms"]/1000):.6f} |')
    profile_stats=[]
    for key,v in sorted(w['by_profile'].items()):
        profile_stats.append(f'| {key} | {v["conditional_U_percent"]:.4f}% | {v["fleet_compute_share"]*100:.4f}% | {v["warm_admission_SLO"]["rate_percent"]:.4f}% | {v["warm_admission_SLO"]["count"]} |')
    randomU=cases['random','baseline']['primary_window']['U_percent']
    onceU=cases['ordered','once']['primary_window']['U_percent']
    maxdisk=max(a['nominal_full_run']['max_ssu_gib_s'] for a in cases.values())
    conditions=all(a['primary_conditions_passed'] for a in cases.values())
    phase=read(AUD/'long_main_ordered_baseline_phase.json')
    phase_max=max(z['first_long_L0_compute_start']['spread'] for z in phase['cycles'])
    group_U=[sum(c['device_U_percent'] for c in w['per_npu'][start:start+16])/16 for start in [0,16]]
    mixing_rows=[];capacity_rows=[]
    for key in [('random','baseline'),('ordered','baseline'),('random','once'),('ordered','once')]:
        a=cases[key];m=a['primary_window'];small=[z for z in a['windows'] if z['duration_ms']==2000]
        warm=next(z for z in small if z['start_ms']==2000)
        mixing_rows.append(f'| {names[key]} | {fast_long_cards(m)}/32 | {m["min_fast_short_fraction"]*100:.4f}% | {min(z["long_fraction"] for z in m["per_npu"])*100:.4f}% | {m["cards_fast_and_long_ge5percent"]}/32 | {m["minimum_per_card_role_switches"]} | {fast_long_cards(warm)}/32 | {min(fast_long_cards(z) for z in small)}–{max(fast_long_cards(z) for z in small)}/32 |')
        proxy=read(AUD/f'long_main_{key[0]}_{key[1]}_prefetch_demand.json')['variants']
        capacity_rows.append(f'| {names[key]} | {a["nominal_full_run"]["max_ssu_gib_s"]:.6f} | {proxy["compute_budget_only"]["max_ssu_gib_s"]:.6f} | {proxy["through_exposed_wait"]["max_ssu_gib_s"]:.6f} | {proxy["through_exposed_wait"]["any_ssu_over40_ms"]:.6f} |')
    text=f'''**总输入超过 10K、每卡反复运行长短流：60 秒测试结果**

本次主输入在 `[2,60)` 秒的 Baseline NPU 平均利用率为 **{w['U_percent']:.4f}%**，同一批请求独立随机打乱卡内顺序后为 **{randomU:.4f}%**，差 **{randomU-w['U_percent']:.4f} 个百分点**。相同定序输入使用 Once per layer 为 **{onceU:.4f}%**。

这是一次连续执行的长输入，原始事件按窗口裁剪；没有拼接低值片段、添加空闲时间或改变核心策略。下表统一使用 seed 7。后续窗口和每个 2 秒子窗一并披露，不能只引用主均值。

| 输入与策略 | U：[2,60) | U：[20,60) | 29 个连续 2 秒子窗 U 范围 | 接纳后 SLO×1.5 达标率：[2,60) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

| 分段 | Random Baseline | Ordered Baseline | Random Once | Ordered Once |
|---|---:|---:|---:|---:|
{chr(10).join(window_rows)}

主四格的技术和预定混合/容量核查：**{'全部通过' if conditions else '未全部通过，请按逐项结果判断'}**。完整数值直接来自 [逐结果独立审计状态](audit_existing/long_runtime_audit_status.json)。这些检查针对整个 `[2,60)` 窗口；不等于任意一个 2 秒小窗内每张卡都包含长、短请求。

**输入是怎样分给 32 张卡的**

32 NPU、6 SSU，每盘 40 GiB/s，每卡接收链路 50 GiB/s；每请求 8 层、每卡同时计算一条请求。每个画像的计算时间和读取量都直接来自原始 `data`，没有插值、缩放、额外 padding 或人为延时。

Baseline 将 I/O 提交到每个 SSU 自己的 Path0；六盘各有队列，并非共享一个全局 FIFO。本报告的 Once per layer 对应策略参数 `once`，使用每5ms采集的共享快照，在每个请求层、每个 SSU 上一次规划该层各块的路径。它不表示整层只能用一条路径，也不是 `new_once`。沿用项目既有模拟器和静态 CIR 配置，没有新增策略代码。

| 作用 | 总输入 | 新计算的 NQL | 单层 C（ms） | 单层读取（MiB） | 读取/C（GiB/s） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(prof_rows)}

K=1,024 token。NQL 是 miss/new-compute token 数，不是总输入长度。32K/NQL4096 的辅助请求需要计算 4K token，因此它的 C=28.593ms，接近长请求；不能把它冒充 C=7.257ms 的快短请求。

- **NPU 0–15：**反复执行 `6 个 176K 长请求 → 1 个 64K/1024 短请求 → 1 个 32K/4096 辅助请求`。
- **NPU 16–31：**反复执行 `46 个 32K/1024 短请求 → 1 个 48K/1024 短请求 → 1 个 64K/1024 短请求 → 1 个 176K 长请求`。每卡从这个 49 项环的不同位置开始，位置为 `floor((NPU编号−16)×49/16)`，分散偶发长请求。
- 每卡队列按完整周期扩充到至少 62 秒纯计算，共 **20,032** 条请求。所有到达时间为 0；每卡依自己的实际进度推进，无全局屏障、墙钟换卡或迁移。
- Random 保留每卡完全相同的请求身份、C/D、到达时间和落盘，只独立打乱完整队列。两策略使用同一份冻结 manifest。

原始输入配额中，每张卡剔除辅助请求后的快短计算至少占自身计算的 **5.4963%**；长、短两类最低份额为 **8.1008%**，后一口径把辅助请求计入短类。实际 `[2,60)` 窗口中，剔除辅助请求后的快短最低份额为 **{w['min_fast_short_fraction']*100:.4f}%**，每卡至少切换 **{w['minimum_per_card_role_switches']} 次**。完整输入配额和实际窗口统计分别核算。

| 输入与策略 | 长窗含快短+长的卡数 | 每卡快短份额的最小值 | 每卡长请求份额的最小值 | 快短与长各占至少5%的卡数 | 每卡最少角色切换 | 原 [2,4) 秒含快短+长的卡数 | 各 2 秒窗含快短+长卡数的范围 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(mixing_rows)}

份额的分母是该卡在 `[2,60)` 内的实际计算时间，快短只包含 NQL=1024 的 32K/48K/64K 请求，辅助请求单独列出。卡数按窗口内实际计算是否同时覆盖快短和长请求判断，分子排除辅助请求。角色切换次数沿用原 long/short 标记，其中辅助请求计入 short，因此不能用切换次数替代快短时间核查。**如果仍要求原 `[2,4)` 秒或每个 2 秒小窗内全部 32 张卡都运行两类，应使用上表核查；长窗通过不能替代这个更强条件。**

Ordered Baseline 的等待分布也不均匀：NPU 0–15 的组平均 U 为 **{group_U[0]:.4f}%**，NPU 16–31 为 **{group_U[1]:.4f}%**。每卡都混合，但两组长短比例明显不同；低值主要集中在快短请求占比更高的后 16 卡，不能说全部 32 卡都降到了约90%。

[输入完整配额 CSV](audit_existing/input_assignment.csv) · [原始画像 CSV](audit_existing/input_profiles.csv) · [输入身份与源码独立核查](audit_existing/long_input_audit.json)

**为什么前一组会恢复，这一构造要改变什么**

前一轮直接把长、短段反复交换，或在连续长请求中插入几个很快的短请求，Baseline 在长窗恢复到约 97%–99.8%。层日志显示：快短内部发生的 I/O 等待让不同卡的进度分开，随后切回长请求时首层预取来不及，又进一步拉开长读取时刻。

新构造选用了 C=12.626ms 的实际短请求，并在它之后放入低读取量、C=28.593ms 的辅助请求。其意图是让下一长请求的首层能在辅助计算期间预取完，从而保持前 16 张卡的长读取集中。另一组卡仍反复执行真正快的短请求，也定期运行长请求。

一个便于理解的量级估算是：16 张卡各读一层长请求，合计 `16×240.625 MiB`；假设均匀分到 6 盘并独占其带宽，读完约需 `16×240.625÷(6×40×1024)×1000 = 15.666ms`。32K 快短每层只算 7.257ms，若它的读取排在这批长读取后面，就可能来不及隐藏等待。辅助请求每层算 28.593ms，为下一批长读取提供了更长预算。这个估算只解释设计方向，忽略其他竞争和链路；实际等待及能否维持同步必须看事件日志。

能否保持相位以日志中的零 stall 与逐轮开始时刻跨度为准，不能仅凭 C 大就作保证。本次完整长运行中，前 16 卡除各自初始化首层外，出现暴露等待的层数为 **{phase['long_group_full_run']['after_initial_L0_exposed_stall_layer_count']}**，合计 **{phase['long_group_full_run']['after_initial_L0_exposed_stall_ms']:.6f}ms**；共记录 **{len(phase['cycles'])} 轮**长段开始，各轮跨卡时间跨度最大 **{phase_max:.9f}ms**。见 [主 Baseline 相位审计](audit_existing/long_main_ordered_baseline_phase.json)。保持时刻集中是人工安排的结果，不代表真实 agent 流量自然具有该顺序。

层日志直接证明：以短请求为主的卡反复等到读取就绪后才能继续计算。结合单 QoS path 的排队规则与集中长读取，SSU 队列竞争是机制解释；本批层日志没有记录每个物理块的 FIFO 前驱，读取生命周期还包含链路传输，因此不能把每次停顿都归给某一条长请求。另一张卡的长计算本身没有占用短请求所在卡的计算单元。辅助计算是真实 `data` 请求的工作，也完整计入各策略的计算量和 SLO；它增加的工作占比必须披露。

[第51秒附近的真实交接放大图](figures/separate/ordered_baseline_late_handoff.png) 画在同一绝对时间轴上：NPU0 的下一长请求首层读取耗时 **15.495ms**，小于前驱辅助层的 **28.593ms** 计算预算，因而无停顿；NPU21 的一层32K快短读取耗时 **15.643ms**，超过 **7.257ms** 计算预算，暴露出 **8.386ms** 等待。图选择50–60秒内第一次完整辅助交接及同窗后16卡最大内部层等待；它展示一个具体实例，频率与整机损失仍以完整长窗统计为准。

以下画像统计对应 **Ordered Baseline、seed7、[2,60) 秒**：

| 画像（总长K:NQL） | 该画像计算/占用时间 | 占整机全部计算的份额 | 接纳后 SLO×1.5 | 接纳人口 |
|---|---:|---:|---:|---:|
{chr(10).join(profile_stats)}

画像自身利用率的分母是该画像占用的卡时间；整机 U 的分母始终为 `32×58秒`，两者不能直接做算术平均。

**带宽约束与统计口径**

四个主运行的全时域逐盘名义峰值最高为 **{maxdisk:.6f} GiB/s**。名义需求沿用原约定：当前已接纳请求的单层逐盘读取量/C，下一请求首层预取不另加一份；所有实际 I/O 完整执行。新条带位置为 `(block_index+npu_id)%6`，即使任意卡同时选择最重画像，其充分上界也只有 **39.898850 GiB/s < 40**。

另有一个计入实际跨请求读取量的描述性核查：对每层，用真实下一层读取量除以当前计算时间，在跨请求位置改用下一请求首层的读取量；分别只覆盖计算预算和延长到暴露等待结束。后两列代理只检查 `[2,60)`；初始化首层没有前一层计算预算，因此不纳入该代理，实际初始化 I/O 仍完整执行且在 warm 前结束。结果另存 `*_prefetch_demand.json`。该代理与原名义需求定义不同，不能笼统称为更严格的约束。

| 输入与策略 | 原名义需求全程峰值 | 下一层真实读取/C：计算区间峰值 | 同一速率延长至等待结束的峰值 | 延长代理超过 40 的时间（ms） |
|---|---:|---:|---:|---:|
{chr(10).join(capacity_rows)}

带宽列单位为 GiB/s，取 6 盘中的最大值。这些代理不是实际 SSD 吞吐，也不是所有预取截止时间必定可满足的证明；不能把瞬时读突发与长期过载混为一谈。

主四格中，只有 Ordered Baseline 在上述补充代理下零超限；Random Baseline、Random Once 以及 Ordered Once 均有超过 40 的区间，其中 Ordered Once 的延长代理超限约7.257ms。因此“随机与定序都欠载”的比较只在原先约定的当前请求 D/C 定义下成立，不能换用新口径仍宣称各格全部通过。代理超限不等于实际 SSD 吞吐超过物理上限。

利用率只计窗口内真实计算；所有 32 卡都保留在分母中，空闲与 I/O 等待分别记录。SLO 只纳入窗口内接纳的请求并跟踪至完成，判断 `完成−接纳 ≤ 1.5×8×原始C`。这是接纳后的处理时间代理，不是包含入队等待的端到端 TTFT；各策略的窗口接纳人口也不同。全部请求在 t=0 到达，不能把高接纳后达标率当成真实在线服务的 TTFT 达标率。

为什么整机利用率约90%，这个 SLO 仍可能接近100%？在本次每卡串行、每请求8层的条件下，单请求的接纳后效率为 `8C/(8C+等待)`；允许耗时达到纯计算的1.5倍，就相当于只要求该请求效率至少 **66.67%**。所以不少请求确实等了 I/O，仍能达标。这里达标率来自逐请求检查，不能仅用画像平均利用率反推；这个阈值对当前等待幅度不敏感。

**稳健性与限制**

60 秒是比原 2 秒长得多的一次有限仿真，不能证明数学上的无限时间行为或真实生产负载的常见性。输入刻意保留同组相近计算时长与相关顺序，采用 batch_size=1、8层、缓存 miss 对应的 `data` 成本。计算抖动、不同模型、连续批处理、真实到达轨迹等仍可能破坏该相位关系。

另用 seed19、43 作了同周期的短时复验，Baseline 在 `[2,12)` 的 U 分别为 **89.7742%、89.7280%**，对应接纳后 SLO×1.5 为 **99.9260%、99.5554%**。32 卡在这个 10 秒窗口内都实际运行了快短和长请求，每个独立 2 秒小窗包含两类的卡数为 25。详见 [复验结果](heldout_12s_retry1/findings.md)。这些 seed 改变提交次序，输入身份完全相同；不是两批独立业务画像，也不是两次新的 60 秒实验。

备选 12L 周期的 `[2,60)` Baseline U 为 **{backup['primary_window']['U_percent']:.4f}%**；它是另一批配额，剔除辅助请求后的快短纯计算配额约3.019%，实际长窗最低快短份额为 **{backup['primary_window']['min_fast_short_fraction']*100:.4f}%**。它通过原筛选，但未满足每卡真正快短至少5%的补充条件；其实际下一层预取代理峰值 **40.627239 GiB/s**，也未通过该补充核查。不能把它混入主输入均值，或当作这两个补充条件的合格例子。

保留了八个初筛/机制对照：四个普通循环失败候选约97%–99.8%，三个多短桥接候选约90.7%–91.7%，一个前16卡只穿插较慢辅助请求、后16卡仍有快短请求的对照约90.1%。未把它们当作主长窗独立重复样本。曾因编排等待上限设置不足停止五个刚启动的长验证，以及两个额外 seed 的一次未知中断；它们没有完整结果，单独保留状态，不计作策略性能失败或成功。正式长结果统一来自 `long_validation_7200s`。

**文件入口**

- [Ordered Baseline：连续长时间线](figures/separate/ordered_baseline_long.png)
- [Random Baseline：连续长时间线](figures/separate/random_baseline_long.png)
- [Ordered Once：连续长时间线](figures/separate/ordered_once_long.png)
- [Random Once：连续长时间线](figures/separate/random_once_long.png)
- [全部独立 PNG / PDF / SVG](figures/separate/README.md)
- [正式长运行：后半段的局部预取与等待](figures/separate/ordered_baseline_late_handoff.png)
- [10秒候选的真实局部交接图与解释](notes/bridge_method.md)
- [预先声明的检查方式](study_plan.md)、[输入构造与运行脚本](run_candidates.py)、[最终输入参数](long_specs.json)
'''
    (HERE/'report.md').write_text(text)
    earlier=HERE/'baseline_long_result.md'
    if earlier.exists():
        earlier.write_text(earlier.read_text().replace('**Baseline 的 60 秒验证已完成；Once 对照仍在运行。**',
            '**这是先行保存的 Baseline 结果。全部 Baseline / Once 长测现已完成，见 [完整报告](report.md)。**'))
    summary=[]
    for key,a in cases.items():
        q=a['primary_window'];summary.append(dict(mode=key[0],policy=key[1],seed=a['submit_seed'],start_ms=2000,end_ms=60000,
            U_percent=q['U_percent'],SLO_percent=q['warm_admission_SLO']['rate_percent'],conditions=a['primary_conditions_passed'],
            max_ssu_gib_s=a['nominal_full_run']['max_ssu_gib_s']))
    with (HERE/'main_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps({'report':str(HERE/'report.md'),'main':summary},ensure_ascii=False))

if __name__=='__main__':main()
