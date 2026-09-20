# 文档原样输入：Random / Ordered 的 ASU 与 OD 对照

**已完成：12/12 次完整运行。** 本次保留文档的 10 种原始 data 画像、每卡每种两条；没有通过改变请求、错峰到达、放大计算时间或删除超限时段制造欠载。

## 主要结果：warm [2,4) 秒

| 顺序 | 策略 | 种子 | NPU 平均利用率 | 接纳起算 SLO×1 | 接纳起算 SLO×1.5 | 全32卡持续活跃 |
|---|---|---|---:|---:|---:|---|
| Random | ASU Baseline | [7, 19, 43] | 100.00% | 100.00% | 100.00% | 是 |
| Random | OD Baseline | [7, 19, 43] | 100.00% | 100.00% | 100.00% | 是 |
| Ordered | ASU Baseline | [7, 19, 43] | 99.99% | 97.71% | 100.00% | 是 |
| Ordered | OD Baseline | [7, 19, 43] | 99.83% | 83.12% | 100.00% | 是 |

- 这组原样输入没有复现 ASU 的明显低利用率：Random 与 Ordered 的 warm 利用率均接近100%。这组结果不能证明 ASU 在欠载时会显著降低利用率。
- Random：OD 相对 ASU 的 warm 利用率 +0.000 个百分点，接纳起算 SLO×1.5 达标率 +0.000 个百分点。
- Ordered：OD 相对 ASU 的 warm 利用率 -0.163 个百分点，接纳起算 SLO×1.5 达标率 +0.000 个百分点。
- Random / ASU Baseline：完整运行逐事件检查全部满足每盘 D<40。
- Random / OD Baseline：完整运行逐事件检查全部满足每盘 D<40。
- Ordered / ASU Baseline：完整运行逐事件检查存在 D≥40 的时段，不能称为逐盘全时严格欠载。
- Ordered / OD Baseline：完整运行逐事件检查存在 D≥40 的时段，不能称为逐盘全时严格欠载。

比较是同一种子、同一冻结输入上的 ASU 与 OD 成对比较。主表汇总 seed 7、19、43，每个种子权重相同。Random 的 seed 同时控制卡内顺序与仿真器同刻 ready NPU 的提交顺序；Ordered 保持相同卡内顺序，但 seed 仍会改变同刻提交仲裁，因此三份 Ordered 不能直接视为完全相同的数值复本。

## 这次输入具体是什么

| 项目 | 配置 |
|---|---|
| 硬件 | 32 NPU、3 SSU，每盘40 GiB/s；每卡接收链路50 GiB/s |
| 请求画像 | 总长度 `[32,64,80,128,160]K` × miss `[2048,4096]`，10种 |
| 配额 | 每卡每种画像2条，每卡20条，共640条 |
| 计算与读取 | 直接取根目录 data 的逐层 C、V，没有缩放 |
| 层数 | 实验8层、batch=1；data 源TTFT字段等价于78层 C |
| 到达 / 接纳 | 全部640条在t=0到达；各卡串行接纳并执行自己的队列 |
| Random | `Random(seed+npu_id).shuffle(20个具体请求)`，与文档画像队列逐项一致 |
| Ordered | 长度递增，同长度miss递增，每个画像两个副本相邻 |
| placement | Ring hash；固定请求身份决定落盘，重新排列不改变物理落盘 |
| 策略 | ASU每盘共享Path0；OD每盘32条独占Path，CIR=1.25 GiB/s、PIR不限 |

文档不是已经存在于旧目录中的640请求归档：旧 `diverse_data_ssu3_l3_20260916` 留存的是其他配比，且历史运行使用条带 placement。本次是**依照文档规则新建输入**，不是声称找回并复跑了不存在的旧640请求日志。[文档核查说明](doc_review.md)列出证据。

文档脚本在排序之后按队列位置编号。若直接把该位置编号用于 hash，同一个具体请求会因换顺序而换盘。因此本次用 `original_request_id=npu_id*20+canonical_ordinal` 固定物理身份，运行中的 `request_id` 仅标识队列位置。该修正保留了文档完整画像顺序；六份输入的物理请求集合和落盘一致，ASU/OD直接共享冻结manifest。

## 指标怎么读

```text
窗口 NPU 利用率 = 窗内全部卡实际计算时间 / (32 × 窗口长度)
接纳起算耗时 = prefill 完成时间 - 当前请求接纳时间
到达起算耗时 = prefill 完成时间 - 原始到达时间
请求纯计算基线 = 8 × data 中每层计算时间 C
SLO×k 达标 = 对应起算耗时 <= k × 请求纯计算基线 + 1e-9 ms
```

这两种耗时都以 prefill 完成为结束点，模拟器未生成真实首token事件。到达起算包括本卡前面请求造成的排队，接纳起算不包括它。本次是t=0到达的有限批次，接纳受执行反馈约束；不能把卡内顺序随机称为泊松或随机到达。

warm 使用 `2000 <= 接纳时刻 < 4000 ms` 的全部请求，跟踪到最终完成；窗后完成者仍计入SLO。两种起算方式使用同一接纳cohort，便于比较排队成本。每个种子先求平均、P50、P95，再对种子等权平均；分位数为nearest-rank `ceil(p*n)-1`，不是混合样本分位数。

| 顺序 | 策略 | 起算时刻 | 平均耗时 ms | P50 ms | P95 ms | SLO×1 | SLO×1.5 |
|---|---|---|---:|---:|---:|---:|---:|
| Random | ASU Baseline | 接纳 | 434.82 | 400.54 | 915.94 | 100.00% | 100.00% |
| Random | ASU Baseline | 到达 | 3419.10 | 3425.18 | 4362.23 | 0.00% | 0.00% |
| Random | OD Baseline | 接纳 | 436.36 | 400.54 | 915.94 | 100.00% | 100.00% |
| Random | OD Baseline | 到达 | 3416.83 | 3435.09 | 4371.46 | 0.00% | 0.00% |
| Ordered | ASU Baseline | 接纳 | 392.44 | 372.65 | 486.44 | 97.71% | 100.00% |
| Ordered | ASU Baseline | 到达 | 3293.52 | 3361.75 | 4107.22 | 0.00% | 0.00% |
| Ordered | OD Baseline | 接纳 | 393.09 | 373.97 | 486.44 | 83.12% | 100.00% |
| Ordered | OD Baseline | 到达 | 3295.05 | 3361.83 | 4111.71 | 0.00% | 0.00% |

| 顺序 | 策略 | seed | warm 接纳数 | warm 完成数 | 窗内接纳但窗后完成数 |
|---|---|---:|---:|---:|---:|
| Ordered | ASU Baseline | 19 | 160 | 160 | 32 |
| Ordered | ASU Baseline | 43 | 160 | 160 | 32 |
| Ordered | ASU Baseline | 7 | 160 | 160 | 32 |
| Ordered | OD Baseline | 19 | 160 | 160 | 32 |
| Ordered | OD Baseline | 43 | 160 | 160 | 32 |
| Ordered | OD Baseline | 7 | 160 | 160 | 32 |
| Random | ASU Baseline | 19 | 142 | 142 | 32 |
| Random | ASU Baseline | 43 | 145 | 145 | 32 |
| Random | ASU Baseline | 7 | 148 | 148 | 32 |
| Random | OD Baseline | 19 | 141 | 141 | 32 |
| Random | OD Baseline | 43 | 143 | 143 | 32 |
| Random | OD Baseline | 7 | 147 | 147 | 32 |

## 逐盘、逐事件验证负载

```text
D_s(t) = 各NPU当前已接纳请求的 sum(V_i,s / C_i)
严格欠载：每个事件区间、每张盘都满足 D_s < 40 GiB/s
超载：D_s > 40；等于40也不满足严格小于40
```

参考需求在当前请求计算和I/O stall期间均保留，按真实落盘量分到三盘，不把下一请求未接纳的首层预取重复算成第二个活跃请求。实际 SSD 供给则统计所有物理读取，包括该首层预取。因此参考需求并不是瞬时发出的I/O速率，两条线之比也不是瞬时NPU利用率。

**warm [2,4) 秒**：

| 顺序 | 策略 | SSU | 最大需求的种子间范围 GiB/s | 超过40的时间比例均值 | 实际供给均值 GiB/s |
|---|---|---:|---|---:|---:|
| Ordered | ASU Baseline | 0 | 37.136～37.186 | 0.0000% | 27.122 |
| Ordered | ASU Baseline | 1 | 40.512～40.555 | 18.6302% | 29.503 |
| Ordered | ASU Baseline | 2 | 39.615～39.748 | 0.0000% | 29.023 |
| Ordered | OD Baseline | 0 | 37.136～37.136 | 0.0000% | 27.121 |
| Ordered | OD Baseline | 1 | 40.775～40.775 | 18.7629% | 29.503 |
| Ordered | OD Baseline | 2 | 39.777～39.777 | 0.0000% | 29.022 |
| Random | ASU Baseline | 0 | 25.814～27.532 | 0.0000% | 23.525 |
| Random | ASU Baseline | 1 | 28.488～29.787 | 0.0000% | 25.932 |
| Random | ASU Baseline | 2 | 27.702～29.620 | 0.0000% | 25.477 |
| Random | OD Baseline | 0 | 25.814～27.532 | 0.0000% | 23.480 |
| Random | OD Baseline | 1 | 28.488～29.787 | 0.0000% | 25.876 |
| Random | OD Baseline | 2 | 27.702～29.575 | 0.0000% | 25.428 |

**完整运行，从0到最后请求完成**：

| 顺序 | 策略 | SSU | 最大需求的种子间范围 GiB/s | 超过40的时间比例均值 | 实际供给均值 GiB/s |
|---|---|---:|---|---:|---:|
| Ordered | ASU Baseline | 0 | 37.749～37.822 | 0.0000% | 23.273 |
| Ordered | ASU Baseline | 1 | 41.102～41.184 | 15.3886% | 25.512 |
| Ordered | ASU Baseline | 2 | 40.135～40.288 | 10.8810% | 25.030 |
| Ordered | OD Baseline | 0 | 37.673～37.673 | 0.0000% | 23.173 |
| Ordered | OD Baseline | 1 | 41.318～41.318 | 15.1918% | 25.404 |
| Ordered | OD Baseline | 2 | 40.103～40.103 | 10.3740% | 24.923 |
| Random | ASU Baseline | 0 | 27.532～28.675 | 0.0000% | 23.241 |
| Random | ASU Baseline | 1 | 29.787～32.287 | 0.0000% | 25.478 |
| Random | ASU Baseline | 2 | 29.620～31.496 | 0.0000% | 24.996 |
| Random | OD Baseline | 0 | 27.532～28.675 | 0.0000% | 23.206 |
| Random | OD Baseline | 1 | 29.787～32.287 | 0.0000% | 25.440 |
| Random | OD Baseline | 2 | 29.575～31.496 | 0.0000% | 24.958 |

完整输入的理论平均整机需求约74.086 GiB/s，低于120，只说明按纯计算时间加权的平均工作量较低，不能推出每盘每个时刻都欠载。Ring hash 的静态最大组合界约为38.806 / 42.335 / 41.293 GiB/s；后两盘超过40仅说明无法靠该界保证欠载，实际是否超限以本次事件轨迹为准。

[demand_intervals.csv](demand_intervals.csv)保存每盘每个完整事件区间及当时活跃卡数；[overload_intervals.csv](overload_intervals.csv)仅含 D>40 的区间；[capacity_violations.csv](capacity_violations.csv)含全部 D≥40 区间，包含恰等于容量的情况。没有抽样漏掉短暂尖峰；图的供给线为10ms平均，而需求和超限审计按精确事件边界进行。多个统计窗口会收录同一物理时段，跨窗口不能重复求和。

## 扩大到 [2,6) 秒

| 顺序 | 策略 | 种子 | NPU 平均利用率 | 接纳起算 SLO×1 | 接纳起算 SLO×1.5 | 全32卡持续活跃 |
|---|---|---|---:|---:|---:|---|
| Random | ASU Baseline | [7, 19, 43] | 100.00% | 100.00% | 100.00% | 是 |
| Random | OD Baseline | [7, 19, 43] | 100.00% | 100.00% | 100.00% | 是 |
| Ordered | ASU Baseline | [7, 19, 43] | 99.99% | 96.61% | 100.00% | 是 |
| Ordered | OD Baseline | [7, 19, 43] | 99.86% | 82.81% | 100.00% | 是 |

## 完整640请求集合

| 顺序 | 策略 | 种子 | NPU 平均利用率 | 接纳起算 SLO×1 | 接纳起算 SLO×1.5 | 全32卡持续活跃 |
|---|---|---|---:|---:|---:|---|
| Random | ASU Baseline | [7, 19, 43] | 99.50% | 94.90% | 100.00% | 否 |
| Random | OD Baseline | [7, 19, 43] | 99.35% | 94.95% | 100.00% | 否 |
| Ordered | ASU Baseline | [7, 19, 43] | 99.63% | 89.79% | 100.00% | 否 |
| Ordered | OD Baseline | [7, 19, 43] | 99.21% | 82.34% | 100.00% | 否 |

全程利用率包含启动和最后部分卡先结束的排空；其“全32卡持续活跃”为否并不表示输入错误。固定窗口是否全部卡持续活跃照实报告，不移动窗口、不剔除空闲卡。两策略warm内接纳的请求集合可能不同，完整640请求表用同一人口补充核对。

## 每张卡与各类请求

- [逐卡逐种子利用率](per_npu_metrics.csv)与[逐卡种子均值](per_npu_macro.csv)。
- [按总长度、类别、具体画像拆分的耗时和SLO](group_metrics.csv)，含接纳/到达两个时钟、×1/×1.5、平均/P50/P95及类内利用率；[种子等权汇总](group_macro.csv)。
- [逐请求样本](request_samples.csv)保留真实原始耗时与比值，不修改浮点值。

该输入的miss只有2048/4096，因此按当前代码分类只出现SL、LL，不存在SS、LS。类内利用率的分母是该类请求在窗口内的活跃卡时间，包含等待；不能把各类利用率直接等权平均成整机利用率。

## OD改变了什么

OD隔离不同NPU的FIFO路径，并把每盘32份CIR设为各1.25 GiB/s；它同时改变路径隔离和QoS分配，不能把差值仅解释为路径数。PIR没有硬上限，空闲份额仍按组间、组内两级WRR借用；等CIR不意味着任意活跃分布下每卡实际带宽都一样。

两策略保持L1分卡、L2卡内顺序及每Path内FIFO。每卡仍只提前一层；当前请求末层开始计算时会预取下一请求首层。预取真实字节已计入盘供给。它仍可能来不及完成，不能仅凭总平均带宽较低就断言没有I/O stall。

## 为什么这次没有明显低利用率

逐层时间戳独立复核显示：Random 两策略在 `[2,4)` 和 `[2,6)` 的真实 I/O barrier 等待为零，100% 不是四舍五入。画像计算时间约14.369～114.492ms，单卡总参考需求约1.315～3.701GiB/s；Random warm 最大逐盘需求只有29.787GiB/s。当前读取及时被上一层计算覆盖，FIFO排队没有暴露为窗口内NPU等待。这是本批时序的结果，不能外推为FIFO永远不会阻塞。

Ordered warm 有少量等待：ASU每种子累计6.009～6.319卡毫秒，OD为110.743卡毫秒，均涉及128K/miss2048。窗口共有 `32*2000=64000` 卡毫秒，OD损失的计算比例仅为 `110.743/64000`，所以利用率仍约99.827%。这些等待与盘1超限高度重叠，但现有日志不足以证明具体每次等待由哪个盘直接造成。

全程还包含启动、后续内部层/请求边界等待和末尾排空，不能把所有差异归于首层；也不能把 `1-U_full` 全当实际I/O等待。详细分解及证据见 [逐层等待分析](audit/notes.md)和[原始核查数据](audit/stall_mechanism.json)。



## 图与复核

- [Random：归一化 CDF（完整）](figures/normalized_cdf/random_asu_od_normalized_cdf.png)
- [Random：归一化 CDF（放大）](figures/normalized_cdf/random_asu_od_normalized_cdf_zoom.png)
- [Random / ASU Baseline：逐盘需求与供给](figures/per_ssu_bandwidth/random_asu_baseline_seed7_per_ssu.png)
- [Random / OD Baseline：逐盘需求与供给](figures/per_ssu_bandwidth/random_od_baseline_seed7_per_ssu.png)
- [Ordered：归一化 CDF（完整）](figures/normalized_cdf/ordered_asu_od_normalized_cdf.png)
- [Ordered：归一化 CDF（放大）](figures/normalized_cdf/ordered_asu_od_normalized_cdf_zoom.png)
- [Ordered / ASU Baseline：逐盘需求与供给](figures/per_ssu_bandwidth/ordered_asu_baseline_seed7_per_ssu.png)
- [Ordered / ASU Baseline：32卡计算及I/O等待](figures/ordered_timeline/ordered_asu_baseline_seed7_all_32npu.png)
- [Ordered / OD Baseline：逐盘需求与供给](figures/per_ssu_bandwidth/ordered_od_baseline_seed7_per_ssu.png)
- [Ordered / OD Baseline：32卡计算及I/O等待](figures/ordered_timeline/ordered_od_baseline_seed7_all_32npu.png)

CDF横轴为接纳起算耗时除以8层纯计算基线，种子等权。绘图副本仅把距离1×或1.5×门槛不超过1e-9ms的浮点边界规范化为精确门槛，使CDF(1)、CDF(1.5)与SLO口径一致；修正数量记录在render_checks，原始样本不变。完整图保留长尾，放大图不改变分母。

- [逐种子结果](comparison.csv)、[总体均值](macro_summary.csv)、[OD−ASU成对差值](paired_deltas.csv)
- [逐盘统计](disk_metrics.csv)、[逐盘种子汇总](disk_macro.csv)
- [输入审计](input_audit.json)、[结果及来源审计](summary_checks.json)、[绘图审计](render_checks.json)
- [原始记录独立复算：12次运行、36个统计窗口](audit/independent_results.json)、[独立审计程序](audit_results.py)
- [输入构造器](prepare_inputs.py)、[输入独立复核](verify_input_design.py)、[执行器](run_trial.py)

本次结论只覆盖这些请求和指定种子；实际控制通信、软件调度开销没有额外建模。既不预设OD必优，也不把“欠载”目录名当成运行已经通过严格欠载验证的证据。
