**旧A/B实验：SSU=3的三策略指标对照**

32 NPU、3 SSU×40 GiB/s，每NPU接收链路50 GiB/s，8层、batch=1；固定warm窗口[2,4)秒。

输入是两种data画像：A总输入128K、miss256；B总输入32K、miss4096。每张卡40A＋80B，即请求数量比1:2。Random每卡独立洗牌，Ordered为ABB重复40轮。这里不是后来的24画像实验。

**同seed7配对结果：与旧图对应**

| 输入顺序 | 策略 | NPU平均利用率 | TTFT SLO×1.5达标率 |
|---|---|---:|---:|
| Random | Baseline | 90.68% | 77.42% |
| Random | 原始 Once per layer | 91.24% | 79.59% |
| Random | 固定候选池 + Once | 85.38% | 91.05% |
| Ordered | Baseline | 71.01% | 71.43% |
| Ordered | 原始 Once per layer | 83.42% | 74.61% |
| Ordered | 固定候选池 + Once | 未测试 | 未测试 |

固定候选池在旧运行记录中叫static_aggressive：已接纳A每盘保留96条合法Path，B收窄到8条；池内继续使用原始Once。未接纳的跨请求首层仍使用完整池。Ordered固定候选池没有现成实测结果，不能用Random结果代替。

Random、seed7下，固定池相对原始Once：SLO变化+11.46个百分点，NPU利用率变化-5.86个百分点。

**Random已有的三种子等权平均：不能与Once单种子混比**

| 策略 | 种子 | NPU平均利用率 | TTFT SLO×1.5达标率 |
|---|---|---:|---:|
| Baseline | 7、19、43 | 92.74% | 79.36% |
| 固定候选池 + Once | 7、19、43 | 86.06% | 91.85% |
| 原始 Once per layer | 尚无完整三种子对照 | — | — |

原始Once仅有seed7。上面的三种子Baseline/固定池均值与seed7表不同，是统计种子集合不同，并非数据矛盾。本次只整理已完成结果，没有补跑。

**统计口径**

NPU平均利用率为窗口内实际计算卡时间/(32×2秒)。SLO选取[2,4)内接纳的请求并跟踪到最终完成，判断`prefill完成−接纳 <= 1.5×8×本请求每层纯计算时间`；不剔除窗后完成请求。不包含接纳前排队，也不是真实首token计时。

来源：[旧图对应的精确SLO和利用率](once_per_layer_ssu3_seed7/ttft_slo.csv)、[L3逐种子精确结果](slo_routing_ssu3_20260915/comparison.csv)、[L3汇总](slo_routing_ssu3_20260915/macro_summary.csv)、[固定池方法](slo_routing_ssu3_20260915/METHOD.md)。Random seed7的Baseline/Once已在两个来源中交叉核对一致，三种子均值已从逐种子表复算。
