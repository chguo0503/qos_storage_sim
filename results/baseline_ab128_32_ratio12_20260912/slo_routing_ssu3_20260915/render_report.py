#!/usr/bin/env python3
"""Render the final report only after all planned full trials are validated."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {
    "baseline": "Baseline Random",
    "once": "Once（原流量分配）",
    "mild": "动态选路：温和",
    "aggressive": "动态选路：较强",
    "static_aggressive": "固定候选池：A全池、B收窄",
}
POLICIES = ("baseline", "mild", "aggressive", "static_aggressive")


def main():
    rows = list(csv.DictReader((HERE / "comparison.csv").open()))
    macro = list(csv.DictReader((HERE / "macro_summary.csv").open()))
    ix = {(r["policy"], r["window"]): r for r in macro}
    single = {(r["policy"], int(r["seed"]), r["window"]): r for r in rows}
    for policy in POLICIES:
        for window in ("warm_2_4s", "long_2_20s", "full_population"):
            assert int(ix[policy, window]["seed_count"]) == 3
            assert {s for p, s, w in single if p == policy and w == window} == {7, 19, 43}

    def value(policy, window, metric):
        return float(ix[policy, window]["macro_mean_" + metric])

    def table(window):
        out = ["| 策略 | NPU平均利用率 | SLO达标率 | A类达标率 | B类达标率 | 实际达标完成数/秒 |",
               "|---|---:|---:|---:|---:|---:|"]
        for policy in POLICIES:
            vals = [value(policy, window, m) for m in
                    ("U_percent", "slo_percent", "A_percent", "B_percent", "completion_window_timely_per_second")]
            out.append(f"| {NAMES[policy]} | " + " | ".join(f"{v:.2f}%" for v in vals[:4]) + f" | {vals[4]:.2f} |")
        return "\n".join(out)

    warm = "warm_2_4s"
    full = "full_population"
    base = ix["baseline", full]
    fixed = ix["static_aggressive", full]
    net = int(fixed["pooled_slo_passed"]) - int(base["pooled_slo_passed"])
    delta_a = int(fixed["pooled_A_passed"]) - int(base["pooled_A_passed"])
    delta_b = int(fixed["pooled_B_passed"]) - int(base["pooled_B_passed"])
    goodput_gain = 100 * (value("static_aggressive", full, "completion_window_timely_per_second") /
                          value("baseline", full, "completion_window_timely_per_second") - 1)
    paired = ["| 策略（全部seed7） | warm NPU利用率 | warm SLO达标率 | 长窗SLO达标率 | 全3840请求SLO达标率 |",
              "|---|---:|---:|---:|---:|"]
    for p in ("baseline", "once", "mild", "aggressive", "static_aggressive"):
        w = single[p, 7, warm]
        paired.append(f"| {NAMES[p]} | {float(w['U_percent']):.2f}% | {float(w['slo_percent']):.2f}% | "
                      f"{float(single[p, 7, 'long_2_20s']['slo_percent']):.2f}% | "
                      f"{float(single[p, 7, full]['slo_percent']):.2f}% |")
    per_seed = ["| seed | Baseline warm SLO / U | 动态较强 warm SLO / U | 固定候选池 warm SLO / U |",
                "|---|---:|---:|---:|"]
    for seed in (7, 19, 43):
        cells = []
        for p in ("baseline", "aggressive", "static_aggressive"):
            r = single[p, seed, warm]
            cells.append(f"{float(r['slo_percent']):.2f}% / {float(r['U_percent']):.2f}%")
        per_seed.append(f"| {seed} | " + " | ".join(cells) + " |")

    text = f"""**SSU3 Random：只改L3，能否改善请求达标率？**

可以。这批输入中，仅改变新I/O的候选path池，就提高了接纳后prefill延迟的SLO达标率；代价是NPU利用率下降、部分B请求超时。**固定候选池比本次第一版动态SLO启发式更有效，因此不能把收益归功于动态SLO判断。** 所有计划的完整仿真均已结束。

这里沿用之前的“SLO×1.5”口径：`prefill完成时刻 - 接纳时刻 <= 1.5 × 本请求总纯计算时间`。**不包含接纳前排队，也没有模拟真实首token事件。不能直接称为从用户到达开始计算的端到端TTFT。**

**同一份输入与硬件配置**

32 NPU，3 SSU × 40 GiB/s，每卡接收链路50 GiB/s，8层，batch=1。每卡40个A、80个B，独立Random打乱；种子7、19、43。每个种子共3840请求，15052800个I/O块，运行到全部完成。请求直接采用原实验冻结的data画像，没有修改读取量或计算时间。

| 请求 | 总输入 / miss | 每层读取量 | 每层计算C | B_i = V/C | 8层纯计算 | SLO允许额外等待 |
|---|---|---:|---:|---:|---:|---:|
| A：大读取、短计算 | 128K / 256 token | 175.66 MiB | 6.024 ms | 28.476 GiB/s | 48.192 ms | 24.096 ms |
| B：小读取、长计算 | 32K / 4096 token | 38.50 MiB | 28.593 ms | 1.315 GiB/s | 228.743 ms | 114.371 ms |

保留原分卡、每卡请求顺序、落盘布局、跨请求首层预取和path内部FIFO。CIR表不变，PIR仍不限速，没有丢请求、盘内重排或主动推迟I/O提交。“固定候选池”指选路规则固定，**每张NPU仍然混合运行Random A/B，不是固定长卡与短卡。**

**先看原来的warm [2,4)秒**

以下为三个种子分别统计后等权平均。NPU利用率按这个固定时间窗积分；SLO选取窗内接纳的请求并跟踪到完成，包括窗后完成的请求。最后一列则直接数这个时间窗内真正完成且达标的请求，除以2秒，两种选样口径不同。

{table(warm)}

固定候选池相对Baseline的warm SLO变化为{value('static_aggressive', warm, 'slo_percent') - value('baseline', warm, 'slo_percent'):+.2f}个百分点，NPU利用率变化为{value('static_aggressive', warm, 'U_percent') - value('baseline', warm, 'U_percent'):+.2f}个百分点。不能只展示前一个数字。

**扩大到[2,20)秒**

{table('long_2_20s')}

两个固定窗口内，32张NPU均持续有请求运行。窗口位置没有根据结果调整。

**再看完全相同的3840个请求，排除窗口选样的影响**

这里每种策略处理同一完整请求集合，每个请求权重相同。全程利用率包含开始和末尾排空，数值不能与warm利用率混作同一个指标。

{table(full)}

三个种子合计11520个相同请求中，固定候选池比Baseline净多{net}个达标，A类净变化{delta_a:+d}个，B类净变化{delta_b:+d}个。这里的“净变化”是通过数量之差，不表示每个原本达标的请求都继续达标。全程实际达标完成数/秒的三种子均值相对变化为{goodput_gain:+.2f}%。

全程还有一个精确关系：`NPU利用率 = 所有请求总纯计算时间 / (32 × 全部请求完成所需时间)`。这里总纯计算工作量固定，Baseline平均总历时为{value('baseline', full, 'makespan_ms') / 1000:.3f}秒，固定候选池为{value('static_aggressive', full, 'makespan_ms') / 1000:.3f}秒。后者用更长的总历时完成同样计算，所以全程利用率下降；这与多救回一些紧期限请求并不矛盾。该全程公式的分子不能直接拿来计算warm窗口，warm必须只累加窗口内实际发生的计算。

因此要分别看：达标率是否提高、是否伤害某一类别、每秒实际达标完成数是否增加。这次是有限输入全部排空的实验，不等于已测出在线稳定服务容量。

**与原Once配对比较**

Once只使用seed7对照，不能与三种子均值直接比较。本次远端重跑Baseline/Once都复现了历史结果。

{chr(10).join(paired)}

**为什么这样选路？用简单公式看**

```text
C = 本请求每层的纯计算时间
总纯计算时间 = 8 * C
接纳后prefill耗时 = 8 * C + 没有被计算覆盖的等待时间
SLO阈值 = 1.5 * 8 * C
所以允许的额外等待 = 0.5 * 8 * C = 4 * C
```

A只能额外等约24ms，B能等约114ms；A的读取量还更大。所以“B的计算时间长”不意味着应先给B更多存储服务。若目标是让更多请求达标，可以尝试在B尚能承受的范围内，将服务机会让给A。这个结论针对本次SLO目标，不是最大化平均NPU利用率的通用最优规则。

原Once在类别合法池里根据5ms拥塞快照选路。固定候选池让已接纳的A保留每盘96条合法path作为候选，B的候选池收窄到每盘8条（每组1条）；池内仍按Once选择。候选池大小不等于每时刻的活跃path数量，后者取决于实际积压。CIR配置虽然没改，但发生竞争时，哪些path有积压会改变实际服务份额。没有工作时的空余带宽借用仍按原模型运行，并非给B强制设置固定带宽上限。

跨请求首层预取时，下一请求尚无接纳时刻，仍走原Once完整合法池；上述收窄规则主要作用于接纳后的层。

动态版则每层重新估计：

```text
剩余等待预算 = 请求截止时刻 - 当前时刻 - 预计剩余纯计算时间
每个剩余读取层的等待预算 = 剩余等待预算 / 剩余读取层数
预算宽裕：每组只使用1条或2条path
预算紧张或已耗尽：使用原完整合法池
```

这里的6ms判断门槛与两档强度在看结果前选定；固定候选池对照是在看到动态版warm结果后增加的消融实验。它们没有经过更大数据集的验证。

固定版更好的一个可能原因来自动态规则本身：A接近完成时，剩余层数减少，算出的“每层剩余预算”可能升高，反而把它的候选池收窄。然而A每层读取量没有减少，这可能增加收尾阶段的等待。动态版还会在B预算耗尽时恢复B的完整池。这些是代码机制和代数支持的解释；本次没有完整记录每次选路决策，不能声称已经逐请求证明了各自的因果贡献。**第一版动态判断没有验证“收窄后的服务时间仍能保证整个请求达标”，不能视为成熟的截止时间调度器。**

只改L3也会改变各卡完成时刻及之后请求的接纳时刻，虽然L2顺序完全没改。因此warm窗口内各策略的请求集合可能不同，必须同时看上面的完整人口结果。

**实际需要什么信息？**

动态版需要上层提供本请求接纳时刻/SLO截止、计算进度、每层计算时间估计和逐盘读取量。固定候选池只用初始画像决定池宽，加上请求是否已经接纳；本次等价于A保留全池、B收窄，不跟踪实时剩余SLO。最强的实验收益并不依赖实时deadline判断。

两者盘侧都仍只提供原有5ms的path拥塞快照及QoS配置。盘本身不必推断哪个请求快超时。本实验没有读取未来事件或精确剩余块服务时间，也没有给新策略增加更频繁的盘侧查询。

当前仍将控制与策略计算延迟设为0，画像计算时间视为已知。真实部署中的画像误差、快照过期、控制成本会改变效果；本次只能证明该仿真和这组输入下存在L3选路的收益与代价。

**随机种子差异与复现材料**

{chr(10).join(per_seed)}

动态较强版在seed19的warm SLO并未优于Baseline，不能宣称每种Random顺序都有收益。下一步若继续研究，应把固定候选池作为必须击败的简单对照，以实际达标完成数/秒为收益指标，同时设置B类达标率或最低服务份额约束。

输入SHA、原核心代码SHA、完整请求数、原分卡/顺序以及FIFO/CIR无修改等检查均通过。前一次统计预览错误导致的失败运行已单独保留，不纳入结果。原有图和结果没有改动。

- [逐种子、逐窗口的精确结果](comparison.csv)
- [三种子等权汇总与合并分母](macro_summary.csv)
- [配对差值](paired_deltas.csv)
- [方法与实现限制](METHOD.md)
- [汇总及来源校验](comparison.json)
- [选路实现](slo_router.py)、[动态运行器](run_trial.py)、[固定候选池运行器](run_static_ablation.py)

在项目根目录运行 `python results/baseline_ab128_32_ratio12_20260912/slo_routing_ssu3_20260915/compare_results.py` 可从完整日志重新汇总；随后运行同目录 `render_report.py` 重新生成本文。
"""
    (HERE / "README.md").write_text(text)
    print(HERE / "README.md")


if __name__ == "__main__":
    main()
