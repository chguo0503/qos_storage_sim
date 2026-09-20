#!/usr/bin/env python3
"""Render the original document population; no input modification or simulation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from summarize_results import HERE, ROOT, ORDERS, POLICIES, SEEDS, read, sha, write_csv

NAMES = {"asu_baseline": "ASU Baseline", "od_baseline": "OD Baseline"}
ORDER_NAMES = {"random": "Random", "ordered": "Ordered"}
COLORS = {"asu_baseline": "#454545", "od_baseline": "#009E73"}


def rows(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def font_setup():
    path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=path).get_name(),
                         "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})


def mean_cdf(arrays, order, policy, x):
    return np.mean([np.searchsorted(arrays[order, policy, seed], x, side="right") /
                    len(arrays[order, policy, seed]) for seed in SEEDS], axis=0)


def cdf_plot(arrays, index, order, output, zoom=False):
    maximum = max(arrays[order, policy, seed][-1] for policy in POLICIES for seed in SEEDS)
    spread = max(0.0, maximum - 1)
    if zoom:
        pad = max(spread * .10, .00001) if spread > 1e-12 else .001
        xmin, xmax = 1 - pad, maximum + pad
    else:
        xmin, xmax = .99, max(1.6, math.ceil((maximum + .02) * 10) / 10)
    all_at_one = all(np.all(arrays[order, policy, seed] == 1) for policy in POLICIES for seed in SEEDS)
    fig, ax = plt.subplots(figsize=(12.8, 7.8))
    fig.subplots_adjust(left=.095, right=.975, bottom=.21, top=.80)
    for policy in POLICIES:
        knots = np.unique(np.concatenate([arrays[order, policy, seed] for seed in SEEDS]))
        xx = np.r_[xmin, knots[(knots > xmin) & (knots < xmax)], xmax]
        yy = mean_cdf(arrays, order, policy, xx)
        assert yy[0] == 0 and np.all(np.diff(yy) >= -1e-12)
        if not zoom:
            assert yy[-1] == 1
        reference = index[order, policy, "warm_2_4s"]
        label = (f"{NAMES[policy]}  |  SLO×1={float(reference['mean_admission_slo1_percent']):.2f}%"
                 f"，×1.5={float(reference['mean_admission_slo1p5_percent']):.2f}%")
        ax.step(xx, yy, where="post", color=COLORS[policy], linewidth=3.2 if policy == "asu_baseline" else 1.8,
                linestyle="--" if policy == "asu_baseline" else "-", label=label)
        for factor in (1, 1.5):
            if xmin <= factor <= xmax:
                ax.plot(factor, mean_cdf(arrays, order, policy, [factor])[0], "s" if policy == "asu_baseline" else "o",
                        color=COLORS[policy], markersize=7 if policy == "asu_baseline" else 4.5,
                        markeredgecolor="white", zorder=5)
    for factor in (1, 1.5):
        if xmin <= factor <= xmax:
            ax.axvline(factor, color="#666666", linestyle=":", linewidth=1)
            ax.text(factor, 1.045, f"SLO × {factor:g}", ha="center", fontsize=10, color="#555555")
    ax.set(xlim=(xmin, xmax), ylim=(0, 1.035), ylabel="累计请求比例（各种子 CDF 等权平均）",
           xlabel="归一化耗时 = 接纳至 prefill 完成耗时 / 本请求 8 层纯计算时间")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks(np.arange(0, 1.01, .1))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower right", framealpha=.97, fontsize=10.5, handlelength=3)
    fig.text(.095, .952, f"{ORDER_NAMES[order]}：ASU 与 OD 的归一化耗时 CDF"
             + ("（1 倍附近放大）" if zoom else ""), fontsize=19, fontweight="bold", va="top")
    fig.text(.095, .892, "原文档 10 画像 × 每卡各 2 次 · 32 NPU / 3 SSU · Ring hash · warm [2,4) 秒", color="#555555")
    if all_at_one:
        note = "本窗口两策略全部请求归一化耗时为 1，两条 CDF 完全重合；未人为错开曲线。"
    elif zoom:
        note = f"自适应放大 1 倍附近（{xmin:.6f}～{xmax:.6f}），完整样本分母不变。"
    else:
        note = "主图保留全部长尾；窗内接纳后追踪至完成，不剔除超时或窗后完成请求。"
    fig.text(.095, .10, note, fontsize=11)
    fig.text(.095, .055, "seed 7、19、43 等权；不含接纳前排队；1×/1.5×边界采用与 SLO 相同的 1e-9 ms 数值容差。",
             fontsize=10, color="#666666")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def disk_plot(raw, order, policy, output):
    window = raw["analysis"][0]
    segments = np.asarray(window["demand"]["segments"])
    edges = np.r_[segments[:, 0], segments[-1, 1]] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.075, top=.80, hspace=.36)
    ymax = max(45, float(segments[:, 2:].max()) * 1.15)
    for disk, ax in enumerate(axes):
        supply = raw["warm_ssd_10ms_GiB_s"][disk]
        assert abs(sum(supply) / 200 - window["SSD_GiB_s"][disk]) < 1e-7
        for a, z, *values in segments:
            if values[disk] > 40:
                ax.axvspan(a / 1000, z / 1000, color="#F4B4B4", alpha=.30, linewidth=0)
        ax.stairs(segments[:, disk + 2], edges, baseline=None, color="#D55E00", linewidth=1.7, label="参考需求 D_s=Σ(V_i,s/C_i)")
        ax.stairs(supply, np.linspace(2, 4, 201), baseline=None, color="#1768B4", linewidth=1.5, label="实际 SSD 供给（10ms 平均）")
        ax.axhline(40, color="#333333", linestyle="--", linewidth=1.2, label="每盘容量 40 GiB/s")
        ax.set(xlim=(2, 4), ylim=(0, ymax), ylabel=f"SSU {disk} 带宽（GiB/s）")
        ax.set_title(f"最大需求 {window['demand']['per_disk_max_GiB_s'][disk]:.3f} GiB/s · "
                     f"超过 40 的时间 {window['demand']['per_disk_overload_percent'][disk]:.3f}% · "
                     f"平均供给 {window['SSD_GiB_s'][disk]:.3f} GiB/s", loc="left", fontsize=11)
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#F4B4B4", alpha=.4)); labels.append("逐事件超载时段")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .902), ncol=2,
               frameon=False, fontsize=10, handlelength=3, columnspacing=2.5)
    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle(f"{ORDER_NAMES[order]} · {NAMES[policy]}：逐盘需求与实际供给\n"
                 "文档原样输入 · seed 7 · warm [2,4) 秒 · 保留真实超限时段", y=.985, fontsize=15)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def ordered_timeline(raw, policy, output):
    window = raw["analysis"][0]
    summary = raw["summary"]
    batches = summary["microbatch_metrics"]
    owners = {rid: batch["npu_id"] for batch in batches for rid in batch["member_request_ids"]}
    fig, ax = plt.subplots(figsize=(14.5, 12.0))
    fig.subplots_adjust(left=.15, right=.98, bottom=.065, top=.85)
    for npu in range(32):
        active, compute = [], []
        for request in summary["request_metrics"]:
            if owners[request["request_id"]] != npu:
                continue
            a, z = max(2000, request["admission_time_ms"]), min(4000, request["completion_time_ms"])
            if z > a:
                active.append((a / 1000, (z - a) / 1000))
        for batch in batches:
            if batch["npu_id"] != npu:
                continue
            for layer in batch["layer_metrics"]:
                a, z = max(2000, layer["compute_start_ms"]), min(4000, layer["compute_end_ms"])
                if z > a:
                    compute.append((a / 1000, (z - a) / 1000))
        ax.broken_barh(active, (npu - .34, .68), facecolors="#E69F00", linewidth=0)
        ax.broken_barh(compute, (npu - .34, .68), facecolors="#1768B4", linewidth=0)
    for a, z, *demand in window["demand"]["segments"]:
        ax.broken_barh([(a / 1000, (z - a) / 1000)], (-1.52, .38),
                      facecolors="#C73535" if max(demand) > 40 else "#CDE5D4", linewidth=0)
    labels = [f"NPU {n:02d}  U={window['per_npu_U_percent'][n]:.2f}%" for n in range(32)]
    ax.set_yticks(list(range(32)), labels=labels, fontsize=9)
    ax.text(1.997, -1.33, "逐盘容量检查", ha="right", va="center", fontsize=9)
    ax.set(xlim=(2, 4), ylim=(31.8, -2.15), xlabel="时间（秒）")
    ax.grid(axis="x", alpha=.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor="#1768B4", label="实际计算"), Patch(facecolor="#E69F00", label="I/O stall"),
               Patch(facecolor="#CDE5D4", label="容量条：各盘 D≤40"), Patch(facecolor="#C73535", label="容量条：至少一盘 D>40")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.565, .915), ncol=4, fontsize=10, frameon=False)
    fig.suptitle(f"Ordered · {NAMES[policy]}：32 张 NPU 的实际计算与 I/O 等待\n"
                 f"seed 7 · warm [2,4) 秒 · 整机 U={window['U_percent']:.2f}% · 文档原样输入",
                 fontsize=16, y=.977)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


def fmt(value, unit="%", digits=2):
    return "—" if value in (None, "") else f"{float(value):.{digits}f}{unit}"


def markdown(checks, comparison, macros, disk_macros, image_links):
    ix = {(row["order"], row["policy"], row["window"]): row for row in macros}
    complete = checks["status"] == "complete"

    def main_table(window):
        out = ["| 顺序 | 策略 | 种子 | NPU 平均利用率 | 接纳起算 SLO×1 | 接纳起算 SLO×1.5 | 全32卡持续活跃 |",
               "|---|---|---|---:|---:|---:|---|"]
        for order in ORDERS:
            for policy in POLICIES:
                row = ix.get((order, policy, window))
                if row:
                    values = [row["seeds"], fmt(row["mean_U_percent"]), fmt(row["mean_admission_slo1_percent"]),
                              fmt(row["mean_admission_slo1p5_percent"]), "是" if row["all_seeds_all_npus_active"] == "True" else "否"]
                else:
                    values = ["无", "未完成", "未完成", "未完成", "未完成"]
                out.append(f"| {ORDER_NAMES[order]} | {NAMES[policy]} | " + " | ".join(values) + " |")
        return "\n".join(out)

    def latency_table():
        out = ["| 顺序 | 策略 | 起算时刻 | 平均耗时 ms | P50 ms | P95 ms | SLO×1 | SLO×1.5 |",
               "|---|---|---|---:|---:|---:|---:|---:|"]
        for order in ORDERS:
            for policy in POLICIES:
                row = ix.get((order, policy, "warm_2_4s"))
                for clock, clock_name in (("admission", "接纳"), ("arrival", "到达")):
                    fields = [(f"mean_{clock}_latency_{stat}_ms", "") for stat in ("mean", "p50", "p95")]
                    fields += [(f"mean_{clock}_slo{factor}_percent", "%") for factor in ("1", "1p5")]
                    values = [fmt(row[field], suffix) if row else "未完成" for field, suffix in fields]
                    out.append(f"| {ORDER_NAMES[order]} | {NAMES[policy]} | {clock_name} | " + " | ".join(values) + " |")
        return "\n".join(out)

    def disk_table(window):
        out = ["| 顺序 | 策略 | SSU | 最大需求的种子间范围 GiB/s | 超过40的时间比例均值 | 实际供给均值 GiB/s |",
               "|---|---|---:|---|---:|---:|"]
        for row in disk_macros:
            if row["window"] != window:
                continue
            span = f"{float(row['min_maximum_demand_GiB_s']):.3f}～{float(row['max_maximum_demand_GiB_s']):.3f}"
            out.append(f"| {ORDER_NAMES[row['order']]} | {NAMES[row['policy']]} | {row['ssu_id']} | {span} | "
                       f"{fmt(row['mean_overload_percent'], digits=4)} | {fmt(row['mean_actual_GiB_s'], '', 3)} |")
        return "\n".join(out)

    findings = []
    if complete:
        if all(float(ix[order, "asu_baseline", "warm_2_4s"]["mean_U_percent"]) >= 99 for order in ORDERS):
            findings.append("- 这组原样输入没有复现 ASU 的明显低利用率：Random 与 Ordered 的 warm 利用率均接近100%。"
                            "这组结果不能证明 ASU 在欠载时会显著降低利用率。")
        for order in ORDERS:
            a, b = (ix[order, p, "warm_2_4s"] for p in POLICIES)
            du = float(b["mean_U_percent"]) - float(a["mean_U_percent"])
            ds = float(b["mean_admission_slo1p5_percent"]) - float(a["mean_admission_slo1p5_percent"])
            findings.append(f"- {ORDER_NAMES[order]}：OD 相对 ASU 的 warm 利用率 {du:+.3f} 个百分点，"
                            f"接纳起算 SLO×1.5 达标率 {ds:+.3f} 个百分点。")
        for order in ORDERS:
            for policy in POLICIES:
                part = [row for row in comparison if row["order"] == order and row["policy"] == policy and row["window"] == "full_population"]
                strict = all(row["strict_underload_all_disks"] == "True" for row in part)
                findings.append(f"- {ORDER_NAMES[order]} / {NAMES[policy]}：完整运行逐事件检查"
                                + ("全部满足每盘 D<40。" if strict else "存在 D≥40 的时段，不能称为逐盘全时严格欠载。"))
    else:
        findings.append("正式结果尚未齐全，下表仅汇总完整运行。各行种子数可能不同，不能将不匹配的暂时均值视作成对结论。")
    status = f"**{'已完成' if complete else '暂时结果'}：{checks['complete_cases']}/{checks['planned_cases']} 次完整运行。**"
    counts = ["| 顺序 | 策略 | seed | warm 接纳数 | warm 完成数 | 窗内接纳但窗后完成数 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in comparison:
        if row["window"] == "warm_2_4s":
            counts.append(f"| {ORDER_NAMES[row['order']]} | {NAMES[row['policy']]} | {row['seed']} | "
                          f"{row['admitted_in_window']} | {row['completed_in_window']} | {row['completed_after_window']} |")
    mechanism = ""
    if complete and (HERE / "audit/stall_mechanism.json").exists():
        assert read(HERE / "audit/stall_mechanism.json")["latency_decomposition_checks_passed"]
        mechanism = """## 为什么这次没有明显低利用率

逐层时间戳独立复核显示：Random 两策略在 `[2,4)` 和 `[2,6)` 的真实 I/O barrier 等待为零，100% 不是四舍五入。画像计算时间约14.369～114.492ms，单卡总参考需求约1.315～3.701GiB/s；Random warm 最大逐盘需求只有29.787GiB/s。当前读取及时被上一层计算覆盖，FIFO排队没有暴露为窗口内NPU等待。这是本批时序的结果，不能外推为FIFO永远不会阻塞。

Ordered warm 有少量等待：ASU每种子累计6.009～6.319卡毫秒，OD为110.743卡毫秒，均涉及128K/miss2048。窗口共有 `32*2000=64000` 卡毫秒，OD损失的计算比例仅为 `110.743/64000`，所以利用率仍约99.827%。这些等待与盘1超限高度重叠，但现有日志不足以证明具体每次等待由哪个盘直接造成。

全程还包含启动、后续内部层/请求边界等待和末尾排空，不能把所有差异归于首层；也不能把 `1-U_full` 全当实际I/O等待。详细分解及证据见 [逐层等待分析](audit/notes.md)和[原始核查数据](audit/stall_mechanism.json)。

"""
    return f"""# 文档原样输入：Random / Ordered 的 ASU 与 OD 对照

{status} 本次保留文档的 10 种原始 data 画像、每卡每种两条；没有通过改变请求、错峰到达、放大计算时间或删除超限时段制造欠载。

## 主要结果：warm [2,4) 秒

{main_table('warm_2_4s')}

{chr(10).join(findings)}

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

{latency_table()}

{chr(10).join(counts)}

## 逐盘、逐事件验证负载

```text
D_s(t) = 各NPU当前已接纳请求的 sum(V_i,s / C_i)
严格欠载：每个事件区间、每张盘都满足 D_s < 40 GiB/s
超载：D_s > 40；等于40也不满足严格小于40
```

参考需求在当前请求计算和I/O stall期间均保留，按真实落盘量分到三盘，不把下一请求未接纳的首层预取重复算成第二个活跃请求。实际 SSD 供给则统计所有物理读取，包括该首层预取。因此参考需求并不是瞬时发出的I/O速率，两条线之比也不是瞬时NPU利用率。

**warm [2,4) 秒**：

{disk_table('warm_2_4s')}

**完整运行，从0到最后请求完成**：

{disk_table('full_population')}

完整输入的理论平均整机需求约74.086 GiB/s，低于120，只说明按纯计算时间加权的平均工作量较低，不能推出每盘每个时刻都欠载。Ring hash 的静态最大组合界约为38.806 / 42.335 / 41.293 GiB/s；后两盘超过40仅说明无法靠该界保证欠载，实际是否超限以本次事件轨迹为准。

[demand_intervals.csv](demand_intervals.csv)保存每盘每个完整事件区间及当时活跃卡数；[overload_intervals.csv](overload_intervals.csv)仅含 D>40 的区间；[capacity_violations.csv](capacity_violations.csv)含全部 D≥40 区间，包含恰等于容量的情况。没有抽样漏掉短暂尖峰；图的供给线为10ms平均，而需求和超限审计按精确事件边界进行。多个统计窗口会收录同一物理时段，跨窗口不能重复求和。

## 扩大到 [2,6) 秒

{main_table('long_2_6s')}

## 完整640请求集合

{main_table('full_population')}

全程利用率包含启动和最后部分卡先结束的排空；其“全32卡持续活跃”为否并不表示输入错误。固定窗口是否全部卡持续活跃照实报告，不移动窗口、不剔除空闲卡。两策略warm内接纳的请求集合可能不同，完整640请求表用同一人口补充核对。

## 每张卡与各类请求

- [逐卡逐种子利用率](per_npu_metrics.csv)与[逐卡种子均值](per_npu_macro.csv)。
- [按总长度、类别、具体画像拆分的耗时和SLO](group_metrics.csv)，含接纳/到达两个时钟、×1/×1.5、平均/P50/P95及类内利用率；[种子等权汇总](group_macro.csv)。
- [逐请求样本](request_samples.csv)保留真实原始耗时与比值，不修改浮点值。

该输入的miss只有2048/4096，因此按当前代码分类只出现SL、LL，不存在SS、LS。类内利用率的分母是该类请求在窗口内的活跃卡时间，包含等待；不能把各类利用率直接等权平均成整机利用率。

## OD改变了什么

OD隔离不同NPU的FIFO路径，并把每盘32份CIR设为各1.25 GiB/s；它同时改变路径隔离和QoS分配，不能把差值仅解释为路径数。PIR没有硬上限，空闲份额仍按组间、组内两级WRR借用；等CIR不意味着任意活跃分布下每卡实际带宽都一样。

两策略保持L1分卡、L2卡内顺序及每Path内FIFO。每卡仍只提前一层；当前请求末层开始计算时会预取下一请求首层。预取真实字节已计入盘供给。它仍可能来不及完成，不能仅凭总平均带宽较低就断言没有I/O stall。

{mechanism}

## 图与复核

{chr(10).join(f'- [{label}]({path})' for label, path in image_links)}

CDF横轴为接纳起算耗时除以8层纯计算基线，种子等权。绘图副本仅把距离1×或1.5×门槛不超过1e-9ms的浮点边界规范化为精确门槛，使CDF(1)、CDF(1.5)与SLO口径一致；修正数量记录在render_checks，原始样本不变。完整图保留长尾，放大图不改变分母。

- [逐种子结果](comparison.csv)、[总体均值](macro_summary.csv)、[OD−ASU成对差值](paired_deltas.csv)
- [逐盘统计](disk_metrics.csv)、[逐盘种子汇总](disk_macro.csv)
- [输入审计](input_audit.json)、[结果及来源审计](summary_checks.json)、[绘图审计](render_checks.json)
- [原始记录独立复算：12次运行、36个统计窗口](audit/independent_results.json)、[独立审计程序](audit_results.py)
- [输入构造器](prepare_inputs.py)、[输入独立复核](verify_input_design.py)、[执行器](run_trial.py)

本次结论只覆盖这些请求和指定种子；实际控制通信、软件调度开销没有额外建模。既不预设OD必优，也不把“欠载”目录名当成运行已经通过严格欠载验证的证据。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    checks = read(HERE / "summary_checks.json")
    if args.require_complete and checks["status"] != "complete":
        raise SystemExit("Formal results incomplete; refusing final report.")
    sources = dict(checks["source_sha256"])
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, ("source changed", name)
    for name in ("summary_checks.json", "comparison.csv", "macro_summary.csv", "request_samples.csv", "disk_macro.csv"):
        sources[str((HERE / name).relative_to(ROOT))] = sha(HERE / name)
    for name in ("audit/stall_mechanism.json", "audit/notes.md", "audit/independent_results.json"):
        if (HERE / name).exists():
            sources[str((HERE / name).relative_to(ROOT))] = sha(HERE / name)
    comparison, macros, samples, disk_macros = (rows(name) for name in
                                               ("comparison.csv", "macro_summary.csv", "request_samples.csv", "disk_macro.csv"))
    index = {(row["order"], row["policy"], row["window"]): row for row in macros}
    arrays = defaultdict(list)
    snapping = {"1": 0, "1.5": 0}
    for row in samples:
        if row["in_warm_2_4s"] != "True":
            continue
        duration, ideal = float(row["admission_latency_ms"]), float(row["own_compute_ms"])
        ratio = duration / ideal
        for factor in (1.0, 1.5):
            if abs(duration - factor * ideal) <= 1e-9:
                if ratio != factor:
                    snapping[f"{factor:g}"] += 1
                ratio = factor
        arrays[row["order"], row["policy"], int(row["seed"])].append(ratio)
    arrays = {key: np.sort(value) for key, value in arrays.items()}
    out = HERE / "figures"
    for name in ("normalized_cdf", "per_ssu_bandwidth", "ordered_timeline"):
        (out / name).mkdir(parents=True, exist_ok=True)
    font_setup()
    links, curve_points, threshold_checks = [], [], []
    for order in ORDERS:
        if all((order, policy, seed) in arrays for policy in POLICIES for seed in SEEDS):
            for policy in POLICIES:
                for factor, code in ((1, "1"), (1.5, "1p5")):
                    value = float(mean_cdf(arrays, order, policy, [factor])[0])
                    expected = float(index[order, policy, "warm_2_4s"][f"mean_admission_slo{code}_percent"])
                    assert abs(value * 100 - expected) < 1e-8
                    threshold_checks.append(dict(order=order, policy=policy, factor=factor, cdf=value, slo_percent=expected))
                knots = np.unique(np.r_[1, 1.5, np.concatenate([arrays[order, policy, seed] for seed in SEEDS])])
                for x, y in zip(knots, mean_cdf(arrays, order, policy, knots)):
                    curve_points.append(dict(order=order, policy=policy, latency_ratio=float(x), mean_cdf=float(y)))
            for zoom in (False, True):
                target = out / "normalized_cdf" / f"{order}_asu_od_normalized_cdf{'_zoom' if zoom else ''}.png"
                cdf_plot(arrays, index, order, target, zoom)
                links.append((f"{ORDER_NAMES[order]}：归一化 CDF" + ("（放大）" if zoom else "（完整）"), str(target.relative_to(HERE))))
        for policy in POLICIES:
            case = next((case for case in checks["cases"] if (case["order"], case["policy"], case["seed"]) == (order, policy, 7)), None)
            if case:
                raw = read(HERE / "runs" / case["case"] / "result.json.gz")
                target = out / "per_ssu_bandwidth" / f"{order}_{policy}_seed7_per_ssu.png"
                disk_plot(raw, order, policy, target)
                links.append((f"{ORDER_NAMES[order]} / {NAMES[policy]}：逐盘需求与供给", str(target.relative_to(HERE))))
                if order == "ordered":
                    target = out / "ordered_timeline" / f"ordered_{policy}_seed7_all_32npu.png"
                    ordered_timeline(raw, policy, target)
                    links.append((f"Ordered / {NAMES[policy]}：32卡计算及I/O等待", str(target.relative_to(HERE))))
    write_csv("cdf_points.csv", curve_points, ("order", "policy", "latency_ratio", "mean_cdf"))
    (HERE / "README.md").write_text(markdown(checks, comparison, macros, disk_macros, links))
    assert all(sha(ROOT / name) == digest for name, digest in sources.items()), "source changed during rendering"
    audit = dict(status=checks["status"], no_simulation_started=True, source_files_unchanged=True,
                 source_sha256=sources, source_script_sha256=sha(Path(__file__)),
                 plotted_policies=list(POLICIES), generated_png=[name for _, name in links],
                 generated_png_sha256={name: sha(HERE / name) for _, name in links},
                 threshold_checks=threshold_checks, cdf_numeric_tolerance_ms=1e-9,
                 cdf_boundary_normalization_count=snapping, original_samples_unchanged=True,
                 full_CDF_tail_preserved=True, zoom_keeps_original_denominator=True, visual_review="pending")
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=checks["status"], png_count=len(links), report=str(HERE / "README.md")), ensure_ascii=False))


if __name__ == "__main__":
    main()
