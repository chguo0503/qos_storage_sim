#!/usr/bin/env python3
"""Render verified tables and separate PNG figures; never run simulations."""
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
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from summarize_results import HERE, ROOT, CLASSES, SEEDS, WINDOWS, read, sha, csv_write

NAMES = {"asu_baseline": "ASU Baseline", "od_baseline": "OD Baseline", "once": "原始 Once per layer",
         "static": "固定候选池 + Once", "mild": "动态候选池：温和", "aggressive": "动态候选池：较强"}
SCENARIOS = {"semi": "间歇过载配比", "full": "持续过载配比"}
MAIN = ("asu_baseline", "od_baseline", "once", "static")
ALL = tuple(NAMES)
STYLE = {"asu_baseline": ("#454545", "--"), "od_baseline": ("#009E73", "-"),
         "once": ("#1768B4", "-."), "static": ("#D55E00", "-"),
         "mild": ("#9B59B6", ":"), "aggressive": ("#CC79A7", (0, (5, 2, 1, 2)))}


def csv_read(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def font_setup():
    path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})


def macro_cdf(arrays, scenario, policy, xx):
    values = [np.searchsorted(arrays[scenario, policy, seed], xx, side="right") /
              len(arrays[scenario, policy, seed]) for seed in SEEDS]
    return np.mean(values, axis=0)


def cdf_figure(arrays, index, scenario, policies, output, *, zoom=False):
    maximum = max(arrays[scenario, policy, seed][-1] for policy in policies for seed in SEEDS)
    xmax = 3.0 if zoom else math.ceil((maximum + .15) * 2) / 2
    fig, ax = plt.subplots(figsize=(13.4, 8.0))
    fig.subplots_adjust(left=.092, right=.975, top=.81, bottom=.19)
    for policy in policies:
        knots = np.unique(np.concatenate([arrays[scenario, policy, seed] for seed in SEEDS]))
        xx = np.r_[.9, knots[(knots > .9) & (knots < xmax)], xmax]
        yy = macro_cdf(arrays, scenario, policy, xx)
        assert yy[0] == 0 and np.all(np.diff(yy) >= -1e-12)
        if not zoom:
            assert yy[-1] == 1
        at_slo = float(macro_cdf(arrays, scenario, policy, [1.5])[0])
        expected = float(index[scenario, policy, "warm_2_4s"]["mean_slo_percent"])
        assert abs(at_slo * 100 - expected) < 1e-8
        color, style = STYLE[policy]
        ax.step(xx, yy, where="post", color=color, linestyle=style, linewidth=2.1,
                label=f"{NAMES[policy]}  |  SLO={expected:.2f}%")
        ax.plot(1.5, at_slo, "o", color=color, markersize=5, markeredgecolor="white", zorder=5)
    ax.set(xlim=(.9, xmax), ylim=(0, 1.035), ylabel="累计请求比例（各种子 CDF 等权平均）",
           xlabel="归一化耗时 = 接纳至 prefill 完成耗时 / 本请求纯计算时间")
    ax.xaxis.labelpad = 12
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if xmax <= 4:
        ax.xaxis.set_major_locator(MultipleLocator(.25 if zoom else .5))
    ax.axvline(1.5, color="#666666", linestyle=":", linewidth=1.4)
    ax.text(1.5, 1.049, "SLO × 1.5", ha="center", fontsize=11, color="#555555")
    ax.grid(axis="y", color="#CDD2D6", alpha=.65, linewidth=.7)
    ax.legend(loc="lower right", frameon=True, framealpha=.97, edgecolor="#DDDDDD",
              fontsize=10.8, handlelength=3.2, labelspacing=.75, borderpad=.8)
    suffix = "（阈值附近放大）" if zoom else ""
    label = f"{len(policies)} 种策略"
    fig.text(.092, .95, f"{SCENARIOS[scenario]}：{label}的归一化耗时 CDF{suffix}",
             ha="left", va="top", fontsize=19, fontweight="bold")
    fig.text(.092, .893, "32 NPU / 3 SSU × 40 GiB/s  ·  Ring hash  ·  Random  ·  warm [2,4) 秒", color="#555555")
    note = ("仅放大横轴 0.9～3 倍；纵轴仍使用全部请求，长尾请看完整图。" if zoom else
            "x=1.5 处就是 SLO×1.5 达标率；主图保留完整长尾，不平滑、不拟合。")
    fig.text(.092, .084, note, fontsize=11)
    fig.text(.092, .045, "seed 7、19、43 等权；窗内接纳后跟踪至完成，不含接纳前排队。各策略窗内请求集合可能不同。",
             fontsize=10, color="#666666")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def disk_figure(raw, scenario, policy, output):
    window = raw["analysis"][0]
    segments = np.asarray(window["demand"]["segments"])
    edges = np.r_[segments[:, 0], segments[-1, 1]] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.075, top=.80, hspace=.36)
    maximum = max(60, float(segments[:, 2:].max()) * 1.08)
    for disk, ax in enumerate(axes):
        supply = raw["warm_ssd_10ms_GiB_s"][disk]
        assert abs(sum(supply) / 200 - window["SSD_GiB_s"][disk]) < 1e-7
        ax.stairs(segments[:, disk + 2], edges, baseline=None, color="#D55E00", linewidth=1.6,
                  label="当前请求参考需求 V/C")
        ax.stairs(supply, np.linspace(2, 4, 201), baseline=None, color="#1768B4", linewidth=1.5,
                  label="SSD 实际供给（10 毫秒平均）")
        ax.axhline(40, color="#333333", linestyle="--", linewidth=1.1, label="每盘容量 40 GiB/s")
        ax.set(xlim=(2, 4), ylim=(0, maximum), ylabel=f"SSU {disk} 带宽（GiB/s）")
        ax.set_title(f"需求超过容量的时间：{window['demand']['per_disk_overload_percent'][disk]:.2f}%  ·  "
                     f"实际平均供给：{window['SSD_GiB_s'][disk]:.2f} GiB/s", loc="left", fontsize=11)
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .905), ncol=3,
               fontsize=10, frameon=False, handlelength=3.0, columnspacing=2.2)
    axes[-1].set_xlabel("时间（秒）")
    fig.suptitle(f"{SCENARIOS[scenario]}：{NAMES[policy]} 逐盘需求与实际供给\n"
                 "32 NPU / 3 SSU · Ring hash · Random · seed 7 · warm [2,4) 秒", fontsize=15, y=.985)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def metric_text(value, suffix="%"):
    return "—" if value in (None, "") else f"{float(value):.2f}{suffix}"


def report_text(checks, rows, macros, images):
    index = {(row["scenario"], row["policy"], row["window"]): row for row in macros}
    complete = checks["status"] == "complete"
    full_input = next(case for case in read(HERE / "input_audit.json")["cases"]
                      if case["scenario"] == "full")
    full_capacity_bound = 100 * min(1, 40 / max(full_input["per_ssu_ideal_mean_GiB_s"]))

    def table(window, policies=MAIN, classes=False):
        out = ["| 配比 | 策略 | 已完成种子 | NPU 平均利用率 | SLO×1.5 达标率 | 达标完成数/秒 |",
               "|---|---|---:|---:|---:|---:|"]
        if classes:
            out[0] += " SS 达标率 | SL 达标率 | LS 达标率 | LL 达标率 |"
            out[1] += "---:|---:|---:|---:|"
        for scenario in SCENARIOS:
            for policy in policies:
                row = index.get((scenario, policy, window))
                if row is None:
                    values = ["无", "未完成", "未完成", "未完成"]
                else:
                    values = ["、".join(str(s) for s in json.loads(row["seeds"])), metric_text(row["mean_U_percent"]),
                              metric_text(row["mean_slo_percent"]),
                              metric_text(row["mean_timely_completions_per_second"], "")]
                line = f"| {SCENARIOS[scenario]} | {NAMES[policy]} | " + " | ".join(values) + " |"
                if classes:
                    line += " " + " | ".join(metric_text(row[f"mean_{c}_slo_percent"]) if row else "未完成"
                                                for c in CLASSES) + " |"
                out.append(line)
        return "\n".join(out)

    findings = []
    if complete:
        for scenario in SCENARIOS:
            for policy, reference in (("od_baseline", "asu_baseline"), ("once", "od_baseline"),
                                      ("static", "od_baseline"), ("static", "once")):
                current = index[scenario, policy, "warm_2_4s"]
                previous = index[scenario, reference, "warm_2_4s"]
                u = float(current["mean_U_percent"]) - float(previous["mean_U_percent"])
                slo = float(current["mean_slo_percent"]) - float(previous["mean_slo_percent"])
                findings.append(f"- {SCENARIOS[scenario]}，{NAMES[policy]} 相对 {NAMES[reference]}："
                                f"NPU 利用率 {u:+.2f} 个百分点，SLO 达标率 {slo:+.2f} 个百分点。")
            warm_delta = (float(index[scenario, "od_baseline", "warm_2_4s"]["mean_slo_percent"]) -
                          float(index[scenario, "asu_baseline", "warm_2_4s"]["mean_slo_percent"]))
            full_delta = (float(index[scenario, "od_baseline", "full_population"]["mean_slo_percent"]) -
                          float(index[scenario, "asu_baseline", "full_population"]["mean_slo_percent"]))
            if warm_delta * full_delta < 0:
                findings.append(f"- {SCENARIOS[scenario]}存在窗口方向变化：OD 相对 ASU 的 SLO 差值，"
                                f"warm 两秒为 {warm_delta:+.2f} 个百分点，完整请求集合为 {full_delta:+.2f} 个百分点。"
                                "因此不能把某个固定窗口的排名直接当作整批请求的排名。")
    else:
        findings.append("正式运行尚未全部完成。表格只使用已完整排空的运行，种子不足 3 的行是暂时结果；"
                        "不将运行中 warm 预览并入正式平均，也不据此宣称策略优劣。")
    status = (f"**已完成 {checks['complete_cases']}/{checks['planned_cases']} 次完整仿真。**" if complete else
              f"**暂时报告：仅完成 {checks['complete_cases']}/{checks['planned_cases']} 次完整仿真。**")
    aggregation_note = ("各种子先独立统计，再对 seed 7、19、43 等权取平均；百分点变化均来自同一种子的成对对照。" if complete else
                        "各行仅对已经完成的种子等权平均，实际种子列在表中。不同种子集合的暂时均值不可直接作成对比较。")
    overload = ["| 配比 | 策略 | seed | SSU 0/1/2 超容量时间 | SSU 0/1/2 实际平均供给（GiB/s） |",
                "|---|---|---:|---|---|"]
    for row in sorted(rows, key=lambda r: (r["scenario"], ALL.index(r["policy"]), int(r["seed"]))):
        if row["window"] != "warm_2_4s" or row["policy"] not in ("asu_baseline", "od_baseline"):
            continue
        rates = " / ".join(metric_text(row[f"SSU{d}_overload_percent"]) for d in range(3))
        supply = " / ".join(metric_text(row[f"SSU{d}_actual_GiB_s"], "") for d in range(3))
        overload.append(f"| {SCENARIOS[row['scenario']]} | {NAMES[row['policy']]} | {row['seed']} | {rates} | {supply} |")
    bycategory = ["| 配比 | 策略 | SS 类内利用率 | SL 类内利用率 | LS 类内利用率 | LL 类内利用率 |",
                  "|---|---|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        for policy in MAIN:
            row = index.get((scenario, policy, "warm_2_4s"))
            values = [metric_text(row[f"mean_{c}_active_U_percent"]) if row else "未完成" for c in CLASSES]
            bycategory.append(f"| {SCENARIOS[scenario]} | {NAMES[policy]} | " + " | ".join(values) + " |")
    tail_table = ["| 配比 | 策略 | 已完成种子 | SLO×1.5 | P95 倍数均值 | P99 倍数均值 | 各种子最大倍数的均值 |",
                  "|---|---|---|---:|---:|---:|---:|"]
    for scenario in SCENARIOS:
        for policy in MAIN:
            row = index.get((scenario, policy, "warm_2_4s"))
            seeds = "、".join(str(s) for s in json.loads(row["seeds"])) if row else "无"
            values = [metric_text(row["mean_slo_percent"]) if row else "未完成"]
            for field in ("latency_ratio_p95_nearest_rank", "latency_ratio_p99_nearest_rank", "latency_ratio_max"):
                values.append(metric_text(row.get("mean_" + field), "") if row else "未完成")
            tail_table.append(f"| {SCENARIOS[scenario]} | {NAMES[policy]} | {seeds} | " + " | ".join(values) + " |")
    tail_findings = []
    if complete:
        for scenario in SCENARIOS:
            old, new = (index[scenario, policy, "warm_2_4s"] for policy in ("asu_baseline", "od_baseline"))
            p95_old = float(old["mean_latency_ratio_p95_nearest_rank"])
            p95_new = float(new["mean_latency_ratio_p95_nearest_rank"])
            if p95_new < p95_old and float(new["mean_slo_percent"]) < float(old["mean_slo_percent"]):
                tail_findings.append(f"{SCENARIOS[scenario]}的 OD 相对 ASU，SLO×1.5 达标率下降，"
                                     f"但各种子 P95 的均值由 {p95_old:.2f} 降至 {p95_new:.2f} 倍。"
                                     "说明更严格阈值下的达标人数与慢请求尾部并不是同一个优化目标。")
    figure_links = "\n".join(f"- [{name}]({path})" for name, path in images)
    return f"""# OD Baseline 与 ASU、Once：多样 data 输入对照

{status} 本目录是新实验，落盘方式全部为 Ring hash。保留旧实验的请求画像、卡内顺序和种子，重新运行各策略，旧条带结果没有混入本次对照。

## warm [2,4) 秒的主要结果

{aggregation_note}

{table('warm_2_4s', classes=True)}

{chr(10).join(findings)}

这些是当前构造配比上的实测差异。OD 同时改变路径隔离和静态 QoS 分配；它与 ASU 的差值不能只归因于“路径数变多”。Once/候选池仍保留原共享类别 QoS，并不是在 OD 的 32 条独占路径上运行。候选池的额外收益应看它与原始 Once 的差值。

## 输入与配置

| 项目 | 本次配置 |
|---|---|
| NPU / SSU | 32 NPU，3 SSU，每盘 40 GiB/s，NPU 接收链路 50 GiB/s |
| 层数 / 并发 | 8 层，batch=1；每卡连续接纳自己的下一请求 |
| 输入画像 | data 中 6 种总长度 `[32,64,80,128,160,200]K` × 4 种 miss `[256,1024,2048,4096]`，共 24 种 |
| L1 绑定 | 固定分卡，每卡包含全部 24 种画像 |
| L2 顺序 | 每卡独立 Random；seed 7、19、43 |
| 到达 | 所有请求在 t=0 到达，有限输入；接纳受前一请求完成约束 |
| placement | Ring hash，同场景同种子的 6 个策略读取完全相同的冻结输入 |
| 主统计窗口 | `[2,4)` 秒，另报 `[2,6)` 秒及完整请求集合 |

间歇配比：每一种总长度下，miss 256/1024/2048/4096 的请求数量为 1/1/1/2；每卡 30 个请求，共 960 个。持续配比为 3/2/1/1；每卡 42 个请求，共 1344 个。计算时间和读取量直接取 data，没有缩放。配比是实验设计，不声称是真实 agent 的经验流量分布。

这里的“间歇/持续”沿用旧实验配比名称。Ring hash 并不保证逐盘严格均分，因此是否及多久过载，以本次下面的逐盘实测为准，不能沿用历史条带的时间比例。

## 两种 Baseline 的具体差别

| 策略 | 每盘路径 | CIR / PIR | 空闲份额 |
|---|---|---|---|
| ASU Baseline | 所有 NPU 使用一个共享 Path 0 | 保留原静态 QoS 配置 | 沿用原生调度 |
| OD Baseline | 32 条有效 Path，一张 NPU 固定独占一条 | 每条 CIR=40/32=1.25 GiB/s，PIR 不设上限，等权 | 有请求的路径可以借用空闲路径的带宽 |
| 原始 Once | 原共享路径池，每层按拥塞选择路径 | 原类别静态 QoS | 原生调度 |
| 固定/动态候选池 + Once | 限制或调整新 I/O 可选池，池内沿用 Once | 同原始 Once | 原生调度 |

OD 的 1.25 GiB/s 是每盘每卡的保障份额，不是任何时刻都不能超过的硬限速。32 条路径同时持续有积压时，各卡长期平分盘带宽；部分路径空闲时，其余卡可以使用剩余带宽。空闲借用仍经过 8 个硬件组及组内 Path 的两级调度，不能进一步声称任意活跃卡集合都获得相同瞬时速率。所有策略保持路径内 FIFO；L1 绑定和 L2 顺序不变，没有运行中 CIR 改写或盘内 I/O 重排。

## 如何统计

```text
NPU 平均利用率 = 窗口内所有卡的实际计算时长之和 / (32 × 窗口长度)
请求归一化耗时 = (prefill 完成时刻 - 接纳时刻) / 本请求 8 层纯计算时间
SLO×1.5 达标 = 请求归一化耗时 <= 1.5
```

SLO 选取窗口内接纳的请求，并追踪到最终完成；窗后完成、已经超时的请求都保留。它不含接纳前排队，也不是真正的首 token 事件，因此本文是沿用前次实验的 TTFT 代理口径。不同策略在同一窗口内接纳的请求集合可能不同；完整请求集表使用同一批请求补充核对。

“达标完成数/秒”按窗口内真正完成的请求统计。`arrival_slo_percent` 在 CSV 另列，从 t=0 的到达时刻起算，包含等待接纳；不能与主表口径混用。

## 请求慢了多少：warm 窗口的尾部

{chr(10).join(tail_table)}

{chr(10).join(tail_findings)}

这里的倍数仍为 `(完成时刻 - 接纳时刻) / 本请求纯计算时间`。先在每个种子内将全部入选请求的倍数从小到大排序，采用不插值的 nearest-rank 定义：

```text
n = 该种子窗内接纳的请求数
P95 = 排序后第 ceil(0.95*n) 个值，即 Python 下标 ceil(0.95*n)-1
P99 = 排序后第 ceil(0.99*n) 个值，即 Python 下标 ceil(0.99*n)-1
最大倍数 = 该种子最慢请求的归一化耗时
主表 P95 均值 = (种子7的P95 + 种子19的P95 + 种子43的P95) / 3
```

P95 表示该种子至少 95% 的请求倍数不超过此值；SLO×1.5 则只问有多少请求落在 1.5 倍以内。某策略可以把极慢请求从十几倍降到几倍，同时让一部分原来刚好达标的请求超过 1.5 倍，因此长尾改善与达标率下降并不矛盾。是否更好取决于目标，不能只选有利的指标。

表中分位数是**各种子分位数的等权均值**，不是混合所有请求后的分位数，也不是平均 CDF 在 95%/99% 处的反函数。最后一列也是各种子最大值的均值，不是跨种子的最坏值。逐种子计数、取值排名及最大值见 [latency_tail.csv](latency_tail.csv)，宏均值和种子间最小/最大值见 [latency_tail_macro.csv](latency_tail_macro.csv)；同样提供 `[2,6)` 和完整集合口径。

## 类别利用率

{chr(10).join(bycategory)}

类内利用率 = 该类别在窗口内的计算卡时间 / 该类别在窗口内的活跃卡时间（包含 I/O 等待）。四类不能直接等权平均成整机利用率。SS/SL/LS/LL 沿用代码标签：第一字母按总长度（≤80K 为 S），第二字母按 miss（<512 为 S）；它们不是直接按读取量与计算毫秒数二分。

## 实测逐盘负载

{chr(10).join(overload)}

参考需求按每卡当前已接纳请求的 `B=每层读取量V/每层纯计算时间C` 计算，再按实际 Ring hash 落盘字节分到三盘。计算和 I/O stall 期间均保留该值；不把排队请求或下一请求未接纳的预取重复叠加。它不是瞬时提交 I/O 的速率，也不是读取指令必须达到的硬期限。

实际供给统计物理 SSD 读取服务，包含跨请求预取。曲线使用 10ms 内实际读量 / 10ms；窗口均值由完整服务区间积分得到。参考需求与实际供给的瞬时比值，不等于瞬时 NPU 利用率。全部策略的逐盘均值、超容量比例、繁忙比例见 `disk_metrics.csv`。

## 扩大窗口：[2,6) 秒

{table('long_2_6s', classes=True)}

## 完整请求集合：包含启动与排空

{table('full_population', classes=True)}

这批持续过载输入即使理想调度，受最忙盘读取工作量限制，**完整输入的 NPU 平均利用率也不超过 {full_capacity_bound:.4f}%**；计算方法见 [input_math.md 第 4 节](input_math.md#4-完整有限输入的npu利用率硬上限)。应把本表全程结果与这个上界比较，判断尚有多少提升空间。它不是 warm `[2,4)` 的上界，也不是 warm 预测值；阶段利用率可以高于它。

正式结果要求两个固定窗口内全部 32 张卡都持续有请求执行；全程利用率的分母则包含末尾部分卡先完成后的排空。因此 full 与 warm 不是同一个指标。固定窗口内不保证每卡已经执行全部 24 种画像；完整队列保证包含全部画像。

## 动态候选池补充结果

{table('warm_2_4s', ('mild', 'aggressive'), classes=True)}

## 公式与解释边界

```text
每卡参考需求 B_i = 该请求每层读取量 V_i / 每层计算时间 C_i
整批纯计算卡时间固定时，全程 U = 总计算卡时间 / (32 × 总历时)
总历时 >= max_盘(该盘整批读取量 / 该盘容量)
```

Ring hash 下应逐盘取最紧的读取时间下界，而不只看总容量 120 GiB/s。改变排队和 QoS 可以改变等待归属、读取重叠及排空时间，所以相同总读取量并不保证相同利用率或 SLO。改善总体达标率也不保证每个类别都受益，需同时看类别 SLO、类内利用率和实际达标吞吐。

本次只检验指定画像及三个种子；没有计入额外主机调度 CPU 成本或控制通信延迟。对其他流量的收益仍需独立验证。

## 图与可复查数据

当前图像展示 ASU Baseline、OD Baseline、原始 Once 和固定候选池 + Once 四种策略。

{figure_links or '完整种子尚未齐全，CDF 暂不生成。'}

CDF 横轴均为归一化耗时；每个种子先算精确经验 CDF，再对三个种子等权平均，x=1.5 与主表 SLO 对应。主图保留完整长尾；放大图只改显示范围，不改分母。

- [逐种子结果](comparison.csv)、[三种子汇总](macro_summary.csv)、[成对差值](paired_deltas.csv)
- [逐类别 SLO / 利用率](category_metrics.csv)、[逐画像计数与 SLO](profile_slo.csv)
- [每张卡的利用率](per_npu_metrics.csv)、[逐盘负载与实际供给](disk_metrics.csv)
- [归一化耗时逐请求样本](request_samples.csv)、[审计结果与来源 SHA](summary_checks.json)
- [输入与数学关系](input_math.md)、[逐画像读取量、计算时间和需求](input_profiles.csv)
- [独立复核脚本](audit_independent.py)
- [冻结输入审计](input_audit.json)、[运行器](run_trial.py)、[原始日志与度量重算](summarize_results.py)

`summary_checks.json` 核对请求数、块数、输入和源码哈希、完整完成、FIFO、卡内顺序、OD 独占路径及 CIR、全程读取字节守恒、窗口计量和逐请求 SLO。历史结果及图像没有被改写。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    checks = read(HERE / "summary_checks.json")
    if args.require_complete and checks["status"] != "complete":
        raise SystemExit("Formal grid is incomplete; run summarize_results.py when the full grid has finished.")
    source_hashes = dict(checks["source_sha256"])
    for path, digest in source_hashes.items():
        assert sha(ROOT / path) == digest, ("source changed since metric audit", path)
    for name in ("summary_checks.json", "comparison.csv", "macro_summary.csv", "request_samples.csv"):
        source_hashes[str((HERE / name).relative_to(ROOT))] = sha(HERE / name)
    rows, macros, samples = (csv_read(name) for name in ("comparison.csv", "macro_summary.csv", "request_samples.csv"))
    index = {(row["scenario"], row["policy"], row["window"]): row for row in macros}
    arrays = defaultdict(list)
    for row in samples:
        if row["in_warm_2_4s"] == "True":
            arrays[row["scenario"], row["policy"], int(row["seed"])].append(float(row["latency_ratio"]))
    arrays = {key: np.sort(value) for key, value in arrays.items()}
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    cdf_dir = out / "normalized_cdf"
    disk_dir = out / "per_ssu_bandwidth"
    cdf_dir.mkdir(exist_ok=True)
    disk_dir.mkdir(exist_ok=True)
    font_setup()
    images, cdf_points = [], []
    for scenario in SCENARIOS:
        if all((scenario, p, seed) in arrays for p in MAIN for seed in SEEDS):
            for zoom in (False, True):
                target = cdf_dir / f"{scenario}_random_ttft_ratio_cdf_four_strategies{'_zoom' if zoom else ''}.png"
                cdf_figure(arrays, index, scenario, MAIN, target, zoom=zoom)
                images.append((f"{SCENARIOS[scenario]}：{len(MAIN)} 策略归一化 CDF"
                               + ("（局部放大）" if zoom else "（完整长尾）"), str(target.relative_to(HERE))))
        for policy in MAIN:
            if all((scenario, policy, seed) in arrays for seed in SEEDS):
                knots = np.unique(np.r_[1.5, np.concatenate([arrays[scenario, policy, seed] for seed in SEEDS])])
                for x, y in zip(knots, macro_cdf(arrays, scenario, policy, knots)):
                    cdf_points.append(dict(scenario=scenario, policy=policy, latency_ratio=float(x), mean_cdf=float(y)))
            case = next((case for case in checks["cases"] if
                         (case["scenario"], case["policy"], case["seed"]) == (scenario, policy, 7)), None)
            if case:
                raw = read(HERE / "runs" / case["case"] / "result.json.gz")
                target = disk_dir / f"{scenario}_{policy}_seed7_per_ssu.png"
                disk_figure(raw, scenario, policy, target)
                images.append((f"{SCENARIOS[scenario]}：{NAMES[policy]} 逐盘需求与供给", str(target.relative_to(HERE))))
    csv_write("cdf_points.csv", cdf_points, ("scenario", "policy", "latency_ratio", "mean_cdf"))
    (HERE / "README.md").write_text(report_text(checks, rows, macros, images))
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items()), "report source changed"
    # Retire only known derived figures from the former six-policy view.
    removed = []
    for scenario in SCENARIOS:
        obsolete = [cdf_dir / f"{scenario}_random_ttft_ratio_cdf_all_six{suffix}.png"
                    for suffix in ("", "_zoom")]
        obsolete += [disk_dir / f"{scenario}_{policy}_seed7_per_ssu.png"
                     for policy in ("mild", "aggressive")]
        for path in obsolete:
            if path.exists():
                path.unlink()
                removed.append(str(path.relative_to(HERE)))
    audit = dict(status=checks["status"], no_new_simulation=True, old_results_unchanged=True,
                 source_files_unchanged=True, source_sha256=source_hashes, normalized_x="(completion-admission)/own_compute",
                 seeds=list(SEEDS), aggregation="equal mean of three seed ECDFs", full_tails_preserved=True,
                 zoom_preserves_full_denominator=True, cdf_1p5_matches_macro_slo=True,
                 plotted_policies=list(MAIN), removed_obsolete_png=removed,
                 generated_png=[path for _, path in images], visual_review="pending",
                 generated_png_sha256={path: sha(HERE / path) for _, path in images},
                 report_script_sha256=sha(Path(__file__)))
    (HERE / "render_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=checks["status"], report=str(HERE / "README.md"), png_count=len(images)), ensure_ascii=False))


if __name__ == "__main__":
    main()
