"""Build audited Chinese multi-SSU reports from saved native simulator results."""
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from run_multi_ssu_stall_experiments import summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/multi_ssu_stall_experiments"
CORE = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")
PRIMARY = (("baseline", "none", 0), ("layer_once", "none", 0),
           ("deadline", "none", 0), ("deadline", "fluid", 0))
NAMES = {("baseline", "none"): "Baseline", ("layer_once", "none"): "Once/layer",
         ("deadline", "none"): "A: deadline", ("deadline", "fluid"): "B: A+fluid选卡",
         ("deadline", "compute"): "A+compute选卡", ("demand", "none"): "V/C max-min",
         ("deadline_reserve", "none"): "deadline预留", ("least_slack", "none"): "最小松弛度",
         ("stall_interchange", "none"): "局部stall交换"}


def collect():
    rows, selected, grouped = [], {}, defaultdict(list)
    core_hash = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE}
    for path in sorted((OUT / "data").glob("npu*.json")):
        data = json.loads(path.read_text())
        meta, summary, window = data["metadata"], data["summary"], data["common_window"]
        recomputed = summarize(summary)
        for field, actual in recomputed.items():
            cached = window[field]
            if isinstance(actual, list):
                assert len(cached) == len(actual), (path, field)
                assert all(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-8)
                           for a, b in zip(cached, actual)), (path, field)
            elif isinstance(actual, bool):
                assert cached == actual, (path, field)
            else:
                assert math.isclose(cached, actual, rel_tol=1e-11, abs_tol=1e-8), (path, field)
        assert all(summary["invariants"].values()), path
        assert data["submit_seed"] == meta["seed"], path
        assert all(data["core_and_policy_sha256"][k] == v for k, v in core_hash.items()), path
        assert summary["request_count"] == meta["request_count"] == 704, path
        assert window["start_ms"] == 1000 and window["end_ms"] == 2000, path
        interval = data["control_min_interval_ms"]
        key = meta["num_ssu"], meta["regime"], meta["seed"], data["strategy"], data["assignment"], interval
        assert key not in selected, key
        selected[key] = data
        grouped[key[:3]].append(data)
        name = NAMES[data["strategy"], data["assignment"]]
        if interval:
            name += f" / {interval:g}ms限频"
        stat = data["control_statistics"]
        row = dict(file=str(path.relative_to(OUT)), num_ssu=meta["num_ssu"], regime=meta["regime"],
                   seed=meta["seed"], strategy=data["strategy"], assignment=data["assignment"],
                   interval_ms=interval, name=name, fingerprint=data["input_fingerprint"],
                   nominal_gib_s=meta["input_demand"]["total_gib_s"],
                   hottest_ssu_gib_s=max(meta["input_demand"]["per_ssu_gib_s"]),
                   **window, control_evaluations=summary["control_evaluations"],
                   cir_path_writes=summary["cir_path_writes"],
                   cir_write_transactions=summary["cir_write_transactions"],
                   pressure_reports=summary["pressure_reports"],
                   planner_mean_wall_us=stat.get("mean_decision_wall_us", 0),
                   planner_total_wall_s=stat.get("total_decision_wall_us", 0) / 1e6,
                   secondary_window=summarize(summary, 2000, 3000))
        rows.append(row)
    for key, group in grouped.items():
        assert len({d["input_fingerprint"] for d in group}) == 1, key
        assert len({d["metadata"]["source"]["source_sha256"] for d in group}) == 1, key
    return rows, selected, core_hash


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def figures(selected):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, s in zip(axes, (6, 7)):
        for offset, regime, label, color, hatch in [(-.2, "near", "Hotspot-safe input", "#0072B2", ""),
                                                    (.2, "replicated", "Unscaled 4-NPU mix", "#E69F00", "//")]:
            d = selected[(s, regime, 20260906, *PRIMARY[0])]["metadata"]["input_demand"]
            bars = ax.bar(np.arange(s) + offset, d["per_ssu_gib_s"], .38,
                          label=label, color=color, hatch=hatch, edgecolor="black", linewidth=.4)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.axhline(40, color="black", linestyle="--", linewidth=1.2, label="40 GiB/s per SSU")
        ax.set_xticks(range(s), [f"S{i}" for i in range(s)])
        ax.set_title(f"32 NPU / {s} SSU")
        ax.set_ylim(0, 66)
        ax.set_ylabel("Ideal input demand (GiB/s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig(OUT / "figures/per_ssu_input_demand.pdf")
    fig.savefig(OUT / "figures/per_ssu_input_demand.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.8), constrained_layout=True)
    for ax, s in zip(axes, (6, 7)):
        matrix = [[100 * u for u in selected[(s, "near", 20260906, *p)]["common_window"]["npu_utilizations"]]
                  for p in PRIMARY]
        im = ax.imshow(matrix, vmin=70, vmax=100, cmap="cividis", aspect="auto")
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                ax.text(x, y, f"{val:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if val < 86 else "black")
        ax.set_xticks(range(32), range(32), fontsize=7)
        ax.set_yticks(range(4), ["Baseline", "Once/layer", "A: deadline", "B: A+assignment"], fontsize=8)
        ax.set_title(f"32 NPU / {s} SSU; fixed [1000, 2000] ms; cell = compute %", fontsize=10)
        ax.set_xlabel("NPU ID (not sorted)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=.75, label="Compute utilization (%)")
    fig.savefig(OUT / "figures/per_npu_window_utilization.pdf")
    fig.savefig(OUT / "figures/per_npu_window_utilization.png", dpi=180)
    plt.close(fig)


def main():
    rows, selected, core_hash = collect()
    for s in (6, 7):
        for regime, seed in (("near", 20260906), ("near", 20260907), ("replicated", 20260906)):
            for policy in PRIMARY:
                assert (s, regime, seed, *policy) in selected, (s, regime, seed, policy)
    (OUT / "docs").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    figures(selected)
    def metric(s, regime, seed, policy):
        return selected[(s, regime, seed, *policy)]["common_window"]
    primary_rows, full_rows, secondary_rows = [], [], []
    for s in (6, 7):
        for seed in (20260906, 20260907):
            values = [metric(s, "near", seed, p) for p in PRIMARY]
            secondary = [summarize(selected[(s, "near", seed, *p)]["summary"], 2000, 3000) for p in PRIMARY]
            secondary_rows.append([s, str(seed)[-2:],
                                   *[f'{v["mean_npu_utilization"]*100:.3f}' for v in secondary],
                                   "是" if all(v["all_npus_active_whole_window"] for v in secondary) else "否"])
            primary_rows.append([s, str(seed)[-2:], *[f'{v["mean_npu_utilization"]*100:.3f}' for v in values],
                                 f'{100*(values[2]["mean_npu_utilization"]-values[1]["mean_npu_utilization"]):+.3f}'])
            for p, v in zip(PRIMARY, values):
                full_rows.append([s, str(seed)[-2:], NAMES[p[:2]], f'{v["makespan_ms"]:.1f}',
                                  f'{v["mean_arrival_to_completion_ms"]:.1f}',
                                  f'{v["p99_arrival_to_completion_ms"]:.1f}',
                                  f'{v["full_run_mean_npu_utilization"]*100:.2f}'])
    overload_rows = []
    for s in (6, 7):
        meta = selected[(s, "replicated", 20260906, *PRIMARY[0])]["metadata"]
        overload_rows.append([s, f'{meta["input_demand"]["total_gib_s"]:.2f}', s*40,
                              f'{max(meta["input_demand"]["per_ssu_gib_s"]):.2f}',
                              *[f'{metric(s,"replicated",20260906,p)["mean_npu_utilization"]*100:.3f}' for p in PRIMARY]])
    profile_rows = []
    for p in selected[(6, "near", 20260906, *PRIMARY[0])]["metadata"]["profiles"]:
        q7 = next(q["quota"] for q in selected[(7, "near", 20260906, *PRIMARY[0])]["metadata"]["profiles"]
                  if (q["seq_len_k"], q["nql"]) == (p["seq_len_k"], p["nql"]))
        profile_rows.append([p["seq_len_k"], p["nql"], p["kv_blocks"],
                             f'{p["kv_blocks"]*176/1024:.3f}', f'{p["compute_ms"]:.6f}', p["quota"], q7])
    ablation_rows, cost_rows, assignment_cost_rows, interchange_rows = [], [], [], []
    for row in rows:
        if row["regime"] == "near" and row["seed"] == 20260906:
            ablation_rows.append([row["num_ssu"], row["name"],
                                  f'{row["mean_npu_utilization"]*100:.3f}',
                                  f'{row["makespan_ms"]:.1f}', f'{row["p99_arrival_to_completion_ms"]:.1f}'])
            if row["assignment"] == "none" and row["strategy"] in ("baseline", "layer_once", "deadline"):
                cost_rows.append([row["num_ssu"], row["name"], row["control_evaluations"],
                                  row["cir_path_writes"], f'{row["planner_mean_wall_us"]:.1f}',
                                  f'{row["planner_total_wall_s"]:.2f}'])
            if row["assignment"] != "none":
                log = selected[(row["num_ssu"], row["regime"], row["seed"],
                                row["strategy"], row["assignment"], row["interval_ms"])]["assignment_log"]
                wall = sum(d["decision_wall_time_us"] for d in log)
                assignment_cost_rows.append([row["num_ssu"], row["name"], len(log),
                                             f'{wall/len(log):.1f}', f'{wall/1e6:.3f}'])
    for s in (6, 7):
        for seed in (20260906, 20260907):
            key = s, "near", seed, "stall_interchange", "none", 0
            if key in selected:
                base = metric(s, "near", seed, ("deadline", "none", 0))
                candidate = selected[key]["common_window"]
                interchange_rows.append([s, str(seed)[-2:], f'{base["mean_npu_utilization"]*100:.3f}',
                                         f'{candidate["mean_npu_utilization"]*100:.3f}',
                                         f'{100*(candidate["mean_npu_utilization"]-base["mean_npu_utilization"]):+.4f}',
                                         f'{candidate["makespan_ms"]-base["makespan_ms"]:+.1f}',
                                         f'{candidate["p99_arrival_to_completion_ms"]-base["p99_arrival_to_completion_ms"]:+.1f}'])
    validation = json.loads((OUT / "data/validation_multi_ssu_predictor.json").read_text())
    assert validation["core_files_unchanged"]
    assert validation["core_sha256"] == core_hash
    assert validation["predictor_sha256"] == hashlib.sha256((ROOT / "multi_ssu_stall_predictor.py").read_bytes()).hexdigest()
    assert validation["validator_sha256"] == hashlib.sha256((ROOT / "validate_multi_ssu_stall_predictor.py").read_bytes()).hexdigest()
    warm_rows = []
    for case in validation["warm_cases"]:
        comp = case["with_candidate"]
        warm_rows.append([case["num_ssu"], case["snapshot_ms"], case["computing_npu_count_at_snapshot"],
                          f'{comp["max_absolute_error_ms"]["io_ready_ms"]:.6f}',
                          f'{comp["max_absolute_error_ms"]["future_stall_ms"]:.6f}',
                          comp["non_layer0_stall_false_negative_count"],
                          comp["non_layer0_stall_false_positive_count"]])
    hotspot_rows = []
    hotspot_tradeoff = ""
    for s in (6, 7):
        key = s, "near_total", 20260906
        if all((*key, *p) in selected for p in PRIMARY):
            meta = selected[(*key, *PRIMARY[0])]["metadata"]["input_demand"]
            hotspot_rows.append([s, f'{meta["total_gib_s"]:.3f}',
                                 f'{max(meta["per_ssu_gib_s"]):.3f}',
                                 *[f'{metric(*key,p)["mean_npu_utilization"]*100:.3f}' for p in PRIMARY]])
            if s == 6:
                base, a, b = (metric(*key, PRIMARY[index]) for index in (0, 2, 3))
                hotspot_tradeoff = (
                    f'特别看6盘：B把主窗口利用率从{base["mean_npu_utilization"]*100:.3f}%提高到'
                    f'{b["mean_npu_utilization"]*100:.3f}%，但完整请求集的完工时间仅从'
                    f'{base["makespan_ms"]:.1f}ms变为{b["makespan_ms"]:.1f}ms，缩短'
                    f'{100*(1-b["makespan_ms"]/base["makespan_ms"]):.2f}%；仍慢于A的'
                    f'{a["makespan_ms"]:.1f}ms。不能把这一秒的约6.5个百分点增益当作同幅度的长期吞吐提升。')
    cold = validation["cold_cases"]
    cold_layers = sum(c["comparison"]["comparison_layer_count"] for c in cold)
    cold_error = max(max(c["comparison"]["max_absolute_error_ms"].values()) for c in cold)
    active = all(r["all_npus_active_whole_window"] for r in rows)
    near_rows = [r for r in rows if r["regime"] == "near"]
    best = max(near_rows, key=lambda r: r["mean_npu_utilization"])
    document = r'''---
title: "32 张 NPU、6/7 个 SSU：公式、策略与同输入实验"
subtitle: "固定中间一秒 + 完整请求集；收益、失败与信息边界一起报告"
date: "2026-09-06"
documentclass: article
fontsize: 10pt
geometry: a4paper,margin=18mm
CJKmainfont: "Noto Sans CJK JP"
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
toc: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \setlength{\emergencystretch}{3em}
---

# 先给结论：不能直接照搬四卡结果

本报告是实际运行原项目仿真器的结果，不是把四卡利用率按比例外推。预测数学已扩展到每盘 FIFO、每卡唯一接收链路和层间反馈；性能实验则分别比较原 Baseline、原 Once per layer、新 A 和新 B。

**必须分开三个问题：公式在限定模型中是否正确、信息不足时在线预测是否可靠、调度是否大幅提高利用率。前者成功不自动证明后两者。**

下表为热点不超盘容量的变化输入。所有数字都是 NPU 的计算占比（%），不是 SSD 忙碌率。06/07 是输入排列种子末两位；同一行严格共用外部请求、到达时间、SSD 放置与原生提交种子。

注意：为分别满足每盘容量，6盘和7盘的near配额不同；本表用于同一行的策略比较，**不能把两行差值全部归因于多加一盘**。第4节的原配比过载组才保留相同用户请求参数与到达规律，仅按各自SSU数量重新执行原生数据放置。

@PRIMARY@

A 固定原 NPU，只调整独占 Path 的 CIR；B 在同一 A 上增加到达时选卡。A−Once 是百分点，不是相对速度百分比。表中负值必须保留，不能只挑一组赢家。

本轮所有已保存性能实验共 @RUNS@ 组；固定窗口内每张卡始终有已接纳工作：@ACTIVE@。本轮最高窗口值为 @BEST@，但它不构成最优证明，也不等于所有种子、P99 和整批完成时间都最好。**近容量原始 data 变化输入没有自动重现旧构造中“baseline 很低、新策略几乎 100%”的大幅差距。**

# 到底给了系统什么输入？

## 单位与拓扑

- 32 张 NPU 独立处理不同请求，不是同一请求做 32 卡张量并行。
- 每 SSU 一个非抢占后端，40 GiB/s；每 NPU 一个 50 GiB/s FCFS 接收链路，所有 SSU 的返回共同排队。
- 一条 I/O 就是一块：128 token 对应的 GLM 每层 KV，176 KiB = 180224 byte。这里不用十进制 GB/s。
- 每请求 8 层，batch=1，启用跨请求 layer0 预取。只有前请求最后一层开始计算时已到达的 successor 才能被预取；不补造迟到预取。
- 数据载荷 SSD→HBM，无 DRAM 缓存/Prefill 命中收益。控制元数据不是把 KV 载荷放入 DRAM。

## 请求参数：不随意指定计算时间

直接解析项目 `data`，验证原文件 SHA256；使用其中计算时长和读取量，未插值、未更换计算模型。下表是每个原始 NPU 的 22 请求配额（种子06），随后每卡独立打乱顺序。长度 K 表示乘 1024 token；NQL 是本次查询 token 数。

@PROFILES@

每层读取块数 $K=({\rm total\ tokens}-NQL)/128$；每层数据量 $V=K\times180224/2^{30}$ GiB。$C$ 是 `data` 给出的单层计算毫秒数；$V/(C/1000)$ 才是无 stall 计算节奏下的名义 GiB/s，不是“读取所需时间”。物理服务时间须除以设备实际服务带宽。

**`data` 是参数表，不是线上概率分布或到达 trace。** 表中取值是真实项目输入，配额、随机顺序与到达规律是本实验构造，不能声称这是线上出现频率。

每卡最初恰有4请求在 $t=0$ 到达；其余按该卡理想计算节奏提前三个请求间隔到达。令请求0起编号、$C_j$ 为单层计算时长：

$$a_j=\max\left(0,\sum_{k<j}8C_k-\sum_{k<3}8C_k\right).$$

每组总计704请求。初始 backlog 是额外已到达工作，因此本文的“名义需求”不冒充整段 trace 的入口 offered bandwidth。所有策略保留相同到达，B 只能改变到达时的卡分配。

## 平均总量不够，还要看每一盘

对原固定分配，统计每卡完整输入总计算 $T_i$（ms）和逐盘读量 $V_{is}$（GiB）：

$$d_{is}=1000V_{is}/T_i,\quad d_s=\sum_i d_{is}.$$

这按计算时间加权，不是直接平均每个请求的 $V/C$。若所有卡要按原完整配比无 stall 持续计算，至少需要每个 $d_s\le40$、每卡 $\sum_s d_{is}\le50$。这不是任意有限一秒利用率的上界，也不是 B 重分配后的实时需求事实。

原生一致性哈希保留不动；6盘 near 总218.316 GiB/s、最热39.877；7盘 near 总249.203、最热39.159。因此 near 是“热点接近容量”，总容量占比约90.96%和89.00%，不是统一98%。原始配比直接扩到32卡则为316.183，6/7盘都承受不了。

![每根柱为完整输入的计算时间加权名义需求，不是某一秒实际盘吞吐。虚线为单盘物理容量；橙色直接放大旧配比，蓝色调整配额使每盘不过载。](../figures/per_ssu_input_demand.pdf){width=100%}

# 比较的是哪些策略？

Baseline 在每一 SSU 都只用 Path0 FIFO。Once/layer 保留项目原 `layer_once` 实现；它也会选择 Path，不能把 A 超过 Once 简单归因于“拥有多个 Path”。

A 使用每 NPU 一条独占 Path，在每 SSU 的256条中实际只占32条；ID 由原 `dedicated_path_id` 生成。在线取得已释放层的逐盘未完成量 $w_{is}$ 和数据 deadline $D_i$，按最早 deadline 排序；给同一层的各盘分量按同一个进度比例分配，使：

$$r_{is}=p_iw_{is},\qquad \sum_i r_{is}\le40,\qquad\sum_s r_{is}\le50.$$

一个层必须等所有盘的数据到齐，不能只优先读完最容易的一盘。已经在计算时，下层 deadline 是**当前层计算结束时刻**；跨请求 L0 也用前驱末层结束时刻，不用新请求自己的计算时间。已 stall 的层已经错过 deadline，需要尽快启动，但不同 overdue 层之间也存在选择冲突。

这是 CIR 优先方向，不是替换 SSD 的真实 EDF：原来的虚拟完成标签、非抢占命令、借带宽机制仍保留。项目不支持有限 PIR，本轮使用 PIR=无穷。**CIR 行和≤50 不意味着实际同时从多盘返回的峰值≤50**；接收链路仍然会排队。

B 在每次真实到达时，枚举32个卡位置，按已知剩余计算及每盘/每卡工作量的流体完工评分选卡。不读取未来请求，也不迁移已分配请求，不改变其数据所在 SSU。这是启发式，不是全局最优或精确 deadline 预测器。

另做三类消融：V/C 共同进度 max-min、按剩余时间预留再分配、只按剩余计算选卡；以及0.1 ms控制最小间隔。收益须从结果看，不能因公式看起来合理就预设胜出。

最后增加一个有针对性的 `stall_interchange` 候选：先按deadline排列，做两遍有限相邻交换。把每层独占服务下界作为串行代理服务时间 $T_i$，当前还能计算的时间为 $d_i=\max(0,D_i-now)$，前缀累计时间为 $q$。比较：

$$L_{ij}=(q+T_i-d_i)^++(q+T_i+T_j-d_j)^+.$$

仅当反序的 $L_{ji}$ 更小时交换。共享单瓶颈时，例如大任务需10ms且已到期，小任务需1ms且2ms后到期，总新增等待由19ms变11ms；多SSU并行下该串行代理不是精确预测。两遍共至多 $2(N-1)$ 次比较，不穷举，可能偏向短任务，必须检查P99。

# 同一秒与完整请求集必须一起看

主指标固定为所有策略同一绝对区间 $[1000,2000]$ ms：

$$U=\frac{\sum_{i=0}^{31}{\rm compute\_overlap}_i}{32\times1000\ {\rm ms}}.$$

每个 batch 接纳至完成的时间与窗口求交，检查每卡 active=1000 ms；然后将 active−compute 标为 I/O 等待，1000−active 标为无已接纳工作 idle。这避免把入口空闲伪装成 I/O stall，也没有移动窗口挑最大提升。

![格中整数为该卡计算利用率（%），横轴是真实卡号，没有逐策略重新排序；色阶固定70–100%。精确小数保存于 JSON。这里只画种子06，同一秒比较。](../figures/per_npu_window_utilization.pdf){width=100%}

下表覆盖完整704请求，不是窗口内标记的一部分。平均/P99延迟从外部到达算至完成，包含 admission 排队。整批利用率以32卡×完整 makespan 为分母，包含启动与尾部空闲。

@FULL@

**选卡可能降低多数请求或 P99 延迟，却使尾部某些 NPU 更晚结束。** 同时看 makespan 与一秒利用率，才能发现这种目标冲突；不能把中间一秒变好一律解释为吞吐变好。

## 不移动主窗口，但检查窗口敏感性

下面额外检查相同请求在 $[2000,3000]$ ms 的计算占比，不替换用户要求的主窗口。各策略经历的请求画像、相位不同，一秒读数不能直接当作长期稳态。

@SECONDARY@

## 原配比不减量：过载组不隐藏

@OVERLOAD@

这里低利用率包含物理容量不足，不能要求只靠 Path/CIR 把所有卡长期推到100%。调度可以改变受害者、相位及有限窗口中的计算组合，却不能让盘多读出字节。某一秒的平均计算占比也不能直接套“总容量/全输入平均需求”作为硬上界。

## 总量接近容量，但最热盘已经超载

@HOTSPOT@

该组的总量约为总容量的96%–97%，但最热盘仍超过40 GiB/s。它直接检验“只看所有SSU加总”会遗漏什么；不把它列为每盘都满足的 near 组。B改变卡位置不改变已有数据的落盘位置。

@HOTSPOT_TRADEOFF@

## 种子06的近容量消融及新候选复核

@ABLATION@

局部stall交换按两个种子单独对照A；下面时间差都是“新候选减A”，完工或P99的正值表示变差。它不是每个指标都优于A，也不能因单瓶颈代理目标下降就宣布真实多盘目标单调下降。

@INTERCHANGE@

这些是受控策略差异的证据，不是每条命令的唯一因果归因。Deadline、长计算的隐藏能力、数据分盘偏斜、CIR虚拟历史、接收链路、到达与请求切换相位都可能共同影响结果。

# 数学和 Python 到底能匹配到什么程度？

配套 `multi_ssu_math.pdf` 展开所有公式与输入含义；`multi_ssu_stall_predictor.py` 不调用原仿真器，独立实现标准库闭环递推。

- **严格筛查**：全部未提交的目标层，可用每盘前方工作下界和接收链路下界证明 must_stall；未超过 deadline 只返回 unknown，不误称必定满足。
- **已经部分提交的当前层**：`screen_pending_layer` 使用自己的 queued/unissued/link 数量给严格下界，不能把整条Path的全部其他I/O都当作自己前方工作。全部数据已到HBM时返回 already_ready，不凭此判断过去是否迟到。512次激活层和192次部分提交快照检查均无下界越界。
- **有完整冷启动输入的条件预测**：本轮 @COLD_CASES@ 个32卡6/7盘case、共 @COLD_LAYERS@ 层，对原生到齐、计算开始、未来stall的最大绝对误差 @COLD_ERROR@ ms。包含原 data 多种请求和单卡接收瓶颈回归。相同初始输入/原生公开种子允许重现同刻提交顺序，不是读取结果后重放。
- **运行中只有数量的预测**：不知道 SSU FIFO 的归属顺序、已经服务多少以及同刻排列时，必须重建一种假设。验证包含非空队列、正在计算的卡和新请求到达；存在毫秒级误差和漏报。保存的场景采样范围不是数学上下界。

本轮真实 warm 验证详情直接保存在 `data/validation_multi_ssu_predictor.json`；不能将冷启动的0误差泛化到线上计数遥测。新 A 使用实际可观测 deadline/剩余量做 CIR 决策，**没有声称完整 baseline 预测器能精确模拟256 Path CIR调度**。

warm验证不是只输入Path总数量：还使用按请求/层/SSU维护的SSD完成计数、客户端发射轮转状态和NPU链路剩余量。SSD完成计数需要SSU回报或实验instrumentation；此时仍不知道盘内顺序。A/B在线策略只用HBM完成回执的保守剩余量，两者遥测条件不同。

@WARM@

表中误差单位为ms，漏报/误报仅统计非L0的stall；“正在计算”列确认测试不只是初始空系统。该样本共2次漏报、1次误报，不是经过统计认证的错误概率。32卡raw-data八层完整预测约2.5秒，故完整递推用于离线分析，不能每层实时调用。

## 为什么层间自然错开不是保证？

“发起时间不同”不等于“每次读完都赶得上”。一层的到齐受最慢 SSU 和同卡接收队列约束。只要某个必经盘前方不可绕过的剩余工作 $H_s$ 足够长，即使卡改变层相位，也可能赶不上：

定义一条I/O的毫秒服务时间 $\tau_s=1000b/(2^{30}B_s)$、$\ell_i=1000b/(2^{30}L_i)$，其中 $b=180224$ byte，$B_s,L_i$ 用GiB/s。条件为：

$$H_s+K_{is}\tau_s+\ell_i>2C_i.$$

在一层前瞻、受害者正计算、还有至少两个后继转移且画像不变时，这是接下来两个转移中至少一次 stall 的充分条件；左侧减 $2C_i$ 是**两次累计 stall**下界，不是单层 stall 下界。它不成立不代表没 stall。有限8层也不能凭一次证书宣称无限循环必然持续。

另一个不可自然错开的因素是单卡接收容量。原 `data` 的192K/NQL128有1535块，$C\approx4.533469$ ms；六/七盘足够快地汇聚时，单卡仍至少需 $\tau+1535\ell\approx5.157089$ ms 才到齐。单活跃卡原生实测后续每层等待约0.623620 ms。这里没有其他卡可供错开；这是机制回归，不是32卡利用率案例。

# 控制成本和现实边界

@COST@

计数覆盖完整请求集；planner是远程Python callback单次/累计实测墙钟耗时，不是严格CPU时间，受并行负载影响，不是硬件固有延迟。它还**不包括**原生控制快照、pressure收集、CIR应用等开销。零值的 Once 不代表没有客户端路径规划成本，只表示没有使用新 CIR 控制器。

在线选卡另有成本，下表是种子06的704次到达决策，只计选卡hook，不能与CIR callback漏项互相替代：

@ASSIGNMENT_COST@

A 默认在实际层开始、层到齐、每盘分量到HBM完成时更新，后者最多每层每盘一次；32卡×6/7盘的事件数会明显超过 Once。SSU 分量完成回执可释放失效预算，但更频繁的全局协调有成本。0.1 ms限频只是限制更新时间，不是完整控制通信延迟仿真。

**仿真没有把控制器计算、跨卡通知或 CIR 寄存器写入延迟加到数据面时间。** 因此表中的提升不能直接兑现为硬件收益。实际部署还要量化遥测年龄、时钟同步、控制接口吞吐、计算时长预测误差和控制失败。SSD介质内部并行、GC、读放大与链路拓扑也不是本单后端模型完整覆盖的现实。

A/B在线策略不调用DRAM缓存、不取得隐藏FIFO；客户端未完成计数包含可能已经离盘但还没到HBM的量，保守重复计作盘剩余工作可能影响CIR。要提高在线预测确定性，需要SSU合作报告命令服务/完成边界或可用的服务保证，不能只增加Python公式。

# 文件和复现

主目录只保留最终PDF，JSON/CSV在data，图在figures，Markdown在docs。没有修改 `sim.py`、`continuous_batch_sim.py`、`continuous_prefill_client.py`、`policy_logic.py`；每组输出保存核心与策略源码哈希、原输入指纹、提交种子和原生 invariants。

实验期间控制器经历三版：初始版、修正新事件的最小更新间隔、增加局部stall交换；旧零间隔模式未改分配算法。不是所有输出都由当前源码版本生成。各历史版本按SHA归档于 `data/source_versions`，最终可运行源文件与原始data打包在 `data/final_source_bundle.tar.gz`。审计列出每份输出对应版本并验证归档内容，不用当前文件冒充历史版本。统计窗口也从逐层原始事件重新计算，而不是只信缓存汇总数。

```bash
python -B -m pytest -q
python -B validate_multi_ssu_stall_predictor.py
python -B sweep_multi_ssu_experiments.py --phase primary --workers 6
python -B sweep_multi_ssu_experiments.py --phase holdout --workers 4
python -B sweep_multi_ssu_experiments.py --phase ablation --workers 4
python -B sweep_multi_ssu_experiments.py --phase hotspot --workers 4
python -B sweep_multi_ssu_experiments.py --phase interchange --workers 4
python -B build_multi_ssu_report.py
```

若只跑一组：

```bash
python -B run_multi_ssu_stall_experiments.py --num-ssu 7 \
  --regime near --seed 20260906 --strategy deadline
```

增加 `--assignment fluid` 即 B。公式文件、CIR控制器、在线选卡器分别为 `multi_ssu_stall_predictor.py`、`multi_ssu_qos_controller.py`、`multi_ssu_npu_assignment.py`。这是可复现实验适配器，不是生产驱动；下一步应根据本轮收益与成本决定是否值得接入真实控制链路，不能先承诺极大提升再挑结果。
'''
    replacements = {
        "@PRIMARY@": table(["SSU", "种子", "Baseline", "Once", "A", "B", "A−Once"], primary_rows),
        "@FULL@": table(["SSU", "种子", "策略", "完工ms", "均延迟ms", "P99 ms", "整批U%"], full_rows),
        "@SECONDARY@": table(["SSU", "种子", "Baseline%", "Once%", "A%", "B%", "全卡active"], secondary_rows),
        "@HOTSPOT@": table(["SSU", "总需求", "最热盘", "Baseline%", "Once%", "A%", "B%"], hotspot_rows),
        "@HOTSPOT_TRADEOFF@": hotspot_tradeoff,
        "@WARM@": table(["SSU", "快照ms", "正在计算", "到齐误差", "stall误差", "漏报", "误报"], warm_rows),
        "@PROFILES@": table(["长度K", "NQL", "块/层", "MiB/层", "C/ms", "6盘配额", "7盘配额"], profile_rows),
        "@OVERLOAD@": table(["SSU", "需求", "总容量", "最热盘", "Baseline%", "Once%", "A%", "B%"], overload_rows),
        "@ABLATION@": table(["SSU", "策略", "窗口U%", "完工ms", "P99 ms"], ablation_rows),
        "@INTERCHANGE@": table(["SSU", "种子", "A%", "交换%", "差/百分点", "完工差ms", "P99差ms"], interchange_rows),
        "@COST@": table(["SSU", "策略", "重算次数", "Path写次数", "均墙钟/us", "总墙钟/s"], cost_rows),
        "@ASSIGNMENT_COST@": table(["SSU", "策略", "决策数", "均墙钟/us", "总墙钟/s"], assignment_cost_rows),
        "@RUNS@": str(len(rows)), "@ACTIVE@": "全部通过" if active else "存在不通过的组，见CSV，不计作纯stall案例",
        "@BEST@": f'{best["mean_npu_utilization"]*100:.3f}%（{best["num_ssu"]}盘，{best["seed"]}，{best["name"]}）',
        "@COLD_CASES@": str(len(cold)), "@COLD_LAYERS@": str(cold_layers), "@COLD_ERROR@": f"{cold_error:.9g}",
    }
    for key, value in replacements.items():
        document = document.replace(key, value)
    source = OUT / "docs/multi_ssu_experiment_report.md"
    source.write_text(document, encoding="utf-8")
    with (OUT / "data/experiment_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"performance_run_count": len(rows), "window_ms": [1000, 2000],
             "all_windows_all_npus_active": active, "same_input_seed_and_core_verified": True,
             "core_sha256": core_hash, "cold_validation_cases": len(cold),
             "cold_validation_layers": cold_layers, "records": rows}
    versions = {}
    for data in selected.values():
        for name, sha in data["core_and_policy_sha256"].items():
            item = versions.setdefault(name, {}).setdefault(sha, {"run_count": 0})
            item["run_count"] += 1
            item["matches_current"] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
            archive = OUT / "data/source_versions" / sha / name
            item["archived_source"] = str(archive.relative_to(OUT)) if archive.exists() else None
            if archive.exists():
                assert hashlib.sha256(archive.read_bytes()).hexdigest() == sha, archive
            assert item["matches_current"] or archive.exists(), (name, sha)
    audit["source_versions"] = versions
    audit["window_metrics_recomputed_from_layers"] = True
    audit["predictor_and_validator_versions_verified"] = True
    (OUT / "data/report_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    for stem in ("multi_ssu_experiment_report", "multi_ssu_math"):
        subprocess.run(["pandoc", str(OUT / "docs" / (stem + ".md")), "--standalone", "--number-sections",
                        "--pdf-engine=xelatex", "--resource-path", str(OUT / "docs"),
                        "--output", str(OUT / (stem + ".pdf"))], check=True)
    print(replacements["@PRIMARY@"])


if __name__ == "__main__":
    main()
