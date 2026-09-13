#!/usr/bin/env python3
"""Derive a fixed, raw-data candidate matrix; never import or run the simulator."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NPU = 32
SSU = 3
DISK_GIB_S = 40.0
CAPACITY_GIB_S = SSU * DISK_GIB_S
LAYERS = 8
IO_GIB = 176 / 1024**2
PURE_HORIZON_MS = 22000.0

# Entries and seeds are fixed independently of the results of the new runs.
# The counts describe the entire shuffled queue, not repeating ordered blocks.
CANDIDATES = [
    dict(id="ref128_32", priority=0, family="reference", blocker="B",
         mix=[("A", (128, 256), 1), ("B", (32, 4096), 2)],
         why="保留当前参考；大读取 A 的计算很短，小读取 B 的 28.59 ms 计算通常能隐藏等待。",
         risk="用于对照，不预期它提供明显更低的 Random 利用率。"),
    dict(id="main512", priority=1, family="primary", blocker="L",
         mix=[("L", (192, 4096), 1), ("S", (32, 512), 9)],
         why="短类计算约 3.70 ms，纯计算贡献约 19.7%；兼顾较紧截止时间与整机影响。",
         risk="仍需多个读取在 FIFO 中积累；一段孤立长读取通常不够。"),
    dict(id="main1024", priority=2, family="primary", blocker="L",
         mix=[("L", (192, 4096), 1), ("S", (32, 1024), 18)],
         why="短类纯计算贡献约 49%；若短类经常等待，对整机平均的影响更大。",
         risk="7.26 ms 计算更能隐藏读取，长类卡数和长读取频率也降低，未必更差。"),
    dict(id="load095", priority=4, family="load_control", blocker="L",
         mix=[("L", (192, 4096), 1), ("S", (32, 512), 8)],
         why="与 main512 同画像，将理想平均负载降到约 95%，检查下降是否依赖略微过载。",
         risk="平均低于容量仍不代表逐盘逐时欠载。"),
    dict(id="load104", priority=5, family="load_control", blocker="L",
         mix=[("L", (192, 4096), 1), ("S", (32, 512), 10)],
         why="与 main512 同画像，将理想平均负载升到约 104%，观察小范围负载响应。",
         risk="必须区分容量不足与额外等待，不能把全部损失归因于 FIFO。"),
    dict(id="tight256", priority=3, family="deadline_control", blocker="L",
         mix=[("L", (192, 4096), 1), ("S", (32, 256), 7)],
         why="截止时间约 2 ms；检查短类更容易被卡住是否会被其较小计算占比掩盖。",
         risk="短类纯计算贡献只有约 9.3%；短类很差未必使整机很差。"),
    dict(id="smaller_blocker", priority=6, family="burst_control", blocker="L",
         mix=[("L", (128, 4096), 5), ("S", (32, 512), 32)],
         why="与 main512 接近相同负载和短计算贡献，但长类单层读取由 258.5 降至 170.5 MiB。",
         risk="长类计算时间也随原始画像变化；这是成组画像对照，不是只改变读取量。"),
    dict(id="three_profile", priority=7, family="heterogeneity", blocker="L",
         mix=[("L", (192, 4096), 1), ("S1", (32, 512), 4), ("S2", (32, 1024), 10)],
         why="混合两种短类，检查现象是否依赖所有短请求采用同一个计算时间。",
         risk="S1/S2 的等待必须分开统计，不能套一个共同截止时间。"),
]


def category(key: tuple[int, int]) -> str:
    """The existing sim.py thresholds, recorded without importing the simulator."""
    seq_k, miss = key
    return ("S" if seq_k <= 80 else "L") + ("L" if miss >= 512 else "S")


def cv(values: list[float], counts: list[int]) -> float:
    count = sum(counts)
    mean = sum(v * n for v, n in zip(values, counts)) / count
    variance = sum(n * (v - mean) ** 2 for v, n in zip(values, counts)) / count
    return math.sqrt(variance) / mean


def derive(spec: dict, data: dict) -> dict:
    profiles = []
    for label, key, count in spec["mix"]:
        raw_B, raw_C_us, source_ttft_ms, V = data[key]
        C = raw_C_us / 1e6
        assert key[0] >= 32 and V > 0 and C > 0
        assert math.isclose(V / C, raw_B, rel_tol=1e-12)
        nblocks = round(V / IO_GIB)
        assert math.isclose(nblocks * IO_GIB, V, abs_tol=1e-12)
        profiles.append(dict(
            label=label, data_key=list(key), ratio_count=count,
            category=category(key), total_input_tokens=key[0] * 1024,
            miss_tokens=key[1], hit_tokens=key[0] * 1024 - key[1],
            V_GiB=V, V_MiB=V * 1024, C_ms=C * 1000,
            B_GiB_s=V / C, source_ttft_ms_not_used_as_threshold=source_ttft_ms,
            blocks_per_layer=nblocks, raw_profile=True,
        ))
    work_C_s = sum(p["ratio_count"] * p["C_ms"] / 1000 for p in profiles)
    work_V = sum(p["ratio_count"] * p["V_GiB"] for p in profiles)
    count = sum(p["ratio_count"] for p in profiles)
    rho = NPU * work_V / (CAPACITY_GIB_S * work_C_s)
    assert 0.95 <= rho <= 1.05
    blocker = next(p for p in profiles if p["label"] == spec["blocker"])
    victims = [p for p in profiles if p["label"] != spec["blocker"]]
    for p in profiles:
        p["request_count_share_percent"] = 100 * p["ratio_count"] / count
        p["pure_compute_share_percent"] = 100 * p["ratio_count"] * p["C_ms"] / (1000 * work_C_s)
        p["ideal_compute_only_mean_current_cards"] = NPU * p["pure_compute_share_percent"] / 100
        p["balanced_3disk_layer_service_ms"] = p["V_GiB"] / CAPACITY_GIB_S * 1000
    victim_count = sum(p["ratio_count"] for p in victims)
    f = sum(p["pure_compute_share_percent"] for p in victims) / 100
    victim_conditions = []
    for p in victims:
        threshold = (CAPACITY_GIB_S * p["C_ms"] / 1000 - p["V_GiB"]) / blocker["V_GiB"]
        victim_conditions.append(dict(
            label=p["label"],
            queued_blocker_layer_equivalent_threshold=threshold,
            strict_condition="queued residual blocker layer equivalents > threshold; balanced disks, victim bytes queued after that prefix",
            condition_is_not_current_card_count=True,
        ))
    # Each queue contains an integer number of ratio units, then is shuffled once
    # in full independently on each NPU. No ratio unit remains an execution block.
    ratio_pure_ms = LAYERS * work_C_s * 1000
    copies = math.ceil(PURE_HORIZON_MS / ratio_pure_ms)
    per_npu_counts = {p["label"]: copies * p["ratio_count"] for p in profiles}
    return dict(
        id=spec["id"], priority=spec["priority"], family=spec["family"],
        blocker_label=spec["blocker"], profiles=profiles,
        ratio_labels=":".join(p["label"] for p in profiles),
        ratio_counts=":".join(str(p["ratio_count"]) for p in profiles),
        rho_ideal=rho,
        ideal_time_weighted_mean_B_per_npu_GiB_s=work_V / work_C_s,
        victim_pure_compute_fraction=f,
        request_weighted_CV_layer_V=cv([p["V_GiB"] for p in profiles], [p["ratio_count"] for p in profiles]),
        request_weighted_CV_layer_C=cv([p["C_ms"] for p in profiles], [p["ratio_count"] for p in profiles]),
        blocker_balanced_3disk_layer_service_ms=blocker["balanced_3disk_layer_service_ms"],
        victim_queue_conditions=victim_conditions,
        common_victim_wait_ms_for_target_U={
            str(target): (1 / target - 1) * work_C_s * 1000 / victim_count
            for target in (0.9, 0.85, 0.8)
        },
        wait_model_assumptions=[
            "blocker has no exposed stalls; each victim layer has the same mean extra wait in milliseconds",
            "mean wait includes zero-wait layers; for unequal victim means use sum(count_j * wait_j)",
            "fixed completion proportions and long-run complete cycles; startup and request-boundary effects omitted",
            "required waiting to obtain target U, not predicted waiting and not an exact warm-window identity",
        ],
        full_finite_work_conservation_U_upper_bound=min(1.0, 1 / rho),
        full_finite_upper_bound_not_warm_bound=True,
        suggested_minimum_22s_population=dict(
            minimum_pure_compute_horizon_ms=PURE_HORIZON_MS,
            ratio_unit_copies=copies, per_npu_counts=per_npu_counts,
            requests_per_npu=sum(per_npu_counts.values()),
            total_requests=NPU * sum(per_npu_counts.values()),
            pure_compute_ms_per_npu=copies * ratio_pure_ms,
            total_8layer_read_GiB=NPU * copies * LAYERS * work_V,
        ),
        rationale=spec["why"], failure_or_confounding_risk=spec["risk"],
    )


def make_markdown(doc: dict) -> str:
    rows = [
        "# Random、接近容量：预先固定的候选矩阵", "",
        "本文件只计算候选参数，没有运行仿真，也没有预测 Baseline 必然很差。所有 V、C 直接读取项目 data；总输入长度最小 32K。", "",
        "配置：32 NPU、3 SSU，每盘 40 GiB/s，NPU 链路 50 GiB/s，8 层、batch=1，保留跨请求首层预取。只测 Random；每卡固定数量后独立打乱整条队列，不把下面的数量比例排成重复执行块。", "",
        "| 候选 | 原始画像（总长度K / miss token）及数量比 | 理想 rho | 受观察类纯计算贡献 | 假设仅受观察类等待，使 U≈80% 所需平均等待/层 |", 
        "|---|---|---:|---:|---:|",
    ]
    for c in doc["candidates"]:
        labels = "；".join(f'{p["label"]}={p["data_key"][0]}/{p["data_key"][1]}' for p in c["profiles"])
        rows.append(f'| {c["id"]} | {labels}，{c["ratio_labels"]}={c["ratio_counts"]} | {c["rho_ideal"]:.6f} | {100*c["victim_pure_compute_fraction"]:.3f}% | {c["common_victim_wait_ms_for_target_U"]["0.8"]:.3f} ms |')
    rows += ["", "参考组的受观察类是 A；其他组是 S 或 S1+S2。最后一列是需要达到的等待量，不是我们预测会发生的等待。三画像组假设 S1/S2 的平均额外等待毫秒数相同；实际统计仍须分开。", "", "## 为什么选这八组", ""]
    for c in sorted(doc["candidates"], key=lambda x: x["priority"]):
        rows += [f'- **{c["id"]}**：{c["rationale"]} 限制：{c["failure_or_confounding_risk"]}']
    unique = {}
    for c in doc["candidates"]:
        for p in c["profiles"]:
            unique[tuple(p["data_key"])] = p
    rows += ["", "## 原始画像", "", "| 总输入K / miss | 每层读取 MiB | 每层计算 ms | Bi=V/C，GiB/s | 当前路由类别 |", "|---|---:|---:|---:|---|"]
    for key, p in sorted(unique.items()):
        rows.append(f'| {key[0]} / {key[1]} | {p["V_MiB"]:.6f} | {p["C_ms"]:.6f} | {p["B_GiB_s"]:.6f} | {p["category"]} |')
    rows += [
        "", "路由类别按当前 sim.py 的阈值标记：总长度不超过 80K 为首字母 S，miss 至少 512 为第二字母 L。它不是“大读取/小读取、长计算/短计算”的直接标签。Baseline 全部走 Path0；后续若对照 Once，这些类别会影响合法 Path 池。", "",
        "## 简单公式和不能跳过的限制", "",
        "令 n_j 为某画像的请求数量比例，V_j 为单层读取 GiB，C_j 为单层计算秒。", "",
        "```text", "rho_ideal = 32 * sum(n_j * V_j) / [120 * sum(n_j * C_j)]",
        "纯计算贡献 f_j = n_j * C_j / sum(n_j * C_j)",
        "理想无等待情况下，该画像平均同时在场卡数 ≈ 32 * f_j", "```", "",
        "rho_ideal 只描述按纯计算时间加权的平均需求。它不是实际到达负载，不保证逐盘逐时欠载，也不能用请求数量占比替代计算时间占比。实际 IO stall 会改变各类驻留时间、同时在场卡数以及预取相位。", "",
        "192K/4096 每层读取 258.5 MiB。均匀分到三盘后的总服务时间约 2.104 ms，但这不等于每个短请求必然等 2.104 ms：单个长读取的剩余部分可能更短，同时开始提交的请求还会逐块交错。", "",
        "```text", "简化 FIFO 读取完成时间 ≈ (Q_ahead + V_short) / 120",
        "发生暴露等待的条件：上式 > C_short", "```", "",
        "Q_ahead 是实际排在短请求前面的剩余字节，不是长请求的完整读取量乘当前长卡数。模型按 176 KiB 的块服务，并非整层原子占盘；严谨验证必须逐盘检查 FIFO 前缀和短层 IO-ready 是否错过计算结束时间。三盘均匀公式只是候选估算。JSON 中读取量的 CV 统计每层总读取量的差异，不代表单个 IO 块大小变化。", "",
        "```text", "如果长类没有 stall，而短类每层平均额外等 w 秒：",
        "U ≈ sum(n_j*C_j) / [sum(n_j*C_j) + n_short*w]",
        "单一短画像时：U ≈ 1 / (1 + f_short*w/C_short)", "```", "",
        "w 要对所有短层取平均，包括不等待的层。此式假设长期完成比例稳定且忽略启动、请求边界；不是 warm[2,4) 的精确公式。它只回答“需要多少等待才能这么差”，不证明实际能产生这些等待。", "",
        "## 探索、复验与输入约束", "",
        "- 以 study_plan.json 为准：初筛仅用种子 7，独立确认种子为 19、43、67、101。每个已启动运行都报告，不在结果出来后只展示最差种子。",
        "- 优先启动 main512、main1024、tight256，再做负载与读取突发对照。可在探索完成后选候选做复验；选候选的依据与停止原因要记录，复验种子不能参与重新调画像或配比。",
        "- 主指标同时报告 warm[2,4) 和 long[2,20) 实际计算时间 / (32×窗口长度)。补充每类条件利用率、IO stall、完整内部周期 b/B、灰区与逐盘实际读取。",
        "- 每卡先固定全队列画像数量，再独立完整随机打乱；不在卡之间复制同一队列，不强制相位同步，不按统计结果重排。",
        "- 每次运行检查 32 卡全窗有已接纳请求，并检查每卡 warm 内长计算、短计算两组均有真实计算：参考组为 B/A，其余组为 L/S，三画像组将 S1+S2 合为短组并另报各自覆盖。完整随机本身不能保证每卡两组都出现；失败时明确标记，不将其悄悄改成分块随机或挑另一个坏种子替代。",
        "- 随机样本不满足混合约束时仍保留结果和失败原因；它不能作为满足该约束的主要证据。候选若持续不满足，须作为新设计重新冻结输入规则。",
        "- 为统计到 20 秒，本轮所有新输入按照 experiment.py 的规则选择最小整数倍，使每卡纯计算时间至少 22 秒。这样提供任务余量，但不保证 warm 中每类都出现。下表对应实际新输入规则。",
        "- 旧参考组 A=40、B=80 的每卡纯计算时间为 20227.103 ms，足以覆盖旧 20 秒窗口，但不同于本轮至少 22 秒的新参考 A=44、B=88。旧结果可作为注明来源的背景参考，不能当作与新参考完全相同的输入；完整队列长度变化也会改变同 seed 的随机排列。",
        "- 接近 1 的平均 rho 不满足历史的严格逐盘逐时欠载要求；如继续要求该条件，需要另行设计并发限制并说明随机定义变化。", "",
        "| 候选 | 建议每卡请求数 | 每卡纯计算 ms | 全部请求数 |", "|---|---|---:|---:|",
    ]
    for c in doc["candidates"]:
        pop = c["suggested_minimum_22s_population"]
        labels = ", ".join(f"{k}={v}" for k, v in pop["per_npu_counts"].items())
        rows.append(f'| {c["id"]} | {labels} | {pop["pure_compute_ms_per_npu"]:.3f} | {pop["total_requests"]} |')
    rows += ["", "这些是研究候选，不保证 Baseline 随机会差，也不保证 Once 会改善。评价策略时要保留完整输入、记录全部种子，并把平均容量压力与额外排队损失分开。", "", "来源：项目 [data](../../data)；[数学脚本](candidate_math.py)；[完整参数 JSON](candidate_math.json)。", f'原始 data SHA-256：`{doc["source"]["data_sha256"]}`。', ""]
    return "\n".join(rows)


def main() -> None:
    source = ROOT / "data"
    plan_path = HERE / "study_plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["minimum_pure_compute_horizon_ms"] == PURE_HORIZON_MS
    assert plan["pilot_seed"] == 7 and plan["confirmation_seeds"] == [19, 43, 67, 101]
    assert plan["num_npu"] == NPU and plan["num_ssu"] == SSU and plan["n_layers"] == LAYERS
    data = ast.literal_eval(source.read_text())
    cases = [derive(spec, data) for spec in CANDIDATES]
    assert len(cases) <= 8 and len({c["id"] for c in cases}) == len(cases)
    doc = dict(
        schema_version=1, status="mathematical_candidates_not_simulation_results",
        source=dict(data_path=str(source.relative_to(ROOT)), data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    study_plan_path=str(plan_path.relative_to(ROOT)),
                    study_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        configuration=dict(num_npu=NPU, num_ssu=SSU, per_ssu_bandwidth_GiB_s=DISK_GIB_S,
                           npu_link_bandwidth_GiB_s=50.0, n_layers=LAYERS, batch_size=1,
                           cross_request_layer0_prefetch=True, order="random",
                           assignment="fixed", primary_strategy="baseline", all_arrivals_ms=0,
                           warm_ms=[2000, 4000], long_ms=[2000, 20000],
                           minimum_pure_compute_horizon_ms=PURE_HORIZON_MS,
                           ideal_rho_range=[0.95, 1.05], minimum_total_input_tokens=32*1024,
                           full_queue_independent_shuffle_per_npu=True,
                           strict_per_disk_per_instant_underload_not_guaranteed=True),
        seed_plan=dict(pilot=[plan["pilot_seed"]], confirmation=plan["confirmation_seeds"],
                       report_every_started_run=True, choose_only_bad_seeds=False,
                       selection="Candidate selection may use pilot seed 7 only; record the rule and all pilot outcomes before confirmation seeds 19/43/67/101.",
                       mixed_window_gate="Each NPU must compute blocker and any victim profile in warm; reference B/A, others L/(S or S1+S2). Report individual subtype coverage separately; retain and flag failures rather than silently resampling."),
        historical_reference=dict(per_npu_counts={"A": 40, "B": 80},
                                  pure_compute_ms_per_npu=8*(40*data[(128, 256)][1]+80*data[(32, 4096)][1])/1000,
                                  covers_20s=True, meets_new_22s_population_rule=False,
                                  identical_to_new_reference_input=False,
                                  use="Context only unless clearly labeled; new 44/88 full-queue shuffle is a different finite input."),
        candidates=cases,
    )
    (HERE / "candidate_math.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
    (HERE / "candidate_math.md").write_text(make_markdown(doc))
    print(json.dumps({"candidates": len(cases), "simulations_run": 0,
                      "outputs": [str(HERE/"candidate_math.json"), str(HERE/"candidate_math.md")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
