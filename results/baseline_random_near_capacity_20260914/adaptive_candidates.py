#!/usr/bin/env python3
"""Raw-only adaptive candidate ranking; no simulator imports or execution."""

import ast
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
N, W, LAYERS = 32, 120.0, 8
COUNT_LIMIT, CLOSE_RHO = 64, 0.005
FAMILIES = ("larger_read_shorter_compute", "extreme_scales", "similar_compute_different_read")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative_rank(row):
    error = abs(row[2] - 1)
    if error <= CLOSE_RHO:
        return (0, row[0] + row[1], error, row[0], row[1])
    return (1, error, row[0] + row[1], row[0], row[1])


def enumerate_pairs(profiles, excluded_pairs):
    options = [(a, b) for a in range(1, COUNT_LIMIT + 1)
               for b in range(1, COUNT_LIMIT + 1) if math.gcd(a, b) == 1]
    pairs, continuously_feasible, configurations = [], 0, 0
    for first, second in itertools.combinations(profiles, 2):
        H, L = sorted((first, second), key=lambda p: p["B_GiB_s"], reverse=True)
        if min(H["B_GiB_s"], L["B_GiB_s"]) < 1.05 * W / N and max(H["B_GiB_s"], L["B_GiB_s"]) > .95 * W / N:
            continuously_feasible += 1
        legal = []
        for nh, nl in options:
            total_C = nh * H["C_s"] + nl * L["C_s"]
            rho = N * (nh * H["V_GiB"] + nl * L["V_GiB"]) / (W * total_C)
            if .95 <= rho <= 1.05:
                legal.append((nh, nl, rho))
        configurations += len(legal)
        if not legal:
            continue
        nh, nl, rho = min(legal, key=representative_rank)
        total_C = nh * H["C_s"] + nl * L["C_s"]
        fH = nh * H["C_s"] / total_C
        cr = max(H["C_s"], L["C_s"]) / min(H["C_s"], L["C_s"])
        vr = max(H["V_GiB"], L["V_GiB"]) / min(H["V_GiB"], L["V_GiB"])
        scores = {}
        if H["V_GiB"] > L["V_GiB"] and H["C_s"] < L["C_s"] and .15 <= fH <= .65 and cr >= 1.5:
            scores[FAMILIES[0]] = fH * math.log2(vr)
        if cr >= 8 and vr >= 2:
            scores[FAMILIES[1]] = math.log2(cr) * math.log2(vr)
        if cr <= 1.35 and vr >= 2:
            scores[FAMILIES[2]] = math.log2(vr) / cr
        pkey = tuple(sorted((tuple(H["data_key"]), tuple(L["data_key"]))))
        copies = math.ceil(22000 / (LAYERS * total_C * 1000))
        target_threshold = (W * H["C_s"] - H["V_GiB"]) / L["V_GiB"]
        shorter_is_H = H["C_s"] < L["C_s"]
        pairs.append(dict(
            pair_id=f'H{H["data_key"][0]}_{H["data_key"][1]}_L{L["data_key"][0]}_{L["data_key"][1]}',
            H=H, L=L, ratio_counts_H_L=[nh, nl], rho_ideal=rho,
            ideal_f_H=fH, ideal_f_short_compute=fH if shorter_is_H else 1-fH,
            shorter_compute_label="H" if shorter_is_H else "L",
            ideal_compute_only_mean_H_cards=N*fH,
            C_scale_ratio=cr, V_scale_ratio=vr,
            legal_coprime_count_combinations=len(legal),
            family_scores=scores, already_in_frozen_two_profile_matrix=pkey in excluded_pairs,
            target_H_queue_threshold_in_L_full_layer_equivalents=target_threshold,
            target_H_mean_extra_wait_ms_for_U80=.25*total_C/nh*1000,
            target_H_mean_extra_wait_ms_for_U90=(1/.9-1)*total_C/nh*1000,
            mean_wait_is_over_all_H_layers_including_zero=True,
            target_model_assumptions="L has no exposed stalls; fixed completed proportions; long-run complete cycles; startup and request-boundary effects omitted. Required waiting, not predicted waiting.",
            both_classes_full_layer_burst_service_ms=N*(fH*H["V_GiB"]+(1-fH)*L["V_GiB"])/W*1000,
            full_layer_burst_estimate_not_observed_backlog=True,
            minimum_22s_population=dict(H=nh*copies, L=nl*copies, repeats=copies,
                                       pure_compute_ms_per_npu=copies*LAYERS*total_C*1000),
        ))
    return pairs, dict(raw_eligible_profiles=len(profiles), all_unordered_profile_pairs=len(profiles)*(len(profiles)-1)//2,
                       continuously_feasible_positive_count_ratio_pairs=continuously_feasible,
                       bounded_integer_feasible_pairs=len(pairs), coprime_count_options_per_pair=len(options),
                       bounded_legal_pair_and_ratio_configurations=configurations)


def select(pairs):
    chosen, rankings, globally_used = [], {}, set()
    for family_index, family in enumerate(FAMILIES, 1):
        ranked = sorted((p for p in pairs if family in p["family_scores"] and not p["already_in_frozen_two_profile_matrix"]),
                        key=lambda p: (-p["family_scores"][family], abs(p["rho_ideal"]-1), p["pair_id"]))
        used_H, used_L, family_chosen = set(), set(), []
        rankings[family] = []
        for rank, p in enumerate(ranked, 1):
            Hkey, Lkey = tuple(p["H"]["data_key"]), tuple(p["L"]["data_key"])
            take = len(family_chosen) < 4 and p["pair_id"] not in globally_used and Hkey not in used_H and Lkey not in used_L
            rankings[family].append(dict(rank=rank, pair_id=p["pair_id"], score=p["family_scores"][family], selected=take))
            if take:
                item = dict(p, candidate_id=f'adapt_f{family_index}_{len(family_chosen)+1:02d}',
                            selected_family=family, family_rank=rank,
                            profiles_cli=":".join(map(str, Hkey))+","+":".join(map(str, Lkey)),
                            counts_cli=",".join(map(str, p["ratio_counts_H_L"])), roles_cli="H,L")
                family_chosen.append(item)
                used_H.add(Hkey); used_L.add(Lkey); globally_used.add(p["pair_id"])
        chosen.extend(family_chosen)
    assert len(chosen) <= 12
    return chosen, rankings


def render(doc):
    c = doc["search_counts"]
    lines = ["# 扩展 Random 候选：把教程公式变成可验证的问题", "",
             "这份文件不修改已冻结的候选或输入，也不运行仿真。筛选只使用原始 data 的 V、C 和预设规则，不把实测利用率加入评分。它安排下一步测试，不保证 Baseline 会差。", "",
             "最终整理本文件时已知初筛 tight256 的 warm U≈93.2903%、long U≈94.6361%。这些结果用于解释原候选为什么没达到目标；本扩展的筛选函数不读取结果文件，也没有根据该 U 调整评分或门槛。", "",
             "## 筛选范围与事先固定的规则", "",
             f'- 原始数据 {doc["raw_profile_count"]} 条；总输入至少 32K、miss 至少 128、hit 可按 128 token 整块读取、Bi<50 GiB/s 后剩 {c["raw_eligible_profiles"]} 条。',
             f'- 共 {c["all_unordered_profile_pairs"]} 个无序画像对，其中 {c["continuously_feasible_positive_count_ratio_pairs"]} 对存在正比例使理想 rho 位于 0.95–1.05。',
             f'- 为有限扫描，将每种画像的比例整数限制为 1–64，并只保留互质比例：得到 {c["bounded_integer_feasible_pairs"]} 对、{c["bounded_legal_pair_and_ratio_configurations"]} 个合法“画像对+配比”。这个整数范围不是所有可能输入的穷尽。',
             '- 每对若存在 |rho−1|≤0.005 的比例，优先总数最少的比例，再按误差及整数顺序打破平局；否则优先误差最小，再选较少计数。',
             '- H 表示 Bi 更高的画像，L 表示 Bi 更低的画像；这两个字母不代表总长度或计算长短。',
             '- 三个方向各选至多 4 对；每个方向内不重复 H 画像或 L 画像，并排除初始冻结矩阵已经安排的两画像对。完整排名见 JSON。',
             '- 初筛 seed=7；确认 seed=19、43、67、101。每卡先取至少 22 秒纯计算所需的最小整数倍人口，再独立完整随机打乱。保留所有结果及混合约束失败，不换坏种子。', "",
             "三个排序方向：", "",
             "1. **大读取、较短计算的 H**：要求 V_H>V_L、C_H<C_L、C 长短比≥1.5，H 纯计算贡献 15%–65%；按 `f_H * log2(V长/V短)` 排序。目的是提高可能受损类对整机的影响，而非只看某类很差。",
             "2. **尺度极端**：要求 C 长短比≥8、V 大小比≥2；按两种比例的 log2 乘积排序。用于测试突发和截止时间尺度，也可能因受损类纯计算贡献很小而仍有很高整机 U。",
             "3. **计算相近、读取相差大**：要求 C 长短比≤1.35、V 大小比≥2；按 `log2(V大小比)/(C长短比)` 排序。这是隔离计算差异影响的画像对照，并非单变量实验。", "",
             "## 最多十二对候选", "",
             "| ID | H 总K/miss；L 总K/miss | H:L 数量比 | rho | f_H | C_H/C_L ms | V_H/V_L MiB | U80 所需 H 均等 ms |", "|---|---|---|---:|---:|---|---|---:|"]
    for p in doc["selected_candidates"]:
        H,L=p["H"],p["L"]
        lines.append(f'| {p["candidate_id"]} | {H["data_key"][0]}/{H["data_key"][1]}；{L["data_key"][0]}/{L["data_key"][1]} | {p["ratio_counts_H_L"][0]}:{p["ratio_counts_H_L"][1]} | {p["rho_ideal"]:.6f} | {100*p["ideal_f_H"]:.2f}% | {H["C_s"]*1000:.3f}/{L["C_s"]*1000:.3f} | {H["V_GiB"]*1024:.3f}/{L["V_GiB"]*1024:.3f} | {p["target_H_mean_extra_wait_ms_for_U80"]:.3f} |')
    lines += ["", "最后一列对所有 H 层取平均，包括不等的层，并假设 L 不发生暴露等待。它是达到 U80 所需要的等待，不是等待预测。计算相近组里 H 未必算得更短，所以使用 f_H；JSON 同时给出真正短计算类的 f_short，避免混淆。", "",
              "## 与教程公式怎么对应", "", "```text",
              "Bi = V / C",
              "rho_ideal = 32 * sum(n_j*V_j) / [120 * sum(n_j*C_j)]",
              "f_short = sum(短计算类 n_j*C_j) / sum(所有类 n_j*C_j)",
              "f_H = n_H*C_H / (n_H*C_H + n_L*C_L)", "```", "",
              "请求数比例不能替代 f_short。n 表示数量，C 表示一次占用 NPU 多久；计算更久的请求，即使个数少，也可能占据大部分 NPU 计算时间。", "",
              "```text", "逐盘估算 W_disk = (真正排在前面的剩余字节 + 本短层本盘字节) / 40",
              "若 max_disk(W_disk) > C_short，读取不能藏在本层计算里。", "```", "",
              "必须用剩余 FIFO 前缀，不能把所有当前长卡都当成仍有一整层未服务。所有块仍为 176 KiB，同时提交的层会逐块交错；总层量是突发尺度，不是不可抢占的整层服务时间。JSON 中的层当量门槛只是均匀分盘估算。", "",
              "```text", "完整内部周期 D = 本层 compute_start 到下一层 compute_start",
              "平均 b = 下一层完整读取 V / D",
              "r = 平均 b / Bi = C / D", "```", "",
              "上述最后一个等式只直接用于同请求完整内部层。跨请求、截断窗口需要单独记账。r 是已发生周期的描述，不是事前预测。", "",
              "```text", "若 L 不 stall，H 每层平均多等 w：",
              "U ≈ 1 / (1 + f_H*w/C_H)",
              "要 U≈0.8，需要 w≈0.25*C_H/f_H", "```", "",
              "这是长期完成比例稳定、忽略启动和请求边界的近似。检验应先比较实测 w 是否接近所需 w，再核对实际 U，不能只引用‘一层长读取大于短计算时间’就宣布会低利用率。", "",
              "## 为什么很多 Random 可能仍然很好", "",
              "32 卡的独立时序容易分散读取突发，使 SSU 在一张卡计算时继续服务其他卡。长计算也给预取留出余量；大量内部层的 IO 即使等待，仍可能在计算截止前完成。", "",
              "还有一个守恒关系，能直接检查我们是否在追求不可能同时成立的数字：", "",
              "```text", "对整个有限输入，使用相同的从开始到全部完成的时间分母：",
              "NPU平均U = 全部计算卡秒 / (32 * 总时长)",
              "SSU平均忙率 = 全部读取GiB / (120 * 总时长)",
              "SSU平均忙率 = rho_ideal * NPU平均U", "```", "",
              "条件是每盘以固定 40 GiB/s 实际服务、所有读取和计算已守恒记全。若 rho≈1 且 Random 让 SSU 始终忙，整机 U 就不能同时很低；要全程 U≈80%，需要大约 20% 的 SSU 空闲。排队可以伤害某一类，却未必降低总吞吐。", "",
              "这个等式对整批完成的相同分母成立，不能把整批 rho 无条件套到 warm：窗口内推进的画像比例、在途字节和边界计算会改变比例。对长窗口也要实际核查，而不是直接当成恒等式。", "",
              "如果这批候选仍然很好，结论可能是这些约束下独立随机足以平滑突发。下一步可以提出每卡不同配比、固定角色、共同相位或有相关性的随机输入；但这些改变了 L1 或随机相关性，是新假设。不得悄悄当成同一种“每卡同配比、独立完整随机”。也可以把研究目标改成短类延迟或公平性，但要明确这与压低整机平均 U 是不同目标。", "",
              "## 实际验证要记录什么", "",
              "- 每个候选和种子的 warm/long U、每卡长短实际覆盖、每类条件 U、所有内部层的平均 stall（包括 0）、b/B 分布。",
              "- 每盘实际忙率、无待处理 IO 的空闲、短层释放时的 FIFO 前缀、IO-ready 相对计算截止的晚到量。",
              "- 全程计算/读取守恒以及上述忙率关系；窗口内结果另外统计，保留边界差异。",
              "- 不以‘理想平均欠载’替代逐盘逐时欠载；不预先承诺 Once 改善。", "",
              "来源：[原始 data](../../data)、[筛选脚本](adaptive_candidates.py)、[完整排名与参数](adaptive_candidates.json)。", ""]
    return "\n".join(lines)


def main():
    frozen_paths = [HERE / name for name in ("candidate_math.py", "candidate_math.json", "candidate_math.md")]
    frozen_paths += sorted((HERE / "inputs").glob("*.json.gz"))
    frozen_before = {str(p.relative_to(ROOT)): sha(p) for p in frozen_paths}
    data = ast.literal_eval((ROOT / "data").read_text())
    frozen = json.loads((HERE / "candidate_math.json").read_text())
    plan = json.loads((HERE / "study_plan.json").read_text())
    excluded = {tuple(sorted(tuple(p["data_key"]) for p in c["profiles"]))
                for c in frozen["candidates"] if len(c["profiles"]) == 2}
    profiles=[]
    for key, (B,C_us,source_ttft,V) in sorted(data.items()):
        if key[0] >= 32 and key[1] >= 128 and (key[0]*1024-key[1]) % 128 == 0 and B < 50:
            assert math.isclose(B, V/(C_us/1e6), rel_tol=1e-12)
            profiles.append(dict(data_key=list(key), B_GiB_s=B, C_s=C_us/1e6, V_GiB=V,
                                 category=("S" if key[0]<=80 else "L")+("L" if key[1]>=512 else "S")))
    pairs, counts=enumerate_pairs(profiles, excluded)
    chosen, rankings=select(pairs)
    assert {str(p.relative_to(ROOT)):sha(p) for p in frozen_paths} == frozen_before
    doc=dict(schema_version=1, status="heuristic_candidates_not_simulation_predictions",
             raw_profile_count=len(data), search_counts=counts,
             source=dict(data_sha256=sha(ROOT/"data"), generator_sha256=sha(Path(__file__)), study_plan_sha256=sha(HERE/"study_plan.json")),
             configuration=dict(num_npu=32,num_ssu=3,per_ssu_GiB_s=40,npu_link_GiB_s=50,n_layers=8,batch_size=1,
                                order="independent full queue random shuffle",same_ratio_on_every_npu=True,
                                minimum_pure_compute_ms=22000,pilot_seed=plan["pilot_seed"],confirmation_seeds=plan["confirmation_seeds"]),
             ranking_rules=dict(count_max_each=COUNT_LIMIT,counts_coprime=True,rho_range=[.95,1.05],
                                representative_ratio="Within |rho-1|<=0.005 minimize count sum, then error; otherwise minimize error, then count sum; integer lexicographic final tie.",
                                family1="VH>VL; CH<CL; C_scale>=1.5; .15<=fH<=.65; score=fH*log2(V_scale)",
                                family2="C_scale>=8; V_scale>=2; score=log2(C_scale)*log2(V_scale)",
                                family3="C_scale<=1.35; V_scale>=2; score=log2(V_scale)/C_scale",
                                diversity="Per family greedy rank scan: up to 4 pairs; no repeated H or L profile key; no globally duplicate pair; exclude initial frozen two-profile pairs.",
                                result_values_used_in_scoring=False),
             transparency=dict(known_before_final_file_write="Parent reported tight256 warm~93.2903%, long~94.6361%, S internal mean long stall~1.25303ms. Selection formulas had already been specified and do not read result files.",
                               outcomes_are_not_a_failure_of_the_equations="Required queue/stall thresholds were not predictions that the queues would occur."),
             frozen_files_unchanged_sha256=frozen_before, eligible_profiles=profiles,
             selected_candidates=chosen, full_family_rankings=rankings,
             legal_pair_inventory=[{k:p[k] for k in ("pair_id","ratio_counts_H_L","rho_ideal","legal_coprime_count_combinations","already_in_frozen_two_profile_matrix")} for p in pairs])
    (HERE/"adaptive_candidates.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    (HERE/"adaptive_candidates.md").write_text(render(doc))
    print(json.dumps(dict(selected=len(chosen),search_counts=counts,simulations_run=0),ensure_ascii=False))
    for p in chosen:
        print(p["candidate_id"],p["profiles_cli"],p["counts_cli"],round(p["rho_ideal"],6),round(p["ideal_f_H"],6))


if __name__ == "__main__":
    main()
