#!/usr/bin/env python3
"""Separate 6/8-SSU sensitivity candidates, derived from raw data only."""

import ast
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LONG_KEYS = ((200, 2048), (200, 4096))
SHORT_KEYS = ((32, 128), (32, 256), (32, 512), (48, 128))
SELECTED = (
    (6, (200, 2048), (32, 128)),
    (8, (200, 2048), (32, 128)),
    (6, (200, 2048), (32, 256)),
    (8, (200, 2048), (32, 256)),
    (6, (200, 2048), (32, 512)),
    (8, (200, 2048), (48, 128)),
    (6, (200, 4096), (32, 256)),
    (8, (200, 4096), (32, 256)),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(key, data):
    B, C_us, _, V = data[key]
    assert key[0] >= 32 and key[1] >= 128 and (key[0]*1024-key[1]) % 128 == 0
    assert B < 50 and math.isclose(B, V/(C_us/1e6), rel_tol=1e-12)
    return dict(data_key=list(key), B_GiB_s=B, C_s=C_us/1e6, V_GiB=V,
                blocks_per_layer=(key[0]*1024-key[1])//128,
                category=("S" if key[0]<=80 else "L")+("L" if key[1]>=512 else "S"))


def derive(disks, Lkey, Skey, data):
    L, S = profile(Lkey,data), profile(Skey,data)
    cap=disks*40.;x=cap/32
    assert L["B_GiB_s"] < x < S["B_GiB_s"]
    exact_k=(x*L["C_s"]-L["V_GiB"])/(S["V_GiB"]-x*S["C_s"])
    feasible=[]
    for nl in range(1,65):
        for ns in range(1,513):
            if math.gcd(nl,ns)!=1:
                continue
            rho=32*(nl*L["V_GiB"]+ns*S["V_GiB"])/(cap*(nl*L["C_s"]+ns*S["C_s"]))
            if .95<=rho<=1.05:
                e=abs(rho-1)
                rank=(0,nl+ns,e,nl,ns)if e<=.005 else(1,e,nl+ns,nl,ns)
                feasible.append((rank,nl,ns,rho))
    if not feasible:
        return dict(num_ssu=disks,L=L,S=S,status="bounded_integer_search_empty",
                    exact_short_per_long_ratio_at_rho1=exact_k,
                    warning="This bounded-search outcome is not mathematical infeasibility.")
    _,nl,ns,rho=min(feasible)
    total_C=nl*L["C_s"]+ns*S["C_s"]
    f=ns*S["C_s"]/total_C
    copies=math.ceil(22/(8*total_C))
    name=f'topo{disks}_m{Lkey[1]}_s{Skey[0]}m{Skey[1]}'
    Lreq=L["C_s"]*8000
    ideal_work_per_L=8*total_C/nl*1000
    link_ms=S["V_GiB"]/50*1000
    slack_ms=S["C_s"]*1000-link_ms
    assert slack_ms>0
    selected=(disks,Lkey,Skey)in SELECTED
    cli=(f'python results/baseline_random_near_capacity_20260914/experiment.py --name {name}'
         f' --profiles {Lkey[0]}:{Lkey[1]},{Skey[0]}:{Skey[1]} --counts {nl},{ns}'
         f' --roles L,S --num-ssu {disks} --horizon-ms 22000 --seed 7 --strategy baseline')
    return dict(candidate_id=name,status="mathematically_feasible_not_simulated_by_this_script",
                selected=selected,num_npu=32,num_ssu=disks,per_ssu_GiB_s=40,npu_link_GiB_s=50,n_layers=8,
                L=L,S=S,ratio_L_S=[nl,ns],exact_short_per_long_ratio_at_rho1=exact_k,
                legal_integer_ratios_in_declared_bounds=len(feasible),rho_ideal=rho,
                f_short=f,ideal_compute_only_mean_short_cards=32*f,
                C_short_ms=S["C_s"]*1000,
                L_layer_balanced_storage_service_ms=L["V_GiB"]/cap*1000,
                S_layer_balanced_storage_service_ms=S["V_GiB"]/cap*1000,
                S_minimum_NPU_link_service_ms=link_ms,
                S_first_link_start_delay_threshold_ms=slack_ms,
                storage_only_L_layer_equivalent_threshold=(cap*S["C_s"]-S["V_GiB"])/L["V_GiB"],
                first_link_delay_L_layer_equivalent_threshold=cap*(slack_ms/1000)/L["V_GiB"],
                queue_threshold_assumption="Balanced residual L FIFO prefixes delay all first S blocks. Actual block interleaving and streaming link delivery require trace validation; a whole current L request is not a full queued layer.",
                required_mean_short_stall_ms_for_U80=.25*S["C_s"]*1000/f,
                wait_model="All S layers, including zero-stall layers; L unstalled; fixed long-run completion proportions; ignores startup and request-boundary effects. Required amount, not prediction.",
                L_request_pure_compute_ms=Lreq,
                ideal_compute_work_per_L_request_ms=ideal_work_per_L,
                warm_mix_risk_flags=(["One L request exceeds 1s; two consecutive L can exceed the 2s warm window"]if Lreq>1000 else[])
                                    +(["Pure compute work per L exceeds 2s; many cards may have no L in warm"]if ideal_work_per_L>2000 else[]),
                warm_mix_always_requires_actual_compute_validation=True,
                minimum_22s_population=dict(L=nl*copies,S=ns*copies,repeats=copies,
                                           pure_compute_ms_per_npu=8*total_C*copies*1000,
                                           expected_blocks=32*8*copies*(nl*L["blocks_per_layer"]+ns*S["blocks_per_layer"])),
                cli=cli if selected else None)


def render(doc):
    rows=["# 改变盘数的二级候选：6 SSU 与 8 SSU", "",
          "主实验仍为 3 SSU。本文件将盘数变化作为单独的拓扑敏感性实验；同时改变了容量与请求配比，不能把其收益或损失解释成只改变 QoS 策略。没有运行新仿真，筛选没有使用已观测 U 排名。", "",
          "固定 32 NPU、每盘 40 GiB/s、NPU 链路 50 GiB/s、8 层、batch=1、原有跨请求首层预取。所有画像直接来自 data，最小总输入 32K，miss 至少 128，单层按 176 KiB 整块读取，Bi<50。每卡相同数量比例，再独立完整打乱队列。", "",
          "## 先算比例和需要发生的等待", "", "```text", "S = 盘数 * 40",
          "rho = 32*(nL*VL+nS*VS) / [S*(nL*CL+nS*CS)]",
          "f_short = nS*CS / (nL*CL+nS*CS)",
          "若长类不 stall：U ≈ 1 / (1 + f_short*w/CS)",
          "要 U≈80%：所有短层平均额外等待 w≈0.25*CS/f_short", "```", "",
          "w 包括不等待的短层。这个近似要求长期完成比例稳定，忽略启动与请求边界；它回答需要多大等待，不预言实际会出现这么多等待。", "",
          "扫描范围是 2 种盘数×2 种长画像×4 种短画像，共 16 个组合；整数配比搜索 nL=1…64、nS=1…512、互质。若存在 |rho−1|≤0.005 的配比，先取计数和最小的，再取误差小的；否则取最接近 rho=1 的。先用代数验证正比例可行，避免把搜索上限误当成数学不可行。", "",
          "## 最多八组可运行候选", "",
          "| ID | 盘数 | L miss / S 总K及miss | L:S | rho | f_short | U80 所需平均等 ms |", "|---|---:|---|---|---:|---:|---:|"]
    for c in doc["selected_candidates"]:
        rows.append(f'| {c["candidate_id"]} | {c["num_ssu"]} | 200K/{c["L"]["data_key"][1]}；{c["S"]["data_key"][0]}K/{c["S"]["data_key"][1]} | {c["ratio_L_S"][0]}:{c["ratio_L_S"][1]} | {c["rho_ideal"]:.6f} | {100*c["f_short"]:.2f}% | {c["required_mean_short_stall_ms_for_U80"]:.3f} |')
    rows += ["", "选择顺序覆盖同一画像在 6/8 盘的变化、较紧/较松短截止时间、48K 高带宽短类、两种长计算时间。没有声称这是全空间中最差的八组。", "",
             "200K/2048 的整请求纯计算约 565.924 ms；200K/4096 为 1130.686 ms。后者连续两个长请求就可能覆盖整个 warm，所以重点先验证前者。即使前者也必须实测每卡 warm 长短覆盖，不能保证独立随机一定满足。", "",
             "8 盘搭 200K/2048 与 32K/512 可以让短计算贡献达到约 81%，但平均每个长请求对应的纯计算工作超过 3 秒，warm 内很多卡可能没有长类；因此它保留在完整扫描记录中，而不列入优先启动的八组。", "",
             "## 盘数变多，有两个相反作用", "",
             "为了仍然 rho≈1，需要更多短请求，因此 f_short 增大：短类一旦等待，对整机影响更明显。但每个长读取跨更多盘并行处理，其服务突发缩短：同样的排前读取量可能更容易被计算隐藏。所以不能只看 f_short 增大就断言 U 会更低。", "",
             "| ID | 长层盘侧服务 ms | 短层 C ms | 短层自身链路最低 ms | 可容忍首次链路延迟 ms |", "|---|---:|---:|---:|---:|"]
    for c in doc["selected_candidates"]:
        rows.append(f'| {c["candidate_id"]} | {c["L_layer_balanced_storage_service_ms"]:.6f} | {c["C_short_ms"]:.6f} | {c["S_minimum_NPU_link_service_ms"]:.6f} | {c["S_first_link_start_delay_threshold_ms"]:.6f} |')
    rows += ["", "这里显式补上 50 GiB/s 的 NPU 链路。仅用 V/(盘数×40) 算的是盘侧时间，不能直接当完整读取完成时间。", "",
             "```text", "短层第一次开始进入 NPU 链路，相对本层计算开始晚了 tau：",
             "整层 IO 完成时间至少为 tau + VS/50",
             "如果 tau > CS - VS/50，就一定不能在本层计算内藏住读取。", "```", "",
             "这是假设同请求内部层在本层计算开始时预取下一层的时长下界。第一次进入链路后，如果还有供给间隙，实际完成会更晚；磁盘与链路可以流式重叠，不应把全部盘服务时间和链路服务时间直接相加。", "",
             "例如 32K/128 的 C≈1.178 ms，自身链路接收至少≈0.856 ms，只剩≈0.322 ms 的首次供给余量。48K/128 自身接收至少≈1.286 ms，而 C≈1.514 ms，余量更小。这个变量比单看长层总服务时间更接近可以验证的截止条件。", "",
             "但真正排前的是剩余 IO 块。Baseline 逐个 176 KiB 块服务；同时释放的层可能交错提交，长卡也可能早已读完。必须通过 trace 量到 tau、FIFO 前缀和最终 IO-ready，再判断这些候选为什么成功或失败。", "",
             "## 统计口径和成本", "",
             "- 新 metadata 分别记录 num_ssu=6/8；文件名和下列命令带不同候选名。主报告应与 3 盘分表。",
             "- 新输入每卡纯计算至少 22 秒；只 Random，pilot=7，确认=19/43/67/101。不能复制所有卡的排列、引入同步屏障或按坏 seed 重抽。",
             "- warm[2,4)、long[2,20) 均报告 U；每卡 warm 是否两类都计算必须单独验证。失败结果保留，但不能当满足混合约束的最终证据。",
             "- 平均 rho≈1 不是逐盘逐时欠载。整个有限批次相同分母下仍有 SSU忙率=rho×NPU U；若 Random 使 SSU一直忙，U仍难很低。",
             "- 增盘并保持接近满载，会增加完成同样计算时长需要模拟的读取块数。下面记录完整预计块数，不能把六小时预算只按候选个数估算。", "",
             "| ID | 每卡 L/S 数量 | 每卡纯计算 s | 总 IO 块数（百万） | warm 风险标记 |", "|---|---|---:|---:|---|"]
    for c in doc["selected_candidates"]:
        p=c["minimum_22s_population"]
        risk="长请求超过1秒"if c["L_request_pure_compute_ms"]>1000 else"仍需日志验证"
        rows.append(f'| {c["candidate_id"]} | {p["L"]}/{p["S"]} | {p["pure_compute_ms_per_npu"]/1000:.3f} | {p["expected_blocks"]/1e6:.3f} | {risk} |')
    rows += ["", "## 可运行命令（由主任务决定启动；本脚本没有执行）", "",
             "下列初筛均为 Baseline、无 trace；若发现值得复验的低 U，再由主任务补充 trace 和确认种子。", "", "```bash"]
    rows += [c["cli"]for c in doc["selected_candidates"]]
    rows += ["```", "", "[完整扫描与参数](topology_candidates.json) · [数学生成器](topology_candidates.py) · [主 3 盘候选](candidate_math.md)", ""]
    return "\n".join(rows)


def main():
    preserved=[HERE/n for n in ("candidate_math.py","candidate_math.json","candidate_math.md","adaptive_candidates.py","adaptive_candidates.json","adaptive_candidates.md","study_plan.json","experiment.py")]
    before={str(p.relative_to(ROOT)):sha(p)for p in preserved}
    data=ast.literal_eval((ROOT/"data").read_text())
    scan=[derive(ssu,L,S,data)for ssu,L,S in itertools.product((6,8),LONG_KEYS,SHORT_KEYS)]
    lookup={(c["num_ssu"],tuple(c["L"]["data_key"]),tuple(c["S"]["data_key"])):c for c in scan}
    selected=[lookup[k]for k in SELECTED]
    assert len(selected)<=8 and all(c["status"]!="bounded_integer_search_empty"for c in selected)
    assert all(.95<=c["rho_ideal"]<=1.05 for c in selected)
    assert before=={str(p.relative_to(ROOT)):sha(p)for p in preserved}
    doc=dict(schema_version=1,status="separate_topology_sensitivity_mathematical_candidates",primary_study_num_ssu=3,
             source=dict(data_sha256=sha(ROOT/"data"),generator_sha256=sha(Path(__file__))),
             no_new_simulations=True,no_observed_utilization_used_for_selection=True,
             scan_scope=dict(num_ssu=[6,8],long_profiles=[list(k)for k in LONG_KEYS],short_profiles=[list(k)for k in SHORT_KEYS],
                             profile_topology_combinations=16,count_bound_L=64,count_bound_S=512,counts_coprime=True,rho_range=[.95,1.05]),
             selection="Explicit coverage: L200/2048 with S32/128 and S32/256 at both 6/8 SSU; S32/512 at 6 SSU; S48/128 at 8 SSU; L200/4096+S32/256 at both. Omit high-f 8SSU S32/512 from priority list because sparse L may fail warm mixed coverage.",
             seed_plan=dict(pilot=[7],confirmation=[19,43,67,101]),
             preserved_files_sha256=before,selected_candidates=selected,full_scan=scan)
    (HERE/"topology_candidates.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    (HERE/"topology_candidates.md").write_text(render(doc))
    print(json.dumps(dict(selected=len(selected),scanned=len(scan),simulations_run=0)))
    for c in selected:
        print(c["candidate_id"],c["ratio_L_S"],round(c["rho_ideal"],6),round(c["f_short"],6),c["minimum_22s_population"])


if __name__=="__main__":
    main()
