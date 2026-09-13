#!/usr/bin/env python3
"""Independent raw-result audit and mathematical follow-up, without simulation."""
import ast
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

from math_followup_constructed import stationary_coverage, ideal_clock_coverage

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=(7,19,43,67,101)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values)>1 else None,
                min=min(values),max=max(values))


def read_gz(path):
    with gzip.open(path,"rt") as f:return json.load(f)


def overlap(a,b,start,end):
    return max(0.,min(b,end)-max(a,start))


def audit_case(name,seed,comparison,data):
    case=HERE/f"runs/{name}_ssu3_h22000_seed{seed}/baseline"
    a=json.loads((case/"analysis.json").read_text())
    manifest=read_gz(case/"manifest.json.gz")
    result=read_gz(case/"result.json.gz")["summary"]
    physical=json.loads((case/"physical_service.json").read_text())
    assert a["all_technical_checks_passed"]
    for path,digest in a["sources"].items():assert sha(Path(path))==digest,path
    q={q["request_id"]:q for q in manifest["requests"]}
    for req in q.values():
        load=req["load"];row=data[load["seq_len_k"],load["nql"]]
        assert load["per_layer_us"]==row[1] and load["per_layer_kv_gb"]==row[3]
    total_c=math.fsum(8*req["load"]["per_layer_us"]/1000 for req in q.values())
    short_c=math.fsum(8*req["load"]["per_layer_us"]/1000 for req in q.values() if req["load"]["role"]=="S")
    total_v=math.fsum(8*req["load"]["per_layer_kv_gb"] for req in q.values())
    rho=32*total_v/(120*total_c/1000);f=short_c/total_c
    cs=next(req["load"]["per_layer_us"]/1000 for req in q.values() if req["load"]["role"]=="S")
    full_u=total_c/(32*result["makespan_ms"])
    full_ssd=total_v/(120*result["makespan_ms"]/1000)
    assert abs(full_u-result["fleet_npu_compute_utilization"])<1e-10
    assert abs(full_ssd-result["ssd_mean_utilization"])<1e-10
    rows=[]
    for start,end in ((2000,4000),(2000,20000)):
        computing=[[] for _ in range(32)];active=[[] for _ in range(32)]
        seen=[set() for _ in range(32)];stall={"S":[],"L":[]}
        for batch in result["microbatch_metrics"]:
            npu=batch["npu_id"];role=q[batch["member_request_ids"][0]]["load"]["role"]
            active[npu].append(overlap(batch["admission_time_ms"],batch["completion_time_ms"],start,end))
            for layer in batch["layer_metrics"]:
                c=overlap(layer["compute_start_ms"],layer["compute_end_ms"],start,end)
                computing[npu].append(c)
                if c>1e-9:seen[npu].add(role)
            for cur,nxt in zip(batch["layer_metrics"],batch["layer_metrics"][1:]):
                if cur["compute_start_ms"]>=start and nxt["compute_start_ms"]<=end:
                    stall[role].append(max(0.,nxt["compute_start_ms"]-cur["compute_end_ms"]))
        u=100*math.fsum(math.fsum(x) for x in computing)/(32*(end-start))
        bins=range(start//2000,end//2000)
        ssd=100*math.fsum(physical["ssd_busy_ms"][d][j] for d in range(3) for j in bins)/(3*(end-start))
        w=math.fsum(stall["S"])/len(stall["S"])
        mixed=sum(len(s)==2 for s in seen)
        all_active=all(abs(math.fsum(v)-(end-start))<1e-6 for v in active)
        published=next(z for z in a["windows"] if (z["start_ms"],z["end_ms"])==(start,end))
        tab=next(z for z in comparison["rows"] if z["candidate"]==name and z["seed"]==seed and
                 z["window_start_s"]==start/1000 and z["window_end_s"]==end/1000)
        assert abs(u-published["U_percent"])<1e-8 and abs(u-tab["U_percent"])<1e-8
        assert abs(ssd-tab["actual_ssd_mean_busy_percent"])<1e-8
        assert abs(w-tab["short_mean_stall_ms_including_zeros"])<1e-8
        assert mixed==published["long_short_mixed_card_count"]==tab["long_short_mixed_cards"]
        assert all_active==published["all_32_active"]==tab["all_32_active"]
        rows.append(dict(candidate=name,seed=seed,window_ms=[start,end],U_percent=u,
            all_32_active=all_active,mixed_cards=mixed,
            missing_mixed_cards=[i for i,s in enumerate(seen) if len(s)!=2],
            actual_SSD_busy_percent=ssd,complete_S_internal_cycles=len(stall["S"]),
            mean_S_internal_stall_ms_including_zeros=w,
            total_L_internal_stall_ms=math.fsum(stall["L"]),
            rho_ideal=rho,f_short=f,C_short_ms=cs,required_w_U80_ms=.25*cs/f,
            approximate_U_from_measured_w_percent=100/(1+f*w/cs),
            input_fingerprint=result["input_fingerprint"],
            sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                (case/"result.json.gz",case/"manifest.json.gz",case/"analysis.json",case/"physical_service.json")}))
    return dict(candidate=name,seed=seed,rows=rows,full_finite_U_percent=100*full_u,
        full_finite_SSD_busy_percent=100*full_ssd,
        full_finite_SSD_minus_rho_U_pp=100*(full_ssd-rho*full_u))


def choose_ratio(vl,cl,vs,cs,disks):
    candidates=[]
    for nl in range(1,65):
        for ns in range(1,1025):
            if math.gcd(nl,ns)!=1:continue
            rho=32*(nl*vl+ns*vs)/(40*disks*(nl*cl+ns*cs)/1000)
            if abs(rho-1)<=.005:candidates.append((nl+ns,abs(rho-1),nl,ns,rho))
    return min(candidates)[2:]


def main():
    protected=[HERE/name for name in ("experiment.py","study_plan.json","candidate_math.json",
        "adaptive_candidates.json","topology_candidates.json","topology_s128_appendix.json","constructed_candidates.json")]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    data=ast.literal_eval((ROOT/"data").read_text())
    comparison_path=HERE/"comparison.json"
    comparison=json.loads(comparison_path.read_text())
    main_cases=[audit_case(name,seed,comparison,data) for name in ("main512","main1024") for seed in SEEDS]
    groups=[]
    for name in ("main512","main1024"):
        for window in ([2000,4000],[2000,20000]):
            rows=[r for c in main_cases if c["candidate"]==name for r in c["rows"] if r["window_ms"]==window]
            groups.append(dict(candidate=name,window_ms=window,seeds=list(SEEDS),
                U_percent=stats([r["U_percent"] for r in rows]),
                actual_SSD_busy_percent=stats([r["actual_SSD_busy_percent"] for r in rows]),
                mean_S_internal_stall_ms=stats([r["mean_S_internal_stall_ms_including_zeros"] for r in rows]),
                approximate_U_from_measured_w_percent=stats([r["approximate_U_from_measured_w_percent"] for r in rows]),
                all_active_pass_count=sum(r["all_32_active"] for r in rows),
                warm_or_long_mixed_pass_count=sum(r["mixed_cards"]==32 for r in rows),
                mixed_cards_by_seed={str(r["seed"]):r["mixed_cards"] for r in rows},
                f_short=rows[0]["f_short"],required_w_U80_ms=rows[0]["required_w_U80_ms"],
                rho_ideal=rows[0]["rho_ideal"]))

    prior=json.loads((HERE/"topology_s128_appendix.json").read_text())
    kappa_old=prior["source_calibration"]["kappa"]
    topology=[]
    for disks,lm in ((6,2048),(8,2048),(6,4096),(8,4096)):
        case=HERE/f"runs/topo{disks}_m{lm}_s32m128_ssu{disks}_h22000_seed7/baseline"
        path=case/"analysis.json"
        if not path.exists():continue
        a=json.loads(path.read_text());w=next(w for w in a["windows"] if (w["start_ms"],w["end_ms"])==(2000,20000))
        actual_wait=w["complete_internal_cycles_inside_window"]["per_role"]["S"]["stall_ms_including_zeros"]["mean"]
        vl=data[200,lm][3];scale=vl/(40*disks)*1000
        published=next((r for r in prior["selected_candidates"] if r["num_ssu"]==disks and lm==4096),None)
        topology.append(dict(num_ssu=disks,long_nql=lm,actual_long_U_percent=w["U_percent"],
            actual_mean_S_stall_ms=actual_wait,L_balanced_service_ms=scale,actual_kappa=actual_wait/scale,
            source=str(path.relative_to(ROOT)),source_sha256=sha(path),
            preregistered_prediction_U_percent=published["conditional_prediction"]["approximate_U_percent"] if published else None,
            preregistered_prediction_w_ms=published["conditional_prediction"]["mean_S_stall_ms"] if published else None))

    uniform=[]
    for disks,lm in ((12,2048),(16,2048),(12,4096),(16,4096),(12,1024)):
        vl=data[200,lm][3];cl=data[200,lm][1]/1000
        vs=data[32,128][3];cs=data[32,128][1]/1000
        nl,ns,rho=choose_ratio(vl,cl,vs,cs,disks)
        f=ns*cs/(nl*cl+ns*cs);rep=math.ceil(22000/(8*(nl*cl+ns*cs)))
        anchors=[r for r in topology if r["long_nql"]==lm]
        anchor=max(anchors,key=lambda r:r["num_ssu"]) if anchors else None
        uniform.append(dict(name=f"followup{disks}_m{lm}_s32m128",num_ssu=disks,
            profiles=[[200,lm],[32,128]],counts=[nl,ns],rho_ideal=rho,f_short=f,
            required_mean_S_stall_ms_for_U80=.25*cs/f,
            per_card_population=dict(L=nl*rep,S=ns*rep,pure_compute_ms=8*(nl*cl+ns*cs)*rep),
            L_request_pure_compute_ms=8*cl,
            stationary_no_stall_coverage=stationary_coverage(cl,cs,nl,ns),
            ideal_compute_clock_coverage=ideal_clock_coverage(cl,cs,nl,ns,rep),
            latest_same_L_kappa_sensitivity=(dict(source=anchor["source"],kappa=anchor["actual_kappa"],
                hypothetical_mean_wait_ms=anchor["actual_kappa"]*vl/(40*disks)*1000,
                hypothetical_U_percent=100/(1+f*(anchor["actual_kappa"]*vl/(40*disks)*1000)/cs)) if anchor else None),
            no_new_simulation=True))

    cl=data[200,1024][1]/1000;vl=data[200,1024][3]
    cs=data[32,128][1]/1000;vs=data[32,128][3]
    hetero_groups=[]
    for label,cards,nl,ns in (("hot",list(range(20)),1,18),("cold",list(range(20,32)),1,2)):
        f=ns*cs/(nl*cl+ns*cs);rate=(nl*vl+ns*vs)/(nl*cl+ns*cs)*1000
        repeats=math.ceil(22000/(8*(nl*cl+ns*cs)))
        hetero_groups.append(dict(group=label,npu_ids=cards,counts=[nl,ns],f_short=f,
            ideal_per_card_rate_GiB_s=rate,population=dict(L=nl*repeats,S=ns*repeats,
                pure_compute_ms=8*repeats*(nl*cl+ns*cs)),
            stationary_no_stall_coverage=stationary_coverage(cl,cs,nl,ns)))
    demand=math.fsum(len(g["npu_ids"])*g["ideal_per_card_rate_GiB_s"] for g in hetero_groups)
    hot_f=hetero_groups[0]["f_short"]
    required_hot_u=(32*.8-12)/20
    required_hot_w=cs*(1/required_hot_u-1)/hot_f
    def fleet_mix_probability(hot_w):
        return ((1-sum(stationary_coverage(cl,cs,1,18,hot_w)[key] for key in ("no_L_per_card","no_S_per_card")))**20
            *(1-sum(stationary_coverage(cl,cs,1,2)[key] for key in ("no_L_per_card","no_S_per_card")))**12)
    hetero=dict(name="hetero12_m1024_s32m128_hot20",status="mathematical proposal, no manifest or simulation generated",
        num_npu=32,num_ssu=12,profiles=[[200,1024],[32,128]],groups=hetero_groups,
        request_order="Each full queue independently shuffled with seed+100003*npu. Cards0..19 and20..31 have different counts; no synchronized deck or ordered input.",
        ideal_total_rate_GiB_s=demand,ideal_sum_card_rate_over_capacity=demand/480,
        L_request_pure_compute_ms=8*cl,
        all32_mixed_probability_no_wait_in_toy_model=fleet_mix_probability(0),
        U80_assuming_cold_cards_full_compute=dict(required_hot_card_U=required_hot_u,
            required_mean_hot_S_stall_ms=required_hot_w,
            toy_all32_mixed_probability_at_required_hot_wait=fleet_mix_probability(required_hot_w)),
        required_control="Same raw pair and12SSU with uniform per-card counts, e.g. followup12_m1024_s32m128, then same fixed seeds for both.",
        limitation="Changing per-card ratios alone is not a proof of harm: if all short layers retain the same average wait, the convex wait model predicts heterogeneity raises fleet mean U at equal mean f. The new input must actually increase or correlate hot-card waiting.")

    doc=dict(schema_version="math-followup-v1",scope="Independent main5seed raw audit, corrected topology extrapolation, optional raw heterogeneous mix",
        no_new_simulations=True,main_cases=main_cases,main_groups=groups,
        raw_topology_observations=topology,unrun_uniform_raw_sensitivity=uniform,
        unrun_heterogeneous_raw_proposal=hetero,
        constructed_next_step="constructed_candidates.json / constructed_candidates.md: new10/12/16K extrapolation changes the mathematical wait threshold, unlike simply reusing a disproven disk-scaling coefficient",
        conditional_bound=dict(identity="Same complete population and[0,makespan]: SSD_busy=rho_population*U",
            example="If rho_population<=1.05 AND SSD_busy>=.90 over that complete interval, U>=.90/1.05=.857142857. The assumptions do not prove SSD_busy is always>=.90 for every random input.",
            cannot_claim="No universal80/90% lower bound has been proved for the actual finite, coupled, layerwise simulation or its warm window.",
            warm_warning="Whole-input rho is not necessarily the work ratio progressing inside fixed T. Compute-only, partial-cycle, startup, and draining behavior must not be conflated."),
        jensen_argument=dict(model="U_i=1/(1+a*f_i), a=w/CS, same a across cards",
            second_derivative="d2U/df2=2*a*a/(1+a*f)^3>0",
            conclusion="At fixed average f, mean_i U_i >= U(mean_i f). Heterogeneous proportions alone do not lower this model's fleet utilization.",
            way_to_break_assumption="Different per-card mixes can change actual queue waits and role phase correlations; measure those changes, rather than assuming them."),
        comparison_snapshot=dict(path=str(comparison_path.relative_to(ROOT)),sha256=sha(comparison_path),updated_utc=comparison["updated_utc"]),
        preserved_files_sha256=before,generator_sha256=sha(Path(__file__)))
    (HERE/"math_followup.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")

    lines=["# 后续数学核对：为什么主组仍然高，下一步改变什么", "",
        "主组的五个预注册种子都已完成。下面直接重读原始manifest、逐层result和物理服务计数，独立裁剪warm[2,4)与长窗[2,20)，再与comparison核对；没有运行新仿真。", "",
        "| 主组 | 窗口 | 平均U ± 样本标准差 | 实际SSD平均忙率 | 32卡都活跃 | 32卡都长短混合 |", "|---|---|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['candidate']} | {g['window_ms'][0]/1000:g}—{g['window_ms'][1]/1000:g}s | {g['U_percent']['mean']:.6f}% ± {g['U_percent']['sample_std']:.6f}pp | {g['actual_SSD_busy_percent']['mean']:.6f}% | {g['all_active_pass_count']}/5 | {g['warm_or_long_mixed_pass_count']}/5 |")
    lines += ["", "种子为7、19、43、67、101，各种子等权。main512的warm只有seed19未全卡混合（30/32）；main1024的warm五个种子都未全卡混合，所以这些warm数字不能被当作满足全部约束的成功样本。两组长窗均5/5全卡混合。", "",
        "没有达到低U，不代表教程公式错了。公式告诉我们‘发生多少等待后会损失多少计算时间’，不会自动制造那段等待：", "",
        "```text", "同请求完整内部周期：D = C + w；平均b/B = C/D",
        "若长类不等、完成比例稳定：U≈1/(1+f_short*w/C_short)",
        "要U≈80%，所有短层的平均等待需要w≈0.25*C_short/f_short", "```", "",
        "| 长窗主组 | 短类纯计算权重 | 实测短层平均等待 ms | U80需要 ms | 由实测等待算出的近似U | 实际U |", "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        if g["window_ms"]!=[2000,20000]:continue
        lines.append(f"| {g['candidate']} | {100*g['f_short']:.4f}% | {g['mean_S_internal_stall_ms']['mean']:.6f} | {g['required_w_U80_ms']:.6f} | {g['approximate_U_from_measured_w_percent']['mean']:.6f}% | {g['U_percent']['mean']:.6f}% |")
    lines += ["", "均等待包括零等待层；不是只平均被阻塞的层。长类完整内部层等待为0。main512的短层等待仅约目标的21%，main1024约13%。增加短计算权重虽然有帮助，但main1024的较长截止时间同时藏住了更多IO，实际平均等待反而更小。这解释了它为何更难压低整机U。", "",
        "一层大读取的服务时间超过某个门槛，只证明‘排到足够多残余块时可能迟到’。32卡独立随机使请求层释放更分散；正在计算的长卡往往已经读完下一层，不能把‘有多少张长卡’当成‘前面排着多少层长读’。各层只有176KiB块依次服务，不是整层原子服务。主组没有观测到足够大的持续平均等待。", "",
        "原来的增盘预测需要明确纠正：", "",
        "| 盘数 / L miss | 实际长窗U | 实际短层w ms | 实际kappa=w/(VL/容量) | 原先已写下的预测U |", "|---|---:|---:|---:|---:|"]
    for r in topology:
        pred=f"{r['preregistered_prediction_U_percent']:.6f}%" if r['preregistered_prediction_U_percent'] is not None else "未预注册该数"
        lines.append(f"| {r['num_ssu']} / {r['long_nql']} | {r['actual_long_U_percent']:.6f}% | {r['actual_mean_S_stall_ms']:.9f} | {r['actual_kappa']:.6f} | {pred} |")
    lines += ["", f"旧3盘L4096的kappa={kappa_old:.6f}。它不能直接迁移到6/8盘：6盘L4096原本按比例猜w≈0.953ms、U≈88.30%，实测w≈0.535ms、U≈93.21%；8盘原猜U≈87.49%，实际≈94.99%。盘数和请求比例同时变化，队列更分散，等待下降比简单3/m缩放快。保留原预测以便核对，不把失败预测改写成事后正确解释。", "",
        "12/16盘只是额外拓扑，不能继续用已失效的旧系数保证低U。下面采用最新相同长画像的8盘系数作另一种条件参照，也并未证明它能推广：", "",
        "| 盘数 / L miss | L:S | rho | 短计算权重 | U80需要w ms | 用最新系数的条件U | 无等待时全卡混合粗略概率 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in uniform:
        pred=r["latest_same_L_kappa_sensitivity"]
        lines.append(f"| {r['num_ssu']} / {r['profiles'][0][1]} | {r['counts'][0]}:{r['counts'][1]} | {r['rho_ideal']:.6f} | {100*r['f_short']:.2f}% | {r['required_mean_S_stall_ms_for_U80']:.6f} | {pred['hypothetical_U_percent']:.2f}% | {100*r['stationary_no_stall_coverage']['all_32_mixed_if_cards_independent']:.2f}% |" if pred else f"| {r['num_ssu']} / {r['profiles'][0][1]} | {r['counts'][0]}:{r['counts'][1]} | {r['rho_ideal']:.6f} | {100*r['f_short']:.2f}% | {r['required_mean_S_stall_ms_for_U80']:.6f} | 未校准 | {100*r['stationary_no_stall_coverage']['all_32_mixed_if_cards_independent']:.2f}% |")
    lines += ["", "概率来自平稳、独立Bernoulli标签、固定时长的简化模型，不是实际warm保证。全队列有限打乱和共享盘反馈都不符合这些理想假设。它提示：L4096一条请求纯计算超过1秒，增加短数量后，两秒内没有长类或没有短类都可能发生。L2048的覆盖风险较小，但实测等待系数也较小。", "",
        "异质配比可以作为独立假设：12盘，相同raw L200K/1024与S32K/128；NPU0—19各自L:S=1:18，NPU20—31各自1:2，然后每张卡独立打乱完整队列。两组分别准备49L+882S、73L+146S，纯计算均超过22秒。", "",
        f"热组单卡理想需求18.317307GiB/s，冷组9.333943GiB/s，总需求{demand:.6f}GiB/s，约容量的{100*demand/480:.4f}%。热组短计算权重37.43%，冷组6.23%；L的整请求纯计算约283.544ms。对照应使用同画像、同12盘、统一比例的独立随机输入，不能与不同画像直接归因。", "",
        "它只是可检验的机制，不是保证变差。若所有卡短层平均等待w都不变：", "",
        "```text", "U_i = 1/(1+a*f_i)，a=w/C_S",
        "d²U/df² = 2*a²/(1+a*f)³ > 0",
        "因此固定平均f时，平均U_i >= U(平均f)", "```", "",
        "也就是说，单把配比分散到不同卡，反而会提高这个简化模型的平均U。只有它实际拉长热卡的短请求连续段，并造成更大或更集中的排队等待，才可能更差。要整机U=80%，若12张冷卡全算，20张热卡平均U要68%，其所有短层平均等待需约{:.6f}ms；需要测量这个等待是否真的出现。该输入的粗略全卡混合概率在不等待时为{:.2f}%，在这个目标等待下约{:.2f}%，也必须实际验证。".format(required_hot_w,100*fleet_mix_probability(0),100*fleet_mix_probability(required_hot_w)), "",
        "目前更值得优先检验的是透明外推10K短请求，见 [constructed_candidates.md](constructed_candidates.md)。它改变V/C关系，使达到U80所需的短层平均等待降到现有测量的数量级；这提供新的数学动机。计算时间必须标为拟合构造，极小训练残差不代表10K实测准确；新读取量也会改变等待，因此仍不是成功保证。", "",
        "能证明的下界有明确条件：同一完整有限输入、同一个[0,makespan]分母，读取和计算全部守恒，则SSD忙率=rho_population×U。如果rho<=1.05且该全程SSD忙率>=90%，U必然>=85.714%。但目前没有证明所有合法独立随机输入的盘忙率都至少90%，所以不能据此宣称普遍不存在低U。中间warm/long窗推进的类别与在途字节不同，也不能把全批rho直接套进去。", "",
        "所有新增提议尚未在本脚本运行，原始种子结果全部保留。脚本：[math_followup.py](math_followup.py)；精确原始核对、源hash和候选参数：[math_followup.json](math_followup.json)。", ""]
    (HERE/"math_followup.md").write_text("\n".join(lines))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    print(json.dumps(dict(audited_main_cases=len(main_cases),all_raw_comparison_checks_passed=True,
        preserved_files_unchanged=True,groups=[dict(candidate=g["candidate"],window=g["window_ms"],meanU=g["U_percent"]["mean"]) for g in groups])))


if __name__=="__main__":
    main()
