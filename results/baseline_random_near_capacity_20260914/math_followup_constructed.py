#!/usr/bin/env python3
"""Transparent below-range data extrapolation; never invokes the simulator."""
import ast
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (7, 19, 43, 67, 101)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def affine(rows):
    mx = statistics.mean(x for x,y in rows)
    my = statistics.mean(y for x,y in rows)
    slope = math.fsum((x-mx)*(y-my) for x,y in rows) / math.fsum((x-mx)**2 for x,y in rows)
    intercept = my-slope*mx
    return intercept, slope


def stationary_coverage(cl_ms, cs_ms, nl, ns, w_ms=0.):
    """Toy infinite-iid renewal approximation, not real window probability."""
    a, b = 8*cl_ms, 8*(cs_ms+w_ms)
    p, q = nl/(nl+ns), ns/(nl+ns)
    mean_request = p*a + q*b
    def excess(duration, continuation_probability):
        z = 2000/duration
        k = math.floor(z)+1
        r = continuation_probability
        return duration*r**k*(k-z+r/(1-r))
    no_l = p*excess(b,q)/mean_request
    no_s = q*excess(a,p)/mean_request
    return dict(no_L_per_card=no_l, no_S_per_card=no_s,
                all_32_mixed_if_cards_independent=(1-no_l-no_s)**32,
                union_bound_all_32_mixed=max(0.,1-32*(no_l+no_s)),
                assumed_mean_S_stall_ms=w_ms,
                assumptions="Infinite iid request labels; stationary observation time; deterministic request durations 8*CL and 8*(CS+w); no storage coupling. This is a risk indicator, not an actual warm pass probability.")


def ideal_clock_coverage(cl_ms, cs_ms, nl, ns, repeats):
    """Only add raw/constructed C after each independent full-queue shuffle."""
    output=[]
    for seed in SEEDS:
        per_npu=[]
        for npu in range(32):
            order=[0]*(nl*repeats)+[1]*(ns*repeats)
            random.Random(seed+100003*npu).shuffle(order)
            t=0.;seen=set()
            for role in order:
                end=t+8*(cl_ms if role==0 else cs_ms)
                if min(end,4000)>max(t,2000): seen.add(role)
                t=end
                if t>=4000:break
            per_npu.append(len(seen)==2)
        output.append(dict(seed=seed,mixed_cards=sum(per_npu),missing_cards=[i for i,x in enumerate(per_npu) if not x]))
    return output


def main():
    protected=[HERE/name for name in ("experiment.py","study_plan.json","candidate_math.json",
                "adaptive_candidates.json","topology_candidates.json","topology_s128_appendix.json")]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    data=ast.literal_eval((ROOT/"data").read_text())
    support=sorted((k[0],row[1]/1000) for k,row in data.items() if k[1]==128)
    intercept,slope=affine(support)
    residuals=[y-intercept-slope*x for x,y in support]
    low_intercept,low_slope=affine(support[:4])
    loo=[]
    for i,(x,y) in enumerate(support):
        a,b=affine(support[:i]+support[i+1:]);loo.append(y-a-b*x)
    for key,row in data.items():
        if key[1]==128:
            assert math.isclose(row[3],(key[0]*1024-128)/128*176/1024**2,rel_tol=1e-12)
    fit=dict(method="ordinary_least_squares_affine_C_at_fixed_miss128",
        C_ms_formula="intercept_ms + slope_ms_per_K * total_input_K",
        intercept_ms=intercept,slope_ms_per_K=slope,
        source_data_sha256=sha(ROOT/"data"),source_points=[dict(total_k=x,C_ms=y) for x,y in support],
        source_total_k_range=[min(x for x,y in support),max(x for x,y in support)],
        max_abs_training_residual_ms=max(map(abs,residuals)),
        training_rmse_ms=math.sqrt(statistics.mean(r*r for r in residuals)),
        leave_one_out_max_abs_residual_ms=max(map(abs,loo)),
        leave_one_out_rmse_ms=math.sqrt(statistics.mean(r*r for r in loo)),
        lowest_four_point_fit=dict(intercept_ms=low_intercept,slope_ms_per_K=low_slope),
        V_GiB_formula="((total_K*1024-128)/128) * (176/1048576)",
        V_rule="One cached 128-token block occupies 176 KiB. Exact integer KV-block construction; verified against every fixed-miss128 raw row.",
        limitation="The requested totals10/12/16K are below the raw minimum32K. Tiny in-range residuals do not establish extrapolation accuracy or a hardware measurement at the new lengths.")

    calibrations={}
    for disks,lmiss in itertools.product((6,8),(2048,4096)):
        name=f"topo{disks}_m{lmiss}_s32m128"
        path=HERE/f"runs/{name}_ssu{disks}_h22000_seed7/baseline/analysis.json"
        if not path.exists():continue
        doc=json.loads(path.read_text())
        win=next(w for w in doc["windows"] if (w["start_ms"],w["end_ms"])==(2000,20000))
        raw_s=win["complete_internal_cycles_inside_window"]["per_role"]["S"]
        w=raw_s["stall_ms_including_zeros"]["mean"]
        calibrations[(disks,lmiss)]=dict(source=str(path.relative_to(ROOT)),source_sha256=sha(path),
            actual_long_U_percent=win["U_percent"],mean_S_internal_stall_ms_including_zeros=w,
            kappa=w/(data[200,lmiss][3]/(40*disks)*1000),
            all_32_long_mixed=win["long_short_mixed_card_count"]==32,
            assumption_for_new_candidate="Only a sensitivity anchor: retain this raw32K short-layer mean wait while replacing short V,C and request counts. New queues can change the wait substantially.")

    selected_keys={(d,l,10) for d,l in itertools.product((6,8),(2048,4096))}
    selected_keys.update((d,4096,s) for d,s in itertools.product((6,8),(12,16)))
    all_rows=[]
    for disks,lmiss,sk in itertools.product((6,8),(2048,4096),(10,12,16)):
        L=data[200,lmiss];cl=L[1]/1000;vl=L[3];cs=intercept+slope*sk
        blocks=(sk*1024-128)//128;vs=blocks*176/1024**2;bs=vs/(cs/1000)
        assert bs<50 and blocks*128==sk*1024-128
        ratios=[]
        for nl in range(1,65):
            for ns in range(1,1025):
                if math.gcd(nl,ns)!=1:continue
                rho=32*(nl*vl+ns*vs)/(40*disks*(nl*cl+ns*cs)/1000)
                if .95<=rho<=1.05:
                    error=abs(rho-1)
                    rank=(0,nl+ns,error,nl,ns) if error<=.005 else (1,error,nl+ns,nl,ns)
                    ratios.append((rank,nl,ns,rho))
        _,nl,ns,rho=min(ratios)
        f=ns*cs/(nl*cl+ns*cs);rep=math.ceil(22000/(8*(nl*cl+ns*cs)))
        name=f"fit{disks}_m{lmiss}_s{sk}m128"
        calibration=calibrations.get((disks,lmiss))
        coverage=stationary_coverage(cl,cs,nl,ns)
        w80=.25*cs/f
        prediction=None
        if calibration:
            predicted_w=calibration["mean_S_internal_stall_ms_including_zeros"]
            prediction=dict(calibration=calibration,
                hypothetical_mean_S_stall_ms=predicted_w,
                approximate_U_percent=100/(1+f*predicted_w/cs),
                mixed_risk_under_same_constant_wait=stationary_coverage(cl,cs,nl,ns,predicted_w),
                not_a_measured_new_result=True)
        row=dict(name=name,candidate_id=name,selected=(disks,lmiss,sk) in selected_keys,
            num_npu=32,num_ssu=disks,per_ssu_GiB_s=40,npu_link_GiB_s=50,n_layers=8,
            long_total_k=200,long_nql=lmiss,short_total_k=sk,short_nql=128,
            long_compute_us=L[1],long_V_GiB=vl,long_B_GiB_s=L[0],
            short_compute_us=cs*1000,short_V_GiB=vs,short_B_GiB_s=bs,
            short_blocks_per_layer=blocks,
            short_constructed_profile=True,
            short_construction="affine_extrapolation_below_raw_minimum",
            counts=[nl,ns],roles=["L","S"],rho_ideal=rho,f_short=f,
            per_card_population=dict(L=nl*rep,S=ns*rep,repeats=rep,
                pure_compute_ms=8*(nl*cl+ns*cs)*rep,
                total_blocks_all_32_npu=32*8*rep*(nl*((200*1024-lmiss)//128)+ns*blocks)),
            C_short_ms=cs,L_request_pure_compute_ms=8*cl,
            mean_pure_compute_work_per_L_request_ms=8*(nl*cl+ns*cs)/nl,
            L_balanced_storage_layer_service_ms=vl/(disks*40)*1000,
            short_link_minimum_ms=vs/50*1000,
            first_link_start_delay_threshold_ms=cs-vs/50*1000,
            required_mean_S_stall_ms_for_U80=w80,
            model_provenance=dict(fit_reference="top_level.fit",data_sha256=fit["source_data_sha256"],
                long_source="direct_data_row",short_source="affine_extrapolation_below_raw_minimum",
                original_short_total_k_not_in_data=sk<fit["source_total_k_range"][0],
                source_ttft_for_short=None,
                low_four_point_fit_C_ms=low_intercept+low_slope*sk,
                all_point_vs_low_four_prediction_difference_ms=cs-(low_intercept+low_slope*sk)),
            stationary_no_stall_coverage_risk=coverage,
            stationary_constant_U80_wait_coverage_risk=stationary_coverage(cl,cs,nl,ns,w80),
            pure_compute_clock_warm_coverage_by_seed=ideal_clock_coverage(cl,cs,nl,ns,rep),
            same_topology_raw_wait_sensitivity=prediction,
            fixed_count_fit_error_sensitivity=[dict(relative_short_C_error=delta,
                resulting_rho_if_counts_unchanged=rho/(1+f*delta),
                resulting_short_B_GiB_s=bs/(1+delta)) for delta in (-.25,-.1,.1,.25)])
        all_rows.append(row)
    chosen=[r for r in all_rows if r["selected"]]
    assert len(chosen)==8
    doc=dict(schema_version="constructed-affine128-candidates-v1",family="data_affine_extrapolation",
        status="Adaptive mathematical candidates, not simulated by this builder",
        no_new_simulations=True,fit=fit,
        protocol=dict(pilot_seed=7,confirmation_seeds=[19,43,67,101],minimum_pure_compute_ms=22000,
            windows_ms=[[2000,4000],[2000,20000]],
            random="Every card independently shuffles its entire fixed-count queue; seed+100003*npu; no synchronized decks or seed replacement"),
        integer_ratio_selection="Coprime nL<=64,nS<=1024. Within abs(rho-1)<=.005 choose minimum count sum, then minimum error; if unavailable choose closest among rho .95..1.05.",
        selection="Eight requested: S10K with both L200/2048 and L200/4096 at6/8 disks; S12K/S16K with L200/4096 at6/8 disks. Full12-cell math scan is retained.",
        source_calibrations=list(calibrations.values()),
        selected_candidates=chosen,all_mathematical_candidates=all_rows,
        warnings=[
            "Fitted short C is constructed, never direct_data_row or a raw10/12/16K measurement.",
            "Smaller total length lowers both V and C but not proportionally: a positive fitted intercept lowers short B. Near-capacity balancing therefore increases short pure-compute share.",
            "The new short V/C/counts alter queue traffic. Retaining observed raw32K short mean wait is a conditional sensitivity, not a validated prediction.",
            "Uniform full finite shuffles are not stationary Bernoulli streams; real waits are variable and cards are storage-coupled. Coverage probabilities and compute-only clocks are diagnostics; real per-card warm compute overlap is mandatory.",
            "Long141ms layers make1.13s requests. Candidate coverage failures must be retained and reported; a good long window does not repair failed2s warm mixed coverage.",
            "Whole-population rho near1 does not enforce per-disk per-instant underload. Fit uncertainty can move fixed-count cases out of the near-capacity range.",
            "U approximation assumes long layers unstalled, stable completion mix, and ignores startup/request-boundary/window-clipping residuals; do not claim Once improvement before paired execution."],
        formulas=dict(U_proxy="1/(1+f_short*w/C_short), with mean w over ALL short layers including zero waits",
            required_w_U80=".25*C_short/f_short",
            same_full_population_conservation="SSD_busy=rho_ideal*U only over same complete batch and time denominator; middle windows must be measured separately"),
        preserved_files_sha256=before,generator_sha256=sha(Path(__file__)))
    (HERE/"constructed_candidates.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")

    lines=["# 10K / 12K / 16K 短请求：明确标注的数据外推", "",
        "这些短请求不是 data 的原始行。总长度低于 data 的最小32K；计算时间由固定 miss=128 的12个数据点作线性拟合后外推，读取量由128 token一块、每块176 KiB直接构造。长请求仍采用原始200K数据。没有修改仿真内核，也没有在本脚本运行仿真。", "",
        "```text", f"C_short_ms(K) = {intercept:.15f} + {slope:.17f} * K",
        "V_short_GiB = ((K*1024-128)/128) * 176/1048576", "```", "",
        f"训练区间为32K—200K，固定miss=128。最大训练残差 {fit['max_abs_training_residual_ms']:.3g} ms，留一法最大残差 {fit['leave_one_out_max_abs_residual_ms']:.3g} ms。极小残差只说明已有数据非常接近直线，不能证明10K硬件耗时也符合外推。短请求原始TTFT留空，不能伪造一个data来源值。", "",
        "| 总长度 | miss | 一层 C ms | 一层 V GiB | B=V/C GiB/s |", "|---:|---:|---:|---:|---:|"]
    for sk in (10,12,16):
        r=next(r for r in all_rows if r["short_total_k"]==sk)
        lines.append(f"| {sk}K | 128 | {r['C_short_ms']:.9f} | {r['short_V_GiB']:.12f} | {r['short_B_GiB_s']:.6f} |")
    lines += ["", "为什么比只取32K更值得尝试：拟合式有约0.507ms的截距，所以缩短总长时，V下降得比C快。短类B降低，在相同盘容量附近可以让更多纯计算时间属于短类。此时同样0.5—0.8ms的平均等待，对整机U影响更大。但读取量变小也会缓解队列，所以不能保证等待保持不变。", "",
        "```text", "rho = 32*(nL*VL+nS*VS)/(SSU*40*(nL*CL+nS*CS))",
        "f_short = nS*CS/(nL*CL+nS*CS)", "U_approx = 1/(1+f_short*w/CS)",
        "若要U=80%，需要平均w=0.25*CS/f_short", "```", "",
        "w包括所有短层，含零等待；近似要求长层不等、完成比例稳定，并忽略启动和跨请求/窗口边界。下面是‘需要发生多大等待’，不是已发生的结果。", "",
        "| 名称 | 盘 | L miss | S总K | L:S | rho | 短计算权重 | U80所需w ms | 每卡L/S数量 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in chosen:
        pop=r["per_card_population"]
        lines.append(f"| `{r['name']}` | {r['num_ssu']} | {r['long_nql']} | {r['short_total_k']} | {r['counts'][0]}:{r['counts'][1]} | {r['rho_ideal']:.6f} | {100*r['f_short']:.2f}% | {r['required_mean_S_stall_ms_for_U80']:.6f} | {pop['L']}/{pop['S']} |")
    lines += ["", "原始32K拓扑结果只是条件参照。下表把相同盘数、相同长画像的实测短层平均等待原封不动放进新公式；这并未证明换短画像后等待不变。", "",
        "| 新候选 | 原始32K短层实测w ms | 若w保持则U约 | 无等待时32卡2秒都混合的粗略概率 | 达到U80等待时同一粗略概率 |", "|---|---:|---:|---:|---:|"]
    for r in chosen:
        pred=r["same_topology_raw_wait_sensitivity"]
        ws=f"{pred['hypothetical_mean_S_stall_ms']:.6f}" if pred else "尚无参照"
        us=f"{pred['approximate_U_percent']:.2f}%" if pred else "—"
        lines.append(f"| `{r['name']}` | {ws} | {us} | {100*r['stationary_no_stall_coverage_risk']['all_32_mixed_if_cards_independent']:.2f}% | {100*r['stationary_constant_U80_wait_coverage_risk']['all_32_mixed_if_cards_independent']:.2f}% |")
    lines += ["", "这两个概率来自简化的平稳Bernoulli请求序列：固定每类时长、独立卡，无共享盘反馈。实际是有限整队列打乱、所有请求同时到达、等待随时间变化，不能把这些概率当实际通过率。JSON还列出五个预注册seed仅累加纯计算时间的覆盖检查；它同样不代替真实warm统计。", "",
        "长画像200K/2048的8层纯计算约566ms，200K/4096约1131ms。后者更容易让某卡两秒窗只有长类；当短层等待变大时，又更容易让某卡两秒窗只有短类。必须同时看两种漏覆盖，不能只检查输入队列中有没有两类。", "",
        "建议先关注10K+L2048的6/8盘：等待目标降低且长请求较短。L4096和16K/12K是不同权重与外推幅度的对照，最终保留所有失败种子。若要求总长严格大于10K，优先12K行。", "",
        "需保留的限制：", "",
        "- 同一组配比在拟合C变化时rho也变：rho_new=rho/(1+f_short*相对C误差)。JSON列出±10%/±25%；近容量结论不能脱离计算模型准确性。",
        "- 旧3盘均等待按VL/容量缩放曾过度预测6/8盘损失；新参照不能再次被当成保证。",
        "- rho近1只是理想平均，不是逐盘逐时刻欠载。全程的SSD忙率=rho×U守恒式不能无条件套中间窗。",
        "- 只观察Baseline变低，不能推断Once一定有效；真实策略差距需同manifest配对。", "",
        "复算：`python results/baseline_random_near_capacity_20260914/math_followup_constructed.py`。完整12组、拟合残差、种子覆盖诊断及所有构造字段见 [constructed_candidates.json](constructed_candidates.json)。", ""]
    (HERE/"constructed_candidates.md").write_text("\n".join(lines))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    print(json.dumps(dict(selected=len(chosen),max_fit_residual_ms=fit["max_abs_training_residual_ms"],preserved_files_unchanged=True)))
    for r in chosen:
        print(r["name"],r["counts"],r["rho_ideal"],r["f_short"],r["required_mean_S_stall_ms_for_U80"])


if __name__=="__main__":
    main()
