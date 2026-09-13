#!/usr/bin/env python3
"""Unused miss64 alternatives: math only; preserve every frozen input/model."""
import ast
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from math_followup_constructed import affine, stationary_coverage

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FULL_BLOCK_GIB=176/1048576


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choose_ratio(vl,cl,vs,cs,disks):
    legal=[]
    for nl in range(1,65):
        for ns in range(1,1025):
            if math.gcd(nl,ns)!=1:continue
            rho=32*(nl*vl+ns*vs)/(40*disks*(nl*cl+ns*cs)/1000)
            if .95<=rho<=1.05:
                error=abs(rho-1)
                rank=(0,nl+ns,error,nl,ns) if error<=.005 else (1,error,nl+ns,nl,ns)
                legal.append((rank,nl,ns,rho))
    return min(legal)[1:]


def main():
    protected=[HERE/n for n in ("constructed_candidates.json","math_followup_constructed.py",
        "experiment.py","study_plan.json","topology_candidates.json","candidate_math.json")]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw=ast.literal_eval((ROOT/"data").read_text())
    fit128=json.loads((HERE/"constructed_candidates.json").read_text())["fit"]
    points=sorted((k[0],v[1]/1000) for k,v in raw.items() if k[1]==64)
    intercept,slope=affine(points)
    residual=[y-(intercept+slope*x) for x,y in points]
    loo=[]
    for i,(x,y) in enumerate(points):
        a,b=affine(points[:i]+points[i+1:]);loo.append(y-a-b*x)
    for key,row in raw.items():
        if key[1]==64:
            assert math.isclose(row[3],(key[0]*1024-64)/128*FULL_BLOCK_GIB,rel_tol=1e-12)
    fit=dict(method="ordinary_least_squares_at_fixed_miss64",
        C_ms_formula="intercept_ms + slope_ms_per_K*total_K",
        intercept_ms=intercept,slope_ms_per_K=slope,
        source_points=[dict(total_K=x,C_ms=y) for x,y in points],
        source_total_K_range=[points[0][0],points[-1][0]],
        maximum_training_abs_residual_ms=max(map(abs,residual)),
        training_rmse_ms=math.sqrt(statistics.mean(e*e for e in residual)),
        maximum_leave_one_out_abs_residual_ms=max(map(abs,loo)),
        source_data_sha256=sha(ROOT/"data"),
        warning="10/12/16K below the raw32K minimum; no measurement at the new inputs. Raw miss64 rows at32K and above have B>50, but only their C values are fit; none is proposed as an input.")

    short_profiles=[]
    for K in (10,12,16):
        c64=intercept+slope*K
        c128=fit128["intercept_ms"]+fit128["slope_ms_per_K"]*K
        hit=K*1024-64;nfull=hit//128;tail=hit%128
        assert tail==64
        v_exact=nfull*FULL_BLOCK_GIB+(tail/128)*FULL_BLOCK_GIB
        v_pad=(nfull+1)*FULL_BLOCK_GIB
        v128=(K*1024-128)//128*FULL_BLOCK_GIB
        short_profiles.append(dict(total_K=K,miss64_C_ms=c64,miss128_C_ms=c128,
            miss64_logical_hit_tokens=hit,full_blocks=nfull,tail_tokens=64,
            exact_tail_KiB=88,exact_command_count=nfull+1,padded_command_count=nfull+1,
            exact_V_GiB=v_exact,padded_V_GiB=v_pad,miss128_V_GiB=v128,
            exact_B_GiB_s=v_exact/(c64/1000),padded_B_GiB_s=v_pad/(c64/1000),
            miss128_B_GiB_s=v128/(c128/1000),padding_KiB=88,
            padding_bytes_fraction_of_exact=(v_pad-v_exact)/v_exact,
            exact_first_link_delay_budget_ms=c64-v_exact/50*1000,
            padded_first_link_delay_budget_ms=c64-v_pad/50*1000,
            miss128_first_link_delay_budget_ms=c128-v128/50*1000,
            initial_full_block_SSD_service_ms=FULL_BLOCK_GIB/40*1000,
            exact_minimum_no_contention_ready_ms=FULL_BLOCK_GIB/40*1000+v_exact/50*1000,
            padded_minimum_no_contention_ready_ms=FULL_BLOCK_GIB/40*1000+v_pad/50*1000))
    candidates=[]
    for disks,lmiss,K,mode in itertools.product((6,8),(2048,4096),(10,12,16),("exact_tail88KiB","pad_tail176KiB")):
        short=next(p for p in short_profiles if p["total_K"]==K)
        bs,cl_us,_,vl=raw[200,lmiss];cl=cl_us/1000
        cs=short["miss64_C_ms"]
        vs=short["exact_V_GiB"] if mode=="exact_tail88KiB" else short["padded_V_GiB"]
        cs128=short["miss128_C_ms"];vs128=short["miss128_V_GiB"]
        cap=40*disks;x=cap/32;Bshort=vs/(cs/1000)
        assert bs<x<Bshort<50
        nl,ns,rho=choose_ratio(vl,cl,vs,cs,disks)
        nl128,ns128,rho128=choose_ratio(vl,cl,vs128,cs128,disks)
        f=ns*cs/(nl*cl+ns*cs)
        k_exact=(x*cl/1000-vl)/(vs-x*cs/1000)
        k128=(x*cl/1000-vl)/(vs128-x*cs128/1000)
        f_exact=(x-bs)/(Bshort-bs)
        f128_exact=(x-bs)/(short["miss128_B_GiB_s"]-bs)
        w80_exact=.25*(vs-bs*cs/1000)/(x-bs)*1000
        w80_128_exact=.25*(vs128-bs*cs128/1000)/(x-bs)*1000
        rep=math.ceil(22000/(8*(nl*cl+ns*cs)))
        short_blocks=[FULL_BLOCK_GIB]*short["full_blocks"]+[
            FULL_BLOCK_GIB/2 if mode=="exact_tail88KiB" else FULL_BLOCK_GIB]
        long_blocks=[FULL_BLOCK_GIB]*((200*1024-lmiss)//128)
        assert math.isclose(math.fsum(short_blocks),vs,rel_tol=1e-14)
        disk_rates=[0.]*disks
        for npu in range(32):
            for count,blocks in ((nl,long_blocks),(ns,short_blocks)):
                for j,size in enumerate(blocks):
                    disk_rates[(j+npu)%disks]+=count*size/((nl*cl+ns*cs)/1000)
        assert math.isclose(math.fsum(disk_rates)/cap,rho,rel_tol=1e-12)
        candidates.append(dict(name=f"backup{disks}_m{lmiss}_s{K}m64_{'exact' if mode.startswith('exact') else 'pad'}",
            status="mathematical_backup_only_no_manifest_no_run",num_npu=32,num_ssu=disks,
            profiles=dict(L=[200,lmiss],S=[K,64]),mode=mode,
            physical_block_rule="Full blocks176KiB; final88KiB" if mode.startswith("exact") else "All blocks176KiB; explicitly64 padding tokens=88KiB per layer",
            short_C_ms=cs,short_V_GiB=vs,short_B_GiB_s=Bshort,
            counts=[nl,ns],rho_ideal=rho,f_short=f,required_all_S_mean_stall_ms_U80=.25*cs/f,
            exact_rho1_reference=dict(short_per_long=k_exact,f_short=f_exact,
                required_all_S_mean_stall_ms_U80=w80_exact,
                long_byte_fraction=vl/(vl+k_exact*vs),
                workload_weighted_layer_service_second_to_first_moment_ms=(vl*vl+k_exact*vs*vs)/(cap*(vl+k_exact*vs))*1000),
            miss128_exact_rho1_reference=dict(short_per_long=k128,f_short=f128_exact,
                required_all_S_mean_stall_ms_U80=w80_128_exact,
                long_byte_fraction=vl/(vl+k128*vs128),
                workload_weighted_layer_service_second_to_first_moment_ms=(vl*vl+k128*vs128*vs128)/(cap*(vl+k128*vs128))*1000),
            wait_target_ratio64_to128_at_exact_rho1=w80_exact/w80_128_exact,
            if_keep_miss128_integer_counts=dict(counts=[nl128,ns128],old_rho128=rho128,
                resulting_rho64=32*(nl128*vl+ns128*vs)/(cap*(nl128*cl+ns128*cs)/1000),
                warning="Keeping old counts after shortening C can overload the disks; this is not the balanced comparison."),
            first_link_start_delay_budget_ms=cs-vs/50*1000,
            isolated_IO_ready_lower_bound_ms=FULL_BLOCK_GIB/40*1000+vs/50*1000,
            L_balanced_storage_layer_service_ms=vl/cap*1000,
            per_ssu_ideal_rate_GiB_s=disk_rates,
            per_card_population=dict(L=nl*rep,S=ns*rep,repeats=rep,pure_compute_ms=8*(nl*cl+ns*cs)*rep),
            stationary_no_stall_warm_mixed_risk=stationary_coverage(cl,cs,nl,ns),
            stationary_constant_U80_wait_mixed_risk=stationary_coverage(cl,cs,nl,ns,.25*cs/f),
            theoretical_mechanism="Shorter C reduces deadline slack; at exact mean rho1 it also reduces short compute share and increases long-byte share. Actual wait must increase enough; no utilization outcome is guaranteed."))
    sources=[ROOT/"sim.py",ROOT/"continuous_batch_sim.py",ROOT/"shared_path_sim_adapter.py",HERE/"experiment.py"]
    doc=dict(schema_version="miss64-backup-math-v1",no_simulations=True,no_manifest_generated=True,
        status="Optional backup requested after frozen miss128 selection; not an extension of the frozen eight inputs",
        num_ssu_scope=[6,8],fit64=fit,fit128_reference=dict(file="constructed_candidates.json",sha256=sha(HERE/"constructed_candidates.json")),
        short_profiles=short_profiles,mathematical_candidates=candidates,
        comparison_proof=dict(assumptions="Exact rho1; same L; capacity per card x; BL<x<BS; C in seconds; long unstalled wait approximation",
            short_compute_fraction="f=(x-BL)/(VS/CS-BL)",
            wait_target="w80=.25*CS/f=.25*(VS-BL*CS)/(x-BL)",
            change128to64="VS increases and CS decreases, with BL>0. Hence VS-BL*CS increases; the required mean short stall for U80 increases slightly after rebalancing rho. Shorter C alone does not lower the fleet U80 wait target.",
            reason_it_might_still_hurt="First-link delay tolerance CS-VS/50 decreases; the exact-rho short count ratio falls, raising long-read byte share and a workload-size second-moment proxy. These can increase real waiting, but the blockwise coupled queue must be measured.",
            static_impossibility_not_claimed="No universal lower bound for actual Random U is proved; this only refutes an inference that halving miss automatically improves the wait threshold."),
        real_block_model=dict(
            sim_py="build_block_placement uses ceil(hit_tokens/128) blocks and min(128, remaining_tokens) for the last size, so its natural miss64 tail is88KiB.",
            continuous_batch_sim_py="Placement holds explicit(ssu_id,size_gb); submission creates each flow with total_gb=size_gb and block_count=1. A half-size tail remains one command.",
            study_builder="experiment.py input preparation currently asserts hit_tokens%128==0 and builds equal176KiB blocks. Existing core supports a precise partial tail, but a separately prepared honest manifest is needed. No frozen builder/runner was changed.",
            padding="If padding is chosen, retain logical hit=total-64 and miss=64, separately state physical_read_tokens=ceil(hit/128)*128, padding_tokens=64 and padding_gib=88KiB. Do not pretend those padding tokens are genuine cache hits.",
            equality_warning="The miss64 exact and padded variants have the same command count. Relative to miss128 each has one additional command per layer; size and command-count pressure effects must be separated, especially for any later Once comparison."),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        protected_files_sha256=before,generator_sha256=sha(Path(__file__)),
        warnings=["Constructed C remains an out-of-range fit; small residuals are not hardware validation.",
            "Tail padding adds88KiB per layer relative to exact miss64, not zero; versus miss128 total read rises176KiB for the padded variant.",
            "rho near1 is an ideal average, not a per-disk per-instant bound. Per-disk ideal means are explicitly retained.",
            "Stationary iid coverage probabilities are diagnostics, not real coupled finite-window pass probabilities. Keep the same seeds7,19,43,67,101 and report failures.",
            "No12/16-disk extension, no input generation, and no simulation were performed for this backup."])
    (HERE/"miss64_math.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")

    lines=["# miss=64 的备用数学分析：不运行、不改冻结输入", "",
        "miss从128降到64，短计算会更短，确实更容易错过单层预取截止。但在重新调整请求比例、保持理想平均rho=1后，达到整机U=80%所需的平均短层等待反而略升。因此不能只凭‘计算更短’断言这组更差。它值得作为可选机制验证，优先级取决于现有miss128实测。", "",
        "计算拟合仍明确是外推：", "", "```text",
        f"C64_ms(K) = {intercept:.15f} + {slope:.17f} * K", "```", "",
        f"拟合来源为data中固定miss64的12个32K—200K数据点，最大残差{fit['maximum_training_abs_residual_ms']:.3g}ms；10K/12K/16K均低于原始区间。这不是这些新长度的测量。原始32K以上miss64行的B均超过50，不能直接作为当前50GiB/s链路下的无单卡过载输入；这里仅用其C拟合，外推的新短画像均低于50。", "",
        "| 总K | C64 ms | 精确V GiB | 精确B GiB/s | 补齐V GiB | 补齐B GiB/s |", "|---:|---:|---:|---:|---:|---:|"]
    for s in short_profiles:
        lines.append(f"| {s['total_K']} | {s['miss64_C_ms']:.9f} | {s['exact_V_GiB']:.12f} | {s['exact_B_GiB_s']:.6f} | {s['padded_V_GiB']:.12f} | {s['padded_B_GiB_s']:.6f} |")
    lines += ["", "命中数为total_tokens−64，总会留下64 token尾块。真实内核支持两种物理清单：", "",
        "- **精确尾块**：10K为79个176KiB块加1个88KiB块；12K为95+半块；16K为127+半块。这与sim.py的ceil分块与最后块min剩余token规则一致。",
        "- **显式补齐**：尾块也读176KiB，每层比精确方案多88KiB。逻辑miss仍64、逻辑命中仍total−64；额外64token对应读取对齐开销，必须记录为padding，不能伪称真命中。",
        "两种方式的命令数相同，都是80/96/128；比miss128各多一条命令。补齐相对精确方案额外字节约0.629%/0.524%/0.392%。physical V和B必须使用真实读取总量。当前研究输入生成器要求整128token，需另做明确标注的manifest；内核本身无须修改。本文件没有生成manifest。", "",
        "链路余量比较：", "",
        "| 总K | miss128首链路启动余量 ms | miss64精确 ms | miss64补齐 ms |", "|---:|---:|---:|---:|"]
    for s in short_profiles:
        lines.append(f"| {s['total_K']} | {s['miss128_first_link_delay_budget_ms']:.9f} | {s['exact_first_link_delay_budget_ms']:.9f} | {s['padded_first_link_delay_budget_ms']:.9f} |")
    lines += ["", "余量=C−V/50：如果首块进入NPU链路已经晚于这个时间，即使后续连续满速接收也来不及。无竞争时还至少先经过一个176KiB块的盘服务，约0.004196ms；JSON也记录这个无竞争读取下界。不能把整层盘服务和整层链路服务相加，因为它们流水重叠。", "",
        "为什么整机等待门槛并没有因短C而降低：", "", "```text", "x = SSU总容量 / 32",
        "rho=1时：f_short = (x-B_L)/(V_S/C_S-B_L)",
        "U≈1/(1+f_short*w/C_S)",
        "U80所需w = .25*C_S/f_short", "          = .25*(V_S-B_L*C_S)/(x-B_L)", "```", "",
        "保持长画像与盘数不变，miss64让V略增、C下降，所以V−B_L*C变大，w80门槛略升。配比调整减少了短请求占据的纯计算时间，抵消了‘短C更容易受伤’的直觉。如果保留miss128的旧数量比例不调，rho会提高；这会混入带宽过载，JSON单独列出这一反事实。", "",
        "下面均为精确尾块方案，整数配比重新取rho约1。所有短层平均w包括零等待层；长层不等、稳定完成比例、忽略边界才适用此近似。", "",
        "| 盘 / L miss / S总K | L:S | rho | 短计算权重 | U80所需w ms | 相对miss128目标(严格rho=1) | 每卡L/S |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in candidates:
        if r["mode"]!="exact_tail88KiB":continue
        pop=r["per_card_population"]
        lines.append(f"| {r['num_ssu']} / {r['profiles']['L'][1]} / {r['profiles']['S'][0]} | {r['counts'][0]}:{r['counts'][1]} | {r['rho_ideal']:.6f} | {100*r['f_short']:.2f}% | {r['required_all_S_mean_stall_ms_U80']:.6f} | {r['wait_target_ratio64_to128_at_exact_rho1']:.4f}× | {pop['L']}/{pop['S']} |")
    lines += ["", "补齐方案也重新调整配比，不能把额外88KiB当作零成本：", "",
        "| 盘 / L miss / S总K | 补齐L:S | rho | U80所需w ms |", "|---|---:|---:|---:|"]
    for r in candidates:
        if r["mode"]!="pad_tail176KiB":continue
        lines.append(f"| {r['num_ssu']} / {r['profiles']['L'][1]} / {r['profiles']['S'][0]} | {r['counts'][0]}:{r['counts'][1]} | {r['rho_ideal']:.6f} | {r['required_all_S_mean_stall_ms_U80']:.6f} |")
    lines += ["", "仍可能更差的机制是实际w改变：更短的截止余量让相同IO延迟露出更多stall；重新配比后大读取的字节份额更高，按层大小计算的二阶矩指标也更高。两者都可能增加短层等待，但实际是176KiB块队列、异步提交和共享盘反馈，不能把按层矩指标当成实测排队公式。", "",
        "若需要测试，先用精确88KiB尾块减少padding混杂，再以同一逻辑请求的补齐版本检验对齐成本。这里仅保存备用数学说明，不生成输入、不启动仿真。保留每卡独立全队列随机、固定种子7/19/43/67/101，并重新核验真实warm两类计算覆盖。L4096的整请求超过1秒仍有覆盖风险，降低miss并不能解决这一点。", "",
        "完整24组数学记录（仅6/8盘）、逐盘理想均值、纯22秒数量、拟合残差和风险：[miss64_math.json](miss64_math.json)。复算脚本：[miss64_math.py](miss64_math.py)。冻结8组miss128文件和runner的SHA保持不变。", ""]
    (HERE/"miss64_math.md").write_text("\n".join(lines))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    print(json.dumps(dict(backups=len(candidates),new_runs=0,preserved_files_unchanged=True,
        fit_intercept=intercept,fit_slope=slope,max_residual_ms=fit["maximum_training_abs_residual_ms"])))


if __name__=="__main__":
    main()
