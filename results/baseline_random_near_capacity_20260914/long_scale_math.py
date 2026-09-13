#!/usr/bin/env python3
"""Four preregistered absolute-layer-scale controls: mathematical plan only."""
import ast
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from math_followup_constructed import affine, stationary_coverage, ideal_clock_coverage

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BLOCK_GIB=176/1048576


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choose(vl,cl,vs,cs,disks):
    ratios=[]
    for nl in range(1,65):
        for ns in range(1,1025):
            if math.gcd(nl,ns)!=1:continue
            rho=32*(nl*vl+ns*vs)/(40*disks*(nl*cl+ns*cs)/1000)
            if .95<=rho<=1.05:
                err=abs(rho-1)
                rank=(0,nl+ns,err,nl,ns) if err<=.005 else (1,err,nl+ns,nl,ns)
                ratios.append((rank,nl,ns,rho))
    return min(ratios)[1:] if ratios else None


def main():
    protected=[HERE/n for n in ('constructed_candidates.json','math_followup_constructed.py',
        'experiment.py','study_plan.json','topology_candidates.json','miss64_math.json')]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    raw=ast.literal_eval((ROOT/'data').read_text())
    source_data_hash=sha(ROOT/'data')
    short_plan=json.loads((HERE/'constructed_candidates.json').read_text())
    short_fit=short_plan['fit']
    fits={}
    for miss in (512,1024,2048):
        rows=sorted((k[0],r[1]/1000) for k,r in raw.items() if k[1]==miss)
        a,b=affine(rows);res=[y-a-b*x for x,y in rows]
        fits[str(miss)]=dict(fixed_miss=miss,intercept_ms=a,slope_ms_per_K=b,
            source_total_K_range=[rows[0][0],rows[-1][0]],
            source_points=[dict(total_K=x,C_ms=y) for x,y in rows],
            max_abs_training_residual_ms=max(map(abs,res)),
            rmse_ms=math.sqrt(statistics.mean(e*e for e in res)),
            source_data_sha256=source_data_hash,
            warning='A near-linear fit inside32K..200K does not validate a model, KV layout, or hardware at256K..512K.')

    def derive(K,miss,sk,disks,exact_B=None,raw_long=False):
        f=fits[str(miss)]
        affine_c_ms=f['intercept_ms']+f['slope_ms_per_K']*K
        blocks=(K*1024-miss)//128
        assert blocks*128==K*1024-miss
        vl=blocks*BLOCK_GIB
        cl=raw[K,miss][1]/1000 if raw_long else affine_c_ms
        if exact_B is not None:cl=vl/exact_B*1000
        cs=short_fit['intercept_ms']+short_fit['slope_ms_per_K']*sk
        vs=((sk*1024-128)//128)*BLOCK_GIB
        bl=vl/(cl/1000);bs=vs/(cs/1000)
        result=choose(vl,cl,vs,cs,disks)
        if result is None:
            return dict(num_ssu=disks,long_total_k=K,long_nql=miss,short_total_k=sk,
                feasible=False,reason='No coprime counts nL<=64,nS<=1024 with .95<=rho<=1.05',
                long_B_GiB_s=bl,short_B_GiB_s=bs)
        nl,ns,rho=result
        total_c=nl*cl+ns*cs;frac=ns*cs/total_c
        repeats=math.ceil(22000/(8*total_c))
        w80=.25*cs/frac
        name=f'context8_L{K}m{miss}_S{sk}m128' if disks==8 else f'context{disks}_L{K}m{miss}_S{sk}m128'
        if exact_B is not None:name+='_matchedB'
        lm=dict(method='direct_data_row' if raw_long else 'affine_extrapolation_above_raw_maximum',
            fixed_miss=miss,intercept_ms=f['intercept_ms'],slope_ms_per_K=f['slope_ms_per_K'],
            raw_support_total_K_range=f['source_total_K_range'])
        if raw_long:lm['raw_key']=[K,miss]
        if exact_B is not None:
            lm.update(method='matched_B_sensitivity_control_not_direct_affine_fit',
                B_anchor_GiB_s=exact_B,C_rule='C_ms=exact_V_GiB/B_anchor_GiB_s*1000')
        return dict(name=name,num_npu=32,num_ssu=disks,long_total_k=K,long_nql=miss,
            long_compute_us=cl*1000,long_is_raw=raw_long,
            long_V_GiB=vl,long_B_GiB_s=bl,
            short_total_k=sk,short_nql=128,short_compute_us=cs*1000,
            short_is_raw=False,short_V_GiB=vs,short_B_GiB_s=bs,
            counts=[nl,ns],roles=['L','S'],feasible=True,rho_ideal=rho,f_short=frac,
            long_to_short_V_ratio=vl/vs,long_to_short_C_ratio=cl/cs,
            long_request_pure_compute_ms=8*cl,long_request_under_1s=8*cl<1000,
            L_balanced_storage_layer_service_ms=vl/(40*disks)*1000,
            short_first_link_start_delay_budget_ms=cs-vs/50*1000,
            U80_required_all_S_mean_stall_ms=w80,
            minimum_22s_population=dict(L=nl*repeats,S=ns*repeats,repeats=repeats,
                pure_compute_ms_per_card=8*total_c*repeats,
                expected_blocks_all_32=32*8*repeats*(nl*blocks+ns*((sk*1024-128)//128))),
            workload_weighted_layer_service_second_to_first_moment_ms=(nl*vl*vl+ns*vs*vs)/(40*disks*(nl*vl+ns*vs))*1000,
            long_byte_fraction=nl*vl/(nl*vl+ns*vs),
            affine_fit_C_ms=affine_c_ms,
            chosen_C_relative_to_affine_fit=cl/affine_c_ms-1,
            stationary_no_stall_mixed_risk=stationary_coverage(cl,cs,nl,ns),
            stationary_constant_U80_wait_mixed_risk=stationary_coverage(cl,cs,nl,ns,w80),
            pure_compute_clock_warm_coverage=ideal_clock_coverage(cl,cs,nl,ns,repeats),
            model_provenance=dict(source_data_sha256=source_data_hash,data_sha256=source_data_hash,
                long_compute_model=lm,
                short_compute_model=dict(method='affine_extrapolation_below_raw_minimum',fixed_miss=128,
                    intercept_ms=short_fit['intercept_ms'],slope_ms_per_K=short_fit['slope_ms_per_K'],
                    raw_support_total_K_range=short_fit['source_total_k_range'],
                    frozen_source='constructed_candidates.json',frozen_source_sha256=sha(HERE/'constructed_candidates.json')),
                V_model='Exact128-token cached blocks at176KiB per block; no tail and no padding',
                long_source_ttft_ms=raw[K,miss][2] if raw_long else None,
                short_source_ttft_ms=None,
                context_support='256K/384K/512K context support is an unverified stress-test assumption; not a claim that a real deployment supports these lengths.'),
            no_simulation_run=True)

    selected=[derive(K,1024,10,8,raw_long=K==200) for K in (200,256,384,512)]
    anchor=selected[0]
    for c in selected:
        c['long_B_relative_change_vs_raw200']=c['long_B_GiB_s']/anchor['long_B_GiB_s']-1
        c['long_V_relative_scale_vs_raw200']=c['long_V_GiB']/anchor['long_V_GiB']
        c['long_C_relative_scale_vs_raw200']=c['long_compute_us']/anchor['long_compute_us']
    matched=derive(512,1024,10,8,exact_B=anchor['long_B_GiB_s'])
    matched['optional_only_not_in_selected_candidates']=True
    # Scan fixed requested ranges to document why8SSU/miss1024 was chosen.
    scan=[derive(K,miss,sk,disks) for K,miss,sk,disks in
          itertools.product((256,384,512),(512,1024,2048),(10,12),(8,12,16))]
    for r in scan:
        r.pop('pure_compute_clock_warm_coverage',None)
    doc=dict(schema_version='long-layer-absolute-scale-controls-v1',
        status='Mathematical pressure-test plan; no new simulation or manifest generated',
        input_family='long_and_short_context_extrapolation',fits_by_fixed_miss=fits,
        selected_candidates=selected,optional_matched_B_control=matched,screening_scan=scan,
        selection='8SSU, frozen S10K/miss128; L200/256/384/512K fixed miss1024. L200 uses the raw row; larger L uses fixed-miss affine extrapolation. Simple ratios independently chosen near rho1.',
        integer_ratio_rule='Coprime nL1..64,nS1..1024; within |rho-1|<=.005 choose smallest nL+nS, then closest rho; else closest within.95..1.05.',
        protocol=dict(num_npu=32,n_layers=8,per_ssu_GiB_s=40,npu_link_GiB_s=50,
            random='Independent full queue shuffle per NPU with seed+100003*npu; no ordered queues or common decks',
            minimum_pure_compute_ms=22000,pilot_seed=7,confirmation_seeds=[19,43,67,101],
            warm_ms=[2000,4000],long_ms=[2000,20000]),
        prior_mathematics=dict(
            first_moment='At fixedBL andshort profile, adjust counts to keep pure-compute mixture f and idealrho almost fixed.',
            scale='L V andC grow together; individual long serviceVL/capacity grows while mean long bandwidthV/C changes little.',
            variance_motivation='In a stationary ideal-time picture with fixedf, long layer releases per second fall as1/C_L. Their mean byte rate stays roughly fixed, but release_rate*V_L^2 grows with the long scale. This is a burst-variance heuristic, not a Poisson/atomic-layer assertion about the actual simulator.',
            required_wait='U~1/(1+f*w/CS); U80 requires all-short-layer meanw=.25*CS/f. Larger service scale matters only if sufficient residual FIFO work is actually ahead of short reads.',
            same_B_control='The four selected fits keepBL approximately, not exactly, fixed. The optional512K control sets C=V/BL_raw200 exactly and separately labels this intentional departure from the affine fit.'),
        limitations=[
            'All short profiles are extrapolated below raw32K. Long256/384/512K profiles are extrapolated above raw200K. Only long200K is directdata.',
            'Tiny in-range fit residuals do not establish real512K context support, model accuracy, memory feasibility, attention cost, KV block bytes, or compute-kernel behavior.',
            'Changing counts to hold meanrho also changes short-run lengths and long-request frequency. The test studies absolute service scale under a fixed mean-load design, not only a scheduler parameter.',
            'Raw200 toaffine512 longB rises by about3.7%; it is not an exact sameB experiment. The optional matchedB case can quantify that residual confound.',
            'Each chosen long request is shorter than1s of purecompute, but this does not guarantee every card computes both roles in the actual2s warm window. Retain all fixedseeds and report actual coverage.',
            '12/16SSU can require near-all short compute, making long requests rare in a2s window; miss2048 at384/512K exceeds1s perrequest. They are screened, not prioritized.',
            'A long-layer balanced service time exceeding the target waiting scale is not proof that such a layer is pending ahead. Actual storage is176KiB blockwise and streams into perNPU50GiB/s links.',
            'No claim of guaranteed lowBaselineU or improvement byOnce. Nominalrho near1 is not per-disk per-instant underload.'],
        preserved_files_sha256=before,generator_sha256=sha(Path(__file__)))
    (HERE/'long_scale_math.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    lines=['# 长层绝对规模对照：平均B接近，单次读取更大','',
        '这是一个假设压力实验。并不主张实际模型支持256K、384K或512K上下文，也不把外推数据称作生产实证。只改变请求数学规模，不改仿真内核；本脚本没有生成输入或运行仿真。','',
        '优先四组都用32 NPU、8 SSU×40GiB/s、50GiB/s单卡链路和8层。短请求固定为此前冻结的10K/miss128外推：C≈0.716653ms、V≈0.013259888GiB、B≈18.5025GiB/s。长请求固定miss1024，总长度依次200K、256K、384K、512K。只有200K长请求直接来自data，其余长C来自固定miss1024拟合。','',
        '```text',f"C_L_ms(K) = {fits['1024']['intercept_ms']:.15f} + {fits['1024']['slope_ms_per_K']:.17f}*K",
        'V_L_GiB(K) = ((K*1024-1024)/128)*176/1048576','```','',
        f"拟合使用原始32K—200K的12个点，最大训练残差{fits['1024']['max_abs_training_residual_ms']:.3g}ms。200K控制组仍使用原始C，不拿拟合值冒充原始值。区间内残差小，不代表区间外硬件计算或注意力实现仍遵循同一模型。",'',
        '| 长总K | 单层C ms | 单层V GiB | B_L GiB/s | 单长层盘侧服务 ms | 整长请求纯计算 ms |',
        '|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        lines.append(f"| {c['long_total_k']} | {c['long_compute_us']/1000:.6f} | {c['long_V_GiB']:.9f} | {c['long_B_GiB_s']:.6f} | {c['L_balanced_storage_layer_service_ms']:.6f} | {c['long_request_pure_compute_ms']:.3f} |")
    lines += ['',f"从200K到512K，B_L只升高{100*selected[-1]['long_B_relative_change_vs_raw200']:.2f}%，而单层盘服务尺度放大{selected[-1]['long_V_relative_scale_vs_raw200']:.3f}倍。四组整长请求纯计算均不到1秒，有利于两秒内混合覆盖；仍须实际逐卡检查，不能保证。",'',
        '每个候选重新选择简单整数比例，使理想平均rho接近1：','','```text',
        'rho = 32*(nL*VL+nS*VS)/(320*(nL*CL+nS*CS))',
        'f_S = nS*CS/(nL*CL+nS*CS)',
        'U_approx = 1/(1+f_S*w/CS)',
        'U80需要w = .25*CS/f_S','```','',
        '| 名称 | L:S | rho | 短计算权重 | U80所需平均短等待 ms | 每卡L/S数量 |',
        '|---|---:|---:|---:|---:|---:|']
    for c in selected:
        p=c['minimum_22s_population']
        lines.append(f"| `{c['name']}` | {c['counts'][0]}:{c['counts'][1]} | {c['rho_ideal']:.6f} | {100*c['f_short']:.3f}% | {c['U80_required_all_S_mean_stall_ms']:.6f} | {p['L']}/{p['S']} |")
    lines += ['', 'w是包含零等待层的短层平均stall。近似要求长层不等待、完成比例稳定，并忽略首层、跨请求和窗口边界；这里先算需要多大等待，不预言真实会等这么久。','',
        '数学动机比只增盘更直接：平均需求接近不变，长读取却更大、更少发生；若纯计算比例固定，长层释放频率约按1/C_L下降，平均字节率仍近似固定，但“释放频率×V_L²”增大。这说明突发方差可能变大。实际释放受闭环依赖影响，盘按176KiB块交错服务，所以不能把它直接当作Poisson到达或整层原子FIFO的排队证明。必须量到短层释放时前面的残余块，以及IO-ready实际晚于计算截止多少。','',
        '同时也有反作用：长层预取截止更宽、长层释放更稀，独立随机可能再次把这些大读取错开。因此该实验有清楚的对照目的，但不保证Baseline显著降低，更不保证Once必然改善。','',
        '严格“同B”另留一个数学敏感性控制，没有放入四组启动清单：512K长C设为V_512/B_L(raw200)，约{:.6f}ms，恰好保持B_L={:.6f}GiB/s。该C比512K直接仿射外推值高{:.3f}%，必须标成有意构造的matched-B控制，而不是原始数据或直接拟合结果。'.format(matched['long_compute_us']/1000,matched['long_B_GiB_s'],100*matched['chosen_C_relative_to_affine_fit']),'',
        '筛选范围还检查了miss512/1024/2048、长256/384/512K、短10/12K及8/12/16盘。优先8盘miss1024，是因为miss512在8盘下单长类B已高于每卡容量份额，不能靠混入更高B的短类得到rho≈1；miss2048在384/512K的一条长请求计算已超过1秒。12/16盘的一些方案需要极大的短数量比例，长请求变稀，warm更容易漏长类；16盘配10K短请求甚至无法达到rho≥0.95（最高B不足容量/卡）。完整可行性与覆盖风险保留在JSON，没有偷偷丢弃。','',
        '| 长总K | 无等待时32卡两秒都混合的粗略概率 | 代入U80等待时同一粗略概率 |',
        '|---:|---:|---:|']
    for c in selected:
        lines.append(f"| {c['long_total_k']} | {100*c['stationary_no_stall_mixed_risk']['all_32_mixed_if_cards_independent']:.3f}% | {100*c['stationary_constant_U80_wait_mixed_risk']['all_32_mixed_if_cards_independent']:.3f}% |")
    lines += ['', '这些概率只来自固定时长、平稳独立标签的简化模型，不是共享盘反馈下真实warm通过概率。固定seed为7、19、43、67、101；实际失败种子仍保留。JSON也记录只累加纯计算时钟的五种子覆盖诊断，不替代仿真。','',
        '主要外推风险还包括：模型最大上下文、位置编码、NPU/HBM容纳能力、注意力随上下文长度增长的耗时、分块或分片方式、每token KV字节及kernel切换。当前数学压力实验把这些保持为同一模型假设，现实可行性未验证。','',
        '参数与来源：[long_scale_math.json](long_scale_math.json)。复算：[long_scale_math.py](long_scale_math.py)。冻结输入、拟合短C和runner均未修改。','']
    (HERE/'long_scale_math.md').write_text('\n'.join(lines))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    print(json.dumps(dict(selected=len(selected),optional_matchedB=1,screened=len(scan),
        simulations=0,protected_files_unchanged=True)))
    for c in selected:
        print(c['name'],c['counts'],c['rho_ideal'],c['f_short'],c['U80_required_all_S_mean_stall_ms'])


if __name__=='__main__':
    main()
