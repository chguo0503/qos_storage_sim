#!/usr/bin/env python3
"""Four low-B short-compute controls. Mathematics only; no inputs or runs."""
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=[7,19,43,67,101]
BLOCK_GIB=176/1048576


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def affine(rows):
    mx=statistics.mean(x for x,y in rows);my=statistics.mean(y for x,y in rows)
    b=math.fsum((x-mx)*(y-my) for x,y in rows)/math.fsum((x-mx)**2 for x,y in rows)
    return my-b*mx,b


def coverage(cl,cs,nl,ns,repeats):
    out=[]
    for seed in SEEDS:
        missing=[]
        for npu in range(32):
            roles=[0]*(nl*repeats)+[1]*(ns*repeats)
            random.Random(seed+100003*npu).shuffle(roles)
            t=0.;seen=set()
            for role in roles:
                end=t+8*(cl if role==0 else cs)
                if min(end,4000)>max(t,2000):seen.add(role)
                t=end
                if t>=4000:break
            if len(seen)!=2:missing.append(npu)
        out.append(dict(seed=seed,mixed_cards=32-len(missing),missing_cards=missing))
    return out


def toy_coverage(cl,cs,nl,ns,w=0.):
    a,b=8*cl,8*cs+7*w
    p,q=nl/(nl+ns),ns/(nl+ns)
    avg=p*a+q*b
    def excess(d,r):
        z=2000/d;k=math.floor(z)+1
        return d*r**k*(k-z+r/(1-r))
    no_l=p*excess(b,q)/avg;no_s=q*excess(a,p)/avg
    return dict(no_L_per_card=no_l,no_S_per_card=no_s,
        all_32_mixed_if_independent=(1-no_l-no_s)**32,
        assumed_internal_S_wait_ms=w,
        assumptions='Stationary infinite iid labels, deterministic request durations8C_L and8C_S+7w, zeroL0, independent cards; not actual shared-storage probability.')


def main():
    protected=[HERE/n for n in ('long_scale_math.json','long_scale_math.py','constructed_candidates.json',
        'math_followup_constructed.py','prepare_context_scale.py','experiment.py','study_plan.json')]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    data=ast.literal_eval((ROOT/'data').read_text());data_sha=sha(ROOT/'data')
    points=sorted((k[0],r[1]/1000) for k,r in data.items() if k[1]==1024)
    a,b=affine(points);residual=[y-a-b*x for x,y in points]
    old=json.loads((HERE/'long_scale_math.json').read_text())
    prior_fit=old['fits_by_fixed_miss']['1024']
    assert a==prior_fit['intercept_ms'] and b==prior_fit['slope_ms_per_K']
    for k,row in data.items():
        if k[1]==1024:assert math.isclose(row[3],(k[0]*1024-1024)/128*BLOCK_GIB,rel_tol=1e-12)
    cs=a+b*10;vs=72*BLOCK_GIB;bs=1000*vs/cs
    fit=dict(method='OLS_affine_C_at_fixed_miss1024',intercept_ms=a,slope_ms_per_K=b,
        source_total_K_range=[points[0][0],points[-1][0]],
        source_points=[dict(total_K=x,C_ms=y) for x,y in points],
        max_abs_training_residual_ms=max(map(abs,residual)),
        training_rmse_ms=math.sqrt(statistics.mean(e*e for e in residual)),
        source_data_sha256=data_sha,
        frozen_source='long_scale_math.json',frozen_source_sha256=sha(HERE/'long_scale_math.json'))
    selected=[]
    for K in (200,256,384,512):
        is_raw=K==200;cl=data[K,1024][1]/1000 if is_raw else a+b*K
        nblocks=(K*1024-1024)//128;vl=nblocks*BLOCK_GIB;bl=1000*vl/cl
        assert bs<5<bl<50
        q=(vl-.005*cl)/(.005*cs-vs)
        nl,ns=1,int(math.floor(q+.5))
        assert 1<=nl<=128 and 1<=ns<=128
        WC=nl*cl+ns*cs;rho=32*(nl*vl+ns*vs)*1000/(160*WC)
        assert abs(rho-1)<=.005
        f=ns*cs/WC;reps=math.ceil(22000/(8*WC))
        goals={str(int(u*100)):dict(target_U_percent=100*u,
            continuous_layer_mean_S_wait_ms=(1/u-1)*cs/f,
            finite8_zero_other_wait_mean_S_wait_ms=(8/7)*(1/u-1)*cs/f) for u in (.89,.85,.80)}
        # Only an unavoidable receive-time bound. SSD and link overlap.
        handoff=max(0.,1000*vl/50-cs)
        populationL=nl*reps;populationS=ns*reps;population=populationL+populationS
        expected_SL=populationL*populationS/population
        pure=8*WC*reps
        long_model=dict(method='direct_data_row' if is_raw else 'affine_extrapolation_above_raw_maximum',
            fixed_miss=1024,intercept_ms=a,slope_ms_per_K=b,raw_support_total_K_range=[32,200],
            frozen_source='long_scale_math.json',frozen_source_sha256=sha(HERE/'long_scale_math.json'))
        if is_raw:long_model['raw_key']=[200,1024]
        short_model=dict(method='affine_extrapolation_below_raw_minimum',fixed_miss=1024,
            intercept_ms=a,slope_ms_per_K=b,raw_support_total_K_range=[32,200],
            frozen_source='long_scale_math.json',frozen_source_sha256=sha(HERE/'long_scale_math.json'))
        selected.append(dict(name=f'lowB4_L{K}m1024_S10m1024',num_npu=32,num_ssu=4,
            long_total_k=K,long_nql=1024,long_compute_us=cl*1000,long_is_raw=is_raw,
            long_V_GiB=vl,long_B_GiB_s=bl,short_total_k=10,short_nql=1024,short_compute_us=cs*1000,
            short_is_raw=False,short_V_GiB=vs,short_B_GiB_s=bs,counts=[nl,ns],roles=['L','S'],
            ideal_continuous_short_per_long=q,rho_ideal=rho,f_short=f,
            ideal_no_stall_active_short_cards=32*f,ideal_no_stall_active_long_cards=32*(1-f),
            goals=goals,long_request_pure_compute_ms=8*cl,short_request_pure_compute_ms=8*cs,
            L_balanced_storage_layer_service_ms=1000*vl/160,
            S_balanced_storage_layer_service_ms=1000*vs/160,
            short_first_link_start_delay_budget_ms=cs-1000*vs/50,
            residual_long_layer_equivalents_exceeding_short_C=cs/(1000*vl/160),
            residual_long_layer_equivalents_exceeding_S_first_link_budget=(cs-1000*vs/50)/(1000*vl/160),
            minimum_S_to_L_exposed_L0_ms_from_NPU_link=handoff,
            expected_S_to_L_transitions_in_uniform_finite_per_card_queue=expected_SL,
            ratio_of_work_sensitivity_U_if_only_link_min_handoff_percent=100*pure/(pure+expected_SL*handoff),
            handoff_bound_conditions='NextL0 can start receiving only at priorS final-layer compute start. Minimum receive timeVL/50; do not add full-disk and full-link durations. This is not FIFO-specific loss. Ratio uses expected transitions, not exact expected utilization.',
            long_byte_fraction=nl*vl/(nl*vl+ns*vs),
            minimum_22s_population=dict(L=populationL,S=populationS,repeats=reps,
                pure_compute_ms_per_card=pure,expected_blocks_all_32=32*8*reps*(nl*nblocks+ns*72)),
            pure_compute_clock_warm_coverage=coverage(cl,cs,nl,ns,reps),
            stationary_no_stall_mixed_risk=toy_coverage(cl,cs,nl,ns),
            stationary_U89_mean_internal_wait_mixed_risk=toy_coverage(cl,cs,nl,ns,goals['89']['finite8_zero_other_wait_mean_S_wait_ms']),
            model_provenance=dict(source_data_sha256=data_sha,data_sha256=data_sha,
                long_compute_model=long_model,short_compute_model=short_model,
                long_source_ttft_ms=data[K,1024][2] if is_raw else None,short_source_ttft_ms=None,
                V_model='Exact128-token cached blocks at176KiB per block; S10K/miss1024 has72blocks; no tail or padding.',
                context_support='10K short and256/384/512K long compute costs are explicit extrapolations, not measurements. Real deployment context/memory/kernel support is unverified.'),
            no_simulation_run=True))
    result=dict(schema_version='low-B-short-controls-v1',status='Mathematical plan only; no manifests or simulations generated by this script',
        input_family='data_affine_low_b_short',fit=fit,selected_candidates=selected,
        selection_rule='FixnL=1 and round idealnS/nL to nearest integer. All four counts<=128, |rho-1|<=.005. No observed outcomes or seed filtering used.',
        protocol=dict(num_npu=32,num_ssu=4,per_ssu_GiB_s=40,npu_link_GiB_s=50,n_layers=8,
            minimum_pure_compute_ms=22000,pilot_seed=7,confirmation_seeds=[19,43,67,101],warm_ms=[2000,4000],long_ms=[2000,20000],
            random='Independent full-queue shuffle with seed+100003*npu; fixed per-card ratio; all arrivals0; no common deck, ordering or seed cherry-picking.'),
        formulas=dict(B='V/C; Cseconds andVGiB',rho='32*(nL*VL+nS*VS)/(160*(nL*CL+nS*CS))',
            ideal_S_per_L='(VL-x*CL)/(x*CS-VS), x=160/32=5GiB/s',
            ideal_rho1_fS='(BL-x)/(BL-BS), for BS<x<BL',
            current_finite8='U≈8*sum(n*C)/[8*sum(n*C)+7*sum(n*w_internal)+sum(n*d_L0)]',
            finite8_short_only_target='w=(8/7)*(1/Utarget-1)*CS/fS only if longinternal andL0 waits are zero',
            fluid_static_marginal='While uncapped, dU/db=1/(N*B): lowerBshort gains more per extraGiB/s in that ideal fixed-profile model, but this is not the actual scheduler objective or guarantee.'),
        limitations=[
            'LowerBshort reverses the previous miss128short ordering and raises short ideal compute weight toabout62–64%; that alone does not establish FIFO waiting.',
            'FourSSU160GiB/s changes topology relative to earlier3/6/8SSU cases. Only the four within-this-family scale controls should be read as controlled comparisons.',
            'Request C andV both change with total length; BL is approximately, not exactly, fixed. Layer-scale variance is a motivation, not a Poisson/atomic-layer theorem.',
            'A balanced long-layer service scale larger thanCS is not proof of such residual work ahead. The simulator serves176KiB blocks and overlaps disk with50GiB/s NPU links.',
            'Same-request internal stalls and S-to-L0 exposed wait must be reported separately. SomeL0wait is unavoidable link receive time even without FIFO competition.',
            'The data minimum total is32K. Allshort10K C andlong256/384/512K C are extrapolated. Small fit residual does not validate newcontext feasibility or performance.',
            'Full-queueRandom cannot ensure everyNPU computes both roles in the fixed2s window. LargeL/shortcount combinations often fail even on the pure-compute clock; retain failures and all fixedseeds.',
            'Idealmeanrho near1 is not per-disk per-instant underload. EveryprofileB<50 does not eliminate cross-request prefetch link bottlenecks.',
            'Onceperlayer uses category-legal paths and snapshot routing; it is not proven to implement smallBpriority or optimize this dynamic U. Measure paired outcomes.'],
        preserved_files_sha256=before,generator_sha256=sha(Path(__file__)))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
    (HERE/'low_b_short_math.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 低带宽需求的短计算：四组机制对照','',
        '这次把短请求改为低B、短计算，长请求为较高B、长计算。方向有数学理由，但低利用率尚未测到；这是明确的外推压力实验。32NPU、4SSU×40GiB/s、每卡50GiB/s链路、8层，全部队列独立Random。','',
        '```text',f'C_ms(K) = {a:.15f} + {b:.17f}*K  （固定miss1024）',
        'V_GiB = ((K*1024-1024)/128)*176/1048576',
        'S10K：C=3.566244545ms，V=0.012084961GiB，B=3.388707865GiB/s','```','',
        f'拟合源为data中32K—200K的12行，最大训练残差{fit["max_abs_training_residual_ms"]:.3g}ms。只有长200K直接采用原始数据；短10K及长256/384/512K明确外推，sourceTTFT均为空，不伪称实测。短总输入10K，命中9K，72个完整176KiB块，无补齐开销。','',
        '## 为什么和前面的高B短计算不同','',
        '4盘总容量160，理想每卡平均5GiB/s；这次 `B短≈3.389 < 5 < B长≈7.54–7.82`。短请求总纯计算占约62%–64%，其等待更容易影响整机平均。此前context短miss128的权重约20%。在固定画像流体模型中，小B的边际利用率收益更高，所以此组也更适合检查“低B请求却被FIFO拖住”的问题；它不证明Once正好实现这种分配。','',
        '```text','rho = 32*(nL*VL+nS*VS)/(160*(nL*CL+nS*CS))',
        'rho=1时：f短 = (B长-5)/(B长-B短)',
        '理想nS/nL = (VL-5*CL)/(5*CS-VS)  （C用秒）','```','',
        '四组均固定nL=1，nS取上式最近整数，没有用已看到的结果或seed选比例。','',
        '| 长总K | L:S | rho | 长C ms | 长B GiB/s | 短纯算权重 | 单长盘侧尺度 ms |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for c in selected:lines.append(f"| {c['long_total_k']} | 1:{c['counts'][1]} | {c['rho_ideal']:.6f} | {c['long_compute_us']/1000:.6f} | {c['long_B_GiB_s']:.6f} | {100*c['f_short']:.2f}% | {c['L_balanced_storage_layer_service_ms']:.6f} |")
    lines+=['','## 需要实际量到多少等待','',
        '```text','U ≈ 8*sum(n*C)/[8*sum(n*C)+7*sum(n*w内部)+sum(n*d首层)]',
        '若长内部和首层都不等：w短目标 = (8/7)*(1/U目标-1)*C短/f短','```','',
        '下面有限8层门槛包括零等待短层，不能当成实测等待预测。早期连续层门槛单列JSON，避免混算。','',
        '| 长总K | 89%需短内部均w ms | 85%需w ms | 大读盘侧尺度/短C | S→L0最小链路暴露 ms |',
        '|---:|---:|---:|---:|---:|']
    for c in selected:lines.append(f"| {c['long_total_k']} | {c['goals']['89']['finite8_zero_other_wait_mean_S_wait_ms']:.6f} | {c['goals']['85']['finite8_zero_other_wait_mean_S_wait_ms']:.6f} | {c['L_balanced_storage_layer_service_ms']/cs:.3f} | {c['minimum_S_to_L_exposed_L0_ms_from_NPU_link']:.6f} |")
    lines+=['', '512K的单长读取盘侧分摊尺度约4.288ms，超过短计算3.566ms，所以一次长读取残余就可能越过短截止。200K的尺度只有1.670ms，需要更多重叠残余才会产生同样效果。但实际FIFO按176KiB块交错，必须测到短层前面的残余队列与IO-ready，不能把整长层当成原子独占。','',
        '首层必须另算：短最后一层只有3.566ms预取下一长首层，而512K长读取进入单卡链路至少要13.723ms，因此S→L0至少露出约10.157ms。这部分即使没有FIFO竞争也可能出现。这里只用 `max(0,VL/50-C短)` 下界；盘与链路流水重叠，不能再把整层盘服务时间加上。按随机队列转移次数估计，它单独造成约0.5个百分点级损失，不能代替短内部等待证据。','',
        '## 两秒warm全32卡混合是这组的主要风险','',
        '| 长总K | 每卡L/S数量 | 每卡纯计算秒 | 五seed纯计算时钟混合卡数 | toy无等待全32混合概率 |',
        '|---:|---:|---:|---|---:|']
    for c in selected:
        p=c['minimum_22s_population'];mix='/'.join(str(x['mixed_cards']) for x in c['pure_compute_clock_warm_coverage'])
        lines.append(f"| {c['long_total_k']} | {p['L']}/{p['S']} | {p['pure_compute_ms_per_card']/1000:.3f} | {mix} | {100*c['stationary_no_stall_mixed_risk']['all_32_mixed_if_independent']:.2f}% |")
    lines+=['', 'seed顺序为7/19/43/67/101。左侧诊断只累加计算，未跑共享盘；右侧是假设平稳独立标签和固定时长的toy概率，二者都不是真实仿真覆盖。等待会改变相位和短串持续时间，不能据它们保证通过或失败。512K按1:43时平均短串纯计算已约1.227秒，固定两秒很容易漏长类；200K更可能满足覆盖。保留全部预注册seed，不按覆盖结果重抽或换窗。','',
        '若512K低U但warm未全32混合，结果只能支持机制，不能说已满足用户全部约束。四组逐卡完整shuffle保持一致；有意识地强制每个小段都出现两类会改变随机输入定义，不能悄悄加入。','',
        '新实验还需报告：短内部均等待、长内部等待、首层交接、idle的精确损失分解；Baseline/Once同输入对照；逐盘真实忙率和名义需求越限时间；固定warm与长窗的各卡混合覆盖。4盘相对旧3/6/8盘本身也有拓扑变化，因果比较首先看本四组内部。','',
        '原始data固定miss1024的最短32K，其B约5.735793已高于4盘平均份额5；不能把本10K低B短请求称作原始data里直接找出的行。更大context的模型支持、HBM容量、注意力实现和拟合外推有效性均未验证。','',
        '[精确候选与来源](low_b_short_math.json) · [复算脚本](low_b_short_math.py)','']
    (HERE/'low_b_short_math.md').write_text('\n'.join(lines))
    print(json.dumps(dict(selected=len(selected),simulations=0,manifests=0,protected_unchanged=True)))
    for c in selected:print(c['name'],c['counts'],c['rho_ideal'],c['f_short'],c['goals']['89'])


if __name__=='__main__':main()
