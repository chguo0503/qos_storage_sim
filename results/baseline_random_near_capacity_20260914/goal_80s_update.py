#!/usr/bin/env python3
"""Independent, rerunnable 89%/85% target review; no simulation or frozen edits."""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
TARGETS = (0.89, 0.85)


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(a, b, label=''):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6), (label, a, b)


def overlap(a, b):
    return max(0., min(20000., b)-max(2000., a))


def mean(values):
    return statistics.mean(values) if values else 0.


def review_case(path):
    a = read(path)
    assert a['all_technical_checks_passed']
    raw_path = path.parent/'result.json.gz'
    raw_sha = digest(raw_path)
    assert a['sources'][str(raw_path.resolve())] == raw_sha
    raw = read(raw_path)
    summary = raw['summary']
    metadata = raw['metadata']
    profiles = {p['role']: p for p in a['profiles']}
    roles = list(profiles)
    shorts = a['short_roles']
    longs = a['long_roles']
    counts = {role: profiles[role]['requests'] for role in roles}
    gcd = math.gcd(*counts.values())
    ratio = {role: n//gcd for role, n in counts.items()}
    C = {r: profiles[r]['C_ms'] for r in roles}
    WC = math.fsum(counts[r]*C[r] for r in roles)
    nS = sum(counts[r] for r in shorts)
    Cs = math.fsum(counts[r]*C[r] for r in shorts)/nS
    f = nS*Cs/WC
    rho = 32*math.fsum(counts[r]*profiles[r]['V_GiB'] for r in roles)*1000/(40*a['num_ssu']*WC)
    warm = next(w for w in a['windows'] if w['start_ms'] == 2000 and w['end_ms'] == 4000)
    long = next(w for w in a['windows'] if w['start_ms'] == 2000 and w['end_ms'] == 20000)
    groups = {r:dict(compute=0., active=0., first=0., internal=0., waits=[], first_waits=[], R=[]) for r in roles}
    # The role is independently recovered from each batch's measured C; all
    # current profiles have distinct C. This fails rather than guessing ties.
    for batch in summary['microbatch_metrics']:
        assert len(batch['member_request_ids']) == 1
        layers = batch['layer_metrics']
        matches = [role for role in roles if math.isclose(C[role], layers[0]['compute_duration_ms'], abs_tol=1e-8)]
        assert len(matches) == 1, matches
        role = matches[0]
        g = groups[role]
        g['active'] += overlap(batch['admission_time_ms'], batch['completion_time_ms'])
        prev_end = batch['admission_time_ms']
        for index, layer in enumerate(layers):
            start, end = layer['compute_start_ms'], layer['compute_end_ms']
            near(end-start, C[role], 'layer C')
            g['compute'] += overlap(start, end)
            g['first' if index == 0 else 'internal'] += overlap(prev_end, start)
            if index == 0:
                if batch['admission_time_ms'] >= 2000 and start <= 20000:
                    g['first_waits'].append(start-batch['admission_time_ms'])
            else:
                previous = layers[index-1]
                cycle_start = previous['compute_start_ms']
                if cycle_start >= 2000 and start <= 20000:
                    wait = start-previous['compute_end_ms']
                    near(wait, layer['io_barrier_wait_ms'], 'internal wait')
                    g['waits'].append(max(0., wait))
                    g['R'].append(layer['io_ready_time_ms']-cycle_start)
            prev_end = end
    denominator = 32*18000
    for role, g in groups.items():
        cls = long['classes'][role]
        for own, logged in [('compute','compute_ms'),('active','active_ms'),('first','first_layer_stall_ms'),('internal','internal_layer_stall_ms')]:
            near(g[own], cls[logged], role+' '+own)
        near(mean(g['waits']), long['complete_internal_cycles_inside_window']['per_role'][role]['stall_ms_including_zeros']['mean'], 'mean wait')
    U = 100*math.fsum(g['compute'] for g in groups.values())/denominator
    near(U, long['U_percent'], 'full window U')
    idle = max(0., denominator-math.fsum(g['active'] for g in groups.values()))
    first = 100*math.fsum(g['first'] for g in groups.values())/denominator
    Sint = 100*math.fsum(groups[r]['internal'] for r in shorts)/denominator
    Lint = 100*math.fsum(groups[r]['internal'] for r in longs)/denominator
    idle_pp = 100*idle/denominator
    near(U+first+Sint+Lint+idle_pp, 100., 'exact loss accounting')
    waits = {r: mean(groups[r]['waits']) for r in roles}
    first_waits = {r: mean(groups[r]['first_waits']) for r in roles}
    short_wait_cost = math.fsum(counts[r]*waits[r] for r in shorts)
    all_wait_cost = math.fsum(counts[r]*waits[r] for r in roles)
    first_wait_cost = math.fsum(counts[r]*first_waits[r] for r in roles)
    exact_source_C_fraction = math.fsum(groups[r]['compute'] for r in shorts)/math.fsum(g['compute'] for g in groups.values())
    row = dict(candidate=metadata['candidate'], num_ssu=a['num_ssu'], seed=a['seed'], strategy=a['strategy'],
        source_family='constructed_C_extrapolation' if any(p.get('constructed_profile') for p in profiles.values()) else 'direct_data',
        profiles=profiles, count_ratio_by_role=ratio, short_roles=shorts, long_roles=longs,
        ideal_load_ratio=rho, ideal_short_compute_fraction=f, actual_long_window_short_compute_fraction=exact_source_C_fraction,
        count_weighted_short_C_ms=Cs, observed_short_internal_wait_ms=short_wait_cost/nS,
        observed_internal_wait_by_role_ms=waits, observed_L0_wait_by_role_ms=first_waits,
        targets={str(int(t*100)):dict(U_percent=100*t,
            continuous_layer_required_mean_short_wait_ms=(1/t-1)*WC/nS,
            finite8_zero_L0_and_zero_long_wait_required_mean_short_wait_ms=(8/7)*(1/t-1)*WC/nS,
            finite8_with_observed_L0_and_long_wait_required_mean_short_wait_ms=max(0.,(8*WC*(1/t-1)-first_wait_cost-7*(all_wait_cost-short_wait_cost))/(7*nS)),
            observed_over_continuous_required=(short_wait_cost/nS)/((1/t-1)*WC/nS)) for t in TARGETS},
        observed_warm_U_percent=warm['U_percent'], observed_long_U_percent=U,
        all_32_active_long=long['all_32_active'], warm_mixed_cards=warm['long_short_mixed_card_count'], long_mixed_cards=long['long_short_mixed_card_count'],
        exact_long_loss_pp=dict(short_internal=Sint,long_internal=Lint,L0_exposed_handoff=first,idle=idle_pp,total=100-U),
        observed_wait_substitution_U_percent=100*WC/(WC+all_wait_cost),
        observed_7internal_1L0_substitution_U_percent=100*WC/(WC+(7*all_wait_cost+first_wait_cost)/8),
        capacity_upper_bound_complete_finite_U_percent=min(100.,100/rho),
        analysis_path=str(path.relative_to(HERE)), analysis_sha256=digest(path), result_sha256=raw_sha)
    # A tightly bounded diagnostic: keep old observed readiness lifetimes and
    # change only C. It ignores altered V, request ratios, arrivals and feedback.
    if row['candidate'] in ('fit6_m2048_s10m128','fit8_m2048_s10m128') and a['strategy']=='baseline':
        newCs = .44900573175788394 + .010485759875732867*10
        replay = mean([max(0., R-newCs) for R in groups['S']['R']])
        row['miss64_frozen_readiness_sensitivity'] = dict(old_C_ms=C['S'], new_C_ms=newCs,
            old_readiness_sample_count=len(groups['S']['R']), new_wait_if_readiness_unchanged_ms=replay,
            status='Counterfactual on old readiness samples; not a new simulation or prediction. New V, mix and queue feedback are not replayed.')
    return row


def contextual_targets():
    rows=[]
    for c in read(HERE/'long_scale_math.json')['selected_candidates']:
        cs=c['short_compute_us']/1000
        rows.append(dict(candidate=c['name'], num_ssu=c['num_ssu'], counts=c['counts'],
            rho=c['rho_ideal'], f_short=c['f_short'], short_C_ms=cs,
            target_wait_ms={str(int(t*100)):(1/t-1)*cs/c['f_short'] for t in TARGETS},
            finite8_zero_other_wait_target_ms={str(int(t*100)):(8/7)*(1/t-1)*cs/c['f_short'] for t in TARGETS},
            long_storage_service_scale_ms=c['L_balanced_storage_layer_service_ms'],
            short_first_link_start_budget_ms=c['short_first_link_start_delay_budget_ms'],
            long_request_compute_ms=c['long_request_pure_compute_ms'],
            provenance=c['model_provenance'], status='Frozen input mathematics; measured outcomes listed separately only if analysis exists.'))
    return rows


def reference_neighbors(reference):
    P=reference['profiles'];CA=P['A']['C_ms'];CB=P['B']['C_ms'];VA=P['A']['V_GiB'];VB=P['B']['V_GiB']
    rows=[]
    for name, a, b, selected in [('reference_load100',7,15,True),('reference_load105',32,63,True),('reference_load110',17,31,True),('reference_load115_math_only',10,17,False)]:
        WC=a*CA+b*CB;rho=32*(a*VA+b*VB)*1000/(120*WC)
        rows.append(dict(candidate=name, counts_A_B=[a,b], input_family='direct_data', selected_for_run_by_root=selected,
            num_ssu=3, rho=rho, f_short=a*CA/WC,
            target_wait_ms={str(int(t*100)):(1/t-1)*WC/a for t in TARGETS},
            finite8_zero_other_wait_target_ms={str(int(t*100)):(8/7)*(1/t-1)*WC/a for t in TARGETS},
            complete_finite_capacity_upper_bound_U_percent=min(100.,100/rho),
            qualification=('Near initial 1.05 upper edge; discrete rho is slightly above 1.05.' if name=='reference_load105' else
                'Outside initial 0.95–1.05 range: explicitly mildly overloaded sensitivity, not underload evidence.' if rho>1.05 else
                'Ideal input mean near capacity; no per-instant underload guarantee.'),
            statement='Capacity bound is for full finite run or stable completed mix, not every arbitrary time window. A Once pair is needed to quantify policy-specific excess loss.'))
    return rows


def main():
    protected=[HERE/n for n in ['study_plan.json','long_scale_math.json','long_scale_math.py','constructed_candidates.json','math_followup_constructed.py','prepare_context_scale.py','experiment.py']]
    before={str(p.relative_to(HERE)):digest(p) for p in protected}
    paths=sorted((HERE/'runs').glob('*/*/analysis.json'))
    rows=[]
    for path in paths:
        row=review_case(path);rows.append(row)
        print(json.dumps(dict(candidate=row['candidate'], strategy=row['strategy'], seed=row['seed'], U=row['observed_long_U_percent'])),flush=True)
    baseline=[r for r in rows if r['strategy']=='baseline']
    grouped=defaultdict(list)
    for row in rows:grouped[(row['candidate'],row['num_ssu'],row['strategy'])].append(row)
    groups=[]
    for (name,ssu,strategy), rr in sorted(grouped.items()):
        us=[r['observed_long_U_percent'] for r in rr]
        groups.append(dict(candidate=name,num_ssu=ssu,strategy=strategy,seeds=sorted(r['seed'] for r in rr),n=len(rr),
            long_U_mean_percent=mean(us),long_U_sample_sd_pp=statistics.stdev(us) if len(us)>1 else None,
            warm_all_mixed_runs=sum(r['warm_mixed_cards']==32 for r in rr),
            long_all_mixed_runs=sum(r['long_mixed_cards']==32 for r in rr),
            observed_short_wait_mean_ms=mean([r['observed_short_internal_wait_ms'] for r in rr]),
            U89_wait_ms=rr[0]['targets']['89']['continuous_layer_required_mean_short_wait_ms'],U85_wait_ms=rr[0]['targets']['85']['continuous_layer_required_mean_short_wait_ms'],
            finite8_U89_zero_other_wait_ms=rr[0]['targets']['89']['finite8_zero_L0_and_zero_long_wait_required_mean_short_wait_ms'],
            finite8_U85_zero_other_wait_ms=rr[0]['targets']['85']['finite8_zero_L0_and_zero_long_wait_required_mean_short_wait_ms'],
            source_family=rr[0]['source_family']))
    reference=next(r for r in baseline if r['candidate']=='reference' and r['seed']==7)
    context=contextual_targets();neighbors=reference_neighbors(reference)
    miss=read(HERE/'miss64_math.json')
    backups=[]
    for c in miss['mathematical_candidates']:
        if c['name'] not in ('backup6_m2048_s10m64_exact','backup8_m2048_s10m64_exact'):continue
        backups.append(dict(candidate=c['name'],status='Mathematical backup only, no new manifest or simulation',num_ssu=c['num_ssu'],
            counts=c['counts'],rho=c['rho_ideal'],short_C_ms=c['short_C_ms'],short_B_GiB_s=c['short_B_GiB_s'],f_short=c['f_short'],
            target_wait_ms={str(int(t*100)):(1/t-1)*c['short_C_ms']/c['f_short'] for t in TARGETS},
            finite8_zero_other_wait_target_ms={str(int(t*100)):(8/7)*(1/t-1)*c['short_C_ms']/c['f_short'] for t in TARGETS},
            physical_block_rule=c['physical_block_rule'],source='raw L200K/miss2048 + extrapolated S10K/miss64 with exact88KiB tail',
            rationale='Tighter first-link-start deadline may expose more old I/O latency, but rho rebalancing lowers f_short. Measured wait must actually rise; no guaranteed U decrease.'))
    after={str(p.relative_to(HERE)):digest(p) for p in protected};assert before==after
    result=dict(schema_version='goal-80s-review-v1',updated_utc=datetime.now(timezone.utc).isoformat(),no_new_simulations=True,
        target_definition='Long fixed [2,20)s U in the 80s, while all32 remain active and mixed in fixed warm[2,4)s; retain every preregistered seed.',
        authoritative_finite8_formula='U≈8*sum(n*C)/[8*sum(n*C)+7*sum(n*w_internal)+sum(n*d_L0)]. Completion mix and window edges still matter.',
        early_continuous_layer_formula='w_target=(1/U_target-1)*C_short/f_short. This treats every layer as an internal stationary cycle.',
        finite8_zero_other_wait_threshold='w_target=(8/7)*(1/U_target-1)*C_short/f_short only if L0 and long internal waits are zero.',
        formula_conditions=['Stable completed request mix near input mix; long internal wait negligible; same eight layers per request.',
            'Approximation treats layers uniformly. Eight-layer correction uses (7*w_internal+w_L0)/8; measured window losses are clipped exactly.',
            'Observed waits explain measured U after the run; they are not an independent prediction.',
            'No universal lower bound on Random U is proved. Mean rho does not guarantee instantaneous per-disk underload.'],
        rows=rows,groups=groups,closest_baseline_groups=sorted([g for g in groups if g['strategy']=='baseline'],key=lambda g:g['long_U_mean_percent'])[:8],
        context_frozen_math=context,reference_local_load_scan=neighbors,miss64_backups=backups,
        protected_files_sha256=before,source_plan_sha256={n:digest(HERE/n) for n in ['long_scale_math.json','miss64_math.json']},generator_sha256=digest(Path(__file__)))
    (HERE/'goal_80s_update.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 目标改为长期 80 几：等待到底还差多少', '',
        f'固定长窗 `[2,20)s`，warm 仍用 `[2,4)s`。本快照独立复核 {len(rows)} 次已完成运行；不挑窗口，不删混合覆盖失败的 seed。没有启动新仿真。', '',
        '```text', 'fS = 输入中短请求纯计算时间 / 全部纯计算时间',
        'w89 = (1 / 0.89 - 1) * CS / fS = 0.1235955 * CS / fS',
        'w85 = (1 / 0.85 - 1) * CS / fS = 0.1764706 * CS / fS', '```', '',
        'w 是包括零等待在内的短内部层平均额外等待。上面是保留早期口径的连续层近似，把8层都视作内部稳态周期，不能当成当前有限8层的严格预测。原80%门槛是 `0.25*CS/fS`，89%只需其49.44%，85%需其70.59%。', '',
        '**当前有限8层采用下面的修正公式**；每请求7次内部等待和1次首层等待分别计算：', '',
        '```text', 'U ≈ 8*sum(n*C) / [8*sum(n*C) + 7*sum(n*w内部) + sum(n*d首层)]',
        '若首层和长内部都不等：w目标 = (8/7)*(1/U目标-1)*CS/fS', '```', '',
        '因此在“只有短内部等待”这一条件下，有限8层门槛是早期门槛的8/7倍。首层与长内部真实有等待时应显式计入，不能假设其必为零；JSON同时列出零其他等待门槛及代入实测首层/长层等待的门槛。所有版本仍要求完成比例稳定，精确窗口结果以实际裁剪为准。', '',
        '## 已完成结果：按长期实际 U 排序', '',
        '| 候选 | 盘 | 来源 | seed 数 | 长窗 U 均值 | 实测短等待 ms | 连续层89/85需 ms | 有限8层89/85需 ms（其他等待为0） | warm全部混合 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for g in sorted([g for g in groups if g['strategy']=='baseline'],key=lambda g:g['long_U_mean_percent'])[:8]:
        sd='' if g['long_U_sample_sd_pp'] is None else f" ± {g['long_U_sample_sd_pp']:.3f}"
        lines.append(f"| {g['candidate']} | {g['num_ssu']} | {'原始data' if g['source_family']=='direct_data' else 'C外推'} | {g['n']} | {g['long_U_mean_percent']:.3f}{sd}% | {g['observed_short_wait_mean_ms']:.4f} | {g['U89_wait_ms']:.4f}/{g['U85_wait_ms']:.4f} | {g['finite8_U89_zero_other_wait_ms']:.4f}/{g['finite8_U85_zero_other_wait_ms']:.4f} | {g['warm_all_mixed_runs']}/{g['n']} |")
    lines+=['', '本表只列最接近目标的8组；JSON保留所有已完成候选、所有策略与逐seed结果。±为种子样本标准差，n=1不是多种子结论。warm混合覆盖是约束，不能因为U较低而忽略；长期18秒也只是固定观察窗，不能冒称无限时间极限。', '',
        '## 首层与交接是否漏算', '',
        '从原始逐层日志重新裁剪：`100%-实际U = 短内部等待 + 长内部等待 + 首层暴露等待 + 空闲`。首层等待从接纳到首层计算；跨请求预取已藏住的部分没有再次记成stall。', '',
        '| 接近目标的基线/seed7 | 短内部损失 pp | 长内部 pp | L0/交接 pp | 空闲 pp | 实际 U |',
        '|---|---:|---:|---:|---:|---:|']
    for r in sorted([r for r in baseline if r['seed']==7],key=lambda r:r['observed_long_U_percent'])[:6]:
        x=r['exact_long_loss_pp'];lines.append(f"| {r['candidate']} | {x['short_internal']:.4f} | {x['long_internal']:.4f} | {x['L0_exposed_handoff']:.4f} | {x['idle']:.4f} | {r['observed_long_U_percent']:.4f}% |")
    lines+=['', '原简式把每层都近似成内部层，不能在它的8份内部等待上再多加1份首层。有限8层更合适的事后近似为：', '',
        '```text', 'U ≈ 8*sum(n*C) / [8*sum(n*C) + sum n*(7*w内部+w首层)]', '```', '',
        '这仍会受长窗完成比例和边界影响；精确损失表才是全窗记账。JSON同时保存两个近似与实测，避免用它们互相冒充。', '',
        '## 正在检验的长读取尺度对照', '',
        '| 长总K/短10K | L:S | rho | 短纯算权重 | 连续层89/85需 ms | 有限8层89/85需 ms（其他等待为0） | 单长盘侧尺度 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in context:
        lines.append(f"| {c['candidate']} | {c['counts'][0]}:{c['counts'][1]} | {c['rho']:.6f} | {100*c['f_short']:.2f}% | {c['target_wait_ms']['89']:.4f}/{c['target_wait_ms']['85']:.4f} | {c['finite8_zero_other_wait_target_ms']['89']:.4f}/{c['finite8_zero_other_wait_target_ms']['85']:.4f} | {c['long_storage_service_scale_ms']:.4f} |")
    lines+=['', '这四组是8盘、长miss1024、短miss128。长200K来自data；长256/384/512K与短10K的C都是明确外推。长B只近似固定，长读取突发变大并不证明实际平均等待同比变大；过去3→6/8盘的等待缩放预测已经失败。先看这组实测，不建议在尚无规模趋势证据时继续盲增context。', '',
        '## 优先补查原始 data 的 reference 邻域', '',
        '固定A=总128K/miss256、B=总32K/miss4096，只改每卡数量比；32卡各自全队列独立Random。', '',
        '| 候选 | A:B | rho | 连续层89/85需 ms | 有限8层89/85需 ms（其他等待为0） | 完整批次容量上界 U |',
        '|---|---:|---:|---:|---:|---:|']
    for c in neighbors:
        lines.append(f"| {c['candidate']} | {c['counts_A_B'][0]}:{c['counts_A_B'][1]} | {c['rho']:.6f} | {c['target_wait_ms']['89']:.4f}/{c['target_wait_ms']['85']:.4f} | {c['finite8_zero_other_wait_target_ms']['89']:.4f}/{c['finite8_zero_other_wait_target_ms']['85']:.4f} | {c['complete_finite_capacity_upper_bound_U_percent']:.3f}% |")
    lines+=['', '100/105/110三组已由主研究安排；115只保留数学值、不建议借其容量上限宣称FIFO劣势。105离散配比rho=1.050812，略超1.05边界；110明确超出初始0.95–1.05主范围，是轻度过载敏感性。容量上界 `min(1,1/rho)` 适用于完整有限批次共同分母或稳定混合，不能直接套任意窗口。', '',
        '若105已达到长期80几，优先固定19/43/67/101复验并做同输入Once；若仅110达到，也要报告容量因素。观察Baseline与Once差距才能区分策略额外损失，不能只展示Baseline绝对值。每个seed都报告warm混合覆盖，不按结果淘汰。', '',
        '## 如果上述仍高于90%：一个有边界条件的备选', '',
        '可选原始长200K/miss2048，短10K/miss64采用透明C外推与精确88KiB尾块。优先6盘1:29（rho≈1.002796），8盘1:56作备选；长请求纯算约566ms，比miss4096长请求更利于warm混合。仍保持每卡独立全队列shuffle，不同步、不按seed选择。', '',
        '| 盘 | L:S | 短C ms | 短B GiB/s | 连续层89/85需 ms | 有限8层89/85需 ms（其他等待为0） |',
        '|---|---:|---:|---:|---:|---:|']
    for c in backups:lines.append(f"| {c['num_ssu']} | {c['counts'][0]}:{c['counts'][1]} | {c['short_C_ms']:.6f} | {c['short_B_GiB_s']:.4f} | {c['target_wait_ms']['89']:.4f}/{c['target_wait_ms']['85']:.4f} | {c['finite8_zero_other_wait_target_ms']['89']:.4f}/{c['finite8_zero_other_wait_target_ms']['85']:.4f} |")
    lines+=['', 'miss64更紧的首块链路启动余量可能露出更多等待，但重新配比保持rho后短计算权重下降，所需平均等待反而略升。不能只凭更短C承诺更低U。', '']
    for r in rows:
        cf=r.get('miss64_frozen_readiness_sensitivity')
        if cf:lines.append(f"- 对 {r['candidate']} 的旧实际读取完成延迟保持不变、仅把C从{cf['old_C_ms']:.6f}改为{cf['new_C_ms']:.6f}ms，短等待会从{r['observed_short_internal_wait_ms']:.6f}变为{cf['new_wait_if_readiness_unchanged_ms']:.6f}ms。这是旧轨迹敏感性，未重新计算字节、比例与排队反馈，不是新实验或预测。")
    lines+=['', '[完整逐seed数值、损失分解与来源哈希](goal_80s_update.json) · [可重运行脚本](goal_80s_update.py)', '']
    (HERE/'goal_80s_update.md').write_text('\n'.join(lines))
    print(json.dumps(dict(reviewed_runs=len(rows),baseline_runs=len(baseline),groups=len(groups),all_loss_checks_passed=True)),flush=True)


if __name__=='__main__':
    main()
