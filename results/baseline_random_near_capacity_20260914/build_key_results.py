#!/usr/bin/env python3
"""Refresh only key_results.md/json from canonical completed audited runs."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
SEEDS=[7,19,43,67,101]
CONFIRMATION_SEEDS=[19,43,67,101]
SHORT_WINDOWS=[(2000,4000),(2000,20000)]
LONG_WINDOWS=[(2000,4000),(2000,20000),(20000,60000),(2000,60000)]
ALPHAS=['1.5','2.0']
SHORT_CASES=[('context384','context8_L384m1024_S10m128',8),('raw105','reference_load105',3),
    ('raw110','reference_load110',3),('hetero_short','hetero_S10_12_14_L384',8)]
HASH_CACHE={}


def sha(path):
    st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
    if key not in HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb') as f:
            for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
        HASH_CACHE[key]=h.hexdigest()
    return HASH_CACHE[key]


def read(path):return json.loads(path.read_text())


def close(a,b):assert math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-9),(a,b)


def small_slo(entry):
    n,p=int(entry['count']),int(entry['passed']);assert 0<=p<=n
    percentage=100*p/n if n else None
    if percentage is None:assert entry['percent'] is None
    else:close(percentage,entry['percent'])
    return dict(passed=p,count=n,percent=percentage)


def verified_documents(case,seed,strategy,disks):
    command_path=case/'command.json'
    if not command_path.exists():return [],'pending_no_completed_run',None
    command=read(command_path)
    if command.get('status')!='complete' or not command.get('completed_simulation') or command.get('returncode')!=0:
        status='failed_attempt_not_counted' if command.get('status') in ('failed','error') or command.get('returncode') not in (None,0) else 'pending_run'
        return [],status,None
    assert command['strategy']==strategy and command['order']=='random'
    result_path=case/'result.json.gz';manifest_path=case/'manifest.json.gz'
    if not result_path.exists():return [],'pending_result_sync',None
    assert sha(result_path)==command['output_sha256']
    assert sha(manifest_path)==command['manifest_sha256']
    docs=[]
    for name in ('analysis.json','extended_analysis.json'):
        path=case/name
        if not path.exists():continue
        doc=read(path)
        assert doc['all_technical_checks_passed'] and doc['seed']==seed and doc['strategy']==strategy
        assert doc['order']=='random' and doc['num_npu']==32 and doc['num_ssu']==disks and doc['n_layers']==8
        assert doc['input_fingerprint']==command['input_fingerprint']
        verified_sources={}
        for declared,h in doc['sources'].items():
            local=case/Path(declared).name
            assert local.exists() and sha(local)==h,(path,declared,'source hash mismatch')
            verified_sources[str(local.relative_to(HERE))]=h
        assert verified_sources[str(result_path.relative_to(HERE))]==command['output_sha256']
        docs.append((doc,path,verified_sources))
    metadata=dict(input_fingerprint=command['input_fingerprint'],manifest_sha256=command['manifest_sha256'],
        result_sha256=command['output_sha256'],command_sha256=sha(command_path))
    return docs,('complete' if docs else 'pending_analysis'),metadata


def extract_window(doc,w,source_path,sources,run_meta):
    left,right=w['start_ms'],w['end_ms'];D=32*(right-left)
    assert len(w['per_npu'])==32
    close(w['U_percent'],100*math.fsum(p['compute_ms'] for p in w['per_npu'])/D)
    close(w['U_percent'],math.fsum(p['U_percent'] for p in w['per_npu'])/32)
    assert w['all_32_active']==all(p['active_whole_window'] for p in w['per_npu'])
    assert w['long_short_mixed_card_count']==sum(p['long_short_mixed'] for p in w['per_npu'])
    classes=w['classes'];close(w['U_percent'],100*math.fsum(c['compute_ms'] for c in classes.values())/D)
    first=100*math.fsum(c['first_layer_stall_ms'] for c in classes.values())/D
    si=100*math.fsum(classes[r]['internal_layer_stall_ms'] for r in doc['short_roles'])/D
    li=100*math.fsum(classes[r]['internal_layer_stall_ms'] for r in doc['long_roles'])/D
    idle=100*math.fsum(p['idle_ms'] for p in w['per_npu'])/D
    close(w['U_percent']+first+si+li+idle,100.)
    slo={}
    for alpha in ALPHAS:
        e=w['slo']['alphas'][alpha];overall=small_slo(e['admission']);per_role={r:small_slo(v) for r,v in e['per_role'].items()}
        assert overall['count']==w['slo']['admission_cohort_count']==sum(x['count'] for x in per_role.values())
        assert overall['passed']==sum(x['passed'] for x in per_role.values())
        slo[alpha]=dict(overall=overall,per_role=per_role,
            arrival_clock_same_admission_cohort=small_slo(e['arrival_clock_same_admission_cohort']),
            arrival_cohort=small_slo(e['arrival_cohort']))
    cycles=w['complete_internal_cycles_inside_window']['per_role']
    return dict(status='complete',U_percent=w['U_percent'],all_32_active=w['all_32_active'],
        mixed_cards=w['long_short_mixed_card_count'],all_32_mixed=w['long_short_mixed_card_count']==32,
        per_npu_U_percent=[p['U_percent'] for p in w['per_npu']],
        per_role_U_percent={r:c['conditional_U_percent'] for r,c in classes.items()},
        loss_pp=dict(short_internal=si,long_internal=li,L0=first,idle=idle),
        short_internal_complete_cycle_mean_wait_ms={r:cycles[r]['stall_ms_including_zeros']['mean'] for r in doc['short_roles']},
        slo=slo,admission_cohort_count=w['slo']['admission_cohort_count'],
        completed_after_window_count=w['slo']['completed_after_window_count'],
        threshold_ms_by_profile=[dict(role=p['role'],total_K=p['seq_len_k'],miss_tokens=p['miss_tokens'],
            compute_ms=p['C_ms'],thresholds={a:float(a)*8*p['C_ms'] for a in ALPHAS}) for p in doc['profiles']],
        constructed_profile=any(p.get('constructed_profile') for p in doc['profiles']),
        profiles=doc['profiles'],nominal_any_ssu_over_capacity_percent=w['nominal']['any_ssu_over_capacity_percent'],
        physical_bandwidth_available=w['physical']['available'],
        analysis=str(source_path.relative_to(HERE)),analysis_sha256=sha(source_path),
        verified_source_sha256=sources,**run_meta)


def one_case(key,population,candidate,disks,seed,strategy,base,windows):
    label=f'{candidate}_ssu{disks}_h{population}_seed{seed}';case=base/label/strategy
    expected=dict(key=key,population_pure_compute_ms=population,candidate=candidate,num_ssu=disks,
        seed=seed,strategy=strategy,case=str(case.relative_to(HERE)))
    try:docs,status,meta=verified_documents(case,seed,strategy,disks)
    except (AssertionError,KeyError,ValueError,OSError) as e:
        docs=[];status='invalid_source_not_counted';meta=None;expected['validation_issue']=str(e)
    output=[]
    for left,right in windows:
        row=dict(**expected,window_ms=[left,right],status=status)
        if docs:
            matching=[(d,p,s,w) for d,p,s in docs for w in d['windows'] if (w['start_ms'],w['end_ms'])==(left,right)]
            if not matching:row['status']='pending_extended_analysis'
            else:
                # Extended analysis is preferred. If both exist, require agreement.
                for x in matching[1:]:
                    close(x[3]['U_percent'],matching[0][3]['U_percent'])
                    assert x[0]['input_fingerprint']==matching[0][0]['input_fingerprint']
                    for alpha in ALPHAS:assert small_slo(x[3]['slo']['alphas'][alpha]['admission'])==small_slo(matching[0][3]['slo']['alphas'][alpha]['admission'])
                d,p,s,w=matching[-1]
                try:row.update(extract_window(d,w,p,s,meta))
                except (AssertionError,KeyError,ValueError) as e:row.update(status='invalid_window_not_counted',validation_issue=str(e))
        output.append(row)
    return output


def average_slo(complete,alpha,role=None):
    values=[(r['slo'][alpha]['overall'] if role is None else r['slo'][alpha]['per_role'][role]) for r in complete]
    percentages=[v['percent'] for v in values if v['percent'] is not None]
    passed=sum(v['passed'] for v in values);count=sum(v['count'] for v in values)
    return dict(seed_mean_percent=statistics.mean(percentages) if percentages else None,
        seed_sample_sd_pp=statistics.stdev(percentages) if len(percentages)>1 else None,
        pooled_passed=passed,pooled_count=count,pooled_percent=100*passed/count if count else None)


def summarize(rows):
    groups=[]
    for key,_,_ in SHORT_CASES:
        for strategy in ['baseline','once']:
            seeds=SEEDS if strategy=='baseline' or key=='context384' else [7]
            for window in SHORT_WINDOWS:
                chosen=[r for r in rows if r['key']==key and r['population_pure_compute_ms']==22000 and r['strategy']==strategy and r['window_ms']==list(window)]
                assert sorted(r['seed'] for r in chosen)==sorted(seeds)
                complete=[r for r in chosen if r['status']=='complete'];us=[r['U_percent'] for r in complete]
                status=('complete_five_seeds' if len(seeds)==5 else 'complete_single_seed') if len(complete)==len(seeds) else 'partial_not_final'
                groups.append(dict(key=key,strategy=strategy,window_ms=list(window),expected_seeds=seeds,
                    status=status,completed_count=len(complete),completed_seeds=sorted(r['seed'] for r in complete),
                    pending_or_invalid=[dict(seed=r['seed'],status=r['status']) for r in chosen if r['status']!='complete'],
                    U_seed_mean_percent=statistics.mean(us) if us else None,U_seed_sample_sd_pp=statistics.stdev(us) if len(us)>1 else None,
                    U_min_percent=min(us) if us else None,U_max_percent=max(us) if us else None,
                    all_active_runs=sum(r['all_32_active'] for r in complete),all_mixed_runs=sum(r['all_32_mixed'] for r in complete),
                    slo={alpha:average_slo(complete,alpha) for alpha in ALPHAS},
                    per_role_slo={role:{alpha:average_slo(complete,alpha,role) for alpha in ALPHAS} for role in (complete[0]['slo']['1.5']['per_role'] if complete else [])}))
    return groups


def pairs(rows):
    groups=defaultdict(dict)
    for row in rows:
        if row['seed']==7 or row['key']=='context384':
            groups[row['key'],row['population_pure_compute_ms'],tuple(row['window_ms']),row['seed']][row['strategy']]=row
    output=[]
    for (key,pop,window,seed),parts in groups.items():
        a,b=parts['baseline'],parts['once']
        entry=dict(key=key,population_pure_compute_ms=pop,window_ms=list(window),seed=seed,
            baseline_status=a['status'],once_status=b['status'],status='pending_pair')
        for label,part in [('baseline',a),('once',b)]:
            if part['status']=='complete':
                entry[f'{label}_U_percent']=part['U_percent']
                entry[f'{label}_slo']={alpha:part['slo'][alpha]['overall'] for alpha in ALPHAS}
                entry[f'{label}_mixed_cards']=part['mixed_cards']
                entry[f'{label}_all_active']=part['all_32_active']
        if a['status']==b['status']=='complete':
            assert a['input_fingerprint']==b['input_fingerprint'] and a['manifest_sha256']==b['manifest_sha256'],entry
            entry.update(status='complete',same_input_verified=True,input_fingerprint=a['input_fingerprint'],
                baseline_U_percent=a['U_percent'],once_U_percent=b['U_percent'],U_once_minus_baseline_pp=b['U_percent']-a['U_percent'],
                baseline_mixed_cards=a['mixed_cards'],once_mixed_cards=b['mixed_cards'],
                baseline_all_active=a['all_32_active'],once_all_active=b['all_32_active'],
                slo={alpha:dict(baseline=a['slo'][alpha]['overall'],once=b['slo'][alpha]['overall'],
                    once_minus_baseline_pp=(b['slo'][alpha]['overall']['percent']-a['slo'][alpha]['overall']['percent']
                        if a['slo'][alpha]['overall']['percent'] is not None and b['slo'][alpha]['overall']['percent'] is not None else None)) for alpha in ALPHAS})
        output.append(entry)
    return output


def paired_context_summaries(paired):
    output=[]
    for stage,seeds in [('pilot_plus_confirmation',SEEDS),('confirmation_only',CONFIRMATION_SEEDS)]:
        for window in SHORT_WINDOWS:
            chosen=[p for p in paired if p['key']=='context384' and p['population_pure_compute_ms']==22000
                and p['seed'] in seeds and p['window_ms']==list(window)]
            assert sorted(p['seed'] for p in chosen)==sorted(seeds)
            complete=[p for p in chosen if p['status']=='complete']
            def statistics_of(values):
                valid=[v for v in values if v is not None]
                return dict(n=len(valid),mean_pp=statistics.mean(valid) if valid else None,
                    sample_sd_pp=statistics.stdev(valid) if len(valid)>1 else None,
                    min_pp=min(valid) if valid else None,max_pp=max(valid) if valid else None)
            output.append(dict(key='context384',stage=stage,window_ms=list(window),expected_seeds=seeds,
                completed_pair_count=len(complete),completed_seeds=sorted(p['seed'] for p in complete),
                status='complete_expected_pairs' if len(complete)==len(seeds) else 'partial_not_final',
                pending_or_invalid=[dict(seed=p['seed'],baseline_status=p['baseline_status'],once_status=p['once_status']) for p in chosen if p['status']!='complete'],
                U_once_minus_baseline=statistics_of([p['U_once_minus_baseline_pp'] for p in complete]),
                slo_once_minus_baseline={a:statistics_of([p['slo'][a]['once_minus_baseline_pp'] for p in complete]) for a in ALPHAS},
                all_active_pair_count=sum(p['baseline_all_active'] and p['once_all_active'] for p in complete),
                all_mixed_pair_count=sum(p['baseline_mixed_cards']==p['once_mixed_cards']==32 for p in complete),
                definition='Equal-weight mean and sample SD of within-seed paired differences; not SD of two independently pooled strategy means. All completed pairs remain even when coverage fails.'))
    return output


def fmt(value,digits=3):return 'pending' if value is None else f'{value:.{digits}f}'


def mean_sd(mean,sd):
    return 'pending' if mean is None else fmt(mean)+(f' ± {fmt(sd)}' if sd is not None else ' (n=1)')


def main():
    rows=[]
    for key,candidate,disks in SHORT_CASES:
        for seed in SEEDS:
            rows+=one_case(key,22000,candidate,disks,seed,'baseline',HERE/'runs',SHORT_WINDOWS)
        for seed in (SEEDS if key=='context384' else [7]):
            rows+=one_case(key,22000,candidate,disks,seed,'once',HERE/'runs',SHORT_WINDOWS)
    for key,candidate,disks,base in [('context384','context8_L384m1024_S10m128',8,HERE/'long_horizon/context384/runs'),
                                    ('raw110','reference_load110_long65',3,HERE/'runs')]:
        for strategy in ['baseline','once']:rows+=one_case(key,65000,candidate,disks,7,strategy,base,LONG_WINDOWS)
    groups=summarize(rows);paired=pairs(rows);context_paired=paired_context_summaries(paired)
    long_checks=[]
    for key in ('context384','raw110'):
        for strategy in ('baseline','once'):
            scope=[r for r in rows if r['key']==key and r['population_pure_compute_ms']==65000 and r['strategy']==strategy]
            bywindow={tuple(r['window_ms']):r for r in scope}
            a,b,c=[bywindow[w] for w in ((2000,20000),(20000,60000),(2000,60000))]
            if all(r['status']=='complete' for r in [a,b,c]):
                close((18*a['U_percent']+40*b['U_percent'])/58,c['U_percent'])
                for alpha in ALPHAS:
                    for v in ('count','passed'):assert a['slo'][alpha]['overall'][v]+b['slo'][alpha]['overall'][v]==c['slo'][alpha]['overall'][v]
                long_checks.append(dict(key=key,strategy=strategy,window_weighted_U_and_SLO_cohort_additivity=True))
    pending=[{k:r[k] for k in ['key','population_pure_compute_ms','seed','strategy','window_ms','status']} for r in rows if r['status']!='complete']
    confirmation_plan=HERE/'audit_remote_context384_once_confirmation_plan.json'
    confirmation_plan_details=None
    if confirmation_plan.exists():
        plan=read(confirmation_plan)
        assert sorted(j['seed'] for j in plan['jobs'])==sorted(CONFIRMATION_SEEDS)
        for job in plan['jobs']:
            assert job['strategy']=='once' and job['order']=='random' and job['num_ssu']==8 and job['trace'] is False
            baseline=next(r for r in rows if r['key']=='context384' and r['strategy']=='baseline'
                and r['population_pure_compute_ms']==22000 and r['seed']==job['seed'] and r['status']=='complete')
            assert job['input_fingerprint']==baseline['input_fingerprint'] and job['manifest_sha256']==baseline['manifest_sha256']
        confirmation_plan_details=dict(created_utc=plan['created_utc'],selection=plan['selection'],
            selection_evidence=plan['selection_evidence'],all_four_plan_inputs_match_existing_baseline=True)
    hetero_audit_path=HERE/'heterogeneity_confirmation_input_audit.json'
    hetero_provenance=None
    if hetero_audit_path.exists():
        audit=read(hetero_audit_path)
        assert audit['all_checks_passed'] and audit['selected_candidate']=='hetero_S10_12_14_L384'
        assert sorted(audit['selected_seeds'])==sorted(CONFIRMATION_SEEDS)
        assert audit['prepared_jsonl_sha256']==sha(HERE/'heterogeneity_S10_12_14_prepared_4seeds.jsonl')
        assert audit['plan_sha256']==sha(HERE/'heterogeneity_math.json')
        for case in audit['cases']:
            path=HERE.parents[1]/case['manifest']
            assert sha(path)==case['manifest_sha256']
        hetero_provenance=dict(audit=str(hetero_audit_path.relative_to(HERE)),audit_sha256=sha(hetero_audit_path),
            prepared_jsonl_sha256=audit['prepared_jsonl_sha256'],plan_sha256=audit['plan_sha256'],
            all_four_input_audits_passed=True)
    result=dict(schema_version='key-results-canonical-v2',updated_utc=datetime.now(timezone.utc).isoformat(),no_simulation_run=True,
        expected_window_rows=len(rows),completed_window_rows=len(rows)-len(pending),pending_window_rows=pending,
        definitions=dict(U='Exact NPU compute overlap/(32*T); full fixed window denominator includes all stalls andidle.',
            slo='Requests admitted in the half-openwindow; complete processing latency completion-admission <= alpha*8*own per-layerC. Follow completions afterwindowend; not first-token measurement or arrival TTFT.',
            seed_statistics='Equal-weight seed means with sampleSD; pooled request pass ratio is separately retained. Partial groups never represented as finished five-seed statistics.',
            pair='Same seed andwholeinput manifest SHA andinputfingerprint verified. Cohorts may differ bystrategy because admission times differ.',
            populations='h65000 rebuilds/re-shuffles a newfullpopulation; it is not an extension sharing theh22000prefix. Their same2–4s windows neednotmatch.',
            physical='Native exactphysicalbin recorder ends20s; noSSDgrant/busybandwidth is inferred for20–60s.',
            scope='Only explicit canonical directories are accepted; no recursive scan into replays, failed attempts or imported runtime copies.'),
        rows=rows,strategy_groups=groups,five_seed_baseline_groups=[g for g in groups if g['strategy']=='baseline'],
        five_seed_once_groups=[g for g in groups if g['strategy']=='once' and len(g['expected_seeds'])==5],
        seed7_pairs=[p for p in paired if p['seed']==7],
        context384_all_seed_pairs=[p for p in paired if p['key']=='context384' and p['population_pure_compute_ms']==22000],
        context384_paired_difference_groups=context_paired,long_window_additivity_checks=long_checks,
        context384_once_confirmation=dict(pilot_seed=7,confirmation_seeds=CONFIRMATION_SEEDS,
            plan=str(confirmation_plan.relative_to(HERE)),plan_sha256=sha(confirmation_plan) if confirmation_plan.exists() else None,
            plan_details=confirmation_plan_details,
            selection_stage='The Once confirmation was commissioned after the seed7 Once gain was observed; it was not preregistered at the beginning of the research.',
            seed_selection='19/43/67/101 fixed before observing those Once outcomes; no substitutions or coverage-based exclusions.',
            reporting='Five-seed strategy means and pair means include discovery seed7; the four confirmation-only paired statistics are separately shown.',
            raw_once_scope='raw105 and raw110 each have Once only at seed7 (n=1), not five-seed Once confirmation.'),
        hetero_short_confirmation=dict(pilot_seed=7,baseline_confirmation_seeds=CONFIRMATION_SEEDS,
            once_seeds=[7],once_sample_size=1,input_provenance=hetero_provenance,
            selection_stage='After the seed7 heterogeneous-short robustness run met the target and coverage conditions, run all four fixed additional Baseline seeds19/43/67/101 and a seed7 Once pair. Retain all outcomes.',
            matched_control_scope='The proper homogeneous control is S12, with the same coarse L/S order and totalC/V. The original context384 S10 run has a different mix and mean short profile; it is not a single-variable heterogeneity comparison.'),
        qualifications=dict(context384='Long384K/m1024 andshort10K/m128 use explicitC extrapolation; idealrho.996878 is not instantaneous underload.',
            raw105='Raw128K/m256 +32K/m4096, count32:63, idealrho1.050812; slightlyaboveinitial1.05 boundary.',
            raw110='Same rawprofiles,count17:31, idealrho1.100796; mildlyoverloaded sensitivity outside initial.95–1.05, notunderload evidence.',
            hetero_short='8SSU; L384K/m1024 and S10/12/14K/m128 in equal short fractions, count1:6:6:6, idealrho1.002029. All C values explicitly extrapolated. Baseline5 seeds; Once only seed7.'),
        generator_sha256=sha(Path(__file__)))
    (HERE/'key_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 重点结果：确认种子、同输入策略对照与长时间验证','',
        f"更新时间：{result['updated_utc']}。本快照完成 {result['completed_window_rows']}/{len(rows)} 个预期窗口结果；缺失保留为pending，不选seed或窗口。",'',
        'U取完整固定窗口的实际计算时间。SLO是窗口内接纳请求的`完成−接纳 <= 倍数×8×本请求单层C`，跟到最终完成；这是prefill完成代理，不是真实首token事件或包含排队的arrival TTFT。','',
        'context384：8个SSU，长384K/miss1024、短10K/miss128，均为明确外推的计算时间。raw105/raw110：3个SSU，原始data的128K/miss256与32K/miss4096，数量比分别32:63与17:31。所有组均为32个NPU，每卡独立打乱完整请求队列。','',
        'hetero_short：8个SSU，长384K/miss1024；短请求由10K、12K、14K/miss128各占1/3，全部C均为明确外推。每卡35个长请求、三种短请求各210个，理想平均rho=1.002029。其同质控制是短12K，二者粗L/S顺序及总C/V相同；它与context384原短10K队列的比较涉及画像及配比变化，不能当作只改变多样性的单变量对照。','',
        '## 各策略的种子统计','',
        'Baseline四组均计划保留5个seed（7、19、43、67、101）。context384的Once计划使用相同5个seed；raw105/raw110及hetero_short的Once各只有seed7，明确为n=1。','',
        '**研究阶段说明：** context384的Once四个确认种子是在观察到seed7的收益后才启动，不是研究开头预注册的Once试验。19、43、67、101在观察其Once结果前已固定；不更换seed，也不因覆盖失败而剔除。下面同时给出含发现seed7的五种子结果与仅四个后续确认种子的配对统计。','',
        'hetero_short也在seed7稳健性结果满足目标后，追加既定四个Baseline确认种子和seed7的Once配对；不为确认重新运行短12K控制组。此组Once只有一个seed，不把它写成多种子策略验证。','',
        '| 组 | 策略 | 窗口秒 | 已完成/计划 | U均值±种子SD % | SLO×1.5种子均值 % | SLO×2种子均值 % | 全活跃/全混合运行数 | 状态 |',
        '|---|---|---|---:|---:|---:|---:|---|---|']
    for g in groups:
        window=f"[{g['window_ms'][0]/1000:g},{g['window_ms'][1]/1000:g})"
        lines.append(f"| {g['key']} | {g['strategy']} | {window} | {g['completed_count']}/{len(g['expected_seeds'])} | {mean_sd(g['U_seed_mean_percent'],g['U_seed_sample_sd_pp'])} | {fmt(g['slo']['1.5']['seed_mean_percent'])} | {fmt(g['slo']['2.0']['seed_mean_percent'])} | {g['all_active_runs']}/{g['completed_count']}；{g['all_mixed_runs']}/{g['completed_count']} | {g['status']} |")
    lines+=['', 'SLO主表按seed等权平均；JSON另保留全部请求合并的通过数/总数及通过率，二者不能混称。覆盖失败的seed仍在均值中。context384为明确外推输入；raw105/110是原始data，rho分别1.050812/1.100796，后者是轻度过载敏感性。', '',
        '## context384：同seed的配对差值','',
        '每个差值先在同一seed内算`Once − Baseline`，再对seed等权取均值和样本SD。单位均为百分点（pp）。SLO按各策略自己的窗口接纳请求统计，两策略窗口内的请求集合可能不同。尚未完成的配对明确列出；未完成时的部分均值不当成最终确认结论。','',
        '| 纳入seed | 窗口秒 | 完成配对/计划 | ΔU均值±SD pp | ΔSLO×1.5均值±SD pp | ΔSLO×2均值±SD pp | 全活跃/全混合配对 | 状态 |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for g in context_paired:
        window=f"[{g['window_ms'][0]/1000:g},{g['window_ms'][1]/1000:g})"
        label='全部5seed（含发现seed7）' if g['stage']=='pilot_plus_confirmation' else '确认4seed（19/43/67/101）'
        values=[g['U_once_minus_baseline']]+[g['slo_once_minus_baseline'][a] for a in ALPHAS]
        means=[mean_sd(v['mean_pp'],v['sample_sd_pp']) for v in values]
        lines.append(f"| {label} | {window} | {g['completed_pair_count']}/{len(g['expected_seeds'])} | {means[0]} | {means[1]} | {means[2]} | {g['all_active_pair_count']}/{g['completed_pair_count']}；{g['all_mixed_pair_count']}/{g['completed_pair_count']} | {g['status']} |")
    lines+=['', '| seed | 窗口秒 | Baseline U % | Once U % | ΔU pp | ΔSLO×1.5 pp | ΔSLO×2 pp | 状态 |',
        '|---:|---|---:|---:|---:|---:|---:|---|']
    for p in result['context384_all_seed_pairs']:
        window=f"[{p['window_ms'][0]/1000:g},{p['window_ms'][1]/1000:g})"
        values=[fmt(p.get('baseline_U_percent')),fmt(p.get('once_U_percent')),fmt(p.get('U_once_minus_baseline_pp'))]
        differences=[fmt(p['slo'][a]['once_minus_baseline_pp']) if p['status']=='complete' else 'pending' for a in ALPHAS]
        status='同manifest已核验' if p['status']=='complete' else f"B:{p['baseline_status']}；O:{p['once_status']}"
        lines.append(f"| {p['seed']} | {window} | {values[0]} | {values[1]} | {values[2]} | {differences[0]} | {differences[1]} | {status} |")
    lines+=['', '## 每个seed的原始结果','',
        '| 组 | 策略 | seed | 窗口秒 | U % | SLO×1.5 %（通过/总数） | SLO×2 %（通过/总数） | 活跃/混合 | 状态 |',
        '|---|---|---:|---|---:|---|---|---|---|']
    def slo_cell(r,a):
        x=r['slo'][a]['overall'];return f"{fmt(x['percent'])} ({x['passed']}/{x['count']})"
    for r in rows:
        if r['population_pure_compute_ms']!=22000:continue
        window=f"[{r['window_ms'][0]/1000:g},{r['window_ms'][1]/1000:g})"
        if r['status']=='complete':lines.append(f"| {r['key']} | {r['strategy']} | {r['seed']} | {window} | {r['U_percent']:.3f} | {slo_cell(r,'1.5')} | {slo_cell(r,'2.0')} | {'32' if r['all_32_active'] else '<32'}/{r['mixed_cards']} | complete |")
        else:lines.append(f"| {r['key']} | {r['strategy']} | {r['seed']} | {window} | pending | pending | pending | pending | {r['status']} |")
    lines+=['', '## seed7：Baseline与Once的同输入配对','',
        '| 组/每卡最低纯计算量 | 窗口秒 | Baseline U | Once U | ΔU pp | SLO×1.5 B→O | SLO×2 B→O | 状态 |',
        '|---|---|---:|---:|---:|---|---|---|']
    for p in result['seed7_pairs']:
        window=f"[{p['window_ms'][0]/1000:g},{p['window_ms'][1]/1000:g})";label=f"{p['key']} / {p['population_pure_compute_ms']/1000:g}s"
        if p['status']=='complete':
            s=p['slo'];lines.append(f"| {label} | {window} | {p['baseline_U_percent']:.3f}% | {p['once_U_percent']:.3f}% | {p['U_once_minus_baseline_pp']:+.3f} | {s['1.5']['baseline']['percent']:.3f}→{s['1.5']['once']['percent']:.3f}% | {s['2.0']['baseline']['percent']:.3f}→{s['2.0']['once']['percent']:.3f}% | 同manifest已核验 |")
        else:
            us=[fmt(p.get(f'{s}_U_percent'))+'%' if f'{s}_U_percent' in p else 'pending' for s in ['baseline','once']]
            slo_pending=[]
            for alpha in ALPHAS:
                values=[fmt(p[f'{s}_slo'][alpha]['percent'])+'%' if f'{s}_slo' in p else 'pending' for s in ['baseline','once']]
                slo_pending.append('→'.join(values))
            lines.append(f"| {label} | {window} | {us[0]} | {us[1]} | pending | {slo_pending[0]} | {slo_pending[1]} | B:{p['baseline_status']}；O:{p['once_status']} |")
    lines+=['', '每个完整配对均核对manifest SHA和input fingerprint相同。SLO仍按各策略自己的窗口接纳cohort，不能声称是同一批进入窗口的请求。', '',
        '## 两组每卡至少65秒纯计算量的输入：固定长时间验证','',
        '65秒表示每张卡的输入队列至少有65秒纯计算工作量；它是重新生成并全量打乱的队列，不是短实验的时间前缀。因此相同seed7的[2,4)秒结果可以与22秒输入不同。', '',
        '| 组 | 策略 | 窗口秒 | U % | SLO×1.5 % | SLO×2 % | 活跃/混合 | 状态 |',
        '|---|---|---|---:|---:|---:|---|---|']
    for r in rows:
        if r['population_pure_compute_ms']!=65000:continue
        window=f"[{r['window_ms'][0]/1000:g},{r['window_ms'][1]/1000:g})"
        if r['status']=='complete':lines.append(f"| {r['key']} | {r['strategy']} | {window} | {r['U_percent']:.3f} | {slo_cell(r,'1.5')} | {slo_cell(r,'2.0')} | {'32' if r['all_32_active'] else '<32'}/{r['mixed_cards']} | complete |")
        else:lines.append(f"| {r['key']} | {r['strategy']} | {window} | pending | pending | pending | pending | {r['status']} |")
    lines+=['', '20秒之后U/SLO/覆盖来自完整请求与逐层日志；不外推20秒前的SSU实际带宽。完整时校验`U[2,60)=(18*U[2,20)+40*U[20,60))/58`及两个相邻接纳cohort的通过数/总数可加。', '',
        '[所有精确数值、类别SLO、来源SHA与pending清单](key_results.json) · [重建脚本](build_key_results.py)', '']
    (HERE/'key_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(expected_windows=len(rows),completed_windows=len(rows)-len(pending),pending_windows=len(pending),
        complete_five_seed_groups=sum(g['status']=='complete_five_seeds' for g in groups),complete_pairs=sum(p['status']=='complete' for p in paired))))
    issues=[r for r in rows if r['status'].startswith('invalid')]
    if issues:print(json.dumps(dict(validation_issues=[dict(case=r['case'],issue=r.get('validation_issue')) for r in issues]),ensure_ascii=False))


if __name__=='__main__':main()
