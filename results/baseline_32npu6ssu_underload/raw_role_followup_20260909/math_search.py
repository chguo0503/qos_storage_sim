#!/usr/bin/env python3
"""Deterministic raw-row fixed-role capacity enumeration. Never runs simulation."""
from __future__ import annotations

import ast
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
IO = 176*1024/2**30
NPU, SSU, DISK, LINK = 32, 6, 40.0, 50.0
SHORT_SEQ_MAX_K = 80
ISSUE_INTERVAL_MS = .0001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog():
    table = ast.literal_eval((ROOT/'data').read_text())
    out = []
    for index, (key, (bw, cu, ttft, v)) in enumerate(sorted(table.items())):
        count = math.ceil((key[0]*1024-key[1])/128)
        c = cu/1000
        by_offset = [[sum((b+o) % SSU == s for b in range(count))*IO for s in range(SSU)] for o in range(SSU)]
        p = {'index': index, 'key': list(key), 'source_C_us': cu, 'C_ms': c,
             'source_V_gib': v, 'physical_V_gib': count*IO, 'padding_gib': count*IO-v,
             'source_bandwidth_gib_s': bw, 'source_78layer_ttft_ms': ttft,
             'blocks_per_layer': count, 'disk_volume_by_stripe_offset': by_offset,
             'disk_rate_by_stripe_offset': [[x*1000/c for x in row] for row in by_offset],
             'total_V_over_C_gib_s': count*IO*1000/c,
             'any_offset_disk_rate_upper_gib_s': math.ceil(count/SSU)*IO*1000/c,
             'client_layer_emission_last_issue_ms': (count-1)*ISSUE_INTERVAL_MS,
             'category': ('S' if key[0]<=SHORT_SEQ_MAX_K else 'L')+('S' if key[1]<512 else 'L'),
             'post_admission_link_lower_ms': c+7*count*IO/LINK*1000,
             'request_fourth_completion_pure_C_ms': 32*c}
        p['link_nominal_strictly_underloaded'] = p['total_V_over_C_gib_s'] < LINK
        p['underload33_eligible'] = 32*p['any_offset_disk_rate_upper_gib_s'] < DISK
        out.append(p)
    assert len(out) == 84 and sum(p['underload33_eligible'] for p in out) == 33
    return out


def static_vector(profiles, n_start, n_end):
    """Actual canonical offset n//4; independently maximize allowed profiles per NPU/disk."""
    return [math.fsum(max(p['disk_rate_by_stripe_offset'][(n//4)%SSU][s] for p in profiles)
                      for n in range(n_start, n_end)) for s in range(SSU)]


def detail(shorts, long, n_long):
    ns = NPU-n_long
    lv = static_vector([long], 0, n_long)
    sv = static_vector(shorts, n_long, NPU)
    bound = [x+y for x,y in zip(lv, sv)]
    work = [math.fsum(long['disk_volume_by_stripe_offset'][(n//4)%SSU][s] for n in range(n_long))
            for s in range(SSU)]
    service = [x/DISK*1000 for x in work]
    d, cmax, csum = max(service), max(s['C_ms'] for s in shorts), sum(s['C_ms'] for s in shorts)
    f = ns/NPU
    return {'short_keys': [s['key'] for s in shorts], 'long_key': long['key'],
            'n_long': n_long, 'n_short': ns, 'short_categories': [s['category'] for s in shorts],
            'long_category': long['category'], 'short_C_ms': [s['C_ms'] for s in shorts], 'long_C_ms': long['C_ms'],
            'short_physical_V_gib': [s['physical_V_gib'] for s in shorts], 'long_physical_V_gib': long['physical_V_gib'],
            'actual_offset_static_peak_by_disk_gib_s': bound,
            'actual_offset_static_peak_gib_s': max(bound),
            'any_offset_static_peak_gib_s': n_long*long['any_offset_disk_rate_upper_gib_s']+ns*max(s['any_offset_disk_rate_upper_gib_s'] for s in shorts),
            'max_link_gib_s': max(p['total_V_over_C_gib_s'] for p in shorts+[long]),
            'fixed_roles_capacity_pass': max(bound)<DISK and all(p['link_nominal_strictly_underloaded'] for p in shorts+[long]),
            'any_role_mixing_universal_certificate_pass': 32*max(p['any_offset_disk_rate_upper_gib_s'] for p in shorts+[long])<DISK,
            'hypothetical_synchronous_long_wave_service_ms_by_disk': service,
            'long_wave_Dmax_over_long_C': d/long['C_ms'],
            'long_wave_Dmax_over_short_C': [d/s['C_ms'] for s in shorts],
            'long_client_emission_last_issue_ms': long['client_layer_emission_last_issue_ms'],
            'scores_are_heuristics_not_utilization_predictions_or_bounds': True,
            'score_zero_allowance_pp': 100*f*d/long['C_ms'],
            'score_one_short_C_allowance_pp': 100*f*max(0, d-cmax)/long['C_ms'],
            'score_two_short_C_allowance_pp': 100*f*max(0, d-2*cmax)/long['C_ms'],
            'score_two_C_equal_short_counts_C_weighted_pp': 100*f*sum(s['C_ms']*max(0,d-2*s['C_ms']) for s in shorts)/csum/long['C_ms'],
            'all_profiles_link_alone_SLO15_possible': all(p['post_admission_link_lower_ms'] <= 12*p['C_ms'] for p in shorts+[long]),
            'long_fourth_completion_pure_C_ms': long['request_fourth_completion_pure_C_ms'],
            'long_fourth_by_1500_mathematically_possible': long['request_fourth_completion_pure_C_ms'] <= 1500,
            'suggested_6000ms_population_per_card': {'long_repeats': math.ceil(6000/(8*long['C_ms'])),
                'equal_short_counts_each': math.ceil(6000/(8*csum))},
            'scope': 'Fixed long NPUs 0..n_long-1 and short NPUs n_long..31, stripe offset n//4. Every short card may choose any listed short at any instant. Additional cross-request L0 excluded by requested nominal definition.'}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    profiles = catalog()
    shorts = [p for p in profiles if p['key'][0]<=SHORT_SEQ_MAX_K and p['link_nominal_strictly_underloaded']]
    longs = [p for p in profiles if p['key'][0]>SHORT_SEQ_MAX_K and p['link_nominal_strictly_underloaded']]
    long_vectors = {(p['index'], n): static_vector([p], 0, n) for p in longs for n in range(1, NPU)}
    rows, attempted, passing = [], Counter(), Counter()
    for size in (1,3):
        for ss in itertools.combinations(shorts, size):
            # Separate real compute duration and I/O size roles, not seq labels alone.
            lc = [l for l in longs if l['C_ms'] > max(s['C_ms'] for s in ss)
                  and l['physical_V_gib'] > max(s['physical_V_gib'] for s in ss)]
            short_vectors = {n:static_vector(ss,n,NPU) for n in range(1,NPU)}
            cmax=max(s['C_ms'] for s in ss)
            for l in lc:
                for n in range(1,NPU):
                    attempted[str(size)] += 1
                    lv = long_vectors[l['index'],n]
                    peak = max(x+y for x,y in zip(lv,short_vectors[n]))
                    if peak >= DISK:
                        continue
                    passing[str(size)] += 1
                    d=max(lv)*l['C_ms']/DISK
                    scale=100*(NPU-n)/NPU/l['C_ms']
                    two=scale*max(0,d-2*cmax)
                    one=scale*max(0,d-cmax)
                    zero=scale*d
                    rows.append((two,one,zero,n,l['index'],tuple(s['index'] for s in ss),peak))
    rows.sort(reverse=True)
    with (HERE/'math_search_all_feasible.csv.gz').open('wb') as base:
        with gzip.GzipFile(fileobj=base,mode='wb',mtime=0) as gz:
            import io
            with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['rank_by_two_C_score','short_profile_indices','long_profile_index','n_long','n_short',
                                 'fixed_offset_static_peak_gib_s','heuristic_two_C_score_pp','heuristic_one_C_score_pp','heuristic_zero_C_score_pp'])
                for rank,r in enumerate(rows,1):
                    two,one,zero,n,li,si,peak=r
                    writer.writerow([rank,':'.join(map(str,si)),li,n,NPU-n,peak,two,one,zero])
    def expand(r):
        return detail([profiles[i] for i in r[5]],profiles[r[4]],r[3])
    groups={
        'single_any_short': [r for r in rows if len(r[5])==1],
        'single_strict_SS': [r for r in rows if len(r[5])==1 and all(profiles[i]['category']=='SS' for i in r[5])],
        'triple_any_short': [r for r in rows if len(r[5])==3],
        'triple_strict_SS': [r for r in rows if len(r[5])==3 and all(profiles[i]['category']=='SS' for i in r[5])],
        'any_mixed_order_universal_underload': [r for r in rows if 32*max(profiles[i]['any_offset_disk_rate_upper_gib_s'] for i in r[5]+(r[4],))<DISK],
    }
    by_key={tuple(p['key']):p for p in profiles}
    recipes=[
        ('clean_fixed_vs_mixed_16L',[(32,1024)],(160,1024),16),
        ('clean_fixed_vs_mixed_23L',[(32,1024)],(160,1024),23),
        ('old_raw_quartet_fixed_16L',[(32,1024),(48,1024),(64,1024)],(160,1024),16),
        ('large_long_wave_single_SL',[(32,1024)],(200,1024),23),
        ('more_short_cards_single_SL',[(32,512)],(200,2048),19),
        ('strict_SS_single',[(32,256)],(192,2048),26),
        ('strict_SS_triple',[(32,256),(48,256),(64,256)],(192,2048),27),
        ('varied_SL_triple',[(32,512),(48,512),(64,512)],(200,2048),21),
    ]
    recommendations=[dict(name=name, **detail([by_key[k] for k in ss],by_key[l],n)) for name,ss,l,n in recipes]
    current_recipes=[
        ('root_raw160_triple_16L',[(32,1024),(48,1024),(64,1024)],(160,1024),16),
        ('root_raw160_single_16L',[(32,1024)],(160,1024),16),
        ('root_raw2001024_single_19L',[(32,1024)],(200,1024),19),
        ('root_raw2002048_single_17L',[(32,512)],(200,2048),17),
        ('root_paired_raw160_triple_20L',[(32,1024),(48,1024),(64,1024)],(160,1024),20),
        ('root_paired_raw2001024_triple_20L',[(32,1024),(48,1024),(64,1024)],(200,1024),20),
    ]
    current_cases=[dict(name=name, **detail([by_key[k] for k in ss],by_key[l],n)) for name,ss,l,n in current_recipes]
    observed_path = HERE/'followup_results.json'
    observed_snapshot = None
    if observed_path.exists():
        observed_bytes = observed_path.read_bytes()
        observed = json.loads(observed_bytes)
        selected = {'raw160_three_l16_fixed_seed7','raw160_one_l16_fixed_seed7',
                    'raw200_one_l19_fixed_seed7','raw200_fast_l17_fixed_seed7'}
        observed_snapshot = {'source':str(observed_path),
            'source_sha256':hashlib.sha256(observed_bytes).hexdigest(),
            'scope':'Read-only snapshot from parent event analysis; not simulated or independently re-aggregated by this mathematical enumerator.',
            'rows':[r for r in observed['rows'] if r['label'] in selected and r['strategy']=='baseline']}
    sources=['data','continuous_batch_sim.py','sim.py','continuous_prefill_client.py','shared_path_baseline.py',
             'run_shared_path_experiments.py','shared_path_sim_adapter.py']
    result={'no_simulation_run_by_this_script':True,'script_sha256':sha(__file__),
        'source_sha256':{str(ROOT/name):sha(ROOT/name) for name in sources},
        'prior_reports_sha256':{str(p):sha(p) for p in (STUDY/'raw_data_random_5seeds/comparison.md',STUDY/'raw_quartet_replacement/comparison.md')},
        'constants':{'NPU':NPU,'SSU':SSU,'disk_GiB_s':DISK,'link_GiB_s':LINK,'IO_bytes':176*1024,'issue_interval_ms':ISSUE_INTERVAL_MS},
        'catalog':profiles,'catalog_counts':{'raw':len(profiles),'link_safe_short':len(shorts),'link_safe_long':len(longs),'underload33':33},
        'enumeration_attempts_after_C_V_role_filter':dict(attempted),'feasible_counts':dict(passing),
        'all_feasible_csv':str(HERE/'math_search_all_feasible.csv.gz'),'all_feasible_csv_sha256':sha(HERE/'math_search_all_feasible.csv.gz'),
        'top8_by_group':{k:[expand(r) for r in v[:8]] for k,v in groups.items()},
        'alternate_score_top8_single':{str(allowance):[expand(r) for r in sorted(groups['single_any_short'], key=lambda r:r[2-allowance],reverse=True)[:8]] for allowance in (0,1)},
        'recommended_recipes':recommendations, 'parent_already_running_recipes':current_cases,
        'observed_parent_baseline_snapshot':observed_snapshot,
        'heuristic_definition':'D = largest per-disk total service work if all fixed Long cards release one layer together; scores=100*nShort/32*max(0,D-k*CmaxShort)/CLong, k=0,1,2. No existence proof of blackout/phase ordering, no low-U prediction, no bound on actual U.',
        'scope_limit':'Exhaustive for 1 or 3 distinct short profiles (seq<=80K), one larger-C/larger-V long profile (seq>80K), nLong1..31, canonical contiguous roles and n//4 stripe. Excludes nominal per-card V/C>=50, arbitrary subset placement, 2 or >3 short types and role rebinding. Static max over short types is sufficient, not necessary for a specially constrained order.',
        'model_cautions':['Each command is 176KiB and nonpreemptive, not a full-layer atomic read.',
            'All ready clients issue interleaved one-block rounds every0.1us with shuffled within-round order.',
            'Current compute starts next-layer prefetch; last layer starts next request L0 prefetch.',
            'Same fixed long profile can preserve phase only if its I/O finishes within C; first phases and nonzero stalls need trace verification.',
            'Long2048 profiles often make fourth request before1500ms mathematically impossible; do not drop such results silently.',
            'A fixed-role certificate need not remain valid after redistributing Long/Short to all cards.']}
    (HERE/'math_search_candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 原始 data 固定长短卡：数学筛选与已完成补测的区分','',
        '本脚本只做数学枚举，不启动仿真。父研究已完成若干原始画像补测，观测单独列在末尾；下方容量证书与启发式记分不由这些U拟合。', '',
        '直接解析全部84行。32卡/6盘，每盘40 GiB/s、每卡接收50 GiB/s；单层C来自原data，实际命令176KiB。若原行尾块88KiB需padding，则容量使用完整176KiB物理量；本轮推荐均整块对齐。', '',
        '角色设定：NPU 0..nLong−1仅一个Long画像，其余卡可任意选择1或3种Short画像。Short按seq≤80K，Long按seq>80K，另要求CLong大于所有CShort且VLong大于所有VShort。SS/SL等实际策略类别单独记录。每卡每盘按stripe offset=npu//4计算；对每卡允许的短画像先取盘上V/C最大，再跨卡相加。只保留每盘严格<40、每卡严格<50的配置。这个证书覆盖固定角色下任意内部顺序；不自动覆盖改为每卡混合的对照。', '',
        f"共检查1短画像配置 {attempted['1']:,} 个、3短画像配置 {attempted['3']:,} 个；固定条带静态界可行分别 {passing['1']:,}、{passing['3']:,} 个。全部参数与排序可由math_search.py复算，完整轻量表见math_search_all_feasible.csv.gz；详细候选见math_search_candidates.json。", '',
        '排序仅用于提出实验：假设全部Long同刻释放一层，令D为最热盘总服务工作；记分为100×短卡占比×max(0,D−k×最长短C)/长C，k=0/1/2。k表示给短卡预留的计算隐藏余量，不是对模拟器推导出的确定黑窗。实际发块会交织，短层可能早于或夹在Long命令中；因此这些分数不是U下降预测、上下界或可实现性证明。', '',
        '|建议|Long|Short|长/短卡|逐盘静态上界|长波D ms|短C ms|任意每卡混合也有通用证书|第四Long纯C≤1500ms|',
        '|---|---|---|---|---:|---:|---|---|---|']
    for r in recommendations:
        key=lambda k:f'{k[0]}K/{k[1]}'
        lines.append(f"|{r['name']}|{key(r['long_key'])}|{','.join(key(k) for k in r['short_keys'])}|{r['n_long']}/{r['n_short']}|{r['actual_offset_static_peak_gib_s']:.6f}|{max(r['hypothetical_synchronous_long_wave_service_ms_by_disk']):.6f}|{','.join(f'{c:.6f}' for c in r['short_C_ms'])}|{r['any_role_mixing_universal_certificate_pass']}|{r['long_fourth_by_1500_mathematically_possible']}|")
    lines += ['', '父研究已启动配置的数学对照（仅分析这些已定配置，不追加实验计划）：', '',
        '|配置|固定角色盘界 GiB/s|Long波D ms|Short C范围 ms|分数k=0 / 1 / 2（非U预测）|mixed任意组合通用证书|',
        '|---|---:|---:|---|---|---|']
    for r in current_cases:
        lines.append(f"|{r['name']}|{r['actual_offset_static_peak_gib_s']:.6f}|{max(r['hypothetical_synchronous_long_wave_service_ms_by_disk']):.6f}|{min(r['short_C_ms']):.6f}–{max(r['short_C_ms']):.6f}|{r['score_zero_allowance_pp']:.3f} / {r['score_one_short_C_allowance_pp']:.3f} / {r['score_two_short_C_allowance_pp']:.3f}|{r['any_role_mixing_universal_certificate_pass']}|")
    lines += ['', '分数对所留隐藏余量很敏感，尤其160K/1024的16长卡D≈14.23ms，小于2×最短C≈14.51ms，k=2得0，k=1却为正。这不意味着实际不可能下降，也不意味着k=1所示幅度可实现；它说明需要实际相位/发块轨迹，不能由D>C这一条比较预测10个百分点。三短画像用最长C计算主分数更保守，JSON另保留等条数时按纯C加权的版本；二者也都不是概率模型。', '',
        '筛选偏差：全部原84行都进入目录，但最终角色组合限定1或3种Short、单一Long、32卡固定角色、统一条带及单卡名义低于50；并按人为同步黑窗启发式挑高分，因此不是业务分布抽样或稳健性证明。原始C/V没有为低U拟合。raw33与本搜索的条件不同：33保证所有卡任意挑画像，而本搜索可以通过限制Long/Short卡数容纳目录外画像。当前mixed配对若含200K/1024，固定比例的证书不能直接继承；失败的mixed轨迹必须保留并单独标记容量条件。']
    lines += ['',
        '优先把160K/1024 Long +32K/1024 Short的16长、23长作为干净fixed/mixed配对：这两个画像任意32卡组合也有欠载证书。200K/1024 Long同波更大，但32个Long同在计算时略超总容量，因此mixed对照必须实扫，不能继承fixed证书。严格SS采用32K/256（或32/48/64K、NQL256）搭配192K/2048，需要很多Long卡才能欠载；短卡少，整机U的损失会被Long卡计算掩盖。NQL2048 Long的第四请求纯C超过1500ms，若保留该暖机规则应明确标记，而不是筛掉。', '',
        '为什么先前raw输入不能代替本轮：raw33筛选要求每个画像任意复制32次都安全，目录无SS；它的每卡混合会改变长请求画像与计算相位。可行raw quartet也在每卡混合长短，且三Short已经是SL，其C为7–13ms。固定角色使相同Long层周期有可能保持相位，这正是本次固定角色补测检验的机制差别。raw84五个冻结样本全部名义过载，不能用高U或低SLO来证明欠载下的FIFO结论。', '',
        '实现依据：shared_path_baseline.py:4把全部块送Path0；sim.py:596追加FIFO、:612取队头。continuous_prefill_client.py:22–23设单块提交、0.1us间隔；continuous_batch_sim.py:3063打乱同刻客户端顺序，:3071–3089按轮发块，故不是完整Long层一次占盘。该文件:3303–3326在当前层compute_start预取下一层，最后一层预取下一请求L0；:3296按前驱compute_end至本层compute_start计算barrier等待。跨请求L0不额外计入本任务current V/C，但会影响实际读取轨迹。', '',
        '必须在仿真中验证的量：整窗32卡active、逐事件每盘及链路名义峰值、Long逐层相位漂移/是否stall、Short实际暴露等待、设备U和接纳后SLO。名义欠载不是突发deadline保证，read lifetime不是SSD busy，数学大D也不能确定物理FIFO队头身份。保持6000ms以上纯C的简单人口建议已写入JSON；不保证第四请求暖机或所有实际TTFT达标。']
    if observed_snapshot is not None:
        lines += ['', '已完成的 seed7 Baseline 补测（只读父分析快照，不是上述分数的预测）：', '',
            '|实际case|设备U %|暖接纳SLO %|全32active|全程名义欠载|第四请求≤1500ms|',
            '|---|---:|---:|---|---|---|']
        for r in observed_snapshot['rows']:
            lines.append(f"|{r['label']}|{r['device_utilization_percent']:.6f}|{r['warm_slo_percent']:.6f}|{r['all_npus_active']}|{r['full_run_capacity_pass']}|{r['fourth_request_by_1500']}|")
        lines += ['', '这些原始data固定角色输入已经产生实际U下降，不能再称其“尚未模拟”，也不能把旧raw quartet未复现外推为原始data不存在反例。它们仍是选择过的固定角色单seed输入；每卡混合配对与多seed复验按父研究已冻结计划继续，不能据当前单seed把幅度推广到任意输入。观测源为[followup_results.json](followup_results.json)，本次读取时的SHA与每份结果path/SHA一并保存在math_search_candidates.json的observed_parent_baseline_snapshot。后续最新观测应读父报告，数学筛选表不承担实时进度显示。']
    (HERE/'math_search.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'attempted':dict(attempted),'passing':dict(passing),'files':[str(HERE/name) for name in ('math_search.md','math_search_candidates.json','math_search_all_feasible.csv.gz')]},ensure_ascii=False))


if __name__=='__main__':
    main()
