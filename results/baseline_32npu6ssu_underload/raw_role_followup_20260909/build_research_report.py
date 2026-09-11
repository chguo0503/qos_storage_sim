#!/usr/bin/env python3
"""Build the fixed/random-mixed report after all declared runs complete."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[1]


def fmt(values):
    return f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}'


def main():
    result=json.loads((HERE/'followup_results.json').read_text())
    assert result['completed']==result['planned'] and result['completed']>=48 and not result['errors'] and not result['pending']
    rows=result['rows']
    assert all(r['audit_pass'] for r in rows)
    with (HERE/'existing_comparison.csv').open() as f:old=list(csv.DictReader(f))
    main_names=['raw160_three_l20','raw200_three_l20']
    primary=[r for r in rows if r['spec_name'] in main_names]
    assert len(primary)==40
    assert all(r['active_underload_valid'] and r['fourth_request_by_1500'] for r in primary), 'Report assumptions changed: retain and explain failed conditions, never filter seeds'
    for name in main_names:
        for mode in ('fixed','mixed'):
            for policy in ('baseline','once'):
                rr=[r for r in primary if (r['spec_name'],r['mode'],r['strategy'])==(name,mode,policy)]
                assert len(rr)==5 and {r['seed'] for r in rr}=={7,19,43,67,101}
    def sample(name,mode,policy,field):
        return [r[field] for r in primary if (r['spec_name'],r['mode'],r['strategy'])==(name,mode,policy)]
    s=lambda mode,policy,field:sample('raw200_three_l20',mode,policy,field)
    drop=statistics.mean(s('mixed','baseline','device_utilization_percent'))-statistics.mean(s('fixed','baseline','device_utilization_percent'))
    gap=statistics.mean(s('fixed','once','device_utilization_percent'))-statistics.mean(s('fixed','baseline','device_utilization_percent'))
    text=['**原始 data 能否使 Baseline 下降，以及固定／混合分卡的对照**','',
      f'可以。本次直接使用 data 原行，未缩放 C、V，也未填充读取量。主反例使用三种不同短请求和一种长请求：同一批请求改为20张长卡／12张短卡后，Baseline 相比每卡混合下降 **{drop:.2f} 个百分点**；固定分卡时 Once 比 Baseline 高 **{gap:.2f} 个百分点**。这是五个预定种子的均值，单次探索结果另列。','',
      '此前“原始可行四画像没有复现下降”是正确的实验观察；把它推广为“data中的数据无法降低Baseline”则没有依据。新测试扩展了分配方式和原始长请求选择，并保留原始数值。','',
      '后续已补测每张卡都执行长短请求的定向排序；五种子同人口最终对照见[三行汇总](final_comparison.md)，方法、失败探索和数量匹配见[混合输入报告](mixed_research_report.md)。','',
      '**统一统计口径**','',
      '32 NPU、6 SSU、每盘40 GiB/s、每卡链路50 GiB/s；每请求8层；统计[2000,4000) ms。五种子为7、19、43、67、101，表中均为“百分比均值 ± 样本标准差（百分点）”。新主组seed7是先行配对探索，其余四seed在追加运行前声明，用来确认稳定性；不能把全部五seed称为未见过的独立验证集。先分别算每个种子的率再取平均，不按请求数合并种子。',
      '',r'$U=\frac{\sum_n |\mathrm{compute}_n\cap[2000,4000)|}{32\times2000}$。',
      '',r'$\mathrm{SLO}=\frac{\#\{r:2000\le a_r<4000,\ f_r-a_r\le1.5\times8C_r\}}{\#\{r:2000\le a_r<4000\}}$。',
      '', '**这里的 TTFT SLO 沿用之前的接纳后处理时间代理。** a是admission、f是完成、C是该请求单层纯计算；所有暖窗接纳请求均跟踪至完成，包含4秒后完成者。它不含arrival到admission的排队，不等于真实用户端到端TTFT。所有输入arrival为0，不能把暖窗到达SLO当成这张表。',
      '', 'Once指Once per layer（实现名once），共享5ms快照、静态CIR，不是NewOnce。全部策略均固定NPU绑定；“混合”指每张卡执行长短请求，“固定长短卡”指不同卡专门执行不同角色，没有运行时迁移。','',
      '分卡方式和策略都会改变暖窗内实际接纳的请求集合，SLO表比较的是同一时间窗的服务表现，分母不是严格同一批请求ID；全局请求人口、C/V、到达和实际落盘仍严格配对。','',
      '**1. 新补测：原始 data 的同人口固定／混合比较**','',
      '| 原始长画像 | 输入分配 | Baseline NPU U% | Once NPU U% | Baseline TTFT SLO% | Once TTFT SLO% |',
      '|---|---|---:|---:|---:|---:|']
    table_rows=[]
    for name in main_names:
        for mode in ('mixed','fixed'):
            desc='每卡长短混合，独立随机' if mode=='mixed' else '20长卡／12短卡，短卡独立随机'
            vals=[fmt(sample(name,mode,p,f)) for f in ('device_utilization_percent','warm_slo_percent') for p in ('baseline','once')]
            text.append('| '+' | '.join([('160K/1024' if '160' in name else '200K/1024'),desc,*vals])+' |')
            table_rows.append(dict(profile=name,mode=mode,**{f'{p}_{f}':statistics.mean(sample(name,mode,p,f)) for p in ('baseline','once') for f in ('device_utilization_percent','warm_slo_percent')}))
    text+=['','“同一批请求”指同一长画像配置内部的fixed/mixed与Baseline/Once配对。160K组全局1252条，200K组1172条；两组因为单条长计算时间不同，按相同的每卡纯计算时长规则确定了不同长请求条数，不能当作跨长画像的同人口消融。']
    valid=sum(r['active_underload_valid'] for r in primary)
    fourth=sum(r['fourth_request_by_1500'] for r in primary)
    warmmixed=[r['warm_mixed_card_count'] for r in primary if r['mode']=='mixed']
    text+=['',f'主组40格中，**{valid}/40** 通过全32卡暖窗active及全程逐盘/链路欠载；**{fourth}/40** 也满足旧的“每卡第4条请求在1500ms前完成”条件。混合组暖窗内长短各有正计算时间的卡数为{min(warmmixed)}–{max(warmmixed)}/32；专用分卡有意不要求每卡混合。','',
      '| 长画像与分配 | 输入静态逐盘上界GiB/s | 10次运行中全程名义峰值的最大值GiB/s | 10次运行任一盘超40时长之和 |',
      '|---|---:|---:|---:|']
    for name in main_names:
        for mode in ('mixed','fixed'):
            rr=[r for r in primary if r['spec_name']==name and r['mode']==mode]
            text.append(f"| {name} / {mode} | {max(r['static_max_ssu_gib_s'] for r in rr):.6f} | {max(r['max_ssu_nominal_gib_s'] for r in rr):.6f} | {sum(r['any_ssu_over_capacity_ms'] for r in rr):.6f} ms |")
    text+=['','静态上界允许各卡任意选择自己队列内的画像，是充分证书。200K长画像的混合方案静态上界略超过40，不表示这些实际运行过载：表中第三列由每次实际运行全程逐事件扫描取得。固定方案静态上界低于40，因而对该分卡方式的任何短卡内部排序都有名义欠载保证。',
      '', '“需求”是每张卡当前接纳请求的单层盘读取量除以单层C，再逐盘相加；按此前约定不额外叠加下一请求首层预取项。仿真仍保留全部真实读取。它不是物理吞吐、I/O突发到达包络或无排队保证。','',
      '**2. 主反例到底输入了什么**','',
      '| 角色 | data键 | 每层C ms | 每层V MiB | V/C GiB/s | 8层纯计算ms |',
      '|---|---|---:|---:|---:|---:|']
    raw=ast.literal_eval((ROOT/'data').read_text())
    for role,key in [('S1',(32,1024)),('S2',(48,1024)),('S3',(64,1024)),('L',(200,1024))]:
        _,us,_,v=raw[key];c=us/1000
        text.append(f'| {role} | {key[0]}K/{key[1]} | {c:.6f} | {v*1024:.6f} | {v*1000/c:.6f} | {8*c:.6f} |')
    text+=['','固定20长／12短的主反例：NPU0–19每卡19条L；NPU20–31每卡S1/S2/S3各22条，共66条短请求。全局是L380条、每种短画像264条，共1172条请求。短卡各自打乱整份队列；长卡全是相同长画像。','',
      '混合方案把这同一批1172条原请求按画像均匀分给全部32卡，再分别打乱各卡完整队列。每张卡每种短画像8或9条、长画像11或12条，余数由预定种子决定。每条请求的C、V、arrival和逐块SSU落盘位置全部保留。两方案只是绑定与卡内排列不同，没有重新采样请求。','',
      '160K组固定时每张长卡23条L、每张短卡三种各22条，全局1252条；混合时每卡每种短画像8或9条、长画像14或15条。data四字段依次是需求带宽GiB/s、单层计算微秒、78层参考TTFT毫秒、单层读取GiB；78层参考TTFT没有用于本实验八层SLO。','',
      '三种“短”是相对长画像的计算时间而言；按模拟器的seq_len/NQL分类，它们属于SL，长画像属于LL。不要把“短计算”一律等同于路由类别SS。','',
      '**3. 为什么之前那组接近100%，这次能下降**','',
      '首先，是否有长短标签不是关键；关键是短卡能用多少计算时间覆盖读取等待，以及长读取是否反复集中到达每块SSD各自的Path0 FIFO。其他卡的长计算本身不会占用这张短卡的计算资源，造成等待的是尚未完成的I/O。','',
      '之前可行四画像的短C为7.26–12.63ms，比外推构造的0.53–0.90ms长很多，实测等待很少。较长的短C提供了更多预取时间，是合理解释之一；但替换同时改变了长短计算比、配额、路由类别及相位，不能只从接近100%的U唯一归因。它不是一个无效实验，但不是搜尽坏输入的结果。','',
      '本次专用长卡持续执行相同的L，长卡的相位更容易长期集中。以主反例作数量级计算：200K/1024每层有1592个176KiB块，每盘约265–266块；20张长卡的一层累计工作在一块40GiB/s盘上约需22.24–22.32ms。这是假设一波长读取形成先行工作时的数量级，不是实测排在某个短请求前面的工作量。短C只有7.26–12.63ms，而长C为35.44ms，所以同一波读取对短卡可能成为暴露的等待，对长卡仍可藏在计算中。',
      '',r'对内部层L1–7，日志恒等式为 $\mathrm{stall}=\max(0,\ \text{本层读取就绪时刻}-\text{前层计算结束时刻})$。它不预测读取就绪时刻，也不是FIFO等待上界或精确利用率公式；具体请求前面有哪些块，需要逐块服务轨迹才可归因。',
      '', '层日志核查见 [机制证据](mechanism_evidence.md)。它量化长卡相位集中及短卡等待，且明确区分读取生命周期与SSD连续服务。',
      '', '另外，之前33行安全子集要求“32卡任意抽到任意允许画像都欠载”，这是一个很强的充分条件。固定角色限制各画像同时活跃的卡数，因此即使某画像不能承受32卡全用，也可能与受限数量的其他画像组合欠载。本主反例中，被33行筛选排除的是200K长画像，三个短画像原本就在安全子集中；长画像的V/C也高于三个短画像。数学枚举与全部候选见 [数学筛选](math_search.md)。','',
      '**4. 之前构造请求的固定／混合完整对照**','',
      '下面保持原19,456条构造请求：1K/128、1K/256、1K/384各6400条，192K/768共256条。这张表不能和上面的原始data表当作同一输入。','',
      '| 分配与顺序 | Baseline NPU U% | Once NPU U% | Baseline TTFT SLO% | Once TTFT SLO% |',
      '|---|---:|---:|---:|---:|']
    for cond in ['mixed_random','mixed_ordered','fixed_11_random','fixed_11_round_robin','fixed_6_random','fixed_6_round_robin']:
        rr=[r for r in old if r['condition']==cond]
        if not rr:continue
        by={r['strategy']:r for r in rr}
        vals=[f"{float(by[p][f+'_mean']):.2f} ± {float(by[p][f+'_sample_sd']):.2f}" for f in ('device_utilization_percent','warm_admission_slo_percent') for p in ('baseline','once')]
        text.append('| '+' | '.join([rr[0]['condition_name'],*vals])+' |')
    # Never silently omit role configurations if source keys changed.
    assert sum(line.startswith('| ') for line in text[text.index('**4. 之前构造请求的固定／混合完整对照**'):])==7
    text+=['','混合Random分别打乱每张卡的完整长短队列；固定角色Random分别打乱各短卡的完整短队列。固定角色的轮转是S1→S2→S3，与每卡混合Ordered的相位构造不是同一个操作。完整来源与旧raw均匀抽样结果见 [已有实验比较](existing_comparison.md)。','',
      '**5. 结论的边界与容易忽略的变量**','',
      '- 这证明存在原始data欠载输入可使Baseline下降，不能据此说真实业务或所有固定分卡都会如此。输入arrival全部为0，长卡重复同一长画像，是针对机制的饱和队列实验。',
      '- 新强反例针对长短角色专用分卡；本次每卡长短混合的原始data随机输入仍接近满利用率。没有把固定角色结果冒充“每张卡暖窗都有长短、只重排卡内顺序”的反例。',
      '- 固定20长／12短并不是搜索出的全局最优，也没有证明任何卡内顺序都同样差。五种子检验随机队列稳定性，不能代替生产到达分布。',
      '- 同人口仍不等于每卡负载相同：角色分卡会改变相位及完成尾部。主表只算全32卡active的同一暖窗，避免把尾部空卡当成FIFO损失。',
      '- SLO×1.5与U不等价。例如原始160K/1024长、32K/1024短各16张卡，单seed Baseline U87.82%而SLO100%；一定量的等待仍可能落在1.5倍预算内。',
      '- 分卡方式和策略会改变暖窗内实际接纳的请求集合，因此暖SLO的分母不是严格同一批ID。各次人数、通过数和4秒后完成数均保存在结果CSV/JSON；不能把所有SLO差异都解释为同一组请求逐条改善。',
      '- 原始2048长画像的探索组无法在1500ms前完成4条，原因是纯计算就超预算；该组单独保留，没有并入上面满足旧暖机判据的五seed主组。','',
      '**可复核文件**','',
      f'[全部{result["planned"]}格分卡逐seed结果](followup_per_seed.csv)；[新实验JSON](followup_results.json)；[全部探索结果表](followup_results.md)；[原选型审计](raw_selection_audit.md)；[独立新结果复核](crosscheck_followup.json)。',
      '', '新实验脚本为run_raw_followup.py，各预定spec和对应冻结plans、inputs、runs、command日志均保存在本目录；核心模拟器和策略代码未修改。','']
    (HERE/'research_report.md').write_text('\n'.join(text))
    manifest=dict(source_files={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        HERE/'followup_results.json',HERE/'existing_comparison.csv',HERE/'mechanism_evidence.md',Path(__file__))},
        primary_rows=40,total_new_runs=result['planned'],primary_tables=table_rows,baseline_fixed_vs_mixed_drop_pp=drop,
        fixed_once_minus_baseline_pp=gap)
    (HERE/'research_report_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(HERE/'research_report.md'),baseline_drop_pp=drop,fixed_once_gain_pp=gap),ensure_ascii=False))


if __name__=='__main__':main()
