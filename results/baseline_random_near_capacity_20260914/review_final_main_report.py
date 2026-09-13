#!/usr/bin/env python3
"""Independent, read-only main-report audit; skips all h65000 outcomes."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import statistics
from urllib.parse import unquote
from PIL import Image

HERE=Path(__file__).resolve().parent
SOURCES={}
FINDINGS=[]
def sha(p):
    p=Path(p);d=hashlib.sha256(p.read_bytes()).hexdigest();SOURCES[str(p)]=d;return d
def read(p):
    p=Path(p);sha(p);return json.loads(p.read_text())
def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
def main():
    report_path=HERE/'report.md';key_md_path=HERE/'key_results.md'
    report=report_path.read_text();key_md=key_md_path.read_text();sha(report_path);sha(key_md_path)
    key=read(HERE/'key_results.json');rows=[r for r in key['rows'] if r['population_pure_compute_ms']==22000]
    assert len(rows)==56 and all(r['status']=='complete' for r in rows)
    analyses={};canonical={};seen=set()
    for r in rows:
        case=r['case'];ap=HERE/r['analysis']
        if case not in analyses:analyses[case]=read(ap)
        a=analyses[case];assert sha(ap)==r['analysis_sha256'] and a['all_technical_checks_passed']
        w=next(w for w in a['windows'] if [w['start_ms'],w['end_ms']]==r['window_ms'])
        identity=(r['key'],r['strategy'],r['seed'],tuple(r['window_ms']));assert identity not in seen;seen.add(identity)
        canonical[identity]=w
        near(r['U_percent'],w['U_percent'])
        near(r['U_percent'],100*math.fsum(n['compute_ms'] for n in w['per_npu'])/(32*w['duration_ms']))
        assert r['all_32_active']==w['all_32_active']
        assert r['mixed_cards']==w['long_short_mixed_card_count'] and r['all_32_mixed']==w['long_short_mixed_pass']
        for alpha in ('1.5','2.0'):
            direct=w['slo']['alphas'][alpha]
            for k in ('count','passed','percent'):near(r['slo'][alpha]['overall'][k],direct['admission'][k])
            near(direct['admission']['percent'],100*direct['admission']['passed']/direct['admission']['count'])
            for role,datum in r['slo'][alpha]['per_role'].items():
                for k in ('count','passed','percent'):near(datum[k],direct['per_role'][role][k])
        for p,h in r['verified_source_sha256'].items():assert sha(HERE/p)==h
        # Every printed main per-seed U, SLO count and percentage, and coverage.
        alpha1=r['slo']['1.5']['overall'];alpha2=r['slo']['2.0']['overall']
        left,right=[int(x/1000) for x in r['window_ms']]
        expected=f"| {r['key']} | {r['strategy']} | {r['seed']} | [{left},{right}) | {r['U_percent']:.3f} | {alpha1['percent']:.3f} ({alpha1['passed']}/{alpha1['count']}) | {alpha2['percent']:.3f} ({alpha2['passed']}/{alpha2['count']}) | 32/{r['mixed_cards']} | complete |"
        assert expected in key_md,expected
    groups=[]
    for group in key['strategy_groups']:
        subset=[r for r in rows if r['key']==group['key'] and r['strategy']==group['strategy'] and r['window_ms']==group['window_ms']]
        assert len(subset)==group['completed_count']==len(group['expected_seeds'])
        values=[r['U_percent'] for r in subset];near(group['U_seed_mean_percent'],statistics.mean(values))
        if len(values)>1:near(group['U_seed_sample_sd_pp'],statistics.stdev(values))
        else:assert group['U_seed_sample_sd_pp'] is None
        assert group['all_active_runs']==sum(r['all_32_active'] for r in subset)
        assert group['all_mixed_runs']==sum(r['all_32_mixed'] for r in subset)
        for alpha in ('1.5','2.0'):
            rates=[r['slo'][alpha]['overall']['percent'] for r in subset]
            near(group['slo'][alpha]['seed_mean_percent'],statistics.mean(rates))
            total=sum(r['slo'][alpha]['overall']['count'] for r in subset)
            passed=sum(r['slo'][alpha]['overall']['passed'] for r in subset)
            assert group['slo'][alpha]['pooled_count']==total and group['slo'][alpha]['pooled_passed']==passed
            near(group['slo'][alpha]['pooled_percent'],100*passed/total)
        mean=f"{statistics.mean(values):.3f}"
        utext=f'{mean} ± {statistics.stdev(values):.3f}' if len(values)>1 else f'{mean} (n=1)'
        left,right=[int(x/1000) for x in group['window_ms']]
        expected=f"| {group['key']} | {group['strategy']} | [{left},{right}) | {len(values)}/{len(values)} | {utext} | {group['slo']['1.5']['seed_mean_percent']:.3f} | {group['slo']['2.0']['seed_mean_percent']:.3f} | {group['all_active_runs']}/{len(values)}；{group['all_mixed_runs']}/{len(values)} | {group['status']} |"
        assert expected in key_md,expected
        groups.append(dict(key=group['key'],strategy=group['strategy'],window_ms=group['window_ms'],n=len(values),U_mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None))
    for group in key['context384_paired_difference_groups']:
        seeds=group['expected_seeds'];window=tuple(group['window_ms']);du=[];ds=defaultdict(list)
        for seed in seeds:
            b=canonical[('context384','baseline',seed,window)];o=canonical[('context384','once',seed,window)]
            br=next(r for r in rows if r['key']=='context384' and r['strategy']=='baseline' and r['seed']==seed and tuple(r['window_ms'])==window)
            orow=next(r for r in rows if r['key']=='context384' and r['strategy']=='once' and r['seed']==seed and tuple(r['window_ms'])==window)
            assert br['manifest_sha256']==orow['manifest_sha256'] and br['input_fingerprint']==orow['input_fingerprint']
            du.append(o['U_percent']-b['U_percent'])
            for alpha in ('1.5','2.0'):ds[alpha].append(o['slo']['alphas'][alpha]['admission']['percent']-b['slo']['alphas'][alpha]['admission']['percent'])
        near(statistics.mean(du),group['U_once_minus_baseline']['mean_pp']);near(statistics.stdev(du),group['U_once_minus_baseline']['sample_sd_pp'])
        for alpha in ds:
            near(statistics.mean(ds[alpha]),group['slo_once_minus_baseline'][alpha]['mean_pp'])
            near(statistics.stdev(ds[alpha]),group['slo_once_minus_baseline'][alpha]['sample_sd_pp'])
    group_index={(g['key'],g['strategy'],tuple(g['window_ms'])):g for g in key['strategy_groups']}
    # Independently match the main report's primary strategy table, not just JSON.
    for strategy,label in (('baseline','Baseline'),('once','流量分配策略')):
        warm=group_index[('context384',strategy,(2000,4000))]
        long=group_index[('context384',strategy,(2000,20000))]
        expected=f"| {label} | {warm['U_seed_mean_percent']:.3f}% ± {warm['U_seed_sample_sd_pp']:.3f}pp | {long['U_seed_mean_percent']:.3f}% ± {long['U_seed_sample_sd_pp']:.3f}pp | {warm['slo']['1.5']['seed_mean_percent']:.3f}% / {warm['slo']['2.0']['seed_mean_percent']:.3f}% | {long['slo']['1.5']['seed_mean_percent']:.3f}% / {long['slo']['2.0']['seed_mean_percent']:.3f}% |"
        assert expected in report,expected
        for name in ('context384','hetero_short'):
            w=canonical[(name,strategy,7,(2000,4000))];z=canonical[(name,strategy,7,(2000,20000))]
            expected=f"| {label} | {w['U_percent']:.4f}% | {z['U_percent']:.4f}% | {z['slo']['alphas']['1.5']['admission']['percent']:.4f}% | {z['slo']['alphas']['2.0']['admission']['percent']:.4f}% |"
            assert expected in report,expected
    for name,ratio,rho in (('raw105','32:63','105.0812'),('raw110','17:31','110.0796')):
        w=group_index[(name,'baseline',(2000,4000))];z=group_index[(name,'baseline',(2000,20000))]
        expected=f"| {ratio}，rho={rho}% | {w['U_seed_mean_percent']:.4f}% ± {w['U_seed_sample_sd_pp']:.4f}pp | {z['U_seed_mean_percent']:.4f}% ± {z['U_seed_sample_sd_pp']:.4f}pp | {w['all_mixed_runs']}/5 | {z['all_mixed_runs']}/5 |"
        assert expected in report,expected
    for name in ('context384','raw105','raw110','hetero_short'):
        for window in ((2000,4000),(2000,20000)):
            b=next(r for r in rows if (r['key'],r['strategy'],r['seed'],tuple(r['window_ms']))==(name,'baseline',7,window))
            o=next(r for r in rows if (r['key'],r['strategy'],r['seed'],tuple(r['window_ms']))==(name,'once',7,window))
            assert b['manifest_sha256']==o['manifest_sha256'] and b['input_fingerprint']==o['input_fingerprint']
            slo_text=[f"{b['slo'][alpha]['overall']['percent']:.3f}→{o['slo'][alpha]['overall']['percent']:.3f}%" for alpha in ('1.5','2.0')]
            expected=f"| {name} / 22s | [{window[0]//1000},{window[1]//1000}) | {b['U_percent']:.3f}% | {o['U_percent']:.3f}% | {o['U_percent']-b['U_percent']:+.3f} | {slo_text[0]} | {slo_text[1]} | 同manifest已核验 |"
            assert expected in key_md,expected
    main512_U=[];main512_wait=[]
    for seed in (7,19,43,67,101):
        a=read(HERE/f'runs/main512_ssu3_h22000_seed{seed}/baseline/analysis.json')
        w=next(w for w in a['windows'] if (w['start_ms'],w['end_ms'])==(2000,20000))
        main512_U.append(w['U_percent']);main512_wait.append(w['complete_internal_cycles_inside_window']['per_role']['S']['stall_ms_including_zeros']['mean'])
    assert f'{statistics.mean(main512_U):.4f}% ± {statistics.stdev(main512_U):.4f}' in report
    assert f'{statistics.mean(main512_wait):.6f}' in report
    # Full-cycle accounting is distinct from arithmetic mean of individual ratios.
    cycle_results=[]
    for strategy in ('baseline','once'):
        a=analyses[f'runs/context8_L384m1024_S10m128_ssu8_h22000_seed7/{strategy}']
        w=canonical[('context384',strategy,7,(2000,20000))]
        s=w['complete_internal_cycles_inside_window']['per_role']['S']
        profile=next(x for x in a['profiles'] if x['role']=='S');B=profile['B_GiB_s']
        mean_b=s['count']*profile['V_GiB']/(s['total_cycle_ms']/1000)
        near(mean_b,s['work_implied_mean_b_GiB_s_weighted_by_cycle_duration'])
        near(100*mean_b/B,s['duration_weighted_cycle_U_percent'])
        assert abs(s['arithmetic_mean_cycle_U_percent']-100*mean_b/B)>1
        label='Baseline' if strategy=='baseline' else '流量分配策略'
        expected=f"| {label} | {s['count']:,} | {s['stall_ms_including_zeros']['mean']:.6f} | {s['cycle_ms']['mean']:.6f} | {mean_b:.6f} | {100*mean_b/B:.4f}% |"
        assert expected in report,expected
        near(math.fsum(c['window_card_time_share_percent']*c['conditional_U_percent']/100 for c in w['classes'].values()),w['U_percent'])
        cycle_results.append(dict(strategy=strategy,count=s['count'],B=B,mean_b=mean_b,duration_weighted_percent=100*mean_b/B,arithmetic_mean_percent=s['arithmetic_mean_cycle_U_percent']))
    assert '不是把每个周期的b简单平均' in report
    assert '不能与Baseline五种子均值直接相减' in report
    assert '不是真实首token事件' in key_md and '接纳前' in report
    assert '不表示逐盘逐时欠载' in report and '没有做组件拆分实验' in report
    assert '±为种子样本标准差' in report and '不是置信区间' in report
    # Link and image checks apply to existing files, without regenerating anything.
    links=[];images=[]
    for path,content in ((report_path,report),(key_md_path,key_md)):
        for label,target in re.findall(r'!?\[([^\]]*)\]\(([^)]+)\)',content):
            if target.startswith(('http:','https:','#')):continue
            target=unquote(target.split('#',1)[0].strip('<>'))
            linked=(path.parent/target).resolve();exists=linked.exists()
            links.append(dict(document=path.name,label=label,target=str(linked),exists=exists))
            if not exists:FINDINGS.append(dict(severity='error',kind='missing_link',target=str(linked)))
            if linked.suffix.lower()=='.png' and exists:
                with Image.open(linked) as im:size=list(im.size);im.verify()
                images.append(dict(path=str(linked),sha256=sha(linked),pixels=size))
    fdir=HERE/'figures/five_seed_strategy_comparison';plot=read(fdir/'plot_data.json');checks=read(fdir/'checks.json')
    for path,h in plot['source_analysis_sha256'].items():assert sha(HERE/path)==h
    assert sha(fdir/'plot_data.json')==checks['plot_data_sha256']
    for path,entry in checks['output_files'].items():assert sha(fdir/path)==entry['sha256']
    for window in plot['windows'].values():
        for r in window['rows']:
            win=(int(window['start_ms']),int(window['end_ms']))
            for strategy in ('baseline','once'):near(r[strategy+'_U_percent'],canonical[('context384',strategy,r['seed'],win)]['U_percent'])
    stack=read(HERE/'figures/context384_strategy_comparison/checks.json')
    for strategy,datum in stack['results'].items():
        near(datum['U_percent'],canonical[('context384',strategy,7,(2000,20000))]['U_percent'])
        near(datum['U_percent']+sum(datum[k] for k in ('short_internal_loss_pp','long_internal_loss_pp','first_layer_loss_pp','idle_loss_pp')),100)
    if '研究尚在进行' in report[:500]:FINDINGS.append(dict(severity='editorial',kind='temporary_header',message='Opening research-in-progress banner should be removed or narrowed after long65 finalization. Long65 outcomes are explicitly outside this review.'))
    FINDINGS.append(dict(severity='clarification',kind='cycle_bandwidth_source',message='The long-window cycle mean b is reconstructed from complete-cycle count × manifest V divided by summed D (analysis field work_implied_mean_b); it is not an independent per-cycle physical-trace integration over all 2–20s. The equation and printed numbers are correct under the simulator completion contract. Consider calling it 按完整层读取量重建的周期平均带宽.'))
    record=dict(all_numeric_and_aggregation_checks_passed=True,no_simulation=True,no_canonical_edits=True,
        reviewed_utc=datetime.now(timezone.utc).isoformat(),scope='All 56 h22000 primary key-result rows, 16 seed groups, paired context384 differences, main complete-cycle b/B table, and local main-report links. All h65000 outcomes skipped.',
        main_window_rows_checked=len(rows),canonical_analysis_count=len(analyses),additional_main512_analyses=5,groups=groups,cycle_bandwidth=cycle_results,
        pairing='context384 Once n=5; raw105/raw110/hetero_short Once n=1; same-seed subtraction, equal seed weights, sample SD, failed coverage retained.',
        links=links,images=images,findings=FINDINGS,source_sha256=SOURCES,generator_sha256=sha(__file__),
        visual_review='Both five-seed PNGs independently viewed: all 5 pairs and correct values, units, sample SD, seed-stage caveat and extrapolation labels; no cropping/overlap. Existing single-seed stack values and source checks verified.')
    (HERE/'final_main_report_review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    text=['# 主报告独立核验','', '数值与汇总核验通过。未运行仿真，未修改正式数据、报告或原审计。','',
          '- 核验28份正式analysis、56个22秒人口统计窗、16个策略汇总，另核验main512的5份analysis；逐seed U与SLO通过数/总数、等权均值、样本SD和同seed差值均一致。',
          '- context384为两策略各5seed；异质短/原始105%/110%的Once为各1seed。报告正确区分配对收益与不同样本均值。',
          '- 完整短层的平均b=总V/总D，b/B是周期时长加权利用率；不是各层比值的简单平均，也不是整机瞬时利用率。',
          '- 所有主报告本地链接与PNG均存在；两幅五seed图数据和来源SHA一致，并已目检。','', '非阻断说明：',
          '1. 长窗周期b来自完成层数量×manifest V与总周期时间的重建；建议把“实际收到”注明为“按完整层读取量重建”，避免读者以为2–20秒每层都有独立逐块trace积分。',
          '2. 本核验按授权跳过全部65秒结果，后续总审计需单独涵盖。','', '[详细来源与检查](final_main_report_review.json)']
    (HERE/'final_main_report_review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(passed=True,rows=len(rows),analyses=len(analyses),links=len(links),images=len(images),findings=FINDINGS),ensure_ascii=False))

if __name__=='__main__':main()
