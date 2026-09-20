#!/usr/bin/env python3
"""Render three-policy CDFs from the 27 existing fully drained simulations."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = (7, 19, 43)
POLICIES = ('asu_baseline', 'od_baseline', 'once')
REGIMES = ('under', 'full', 'semi')
NAMES = {'asu_baseline': 'ASU Baseline', 'od_baseline': 'OD Baseline', 'once': '流量分配策略（Once）'}
TITLES = {'under': '持续欠载', 'full': '持续过载', 'semi': '局部过载（间歇过载）'}
STYLES = {'asu_baseline': ('#555555', '--', 3.6), 'od_baseline': ('#D55E00', '-.', 2.7),
          'once': ('#0072B2', '-', 1.8)}


def read(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select():
    cases=[]
    for regime in REGIMES:
        name='continuous_underload_asu_od_20260918' if regime=='under' else 'od_baseline_diverse_ssu3_20260918'
        source=ROOT/'results'/name
        checks=read(source/'summary_checks.json')
        assert checks['status']=='complete'
        for row in checks['cases']:
            desired=(row.get('order')=='random') if regime=='under' else row.get('scenario')==regime
            if desired and row['policy'] in POLICIES:
                cases.append((regime, source/'runs'/row['case']))
        if regime=='under':
            for seed in SEEDS:
                cases.append((regime, HERE.parent/'runs'/f'under_once_seed{seed}_local'))
    assert len(cases)==27
    return cases


def prepare():
    cases=[];samples=[];sources={};fingerprints={};source_maps=[]
    for regime, path in select():
        cmd=read(path/'command.json')
        assert cmd['status']=='complete' and cmd['completed_simulation'] and not cmd['smoke']
        assert all(cmd['checks'].values()) and cmd['core_unchanged']
        for name,key in [('result.json.gz','result_sha256'),('manifest.json.gz','manifest_sha256')]:
            p=path/name; value=sha(p)
            assert value==cmd[key]
            sources[str(p.relative_to(ROOT))]=value
        source_maps.append(cmd['core_source_sha256'])
        raw=read(path/'result.json.gz'); s=raw['summary']
        assert (s['num_npu'],s['num_ssu'],s['n_layers'],s['batch_size'])==(32,3,8,1)
        assert all(math.isfinite(r['completion_time_ms']) for r in s['request_metrics'])
        policy,seed=cmd['policy'],cmd['seed']
        assert seed in SEEDS and policy in POLICIES
        key=(regime,seed)
        identity=(cmd['manifest_sha256'],raw['input_fingerprint'])
        assert fingerprints.setdefault(key,identity)==identity
        window=next(a for a in raw['analysis'] if a['start_ms']==2000 and a['end_ms']==4000)
        cohort=[r for r in s['request_metrics'] if 2000<=r['admission_time_ms']<4000]
        ratios=[];passed=0
        for r in cohort:
            latency=r['completion_time_ms']-r['admission_time_ms'];base=r['own_compute_ms']
            assert base>0 and latency+1e-9>=base
            ratio=latency/base; plotted=ratio
            # Preserve the existing 1e-9 ms numerical boundary tolerance.
            for factor in (1.,1.5):
                if abs(latency-factor*base)<=1e-9:plotted=factor
            ok=latency<=1.5*base+1e-9
            assert (plotted<=1.5)==ok
            passed+=ok;ratios.append(plotted)
            samples.append(dict(regime=regime,policy=policy,seed=seed,request_id=r['request_id'],
                                admission_ms=r['admission_time_ms'],completion_ms=r['completion_time_ms'],
                                own_compute_ms=base,latency_ms=latency,raw_ratio=ratio,plot_ratio=plotted,
                                slo_1p5_pass=ok))
        assert (len(cohort),passed)==(window['slo']['count'],window['slo']['passed'])
        slo=100*passed/len(cohort)
        assert abs(slo-window['slo']['percent'])<1e-9
        cases.append(dict(regime=regime,policy=policy,seed=seed,case=path.name,
                          result=str((path/'result.json.gz').relative_to(ROOT)),
                          n=len(cohort),passed=passed,slo_percent=slo,ratios=sorted(ratios),
                          completed_after_window=sum(r['completion_time_ms']>4000 for r in cohort)))
    assert len({(c['regime'],c['policy'],c['seed']) for c in cases})==27
    assert all(x==source_maps[0] for x in source_maps)
    return cases,samples,sources


def cdf(cases, policy, xx):
    rows=[c for c in cases if c['policy']==policy]
    assert sorted(r['seed'] for r in rows)==list(SEEDS)
    return np.mean([np.searchsorted(r['ratios'],xx,side='right')/r['n'] for r in rows],axis=0)


def draw(cases, regime, *, zoom):
    maximum=max(c['ratios'][-1] for c in cases)
    xmax=3.0 if zoom else max(1.8,math.ceil((maximum+.05)*2)/2)
    fig,ax=plt.subplots(figsize=(12.8,7.8))
    fig.subplots_adjust(left=.095,right=.965,top=.79,bottom=.22)
    for policy in POLICIES:
        knots=np.unique(np.concatenate([c['ratios'] for c in cases if c['policy']==policy]))
        xx=np.r_[.9,knots[(knots>.9)&(knots<xmax)],xmax]
        yy=cdf(cases,policy,xx)
        assert yy[0]==0 and np.all(np.diff(yy)>=-1e-12)
        if not zoom:assert yy[-1]==1
        slo=100*float(cdf(cases,policy,[1.5])[0])
        expected=np.mean([c['slo_percent'] for c in cases if c['policy']==policy])
        assert abs(slo-expected)<1e-9
        color,style,width=STYLES[policy]
        ax.step(xx,yy,where='post',color=color,linestyle=style,linewidth=width,
                label=f'{NAMES[policy]}    SLO×1.5：{slo:.2f}%')
        ax.scatter([1.5],[slo/100],color=color,edgecolor='white',linewidth=.6,
                   marker={'asu_baseline':'s','od_baseline':'D','once':'o'}[policy],
                   s={'asu_baseline':76,'od_baseline':48,'once':26}[policy],zorder=6)
    ax.axvline(1.5,color='#777777',linestyle=':',linewidth=1.3,zorder=0)
    ax.text(1.5,1.043,'SLO×1.5',ha='center',va='bottom',fontsize=11,color='#555555')
    ax.set(xlim=(.9,xmax),ylim=(0,1.025),ylabel='累计请求比例（CDF）',
           xlabel='归一化 TTFT = 接纳后 prefill 完成耗时 / 本请求纯计算时间')
    ax.xaxis.labelpad=12
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if xmax<=3:ax.xaxis.set_major_locator(MultipleLocator(.25 if zoom else .2))
    ax.grid(axis='y',color='#D5DADF',alpha=.75,linewidth=.7)
    ax.legend(loc='lower right',fontsize=10.5,framealpha=.97,edgecolor='#DDDDDD',handlelength=3.8,borderpad=.9,labelspacing=.8)
    if regime=='under':
        ax.text(.42,.56,'三条 CDF 重合\n三种策略均在 1 倍纯计算时间处达到 100%',
                transform=ax.transAxes,ha='center',va='center',fontsize=12,color='#555555')
    suffix=' · 阈值附近放大' if zoom else ''
    fig.text(.095,.95,f'{TITLES[regime]}：三种策略的 TTFT / SLO CDF{suffix}',
             fontsize=18,fontweight='bold',ha='left',va='top')
    fig.text(.095,.885,'32 NPU / 3 SSU × 40 GiB/s  ·  Ring hash  ·  Random  ·  warm [2,4) 秒',
             fontsize=11,color='#555555')
    note='仅放大横轴 0.9–3 倍；长尾样本保留在分母中，完整分布见对应全图。' if zoom else '横轴 1.5 处的 CDF 值就是 SLO×1.5 达标率；图中保留全部长尾。'
    fig.text(.095,.126,note,fontsize=10.4)
    fig.text(.095,.087,'seed 7、19、43 的 CDF 等权平均；窗内接纳的请求跟踪到完成，不含接纳前排队。',fontsize=10,color='#666666')
    footer=('局部过载对应此前 semi / 局部欠载：欠载与过载阶段交替出现。' if regime=='semi' else
            '负载名称沿用原输入组分类；各策略执行进度不同，窗内具体请求集合可能不同。')
    fig.text(.095,.049,footer,fontsize=9.8,color='#666666')
    out=HERE/'figures'/f'{regime}_ttft_slo_cdf{"_zoom" if zoom else ""}.png'
    fig.savefig(out,dpi=180,facecolor='white');plt.close(fig)
    return out


def main():
    (HERE/'figures').mkdir(parents=True,exist_ok=True)
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
                         'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    cases,samples,sources=prepare();macros=[];outputs=[]
    for regime in REGIMES:
        rows=[c for c in cases if c['regime']==regime]
        for policy in POLICIES:
            selected=[c for c in rows if c['policy']==policy]
            macros.append(dict(regime=regime,policy=policy,slo_1p5_percent=float(np.mean([c['slo_percent'] for c in selected])),
                               max_ratio=max(c['ratios'][-1] for c in selected),n_by_seed={c['seed']:c['n'] for c in selected}))
        for zoom in (False,True):outputs.append(draw(rows,regime,zoom=zoom))
    for name,rows in [('request_samples.csv',samples),('macro_summary.csv',macros)]:
        with (HERE/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    data=dict(config=dict(seeds=SEEDS,policies=POLICIES,window_ms=[2000,4000],
                          x='(completion_ms-admission_ms)/own_compute_ms',threshold=1.5,
                          aggregation='equal mean of per-seed ECDFs; not pooled requests'),cases=cases,macro=macros)
    (HERE/'plot_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    audit=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
               run_count=27,checks=dict(completed_results=True,same_manifest_per_seed=True,
               unchanged_equal_core_hashes=True,slo_matches_raw_results=True,cdf_at_1p5_matches_slo=True,
               full_tail_reaches_one=True,monotone_ECDF=True),sources=sources,
               outputs={str(p.relative_to(HERE)):sha(p) for p in outputs},visual_review='pending')
    (HERE/'render_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    lines=['# 三类负载：ASU、OD、流量分配的 TTFT/SLO CDF','',
           '复用之前已完成的27次正式仿真。32 NPU、3 SSU×40 GiB/s、Ring hash、Random、每请求8层；主窗口为 [2,4) 秒。没有运行新仿真，旧图保持原样。','',
           '| 负载 | ASU SLO×1.5 | OD SLO×1.5 | 流量分配（Once）SLO×1.5 | 图片 |',
           '|---|---:|---:|---:|---|']
    for regime in REGIMES:
        vals=[next(m['slo_1p5_percent'] for m in macros if m['regime']==regime and m['policy']==p) for p in POLICIES]
        lines.append(f'| {TITLES[regime]} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | [完整CDF](figures/{regime}_ttft_slo_cdf.png) · [阈值附近放大](figures/{regime}_ttft_slo_cdf_zoom.png) |')
    lines+=['','```text','横轴 = (prefill 完成时刻 - 请求接纳时刻) / 本请求 8 层纯计算时间',
            '纵轴 = 三个 seed 各自经验 CDF 的等权平均','横轴 1.5 处的纵轴值 = SLO×1.5 达标率','```','',
            'seed 为7、19、43。每个seed内，三策略使用逐字节相同的冻结输入；只统计窗口内接纳的请求，并一直跟踪至完成，不删除窗后完成或不达标请求。不同策略推进速度不同，窗口内具体请求和数量可以不同。三个seed先各自计算CDF再等权平均，不能将全部请求简单拼接。','',
            'SLO沿用接纳后prefill完成的代理口径，不含接纳前队列等待，也未模拟真正的首token事件。分母是每个请求自己的纯计算时间，不是除以总体平均，也没有再除以1.5。数值边界沿用1e-9毫秒容差。','',
            '局部过载对应此前的semi输入，旧报告称“局部欠载”或“间歇过载”，表示有些阶段欠载、有些阶段至少一盘过载。负载名称沿用原组分类，不能自动推广为Once也在每个时刻满足同一状态。这里不是上一轮新构造的94% OD输入。','',
            '持续欠载三条曲线在归一化耗时1处重合，重合与达标率100%已明确标注。放大图仅改变显示范围，不改变CDF分母；完整图保留长尾。','',
            '- full/semi：`results/od_baseline_diverse_ssu3_20260918/runs/` 中对应三策略与三个seed。',
            '- under ASU/OD：`results/continuous_underload_asu_od_20260918/runs/random_*`。',
            '- under Once：`results/od_vs_once_three_loads_20260918/runs/under_once_seed*_local/`。',
            '- [逐请求样本](request_samples.csv) · [画图数据](plot_data.json) · [绘图校验](render_checks.json) · [独立复核](independent_audit.md)','',
            '复现：`python results/od_vs_once_three_loads_20260918/three_strategy_cdf/render_cdf.py`','']
    (HERE/'README.md').write_text('\n'.join(lines))
    print(json.dumps(macros,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
