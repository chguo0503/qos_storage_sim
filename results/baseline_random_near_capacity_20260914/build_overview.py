#!/usr/bin/env python3
"""Refresh two study-overview PNGs using completed audited table snapshots only.

Does not launch simulations, refresh source tables, or touch existing case plots.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qos_overview_mpl'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'/'overview'
FONT=Path('/home/chguo/.fonts/msyh.ttc')
if not FONT.exists():FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
    'axes.unicode_minus':False,'font.size':10,'savefig.facecolor':'white'})
INK='#183147';MUTED='#5e7080';BLUE='#165dce';ORANGE='#dd8506'
GREEN='#087f68';PURPLE='#a34393';MODEL='#7949a9';TARGET='#b64f37'
COLORS={('baseline','long'):BLUE,('baseline','warm'):ORANGE,
        ('once','long'):GREEN,('once','warm'):PURPLE}


def snapshot(path):
    for attempt in range(5):
        data=path.read_bytes()
        try:return json.loads(data),hashlib.sha256(data).hexdigest()
        except json.JSONDecodeError:
            if attempt==4:raise
            time.sleep(.15)


def stats(values):
    vals=[float(v) for v in values]
    return dict(n=len(vals),mean=statistics.mean(vals),
        sample_std=statistics.stdev(vals) if len(vals)>1 else 0.,
        min=min(vals),max=max(vals))


def family(row):
    explicit=row.get('input_family','')
    if row.get('constructed_profile') or explicit.startswith('data_affine') or explicit in ('constructed','extrapolated'):
        return 'extrapolated'
    if explicit=='raw' or row.get('constructed_profile') is False:return 'raw'
    return 'unknown'


def run_key(row):
    return row['candidate'],int(row['num_ssu']),row['strategy'],int(row['seed'])


def label(row):
    tokens=[]
    for item in row['profile_keys'].split(','):
        total,miss=item.split(':');tokens.append(f'{total}K/{miss}')
    count=':'.join(map(str,row['count_ratio']))
    strategy='Baseline' if row['strategy']=='baseline' else ('流量分配策略' if row['strategy']=='once' else row['strategy'])
    return f"{row['candidate']}\n{' + '.join(tokens)}；{count} · {strategy}"


def section_name(fam,disks):
    title={'raw':'原始 data 输入','extrapolated':'构造输入：C 外推（详见画像来源）','unknown':'输入来源未分类'}[fam]
    return f'{title}  |  {disks} SSU × 40 GiB/s'


def prep(comparison,math_table):
    by_run={};duplicates=[]
    for r in comparison['rows']:
        if (r['window_start_s'],r['window_end_s']) not in ((2.,4.),(2.,20.)):continue
        if r.get('order')!='random':raise ValueError('This overview is Random-only')
        k=run_key(r)+(r['window_end_s'],)
        if k in by_run:raise ValueError(f'Duplicate seed/window row: {k}')
        by_run[k]=r
    grouped=defaultdict(list)
    for k,r in by_run.items():
        if k[-1]!=20.:continue
        warm=by_run.get(k[:-1]+(4.,))
        if warm is None:raise ValueError(f'Missing warm row for completed long row: {k}')
        assert r['analysis_sha256']==warm['analysis_sha256']
        grouped[k[:3]].append((r,warm))
    overview=[]
    for k,pairs in sorted(grouped.items()):
        longs=[r for r,w in pairs];warms=[w for r,w in pairs]
        configs={(r['profile_keys'],tuple(r['count_ratio']),family(r)) for r in longs}
        if len(configs)!=1:raise ValueError(f'Mixed configurations within candidate: {k}')
        first=longs[0]
        row=dict(candidate=k[0],num_ssu=k[1],strategy=k[2],family=family(first),
            profile_keys=first['profile_keys'],count_ratio=first['count_ratio'],
            seeds=sorted(r['seed'] for r in longs),n=len(longs),
            long_U=stats(r['U_percent'] for r in longs),warm_U=stats(r['U_percent'] for r in warms),
            long_mixed_pass=sum(r['long_short_mixed_pass'] for r in longs),
            warm_mixed_pass=sum(r['long_short_mixed_pass'] for r in warms),
            long_active_pass=sum(r['all_32_active'] for r in longs),
            warm_active_pass=sum(r['all_32_active'] for r in warms),
            source_cases=[dict(seed=r['seed'],case=r['case'],analysis_sha256=r['analysis_sha256'],
                input_fingerprint=r['input_fingerprint'],long_U_percent=r['U_percent'],warm_U_percent=w['U_percent'],
                long_mixed_cards=r['long_short_mixed_cards'],warm_mixed_cards=w['long_short_mixed_cards']) for r,w in pairs])
        if k[2]!='baseline':
            paired=[];unpaired=[]
            for r,w in pairs:
                bk=(k[0],k[1],'baseline',r['seed'],20.)
                base=by_run.get(bk)
                if base is None:unpaired.append(r['seed']);continue
                if base['input_fingerprint']!=r['input_fingerprint']:
                    raise ValueError(f'Paired-strategy input mismatch: {k}, seed={r["seed"]}')
                paired.append(dict(seed=r['seed'],long_delta_pp=r['U_percent']-base['U_percent'],
                    input_fingerprint=r['input_fingerprint']))
            row['baseline_pairs']=paired;row['unpaired_seeds']=unpaired
            row['paired_long_delta_pp']=stats(p['long_delta_pp'] for p in paired) if paired else None
        overview.append(row)

    matched=[];unmatched=[];stale=[]
    for r in math_table.get('rows',[]):
        k=run_key(r);ref=by_run.get(k+(20.,))
        if ref is None:unmatched.append(k);continue
        if r['analysis_sha256']!=ref['analysis_sha256'] or abs(r['observed_long_window_U_percent']-ref['U_percent'])>1e-8:
            stale.append(k);continue
        matched.append((r,ref))
    math_groups=defaultdict(list)
    for r,ref in matched:math_groups[run_key(r)[:3]].append((r,ref))
    formulas=[]
    for k,pairs in sorted(math_groups.items()):
        seen=set()
        for r,_ in pairs:
            if r['seed'] in seen:raise ValueError(f'Duplicate formula seed: {k}')
            seen.add(r['seed'])
        records=[r for r,ref in pairs];ref=pairs[0][1]
        targets=[r['short_mean_wait_needed_for_U80_ms'] for r in records]
        if max(targets)-min(targets)>1e-8:raise ValueError(f'Changed math target across seeds: {k}')
        formulas.append(dict(candidate=k[0],num_ssu=k[1],strategy=k[2],family=family(ref),
            profile_keys=ref['profile_keys'],count_ratio=ref['count_ratio'],
            seeds=sorted(seen),n=len(records),
            target_wait_ms=stats(targets),
            observed_wait_ms=stats(r['count_weighted_observed_short_wait_ms'] for r in records),
            observed_over_target=stats(r['count_weighted_observed_short_wait_ms']/r['short_mean_wait_needed_for_U80_ms'] for r in records),
            actual_U=stats(r['observed_long_window_U_percent'] for r in records),
            substituted_U=stats(r['conditional_U_all_internal_wait_percent'] for r in records),
            formula_error_pp=stats(r['approximation_error_all_internal_pp'] for r in records),
            any_long_internal_wait=any(any(w>1e-8 for role,w in r['observed_internal_wait_by_role_ms'].items()
                if role not in r['short_roles']) for r in records),
            rows=records))
    math_keys={run_key(r) for r,ref in matched}
    missing_math=[k[:-1] for k in by_run if k[-1]==20. and k[:-1] not in math_keys]
    return dict(overview_rows=overview,formula_rows=formulas,
        completed_long_run_count=len(grouped) and sum(len(v) for v in grouped.values()),
        formula_matched_run_count=len(matched),comparison_runs_without_current_math=missing_math,
        math_without_comparison=unmatched,stale_math_rows=stale,
        pending_or_unanalyzed=comparison.get('pending_or_unanalyzed',[]),
        all_seed_rows_retained=True,error_bar_definition='Equal seed weighting, sample standard deviation; one seed has no error bar',
        pairing_definition='Same candidate, SSU, seed, and identical input_fingerprint before reporting a strategy delta')


def sections(rows,include_fit_placeholder=True):
    keys=sorted({(r['family'],r['num_ssu']) for r in rows},key=lambda k:({'raw':0,'extrapolated':1,'unknown':2}[k[0]],k[1]))
    result=[]
    for fam,disks in keys:
        part=sorted((r for r in rows if (r['family'],r['num_ssu'])==(fam,disks)),
            key=lambda r:(r['candidate'],r['strategy']!='baseline',r['strategy']))
        result.append((fam,disks,part))
    if include_fit_placeholder and not any(fam=='extrapolated' for fam,disks,part in result):
        result.append(('extrapolated',None,[]))
    return result


def point(ax,x,y,error,color,marker='o',fill=True):
    ax.errorbar(x,y,xerr=error if error>0 else None,fmt=marker,markersize=5.5,
        color=color,markerfacecolor=color if fill else 'white',markeredgewidth=1.2,
        elinewidth=1.35,capsize=3,zorder=4)


def style(ax,y,labels):
    ax.set_yticks(y,labels,fontsize=8.8)
    ax.tick_params(axis='y',length=0,pad=12)
    ax.set_ylim(len(y)-.55,-.55)
    ax.grid(axis='x',alpha=.17,zorder=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for j in range(len(y)):
        if j%2==0:ax.axhspan(j-.45,j+.45,color='#f3f6fa',zorder=-1)


def build_utilization(data,dpi):
    parts=sections(data['overview_rows'])
    block_heights=[max(1.5,.52*len(part)+.9) for fam,d,part in parts]
    height=2.15+sum(block_heights)+1.1
    fig=plt.figure(figsize=(20,height))
    fig.text(.04,1-.40/height,'Random 研究总览：warm 与长窗的实际 NPU 利用率',fontsize=20,fontweight='bold',color=INK)
    fig.text(.04,1-.80/height,'32 NPU · 每卡完整队列独立打乱 · 每盘 40 GiB/s · 原始 data 与 C 外推分区',fontsize=11,color=MUTED)
    handles=[Line2D([],[],marker='o',color=BLUE,ls='',label='Baseline 长窗 [2,20)s'),
             Line2D([],[],marker='D',color=ORANGE,ls='',label='Baseline warm [2,4)s')]
    if any(r['strategy']!='baseline' for r in data['overview_rows']):
        handles += [Line2D([],[],marker='^',color=GREEN,ls='',label='流量分配策略 长窗'),
                    Line2D([],[],marker='s',color=PURPLE,ls='',label='流量分配策略 warm')]
    handles.append(Line2D([],[],color=TARGET,lw=7,alpha=.15,label='80%～90% 目标区间'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.037,1-1.0/height),ncol=len(handles),frameon=False,fontsize=10)
    fig.text(.04,1-1.66/height,'每个点保留该组所有完成种子；误差棒为样本标准差（不是置信区间）。n=1 只是 pilot，不能代表多种子结论。',fontsize=10,color=MUTED)
    minimum=min((r[w]['min']-r[w]['sample_std'] for r in data['overview_rows'] for w in ('long_U','warm_U')),default=70)
    xmin=max(0,min(70,math.floor((minimum-3)/10)*10))
    top=height-2.15
    for (fam,disks,rows),block_height in zip(parts,block_heights):
        fig.text(.04,(top-.03)/height,section_name(fam,disks) if disks else '构造输入：C 外推（详见画像来源）',
            fontsize=13,fontweight='bold',color='#87551a' if fam=='extrapolated' else INK)
        if not rows:
            fig.text(.04,(top-.58)/height,'当前 comparison.json 尚无该类完成实测；不画候选预测值。后续刷新源表并重跑脚本即可补全。',fontsize=11,color=MUTED)
            top-=block_height;continue
        h=.52*len(rows)
        ax=fig.add_axes([.365,(top-.42-h)/height,.365,h/height])
        ys=np.arange(len(rows));style(ax,ys,[label(r) for r in rows]);ax.set_xlim(xmin,100.8)
        ax.axvspan(80,90,color=TARGET,alpha=.06,zorder=0)
        ax.axvline(80,color=TARGET,ls='--',lw=.7,alpha=.6)
        ax.axvline(90,color=TARGET,ls='--',lw=.7,alpha=.6)
        ax.set_xlabel('实际平均 NPU 利用率 (%)',fontsize=10)
        ax.text(1.04,1.027,'长窗 / warm 均值   |   种子与混合覆盖',transform=ax.transAxes,fontsize=9.5,color=MUTED)
        for j,row in enumerate(rows):
            strategy=row['strategy'];fallback=strategy if strategy in ('baseline','once') else 'once'
            for name,offset,marker in (('long',-.13,'o' if strategy=='baseline' else '^'),
                                       ('warm',.13,'D' if strategy=='baseline' else 's')):
                s=row[name+'_U'];point(ax,s['mean'],j+offset,s['sample_std'],COLORS[(fallback,name)],marker)
            text=f"{row['long_U']['mean']:.2f}% / {row['warm_U']['mean']:.2f}%   n={row['n']}\n混合 warm {row['warm_mixed_pass']}/{row['n']}；长窗 {row['long_mixed_pass']}/{row['n']}"
            if row.get('paired_long_delta_pp') is not None:
                d=row['paired_long_delta_pp'];text+=f"；配对 Δ长 {d['mean']:+.2f}pp (n={d['n']})"
            if row['long_active_pass']<row['n'] or row['warm_active_pass']<row['n']:
                text+='；存在未全卡活跃窗口'
            ax.text(1.04,j,text,transform=ax.get_yaxis_transform(),va='center',fontsize=9.2,
                color=INK if row['warm_mixed_pass']==row['n'] and row['long_mixed_pass']==row['n'] else '#9a5c17')
        top-=block_height
    fig.text(.04,.58/height,'混合 a/n：n 个完成种子中，a 个种子的32张卡在该窗口均实际计算过长、短请求。覆盖失败仍计入均值，不换 seed。',fontsize=9.5,color=MUTED)
    fig.text(.04,.20/height,f"来源：comparison.json 快照 · {data['completed_long_run_count']} 个完成案例；画像标签=总输入K/miss token；数量按标签顺序。",fontsize=9,color=MUTED)
    path=OUT/'utilization_overview.png';fig.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    return path


def build_formula(data,dpi):
    parts=sections(data['formula_rows'])
    block_heights=[max(1.45,.56*len(part)+1.) for fam,d,part in parts]
    height=2.4+sum(block_heights)+1.4
    fig=plt.figure(figsize=(22,height))
    fig.text(.035,1-.42/height,'公式核对：实测等待有多大，能解释多少利用率损失？',fontsize=20,fontweight='bold',color=INK)
    fig.text(.035,1-.85/height,'仅长窗 [2,20)s · 左：历史80%阈值核对；右：代入已测等待的近似 U 与实际 U；当前目标为80多',fontsize=11,color=MUTED)
    fig.text(.035,1-1.25/height,'右图是事后解释，不是独立预测。内部周期为同请求 compute_start → next_compute_start；平均等待包含零等待层。',fontsize=11,color='#9b452e')
    handles=[Line2D([],[],marker='o',color=BLUE,ls='',label='实际值（Baseline）'),
        Line2D([],[],marker='o',color=MODEL,markerfacecolor='white',ls='',label='代入实测内部等待的近似 U'),
        Line2D([],[],color=TARGET,ls='--',label='左：等待目标 1 倍；右：U=80%')]
    if any(r['strategy']!='baseline' for r in data['formula_rows']):
        handles.insert(1,Line2D([],[],marker='^',color=GREEN,ls='',label='实际值（流量分配策略）'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.032,1-1.48/height),frameon=False,ncol=len(handles),fontsize=10)
    xmax=max(1.15,max((r['observed_over_target']['max']+r['observed_over_target']['sample_std'] for r in data['formula_rows']),default=1)*1.15)
    umin=max(0,min(70,math.floor((min((min(r['actual_U']['min'],r['substituted_U']['min']) for r in data['formula_rows']),default=80)-4)/10)*10))
    top=height-2.4
    for (fam,disks,rows),block_height in zip(parts,block_heights):
        fig.text(.035,(top-.01)/height,section_name(fam,disks) if disks else '构造输入：C 外推（详见画像来源）',fontsize=13,fontweight='bold',color='#87551a' if fam=='extrapolated' else INK)
        if not rows:
            fig.text(.035,(top-.60)/height,'当前公式源表尚无此类可匹配实测。不会将未运行候选、缺失行或条件预测画成结果。',fontsize=11,color=MUTED)
            top-=block_height;continue
        h=.56*len(rows);bottom=(top-.46-h)/height
        left=fig.add_axes([.325,bottom,.19,h/height]);right=fig.add_axes([.685,bottom,.17,h/height])
        ys=np.arange(len(rows));style(left,ys,[label(r) for r in rows]);style(right,ys,['']*len(rows))
        left.set_xlim(0,xmax);right.set_xlim(umin,100.8)
        left.axvline(1,color=TARGET,ls='--',lw=1);right.axvline(80,color=TARGET,ls='--',lw=1)
        left.set_xlabel('实测平均短等待 / U80 所需等待',fontsize=9.5)
        right.set_xlabel('平均 NPU 利用率 (%)',fontsize=9.5)
        left.text(1.03,1.025,'实测 / 目标 (ms)',transform=left.transAxes,fontsize=9.2,color=MUTED)
        right.text(1.04,1.025,'实际 / 代入 (%)',transform=right.transAxes,fontsize=9.2,color=MUTED)
        for j,row in enumerate(rows):
            color=BLUE if row['strategy']=='baseline' else GREEN;marker='o' if row['strategy']=='baseline' else '^'
            r=row['observed_over_target'];point(left,r['mean'],j,r['sample_std'],color,marker)
            left.text(1.03,j,f"{row['observed_wait_ms']['mean']:.3f} / {row['target_wait_ms']['mean']:.3f}\n{r['mean']:.2f} 倍 · n={row['n']}"+(' · 长层也有等待' if row['any_long_internal_wait'] else ''),
                transform=left.get_yaxis_transform(),va='center',fontsize=8.8,color=INK)
            a=row['actual_U'];m=row['substituted_U']
            point(right,a['mean'],j-.13,a['sample_std'],color,marker)
            point(right,m['mean'],j+.13,m['sample_std'],MODEL,'o',False)
            right.text(1.04,j,f"{a['mean']:.2f} / {m['mean']:.2f}\n差 {row['formula_error_pp']['mean']:+.2f}pp",
                transform=right.get_yaxis_transform(),va='center',fontsize=9,color=INK)
        top-=block_height
    note=f"源表同步：{data['formula_matched_run_count']} 个完成案例有匹配公式记录；comparison 中另 {len(data['comparison_runs_without_current_math'])} 个案例待公式表更新。"
    fig.text(.035,1.05/height,note,fontsize=9.6,color=MUTED)
    fig.text(.035,.67/height,'目标 w80 假设长层不等待、长期完成比例接近输入比例；多种短画像按输入数量加权。误差棒为种子样本标准差，n=1 无误差棒。',fontsize=9.4,color=MUTED)
    fig.text(.035,.29/height,'近似 U 用各画像内部层已测等待，忽略首层、跨请求与窗口组成变化；实际 U 保留全窗计算。相近说明可解释，不证明调度因果或独立预测。',fontsize=9.4,color=MUTED)
    path=OUT/'waiting_formula_overview.png';fig.savefig(path,dpi=dpi,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dpi',type=int,default=160)
    args=parser.parse_args()
    comparison,csha=snapshot(HERE/'comparison.json');math_table,msha=snapshot(HERE/'math_result_table.json')
    data=prep(comparison,math_table)
    OUT.mkdir(parents=True,exist_ok=True)
    data['definitions']=dict(actual_U='Actual compute overlap / (32*window duration)',
        warm_window_s=[2,4],long_window_s=[2,20],
        formula_source='math_result_table.json; observed per-profile complete internal-layer waits, weighted by input request counts',
        not_prediction=True,unrun_candidates_never_plotted=True,
        constructed_origin_note='Constructed cases can extrapolate the long profile, short profile, or both; audited manifests preserve each profile source.')
    paths=[build_utilization(data,args.dpi),build_formula(data,args.dpi)]
    sources=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input_snapshots={'comparison.json':csha,'math_result_table.json':msha},
        comparison_updated_utc=comparison.get('updated_utc'),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        font=str(FONT),font_sha256=hashlib.sha256(FONT.read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        status='Completed source rows only; can be rerun after source-table refresh',
        writes_restricted_to=str(OUT.relative_to(HERE)),
        source_tables_not_modified=True,simulations_started=0)
    (OUT/'overview_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    sources['output_sha256']['overview_data.json']=hashlib.sha256((OUT/'overview_data.json').read_bytes()).hexdigest()
    (OUT/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(outputs=[str(p.relative_to(HERE)) for p in paths],
        completed_runs=data['completed_long_run_count'],formula_runs=data['formula_matched_run_count'],
        pending_formula_count=len(data['comparison_runs_without_current_math']),
        stale_math_rows=len(data['stale_math_rows'])),ensure_ascii=False))


if __name__=='__main__':
    main()
