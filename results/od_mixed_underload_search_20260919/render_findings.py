#!/usr/bin/env python3
"""Render completed OD/Once evidence only; never draw formal figures from previews."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import analyze_completed as audit

PER_NPU_HELPER=ROOT/'results/od_vs_once_three_loads_20260918/per_npu_bandwidth/render_figures.py'
spec=importlib.util.spec_from_file_location('previous_per_npu_plot_helpers',PER_NPU_HELPER)
card_plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(card_plot)
START,END=2000.,4000.
COLORS={'A':'#D55E00','B':'#0072B2','wait':'#C0C5CA'}
POLICIES={'od_baseline':'OD Baseline','once':'Once per layer（流量分配策略）'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(path):
    return {name:path/name for name in ('result.json.gz','manifest.json.gz','command.json')}


def build_case(path,label):
    paths=source_files(path)
    assert all(p.exists() for p in paths.values()),f'Completed raw files required: {path}'
    hashes={name:sha(p) for name,p in paths.items()}
    result,manifest,command,requests,profiles,raw_count=audit.load_raw(paths['result.json.gz'],paths['manifest.json.gz'],paths['command.json'])
    analysis,demand,overloads,cycles,stalls=audit.analyze(result,requests,profiles,START,END,8)
    compute=[defaultdict(list) for _ in range(32)]
    all_compute=[]
    compute_by_role_npu=defaultdict(list)
    for batch in result['summary']['microbatch_metrics']:
        rid=batch['member_request_ids'][0];role=profiles[rid]['role'];npu=batch['npu_id']
        assert role in ('A','B')
        for layer in batch['layer_metrics']:
            a,z=layer['compute_start_ms'],layer['compute_end_ms']
            all_compute.append((a,z))
            compute_by_role_npu[(npu,role)].append((a,z))
            a,z=max(a,START),min(z,END)
            if z>a:compute[npu][role].append((a/1000,(z-a)/1000))
    waiting=[[] for _ in range(32)]
    for row in stalls:
        a,z=max(START,row['barrier_budget_end_ms']),min(END,row['io_ready_time_ms'])
        assert z>a
        waiting[row['npu_id']].append((a/1000,(z-a)/1000))
    supply=np.asarray(result['warm_ssd_10ms_GiB_s'],dtype=float)
    assert supply.shape==(3,200) and supply.min()>=-1e-8 and supply.max()<=40+1e-6
    saved=next(x for x in result['analysis'] if x['start_ms']==START and x['end_ms']==END)
    assert np.max(np.abs(supply.mean(axis=1)-saved['SSD_GiB_s']))<1e-6
    per_npu_supply=np.asarray(result['warm_ssd_GiB_s_by_ssu_npu'],dtype=float)
    assert per_npu_supply.shape==(3,32)
    assert np.max(np.abs(per_npu_supply.sum(axis=1)-saved['SSD_GiB_s']))<1e-6
    window_u={}
    for name,a,z in (('warm [2,4)',2000.,4000.),('warm [2,6)',2000.,6000.),('later [4,8)',4000.,8000.),('later [8,12)',8000.,12000.),('later [12,16)',12000.,16000.)):
        if z<=result['summary']['makespan_ms']:
            window_u[name]=100*math.fsum(audit.overlap(s,e,a,z) for s,e in all_compute)/(32*(z-a))
    recovery_windows=[]
    for a,z in ((2000.,4000.),(4000.,8000.),(8000.,12000.),(12000.,16000.)):
        if z>result['summary']['makespan_ms']:continue
        active_by_npu=[math.fsum(audit.overlap(r['admission_time_ms'],r['completion_time_ms'],a,z) for rid,r in requests.items() if profiles[rid]['npu_id']==npu) for npu in range(32)]
        A_active=math.fsum(audit.overlap(r['admission_time_ms'],r['completion_time_ms'],a,z) for rid,r in requests.items() if profiles[rid]['role']=='A')
        both=sum(all(math.fsum(audit.overlap(s,e,a,z) for s,e in compute_by_role_npu[(npu,role)])>0 for role in ('A','B')) for npu in range(32))
        U=100*math.fsum(audit.overlap(s,e,a,z) for s,e in all_compute)/(32*(z-a))
        saved_window=next((x for x in result['analysis'] if x['start_ms']==a and x['end_ms']==z),None)
        if saved_window is not None:audit.close(U,saved_window['U_percent'])
        recovery_windows.append(dict(start_ms=a,end_ms=z,U_percent=U,A_residence_percent=100*A_active/(32*(z-a)),
            all_npus_active=all(abs(v-(z-a))<1e-6 for v in active_by_npu),mixed_compute_npu_count=both,
            strict_underload=None if saved_window is None else saved_window['demand']['strict_underload_all_disks'],
            per_disk_peak_GiB_s=None if saved_window is None else saved_window['demand']['per_disk_max_GiB_s']))
    direct=raw_count==len(profiles)
    candidate=manifest['metadata'].get('candidate_spec',{})
    address_selection=manifest['metadata'].get('address_selection',{})
    address_selected=bool(candidate.get('address_pool')) or 'address_pool' in manifest['metadata'] or (address_selection.get('is_random_address_population') is False and bool(address_selection.get('pool_file')))
    origin=('原始data画像' if direct else '合成画像')+' / '+('对抗性地址选择' if address_selected else '普通地址')
    own_cards=[]
    for npu in range(32):
        ds=[]
        for rid,r in sorted(requests.items(),key=lambda item:item[1]['admission_time_ms']):
            if profiles[rid]['npu_id']!=npu:continue
            a,z=max(START,r['admission_time_ms']),min(END,r['completion_time_ms'])
            if z>a:ds.append([a,z,sum(profiles[rid]['B_per_ssu_GiB_s'])])
        ss=[[max(START,c['start_ms']),min(END,c['end_ms']),sum(c['derived_complete_cycle_supply_per_ssu_GiB_s'])]
            for c in cycles if c['npu_id']==npu]
        ss.sort()
        own_cards.append(dict(npu_id=npu,U_percent=analysis['per_npu'][npu]['U_percent'],
            mean_demand_GiB_s=math.fsum((z-a)*v for a,z,v in ds)/(END-START),
            mean_supply_GiB_s=float(per_npu_supply[:,npu].sum()),demand_segments=ds,supply_segments=ss,
            stall_segments=[[a*1000,(a+w)*1000] for a,w in waiting[npu]]))
    queue_roles=[[] for _ in range(32)]
    population=[Counter() for _ in range(32)]
    for row in manifest['requests']:
        rec=profiles[row['request_id']]
        queue_roles[rec['npu_id']].append(rec['role'])
        population[rec['npu_id']][(rec['role'],rec['C_ms'],tuple(rec['V_per_ssu_GiB']))]+=1
    assert hashes=={name:sha(p) for name,p in paths.items()}
    return dict(label=label,path=path,source_sha256=hashes,result=result,manifest=manifest,
                requests=requests,profiles=profiles,analysis=analysis,demand=demand,overloads=overloads,
                cycles=cycles,compute=compute,waiting=waiting,supply_10ms=supply,
                per_npu=own_cards,window_U=window_u,origin=origin,queue_roles=queue_roles,
                recovery_windows=recovery_windows,
                raw_profile_matches=raw_count,profile_count=len(profiles),
                policy=result['strategy'],population=population)


def save(fig,directory,name):
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/name,dpi=200,facecolor='white');plt.close(fig)
    return name


def timeline_axis(ax,case):
    for npu in range(32):
        y=npu-.34
        for role in ('A','B'):
            ax.broken_barh(case['compute'][npu].get(role,[]),(y,.68),facecolors=COLORS[role],edgecolors='none')
        ax.broken_barh(case['waiting'][npu],(y,.68),facecolors=COLORS['wait'],edgecolors='none')
    labels=[f"NPU {n:02d}  U={case['analysis']['per_npu'][n]['U_percent']:.2f}%" for n in range(32)]
    ax.set(yticks=range(32),yticklabels=labels,xlim=(2,4),ylim=(31.65,-.65),xlabel='时间（秒）')
    ax.set_xticks(np.arange(2,4.001,.25));ax.tick_params(axis='y',labelsize=9,length=0,pad=6)
    ax.tick_params(axis='x',labelsize=10)
    ax.grid(axis='x',alpha=.22,linewidth=.6)
    ax.set_axisbelow(True)


def timeline_legend(fig,y):
    fig.legend(handles=[Patch(color=COLORS['A'],label='短请求（A）计算'),
                        Patch(color=COLORS['B'],label='长请求（B）计算'),
                        Patch(color=COLORS['wait'],label='IO 等待')],
               loc='upper center',bbox_to_anchor=(.5,y),ncol=3,frameon=False,fontsize=11)


def timeline_single(case,directory,name):
    fig,ax=plt.subplots(figsize=(18,14.5))
    fig.subplots_adjust(left=.145,right=.98,bottom=.095,top=.855)
    timeline_axis(ax,case)
    fig.suptitle(f"{POLICIES.get(case['policy'],case['policy'])}：32张卡的计算与IO等待",fontsize=21,y=.982)
    fig.text(.5,.944,f"{case['label']} · {case['origin']} · 32 NPU / 3 SSU × 40 GiB/s · warm [2,4) 秒 · 整机 U={case['analysis']['fleet_U_percent']:.2f}%",
             ha='center',fontsize=10.5,color='#555555')
    timeline_legend(fig,.918)
    fig.text(.145,.046,'每行一张NPU；彩色仅为真实计算时间，灰色为计算已结束但下一层数据尚未到齐的等待。层之间和跨请求的等待均显示。',fontsize=10,color='#555555')
    fig.text(.145,.025,'该图展示固定窗口内实际执行；输入队列相同不意味着两策略同一时刻正在执行相同请求。',fontsize=10,color='#555555')
    return save(fig,directory,name)


def timeline_pair(od,once,directory):
    fig,axes=plt.subplots(1,2,figsize=(28,15))
    fig.subplots_adjust(left=.083,right=.985,bottom=.082,top=.85,wspace=.25)
    for ax,case in zip(axes,(od,once)):
        timeline_axis(ax,case)
        ax.set_title(f"{POLICIES[case['policy']]} · 整机 U={case['analysis']['fleet_U_percent']:.2f}%",fontsize=15,pad=16)
    fig.suptitle('相同输入：OD 与 Once 的32卡计算时序',fontsize=23,y=.982)
    fig.text(.5,.947,f"{od['origin']} · Ring hash · 32 NPU / 3 SSU × 40 GiB/s · warm [2,4) 秒",ha='center',fontsize=12,color='#555555')
    timeline_legend(fig,.918)
    fig.text(.083,.035,'对应行是同一张NPU；彩色为计算，灰色为IO等待。所有颜色均来自完成运行后的原始时刻，不从带宽比值反推。',fontsize=11,color='#555555')
    return save(fig,directory,'od_vs_once_32npu_timeline.png')


def disk_figure(od,once,directory):
    fig,axes=plt.subplots(3,2,figsize=(20,12.5),sharex=True)
    fig.subplots_adjust(left=.08,right=.985,top=.80,bottom=.11,hspace=.38,wspace=.18)
    ymax=max(44,max(max(row[f'D{s}_GiB_s'] for row in case['demand']) for case in (od,once) for s in range(3))*1.16)
    for col,case in enumerate((od,once)):
        for s in range(3):
            ax=axes[s,col];segments=case['demand'];values=[r[f'D{s}_GiB_s'] for r in segments]
            edges=np.array([r['start_ms'] for r in segments]+[segments[-1]['end_ms']])/1000
            for row,value in zip(segments,values):
                if value>40:ax.axvspan(row['start_ms']/1000,row['end_ms']/1000,color='#E8B5B5',alpha=.28,linewidth=0)
            ax.stairs(values,edges,baseline=None,color='#D55E00',linewidth=1.6)
            ax.stairs(case['supply_10ms'][s],np.linspace(2,4,201),baseline=None,color='#0072B2',linewidth=1.35)
            ax.axhline(40,color='#555555',linestyle='--',linewidth=1)
            ax.set(xlim=(2,4),ylim=(0,ymax),xlabel='时间（秒）',ylabel=f'SSU {s}（GiB/s）')
            ax.grid(alpha=.16)
            ax.text(.01,1.035,f"需求峰值 {max(values):.3f}；D>40 时间 {case['analysis']['demand']['overload_percent_by_ssu'][s]:.3f}%；真实平均供给 {case['supply_10ms'][s].mean():.3f}",
                    transform=ax.transAxes,va='bottom',fontsize=9)
            if s==0:ax.set_title(POLICIES[case['policy']],fontsize=15,pad=33)
    fig.suptitle('逐盘容量核查：名义需求与物理SSD实际供给',fontsize=22,y=.982)
    fig.text(.5,.941,f"{od['origin']} · 32 NPU / 3 SSU × 40 GiB/s · warm [2,4) 秒",ha='center',fontsize=11,color='#555555')
    handles=[Line2D([],[],color='#D55E00',lw=2,label='当前请求参考需求 Σ(V_is/C_i)'),
             Line2D([],[],color='#0072B2',lw=2,label='物理SSD实际供给（10ms平均）'),
             Line2D([],[],color='#555555',ls='--',label='每盘物理容量40 GiB/s'),
             Patch(facecolor='#E8B5B5',alpha=.28,label='该盘需求>40的精确时段')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.903),ncol=4,frameon=False,fontsize=9.5)
    fig.text(.08,.05,'这里的蓝线采用所有卡共享的10ms时间片，因此不能超过40；每卡层周期平均线属于另一种统计口径。',fontsize=10,color='#555555')
    fig.text(.08,.027,'是否欠载由各盘原始逐事件需求判定。OD与Once的执行进度可能改变窗口中的活跃画像，必须分别检验。',fontsize=10,color='#555555')
    return save(fig,directory,'od_once_per_ssu_physical_bandwidth.png')


def od_card_bandwidth(case,directory):
    cards=case['per_npu']
    for card in cards:
        for key in ('demand_segments','supply_segments'):
            rows=card[key]
            assert rows and rows[0][0]==START and rows[-1][1]==END
            for first,second in zip(rows,rows[1:]):audit.close(first[1],second[0])
    ymax=card_plot.nice_upper(max(row[2] for card in cards for key in ('demand_segments','supply_segments') for row in card[key]))
    fig,axes=plt.subplots(32,1,figsize=(18.5,28),sharex=True)
    fig.subplots_adjust(left=.13,right=.865,top=.922,bottom=.057,hspace=.31)
    for ax,card in zip(axes,cards):card_plot.draw_card(ax,card,ymax,last=card['npu_id']==31)
    fig.suptitle('OD Baseline：每张NPU的参考需求与完整层周期平均供给',fontsize=20,y=.989)
    fig.text(.5,.970,f"{case['origin']} · warm [2,4) 秒 · 整机 U={case['analysis']['fleet_U_percent']:.2f}% · 32 NPU / 3 SSU",ha='center',fontsize=11,color='#555555')
    card_plot.legend(fig,.958)
    fig.text(.13,.938,f'全部32卡共用纵轴0～{ymax:g} GiB/s；左侧为实测U，右侧为真实完整warm均值。',fontsize=10,color='#555555')
    axes[0].text(1.012,1.45,'真实warm均值\nGiB/s',transform=axes[0].transAxes,ha='left',va='bottom',fontsize=9.5,fontweight='bold')
    fig.text(.065,.027,'蓝线=下一层真实读取量/完整计算开始间隔，按预取时序与工作量守恒重建，非瞬时供给；窗边仅裁显示，均值仍用完整周期。',fontsize=9.1,color='#555555')
    fig.text(.065,.014,'跨请求周期中，蓝线读取下一请求，不能直接与当前请求橙线作比。右侧供给均值来自SSD真实服务积分，不等于蓝线可见面积。',fontsize=9.1,color='#555555')
    return save(fig,directory,'od_32npu_layer_average_bandwidth.png')


def control_figure(cases,directory):
    windows=('warm [2,4)','warm [2,6)','later [4,8)')
    fig,axes=plt.subplots(1,3,figsize=(21,5.8),sharey=True)
    fig.subplots_adjust(left=.23,right=.98,bottom=.22,top=.79,wspace=.15)
    labels=[case['label']+(' [warm超限]' if not case['analysis']['demand']['strict_underload_all_disks'] else '') for case in cases]
    for ax,window in zip(axes,windows):
        for i,case in enumerate(cases):
            value=case['window_U'].get(window)
            if value is None:continue
            color='#0072B2' if case['policy']=='once' else '#D55E00'
            ax.barh(i,value,color=color,alpha=.75,height=.6)
            ax.text(value+.5,i,f'{value:.2f}%',va='center',fontsize=10)
        ax.set(xlim=(0,110),yticks=range(len(labels)),yticklabels=labels,xlabel='平均NPU利用率（%）')
        ax.set_title(window.replace('warm ','').replace('later ','')+' 秒',fontsize=13)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle(cases[0]['origin'].split(' / ')[0]+'：策略、地址与输入顺序对照',fontsize=20,y=.97)
    fig.text(.08,.09,'每条是一个完成运行；橙=OD，蓝=Once。标注warm超限的对照在[2,4)内存在盘需求>40，不能作为严格欠载反例。',fontsize=10,color='#555555')
    fig.text(.08,.045,'不同地址/排列用于检验输入敏感性，不是多种子的统计均值；扩展窗口仅列U，不能据此证明扩展窗口逐盘欠载。',fontsize=10,color='#555555')
    return save(fig,directory,'controls_npu_utilization.png')


def recovery_figure(od,once,directory):
    fig,axes=plt.subplots(2,1,figsize=(17,10.5),sharex=True)
    fig.subplots_adjust(left=.095,right=.96,top=.81,bottom=.17,hspace=.32)
    windows=od['recovery_windows']
    assert len(windows)==4 and len(once['recovery_windows'])==4
    labels=[f'[{w["start_ms"]/1000:g},{w["end_ms"]/1000:g})' for w in windows]
    x=np.arange(4)
    for case,color,offset in ((od,'#D55E00',-1),(once,'#0072B2',1)):
        rows=case['recovery_windows']
        assert [(w['start_ms'],w['end_ms']) for w in rows]==[(w['start_ms'],w['end_ms']) for w in windows]
        for ax,key,unit in ((axes[0],'U_percent','%'),(axes[1],'A_residence_percent','%')):
            y=[w[key] for w in rows]
            ax.plot(x,y,color=color,marker='o',linewidth=2.3,markersize=7,label=POLICIES[case['policy']])
            delta=.26 if key=='U_percent' else .38
            other=once if case is od else od
            for xi,yi,w,other_w in zip(x,y,rows,other['recovery_windows']):
                direction=1 if yi>other_w[key] or (yi==other_w[key] and case is once) else -1
                mark='*' if key=='U_percent' and w['strict_underload'] is not True else ''
                ax.text(xi,yi+direction*delta,f'{yi:.2f}{unit}{mark}',ha='center',va='bottom' if direction>0 else 'top',fontsize=11,color=color)
    axes[0].set_ylabel('平均NPU利用率（%）',fontsize=12)
    min_u=min(w['U_percent'] for case in (od,once) for w in case['recovery_windows'])
    axes[0].set_ylim(math.floor(min_u)-1,100.7)
    axes[0].set_title('相同输入，分别在预先固定的窗口积分计算时间',loc='left',fontsize=13,pad=12)
    axes[1].set_ylabel('短请求A的驻留份额（%）',fontsize=12)
    vals=[w['A_residence_percent'] for case in (od,once) for w in case['recovery_windows']]
    axes[1].set_ylim(max(0,min(vals)-3),max(vals)+3)
    axes[1].set_title('A驻留包括计算与等待；份额会受到执行速度影响，并非固定请求数量比例',loc='left',fontsize=12,pad=12)
    axes[1].set_xticks(x,labels);axes[1].set_xlabel('统计时间窗口（秒）',fontsize=12)
    for ax in axes:ax.grid(alpha=.20);ax.set_xlim(-.25,3.25)
    fig.suptitle('OD与Once：warm低点之后，利用率如何变化',fontsize=22,y=.975)
    fig.text(.5,.936,f'{od["origin"]} · 同一manifest · 32 NPU / 3 SSU × 40 GiB/s',ha='center',fontsize=12,color='#555555')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.909),ncol=2,frameon=False,fontsize=12)
    all_valid=all(w['all_npus_active'] and w['mixed_compute_npu_count']==32 and w['strict_underload'] is True for case in (od,once) for w in case['recovery_windows'])
    capacity_note='两策略全部4个窗口均逐盘严格欠载、32卡持续有请求，并且每张卡都实际计算过A和B。' if all_valid else '带*的点未通过逐盘严格欠载检查，不能作为欠载证据；各窗盘峰值及32卡覆盖检查见配套表。'
    fig.text(.095,.103,capacity_note+'点之间连线仅引导阅读。',fontsize=10.5,color='#555555')
    fig.text(.095,.075,'首窗长2秒，其余各长4秒；每个点分别以32×该窗口时长为分母。纵轴局部放大，避免将小变化误读为绝对吞吐差。',fontsize=10.5,color='#555555')
    fig.text(.095,.047,'全程数值含启动与末尾排空，未绘作稳态表现。A驻留变化与相位证据应结合原始层时刻分析，不能仅凭U推断原因。',fontsize=10.5,color='#555555')
    return save(fig,directory,'utilization_recovery_windows.png')


def write_readme(cases,outputs,target):
    od,once=cases[:2]
    lines=['# 正式完成结果的绘图说明','',
           '本目录仅使用completed result和对应manifest，不使用warm预览。OD/Once主对照使用相同输入指纹；地址和排列控制另行标注。','',
           '|case|输入性质|warm U|最忙盘参考需求峰值 GiB/s|盘0/1/2超限时间占比|欠载审计结论|32卡均计算A/B|',
           '|---|---|---:|---:|---|---|---|']
    for case in cases:
        a=case['analysis']
        overload='/'.join(f'{x:.4f}%' for x in a['demand']['overload_percent_by_ssu'])
        verdict='通过：warm逐盘严格欠载' if a['demand']['strict_underload_all_disks'] else '**未通过：窗口越限，不是有效欠载反例**'
        lines.append(f"|{case['label']}|{case['origin']}|{a['fleet_U_percent']:.4f}%|{max(a['demand']['maximum_GiB_s_by_ssu']):.6f}|{overload}|{verdict}|{a['all_32_npus_have_A_and_B_compute']}|")
    invalid=[case['label'] for case in cases if not case['analysis']['demand']['strict_underload_all_disks']]
    load_note='本表所列主对照均通过warm逐盘严格欠载检查。' if not invalid else '、'.join(invalid)+'在warm内出现需求越限，只能说明输入敏感性，不能用其U证明严格欠载下的退化。'
    lines.extend(['', '这里的欠载口径是当前已接纳请求的逐盘V/C名义需求，跨请求预取不额外叠加。'+load_note,
        '', '|case|[2,4) U|[2,6) U|[4,8) U|', '|---|---:|---:|---:|'])
    for case in cases:
        values=[f"{case['window_U'][name]:.4f}%" if name in case['window_U'] else '未覆盖' for name in ('warm [2,4)','warm [2,6)','later [4,8)')]
        lines.append('|'+case['label']+'|'+'|'.join(values)+'|')
    lines.extend(['', '上述固定窗口用真实计算时段独立积分。全程结果会包含启动与末尾排空，不能把较低的全程U称为稳态退化；本图不以全程U替代warm U。'])
    lines.extend(['','计算时序图中，橙色是短请求A的真实计算区间，蓝色是长请求B的真实计算区间，灰色是数据未到齐导致的IO等待。U来自计算时间积分，不从b/B反推。',
        '逐盘图蓝线是物理SSD的统一10ms平均服务速率，40 GiB/s是它的点值上限；每张盘、每种策略分别判断需求是否超限。',
        '每卡带宽图蓝线使用完整层周期（本层开始计算到下一层开始计算，含等待）平均。在已核验的一层预取语义下，完整周期读取量等于下一层manifest读取量，因此可由工作量守恒重建；它不是独立逐块服务日志。',
        '窗口边缘只裁显示，完整周期均值含窗外时段。跨请求时，蓝线读下一请求、橙线仍描述当前请求，不能直接把它们的比值当利用率。右侧是原始collector的真实warm平均供给。','',
        '|策略|层内等待卡ms|跨请求L0等待卡ms|', '|---|---:|---:|'])
    for case in (od,once):
        waits=case['analysis']['io_stall_card_ms_by_kind']
        lines.append(f"|{case['label']}|{waits.get('internal_L1_to_L7',0):.6f}|{waits.get('cross_request_L0',0):.6f}|")
    lines.extend(['','地址筛选固定了瓶颈盘块数，用于有意保持相位；这是对抗性地址选择，不能冒充普通随机请求，也不能只凭一个候选推广为所有欠载输入的结论。',
        '原始结果通常没有每层SSD最后服务完成时刻；io_ready表示数据经NPU链路到齐。对具体哪一盘、Path或group导致等待的归因还应结合更细服务记录，不把到齐时刻误写成SSD完成。','', '图片：',''])
    lines.extend(f'- [{name}](figures/{name})' for name in outputs)
    (target/'README_figures.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--od-run',type=Path,required=True)
    parser.add_argument('--once-run',type=Path,required=True)
    parser.add_argument('--label',default='selected_completed')
    parser.add_argument('--control',action='append',default=[],help='LABEL=completed_run_directory')
    parser.add_argument('--recovery-plot',action='store_true',help='Add fixed 2–4 / 4–8 / 8–12 / 12–16 second recovery comparison')
    args=parser.parse_args();assert Path(args.label).name==args.label
    helpers={str(path):sha(path) for path in (HERE/'analyze_completed.py',PER_NPU_HELPER)}
    od=build_case(args.od_run.resolve(),'OD · 选址+错相输入')
    once=build_case(args.once_run.resolve(),'Once · 同一批输入')
    assert od['policy']=='od_baseline' and once['policy']=='once'
    assert od['result']['input_fingerprint']==once['result']['input_fingerprint']
    cases=[od,once]
    for item in args.control:
        label,path=item.split('=',1);cases.append(build_case(Path(path).resolve(),label))
    target=HERE/'findings'/args.label;directory=target/'figures';target.mkdir(parents=True,exist_ok=True)
    old_pngs=[p for p in HERE.rglob('*.png') if target not in p.parents]
    old_hashes={str(path):sha(path) for path in old_pngs}
    card_plot.setup_font()
    outputs=[timeline_single(od,directory,'od_32npu_timeline.png'),
             timeline_single(once,directory,'once_32npu_timeline.png'),
             timeline_pair(od,once,directory),disk_figure(od,once,directory),
             od_card_bandwidth(od,directory)]
    if len(cases)>2:outputs.append(control_figure(cases,directory))
    if args.recovery_plot:outputs.append(recovery_figure(od,once,directory))
    write_readme(cases,outputs,target)
    if args.recovery_plot:
        with (target/'README_figures.md').open('a') as out:
            out.write('\n## 固定窗口恢复检查\n\n|策略|窗口 秒|U|A驻留份额|持续active卡数|窗口实际计算A/B卡数|逐盘严格欠载|盘0/1/2峰值 GiB/s|\n|---|---|---:|---:|---:|---:|---|---|\n')
            for case in (od,once):
                for w in case['recovery_windows']:
                    peaks='未核验' if w['per_disk_peak_GiB_s'] is None else '/'.join(f'{x:.6f}' for x in w['per_disk_peak_GiB_s'])
                    out.write(f'|{case["label"]}|[{w["start_ms"]/1000:g},{w["end_ms"]/1000:g})|{w["U_percent"]:.6f}%|{w["A_residence_percent"]:.4f}%|{32 if w["all_npus_active"] else "未全覆盖"}|{w["mixed_compute_npu_count"]}|{w["strict_underload"]}|{peaks}|\n')
            out.write('\nA驻留包括计算和等待，分母为32×窗口时间。全程指标含启动和排空，不用作长期稳态证据。相位恢复分析另见mechanism_audit中的原始事件核算。\n')
    for case in cases:
        assert case['source_sha256']=={name:sha(path) for name,path in source_files(case['path']).items()}
    assert old_hashes=={str(path):sha(path) for path in old_pngs}
    assert helpers=={name:sha(Path(name)) for name in helpers}
    checks=dict(all_checks_passed=True,only_completed_raw_results=True,primary_input_fingerprints_equal=True,
        primary_manifest_bytes_equal=od['source_sha256']['manifest.json.gz']==once['source_sha256']['manifest.json.gz'],
        helper_sha256=helpers,renderer_sha256=sha(__file__),old_png_sha256=old_hashes,old_png_unchanged=True,
        cases=[dict(label=c['label'],path=str(c['path']),policy=c['policy'],source_sha256=c['source_sha256'],
                    origin=c['origin'],raw_profile_matches=c['raw_profile_matches'],request_count=c['profile_count'],
                    U_percent=c['analysis']['fleet_U_percent'],window_U=c['window_U'],
                    recovery_windows=c['recovery_windows'],
                    maximum_demand_GiB_s_by_ssu=c['analysis']['demand']['maximum_GiB_s_by_ssu'],
                    overload_percent_by_ssu=c['analysis']['demand']['overload_percent_by_ssu'],
                    strict_underload=c['analysis']['demand']['strict_underload_all_disks'],
                    wait_ms_by_kind=c['analysis']['io_stall_card_ms_by_kind'],
                    same_role_queue_as_OD=c['queue_roles']==od['queue_roles']) for c in cases],
        physical_supply='raw SSD service integrated over common 10ms bins; each disk <=40 GiB/s',
        per_card_supply='complete-cycle next-layer workload / interval; conservation reconstruction; not raw instantaneous rate',
        outputs=['figures/'+name for name in outputs],
        output_sha256={'figures/'+name:sha(directory/name) for name in outputs},visual_review='pending')
    (target/'render_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(png_count=len(outputs),output=str(target)),ensure_ascii=False))


if __name__=='__main__':main()
