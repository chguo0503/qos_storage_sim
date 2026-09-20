#!/usr/bin/env python3
"""Export selected frozen random multi-profile cases; never run a simulation.

Uses the existing audited layer/byte accounting from export_fixed128_32_results
and plot_fifo_mixed8. All titles/profile rows are derived from the actual input.
"""
from __future__ import annotations
import argparse
import io
import os
import tempfile
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

import export_fixed128_32_results as shared
import plot_fifo_mixed8 as accounting

LEFT,RIGHT=2000.,4000.
INK,MUTED=shared.INK,shared.MUTED
BLUE,GREEN,YELLOW=accounting.BLUE,accounting.GREEN,accounting.YELLOW
POLICY,COLORS=shared.POLICY,shared.COLORS
read,write_json,write_csv=shared.read,shared.write_json,shared.write_csv


def save_checked(fig,path):
    """Validate pixels in memory, then atomically replace the destination PNG."""
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    overflow=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            if box.x0 < -2 or box.y0 < -2 or box.x1 > width+2 or box.y1 > height+2:
                overflow.append((artist.get_text(),box.bounds))
    assert not overflow,overflow
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=150);plt.close(fig)
    payload=stream.getvalue()
    with Image.open(io.BytesIO(payload)) as png:png.load()
    with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,prefix='.png-write-',delete=False) as f:
        temporary=Path(f.name);f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    expected=hashlib.sha256(payload).hexdigest()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    return dict(file=path.name,pixels=[width,height],visible_labels_inside_canvas=True,
        sha256=expected,selected_npus=list(range(8)),atomic_png_write=True)


def descriptions(case, actual=True):
    result=[]
    for tag,gid in case['tags'].items():
        p=case['metadata']['profiles'][gid]
        nql=(f'{p["nql_range"][0]}–{p["nql_range"][1]}' if actual else str(p['nql']))
        result.append(f'{tag}={gid}：{p["total_k"]:g}K / NQL {nql}')
    return result


def profile_text_lines(case):
    items=descriptions(case)
    return ['； '.join(items[i:i+2]) for i in range(0,len(items),2)]


def header(fig,case,title,detail=None):
    fig.text(.045,.970,f'{POLICY[case["policy"]]}：{title}',fontsize=21,color=INK)
    fig.text(.045,.940,f'{case["name"]} · 8 NPU / {case["num_ssu"]} SSU × 40 GiB/s · ring hash · 展示单次 seed {case["metadata"]["seed"]}（非跨seed均值）',fontsize=12,color=MUTED)
    m=case['metrics']
    fig.text(.045,.915,f'warm [2,4)秒：整机 U={m["U_percent"]:.2f}% · 短类 U={m["short_U_percent"]:.2f}% · TTFT SLO×1.5={m["slo_1p5_percent"]:.2f}%',fontsize=12,color=MUTED)
    for i,line in enumerate(profile_text_lines(case)):
        fig.text(.045,.889-i*.022,line,fontsize=10.5,color=MUTED)
    if detail:fig.text(.045,.795,detail,fontsize=10.5,color=MUTED)


def request_rows(case):
    rows=shared.request_rows(case)
    lookup={q['request_id']:q['load'] for q in case['manifest']['requests']}
    for row in rows:
        q=lookup[row['request_id']]
        row['profile_group']=q['profile_group']
        row['profile_tag']=case['group_tags'][q['profile_group']]
        row['scenario_name']=case['name']
        row['seed']=case['metadata']['seed']
    return rows


def profile_aggregates(case,rows):
    output=[]
    for gid in case['metadata']['profiles']:
        pool=[r for r in rows if r['profile_group']==gid]
        for cohort in ['warm_admissions','all_requests']:
            sample=[r for r in pool if cohort=='all_requests' or r['admitted_in_warm']]
            warm=cohort=='warm_admissions'
            c=sum(r['warm_compute_ms'] if warm else r['ideal_compute_ms'] for r in pool)
            a=sum(r['warm_active_ms'] if warm else r['ttft_admission_ms'] for r in pool)
            output.append(dict(configuration=case['group'],scenario_name=case['name'],policy=case['policy'],
                profile_group=gid,profile_tag=case['group_tags'][gid],role=pool[0]['role'],cohort=cohort,
                request_count=len(sample),slo_passed=sum(r['slo_1p5_passed'] for r in sample),
                slo_1p5_percent=100*sum(r['slo_1p5_passed'] for r in sample)/len(sample) if sample else None,
                compute_ms=c,active_ms=a,stall_ms=a-c,active_time_U_percent=100*c/a if a else None,
                utilization_scope='all_active_intervals_clipped_to_warm' if warm else 'all_request_active_intervals'))
    return output


def zoom_bounds(case,width=250.):
    # A severe local slice is identified from all-NPU exposed stall, never used
    # as the reported warm-window metric. Fixed 250-ms candidates, deterministic.
    candidates=np.arange(LEFT,RIGHT-width+1e-9,25.)
    def score(a):
        return sum(max(0.,min(e,a+width)-max(s,a)) for s,e in case['stall_intervals'])
    left=max(candidates,key=score)
    return float(left),float(left+width)


def draw_timeline(case,path,zoom=False):
    left,right=zoom_bounds(case) if zoom else (LEFT,RIGHT)
    fig,axes=plt.subplots(8,1,figsize=(19,14),dpi=150,sharex=True,facecolor='white')
    fig.subplots_adjust(left=.105,right=.965,top=.740,bottom=.105,hspace=.55)
    scope=f'局部 [{left/1000:.3f},{right/1000:.3f})秒' if zoom else '完整 warm [2,4)秒'
    header(fig,case,f'每卡混合请求：计算、读取与 IO Stall（{scope}）')
    handles=[Patch(facecolor=GREEN,label='计算'),Patch(facecolor=BLUE,label='端到端读取（含排队）'),
        Patch(facecolor=YELLOW,label='暴露的 IO Stall'),Line2D([],[],color='#222',marker='|',lw=0,markersize=11,label='请求完成')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.041,.810),frameon=False,ncol=4,fontsize=11)
    text='局部窗口按总 Stall 最多选取，不代表平均；标签为画像 / 总长K / 实际NQL。' if zoom else '各行自上而下为计算、读取、Stall；G标签对应上方画像，短段未逐一写值，完整实际NQL见逐请求CSV和局部图。'
    fig.text(.045,.766,text,fontsize=10.5,color=MUTED)
    lookup={q['request_id']:q['load'] for q in case['manifest']['requests']}
    clip=lambda a,b:max(0.,min(b,right)-max(a,left))
    for npu,ax in enumerate(axes):
        lane=case['analysed']['lanes'][npu]
        ax.set(xlim=(left/1000,right/1000),ylim=(0,1.35),yticks=[],xticks=np.linspace(left/1000,right/1000,6 if zoom else 9))
        for batch in lane['batches']:
            q=lookup[batch['request_id']]
            a,b=max(left,batch['admission_ms']),min(right,batch['completion_ms'])
            if b<=a:continue
            tag=case['group_tags'][q['profile_group']]
            ax.axvspan(a/1000,b/1000,color='#e8eef8' if q['role']=='L' else '#f8f5e9',alpha=.48,zorder=0)
            if zoom and b-a>=8:
                label=f'{tag}\n{q["total_tokens"]/1024:g}K/{q["nql"]}'
                ax.text((a+b)/2000,1.145,label,ha='center',va='center',fontsize=7.1,color=INK,clip_on=True)
            elif not zoom and b-a>=25:
                ax.text((a+b)/2000,1.145,tag,ha='center',va='center',fontsize=8,color=INK,clip_on=True)
            deadline=batch['admission_ms']
            for layer in batch['layers']:
                for start,end,y,color in [(layer['compute_start_ms'],layer['compute_end_ms'],.71,GREEN),
                        (layer['io_start_time_ms'],layer['io_ready_time_ms'],.40,BLUE),
                        (deadline,layer['compute_start_ms'],.09,YELLOW)]:
                    length=clip(start,end)
                    if length>0:ax.broken_barh([(max(start,left)/1000,length/1000)],(y,.22),facecolors=color,linewidth=0,zorder=2)
                deadline=layer['compute_end_ms']
            if left<=batch['completion_ms']<right:
                ax.plot([batch['completion_ms']/1000]*2,[.96,1.32],color='#222',lw=.85,zorder=4)
        shown={b['request_id'] for b in lane['batches']}
        for layer in lane['layers']:
            if layer['request_id'] not in shown and clip(layer['io_start_time_ms'],layer['io_ready_time_ms'])>0:
                a,b=layer['io_start_time_ms'],layer['io_ready_time_ms']
                ax.broken_barh([(max(a,left)/1000,clip(a,b)/1000)],(.40,.22),facecolors=BLUE,linewidth=0,zorder=2)
        ax.set_ylabel(f'NPU {npu}\nwarm U={lane["U_percent"]:.2f}%',fontsize=10,rotation=0,ha='right',va='center',labelpad=10,color=INK)
        ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.16)
        ax.tick_params(axis='x',labelsize=9,labelbottom=npu==7)
    axes[-1].set_xlabel('仿真时间（秒）',fontsize=12)
    fig.text(.045,.061,'绿色与黄色覆盖完整 active 时间；蓝色可与计算重叠，不能把蓝条长度直接当作 Stall。图中 warm U 始终统计完整2秒。',fontsize=11,color=INK)
    fig.text(.045,.036,'所有卡都独立随机混合，固定NPU归属；每卡(total,NQL)不重复。数据及逐请求完成时间均单独导出。',fontsize=11,color=MUTED)
    return save_checked(fig,path)


def draw_physical(case,values,path):
    s=case['num_ssu'];count=3 if s==1 else s+3
    fig,axes=plt.subplots(count,1,figsize=(18,3.05*count+3.8),sharex=True,dpi=150)
    fig.subplots_adjust(left=.08,right=.96,top=.75,bottom=.11,hspace=.66)
    average=case['metadata']['input_demand']['total_gib_s'];over=case['metrics']['demand_audit']['any_ssu_overload_fraction']
    detail=f'输入理想平均需求 {average:.3f}/{40*s:g} GiB/s = {100*average/(40*s):.2f}% 容量；运行中任一盘当前需求超过40的时间占 {100*over:.2f}%。'
    header(fig,case,'真实 SSD 带宽、当前需求与预取需求',detail=detail)
    x=values['edges']/1000
    def panel(ax,actual,nominal,cap,title):
        ax.stairs(actual,x,color=BLUE,lw=1.1,label='实际 SSD 服务字节 / 2 ms')
        ax.stairs(nominal,x,color=shared.PURPLE,lw=1.25,label='当前请求 V/C（同窗均值）')
        ax.axhline(cap,color='#333',lw=1.1,ls='--',label=f'容量 {cap:g} GiB/s')
        ax.set_ylim(0,max(cap*1.15,float(max(nominal))*1.12))
        ax.set_title(title,loc='left',fontsize=12,pad=9);ax.set_ylabel('GiB/s')
        ax.legend(loc='upper left',ncol=3,fontsize=9,facecolor='white',framealpha=.95)
    panel(axes[0],np.sum(values['physical'],axis=0),np.sum(values['nominal'],axis=0),40*s,
        f'整机：实际平均 {values["audit"]["physical_fleet_mean_gib_s"]:.3f} GiB/s；实际峰值 {values["audit"]["physical_fleet_max_gib_s"]:.3f} GiB/s')
    i=1
    if s>1:
        for disk in range(s):
            panel(axes[i],values['physical'][disk],values['nominal'][disk],40,f'SSU {disk}：实际服务与该盘当前需求');i+=1
    ax=axes[i]
    for disk in range(s):
        ax.stairs(values['deadline'][disk],x,color=['#d97b12','#bc354e','#743c9f','#488577'][disk%4],lw=1.15,label=f'SSU {disk} 下一层 V / 当前层 C')
    ax.axhline(40,color='#333',ls='--',lw=1.1,label='单盘容量 40 GiB/s')
    ax.set_ylim(0,max(45.,float(np.max(values['deadline']))*1.18));ax.set_ylabel('GiB/s')
    ax.set_title('预取截止参考需求：含跨请求 L0，只在当前层计算窗口累计',loc='left',fontsize=12,pad=9)
    ax.legend(loc='upper left',ncol=min(3,s+1),fontsize=9,facecolor='white',framealpha=.96)
    ax=axes[-1];ax.stairs(values['stalled'],x,color='#b87500',fill=True,alpha=.65)
    ax.set(ylim=(0,8.4),yticks=[0,2,4,6,8],ylabel='NPU 数',xlabel='仿真时间（秒）')
    ax.set_title('同一个2 ms内正在等待 IO 的平均 NPU 数',loc='left',fontsize=12,pad=9)
    for ax in axes:
        ax.set_xlim(2,4);ax.set_xticks(np.arange(2,4.01,.25));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.text(.045,.060,'实际蓝线直接统计每盘共同2 ms窗口内的服务字节，未限幅；每盘≤40 GiB/s，整机≤40×SSU数。',fontsize=11,color=INK)
    fig.text(.045,.034,'平均需求欠载允许短期争用；预取参考曲线越线本身不是截止时间不可满足的充分证明。',fontsize=11,color=MUTED)
    return save_checked(fig,path)


def draw_cdf(group,cases,cohort,path,points):
    first=cases[0]
    panel_groups=['all',*first['metadata']['profiles']]
    count=len(panel_groups);cols=2 if count==4 else min(3,count);rows_count=math.ceil(count/cols)
    fig,grid=plt.subplots(rows_count,cols,figsize=(19,7.5+3.5*(rows_count-1)),dpi=150,sharey=True,squeeze=False)
    axes=grid.ravel()
    top=.69 if rows_count>1 else .61
    fig.subplots_adjust(left=.07,right=.965,bottom=.16 if rows_count>1 else .21,top=top,wspace=.20,hspace=.39)
    label='warm [2,4)秒内接纳' if cohort=='warm_admissions' else '相同全部输入请求'
    fig.text(.045,.954,f'TTFT SLO × 1.5：{first["name"]} · {label}',fontsize=22,color=INK)
    fig.text(.045,.915,f'8 NPU / {first["num_ssu"]} SSU · 展示单次 seed {first["metadata"]["seed"]}（非跨seed均值）· 相同输入策略配对',fontsize=12,color=MUTED)
    for i,line in enumerate(profile_text_lines(first)):
        fig.text(.045,.877-i*.030,line,fontsize=10.5,color=MUTED)
    for gid,ax in zip(panel_groups,axes):
        samples={c['policy']:[r for r in c['rows'] if (cohort=='all_requests' or r['admitted_in_warm']) and (gid=='all' or r['profile_group']==gid)] for c in cases}
        values=[r['ttft_over_ideal'] for rr in samples.values() for r in rr]
        xmax=max(1.7,max(values,default=1.0)*1.025)
        for case in cases:
            pool=samples[case['policy']]
            if not pool:continue
            unique,counts=np.unique([r['ttft_over_ideal'] for r in pool],return_counts=True)
            pct=np.cumsum(counts)*100/len(pool);passed=sum(r['slo_1p5_passed'] for r in pool)
            ax.step(np.r_[.98,unique,xmax],np.r_[0.,pct,100.],where='post',lw=1.8,color=COLORS[case['policy']],label=f'{POLICY[case["policy"]]}\n{passed}/{len(pool)}={100*passed/len(pool):.2f}%')
            for v,y in zip(unique,pct):
                points.append(dict(configuration=group,policy=case['policy'],profile_group=gid,cohort=cohort,ttft_over_ideal=float(v),cumulative_percent=float(y),request_count=len(pool)))
        ax.axvline(1.5,color='#555',ls='--',lw=1.2)
        tick_step=.2 if xmax<3 else .5 if xmax<6 else 1.0
        if gid=='all':title='所有请求'
        else:
            spec=first['metadata']['profiles'][gid]
            title=f'{first["group_tags"][gid]} = {gid}：{spec["total_k"]:g}K / NQL {spec["nql_range"][0]}–{spec["nql_range"][1]}'
        ax.set(xlim=(.98,xmax),ylim=(0,102),xticks=np.arange(1.,xmax+1e-9,tick_step),xlabel='TTFT / 纯计算时间',title=title)
        ax.grid(alpha=.17);ax.legend(loc='lower right',fontsize=9);ax.spines[['top','right']].set_visible(False)
    for ax in axes[count:]:ax.set_visible(False)
    for row in grid:row[0].set_ylabel('累计请求比例（%）')
    fig.text(.045,.095 if rows_count>1 else .125,'TTFT = 完成时间 - NPU接纳时间，不含接纳前输入队列等待；达标阈值为8层纯计算时间之和 × 1.5。',fontsize=11,color=INK)
    fig.text(.045,.055 if rows_count>1 else .074,'各画像分别绘制：不同NQL的32K请求不会被合并。warm接纳集合可能不同，全部输入图对齐相同请求ID。',fontsize=11,color=MUTED)
    return save_checked(fig,path)


def draw_hol(case,evidence,path):
    assert all(evidence['validation_checks'].values())
    for name,expected in evidence['source_sha256'].items():
        assert hashlib.sha256((case['folder']/name).read_bytes()).hexdigest()==expected
    short=evidence['short_layer'];longs=evidence['preceding_long_layers'];local=evidence['local_feasibility_witness'];time=evidence['time_accounting']
    origin=local['start_ms'];rel=lambda t:t-origin
    fig,ax=plt.subplots(figsize=(18,9),dpi=150)
    fig.subplots_adjust(left=.12,right=.955,top=.64,bottom=.245)
    fig.text(.045,.950,'真实 FIFO 局部证据：一个长层读取，使后面的短层错过计算窗口',fontsize=22,color=INK)
    fig.text(.045,.905,f'{case["name"]} · 展示单次 seed {case["metadata"]["seed"]}，不是跨seed均值 · 1 SSU × 40 GiB/s · 两条均为内部层',fontsize=12,color=MUTED)
    fig.text(.045,.860,f'短：NPU {short["npu_id"]} / 请求 {short["request_id"]} / L{short["layer"]} · {short["total_length_k"]:g}K / NQL {short["nql"]} · 读取 {short["bytes_mib"]:.3f} MiB · 计算窗口 {short["per_layer_compute_ms"]:.3f} ms',fontsize=12,color=INK)
    description='； '.join(f'NPU {r["npu_id"]} / 请求 {r["request_id"]} / L{r["layer"]} · {r["total_length_k"]:g}K / NQL {r["nql"]}' for r in longs)
    fig.text(.045,.818,'前面的长层：'+description,fontsize=12,color=INK)
    fig.text(.045,.777,f'实际：短层自己只需 {short["minimum_ssd_service_ms"]:.3f} ms，却暴露 {short["layer_stall_ms"]:.3f} ms Stall；该层周期利用率 {100*time["short_layer_cycle_utilization"]:.2f}%。',fontsize=12,color=MUTED)
    LONG,SHORT='#225ea8','#41b6c4'
    handles=[Patch(facecolor=GREEN,label='短请求当前层计算'),Patch(facecolor=YELLOW,label='短请求暴露 Stall'),Patch(facecolor=LONG,label='长层 SSD 服务'),Patch(facecolor=SHORT,label='短层 SSD 服务')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.041,.741),ncol=4,frameon=False,fontsize=11)
    def bar(start,end,row,color,label=None):
        ax.broken_barh([(rel(start),end-start)],(row-.19,.38),facecolors=color,edgecolor='white',linewidth=.6)
        if label:ax.text((rel(start)+rel(end))/2,row,label,color='white' if color==LONG else INK,ha='center',va='center',fontsize=10)
    bar(short['io_release_ms'],short['compute_deadline_ms'],2,GREEN,f'计算 {short["per_layer_compute_ms"]:.3f} ms')
    bar(short['compute_deadline_ms'],short['last_link_end_ms'],2,YELLOW,f'Stall {short["layer_stall_ms"]:.3f} ms')
    for r in longs:bar(r['first_ssd_start_ms'],r['last_ssd_end_ms'],1,LONG,f'长 {r["minimum_ssd_service_ms"]:.3f} ms')
    bar(short['first_ssd_start_ms'],short['last_ssd_end_ms'],1,SHORT,f'短 {short["minimum_ssd_service_ms"]:.3f} ms')
    bar(origin,local['short_ssd_end_ms'],0,SHORT,f'短 {short["minimum_ssd_service_ms"]:.3f} ms')
    for r in local['reordered_long_intervals']:bar(r['start_ms'],r['ssd_end_ms'],0,LONG,'随后读长层')
    sd=rel(short['compute_deadline_ms']);ld=rel(local['earliest_long_deadline_ms'])
    ax.axvline(sd,color='#c23b42',ls='--',lw=1.3)
    ax.axvline(ld,color='#596c80',ls='--',lw=1.3)
    ax.text(sd,2.53,f'短截止\n{short["compute_deadline_ms"]:.3f} ms',ha='center',va='center',fontsize=10,color='#c23b42')
    ax.text(ld,2.53,f'长截止\n{local["earliest_long_deadline_ms"]:.3f} ms',ha='center',va='center',fontsize=10,color=MUTED)
    ax.annotate(f'实际短数据到齐\n{short["last_link_end_ms"]:.3f} ms',xy=(rel(short['last_link_end_ms']),2),xytext=(rel(short['last_link_end_ms'])+1.1,2.42),arrowprops=dict(arrowstyle='->',color=MUTED),fontsize=10,color=MUTED)
    ax.annotate(f'局部短到齐上界 {local["short_link_end_upper_bound_ms"]:.3f} ms\n早于短截止 {local["short_deadline_slack_lower_bound_ms"]:.3f} ms',xy=(rel(local['short_link_end_upper_bound_ms']),0),xytext=(3.1,-.60),arrowprops=dict(arrowstyle='->',color=MUTED),fontsize=10,color=MUTED)
    xmax=math.ceil(ld+1);xmin=min(-.7,rel(short['io_release_ms'])-.2)
    ax.set(xlim=(xmin,xmax),ylim=(-.85,2.85),yticks=[0,1,2],yticklabels=['局部离线可行次序\n未重跑仿真','实际 SSD FIFO','实际 NPU '+str(short['npu_id'])],xticks=np.arange(0,xmax+1e-9,2),xlabel=f'相对于真实 IO 边界 {origin:.6f} ms 的时间（ms）')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.tick_params(axis='y',length=0,labelsize=11)
    fig.text(.045,.178,f'在这一个真实 IO 边界，短层全部 IO 已提交。交换这两层的先后后，总 SSD 完成时刻仍为 {local["all_jobs_ssd_end_ms"]:.3f} ms，短、长均可按期到齐。',fontsize=11,color=INK)
    fig.text(.045,.128,'这里证明的是该局部已排队任务集合存在可行次序；后续新层未重新释放、未重跑整条轨迹，不能把它当作 Once 的整体收益。',fontsize=11,color=MUTED)
    fig.text(.045,.082,'示例按严格的连续整层服务条件事后筛选；它不表示单长阻塞的发生频率，也不证明整组输入所有时刻都能满足截止时间。',fontsize=11,color=MUTED)
    return save_checked(fig,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,action='append',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--font',type=Path,required=True)
    args=p.parse_args()
    for d in ['images','data']:(args.out/d).mkdir(parents=True,exist_ok=True)
    fontManager.addfont(str(args.font));plt.rcParams.update({'font.family':FontProperties(fname=str(args.font)).get_name(),'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white'})
    cases=[];groups=defaultdict(list)
    for folder in args.case:
        folder=folder.resolve();meta=read(folder/'metadata.json');assert meta['order']=='random'
        analysed=accounting.analyse(folder)
        name=meta['case']['name'];fp=meta['input_fingerprint'];group=f'{name}_seed{meta["seed"]}_s{meta["num_ssu"]}_{fp[:8]}'
        tags={f'G{i+1}':gid for i,gid in enumerate(meta['profiles'])}
        case=dict(folder=folder,name=name,analysed=analysed,manifest=read(folder/'manifest.json.gz'),metadata=meta,
            metrics=read(folder/'metrics.json'),receipt=read(folder/'receipts.json'),raw=read(folder/'result.json.gz'),
            group=group,fingerprint=fp,num_ssu=meta['num_ssu'],tags=tags,group_tags={v:k for k,v in tags.items()})
        case['policy']=case['metrics']['policy'];case['stem']=f'{group}_{case["policy"]}'
        case['rows']=request_rows(case)
        assert all(32<=r['total_length_k']<=200 for r in case['rows'])
        case['stall_intervals']=[]
        for lane in analysed['lanes']:
            for batch in lane['batches']:
                end=batch['admission_ms']
                for layer in batch['layers']:
                    case['stall_intervals'].append((end,layer['compute_start_ms']));end=layer['compute_end_ms']
        cases.append(case);groups[group].append(case)
    assert len({c['stem'] for c in cases})==len(cases),'Duplicate policy/input case'
    allrows=[];classrows=[];grouprows=[];audits=[];scenarios=[];points=[]
    for case in cases:
        stem=case['stem'];values=shared.physical_data(case)
        audits.extend([draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline.png'),
            draw_timeline(case,args.out/'images'/f'{stem}_8npu_timeline_zoom.png',True),
            draw_physical(case,values,args.out/'images'/f'{stem}_physical_ssd_bandwidth.png')])
        evidence_path=case['folder']/'fifo_hol_evidence.json'
        if evidence_path.exists() and case['policy']=='fifo':
            evidence=read(evidence_path)
            audits.append(draw_hol(case,evidence,args.out/'images'/f'{stem}_fifo_hol_local.png'))
            write_json(args.out/'data'/f'{stem}_fifo_hol_evidence.json',evidence)
        rs=case['rows'];classes=shared.aggregate_rows(case,rs);pr=profile_aggregates(case,rs)
        write_csv(args.out/'data'/f'{stem}_requests.csv',rs)
        write_csv(args.out/'data'/f'{stem}_layer_cycles.csv',case['analysed']['cycles'])
        write_csv(args.out/'data'/f'{stem}_per_npu_class.csv',case['analysed']['class_rows'])
        write_csv(args.out/'data'/f'{stem}_physical_2ms.csv',values['rows'])
        write_csv(args.out/'data'/f'{stem}_prefetch_deadline_jobs.csv',values['jobs'])
        write_json(args.out/'data'/f'{stem}_physical_audit.json',values['audit'])
        allrows.extend(rs);classrows.extend(classes);grouprows.extend(pr)
        full=next(r for r in classes if r['npu_id']=='all' and r['role']=='all' and r['cohort']=='all_requests')
        scenarios.append(dict(configuration=case['group'],scenario_name=case['name'],policy=case['policy'],seed=case['metadata']['seed'],
            case_directory=str(case['folder']),num_ssu=case['num_ssu'],profile_group_count=len(case['tags']),
            input_fingerprint=case['fingerprint'],U_percent=case['metrics']['U_percent'],short_U_percent=case['metrics']['short_U_percent'],
            long_U_percent=case['metrics']['long_U_percent'],warm_slo_percent=case['metrics']['slo_1p5_percent'],
            full_slo_percent=full['slo_1p5_percent'],warm_slo_passed=case['metrics']['slo_passed'],warm_slo_count=case['metrics']['slo_count'],
            full_slo_passed=full['slo_passed'],request_count=len(rs),input_average_demand_gib_s=case['metadata']['input_demand']['total_gib_s'],
            input_average_capacity_ratio=case['metadata']['input_demand']['total_gib_s']/(40*case['num_ssu']),
            all_npus_both_roles_computed=case['metrics']['all_npus_both_roles_computed'],
            static_current_underload=case['metadata']['static_underload_all_request_combinations'],
            source_hashes=case['analysed']['source_hashes'],**values['audit']))
    for group,items in groups.items():
        ids=[{r['request_id'] for r in c['rows']} for c in items];assert all(x==ids[0] for x in ids)
        for cohort in ['warm_admissions','all_requests']:
            audits.append(draw_cdf(group,items,cohort,args.out/'images'/f'{group}_ttft_slo15_{cohort}.png',points))
        fields=['configuration','scenario_name','seed','npu_id','request_id','input_order_1based','profile_group','profile_tag','role','qos_category','total_tokens','total_length_k','nql_tokens','hit_prefix_tokens','per_layer_read_gib','per_layer_read_mib','per_layer_compute_ms','bandwidth_demand_gib_s','compute_method','compute_extrapolated']
        write_csv(args.out/'data'/f'{group}_input_profiles.csv',[{k:r[k] for k in fields} for r in items[0]['rows']])
        write_json(args.out/'data'/f'{group}_profile_groups.json',dict(tags=items[0]['tags'],actual_profiles=items[0]['metadata']['profiles'],input_fingerprint=items[0]['fingerprint']))
    write_csv(args.out/'data'/'all_request_execution.csv',allrows);write_csv(args.out/'data'/'slo_and_class_utilization.csv',classrows)
    write_csv(args.out/'data'/'profile_group_summary.csv',grouprows);write_csv(args.out/'data'/'ttft_cdf_points.csv',points)
    write_json(args.out/'data'/'scenario_summary.json',scenarios)
    write_csv(args.out/'data'/'scenario_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in scenarios])
    for info in audits:
        path=args.out/'images'/info['file']
        with Image.open(path) as png:png.load()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
    write_json(args.out/'data'/'figure_audit.json',dict(figures=audits,all_checks_passed=True,input_groups=list(groups),
        request_execution_rows=len(allrows),no_simulation_executed=True,
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(shared.__file__),Path(accounting.__file__)]}))
    notes='''# 随机多画像请求实验导出

每张 NPU 独立随机混合全部画像；总长和实际 NQL 以输入 CSV 为准。G1、G2 等只是图中画像简称，不是 QoS Path。\n\n本图包展示选定 seed=7 的单次结果，主示例中它是 Baseline 利用率较低的一次，不能当作典型随机输入或跨seed平均；跨seed结果以单独汇总为准。

- `*_input_profiles.csv`：每条输入的画像组、总长、NQL、每层精确读取量、计算时间和带宽需求。
- `*_requests.csv`、`all_request_execution.csv`：逐请求实际接纳、完成、TTFT、SLO 和 warm 时间账。
- CDF 按所有请求和每个 `profile_group` 分面，自动安排面板；相同32K但不同NQL的画像不会合并。\n- `profile_group_summary.csv`：各画像类别的利用率与 SLO；`slo_and_class_utilization.csv` 还区分各卡及 L/S。
- `*_physical_2ms.csv`：真实 SSD 服务字节除以共同 2 ms；按实际 ring hash 和尾块大小统计，无限幅。
- 常规需求是当前请求 V/C，预取截止参考需求是下一层 V/当前层 C。平均欠载不保证瞬时欠载；参考曲线越线本身不证明不可调度。
- 时序图保持绿色计算、蓝色端到端读取、黄色 IO Stall。局部图挑选 total Stall 最大的 250 ms 窗口，只用来观察排队；所有图上 warm U 都来自完整 [2,4) 秒。局部标签给出画像、总长和实际 NQL，完整逐请求值以 CSV 为准。
- TTFT 从 NPU 接纳开始计时，不含接纳之前的输入队列等待。SLO 阈值为 8 层纯计算时间之和 × 1.5。
- warm 的 SLO 取窗口内接纳集合，利用率取所有与窗口相交的 active 区间；全部输入 CDF 对齐完全相同请求 ID。
- 此导出器只读取冻结结果，不修改原输入、不重跑仿真。
'''
    (args.out/'README.md').write_text(notes,encoding='utf-8')
    print(shared.json.dumps(dict(output=str(args.out.resolve()),cases=len(cases),input_groups=len(groups),figures=len(audits),request_execution_rows=len(allrows),all_checks_passed=True),ensure_ascii=False))

if __name__=='__main__':main()
