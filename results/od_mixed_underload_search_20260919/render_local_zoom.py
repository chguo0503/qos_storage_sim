#!/usr/bin/env python3
"""A same-request, completed-raw cycle example; preserve the six overview PNGs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import analyze_completed as audit
import render_findings as shared

HERE=Path(__file__).resolve().parent
ORANGE='#D55E00';BLUE='#0072B2';GRAY='#C8CDD2';DARK='#333333'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=HERE/'runs/fixedssu1_safe38_od_local')
    p.add_argument('--output-dir',type=Path,default=HERE/'findings/fixed_bottleneck_safe38')
    p.add_argument('--npu',type=int,default=19)
    p.add_argument('--request-id',type=int,default=19000031)
    p.add_argument('--start-layer',type=int,default=0)
    p.add_argument('--cycles',type=int,default=4)
    args=p.parse_args()
    assert 3<=args.cycles<=5
    target=args.output_dir.resolve();figdir=target/'figures';figdir.mkdir(parents=True,exist_ok=True)
    name='od_internal_cycle_zoom.png'
    old_pngs={str(x):sha(x) for x in figdir.glob('*.png') if x.name!=name}
    sources={n:args.run_dir.resolve()/n for n in ('result.json.gz','manifest.json.gz','command.json')}
    source_sha={n:sha(x) for n,x in sources.items()}
    result,manifest,command,requests,records,_=audit.load_raw(sources['result.json.gz'],sources['manifest.json.gz'],sources['command.json'])
    warm,demand,_,all_cycles,_=audit.analyze(result,requests,records,2000.,4000.,8)
    cycles=[c for c in all_cycles if c['npu_id']==args.npu and c['request_id']==args.request_id and c['same_request']
            and args.start_layer<=c['previous_layer']<args.start_layer+args.cycles]
    cycles.sort(key=lambda c:c['start_ms']);assert len(cycles)==args.cycles
    assert [c['previous_layer'] for c in cycles]==list(range(args.start_layer,args.start_layer+args.cycles))
    record=records[args.request_id];assert record['role']=='A'
    start,end=cycles[0]['start_ms'],cycles[-1]['end_ms'];duration=end-start
    assert 2000<=start<end<=4000
    volume=sum(record['V_per_ssu_GiB']);C=record['C_ms'];B=volume/(C/1000)
    rows=[]
    for c in cycles:
        assert c['previous_request_id']==args.request_id and c['full_io_stall_ms']>0
        assert c['warm_queue_ms']<1e-8
        audit.close(c['current_compute_C_ms'],C)
        audit.close(c['period_ms'],C+c['full_io_stall_ms'])
        audit.close(c['io_start_time_ms'],c['start_ms'])
        audit.close(c['io_ready_time_ms'],c['end_ms'])
        b=volume/(c['period_ms']/1000)
        audit.close(b,sum(c['derived_complete_cycle_supply_per_ssu_GiB_s']))
        audit.close(b/B,C/c['period_ms'])
        rows.append(dict(previous_layer=c['previous_layer'],next_layer=c['layer'],start_ms=c['start_ms'],end_ms=c['end_ms'],
                         C_ms=C,I_ms=c['full_io_stall_ms'],period_ms=c['period_ms'],V_GiB=volume,B_GiB_s=B,b_GiB_s=b,
                         ratio=b/B,io_start_ms=c['io_start_time_ms'],io_ready_ms=c['io_ready_time_ms'],ssd_complete_ms=None))
    for a,z in zip(rows,rows[1:]):audit.close(a['end_ms'],z['start_ms'])
    # A concurrent B example chosen by full coverage of this local interval.
    b_examples=[]
    for batch in result['summary']['microbatch_metrics']:
        rid=batch['member_request_ids'][0]
        if records[rid]['role']!='B':continue
        layers=batch['layer_metrics']
        for prev,nxt in zip(layers,layers[1:]):
            if prev['compute_start_ms']<=start and prev['compute_end_ms']>=end and nxt['io_start_time_ms']<=start and nxt['io_ready_time_ms']>=end:
                b_examples.append((batch['npu_id'],rid,prev,nxt))
    assert b_examples
    npu_b,rid_b,prev_b,next_b=sorted(b_examples,key=lambda x:(x[0],x[1]))[0]
    br=records[rid_b]
    b_context=dict(npu_id=npu_b,request_id=rid_b,compute_layer=prev_b['layer'],prefetch_layer=next_b['layer'],
                   compute_start_ms=prev_b['compute_start_ms'],compute_end_ms=prev_b['compute_end_ms'],C_ms=br['C_ms'],
                   next_layer_V_GiB=sum(br['V_per_ssu_GiB']),next_layer_V_per_ssu_GiB=br['V_per_ssu_GiB'],
                   io_submit_ms=next_b['io_start_time_ms'],io_ready_ms=next_b['io_ready_time_ms'],
                   io_outstanding_ms=next_b['io_ready_time_ms']-next_b['io_start_time_ms'],
                   next_compute_start_ms=next_b['compute_start_ms'],ssd_complete_ms=None)
    local_demand=[dict(start_ms=max(start,r['start_ms']),end_ms=min(end,r['end_ms']),
                       values=[r[f'D{s}_GiB_s'] for s in range(3)]) for r in demand if r['end_ms']>start and r['start_ms']<end]
    local_max=[max(r['values'][s] for r in local_demand) for s in range(3)]
    assert max(local_max)<40 and warm['demand']['strict_underload_all_disks']
    local_U=100*args.cycles*C/duration
    weighted_b=volume*args.cycles/(duration/1000)
    audit.close(weighted_b/B,local_U/100)

    shared.card_plot.setup_font()
    fig=plt.figure(figsize=(18,16.5))
    gs=fig.add_gridspec(5,1,height_ratios=[1.12,1.0,.83,1.10,1.12],hspace=.62,left=.095,right=.97,bottom=.12,top=.865)
    ax_t=fig.add_subplot(gs[0]);ax_b=fig.add_subplot(gs[1],sharex=ax_t)
    ax_context=fig.add_subplot(gs[2],sharex=ax_t);ax_d=fig.add_subplot(gs[3],sharex=ax_t)
    ax_table=fig.add_subplot(gs[4]);ax_table.axis('off')
    fig.suptitle('OD局部放大：名义需求低于盘容量，短请求仍会等数据',fontsize=22,y=.980)
    fig.text(.5,.951,'合成画像 + 对抗性地址选择 · 32 NPU / 3 SSU × 40 GiB/s · 同一张卡、同一个请求的4个连续完整层周期',ha='center',fontsize=12,color=DARK)
    fig.text(.5,.928,f'NPU {args.npu:02d} · A请求 {args.request_id} · 绝对时间 {start/1000:.6f}–{end/1000:.6f} 秒 · 原始层 L{args.start_layer}→L{args.start_layer+args.cycles}',ha='center',fontsize=12,color=DARK)
    fig.legend(handles=[Patch(color=ORANGE,label='A层计算'),Patch(color=GRAY,label='真实IO等待'),Line2D([],[],color=ORANGE,ls='--',lw=2,label='参考需求 B'),Line2D([],[],color=BLUE,lw=2,label='完整层周期平均 b')],loc='upper center',bbox_to_anchor=(.5,.911),ncol=4,frameon=False,fontsize=11)

    ax_t.set_title('1  计算结束后，下一层数据还没到：灰色就是等待 I',loc='left',fontsize=13,pad=14)
    for r in rows:
        x=r['start_ms']-start;T=r['period_ms'];I=r['I_ms']
        ax_t.broken_barh([(x,C)],(.25,.5),facecolor=ORANGE)
        ax_t.broken_barh([(x+C,I)],(.25,.5),facecolor=GRAY)
        ax_t.text(x+C/2,.50,f'C={C:.3f} ms',ha='center',va='center',fontsize=10,color='white')
        ax_t.text(x+C+I/2,.50,f'I={I:.3f} ms',ha='center',va='center',fontsize=10,color=DARK)
        ax_t.text(x+T/2,.91,f'L{r["previous_layer"]} → L{r["next_layer"]}；周期 {T:.3f} ms',ha='center',va='center',fontsize=10.5)
        ax_t.axvline(x,color='#999999',lw=.7,ls=':')
    ax_t.axvline(duration,color='#999999',lw=.7,ls=':')
    ax_t.set(ylim=(0,1.08),yticks=[],xlim=(0,duration));ax_t.tick_params(axis='x',labelbottom=False)
    ax_t.spines[['top','right','left']].set_visible(False)

    ax_b.set_title(f'2  A每层读取 {volume*1024:.2f} MiB：B=V/C={B:.3f} GiB/s，实际周期 b 约为其一半',loc='left',fontsize=13,pad=14)
    edges=[r['start_ms']-start for r in rows]+[duration]
    ax_b.axhline(B,color=ORANGE,ls='--',lw=2.3)
    ax_b.stairs([r['b_GiB_s'] for r in rows],edges,baseline=None,color=BLUE,lw=2.2)
    for r in rows:
        ax_b.text((r['start_ms']+r['end_ms'])/2-start,r['b_GiB_s']-.5,f'b={r["b_GiB_s"]:.3f}\nb/B={100*r["ratio"]:.2f}%',ha='center',va='top',fontsize=10.5,color=BLUE)
    ax_b.set(ylim=(0,8.2),ylabel='GiB/s',yticks=[0,2,4,6,8]);ax_b.grid(alpha=.18)
    ax_b.tick_params(axis='x',labelbottom=False)

    ax_context.set_title(f'3  同时存在大读取：NPU {npu_b:02d} 的B请求 {rid_b} 正在计算，下一层IO仍在途',loc='left',fontsize=13,pad=14)
    ax_context.broken_barh([(0,duration)],(.60,.25),facecolors=BLUE)
    ax_context.text(duration/2,.725,f'B本层计算：完整 C={br["C_ms"]:.3f} ms（此处仅显示局部）',ha='center',va='center',fontsize=11,color='white')
    ax_context.broken_barh([(0,duration)],(.15,.25),facecolors='#EDF5FA',edgecolors=BLUE,linestyles='--',linewidth=1.3)
    ax_context.text(duration/2,.275,f'下一层 {b_context["next_layer_V_GiB"]*1024:.3f} MiB IO在途：提交→数据到齐共 {b_context["io_outstanding_ms"]:.3f} ms；不是SSD服务条',ha='center',va='center',fontsize=11,color=BLUE)
    ax_context.set(ylim=(0,1),yticks=[]);ax_context.tick_params(axis='x',labelbottom=False)
    ax_context.spines[['top','right','left']].set_visible(False)

    ax_d.set_title('4  同一时段逐盘名义需求均 <40 GiB/s；这不保证每次预取都能赶上各自计算期限',loc='left',fontsize=13,pad=14)
    d_edges=[r['start_ms']-start for r in local_demand]+[duration]
    d_colors=['#0072B2','#009E73','#CC79A7']
    for s in range(3):ax_d.stairs([r['values'][s] for r in local_demand],d_edges,baseline=None,color=d_colors[s],lw=2,label=f'SSU{s}：峰值 {local_max[s]:.3f}')
    ax_d.axhline(40,color='#333333',ls='--',lw=1,label='每盘容量40')
    ax_d.set(ylim=(0,43),ylabel='GiB/s',xlabel=f'相对 {start/1000:.6f} 秒的时间（毫秒）')
    ax_d.set_xticks(np.arange(0,math.ceil(duration)+1,2));ax_d.set_xlim(0,duration);ax_d.grid(alpha=.18)
    ax_d.legend(loc='lower left',ncol=4,frameon=False,fontsize=10)

    ax_table.set_title('四个完整周期的核算（原始层编号从L0开始；所有周期都在同一个A请求内）',loc='left',fontsize=12,pad=12)
    table_rows=[[f'L{r["previous_layer"]}→L{r["next_layer"]}',f'{C:.6f}',f'{r["I_ms"]:.6f}',f'{r["period_ms"]:.6f}',f'{r["b_GiB_s"]:.6f}',f'{r["ratio"]*100:.3f}%'] for r in rows]
    table=ax_table.table(cellText=table_rows,colLabels=['周期','计算 C（ms）','等待 I（ms）','周期 C+I（ms）','b=V/(C+I)（GiB/s）','b/B=C/(C+I)'],loc='center',cellLoc='center',colWidths=[.12,.16,.16,.18,.20,.18])
    table.auto_set_font_size(False);table.set_fontsize(10.5);table.scale(1,1.75)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#D7DEE4')
        if row==0:cell.set_facecolor('#E9EFF4');cell.set_text_props(weight='bold')
        elif row%2==0:cell.set_facecolor('#F6F8FA')

    fig.text(.095,.079,f'这4个周期的该卡U={local_U:.2f}%；整机warm [2,4)的U仍为{warm["fleet_U_percent"]:.2f}%。局部示例用于解释等待，不能代替整机或长期统计。',fontsize=11,color=DARK)
    fig.text(.095,.055,'b由完整周期的下一层读取量除以周期时长重建；这里同一请求各层V、C不变且无额外计算排队，因此 b/B 与该周期计算占比相等。',fontsize=10.5,color=DARK)
    fig.text(.095,.032,'io_ready是数据到达NPU的时刻，不是SSD完成时刻。B的IO在途仅证明并发重叠，不能单凭该图确定某盘或某Path造成了A的等待。',fontsize=10.5,color=DARK)
    fig.savefig(figdir/name,dpi=200,facecolor='white');plt.close(fig)

    assert source_sha=={n:sha(x) for n,x in sources.items()}
    assert old_pngs=={n:sha(Path(n)) for n in old_pngs}
    evidence=dict(all_checks_passed=True,source_paths={n:str(x) for n,x in sources.items()},source_sha256=source_sha,
        script_sha256=sha(__file__),policy=result['strategy'],input_kind='synthetic profiles and adversarial address selection',
        selection='A request with four clear consecutive internal waits; illustrative local interval, not a new fleet metric window',
        selected_npu=args.npu,selected_request=args.request_id,window_ms=[start,end],cycles=rows,
        A_volume_GiB=volume,A_C_ms=C,A_B_GiB_s=B,local_per_npu_U_percent=local_U,local_weighted_b_GiB_s=weighted_b,
        warm_fleet_U_percent=warm['fleet_U_percent'],warm_per_disk_demand_max_GiB_s=warm['demand']['maximum_GiB_s_by_ssu'],
        local_per_disk_demand_max_GiB_s=local_max,local_demand_segments=local_demand,B_context=b_context,
        source_ssd_layer_completion_available=False,ssd_service_timeline_drawn=False,same_request_only=True,
        old_png_sha256=old_pngs,old_png_unchanged=True,output_png='figures/'+name,output_sha256=sha(figdir/name),visual_review='pending')
    (target/'local_zoom_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    lines=['# 同一短请求的完整层周期放大','',f'[打开PNG](figures/{name})','',
        f'采用正式完成的OD结果，NPU {args.npu}、请求 {args.request_id}，{start/1000:.9f}–{end/1000:.9f}秒；原始L{args.start_layer}到L{args.start_layer+args.cycles}的4个连续周期，均不跨请求。合成画像和对抗性地址选择，不是原始data样本。','',
        f'A每层V={volume*1024:.6f} MiB，C={C:.9f} ms，B=V/C={B:.9f} GiB/s。I由数据到齐时刻减去上一层计算结束得到，逐周期与原始io_barrier_wait匹配。','',
        '|周期|C ms|I ms|C+I ms|b GiB/s|b/B|','|---|---:|---:|---:|---:|---:|']
    lines.extend('|'+ '|'.join(row)+'|' for row in table_rows)
    lines.extend(['',f'4周期该卡U={local_U:.6f}%，整机warm U={warm["fleet_U_percent"]:.6f}%。这里V、C均恒定、没有额外计算排队；b按完整周期平均，b/B=C/(C+I)。这不是瞬时公式，不适用于把跨请求读取直接除以当前请求需求。',
        '',f'局部逐盘名义需求峰值为 {local_max} GiB/s；全warm逐盘峰值为 {warm["demand"]["maximum_GiB_s_by_ssu"]} GiB/s，均低于40。名义需求仍按当前请求V/C统计，跨请求预取不额外叠加。',
        '',f'并发B例子：NPU{npu_b}、请求{rid_b}，本层计算 [{prev_b["compute_start_ms"]/1000:.9f}, {prev_b["compute_end_ms"]/1000:.9f}) 秒，C={br["C_ms"]:.9f}ms。下一层读取{b_context["next_layer_V_GiB"]*1024:.6f}MiB，IO提交{next_b["io_start_time_ms"]/1000:.9f}秒，数据到齐{next_b["io_ready_time_ms"]/1000:.9f}秒，IO在途{b_context["io_outstanding_ms"]:.9f}ms。',
        '', 'IO在途包含排队、SSD服务与传输，不能画作全程SSD服务。源结果没有层级SSD最后完成时刻，图中明确不提供虚构服务条，也不把io_ready标成SSD完成。并发B只说明重叠，不能单独识别哪张盘或哪条Path的因果贡献。',
        '', '图解释的是：总的平均速率约束成立，仍不保证每次预取在各自计算预算内完成。此局部例子不证明整机利用率已降至80几，也不证明长期保持同样程度的退化。',''])
    (target/'README_local_zoom.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(figdir/name),local_npu_U_percent=local_U,warm_fleet_U_percent=warm['fleet_U_percent'],local_D_max=local_max),ensure_ascii=False))


if __name__=='__main__':main()
