#!/usr/bin/env python3
"""Plot admission-relative TTFT CDFs and SLO x1.5 rates from frozen exports."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'mixed8_slo_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mixed8_complete_figures_20260914'
POLICIES=['fifo','once','short_first']
LABELS={'fifo':'Baseline FIFO','once':'Once per layer (5 ms)','short_first':'短读取优先（诊断）'}
COLORS={'fifo':'#0068d9','once':'#258a62','short_first':'#aa4a94'}
INK,MUTED='#172d45','#546980'


def verify(rows, summary):
    assert len(rows)==1728 and len({(q['policy'],q['request_id']) for q in rows})==1728
    for q in rows:
        ttft=q['completion_time_ms']-q['admission_time_ms']
        ideal=8*q['per_layer_compute_ms']
        assert math.isclose(ttft,q['ttft_admission_ms'],abs_tol=1e-8)
        assert math.isclose(ideal,q['ideal_compute_8layers_ms'],abs_tol=1e-8)
        assert math.isclose(ttft/ideal,q['ttft_over_ideal'],abs_tol=1e-10)
        assert q['slo_1p5_passed']==(ttft<=1.5*ideal+1e-8)
        assert q['window_admitted']==(2000<=q['admission_time_ms']<4000)
    for s in summary:
        selected=[q for q in rows if q['policy']==s['policy']
            and (s['cohort']=='all_requests' or q['window_admitted'])
            and (s['role']=='all' or q['role']==s['role'])]
        assert len(selected)==s['request_count']
        assert sum(q['slo_1p5_passed'] for q in selected)==s['passed_count']
    return True


def style_axis(ax):
    ax.grid(axis='both',color='#e8edf2',linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=12)


def cdf_figure(rows, summary, cohort, path):
    fig,ax=plt.subplots(figsize=(14,7.5),dpi=180)
    fig.subplots_adjust(left=.085,right=.975,bottom=.23,top=.71)
    title='Warm [2,4)s 内接纳的请求' if cohort=='window_admissions' else '完整的同一组 576 条请求'
    fig.text(.065,.95,'TTFT SLO × 1.5：三种策略的累积分布',fontsize=23,color=INK)
    fig.text(.065,.89,f'8 NPU / 1 SSU × 40 GiB/s · 每卡长短随机混合 · {title}',fontsize=14,color=MUTED)
    maximum=max(q['ttft_over_ideal'] for q in rows)
    xmax=math.ceil(maximum*10)/10+.04
    points=[]
    for p in POLICIES:
        selected=[q for q in rows if q['policy']==p and (cohort=='all_requests' or q['window_admitted'])]
        x=np.sort([q['ttft_over_ideal'] for q in selected])
        y=np.arange(1,len(x)+1)/len(x)*100
        s=next(s for s in summary if s['policy']==p and s['cohort']==cohort and s['role']=='all')
        label=f"{LABELS[p]}：{s['passed_count']}/{s['request_count']} = {s['pass_rate']*100:.2f}%"
        ax.step(np.r_[.98,x,xmax],np.r_[0,y,100],where='post',color=COLORS[p],linewidth=2.2,label=label)
        for a,b in zip(x,y):points.append(dict(cohort=cohort,policy=p,ttft_over_ideal=float(a),cdf_percent=float(b)))
    ax.axvline(1.5,color='#697481',linestyle='--',linewidth=1.5)
    ax.text(1.515,7,'达标阈值 1.5',color=INK,fontsize=12)
    ax.set(xlim=(.98,xmax),ylim=(0,103),xlabel='TTFT / 纯计算时间（8 × 每层计算时间）',ylabel='累计请求比例（%）')
    ax.set_yticks(np.arange(0,101,20))
    ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
    style_axis(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.063,.84),frameon=False,ncol=1,fontsize=12)
    fig.text(.065,.12,'TTFT = 完成时刻 - NPU 接纳时刻；每请求阈值 = 1.5 × 自身 8 层纯计算时间。',fontsize=12,color=INK)
    note='三种策略执行进度不同，窗口内接纳的请求集合不同；使用完整运行中的完成时间判定达标。' if cohort=='window_admissions' else '三种策略逐条比较同一批输入；统计包含预热和运行尾段，因此与 warm 窗口达标率不同。'
    fig.text(.065,.075,note,fontsize=11.5,color=MUTED)
    fig.text(.065,.035,'20K 计算时间含 data 外推；短读取优先仅为诊断。虚线左侧（含阈值）的累计比例即 SLO 达标率。',fontsize=11.5,color=MUTED)
    fig.savefig(path,facecolor='white');plt.close(fig)
    return points


def rates_figure(summary,path):
    fig,axes=plt.subplots(1,2,figsize=(15,7.4),dpi=180,sharey=True)
    fig.subplots_adjust(left=.065,right=.98,bottom=.245,top=.73,wspace=.14)
    fig.text(.055,.95,'TTFT SLO × 1.5：整体、短流与长流达标率',fontsize=23,color=INK)
    fig.text(.055,.89,'8 NPU / 1 SSU · 每卡 L:S = 1:11 · 3 种策略使用相同完整输入',fontsize=14,color=MUTED)
    for ax,cohort,title in zip(axes,['window_admissions','all_requests'],['Warm [2,4)s 接纳集合','完整的同一组 576 条请求']):
        for j,p in enumerate(POLICIES):
            vals=[next(s for s in summary if s['cohort']==cohort and s['role']==r and s['policy']==p) for r in ['all','S','L']]
            xs=np.arange(3)+(j-1)*.245
            ax.bar(xs,[100*s['pass_rate'] for s in vals],width=.22,color=COLORS[p],label=LABELS[p],zorder=2)
            for x,s in zip(xs,vals):
                ax.text(x,100*s['pass_rate']+1.4,f"{100*s['pass_rate']:.2f}%\n{s['passed_count']}/{s['request_count']}",ha='center',va='bottom',fontsize=9.5,color=INK)
        ax.set_xticks(range(3),['整体','短流 S（约20K）','长流 L（约200K）'])
        ax.set_ylim(0,115);ax.set_yticks(np.arange(0,101,20))
        ax.set_title(title,fontsize=15,color=INK,pad=15)
        style_axis(ax);ax.grid(axis='x',visible=False)
    axes[0].set_ylabel('达标请求比例（%）',fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.05,.83),frameon=False,ncol=3,fontsize=12)
    fig.text(.055,.13,'柱顶数字为达标率及“达标数 / 请求数”；SLO = TTFT ≤ 1.5 × 每请求自身 8 层纯计算时间。',fontsize=12,color=INK)
    fig.text(.055,.082,'Warm 集合随执行进度变化；右图对比同一完整输入集合。TTFT 从 NPU 接纳开始，不计入接纳前排队。',fontsize=11.5,color=MUTED)
    fig.text(.055,.038,'L/S 是本实验长短标签（源码 QoS 分类分别为 LL/SL）；20K 计算时间含 data 外推。',fontsize=11.5,color=MUTED)
    fig.savefig(path,facecolor='white');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--font',type=Path,required=True)
    a=p.parse_args()
    matplotlib.font_manager.fontManager.addfont(str(a.font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(a.font)).get_name(),'axes.unicode_minus':False,'font.size':12})
    rows=json.loads((OUT/'data/request_execution.json').read_text())
    summary=json.loads((OUT/'data/slo_1p5_summary_from_details.json').read_text())
    assert verify(rows,summary)
    dest=OUT/'images/ttft_slo';dest.mkdir(parents=True,exist_ok=True)
    points=[]
    for cohort in ['window_admissions','all_requests']:
        points.extend(cdf_figure(rows,summary,cohort,dest/f'ttft_slo15_cdf_{cohort}.png'))
    rates_figure(summary,dest/'ttft_slo15_rates_by_class.png')
    with (OUT/'data/ttft_slo15_cdf_points.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    audit=dict(request_records=len(rows),all_ttft_and_thresholds_verified=True,all_cohort_counts_match_export=True,
        input_sha256=hashlib.sha256((OUT/'data/request_execution.json').read_bytes()).hexdigest(),
        cdf_x_definition='admission-relative TTFT / (8 * per-layer compute ms)',threshold=1.5,
        images=[p.name for p in sorted(dest.glob('*.png'))],summary=summary)
    (OUT/'data/ttft_slo15_figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='summary'},ensure_ascii=False))


if __name__=='__main__':main()
