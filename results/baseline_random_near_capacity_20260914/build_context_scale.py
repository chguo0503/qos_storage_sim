#!/usr/bin/env python3
"""Four separate PNGs for the completed high-B-short context seed7 controls."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qos_context_scale_mpl'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import goal_80s_update as independent

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'/'context_scale'
FONT=Path('/home/chguo/.fonts/msyh.ttc')
if not FONT.exists():FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
    'axes.unicode_minus':False,'font.size':11,'savefig.facecolor':'white'})
BLUE='#1459C9';ORANGE='#DB7B08';RED='#BE453F';PURPLE='#8D4CA5';GRAY='#798794';INK='#17354D'
KS=[200,256,384,512]
SUBTITLE='32 NPU · 8 SSU × 40 GiB/s · Baseline / 独立 Random · 探索 seed7（n=1）'
PROVENANCE='短请求固定总10K / miss128，C为外推；长请求miss1024：200K长C来自data，256/384/512K长C外推。'
LIMIT='理想输入平均 rho≈1，不等于逐盘逐时欠载；外推不代表真实硬件支持该上下文或已测得其计算时间。'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def base(title,ylabel,foot_extra=''):
    fig,ax=plt.subplots(figsize=(13.5,7.6))
    fig.subplots_adjust(left=.105,right=.97,top=.79,bottom=.245)
    fig.text(.055,.945,title,fontsize=20,fontweight='bold',color=INK)
    fig.text(.055,.89,SUBTITLE,fontsize=11.5,color=GRAY)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ACB8C3')
    ax.grid(axis='y',color='#DDE4EA',alpha=.75)
    ax.set_axisbelow(True)
    ax.set_xlabel('长请求总输入长度（K token；1K=1024 token）',labelpad=12)
    ax.set_ylabel(ylabel,labelpad=12)
    ax.set_xticks(KS)
    ax.set_xlim(180,535)
    fig.text(.055,.14,foot_extra,fontsize=11,color=INK)
    fig.text(.055,.094,PROVENANCE,fontsize=10,color=GRAY)
    fig.text(.055,.05,LIMIT,fontsize=10,color=GRAY)
    return fig,ax


def save(fig,name):
    path=OUT/name;fig.savefig(path,dpi=180);plt.close(fig)
    return path


def collect():
    rows=[];sources={}
    for K in KS:
        case=HERE/f'runs/context8_L{K}m1024_S10m128_ssu8_h22000_seed7/baseline'
        path=case/'analysis.json';a=json.loads(path.read_text())
        assert a['seed']==7 and a['strategy']=='baseline' and a['num_ssu']==8
        row=independent.review_case(path)
        raw_path=case/'result.json.gz'
        with gzip.open(raw_path,'rt') as stream:raw=json.load(stream)
        waits=[]
        for batch in raw['summary']['microbatch_metrics']:
            layers=batch['layer_metrics']
            if not math.isclose(layers[0]['compute_duration_ms'],row['profiles']['S']['C_ms'],abs_tol=1e-8):continue
            for previous,current in zip(layers,layers[1:]):
                if previous['compute_start_ms']>=2000 and current['compute_start_ms']<=20000:
                    waits.append(max(0.,current['compute_start_ms']-previous['compute_end_ms']))
        positives=[w for w in waits if w>1e-7]
        long=next(w for w in a['windows'] if (w['start_ms'],w['end_ms'])==(2000,20000))
        expected=long['complete_internal_cycles_inside_window']['per_role']['S']
        assert len(waits)==expected['count'] and len(positives)==expected['late_layer_count']
        independent.near(statistics.mean(waits),row['observed_short_internal_wait_ms'])
        row.update(long_total_K=K,short_internal_complete_count=len(waits),
            short_internal_positive_count=len(positives),short_internal_zero_fraction=1-len(positives)/len(waits),
            short_internal_all_median_ms=statistics.median(waits),
            short_internal_positive_mean_ms=statistics.mean(positives),
            short_internal_positive_median_ms=statistics.median(positives),
            finite8_model_error_pp=row['observed_7internal_1L0_substitution_U_percent']-row['observed_long_U_percent'])
        for filename in ('analysis.json','result.json.gz','manifest.json.gz'):
            p=case/filename;sources[str(p.relative_to(HERE))]=sha(p)
        assert row['warm_mixed_cards']==row['long_mixed_cards']==32 and row['all_32_active_long']
        warm=next(w for w in a['windows'] if (w['start_ms'],w['end_ms'])==(2000,4000))
        assert warm['all_32_active']
        rows.append(row)
    return rows,sources


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows,sources=collect();artifacts=[]
    fig,ax=base('01  长读取变大，但每层带宽需求 B 基本不变','每层需求 B=V/C（GiB/s）',
        '灰线=总容量320/32=10 GiB/s，仅为均分容量参考；蓝、紫线都是请求需求，不是SSU实际供给。')
    long_b=[r['profiles']['L']['B_GiB_s'] for r in rows]
    short_b=[r['profiles']['S']['B_GiB_s'] for r in rows]
    ax.plot(KS,long_b,'o-',color=BLUE,lw=2.6,ms=8,label='长请求 B_L')
    ax.plot(KS,short_b,'s-',color=PURPLE,lw=2.6,ms=7,label='短请求 B_S（恒定）')
    ax.axhline(10,color=GRAY,ls='--',lw=1.8,label='容量 / 32：参考值10')
    ax.set_ylim(0,22)
    for k,v in zip(KS,long_b):ax.annotate(f'{v:.3f}',(k,v),xytext=(0,-21),textcoords='offset points',ha='center',color=BLUE)
    ax.annotate(f'{short_b[-1]:.3f}',(KS[-1],short_b[-1]),xytext=(0,12),textcoords='offset points',ha='center',color=PURPLE)
    ax.legend(loc='upper left',ncol=3,frameon=False,fontsize=11)
    artifacts.append(save(fig,'01_bandwidth_demand.png'))

    fig,ax=base('02  长读取越大，短层平均暴露等待越长','短内部层平均等待 w（ms，包含零等待）',
        '同请求完整周期：当前层compute_start→下一层compute_start；只收完整落在长窗[2,20)s的周期。')
    means=[r['observed_short_internal_wait_ms'] for r in rows]
    ax.plot(KS,means,'o-',color=ORANGE,lw=2.8,ms=9)
    ax.set_ylim(0,.7)
    for k,v,r in zip(KS,means,rows):
        ax.annotate(f'{v:.6f} ms',(k,v),xytext=(0,13),textcoords='offset points',ha='center',fontsize=12,fontweight='bold',color=ORANGE)
        ax.text(k,.052,f"N={r['short_internal_complete_count']:,}\n正等待 {100*(1-r['short_internal_zero_fraction']):.2f}%",ha='center',va='bottom',fontsize=10,color=GRAY)
    ax.text(.02,.94,'四组全体短层等待的中位数均为0；有等待的少数层仍可拉高平均值。',transform=ax.transAxes,color=INK,fontsize=11)
    artifacts.append(save(fig,'02_short_internal_mean_wait.png'))

    fig,ax=base('03  固定 warm 与长窗，都看真实 NPU 利用率','整机实际平均 NPU 利用率（%）',
        '实测：计算区间与统计窗的交集 / (32×窗口长度)。四组两个窗口均32卡全程活跃、每卡两类都有计算。')
    warm=[r['observed_warm_U_percent'] for r in rows];long=[r['observed_long_U_percent'] for r in rows]
    ax.plot(KS,warm,'s-',color=ORANGE,lw=2.4,ms=7,label='warm [2,4)s：真实 U')
    ax.plot(KS,long,'o-',color=BLUE,lw=2.4,ms=8,label='长窗 [2,20)s：真实 U')
    ax.axhline(90,color=GRAY,lw=1.3,ls='--')
    ax.set_ylim(80,100)
    for k,w,l in zip(KS,warm,long):
        ax.annotate(f'{w:.3f}%',(k,w),xytext=(0,13),textcoords='offset points',ha='center',color=ORANGE,fontsize=11)
        ax.annotate(f'{l:.3f}%',(k,l),xytext=(0,-23),textcoords='offset points',ha='center',color=BLUE,fontsize=11)
    ax.legend(loc='upper right',frameon=False)
    ax.text(.02,.94,'单seed探索：未画误差条，不冒称多种子或无限长期结论。',transform=ax.transAxes,color=GRAY)
    artifacts.append(save(fig,'03_actual_utilization.png'))

    fig,ax=base('04  少掉的利用率，主要损失在哪里','长窗 [2,20)s 的全部卡时间（% / 百分点）',
        '精确记账：计算U + 短内部等待 + 长内部等待 + 首层暴露等待 + 空闲 = 100%；没有丢弃边界时间。')
    ax.set_xlabel('长请求总输入长度（K token；宽度仅用于排版）')
    ax.set_xticks(range(4),[str(k) for k in KS]);ax.set_xlim(-.6,3.6);ax.set_ylim(0,112)
    colors=[BLUE,ORANGE,PURPLE,RED,'#BBC7D1']
    names=['计算（实际U）','短内部等待','长内部等待','L0/交接等待','空闲']
    matrix=[]
    for r in rows:
        loss=r['exact_long_loss_pp']
        vals=[r['observed_long_U_percent'],loss['short_internal'],loss['long_internal'],loss['L0_exposed_handoff'],loss['idle']]
        independent.near(sum(vals),100.)
        matrix.append(vals)
    x=np.arange(4);bottom=np.zeros(4)
    for j,(color,name) in enumerate(zip(colors,names)):
        values=np.array([v[j] for v in matrix]);ax.bar(x,values,bottom=bottom,width=.56,color=color,label=name)
        if j==0:
            for k,v in enumerate(values):ax.text(k,45,f'{v:.3f}%',ha='center',va='center',color='white',fontweight='bold',fontsize=13)
        elif j==1:
            for k,v in enumerate(values):ax.text(k,bottom[k]+v/2,f'{v:.3f}pp',ha='center',va='center',color='white',fontsize=10)
        elif j==3:
            for k,v in enumerate(values):ax.annotate(f'L0 {v:.3f}pp',(k,bottom[k]+v/2),xytext=(k,103),textcoords='data',ha='center',fontsize=10,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=.8))
        bottom+=values
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.08),ncol=5,frameon=False,fontsize=10)
    artifacts.append(save(fig,'04_exact_time_loss.png'))
    definitions=dict(B='Per-layer required bandwidthV/C; both class lines are demand. 320/32=10 is only a mean-capacity reference, not actual grant.',
        mean_wait='All same-request internal cycles entirely inside[2000,20000)ms, zero waits included. Positive threshold>1e-7ms.',
        actual_U='Exact compute-window overlap/(32*T); fullwarm[2,4)s andlong[2,20)s, no clipping of denominator by activity.',
        exact_loss='Exact window overlap: compute+short_internal+long_internal+exposedL0+idle=100%.',
        finite8='U≈8ΣnC/(8ΣnC+7Σn*w_internal+Σn*d_L0); substitutes observed per-role waits, so it is posthoc explanation, not independent prediction.',
        zoom='A positive-wait selected layer or positive-only median is not the all-cycle mean. Context384 all-cycle mean is.467224ms; context512 is.529443ms.')
    data=dict(updated_utc=datetime.now(timezone.utc).isoformat(),scope=SUBTITLE,definitions=definitions,rows=rows,
        warnings=[PROVENANCE,LIMIT,'n=1探索，完整32NPU独立随机输入；其他确认seed结果不在本四图。'],source_sha256=sources)
    data_path=OUT/'plot_data.json';data_path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# Context尺度：四张独立PNG','',SUBTITLE+'。本页仅固定seed7四组，后续确认seed不混入。','',
        '1. [每层带宽需求B](01_bandwidth_demand.png)：长B约7.54→7.82，短B18.50恒定。10GiB/s是容量/32参考，不是实际供给。',
        '2. [短内部层平均等待](02_short_internal_mean_wait.png)：所有完整短内部周期，包括零等待。',
        '3. [warm与长窗实际U](03_actual_utilization.png)：固定两个窗口，全32卡活跃、每卡均有长短计算。',
        '4. [精确卡时间分解](04_exact_time_loss.png)：实际计算、短/长内部等待、L0/交接与idle加总100%。','',
        PROVENANCE+' '+LIMIT,'',
        '| 长总K | L:S | rho | 全体短内部均w ms | 正等待层比例 | 正等待子集均值/中位数 ms | warm U | 长窗 U |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        ratio=r['count_ratio_by_role'];lines.append(f"| {r['long_total_K']} | {ratio['L']}:{ratio['S']} | {r['ideal_load_ratio']:.6f} | {r['observed_short_internal_wait_ms']:.6f} | {100*(1-r['short_internal_zero_fraction']):.2f}% | {r['short_internal_positive_mean_ms']:.6f}/{r['short_internal_positive_median_ms']:.6f} | {r['observed_warm_U_percent']:.6f}% | {r['observed_long_U_percent']:.6f}% |")
    lines+=['', '全体短内部层等待中位数四组均为0。`.467224ms`来自384K组，512K全体均值为`.529443ms`。局部zoom选出的正等待层（例如约1.467ms）既不是全体均值，也不能跨组比较；正等待子集的具体均值/中位数见表。', '',
        '有限8层公式与实测：','', '```text',
        'U ≈ 8*sum(n*C) / [8*sum(n*C)+7*sum(n*w内部)+sum(n*d首层)]',
        '这里n是输入请求数比例；C是每层计算，w和d来自本次已经完成的日志。','```','',
        '这是代入实测的解释，不是独立预测。真实长窗还会有完成混合比例及边界差异；精确损失分解不依赖此近似。', '',
        '| 长总K | 代入实测的有限8层U | 实际长窗U | 近似−实际 pp | 短内部损失pp | 长内部pp | L0损失pp | idle pp |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        l=r['exact_long_loss_pp'];lines.append(f"| {r['long_total_K']} | {r['observed_7internal_1L0_substitution_U_percent']:.6f}% | {r['observed_long_U_percent']:.6f}% | {r['finite8_model_error_pp']:+.6f} | {l['short_internal']:.6f} | {l['long_internal']:.6f} | {l['L0_exposed_handoff']:.6f} | {l['idle']:.6f} |")
    lines+=['', '可支持的结论：这四组输入中，长V和C同时增大、B变化较小，实测短内部等待增加，实际U下降；精确损失显示短内部等待是主要变化项。它支持本仿真中绝对读取规模影响layerwise等待的机制，不等于已证明一般FIFO必然下降或Once必然改善。比例、长请求频率、统计窗内完成混合也随输入改变，不能忽略。', '',
        '[完整数值与逐文件哈希](plot_data.json) · [生成器与PNG哈希](sources.json) · [重绘脚本](../../build_context_scale.py)', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    assert all(sha(HERE/path)==digest for path,digest in sources.items())
    own_sources=dict(generator_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(HERE/'goal_80s_update.py'),
        source_sha256=sources,plot_data_sha256=sha(data_path),artifacts={p.name:sha(p) for p in artifacts},
        untouched='Only figures/context_scale outputs were written; no source data, frozen plans, originaloverview or previous case PNG modified.',
        no_new_simulation=True)
    (OUT/'sources.json').write_text(json.dumps(own_sources,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(outputs=[str(p.relative_to(HERE)) for p in artifacts],verified_cases=len(rows))))


if __name__=='__main__':main()
