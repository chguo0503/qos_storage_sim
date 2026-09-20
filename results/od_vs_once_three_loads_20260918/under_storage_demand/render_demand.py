#!/usr/bin/env python3
"""Plot the original underload workload's exact layer V/C demand, seed 7."""
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
POLICIES=('asu_baseline','od_baseline','once')
NAMES={'asu_baseline':'ASU Baseline','od_baseline':'OD Baseline','once':'流量分配策略（Once）'}
STYLE={'asu_baseline':('#555555','--',3.0),'od_baseline':('#D55E00','-.',2.2),'once':('#0072B2','-',1.4)}


def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    cdf=read(HERE.parent/'three_strategy_cdf/plot_data.json')
    cases=[];fingerprints=set();sources={}
    for policy in POLICIES:
        entry=next(c for c in cdf['cases'] if c['regime']=='under' and c['policy']==policy and c['seed']==7)
        path=ROOT/entry['result'];command=read(path.parent/'command.json')
        assert command['status']=='complete' and command['completed_simulation']
        assert sha(path)==command['result_sha256']
        assert sha(path.parent/'manifest.json.gz')==command['manifest_sha256']
        fingerprints.add((command['manifest_sha256'],command['input_fingerprint']))
        sources[str(path.relative_to(ROOT))]=sha(path)
        raw=read(path);warm=next(w for w in raw['analysis'] if w['start_ms']==2000 and w['end_ms']==4000)
        assert warm['all_npus_active'] and warm['demand']['strict_underload_all_disks']
        segments=np.asarray(warm['demand']['segments'],dtype=float)
        assert segments.shape[1]==5 and segments[0,0]==2000 and segments[-1,1]==4000
        assert np.allclose(segments[:-1,1],segments[1:,0],rtol=0,atol=1e-9)
        duration=segments[:,1]-segments[:,0];disk=segments[:,2:]
        assert np.all(disk<40) and np.all(duration>0)
        mean=(disk*duration[:,None]).sum(axis=0)/2000
        peak=disk.max(axis=0);total=disk.sum(axis=1)
        assert np.allclose(mean,warm['demand']['per_disk_mean_GiB_s'],rtol=0,atol=1e-9)
        assert np.allclose(peak,warm['demand']['per_disk_max_GiB_s'],rtol=0,atol=1e-9)
        cases.append(dict(policy=policy,source=str(path.relative_to(ROOT)),segments=segments.tolist(),
                          per_disk_mean_GiB_s=mean.tolist(),per_disk_peak_GiB_s=peak.tolist(),
                          total_mean_GiB_s=float(np.dot(total,duration)/2000),
                          total_peak_GiB_s=float(total.max())))
    assert len(fingerprints)==1
    return cases,sources


def plot_line(ax,case,disk=None,label=None):
    s=np.asarray(case['segments']);y=s[:,2:].sum(axis=1) if disk is None else s[:,2+disk]
    x=np.r_[s[:,0],s[-1,1]]/1000
    color,style,width=STYLE[case['policy']]
    ax.step(x,np.r_[y,y[-1]],where='post',color=color,linestyle=style,linewidth=width,
            label=label or NAMES[case['policy']])


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    font=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,
                         'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    cases,sources=prepare()
    fig,ax=plt.subplots(figsize=(12.8,7.0))
    fig.subplots_adjust(left=.09,right=.975,top=.78,bottom=.22)
    for c in cases:
        plot_line(ax,c,label=f"{NAMES[c['policy']]}  ·  均值 {c['total_mean_GiB_s']:.2f} GiB/s")
    ax.axhline(120,color='#777777',linestyle=':',linewidth=1.4)
    ax.text(3.98,121.5,'3 盘总容量：120 GiB/s',ha='right',color='#555555')
    ax.set(xlim=(2,4),ylim=(0,128),xlabel='时间（秒）',ylabel='32 张 NPU 的存储总带宽需求（GiB/s）')
    ax.xaxis.set_major_locator(MultipleLocator(.25));ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis='y',alpha=.25)
    ax.legend(loc='lower left',framealpha=.96,handlelength=3.6,labelspacing=.75)
    fig.text(.09,.95,'持续欠载：存储读取带宽需求',fontsize=19,fontweight='bold',va='top')
    fig.text(.09,.887,'32 NPU / 3 SSU × 40 GiB/s  ·  Ring hash  ·  Random  ·  seed 7  ·  warm [2,4) 秒',color='#555555')
    fig.text(.09,.828,f"总需求峰值 {max(c['total_peak_GiB_s'] for c in cases):.2f} GiB/s；三策略曲线高度重合。",fontsize=11.5)
    fig.text(.09,.108,'需求 = 各卡当前请求每层读取量 V / 每层纯计算时间 C，再对 32 张卡求和。',fontsize=10.5)
    fig.text(.09,.054,'这是参考需求，不是 SSD 实际读取速率；曲线保留真实请求切换时刻，不把多个种子的时序平均。',fontsize=10,color='#666666')
    outputs=[HERE/'under_total_storage_demand.png'];fig.savefig(outputs[-1],dpi=180,facecolor='white');plt.close(fig)

    fig,axes=plt.subplots(3,1,figsize=(12.8,10.0),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.975,top=.80,bottom=.16,hspace=.28)
    for disk,ax in enumerate(axes):
        for c in cases:plot_line(ax,c,disk)
        ax.axhline(40,color='#777777',linestyle=':',linewidth=1.2)
        ax.text(3.985,40.8,'单盘容量 40',ha='right',fontsize=9.5,color='#555555')
        peak=max(c['per_disk_peak_GiB_s'][disk] for c in cases)
        ax.set_title(f'SSU {disk}  ·  最高需求 {peak:.2f} GiB/s',loc='left',fontsize=11.5,pad=8)
        ax.set(xlim=(2,4),ylim=(0,43),ylabel='带宽需求（GiB/s）')
        ax.yaxis.set_major_locator(MultipleLocator(10));ax.xaxis.set_major_locator(MultipleLocator(.25))
        ax.grid(axis='y',alpha=.25)
    axes[-1].set_xlabel('时间（秒）')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.083,.866),ncol=3,frameon=False,handlelength=3.6)
    fig.text(.09,.965,'持续欠载：逐盘存储带宽需求',fontsize=19,fontweight='bold',va='top')
    fig.text(.09,.918,'32 NPU / 3 SSU × 40 GiB/s  ·  Ring hash  ·  Random  ·  seed 7  ·  warm [2,4) 秒',color='#555555')
    fig.text(.09,.085,'每盘需求 = 所有当前请求落在该盘的每层读取量 / 各自纯计算时间，再求和。',fontsize=10.5)
    fig.text(.09,.044,'三策略逐盘、逐事件均低于 40 GiB/s；参考需求按每层 V/C 计算，同一请求各层相同。',fontsize=10,color='#666666')
    outputs.append(HERE/'under_per_ssu_storage_demand.png');fig.savefig(outputs[-1],dpi=180,facecolor='white');plt.close(fig)
    data=dict(seed=7,window_ms=[2000,4000],units='GiB/s',definition='current request actual per-disk layer V / pure C, summed over 32 active NPUs',
              no_temporal_averaging=True,no_cross_seed_time_averaging=True,cases=cases)
    (HERE/'plot_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    checks=dict(created_utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=True,
                same_frozen_input=True,no_simulation_started=True,all_npus_active=True,
                all_disk_event_demand_below_40=True,sources=sources,
                outputs={p.name:sha(p) for p in outputs},visual_review='pending')
    (HERE/'render_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    rows=['| 策略 | 整机平均需求 GiB/s | 整机峰值 GiB/s |','|---|---:|---:|']
    for c in cases:rows.append(f"| {NAMES[c['policy']]} | {c['total_mean_GiB_s']:.4f} | {c['total_peak_GiB_s']:.4f} |")
    (HERE/'README.md').write_text('# 持续欠载输入：存储带宽需求\n\n对应刚才三策略CDF的原始持续欠载输入；带宽时序取 seed 7，32 NPU、3 SSU×40 GiB/s、Random、Ring hash、warm [2,4) 秒。CDF使用三个seed等权，时序图不混合种子。\n\n'
        '- [整机总需求图](under_total_storage_demand.png)\n- [逐盘需求图](under_per_ssu_storage_demand.png)\n\n'+'\n'.join(rows)+
        '\n\n需求按每层 V/C 计算，V 是每个请求实际落在相应盘的数据量，C 是该请求每层纯计算时间。I/O等待时仍保留需求；跨请求首层预取不重复叠加为第二份当前请求。图直接画真实请求切换事件之间的需求，没有做10ms或跨种子平均。\n\n'
        '三策略输入逐字节相同。整机峰值按同一时刻三盘需求之和取最大值，不是三张盘各自峰值相加。三盘分别均低于40 GiB/s，单盘最高29.786927 GiB/s。图展示参考需求，不是SSD实际供给。\n\n'
        '旧图与原始仿真保持不变；[绘图数据](plot_data.json) 和 [核验记录](render_checks.json) 包含原始结果路径、SHA及完整事件区间。\n')
    print('\n'.join(rows))


if __name__=='__main__':main()
