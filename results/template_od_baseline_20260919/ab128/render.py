#!/usr/bin/env python3
"""Add OD to the original A/B CDF and produce matching OD-only timeline/32-lane plots."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = ROOT / 'results/baseline_ab128_32_ratio12_20260912'
ARCHIVE = ROOT / 'template/qos_experiments_20260919/repository_source/results/baseline_ab128_32_ratio12_20260912'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
font = Path('/home/chguo/.fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
STYLES = {'asu_baseline': ('ASU baseline', '#4b5563', '--'),
          'od_baseline': ('OD baseline', '#d45e00', '-.'),
          'once': ('流量分配（Once）', '#2474b7', '-')}
BLUE, PURPLE, GRAY = '#0780ff', '#b12bac', '#e0e4eb'


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def overlap(a, b):
    return max(0., min(4000., b)-max(2000., a))


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -2 and box.y0 >= -2 and box.x1 <= width+2 and box.y1 <= height+2, (artist.get_text(),box.bounds)
    path = OUT / name
    fig.savefig(path, dpi=fig.dpi, facecolor='white')
    plt.close(fig)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'pixels': [width,height], 'text_inside_canvas': True}


def get_case(order, strategy):
    if strategy == 'asu_baseline':
        rel = Path('validation20s/runs') / f'ssu3_{order}_k1_sync_seed7/baseline'
    elif strategy == 'once':
        rel = Path('once_per_layer_ssu3_seed7/runs') / order / 'once'
    else:
        rel = None
    directory = ORIGINAL/rel if rel else HERE/'runs'/f'{order}_{strategy}'
    man, raw, command = [read(directory / n) for n in ('manifest.json.gz','result.json.gz','command.json')]
    assert command['status'] == 'complete'
    assert sha(directory/'result.json.gz') == command['output_sha256']
    reference = ARCHIVE/rel if rel else directory
    assert sha(reference/'manifest.json.gz') == sha(directory/'manifest.json.gz')
    if rel:
        assert read(reference/'command.json')['output_sha256'] == sha(directory/'result.json.gz')
    assert raw['input_fingerprint'] == man['input_fingerprint']
    assert all(raw['summary']['invariants'].values())
    return directory, man, raw


def samples_and_u(order, strategy):
    directory, man, raw = get_case(order,strategy)
    requests = {q['request_id']: q for q in man['requests']}
    samples = []
    for q in raw['summary']['request_metrics']:
        if not 2000 <= q['admission_time_ms'] < 4000:
            continue
        ideal = requests[q['request_id']]['load']['per_layer_us'] * 8 / 1000
        latency = q['completion_time_ms'] - q['admission_time_ms']
        assert math.isclose(q['own_compute_ms'],ideal,abs_tol=1e-8)
        ratio = latency/ideal
        for threshold in (1., 1.5):
            if abs(latency-threshold*ideal)<=1e-9:
                ratio = threshold
        samples.append(dict(order=order,strategy=strategy,request_id=q['request_id'],npu_id=q['npu_id'],
                            role=requests[q['request_id']]['load']['role'], admission_ms=q['admission_time_ms'],
                            completion_ms=q['completion_time_ms'],ttft_ms=latency,ideal_ms=ideal,ratio=ratio,
                            slo_1p5_pass=latency<=1.5*ideal+1e-9))
    compute = sum(overlap(l['compute_start_ms'], l['compute_end_ms']) for b in raw['summary']['microbatch_metrics'] for l in b['layer_metrics'])
    u = compute/64000*100
    assert math.isclose(u, 100*raw['common_window']['mean_npu_utilization'],abs_tol=1e-8)
    return samples, dict(scenario='ab128_'+order,strategy=strategy,seed=7,U_percent=u,
                        slo_1p5_percent=100*sum(q['slo_1p5_pass'] for q in samples)/len(samples),
                        sample_count=len(samples),window_start_ms=2000,window_end_ms=4000,
                        cohort='admission_in_window_follow_to_completion',input_fingerprint=man['input_fingerprint'])


def cdf(order):
    all_samples, summary = [], []
    fig, ax = plt.subplots(figsize=(11.2,7),dpi=180)
    fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.80)
    for strategy,(label,color,style) in STYLES.items():
        samples, row = samples_and_u(order,strategy)
        summary.append(row); all_samples.extend(samples)
        values,counts = np.unique([r['ratio'] for r in samples],return_counts=True)
        ax.step(np.r_[.88,values],np.r_[0,np.cumsum(counts)/len(samples)],where='post',
                label=f'{label} · U {row["U_percent"]:.2f}% · SLO×1.5 {row["slo_1p5_percent"]:.2f}%',
                color=color,linestyle=style,lw=2.3)
        for t in (1,1.5):
            ax.scatter(t,np.mean([r['ratio']<=t for r in samples]),color=color,s=34,zorder=5)
    maxratio = max(r['ratio'] for r in all_samples)
    ax.set(xlim=(.88,maxratio*1.05),ylim=(0,1.03),xlabel='TTFT / SLO（SLO = 8 层纯计算时间）',ylabel='累计请求比例（CDF）')
    ax.set_xticks(np.arange(1.,maxratio*1.05,.5))
    ax.set_yticks(np.arange(0.,1.01,.2))
    for t in (1,1.5):
        ax.axvline(t,color='#8b929a',ls=':',lw=1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',alpha=.22)
    ax.legend(loc='lower right',fontsize=10,framealpha=.95)
    fig.suptitle(f'A/B {order.title()}：原输入上的三策略 TTFT CDF',y=.958,fontsize=20)
    fig.text(.1,.86,'32 NPU / 3 SSU × 40 GiB/s · seed 7 · 接纳窗口 [2,4) 秒',color='#526476')
    fig.text(.1,.065,'A: 128K / miss 256；B: 32K / miss 4096；每卡 40A + 80B。',fontsize=10)
    fig.text(.1,.030,'TTFT 指接纳至 prefill 完成；包括窗后完成请求，不含接纳前排队。',fontsize=10)
    image=save(fig,f'AB_{order}_TTFT_CDF_three_strategies.png')
    csvout(HERE/f'{order}_request_samples.csv',all_samples)
    return summary,image


def od_analysis(order):
    directory,man,raw=get_case(order,'od_baseline')
    audit=read(directory/'io_audit.json.gz')
    io={(r[0],r[1]):r[2:] for r in audit['rows']}
    req={r['request_id']:r for r in man['requests']}
    lanes=[]; cycle_rows=[]
    for npu in range(32):
        batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
        flat=[]; demands=[]; spans={'A':[],'B':[],'stall':[]}; active=0
        for b in batches:
            rid=b['member_request_ids'][0];q=req[rid]['load']
            active+=overlap(b['admission_time_ms'],b['completion_time_ms'])
            if overlap(b['admission_time_ms'],b['completion_time_ms']):
                demands.append((b['admission_time_ms'],b['completion_time_ms'],q['per_layer_kv_gb']*1e6/q['per_layer_us']))
            previous=b['admission_time_ms']
            for l in b['layer_metrics']:
                flat.append(dict(rid=rid,**l))
                for state,a,z in (('stall',previous,l['compute_start_ms']),(q['role'],l['compute_start_ms'],l['compute_end_ms'])):
                    if overlap(a,z):spans[state].append((max(a,2000)/1000,overlap(a,z)/1000))
                previous=l['compute_end_ms']
        assert math.isclose(active,2000,abs_tol=1e-7)
        assert math.isclose(sum(d for ss in spans.values() for _,d in ss),2,abs_tol=1e-8)
        cycles=[]
        for a,z in zip(flat,flat[1:]):
            start,end=a['compute_start_ms'],z['compute_start_ms']
            if not overlap(start,end):continue
            c=a['compute_end_ms']-start;d=end-start
            assert math.isclose(z['io_start_time_ms'],start,abs_tol=1e-7)
            assert math.isclose(end,max(a['compute_end_ms'],z['io_ready_time_ms']),abs_tol=1e-7)
            complete=2000<=start<end<=4000 and a['rid']==z['rid']
            q=req[a['rid']]['load'];B=q['per_layer_kv_gb']*1000/c
            b=None
            if complete:
                count,v,first,last=io[(z['rid'],z['layer'])]
                assert math.isclose(v,q['per_layer_kv_gb'],abs_tol=1e-12)
                assert first>=start-1e-7 and last<=end+1e-7
                assert math.isclose(last,z['io_ready_time_ms'],abs_tol=1e-7)
                b=v*1000/d
                assert math.isclose(b/B,c/d,abs_tol=1e-9)
            row=dict(npu=npu,request_id=a['rid'],compute_layer=a['layer'],start_ms=start,end_ms=end,
                     clipped_start_ms=max(start,2000),clipped_end_ms=min(end,4000),
                     B_GiB_s=B,b_GiB_s=b,C_ms=c,D_ms=d,complete_internal_cycle=complete)
            cycles.append(row);cycle_rows.append(row)
        u=100*sum(d for state,ss in spans.items() if state!='stall' for _,d in ss)/2
        lanes.append(dict(npu=npu,cycles=cycles,demands=demands,spans=spans,U_percent=u,
                          mean_demand_GiB_s=sum(overlap(a,z)*B for a,z,B in demands)/2000,
                          mean_supply_GiB_s=audit['warm_received_GiB_by_npu'][npu]/2))
    assert math.isclose(np.mean([l['U_percent'] for l in lanes]),100*raw['common_window']['mean_npu_utilization'],abs_tol=1e-8)
    csvout(HERE/f'{order}_od_layer_cycles.csv',cycle_rows)
    csvout(HERE/f'{order}_od_per_npu.csv',[{k:v for k,v in l.items() if k not in ('cycles','demands','spans')} for l in lanes])
    return lanes


def timeline(order,lanes):
    fig,ax=plt.subplots(figsize=(16,10.8),dpi=155)
    fig.subplots_adjust(left=.07,right=.95,bottom=.12,top=.81)
    colors={'A':'#277da8','B':'#52a68a','stall':'#f0a340'}
    for n,lane in enumerate(lanes):
        for state,color in colors.items():ax.broken_barh(lane['spans'][state],(n-.4,.8),facecolors=color,linewidth=0)
        ax.text(4.016,n,f'{lane["U_percent"]:.1f}%',va='center',fontsize=8)
    ax.set(xlim=(2,4),ylim=(31.8,-.8),yticks=range(32),xticks=np.arange(2.,4.01,.25),xlabel='时间（秒）',ylabel='NPU 编号')
    ax.grid(axis='x',alpha=.15)
    fig.suptitle(f'OD baseline {order.title()}：32 张 NPU 的计算与 I/O 等待',y=.97,fontsize=21)
    fig.text(.5,.923,f'32 NPU / 3 SSU × 40 GiB/s · [2,4) 秒 · NPU 平均利用率 {np.mean([l["U_percent"] for l in lanes]):.2f}%',ha='center',fontsize=15)
    fig.text(.5,.884,'A：128K 总长 / miss 256；B：32K 总长 / miss 4096；每卡 40A + 80B',ha='center')
    fig.legend(handles=[Patch(color=colors[s],label=label) for s,label in [('A','A 请求计算'),('B','B 请求计算'),('stall','I/O 等待')]],loc='upper center',bbox_to_anchor=(.5,.867),ncol=3,frameon=False)
    fig.text(.07,.065,'32/32 卡全窗有任务；右侧为每卡利用率。OD 每卡独占一个 Path，等额 CIR，允许借用空闲带宽。',fontsize=10)
    fig.text(.07,.03,'原冻结输入、条带放置与提交 seed 均保持不变；I/O 等待不含接纳前排队。',fontsize=10)
    return save(fig,f'od_baseline_{order}_timeline.png')


def bandwidth(order,lanes):
    fig,axes=plt.subplots(32,1,figsize=(18,32),dpi=150,sharex=True,sharey=True)
    fig.subplots_adjust(left=.108,right=.848,top=.933,bottom=.048,hspace=.34)
    fig.text(.045,.985,f'OD baseline {order.title()}：32 张 NPU 的每层平均带宽与需求',fontsize=24,color='#20344c')
    fig.text(.045,.970,f'32 NPU / 3 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 整机 U={np.mean([l["U_percent"] for l in lanes]):.2f}%',fontsize=15,color='#5a6e86')
    fig.legend(handles=[Line2D([],[],color=PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='平均 b_i：每个完整内部层周期一个值'),Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')],loc='upper left',bbox_to_anchor=(.043,.964),ncol=3,frameon=False,fontsize=13)
    fig.text(.865,.940,'整窗平均（GiB/s）',fontsize=12)
    for ax,lane in zip(axes,lanes):
        blue=[];purple=[];points=[];previous=None
        for c in lane['cycles']:
            a,z=c['clipped_start_ms']/1000,c['clipped_end_ms']/1000
            if not c['complete_internal_cycle']:
                ax.axvspan(a,z,color=GRAY,zorder=0);previous=None;continue
            b=c['b_GiB_s'];blue.append([(a,b),(z,b)]);points.append(((a+z)/2,b))
            if previous and math.isclose(previous[0],a,abs_tol=1e-9):blue.append([(a,previous[1]),(a,b)])
            previous=(z,b)
        previous=None
        for a,z,B in lane['demands']:
            a,z=max(a,2000)/1000,min(z,4000)/1000
            purple.append([(a,B),(z,B)])
            if previous and math.isclose(previous[0],a,abs_tol=1e-9):purple.append([(a,previous[1]),(a,B)])
            previous=(z,B)
        ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.3,zorder=3))
        ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.8,linestyles='--',zorder=4))
        if points:
            px,py=zip(*points);ax.scatter(px,py,s=13,facecolors='white',edgecolors=BLUE,linewidths=1.1,zorder=5)
        ax.set(xlim=(2,4),ylim=(0,32.5),xticks=np.arange(2,4.01,.25),yticks=[0,10,28.476])
        ax.set_yticklabels(['0','10','28.48'])
        ax.set_ylabel(f'NPU {lane["npu"]:02d}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=16)
        ax.tick_params(axis='y',labelsize=9,length=3)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=lane['npu'] in (7,15,23,31))
        ax.grid(alpha=.13,lw=.6)
        ax.text(1.022,.70,f'需求 {lane["mean_demand_GiB_s"]:.3f}',transform=ax.transAxes,fontsize=11,color=PURPLE)
        ax.text(1.022,.28,f'供给 {lane["mean_supply_GiB_s"]:.3f}',transform=ax.transAxes,fontsize=11,color=BLUE)
    axes[-1].set_xlabel('时间（秒）',fontsize=13,labelpad=10)
    fig.text(.045,.020,'周期 D：当前层开始计算 → 下一层开始计算（含等待）；b_i = 实际接收的下一层数据量 / D。',fontsize=12)
    fig.text(.045,.009,'右侧统计完整 [2,4) 秒，供给含灰区；两个整窗平均值相除不等于 NPU 利用率。',fontsize=11,color='#5a6e86')
    return save(fig,f'od_{order}_all_32npu_layer_average.png')


def main():
    summaries=[];images=[];parity=[]
    for order in ('random','ordered'):
        p=read(HERE/'runs'/f'{order}_asu_baseline/command.json')
        assert p['status']=='complete' and p['parity']['max_layer_time_error_ms']==0
        parity.append(p['parity'])
        rows,image=cdf(order);summaries.extend(rows);images.append(image)
        lanes=od_analysis(order)
        images.extend([timeline(order,lanes),bandwidth(order,lanes)])
    csvout(HERE/'summary.csv',summaries)
    write(HERE/'summary.json',summaries)
    write(HERE/'checks.json',dict(all_checks_passed=True,original_asu_parity=parity,images=images,
                                inputs_unchanged=True,legacy_placement_preserved=True,
                                all_internal_cycle_b_B_equal_C_D=True,all_original_result_hashes_match_archive=True,
                                cohort='warm [2000,4000) admission, followed to full completion',
                                ttft_proxy='admission to prefill completion',formats_generated=['png'],
                                renderer_sha256=sha(Path(__file__))))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
