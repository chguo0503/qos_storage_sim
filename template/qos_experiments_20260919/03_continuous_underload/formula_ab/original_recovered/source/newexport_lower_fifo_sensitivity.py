#!/usr/bin/env python3
"""Small native FIFO/Once figures; 20K COMPUTE EXTRAPOLATION SENSITIVITY ONLY."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.patches import Patch
from PIL import Image
import export_boundary_underload as eb
import audit_boundary_20k_sensitivity as audit20
import export_fixed128_32_results as shared
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914/figures';IMG=OUT/'images';DATA=OUT/'data'
FONT=ROOT/'assets/KaiXinSong-Charts.ttf'
if not FONT.exists():FONT=ROOT.parents[1]/'fifo_figure_reference/fonts/KaiXinSong-Charts.ttf'
fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':FontProperties(fname=str(FONT)).get_name(),'font.size':10,'axes.unicode_minus':False,'figure.facecolor':'white','path.simplify':False})
eb.boundary=audit20  # Use the explicit extrapolation-aware auditor, in memory only.
COL={'fifo':'#355d8a','once':'#1a896f'};INK='#263445';MUTED='#596579'
for p in (IMG,DATA):p.mkdir(parents=True,exist_ok=True)
figaudit=[];cases=[]

def save(fig,name):
    path=IMG/name;fig.savefig(path,dpi=160,facecolor='white');plt.close(fig)
    im=Image.open(path);im.verify();figaudit.append(dict(file=name,pixels=list(Image.open(path).size),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),every_image_labeled_20k_compute_extrapolation=True))

def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)

def title(fig,heading,sub):
    fig.text(.055,.96 if fig.get_figheight()<6 else .978,heading+'｜20K 计算外推',ha='left',va='top',fontsize=17,color=INK)
    fig.text(.055,.86 if fig.get_figheight()<6 else .936,sub+'；敏感性实验，20K 计算时间未实测校准。',fontsize=10,color=MUTED)

for policy in ('fifo','once'):
    folder=ROOT/f'results/lower_fifo_followup_20260914/native20/sensitivity20k_076_seed7_{policy}'
    case=dict(folder=folder,policy=policy,group='sensitivity20k_076_seed7',num_ssu=1)
    for key,name in [('manifest','manifest.json.gz'),('metadata','metadata.json'),('metrics','metrics.json'),('receipt','receipts.json'),('raw','result.json.gz')]:case[key]=shared.read(folder/name)
    case['fingerprint']=case['metadata']['input_fingerprint'];case['rows']=shared.request_rows(case)
    assert not case['metadata']['compute_calibrated'] and case['metadata']['contains_extrapolated_compute']
    case['event']=eb.event_data(case)
    receipt=case['receipt'];assert all(receipt['observer_checks'].values())
    edges=np.asarray(receipt['bin_edges_ms']);widths=np.diff(edges);assert np.allclose(widths,2)
    physical=np.asarray(receipt['ssu_bin_read_gib'])[0]*1000/widths
    assert physical.max()<=40+1e-7
    assert math.isclose(float(np.dot(physical,widths/1000)),sum(receipt['ssu_bin_read_gib'][0]),abs_tol=1e-10)
    case.update(physical=physical,physical_edges=edges)
    shared.write_csv(DATA/f'{policy}_exact_demand_segments.csv',case['event']['rows'])
    shared.write_csv(DATA/f'{policy}_physical_2ms.csv',[dict(start_ms=float(a),end_ms=float(b),physical_gib_s=float(v)) for a,b,v in zip(edges,edges[1:],physical)])
    shared.write_csv(DATA/f'{policy}_all_input_ttft.csv',case['rows'])
    cases.append(case)
assert cases[0]['fingerprint']==cases[1]['fingerprint']
assert {r['request_id'] for r in cases[0]['rows']}=={r['request_id'] for r in cases[1]['rows']}

fig,axs=plt.subplots(2,2,figsize=(14,8),sharex='row');fig.subplots_adjust(left=.07,right=.97,bottom=.105,top=.82,wspace=.14,hspace=.40)
title(fig,'FIFO / Once：需求与真实 SSD 服务','相同输入，8 NPU / 1 SSU，SSU 40 GiB/s，NPU 接收上限 50 GiB/s')
curves=[('current','#82579e','当前 V/C'),('continuing','#17866a','非交界预取'),('boundary','#bb6763','交界预取'),('total_prefetch','#da8a19','总预取')]
ymax=max(np.max(c['event']['total_prefetch']) for c in cases)*1.15
for j,c in enumerate(cases):
    ax=axs[0,j];e=c['event'];x=e['edges']/1000
    for k,color,label in curves:ax.stairs(e[k][0],x,color=color,lw=.85,label=label)
    ax.axhline(40,color='#323c47',ls='--',lw=1,label='容量 40')
    ax.set(xlim=(0,max(cc['event']['edges'][-1] for cc in cases)/1000),ylim=(0,ymax),ylabel='GiB/s',xlabel='仿真时间（秒）')
    ax.set_title(c['policy'].upper()+' · 20K 计算外推 · 全程精确需求',loc='left',fontsize=11)
    ax=axs[1,j];ax.stairs(c['physical'],c['physical_edges']/1000,color=COL[c['policy']],lw=.8)
    ax.axhline(40,color='#323c47',ls='--',lw=1)
    ax.set(xlim=(2,4),ylim=(0,44),ylabel='GiB/s',xlabel='仿真时间（秒）')
    ax.set_title(f'{c["policy"].upper()} · 20K 计算外推 · 实际服务 / 2 ms，峰值 {c["physical"].max():.3f}',loc='left',fontsize=10)
for ax in axs.ravel():style(ax)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.065,.899),ncol=5,frameon=False,fontsize=10)
fig.text(.055,.033,'总预取 = 非交界 + 交界；交界需求单独豁免。实际服务未经限幅，按共同 2 ms 窗口计算，始终 ≤40 GiB/s。',fontsize=10,color=MUTED)
save(fig,'sensitivity20k_bandwidth_pair.png')

fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True);fig.subplots_adjust(left=.08,right=.98,bottom=.115,top=.835,hspace=.40)
title(fig,'FIFO / Once：8 张 NPU 的完整 warm 时间线','固定窗口 [2,4) 秒，绿色为短请求计算，蓝色为长请求计算，橙色为 IO 等待')
for ax,c in zip(axs,cases):
    for b in c['raw']['summary']['microbatch_metrics']:
        n=b['npu_id'];rid=b['member_request_ids'][0];role=next(r['role'] for r in c['rows'] if r['request_id']==rid);prev=b['admission_time_ms']
        for l in sorted(b['layer_metrics'],key=lambda z:z['layer']):
            for a,z,color in [(prev,l['compute_start_ms'],'#e9b157'),(l['compute_start_ms'],l['compute_end_ms'],'#209f7a' if role=='S' else '#447eac')]:
                a,z=max(a,2000),min(z,4000)
                if z>a:ax.broken_barh([(a/1000,(z-a)/1000)],(n-.34,.68),facecolors=color,linewidth=0)
            prev=l['compute_end_ms']
    ax.set(yticks=list(range(8)),yticklabels=[f'NPU {n}' for n in range(8)],ylim=(7.65,-.65),xlim=(2,4),xlabel='仿真时间（秒）')
    ax.set_title(f'{c["policy"].upper()} · 20K 计算外推 · 全机 U={c["metrics"]["U_percent"]:.3f}%',loc='left',fontsize=11)
    style(ax);ax.grid(axis='y',visible=False);ax.tick_params(axis='x',labelbottom=True)
fig.legend(handles=[Patch(color='#209f7a',label='短请求计算（20K，外推）'),Patch(color='#447eac',label='长请求计算（200K）'),Patch(color='#e9b157',label='IO 等待')],loc='upper left',bbox_to_anchor=(.065,.903),ncol=3,frameon=False)
fig.text(.055,.036,'显示全部 8 张卡及完整 2 秒窗口；交界 IO 等待保留在图中与利用率统计中。时间线显示计算与暴露等待，不把预取重叠计作等待。',fontsize=9.5,color=MUTED)
save(fig,'sensitivity20k_timeline_pair.png')

fig,axs=plt.subplots(1,3,figsize=(14,5.5),sharey=True);fig.subplots_adjust(left=.07,right=.97,bottom=.23,top=.79,wspace=.17)
title(fig,'FIFO / Once：TTFT SLO × 1.5 分布','相同全部输入请求，共 392 个；两策略严格按相同请求 ID 配对，seed 7')
cdfrows=[]
for ax,role,label in zip(axs,('all','S','L'),('全部请求','短请求：20K（计算外推）','长请求：200K')):
    vals=[r['ttft_over_ideal'] for c in cases for r in c['rows'] if role=='all' or r['role']==role];xmax=max(1.65,max(vals)*1.025)
    for c in cases:
        pool=[r for r in c['rows'] if role=='all' or r['role']==role];u,cnt=np.unique([r['ttft_over_ideal'] for r in pool],return_counts=True);y=np.cumsum(cnt)*100/len(pool);passed=sum(r['slo_1p5_passed'] for r in pool)
        ax.step(np.r_[.98,u,xmax],np.r_[0,y,100],where='post',color=COL[c['policy']],lw=1.6,label=f'{c["policy"].upper()}：{passed}/{len(pool)}\n达标 {passed/len(pool)*100:.2f}%')
        cdfrows.extend(dict(policy=c['policy'],role=role,ratio=float(x),cdf_percent=float(z),count=len(pool)) for x,z in zip(u,y))
    ax.axvline(1.5,color='#48525c',ls='--',lw=1);ax.set(xlim=(.98,xmax),ylim=(0,102),title=label,xlabel='TTFT / 8 层纯计算时间');ax.legend(loc='lower right',fontsize=9);style(ax)
axs[0].set_ylabel('累计请求比例（%）')
fig.text(.055,.13,'TTFT = 完成时间减去 NPU 接纳时间，不含接纳前排队。虚线阈值：TTFT ≤1.5×纯计算时间。全量分布不混用 warm 接纳集合。',fontsize=10,color=MUTED)
fig.text(.055,.075,'FIFO 与 Once 均含同一组 20K 计算外推画像；这些结果仅是敏感性实验，不能作为 20K 实测性能结论。',fontsize=10,color=MUTED)
save(fig,'sensitivity20k_ttft_cdf_pair.png');shared.write_csv(DATA/'ttft_cdf_points.csv',cdfrows)
summary=dict(scope='20K compute extrapolation sensitivity only; native simulator; not measured20K data',figures=figaudit,input_fingerprint=cases[0]['fingerprint'],matched_input_request_ids=True,request_count=len(cases[0]['rows']),new_auditor='audit_boundary_20k_sensitivity.build',cases=[])
for c in cases:
    e=c['event'];summary['cases'].append(dict(policy=c['policy'],U_percent=c['metrics']['U_percent'],current_peak_gib_s=float(e['current'].max()),continuing_peak_gib_s=float(e['continuing'].max()),boundary_peak_gib_s=float(e['boundary'].max()),total_prefetch_peak_gib_s=float(e['total_prefetch'].max()),physical_peak_gib_s=float(c['physical'].max()),physical_capacity_passed=bool(c['physical'].max()<=40+1e-7),observer_checks=c['receipt']['observer_checks'],source_sha256={name:hashlib.sha256((c['folder']/name).read_bytes()).hexdigest() for name in ('manifest.json.gz','result.json.gz','receipts.json','metadata.json')}))
shared.write_json(DATA/'sensitivity20k_figure_audit.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
