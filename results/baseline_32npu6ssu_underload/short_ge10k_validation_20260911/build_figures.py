#!/usr/bin/env python3
"""Draw separate figures from existing raw-data experiments; no simulation."""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
FOLLOWUP = BASE / 'raw_role_followup_20260909'
SOURCE = FOLLOWUP / 'mixed_rebinding'
OUT = HERE / 'figures/separate'
COLORS = {'short':'#26749b','long':'#72b5a5','stall':'#efa640'}
FONT = Path('/home/chguo/.fonts/msyh.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
    'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
    'svg.hashsalt':'short-ge10k-20260911'})


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clip(a,b,left,right):
    return max(0.0,min(b,right)-max(a,left))


def load_case(mode, strategy, left=2000., right=4000., repeated=False):
    folder=FOLLOWUP/'window_sensitivity/repeated_queues' if repeated else SOURCE
    label=f'raw176_{mode}_repeat3_seed7' if repeated else f'raw176_extendedhot_{mode}_seed7'
    manifest_path=folder/'inputs'/f'{label}.json.gz'
    paths=list((folder/'runs'/label/strategy).glob('*.json.gz'))
    assert len(paths)==1,paths
    result_path=paths[0]
    manifest,result=read(manifest_path),read(result_path)
    assert manifest['input_fingerprint']==result['input_fingerprint']
    assert result['strategy']==strategy
    assert all(result['summary']['invariants'].values())
    byid={r['request_id']:r for r in manifest['requests']}
    assert all(r['load']['seq_len_k']*1024>=10000 for r in byid.values())
    volumes={}
    for idx,placement in enumerate(manifest['placements']):
        vv=[[math.fsum(v for s,v in layer if s==ssu) for ssu in range(6)] for layer in placement]
        assert all(np.allclose(v,vv[0],rtol=0,atol=1e-12) for v in vv)
        volumes[idx]=np.array(vv[0])
    spans=[{key:[] for key in COLORS} for _ in range(32)]
    compute=np.zeros(32);active=np.zeros(32);stall=np.zeros(32)
    events=defaultdict(lambda:np.zeros(6))
    role_compute={key:np.zeros(32) for key in ['short','long']}
    slo_pass=slo_count=0
    for b in result['summary']['microbatch_metrics']:
        assert b['batch_size']==1
        r=byid[b['member_request_ids'][0]];n=b['npu_id'];role=r['load']['role']
        assert n==r['npu_id']
        a,z=b['admission_time_ms'],b['completion_time_ms']
        active[n]+=clip(a,z,left,right)
        rate=volumes[r['placement_index']]/(r['load']['per_layer_us']/1e6)
        events[a]+=rate;events[z]-=rate
        previous=a
        for layer in b['layer_metrics']:
            cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
            assert math.isclose(cs-previous,layer['io_barrier_wait_ms'],abs_tol=1e-7)
            for key,start,end in [('stall',previous,cs),(role,cs,ce)]:
                dt=clip(start,end,left,right)
                if dt:
                    spans[n][key].append((max(start,left),dt))
                    if key=='stall':stall[n]+=dt
                    else:compute[n]+=dt;role_compute[role][n]+=dt
            previous=ce
        assert math.isclose(previous,z,abs_tol=1e-7)
        if left<=a<right:
            slo_count+=1
            slo_pass+=z-a<=1.5*8*r['load']['per_layer_us']/1000+1e-8
    assert np.allclose(active,right-left,atol=1e-6,rtol=0)
    assert np.allclose(compute+stall,active,atol=1e-6,rtol=0)
    assert all(np.all(a>0) for a in role_compute.values())
    times=[];rates=[];current=np.zeros(6);fullpeak=np.zeros(6)
    et=sorted(events)
    for i,t in enumerate(et[:-1]):
        current+=events[t]
        fullpeak=np.maximum(fullpeak,current)
        if et[i+1]>left and t<right:
            times.append(max(left,t));rates.append(current.copy())
    times.append(right);rates.append(rates[-1].copy())
    rates=np.array(rates)
    assert max(fullpeak)<40
    util=float(compute.sum()/32/(right-left))
    if left==2000 and right==4000:
        ww=next(w for w in result['windows'] if w['start_ms']==left and w['end_ms']==right)
        assert math.isclose(util,ww['mean_npu_utilization'],abs_tol=1e-10)
    audit={'mode':mode,'strategy':strategy,'label':label,'window_ms':[left,right],
        'repeated_queue':repeated,'utilization':util,'slo_passed':slo_pass,'slo_count':slo_count,
        'slo_rate':slo_pass/slo_count,'all_active':True,'all_npus_both_roles_compute':True,
        'role_compute_ms_by_npu':{k:v.tolist() for k,v in role_compute.items()},
        'full_peak_per_ssu_gib_s':fullpeak.tolist(),'warm_peak_per_ssu_gib_s':rates.max(axis=0).tolist(),
        'manifest':str(manifest_path.relative_to(BASE)),'manifest_sha256':sha(manifest_path),
        'result':str(result_path.relative_to(BASE)),'result_sha256':sha(result_path),
        'input_fingerprint':manifest['input_fingerprint']}
    return {'spans':spans,'times':np.array(times),'rates':rates,'audit':audit}


def draw(case,kind,stem):
    a=case['audit'];left,right=a['window_ms'];extended=a['repeated_queue']
    policy='Baseline' if a['strategy']=='baseline' else 'Once per layer'
    order='随机顺序' if a['mode']=='random' else '定序混合'
    subject='32 卡计算与等待' if kind=='timeline' else '六盘名义需求'
    fig,ax=plt.subplots(figsize=(15,10.5 if kind=='timeline' else 7.4))
    fig.subplots_adjust(left=.075,right=.985,top=.765 if kind=='timeline' else .735,
                        bottom=.185 if kind=='timeline' else .260)
    fig.suptitle(f'{order} · {policy}：{subject}',fontsize=23,y=.976)
    subtitle_y, profiles_y, note_y = (.921,.875,.836) if kind=='timeline' else (.890,.839,.797)
    fig.text(.5,subtitle_y,f'32 NPU / 6 SSU · 每盘 40 GiB/s · [{left/1000:g}, {right/1000:g}) 秒 · U = {a["utilization"]*100:.4f}%',
             ha='center',fontsize=16)
    fig.text(.5,profiles_y,'短请求总长 32K / 48K / 64K；长请求 176K；全部直接取自 data，每请求 8 层',ha='center',fontsize=14)
    note='每卡完整队列重复 3 遍：新长输入，非旧日志拼接；仅 seed 7' if extended else '同种子 Random / Ordered 保留每卡同一批请求，只改变卡内顺序；每卡暖窗都有长短计算'
    fig.text(.5,note_y,note,ha='center',fontsize=12.7,color='#444444')
    if kind=='timeline':
        fig.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in
                   [('short','短请求计算'),('long','长请求计算'),('stall','接纳后 I/O 等待')]],
                   loc='upper center',bbox_to_anchor=(.52,.811),ncol=3,frameon=False,fontsize=12.8)
        for n in range(32):
            for key in ['short','long','stall']:
                ax.broken_barh(case['spans'][n][key],(n-.40,.80),facecolors=COLORS[key],
                               linewidth=0,edgecolors='none',rasterized=False)
        ax.set(ylim=(31.8,-.8),yticks=list(range(32)),ylabel='NPU 编号')
        ax.tick_params(axis='y',labelsize=10)
        for boundary in [16.5,19.5]:ax.axhline(boundary,color='#9AA4AD',lw=.6)
        ax.grid(axis='x',alpha=.15)
        fig.text(.075,.119,'按真实层事件裁剪；无时间移动或拼接。32 卡全窗有任务，橙色不含接纳前排队。',fontsize=12)
        if extended:
            fig.text(.075,.081,'前段的等待后来显著减少：这组混合输入不能证明长期维持 10 个百分点损失。',fontsize=12,color='#9D4221')
        else:
            fig.text(.075,.081,f'全程最热盘名义需求 {max(a["full_peak_per_ssu_gib_s"]):.6f} < 40 GiB/s；窗口 SLO×1.5 = {100*a["slo_rate"]:.2f}%（接纳后代理）。',fontsize=12)
    else:
        for s in range(6):
            ax.step(case['times'],case['rates'][:,s],where='post',label=f'SSU {s}',lw=1.1,alpha=.80)
        ax.axhline(40,color='#B42B28',lw=1.4,ls='--',label='单盘容量')
        ax.set(ylim=(0,43),yticks=[0,10,20,30,40],ylabel='逐盘名义需求（GiB/s）')
        ax.grid(axis='y',alpha=.2)
        fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.148),ncol=7,frameon=False,fontsize=11)
        fig.text(.075,.107,f'全程逐事件峰值 {max(a["full_peak_per_ssu_gib_s"]):.6f} GiB/s；本窗口峰值 {max(a["warm_peak_per_ssu_gib_s"]):.6f} GiB/s。',fontsize=12)
        fig.text(.075,.069,'需求 = 当前已接纳请求逐盘 D/C；沿用原约定，下一请求首层预取不另加。此图不是实际吞吐。',fontsize=11.5)
    ax.set_xlim(left,right)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1000:g}'))
    ax.set_xlabel('仿真时间（秒）',fontsize=14)
    fig.text(.075,.025,f'seed 7 · 输入指纹 {a["input_fingerprint"][:16]} · 原始仿真日志离线重绘；非真实硬件测量',fontsize=10.5,color='#59636C')
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=190)
    plt.close(fig)
    return {'stem':stem,'kind':kind,**a}


def main():
    rows=[]
    for mode in ['random','ordered']:
        for strategy in ['baseline','once']:
            case=load_case(mode,strategy)
            for kind in ['timeline','demand']:
                stem=f'{mode}_{strategy}'+('_demand' if kind=='demand' else '')
                rows.append(draw(case,kind,stem))
    rows.append(draw(load_case('ordered','baseline',2000,12000,True),'timeline','ordered_baseline_long_window'))
    (HERE/'figure_audit.json').write_text(json.dumps({'runs_no_simulation':True,'figures':rows,
        'script_sha256':sha(__file__)},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(rows),'out':str(OUT),'main_U':[
        (r['stem'],r['utilization']) for r in rows if r['kind']=='timeline']},ensure_ascii=False))


if __name__=='__main__':main()
