#!/usr/bin/env python3
"""Standalone, separate scientific figures drawn from saved layer events."""
import argparse
from pathlib import Path
import gzip,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
START,END=2000.,4000.
plt.rcParams.update({'font.family':FontProperties(fname='/home/chguo/.fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})

def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);a=ap.parse_args()
    label=a.label;folder=HERE/'mixed_rebinding';out=folder/'figures';out.mkdir(exist_ok=True)
    manifest=read(folder/'inputs'/f'{label}.json.gz');byid={r['request_id']:r for r in manifest['requests']}
    raw=read(next((folder/'runs'/label/'baseline').glob('*.json.gz')))
    batches=raw['summary']['microbatch_metrics'];bars={role:[[] for _ in range(32)] for role in ('long','short','stall')};events={START:[0,0],END:[0,0]}
    for b in batches:
        role=byid[b['member_request_ids'][0]]['load']['role'];n=b['npu_id'];idx=0 if role=='long' else 1
        a=max(START,b['admission_time_ms']);z=min(END,b['completion_time_ms'])
        if z>a:
            events.setdefault(a,[0,0])[idx]+=1;events.setdefault(z,[0,0])[idx]-=1
        prev=b['admission_time_ms']
        for layer in b['layer_metrics']:
            cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
            for key,x,y in ((role,cs,ce),('stall',prev,cs)):
                x=max(x,START);y=min(y,END)
                if y>x:bars[key][n].append((x/1000,(y-x)/1000))
            prev=ce
    total=sum(d for k in ('long','short') for lane in bars[k] for _,d in lane)
    U=total/64*100
    colors={'long':'#D17A24','short':'#138C9E','stall':'#D4D8DD'}
    fig,ax=plt.subplots(figsize=(14,10))
    for n in range(32):
        for role in ('stall','long','short'):ax.broken_barh(bars[role][n],(n-.38,.76),facecolors=colors[role],linewidth=0)
    ax.set(xlim=(2,4),ylim=(31.8,-.8),yticks=list(range(32)),xlabel='仿真时间（秒）',ylabel='NPU 编号',title=f'每卡长短混合：Baseline 的计算与 I/O 等待\n32 NPU · 6 SSU · warm [2,4)s · 平均利用率 {U:.2f}%')
    ax.axhline(16.5,color='#777',lw=.8);ax.axhline(19.5,color='#777',lw=.8);ax.grid(axis='x',alpha=.18)
    ax.legend(handles=[Patch(color=colors[k],label=v) for k,v in [('long','长请求计算'),('short','短请求计算'),('stall','I/O stall')]],loc='upper center',bbox_to_anchor=(.5,-.06),ncol=3,frameon=False)
    fig.text(.02,.015,'按真实层事件裁剪绘制；灰色是计算等待，不能当作 SSD 连续服务时间。',fontsize=10,color='#555')
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('png','pdf','svg'):fig.savefig(out/f'{label}_npu_timeline.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    times=sorted(events);current=[0,0];ts=[];ls=[];ss=[];hist={}
    for i,t in enumerate(times[:-1]):
        current=[x+y for x,y in zip(current,events[t])];dt=times[i+1]-t
        ts.append(t/1000);ls.append(current[0]);ss.append(current[1]);hist[current[0]]=hist.get(current[0],0)+dt
    ts.append(4);ls.append(ls[-1]);ss.append(ss[-1]);mean=sum(n*d for n,d in hist.items())/2000
    fig,ax=plt.subplots(figsize=(13,4.8))
    ax.step(ts,ls,where='post',color=colors['long'],label=f'长请求卡数（时间平均 {mean:.2f}）',lw=1.7)
    ax.step(ts,ss,where='post',color=colors['short'],label=f'短请求卡数（时间平均 {32-mean:.2f}）',lw=1.5)
    ax.axhline(20,color=colors['long'],ls='--',alpha=.45,label='固定分卡对照：20 长卡')
    ax.axhline(12,color=colors['short'],ls='--',alpha=.45,label='固定分卡对照：12 短卡')
    ax.set(xlim=(2,4),ylim=(-.5,32.5),yticks=[0,4,8,12,16,20,24,28,32],xlabel='仿真时间（秒）',ylabel='当前接纳该角色请求的 NPU 数',title='混合输入在 warm 窗口内的长短卡数量（Baseline）')
    ax.grid(alpha=.15);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=2,frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'{label}_role_counts.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'label':label,'U':U,'mean_long':mean,'out':str(out)}))

if __name__=='__main__':main()
