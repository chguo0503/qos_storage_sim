#!/usr/bin/env python3
"""Draw existing exact-service analysis; never imports or runs the simulator."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
FONT=Path('/home/chguo/.fonts/msyh.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
    'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'bandwidth-delivery-20260911'})
COLORS={'demand':'#b94641','ssd':'#389575','link':'#357bb1','stall':'#efa640'}
POLICIES=('baseline','once')
NAMES={'baseline':'Baseline','once':'Once per layer'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig,stem):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();canvas=fig.bbox
    outside=[]
    for item in fig.findobj(Text):
        if not item.get_visible() or not item.get_text().strip():continue
        box=item.get_window_extent(renderer)
        if box.x0<canvas.x0-1 or box.y0<canvas.y0-1 or box.x1>canvas.x1+1 or box.y1>canvas.y1+1:
            outside.append(item.get_text())
    assert not outside, outside
    stem.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for ext in ('png','pdf','svg'):
        target=stem.with_suffix('.'+ext);fig.savefig(target,dpi=175)
        files.append({'path':str(target),'sha256':sha(target)})
    plt.close(fig)
    return {'stem':str(stem),'layout':{'all_text_inside_canvas':True,'outside_text':[]},'files':files}


def validate(data):
    edges=np.array(data['bin_edges_ms'],dtype=float)
    a,z=data['window_ms'];dt=float(data['bin_ms'])
    assert len(edges)==1601 and a==3200 and z==4000 and dt==.5
    assert edges[0]==a and edges[-1]==z and np.allclose(np.diff(edges),dt,atol=1e-10,rtol=0)
    checks=[]
    for policy in POLICIES:
        item=data['policies'][policy]
        assert item['checks'] and all(v is True for v in item['checks'].values())
        for path_key in ('manifest','result','trace'):
            assert sha(item['source'][path_key])==item['source'][path_key+'_sha256']
        for key in ('demand_gib_s','ssd_gib_s','link_gib_s','stall_fraction'):
            arr=np.array(item['bins'][key],dtype=float)
            assert arr.shape==(32,1600) and np.all(np.isfinite(arr)) and np.min(arr)>-1e-7,(policy,key)
            if key=='stall_fraction':assert np.max(arr)<1+1e-7
        assert len(item['stall_intervals_ms'])==32 and len(item['npu_summary'])==32
        for n,summary in enumerate(item['npu_summary']):
            assert summary['npu']==n
            ssd=np.sum(item['bins']['ssd_gib_s'][n])*dt/1000*1024
            link=np.sum(item['bins']['link_gib_s'][n])*dt/1000*1024
            stall=np.sum(item['bins']['stall_fraction'][n])*dt
            exact_stall=math.fsum(max(0.,min(v,z)-max(u,a)) for u,v in item['stall_intervals_ms'][n])
            assert math.isclose(ssd,summary['ssd_MiB'],abs_tol=1e-5,rel_tol=1e-9)
            assert math.isclose(link,summary['link_MiB'],abs_tol=1e-5,rel_tol=1e-9)
            assert math.isclose(stall,summary['stall_ms'],abs_tol=1e-6,rel_tol=1e-9)
            assert math.isclose(stall,exact_stall,abs_tol=1e-6,rel_tol=1e-9)
            assert math.isclose(summary['U_percent'],100*(1-stall/(z-a)),abs_tol=1e-7)
            checks.append({'policy':policy,'npu':n,'ssd_bin_bytes_match':True,'link_bin_bytes_match':True,
                           'binned_and_exact_stall_match':True,'U_stall_accounting_match':True})
        for example in item['examples'].values():
            t=np.array(example['curves']['time_ms']);ss=np.array(example['curves']['ssd_cumulative_MiB']);lk=np.array(example['curves']['link_cumulative_MiB'])
            assert len(t)==len(ss)==len(lk) and np.all(np.diff(t)>=0)
            assert np.all(np.diff(ss)>=-1e-7) and np.all(np.diff(lk)>=-1e-7)
            assert np.min(ss)>-1e-7 and np.min(lk)>-1e-7 and np.all(lk<=ss+1e-6)
            D=example['payload_MiB'];deadline=example['deadline_ms']
            assert math.isclose(deadline-example['release_ms'],example['budget_C_ms'],abs_tol=1e-6)
            assert math.isclose(t[0],example['release_ms'],abs_tol=1e-6)
            assert math.isclose(t[-1],max(deadline,example['ready_ms']),abs_tol=1e-6)
            assert abs(ss[0])<1e-6 and abs(lk[0])<1e-6
            assert math.isclose(ss[-1],D,abs_tol=1e-5) and math.isclose(lk[-1],D,abs_tol=1e-5)
            assert math.isclose(np.interp(deadline,t,ss),example['ssd_before_deadline_MiB'],abs_tol=1e-5)
            assert math.isclose(np.interp(deadline,t,lk),example['link_before_deadline_MiB'],abs_tol=1e-5)
            assert math.isclose(max(0,example['ready_ms']-deadline),example['stall_ms'],abs_tol=1e-6)
    return checks


def rebinned(item,n,factor):
    answer={}
    for key in ('demand_gib_s','ssd_gib_s','link_gib_s'):
        v=np.array(item['bins'][key][n]);answer[key]=v.reshape(-1,factor).mean(axis=1)
    return answer


def step(ax,edges,y,**kwargs):
    ax.step(edges,np.r_[y,y[-1]],where='post',**kwargs)


def draw_npu(data,n,out,bin_ms):
    factor=round(bin_ms/data['bin_ms']);edges=np.array(data['bin_edges_ms'])[::factor]
    series={p:rebinned(data['policies'][p],n,factor) for p in POLICIES}
    maximum=max(float(max(v)) for item in series.values() for v in item.values())
    ceiling=math.ceil(max(50,maximum)*1.06/10)*10
    fig,axes=plt.subplots(2,1,figsize=(16,9.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.985,top=.735,bottom=.225,hspace=.30)
    fig.suptitle(f'NPU {n}：带宽诉求参考、SSD服务与NPU交付',fontsize=23,y=.968)
    fig.text(.5,.910,f'定序输入 · 32 NPU / 6 SSU · 每盘40 GiB/s · NPU链路50 GiB/s · seed7 · [{edges[0]/1000:g},{edges[-1]/1000:g})秒',ha='center',fontsize=13)
    fig.text(.5,.865,f'绿/蓝线按真实服务片段积分取{bin_ms:g}ms均值，红线是名义参考；两策略共用纵轴，不裁切服务峰值。',ha='center',fontsize=12)
    fig.legend(handles=[Line2D([],[],color=COLORS['demand'],ls='--',label='当前请求 D/C（名义参考）'),
                        Line2D([],[],color=COLORS['ssd'],label='6 SSU真实服务之和'),
                        Line2D([],[],color=COLORS['link'],label='NPU链路实际交付'),
                        Patch(color=COLORS['stall'],alpha=.24,label='实际暴露等待')],
               loc='upper center',bbox_to_anchor=(.53,.818),ncol=4,frameon=False,fontsize=12)
    for ax,policy in zip(axes,POLICIES):
        item=data['policies'][policy];summary=item['npu_summary'][n]
        for a,z in item['stall_intervals_ms'][n]:
            a,z=max(a,edges[0]),min(z,edges[-1])
            if z>a:ax.axvspan(a,z,color=COLORS['stall'],alpha=.24,lw=0,zorder=0)
        step(ax,edges,series[policy]['ssd_gib_s'],color=COLORS['ssd'],lw=1.10,zorder=2)
        step(ax,edges,series[policy]['link_gib_s'],color=COLORS['link'],lw=1.05,zorder=3)
        step(ax,edges,series[policy]['demand_gib_s'],color=COLORS['demand'],ls='--',lw=1.2,zorder=4)
        ax.axhline(50,color='#777777',ls=':',lw=.7,zorder=1)
        ax.set_title(f'{NAMES[policy]} · 本卡本窗U {summary["U_percent"]:.4f}% · 等待 {summary["stall_ms"]:.3f}ms',loc='left',fontsize=14)
        ax.set(ylim=(0,ceiling),xlim=(edges[0],edges[-1]),ylabel='GiB/s');ax.grid(alpha=.14)
    axes[-1].set(xlabel='仿真绝对时间（ms）',xticks=np.arange(edges[0],edges[-1]+1,100))
    fig.text(.08,.155,'红线仅为当前已接纳请求的每层D/C参考，不能沿请求寿命积分当作实际所需字节；下一请求L0仍真实执行。',fontsize=11.5)
    fig.text(.08,.108,'绿线汇总六盘归属于本NPU的实际服务，不是一块盘的带宽；蓝线计链路传输交付，不是进入链路队列的到达量。',fontsize=11.5)
    fig.text(.08,.061,'橙色严格按层日志等待区间绘制。两策略同卡同绝对窗可能处理不同请求；图中U是0.8秒局部窗，不是58秒均值。',fontsize=11.5)
    return save(fig,out/f'npu_{n:02d}')


def draw_overview(data,policy,out,bin_ms):
    factor=round(bin_ms/data['bin_ms']);edges=np.array(data['bin_edges_ms'])[::factor]
    maxima=[max(max(item['bins'][key][n]) for key in ('ssd_gib_s','link_gib_s','demand_gib_s'))
            for item in data['policies'].values() for n in range(32)]
    scale=math.ceil(max(50,max(maxima))/10)*10
    item=data['policies'][policy]
    fig,ax=plt.subplots(figsize=(16,14));fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.135)
    mean_u=math.fsum(x['U_percent'] for x in item['npu_summary'])/32
    fig.suptitle(f'{NAMES[policy]}：32卡带宽服务与等待概览',fontsize=23,y=.973)
    fig.text(.5,.926,f'定序输入 · [3.2,4.0)秒 · 本窗整机U {mean_u:.4f}% · 每行统一0–{scale:g} GiB/s · 箱宽{bin_ms:g}ms',ha='center',fontsize=14)
    fig.legend(handles=[Line2D([],[],color=COLORS['demand'],ls='--',label='当前请求 D/C（名义参考）'),
                        Line2D([],[],color=COLORS['ssd'],label='6 SSU真实服务之和'),
                        Line2D([],[],color=COLORS['link'],label='NPU链路实际交付'),
                        Patch(color=COLORS['stall'],alpha=.24,label='实际暴露等待')],
               loc='upper center',bbox_to_anchor=(.53,.885),ncol=4,frameon=False,fontsize=12)
    for n in range(32):
        s=rebinned(item,n,factor)
        ax.axhline(n+.39,color='#eeeeee',lw=.5)
        for a,z in item['stall_intervals_ms'][n]:
            a,z=max(a,edges[0]),min(z,edges[-1])
            if z>a:ax.add_patch(Rectangle((a,n-.42),z-a,.84,color=COLORS['stall'],alpha=.25,lw=0))
        for key,color,style in [('ssd_gib_s','ssd','-'),('link_gib_s','link','-'),('demand_gib_s','demand','--')]:
            step(ax,edges,n+.39-.78*s[key]/scale,color=COLORS[color],ls=style,lw=.55)
    ax.axhline(15.5,color='#888888',ls=':',lw=.8)
    ax.set(xlim=(edges[0],edges[-1]),ylim=(31.7,-.7),yticks=range(32),xticks=np.arange(edges[0],edges[-1]+1,100),
           ylabel='NPU编号；各行曲线向上表示带宽增加',xlabel='仿真绝对时间（ms）')
    ax.tick_params(axis='y',labelsize=10,length=0);ax.grid(axis='x',alpha=.12)
    fig.text(.065,.083,'每行下缘表示0，上缘为统一带宽上限；上下平移仅用于逐卡排版。所有曲线保留真实绝对时间。',fontsize=12)
    fig.text(.065,.045,'名义D/C不是实际字节到达曲线。细节请看每NPU独立图和单层累计字节图；本图不会把无服务直接判为stall。',fontsize=11.5)
    return save(fig,out/f'overview_{policy}')


def draw_cumulative(data,out):
    keys=('same_time_short','long_l0');fig,axes=plt.subplots(1,2,figsize=(16,10.2))
    fig.subplots_adjust(left=.075,right=.98,top=.71,bottom=.325,wspace=.21)
    fig.suptitle('同一真实层：预算截止前收到了多少字节',fontsize=23,y=.967)
    fig.text(.5,.913,'每条曲线以该策略此层的释放时刻为0；显示完整服务片段，未拼接不同层或改变实际持续时间。',ha='center',fontsize=13)
    fig.legend(handles=[Line2D([],[],color=COLORS['ssd'],label='Baseline · SSD服务'),
                        Line2D([],[],color=COLORS['link'],label='Baseline · NPU交付'),
                        Line2D([],[],color=COLORS['ssd'],ls='--',label='Once · SSD服务'),
                        Line2D([],[],color=COLORS['link'],ls='--',label='Once · NPU交付'),
                        Line2D([],[],color=COLORS['demand'],ls=':',label='均匀完成参考（封顶于D）')],
               loc='upper center',bbox_to_anchor=(.53,.874),ncol=3,frameon=False,fontsize=11.5)
    annotations=[]
    for ax,key,title in zip(axes,keys,('短请求内部层','长请求L0：前驱桥接请求提供预算')):
        samples={p:data['policies'][p]['examples'][key] for p in POLICIES};b=samples['baseline'];o=samples['once']
        assert (b['request_id'],b['layer'],b['npu'])==(o['request_id'],o['layer'],o['npu'])
        assert math.isclose(b['payload_MiB'],o['payload_MiB'],abs_tol=1e-7)
        D=b['payload_MiB'];end=max(max(e['curves']['time_ms'])-e['release_ms'] for e in samples.values())
        end=max(end,max(e['budget_C_ms'] for e in samples.values()))*1.1
        for policy,e in samples.items():
            t=np.array(e['curves']['time_ms'])-e['release_ms'];ls='-' if policy=='baseline' else '--'
            ax.plot(t,e['curves']['ssd_cumulative_MiB'],color=COLORS['ssd'],ls=ls,lw=1.6)
            ax.plot(t,e['curves']['link_cumulative_MiB'],color=COLORS['link'],ls=ls,lw=1.6)
        budget=b['budget_C_ms'];assert math.isclose(budget,o['budget_C_ms'],abs_tol=1e-7)
        ax.plot([0,budget,end],[0,D,D],color=COLORS['demand'],ls=':',lw=1.3)
        ax.axvline(budget,color='#666666',ls=':',lw=1)
        ax.text(budget+.15,D*.40,f'预算截止\n{budget:.3f}ms',fontsize=10,color='#555555')
        if b['stall_ms']>0:
            ax.axvspan(budget,b['ready_ms']-b['release_ms'],color=COLORS['stall'],alpha=.20,lw=0)
        ax.set_title(f'{title}\nNPU{b["npu"]} / rid{b["request_id"]} / L{b["layer"]}\nD={D:g}MiB / C={budget:.3f}ms = {b["budget_gib_s"]:.3f}GiB/s',fontsize=12,pad=15)
        ax.set(xlim=(0,end),xticks=np.arange(0,end+1e-8,5 if end>=15 else 2),ylim=(0,D*1.07),xlabel='距此层释放的时间（ms）',ylabel='累计MiB');ax.grid(alpha=.15)
        abs_text=f'实际释放：Baseline {b["release_ms"]:.6f}ms；Once {o["release_ms"]:.6f}ms'
        annotations.append((abs_text,f'{title}：Baseline截止欠交 {max(0,D-b["link_before_deadline_MiB"]):.6f}MiB，等待 {b["stall_ms"]:.6f}ms；Once欠交 {max(0,D-o["link_before_deadline_MiB"]):.6f}MiB，等待 {o["stall_ms"]:.6f}ms。'))
    fig.text(.075,.220,annotations[0][0],fontsize=11.5);fig.text(.075,.178,annotations[1][0],fontsize=11.5)
    fig.text(.075,.136,annotations[0][1],fontsize=11);fig.text(.075,.095,annotations[1][1],fontsize=11)
    fig.text(.075,.045,'红斜线仅示均匀完成路径，中途低于它不等于stall；仅在deadline核缺口，橙色为Baseline实际截止后等待。',fontsize=11.3)
    return save(fig,out/'cumulative_examples')


def draw_high_bandwidth_counterexample(data,out):
    e=data['policies']['baseline']['examples'].get('high_bandwidth_during_stall')
    if e is None:return None
    D=e['payload_MiB'];C=e['budget_C_ms'];wait=e['stall_ms'];ref=e['budget_gib_s']
    assert wait>0 and e['ssd_mean_during_stall_gib_s']>ref and e['role']=='short'
    times=np.array(e['curves']['time_ms'])-e['release_ms']
    fig,axes=plt.subplots(1,2,figsize=(16,9.7));fig.subplots_adjust(left=.075,right=.98,top=.72,bottom=.315,wspace=.22)
    fig.suptitle('带宽后来超过参考值，先前错过的计算截止仍会造成等待',fontsize=22,y=.965)
    fig.text(.5,.905,f'Baseline · 短请求 {e["profile"]} · NPU{e["npu"]} / rid{e["request_id"]} / L{e["layer"]}',ha='center',fontsize=14)
    fig.text(.5,.863,f'此层D={D:g}MiB；前驱计算预算C={C:.6f}ms；D/C={ref:.6f}GiB/s。此图仅展示这一真实层。',ha='center',fontsize=13)
    fig.legend(handles=[Line2D([],[],color=COLORS['ssd'],lw=2,label='SSD真实服务'),
                        Line2D([],[],color=COLORS['link'],lw=2,label='NPU链路实际交付'),
                        Line2D([],[],color=COLORS['demand'],ls=':',lw=2,label='均匀完成 / D/C参考'),
                        Patch(color=COLORS['stall'],alpha=.22,label='真实暴露等待')],
               ncol=4,loc='upper center',bbox_to_anchor=(.53,.824),frameon=False,fontsize=12)
    ax=axes[0]
    ax.plot(times,e['curves']['ssd_cumulative_MiB'],color=COLORS['ssd'],lw=1.8)
    ax.plot(times,e['curves']['link_cumulative_MiB'],color=COLORS['link'],lw=1.8)
    ax.plot([0,C,times[-1]],[0,D,D],color=COLORS['demand'],ls=':',lw=1.4)
    ax.axvline(C,color='#666666',ls=':',lw=1)
    ax.axvspan(C,C+wait,color=COLORS['stall'],alpha=.22,lw=0)
    ax.set(title='累计字节：只在真实截止时刻判断缺口',xlabel='距此层释放的时间（ms）',ylabel='累计MiB',
           xlim=(0,times[-1]*1.06),xticks=np.arange(0,times[-1]*1.06+1e-8,2),ylim=(0,D*1.08))
    ax.text(C+.2,D*.62,f'截止欠交\n{e["link_deficit_at_deadline_MiB"]:.3f}MiB',fontsize=11)
    ax.grid(alpha=.15)
    ax=axes[1];x=np.arange(2);width=.28
    ssd=[e['ssd_mean_during_budget_gib_s'],e['ssd_mean_during_stall_gib_s']]
    link=[e['link_mean_during_budget_gib_s'],e['link_mean_during_stall_gib_s']]
    ax.axvspan(.55,1.45,color=COLORS['stall'],alpha=.15,lw=0)
    for offset,values,color in [(-width/2,ssd,'ssd'),(width/2,link,'link')]:
        bars=ax.bar(x+offset,values,width,color=COLORS[color])
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.06,f'{v:.3f}',ha='center',fontsize=11)
    ax.axhline(ref,color=COLORS['demand'],ls=':',lw=1.3)
    ax.text(-.40,ref+.10,f'D/C参考 {ref:.3f}',fontsize=10,color=COLORS['demand'])
    ax.set(title='同一层：预算期与等待期的真实服务均值',ylabel='阶段平均GiB/s',xticks=x,
           xticklabels=[f'预算期\n{C:.3f}ms',f'等待期\n{wait:.3f}ms'],ylim=(0,max(ssd+link+[ref])*1.20),xlim=(-.5,1.5))
    ax.grid(axis='y',alpha=.15)
    fig.text(.075,.190,f'绝对时间：释放 {e["release_ms"]:.6f}ms → 预算截止 {e["deadline_ms"]:.6f}ms → ready {e["ready_ms"]:.6f}ms。',fontsize=12)
    fig.text(.075,.145,f'截止时NPU尚缺 {e["link_deficit_at_deadline_MiB"]:.6f}MiB，因此暴露等待 {wait:.6f}ms；等待期的SSD平均 {ssd[1]:.6f}GiB/s 已高于 {ref:.6f}GiB/s。',fontsize=11.5)
    fig.text(.075,.100,'阶段平均按该真实阶段内服务字节/阶段长度计算；高于参考的均值不表示每一瞬间都高于参考，也不会消除已经产生的等待。',fontsize=11)
    fig.text(.075,.052,'红色累计斜线仅为均匀完成示意，中途低于它不等于stall；实际缺口只在deadline计算，累计需求最多为该层D。',fontsize=11.3)
    answer=save(fig,out/'stall_despite_high_bandwidth')
    answer['example']={k:e[k] for k in ['request_id','layer','npu','release_ms','deadline_ms','ready_ms','payload_MiB','budget_C_ms','budget_gib_s','stall_ms','ssd_mean_during_budget_gib_s','ssd_mean_during_stall_gib_s','link_mean_during_budget_gib_s','link_mean_during_stall_gib_s']}
    return answer


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,default=HERE/'analysis.json')
    ap.add_argument('--output',type=Path)
    ap.add_argument('--bin-ms',type=float,choices=(.5,1.0),default=.5)
    ap.add_argument('--npu',type=int,nargs='+',default=list(range(32)))
    args=ap.parse_args();assert len(set(args.npu))==len(args.npu) and all(0<=n<32 for n in args.npu)
    out=args.output or HERE/('figures' if args.bin_ms==.5 else 'figures_1ms')
    data=read(args.analysis);checks=validate(data)
    figures=[draw_npu(data,n,out,args.bin_ms) for n in args.npu]
    figures += [draw_overview(data,p,out,args.bin_ms) for p in POLICIES]
    figures += [draw_cumulative(data,out)]
    counter=draw_high_bandwidth_counterexample(data,out)
    if counter is not None:figures.append(counter)
    audit=dict(no_simulation_run=True,analysis=str(args.analysis.resolve()),analysis_sha256=sha(args.analysis),
               script_sha256=sha(__file__),font_sha256=sha(FONT),window_ms=data['window_ms'],bin_ms=args.bin_ms,
               requested_npus=args.npu,full_32_npu_set=(sorted(args.npu)==list(range(32))),
               numerical_checks=checks,all_numerical_checks_passed=True,all_layout_checks_passed=True,
               figures=figures,figure_count=len(figures),image_file_count=sum(len(f['files']) for f in figures),
               reviewed_utc=datetime.now(timezone.utc).isoformat())
    (out/'render_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figure_count':len(figures),'all_layout_checks_passed':True,'output':str(out)}))


if __name__=='__main__':main()
