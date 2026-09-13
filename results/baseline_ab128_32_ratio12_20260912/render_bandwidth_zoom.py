#!/usr/bin/env python3
"""Readable per-card A-to-B bandwidth zooms, PNG only, from existing logs."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qos_bandwidth_zoom_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'figures/ssu4/per_npu_zoom'
FONT_PATH = Path('/home/chguo/.fonts/msyh.ttc')
if not FONT_PATH.exists():
    FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.font_manager.fontManager.addfont(str(FONT_PATH))
FONT = FontProperties(fname=str(FONT_PATH))
plt.rcParams.update({'font.family': FONT.get_name(), 'axes.unicode_minus': False, 'font.size': 12})
BLUE, PURPLE, ORANGE = '#0068d9', '#8b208f', '#e87900'
INK, GREEN, RED = '#192d45', '#148065', '#c83c48'
COLORS = {'A':'#52657c', 'B':'#16866d', 'stall':'#f8d4d8'}
COLS = ['request_id','npu_id','layer','block_idx','ssu_id','path_id','size_gib',
        'block_count','enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms']
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    with (gzip.open if path.suffix == '.gz' else open)(path,'rt') as stream:
        return json.load(stream)


def close(a,b):
    assert np.isclose(a,b,atol=1e-7,rtol=1e-10), (a,b)


def clip(a,z,left,right):
    return max(0.,min(z,right)-max(a,left))


def curve(rows,start,end,rate,left,right):
    """Exact rate steps, clipped intervals; no bins or moving average."""
    a,z = np.maximum(rows[:,start],left),np.minimum(rows[:,end],right)
    valid = z>a
    a,z = a[valid],z[valid]
    times,inv = np.unique(np.r_[left,right,a,z],return_inverse=True)
    changes = np.zeros(len(times))
    np.add.at(changes,inv[2:2+len(a)],rate)
    np.add.at(changes,inv[2+len(a):],-rate)
    rates = np.cumsum(changes)[:-1]
    volume = np.r_[0.,np.cumsum(rates*np.diff(times)/1000)]
    close(volume[-1],float(np.sum((z-a)*rate/1000)))
    assert rates.min()>=-1e-8
    return times,rates,volume


def select_events(lane,reqs):
    candidates=[]
    for prev,batch in zip(lane,lane[1:]):
        ar,br = [b['member_request_ids'][0] for b in (prev,batch)]
        if reqs[ar]['load']['role']!='A' or reqs[br]['load']['role']!='B':
            continue
        last,first = prev['layer_metrics'][-1],batch['layer_metrics'][0]
        release,deadline,ready = first['io_start_time_ms'],last['compute_end_ms'],first['io_ready_time_ms']
        if release-8<2000 or release+65>4000:
            continue
        close(release,last['compute_start_ms'])
        close(deadline,batch['admission_time_ms'])
        close(max(deadline,ready),first['compute_start_ms'])
        close(first['io_barrier_wait_ms'],max(0.,ready-deadline))
        candidates.append(dict(Arid=ar,Brid=br,release=release,deadline=deadline,ready=ready,
            stall_ms=first['io_barrier_wait_ms'],last=last,first=first,batch=batch))
    assert candidates
    longest=max(c['stall_ms'] for c in candidates)
    selected=min((c for c in candidates if longest-c['stall_ms']<1e-8),key=lambda c:c['release'])
    return selected,len(candidates)


def draw(order,npu,lane,reqs,man,rows):
    event,count=select_events(lane,reqs)
    release,deadline,ready=[event[k] for k in ('release','deadline','ready')]
    left,right=release-8,release+65
    C,R=deadline-release,ready-release
    target=rows[(rows[:,0]==event['Brid'])&(rows[:,2]==0)]
    target=target[np.argsort(target[:,3])]
    request=reqs[event['Brid']]
    placements=man['placements'][request['placement_index']]
    placement=placements[0]
    assert len(target)==len(placement)==224
    assert np.array_equal(target[:,3],np.arange(len(placement)))
    for row,(disk,size) in zip(target,placement):
        assert row[1]==npu and row[4]==disk and row[6]==size
    close(float(target[:,12].max()),ready)
    V=float(target[:,6].sum())*1024
    ref=V/1024*1000/C
    ssd=curve(rows,9,10,40,left,right)
    link=curve(rows,11,12,50,left,right)
    ts,rs,vs=curve(target,9,10,40,left,right)
    tl,rl,vl=curve(target,11,12,50,left,right)
    close(vs[-1]*1024,V);close(vl[-1]*1024,V)
    assert ssd[1].max()<=160+1e-8 and link[1].max()<=50+1e-8
    arrived=float(np.sum(target[:,6]*1024*np.clip((deadline-target[:,11])/(target[:,12]-target[:,11]),0,1)))
    missing=max(0.,V-arrived)
    # Label actual reads, separately from the layer being computed below.
    visible=rows[(rows[:,11]<right)&(rows[:,12]>left)]
    pulse_layers=[]
    for rid,layer_index in np.unique(visible[:,[0,2]],axis=0):
        group=rows[(rows[:,0]==rid)&(rows[:,2]==layer_index)]
        q=reqs[int(rid)]['load']
        received_ms=float(np.maximum(0.,np.minimum(group[:,12],right)-np.maximum(group[:,11],left)).sum())
        shown_MiB=received_ms*50/1000*1024
        full_MiB=q['per_layer_kv_gb']*1024
        assert shown_MiB<=full_MiB+1e-6
        pulse_layers.append(dict(request_id=int(rid),role=q['role'],layer_human=int(layer_index)+1,
            full_layer_MiB=full_MiB,visible_received_MiB=shown_MiB,
            full_layer_effective_receive_ms=q['per_layer_kv_gb']*1000/50,
            arrival_span_relative_ms=[float(group[:,11].min()-release),float(group[:,12].max()-release)],
            clipped=shown_MiB<full_MiB-1e-6))
    pulse_layers.sort(key=lambda row:row['arrival_span_relative_ms'][0])
    A_read=next(row for row in pulse_layers if row['request_id']==event['Arid'] and row['layer_human']==8)
    B_read=next(row for row in pulse_layers if row['request_id']==event['Brid'] and row['layer_human']==1)
    edge_reads=[row for row in pulse_layers if row['clipped']]
    if edge_reads:
        edge_caption='边缘裁剪：'+'；'.join(f'{row["role"]}第{row["layer_human"]}层仅显示 {row["visible_received_MiB"]:.2f} / 全层 {row["full_layer_MiB"]:.2f} MiB' for row in edge_reads)
    else:
        edge_caption='本图可见的各层读取均完整；B 的其余各层也都是 38.50 MiB。'
    next_layer=event['batch']['layer_metrics'][1]
    close(next_layer['io_start_time_ms'],event['first']['compute_start_ms'])
    hidden=event['first']['compute_end_ms']-next_layer['io_ready_time_ms']
    assert hidden>0
    compute=sum(clip(l['compute_start_ms'],l['compute_end_ms'],left,right) for b in lane for l in b['layer_metrics'])
    spans=[]
    for b in lane:
        rid=b['member_request_ids'][0];role=reqs[rid]['load']['role']
        start=b['admission_time_ms']
        for layer in b['layer_metrics']:
            cs,ce=layer['compute_start_ms'],layer['compute_end_ms']
            for a,z,k in ((start,cs,'stall'),(cs,ce,role)):
                a,z=max(left,a),min(right,z)
                if z>a:spans.append((a,z,k,layer['layer']))
            start=ce
    close(sum(z-a for a,z,_,_ in spans),right-left)

    fig=plt.figure(figsize=(16,13.4),dpi=120,facecolor='white')
    fig.text(.065,.966,f'Baseline {order.title()} · NPU {npu:02d}：把每条带宽线看清楚',fontsize=23,color=INK)
    fig.text(.065,.937,'32 NPU / 4 SSU × 40 GiB/s · NPU 接收上限 50 GiB/s · seed 7 · 逐块真实服务，无平滑',fontsize=13,color='#53677e')
    fig.text(.065,.906,'紫虚线 B_i：当前请求参考需求',fontsize=14,color=PURPLE)
    fig.text(.395,.906,'蓝实线 b_i(t)：NPU 实际收到',fontsize=14,color=BLUE)
    fig.text(.725,.906,'橙实线：SSD 为本卡读出',fontsize=14,color=ORANGE)
    fig.text(.065,.879,f'选取本卡 warm 窗内首层等待最长的 A→B 交接（共 {count} 次候选）；局部示例，不代表平均状态。',fontsize=12,color='#53677e')
    fig.text(.065,.856,f'本次 B 首层读取 {V:.2f} MiB；{C:.3f} ms 时应到齐，实际 {R:.3f} ms 到齐；IO stall = {event["stall_ms"]:.3f} ms。',fontsize=14,color=INK)
    ax_b=fig.add_axes([.085,.655,.87,.158])
    ax_s=fig.add_axes([.085,.445,.87,.12],sharex=ax_b)
    ax_v=fig.add_axes([.085,.277,.87,.12],sharex=ax_b)
    ax_c=fig.add_axes([.085,.145,.87,.065],sharex=ax_b)
    axes=(ax_b,ax_s,ax_v,ax_c)
    for ax in axes:
        ax.set_xlim(-8,65)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.12)
        ax.axvline(0,color='#8e99a7',lw=.9,ls=':')
        ax.axvline(C,color='#222',ls='--',lw=1.15)
        ax.axvline(R,color=GREEN,ls=':',lw=1.5)
        if R>C:ax.axvspan(C,R,color='#f7dadd',alpha=.48,zorder=0)
    for ax in axes[:-1]:ax.tick_params(labelbottom=False)
    ax_b.stairs(link[1],link[0]-release,color=BLUE,lw=1.9,fill=True,alpha=.16)
    ax_b.stairs(link[1],link[0]-release,color=BLUE,lw=1.9)
    demands=[]
    for b in lane:
        a,z=max(left,b['admission_time_ms']),min(right,b['completion_time_ms'])
        if z<=a:continue
        q=reqs[b['member_request_ids'][0]]['load'];d=q['per_layer_kv_gb']*1e6/q['per_layer_us']
        ax_b.hlines(d,a-release,z-release,color=PURPLE,ls='--',lw=2.5)
        if demands:ax_b.vlines(a-release,demands[-1][2],d,color=PURPLE,ls='--',lw=2.5)
        demands.append((a,z,d))
    ax_b.set(ylim=(0,68),yticks=[0,25,50],ylabel='GiB/s')
    ax_b.set_title('① 本卡接收：蓝线有高度才表示正在收数据；紫线是参考值，不是实际流量',loc='left',fontsize=14,pad=14,color=INK)
    ax_b.text(.19,.59,'A 请求：B_i = 28.48    →    B 请求：B_i = 1.31',transform=ax_b.transAxes,
        fontsize=12,color=PURPLE,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax_b.text(.19,.31,'黑虚线 = A 末层算完（截止）    绿点线 = 本次 B 首层数据收齐',transform=ax_b.transAxes,
        fontsize=11.5,color=INK,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    for read_event,text_x in ((A_read,-7.5),(B_read,max(6.,R-7.))):
        start,end=read_event['arrival_span_relative_ms']
        ax_b.annotate(f'{read_event["role"]} 第{read_event["layer_human"]}层：{read_event["full_layer_MiB"]:.2f} MiB\n有效接收 {read_event["full_layer_effective_receive_ms"]:.3f} ms',
            xy=((max(-8,start)+min(65,end))/2,50),xytext=(text_x,65),
            color=BLUE,fontsize=10,ha='left',va='top',
            arrowprops={'arrowstyle':'->','color':BLUE,'lw':.9},
            bbox={'facecolor':'white','alpha':.92,'edgecolor':'none','pad':1})
    fig.text(.085,.631,'蓝色脉冲：高度 = 速率，横向宽度 = 接收持续时间，面积 = 数据量。A 每层 175.66 MiB，B 每层 38.50 MiB；同类各层相同。',fontsize=11.5,color=BLUE)
    fig.text(.085,.608,edge_caption+' 一层若分段到达，需把各段面积相加。',fontsize=11,color='#53677e')
    ax_s.stairs(ssd[1],ssd[0]-release,color=ORANGE,lw=1.8,fill=True,alpha=.35)
    ax_s.stairs(ssd[1],ssd[0]-release,color=ORANGE,lw=1.8)
    ax_s.set(ylim=(0,180),yticks=[0,40,80,160],ylabel='GiB/s')
    ax_s.set_title('② 盘端读出：橙线是四盘为本卡服务的合计速率；读出后还需经过 NPU 接收链路',loc='left',fontsize=14,pad=14,color=INK)
    ax_s.text(.22,.64,'橙线可达 160，蓝线最多 50；两图纵轴不同，比较时请看刻度。',transform=ax_s.transAxes,
        fontsize=12,color='#975207',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax_v.plot(ts-release,vs*1024,color=ORANGE,lw=2,label='SSD 已读出')
    ax_v.plot(tl-release,vl*1024,color=BLUE,lw=2,label='NPU 已收到')
    ax_v.plot([0,C,65],[0,V,V],color=PURPLE,ls='--',lw=1.8,label='截止前均匀到齐的参考进度')
    ax_v.set(ylim=(-2,53),yticks=[0,20,38.5],ylabel='累计 MiB')
    ax_v.set_title(f'③ 只看本次 B 首层：借用前一个 A 的 {C:.3f} ms，需 {ref:.2f} GiB/s；不是 B 自身的 1.31',loc='left',fontsize=14,pad=14,color=INK)
    ax_v.legend(loc='upper left',ncol=3,fontsize=10.5,frameon=False)
    if missing>1e-6:
        ax_v.text(.24,.30,f'截止时还缺 {missing:.2f} MiB → 必须等到蓝线达到 38.5 MiB',transform=ax_v.transAxes,
            fontsize=12,color=RED,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    else:
        ax_v.text(.24,.30,f'比截止提前 {C-R:.3f} ms 收齐 → 本次首层不用等',transform=ax_v.transAxes,
            fontsize=12,color=GREEN,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    ax_c.set(ylim=(0,1),yticks=[],xlabel='相对本次 B 首层预取发出时刻的时间（ms）；0 = 预取发出',xticks=[-8,0,10,20,30,40,50,60])
    ax_c.spines['left'].set_visible(False)
    ax_c.set_title('④ NPU 状态：计算色块各对应一层；灰蓝 = A 请求计算，绿色 = B 请求计算，粉红 = IO 等待',loc='left',fontsize=14,pad=14,color=INK)
    for a,z,k,layer in spans:
        ax_c.broken_barh([(a-release,z-a)],(.06,.88),facecolor=COLORS[k],edgecolor='none')
        if z-a>=5:
            label=f'IO 等待\n{z-a:.2f} ms' if k=='stall' else f'{k} 请求 第{layer+1}层\n计算 {z-a:.2f} ms'
            if abs(a-left)<1e-8 or abs(z-right)<1e-8:
                label+='\n（窗内片段）'
            ax_c.text((a+z)/2-release,.5,label,ha='center',va='center',fontsize=10.5,
                color=RED if k=='stall' else 'white')
    fig.text(.065,.080,'上两图是本卡全部预取 IO；蓝色读取某层时，下方可能正在算前一层。累计图仅统计目标首层；紫线面积不是新增读取量。',fontsize=11.5,color='#53677e')
    fig.text(.065,.052,f'B 的下一层数据提前 {hidden:.3f} ms 到齐：之后即使蓝线为 0，也能继续计算。瞬时 b_i(t)/B_i 不是瞬时利用率。',fontsize=12,color=INK)
    fig.text(.065,.029,f'绝对窗口 [{left:.6f}, {right:.6f}) ms；A 请求 {event["Arid"]} → B 请求 {event["Brid"]}。局部本卡 U = {100*compute/(right-left):.2f}%。',fontsize=11,color='#53677e')
    fig.text(.065,.009,'边缘色块只显示窗内部分；这里的 73 ms 局部利用率不是两秒 warm 窗口利用率。层号从第 1 层开始。',fontsize=10,color='#53677e')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(npu,artist.get_text(),box.bounds)
    dest=OUT/f'{order}_npu_{npu:02d}.png'
    fig.savefig(dest,dpi=120)
    plt.close(fig)
    return dict(order=order,npu=npu,seed=7,num_ssu=4,num_npu=32,candidate_count=count,
        selection='Maximum B first-layer stall among full A-to-B events with plot inside warm [2000,4000)ms; ties within 1e-8ms select earliest.',
        window_ms=[left,right],relative_window_ms=[-8,65],Arid=event['Arid'],Brid=event['Brid'],
        release_ms=release,deadline_ms=deadline,ready_ms=ready,stall_ms=event['stall_ms'],
        payload_MiB=V,missing_at_deadline_MiB=missing,first_layer_budget_GiB_s=ref,
        pulse_layers=pulse_layers,blue_stroke_width_points=1.9,
        next_layer_early_ready_ms=hidden,local_npu_U_percent=100*compute/(right-left),
        ssd_GiB=ssd[2][-1],received_GiB=link[2][-1],rate_peaks_GiB_s={'ssd':float(ssd[1].max()),'npu':float(link[1].max())},
        byte_conservation=True,full_target_block_identity_verified=True,visible_labels_inside_canvas=True,
        image=str(dest.relative_to(ROOT)),sha256=sha(dest))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for order in ('random','ordered'):
        case=HERE/'validation20s/runs'/f'ssu4_{order}_k1_sync_seed7'/'baseline'
        man=read(case/'manifest.json.gz');raw=read(case/'result.json.gz')
        command=read(case/'command.json');trace=read(case/'trace.json.gz')
        for name,key in [('manifest','manifest_sha256'),('result','output_sha256'),('trace','trace_sha256')]:
            assert SOURCES[str((case/(name+'.json.gz')).relative_to(ROOT))]==command[key]
        assert trace['columns']==COLS and trace['completed_simulation'] and all(trace['checks'].values())
        assert trace['window_ms'][0]<=2000 and trace['window_ms'][1]>=4000
        assert raw['strategy']=='baseline' and raw['submit_seed']==7 and all(raw['summary']['invariants'].values())
        assert man['input_fingerprint']==raw['input_fingerprint']==trace['source']['input_fingerprint']
        arr=np.asarray(trace.pop('rows'),dtype=float)
        assert np.allclose(arr[:,10]-arr[:,9],arr[:,6]*1000/40,atol=1e-8)
        assert np.allclose(arr[:,12]-arr[:,11],arr[:,6]*1000/50,atol=1e-8)
        reqs={r['request_id']:r for r in man['requests']}
        for npu in range(32):
            lane=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
            outputs.append(draw(order,npu,lane,reqs,man,arr[arr[:,1]==npu]))
            if npu%8==7:print(f'{order}: {npu+1}/32 PNG complete',flush=True)
        del arr,trace,raw,man
    audit=dict(no_simulation=True,formats_created=['png'],sources=SOURCES,builder_sha256=sha(Path(__file__)),
        all_checks_passed=True,images=outputs)
    (OUT/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 4 SSU：每张 NPU 的带宽局部放大（只生成 PNG）','',
        '此目录保留历史 A→B 交接图。理解 b_i、B_i 请优先看 [单个请求内部的新图](../internal_layer_zoom/index.md)，避免混入跨请求首层预算。','',
        '使用周六长验证 seed 7 原日志，32 NPU、4 SSU × 40 GiB/s。每张卡分别选择 warm [2,4) 秒内首层等待最长的一次 A→B 交接；并列取最早。图是局部案例，不是整窗或整机平均。','',
        '所有图统一横轴 [-8,65] ms，0 为目标 B 首层预取发出。紫虚线是当前请求自身的 V/C，蓝线是 NPU 实际接收，橙线是 SSD 为本卡读出。两种实际带宽分别绘图；不要把不同纵轴的视觉高度直接比较。','',
        '蓝色笔画统一为 1.9 pt；看到的宽窄主要是接收脉冲持续时间。高度=速率，横向宽度=接收时间，曲线下方面积=数据量。A 每层 175.65625 MiB，在 50 GiB/s 下有效接收 3.430786 ms；B 每层 38.5 MiB，对应 0.751953 ms。同一画像各层读取量相同；有间隙时加总各段面积，边缘裁剪按图中说明理解。','',
        '累计面板只统计目标 B 首层，其预算来自前一个 A 的 6.024 ms，对应约 6.24 GiB/s，不是 B 内部层的 1.31 GiB/s。计算与等待条和带宽曲线使用同一条时间轴。','',
        '推荐先看 NPU 15：Random 首层不用等待，Ordered 首层等待约 25.605 ms。逐块服务重建、无分箱或平滑，没有运行新仿真。','',
        '| NPU | Random PNG | Ordered PNG |','|---:|---|---|']
    for n in range(32):lines.append(f'| {n} | [Random](random_npu_{n:02d}.png) | [Ordered](ordered_npu_{n:02d}.png) |')
    lines+=['','[来源与数值校验](checks.json) · [生成代码](../../../render_bandwidth_zoom.py)','']
    (OUT/'index.md').write_text('\n'.join(lines))
    print(json.dumps({'png_count':len(outputs),'output':str(OUT),'NPU15':[o for o in outputs if o['npu']==15]},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
