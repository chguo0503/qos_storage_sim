#!/usr/bin/env python3
"""PNG-only zooms of three complete internal cycles of one unchanged request."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import render_bandwidth_zoom as base

HERE,ROOT=base.HERE,base.ROOT
OUT=HERE/'figures/ssu3/internal_layer_zoom'
BLUE,PURPLE,ORANGE=base.BLUE,base.PURPLE,base.ORANGE
INK,GREEN,RED=base.INK,base.GREEN,base.RED
read,sha,close,clip,curve=base.read,base.sha,base.close,base.clip,base.curve


def select(lane,reqs,role):
    candidates=[]
    for b in lane:
        rid=b['member_request_ids'][0]
        if reqs[rid]['load']['role']!=role:continue
        layers=b['layer_metrics']
        left,right=layers[1]['compute_start_ms'],layers[4]['compute_start_ms']
        if 2000<=left<right<=4000:candidates.append((left,b))
    assert candidates,(role,'No complete internal window')
    return min(candidates,key=lambda x:x[0])[1],len(candidates)


def draw(order,npu,role,lane,reqs,man,rows):
    batch,count=select(lane,reqs,role)
    rid=batch['member_request_ids'][0];q=reqs[rid]['load'];layers=batch['layer_metrics']
    left,right=layers[1]['compute_start_ms'],layers[4]['compute_start_ms']
    duration=right-left;C=q['per_layer_us']/1000;V=q['per_layer_kv_gb']*1024;B=V/1024*1000/C
    events=[]
    selected=rows[(rows[:,0]==rid)&np.isin(rows[:,2],[2,3,4])]
    assert np.all(selected[:,1]==npu)
    placement_set=man['placements'][reqs[rid]['placement_index']]
    for k in (1,2,3):
        current,next_layer=layers[k:k+2]
        release,deadline,ready=current['compute_start_ms'],current['compute_end_ms'],next_layer['io_ready_time_ms']
        end=next_layer['compute_start_ms']
        close(next_layer['io_start_time_ms'],release)
        close(end,max(deadline,ready));close(deadline-release,C)
        wait=max(0.,ready-deadline);close(wait,next_layer['io_barrier_wait_ms'])
        blocks=selected[selected[:,2]==k+1];blocks=blocks[np.argsort(blocks[:,3])]
        placement=placement_set[0 if len(placement_set)==1 else k+1]
        assert len(blocks)==len(placement)
        assert np.array_equal(blocks[:,3],np.arange(len(placement)))
        for row,(disk,size) in zip(blocks,placement):
            assert row[4]==disk and row[6]==size
        close(blocks[:,6].sum()*1024,V);close(blocks[:,12].max(),ready)
        assert blocks[:,11].min()>=release-1e-7 and ready<=end+1e-7
        events.append(dict(compute_layer=k+1,read_layer=k+2,release_ms=release,deadline_ms=deadline,
            ready_ms=ready,end_ms=end,stall_ms=wait,blocks=blocks,
            receive_start_ms=float(blocks[:,11].min()),effective_receive_ms=float((blocks[:,12]-blocks[:,11]).sum())))
    # All IO in the window must belong to these three layers of this request.
    foreign=rows[~((rows[:,0]==rid)&np.isin(rows[:,2],[2,3,4]))]
    for start,end,rate in ((9,10,40),(11,12,50)):
        other_ms=np.maximum(0,np.minimum(foreign[:,end],right)-np.maximum(foreign[:,start],left)).sum()
        close(float(other_ms),0.)
    ssd=curve(rows,9,10,40,left,right);link=curve(rows,11,12,50,left,right)
    close(ssd[2][-1]*1024,3*V);close(link[2][-1]*1024,3*V)
    assert ssd[1].max()<=120+1e-8 and link[1].max()<=50+1e-8
    wait=sum(e['stall_ms'] for e in events)
    close(3*C+wait,duration)
    actual_compute=sum(clip(l['compute_start_ms'],l['compute_end_ms'],left,right) for l in layers)
    close(actual_compute,3*C)
    mean_b=link[2][-1]*1000/duration;U=100*actual_compute/duration
    close(100*mean_b/B,U)
    example=events[0];deadline=example['deadline_ms']-left;ready=example['ready_ms']-left
    ts,rs,vs=curve(example['blocks'],9,10,40,left,right)
    tl,rl,vl=curve(example['blocks'],11,12,50,left,right)
    arrived=float(np.sum(example['blocks'][:,6]*1024*np.clip(
        (example['deadline_ms']-example['blocks'][:,11])/(example['blocks'][:,12]-example['blocks'][:,11]),0,1)))
    missing=max(0.,V-arrived)

    fig=plt.figure(figsize=(16,12.7),dpi=120,facecolor='white')
    fig.text(.065,.966,f'Baseline {order.title()} · NPU {npu:02d} · 同一个 {role} 请求内部的 3 层',fontsize=23,color=INK)
    fig.text(.065,.935,f'32 NPU / 3 SSU × 40 GiB/s · seed 7 · 请求 {rid} · 只看第 2～4 层计算和第 3～5 层预取',fontsize=13,color='#53677e')
    fig.text(.065,.904,f'每层读取 V = {V:.2f} MiB    每层计算 C = {C:.3f} ms    B_i = {B:.2f} GiB/s，全图不变',fontsize=16,color=INK)
    fig.text(.065,.874,'紫虚线：参考需求 B_i',fontsize=13,color=PURPLE)
    fig.text(.365,.874,'蓝实线：NPU 实际收到 b_i(t)',fontsize=13,color=BLUE)
    fig.text(.725,.874,'橙实线：SSD 为本卡读出',fontsize=13,color=ORANGE)
    fig.text(.065,.847,f'warm 窗内最早完整包含这 3 轮的 {role} 请求（{count} 个候选）；只有同一个请求，三层读取均完整。',fontsize=12,color='#53677e')
    ax_b=fig.add_axes([.085,.639,.87,.16])
    ax_s=fig.add_axes([.085,.432,.87,.12],sharex=ax_b)
    ax_v=fig.add_axes([.085,.271,.87,.115],sharex=ax_b)
    ax_c=fig.add_axes([.085,.134,.87,.067],sharex=ax_b)
    for ax in (ax_b,ax_s,ax_v,ax_c):
        ax.set_xlim(0,duration)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.13)
        # These shared markers refer only to the first illustrated prefetch.
        ax.axvline(deadline,color='#222',ls='--',lw=1.15)
        ax.axvline(ready,color=GREEN,ls=':',lw=1.5)
        if ready>deadline:ax.axvspan(deadline,ready,color='#f7dadd',alpha=.5,zorder=0)
    for ax in (ax_b,ax_s,ax_v):ax.tick_params(labelbottom=False)
    ax_b.stairs(link[1],link[0]-left,color=BLUE,lw=1.8,fill=True,alpha=.16)
    ax_b.stairs(link[1],link[0]-left,color=BLUE,lw=1.8)
    ax_b.axhline(B,color=PURPLE,ls='--',lw=2.3)
    ax_b.set(ylim=(0,70),yticks=[0,25,50],ylabel='GiB/s')
    ax_b.set_title('① NPU 接收：每一层读取量相同；蓝色脉冲的高度是速率，面积是该层数据量',loc='left',fontsize=14,pad=14,color=INK)
    for e in events:
        a,z=e['release_ms']-left,e['end_ms']-left
        p=(max(left,e['receive_start_ms'])+e['ready_ms'])/2-left
        ax_b.annotate(f'读第{e["read_layer"]}层\n{V:.2f} MiB',xy=(p,50),xytext=((a+z)/2,66),
            ha='center',va='top',fontsize=11,color=BLUE,
            arrowprops={'arrowstyle':'->','lw':.85,'color':BLUE},
            bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':1})
    ax_b.text(.70,.14,f'B_i = {B:.2f}（固定）',transform=ax_b.transAxes,fontsize=12,color=PURPLE,
        bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    fig.text(.085,.612,f'每层在 50 GiB/s 下的有效接收时间都是 {V/1024*1000/50:.3f} ms；若分段到达，把各段面积相加。蓝色正在读下一层。',fontsize=11.5,color=BLUE)
    fig.text(.085,.589,'以下黑虚线、绿点线只标记第 3 层这次预取：黑 = 第 2 层计算结束，绿 = 第 3 层数据收齐。',fontsize=11,color='#53677e')
    ax_s.stairs(ssd[1],ssd[0]-left,color=ORANGE,lw=1.8,fill=True,alpha=.33)
    ax_s.stairs(ssd[1],ssd[0]-left,color=ORANGE,lw=1.8)
    ax_s.set(ylim=(0,135),yticks=[0,40,80,120],ylabel='GiB/s')
    ax_s.set_title('② SSD 读出：三盘为本卡服务的速率之和；它与上方 NPU 接收的纵轴刻度不同',loc='left',fontsize=14,pad=14,color=INK)
    ax_v.plot(ts-left,vs*1024,color=ORANGE,lw=2,label='第3层：SSD 已读出')
    ax_v.plot(tl-left,vl*1024,color=BLUE,lw=2,label='第3层：NPU 已收到')
    ax_v.plot([0,C,duration],[0,V,V],color=PURPLE,ls='--',lw=1.8,label='按 B_i 均匀到齐的参考进度')
    ax_v.set(ylim=(-.05*V,1.42*V),yticks=[0,V/2,V],ylabel='累计 MiB')
    ax_v.set_title(f'③ 只看第 3 层：算第 2 层时预取第 3 层，预算就是本请求的 {C:.3f} ms',loc='left',fontsize=14,pad=14,color=INK)
    ax_v.legend(loc='upper left',ncol=3,fontsize=10.5,frameon=False)
    if missing>1e-6:
        caption=f'截止还缺 {missing:.2f} MiB → 等待 {example["stall_ms"]:.3f} ms'
        caption_color=RED
    else:
        caption=f'提前 {deadline-ready:.3f} ms 到齐 → 下一层无需等待'
        caption_color=GREEN
    ax_v.text(.42,.25,caption,transform=ax_v.transAxes,fontsize=12,color=caption_color,
        bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    ax_c.set(ylim=(0,1),yticks=[],xlabel='从这个请求第 2 层计算开始计时（ms）',xticks=np.linspace(0,duration,7))
    ax_c.set_xticklabels([f'{t:.1f}' for t in np.linspace(0,duration,7)])
    ax_c.spines['left'].set_visible(False)
    ax_c.set_title('④ NPU 的 3 个完整周期：深色 = 计算，粉红 = 等待下一层数据；没有跨请求或边缘裁剪',loc='left',fontsize=14,pad=14,color=INK)
    for e in events:
        a,d,z=e['release_ms']-left,e['deadline_ms']-left,e['end_ms']-left
        ax_c.broken_barh([(a,d-a)],(.05,.9),facecolor=base.COLORS[role],edgecolor='none')
        ax_c.text((a+d)/2,.5,f'算第{e["compute_layer"]}层\n{C:.3f} ms',ha='center',va='center',fontsize=10.5,color='white')
        if z>d:
            ax_c.broken_barh([(d,z-d)],(.05,.9),facecolor=base.COLORS['stall'],edgecolor='none')
            if z-d>duration*.035:
                ax_c.text((d+z)/2,.5,f'等第{e["read_layer"]}层\n{z-d:.3f} ms',ha='center',va='center',fontsize=11,color=RED)
    fig.text(.065,.075,'同一张图上下共用时间轴；不同图的时间跨度可能不同，跨图请看毫秒刻度，不要直接比较屏幕上的脉冲宽度。',fontsize=11.5,color='#53677e')
    fig.text(.065,.050,f'本图 3 轮：计算 {3*C:.3f} ms，等待 {wait:.3f} ms，局部 U = {U:.2f}%；平均接收 b = {mean_b:.3f} GiB/s（包含未接收时段）。',fontsize=12,color=INK)
    fig.text(.065,.028,f'绝对窗口 [{left:.6f}, {right:.6f}) ms，长度 {duration:.3f} ms；这是局部 3 轮，不是两秒 warm 平均。',fontsize=11,color='#53677e')
    fig.text(.065,.009,'逐块日志重建，无分箱、无平滑；层号从第 1 层开始。平均接收速率不等于链路限速，瞬时 b_i(t)/B_i 不等于瞬时利用率。',fontsize=10.5,color='#53677e')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=width+2 and b.y1<=height+2,(order,npu,role,artist.get_text(),b.bounds)
    path=OUT/f'{order}_{role.lower()}_npu_{npu:02d}.png'
    fig.savefig(path,dpi=120);plt.close(fig)
    return dict(order=order,role=role,npu=npu,num_ssu=3,num_npu=32,ssd_capacity_GiB_s=120,request_id=rid,candidate_count=count,
        selection='Earliest same-role request whose complete compute layers 1..3 and prefetch layers 2..4 fit warm [2000,4000)ms; layer indices here are zero-based.',
        window_ms=[left,right],duration_ms=duration,C_ms=C,V_MiB=V,B_GiB_s=B,
        mean_received_GiB_s=mean_b,local_U_percent=U,compute_ms=actual_compute,stall_ms=wait,
        layer_events=[{k:v for k,v in e.items() if k!='blocks'} for e in events],
        total_received_MiB=link[2][-1]*1024,total_ssd_MiB=ssd[2][-1]*1024,
        no_other_requests_or_layers=True,all_three_layers_complete=True,visible_labels_inside_canvas=True,
        bandwidth_ratio_note='mean b / B = U here is a full-three-cycle work accounting identity, not an independent allocation prediction.',
        image=str(path.relative_to(ROOT)),sha256=sha(path))


def main():
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    for order in ('random','ordered'):
        case=HERE/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
        man=read(case/'manifest.json.gz');raw=read(case/'result.json.gz');command=read(case/'command.json');trace=read(case/'trace.json.gz')
        meta=man['metadata']
        assert meta['num_ssu']==3 and meta['num_npu']==32
        assert meta['disk_bw_gib_s']==40 and meta['npu_bw_gib_s']==50
        assert meta['n_layers']==8 and meta['seed']==7 and meta['order']==order
        for name,key in [('manifest','manifest_sha256'),('result','output_sha256'),('trace','trace_sha256')]:
            assert base.SOURCES[str((case/(name+'.json.gz')).relative_to(ROOT))]==command[key]
        assert trace['columns']==base.COLS and trace['completed_simulation'] and all(trace['checks'].values())
        assert trace['window_ms'][0]<=2000 and trace['window_ms'][1]>=4000
        assert raw['strategy']=='baseline' and raw['submit_seed']==7 and all(raw['summary']['invariants'].values())
        assert man['input_fingerprint']==raw['input_fingerprint']==trace['source']['input_fingerprint']
        arr=np.asarray(trace.pop('rows'),dtype=float)
        assert np.array_equal(np.unique(arr[:,4]),[0,1,2])
        assert np.allclose(arr[:,10]-arr[:,9],arr[:,6]*1000/40,atol=1e-8)
        assert np.allclose(arr[:,12]-arr[:,11],arr[:,6]*1000/50,atol=1e-8)
        reqs={r['request_id']:r for r in man['requests']}
        for npu in range(32):
            lane=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
            rows=arr[arr[:,1]==npu]
            for role in ('A','B'):results.append(draw(order,npu,role,lane,reqs,man,rows))
            if npu%8==7:print(f'{order}: {(npu+1)*2}/64 PNG complete',flush=True)
        del arr,trace,raw,man
    audit=dict(no_new_simulation=True,formats_created=['png'],sources=base.SOURCES,
        builders={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(base.__file__))},
        all_checks_passed=True,num_ssu=3,num_npu=32,ssd_capacity_GiB_s=120,images=results)
    (OUT/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 单个请求内部的逐层带宽图（PNG）','',
        '主入口：32 NPU、3 SSU × 40 GiB/s、长验证 seed 7。每卡有 Random A、Ordered A、Random B、Ordered B 四张图。每图只包含同一个请求的第2～4层计算及第3～5层预取，均完整、无跨请求、无读取边缘裁剪。','',
        '选 warm [2,4) 秒内最早能完整包含上述片段的对应画像请求，不选最大等待。A 各层175.65625 MiB / 6.024012 ms / B_i=28.475926 GiB/s；B各层38.5 MiB / 28.592842 ms / B_i=1.314932 GiB/s。','',
        '紫虚线是恒定需求 B_i，蓝线是 NPU 实际接收，橙线是 SSD 为这张卡读出。蓝色每层总面积相同；实际预取的层比当时计算的层晚一层。黑虚线/绿点线只标记第3层这次预取的截止/到齐。','',
        '各图时间跨度可能不同：比较时看毫秒刻度，不能只看屏幕上的宽度。图内的利用率、平均带宽只统计这3轮，不是整机或两秒warm统计。','',
        '完整3轮恰好计算3C、收到3V，因此 mean(b)/B=(3V/T)/(V/C)=3C/T=U 是这里的工作量核对，不能称为独立预测。平均到达带宽包含未接收时段，不表示链路限速。','',
        '| NPU | Random A | Ordered A | Random B | Ordered B |','|---:|---|---|---|---|']
    for n in range(32):
        links=[f'[{o.title()} {r.upper()}]({o}_{r}_npu_{n:02d}.png)' for o,r in [('random','a'),('ordered','a'),('random','b'),('ordered','b')]]
        lines.append(f'| {n} | '+' | '.join(links)+' |')
    lines+=['','[数据和来源校验](checks.json) · [生成代码](../../../render_internal_layer_zoom_ssu3.py)','']
    (OUT/'index.md').write_text('\n'.join(lines))
    print(json.dumps({'png_count':len(results),'out':str(OUT),'npu15':[{k:r[k] for k in ('order','role','window_ms','local_U_percent','mean_received_GiB_s')} for r in results if r['npu']==15]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
