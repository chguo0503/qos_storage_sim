#!/usr/bin/env python3
"""32-card warm-window ratio maps, from actual complete internal layer cycles."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import render_bandwidth_zoom as base

HERE,ROOT=base.HERE,base.ROOT
OUT=HERE/'figures/ssu4/fleet_cycle_ratio'
LEFT,RIGHT,N=2000.,4000.,32
DEN=N*(RIGHT-LEFT)
INK,MUTED,GRAY='#172d45','#546980','#c7cdd5'
read,sha,close,clip=base.read,base.sha,base.close,base.clip


def arrival_prefix(rows):
    rows=rows[np.argsort(rows[:,11],kind='stable')]
    starts,ends=rows[:,11],rows[:,12]
    assert np.all(starts[1:]>=ends[:-1]-1e-8)
    sums=np.r_[0.,np.cumsum(rows[:,6])]
    def at(t):
        idx=int(np.searchsorted(ends,t,side='right'))
        value=float(sums[idx])
        if idx<len(rows) and starts[idx]<t:
            value+=(t-starts[idx])*50/1000
        return value
    return at


def analyse(order):
    case=HERE/'validation20s/runs'/f'ssu4_{order}_k1_sync_seed7'/'baseline'
    man=read(case/'manifest.json.gz');raw=read(case/'result.json.gz')
    command=read(case/'command.json');trace=read(case/'trace.json.gz')
    for name,key in [('manifest','manifest_sha256'),('result','output_sha256'),('trace','trace_sha256')]:
        assert base.SOURCES[str((case/(name+'.json.gz')).relative_to(ROOT))]==command[key]
    assert trace['columns']==base.COLS and trace['completed_simulation'] and all(trace['checks'].values())
    assert trace['window_ms'][0]<=LEFT and trace['window_ms'][1]>=RIGHT
    assert raw['strategy']=='baseline' and raw['submit_seed']==7 and all(raw['summary']['invariants'].values())
    assert man['input_fingerprint']==raw['input_fingerprint']==trace['source']['input_fingerprint']
    assert man['metadata']['num_npu']==32 and man['metadata']['num_ssu']==4
    rows=np.asarray(trace.pop('rows'),dtype=float)
    assert np.allclose(rows[:,12]-rows[:,11],rows[:,6]*1000/50,atol=1e-8)
    assert np.allclose(rows[:,10]-rows[:,9],rows[:,6]*1000/40,atol=1e-8)
    assert np.all(rows[:,4]==(rows[:,3]+rows[:,1])%4)
    keys=rows[:,0].astype(np.int64)*8+rows[:,2].astype(np.int64)
    assert np.all((rows[:,3]>=0)&(rows[:,3]<2048))
    identities=keys*2048+rows[:,3].astype(np.int64)
    assert len(np.unique(identities))==len(rows)
    unique,inv=np.unique(keys,return_inverse=True)
    counts=np.bincount(inv);volumes=np.bincount(inv,weights=rows[:,6])
    first=np.full(len(unique),np.inf);last=np.full(len(unique),-np.inf)
    np.minimum.at(first,inv,rows[:,11]);np.maximum.at(last,inv,rows[:,12])
    stats={int(key):(int(counts[j]),float(volumes[j]),float(first[j]),float(last[j])) for j,key in enumerate(unique)}
    reqs={r['request_id']:r for r in man['requests']}
    groups={k:dict(cycles=0,wall_ms=0.,compute_ms=0.,received_GiB=0.) for k in ('A','B','gray')}
    cycles=[];per_npu=[]
    for npu in range(N):
        lane=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
        flat=[]
        for batch in lane:
            rid=batch['member_request_ids'][0]
            for layer in batch['layer_metrics']:flat.append(dict(rid=rid,**layer))
        at=arrival_prefix(rows[rows[:,1]==npu])
        window_wall=window_C=0.
        for current,following in zip(flat,flat[1:]):
            start,end=current['compute_start_ms'],following['compute_start_ms']
            visible=clip(start,end,LEFT,RIGHT)
            if visible<=0:continue
            deadline=current['compute_end_ms'];C=deadline-start;D=end-start
            close(following['io_start_time_ms'],start)
            close(end,max(deadline,following['io_ready_time_ms']))
            same=current['rid']==following['rid']
            complete=start>=LEFT and end<=RIGHT
            q=reqs[current['rid']]['load'];B=q['per_layer_kv_gb']*1000/C
            actual_C=clip(start,deadline,LEFT,RIGHT)
            group=q['role'] if same and complete else 'gray'
            actual_b=r=None
            if group!='gray':
                count,volume,first_received,last_received=stats[following['rid']*8+following['layer']]
                placements=man['placements'][reqs[following['rid']]['placement_index']]
                placement=placements[0 if len(placements)==1 else following['layer']]
                assert count==len(placement)
                close(volume,q['per_layer_kv_gb'])
                assert first_received>=start-1e-7 and last_received<=end+1e-7
                close(last_received,following['io_ready_time_ms'])
                measured=at(end)-at(start)
                close(measured,volume)
                actual_b=measured*1000/D;r=actual_b/B
                close(r,C/D);close(r*D,actual_C)
                assert 0<r<=1+1e-8
                groups[group]['received_GiB']+=measured
            g=groups[group];g['cycles']+=1;g['wall_ms']+=visible;g['compute_ms']+=actual_C
            window_wall+=visible;window_C+=actual_C
            cycles.append(dict(npu=npu,request_id=current['rid'],next_request_id=following['rid'],role=q['role'],
                compute_layer=current['layer']+1,read_layer=following['layer']+1,
                start_ms=start,end_ms=end,D_ms=D,C_ms=C,deadline_ms=deadline,
                clipped_start_ms=max(LEFT,start),clipped_end_ms=min(RIGHT,end),window_ms=visible,
                actual_window_compute_ms=actual_C,group=group,same_request=same,complete_cycle=complete,
                V_GiB=q['per_layer_kv_gb'],B_GiB_s=B,mean_b_GiB_s=actual_b,r=r))
        close(window_wall,RIGHT-LEFT)
        direct_C=sum(clip(l['compute_start_ms'],l['compute_end_ms'],LEFT,RIGHT) for l in flat)
        close(window_C,direct_C)
        per_npu.append(dict(npu=npu,compute_ms=direct_C,U_percent=100*direct_C/(RIGHT-LEFT)))
    for name,g in groups.items():
        g['share_percent']=100*g['wall_ms']/DEN
        g['U_contribution_pp']=100*g['compute_ms']/DEN
        g['loss_pp']=100*(g['wall_ms']-g['compute_ms'])/DEN
        g['weighted_r']=g['compute_ms']/g['wall_ms'] if name!='gray' else None
        g['weighted_mean_b_GiB_s']=g['received_GiB']*1000/g['wall_ms'] if name!='gray' else None
    direct_U=sum(x['U_percent'] for x in per_npu)/N
    rebuilt_U=sum(g['U_contribution_pp'] for g in groups.values())
    close(sum(g['wall_ms'] for g in groups.values()),DEN);close(direct_U,rebuilt_U)
    close(rebuilt_U+sum(g['loss_pp'] for g in groups.values()),100.)
    return dict(order=order,seed=7,num_npu=32,num_ssu=4,window_ms=[LEFT,RIGHT],denominator_card_ms=DEN,
        U_percent=direct_U,reconstructed_U_percent=rebuilt_U,groups=groups,per_npu=per_npu,cycles=cycles,
        all_cycles_cover_common_window=True,complete_internal_ratios_verified_with_physical_arrivals=True,
        no_new_simulation=True)


def draw(data):
    fig=plt.figure(figsize=(17.5,14),dpi=140,facecolor='white')
    fig.text(.055,.971,'32 张 NPU：按层周期看 b_i/B_i，再汇总整机利用率',fontsize=23,color=INK)
    fig.text(.79,.970,f'整机 U = {data["U_percent"]:.2f}%',fontsize=23,color='#174d76',weight='bold')
    fig.text(.055,.942,f'Baseline {data["order"].title()} · 32 NPU / 4 SSU × 40 GiB/s · seed 7 · 同一个 warm 窗口 [2,4) 秒',fontsize=14,color=MUTED)
    fig.text(.055,.913,'一个完整内部周期：D = 计算 C + 等下一层数据；平均 b_i = 周期内实际收到的 V / D；r = 平均 b_i / B_i = C / D',fontsize=14,color=INK)
    fig.text(.055,.886,'A（大读取·短计算）：V = 175.656 MiB，C = 6.024 ms，B_i = 28.476 GiB/s；B（小读取·长计算）：V = 38.5 MiB，C = 28.593 ms，B_i = 1.315 GiB/s',fontsize=11.5,color=MUTED)
    fig.text(.055,.861,'颜色表示整轮比值，不表示某一瞬间是否正在计算。每张卡独立按自己的层周期划分；最终统一汇总 [2,4) 秒。',fontsize=12,color=MUTED)
    ax=fig.add_axes([.065,.355,.68,.46])
    bar=fig.add_axes([.79,.355,.145,.46],sharey=ax)
    norm=Normalize(0,1);cmap=matplotlib.colormaps['RdYlGn']
    polys=[];values=[];gray=[]
    labelled={n:set() for n in range(N)}
    last_label_x={n:-math.inf for n in range(N)}
    labels=[]
    for cycle in data['cycles']:
        n=cycle['npu'];a,z=cycle['clipped_start_ms']/1000,cycle['clipped_end_ms']/1000
        shape=[(a,n-.43),(z,n-.43),(z,n+.43),(a,n+.43)]
        if cycle['group']=='gray':gray.append(shape)
        else:
            r=min(1.,cycle['r']);polys.append(shape);values.append(r)
            key=(cycle['group'],round(r,2))
            if cycle['D_ms']>=26 and key not in labelled[n] and len(labelled[n])<4 and (a+z)/2-last_label_x[n]>=.045:
                labels.append(((a+z)/2,n,f'{r:.2f}',r))
                labelled[n].add(key)
                last_label_x[n]=(a+z)/2
    ax.add_collection(PolyCollection(gray,facecolors=GRAY,edgecolors='white',linewidths=.15))
    colors=PolyCollection(polys,array=np.asarray(values),cmap=cmap,norm=norm,edgecolors='white',linewidths=.15)
    ax.add_collection(colors)
    for x,y,label,r in labels:
        ax.text(x,y,label,ha='center',va='center',fontsize=6.3,color='white' if r>.85 or r<.22 else INK)
    ax.set(xlim=(2,4),ylim=(31.6,-.6),yticks=np.arange(N),yticklabels=[str(i) for i in range(N)],
        xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4],xlabel='仿真时间（秒）',ylabel='NPU 编号')
    ax.tick_params(axis='y',labelsize=9);ax.tick_params(axis='x',labelsize=11)
    ax.set_title('色块宽度 = 这一轮耗时 D；颜色 = 这一轮 r（部分较宽色块直接标值）',loc='left',fontsize=14,pad=16,color=INK)
    for x in (2.25,2.5,2.75,3,3.25,3.5,3.75):ax.axvline(x,color='#c8cdd3',lw=.55,alpha=.3,zorder=0)
    u=np.array([p['U_percent'] for p in data['per_npu']])
    bar.barh(np.arange(N),100,color='#edf1f5',height=.72)
    bar.barh(np.arange(N),u,color='#277ca4',height=.72)
    for n,value in enumerate(u):bar.text(value-1.5,n,f'{value:.2f}%',ha='right',va='center',fontsize=8.5,color='white')
    bar.set(xlim=(0,100),xticks=[0,50,100],xlabel='实际 U（%）')
    bar.tick_params(axis='y',left=False,labelleft=False)
    bar.set_title('每张卡的真实计算占比',fontsize=12,pad=16,color=INK)
    for a in (ax,bar):a.spines[['top','right']].set_visible(False)
    cax=fig.add_axes([.075,.288,.43,.014])
    cbar=fig.colorbar(colors,cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1])
    cbar.set_label('r = 周期平均 b_i / B_i：越接近 0，整轮等待占比越高',fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    grayax=fig.add_axes([.575,.277,.025,.025]);grayax.set_facecolor(GRAY);grayax.set(xticks=[],yticks=[])
    for spine in grayax.spines.values():spine.set_visible(False)
    fig.text(.612,.291,'灰色：跨请求或窗口截断的周期',fontsize=12,color=MUTED)
    fig.text(.612,.273,'不套内部层比值，按真实计算时间补回整机 U',fontsize=11,color=MUTED)
    tableax=fig.add_axes([.06,.091,.875,.14]);tableax.axis('off')
    rows=[]
    for name in ('A','B','gray'):
        g=data['groups'][name]
        label=f'{name} 完整内部周期（{g["cycles"]} 层）' if name!='gray' else '跨请求 / 窗口截断，按日志补回'
        rows.append([label,f'{g["share_percent"]:.2f}%',
            f'{g["weighted_mean_b_GiB_s"]:.3f}' if name!='gray' else '—',
            f'{g["weighted_r"]:.4f}' if name!='gray' else '不套比值',
            f'{g["U_contribution_pp"]:.2f}',f'{g["loss_pp"]:.2f}'])
    rows.append(['整机合计','100.00%','—','—',f'{data["U_percent"]:.2f}',f'{100-data["U_percent"]:.2f}'])
    headers=['窗口分组','卡时间占比','周期平均 b_i\nGiB/s','平均 r\n按周期时长加权','贡献给整机 U\n百分点','等待造成的损失\n百分点']
    table=tableax.table(cellText=rows,colLabels=headers,colWidths=[.28,.115,.125,.13,.175,.175],
        cellLoc='center',loc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d3dce5');cell.set_linewidth(.6)
        if row==0:cell.set_facecolor('#e8eff6');cell.set_text_props(color=INK,weight='bold')
        elif row==4:cell.set_facecolor('#eaf2f7');cell.set_text_props(weight='bold',color=INK)
        elif row==3:cell.set_facecolor('#f0f1f3')
        elif data['order']=='ordered' and row==1:cell.set_facecolor('#fff0e5')
        else:cell.set_facecolor('white')
    fig.text(.06,.062,f'U = [完整内部周期的 r×D 之和 + 灰色部分实际计算时间] / (32×2000 ms) = {data["U_percent"]:.2f}%',fontsize=14,color=INK)
    fig.text(.06,.038,'必须按周期时长加权，不能对层数简单平均。这里是在核对已有运行结果，不能仅凭这个比值预测另一种调度策略。',fontsize=11.5,color=MUTED)
    fig.text(.06,.017,'A/B 两行只含完整内部周期，不等同整请求类别利用率；本图为 seed 7 单次结果。所有周期精确数值保存在同目录 CSV。',fontsize=11,color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(artist.get_text(),box.bounds)
    path=OUT/f'fleet_cycle_ratio_{data["order"]}.png'
    fig.savefig(path,dpi=140);plt.close(fig)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),pixels=[width,height],visible_labels_inside_canvas=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results=[];images=[]
    for order in ('random','ordered'):
        data=analyse(order);results.append(data)
        image=draw(data);images.append(image)
        fields=list(data['cycles'][0])
        with (OUT/f'cycles_{order}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(data['cycles'])
        print(json.dumps({'order':order,'U':data['U_percent'],'groups':data['groups'],'image':image},ensure_ascii=False),flush=True)
    sources=dict(base.SOURCES)
    audit=dict(no_new_simulation=True,formats_created=['png'],sources=sources,
        builders={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),Path(base.__file__))},
        all_checks_passed=True,images=images,results=results)
    (OUT/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    md=['# 32 张 NPU：按层周期的 b_i/B_i 如何汇总成整机平均利用率','',
        '使用周六32 NPU、4 SSU × 40 GiB/s、长验证 seed 7 的既有日志；两图均统计同一个 warm [2,4) 秒。没有新增仿真，只生成 PNG。','',
        '[Random 全32卡PNG](fleet_cycle_ratio_random.png) · [Ordered 全32卡PNG](fleet_cycle_ratio_ordered.png)','',
        '每张卡都使用40个A、80个B。Ordered各卡都按ABB重复；Random各卡独立打乱这120个请求。没有运行时同步屏障。A/B来自data，输入画像如下：','',
        '| 请求 | 总输入token | miss token | 每层读取V | 每层计算C | B_i = V/C |',
        '|---|---:|---:|---:|---:|---:|',
        '| A：大读取·短计算 | 131072（128K） | 256 | 175.65625 MiB | 6.024012 ms | 28.475926 GiB/s |',
        '| B：小读取·长计算 | 32768（32K） | 4096 | 38.5 MiB | 28.592842 ms | 1.314932 GiB/s |','',
        '## 一层周期怎样定义','',
        'i 始终是 NPU 编号。每张卡单独划分周期，从当前层开始计算到下一层开始计算，包含当前层计算，以及必要的下一层等待。这里只给同一请求内部的完整周期定义带宽比值：','',
        '```text','D = 当前层开始计算，到下一层开始计算的经过时间','平均 b_i = 该周期内实际收到的下一层完整数据量 / D',
        'B_i = 该请求每层读取量 V / 每层计算时间 C','r = 平均 b_i / B_i = C / D','```','',
        'b_i 根据逐块 NPU 接收起止时间积分，并核实一个周期恰好收到下一层 V；没有把 SSD 排队时间误当成持续传输时间。周期平均包含没有数据到达的时段。r=1 只说明该周期没有暴露等待，不说明链路没有更多余量。','',
        '## 怎样看图','',
        '每行是一张 NPU，每个彩色块是一轮完整内部层周期。宽度=D，颜色=r；部分足够宽的色块直接标出比值。红色偏低、绿色接近1。灰色是跨请求或被warm边缘截断的周期，不套这个内部公式；其真实计算时间没有漏掉，而是单独补回。右边是同一warm窗口每张卡真实利用率。','',
        '不能把所有色块的r直接算术平均。一个耗时34ms的周期，比一个耗时6ms的周期占用更多窗口卡时间。正确汇总为：','',
        '```text','U = [所有完整内部周期的 r×D 之和 + 灰色部分实际计算时间]','    / (32 × 2000 ms)','```','',
        '如果窗口恰好完整覆盖全部周期且不跨请求，这就是按周期时长加权的r平均。现在有请求切换及窗口边界，因此保留灰色补账，避免更换B_i口径或凭比例估算截断周期。','',
        '注意：32张卡不需要同时开始、同时结束一层。先按各自周期算，再累加到共同的2秒窗口。窗口的总卡时间是32×2秒=64卡秒；卡时间占比就是该类周期的总时长除以64卡秒。','',
        '## 这次的具体值','',
        '| 情况 | 分组 | 卡时间占比 | 周期平均b GiB/s | 按时长加权r | U贡献，百分点 | 等待损失，百分点 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for data in results:
        for name,g in data['groups'].items():
            label=f'{name}完整内部周期' if name!='gray' else '跨请求或窗口截断'
            b=f'{g["weighted_mean_b_GiB_s"]:.6f}' if name!='gray' else '—'
            r=f'{g["weighted_r"]:.6f}' if name!='gray' else '不套比值'
            md.append(f'| {data["order"].title()} | {label} | {g["share_percent"]:.4f}% | {b} | {r} | {g["U_contribution_pp"]:.4f} | {g["loss_pp"]:.4f} |')
    md+=['','Random真实整机U为99.484872%，Ordered为70.234884%。这里都是seed7，不能写成三种子均值。','',
        'Ordered的完整A内部周期平均r约0.1850，占34.1839%的窗口卡时间；这些周期贡献6.3252个百分点计算利用率，等待损失27.8587个百分点。这是低r周期占用大量时间后怎样拉低整窗U的直接记账证据。B完整内部周期r为1。','',
        '把Ordered表格直接代入，可以看到比值和最终U如何对上（下式有四舍五入）：','',
        '```text','A贡献：34.1839% × 0.185035 = 6.3252个百分点','B贡献：54.7285% × 1.000000 = 54.7285个百分点',
        '灰色贡献：按日志统计的真实计算时间 / 64卡秒 = 9.1812个百分点',
        '整机U = 6.3252% + 54.7285% + 9.1812% = 70.2349%',
        'A的等待损失 = 34.1839% × (1 − 0.185035) = 27.8587个百分点','```','',
        '例如之前NPU15的一个Ordered A内部周期：计算6.024ms，等待28.284ms，共34.308ms。实际接收175.656MiB，摊到整轮平均为5GiB/s；5/28.476=0.1756，也等于6.024/34.308。意思是这一轮约17.56%的时间在计算，其余在等。整窗A平均0.1850略高，是因为不是每个A周期都完全相同。','',
        '**解释边界：**这里的平均b与D来自同一次真实运行，因此r=C/D是工作量核对，不是独立预测。调整策略后，b、周期长度、A/B在warm窗口的占比和请求推进都会一起变化；不能固定其他量，仅凭r推断Once能追回多少。周六这批没有Once对照，也不满足严格逐盘逐时刻欠载。','',
        'A/B表中分组只含完整内部周期，不是此前按已接纳请求生命周期统计的类别利用率。跨请求首层的等待在这里落入前一层到下一层的灰色周期；没有改变原始请求统计口径。','',
        '[Random全部逐周期数值CSV](cycles_random.csv) · [Ordered全部逐周期数值CSV](cycles_ordered.csv) · [完整来源和校验](checks.json) · [生成代码](../../../render_fleet_cycle_ratio.py)','']
    (OUT/'README.md').write_text('\n'.join(md))


if __name__=='__main__':main()
