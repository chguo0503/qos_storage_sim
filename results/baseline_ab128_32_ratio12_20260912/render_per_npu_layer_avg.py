#!/usr/bin/env python3
"""Per-card layer-cycle averages; reuse verified physical-arrival accounting."""
from pathlib import Path
import json
import math
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import render_fleet_cycle_ratio as fleet

base = fleet.base
plt = fleet.plt
ROOT, HERE = base.ROOT, base.HERE
OUT = HERE / 'figures/ssu4/per_npu'
CACHE = HERE / 'figures/ssu4/fleet_cycle_ratio/checks.json'
BLUE, PURPLE, GRAY = '#0068d9', '#a32b91', '#e0e4e9'
INK, MUTED = '#172d45', '#546980'
STATE = {'A': '#52657c', 'B': '#16866d', 'stall': '#e99925'}
IMAGES = []


def verified_data():
    cached = json.loads(CACHE.read_text())
    assert cached['all_checks_passed']
    for section in ('sources', 'builders'):
        for name, digest in cached[section].items():
            assert base.sha(ROOT/name) == digest, name
    result = {}
    for data in cached['results']:
        order = data['order']
        case = HERE/'validation20s/runs'/f'ssu4_{order}_k1_sync_seed7'/'baseline'
        man, raw = base.read(case/'manifest.json.gz'), base.read(case/'result.json.gz')
        reqs = {q['request_id']: q for q in man['requests']}
        lanes = {}
        for npu in range(32):
            cycles = sorted((c for c in data['cycles'] if c['npu'] == npu), key=lambda c:c['start_ms'])
            batches = sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id'] == npu),
                             key=lambda b:b['admission_time_ms'])
            demands = []
            for batch in batches:
                q = reqs[batch['member_request_ids'][0]]['load']
                if base.clip(batch['admission_time_ms'], batch['completion_time_ms'], 2000., 4000.) > 0:
                    demands.append((batch['admission_time_ms'], batch['completion_time_ms'],
                                    q['per_layer_kv_gb']*1e6/q['per_layer_us']))
            assert math.isclose(sum(base.clip(a,z,2000.,4000.) for a,z,_ in demands),2000.,abs_tol=1e-7)
            edge, actual_C = 2000., 0.
            for cycle in cycles:
                base.close(cycle['clipped_start_ms'], edge)
                edge = cycle['clipped_end_ms']
                actual_C += cycle['actual_window_compute_ms']
                if cycle['group'] != 'gray':
                    base.close(cycle['mean_b_GiB_s']/cycle['B_GiB_s'], cycle['r'])
                    base.close(cycle['mean_b_GiB_s']*cycle['D_ms']/1000., cycle['V_GiB'])
                    base.close(cycle['r']*cycle['D_ms'], cycle['C_ms'])
                    assert cycle['same_request'] and cycle['complete_cycle']
            base.close(edge,4000.)
            direct = sum(base.clip(l['compute_start_ms'],l['compute_end_ms'],2000.,4000.)
                         for b in batches for l in b['layer_metrics'])
            base.close(direct,actual_C)
            base.close(100*direct/2000.,data['per_npu'][npu]['U_percent'])
            zooms = {}
            for role in ('A','B'):
                candidates = []
                for rid in {c['request_id'] for c in cycles if c['group'] == role}:
                    chosen = [c for c in cycles if c['request_id']==rid and c['group']==role and c['compute_layer'] in (2,3,4)]
                    if len(chosen)==3:
                        assert [c['compute_layer'] for c in chosen] == [2,3,4]
                        for a,z in zip(chosen,chosen[1:]): base.close(a['end_ms'],z['start_ms'])
                        candidates.append(chosen)
                assert candidates, (order,npu,role)
                zooms[role] = min(candidates,key=lambda cs:cs[0]['start_ms'])
            lanes[npu] = dict(cycles=cycles,demands=demands,zooms=zooms,U_percent=100*direct/2000.)
        result[order] = dict(lanes=lanes,U_percent=data['U_percent'])
    return cached,result


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y',alpha=.12)


def bandwidth(ax,lane,left=2000.,right=4000.,relative=False,zoom=None):
    offset = left if relative else 0.
    scale = 1. if relative else 1000.
    x = lambda t:(t-offset)/scale
    cycles = lane['cycles'] if zoom is None else zoom
    blue, purple, points = [], [], []
    previous = None
    for c in cycles:
        a,z=max(left,c['start_ms']),min(right,c['end_ms'])
        if z<=a: continue
        if c['group']=='gray':
            ax.axvspan(x(a),x(z),color=GRAY,zorder=0)
            previous=None
            continue
        b=c['mean_b_GiB_s']
        blue.append([(x(a),b),(x(z),b)])
        if previous is not None and math.isclose(previous['end_ms'],a,abs_tol=1e-7):
            blue.append([(x(a),previous['mean_b_GiB_s']),(x(a),b)])
        points.append(((x(a)+x(z))/2,b))
        previous=c
    previous=None
    for a,z,B in lane['demands']:
        a,z=max(left,a),min(right,z)
        if z<=a:continue
        purple.append([(x(a),B),(x(z),B)])
        if previous is not None:purple.append([(x(a),previous),(x(a),B)])
        previous=B
    ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.3,zorder=3))
    ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.8,linestyles='--',zorder=4))
    if points:
        px,py=zip(*points)
        ax.scatter(px,py,s=13 if zoom is None else 28,facecolors='white',edgecolors=BLUE,linewidths=1.1,zorder=5)
    ax.set(xlim=(x(left),x(right)),ylabel='GiB/s')
    if zoom is None:
        ax.set(ylim=(0,32.5),yticks=[0,5,10,20,28.476],xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4])
    else:
        B=zoom[0]['B_GiB_s']
        ax.set(ylim=(0,1.43*B),yticks=[0,B/2,B],yticklabels=['0',f'{B/2:.3f}',f'{B:.3f}'])
        edges=[c['start_ms'] for c in zoom]+[zoom[-1]['end_ms']]
        ax.set_xticks([x(t) for t in edges],[f'{x(t):.2f}' for t in edges])
        for c in zoom:
            a,z=x(c['start_ms']),x(c['end_ms'])
            ax.axvline(a,color='#abb6c2',ls=':',lw=.9)
            ax.text((a+z)/2,1.32*B,f'第{c["compute_layer"]}层周期\nD={c["D_ms"]:.3f} ms',
                    ha='center',va='top',fontsize=10,color=INK)
            b=c['mean_b_GiB_s']
            ax.text((a+z)/2,b+(.08*B if b<.7*B else -.16*B),f'b={b:.3f}',
                    ha='center',va='center',fontsize=10,color=BLUE,
                    bbox=dict(facecolor='white',alpha=.88,edgecolor='none',pad=1))
    style(ax)


def state(ax,cycles,left,right,relative=False):
    offset=left if relative else 0.
    scale=1. if relative else 1000.
    total=0.
    for c in cycles:
        for a,z,kind in ((c['start_ms'],c['deadline_ms'],c['role']),
                         (c['deadline_ms'],c['end_ms'],'stall')):
            a,z=max(left,a),min(right,z)
            if z>a:
                ax.broken_barh([((a-offset)/scale,(z-a)/scale)],(0,1),facecolors=STATE[kind])
                total+=z-a
    base.close(total,right-left)
    ax.set(xlim=((left-offset)/scale,(right-offset)/scale),ylim=(0,1),yticks=[])
    ax.tick_params(axis='x',bottom=False,labelbottom=False)
    ax.spines[['top','right','left','bottom']].set_visible(False)


def save(fig,path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.canvas.get_width_height()
    for text in fig.findobj(fleet.matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(text.get_text(),box.bounds)
    fig.savefig(path,dpi=140)
    plt.close(fig)
    IMAGES.append(dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path),pixels=[width,height]))


def legend(fig,y):
    handles=[Line2D([],[],color=PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),
             Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='平均 b_i：每个完整内部层周期取一个值'),
             Patch(facecolor=GRAY,label='灰区：跨请求或窗口截断，蓝线不填值')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,y),ncol=3,frameon=False,fontsize=11)


def single(order,npu,data):
    lane=data['lanes'][npu]
    fig=plt.figure(figsize=(16,10.5),dpi=140,facecolor='white')
    fig.text(.065,.963,f'NPU {npu:02d} · Baseline {order.title()}：把 b_i 改成每层周期平均值',fontsize=22,color=INK)
    fig.text(.065,.930,f'32 NPU / 4 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 本卡真实 U={lane["U_percent"]:.2f}% · 整机真实 U={data["U_percent"]:.2f}%',fontsize=12,color=MUTED)
    fig.text(.065,.902,'周期 D：开始算第 k 层 → 开始算第 k+1 层，包含中间等待；平均 b_i = 这段时间实际收到的第 k+1 层数据量 / D',fontsize=12,color=INK)
    legend(fig,.881)
    ax=fig.add_axes([.08,.535,.865,.275])
    bandwidth(ax,lane)
    ax.set_xlabel('仿真时间（秒）',fontsize=11)
    ax.set_title('完整 warm 窗口：蓝线高度是这一轮平均值，圆点标记各个内部周期；与紫虚线重合时，蓝线仍然存在。',loc='left',fontsize=12,pad=10,color=INK)
    st=fig.add_axes([.08,.449,.865,.025])
    state(st,lane['cycles'],2000.,4000.)
    fig.text(.08,.482,'实际计算 / 等待',fontsize=11,color=INK)
    for x,kind,label in ((.31,'A','A：大读取·短计算'),(.51,'B','B：小读取·长计算'),(.74,'stall','I/O 等待')):
        fig.text(x,.482,'■ '+label,color=STATE[kind],fontsize=11)
    for role,x in (('A',.08),('B',.56)):
        zoom=lane['zooms'][role]
        left,right=zoom[0]['start_ms'],zoom[-1]['end_ms']
        D=right-left
        C=sum(c['C_ms'] for c in zoom)
        b=sum(c['mean_b_GiB_s']*c['D_ms'] for c in zoom)/D
        zax=fig.add_axes([x,.148,.385,.215])
        bandwidth(zax,lane,left,right,True,zoom)
        zax.set_xlabel('从本局部起点经过的时间（ms）',fontsize=10)
        zax.set_title(f'{role} 同一请求内部 3 周期：平均 b/B={C/D:.4f}\n请求 {zoom[0]["request_id"]}；起点 {left/1000:.6f} 秒',loc='left',fontsize=12,pad=10,color=INK)
        zst=fig.add_axes([x,.077,.385,.022])
        state(zst,zoom,left,right,True)
        base.close(b/zoom[0]['B_GiB_s'],C/D)
    fig.text(.08,.040,'下方只放大同一请求内部，A/B 横轴范围及纵轴刻度不同；每张卡均取 warm 内首个符合条件的第 2～4 层周期。',fontsize=10.5,color=MUTED)
    fig.text(.08,.018,'灰色不代表 b_i=0，真实计算和等待仍计入 U；周期均值不是瞬时传输速率，也不是只对传输脉冲取平均。',fontsize=10.5,color=MUTED)
    save(fig,OUT/f'{order}_npu_{npu:02d}.png')


def paired(npu,data):
    fig=plt.figure(figsize=(16,10),dpi=140,facecolor='white')
    fig.text(.065,.963,f'NPU {npu:02d}：B_i 保持原义，b_i 改为每层周期平均值',fontsize=22,color=INK)
    fig.text(.065,.928,'32 NPU / 4 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 本图为 Random / Ordered 对照；分开的 PNG 在同目录',fontsize=12,color=MUTED)
    fig.text(.065,.894,'周期 D = 开始算第 k 层 → 开始算第 k+1 层（含等待）；平均 b_i = 这段时间收到的第 k+1 层数据量 / D',fontsize=12,color=INK)
    legend(fig,.875)
    for order,y in (('random',.55),('ordered',.205)):
        lane=data[order]['lanes'][npu]
        ax=fig.add_axes([.08,y,.865,.24])
        bandwidth(ax,lane)
        ax.set_title(f'{order.title()} · 本卡真实 U={lane["U_percent"]:.2f}% · 整机真实 U={data[order]["U_percent"]:.2f}%',loc='left',fontsize=13,pad=12,color=INK)
        st=fig.add_axes([.08,y-.060,.865,.024])
        state(st,lane['cycles'],2000.,4000.)
    fig.text(.51,.119,'仿真时间（秒）',ha='center',fontsize=11,color=INK)
    fig.text(.08,.092,'状态条：',fontsize=11,color=INK)
    for x,kind,label in ((.19,'A','A：大读取·短计算'),(.42,'B','B：小读取·长计算'),(.67,'stall','I/O 等待')):
        fig.text(x,.092,'■ '+label,color=STATE[kind],fontsize=11)
    fig.text(.08,.053,'灰区蓝线留空，不代表实际没有读取；这些时段的真实计算和等待均计入 U。蓝线与紫线重合时，用蓝色圆点标出。',fontsize=11,color=MUTED)
    fig.text(.08,.025,'b_i 的均值包含整轮计算和等待，不是瞬时传输速率；每个周期独立计算，不使用固定宽度的移动平均。',fontsize=11,color=MUTED)
    save(fig,OUT/f'npu_{npu:02d}.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cached,data=verified_data()
    for npu in range(32):
        for order in ('random','ordered'):single(order,npu,data[order])
        paired(npu,data)
        if npu%8==7:print(f'completed {npu+1}/32 NPUs',flush=True)
    audit=dict(all_checks_passed=True,no_new_simulation=True,generated_image_format='png',
        builder_sha256=base.sha(Path(__file__)),dependency_checks_sha256=base.sha(CACHE),
        verified_sources=cached['sources'],images=IMAGES,
        U_percent={order:d['U_percent'] for order,d in data.items()},
        local_examples={order:{str(n):{role:cs for role,cs in lane['zooms'].items()}
                              for n,lane in d['lanes'].items()} for order,d in data.items()})
    (OUT/'layer_average_checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    md=['# 逐卡带宽：按完整内部层周期平均的 b_i','',
        '32 NPU、4 SSU × 40 GiB/s、seed 7、同一warm [2,4) 秒。使用既有物理接收日志的逐层积分结果，没有新增仿真。','',
        '紫虚线B_i保持原义：当前已接纳请求每层读取量V / 每层纯计算时间C，在请求切换时改变。蓝线改为按层周期的实际平均接收速率。','',
        '```text','周期D：开始计算第k层 → 开始计算第k+1层','D = 第k层计算C + 等待第k+1层数据的时间',
        '平均b_i = 这整个周期内实际收到的第k+1层数据量 / D','```','',
        '例如Ordered NPU15的一个A内部周期：收到175.65625MiB，计算6.024ms，等待28.284ms，完整周期34.308ms。因此蓝线画为5GiB/s，紫线仍为28.476GiB/s。没有把实际50GiB/s的短暂传输画成密集尖峰，也没有只对传输中的时间取平均。','',
        '每个完整内部周期是一段水平蓝线，圆点标记这个周期；相邻周期均值相同会连成较长水平线。若b_i与B_i相等，蓝线和紫虚线重叠，蓝色圆点仍可见。','',
        '灰区表示跨请求或被warm窗口边缘截断的周期，蓝线不填值，不表示零带宽。跨请求时B_i可能在周期中途变化，不能直接套用内部的b_i/B_i=C/D；边界也不能拿完整V除以裁剪时长。所有灰区的真实计算和等待仍计入原来的warm利用率。','',
        '每张独立PNG下方还放大A、B各一个请求的3个完整内部周期（第2～4层计算），标明均值和周期耗时；按warm内最早符合条件选取，没有选择最大等待。两局部的横轴范围和纵轴刻度不同。','',
        '当前生成64张分开的PNG，并更新32张原npu_XX.png合并对照，方便旧链接继续使用。同名npu_XX.pdf是历史瞬时带宽版本，本次没有生成或修改PDF。','',
        '| NPU | Random：每层平均 PNG | Ordered：每层平均 PNG | 合并对照 PNG |',
        '|---:|---|---|---|']
    for n in range(32):md.append(f'| {n} | [Random](random_npu_{n:02d}.png) | [Ordered](ordered_npu_{n:02d}.png) | [对照](npu_{n:02d}.png) |')
    md+=['','[完整周期定义与整机汇总](../fleet_cycle_ratio/README.md) · [逐周期数据和来源校验](../fleet_cycle_ratio/checks.json) · [本次生成校验](layer_average_checks.json) · [生成代码](../../../render_per_npu_layer_avg.py)','']
    (OUT/'index.md').write_text('\n'.join(md))
    print(f'PASS: {len(IMAGES)} PNGs; no simulator or PDF generation',flush=True)


if __name__=='__main__':main()
