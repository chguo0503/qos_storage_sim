#!/usr/bin/env python3
"""New two-line fleet mean curves; never rewrite prior figures."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import render_per_npu_layer_avg_ssu3 as source
import warm_bandwidth_means_ssu3 as means

ROOT,HERE = source.ROOT,source.HERE
OUT = HERE/'figures/ssu3/fleet_mean_bandwidth_curves'


def combined(pieces,edges):
    changes=np.zeros(len(edges))
    for a,z,value in pieces:
        changes[np.searchsorted(edges,a)]+=value/32
        changes[np.searchsorted(edges,z)]-=value/32
    rates=np.cumsum(changes)
    source.base.close(rates[-1],0.)
    assert rates[:-1].min()>-1e-7
    return np.maximum(0.,rates[:-1])


def build(order,data):
    case=HERE/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
    trace=source.base.read(case/'trace.json.gz')
    command=source.base.read(case/'command.json')
    assert source.base.sha(case/'trace.json.gz')==command['trace_sha256']
    assert trace['columns']==source.base.COLS and trace['completed_simulation']
    assert all(trace['checks'].values()) and trace['window_ms'][0]<=2000<4000<=trace['window_ms'][1]
    arr=np.asarray(trace.pop('rows'),dtype=float)
    assert np.allclose(arr[:,12]-arr[:,11],arr[:,6]*1000/50,atol=1e-8)
    demand=[];supply=[];periods=[]
    for npu,lane in data['lanes'].items():
        prefix=source.fleet.arrival_prefix(arr[arr[:,1]==npu])
        for a,z,B in lane['demands']:
            a,z=max(2000.,a),min(4000.,z)
            if z>a:demand.append((a,z,B))
        wall=0.
        for c in lane['cycles']:
            a,z=c['clipped_start_ms'],c['clipped_end_ms']
            received=prefix(z)-prefix(a)
            b=received*1000/(z-a)
            assert received>=-1e-9 and 0<=b<=50+1e-6
            if c['group']!='gray':source.base.close(b,c['mean_b_GiB_s'])
            supply.append((a,z,b));wall+=z-a
            periods.append(dict(npu=npu,start_ms=a,end_ms=z,received_GiB=received,
                                mean_supply_GiB_s=b,source_group=c['group']))
        source.base.close(wall,2000.)
    edges=np.unique([t for pieces in (demand,supply) for a,z,_ in pieces for t in (a,z)])
    B,b=combined(demand,edges),combined(supply,edges)
    avg_B=float(np.dot(B,np.diff(edges))/2000.)
    avg_b=float(np.dot(b,np.diff(edges))/2000.)
    reference=means.load_verified(order)['totals']
    source.base.close(avg_B,reference['per_card_mean_demand_GiB_s'])
    source.base.close(avg_b,reference['per_card_mean_supply_GiB_s'])
    # Independent physical whole-window byte integral, including partial blocks.
    overlap=np.maximum(0.,np.minimum(arr[:,12],4000.)-np.maximum(arr[:,11],2000.))
    physical_GiB=float(np.sum(overlap)*50/1000.)
    source.base.close(avg_b*2*32,physical_GiB)
    fig=plt.figure(figsize=(15,6.7),dpi=160,facecolor='white')
    fig.text(.065,.945,f'Baseline {order.title()}：32 张 NPU 的平均带宽需求与供给',
             fontsize=23,color=source.INK)
    fig.text(.065,.894,'32 NPU / 3 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 纵轴是每卡平均值',
             fontsize=13,color=source.MUTED)
    fig.text(.065,.850,f'整个窗口的时间平均：需求 {avg_B:.3f} GiB/s / 卡；实际供给 {avg_b:.3f} GiB/s / 卡',
             fontsize=13,color=source.INK)
    ax=fig.add_axes([.085,.225,.87,.555])
    ax.stairs(B,edges/1000,baseline=None,color=source.PURPLE,lw=2.3,ls='--',label='平均需求：32 卡当前 B_i 的均值')
    ax.stairs(b,edges/1000,baseline=None,color=source.BLUE,lw=2.0,label='平均供给：各卡按层周期平均后，再对 32 卡平均')
    ax.set(xlim=(2,4),ylim=(0,32.5),xticks=np.arange(2,4.01,.25),yticks=[0,5,10,15,20,25,30],
           xlabel='仿真时间（秒）',ylabel='平均带宽（GiB/s / 卡）')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,
              frameon=False,fontsize=10.5,borderaxespad=0)
    fig.text(.065,.115,'供给先按每张卡各自的层周期取平均（包含等待），再求 32 卡均值；不是原始瞬时传输速率。',
             fontsize=12,color=source.MUTED)
    fig.text(.065,.073,'跨请求周期也计入；窗口两端只用窗内实际收到的字节和片段时长，因此曲线面积与整窗收到量一致。',
             fontsize=11.5,color=source.MUTED)
    fig.text(.065,.032,'需求是 V/C 参考值的跨卡平均；两条线之比不能作为 NPU 利用率。',fontsize=11.5,color=source.MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for artist in fig.findobj(source.fleet.matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(artist.get_text(),box.bounds)
    png=OUT/f'{order}_mean_demand_supply.png'
    fig.savefig(png,dpi=160);plt.close(fig)
    with (OUT/f'{order}_curves.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['start_ms','end_ms','mean_demand_GiB_s_per_card','mean_supply_GiB_s_per_card'])
        w.writerows(zip(edges[:-1],edges[1:],B,b))
    return dict(order=order,image=str(png.relative_to(ROOT)),sha256=source.base.sha(png),
                pixels=[width,height],mean_demand_GiB_s_per_card=avg_B,mean_supply_GiB_s_per_card=avg_b,
                received_GiB=physical_GiB,curve_segments=len(B),periods=periods,
                exactly_two_curves=True,area_matches_whole_window=True,visible_labels_inside_canvas=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache,data=source.verified_data()
    results=[build(order,data[order]) for order in ('random','ordered')]
    sources=dict(cache['sources']);sources.update(source.base.SOURCES)
    audit=dict(all_checks_passed=True,no_new_simulation=True,formats_created=['png'],
               num_npu=32,num_ssu=3,seed=7,window_ms=[2000.,4000.],sources=sources,
               builders={str(p.relative_to(ROOT)):source.base.sha(p) for p in (Path(__file__),Path(source.__file__),Path(means.__file__))},
               results=results)
    (OUT/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'README.md').write_text('''# 32 卡平均需求与供给：两条曲线

独立新增的PNG，Random、Ordered分开。32 NPU、3 SSU × 40 GiB/s、seed 7、warm [2,4) 秒。

- [Random PNG](random_mean_demand_supply.png)
- [Ordered PNG](ordered_mean_demand_supply.png)

横轴为仿真时间，纵轴为每卡平均带宽（GiB/s）。每张图只有两条曲线：

1. 需求：同一时刻32张卡各自当前请求的B_i=V/C相加，再除以32。
2. 供给：先把每张卡实际收到的数据量摊到自己的层周期（当前层开始计算到下一层开始计算，包含等待），再在同一时刻对32张卡求平均。

完整跨请求周期也按实际收到的字节数计算。窗口两端不完整的周期只使用窗内字节量除以窗内片段长度，避免漏计或额外计入窗外数据。因此新曲线面积可精确对齐原整窗统计。两图使用相同的坐标尺度。

蓝线不是原始瞬时吞吐，也不是存储承诺的可用带宽；不同卡的周期并不对齐。不要用它的局部峰值判断磁盘是否超速，也不要把两线之比当作瞬时或整窗NPU利用率。

所有旧PNG/PDF/SVG均保留，没有改写。

[Random曲线CSV](random_curves.csv) · [Ordered曲线CSV](ordered_curves.csv) · [来源与校验](checks.json)
''')
    print(json.dumps([{k:r[k] for k in ('order','image','pixels','mean_demand_GiB_s_per_card','mean_supply_GiB_s_per_card')} for r in results],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
