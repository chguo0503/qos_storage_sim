#!/usr/bin/env python3
"""Render two 32-card Once-per-layer bandwidth PNGs from immutable raw logs.

No simulator is started and no pre-existing study artifact or hash cache is changed.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'qos_once_fleet_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE/'figures'
LEFT, RIGHT, N, DISKS = 2000., 4000., 32, 3
WINDOW = RIGHT-LEFT
COLS = ['request_id','npu_id','layer','block_idx','ssu_id','path_id','size_gib',
        'block_count','enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms']
BLUE, PURPLE, GRAY = '#0068d9', '#a32b91', '#e0e4e9'
INK, MUTED = '#172d45', '#546980'
FONT_PATH = Path('/home/chguo/.fonts/msyh.ttc')
if not FONT_PATH.exists():
    FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.font_manager.fontManager.addfont(str(FONT_PATH))
plt.rcParams.update({'font.family':FontProperties(fname=str(FONT_PATH)).get_name(),
                     'axes.unicode_minus':False, 'font.size':12})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-9), (a,b)


def clip(a, z, left=LEFT, right=RIGHT):
    return max(0.,min(z,right)-max(a,left))


def read(path, sources):
    sources[str(path.relative_to(ROOT))] = sha(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:
        return json.load(stream)


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
    sources={}
    case=HERE/'runs'/order/'once'
    man=read(case/'manifest.json.gz',sources)
    raw=read(case/'result.json.gz',sources)
    command=read(case/'command.json',sources)
    trace=read(case/'trace.json.gz',sources)
    for name,key in [('manifest','manifest_sha256'),('result','output_sha256'),('trace','trace_sha256')]:
        assert sources[str((case/(name+'.json.gz')).relative_to(ROOT))]==command[key]
    assert raw['strategy']=='once' and trace['strategy']=='once'
    assert command['strategy']=='once' and command['status']=='complete'
    assert raw['submit_seed']==7 and all(raw['summary']['invariants'].values())
    assert trace['columns']==COLS and trace['completed_simulation'] and all(trace['checks'].values())
    assert trace['window_ms'][0]<=LEFT<RIGHT<=trace['window_ms'][1]
    assert man['input_fingerprint']==raw['input_fingerprint']==trace['source']['input_fingerprint']
    meta=man['metadata']
    assert meta['num_npu']==N and meta['num_ssu']==DISKS and meta['n_layers']==8
    assert meta['disk_bw_gib_s']==40 and meta['npu_bw_gib_s']==50
    assert meta['seed']==7 and meta['order']==order
    reqs={r['request_id']:r for r in man['requests']}
    rows=np.asarray(trace.pop('rows'),dtype=float)
    del trace
    assert rows.ndim==2 and rows.shape[1]==len(COLS)
    assert np.allclose(rows[:,12]-rows[:,11],rows[:,6]*1000/50,atol=1e-8)
    assert np.allclose(rows[:,10]-rows[:,9],rows[:,6]*1000/40,atol=1e-8)
    assert np.all((rows[:,4]>=0)&(rows[:,4]<DISKS))
    assert np.all((rows[:,1]>=0)&(rows[:,1]<N))
    assert np.all(rows[:,4]==(rows[:,3]+rows[:,1])%DISKS)
    assert np.all(rows[:,6]==176/(1024*1024))
    assert np.all(rows[:,7]==1)
    keys=rows[:,0].astype(np.int64)*8+rows[:,2].astype(np.int64)
    assert np.all((rows[:,2]>=0)&(rows[:,2]<8))
    assert np.all((rows[:,3]>=0)&(rows[:,3]<2048))
    identities=keys*2048+rows[:,3].astype(np.int64)
    assert len(np.unique(identities))==len(rows)
    unique,inv=np.unique(keys,return_inverse=True)
    counts=np.bincount(inv);volumes=np.bincount(inv,weights=rows[:,6])
    first=np.full(len(unique),np.inf);last=np.full(len(unique),-np.inf)
    first_block=np.full(len(unique),np.inf);last_block=np.full(len(unique),-np.inf)
    np.minimum.at(first,inv,rows[:,11]);np.maximum.at(last,inv,rows[:,12])
    np.minimum.at(first_block,inv,rows[:,3]);np.maximum.at(last_block,inv,rows[:,3])
    stats={int(key):(int(counts[j]),float(volumes[j]),float(first[j]),float(last[j]),
                    int(first_block[j]),int(last_block[j])) for j,key in enumerate(unique)}
    # Explicitly validate the immutable placement for every request retained in the trace.
    trace_rids,rid_inverse=np.unique(rows[:,0].astype(np.int64),return_inverse=True)
    expected_npus=[]
    for rid in trace_rids:
        request=reqs[int(rid)]
        placements=man['placements'][request['placement_index']]
        npu=int(request['npu_id']) if 'npu_id' in request else int(request['load']['npu_id'])
        expected_npus.append(npu)
        for placement in placements:
            assert placement and all(int(disk)==(idx+npu)%DISKS and size==176/(1024*1024)
                                     for idx,(disk,size) in enumerate(placement))
    assert np.array_equal(rows[:,1],np.asarray(expected_npus)[rid_inverse])
    for start,end,resource,count in ((9,10,4,DISKS),(11,12,1,N)):
        for number in range(count):
            intervals=rows[rows[:,resource]==number]
            intervals=intervals[np.argsort(intervals[:,start],kind='stable')]
            assert np.all(intervals[1:,start]>=intervals[:-1,end]-1e-8)
    receive_ms=np.maximum(0.,np.minimum(rows[:,12],RIGHT)-np.maximum(rows[:,11],LEFT))
    received=np.bincount(rows[:,1].astype(int),weights=receive_ms*50/1000,minlength=N)
    disk_ms=np.maximum(0.,np.minimum(rows[:,10],RIGHT)-np.maximum(rows[:,9],LEFT))
    disk_bytes=np.bincount(rows[:,4].astype(int),weights=disk_ms*40/1000,minlength=DISKS)
    assert np.all(disk_bytes*1000/WINDOW<=40+1e-7)
    lanes=[];all_cycles=[];per_npu=[]
    for npu in range(N):
        batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),
                       key=lambda b:b['admission_time_ms'])
        flat=[];demands=[]
        for batch in batches:
            assert len(batch['member_request_ids'])==1
            rid=batch['member_request_ids'][0];q=reqs[rid]['load']
            for layer in batch['layer_metrics']:flat.append(dict(rid=rid,**layer))
            if clip(batch['admission_time_ms'],batch['completion_time_ms'])>0:
                demands.append((batch['admission_time_ms'],batch['completion_time_ms'],
                                q['per_layer_kv_gb']*1e6/q['per_layer_us']))
        assert flat
        for a,z in zip(flat,flat[1:]):assert a['compute_end_ms']<=z['compute_start_ms']+1e-7
        direct_C=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for l in flat)
        close(raw['common_window']['start_ms'],LEFT);close(raw['common_window']['end_ms'],RIGHT)
        close(direct_C,raw['common_window']['compute_ms_by_npu'][npu])
        at=arrival_prefix(rows[rows[:,1]==npu]);close(at(RIGHT)-at(LEFT),received[npu])
        cycles=[]
        for current,following in zip(flat,flat[1:]):
            start,end=current['compute_start_ms'],following['compute_start_ms']
            if clip(start,end)<=0:continue
            deadline=current['compute_end_ms'];C=deadline-start;D=end-start
            assert D>0
            close(following['io_start_time_ms'],start)
            close(end,max(deadline,following['io_ready_time_ms']))
            same=current['rid']==following['rid'];complete=LEFT<=start<end<=RIGHT
            q=reqs[current['rid']]['load'];B=q['per_layer_kv_gb']*1000/C
            actual_C=clip(start,deadline);group=q['role'] if same and complete else 'gray'
            actual_b=ratio=measured=None
            if group!='gray':
                count,volume,first_received,last_received,min_block,max_block=stats[following['rid']*8+following['layer']]
                placements=man['placements'][reqs[following['rid']]['placement_index']]
                placement=placements[0 if len(placements)==1 else following['layer']]
                assert count==len(placement)
                assert min_block==0 and max_block==count-1
                close(volume,q['per_layer_kv_gb'])
                assert first_received>=start-1e-7 and last_received<=end+1e-7
                close(last_received,following['io_ready_time_ms'])
                measured=at(end)-at(start);close(measured,volume)
                actual_b=measured*1000/D;ratio=actual_b/B
                close(ratio,C/D);close(ratio*D,actual_C)
                assert 0<ratio<=1+1e-8
            cycles.append(dict(npu=npu,request_id=current['rid'],next_request_id=following['rid'],role=q['role'],
                compute_layer=current['layer']+1,read_layer=following['layer']+1,start_ms=start,end_ms=end,
                D_ms=D,C_ms=C,deadline_ms=deadline,clipped_start_ms=max(LEFT,start),clipped_end_ms=min(RIGHT,end),
                actual_window_compute_ms=actual_C,group=group,same_request=same,complete_cycle=complete,
                V_GiB=q['per_layer_kv_gb'],B_GiB_s=B,mean_b_GiB_s=actual_b,r=ratio,
                physically_received_GiB=measured))
        # Any exposed start/end region remains gray; its compute still contributes to U.
        covered_start=cycles[0]['clipped_start_ms'] if cycles else RIGHT
        covered_end=cycles[-1]['clipped_end_ms'] if cycles else LEFT
        edges=[(LEFT,covered_start)] if not cycles else [(LEFT,covered_start),(covered_end,RIGHT)]
        for a,z in edges:
            if z<=a:continue
            c=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms'],a,z) for l in flat)
            cycles.append(dict(npu=npu,request_id=None,next_request_id=None,role=None,compute_layer=None,
                read_layer=None,start_ms=a,end_ms=z,D_ms=z-a,C_ms=None,deadline_ms=None,
                clipped_start_ms=a,clipped_end_ms=z,actual_window_compute_ms=c,group='gray',
                same_request=False,complete_cycle=False,V_GiB=None,B_GiB_s=None,mean_b_GiB_s=None,
                r=None,physically_received_GiB=None))
        cycles.sort(key=lambda c:c['clipped_start_ms'])
        edge=LEFT
        for cycle in cycles:close(cycle['clipped_start_ms'],edge);edge=cycle['clipped_end_ms']
        close(edge,RIGHT)
        close(math.fsum(c['actual_window_compute_ms'] for c in cycles),direct_C)
        weighted=math.fsum(clip(a,z)*B for a,z,B in demands)
        row=dict(npu=npu,U_percent=100*direct_C/WINDOW,compute_ms=direct_C,
                 mean_demand_GiB_s=weighted/WINDOW,mean_supply_GiB_s=float(received[npu])*1000/WINDOW,
                 received_GiB=float(received[npu]),active_ms=math.fsum(clip(a,z) for a,z,_ in demands),
                 internal_cycle_count=sum(c['group']!='gray' for c in cycles),
                 gray_ms=math.fsum(c['clipped_end_ms']-c['clipped_start_ms'] for c in cycles if c['group']=='gray'))
        lanes.append(dict(npu=npu,cycles=cycles,demands=demands,**{k:v for k,v in row.items() if k!='npu'}))
        per_npu.append(row);all_cycles.extend(cycles)
    totals=dict(U_percent=math.fsum(r['U_percent'] for r in per_npu)/N,
        sum_mean_demand_GiB_s=math.fsum(r['mean_demand_GiB_s'] for r in per_npu),
        sum_mean_supply_GiB_s=math.fsum(r['mean_supply_GiB_s'] for r in per_npu),
        per_card_mean_demand_GiB_s=math.fsum(r['mean_demand_GiB_s'] for r in per_npu)/N,
        per_card_mean_supply_GiB_s=math.fsum(r['mean_supply_GiB_s'] for r in per_npu)/N,
        disk_mean_service_GiB_s=(disk_bytes*1000/WINDOW).tolist())
    close(totals['U_percent'],100*raw['common_window']['mean_npu_utilization'])
    return dict(order=order,strategy=raw['strategy'],seed=7,num_npu=N,num_ssu=DISKS,
        window_ms=[LEFT,RIGHT],totals=totals,per_npu=per_npu,lanes=lanes,cycles=all_cycles,sources=sources,
        input_fingerprint=raw['input_fingerprint'],all_checks_passed=True,
        physically_verified_internal_cycles=True,gray_regions_include_real_compute=True)


def bandwidth(ax,lane):
    blue=[];purple=[];points=[];previous=None
    for c in lane['cycles']:
        a,z=c['clipped_start_ms']/1000,c['clipped_end_ms']/1000
        if c['group']=='gray':
            ax.axvspan(a,z,color=GRAY,zorder=0);previous=None;continue
        b=c['mean_b_GiB_s'];blue.append([(a,b),(z,b)])
        if previous is not None and math.isclose(previous['end_ms']/1000,a,abs_tol=1e-9):
            blue.append([(a,previous['mean_b_GiB_s']),(a,b)])
        points.append(((a+z)/2,b));previous=c
    previous=None
    for a,z,B in lane['demands']:
        a,z=max(LEFT,a)/1000,min(RIGHT,z)/1000
        purple.append([(a,B),(z,B)])
        if previous is not None and math.isclose(previous[0],a,abs_tol=1e-9):
            purple.append([(a,previous[1]),(a,B)])
        previous=(z,B)
    ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.3,zorder=3))
    ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.8,linestyles='--',zorder=4))
    if points:
        px,py=zip(*points);ax.scatter(px,py,s=13,facecolors='white',edgecolors=BLUE,linewidths=1.1,zorder=5)
    ax.set(xlim=(2,4),ylim=(0,32.5),xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4])
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)


def draw(data):
    order=data['order'];totals=data['totals']
    fig,axes=plt.subplots(N,1,figsize=(18,32),dpi=150,sharex=True,sharey=True,facecolor='white')
    fig.subplots_adjust(left=.108,right=.848,top=.930,bottom=.054,hspace=.34)
    fig.text(.045,.986,f'Once per layer {order.title()}：32 张 NPU 的每层平均带宽与需求',fontsize=25,color=INK)
    fig.text(.045,.972,f'32 NPU / 3 SSU × 40 GiB/s · seed 7 · warm [2,4) 秒 · 整机 U={totals["U_percent"]:.2f}% · 平均每卡需求={totals["per_card_mean_demand_GiB_s"]:.3f}、供给={totals["per_card_mean_supply_GiB_s"]:.3f} GiB/s',fontsize=15,color=MUTED)
    handles=[Line2D([],[],color=PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),
             Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='平均 b_i：每个完整内部层周期一个值'),
             Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.043,.965),frameon=False,ncol=3,fontsize=13)
    fig.text(.045,.943,'每行一张卡；左侧 U 和右侧带宽均统计完整 warm [2,4) 秒。蓝线仍按各自层周期取均值。',fontsize=13,color=MUTED)
    fig.text(.865,.937,'整窗平均（GiB/s）',fontsize=12,color=INK)
    for npu,ax in enumerate(axes):
        lane=data['lanes'][npu];assert lane['npu']==npu
        bandwidth(ax,lane)
        ax.set_yticks([0,10,28.476],['0','10','28.48'])
        ax.set_ylabel(f'NPU {npu:02d}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=16,color=INK)
        ax.tick_params(axis='y',labelsize=9,length=3)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu in (7,15,23,31))
        ax.grid(axis='x',alpha=.15,lw=.6)
        ax.text(1.022,.70,f'需求 {lane["mean_demand_GiB_s"]:.3f}',transform=ax.transAxes,ha='left',va='center',fontsize=11,color=PURPLE)
        ax.text(1.022,.28,f'供给 {lane["mean_supply_GiB_s"]:.3f}',transform=ax.transAxes,ha='left',va='center',fontsize=11,color=BLUE)
    axes[-1].set_xlabel('仿真时间（秒）；所有行带宽单位均为 GiB/s',fontsize=13,labelpad=10)
    footer=fig.text(.045,.024,'周期 D = 当前层开始计算 → 下一层开始计算（含等待）；平均 b_i = 周期内实际收到的下一层数据量 / D。',fontsize=13,color=INK)
    fig.text(.045,.013,'右侧需求 = B_i 按时间加权；供给 = 2秒内实际收到的数据量 / 2秒（含灰区）。两个整窗均值相除不等于 U。',fontsize=12,color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    assert footer.get_window_extent(renderer).y1+6<axes[-1].xaxis.label.get_window_extent(renderer).y0
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            assert b.x0>=-2 and b.y0>=-2 and b.x1<=width+2 and b.y1<=height+2,(artist.get_text(),b.bounds)
    dest=OUT/f'{order}_all_32npu_layer_average.png'
    fig.savefig(dest,dpi=150);plt.close(fig)
    return dict(image=str(dest.relative_to(ROOT)),sha256=sha(dest),pixels=[width,height],
        shared_xlim_seconds=[2.,4.],shared_ylim_GiB_s=[0.,32.5],visible_labels_inside_canvas=True,
        whole_window_bandwidth_displayed=True,whole_window_bandwidth_includes_gray=True)


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders',nargs='+',choices=['random','ordered'],default=['random','ordered'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for order in args.orders:
        data=analyse(order);image=draw(data)
        with (OUT/f'cycles_{order}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data['cycles'][0]));writer.writeheader();writer.writerows(data['cycles'])
        write_json(OUT/f'cycles_{order}.json',data['cycles'])
        write_json(OUT/f'summary_{order}.json',{k:v for k,v in data.items() if k not in ('cycles','lanes')})
        checks={k:v for k,v in data.items() if k not in ('cycles','lanes')}
        checks.update(no_simulation_started_by_renderer=True,formats_created=['png'],
                      builder_sha256=sha(Path(__file__)),builder=str(Path(__file__).relative_to(ROOT)),
                      cycle_count=len(data['cycles']),image=image,
                      demand_definition='Time integral of current admitted request V/C over [2,4)s divided by 2s.',
                      supply_definition='Physical received GiB inside [2,4)s divided by 2s, including partial blocks and gray regions.',
                      internal_b_definition='Physical next-layer GiB received over a complete same-request compute-start-to-next-compute-start cycle divided by its full duration.',
                      cycle_ratio_note='b/B=C/D here is a same-cycle work accounting identity, not an independent allocation prediction.')
        write_json(OUT/f'{order}_all_32npu_layer_average.checks.json',checks)
        print(json.dumps(dict(order=order,**data['totals'],**image),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
