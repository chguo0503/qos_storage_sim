#!/usr/bin/env python3
"""PNG-only case figures for the new Random study; immutable logs only."""
from pathlib import Path
import argparse
import json
import math
import os
import tempfile
import numpy as np
import analyze as audit

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qos_random_near_capacity_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=audit.HERE
BLUE,PURPLE,GRAY='#0068d9','#a32b91','#e0e4e9'
INK,MUTED,STALL='#172d45','#546980','#e99925'
FONT=Path('/home/chguo/.fonts/msyh.ttc')
if not FONT.exists():FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False,'font.size':12})
COLS=['request_id','npu_id','layer','block_idx','ssu_id','path_id','size_gib','block_count',
      'enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms']


def role_label(role,analysis):
    if 'H' in analysis['roles'] and 'L' in analysis['roles']:
        if role=='H':return '高带宽需求请求（H）'
        if role=='L':return '低带宽需求请求（L）'
    if role in analysis['short_roles']:
        return '短请求' if len(analysis['short_roles'])==1 else f'短请求（{role}）'
    if role in analysis['long_roles']:
        return '长请求' if len(analysis['long_roles'])==1 else f'长请求（{role}）'
    return f'请求 {role}'


def arrival_prefix(rows,rate):
    rows=rows[np.argsort(rows[:,11],kind='stable')]
    starts,ends=rows[:,11],rows[:,12]
    assert np.all(starts[1:]>=ends[:-1]-1e-8)
    prefix=np.r_[0.,np.cumsum(rows[:,6])]
    def at(t):
        index=int(np.searchsorted(ends,t,side='right'));value=float(prefix[index])
        if index<len(rows) and starts[index]<t:value+=(t-starts[index])*rate/1000
        return value
    return at


def trace_evidence(case,man,raw,command,left,right):
    path=case/'trace.json.gz'
    if not path.exists():return None
    assert audit.sha(path)==command['trace_sha256']
    trace=audit.read(path)
    assert trace['columns']==COLS and trace['completed_simulation'] and all(trace['checks'].values())
    assert trace['window_ms'][0]<=left<right<=trace['window_ms'][1]
    assert trace['strategy']==raw['strategy']
    assert trace['source']['manifest_sha256']==audit.sha(case/'manifest.json.gz')
    assert trace['source']['reference_sha256']==audit.sha(case/'result.json.gz')
    rows=np.asarray(trace.pop('rows'),dtype=float)
    meta=man['metadata'];n_layers=meta['n_layers'];rate=float(meta['npu_bw_gib_s']);disk_rate=float(meta['disk_bw_gib_s'])
    assert np.allclose(rows[:,12]-rows[:,11],rows[:,6]*1000/rate,atol=1e-8)
    assert np.allclose(rows[:,10]-rows[:,9],rows[:,6]*1000/disk_rate,atol=1e-8)
    assert np.all(rows[:,7]==1)
    reqs={r['request_id']:r for r in man['requests']}
    keys=rows[:,0].astype(np.int64)*n_layers+rows[:,2].astype(np.int64)
    # Sorting identities avoids a hard-coded maximum block count.
    order=np.lexsort((rows[:,3],keys));sorted_keys=keys[order];sorted_blocks=rows[order,3]
    assert np.all((sorted_keys[1:]!=sorted_keys[:-1])|(sorted_blocks[1:]!=sorted_blocks[:-1]))
    unique,inv=np.unique(keys,return_inverse=True)
    count=np.bincount(inv);volume=np.bincount(inv,weights=rows[:,6])
    first=np.full(len(unique),np.inf);last=np.full(len(unique),-np.inf)
    lo=np.full(len(unique),np.inf);hi=np.full(len(unique),-np.inf)
    np.minimum.at(first,inv,rows[:,11]);np.maximum.at(last,inv,rows[:,12])
    np.minimum.at(lo,inv,rows[:,3]);np.maximum.at(hi,inv,rows[:,3])
    stats={int(k):dict(count=int(count[i]),volume=float(volume[i]),first=float(first[i]),last=float(last[i]),
                       min_block=int(lo[i]),max_block=int(hi[i])) for i,k in enumerate(unique)}
    for rid in np.unique(rows[:,0].astype(np.int64)):
        request=reqs[int(rid)]
        selected=rows[rows[:,0]==rid]
        assert np.all(selected[:,1]==request['npu_id'])
        placements=man['placements'][request['placement_index']]
        for layer in np.unique(selected[:,2].astype(int)):
            part=selected[selected[:,2]==layer]
            placement=np.asarray(placements[0 if len(placements)==1 else layer],float)
            idx=part[:,3].astype(int)
            assert np.all((idx>=0)&(idx<len(placement)))
            assert np.array_equal(part[:,4],placement[idx,0])
            assert np.array_equal(part[:,6],placement[idx,1])
    prefixes={n:arrival_prefix(rows[rows[:,1]==n],rate) for n in range(meta['num_npu'])}
    return dict(path=str(path),sha256=audit.sha(path),stats=stats,prefixes=prefixes,
                n_layers=n_layers,all_retained_block_identities_verified=True)


def prepare(case,left,right):
    analysis_path=case/'analysis.json';analysis=audit.read(analysis_path)
    assert analysis['all_technical_checks_passed']
    for name,digest in analysis['sources'].items():assert audit.sha(Path(name))==digest,name
    assert analysis['builder_sha256']==audit.sha(Path(audit.__file__))
    window=next(w for w in analysis['windows'] if w['start_ms']==left and w['end_ms']==right)
    man=audit.read(case/'manifest.json.gz');raw=audit.read(case/'result.json.gz');command=audit.read(case/'command.json')
    assert command['status']=='complete' and man['metadata']['order']=='random'
    reqs={r['request_id']:r for r in man['requests']};meta=man['metadata']
    trace=trace_evidence(case,man,raw,command,left,right)
    lanes=[];verified_cycles=0
    for npu in range(meta['num_npu']):
        batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
        flat=[];demands=[];states=[]
        for batch in batches:
            assert len(batch['member_request_ids'])==1
            rid=batch['member_request_ids'][0];q=reqs[rid]['load']
            for layer in batch['layer_metrics']:flat.append(dict(rid=rid,**layer))
            if audit.overlap(batch['admission_time_ms'],batch['completion_time_ms'],left,right)<=0:continue
            demands.append((batch['admission_time_ms'],batch['completion_time_ms'],q['per_layer_kv_gb']*1e6/q['per_layer_us']))
            previous=batch['admission_time_ms']
            for layer in batch['layer_metrics']:
                for a,z,role in ((previous,layer['compute_start_ms'],'stall'),(layer['compute_start_ms'],layer['compute_end_ms'],q['role'])):
                    a,z=max(left,a),min(right,z)
                    if z>a:states.append((a,z,role))
                previous=layer['compute_end_ms']
        cycles=[]
        for current,following in zip(flat,flat[1:]):
            a,z=current['compute_start_ms'],following['compute_start_ms']
            if audit.overlap(a,z,left,right)<=0:continue
            q=reqs[current['rid']]['load'];complete=left<=a<z<=right;same=current['rid']==following['rid']
            gray=not(same and complete);b=None
            if not gray:
                V=q['per_layer_kv_gb'];C=current['compute_end_ms']-a
                audit.close(following['io_start_time_ms'],a)
                audit.close(z,max(current['compute_end_ms'],following['io_ready_time_ms']))
                if trace:
                    stat=trace['stats'][following['rid']*trace['n_layers']+following['layer']]
                    placement=man['placements'][reqs[following['rid']]['placement_index']]
                    expected=placement[0 if len(placement)==1 else following['layer']]
                    assert stat['count']==len(expected) and stat['min_block']==0 and stat['max_block']==len(expected)-1
                    audit.close(stat['volume'],V)
                    assert stat['first']>=a-1e-7 and stat['last']<=z+1e-7
                    audit.close(stat['last'],following['io_ready_time_ms'])
                    received=trace['prefixes'][npu](z)-trace['prefixes'][npu](a)
                    audit.close(received,V);verified_cycles+=1
                else:received=V
                b=received*1000/(z-a)
                B=q['per_layer_kv_gb']*1e6/q['per_layer_us'];audit.close(b/B,C/(z-a))
            cycles.append(dict(start_ms=a,end_ms=z,gray=gray,mean_b_GiB_s=b,role=q['role']))
        p=window['per_npu'][npu]
        actual=math.fsum(z-a for a,z,role in states if role!='stall')
        audit.close(actual,p['compute_ms'])
        mean_B=math.fsum(audit.overlap(a,z,left,right)*B for a,z,B in demands)/(right-left)
        received_mean=None
        if window['physical']['available']:
            received_mean=window['physical']['per_npu_mean_received_GiB_s'][npu]
        if trace:
            trace_mean=(trace['prefixes'][npu](right)-trace['prefixes'][npu](left))*1000/(right-left)
            if received_mean is not None:audit.close(trace_mean,received_mean)
            received_mean=trace_mean
        lanes.append(dict(npu=npu,cycles=cycles,demands=demands,states=states,U_percent=p['U_percent'],
                          mean_B_GiB_s=mean_B,mean_received_GiB_s=received_mean))
    evidence=dict(trace_used=trace is not None,verified_complete_internal_cycles=verified_cycles,
        trace_path=trace['path'] if trace else None,trace_sha256=trace['sha256'] if trace else None,
        complete_cycles_b_source='Physical NPU arrival integrals verified against immutable next-layer V' if trace else 'V/D from complete internal layer work; no independent block trace available',
        analysis_path=str(analysis_path),analysis_sha256=audit.sha(analysis_path),all_checks_passed=True)
    return dict(case=case,analysis=analysis,window=window,metadata=meta,lanes=lanes,evidence=evidence,
                strategy=raw['strategy'],left=left,right=right)


def configuration(data):
    m=data['metadata'];candidate=m.get('candidate',m.get('label',data['case'].parent.name))
    strategy='Baseline' if data['strategy']=='baseline' else '流量分配策略'
    return f'{strategy} Random · {candidate}',f'{m["num_npu"]} NPU / {m["num_ssu"]} SSU × {m["disk_bw_gib_s"]:g} GiB/s · seed {m["seed"]} · warm [{data["left"]/1000:g},{data["right"]/1000:g}) 秒'


def save(fig,path):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(text.get_text(),box.bounds)
    fig.savefig(path,dpi=150);plt.close(fig)
    return dict(path=str(path),sha256=audit.sha(path),pixels=[width,height],visible_text_inside_canvas=True)


def fleet_bandwidth(data,out):
    title,config=configuration(data);window=data['window'];left,right=data['left'],data['right']
    max_B=max(p['B_GiB_s'] for p in data['analysis']['profiles'])
    fig,axes=plt.subplots(32,1,figsize=(18,32),dpi=150,sharex=True,sharey=True,facecolor='white')
    fig.subplots_adjust(left=.108,right=.848,top=.930,bottom=.054,hspace=.34)
    fig.text(.045,.986,title+'：32卡层周期平均带宽',fontsize=25,color=INK)
    fig.text(.045,.972,config+f' · 整机 U={window["U_percent"]:.2f}%',fontsize=15,color=MUTED)
    fig.legend(handles=[Line2D([],[],color=PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),
        Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='平均 b_i：完整内部层的 V/D'),
        Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线留空')],loc='upper left',bbox_to_anchor=(.043,.965),frameon=False,ncol=3,fontsize=13)
    evidence='周期接收量已与逐块日志核对' if data['evidence']['trace_used'] else '周期蓝线按V/D计算；本case没有逐块trace'
    fig.text(.045,.943,'每行一张卡；U、右侧供需均统计完整窗口。'+evidence+'。',fontsize=13,color=MUTED)
    fig.text(.865,.937,'整窗平均（GiB/s）',fontsize=12,color=INK)
    for npu,ax in enumerate(axes):
        lane=data['lanes'][npu];blue=[];purple=[];points=[];previous=None
        for c in lane['cycles']:
            a,z=max(left,c['start_ms'])/1000,min(right,c['end_ms'])/1000
            if c['gray']:ax.axvspan(a,z,color=GRAY,zorder=0);previous=None;continue
            b=c['mean_b_GiB_s'];blue.append([(a,b),(z,b)])
            if previous is not None and math.isclose(previous['end_ms']/1000,a,abs_tol=1e-9):blue.append([(a,previous['mean_b_GiB_s']),(a,b)])
            points.append(((a+z)/2,b));previous=c
        previous=None
        for a,z,B in lane['demands']:
            a,z=max(left,a)/1000,min(right,z)/1000;purple.append([(a,B),(z,B)])
            if previous is not None and math.isclose(previous[0],a,abs_tol=1e-9):purple.append([(a,previous[1]),(a,B)])
            previous=(z,B)
        ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.3,zorder=3))
        ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.8,linestyles='--',zorder=4))
        if points:
            x,y=zip(*points);ax.scatter(x,y,s=12,facecolors='white',edgecolors=BLUE,linewidths=1.1,zorder=5)
        ax.set(xlim=(left/1000,right/1000),ylim=(0,max_B*1.15),yticks=[0,max_B/2,max_B],
               yticklabels=['0',f'{max_B/2:.2f}',f'{max_B:.2f}'])
        ax.set_ylabel(f'NPU {npu:02d}\nU={lane["U_percent"]:.2f}%',fontsize=11,rotation=0,ha='right',va='center',labelpad=16,color=INK)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15,lw=.6)
        ax.tick_params(axis='y',labelsize=9,length=3);ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu in (7,15,23,31))
        ax.text(1.022,.70,f'需求 {lane["mean_B_GiB_s"]:.3f}',transform=ax.transAxes,ha='left',va='center',fontsize=11,color=PURPLE)
        supply='N/A' if lane['mean_received_GiB_s'] is None else f'{lane["mean_received_GiB_s"]:.3f}'
        ax.text(1.022,.28,'供给 '+supply,transform=ax.transAxes,ha='left',va='center',fontsize=11,color=BLUE)
    axes[-1].set_xlabel('时间（秒）',fontsize=13,labelpad=10)
    fig.text(.045,.025,'周期 D 从本层计算开始，到下一层计算开始，包含等待。灰区未填蓝线，但真实等待和接收仍计入整窗统计。',fontsize=12,color=MUTED)
    fig.text(.045,.013,'供给是整窗实际收到的数据量 / 窗口时长；它与整窗平均需求相除不等于 U。',fontsize=12,color=MUTED)
    return save(fig,out/f'{data["strategy"]}_random_all_32npu_layer_average.png')


def timeline(data,out):
    title,config=configuration(data);window=data['window'];left,right=data['left'],data['right']
    roles=data['analysis']['roles'];palette=['#52657c','#16866d','#49a7aa','#75b855','#8871bb']
    colors={role:palette[i%len(palette)] for i,role in enumerate(roles)};colors['stall']=STALL
    fig,ax=plt.subplots(figsize=(18,12),dpi=150,facecolor='white')
    extra_profiles=max(0,len(data['analysis']['profiles'])-2)
    fig.subplots_adjust(left=.07,right=.89,top=.80-.024*extra_profiles,bottom=.095)
    fig.text(.06,.961,title+'：32卡计算与IO等待',fontsize=24,color=INK)
    fig.text(.06,.925,config+f' · 整机 U={window["U_percent"]:.2f}%',fontsize=15,color=MUTED)
    labels=[]
    for p in sorted(data['analysis']['profiles'],key=lambda p:p['role']):
        labels.append(f'{role_label(p["role"],data["analysis"])}：总长{p["seq_len_k"]}K / miss {p["miss_tokens"]}；每层C={p["C_ms"]:.3f}ms，V={1024*p["V_GiB"]:.2f}MiB')
    for idx,text in enumerate(labels):fig.text(.06,.890-.024*idx,text,fontsize=12,color=MUTED)
    fig.legend(handles=[Patch(facecolor=colors[r],label=role_label(r,data['analysis'])) for r in roles]+[Patch(facecolor=STALL,label='IO 等待')],
               loc='upper left',bbox_to_anchor=(.057,.842-.024*extra_profiles),frameon=False,ncol=len(roles)+1,fontsize=13)
    for lane in data['lanes']:
        n=lane['npu'];ax.broken_barh([(left/1000,(right-left)/1000)],(n-.36,.72),facecolors='#f0f2f5',edgecolors='none')
        for role in roles+['stall']:
            intervals=[(a/1000,(z-a)/1000) for a,z,r in lane['states'] if r==role]
            ax.broken_barh(intervals,(n-.36,.72),facecolors=colors[role],edgecolors='none')
        ax.text(1.012,n,f'U={lane["U_percent"]:.2f}%',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=11,color=INK)
    ax.set(xlim=(left/1000,right/1000),ylim=(31.7,-.7),yticks=list(range(32)),
           yticklabels=[f'{n:02d}' for n in range(32)],xlabel='时间（秒）',ylabel='NPU 编号')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.text(.06,.038,'色块显示真实计算和IO等待；不同卡独立推进，没有层间同步屏障。请求内部每层计算时间固定，IO等待可能不同。',fontsize=12,color=MUTED)
    return save(fig,out/f'{data["strategy"]}_random_timeline.png')


def history(data,out):
    title,_=configuration(data);meta=data['metadata']
    rows=sorted((w for w in data['analysis']['windows'] if abs(w['duration_ms']-2000)<1e-7),key=lambda w:w['start_ms'])
    fig,ax=plt.subplots(figsize=(13.5,6),dpi=150,facecolor='white')
    fig.subplots_adjust(left=.085,right=.96,top=.78,bottom=.18)
    fig.text(.06,.936,title+'：连续两秒窗口利用率',fontsize=21,color=INK)
    fig.text(.06,.877,f'32 NPU / {meta["num_ssu"]} SSU × {meta["disk_bw_gib_s"]:g} GiB/s · seed {meta["seed"]}；每点使用同一32卡固定分母',fontsize=13,color=MUTED)
    x=[(w['start_ms']+w['end_ms'])/2000 for w in rows];y=[w['U_percent'] for w in rows]
    ax.plot(x,y,color=BLUE,marker='o',lw=2)
    for t,u in zip(x,y):ax.annotate(f'{u:.2f}%',(t,u),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10,color=INK)
    ax.set(ylim=(0,105),yticks=[0,20,40,60,80,100],ylabel='NPU 平均利用率（%）',xlabel='统计窗口（秒）',
           xticks=x,xticklabels=[f'[{w["start_ms"]/1000:g},{w["end_ms"]/1000:g})' for w in rows])
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    long=next((w for w in data['analysis']['windows'] if w['start_ms']==2000 and w['end_ms']==20000),None)
    if long:
        active='是' if long['all_32_active'] else '否'
        fig.text(.08,.045,f'[2,20) 秒整体 U={long["U_percent"]:.2f}%；32卡全程有请求：{active}；实际计算过所有类别的卡：{long["mixed_card_count"]}/32。',fontsize=12,color=MUTED)
    return save(fig,out/f'{data["strategy"]}_random_window_U.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,action='append',required=True)
    parser.add_argument('--window',type=audit.parse_window,default=(2000.,4000.))
    args=parser.parse_args()
    for case in args.case:
        case=case.resolve();assert case.is_relative_to(HERE)
        data=prepare(case,*args.window);out=case/'figures';out.mkdir(exist_ok=True)
        images=[fleet_bandwidth(data,out),timeline(data,out),history(data,out)]
        record=dict(all_checks_passed=True,no_new_simulation=True,formats_created=['png'],
            case=str(case),strategy=data['strategy'],window_ms=list(args.window),
            metadata_label=data['metadata']['label'],U_percent=data['window']['U_percent'],
            evidence=data['evidence'],builder_sha256=audit.sha(Path(__file__)),images=images)
        (out/'checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
