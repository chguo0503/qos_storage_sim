#!/usr/bin/env python3
"""Four-color timeline from complete existing logs; no simulation."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
from collections import defaultdict

HERE=Path(__file__).resolve().parent
BASE=HERE/'draw_timeline.py'
spec=importlib.util.spec_from_file_location('timeline_base',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter

COLORS={'fast':'#26749b','bridge':'#8c78ab','long':'#72b5a5','stall':'#efa640'}


def kind(req):
    q=req['load']
    if q['role']=='long':return 'long'
    if int(q['nql'])==4096:
        assert int(q['seq_len_k'])==32
        return 'bridge'
    assert int(q['nql'])==1024 and int(q['seq_len_k']) in (32,48,64)
    return 'fast'


def layout(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;outside=[]
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(renderer)
        if b.x0<bounds.x0-1 or b.y0<bounds.y0-1 or b.x1>bounds.x1+1 or b.y1>bounds.y1+1:outside.append(t.get_text())
    return dict(all_text_inside_canvas=not outside,outside_text=outside)


def collect(man,raw,left,right):
    _,audit=base.collect(man,raw,left,right)
    reqs={r['request_id']:r for r in man['requests']}
    spans=[{k:[] for k in COLORS} for _ in range(32)]
    sums={k:[0.]*32 for k in COLORS};lanes=defaultdict(list)
    def add(n,k,a,z):
        a,z=max(a,left),min(z,right)
        if z>a:spans[n][k].append((a,z-a));sums[k][n]+=z-a
    for b in raw['summary']['microbatch_metrics']:
        assert b['batch_size']==1
        n=b['npu_id'];r=reqs[b['member_request_ids'][0]];k=kind(r)
        assert n==r['npu_id'];lanes[n].append((b['admission_time_ms'],r['load']['role']))
        previous=b['admission_time_ms']
        for l in b['layer_metrics']:
            cs,ce=l['compute_start_ms'],l['compute_end_ms']
            assert math.isclose(cs-previous,l['io_barrier_wait_ms'],abs_tol=1e-6)
            add(n,'stall',previous,cs);add(n,k,cs,ce);previous=ce
    for n in range(32):
        assert math.isclose(sum(sums[k][n] for k in ('fast','bridge','long')),audit['C_ms'][n],abs_tol=1e-5)
        assert math.isclose(sums['fast'][n]+sums['bridge'][n],audit['role_C_ms']['short'][n],abs_tol=1e-5)
        assert math.isclose(sums['stall'][n],audit['stall_ms'][n],abs_tol=1e-5)
    switches=[]
    for n in range(32):
        lane=sorted(lanes[n]);switches.append(sum(left<=b[0]<right and a[1]!=b[1] for a,b in zip(lane,lane[1:])))
    audit.update(compute_ms_by_kind={k:sums[k] for k in ('fast','bridge','long')},
        role_switches=switches,both_fast_and_long_all_cards=all(sums['fast'][n]>0 and sums['long'][n]>0 for n in range(32)),
        min_fast_compute_fraction=min(sums['fast'][n]/audit['C_ms'][n] for n in range(32)),
        min_long_compute_fraction=min(sums['long'][n]/audit['C_ms'][n] for n in range(32)),
        bridge_fraction_of_fleet_compute=sum(sums['bridge'])/sum(audit['C_ms']),
        color_definition={'fast':'short role and NQL1024; total32/48/64K','bridge':'short role, total32K/NQL4096','long':'Long role, total176K/NQL1024','stall':'exposed admission-clock IO wait'})
    return spans,audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=float,default=2);p.add_argument('--end',type=float,default=60);p.add_argument('--caption',default='混合输入')
    args=p.parse_args();man,raw=base.read(args.manifest),base.read(args.result);left,right=args.start*1000,args.end*1000
    spans,audit=collect(man,raw,left,right)
    policy={'baseline':'Baseline','once':'Once per layer'}.get(raw['strategy'],raw['strategy'])
    fig,ax=plt.subplots(figsize=(16,11.8));fig.subplots_adjust(left=.067,right=.985,top=.766,bottom=.200)
    fig.suptitle(f'{args.caption} · {policy}：计算与读取等待',fontsize=23,y=.970)
    fig.text(.5,.919,f'32 NPU / 6 SSU · 每盘40 GiB/s · [{args.start:g}, {args.end:g})秒 · 平均利用率 {audit["U"]*100:.4f}%',ha='center',fontsize=16)
    fig.text(.5,.873,'短请求：32/48/64K、NQL1024；桥接请求：32K、NQL4096；长请求：176K、NQL1024',ha='center',fontsize=13)
    fig.text(.5,.839,'原始data画像，每请求8层；桥接层C=28.593ms，单独计色；全部真实计算均计入利用率',ha='center',fontsize=12.5,color='#444444')
    fig.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in [('fast','短请求'),('bridge','桥接请求'),('long','长请求'),('stall','接纳后 I/O 等待')]],loc='upper center',bbox_to_anchor=(.5,.817),ncol=4,frameon=False,fontsize=12.5)
    for n in range(32):
        for k in COLORS:ax.broken_barh(spans[n][k],(n-.4,.8),facecolors=COLORS[k],linewidth=0,edgecolors='none')
    ax.axhline(15.5,color='#666666',lw=.7,ls='--',alpha=.6)
    ax.set(xlim=(left,right),ylim=(31.8,-.8),yticks=range(32),ylabel='NPU 编号',xlabel='仿真时间（秒）')
    ticks=[args.start]+[v for v in (4,6,8,10,12) if args.start<v<args.end]+[args.end] if args.end<=12 else [args.start]+[v for v in range(10,61,10) if args.start<v<args.end]+[args.end]
    ax.set_xticks([v*1000 for v in ticks]);ax.tick_params(axis='y',labelsize=10);ax.grid(axis='x',alpha=.16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1000:g}'))
    fig.text(.067,.145,'全部32卡全窗有任务；橙色为接纳后的真实I/O等待，不含排队等接纳，也不含耗尽后的空闲。',fontsize=11.5)
    role='每卡均有短请求与长请求的计算' if audit['both_fast_and_long_all_cards'] else '本窗并非每卡都有短请求与长请求的计算'
    fig.text(.067,.108,f'{role}；最低短请求计算占比 {audit["min_fast_compute_fraction"]*100:.2f}%，最低长请求计算占比 {audit["min_long_compute_fraction"]*100:.2f}%。',fontsize=11.5)
    fig.text(.067,.071,f'每卡至少 {min(audit["role_switches"])} 次长/短角色切换；输入序列人工安排，无运行时同步屏障；下一请求L0预取仍完整执行。',fontsize=11)
    fig.text(.067,.034,f'seed {man["metadata"]["seed"]} · 输入指纹 {man["input_fingerprint"][:16]} · 图为完整日志真实区间；非硬件测量',fontsize=10.5,color='#59636c')
    audit['layout']=layout(fig);assert audit['layout']['all_text_inside_canvas'],audit['layout']
    args.out.parent.mkdir(parents=True,exist_ok=True);files=[]
    for ext in ('png','pdf','svg'):
        path=args.out.with_suffix('.'+ext);fig.savefig(path,dpi=175);files.append(dict(path=str(path),sha256=base.sha(path)))
    plt.close(fig)
    audit.update(manifest=str(args.manifest),result=str(args.result),manifest_sha256=base.sha(args.manifest),result_sha256=base.sha(args.result),
        script_sha256=base.sha(__file__),base_collect_source_sha256=base.sha(BASE),window_ms=[left,right],files=files)
    args.out.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.out),U=audit['U'],min_fast_compute_fraction=audit['min_fast_compute_fraction'],bridge_fleet_compute_fraction=audit['bridge_fraction_of_fleet_compute']),ensure_ascii=False))


if __name__=='__main__':main()
