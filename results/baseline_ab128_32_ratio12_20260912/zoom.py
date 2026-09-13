#!/usr/bin/env python3
"""Local causal zoom: one NPU, layer read deadlines, and physical SSD service."""
from pathlib import Path
import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import render


def merged(blocks, left, right, color_for):
    spans = []
    for block in blocks[np.argsort(blocks[:, 9], kind='stable')]:
        a, z = max(left, block[9]), min(right, block[10])
        if z <= a:
            continue
        color = color_for(block)
        if spans and color == spans[-1][2] and abs(a-spans[-1][1]) < 1e-8:
            spans[-1][1] = z
        else:
            spans.append([a, z, color])
    return spans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True); p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--role', choices=['A','B'], default='B'); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    example = render.read(args.examples)[args.role]
    npu, rid, target_layer = example['npu'], example['request_id'], example['layer']
    left, right = example['release_ms']-40, max(example['ready_ms'],example['deadline_ms'])+65
    man = render.read(args.case/'manifest.json.gz'); raw = render.read(args.case/'result.json.gz')
    trace = render.read(args.case/'trace.json.gz')
    assert trace['window_ms'][0] <= left < right <= trace['window_ms'][1]
    arr = np.asarray(trace['rows'], dtype=float)
    reqs = {r['request_id']:r for r in man['requests']}
    roles = {rid:render.role(r['load']) for rid,r in reqs.items()}
    lanes = sorted([b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu], key=lambda b:b['admission_time_ms'])
    timeline=[]; reads=[]; previous=None
    for b in lanes:
        r = b['member_request_ids'][0]; kind=roles[r]; wait_start=b['admission_time_ms']
        for l in b['layer_metrics']:
            cs,ce=l['compute_start_ms'],l['compute_end_ms']
            if ce>left and wait_start<right:
                timeline.append((wait_start,cs,'stall',r,l['layer']))
                timeline.append((cs,ce,kind,r,l['layer']))
            if previous is not None and l['io_ready_time_ms']>left and l['io_start_time_ms']<right:
                reads.append(dict(rid=r,kind=kind,layer=l['layer'],release=l['io_start_time_ms'],deadline=previous['compute_end_ms'],
                                  ready=l['io_ready_time_ms'],stall=l['io_barrier_wait_ms']))
            previous=l; wait_start=ce
    num_ssu=man['metadata']['num_ssu']; label=man['metadata']['order'].title()
    target_color='#c84976'
    fig,axes=plt.subplots(3,1,figsize=(15,10.5),sharex=True,gridspec_kw={'height_ratios':[1.2,max(1.8,len(reads)*.3),max(1.5,num_ssu*.35)]})
    fig.subplots_adjust(top=.82,bottom=.12,left=.11,right=.985,hspace=.55)
    fig.suptitle(f'Baseline {label} · {num_ssu} SSU · NPU {npu}：为什么发生等待',fontsize=20,y=.97)
    fig.text(.5,.924,render.DESCRIPTION,ha='center',fontsize=11)
    fig.text(.5,.884,f'目标 {args.role} 请求 L{target_layer}：预算 {example["deadline_ms"]-example["release_ms"]:.3f} ms；'
             f'读取 {example["payload_MiB"]:.3f} MiB；等待 {example["stall_ms"]:.3f} ms',ha='center',fontsize=12)
    ax=axes[0]
    for a,z,kind,r,k in timeline:
        a,z=max(a,left),min(z,right)
        if z<=a:continue
        ax.broken_barh([(a,z-a)],(.05,.6),facecolors=render.COLOR[kind],linewidth=0)
        if z-a>3:
            text=f'等待\n{z-a:.2f} ms' if kind=='stall' else f'{kind} L{k}\n{z-a:.2f} ms'
            ax.text((a+z)/2,.35,text,ha='center',va='center',fontsize=8)
    ax.set(ylim=(0,1),yticks=[],title=f'NPU {npu} 的计算与等待（边缘区间可能被裁剪）')
    ax=axes[1]
    for i,row in enumerate(reads):
        color=target_color if (row['rid'],row['layer'])==(rid,target_layer) else render.COLOR[row['kind']]
        ax.plot([row['release'],row['ready']],[i,i],color=color,lw=3)
        ax.scatter([row['release'],row['ready']],[i,i],c=color,s=22,zorder=3)
        ax.scatter([row['deadline']],[i],marker='|',c='#333',s=200,zorder=4)
        if row['stall']>1e-7:
            ax.plot([row['deadline'],row['ready']],[i+.13,i+.13],color=render.COLOR['stall'],lw=5)
        ax.text(min(right-1,row['ready']+1),i-.13,f'等 {row["stall"]:.2f} ms',fontsize=8,va='bottom',ha='left' if row['ready']<right-12 else 'right')
    ax.set(yticks=range(len(reads)),yticklabels=[f'{r["kind"]} L{r["layer"]}' for r in reads],ylim=(len(reads)-.5,-.7),
           title='本卡各层：圆点为读取释放/到齐，黑竖线为计算截止；线段包含排队，不是 SSD 持续服务')
    ax=axes[2]
    def color_for(block):
        return target_color if (int(block[0]),int(block[2]))==(rid,target_layer) else render.COLOR[roles[int(block[0])]]
    for disk in range(num_ssu):
        blocks=arr[(arr[:,4]==disk)&(arr[:,9]<right)&(arr[:,10]>left)]
        for a,z,color in merged(blocks,left,right,color_for):
            ax.broken_barh([(a,z-a)],(disk-.32,.64),facecolors=color,linewidth=0)
    ax.set(ylim=(num_ssu-.5,-.5),yticks=range(num_ssu),yticklabels=[f'SSU {d} Path0' for d in range(num_ssu)],
           xlabel='仿真时间（ms）',title='各盘真正服务的 IO 块：白色为空闲；粉色为目标层；其余颜色含其他 NPU 的请求')
    for ax in axes:
        ax.axvline(example['deadline_ms'],color='#b45e2d',ls='--',lw=.9)
        ax.axvline(example['ready_ms'],color=target_color,ls=':',lw=.9)
        ax.grid(axis='x',alpha=.12);ax.set_xlim(left,right)
    fig.legend(handles=[Patch(color=render.COLOR['A'],label='A 请求'),Patch(color=render.COLOR['B'],label='B 请求'),
                        Patch(color=render.COLOR['stall'],label='I/O 等待'),Patch(color=target_color,label='目标层 IO')],
               loc='lower center',bbox_to_anchor=(.5,.044),ncol=4,frameon=False)
    fig.text(.11,.022,'每盘物理服务速度 40 GiB/s；读取后还经过 NPU 的 50 GiB/s 接收链路，最后一块到达后才能继续计算。',fontsize=10)
    render.save(fig,args.out)
    args.out.with_suffix('.json').write_text(json.dumps(dict(window_ms=[left,right],example=example,reads=reads),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
