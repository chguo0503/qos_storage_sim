#!/usr/bin/env python3
"""Recompute common-window accounts and render original event timelines.

No simulation is run. All plots use exact archived event timestamps; only
display clipping is applied. CSV files are machine-readable analysis exports.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PALETTE = ['#2463A0','#22A6A1','#D6A51D','#9A72B0','#508C46','#C87632','#596573']
CATCOL = {'SS':'#2463A0','SL':'#22A6A1','LS':'#D6A51D','LL':'#596573'}
STALL = '#DF5353'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})


def read_json(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if str(path).endswith('.gz') else open
    temporary = path.with_name(path.name + '.tmp')
    with op(temporary, 'wt', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=None if str(path).endswith('.gz') else 2)
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'wt', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def overlap(x, y, a, b):
    if x is None:
        return 0.0
    return max(0., min(b, b if y is None else y)-max(a, x))


def classify(seq, nql):
    return ('S' if seq <= 80 else 'L') + ('L' if nql >= 512 else 'S')


def npu_group_table(case, by, key):
    labels = sorted({r[key] for r in case['inputs']})
    out = []
    a, b = case['window_start_ms'], case['window_end_ms']
    for label in labels:
        cells = [by[(n, label)] for n in range(32)]
        active = math.fsum(c['active_ms'] for c in cells)
        comp = math.fsum(c['compute_ms'] for c in cells)
        present = [c['compute_ms']/c['active_ms'] for c in cells if c['active_ms'] > 1e-8]
        inputs = [r for r in case['inputs'] if r[key] == label]
        out.append(dict(label=label,input_count=len(inputs),warm_active_npus=len(present),warm_request_count=sum(c['request_count'] for c in cells),
                        warm_completed_count=sum(c['completed_count'] for c in cells),active_ms=active,compute_ms=comp,stall_ms=active-comp,
                        mean_per_npu_conditional_utilization=math.fsum(present)/len(present) if present else None,
                        pooled_conditional_utilization=comp/active if active else None,
                        occupancy_share=active/(32*(b-a)),fleet_compute_contribution=comp/(32*(b-a))))
    return out


def analyze(case):
    a,b=case['window_start_ms'],case['window_end_ms']; dur=b-a
    assert abs(dur-1000)<1e-7
    records=case['requests']
    for r in records:
        r.setdefault('first_layer_io_start_ms',r['layers'][0]['io_start_time_ms'])
    inputs=case.get('inputs', records);case['inputs']=inputs
    assert len({r['request_id'] for r in inputs}) == len(inputs)
    sums={key:defaultdict(lambda:dict(active_ms=0.,compute_ms=0.,request_count=0,completed_count=0)) for key in ('profile_id','category')}
    npu=[dict(npu_id=n,active_ms=0.,compute_ms=0.,warm_request_count=0,warm_completed_count=0) for n in range(32)]
    warm=[]; layer_export=[]; request_export=[]
    for r in records:
        assert r['category']==classify(r['seq_len_k'], r['nql'])
        n=r['npu_id']; act=overlap(r['admission_ms'],r['completion_ms'],a,b)
        comp=math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],a,b) for l in r['layers'])
        assert comp<=act+1e-6, (r['request_id'],comp,act)
        done=r['completion_ms'] is not None and a<=r['completion_ms']<b
        if act>1e-8:
            warm.append(r)
            row={k:r.get(k) for k in ('request_id','npu_id','input_order','profile_id','seq_len_k','nql','category','arrival_ms','admission_ms','completion_ms','layer_compute_ms','layer_kv_gib')}
            row.update(warm_active_ms=act,warm_compute_ms=comp,warm_stall_ms=act-comp,warm_conditional_utilization=comp/act)
            request_export.append(row)
            for l in r['layers']:
                layer_export.append(dict(request_id=r['request_id'],npu_id=n,profile_id=r['profile_id'],category=r['category'],**l,warm_compute_ms=overlap(l['compute_start_ms'],l['compute_end_ms'],a,b)))
        npu[n]['active_ms']+=act;npu[n]['compute_ms']+=comp;npu[n]['warm_request_count']+=act>1e-8;npu[n]['warm_completed_count']+=done
        for key in sums:
            cell=sums[key][(n,r[key])]
            cell['active_ms']+=act;cell['compute_ms']+=comp;cell['request_count']+=act>1e-8;cell['completed_count']+=done
    for row in npu:
        assert abs(row['active_ms']-dur)<1e-5, row
        row.update(stall_ms=row['active_ms']-row['compute_ms'],idle_ms=dur-row['active_ms'],utilization=row['compute_ms']/dur)
    per_npu_category=[];per_npu_profile=[]
    for key,target in [('category',per_npu_category),('profile_id',per_npu_profile)]:
        for n in range(32):
            for label in sorted({r[key] for r in inputs}):
                cell=sums[key][(n,label)]
                target.append(dict(npu_id=n,label=label,**cell,conditional_utilization=cell['compute_ms']/cell['active_ms'] if cell['active_ms']>1e-8 else None))
    allocation=[]
    for n in range(32):
        lane=[r for r in inputs if r['npu_id']==n]
        assert [r['input_order'] for r in lane]==list(range(len(lane))), (n,'order')
        allocation.append(dict(npu_id=n,input_count=len(lane),profile_counts=dict(Counter(r['profile_id'] for r in lane)),category_counts=dict(Counter(r['category'] for r in lane)),request_id_first=lane[0]['request_id'],request_id_last=lane[-1]['request_id'],profile_order=[r['profile_id'] for r in lane],first_six=[{k:r.get(k) for k in ('request_id','input_order','seq_len_k','nql','category','arrival_ms')} for r in lane[:6]],unique_pairs=len({(r['seq_len_k'],r['nql']) for r in lane})==len(lane)))
    profile_summary=npu_group_table(case,sums['profile_id'],'profile_id')
    category_summary=npu_group_table(case,sums['category'],'category')
    fleet=math.fsum(r['utilization'] for r in npu)/32
    assert abs(fleet-math.fsum(r['fleet_compute_contribution'] for r in profile_summary))<1e-10
    result=dict(case_id=case['case_id'],num_ssu=case['num_ssu'],window_start_ms=a,window_end_ms=b,input_count=len(inputs),observed_admitted_count=len(records),warm_overlapping_requests=len(warm),warm_completed_count=sum(r['warm_completed_count'] for r in npu),fleet_utilization=fleet,compute_ms=math.fsum(r['compute_ms'] for r in npu),stall_ms=32000-math.fsum(r['compute_ms'] for r in npu),all_npus_active=True,per_npu=npu,profile_summary=profile_summary,category_summary=category_summary,per_npu_category=per_npu_category,per_npu_profile=per_npu_profile,allocation=allocation)
    return result,warm,request_export,layer_export


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def timeline(case, stats, out, label):
    a,b=case['window_start_ms'],case['window_end_ms']
    profiles=sorted({r['profile_id'] for r in case['inputs']})
    colors={p:PALETTE[i] for i,p in enumerate(profiles)}
    # Keep intuitive short=blue / long=gray for fixed-role cases.
    if profiles==['L','S']:colors={'S':'#2463A0','L':'#596573'}
    fig,ax=plt.subplots(figsize=(17,12.3))
    polys=[];polycolors=[];end_x=[];end_y=[]
    for r in case['requests']:
        start=max(a,r['admission_ms']);end=min(b,r['completion_ms'] if r['completion_ms'] is not None else b)
        if end<=start:continue
        n=r['npu_id'];lo=n-.33;hi=n+.33
        polys.append([(start-a,lo),(end-a,lo),(end-a,hi),(start-a,hi)]);polycolors.append(STALL)
        for l in r['layers']:
            cs=l['compute_start_ms'];ce=l['compute_end_ms']
            if cs is None:continue
            x=max(a,cs);z=min(b,ce)
            if z>x:
                polys.append([(x-a,lo),(z-a,lo),(z-a,hi),(x-a,hi)]);polycolors.append(colors[r['profile_id']])
        if end-start>55:
            ax.text((start+end)/2-a,n,f"{r['profile_id']} / #{r['input_order']}",ha='center',va='center',fontsize=7.5,color='white',zorder=5)
        if r['completion_ms'] is not None and a<=r['completion_ms']<b:
            end_x.append(r['completion_ms']-a);end_y.append(n)
    ax.add_collection(PolyCollection(polys,facecolors=polycolors,edgecolors='none',rasterized=False))
    ax.scatter(end_x,end_y,marker='|',s=38,c='#111111',linewidths=.65,zorder=4)
    ax.set(xlim=(0,1000),ylim=(31.75,-.8),xlabel='Elapsed time in this warm window (ms)',ylabel='NPU ID')
    ax.set_xticks(np.arange(0,1001,100));ax.set_yticks(range(32));ax.grid(axis='x',color='#dfe3e8',linewidth=.5,zorder=0)
    for row in stats['per_npu']:
        ax.text(1.012,row['npu_id'],f"{100*row['utilization']:.2f}%",transform=ax.get_yaxis_transform(),va='center',fontsize=8)
    ax.text(1.012,-1.1,'NPU U',transform=ax.get_yaxis_transform(),fontsize=9,fontweight='bold')
    handles=[Patch(facecolor=colors[p],label=f'{p} compute') for p in profiles]+[Patch(facecolor=STALL,label='I/O stall'),Line2D([0],[0],color='#111111',marker='|',linestyle='None',label='Request completed')]
    ax.legend(handles=handles,ncol=min(5,len(handles)),loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,fontsize=8.5)
    fig.suptitle(f'{label} | Baseline | {case["num_ssu"]} SSU | Fleet U = {100*stats["fleet_utilization"]:.4f}%',x=.07,ha='left',fontsize=16,y=.992)
    fig.text(.07,.958,f'Absolute interval [{a:.6f}, {b:.6f}) ms; every NPU is active for all 1000 ms. Labels: profile / per-NPU input order.',fontsize=10,color='#465166')
    fig.subplots_adjust(top=.875,right=.91,left=.07,bottom=.065)
    save_figure(fig,out)


def heatmap(case, stats, out, key):
    rows=stats['per_npu_'+('profile' if key=='profile_id' else 'category')]
    labels=sorted({r['label'] for r in rows});values=np.full((32,len(labels)),np.nan)
    for r in rows:
        if r['conditional_utilization'] is not None:values[r['npu_id'],labels.index(r['label'])]=r['conditional_utilization']*100
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#edf0f3')
    fig,ax=plt.subplots(figsize=(max(7,1.4*len(labels)+2),12))
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(32));ax.set_ylabel('NPU ID')
    for n in range(32):
        for j in range(len(labels)):
            v=values[n,j];ax.text(j,n,'--' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=8,color='#718096' if np.isnan(v) else ('white' if v<60 else '#152235'))
    ax.set_title('Per-NPU conditional utilization by '+('profile' if key=='profile_id' else 'simulator category')+' (%)\nCompute / active time for that class within the SAME 1 s window',fontsize=12,pad=16)
    fig.colorbar(im,ax=ax,pad=.035,shrink=.65,label='Conditional utilization (%)')
    fig.text(.12,.025,'-- = this class did not occupy this NPU in the window; excluded from its per-NPU mean.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));save_figure(fig,out)


def input_matrix(case,out):
    profiles=sorted({r['profile_id'] for r in case['inputs']});lanes=[[r for r in case['inputs'] if r['npu_id']==n] for n in range(32)]
    assert len({len(l) for l in lanes})==1
    mat=np.array([[profiles.index(r['profile_id']) for r in lane] for lane in lanes])
    fig,ax=plt.subplots(figsize=(16,12))
    ax.imshow(mat,cmap=ListedColormap(PALETTE[:len(profiles)]),vmin=-.5,vmax=len(profiles)-.5,aspect='auto')
    for n,lane in enumerate(lanes):
        for j,r in enumerate(lane):ax.text(j,n,r['profile_id'],ha='center',va='center',fontsize=7.5,color='white')
    ax.set_xticks(range(mat.shape[1]));ax.set_yticks(range(32));ax.set(xlabel='Input order on each NPU (one equal-width cell = one request, NOT elapsed time)',ylabel='NPU ID')
    ax.set_title('Original mixed input | 22 requests on every NPU | Seven profiles, independently shuffled',pad=18,fontsize=14)
    fig.tight_layout();save_figure(fig,out)


def detail(case,stats,out):
    a,b=case['window_start_ms'],case['window_end_ms']
    candidates=[]
    for r in case['requests']:
        act=overlap(r['admission_ms'],r['completion_ms'],a,b)
        if act<=0 or r['category']!='SS':continue
        wait=act-math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],a,b) for l in r['layers'])
        candidates.append((wait,r))
    if not candidates:return None
    _,r=max(candidates,key=lambda x:x[0]);start=r['admission_ms'];end=r['completion_ms']
    if end is None:end=b
    fig,ax=plt.subplots(figsize=(15,6.3))
    for l in r['layers']:
        n=l['layer'];cs=l['compute_start_ms'];ce=l['compute_end_ms'];io=l['io_start_time_ms'];ready=l['io_ready_time_ms']
        if io is not None and ready is not None:ax.broken_barh([(io,ready-io)],(n-.34,.2),facecolors='#8995A6')
        prior=start if n==0 else r['layers'][n-1]['compute_end_ms']
        if cs is not None:
            if prior is not None and cs>prior:ax.broken_barh([(prior,cs-prior)],(n-.08,.27),facecolors=STALL)
            ax.broken_barh([(cs,ce-cs)],(n-.08,.27),facecolors='#2463A0')
            ax.text(ce+.003*(end-start),n+.035,f'{cs:.3f} - {ce:.3f}',fontsize=8,va='center')
    ax.axvspan(a,b,facecolor='#E4EEFF',alpha=.35,zorder=-2,label='Main warm window')
    ax.axvline(start, color='#263344', linestyle='--', linewidth=.8)
    ax.axvline(end, color='#263344', linestyle='--', linewidth=.8)
    ax.text(start,-.50,f'Admission {start:.3f}',fontsize=8,ha='left',color='#263344')
    ax.text(end,-.50,f'Done {end:.3f}',fontsize=8,ha='right',color='#263344')
    ax.set_xlim(min(start,min(l['io_start_time_ms'] for l in r['layers'] if l['io_start_time_ms'] is not None)),end+(end-start)*.20)
    ax.set_ylim(7.6,-.75);ax.set_yticks(range(8),[f'L{i}' for i in range(8)])
    ax.set_xlabel('Absolute simulation time (ms)');ax.grid(axis='x',alpha=.2)
    ax.set_title(f'Largest SS request stall overlap in the warm window | NPU {r["npu_id"]} | Request {r["request_id"]}\nProfile {r["profile_id"]}, input order {r["input_order"]}, {r["seq_len_k"]:.3f}K / NQL {r["nql"]}',loc='left',fontsize=12,pad=16)
    ax.legend(handles=[Patch(facecolor='#8995A6',label='I/O outstanding (queue + service + link)'),Patch(facecolor=STALL,label='NPU waits for I/O'),Patch(facecolor='#2463A0',label='NPU compute; labels are start - end')],loc='upper left',bbox_to_anchor=(0,-.15),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout();save_figure(fig,out)
    return {k:r[k] for k in ('request_id','npu_id','input_order','profile_id','category','admission_ms','completion_ms')}


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--key',required=True);p.add_argument('--title',required=True)
    args=p.parse_args();case=read_json(args.input);stats,warm,request_rows,layer_rows=analyze(case)
    root=args.output;data=root/'data'/args.key;images=root/'images';data.mkdir(parents=True,exist_ok=True);images.mkdir(parents=True,exist_ok=True)
    write_json(data/'statistics.json',stats)
    write_json(data/'normalized.json.gz',case)
    write_csv(data/'warm_requests.csv',request_rows);write_csv(data/'warm_layers.csv.gz',layer_rows)
    write_csv(data/'per_npu_utilization.csv',stats['per_npu']);write_csv(data/'per_npu_category.csv',stats['per_npu_category']);write_csv(data/'per_npu_profile.csv',stats['per_npu_profile'])
    write_csv(data/'profile_summary.csv',stats['profile_summary']);write_csv(data/'category_summary.csv',stats['category_summary'])
    write_csv(data/'all_inputs_and_times.csv.gz',[{k:v for k,v in r.items() if k!='layers'} for r in case['inputs']])
    timeline(case,stats,images/(args.key+'_timeline_1s'),args.title)
    heatmap(case,stats,images/(args.key+'_category_utilization'),'category')
    if args.key=='high':
        input_matrix(case,images/(args.key+'_input_order'))
        heatmap(case,stats,images/(args.key+'_profile_utilization'),'profile_id')
    selected=detail(case,stats,images/(args.key+'_request_detail'))
    write_json(data/'detail_selection.json',selected)
    print(json.dumps(dict(key=args.key,fleet_utilization=stats['fleet_utilization'],input_count=stats['input_count'],profile_summary=stats['profile_summary'],category_summary=stats['category_summary'],detail=selected),ensure_ascii=False))


if __name__=='__main__':main()
