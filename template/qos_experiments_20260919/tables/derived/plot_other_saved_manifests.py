#!/usr/bin/env python3
"""Render saved manifest sequences. No simulation or sampling is performed."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch

ap=argparse.ArgumentParser()
ap.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'other_scenarios')
args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);root=args.workspace
sources={s:root/f'repository_source/results/diverse_data_ssu3_l3_20260916/inputs/{s}_seed7.json.gz' for s in ['full','semi']}
sources['near35']=root/'03_continuous_underload/near35/source/results/diverse_near35_20260916/inputs/near35_seed7.json.gz'
labels={'full':'Sustained overload','semi':'Partial overload / underload','near35':'Near 35 GiB/s per disk'}
cats=['SS','SL','LS','LL'];colors=['#b7ddcb','#63a3ca','#eab878','#ba7469']
def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
for scenario,source in sources.items():
    m=json.load(gzip.open(source));meta=m['metadata'];profiles=meta['profiles'];npu=meta['num_npu']
    byn=defaultdict(list)
    for q in m['requests']:byn[q['npu_id']].append(q)
    lens={len(x) for x in byn.values()};assert len(lens)==1;cols=lens.pop()
    arr=np.zeros((npu,cols),int);cls=np.zeros((npu,cols),int);seq=[]
    counts=Counter((q['load']['seq_len_k'],q['load']['nql']) for q in m['requests'])
    catcounts=Counter(q['load']['category'] for q in m['requests'])
    prows=[]
    for i,p in enumerate(profiles):
        count=counts[p['seq_len_k'],p['nql']]
        prows.append(dict(scenario=scenario,profile=f'P{i+1:02}',total_input_K=p['seq_len_k'],NQL=p['nql'],category=p['category'],count_per_NPU=count//npu,count_total=count,count_share_percent=count/len(m['requests'])*100,per_layer_read_MiB=p['per_layer_kv_gib']*1024,per_layer_compute_ms=p['per_layer_compute_us']/1000,required_bandwidth_GiB_s=p['required_bandwidth_gibps']))
    write(out/f'{scenario}_input_profile_ratios.csv',prows)
    write(out/f'{scenario}_input_category_ratios.csv',[dict(scenario=scenario,category=c,count_per_NPU=catcounts[c]//npu,count_total=catcounts[c],count_share_percent=100*catcounts[c]/len(m['requests'])) for c in cats])
    for n,reqs in byn.items():
        reqs.sort(key=lambda x:x['request_id'])
        for j,q in enumerate(reqs):
            p=q['load'];arr[n,j]=p['profile_index'];cls[n,j]=cats.index(p['category'])
            seq.append(dict(scenario=scenario,seed=7,npu=n,position_1based=j+1,request_id=q['request_id'],profile=f"P{p['profile_index']+1:02}",total_input_K=p['seq_len_k'],NQL=p['nql'],category=p['category']))
    write(out/f'{scenario}_original_manifest_seed7_sequence.csv',seq)
    fig,ax=plt.subplots(figsize=(19 if cols>35 else 16,13.5),layout='constrained')
    ax.imshow(cls,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,4.5),4),aspect='auto',interpolation='nearest')
    for n in range(npu):
        for j in range(cols):ax.text(j,n,str(arr[n,j]+1),ha='center',va='center',fontsize=7.5,color='#102631')
    ax.set(xticks=np.arange(cols),xticklabels=np.arange(1,cols+1),yticks=np.arange(npu),yticklabels=[f'NPU {n}' for n in range(npu)],xlabel='Request order on each NPU (not time)',title=f'{labels[scenario]} | original seed 7 manifest | {cols} requests per NPU')
    handles=[Patch(facecolor=colors[cats.index(p['category'])],label=f"{i+1}: {p['seq_len_k']}K / {p['nql']} ({p['category']})") for i,p in enumerate(profiles)]
    fig.legend(handles=handles,loc='outside lower center',ncol=6,frameon=False,fontsize=8.5,title='Profile number: total input / NQL (category)')
    fig.suptitle('New organization plot from saved manifest; no simulation rerun',fontsize=11,color='#555555')
    fig.savefig(out/f'NEW_{scenario}_seed7_profile_shuffle.png',dpi=175);plt.close(fig)
    print(scenario,'requests',len(m['requests']),'category count',dict(catcounts))
