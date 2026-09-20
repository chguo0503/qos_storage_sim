#!/usr/bin/env python3
"""Render the reference study's 32-NPU layer-average bandwidth view.

Consumes frozen results and independently observed physical byte receipts.
This renderer never starts the simulator. Blue values are complete internal
layer-cycle accounting, not instantaneous link rate or allocation predictions.
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

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fifo_fleet_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/fifo_underload_exploration_20260914'
RECEIPTS=ROOT/'results/fifo_figure_receipts_20260914'
OUT=ROOT/'results/fifo_fleet_figures_20260914'
LEFT,RIGHT,N=2000.,4000.,32
BLUE,PURPLE,GRAY='#0068d9','#a32b91','#e0e4e9'
INK,MUTED='#172d45','#546980'
CASES=[
 ('formal/s3_L19_S13_seed7_fifo','01_ssu3_fifo_separated','Baseline FIFO','固定长短卡','19张长卡：200K/2048；13张短卡：32K/2048'),
 ('formal/s4_L20_S12_seed7_fifo','02_ssu4_fifo_separated','Baseline FIFO','固定长短卡','20张长卡：200K/2048；12张短卡：32K/1024'),
 ('formal/s3_L19_S13_seed7_short_first','03_ssu3_short_first_separated','短读取优先（诊断）','固定长短卡','相同输入与带宽，仅重排Path0已入队IO；19长卡＋13短卡'),
 ('formal/s4_L20_S12_seed7_short_first','04_ssu4_short_first_separated','短读取优先（诊断）','固定长短卡','相同输入与带宽，仅重排Path0已入队IO；20长卡＋12短卡'),
 ('tiny_formal/tiny_jitter_s3_L19_S13_seed7_baseline_fifo','05_ssu3_fifo_unique_jitter','Baseline FIFO','固定角色、小幅随机','19长卡＋13短卡；每卡长度/NQL组合不重复；计算时间由data内插'),
 ('tiny_formal/tiny_jitter_s3_L19_S13_seed7_baseline_short_first','06_ssu3_short_first_unique_jitter','短读取优先（诊断）','固定角色、小幅随机','相同不重复输入；仅重排Path0已入队IO'),
 ('mixed_formal/mixed20n1536_s3_L1_S8_seed7_baseline','07_ssu3_fifo_random_mixed','Baseline FIFO','全卡随机混排','每卡200K/2048 : 20K/1536 = 1:8；20K计算时间含外推'),
 ('mixed_formal/mixed24_s4_L1_S4_seed7_baseline','08_ssu4_fifo_random_mixed','Baseline FIFO','全卡随机混排','每卡200K/2048 : 24K/1024 = 1:4；24K计算时间含外推'),
]

def read(path):
 with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a,b,context=''):
 assert math.isclose(a,b,abs_tol=1e-7,rel_tol=1e-9),(context,a,b)

def clip(a,b,left=LEFT,right=RIGHT):return max(0.,min(b,right)-max(a,left))

def write_json(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def analyse(case):
 relative,stem,policy,scope,description=case
 folder=DATA/relative
 raw=read(folder/'result.json.gz');man=read(folder/'manifest.json.gz');metric=read(folder/'metrics.json')
 receipt_path=RECEIPTS/folder.name/'receipts.json';physical=read(receipt_path)
 assert all(physical['replay_checks'].values())
 assert physical['window_ms']==[LEFT,RIGHT]
 assert physical['source_result_sha256']==sha(folder/'result.json.gz')
 assert physical['source_manifest_sha256']==sha(folder/'manifest.json.gz')
 assert all(raw['summary']['invariants'].values()) and metric['all_active']
 assert metric['window_ms']==[LEFT,RIGHT]
 requests={q['request_id']:q for q in man['requests']}
 # Every internal layer's physical byte count is independently observed on replay.
 layer_stats={(int(q['request_id']),int(q['layer'])):q for q in physical['layers']}
 received=physical['window_npu_received_gib'];disk_bytes=physical['window_ssu_read_gib']
 assert len(received)==N
 lanes=[];all_cycles=[];completion_rows=[]
 for npu in range(N):
  batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
  flat=[];demands=[];roles=set()
  for batch in batches:
   assert len(batch['member_request_ids'])==1
   rid=batch['member_request_ids'][0];q=requests[rid]['load'];roles.add(q['role'])
   flat.extend(dict(request_id=rid,**layer) for layer in batch['layer_metrics'])
   if clip(batch['admission_time_ms'],batch['completion_time_ms'])>0:
    demands.append((batch['admission_time_ms'],batch['completion_time_ms'],q['per_layer_kv_gb']*1e6/q['per_layer_us']))
   if LEFT<=batch['completion_time_ms']<RIGHT:
    completion_rows.append(dict(npu_id=npu,request_id=rid,role=q['role'],completion_ms=batch['completion_time_ms']))
  direct_c=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for l in flat)
  close(direct_c,raw['windows'][0]['compute_ms_by_npu'][npu],'window compute')
  close(math.fsum(clip(a,b) for a,b,_ in demands),RIGHT-LEFT,'whole-window active')
  cycles=[]
  for current,following in zip(flat,flat[1:]):
   start,end=current['compute_start_ms'],following['compute_start_ms']
   if clip(start,end)<=0:continue
   deadline=current['compute_end_ms'];compute=deadline-start;duration=end-start
   close(following['io_start_time_ms'],start,'next read begins with compute')
   close(end,max(deadline,following['io_ready_time_ms']),'barrier')
   same=current['request_id']==following['request_id'];complete=LEFT<=start<end<=RIGHT
   q=requests[current['request_id']]['load'];demand=q['per_layer_kv_gb']*1000/compute
   group=q['role'] if same and complete else 'gray'
   actual=ratio=volume=None
   if group!='gray':
    observed=layer_stats[following['request_id'],following['layer']]
    volume=observed['bytes_gib'];close(volume,q['per_layer_kv_gb'],'observed layer bytes')
    placement=man['placements'][requests[following['request_id']]['placement_index']]
    assert observed['completed_blocks']==len(placement[0 if len(placement)==1 else following['layer']])
    assert observed['first_link_start_ms']>=start-1e-7
    close(observed['last_link_end_ms'],following['io_ready_time_ms'],'physical last byte')
    assert observed['last_link_end_ms']<=end+1e-7
    actual=volume*1000/duration;ratio=actual/demand
    close(ratio,compute/duration,'same-cycle ratio')
   cycles.append(dict(npu=npu,request_id=current['request_id'],next_request_id=following['request_id'],
    role=q['role'],compute_layer=current['layer'],read_layer=following['layer'],
    start_ms=start,end_ms=end,C_ms=compute,T_ms=duration,deadline_ms=deadline,
    clipped_start_ms=max(LEFT,start),clipped_end_ms=min(RIGHT,end),
    actual_window_compute_ms=clip(start,deadline),group=group,
    B_GiB_s=demand,mean_b_GiB_s=actual,received_GiB=volume,cycle_ratio=ratio))
  edge_start=cycles[0]['clipped_start_ms'] if cycles else RIGHT
  edge_end=cycles[-1]['clipped_end_ms'] if cycles else LEFT
  for a,b in [(LEFT,edge_start),(edge_end,RIGHT)]:
   if b>a:
    cycles.append(dict(npu=npu,request_id=None,next_request_id=None,role=None,compute_layer=None,read_layer=None,
     start_ms=a,end_ms=b,C_ms=None,T_ms=b-a,deadline_ms=None,clipped_start_ms=a,clipped_end_ms=b,
     actual_window_compute_ms=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms'],a,b) for l in flat),
     group='gray',B_GiB_s=None,mean_b_GiB_s=None,received_GiB=None,cycle_ratio=None))
  cycles.sort(key=lambda c:c['clipped_start_ms']);cursor=LEFT
  for c in cycles:close(c['clipped_start_ms'],cursor,'cycle partition');cursor=c['clipped_end_ms']
  close(cursor,RIGHT);close(math.fsum(c['actual_window_compute_ms'] for c in cycles),direct_c)
  lanes.append(dict(npu=npu,role=''.join(sorted(roles)),U_percent=100*direct_c/(RIGHT-LEFT),
    compute_ms=direct_c,mean_demand_GiB_s=math.fsum(clip(a,b)*d for a,b,d in demands)/(RIGHT-LEFT),
    mean_supply_GiB_s=received[npu]*1000/(RIGHT-LEFT),received_GiB=received[npu],
    gray_ms=math.fsum(c['clipped_end_ms']-c['clipped_start_ms'] for c in cycles if c['group']=='gray'),
    cycles=cycles,demands=demands))
  all_cycles.extend(cycles)
 close(math.fsum(l['U_percent'] for l in lanes)/N,metric['U_percent'],'fleet U')
 totals=dict(U_percent=metric['U_percent'],short_U_percent=metric['short_U_percent'],
  slo_1p5_percent=metric['slo_1p5_percent'],
  per_card_mean_demand_GiB_s=math.fsum(l['mean_demand_GiB_s'] for l in lanes)/N,
  per_card_mean_supply_GiB_s=math.fsum(l['mean_supply_GiB_s'] for l in lanes)/N,
  disk_mean_service_GiB_s=[v*1000/(RIGHT-LEFT) for v in disk_bytes])
 return dict(stem=stem,policy=policy,scope=scope,description=description,source=relative,
  num_ssu=metric['num_ssu'],totals=totals,lanes=lanes,cycles=all_cycles,completion_rows=completion_rows,
  window_nominal_underloaded=metric['demand_audit']['underloaded_every_admission_interval'],
  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [folder/'result.json.gz',folder/'manifest.json.gz',receipt_path]},
  all_checks_passed=True)

def bandwidth(ax,lane):
 blue=[];purple=[];points=[];previous=None
 for c in lane['cycles']:
  a,b=c['clipped_start_ms']/1000,c['clipped_end_ms']/1000
  if c['group']=='gray':ax.axvspan(a,b,color=GRAY,zorder=0);previous=None;continue
  v=c['mean_b_GiB_s'];blue.append([(a,v),(b,v)])
  if previous is not None and math.isclose(previous['end_ms']/1000,a,abs_tol=1e-9):
   blue.append([(a,previous['mean_b_GiB_s']),(a,v)])
  points.append(((a+b)/2,v));previous=c
 previous=None
 for a,b,v in lane['demands']:
  a,b=max(LEFT,a)/1000,min(RIGHT,b)/1000;purple.append([(a,v),(b,v)])
  if previous is not None and math.isclose(previous[0],a,abs_tol=1e-9):purple.append([(a,previous[1]),(a,v)])
  previous=(b,v)
 ax.add_collection(LineCollection(blue,colors=BLUE,linewidths=2.3,zorder=3))
 ax.add_collection(LineCollection(purple,colors=PURPLE,linewidths=2.5,linestyles='--',zorder=4))
 if points:
  x,y=zip(*points);ax.scatter(x,y,s=13,facecolors='white',edgecolors=BLUE,linewidths=1.1,zorder=5)
 ax.set(xlim=(2,4),ylim=(0,6.5),xticks=[2,2.25,2.5,2.75,3,3.25,3.5,3.75,4],yticks=[0,3,6])
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='both',alpha=.14,lw=.6)

def draw(data,output,selected=None):
 ids=list(range(N)) if selected is None else selected
 short=len(ids)<N
 fig,axes=plt.subplots(len(ids),1,figsize=(18,12 if short else 32),dpi=150,sharex=True,sharey=True,facecolor='white')
 fig.subplots_adjust(left=.115,right=.847,top=.790 if short else .917,bottom=.145 if short else .065,hspace=.35)
 suffix='（选取8张卡放大）' if short else '（全部32张卡）'
 fig.text(.035,.984 if not short else .964,f'{data["policy"]} · {data["scope"]}：每层平均带宽与需求{suffix}',fontsize=23,color=INK)
 t=data['totals']
 fig.text(.035,.970 if not short else .932,f'32 NPU / {data["num_ssu"]} SSU × 40 GiB/s · ring hash · seed 7 · warm [2,4)秒 · 整机U={t["U_percent"]:.2f}% · TTFT SLO×1.5={t["slo_1p5_percent"]:.2f}%',fontsize=14,color=MUTED)
 fig.text(.035,.956 if not short else .900,data['description']+'；长度/NQL中的长度含新增token，K=1024。',fontsize=13,color=MUTED)
 handles=[Line2D([],[],color=PURPLE,lw=2.5,ls='--',label='需求 B：当前请求每层读取量 V / 计算时间 C'),
  Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='供给 b：完整内部层周期平均带宽'),
  Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')]
 fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.031,.951 if not short else .876),frameon=False,ncol=3,fontsize=11.5)
 fig.text(.035,.929 if not short else .819,'每行一张卡；左右数字统计完整[2,4)秒。统一纵轴0–6.5 GiB/s；灰区仍计入整窗统计。',fontsize=12,color=MUTED)
 fig.text(.864,.925 if not short else .811,'整窗平均（GiB/s）',fontsize=11,color=INK)
 for row,(npu,ax) in enumerate(zip(ids,axes)):
  lane=data['lanes'][npu];bandwidth(ax,lane)
  role='混排' if len(lane['role'])>1 else lane['role']
  ax.set_ylabel(f'NPU {npu:02d} · {role}\nU={lane["U_percent"]:.2f}%',fontsize=10.5,rotation=0,ha='right',va='center',labelpad=15,color=INK)
  ax.tick_params(axis='y',labelsize=9,length=3)
  ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=row==len(ids)-1 or (not short and row in (7,15,23)))
  ax.text(1.022,.70,f'需求 {lane["mean_demand_GiB_s"]:.3f}',transform=ax.transAxes,ha='left',va='center',fontsize=11,color=PURPLE)
  ax.text(1.022,.28,f'供给 {lane["mean_supply_GiB_s"]:.3f}',transform=ax.transAxes,ha='left',va='center',fontsize=11,color=BLUE)
 axes[-1].set_xlabel('仿真时间（秒）；所有行带宽单位均为 GiB/s',fontsize=13,labelpad=10)
 fig.text(.035,.036 if not short else .078,'周期 T = 当前层开始计算 → 下一层开始计算（包含IO等待）；b = 周期内实际收到的下一层字节量 / T。',fontsize=12,color=INK)
 fig.text(.035,.024 if not short else .049,'右侧需求按时间加权；供给=2秒内实际收到的字节量/2秒（含灰区、部分IO）。两个整窗带宽均值相除不等于U。',fontsize=11.5,color=MUTED)
 note='窗口内名义需求逐盘低于40 GiB/s；这允许IO突发排队。' if data['window_nominal_underloaded'] else '本组窗口存在逐盘名义需求超过40 GiB/s的区间，不作为逐盘欠载反例。'
 fig.text(.035,.013 if not short else .021,note+'  短读取优先仅为诊断对照，不是Once策略。',fontsize=11,color=MUTED)
 fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
 overflow=[]
 for artist in fig.findobj(matplotlib.text.Text):
  if artist.get_visible() and artist.get_text():
   b=artist.get_window_extent(renderer)
   if b.x0 < -2 or b.y0 < -2 or b.x1>width+2 or b.y1>height+2:overflow.append((artist.get_text(),b.bounds))
 assert not overflow,overflow
 fig.savefig(output,dpi=150);plt.close(fig)
 return dict(file=output.name,pixels=[width,height],visible_labels_inside_canvas=True,sha256=sha(output),selected_npus=ids)

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=int,nargs='+',default=list(range(len(CASES))))
 p.add_argument('--font',type=Path,required=True);p.add_argument('--out',type=Path,default=OUT)
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 matplotlib.font_manager.fontManager.addfont(str(a.font))
 plt.rcParams.update({'font.family':FontProperties(fname=str(a.font)).get_name(),'axes.unicode_minus':False,'font.size':12})
 for i in a.cases:
  data=analyse(CASES[i]);stem=data['stem'];image=draw(data,a.out/(stem+'_all_32npu_layer_average.png'))
  if i in (0,1):
   selected=[0,8,16,18,19,21,25,31] if i==0 else [0,8,16,19,20,23,27,31]
   draw(data,a.out/(stem+'_8npu_detail.png'),selected)
  for name,rows in [('cycles',data['cycles']),('request_completions',data['completion_rows'])]:
   with (a.out/(stem+'_'+name+'.csv')).open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  summary={k:v for k,v in data.items() if k not in ['cycles','lanes','completion_rows']}
  summary['per_npu']=[{k:v for k,v in l.items() if k not in ['cycles','demands']} for l in data['lanes']]
  summary['image']=image;summary['renderer_sha256']=sha(Path(__file__))
  write_json(a.out/(stem+'_summary.json'),summary)
  print(json.dumps(dict(case=i,**data['totals'],**image),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
