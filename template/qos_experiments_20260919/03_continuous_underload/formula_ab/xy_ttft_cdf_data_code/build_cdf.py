#!/usr/bin/env python3
"""Extract matched native requests and plot un-smoothed empirical TTFT CDFs."""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parent
IDS=['XY12_32','XY12_24','XY12_20','XY12_16','X16']
SEEDS=[7,19,43]
COLORS={'baseline':'#315b83','once':'#e28536'}
LABELS={'baseline':'Baseline FIFO','once':'Once'}

def atomic_bytes(path, data):
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def write_csv(path, rows):
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    data=stream.getvalue().encode('utf-8-sig')
    atomic_bytes(path,gzip.compress(data,mtime=0) if path.suffix=='.gz' else data)

def save_figure(fig,path):
    from PIL import Image
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=220);data=stream.getvalue()
    Image.open(io.BytesIO(data)).load();atomic_bytes(path,data)
    assert path.read_bytes()==data

def extract(source):
    rows=[];configs=[];matched=[]
    for index,id in enumerate(IDS,1):
        c=json.loads((source/'configs'/f'{id}_random_s1_seed7.json').read_text());configs.append(c)
        for seed in SEEDS:
            pair={}
            for strategy in COLORS:
                folder=source/'results'/f'{id}_random_s1_seed{seed}'/strategy
                raw=(folder/'requests.json').read_bytes(); loads=json.loads(raw);meta={r['request_id']:r for r in loads}
                with gzip.open(folder/'native_summary.json.gz','rt') as f:s=json.load(f)
                assert all(s['invariants'].values())
                with (folder/'analysis.json').open() as f: audit=json.load(f)
                assert audit['full']['ordinary_demand']['strictly_under_capacity']
                pair[strategy]=(hashlib.sha256(raw).hexdigest(), audit['input_fingerprint'])
                for q in s['request_metrics']:
                    m=meta[q['request_id']];ttft=q['completion_time_ms']-q['admission_time_ms'];comp=q['own_compute_ms']
                    assert abs(ttft-q['processing_latency_ms'])<1e-7
                    assert ttft>=comp-1e-7
                    rows.append(dict(group=index,id=id,seed=seed,strategy=strategy,request_id=q['request_id'],npu_id=q['npu_id'],request_type=m['profile_group'],ttft_ms=ttft,compute_ms=comp,ttft_ratio=ttft/comp,arrival_ms=q['arrival_time_ms'],admission_ms=q['admission_time_ms'],completion_ms=q['completion_time_ms'],initial_npu_queue_ms=q['admission_wait_ms'],cold_start=m['generation']==0))
            assert pair['baseline']==pair['once']
            matched.append(dict(id=id,seed=seed,matched_request_manifest=True,matched_native_input_fingerprint=True))
    write_csv(ROOT/'cdf_requests.csv.gz',rows)
    (ROOT/'request_configs.json').write_text(json.dumps(configs,ensure_ascii=False,indent=2))
    (ROOT/'cdf_verification.json').write_text(json.dumps(dict(seeds=SEEDS,groups=IDS,all_requests=True,no_warm_filter=True,ttft_definition='completion minus NPU admission; initial NPU queue excluded',cdf='Empirical step CDF, no smoothing; per-request pooling of three seeds',checks=matched),indent=2))

def read_rows():
    with gzip.open(ROOT/'cdf_requests.csv.gz','rt',encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['group','seed','request_id','npu_id']:r[k]=int(r[k])
        for k in ['ttft_ms','compute_ms','ttft_ratio','arrival_ms','admission_ms','completion_ms','initial_npu_queue_ms']:r[k]=float(r[k])
    for id in IDS:
        pairs={s:{(r['seed'],r['request_id']):(r['request_type'],r['compute_ms']) for r in rows if r['id']==id and r['strategy']==s} for s in COLORS}
        assert pairs['baseline']==pairs['once']
    return rows

def configure_font():
    import fitz
    tmp=ROOT/'tmp';tmp.mkdir(exist_ok=True)
    p=tmp/'DroidSansFallback.ttf'
    if not p.exists():p.write_bytes(fitz.Font('china-s').buffer)
    font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(p)).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6,'savefig.facecolor':'white'})

def values(rows,id,strategy,scope,metric):
    return np.array([r[metric] for r in rows if r['id']==id and r['strategy']==strategy and (scope=='all' or r['request_type']==scope)])

def ecdf(v):
    x,n=np.unique(v,return_counts=True);return x,np.cumsum(n)/len(v)

def stats_and_curves(rows):
    stats=[];curves=[]
    for index,id in enumerate(IDS,1):
        for strategy in COLORS:
            for scope in ['all','A','B']:
                t=values(rows,id,strategy,scope,'ttft_ms');ratio=values(rows,id,strategy,scope,'ttft_ratio')
                qs=np.quantile(t,[.5,.9,.95,.99],method='inverted_cdf')
                stats.append(dict(group=index,id=id,strategy=strategy,scope=scope,n=len(t),mean_ms=float(np.mean(t)),p50_ms=qs[0],p90_ms=qs[1],p95_ms=qs[2],p99_ms=qs[3],max_ms=float(np.max(t)),slo15_pct=float(np.mean(ratio<=1.5+1e-10)*100)))
                for metric,v in [('ttft_ms',t),('ttft_ratio',ratio)]:
                    x,y=ecdf(v);assert np.all(np.diff(x)>=0) and np.all(np.diff(y)>0) and y[-1]==1
                    curves.extend(dict(group=index,id=id,strategy=strategy,scope=scope,metric=metric,x=float(a),cdf=float(b),n=len(v)) for a,b in zip(x,y))
    write_csv(ROOT/'ttft_cdf_statistics.csv',stats);write_csv(ROOT/'ttft_ecdf_points.csv.gz',curves)
    return stats

def draw(ax,rows,id,scope,metric,xlim,log=False):
    for strategy in COLORS:
        v=values(rows,id,strategy,scope,metric);x,y=ecdf(v)
        ax.step(np.r_[xlim[0],x,xlim[1]],np.r_[0,y,1],where='post',label=LABELS[strategy],color=COLORS[strategy],lw=1.6,linestyle='-' if strategy=='baseline' else '--')
    ax.set_xlim(*xlim);ax.set_ylim(0,1.02);ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('CDF')
    if log:
        ax.set_xscale('log');ax.set_xticks([40,60,100,200,400,800,1200]);ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter());ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(labelsize=9)

def config_label(c):
    a,b=c['profile_A'],c['profile_B'];x=a['read_gib']/b['read_gib'];y=a['compute_us']/b['compute_us']
    return f"A请求 {a['total_length_k']:g}K/{a['nql']}  |  B请求 {b['total_length_k']:g}K/{b['nql']}",x,y

def plots(rows,configs,stats):
    configure_font();figdir=ROOT/'figures';figdir.mkdir(exist_ok=True);paths=[]
    for normalized in [False,True]:
        fig,axes=plt.subplots(3,2,figsize=(8.27,11.69));fig.subplots_adjust(left=.10,right=.96,top=.83,bottom=.115,hspace=.58,wspace=.27)
        title='TTFT CDF：五组随机请求' if not normalized else 'TTFT / 纯计算时间：五组随机请求'
        fig.suptitle(title,fontsize=18,y=.97)
        fig.text(.5,.933,'8 NPU | 1 SSU × 40 GB/s | ring hash | seed 7, 19, 43',ha='center',fontsize=10)
        for i,(id,c) in enumerate(zip(IDS,configs)):
            ax=axes.flat[i];label,x,y=config_label(c)
            if normalized:lim=(.98,2.5);metric='ttft_ratio'
            else:lim=(30,1250);metric='ttft_ms'
            draw(ax,rows,id,'all',metric,lim,not normalized)
            if normalized:ax.axvline(1.5,color='#999999',lw=1,ls=':')
            a,b=c['profile_A'],c['profile_B']
            ax.set_title(f"组合{i+1}  x={x:.2f}, y={y:.2f}\nA {a['total_length_k']:g}K/{a['nql']}  |  B {b['total_length_k']:g}K/{b['nql']}",fontsize=10,pad=9)
            ax.set_xlabel('TTFT / 纯计算时间' if normalized else 'TTFT (ms，对数轴)')
        ax=axes.flat[5];ax.axis('off');table=[]
        for i,id in enumerate(IDS,1):
            ss={s:next(v for v in stats if v['id']==id and v['strategy']==s and v['scope']==('all' if normalized else 'B')) for s in COLORS}
            table.append([str(i)]+[f"{ss[s]['slo15_pct' if normalized else 'p95_ms']:.2f}" for s in COLORS])
        ax.text(.5,.96,'SLO ×1.5 达标率 (%)' if normalized else 'B请求 P95 TTFT (ms)',ha='center',fontsize=11,transform=ax.transAxes)
        t=ax.table(cellText=table,colLabels=['组合','FIFO','Once'],cellLoc='center',loc='center',bbox=[.02,.15,.96,.68]);t.auto_set_font_size(False);t.set_fontsize(10)
        for (ri,ci),cell in t.get_celld().items():
            cell.set_edgecolor('#d6e0e7')
            if ri==0:cell.set_facecolor('#e8f0f5')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.916),ncol=2,frameon=False,fontsize=11)
        fig.text(.5,.029,'TTFT = NPU接纳至完成；三个种子的全部相同请求合并，包含冷启动。\nCDF为经验阶梯曲线；补充图中竖线表示1.5倍纯计算时间。' if normalized else 'TTFT = NPU接纳至完成，不含初始NPU输入排队。\n三个种子的全部相同请求合并，包含冷启动；CDF为经验阶梯曲线。',ha='center',fontsize=9)
        name='ttft_ratio_cdf_overview.png' if normalized else 'ttft_cdf_overview.png';path=figdir/name;save_figure(fig,path);paths.append(path);plt.close(fig)
    detail=[]
    for i,(id,c) in enumerate(zip(IDS,configs),1):
        fig,axes=plt.subplots(3,1,figsize=(8.27,11.69));fig.subplots_adjust(left=.12,right=.95,top=.85,bottom=.115,hspace=.48)
        label,x,y=config_label(c);fig.suptitle(f'组合{i}：TTFT CDF',fontsize=19,y=.97);fig.text(.5,.933,label,ha='center',fontsize=11);fig.text(.5,.906,f'x={x:.3f} | y={y:.3f} | 8 NPU | 1 SSU | ring hash',ha='center',fontsize=10)
        for ax,scope,title in zip(axes,['all','A','B'],['全部请求','A请求','B请求']):
            v=np.concatenate([values(rows,id,s,scope,'ttft_ms') for s in COLORS]);lo,hi=float(v.min()),float(v.max());pad=max((hi-lo)*.07,lo*.003,.1)
            draw(ax,rows,id,scope,'ttft_ms',(max(0,lo-pad),hi+pad))
            n=len(values(rows,id,'baseline',scope,'ttft_ms'));ax.set_title(f'{title} | 每策略 n={n}',loc='left',fontsize=11);ax.set_xlabel('TTFT (ms)');ax.legend(loc='lower right',frameon=False)
        fig.text(.5,.025,'TTFT = NPU接纳至完成；seed 7、19、43全部请求合并，两个策略的请求集合一致。\nCDF未做平滑；三个子图横轴范围不同，分别展示全部、A和B请求。',ha='center',fontsize=9)
        path=figdir/f'ttft_cdf_{i:02d}_{id}.png';save_figure(fig,path);detail.append(path);plt.close(fig)
    return [paths[0],*detail,paths[1]]

def table_exports(configs,audit):
    result=[]
    for i,(c,a) in enumerate(zip(configs,audit['groups']),1):
        assert c['id']==a['id'];ba,bb=a['B_A_GBs'],a['B_B_GBs'];L=8*max(ba,bb);U=a['V_A_GB']/a['C_B_s']
        result.append(dict(group=i,id=c['id'],A_request=f"{a['A_total_length_k']:g}K/{a['A_nql']}",B_request=f"{a['B_total_length_k']:g}K/{a['B_nql']}",x=a['x'],y=a['y'],A_demand_GB_s=ba,B_demand_GB_s=bb,input_ideal_fleet_mean_GB_s=a['input_ideal_mean_fleet_GBs'],runtime_nominal_fleet_mean_FIFO_GB_s=a['full_ordinary_demand_mean_by_policy']['baseline'],runtime_nominal_fleet_mean_Once_GB_s=a['full_ordinary_demand_mean_by_policy']['once'],S_used=1,S_min=math.floor(L/40)+1,S_max=math.ceil(U/40)-1,both_conditions_satisfied=L<40<U,pooled_request_count=a['pooled_CDF_request_count_per_policy']))
    write_csv(ROOT/'request_bandwidth_and_S.csv',result);return result

def pdf_images(paths):
    out=ROOT/'xy_ttft_cdf.pdf';stream=io.BytesIO();pdf=canvas.Canvas(stream,pagesize=A4);pdf.setTitle('五组请求的TTFT CDF：FIFO与Once')
    for p in paths:
        pdf.drawImage(ImageReader(str(p)),0,0,width=A4[0],height=A4[1]);pdf.showPage()
    pdf.save();atomic_bytes(out,stream.getvalue())

def readme(table,stats):
    lines=['# S、x、y、带宽需求与TTFT CDF','S与x、y不是独立判据，而是同一组约束的两个检查步骤。x/y给出BA/BB，实际整数盘数还依赖BB的绝对大小。',
    '强约束下：BA=(x/y)BB；8×max(x/y,1) < 40S/BB < x。消去容量得到x>8且y>8，但固定40 GB/s的整数盘数仍须满足原区间。固定并发时左端改为nA×x/y+nB。',
    '## 请求与带宽表','单位为十进制GB/s。A/B需求为单层读取量/单层计算时间；实际均值为完整轨迹中当前请求V/C需求的时间平均，再对三个seed取简单均值。它不是SSD实际传输带宽。',
    '|组合|A请求|B请求|x|y|A需求|B需求|输入理想8卡均值|实际FIFO均值|实际Once均值|S|满足欠载+单A条件|',
    '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in table:lines.append('|'+ '|'.join([str(r['group']),r['A_request'],r['B_request'],f"{r['x']:.3f}",f"{r['y']:.3f}",f"{r['A_demand_GB_s']:.3f}",f"{r['B_demand_GB_s']:.3f}",f"{r['input_ideal_fleet_mean_GB_s']:.3f}",f"{r['runtime_nominal_fleet_mean_FIFO_GB_s']:.3f}",f"{r['runtime_nominal_fleet_mean_Once_GB_s']:.3f}",'1','是' if r['both_conditions_satisfied'] else '否'])+'|')
    lines.extend(['所有组均欠载，只有3/4/5组满足这里的理想单A阻塞条件。平均需求不能用于保证瞬时或逐盘欠载。',
    '输入数量比A:B=1:12，输入理想均值=8×(VA+12VB)/(CA+12CB)，只用于描述负载；它不替代实际并发的需求检查。前四组平均负载也在增大，因此不能把FIFO下降全部解释成隔离了负载影响后的纯x效应。',
    '16K/20K/24K计算时间为长度外推，NQL包含插值，不满足之前要求的全部NQL为512倍数。这里沿用既有实验记录，没有重新仿真或改变请求。',
    '## CDF定义',
    '- TTFT沿用本实验口径：completion_time_ms - admission_time_ms，即NPU接纳到完成，不包含t=0到接纳之间的初始输入队列等待。',
    '- 每组分别合并seed=7、19、43的全部完成请求，包含冷启动；基线和Once使用同一组(seed,request_id)集合。',
    '- 主CDF不是2-4秒窗口的样本，不能与此前warm利用率直接当成同一统计集合。',
    '- 所有CDF均为经验阶梯函数F(t)=数量(TTFT<=t)/总数量，未做平滑。所有CDF终点为1。',
    '- 分别给出全部请求、A请求、B请求。A:B数量为1:12，整体CDF约92.3%的请求为B；全体P95主要落在A上，未必反映B的改善。',
    '- 补充图绘制TTFT/该请求纯计算时间，1.5竖线对应SLO×1.5。这是归一化指标，不是毫秒TTFT。',
    '- 分位数采用经验分布逆函数(np.quantile method=inverted_cdf)，并非三个seed分位数的平均。',
    '## 文件',
    '- xy_ttft_cdf.pdf：7页，主概览、5组详细CDF（全部/A/B）、归一化补充概览。',
    '- figures/：7张PNG。主概览的毫秒轴为对数轴；每组详细图为线性轴。',
    '- request_bandwidth_and_S.csv：完整精度的请求带宽、平均需求和整数S判据。',
    '- ttft_cdf_statistics.csv：每组/策略/请求类型的数量、P50/P90/P95/P99、均值和SLO×1.5。',
    '- cdf_requests.csv.gz：画图使用的完整逐请求数据，两策略可按(group,seed,request_id)配对。',
    '- ttft_ecdf_points.csv.gz：全部CDF点；request_configs.json：原输入及插值/外推锚点。',
    '- cdf_verification.json和independent_bandwidth_audit.json：集合一致性与带宽独立核查。',
    '## 复现',
    '安装numpy、matplotlib、reportlab、PyMuPDF、Pillow，运行python build_cdf.py。使用包内逐请求CSV直接复现，不需重新仿真。要从完整仿真包重新提取，运行python build_cdf.py --source /path/to/xy_underload_experiments --extract。'])
    atomic_bytes(ROOT/'xy_ttft_cdf_notes.md', ('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n').encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT.parent/'xy_underload_experiments');ap.add_argument('--extract',action='store_true');a=ap.parse_args()
    if a.extract or not (ROOT/'cdf_requests.csv.gz').exists():extract(a.source)
    rows=read_rows();configs=json.loads((ROOT/'request_configs.json').read_text());stats=stats_and_curves(rows)
    audit=json.loads((ROOT/'independent_bandwidth_audit.json').read_text());table=table_exports(configs,audit)
    paths=plots(rows,configs,stats);pdf_images(paths);readme(table,stats)
    print(json.dumps(dict(request_rows=len(rows),groups=len(configs),pdf_pages=len(paths),pngs=len(paths),matched=True),ensure_ascii=False))
    for r in stats:
        if r['scope']=='B':print(json.dumps({k:r[k] for k in ['group','strategy','n','p95_ms','p99_ms','slo15_pct']}))
if __name__=='__main__':main()
