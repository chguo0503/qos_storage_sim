#!/usr/bin/env python3
"""Replot saved, paired requests as an empirical CDF of TTFT / base SLO."""
from pathlib import Path
import csv, gzip, io, json, os, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from PIL import Image
import fitz
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
IDS = ['XY12_32','XY12_24','XY12_20','XY12_16','X16']
STYLES = {'baseline': ('Baseline FIFO','#315b83','-'), 'once': ('Once','#e28536','--')}

def write(path, data):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)

def csv_bytes(rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return s.getvalue().encode('utf-8-sig')

def vals(rows, group, strategy, scope='all'):
    return np.array([r['ttft_ratio'] for r in rows if r['id']==group and r['strategy']==strategy and (scope=='all' or r['request_type']==scope)])

def panel(ax, rows, group, scope='all'):
    for strategy, (label,color,style) in STYLES.items():
        v = vals(rows,group,strategy,scope)
        xx, counts = np.unique(v, return_counts=True)
        yy = np.cumsum(counts)/len(v)
        ax.step(np.r_[.98,xx,2.5],np.r_[0,yy,1],where='post',color=color,linestyle=style,lw=1.8,label=label)
    ax.axvline(1.5,color='#8c9399',ls=':',lw=1)
    ax.set_xlim(.98,2.5); ax.set_ylim(0,1.025)
    ax.set_xticks([1,1.25,1.5,1.75,2,2.25,2.5])
    ax.set_xticklabels(['1×','1.25×','1.5×','1.75×','2×','2.25×','2.5×'])
    ax.set_yticks([0,.25,.5,.75,1]); ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel('SLO倍数（TTFT / 基准SLO）',fontsize=10)
    ax.set_ylabel('累计达标比例',fontsize=10)
    ax.tick_params(labelsize=9)

def save_fig(fig, path):
    data = io.BytesIO(); fig.savefig(data,format='png',dpi=220)
    Image.open(io.BytesIO(data.getvalue())).load()
    write(path,data.getvalue()); plt.close(fig)

def main():
    data_path=ROOT/'cdf_requests.csv.gz'; config_path=ROOT/'request_configs.json'
    for path in [data_path,config_path]:
        if not path.exists():
            write(path,(ROOT.parent/'xy_cdf_followup'/path.name).read_bytes())
    with gzip.open(data_path,'rt',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    for r in rows:
        r['ttft_ratio']=float(r['ttft_ms'])/float(r['compute_ms'])
        assert 1-1e-8 <= r['ttft_ratio'] <= 2.5
    configs=json.loads(config_path.read_text())
    for group in IDS:
        pairs={s:{(r['seed'],r['request_id']) for r in rows if r['id']==group and r['strategy']==s} for s in STYLES}
        assert pairs['baseline']==pairs['once']
    tmp=ROOT/'tmp'; tmp.mkdir(exist_ok=True)
    font=tmp/'DroidSansFallback.ttf'; write(font,fitz.Font('china-s').buffer)
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6})
    figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
    stats=[]; points=[]
    for i,group in enumerate(IDS,1):
        for s in STYLES:
            for scope in ['all','A','B']:
                v=vals(rows,group,s,scope); xx,counts=np.unique(v,return_counts=True); yy=np.cumsum(counts)/len(v)
                stats.append(dict(group=i,id=group,strategy=s,scope=scope,n=len(v),slo15_pct=float(np.mean(v<=1.5+1e-10)*100),p95_slo_multiple=float(np.quantile(v,.95,method='inverted_cdf')),max_slo_multiple=float(v.max())))
                points.extend(dict(group=i,id=group,strategy=s,scope=scope,slo_multiple=float(x),cdf=float(y)) for x,y in zip(xx,yy))
    write(ROOT/'slo_cdf_statistics.csv',csv_bytes(stats))
    write(ROOT/'slo_cdf_points.csv.gz',gzip.compress(csv_bytes(points),mtime=0))

    fig,axes=plt.subplots(3,2,figsize=(8.27,11.69))
    fig.subplots_adjust(left=.10,right=.96,top=.83,bottom=.115,hspace=.58,wspace=.30)
    fig.suptitle('TTFT CDF：按SLO倍数比较',fontsize=18,y=.97)
    fig.text(.5,.935,'8 NPU | 1 SSU × 40 GB/s | ring hash | seed 7, 19, 43',ha='center',fontsize=10)
    for i,(group,c) in enumerate(zip(IDS,configs)):
        a,b=c['profile_A'],c['profile_B']; x=a['read_gib']/b['read_gib']; y=a['compute_us']/b['compute_us']
        ax=axes.flat[i]; panel(ax,rows,group)
        ax.set_title(f"组合{i+1}  x={x:.2f}, y={y:.2f}\nA {a['total_length_k']:g}K/{a['nql']}  |  B {b['total_length_k']:g}K/{b['nql']}",fontsize=10,pad=9)
    ax=axes.flat[5]; ax.axis('off')
    ax.text(.5,.96,'SLO ×1.5 达标率 (%)',ha='center',fontsize=11,transform=ax.transAxes)
    cells=[]
    for i,group in enumerate(IDS,1):
        cells.append([str(i)]+[f"{next(r['slo15_pct'] for r in stats if r['id']==group and r['strategy']==s and r['scope']=='all'):.2f}" for s in STYLES])
    table=ax.table(cellText=cells,colLabels=['组合','FIFO','Once'],cellLoc='center',bbox=[.02,.15,.96,.68])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (ri,ci),cell in table.get_celld().items():
        cell.set_edgecolor('#d6e0e7')
        if ri==0: cell.set_facecolor('#e8f0f5')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.916),ncol=2,frameon=False,fontsize=11)
    fig.text(.5,.035,'基准SLO = 各请求自身的总计算时间；竖线为SLO ×1.5。\n每组汇总三个种子的全部相同请求，包含冷启动；CDF未做平滑。',ha='center',fontsize=9)
    paths=[figdir/'ttft_slo_cdf_overview.png']; save_fig(fig,paths[0])

    for i,(group,c) in enumerate(zip(IDS,configs),1):
        a,b=c['profile_A'],c['profile_B']
        fig,axes=plt.subplots(3,1,figsize=(8.27,11.69))
        fig.subplots_adjust(left=.12,right=.95,top=.86,bottom=.115,hspace=.47)
        fig.suptitle(f'组合{i}：TTFT / 基准SLO',fontsize=18,y=.97)
        fig.text(.5,.935,f"A请求 {a['total_length_k']:g}K/{a['nql']}  |  B请求 {b['total_length_k']:g}K/{b['nql']}",ha='center',fontsize=11)
        fig.text(.5,.905,'8 NPU | 1 SSU | ring hash | seed 7, 19, 43',ha='center',fontsize=10)
        for ax,scope,title in zip(axes,['all','A','B'],['全部请求','A请求','B请求']):
            panel(ax,rows,group,scope)
            ax.set_title(f'{title} | 每策略 n={len(vals(rows,group,"baseline",scope))}',loc='left',fontsize=11)
            ax.legend(loc='lower right',frameon=False)
        fig.text(.5,.035,'横轴1.5×处的CDF = TTFT ≤ 1.5 × 基准SLO 的请求比例。\n基准SLO为该请求总计算时间；TTFT不含接纳前排队，包含冷启动。',ha='center',fontsize=9)
        path=figdir/f'ttft_slo_cdf_{i:02d}_{group}.png'; save_fig(fig,path); paths.append(path)

    pdf_data=io.BytesIO(); pdf=canvas.Canvas(pdf_data,pagesize=A4)
    pdf.setTitle('五组请求的TTFT SLO倍数CDF')
    for path in paths:
        pdf.drawImage(ImageReader(str(path)),0,0,width=A4[0],height=A4[1]); pdf.showPage()
    pdf.save(); write(ROOT/'ttft_slo_cdf.pdf',pdf_data.getvalue())
    note='''# TTFT CDF：SLO倍数横轴

横轴 k = TTFT / 该请求的基准SLO。基准SLO沿用既有实验：该请求的总纯计算时间（8层计算时间之和）。因此横轴1.5×处的CDF就是SLO×1.5达标率。

TTFT = NPU接纳到请求完成，不包含接纳前的初始输入队列等待。合并seed=7、19、43的全部相同完成请求，包含冷启动，不使用2–4秒筛选，不做平滑。

原毫秒CDF在约92.3%处有平台，是因为A:B数量比为1:12，B占12/13，且两类TTFT相隔较大；A请求自身参数相近，余下7.7%的请求集中完成，最后便出现陡升。CDF本来就是阶梯函数。

这里逐请求归一化：A除以自己的基准SLO，B除以自己的基准SLO，不是所有请求除以同一个常数。归一化可消除固有计算时长的差异，但相近值仍可能造成阶梯。

配置保持原样。16K/20K/24K计算时间为外推，NQL包含插值；本次只重绘既有实验，不重新仿真。计算比x=VA/VB与横轴SLO倍数k不是同一个量。

运行 python build_slo_cdf.py 可重现全部图片和PDF。依赖 numpy、matplotlib、Pillow、PyMuPDF、reportlab。
'''
    write(ROOT/'README.md',note.encode())
    packed=[*paths,ROOT/'ttft_slo_cdf.pdf',ROOT/'build_slo_cdf.py',data_path,config_path,ROOT/'slo_cdf_statistics.csv',ROOT/'slo_cdf_points.csv.gz',ROOT/'README.md']
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for path in packed:z.write(path,str(path.relative_to(ROOT)))
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z: assert z.testzip() is None
    write(ROOT/'ttft_slo_cdf_bundle.zip',buf.getvalue())
    doc=fitz.open(ROOT/'ttft_slo_cdf.pdf')
    previews=[]
    for i,page in enumerate(doc):
        pix=page.get_pixmap(matrix=fitz.Matrix(.75,.75))
        previews.append(Image.frombytes('RGB',[pix.width,pix.height],pix.samples))
    contact=Image.new('RGB',(previews[0].width*3,previews[0].height*2),'#dddddd')
    for i,img in enumerate(previews):contact.paste(img,((i%3)*img.width,(i//3)*img.height))
    contact.save(tmp/'pdf_contact.png')
    print(json.dumps({'rows':len(rows),'pdf_pages':len(doc),'figures':len(paths),'slo15_table':cells},ensure_ascii=False))

if __name__=='__main__': main()
