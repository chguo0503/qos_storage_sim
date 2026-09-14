#!/usr/bin/env python3
"""Four vector PDF presentations of the existing 32-NPU bandwidth results."""
from pathlib import Path
import gzip
import hashlib
import json
import math
import os

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
STUDY = STAGE.parent
ROOT = STUDY.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pypdf import PdfReader

FONT_PATH=Path('/home/chguo/.fonts/msyh.ttc')
if not FONT_PATH.exists():FONT_PATH=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
matplotlib.font_manager.fontManager.addfont(str(FONT_PATH))
plt.rcParams.update({'font.family':FontProperties(fname=str(FONT_PATH)).get_name(),
                     'axes.unicode_minus':False,'font.size':12,'pdf.fonttype':42})
BLUE,PURPLE,GRAY='#0068d9','#a32b91','#e0e4e9'
INK,MUTED='#172d45','#546980'
LEFT,RIGHT,N=2000.,4000.,32
SOURCES={}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=1e-7,rel_tol=1e-9),(a,b)


def clip(a,z):return max(0.,min(z,RIGHT)-max(a,LEFT))


def read(path):
    SOURCES[str(path.relative_to(ROOT))]=sha(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:return json.load(stream)


def verify_sources(record):
    assert record['all_checks_passed']
    for section in ('sources','builders'):
        for name,digest in record.get(section,{}).items():
            assert sha(ROOT/name)==digest,name
            SOURCES[name]=digest
    if 'builder' in record:
        assert sha(ROOT/record['builder'])==record['builder_sha256']
        SOURCES[record['builder']]=record['builder_sha256']


def load(strategy,order):
    if strategy=='baseline':
        case=STUDY/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
        cached=read(STUDY/'figures/ssu3/fleet_cycle_ratio/checks.json')
        means=read(STUDY/'figures/ssu3/warm_bandwidth_means.json')
        verify_sources(cached);verify_sources(means)
        data=next(r for r in cached['results'] if r['order']==order)
        cycles=data['cycles'];totals=means['results'][order]['totals']
        per_npu=means['results'][order]['rows']
        reference=read(STUDY/'figures/ssu3'/f'{order}_all_32npu_layer_average.checks.json')
        assert reference['all_checks_passed']
        close(data['U_percent'],totals['U_percent'])
        close(reference['U_percent'],totals['U_percent'])
        original_png=ROOT/reference['image']
        assert sha(original_png)==reference['sha256']
        SOURCES[str(original_png.relative_to(ROOT))]=reference['sha256']
    else:
        case=STAGE/'runs'/order/'once'
        cached=read(STAGE/'figures'/f'{order}_all_32npu_layer_average.checks.json')
        verify_sources(cached)
        summary=read(STAGE/'figures'/f'summary_{order}.json')
        verify_sources(summary)
        cycles=read(STAGE/'figures'/f'cycles_{order}.json')
        totals=summary['totals'];per_npu=summary['per_npu']
        assert summary['strategy']=='once'
        assert per_npu==cached['per_npu'] and totals==cached['totals']
        original_png=ROOT/cached['image']['image']
        assert sha(original_png)==cached['image']['sha256']
        SOURCES[str(original_png.relative_to(ROOT))]=cached['image']['sha256']
    man=read(case/'manifest.json.gz');raw=read(case/'result.json.gz');command=read(case/'command.json')
    assert sha(case/'manifest.json.gz')==command['manifest_sha256']
    assert sha(case/'result.json.gz')==command['output_sha256']
    assert raw['strategy']==('baseline' if strategy=='baseline' else 'once')
    assert man['metadata']['num_ssu']==3 and man['metadata']['num_npu']==N
    assert man['metadata']['seed']==7 and man['metadata']['order']==order
    assert raw['common_window']['start_ms']==LEFT and raw['common_window']['end_ms']==RIGHT
    close(totals['U_percent'],100*raw['common_window']['mean_npu_utilization'])
    reqs={q['request_id']:q for q in man['requests']};lanes=[]
    for npu in range(N):
        row=per_npu[npu];assert row['npu']==npu
        batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),
                       key=lambda b:b['admission_time_ms'])
        direct_C=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for b in batches for l in b['layer_metrics'])
        close(100*direct_C/(RIGHT-LEFT),row['U_percent'])
        close(direct_C,raw['common_window']['compute_ms_by_npu'][npu])
        demands=[]
        for b in batches:
            if clip(b['admission_time_ms'],b['completion_time_ms'])<=0:continue
            assert len(b['member_request_ids'])==1
            q=reqs[b['member_request_ids'][0]]['load']
            demands.append((b['admission_time_ms'],b['completion_time_ms'],q['per_layer_kv_gb']*1e6/q['per_layer_us']))
        close(math.fsum(clip(a,z)*B for a,z,B in demands)/(RIGHT-LEFT),row['mean_demand_GiB_s'])
        cs=sorted((c for c in cycles if c['npu']==npu),key=lambda c:c['clipped_start_ms'])
        edge=LEFT
        for c in cs:
            close(edge,c['clipped_start_ms']);edge=c['clipped_end_ms']
            if c['group']!='gray':
                assert c['same_request'] and c['complete_cycle']
                close(c['mean_b_GiB_s']*c['D_ms']/1000,c['V_GiB'])
                close(c['mean_b_GiB_s']/c['B_GiB_s'],c['C_ms']/c['D_ms'])
        close(edge,RIGHT)
        close(math.fsum(c['actual_window_compute_ms'] for c in cs),direct_C)
        lanes.append(dict(cycles=cs,demands=demands,**row))
    return dict(strategy=strategy,order=order,totals=totals,lanes=lanes,
                original_png=str(original_png.relative_to(ROOT)),original_png_sha256=sha(original_png))


def bandwidth(ax,lane):
    blue=[];purple=[];points=[];previous=None
    for c in lane['cycles']:
        a,z=c['clipped_start_ms']/1000,c['clipped_end_ms']/1000
        if c['group']=='gray':ax.axvspan(a,z,color=GRAY,zorder=0);previous=None;continue
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
    label='Baseline' if data['strategy']=='baseline' else '流量分配策略'
    title=f'{label} {data["order"].title()}：32 张 NPU 的每层平均带宽与需求'
    totals=data['totals']
    fig,axes=plt.subplots(N,1,figsize=(18,32),dpi=150,sharex=True,sharey=True,facecolor='white')
    fig.subplots_adjust(left=.108,right=.848,top=.930,bottom=.054,hspace=.34)
    fig.text(.045,.986,title,fontsize=25,color=INK)
    handles=[Line2D([],[],color=PURPLE,lw=2.8,ls='--',label='B_i：当前请求每层 V/C'),
        Line2D([],[],color=BLUE,lw=2.3,marker='o',markerfacecolor='white',label='平均 b_i：每个完整内部层周期一个值'),
        Patch(facecolor=GRAY,label='灰区：跨请求 / 窗口截断；蓝线不填值')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.043,.965),frameon=False,ncol=3,fontsize=13)
    fig.text(.865,.937,'整窗平均（GiB/s）',fontsize=12,color=INK)
    displayed=[]
    for npu,ax in enumerate(axes):
        lane=data['lanes'][npu];assert lane['npu']==npu
        bandwidth(ax,lane)
        ax.set_yticks([0,10,28.476],['0','10','28.48'])
        ulabel=f'NPU {npu:02d}\nU={lane["U_percent"]:.2f}%'
        ax.set_ylabel(ulabel,fontsize=11,rotation=0,ha='right',va='center',labelpad=16,color=INK)
        ax.tick_params(axis='y',labelsize=9,length=3)
        ax.tick_params(axis='x',labelsize=10,length=3,labelbottom=npu in (7,15,23,31))
        ax.grid(axis='x',alpha=.15,lw=.6)
        dlabel=f'需求 {lane["mean_demand_GiB_s"]:.3f}'
        blabel=f'供给 {lane["mean_supply_GiB_s"]:.3f}'
        ax.text(1.022,.70,dlabel,transform=ax.transAxes,ha='left',va='center',fontsize=11,color=PURPLE)
        ax.text(1.022,.28,blabel,transform=ax.transAxes,ha='left',va='center',fontsize=11,color=BLUE)
        displayed.append(dict(npu=npu,U_percent=lane['U_percent'],U_label=ulabel,
                              demand_label=dlabel,supply_label=blabel))
    axes[-1].set_xlabel('时间（秒）',fontsize=13,labelpad=10)
    assert axes[-1].get_xlabel()=='时间（秒）'
    fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box=artist.get_window_extent(renderer)
            assert box.x0>=-2 and box.y0>=-2 and box.x1<=width+2 and box.y1<=height+2,(artist.get_text(),box.bounds)
    path=HERE/f'{data["strategy"]}_{data["order"]}_all_32npu_layer_average.pdf'
    fig.savefig(path,format='pdf',metadata={'Title':title,'Author':'qos_storage_sim','Subject':'32 NPU / 3 SSU, warm [2,4) seconds; existing logs'})
    plt.close(fig)
    pdf=PdfReader(path);assert len(pdf.pages)==1
    text=pdf.pages[0].extract_text()
    normalized=''.join(text.split())
    assert ''.join(title.split()) in normalized
    assert 'Onceperlayer' not in normalized
    assert '时间（秒）' in normalized
    for forbidden in ('周期D=','右侧需求=','两个整窗均值相除','仿真时间（秒）','所有行带宽单位',
                      '32NPU/3SSU','平均每卡需求','每行一张卡；左侧U'):
        assert forbidden not in normalized,forbidden
    for row in displayed:
        assert ''.join(row['U_label'].split()) in normalized,row
        assert ''.join(row['demand_label'].split()) in normalized,row
        assert ''.join(row['supply_label'].split()) in normalized,row
    # Matplotlib paths and text remain vector content; no PNG is embedded.
    assert len(list(pdf.pages[0].images))==0
    pagesize=[float(pdf.pages[0].mediabox.width),float(pdf.pages[0].mediabox.height)]
    return dict(strategy=data['strategy'],order=data['order'],path=str(path.relative_to(ROOT)),
        sha256=sha(path),pages=1,title=title,pagesize_pt=pagesize,vector_no_embedded_images=True,
        x_axis_label='时间（秒）',old_bottom_explanations_removed=True,all_32_U_labels_verified=True,
        top_context_and_reading_notes_removed=True,
        all_32_mean_bandwidth_labels_verified=True,visible_labels_inside_canvas=True,
        original_png=data['original_png'],original_png_sha256=data['original_png_sha256'],
        U_percent=totals['U_percent'],per_npu=displayed)


def main():
    outputs=[]
    for strategy in ('baseline','allocation'):
        for order in ('random','ordered'):
            record=draw(load(strategy,order));outputs.append(record)
            print(json.dumps({k:record[k] for k in ('path','pages','U_percent','vector_no_embedded_images')},ensure_ascii=False),flush=True)
    for name,digest in SOURCES.items():assert sha(ROOT/name)==digest,name
    audit=dict(all_checks_passed=True,no_new_simulation=True,num_npu=N,num_ssu=3,seed=7,
        window_ms=[LEFT,RIGHT],sources=SOURCES,builder=str(Path(__file__).relative_to(ROOT)),
        builder_sha256=sha(Path(__file__)),outputs=outputs,formats_created=['pdf'],
        existing_sources_and_pngs_unchanged=True)
    (HERE/'checks.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
