#!/usr/bin/env python3
"""Plot one observed FIFO blocking episode, with exact block-level provenance."""
from pathlib import Path
from collections import defaultdict,Counter
import gzip
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch,Rectangle

BASE=Path(__file__).resolve().parent
OUT=BASE/'figures'/'separate'
LABEL='concurrency_l768_seed7__exact_cohort4_p1'
LEFT,RIGHT=3094.0,3105.0
SHORT_ID,LONG_ID,LAYER=8000307,2000304,2
BLUE='#26749B';GREEN='#4EA889';ORANGE='#ECA13B';INK='#17364B';GRAY='#526575'
PURPLE='#8154A8';PALE_ORANGE='#F9DEB1'


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def _load_analysis(directory):
    path = directory / 'analysis.json.gz'
    if not path.exists():
        path = directory / 'analysis.json'
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as handle:
        payload = handle.read()
    return path, json.loads(payload), hashlib.sha256(payload).hexdigest()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a,b,c,d):return max(0,min(b,d)-max(a,c))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(BASE/'inputs'/f'{LABEL}.json.gz')
    reference_path=next((BASE/'runs'/LABEL/'baseline').glob('*.json.gz'))
    reference=read(reference_path)
    trace_path=BASE/'diagnostics'/'ordered_block_trace.json.gz'
    trace=read(trace_path)
    assert trace['audit']['passed']
    assert trace['audit']['reference_sha256']==sha(reference_path)
    requests={r['request_id']:r for r in manifest['requests']}
    roles={rid:r['load']['role'] for rid,r in requests.items()}
    batches={b['member_request_ids'][0]:b for b in reference['summary']['microbatch_metrics']}
    rows=[dict(zip(trace['columns'],r)) for r in trace['rows']]

    def layer_info(rid):
        batch=batches[rid]
        layers={l['layer']:l for l in batch['layer_metrics']}
        previous,current=layers[LAYER-1],layers[LAYER]
        flows=[r for r in rows if r['request_id']==rid and r['layer']==LAYER]
        assert len(flows)==len(manifest['placements'][requests[rid]['placement_index']][0])
        assert abs(max(r['link_end_ms'] for r in flows)-current['io_ready_time_ms'])<1e-7
        assert abs(current['io_start_time_ms']-previous['compute_start_ms'])<1e-7
        return {'request_id':rid,'npu':batch['npu_id'],'previous_layer':previous,'layer':current,
                'flows':flows,'compute_budget_ms':previous['compute_duration_ms'],
                'read_span_ms':current['io_ready_time_ms']-current['io_start_time_ms'],
                'stall_ms':current['io_barrier_wait_ms'],
                'size_mib':sum(r['size_gib'] for r in flows)*1024,
                'ssd_service_sum_by_ssu_ms':{s:sum(r['ssd_end_ms']-r['ssd_start_ms'] for r in flows if r['ssu_id']==s) for s in range(6)},
                'link_service_sum_ms':sum(r['link_end_ms']-r['link_start_ms'] for r in flows)}
    short,long=layer_info(SHORT_ID),layer_info(LONG_ID)
    last=max(short['flows'],key=lambda r:r['link_end_ms'])
    qstart,qend=last['enqueue_ms'],last['ssd_start_ms']
    disk=last['ssu_id']
    blockers=[r for r in rows if r['ssu_id']==disk and overlap(qstart,qend,r['ssd_start_ms'],r['ssd_end_ms'])>0]
    service_by_role=defaultdict(float);blocks_by_role=Counter();long_by_npu=Counter()
    for r in blockers:
        role=roles[r['request_id']]
        service_by_role[role]+=overlap(qstart,qend,r['ssd_start_ms'],r['ssd_end_ms'])
        blocks_by_role[role]+=1
        assert r['enqueue_ms']<=qstart+1e-8,'Later arrival bypassed FIFO before the target block'
        if role=='long':
            long_by_npu[r['npu_id']]+=1
            assert r['layer']==2,'This illustration requires only internal long-layer blockers'
    assert abs(sum(service_by_role.values())-(qend-qstart))<1e-7
    assert LONG_ID in {r['request_id'] for r in blockers}
    long_l0_overlaps=[]
    for rid,batch in batches.items():
        if roles[rid]!='long':continue
        l=next(x for x in batch['layer_metrics'] if x['layer']==0)
        if overlap(LEFT,RIGHT,l['io_start_time_ms'],l['io_ready_time_ms'])>0:long_l0_overlaps.append(rid)
    assert not long_l0_overlaps
    # Independent reconstruction of nonoverlapping physical SSD services.
    disk_rows=sorted((r for r in rows if r['ssu_id']==disk and overlap(LEFT,RIGHT,r['ssd_start_ms'],r['ssd_end_ms'])>0),key=lambda r:r['ssd_start_ms'])
    assert all(a['ssd_end_ms']<=b['ssd_start_ms']+1e-8 for a,b in zip(disk_rows,disk_rows[1:]))
    long_fraction=service_by_role['long']/(qend-qstart)
    _,nominal,_=_load_analysis(BASE)
    full_run=next(r for r in nominal['runs'] if r['label']==LABEL and r['strategy']=='baseline')
    info={
        'label':LABEL,'window_ms':[LEFT,RIGHT],
        'selection':'A clear internal-layer FIFO episode within the fixed warm window; selected for explanation, not an average or a new utilization measurement window.',
        'layer_numbering':'L0 is first; L1 is second; L2 is third.',
        'short':{k:v for k,v in short.items() if k!='flows'},
        'long':{k:v for k,v in long.items() if k!='flows'},
        'critical_short_block':last,'critical_ssu':disk,
        'critical_block_queue_wait_ms':qend-qstart,
        'blocking_service_ms_by_role':dict(service_by_role),
        'blocking_count_by_role':dict(blocks_by_role),
        'blocking_long_blocks_by_npu':dict(long_by_npu),
        'long_service_fraction_of_queue_wait':long_fraction,
        'queue_wait_exactly_covered_by_disk_service':True,
        'all_preceding_blocks_enqueued_before_target':True,
        'no_long_l0_io_overlaps_plot':True,
        'full_warm_device_utilization':full_run['windows'][0]['device_utilization'],
        'full_run_nominal_ssu_peak_gib_s':full_run['nominal_demand_scan']['full_run']['max_ssu_gib_s'],
        'reference_sha256':sha(reference_path),'trace_sha256':sha(trace_path),
        'plot_source_sha256':sha(Path(__file__)),
        'read_span_definition':'Layer I/O trigger -> all blocks at NPU memory; includes submit, SSD queue, SSD service, link queue and link service.',
        'service_definition':'Actual SSD activation -> SSD completion (link enqueue); distinct from I/O read span.',
        'block_count_caveat':'The first blocker can already be in service when the target enqueues; service sums include only the overlapping remainder.',
    }
    (OUT/'ordered_zoom_evidence.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')

    for f in ['/home/chguo/.fonts/msyh.ttc','/home/chguo/.fonts/msyhbd.ttc']:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':11,'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
                         'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'ordered-block-evidence'})
    fig=plt.figure(figsize=(16.8,10.8),facecolor='white')
    fig.suptitle('Ordered 局部放大：短流算完了，数据还在前方长流后面排队',x=.51,y=.975,fontsize=20,fontweight='bold',color=INK)
    fig.text(.51,.941,'真实片段 3094–3105 ms（3.094–3.105 秒）  |  32 NPU / 6 SSU  |  Baseline，所有 I/O 使用 Path 0',ha='center',fontsize=11.5,color=GRAY)
    fig.text(.07,.887,f"长流 NPU {long['npu']}：读取 {long['read_span_ms']:.3f} ms < 计算 {long['compute_budget_ms']:.3f} ms → IO stall = 0",fontsize=12,color=INK,
             bbox={'boxstyle':'round,pad=.7','facecolor':'#E8F4EF','edgecolor':'none'})
    fig.text(.54,.887,f"短流 NPU {short['npu']}：读取 {short['read_span_ms']:.3f} ms − 计算 {short['compute_budget_ms']:.3f} ms → 停工 {short['stall_ms']:.3f} ms",fontsize=12,color=INK,
             bbox={'boxstyle':'round,pad=.7','facecolor':'#FEF0D9','edgecolor':'none'})

    ax=fig.add_axes([.15,.287,.815,.49])
    ax.set_xlim(LEFT,RIGHT)
    ax.set_ylim(-1.08,7.45)
    ypos={'long_compute':6.65,'long_io':5.58,'short_compute':3.92,'short_io':2.87,'ssd':.90,'queue':-.25}
    ticks=list(ypos.values())
    labels=[f"长流 NPU {long['npu']}\n计算第 2 层（L1）",'长流\n预取第 3 层（L2）',
            f"短流 NPU {short['npu']}\n计算 / IO stall",'短流\n预取第 3 层（L2）',
            f'SSU {disk} · Path 0\n实际盘服务顺序','短层最后到齐的一块\n排队与实际服务']
    ax.set_yticks(ticks,labels)
    ax.tick_params(axis='y',length=0,pad=12,labelsize=10.8)
    ax.set_xticks(range(3094,3106))
    ax.set_xlabel('仿真时间（ms）—— 同一时间轴；所有条带按真实时长绘制',labelpad=12,fontsize=11.5)
    ax.grid(axis='x',color='#E8EDEF',linewidth=.6,zorder=0)
    ax.spines['left'].set_visible(False)
    for y in [4.78,1.87]:ax.axhline(y,color='#D2DFE4',linewidth=.8)

    def bar(a,z,y,color,height=.55,alpha=1,hatch=None,edge=None,zorder=3):
        lo,hi=max(LEFT,a),min(RIGHT,z)
        if hi<=lo:return
        ax.add_patch(Rectangle((lo,y-height/2),hi-lo,height,facecolor=color,edgecolor=edge or color,
                               linewidth=.8 if edge else 0,alpha=alpha,hatch=hatch,zorder=zorder))
    def txt(a,z,y,text,color='white',size=10.5):
        ax.text((max(LEFT,a)+min(RIGHT,z))/2,y,text,ha='center',va='center',fontsize=size,color=color,zorder=8)

    lc=long['previous_layer'];li=long['layer'];sc=short['previous_layer'];si=short['layer']
    bar(lc['compute_start_ms'],lc['compute_end_ms'],ypos['long_compute'],GREEN)
    txt(lc['compute_start_ms'],RIGHT-.1,ypos['long_compute'],f"上一层计算总长 {long['compute_budget_ms']:.3f} ms（右侧尚未画完）",size=11)
    ax.annotate(f"继续计算到 {lc['compute_end_ms']:.3f} ms →",xy=(RIGHT-.02,ypos['long_compute']),xytext=(RIGHT-2.8,ypos['long_compute']+.57),
                fontsize=10.5,color=GREEN,arrowprops={'arrowstyle':'->','color':GREEN,'lw':1.2},ha='left')
    bar(li['io_start_time_ms'],li['io_ready_time_ms'],ypos['long_io'],'#DBEEE6',hatch='///',edge=GREEN)
    txt(li['io_start_time_ms'],li['io_ready_time_ms'],ypos['long_io'],f"读取跨度 {long['read_span_ms']:.3f} ms：数据提前到齐",INK,10.7)
    ax.plot([li['io_ready_time_ms'],li['io_ready_time_ms']],[ypos['long_io']+.3,ypos['long_compute']-.3],ls=':',lw=1.2,color=GREEN)

    # Show the surrounding NPU context faintly, then emphasize the selected budget/wait.
    for batch in reference['summary']['microbatch_metrics']:
        if batch['npu_id']!=short['npu']:continue
        prev=batch['admission_time_ms']
        for layer in batch['layer_metrics']:
            bar(prev,layer['compute_start_ms'],ypos['short_compute'],ORANGE,alpha=.12)
            bar(layer['compute_start_ms'],layer['compute_end_ms'],ypos['short_compute'],BLUE,alpha=.12)
            prev=layer['compute_end_ms']
    bar(sc['compute_start_ms'],sc['compute_end_ms'],ypos['short_compute'],BLUE)
    txt(sc['compute_start_ms'],sc['compute_end_ms'],ypos['short_compute'],f"计算\n{short['compute_budget_ms']:.3f}",size=8.8)
    bar(sc['compute_end_ms'],si['compute_start_ms'],ypos['short_compute'],ORANGE)
    txt(sc['compute_end_ms'],si['compute_start_ms'],ypos['short_compute'],f"IO stall = {short['stall_ms']:.3f} ms：算完了，却没有数据",INK,11.5)
    bar(si['compute_start_ms'],si['compute_end_ms'],ypos['short_compute'],BLUE)
    txt(si['compute_start_ms'],si['compute_end_ms'],ypos['short_compute'],'下一层\n0.528',size=8.8)
    bar(si['io_start_time_ms'],si['io_ready_time_ms'],ypos['short_io'],'#D9EAF3',hatch='///',edge=BLUE)
    txt(si['io_start_time_ms'],si['io_ready_time_ms'],ypos['short_io'],f"读取跨度 {short['read_span_ms']:.3f} ms（含排队、盘读与传输）",INK,10.5)
    for x,label,yoff in [(sc['compute_end_ms'],'本该继续计算',.77),(si['compute_start_ms'],'数据终于到齐',.77)]:
        ax.plot([x,x],[ypos['short_io']-.4,ypos['short_compute']+.4],ls='--',color=ORANGE,lw=1.1)
        ax.text(x,ypos['short_compute']+yoff,label,ha='center',va='center',fontsize=10,color='#98571A')

    for r in disk_rows:
        bar(r['ssd_start_ms'],r['ssd_end_ms'],ypos['ssd'],GREEN if roles[r['request_id']]=='long' else BLUE,height=.44)
    ax.add_patch(Rectangle((qstart,ypos['ssd']-.39),qend-qstart,.78,fill=False,ec=ORANGE,lw=1.6,zorder=7))
    ax.text((qstart+qend)/2,ypos['ssd']+.64,
            f"框内先服务：{blocks_by_role['long']} 个长块（含跨界块） + {blocks_by_role['short']} 个其它短块",ha='center',va='center',fontsize=10.4,color=INK)
    bar(qstart,qend,ypos['queue'],PALE_ORANGE,edge=ORANGE)
    txt(qstart,qend,ypos['queue'],f"目标短块在盘队列里等待 {qend-qstart:.3f} ms",INK,11)
    bar(last['ssd_start_ms'],last['ssd_end_ms'],ypos['queue'],BLUE,height=.68)
    bar(last['ssd_end_ms'],last['link_start_ms'],ypos['queue'],'#B8C0C5',height=.68)
    bar(last['link_start_ms'],last['link_end_ms'],ypos['queue'],PURPLE,height=.68)
    ax.scatter([last['ssd_start_ms']],[ypos['ssd']],marker='v',s=48,color='#A23B32',zorder=9)
    ax.annotate('最后一个短块\n真正开始读取',xy=(last['ssd_start_ms'],ypos['ssd']-.02),xytext=(RIGHT-.22,ypos['ssd']+.72),
                ha='right',va='bottom',fontsize=9.7,color='#A23B32',arrowprops={'arrowstyle':'->','color':'#A23B32','lw':1.1})
    fig.legend(handles=[Patch(color=BLUE,label='短流计算 / 短块盘服务'),Patch(color=GREEN,label='长流计算 / 长块盘服务'),
                        Patch(color=ORANGE,label='NPU 停工等数据'),Patch(facecolor='#EDF3F4',edgecolor=GRAY,hatch='///',label='读取跨度，包含排队')],
               loc='upper center',bbox_to_anchor=(.535,.83),ncol=4,fontsize=10.5,frameon=False)

    fig.text(.065,.200,f"直接证据：目标短块排队的 {qend-qstart:.3f} ms 中，{service_by_role['long']:.3f} ms（{100*long_fraction:.1f}%）是盘在服务前方长流。",fontsize=13,fontweight='bold',color=INK)
    fig.text(.065,.168,
             f"该短块真正盘读仅 {(last['ssd_end_ms']-last['ssd_start_ms'])*1000:.3f} μs；链路传输 {(last['link_end_ms']-last['link_start_ms'])*1000:.3f} μs，链路排队为 0。不能把 5.817 ms 读取跨度当成盘服务时间。",
             fontsize=10.8,color=INK)
    fig.text(.065,.137,
             f"实际服务量：长层 {long['size_mib']:.3f} MiB，每盘 255 块，累计盘服务 1.070 ms/盘；短层 {short['size_mib']:.3f} MiB，共 7 块，每盘累计盘服务 0.004–0.008 ms（6 盘并行）。",
             fontsize=10.5,color=INK)
    fig.text(.065,.108,
             '阻塞长块来自 NPU 0、1、2、4、5、6，全部是内部第 3 层（L2）；本图没有长流首层 L0 读取。框内首个长块仅计其与等待区间重叠的剩余服务时间。',
             fontsize=9.5,color=GRAY)
    fig.text(.065,.078,
             f"本图用局部解释一次等待，不代表整窗平均。原 [2,4) 秒全卡平均利用率仍为 {100*full_run['windows'][0]['device_utilization']:.2f}%；全程逐盘名义需求峰值 {info['full_run_nominal_ssu_peak_gib_s']:.2f} < 40 GiB/s。",
             fontsize=9.5,color=GRAY)
    fig.text(.065,.049,'读取跨度、计算与停工来自层事件；盘服务顺序来自逐块重放记录。重放的全部请求、层时间及原窗口结果与原运行完全一致。微秒服务条按真实比例绘制，箭头仅定位。',fontsize=9.3,color=GRAY)
    fig.text(.065,.021,f"短 request={SHORT_ID} / 长示例 request={LONG_ID}  ·  原结果 SHA256 {sha(reference_path)[:16]}  ·  数据与推导：ordered_zoom_evidence.json",fontsize=8.7,color=GRAY)

    stem=OUT/'ordered_zoom'
    description=json.dumps({k:v for k,v in info.items() if k not in ('short','long')},ensure_ascii=False)
    fig.savefig(stem.with_suffix('.png'),dpi=240,metadata={'Description':description})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Description':description})
    fig.savefig(stem.with_suffix('.pdf'),metadata={'Title':'Ordered 局部放大：逐块 FIFO 阻塞证据','Subject':description})
    plt.close(fig)
    print(json.dumps({k:info[k] for k in ['window_ms','critical_ssu','critical_block_queue_wait_ms',
          'blocking_service_ms_by_role','blocking_count_by_role','long_service_fraction_of_queue_wait']},ensure_ascii=False))
    print(stem.with_suffix('.png'))


if __name__=='__main__':main()
