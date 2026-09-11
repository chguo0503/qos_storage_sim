#!/usr/bin/env python3
"""Plot only existing bridge16 pilot logs; never starts a simulation."""
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
LABEL='bridge16_l176_6l_s64_s32q4k_stagger_ordered_seed7'
MANIFEST=STUDY/'pilot_bridge/inputs'/f'{LABEL}.json.gz'
RESULT=next((STUDY/'pilot_bridge/runs'/LABEL/'baseline').glob('*.json.gz'))
spec=importlib.util.spec_from_file_location('root_timeline',STUDY/'draw_timeline.py')
draw=importlib.util.module_from_spec(spec);spec.loader.exec_module(draw)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.patches import Patch

COLORS={'long':'#72b5a5','short':'#26749b','bridge':'#8c78ab','stall':'#efa640'}


def layout(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox
    outside=[]
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():continue
        b=text.get_window_extent(renderer)
        if b.x0<box.x0-1 or b.y0<box.y0-1 or b.x1>box.x1+1 or b.y1>box.y1+1:
            outside.append(dict(text=text.get_text(),bbox=list(b.bounds)))
    return dict(canvas_pixels=list(box.bounds),outside_text=outside,passed=not outside)


def key(req):
    return 'long' if req['load']['role']=='long' else 'bridge' if req['load']['nql']==4096 else 'short'


def main():
    man,raw=draw.read(MANIFEST),draw.read(RESULT)
    assert man['input_fingerprint']==raw['input_fingerprint']
    assert all(raw['summary']['invariants'].values())
    reqs={r['request_id']:r for r in man['requests']}
    batches=raw['summary']['microbatch_metrics']
    lanes={n:sorted([b for b in batches if b['npu_id']==n],key=lambda b:b['admission_time_ms']) for n in range(32)}
    figures=[];layout_checks=[]
    original=Figure.savefig
    def checked_save(fig,path,*args,**kwargs):
        if str(path).endswith('.png'):
            item=dict(path=str(path),**layout(fig));layout_checks.append(item)
            assert item['passed'],item['outside_text']
        return original(fig,path,*args,**kwargs)
    Figure.savefig=checked_save
    try:
        for a,z in [(2,12),(4,6)]:
            stem=HERE/f'bridge16_baseline_{a}_{z}'
            old=sys.argv
            try:
                sys.argv=['draw_timeline.py','--manifest',str(MANIFEST),'--result',str(RESULT),
                    '--out',str(stem),'--start',str(a),'--end',str(z),'--caption','原始画像桥接输入（12秒试验）']
                draw.main()
            finally:sys.argv=old
            figures.append(dict(stem=str(stem),audit=draw.read(stem.with_suffix('.audit.json'))))

        # Choose the bridge on NPU0 wholly within [4,6), then follow both
        # adjacent requests. All plotted tracks use the same absolute axis.
        candidates=[(i,b) for i,b in enumerate(lanes[0]) if key(reqs[b['member_request_ids'][0]])=='bridge'
                    and b['admission_time_ms']>=4000 and b['completion_time_ms']<=6000]
        i,bridge=min(candidates,key=lambda x:x[1]['admission_time_ms'])
        fast,long=lanes[0][i-1],lanes[0][i+1]
        assert reqs[fast['member_request_ids'][0]]['load']['seq_len_k']==64
        assert key(reqs[long['member_request_ids'][0]])=='long'
        left=math.floor((fast['layer_metrics'][0]['io_start_time_ms']-10)/10)*10
        right=math.ceil((long['layer_metrics'][0]['compute_end_ms']+10)/10)*10
        victims=[]
        for n in range(16,32):
            for b in lanes[n]:
                if key(reqs[b['member_request_ids'][0]])!='short':continue
                for l in b['layer_metrics'][1:]:
                    start=l['compute_start_ms']-l['io_barrier_wait_ms']
                    if start>=left and l['compute_start_ms']<=right:
                        victims.append((l['io_barrier_wait_ms'],n,b,l))
        wait,victim_n,victim_batch,victim_layer=max(victims,key=lambda x:(x[0],-x[1],-x[3]['compute_start_ms']))
        l0=long['layer_metrics'][0];budget=bridge['layer_metrics'][-1]['compute_duration_ms']
        read=l0['io_ready_time_ms']-l0['io_start_time_ms']
        assert math.isclose(l0['io_start_time_ms'],bridge['layer_metrics'][-1]['compute_start_ms'],abs_tol=1e-8)
        assert l0['io_ready_time_ms']<=bridge['completion_time_ms'] and l0['io_barrier_wait_ms']==0

        fig,ax=plt.subplots(figsize=(16,8.8))
        fig.subplots_adjust(left=.14,right=.975,top=.745,bottom=.235)
        fig.suptitle('真实交接：快短请求 → 低读取量的桥接请求 → 下一 Long',fontsize=21,y=.965)
        fig.text(.5,.908,f'Baseline · 32 NPU / 6 SSU · 每盘40 GiB/s · seed7 · 共用绝对时间轴 [{left:g}, {right:g}) ms',ha='center',fontsize=13)
        fig.text(.5,.866,'快短 64K/1024：层C=12.626ms；桥接 32K/4096：28.593ms；Long 176K/1024：31.416ms',ha='center',fontsize=13)
        fig.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in [('long','Long计算'),('short','快短计算'),('bridge','桥接计算'),('stall','暴露 I/O 等待')]],
                   ncol=4,loc='upper center',bbox_to_anchor=(.54,.828),frameon=False,fontsize=12)
        for compute_y,io_y,n in [(3,2,0),(1,0,victim_n)]:
            for b in lanes[n]:
                req=reqs[b['member_request_ids'][0]];k=key(req);previous=b['admission_time_ms']
                for l in b['layer_metrics']:
                    cs,ce=l['compute_start_ms'],l['compute_end_ms']
                    for color,a,z in [(COLORS['stall'],previous,cs),(COLORS[k],cs,ce)]:
                        a,z=max(a,left),min(z,right)
                        if z>a:ax.broken_barh([(a,z-a)],(compute_y-.28,.56),facecolors=color,edgecolors='white',linewidth=.5)
                    a,z=max(l['io_start_time_ms'],left),min(l['io_ready_time_ms'],right)
                    if z>a:ax.broken_barh([(a,z-a)],(io_y-.2,.4),facecolors=COLORS[k],alpha=.43,edgecolors=COLORS[k],linewidth=.6)
                    previous=ce
        for t in [fast['admission_time_ms'],bridge['admission_time_ms'],long['admission_time_ms']]:ax.axvline(t,color='#707070',linestyle='--',alpha=.55,linewidth=.8)
        for a,z,label in [(fast['admission_time_ms'],fast['completion_time_ms'],'64K/1024'),
                          (bridge['admission_time_ms'],bridge['completion_time_ms'],'32K/4096 桥接'),
                          (long['admission_time_ms'],right,'Long')]:
            ax.text((a+z)/2,3.42,label,ha='center',fontsize=10.5)
        ax.broken_barh([(l0['io_start_time_ms'],read)],(1.76,.48),facecolors='none',edgecolors='#285d4f',linewidth=2)
        ax.text((l0['io_start_time_ms']+l0['io_ready_time_ms'])/2,1.59,'下一 Long L0',ha='center',fontsize=10,color='#285d4f')
        vstart=victim_layer['compute_start_ms']-wait
        ax.annotate(f'真实等待 {wait:.3f}ms',xy=((vstart+victim_layer['compute_start_ms'])/2,1.27),
                    xytext=(max(left+65,min(right-65,vstart-85)),1.48),ha='center',fontsize=10.5,
                    arrowprops=dict(arrowstyle='->',color='#915b0b'),color='#915b0b')
        ax.set(xlim=(left,right),ylim=(-.52,3.8),yticks=[0,1,2,3],
            yticklabels=[f'NPU{victim_n}\n层读取生命周期',f'NPU{victim_n}\n计算 / 等待','NPU0\n层读取生命周期','NPU0\n计算 / 等待'],
            xlabel='仿真绝对时间（ms）')
        ax.set_xticks(list(range(math.ceil(left/50)*50,math.floor(right/50)*50+1,50)))
        ax.tick_params(axis='y',length=0,labelsize=11);ax.grid(axis='x',alpha=.16)
        fig.text(.14,.166,f'Long L0：释放 {l0["io_start_time_ms"]:.3f} → ready {l0["io_ready_time_ms"]:.3f}ms；读取生命周期 {read:.3f}ms < 前驱桥接预算 {budget:.3f}ms。',fontsize=11.5)
        fig.text(.14,.123,'淡色条是该层从释放到所有块到达NPU的时间，包含排队与传输；不是SSD独占服务或盘busy时间。',fontsize=11.5)
        fig.text(.14,.080,'两张卡均画真实、连续的同一时间窗。等待例子取本窗后16卡最大内部层等待；不认定某一Long为其FIFO队头。',fontsize=11)
        fig.text(.14,.038,'本图来自已完成的12秒试验；该图不包含或推断尚未完成的60秒试验。',fontsize=11,color='#59636c')
        stem=HERE/'bridge16_handoff_real_time'
        for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=175)
        plt.close(fig)
        front_noninitial=[l for n in range(16) for j,b in enumerate(lanes[n]) for l in b['layer_metrics'] if not(j==0 and l['layer']==0)]
        assert all(abs(l['io_barrier_wait_ms'])<1e-8 for l in front_noninitial)
        phase=[]
        for position in range(min(map(len,[lanes[n] for n in range(16)]))):
            bs=[lanes[n][position] for n in range(16)]
            seq=[(reqs[b['member_request_ids'][0]]['load']['seq_len_k'],reqs[b['member_request_ids'][0]]['load']['nql']) for b in bs]
            assert len(set(seq))==1
            phase.append(dict(position=position,profile=list(seq[0]),
                compute_start_spread_ms=max(b['layer_metrics'][0]['compute_start_ms'] for b in bs)-min(b['layer_metrics'][0]['compute_start_ms'] for b in bs)))
        assert max(p['compute_start_spread_ms'] for p in phase)<.014
        audit=dict(no_simulation_run=True,manifest=str(MANIFEST),result=str(RESULT),
            manifest_sha256=draw.sha(MANIFEST),result_sha256=draw.sha(RESULT),
            input_fingerprint=man['input_fingerprint'],script_sha256=draw.sha(__file__),root_plot_sha256=draw.sha(STUDY/'draw_timeline.py'),
            all_summary_invariants=True,timeline_figures=figures,layout_checks=layout_checks,
            layout_all_text_inside_canvas=all(x['passed'] for x in layout_checks),
            handoff_window_ms=[left,right],handoff_npu=0,fast_request=fast,bridge_request=bridge,next_long_request=long,
            handoff_read_lifetime_ms=read,predecessor_C_budget_ms=budget,
            ready_before_compute_ms=long['admission_time_ms']-l0['io_ready_time_ms'],
            victim_selection='Largest fully window-contained internal-layer exposed wait among NPU16–31; no FIFO head identity attribution.',
            victim_npu=victim_n,victim_request_id=victim_batch['member_request_ids'][0],victim_layer=victim_layer,
            front16_noninitial_layer_count=len(front_noninitial),front16_noninitial_stall_sum_ms=sum(l['io_barrier_wait_ms'] for l in front_noninitial),
            front16_matched_request_compute_phase=phase,front16_max_compute_phase_spread_ms=max(p['compute_start_spread_ms'] for p in phase),
            files=[dict(path=str(p),sha256=draw.sha(p)) for p in sorted(HERE.glob('*.png'))+sorted(HERE.glob('*.pdf'))+sorted(HERE.glob('*.svg'))])
        (HERE/'bridge_pilot_evidence.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:audit[k] for k in ['handoff_window_ms','handoff_read_lifetime_ms','predecessor_C_budget_ms','victim_npu','front16_noninitial_stall_sum_ms','front16_max_compute_phase_spread_ms','layout_all_text_inside_canvas']},ensure_ascii=False))
    finally:Figure.savefig=original


if __name__=='__main__':main()
