#!/usr/bin/env python3
"""Plot common-bin physical SSD throughput and explicitly defined deadline demand.

Run replay_mixed8_physical.py first. This script only reads frozen results and
observations; it does not execute or change scheduling. No rate is capped.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mixed8_physical_bandwidth_20260914'
OLD=ROOT/'results/fifo_mixed_unique_20260914'
LEFT,RIGHT,DT,CAP=2000.,4000.,2.,40.
EDGES=[LEFT+i*DT for i in range(1001)]
LABELS={'fifo':'Baseline FIFO','once':'Once per layer（5 ms）','short_first':'短读取优先（诊断）'}
BLUE,PURPLE,ORANGE,RED,INK='#086bd9','#a32b91','#d97b12','#c92f40','#17324d'

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def savecsv(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def intersect(s,e):return max(0.,min(e,RIGHT)-max(s,LEFT))
def add(row,s,e,rate):
    s,e=max(s,LEFT),min(e,RIGHT)
    if s>=e:return
    i=min(len(row)-1,int((s-LEFT)/DT))
    while s<e:
        z=min(e,EDGES[i+1]);row[i]+=(z-s)*rate/DT;s=z;i+=1
def exact(intervals):
    events={LEFT:[],RIGHT:[]}
    for s,e,v in intervals:
        s,e=max(s,LEFT),min(e,RIGHT)
        if s<e:events.setdefault(s,[]).append(v);events.setdefault(e,[]).append(-v)
    times=sorted(events);v=0.;result=[]
    for s,e in zip(times,times[1:]):
        v=math.fsum([v,*events[s]])
        if abs(v)<1e-10:v=0.
        result.append(dict(start_ms=s,end_ms=e,rate_gib_s=v))
    return result
def locate(policy):
    stage='formal_random_s1_once' if policy=='once' else 'formal_random_s1'
    cases=[p.parent for p in (OLD/stage).glob('*/metrics.json') if read(p)['policy']==policy]
    assert len(cases)==1;return cases[0]
def analyse(policy):
    folder=locate(policy);raw=read(folder/'result.json.gz');man=read(folder/'manifest.json.gz')
    req={r['request_id']:r for r in man['requests']};metrics=read(folder/'metrics.json')
    nominal=[];deadline=[];stall=[];jobs=[]
    for npu in range(8):
        batches=sorted((b for b in raw['summary']['microbatch_metrics'] if b['npu_id']==npu),key=lambda b:b['admission_time_ms'])
        flat=[]
        for batch in batches:
            rid=batch['member_request_ids'][0];q=req[rid]['load']
            nominal.append((batch['admission_time_ms'],batch['completion_time_ms'],q['per_layer_kv_gb']*1e6/q['per_layer_us']))
            prior=batch['admission_time_ms']
            for l in sorted(batch['layer_metrics'],key=lambda l:l['layer']):
                flat.append(dict(request_id=rid,**l))
                if l['compute_start_ms']>prior:stall.append((prior,l['compute_start_ms'],1.))
                prior=l['compute_end_ms']
        for cur,nxt in zip(flat,flat[1:]):
            s,d=cur['compute_start_ms'],cur['compute_end_ms'];C=d-s
            assert math.isclose(nxt['io_start_time_ms'],s,abs_tol=1e-7)
            r,q=req[cur['request_id']]['load'],req[nxt['request_id']]['load']
            V=q['per_layer_kv_gb'];B=V*1000/C
            deadline.append((s,d,B))
            if intersect(s,d)>0:
                jobs.append(dict(npu_id=npu,current_request_id=cur['request_id'],current_layer=cur['layer'],
                    next_request_id=nxt['request_id'],next_layer=nxt['layer'],current_role=r['role'],next_role=q['role'],
                    release_ms=s,deadline_ms=d,read_ready_ms=nxt['io_ready_time_ms'],next_compute_start_ms=nxt['compute_start_ms'],
                    compute_window_ms=C,next_layer_read_gib=V,deadline_reference_gib_s=B,
                    original_current_request_nominal_gib_s=r['per_layer_kv_gb']*1000/C,
                    complete_compute_window_in_warm=LEFT<=s<d<=RIGHT,
                    impossible_even_with_exclusive_ssd=V>CAP*C/1000+1e-12,
                    exclusive_ssd_stall_lower_bound_ms=max(0.,V/CAP*1000-C),
                    actual_io_stall_ms=max(0.,nxt['compute_start_ms']-d)))
    nb,db,sb=[0.]*1000,[0.]*1000,[0.]*1000
    for intervals,target in [(nominal,nb),(deadline,db),(stall,sb)]:
        for s,e,v in intervals:add(target,s,e,v)
    es=exact(deadline);ns=exact(nominal)
    hard=[j for j in jobs if j['complete_compute_window_in_warm'] and j['impossible_even_with_exclusive_ssd']]
    assert all(j['actual_io_stall_ms']+1e-7>=j['exclusive_ssd_stall_lower_bound_ms'] for j in hard)
    stall_card_ms=sum(intersect(s,e)*v for s,e,v in stall)
    assert math.isclose(100*(1-stall_card_ms/(8*(RIGHT-LEFT))),metrics['U_percent'],abs_tol=1e-7)
    summary=dict(policy=policy,window_ms=[LEFT,RIGHT],bin_ms=DT,capacity_gib_s=CAP,
        nominal_demand_mean_gib_s=sum(nb)/1000,nominal_demand_exact_peak_gib_s=max(x['rate_gib_s'] for x in ns),
        deadline_reference_mean_gib_s=sum(db)/1000,deadline_reference_exact_peak_gib_s=max(x['rate_gib_s'] for x in es),
        deadline_reference_exact_above_capacity_ms=sum(x['end_ms']-x['start_ms'] for x in es if x['rate_gib_s']>CAP+1e-8),
        deadline_reference_2ms_peak_gib_s=max(db),deadline_reference_2ms_above_capacity_bins=sum(x>CAP+1e-8 for x in db),
        complete_individually_impossible_prefetches=len(hard),
        exclusive_ssd_lower_bound_total_card_ms=sum(j['exclusive_ssd_stall_lower_bound_ms'] for j in hard),
        actual_stall_for_individually_impossible_prefetches_card_ms=sum(j['actual_io_stall_ms'] for j in hard),
        warm_stall_card_ms=stall_card_ms,npu_utilization_percent=metrics['U_percent'],slo_1p5_percent=metrics['slo_1p5_percent'])
    savecsv(OUT/'data'/f'{policy}_prefetch_deadline_jobs.csv',jobs)
    savecsv(OUT/'data'/f'{policy}_deadline_reference_exact_segments.csv',es)
    return dict(policy=policy,folder=folder,summary=summary,nominal=nb,deadline=db,stall=sb,jobs=jobs)

def draw(case):
    p=case['policy'];a=case['summary'];physical=read(OUT/'data'/f'{p}_physical_bins.json')
    assert physical['bin_edges_ms']==EDGES
    assert all(physical['replay_checks'].values())
    supply=[x*1000/DT for x in physical['ssu_bin_read_gib'][0]]
    assert max(supply)<=CAP+1e-7
    assert math.isclose(sum(supply)/len(supply),physical['window_ssu_read_bandwidth_gib_s'][0],abs_tol=1e-7)
    a.update(physical_ssd_mean_gib_s=sum(supply)/len(supply),physical_ssd_2ms_max_gib_s=max(supply),
        physical_ssd_capacity_violation_bins=sum(x>CAP+1e-7 for x in supply),all_replay_checks_passed=True)
    combined=[dict(start_ms=EDGES[i],end_ms=EDGES[i+1],ssd_actual_gib_s=supply[i],
        current_request_nominal_gib_s=case['nominal'][i],prefetch_deadline_reference_gib_s=case['deadline'][i],
        io_stalled_npus_time_average=case['stall'][i],ssd_read_gib=physical['ssu_bin_read_gib'][0][i]) for i in range(1000)]
    savecsv(OUT/'data'/f'{p}_physical_demand_2ms.csv',combined)
    write(OUT/'data'/f'{p}_plot_audit.json',a)
    fig=plt.figure(figsize=(16,12.2))
    fig.text(.065,.952,f'{LABELS[p]}：真实盘吞吐、预取截止需求与 IO 等待',fontsize=24,color=INK)
    fig.text(.065,.916,f'8 NPU / 1 SSU × 40 GiB/s · ring hash · 长短请求每卡随机混合 · warm [2,4) s · 统一 2 ms 时间窗',fontsize=14,color='#596c80')
    fig.text(.065,.883,f'实际盘平均 {a["physical_ssd_mean_gib_s"]:.3f} GiB/s；实际最大 {a["physical_ssd_2ms_max_gib_s"]:.3f}；容量突破 0 次；NPU 利用率 {a["npu_utilization_percent"]:.2f}%',fontsize=14,color=INK)
    axes=[fig.add_axes([.10,.625,.84,.195]),fig.add_axes([.10,.355,.84,.195]),fig.add_axes([.10,.145,.84,.125])]
    x=[z/1000 for z in EDGES]
    for ax in axes:
        ax.set_xlim(2,4);ax.set_xticks([2,2.25,2.5,2.75,3,3.25,3.5,3.75,4]);ax.grid(alpha=.14)
    ax=axes[0]
    ax.stairs(supply,x,color=BLUE,lw=1.3,label='盘实际读取字节 / 2 ms')
    ax.stairs(case['nominal'],x,color=PURPLE,lw=1.6,label='当前请求常规 V/C（2 ms 均值）')
    ax.axhline(CAP,color='#333',ls='--',lw=1.3,label='盘容量 40 GiB/s')
    ax.set_ylim(0,45);ax.set_yticks([0,10,20,30,40]);ax.set_ylabel('GiB/s')
    ax.set_title('1. 物理供给：从每条 IO 的实际服务起止时刻精确计量，全部 NPU 共用同一时间窗',loc='left',fontsize=14,pad=32)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=3,frameon=False,fontsize=11)
    ax=axes[1]
    ax.stairs(case['deadline'],x,color=ORANGE,lw=1.5,label='下一层 V / 当前层 C（仅计算窗口；2 ms 均值）')
    ax.fill_between(x,[*case['deadline'],case['deadline'][-1]],CAP,where=[v>CAP for v in [*case['deadline'],case['deadline'][-1]]],step='post',color=ORANGE,alpha=.16)
    hard=[j for j in case['jobs'] if j['complete_compute_window_in_warm'] and j['impossible_even_with_exclusive_ssd']]
    ax.scatter([j['deadline_ms']/1000 for j in hard],[j['deadline_reference_gib_s'] for j in hard],marker='x',s=38,color=RED,zorder=5,label='单项预取已超容量：独占盘也无法按时完成')
    ax.axhline(CAP,color='#333',ls='--',lw=1.3)
    ax.set_ylim(0,140);ax.set_yticks([0,40,80,120]);ax.set_ylabel('GiB/s')
    ax.set_title(f'2. 截止时间参考需求：包含短→长 L0；2 ms 均值峰 {a["deadline_reference_2ms_peak_gib_s"]:.3f}；红叉共 {len(hard)} 次',loc='left',fontsize=14,pad=32)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=2,frameon=False,fontsize=10.8)
    ax=axes[2]
    ax.stairs(case['stall'],x,color='#b87500',fill=True,alpha=.6,lw=1)
    ax.set_ylim(0,8);ax.set_yticks([0,2,4,6,8]);ax.set_ylabel('NPU 数');ax.set_xlabel('仿真时间（秒）')
    ax.set_title(f'3. 实际 IO 等待：每个 2 ms 内正在等待 IO 的平均 NPU 数；全窗合计 {a["warm_stall_card_ms"]:.3f} 卡·ms',loc='left',fontsize=14,pad=10)
    fig.text(.065,.07,'橙线越过 40 表示匀速参考需求超过容量，不能单凭这一点证明所有截止时间不可满足；红叉表示单项读取量已超过窗口可服务量。',fontsize=11.7,color=INK)
    fig.text(.065,.035,'红叉统计完整预取窗口；不同策略在 warm 内执行的请求集合不同，次数不能直接衡量策略好坏。真实盘吞吐未做截断或限幅。',fontsize=11.7,color='#596c80')
    fig.savefig(OUT/'images'/f'{p}_total_demand_supply.png',dpi=150);plt.close(fig)

def draw_witness(fifo):
    selected=[]
    for npu,rid in [(7,7000042),(4,4000043)]:
        selected.append(next(j for j in fifo['jobs'] if j['npu_id']==npu and j['next_request_id']==rid and j['next_layer']==0))
    a=min(j['release_ms'] for j in selected);b=max(j['deadline_ms'] for j in selected);duration=b-a
    V=sum(j['next_layer_read_gib'] for j in selected);available=CAP*duration/1000;required=V*1000/duration
    assert V>available
    receipt=read(fifo['folder']/'receipts.json')
    indexed={(r['request_id'],r['layer']):r for r in receipt['layers']}
    proof=dict(window_start_ms=a,window_end_ms=b,window_duration_ms=duration,required_read_gib=V,
        maximum_possible_service_gib=available,minimum_required_average_gib_s=required,
        shortage_gib=V-available,jobs=selected,ignores_other_six_npus=True,strict_deadline_capacity_infeasibility=True)
    write(OUT/'data'/'fifo_deadline_capacity_witness.json',proof)
    fig=plt.figure(figsize=(14,9.5))
    fig.text(.06,.947,'FIFO 的真实容量不足：6.602 ms 内，必须读完两条长请求的 L0',fontsize=22,color=INK)
    fig.text(.06,.897,'只看 NPU 7 和 NPU 4，忽略其他六卡，也无法同时赶上两项预取的截止时间。',fontsize=14,color='#596c80')
    ax=fig.add_axes([.12,.48,.82,.31])
    ax.axvspan(a,b,color=RED,alpha=.075)
    for y,j in zip([1,0],selected):
        s,d=j['release_ms'],j['deadline_ms'];r=indexed[j['next_request_id'],0]
        rs,re=r['first_ssd_start_ms'],r['last_ssd_end_ms']
        assert math.isclose(re-rs,j['next_layer_read_gib']/CAP*1000,abs_tol=1e-7)
        ax.broken_barh([(s,d-s)],(y+.05,.24),facecolors='#dce2e8')
        ax.broken_barh([(rs,re-rs)],(y-.29,.24),facecolors=BLUE)
        ax.plot([d,d],[y-.36,y+.36],color=RED,ls='--',lw=1.5)
        ax.text((s+d)/2,y+.17,f'允许预取窗口：{s:.3f} → {d:.3f}',ha='center',va='center',fontsize=11,color=INK)
        ax.text((rs+re)/2,y-.17,'实际盘服务：40 GiB/s',ha='center',va='center',fontsize=12,color='white')
        ax.text(d,y+.38,'截止',ha='center',color=RED,fontsize=11)
    ax.set_ylim(-.7,1.7);ax.set_yticks([0,1],['NPU 4','NPU 7']);ax.set_xlim(3757,3773)
    ax.set_xticks([3758,3760,3762,3764,3766,3768,3770,3772]);ax.ticklabel_format(useOffset=False,style='plain',axis='x')
    ax.set_xlabel('仿真时间（ms）');ax.grid(axis='x',alpha=.14)
    ax.set_title('灰条：从发起读取到“希望就绪”的时间；蓝条：实际盘服务；红线：各自截止时间',loc='left',fontsize=13,pad=14)
    ax=fig.add_axes([.24,.215,.64,.15])
    ax.barh([1,0],[available,V],color=[BLUE,RED],height=.48)
    for y,v in [(1,available),(0,V)]:ax.text(v+.008,y,f'{v:.6f} GiB',va='center',fontsize=13)
    ax.set_yticks([0,1],['两条 L0 必须读取','这段时间盘最多能读']);ax.set_xlim(0,.66);ax.set_xticks([0,.2,.4,.6]);ax.set_xlabel('读取量（GiB）')
    fig.text(.06,.125,f'共同截止区间 [{a:.3f}, {b:.3f}] ms：需要 {V:.6f} GiB > 容量 {available:.6f} GiB。',fontsize=14,color=INK)
    fig.text(.06,.077,f'满足这些截止时间至少需要 {required:.3f} GiB/s；盘只有 40 GiB/s，因此等待无法全部避免。',fontsize=15,color=RED)
    fig.text(.06,.033,'这是读取量与可用时间直接证明的容量不足；实际盘吞吐依然为 40 GiB/s，没有突破物理上限。',fontsize=12.5,color='#596c80')
    fig.savefig(OUT/'images'/'fifo_capacity_overload_zoom.png',dpi=150);plt.close(fig)
    return proof

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    for d in ['data','images']:(OUT/d).mkdir(parents=True,exist_ok=True)
    fontManager.addfont(args.font)
    plt.rcParams.update({'font.family':FontProperties(fname=args.font).get_name(),'font.size':12,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    cases=[analyse(p) for p in LABELS]
    proof=draw_witness(cases[0])
    if not args.prepare_only:
        for case in cases:draw(case)
    write(OUT/'data'/'physical_bandwidth_summary.json',{'scenarios':[c['summary'] for c in cases],'fifo_strict_capacity_witness':proof})
    savecsv(OUT/'data'/'physical_bandwidth_summary.csv',[c['summary'] for c in cases])
    print(json.dumps({'prepared_only':args.prepare_only,'output':str(OUT),'summaries':[c['summary'] for c in cases]},ensure_ascii=False))
if __name__=='__main__':main()
