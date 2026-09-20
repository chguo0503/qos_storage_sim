"""Three-way PNG comparisons; archived ASU/Once data remain unchanged."""
from pathlib import Path
from collections import defaultdict
import csv,gzip,hashlib,json,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.patches import Patch
from run_case import metrics, load_manifest

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
FIG=ROOT/'figures'
STYLES={'asu_baseline':('ASU baseline','#4b5563','-'),
        'od_baseline':('OD baseline','#d45e00','--'),
        'once':('流量分配策略（Once）','#2474b7','-.')}
GROUPS=['XY12_32','XY12_24','XY12_20','XY12_16','X16','sensitivity20k_076']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if p.name.endswith('.gz') else json.loads(p.read_text())
def write_csv(p,rows):
    opener=gzip.open if p.name.endswith('.gz') else open
    with opener(p,'wt',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
    fig.savefig(FIG/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True)
    font=Path('/home/chguo/.fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    samples=[];per_seed=[];loaded={};provenance={}
    jobs=read(ROOT/'jobs.json')
    for job in jobs:
        requests,meta=load_manifest(ROOT/'inputs'/f"{job['case']}.json.gz")
        original=PROJECT/job['original_directory']
        expected_ids={(q.npu_id,q.request_id) for q in requests}
        for strategy in STYLES:
            if strategy=='od_baseline':path=ROOT/'runs'/f"{job['case']}_od_baseline/native_summary.json.gz"
            elif job['family']=='formula':path=original.parent/('baseline' if strategy=='asu_baseline' else 'once')/'native_summary.json.gz'
            else:path=original.parent/f"sensitivity20k_076_seed7_{'fifo' if strategy=='asu_baseline' else 'once'}/result.json.gz"
            raw=read(path);summary=raw.get('summary',raw)
            assert summary['input_fingerprint']==job['input_fingerprint']
            result,rows=metrics(summary,requests)
            if job['family']=='sensitivity':
                # Preserve the archived plot's exact arithmetic: sum measured
                # compute intervals, not input C*8. They differ only at ~1e-14,
                # but equality at x=1 changes np.unique step aggregation.
                denominator={b['member_request_ids'][0]:sum(l['compute_end_ms']-l['compute_start_ms']
                    for l in sorted(b['layer_metrics'],key=lambda x:x['layer'])) for b in summary['microbatch_metrics']}
                for row in rows:
                    assert abs(denominator[row['request_id']]-row['compute_ms'])<1e-8
                    row['compute_ms']=denominator[row['request_id']]
                    row['ttft_ratio']=row['ttft_ms']/row['compute_ms']
            assert {(r['npu_id'],r['request_id']) for r in rows}==expected_ids
            provenance[str(path.relative_to(PROJECT))]=sha(path)
            loaded[job['group'],job['seed'],strategy]=(summary,requests,meta)
            for row in rows:samples.append(dict(case_group=job['group'],seed=job['seed'],strategy=strategy,**row))
            all_=result['cohorts']['all_requests']['all'];warm=result['cohorts']['warm_admissions']['all']
            per_seed.append(dict(group=job['group'],seed=job['seed'],strategy=strategy,
                U_warm_percent=result['warm_U_percent'],U_full_percent=result['full_U_percent'],
                SLO_full_1p5_percent=all_['slo1.5_percent'],SLO_full_1p5_passed=all_['slo1.5_passed'],
                SLO_full_count=all_['count'],SLO_warm_1p5_percent=warm['slo1.5_percent'],
                SLO_warm_1p5_passed=warm['slo1.5_passed'],SLO_warm_count=warm['count'],
                U_A_warm_percent=result['warm_group_U_percent']['A'],U_B_warm_percent=result['warm_group_U_percent']['B'],
                capacity=meta['capacity_display'],input_fingerprint=job['input_fingerprint']))
    summary_rows=[];stats=[];cdfpoints=[]
    for group in GROUPS:
        for strategy in STYLES:
            seeds=[r for r in per_seed if r['group']==group and r['strategy']==strategy]
            total=sum(r['SLO_full_count'] for r in seeds);passed=sum(r['SLO_full_1p5_passed'] for r in seeds)
            warmtotal=sum(r['SLO_warm_count'] for r in seeds);warmpassed=sum(r['SLO_warm_1p5_passed'] for r in seeds)
            summary_rows.append(dict(group=group,strategy=strategy,seeds=','.join(str(r['seed']) for r in seeds),
                U_warm_percent=np.mean([r['U_warm_percent'] for r in seeds]),
                U_full_percent=np.mean([r['U_full_percent'] for r in seeds]),
                SLO_full_1p5_percent=100*passed/total,SLO_full_1p5_passed=passed,SLO_full_count=total,
                SLO_warm_1p5_percent=100*warmpassed/warmtotal,SLO_warm_1p5_passed=warmpassed,SLO_warm_count=warmtotal,
                SLO_warm_1p5_seed_mean_percent=np.mean([r['SLO_warm_1p5_percent'] for r in seeds]),
                U_A_warm_percent=np.mean([r['U_A_warm_percent'] for r in seeds]),
                U_B_warm_percent=np.mean([r['U_B_warm_percent'] for r in seeds]),capacity=seeds[0]['capacity']))
            for category in ['all','A','B']:
                data=[r for r in samples if r['case_group']==group and r['strategy']==strategy and (category=='all' or r['group']==category)]
                # Population is all matching IDs across all three seeds, not warm-admission cohorts.
                values=np.array([r['ttft_ratio'] for r in data])
                xx,counts=np.unique(values,return_counts=True);yy=np.cumsum(counts)/len(values)
                stats.append(dict(group=group,strategy=strategy,category=category,count=len(values),
                     SLO_full_1p5_percent=100*np.mean(values<=1.5+1e-10),
                     p50=float(np.quantile(values,.5,method='inverted_cdf')),
                     p95=float(np.quantile(values,.95,method='inverted_cdf')),
                     p99=float(np.quantile(values,.99,method='inverted_cdf')),maximum=float(max(values))))
                cdfpoints.extend(dict(group=group,strategy=strategy,category=category,x_ratio=float(x),cdf=float(y)) for x,y in zip(xx,yy))
    def panel(ax,group,category):
        points=[r for r in cdfpoints if r['group']==group and r['category']==category]
        xmax=max(1.6,max(r['x_ratio'] for r in points)*1.02)
        for strategy,(label,color,style) in STYLES.items():
            p=[r for r in points if r['strategy']==strategy]
            pct=next(r['SLO_full_1p5_percent'] for r in stats if r['group']==group and r['strategy']==strategy and r['category']==category)
            xx=[.98]+[r['x_ratio'] for r in p]+[xmax];yy=[0.]+[r['cdf']*100 for r in p]+[100.]
            ax.step(xx,yy,where='post',color=color,linestyle=style,lw=2,label=f'{label} · SLO×1.5 {pct:.2f}%')
        ax.axvline(1.5,color='#667085',ls=':',lw=1)
        ax.set(xlim=(.98,xmax),ylim=(0,102),xlabel='TTFT / 该请求的 8 层纯计算时间',ylabel='累计请求比例 (%)')
        n=next(r['count'] for r in stats if r['group']==group and r['strategy']=='od_baseline' and r['category']==category)
        ax.set_title(f"{'全部请求' if category=='all' else category+' 请求'} · 每策略 {n} 个",loc='left',fontsize=11)
        ax.legend(loc='lower right',frameon=False,fontsize=9)
    figure_files=[]
    for i,group in enumerate(GROUPS,1):
        fig,axes=plt.subplots(3,1,figsize=(10,11))
        fig.subplots_adjust(top=.85,bottom=.13,hspace=.52)
        summary,requests,meta=loaded[group,7,'od_baseline']
        if group!='sensitivity20k_076':
            a,b=meta['config']['profile_A'],meta['config']['profile_B']
            subtitle=f"A {a['total_length_k']:g}K / miss {a['nql']}；B {b['total_length_k']:g}K / miss {b['nql']}；A:B=1:12"
            filename=f'formula_{i:02d}_{group}_ttft_cdf_three_strategies.png'
            title=f'公式实验 {i} · ASU / OD / 流量分配'
            seedtext='7、19、43'
        else:
            subtitle='A：200K，miss 2042–2054；B：20K，miss 1126–1178；A:B=1:6'
            filename='sensitivity20k_076_ttft_cdf_three_strategies.png'
            title='sensitivity20k_076 · ASU / OD / 流量分配';seedtext='7'
        fig.suptitle(title,fontsize=17,y=.97)
        fig.text(.5,.935,subtitle,ha='center',fontsize=10)
        fig.text(.5,.905,f"8 NPU · 1 SSU × {meta['capacity_display']} · ring hash · seed {seedtext}",ha='center',fontsize=10)
        for ax,category in zip(axes,['all','A','B']):panel(ax,group,category)
        fig.text(.5,.025,'全输入、相同请求 ID，包含冷启动；TTFT = 完成 − NPU 接纳，不含接纳前排队。\n画像及原队列保持原样；部分计算时间来自原实验插值或外推。',ha='center',fontsize=9,color='#475467')
        save(fig,filename);figure_files.append(filename)
    fig,axes=plt.subplots(3,1,figsize=(15,10),sharex=True)
    fig.subplots_adjust(top=.89,bottom=.1,hspace=.42)
    for ax,(strategy,(label,color,style)) in zip(axes,STYLES.items()):
        raw,requests,meta=loaded['sensitivity20k_076',7,strategy]
        byid={q.request_id:q.load for q in requests}
        for batch in raw['microbatch_metrics']:
            load=byid[batch['member_request_ids'][0]]
            a=max(2.,batch['admission_time_ms']/1000);b=min(4.,batch['completion_time_ms']/1000)
            if b<=a:continue
            ax.broken_barh([(a,b-a)],(batch['npu_id']-.32,.64),facecolors='#e7eaee',edgecolors='none')
            intervals=[]
            for layer in batch['layer_metrics']:
                a=max(2.,layer['compute_start_ms']/1000);b=min(4.,layer['compute_end_ms']/1000)
                if b>a:intervals.append((a,b-a))
            ax.broken_barh(intervals,(batch['npu_id']-.32,.64),facecolors='#38588c' if load['role']=='L' else '#eeb45f',edgecolors='none')
        row=next(r for r in summary_rows if r['group']=='sensitivity20k_076' and r['strategy']==strategy)
        ax.set_title(f"{label} · NPU 利用率 {row['U_warm_percent']:.3f}% · warm 入场 SLO×1.5 {row['SLO_warm_1p5_percent']:.2f}%",loc='left',fontsize=12,color=color)
        ax.set(yticks=range(8),ylim=(7.7,-.7),ylabel='NPU 编号',xlim=(2.,4.))
        ax.grid(axis='y',visible=False)
    axes[-1].set_xlabel('时间（秒）')
    fig.suptitle('sensitivity20k_076 · 同一批输入的计算时序',fontsize=18,y=.99)
    fig.text(.5,.945,'8 NPU · 1 SSU × 40 GiB/s · 8 层 · seed 7 · 固定分卡、每卡随机混排',ha='center')
    fig.legend(handles=[Patch(color='#38588c',label='A 长请求计算'),Patch(color='#eeb45f',label='B 短请求计算'),Patch(color='#e7eaee',label='已接纳、等待计算（主要为 I/O 等待）')],loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False)
    filename='sensitivity20k_076_timeline_three_strategies.png';save(fig,filename);figure_files.append(filename)
    write_csv(ROOT/'summary.csv',summary_rows);write_csv(ROOT/'per_seed_metrics.csv',per_seed)
    write_csv(ROOT/'cdf_samples.csv.gz',samples);write_csv(ROOT/'cdf_statistics.csv',stats)
    write_csv(ROOT/'cdf_points.csv.gz',cdfpoints)
    report=dict(rows=summary_rows,per_seed=per_seed,figures=figure_files,
        definitions={'U_warm_percent':'Exact compute overlap in [2,4) / (8*2sec), arithmetic mean over seeds',
                     'U_full_percent':'Per-seed full makespan utilization, arithmetic mean over seeds',
                     'SLO_full_1p5_percent':'Pooled all requests TTFT<=1.5*own_compute; same (seed,NPU,requestID) across policies',
                     'SLO_warm_1p5_percent':'Pooled admissions in [2,4); preserve final completions beyond window',
                     'SLO_warm_1p5_seed_mean_percent':'Arithmetic mean of per-seed warm-admission pass rates',
                     'U_A_warm_percent':'A compute overlap / A active overlap, arithmetic seed mean',
                     'U_B_warm_percent':'B compute overlap / B active overlap, arithmetic seed mean'},
        archived_cdf_arithmetic={'formula':'TTFT / raw own_compute_ms; exact original CSV ratio',
                                 'sensitivity':'TTFT / sum(compute_end_ms-compute_start_ms) over 8 layers; exact original plotting code'},
        source_hashes=provenance,input_identity_and_cohort_pairing_verified=True,
        excluded_outputs=['sensitivity bandwidth_pair','mixed8','E1 standalone'])
    (ROOT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'figures':len(figure_files),'summary_rows':len(summary_rows),'cdf_samples':len(samples)}))

if __name__=='__main__':main()
