#!/usr/bin/env python3
"""Report frozen near-35GiB/s input; no selection based on measured outcomes."""
from pathlib import Path
import csv,gzip,json,math,shutil,zipfile,hashlib,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'source/results/diverse_near35_20260916'
OUT=ROOT/'near35_random_results';FIG=OUT/'figures';DATA=OUT/'data';ASSET=OUT/'assets'
SEEDS=(7,19,43);POLICIES=('baseline','once')
NAMES={'baseline':'Baseline Random','once':'流量分配 Random'}
WINDOWS=('warm_2_4s','warm_2_6s','full_population')

def read(p):
    b=Path(p).read_bytes();return json.loads(gzip.decompress(b) if str(p).endswith('.gz') else b)
def csvout(name,rows):
    # Render strategy names for readers; raw simulator policy IDs stay stable.
    rows=[{(k.replace('once_', '流量分配_', 1) if k.startswith('once_') else k):
           (NAMES.get(v, v) if k == 'policy' else v) for k,v in r.items()} for r in rows]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (DATA/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def mean(values):return float(np.mean(values))

def main():
    for p in (OUT,FIG,DATA,ASSET):p.mkdir(parents=True,exist_ok=True)
    font=ASSET/'DroidSansFallback.ttf'
    if not font.exists():raise FileNotFoundError(f'Missing bundled font: {font}')
    font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.family':['DejaVu Sans',family],'font.size':11,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    perseed=[];samples=[];pernpu=[];perclass=[];perdisk=[];results={};manifests={};checks=[]
    for policy in POLICIES:
        for seed in SEEDS:
            case=EXP/'runs'/f'{policy}_seed{seed}'
            r=read(case/'result.json.gz');m=read(case/'manifest.json.gz');command=read(case/'command.json')
            assert command['status']=='complete' and all(command['checks'].values())
            assert command['core_unchanged'] and command['extension_unchanged']
            assert all(r['summary']['invariants'].values())
            results[policy,seed]=r;manifests[policy,seed]=m;checks.append(command)
            byid={q['request_id']:q for q in m['requests']}
            for i,a in enumerate(r['analysis']):
                window=WINDOWS[i]
                cohort=r['summary']['request_metrics']
                if i<2:cohort=[q for q in cohort if a['start_ms']<=q['admission_time_ms']<a['end_ms']]
                assert all(math.isfinite(q['completion_time_ms']) for q in cohort)
                ttft=np.array([q['completion_time_ms']-q['admission_time_ms'] for q in cohort])
                base=np.array([q['own_compute_ms'] for q in cohort])
                ratios=ttft/base;assert ratios.min()>1-1e-8
                ratios[np.abs(ttft-base)<=1e-9]=1.
                ratios[(ttft>1.5*base)&(ttft<=1.5*base+1e-9)]=1.5
                slo15=float(np.mean(ratios<=1.5)*100)
                assert abs(slo15-a['slo']['percent'])<1e-8
                row=dict(policy=policy,seed=seed,window=window,NPU_utilization_percent=a['U_percent'],
                    SLO1_percent=float(np.mean(ratios<=1)*100),SLO1_5_percent=slo15,requests=len(cohort),
                    TTFT_mean_ms=float(np.mean(ttft)),TTFT_p99_ms=float(np.percentile(ttft,99)),
                    ratio_max=float(ratios.max()),completed_after_window=a['completed_after_window'],
                    any_disk_over40_percent=a['demand']['any_disk_overload_percent'])
                for d in range(3):
                    row[f'SSU{d}_mean_demand_GiB_s']=a['demand']['per_disk_mean_GiB_s'][d]
                    row[f'SSU{d}_over40_percent']=a['demand']['per_disk_overload_percent'][d]
                    row[f'SSU{d}_max_demand_GiB_s']=a['demand']['per_disk_max_GiB_s'][d]
                    if i<2:row[f'SSU{d}_mean_supply_GiB_s']=a['SSD_GiB_s'][d]
                perseed.append(row)
                if i==0:
                    for q,t,x in zip(cohort,ttft,ratios):
                        inp=byid[q['request_id']]
                        samples.append(dict(policy=policy,seed=seed,request_id=q['request_id'],npu=inp['npu_id'],
                            total_K=inp['load']['seq_len_k'],miss=inp['load']['nql'],category=inp['load']['category'],
                            admission_ms=q['admission_time_ms'],completion_ms=q['completion_time_ms'],
                            TTFT_ms=float(t),SLO_base_ms=q['own_compute_ms'],SLO_multiple=float(x)))
                    pernpu.extend(dict(policy=policy,seed=seed,npu=n,U_percent=u) for n,u in enumerate(a['per_npu_U_percent']))
                    for cat,stats in a['slo_by_category'].items():
                        perclass.append(dict(policy=policy,seed=seed,category=cat,count=stats['count'],
                            SLO1_5_percent=stats['percent'],active_U_percent=a['category_active_U_percent'][cat]))
    for seed in SEEDS:assert manifests['baseline',seed]==manifests['once',seed]
    meta=manifests['baseline',7]['metadata'];profiles=meta['profiles']
    profile_rows=[dict(total_K=p['seq_len_k'],miss=p['nql'],hit_tokens=p['ssd_prefix_tokens'],category=p['category'],
        layer_read_MiB=p['per_layer_kv_gib']*1024,layer_compute_ms=p['per_layer_compute_us']/1000,
        required_bandwidth_GiB_s=p['required_bandwidth_gibps'],count_per_npu=int(meta['per_length_miss_counts'][str(p['nql'])])) for p in profiles]
    csvout('input_profiles.csv',profile_rows);csvout('per_seed_metrics.csv',perseed)
    csvout('request_samples.csv',samples);csvout('per_npu_utilization.csv',pernpu);csvout('per_class_metrics.csv',perclass)
    macro=[]
    for window in WINDOWS:
        for policy in POLICIES:
            rr=[r for r in perseed if r['policy']==policy and r['window']==window]
            row=dict(policy=policy,window=window,seed_count=3,requests=sum(r['requests'] for r in rr))
            keys=['NPU_utilization_percent','SLO1_percent','SLO1_5_percent','TTFT_mean_ms','any_disk_over40_percent']
            for d in range(3):
                keys.extend([f'SSU{d}_mean_demand_GiB_s',f'SSU{d}_over40_percent'])
                if window!='full_population':keys.append(f'SSU{d}_mean_supply_GiB_s')
            row.update({k:mean([r[k] for r in rr]) for k in keys});macro.append(row)
    csvout('summary.csv',macro)
    classmacro=[]
    for cat in ('SS','SL','LS','LL'):
        row=dict(category=cat)
        for policy in POLICIES:
            rr=[r for r in perclass if r['policy']==policy and r['category']==cat]
            assert len(rr)==3 and all(r['count']>0 for r in rr)
            row[policy+'_count']=sum(r['count'] for r in rr)
            row[policy+'_SLO1_5_percent']=mean([r['SLO1_5_percent'] for r in rr])
            row[policy+'_active_U_percent']=mean([r['active_U_percent'] for r in rr])
        classmacro.append(row)
    csvout('per_class_summary.csv',classmacro)
    thresholds=[.5,.75,1.,1.1,1.25,1.5,1.75,2.]
    xmax=max(2.,math.ceil(max(r['SLO_multiple'] for r in samples)*4)/4)
    thresholds.append(xmax)
    fig,ax=plt.subplots(figsize=(12,5.7),layout='constrained');points=[]
    for policy,color,style in [('baseline','#2868ac','-'),('once','#db7925','--')]:
        pp=[r for r in samples if r['policy']==policy];x=np.unique([r['SLO_multiple'] for r in pp]+thresholds)
        y=np.mean([np.searchsorted(np.sort([r['SLO_multiple'] for r in pp if r['seed']==seed]),x,side='right')/sum(r['seed']==seed for r in pp) for seed in SEEDS],axis=0)*100
        assert abs(y[np.where(x==1.5)[0][0]]-next(r['SLO1_5_percent'] for r in macro if r['policy']==policy and r['window']=='warm_2_4s'))<1e-9
        ax.step(x,y,where='post',lw=2.2,label=NAMES[policy],color=color,ls=style)
        points.extend(dict(policy=policy,SLO_multiple=float(xx),CDF_percent=float(yy)) for xx,yy in zip(x,y))
    ticks=np.arange(.5,xmax+.001,.25 if xmax<=2.5 else .5 if xmax<=5 else 1.)
    ax.set_xticks(ticks,[f'SLO×{x:g}' for x in ticks]);ax.set(xlim=(.5,xmax),ylim=(0,102),xlabel='TTFT 阈值（SLO × x）',ylabel='达标率 / CDF（%）')
    ax.axvline(1.5,color='#777777',ls=':',lw=1);ax.legend(loc='lower right')
    fig.suptitle('每盘平均需求目标35 GiB/s：TTFT SLO CDF\n32 NPU / 3 SSU · 24种data画像 · warm [2,4)秒 · 三种子等权平均',fontsize=15)
    fig.savefig(FIG/'near35_ttft_slo_cdf.png',dpi=180);plt.close(fig);csvout('cdf_points.csv',points)
    for seed in SEEDS:
        fleet,fa=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
        for col,policy in enumerate(POLICIES):
            r=results[policy,seed];a=r['analysis'][0]
            seg=np.array(a['demand']['segments']);edges=np.r_[seg[:,0],seg[-1,1]]/1000
            s2=np.array(r['bandwidth_2ms']['ssd_GiB_s']);assert s2.shape==(3,1000)
            s10=s2.reshape(3,200,5).mean(axis=2);bins=np.linspace(2,4,201)
            assert s10.max()<=40+1e-8 and np.allclose(s10.mean(axis=1),a['SSD_GiB_s'],atol=1e-8)
            csvout(f'{policy}_seed{seed}_demand_segments.csv',[dict(start_ms=q[0],end_ms=q[1],SSU0_GiB_s=q[2],SSU1_GiB_s=q[3],SSU2_GiB_s=q[4]) for q in seg])
            fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
            peak=max(60.,math.ceil(max(np.max(np.array(results[p,seed]['analysis'][0]['demand']['segments'])[:,2:]) for p in POLICIES)/10)*10)
            for d,ax in enumerate(axs):
                ax.stairs(seg[:,d+2],edges,baseline=None,color='#d76d00',lw=1.7,zorder=4,label='当前请求参考需求 V/C')
                ax.stairs(s10[d],bins,baseline=None,color='#1767ac',lw=1.5,label='实际SSD供给（10毫秒平均）')
                ax.axhline(40,color='#333333',lw=1,ls='--',label='容量40 GiB/s')
                ax.set(xlim=(2,4),ylim=(0,peak*1.03),ylabel=f'SSU {d} 带宽（GiB/s）')
                ax.set_title(f'平均需求：{a["demand"]["per_disk_mean_GiB_s"][d]:.2f} GiB/s   平均供给：{a["SSD_GiB_s"][d]:.2f} GiB/s   需求>40：{a["demand"]["per_disk_overload_percent"][d]:.2f}%',loc='left',fontsize=11)
                if d==0:ax.legend(loc='upper right',fontsize=9,ncol=3)
                csvout(f'{policy}_seed{seed}_SSU{d}_supply_10ms.csv',[dict(start_s=bins[j],end_s=bins[j+1],supply_GiB_s=float(v)) for j,v in enumerate(s10[d])])
            axs[-1].set_xlabel('时间（秒）')
            fig.suptitle(f'{NAMES[policy]}：逐盘需求与实际供给\n平均需求目标35 GiB/s · 32 NPU / 3 SSU · seed{seed} · warm [2,4)秒',fontsize=15)
            fig.savefig(FIG/f'near35_{policy}_per_ssu_seed{seed}.png',dpi=170);plt.close(fig)
            ax=fa[col];ax.stairs(seg[:,2:].sum(axis=1),edges,baseline=None,color='#d76d00',lw=1.8,zorder=4,label='当前请求参考需求 V/C')
            ax.stairs(s10.sum(axis=0),bins,baseline=None,color='#1767ac',lw=1.4,label='实际SSD供给（10毫秒平均）')
            ax.axhline(120,color='#333333',ls='--',lw=1,label='总容量120 GiB/s');ax.set(ylabel='带宽（GiB/s）',title=NAMES[policy],xlim=(2,4));ax.legend(loc='upper right',ncol=3,fontsize=9)
        fa[-1].set_xlabel('时间（秒）');fleet.suptitle(f'平均需求目标35 GiB/s：整机带宽 · seed{seed}',fontsize=15)
        fleet.savefig(FIG/f'near35_fleet_seed{seed}.png',dpi=170);plt.close(fleet)
    (OUT/'metrics.json').write_text(json.dumps(dict(summary=macro,per_seed=perseed),indent=2),encoding='utf-8')
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    write_report(meta,macro,profile_rows,classmacro)
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in (ROOT/'source',OUT):
            for p in sorted(folder.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.log'):continue
                rel=p.relative_to(ROOT)
                if 'diverse_underload_20260916' in rel.parts:continue
                z.write(p,rel)
        z.write(Path(__file__),Path(__file__).name)
    (ROOT/'near35_random_bundle.zip').write_bytes(archive.getvalue())
    with zipfile.ZipFile(ROOT/'near35_random_bundle.zip') as z:
        assert z.testzip() is None
    print(json.dumps(macro,indent=2))

def write_report(meta,macro,profiles,classmacro):
    lines=['# 每盘平均需求约35 GiB/s：Baseline Random与流量分配 Random','',
        '本次按用户更新的目标重新选择请求：关注每盘平均需求约35，不再要求任意画像组合或任意时刻都低于40。跨请求的下一请求L0预取不叠加到当前请求参考需求中；全部真实I/O仍参与仿真并计入实际供给。','',
        '32 NPU、3 SSU×40 GiB/s、每NPU接收链路50 GiB/s、8层、batch=1；保留原实验条带放置 `(block_index+npu_id)%3`。单位沿用源码GiB/s。Baseline使用原path0 FIFO；流量分配使用原完整类别合法Path池、每层规划和5ms快照。','',
        '总长度32/64/80/128/160/200K，每种长度按miss=256/1024/2048/4096配置1/1/3/2条，合计24种画像、每卡42条、每种子1344条。全部读取量和计算时间直接来自data，不缩放、不填充；每卡队列独立随机打乱。两策略按种子共用完全相同的输入和卡内顺序。','',
        '各盘理想平均需求：'+ ' / '.join(f'{x:.6f}' for x in meta['time_weighted_per_ssu_nominal_gib_s'])+' GiB/s。计算为每卡总读取量/总纯计算时间，再对32卡求和；各卡纯计算总时长14.9193秒。实测平均按当前已接纳请求的逐盘V/C对时间积分，受等待导致的驻留时间和有限窗口影响。','',
        '## 利用率、TTFT SLO和实测平均需求','',
        '| 窗口 | 策略 | NPU利用率 | SLO×1 | SLO×1.5 | SSU0平均需求 | SSU1平均需求 | SSU2平均需求 | 任意盘需求>40时间 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in macro:
        lines.append(f'| {r["window"]} | {NAMES[r["policy"]]} | {r["NPU_utilization_percent"]:.3f}% | {r["SLO1_percent"]:.2f}% | {r["SLO1_5_percent"]:.2f}% | {r["SSU0_mean_demand_GiB_s"]:.3f} | {r["SSU1_mean_demand_GiB_s"]:.3f} | {r["SSU2_mean_demand_GiB_s"]:.3f} | {r["any_disk_over40_percent"]:.2f}% |')
    lines+=['','表格单位GiB/s。seed7/19/43的指标分别计算后等权平均；并非合并所有请求后求达标率。warm窗口全部32张卡持续有请求，利用率=计算卡时间/(32×窗口长度)。完整人口含启动和排空，单独展示。','',
        '### [2,4)秒逐类结果','',
        '| 类别 | Baseline SLO×1.5 | 流量分配 SLO×1.5 | Baseline类别活跃期利用率 | 流量分配类别活跃期利用率 |',
        '|---|---:|---:|---:|---:|']
    for r in classmacro:
        lines.append(f'| {r["category"]} | {r["baseline_SLO1_5_percent"]:.2f}% | {r["once_SLO1_5_percent"]:.2f}% | {r["baseline_active_U_percent"]:.2f}% | {r["once_active_U_percent"]:.2f}% |')
    lines+=['','SS/SL/LS/LL沿用源代码：第一个字母按总长度≤80K划S，第二个字母按miss<512划S。这里SS/LS均为miss256，并不意味着读取量小。类别利用率=该类别窗口内计算卡时间/该类别活跃卡时间，先逐种子计算再等权平均。','',
        '本组miss256请求在完整输入中占14.29%，却只占纯计算工作量约1.70%。因此短计算请求的等待减少可以使按请求数计的SLO明显改善，而按计算卡时间计的整机利用率提升较小。两策略warm接纳集合会不同，不能将窗口统计差异直接表述为同一批请求逐一被救回。','',
        'SLO基准=请求8层纯计算时间，SLO×x表示其x倍。TTFT沿用原工程：prefill完成时刻−接纳时刻，不包含接纳前的卡内队列等待，也不模拟真实首token。CDF取[2,4)秒接纳请求并追踪全部到最终完成，保留窗后完成和超时。SLO×1与×1.5边界采用原统计的1e-9ms浮点容差，CDF在对应关键点使用同一容差；原始时间同时保留。','',
        '平均需求约35不等于全程不过载；超过40的参考需求时间比例在表中明确列出。这不包括将下一请求首层预取重复叠加成参考需求。图上10ms实际供给包含所有物理读取，最大不超过单盘40。','',
        '![TTFT SLO CDF](figures/near35_ttft_slo_cdf.png)','',
        '![Baseline逐盘带宽](figures/near35_baseline_per_ssu_seed7.png)','',
        '![流量分配逐盘带宽](figures/near35_once_per_ssu_seed7.png)','',
        '橙线是逐事件当前请求V/C之和，计算和stall期间都保留。蓝线按真实SSD服务区间积分，每10ms平均，由5个原始2ms格精确合并；不平滑拟合、不裁剪。图例标明seed7，seed19/43的独立图也附在figures中。','',
        '## 全部请求画像','',
        '| 总长度K | miss | 类别 | 每层读取MiB | 每层计算ms | 单卡需求GiB/s | 每卡条数 |',
        '|---:|---:|---|---:|---:|---:|---:|']
    for p in profiles:lines.append(f'| {p["total_K"]} | {p["miss"]} | {p["category"]} | {p["layer_read_MiB"]:.5f} | {p["layer_compute_ms"]:.5f} | {p["required_bandwidth_GiB_s"]:.5f} | {p["count_per_npu"]} |')
    lines+=['','## 复现与文件','',
        '`data/summary.csv`与`per_seed_metrics.csv`保存汇总及逐种子指标；`per_class_metrics.csv`为逐类指标；`request_samples.csv`与`cdf_points.csv`为CDF依据；`input_profiles.csv`列出所有画像。','',
        '完整包内`source/results/diverse_near35_20260916`包含冻结输入、全部6次完整结果、运行命令、校验和新增构造/运行脚本。原核心仿真源码未改；source_provenance.json记录源仓库版本及文本落地换行差异。旧实验结果独立保留。','',
        '可直接运行`python report_near35.py`从包内结果重新生成报告和图；需Python 3、NumPy、Matplotlib。若需重新跑仿真，先移动已有runs目录，再运行新增实验中的运行脚本；输入构造器也保护已有输入，不覆盖。']
    (OUT/'near35_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
