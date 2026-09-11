#!/usr/bin/env python3
"""Render complete 60-second result files only. Never runs a simulation."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
RUNS=HERE/'long_validation_7200s'
OUT=HERE/'figures/separate'
PREFIX='sustained16_l176_6l_s64_bridge_48s1l'
DRAW=HERE/'draw_final_timeline.py'
BASE_DRAW=HERE/'draw_timeline.py'
MODES=('random','ordered')
POLICIES=('baseline','once')


def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def prepare():
    source_plan=read(RUNS/'plan.json')
    jobs=[]
    for mode in MODES:
        for policy in POLICIES:
            label=f'{PREFIX}_{mode}_seed7'
            matches=[x for x in source_plan['jobs'] if x['input']['label']==label and x['strategy']==policy]
            assert len(matches)==1
            item=matches[0]['input'];assert sha(item['manifest'])==item['manifest_sha256']
            jobs.append(dict(label=label,mode=mode,strategy=policy,manifest=item['manifest'],
                manifest_sha256=item['manifest_sha256'],input_fingerprint=item['input_fingerprint'],
                command=str(RUNS/'runs'/label/policy/'command.json'),
                planned_timeline_stems=[f'{mode}_{policy}_long',f'{mode}_{policy}_detail'],
                timeline_windows_ms=[[2000,60000],[2000,12000]],demand_stem=f'{mode}_{policy}_demand'))
    result=dict(no_simulation_run=True,scope='Four main population cases only; 12L alternate excluded.',
        source_plan=str(RUNS/'plan.json'),source_plan_sha256=sha(RUNS/'plan.json'),
        script_sha256=sha(__file__),renderer=str(DRAW),renderer_sha256=sha(DRAW),base_collect_source_sha256=sha(BASE_DRAW),jobs=jobs,
        rolling_2s_windows_ms=[[a,a+2000] for a in range(2000,60000,2000)],
        condition='Only command.status=complete, returncode=0 and matching manifest/result hashes are rendered. No simulated values are filled for pending cases.')
    write(OUT/'figure_plan.json',result)
    return result


def plotting():
    spec=importlib.util.spec_from_file_location('draw_final_base',BASE_DRAW)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def text_bounds(fig):
    from matplotlib.text import Text
    fig.canvas.draw();renderer=fig.canvas.get_renderer();b=fig.bbox
    errors=[]
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():continue
        a=obj.get_window_extent(renderer)
        if a.x0<b.x0-1 or a.y0<b.y0-1 or a.x1>b.x1+1 or a.y1>b.y1+1:errors.append(obj.get_text())
    return dict(all_text_inside_canvas=not errors,outside_text=errors)


def nominal_series(man,raw):
    requests={r['request_id']:r for r in man['requests']}
    rates={}
    for rid,r in requests.items():
        placement=r['placement'] if 'placement' in r else man['placements'][r['placement_index']]
        layer=placement[0]
        assert len(placement) in (1,8)
        assert all(x==layer for x in placement)
        rates[rid]=[math.fsum(v for s,v in layer if s==disk)*1e6/r['load']['per_layer_us'] for disk in range(6)]
    events=defaultdict(lambda:[[] for _ in range(6)])
    for b in raw['summary']['microbatch_metrics']:
        assert b['batch_size']==1
        rid=b['member_request_ids'][0]
        assert b['npu_id']==requests[rid]['npu_id']
        for s,v in enumerate(rates[rid]):
            events[b['admission_time_ms']][s].append(v)
            events[b['completion_time_ms']][s].append(-v)
    times=sorted(events);v=[0.]*6;values=[];peak=[0.]*6;over=[0.]*6
    for i,t in enumerate(times):
        # Aggregate ALL admission/completion changes at the same exact time
        # before evaluating the following positive-length interval.
        v=[x+math.fsum(ds) for x,ds in zip(v,events[t])]
        values.append(v[:])
        if i+1<len(times) and times[i+1]>t:
            for s,x in enumerate(v):
                peak[s]=max(peak[s],x)
                if x>40+1e-8:over[s]+=times[i+1]-t
    assert all(abs(x)<1e-7 for x in v)
    return times,values,dict(full_run_peak_by_ssu_gib_s=peak,full_run_over40_ms_by_ssu=over,
        current_request_only=True,next_request_L0_excluded_from_nominal=True,
        same_time_events_batched=True,all_rates_return_to_zero=True,
        overload_comparison_tolerance_gib_s=1e-8,
        capacity_classification_scope='Plot consistency scan. Final strict underload classification comes from independent fresh-sum auditor.')


def render_demand(job,man,raw):
    m=plotting();times,values,audit=nominal_series(man,raw)
    # Retain last state before the left edge, so step-post is exact at 2s.
    before=max((i for i,t in enumerate(times) if t<=2000),default=0)
    inds=[before]+[i for i in range(before+1,len(times)) if times[i]<=60000]
    xx=[max(2000,times[i])/1000 for i in inds]+[60.]
    vv=[values[i] for i in inds]+[values[inds[-1]]]
    fig,ax=m.plt.subplots(figsize=(16,7.8));fig.subplots_adjust(left=.075,right=.98,top=.73,bottom=.20)
    title=('随机' if job['mode']=='random' else '定序')+'输入 · '+('Baseline' if job['strategy']=='baseline' else 'Once per layer')
    fig.suptitle(title+'：6盘当前请求名义需求',fontsize=22,y=.96)
    fig.text(.5,.897,'32 NPU / 6 SSU · 每盘40 GiB/s · seed7 · [2,60)秒',ha='center',fontsize=14)
    peak=max(audit['full_run_peak_by_ssu_gib_s'])
    fig.text(.5,.85,f'全程实际事件扫描最热盘峰值 {peak:.6f} GiB/s；同刻接纳/完成合并后计算',ha='center',fontsize=13)
    for s in range(6):ax.step(xx,[row[s] for row in vv],where='post',lw=.8,label=f'SSU{s}',alpha=.8)
    ax.axhline(40,color='#ad433d',ls='--',lw=1.4,label='单盘容量40')
    ax.set(xlim=(2,60),ylim=(0,42),xticks=[2,10,20,30,40,50,60],yticks=[0,10,20,30,40],xlabel='仿真时间（秒）',ylabel='当前请求名义需求（GiB/s）')
    ax.grid(alpha=.14);fig.legend(loc='upper center',bbox_to_anchor=(.53,.798),ncol=7,frameon=False,fontsize=11)
    fig.text(.075,.112,'每盘曲线 = 各当前已接纳请求该盘每层D / 每层C之和；这是名义需求，不是盘busy或瞬间释放量。',fontsize=11.5)
    fig.text(.075,.065,'下一请求L0按既定口径不额外加入此图；原模拟仍执行全部跨请求预取。全部数据来自该次完整结果。',fontsize=11.5)
    audit['layout']=text_bounds(fig);assert audit['layout']['all_text_inside_canvas']
    stem=OUT/job['demand_stem']
    files=[]
    for ext in ('png','pdf','svg'):
        p=stem.with_suffix('.'+ext);fig.savefig(p,dpi=175);files.append(dict(path=str(p),sha256=sha(p)))
    m.plt.close(fig)
    audit.update(files=files,result_sha256=sha(job['result']),manifest_sha256=sha(job['manifest']),script_sha256=sha(__file__))
    write(stem.with_suffix('.audit.json'),audit)
    return audit


def render_job(job,record):
    assert record.get('returncode')==0
    result=Path(record['output']);assert result.exists() and sha(result)==record['output_sha256']
    assert record['manifest_sha256']==job['manifest_sha256']==sha(job['manifest'])
    cache_path=OUT/f'{job["mode"]}_{job["strategy"]}.render.json'
    hashes=dict(script=sha(__file__),draw=sha(DRAW),base_draw=sha(BASE_DRAW),manifest=sha(job['manifest']),result=sha(result))
    if cache_path.exists():
        cached=read(cache_path)
        if cached.get('source_hashes')==hashes and all(Path(x['path']).exists() and sha(x['path'])==x['sha256'] for x in cached['files']):return cached
    # Reuse root rendering in isolated short-lived processes; no simulator.
    timelines=[]
    for (a,z),stem in zip(job['timeline_windows_ms'],job['planned_timeline_stems']):
        target=OUT/stem
        cmd=[sys.executable,'-B',str(DRAW),'--manifest',job['manifest'],'--result',str(result),
             '--out',str(target),'--start',str(a/1000),'--end',str(z/1000),
             '--caption',('随机' if job['mode']=='random' else '定序')+'混合输入（seed 7）']
        completed=subprocess.run(cmd,capture_output=True,text=True,check=True)
        audit=read(target.with_suffix('.audit.json'))
        assert audit['manifest_sha256']==hashes['manifest'] and audit['result_sha256']==hashes['result']
        timelines.append(dict(stem=stem,command=cmd,stdout=completed.stdout,audit=audit))
    man,raw=read(job['manifest']),read(result)
    assert raw['input_fingerprint']==job['input_fingerprint']==man['input_fingerprint']
    assert raw['strategy']==job['strategy'] and all(raw['summary']['invariants'].values())
    bins=[]
    for start in range(2000,60000,2000):
        matches=[w for w in raw['windows'] if w['start_ms']==start and w['end_ms']==start+2000]
        assert len(matches)==1
        w=matches[0]
        bins.append(dict(start_ms=start,end_ms=start+2000,U_percent=100*w['mean_npu_utilization'],all_active=w['all_npus_active_whole_window']))
    demand=render_demand(dict(job,result=str(result)),man,raw)
    files=[item for t in timelines for item in t['audit']['files']]+demand['files']
    output=dict(label=job['label'],mode=job['mode'],strategy=job['strategy'],source_hashes=hashes,
        result=str(result),command_sha256=sha(job['command']),timelines=timelines,demand=demand,rolling_2s_bins=bins,files=files)
    write(cache_path,output)
    print(json.dumps(dict(rendered=job['label'],strategy=job['strategy'],main_U=timelines[0]['audit']['U'],peak=max(demand['full_run_peak_by_ssu_gib_s'])),ensure_ascii=False),flush=True)
    return output


def render_comparison(rows):
    assert len(rows)==4 and len({(r['mode'],r['strategy']) for r in rows})==4
    m=plotting();fig,ax=m.plt.subplots(figsize=(16,7.4));fig.subplots_adjust(left=.075,right=.98,top=.72,bottom=.20)
    styles={('random','baseline'):('#567080','--'),('ordered','baseline'):('#cf8a27','-'),('random','once'):('#8c78ab','--'),('ordered','once'):('#3a9a80','-')}
    fig.suptitle('每个独立2秒窗口：设备平均利用率',fontsize=22,y=.96)
    fig.text(.5,.89,'32 NPU / 6 SSU · 每盘40 GiB/s · 同一每卡人口，仅请求顺序不同 · seed7',ha='center',fontsize=14)
    output=[]
    for r in rows:
        color,ls=styles[(r['mode'],r['strategy'])];bins=r['rolling_2s_bins']
        label=('随机' if r['mode']=='random' else '定序')+' / '+('Baseline' if r['strategy']=='baseline' else 'Once per layer')
        ax.step([b['start_ms']/1000 for b in bins]+[60],[b['U_percent'] for b in bins]+[bins[-1]['U_percent']],where='post',color=color,ls=ls,lw=1.8,label=label)
        output.extend(dict(mode=r['mode'],strategy=r['strategy'],**b) for b in bins)
    low=max(0,math.floor(min(x['U_percent'] for x in output)/5)*5-5)
    ax.set(xlim=(2,60),ylim=(low,101),xticks=[2,10,20,30,40,50,60],yticks=list(range(low,101,5)),xlabel='仿真时间（秒）；每段覆盖相邻2秒',ylabel='设备利用率（%）')
    ax.grid(alpha=.18);fig.legend(loc='upper center',bbox_to_anchor=(.53,.82),ncol=4,frameon=False)
    fig.text(.075,.11,'每段U = 该2秒内32卡真实计算总时长 / (32 × 2秒)；各段独立统计，不是累积均值。',fontsize=12)
    fig.text(.075,.06,'读取完整结果中的实际2秒窗口；只在四格结果全部完成后绘制，不用缺失值或试验值填充。',fontsize=11.5)
    audit=text_bounds(fig);assert audit['all_text_inside_canvas']
    stem=OUT/'rolling_2s_utilization';files=[]
    for ext in ('png','pdf','svg'):
        p=stem.with_suffix('.'+ext);fig.savefig(p,dpi=175);files.append(dict(path=str(p),sha256=sha(p)))
    m.plt.close(fig)
    with stem.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['mode','strategy','start_ms','end_ms','U_percent','all_active']);w.writeheader();w.writerows(output)
    write(stem.with_suffix('.audit.json'),dict(layout=audit,files=files,rows=output,source_hashes={f'{r["mode"]}_{r["strategy"]}':r['source_hashes'] for r in rows},script_sha256=sha(__file__)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--watch',action='store_true');ap.add_argument('--require-complete',action='store_true');args=ap.parse_args()
    plan=prepare()
    if args.prepare_only:print(json.dumps(dict(prepared_jobs=len(plan['jobs']),plan=str(OUT/'figure_plan.json'))));return
    while True:
        rendered=[];pending=[];failed=[]
        for job in plan['jobs']:
            try:record=read(job['command'])
            except (FileNotFoundError,json.JSONDecodeError):pending.append(job['label']+'/'+job['strategy']);continue
            if record.get('status')=='complete':rendered.append(render_job(job,record))
            elif record.get('status') in ('failed','timeout','cancelled','user-cancelled'):failed.append(dict(job=job,status=record.get('status')))
            else:pending.append(job['label']+'/'+job['strategy'])
        state=dict(updated_utc=datetime.now(timezone.utc).isoformat(),complete=len(rendered),expected=4,pending=pending,failed=failed,no_simulation_run=True)
        write(OUT/'render_status.json',state)
        if len(rendered)==4:render_comparison(rendered);print(json.dumps(dict(complete=4,comparison='written')),flush=True);return
        if failed:raise RuntimeError('Existing simulation ended unsuccessfully; no retry: '+str(failed))
        if not args.watch:
            print(json.dumps(state),flush=True)
            if args.require_complete:raise RuntimeError('Some source results are incomplete')
            return
        time.sleep(30)


if __name__=='__main__':main()
