#!/usr/bin/env python3
"""Independent window-clipped wait accounting; no simulator or analyzer import."""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import json
import math

CASE = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path,'rt') as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip(a,z,left,right):
    return max(0.,min(z,right)-max(a,left))


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6),(a,b)


def main():
    files = {n:CASE/n for n in ('manifest.json.gz','result.json.gz','command.json','analysis.json')}
    man,raw,command,analysis = [read(files[n]) for n in files]
    assert command['status'] == 'complete'
    assert sha(files['manifest.json.gz']) == command['manifest_sha256']
    assert sha(files['result.json.gz']) == command['output_sha256']
    reqs = {r['request_id']:r for r in man['requests']}
    meta = man['metadata'];rate = meta['npu_bw_gib_s']
    lanes = defaultdict(list)
    for batch in raw['summary']['microbatch_metrics']:
        assert len(batch['member_request_ids']) == 1
        lanes[batch['npu_id']].append(batch)
    results = []
    for left,right in ((2000.,4000.),(2000.,20000.)):
        groups = {r:dict(compute_ms=0.,active_ms=0.,internal_wait_ms=0.,first_wait_ms=0.,
                        first_wait_link_lower_bound_ms=0.) for r in ('L','S')}
        transitions = defaultdict(lambda:dict(first_wait_ms=0.,link_lower_bound_ms=0.,
                                              admitted_inside_count=0))
        release_count = 0
        for npu,lane in lanes.items():
            prior = None
            for batch in sorted(lane,key=lambda b:b['admission_time_ms']):
                rid = batch['member_request_ids'][0]
                q = reqs[rid]['load'];role=q['role'];g=groups[role]
                admission = batch['admission_time_ms'];layers=batch['layer_metrics']
                g['active_ms'] += clip(admission,batch['completion_time_ms'],left,right)
                previous = admission
                for index,layer in enumerate(layers):
                    key='first_wait_ms' if index==0 else 'internal_wait_ms'
                    g[key] += clip(previous,layer['compute_start_ms'],left,right)
                    g['compute_ms'] += clip(layer['compute_start_ms'],layer['compute_end_ms'],left,right)
                    previous=layer['compute_end_ms']
                first=layers[0]
                earliest_ready = first['io_start_time_ms']+q['per_layer_kv_gb']*1000/rate
                assert first['compute_start_ms'] >= earliest_ready-1e-6
                floor_end=max(admission,earliest_ready)
                floor=clip(admission,floor_end,left,right)
                actual=clip(admission,first['compute_start_ms'],left,right)
                assert floor <= actual+1e-6
                g['first_wait_link_lower_bound_ms'] += floor
                if prior is not None:
                    prior_role=reqs[prior['member_request_ids'][0]]['load']['role']
                    # This exact replay uses the previous final computation as
                    # the sole budget for next-request first-layer prefetch.
                    near(first['io_start_time_ms'],prior['layer_metrics'][-1]['compute_start_ms'])
                    near(admission,prior['completion_time_ms'])
                    release_count+=1
                    t=transitions[prior_role+'→'+role]
                    t['first_wait_ms']+=actual;t['link_lower_bound_ms']+=floor
                    t['admitted_inside_count']+=int(left<=admission<right)
                prior=batch
        denom=meta['num_npu']*(right-left)
        compute=math.fsum(g['compute_ms'] for g in groups.values())
        active=math.fsum(g['active_ms'] for g in groups.values())
        idle=denom-active
        short=groups['S']['internal_wait_ms'];long=groups['L']['internal_wait_ms']
        first=math.fsum(g['first_wait_ms'] for g in groups.values())
        floor=math.fsum(g['first_wait_link_lower_bound_ms'] for g in groups.values())
        near(compute+short+long+first+idle,denom)
        prior_analysis=next(w for w in analysis['windows'] if w['start_ms']==left and w['end_ms']==right)
        near(100*compute/denom,prior_analysis['U_percent'])
        for role,g in groups.items():
            for own,stored in (('compute_ms','compute_ms'),('active_ms','active_ms'),
                              ('internal_wait_ms','internal_layer_stall_ms'),('first_wait_ms','first_layer_stall_ms')):
                near(g[own],prior_analysis['classes'][role][stored])
        results.append(dict(start_ms=left,end_ms=right,U_percent=100*compute/denom,
            short_internal_loss_pp=100*short/denom,long_internal_loss_pp=100*long/denom,
            first_layer_loss_pp=100*first/denom,idle_loss_pp=100*idle/denom,
            first_layer_link_only_lower_bound_pp=100*floor/denom,
            first_layer_above_link_lower_bound_pp=100*(first-floor)/denom,
            denominator_card_ms=denom,groups=groups,transitions=dict(transitions),
            all_adjacent_request_prefetch_release_times_verified=release_count))
    record=dict(all_checks_passed=True,no_new_simulation=True,no_analyzer_import=True,
        source_sha256={str(p):sha(p) for p in files.values()},builder_sha256=sha(Path(__file__)),
        ideal_load_ratio=meta['ideal_load_ratio'],results=results,
        link_lower_bound_formula='max(0, V_next/50GiBps - C_previous_last_layer), given the replayed prefetch release at previous final compute start; each exposed interval is clipped to the window.',
        limitation='A lower bound under the existing release rule and one 50GiB/s receive link. It is not an exact FIFO penalty. The residual also includes SSD startup/transfer, issue timing, queues and other effects; it must not all be called FIFO.')
    out=CASE/'figures';(out/'loss_decomposition.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    lines=['# 512K案例：把内部层等待和首层等待分开','',
           '32 NPU、8 SSU×40 GiB/s；长短请求 C 均为外推。理想平均负载99.74%，不代表逐盘逐时欠载。以下直接重读原始层日志并裁剪窗口，未导入已有分析器。','',
           '| 窗口 | U | 短内部损失 pp | 长内部损失 pp | 首层损失 pp | 空闲 pp | 首层中链路固有下界 pp |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f'| [{r["start_ms"]/1000:g},{r["end_ms"]/1000:g})s | {r["U_percent"]:.6f}% | {r["short_internal_loss_pp"]:.6f} | {r["long_internal_loss_pp"]:.6f} | {r["first_layer_loss_pp"]:.6f} | {r["idle_loss_pp"]:.6f} | {r["first_layer_link_only_lower_bound_pp"]:.6f} |')
    lines+=['','前四项损失合计等于100%−U。最后一列包含在首层损失内，不能再加一次。','',
            '短请求最后一层只算0.716653 ms，而下一长请求首层需要接收702.625 MiB；即使数据源没有排队，50 GiB/s链路也至少需要13.723145 ms。因此按当前预取起点，S→L首层至少暴露13.006492 ms等待。这一部分不能归因于FIFO。','',
            '下界使用每个请求的真实预取释放与接纳时刻，并逐请求裁剪窗口；已验证所有相邻请求首层均从前一请求最后一层计算开始预取。高于该下界的剩余首层时间还包含SSD首块延迟、提交时间和实际排队，不能未经对照便全部称为FIFO损失。','',
            '[逐角色、交接类型与来源SHA](loss_decomposition.json)']
    (out/'loss_decomposition.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict((k,r[k]) for k in ('start_ms','end_ms','U_percent','short_internal_loss_pp',
          'long_internal_loss_pp','first_layer_loss_pp','first_layer_link_only_lower_bound_pp')) for r in results]),flush=True)


if __name__=='__main__':
    main()
