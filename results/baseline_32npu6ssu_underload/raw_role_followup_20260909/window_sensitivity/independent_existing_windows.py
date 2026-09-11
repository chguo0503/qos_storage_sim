#!/usr/bin/env python3
"""Recompute seven fixed windows from the existing 30 raw result files only.

No simulation or parent metric implementation is imported. All intervals are
half-open; device time always uses 32 cards, including drained cards.
"""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
F = HERE.parent
ROOT = F.parents[2]
SEEDS = [7, 19, 43, 67, 101]
WINDOWS = [(2000, 4000), (1000, 4000), (0, 4000), (2000, 4250),
           (2000, 5000), (2000, 6000), (2000, 8000)]
GROUPS = [('fixed', '固定20长卡/12短卡'), ('random', '每卡混合、随机'),
          ('ordered', '每卡混合、定序')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7)


def clip(a, b, start, end):
    return max(0.0, min(b, end) - max(a, start))


def stat(v):
    return dict(n=len(v), mean=statistics.mean(v), sample_sd=statistics.stdev(v),
                min=min(v), max=max(v))


def inspect(row, group, parent_path):
    path = Path(row['result_path'])
    raw = read(path)
    s = raw['summary']
    requests = s['request_metrics']
    request_by_id = {r['request_id']: r for r in requests}
    assert len(requests) == len(request_by_id) == s['request_count'] == 1212
    assert (s['num_npu'], s['num_ssu'], s['n_layers'], s['batch_size']) == (32, 6, 8, 1)
    assert raw['strategy'] == row['strategy'] in ('baseline', 'once')
    assert raw['submit_seed'] == row['seed']
    assert row['result_sha256'] == sha(path)
    assert row['audit_pass'] and row['active_underload_valid']
    assert all(s['invariants'].values())
    assert raw['policy_config']['assignment'] == 'fixed' and raw['assignment_log'] == []
    assert raw['execution_placement_fingerprint'] == raw['input_placement_fingerprint']
    assert raw['input_fingerprint'] == s['input_fingerprint']
    assert raw['collector_interval_ms'] == 5
    # Core sources are independently bound to the current files; parent audit
    # additionally checks their frozen plans and physical input placements.
    for name, expected in raw['core_and_policy_sha256'].items():
        assert sha(ROOT / name) == expected, (row['label'], name)
    assert sha(ROOT / 'run_baseline_npu32_stress.py') == raw['stress_runner_sha256']
    lanes = [[] for _ in range(32)]
    compute = [[] for _ in range(32)]
    stalls = [[] for _ in range(32)]
    batches_seen = set()
    full_compute = 0.0
    for batch in s['microbatch_metrics']:
        assert batch['batch_size'] == len(batch['member_request_ids']) == 1
        rid = batch['member_request_ids'][0]
        assert rid not in batches_seen
        batches_seen.add(rid)
        r = request_by_id[rid]
        n = r['npu_id']
        assert n == batch['npu_id'] and 0 <= n < 32
        a, f = r['admission_time_ms'], r['completion_time_ms']
        assert r['arrival_time_ms'] == 0 and math.isfinite(f) and 0 <= a < f
        assert close(a, batch['admission_time_ms']) and close(f, batch['completion_time_ms'])
        lanes[n].append((a, f, rid, r['category']))
        previous = a
        amount = 0.0
        layers = sorted(batch['layer_metrics'], key=lambda x: x['layer'])
        assert [x['layer'] for x in layers] == list(range(8))
        for layer in layers:
            cs, ce = layer['compute_start_ms'], layer['compute_end_ms']
            assert previous <= cs + 1e-8 and cs < ce <= f + 1e-8
            compute[n].append((cs, ce, r['category']))
            stalls[n].append((previous, cs, layer['layer']))
            amount += ce - cs
            previous = ce
        assert close(previous, f) and close(amount, r['own_compute_ms'])
        full_compute += amount
    assert batches_seen == set(request_by_id)
    per_npu = []
    for n, lane in enumerate(lanes):
        lane.sort()
        assert lane
        gaps = []
        for previous, current in zip(lane, lane[1:]):
            assert previous[1] <= current[0] + 1e-8
            gaps.append((previous[1], current[0]))
        per_npu.append(dict(npu=n, request_count=len(lane), first_admission_ms=lane[0][0],
                            last_admission_ms=lane[-1][0], final_completion_ms=lane[-1][1],
                            full_compute_ms=math.fsum(b-a for a,b,_ in compute[n]),
                            full_occupied_ms=math.fsum(b-a for a,b,_,_ in lane),
                            before_first_admission_idle_ms=lane[0][0],
                            between_requests_idle_ms=math.fsum(b-a for a,b in gaps)))
    first_drain = min(p['final_completion_ms'] for p in per_npu)
    makespan = max(p['final_completion_ms'] for p in per_npu)
    assert close(makespan, row['makespan_ms']) and close(makespan, s['makespan_ms'])
    assert close(100*full_compute/(32*makespan), row['full_run_U_percent'])
    windows = []
    for start, end in WINDOWS:
        lane_windows = []
        for n, lane in enumerate(lanes):
            c = math.fsum(clip(a,b,start,end) for a,b,_ in compute[n])
            occupied = math.fsum(clip(a,b,start,end) for a,b,_,_ in lane)
            l0 = math.fsum(clip(a,b,start,end) for a,b,l in stalls[n] if l == 0)
            inner = math.fsum(clip(a,b,start,end) for a,b,l in stalls[n] if l != 0)
            tail = clip(lane[-1][1], end, start, end)
            startup = clip(0, lane[0][0], start, end)
            gaps = math.fsum(clip(p[1],q[0],start,end) for p,q in zip(lane,lane[1:]))
            assert close(occupied, c+l0+inner)
            assert close(end-start, occupied+tail+startup+gaps)
            by_category = {cat: math.fsum(clip(a,b,start,end) for a,b,cc in compute[n] if cc==cat)
                           for cat in ('SS','SL','LS','LL')}
            lane_windows.append(dict(npu=n, compute_ms=c, occupied_ms=occupied, l0_stall_ms=l0,
                l1_7_stall_ms=inner, tail_idle_ms=tail, startup_idle_ms=startup,
                between_requests_idle_ms=gaps, compute_by_category_ms=by_category))
        totals = {key:math.fsum(z[key] for z in lane_windows) for key in
                  ('compute_ms','occupied_ms','l0_stall_ms','l1_7_stall_ms','tail_idle_ms',
                   'startup_idle_ms','between_requests_idle_ms')}
        denom = 32*(end-start)
        w = dict(start_ms=start,end_ms=end,duration_ms=end-start,card_time_ms=denom,**totals)
        w.update(device_U_percent=100*totals['compute_ms']/denom,
                 occupied_percent=100*totals['occupied_ms']/denom,
                 tail_idle_percent=100*totals['tail_idle_ms']/denom,
                 occupied_compute_fraction_percent=100*totals['compute_ms']/totals['occupied_ms'],
                 exposed_stall_percent=100*(totals['l0_stall_ms']+totals['l1_7_stall_ms'])/denom,
                 all_32_active=all(close(z['occupied_ms'],end-start) for z in lane_windows),
                 cards_drained_before_end=sum(p['final_completion_ms']<end for p in per_npu),
                 cards_with_positive_SL_and_LL_compute=sum(z['compute_by_category_ms']['SL']>0 and
                     z['compute_by_category_ms']['LL']>0 for z in lane_windows),
                 after_whole_batch_finished_idle_ms=32*clip(makespan,end,start,end),
                 per_npu=lane_windows)
        if (start,end)==(2000,4000):
            assert close(w['device_U_percent'],row['device_utilization_percent'])
            assert w['all_32_active']
        windows.append(w)
    return dict(label=row['label'],group=group,seed=row['seed'],strategy=row['strategy'],
        result_path=str(path),result_sha256=sha(path),input_sha256=row['input_sha256'],
        input_fingerprint=raw['input_fingerprint'],parent_path=str(parent_path),audit_passed=True,
        inherited_full_run_nominal_capacity_pass=row['full_run_capacity_pass'],
        inherited_max_ssu_nominal_gib_s=row['max_ssu_nominal_gib_s'],
        first_card_final_completion_ms=first_drain,
        first_finished_npus=[p['npu'] for p in per_npu if close(p['final_completion_ms'],first_drain)],
        last_card_final_completion_ms=makespan,full_compute_ms=full_compute,
        full_device_U_percent=100*full_compute/(32*makespan),
        per_npu=per_npu,windows=windows)


def main():
    parent_paths = [F/'followup_results.json', F/'mixed_rebinding/results.json']
    source_paths = parent_paths+[Path(__file__)]
    source_before = {str(p):sha(p) for p in source_paths}
    fixed, mixed = map(read,parent_paths)
    assert fixed['completed']==fixed['planned']==60 and not fixed['errors'] and not fixed['pending']
    assert mixed['all_complete'] and not mixed['errors'] and not mixed['pending']
    selected = [(r,'fixed',parent_paths[0]) for r in fixed['rows']
                if r['spec_name']=='raw176_three_l20' and r['mode']=='fixed']
    selected += [(r,r['mode'],parent_paths[1]) for r in mixed['rows'] if r['spec_name']=='raw176_extendedhot']
    assert len(selected)==30
    assert {(g,p,s) for r,g,_ in selected for p,s in [(r['strategy'],r['seed'])]} == {
        (g,p,s) for g,_ in GROUPS for p in ('baseline','once') for s in SEEDS}
    runs = [inspect(row,group,parent_path) for row,group,parent_path in sorted(
        selected,key=lambda t:(t[1],t[0]['strategy'],t[0]['seed']))]
    assert max(r['full_compute_ms'] for r in runs)-min(r['full_compute_ms'] for r in runs)<1e-6
    groups=[]
    metrics=('device_U_percent','occupied_percent','tail_idle_percent','exposed_stall_percent',
             'occupied_compute_fraction_percent','compute_ms','occupied_ms','tail_idle_ms',
             'l0_stall_ms','l1_7_stall_ms')
    for group,name in GROUPS:
        for policy in ('baseline','once'):
            rr=[r for r in runs if r['group']==group and r['strategy']==policy]
            for start,end in WINDOWS:
                ww=[next(w for w in r['windows'] if (w['start_ms'],w['end_ms'])==(start,end)) for r in rr]
                groups.append(dict(group=group,group_name=name,strategy=policy,start_ms=start,end_ms=end,
                    seeds=SEEDS,all_active_count=sum(w['all_32_active'] for w in ww),
                    all_32_mixed_count=sum(w['cards_with_positive_SL_and_LL_compute']==32 for w in ww),
                    statistics={k:stat([w[k] for w in ww]) for k in metrics}))
    paired=[]
    for policy in ('baseline','once'):
        for start,end in WINDOWS:
            drops={}
            for seed in SEEDS:
                rr={r['group']:r for r in runs if r['seed']==seed and r['strategy']==policy}
                uu={g:next(w['device_U_percent'] for w in r['windows'] if
                          (w['start_ms'],w['end_ms'])==(start,end)) for g,r in rr.items()}
                drops[str(seed)]=uu['random']-uu['ordered']
            paired.append(dict(strategy=policy,start_ms=start,end_ms=end,
                per_seed_random_minus_ordered_pp=drops,statistics=stat(list(drops.values()))))
    drainage=[]
    for group,name in GROUPS:
        for policy in ('baseline','once'):
            rr=[r for r in runs if r['group']==group and r['strategy']==policy]
            drainage.append(dict(group=group,group_name=name,strategy=policy,
                first_card_final_completion_ms=stat([r['first_card_final_completion_ms'] for r in rr]),
                last_card_final_completion_ms=stat([r['last_card_final_completion_ms'] for r in rr]),
                full_device_U_percent=stat([r['full_device_U_percent'] for r in rr])))
    source_after={str(p):sha(p) for p in source_paths}
    assert source_before==source_after
    out=dict(all_30_runs_audited=True,run_count=30,window_count=210,windows_ms=WINDOWS,
        definitions=dict(device_U='clipped actual layer compute / [32*(window_end-window_start)]',
            occupied='clipped union of per-NPU [admission,completion); disjointness independently checked',
            tail_idle='per-NPU interval after its last request completion, including after whole-batch completion',
            first_card_empty='earliest per-NPU LAST completion, not last admission when waiting FIFO first becomes empty',
            active_normalized='compute/occupied is supplementary and changes the denominator; not device U',
            full_population='same 1212 requests across all 30 runs; original C/V preserved by prior audits',
            statistics='five seeds equally weighted; sample SD uses n-1; all seven windows retained',
            capacity='inherited prior complete event scan bound to raw result SHA; not rescanned here',
            scope='existing finite-batch results only; no longer-input or steady-state experiment was run'),
        source_sha256=source_before,drainage=drainage,groups=groups,paired_differences=paired,runs=runs)
    target=HERE/'independent_existing_windows.json'
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines=['# 现有 30 格结果的窗口敏感性独立复算','',
        '只读取冻结结果，不运行仿真。每组五种子等权，表中为百分比均值 ± 样本标准差。所有 U 的分母始终是 32 张卡 × 窗口时长。',
        '', '“卡耗尽”指该卡最后一条请求**完成**，不是最后一次接纳。请求占用是 admission→completion；占用中又分实际计算与 exposed I/O stall。卡耗尽后的空闲单独统计。',
        '', '## 利用率：七个窗口全部保留', '', '| 窗口(s) | 固定 Baseline | 固定 Once | 混合随机 Baseline | 混合随机 Once | 混合定序 Baseline | 混合定序 Once | 随机−定序 Baseline (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def pick(group,policy,start,end):
        return next(x for x in groups if (x['group'],x['strategy'],x['start_ms'],x['end_ms'])==(group,policy,start,end))
    def fmt(s):return f"{s['mean']:.4f} ± {s['sample_sd']:.4f}"
    for start,end in WINDOWS:
        vals=[fmt(pick(g,p,start,end)['statistics']['device_U_percent']) for g,_ in GROUPS for p in ('baseline','once')]
        d=next(x for x in paired if (x['strategy'],x['start_ms'],x['end_ms'])==('baseline',start,end))
        lines.append('| '+f'[{start/1000:g},{end/1000:g})'+' | '+' | '.join(vals+[fmt(d['statistics'])])+' |')
    lines+=['','## 有限队列耗尽时刻','',
            '| 分配/策略 | 首卡完成：均值 ± SD(ms) | 首卡完成范围(ms) | 最后完工：均值 ± SD(ms) | 最后完工范围(ms) |',
            '|---|---:|---:|---:|---:|']
    for x in drainage:
        a,b=x['first_card_final_completion_ms'],x['last_card_final_completion_ms']
        lines.append(f"| {x['group_name']} / {x['strategy']} | {fmt(a)} | {a['min']:.3f}–{a['max']:.3f} | {fmt(b)} | {b['min']:.3f}–{b['max']:.3f} |")
    lines+=['','## 占用与尾部空闲','',
            '下表占用率、尾部空闲率使用同一固定卡时间分母；计算率 + 暴露 stall 率 = 占用率。首卡耗尽后的区间包含卡间排空不同步，不能将其低 U 全部归因为 I/O 阻塞。',
            '', '| 窗口(s) | 分配/策略 | 占用% | 尾部空闲% | 暴露stall% | 计算/占用% | 32卡全程active种子数 | 每卡长短都有计算种子数 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for start,end in WINDOWS:
        for g,_ in GROUPS:
            for p in ('baseline','once'):
                x=pick(g,p,start,end);st=x['statistics']
                vals=[f"{st[k]['mean']:.4f}" for k in ('occupied_percent','tail_idle_percent','exposed_stall_percent','occupied_compute_fraction_percent')]
                lines.append(f"| [{start/1000:g},{end/1000:g}) | {x['group_name']} / {p} | "+' | '.join(vals)+f" | {x['all_active_count']}/5 | {x['all_32_mixed_count']}/5 |")
    lines+=['','## 口径边界','',
        '- [0,4) 包含冷启动；[1,4) 也早于原先预设的 2 s 暖窗起点。它们用于敏感性核查，不能自动称为相同暖机条件。',
        '- 扩展终点至首卡耗尽之后时，固定 32 卡分母保留排空空闲；这是该有限批次的真实设备利用率，但不等同于持续饱和稳态。',
        '- 计算/占用比例去掉了无请求时间，仅用于解释时间账，不能替代原设备 U，也不能证明长时间全卡饱和时的表现。',
        '- 逐 run、逐卡完成时间和全部裁剪时间账保存在 JSON。原 [2,4) U 与父汇总逐格交叉核对；层时间、请求/批次身份、无区间重叠、源码和 raw SHA 均独立核验。',
        '- 本复算没有新增到达流量、补齐已耗尽的卡或延长队列；关于更长持续输入的结论，需要另外的输入和仿真。','']
    (HERE/'independent_existing_windows.md').write_text('\n'.join(lines))
    # Compact raw window rows make downstream plotting reproducible.
    fields=['label','group','seed','strategy','start_ms','end_ms','device_U_percent','occupied_percent',
            'tail_idle_percent','exposed_stall_percent','occupied_compute_fraction_percent','all_32_active',
            'compute_ms','occupied_ms','tail_idle_ms','l0_stall_ms','l1_7_stall_ms']
    with (HERE/'independent_existing_windows.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader()
        for r in runs:
            for w in r['windows']:
                writer.writerow({k:r[k] if k in r else w[k] for k in fields})
    print(json.dumps(dict(runs=30,windows=210,all_audits=True,source_sha256=sha(__file__),
                         output_sha256=sha(target)),ensure_ascii=False))


if __name__=='__main__':
    main()
