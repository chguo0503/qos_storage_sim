#!/usr/bin/env python3
"""Independent raw-result mechanism audit. No runner/metrics/simulator imports."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(Path(path).read_text())


def overlap(a, z, start, end):
    return max(0., min(z, end) - max(a, start))


def close(a, b, tolerance=1e-6):
    assert math.isclose(a, b, abs_tol=tolerance, rel_tol=1e-10), (a, b)


def csv_write(path, rows):
    if not rows:
        Path(path).write_text('no_records\n')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:json.dumps(value, separators=(',', ':')) if isinstance(value, (list,dict)) else value
                         for key,value in row.items()} for row in rows)


def load_raw(result_path, manifest_path, command_path):
    result, manifest = read(result_path), read(manifest_path)
    if command_path.exists():
        command = read(command_path)
        assert command['status'] == 'complete' and command['completed_simulation']
        assert not command.get('smoke', False)
        if 'result_sha256' in command:
            assert command['result_sha256'] == sha(result_path)
        if 'manifest_sha256' in command:
            assert command['manifest_sha256'] == sha(manifest_path)
    else:
        command = None
    requests = {row['request_id']:row for row in manifest['requests']}
    metrics = {row['request_id']:row for row in result['summary']['request_metrics']}
    assert len(requests) == len(manifest['requests']) == len(metrics)
    assert set(requests) == set(metrics)
    assert all(math.isfinite(r['completion_time_ms']) for r in metrics.values())
    assert result['summary']['batch_size'] == 1
    assert result['summary']['cross_request_layer0_prefetch'] is True
    assert (result['summary']['num_npu'], result['summary']['num_ssu']) == (32,3)
    assert result['input_fingerprint'] == manifest['input_fingerprint']
    table = ast.literal_eval((ROOT/'data').read_text())
    records, original_data_count = {}, 0
    for rid, row in requests.items():
        load = row['load']
        placement = manifest['placements'][row['placement_index']]
        assert len(placement) == 1, 'This audit expects one placement reused by all layers'
        volume = [math.fsum(size for disk,size in placement[0] if disk == s) for s in range(3)]
        close(sum(volume), load['per_layer_kv_gb'], 1e-11)
        c = load['per_layer_us']/1000
        key = (load['seq_len_k'],load['nql'])
        original = key in table and math.isclose(c*1000, table[key][1], rel_tol=1e-12) and math.isclose(sum(volume),table[key][3],rel_tol=1e-12)
        original_data_count += original
        records[rid] = dict(request_id=rid, npu_id=row['npu_id'], role=load['role'],
                            profile=f'{key[0]}:{key[1]}', C_ms=c, V_per_ssu_GiB=volume,
                            B_per_ssu_GiB_s=[v*1000/c for v in volume],
                            original_request_id=load.get('original_request_id',rid),
                            matches_raw_data=original)
    return result, manifest, command, metrics, records, original_data_count


def demand_sweep(metrics, records, start, end):
    events = defaultdict(lambda: {'end':[], 'start':[]})
    events[start]; events[end]
    for rid, r in metrics.items():
        a,z=max(start,r['admission_time_ms']),min(end,r['completion_time_ms'])
        if z>a:
            events[a]['start'].append(rid); events[z]['end'].append(rid)
    active, segments, overloads = {}, [], []
    totals, maximum = [0.]*3, [0.]*3
    peak_segments = [None]*3
    over_ms = [0.]*3
    times = sorted(events)
    for a,z in zip(times[:-1],times[1:]):
        for rid in events[a]['end']:
            assert active[records[rid]['npu_id']] == rid
            del active[records[rid]['npu_id']]
        for rid in events[a]['start']:
            npu=records[rid]['npu_id']; assert npu not in active
            active[npu]=rid
        d=[math.fsum(records[rid]['B_per_ssu_GiB_s'][s] for rid in active.values()) for s in range(3)]
        roles={role:sum(records[rid]['role']==role for rid in active.values()) for role in ('A','B')}
        row=dict(start_ms=a,end_ms=z,D0_GiB_s=d[0],D1_GiB_s=d[1],D2_GiB_s=d[2],
                 active_npus=len(active),A_active_npus=roles['A'],B_active_npus=roles['B'])
        segments.append(row)
        for s in range(3):
            totals[s]+=d[s]*(z-a)
            if d[s]>maximum[s]:
                maximum[s]=d[s];peak_segments[s]=dict(row,active_request_ids=[active.get(n) for n in range(32)])
            if d[s]>40:
                over_ms[s]+=z-a
                overloads.append(dict(ssu_id=s,start_ms=a,end_ms=z,demand_GiB_s=d[s],duration_ms=z-a,
                                      A_active_npus=roles['A'],B_active_npus=roles['B']))
    return dict(maximum_GiB_s_by_ssu=maximum,mean_GiB_s_by_ssu=[v/(end-start) for v in totals],
                strict_underload_all_disks=all(value<40 for value in maximum),
                overload_percent_by_ssu=[100*v/(end-start) for v in over_ms],
                all_npus_active=all(row['active_npus']==32 for row in segments),
                A_active_count_min=min(row['A_active_npus'] for row in segments),
                A_active_count_max=max(row['A_active_npus'] for row in segments),
                A_active_count_time_average=math.fsum(row['A_active_npus']*(row['end_ms']-row['start_ms']) for row in segments)/(end-start),
                B_active_count_time_average=math.fsum(row['B_active_npus']*(row['end_ms']-row['start_ms']) for row in segments)/(end-start),
                peak_segments_by_ssu=peak_segments),segments,overloads


def analyze(result, metrics, records, start, end, top):
    summary=result['summary'];duration=end-start
    assert 0 <= start < end <= summary['makespan_ms']+1e-8
    lanes=[[] for _ in range(32)]
    npu=[dict(npu_id=i,compute_ms=0.,active_ms=0.,io_stall_ms=0.,compute_queue_ms=0.,
              role_compute_ms=defaultdict(float),role_active_ms=defaultdict(float),
              role_requests_computing=defaultdict(set)) for i in range(32)]
    for rid,r in metrics.items():
        rec=records[rid];n=npu[rec['npu_id']]
        active=overlap(r['admission_time_ms'],r['completion_time_ms'],start,end)
        n['active_ms']+=active;n['role_active_ms'][rec['role']]+=active
    all_layers, stalls=[],[]
    for batch in summary['microbatch_metrics']:
        assert batch['batch_size']==1 and len(batch['member_request_ids'])==1
        rid=batch['member_request_ids'][0];rec=records[rid];r=metrics[rid]
        assert batch['npu_id']==rec['npu_id']
        previous_end=r['admission_time_ms']
        for layer in sorted(batch['layer_metrics'],key=lambda q:q['layer']):
            a,z=layer['compute_start_ms'],layer['compute_end_ms']
            close(z-a,rec['C_ms'])
            io_ready=layer['io_ready_time_ms']
            exposed_end=max(previous_end,io_ready)
            assert a+1e-7>=exposed_end
            wait=overlap(previous_end,exposed_end,start,end)
            queue=overlap(exposed_end,a,start,end)
            close(max(0,io_ready-previous_end),layer['io_barrier_wait_ms'])
            compute=overlap(a,z,start,end)
            n=npu[rec['npu_id']]
            n['compute_ms']+=compute;n['io_stall_ms']+=wait;n['compute_queue_ms']+=queue
            n['role_compute_ms'][rec['role']]+=compute
            if compute>0:n['role_requests_computing'][rec['role']].add(rid)
            kind='internal_L1_to_L7' if layer['layer'] else ('cross_request_L0' if r['layer0_cross_request_prefetched'] else 'initial_L0')
            record=dict(**rec,layer=layer['layer'],compute_start_ms=a,compute_end_ms=z,
                        io_start_time_ms=layer['io_start_time_ms'],io_ready_time_ms=io_ready,
                        ssd_complete_time_ms=layer.get('ssd_complete_time_ms'),
                        barrier_budget_end_ms=previous_end,full_io_stall_ms=max(0,io_ready-previous_end),
                        warm_io_stall_ms=wait,warm_compute_ms=compute,warm_queue_ms=queue,kind=kind)
            lanes[rec['npu_id']].append(record);all_layers.append(record)
            if wait>1e-10:stalls.append(record)
            previous_end=z
        close(previous_end,r['completion_time_ms'])
    cycles=[]
    transition_wait=defaultdict(float)
    for lane in lanes:
        lane.sort(key=lambda q:q['compute_start_ms'])
        for previous,current in zip(lane,lane[1:]):
            assert previous['compute_end_ms']<=current['compute_start_ms']+1e-7
            close(current['io_start_time_ms'],previous['compute_start_ms'])
            same=previous['request_id']==current['request_id']
            assert same == (current['layer']!=0)
            transition=f"{previous['role']}->{current['role']}"
            if not same:transition_wait[transition]+=current['warm_io_stall_ms']
            a,z=previous['compute_start_ms'],current['compute_start_ms']
            if overlap(a,z,start,end)<=0:continue
            c=previous['C_ms'];period=z-a
            supply=[v*1000/period for v in current['V_per_ssu_GiB']]
            if same:
                for b,B in zip(supply,previous['B_per_ssu_GiB_s']):
                    if B>0:close(b/B,c/period,1e-9)
            cycles.append(dict(npu_id=current['npu_id'],same_request=same,
                previous_request_id=previous['request_id'],previous_layer=previous['layer'],previous_role=previous['role'],
                request_id=current['request_id'],layer=current['layer'],role=current['role'],profile=current['profile'],
                previous_profile=previous['profile'],next_request_role=current['role'],
                start_ms=a,end_ms=z,current_compute_C_ms=c,next_compute_C_ms=current['C_ms'],period_ms=period,
                current_V_per_ssu_GiB=previous['V_per_ssu_GiB'],next_V_per_ssu_GiB=current['V_per_ssu_GiB'],
                current_B_per_ssu_GiB_s=previous['B_per_ssu_GiB_s'],
                prefetch_budget_demand_per_ssu_GiB_s=[v*1000/c for v in current['V_per_ssu_GiB']],
                derived_complete_cycle_supply_per_ssu_GiB_s=supply,
                supply_provenance='complete-cycle volume conservation, not independently logged SSD rates',
                io_start_time_ms=current['io_start_time_ms'],io_ready_time_ms=current['io_ready_time_ms'],
                ssd_complete_time_ms=current['ssd_complete_time_ms'],
                ssd_complete_timestamp_available=current['ssd_complete_time_ms'] is not None,
                budget_end_ms=previous['compute_end_ms'],io_elapsed_ms=current['io_ready_time_ms']-current['io_start_time_ms'],
                full_io_stall_ms=current['full_io_stall_ms'],warm_io_stall_ms=current['warm_io_stall_ms'],
                warm_queue_ms=current['warm_queue_ms'],cycle_compute_fraction_percent=100*c/period,
                fixed_CIR_read_time_reference_ms=max(current['V_per_ssu_GiB'])*1000/1.25,
                fixed_CIR_reference_is_actual_measurement=False,
                prior_role=previous['role'],following_role=current['role']))
    by_kind=defaultdict(float)
    for row in stalls:by_kind[row['kind']]+=row['warm_io_stall_ms']
    for n in npu:
        close(n['compute_ms']+n['io_stall_ms']+n['compute_queue_ms'],n['active_ms'])
        n['U_percent']=100*n['compute_ms']/duration
        n['idle_ms']=max(0.,duration-n['active_ms'])
        n['role_compute_ms']=dict(n['role_compute_ms']);n['role_active_ms']=dict(n['role_active_ms'])
        n['role_requests_computing']={role:len(ids) for role,ids in n['role_requests_computing'].items()}
        n['A_and_B_actually_computing']=all(n['role_compute_ms'].get(role,0)>0 for role in ('A','B'))
    demand,segments,overloads=demand_sweep(metrics,records,start,end)
    U=math.fsum(n['compute_ms'] for n in npu)/(32*duration)*100
    total_stall=math.fsum(n['io_stall_ms'] for n in npu)
    role_summary={role:dict(compute_card_ms=math.fsum(n['role_compute_ms'].get(role,0) for n in npu),
                            active_card_ms=math.fsum(n['role_active_ms'].get(role,0) for n in npu)) for role in ('A','B')}
    for row in role_summary.values():row['active_U_percent']=100*row['compute_card_ms']/row['active_card_ms'] if row['active_card_ms'] else None
    result_checks=[]
    for saved in result.get('analysis',[]):
        if saved['start_ms']==start and saved['end_ms']==end:
            close(U,saved['U_percent'])
            for n,value in zip(npu,saved['per_npu_U_percent']):close(n['U_percent'],value)
            for actual,expected in zip(demand['maximum_GiB_s_by_ssu'],saved['demand']['per_disk_max_GiB_s']):close(actual,expected)
            for actual,expected in zip(demand['mean_GiB_s_by_ssu'],saved['demand']['per_disk_mean_GiB_s']):close(actual,expected)
            if 'role_and_stall' in saved:
                for role in ('A','B'):
                    for n,value in zip(npu,saved['role_and_stall']['per_npu_role_compute_ms']):close(n['role_compute_ms'].get(role,0),value.get(role,0))
            result_checks.append('independent U/per-NPU U/per-role compute/per-disk demand matched saved analysis')
    audited=dict(window_ms=[start,end],fleet_U_percent=U,demand=demand,role_summary=role_summary,
        all_32_npus_have_A_and_B_compute=all(n['A_and_B_actually_computing'] for n in npu),
        total_io_stall_card_ms=total_stall,io_stall_card_ms_by_kind=dict(by_kind),
        io_stall_loss_pp_by_kind={kind:value/(32*duration)*100 for kind,value in by_kind.items()},
        cross_request_wait_card_ms_by_role_transition=dict(transition_wait),
        idle_card_ms=math.fsum(n['idle_ms'] for n in npu),
        compute_queue_card_ms=math.fsum(n['compute_queue_ms'] for n in npu),
        saved_analysis_comparisons=result_checks,
        SSD_completion_times_available_for_cycles=sum(c['ssd_complete_timestamp_available'] for c in cycles),
        per_npu=npu,
        worst_internal_cycles=sorted([c for c in cycles if c['same_request'] and c['warm_io_stall_ms']>1e-10],key=lambda c:c['warm_io_stall_ms'],reverse=True)[:top],
        worst_cross_request_cycles=sorted([c for c in cycles if not c['same_request'] and c['warm_io_stall_ms']>1e-10],key=lambda c:c['warm_io_stall_ms'],reverse=True)[:top])
    return audited,segments,overloads,cycles,stalls


def markdown(audit):
    a=audit['analysis'];d=a['demand'];start,end=a['window_ms']
    out=[f"# {audit['case']}：独立机制核查",'',
         f"窗口固定为[{start/1000:g},{end/1000:g})秒；整机U={a['fleet_U_percent']:.6f}%。逐盘最大参考需求为"+
         ', '.join(f'{x:.6f}' for x in d['maximum_GiB_s_by_ssu'])+' GiB/s。',
         f"逐时逐盘严格欠载：{d['strict_underload_all_disks']}；32卡全窗活跃：{d['all_npus_active']}；32卡都实际计算过A和B：{a['all_32_npus_have_A_and_B_compute']}。",'',
         '|等待来源|窗口卡毫秒|对应整机U损失百分点|','|---|---:|---:|']
    for kind,value in a['io_stall_card_ms_by_kind'].items():out.append(f"|{kind}|{value:.6f}|{a['io_stall_loss_pp_by_kind'][kind]:.6f}|")
    out.extend(['',f"跨请求L0等待按前后角色分解：{a['cross_request_wait_card_ms_by_role_transition']}。计算排队{a['compute_queue_card_ms']:.6f}卡ms，窗口空闲{a['idle_card_ms']:.6f}卡ms。",'',
        '|角色|实际计算卡ms|请求驻留卡ms|角色内部U|','|---|---:|---:|---:|'])
    for role,row in a['role_summary'].items():
        u=f"{row['active_U_percent']:.6f}%" if row['active_U_percent'] is not None else '无该角色驻留'
        out.append(f"|{role}|{row['compute_card_ms']:.6f}|{row['active_card_ms']:.6f}|{u}|")
    out.extend(['',f"窗口平均同时驻留A的卡数={d['A_active_count_time_average']:.6f}，B={d['B_active_count_time_average']:.6f}；A瞬时卡数范围[{d['A_active_count_min']},{d['A_active_count_max']}]。这是按真实驻留时间计算，不是请求条数占比。"])
    out.extend(['','## 最差同请求内部周期','',
        '起点是上一层开始计算，终点是本层开始计算；等待为上一层计算结束后、本层数据到齐前的暴露等待。','',
        '|NPU|角色|请求/待读层|周期起止ms|当前C ms|下一层逐盘MiB|io_ready ms|完整等待ms|窗内等待ms|',
        '|---:|---|---|---|---:|---|---:|---:|---:|'])
    for c in a['worst_internal_cycles']:
        out.append(f"|{c['npu_id']}|{c['role']}|{c['request_id']}/L{c['layer']}|{c['start_ms']:.3f}→{c['end_ms']:.3f}|{c['current_compute_C_ms']:.3f}|"+
                   '/'.join(f'{v*1024:.3f}' for v in c['next_V_per_ssu_GiB'])+f"|{c['io_ready_time_ms']:.3f}|{c['full_io_stall_ms']:.3f}|{c['warm_io_stall_ms']:.3f}|")
    out.extend(['','## 证据能够说明什么','',
        '这份脚本独立从实际计算时刻求U、从manifest落盘量和admission/completion事件求逐盘需求，未调用runner的metrics函数。短计算时间本身不是充分原因：需要同时看读取量、上一层的预取预算、实际IO到齐时刻，以及是否跨请求。',
        '同请求内部周期满足完整周期供给=b=V/(C+等待)，故b/B=C/(C+等待)；这里的b来自完整周期工作量守恒，不是独立保存的逐块速率测量，也不是瞬时利用率。跨请求周期读取下一请求，不能直接用当前请求V/C作这个比值。',
        '固定CIR读取时间列只是max(V_s/1.25)参考。OD会借用空闲份额，所以不能拿它冒充实际读取时间或根据C最小值就断言缺带宽。',
        'io_ready表示该层数据已通过NPU链路到齐；通常原result未记录各层SSD最后服务完成时刻，本审计把该字段留空，不与io_ready混用。没有逐块SSD/Path/group服务日志时，不能仅凭这些时序就断言某一盘、某条Path或两级WRR是直接瓶颈。',
        '逐盘需求超40是实测结果，不会作为审计失败或被删除；详见overload_intervals.csv。跨请求等待详见worst_cross_request_cycles.csv；每卡A/B实际计算、驻留与U详见per_npu.csv。',
        '',f"源result SHA256：`{audit['source_sha256']['result']}`；manifest SHA256：`{audit['source_sha256']['manifest']}`。",''])
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--command',type=Path)
    parser.add_argument('--start-ms',type=float,default=2000.)
    parser.add_argument('--end-ms',type=float,default=4000.)
    parser.add_argument('--top',type=int,default=16)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if not args.run_dir and not (args.result and args.manifest):parser.error('provide --run-dir or both --result and --manifest')
    result_path=(args.result or args.run_dir/'result.json.gz').resolve()
    manifest_path=(args.manifest or args.run_dir/'manifest.json.gz').resolve()
    command_path=(args.command or result_path.parent/'command.json').resolve()
    sources={'result':result_path,'manifest':manifest_path,'data':ROOT/'data'}
    if command_path.exists():sources['command']=command_path
    initial={key:sha(path) for key,path in sources.items()}
    result,manifest,command,metrics,records,raw_count=load_raw(result_path,manifest_path,command_path)
    analysis,segments,overloads,cycles,stalls=analyze(result,metrics,records,args.start_ms,args.end_ms,args.top)
    assert initial=={key:sha(path) for key,path in sources.items()}
    target=args.output_dir or HERE/'mechanism_audit'/f'{result_path.parent.name}_{args.start_ms:g}_{args.end_ms:g}'
    target.mkdir(parents=True,exist_ok=True)
    audit=dict(all_checks_passed=True,case=result_path.parent.name,policy=result['strategy'],
               source_sha256=initial,source_paths={key:str(path) for key,path in sources.items()},
               audit_script_sha256=sha(__file__),sources_unchanged=True,
               request_count=len(records),requests_matching_raw_data=raw_count,
               no_simulation_started=True,no_runner_or_metrics_imported=True,analysis=analysis)
    (target/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    csv_write(target/'per_npu.csv',analysis['per_npu'])
    csv_write(target/'demand_segments.csv',segments)
    csv_write(target/'overload_intervals.csv',overloads)
    csv_write(target/'cycles.csv',cycles)
    csv_write(target/'stall_intervals.csv',stalls)
    csv_write(target/'worst_internal_cycles.csv',analysis['worst_internal_cycles'])
    csv_write(target/'worst_cross_request_cycles.csv',analysis['worst_cross_request_cycles'])
    (target/'README.md').write_text(markdown(audit))
    print(json.dumps(dict(case=audit['case'],U_percent=analysis['fleet_U_percent'],
                         demand_max=analysis['demand']['maximum_GiB_s_by_ssu'],
                         stall_ms_by_kind=analysis['io_stall_card_ms_by_kind'],
                         all_32_both_roles=analysis['all_32_npus_have_A_and_B_compute'],output=str(target)),ensure_ascii=False))


if __name__=='__main__':main()
