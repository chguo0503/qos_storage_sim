#!/usr/bin/env python3
"""Collect independently audited Random cases without filtering bad seeds."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parent


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def main():
    records, pending, groups = [], [], defaultdict(list)
    for command_path in sorted((HERE/'runs').glob('*/*/command.json')):
        case = command_path.parent
        command = read(command_path)
        assert command['strategy'] in ('baseline', 'once') and command.get('order', 'random') == 'random'
        analysis_path = case/'analysis.json'
        if command['status'] != 'complete' or not analysis_path.exists():
            pending.append(dict(case=str(case.relative_to(HERE)), command_status=command['status'],
                                independently_analyzed=analysis_path.exists()))
            continue
        a = read(analysis_path)
        assert a['all_technical_checks_passed']
        assert a['builder_sha256'] == sha(HERE/'analyze.py')
        assert sha(case/'result.json.gz') == command['output_sha256']
        m = read(case/'manifest.json.gz')['metadata']
        assert a['input_fingerprint'] == m['input_fingerprint'] == command['input_fingerprint']
        for w in a['windows']:
            short_compute = sum(w['classes'][r]['compute_ms'] for r in a['short_roles'])
            short_active = sum(w['classes'][r]['active_ms'] for r in a['short_roles'])
            cycles = w['complete_internal_cycles_inside_window']['per_role']
            short_cycles = [cycles[r] for r in a['short_roles'] if r in cycles]
            layer_count = sum(c['count'] for c in short_cycles)
            short_stall = sum(c['total_stall_ms'] for c in short_cycles)
            short_C = sum(c['total_compute_ms'] for c in short_cycles)
            row = dict(case=str(case.relative_to(HERE)), candidate=m['candidate'], strategy=a['strategy'],
                       input_family=m['family'], constructed_profile=m.get('constructed_profile',False),
                       constructed_long_profile=m.get('constructed_long_profile',False),
                       order=a['order'], seed=a['seed'], num_ssu=a['num_ssu'],
                       window_start_s=w['start_ms']/1000, window_end_s=w['end_ms']/1000,
                       U_percent=w['U_percent'], all_32_active=w['all_32_active'],
                       long_short_mixed_cards=w['long_short_mixed_card_count'],
                       long_short_mixed_pass=w['long_short_mixed_pass'],
                       all_profile_mixed_cards=w['mixed_card_count'],
                       short_conditional_U_percent=100*short_compute/short_active if short_active else None,
                       short_window_card_time_share_percent=100*short_active/w['denominator_card_ms'],
                       short_complete_internal_cycles=layer_count,
                       short_mean_stall_ms_including_zeros=short_stall/layer_count if layer_count else None,
                       short_duration_weighted_internal_U_percent=100*short_C/(short_C+short_stall) if short_C+short_stall else None,
                       ideal_load_ratio=a['ideal_no_stall_population']['mean_demand_capacity_ratio'],
                       per_ssu_peak_demand_GiB_s=w['nominal']['per_ssu_peak_GiB_s'],
                       any_ssu_over_nominal_percent=w['nominal']['any_ssu_over_capacity_percent'],
                       nominal_within_disk_capacity=w['nominal']['within_disk_capacity'],
                       actual_ssd_mean_busy_percent=statistics.mean(w['physical']['per_ssu_utilization_percent']),
                       actual_ssd_total_GiB_s=w['physical']['total_mean_service_GiB_s'],
                       actual_npu_receive_total_GiB_s=w['physical']['total_mean_received_GiB_s'],
                       slo_1p5_percent=w['slo']['alphas']['1.5']['admission']['percent'],
                       slo_2_percent=w['slo']['alphas']['2.0']['admission']['percent'],
                       slo_cohort_requests=w['slo']['admission_cohort_count'],
                       count_ratio=m['count_ratio'], profile_keys=m['profile_keys'],
                       per_card_profile_counts=m['per_npu_assignment'][0]['role_counts'],
                       completed_requests=a['completed_request_count'],
                       analysis_sha256=sha(analysis_path), input_fingerprint=a['input_fingerprint'])
            records.append(row)
            groups[(row['candidate'],row['num_ssu'],row['strategy'],row['window_start_s'],row['window_end_s'])].append(row)
    summaries = []
    metrics = ['U_percent','short_conditional_U_percent','short_mean_stall_ms_including_zeros',
               'actual_ssd_mean_busy_percent','slo_1p5_percent','slo_2_percent','any_ssu_over_nominal_percent']
    for key, rows in sorted(groups.items()):
        summaries.append(dict(candidate=key[0], num_ssu=key[1], strategy=key[2],
                              input_family=rows[0]['input_family'], constructed_profile=rows[0]['constructed_profile'],
                              constructed_long_profile=rows[0]['constructed_long_profile'],
                              start_s=key[3], end_s=key[4], seeds=[r['seed'] for r in rows],
                              seed_count=len(rows), all_active_pass_count=sum(r['all_32_active'] for r in rows),
                              long_short_mixed_pass_count=sum(r['long_short_mixed_pass'] for r in rows),
                              profile_keys=rows[0]['profile_keys'], count_ratio=rows[0]['count_ratio'],
                              ideal_load_ratio=rows[0]['ideal_load_ratio'],
                              metrics={k:describe([r[k] for r in rows]) for k in metrics}))
    result = dict(updated_utc=datetime.now(timezone.utc).isoformat(), builder_sha256=sha(Path(__file__)),
                  completed_analyzed_cases=len({r['case'] for r in records}), pending_or_unanalyzed=pending,
                  source='Only complete immutable results with independently verified analysis',
                  all_seeds_retained=True, no_ordered_runs=True, group_statistics='Equal seed weighting; sample standard deviation',
                  caveat='Different topology/profile groups are separate experiments; partial groups are explicitly counted',
                  rows=records, groups=summaries)
    (HERE/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if records:
        with (HERE/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# Baseline Random：已完成实验汇总','',
           f'独立审计完成 {result["completed_analyzed_cases"]} 个案例；另有 {len(pending)} 个已启动案例仍在运行或待审计。只有已完成案例进入下表。',
           '', '全部为独立 Random。单位为百分比；多种子结果为均值 ± 样本标准差。未按利用率或每卡混合是否通过剔除种子。', '',
           '| 配方 | 输入 | SSU | 策略 | 窗口 | 种子数 | 平均 NPU U | SLO×1.5 | SLO×2 | 每卡长短覆盖通过 |',
           '|---|---|---:|---|---|---:|---:|---:|---:|---:|']
    def fmt(s):
        return 'N/A' if s['mean'] is None else f'{s["mean"]:.2f}' + (f' ± {s["std"]:.2f}' if s['std'] is not None else '')
    for g in summaries:
        if (g['start_s'],g['end_s']) not in ((2.,4.),(2.,20.)):continue
        met=g['metrics'];label='Baseline' if g['strategy']=='baseline' else '流量分配策略'
        source='长短C外推' if g['constructed_long_profile'] else '短C外推' if g['constructed_profile'] else '原始data'
        lines.append(f'| {g["candidate"]} | {source} | {g["num_ssu"]} | {label} | {g["start_s"]:g}–{g["end_s"]:g}s | {g["seed_count"]} | {fmt(met["U_percent"])} | {fmt(met["slo_1p5_percent"])} | {fmt(met["slo_2_percent"])} | {g["long_short_mixed_pass_count"]}/{g["seed_count"]} |')
    lines += ['', 'SLO 为接纳后 prefill 完成代理，不含入卡前排队。每卡长短覆盖要求窗口内两类都有正计算时间；没有通过的种子仍在均值中，并明确报告。',
              '', '理想平均负载接近容量不表示逐盘逐时欠载。完整数值、实际盘忙时及名义需求超限比例见 [CSV](comparison.csv) 和 [JSON](comparison.json)。',
              '', '主组数学与实际等待的对应见 [公式核对](math_to_measurements.md)。新 22 秒参考队列的每卡配额为44A/88B；它与周六40A/80B的完整shuffle不同，不能称为同输入复现。']
    (HERE/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(completed_analyzed_cases=result['completed_analyzed_cases'],pending=len(pending)),ensure_ascii=False))


if __name__ == '__main__':
    main()
