#!/usr/bin/env python3
"""Render all fixed-window runs, including invalid and non-winning cases."""
import gzip
import hashlib
import json
from pathlib import Path


def _load_analysis(directory):
    path = directory / 'analysis.json.gz'
    if not path.exists():
        path = directory / 'analysis.json'
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rb') as handle:
        payload = handle.read()
    return path, json.loads(payload), hashlib.sha256(payload).hexdigest()


BASE = Path(__file__).resolve().parent
ANALYSIS_PATH, data, _ = _load_analysis(BASE)
assert not data['errors'], data['errors']
paired = {}
for pair in data['order_comparisons']:
    for strategy in pair['strategies']:
        if strategy.get('status') == 'complete':
            paired[pair['ordered_label'], strategy['strategy']] = strategy

lines = [
    '**全部结果：固定 [2000,4000) ms；32 NPU / 6 SSU**', '',
    f'由 `build_overview.py` 从 `{ANALYSIS_PATH.name}` 生成。利用率单位为 %，相对下降以同人口随机 Baseline 为参照；负数代表重排后提高。', '',
    '“有效”要求源码/输入审计、主窗暖机与每卡混合、全运行逐事件名义容量检查全部通过。不能只看利用率一列。不同种子可能共享同一输入指纹，仍分别保留各自提交种子的运行。', '',
    '| 输入标签 | 策略 | 设备 U | 请求等权 U | 相对随机下降 | 全程逐盘峰值 GiB/s | 暖窗/混合 | 欠载 | 有效 |',
    '|---|---|---:|---:|---:|---:|---|---|---|',
]
accepted = 0
for run in data['runs']:
    window = run['windows'][0]
    scan = run['nominal_demand_scan']['full_run']
    capacity = scan['all_ssu_within_capacity'] and scan['all_npu_links_within_capacity']
    valid = run['audit']['passed'] and run['main_window_valid'] and capacity
    accepted += int(valid)
    comparison = paired.get((run['label'], run['strategy']))
    delta = '—'
    if comparison:
        reduction = comparison['windows'][0]['device_utilization']['relative_reduction_vs_random']
        delta = f'{100 * reduction:.4f}%'
    yes = lambda value: '通过' if value else '**失败**'
    lines.append(
        f"| [{run['label']}]({run['path']}) | {run['strategy']} | "
        f"{100 * window['device_utilization']:.4f} | {100 * window['request_equal_utilization']:.4f} | "
        f"{delta} | {scan['max_ssu_gib_s']:.6f} | {yes(run['main_window_valid'])} | {yes(capacity)} | {yes(valid)} |"
    )
lines += ['', f"共 {len(data['inputs'])} 份输入 manifest，{len(data['runs'])} 次策略运行，{accepted} 次满足全部主窗与名义容量条件。源码/执行审计通过 {data['counts']['audit_passed']} 次。", '',
          '仅 `concurrency_l768_seed7__exact_cohort4_p1` 的 Baseline 同时满足所有条件与相对随机降低至少 10% 的目标。所有结果并不构成生产分布的独立随机样本。', '',
          f'[主报告](report.md) · [详细审计]({ANALYSIS_PATH.name}) · [全部 250 ms 子窗口及指标](summary.csv)', '']
(BASE / 'all_results.md').write_text('\n'.join(lines))
print(json.dumps({'runs':len(data['runs']), 'accepted':accepted, 'output':str(BASE / 'all_results.md')}))
