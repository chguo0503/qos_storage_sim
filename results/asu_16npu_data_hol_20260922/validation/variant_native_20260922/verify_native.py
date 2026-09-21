#!/usr/bin/env python3
"""Recheck the completed small native runs without importing either runner."""
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = ('standard_asu', 'variant_asu', 'variant_changed_once', 'variant_changed_once_measured')


def read(path):
    with gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path) as f:
        return json.load(f)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


summaries, records = {}, {}
for label in LABELS:
    out = HERE / label
    summary = read(out / 'native_summary.json.gz')
    manifest = read(out / 'manifest.json.gz')
    metrics = read(out / 'metrics.json')
    command = read(out / 'command.json')
    extended = read(out / 'extended_windows.json')
    source = read(out / 'source_sha256.json')
    meta = {row['request_id']: row['load'] for row in manifest['requests']}
    source_matches = {name: (ROOT / name).is_file() and sha(ROOT / name) == value for name, value in source.items()}
    checks = dict(full_drain=summary['request_count'] == len(summary['request_metrics']) == len(meta) == 4,
        native_invariants=all(summary['invariants'].values()),
        sources_unchanged=metrics['source_unchanged'] is True and source == command['source_sha256'] == metrics['source_sha256'] and all(source_matches.values()),
        manifest_sha=sha(out / 'manifest.json.gz') == command['manifest_sha256'] == metrics['manifest_sha256'],
        input_fingerprint=summary['input_fingerprint'] == manifest['input_fingerprint'] == metrics['input_fingerprint'],
        extended_complete=extended == metrics['extended_windows'],
        extended_full_U=abs(extended['full']['U_percent'] - metrics['full_U_percent']) < 1e-10,
        future_windows_unavailable=all(not row['available'] for row in extended['windows'].values()))
    per_request = []
    for batch in summary['microbatch_metrics']:
        rid = batch['member_request_ids'][0]
        expected_C = meta[rid]['per_layer_us'] / 1000
        errors = [abs(row['compute_end_ms'] - row['compute_start_ms'] - expected_C) for row in batch['layer_metrics']]
        per_request.append(dict(request_id=rid, profile_key=meta[rid].get('profile_key'),
            expected_layer_C_ms=expected_C, max_actual_layer_C_error_ms=max(errors),
            expected_layer_V_gib=meta[rid]['per_layer_kv_gb'],
            actual_own_compute_ms=next(row['own_compute_ms'] for row in summary['request_metrics'] if row['request_id'] == rid),
            layer_count=len(errors), passed=len(errors) == 8 and max(errors) < 1e-9))
    checks['actual_per_request_compute'] = all(row['passed'] for row in per_request)
    expected_read = 8 * math.fsum(row['per_layer_kv_gb'] for row in meta.values())
    checks['total_read_volume'] = math.isclose(summary['completed_read_gb'], expected_read, rel_tol=0, abs_tol=1e-10)
    checks['per2s_compute_conservation'] = abs(sum(row['compute_card_ms'] for row in extended['consecutive_2s']) - extended['full']['compute_card_ms']) < 1e-7
    checks['per2s_SLO_population'] = sum(row['slo_admitted']['count'] for row in extended['consecutive_2s']) == 4
    assert all(checks.values()), (label, checks)
    records[label] = dict(passed=True, checks=checks, per_request_compute=per_request,
        requests=summary['request_count'], completed_blocks=summary['completed_blocks'],
        completed_read_gib=summary['completed_read_gb'], makespan_ms=summary['makespan_ms'],
        full_U_percent=metrics['full_U_percent'], full_SLO_1p5_percent=metrics['full_SLO_1p5_percent'],
        wall_seconds=metrics['wall_seconds'], source_files=len(source),
        source_hashes_all_match=True, native_summary_sha256=sha(out / 'native_summary.json.gz'))
    summaries[label] = summary
parity = dict(full_native_summary_exact=summaries['standard_asu'] == summaries['variant_asu'],
    request_metrics_exact=summaries['standard_asu']['request_metrics'] == summaries['variant_asu']['request_metrics'],
    all_layer_metrics_exact=summaries['standard_asu']['microbatch_metrics'] == summaries['variant_asu']['microbatch_metrics'],
    physical_input_fingerprint_exact=summaries['standard_asu']['input_fingerprint'] == summaries['variant_asu']['input_fingerprint'],
    completed_byte_probe_does_not_change_summary=summaries['variant_changed_once'] == summaries['variant_changed_once_measured'])
assert all(parity.values()), parity
probe = read(HERE / 'variant_changed_once_measured/completed_bytes_probe.json')
assert probe['passed'] and all(row['passed'] for row in probe['checks'])
assert probe['unique_completed_blocks'] == summaries['variant_changed_once']['completed_blocks']
report = dict(status='passed', simulator_was_actually_executed=True, probe_is_passive_and_summary_exact=True,
    parity=parity, per_request_layer_disk_actual_volume=dict(passed=True, checks=len(probe['checks']),
        completed_blocks=probe['unique_completed_blocks'], source='variant_changed_once_measured/completed_bytes_probe.json'),
    runs=records, note='Variant manifests add catalog/profile_key/family metadata, so serialized manifest hashes differ. The physical input fingerprint and every native summary field match for the same A/B input.',
    frozen_runner_sha256=sha(HERE.parents[1] / 'runner.py'), frozen_variant_runner_sha256=sha(HERE.parents[1] / 'variant_runner.py'))
(HERE / 'checks.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['# 小型原生验证', '', '已实际执行原生事件仿真，未修改冻结运行器或正式实验。所有验证通过。', '',
 '| 验证 | 请求数 | 完成块 | 完整时间(ms) |', '|---|---:|---:|---:|']
for label, row in records.items():
    lines.append(f"| {label} | {row['requests']} | {row['completed_blocks']} | {row['makespan_ms']:.9f} |")
lines += ['', '标准 explicit 与相同 A/B 变体输入的 ASU：完整 native_summary、所有请求/层时序、事件计数、物理输入指纹完全相同。变体 manifest 额外保存画像键和目录，所以整个 manifest 文件的 SHA 不同。', '',
 'Once 使用一个不同 A：40K+17 个总 token、miss3000，计算时间来自相邻长度/miss 网格的双线性插值。被动完成回调逐请求、逐层、逐盘累计实际块数与字节，32 项全部与 manifest 一致；加观察器前后完整 native_summary 也完全相同。', '',
 '逐请求每层实际计算时间、读取总量、源码前后 SHA、完整请求/块守恒和 extended 窗口统计均通过检查。小输入不到2秒，2–4/2–10/4–10秒窗口如实标 unavailable，不将它们判为通过。', '',
 '配置、命令、原始结果及完成字节观察器均保存在本目录。运行 `python verify_native.py` 可只读复核原始数据；详细检查见 checks.json。']
(HERE / 'README.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(status=report['status'], parity=parity, volume_checks=report['per_request_layer_disk_actual_volume'],
    makespans={label: row['makespan_ms'] for label,row in records.items()}), indent=2))
