#!/usr/bin/env python3
"""Compare conditional layer-cycle arithmetic with independently audited logs."""
from pathlib import Path
import gzip
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def read(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def main():
    rows = []
    for path in sorted((HERE / 'runs').glob('*/*/analysis.json')):
        a = read(path)
        assert a['all_technical_checks_passed']
        case = path.parent
        m = read(case / 'manifest.json.gz')['metadata']
        raw = read(case / 'result.json.gz')
        profiles = dict(zip(m['role_names'], m['profiles']))
        counts = dict(zip(m['role_names'], m['count_ratio']))
        C = {r: p['per_layer_compute_us'] / 1000 for r, p in profiles.items()}
        V = {r: p['per_layer_kv_gib'] for r, p in profiles.items()}
        W_C = math.fsum(counts[r] * C[r] for r in counts)
        f = math.fsum(counts[r] * C[r] for r in a['short_roles']) / W_C
        rho = 32 * math.fsum(counts[r] * V[r] for r in counts) * 1000 / (m['num_ssu'] * 40 * W_C)
        assert math.isclose(rho, m['ideal_load_ratio'], abs_tol=1e-12)
        long = next(w for w in a['windows'] if w['start_ms'] == 2000 and w['end_ms'] == 20000)
        cycles = long['complete_internal_cycles_inside_window']['per_role']
        waits = {r: cycles[r]['stall_ms_including_zeros']['mean'] for r in counts}
        # This deliberately uses observed waits. It is explanatory, not a forecast.
        short_wait_cost = math.fsum(counts[r] * waits[r] for r in a['short_roles'])
        all_wait_cost = math.fsum(counts[r] * waits[r] for r in counts)
        estimate_short = 100 * W_C / (W_C + short_wait_cost)
        estimate_all = 100 * W_C / (W_C + all_wait_cost)
        full_U = 100 * raw['summary']['fleet_npu_compute_utilization']
        full_SSD = 100 * raw['summary']['ssd_mean_utilization']
        identity_error = full_SSD - rho * full_U
        assert abs(identity_error) < 1e-7, (case, identity_error)
        n_short = sum(counts[r] for r in a['short_roles'])
        # For several short profiles this is a count-weighted mean-wait target.
        target_wait = .25 * W_C / n_short
        actual_short_wait = short_wait_cost / n_short
        rows.append(dict(candidate=m['candidate'], num_ssu=m['num_ssu'], seed=m['seed'],
            strategy=a['strategy'], profile_keys=m['profile_keys'], count_ratio=m['count_ratio'],
            C_ms=C, V_GiB=V, B_GiB_s={r: V[r]*1000/C[r] for r in counts},
            short_roles=a['short_roles'], ideal_load_ratio=rho, ideal_short_compute_fraction=f,
            observed_internal_wait_by_role_ms=waits,
            count_weighted_observed_short_wait_ms=actual_short_wait,
            short_mean_wait_needed_for_U80_ms=target_wait,
            conditional_U_short_wait_only_percent=estimate_short,
            conditional_U_all_internal_wait_percent=estimate_all,
            observed_long_window_U_percent=long['U_percent'],
            approximation_error_all_internal_pp=estimate_all-long['U_percent'],
            full_finite_fleet_U_percent=full_U, full_finite_SSD_busy_percent=full_SSD,
            full_finite_conservation_error_pp=identity_error,
            warm_long_short_mixed_cards=a['windows'][0]['long_short_mixed_card_count'],
            long_long_short_mixed_cards=long['long_short_mixed_card_count'],
            analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(definition='Observed internal-layer waiting explains utilization under fixed completion mix; not independent prediction.',
        caveats=['The early continuous-layer approximation treats all eight layers as identical internal cycles; finite requests have seven internal cycles plus a separate first-layer handoff.',
                 'First-layer/handoff waiting and finite-window composition are excluded from the approximation.',
                 'Whole-finite-batch SSD_busy=rho*U uses a common makespan denominator; do not apply it directly to arbitrary windows.',
                 'All selected seeds are retained. Per-disk instantaneous underload is a separate test.'], rows=rows)
    (HERE / 'math_result_table.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 从输入公式到实际等待：全案例核对', '',
        '这里用已经测到的等待解释利用率，**不是独立预测**。所有数字都能追溯到同目录的 manifest、result、analysis。', '',
        '```text', '输入纯计算权重 f = 短请求总计算时间 / 全部请求总计算时间',
        '若长期完成比例接近输入比例，且长层无等待：',
        'U ≈ 1 / (1 + f * 短层平均额外等待 / 短层计算时间)',
        'U≈80% 所需平均短层等待 = 0.25 * 全部纯计算工作量 / 短层数量',
        '```', '',
        '此表保留早期80%筛选阈值和连续层近似。用户最终目标为长期80几；[89%/85%目标与有限8层修正](goal_80s_update.md)另列，避免事后改写预测。', '',
        '平均等待包括零等待层。这里把每层都视为相同内部周期，忽略首层交接和窗口边界。本次实际只有7次内部预取，应使用 `U≈8ΣnC/(8ΣnC+7Σn内部等待+Σn首层等待)`；不能把8次内部等待再加一遍首层。实测 U 始终直接从时间线核算。', '',
        '| 配方 | 盘数 | 策略 | seed | f | U80需等待 ms | 实测短等待 ms | 代入内部等待估算 U | 实际长窗 U |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        label = 'Baseline' if r['strategy'] == 'baseline' else '流量分配策略'
        lines.append(f"| {r['candidate']} | {r['num_ssu']} | {label} | {r['seed']} | {100*r['ideal_short_compute_fraction']:.2f}% | {r['short_mean_wait_needed_for_U80_ms']:.3f} | {r['count_weighted_observed_short_wait_ms']:.3f} | {r['conditional_U_all_internal_wait_percent']:.2f}% | {r['observed_long_window_U_percent']:.2f}% |")
    lines += ['', '最后两列接近，说明内部层等待能解释主要损失；不接近时，需要检查首层交接、长短完成比例和统计窗边界，而不是改写实测 U。', '',
        '另按完整有限批次核验 `平均SSD忙率 = rho * 整机U`，全部使用同一个结束时间作分母；每例误差均小于 1e-7 个百分点。它是读取工作量守恒，不是说任何两秒窗都满足这个等式。', '',
        '[完整数值及来源哈希](math_result_table.json) · [全部窗口和SLO](comparison.md)']
    (HERE / 'math_result_table.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(verified_cases=len(rows))))


if __name__ == '__main__':
    main()
