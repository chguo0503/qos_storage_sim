#!/usr/bin/env python3
"""Read-only source-data and ring-placement screening; no simulation execution."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulator.core import sim

NPUS, DISKS, LAYERS, CAPACITY = 32, 3, 8, 40.0
CIR = CAPACITY / NPUS
SAMPLE_IDS = 2048
BLOCK_GIB = 176 * 1024 / 2**30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ring_weights():
    positions, disks = sim._block_hash_ring(DISKS)
    weights = [0, 0, 0]
    for index, (position, disk) in enumerate(zip(positions, disks)):
        weights[disk] += (position - positions[index - 1]) % (1 << 256)
    assert sum(weights) == 1 << 256
    return [value / (1 << 256) for value in weights]


def sampled_profiles(table):
    profiles = []
    for (length, miss), (source_bw, compute_us, source_ttft_ms, volume) in sorted(table.items()):
        if length * 1024 < 10 * 1024:
            continue
        hit = length * 1024 - miss
        assert math.isclose(volume, hit / 128 * BLOCK_GIB, abs_tol=1e-12)
        c = compute_us / 1e6
        assert math.isclose(volume / c, source_bw, rel_tol=1e-12)
        assert math.isclose(source_ttft_ms, 78 * compute_us / 1000, abs_tol=1e-8)
        profiles.append(dict(key=f"{length}:{miss}", total_K=length, miss_tokens=miss,
                             hit_tokens=hit, C_ms=compute_us / 1000, V_GiB=volume,
                             V_MiB=volume * 1024, B_total_GiB_s=volume / c,
                             B_balanced_per_disk_GiB_s=volume / (3 * c),
                             source_ttft_78_layers_ms=source_ttft_ms,
                             experiment_pure_compute_8_layers_ms=LAYERS * compute_us / 1000,
                             category=sim.classify_request(length, miss)))
    needed = sorted({p["hit_tokens"] // 128 for p in profiles} |
                    {math.ceil(p["hit_tokens"] / 128) for p in profiles})
    checkpoint_index = {n: index for index, n in enumerate(needed)}
    counters = np.zeros((len(needed), SAMPLE_IDS, DISKS), dtype=np.int32)
    for rid in range(SAMPLE_IDS):
        running = np.zeros(3, dtype=np.int32)
        for block in range(max(needed)):
            running[sim.block_ring_hash_disk_id(rid, block, DISKS)] += 1
            if block + 1 in checkpoint_index:
                counters[checkpoint_index[block + 1], rid] = running
    rates = {}
    for p in profiles:
        exact_blocks = p["hit_tokens"] / 128
        lower, upper = math.floor(exact_blocks), math.ceil(exact_blocks)
        counts = counters[checkpoint_index[lower]].astype(float)
        if upper != lower:
            counts += (counters[checkpoint_index[upper]] - counters[checkpoint_index[lower]]) * (exact_blocks - lower)
        values = counts * BLOCK_GIB / (p["C_ms"] / 1000)
        rates[p["key"]] = values
        p.update(sample_min_B_per_ssu_GiB_s=values.min(axis=0).tolist(),
                 sample_mean_B_per_ssu_GiB_s=values.mean(axis=0).tolist(),
                 sample_max_B_per_ssu_GiB_s=values.max(axis=0).tolist(),
                 sample_argmax_request_id_by_ssu=values.argmax(axis=0).tolist(),
                 sample_max_any_disk_GiB_s=float(values.max()),
                 all_sampled_per_disk_B_at_most_CIR=bool(np.all(values <= CIR)))
    return profiles, rates


def cohort_peak(rates_a, rates_b, k):
    """64 disjoint 32-ID cohorts, 32 cyclic choices of which k cards run A."""
    peak = np.zeros(3)
    for offset in range(0, SAMPLE_IDS, NPUS):
        aa, bb = rates_a[offset:offset + NPUS], rates_b[offset:offset + NPUS]
        base, delta = bb.sum(axis=0), aa - bb
        for start in range(NPUS):
            indices = [(start + j) % NPUS for j in range(k)]
            value = base + delta[indices].sum(axis=0)
            peak = np.maximum(peak, value)
    return peak


def pair_analysis(a, b, rates, weights, *, control=False):
    ca, cb = a["C_ms"], b["C_ms"]
    ba, bb = a["B_balanced_per_disk_GiB_s"], b["B_balanced_per_disk_GiB_s"]
    hypotheses = []
    for k in range(8, 17):
        expected = (k * a["B_total_GiB_s"] + (NPUS - k) * b["B_total_GiB_s"]) * np.asarray(weights)
        sampled = cohort_peak(rates[a["key"]], rates[b["key"]], k)
        envelope = k * np.asarray(a["sample_max_B_per_ssu_GiB_s"]) + (NPUS - k) * np.asarray(b["sample_max_B_per_ssu_GiB_s"])
        theta = min(1., bb / CIR)
        ua = 1 - theta * max(0., 1 - CIR / ba)
        fleet = 1 - k / NPUS * (1 - ua)
        count_ratio_no_wait = k / (NPUS - k) * cb / ca
        count_ratio_fluid = count_ratio_no_wait * ua
        run_lengths = sorted({max(1, math.floor(count_ratio_fluid)), max(1, round(count_ratio_fluid)),
                              max(1, math.ceil(count_ratio_fluid)), max(1, round(count_ratio_no_wait))})
        hypotheses.append(dict(k_A_active_target=k, k_B_active_target=NPUS-k,
            expected_ring_D_per_ssu_GiB_s=expected.tolist(),
            sampled_cohort_peak_D_per_ssu_GiB_s=sampled.tolist(),
            independent_sample_max_envelope_D_per_ssu_GiB_s=envelope.tolist(),
            expected_D_below_39_0=bool(expected.max() < 39),
            sampled_cohorts_all_below_39_5=bool(sampled.max() < 39.5),
            finite_ID_envelope_all_below_40=bool(envelope.max() < 40),
            fluid_B_burst_time_fraction=theta, fluid_A_U_percent=100*ua,
            fluid_fleet_U_percent=100*fleet, fluid_loss_pp=100*(1-fleet),
            A_to_B_request_count_ratio_no_wait=count_ratio_no_wait,
            A_to_B_request_count_ratio_fluid_corrected=count_ratio_fluid,
            short_request_run_length_candidates=run_lengths,
            long_requests_per_cycle=1,
            estimated_cycle_pure_compute_ms_by_run={str(r): LAYERS*(r*ca+cb) for r in run_lengths}))
    eligible = [h for h in hypotheses if h["sampled_cohorts_all_below_39_5"] and h["expected_D_below_39_0"]]
    chosen = max(eligible, key=lambda h:h["fluid_loss_pp"]) if eligible else hypotheses[0]
    # Concrete boundary micro-test: B request IDs0..31 are all released together.
    # Disk capacity lower-bounds the LAST NPU's completion, not the average NPU.
    b_disk_bytes = b["V_GiB"] * np.asarray(weights)
    cohort_b_disk_bytes = rates[b["key"]][:32].sum(axis=0) * cb / 1000
    all_32_b_read_ms_lb = 1000 * float(cohort_b_disk_bytes.max()) / CAPACITY
    link_read_ms_lb = 1000 * b["V_GiB"] / sim.NPU_BW_LIMIT
    boundary_wait_ms_lb = max(0., max(all_32_b_read_ms_lb, link_read_ms_lb) - ca)
    alternating_u_reference = 100 * LAYERS * (ca+cb) / (LAYERS*(ca+cb)+boundary_wait_ms_lb)
    return dict(A=a["key"], B=b["key"], control=control,
                C_B_over_C_A=cb/ca, V_B_over_V_A=b["V_GiB"]/a["V_GiB"],
                B_read_burst_ms_balanced=1000*b["V_GiB"]/(DISKS*CIR),
                B_read_burst_ms_expected_bottleneck=1000*float(b_disk_bytes.max())/CIR,
                A_compute_ms=ca, B_compute_ms=cb,
                long_request_residence_pure_ms=LAYERS*cb,
                recommended=chosen, alternatives=hypotheses,
                conservative_k_values_for_any_assignment_of_sampled_IDs=[h['k_A_active_target'] for h in hypotheses if h['finite_ID_envelope_all_below_40']],
                synchronized_all_32_A_to_B_boundary_read_lower_bound_ms=all_32_b_read_ms_lb,
                boundary_microtest_B_request_ids=list(range(32)),
                boundary_microtest_actual_B_GiB_per_ssu=cohort_b_disk_bytes.tolist(),
                synchronized_A_to_B_boundary_wait_lower_bound_ms=boundary_wait_ms_lb,
                synchronized_A1_B1_similar_wait_U_reference_percent=alternating_u_reference,
                both_profiles_all_sampled_per_disk_B_at_most_CIR=(a["all_sampled_per_disk_B_at_most_CIR"] and b["all_sampled_per_disk_B_at_most_CIR"]))


def report(result):
    profiles = {p["key"]:p for p in result["profiles"]}
    out = ["# 原始data画像筛选：OD在逐盘参考欠载下的候选输入", "",
           "这是输入筛选和近似数学分析，不是仿真结果。未修改data、核心模拟器、原实验输入或旧图。所有候选来自原始data行，未缩放；最短总输入32K，满足≥10K。", "",
           "## 先说能够支持和不能支持的结论", "",
           "OD每盘32条独占Path的CIR为40/32=1.25 GiB/s。需要区分：当前请求参考需求欠载，不能自动证明所有跨请求预取都有足够截止时间；也不能把CIR当成始终不变的实际供给，空闲份额仍会借用。", "",
           "在下述简化的同步大读取模型中，主要候选通常只产生约几个到8个百分点的平均利用率损失，不能据此许诺80几的长期U。若实测更差，应分别核对跨请求L0读取、两级WRR借用、多盘完成barrier、相位漂移和边界，而不是把差值全部归给均分。", "",
           "## 1. 画像的实际含义和单位", "",
           "A是短计算、高参考带宽；B是长计算、较大读取、低参考带宽。这里A/B是本页角色，不是已有其他实验中固定不变的名字。B=V/C中的B表示带宽，不表示角色B。", "",
           "```text", "V = 当前请求每层全部读取量（GiB）", "C = 每层纯计算时间（秒）",
           "参考带宽 = V/C；逐盘需求 = 该层实际落盘GiB/C", "OD每盘每卡保证份额 q = 40/32 = 1.25 GiB/s", "```", "",
           "data中的第二列是每层微秒；第三列TTFT等于78层纯计算，不直接作为本实验8层请求耗时。K=1024 token；miss是token数量，不是百分比。", "",
           "|角色候选|总输入|miss|每层读取MiB|每层计算ms|总B GiB/s|假设均分每盘B|采样最忙单卡盘B|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = sorted({p[role] for p in result["recommended_pairs"] + result["controls"] for role in ("A","B")}, key=lambda x:tuple(map(int,x.split(':'))))
    for key in keys:
        p=profiles[key]
        out.append(f"|{key}|{p['total_K']}K|{p['miss_tokens']}|{p['V_MiB']:.3f}|{p['C_ms']:.3f}|{p['B_total_GiB_s']:.3f}|{p['B_balanced_per_disk_GiB_s']:.3f}|{p['sample_max_any_disk_GiB_s']:.3f}|")
    out.extend(["", "## 2. 优先验证的十对输入", "",
        "k表示某一时刻执行A画像的卡数目标，不是永久绑卡。每卡循环运行若干A后一个B，并错开各卡相位，才能让每张卡都出现两种请求。推荐k要求：固定ring期望最忙盘<39.0、下述有限队列样本的最忙盘<39.5。它不是任意真实运行的硬保证，真实manifest和warm事件仍必须逐点审计。", "",
        "|A / B（长度K:miss）|C比 B/A|V比 B/A|目标k(A/B卡数)|样本最忙盘峰值|简化模型U|建议A连跑条数，再B×1|", "|---|---:|---:|---:|---:|---:|---|"])
    for p in result["recommended_pairs"]:
        r=p['recommended']
        out.append(f"|{p['A']} / {p['B']}|{p['C_B_over_C_A']:.2f}|{p['V_B_over_V_A']:.2f}|{r['k_A_active_target']}/{r['k_B_active_target']}|{max(r['sampled_cohort_peak_D_per_ssu_GiB_s']):.3f}|{r['fluid_fleet_U_percent']:.2f}%|{', '.join(map(str,r['short_request_run_length_candidates']))}|")
    out.extend(["", "## 3. 为什么会损失，以及这个预测的限制", "",
        "假设B卡的下一层读取同时开始，开始时32条Path都繁忙，每卡每盘暂时拿q。A需要的速率a>q，因此在这段期间反复读不及；B计算长、需求b<q，读完后会释放份额，A在剩余期间可恢复满速。假设均匀三盘、同一时期始终k张A卡、其余B卡，忽略层粒度与边界：", "",
        "```text", "B卡读取繁忙时间 t_busy ≈ V_B,s / q", "B一个计算周期中繁忙比例 theta ≈ b/q",
        "A在繁忙阶段U≈q/a，在其余阶段U≈1", "A平均U≈1−(b/q)*(1−q/a)",
        "整机U≈1−(k/32)*(b/q)*(1−q/a)", "逐盘参考欠载要求 k*a+(32−k)*b < 40", "```", "",
        "这个表达式是用于排序候选的流体近似，不是严格上下界。它要求C_B显著大于C_A、B同步、A不改变角色；真正8层有限请求会跨边界、错相并产生角色驻留反馈。独立Random可能自然减少同步争用，也可能某时刻A卡过多直接过载。", "",
        "运行中要特别看：总U下降发生在A层内部，还是A最后层切到B首层；没有这个分解，就无法证明是同一个原因。", "",
        "## 4. 请求数量比例不是时间占用比例", "",
        "```text", "忽略等待，A:B请求数≈r:1", "r = k/(32−k) * C_B/C_A", "若A平均利用率为U_A、B约1：", "r ≈ k/(32−k) * C_B/C_A * U_A", "```", "",
        "A被阻塞后同一请求在卡上驻留更久，会使实际A卡数量超过按纯计算比例估计的k。表中的多个A连跑长度包含这两种估计。8层在时间比例里抵消，但一个请求的绝对纯计算时长仍是8C。比如A32K/1024与B200K/4096，单个B约1130.69ms纯算，单个A约58.06ms纯算，不能按请求条数1:1理解成一半时间执行A。", "",
        "为保证所有卡长期都运行A/B，可循环A^r B并轮换初始相位。要保证逐时欠载，仍要针对运行产生的真实admission/completion扫描每盘需求；只看平均k、整机总量或输入比例都不够。", "",
        "## 5. Ring hash风险与采样范围", "",
        f"固定三盘ring的精确哈希空间占比为：{', '.join(f'{w*100:.6f}%' for w in result['ring_space_weights'])}。因此V/3不是每盘精确值。某一层的盘间分配还会围绕这些占比波动，且同一请求各层复用相同分盘，不会靠层数自动平均消失。", "",
        "本页枚举request_id=0..2047的真实block_ring_hash分配；每画像保留各盘最小/均值/最大。组合检验用64个连续32-ID组，并枚举32种循环A卡位置，共2048个群组/每个k。另给出k*max(A)+(32−k)*max(B)的更保守有限ID包络，详见JSON。", "",
        "若要求在这2048个ID范围内连最坏画像组合都不超40，可先测更保守的k：A32K/1024+B200K/4096用k≤8；A48K/1024+B200K/4096用k≤8；A32K/1024+B160K/4096或B128K/4096用k≤9。这仍要求真实运行同刻A卡数不超过该k，且ID确实在所审计范围内。", "",
        "有限样本的最大值不是所有未来ID的数学最大值。若完全不限制request_id，理论上块可高度集中；必须对最终manifest精确检查每盘，而不能把这些样本数字当绝对保证。随机独立选卡使k本身波动，且闭环等待反馈会进一步改变它。", "",
        "## 6. 每卡每盘B≤1.25的边界对照", "",
        "以下两画像在所枚举的每个ID、每个盘上都≤1.25。若最终manifest也满足这个条件，那么只看当前请求V/C时，任意组合的32卡逐盘需求都≤40；不依赖同刻A卡数。它适合检验跨请求预取，而不是A层内部的保证份额不足。", "",
        "|A / B|C比 B/A|两画像样本最大逐盘B|32卡同步A→B全部读完下界ms|至少一张卡的边界等待下界ms|假设相近等待的A1B1参考U|", "|---|---:|---:|---:|---:|---:|"])
    for p in result['controls']:
        peak=max(profiles[p['A']]['sample_max_any_disk_GiB_s'],profiles[p['B']]['sample_max_any_disk_GiB_s'])
        out.append(f"|{p['A']} / {p['B']}|{p['C_B_over_C_A']:.2f}|{peak:.3f}|{p['synchronized_all_32_A_to_B_boundary_read_lower_bound_ms']:.3f}|{p['synchronized_A_to_B_boundary_wait_lower_bound_ms']:.3f}|{p['synchronized_A1_B1_similar_wait_U_reference_percent']:.2f}%|")
    out.extend(["", "边界表明确使用request_id=0..31这32个B的真实ring分盘，并假设32卡在A最后层同时发出B首层。最忙盘全部字节/40给出全部读完的下界，所以至少一张卡要等待；它不是平均每张卡等待的下界。最后一列另外假设各卡等待相近，仅为同步边界参考，不能当严格U上限或整个混合实验的U预测。", "",
        "32K/2048虽然平均每盘仅0.934 GiB/s，但2048个ID采样中存在单盘超过1.25的尾部，不能作为无需核验的对照。JSON另保留32K/2048、48K/2048的条件候选；只有最终manifest逐请求逐盘通过≤1.25才满足这项对照条件。", "",
        "关键区别：此时要隐藏的是V_next，预算却是C_current；应比较V_next/C_current，不能拿当前请求V_current/C_current证明跨请求也一定读得及。同步切换时所有B首层同发是一种真实额外时间压力。随着卡相位自然错开，边界等待可能缩小；是否长期重现需要真模拟。", "",
        "## 7. 建议记录而不混淆的证据", "",
        "- 逐时逐盘当前请求需求最大值、超40时长及精确区间；warm每卡都出现A/B的驻留比例。",
        "- 每卡计算时间、层内部IO等待、跨请求L0等待分别求和；全程和多个固定warm窗口同时看。",
        "- 按A/B分别记录U、每盘完整层周期供给、OD Path/group忙闲；CIR只是保证额，不能代替实际服务。",
        "- 原始数据、manifest身份、分盘必须在策略间一致；禁止重排后改变request_id从而更换hash。",
        "", f"数据SHA256：`{result['source_sha256']['data']}`。完整画像、所有k=8..16备选、原始精度见[profile_analysis.json](profile_analysis.json)。", ""])
    return '\n'.join(out)


def main():
    sources = {name:sha(ROOT/name) for name in ('data','sim.py','continuous_batch_sim.py')}
    table = ast.literal_eval((ROOT/'data').read_text())
    profiles, rates = sampled_profiles(table)
    index = {p['key']:p for p in profiles}
    weights = ring_weights()
    targets = [('32:1024','200:4096'),('48:1024','200:4096'),('64:1024','200:4096'),
               ('80:1024','200:4096'),('32:1024','160:4096'),('48:1024','160:4096'),
               ('64:1024','160:4096'),('32:1024','128:4096'),('48:1024','192:4096'),('96:1024','200:4096')]
    pairs = [pair_analysis(index[a],index[b],rates,weights) for a,b in targets]
    controls = [pair_analysis(index[a],index[b],rates,weights,control=True)
                for a,b in [('32:4096','200:4096'),('32:4096','160:4096'),('48:4096','200:4096')]]
    conditional_controls = [pair_analysis(index[a],index[b],rates,weights,control=True)
                for a,b in [('32:2048','200:4096'),('48:2048','200:4096')]]
    assert all(p['both_profiles_all_sampled_per_disk_B_at_most_CIR'] for p in controls)
    assert sources == {name:sha(ROOT/name) for name in sources}
    result = dict(source_sha256=sources, script_sha256=sha(__file__), no_simulation_run=True,
                  npu_count=32, ssu_count=3, layers=8, capacity_per_ssu_GiB_s=40,
                  OD_per_npu_per_ssu_CIR_GiB_s=1.25, raw_profile_count=len(profiles),
                  sampled_request_ids=[0,SAMPLE_IDS-1], ring_space_weights=weights,
                  profiles=profiles, recommended_pairs=pairs, controls=controls,
                  conditional_controls_requiring_manifest_per_disk_CIR_check=conditional_controls,
                  cautions=['Model values are approximations, not simulated utilization.',
                            'Finite-ID hash samples do not certify arbitrary future manifests.',
                            'Current-request V/C underload does not bound cross-request V_next/C_current.',
                            'Request count ratio is not wall-time occupancy ratio; stalls change occupancy.'])
    (HERE/'profile_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'profile_analysis.md').write_text(report(result))
    print(json.dumps(dict(raw_profiles=len(profiles),pairs=len(pairs),controls=len(controls),
                         ring_weights=weights,outputs=['profile_analysis.json','profile_analysis.md']),ensure_ascii=False))


if __name__=='__main__':
    main()
