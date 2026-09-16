#!/usr/bin/env python3
"""Audit raw profiles and predict load from pure-compute residence weights.

This is not an SSD simulation. Monte Carlo observations are independent draws
from the ideal stationary renewal distribution, without stalls or correlations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from authenticated_workload_inputs import load_authenticated_bw_table
import sim

LENGTHS = (32, 64, 80, 128, 160, 200)
MISSES = (256, 1024, 2048, 4096)
BLOCK_GIB = 176 * 1024 / 2**30
RECIPES = {
    "semi_low": (1, 1, 1, 2),
    "semi_middle": (1, 2, 2, 2),
    "semi_near": (1, 1, 1, 1),
    "full_mild": (1, 2, 1, 1),
    "full_middle": (1, 4, 1, 1),
    "full_primary": (3, 2, 1, 1),
    "full_strong": (4, 4, 1, 1),
}


def profile_rows():
    table, provenance = load_authenticated_bw_table(32)
    rows = []
    for (length, miss), (bandwidth, compute_us, ttft_ms, volume) in sorted(table.items()):
        cached = length * 1024 - miss
        exact = cached % 128 == 0
        derived = volume / (compute_us / 1e6)
        assert abs(derived - bandwidth) <= 1e-8 * max(1, bandwidth)
        rows.append(dict(profile_id=f"L{length}_M{miss}", total_length_k=length,
                         miss_tokens=miss, miss_fraction=miss / (length * 1024),
                         cached_tokens=cached, category=sim.classify_request(length, miss),
                         layer_compute_ms=compute_us / 1000, layer_read_gib=volume,
                         layer_read_mib=volume * 1024, nominal_B_gib_s=derived,
                         source_ttft_78_layers_ms=ttft_ms, exact_176kib_blocks=exact,
                         chosen_diverse_catalog=(length in LENGTHS and miss in MISSES)))
    return rows, provenance


def analyze_recipe(name, weights, rows, samples=100000, seed=20260916):
    selected = [r for r in rows if r["chosen_diverse_catalog"]]
    counts = np.array([weights[MISSES.index(r["miss_tokens"])] for r in selected], dtype=float)
    computes = np.array([r["layer_compute_ms"] / 1000 for r in selected])
    volumes = np.array([r["layer_read_gib"] for r in selected])
    probs = counts * computes / np.dot(counts, computes)
    per_card = np.dot(counts, volumes) / np.dot(counts, computes)
    rate = np.zeros((32, len(selected), 3))
    for npu in range(32):
        for p, row in enumerate(selected):
            blocks = row["cached_tokens"] // 128
            for disk in range(3):
                num = (blocks + 2 - ((disk - npu) % 3)) // 3
                rate[npu, p, disk] = num * BLOCK_GIB / computes[p]
    rng = np.random.default_rng(seed)
    demands = []
    for offset in range(0, samples, 4096):
        chosen = rng.choice(len(selected), size=(min(4096, samples-offset), 32), p=probs)
        demands.append(rate[np.arange(32)[None, :], chosen].sum(axis=1))
    demands = np.concatenate(demands)
    categories = sorted({r["category"] for r in selected})
    return dict(name=name, counts_per_miss=dict(zip(map(str, MISSES), weights)),
                distinct_profiles=int(sum(counts > 0)), requests_per_deck=int(counts.sum()),
                pure_compute_ms_per_deck=float(np.dot(counts, computes) * 8 * 1000),
                fluid_fleet_nominal_gib_s=float(32 * per_card), fluid_rho=float(32 * per_card / 120),
                fluid_disk_nominal_gib_s=(rate * probs[None, :, None]).sum(axis=(0,1)).tolist(),
                ideal_request_count_share_by_category={c: float(sum(counts[i] for i,r in enumerate(selected) if r["category"] == c) / sum(counts)) for c in categories},
                ideal_compute_time_share_by_category={c: float(sum(probs[i] for i,r in enumerate(selected) if r["category"] == c)) for c in categories},
                ideal_mc_samples=samples, ideal_mc_seed=seed,
                ideal_mc_fraction_over_40_each_disk=(demands > 40).mean(axis=0).tolist(),
                ideal_mc_fraction_all_disks_over_40=float((demands > 40).all(axis=1).mean()),
                ideal_mc_disk_quantiles_01_50_99=np.quantile(demands, [0.01,0.5,0.99], axis=0).tolist(),
                caveat="Ideal independent pure-compute residence samples only; not measured overload fractions. Stalls extend residence, and layer feedback creates correlations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    args = parser.parse_args()
    rows, provenance = profile_rows()
    with (HERE / "profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    results = [analyze_recipe(name, weights, rows, args.samples) for name, weights in RECIPES.items()]
    output = dict(source=provenance, analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  total_raw_profiles=len(rows), exact_block_profiles=sum(r["exact_176kib_blocks"] for r in rows),
                  selected_lengths_k=LENGTHS, selected_misses=MISSES, scenarios=results,
                  formulas={"fluid_fleet_demand": "32 * sum(profile_count * per_layer_V) / sum(profile_count * per_layer_C)",
                            "ideal_residence_probability": "profile_count * C / sum(profile_count * C)"})
    (HERE / "workload_analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 多画像输入的数学设计", "", "本文件由analyze_data.py生成。只做原始数据审计和理想负载估计，没有运行SSD仿真。", "",
             "- 直接使用data中的计算时间和读取量，不缩放计算、不外推长度。",
             "- data包含84个画像；miss=64的12行需要半块读取。为保持现有策略的176KiB等长块约束，本次不使用这12行，不对它们填充。",
             "- 候选目录：总长度32/64/80/128/160/200K，miss=256/1024/2048/4096，共24种画像，覆盖SS、SL、LS、LL四类。SS/LS仅指总长度和miss类别，不直接代表带宽大小。",
             "- 每张卡使用同样的计数比例，但用seed+100003*npu独立打乱整个有限请求集合；不是反复播放同一小段随机顺序。", "",
             "```text", "预计每盘平均需求 = (32 / 3) × sum(数量 × V) / sum(数量 × C)",
             "理想时间占比 = 数量 × C / sum(数量 × C)", "```", "",
             "不能直接平均每个请求的V/C。长计算请求在时间轴上停留更久；实际有等待时，还应把停留时间改成计算加等待。预测不是过载证明，最终必须逐个接纳/完成事件积分当前请求的逐盘V/C。", "",
             "| 配方 | miss256:1024:2048:4096的每长度请求数 | 纯计算预测总需求 GiB/s | 预测/120 | 理想独立抽样三盘同时>40的占比 |", "|---|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['name']} | {':'.join(map(str, row['counts_per_miss'].values()))} | {row['fluid_fleet_nominal_gib_s']:.3f} | {row['fluid_rho']:.3f} | {100*row['ideal_mc_fraction_all_disks_over_40']:.2f}% |")
    lines += ["", "Monte Carlo中的占比是无I/O等待、各卡独立的理想时间分布，不能作为正式仿真结果。FIFO等待通常会拉长高带宽画像的驻留时间，使实际名义需求更高。", "",
              "首轮冻结输入选择semi_low的1:1:1:2，以及full_primary的3:2:1:1。其他行供校准参考。名字只是候选标签，实际事件审计未通过之前不能宣称已满足条件。full_strong用于更强压力，但代价是离接近容量的工作区更远。", "",
              "若需要不依赖运行结果的严格全状态过载保证，可只用miss<=1024：最小单卡B=5.7358GiB/s，32张卡全部有接纳请求时，精确条带后每盘最小总需求也超过40。但这排除了低带宽画像，会把理想负载明显提高；不作为首选。", "",
              "所有输入仍为t=0一次到达的有限队列，每卡持续取下一请求。它们用于固定L1/L2、比较L3，不代表真实在线到达分布。TTFT继续分清接纳后耗时和包含队列等待的端到端耗时。", ""]
    (HERE / "workload_plan.md").write_text("\n".join(lines), encoding="utf-8")
    for row in results:
        print(row["name"], f"fluid={row['fluid_fleet_nominal_gib_s']:.3f}", f"ideal_Pallover={row['ideal_mc_fraction_all_disks_over_40']:.5f}")


if __name__ == "__main__":
    main()
