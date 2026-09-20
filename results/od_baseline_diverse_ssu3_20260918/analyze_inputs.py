#!/usr/bin/env python3
"""Read the six frozen manifests; write only input_profiles.csv/input_math.md.

No simulator imports, simulation execution, input rewriting, or result reads.
All numerical fields are recomputed from request records and block placement;
manifest summaries are used only as independent consistency checks.
"""

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCENARIOS = ("semi", "full")
SEEDS = (7, 19, 43)
LAYERS, NPUS, SSUS = 8, 32, 3
DISK_GIB_S, IO_GIB = 40.0, 176 * 1024 / 2**30
FIELDS = (
    "total_input_Ki_tokens", "total_input_tokens", "miss_tokens", "hit_tokens",
    "category", "layer_read_MiB", "layer_compute_ms", "required_B_GiB_s",
    "semi_requests_per_npu", "full_requests_per_npu",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10), (actual, expected)


def analyze_manifest(path, profiles, quotas, data_sha):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    meta = doc["metadata"]
    scenario, seed = meta["scenario_candidate"], meta["seed"]
    assert scenario in SCENARIOS and seed in SEEDS
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (NPUS, SSUS, LAYERS)
    assert meta["layout"] == "block_ring_hash" and meta["order"] == "random"
    assert meta["source_data_sha256"] == data_sha
    counts = [Counter() for _ in range(NPUS)]
    compute = [[] for _ in range(NPUS)]
    work = [[[] for _ in range(SSUS)] for _ in range(NPUS)]
    placement_cache = {}
    request_ids = set()
    for request in doc["requests"]:
        rid, npu, load = request["request_id"], request["npu_id"], request["load"]
        assert rid not in request_ids
        request_ids.add(rid)
        assert 0 <= npu < NPUS and request["arrival_time_ms"] == 0
        assert not load["constructed_profile"]
        assert load["profile_construction"]["method"] == "direct_data_row"
        key = (int(load["seq_len_k"]), int(load["nql"]))
        counts[npu][key] += 1
        pindex = request["placement_index"]
        if pindex not in placement_cache:
            placement = doc["placements"][pindex]
            assert len(placement) == 1
            blocks = [Counter() for _ in range(SSUS)]
            for disk, volume in placement[0]:
                assert 0 <= disk < SSUS
                close(volume, IO_GIB)
                blocks[disk][volume] += 1
            placement_cache[pindex] = tuple(math.fsum(v * n for v, n in count.items()) for count in blocks)
        volumes = placement_cache[pindex]
        layer_gib = math.fsum(volumes)
        c_ms = float(load["per_layer_us"]) / 1000.
        bandwidth = layer_gib * 1000. / c_ms
        close(layer_gib, load["per_layer_kv_gb"])
        close(bandwidth, load["required_bw_input_gbps"])
        total_tokens = key[0] * 1024
        assert load["total_tokens"] == total_tokens
        assert load["ssd_prefix_tokens"] == total_tokens - key[1]
        close(layer_gib, (total_tokens - key[1]) / 128 * IO_GIB)
        row = dict(total_input_Ki_tokens=key[0], total_input_tokens=total_tokens,
                   miss_tokens=key[1], hit_tokens=total_tokens - key[1],
                   category=load["category"], layer_read_MiB=layer_gib * 1024.,
                   layer_compute_ms=c_ms, required_B_GiB_s=bandwidth)
        if key in profiles:
            assert profiles[key] == row, (key, profiles[key], row)
        else:
            profiles[key] = row
        compute[npu].append(LAYERS * c_ms)
        for disk in range(SSUS):
            work[npu][disk].append(LAYERS * volumes[disk])
    assert len(request_ids) == meta["request_count"]
    assert all(count == counts[0] for count in counts)
    if scenario in quotas:
        assert quotas[scenario] == counts[0]
    else:
        quotas[scenario] = counts[0]
    per_npu_compute = [math.fsum(values) for values in compute]
    per_npu_work = [[math.fsum(values) for values in disks] for disks in work]
    for c_ms in per_npu_compute:
        close(c_ms, meta["actual_pure_compute_ms_per_npu"])
    per_disk_demand = [math.fsum(1000. * per_npu_work[n][s] / per_npu_compute[n]
                                for n in range(NPUS)) for s in range(SSUS)]
    per_disk_work = [math.fsum(per_npu_work[n][s] for n in range(NPUS)) for s in range(SSUS)]
    all_work, all_demand = math.fsum(per_disk_work), math.fsum(per_disk_demand)
    shares = [v / all_work for v in per_disk_work]
    compute_card_seconds = math.fsum(per_npu_compute) / 1000.
    disk_finish_lower_bound_seconds = max(per_disk_work) / DISK_GIB_S
    whole_input_u_bound = min(1., compute_card_seconds / (NPUS * disk_finish_lower_bound_seconds))
    # Every NPU has exactly the same multiset of requests and pure compute.
    # This simplification is not valid for arbitrary unequal-C workloads.
    assert len(set(per_npu_compute)) == 1
    close(whole_input_u_bound, min(1., DISK_GIB_S / max(per_disk_demand)))
    aggregate_only_u_bound = min(1., compute_card_seconds /
                                 (NPUS * all_work / (SSUS * DISK_GIB_S)))
    close(all_demand, meta["time_weighted_fleet_nominal_gib_s"])
    close(all_demand / (SSUS * DISK_GIB_S), meta["fluid_rho"])
    for actual, expected in zip(per_disk_demand, meta["time_weighted_per_ssu_nominal_gib_s"]):
        close(actual, expected)
    return dict(scenario=scenario, seed=seed, request_count=len(request_ids),
                requests_per_npu=sum(counts[0].values()),
                pure_compute_ms_per_npu=per_npu_compute[0],
                per_disk_demand=per_disk_demand, total_demand=all_demand,
                per_disk_rho=[x / DISK_GIB_S for x in per_disk_demand],
                total_rho=all_demand / (SSUS * DISK_GIB_S),
                per_disk_work_gib=per_disk_work, per_disk_work_share=shares,
                reference_throughput_gib_s=DISK_GIB_S / max(shares),
                bottleneck_disk=max(range(SSUS), key=lambda s: shares[s]),
                compute_card_seconds=compute_card_seconds,
                disk_finish_lower_bound_seconds=disk_finish_lower_bound_seconds,
                whole_input_u_upper_bound=whole_input_u_bound,
                aggregate_only_u_upper_bound=aggregate_only_u_bound)


def format_report(rows, cases, hashes, data_sha):
    case_by_scenario = {}
    for case in cases:
        scenario = case["scenario"]
        if scenario in case_by_scenario:
            before = {k: v for k, v in case_by_scenario[scenario].items() if k != "seed"}
            after = {k: v for k, v in case.items() if k != "seed"}
            assert before == after, "Random order changed population or aggregate physical work"
        else:
            case_by_scenario[scenario] = case
    text = [
        "# 冻结输入与OD均分带宽：数学核对", "",
        "本页只读取6份冻结Ring hash输入，不读取运行结果，也不启动仿真。画像的读取量、计算时间来自manifest中的原始data画像；配比与Random顺序是构造输入，不代表生产流量分布。",
        "",
        "32张NPU、3张SSU，每盘40 GiB/s，8层；所有请求在0秒到达，每卡固定执行自身的Random队列。seed为7、19、43，同一场景各种子保留相同画像数量和总体落盘工作量，只改变顺序。K表示1024 token；miss列是未命中token数量，不是百分比。",
        "", "## 1. 每种请求真正需要多少带宽", "",
        "```text",
        "V = 一层需要从所有SSU读取的总量，单位GiB",
        "C = 一层纯计算时间，单位秒",
        "B = V / C，单位GiB/s",
        "```", "",
        "B回答的是：想把下一层读取藏在一层计算期间，大约需要多大的平均读取速率。它既不是实际供给，也不是外部请求到达速率。每层读取量按实际块清单求和，1块=176 KiB，对应128个命中token。",
        "",
        "| 总输入长度 | miss token | 原分类 | 每层读取 MiB | 每层计算 ms | B GiB/s | semi每卡数量 | full每卡数量 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(f"| {row['total_input_Ki_tokens']}K | {row['miss_tokens']} | {row['category']} | "
                    f"{row['layer_read_MiB']:.3f} | {row['layer_compute_ms']:.3f} | "
                    f"{row['required_B_GiB_s']:.3f} | {row['semi_requests_per_npu']} | {row['full_requests_per_npu']} |")
    text.extend([
        "", "CSV保留未四舍五入数值：[input_profiles.csv](input_profiles.csv)。每个画像均在32张卡出现；semi每卡30个请求、full每卡42个请求，并非给不同卡永久绑定不同画像。", "",
        "## 2. 整批输入的理论平均需求", "",
        "不能直接把24个B做算术平均，因为不同请求占用的纯计算时间不同。先对每张卡求总读取量与总纯计算时间之比，再把32张卡相加：", "",
        "```text",
        "每卡对盘s的理论平均需求 = 该卡全部8层在盘s的读取GiB / 该卡全部请求纯计算秒数",
        "D_s = 32张卡对盘s的上述需求之和",
        "整机理论需求 D = D_0 + D_1 + D_2",
        "逐盘负载比 rho_s = D_s / 40；整机负载比 rho = D / 120",
        "```", "",
        "这是没有I/O等待时，按纯计算时间加权的参考需求。加入等待会改变某个时刻和warm窗口里正在运行的画像，因此它不等于仿真的逐时需求或warm平均需求。", "",
        "| 场景 | 每卡请求数 | 总请求数 | 每卡纯计算秒数 | 整机理论需求 GiB/s | rho |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for scenario in SCENARIOS:
        c = case_by_scenario[scenario]
        text.append(f"| {scenario} | {c['requests_per_npu']} | {c['request_count']} | "
                    f"{c['pure_compute_ms_per_npu']/1000:.6f} | {c['total_demand']:.6f} | {c['total_rho']:.6f} |")
    text.extend(["", "| 场景 | D_0 GiB/s | D_1 GiB/s | D_2 GiB/s | rho_0 | rho_1 | rho_2 |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
    for scenario in SCENARIOS:
        c = case_by_scenario[scenario]
        text.append(f"| {scenario} | " + " | ".join(f"{v:.6f}" for v in c["per_disk_demand"] + c["per_disk_rho"]) + " |")
    text.extend([
        "", "semi的整批平均需求低于容量，仍可有局部突发；full的整批平均需求高于容量，也不能仅凭该平均数就宣称任意时刻都过载。实际的逐时超载比例必须另从运行时窗口检查。", "",
        "## 3. 为什么三盘额定120，也不必同时供给120", "",
        "Ring hash并不强制三盘字节数完全相等。令f_s为完整输入中分到盘s的字节占比，若持续按这个工作比例完成输入：", "",
        "```text",
        "f_s = 盘s全部读取字节 / 三盘全部读取字节",
        "该工作比例下的总吞吐参考上限 R_ref = min_s(40 / f_s) = 40 / max_s(f_s)",
        "```", "",
        "| 场景 | 盘0字节占比 | 盘1字节占比 | 盘2字节占比 | 最忙盘 | R_ref GiB/s |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for scenario in SCENARIOS:
        c = case_by_scenario[scenario]
        shares = " | ".join(f"{100*v:.4f}%" for v in c["per_disk_work_share"])
        text.append(f"| {scenario} | {shares} | {c['bottleneck_disk']} | {c['reference_throughput_gib_s']:.6f} |")
    text.extend([
        "", "该上限适用于完整输入的固定工作比例，是磁盘容量给出的必要限制，不承诺实际吞吐一定达到它。短窗口可以先完成不同的字节比例，不能把R_ref当成每个2秒窗口的硬上限。NPU接收链路、计算依赖和请求边界也可能让实际吞吐更低。", "",
        "每层还需要等涉及的所有盘都读完，这就是多盘完成barrier：两盘早完成，第三盘的最后几个块没到齐，NPU仍不能开始该层计算。在近似恒定服务份额下，读取时间可粗略看成`max_s(V_s / b_s)`，还需考虑链路和提交延迟；只看三盘供给相加，可能掩盖某一盘拖尾。", "",
        "## 4. 完整有限输入的NPU利用率硬上限", "",
        "下面讨论从0秒开始、直到最后一个请求完成的**全输入平均利用率**，不讨论中间warm窗口。设total_V_s是全部请求、全部8层在盘s上的读取GiB，total_compute是全部请求的纯计算卡秒。无论如何选Path，盘s都必须把自己的字节读完：", "",
        "```text",
        "t_final >= max_s(total_V_s / 40)",
        "U_full = total_compute / (32 * t_final)",
        "U_full <= min(1, total_compute / (32 * max_s(total_V_s / 40)))",
        "```", "",
        "本实验每张卡的请求画像数量相同，所以每卡总纯计算时间C_card也相同。此时total_compute=32*C_card，且D_s=total_V_s/C_card，可以化简：", "",
        "```text",
        "U_full <= min(1, 40 / max_s(D_s))",
        "        = min(1, 1 / max_s(rho_s))",
        "```", "",
        "| 场景 | 全部纯计算卡秒 | 磁盘容量给出的t_final下界（秒） | 逐盘约束的全程U上限 | 只看总容量的较松上限 |",
        "|---|---:|---:|---:|---:|",
    ])
    for scenario in SCENARIOS:
        c = case_by_scenario[scenario]
        text.append(f"| {scenario} | {c['compute_card_seconds']:.6f} | "
                    f"{c['disk_finish_lower_bound_seconds']:.6f} | "
                    f"{100*c['whole_input_u_upper_bound']:.4f}% | {100*c['aggregate_only_u_upper_bound']:.4f}% |")
    text.extend([
        "", "只看总容量会使用`min(1, 120 / D)`，忽略字节分布不均。取最忙盘的`max_s`更严格：其他盘剩余的容量不能替最忙盘读取已固定放在它上面的块。semi的100%仅表示磁盘容量推导没有给出低于100%的约束，不表示实际能达到100%；首层读取、链路、有限输入起止和等待都可能进一步降低利用率。", "",
        "full的全程U最多约62.78%，这是对任何策略都适用的容量上限；因此评估全程提升空间应看当前全程U距离该上限还有多少，不能预设能靠选路把全程U提高到90%。这个上限本身也不保证可达。", "",
        "**它不是warm `[2,4)`秒的上限。** warm可能恰好包含较多计算、而把必要读取或其他等待留在窗口外，所以阶段利用率可以高于62.78%。同样不能把这个数当warm的预测值或目标值。仅给出24种画像的B，也不能推出任意窗口的真实计算比例。", "",
        "## 5. OD均分了CIR，为什么不保证SLO", "",
        "```text",
        "每卡每盘CIR = 40 / 32 = 1.25 GiB/s",
        "每卡跨3盘保证份额之和 = 3 * 1.25 = 3.75 GiB/s",
        "本实验SLO判定：完成时间 - 接纳时间 <= 1.5 * (8 * C)",
        "```", "",
        "但B从约1.315到29.657 GiB/s不等。比如200K、miss=1024的每层读取273.625 MiB，计算35.443 ms，B约7.539 GiB/s；32K、miss=4096则只需约1.315 GiB/s。同样的保证份额对这两类请求意味着不同的覆盖程度。", "",
        "**B高于3.75不等于必定不达标。** OD的PIR没有硬限，活跃Path能借用空闲带宽；借用沿用组间、组内两级WRR，实际供给b未必是3.75，也未必在每张活跃卡之间完全相同。加上首层可能在接纳前已预取、SLO允许1.5倍耗时、请求接续时的计算画像不同，必须查看实际层时序与完整请求耗时才能判断是否达标。B低于3.75同样不单独保证达标，因为还存在首层等待、逐盘工作量不均衡与多盘barrier。", "",
        "OD对NPU身份均分保证份额；它没有根据当前请求B或剩余deadline重新配置CIR。因此独占Path与请求SLO公平是不同目标，不能预先承诺利用率和SLO都会提高。", "",
        "## 6. b/B怎样与利用率联系", "",
        "不要拿某个瞬间的SSU速率除以B，当作该瞬间的NPU利用率。某卡正在计算时，读取可能已经结束；某卡正在等I/O时，盘也可能正在全速给它传输。", "",
        "在同一请求的一个完整内部层周期，C固定，令P为连续两层计算开始之间的间隔；这一周期预取下一层的完整V，把没有接收的时间也算进平均值：", "",
        "```text",
        "P = C + 暴露在计算外的等待时间",
        "b_cycle = V / P",
        "b_cycle / B = (V/P) / (V/C) = C/P",
        "```", "",
        "在这些限定下，C/P正是这个完整内部周期的计算比例。若改用固定窗口供给、瞬时b、不同请求之间的周期或有边界截断的周期，该等式不能直接照搬，只能结合周期模型作近似解释。整机warm利用率仍应按`窗口内全部真实计算卡时间 / (32 * 窗口长度)`计算；不能把不同长度周期的b/B简单等权平均。", "",
        "## 来源与复算", "",
        "```bash",
        "python results/od_baseline_diverse_ssu3_20260918/analyze_inputs.py",
        "```", "",
        "脚本只写本页和input_profiles.csv。它逐请求核验24种画像跨6份输入一致、每张卡配比一致、读取块量与V一致、B与manifest原始字段一致、全输入理论需求与metadata一致；读取前后验证manifest和data的SHA没有变化。", "",
        f"原始data SHA256：`{data_sha}`。", "",
        "| 冻结输入 | SHA256 |", "|---|---|",
    ])
    for path, digest in hashes.items():
        text.append(f"| [{path.name}](inputs/{path.name}) | `{digest}` |")
    text.extend(["", "上述数值完全由输入推导，不包含尚未完成的策略结果或跨种子性能结论。", ""])
    return "\n".join(text)


def main():
    paths = [HERE / "inputs" / f"{scenario}_seed{seed}_ring_hash.json.gz"
             for scenario in SCENARIOS for seed in SEEDS]
    hashes = {path: sha(path) for path in paths}
    data_sha = sha(ROOT / "data")
    profiles, quotas = {}, {}
    cases = [analyze_manifest(path, profiles, quotas, data_sha) for path in paths]
    assert len(profiles) == 24
    rows = [{**profiles[key], **{f"{s}_requests_per_npu": quotas[s][key] for s in SCENARIOS}}
            for key in sorted(profiles)]
    report = format_report(rows, cases, hashes, data_sha)
    assert all(sha(path) == digest for path, digest in hashes.items())
    assert sha(ROOT / "data") == data_sha
    with (HERE / "input_profiles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "input_math.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(manifests_checked=len(paths), profiles=len(rows),
                         output_files=["input_profiles.csv", "input_math.md"],
                         source_files_unchanged=True,
                         cases=[{k:v for k,v in c.items() if k not in ("per_disk_work_gib",)} for c in cases]),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
