#!/usr/bin/env python3
"""Explain three completed pilots using raw compute intervals and fixed math."""

import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("tight256", "main512", "main1024")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    with gzip.open(p, "rt") as stream:
        return json.load(stream)


def clip(a, b, lo, hi):
    return max(0., min(b, hi) - max(a, lo))


def analyse(name, theory):
    directory = HERE / "runs" / f"{name}_ssu3_h22000_seed7" / "baseline"
    result, manifest = read(directory/"result.json.gz"), read(directory/"manifest.json.gz")
    command = json.loads((directory/"command.json").read_text())
    physical = json.loads((directory/"physical_service.json").read_text())
    assert command["completed_simulation"] and command["returncode"] == 0
    assert command["output_sha256"] == sha(directory/"result.json.gz")
    assert command["manifest_sha256"] == sha(directory/"manifest.json.gz")
    assert command["physical_service_sha256"] == sha(directory/"physical_service.json")
    assert all(result["summary"]["invariants"].values())
    requests = {q["request_id"]: q for q in manifest["requests"]}
    batches = result["summary"]["microbatch_metrics"]
    assert all(b["batch_size"] == 1 for b in batches)
    Cs = next(p["C_ms"] for p in theory["profiles"] if p["label"] == "S")
    f = theory["victim_pure_compute_fraction"]
    rho = manifest["metadata"]["ideal_load_ratio"]
    windows = []
    for lo, hi in ((2000, 4000), (2000, 20000)):
        Ctotal = math.fsum(clip(x["compute_start_ms"],x["compute_end_ms"],lo,hi)
                          for b in batches for x in b["layer_metrics"])
        U = Ctotal/(32*(hi-lo))
        groups = {role:dict(count=0,D_ms=0.,C_ms=0.) for role in ("L","S")}
        seen = [set() for _ in range(32)]
        active = [0.]*32
        for batch in batches:
            role = requests[batch["member_request_ids"][0]]["load"]["role"]
            active[batch["npu_id"]] += clip(batch["admission_time_ms"],batch["completion_time_ms"],lo,hi)
            if any(clip(x["compute_start_ms"],x["compute_end_ms"],lo,hi)>0 for x in batch["layer_metrics"]):
                seen[batch["npu_id"]].add(role)
            for cur,nxt in zip(batch["layer_metrics"],batch["layer_metrics"][1:]):
                a,b = cur["compute_start_ms"],nxt["compute_start_ms"]
                if a>=lo and b<=hi:
                    g=groups[role];g["count"]+=1;g["D_ms"]+=b-a;g["C_ms"]+=cur["compute_end_ms"]-a
        for g in groups.values():
            g["mean_stall_ms_all_complete_internal_layers"]=(g["D_ms"]-g["C_ms"])/g["count"]
            g["weighted_b_over_B"]=g["C_ms"]/g["D_ms"]
        w=groups["S"]["mean_stall_ms_all_complete_internal_layers"]
        proxy=1/(1+f*w/Cs)
        busy=math.fsum(sum(row[lo//2000:hi//2000]) for row in physical["ssd_busy_ms"])/(3*(hi-lo))
        actual_summary=next(w for w in result["windows"] if w["start_ms"]==lo and w["end_ms"]==hi)
        assert math.isclose(U,actual_summary["mean_npu_utilization"],abs_tol=1e-10)
        windows.append(dict(window_ms=[lo,hi],actual_U_percent=100*U,
                            all32_admitted_active=all(math.isclose(x,hi-lo,abs_tol=1e-7)for x in active),
                            mixed_compute_npus=sum(s=={"L","S"}for s in seen),
                            missing_mixed_compute_npus=[i for i,s in enumerate(seen)if s!={"L","S"}],
                            internal_groups=groups,proxy_U_from_ideal_f_and_internal_w_percent=100*proxy,
                            actual_ssu_busy_percent=100*busy,
                            window_busy_minus_whole_population_rho_times_U_pp=100*(busy-rho*U)))
    full_C=math.fsum(x["compute_end_ms"]-x["compute_start_ms"]for b in batches for x in b["layer_metrics"])
    expected_C=8*math.fsum(q["load"]["per_layer_us"]/1000 for q in requests.values())
    full_V=8*math.fsum(q["load"]["per_layer_kv_gb"]for q in requests.values())
    T=result["summary"]["makespan_ms"]
    full_U=full_C/(32*T);full_busy=full_V/(120*T/1000)
    assert math.isclose(full_C,expected_C,abs_tol=1e-6)
    assert math.isclose(full_busy,rho*full_U,abs_tol=1e-10)
    assert math.isclose(full_busy,result["summary"]["ssd_mean_utilization"],abs_tol=1e-10)
    return dict(candidate=name,seed=7,rho_ideal=rho,ideal_f_short=f,C_short_ms=Cs,
                required_mean_short_stall_for_U80_ms=theory["common_victim_wait_ms_for_target_U"]["0.8"],
                windows=windows,full_finite=dict(makespan_ms=T,U_percent=100*full_U,ssu_busy_percent=100*full_busy,
                  rho_times_U_percent=100*rho*full_U,identity_error_pp=100*(full_busy-rho*full_U)),
                source={str((directory/n).relative_to(HERE)):sha(directory/n)
                        for n in ("manifest.json.gz","result.json.gz","physical_service.json","command.json")})


def main():
    math_path=HERE/"candidate_math.json"
    theory={c["id"]:c for c in json.loads(math_path.read_text())["candidates"]}
    cases=[analyse(name,theory[name])for name in CASES]
    report=dict(status="independently_recomputed",scope="Three initial seed7 Baseline pilots; no new simulations",
                source_math_sha256=sha(math_path),generator_sha256=sha(Path(__file__)),cases=cases)
    (HERE/"math_to_measurements.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    lines=["# 公式对得上，为什么 Random 仍然没有降到 80%？", "",
           "三个已完成的 seed 7 初筛都没有达到整机 U≈80%。原因是实际出现的平均等待远小于达到该目标所需要的等待；不能把“队列够长就会卡住”误当成“随机输入一定会形成足够长的队列”。", "",
           "所有请求都直接来自 data。配置为 32 NPU、3 SSU（每盘 40 GiB/s）、8 层、每卡独立完整随机。这里展示初筛，不代表四个确认种子的结论。", "",
           "## 先写目标条件，再对照实测", "", "```text", "rho = 32*sum(n*V) / [120*sum(n*C)]",
           "f_short = sum(短类 n*C) / sum(所有类 n*C)",
           "如果长类不 stall，短类平均每层额外等 w：",
           "U ≈ 1 / (1 + f_short*w/C_short)",
           "要 U≈80%，需要 w≈0.25*C_short/f_short", "```", "",
           "w 对所有短层取平均，包括不等的层。公式假设长期完成比例稳定、忽略启动与请求边界；它给出需要多少等待，不是承诺实际会等这么久。", "",
           "| 候选 | rho | f_short | C_short ms | U80 所需平均等 ms | 实测长窗内部平均等 ms | 代入近似公式 U | 实际长窗 U |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in cases:
        w=c["windows"][1]
        lines.append(f'| {c["candidate"]} | {c["rho_ideal"]:.6f} | {100*c["ideal_f_short"]:.3f}% | {c["C_short_ms"]:.6f} | {c["required_mean_short_stall_for_U80_ms"]:.6f} | {w["internal_groups"]["S"]["mean_stall_ms_all_complete_internal_layers"]:.6f} | {w["proxy_U_from_ideal_f_and_internal_w_percent"]:.4f}% | {w["actual_U_percent"]:.4f}% |')
    lines += ["", "长窗为 [2,20) 秒。所有实测平均等待采用同请求、完整落在窗口内的内部周期；跨请求与截断周期不混入这个均值。实际 U 则按全部真实计算段裁剪统计，包含边界，因此两列不必完全相等。", "",
              "tight256 的短类完整内部周期 b/B 只有约 0.6145，说明短类确实在等。然而其纯计算贡献只有约 9.32%，实测平均只多等约 1.253 ms，远低于需要的 5.355 ms，所以整机长窗 U 仍约 94.64%。这是“某类明显受损，却未使整机平均很低”的实测例子。", "",
              "main512 把短类计算贡献提高到约 19.68%，但计算时间也变长，更多读取可以隐藏，实际平均等待约 0.907 ms。main1024 的短类计算贡献接近 49%，但约 7.257 ms 的计算更能隐藏读取，实际平均只等约 0.368 ms。因此，提高短类计算贡献本身仍不能保证整机 U 降低。", "",
              "## b/B 具体对应什么", "", "```text", "D = 本层开始计算，到下一层开始计算",
              "同请求完整内部周期：平均 b = V/D；Bi = V/C",
              "所以：平均 b/Bi = C/D = C/(C+stall)", "```", "",
              "这三个实验的长类完整内部周期均没有暴露等待。短类是否低效，需要看平均 b/Bi；但整机 U 还取决于该类占了多少时间，以及跨请求和窗口边界。不能把不同层的 b/Bi 不加权平均后当作整机 U。", "",
              "## warm 混合要求需要单独检查", "", "| 候选 | warm U | warm 两类都计算的卡数 | 未满足卡号 | 长窗两类都计算卡数 |", "|---|---:|---:|---|---:|"]
    for c in cases:
        warm,long=c["windows"]
        missing=", ".join(map(str,warm["missing_mixed_compute_npus"])) or "无"
        lines.append(f'| {c["candidate"]} | {warm["actual_U_percent"]:.4f}% | {warm["mixed_compute_npus"]}/32 | {missing} | {long["mixed_compute_npus"]}/32 |')
    lines += ["", "三个实验全部 32 卡在窗口内都有已接纳请求。但 tight256、main1024 的 seed 7 不满足“每张卡在 warm 内两类都实际计算”的要求，不能把它们当作满足该条件的最终证据。保留这些结果用于诊断，不换一个更坏的 seed 来掩盖失败。", "",
              "## 为什么 SSU 忙时整机平均很难很低", "", "```text", "整个有限批次，从开始到全部完成，使用相同总时长：",
              "SSU平均忙率 = rho_ideal * NPU平均U", "```", "",
              "读取与计算总量固定，每盘以 40 GiB/s 服务。如果 rho≈1，而独立随机让 SSU 一直有 IO 可做，整机 U 就很难同时很低。只有很多读取等待却不影响 SSU 忙碌，并不足以证明总吞吐会明显下降。下面的整批守恒关系已从原始计算记录、全部读取量及原结果 SSU 忙率交叉核对。", "",
              "| 候选 | 整批 NPU U | 整批 SSU 忙率 | rho×U |", "|---|---:|---:|---:|"]
    for c in cases:
        f=c["full_finite"]
        lines.append(f'| {c["candidate"]} | {f["U_percent"]:.6f}% | {f["ssu_busy_percent"]:.6f}% | {f["rho_times_U_percent"]:.6f}% |')
    lines += ["", "整批结果包含启动及最后任务陆续结束的阶段，与中间窗口不同。不能直接将整批 rho 套进 warm/long：窗口内推进的画像比例不同，还有在途读取与边界计算。此次长窗实测 SSU 忙率与“整批 rho×窗口 U”仍相差约 2.59–3.79 个百分点，已经足以说明这不是窗口恒等式。", "",
              "扩展候选继续检查更高受损类计算贡献、极端尺度、计算相近但读取差大的画像。如果独立随机仍持续让 SSU 很忙，应接受“这些限制下整机 U 不容易明显下降”的结果。改变每卡配比或引入相关随机需要另列为新输入假设。", "",
              "[初始数学设计](candidate_math.md) · [扩展候选](adaptive_candidates.md) · [精确数值及来源哈希](math_to_measurements.json) · [复算脚本](build_math_to_measurements.py)", ""]
    (HERE/"math_to_measurements.md").write_text("\n".join(lines))
    print(json.dumps({"cases":len(cases),"outputs":["math_to_measurements.md","math_to_measurements.json"],"simulations_run":0}))


if __name__ == "__main__":
    main()
