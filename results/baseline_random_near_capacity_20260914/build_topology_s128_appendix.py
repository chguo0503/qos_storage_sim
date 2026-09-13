#!/usr/bin/env python3
"""Record two adaptive hypotheses; read only the pre-existing 3-SSU pilot.

Does not launch simulations or read the outcomes of the two new candidates.
"""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parent.parent
SOURCE = STUDY / "runs/adapt_f2_01_ssu3_h22000_seed7/baseline"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = {str(STUDY / f): sha(STUDY / f) for f in (
        "topology_candidates.py", "topology_candidates.json", "topology_candidates.md")}
    raw = ast.literal_eval((ROOT / "data").read_text())
    analysis = json.loads((SOURCE / "analysis.json").read_text())
    with gzip.open(SOURCE / "manifest.json.gz", "rt") as handle:
        manifest = json.load(handle)
    with gzip.open(SOURCE / "result.json.gz", "rt") as handle:
        result = json.load(handle)
    assert analysis["all_technical_checks_passed"]
    for path, digest in analysis["sources"].items():
        assert sha(Path(path)) == digest, path
    window = next(w for w in analysis["windows"] if
                  (w["start_ms"], w["end_ms"]) == (2000, 20000))
    warm = next(w for w in analysis["windows"] if
                (w["start_ms"], w["end_ms"]) == (2000, 4000))
    roles = {q["request_id"]: q["load"]["role"] for q in manifest["requests"]}
    stalls, computes, durations = [], [], []
    for batch in result["summary"]["microbatch_metrics"]:
        if roles[batch["member_request_ids"][0]] != "H":
            continue
        layers = batch["layer_metrics"]
        for cur, nxt in zip(layers, layers[1:]):
            if cur["compute_start_ms"] >= 2000 and nxt["compute_start_ms"] <= 20000:
                stalls.append(max(0., nxt["compute_start_ms"] - cur["compute_end_ms"]))
                computes.append(cur["compute_end_ms"] - cur["compute_start_ms"])
                durations.append(nxt["compute_start_ms"] - cur["compute_start_ms"])
    observed_w = math.fsum(stalls) / len(stalls)
    published = window["complete_internal_cycles_inside_window"]["per_role"]["H"]
    assert len(stalls) == published["count"] == 21530
    assert abs(observed_w - published["stall_ms_including_zeros"]["mean"]) < 1e-12

    def profile(key):
        b, c_us, source_ttft, v = raw[key]
        return dict(data_key=list(key), C_ms=c_us / 1000, V_GiB=v,
                    B_GiB_s=v / (c_us / 1e6), source_B_GiB_s=b,
                    source_ttft_ms=source_ttft,
                    blocks_per_layer=(key[0] * 1024 - key[1]) // 128,
                    original_data_row=list(raw[key]))

    long, short, short256 = profile((200, 4096)), profile((32, 128)), profile((32, 256))
    vl, cl, vs, cs = long["V_GiB"], long["C_ms"], short["V_GiB"], short["C_ms"]
    source_scale = vl / 120 * 1000
    kappa = observed_w / source_scale
    rho3 = 32 * (vl + 7 * vs) / (120 * (cl + 7 * cs) / 1000)
    f3 = 7 * cs / (cl + 7 * cs)
    source_proxy = 100 / (1 + f3 * observed_w / cs)
    full_u = result["summary"]["fleet_npu_compute_utilization"]
    full_ssd = result["summary"]["ssd_mean_utilization"]
    assert abs(full_ssd - rho3 * full_u) < 1e-10

    rows = []
    for nssu, nl, ns in ((6, 2, 47), (8, 1, 37)):
        identity = f"topo{nssu}_m4096_s32m128"
        cycle_compute_ms = nl * cl + ns * cs
        ratio = 32 * (nl * vl + ns * vs) / (nssu * 40 * cycle_compute_ms / 1000)
        f = ns * cs / cycle_compute_ms
        scale = vl / (nssu * 40) * 1000
        predicted_w = kappa * scale
        required_w = 0.25 * cs / f
        repeats = math.ceil(22000 / (8 * cycle_compute_ms))
        rows.append(dict(
            candidate_id=identity, num_npu=32, num_ssu=nssu,
            per_ssu_GiB_s=40, npu_link_GiB_s=50, n_layers=8,
            profiles=[long, short], roles=["L", "S"], counts=[nl, ns],
            ratio_L_S=[nl, ns], rho_ideal=ratio, f_short=f,
            ideal_compute_only_mean_short_cards=32*f,
            L_balanced_storage_layer_service_ms=scale,
            S_minimum_NPU_link_service_ms=vs / 50 * 1000,
            S_first_link_start_delay_threshold_ms=cs-vs/50*1000,
            conditional_prediction=dict(
                assumption="All-S mean internal stall scales with VL/(num_ssu*40), using the measured 3-SSU coefficient. This is a hypothesis, not a guarantee.",
                kappa_from_3ssu=kappa, mean_S_stall_ms=predicted_w,
                approximate_U_percent=100 / (1 + f * predicted_w / cs)),
            U80_required_mean_S_stall_ms=required_w,
            U80_required_kappa=required_w/scale,
            minimum_22s_population=dict(repeats=repeats,
                L=nl*repeats, S=ns*repeats,
                pure_compute_ms_per_npu=8*cycle_compute_ms*repeats),
            L_request_pure_compute_ms=8*cl,
            results_read_for_this_appendix=False,
            cli=f"python results/{STUDY.name}/experiment.py --name {identity} --profiles 200:4096,32:128 --counts {nl},{ns} --roles L,S --num-ssu {nssu} --horizon-ms 22000 --seed 7 --strategy baseline"))

    record = dict(
        schema_version="topology-s128-adaptive-hypothesis-v1",
        status="Adaptive extension after seeing adapt_f2_01; candidate outcomes are not read by this builder.",
        no_simulations_launched_by_builder=True,
        new_candidates_outcomes_used=False,
        primary_study_num_ssu=3,
        secondary_topology_comparison=[6, 8],
        seed_plan=dict(pilot=7, confirmation=[19,43,67,101]),
        random_definition="Same fixed counts on every card; shuffle each complete queue independently with seed+100003*npu; all requests arrive at time 0; no fixed-card roles or aligned decks.",
        active_and_mixed_rule="Report actual compute overlap separately for warm[2,4) and long[2,20). Keep failed warm-mix seeds; do not replace them.",
        raw_profiles=dict(L=long, S128=short, S256_comparison=short256),
        source_calibration=dict(
            directory=str(SOURCE.relative_to(ROOT)), window_ms=[2000,20000],
            source_role_mapping="H is S=(32,128), L is (200,4096)",
            ratio_L_S=[1,7], rho_ideal=rho3, f_short=f3,
            complete_S_internal_cycles=len(stalls),
            mean_S_internal_stall_ms_including_zeros=observed_w,
            S_complete_internal_weighted_U_percent=100*math.fsum(computes)/math.fsum(durations),
            L_complete_internal_stall_ms=window["complete_internal_cycles_inside_window"]["per_role"]["L"]["total_stall_ms"],
            L_balanced_storage_layer_service_ms=source_scale, kappa=kappa,
            approximate_U_percent_from_wait_model=source_proxy,
            actual_long_U_percent=window["U_percent"],
            actual_long_SSD_busy_percent=math.fsum(window["physical"]["per_ssu_utilization_percent"])/3,
            long_mixed_cards=window["long_short_mixed_card_count"],
            warm_U_percent=warm["U_percent"], warm_mixed_cards=warm["long_short_mixed_card_count"],
            full_finite_U_percent=100*full_u,
            full_finite_SSD_busy_percent=100*full_ssd,
            full_finite_SSD_minus_rho_U_pp=100*(full_ssd-rho3*full_u)),
        formulas=dict(
            rho_ideal="32*(nL*VL+nS*VS)/(num_ssu*40*(nL*CL+nS*CS)); C in seconds",
            f_short="nS*CS/(nL*CL+nS*CS)",
            internal_cycle="D=current.compute_start to next.compute_start inside the same request; w=D-C; include zero-stall cycles",
            wait_approximation="U ~= 1/(1+f_short*mean_wS/CS); long layers unstalled; stable completion mix; request boundaries/startup ignored",
            required_wait_U80="mean_wS_required=0.25*CS/f_short; a required amount under the approximation, not a prediction",
            full_finite_identity="SSD_busy_fraction = rho_ideal * U_fraction over exactly the same complete finite population and [0,makespan], fixed service rate40 per SSD, every original byte read once and every compute stage included",
            window_caution="For a fixed middle T window, whole-input rho is generally not the byte/compute ratio inside T; request mix and prefetch carry-in/out differ. Do not apply the full-finite identity directly.",
            link_delay_bound="If next-S first link-service start is delayed by tau from current compute_start, readiness >= tau+VS/50. Therefore tau>CS-VS/50 is sufficient for a stall.",
            streaming_caution="SSD and NPU-link service overlap. Do not add full SSD service time and full link service time. VL/(num_ssu*40) is a balanced storage-work scale, not measured FIFO delay."),
        limitations=[
            "Changing SSD count and L:S ratio changes queue depth, effective residual-prefix distribution, link overlap, per-disk striping, and temporal role mix. kappa need not transfer.",
            "L requests compute for 1.130686 seconds; consecutive L requests can occupy most/all of a 2-second window. Warm all-card mixed status is not guaranteed by independent random input.",
            "rho near 1 is only an ideal whole-input average. It does not guarantee per-disk per-instant underload.",
            "The predicted 88.30%/87.49% do not imply 80%, and cannot establish a >=10 percentage-point improvement for Once; Once requires paired measurement.",
            "The two additional candidates were motivated by already observed 3-SSU results. Report them as adaptive hypotheses, not as part of the frozen original eight."],
        selected_candidates=rows,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/"data", SOURCE/"analysis.json", SOURCE/"manifest.json.gz", SOURCE/"result.json.gz", STUDY/"study_plan.json", STUDY/"experiment.py", Path(__file__).resolve())},
        preserved_frozen_topology_files_sha256=frozen)

    lines = [
        "# S128 两组拓扑追加：先写预测，再看结果", "",
        "这两组是看到 3 盘 `adapt_f2_01` 后提出的追加假设。原先冻结的八组拓扑候选保持不变。本文件只读取旧的 3 盘结果，不读取追加两组结果，也不启动仿真。", "",
        "保持 32 NPU、每盘 40 GiB/s、每卡链路 50 GiB/s、8 层、每卡相同配比但整队列独立随机；只将盘数改成 6 / 8，并重新选比例使理想平均需求接近容量。所有请求直接取 `data`。这是拓扑和输入比例的追加实验。", "",
        "| 画像 | data 总长度 / miss | 单层读取 GiB | 单层计算 ms | B=V/C GiB/s |", "|---|---|---:|---:|---:|",
        f"| L | 200K / 4096 | {vl:.12f} | {cl:.9f} | {long['B_GiB_s']:.9f} |",
        f"| S | 32K / 128 | {vs:.12f} | {cs:.9f} | {short['B_GiB_s']:.9f} |", "",
        "每卡 S 的带宽需求低于 50 GiB/s 链路上限。S128 的一层计算只有 1.178026 ms，自己接收一层数据至少需要 0.856018 ms，因此留给首次接收启动延迟的余量只有 **0.322008 ms**。S256 对应余量为 **1.144819 ms**。这就是补 S128 的理由：同一段排队等待，S256 可能藏进计算里，S128 更容易露出等待。", "",
        "这里不能把一整层的盘读取时间再加一整层的链路接收时间，因为两者会流水重叠。准确的下界是：若下一 S 层首次链路服务的启动已推迟 tau，则完整收到不早于 tau+V_S/50。只有在残余 FIFO 前缀确实推迟首块时，才能用盘队列解释这个 tau。当前有一张 L 卡，不能直接当作前面排着完整一层 L。", "",
        "旧的 3 盘数据给出的实测依据：", "",
        f"- 长窗 [2,20) 实际 U={window['U_percent']:.9f}%，SSD 平均忙率={record['source_calibration']['actual_long_SSD_busy_percent']:.9f}%。32 张卡均活跃且均计算过两类请求。",
        f"- 窗内完整 S 内部周期 {len(stalls)} 个；含零等待层的平均 IO stall w={observed_w:.12f} ms。独立从 raw result 重算与 analysis.json 一致。L 的完整内部周期等待合计为 0。",
        f"- 单个 L 层的均衡盘服务尺度 V_L/120={source_scale:.12f} ms，实测系数 kappa=w/(V_L/120)={kappa:.12f}。",
        f"- 原来 S 的纯计算权重仅 {100*f3:.6f}%；即使 S 完整内部周期的加权利用率只有 {record['source_calibration']['S_complete_internal_weighted_U_percent']:.6f}%，整机仍可达到约 91.56%。",
        f"- 该次 warm [2,4) U={warm['U_percent']:.6f}%，但只有 {warm['long_short_mixed_card_count']}/32 卡两类均出现，因此此次校准来自长窗，不将 warm 结果称作满足全卡混合约束。", "",
        "使用教程中的 C/D 记账关系，作一个需要实测检验的简化：", "",
        "```text", "f_S = nS*CS / (nL*CL+nS*CS)", "U_approx = 1 / (1 + f_S*mean_wS/CS)", "mean_wS_for_U80 = 0.25*CS/f_S", "", "假设：mean_wS_new = kappa_from_3SSU * VL/(SSU*40)", "```", "",
        "其中 mean_wS 必须包含没有等待的 S 层。这个 U 近似要求 L 不等待、完成的请求比例稳定，并忽略首层、跨请求和统计窗裁剪。它不是 warm 窗口恒等式。把旧的 3 盘数代入，近似 U 为 {:.6f}%，实际长窗为 {:.6f}%；两者接近不等于新拓扑一定遵循同一系数。".format(source_proxy, window["U_percent"]), "",
        "| ID | L:S 请求数 | 理想 rho | S 纯计算权重 | L 层盘服务尺度 ms | 假设均等待 ms | 条件预测 U | U=80% 所需均等待 ms |", "|---|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in rows:
        lines.append(f"| `{row['candidate_id']}` | {row['counts'][0]}:{row['counts'][1]} | {row['rho_ideal']:.9f} | {100*row['f_short']:.6f}% | {row['L_balanced_storage_layer_service_ms']:.9f} | {row['conditional_prediction']['mean_S_stall_ms']:.9f} | {row['conditional_prediction']['approximate_U_percent']:.6f}% | {row['U80_required_mean_S_stall_ms']:.9f} |")
    lines += ["", "这两组的条件预测是 **88.30% / 87.49%**，并不是 80%。若要降到 80%，kappa 分别需要约 **1.640 / 1.519**，高于旧值 0.869。增加盘数会提高为保持 rho 约 1 所需的短请求数量和短计算权重，却也缩短每次长读取带来的等待尺度；这两个作用相反。实际队列规模、条带分配、链路重叠和角色相位都会变化，等待系数可能增大，也可能减小。", "",
        "区分整批全程恒等式与中间 T 窗：", "",
        "```text", "rho_ideal = 32*(nL*VL+nS*VS) / (SSU*40*(nL*CL+nS*CS))", "", "全批完成、同一个 [0,makespan] 区间：", "SSD_busy_fraction = rho_ideal * U_fraction", "```", "",
        "全程式子来自同一批工作总量守恒：读取总量决定盘忙时，纯计算总量决定卡忙时，再除以同一个 makespan。它要求原始工作全部算入、每字节恰读一次、每盘服务速率固定。旧 3 盘全程实测 U={:.9f}%、SSD 忙率={:.9f}%，该恒等式误差 {:.3g} 个百分点。".format(100*full_u,100*full_ssd,100*(full_ssd-rho3*full_u)), "",
        "中间 [2,4) 或 [2,20) 的读取总量与计算类别比例可偏离全批比例，还存在预取跨窗。因此不能把全批 rho 直接乘中间窗 U 预测盘忙率；中间窗必须独立积分实际计算时间和盘服务时间。上述条件 U 预测也只能作为稳定混合情况下的参考，需要用真实窗口统计核对。", "",
        "rho 约 1 仅代表理想平均负载接近总盘容量，不保证逐盘逐时刻欠载。不能据此把所有等待归为 FIFO，也不能先认定 Once 会接近 100% 或比 Baseline 高 10 个百分点。", "",
        "实际运行参数：", "",
        "| ID | 每卡 L / S 数量 | 每卡最低纯计算 ms |", "|---|---:|---:|"
    ]
    for row in rows:
        pop = row["minimum_22s_population"]
        lines.append(f"| `{row['candidate_id']}` | {pop['L']} / {pop['S']} | {pop['pure_compute_ms_per_npu']:.6f} |")
    lines += ["", "一条 L 请求单纯计算约 1.130686 秒，随机连续 L 可能占满两秒窗。必须逐卡核验 warm 和长窗都实际计算过长、短请求；失败也保留，不换成恰好很差的种子。pilot 为 seed 7，确认种子固定为 19 / 43 / 67 / 101。", "", "```bash"]
    lines += [row["cli"] for row in rows]
    lines += ["```", "", "复算：`python results/baseline_random_near_capacity_20260914/build_topology_s128_appendix.py`。原始数字、原始 data 行、源文件 SHA256 和假设见同目录 `topology_s128_appendix.json`。该脚本只读取上述旧 3 盘结果；追加两组的测量结果应另行报告。", ""]

    (STUDY / "topology_s128_appendix.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    (STUDY / "topology_s128_appendix.md").write_text("\n".join(lines))
    assert frozen == {p:sha(Path(p)) for p in frozen}, "Frozen topology files changed"
    print(json.dumps({"outputs":["topology_s128_appendix.json","topology_s128_appendix.md"],
                      "raw_S_cycle_count":len(stalls), "raw_mean_w_ms":observed_w,
                      "frozen_topology_files_unchanged":True,
                      "candidate_ids":[r["candidate_id"] for r in rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
