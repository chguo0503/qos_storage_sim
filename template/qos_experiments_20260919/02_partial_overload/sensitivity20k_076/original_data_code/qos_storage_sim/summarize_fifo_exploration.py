#!/usr/bin/env python3
"""Read completed FIFO experiments, independently audit, and write a Chinese report.

This is a read-only consumer of experiment directories. It never starts a
simulation or changes a manifest/result. Derived summary files may be replaced.
Run when all 24 screening runs and the six paired formal runs have completed.
Completed mixed_formal runs are discovered automatically on every invocation.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/fifo_underload_exploration_20260914"
SCREEN_COUNTS = {"screen": 12, "variant_screen": 2, "mixed20_screen": 10}
PAIRS = [
    ("formal/s3_L19_S13_seed7_fifo", "formal/s3_L19_S13_seed7_short_first"),
    ("formal/s4_L20_S12_seed7_fifo", "formal/s4_L20_S12_seed7_short_first"),
    ("tiny_formal/tiny_jitter_s3_L19_S13_seed7_baseline_fifo",
     "tiny_formal/tiny_jitter_s3_L19_S13_seed7_baseline_short_first"),
]
CAPACITY_GIB_S = 40.0


def read_json(path):
    path = Path(path)
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as stream:
        return json.load(stream)


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(actual, expected, label, tolerance=1e-7):
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: recomputed {actual} != saved {expected}")


def overlap(a, b, left, right):
    return max(0.0, min(b, right) - max(a, left))


def independent_window(result, manifest, metric):
    """Recompute utilization from raw compute intervals and SLO from endpoints."""
    summary = result["summary"]
    left, right = metric["window_ms"]
    width = right - left
    num_npu = manifest["metadata"]["num_npu"]
    layers = manifest["metadata"]["n_layers"]
    inputs = {q["request_id"]: q for q in manifest["requests"]}
    completed = summary["request_metrics"]
    assert len(inputs) == len(completed)
    assert set(inputs) == {q["request_id"] for q in completed}
    assert all(summary["invariants"].values())
    active_pieces = defaultdict(list)
    compute_pieces = defaultdict(list)
    role_active = defaultdict(list)
    role_compute = defaultdict(list)
    npu_intervals = defaultdict(list)
    slo = dict(count=0, passed=0)
    slo_all = dict(count=0, passed=0)
    role_slo = defaultdict(lambda: dict(count=0, passed=0))
    for q in completed:
        source = inputs[q["request_id"]]
        assert q["npu_id"] == source["npu_id"]
        a, b = q["admission_time_ms"], q["completion_time_ms"]
        assert math.isfinite(a) and math.isfinite(b) and b >= a
        npu, role = q["npu_id"], source["load"]["role"]
        elapsed = overlap(a, b, left, right)
        active_pieces[npu].append(elapsed)
        role_active[role].append(elapsed)
        if elapsed:
            npu_intervals[npu].append((max(a, left), min(b, right)))
        ideal = layers * source["load"]["per_layer_us"] / 1000.0
        close(ideal, q["own_compute_ms"], "own_compute_ms")
        passed = int(b - a <= 1.5 * ideal + 1e-9)
        slo_all["count"] += 1
        slo_all["passed"] += passed
        if left <= a < right:
            slo["count"] += 1
            slo["passed"] += passed
            role_slo[role]["count"] += 1
            role_slo[role]["passed"] += passed
    for batch in summary["microbatch_metrics"]:
        assert batch["batch_size"] == 1 and len(batch["member_request_ids"]) == 1
        source = inputs[batch["member_request_ids"][0]]
        role, npu = source["load"]["role"], batch["npu_id"]
        assert npu == source["npu_id"]
        assert len(batch["layer_metrics"]) == layers
        for layer in batch["layer_metrics"]:
            a, b = layer["compute_start_ms"], layer["compute_end_ms"]
            elapsed = overlap(a, b, left, right)
            compute_pieces[npu].append(elapsed)
            role_compute[role].append(elapsed)
    active = [math.fsum(active_pieces[n]) for n in range(num_npu)]
    compute = [math.fsum(compute_pieces[n]) for n in range(num_npu)]
    for n in range(num_npu):
        intervals = sorted(npu_intervals[n])
        assert intervals, f"NPU {n} has no active interval"
        cursor = left
        for a, b in intervals:
            close(a, cursor, f"NPU {n} contiguous active coverage")
            cursor = b
        close(cursor, right, f"NPU {n} full active coverage")
        close(active[n], width, f"NPU {n} active duration")
    roles = {}
    for role in role_active:
        active_ms = math.fsum(role_active[role])
        compute_ms = math.fsum(role_compute[role])
        roles[role] = dict(active_ms=active_ms, compute_ms=compute_ms,
            U_percent=100.0 * compute_ms / active_ms,
            exposed_stall_card_ms=active_ms - compute_ms,
            slo_1p5=role_slo[role])
    for item in [slo, slo_all, *role_slo.values()]:
        item["rate"] = item["passed"] / item["count"] if item["count"] else None
    recalculated = dict(window_ms=[left, right],
        U_percent=100.0 * math.fsum(compute) / (num_npu * width),
        by_role=roles, slo_1p5=slo, all_requests_slo_1p5=slo_all,
        active_ms_by_npu=active, compute_ms_by_npu=compute,
        all_32_npus_active_whole_window=num_npu == 32,
        all_input_requests_completed=True, complete_request_count=len(completed),
        all_invariants_passed=True)
    close(recalculated["U_percent"], metric["U_percent"], "fleet U")
    close(recalculated["U_percent"], 100 * result["windows"][0]["mean_npu_utilization"], "saved window U")
    for role, field in [("L", "long_U_percent"), ("S", "short_U_percent")]:
        close(roles[role]["U_percent"], metric[field], f"{role} U")
        close(roles[role]["U_percent"], 100 * result["windows"][0]["by_role"][role]["active_compute_fraction"], f"{role} saved window U")
    close(roles["S"]["exposed_stall_card_ms"], metric["short_stall_card_ms"], "short stall")
    close(100 * slo["rate"], metric["slo_1p5_percent"], "SLO1.5")
    assert slo["count"] == metric["slo_count"] and slo["passed"] == metric["slo_passed"]
    for name, item in [("window_admissions", slo), ("all_requests", slo_all)]:
        saved = result["slo"][name]["admission"]
        assert item["count"] == saved["count"] and item["passed"] == saved["passed"]
        close(item["rate"], saved["rate"], f"{name} SLO")
    assert metric["all_active"] is True
    return recalculated


def whole_run_demand(result, manifest):
    """Sweep every admission/completion breakpoint; no time-bin sampling."""
    summary = result["summary"]
    num_ssu = manifest["metadata"]["num_ssu"]
    npu_count = manifest["metadata"]["num_npu"]
    vectors = {}
    maxima = [[0.0] * num_ssu for _ in range(npu_count)]
    for q in manifest["requests"]:
        placement = manifest["placements"][q["placement_index"]]
        assert len(placement) == 1 or all(p == placement[0] for p in placement)
        volume = [math.fsum(size for disk, size in placement[0] if disk == s)
                  for s in range(num_ssu)]
        close(math.fsum(volume), q["load"]["per_layer_kv_gb"], "placement bytes", 1e-12)
        compute_s = q["load"]["per_layer_us"] / 1e6
        vector = [v / compute_s for v in volume]
        vectors[q["request_id"]] = vector
        maxima[q["npu_id"]] = [max(a, b) for a, b in zip(maxima[q["npu_id"]], vector)]
    events = defaultdict(lambda: [[] for _ in range(num_ssu)])
    for q in summary["request_metrics"]:
        vector = vectors[q["request_id"]]
        for t, sign in [(q["admission_time_ms"], 1), (q["completion_time_ms"], -1)]:
            for s, value in enumerate(vector):
                events[t][s].append(sign * value)
    current = [0.0] * num_ssu
    peaks = [0.0] * num_ssu
    peak_intervals = [None] * num_ssu
    areas = [0.0] * num_ssu
    overload_ms = [0.0] * num_ssu
    any_overload = 0.0
    previous = 0.0
    end = float(summary["makespan_ms"])
    events[end]
    interval_count = 0
    for t, changes in sorted(events.items()):
        duration = t - previous
        assert duration >= 0
        if duration > 0:
            interval_count += 1
            overloaded = False
            for s in range(num_ssu):
                if current[s] > peaks[s]:
                    peaks[s], peak_intervals[s] = current[s], [previous, t]
                areas[s] += duration * current[s]
                if current[s] > CAPACITY_GIB_S + 1e-8:
                    overload_ms[s] += duration
                    overloaded = True
            if overloaded:
                any_overload += duration
        current = [math.fsum([current[s], *changes[s]]) for s in range(num_ssu)]
        previous = t
    assert all(abs(v) < 1e-7 for v in current), current
    static = [math.fsum(row[s] for row in maxima) for s in range(num_ssu)]
    return dict(start_ms=0.0, end_ms=end, admission_completion_intervals=interval_count,
        peak_nominal_gib_s_by_ssu=peaks, peak_intervals_ms_by_ssu=peak_intervals,
        mean_nominal_gib_s_by_ssu=[v / end for v in areas],
        overload_ms_by_ssu=overload_ms, any_ssu_overload_ms=any_overload,
        any_ssu_overload_fraction=any_overload / end,
        underloaded_every_admission_interval=any_overload == 0.0,
        per_ssu_static_upper_bound_gib_s=static,
        static_underload_all_request_combinations=max(static) <= CAPACITY_GIB_S,
        capacity_gib_s_per_ssu=CAPACITY_GIB_S,
        definition="sum admitted request per-layer bytes on SSU / own layer compute seconds",
        caveat="Layer-granularity nominal demand, not instantaneous command arrival bandwidth. "
               "Next-request L0 prefetch is not counted as a second admitted request. "
               "The observed sequence can be underloaded even when the sum of per-NPU maxima exceeds capacity.")


def paired_input_check(left, right, left_result, right_result):
    for key in ["input_fingerprint", "requests", "placements"]:
        assert left[key] == right[key], f"Paired manifest {key} differ"
    for result, manifest in [(left_result, left), (right_result, right)]:
        assert result["input_fingerprint"] == manifest["input_fingerprint"]
        assert result["input_placement_fingerprint"] == result["execution_placement_fingerprint"]
    for key in ["input_placement_fingerprint", "execution_placement_fingerprint", "static_path_cirs_gib_s"]:
        assert left_result[key] == right_result[key], key
    assert left_result["summary"]["routing_mode"] == right_result["summary"]["routing_mode"]
    assert left_result["strategy"] == right_result["strategy"] == "baseline"
    assert not left_result["assignment_log"] and not right_result["assignment_log"]
    for result in (left_result, right_result):
        assert not result["adapter_statistics"]["cir_write_events"]
        if "io_audit" in result:
            assert result["io_audit"]["nonzero_path_io"] == 0
    by_npu = defaultdict(list)
    for q in left["requests"]:
        by_npu[q["npu_id"]].append((q["load"]["seq_len_k"], q["load"]["nql"]))
    return dict(exact_request_loads_equal=True, exact_placements_equal=True,
        exact_input_fingerprint_equal=True, npu_assignment_equal=True,
        cir_configuration_equal=True, no_dynamic_cir_writes=True,
        input_fingerprint=left["input_fingerprint"],
        request_load_sha256=digest(left["requests"]),
        manifest_placements_sha256=digest(left["placements"]),
        execution_placement_fingerprint=left_result["execution_placement_fingerprint"],
        unique_length_nql_within_every_npu=all(len(v) == len(set(v)) for v in by_npu.values()),
        only_intentional_change="FIFO versus ascending request per-layer bytes within Path0; FIFO ties")


def normalize_metric(path, out):
    metric = read_json(path)
    metadata = read_json(path.with_name("metadata.json"))
    row = dict(metric)
    row["experiment_path"] = path.parent.relative_to(out).as_posix()
    row["stage"] = path.parent.parent.name
    row["mode"] = metadata["case"].get("mode", metric.get("mode", "unknown"))
    row["policy"] = metric.get("policy", metadata.get("probe_policy", "fifo"))
    row["strategy"] = metric.get("strategy", "baseline")
    row["case"] = metadata["case"]
    row["profiles"] = metadata["profiles"]
    row["scope"] = metadata.get("workload_scope", "")
    row["input_fingerprint"] = metadata["input_fingerprint"]
    row["input_average_gib_s"] = metadata["input_demand"]["per_ssu_gib_s"]
    row["input_total_average_gib_s"] = metadata["input_demand"]["total_gib_s"]
    row["window_peak_max_gib_s"] = max(metric["demand_audit"]["peak_nominal_gib_s_by_ssu"])
    row["window_any_ssu_overload_ms"] = metric["demand_audit"]["any_ssu_overload_ms"]
    row["window_underloaded"] = metric["demand_audit"]["underloaded_every_admission_interval"]
    row["length_nql_unique_per_npu"] = metadata.get("all_length_nql_unique_within_each_npu", False)
    return row


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(v) for v in row) + " |" for row in rows]])


def report_text(rows, formal, audits, whole, pairs):
    by_path = {r["experiment_path"]: r for r in rows}
    table = ast.literal_eval((ROOT / "data").read_text())
    a, b = table[128, 256], table[32, 4096]
    b_share = 2 * b[1] / (a[1] + 2 * b[1])
    old_bw = 32 * (a[3] + 2 * b[3]) / ((a[1] + 2 * b[1]) / 1e6)
    s3 = by_path[PAIRS[0][0]]
    s4 = by_path[PAIRS[1][0]]
    tiny = by_path[PAIRS[2][0]]
    tiny_priority = by_path[PAIRS[2][1]]
    p_rows = []
    for label, key, count in [("旧 A", (128, 256), "每卡40"), ("旧 B", (32, 4096), "每卡80"),
                             ("新 L", (200, 2048), "3盘19卡；4盘20卡"),
                             ("新 S（3盘）", (32, 2048), "13卡"),
                             ("新 S（4盘）", (32, 1024), "12卡")]:
        bw, us, _, size = table[key]
        p_rows.append([label, f"{key[0]}K", key[1], count, f"{size:.9f}", f"{us/1000:.6f}", f"{bw:.6f}"])
    for path in formal:
        if path.startswith("mixed_formal/"):
            r = by_path[path]
            p = r["profiles"]["S"]
            p_rows.append(["混排 S（外推）", f"{p['total_length_k']:g}K", p["nql"],
                f"每卡L:S=1:{r['case']['short_per_long']}", f"{p['read_gib']:.9f}",
                f"{p['compute_us']/1000:.6f}", f"{p['B_gib_s']:.6f}"])
    formal_rows = []
    for path in formal:
        r = by_path[path]
        label = ("小扰动、不重复" if path.startswith("tiny") else
                 "全卡混排、含外推" if path.startswith("mixed") else "原始data、固定角色")
        formal_rows.append([r["num_ssu"], label,
            "FIFO" if r["policy"] == "fifo" else "短读取优先诊断",
            f"{r['U_percent']:.2f}%", f"{r['long_U_percent']:.2f}%", f"{r['short_U_percent']:.2f}%",
            f"{r['slo_1p5_percent']:.2f}% ({r['slo_passed']}/{r['slo_count']})"])
    demand_rows = []
    for path in formal:
        r, w = by_path[path], whole[path]
        demand_rows.append([r["name"], r["policy"],
            "/".join(f"{v:.3f}" for v in w["peak_nominal_gib_s_by_ssu"]),
            f"{w['any_ssu_overload_ms']:.6f}",
            f"{r['window_peak_max_gib_s']:.6f}", f"{r['window_any_ssu_overload_ms']:.6f}"])
    screening = [r for r in rows if r["stage"] in SCREEN_COUNTS]
    screened_over = [r for r in screening if not r["window_underloaded"]]
    mixed = [r for r in screening if r["mode"] == "mixed"]
    mixed_under = [r for r in mixed if r["window_underloaded"]]
    variant = [r for r in screening if r["stage"] == "variant_screen"]
    variant_ranges = sorted({(tuple(r["profiles"]["L"]["total_length_k_range"]),
                              tuple(r["profiles"]["S"]["total_length_k_range"])) for r in variant})
    variant_range_text = "；".join(f"L{long[0]:g}–{long[1]:g}K、S{short[0]:g}–{short[1]:g}K"
                                    for long, short in variant_ranges)
    mixed_formal = [by_path[path] for path in formal if path.startswith("mixed_formal/")]
    mixed_followup_lines = []
    for r in mixed_formal:
        screened = next((s for s in screening if s["name"] == r["name"]), None)
        if screened is not None:
            state = (f"出现{r['window_any_ssu_overload_ms']:.6f}ms名义超载，不能作为该窗口的欠载证据"
                     if not r["window_underloaded"] else "逐事件名义需求未超40")
            mixed_followup_lines.append(f"`{r['name']}`筛选窗口U={screened['U_percent']:.2f}%，"
                f"正式窗口U={r['U_percent']:.2f}%，原短窗口的下降幅度未复现；正式窗最高盘需求"
                f"{r['window_peak_max_gib_s']:.6f}GiB/s，{state}。")
    all_whole_underload = all(w["underloaded_every_admission_interval"] for w in whole.values())
    screen_rows = []
    for r in screening:
        case = r["case"]
        ratio = f"{case['nlong']}:{32-case['nlong']}卡" if "nlong" in case else f"1:{case['short_per_long']}请求"
        screen_rows.append([r["stage"], r["name"], ratio,
            f"{r['U_percent']:.2f}%", f"{r['short_U_percent']:.2f}%", f"{r['slo_1p5_percent']:.2f}%",
            f"{r['window_peak_max_gib_s']:.3f}",
            "欠载" if r["window_underloaded"] else f"排除：超载{r['window_any_ssu_overload_ms']:.1f}ms"])
    slo_matched = []
    for path in formal:
        r, item = by_path[path], audits[path]["all_requests_slo_1p5"]
        slo_matched.append([r["name"], r["policy"], f"{100*item['rate']:.2f}%", f"{item['passed']}/{item['count']}"])
    texts = [
        "# Ring hash 下 FIFO 欠载探索\n",
        f"找到了固定长短角色下的 FIFO 代价：3盘时整机利用率 **{s3['U_percent']:.2f}%**，"
        f"4盘时 **{s4['U_percent']:.2f}%**；短卡分别只有 **{s3['short_U_percent']:.2f}% / {s4['short_U_percent']:.2f}%**。"
        "相同输入仅改变 Path0 已入队 I/O 的顺序，两个原始data实例的利用率均达到100%。"
        "这是指定输入下的因果对照，不是 once 的实现或最优性证明。固定角色与全卡长短混排分别报告。\n",
        "## 为什么旧 Baseline Random 很高\n",
        f"旧输入 A=128K/256，B=32K/4096，每卡A:B=1:2。A是长读取、短计算；B虽总长度短，却是长计算。"
        f"B占理想计算时间的 **{100*b_share:.2f}%**：`2×C_B/(C_A+2×C_B)`。"
        "因此大部分计算时间来自能用较长计算窗口掩盖读取的B；少量A被阻塞，不必使整机时间平均利用率明显下降。"
        "随机顺序还会改变各卡的层相位，不能仅看长短请求数量判断损失。\n",
        f"旧输入平均名义需求为 **{old_bw:.2f} GiB/s**。3盘只有120 GiB/s，原3盘结果并非欠载；4盘160 GiB/s仅说明平均欠载，"
        "不自动证明每盘、每个运行区间都欠载。以下单位统一为GiB/s，沿用模拟器的40 GiB/s/盘、50 GiB/s/NPU。\n",
        "## 输入与统计口径\n",
        md_table(["类别", "总长度", "NQL", "配比", "每层读取GiB", "每层计算ms", "D/C GiB/s"], p_rows) + "\n",
        "总长度包含新增NQL，K=1024 tokens。新L有较多block，但计算窗口也长；新S的读取本身可供给，却更容易因前方已排入的大批IO超过其计算窗口而等待。"
        "32 NPU、8层、seed=7、每盘256虚拟节点ring hash；key=(request_id,block_index)，同一block全部层同盘。"
        "所有请求t=0进入各卡输入队列，batch=1，逐卡连续执行，并保留跨请求L0预取。\n",
        "正式窗口为 **[2,4)秒**，各卡均有请求在运行。整机U=窗口内计算时间之和/(32×2000ms)；"
        "类别U=该类计算时间/该类active时间。TTFT=完成−接纳，不含接纳前排队；SLO×1.5阈值=1.5×8层纯计算时间，"
        "对窗口内接纳的请求跟踪到全部完成。\n",
        md_table(["SSU", "输入", "队列顺序", "整机U", "长类U", "短类U", "TTFT SLO×1.5"], formal_rows) + "\n",
        f"3盘固定分离可手算：`U=(19×1+13×{s3['short_U_percent']/100:.8f})/32={s3['U_percent']:.4f}%`。"
        "这说明长卡保持计算时，13张短卡的IO stall已足以压低整机利用率。短读取优先按请求每层总读取量选取已入队IO，层总读取量相同时FIFO，每次IO完成重新选择，不抢占正在服务的IO；"
        "硬件服务粒度、Path0、CIR、盘与NPU带宽、输入及落盘保持一致。\n",
        "## 欠载与局部阻塞证据\n",
        "欠载量定义为 `R_s(t)=Σ 当前接纳请求 D(i,s)/C_i`。以下重新从manifest和完整完成记录扫描每个接纳/完成事件之间的区间，"
        "没有用粗采样隐藏峰值；表中是整个运行过程各盘的最大名义需求。\n",
        md_table(["实例", "队列", "逐盘全程峰值GiB/s", "全程超40时长ms", "正式窗最高盘需求", "正式窗超40时长ms"], demand_rows) + "\n",
        ("上述正式实际运行序列逐事件未超过40。" if all_whole_underload else
         "各组是否全程欠载以表中超载时长为准；出现超载的正式组不作为全程欠载证据。") +
        "运行序列欠载不代表任意排列都满足：逐卡取其所有请求最大需求再相加的静态上界有盘超过40。"
        "更不能把D/C欠载解释为没有突发IO排队：整层block集中到达，短层有截止时间，L0还可提前读取。"
        "这些定义与静态上界都保留在 `whole_run_underload.json`。\n",
        "局部证据见 [时间账与解释](formal/s3_L19_S13_seed7_fifo/diagnostics/fifo_short_victim_explained.md)。"
        "NPU21请求21000011的L2层暴露等待 **27.822057ms**，其中 **23.102209ms** 是前方 **17个长请求层** 的盘服务。"
        "不是单个长请求独占27.8ms。该例从1800–2400ms完整覆盖的短层中选最大stall，不能代表平均；"
        "关键路径时间账按不重叠区间拆分，没有把大量IO的排队时间重复相加。\n",
        "![FIFO局部时间线](formal/s3_L19_S13_seed7_fifo/diagnostics/fifo_short_victim_timeline.png)\n",
        "## 随机变化与适用边界\n",
        f"小扰动组保持19长卡+13短卡，每卡长度/NQL组合不重复且随机排序。FIFO利用率 **{tiny['U_percent']:.2f}%**，"
        f"短卡 **{tiny['short_U_percent']:.2f}%**，优先对照整机 **{tiny_priority['U_percent']:.2f}%**；"
        f"FIFO的SLO×1.5已有 **{tiny['slo_1p5_percent']:.2f}%**。因此利用率仍有损失，不能宣称TTFT达标率也同等恶化。"
        "长类总长199.984375–200K/NQL2032–2048，短类32–32.061523K/NQL2048–2111；固定命中前缀，"
        "计算时间在原data网格内双线性插值，没有外推或倍率缩放，插值点不是新增实测数据。\n",
        f"较宽的随机长度组（{variant_range_text}，固定角色且每卡不重复）整机U为 **{min(r['U_percent'] for r in variant):.2f}%–{max(r['U_percent'] for r in variant):.2f}%**。"
        f"全卡随机长短混排共筛了{len(mixed)}组，其中窗口逐盘欠载的{len(mixed_under)}组利用率范围为"
        f" **{min(r['U_percent'] for r in mixed_under):.2f}%–{max(r['U_percent'] for r in mixed_under):.2f}%**。"
        "这些结果一并保留：层相位分散及长计算时间占比会削弱短流stall对整机U的影响。不能将固定分离结果外推成所有随机输入结论。\n",
        "全卡混排追加正式复测（相同[2,4)秒口径）：" + "；".join(
            f"{r['num_ssu']}盘、L{r['case']['long'][0]}K/{r['case']['long'][1]}:S{r['case']['short'][0]}K/{r['case']['short'][1]}="
            f"1:{r['case']['short_per_long']}，整机U **{r['U_percent']:.2f}%**、短类U **{r['short_U_percent']:.2f}%**、"
            f"SLO×1.5 **{r['slo_1p5_percent']:.2f}%**"
            for r in mixed_formal) +
        "。这些输入把长短请求随机混入每张卡，仍重复有限profile；短类计算含20–32K范围外推。"
        "它们没有短读取优先配对实验，不能借用固定分离的100%对照宣称这类混排也可达到100%。\n",
        "\n".join(mixed_followup_lines) + "\n",
        "原始筛选重复data中的长度/NQL，只作机制定位；20K/24K短流计算由32K/48K行线性外推，NQL1536另做NQL插值，均非原始实测。"
        "只有宽范围随机组及小扰动组满足每卡长度/NQL组合不重复。筛选是有限候选、单seed探索，不是整个20–200K域的穷举或随机负载保证。\n",
        f"## 附录：全部{len(screening)}组筛选\n",
        f"窗口均为[0.2,0.8)秒，仅用于筛选；其中{len(screened_over)}组出现逐盘名义超载，已标记，不能作为欠载证据。\n",
        md_table(["阶段", "实例", "L:S", "整机U", "短类U", "SLO×1.5", "最高盘需求", "窗口判定"], screen_rows) + "\n",
        "正式窗口的接纳请求集合会随调度变化。为提供完全相同输入总体的参照，以下统计每组全部请求，包含启动与排空阶段：\n",
        md_table(["实例", "顺序", "全部请求SLO×1.5", "通过/总数"], slo_matched) + "\n",
        f"{len(formal)}组正式结果已独立复算U、类别U、SLO；3对输入的load、placement、指纹逐项相同，全部请求完成、32卡覆盖完整窗口。"
        "核验明细见 `formal_recalculation.json`，输入匹配及核心源码SHA256见 `verification.json`；"
        "每个实验目录保留manifest、完整result、metadata和metrics。复现命令见 `README.md`。\n",
    ]
    return "\n".join(texts)


def readme_text():
    return """# FIFO 欠载实验复现

请在本项目的 `source` 目录执行。Python依赖沿用原仿真项目（含numpy）。
读取已完成结果并重建汇总，不重新仿真：

```bash
python summarize_fifo_exploration.py
```

代码先要求24组screen与6组配对formal结果齐全，另自动纳入mixed_formal下已完成的组。
再独立核对时间窗口、完整请求总体、配对输入与核心源码。
只覆盖派生的CSV/JSON/Markdown汇总；不会改动任何实验manifest、metadata、metrics或result。

重新仿真请选新的 `--stage`，例如 `reproduce_20260914`，保护已保存结果。已有result目录会主动报错。

```bash
python run_fifo_proof.py --case 0 --policy fifo --horizon-ms 4500 --window 2000 4000 --stage reproduce_20260914
python run_fifo_proof.py --case 0 --policy short_first --horizon-ms 4500 --window 2000 4000 --stage reproduce_20260914
python run_fifo_proof.py --case 2 --policy fifo --horizon-ms 4500 --window 2000 4000 --stage reproduce_20260914
python run_fifo_proof.py --case 2 --policy short_first --horizon-ms 4500 --window 2000 4000 --stage reproduce_20260914
python explore_fifo_tiny_jitter.py --horizon-ms 4500 --window 2000 4000 --policy fifo --stage reproduce_tiny_20260914
python explore_fifo_tiny_jitter.py --horizon-ms 4500 --window 2000 4000 --policy short_first --stage reproduce_tiny_20260914
```

探索脚本：`explore_fifo_underload.py --case N`（N=0…11）、
`explore_fifo_variants.py --case N`（本轮N=1、3）、
`explore_fifo_mixed20.py --case N`（N=0…9）。筛选默认horizon=1000ms、window=[200,800)ms。
每个脚本显式支持 `--seed` 与 `--stage`；本轮seed=7。
本轮宽随机组另指定 `--long-range 176 200 --short-range 20 35`，实际范围以各组metadata为准。

主要文件：

- `fifo_underload_report.md`：中文结论、口径、阴性结果及完整筛选表。
- `all_experiments.csv/json`：全部已完成组的统计及输入来源。
- `formal_recalculation.json`：正式结果独立复算，包括逐卡计算时间与类别SLO。
- `whole_run_underload.json`：从manifest重建D/C并逐事件扫描全程逐盘峰值/超载时长。
- `verification.json`：配对输入相等、全部请求完成、核心源码哈希等核验。
- `formal/s3_L19_S13_seed7_fifo/diagnostics/`：局部时间线、关键路径时间账与同请求对照。

短读取优先只是保留全部硬件/输入条件后改变Path0顺序的诊断；不是once，也没有最优上界证明。
参考报告严格区分raw重复profile、外推候选和每卡不重复的插值输入。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    metric_paths = []
    for stage, count in SCREEN_COUNTS.items():
        paths = sorted((out / stage).glob("*/metrics.json"))
        if len(paths) != count:
            raise RuntimeError(f"Need {count} completed {stage} runs; found {len(paths)}")
        metric_paths.extend(paths)
    formal_paths = [p for pair in PAIRS for p in pair]
    formal_paths += [p.parent.relative_to(out).as_posix()
                     for p in sorted((out / "mixed_formal").glob("*/metrics.json"))]
    for path in formal_paths:
        for filename in ["metrics.json", "metadata.json", "manifest.json.gz", "result.json.gz"]:
            if not (out / path / filename).is_file():
                raise FileNotFoundError(f"Formal run incomplete: {path}/{filename}")
        metric_paths.append(out / path / "metrics.json")
    rows = [normalize_metric(path, out) for path in metric_paths]
    data = {}
    audits, whole, pair_checks = {}, {}, {}
    source_hashes = None
    for path in formal_paths:
        destination = out / path
        result, manifest, metric = [read_json(destination / name) for name in
            ["result.json.gz", "manifest.json.gz", "metrics.json"]]
        audits[path] = independent_window(result, manifest, metric)
        whole[path] = whole_run_demand(result, manifest)
        hashes = result["core_and_policy_sha256"]
        if source_hashes is None:
            source_hashes = hashes
        assert hashes == source_hashes, f"Core source hashes differ for {path}"
        data[path] = (result, manifest)
    for left, right in PAIRS:
        lr, lm = data[left]
        rr, rm = data[right]
        pair_checks[left.rsplit("_", 1)[0]] = paired_input_check(lm, rm, lr, rr)
    local_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in source_hashes}
    assert local_hashes == source_hashes, "Current core files differ from simulation snapshots"
    verification = dict(schema_version=1, all_checks_passed=True,
        formal_run_count=len(formal_paths),
        all_formal_window_metrics_independently_recomputed=True,
        all_formal_requests_completed=True, all_formal_npus_active_whole_window=True,
        all_formal_core_hash_sets_equal=True, core_hashes_match_current_source=True,
        core_and_policy_sha256=source_hashes, input_pairs=pair_checks,
        whole_run_underload_definition=next(iter(whole.values()))["definition"],
        whole_run_all_formal_underloaded=all(w["underloaded_every_admission_interval"] for w in whole.values()),
        screening_counts=SCREEN_COUNTS, total_experiments=len(rows),
        summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # No output is changed until every required source and audit is available.
    report = report_text(rows, formal_paths, audits, whole, pair_checks)
    flat_columns = ["stage", "name", "case_index", "policy", "mode", "num_ssu", "seed", "window_ms",
        "U_percent", "long_U_percent", "short_U_percent", "short_stall_card_ms",
        "slo_1p5_percent", "slo_passed", "slo_count", "all_active",
        "input_total_average_gib_s", "input_average_gib_s", "window_peak_max_gib_s",
        "window_any_ssu_overload_ms", "window_underloaded", "length_nql_unique_per_npu",
        "experiment_path", "input_fingerprint", "scope"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=flat_columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (list, dict))
                         else row.get(k, "") for k in flat_columns})
    write_json(out / "all_experiments.json", rows)
    atomic_text(out / "all_experiments.csv", stream.getvalue())
    write_json(out / "formal_recalculation.json", audits)
    write_json(out / "whole_run_underload.json", whole)
    write_json(out / "verification.json", verification)
    atomic_text(out / "fifo_underload_report.md", report)
    atomic_text(out / "README.md", readme_text())
    print(json.dumps(dict(output=str(out), experiments=len(rows), all_checks_passed=True,
        whole_run_all_formal_underloaded=verification["whole_run_all_formal_underloaded"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
