#!/usr/bin/env python3
"""Explain one fully traced short-request layer's FIFO stall without summing IO waits.

Run after both matched-input simulations finish.  Default paths target the
s3_L19_S13 formal seed-7 FIFO / short-first probe.  This script never simulates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / "results/fifo_underload_exploration_20260914/formal"
EPS = 1e-7
COLORS = {"ahead_L": "#2364a5", "other_S": "#a8cee8", "own_S": "#9273ba",
          "idle": "#eeeeee", "critical_ssd": "#b04b98", "link_wait": "#f7c75d",
          "link_service": "#65a7a1", "compute": "#48a577", "stall": "#f3c34f"}
LABELS = {"ahead_L": "前方长类 L 的盘服务", "other_S": "其他短类 S 的盘服务",
          "own_S": "本请求其他 block 的盘服务", "idle": "盘空闲",
          "critical_ssd": "关键 block 自身盘服务", "link_wait": "关键 block 链路排队",
          "link_service": "关键 block 链路传输"}


def read_json(path):
    path = Path(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def overlap(a, b, left, right):
    return max(0.0, min(b, right) - max(a, left))


def request_batches(result):
    answer = {}
    for batch in result["summary"]["microbatch_metrics"]:
        if len(batch["member_request_ids"]) != 1:
            raise ValueError("This accounting requires one request per batch.")
        answer[batch["member_request_ids"][0]] = batch
    return answer


def read_case(path, with_trace=False):
    path = Path(path)
    result = read_json(path / "result.json.gz")
    manifest = read_json(path / "manifest.json.gz")
    answer = dict(directory=str(path), result=result, manifest=manifest,
                  requests={r["request_id"]: r for r in manifest["requests"]},
                  batches=request_batches(result))
    if with_trace:
        trace = read_json(path / "trace.json.gz")
        answer["trace_window"] = trace["window_ms"]
        answer["trace"] = [dict(zip(trace["columns"], row)) for row in trace["rows"]]
    return answer


def layer_view(batch, layer):
    current = batch["layer_metrics"][layer]
    previous = batch["layer_metrics"][layer - 1]
    admission = batch["admission_time_ms"]
    need = previous["compute_end_ms"]
    ready = current["io_ready_time_ms"]
    stall = current["compute_start_ms"] - need
    return dict(admission_ms=admission, layer=layer, io_start_ms=current["io_start_time_ms"],
                previous_compute_start_ms=previous["compute_start_ms"], deadline_ms=need,
                ready_ms=ready, compute_start_ms=current["compute_start_ms"],
                compute_end_ms=current["compute_end_ms"], layer_stall_ms=stall,
                io_start_since_admission_ms=current["io_start_time_ms"] - admission,
                deadline_since_admission_ms=need - admission, ready_since_admission_ms=ready - admission,
                compute_start_since_admission_ms=current["compute_start_ms"] - admission,
                ttft_ms=batch["completion_time_ms"] - admission,
                request_total_io_stall_ms=sum(m["io_barrier_wait_ms"] for m in batch["layer_metrics"]))


def select_victim(case):
    left, right = case["trace_window"]
    candidates = []
    for rid, batch in case["batches"].items():
        if case["requests"][rid]["load"]["role"] != "S":
            continue
        for layer in range(1, len(batch["layer_metrics"])):
            m = layer_view(batch, layer)
            if m["io_start_ms"] >= left - EPS and m["ready_ms"] <= right + EPS:
                candidates.append((m["layer_stall_ms"], rid, layer))
    if not candidates:
        raise ValueError("No fully traced S layer L1+ is available.")
    _, rid, layer = max(candidates)
    view = layer_view(case["batches"][rid], layer)
    if view["layer_stall_ms"] <= EPS:
        raise ValueError("The fully traced short layers have no positive exposed stall.")
    flows = [f for f in case["trace"] if f["request_id"] == rid and f["layer"] == layer]
    manifest_row = case["requests"][rid]
    expected = len(case["manifest"]["placements"][manifest_row["placement_index"]][0])
    if len(flows) != expected:
        raise AssertionError(f"Incomplete victim layer: got {len(flows)}, expected {expected} IOs")
    critical = max(flows, key=lambda f: (f["link_end_ms"], f["block_idx"]))
    assert abs(critical["link_end_ms"] - view["ready_ms"]) < EPS
    assert abs(view["ready_ms"] - view["compute_start_ms"]) < EPS
    assert critical["enqueue_ms"] <= view["deadline_ms"] + EPS
    return rid, layer, view, critical, len(candidates)


def account(case, rid, view, critical):
    left, right = critical["enqueue_ms"], critical["ssd_start_ms"]
    deadline, ready = view["deadline_ms"], view["ready_ms"]
    phases = []
    def add(category, start, end, flow=None):
        if end <= start + 1e-12:
            return
        duration = end - start
        exposed = overlap(start, end, deadline, ready)
        row = dict(category=category, description_zh=LABELS[category],
                   start_ms=start, end_ms=end, duration_ms=duration,
                   hidden_by_previous_compute_ms=duration - exposed,
                   exposed_npu_stall_ms=exposed, service_io_count=0,
                   service_bytes_during_interval=0.0, full_io_bytes=0,
                   serving_request_id=None, serving_npu_id=None, serving_layer=None,
                   serving_block_idx=None, full_io_service_ms=None)
        if flow is not None:
            service = flow["ssd_end_ms"] - flow["ssd_start_ms"]
            size_bytes = flow["size_gib"] * 2**30
            row.update(service_io_count=1, service_bytes_during_interval=size_bytes * duration / service,
                       full_io_bytes=round(size_bytes), serving_request_id=flow["request_id"],
                       serving_npu_id=flow["npu_id"], serving_layer=flow["layer"],
                       serving_block_idx=flow["block_idx"], full_io_service_ms=service)
        phases.append(row)

    serving = sorted((f for f in case["trace"] if f["ssu_id"] == critical["ssu_id"]
                      and overlap(f["ssd_start_ms"], f["ssd_end_ms"], left, right) > 1e-12),
                     key=lambda f: f["ssd_start_ms"])
    cursor = left
    for flow in serving:
        a, b = max(left, flow["ssd_start_ms"]), min(right, flow["ssd_end_ms"])
        if a < cursor - EPS:
            raise AssertionError("SSD service intervals overlap; serial-service accounting is invalid")
        if a > cursor + 1e-12:
            add("idle", cursor, a)
        role = case["requests"][flow["request_id"]]["load"]["role"]
        if flow["request_id"] == rid:
            category = "own_S"
        elif role == "L":
            category = "ahead_L"
        elif role == "S":
            category = "other_S"
        else:
            raise ValueError(f"Unexpected request role {role}")
        if flow["enqueue_ms"] > left + EPS:
            raise AssertionError("Later enqueued IO served before victim in asserted FIFO case")
        add(category, a, b, flow)
        cursor = b
    if right > cursor + 1e-12:
        add("idle", cursor, right)
    add("critical_ssd", critical["ssd_start_ms"], critical["ssd_end_ms"], critical)
    add("link_wait", critical["ssd_end_ms"], critical["link_start_ms"])
    add("link_service", critical["link_start_ms"], critical["link_end_ms"])
    total_duration = sum(p["duration_ms"] for p in phases)
    total_exposed = sum(p["exposed_npu_stall_ms"] for p in phases)
    assert math.isclose(total_duration, ready - left, abs_tol=EPS)
    assert math.isclose(total_exposed, view["layer_stall_ms"], abs_tol=EPS)
    aggregates = []
    for category in LABELS:
        subset = [p for p in phases if p["category"] == category]
        aggregates.append(dict(category=category, description_zh=LABELS[category],
                               **{key: sum(p[key] for p in subset) for key in
                                  ("duration_ms", "hidden_by_previous_compute_ms", "exposed_npu_stall_ms",
                                   "service_io_count", "service_bytes_during_interval", "full_io_bytes")}))
    return phases, aggregates


def save_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merged_intervals(case, rid, ssu, left, right):
    intervals = []
    for f in sorted(case["trace"], key=lambda f: f["ssd_start_ms"]):
        a, b = max(left, f["ssd_start_ms"]), min(right, f["ssd_end_ms"])
        if f["ssu_id"] != ssu or b <= a:
            continue
        role = case["requests"][f["request_id"]]["load"]["role"]
        category = "own_S" if f["request_id"] == rid else "ahead_L" if role == "L" else "other_S"
        if intervals and intervals[-1][2] == category and abs(intervals[-1][1] - a) < EPS:
            intervals[-1] = intervals[-1][0], b, category
        else:
            intervals.append((a, b, category))
    return intervals


def draw_figures(output, case, rid, layer, view, critical, aggregates, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    origin = min(view["previous_compute_start_ms"], critical["enqueue_ms"])
    left = origin
    # Stay inside captured SSD service coverage; show the current computation as far as recorded.
    right = min(view["compute_end_ms"], case["trace_window"][1])
    if right <= view["ready_ms"]:
        right = view["compute_end_ms"]
    span = right - left
    fig = plt.figure(figsize=(12, 7.2), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[0.75, 2.2, 1.65])
    header = fig.add_subplot(gs[0]); header.axis("off")
    header.text(0, 0.95, f"FIFO: a short request waits behind long-request IOs", fontsize=16, weight="bold", va="top")
    header.text(0, 0.46, f"Victim: NPU {critical['npu_id']} / request {rid} / layer L{layer}   |   Critical SSU {critical['ssu_id']} / block {critical['block_idx']}", va="top")
    header.text(0, 0.05, f"Absolute origin = {origin:.6f} ms. Labels below are milliseconds after origin.", color="#555555", va="top")
    ax = fig.add_subplot(gs[1])
    for a, b, category in merged_intervals(case, rid, critical["ssu_id"], left, right):
        ax.broken_barh([(a - origin, b - a)], (1.12, .40), facecolor=COLORS[category], edgecolor="none")
    a, b = max(left, critical["ssd_start_ms"]), min(right, critical["ssd_end_ms"])
    ax.broken_barh([(a - origin, b - a)], (1.08, .48), facecolor="#c63d54", edgecolor="none", zorder=5)
    ax.broken_barh([(view["previous_compute_start_ms"] - origin,
                     view["deadline_ms"] - view["previous_compute_start_ms"])], (.20, .44), facecolor=COLORS["compute"])
    ax.broken_barh([(view["deadline_ms"] - origin, view["layer_stall_ms"])], (.20, .44), facecolor=COLORS["stall"])
    if right > view["compute_start_ms"]:
        ax.broken_barh([(view["compute_start_ms"] - origin, right - view["compute_start_ms"])], (.20, .44), facecolor=COLORS["compute"])
    for time, label, y, color in [
        (critical["enqueue_ms"], "Block enqueued", 1.87, "#555555"),
        (critical["ssd_start_ms"], "Block SSD start", 1.65, "#a72d48"),
        (view["deadline_ms"], "Data needed", -.03, "#997000"),
        (view["ready_ms"], "All data ready", -.25, "#28764d")]:
        x = time - origin
        ax.axvline(x, ymin=.21, ymax=.77, linestyle="--", color=color, linewidth=.85, alpha=.85)
        ax.text(x, y, f"{label}\n{x:.4f} ms", ha="left" if x < span * .5 else "right", va="center", fontsize=8.6, color=color)
    center = (view["deadline_ms"] + view["ready_ms"]) / 2 - origin
    ax.text(center, .42, f"IO stall\n{view['layer_stall_ms']:.4f} ms", ha="center", va="center", fontsize=9, weight="bold")
    ax.set_yticks([.42, 1.32], [f"Victim NPU {critical['npu_id']}", f"SSU {critical['ssu_id']} service"])
    ax.set_ylim(-.55, 2.1); ax.set_xlim(-span*.025, span*1.025)
    ax.set_xlabel("Time after origin (ms)"); ax.grid(axis="x", alpha=.15)
    ax.legend(handles=[Patch(color=COLORS[k], label=v) for k, v in [
        ("ahead_L", "Long L"), ("other_S", "Other short S"), ("own_S", "Victim S"),
        ("compute", "Compute"), ("stall", "NPU stall")]], loc="upper center", bbox_to_anchor=(.5, -0.26), ncol=5, frameon=False)
    table_ax = fig.add_subplot(gs[2]); table_ax.axis("off")
    names = ["Ahead long L", "Other short S", "Victim's other blocks", "SSD idle",
             "Critical block SSD", "Critical block link wait", "Critical block transfer"]
    rows = [[names[i], f"{a['duration_ms']:.5f}", f"{a['hidden_by_previous_compute_ms']:.5f}",
             f"{a['exposed_npu_stall_ms']:.5f}"] for i, a in enumerate(aggregates)]
    rows.append(["TOTAL", f"{view['ready_ms'] - critical['enqueue_ms']:.5f}",
                 f"{view['deadline_ms'] - critical['enqueue_ms']:.5f}", f"{view['layer_stall_ms']:.5f}"])
    table = table_ax.table(cellText=rows, colLabels=["Critical block time account", "Duration (ms)", "Hidden (ms)", "Exposed stall (ms)"],
                           cellLoc="right", colWidths=[.43, .19, .19, .19], bbox=[0, .08, 1, .92])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0 or r == len(rows): cell.set_facecolor("#eef2f6"); cell.set_text_props(weight="bold")
        if c == 0: cell.set_text_props(ha="left")
    table_ax.text(0, -.04, "Each millisecond is counted once. SSD service before the data-needed deadline is hidden by computation.", fontsize=9, color="#555555")
    fig.savefig(output / "fifo_short_victim_timeline.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.2, 3.8), layout="constrained"); ax.axis("off")
    ax.text(0, 1, f"Same short request {rid}, same layer L{layer}", fontsize=16, weight="bold", va="top")
    ax.text(0, .86, "Timing is aligned to each policy's own request admission; absolute simulation times differ.", fontsize=10, va="top")
    rows = [[r["policy"], f"{r['io_start_since_admission_ms']:.4f}", f"{r['deadline_since_admission_ms']:.4f}",
             f"{r['ready_since_admission_ms']:.4f}", f"{r['layer_stall_ms']:.4f}", f"{r['ttft_ms']:.4f}"] for r in comparison]
    table = ax.table(cellText=rows, colLabels=["Policy", "Read start", "Data needed", "Data ready", "Layer stall", "Request TTFT"],
                     cellLoc="center", bbox=[0, .32, 1, .36]); table.auto_set_font_size(False); table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0: cell.set_facecolor("#eef2f6"); cell.set_text_props(weight="bold")
    delta = comparison[0]["layer_stall_ms"] - comparison[1]["layer_stall_ms"]
    ax.text(0, .19, f"All values in ms. Layer stall reduction: {delta:.4f} ms.\nMatched inputs, placement and Path0; queued IO selection order is the changed factor.", fontsize=11, va="top")
    fig.savefig(output / "same_victim_policy_comparison.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fifo-dir", type=Path, default=FORMAL / "s3_L19_S13_seed7_fifo")
    parser.add_argument("--control-dir", type=Path, default=FORMAL / "s3_L19_S13_seed7_short_first")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    fifo, control = read_case(args.fifo_dir, True), read_case(args.control_dir)
    assert fifo["manifest"]["input_fingerprint"] == control["manifest"]["input_fingerprint"], "Control inputs differ"
    rid, layer, view, critical, candidate_count = select_victim(fifo)
    phases, aggregates = account(fifo, rid, view, critical)
    long_groups = defaultdict(lambda: dict(service_ms=0.0, exposed_stall_ms=0.0, io_count=0))
    for phase in phases:
        if phase["category"] != "ahead_L":
            continue
        key = (phase["serving_request_id"], phase["serving_npu_id"], phase["serving_layer"])
        group = long_groups[key]
        group["service_ms"] += phase["duration_ms"]
        group["exposed_stall_ms"] += phase["exposed_npu_stall_ms"]
        group["io_count"] += phase["service_io_count"]
    long_request_contributions = sorted(
        [dict(request_id=k[0], npu_id=k[1], layer=k[2], **v) for k, v in long_groups.items()],
        key=lambda r: r["exposed_stall_ms"], reverse=True)
    comparison = [dict(policy="FIFO", **view),
                  dict(policy="Short first", **layer_view(control["batches"][rid], layer))]
    output = args.out_dir or args.fifo_dir / "diagnostics"; output.mkdir(parents=True, exist_ok=True)
    payload = dict(selection_zh="在 trace 完整覆盖的短类 S 的 L1-L7 中选择暴露 IO stall 最大的一层；这是局部最坏例，不代表平均值。",
                   fifo_dir=str(args.fifo_dir.resolve()), control_dir=str(args.control_dir.resolve()),
                   matched_input_fingerprint=fifo["manifest"]["input_fingerprint"],
                   trace_window_ms=fifo["trace_window"], candidate_layer_count=candidate_count,
                   request_id=rid, npu_id=critical["npu_id"], layer=layer, request_load=fifo["requests"][rid]["load"],
                   victim=view, critical_block=critical, accounting_by_category=aggregates,
                   ahead_long_request_contributions=long_request_contributions,
                   chronological_accounting=phases, policy_comparison=comparison,
                   accounting_notes_zh=["关键 block 是本层最后到达 NPU 的 block，所在 SSU 是本层数据到齐的关键路径。",
                       "盘队列等待按该 SSU 实际服务区间交集拆分为前方 L、其他 S、本请求其他 block 和空闲。",
                       "service_io_count 是与区间相交的 IO 个数；service_bytes_during_interval 按服务交集比例计字节；full_io_bytes 是这些 IO 完整大小。",
                       "链路等待是关键 block 的盘服务结束到链路传输开始；链路传输单独计算。",
                       "每阶段再与 [上一层计算结束, 本层计算开始) 求交，其和才是本层 NPU stall，不能加总各个 block 的排队时间。",
                       "deadline 是本层数据需求时刻（上一层计算结束），不是请求 TTFT SLO deadline。",
                       "跨策略对比同一 request_id 和同一层，以各自 admission 为零点；背景相位变化也属于调度顺序变化的系统效应。"],
                   checks=dict(input_identical=True, victim_layer_trace_complete=True, critical_end_equals_layer_ready=True,
                               critical_path_partition_exact=True, exposed_partition_equals_layer_stall=True,
                               serial_ssd_service=True, fifo_no_later_enqueue_overtakes=True))
    write_json(output / "fifo_short_victim_accounting.json", payload)
    save_csv(output / "fifo_short_victim_time_account.csv", aggregates)
    save_csv(output / "fifo_short_victim_service_segments.csv", phases)
    if long_request_contributions:
        save_csv(output / "ahead_long_request_contributions.csv", long_request_contributions)
    save_csv(output / "same_victim_policy_comparison.csv", comparison)
    draw_figures(output, fifo, rid, layer, view, critical, aggregates, comparison)
    long_row = next(a for a in aggregates if a["category"] == "ahead_L")
    report = [f"# FIFO 短流阻塞的局部证据\n", f"请求 **{rid}**，NPU **{critical['npu_id']}**，第 **L{layer}** 层。",
              f"选择范围为 {fifo['trace_window'][0]:.3f}–{fifo['trace_window'][1]:.3f} ms 内完整覆盖的 {candidate_count} 个短类层；展示其中 stall 最大的层，不能将其当作平均值。\n",
              f"本层数据在 **{view['deadline_ms']:.6f} ms** 需要就绪，实际在 **{view['ready_ms']:.6f} ms** 到齐；NPU 暴露等待 **{view['layer_stall_ms']:.6f} ms**。",
              f"最后到达的 block {critical['block_idx']} 位于 SSU {critical['ssu_id']}。该 block 入队后、获得盘服务前，前方 L 类占用盘服务 **{long_row['duration_ms']:.6f} ms**，其中落在 NPU 暴露等待区间的部分为 **{long_row['exposed_npu_stall_ms']:.6f} ms**。\n",
              f"上述 L 类服务来自 **{len(long_request_contributions)} 个长请求层** 的累计贡献，不能解读为单个长请求独占这段时间；逐请求贡献见 `ahead_long_request_contributions.csv`。\n",
              "| 时间段 | 总时长 ms | 被计算掩盖 ms | 暴露 NPU stall ms | 相交 IO 数 |\n|---|---:|---:|---:|---:|"]
    report.extend(f"| {a['description_zh']} | {a['duration_ms']:.6f} | {a['hidden_by_previous_compute_ms']:.6f} | {a['exposed_npu_stall_ms']:.6f} | {a['service_io_count']} |" for a in aggregates)
    report += ["\n**以上是关键路径的不重叠时间账，不是将每个 IO 的排队时间相加。**\n",
               "同一请求、同一层在两策略中的比较（读开始/需要/到齐时间均相对本策略的请求接纳时刻）：\n",
               "| 策略 | 读开始 ms | 数据需要 ms | 数据到齐 ms | 本层 stall ms | 请求 TTFT ms |\n|---|---:|---:|---:|---:|---:|"]
    report.extend(f"| {r['policy']} | {r['io_start_since_admission_ms']:.6f} | {r['deadline_since_admission_ms']:.6f} | {r['ready_since_admission_ms']:.6f} | {r['layer_stall_ms']:.6f} | {r['ttft_ms']:.6f} |" for r in comparison)
    report += ["\n相同输入、放置和 Path0 配置，改变已入队 IO 的选择顺序；调度也会改变整个系统后续的读写相位，因此不能把不同策略的同一绝对时刻直接比较。",
               "\n这里的长/短按本实验输入类别 L/S 标识；每次实际服务的 IO 大小仍相同，长类的一层包含更多 IO。",
               "\n数据文件：`fifo_short_victim_accounting.json` 包含精确时间、完整检查和说明；`fifo_short_victim_time_account.csv` 为分类时间账；`fifo_short_victim_service_segments.csv` 为逐个服务区间。"]
    (output / "fifo_short_victim_explained.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output.resolve()), request_id=rid, layer=layer,
                          fifo_stall_ms=view["layer_stall_ms"], control_stall_ms=comparison[1]["layer_stall_ms"],
                          ahead_L_queue_service_ms=long_row["duration_ms"],
                          ahead_L_exposed_ms=long_row["exposed_npu_stall_ms"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
