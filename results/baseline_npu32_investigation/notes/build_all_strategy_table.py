"""Read-only, repeatable table of every planned screen/formal/holdout job.

Missing results remain pending; policy comparisons require the same complete
frozen input fingerprint, native submission seed, and core simulator hashes.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

STAGES = ("screen", "formal", "holdout")
CORE = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py", "policy_logic.py")
NEW = ("strategy1", "strategy2", "strategy3")
COFLOW = ("baseline", "once", "new_once") + NEW
LABELS = {"baseline": "Baseline", "baseline_native": "Baseline", "once": "Once",
          "once_native": "Once", "new_once": "New once", "strategy1": "S1",
          "strategy2": "S2", "strategy3": "S3", "deadline": "A deadline",
          "stall_interchange": "Stall interchange"}


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_mode(strategy, config):
    if strategy in NEW:
        return "5ms/" + config.get("assignment", "pipeline")
    return "5ms/fixed" if strategy in COFLOW else "native/fixed"


def profile_kind(metadata):
    methods = {p["construction"]["method"] for p in metadata["profiles"]}
    direct = methods == {"direct_data_row"}
    scaled = metadata["compute_scale_actual"] != 1.0
    padded = metadata["padding_total_gib"] != 0.0
    parts = ["原始data参数" if direct else "外推/插值画像"]
    if scaled:
        parts.append("C缩放")
    if padded:
        parts.append("尾块补齐")
    return "+".join(parts)


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}"


def num(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def fraction(row, origin):
    count = row.get("request_count")
    passed = row.get(f"{origin}_slo_passed")
    return "—" if passed is None else f"{passed}/{count}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, output = args.root, args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, inputs = [], [], {}
    generated = datetime.now(timezone.utc).isoformat()
    metric_fields = ("u_w1", "u_w2", "makespan_ms", "full_run_u", "admission_slo_passed",
                     "arrival_slo_passed", "admission_slo_rate", "arrival_slo_rate",
                     "active_cards_w1", "active_cards_w2", "all_active_w1", "all_active_w2",
                     "delta_u_w1_pp", "delta_u_w2_pp", "makespan_change_pct")
    for stage in STAGES:
        plan_path = root / stage / f"{stage}_plan.json"
        if not plan_path.exists():
            plan_path = root / f"{stage}_plan.json"
        plan = read(plan_path)
        sources.append({"kind": "plan", "path": str(plan_path), "sha256": sha(plan_path)})
        metadata_by_label = {}
        for label in plan["inputs"]:
            path = root / stage / "inputs" / f"{label}.json.gz"
            if not path.exists():
                metadata_by_label[label] = None
                continue
            payload = read(path)
            metadata = payload["metadata"]
            assert metadata["input_fingerprint"] == payload["input_fingerprint"]
            metadata_by_label[label] = metadata
            fp = payload["input_fingerprint"]
            if fp not in inputs:
                inputs[fp] = {"metadata": metadata, "aliases": [], "manifest_files": []}
            inputs[fp]["aliases"].append(f"{stage}/{label}")
            inputs[fp]["manifest_files"].append({"path": str(path), "sha256": sha(path)})
        for job in plan["jobs"]:
            label, strategy = job["input"], job["strategy"]
            variant = job.get("variant", strategy)
            config = job.get("config", {})
            metadata = metadata_by_label[label]
            row = {"stage": stage, "input_label": label, "strategy": strategy, "variant": variant,
                   "effective_mode": effective_mode(strategy, config),
                   "status": "pending", "status_detail": "未完成或结果尚未同步到本地",
                   "input_fingerprint": metadata["input_fingerprint"] if metadata else "",
                   "num_npu": metadata["num_npu"] if metadata else None,
                   "num_ssu": metadata["num_ssu"] if metadata else None,
                   "input_kind": profile_kind(metadata) if metadata else "manifest pending",
                   "compute_scale": metadata["compute_scale_actual"] if metadata else None,
                   "rho_ssu": metadata["input_demand"]["hottest_ssu_load_ratio"] if metadata else None,
                   "capacity_feasible": metadata["load_within_disk_and_link_capacity"] if metadata else None,
                   "request_count": metadata["request_count"] if metadata else None,
                   "submit_seed": metadata["seed"] if metadata else None,
                   **{key: None for key in metric_fields},
                   "baseline_stage": "", "baseline_variant": "", "regresses_window_u": None,
                   "result_path": "", "result_sha256": "", "core_signature": ""}
            directory = root / stage / "runs" / label / variant
            paths = sorted(directory.glob("*.json.gz"))
            failures = sorted(directory.glob("*.failure.json"))
            if not paths and failures:
                row.update(status="failed", status_detail="; ".join(str(p) for p in failures))
            elif len(paths) > 1:
                row.update(status="invalid", status_detail="结果目录存在多个文件，未任意选择")
            elif len(paths) == 1:
                path = paths[0]
                row["result_path"] = str(path)
                try:
                    result = read(path)
                    summary = result["summary"]
                    assert metadata is not None, "frozen manifest is not locally available"
                    assert result["strategy"] == strategy
                    assert result["input_fingerprint"] == metadata["input_fingerprint"]
                    assert result["submit_seed"] == metadata["seed"]
                    assert all(summary["invariants"].values())
                    assert summary["request_count"] == metadata["request_count"]
                    if strategy in NEW:
                        assert result["policy_config"]["assignment"] == config.get("assignment", "pipeline")
                    elif strategy not in COFLOW:
                        assert result.get("assignment", "none") == "none"
                    windows = {(w["start_ms"], w["end_ms"]): w for w in result["windows"]}
                    for number, key in enumerate(((1000, 2000), (2000, 3000)), 1):
                        window = windows[key]
                        assert abs(sum(window["npu_utilizations"]) / metadata["num_npu"]
                                   - window["mean_npu_utilization"]) < 1e-10
                        duration = key[1] - key[0]
                        active = sum(abs(a - duration) < 1e-7 for a in window["active_ms_by_npu"])
                        assert (active == metadata["num_npu"]) == window["all_npus_active_whole_window"]
                        row[f"u_w{number}"] = window["mean_npu_utilization"]
                        row[f"active_cards_w{number}"] = active
                        row[f"all_active_w{number}"] = active == metadata["num_npu"]
                    request_rows = summary["request_metrics"]
                    assert len(request_rows) == metadata["request_count"]
                    for origin in ("admission", "arrival"):
                        passed = sum(r["completion_time_ms"] - r[f"{origin}_time_ms"]
                                     <= 1.5 * r["own_compute_ms"] + 1e-9 for r in request_rows)
                        assert passed == result["slo"]["all_requests"][origin]["passed"]
                        row[f"{origin}_slo_passed"] = passed
                        row[f"{origin}_slo_rate"] = passed / len(request_rows)
                    core = {name: result["core_and_policy_sha256"][name] for name in CORE}
                    row.update(status="complete", status_detail="",
                               makespan_ms=summary["makespan_ms"], full_run_u=summary["fleet_npu_compute_utilization"],
                               core_signature=json.dumps(core, sort_keys=True), result_sha256=sha(path))
                    sources.append({"kind": "result", "path": str(path), "sha256": row["result_sha256"],
                                    "input_fingerprint": row["input_fingerprint"],
                                    "core_and_policy_sha256": result["core_and_policy_sha256"],
                                    "stress_runner_sha256": result["stress_runner_sha256"]})
                except (OSError, EOFError, json.JSONDecodeError) as exc:
                    row.update(status="pending", status_detail="本地文件尚不可完整读取: " + repr(exc))
                    row.update({key: None for key in metric_fields})
                except (AssertionError, KeyError, ValueError) as exc:
                    row.update(status="invalid", status_detail="完整性/配对校验失败: " + repr(exc))
                    row.update({key: None for key in metric_fields})
            rows.append(row)
    baselines = defaultdict(list)
    pair_key = lambda r: (r["input_fingerprint"], r["submit_seed"], r["core_signature"])
    for row in rows:
        if row["status"] == "complete" and row["strategy"] in ("baseline", "baseline_native"):
            baselines[pair_key(row)].append(row)
    for row in rows:
        if row["status"] != "complete":
            continue
        candidates = baselines[pair_key(row)]
        if not candidates:
            row["status_detail"] = "结果完成；匹配Baseline尚未就绪，差值pending"
            continue
        baseline = min(candidates, key=lambda r: r["effective_mode"] != row["effective_mode"])
        if len(candidates) > 1:
            assert all(abs(r["u_w1"] - baseline["u_w1"]) < 1e-8
                       and abs(r["u_w2"] - baseline["u_w2"]) < 1e-8 for r in candidates), "multiple baselines differ"
        row.update(baseline_stage=baseline["stage"], baseline_variant=baseline["variant"],
                   delta_u_w1_pp=100 * (row["u_w1"] - baseline["u_w1"]),
                   delta_u_w2_pp=100 * (row["u_w2"] - baseline["u_w2"]),
                   makespan_change_pct=100 * (row["makespan_ms"] / baseline["makespan_ms"] - 1))
        row["regresses_window_u"] = row["delta_u_w1_pp"] < -1e-8 or row["delta_u_w2_pp"] < -1e-8
    with (output / "all_strategy_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = Counter(r["status"] for r in rows)
    negative = [r for r in rows if r["regresses_window_u"]]
    tradeoffs = [r for r in rows if r["delta_u_w1_pp"] is not None
                 and r["delta_u_w1_pp"] > 1e-8 and r["makespan_change_pct"] > 1e-8]
    lines = ["# 当前全部计划策略表", "", f"生成时间UTC：{generated}。计划{len(rows)}作业：" +
             "，".join(f"{status}={counts[status]}" for status in ("complete", "pending", "failed", "invalid")) + "。", "",
             "本表只读取本地已同步的完整结果。pending表示未完成或尚未同步，不赋值为0；完整结果若缺匹配Baseline，差值也保持空。按完整冻结input_fingerprint、submit seed和四个核心源码哈希配对，不按名称猜测相同输入。", "",
             "模式：5ms/fixed保留原分卡，5ms/pipeline仅S1–S3实际启用到达选卡；baseline/Once/New once忽略CLI中的assignment，仍属fixed。native/fixed为原生即时压力/控制分支，deadline等不会因命令写assignment=pipeline而重分卡。跨模式差异包含不同遥测/控制权限。", "",
             "U1=[1000,2000]ms，U2=[2000,3000]ms，单位%；接纳/到达SLO均为完整请求集completion−对应起点≤1.5×8C。SLO列为通过数/总数；接纳排队未计入接纳SLO。active为两个窗口分别全程有已接纳工作的卡数。", "",
             "所有输入的配额、到达和placement均为构造；“原始data参数”仅表示画像数值未经外推/插值/C缩放/尾块补齐，不表示生产到达分布。", "", "## 已完成的负例：任一窗口低于同输入Baseline", "",
             "只筛选已完成的严格配对；负差全部保留，包括小差异，不据此做统计显著性推断。Δmakespan正值表示完整排空更慢。", "",
             "| 输入 | 策略/模式 | ΔU1/pp | ΔU2/pp | Δmakespan/% | active |",
             "|---|---|---:|---:|---:|---|"]
    for r in sorted(negative, key=lambda r: min(r["delta_u_w1_pp"], r["delta_u_w2_pp"])):
        lines.append(f'| {r["input_label"]} | {LABELS.get(r["strategy"], r["strategy"])} / {r["effective_mode"]} | {r["delta_u_w1_pp"]:+.3f} | {r["delta_u_w2_pp"]:+.3f} | {r["makespan_change_pct"]:+.2f} | {r["active_cards_w1"]},{r["active_cards_w2"]}/{r["num_npu"]} |')
    if not negative:
        lines.append("| 尚无已完成负例 | — | — | — | — | — |")
    lines.extend(["", "## 主窗口U更高但完整排空更慢", "",
                  "同样只列已完成的同输入配对；窗口U收益不能替代完整吞吐证据。", "",
                  "| 输入 | 策略/模式 | ΔU1/pp | ΔU2/pp | Δmakespan/% |",
                  "|---|---|---:|---:|---:|"])
    for r in sorted(tradeoffs, key=lambda r: -r["makespan_change_pct"]):
        lines.append(f'| {r["input_label"]} | {LABELS.get(r["strategy"], r["strategy"])} / {r["effective_mode"]} | {r["delta_u_w1_pp"]:+.3f} | {r["delta_u_w2_pp"]:+.3f} | {r["makespan_change_pct"]:+.2f} |')
    if not tradeoffs:
        lines.append("| 尚无已完成匹配项 | — | — | — | — |")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["input_fingerprint"] or f'{row["stage"]}/{row["input_label"]}'].append(row)
    for fp, group in grouped.items():
        first = group[0]
        aliases = inputs.get(fp, {}).get("aliases", [f'{first["stage"]}/{first["input_label"]}'])
        lines.extend(["", "## " + " / ".join(dict.fromkeys(a.split("/", 1)[1] for a in aliases)), "",
                      f'冻结指纹：`{fp}`。{first["input_kind"]}；{first["num_npu"]}NPU/{first["num_ssu"]}SSU；ρmax={num(first["rho_ssu"], 4)}；C倍率={num(first["compute_scale"], 6)}；容量条件={first["capacity_feasible"]}。', "",
                      "| 来源 | 策略/模式 | 状态 | U1% | U2% | makespan ms | 整批U% | 接纳SLO | 到达SLO | active |",
                      "|---|---|---|---:|---:|---:|---:|---|---|---|"])
        for r in group:
            active = "—" if r["all_active_w1"] is None else f'{r["active_cards_w1"]},{r["active_cards_w2"]}/{r["num_npu"]}'
            lines.append(f'| {r["stage"]} | {LABELS.get(r["strategy"], r["strategy"])} / {r["effective_mode"]} | {r["status"]} | {pct(r["u_w1"])} | {pct(r["u_w2"])} | {num(r["makespan_ms"])} | {pct(r["full_run_u"])} | {fraction(r,"admission")} | {fraction(r,"arrival")} | {active} |')
        issues = [r for r in group if r["status"] in ("failed", "invalid") or (r["status"] == "complete" and r["status_detail"])]
        for r in issues:
            lines.extend(["", f'{r["variant"]}: {r["status_detail"]}'])
    lines.extend(["", "复算：`python results/baseline_npu32_investigation/notes/build_all_strategy_table.py`。CSV中U/rate为0–1，delta_u为百分点；完整路径、结果SHA及源代码指纹保存在CSV/JSON。", ""])
    (output / "all_strategy_table.md").write_text("\n".join(lines))
    (output / "all_strategy_table.json").write_text(json.dumps({
        "generated_utc": generated, "counts": dict(counts), "planned_jobs": len(rows),
        "rows": rows, "input_manifests": inputs, "sources": sources,
        "negative_case_count": len(negative), "higher_u_slower_makespan_count": len(tradeoffs),
        "pairing_keys": ["input_fingerprint", "submit_seed", *CORE]},
        ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"planned": len(rows), "status": dict(counts), "negative_cases": len(negative),
                      "markdown": str(output / "all_strategy_table.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
