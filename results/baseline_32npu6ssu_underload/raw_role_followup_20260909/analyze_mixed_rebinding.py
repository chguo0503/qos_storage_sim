#!/usr/bin/env python3
"""Independent global-source and per-NPU order-pair audit; no simulations.

All planned statuses and scientific failures are retained. Source populations
and raw profile keys are read dynamically, supporting the v4/v5 designs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

FOLLOWUP = Path(__file__).resolve().parent
HERE = FOLLOWUP / "mixed_rebinding"
ROOT = FOLLOWUP.parent.parents[1]
sys.path[:0] = [str(ROOT), str(FOLLOWUP), str(FOLLOWUP.parent)]
import analyze_followup as base
base.HERE = HERE
from run_baseline_npu32_stress import load_manifest, read_json, write_json
from run_coflow_experiments import source_files

START, END = 2000.0, 4000.0
BLOCK = 176 * 1024 / 2**30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, message):
    if not value:
        raise AssertionError(message)


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return path
    for root in (ROOT, FOLLOWUP, HERE):
        if (root / path).exists():
            return root / path
    return ROOT / path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_input(item, table):
    path = resolve(item["manifest"])
    check(sha(path) == item["manifest_sha256"], "input bytes differ from plan")
    requests, meta = load_manifest(path)
    source_path = resolve(meta["source_fixed_manifest"])
    check(sha(source_path) == meta["source_fixed_manifest_sha256"], "fixed source bytes changed")
    if item.get("source_manifest"):
        check(resolve(item["source_manifest"]).resolve() == source_path.resolve(), "plan/source path mismatch")
    if item.get("source_sha256"):
        check(item["source_sha256"] == sha(source_path), "plan/source hash mismatch")
    source, source_meta = load_manifest(source_path)
    check(meta["source_fixed_input_fingerprint"] == source_meta["input_fingerprint"], "source fingerprint mismatch")
    check(meta["input_fingerprint"] == item["input_fingerprint"], "input fingerprint mismatch")
    check(meta["label"] == item["label"] and meta["seed"] == item["seed"], "input label/seed mismatch")
    check(meta["order"] == item["mode"] and item["mode"] in ("random", "ordered"), "wrong order mode")
    check((meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (32, 6, 8), "wrong dimensions")
    check(meta["long_cards"] is None and meta["short_cards"] is None, "new mixed lanes labelled as fixed roles")
    check(meta["source_data_sha256"] == sha(ROOT / "data"), "data source changed")
    check(len(requests) == len(source) == meta["request_count"], "source population size changed")
    byid = {r.request_id: r for r in source}
    check(len(byid) == len(source), "duplicate source identity")
    seen, science, lanes = [], {}, defaultdict(list)
    maxima = [[0.0] * 6 for _ in range(32)]
    link_max = [0.0] * 32
    for r in requests:
        load = r.load
        oldid = load["source_fixed_request_id"]
        check(oldid in byid, "foreign fixed source identity")
        old = byid[oldid]
        check(load["source_fixed_npu_id"] == old.npu_id, "wrong source fixed NPU")
        check(r.arrival_time_ms == old.arrival_time_ms == 0.0, "arrival changed")
        check(r.placement == old.placement, "physical placement changed on rebinding")
        check(all(load[k] == v for k, v in old.load.items() if k not in ("request_id", "npu_id", "generation")),
              "old scientific/provenance field changed")
        check(set(load) - set(old.load) == {"source_fixed_request_id", "source_fixed_npu_id"}, "unexpected new load fields")
        n, rid = r.npu_id, r.request_id
        check(0 <= n < 32 and load["npu_id"] == n and load["request_id"] == rid, "new binding/ID mismatch")
        check(rid == n * 1_000_000 + load["generation"], "position/generation mismatch")
        key = (int(load["seq_len_k"]), int(load["nql"]))
        check(key in table, "profile absent from raw data")
        bw, cu, ttft, raw_v = table[key]
        check(load["per_layer_us"] == load["original_compute_us"] == cu, "C not exact raw value")
        check(load["source_ttft_ms"] == ttft, "source 78-layer TTFT altered")
        check(load["constructed_profile"] is False and load["padding_gib_per_layer"] == 0,
              "constructed or padded profile")
        check(load["profile_construction"]["method"] == "direct_data_row", "not direct raw profile")
        check(load["category"] == ("S" if key[0] <= 80 else "L") + ("S" if key[1] < 512 else "L"), "wrong category")
        check(load["role"] == ("short" if key[0] <= 80 else "long"), "wrong analysis role")
        check(len(r.placement) in (1, 8) and all(layer == r.placement[0] for layer in r.placement), "layer placement varies")
        check(all(0 <= s < 6 and v == BLOCK for layer in r.placement for s, v in layer), "non-176KiB or foreign disk")
        volume = math.fsum(v for _, v in r.placement[0])
        check(math.isclose(volume, raw_v, rel_tol=0, abs_tol=1e-12)
              and math.isclose(load["per_layer_kv_gb"], raw_v, rel_tol=0, abs_tol=1e-12), "raw V changed")
        rates = [math.fsum(v for s, v in r.placement[0] if s == d) * 1e6 / cu for d in range(6)]
        maxima[n] = [max(a, b) for a, b in zip(maxima[n], rates)]
        link_max[n] = max(link_max[n], math.fsum(rates))
        science[oldid] = digest({"source_id": oldid, "new_npu": n, "arrival": r.arrival_time_ms,
            "load": {k: v for k, v in load.items() if k not in ("request_id", "generation")}, "placement": r.placement})
        seen.append(oldid)
        lanes[n].append(r)
    check(len(seen) == len(set(seen)) == len(byid) and set(seen) == set(byid), "global source population not bijective")
    description = meta["new_binding_design"]
    check(description["seed"] == item["seed"] and description["mode"] == item["mode"],
          "design seed/mode mismatch")
    check(description["requests"] == len(requests), "design population size mismatch")
    check(meta["per_npu_assignment"] == description["per_npu_assignment"],
          "top-level assignment metadata differs from design")
    assignment = {r["npu_id"]: r for r in description["per_npu_assignment"]}
    ordering = {r["npu_id"]: r for r in description["per_npu_order"]}
    profile_keys = [tuple(k) for k in description["profile_keys"]]
    global_counts = Counter((int(r.load["seq_len_k"]), int(r.load["nql"])) for r in requests)
    check([global_counts[k] for k in profile_keys] == description["global_profile_counts"], "global profile metadata mismatch")
    per_npu = []
    for n in range(32):
        lane = sorted(lanes[n], key=lambda r: r.request_id)
        check([r.load["generation"] for r in lane] == list(range(len(lane))), "non-contiguous new positions")
        identities = [r.load["source_fixed_request_id"] for r in lane]
        profiles = [profile_keys.index((int(r.load["seq_len_k"]), int(r.load["nql"]))) for r in lane]
        counts = [profiles.count(p) for p in range(len(profile_keys))]
        pure = 8 * math.fsum(r.load["per_layer_us"] / 1000 for r in lane)
        check(counts == assignment[n]["profile_counts"] and sorted(identities) == assignment[n]["source_request_ids"], "assignment description mismatch")
        check(profiles == ordering[n]["profile_sequence"] and identities == ordering[n]["source_request_ids"], "order description mismatch")
        check(math.isclose(pure, assignment[n]["pure_compute_ms"], abs_tol=1e-8), "per-card pure C mismatch")
        per_npu.append({"npu_id": n, "request_count": len(lane), "profile_counts": counts, "pure_compute_ms": pure,
            "source_fixed_ids": sorted(identities), "has_both_roles_in_input": {r.load["role"] for r in lane} == {"short", "long"}})
    bounds = [math.fsum(v[d] for v in maxima) for d in range(6)]
    stored = meta["active_profile_rate_certificate"]
    check(all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(bounds, stored["per_ssu_upper_bound_gib_s"])), "static disk proof mismatch")
    check(all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(link_max, stored["per_npu_receive_upper_bound_gib_s"])), "static link proof mismatch")
    summary = {"label": item["label"], "spec_name": item["spec_name"], "seed": item["seed"], "mode": item["mode"],
        "status": "passed", "manifest": str(path), "manifest_sha256": sha(path), "input_fingerprint": meta["input_fingerprint"],
        "source_fixed_manifest": str(source_path), "source_fixed_manifest_sha256": sha(source_path),
        "source_fixed_input_fingerprint": source_meta["input_fingerprint"], "request_count": len(requests),
        "global_profile_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(global_counts.items())},
        "all_global_scientific_fields_and_placement_preserved": True, "per_npu": per_npu,
        "all_per_npu_pure_compute_above_4000": all(n["pure_compute_ms"] > END for n in per_npu),
        "input_all_npus_have_both_roles": all(n["has_both_roles_in_input"] for n in per_npu),
        "static_per_ssu_upper_bound_gib_s": bounds, "static_max_ssu_gib_s": max(bounds),
        "static_per_npu_link_upper_bound_gib_s": link_max}
    return requests, summary, science


def role_counts(raw, requests):
    """Count actual active roles; preserve idle states instead of rejecting them."""
    roles = {r.request_id: r.load["role"] for r in requests}
    events = defaultdict(lambda: {"add": [], "remove": []})
    events[START]; events[END]
    for b in raw["summary"]["microbatch_metrics"]:
        a, z = max(START, b["admission_time_ms"]), min(END, b["completion_time_ms"])
        if z > a:
            pair = (b["npu_id"], b["member_request_ids"][0])
            events[a]["add"].append(pair); events[z]["remove"].append(pair)
    active, hist, joint, intervals = {}, defaultdict(float), defaultdict(float), []
    times = sorted(events)
    for j, t in enumerate(times):
        for n, rid in events[t]["remove"]:
            check(active.get(n) == rid, "unmatched role completion")
            del active[n]
        for n, rid in events[t]["add"]:
            check(n not in active, "two admitted requests on same NPU")
            active[n] = rid
        if j + 1 == len(times):
            break
        dt = times[j + 1] - t
        long = sum(roles[r] == "long" for r in active.values())
        short = sum(roles[r] == "short" for r in active.values())
        check(long + short == len(active) <= 32, "unknown role or invalid active count")
        hist[long] += dt; joint[long, short] += dt
        intervals.append({"start_ms": t, "end_ms": times[j + 1], "long_cards": long,
                          "short_cards": short, "active_cards": len(active), "idle_cards": 32 - len(active)})
    check(not active and math.isclose(math.fsum(hist.values()), END - START, abs_tol=1e-7), "incomplete role timeline")
    duration = END - START
    average_long = math.fsum(n * dt for n, dt in hist.items()) / duration
    average_short = math.fsum(s * dt for (l, s), dt in joint.items()) / duration
    return {"mean_long_cards": average_long, "mean_short_cards": average_short,
        "mean_idle_cards": 32 - average_long - average_short, "min_long_cards": min(hist), "max_long_cards": max(hist),
        "min_active_cards": min(l + s for l, s in joint), "max_active_cards": max(l + s for l, s in joint),
        "fraction_all_32_active": math.fsum(dt for (l, s), dt in joint.items() if l + s == 32) / duration,
        "fraction_long_exactly_20": hist.get(20, 0.0) / duration,
        "fraction_long_18_to_22": math.fsum(dt for n, dt in hist.items() if 18 <= n <= 22) / duration,
        "fraction_long_16_to_24": math.fsum(dt for n, dt in hist.items() if 16 <= n <= 24) / duration,
        "histogram_ms": dict(sorted(hist.items())), "joint_histogram_ms": [
            {"long_cards": l, "short_cards": s, "duration_ms": dt} for (l, s), dt in sorted(joint.items())], "intervals": intervals}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    jobs, items, plans, errors = {}, {}, [], []
    expected_core = set(source_files())
    for p in sorted((HERE / "plans").glob("*.json")):
        plan = read_json(p)
        try:
            check(sha(resolve(plan["spec_file"])) == plan["spec_sha256"], "spec bytes changed")
            check(plan["window_ms"] == [2000, 4000], "wrong planned window")
            check(expected_core.issubset(plan["source_sha256"]), "plan lacks core files")
            for name, value in plan["source_sha256"].items():
                snapshot = HERE / "sources" / value / Path(name).name
                check(sha(snapshot) == value, f"snapshot changed: {name}")
                if not name.startswith("results/"):
                    check(sha(ROOT / name) == value, f"core/data changed: {name}")
            plans.append({"path": str(p), "sha256": sha(p), "status": "passed"})
        except Exception as e:
            errors.append({"scope": "plan", "path": str(p), "error": f"{type(e).__name__}: {e}"})
        # Input-only controls (strategies=[]) are still audited and paired.
        # They are deliberately absent from the planned simulation count.
        plan_items = {i["label"]: i for i in plan["inputs"]}
        for item in plan_items.values():
            label = item["label"]
            if label in items:
                check(all(items[label][k] == item[k] for k in ("input_fingerprint", "manifest_sha256", "seed", "mode", "spec_name")), "conflicting duplicate input")
            items[label] = item
        for j in plan["jobs"]:
            item = j["input"]; label = item["label"]
            check(label in plan_items and item == plan_items[label], "job input differs from plan inputs")
            key = label, j["strategy"]
            if key in jobs:
                check(jobs[key]["input"]["input_fingerprint"] == item["input_fingerprint"], "conflicting duplicate job")
            jobs[key] = {**j, "_plan_sources": plan["source_sha256"]}
    table = ast.literal_eval((ROOT / "data").read_text())
    input_audits, payloads = {}, {}
    for label, item in items.items():
        try:
            requests, record, science = audit_input(item, table)
            input_audits[label] = record; payloads[label] = (requests, science)
        except Exception as e:
            record = {"label": label, "status": "audit_failed", "error": f"{type(e).__name__}: {e}"}
            input_audits[label] = record; errors.append({"scope": "input", **record})
    order_pairs, pair_valid = [], {}
    grouped = defaultdict(dict)
    for item in items.values():
        grouped[item["spec_name"], item["seed"]][item["mode"]] = item["label"]
    for (spec_name, seed), modes in sorted(grouped.items()):
        record = {"spec_name": spec_name, "seed": seed, "labels": modes, "status": "pending"}
        if set(modes) == {"random", "ordered"} and all(label in payloads for label in modes.values()):
            a, b = (input_audits[modes[k]] for k in ("random", "ordered"))
            passed = (a["source_fixed_manifest_sha256"] == b["source_fixed_manifest_sha256"]
                      and payloads[modes["random"]][1] == payloads[modes["ordered"]][1])
            record.update(status="passed" if passed else "audit_failed", same_per_npu_population_C_V_arrival_placement=passed)
            if not passed:
                errors.append({"scope": "order_pair", **record})
        order_pairs.append(record)
        for label in modes.values():
            pair_valid[label] = record["status"] == "passed"
    rows, role_records = [], {}
    for (label, strategy), job in sorted(jobs.items()):
        item = job["input"]
        row = {"label": label, "spec_name": item["spec_name"], "seed": item["seed"], "mode": item["mode"],
               "strategy": strategy, "status": "pending", "audit_pass": False,
               "order_pair_pass": pair_valid.get(label, False)}
        directory = HERE / "runs" / label / strategy
        paths = list(directory.glob("*.json.gz"))
        command = read_json(directory / "command.json") if (directory / "command.json").exists() else {}
        if command.get("status") in ("failed", "timeout", "audit_failed"):
            row.update(status=command["status"], returncode=command.get("returncode"), error=command.get("error"))
        elif not paths and command.get("status") == "complete":
            row.update(status="audit_failed", error="command completed but result file is missing")
        elif len(paths) > 1:
            row.update(status="audit_failed", error="multiple result files")
        elif len(paths) == 1 and command.get("status") == "complete":
            try:
                check(label in payloads, "input audit failed")
                raw = read_json(paths[0])
                check(raw["strategy"] == strategy and raw["input_fingerprint"] == item["input_fingerprint"], "wrong policy/input")
                check(set(raw["core_and_policy_sha256"]) == expected_core, "incomplete core hash set")
                check(all(job["_plan_sources"].get(n) == h for n, h in raw["core_and_policy_sha256"].items()), "result/plan source mismatch")
                check(all(command["source_sha256"].get(n) == h for n, h in job["_plan_sources"].items()), "command/plan source mismatch")
                out = base.analyze(item, strategy, paths[0])
                row.update(out["row"])
                counts = role_counts(raw, payloads[label][0])
                check(math.isclose(counts["mean_long_cards"] * 2000, out["technical"]["windows"][0]["by_role"]["long"]["active_ms"], abs_tol=1e-6), "long card-time mismatch")
                write_json(HERE / "role_counts" / label / f"{strategy}.json", counts)
                role_records[label, strategy] = counts
                row.update({k: v for k, v in counts.items() if k not in ("histogram_ms", "joint_histogram_ms", "intervals")})
                cards = out["technical"]["windows"][0]["per_npu"]
                for role in ("short", "long"):
                    row[f"min_card_{role}_active_ms"] = min(c["by_role"][role]["active_ms"] for c in cards)
                    row[f"min_card_{role}_compute_ms"] = min(c["by_role"][role]["compute_ms"] for c in cards)
                row["warm_mixed_100ms_card_count"] = sum(all(c["by_role"][role]["compute_ms"] >= 100 for role in ("short", "long")) for c in cards)
                row["global_source_population_and_placement_preserved"] = True
                row["per_npu_random_ordered_population_preserved"] = pair_valid.get(label, False)
                row["audit_pass"] = row["audit_pass"] and not any(e.get("scope") == "plan" for e in errors)
                row["strict_mixed_underload_valid"] = row["audit_pass"] and row["full_run_capacity_pass"] and row["all_npus_active"] and row["warm_mixed_card_count"] == 32
                row["strict_mixed_100ms_underload_valid"] = row["strict_mixed_underload_valid"] and row["warm_mixed_100ms_card_count"] == 32
                row["paired_study_conditions_met"] = row["strict_mixed_underload_valid"] and row["order_pair_pass"]
                row["mean_role_count_near_20_12"] = 18 <= row["mean_long_cards"] <= 22
                row["status"] = "complete" if row["audit_pass"] else "audit_failed"
            except Exception as e:
                row.update(status="audit_failed", audit_pass=False, error=f"{type(e).__name__}: {e}")
        elif len(paths) == 1:
            row["status"] = "running"  # Atomic result can precede final command update.
        elif command.get("pid") and command.get("status") not in ("complete", "failed", "timeout", "audit_failed"):
            row["status"] = "running"
        if row["status"] not in ("pending", "running", "complete"):
            errors.append({"scope": "run", "label": label, "strategy": strategy, "status": row["status"], "error": row.get("error")})
        rows.append(row)
    reused_controls = []
    for label, item in sorted(items.items()):
        if item["mode"] != "random" or item.get("strategies") != []:
            continue
        matches = [r for r in rows if r["mode"] == "random" and r["seed"] == item["seed"]
                   and items[r["label"]]["input_fingerprint"] == item["input_fingerprint"]
                   and r["label"] in payloads and label in payloads
                   and payloads[r["label"]][1] == payloads[label][1]]
        record = {"input_label": label, "spec_name": item["spec_name"], "seed": item["seed"],
            "input_fingerprint": item["input_fingerprint"], "status": "matched" if matches else "pending",
            "all_source_results_complete_and_audited": bool(matches) and all(r["status"] == "complete" and r["audit_pass"] for r in matches),
            "counts_as_new_simulation": False, "counts_as_independent_seed": False,
            "source_results": [{"label": r["label"], "spec_name": r["spec_name"],
                "strategy": r["strategy"], "status": r["status"], "audit_pass": r["audit_pass"],
                "result_path": r.get("result_path"), "result_sha256": r.get("result_sha256"),
                "device_utilization_percent": r.get("device_utilization_percent"),
                "warm_slo_percent": r.get("warm_slo_percent")} for r in matches]}
        reused_controls.append(record)
    complete = [r for r in rows if r["status"] == "complete"]
    groups = []
    for spec_name, mode, strategy in sorted({(r["spec_name"], r["mode"], r["strategy"]) for r in rows}):
        planned = [r for r in rows if (r["spec_name"], r["mode"], r["strategy"]) == (spec_name, mode, strategy)]
        completed = [r for r in planned if r["status"] == "complete"]
        ready = len(completed) == len(planned)
        group = {"spec_name": spec_name, "mode": mode, "strategy": strategy, "planned": len(planned), "completed": len(completed),
            "seeds": [r["seed"] for r in planned], "means_published": ready,
            "strict_mixed_underload_valid_count": sum(r["strict_mixed_underload_valid"] for r in completed)}
        if ready:
            for field in ("device_utilization_percent", "warm_slo_percent", "mean_long_cards", "short_pooled_U_percent", "long_pooled_U_percent"):
                values = [r[field] for r in completed]
                group[field] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None}
        groups.append(group)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "analyzer_sha256": sha(__file__),
        "planned": len(rows), "completed": len(complete), "all_complete": bool(rows) and len(complete) == len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)), "pending": [r for r in rows if r["status"] in ("pending", "running")],
        "all_completed_technical_audits_passed": not errors and all(r["audit_pass"] for r in complete),
        "errors": errors, "rows": rows, "groups": groups, "order_pairs": order_pairs,
        "reused_random_controls": reused_controls,
        "all_reused_controls_available": all(r["all_source_results_complete_and_audited"] for r in reused_controls),
        "all_order_pairs_passed": bool(order_pairs) and all(p["status"] == "passed" for p in order_pairs),
        "definition": "U uses actual clipped compute/[32*2000ms]. Warm SLO uses admissions in [2,4)s, completion-admission<=1.5*8C followed to completion. Role counts are admitted active requests, including their stalls; idle is retained. Nominal capacity is scanned over the entire run; extra next-request L0 is not added to current-profile V/C. Scientific failures remain in all rows and completed-group means."}
    write_json(HERE / "audit.json", {"created_utc": summary["created_utc"], "plans": plans, "inputs": list(input_audits.values()), "order_pairs": order_pairs, "reused_random_controls": reused_controls, "errors": errors})
    write_json(HERE / "results.json", summary)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / "per_seed.csv").open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["**新绑定下的同每卡人口随机／重排比较**", "", summary["definition"], "",
        f"完成 {len(complete)}/{len(rows)}；所有运行状态、科学条件失败均保留。原固定人口只改一次绑定，之后 random/ordered 每卡身份相同。", "",
        "|组|seed|顺序|策略|状态|设备U%|暖接纳SLO%|平均Long/Short|暖混合卡数|两角色各≥100ms卡数|全程盘峰GiB/s|每卡混合且欠载|",
        "|---|---:|---|---|---|---:|---:|---|---:|---:|---:|---|"]
    for r in rows:
        def f(k, digits=4):
            value = r.get(k); return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"
        lines.append(f"|{r['spec_name']}|{r['seed']}|{r['mode']}|{r['strategy']}|{r['status']}|{f('device_utilization_percent')}|{f('warm_slo_percent')}|{f('mean_long_cards',3)}/{f('mean_short_cards',3)}|{r.get('warm_mixed_card_count','—')}|{r.get('warm_mixed_100ms_card_count','—')}|{f('max_ssu_nominal_gib_s',6)}|{r.get('strict_mixed_underload_valid','—')}|")
    lines += ["", "暖窗混合要求每卡两角色均有正计算；输入含两类不等于实际暖窗混合。100ms阈值另列，不偷换正计算定义。平均20/12是描述目标，两策略速度不同会产生不同角色数量。名义V/C不是SSD吞吐或无突发保证。暖接纳SLO人口随顺序/策略变化，完整人口指标保留在analysis_cache。", ""]
    for control in reused_controls:
        sources = "; ".join(f"{r['label']}/{r['strategy']} ({r['status']})" for r in control["source_results"]) or "尚未找到同指纹计划结果"
        lines.append(f"随机对照复用：{control['input_label']} → {sources}。完整输入指纹相同且逐卡科学身份相同；不计为新增仿真或独立种子。")
    (HERE / "results.md").write_text("\n".join(lines))
    print(json.dumps({"planned": len(rows), "completed": len(complete), "statuses": summary["status_counts"], "errors": errors,
                      "order_pairs_passed": sum(p["status"] == "passed" for p in order_pairs)}, ensure_ascii=False))
    if errors or (args.require_complete and not (summary["all_complete"] and summary["all_order_pairs_passed"] and summary["all_reused_controls_available"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
