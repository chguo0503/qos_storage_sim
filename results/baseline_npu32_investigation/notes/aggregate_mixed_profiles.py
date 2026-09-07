#!/usr/bin/env python3
"""Read-only manifest audit and complete-ID profile reaggregation; no simulation."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/baseline_npu32_investigation"
WINDOWS = ((1000, 2000), (2000, 3000))


def read(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else path.open()) as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def csv_dump(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-9, abs_tol=2e-7), (a, b)


def overlap(a, b, start, end):
    return max(0.0, min(b, end) - max(a, start))


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def audit_manifest(path, short_only_control=False, required_short_category=None):
    x = read(path)
    m, requests = x["metadata"], x["requests"]
    npu_count, disk_count, layers = m["num_npu"], m["num_ssu"], m["n_layers"]
    assert len(requests) == m["request_count"] == len({r["request_id"] for r in requests})
    placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p)
                  for p in x["placements"]]
    # Reproduce the simulator's documented fingerprint serialization independently.
    digest = hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    for r in sorted(requests, key=lambda z: z["request_id"]):
        digest.update(repr((r["request_id"], r["npu_id"], float(r["arrival_time_ms"]),
                            r["load"]["category"], float(r["load"]["per_layer_us"]),
                            placements[r["placement_index"]])).encode())
    assert digest.hexdigest() == x["input_fingerprint"] == m["input_fingerprint"]
    profiles = [(p["seq_len_k"], p["nql"]) for p in m["profiles"]]
    authenticated_data = ast.literal_eval((ROOT / "data").read_text())
    for profile in m["profiles"]:
        if profile["construction"]["method"] == "direct_data_row":
            raw = authenticated_data[profile["seq_len_k"], profile["nql"]]
            for actual, expected in zip((profile["required_bandwidth_gibps"], profile["per_layer_compute_us"],
                                         profile["source_equivalent_ttft_78_layers_ms"], profile["per_layer_kv_gib"]), raw):
                close(actual, expected)
    profile_index = {key: i for i, key in enumerate(profiles)}
    assert len(profile_index) == len(profiles) >= (3 if short_only_control else 4)
    assert m["short_profile_count_per_npu"] >= 3
    if short_only_control:
        assert m["short_profile_count_per_npu"] == len(profiles)
    assert m["compute_scale_actual"] == 1
    lanes = defaultdict(list)
    lookup = {}
    counts, computation = Counter(), Counter()
    placement_work = []
    io_gib = 176 * 1024 / 2**30
    for placement in placements:
        per_disk = [0.0] * disk_count
        multiplier = layers if len(placement) == 1 else 1
        for layer in placement:
            for s, volume in layer:
                assert volume == io_gib
                per_disk[s] += multiplier * volume
        placement_work.append(per_disk)
    matrix, lane_rows = [], []
    for r in requests:
        load = r["load"]
        key = (load["seq_len_k"], load["nql"])
        p = m["profiles"][profile_index[key]]
        assert load["per_layer_us"] == p["per_layer_compute_us"]
        assert load["per_layer_kv_gb"] == p["per_layer_kv_gib"]
        assert r["request_id"] == load["request_id"]
        assert r["npu_id"] == load["npu_id"]
        assert r["arrival_time_ms"] == load["arrival_time"] == load["arrival_ms"] == 0
        assert load["padding_gib_per_layer"] == 0
        assert load["role"] == ("short" if profile_index[key] < m["short_profile_count_per_npu"] else "long")
        expected_category = ("S" if key[0] <= 80 else "L") + ("L" if key[1] >= 512 else "S")
        assert load["category"] == expected_category
        if required_short_category and load["role"] == "short":
            assert load["category"] == required_short_category
        actual_placement = placements[r["placement_index"]]
        expected = (tuple(((j + r["npu_id"] // 4) % disk_count, io_gib)
                          for j in range(p["ssd_prefix_tokens"] // 128)),)
        assert actual_placement == expected
        close(sum(placement_work[r["placement_index"]]), layers * p["per_layer_kv_gib"])
        lanes[r["npu_id"]].append(r)
        lookup[r["request_id"]] = {"profile": key, "role": load["role"],
                                   "sim_category": load["category"],
                                   "read_gib_by_ssu": placement_work[r["placement_index"]],
                                   "npu_id": r["npu_id"], "compute_ms": layers * load["per_layer_us"] / 1000}
        counts[key] += 1
        computation[key] += lookup[r["request_id"]]["compute_ms"]
    assert sorted(lanes) == list(range(npu_count))
    for n, rows in sorted(lanes.items()):
        rows.sort(key=lambda r: r["load"]["generation"])
        observed = [profile_index[lookup[r["request_id"]]["profile"]] for r in rows]
        assert [r["load"]["generation"] for r in rows] == list(range(len(rows)))
        assert [r["request_id"] for r in rows] == [n * 1_000_000 + g for g in range(len(rows))]
        expected = [i for i, q in enumerate(m["profile_quotas_per_unit"]) for _ in range(q)] * m["quota_units_per_npu"]
        random.Random(m["seed"] + n * 100003).shuffle(expected)
        assert observed == expected
        assert hashlib.sha256(json.dumps(observed).encode()).hexdigest() == m["per_npu_deck_sha256"][n]
        assert Counter(observed) == dict(enumerate(m["profile_counts_per_npu"]))
        compute_ms = sum(lookup[r["request_id"]]["compute_ms"] for r in rows)
        short_ms = sum(lookup[r["request_id"]]["compute_ms"] for r in rows if r["load"]["role"] == "short")
        work = [sum(placement_work[r["placement_index"]][s] for r in rows) for s in range(disk_count)]
        matrix.append([1000 * w / compute_ms for w in work])
        close(short_ms / compute_ms, m["short_compute_fraction_per_npu"])
        close(compute_ms, m["input_demand"]["per_npu_ideal_compute_ms"][n])
        assert compute_ms >= m["required_minimum_compute_ms_per_npu"] - 1e-7
        lane_rows.append({"npu_id": n, "profiles": len(set(observed)), "short_profiles": len({lookup[r["request_id"]]["profile"] for r in rows if r["load"]["role"] == "short"}),
                          "requests": len(rows), "ideal_compute_ms": compute_ms, "short_compute_share": short_ms / compute_ms,
                          "shuffle_reproduces": True, "deck_sha256": m["per_npu_deck_sha256"][n]})
    disk_load = [sum(row[s] for row in matrix) for s in range(disk_count)]
    rho = max(disk_load) / 40
    link_rho = max(map(sum, matrix)) / 50
    close(rho, m["input_demand"]["hottest_ssu_load_ratio"])
    close(link_rho, m["input_demand"]["largest_npu_receive_load_ratio"])
    for a, b in zip(matrix, m["input_demand"]["matrix_gib_s"]):
        for v, w in zip(a, b):
            close(v, w)
    total_compute = sum(computation.values())
    categories = {row["profile"]: row["sim_category"] for row in lookup.values()}
    profile_rows = [{"seq_len_k": key[0], "nql": key[1], "role": "short" if i < m["short_profile_count_per_npu"] else "long", "sim_category": categories[key],
                     "requests": counts[key], "request_share": counts[key] / len(requests),
                     "compute_ms": computation[key], "compute_share": computation[key] / total_compute,
                     "per_request_compute_ms": layers * m["profiles"][i]["per_layer_compute_us"] / 1000,
                     "per_request_read_gib": layers * m["profiles"][i]["per_layer_kv_gib"],
                     "construction": m["profiles"][i]["construction"]["method"]} for i, key in enumerate(profiles)]
    audit = {"label": m["label"], "family": m["family"], "seed": m["seed"], "manifest": str(path), "manifest_sha256": sha(path),
             "input_fingerprint": x["input_fingerprint"], "request_count": len(requests), "profile_count_per_npu": len(profiles),
             "short_profile_count_per_npu": m["short_profile_count_per_npu"], "short_compute_share": m["short_compute_fraction_per_npu"],
             "short_request_share": sum(r["requests"] for r in profile_rows if r["role"] == "short") / len(requests),
             "hottest_ssu_rho": rho, "largest_link_rho": link_rho, "mean_capacity_feasible": max(rho, link_rho) <= 1 + 1e-9,
             "minimum_ideal_compute_ms_per_npu": min(r["ideal_compute_ms"] for r in lane_rows),
             "per_ssu_gib_s": disk_load, "unique_full_lane_orders": len(set(m["per_npu_deck_sha256"])),
             "required_short_category": required_short_category,
             "lane_audit": lane_rows, "profiles": profile_rows, "independent_audit_passed": True}
    return m, lookup, audit


def cohort(rows):
    total_c = sum(r["own_compute_ms"] for r in rows)
    active = sum(r["processing_latency_ms"] for r in rows)
    admission = sum(r["completion_time_ms"] - r["admission_time_ms"] <= 1.5 * r["own_compute_ms"] + 1e-9 for r in rows)
    arrival = sum(r["completion_time_ms"] - r["arrival_time_ms"] <= 1.5 * r["own_compute_ms"] + 1e-9 for r in rows)
    fractions = [r["own_compute_ms"] / r["processing_latency_ms"] for r in rows]
    total_stall = sum(r["io_stall_ms"] for r in rows)
    warm_stall = sum(r["warm_layers_stall_ms"] for r in rows)
    return {"request_count": len(rows), "compute_ms": total_c, "active_ms": active,
            "aggregate_compute_fraction": total_c / active, "mean_request_compute_fraction": sum(fractions) / len(rows),
            "min_request_compute_fraction": min(fractions), "p05_request_compute_fraction": percentile(fractions, .05),
            "median_request_compute_fraction": percentile(fractions, .5),
            "mean_io_stall_ms": sum(r["io_stall_ms"] for r in rows) / len(rows),
            "mean_layer0_stall_ms": sum(r["layer0_stall_ms"] for r in rows) / len(rows),
            "mean_warm_layers_stall_ms": warm_stall / len(rows),
            "warm_layers_share_of_total_stall": warm_stall / total_stall if total_stall else 0.0,
            "p95_io_stall_ms": percentile([r["io_stall_ms"] for r in rows], .95),
            "admission_slo_passed": admission, "admission_slo_pass_rate": admission / len(rows),
            "arrival_slo_passed": arrival, "arrival_slo_pass_rate": arrival / len(rows),
            "cohort_id_sha256": hashlib.sha256(json.dumps(sorted(r["request_id"] for r in rows)).encode()).hexdigest()}


def aggregate_result(path, strategy, m, lookup, assignment="fixed", variant=None):
    x = read(path)
    assert x["input_fingerprint"] == m["input_fingerprint"]
    assert x["strategy"] == strategy
    assert x["collector_interval_ms"] == 5
    assert x["stress_runner_sha256"] == sha(ROOT / "run_baseline_npu32_stress.py")
    for name, expected_sha in x["core_and_policy_sha256"].items():
        assert sha(ROOT / name) == expected_sha, ("Result/source mismatch", path, name)
    summary = x["summary"]
    assert all(summary["invariants"].values())
    assert summary["batch_size"] == 1
    assert {r["request_id"] for r in summary["request_metrics"]} == set(lookup)
    assert len(summary["request_metrics"]) == len(lookup)
    assert x["input_placement_fingerprint"] == x["execution_placement_fingerprint"]
    if strategy in ("strategy1", "strategy2", "strategy3"):
        assert x.get("policy_config", {}).get("assignment", "fixed") == assignment
    else:
        # Baseline/Once/New Once ignore the CLI assignment field in this runner.
        assert assignment == "fixed"
    assigned_npu = {rid: info["npu_id"] for rid, info in lookup.items()}
    if x["assignment_log"]:
        assert len(x["assignment_log"]) == len(lookup)
        assert {a["request_id"] for a in x["assignment_log"]} == set(lookup)
        for record in x["assignment_log"]:
            rid = record["request_id"]
            assert record["original_npu_id"] == lookup[rid]["npu_id"]
            assert record["score_model"] == assignment
            assert record["arrival_time_ms"] == 0
            assert 0 <= record["assigned_npu_id"] < m["num_npu"]
            if assignment == "fixed":
                assert record["assigned_npu_id"] == record["original_npu_id"]
            assigned_npu[rid] = record["assigned_npu_id"]
    else:
        assert assignment == "fixed"
    assigned_compute = [0.] * m["num_npu"]
    assigned_work = [[0.] * m["num_ssu"] for _ in range(m["num_npu"])]
    for rid, info in lookup.items():
        n = assigned_npu[rid]
        assigned_compute[n] += info["compute_ms"]
        for s, volume in enumerate(info["read_gib_by_ssu"]):
            assigned_work[n][s] += volume
    assigned_matrix = [[1000*v/c if c else 0. for v in row] for row,c in zip(assigned_work,assigned_compute)]
    assigned_ssu = [sum(row[s] for row in assigned_matrix) for s in range(m["num_ssu"])]
    close(summary["expected_read_gb"], m["input_demand"]["total_read_gib"])
    close(summary["completed_read_gb"], m["input_demand"]["total_read_gib"])
    grouped = defaultdict(list)
    batches = {b["member_request_ids"][0]: b for b in summary["microbatch_metrics"]}
    base = {"label": m["label"], "family": m["family"], "seed": m["seed"], "strategy": variant or strategy,
            "policy_strategy": strategy, "assignment": assignment, "input_fingerprint": m["input_fingerprint"]}
    for r in summary["request_metrics"]:
        expected = lookup[r["request_id"]]
        assert r["npu_id"] == assigned_npu[r["request_id"]]
        close(r["own_compute_ms"], expected["compute_ms"])
        close(r["processing_latency_ms"], r["completion_time_ms"] - r["admission_time_ms"])
        close(r["processing_latency_ms"], r["own_compute_ms"] + r["io_stall_ms"])
        layer_rows = batches[r["request_id"]]["layer_metrics"]
        r["layer0_stall_ms"] = layer_rows[0]["io_barrier_wait_ms"]
        r["warm_layers_stall_ms"] = sum(layer["io_barrier_wait_ms"] for layer in layer_rows[1:])
        close(r["layer0_stall_ms"] + r["warm_layers_stall_ms"], r["io_stall_ms"])
        grouped[expected["profile"]].append(r)
    full = cohort(summary["request_metrics"])
    short = cohort([r for r in summary["request_metrics"] if lookup[r["request_id"]]["role"] == "short"])
    profile_rows = []
    for key, rows in sorted(grouped.items()):
        profile_rows.append({**base, "seq_len_k": key[0], "nql": key[1], "role": lookup[rows[0]["request_id"]]["role"],
                             "sim_category": lookup[rows[0]["request_id"]]["sim_category"],
                             "compute_share_of_full_input": sum(r["own_compute_ms"] for r in rows) / full["compute_ms"], **cohort(rows)})
    window_rows, window_profiles, card_rows = [], [], []
    for start, end in WINDOWS:
        active, busy = [0.] * m["num_npu"], [0.] * m["num_npu"]
        weights = defaultdict(lambda: [0., 0.])
        for batch in summary["microbatch_metrics"]:
            assert len(batch["member_request_ids"]) == 1
            expected = lookup[batch["member_request_ids"][0]]
            n = batch["npu_id"]
            assert n == assigned_npu[batch["member_request_ids"][0]]
            a = overlap(batch["admission_time_ms"], batch["completion_time_ms"], start, end)
            c = sum(overlap(layer["compute_start_ms"], layer["compute_end_ms"], start, end) for layer in batch["layer_metrics"])
            active[n] += a
            busy[n] += c
            weights[expected["profile"]][0] += c
            weights[expected["profile"]][1] += a
        length = end - start
        all_active = all(abs(a - length) <= 1e-6 for a in active)
        recorded = next(w for w in x["windows"] if w["start_ms"] == start and w["end_ms"] == end)
        close(sum(busy) / (m["num_npu"] * length), recorded["mean_npu_utilization"])
        assert recorded["all_npus_active_whole_window"] == all_active
        for n, (a, c) in enumerate(zip(active, busy)):
            assert c <= a + 1e-7 <= length + 2e-7
            card_rows.append({**base, "window_start_ms": start, "window_end_ms": end, "npu_id": n,
                              "utilization": c / length, "active_ms": a, "stall_ms": a-c, "idle_ms": length-a})
        window_rows.append({**base, "window_start_ms": start, "window_end_ms": end,
                            "utilization": sum(busy) / (m["num_npu"] * length),
                            "all_active": all_active, "fully_active_npu_count": sum(abs(a-length) <= 1e-6 for a in active),
                            "minimum_npu_utilization": min(busy) / length, "maximum_npu_utilization": max(busy) / length,
                            "fleet_idle_ms": m["num_npu"] * length-sum(active)})
        for key, (c, a) in sorted(weights.items()):
            window_profiles.append({**base, "window_start_ms": start, "window_end_ms": end,
                                    "seq_len_k": key[0], "nql": key[1], "compute_ms": c, "active_ms": a,
                                    "fleet_time_compute_share": c / (m["num_npu"]*length),
                                    "fleet_time_active_share": a / (m["num_npu"]*length),
                                    "profile_active_compute_fraction": c/a if a else None, "all_active": all_active})
    provenance = {**base, "status": "complete", "path": str(path), "sha256": sha(path),
                  "every_core_and_policy_hash_matches_current_frozen_source": True,
                  "core_and_policy_sha256": x["core_and_policy_sha256"], "stress_runner_sha256": x["stress_runner_sha256"],
                  "submit_seed": x["submit_seed"], "makespan_ms": summary["makespan_ms"],
                  "reassigned_request_count": sum(assigned_npu[rid] != info["npu_id"] for rid, info in lookup.items()),
                  "assigned_ideal_compute_ms_by_npu": assigned_compute,
                  "assigned_mean_hottest_ssu_rho": max(assigned_ssu) / 40,
                  "assigned_mean_largest_link_rho": max(map(sum,assigned_matrix)) / 50,
                  "execution_input_fingerprint": summary["input_fingerprint"],
                  "policy_config": x["policy_config"],
                  "input_and_execution_placement_fingerprint": x["input_placement_fingerprint"],
                  "fleet_full_run_utilization": summary["fleet_npu_compute_utilization"],
                  "short": short, "all_requests": full}
    close(full["compute_ms"] / (m["num_npu"] * summary["makespan_ms"]), summary["fleet_npu_compute_utilization"])
    return provenance, profile_rows, window_rows, window_profiles, card_rows


def plot(output, profiles, windows):
    if not profiles:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline": "#6b7280", "once": "#2563eb", "new_once": "#059669"}
    labels = sorted({r["label"] for r in profiles})
    for label in labels:
        pr = [r for r in profiles if r["label"] == label]
        short_keys = sorted({(r["seq_len_k"],r["nql"]) for r in pr if r["role"] == "short"})
        strategies = [s for s in colors if any(r["strategy"] == s for r in pr)]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
        width = .8 / len(strategies)
        for i, strategy in enumerate(strategies):
            offsets = [j - .4 + width*(i+.5) for j in range(len(short_keys))]
            rows = [next(r for r in pr if r["strategy"] == strategy and (r["seq_len_k"],r["nql"]) == key) for key in short_keys]
            for axis, metric, factor in ((axes[0], "aggregate_compute_fraction", 100), (axes[1], "admission_slo_pass_rate", 100), (axes[2], "mean_io_stall_ms", 1)):
                axis.bar(offsets, [r[metric]*factor for r in rows], width, color=colors[strategy], label=strategy)
        for axis in axes:
            axis.set_xticks(range(len(short_keys)), [f"{s}K/{q}" for s,q in short_keys])
            axis.set_xlabel("Short profile (sequence / NQL)")
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Complete-cohort compute / active (%)")
        axes[1].set_ylabel("Admission SLO passed (%)")
        axes[2].set_ylabel("Mean request IO stall (ms)")
        axes[0].set_ylim(0, 103); axes[1].set_ylim(0, 103)
        axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.24), ncol=3)
        seen = {r["strategy"] for r in pr}
        pending = [s for s in colors if s not in seen]
        fig.suptitle(label + (" | pending: " + ", ".join(pending) if pending else "") + "\nSame complete request-ID cohorts; synthetic t=0 backlog", fontsize=10)
        fig.savefig(output / f"{label}_short_profiles.png", dpi=180)
        fig.savefig(output / f"{label}_short_profiles.svg")
        fig.savefig(output / f"{label}_short_profiles.pdf")
        plt.close(fig)
        # A companion chart retains the long-profile cost of protecting shorts.
        keys = sorted({(r["seq_len_k"], r["nql"]) for r in pr})
        fig, axis = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
        for i, strategy in enumerate(strategies):
            rows = [next(r for r in pr if r["strategy"] == strategy and (r["seq_len_k"],r["nql"]) == key) for key in keys]
            axis.bar([j-.4+width*(i+.5) for j in range(len(keys))],
                     [100*r["aggregate_compute_fraction"] for r in rows],width,
                     color=colors[strategy],label=strategy)
        tick_labels = []
        for key in keys:
            row = next(r for r in pr if (r["seq_len_k"],r["nql"]) == key)
            tick_labels.append(f"{key[0]}K/{key[1]}\n{row['role']}; {100*row['compute_share_of_full_input']:.1f}% of C")
        axis.set_xticks(range(len(keys)),tick_labels)
        axis.set_ylim(0,103)
        axis.set_ylabel("Complete-cohort compute / active (%)")
        axis.grid(axis="y",alpha=.2); axis.set_axisbelow(True)
        axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.19), ncol=3)
        axis.set_title(label + (" | pending: " + ", ".join(pending) if pending else "") +
                       "\nEvery profile retained, including long-profile tradeoffs", fontsize=10)
        fig.savefig(output / f"{label}_all_profiles_tradeoff.png",dpi=180)
        fig.savefig(output / f"{label}_all_profiles_tradeoff.svg")
        fig.savefig(output / f"{label}_all_profiles_tradeoff.pdf")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=BASE / "mixed_varied")
    p.add_argument("--output", type=Path)
    p.add_argument("--short-only-control", action="store_true", help="Separate all-short control; permit 3 profiles, require every profile short")
    p.add_argument("--require-short-category", choices=("SS","SL","LS","LL"), help="Optional semantic control: require all short-group profiles to have this actual sim category")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    args.output = args.output or BASE / ("notes/mixed_control_audit" if args.short_only_control else "notes/mixed_profile_audit")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read(args.study / "plan.json")
    audit, results, profiles, windows, weighted, cards = [], [], [], [], [], []
    for item in plan["inputs"]:
        manifest_path = args.study / "inputs" / Path(item["manifest"]).name
        m, lookup, audited = audit_manifest(manifest_path, args.short_only_control, args.require_short_category)
        assert m["input_fingerprint"] == item["input_fingerprint"]
        audit.append(audited)
        codes = None
        for strategy in plan["strategies"]:
            directories = [args.study / "runs" / item["label"] / strategy]
            directories += [args.study / host / "runs" / item["label"] / strategy
                            for host in ("local", "remote", "extra_local", "extra_remote")]
            paths = sorted(path for directory in directories for path in directory.glob("*.json.gz"))
            if not paths:
                commands = [read(directory / "command.json") for directory in directories
                            if (directory / "command.json").exists()]
                results.append({"label": item["label"], "input_fingerprint": item["input_fingerprint"], "strategy": strategy,
                                "status": "failed" if any(c.get("returncode", 0) for c in commands) else "pending",
                                "note": "No complete result locally; may be queued/running/not synced."})
                continue
            assert len(paths) == 1, (item["label"], strategy, paths)
            result, pr, wr, wp, cr = aggregate_result(paths[0], strategy, m, lookup)
            signature = (result["core_and_policy_sha256"], result["stress_runner_sha256"], result["submit_seed"])
            if codes is not None:
                assert codes == signature, "Do not pair different code or submission seeds"
            codes = signature
            results.append(result); profiles.extend(pr); windows.extend(wr); weighted.extend(wp); cards.extend(cr)
    paired = []
    index = {(r["input_fingerprint"], r["strategy"],r["seq_len_k"],r["nql"]): r for r in profiles}
    for r in profiles:
        if r["strategy"] == "baseline":
            continue
        baseline = index.get((r["input_fingerprint"],"baseline",r["seq_len_k"],r["nql"]))
        if baseline:
            assert r["cohort_id_sha256"] == baseline["cohort_id_sha256"]
            paired.append({k:r[k] for k in ("label","input_fingerprint","strategy","seq_len_k","nql","role","request_count")} |
                          {"delta_compute_fraction_pp": 100*(r["aggregate_compute_fraction"]-baseline["aggregate_compute_fraction"]),
                           "delta_admission_slo_pp":100*(r["admission_slo_pass_rate"]-baseline["admission_slo_pass_rate"]),
                           "delta_mean_io_stall_ms": r["mean_io_stall_ms"]-baseline["mean_io_stall_ms"]})
    fleet_pairs = []
    result_index = {(r["input_fingerprint"],r["strategy"]):r for r in results if r["status"] == "complete"}
    window_index = {(w["input_fingerprint"],w["strategy"],w["window_start_ms"]):w for w in windows}
    for r in results:
        if r["status"] != "complete" or r["strategy"] == "baseline":
            continue
        baseline = result_index.get((r["input_fingerprint"],"baseline"))
        if baseline is None:
            continue
        ds = [100*(window_index[r["input_fingerprint"],r["strategy"],s]["utilization"] -
                   window_index[r["input_fingerprint"],"baseline",s]["utilization"]) for s in (1000,2000)]
        fleet_pairs.append({"label":r["label"],"strategy":r["strategy"],"input_fingerprint":r["input_fingerprint"],
                            "delta_U1_pp":ds[0],"delta_U2_pp":ds[1],
                            "delta_makespan_percent":100*(r["makespan_ms"]/baseline["makespan_ms"]-1),
                            "delta_short_cohort_U_pp":100*(r["short"]["aggregate_compute_fraction"]-baseline["short"]["aggregate_compute_fraction"]),
                            "delta_short_admission_SLO_pp":100*(r["short"]["admission_slo_pass_rate"]-baseline["short"]["admission_slo_pass_rate"])})
    for name, rows in (("inputs", [{k:v for k,v in a.items() if k not in ("profiles","lane_audit","per_ssu_gib_s")} for a in audit]),
                       ("input_profiles",[{"label":a["label"],"input_fingerprint":a["input_fingerprint"],**r} for a in audit for r in a["profiles"]]),
                       ("profiles",profiles),("profile_pairs",paired),("fleet_pairs",fleet_pairs),("windows",windows),("window_profiles",weighted),("cards",cards),
                       ("status",[{k:v for k,v in r.items() if k not in ("short","all_requests","core_and_policy_sha256")} for r in results])):
        csv_dump(args.output/f"{name}.csv", rows)
    artifact = {"created_utc":datetime.now(timezone.utc).isoformat(), "source_plan": str(args.study/"plan.json"),
                "study_kind":"separate_all_short_control" if args.short_only_control else "main_multi_short_and_long_profiles",
                "source_plan_sha256":sha(args.study/"plan.json"),"aggregation_script_sha256":sha(Path(__file__)),
                "method":"Independent read-only frozen manifest and complete same-ID cohorts; all results, including losses, retained.",
                "inputs":audit,"results":results,"profiles":profiles,"profile_pairs":paired,"fleet_pairs":fleet_pairs,"windows":windows,
                "window_profiles":weighted,"cards":cards}
    dump(args.output/"audit.json",artifact)
    lines=["# 持续混合多短画像独立重聚合", "", f"计划：`{args.study}`；生成 UTC：{artifact['created_utc']}。", "",
           "所有画像统计使用完整同 request-ID cohort；U 是 compute/active；SLO 是 admission 后 1.5×自身计算时间。图与表包含尚缺策略的 pending 标记。", "",
           "|输入|策略|状态|1–2s U|2–3s U|两窗 all-active|full makespan ms|短类 U|短类 SLO|", "|---|---|---|---:|---:|---|---:|---:|---:|"]
    for r in results:
        if r["status"] != "complete":
            lines.append(f"|{r['label']}|{r['strategy']}|{r['status']}|—|—|—|—|—|—|")
            continue
        ww=sorted([w for w in windows if w["input_fingerprint"]==r["input_fingerprint"] and w["strategy"]==r["strategy"]],key=lambda w:w["window_start_ms"])
        lines.append(f"|{r['label']}|{r['strategy']}|complete|{100*ww[0]['utilization']:.3f}%|{100*ww[1]['utilization']:.3f}%|{all(w['all_active'] for w in ww)}|{r['makespan_ms']:.3f}|{100*r['short']['aggregate_compute_fraction']:.3f}%|{r['short']['admission_slo_passed']}/{r['short']['request_count']}|")
    lines += ["", "## 相对baseline的窗口或完整完工负例", "", "正Δmakespan表示变慢；全部配对（含正例）在fleet_pairs.csv。", "",
              "|输入|策略|ΔU1 pp|ΔU2 pp|Δmakespan %|", "|---|---|---:|---:|---:|"]
    for r in fleet_pairs:
        if r["delta_U1_pp"] < -1e-9 or r["delta_U2_pp"] < -1e-9 or r["delta_makespan_percent"] > 1e-9:
            lines.append(f"|{r['label']}|{r['strategy']}|{r['delta_U1_pp']:+.3f}|{r['delta_U2_pp']:+.3f}|{r['delta_makespan_percent']:+.3f}|")
    lines += ["", "## 逐短画像的同 cohort 指标", "", "|输入|策略|短画像 seqK/NQL|完整请求数|输入 compute 占比|compute/active|admission SLO|平均 stall ms|", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in profiles:
        if r["role"]=="short":
            lines.append(f"|{r['label']}|{r['strategy']}|{r['seq_len_k']}/{r['nql']}|{r['request_count']}|{100*r['compute_share_of_full_input']:.3f}%|{100*r['aggregate_compute_fraction']:.3f}%|{r['admission_slo_passed']}/{r['request_count']}|{r['mean_io_stall_ms']:.3f}|")
    (args.output/"summary.md").write_text("\n".join(lines)+"\n")
    if not args.no_plots:
        plot(args.output,profiles,windows)
    print(json.dumps({"audited_inputs":len(audit),"complete":sum(r["status"]=="complete" for r in results),
                      "pending":sum(r["status"]=="pending" for r in results),"profile_rows":len(profiles),"output":str(args.output)}))


if __name__ == "__main__":
    main()
