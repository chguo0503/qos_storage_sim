#!/usr/bin/env python3
"""Independent audit of the pure-role NPU allocation study; no simulations.

Writes only audit/. The source population is each seed's original 19456-request
random mixed manifest. New NPU bindings are permitted; original SSD placements,
scientific request fields, and the complete original identity population are not.
The warm window requires every NPU active, not every NPU mixed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(BASE)]
START, END = 2000.0, 4000.0
SEEDS = (7, 19, 43, 67, 101)
STRATEGIES = ("baseline", "once")
EXPECTED_PROFILES = {(1, 128): 6400, (1, 256): 6400, (1, 384): 6400, (192, 768): 256}
MUTABLE_LOAD_FIELDS = {"request_id", "npu_id", "generation", "original_request_id", "source_original_npu_id"}


def read(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic_hash(request, placements):
    return digest({"arrival_time_ms": request["arrival_time_ms"],
                   "load": {k: v for k, v in request["load"].items() if k not in MUTABLE_LOAD_FIELDS},
                   "placement": placements[request["placement_index"]]})


@lru_cache(maxsize=5)
def source_population(seed):
    path = BASE / "inputs" / f"concurrency_l768_seed{seed}.json.gz"
    data = read(path)
    rows, maxima = {}, {role: [0.0]*6 for role in ("short", "long")}
    for r in data["requests"]:
        rid, load = r["request_id"], r["load"]
        require(rid not in rows, "Duplicate original source identity")
        rows[rid] = {"npu": r["npu_id"], "semantic_sha256": semantic_hash(r, data["placements"])}
        for layer in data["placements"][r["placement_index"]]:
            for disk in range(6):
                rate = math.fsum(v for s, v in layer if s == disk)/(load["per_layer_us"]/1e6)
                maxima[load["role"]][disk] = max(maxima[load["role"]][disk], rate)
    require(len(rows) == 19456, "Source population is not 19456 requests")
    return {"path": str(path), "sha256": sha(path), "input_fingerprint": data["input_fingerprint"],
            "rows": rows, "role_pool_max_per_ssu_gib_s": maxima}


def order_name(metadata, label):
    values = [metadata.get(k) for k in ("short_order", "short_order_mode", "order_mode", "order")]
    for value in values:
        if value in ("random", "round_robin"):
            return value
    if "round_robin" in label:
        return "round_robin"
    if "random" in label:
        return "random"
    raise ValueError(f"Cannot identify declared short order: {label}, {values}")


def audit_input(path, auditor):
    data = read(path)
    meta = data["metadata"]
    seed, label = meta["seed"], meta["label"]
    require(seed in SEEDS, "Unplanned seed")
    require((meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (32, 6, 8), "Wrong topology/layers")
    require(len(data["requests"]) == 19456, "Changed global request count")
    source = source_population(seed)
    require(meta.get("source_manifest_sha256") == source["sha256"], "Wrong recorded source manifest hash")
    require(meta.get("source_input_fingerprint") == source["input_fingerprint"], "Wrong recorded source input fingerprint")
    require(meta.get("source_request_count") == 19456, "Wrong recorded source population count")
    require(all(sha(ROOT/name) == expected for name, expected in meta.get("construction_source_sha256", {}).items()),
            "Recorded construction source hash changed")
    order = order_name(meta, label)
    seen, new_seen = set(), set()
    lanes = defaultdict(list)
    profiles = Counter()
    for r in data["requests"]:
        load = r["load"]
        oid = load["original_request_id"]
        require(oid not in seen and oid in source["rows"], "Original identity missing, duplicated, or foreign")
        seen.add(oid)
        rid, npu = r["request_id"], r["npu_id"]
        require(rid not in new_seen, "New position ID duplicated")
        new_seen.add(rid)
        require(0 <= npu < 32 and load["npu_id"] == npu, "New NPU fields inconsistent")
        require(load["request_id"] == rid, "Top-level and load request IDs differ")
        require(load["source_original_npu_id"] == source["rows"][oid]["npu"], "Original NPU provenance incorrect")
        require(semantic_hash(r, data["placements"]) == source["rows"][oid]["semantic_sha256"],
                f"Scientific request fields or original SSD placement changed: {oid}")
        require(r["arrival_time_ms"] == 0, "Arrival changed")
        lanes[npu].append(r)
        profiles[(load["seq_len_k"], load["nql"])] += 1
    require(seen == source["rows"].keys(), "Original global population is not a bijection")
    require(profiles == EXPECTED_PROFILES, "Global profile population changed")
    require(set(lanes) == set(range(32)), "An NPU has no input")
    per_npu, roles, assignment, long_sequence = [], [], [], []
    for npu in range(32):
        lane = sorted(lanes[npu], key=lambda r: r["request_id"])
        require(all(r["request_id"] == npu*1_000_000+i and r["load"]["generation"] == i
                    for i, r in enumerate(lane)), "Position ID/generation does not match queue position")
        role_set = {r["load"]["role"] for r in lane}
        require(len(role_set) == 1, "A role-separated NPU has mixed roles")
        role = next(iter(role_set))
        require(role in ("short", "long"), "Unknown role")
        roles.append(role)
        counts = Counter((r["load"]["seq_len_k"], r["load"]["nql"]) for r in lane)
        pure_c = math.fsum(8*r["load"]["per_layer_us"]/1000 for r in lane)
        require(pure_c > END, "Pure compute does not cover the entire 4-second prefix")
        if role == "short":
            require(set(counts) == {(1, 128), (1, 256), (1, 384)}, "Short card lacks a short profile")
            if order == "round_robin":
                expected = [key for i in range(max(counts.values())) for key in sorted(counts) if i < counts[key]]
                require([(r["load"]["seq_len_k"], r["load"]["nql"]) for r in lane] == expected,
                        "Short order is not exact per-profile round robin")
        else:
            require(set(counts) == {(192, 768)}, "Long card has wrong profile")
            long_sequence.append([npu, [r["load"]["original_request_id"] for r in lane]])
        original_ids = sorted(r["load"]["original_request_id"] for r in lane)
        assignment.append([npu, original_ids])
        per_npu.append({"npu_id": npu, "role": role, "request_count": len(lane),
                        "ideal_compute_ms": pure_c,
                        "profile_counts": {f"{key[0]}:{key[1]}": value for key, value in sorted(counts.items())},
                        "assigned_original_population_sha256": digest(original_ids)})
    long_count = roles.count("long")
    require(long_count in (11, 6), "Unplanned long/short card ratio")
    require(roles == ["long"]*long_count + ["short"]*(32-long_count), "Role card ranges changed")
    require(meta.get("assigned_role_by_npu") == roles, "Recorded assigned roles disagree with actual requests")
    recorded_lanes = meta.get("per_npu_assignment", [])
    require(len(recorded_lanes) == 32, "Missing recorded per-NPU assignment")
    for actual, recorded in zip(per_npu, recorded_lanes):
        require((actual["npu_id"], actual["role"], actual["request_count"]) ==
                (recorded["npu_id"], recorded["assigned_role"], recorded["request_count"]),
                "Recorded role or lane count differs")
        require(math.isclose(actual["ideal_compute_ms"], recorded["ideal_compute_ms"], rel_tol=0, abs_tol=1e-8),
                "Recorded per-NPU pure compute differs")
        require(actual["assigned_original_population_sha256"] == recorded["original_identity_multiset_sha256"],
                "Recorded original per-NPU population hash differs")
        require([actual["profile_counts"].get(f"{key[0]}:{key[1]}", 0) for key in EXPECTED_PROFILES]
                == recorded["profile_counts"], "Recorded per-NPU profile counts differ")
    for key in EXPECTED_PROFILES:
        group = [r["profile_counts"].get(f"{key[0]}:{key[1]}", 0) for r in per_npu
                 if r["role"] == ("long" if key == (192, 768) else "short")]
        require(max(group)-min(group) <= 1, "A profile is not evenly distributed within its role cards")

    # This loader independently verifies the simulator's input fingerprint and
    # computes exact max-per-declared-NPU physical SSD vectors.
    info = auditor.load_input(path, HERE)
    exact = info["static_max_proof"]
    broad = [long_count*source["role_pool_max_per_ssu_gib_s"]["long"][s]
             + (32-long_count)*source["role_pool_max_per_ssu_gib_s"]["short"][s] for s in range(6)]
    require(max(exact["per_ssu_gib_s"]) < 40 and max(broad) < 40, "Static disk certificate fails")
    require(exact["max_npu_link_gib_s"] < 50, "Static receive-link certificate fails")
    require(all(a <= b+1e-10 for a, b in zip(exact["per_ssu_gib_s"], broad)), "Exact bound exceeds role-only envelope")
    recorded_proof = meta["active_profile_rate_certificate"]
    require(recorded_proof["passes"] and all(math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
            for a, b in zip(exact["per_ssu_gib_s"], recorded_proof["per_ssu_upper_bound_gib_s"])),
            "Recorded static capacity certificate differs")
    return {"label": label, "seed": seed, "long_npus": long_count, "short_npus": 32-long_count,
            "order": order, "manifest_path": str(path), "manifest_sha256": sha(path),
            "input_fingerprint": info["input_fingerprint"], "status": "passed", "request_count": len(seen),
            "source_manifest_path": source["path"], "source_manifest_sha256": source["sha256"],
            "source_input_fingerprint": source["input_fingerprint"],
            "original_population_bijection": True, "scientific_fields_and_placement_exact": True,
            "assignment_population_sha256": digest(assignment), "long_identity_order_sha256": digest(long_sequence),
            "min_per_npu_ideal_compute_ms": min(x["ideal_compute_ms"] for x in per_npu),
            "max_per_npu_ideal_compute_ms": max(x["ideal_compute_ms"] for x in per_npu),
            "total_ideal_compute_ms": math.fsum(x["ideal_compute_ms"] for x in per_npu),
            "full_run_device_U_upper_even_without_IO_wait": math.fsum(x["ideal_compute_ms"] for x in per_npu)
                / (32*max(x["ideal_compute_ms"] for x in per_npu)),
            "per_npu": per_npu, "exact_assigned_npu_static_per_ssu_gib_s": exact["per_ssu_gib_s"],
            "role_count_only_static_per_ssu_gib_s": broad,
            "static_max_npu_link_gib_s": exact["max_npu_link_gib_s"]}


def audit_result(path, item, strategy, plan, auditor, paired):
    from run_coflow_experiments import source_files
    info = auditor.load_input(Path(item["manifest_path"]), HERE)
    raw = read(path)
    parent = auditor.analyze_result(path, info, HERE)
    metrics, metric_checks = paired.independent_metrics(raw, info, auditor)
    checks = dict(parent["audit"]["checks"])
    checks["runtime_uses_declared_new_npu_binding"] = checks.pop("original_npu_assignment")
    checks.update(metric_checks)
    checks.update(paired.policy_checks(raw, info))
    command = read(path.parent/"command.json")
    argv = command.get("command", [])
    options = {argv[i]: argv[i+1] for i in range(len(argv)-1) if argv[i].startswith("--")}
    checks["result_matches_planned_strategy_and_seed"] = raw.get("strategy") == strategy and raw.get("submit_seed") == item["seed"]
    checks["completed_matching_command"] = (
        command.get("status") == "complete" and command.get("returncode") == 0
        and command.get("strategy") == strategy and command.get("seed") == item["seed"]
        and command.get("label") == item["label"] and command.get("input_fingerprint") == item["input_fingerprint"])
    checks["command_matches_frozen_manifest_window_policy"] = (
        Path(options.get("--manifest", "")).resolve() == Path(item["manifest_path"]).resolve()
        and options.get("--strategy") == strategy and options.get("--assignment") == "fixed"
        and options.get("--window") == "2000:4000")
    checks["command_matches_all_frozen_source_hashes"] = command.get("source_sha256") == plan["source_sha256"]
    checks["complete_expected_core_hash_set"] = set(raw.get("core_and_policy_sha256", {})) == set(source_files())
    checks["fixed_policy_arguments"] = raw.get("policy_config") == {
        "assignment": "fixed", "joint_rule": "urgent_short", "queue_window_ms": 1.0}
    checks["cir_interval_100ms"] = raw.get("cir_min_interval_ms") == 100.0
    checks["core_own_window_matches_manifest"] = (
        raw.get("metadata", {}).get("input_fingerprint") == item["input_fingerprint"]
        and (raw.get("common_window", {}).get("start_ms"), raw.get("common_window", {}).get("end_ms")) == (START, END))
    # A second small count directly from request timing fields catches accidental
    # completion-window clipping or a changed denominator in the shared helper.
    admissions = [r for r in raw["summary"]["request_metrics"] if START <= r["admission_time_ms"] < END]
    passed = sum(r["completion_time_ms"]-r["admission_time_ms"] <= 1.5*r["own_compute_ms"]+1e-9
                 for r in admissions)
    reported = metrics["cohorts"]["window_admissions"]
    checks["independent_uncensored_admission_SLO_count"] = (
        reported["count"] == len(admissions) and reported["admission"]["passed"] == passed
        and reported["completion_after_window_end_count"] == sum(r["completion_time_ms"] > END for r in admissions))
    window = parent["windows"][0]
    role_ok = all(card["by_role"][item["per_npu"][card["npu_id"]]["role"]]["compute_ms"] > 0
                  and all(group["active_ms"] == 0 for role, group in card["by_role"].items()
                          if role != item["per_npu"][card["npu_id"]]["role"])
                  for card in window["per_npu"])
    checks["warm_execution_obeys_declared_pure_roles"] = role_ok
    technical = all(checks.values())
    warm = parent["warmup"]["all_fourth_completions_by_1500"] and window["all_npus_active"] and role_ok
    scan = parent["nominal_demand_scan"]["full_run"]
    capacity = scan["max_ssu_gib_s"] < 40 and scan["max_npu_link_gib_s"] < 50
    slim_window = {k: v for k, v in window.items() if k != "requests"}
    return {"label": item["label"], "seed": item["seed"], "long_npus": item["long_npus"],
            "short_npus": item["short_npus"], "order": item["order"], "strategy": raw["strategy"],
            "path": str(path), "file_sha256": sha(path), "input_fingerprint": item["input_fingerprint"],
            "status": "passed" if technical else "audit_failed", "checks": checks,
            "warm_window_valid_without_per_card_mixing_requirement": warm,
            "full_run_strict_nominal_capacity_satisfied": capacity,
            "all_study_conditions_met": technical and warm and capacity,
            "warmup": parent["warmup"], "window": slim_window,
            "nominal_demand_scan": parent["nominal_demand_scan"], "metrics": metrics,
            "core_source_hashes": raw.get("core_and_policy_sha256"),
            "static_path_cirs": raw.get("static_path_cirs_gib_s"), "submit_seed": raw.get("submit_seed")}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    auditor = load_module("role_separated_parent_auditor", BASE/"analyze.py")
    auditor.WINDOWS = ((START, END),)
    paired = load_module("role_separated_slo_auditor", BASE/"paired_once_5seeds/analyze_paired.py")
    inputs, runs, failures, pairs = [], [], [], []
    plan = read(HERE/"plan.json")
    require(tuple(plan["seeds"]) == SEEDS and tuple(plan["strategies"]) == STRATEGIES,
            "Frozen plan seeds or strategies differ")
    require(set(plan["orders"]) == {"random", "round_robin"} and set(plan["long_npu_counts"]) == {11, 6},
            "Frozen plan order/ratio grid differs")
    require(len(plan["inputs"]) == 20 and len(plan["jobs"]) == 40, "Frozen plan grid size differs")
    require(all(sha(ROOT/name) == expected for name, expected in plan["source_sha256"].items()),
            "Frozen plan source hash changed")
    planned_inputs = {x["label"]: x for x in plan["inputs"]}
    for path in sorted((HERE/"inputs").glob("*.json.gz")):
        try:
            item = audit_input(path, auditor)
            expected = planned_inputs[item["label"]]
            require(item["manifest_sha256"] == expected["manifest_sha256"]
                    and item["input_fingerprint"] == expected["input_fingerprint"],
                    "Manifest bytes/fingerprint differ from frozen plan")
            inputs.append(item)
            print(json.dumps({"input": item["label"], "status": item["status"],
                              "minimum_C_ms": item["min_per_npu_ideal_compute_ms"],
                              "static_disk_bound": item["exact_assigned_npu_static_per_ssu_gib_s"]}), flush=True)
        except Exception as error:
            failures.append({"input": str(path), "error": str(error), "traceback": traceback.format_exc()})
    groups = defaultdict(dict)
    for item in inputs:
        key = (item["seed"], item["long_npus"])
        if item["order"] in groups[key]:
            failures.append({"error": "Duplicate input order", "group": key, "order": item["order"]})
        groups[key][item["order"]] = item
    for key, group in sorted(groups.items()):
        row = {"seed": key[0], "long_npus": key[1], "status": "pending"}
        if set(group) == {"random", "round_robin"}:
            a, b = group["random"], group["round_robin"]
            checks = {"same_per_new_npu_original_population": a["assignment_population_sha256"] == b["assignment_population_sha256"],
                      "same_long_identity_order": a["long_identity_order_sha256"] == b["long_identity_order_sha256"],
                      "same_source_manifest": a["source_manifest_sha256"] == b["source_manifest_sha256"],
                      "same_per_npu_compute_and_static_certificate": a["per_npu"] == b["per_npu"]
                          and a["exact_assigned_npu_static_per_ssu_gib_s"] == b["exact_assigned_npu_static_per_ssu_gib_s"]}
            row.update(status="passed" if all(checks.values()) else "audit_failed", checks=checks)
        pairs.append(row)
    if not args.inputs_only:
        for item in inputs:
            for strategy in STRATEGIES:
                directory = HERE/"runs"/item["label"]/strategy
                outputs = [p for p in directory.glob("*.json.gz") if ".failure." not in p.name]
                if len(outputs) != 1:
                    status = "pending" if not outputs else "duplicate_results"
                    command = directory/"command.json"
                    if not outputs and command.exists():
                        status = read(command).get("status", "running")
                    runs.append({"label": item["label"], "strategy": strategy, "status": status,
                                 "failure_artifacts": [str(p) for p in directory.glob("*.failure*")]})
                    continue
                try:
                    record = audit_result(outputs[0], item, strategy, plan, auditor, paired)
                except Exception as error:
                    record = {"label": item["label"], "strategy": strategy, "path": str(outputs[0]),
                              "status": "audit_failed", "error": str(error), "traceback": traceback.format_exc()}
                runs.append(record)
                print(json.dumps({"result": item["label"], "strategy": strategy,
                                  "status": record["status"], "valid": record.get("all_study_conditions_met")}), flush=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "metric_sources_sha256": {str(p.relative_to(BASE)): sha(p) for p in (BASE/"analyze.py", BASE/"paired_once_5seeds/analyze_paired.py")},
              "definitions": {"source_population": "All original 19456 requests from the same seed; preserve every SSD block placement",
                  "binding": "Only declared input NPU binding and queue-position IDs may change; runtime must keep the new manifest binding",
                  "warm_validity": "Every declared pure-role NPU active throughout [2000,4000), positive compute in its assigned role; four completions by1500; no per-card mixing requirement",
                  "main_slo": "Requests admitted in [2000,4000), followed through completion; completion-admission <=1.5*8*per_layer_C",
                  "static_role_bound": "Per SSD, n_long*max_over_source_long(V_s/C)+n_short*max_over_source_short(V_s/C); any pure-role binding/order, fixed physical placements",
                  "capacity_scope": "Current admitted-profile nominal V/C, not SSD throughput, released-I/O burst capacity, or a deadline guarantee",
                  "warm_cohort_caveat": "Warm admission populations depend on policy/order; complete populations remain matched"},
              "expected_inputs": 20, "expected_results": 40,
              "counts": {"inputs": len(inputs), "input_failures": len(failures), "order_pairs": len(pairs),
                         "order_pairs_passed": sum(x["status"] == "passed" for x in pairs),
                         "results": len(runs), "result_statuses": dict(Counter(x["status"] for x in runs))},
              "inputs": inputs, "input_failures": failures, "order_pairs": pairs, "runs": runs}
    atomic_json(HERE/"audit"/("inputs.json" if args.inputs_only else "audit.json"), report)
    print(json.dumps(report["counts"]), flush=True)
    bad = failures or any(x["status"] == "audit_failed" for x in pairs+runs)
    if args.require_complete:
        bad = bad or len(inputs) != 20 or len(pairs) != 10 or any(x["status"] != "passed" for x in pairs)
        if not args.inputs_only:
            bad = bad or len(runs) != 40 or any(x.get("all_study_conditions_met") is not True for x in runs)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
