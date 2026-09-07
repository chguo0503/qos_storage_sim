#!/usr/bin/env python3
"""Read frozen stress results, audit matched inputs, and export CSV/figures.

No simulation or source mutation. Re-run while results are being collected:
    python results/baseline_npu32_investigation/analyze_results.py
Incomplete jobs and every recorded failure remain visible in the exports.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
CORE = ("sim.py", "continuous_batch_sim.py", "continuous_prefill_client.py",
        "policy_logic.py", "strategy_profiles.py", "data")
EPS = 1e-6


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})
    temporary.replace(path)


def suite_name(stage):
    try:
        return str(stage.relative_to(BASE))
    except ValueError:
        return stage.name


def is_current_artifact(path):
    return not any(part in ("archive", "_archive") or part.startswith("interrupted_attempt_")
                   for part in path.parts)


def identity(path, result=None):
    if "runs" in path.parts:
        i = path.parts.index("runs")
        return suite_name(Path(*path.parts[:i])), path.parts[i + 1], path.parts[i + 2]
    meta = (result or {}).get("metadata", {})
    return "pilot", meta.get("case_id", path.stem), (result or {}).get("strategy", "unknown")


def manifest_info(path, result, cache):
    candidates = []
    if "manifest_path" in result:
        candidates.append(Path(result["manifest_path"]))
    if "runs" in path.parts:
        candidates.append(path.parents[3] / "inputs" / (path.parents[1].name + ".json.gz"))
        # Split-host custom plans keep one shared frozen input directory above
        # their local/remote run directories.
        candidates.append(path.parents[4] / "inputs" / (path.parents[1].name + ".json.gz"))
    candidates.append(path.parent / "inputs" / (result["metadata"]["case_id"] + ".json.gz"))
    manifest = next((p for p in candidates if p.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(f"No local frozen manifest for {path}")
    key = str(manifest.resolve())
    if key in cache:
        return cache[key]
    payload = read_json(manifest)
    placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p)
                  for p in payload["placements"]]
    placement_reprs = [repr(p) for p in placements]
    digest = hashlib.sha256(b"full-prefill-microbatch-des-input-v2\0")
    role_map, input_rows, original_npus, expected_io, expected_io_by_ssu, homes = {}, [], {}, {}, {}, {}
    layers, num_ssu = payload["metadata"]["n_layers"], payload["metadata"]["num_ssu"]
    placement_counts, placement_bytes = [], []
    for placement in placements:
        counts, volumes = [0]*num_ssu, [0.0]*num_ssu
        factor = layers if len(placement)==1 else 1
        for layer in placement:
            for ssu,volume in layer:
                counts[ssu] += factor
                volumes[ssu] += factor*volume
        placement_counts.append(counts)
        placement_bytes.append(volumes)
    expected_gib = [0.0]*num_ssu
    # Recompute the native fingerprint protocol, caching repeated placement text.
    for r in sorted(payload["requests"], key=lambda row: row["request_id"]):
        load = r["load"]
        prefix = repr((r["request_id"], r["npu_id"], r["arrival_time_ms"],
                       load["category"], load["per_layer_us"]))[:-1]
        digest.update((prefix + ", " + placement_reprs[r["placement_index"]] + ")").encode())
        role_map[r["request_id"]] = load.get("role", "unknown")
        original_npus[r["request_id"]] = r["npu_id"]
        counts = placement_counts[r["placement_index"]]
        expected_io[r["request_id"]] = sum(counts)
        expected_io_by_ssu[r["request_id"]] = counts
        nonempty = [s for s,count in enumerate(counts) if count]
        homes[r["request_id"]] = nonempty[0] if len(nonempty)==1 else None
        for s,v in enumerate(placement_bytes[r["placement_index"]]):
            expected_gib[s] += v
        input_rows.append((r["request_id"], r["arrival_time_ms"],
                           payload["metadata"]["n_layers"] * load["per_layer_us"] / 1000))
    info = {"path": key, "sha256": file_hash(manifest),
            "embedded_fingerprint": payload["input_fingerprint"],
            "recomputed_fingerprint": digest.hexdigest(), "role_map": role_map,
            "input_rows": input_rows, "request_count": len(input_rows),
            "original_npus": original_npus, "expected_io": expected_io,
            "expected_io_by_ssu": expected_io_by_ssu, "home_ssu_by_request": homes,
            "expected_completed_gib_by_ssu": expected_gib,
            "computed_padding_total_gib": (layers * sum(r["load"]["padding_gib_per_layer"] for r in payload["requests"])
                if all("padding_gib_per_layer" in r["load"] for r in payload["requests"]) else None)}
    cache[key] = info
    return info


def overlap(a, b, left, right):
    return max(0.0, min(b, right) - max(a, left))


def max_error(a, b):
    if len(a) != len(b):
        return math.inf
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def full_time_account(summary):
    n, makespan = summary["num_npu"], summary["makespan_ms"]
    compute, active, last = [0.0]*n, [0.0]*n, [0.0]*n
    counts = [0]*n
    for batch in summary["microbatch_metrics"]:
        i = batch["npu_id"]
        compute[i] += sum(layer["compute_end_ms"]-layer["compute_start_ms"]
                          for layer in batch["layer_metrics"])
        active[i] += batch["completion_time_ms"]-batch["admission_time_ms"]
        last[i] = max(last[i],batch["completion_time_ms"])
        counts[i] += len(batch["member_request_ids"])
    stall = [a-c for a,c in zip(active,compute)]
    idle = [makespan-a for a in active]
    tail_idle = [makespan-t for t in last]
    account = {"full_compute_ms_by_npu":compute,"full_active_ms_by_npu":active,
               "full_stall_ms_by_npu":stall,"full_idle_ms_by_npu":idle,
               "tail_idle_ms_by_npu":tail_idle,"last_completion_ms_by_npu":last,
               "executed_request_count_by_npu":counts}
    for key,values in (("compute",compute),("active",active),("stall",stall),("idle",idle),("tail_idle",tail_idle)):
        account[f"full_mean_{key}_ms_per_npu"] = sum(values)/n
        account[f"full_fleet_{key}_fraction"] = sum(values)/(n*makespan)
    return account


def native_assignment_details(result, manifest, result_path):
    """Independent home-SSU attribution; read lifetimes are NOT SSD busy time."""
    if "native_strategy" not in result:
        return None
    summary, meta = result["summary"], result["metadata"]
    root = BASE.parents[1]
    source_checks = {name: (root/name).is_file() and file_hash(root/name)==digest
                     for name,digest in result.get("probe_source_sha256",{}).items()}
    wrapper = BASE/"run_native_assignment_probe.py"
    own_hash_match = wrapper.is_file() and file_hash(wrapper)==result.get("runner_sha256")
    windows = []
    for w in result["windows"]:
        left,right = w["start_ms"],w["end_ms"]
        by_home = [{"home_ssu":s,"active_npu_ms":0.0,"compute_npu_ms":0.0,
                    "warm_stall_npu_ms":0.0,"layer0_stall_npu_ms":0.0,
                    "layer_read_lifetime_sum_ms":0.0,"execution_npus":set(),"original_npus":set()}
                   for s in range(meta["num_ssu"])]
        multi_ssu_batches = 0
        for batch in summary["microbatch_metrics"]:
            rid = batch["member_request_ids"][0]
            home = manifest["home_ssu_by_request"][rid]
            if home is None:
                multi_ssu_batches += 1
                continue
            row = by_home[home]
            active = overlap(batch["admission_time_ms"],batch["completion_time_ms"],left,right)
            row["active_npu_ms"] += active
            if active:
                row["execution_npus"].add(batch["npu_id"])
                row["original_npus"].add(manifest["original_npus"][rid])
            previous = batch["admission_time_ms"]
            for layer in batch["layer_metrics"]:
                row["compute_npu_ms"] += overlap(layer["compute_start_ms"],layer["compute_end_ms"],left,right)
                row["layer_read_lifetime_sum_ms"] += overlap(layer["io_start_time_ms"],layer["io_ready_time_ms"],left,right)
                key = "layer0_stall_npu_ms" if layer["layer"]==0 else "warm_stall_npu_ms"
                row[key] += overlap(previous,layer["compute_start_ms"],left,right)
                previous = layer["compute_end_ms"]
        for row in by_home:
            row["execution_npus"] = sorted(row["execution_npus"])
            row["original_npus"] = sorted(row["original_npus"])
        windows.append({"start_ms":left,"end_ms":right,"by_home_ssu":by_home,
                        "multi_ssu_batches_excluded":multi_ssu_batches,
                        "active_compute_stall_error_ms":sum(row["active_npu_ms"]-row["compute_npu_ms"]
                            -row["warm_stall_npu_ms"]-row["layer0_stall_npu_ms"] for row in by_home)})
    return {"result":str(result_path),"strategy":result["strategy"],"assignment":result.get("assignment"),
        "input_fingerprint":result["input_fingerprint"],"request_count":summary["request_count"],
        "source_hashes_match_local":source_checks,"wrapper_source_hash_match":own_hash_match,
        "submitted_blocks":summary["submitted_blocks"],"completed_blocks":summary["completed_blocks"],
        "full_run_makespan_ms":summary["makespan_ms"],
        "full_run_physical_ssu_stats":summary["disk_stats"],"windows":windows,
        "native_arrival_tie_order":"(arrival_time_ms, original_npu_id, request_id); shuffling only the input tuple does not change this order",
        "first_32_arrival_assignments":[{"request_id":r["request_id"],"arrival_time_ms":r["arrival_time_ms"],
            "original_npu_id":r["original_npu_id"],"assigned_npu_id":r["assigned_npu_id"],
            "home_ssu":manifest["home_ssu_by_request"][r["request_id"]]}
            for r in result.get("assignment_log",[])[:32]],
        "evidence_boundary":"NPU binding intentionally changes under fluid assignment. Original SSD placement is retained by source-level dataclass replacement and checked at every physical _register_submit; per-request and per-disk I/O conservation is independently rechecked. This wrapper did not store an execution-placement fingerprint.",
        "read_lifetime_caveat":"layer_read_lifetime_sum_ms adds release-to-HBM-ready lifetimes over concurrent layers; it can exceed window duration and is NOT SSD busy time. Zero on an SSU means no layer read lifespan for that one-SSU-per-layer workload intersects the window. Exact per-window SSD busy time is not recorded here; full-run physical utilization is separately available."}


def audit_result(path, result, cache):
    summary, meta = result["summary"], result["metadata"]
    manifest = manifest_info(path, result, cache)
    checks = {"simulator_invariants": all(summary["invariants"].values()),
              "manifest_recomputed_fingerprint": manifest["embedded_fingerprint"] == manifest["recomputed_fingerprint"],
              "result_manifest_fingerprint": result["input_fingerprint"] == manifest["recomputed_fingerprint"],
              "metadata_fingerprint": result["input_fingerprint"] == meta["input_fingerprint"]}
    request_rows = sorted(summary["request_metrics"], key=lambda r: r["request_id"])
    execution_inputs = [(r["request_id"], r["arrival_time_ms"], r["own_compute_ms"]) for r in request_rows]
    expected = manifest["input_rows"]
    checks["request_ids_and_arrivals"] = (len(execution_inputs) == len(expected)
        and all(a[:2] == b[:2] for a, b in zip(execution_inputs, expected)))
    checks["per_request_ideal_compute"] = (len(execution_inputs) == len(expected)
        and all(abs(a[2] - b[2]) < EPS for a, b in zip(execution_inputs, expected)))
    checks["all_requests_completed_finite"] = all(math.isfinite(r["completion_time_ms"]) for r in request_rows)
    checks["per_request_io_count"] = all(r["io_count"]==manifest["expected_io"][r["request_id"]] for r in request_rows)
    actual_gib = {r["ssu_id"]:r["completed_gb"] for r in summary["disk_stats"]}
    checks["physical_per_ssu_completed_gib"] = all(abs(actual_gib[s]-v)<EPS
        for s,v in enumerate(manifest["expected_completed_gib_by_ssu"]))
    assignments = result.get("assignment_log",[])
    moved = sum(r["npu_id"] != manifest["original_npus"][r["request_id"]] for r in request_rows)
    if assignments:
        assigned = {row["request_id"]:row for row in assignments}
        checks["assignment_log_complete_unique"] = len(assignments)==len(assigned)==len(request_rows)
        checks["assignment_log_matches_execution"] = all(r["request_id"] in assigned
            and r["npu_id"]==assigned[r["request_id"]]["assigned_npu_id"] for r in request_rows)
        checks["assignment_log_original_npu"] = all(row["original_npu_id"]==manifest["original_npus"][row["request_id"]]
                                                     for row in assignments)
        if all("incoming_total_io_by_ssu" in row for row in assignments):
            checks["assignment_incoming_placement_counts"] = all(row["incoming_total_io_by_ssu"]==manifest["expected_io_by_ssu"][row["request_id"]]
                                                                  for row in assignments)
    else:
        checks["fixed_assignment_preserved"] = moved==0
    if "manifest_file_sha256" in result:
        checks["wrapper_recorded_manifest_file_hash"] = result["manifest_file_sha256"]==manifest["sha256"]
    full = full_time_account(summary)
    checks["full_time_nonnegative"] = all(v >= -EPS for key,values in full.items()
        if key.endswith("_ms_by_npu") for v in values)
    checks["full_time_compute_within_active"] = all(c <= a+EPS for c,a in zip(full["full_compute_ms_by_npu"],full["full_active_ms_by_npu"]))
    checks["full_time_fleet_utilization"] = abs(full["full_fleet_compute_fraction"]-summary["fleet_npu_compute_utilization"]) < EPS
    checks["full_time_sum_compute_stall_idle"] = abs(full["full_fleet_compute_fraction"]+full["full_fleet_stall_fraction"]+full["full_fleet_idle_fraction"]-1) < EPS
    checks["full_time_active_equals_compute_stall"] = abs(full["full_fleet_compute_fraction"]+full["full_fleet_stall_fraction"]-full["full_fleet_active_fraction"]) < EPS
    checks["full_time_ideal_compute_conservation"] = abs(sum(full["full_compute_ms_by_npu"])-sum(r["own_compute_ms"] for r in request_rows)) < EPS
    windows = []
    for stored in result["windows"]:
        left, right = stored["start_ms"], stored["end_ms"]
        n, duration = meta["num_npu"], right - left
        compute, active = [0.0] * n, [0.0] * n
        roles = defaultdict(lambda: [0.0, 0.0])
        for batch in summary["microbatch_metrics"]:
            role = manifest["role_map"][batch["member_request_ids"][0]]
            a = overlap(batch["admission_time_ms"], batch["completion_time_ms"], left, right)
            c = sum(overlap(layer["compute_start_ms"], layer["compute_end_ms"], left, right)
                    for layer in batch["layer_metrics"])
            active[batch["npu_id"]] += a
            compute[batch["npu_id"]] += c
            roles[role][0] += c
            roles[role][1] += a
        errors = {"per_npu_compute_ms": max_error(compute, stored["compute_ms_by_npu"]),
                  "per_npu_active_ms": max_error(active, stored["active_ms_by_npu"]),
                  "per_npu_stall_ms": max_error([a-c for a,c in zip(active, compute)], stored["io_stall_ms_by_npu"]),
                  "per_npu_idle_ms": max_error([duration-a for a in active], stored["idle_ms_by_npu"]),
                  "fleet_utilization": abs(sum(compute)/(n*duration)-stored["mean_npu_utilization"])}
        role_keys_match = set(roles) == set(stored["by_role"])
        role_error = 0.0
        for role, (c, a) in roles.items():
            row = stored["by_role"].get(role, {})
            role_error = max(role_error, abs(c-row.get("compute_ms", 0)),
                             abs(a-row.get("active_ms", 0)), abs(a-c-row.get("exposed_stall_ms", 0)))
        errors["role_compute_active_stall_ms"] = role_error
        errors["role_compute_sum_ms"] = abs(sum(v["compute_ms"] for v in stored["by_role"].values())-sum(compute))
        errors["role_active_sum_ms"] = abs(sum(v["active_ms"] for v in stored["by_role"].values())-sum(active))
        errors["role_stall_sum_ms"] = abs(sum(v["exposed_stall_ms"] for v in stored["by_role"].values())-sum(a-c for a,c in zip(active,compute)))
        valid = role_keys_match and max(errors.values()) < EPS
        windows.append({"start_ms": left, "end_ms": right, "checks_pass": valid,
                        "roles_match": role_keys_match, "max_errors": errors})
    checks["all_window_accounting"] = all(w["checks_pass"] for w in windows)
    native_details = native_assignment_details(result,manifest,path)
    return {"result": str(path), "result_sha256": file_hash(path), "checks": checks,
            "all_checks_pass": all(checks.values()), "windows": windows,
            "full_time_account": full,
            "assignment_moved_request_count": moved,
            "native_assignment_audit": native_details,
            "placement_evidence": "Frozen original manifest and full per-request/per-SSU I/O conservation; native _register_submit checks each physical command's SSU and size against execution manifest. Native assignment source replaces only NPU ID, retaining placement.",
            "manifest": {k:v for k,v in manifest.items() if k not in ("role_map", "input_rows", "original_npus", "expected_io", "expected_io_by_ssu", "home_ssu_by_request")},
            "executed_request_population_sha256": value_hash(execution_inputs),
            "failed_invariants": [k for k,v in summary["invariants"].items() if not v]}


def result_rows(path, result, audit):
    suite, label, variant = identity(path, result)
    meta, summary = result["metadata"], result["summary"]
    policy = result.get("policy_config", {})
    strategy = result["strategy"]
    actual_assignment = (policy.get("assignment", "fixed") if strategy.startswith("strategy")
                         else result.get("assignment", "fixed"))
    if actual_assignment == "none":
        actual_assignment = "fixed"
    common = {"suite": suite, "input": label, "variant": variant, "strategy": strategy,
        "assignment": actual_assignment, "assignment_moved_request_count": audit["assignment_moved_request_count"],
        "native_strategy": result.get("native_strategy"),
        "status": "complete" if audit["all_checks_pass"] else "audit_failed",
        "input_fingerprint": result["input_fingerprint"], "result": str(path),
        "result_sha256": audit["result_sha256"], "audit_pass": audit["all_checks_pass"],
        "num_npu": meta["num_npu"], "num_ssu": meta["num_ssu"], "seed": meta["seed"],
        "family": meta["family"], "layout": meta["layout"], "order": meta["order"], "blocks": meta["blocks"],
        "profile_keys": meta.get("profile_keys") or meta.get("raw_keys"),
        "compute_scale": meta["compute_scale_actual"],
        "hottest_ssu_rho": meta["input_demand"]["hottest_ssu_load_ratio"],
        "largest_npu_receive_rho": meta["input_demand"]["largest_npu_receive_load_ratio"],
        "total_nominal_gib_s": meta["input_demand"]["total_gib_s"],
        "padding_total_gib": meta.get("padding_total_gib", audit["manifest"]["computed_padding_total_gib"]),
        "collector_interval_ms": result.get("collector_interval_ms", 0),
        "pressure_semantics": "none" if strategy in ("baseline", "baseline_native") else "5 ms collector" if "collector_interval_ms" in result else "instant native",
        "queue_window_ms": policy.get("queue_window_ms"), "joint_rule": policy.get("joint_rule"),
        "request_count": summary["request_count"], "makespan_ms": summary["makespan_ms"],
        "full_run_utilization": summary["fleet_npu_compute_utilization"],
        "mean_arrival_latency_ms": summary["avg_request_latency_ms"],
        "p99_arrival_latency_ms": summary["p99_request_latency_ms"],
        "mean_admission_wait_ms": summary["avg_admission_wait_ms"],
        "admission_slo": result["slo"]["all_requests"]["admission"]["rate"],
        "arrival_slo": result["slo"]["all_requests"]["arrival"]["rate"],
        "wall_seconds": result["wall_seconds_total"], **audit["full_time_account"]}
    rows, role_rows = [], []
    for w in result["windows"]:
        row = {**common, "start_ms": w["start_ms"], "end_ms": w["end_ms"],
            "fleet_utilization": w["mean_npu_utilization"], "min_npu_utilization": min(w["npu_utilizations"]),
            "all_npus_active": w["all_npus_active_whole_window"],
            "idle_ms": sum(w["idle_ms_by_npu"]), "stall_ms": sum(w["io_stall_ms_by_npu"]),
            "npu_utilizations": w["npu_utilizations"],
            "role_compute_fractions": {k:v["active_compute_fraction"] for k,v in w["by_role"].items()}}
        rows.append(row)
        for role, values in w["by_role"].items():
            role_rows.append({**common, "start_ms": w["start_ms"], "end_ms": w["end_ms"], "role": role, **values})
    return rows, role_rows


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_existing(base, output):
    import matplotlib.pyplot as plt
    source = base / "notes/existing_audit_data/short_cohort_metrics.csv"
    if not source.exists():
        return {"name": "01_existing_coflow", "status": "waiting_for_existing_audit"}
    with source.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["experiment"] == "coflow"]
    strategies = ["baseline", "once", "new_once", "strategy1", "strategy2", "strategy3"]
    labels = ["Baseline", "Once", "New Once", "S1", "S2", "S3"]
    groups = defaultdict(dict)
    for row in rows:
        groups[(int(row["ssu"]), int(row["seed"]))][row["strategy"]] = row
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.2), sharey=True)
    colors = {5: "#B84A48", 6: "#3676A2", 7: "#4E8C62"}
    for (ssu, seed), by_strategy in sorted(groups.items()):
        if not all(s in by_strategy for s in strategies):
            continue
        label = f"{ssu} SSUs, seed {str(seed)[-2:]}"
        style = "-" if seed % 2 == 0 else "--"
        for ax, metric in zip(axes, ("window_utilization", "aggregate_compute_fraction")):
            ax.plot(range(6), [100*float(by_strategy[s][metric]) for s in strategies],
                    marker="o", markersize=4, color=colors[ssu], linestyle=style, label=label)
    axes[0].set_title("Fleet computation in [1000, 2000] ms")
    axes[1].set_title("Short cohort per input: compute / active time")
    for ax in axes:
        ax.set_xticks(range(6), labels)
        ax.set_ylim(0, 103)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Original coflow results: a high fleet average can hide short-request stalls", fontsize=12)
    fig.text(.5, -.01, "Right: sum(compute) / sum(completion - admission); 192 short requests per input, matched across policies (32K/128 + 32K/256). 5 ms setup.",
             ha="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "01_existing_coflow")
    write_csv(output / "01_existing_coflow_data.csv", rows)
    return {"name": "01_existing_coflow", "status": "written", "source": str(source), "source_sha256": file_hash(source)}


def plot_screen(rows, output):
    import matplotlib.pyplot as plt
    anchors = {r["input_fingerprint"]: r["input"] for r in rows
               if r.get("suite") == "screen" and r.get("status") == "complete"}
    label_counts = defaultdict(int)
    for label in anchors.values():
        label_counts[label] += 1
    display = {fp: label if label_counts[label] == 1 else f"{label} [{fp[:8]}]"
               for fp, label in anchors.items()}
    candidates = [r for r in rows if r.get("input_fingerprint") in anchors and r.get("start_ms") == 1000
                  and r.get("end_ms") == 2000 and r.get("status") == "complete"]
    data, seen = [], set()
    # Join missing policies from later phases by fingerprint. Prefer the original
    # screen run for duplicate policies; never choose the highest utilization.
    for r in sorted(candidates, key=lambda row: (row["suite"] != "screen", row["result"])):
        if r["strategy"] == "strategy3" and (r["assignment"] != "fixed"
                or r["queue_window_ms"] != 1.0 or r["joint_rule"] != "urgent_short"):
            continue
        key = r["input_fingerprint"], r["strategy"]
        if key in seen:
            continue
        seen.add(key)
        data.append({**r, "source_input_label": r["input"], "input": display[r["input_fingerprint"]]})
    native = sorted({r["input"] for r in data if r["strategy"] == "baseline_native"})
    shared = sorted({r["input"] for r in data if r["strategy"] == "baseline"})
    groups = [(native, ("baseline_native", "once_native"), "Native comparison: instantaneous Once pressure")]
    groups += [(shared, ("baseline", "once", "strategy3"), "5 ms comparison: Once telemetry; S3 uses fixed NPU assignment")]
    groups = [g for g in groups if g[0]]
    if not groups:
        return {"name": "02_screen_comparison", "status": "waiting_for_screen_results"}
    fig, axes = plt.subplots(len(groups), 1, figsize=(11, max(4, .42*(len(native)+len(shared))+2)),
                             squeeze=False, gridspec_kw={"height_ratios": [len(g[0])+1 for g in groups]})
    selected = []
    markers = ("o", "s", "D")
    colors = ("#333333", "#3676A2", "#C67C2C")
    for ax, (labels, strategies, title) in zip(axes[:, 0], groups):
        for idx, strategy in enumerate(strategies):
            choices = [r for r in data if r["strategy"] == strategy and r["input"] in labels
                       and (strategy != "strategy3" or r["assignment"] == "fixed")]
            xs, ys = [], []
            for r in choices:
                xs.append(100*r["fleet_utilization"])
                ys.append(labels.index(r["input"]))
                selected.append(r)
            ax.scatter(xs, ys, marker=markers[idx], color=colors[idx], s=42,
                       label=("Baseline", "Once", "S3 fixed")[idx], zorder=3)
        ax.set_yticks(range(len(labels)), [label.replace("_", " ") for label in labels])
        ax.invert_yaxis()
        ax.set_xlim(0, 103)
        ax.set_xlabel("Fleet NPU compute utilization in [1000, 2000] ms (%)")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "02_screen_comparison")
    write_csv(output / "02_screen_comparison_data.csv", selected)
    return {"name": "02_screen_comparison", "status": "written", "plotted_rows": len(selected),
            "caveat": "Partial result collection remains partial; missing policies are not imputed."}


def timeline_segments(result, npus, left, right):
    records = []
    for batch in result["summary"]["microbatch_metrics"]:
        if batch["npu_id"] not in npus:
            continue
        previous_end = batch["admission_time_ms"]
        for layer in batch["layer_metrics"]:
            for state, start, end in (("stall", previous_end, layer["compute_start_ms"]),
                                      ("compute", layer["compute_start_ms"], layer["compute_end_ms"])):
                if overlap(start, end, left, right) > 1e-10:
                    records.append({"strategy": result["strategy"], "input_fingerprint": result["input_fingerprint"],
                        "npu_id": batch["npu_id"], "request_id": batch["member_request_ids"][0],
                        "layer": layer["layer"], "state": state, "actual_start_ms": start, "actual_end_ms": end,
                        "clipped_start_ms": max(start,left), "clipped_end_ms": min(end,right)})
            previous_end = layer["compute_end_ms"]
    return records


def plot_timeline(records, output, label="strong32_local", left=1000, right=1030):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    anchors = {r["input_fingerprint"] for r in records
               if r["suite"] == "screen" and r["input"] == label and r["audit_pass"]}
    if len(anchors) > 1:
        raise AssertionError(f"Ambiguous timeline label with multiple manifests: {label}")
    choices = {}
    for r in sorted(records, key=lambda row: (row["suite"] != "screen", row["result"])):
        if r["input_fingerprint"] in anchors and r["audit_pass"]:
            choices.setdefault(r["strategy"], r)
    if "baseline" not in choices or "once" not in choices:
        return {"name": "03_strong32_timeline", "status": "waiting_for_matched_baseline_once", "input": label}
    paired = [choices[strategy] for strategy in ("baseline", "once")]
    if paired[0]["input_fingerprint"] != paired[1]["input_fingerprint"]:
        raise AssertionError("timeline inputs do not match")
    npus = (0,1,2,3)
    fig, axes = plt.subplots(2, 1, figsize=(12,5.6), sharex=True)
    rows = []
    colors = {"compute": "#4C936E", "stall": "#C65B49"}
    for ax, rec in zip(axes, paired):
        result = read_json(Path(rec["result"]))
        segments = timeline_segments(result, npus, left, right)
        rows.extend(segments)
        for n in npus:
            for state in ("compute", "stall"):
                intervals = [(r["clipped_start_ms"],r["clipped_end_ms"]-r["clipped_start_ms"])
                             for r in segments if r["npu_id"]==n and r["state"]==state]
                ax.broken_barh(intervals, (n-.32,.64), facecolors=colors[state], linewidth=0)
            starts = [r["actual_start_ms"] for r in segments if r["npu_id"]==n
                      and r["state"]=="compute" and left < r["actual_start_ms"] < right]
            ax.vlines(starts,n-.32,n+.32,color="white",linewidth=.5,alpha=.7)
        ax.set_yticks(npus,[f"NPU {n}" for n in npus])
        ax.set_ylim(3.65,-.65)
        ax.set_title("Baseline: actual compute and exposed I/O stall" if result["strategy"]=="baseline"
                     else "Once (5 ms): actual compute and exposed I/O stall",loc="left",fontsize=11)
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[-1].set_xlim(left,right)
    axes[-1].set_xlabel("Absolute simulation time (ms)")
    axes[0].legend(handles=[Patch(facecolor=colors[k],label=k.title()) for k in colors],
                   loc="lower right",bbox_to_anchor=(1,1.005),ncol=2,fontsize=8,frameon=False)
    fig.suptitle(f"{paired[0]['num_npu']} NPUs / {paired[0]['num_ssu']} SSUs: cards 0-3 form one local SSU group",fontsize=12)
    fig.text(.5,.005,"Each panel uses that policy's own chronological events; thin white ticks mark layer starts. Boundary intervals are clipped; blank space means no admitted work.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.97))
    save_figure(fig,output,"03_strong32_timeline")
    write_csv(output / "03_strong32_timeline_segments.csv",rows)
    return {"name":"03_strong32_timeline","status":"written","input_fingerprint":paired[0]["input_fingerprint"],
            "sources":[r["result"] for r in paired],"npu_ids":npus,"window_ms":[left,right],"segments":len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=BASE)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--no-figures",action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    output = (args.output or base / "analysis").resolve()
    output.mkdir(parents=True,exist_ok=True)
    paths = sorted({p for p in base.glob("**/runs/*/*/*.json.gz") if is_current_artifact(p)} | {
        p for p in (base/"pilot").rglob("*.json.gz") if "inputs" not in p.relative_to(base).parts and is_current_artifact(p)})
    rows, roles, audits, records, failures, full_cards = [], [], [], [], [], []
    manifest_cache, groups, successful_jobs = {}, defaultdict(list), set()
    for index,path in enumerate(paths,1):
        try:
            result = read_json(path)
            if "summary" not in result:
                continue
            audit = audit_result(path,result,manifest_cache)
            audits.append(audit)
            new_rows,new_roles = result_rows(path,result,audit)
            rows.extend(new_rows);roles.extend(new_roles)
            record = {k:v for k,v in new_rows[0].items() if k not in ("npu_utilizations","role_compute_fractions")}
            records.append(record)
            full = audit["full_time_account"]
            for i in range(record["num_npu"]):
                full_cards.append({"suite":record["suite"],"input":record["input"],
                    "variant":record["variant"],"strategy":record["strategy"],"assignment":record["assignment"],
                    "input_fingerprint":record["input_fingerprint"],"result":str(path),"npu_id":i,
                    "makespan_ms":record["makespan_ms"],
                    "compute_ms":full["full_compute_ms_by_npu"][i],"active_ms":full["full_active_ms_by_npu"][i],
                    "stall_ms":full["full_stall_ms_by_npu"][i],"idle_ms":full["full_idle_ms_by_npu"][i],
                    "tail_idle_ms":full["tail_idle_ms_by_npu"][i],"last_completion_ms":full["last_completion_ms_by_npu"][i],
                    "executed_request_count":full["executed_request_count_by_npu"][i]})
            successful_jobs.add((record["suite"],record["input"],record["variant"]))
            groups[result["input_fingerprint"]].append({"suite":record["suite"],"input":record["input"],
                "strategy":result["strategy"],"variant":record["variant"],"result":str(path),
                "audit_pass":audit["all_checks_pass"],"submit_seed":result["submit_seed"],
                "request_population_sha256":audit["executed_request_population_sha256"],
                "core_sha256":{k:v for k,v in result["core_and_policy_sha256"].items() if k in CORE},
                "window_ms":[[w["start_ms"],w["end_ms"]] for w in result["windows"]]})
        except Exception as exc:
            suite,label,variant=identity(path)
            row={"suite":suite,"input":label,"variant":variant,"status":"analysis_error", "result":str(path),"error":repr(exc)}
            rows.append(row);failures.append(row)
        print(json.dumps({"read_results":index,"total_snapshot":len(paths),"file":path.name}),flush=True)
    status_rows={}
    status_records=[]
    pending_status_records=[]
    for path in sorted(p for p in base.glob("**/*_status.json") if is_current_artifact(p)):
        payload=read_json(path)
        for row in payload.get("rows",[]):
            status_suite=path.parent/row["batch"] if row.get("batch") else path.parent
            key=(suite_name(status_suite),row.get("input",row.get("label")),row.get("variant",row["strategy"]))
            status_rows[key]=row
            status_record={"suite":key[0],"input":key[1],"variant":key[2],"source":str(path),**row,
                           "complete_result_collected":key in successful_jobs}
            status_records.append(status_record)
            if row["status"] in ("failed","failure","timeout","timed_out","error","analysis_error","orchestrator_error","cancelled","interrupted") or row.get("returncode") not in (None,0):
                failures.append(status_record)
            elif row["status"] not in ("complete","existing"):
                pending_status_records.append(status_record)
    planned_jobs_seen=set()
    plan_paths=set(base.glob("**/*_plan.json")) | set(base.glob("**/plan.json"))
    for path in sorted(p for p in plan_paths if p.parent!=base and is_current_artifact(p)):
        plan=read_json(path)
        jobs=plan.get("jobs",[])
        if not jobs and isinstance(plan.get("inputs"),list) and (path.parent/"runs").is_dir():
            jobs=[{"input":item["label"],"strategy":strategy}
                  for item in plan["inputs"] for strategy in plan.get("strategies",[])]
        for job in jobs:
            key=(suite_name(path.parent),job["input"],job.get("variant",job["strategy"]))
            if key in planned_jobs_seen:
                continue
            planned_jobs_seen.add(key)
            if key in successful_jobs:
                continue
            saved=status_rows.get(key)
            command=path.parent/"runs"/key[1]/key[2]/"command.json"
            status=saved["status"] if saved else "started_uncollected" if command.exists() else "not_started_or_not_collected"
            if status in ("complete","existing"):
                status="complete_result_not_collected"
            rows.append({"suite":key[0],"input":key[1],"variant":key[2],"strategy":job["strategy"],"status":status,"plan":str(path)})
    for path in sorted(p for p in base.glob("**/runs/*/*/*.failure.json") if is_current_artifact(p)):
        failures.append({"source":str(path),**read_json(path)})
    comparisons=[]
    comparison_groups=defaultdict(list)
    for row in rows:
        if row.get("status")=="complete":
            comparison_groups[(row["input_fingerprint"],row["start_ms"],row["end_ms"])].append(row)
    for key,members in sorted(comparison_groups.items()):
        baselines=[r for r in members if r["strategy"] in ("baseline","baseline_native")]
        for baseline in baselines:
            for candidate in members:
                if baseline["result"]==candidate["result"]:
                    continue
                comparisons.append({"input_fingerprint":key[0],"start_ms":key[1],"end_ms":key[2],
                    "reference_suite":baseline["suite"],"reference_input":baseline["input"],
                    "reference_strategy":baseline["strategy"],"reference_result":baseline["result"],
                    "candidate_suite":candidate["suite"],"candidate_input":candidate["input"],
                    "candidate_strategy":candidate["strategy"],"candidate_variant":candidate["variant"],
                    "candidate_assignment":candidate["assignment"],"candidate_result":candidate["result"],
                    "reference_utilization":baseline["fleet_utilization"],"candidate_utilization":candidate["fleet_utilization"],
                    "delta_utilization_percentage_points":100*(candidate["fleet_utilization"]-baseline["fleet_utilization"]),
                    "delta_makespan_ms":candidate["makespan_ms"]-baseline["makespan_ms"],
                    "delta_admission_slo_percentage_points":100*(candidate["admission_slo"]-baseline["admission_slo"]),
                    "reference_all_active":baseline["all_npus_active"],"candidate_all_active":candidate["all_npus_active"]})
    matched=[]
    for fingerprint,members in sorted(groups.items()):
        cores=defaultdict(set)
        for member in members:
            for name,digest in member["core_sha256"].items():cores[name].add(digest)
        checks={"all_individual_audits":all(m["audit_pass"] for m in members),
                "same_submit_seed":len({m["submit_seed"] for m in members})==1,
                "same_request_population":len({m["request_population_sha256"] for m in members})==1,
                "same_recorded_core_sources":all(len(v)==1 for v in cores.values()),
                "same_windows":len({json.dumps(m["window_ms"]) for m in members})==1}
        matched.append({"input_fingerprint":fingerprint,"result_count":len(members),
                        "checks":checks,"all_checks_pass":all(checks.values()),"members":members})
    figures=[]
    if not args.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,"svg.fonttype":"none"})
        figures=[plot_existing(base,output),plot_screen(rows,output),plot_timeline(records,output)]
    write_csv(output/"summary.csv",rows)
    write_csv(output/"roles.csv",roles)
    write_csv(output/"matched_comparisons.csv",comparisons)
    write_csv(output/"full_run_per_npu.csv",full_cards)
    write_json(output/"failures.json",failures)
    write_json(output/"source_status_records.json",status_records)
    write_json(output/"pending_status_records.json",pending_status_records)
    write_json(output/"native_assignment_audit.json",[a["native_assignment_audit"] for a in audits if a["native_assignment_audit"] is not None])
    artifact={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),
        "method":"read-only independent reaggregation of complete simulator event summaries; no simulation",
        "analysis_source_sha256":file_hash(Path(__file__)),"result_files_read":len(audits),
        "result_files_in_snapshot":len(paths),"all_result_audits_pass":bool(audits) and len(audits)==len(paths) and len(records)==len(paths) and all(a["all_checks_pass"] for a in audits),
        "all_result_exports_pass":len(records)==len(paths),
        "all_matched_groups_pass":all(g["all_checks_pass"] for g in matched),
        "summary_rows":len(rows),"failure_record_count":len(failures),
        "pending_source_status_record_count":len(pending_status_records),
        "status_caveat":"Collector status files can lag result files. Historical queued/running entries are retained separately from explicit failures and marked when a complete result has since been collected.",
        "coverage_caveat":"Snapshot of result artifacts and executable plans at created_utc. Any pending, timeout, failed or uncollected jobs remain visible; missing values are never imputed. Final completeness is separately checked by ops/finalize_study.py.",
        "metric_caveat":"Fleet utilization uses all NPU wall time; role fractions use that role's admitted active time. Arrival SLO includes the deliberately saturated admission backlog.",
        "results":audits,"matched_input_groups":matched,"figures":figures}
    write_json(output/"matched_audit.json",artifact)
    print(json.dumps({k:artifact[k] for k in ("result_files_read","all_result_audits_pass","all_matched_groups_pass","summary_rows","failure_record_count")}),flush=True)
    if any(row.get("status")=="analysis_error" for row in failures):
        raise SystemExit(1)


if __name__=="__main__":
    main()
