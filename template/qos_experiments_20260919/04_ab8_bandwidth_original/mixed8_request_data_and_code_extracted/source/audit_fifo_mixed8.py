#!/usr/bin/env python3
"""Independently audit an existing paired eight-NPU mixed-input experiment.

This reads frozen manifests and raw per-layer times. It never runs or changes
the simulator. Example:
  python audit_fifo_mixed8.py results/fifo_mixed_unique_20260914/formal_random_s1
The default output is INPUT_DIRECTORY/audit.json; the window is [2000,4000) ms.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parent


def read_json(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-7), message)


def audit_run(directory, window):
    result = read_json(directory / "result.json.gz")
    manifest = read_json(directory / "manifest.json.gz")
    metadata = manifest["metadata"]
    npu_count = metadata["num_npu"]
    require(npu_count == 8 and metadata["num_ssu"] == 1,
            "This audit's physical bounds apply to eight NPUs and one SSU.")
    require(metadata["n_layers"] == 8, "Expected eight layers.")
    require(window[0] < window[1], "Window must have positive duration.")
    clip = lambda a, b: max(0.0, min(window[1], b) - max(window[0], a))
    denominator = npu_count * (window[1] - window[0])
    requests = {q["request_id"]: q for q in manifest["requests"]}
    loads = {rid: q["load"] for rid, q in requests.items()}
    completed = {q["request_id"]: q for q in result["summary"]["request_metrics"]}
    require(set(requests) == set(completed), "Not all frozen requests completed.")
    rows = {n: dict(npu_id=n, compute_ms=0.0, active_ms=0.0, stall_ms=0.0,
                    roles={z: dict(compute_ms=0.0, active_ms=0.0, stall_ms=0.0,
                                   computed_ids=set()) for z in ("L", "S")})
            for n in range(npu_count)}
    batches, rates = {}, []
    layer_stalls = collections.defaultdict(float)
    demand_events = collections.defaultdict(float)
    for batch in result["summary"]["microbatch_metrics"]:
        require(batch["batch_size"] == 1 and len(batch["member_request_ids"]) == 1,
                "Only serial batch-size-one execution is supported.")
        rid = batch["member_request_ids"][0]
        require(rid not in batches, "Repeated batch membership.")
        batches[rid] = batch
        load, npu = loads[rid], batch["npu_id"]
        role = load["role"]
        dst, cls = rows[npu], rows[npu]["roles"][role]
        active = clip(batch["admission_time_ms"], batch["completion_time_ms"])
        dst["active_ms"] += active
        cls["active_ms"] += active
        placement = manifest["placements"][requests[rid]["placement_index"]]
        require(len(placement) == 1, "Expected one block map reused for every layer.")
        require(all(ssu == 0 for ssu, _ in placement[0]), "Unexpected SSU ID.")
        volume = math.fsum(size for _, size in placement[0])
        close(volume, load["per_layer_kv_gb"], "Placement bytes differ from profile.")
        rate = volume / (load["per_layer_us"] / 1e6)
        rates.append((npu, rid, rate))
        demand_events[batch["admission_time_ms"]] += rate
        demand_events[batch["completion_time_ms"]] -= rate
        previous_end = batch["admission_time_ms"]
        require(len(batch["layer_metrics"]) == 8, "Incomplete layer trace.")
        for index, layer in enumerate(batch["layer_metrics"]):
            require(layer["layer"] == index, "Layer order differs from expected order.")
            close(layer["compute_duration_ms"], load["per_layer_us"] / 1000,
                  "Layer compute differs from frozen profile.")
            compute = clip(layer["compute_start_ms"], layer["compute_end_ms"])
            stall = clip(previous_end, layer["compute_start_ms"])
            previous_end = layer["compute_end_ms"]
            dst["compute_ms"] += compute
            cls["compute_ms"] += compute
            dst["stall_ms"] += stall
            cls["stall_ms"] += stall
            layer_stalls[role + ("_L0" if index == 0 else "_L1_L7")] += stall
            if compute > 1e-9:
                cls["computed_ids"].add(rid)
    require(set(batches) == set(requests), "Incomplete batch membership.")
    static = sum(max(rate for n, _, rate in rates if n == npu)
                 for npu in range(npu_count))
    current = peak = over = previous_time = 0.0
    for time, change in sorted(demand_events.items()):
        if current > 40 + 1e-9:
            over += time - previous_time
        current += change
        peak = max(peak, current)
        previous_time = time
    total_compute = sum(row["compute_ms"] for row in rows.values())
    total_stall = sum(row["stall_ms"] for row in rows.values())
    close(total_compute + total_stall, denominator, "Window contains idle gaps.")
    per_npu = []
    for npu, row in rows.items():
        deck = [q for q in manifest["requests"] if q["npu_id"] == npu]
        pairs = {(q["load"]["total_tokens"], q["load"]["nql"]) for q in deck}
        require(len(pairs) == len(deck), f"NPU {npu} repeats a length/NQL pair.")
        close(row["active_ms"], window[1] - window[0], "NPU inactive within window.")
        row["U_percent"] = 100 * row["compute_ms"] / (window[1] - window[0])
        row["input_requests"] = len(deck)
        row["unique_length_nql_pairs"] = len(pairs)
        for cls in row["roles"].values():
            cls["computed_request_count"] = len(cls.pop("computed_ids"))
            require(cls["computed_request_count"] > 0,
                    f"NPU {npu} did not compute both classes in the window.")
            cls["U_percent"] = 100 * cls["compute_ms"] / cls["active_ms"]
        per_npu.append(row)
    cohort = [q for q in completed.values()
              if window[0] <= q["admission_time_ms"] < window[1]]
    passed = sum(q["completion_time_ms"] - q["admission_time_ms"] <=
                 1.5 * 8 * loads[q["request_id"]]["per_layer_us"] / 1000 + 1e-8
                 for q in cohort)
    transfers = []
    for npu in range(npu_count):
        ordered = sorted((b for b in batches.values() if b["npu_id"] == npu),
                         key=lambda b: b["admission_time_ms"])
        for previous, following in zip(ordered, ordered[1:]):
            ida, idb = previous["member_request_ids"][0], following["member_request_ids"][0]
            if (loads[ida]["role"], loads[idb]["role"]) != ("S", "L"):
                continue
            volume = loads[idb]["per_layer_kv_gb"]
            compute = previous["layer_metrics"][-1]["compute_duration_ms"]
            ssd_bound = max(0.0, volume / 40 * 1000 - compute)
            link_bound = max(0.0, volume / 50 * 1000 - compute)
            admission = following["admission_time_ms"]
            actual = following["layer_metrics"][0]["compute_start_ms"] - admission
            require(actual + 1e-7 >= ssd_bound, "Physical S-to-L lower bound violated.")
            transfers.append(dict(npu_id=npu, previous_request_id=ida, long_request_id=idb,
                admission_ms=admission, previous_compute_ms=compute, L_bytes_gib=volume,
                link50_lower_bound_ms=link_bound, ssd40_lower_bound_ms=ssd_bound,
                actual_L0_stall_ms=actual, window_clipped_actual_ms=clip(admission, admission + actual),
                window_clipped_ssd_lower_bound_ms=clip(admission, admission + ssd_bound),
                admitted_in_window=window[0] <= admission < window[1]))
    window_transfers = [t for t in transfers if t["admitted_in_window"]]
    transfer_range = lambda key: ([min(t[key] for t in window_transfers),
                                   max(t[key] for t in window_transfers)]
                                  if window_transfers else None)
    roles = {z: {k: sum(row["roles"][z][k] for row in per_npu)
                 for k in ("active_ms", "compute_ms", "stall_ms")} for z in ("L", "S")}
    for role in roles.values():
        role["U_percent"] = 100 * role["compute_ms"] / role["active_ms"]
    audit = dict(input_request_count=len(requests), completed_request_count=len(completed),
        U_percent=100 * total_compute / denominator, compute_card_ms=total_compute,
        stall_card_ms=total_stall, by_role=roles, by_npu=per_npu,
        all_8_npus_compute_both_roles=True, all_input_pairs_unique_per_npu=True,
        static_worst_combination_GiB_s=static, whole_run_actual_nominal_peak_GiB_s=peak,
        whole_run_over40_ms=over,
        SLO_1p5_admission_cohort=dict(passed=passed, count=len(cohort), percent=100 * passed / len(cohort)),
        window_stall_by_role_and_layer_card_ms=dict(layer_stalls),
        S_to_L=dict(whole_run_transition_count=len(transfers),
            window_admitted_transition_count=len(window_transfers),
            window_ssd40_lower_bound_sum_card_ms=sum(t["window_clipped_ssd_lower_bound_ms"] for t in transfers),
            window_link50_lower_bound_sum_card_ms=sum(t["link50_lower_bound_ms"] for t in window_transfers),
            window_actual_stall_card_ms=sum(t["window_clipped_actual_ms"] for t in transfers),
            window_admitted_ssd40_lower_bound_range_ms=transfer_range("ssd40_lower_bound_ms"),
            window_admitted_actual_L0_stall_range_ms=transfer_range("actual_L0_stall_ms"),
            transitions=transfers))
    stored_window = next(w for w in result["windows"]
                         if w["start_ms"] == window[0] and w["end_ms"] == window[1])
    close(audit["U_percent"], 100 * stored_window["mean_npu_utilization"], "Stored utilization differs.")
    close(static, metadata["per_ssu_static_upper_bound_gib_s"][0], "Stored static bound differs.")
    stored_slo = result["slo"]["window_admissions"]["admission"]
    require(passed == stored_slo["passed"] and len(cohort) == stored_slo["count"],
            "Stored SLO admission cohort differs.")
    return audit, result, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path, nargs="?",
                        default=SOURCE / "results/fifo_mixed_unique_20260914/formal_random_s1")
    parser.add_argument("--window", type=float, nargs=2, default=[2000.0, 4000.0], metavar=("START_MS", "END_MS"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.input_directory.resolve()
    output = args.output or root / "audit.json"
    document = dict(method="Independent raw manifest plus microbatch layer interval audit; does not reuse stored window aggregates.",
                    window_ms=args.window, policies={})
    raw = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "result.json.gz").exists()):
        audit, result, manifest = audit_run(directory, args.window)
        policy = result["probe_policy"]
        require(policy in ("fifo", "short_first") and policy not in raw, "Expected one run per policy.")
        document["policies"][policy] = audit
        raw[policy] = result, manifest
    require(set(raw) == {"fifo", "short_first"}, "Missing paired FIFO/short_first runs.")
    fifo, fifo_manifest = raw["fifo"]
    short, short_manifest = raw["short_first"]
    checks = dict(requests_exactly_identical=fifo_manifest["requests"] == short_manifest["requests"],
        placements_exactly_identical=fifo_manifest["placements"] == short_manifest["placements"],
        input_fingerprint_equal=fifo["input_fingerprint"] == short["input_fingerprint"],
        logical_input_fingerprint_equal=fifo["logical_input_fingerprint"] == short["logical_input_fingerprint"],
        execution_placement_fingerprint_equal=fifo["execution_placement_fingerprint"] == short["execution_placement_fingerprint"],
        core_and_policy_sha256_equal=fifo["core_and_policy_sha256"] == short["core_and_policy_sha256"],
        current_core_files_match_recorded_hashes=all(
            hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
            for name, digest in fifo["core_and_policy_sha256"].items()))
    require(all(checks.values()), f"Paired validation failed: {checks}")
    document["paired_checks"] = checks
    a, b = document["policies"]["fifo"], document["policies"]["short_first"]
    document["window_difference"] = dict(fleet_improvement_pp=b["U_percent"] - a["U_percent"],
        short_stall_saved_card_ms=a["by_role"]["S"]["stall_ms"] - b["by_role"]["S"]["stall_ms"],
        long_stall_added_card_ms=b["by_role"]["L"]["stall_ms"] - a["by_role"]["L"]["stall_ms"],
        net_stall_saved_card_ms=a["stall_card_ms"] - b["stall_card_ms"])
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(output), paired_checks=checks,
                          window_difference=document["window_difference"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
