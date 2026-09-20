#!/usr/bin/env python3
"""Audit a physical lower bound on cross-type request-boundary stalls.

Reads completed native results only; never runs or imports the simulator.
T_min=max(max_s V_s/40, V_total/50) is a lower bound in the pipelined
SSD-to-NPU model, not the exact end-to-end transfer time. A residual above
this bound is deliberately NOT classified as recoverable scheduling loss.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path


GIB_TO_GB = 2**30 / 1e9
TIME_TOL_MS = 1e-6


def overlap(start, end, window):
    return max(0.0, min(float(end), window[1])-max(float(start), window[0]))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_run(run_dir):
    """Return a flat JSON/CSV-safe summary for one completed run directory."""
    run_dir = Path(run_dir)
    config = load_json(run_dir/"config.json")
    metadata = {int(r["request_id"]): r for r in load_json(run_dir/"requests.json")}
    with gzip.open(run_dir/"native_summary.json.gz", "rt", encoding="utf-8") as stream:
        summary = json.load(stream)
    window = tuple(map(float, config.get("window_ms", [2000, 4000])))
    if not 0 <= window[0] < window[1] <= float(summary["makespan_ms"])+TIME_TOL_MS:
        raise ValueError("Warm window falls outside the completed run")
    by_npu = defaultdict(list)
    for batch in summary["microbatch_metrics"]:
        if len(batch["member_request_ids"]) != 1:
            raise ValueError("Boundary analysis requires batch_size=1")
        by_npu[int(batch["npu_id"])].append(batch)
    a, b = config["profile_A"], config["profile_B"]
    result = {
        "case": config["name"], "id": config["id"], "mode": config["mode"],
        "ssu": config["ssu"], "seed": config["seed"], "strategy": run_dir.name,
        "x": float(a["read_gib"])/float(b["read_gib"]),
        "y": float(a["compute_us"])/float(b["compute_us"]),
        "warm_start_ms": window[0], "warm_end_ms": window[1],
        "disk_GB_s": 40.0, "npu_receive_GB_s": 50.0,
        "cross_type_boundary_count_full": 0,
        "cross_type_bound_eligible_count_full": 0,
        "cross_type_late_submit_count_full": 0,
        "cross_type_admitted_count_warm": 0,
        "cross_type_positive_stall_count_warm": 0,
        "cross_type_positive_bound_count_warm": 0,
        "cross_type_l0_actual_stall_card_ms": 0.0,
        "cross_type_l0_physical_bound_card_ms": 0.0,
        "same_type_l0_actual_stall_card_ms": 0.0,
        "cold_start_l0_actual_stall_card_ms": 0.0,
        "internal_actual_stall_card_ms": 0.0,
        "all_l0_actual_stall_card_ms": 0.0,
        "bound_violation_count": 0,
        "maximum_bound_minus_actual_ms": 0.0,
    }
    for direction in ("A_to_B", "B_to_A"):
        result[direction+"_actual_stall_card_ms"] = 0.0
        result[direction+"_physical_bound_card_ms"] = 0.0
        result[direction+"_admitted_count_warm"] = 0
    for npu, batches in sorted(by_npu.items()):
        batches.sort(key=lambda batch: (float(batch["admission_time_ms"]), int(batch["batch_id"])))
        previous = None
        for batch in batches:
            rid = int(batch["member_request_ids"][0])
            meta = metadata[rid]
            role = str(meta["profile_group"])
            admission = float(batch["admission_time_ms"])
            layers = sorted(batch["layer_metrics"], key=lambda layer: int(layer["layer"]))
            layer0 = layers[0]
            if int(layer0["layer"]) != 0:
                raise ValueError("Missing layer-0 record")
            l0_compute_start = float(layer0["compute_start_ms"])
            io_start = float(layer0["io_start_time_ms"])
            actual = max(0.0, l0_compute_start-admission)
            if not math.isclose(actual, float(layer0["io_barrier_wait_ms"]), rel_tol=0.0, abs_tol=TIME_TOL_MS):
                raise AssertionError("Layer-0 stall accounting differs from admission-to-compute interval")
            actual_warm = overlap(admission, l0_compute_start, window)
            result["all_l0_actual_stall_card_ms"] += actual_warm
            for layer in layers[1:]:
                compute_start = float(layer["compute_start_ms"])
                wait = float(layer["io_barrier_wait_ms"])
                result["internal_actual_stall_card_ms"] += overlap(compute_start-wait, compute_start, window)
            if previous is None:
                result["cold_start_l0_actual_stall_card_ms"] += actual_warm
            else:
                previous_meta = metadata[int(previous["member_request_ids"][0])]
                previous_role = str(previous_meta["profile_group"])
                if previous_role == role:
                    result["same_type_l0_actual_stall_card_ms"] += actual_warm
                else:
                    if previous_role not in ("A", "B") or role not in ("A", "B"):
                        raise ValueError("Boundary analysis expects A/B profiles")
                    direction = previous_role+"_to_"+role
                    result["cross_type_boundary_count_full"] += 1
                    result["cross_type_l0_actual_stall_card_ms"] += actual_warm
                    result[direction+"_actual_stall_card_ms"] += actual_warm
                    admitted_warm = int(window[0] <= admission < window[1])
                    result["cross_type_admitted_count_warm"] += admitted_warm
                    result[direction+"_admitted_count_warm"] += admitted_warm
                    result["cross_type_positive_stall_count_warm"] += int(actual_warm > TIME_TOL_MS)
                    if io_start <= admission+TIME_TOL_MS:
                        result["cross_type_bound_eligible_count_full"] += 1
                        volumes_gb = [float(v)*GIB_TO_GB for v in meta["disk_gib"]]
                        v_total_gb = float(meta["per_layer_kv_gb"])*GIB_TO_GB
                        if not math.isclose(math.fsum(volumes_gb),v_total_gb,rel_tol=1e-10,abs_tol=1e-10):
                            raise AssertionError("Per-SSU volumes do not sum to total layer volume")
                        minimum_transfer_ms = 1000*max(max(volumes_gb)/40.0, v_total_gb/50.0)
                        overlap_window_ms = max(admission-io_start,0.0)
                        bound = max(minimum_transfer_ms-overlap_window_ms,0.0)
                        result["maximum_bound_minus_actual_ms"] = max(result["maximum_bound_minus_actual_ms"],bound-actual)
                        if bound > actual+TIME_TOL_MS:
                            result["bound_violation_count"] += 1
                            raise AssertionError(f"Physical lower bound exceeds actual stall: {run_dir.name}, NPU {npu}, rid {rid}, bound={bound}, actual={actual}")
                        bound_warm = overlap(admission, admission+bound, window)
                        result["cross_type_l0_physical_bound_card_ms"] += bound_warm
                        result[direction+"_physical_bound_card_ms"] += bound_warm
                        result["cross_type_positive_bound_count_warm"] += int(bound_warm > TIME_TOL_MS)
                    else:
                        result["cross_type_late_submit_count_full"] += 1
            previous = batch
    classified_l0 = math.fsum(result[key] for key in (
        "cross_type_l0_actual_stall_card_ms", "same_type_l0_actual_stall_card_ms", "cold_start_l0_actual_stall_card_ms"))
    result["l0_classification_error_card_ms"] = result["all_l0_actual_stall_card_ms"]-classified_l0
    if abs(result["l0_classification_error_card_ms"]) > TIME_TOL_MS:
        raise AssertionError("Layer-0 classes fail conservation")
    bound = result["cross_type_l0_physical_bound_card_ms"]
    actual = result["cross_type_l0_actual_stall_card_ms"]
    total = result["all_l0_actual_stall_card_ms"]+result["internal_actual_stall_card_ms"]
    result["total_actual_stall_card_ms"] = total
    result["cross_type_actual_minus_bound_card_ms_unclassified"] = actual-bound
    result["bound_fraction_of_cross_type_actual"] = bound/actual if actual > 0 else None
    result["bound_fraction_of_total_actual_stall"] = bound/total if total > 0 else None
    result["bound_checks_pass"] = result["bound_violation_count"] == 0
    result["interpretation"] = "Transfer lower bound only; actual minus bound is not necessarily recoverable by scheduling"
    # Independently cross-check against the separate whole-run audit module.
    analysis_path = run_dir/"analysis.json"
    if analysis_path.exists():
        audit = load_json(analysis_path)["warm"]
        l0_error = result["all_l0_actual_stall_card_ms"]-audit["stall"]["layer0"]["card_ms"]
        internal_error = result["internal_actual_stall_card_ms"]-audit["stall"]["internal"]["card_ms"]
        result["saved_audit_l0_error_card_ms"] = l0_error
        result["saved_audit_internal_error_card_ms"] = internal_error
        if max(abs(l0_error),abs(internal_error)) > TIME_TOL_MS:
            raise AssertionError("Boundary analysis disagrees with saved warm stall accounting")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    skipped = []
    for marker in sorted((root/"results").glob("*/*/analysis.json")):
        if marker.parent.name not in ("baseline", "once"):
            continue
        try:
            rows.append(analyze_run(marker.parent))
        except (OSError, EOFError, json.JSONDecodeError) as error:
            # A run can be concurrently finishing or being refreshed. Never
            # invent data from a partial gzip/JSON; include its path in status.
            skipped.append({"run":str(marker.parent),"reason":str(error)})
    output = root/"outputs"/"boundary_bounds.csv"
    output.parent.mkdir(parents=True,exist_ok=True)
    fields = list(rows[0]) if rows else []
    with output.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"completed_runs":len(rows),"output":str(output),
                      "all_lower_bound_checks_pass":all(r["bound_checks_pass"] for r in rows),
                      "late_cross_type_submissions":sum(r["cross_type_late_submit_count_full"] for r in rows),
                      "skipped_incomplete":skipped}))


if __name__ == "__main__":
    main()
