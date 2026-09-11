#!/usr/bin/env python3
"""Compare independent role audit values with every final report CSV/group."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-8)


def main():
    audit = read(HERE/"audit.json")
    assert audit["counts"]["result_statuses"] == {"passed": 40}
    assert all(x["all_study_conditions_met"] for x in audit["runs"])
    summary = read(BASE/"summary.json")
    assert summary["all_complete"] and summary["complete_count"] == 40
    csv_rows = {(int(x["seed"]), int(x["long_npu_count"]), x["order"], x["strategy"]): x
                for x in csv.DictReader((BASE/"per_seed.csv").open())}
    inputs = {x["label"]: x for x in audit["inputs"]}
    rows, comparisons, failures = [], [], []

    def check(actual, expected, context):
        ok = close(actual, expected)
        comparisons.append({"context": context, "matches": ok})
        if not ok:
            failures.append({"context": context, "independent": actual, "reported": expected})

    for run in audit["runs"]:
        key = (run["seed"], run["long_npus"], run["order"], run["strategy"])
        target = csv_rows[key]
        m, w, inp = run["metrics"], run["window"], inputs[run["label"]]
        v = {"device_utilization_percent": 100*m["device_utilization"],
             "window_request_equal_utilization_percent": 100*w["request_equal_utilization"],
             "makespan_ms": m["makespan_ms"],
             "full_run_device_utilization_percent": 100*inp["total_ideal_compute_ms"]/(32*m["makespan_ms"]),
             "full_population_binding_U_upper_bound_percent": 100*inp["full_run_device_U_upper_even_without_IO_wait"],
             "full_population_ideal_max_card_compute_ms": inp["max_per_npu_ideal_compute_ms"]}
        for name, prefix in (("window_admissions", "warm"), ("window_arrivals", "warm_arrivals"), ("all_requests", "full")):
            c = m["cohorts"][name]
            v[prefix+"_count"] = c["count"]
            v[prefix+"_completion_after_4000_count"] = c["completion_after_window_end_count"]
            for clock in ("admission", "arrival"):
                v[f"{prefix}_{clock}_passed"] = c[clock]["passed"]
                if c[clock]["rate"] is not None:
                    v[f"{prefix}_{clock}_slo_percent"] = 100*c[clock]["rate"]
            for role in ("short", "long"):
                rc = m["by_role"][role][name]
                v[f"{prefix}_{role}_count"] = rc["count"]
                v[f"{prefix}_{role}_admission_passed"] = rc["admission"]["passed"]
                if rc["admission"]["rate"] is not None:
                    v[f"{prefix}_{role}_admission_slo_percent"] = 100*rc["admission"]["rate"]
        for role in ("short", "long"):
            role_count = run[role+"_npus"]
            v[f"{role}_card_utilization_percent"] = 100*w["by_role"][role]["compute_ms"]/(2000*role_count)
            for metric in ("l0_exposed_stall_ms", "l1_7_exposed_stall_ms"):
                v[f"{role}_{metric}"] = w["by_role"][role][metric]
        v.update(max_fourth_completion_ms=max(run["warmup"]["fourth_completion_by_npu_ms"]),
                 full_run_max_ssu_gib_s=run["nominal_demand_scan"]["full_run"]["max_ssu_gib_s"],
                 full_run_any_ssu_over_capacity_ms=run["nominal_demand_scan"]["full_run"]["any_ssu_over_capacity_ms"],
                 static_max_ssu_gib_s=max(inp["exact_assigned_npu_static_per_ssu_gib_s"]))
        for metric, value in v.items():
            if metric in target:
                check(value, target[metric], [*key, metric])
        assert all(target[k] == "True" for k in ("audit_passed", "main_window_valid", "full_run_nominal_capacity_satisfied", "all_study_conditions_met"))
        rows.append({"key": key, "values": v})
    groups = []
    for group in summary["groups"]:
        key = (group["long_npu_count"], group["order"], group["strategy"])
        members = [r["values"] for r in rows if r["key"][1:] == key]
        assert len(members) == 5 and group["all_conditions_count"] == 5
        n_metrics = 0
        for metric, saved in group.items():
            if not isinstance(saved, dict) or "mean" not in saved or metric not in members[0]:
                continue
            values = [m[metric] for m in members]
            calculated = {"n": 5, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                          "min": min(values), "max": max(values)}
            for stat, expected in saved.items():
                check(calculated[stat], expected, [*key, metric, stat])
            n_metrics += 1
        groups.append({"key": key, "matched_metric_groups": n_metrics})
    assert len(groups) == 8
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__)),
               "inputs_sha256": {p.name: sha(p) for p in (HERE/"audit.json", BASE/"summary.json", BASE/"per_seed.csv")},
               "independent_complete_valid_results": 40, "independent_order_pairs": 10, "per_seed_rows_checked": len(rows),
               "aggregate_rows_checked": len(groups), "groups": groups, "numeric_comparisons": len(comparisons),
               "failed_checks": failures, "all_match": not failures,
               "full_run_U_note": "Independently recomputed from frozen total pure compute divided by 32*makespan; includes inevitable tail imbalance."}
    (HERE/"crosscheck_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(payload, ensure_ascii=False))
    assert not failures


if __name__ == "__main__":
    main()
