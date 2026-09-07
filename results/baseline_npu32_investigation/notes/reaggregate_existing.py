"""Recompute profile and window evidence from frozen results; no simulation."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_requests(rows):
    compute = sum(r["own_compute_ms"] for r in rows)
    active = sum(r["processing_latency_ms"] for r in rows)
    passed = sum(r["completion_time_ms"] - r["admission_time_ms"]
                 <= 1.5 * r["own_compute_ms"] + 1e-9 for r in rows)
    arrival_passed = sum(r["completion_time_ms"] - r["arrival_time_ms"]
                         <= 1.5 * r["own_compute_ms"] + 1e-9 for r in rows)
    return {
        "request_count": len(rows),
        "compute_ms": compute,
        "processing_ms": active,
        "aggregate_compute_fraction": compute / active,
        "mean_request_compute_fraction": sum(r["request_compute_fraction"] for r in rows) / len(rows),
        "min_request_compute_fraction": min(r["request_compute_fraction"] for r in rows),
        "mean_io_stall_ms": sum(r["io_stall_ms"] for r in rows) / len(rows),
        "mean_admission_wait_ms": sum(r["admission_wait_ms"] for r in rows) / len(rows),
        "admission_slo_passed": passed,
        "arrival_slo_passed": arrival_passed,
    }


def profile_keys(result):
    pairs = [(round(p["compute_ms"] * 8, 7), (p["seq_len_k"], p["nql"]))
             for p in result["metadata"]["profiles"]]
    assert len(dict(pairs)) == len(pairs), "compute duration must identify profiles uniquely"
    return dict(pairs)


def save_csv(path, rows):
    assert rows
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or Path(__file__).resolve().parent / "existing_audit_data"
    output.mkdir(parents=True, exist_ok=True)
    formal = root / "results/coflow_global_5ms_experiments/data/formal_results"
    multi = root / "results/multi_ssu_stall_experiments/data"
    paths = sorted(formal.glob("npu32_ssu*_arrival218_seed*.json"))
    paths += sorted(multi.glob("npu32_ssu*_baseline_none.json"))
    results, sources = {}, []
    profiles, short, windows, cards = [], [], [], []
    short_ids, inputs = {}, {}
    for path in paths:
        result = json.loads(path.read_text())
        results[path.name] = result
        metadata, summary = result["metadata"], result["summary"]
        kind = "coflow" if path.parent == formal else "multi_ssu"
        label = {
            "experiment": kind, "case": path.stem,
            "ssu": metadata["num_ssu"], "seed": metadata["seed"],
            "strategy": result["strategy"],
        }
        sources.append({"path": str(path.relative_to(root)), "sha256": digest(path),
                        "input_fingerprint": result["input_fingerprint"],
                        "core_and_policy_sha256": result["core_and_policy_sha256"]})
        pmap = profile_keys(result)
        rows = summary["request_metrics"]
        assert len(rows) == len({r["request_id"] for r in rows}) == metadata["request_count"]
        groups = defaultdict(list)
        for row in rows:
            groups[pmap[round(row["own_compute_ms"], 7)]].append(row)
        total = sum(r["own_compute_ms"] for r in rows)
        for (seq, nql), group in sorted(groups.items()):
            aggregated = aggregate_requests(group)
            profiles.append({**label, "seq_len_k": seq, "nql": nql,
                             "share_of_total_compute": aggregated["compute_ms"] / total,
                             **aggregated})
        short_rows = groups[32, 128] + groups[32, 256]
        cohort = tuple(sorted(r["request_id"] for r in short_rows))
        if kind == "coflow":
            pair = (metadata["num_ssu"], metadata["seed"])
            if pair in short_ids:
                assert short_ids[pair] == cohort
                assert inputs[pair] == result["input_fingerprint"]
            short_ids[pair] = cohort
            inputs[pair] = result["input_fingerprint"]
        short.append({**label, "window_utilization": result["common_window"]["mean_npu_utilization"],
                      "full_run_utilization": summary["fleet_npu_compute_utilization"],
                      "makespan_ms": summary["makespan_ms"], **aggregate_requests(short_rows)})
        if result["strategy"] != "baseline":
            continue
        u = result["common_window"]["npu_utilizations"]
        cards.append({**label, "mean_utilization": sum(u) / len(u),
                      "min_npu_id": u.index(min(u)), "min_utilization": min(u),
                      "cards_below_80pct": sum(x < .8 for x in u),
                      "full_run_utilization": summary["fleet_npu_compute_utilization"],
                      "mean_request_compute_fraction": summary["avg_request_compute_fraction"]})
        for start in (0, 1000, 2000, 3000):
            end = start + 1000
            active = [0.] * metadata["num_npu"]
            group_window = defaultdict(lambda: [0., 0.])
            for batch in summary["microbatch_metrics"]:
                key = pmap[round(batch["compute_busy_ms"], 7)]
                a = max(0., min(end, batch["completion_time_ms"])
                        - max(start, batch["admission_time_ms"]))
                c = sum(max(0., min(end, layer["compute_end_ms"])
                            - max(start, layer["compute_start_ms"]))
                        for layer in batch["layer_metrics"])
                active[batch["npu_id"]] += a
                group_window[key][0] += c
                group_window[key][1] += a
            for (seq, nql), (compute, processing) in sorted(group_window.items()):
                windows.append({**label, "start_ms": start, "end_ms": end,
                                "seq_len_k": seq, "nql": nql,
                                "compute_ms": compute, "active_ms": processing,
                                "share_of_fleet_time_compute": compute / (len(active) * 1000),
                                "share_of_fleet_time_active": processing / (len(active) * 1000),
                                "profile_compute_fraction": compute / processing if processing else None,
                                "all_cards_active": all(abs(a - 1000) < 1e-7 for a in active)})
    comparisons = []
    for ssu in (6, 7):
        for seed in (20260906, 20260907):
            a = results[f"npu32_ssu{ssu}_arrival218_seed{seed}_baseline.json"]
            b = results[f"npu32_ssu{ssu}_near_seed{seed}_baseline_none.json"]
            ar = {r["request_id"]: r for r in a["summary"]["request_metrics"]}
            br = {r["request_id"]: r for r in b["summary"]["request_metrics"]}
            fields = ("arrival_time_ms", "admission_time_ms", "completion_time_ms",
                      "own_compute_ms", "io_stall_ms")
            comparisons.append({
                "ssu": ssu, "seed": seed,
                "input_fingerprints_equal": a["input_fingerprint"] == b["input_fingerprint"],
                "microbatch_metrics_equal": a["summary"]["microbatch_metrics"] == b["summary"]["microbatch_metrics"],
                "differing_request_counts": {k: sum(ar[i][k] != br[i][k] for i in ar) for k in fields},
                "max_absolute_difference_ms": {k: max(abs(ar[i][k] - br[i][k]) for i in ar) for k in fields},
                "mean_arrival_latency_ms": {"coflow": a["summary"]["avg_request_latency_ms"],
                                            "multi_ssu": b["summary"]["avg_request_latency_ms"]},
            })
    for filename, rows in (("profile_metrics.csv", profiles), ("short_cohort_metrics.csv", short),
                           ("window_profile_metrics.csv", windows), ("baseline_card_metrics.csv", cards)):
        save_csv(output / filename, rows)
    artifact = {"schema_version": 1, "method": "read-only aggregation; no simulation",
                "profile_metric_denominator": "sum processing_latency_ms, excludes admission queue",
                "main_window_ms": [1000, 2000], "short_profiles": [[32, 128], [32, 256]],
                "admission_slo": "completion-admission <= 1.5*own_compute_ms + 1e-9 ms",
                "matched_short_cohort_request_ids": {f"ssu{s}_seed{seed}": ids
                                                      for (s, seed), ids in short_ids.items()},
                "source_result_files": sources, "cross_experiment_baseline_comparisons": comparisons,
                "profile_metrics": profiles, "short_cohort_metrics": short,
                "window_profile_metrics": windows, "baseline_card_metrics": cards}
    (output / "audit_metrics.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result_files_read": len(paths), "profile_rows": len(profiles),
                      "window_profile_rows": len(windows), "output": str(output)}))


if __name__ == "__main__":
    main()
