"""Compare only completed clean OD/Once replays of the same frozen manifest."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "transition"))
import metrics
from inputs.manifest import load_manifest, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brief(raw, requests, left, right):
    measured = metrics.summarize(raw, requests, left, right)
    demand = measured["demand"]
    return dict(
        start_ms=left, end_ms=right, U_percent=measured["U_percent"],
        admission_slo_1p5=measured["slo"], arrival_slo_1p5=measured["arrival_slo"],
        all_32_npus_active=measured["all_npus_active"],
        cards_computing_both_roles=measured["role_and_stall"]["npus_with_A_and_B_compute"],
        by_category={key: {"active_U_percent": value["active_U_percent"],
                           "slo_1p5": value["admission"]["slo"]["1.5"],
                           "normalized_latency": value["admission"]["normalized_latency"]}
                     for key, value in measured["by_category"].items()},
        demand={key: demand[key] for key in (
            "per_disk_mean_GiB_s", "per_disk_min_GiB_s", "per_disk_max_GiB_s",
            "per_disk_overload_percent", "all_disks_overload_percent",
            "any_disk_overload_percent", "strict_underload_all_disks")},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="abb_interp_unique_50")
    args = parser.parse_args()
    manifest = HERE / "inputs" / f"{args.name}.json.gz"
    requests, metadata = load_manifest(manifest)
    manifest_sha = sha(manifest)
    expected = {q.request_id: q for q in requests}
    runs = {}
    commands = {}
    details = {}
    for strategy in ("od_baseline", "once"):
        directory = HERE / "formal" / f"{args.name}_{strategy}"
        command = json.loads((directory / "command.json").read_text())
        assert command["status"] == "complete" and command["runtime_hooks"] is False
        assert command["source_and_artifacts_unchanged"]
        assert command["manifest_sha256"] == manifest_sha
        with gzip.open(directory / "result.json.gz", "rt") as stream:
            raw = json.load(stream)["summary"]
        population = {q["request_id"]: q for q in raw["request_metrics"]}
        assert len(population) == len(requests) == len(raw["request_metrics"])
        assert population.keys() == expected.keys()
        assert raw["request_count"] == len(requests)
        assert raw["num_npu"] == 32 and raw["num_ssu"] == 3 and raw["n_layers"] == 8
        assert all(raw["invariants"].values())
        for request_id, q in expected.items():
            row = population[request_id]
            assert row["npu_id"] == q.npu_id
            assert row["arrival_time_ms"] == q.arrival_time_ms
            assert abs(row["own_compute_ms"]-8*q.load["per_layer_us"]/1000) < 1e-9
            assert row["io_count"] == 8*len(q.placement[0])
        expected_blocks = sum(8*len(q.placement[0]) for q in requests)
        assert raw["submitted_blocks"] == raw["completed_blocks"] == expected_blocks
        own_compute = math.fsum(r["own_compute_ms"] for r in population.values())
        first_drain = min(max(r["completion_time_ms"] for r in population.values()
                              if r["npu_id"] == n) for n in range(32))
        assert first_drain > 60000
        details[strategy] = dict(
            request_count=len(population), completed_blocks=raw["completed_blocks"],
            total_own_compute_card_ms=own_compute,
            makespan_ms=raw["makespan_ms"], first_card_drains_ms=first_drain,
            source_and_artifact_sha256=command["source_and_artifact_sha256"],
            result_sha256=sha(directory / "result.json.gz"),
            disk_stats_full_input=raw["disk_stats"],
            full_input=brief(raw, requests, 0., raw["makespan_ms"]),
        )
        runs[strategy] = raw
        commands[strategy] = command
    shared = set(commands["od_baseline"]["source_and_artifact_sha256"]) & set(commands["once"]["source_and_artifact_sha256"])
    assert all(commands["od_baseline"]["source_and_artifact_sha256"][key]
               == commands["once"]["source_and_artifact_sha256"][key] for key in shared)
    assert details["od_baseline"]["total_own_compute_card_ms"] == details["once"]["total_own_compute_card_ms"]
    windows = []
    csv_rows = []
    for left, right in ((2000., 4000.), (8000., 12000.), (12000., 20000.),
                        (20000., 40000.), (40000., 60000.), (20000., 60000.)):
        row = dict(start_ms=left, end_ms=right)
        for strategy, raw in runs.items():
            row[strategy] = brief(raw, requests, left, right)
            assert row[strategy]["all_32_npus_active"]
            assert row[strategy]["cards_computing_both_roles"] == 32
            m = row[strategy]
            csv_rows.append(dict(window=f"[{left/1000:g},{right/1000:g})", strategy=strategy,
                                 U_percent=m["U_percent"], SLO_1p5_percent=m["admission_slo_1p5"]["percent"],
                                 cohort_count=m["admission_slo_1p5"]["count"],
                                 passed=m["admission_slo_1p5"]["passed"], all32active=True, mixed_cards=32))
        row["Once_minus_OD_U_percentage_points"] = row["once"]["U_percent"]-row["od_baseline"]["U_percent"]
        windows.append(row)
    output = dict(
        name=args.name, manifest_sha256=manifest_sha,
        same_frozen_input_exact=True, identical_request_population_and_compute=True,
        shared_artifact_hashes_exact=True, both_clean_original_API_no_hooks=True,
        baseline_queue_depth_per_disk=8192, baseline_queue_depth_per_npu=256,
        once_queue_depth=None, capacity_per_disk_GiB_s=40., npu_count=32, ssu_count=3,
        input_mode=metadata["input_mode"], cycles=metadata["cycles"],
        scenarios=details, windows=windows,
        definitions={
            "U": "sum compute time intersecting window / (32 * window duration)",
            "SLO_1p5": "admission cohort in window; uncensored (completion-admission) <= 1.5*own frozen 8-layer compute",
            "full_input": "same 4800 requests; each strategy uses its own 0-to-final-completion interval",
            "demand": "current admitted request per-disk V/C, evaluated at all request-switch intervals; not actual SSD service",
        },
        cautions=[
            "Data-interpolated OD-tailored offline input; not raw measured rows, natural random input, or strict continuous underload.",
            "The input is static during both formal replays; planning output is not used as a measured result.",
            "Window admission cohorts differ when strategies progress at different speeds; full request population is identical.",
            "Once comparison includes original category CIR/path pools, not routing alone. Separate depth ablation shows cap does not change this input's actual request/layer times.",
        ],
    )
    write_json(HERE / "paired_formal_results.json", output)
    with (HERE / "paired_formal_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"manifest_sha256": manifest_sha, "windows": csv_rows,
                      "full": {k: {"U_percent": v["full_input"]["U_percent"],
                                    "SLO_1p5": v["full_input"]["admission_slo_1p5"],
                                    "first_card_drains_ms": v["first_card_drains_ms"]}
                               for k, v in details.items()}}, indent=2))


if __name__ == "__main__":
    main()
