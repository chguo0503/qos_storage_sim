"""Independent no-hook OD replay; vary only the SSD outstanding queue cap.

This runner deliberately makes no phase-lock/target-E assumption.  Its output
is compared with the capped replay after both simulations have completed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "transition"))
import metrics
from inputs.manifest import load_manifest, write_json
from simulator.api import run_simulation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    manifest = HERE / "inputs" / f"{args.name}.json.gz"
    requests, metadata = load_manifest(manifest)
    output = HERE / "formal" / f"{args.name}_od_depth_none"
    output.mkdir(parents=True, exist_ok=False)
    planning = HERE / "planning" / args.name
    sources = list((ROOT / "simulator").rglob("*.py")) + [
        ROOT / "data", ROOT / "inputs/manifest.py",
        HERE.parent / "transition/metrics.py", Path(__file__), manifest,
        planning / "generator_source.py", planning / "command.json",
        planning / "planning_records.json", planning / "PLANNING_ONLY_result.json.gz",
    ]
    before = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    planned = json.loads((planning / "command.json").read_text())
    assert planned["status"] == "complete"
    assert planned["manifest_sha256"] == digest(manifest)
    assert metadata["data_sha256"] == digest(ROOT / "data")
    assert all(before[key] == value for key, value in metadata["planning_source_sha256"].items())
    assert all(math.isfinite(q.load["per_layer_us"]) and q.load["per_layer_us"] > 0 for q in requests)
    command = dict(
        status="running", name=args.name, strategy="od_baseline", seed=7,
        runtime_hooks=False, od_queue_depth_per_ssu=None,
        depth_per_npu_per_ssu=None, manifest_sha256=digest(manifest),
        source_and_artifact_sha256=before,
        only_intended_change="OD SSD queue depth: 8192 per disk -> unlimited",
        phase_target_assertions=False,
    )
    write_json(output / "command.json", command)
    started = time.perf_counter()
    result = run_simulation(
        requests, strategy="od_baseline", num_npu=32, num_ssu=3,
        n_layers=8, seed=7, od_queue_depth_per_ssu=None,
    )
    assert before == {str(path.relative_to(ROOT)): digest(path) for path in sources}
    write_json(output / "result.json.gz", result)
    raw = result["summary"]
    by_id = {q.request_id: q for q in requests}
    batches = {
        (b["npu_id"], by_id[b["member_request_ids"][0]].load["cycle"],
         by_id[b["member_request_ids"][0]].load["position"]): b
        for b in raw["microbatch_metrics"]
    }
    cycles = []
    for cycle in range(metadata["cycles"]):
        tails = [batches[n, cycle, 2] for n in range(32)]
        starts = [b["layer_metrics"][0]["compute_start_ms"] for b in tails]
        ends = [b["completion_time_ms"] for b in tails]
        waits = [layer["io_barrier_wait_ms"] for n in range(32)
                 for position in (1, 2)
                 for layer in batches[n, cycle, position]["layer_metrics"][1:]]
        row = dict(cycle=cycle, start_spread_ms=max(starts)-min(starts),
                   end_spread_ms=max(ends)-min(ends),
                   max_E_error_ms=max(abs(end-(cycle+1)*metadata["period_ms"]) for end in ends),
                   B_internal_stall_card_ms=sum(waits), B_internal_stall_max_ms=max(waits))
        if cycle+1 < metadata["cycles"]:
            row["next_A0_min_prefetch_margin_ms"] = min(
                ends[n]-batches[n, cycle+1, 0]["layer_metrics"][0]["io_ready_time_ms"]
                for n in range(32))
        cycles.append(row)
    windows = []
    for left, right in [(2000., 4000.), (8000., 12000.), (12000., 20000.),
                        (20000., 40000.), (40000., 60000.), (20000., 60000.),
                        (60000., 78000.), (0., raw["makespan_ms"])]:
        if right <= raw["makespan_ms"]+1e-7:
            row = metrics.summarize(raw, requests, left, right)
            row["measurement_definition"]["own_compute"] = "8 times frozen request C; identical to capped OD and Once input"
            windows.append(row)
    analysis = dict(
        cycles=cycles, windows=windows, makespan_ms=raw["makespan_ms"],
        first_card_drains_ms=min(max(m["completion_time_ms"] for m in raw["request_metrics"]
                                    if m["npu_id"] == n) for n in range(32)),
        all_native_invariants=all(raw["invariants"].values()),
        synthetic_OD_tailored_input=True, runtime_hooks=False,
        depth_per_npu_per_ssu=None,
    )
    assert "ssd_queue_depth" not in raw
    write_json(output / "analysis.json", analysis)
    (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    command.update(status="complete", source_and_artifacts_unchanged=True,
                   wall_seconds=time.perf_counter()-started)
    write_json(output / "command.json", command)
    print(json.dumps(dict(status="complete", output=str(output),
                         wall_seconds=command["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
