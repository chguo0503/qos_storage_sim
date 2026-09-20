"""Read-only comparison of the completed capped and uncapped OD runs."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "transition"))
import metrics
from inputs.manifest import load_manifest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="abb_interp_unique_50")
    args = parser.parse_args()
    source = HERE / "inputs" / f"{args.name}.json.gz"
    manifest_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    requests, _ = load_manifest(source)
    runs = {}
    source_hashes = {}
    artifact_hashes = {}
    for label, suffix in (("depth256", "od_baseline"), ("depthNone", "od_depth_none")):
        directory = HERE / "formal" / f"{args.name}_{suffix}"
        command = json.loads((directory / "command.json").read_text())
        assert command["status"] == "complete"
        assert command["runtime_hooks"] is False
        assert command["source_and_artifacts_unchanged"]
        assert command["manifest_sha256"] == manifest_sha
        source_hashes[label] = {k: v for k, v in command["source_and_artifact_sha256"].items()
                               if k.startswith("simulator/")}
        artifact_hashes[label] = command["source_and_artifact_sha256"]
        with gzip.open(directory / "result.json.gz", "rt") as stream:
            runs[label] = json.load(stream)["summary"]
    assert source_hashes["depth256"] == source_hashes["depthNone"]
    common_artifacts = set(artifact_hashes["depth256"]) & set(artifact_hashes["depthNone"])
    assert all(artifact_hashes["depth256"][key] == artifact_hashes["depthNone"][key]
               for key in common_artifacts)
    capped, unlimited = runs["depth256"], runs["depthNone"]
    request_maps = [{r["request_id"]: r for r in raw["request_metrics"]} for raw in (capped, unlimited)]
    assert set(request_maps[0]) == set(request_maps[1])
    layer_maps = [{(b["npu_id"], tuple(b["member_request_ids"]), layer["layer"]): layer
                   for b in raw["microbatch_metrics"] for layer in b["layer_metrics"]}
                  for raw in (capped, unlimited)]
    assert set(layer_maps[0]) == set(layer_maps[1])
    layer_fields = ("compute_start_ms", "compute_end_ms", "compute_duration_ms",
                    "io_start_time_ms", "io_ready_time_ms", "io_barrier_wait_ms")
    differences = {
        field: {"max_absolute_ms": max(abs(layer_maps[0][k][field]-layer_maps[1][k][field])
                                     for k in layer_maps[0]),
                "nonidentical_layers": sum(layer_maps[0][k][field] != layer_maps[1][k][field]
                                           for k in layer_maps[0])}
        for field in layer_fields
    }
    output = dict(
        name=args.name, manifest_sha256=manifest_sha,
        identical_manifest_and_core=True, both_formal_no_runtime_hooks=True,
        request_count=len(request_maps[0]), layer_count=len(layer_maps[0]),
        all_request_metrics_exact=capped["request_metrics"] == unlimited["request_metrics"],
        all_microbatch_metrics_exact=capped["microbatch_metrics"] == unlimited["microbatch_metrics"],
        nonidentical_request_records=sum(request_maps[0][k] != request_maps[1][k] for k in request_maps[0]),
        changed_request_fields=dict(Counter(
            field for rid in request_maps[0]
            for field in request_maps[0][rid]
            if request_maps[0][rid][field] != request_maps[1][rid][field])),
        request_time_differences={
            field: {"max_absolute_ms": max(abs(request_maps[0][k][field]-request_maps[1][k][field])
                                            for k in request_maps[0]),
                    "nonidentical_requests": sum(request_maps[0][k][field] != request_maps[1][k][field]
                                                for k in request_maps[0])}
            for field in ("arrival_time_ms", "admission_time_ms", "completion_time_ms", "latency_ms",
                          "processing_latency_ms", "io_stall_ms", "own_compute_ms")
        },
        layer_field_differences=differences,
        common_artifact_hashes_identical=True,
        depth256_summary=capped.get("ssd_queue_depth"),
        depthNone_has_depth_summary="ssd_queue_depth" in unlimited,
        windows=[],
        interpretation="Only OD queue depth changed. Equal request/layer times would exclude the queue cap as the cause in this fixed input, not in other inputs. Full request records can differ merely because wait moves between host and SSD queues; such record inequality alone is not a performance change.",
    )
    for left, right in ((2000., 4000.), (20000., 40000.), (40000., 60000.), (20000., 60000.)):
        row = dict(start_ms=left, end_ms=right)
        for label, raw in runs.items():
            m = metrics.summarize(raw, requests, left, right)
            row[label] = dict(
                U_percent=m["U_percent"], slo=m["slo"],
                all_npus_active=m["all_npus_active"],
                mixed_cards=m["role_and_stall"]["npus_with_A_and_B_compute"],
            )
        output["windows"].append(row)
    output["full"] = {}
    for label, raw in runs.items():
        m = metrics.summarize(raw, requests, 0., raw["makespan_ms"])
        output["full"][label] = dict(makespan_ms=raw["makespan_ms"], U_percent=m["U_percent"], slo=m["slo"])
    write_json(HERE / "depth_comparison.json", output)
    print(json.dumps({k: v for k, v in output.items() if k != "depth256_summary"}, indent=2))


if __name__ == "__main__":
    main()
