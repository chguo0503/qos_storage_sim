#!/usr/bin/env python3
"""Compare frozen original results with tripled queues, without simulation.

The primary bridge is logged request/layer milestones in [0,4000) and clipped
[2000,4000) compute/occupancy/stall. It does not compare individual SSD block
events or the entire simulator event stream. Full cycle-0 equality is diagnostic
only: new cycle-1 prefetch may affect another lane's late cycle-0 requests.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = HERE.parent
OUT = HERE / "repeated_queues"
ROOT = F.parents[2]
START, END = 2000.0, 4000.0


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_raw(directory):
    paths = [p for p in Path(directory).glob("*.json.gz") if ".failure." not in p.name]
    assert len(paths) == 1, (directory, paths)
    return paths[0]


def clip(a, b):
    return max(0.0, min(END, b) - max(START, a))


def events(summary):
    """Keep logged request/layer milestones inside the common prefix."""
    rows = {}
    for batch in summary["microbatch_metrics"]:
        assert len(batch["member_request_ids"]) == 1
        rid = batch["member_request_ids"][0]
        npu = batch["npu_id"]
        for key in ("admission_time_ms", "completion_time_ms"):
            if 0 <= batch[key] < END:
                rows[f"{rid}/{npu}/batch/{key}"] = batch[key]
        for layer in batch["layer_metrics"]:
            for key in ("io_start_time_ms", "io_ready_time_ms", "compute_start_ms", "compute_end_ms"):
                if 0 <= layer[key] < END:
                    rows[f"{rid}/{npu}/layer{layer['layer']}/{key}"] = layer[key]
    return rows


def clipped(summary):
    lanes = [dict(compute=[], occupied=[], stall=[]) for _ in range(32)]
    for batch in summary["microbatch_metrics"]:
        lane = lanes[batch["npu_id"]]
        admission, completion = batch["admission_time_ms"], batch["completion_time_ms"]
        lane["occupied"].append(clip(admission, completion))
        prior = admission
        for layer in sorted(batch["layer_metrics"], key=lambda x: x["layer"]):
            start, end = layer["compute_start_ms"], layer["compute_end_ms"]
            assert start >= prior - 1e-8
            lane["compute"].append(clip(start, end))
            lane["stall"].append(clip(prior, start))
            prior = end
        assert math.isclose(prior, completion, abs_tol=1e-8)
    totals = [{k: math.fsum(v) for k, v in row.items()} for row in lanes]
    for row in totals:
        assert math.isclose(row["compute"] + row["stall"], row["occupied"], abs_tol=1e-8)
    return dict(per_npu=totals, device_U_percent=100 * math.fsum(x["compute"] for x in totals) / (32*(END-START)))


def compare_maps(a, b):
    keys = sorted(set(a) | set(b))
    different = [k for k in keys if a.get(k) != b.get(k)]
    numerical = [abs(a[k]-b[k]) for k in keys if k in a and k in b]
    return dict(exact_equal=not different, original_count=len(a), repeated_count=len(b),
        differing_count=len(different), missing_from_original=sum(k not in a for k in keys),
        missing_from_repeated=sum(k not in b for k in keys),
        max_absolute_difference_ms=max(numerical, default=0.0),
        first_differences=[dict(event=k, original=a.get(k), repeated=b.get(k)) for k in different[:20]])


def compare(job):
    item, policy = job["input"], job["strategy"]
    directory = OUT / "runs" / item["label"] / policy
    command_path = directory / "command.json"
    command = read(command_path) if command_path.exists() else {}
    if command.get("status") != "complete":
        return dict(label=item["label"], mode=item["mode"], strategy=policy,
                    status=command.get("status", "pending"))
    assert command["returncode"] == 0
    original_label = Path(item["base_manifest"]).name.removesuffix(".json.gz")
    original_dir = (F if item["mode"] == "fixed" else F/"mixed_rebinding") / "runs" / original_label / policy
    original_path, repeated_path = one_raw(original_dir), one_raw(directory)
    original, repeated = read(original_path), read(repeated_path)
    source_manifest, new_manifest = read(item["base_manifest"]), read(item["manifest"])
    assert sha(item["base_manifest"]) == item["base_manifest_sha256"]
    assert sha(item["manifest"]) == item["manifest_sha256"]
    assert original["input_fingerprint"] == source_manifest["input_fingerprint"]
    assert repeated["input_fingerprint"] == new_manifest["input_fingerprint"] == item["input_fingerprint"]
    assert repeated["strategy"] == original["strategy"] == policy
    assert repeated["submit_seed"] == original["submit_seed"] == 7
    assert repeated["python_version"] == original["python_version"]
    assert repeated["core_and_policy_sha256"] == original["core_and_policy_sha256"]
    assert repeated["stress_runner_sha256"] == original["stress_runner_sha256"]
    assert repeated["policy_config"] == original["policy_config"]
    for name, digest in repeated["core_and_policy_sha256"].items():
        assert sha(ROOT/name) == digest
    source_ids = {r["request_id"] for r in source_manifest["requests"]}
    cycle0 = [r for r in new_manifest["requests"] if r["load"]["source_cycle"] == 0]
    assert source_ids == {r["request_id"] for r in cycle0}
    assert all(r["request_id"] == r["load"]["base_request_id"] for r in cycle0)
    a, b = original["summary"], repeated["summary"]
    assert a["request_count"] == 1212 and b["request_count"] == 3636
    assert all(a["invariants"].values()) and all(b["invariants"].values())
    prefix = compare_maps(events(a), events(b))
    original_clip, repeated_clip = clipped(a), clipped(b)
    clip_differences = [abs(x[k]-y[k]) for x,y in zip(original_clip["per_npu"], repeated_clip["per_npu"])
                        for k in ("compute", "occupied", "stall")]
    # Deliberately diagnostic only; later cycle-0 interference is possible.
    old_requests = {r["request_id"]: r for r in a["request_metrics"]}
    new_requests = {r["request_id"]: r for r in b["request_metrics"] if r["request_id"] in source_ids}
    old_batches = {r["member_request_ids"][0]: r for r in a["microbatch_metrics"]}
    new_batches = {r["member_request_ids"][0]: r for r in b["microbatch_metrics"] if r["member_request_ids"][0] in source_ids}
    old_window = next(w for w in original["windows"] if (w["start_ms"],w["end_ms"])==(START,END))
    new_window = next(w for w in repeated["windows"] if (w["start_ms"],w["end_ms"])==(START,END))
    assert math.isclose(original_clip["device_U_percent"],100*old_window["mean_npu_utilization"],abs_tol=1e-9)
    assert math.isclose(repeated_clip["device_U_percent"],100*new_window["mean_npu_utilization"],abs_tol=1e-9)
    return dict(label=item["label"],mode=item["mode"],strategy=policy,status="complete",
        original_result=str(original_path),original_sha256=sha(original_path),
        repeated_result=str(repeated_path),repeated_sha256=sha(repeated_path),
        original_input_sha256=sha(item["base_manifest"]),repeated_input_sha256=sha(item["manifest"]),
        cycle0_request_id_preserved=True,source_policy_python_equal=True,
        event_prefix_0_4000=prefix,clipped_2000_4000_exact_equal=original_clip==repeated_clip,
        clipped_max_absolute_difference_ms=max(clip_differences),
        original_clipped=original_clip,repeated_clipped=repeated_clip,
        device_U_absolute_difference_pp=abs(original_clip["device_U_percent"]-repeated_clip["device_U_percent"]),
        full_cycle0_diagnostic=dict(request_rows_exact_equal=old_requests==new_requests,
            request_different_count=sum(old_requests[k]!=new_requests[k] for k in source_ids),
            microbatch_rows_exact_equal=old_batches==new_batches,
            microbatch_different_count=sum(old_batches[k]!=new_batches[k] for k in source_ids)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    plan_path=OUT/"plan.json"
    plan=read(plan_path)
    rows=[compare(job) for job in plan["jobs"]]
    complete=[r for r in rows if r["status"]=="complete"]
    result=dict(script_sha256=sha(__file__),plan_sha256=sha(plan_path),complete=len(complete),planned=6,
        all_complete=len(complete)==6,
        all_completed_prefix_events_exact=all(r["event_prefix_0_4000"]["exact_equal"] for r in complete) if complete else None,
        all_completed_clipped_windows_exact=all(r["clipped_2000_4000_exact_equal"] for r in complete) if complete else None,
        scope="Primary comparison only uses logged request/layer milestones in [0,4000), not individual SSD block events or the entire simulator event stream; compute/occupancy/stall is clipped to [2000,4000). Full cycle-0 rows are diagnostic and may differ after new-cycle prefetch begins; they are not a required equivalence condition. --require-complete checks availability only; the two explicit exact-equality flags state the observed bridge result without suppressing differences.",rows=rows)
    (OUT/"prefix_bridge.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False))
    if args.require_complete:
        assert len(complete)==6


if __name__ == "__main__":
    main()
