#!/usr/bin/env python3
"""Export both frozen v8 seed7 queues, without simulation or input mutation."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from run_baseline_npu32_stress import load_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    profiles = {(32, 1024): "S1", (48, 1024): "S2", (64, 1024): "S3", (176, 1024): "L"}
    rows = []
    audit = []
    paired = {}
    for mode in ("random", "ordered"):
        path = HERE / "inputs" / f"raw176_extendedhot_{mode}_seed7.json.gz"
        requests, metadata = load_manifest(path)
        source_path = Path(metadata["source_fixed_manifest"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        source, source_metadata = load_manifest(source_path)
        assert sha(source_path) == metadata["source_fixed_manifest_sha256"]
        by_source_id = {request.request_id: request for request in source}
        assert len(requests) == len(source) == len(by_source_id) == 1212
        mode_rows = []
        lane_audit = []
        for npu in range(32):
            lane = sorted((r for r in requests if r.npu_id == npu), key=lambda r: r.request_id)
            old_ids = []
            counts = Counter()
            for position, request in enumerate(lane):
                source_id = int(request.load["source_fixed_request_id"])
                original = by_source_id[source_id]
                assert request.request_id == npu * 1_000_000 + position
                assert int(request.load["generation"]) == position
                assert request.arrival_time_ms == original.arrival_time_ms == 0.0
                assert request.placement == original.placement
                assert int(request.load["source_fixed_npu_id"]) == original.npu_id
                for key in ("seq_len_k", "nql", "per_layer_us", "per_layer_kv_gb", "role", "category"):
                    assert request.load[key] == original.load[key]
                assert len(request.placement) == 1 and metadata["n_layers"] == 8
                volumes = [0.0] * 6
                blocks = [0] * 6
                for ssu, volume in request.placement[0]:
                    volumes[ssu] += volume
                    blocks[ssu] += 1
                total = math.fsum(volumes)
                assert math.isclose(total, float(request.load["per_layer_kv_gb"]), abs_tol=1e-12)
                key = int(request.load["seq_len_k"]), int(request.load["nql"])
                profile = profiles[key]
                counts[profile] += 1
                old_ids.append(source_id)
                compute_ms = float(request.load["per_layer_us"]) / 1000
                row = dict(order_mode=mode, npu=npu, rowposition=position,
                    role=request.load["role"], profile=profile,
                    data_key=f"{key[0]}K/{key[1]}", sim_category=request.load["category"],
                    request_id=request.request_id, sourceid=source_id, source_npu=original.npu_id,
                    original_request_id=request.load["original_request_id"],
                    arrival_ms=0.0, n_layers=8, per_layer_compute_ms=compute_ms,
                    request_pure_compute_ms=8 * compute_ms, total_gib_per_layer=total,
                    request_total_read_gib=8 * total)
                row.update({f"ssu{s}_gib_per_layer": volumes[s] for s in range(6)})
                row.update({f"ssu{s}_blocks_per_layer": blocks[s] for s in range(6)})
                mode_rows.append(row)
            population = tuple(sorted(old_ids))
            if mode == "random":
                paired[npu] = population
            else:
                assert paired[npu] == population
            lane_audit.append(dict(npu=npu, requests=len(lane), profile_counts=dict(counts),
                sourceids=population,
                pure_compute_ms=math.fsum(row["request_pure_compute_ms"] for row in mode_rows if row["npu"] == npu)))
        assert len({row["sourceid"] for row in mode_rows}) == 1212
        assert Counter(row["profile"] for row in mode_rows) == {"S1": 264, "S2": 264, "S3": 264, "L": 420}
        rows.extend(mode_rows)
        audit.append(dict(order_mode=mode, manifest=str(path), manifest_sha256=sha(path),
            input_fingerprint=metadata["input_fingerprint"], source_manifest=str(source_path),
            source_manifest_sha256=sha(source_path), rows=len(mode_rows), npus=32,
            all_original_identity_C_V_arrival_placement_checks_pass=True, per_npu=lane_audit))
    output = HERE / "assignments_seed7.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence = dict(scope="v8 seed7 ordered AND its exact new-binding random control; input queues only",
        rowposition_definition="zero-based queue position within (order_mode,npu); not observed admission order under any changed assignment policy",
        volume_definition="ssu*_gib_per_layer is one immutable placement template, reused for all8layers; multiply by8 for whole-request per-SSU read work",
        sourceid_definition="source_fixed_request_id in frozen raw176 fixed source; old physical placement retained after rebinding",
        output=str(output), csv_sha256=sha(output), generator_sha256=sha(Path(__file__)),
        rows=len(rows), unique_requests_per_mode=1212, modes=2, exact_per_card_random_ordered_population_match=True,
        all_checks_pass=True, sources=audit)
    (HERE / "assignments_seed7_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k != "sources"}))


if __name__ == "__main__":
    main()
