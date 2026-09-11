#!/usr/bin/env python3
"""Check global identity, the new binding's paired populations, and capacity bound."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from run_baseline_npu32_stress import load_manifest
import mixed_design_v5 as design


def main():
    rows = []
    for seed in (7, 19, 43, 67, 101):
        path = HERE / "inputs" / f"raw200_three_l20_fixed_seed{seed}.json.gz"
        if not path.exists():
            continue
        source, metadata = load_manifest(path)
        random, random_desc = design.build_queues(source, metadata, seed, "random")
        ordered, ordered_desc = design.build_queues(source, metadata, seed, "ordered")
        replay, replay_desc = design.build_queues(tuple(reversed(source)), metadata, seed, "ordered")
        assert replay_desc == ordered_desc
        assert Counter(id(r) for lane in random.values() for r in lane) == Counter(id(r) for r in source)
        assert Counter(id(r) for lane in ordered.values() for r in lane) == Counter(id(r) for r in source)
        lanes = []
        certificate = [0.0] * 6
        for npu in range(32):
            assert Counter(id(r) for r in random[npu]) == Counter(id(r) for r in ordered[npu])
            assert all(a is b for a, b in zip(ordered[npu], replay[npu]))
            expected = [3, 3, 3, 13] if npu < 17 else [4, 4, 4, 11] if npu < 20 else [17 if npu < 29 else 16] * 3 + [11 if npu < 26 else 10]
            profile_counts = Counter((int(r.load["seq_len_k"]), int(r.load["nql"])) for r in ordered[npu])
            actual = [profile_counts[k] for k in design.PROFILE_KEYS]
            assert actual == expected
            pure_compute = sum(8 * float(r.load["per_layer_us"]) / 1000 for r in ordered[npu])
            assert abs(pure_compute - ordered_desc["per_npu_assignment"][npu]["pure_compute_ms"]) < 1e-8
            rates = [0.0] * 6
            for request in ordered[npu]:
                per_disk = [0.0] * 6
                for ssu, amount in request.placement[0]:
                    per_disk[ssu] += amount
                compute_seconds = float(request.load["per_layer_us"]) / 1e6
                rates = [max(old, value / compute_seconds) for old, value in zip(rates, per_disk)]
            certificate = [a + b for a, b in zip(certificate, rates)]
            lanes.append(dict(npu=npu, quotas=actual, pure_compute_ms=pure_compute,
                paired_original_objects=True, deterministic_despite_source_enumeration=True,
                original_request_ids=sorted(r.request_id for r in ordered[npu]), max_profile_per_disk_gib_s=rates))
        rows.append(dict(seed=seed, source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_requests=len(source), modes=["random", "ordered"], all_checks_pass=True,
            all_lane_pure_compute_exceeds4000=all(row["pure_compute_ms"] > 4000 for row in lanes),
            min_lane_pure_compute_ms=min(row["pure_compute_ms"] for row in lanes),
            per_disk_static_max_certificate_gib_s=certificate,
            static_sufficient_certificate_passes=all(x < 40.0 for x in certificate),
            static_certificate_is_not_actual_trace=True,
            capacity_failure_not_an_input_integrity_failure=True,
            per_npu=lanes))
    assert rows
    output = dict(scope="Pure generation checks; no simulation or claims about warm mix/actual capacity/U",
        design_sha256=hashlib.sha256(Path(design.__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_inputs=len(rows), modes_per_source=2, lanes_paired=32 * len(rows), all_checks_pass=True, rows=rows)
    (HERE / "mixed_design_v5_checks.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}))
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "per_npu"}))


if __name__ == "__main__":
    main()
