#!/usr/bin/env python3
"""Two frozen 32-NPU / 6-SSU mixed-input controls; four local jobs only."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(BASE), str(BASE / "notes")]

import numpy as np
import run_mixed_sustained_probe as frozen_helper
from aggregate_mixed_profiles import audit_manifest
from run_baseline_npu32_stress import load_manifest, save_manifest, read_json, write_json

STRATEGIES = ("baseline", "new_once")
SPECS = (
    {"label": "aligned_short080_ssu6", "family": "aligned",
     "profile_keys": "1:128,1:256,1:384,192:256", "quotas": (65, 65, 65, 4),
     "short_profile_count": 3},
    {"label": "raw_size_varied_ssu6", "family": "raw",
     "profile_keys": "32:128,48:256,64:512,192:1024,192:2048",
     "quotas": (6, 6, 6, 1, 3), "short_profile_count": 3},
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "data", Path(frozen_helper.__file__), Path(__file__)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare():
    # Only this Python process's configuration is changed. No helper/source edit.
    frozen_helper.NUM_SSU = 6
    frozen_helper.STRATEGIES = STRATEGIES
    before = sources()
    inputs, audits = [], []
    for spec in SPECS:
        requests, metadata = frozen_helper.build_input(seed=7, **spec)
        metadata.update({
            "experiment": "mixed_six_ssu_topology_controls_v1",
            "stripe_group_rule": "group = original_npu_id // 4; ssu=(block_index+group)%6",
            "construction_wrapper": str(Path(__file__).relative_to(ROOT)),
            "construction_wrapper_sha256": sha(Path(__file__)),
            "construction_helper_sha256": sha(Path(frozen_helper.__file__)),
            "process_configuration_override": {"run_mixed_sustained_probe.NUM_SSU": 6},
            "comparison_scope": "Only placement/topology differs from frozen 8-SSU E80 input" if spec["family"] == "aligned" else
                "New 6-SSU raw population, quota 6:6:6:1:3; not a topology-only change to the 8-SSU population",
        })
        bridge = None
        if spec["family"] == "aligned":
            reference = BASE / "mixed_varied/inputs/aligned_short080_seed7.json.gz"
            original, original_meta = load_manifest(reference)
            assert len(requests) == len(original)
            checks = {
                "request_list_order_same": [r.request_id for r in requests] == [r.request_id for r in original],
                "all_request_nonplacement_fields_exact": all(
                    (a.request_id, a.npu_id, a.arrival_time_ms, dict(a.load)) ==
                    (b.request_id, b.npu_id, b.arrival_time_ms, dict(b.load))
                    for a, b in zip(requests, original)),
                "logical_input_fingerprint_same": metadata["logical_input_fingerprint"] == original_meta["logical_input_fingerprint"],
                "per_npu_deck_hashes_same": metadata["per_npu_deck_sha256"] == original_meta["per_npu_deck_sha256"],
                "physical_input_fingerprint_changed": metadata["input_fingerprint"] != original_meta["input_fingerprint"],
            }
            assert all(checks.values()), checks
            bridge = {"reference_manifest": str(reference), "reference_manifest_sha256": sha(reference),
                      "reference_input_fingerprint": original_meta["input_fingerprint"],
                      "new_input_fingerprint": metadata["input_fingerprint"], "checks": checks,
                      "all_checks_passed": all(checks.values()), "request_count": len(requests)}
            metadata["reference_8_ssu"] = bridge
        path = HERE / "inputs" / (metadata["label"] + ".json.gz")
        save_manifest(path, requests, metadata)
        write_json(path.with_name(path.stem + ".description.json"), metadata)
        _, _, audit = audit_manifest(path)
        assert audit["mean_capacity_feasible"]
        audits.append({"input_audit": audit, "topology_bridge": bridge})
        inputs.append({"label": metadata["label"], "manifest": str(path),
                       "input_fingerprint": metadata["input_fingerprint"], "request_count": len(requests)})
    assert sources() == before
    plan = {"inputs": inputs, "strategies": list(STRATEGIES), "job_count": 4,
            "windows_ms": [[1000, 2000], [2000, 3000]], "fixed_npu_binding": True,
            "collector_interval_ms": 5.0, "worker_count": 2,
            "source_sha256": before, "wrapper_sha256": sha(Path(__file__)),
            "python_executable": sys.executable, "python_version": sys.version,
            "numpy_version": np.__version__, "platform": platform.platform(),
            "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(HERE / "plan.json", plan)
    write_json(HERE / "input_audit.json", {"audits": audits, "source_sha256": before})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()
    plan = read_json(HERE / "plan.json") if args.run_only else prepare()
    assert plan["source_sha256"] == sources(), "Frozen source hash changed"
    print(json.dumps({"prepared": len(plan["inputs"]), "jobs": 4,
                      "input_fingerprints": [x["input_fingerprint"] for x in plan["inputs"]]}), flush=True)
    if args.prepare_only:
        return
    rows = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = {executor.submit(frozen_helper.run_job, item, strategy, HERE)
                   for item in plan["inputs"] for strategy in STRATEGIES}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(HERE / "run_status.json", {"completed": len(rows), "total": 4, "rows": rows})
                print(json.dumps(row), flush=True)
            if not done:
                print(json.dumps({"complete": len(rows), "pending": len(pending)}), flush=True)
    unchanged = plan["source_sha256"] == sources()
    write_json(HERE / "source_integrity_after.json", {"all_sources_unchanged": unchanged,
               "source_sha256": sources(), "finished_utc": datetime.now(timezone.utc).isoformat()})
    assert unchanged
    assert len(rows) == 4 and all(r["status"] in ("complete", "existing") for r in rows), rows


if __name__ == "__main__":
    main()
