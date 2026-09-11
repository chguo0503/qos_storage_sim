#!/usr/bin/env python3
"""Broader candidates accepted only after actual per-event demand validation."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "results/baseline_npu32_investigation")]
import run_mixed_sustained_probe as helper
from run_study import certificate, run_job, sha
from run_baseline_npu32_stress import save_manifest, write_json

SPECS = {
    "concurrency_l512": ("1:128,1:256,1:384,192:512", (45, 45, 45, 1), 12),
    "concurrency_l768": ("1:128,1:256,1:384,192:768", (25, 25, 25, 1), 20),
    "concurrency_l1024": ("1:128,1:256,1:384,192:1024", (18, 18, 18, 1), 31),
}


if __name__ == "__main__":
    helper.NUM_SSU = 6
    items = []
    for label, (keys, quotas, k_limit) in SPECS.items():
        requests, meta = helper.build_input(family="aligned", seed=7, profile_keys=keys,
                                          quotas=quotas, short_profile_count=3, label=label)
        meta.update(experiment="32npu_6ssu_actual_concurrency_underload_candidates_v1",
                    stripe_group_rule="group=original_npu_id//4; ssu=(block_index+group)%6",
                    active_profile_rate_certificate=certificate(requests),
                    theoretical_sufficient_max_concurrent_long=k_limit,
                    capacity_acceptance="Universal maximum may exceed capacity. Accept only if actual full-run event scan has per-SSU V/C strictly below 40 GiB/s, excluding extra L0 term as agreed.",
                    measurement_window_ms=[2000,4000],
                    warmup_requirement="Four completed requests per card by 1500 ms; fixed [2000,4000) window, all cards continuously active and each with short+long compute.",
                    construction_runner_sha256=sha(__file__))
        path = HERE / "inputs" / (meta["label"] + ".json.gz")
        save_manifest(path, requests, meta)
        write_json(path.with_name(path.stem + ".description.json"), meta)
        items.append({"label": meta["label"], "manifest": str(path), "input_fingerprint": meta["input_fingerprint"], "request_count": len(requests)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    write_json(HERE / f"concurrency_plan_{stamp}.json", {"inputs": items, "strategies": ["baseline"], "source_sha256": {"run_concurrency_candidates.py": sha(__file__)}})
    print(json.dumps({"prepared": items}), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(run_job, item, "baseline", 1800) for item in items}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(HERE / f"concurrency_status_{stamp}.json", {"total": len(items), "rows": rows})
                print(json.dumps(row), flush=True)
            if not done:
                print(json.dumps({"finished": len(rows), "total": len(items), "pending_including_queued": len(pending)}), flush=True)
    if any(r["status"] in ("failed", "timeout") for r in rows):
        raise SystemExit(1)
