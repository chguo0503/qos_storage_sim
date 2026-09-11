#!/usr/bin/env python3
"""Predeclare a selected population/order and repeat with additional seeds."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "results/baseline_npu32_investigation")]
import run_mixed_sustained_probe as helper
from run_baseline_npu32_stress import load_manifest, save_manifest, write_json
from run_study import certificate, run_job, sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--order-runner", choices=("run_exact_orders.py", "run_phase_orders.py"), default="run_exact_orders.py")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--packets", type=int, default=1)
    args = parser.parse_args()
    _, ref = load_manifest(HERE / "inputs" / (args.reference_label + ".json.gz"))
    helper.NUM_SSU = 6
    items = []
    for seed in args.seed:
        requests, meta = helper.build_input(
            family=ref["family"], seed=seed, profile_keys=ref["profile_keys"],
            quotas=ref["profile_quotas_per_unit"], short_profile_count=ref["short_profile_count_per_npu"],
            label=args.reference_label.rsplit("_seed", 1)[0])
        meta.update(experiment="32npu6ssu_order_replication_v1",
                    stripe_group_rule="group=original_npu_id//4; ssu=(block_index+group)%6",
                    active_profile_rate_certificate=certificate(requests),
                    measurement_window_ms=[2000,4000], replication_reference_label=args.reference_label,
                    preregistered_order={"runner": args.order_runner, "mode": args.mode, "packets": args.packets},
                    construction_runner_sha256=sha(__file__),
                    replication_scope="Same per-NPU profile quotas and physical layout; new independent random input order and submission seed. Fixed order design, no retuning to new results.")
        path = HERE / "inputs" / (meta["label"] + ".json.gz")
        save_manifest(path, requests, meta)
        write_json(path.with_name(path.stem + ".description.json"), meta)
        items.append({"label": meta["label"], "manifest": str(path), "input_fingerprint": meta["input_fingerprint"], "request_count": len(requests)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    plan = {"inputs": items, "strategies": ["baseline"], "selected_reference": args.reference_label,
            "order": {"runner": args.order_runner, "mode": args.mode, "packets": args.packets},
            "source_sha256": {name: sha(HERE / name) for name in ("run_replication.py", args.order_runner)}}
    write_json(HERE / f"replication_plan_{stamp}.json", plan)
    print(json.dumps({"prepared_replication": plan}, ensure_ascii=False), flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(run_job, item, "baseline", 1800) for item in items}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(HERE / f"replication_status_{stamp}.json", {"rows": rows})
                print(json.dumps(row), flush=True)
            if not done:
                print(json.dumps({"random_finished": len(rows), "random_total": len(items)}), flush=True)
    if any(row["status"] in ("failed", "timeout") for row in rows):
        raise SystemExit(1)
    command = [sys.executable, "-B", str(HERE / args.order_runner)]
    for item in items:
        command += ["--source-label", item["label"]]
    command += ["--mode", args.mode, "--packets", str(args.packets), "--run", "--workers", "2"]
    raise SystemExit(subprocess.call(command, cwd=ROOT))
