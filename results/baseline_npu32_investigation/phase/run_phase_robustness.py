"""Perturb only per-NPU startup arrivals of a frozen strong32_stripe input."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, write_json
from run_shared_path_experiments import logical_input_fingerprint
from sweep_baseline_npu32_stress import execute


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, output, jitter_seed):
    original, original_metadata = load_manifest(source)
    assert original_metadata["num_npu"] == 32
    assert original_metadata["num_ssu"] == 8
    assert all(r.arrival_time_ms == 0 for r in original)
    rng = random.Random(jitter_seed)
    offsets = [rng.uniform(0, 10) for _ in range(32)]
    requests = []
    for request in original:
        arrival = offsets[request.npu_id]
        load = dict(request.load)
        load.update(arrival_time=arrival, arrival_ms=arrival)
        updated = ContinuousBatchRequest.from_normalized(
            request.request_id, request.npu_id, arrival, load, request.placement)
        assert updated.request_id == request.request_id
        assert updated.npu_id == request.npu_id
        assert updated.placement == request.placement
        assert {k: v for k, v in updated.load.items() if k not in ("arrival_time", "arrival_ms")} == {
            k: v for k, v in request.load.items() if k not in ("arrival_time", "arrival_ms")}
        requests.append(updated)
    requests = tuple(requests)
    fingerprint = continuous_batch_input_fingerprint(requests)
    label = f"strong32_stripe_jitter_seed{jitter_seed}"
    metadata = dict(original_metadata)
    metadata.update({
        "case_id": f"{label}_{fingerprint[:12]}",
        "regime": "finite_backlog_per_npu_startup_jitter",
        "last_arrival_ms": max(offsets),
        "initial_arrived_request_count": sum(r.arrival_time_ms == 0 for r in requests),
        "input_fingerprint": fingerprint,
        "logical_input_fingerprint": logical_input_fingerprint(requests),
        "base_manifest": str(source.resolve()), "base_manifest_sha256": sha(source),
        "base_input_fingerprint": original_metadata["input_fingerprint"],
        "base_logical_input_fingerprint": original_metadata["logical_input_fingerprint"],
        "startup_offset_rng_seed": jitter_seed,
        "startup_offset_ms_by_npu": offsets,
        "startup_offset_rule": "random.Random(jitter_seed).uniform(0,10), once per NPU in ID order; all requests on that lane arrive at this offset",
        "native_submit_seed_preserved": original_metadata["seed"],
        "sampling_caveat": "Only startup arrival phases differ from frozen strong32_stripe. Per-lane finite backlog, not an observed arrival distribution.",
        "treatment_caveat": "Preserves request IDs, original NPU bindings, population, profile values, physical placement, compute scale, nominal demand and native submission seed. Jittered versus original is not an identical-input policy comparison.",
    })
    assert metadata["seed"] == original_metadata["seed"]
    assert fingerprint != original_metadata["input_fingerprint"]
    spec = {"base_manifest": str(source.resolve()), "base_manifest_sha256": sha(source),
            "startup_offset_rng_seed": jitter_seed, "startup_offset_ms_by_npu": offsets,
            "preserved_native_submit_seed": original_metadata["seed"]}
    save_manifest(output / "inputs" / f"{label}.json.gz", requests, metadata)
    write_json(output / "inputs" / f"{label}.spec.json", spec)
    return label, spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/baseline_npu32_investigation/screen/inputs/strong32_stripe.json.gz")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = dict(prepare(args.source, output, seed) for seed in (7, 123))
    jobs = [{"input": label, "strategy": strategy,
             "config": {"windows": [[1000, 2000], [2000, 3000]]}}
            for label in inputs for strategy in ("baseline", "once")]
    write_json(output / "phase_plan.json", {"inputs": inputs, "jobs": jobs,
        "policy_information": "Both Baseline and Once use the existing coflow wrapper with true 5ms collector; no dynamic CIR writes."})
    sources = {p.name: sha(p) for p in ROOT.glob("*.py")}
    write_json(output / "phase_environment.json", {
        "at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "source_sha256": sources,
        "phase_script_sha256": sha(Path(__file__)), "workers": args.workers,
        "base_manifest_sha256": sha(args.source)})
    if args.prepare_only:
        return
    rows, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(execute, job, output, args.timeout): job for job in jobs}
        while pending:
            finished, _ = wait(pending, timeout=45, return_when=FIRST_COMPLETED)
            for future in finished:
                job = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"input": job["input"], "strategy": job["strategy"],
                           "status": "orchestrator_error", "error": repr(exc)}
                rows.append(row)
                print(json.dumps({"finished": row}), flush=True)
            status = {"elapsed_seconds": time.monotonic() - started,
                      "completed": len(rows), "total": len(jobs),
                      "failures": sum(r["status"] not in ("complete", "existing") for r in rows),
                      "rows": rows}
            write_json(output / "phase_status.json", status)
            if not finished:
                print(json.dumps({k: v for k, v in status.items() if k != "rows"}), flush=True)
    assert all(sha(ROOT / name) == value for name, value in sources.items()), "root sources changed during phase sweep"
    if any(r["status"] not in ("complete", "existing") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
