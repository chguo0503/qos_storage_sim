#!/usr/bin/env python3
"""Reorder the same per-NPU population, keeping every physical placement."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_study import run_job, sha
from order_design import MODES, order_indices, describe_order


def prepare(label, mode, packets, jitter):
    source = HERE / "inputs" / (label + ".json.gz")
    original, meta = load_manifest(source)
    profiles = [dict(p, analysis_role="short" if i < meta["short_profile_count_per_npu"] else "long")
                for i, p in enumerate(meta["profiles"])]
    by_profile = {(p["seq_len_k"], p["nql"]): i for i, p in enumerate(profiles)}
    new_label = f"{label}__{mode}_p{packets}" + (f"_j{jitter:g}" if jitter else "")
    reordered, descriptions, deck_hashes = [], [], []
    for npu in range(32):
        lane = sorted((r for r in original if r.npu_id == npu), key=lambda r: (r.arrival_time_ms, r.request_id))
        indices = [by_profile[(r.load["seq_len_k"], r.load["nql"])] for r in lane]
        kwargs = dict(npu=npu, seed=meta["seed"], mode=mode, period_packets=packets, phase_jitter=jitter)
        permutation = order_indices(indices, profiles, **kwargs)
        descriptions.append(dict(npu_id=npu, **describe_order(indices, profiles, **kwargs)))
        assert sorted(permutation) == list(range(len(lane)))
        deck_hashes.append(hashlib.sha256(json.dumps([indices[i] for i in permutation]).encode()).hexdigest())
        for position, index in enumerate(permutation):
            old = lane[index]
            new_id = npu * 1_000_000 + position
            load = dict(old.load, request_id=new_id, generation=position, original_request_id=old.request_id)
            request = ContinuousBatchRequest.from_normalized(new_id, npu, old.arrival_time_ms, load, old.placement)
            assert request.placement == old.placement
            assert all(request.load[k] == v for k, v in old.load.items() if k not in ("request_id", "generation"))
            reordered.append(request)
    requests = tuple(reordered)
    fingerprint = continuous_batch_input_fingerprint(requests)
    metadata = dict(meta)
    metadata.update(label=new_label, case_id=f"{new_label}_{fingerprint[:12]}",
                    experiment="32npu_6ssu_same_population_order_intervention_v1",
                    input_fingerprint=fingerprint, logical_input_fingerprint=logical_input_fingerprint(requests),
                    random_reference_label=label, random_reference_input_fingerprint=meta["input_fingerprint"],
                    random_reference_manifest_sha256=sha(source),
                    order_mode=mode, order_period_packets=packets, phase_jitter=jitter,
                    order="deterministic_role_packets_with_per_npu_phase",
                    per_npu_deck_sha256=deck_hashes, per_npu_order_design=descriptions,
                    shuffle_rule="Each original role queue is shuffled independently, then arranged into packets and phase-rotated by pure compute. See per_npu_order_design.",
                    id_rule="request_id=npu*1000000+new_queue_position; original_request_id maps exactly to random reference. Placement is kept verbatim, never rehashed.",
                    order_source_sha256={name: sha(HERE / name) for name in ("run_orders.py", "order_design.py")},
                    sampling_caveat="Same per-NPU request population and all-t=0 arrivals as random reference; only input queue order and its position IDs change. Structured mechanism input, not a production trace.")
    path = HERE / "inputs" / (new_label + ".json.gz")
    save_manifest(path, requests, metadata)
    write_json(path.with_name(path.stem + ".description.json"), metadata)
    return {"label": new_label, "manifest": str(path), "input_fingerprint": fingerprint,
            "random_reference_label": label, "mode": mode, "period_packets": packets,
            "request_count": len(requests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-label", action="append", required=True)
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--packets", type=int, action="append")
    parser.add_argument("--jitter", type=float, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    for name in ("run_orders.py", "order_design.py"):
        target = HERE / "sources" / "order_intervention" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(HERE / name)
        else:
            shutil.copyfile(HERE / name, target)
    inputs = [prepare(label, mode, packets, args.jitter)
              for label in args.source_label for mode in (args.mode or MODES)
              for packets in (args.packets or [1])]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    plan_path = HERE / f"order_plan_{stamp}.json"
    write_json(plan_path, {"inputs": inputs, "strategies": ["baseline"], "window_ms": [2000, 4000]})
    print(json.dumps({"prepared": inputs}, ensure_ascii=False), flush=True)
    if not args.run:
        return
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, item, "baseline", args.timeout) for item in inputs}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(HERE / f"order_status_{stamp}.json", {"total": len(inputs), "rows": rows})
                print(json.dumps(row, ensure_ascii=False), flush=True)
            if not done:
                print(json.dumps({"finished": len(rows), "total": len(inputs), "pending_including_queued": len(pending)}), flush=True)
    if any(r["status"] in ("failed", "timeout") for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
