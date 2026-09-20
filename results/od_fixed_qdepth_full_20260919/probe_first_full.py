#!/usr/bin/env python3
"""Replay frozen input only through the first queue-full events; preserve old runs."""
from pathlib import Path
from unittest.mock import patch
import gzip
import hashlib
import json
import math
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))
from inputs.manifest import load_manifest
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core


class ProbeFinished(Exception):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(seed):
    path = HERE / "inputs" / f"full_seed{seed}.json.gz"
    requests, meta = load_manifest(path)
    reference = HERE / "runs" / f"full_od_depth256_seed{seed}" / "command.json"
    command = json.loads(reference.read_text())
    assert sha(path) == command["manifest_sha256"]
    for rel, expected in command["source_sha256"].items():
        if rel.startswith("runtime/"):
            assert sha(HERE / rel) == expected
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (32, 3, 8)
    fills, blocked = [], []
    original_submit = core._register_submit
    original_can_submit = core._depth_can_submit
    original_client = core._handle_client_submission
    observed_until = None

    def row(context, npu, ssu, rid, layer):
        load = context.requests[rid].manifest.load
        return dict(time_ms=context.current_time_ms,
                    time_us=context.current_time_ms * 1000,
                    npu_id=npu, ssu_id=ssu, request_id=rid, layer=layer,
                    total_K=load["seq_len_k"], miss=load["nql"],
                    category=load["category"])

    def submit(context, flow):
        before = context.ssd_depth_outstanding[flow.npu_id][flow.disk_id]
        result = original_submit(context, flow)
        after = context.ssd_depth_outstanding[flow.npu_id][flow.disk_id]
        if before == 255 and after == 256:
            fills.append(row(context, flow.npu_id, flow.disk_id, flow.request_id, flow.layer))
        return result

    def can_submit(context, state):
        before = state.depth_blocked_since_ms
        result = original_can_submit(context, state)
        if before is None and state.depth_blocked_since_ms is not None:
            blocked.append(row(context, state.npu_id, state.disk_id, state.request_id, state.layer))
        return result

    def client(context, generation, now):
        nonlocal observed_until
        result = original_client(context, generation, now)
        if fills and len({r["ssu_id"] for r in fills}) == 3 and now >= fills[0]["time_ms"] + .001:
            observed_until = now
            raise ProbeFinished()
        if now > 1.0:
            raise AssertionError("No full event observed within the bounded startup replay")
        return result

    with patch.object(core, "_register_submit", submit), \
         patch.object(core, "_depth_can_submit", can_submit), \
         patch.object(core, "_handle_client_submission", client):
        try:
            run_simulation(requests, strategy="od_baseline", num_npu=32, num_ssu=3,
                           n_layers=8, seed=seed, od_queue_depth_per_ssu=8192)
        except ProbeFinished:
            pass
    assert observed_until is not None and fills and blocked
    first = min(r["time_ms"] for r in fills)
    first_blocked = min(r["time_ms"] for r in blocked)
    return dict(seed=seed, manifest_sha256=sha(path),
                input_fingerprint=core.continuous_batch_input_fingerprint(requests),
                replay_stopped_at_ms=observed_until,
                first_reached_256_ms=first, first_blocked_ms=first_blocked,
                first_fill_ties=[r for r in fills if math.isclose(r["time_ms"], first, rel_tol=0, abs_tol=1e-12)],
                first_blocked_ties=[r for r in blocked if math.isclose(r["time_ms"], first_blocked, rel_tol=0, abs_tol=1e-12)],
                first_per_ssu=[min((r for r in fills if r["ssu_id"] == s), key=lambda r:r["time_ms"]) for s in range(3)],
                all_observed_fill_transitions=fills, all_observed_blocked_entries=blocked)


def main():
    result = dict(scope="startup from simulation t=0; early stopped, not a new full performance run",
                  slot_definition="per execution NPU and SSD; queued plus SSD service; excludes HBM link",
                  fill_definition="actual submission changes occupancy from 255 to 256",
                  blocked_definition="submission state enters full-quota wait with unissued work",
                  command="python results/od_fixed_qdepth_full_20260919/probe_first_full.py",
                  cases=[probe(seed) for seed in (7, 19, 43)])
    target = HERE / "data/first_full_events.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"file": str(target), "cases": [
        {k:v for k,v in c.items() if k not in ("all_observed_fill_transitions", "all_observed_blocked_entries")}
        for c in result["cases"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
