#!/usr/bin/env python3
"""Freeze and run role-separated placements of the same 19,456 requests."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[1]
sys.path[:0] = [str(ROOT), str(PARENT)]
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, read_json, write_json
from run_shared_path_experiments import logical_input_fingerprint
from run_multi_ssu_stall_experiments import input_demand
from run_coflow_experiments import source_files
from run_study import certificate

SEEDS = (7, 19, 43, 67, 101)
LONG_COUNTS = (11, 6)
ORDERS = ("random", "round_robin")
STRATEGIES = ("baseline", "once")
PROFILES = ((1, 128), (1, 256), (1, 384), (192, 768))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def label_for(seed, n_long, order):
    return f"role_l{n_long}_s{32-n_long}_seed{seed}_{order}"


def source_hashes():
    paths = {ROOT / name for name in source_files()}
    paths.update((ROOT / "run_baseline_npu32_stress.py", ROOT / "data",
                  PARENT / "run_study.py", Path(__file__)))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def allocate(original, seed, n_long):
    """Profile-wise allocation is frozen before either within-lane ordering."""
    lanes = [[] for _ in range(32)]
    for profile_index, profile in enumerate(PROFILES):
        pool = sorted((r for r in original if (r.load["seq_len_k"], r.load["nql"]) == profile),
                      key=lambda r: r.request_id)
        assert len(pool) == (256 if profile_index == 3 else 6400)
        # Separate fixed streams: assignment never depends on within-lane order.
        random.Random(seed * 1_000_003 + n_long * 10_007 + profile_index * 101 + 17).shuffle(pool)
        targets = list(range(n_long)) if profile_index == 3 else list(range(n_long, 32))
        for i, request in enumerate(pool):
            lanes[targets[i % len(targets)]].append(request)
    assert sum(map(len, lanes)) == 19456
    return lanes


def reorder(lane, seed, npu, order):
    if lane[0].load["role"] == "long":
        # Long identities and order are identical in random and round_robin.
        return sorted(lane, key=lambda r: r.request_id)
    if order == "random":
        result = sorted(lane, key=lambda r: r.request_id)
        random.Random(seed * 1_000_003 + npu * 100_003 + 71_923).shuffle(result)
        return result
    queues = [sorted((r for r in lane if (r.load["seq_len_k"], r.load["nql"]) == p),
                     key=lambda r: r.request_id) for p in PROFILES[:3]]
    return [queue[turn] for turn in range(max(map(len, queues)))
            for queue in queues if turn < len(queue)]


def prepare():
    if (HERE / "plan.json").exists():
        raise FileExistsError("Frozen plan already exists; use --run without --prepare")
    hashes = source_hashes()
    for name, digest in hashes.items():
        target = HERE / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == digest
        else:
            shutil.copyfile(ROOT / name, target)
    items = []
    for seed in SEEDS:
        source = PARENT / "inputs" / f"concurrency_l768_seed{seed}.json.gz"
        original, oldmeta = load_manifest(source)
        assert len(original) == 19456 and all(r.arrival_time_ms == 0 for r in original)
        original_ids = {r.request_id for r in original}
        assert len(original_ids) == len(original)
        for n_long in LONG_COUNTS:
            lanes = allocate(original, seed, n_long)
            lane_members = [canonical(sorted(r.request_id for r in lane)) for lane in lanes]
            for order in ORDERS:
                label = label_for(seed, n_long, order)
                requests, lane_rows, order_hashes = [], [], []
                for npu, lane in enumerate(lanes):
                    ordered = reorder(lane, seed, npu, order)
                    assert Counter(r.request_id for r in ordered) == Counter(r.request_id for r in lane)
                    role = "long" if npu < n_long else "short"
                    assert {r.load["role"] for r in lane} == {role}
                    counts = [sum((r.load["seq_len_k"], r.load["nql"]) == p for r in lane) for p in PROFILES]
                    compute = 8 * math.fsum(r.load["per_layer_us"] / 1000 for r in lane)
                    assert compute > 4000
                    lane_rows.append(dict(npu_id=npu, assigned_role=role, request_count=len(lane),
                                          profile_counts=counts, ideal_compute_ms=compute,
                                          original_identity_multiset_sha256=lane_members[npu]))
                    order_hashes.append(canonical([r.request_id for r in ordered]))
                    for position, old in enumerate(ordered):
                        rid = npu * 1_000_000 + position
                        load = dict(old.load, request_id=rid, npu_id=npu, generation=position,
                                    original_request_id=old.request_id,
                                    source_original_npu_id=old.npu_id)
                        new = ContinuousBatchRequest.from_normalized(rid, npu, old.arrival_time_ms, load, old.placement)
                        assert new.placement == old.placement
                        assert all(new.load[k] == v for k, v in old.load.items()
                                   if k not in ("request_id", "npu_id", "generation"))
                        requests.append(new)
                requests = tuple(requests)
                assert {r.load["original_request_id"] for r in requests} == original_ids
                fingerprint = continuous_batch_input_fingerprint(requests)
                demand = input_demand(requests, 32, 6)
                proof = certificate(requests)
                assert proof["passes"]
                # Retain source profile measurements, but replace all derived
                # lane/mixing/placement fields of the original mixed workload.
                metadata = {k: oldmeta[k] for k in (
                    "family", "profiles", "profile_keys", "source", "blocks", "equal_176kib_blocks",
                    "n_layers", "num_npu", "num_ssu", "compute_scale_actual")}
                metadata["profiles"] = [dict((k,v) for k,v in p.items() if k not in ("npu_id", "role"))
                                        for p in oldmeta["profiles"]]
                metadata.update(
                    experiment="32npu6ssu_role_separated_same_global_population_v1",
                    label=label, case_id=f"{label}_{fingerprint[:12]}", seed=seed,
                    input_fingerprint=fingerprint, logical_input_fingerprint=logical_input_fingerprint(requests),
                    source_manifest=str(source), source_manifest_sha256=sha(source),
                    source_input_fingerprint=oldmeta["input_fingerprint"], source_request_count=19456,
                    source_profile_counts=[6400, 6400, 6400, 256],
                    long_npu_count=n_long, short_npu_count=32-n_long,
                    assigned_role_by_npu=[x["assigned_role"] for x in lane_rows],
                    per_npu_assignment=lane_rows,
                    profile_count_per_npu=[sum(c > 0 for c in x["profile_counts"]) for x in lane_rows],
                    short_profile_count_per_npu=[3 if x["assigned_role"] == "short" else 0 for x in lane_rows],
                    profile_counts_per_npu=[x["profile_counts"] for x in lane_rows],
                    short_compute_fraction_per_npu=[1.0 if x["assigned_role"] == "short" else 0.0 for x in lane_rows],
                    requests_per_npu=[x["request_count"] for x in lane_rows], request_count=len(requests),
                    short_requests_per_npu=[x["request_count"] if x["assigned_role"] == "short" else 0 for x in lane_rows],
                    long_requests_per_npu=[x["request_count"] if x["assigned_role"] == "long" else 0 for x in lane_rows],
                    order=order, order_mode=order, per_npu_deck_sha256=order_hashes,
                    per_npu_population_sha256=lane_members,
                    assignment_rule="For each source profile, sort original request IDs, independently shuffle with seed*1000003+long_npu_count*10007+profile_index*101+17, and round-robin across its assigned role NPUs. Allocation is independent of internal order.",
                    shuffle_rule="random: independently shuffle each full short-card list using seed*1000003+npu*100003+71923; round_robin: consume sorted-identity queues S1,S2,S3 until exhausted. Long-card identities stay sorted in both orders.",
                    id_rule="request_id=new_npu*1000000+position; generation=position; load.npu_id=new_npu. original_request_id and source_original_npu_id retain the exact source mapping.",
                    layout="preserved_original_placement_after_role_rebinding",
                    stripe_group_rule="Placement retained verbatim from each original request; original stripe offset is source_original_npu_id//4, NOT new NPU ID. No blocks are re-striped.",
                    input_demand=demand, active_profile_rate_certificate=proof,
                    load_within_disk_and_link_capacity=proof["passes"],
                    last_arrival_ms=0.0, measurement_window_ms=[2000, 4000],
                    required_minimum_compute_ms_per_npu=4000,
                    warmup_requirement="Every NPU completes at least four requests by 1500 ms, then settles until the [2000,4000) window.",
                    active_window_requirement="All 32 NPUs remain active throughout [2000,4000); each NPU computes only its assigned role. Per-NPU mixing is intentionally not required.",
                    sampling_caveat="Role rebinding changes per-NPU population relative to the original mixed study. Within each ratio and seed, random versus round_robin preserves each new NPU's exact request population, placement and arrivals. Synthetic 1K short profiles and interpolated 192K/768 long profile are unchanged.",
                    full_run_tail_caveat="Role allocation changes ideal completion times; especially 6 long NPUs create a long tail. Full-run device utilization must not be interpreted as warm-window FIFO loss.",
                    construction_source_sha256=hashes,
                )
                path = HERE / "inputs" / f"{label}.json.gz"
                assert not path.exists()
                save_manifest(path, requests, metadata)
                write_json(path.with_name(path.stem + ".description.json"), metadata)
                items.append(dict(label=label, seed=seed, long_npu_count=n_long, short_npu_count=32-n_long,
                                  order=order, manifest=str(path), manifest_sha256=sha(path),
                                  input_fingerprint=fingerprint, request_count=len(requests),
                                  minimum_compute_ms=min(x["ideal_compute_ms"] for x in lane_rows),
                                  rate_certificate=proof))
                print(json.dumps(dict(prepared=label, minimum_compute_ms=items[-1]["minimum_compute_ms"],
                                      max_ssu_gib_s=max(proof["per_ssu_upper_bound_gib_s"])), ensure_ascii=False), flush=True)
    jobs = [dict(label=item["label"], strategy=strategy) for strategy in STRATEGIES for item in items]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), seeds=list(SEEDS),
                long_npu_counts=list(LONG_COUNTS), orders=list(ORDERS), strategies=list(STRATEGIES),
                inputs=items, jobs=jobs, source_sha256=hashes,
                python_executable=sys.executable, python_version=sys.version, platform=platform.platform(),
                num_npu=32, num_ssu=6, n_layers=8, fixed_npu_binding=True, window_ms=[2000,4000],
                collector_interval_ms=5.0, slo_alpha=1.5,
                primary_slo="Admissions in [2000,4000), completion-admission <= 1.5*8*C, every member followed to completion without censoring at 4000 ms.",
                arrival_caveat="Every true arrival is zero; warm arrival cohort is empty. Admission TTFT is a processing proxy and excludes external queue waiting.",
                main_window_valid="Technical/source audits, fourth completion by 1500 ms, all 32 NPUs active throughout [2000,4000), assigned role only, and nominal per-SSU/link capacity for the entire run.")
    write_json(HERE / "plan.json", plan)
    return plan


def assert_sources(plan):
    assert all(sha(ROOT / name) == digest for name,digest in plan["source_sha256"].items())


def run_job(item, strategy, plan, timeout):
    assert sha(item["manifest"]) == item["manifest_sha256"]
    directory = HERE / "runs" / item["label"] / strategy
    directory.mkdir(parents=True, exist_ok=True)
    outputs = list(directory.glob("*.json.gz"))
    if outputs:
        assert len(outputs) == 1
        result = read_json(outputs[0])
        assert result["input_fingerprint"] == item["input_fingerprint"]
        assert result["strategy"] == strategy and result["submit_seed"] == item["seed"]
        assert result["collector_interval_ms"] == 5.0
        assert result["stress_runner_sha256"] == sha(ROOT / "run_baseline_npu32_stress.py")
        assert all(sha(ROOT / name) == digest for name,digest in result["core_and_policy_sha256"].items())
        assert all(result["summary"]["invariants"].values())
        return dict(label=item["label"], strategy=strategy, status="existing", output=str(outputs[0]))
    command = [sys.executable, "-B", str(ROOT / "run_baseline_npu32_stress.py"),
               "--manifest", item["manifest"], "--strategy", strategy, "--assignment", "fixed",
               "--window", "2000:4000", "--output", str(directory)]
    started = time.perf_counter()
    record = dict(label=item["label"], seed=item["seed"], strategy=strategy, command=command,
                  cwd=str(ROOT), start_utc=datetime.now(timezone.utc).isoformat(),
                  input_fingerprint=item["input_fingerprint"], python_version=sys.version,
                  platform=platform.platform(), worker_limit=4, source_sha256=plan["source_sha256"])
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        write_json(directory / "command.json", record)
        try:
            code = process.wait(timeout=timeout)
            status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code, status = process.returncode, "timeout"
    record.update(returncode=code, status=status, wall_seconds=time.perf_counter()-started,
                  end_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "command.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2400)
    args = parser.parse_args()
    plan = prepare() if args.prepare else read_json(HERE / "plan.json")
    if not args.run:
        return
    assert 1 <= args.workers <= 4
    assert_sources(plan)
    inputs = {x["label"]:x for x in plan["inputs"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    status_path = HERE / f"status_{stamp}.json"
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_job, inputs[job["label"]], job["strategy"], plan, args.timeout)
                   for job in plan["jobs"]}
        while pending:
            done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                row = future.result()
                rows.append(row)
                write_json(status_path, dict(total=len(plan["jobs"]), complete=len(rows), rows=rows))
                print(json.dumps({k:row[k] for k in ("label","strategy","status","wall_seconds") if k in row}), flush=True)
            if not done:
                print(json.dumps(dict(finished=len(rows), total=len(plan["jobs"]), pending_including_queued=len(pending))), flush=True)
    assert_sources(plan)
    if any(row["status"] in ("failed", "timeout") for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
