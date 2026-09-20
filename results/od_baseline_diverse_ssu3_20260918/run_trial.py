#!/usr/bin/env python3
"""Run one immutable ring-hash diverse-input case, through complete drainage.

ASU and OD use the core baseline implementations. Original Once is unchanged;
static/mild/aggressive reuse the archived candidate-pool extension explicitly.
All policies preserve NPU binding, per-NPU request order and in-Path FIFO.
"""
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / "results/diverse_data_ssu3_l3_20260916"
sys.path[:0] = [str(HERE), str(ROOT)]

from simulator.core import continuous_batch_sim as native
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint
from metrics import live_summary, overlap, summarize
from inputs.runners.run_baseline_npu32_stress import load_manifest, read_json, run_case, save_manifest, write_json
from inputs.runners.run_coflow_experiments import source_files
from simulator.core import sim

POLICIES = ("asu_baseline", "od_baseline", "once", "static", "mild", "aggressive")
WINDOWS = ((2000.0, 4000.0), (2000.0, 6000.0))
NUM_NPU, NUM_SSU, LAYERS, DISK_BW = 32, 3, 8, 40.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha(ROOT / name) for name in source_files()}


def extension_hashes():
    paths = (Path(__file__), HERE / "metrics.py", LEGACY / "policy.py",
             ROOT / "inputs/runners/run_baseline_npu32_stress.py", ROOT / "data")
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def load_pool_extension():
    """Import archived implementation by file path, never ambiguous sys.path."""
    name = "od_diverse_archived_pool_policy"
    spec = importlib.util.spec_from_file_location(name, LEGACY / "policy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def qos_snapshot(context):
    """Capture the configuration actually installed on each disk, not defaults."""
    return [dict(path_cirs_gib_s=list(q.path_cirs),
                 path_pirs_gib_s=["unlimited" if math.isinf(x) else x for x in q.path_pirs],
                 path_weights=list(q.path_weights), group_weights=list(q.group_weights),
                 category_paths_per_group=list(q.category_paths_per_group),
                 category_labels=list(q.category_labels))
            for q in context.qos_configs_by_ssu]


def check_baseline_ownership(policy, path_ids_by_ssu_npu, qos):
    """Audit actual completed I/O and actual initial QoS on every disk."""
    if policy not in ("asu_baseline", "od_baseline"):
        return dict(applicable=False, reason="shared-pool routing policy")
    from simulator.policies.baselines import od_npu_path_ids
    expected = (0,) * NUM_NPU if policy == "asu_baseline" else od_npu_path_ids(NUM_NPU)
    for disk in path_ids_by_ssu_npu:
        for npu, ids in enumerate(disk):
            assert ids == {expected[npu]}, (policy, npu, ids, expected[npu])
    if policy == "od_baseline":
        assert len(set(expected)) == NUM_NPU
        for q in qos:
            assert all(abs(q["path_cirs_gib_s"][p] - DISK_BW / NUM_NPU) < 1e-12 for p in expected)
            assert sum(x > 0 for x in q["path_cirs_gib_s"]) == NUM_NPU
            assert abs(sum(q["path_cirs_gib_s"]) - DISK_BW) < 1e-12
            assert all(q["path_pirs_gib_s"][p] == "unlimited" for p in expected)
            assert len({q["path_weights"][p] for p in expected}) == 1
            assert len(set(q["group_weights"])) == 1
    return dict(applicable=True, passed=True, npu_path_ids=list(expected),
                all_blocks_on_owner_path=True, exclusive_per_npu=policy == "od_baseline",
                configured_paths_per_ssu=len(set(expected)),
                od_cir_gib_s=DISK_BW / NUM_NPU if policy == "od_baseline" else None,
                idle_bandwidth_borrowing=True, runtime_cir_changes=0)


class PilotFinished(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--pilot", action="store_true", help="Stop only after the uncensored warm cohort and SSD service have drained")
    ap.add_argument("--smoke", action="store_true", help="Run only the first request on each NPU; not a measured experiment")
    args = ap.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    assert not (args.pilot and args.smoke)
    manifest_path = args.manifest.resolve()
    manifest_sha = sha(manifest_path)
    requests, meta = load_manifest(manifest_path)
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (NUM_NPU, NUM_SSU, LAYERS)
    assert meta["order"] == "random"
    assert meta["source_data_sha256"] == sha(ROOT / "data")
    assert meta["layout"] == sim.PLACEMENT_BLOCK_RING_HASH, "New comparison requires ring-hash inputs"
    assert all(q.arrival_time_ms == 0.0 for q in requests)
    assert all(len(q.placement) == 1 for q in requests)
    assert len({q.npu_id for q in requests}) == NUM_NPU
    scenario = meta["scenario_candidate"]
    target = HERE / "runs" / args.label
    target.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        first = {}
        for q in requests:
            first.setdefault(q.npu_id, q)
        requests = tuple(first.values())
        meta = dict(meta, smoke=True, original_full_input_fingerprint=meta["input_fingerprint"],
                    request_count=len(requests), input_fingerprint=continuous_batch_input_fingerprint(requests))
        from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
        meta["logical_input_fingerprint"] = logical_input_fingerprint(requests)
        save_manifest(target / "manifest.json.gz", requests, meta)
    else:
        (target / "manifest.json.gz").write_bytes(manifest_path.read_bytes())
        assert sha(target / "manifest.json.gz") == manifest_sha
    core_before, extension_before = source_hashes(), extension_hashes()
    expected = sum(len(q.placement[0]) * LAYERS for q in requests)
    extension = load_pool_extension()
    stats = extension.make_stats()
    managed = extension.install_policy(args.policy, stats) if args.policy in ("static", "mild", "aggressive") else nullcontext()
    backbone = "once" if args.policy in ("static", "mild", "aggressive") else args.policy
    observed = 0
    warm_saved = False
    busy = [[0.0] * NUM_SSU for _ in WINDOWS]
    bins = [[0.0] * 200 for _ in range(NUM_SSU)]
    per_npu_warm_service_ms = [[0.0] * NUM_NPU for _ in range(NUM_SSU)]
    all_service_ms = [0.0] * NUM_SSU
    block_counts = [[0] * NUM_NPU for _ in range(NUM_SSU)]
    seen_paths = [[set() for _ in range(NUM_NPU)] for _ in range(NUM_SSU)]
    actual_qos = []
    started = last_progress = time.perf_counter()
    record = dict(status="running", completed_simulation=False, policy=args.policy,
                  strategy=args.policy, simulator_strategy=backbone, assignment="fixed",
                  seed=meta["seed"], scenario=scenario, pilot=args.pilot, smoke=args.smoke,
                  pid=os.getpid(), started_utc=utc(), argv=sys.argv,
                  original_manifest=str(manifest_path), original_manifest_sha256=manifest_sha,
                  manifest_sha256=sha(target / "manifest.json.gz"),
                  input_fingerprint=meta["input_fingerprint"],
                  source_data_sha256=sha(ROOT / "data"), core_source_sha256=core_before,
                  extension_source_sha256=extension_before, expected_blocks=expected,
                  expected_requests=len(requests), windows_ms=WINDOWS,
                  placement=sim.PLACEMENT_BLOCK_RING_HASH, collector_interval_ms=5,
                  modeled_control_latency_ms=0)
    write_json(target / "command.json", record)
    callback = native._register_complete

    def preview(context):
        nonlocal warm_saved
        if args.smoke or warm_saved or context.current_time_ms < 4100:
            return
        cohort = [q for q in context.requests.values() if q.admitted and 2000 <= q.admission_time_ms < 4000]
        if not cohort or not all(q.completed for q in cohort):
            return
        for npu in context.npus:
            if npu.link_active_flow is not None and npu.link_active_flow.ssd_activation_time < 4000:
                return
            if any(f.ssd_activation_time < 4000 for f in npu.link_pending):
                return
        row = summarize(live_summary(context), requests, *WINDOWS[0])
        assert row["all_npus_active"]
        row.update(policy=args.policy, seed=meta["seed"], scenario=scenario,
                   SSD_GiB_s=[x * DISK_BW / 2000 for x in busy[0]],
                   observed_at_ms=context.current_time_ms,
                   warm_exact_full_run_continues=not args.pilot)
        write_json(target / "warm_preview.json", row)
        warm_saved = True
        print(json.dumps(dict(warm=True, policy=args.policy, scenario=scenario,
                              seed=meta["seed"], U=row["U_percent"], slo=row["slo"],
                              overload_percent=row["demand"]["per_disk_overload_percent"])), flush=True)
        if args.pilot:
            raise PilotFinished()

    def observe(context, flow):
        nonlocal observed, last_progress
        observed += 1
        if not actual_qos:
            actual_qos.extend(qos_snapshot(context))
        d, n = flow.disk_id, flow.npu_id
        seen_paths[d][n].add(flow.queue_id)
        block_counts[d][n] += 1
        a, z = flow.ssd_activation_time, flow.link_enqueue_time
        all_service_ms[d] += z - a
        for i, (left, right) in enumerate(WINDOWS):
            busy[i][d] += overlap(a, z, left, right)
        if a < 4000 and z > 2000:
            a1, z1 = max(a, 2000.0), min(z, 4000.0)
            per_npu_warm_service_ms[d][n] += z1 - a1
            first, last = int((a1 - 2000) // 10), min(199, int((z1 - 2000) // 10))
            for j in range(first, last + 1):
                bins[d][j] += overlap(a1, z1, 2000 + j * 10, 2010 + j * 10)
        ret = callback(context, flow)
        if observed % 10000 == 0:
            preview(context)
            now = time.perf_counter()
            if now - last_progress >= 15:
                p = dict(completed_blocks=observed, expected_blocks=expected,
                         simulation_ms=context.current_time_ms, wall_seconds=now - started,
                         completed_requests=context.completed_requests)
                write_json(target / "progress.json", p)
                print(json.dumps(p), flush=True)
                last_progress = now
        return ret

    try:
        with patch.object(native, "_register_complete", observe), managed:
            result = run_case(requests, meta, strategy=backbone, assignment="fixed", windows=WINDOWS)
        assert observed == expected
        assert all(abs(sum(bins[d]) - busy[0][d]) < 1e-7 for d in range(NUM_SSU))
        assert all(v <= z - a + 1e-7 for i, (a, z) in enumerate(WINDOWS) for v in busy[i])
        assert all(result["summary"]["invariants"].values())
        adapter = result["adapter_statistics"]
        assert adapter["reorder_calls"] == 0
        assert adapter["cir_write_events"] == []
        assert adapter["assignment_count"] == 0
        assert result["input_fingerprint"] == meta["input_fingerprint"]
        assert result["input_placement_fingerprint"] == result["execution_placement_fingerprint"]
        byid = {q.request_id: q for q in requests}
        rows = result["summary"]["request_metrics"]
        assert len(rows) == len(requests)
        for npu in range(NUM_NPU):
            actual = [r["request_id"] for r in sorted(rows, key=lambda x: x["admission_time_ms"])
                      if byid[r["request_id"]].npu_id == npu]
            assert actual == [q.request_id for q in requests if q.npu_id == npu]
        for row in rows:
            expected_compute = LAYERS * byid[row["request_id"]].load["per_layer_us"] / 1000
            assert abs(row["own_compute_ms"] - expected_compute) < 1e-8
        ownership = check_baseline_ownership(args.policy, seen_paths, actual_qos)
        assert all(q["path_cirs_gib_s"] == result["static_path_cirs_gib_s"] for q in actual_qos)
        analyses = []
        if not args.smoke:
            for i, (a, z) in enumerate(WINDOWS):
                row = summarize(result["summary"], requests, a, z)
                assert row["all_npus_active"]
                assert abs(row["U_percent"] - 100 * result["windows"][i]["mean_npu_utilization"]) < 1e-7
                row["SSD_GiB_s"] = [x * DISK_BW / (z - a) for x in busy[i]]
                row["SSD_busy_percent"] = [100 * x / (z - a) for x in busy[i]]
                analyses.append(row)
            full = summarize(result["summary"], requests, 0.0, result["summary"]["makespan_ms"], full=True)
            assert abs(full["U_percent"] - 100 * result["summary"]["fleet_npu_compute_utilization"]) < 1e-7
            full["SSD_GiB_s"] = [x * DISK_BW / full["end_ms"] for x in all_service_ms]
            full["SSD_busy_percent"] = [100 * x / full["end_ms"] for x in all_service_ms]
            analyses.append(full)
            if (target / "warm_preview.json").exists():
                p = read_json(target / "warm_preview.json")
                assert p["slo"] == analyses[0]["slo"] and abs(p["U_percent"] - analyses[0]["U_percent"]) < 1e-7
                assert p["demand"] == analyses[0]["demand"]
                assert all(abs(a - b) < 1e-7 for a, b in zip(p["SSD_GiB_s"], analyses[0]["SSD_GiB_s"]))
        result.update(experiment="od_baseline_diverse_ssu3_20260918", strategy=args.policy,
                      simulator_strategy=backbone, analysis=analyses, routing_statistics=stats,
                      actual_qos_by_ssu=actual_qos, baseline_ownership_audit=ownership,
                      observed_path_ids_by_ssu_npu=[[sorted(paths) for paths in disk] for disk in seen_paths],
                      completed_blocks_by_ssu_npu=block_counts,
                      warm_ssd_10ms_GiB_s=[[v * 4 for v in disk] for disk in bins],
                      warm_ssd_GiB_s_by_ssu_npu=[[v * DISK_BW / 2000 for v in disk]
                                                 for disk in per_npu_warm_service_ms])
        write_json(target / "result.json.gz", result)
        record.update(status="complete", completed_simulation=True,
                      result_sha256=sha(target / "result.json.gz"),
                      output_sha256=sha(target / "result.json.gz"),
                      completed_requests=len(rows), observed_blocks=observed,
                      checks=dict(all_invariants=True, FIFO_preserved=True, static_QoS=True,
                                  unchanged_L1_L2=True, same_input=True, placement_preserved=True,
                                  actual_QoS_matches_record=True, baseline_path_ownership=True),
                      baseline_ownership_audit=ownership,
                      metrics=[dict(start_ms=r["start_ms"], end_ms=r["end_ms"], U=r["U_percent"],
                                    slo=r["slo"], SSD_GiB_s=r["SSD_GiB_s"],
                                    overload_percent=r["demand"]["per_disk_overload_percent"])
                               for r in analyses])
    except PilotFinished:
        record.update(status="pilot_complete", completed_simulation=False,
                      observed_blocks=observed, warm_exact=True)
    except BaseException as exc:
        record.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        core_ok = source_hashes() == core_before
        extension_ok = extension_hashes() == extension_before
        manifest_ok = sha(manifest_path) == manifest_sha
        record.update(ended_utc=utc(), wall_seconds=time.perf_counter() - started,
                      core_unchanged=core_ok, extension_unchanged=extension_ok,
                      original_manifest_unchanged=manifest_ok, routing_statistics=stats)
        if not (core_ok and extension_ok and manifest_ok):
            record["status"] = "failed_source_changed"
        write_json(target / "command.json", record)
        assert core_ok and extension_ok and manifest_ok
    print(json.dumps({k: record[k] for k in ("status", "policy", "scenario", "seed", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
