#!/usr/bin/env python3
"""Run a frozen mixed-role population with ASU, OD or original full-pool Once.

Both random and ordered inputs run through complete drainage. Capacity
violations and incomplete active-window coverage are findings, not failures.
Neither request arrival times nor computation is changed to enforce underload.
The optional ASU/OD compatibility routes retain the same validation. No
candidate-pool extension is imported or installed for the Once control.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from simulator.core import continuous_batch_sim as native
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint
from metrics import live_summary, overlap, summarize
from inputs.runners.run_baseline_npu32_stress import load_manifest, read_json, run_case, save_manifest, write_json
from inputs.runners.run_coflow_experiments import source_files
from simulator.core import sim

POLICIES = ("asu_baseline", "od_baseline", "once")
WINDOWS = ((2000.0, 4000.0), (2000.0, 6000.0), (4000.0, 8000.0))
NUM_NPU, NUM_SSU, LAYERS, DISK_BW = 32, 3, 8, 40.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha(ROOT / name) for name in source_files()}


def extension_hashes():
    paths = (Path(__file__), HERE / "metrics.py",
             ROOT / "inputs/runners/run_baseline_npu32_stress.py", ROOT / "data")
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


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
            assert not ids or ids == {expected[npu]}, (policy, npu, ids, expected[npu])
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


def check_prefetch(summary, requests):
    """Verify actual cross-request L0 release against the predecessor's L7."""
    assert summary['n_layers'] == LAYERS and summary['batch_size'] == 1
    assert summary['cross_request_layer0_prefetch'] is True
    byid = {r['request_id']: r for r in summary['request_metrics']}
    batches = {b['member_request_ids'][0]: b for b in summary['microbatch_metrics']}
    checked = 0; maximum_error = 0.0
    for npu in range(NUM_NPU):
        ids = [q.request_id for q in requests if q.npu_id == npu]
        assert not byid[ids[0]]['layer0_cross_request_prefetched']
        for previous_id, rid in zip(ids, ids[1:]):
            r, previous = byid[rid], byid[previous_id]
            assert r['layer0_cross_request_prefetched']
            error = abs(r['layer0_io_start_time_ms'] - batches[previous_id]['layer_metrics'][-1]['compute_start_ms'])
            assert error < 1e-7
            assert abs(r['admission_time_ms'] - previous['completion_time_ms']) < 1e-7
            assert r['layer0_io_start_time_ms'] <= r['admission_time_ms'] + 1e-7
            maximum_error = max(maximum_error, error); checked += 1
    assert checked == len(requests) - NUM_NPU == summary['cross_request_layer0_prefetches']
    return dict(enabled=True, expected_cross_request_prefetches=len(requests)-NUM_NPU,
                observed_cross_request_prefetches=checked, all_releases_at_predecessor_last_compute_start=True,
                maximum_release_timestamp_error_ms=maximum_error,
                physical_L0_service_included=True, reference_demand_double_counts_L0=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--smoke", action="store_true", help="Run only the first request on each NPU; not a measured experiment")
    ap.add_argument("--validate-only", action="store_true", help="Check frozen inputs and print configuration without creating a run")
    args = ap.parse_args()
    assert Path(args.label).name == args.label and args.label not in (".", "..")
    manifest_path = args.manifest.resolve()
    manifest_sha = sha(manifest_path)
    requests, meta = load_manifest(manifest_path)
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (NUM_NPU, NUM_SSU, LAYERS)
    assert isinstance(meta["order"], str) and meta["order"]
    if "source_data_sha256" in meta:
        assert meta["source_data_sha256"] == sha(ROOT / "data")
    assert meta["layout"] == sim.PLACEMENT_BLOCK_RING_HASH, "New comparison requires ring-hash inputs"
    assert all(q.arrival_time_ms == 0.0 for q in requests)
    assert all(len(q.placement) == 1 for q in requests)
    assert {q.npu_id for q in requests} == set(range(NUM_NPU))
    assert len({q.request_id for q in requests}) == len(requests)
    assert meta["input_fingerprint"] == continuous_batch_input_fingerprint(requests)
    assert meta["equal_176kib_blocks"] is True
    assert meta.get("disk_bw_gib_s", DISK_BW) == DISK_BW
    assert len({q.load['original_request_id'] for q in requests}) == len(requests)
    for q in requests:
        assert q.load['per_layer_us'] > 0 and math.isfinite(q.load['per_layer_us'])
        assert q.placement[0] and all(abs(v - 176 * 1024 / 2**30) < 1e-15 for _, v in q.placement[0])
        assert abs(math.fsum(v for _, v in q.placement[0]) - q.load['per_layer_kv_gb']) < 1e-10
        assert all(k in q.load for k in ('role', 'category', 'seq_len_k', 'nql'))
        assert all(disk == sim.block_ring_hash_disk_id(int(q.load['original_request_id']), i, NUM_SSU)
                   for i, (disk, _) in enumerate(q.placement[0])), q.request_id
    input_audit = dict(request_count=len(requests), blocks=sum(len(q.placement[0])*LAYERS for q in requests),
        per_npu_request_counts=[sum(q.npu_id == n for q in requests) for n in range(NUM_NPU)],
        role_counts=dict(Counter(q.load['role'] for q in requests)),
        per_npu_roles=[sorted({q.load['role'] for q in requests if q.npu_id == n}) for n in range(NUM_NPU)],
        pure_compute_ms_per_npu=[math.fsum(LAYERS*q.load['per_layer_us']/1000 for q in requests if q.npu_id == n) for n in range(NUM_NPU)],
        constructed_requests=sum(bool(q.load.get('constructed_profile', False)) for q in requests),
        synthetic_metadata=meta.get('synthetic', meta.get('constructed_profile', 'unspecified')),
        source_claim='Manifest values are used unchanged; original-data authenticity is not inferred',
        input_fingerprint=meta['input_fingerprint'], manifest_sha256=manifest_sha)
    if args.validate_only:
        print(json.dumps(dict(valid=True, policy=args.policy, windows_ms=WINDOWS, input_audit=input_audit)), flush=True)
        return
    scenario = meta.get("scenario_candidate", "mixed_underload_search")
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
    stats = dict(calls=0, io_blocks=0, extension_installed=False)
    backbone = args.policy
    from simulator.policies import once as shared_path_once
    from simulator.adapters import shared_path as shared_path_sim_adapter
    original_once_function = shared_path_once.once_path_ids
    original_adapter_function = shared_path_sim_adapter.shared_path_adapter
    assert original_once_function.__module__ == 'shared_path_once'
    assert original_adapter_function.__module__ == 'shared_path_sim_adapter'
    observed = 0
    warm_saved = False
    busy = [[0.0] * NUM_SSU for _ in WINDOWS]
    service_by_window_ssu_npu = [[[0.0] * NUM_NPU for _ in range(NUM_SSU)] for _ in WINDOWS]
    bins = [[0.0] * 200 for _ in range(NUM_SSU)]
    per_npu_warm_service_ms = [[0.0] * NUM_NPU for _ in range(NUM_SSU)]
    all_service_ms = [0.0] * NUM_SSU
    block_counts = [[0] * NUM_NPU for _ in range(NUM_SSU)]
    seen_paths = [[set() for _ in range(NUM_NPU)] for _ in range(NUM_SSU)]
    actual_qos = []
    started = last_progress = time.perf_counter()
    record = dict(status="running", completed_simulation=False, policy=args.policy,
                  strategy=args.policy, simulator_strategy=backbone, assignment="fixed",
                  seed=meta["seed"], scenario=scenario, order=meta['order'], pilot=False, smoke=args.smoke,
                  pid=os.getpid(), started_utc=utc(), argv=sys.argv,
                  original_manifest=str(manifest_path), original_manifest_sha256=manifest_sha,
                  manifest_sha256=sha(target / "manifest.json.gz"),
                  input_fingerprint=meta["input_fingerprint"],
                  source_data_sha256=sha(ROOT / "data"), core_source_sha256=core_before,
                  extension_source_sha256=extension_before, expected_blocks=expected,
                  expected_requests=len(requests), windows_ms=WINDOWS,
                  placement=sim.PLACEMENT_BLOCK_RING_HASH, collector_interval_ms=5,
                  modeled_control_latency_ms=0,
                  input_audit=input_audit,
                  experiment_conditions=dict(num_npu=NUM_NPU, num_ssu=NUM_SSU, disk_GiB_s=DISK_BW,
                      n_layers=LAYERS, batch_size=1, all_arrivals_at_zero=True,
                      manifest_compute_and_volume_unchanged=True,
                      constructed_requests=input_audit['constructed_requests'],
                      admission_reference='8 * own per-layer compute',
                      cross_request_layer0_prefetch=True,
                      capacity_violation_is_result_not_failure=True))
    if args.policy == 'once':
        record['once_control'] = dict(candidate_pool='unchanged full category-legal pool',
            routing='shared_path_once.once_path_ids', candidate_pool_extension_installed=False,
            collector_interval_ms=5, runtime_CIR_changes=False)
    write_json(target / "command.json", record)
    callback = native._register_complete

    def preview(context):
        nonlocal warm_saved
        if args.smoke or warm_saved or context.current_time_ms < 4100:
            return
        cohort = [q for q in context.requests.values() if q.admitted and 2000 <= q.admission_time_ms < 4000]
        if not cohort or not all(q.completed for q in cohort):
            return
        if any(q.admitted and q.admission_time_ms < 4000 and not q.completed for q in context.requests.values()):
            return
        for npu in context.npus:
            if npu.link_active_flow is not None and npu.link_active_flow.ssd_activation_time < 4000:
                return
            if any(f.ssd_activation_time < 4000 for f in npu.link_pending):
                return
        row = summarize(live_summary(context), requests, *WINDOWS[0])
        row.update(policy=args.policy, seed=meta["seed"], scenario=scenario, order=meta['order'],
                   SSD_GiB_s=[x * DISK_BW / 2000 for x in busy[0]],
                   observed_at_ms=context.current_time_ms,
                   warm_exact_full_run_continues=True)
        write_json(target / "warm_preview.json", row)
        warm_saved = True
        print(json.dumps(dict(warm=True, policy=args.policy, scenario=scenario,
                              seed=meta["seed"], U=row["U_percent"], slo=row["slo"],
                              overload_percent=row["demand"]["per_disk_overload_percent"])), flush=True)

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
            duration = overlap(a, z, left, right)
            busy[i][d] += duration
            service_by_window_ssu_npu[i][d][n] += duration
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
        with patch.object(native, "_register_complete", observe):
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
        assert {r['request_id'] for r in rows} == set(byid)
        assert result['summary']['submitted_blocks'] == result['summary']['completed_blocks'] == expected
        expected_by_disk_npu = [[0] * NUM_NPU for _ in range(NUM_SSU)]
        for q in requests:
            for disk, count in Counter(d for d, _ in q.placement[0]).items():
                expected_by_disk_npu[disk][q.npu_id] += count * LAYERS
        assert block_counts == expected_by_disk_npu
        for npu in range(NUM_NPU):
            actual = [r["request_id"] for r in sorted(rows, key=lambda x: x["admission_time_ms"])
                      if byid[r["request_id"]].npu_id == npu]
            assert actual == [q.request_id for q in requests if q.npu_id == npu]
        for row in rows:
            expected_compute = LAYERS * byid[row["request_id"]].load["per_layer_us"] / 1000
            assert abs(row["own_compute_ms"] - expected_compute) < 1e-8
        ownership = check_baseline_ownership(args.policy, seen_paths, actual_qos)
        assert all(q["path_cirs_gib_s"] == result["static_path_cirs_gib_s"] for q in actual_qos)
        ownership_counts = {(r['ssu_id'], r['npu_id'], r['path_id']): r['blocks']
                            for r in adapter['routed_blocks_by_ssu_npu_path']}
        if ownership['applicable']:
            assert ownership_counts == {(d, n, ownership['npu_path_ids'][n]): block_counts[d][n]
                                        for d in range(NUM_SSU) for n in range(NUM_NPU) if block_counts[d][n]}
        if args.policy == 'once':
            assert adapter['strategy'] == 'once' and adapter['routing_calls'] > 0
            assert adapter['collector_interval_ms'] == 5
            assert adapter['max_snapshot_age_ms'] <= 5 + 1e-7
            assert shared_path_once.once_path_ids is original_once_function
            assert shared_path_sim_adapter.shared_path_adapter is original_adapter_function
            assert stats['calls'] == stats['io_blocks'] == 0
            assert ownership['applicable'] is False and ownership_counts == {}
            record['once_control']['all_checks_passed'] = True
        prefetch_audit = check_prefetch(result['summary'], requests)
        analyses = []
        if not args.smoke:
            for i, (a, z) in enumerate(WINDOWS):
                row = summarize(result["summary"], requests, a, z)
                assert abs(row["U_percent"] - 100 * result["windows"][i]["mean_npu_utilization"]) < 1e-7
                row.update(order=meta['order'], policy=args.policy, seed=meta['seed'], scenario=scenario)
                row["SSD_GiB_s"] = [x * DISK_BW / (z - a) for x in busy[i]]
                row["SSD_busy_percent"] = [100 * x / (z - a) for x in busy[i]]
                row["SSD_GiB_s_by_ssu_npu"] = [[v * DISK_BW / (z-a) for v in disk]
                                                for disk in service_by_window_ssu_npu[i]]
                analyses.append(row)
            full = summarize(result["summary"], requests, 0.0, result["summary"]["makespan_ms"], full=True)
            full.update(order=meta['order'], policy=args.policy, seed=meta['seed'], scenario=scenario)
            assert abs(full["U_percent"] - 100 * result["summary"]["fleet_npu_compute_utilization"]) < 1e-7
            full["SSD_GiB_s"] = [x * DISK_BW / full["end_ms"] for x in all_service_ms]
            full["SSD_busy_percent"] = [100 * x / full["end_ms"] for x in all_service_ms]
            analyses.append(full)
            if (target / "warm_preview.json").exists():
                p = read_json(target / "warm_preview.json")
                assert p["slo"] == analyses[0]["slo"] and abs(p["U_percent"] - analyses[0]["U_percent"]) < 1e-7
                assert p["demand"] == analyses[0]["demand"]
                assert p['request_statistics'] == analyses[0]['request_statistics']
                assert all(abs(a - b) < 1e-7 for a, b in zip(p["SSD_GiB_s"], analyses[0]["SSD_GiB_s"]))
        result.update(experiment="od_mixed_underload_search_20260919", strategy=args.policy,
                      input_audit=input_audit,
                      order=meta['order'], scenario=scenario, prefetch_audit=prefetch_audit,
                      simulator_strategy=backbone, analysis=analyses, routing_statistics=stats,
                      actual_qos_by_ssu=actual_qos, baseline_ownership_audit=ownership,
                      observed_path_ids_by_ssu_npu=[[sorted(paths) for paths in disk] for disk in seen_paths],
                      completed_blocks_by_ssu_npu=block_counts,
                      warm_ssd_10ms_GiB_s=[[v * 4 for v in disk] for disk in bins],
                      warm_ssd_GiB_s_by_ssu_npu=[[v * DISK_BW / 2000 for v in disk]
                                                 for disk in per_npu_warm_service_ms])
        if args.policy == 'once':
            result['once_control'] = record['once_control']
        write_json(target / "result.json.gz", result)
        record.update(status="complete", completed_simulation=True,
                      result_sha256=sha(target / "result.json.gz"),
                      output_sha256=sha(target / "result.json.gz"),
                      completed_requests=len(rows), observed_blocks=observed,
                      checks=dict(all_invariants=True, FIFO_preserved=True, static_QoS=True,
                                  unchanged_L1_L2=True, same_input=True, placement_preserved=True,
                                  actual_QoS_matches_record=True, baseline_path_ownership=True,
                                  block_and_request_conservation=True, cross_request_prefetch=True),
                      baseline_ownership_audit=ownership,
                      prefetch_audit=prefetch_audit,
                      metrics=[dict(start_ms=r["start_ms"], end_ms=r["end_ms"], U=r["U_percent"],
                                    slo=r["slo"], SSD_GiB_s=r["SSD_GiB_s"],
                                    all_npus_active=r['all_npus_active'],
                                    strict_underload=r['demand']['strict_underload_all_disks'],
                                    at_or_above_capacity_percent=r['demand']['per_disk_at_or_above_capacity_percent'],
                                    overload_percent=r["demand"]["per_disk_overload_percent"])
                               for r in analyses])
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
