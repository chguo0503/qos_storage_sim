#!/usr/bin/env python3
"""Freeze and run exactly six tripled-queue seed7 cases using the full runner."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent
ROOT = FOLLOWUP.parent.parents[1]
OUT = HERE / "repeated_queues"
sys.path.insert(0, str(ROOT))
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_baseline_npu32_stress import load_manifest, save_manifest, read_json, write_json
from run_multi_ssu_stall_experiments import input_demand
from run_shared_path_experiments import logical_input_fingerprint, summarize_slo
from run_coflow_experiments import source_files

REPEATS = 3
SEED = 7
WINDOWS = ((2000, 4000), (2000, 6000), (2000, 8000), (2000, 10000), (2000, 12000))
STRATEGIES = ("baseline", "once")
TIMEOUT = 3600
BASE_PATHS = {
    "fixed": FOLLOWUP / "inputs/raw176_three_l20_fixed_seed7.json.gz",
    "random": FOLLOWUP / "mixed_rebinding/inputs/raw176_extendedhot_random_seed7.json.gz",
    "ordered": FOLLOWUP / "mixed_rebinding/inputs/raw176_extendedhot_ordered_seed7.json.gz",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def certificate(requests):
    maxima = [[0.0] * 6 for _ in range(32)]
    link_maxima = [0.0] * 32
    for request in requests:
        compute_seconds = float(request.load["per_layer_us"]) / 1e6
        for layer in request.placement:
            work = [math.fsum(volume for disk, volume in layer if disk == s) for s in range(6)]
            rates = [volume / compute_seconds for volume in work]
            npu = request.npu_id
            maxima[npu] = [max(a, b) for a, b in zip(maxima[npu], rates)]
            link_maxima[npu] = max(link_maxima[npu], math.fsum(rates))
    disk = [math.fsum(row[s] for row in maxima) for s in range(6)]
    return dict(definition="sum_n max_request V[n,s]/C[n]; current admitted request only",
        per_ssu_upper_bound_gib_s=disk, per_npu_receive_upper_bound_gib_s=link_maxima,
        total_upper_bound_gib_s=math.fsum(link_maxima),
        passes=max(disk) < 40 and max(link_maxima) < 50 and math.fsum(link_maxima) < 240,
        caveat="Sufficient bound only; failed bound is not an observed overload. Actual event capacity is audited separately. Next-request L0 is excluded from nominal current demand.")


def canonical_original_id(request, mode):
    return int(request.request_id if mode == "fixed" else request.load["source_fixed_request_id"])


def build(mode):
    source = BASE_PATHS[mode]
    base, base_meta = load_manifest(source)
    assert len(base) == 1212 and base_meta["n_layers"] == 8
    assert base_meta["num_npu"] == 32 and base_meta["num_ssu"] == 6
    requests, mapping, lane_rows = [], [], []
    for npu in range(32):
        lane = sorted((r for r in base if r.npu_id == npu), key=lambda r: r.request_id)
        assert lane and all(r.arrival_time_ms == 0.0 for r in lane)
        for cycle in range(REPEATS):
            for base_position, original in enumerate(lane):
                position = cycle * len(lane) + base_position
                rid = npu * 1_000_000 + position
                source_id = canonical_original_id(original, mode)
                load = dict(original.load, request_id=rid, npu_id=npu, generation=position,
                    base_request_id=original.request_id, source_cycle=cycle,
                    source_base_position=base_position, source_base_npu=npu,
                    source_global_request_id=source_id)
                request = ContinuousBatchRequest.from_normalized(rid, npu, original.arrival_time_ms,
                                                               load, original.placement)
                assert request.placement is original.placement
                for key, value in original.load.items():
                    if key not in ("request_id", "npu_id", "generation"):
                        assert request.load[key] == value
                requests.append(request)
                mapping.append(dict(request_id=rid, npu_id=npu, rowposition=position,
                    source_cycle=cycle, base_request_id=original.request_id,
                    base_position=base_position, source_global_request_id=source_id,
                    original_request_id=original.load["original_request_id"],
                    canonical_identity=[source_id, cycle]))
        pure = REPEATS * 8 * math.fsum(float(r.load["per_layer_us"]) / 1000 for r in lane)
        counts = [REPEATS * sum(r.load["profile_index"] == p for r in lane) for p in range(4)]
        lane_rows.append(dict(npu_id=npu, base_request_count=len(lane), request_count=REPEATS * len(lane),
            profile_counts=counts, pure_compute_ms=pure))
    requests = tuple(requests)
    assert len(requests) == 3636 and len({r.request_id for r in requests}) == 3636
    assert min(row["pure_compute_ms"] for row in lane_rows) > 12000
    fp = continuous_batch_input_fingerprint(requests)
    label = f"raw176_{mode}_repeat3_seed7"
    proof = certificate(requests)
    meta = dict(experiment="raw176_repeated_full_queues_3_v1", label=label,
        case_id=f"{label}_{fp[:12]}", seed=SEED, num_npu=32, num_ssu=6, n_layers=8,
        family="raw", equal_176kib_blocks=True, blocks="exact", compute_scale_actual=1.0,
        profiles=base_meta["profiles"], profile_keys=base_meta["profile_keys"], source=base_meta["source"],
        source_data_sha256=base_meta["source_data_sha256"], request_count=len(requests),
        source_profile_counts=[792, 792, 792, 1260], per_npu_assignment=lane_rows,
        order=f"repeat_complete_{mode}_lane_3_times", assignment_mode=f"preserved_{mode}_binding",
        long_cards=20 if mode == "fixed" else None, short_cards=12 if mode == "fixed" else None,
        layout="preserved_base_request_placement", last_arrival_ms=0.0,
        input_fingerprint=fp, logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests, 32, 6), active_profile_rate_certificate=proof,
        load_within_disk_and_link_capacity=proof["passes"],
        measurement_windows_ms=[list(w) for w in WINDOWS], measurement_window_ms=list(WINDOWS[0]),
        base_manifest=str(source), base_manifest_sha256=sha(source),
        base_input_fingerprint=base_meta["input_fingerprint"], base_manifest_metadata=base_meta,
        repetitions=REPEATS, repetition_unit="each NPU's entire original finite queue",
        population_rule="New tripled input, not merely a new statistical window. Each original global identity has cycles0,1,2; random/ordered retain equal per-card populations. Fixed has different binding but same global repeated population.",
        placement_rule="Every copy reuses its source request's immutable complete placement; no new striping or C/V scaling",
        clock_rule="All copied arrivals remain0. Each lane advances by its own actual service progress; no global cycle barrier or artificial delay",
        slo_rule="Each selected window uses admissions in its half-open interval, followed to final completion; completion-admission<=1.5*8*C",
        construction_runner_sha256=sha(Path(__file__)))
    assert Counter(r.load["profile_index"] for r in requests) == {0: 792, 1: 792, 2: 792, 3: 1260}
    return requests, meta, mapping


def check_frozen(plan):
    for path, digest in plan["source_sha256"].items():
        assert sha(ROOT / path) == digest, f"frozen source changed: {path}"
    for item in plan["inputs"]:
        assert sha(item["manifest"]) == item["manifest_sha256"]
        assert sha(item["base_manifest"]) == item["base_manifest_sha256"]
        assert sha(item["mapping"]) == item["mapping_sha256"]


def prepare():
    plan_path = OUT / "plan.json"
    if plan_path.exists():
        plan = read_json(plan_path)
        check_frozen(plan)
        return plan
    OUT.mkdir(parents=True, exist_ok=True)
    items, built = [], {}
    for mode in ("fixed", "random", "ordered"):
        requests, metadata, mapping = build(mode)
        label = metadata["label"]
        manifest = OUT / "inputs" / f"{label}.json.gz"
        mapping_path = OUT / "mappings" / f"{label}.json.gz"
        write_json(mapping_path, dict(schema_version=1, label=label,
            identity_definition="(source_global_request_id,source_cycle) references one immutable original fixed-source request copy",
            base_manifest=metadata["base_manifest"], base_manifest_sha256=metadata["base_manifest_sha256"], rows=mapping))
        metadata["identity_mapping"] = str(mapping_path)
        metadata["identity_mapping_sha256"] = sha(mapping_path)
        save_manifest(manifest, requests, metadata)
        loaded, loaded_meta = load_manifest(manifest)
        assert continuous_batch_input_fingerprint(loaded) == metadata["input_fingerprint"]
        snapshot = OUT / "base_manifests" / f"{mode}.json.gz"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(metadata["base_manifest"], snapshot)
        assert sha(snapshot) == metadata["base_manifest_sha256"]
        built[mode] = requests
        items.append(dict(mode=mode, label=label, seed=SEED, manifest=str(manifest),
            manifest_sha256=sha(manifest), input_fingerprint=metadata["input_fingerprint"],
            base_manifest=metadata["base_manifest"], base_manifest_sha256=metadata["base_manifest_sha256"],
            mapping=str(mapping_path), mapping_sha256=sha(mapping_path), request_count=len(requests),
            min_per_npu_pure_compute_ms=min(row["pure_compute_ms"] for row in metadata["per_npu_assignment"]),
            per_ssu_static_max_gib_s=metadata["active_profile_rate_certificate"]["per_ssu_upper_bound_gib_s"]))
    signatures = {}
    populations = {}
    for mode, requests in built.items():
        mode_map = {}
        populations[mode] = {}
        for request in requests:
            uid = (request.load["source_global_request_id"], request.load["source_cycle"])
            assert uid not in mode_map
            mode_map[uid] = (request.load["seq_len_k"], request.load["nql"], request.load["per_layer_us"],
                            request.load["per_layer_kv_gb"], request.arrival_time_ms, request.placement)
            populations[mode].setdefault(request.npu_id, []).append(uid)
        signatures[mode] = mode_map
    assert signatures["fixed"] == signatures["random"] == signatures["ordered"]
    assert all(sorted(populations["random"][n]) == sorted(populations["ordered"][n]) for n in range(32))
    source_paths = {ROOT / name for name in source_files()}
    source_paths.update((ROOT / "data", ROOT / "run_baseline_npu32_stress.py",
                         ROOT / "run_multi_ssu_stall_experiments.py", Path(__file__)))
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(source_paths)}
    for path in sorted(source_paths):
        target = OUT / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(path) == sha(target)
    audit = dict(all_checks_pass=True, global_3636_copy_identities_C_V_arrival_placement_match_all_three_modes=True,
        random_ordered_exact_population_match_all_32_npus=True,
        all_original_fields_except_position_id_generation_retained=True,
        all_per_npu_pure_compute_above_12000=True, fixed_binding_intentionally_differs_from_random_ordered=True,
        inputs=items)
    write_json(OUT / "input_audit.json", audit)
    plan = dict(schema_version=1, created_utc=utc(), predeclared=True, seed=SEED, repetitions=REPEATS,
        strategies=list(STRATEGIES), windows_ms=[list(w) for w in WINDOWS], subprocess_timeout_seconds=TIMEOUT,
        workers=6, input_audit=str(OUT / "input_audit.json"), input_audit_sha256=sha(OUT / "input_audit.json"),
        inputs=items, jobs=[dict(input=item, strategy=s) for item in items for s in STRATEGIES],
        source_sha256=hashes, python_executable=sys.executable, python_version=sys.version,
        platform=platform.platform(), cpu_count=os.cpu_count(),
        study_scope="Exactly six full finite runs, no steady-state stop, no added seeds. Windows may include phase drift across per-NPU cycle boundaries.")
    assert len(plan["jobs"]) == 6
    write_json(plan_path, plan)
    check_frozen(plan)
    print(json.dumps(dict(prepared=True, inputs=items, jobs=6, input_audit_pass=True)), flush=True)
    return plan


def run_job(job, plan):
    item, strategy = job["input"], job["strategy"]
    directory = OUT / "runs" / item["label"] / strategy
    directory.mkdir(parents=True, exist_ok=True)
    check_frozen(plan)
    prior = directory / "command.json"
    if prior.exists():
        record = read_json(prior)
        assert record.get("status") == "complete", "preserving incomplete prior command; inspect before resuming"
        raw = read_json(record["output"])
        assert raw["input_fingerprint"] == item["input_fingerprint"] and raw["strategy"] == strategy
        assert all(raw["summary"]["invariants"].values())
        return record
    command = [sys.executable, "-B", str(ROOT / "run_baseline_npu32_stress.py"),
        "--manifest", item["manifest"], "--strategy", strategy, "--assignment", "fixed"]
    for start, end in WINDOWS:
        command.extend(("--window", f"{start}:{end}"))
    command.extend(("--output", str(directory)))
    record = dict(label=item["label"], mode=item["mode"], strategy=strategy, status="starting",
        command=command, cwd=str(ROOT), manifest=item["manifest"], input_sha256=item["manifest_sha256"],
        input_fingerprint=item["input_fingerprint"], source_sha256=plan["source_sha256"],
        started_utc=utc(), simultaneous_workers=6, timeout_seconds=TIMEOUT)
    start = time.perf_counter()
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(pid=process.pid, status="running")
        write_json(prior, record)
        print(json.dumps(dict(started=item["label"], strategy=strategy, pid=process.pid)), flush=True)
        try:
            code = process.wait(timeout=TIMEOUT)
            status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code, status = process.returncode, "timeout"
    record.update(returncode=code, status=status, wall_seconds=time.perf_counter()-start, ended_utc=utc())
    if status == "complete":
        try:
            outputs = [p for p in directory.glob("*.json.gz") if ".failure." not in p.name]
            assert len(outputs) == 1
            raw = read_json(outputs[0])
            assert raw["input_fingerprint"] == item["input_fingerprint"] and raw["strategy"] == strategy
            assert all(raw["summary"]["invariants"].values())
            assert [(w["start_ms"], w["end_ms"]) for w in raw["windows"]] == list(WINDOWS)
            assert raw["summary"]["request_count"] == 3636
            assert all(plan["source_sha256"][key] == value for key, value in raw["core_and_policy_sha256"].items())
            check_frozen(plan)
            requests, _ = load_manifest(item["manifest"])
            write_json(directory / "window_slo.json", dict(input_fingerprint=item["input_fingerprint"],
                raw_sha256=sha(outputs[0]), windows=[dict(start_ms=a, end_ms=b,
                    slo=summarize_slo(raw["summary"], requests, start_ms=a, end_ms=b)) for a,b in WINDOWS]))
            record.update(output=str(outputs[0]), output_sha256=sha(outputs[0]),
                invariants_pass=True, source_hashes_pass=True,
                window_utilization=[dict(start_ms=w["start_ms"], end_ms=w["end_ms"],
                    utilization=w["mean_npu_utilization"], all_active=w["all_npus_active_whole_window"])
                    for w in raw["windows"]])
        except Exception as error:
            record.update(status="audit_failed", error=repr(error))
    write_json(prior, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, choices=(6,), default=6)
    args = parser.parse_args()
    plan = prepare()
    if not args.run:
        return
    records = []
    write_json(OUT / "run_status.json", dict(status="running", started_utc=utc(), total=6, finished=0, rows=[]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job, plan) for job in plan["jobs"]]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_json(OUT / "run_status.json", dict(status="running" if len(records)<6 else "finished",
                total=6, finished=len(records), rows=records))
            print(json.dumps({key:record[key] for key in ("label", "strategy", "status", "wall_seconds")}), flush=True)
    check_frozen(plan)
    assert len(records) == 6 and all(row["status"] == "complete" for row in records)
    write_json(OUT / "completion.json", dict(completed_utc=utc(), complete=6, planned=6,
        all_returncodes_zero=True, all_sources_unchanged=True, rows=records))


if __name__ == "__main__":
    main()
