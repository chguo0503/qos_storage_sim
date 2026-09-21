#!/usr/bin/env python3
"""Extend the archived formula AB experiment without changing its data plane.

Separate processes only. The archived flat runtime and exact-tail adapter are
loaded by absolute repository-relative paths. This runner changes topology and
OD static path assignment, not event ordering, byte sizes, or prefetch behavior.
All capacities displayed as GB/s are decimal; native volumes/rates are GiB.
OD has unlimited queue depth, matching the original formula comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LEGACY = PROJECT / "results/template_od_baseline_20260919/formula_sensitivity"
FROZEN = LEGACY / "frozen_source"
ORIGINAL = PROJECT / "template/qos_experiments_20260919/03_continuous_underload/formula_ab/original_recovered"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(FROZEN))
import sim
import continuous_batch_sim as native
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from fixed_total_compat import exact_tail_adapter, validate_blocks
from run_baseline_npu32_stress import load_manifest

LAYERS = 8
GB_TO_GIB = 1e9 / 2**30


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


od = load_module("formula32_od_policy_snapshot", LEGACY / "od_policy_snapshot.py")
audit = load_module("formula32_archived_audit", ORIGINAL / "audit_results.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save(path, value):
    """Atomic JSON; deterministic gzip makes identical manifests byte-identical."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.name.endswith(".gz"):
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    stream.write("\n")
            raw.flush()
            os.fsync(raw.fileno())
    else:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def validate_config(config):
    config = dict(config)
    config.setdefault("num_npu", 32)
    config.setdefault("mode", "random")
    config.setdefault("window_ms", [2000, 4000])
    config.setdefault("disk_gb_s", 40.0)
    config.setdefault("npu_gb_s", 50.0)
    for key in ("num_npu", "ssu", "seed"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    if not 1 <= config["num_npu"] <= 256 or config["ssu"] < 1:
        raise ValueError("Require 1..256 NPUs and at least one SSU")
    if config["disk_gb_s"] != 40 or config["npu_gb_s"] != 50:
        raise ValueError("This reproduction fixes disk=40 and NPU link=50 decimal GB/s")
    if config.get("n_layers", 8) != 8 or config.get("batch_size", 1) != 1:
        raise ValueError("This reproduction fixes eight layers and batch_size=1")
    if config.get("slo_alpha", 1.5) != 1.5:
        raise ValueError("This comparison fixes TTFT SLO multiplier=1.5")
    start, end = map(float, config["window_ms"])
    if not all(math.isfinite(v) for v in (start, end, config["horizon_ms"])) or not 0 <= start < end:
        raise ValueError("Invalid warm window or horizon")
    if config["horizon_ms"] < end:
        raise ValueError("Ideal compute horizon must cover the complete warm window")
    if config["mode"] == "random":
        counts = config["input_counts"]
        if len(counts) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError("Random input_counts needs two positive integers")
    elif config["mode"] == "fixed":
        if not 0 <= config["n_A"] <= config["num_npu"]:
            raise ValueError("Invalid fixed A-card count")
    else:
        raise ValueError("Only original random and fixed order are supported")
    for role in ("A", "B"):
        p = config["profile_" + role]
        prefix = p["ssd_prefix_tokens"]
        if not isinstance(prefix, int) or prefix <= 0:
            raise ValueError("Exact prefix tokens must be positive integers")
        if p["total_tokens"] - p["nql"] != prefix:
            raise ValueError("total_tokens - miss must equal exact SSD prefix tokens")
        if p["total_length_k"] * 1024 != p["total_tokens"]:
            raise ValueError("Total length K means 1024 tokens")
        if not math.isfinite(p["compute_us"]) or p["compute_us"] <= 0:
            raise ValueError("Compute time must be positive finite microseconds")
        if not math.isclose(p["read_gib"], prefix * 1408 / 2**30, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Exact prefix bytes differ from profile volume")
        if not math.isclose(p["B_gib_s"], p["read_gib"] * 1e6 / p["compute_us"], rel_tol=1e-12):
            raise ValueError("Profile B must equal V/C in GiB/s")
    return config


def build(config):
    """Original generator with range(8) replaced by config.num_npu only."""
    profiles = {r: config["profile_" + r] for r in ("A", "B")}
    s, seed, horizon = config["ssu"], config["seed"], config["horizon_ms"]
    requests, metadata, decks = [], {}, []
    for npu in range(config["num_npu"]):
        if config["mode"] == "fixed":
            role = "A" if npu < config["n_A"] else "B"
            deck = [role] * (math.ceil(horizon / (8 * profiles[role]["compute_us"] / 1000)) + 1)
        else:
            na, nb = config["input_counts"]
            cycle_ms = 8 * (na * profiles["A"]["compute_us"] + nb * profiles["B"]["compute_us"]) / 1000
            cycles = math.ceil(horizon / cycle_ms) + 1
            deck = ["A"] * (cycles * na) + ["B"] * (cycles * nb)
            random.Random(seed + 100003 * npu).shuffle(deck)
        decks.append(deck)
        if len(deck) >= 1_000_000:
            raise ValueError("Request ID stride exhausted")
        for generation, role in enumerate(deck):
            p = profiles[role]
            rid = npu * 1_000_000 + generation
            prefix, disk_gib, layer = p["ssd_prefix_tokens"], [0.0] * s, []
            for block in range(math.ceil(prefix / 128)):
                disk = sim.block_ring_hash_disk_id(rid, block, s)
                size = min(128, prefix - 128 * block) * 1408 / 2**30
                layer.append((disk, size))
                disk_gib[disk] += size
            assert math.isclose(sum(disk_gib), p["read_gib"], abs_tol=1e-12)
            load = dict(request_id=rid, npu_id=npu, generation=generation, original_request_id=rid,
                total_tokens=p["total_tokens"], seq_len_k=p["total_length_k"], nql=p["nql"],
                role="L" if role == "A" else "S", profile_group=role,
                ssd_prefix_tokens=prefix, category=sim.classify_request(p["total_length_k"], p["nql"]),
                per_layer_us=p["compute_us"], per_layer_kv_gb=p["read_gib"],
                required_bw_input_gbps=p["B_gib_s"], arrival_time=0.0, arrival_ms=0.0, initial=True,
                constructed_profile=p["constructed_profile"], profile_construction=p["profile_construction"],
                original_compute_us=p["compute_us"], padding_gib_per_layer=0.0, disk_gib=disk_gib)
            requests.append(native.ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, (tuple(layer),)))
            metadata[rid] = load
    return tuple(requests), metadata, decks


def freeze_manifest(path, requests, config, decks):
    placements, indices, rows = [], {}, []
    for request in requests:
        if request.placement not in indices:
            indices[request.placement] = len(placements)
            placements.append(request.placement)
        rows.append(dict(request_id=request.request_id, npu_id=request.npu_id,
            arrival_time_ms=request.arrival_time_ms, load=dict(request.load),
            placement_index=indices[request.placement]))
    metadata = dict(config=config, num_npu=config["num_npu"], num_ssu=config["ssu"], n_layers=LAYERS,
        seed=config["seed"], disk_bw_gib_s=40 * GB_TO_GIB, npu_bw_gib_s=50 * GB_TO_GIB,
        placement="block_ring_hash", physical_identity="npu_id * 1000000 + queue_position",
        exact_partial_tail=True, arrival_ms=0.0, queues=decks,
        count_rule="ceil(horizon / (8*(nA*C_A+nB*C_B)))+1 complete AB decks per card; independently shuffled",
        queue_depth_limit=None)
    save(path, dict(schema_version=1, metadata=metadata, placements=placements, requests=rows,
        input_fingerprint=native.continuous_batch_input_fingerprint(requests)))
    restored, _ = load_manifest(path)
    assert native.continuous_batch_input_fingerprint(restored) == native.continuous_batch_input_fingerprint(requests)


def source_hashes():
    paths = list(FROZEN.glob("*.py")) + [Path(__file__), LEGACY / "od_policy_snapshot.py", ORIGINAL / "audit_results.py"]
    # OD snapshot.qos_config imports only the current StaticQoSConfig definition.
    # It is converted immediately to the archived dataclass, never used to run events.
    for module in tuple(sys.modules.values()):
        name, filename = getattr(module, "__name__", ""), getattr(module, "__file__", None)
        if name.startswith("simulator.") and filename and filename.endswith(".py"):
            paths.append(Path(filename))
    return {str(p.resolve().relative_to(PROJECT)): sha(p) for p in sorted(set(paths))}


def nominal_event_intervals(summary, metadata, ssu):
    """Exact request-switch events; demand excludes extra next-request L0 term."""
    events = defaultdict(lambda: [0.0] * ssu)
    events[0.0]
    events[summary["makespan_ms"]]
    for row in summary["request_metrics"]:
        meta = metadata[row["request_id"]]
        rates = [v / GB_TO_GIB / (meta["per_layer_us"] / 1e6) for v in meta["disk_gib"]]
        for time_ms, sign in ((row["admission_time_ms"], 1), (row["completion_time_ms"], -1)):
            for disk, rate in enumerate(rates):
                events[time_ms][disk] += sign * rate
    points, rates, intervals = sorted(events), [0.0] * ssu, []
    for index, t in enumerate(points[:-1]):
        rates = [a + b for a, b in zip(rates, events[t])]
        rates = [0.0 if abs(v) < 1e-9 else v for v in rates]
        assert min(rates) > -1e-7
        intervals.append(dict(start_ms=t, end_ms=points[index + 1], demand_decimal_gb_s_by_ssu=list(rates)))
    return dict(definition="Sum active request V_i,d/C_i over admission-to-completion; not physical service; no extra cross-request L0 term",
        capacity_decimal_gb_s_per_ssu=40.0, intervals=intervals)


def cdf_rows(summary, metadata):
    return [dict(request_id=r["request_id"], npu_id=r["npu_id"], group=metadata[r["request_id"]]["profile_group"],
        admission_ms=r["admission_time_ms"], completion_ms=r["completion_time_ms"],
        compute_ms=r["own_compute_ms"], ttft_ms=r["completion_time_ms"] - r["admission_time_ms"],
        ttft_ratio=(r["completion_time_ms"] - r["admission_time_ms"]) / r["own_compute_ms"])
        for r in summary["request_metrics"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--strategy", required=True, choices=("asu_baseline", "od_baseline", "once"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Optional archived native_summary.json.gz for exact parity")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing any nonempty output directory: {out}")
    config = validate_config(read_json(args.config))
    out.mkdir(parents=True, exist_ok=True)
    save(out / "config.json", config)
    started, last_progress = time.monotonic(), time.monotonic()
    requests, metadata, decks = build(config)
    block_audit = validate_blocks(requests)
    fingerprint = native.continuous_batch_input_fingerprint(requests)
    freeze_manifest(out / "manifest.json.gz", requests, config, decks)
    manifest_hash, config_hash = sha(out / "manifest.json.gz"), sha(args.config)
    q0 = static_qos_config()
    if args.strategy == "od_baseline":
        qos = replace(q0, **asdict(od.qos_config(config["num_npu"], 40 * GB_TO_GIB)))
    else:
        qos = replace(q0, path_cirs=tuple(v * GB_TO_GIB for v in q0.path_cirs),
            path_pirs=tuple(v * GB_TO_GIB for v in q0.path_pirs))
    source_before = source_hashes()
    save(out / "source_sha256.json", source_before)
    save(out / "command.json", dict(argv=sys.argv, config_sha256=config_hash, manifest_sha256=manifest_hash,
        source_sha256=source_before, strategy=args.strategy, od_queue_depth_limit=None,
        native_runtime="archived flat runtime; no event-engine changes", python=sys.version))
    route = "layer_once" if args.strategy == "once" else "baseline"
    client = next(s for s in routing_strategy_specs() if s.name == route).client_config()
    paths, layer0_paths = Counter(), Counter()
    expected_paths = od.od_npu_path_ids(config["num_npu"])
    print(json.dumps(dict(event="start", case=config["name"], strategy=args.strategy,
        num_npu=config["num_npu"], num_ssu=config["ssu"], requests=len(requests),
        expected_blocks=LAYERS * block_audit["manifest_block_count"])), flush=True)
    with exact_tail_adapter("once" if args.strategy == "once" else "baseline", collector_interval_ms=5.0) as adapter:
        def plan_od(context, state, now_ms):
            count = len(state.blocks) - len(state.planned_path_ids)
            ids = od.path_ids(count, state.npu_id, context.num_npu)
            state.planned_path_ids.extend(ids)
            for path, number in Counter(ids).items():
                adapter.pending[state.disk_id][path] += number
                adapter.max_ledger_count = max(adapter.max_ledger_count, adapter.pending[state.disk_id][path])
            adapter.reserved_blocks += len(ids)
            adapter.routing_calls += 1

        original_complete = native._register_complete

        def completed(context, flow):
            nonlocal last_progress
            if args.strategy != "once":
                expected = expected_paths[flow.npu_id] if args.strategy == "od_baseline" else 0
                assert flow.queue_id == expected, (flow.npu_id, flow.disk_id, flow.queue_id, expected)
            paths[(flow.npu_id, flow.disk_id, flow.queue_id)] += flow.block_count
            if flow.layer == 0:
                layer0_paths[(flow.npu_id, flow.disk_id, flow.queue_id)] += flow.block_count
            if context.completed_blocks % 1024 == 0:
                now = time.monotonic()
                if now - last_progress >= 30:
                    last_progress = now
                    print(json.dumps(dict(event="progress", strategy=args.strategy,
                        simulation_ms=context.current_time_ms, completed_requests=context.completed_requests,
                        completed_blocks=context.completed_blocks, wall_seconds=now - started)), flush=True)
            return original_complete(context, flow)

        with ExitStack() as stack:
            if args.strategy == "od_baseline":
                stack.enter_context(patch.object(native, "_plan_paths", plan_od))
            stack.enter_context(patch.object(native, "_register_complete", completed))
            summary = native.simulate_continuous_batch(requests, num_npu=config["num_npu"], num_ssu=config["ssu"],
                n_layers=LAYERS, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
                qos_config=qos, client_io_config=client, cross_request_layer0_prefetch=True,
                pressure_ttl_ms=5.0, disk_bw_gbps=40 * GB_TO_GIB, npu_bw_gbps=50 * GB_TO_GIB,
                submit_order_seed=config["seed"], control=None)
        adapter_stats = adapter.statistics()
        assert all(actual == qos for actual in adapter.context.qos_configs_by_ssu)
    expected_blocks = LAYERS * block_audit["manifest_block_count"]
    assert all(summary["invariants"].values()), summary["invariants"]
    assert summary["request_count"] == len(requests) == len(summary["request_metrics"])
    assert len(summary["microbatch_metrics"]) == len(requests)
    assert all(len(b["layer_metrics"]) == LAYERS for b in summary["microbatch_metrics"])
    assert sum(paths.values()) == summary["completed_blocks"] == summary["submitted_blocks"] == expected_blocks
    assert adapter_stats["acknowledged_blocks"] == adapter_stats["reserved_blocks"] == expected_blocks
    assert adapter_stats["ledger_end_counts_by_ssu"] == [0] * config["ssu"]
    assert adapter_stats["min_ledger_count"] == 0 and not adapter_stats["cir_write_events"]
    assert native.continuous_batch_input_fingerprint(requests) == fingerprint
    assert source_before == source_hashes(), "Source changed during execution"
    assert config_hash == sha(args.config) and manifest_hash == sha(out / "manifest.json.gz")
    expected_read = LAYERS * math.fsum(m["per_layer_kv_gb"] for m in metadata.values())
    assert math.isclose(summary["completed_read_gb"], expected_read, rel_tol=1e-8, abs_tol=1e-8)
    analysis = audit.analyze(summary, metadata, config)
    for window in (analysis["warm"], analysis["full"]):
        window["all_npus_compute_both_groups"] = all(all(p["by_group"][g]["compute_card_ms"] > 1e-9
            for g in ("A", "B")) for p in window["per_npu"])
    m = dict(analysis, name=config["name"], strategy=args.strategy, seed=config["seed"],
        npu_count=config["num_npu"], ssu_count=config["ssu"], n_layers=LAYERS, batch_size=1,
        warm_U_percent=100 * analysis["warm"]["fleet_utilization"],
        full_U_percent=100 * analysis["full"]["fleet_utilization"],
        warm_SLO_1p5_percent=100 * analysis["warm"]["slo_admitted"]["rate"] if analysis["warm"]["slo_admitted"]["count"] else None,
        full_SLO_1p5_percent=100 * analysis["slo_all_requests"]["rate"],
        makespan_ms=summary["makespan_ms"], wall_seconds=time.monotonic() - started,
        input_fingerprint=fingerprint, manifest_sha256=manifest_hash, config_sha256=config_hash,
        source_sha256=source_before, source_unchanged=True, n_requests=len(requests),
        units=dict(capacity="decimal GB/s", native_volume="GiB", time="ms"),
        disk_bw_gib_s=40 * GB_TO_GIB, npu_bw_gib_s=50 * GB_TO_GIB,
        cdf_cohort="all completed input requests, including startup; ratio=(completion-admission)/own_compute",
        queue_depth_limit=None, static_qos=asdict(qos),
        od_configuration=od.configuration(config["num_npu"], 40 * GB_TO_GIB) if args.strategy == "od_baseline" else None,
        observed_paths=[dict(npu=n, ssu=s, path=p, blocks=c) for (n, s, p), c in sorted(paths.items())],
        observed_layer0_paths=[dict(npu=n, ssu=s, path=p, blocks=c) for (n, s, p), c in sorted(layer0_paths.items())],
        block_audit=block_audit, completed_blocks=expected_blocks, completed_read_gib=summary["completed_read_gb"],
        invariants=summary["invariants"])
    save(out / "native_summary.json.gz", summary)
    save(out / "adapter_statistics.json", adapter_stats)
    save(out / "cdf_samples.json.gz", cdf_rows(summary, metadata))
    save(out / "nominal_demand_events.json.gz", nominal_event_intervals(summary, metadata, config["ssu"]))
    if args.reference:
        reference = read_json(args.reference)
        fields = ["request_metrics", "microbatch_metrics", "makespan_ms", "input_fingerprint", "events_processed",
            "completed_blocks", "completed_read_gb", "fleet_npu_compute_utilization", "disk_stats"]
        exact = {k: summary[k] == reference[k] for k in fields}
        checked = dict(exact)
        # Archived scalar sums may differ by a final floating-point bit across
        # Python versions. Every request/layer record remains strict equality.
        scalar = "fleet_npu_compute_utilization"
        checked[scalar] = math.isclose(summary[scalar], reference[scalar], rel_tol=0.0, abs_tol=1e-12)
        parity = dict(reference=str(args.reference.resolve()), reference_sha256=sha(args.reference),
            fields=checked, fields_exact=exact,
            scalar_aggregation_absolute_tolerance=1e-12,
            scalar_aggregation_difference=summary[scalar] - reference[scalar],
            output_sha256={k: object_sha(summary[k]) for k in fields},
            reference_field_sha256={k: object_sha(reference[k]) for k in fields})
        save(out / "parity.json", parity)
        assert all(parity["fields"].values()), parity["fields"]
        m["parity_all_pass"] = True
    save(out / "metrics.json", m)
    print(json.dumps(dict(event="done", case=config["name"], strategy=args.strategy,
        warm_U_percent=m["warm_U_percent"], warm_SLO_1p5_percent=m["warm_SLO_1p5_percent"],
        full_SLO_1p5_percent=m["full_SLO_1p5_percent"], wall_seconds=m["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
