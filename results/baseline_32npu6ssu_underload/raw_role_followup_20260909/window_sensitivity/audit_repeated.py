#!/usr/bin/env python3
"""Independent audit of tripled queues and complete saved event logs; no simulation.

Only manifest deserialization is shared with the runner. Time accounting,
nominal capacity, SLO cohorts, and the old-run prefix comparison are recomputed.
Scientific condition failures are retained as completed audited observations.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
F = HERE.parent
ROOT = F.parent.parents[1]
OUT = HERE / "repeated_queues"
sys.path.insert(0, str(ROOT))
from run_baseline_npu32_stress import load_manifest
from run_coflow_experiments import source_files

NPU, SSU, LAYERS, REPEATS = 32, 6, 8, 3
TIME_TOL, RATE_TOL = 1e-6, 1e-9
BLOCK = 176 * 1024 / 2**30
EXTRA_WINDOWS = [(4000, 12000), (6000, 12000)]
ADDED = {"base_request_id", "source_cycle", "source_base_position", "source_base_npu", "source_global_request_id"}
POSITION = {"request_id", "npu_id", "generation"}


def read(path):
    p = Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(condition, description):
    if not condition:
        raise AssertionError(description)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=TIME_TOL)


def clip(a, b, start, end):
    return max(0.0, min(b, end) - max(a, start))


def scientific_record(r):
    ignored = POSITION | ADDED | {"source_fixed_request_id", "source_fixed_npu_id"}
    return {"arrival_ms": r.arrival_time_ms, "placement": r.placement,
            "load": {k: v for k, v in r.load.items() if k not in ignored}}


def audit_input(item, table):
    path, source = Path(item["manifest"]), Path(item["base_manifest"])
    check(sha(path) == item["manifest_sha256"], "manifest bytes changed")
    check(sha(source) == item["base_manifest_sha256"], "base manifest changed")
    requests, meta = load_manifest(path)
    base, bm = load_manifest(source)
    check(meta["label"] == item["label"] and Path(meta["base_manifest"]).resolve() == source.resolve()
          and meta["base_manifest_sha256"] == sha(source), "base/input label or path metadata differs")
    check(meta["construction_runner_sha256"] == sha(HERE / "run_repeated_queues.py"), "construction runner provenance differs")
    check((meta["num_npu"], meta["num_ssu"], meta["n_layers"], meta["seed"], meta["repetitions"]) == (32, 6, 8, 7, 3), "wrong dimensions/seed/repetition")
    check(meta["input_fingerprint"] == item["input_fingerprint"], "plan/input fingerprint mismatch")
    check(meta["base_input_fingerprint"] == bm["input_fingerprint"], "wrong base fingerprint")
    check(meta["base_manifest_metadata"] == bm, "embedded base metadata changed")
    check(meta["source_data_sha256"] == sha(ROOT / "data"), "data changed")
    check(len(base) == 1212 and len(requests) == meta["request_count"] == item["request_count"] == 3636, "wrong population size")
    check(sha(OUT / "base_manifests" / f"{item['mode']}.json.gz") == sha(source), "base snapshot changed")
    check(sha(item["mapping"]) == item["mapping_sha256"] == meta["identity_mapping_sha256"], "mapping bytes changed")
    mapping = read(item["mapping"])
    check(mapping["label"] == item["label"] and mapping["base_manifest_sha256"] == sha(source), "mapping header differs")
    mapped = {r["request_id"]: r for r in mapping["rows"]}
    check(len(mapped) == len(mapping["rows"]) == len(requests), "mapping is not one-to-one")
    base_lanes = {n: sorted((r for r in base if r.npu_id == n), key=lambda r: r.request_id) for n in range(NPU)}
    new_lanes = {n: sorted((r for r in requests if r.npu_id == n), key=lambda r: r.request_id) for n in range(NPU)}
    stored_lanes = {r["npu_id"]: r for r in meta["per_npu_assignment"]}
    science, global_ids, per_npu, per_npu_ids = {}, {}, [], {}
    maxima = [[0.0] * SSU for _ in range(NPU)]
    link_max = [0.0] * NPU
    seen = set()
    for n in range(NPU):
        lane, original = new_lanes[n], base_lanes[n]
        check(len(lane) == REPEATS * len(original), "lane population not exactly tripled")
        lane_ids = []
        for pos, r in enumerate(lane):
            cycle, base_position = divmod(pos, len(original))
            old = original[base_position]
            expected_global_id = old.request_id if item["mode"] == "fixed" else old.load["source_fixed_request_id"]
            check(r.request_id == n * 1_000_000 + pos and r.request_id not in seen, "duplicate or wrong new position ID")
            seen.add(r.request_id)
            check(r.npu_id == old.npu_id == n and r.load["npu_id"] == n, "binding changed during repetition")
            check(r.load["request_id"] == r.request_id and r.load["generation"] == pos, "load position ID differs")
            check(set(r.load) - set(old.load) == ADDED, "unexpected load fields added")
            check(all(r.load[k] == v for k, v in old.load.items() if k not in POSITION), "original scientific/provenance field modified")
            check(r.arrival_time_ms == old.arrival_time_ms == 0 and r.placement == old.placement, "arrival or placement modified")
            extra = dict(base_request_id=old.request_id, source_cycle=cycle, source_base_position=base_position,
                         source_base_npu=n, source_global_request_id=expected_global_id)
            check(all(r.load[k] == v for k, v in extra.items()), "wrong copy provenance")
            mr = mapped[r.request_id]
            check(mr == dict(request_id=r.request_id, npu_id=n, rowposition=pos, source_cycle=cycle,
                base_request_id=old.request_id, base_position=base_position, source_global_request_id=expected_global_id,
                original_request_id=old.load["original_request_id"], canonical_identity=[expected_global_id, cycle]), "mapping row differs from actual copy")
            key = int(r.load["seq_len_k"]), int(r.load["nql"])
            _, raw_c, raw_ttft, raw_v = table[key]
            check(r.load["per_layer_us"] == r.load["original_compute_us"] == raw_c, "C is not exact raw data")
            check(r.load["source_ttft_ms"] == raw_ttft and not r.load["constructed_profile"] and r.load["padding_gib_per_layer"] == 0, "raw provenance changed")
            check(r.load["category"] == ("S" if key[0] <= 80 else "L") + ("S" if key[1] < 512 else "L"), "wrong category")
            check(len(r.placement) in (1, 8) and all(p == r.placement[0] for p in r.placement), "layer placement varies")
            check(all(0 <= s < SSU and v == BLOCK for s, v in r.placement[0]), "invalid physical block")
            volume = math.fsum(v for _, v in r.placement[0])
            check(math.isclose(volume, raw_v, abs_tol=1e-12, rel_tol=0) and math.isclose(r.load["per_layer_kv_gb"], raw_v, abs_tol=1e-12, rel_tol=0), "V differs from raw data")
            rates = [math.fsum(v for s, v in r.placement[0] if s == d) * 1e6 / raw_c for d in range(SSU)]
            maxima[n] = [max(a, b) for a, b in zip(maxima[n], rates)]
            link_max[n] = max(link_max[n], math.fsum(rates))
            identity = cycle, r.load["original_request_id"]
            check(identity not in science, "duplicate (cycle, original_request_id)")
            science[identity] = digest(scientific_record(r))
            global_ids[cycle, expected_global_id] = identity
            lane_ids.append(identity)
        pure = 8 * math.fsum(r.load["per_layer_us"] / 1000 for r in lane)
        counts = [sum(r.load["profile_index"] == p for r in lane) for p in range(4)]
        expected = stored_lanes[n]
        check(expected["profile_counts"] == counts and expected["request_count"] == len(lane) and expected["base_request_count"] == len(original), "lane description incorrect")
        check(close(expected["pure_compute_ms"], pure) and pure > 12000, "insufficient or wrong pure computation horizon")
        first_volume = math.fsum(v for _,v in original[0].placement[0])
        second_cycle_release_lower_bound = pure/REPEATS - original[-1].load["per_layer_us"]/1000 + first_volume/50*1000
        per_npu.append(dict(npu_id=n, base_requests=len(original), requests=len(lane), profile_counts=counts, pure_compute_ms=pure,
            second_cycle_layer0_release_lower_bound_ms=second_cycle_release_lower_bound))
        per_npu_ids[n] = sorted(lane_ids)
    check(len(global_ids) == len(science) == len(requests), "global copy identity not bijective")
    check(Counter(r.load["source_cycle"] for r in requests) == {0: 1212, 1: 1212, 2: 1212}, "copy counts differ")
    check([sum(r.load["profile_index"] == p for r in requests) for p in range(4)] == meta["source_profile_counts"] == [792, 792, 792, 1260], "profile totals differ")
    bounds = [math.fsum(row[d] for row in maxima) for d in range(SSU)]
    proof = meta["active_profile_rate_certificate"]
    check(len(proof["per_ssu_upper_bound_gib_s"]) == SSU and all(close(a, b) for a, b in zip(bounds, proof["per_ssu_upper_bound_gib_s"])), "static disk certificate incorrect")
    check(len(proof["per_npu_receive_upper_bound_gib_s"]) == NPU and all(close(a, b) for a, b in zip(link_max, proof["per_npu_receive_upper_bound_gib_s"])), "static link certificate incorrect")
    passed = max(bounds) < 40 and max(link_max) < 50 and math.fsum(link_max) < 240
    check(passed == proof["passes"] == meta["load_within_disk_and_link_capacity"], "static certificate result incorrect")
    summary = dict(mode=item["mode"], label=item["label"], passed=True, manifest_sha256=sha(path),
        input_fingerprint=meta["input_fingerprint"], base_manifest=str(source), base_manifest_sha256=sha(source),
        requests=len(requests), source_requests=len(base), source_cycles=[0, 1, 2], per_npu=per_npu,
        global_copy_identity_science_sha256=digest(sorted((list(k), v) for k, v in science.items())),
        min_per_npu_pure_compute_ms=min(x["pure_compute_ms"] for x in per_npu),
        min_second_cycle_layer0_release_lower_bound_ms=min(x["second_cycle_layer0_release_lower_bound_ms"] for x in per_npu),
        per_ssu_static_upper_bound_gib_s=bounds, per_npu_static_link_upper_bound_gib_s=link_max,
        static_sufficient_certificate_passed=passed)
    return summary, requests, meta, science, global_ids, per_npu_ids


def records_from_raw(raw, requests):
    s = raw["summary"]
    byid = {r.request_id: r for r in requests}
    check(len(byid) == len(requests) == s["request_count"], "wrong completed request population")
    check((s["num_npu"], s["num_ssu"], s["n_layers"], s["batch_size"]) == (32, 6, 8, 1), "wrong execution dimensions")
    check(all(s["invariants"].values()), "simulator invariant failure")
    request_metrics = {r["request_id"]: r for r in s["request_metrics"]}
    check(len(request_metrics) == len(s["request_metrics"]) == len(byid), "duplicate/missing request metric")
    records = []
    seen = set()
    for b in s["microbatch_metrics"]:
        check(b["batch_size"] == len(b["member_request_ids"]) == 1, "batch size changed")
        rid = b["member_request_ids"][0]
        check(rid in byid and rid not in seen, "duplicate or foreign executed request")
        seen.add(rid)
        r, rm = byid[rid], request_metrics[rid]
        n, a, end = b["npu_id"], b["admission_time_ms"], b["completion_time_ms"]
        check(n == r.npu_id == rm["npu_id"] and close(a, rm["admission_time_ms"]) and close(end, rm["completion_time_ms"]), "binding/timestamp changed")
        check(rm["arrival_time_ms"] == r.arrival_time_ms == 0 and math.isfinite(end) and 0 <= a < end, "invalid request interval")
        layers = sorted(b["layer_metrics"], key=lambda z: z["layer"])
        check([z["layer"] for z in layers] == list(range(LAYERS)), "incomplete layers")
        previous, compute, stall = a, 0.0, 0.0
        for j, layer in enumerate(layers):
            cs, ce = layer["compute_start_ms"], layer["compute_end_ms"]
            ready, release = layer["io_ready_time_ms"], layer["io_start_time_ms"]
            check(previous <= cs + TIME_TOL and cs < ce and release <= ready + TIME_TOL, "invalid layer ordering")
            check(close(ce-cs, r.load["per_layer_us"]/1000) and close(layer["compute_duration_ms"], ce-cs), "executed C differs from manifest")
            check(close(cs, max(previous, ready)) and close(cs-previous, layer["io_barrier_wait_ms"]), "stall not explained by readiness barrier")
            if j:
                check(close(release, layers[j-1]["compute_start_ms"]), "internal prefetch release changed")
            compute += ce-cs; stall += cs-previous; previous = ce
        check(close(previous, end) and close(compute+stall, end-a), "request compute/stall accounting failed")
        check(close(compute, rm["own_compute_ms"]) and close(stall, rm["io_stall_ms"]), "request summary time mismatch")
        rates = [math.fsum(v for d, v in r.placement[0] if d == disk) * 1e6/r.load["per_layer_us"] for disk in range(SSU)]
        records.append(dict(rid=rid, npu=n, role=r.load["role"], admission=a, completion=end, layers=layers,
            pure_compute_ms=LAYERS*r.load["per_layer_us"]/1000, rates=rates,
            source_cycle=r.load.get("source_cycle", 0), base_request_id=r.load.get("base_request_id", rid)))
    check(seen == set(byid), "not all requests completed")
    for n in range(NPU):
        lane = sorted((r for r in records if r["npu"] == n), key=lambda r: r["admission"])
        check(lane and [r["rid"] for r in lane] == sorted(r["rid"] for r in lane), "queue order changed")
        check(all(x["completion"] <= y["admission"] + TIME_TOL for x, y in zip(lane, lane[1:])), "overlapping admitted requests")
    makespan = max(r["completion"] for r in records)
    check(close(makespan, s["makespan_ms"]), "makespan incorrect")
    expected_blocks = LAYERS * sum(len(r.placement[0]) for r in requests)
    expected_volume = LAYERS * math.fsum(math.fsum(v for _, v in r.placement[0]) for r in requests)
    check(s["submitted_blocks"] == s["completed_blocks"] == expected_blocks, "incomplete/extra physical blocks")
    check(close(s["expected_read_gb"], expected_volume) and close(s["completed_read_gb"], expected_volume), "incomplete/extra physical volume")
    check(s["completed_batch_layers"] == len(requests)*LAYERS, "incomplete executed layers")
    return records, makespan


def accounting(records, start, end):
    check(0 <= start < end, "invalid window")
    duration = end-start
    cards = []
    for n in range(NPU):
        lane = sorted((r for r in records if r["npu"] == n), key=lambda r: r["admission"])
        roles = {role: dict(compute_ms=0.0, active_ms=0.0, l0_stall_ms=0.0, internal_stall_ms=0.0) for role in ("short", "long")}
        for r in lane:
            role = roles[r["role"]]
            role["active_ms"] += clip(r["admission"], r["completion"], start, end)
            previous = r["admission"]
            for layer in r["layers"]:
                cs, ce = layer["compute_start_ms"], layer["compute_end_ms"]
                role["compute_ms"] += clip(cs, ce, start, end)
                role["l0_stall_ms" if layer["layer"] == 0 else "internal_stall_ms"] += clip(previous, cs, start, end)
                previous = ce
        totals = {k: math.fsum(v[k] for v in roles.values()) for k in ("compute_ms", "active_ms", "l0_stall_ms", "internal_stall_ms")}
        totals.update(startup_idle_ms=clip(0, lane[0]["admission"], start, end),
            between_requests_idle_ms=math.fsum(clip(a["completion"], b["admission"], start, end) for a, b in zip(lane, lane[1:])),
            tail_idle_ms=clip(lane[-1]["completion"], end, start, end))
        totals["idle_ms"] = sum(totals[k] for k in ("startup_idle_ms", "between_requests_idle_ms", "tail_idle_ms"))
        check(close(totals["compute_ms"]+totals["l0_stall_ms"]+totals["internal_stall_ms"], totals["active_ms"]), "active != compute+stall")
        check(close(totals["active_ms"]+totals["idle_ms"], duration), "card-time conservation failed")
        for role in roles.values():
            check(close(role["compute_ms"]+role["l0_stall_ms"]+role["internal_stall_ms"], role["active_ms"]), "role-time conservation failed")
        cards.append(dict(npu_id=n, **totals, by_role=roles))
    totals = {k: math.fsum(c[k] for c in cards) for k in cards[0] if k not in ("npu_id", "by_role")}
    by_role = {role: {key: math.fsum(c["by_role"][role][key] for c in cards) for key in cards[0]["by_role"][role]} for role in ("short", "long")}
    admitted = [r for r in records if start <= r["admission"] < end]
    passed = [r for r in admitted if r["completion"]-r["admission"] <= 1.5*r["pure_compute_ms"]+1e-8]
    den = NPU*duration
    return dict(start_ms=start, end_ms=end, duration_ms=duration, card_time_ms=den, **totals,
        device_U_percent=100*totals["compute_ms"]/den,
        stall_percent=100*(totals["l0_stall_ms"]+totals["internal_stall_ms"])/den,
        idle_percent=100*totals["idle_ms"]/den,
        active_compute_fraction_percent=100*totals["compute_ms"]/totals["active_ms"] if totals["active_ms"] else None,
        all_32_active=all(c["idle_ms"] <= TIME_TOL for c in cards),
        cards_with_both_roles_positive_compute=sum(all(c["by_role"][role]["compute_ms"] > 1e-8 for role in ("short", "long")) for c in cards),
        cards_with_both_roles_100ms_compute=sum(all(c["by_role"][role]["compute_ms"] >= 100 for role in ("short", "long")) for c in cards),
        mean_long_active_cards=by_role["long"]["active_ms"]/duration,
        mean_short_active_cards=by_role["short"]["active_ms"]/duration,
        admission_slo_count=len(admitted), admission_slo_passed=len(passed),
        admission_slo_percent=100*len(passed)/len(admitted) if admitted else None,
        by_role=by_role, per_npu=cards)


def nominal_segments(records):
    events = defaultdict(lambda: {"add": [], "remove": []})
    for r in records:
        events[r["admission"]]["add"].append(r)
        events[r["completion"]]["remove"].append(r)
    events[0.0]
    times, active, segments = sorted(events), {}, []
    for i, time in enumerate(times):
        for r in events[time]["remove"]:
            check(active.get(r["npu"], {}).get("rid") == r["rid"], "nominal scan unmatched completion")
            del active[r["npu"]]
        for r in events[time]["add"]:
            check(r["npu"] not in active, "nominal scan double admission")
            active[r["npu"]] = r
        if i+1 < len(times):
            disk = [math.fsum(r["rates"][d] for r in active.values()) for d in range(SSU)]
            link = [math.fsum(active[n]["rates"]) if n in active else 0.0 for n in range(NPU)]
            segments.append((time, times[i+1], disk, link, len(active), sum(r["role"] == "long" for r in active.values())))
    check(not active, "nominal scan ends with unfinished requests")
    return segments


def nominal_window(segments, start, end):
    maxima, link_max, excess = [0.0]*SSU, [0.0]*NPU, [0.0]*SSU
    any_excess, link_excess, hist = 0.0, 0.0, defaultdict(float)
    for a, b, disk, link, active, long in segments:
        dt = clip(a, b, start, end)
        if not dt:
            continue
        maxima = [max(x, y) for x, y in zip(maxima, disk)]
        link_max = [max(x, y) for x, y in zip(link_max, link)]
        for d in range(SSU):
            if disk[d] > 40+RATE_TOL:
                excess[d] += dt
        any_excess += dt if max(disk) > 40+RATE_TOL else 0.0
        link_excess += dt if max(link) > 50+RATE_TOL else 0.0
        hist[long, active-long, NPU-active] += dt
    covered = math.fsum(hist.values())
    check(covered <= end-start+TIME_TOL, "overlapping nominal segments")
    # A requested window may extend beyond the complete finite run.
    if end-start > covered:
        hist[0, 0, NPU] += end-start-covered
    return dict(start_ms=start, end_ms=end, per_ssu_peak_gib_s=maxima, max_ssu_gib_s=max(maxima),
        per_npu_link_peak_gib_s=link_max, max_npu_link_gib_s=max(link_max), per_ssu_over_capacity_ms=excess,
        any_ssu_over_capacity_ms=any_excess, any_npu_link_over_capacity_ms=link_excess,
        all_ssu_within_capacity=max(maxima) <= 40+RATE_TOL,
        all_npu_links_within_capacity=max(link_max) <= 50+RATE_TOL,
        active_role_histogram_ms=[dict(long_cards=l, short_cards=s, idle_cards=i, duration_ms=t) for (l,s,i),t in sorted(hist.items())])


def prefix_events(records, end=4000):
    result = {}
    for r in records:
        prefix = r["npu"], r["source_cycle"], r["base_request_id"]
        for kind, time in (("admission", r["admission"]), ("completion", r["completion"])):
            if time < end:
                result[(*prefix, -1, kind)] = time
        for layer in r["layers"]:
            for kind in ("io_start_time_ms", "io_ready_time_ms", "compute_start_ms", "compute_end_ms"):
                if layer[kind] < end:
                    result[(*prefix, layer["layer"], kind)] = layer[kind]
    return result


def compare_prefix(raw, records, item, meta):
    parent = F if item["mode"] == "fixed" else F / "mixed_rebinding"
    parent_summary = parent / ("followup_results.json" if item["mode"] == "fixed" else "results.json")
    label, strategy = meta["base_manifest_metadata"]["label"], raw["strategy"]
    reference = [r for r in read(parent_summary)["rows"] if r["label"] == label and r["strategy"] == strategy]
    check(len(reference) == 1 and reference[0]["audit_pass"], "missing audited base result")
    reference = reference[0]
    base_path = Path(reference["result_path"])
    check(sha(base_path) == reference["result_sha256"], "base result bytes changed")
    old = read(base_path)
    for key in ("policy_config", "submit_seed", "collector_interval_ms", "cir_policy", "static_path_cirs_gib_s", "modeled_control_latency_ms"):
        check(raw[key] == old[key], f"base/repeated policy configuration differs: {key}")
    for key in ("cross_request_layer0_prefetch", "prefetch_policy", "client_submit_batch_size", "client_issue_interval_us"):
        check(raw["summary"][key] == old["summary"][key], f"base/repeated data-plane configuration differs: {key}")
    check(raw["core_and_policy_sha256"] == old["core_and_policy_sha256"], "base/repeated core sources differ")
    base, _ = load_manifest(item["base_manifest"])
    old_records, _ = records_from_raw(old, base)
    new_events, old_events = prefix_events(records), prefix_events(old_records)
    shared = set(new_events) & set(old_events)
    exact = new_events == old_events
    diffs = [abs(new_events[k]-old_events[k]) for k in shared]
    maxdiff = max(diffs, default=0.0)
    within = set(new_events) == set(old_events) and maxdiff <= 1e-7
    mismatches = [dict(event=list(k), old_ms=old_events.get(k), repeated_ms=new_events.get(k))
                  for k in sorted(set(new_events) | set(old_events)) if new_events.get(k) != old_events.get(k)][:20]
    comparisons = []
    for start, end in ((0, 4000), (2000, 4000)):
        a, b = accounting(old_records, start, end), accounting(records, start, end)
        fields = ("compute_ms", "active_ms", "l0_stall_ms", "internal_stall_ms", "idle_ms")
        error = max(abs(x[k]-y[k]) for x,y in zip(a["per_npu"],b["per_npu"]) for k in fields)
        comparisons.append(dict(start_ms=start, end_ms=end, max_per_card_accounting_difference_ms=error, passed=error <= TIME_TOL))
    return dict(base_result_path=str(base_path), base_result_sha256=sha(base_path),
        prefix_interval_ms=[0,4000], old_event_count=len(old_events), repeated_event_count=len(new_events),
        exact_event_times_and_identities_equal=exact, max_event_time_difference_ms=maxdiff,
        events_equal_within_1e_7_ms=within, accountings=comparisons,
        passed=within and all(x["passed"] for x in comparisons), first_exact_mismatches=mismatches,
        limitation="Only events strictly before4000ms and clipped compute/stall/active intervals are compared; post-window completion metrics are not treated as prefix events.")


def audit_result(path, command, item, plan, payload, windows, expected_strategy):
    _, requests, meta, *_ = payload
    raw = read(path)
    check(command["status"] == "complete" and command["returncode"] == 0, "command did not complete successfully")
    check(command["label"] == item["label"] and command["mode"] == item["mode"] and command["strategy"] == expected_strategy,
          "command label/mode/strategy differs from job")
    check(command["source_sha256"] == plan["source_sha256"], "command source hashes differ from plan")
    check(command["input_sha256"] == item["manifest_sha256"] and command["input_fingerprint"] == item["input_fingerprint"], "command input differs from plan")
    expected_args = ["-B", str(ROOT/"run_baseline_npu32_stress.py"), "--manifest", item["manifest"],
                     "--strategy", command["strategy"], "--assignment", "fixed"]
    for start, end in plan["windows_ms"]:
        expected_args.extend(("--window", f"{start}:{end}"))
    expected_args.extend(("--output", str(path.parent)))
    check(command["command"][1:] == expected_args and Path(command["cwd"]).resolve() == ROOT.resolve(), "CLI differs from frozen experiment")
    check(sha(path) == command["output_sha256"] and Path(command["output"]).resolve() == path.resolve(), "command/output hash mismatch")
    check(raw["input_fingerprint"] == raw["summary"]["input_fingerprint"] == item["input_fingerprint"], "executed input mismatch")
    check(Path(raw["manifest_path"]).resolve() == Path(item["manifest"]).resolve(), "raw result names another manifest")
    check(raw["strategy"] == command["strategy"] == expected_strategy and raw["submit_seed"] == 7, "wrong policy or seed")
    check(raw["collector_interval_ms"] == 5 and raw["policy_config"]["assignment"] == "fixed" and raw["assignment_log"] == [], "policy/snapshot/assignment changed")
    check(raw["execution_placement_fingerprint"] == raw["input_placement_fingerprint"], "execution moved physical data")
    core = raw["core_and_policy_sha256"]
    check(set(core) == set(source_files()), "core keyset incomplete")
    check(all(plan["source_sha256"][k] == v for k,v in core.items()), "core does not match frozen plan")
    check(raw["stress_runner_sha256"] == plan["source_sha256"]["run_baseline_npu32_stress.py"], "stress runner changed")
    check([(w["start_ms"],w["end_ms"]) for w in raw["windows"]] == [tuple(w) for w in plan["windows_ms"]], "planned window list differs")
    s = raw["summary"]
    # The shared-path adapter uses one block per issue event, overriding the
    # generic sim.ClientIOConfig default batch of eight. Bind to observed base
    # configuration again in compare_prefix, not merely to that generic default.
    check(s["cross_request_layer0_prefetch"] and s["client_submit_batch_size"] == 1 and s["client_issue_interval_us"] == 0.1, "data-plane prefetch/issue config changed")
    records, makespan = records_from_raw(raw, requests)
    segments = nominal_segments(records)
    full_accounting = accounting(records, 0, makespan)
    check(close(full_accounting["device_U_percent"], s["fleet_npu_compute_utilization"]*100), "full-run fleet utilization differs")
    full_nominal = nominal_window(segments, 0, makespan)
    outputs = []
    for start, end in windows:
        w = accounting(records, start, end)
        w["nominal"] = nominal_window(segments, start, end)
        outputs.append(w)
        saved = [z for z in raw["windows"] if (z["start_ms"],z["end_ms"]) == (start,end)]
        if saved:
            saved = saved[0]
            check(close(w["device_U_percent"], 100*saved["mean_npu_utilization"]), "saved window U differs")
            for key, saved_key in (("compute_ms","compute_ms_by_npu"),("active_ms","active_ms_by_npu"),("idle_ms","idle_ms_by_npu")):
                check(all(close(c[key],value) for c,value in zip(w["per_npu"],saved[saved_key])), f"saved per-NPU {key} differs")
            check(all(close(c["l0_stall_ms"]+c["internal_stall_ms"],value) for c,value in zip(w["per_npu"],saved["io_stall_ms_by_npu"])), "saved stall differs")
            check(w["all_32_active"] == saved["all_npus_active_whole_window"], "saved all-active flag differs")
    prefix = compare_prefix(raw, records, item, meta)
    return dict(label=item["label"], mode=item["mode"], strategy=raw["strategy"], status="complete", technical_audit_passed=True,
        request_count=len(records), result_path=str(path), result_sha256=sha(path), input_sha256=item["manifest_sha256"],
        makespan_ms=makespan, first_card_final_completion_ms=min(max(r["completion"] for r in records if r["npu"]==n) for n in range(NPU)),
        full_run_accounting=full_accounting, full_run_nominal=full_nominal, windows=outputs, original_prefix=prefix,
        scientific_full_run_nominal_capacity_passed=full_nominal["all_ssu_within_capacity"] and full_nominal["all_npu_links_within_capacity"],
        scientific_all_declared_windows_all_32_active=all(w["all_32_active"] for w in outputs),
        scientific_all_windows_every_card_mixed=all(w["cards_with_both_roles_positive_compute"] == 32 for w in outputs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--inputs-only", action="store_true")
    args = ap.parse_args()
    plan_path = OUT / "plan.json"
    plan = read(plan_path)
    errors, input_rows, payloads = [], [], {}
    check((plan["repetitions"],plan["seed"],len(plan["inputs"]),len(plan["jobs"])) == (3,7,3,6), "wrong experiment scope")
    check({(j["input"]["mode"],j["strategy"]) for j in plan["jobs"]} == {(m,s) for m in ("fixed","random","ordered") for s in ("baseline","once")}, "wrong planned jobs")
    for name, expected in plan["source_sha256"].items():
        check(sha(ROOT/name) == sha(OUT/"sources"/name) == expected, f"frozen source changed: {name}")
    check(sha(plan["input_audit"]) == plan["input_audit_sha256"], "runner input audit changed")
    table = ast.literal_eval((ROOT / "data").read_text())
    for item in plan["inputs"]:
        try:
            payload = audit_input(item, table)
            payloads[item["mode"]] = payload
            input_rows.append(payload[0])
        except Exception as e:
            record = dict(scope="input", mode=item["mode"], error=f"{type(e).__name__}: {e}")
            errors.append(record); input_rows.append(dict(**record, passed=False))
    paired = {}
    if len(payloads) == 3:
        paired = dict(global_cycle_original_id_science_and_placement_equal=payloads["fixed"][3] == payloads["random"][3] == payloads["ordered"][3],
            global_source_id_and_original_id_bijections_equal=payloads["fixed"][4] == payloads["random"][4] == payloads["ordered"][4],
            random_ordered_per_npu_copy_identities_equal=payloads["random"][5] == payloads["ordered"][5])
        if not all(paired.values()):
            errors.append(dict(scope="pairing", error="cross-mode population/placement mismatch"))
    windows = list(dict.fromkeys([tuple(w) for w in plan["windows_ms"]] + EXTRA_WINDOWS))
    runs = []
    for job in plan["jobs"]:
        item, strategy = job["input"], job["strategy"]
        directory = OUT / "runs" / item["label"] / strategy
        command_path = directory / "command.json"
        command = read(command_path) if command_path.exists() else {}
        status = command.get("status", "pending")
        row = dict(label=item["label"], mode=item["mode"], strategy=strategy, status=status)
        if args.inputs_only:
            row["status"] = "not_inspected"
        elif status == "complete":
            try:
                check(item["mode"] in payloads, "input audit failed")
                files = [p for p in directory.glob("*.json.gz") if ".failure." not in p.name]
                check(len(files) == 1, "ambiguous or missing complete result")
                row = audit_result(files[0], command, item, plan, payloads[item["mode"]], windows, strategy)
            except Exception as e:
                row.update(status="audit_failed", error=f"{type(e).__name__}: {e}")
        if row["status"] in ("failed", "timeout", "audit_failed"):
            errors.append(dict(scope="run", **row))
        runs.append(row)
    complete = [r for r in runs if r["status"] == "complete"]
    all_complete = len(complete) == len(plan["jobs"])
    all_prefixes = all_complete and all(r["original_prefix"]["passed"] for r in complete)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), analyzer_sha256=sha(__file__),
        plan_sha256=sha(plan_path), data_sha256=sha(ROOT/"data"), source_sha256=plan["source_sha256"],
        inputs=input_rows, input_pairing=paired, all_input_audits_passed=len(payloads)==3 and all(paired.values()) and not any(e["scope"]=="input" for e in errors),
        planned=len(plan["jobs"]), completed=len(complete), all_complete=all_complete,
        statuses=dict(Counter(r["status"] for r in runs)), all_completed_technical_audits_passed=not errors,
        all_original_prefixes_match=all_prefixes,
        all_scientific_full_run_nominal_capacity_passed=all_complete and all(r["scientific_full_run_nominal_capacity_passed"] for r in complete),
        windows_ms=windows, runs=runs, errors=errors,
        definitions=dict(population="3636 copies of1212 original requests; (cycle,original_request_id) is unique. Three repeats are not three independent seeds.",
            accounting="Compute from clipped layer intervals; I/O stall only inside admission→completion; idle is startup+between-request+tail. Per-card and per-role conservation checked.",
            nominal="Sum of per-SSD V/C for actual currently admitted requests, equal-time handoffs atomic. Next-request L0 excluded by study definition. Neither read lifecycle nor nominal V/C is physical SSD busy time/throughput.",
            capacity="All-run event sweep; failed static sufficient certificate does not imply overload. Observed scientific failures remain complete rows.",
            SLO="Window admissions followed to completion; latency<=1.5*8C. Cohorts change with mode/policy/window.",
            prefix="Compare exact identities/timestamps before4000ms and per-card clipped [0,4000),[2000,4000) accounting; also report floating-point tolerance separately."))
    target = HERE / "audit_repeated.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("all_input_audits_passed","planned","completed","statuses","all_original_prefixes_match","errors")},ensure_ascii=False))
    if errors or (args.require_complete and not (all_complete and all_prefixes and result["all_input_audits_passed"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
