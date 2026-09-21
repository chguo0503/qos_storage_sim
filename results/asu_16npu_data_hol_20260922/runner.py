#!/usr/bin/env python3
"""Native ASU/OD/Once research with measured A/B profiles and explicit ordering.

CLI matches ../formula_ab_32npu_20260921/runner.py. Random and fixed inputs
are delegated unchanged to that audited runner. Additional input modes:

  explicit: decks=["ABAB", "BABA", ...], one complete deck per NPU.
  cyclic: cycle="ABBB", optional cycle_offsets=[0,1,...] and
          cycle_repetitions=integer. Alternatively cycles_by_npu supplies
          equal-count per-card cycles. Default repetitions cover horizon_ms
          plus one complete cycle, exactly like the original count rule.

Explicit/cyclic decks must contain both roles on every card and the same A/B
counts on all cards. Profiles must be exact archived data rows or verified
linear miss interpolation between adjacent rows at the same measured length.
Length interpolation/extrapolation, compute scaling and read padding are rejected.
No native event-engine or policy implementation is changed. OD retains the
original unlimited submission depth. Nominal demand still excludes the extra
cross-request L0 prefetch term; all resulting waits remain in U and latency.
"""
from __future__ import annotations

import ast
from collections import Counter
import importlib.util
import math
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE_PATH = HERE.parent / "formula_ab_32npu_20260921/runner.py"
_spec = importlib.util.spec_from_file_location("asu16_audited_formula_runner", BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
DATA_PATH = base.ORIGINAL / "source/data"
DATA = ast.literal_eval(DATA_PATH.read_text())
DATA_SHA256 = base.sha(DATA_PATH)
_BASE_VALIDATE = base.validate_config
_BASE_BUILD = base.build
_BASE_FREEZE = base.freeze_manifest
_BASE_SOURCE_HASHES = base.source_hashes
_BASE_ANALYZE = base.audit.analyze
_BASE_SAVE = base.save


def _deck(value, name):
    if not isinstance(value, (str, list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty A/B string or sequence")
    result = list(value)
    if any(not isinstance(role, str) or role not in ("A", "B") for role in result):
        raise ValueError(f"{name} contains a role other than A or B")
    if set(result) != {"A", "B"}:
        raise ValueError(f"{name} must include both A and B")
    return result


def controlled_decks(config):
    """Generate labeled deterministic order only; never alter profile work."""
    n = config["num_npu"]
    if config["mode"] == "explicit":
        values = config.get("decks")
        if not isinstance(values, (list, tuple)) or len(values) != n:
            raise ValueError("explicit decks must provide exactly one deck per NPU")
        decks = [_deck(value, f"decks[{i}]") for i, value in enumerate(values)]
    elif config["mode"] == "cyclic":
        if "cycles_by_npu" in config:
            if "cycle" in config:
                raise ValueError("Specify cycle or cycles_by_npu, not both")
            values = config["cycles_by_npu"]
            if not isinstance(values, (list, tuple)) or len(values) != n:
                raise ValueError("cycles_by_npu must contain one cycle per NPU")
            cycles = [_deck(value, f"cycles_by_npu[{i}]") for i, value in enumerate(values)]
        else:
            common = _deck(config.get("cycle"), "cycle")
            cycles = [list(common) for _ in range(n)]
        if any(Counter(c) != Counter(cycles[0]) for c in cycles):
            raise ValueError("Every NPU cycle must have identical counts of A and B")
        offsets = config.get("cycle_offsets", [0] * n)
        if (not isinstance(offsets, (list, tuple)) or len(offsets) != n
                or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets)):
            raise ValueError("cycle_offsets must contain one integer per NPU")
        cycle_ms = 8 * math.fsum(config["profile_" + role]["compute_us"] for role in cycles[0]) / 1000
        repetitions = config.get("cycle_repetitions", math.ceil(config["horizon_ms"] / cycle_ms) + 1)
        if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
            raise ValueError("cycle_repetitions must be a positive integer")
        decks = []
        for cycle, offset in zip(cycles, offsets):
            offset %= len(cycle)
            decks.append((cycle[offset:] + cycle[:offset]) * repetitions)
    else:
        raise ValueError("controlled_decks only supports explicit and cyclic")
    expected = Counter(decks[0])
    if any(Counter(deck) != expected for deck in decks):
        raise ValueError("Every NPU must receive identical counts of A and B")
    for deck in decks:
        if len(deck) >= 1_000_000:
            raise ValueError("Request ID stride exhausted")
        compute_ms = 8 * math.fsum(config["profile_" + role]["compute_us"] for role in deck) / 1000
        if compute_ms + 1e-9 < config["horizon_ms"]:
            raise ValueError("Explicit work must cover horizon_ms on every NPU")
    return decks


def validate_profile_source(profile, role):
    """Recompute every accepted profile from the immutable measured table."""
    key = (profile["total_length_k"], profile["nql"])
    construction = profile.get("profile_construction", {})
    if not isinstance(construction, dict) or construction.get("source") != "data":
        raise ValueError(f"{role} must identify archived data as its source")
    hashes = [construction[k] for k in ("data_sha256", "source_sha256") if k in construction]
    if not hashes or any(value != DATA_SHA256 for value in hashes):
        raise ValueError(f"{role} data SHA256 is missing or differs from archived data")
    if (construction.get("compute_scale", 1) != 1 or construction.get("extrapolated", False)
            or construction.get("total_length_interpolated", False)):
        raise ValueError(f"{role} cannot scale compute, extrapolate, or interpolate total length")
    if isinstance(profile["nql"], bool) or not isinstance(profile["nql"], int):
        raise ValueError(f"{role} miss count must be an integer token count")
    constructed = profile.get("constructed_profile")
    if constructed is False:
        if key not in DATA:
            raise ValueError(f"{role} direct profile is not a measured length/miss row")
        if construction.get("method") not in ("direct_data_row", "exact_data_row", "original_data_row"):
            raise ValueError(f"{role} direct profile must explicitly identify the direct-row method")
        if "source_key" in construction and tuple(construction["source_key"]) != key:
            raise ValueError(f"{role} source_key differs from the profile")
        for field in ("original_row", "original_data_row"):
            if field in construction and tuple(construction[field]) != DATA[key]:
                raise ValueError(f"{role} embedded source row differs from archived data")
        if "anchors" in construction:
            anchors = construction["anchors"]
            if not isinstance(anchors, list) or len(anchors) != 1:
                raise ValueError(f"{role} direct profile must have exactly one source anchor")
            anchor = anchors[0]
            if ((anchor.get("seq_len_k"), anchor.get("nql")) != key
                    or isinstance(anchor.get("weight"), bool) or anchor.get("weight") != 1
                    or anchor.get("compute_us") != DATA[key][1]):
                raise ValueError(f"{role} direct source anchor does not match the measured row")
            if "original_data_row" in anchor and tuple(anchor["original_data_row"]) != DATA[key]:
                raise ValueError(f"{role} direct anchor's embedded row differs from archived data")
        expected_compute = DATA[key][1]
    elif constructed is True:
        if construction.get("method") != "linear_interpolation_miss_same_length":
            raise ValueError(f"{role} only same-length miss interpolation is supported")
        if key in DATA:
            raise ValueError(f"{role} measured grid point must be labeled as a direct data row")
        misses = sorted(m for length, m in DATA if length == key[0])
        if not misses or not misses[0] < key[1] < misses[-1]:
            raise ValueError(f"{role} miss interpolation must stay inside a measured length's grid")
        lo = max(m for m in misses if m < key[1])
        hi = min(m for m in misses if m > key[1])
        anchors = construction.get("anchors")
        if not isinstance(anchors, list) or len(anchors) != 2:
            raise ValueError(f"{role} miss interpolation requires exactly two adjacent anchors")
        expected_weights = {lo: (hi - key[1]) / (hi - lo), hi: (key[1] - lo) / (hi - lo)}
        seen, weighted_compute, weights = set(), [], []
        for anchor in anchors:
            anchor_key = (anchor.get("seq_len_k"), anchor.get("nql"))
            if anchor_key not in DATA or anchor_key[0] != key[0] or anchor_key[1] not in expected_weights:
                raise ValueError(f"{role} anchor is not an adjacent measured row at the same length")
            if anchor_key in seen:
                raise ValueError(f"{role} interpolation anchors must be distinct")
            seen.add(anchor_key)
            weight = anchor.get("weight")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError(f"{role} anchor weights must be finite and nonnegative")
            if not math.isclose(weight, expected_weights[anchor_key[1]], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{role} anchor weight does not interpolate the declared miss count")
            if anchor.get("compute_us") != DATA[anchor_key][1]:
                raise ValueError(f"{role} anchor compute differs from archived data")
            if "original_data_row" in anchor and tuple(anchor["original_data_row"]) != DATA[anchor_key]:
                raise ValueError(f"{role} anchor's embedded row differs from archived data")
            weights.append(weight)
            weighted_compute.append(weight * DATA[anchor_key][1])
        if not math.isclose(math.fsum(weights), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{role} anchor weights must sum to one")
        expected_compute = math.fsum(weighted_compute)
    else:
        raise ValueError(f"{role} must explicitly label constructed_profile as true or false")
    if not math.isclose(profile["compute_us"], expected_compute, rel_tol=1e-14, abs_tol=1e-9):
        raise ValueError(f"{role} compute time differs from its verified data construction")
    exact_volume = (profile["total_tokens"] - profile["nql"]) * 1408 / 2**30
    if not math.isclose(profile["read_gib"], exact_volume, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{role} read volume must equal actual hit tokens times 1408 bytes")


def validate_config(config):
    config = dict(config)
    config.setdefault("num_npu", 16)
    mode = config.get("mode", "random")
    if mode in ("random", "fixed"):
        config = _BASE_VALIDATE(config)
    elif mode in ("explicit", "cyclic"):
        # Reuse all topology/profile checks, then restore the explicit mode.
        proxy = dict(config, mode="random", input_counts=[1, 1])
        config = _BASE_VALIDATE(proxy)
        config["mode"] = mode
        # Do not expose the validation-only [1,1] as an actual input ratio.
        counts = Counter(controlled_decks(config)[0])
        config["input_counts"] = [counts["A"], counts["B"]]
        config["input_counts_semantics"] = "complete per-card counts; ordering is explicit/cyclic, not random"
    else:
        raise ValueError("mode must be random, fixed, explicit, or cyclic")
    for role in ("A", "B"):
        validate_profile_source(config["profile_" + role], role)
    return config


def build(config):
    if config["mode"] in ("random", "fixed"):
        return _BASE_BUILD(config)
    profiles = {role: config["profile_" + role] for role in ("A", "B")}
    decks = controlled_decks(config)
    requests, metadata = [], {}
    for npu, deck in enumerate(decks):
        for generation, role in enumerate(deck):
            p = profiles[role]
            rid = npu * 1_000_000 + generation
            prefix, disk_gib, layer = p["ssd_prefix_tokens"], [0.0] * config["ssu"], []
            for block in range(math.ceil(prefix / 128)):
                disk = base.sim.block_ring_hash_disk_id(rid, block, config["ssu"])
                size = min(128, prefix - 128 * block) * 1408 / 2**30
                layer.append((disk, size))
                disk_gib[disk] += size
            assert math.isclose(sum(disk_gib), p["read_gib"], abs_tol=1e-12)
            load = dict(request_id=rid, npu_id=npu, generation=generation, original_request_id=rid,
                total_tokens=p["total_tokens"], seq_len_k=p["total_length_k"], nql=p["nql"],
                role="L" if role == "A" else "S", profile_group=role,
                ssd_prefix_tokens=prefix, category=base.sim.classify_request(p["total_length_k"], p["nql"]),
                per_layer_us=p["compute_us"], per_layer_kv_gb=p["read_gib"],
                required_bw_input_gbps=p["B_gib_s"], arrival_time=0.0, arrival_ms=0.0, initial=True,
                constructed_profile=p["constructed_profile"], profile_construction=p["profile_construction"],
                original_compute_us=p["compute_us"], padding_gib_per_layer=0.0, disk_gib=disk_gib)
            requests.append(base.native.ContinuousBatchRequest.from_normalized(rid, npu, 0.0, load, (tuple(layer),)))
            metadata[rid] = load
    return tuple(requests), metadata, decks


def freeze_manifest(path, requests, config, decks):
    _BASE_FREEZE(path, requests, config, decks)
    if config["mode"] in ("explicit", "cyclic"):
        manifest = base.read_json(path)
        manifest["metadata"]["count_rule"] = (
            "Explicit equal-count complete per-NPU A/B decks; no shuffle" if config["mode"] == "explicit" else
            "Equal-count per-NPU A/B cycles, rotated by configured offsets and repeated; no shuffle")
        manifest["metadata"]["input_order_mode"] = config["mode"]
        manifest["metadata"]["per_npu_equal_AB_counts"] = True
        _BASE_SAVE(path, manifest)


def source_hashes():
    result = _BASE_SOURCE_HASHES()
    for path in (Path(__file__), DATA_PATH):
        result[str(path.resolve().relative_to(PROJECT))] = base.sha(path)
    return result


def _enrich_window(window):
    window["U_percent"] = 100 * window["fleet_utilization"]
    slo = window["slo_admitted"]
    window["SLO_1p5_percent"] = 100 * slo["rate"] if slo["count"] else None
    window["all_npus_compute_both_groups"] = all(
        all(row["by_group"][group]["compute_card_ms"] > 1e-9 for group in ("A", "B"))
        for row in window["per_npu"])
    window["npu_count_computing_both_groups"] = sum(
        all(row["by_group"][group]["compute_card_ms"] > 1e-9 for group in ("A", "B"))
        for row in window["per_npu"])
    window["available"] = True
    return window


def extended_windows(summary, metadata, config, primary):
    audit = base.audit
    n, s = config["num_npu"], config["ssu"]
    end = float(summary["makespan_ms"])
    rows, batches = summary["request_metrics"], summary["microbatch_metrics"]
    ordinary, pending, composition = [], [], []
    finish = [0.0] * n
    for batch in batches:
        rid = batch["member_request_ids"][0]
        meta = audit._meta_for(metadata, rid)
        rates = tuple(v * audit.GIB_TO_GB / (meta["per_layer_us"] / 1e6) for v in meta["disk_gib"])
        a, b = float(batch["admission_time_ms"]), float(batch["completion_time_ms"])
        ordinary.append((a, b, rates))
        composition.append((a, b, meta["profile_group"]))
        finish[int(batch["npu_id"])] = max(finish[int(batch["npu_id"])], b)
        for layer in batch["layer_metrics"]:
            if int(layer["layer"]) >= 1:
                pending.append((layer["io_start_time_ms"], layer["io_ready_time_ms"], rates))

    def one(start, stop):
        if stop > end + 1e-9:
            return dict(start_ms=start, end_ms=stop, available=False,
                        reason="requested window extends beyond completed finite run")
        window = audit._window_metrics(batches, rows, metadata, start, stop, n, config.get("slo_alpha", 1.5))
        for name, intervals in (("ordinary_demand", ordinary), ("ordinary_pending_demand", pending)):
            window[name] = audit._demand_scan(intervals, s, start, stop,
                config.get("disk_gb_s", 40.), config.get("near_gb_s", 36.))
        window["composition"] = audit._composition_scan(composition, start, stop, n)
        window["ends_before_first_npu_finishes"] = stop <= min(finish) + 1e-9
        return _enrich_window(window)

    windows = {"warm_2_4": one(2000., 4000.), "steady_2_10": one(2000., 10000.),
               "late_4_10": one(4000., 10000.)}
    bins = []
    for start in range(0, math.ceil(end), 2000):
        stop = min(float(start + 2000), end)
        if stop <= start:
            continue
        item = one(float(start), stop)
        item["full_2s_bin"] = stop == start + 2000
        bins.append(item)
    return dict(schema_version=1, order_mode=config["mode"], npu_count=n, ssu_count=s,
        full=_enrich_window(dict(primary["full"])), windows=windows, consecutive_2s=bins,
        per_npu_final_completion_ms=finish, first_npu_finishes_ms=min(finish), makespan_ms=end,
        definitions=dict(utilization="Exact compute overlap / (NPU count * window duration)",
            slo="Uncensored admission cohort [start,end), completion-admission <=1.5*own compute",
            AB_coverage="Both groups require positive actual compute overlap on each individual NPU",
            demand="Decimal GB/s. Sum active request per-disk V/C; extra next-request L0 excluded",
            physical_bandwidth="Nominal demand and pending envelope are not physical SSD throughput",
            short_runs="Unavailable named windows are not clipped; final 2s bin may be partial",
            ordering="explicit/cyclic is phase-controlled deterministic ordering, not random"))


def analyze(summary, metadata, config):
    result = _BASE_ANALYZE(summary, metadata, config)
    result["extended_windows"] = extended_windows(summary, metadata, config, result)
    return result


def save(path, value):
    if Path(path).name == "metrics.json" and "extended_windows" in value:
        _BASE_SAVE(Path(path).with_name("extended_windows.json"), value["extended_windows"])
    _BASE_SAVE(path, value)


def main():
    with patch.multiple(base, validate_config=validate_config, build=build,
                        freeze_manifest=freeze_manifest, source_hashes=source_hashes, save=save):
        with patch.object(base.audit, "analyze", analyze):
            base.main()


if __name__ == "__main__":
    main()
