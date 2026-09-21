#!/usr/bin/env python3
"""Per-request A/B-family variants, using only interpolation inside frozen data.

CLI: --config FILE --strategy asu_baseline|od_baseline|once --output DIRECTORY.
Configuration adds mode="variant_explicit", profiles_catalog={key: profile}, and
per_npu_sequences=[[profile_key,...],...]. Every profile has family="A" or "B"
and the ordinary total_tokens/nql/compute_us/read_gib/B_gib_s/provenance fields.
All cards receive equal A/B family counts, but their exact profiles and total
compute work can differ. Work on every card must cover horizon_ms. Arrival is
zero, no idle/delay tokens are accepted, and IDs remain npu*1000000+position.

make_profile(total_tokens, miss, family) builds canonical auditable profiles.
Only adjacent measured length/miss cells are allowed, with nonnegative bilinear
weights. An interpolation is a modeled compute estimate, not a new data row.
No event engine, old runner, or strategy is modified. OD remains the original
unlimited-submission-depth implementation. Extra cross-request L0 is excluded
from nominal demand, while all actual service and waiting remain in the run.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import math
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("asu16_variant_standard_runner", HERE / "runner.py")
standard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(standard)
engine = standard.base
DATA = standard.DATA
DATA_SHA256 = standard.DATA_SHA256
LENGTHS = sorted({length for length, _ in DATA})
MISSES = sorted({miss for _, miss in DATA})


def _bracket(grid, target):
    if not grid[0] <= target <= grid[-1]:
        raise ValueError("Interpolation target lies outside the measured grid")
    if target in grid:
        return [(target, 1.)]
    lo = max(value for value in grid if value < target)
    hi = min(value for value in grid if value > target)
    return [(lo, (hi - target) / (hi - lo)), (hi, (target - lo) / (hi - lo))]


def _expected_weights(total_tokens, miss):
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or not 32 * 1024 <= total_tokens <= 200 * 1024:
        raise ValueError("total_tokens must be an integer in [32K,200K], K=1024")
    if isinstance(miss, bool) or not isinstance(miss, int) or not 64 <= miss <= 4096:
        raise ValueError("miss must be an integer in the measured [64,4096] range")
    weights = {(length, nql): wl * wm for length, wl in _bracket(LENGTHS, total_tokens / 1024)
               for nql, wm in _bracket(MISSES, miss)}
    if any(key not in DATA for key in weights):
        raise ValueError("Required interpolation corner is not present in data")
    return weights


def make_profile(total_tokens, miss, family):
    if family not in ("A", "B"):
        raise ValueError("family must be A or B")
    weights = _expected_weights(total_tokens, miss)
    key = (total_tokens / 1024, miss)
    direct = key in DATA
    length_interpolated = key[0] not in LENGTHS
    method = ("direct_data_row" if direct else "bilinear_interpolation_length_nql" if length_interpolated
              else "linear_interpolation_miss_same_length")
    compute = math.fsum(DATA[anchor][1] * weight for anchor, weight in weights.items())
    volume = (total_tokens - miss) * 1408 / 2**30
    return dict(family=family, total_tokens=total_tokens, total_length_k=total_tokens / 1024,
        nql=miss, ssd_prefix_tokens=total_tokens - miss, compute_us=compute,
        read_gib=volume, B_gib_s=volume * 1e6 / compute, constructed_profile=not direct,
        profile_construction=dict(method=method, source="data", data_sha256=DATA_SHA256,
            extrapolated=False, total_length_interpolated=length_interpolated, compute_scale=1.,
            kv_formula="exact hit prefix tokens * 1408 / 2**30; partial final block",
            interpretation="direct measured data row" if direct else "Compute time is a grid-internal interpolation estimate; KV volume uses exact hit tokens.",
            anchors=[dict(seq_len_k=length, nql=nql, weight=weight, compute_us=DATA[length, nql][1],
                          original_data_row=list(DATA[length, nql]))
                     for (length, nql), weight in weights.items()]))


def validate_profile(profile, key):
    if not isinstance(profile, dict) or profile.get("family") not in ("A", "B"):
        raise ValueError(f"{key}: profile needs family A or B")
    weights = _expected_weights(profile.get("total_tokens"), profile.get("nql"))
    length, miss = profile["total_tokens"] / 1024, profile["nql"]
    if profile.get("total_length_k") != length:
        raise ValueError(f"{key}: total_length_k does not match integer total_tokens")
    prefix = profile["total_tokens"] - miss
    if profile.get("ssd_prefix_tokens") != prefix:
        raise ValueError(f"{key}: prefix must equal total tokens minus miss")
    volume = prefix * 1408 / 2**30
    if not math.isclose(profile.get("read_gib", math.nan), volume, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{key}: read volume must be exact hit tokens times 1408 bytes")
    construction = profile.get("profile_construction", {})
    if not isinstance(construction, dict) or construction.get("source") != "data":
        raise ValueError(f"{key}: source must be data")
    hashes = [construction[field] for field in ("data_sha256", "source_sha256") if field in construction]
    if not hashes or any(value != DATA_SHA256 for value in hashes):
        raise ValueError(f"{key}: data SHA256 is missing or incorrect")
    if construction.get("extrapolated", False) or construction.get("compute_scale", 1) != 1:
        raise ValueError(f"{key}: extrapolation and compute scaling are prohibited")
    if profile.get("padding_gib_per_layer", 0) != 0:
        raise ValueError(f"{key}: read padding is prohibited")
    direct = (length, miss) in DATA
    if profile.get("constructed_profile") is not (not direct):
        raise ValueError(f"{key}: constructed_profile must distinguish grid points from interpolation")
    length_interpolated = length not in LENGTHS
    if "total_length_interpolated" in construction and construction["total_length_interpolated"] is not length_interpolated:
        raise ValueError(f"{key}: total_length_interpolated label is incorrect")
    methods = (("direct_data_row", "exact_data_row", "original_data_row") if direct else
               ("bilinear_interpolation_length_nql", "bilinear_interpolation_length_miss") if length_interpolated else
               ("linear_interpolation_miss_same_length", "bilinear_interpolation_length_nql"))
    if construction.get("method") not in methods:
        raise ValueError(f"{key}: provenance method does not describe the accepted construction")
    anchors = construction.get("anchors")
    # Old direct-row configs can identify the exact source with source_key.
    if direct and anchors is None:
        if tuple(construction.get("source_key", ())) != (length, miss):
            raise ValueError(f"{key}: direct profile requires an exact source_key or anchor")
        anchors = [dict(seq_len_k=length, nql=miss, weight=1., compute_us=DATA[length, miss][1])]
        if "original_row" in construction and tuple(construction["original_row"]) != DATA[length, miss]:
            raise ValueError(f"{key}: embedded measured row differs from data")
    if not isinstance(anchors, list) or len(anchors) != len(weights):
        raise ValueError(f"{key}: anchors must be exactly the nonzero enclosing cell corners")
    seen, submitted_weights = set(), []
    for anchor in anchors:
        source = (anchor.get("seq_len_k"), anchor.get("nql"))
        if source not in DATA or source not in weights or source in seen:
            raise ValueError(f"{key}: anchor is duplicated, missing, or outside the enclosing cell")
        seen.add(source)
        weight = anchor.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"{key}: weights must be finite and nonnegative")
        if not math.isclose(weight, weights[source], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{key}: weights do not interpolate the requested length and miss")
        if anchor.get("compute_us") != DATA[source][1]:
            raise ValueError(f"{key}: anchor compute does not match archived data")
        if "original_data_row" in anchor and tuple(anchor["original_data_row"]) != DATA[source]:
            raise ValueError(f"{key}: embedded anchor row does not match archived data")
        submitted_weights.append(weight)
    if not math.isclose(math.fsum(submitted_weights), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{key}: weights must sum to one")
    expected_compute = math.fsum(DATA[source][1] * weight for source, weight in weights.items())
    actual_compute = profile.get("compute_us", math.nan)
    if not math.isclose(actual_compute, expected_compute, rel_tol=1e-14, abs_tol=1e-9):
        raise ValueError(f"{key}: compute is not the verified weighted data value")
    if not math.isclose(profile.get("B_gib_s", math.nan), volume * 1e6 / actual_compute, rel_tol=1e-12):
        raise ValueError(f"{key}: bandwidth demand must equal verified V/C")


def validate_config(config):
    config = dict(config)
    if config.get("mode", "variant_explicit") != "variant_explicit":
        raise ValueError("Variant inputs must be explicitly labeled mode=variant_explicit")
    catalog = config.get("profiles_catalog")
    if not isinstance(catalog, dict) or not catalog or any(not isinstance(key, str) or not key for key in catalog):
        raise ValueError("profiles_catalog must be a nonempty map with string keys")
    for key, profile in catalog.items():
        validate_profile(profile, key)
    by_family = {family: [p for p in catalog.values() if p["family"] == family] for family in ("A", "B")}
    if any(not values for values in by_family.values()):
        raise ValueError("profiles_catalog must contain both families")
    if any(field in config for field in ("profile_A", "profile_B", "decks", "cycle", "cycles_by_npu")):
        raise ValueError("Use profiles_catalog/per_npu_sequences exclusively; legacy order fields are ambiguous")
    # Reuse the original topology/time/byte checks with temporary representatives.
    proxy = dict(config, mode="random", num_npu=config.get("num_npu", 16), input_counts=[1, 1],
                 profile_A=by_family["A"][0], profile_B=by_family["B"][0])
    normalized = standard._BASE_VALIDATE(proxy)
    for field in ("profile_A", "profile_B", "input_counts"):
        normalized.pop(field)
    normalized["mode"] = "variant_explicit"
    n = normalized["num_npu"]
    sequences = normalized.get("per_npu_sequences")
    if not isinstance(sequences, list) or len(sequences) != n:
        raise ValueError("per_npu_sequences must provide one sequence per NPU")
    expected_counts = None
    for card, sequence in enumerate(sequences):
        if not isinstance(sequence, list) or not sequence or len(sequence) >= 1_000_000:
            raise ValueError(f"NPU {card}: sequence must be nonempty and fit the request-ID stride")
        if any(not isinstance(key, str) or key not in catalog for key in sequence):
            raise ValueError(f"NPU {card}: sequence contains unknown profiles or an idle/delay token")
        counts = Counter(catalog[key]["family"] for key in sequence)
        if set(counts) != {"A", "B"}:
            raise ValueError(f"NPU {card}: both families must occur")
        if expected_counts is None:
            expected_counts = counts
        elif counts != expected_counts:
            raise ValueError("Every NPU must receive equal A/B family counts")
        compute_ms = 8 * math.fsum(catalog[key]["compute_us"] for key in sequence) / 1000
        if compute_ms + 1e-9 < normalized["horizon_ms"]:
            raise ValueError(f"NPU {card}: pure compute work does not cover horizon_ms")
    normalized["input_counts"] = [expected_counts["A"], expected_counts["B"]]
    normalized["input_counts_semantics"] = "per-card complete A/B-family counts; per-request profiles may vary"
    return normalized


def build(config):
    requests, metadata, decks = [], {}, []
    catalog, ssu = config["profiles_catalog"], config["ssu"]
    for card, sequence in enumerate(config["per_npu_sequences"]):
        decks.append([catalog[key]["family"] for key in sequence])
        for position, key in enumerate(sequence):
            profile = catalog[key]
            family = profile["family"]
            rid = card * 1_000_000 + position
            prefix, disk_gib, blocks = profile["ssd_prefix_tokens"], [0.] * ssu, []
            for block in range(math.ceil(prefix / 128)):
                disk = engine.sim.block_ring_hash_disk_id(rid, block, ssu)
                size = min(128, prefix - block * 128) * 1408 / 2**30
                blocks.append((disk, size));disk_gib[disk] += size
            assert math.isclose(math.fsum(disk_gib), profile["read_gib"], abs_tol=1e-12)
            load = dict(request_id=rid, npu_id=card, generation=position, original_request_id=rid,
                total_tokens=profile["total_tokens"], seq_len_k=profile["total_length_k"], nql=profile["nql"],
                role="L" if family == "A" else "S", profile_group=family, family=family, profile_key=key,
                ssd_prefix_tokens=prefix, category=engine.sim.classify_request(profile["total_length_k"], profile["nql"]),
                per_layer_us=profile["compute_us"], per_layer_kv_gb=profile["read_gib"],
                required_bw_input_gbps=profile["B_gib_s"], arrival_time=0., arrival_ms=0., initial=True,
                constructed_profile=profile["constructed_profile"], profile_construction=profile["profile_construction"],
                original_compute_us=profile["compute_us"], padding_gib_per_layer=0., disk_gib=disk_gib)
            requests.append(engine.native.ContinuousBatchRequest.from_normalized(rid, card, 0., load, (tuple(blocks),)))
            metadata[rid] = load
    return tuple(requests), metadata, decks


def freeze_manifest(path, requests, config, decks):
    standard._BASE_FREEZE(path, requests, config, decks)
    manifest = engine.read_json(path)
    per_card_compute = [8 * math.fsum(config["profiles_catalog"][key]["compute_us"] for key in sequence) / 1000
                        for sequence in config["per_npu_sequences"]]
    manifest["metadata"].update(input_order_mode="variant_explicit",
        count_rule="Explicit per-card profile-key sequences; equal family counts, possibly different measured/interpolated profiles; no idle and no shuffle",
        per_npu_equal_AB_counts=True, per_npu_equal_compute_work_required=False,
        per_npu_equal_compute_work=all(math.isclose(value, per_card_compute[0], rel_tol=0, abs_tol=1e-8)
                                     for value in per_card_compute),
        per_npu_pure_compute_ms=per_card_compute,
        profiles_catalog=config["profiles_catalog"], per_npu_profile_keys=config["per_npu_sequences"],
        profile_source="Archived measured data or adjacent grid-internal bilinear interpolation; no extrapolation")
    standard._BASE_SAVE(path, manifest)
    restored, _ = engine.load_manifest(path)
    assert engine.native.continuous_batch_input_fingerprint(restored) == engine.native.continuous_batch_input_fingerprint(requests)


def source_hashes():
    hashes = standard.source_hashes()
    hashes[str(Path(__file__).resolve().relative_to(standard.PROJECT))] = engine.sha(Path(__file__))
    return hashes


def main():
    with patch.multiple(engine, validate_config=validate_config, build=build,
                        freeze_manifest=freeze_manifest, source_hashes=source_hashes, save=standard.save):
        with patch.object(engine.audit, "analyze", standard.analyze):
            engine.main()


if __name__ == "__main__":
    main()
