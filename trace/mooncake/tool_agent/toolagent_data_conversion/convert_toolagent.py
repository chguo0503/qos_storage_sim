#!/usr/bin/env python3
"""Convert Mooncake toolagent JSONL to a synthetic qos_storage_sim data table.

Standard library only. This conversion does not run a simulator or assign NPUs.
The output dictionary uses exact fractional Ki-token keys and keeps zero-KV
profiles, so it must not be read by loaders that round K or reject zero values.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
from typing import Any

BLOCK_TOKENS = 512
BYTES_PER_TOKEN_PER_LAYER = 4096
LAYERS = 32
REFERENCE_BYTES_PER_SECOND = 80e9
GIB = 2**30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append({**value, "source_line": line_number})
    return rows


def derive_hits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (row["timestamp"], row["source_line"]))
    seen: set[Any] = set()
    result = []
    for _, group_iterator in groupby(rows, key=lambda row: row["timestamp"]):
        group = list(group_iterator)
        for row in group:
            length = row["input_length"]
            if type(length) is not int or length < 1:
                raise ValueError(f"line {row['source_line']}: input_length must be a positive integer")
            if not isinstance(row["hash_ids"], list):
                raise ValueError(f"line {row['source_line']}: hash_ids must be a list")
            if not isinstance(row["timestamp"], (int, float)) or not math.isfinite(row["timestamp"]):
                raise ValueError(f"line {row['source_line']}: timestamp must be finite")
            prefix_blocks = 0
            for block in row["hash_ids"]:  # Includes the final partial block.
                if block not in seen:
                    break
                prefix_blocks += 1
            hit = min(prefix_blocks * BLOCK_TOKENS, length - 1)
            result.append({
                **row,
                "hit_tokens": hit,
                "u_tokens": length - hit,
                "full_hit_adjusted": hit >= length - 1,
            })
        # All arrivals at the same timestamp see only earlier timestamp groups.
        for row in group:
            seen.update(row["hash_ids"])
    return result


def synthetic_profile(hit: int, uncached: int) -> tuple[float, float, float, float]:
    eta = min(1.0, max(1.0 / 16.0, uncached / 512.0))
    attention = uncached * hit + uncached * (uncached + 1) / 2
    compute_s = 50e-6 + 2e-7 * uncached / eta + 1e-10 * attention
    layer_gib = hit * BYTES_PER_TOKEN_PER_LAYER / GIB
    return (
        layer_gib / compute_s,  # GiB/s: legacy column name contains "gbps".
        compute_s * 1e6,  # Microseconds per layer.
        LAYERS * compute_s * 1000,  # Milliseconds of pure compute, NOT T0.
        layer_gib,  # GiB per layer: legacy column name contains "gb".
    )


def verify_derived(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    expected = read_jsonl(path)
    if len(rows) != len(expected):
        raise ValueError(f"derived row count differs: {len(expected)} != {len(rows)}")
    fields = (
        "timestamp", "input_length", "output_length", "hash_ids",
        "hit_tokens", "u_tokens", "full_hit_adjusted",
    )
    for position, (got, reference) in enumerate(zip(rows, expected), 1):
        for field in fields:
            if got[field] != reference[field]:
                raise ValueError(
                    f"derived mismatch at sorted row {position}, field {field}: "
                    f"{got[field]!r} != {reference[field]!r}"
                )
    return {
        "performed": True,
        "passed": True,
        "rows_compared": len(rows),
        "fields_compared": list(fields),
        "source_path": str(path.resolve()),
        "source_sha256": sha256(path),
    }


def convert(input_path: Path, output_dir: Path, derived_path: Path | None) -> dict[str, Any]:
    rows = derive_hits(read_jsonl(input_path))
    if not rows:
        raise ValueError("input contains no requests")
    derived_check = verify_derived(rows, derived_path) if derived_path else {"performed": False}
    profiles: dict[tuple[float, int], tuple[float, float, float, float]] = {}
    requests: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    max_bandwidth_volume_error = 0.0
    checks = {
        "input_equals_hit_plus_uncached": True,
        "uncached_at_least_one": True,
        "fractional_k_round_trip_exact": True,
        "kv_bytes_round_trip_exact": True,
        "bandwidth_times_compute_equals_kv_within_float_tolerance": True,
        "profile_key_maps_to_unique_hit_uncached": True,
        "every_request_has_profile": True,
        "stable_timestamp_source_line_order": True,
        "request_count_preserved": True,
    }
    for row in rows:
        length, hit, uncached = row["input_length"], row["hit_tokens"], row["u_tokens"]
        key = (length / 1024, uncached)
        value = synthetic_profile(hit, uncached)
        if key in profiles and profiles[key] != value:
            raise ValueError(f"profile key collision: {key!r}")
        profiles[key] = value
        bandwidth_gibs, compute_us, compute_ttft_ms, layer_gib = value
        compute_s = compute_us * 1e-6
        read_s = hit * BYTES_PER_TOKEN_PER_LAYER / REFERENCE_BYTES_PER_SECOND
        t0_ms = (read_s + LAYERS * compute_s + (LAYERS - 1) * max(0, read_s - compute_s)) * 1000
        volume_error = abs(bandwidth_gibs * compute_s - layer_gib)
        max_bandwidth_volume_error = max(max_bandwidth_volume_error, volume_error)
        checks["input_equals_hit_plus_uncached"] &= length == hit + uncached
        checks["uncached_at_least_one"] &= uncached >= 1
        checks["fractional_k_round_trip_exact"] &= key[0] * 1024 == length
        checks["kv_bytes_round_trip_exact"] &= layer_gib * GIB == hit * BYTES_PER_TOKEN_PER_LAYER
        checks["bandwidth_times_compute_equals_kv_within_float_tolerance"] &= math.isclose(
            bandwidth_gibs * compute_s, layer_gib, rel_tol=1e-12, abs_tol=1e-15
        )
        counts["zero_hit_requests"] += hit == 0
        counts["u_equals_one_requests"] += uncached == 1
        counts["full_hit_adjusted_requests"] += row["full_hit_adjusted"]
        totals["input_tokens"] += length
        totals["hit_tokens"] += hit
        totals["uncached_tokens"] += uncached
        requests.append({
            **row,
            "request_id": len(requests),
            "arrival_time": row["timestamp"],
            "arrival_time_ms": row["timestamp"],
            "profile_key": list(key),
            "seq_len_k": key[0],
            "nql": uncached,
            "required_bw_input_gbps": bandwidth_gibs,
            "per_layer_us": compute_us,
            "per_layer_kv_gb": layer_gib,
            "source_compute_ttft_ms": compute_ttft_ms,
            "source_T0_ms": t0_ms,
            "source_slo4_ms": 4 * t0_ms,
        })
    counts["zero_hit_profiles"] = sum(value[3] == 0 for value in profiles.values())
    counts["u_equals_one_profiles"] = sum(key[1] == 1 for key in profiles)
    checks["every_request_has_profile"] = all(tuple(row["profile_key"]) in profiles for row in requests)
    ordering = [(row["timestamp"], row["source_line"]) for row in requests]
    checks["stable_timestamp_source_line_order"] = ordering == sorted(ordering)
    if not all(checks.values()):
        raise ValueError(f"conversion invariant failed: {checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "data_toolagent"
    request_path = output_dir / "toolagent_requests.jsonl"
    report_path = output_dir / "conversion_report.json"
    with profile_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "# Synthetic-v1 singleton profiles; not measured device timings.\n"
            "# key: (input_tokens / 1024, uncached_tokens); fractional K is exact, never rounded.\n"
            "# value: (required bandwidth GiB/s, compute us/layer, pure-compute TTFT ms, KV GiB/layer).\n"
            "# Third field = 32 * layer compute, retaining source L=32; original GLM data uses L=78.\n"
            "# Zero-hit profiles are retained with zero bandwidth and zero KV bytes.\n"
            "# Not a drop-in file for loaders that round fractional K or discard zero values.\n"
            "{\n"
        )
        for key in sorted(profiles):
            stream.write(f"    {key!r}: {profiles[key]!r},\n")
        stream.write("}\n")
    with request_path.open("w", encoding="utf-8") as stream:
        for row in requests:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    # Validate the actual serialized dictionary, including exact keys and zeros.
    parsed_profiles = ast.literal_eval(profile_path.read_text(encoding="utf-8"))
    checks["serialized_profile_round_trip_exact"] = parsed_profiles == profiles
    with request_path.open(encoding="utf-8") as stream:
        parsed_requests = [json.loads(line) for line in stream]
    checks["serialized_request_round_trip_exact"] = parsed_requests == requests
    checks["request_count_preserved"] = len(parsed_requests) == len(rows)
    if not all(checks.values()):
        raise ValueError(f"serialized output invariant failed: {checks}")

    report = {
        "schema_version": 1,
        "source": {
            "path": str(input_path.resolve()),
            "sha256": sha256(input_path),
            "rows": len(rows),
        },
        "output": {
            "requests": len(requests),
            "unique_profiles": len(profiles),
            "profiles_file": profile_path.name,
            "profiles_sha256": sha256(profile_path),
            "requests_file": request_path.name,
            "requests_sha256": sha256(request_path),
        },
        "statistics": {**counts, **totals,
            "token_hit_ratio": totals["hit_tokens"] / totals["input_tokens"],
            "distinct_timestamps": len(set(row["timestamp"] for row in requests)),
        },
        "profile_model": {
            "name": "synthetic-v1-singleton",
            "measured_timings": False,
            "layers": LAYERS,
            "block_tokens": BLOCK_TOKENS,
            "bytes_per_token_per_layer": BYTES_PER_TOKEN_PER_LAYER,
            "compute_seconds_formula": "50e-6 + 2e-7*u/eta + 1e-10*(u*h + u*(u+1)/2)",
            "eta_formula": "min(1, max(1/16, u/512))",
            "source_T0_seconds_formula": "r + 32*c + 31*max(0, r-c)",
            "read_seconds_formula": "r = 4096*h/(80e9)",
            "reference_bandwidth_bytes_per_second": REFERENCE_BYTES_PER_SECOND,
            "source_slo_multiplier": 4,
            "third_field_definition": "32*c*1000 ms of pure compute; excludes IO and queueing",
            "third_field_difference_from_original_data": "original GLM data uses 78*c*1000 ms; this source model uses 32 layers",
        },
        "units": {
            "timestamp": "milliseconds, original source timestamp retained",
            "arrival_time": "milliseconds, equal to timestamp; no arrival scaling",
            "arrival_time_ms": "milliseconds, same value as arrival_time",
            "source_line": "1-based original physical JSONL line number",
            "seq_len_k": "input tokens / 1024, exact fractional Ki-tokens",
            "nql": "uncached tokens, integer >= 1; not rounded to multiples of 512",
            "profile_key": "[seq_len_k, nql] in JSON; tuple in data_toolagent",
            "required_bw_input_gbps": "GiB/s despite legacy field spelling; not decimal GB/s or bits/s",
            "per_layer_us": "microseconds",
            "per_layer_kv_gb": "GiB despite legacy field spelling; not decimal GB",
            "source_compute_ttft_ms": "milliseconds, pure compute for 32 layers",
            "source_T0_ms": "milliseconds, isolated reference read/compute overlap time",
            "source_slo4_ms": "milliseconds, 4 * source_T0_ms",
        },
        "hit_derivation": {
            "algorithm": "first_seen, contiguous prefix; same-timestamp arrivals mutually invisible; tail block participates",
            "hit_formula": "min(512 * matching_prefix_blocks, input_length - 1)",
            "u_formula": "input_length - hit_tokens",
            "full_hit_adjusted_formula": "hit_tokens >= input_length - 1",
            "cache_capacity_or_eviction_modeled": False,
            "wait_for_previous_request_completion_before_visibility": False,
        },
        "scope": {
            "simulator_run": False,
            "npu_assignment": False,
            "batching_modeled": False,
            "request_order": "stable sort by (timestamp, original source_line)",
            "compatibility": "dictionary syntax matches data; use the supplied exact-key loader; not all existing loaders preserve fractional K or zero profiles",
        },
        "derived_validation": derived_check,
        "checks": checks,
        "max_bandwidth_times_compute_volume_error_gib": max_bandwidth_volume_error,
        "all_checks_passed": all(checks.values()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw toolagent trace JSONL")
    parser.add_argument("--derived", type=Path, help="Optional upstream .hit.jsonl; require exact row-by-row match")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = convert(args.input, args.output_dir, args.derived)
    print(json.dumps({
        "requests": report["output"]["requests"],
        "unique_profiles": report["output"]["unique_profiles"],
        "checks_passed": report["all_checks_passed"],
        "derived_validation": report["derived_validation"].get("passed", "not requested"),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
