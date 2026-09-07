"""Authenticated request-profile loading for report-grade experiments."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import sim
from random_steady_state_workload import catalog_hash


ROOT = Path(__file__).resolve().parent


def canonical_bw_table(table) -> list[list[object]]:
    """Return a stable, JSON-serializable representation of a BW table."""

    return [
        [int(key[0]), int(key[1]), [float(value) for value in table[key]]]
        for key in sorted(table)
    ]


def bw_table_fingerprint(table) -> str:
    encoded = json.dumps(
        canonical_bw_table(table),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(b"authenticated-bw-table:v1\0" + encoded).hexdigest()


def load_authenticated_bw_table(num_npu: int):
    """Load directly from tracked ``data`` and reject cache substitution.

    ``sim.load_bw_table_cache`` normally prefers an optional results-side NPZ.
    Formal experiment provenance authenticates ``data``, so this loader parses
    that file itself. If a normal cache exists, it is also decoded and required
    to be scientifically identical; the returned table is always the direct
    parse.
    """

    num_npu = int(num_npu)
    if num_npu <= 0:
        raise ValueError("num_npu must be positive")
    data_path = ROOT / "data"
    raw = ast.literal_eval(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("data must contain a non-empty profile dictionary")
    direct = {}
    for raw_key, raw_value in raw.items():
        key = ast.literal_eval(raw_key) if isinstance(raw_key, str) else raw_key
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(f"invalid profile key: {raw_key!r}")
        profile_key = (int(key[0]), int(key[1]))
        if profile_key in direct or any(value <= 0 for value in profile_key):
            raise ValueError(f"duplicate or invalid profile key: {profile_key!r}")
        values = tuple(raw_value)
        if len(values) == 3:
            required_bw, per_layer_us, ttft_ms = values
            values = (
                required_bw,
                per_layer_us,
                ttft_ms,
                required_bw * per_layer_us / 1e6,
            )
        if len(values) != 4:
            raise ValueError(f"profile {profile_key!r} must have 3 or 4 values")
        converted = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value <= 0.0 for value in converted):
            raise ValueError(f"profile {profile_key!r} contains invalid values")
        direct[profile_key] = converted
    cache_path = ROOT / "results" / f"bw_table_cache_v2_{num_npu}npu.npz"
    if cache_path.exists():
        cached = sim.load_bw_table_cache(
            results_dir=str(ROOT / "results"), num_npu=num_npu
        )
        if canonical_bw_table(cached) != canonical_bw_table(direct):
            raise RuntimeError(
                f"unauthenticated bandwidth-table cache differs from data: {cache_path}"
            )
    return direct, {
        "source": "data",
        "source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "catalog_hash": catalog_hash(direct),
        "table_fingerprint": bw_table_fingerprint(direct),
        "profile_count": len(direct),
        "cache_path": str(cache_path.relative_to(ROOT)),
        "cache_present": cache_path.exists(),
        "cache_verified_equal": cache_path.exists(),
    }
