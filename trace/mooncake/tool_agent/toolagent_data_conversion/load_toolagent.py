"""Exact-key readers for the converted toolagent artifacts (standard library).

These readers preserve fractional Ki-token keys and zero-KV profiles. They are
not adapters for existing simulator scheduling, batching, or request routing.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path


def load_profiles(path: str | Path) -> dict[tuple[float, int], tuple[float, float, float, float]]:
    profiles = ast.literal_eval(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profiles, dict):
        raise ValueError("profile file must contain a dictionary")
    for key, value in profiles.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"invalid profile key: {key!r}")
        seq_len_k, uncached = key
        if not isinstance(seq_len_k, (int, float)) or not math.isfinite(seq_len_k):
            raise ValueError(f"nonfinite input length: {key!r}")
        length = seq_len_k * 1024
        if length < 1 or length != int(length) or type(uncached) is not int or not 1 <= uncached <= length:
            raise ValueError(f"nonreversible or invalid token key: {key!r}")
        if not isinstance(value, tuple) or len(value) != 4:
            raise ValueError(f"invalid four-field profile: {key!r}")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
            raise ValueError(f"invalid profile number: {key!r}")
        bandwidth, compute_us, compute_ttft_ms, layer_gib = value
        if bandwidth < 0 or compute_us <= 0 or compute_ttft_ms <= 0 or layer_gib < 0:
            raise ValueError(f"invalid profile sign: {key!r}")
        expected_gib = (length - uncached) * 4096 / 2**30
        if layer_gib != expected_gib:
            raise ValueError(f"token/byte mismatch: {key!r}")
        if not math.isclose(bandwidth * compute_us * 1e-6, layer_gib, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"bandwidth/compute/volume mismatch: {key!r}")
        if not math.isclose(compute_ttft_ms, 32 * compute_us / 1000, rel_tol=1e-12):
            raise ValueError(f"32-layer pure compute mismatch: {key!r}")
        if layer_gib == 0 and bandwidth != 0:
            raise ValueError(f"zero-KV bandwidth mismatch: {key!r}")
    return profiles


def load_requests(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if row["hit_tokens"] + row["u_tokens"] != row["input_length"]:
            raise ValueError(f"token conservation failed at source_line {row['source_line']}")
        if row["u_tokens"] < 1:
            raise ValueError("uncached token count must be positive")
        if row["profile_key"] != [row["input_length"] / 1024, row["u_tokens"]]:
            raise ValueError("request profile key does not preserve token counts")
    return rows
