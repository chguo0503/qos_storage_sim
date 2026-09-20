"""Frozen request manifests shared by input generators and experiment runners.

The schema stores normalized requests plus deduplicated placements. Loading
verifies the complete request fingerprint; saving refuses to replace a
manifest with different request content. Both JSON and JSON.gz are supported.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from simulator.core.continuous_batch_sim import (
    ContinuousBatchRequest,
    continuous_batch_input_fingerprint,
)


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def save_manifest(path, requests, metadata):
    """Deduplicate identical placements while preserving exact binary floats."""
    placements, indices, rows = [], {}, []
    for request in requests:
        placement = request.placement
        if placement not in indices:
            indices[placement] = len(placements)
            placements.append(placement)
        rows.append({"request_id": request.request_id, "npu_id": request.npu_id,
                     "arrival_time_ms": request.arrival_time_ms, "load": dict(request.load),
                     "placement_index": indices[placement]})
    payload = {"schema_version": 1, "metadata": metadata, "placements": placements,
               "requests": rows, "input_fingerprint": continuous_batch_input_fingerprint(requests)}
    path = Path(path)
    if path.exists():
        existing = read_json(path)
        if existing["input_fingerprint"] != payload["input_fingerprint"]:
            raise FileExistsError(f"preserving different manifest: {path}")
        return
    write_json(path, payload)


def load_manifest(path):
    payload = read_json(path)
    placements = [tuple(tuple((int(s), float(v)) for s, v in layer) for layer in p)
                  for p in payload["placements"]]
    requests = tuple(ContinuousBatchRequest.from_normalized(
        row["request_id"], row["npu_id"], row["arrival_time_ms"], row["load"],
        placements[row["placement_index"]]) for row in payload["requests"])
    if continuous_batch_input_fingerprint(requests) != payload["input_fingerprint"]:
        raise AssertionError("frozen manifest fingerprint mismatch")
    return requests, payload["metadata"]


