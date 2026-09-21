#!/usr/bin/env python3
"""Validation-only passive per-request/layer SSD+HBM completed-byte probe."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[1] / "variant_runner.py"
spec = importlib.util.spec_from_file_location("variant_native_byte_probe", RUNNER)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
original_complete = v.engine.native._register_complete
original_hashes = v.source_hashes
observed, blocks = {}, set()


def completed(context, flow):
    result = original_complete(context, flow)
    identity = (flow.request_id, flow.layer, flow.block_idx)
    assert identity not in blocks
    blocks.add(identity)
    key = (flow.request_id, flow.layer, flow.disk_id)
    row = observed.setdefault(key, dict(blocks=0, gib=0.))
    row["blocks"] += flow.block_count
    row["gib"] += flow.total_gb
    return result


def hashes():
    result = original_hashes()
    result[str(Path(__file__).resolve().relative_to(v.standard.PROJECT))] = v.engine.sha(Path(__file__))
    return result


with patch.object(v.engine.native, "_register_complete", completed), patch.object(v, "source_hashes", hashes):
    v.main()
out = Path(sys.argv[sys.argv.index("--output") + 1])
manifest = v.engine.read_json(out / "manifest.json.gz")
expected = {}
for request in manifest["requests"]:
    placement = manifest["placements"][request["placement_index"]]
    assert len(placement) == 1
    for layer in range(8):
        for disk, size in placement[0]:
            key = (request["request_id"], layer, disk)
            row = expected.setdefault(key, dict(blocks=0, gib=0.))
            row["blocks"] += 1
            row["gib"] += size
assert set(observed) == set(expected)
checks = []
for key in sorted(expected):
    e, o = expected[key], observed[key]
    checks.append(dict(request_id=key[0], layer=key[1], ssu=key[2],
        expected_blocks=e["blocks"], completed_blocks=o["blocks"],
        expected_gib=e["gib"], completed_gib=o["gib"],
        passed=e["blocks"] == o["blocks"] and math.isclose(e["gib"], o["gib"], rel_tol=0, abs_tol=1e-12)))
assert all(row["passed"] for row in checks)
v.standard._BASE_SAVE(out / "completed_bytes_probe.json", dict(passed=True,
    definition="Passive completion hook; actual completed flow.total_gb and block_count per request/layer/disk",
    unique_completed_blocks=len(blocks), probe_source_sha256=v.engine.sha(Path(__file__)), checks=checks))
