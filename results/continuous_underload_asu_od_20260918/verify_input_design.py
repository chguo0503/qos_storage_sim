#!/usr/bin/env python3
"""Read-only verification of frozen manifests against the document's queues.

Prints JSON to stdout. Does not generate/rewrite inputs, run a simulation,
write reports, or change the simulator. Repeat --manifest for paired inputs.
"""

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulator.core import sim

NUM_NPU, NUM_SSU, N_LAYERS = 32, 3, 8
CAPACITY_GIB_S = 40.
IO_BYTES, IO_GIB = 176 * 1024, 176 * 1024 / 2**30
KEYS = tuple((length, miss) for length in (32, 64, 80, 128, 160)
             for miss in (2048, 4096))
CANONICAL = tuple(key for key in KEYS for _ in range(2))
REQUESTS_PER_NPU = len(CANONICAL)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11), (actual, expected)


def reference_profiles():
    table = ast.literal_eval((ROOT / "data").read_text(encoding="utf-8"))
    result = {}
    for key in KEYS:
        assert key in table and len(table[key]) == 4
        b, c_us, source_ttft_ms, v_gib = map(float, table[key])
        close(b, v_gib * 1e6 / c_us)
        close(source_ttft_ms, 78 * c_us / 1000.)
        blocks = (key[0] * 1024 - key[1]) // 128
        close(v_gib, blocks * IO_GIB)
        result[key] = dict(B_gib_s=b, C_us=c_us, V_gib=v_gib,
                           source_ttft_ms=source_ttft_ms, blocks=blocks)
    return result


def read_manifest(path, profiles):
    path = path.resolve()
    digest = sha(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        doc = json.load(stream)
    meta = doc["metadata"]
    assert (meta["num_npu"], meta["num_ssu"], meta["n_layers"]) == (NUM_NPU, NUM_SSU, N_LAYERS)
    assert meta["layout"] == sim.PLACEMENT_BLOCK_RING_HASH
    order, seed = meta["order"], int(meta["seed"])
    assert order in ("random", "ordered")
    if "source_data_sha256" in meta:
        assert meta["source_data_sha256"] == sha(ROOT / "data")
    assert len(doc["requests"]) == NUM_NPU * REQUESTS_PER_NPU == 640
    by_npu = [[] for _ in range(NUM_NPU)]
    for q in doc["requests"]:
        assert 0 <= q["npu_id"] < NUM_NPU
        by_npu[q["npu_id"]].append(q)
    inventory = {}
    rates_by_npu, stripe_by_npu = [], []
    compute_ms_by_npu, work_by_npu = [], []
    for npu, requests in enumerate(by_npu):
        assert len(requests) == REQUESTS_PER_NPU
        ordinals = list(range(REQUESTS_PER_NPU))
        if order == "random":
            random.Random(seed + npu).shuffle(ordinals)
        # Independently shuffle the tuples exactly as in the user document;
        # stable identities must not change its profile queue sequence.
        documented_queue = list(CANONICAL)
        if order == "random":
            random.Random(seed + npu).shuffle(documented_queue)
        assert [CANONICAL[o] for o in ordinals] == documented_queue
        rates, stripe_rates, computes, works = [], [], [], []
        seen = Counter()
        for position, (q, ordinal, expected_key) in enumerate(zip(requests, ordinals, documented_queue)):
            runtime_id = npu * REQUESTS_PER_NPU + position
            stable_id = npu * REQUESTS_PER_NPU + ordinal
            load = q["load"]
            key = (int(load["seq_len_k"]), int(load["nql"]))
            assert key == expected_key
            seen[key] += 1
            assert q["request_id"] == runtime_id
            assert load["original_request_id"] == stable_id
            assert q["arrival_time_ms"] == 0.
            if "generation" in load:
                assert load["generation"] == position
            p = profiles[key]
            close(load["per_layer_us"], p["C_us"])
            close(load["per_layer_kv_gb"], p["V_gib"])
            close(load["required_bw_input_gbps"], p["B_gib_s"])
            if "source_ttft_ms" in load:
                close(load["source_ttft_ms"], p["source_ttft_ms"])
            assert not load.get("constructed_profile", False)
            placement = doc["placements"][q["placement_index"]]
            assert len(placement) == 1
            layer = placement[0]
            assert len(layer) == p["blocks"]
            counts, stripe_counts = [0] * NUM_SSU, [0] * NUM_SSU
            physical = []
            for block_index, (disk, size) in enumerate(layer):
                close(size, IO_GIB)
                assert disk == sim.block_ring_hash_disk_id(stable_id, block_index, NUM_SSU)
                counts[disk] += 1
                stripe_counts[(block_index + npu) % NUM_SSU] += 1
                physical.append((disk, size))
            volumes = tuple(count * IO_GIB for count in counts)
            close(math.fsum(volumes), p["V_gib"])
            rates.append(tuple(v * 1e6 / p["C_us"] for v in volumes))
            stripe_rates.append(tuple(count * IO_GIB * 1e6 / p["C_us"] for count in stripe_counts))
            computes.append(N_LAYERS * p["C_us"] / 1000.)
            works.append(tuple(N_LAYERS * v for v in volumes))
            assert stable_id not in inventory
            inventory[stable_id] = (npu, key, p["C_us"], tuple(physical))
        assert seen == Counter({key: 2 for key in KEYS})
        rates_by_npu.append(rates)
        stripe_by_npu.append(stripe_rates)
        compute_ms_by_npu.append(math.fsum(computes))
        work_by_npu.append(tuple(math.fsum(w[s] for w in works) for s in range(NUM_SSU)))
    assert len(set(compute_ms_by_npu)) == 1
    upper = [math.fsum(max(r[s] for r in rates) for rates in rates_by_npu) for s in range(NUM_SSU)]
    stripe_upper = [math.fsum(max(r[s] for r in rates) for rates in stripe_by_npu) for s in range(NUM_SSU)]
    mean = [math.fsum(work_by_npu[n][s] * 1000 / compute_ms_by_npu[n]
                      for n in range(NUM_NPU)) for s in range(NUM_SSU)]
    synchronized = []
    for key in ((128, 2048), (160, 2048)):
        for copy in range(2):
            ordinal = CANONICAL.index(key) + copy
            values = []
            for npu in range(NUM_NPU):
                stable_id = npu * REQUESTS_PER_NPU + ordinal
                physical = inventory[stable_id][3]
                counts = Counter(d for d, _ in physical)
                values.append([counts[s] * IO_GIB * 1e6 / profiles[key]["C_us"] for s in range(NUM_SSU)])
            synchronized.append(dict(profile=f"{key[0]}:{key[1]}", copy=copy,
                                     per_disk_demand_gib_s=[math.fsum(v[s] for v in values) for s in range(NUM_SSU)]))
    assert sha(path) == digest
    audit = dict(manifest=str(path), manifest_sha256=digest, order=order, seed=seed,
                 request_count=len(doc["requests"]), profiles=len(profiles), requests_per_npu=REQUESTS_PER_NPU,
                 stable_identity="npu_id * 20 + canonical_ordinal",
                 runtime_identity="npu_id * 20 + queue_position",
                 shuffle="Random(seed + npu_id); same profile queue as document",
                 per_npu_pure_compute_ms=compute_ms_by_npu[0],
                 static_upper_gib_s_by_disk=upper,
                 static_upper_proves_strict_underload=all(v < CAPACITY_GIB_S for v in upper),
                 theoretical_mean_gib_s_by_disk=mean,
                 theoretical_mean_total_gib_s=math.fsum(mean),
                 counterfactual_old_stripe_upper_gib_s_by_disk=stripe_upper,
                 synchronized_profile_demand_examples=synchronized,
                 caveat="A static upper bound >40 disproves the input-only guarantee, not by itself proof of actual run overload. No simulator was run.",
                 checks="Passed profile/raw data, identity, exact documented queue, placement, layer/arrival/count checks; input unchanged")
    return audit, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="Frozen JSON or JSON.gz manifest; repeat to verify common physical request inventory")
    args = parser.parse_args()
    data_sha = sha(ROOT / "data")
    profiles = reference_profiles()
    audits, expected_inventory = [], None
    for path in args.manifest:
        audit, inventory = read_manifest(path, profiles)
        if expected_inventory is None:
            expected_inventory = inventory
        else:
            assert inventory == expected_inventory, "Paired order/seed inputs changed physical request inventory"
        audits.append(audit)
    assert sha(ROOT / "data") == data_sha
    print(json.dumps(dict(source_data_sha256=data_sha, files_checked=len(audits),
                          identical_physical_inventory=True, audits=audits), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
