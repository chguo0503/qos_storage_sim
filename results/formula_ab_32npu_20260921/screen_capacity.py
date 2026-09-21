#!/usr/bin/env python3
"""Reproduce the 32-NPU static capacity screen without running a simulation.

By default this recomputes and verifies static_capacity_screen.json, without
rewriting it. Use --output NEW_FILE to save a new copy. Frozen source/configs
are read only; only build()'s NPU loop changes from range(8) to range(32).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = REPO / "template/qos_experiments_20260919/03_continuous_underload/formula_ab/original_recovered"
GROUPS = ("XY12_32", "XY12_24", "XY12_20", "XY12_16", "X16")
SEEDS = (7, 19, 43)
NUM_NPU = 32
CAPACITY_GB_S = 40.0
CIR_GB_S = CAPACITY_GB_S / NUM_NPU
METHOD = "Frozen build source with only range(8)->range(32); original random queue/hash/exact tail; nominal V/C, decimal GB/s"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_builder():
    # Avoid creating __pycache__ inside the historical archive.
    sys.dont_write_bytecode = True
    path = FROZEN / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("formula_capacity_frozen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.sim.__file__).resolve() == FROZEN / "source/sim.py"
    source = inspect.getsource(module.build)
    assert source.count("range(8)") == 1
    # This leaves all other uses of 8 (the number of layers) unchanged.
    exec(compile(source.replace("range(8)", "range(32)"), str(path), "exec"), module.__dict__)
    return module.build


def screen_one(build, group: str, disks: int, seed: int) -> dict:
    path = FROZEN / "configs" / f"{group}_random_s1_seed{seed}.json"
    config = json.loads(path.read_text())
    assert config["horizon_ms"] == 4500
    assert config["input_counts"] == [1, 12]
    assert config["mode"] == "random"
    config["ssu"] = disks
    requests, metadata, decks = build(config)
    assert len(decks) == NUM_NPU
    assert len(metadata) == len(requests)
    assert all(deck.count("A") > 0 and deck.count("B") == 12 * deck.count("A") for deck in decks)

    # Even a perfectly balanced three-disk placement cannot cover every
    # possible current-request composition: every card has both profiles.
    largest_profile = max(
        config[f"profile_{role}"]["read_gib"] * 2**30
        / (config[f"profile_{role}"]["compute_us"] / 1e6) / 1e9
        for role in ("A", "B")
    )
    assert NUM_NPU * largest_profile > 3 * CAPACITY_GB_S

    worst = np.zeros((NUM_NPU, disks))
    profile_max = {role: 0.0 for role in ("A", "B")}
    above_cir = 0
    peak_request = None
    for rid, load in metadata.items():
        # The frozen builder calls block_ring_hash_disk_id for every block,
        # and uses min(128, remaining_tokens)*1408 bytes for the final block.
        sizes = np.array(load["disk_gib"]) * 2**30
        expected_bytes = load["ssd_prefix_tokens"] * 1408
        assert abs(float(sizes.sum()) - expected_bytes) < 1e-5
        assert all(abs(value - round(value)) < 1e-6 for value in sizes)
        assert math.isclose(load["per_layer_kv_gb"] * 2**30, expected_bytes, abs_tol=1e-5)
        rates = sizes / (load["per_layer_us"] / 1e6) / 1e9
        npu = load["npu_id"]
        worst[npu] = np.maximum(worst[npu], rates)
        maximum = float(rates.max())
        role = load["profile_group"]
        profile_max[role] = max(profile_max[role], maximum)
        above_cir += int(maximum > CIR_GB_S)
        if peak_request is None or maximum > peak_request["max_GB_s"]:
            peak_request = dict(
                rid=rid, npu=npu, role=role, max_GB_s=maximum,
                bytes_by_disk=sizes.astype("int64").tolist(),
                C_ms=load["per_layer_us"] / 1000,
            )

    bounds = worst.sum(axis=0)
    return dict(
        group=group, S=disks, seed=seed,
        requests=len(requests), requests_per_npu=len(decks[0]),
        cycles_per_npu=decks[0].count("A"),
        per_disk_static_upper_GB_s=bounds.tolist(),
        peak_static_upper_GB_s=float(bounds.max()),
        strict_any_current_combination_under40=bool(bounds.max() < CAPACITY_GB_S),
        max_single_npu_disk_GB_s=float(worst.max()),
        npus_with_some_request_disk_above_CIR=int((worst.max(axis=1) > CIR_GB_S).sum()),
        requests_with_some_disk_above_CIR=above_cir,
        per_profile_peak_single_disk_GB_s=profile_max,
        peak_request=peak_request,
    )


def reproduce() -> dict:
    started = time.monotonic()
    source = FROZEN / "run_experiment.py"
    before = sha(source)
    build = load_frozen_builder()
    # Preserve the archived screen's row order: S4/S5, then the S6 extension.
    plan = [(g, s, seed) for g in GROUPS for s in (4, 5) for seed in SEEDS]
    plan += [(g, 6, seed) for g in GROUPS for seed in SEEDS]
    rows = []
    for group, disks, seed in plan:
        rows.append(screen_one(build, group, disks, seed))
        gc.collect()
    assert sha(source) == before
    return dict(method=METHOD, source_sha256=before, rows=rows,
                wall_seconds=time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify", nargs="+", type=Path,
                         help="Recompute and compare existing JSON files without changing them")
    actions.add_argument("--output", type=Path, help="Save a newly computed screen to this path")
    args = parser.parse_args()
    report = reproduce()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}")
    else:
        for path in args.verify or [HERE / "static_capacity_screen.json"]:
            before = sha(path)
            reference = json.loads(path.read_text())
            # Runtime is intentionally excluded: it is not a model result.
            for key in ("method", "source_sha256", "rows"):
                assert report[key] == reference[key], f"Mismatch: {path}: {key}"
            assert sha(path) == before
            print(f"PASS: all 45 rows exactly match {path}; file unchanged")
    for group in GROUPS:
        rows = [row for row in report["rows"] if row["group"] == group]
        peaks = {s: max(row["peak_static_upper_GB_s"] for row in rows if row["S"] == s)
                 for s in (4, 5, 6)}
        minimum = next(s for s in (4, 5, 6) if peaks[s] < CAPACITY_GB_S)
        print(f"{group}: S4={peaks[4]:.8f}, S5={peaks[5]:.8f}, "
              f"S6={peaks[6]:.8f} decimal GB/s; minimum certified S={minimum}")


if __name__ == "__main__":
    main()
