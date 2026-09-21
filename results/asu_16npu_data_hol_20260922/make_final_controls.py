#!/usr/bin/env python3
"""Freeze input-order controls without changing the simulator submission seed."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "configs/native_aligned_A10B45_60s.json")
    parser.add_argument("--prefix", default="native_aligned_pool_random_60s")
    parser.add_argument("--audit-file", default="final_control_construction.json")
    args = parser.parse_args()
    source = args.source.resolve()
    original = json.loads(source.read_text())
    records = []
    for input_seed in (7, 17, 27):
        config = json.loads(json.dumps(original))
        config["name"] = f"{args.prefix}_inputseed{input_seed}"
        # seed belongs to the simulator's simultaneous-submit ordering.
        # Keep it fixed at the constructed case's value for an order-only control.
        for npu, sequence in enumerate(config["per_npu_sequences"]):
            random.Random(input_seed * 1000003 + npu).shuffle(sequence)
            assert Counter(sequence) == Counter(original["per_npu_sequences"][npu])
        destination = HERE / "configs" / f"{config['name']}.json"
        destination.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        records.append(dict(config=str(destination.relative_to(HERE)), input_shuffle_seed=input_seed,
                            simulator_submit_seed=config["seed"],
                            config_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                            same_per_npu_profile_multiset=True))
        print(destination)
    evidence = dict(source=str(source.relative_to(HERE)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    shuffle="random.Random(input_seed * 1000003 + npu_id).shuffle(sequence)",
                    controls=records)
    (HERE / args.audit_file).write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
