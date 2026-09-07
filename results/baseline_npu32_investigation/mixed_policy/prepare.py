#!/usr/bin/env python3
"""Prepare additional unchanged policies on two already-frozen varied inputs.

Run this preparer before invoking the original sweep against this directory;
the plan intentionally references frozen custom manifests, not build_workload.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil

STUDY = Path(__file__).resolve().parents[1]
LABELS = ("aligned_short072_seed7", "raw_size_varied_rho097_seed7")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    (args.output / "inputs").mkdir(parents=True, exist_ok=True)
    inputs, jobs = {}, []
    for label in LABELS:
        original = STUDY / "mixed_varied/inputs" / (label + ".json.gz")
        with gzip.open(original, "rt") as source:
            frozen = json.load(source)
        spec = {"frozen_input_fingerprint": frozen["input_fingerprint"],
                "source_relative_to_study": str(original.relative_to(STUDY))}
        dest = args.output / "inputs" / original.name
        if dest.exists():
            with gzip.open(dest, "rt") as source:
                assert json.load(source) == frozen, "preserve conflicting frozen input"
        else:
            shutil.copyfile(original, dest)
        (dest.parent / (label + ".spec.json")).write_text(json.dumps(spec, indent=2) + "\n")
        inputs[label] = spec
        for strategy, assignment in [("strategy1", "pipeline"), ("strategy2", "pipeline"),
                                     ("strategy3", "pipeline"), ("strategy3", "fixed")]:
            jobs.append({"input": label, "strategy": strategy,
                         "variant": strategy + "_" + assignment,
                         "config": {"assignment": assignment}})
    (args.output / "plan.json").write_text(json.dumps({"inputs": inputs, "jobs": jobs}, indent=2) + "\n")
    print({"frozen_inputs": len(inputs), "jobs": len(jobs), "output": str(args.output)})


if __name__ == "__main__":
    main()
