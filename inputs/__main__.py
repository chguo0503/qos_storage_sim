"""Run synthetic, catalog, or saved-manifest input through simulator.api."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.api import STRATEGIES, run_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("synthetic", "data", "manifest"), default="synthetic")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--strategy", choices=STRATEGIES, default="asu_baseline")
    parser.add_argument("--num-npu", type=int)
    parser.add_argument("--num-ssu", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--requests-per-npu", type=int, default=2)
    parser.add_argument("--keys", default="32:256,128:4096", help="exact data keys: total_K:miss,...")
    parser.add_argument("--seed", type=int, help="override submission/input seed; otherwise manifest seed or 7")
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are preserved")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    metadata = {}
    if args.input == "manifest":
        if args.manifest is None:
            parser.error("--input manifest requires --manifest")
        from inputs.manifest import load_manifest
        requests, metadata = load_manifest(args.manifest)
    elif args.manifest is not None:
        parser.error("--manifest requires --input manifest")
    num_npu = args.num_npu if args.num_npu is not None else metadata.get("num_npu", 32)
    num_ssu = args.num_ssu if args.num_ssu is not None else metadata.get("num_ssu", 3)
    n_layers = args.n_layers if args.n_layers is not None else metadata.get("n_layers", 8)
    seed = args.seed if args.seed is not None else metadata.get("seed", 7)
    if args.input != "manifest":
        kwargs = dict(num_npu=num_npu, num_ssu=num_ssu,
                      requests_per_npu=args.requests_per_npu, seed=seed)
        if args.input == "data":
            from inputs.from_data import build_requests
            try:
                keys = tuple(tuple(map(int, key.split(":"))) for key in args.keys.split(","))
                if any(len(key) != 2 for key in keys):
                    raise ValueError("each key must be total_K:miss")
                requests, metadata = build_requests(**kwargs, keys=keys)
            except ValueError as error:
                parser.error(str(error))
        else:
            from inputs.synthetic import build_requests
            requests = build_requests(**kwargs)
    result = run_simulation(requests, strategy=args.strategy, num_npu=num_npu,
                            num_ssu=num_ssu, n_layers=n_layers, seed=seed)
    result["input_metadata"] = metadata
    result["configuration"] = dict(num_npu=num_npu, num_ssu=num_ssu,
                                   n_layers=n_layers, seed=seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    summary = result["summary"]
    # The reported mean is over the complete finite run, not the warm [2,4)s
    # experimental window. Full native metrics are available in --output.
    utilization = summary["fleet_npu_compute_utilization"]
    print(json.dumps(dict(strategy=args.strategy, requests=len(requests),
                         makespan_ms=summary["makespan_ms"],
                         whole_run_npu_utilization_percent=100 * utilization,
                         invariants_passed=all(summary["invariants"].values()),
                         output=str(args.output) if args.output else None)))


if __name__ == "__main__":
    main()
