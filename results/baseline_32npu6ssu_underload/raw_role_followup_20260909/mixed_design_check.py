#!/usr/bin/env python3
"""Recheck v1 permutations and tail lower bounds without simulation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from run_baseline_npu32_stress import load_manifest
import mixed_design as design


def main():
    rows = []
    sources = []
    for seed in (7, 19, 43, 67, 101):
        path = HERE / "inputs" / f"raw200_three_l20_mixed_seed{seed}.json.gz"
        if not path.exists():
            continue
        requests, meta = load_manifest(path)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed=seed))
        for variant in design.VARIANTS:
            lanes = []
            for npu in range(32):
                original = sorted((r for r in requests if r.npu_id == npu), key=lambda r: r.request_id)
                ordered, desc = design.order_lane(original, meta, npu, seed, variant)
                repeated, repeated_desc = design.order_lane(original, meta, npu, seed, variant)
                assert desc == repeated_desc and all(a is b for a, b in zip(ordered, repeated))
                assert Counter(id(r) for r in original) == Counter(id(r) for r in ordered)
                core = ordered[:desc["core_requests"]]
                extras = ordered[desc["core_requests"]:]
                core_ids = {r.request_id for r in core}
                assert [r.request_id for r in extras] == [r.request_id for r in original if r.request_id not in core_ids]
                pure = sum(8 * float(r.load["per_layer_us"]) / 1000 for r in core)
                assert abs(pure - desc["core_pure_compute_ms"]) < 1e-8
                first_layer_gib = sum(size for _, size in core[0].placement[0])
                first_link_lower_bound_ms = 1000 * first_layer_gib / 50.0
                final_compute_ms = float(core[-1].load["per_layer_us"]) / 1000.0
                # At t=0 no previous request exists to prefetch first request's
                # first layer; tail L0 starts at the last core-layer C start.
                tail_l0_lower_bound = pure + first_link_lower_bound_ms - final_compute_ms
                lanes.append(dict(npu=npu, core_compute_ms=pure,
                    first_layer_receive_lower_bound_ms=first_link_lower_bound_ms,
                    final_core_layer_compute_ms=final_compute_ms,
                    tail_admission_lower_bound_ms=pure + first_link_lower_bound_ms,
                    tail_L0_prefetch_lower_bound_ms=tail_l0_lower_bound,
                    excludes_tail_L0_before_4000=tail_l0_lower_bound >= 4000,
                    initial_role=desc["initial_role"], cut=desc["cut_request_index"],
                    short_template=desc["short_template_profile_indices"],
                    core_counts=desc["core_counts"], extras_counts=desc["extras_counts"]))
            rows.append(dict(seed=seed, variant=variant, all_32_permutations_pass=True,
                initial_long_count=sum(x["initial_role"] == "long" for x in lanes),
                min_tail_admission_lower_bound_ms=min(x["tail_admission_lower_bound_ms"] for x in lanes),
                min_tail_L0_lower_bound_ms=min(x["tail_L0_prefetch_lower_bound_ms"] for x in lanes),
                all_lanes_exclude_tail_L0_before_4000=all(x["excludes_tail_L0_before_4000"] for x in lanes),
                unique_short_templates=len({tuple(x["short_template"]) for x in lanes}), lanes=lanes))
    assert rows
    result = dict(scope="Pure input permutation and analytic tail bounds; no simulations",
        design_sha256=hashlib.sha256(Path(design.__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=sources,
        source_semantics="metadata n_layers=8; placement length1 repeats the same layer template",
        tail_bound_scope="No initial prefetched I/O; first layer must traverse the 50GiB/s NPU link. Cross-request tail L0 releases at final core layer compute start.",
        scenarios=len(rows), lanes_checked=32 * len(rows), all_checks_pass=True, rows=rows)
    output = HERE / "mixed_design_checks.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "sources")}))
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "lanes"}))


if __name__ == "__main__":
    main()
