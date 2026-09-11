#!/usr/bin/env python3
"""Input-only checks and explicitly hypothetical handoff sensitivity table."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[1]
sys.path[:0] = [str(ROOT), str(HERE)]
from run_baseline_npu32_stress import load_manifest
import mixed_design_v3 as design


def role_sensitivity(descriptions, prefix_end_ms, short_wall_factor):
    first = descriptions[0]
    long_c = first["per_request_pure_compute_ms"][3]
    roles = []
    for d in descriptions:
        if d["cohort"] == "early_12":
            predicted = float(d["early_selection_predicted_join_ms"][str(d["npu"])])
            join = 3 * long_c + (predicted - 3 * long_c) * short_wall_factor
            # The initial 3L segment lies before [2,4)s in this proxy model.
            roles.append((join, 4000.0))
        else:
            roles.append((prefix_end_ms, prefix_end_ms + d["late_main_Long_requests"] * long_c))
    times = {2000.0, 4000.0}
    for start, end in roles:
        if 2000 < start < 4000:
            times.add(start)
        if 2000 < end < 4000:
            times.add(end)
    times = sorted(times)
    max_long = 0
    long_card_ms = 0.0
    over29_ms = 0.0
    for start, end in zip(times, times[1:]):
        mid = (start + end) / 2
        count = sum(a <= mid < b for a, b in roles)
        max_long = max(max_long, count)
        long_card_ms += count * (end - start)
        over29_ms += (end - start) if count > 29 else 0
    warm_roles = []
    for d, (start, end) in zip(descriptions, roles):
        long_wall = max(0.0, min(4000, end) - max(2000, start))
        warm_roles.append(dict(npu=d["npu"], long_role_wall_ms=long_wall,
                               short_role_wall_ms=2000 - long_wall))
    return dict(late_prefix_end_ms=prefix_end_ms,
        early_short_wall_factor_vs_fixed20L=short_wall_factor,
        warm_mean_Long_cards=long_card_ms / 2000.0, warm_max_Long_cards=max_long,
        warm_KLong_above29_ms=over29_ms,
        min_per_card_Long_role_wall_ms=min(x["long_role_wall_ms"] for x in warm_roles),
        min_per_card_Short_role_wall_ms=min(x["short_role_wall_ms"] for x in warm_roles),
        note="Hypothetical wall-role intervals, NOT measured compute and NOT a capacity certificate")


def main():
    sources = []
    scenarios = []
    for seed in (7, 19, 43, 67, 101):
        source = HERE / "inputs" / f"raw176_three_l20_mixed_seed{seed}.json.gz"
        if not source.exists():
            continue
        requests, metadata = load_manifest(source)
        sources.append(dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(), seed=seed))
        for variant in design.VARIANTS:
            descriptions = []
            output = []
            for npu in range(32):
                lane = sorted((r for r in requests if r.npu_id == npu), key=lambda r: r.request_id)
                ordered, desc = design.order_lane(lane, metadata, npu, seed, variant)
                repeated, desc2 = design.order_lane(lane, metadata, npu, seed, variant)
                assert desc == desc2
                assert all(a is b for a, b in zip(ordered, repeated))
                assert Counter(id(r) for r in lane) == Counter(id(r) for r in ordered)
                descriptions.append(desc)
                output.extend(ordered)
            assert Counter(id(r) for r in requests) == Counter(id(r) for r in output)
            assert all(len(d["early_cards"]) == 12 and len(d["late_cards"]) == 20 for d in descriptions)
            assert len({tuple(d["early_cards"]) for d in descriptions}) == 1
            assert len({tuple(d["guard_cards"]) for d in descriptions}) == 1
            first = descriptions[0]
            scenarios.append(dict(seed=seed, variant=variant, requests=len(requests), all_checks_pass=True,
                early_cards=first["early_cards"], late_cards=first["late_cards"], guard_cards=first["guard_cards"],
                early_join_proxy_ms=first["early_selection_predicted_join_ms"],
                prefix_pure_compute_ms=first["late_Short_prefix_pure_compute_ms"],
                prefix_fixed20L_proxy_ms=first["late_Short_prefix_fixed20L_proxy_ms"],
                per_npu=descriptions,
                sensitivity=[role_sensitivity(descriptions, end, factor)
                             for end in (960.0, 1000.0, 1040.0, 1080.0, 1120.0, 1200.0)
                             for factor in (0.9, 0.95, 1.0, 1.05)]))
    assert scenarios
    result = dict(scope="Pure permutations and hypothetical wall-role sensitivity; no simulation",
        design_sha256=hashlib.sha256(Path(design.__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), sources=sources,
        scenarios=len(scenarios), lanes_checked=32 * len(scenarios), all_checks_pass=True, rows=scenarios)
    output = HERE / "mixed_design_v3_checks.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "rows")}))
    for scenario in scenarios[:3]:
        print(scenario["variant"], scenario["early_cards"], scenario["guard_cards"])
        for row in scenario["sensitivity"]:
            if row["early_short_wall_factor_vs_fixed20L"] == 1 and row["late_prefix_end_ms"] in (1000, 1040, 1080):
                print(json.dumps(row))


if __name__ == "__main__":
    main()
