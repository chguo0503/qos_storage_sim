#!/usr/bin/env python3
"""Probe two declared pure-compute phases of the frozen exact-packet design.

Only original request positions are permuted.  Actual compute/stall timing is
not consulted.  Group phases snap to the nearest whole-request boundary in
the first macroperiod, with ties resolved toward the earlier boundary.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import run_orders
from exact_order_design import (
    _compute, _hash, describe_exact_order, exact_order_indices,
)


HERE = Path(__file__).resolve().parent
MODES = ("phase_cohort3", "phase_cohort4_bias05")
ORIGINAL_SAVE = run_orders.save_manifest
SOURCE_NAMES = (
    "run_orders.py", "order_design.py", "exact_order_design.py",
    "run_phase_orders.py",
)


def _design(profile_indices, profiles, npu, seed, mode,
            period_packets=1, phase_jitter=0.0):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if phase_jitter != 0:
        raise ValueError("declared phase probes require phase_jitter=0")
    labels = list(profile_indices)
    base_kwargs = dict(npu=npu, seed=seed, mode="exact_burst",
                       period_packets=period_packets, phase_jitter=0.0)
    base = exact_order_indices(labels, profiles, **base_kwargs)
    info = describe_exact_order(labels, profiles, **base_kwargs)
    if mode == "phase_cohort3":
        group_sizes = [11, 11, 10]
        group = 0 if npu < 11 else (1 if npu < 22 else 2)
        bias = 0.0
    else:
        group_sizes = [8, 8, 8, 8]
        group = int(npu) // 8
        bias = 0.05
    fraction = (group / len(group_sizes) + bias) % 1.0
    first_units = min(int(period_packets), info["long_request_count"])
    packet_count = first_units * (
        1 + sum(row["quota"] for row in info["short_profile_quotas_per_long"])
    )
    boundaries = [0.0]
    for index in base[:packet_count]:
        boundaries.append(boundaries[-1] + _compute(profiles[labels[index]]))
    period = boundaries[-1]
    cut = min(range(len(boundaries)),
              key=lambda k: (abs(boundaries[k] - fraction * period), k))
    result = base[cut:] + base[:cut]
    assert sorted(result) == list(range(len(labels)))
    realized = boundaries[cut] / period
    info.update(
        mode=mode,
        base_constructor_mode="exact_burst",
        group=group,
        group_count=len(group_sizes),
        group_sizes=group_sizes,
        group_assignment="contiguous NPU identifiers with listed group sizes",
        phase_bias_fraction=bias,
        requested_phase_fraction=fraction,
        actual_phase_fraction_of_first_packet=realized,
        phase_snap_error_fraction=realized - fraction,
        phase_boundary_request_count=cut,
        first_packet_request_count=packet_count,
        actual_full_compute_phase_ms_for_8_layers=boundaries[cut] * 8,
        first_packet_full_compute_ms_for_8_layers=period * 8,
        phase_boundary_rule="nearest cumulative pure-compute whole-request boundary; ties choose earlier boundary",
        unrotated_profile_template_sha256=_hash([labels[i] for i in base]),
        profile_template_sha256=_hash([labels[i] for i in result]),
        phase_constructor_sha256=run_orders.sha(HERE / "run_phase_orders.py"),
        phase_is_nominal_pure_compute_not_actual_time=True,
    )
    return result, info


def phase_order_indices(profile_indices, profiles, npu, seed, mode,
                        period_packets=1, phase_jitter=0.0):
    """Return a permutation of original request positions, with exact quotas."""
    return _design(profile_indices, profiles, npu, seed, mode,
                   period_packets, phase_jitter)[0]


def describe_phase_order(profile_indices, profiles, npu, seed, mode,
                         period_packets=1, phase_jitter=0.0):
    """Report the realized phase, exact template hashes and source hash."""
    return _design(profile_indices, profiles, npu, seed, mode,
                   period_packets, phase_jitter)[1]


def save_phase(path, requests, metadata):
    metadata.update(
        order_source_sha256={name: run_orders.sha(HERE / name)
                             for name in SOURCE_NAMES},
        order_constructor_override="run_phase_orders installs exact-packet phase permutations process-locally; simulator is unchanged",
        shuffle_rule="Original identities are shuffled within each profile by exact_burst, then the exact-quota full sequence is rotated at the nearest whole-request pure-compute phase boundary. See per_npu_order_design.",
        sampling_caveat="Declared synchronized phase probe with identical profile templates within each group. Same per-NPU population, arrivals, C, V and placement as the random reference; phase is nominal pure compute, not guaranteed actual execution timing or performance.",
    )
    return ORIGINAL_SAVE(path, requests, metadata)


def main():
    for name in ("run_phase_orders.py", "exact_order_design.py"):
        target = HERE / "sources" / "phase_order_intervention" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert run_orders.sha(target) == run_orders.sha(HERE / name)
        else:
            shutil.copyfile(HERE / name, target)
    run_orders.order_indices = phase_order_indices
    run_orders.describe_order = describe_phase_order
    run_orders.save_manifest = save_phase
    run_orders.MODES = MODES
    if not any(arg == "--source-label" or arg.startswith("--source-label=")
               for arg in sys.argv[1:]):
        sys.argv.extend(["--source-label", "synthetic_s60_l1_seed7"])
    run_orders.main()


if __name__ == "__main__":
    main()
