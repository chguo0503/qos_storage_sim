"""Deliberately synchronized, exact-profile packet orderings; no simulation.

Every long request is paired with the same integer quota of EVERY short
profile.  A full macroperiod has ``period_packets`` long requests followed by
their exact short quotas, in profile round-robin order.  If the finite long
count is not divisible by period_packets, the final macroperiod contains the
remaining exact units; it is explicitly reported and never padded or dropped.

Within a cohort, the PROFILE template is identical across NPUs.  Randomness
only permutes original request identities within each individual profile.
The returned list permutes positions in profile_indices: callers must retain
each original request's C, V, placement, arrival and NPU assignment.

Phase rotation uses cumulative pure compute, snapped to a whole-request
boundary.  Because all requests have eight layers, per-layer C and full 8C
give identical cuts.  These are input constructions, not performance claims
or guarantees of actual warm-window coverage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
import random


MODES = ("exact_cohort2", "exact_cohort4", "exact_burst")


def _get(p, name, default=None):
    return p.get(name, default) if isinstance(p, Mapping) else getattr(p, name, default)


def _compute(p):
    for name, scale in (("layer_compute_ms", 1), ("per_layer_compute_ms", 1),
                        ("per_layer_compute_us", .001), ("per_layer_us", .001),
                        ("compute_us", .001)):
        value = _get(p, name)
        if value is not None:
            value = float(value) * scale
            if math.isfinite(value) and value > 0:
                return value
            raise ValueError("compute duration must be positive and finite")
    raise ValueError("profile lacks per-layer compute duration")


def _role(p):
    role = str(_get(p, "analysis_role", _get(p, "role"))).lower()
    if role in ("s", "short"):
        return "short"
    if role in ("l", "long"):
        return "long"
    raise ValueError("explicit short/long analysis_role or role is required")


def _hash(template):
    # repr also gives a deterministic encoding if profile keys are tuples.
    return hashlib.sha256(json.dumps([repr(x) for x in template], separators=(",", ":")).encode()).hexdigest()


def _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if phase_jitter != 0:
        raise ValueError("exact synchronized designs require phase_jitter=0")
    if isinstance(period_packets, bool) or int(period_packets) != period_packets or period_packets < 1:
        raise ValueError("period_packets must be a positive integer")
    if int(npu) != npu or not 0 <= npu < 32:
        raise ValueError("NPU must be an integer in [0,31]")
    period_packets = int(period_packets)
    labels = list(profile_indices)
    counts = Counter(labels)
    costs = {key: _compute(profiles[key]) for key in counts}
    roles = {key: _role(profiles[key]) for key in counts}
    longs = [key for key in counts if roles[key] == "long"]
    shorts = sorted((key for key in counts if roles[key] == "short"), key=lambda key: (costs[key], repr(key)))
    if len(longs) != 1 or not shorts:
        raise ValueError("exact packet design requires one long profile and at least one short profile")
    long = longs[0]
    long_count = counts[long]
    if any(counts[key] % long_count for key in shorts):
        raise ValueError("every short-profile count must be divisible by long-profile count")
    quotas = {key: counts[key] // long_count for key in shorts}
    packets = []
    for start in range(0, long_count, period_packets):
        units = min(period_packets, long_count - start)
        packet = [long] * units
        packet += [key for turn in range(max(quotas.values()) * units)
                   for key in shorts if turn < quotas[key] * units]
        packets.append(packet)
    template = [key for packet in packets for key in packet]
    assert Counter(template) == counts
    groups = {"exact_cohort2": 2, "exact_cohort4": 4, "exact_burst": 1}[mode]
    group = int(npu) // (32 // groups)
    fraction = group / groups
    boundaries = [0.0]
    for key in packets[0]:
        boundaries.append(boundaries[-1] + costs[key])
    period = boundaries[-1]
    cut = min(range(len(boundaries)), key=lambda k: (abs(boundaries[k] - fraction * period), k))
    rotated_template = template[cut:] + template[:cut]
    queues = {key: [i for i, k in enumerate(labels) if k == key] for key in counts}
    rng = random.Random(int(seed) * 1_000_003 + int(npu) * 100_003 + 92_821)
    for key in sorted(queues, key=repr):
        rng.shuffle(queues[key])
    cursors = {key: 0 for key in queues}
    result = []
    for key in rotated_template:
        result.append(queues[key][cursors[key]])
        cursors[key] += 1
    assert sorted(result) == list(range(len(labels)))
    info = {
        "mode": mode, "period_packets": period_packets, "group": group,
        "group_count": groups, "phase_jitter_fraction": 0.0,
        "requested_phase_fraction": fraction,
        "actual_phase_fraction_of_first_packet": boundaries[cut] / period,
        "phase_boundary_request_count": cut,
        "actual_full_compute_phase_ms_for_8_layers": boundaries[cut] * 8,
        "first_packet_full_compute_ms_for_8_layers": period * 8,
        "full_macroperiod_count": long_count // period_packets,
        "finite_tail_long_units": long_count % period_packets,
        "finite_tail_has_exact_proportional_profile_quotas": True,
        "long_profile": long, "long_request_count": long_count,
        "short_profile_quotas_per_long": [{"profile": key, "quota": quotas[key]} for key in shorts],
        "short_profile_layout": "round_robin_by_compute_then_profile_key",
        "unrotated_profile_template_sha256": _hash(template),
        "profile_template_sha256": _hash(rotated_template),
        "same_original_request_position_multiset": True,
        "randomness_changes_only_within_profile_identity_order": True,
        "phase_is_nominal_pure_compute_not_actual_time": True,
        "deliberately_synchronized_profile_template": True,
    }
    return result, info


def exact_order_indices(profile_indices, profiles, npu, seed, mode, period_packets=1, phase_jitter=0.0):
    """Return original request-position permutation without changing population."""
    return _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter)[0]


def describe_exact_order(profile_indices, profiles, npu, seed, mode, period_packets=1, phase_jitter=0.0):
    """Report exact quotas, finite tail and realized pure-compute phase."""
    return _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter)[1]
