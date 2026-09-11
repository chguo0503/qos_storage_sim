"""Pure request-order designs for the 32-NPU, six-SSU experiment.

The result is a permutation of POSITIONS in ``profile_indices``.  The caller
must move each complete original request (including its immutable placement),
not recreate the request from its profile.  No profile, count, or arrival is
changed here.  These designs are hypotheses, not utilization guarantees.

Profiles are indexed by entries of ``profile_indices`` and must provide a
``role`` or ``analysis_role`` of short/long, plus a positive per-layer compute
duration.  Supported duration fields are documented in ``_compute_ms``.
All requests have the same layer count, so cutting by per-layer C is exactly
equivalent to cutting by total pure request compute (8C in this study).

Modes:
* cohort2: two consecutive groups of 16 cards, phases 0 and 1/2.
* cohort4: four consecutive groups of 8 cards, phases 0, 1/4, 1/2, 3/4.
* burst_all: all 32 cards start at the same nominal long-packet phase.

A packet contains ``period_packets`` long requests and a proportional amount
of short pure compute.  Short counts and individual short-profile order are
allowed to vary by packet, without changing the finite input population.
Both role queues are independently shuffled for each NPU.  With only one
profile per role and zero jitter, cohort members can intentionally have the
same role pattern; their original request identities are still permuted.

Phase cuts are snapped to the nearest whole-request boundary.  In particular,
cohort4 with one long request per packet can collapse two nominal phases onto
one actual boundary.  ``describe_order`` exposes that quantization.  No claim
about actual warm-window role coverage is made; the simulator must audit it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping


MODES = ("cohort2", "cohort4", "burst_all")


def _get(profile, field, default=None):
    return profile.get(field, default) if isinstance(profile, Mapping) else getattr(profile, field, default)


def _compute_ms(profile):
    for name, factor in (
        ("layer_compute_ms", 1.0),
        ("per_layer_compute_ms", 1.0),
        ("per_layer_compute_us", 0.001),
        ("per_layer_us", 0.001),
        ("compute_us", 0.001),
    ):
        value = _get(profile, name)
        if value is not None:
            value = float(value) * factor
            if not math.isfinite(value) or value <= 0:
                raise ValueError("profile compute duration must be finite and positive")
            return value
    raise ValueError("profile lacks a supported per-layer compute duration")


def _role(profile):
    value = _get(profile, "analysis_role", _get(profile, "role"))
    value = str(value).lower()
    if value in ("s", "short"):
        return "short"
    if value in ("l", "long"):
        return "long"
    raise ValueError("profiles must explicitly label analysis_role/role as short or long")


def _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if isinstance(period_packets, bool) or int(period_packets) != period_packets or period_packets < 1:
        raise ValueError("period_packets must be a positive integer")
    if int(npu) != npu or not 0 <= npu < 32:
        raise ValueError("npu must be an integer in [0, 31]")
    if not math.isfinite(float(phase_jitter)) or not 0 <= phase_jitter < 0.5:
        raise ValueError("phase_jitter must be in [0, 0.5)")
    period_packets = int(period_packets)
    labels = list(profile_indices)
    weights = [_compute_ms(profiles[key]) for key in labels]
    roles = [_role(profiles[key]) for key in labels]
    queues = {role: [i for i, r in enumerate(roles) if r == role] for role in ("short", "long")}
    if not queues["short"] or not queues["long"]:
        raise ValueError("every NPU input must contain both short and long requests")
    rng = random.Random(int(seed) * 1_000_003 + int(npu) * 100_003 + 92_821)
    rng.shuffle(queues["short"])
    rng.shuffle(queues["long"])
    short_total = math.fsum(weights[i] for i in queues["short"])
    long_total = math.fsum(weights[i] for i in queues["long"])
    short_used = 0.0
    long_used = 0.0
    short_cursor = 0
    packets = []
    long_queue = queues["long"]
    short_queue = queues["short"]
    for start in range(0, len(long_queue), period_packets):
        longs = long_queue[start:start + period_packets]
        long_used += math.fsum(weights[i] for i in longs)
        target = short_total * long_used / long_total
        shorts = []
        last = start + period_packets >= len(long_queue)
        while short_cursor < len(short_queue):
            i = short_queue[short_cursor]
            if not last and abs(short_used + weights[i] - target) >= abs(short_used - target):
                break
            shorts.append(i)
            short_used += weights[i]
            short_cursor += 1
        packets.append(longs + shorts)
    ordered = [i for packet in packets for i in packet]
    assert sorted(ordered) == list(range(len(labels)))
    groups = {"cohort2": 2, "cohort4": 4, "burst_all": 1}[mode]
    group = int(npu) // (32 // groups)
    nominal_fraction = group / groups
    jitter = rng.uniform(-float(phase_jitter), float(phase_jitter)) if phase_jitter else 0.0
    fraction = (nominal_fraction + jitter) % 1.0
    first_packet = packets[0]
    boundaries = [0.0]
    for i in first_packet:
        boundaries.append(boundaries[-1] + weights[i])
    period = boundaries[-1]
    target_phase = fraction * period
    cut = min(range(len(boundaries)), key=lambda k: (abs(boundaries[k] - target_phase), k))
    result = ordered[cut:] + ordered[:cut]
    info = {
        "mode": mode,
        "period_packets": period_packets,
        "group": group,
        "group_count": groups,
        "phase_jitter_fraction": phase_jitter,
        "nominal_phase_fraction": nominal_fraction,
        "requested_phase_fraction": fraction,
        "actual_phase_fraction_of_first_packet": boundaries[cut] / period,
        "phase_boundary_request_count": cut,
        "first_packet_per_layer_compute_ms": period,
        "first_packet_full_compute_ms_for_8_layers": period * 8,
        "actual_full_compute_phase_ms_for_8_layers": boundaries[cut] * 8,
        "max_packet_full_compute_ms_for_8_layers": max(math.fsum(weights[i] for i in x) * 8 for x in packets),
        "short_request_count": len(short_queue),
        "long_request_count": len(long_queue),
        "short_pure_compute_fraction": short_total / (short_total + long_total),
        "same_request_position_multiset": True,
        "phase_is_nominal_pure_compute_not_actual_time": True,
    }
    return result, info


def order_indices(profile_indices, profiles, npu, seed, mode, period_packets=1, phase_jitter=0.0):
    """Return an immutable-input-preserving permutation of original positions."""
    return _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter)[0]


def describe_order(profile_indices, profiles, npu, seed, mode, period_packets=1, phase_jitter=0.0):
    """Return deterministic design metadata, including realized phase snapping."""
    return _design(profile_indices, profiles, npu, seed, mode, period_packets, phase_jitter)[1]
