"""Frozen v1 pure queue permutations for the raw200 three-short mixed input.

No request is constructed or mutated here.  Predicted wall times choose queue
boundaries only; they are NOT sleeps, release times, or scheduling constraints.
"""
from collections import Counter, defaultdict
import hashlib
import math
import random


VARIANTS = ("5l8s422", "5l7s133", "5l7s133_shuffled", "2l3s111")
PROFILE_KEYS = ((32, 1024), (48, 1024), (64, 1024), (200, 1024))
# Observed completion-admission means of Short requests entirely inside [2,4)s
# in the pre-existing raw200_three_l20_fixed_seed7 Baseline.  These constants
# only guide ordering.  Reordering changes contention and thus realized times.
SHORT_WALL_PROXY_MS = (95.59614065168324, 113.47015757903526, 137.67019174301566)
PROXY_SOURCE = (
    "runs/raw200_three_l20_fixed_seed7/baseline/"
    "raw200_three_l20_fixed_seed7_754f92e5ffba_baseline_104671470b.json.gz"
)
PROXY_SOURCE_SHA256 = "70e6b50e23beeffe0e36d64e0676e5e304d3a923c3bc57a9df6289515b3b0f8b"


def _rng(seed, npu, purpose):
    payload = f"mixed_rotation_v1:{int(seed)}:{int(npu)}:{purpose}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def _key(request):
    return int(request.load["seq_len_k"]), int(request.load["nql"])


def _compute_ms(request):
    # The source stores one reusable placement template for all eight layers.
    return 8 * float(request.load["per_layer_us"]) / 1000.0


def order_lane(lane, metadata, npu, seed, variant):
    """Return (same request objects in new order, complete ordering description).

    Input must be one raw200 mixed lane, with at least 10 Long and the stated
    Short quotas.  Eight contiguous groups of four cards select phases.  The
    repeated, exact-quota core alone has >4000ms pure compute; unused requests
    retain their input relative order and are appended after the rotated core.
    """
    if int(metadata.get("n_layers", 0)) != 8:
        raise ValueError("this frozen design requires metadata n_layers=8")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose {VARIANTS}")
    lane = list(lane)
    if not lane or not 0 <= int(npu) < 32:
        raise ValueError("expected a nonempty lane and an NPU in [0,32)")
    if any(r.npu_id != int(npu) for r in lane):
        raise ValueError("lane contains requests bound to another NPU")
    if len({r.request_id for r in lane}) != len(lane):
        raise ValueError("duplicate request identity in lane")
    if any(r.arrival_time_ms != 0.0 or len(r.placement) not in (1, 8) for r in lane):
        raise ValueError("this frozen design requires arrival=0 and 1 reusable or 8 layer placements")
    key_to_profile = {key: p for p, key in enumerate(PROFILE_KEYS)}
    buckets = defaultdict(list)
    for request in lane:
        if _key(request) not in key_to_profile:
            raise ValueError(f"unexpected profile {_key(request)}")
        buckets[key_to_profile[_key(request)]].append(request)
    if set(buckets) != set(range(4)):
        raise ValueError("all three Short profiles and the Long profile are required")
    compute = []
    for p in range(4):
        values = [_compute_ms(r) for r in buckets[p]]
        if max(values) - min(values) > 1e-9:
            raise ValueError("same profile has unequal compute")
        compute.append(values[0])

    if variant == "5l8s422":
        long_count, repeats = 5, 2
        short_template = [2, 2, 0, 1, 0, 0, 1, 0]
    elif variant in ("5l7s133", "5l7s133_shuffled"):
        long_count, repeats = 5, 2
        short_template = [2, 2, 0, 1, 1, 1, 2]
        if variant.endswith("_shuffled"):
            _rng(seed, npu, "short_profile_template").shuffle(short_template)
    else:
        long_count, repeats = 2, 5
        short_template = [0, 1, 2]
    packet = [3] * long_count + short_template
    packet_counts = Counter(packet)
    core_counts = {p: repeats * packet_counts[p] for p in range(4)}
    if any(len(buckets[p]) < core_counts[p] for p in range(4)):
        raise ValueError(f"insufficient exact per-lane quota; need {core_counts}")
    core_compute = sum(core_counts[p] * compute[p] for p in range(4))
    if core_compute <= 4000.0:
        raise ValueError(f"core pure compute {core_compute}ms does not protect [2,4)s")

    # A packet is unchanged within a lane on subsequent repetitions.  Only
    # request identity selection is randomized; no placement is reconstructed.
    selected = {}
    for p in range(4):
        pool = list(buckets[p])
        _rng(seed, npu, f"profile_identity_{p}").shuffle(pool)
        selected[p] = pool[:core_counts[p]]
    positions = Counter()
    core = []
    for p in packet * repeats:
        core.append(selected[p][positions[p]])
        positions[p] += 1

    wall_proxy = list(SHORT_WALL_PROXY_MS) + [compute[3]]
    wall_prefix = [0.0]
    pure_prefix = [0.0]
    for p in packet:
        wall_prefix.append(wall_prefix[-1] + wall_proxy[p])
        pure_prefix.append(pure_prefix[-1] + compute[p])
    group = int(npu) // 4
    if long_count == 5:
        target_phase = group * compute[3]
        if group < 5:
            cut = group
            cut_rule = "successive exact Long request boundaries"
        elif variant == "5l7s133_shuffled":
            # Keep the initial 20L/12S split while adapting a Short boundary to
            # the shuffled lane's predicted progress through its Short segment.
            cut = min(range(5, len(packet)), key=lambda j: (abs(wall_prefix[j] - target_phase), j))
            cut_rule = "nearest Short request boundary to group*T_Long, proxy wall time"
        else:
            cut = (0, 1, 2, 3, 4, 5, 7, 10)[group]
            cut_rule = "predeclared cuts [0,1,2,3,4,5,7,10]"
    else:
        target_phase = group * wall_prefix[-1] / 8.0
        boundary = min(range(len(packet) + 1), key=lambda j: (abs(wall_prefix[j] - target_phase), j))
        cut = boundary % len(packet)
        cut_rule = "nearest packet boundary to group*proxy_cycle/8; endpoint wraps to zero"

    rotated = core[cut:] + core[:cut]
    used_ids = {r.request_id for r in core}
    extras = [r for r in lane if r.request_id not in used_ids]
    ordered = rotated + extras
    assert len(ordered) == len(lane)
    assert {id(r) for r in ordered} == {id(r) for r in lane}
    assert Counter(_key(r) for r in ordered) == Counter(_key(r) for r in lane)
    assert math.isclose(sum(_compute_ms(r) for r in rotated), core_compute, abs_tol=1e-8)
    sequence = [key_to_profile[_key(r)] for r in ordered]
    desc = dict(
        variant=variant, design_version="mixed_rotation_v1", seed=int(seed), group=group,
        group_npus=[group * 4, group * 4 + 3], profile_keys=[list(x) for x in PROFILE_KEYS],
        profile_categories=[str(buckets[p][0].load.get("category")) for p in range(4)],
        per_request_pure_compute_ms=compute, input_counts=[len(buckets[p]) for p in range(4)],
        packet_profile_indices=packet, short_template_profile_indices=short_template,
        packet_counts=[packet_counts[p] for p in range(4)], core_repetitions=repeats,
        core_counts=[core_counts[p] for p in range(4)], core_requests=len(core),
        core_pure_compute_ms=core_compute, core_pure_compute_exceeds_4000_ms=True,
        extras_counts=[len(buckets[p]) - core_counts[p] for p in range(4)],
        extras_rule="Append unused original objects in their original lane relative order; never rotate extras",
        extras_cannot_begin_before_ms=core_compute,
        packet_pure_compute_ms=pure_prefix[-1], packet_proxy_wall_ms=wall_prefix[-1],
        packet_short_pure_compute_ms=sum(compute[p] for p in short_template),
        packet_short_proxy_wall_ms=sum(wall_proxy[p] for p in short_template),
        proxy_long_active_count=32.0 * long_count * compute[3] / wall_prefix[-1],
        cut_request_index=cut, cut_rule=cut_rule,
        target_phase_proxy_ms=target_phase, implemented_phase_proxy_ms=wall_prefix[cut],
        implemented_phase_pure_compute_ms=pure_prefix[cut],
        phase_proxy_error_ms=wall_prefix[cut] - target_phase,
        initial_profile_index=sequence[0], initial_role="long" if sequence[0] == 3 else "short",
        output_profile_sequence=sequence,
        output_source_request_ids=[r.request_id for r in ordered],
        profile_sequence_sha256=hashlib.sha256(bytes(sequence)).hexdigest(),
        short_wall_proxy_ms=list(SHORT_WALL_PROXY_MS),
        proxy_source=PROXY_SOURCE, proxy_source_sha256=PROXY_SOURCE_SHA256,
        proxy_definition="Per-profile completion-admission mean, fixed seed7 Baseline requests entirely in [2000,4000)ms",
        limitations=("Pure queue-order hypothesis, no wall-clock control. Target role count, phase coherence, "
                     "per-card warm mixing, full-window activity, nominal per-disk capacity and utilization "
                     "must all be checked from each strategy's actual events. A >4000ms core only excludes "
                     "extras from the window; it does not prove mixing or feasible instantaneous demand."),
    )
    return ordered, desc
