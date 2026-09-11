"""Raw176 backup single-handoff permutations; no simulation or timing controls.

This module is independent of frozen v1/v2. The Short wall proxy was measured with Long200, not Long176.  All wall-time values
below are planning proxies, not request arrivals, sleeps, or admission gates.
"""
from collections import Counter, defaultdict
import hashlib
import random


VARIANTS = ("handoff176_11l_344", "handoff176_11l_434", "handoff176_guard4_344")
PROFILE_KEYS = ((32, 1024), (48, 1024), (64, 1024), (176, 1024))
SHORT_WALL_PROXY_MS = (95.59614065168324, 113.47015757903526, 137.67019174301566)
PROXY_SOURCE = "runs/raw200_three_l20_fixed_seed7/baseline/raw200_three_l20_fixed_seed7_754f92e5ffba_baseline_104671470b.json.gz"
PROXY_SOURCE_SHA256 = "70e6b50e23beeffe0e36d64e0676e5e304d3a923c3bc57a9df6289515b3b0f8b"
EARLY_HANDOFF_TARGET_MS = 3700.0


def _round_robin(counts):
    counts = list(counts)
    result = []
    while any(counts):
        for profile in range(len(counts)):
            if counts[profile]:
                result.append(profile)
                counts[profile] -= 1
    return result


def _rng(seed, npu, profile):
    encoded = f"single_handoff_176_v3:{int(seed)}:{int(npu)}:{profile}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(encoded).digest(), "big"))


def order_lane(lane, metadata, npu, seed, variant):
    """Return the same lane's original request objects and explicit description.

    Select 12 early-Long cards using full source per-lane quotas, then assign
    the remaining 20 cards a common Short prefix and a 10/11-Long middle segment.
    Selection depends on quotas, not simulated results from this new ordering.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected {VARIANTS}")
    if int(metadata.get("n_layers", 0)) != 8 or int(metadata.get("num_npu", 0)) != 32:
        raise ValueError("this design requires 32 NPUs, eight layers")
    lane = list(lane)
    if not lane or not 0 <= int(npu) < 32:
        raise ValueError("expected one nonempty lane of NPU 0..31")
    mapping = {key: p for p, key in enumerate(PROFILE_KEYS)}
    buckets = defaultdict(list)
    for request in lane:
        key = int(request.load["seq_len_k"]), int(request.load["nql"])
        if key not in mapping or request.npu_id != int(npu):
            raise ValueError("unexpected profile or assigned NPU")
        if request.arrival_time_ms != 0.0 or len(request.placement) not in (1, 8):
            raise ValueError("requires original arrival0 and reusable/eight-layer placement")
        buckets[mapping[key]].append(request)
    if set(buckets) != set(range(4)):
        raise ValueError("every lane must retain all four source profiles")
    counts = [len(buckets[p]) for p in range(4)]
    computes = []
    for p in range(4):
        values = [8 * float(r.load["per_layer_us"]) / 1000 for r in buckets[p]]
        if max(values) - min(values) > 1e-9:
            raise ValueError("unequal compute within a profile")
        computes.append(values[0])
    source_counts = {int(row["npu_id"]): list(row["profile_counts"])
                     for row in metadata["per_npu_assignment"]}
    if set(source_counts) != set(range(32)) or source_counts[int(npu)] != counts:
        raise ValueError("source per-NPU quota metadata mismatch")
    declared_keys = metadata["profile_keys"]
    if isinstance(declared_keys, str):
        declared_keys = [tuple(map(int, item.split(":"))) for item in declared_keys.split(",")]
    if list(map(tuple, declared_keys)) != list(PROFILE_KEYS):
        raise ValueError("source profile-index order mismatch")
    if any(min(c[:3]) < 5 or c[3] < 11 for c in source_counts.values()):
        raise ValueError("insufficient source population for common handoff segments")

    # Exclude cards with exceptionally little/much Short work by choosing the
    # twelve nearest the desired crossing under a fixed-20L reference proxy.
    early_short_wall = {n: sum(c[p] * SHORT_WALL_PROXY_MS[p] for p in range(3))
                        for n, c in source_counts.items()}
    early_join_proxy = {n: 3 * computes[3] + w for n, w in early_short_wall.items()}
    early_cards = sorted(sorted(range(32), key=lambda n: (
        abs(early_join_proxy[n] - EARLY_HANDOFF_TARGET_MS), n))[:12])
    late_cards = [n for n in range(32) if n not in early_cards]
    # Spread four guard cards across the rank order, with no topology remapping.
    guard_cards = [late_cards[j] for j in (0, 5, 10, 15)] if "guard4" in variant else []
    prefix_counts = [4, 3, 4] if variant.endswith("434") else [3, 4, 4]
    short_prefix = _round_robin(prefix_counts)
    if int(npu) in early_cards:
        segments = [[3] * 3, _round_robin(counts[:3]), [3] * (counts[3] - 3)]
        segment_names = ["early_Long", "middle_all_Short", "remaining_Long"]
        cohort = "early_12"
        intended_main_long_count = None
    else:
        intended_main_long_count = 10 if int(npu) in guard_cards else 11
        remaining_short = [counts[p] - prefix_counts[p] for p in range(3)]
        if min(remaining_short) < 0:
            raise ValueError("insufficient source Short quota")
        segments = [short_prefix, [3] * intended_main_long_count,
                    _round_robin(remaining_short), [3] * (counts[3] - intended_main_long_count)]
        segment_names = ["early_Short_prefix", "middle_Long", "remaining_Short", "remaining_Long"]
        cohort = "late_guard_4" if int(npu) in guard_cards else "late_main"
    sequence = [p for segment in segments for p in segment]
    if Counter(sequence) != Counter({p: counts[p] for p in range(4)}):
        raise AssertionError("profile population changed")
    pools = {}
    for p in range(4):
        pool = list(buckets[p])
        _rng(seed, npu, p).shuffle(pool)
        pools[p] = iter(pool)
    ordered = [next(pools[p]) for p in sequence]
    if Counter(id(r) for r in ordered) != Counter(id(r) for r in lane):
        raise AssertionError("original request identities changed")
    segment_desc = []
    for name, segment in zip(segment_names, segments):
        quota = Counter(segment)
        segment_desc.append(dict(name=name, count=len(segment), profile_counts=[quota[p] for p in range(4)],
            pure_compute_ms=sum(computes[p] for p in segment)))
    prefix_pure = sum(prefix_counts[p] * computes[p] for p in range(3))
    proxy_prefix = sum(prefix_counts[p] * SHORT_WALL_PROXY_MS[p] for p in range(3))
    desc = dict(design_version="single_handoff_176_v3", variant=variant, npu=int(npu), seed=int(seed),
        cohort=cohort, early_cards=early_cards, late_cards=late_cards, guard_cards=guard_cards,
        source_counts=counts, profile_keys=[list(k) for k in PROFILE_KEYS],
        per_request_pure_compute_ms=computes, profile_categories=[buckets[p][0].load["category"] for p in range(4)],
        segments=segment_desc, profile_sequence=sequence,
        source_request_ids=[r.request_id for r in ordered],
        profile_sequence_sha256=hashlib.sha256(bytes(sequence)).hexdigest(),
        early_selection_rule="12 cards nearest 3*C_Long+sum(all Short counts*fixed-20L wall proxy)=3700ms, ties by NPU; Short proxy measured under Long200",
        early_selection_predicted_join_ms={str(n): early_join_proxy[n] for n in early_cards},
        early_selection_ignores_initial_IO_and_feedback=True,
        early_prefix_Long_pure_compute_ms=3 * computes[3],
        late_Short_prefix_counts=prefix_counts, late_Short_prefix_pure_compute_ms=prefix_pure,
        late_Short_prefix_fixed20L_proxy_ms=proxy_prefix,
        late_Short_prefix_minus_early_Long_pure_compute_ms=prefix_pure - 3 * computes[3],
        late_main_Long_requests=intended_main_long_count,
        late_11L_pure_compute_ms=11 * computes[3], late_10L_pure_compute_ms=10 * computes[3],
        late_prefix_latest_end_for_11L_then_100ms_compute=3900.0 - 11 * computes[3],
        proxy_source=PROXY_SOURCE, proxy_source_sha256=PROXY_SOURCE_SHA256,
        proxy_is_cross_Long_profile=True, proxy_Long_key=[200, 1024], target_Long_key=[176, 1024],
        short_wall_proxy_ms=list(SHORT_WALL_PROXY_MS),
        limitations=("No absolute-time guarantees or delays. Initial Short prefixes run against 12 Long cards, "
                     "so a fixed-20-Long200 processing proxy is not their predicted completion time under Long176. The 28-Long "
                     "guard is conditional on all four 10L segments ending before early cards rejoin, and "
                     "on no guard re-entry during that handoff. Full-run capacity, every-card >=100ms "
                     "warm compute per role, activity, and utilization require actual event validation."))
    return ordered, desc
