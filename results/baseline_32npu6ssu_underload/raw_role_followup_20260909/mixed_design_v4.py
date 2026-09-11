"""New binding plus paired queue orders; original raw176 objects are retained.

build_queues returns {new_npu: [original request objects]} and a description.
The caller must copy each original request with its NEW NPU/position ID and
record the source request ID/NPU.  No request or physical placement is mutated
by this module.  Assignment is exactly independent of random/ordered mode.
"""
from collections import Counter, defaultdict
import hashlib
import random


MODES = ("random", "ordered")
PROFILE_KEYS = ((32, 1024), (48, 1024), (64, 1024), (176, 1024))


def _rng(seed, purpose):
    text = f"raw176_rebinding_v4:{int(seed)}:{purpose}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(text).digest(), "big"))


def _round_robin(counts):
    counts = list(counts)
    result = []
    while any(counts):
        for profile in range(len(counts)):
            if counts[profile]:
                result.append(profile)
                counts[profile] -= 1
    return result


def _quota(npu):
    if npu < 17:
        return [3, 3, 3, 15]
    if npu < 20:
        return [4, 4, 4, 13]
    position = npu - 20
    short_count = 17 if position < 9 else 16
    return [short_count] * 3 + [11 if position < 6 else 10]


def build_queues(source_requests, source_metadata, seed, mode):
    """Return new bindings/orders of the unchanged global source population.

    mode=random uniformly shuffles each entire newly assigned lane.  Ordered
    mode starts all front20 lanes with their whole Long population; the last12
    lanes use shuffled 9*S2+8*S3, then16*S1, then all Long, then remaining Short.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {MODES}")
    if source_metadata.get("n_layers") != 8 or source_metadata.get("num_npu") != 32:
        raise ValueError("expected 32 NPUs and eight-layer requests")
    if source_metadata.get("num_ssu") != 6:
        raise ValueError("expected six SSUs")
    source = list(source_requests)
    if len(source) != 1212 or len({r.request_id for r in source}) != 1212:
        raise ValueError("expected exactly 1212 unique source requests")
    key_to_profile = {key: p for p, key in enumerate(PROFILE_KEYS)}
    pools = defaultdict(list)
    for request in source:
        key = int(request.load["seq_len_k"]), int(request.load["nql"])
        if key not in key_to_profile or request.arrival_time_ms != 0.0:
            raise ValueError("unexpected raw profile or nonzero source arrival")
        if len(request.placement) not in (1, 8):
            raise ValueError("expected one reused or eight explicit layer placements")
        pools[key_to_profile[key]].append(request)
    if [len(pools[p]) for p in range(4)] != [264, 264, 264, 420]:
        raise ValueError("source global population does not match v4 design")
    computes = []
    for profile in range(4):
        values = [8 * float(r.load["per_layer_us"]) / 1000 for r in pools[profile]]
        if max(values) - min(values) > 1e-9:
            raise ValueError("same profile has inconsistent C")
        computes.append(values[0])
    quotas = {npu: _quota(npu) for npu in range(32)}
    assert [sum(q[p] for q in quotas.values()) for p in range(4)] == [264, 264, 264, 420]

    # Different RNG purpose strings isolate binding from all ordering choices.
    assigned = {npu: [] for npu in range(32)}
    for profile in range(4):
        pool = sorted(pools[profile], key=lambda r: r.request_id)
        _rng(seed, f"assignment_profile_{profile}").shuffle(pool)
        cursor = 0
        remaining = {npu: quotas[npu][profile] for npu in range(32)}
        while cursor < len(pool):
            for npu in range(32):
                if remaining[npu]:
                    assigned[npu].append(pool[cursor])
                    remaining[npu] -= 1
                    cursor += 1
        assert not any(remaining.values())

    result = {}
    assignment_rows = []
    ordering_rows = []
    for npu in range(32):
        assigned_lane = assigned[npu]
        if mode == "random":
            ordered = list(assigned_lane)
            _rng(seed, f"random_whole_lane_{npu}").shuffle(ordered)
            template = [key_to_profile[(int(r.load["seq_len_k"]), int(r.load["nql"]))] for r in ordered]
            segment_description = "one independent shuffle of the complete assigned lane"
        else:
            quota = quotas[npu]
            if npu < 20:
                template = [3] * quota[3] + _round_robin(quota[:3])
                segment_description = "all Long then remaining Short in profile round robin"
            else:
                prefix = [1] * 9 + [2] * 8
                _rng(seed, f"ordered_short_prefix_{npu}").shuffle(prefix)
                tail_counts = [quota[0] - 16, quota[1] - 9, quota[2] - 8]
                template = prefix + [0] * 16 + [3] * quota[3] + _round_robin(tail_counts)
                segment_description = "independent shuffle of9*S2+8*S3; then16*S1; then all Long; then remaining Short"
            by_profile = defaultdict(list)
            for request in assigned_lane:
                key = int(request.load["seq_len_k"]), int(request.load["nql"])
                by_profile[key_to_profile[key]].append(request)
            positions = Counter()
            ordered = []
            for profile in template:
                ordered.append(by_profile[profile][positions[profile]])
                positions[profile] += 1
        assert Counter(template) == Counter({p: quotas[npu][p] for p in range(4)})
        assert Counter(id(r) for r in assigned_lane) == Counter(id(r) for r in ordered)
        result[npu] = ordered
        pure = sum(quotas[npu][p] * computes[p] for p in range(4))
        assignment_rows.append(dict(npu_id=npu, profile_counts=quotas[npu], pure_compute_ms=pure,
            request_count=len(ordered), group="main17" if npu < 17 else "guard3" if npu < 20 else "initial_short12",
            source_request_ids=sorted(r.request_id for r in assigned_lane)))
        ordering_rows.append(dict(npu_id=npu, mode=mode, rule=segment_description,
            profile_sequence=template, source_request_ids=[r.request_id for r in ordered],
            profile_sequence_sha256=hashlib.sha256(bytes(template)).hexdigest()))
    assert Counter(id(r) for lane in result.values() for r in lane) == Counter(id(r) for r in source)

    description = dict(design_version="raw176_new_binding_v4", mode=mode, seed=int(seed),
        requests=1212, profile_keys=[list(x) for x in PROFILE_KEYS], global_profile_counts=[264, 264, 264, 420],
        categories=[str(pools[p][0].load["category"]) for p in range(4)],
        per_request_pure_compute_ms=computes, per_npu_assignment=assignment_rows,
        per_npu_order=ordering_rows, main_cards=list(range(17)), guard_cards=[17, 18, 19], initial_short_cards=list(range(20, 32)),
        assignment_seed_rule="SHA256-derived RNG per profile, independent of mode; globally shuffled identities allocated round robin subject to declared quota",
        population_rule="Same1212 global original requests. Per-card population is changed once by binding, then identical for random and ordered.",
        placement_rule="Preserve every source request's physical SSU/block placement; DO NOT recompute placement from new NPU",
        source_mapping_required="Caller records old.request_id and old.npu_id before copying with new position request_id and new_npu",
        ordered_first20_initial_Long_pure_compute_ms=[quotas[n][3] * computes[3] for n in range(20)],
        ordered_short12_first_mixed_prefix_counts=[0, 9, 8, 0],
        ordered_short12_first_mixed_prefix_pure_compute_ms=9 * computes[1] + 8 * computes[2],
        ordered_short12_following_S1_count=16,
        ordered_short12_total_prefix_pure_compute_ms=16 * computes[0] + 9 * computes[1] + 8 * computes[2],
        min_per_card_total_pure_compute_ms=min(row["pure_compute_ms"] for row in assignment_rows),
        metadata_fields_to_recompute=["input_fingerprint", "logical_input_fingerprint", "input_demand", "active_profile_rate_certificate", "load_within_disk_and_link_capacity", "case_id", "label"],
        metadata_fields_to_replace={"assignment_mode": "new_binding_mixed_v4", "order": mode,
            "long_cards": None, "short_cards": None, "per_npu_assignment": assignment_rows,
            "placement_rule": "Preserved original per-request physical placement after explicit NPU reassignment",
            "population_rule": "Same global1212 original raw176 requests; new per-card quotas paired exactly between random and ordered",
            "layout": "preserved_source_placement_after_rebinding"},
        limitations=("This changes per-card population; the required ordering control is this new binding's random mode. "
                     "No guarantee that random or ordered has both roles on every card in the warm window. "
                     "Role counts, >=100ms per-role warm compute, all-active, full-run per-disk capacity, "
                     "synchronous Long layer waves and device U must be measured without screening failed seeds."))
    return result, description
