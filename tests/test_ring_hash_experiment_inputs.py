"""Ring placement preserves physical identities in retained experiment inputs.

The retired AB experiment is available in Git history. Its shuffle and placement
invariants are exercised here through the maintained diverse input builder.
"""
from collections import Counter, defaultdict
import importlib.util
import math
from pathlib import Path

import pytest

from simulator.core import sim

ROOT = Path(__file__).resolve().parent.parent


def _load_builder(relative_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diverse_inputs():
    builder = _load_builder(
        "results/diverse_data_ssu3_l3_20260916/construct_manifest.py",
        "ring_test_diverse_builder",
    )
    # One full weighted catalog per NPU covers all 24 profiles without a run.
    return [builder.build_workload("semi", seed, horizon_ms=1.0)
            for seed in (7, 19)]


def _physical_requests(requests):
    result = {}
    for request in requests:
        physical_id = request.load["original_request_id"]
        assert physical_id not in result
        # Only queue position and its admission-order ID may change.
        load = {key: value for key, value in request.load.items()
                if key not in ("request_id", "generation")}
        result[physical_id] = (request.npu_id, request.arrival_time_ms,
                               load, request.placement)
    return result


def _check_population(requests, metadata):
    per_npu = defaultdict(Counter)
    profiles = metadata["profiles"]
    for request in requests:
        profile = profiles[request.load["profile_index"]]
        assert request.load["per_layer_us"] == profile["per_layer_compute_us"]
        assert request.load["per_layer_kv_gb"] == profile["per_layer_kv_gib"]
        assert request.load["total_tokens"] == profile["total_tokens"]
        assert request.load["nql"] == profile["nql"]
        assert request.arrival_time_ms == 0.0
        assert math.fsum(size for _, size in request.placement[0]) == pytest.approx(
            profile["per_layer_kv_gib"], abs=1e-12)
        per_npu[request.npu_id][profile["role"]] += 1
    assert len(requests) == metadata["request_count"]
    assert set(per_npu) == set(range(32))
    return per_npu


def _check_hash_and_disk_metadata(requests, metadata):
    assert metadata["layout"] == sim.PLACEMENT_BLOCK_RING_HASH
    assert metadata["placement_identity_key"] == "original_request_id"
    assert "ring_hash" in metadata["label"]
    num_ssu = metadata["num_ssu"]
    per_npu = defaultdict(list)
    for request in requests:
        assert len(request.placement) == 1  # Reused for every layer.
        layer = request.placement[0]
        physical_id = request.load["original_request_id"]
        assert [disk for disk, _ in layer] == [
            sim.block_ring_hash_disk_id(physical_id, j, num_ssu)
            for j in range(len(layer))
        ]
        seconds = request.load["per_layer_us"] / 1e6
        per_disk = [math.fsum(size for disk, size in layer if disk == s)
                    for s in range(num_ssu)]
        per_npu[request.npu_id].append((per_disk, seconds))
    # Derive means directly from total bytes / total compute per NPU, without
    # relying on the builders' profile/rate tables or weighted-mean code.
    for s in range(num_ssu):
        expected_mean = math.fsum(
            math.fsum(volumes[s] for volumes, _ in rows) /
            math.fsum(seconds for _, seconds in rows)
            for rows in per_npu.values()
        )
        expected_max = math.fsum(
            max(volumes[s] / seconds for volumes, seconds in rows)
            for rows in per_npu.values()
        )
        assert metadata["time_weighted_per_ssu_nominal_gib_s"][s] == pytest.approx(expected_mean)
        assert metadata["static_per_ssu_upper_bound_gib_s"][s] == pytest.approx(expected_max)
        if "static_per_ssu_lower_bound_gib_s" in metadata:
            expected_min = math.fsum(
                min(volumes[s] / seconds for volumes, seconds in rows)
                for rows in per_npu.values()
            )
            assert metadata["static_per_ssu_lower_bound_gib_s"][s] == pytest.approx(expected_min)
    assert metadata["time_weighted_fleet_nominal_gib_s"] == pytest.approx(
        math.fsum(metadata["time_weighted_per_ssu_nominal_gib_s"]))


def test_diverse_shuffle_preserves_requests_and_placement(diverse_inputs):
    first, second = diverse_inputs
    assert _physical_requests(first[0]) == _physical_requests(second[0])
    assert [r.load["original_request_id"] for r in first[0]] != [
        r.load["original_request_id"] for r in second[0]]
    for requests, metadata in diverse_inputs:
        per_npu = _check_population(requests, metadata)
        expected = {
            profile["role"]: metadata["per_length_miss_counts"][str(profile["nql"])]
            for profile in metadata["profiles"]
        }
        assert len(expected) == 24
        assert metadata["quota_repeats"] == 1
        assert len(requests) == 32 * 30
        assert all(counts == expected for counts in per_npu.values())


def test_diverse_hash_and_capacity_metadata(diverse_inputs):
    for requests, metadata in diverse_inputs:
        _check_hash_and_disk_metadata(requests, metadata)
