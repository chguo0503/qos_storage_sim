"""Ring-only generation, immutable placement, and historical input preservation."""
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from simulator.core import sim
from simulator.core.continuous_batch_sim import continuous_batch_input_fingerprint
from inputs.runners.run_baseline_npu32_stress import build_workload, load_manifest, save_manifest

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=2)
def generated(blocks='exact'):
    return build_workload(family='raw', num_npu=2, num_ssu=3,
                          raw_keys='32:256,32:4096', horizon_ms=1,
                          seed=7, blocks=blocks)


def test_default_generator_uses_ring_and_conserves_profile_bytes():
    requests, metadata = generated()
    assert metadata['layout'] == sim.PLACEMENT_BLOCK_RING_HASH
    assert metadata['placement_identity_key'] == 'request_id'
    for request in requests:
        assert len(request.placement) == 1  # Same placement reused by all layers.
        layer = request.placement[0]
        assert [disk for disk, _ in layer] == [
            sim.block_ring_hash_disk_id(request.request_id, j, 3) for j in range(len(layer))]
        assert math.fsum(size for _, size in layer) == pytest.approx(
            request.load['per_layer_kv_gb'], abs=1e-12)
    explicit, explicit_meta = build_workload(family='raw', num_npu=2, num_ssu=3,
                                             raw_keys='32:256,32:4096', horizon_ms=1,
                                             seed=7, layout='hash')
    assert continuous_batch_input_fingerprint(requests) == continuous_batch_input_fingerprint(explicit)
    assert metadata == explicit_meta


@pytest.mark.parametrize('layout', ('local', 'stripe', 'hotspot'))
def test_retired_layouts_are_rejected_by_api_and_cli(layout):
    with pytest.raises(ValueError, match='Only ring-hash'):
        build_workload(layout=layout)
    result = subprocess.run([sys.executable, str(ROOT/'inputs/runners/run_baseline_npu32_stress.py'),
                             '--layout', layout, '--describe-only'],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr


def test_manifest_roundtrip_retains_hash_placement_and_identity(tmp_path):
    requests, metadata = generated()
    path = tmp_path/'hash.json.gz'
    save_manifest(path, requests, metadata)
    restored, actual_meta = load_manifest(path)
    assert actual_meta == metadata
    assert continuous_batch_input_fingerprint(restored) == continuous_batch_input_fingerprint(requests)
    before = path.read_bytes()
    save_manifest(path, restored, actual_meta)
    assert path.read_bytes() == before


def test_archived_stripe_manifest_is_read_without_relabeling_or_rewriting():
    path = ROOT/'tests/fixtures/archived_stripe_two_requests.json.gz'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == '2845b8214be2f7a27a91924889f8bdb675b5e813ae8dc043d4c87b04e20109ba'
    with gzip.open(path, 'rt') as stream:
        frozen = json.load(stream)
    requests, metadata = load_manifest(path)
    assert len(requests) == 2
    assert metadata == frozen['metadata']
    assert metadata['layout'] == 'stripe_npu_mod_ssu'
    assert metadata['placement_rule'] == (
        '(block_index + npu_id) % num_ssu, exact 176KiB per block; '
        'physical placement stays with original identity')
    assert (metadata['num_npu'], metadata['num_ssu'], metadata['n_layers']) == (32, 3, 8)
    assert metadata['order'] == 'random' and metadata['seed'] == 7
    assert [(q.request_id, q.npu_id, q.load['original_request_id']) for q in requests] == [
        (0, 0, 67), (1000003, 1, 1000069)]
    for request, frozen_row, block_count in zip(requests, frozen['requests'], (224, 1022)):
        assert request.load == frozen_row['load']
        assert request.arrival_time_ms == 0
        assert request.placement == (tuple(
            ((block_index + request.npu_id) % 3, 176 * 1024 / 2**30)
            for block_index in range(block_count)),)
    assert continuous_batch_input_fingerprint(requests) == (
        'ab48a24b4c011ec24c00f7fef0a18c6de8dc133fc7c1aa40dcf71230af9a18c2')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_single_ssu_generator_is_ring_hash_degenerate_case():
    from inputs.runners.run_baseline_4npu_ssu1_low_utilization import _placement
    profile = dict(seq_len_k=1, nql=128, per_layer_kv_gib=7*176*1024/2**30)
    layer = _placement(profile, request_id=23)
    assert len(layer) == 7
    assert {disk for disk, _ in layer} == {0}
    assert math.fsum(size for _, size in layer) == profile['per_layer_kv_gib']
