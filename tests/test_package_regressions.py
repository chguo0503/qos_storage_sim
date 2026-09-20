"""Behavioral and dependency boundaries for the simulator package migration."""
import ast
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from simulator.core import continuous_batch_sim as native
from simulator.core import sim
from simulator.adapters.shared_path import shared_path_adapter


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def _patch_sites():
    return [(native._Context, '__init__'),
            (sim.DiskIOScheduler, 'report_path_pressure_analysis'),
            *((native, name) for name in ('_handle_control', '_plan_paths',
               '_register_complete', '_handle_arrival', '_start_layer_io')),
            (sim.PathQueue, 'enqueue'), (sim.PathQueue, 'activate_next')]


@pytest.mark.parametrize('policy', ('asu_baseline', 'od_baseline', 'once'))
def test_adapter_restores_all_patch_sites_after_nested_exception(policy):
    """A failed inner use must restore the active outer adapter and then core."""
    sites = _patch_sites()
    original = [getattr(obj, name) for obj, name in sites]
    with shared_path_adapter(policy):
        outer = [getattr(obj, name) for obj, name in sites]
        assert all(a is not b for a, b in zip(original, outer))
        with pytest.raises(RuntimeError, match='intentional inner failure'):
            with shared_path_adapter('once'):
                inner = [getattr(obj, name) for obj, name in sites]
                assert all(a is not b for a, b in zip(outer, inner))
                raise RuntimeError('intentional inner failure')
        assert all(getattr(obj, name) is value for (obj, name), value in zip(sites, outer))
    assert all(getattr(obj, name) is value for (obj, name), value in zip(sites, original))


def test_simulator_source_does_not_import_inputs_or_results():
    """The reusable simulator must not depend on experiment/input packages."""
    violations = []
    for path in sorted((ROOT / 'simulator').rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module or ''] if isinstance(node, ast.ImportFrom)
                       else [name.name for name in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if module.split('.')[0] in ('inputs', 'results'):
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}: {module}')
    assert violations == []


def test_core_reexports_identical_controller_contract_types():
    from simulator import contracts
    for name in ('ControlRequestView', 'CIRControlSnapshot', 'CIRControlDecision',
                 'CIRControlConfig', 'CausalLayerObservation', 'CausalLayerSnapshot',
                 'CausalLayerControlConfig'):
        assert getattr(native, name) is getattr(contracts, name)


def test_public_api_rejects_nonfinite_physical_bandwidth():
    """NaN/inf must not silently turn receive service into zero-time events."""
    from simulator.api import run_simulation
    from inputs.synthetic import build_requests
    requests = build_requests(num_npu=1, num_ssu=1, requests_per_npu=1)
    for parameter in ("disk_bw_gib_s", "npu_bw_gib_s"):
        for value in (float("nan"), float("inf"), 0.0, -1.0):
            with pytest.raises(ValueError, match="finite and positive"):
                run_simulation(requests, num_npu=1, num_ssu=1, **{parameter: value})


def _isolated_python(body, tmp_path, source_root=ROOT):
    # -I ignores PYTHONPATH and user site packages; only this explicit project
    # root is added. The import finder fails even on an optional/lazy dependency.
    prefix = f'''
import importlib.abc
import sys
sys.path.insert(0, {str(source_root)!r})
class NoExperimentInputs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('inputs', 'results'):
            raise AssertionError('Simulator imported forbidden dependency: ' + fullname)
sys.meta_path.insert(0, NoExperimentInputs())
'''
    return subprocess.run([sys.executable, '-I', '-c', prefix + body], cwd=tmp_path,
                          text=True, capture_output=True, timeout=15)


def test_core_runs_supplied_requests_without_experiment_inputs(tmp_path):
    # Copy only the reusable package: project data and input/result packages
    # are physically absent, not just omitted from the test's own imports.
    shutil.copytree(ROOT / 'simulator', tmp_path / 'simulator',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    result = _isolated_python('''
from simulator.core import continuous_batch_sim as native
from simulator.core import sim
from simulator.config import static_qos_config
gib = 176 * 1024 / 2**30
requests = tuple(native.ContinuousBatchRequest.from_normalized(
    i, i, 0.0, dict(request_id=i, npu_id=i, seq_len_k=32, nql=128,
        category='SS', per_layer_us=100.0, per_layer_kv_gb=4*gib),
    (tuple((sim.block_ring_hash_disk_id(i,k,2),gib) for k in range(4)),)) for i in range(2))
summary = native.simulate_continuous_batch(requests, num_npu=2, num_ssu=2,
    n_layers=2, batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
    qos_config=static_qos_config())
assert summary['request_count'] == 2
assert summary['completed_blocks'] == 16
assert all(summary['invariants'].values())
from simulator.api import run_simulation
for policy in ('asu_baseline', 'od_baseline', 'once'):
    result = run_simulation(requests, strategy=policy, num_npu=2, num_ssu=2, n_layers=2)
    assert result['summary']['completed_blocks'] == 16
    assert all(result['summary']['invariants'].values())
assert not any(n.split('.')[0] in ('inputs','results') for n in sys.modules)
print('core fixture completed')
''', tmp_path, source_root=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'core fixture completed' in result.stdout


@pytest.mark.parametrize('name', ('asu_baseline', 'od_baseline', 'once', 'new_once', 'strategy1', 'strategy2'))
def test_policy_main_runs_without_inputs_or_results(name, tmp_path):
    result = _isolated_python(f'''
import runpy
runpy.run_module('simulator.policies.{name}', run_name='__main__')
assert not any(n.split('.')[0] in ('inputs','results') for n in sys.modules)
''', tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip()


@pytest.mark.parametrize('policy', ('asu_baseline', 'od_baseline', 'once'))
def test_data_plane_matches_frozen_pre_refactor_execution(policy):
    """Compare real per-request/per-layer timing against captured old execution."""
    from inputs.runners.run_baseline_npu32_stress import load_manifest
    from inputs.runners.run_coflow_experiments import run_case
    requests, metadata = load_manifest(FIXTURES / 'pre_refactor_trace.json.gz')
    with gzip.open(FIXTURES / 'pre_refactor_expected.json.gz', 'rt') as stream:
        fixture = json.load(stream)
    expected = fixture['policies'][policy]
    observed = []
    original = native._register_complete
    def complete(context, flow):
        observed.append((flow.request_id, flow.layer, flow.block_idx))
        return original(context, flow)
    with patch.object(native, '_register_complete', complete):
        result = run_case(requests, metadata, strategy=policy, assignment='fixed')
    summary = result['summary']
    assert result['input_fingerprint'] == fixture['input_fingerprint']
    assert all(summary['invariants'].values())
    assert summary['request_count'] == expected['request_count']
    assert len(observed) == summary['completed_blocks'] == expected['completed_blocks']
    assert summary['cross_request_layer0_prefetches'] == expected['cross_request_L0_prefetches']
    assert summary['makespan_ms'] == pytest.approx(expected['makespan_ms'], abs=1e-9, rel=0)
    assert 100*summary['fleet_npu_compute_utilization'] == pytest.approx(expected['fleet_U_percent'], abs=1e-9, rel=0)
    completions = sorted([[r['request_id'], r['admission_time_ms'], r['completion_time_ms'], r['own_compute_ms']]
                          for r in summary['request_metrics']])
    layers = sorted([[b['member_request_ids'][0], m['layer'], m['io_start_time_ms'],
                      m['io_ready_time_ms'], m['compute_start_ms'], m['compute_end_ms'], m['io_barrier_wait_ms']]
                     for b in summary['microbatch_metrics'] for m in b['layer_metrics']])
    np.testing.assert_allclose(completions, expected['completion_times'], rtol=0, atol=1e-9)
    np.testing.assert_allclose(layers, expected['layer_times'], rtol=0, atol=1e-9)
    from simulator.api import run_simulation
    public = run_simulation(requests, strategy=policy, num_npu=metadata['num_npu'],
                            num_ssu=metadata['num_ssu'], n_layers=8, seed=metadata['seed'])
    assert public['input_fingerprint'] == result['input_fingerprint']
    assert public['summary'] == summary
    # These three fields measure execution on the host CPU, not simulated
    # time. All routing, queue, conservation and ownership counters must match.
    host_timers = {'routing_wall_us', 'collector_wall_us', 'ack_ledger_wall_us'}
    assert public['adapter_statistics'].keys() == result['adapter_statistics'].keys()
    assert {k: v for k, v in public['adapter_statistics'].items() if k not in host_timers} == {
        k: v for k, v in result['adapter_statistics'].items() if k not in host_timers}


@pytest.mark.parametrize('policy', ('static', 'mild', 'aggressive'))
def test_slo_adapter_restores_hooks_after_exception(policy):
    from simulator.adapters import shared_path, slo_pool
    from simulator.policies import once
    sites = _patch_sites() + [(shared_path, 'shared_path_adapter'), (once, 'once_path_ids')]
    original = [getattr(obj, name) for obj, name in sites]
    with pytest.raises(RuntimeError, match='intentional SLO failure'):
        with slo_pool.install_policy(policy):
            assert shared_path.shared_path_adapter is not original[-2]
            with shared_path.shared_path_adapter(strategy='once'):
                assert once.once_path_ids is not original[-1]
                assert native._plan_paths is not original[3]
                raise RuntimeError('intentional SLO failure')
    assert all(getattr(obj, name) is value for (obj, name), value in zip(sites, original))


def test_source_inventory_tracks_all_nested_package_files():
    from inputs.provenance import source_files
    from inputs.runners import run_coflow_experiments, run_shared_path_experiments
    from inputs.runners import run_multi_ssu_stall_experiments
    expected = {'data'} | {str(p.relative_to(ROOT)) for package in ('simulator', 'inputs')
                           for p in (ROOT / package).rglob('*.py')}
    assert set(source_files()) == expected
    assert len(source_files()) == len(expected)
    for runner in (run_coflow_experiments, run_shared_path_experiments,
                   run_multi_ssu_stall_experiments):
        assert runner.source_files is source_files


def test_manifest_legacy_exports_are_identical_and_roundtrip(tmp_path):
    from inputs import manifest
    from inputs.runners import run_baseline_npu32_stress as legacy
    for name in ('read_json', 'write_json', 'save_manifest', 'load_manifest'):
        assert getattr(legacy, name) is getattr(manifest, name)
    requests, metadata = manifest.load_manifest(FIXTURES / 'pre_refactor_trace.json.gz')
    target = tmp_path / 'frozen.json.gz'
    manifest.save_manifest(target, requests, metadata)
    restored, restored_metadata = manifest.load_manifest(target)
    assert native.continuous_batch_input_fingerprint(restored) == native.continuous_batch_input_fingerprint(requests)
    assert restored_metadata == metadata


def test_catalog_prepared_input_roundtrip_matches_convenience_runner():
    from inputs import catalog
    from simulator.config import static_qos_config
    gib = 4 * 176 * 1024 / 2**30
    table = {(32, 128): (gib / 0.0001, 100., 0.2, gib)}
    prepared = catalog.prepare_simulation_inputs(table, total_requests=2, n_layers=2,
        num_disk=2, workload_seed=17, placement_seed=19, arrival_delay_seed=23,
        arrival_delay_max_ms=0.)
    kwargs = dict(num_npu=2, num_disk=2, n_layers=2, qos_config=static_qos_config(),
        workload_seed=17, placement_seed=19, arrival_delay_seed=23,
        arrival_delay_max_ms=0.)
    direct = sim.simulate_continuous(prepared_inputs=prepared, **kwargs)
    automatic = catalog.simulate_continuous(table, **kwargs)
    assert automatic[1] == direct[1]
    for a, b in zip(automatic[0], direct[0]):
        for name in ('npu_id', 'done', 'current_load', 'block_placement',
                     'total_compute_ms', 'ttft_ms', 'processing_ttft_ms',
                     'io_wait_L0_ms', 'io_wait_L1_ms', 'io_wait_L2plus_ms',
                     'link_completed_gb', 'request_io_count', 'compute_done_up_to'):
            assert getattr(a, name) == getattr(b, name)


def test_manifest_cli_inherits_seed_and_allows_explicit_override(tmp_path):
    from inputs.manifest import save_manifest
    from inputs.synthetic import build_requests
    from simulator.api import run_simulation
    requests = build_requests(num_npu=2, num_ssu=2, requests_per_npu=2, seed=19)
    metadata = dict(num_npu=2, num_ssu=2, n_layers=2, seed=19)
    path = tmp_path / 'seed19.json.gz'
    save_manifest(path, requests, metadata)
    fingerprint = native.continuous_batch_input_fingerprint(requests)
    for args, expected_seed in (([], 19), (['--seed', '7'], 7)):
        output = tmp_path / f'result_seed{expected_seed}.json'
        command = [sys.executable, '-m', 'inputs', '--input', 'manifest',
                   '--manifest', str(path), '--strategy', 'od_baseline',
                   '--output', str(output), *args]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(output.read_text())
        assert result['configuration']['seed'] == expected_seed
        assert result['input_metadata'] == metadata
        assert result['input_fingerprint'] == fingerprint
        expected = run_simulation(requests, strategy='od_baseline', num_npu=2,
                                  num_ssu=2, n_layers=2, seed=expected_seed)
        # JSON normalizes integer mapping keys, so compare like for like.
        assert result['summary'] == json.loads(json.dumps(expected['summary']))
