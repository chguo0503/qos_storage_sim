#!/usr/bin/env python3
"""Run original full-pool Once with the frozen diverse-input measurement runner.

Only the runner's policy dispatch is extended in memory. No candidate-pool
extension is installed, no simulator/routing function is replaced, and the
original frozen source files are not edited. The same manifests and all
measurement/completion checks are retained.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib

import run_trial as trial
from simulator.policies import once as shared_path_once
from simulator.adapters import shared_path as shared_path_sim_adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def original_once(policy, stats):
    assert policy == 'once'
    # Mere importing of the extension does not install its context-manager
    # patches. Assert that original routing and adapter entry points persist.
    identities = (shared_path_once.once_path_ids,
                  shared_path_sim_adapter.shared_path_adapter)
    assert identities[0].__module__ == 'shared_path_once'
    assert identities[1].__module__ == 'shared_path_sim_adapter'
    yield
    assert identities == (shared_path_once.once_path_ids,
                          shared_path_sim_adapter.shared_path_adapter)
    assert stats['calls'] == stats['io_blocks'] == 0


def strategy(policy):
    assert policy == 'once'
    return 'once'


def main():
    trial.POLICIES = ('once',)
    trial.install_policy = original_once
    trial.simulator_strategy = strategy
    runner_digest = sha(__file__)
    writer = trial.original.write_json

    def write_record(path, obj):
        path = Path(path)
        if path.name == 'command.json':
            obj = dict(obj)
            obj['once_control'] = dict(
                runner='run_once_control.py', runner_sha256=runner_digest,
                candidate_pool='original complete category-legal pool',
                new_candidate_pool_extension_installed=False,
                routing='original shared_path_once.once_path_ids',
                sample_interval_ms=5,
            )
            assert sha(__file__) == runner_digest
            if obj['status'] == 'complete':
                raw = trial.original.read_json(path.parent/'result.json.gz')
                adapter = raw['adapter_statistics']
                assert adapter['strategy'] == 'once'
                assert adapter['routing_calls'] > 0
                assert adapter['collector_interval_ms'] == 5
                assert adapter['max_snapshot_age_ms'] <= 5 + 1e-7
                assert adapter['reorder_calls'] == 0
                assert adapter['assignment_count'] == 0
                assert adapter['cir_write_events'] == []
                assert raw['routing_statistics']['calls'] == 0
                assert raw['routing_statistics']['io_blocks'] == 0
                obj['once_control']['checks_passed'] = True
        return writer(path, obj)

    trial.original.write_json = write_record
    trial.main()


if __name__ == '__main__':
    main()
