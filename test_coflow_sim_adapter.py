"""End-to-end global client grants: actual sends, ACKs and backend choices.

Synthetic traces below are mechanism tests, not GLM utilization benchmarks.
The standalone source-copy replay additionally exercises a raw-data smoke
input through the production experiment runner.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import continuous_batch_sim as native
import sim
from coflow_sim_adapter import coflow_adapter
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from run_coflow_experiments import ROOT, placement_fingerprint, source_files
from shared_path_common import IO_GIB
from test_shared_path_sim_adapter import example_trace


def run(policy, requests=None, *, num_npu=4, num_ssu=2,
        queue_window_ms=.25, assignment="compute"):
    requests = example_trace() if requests is None else requests
    original = continuous_batch_input_fingerprint(requests)
    with coflow_adapter(policy, queue_window_ms=queue_window_ms,
                        assignment=assignment) as adapter:
        result = native.simulate_continuous_batch(
            requests, num_npu=num_npu, num_ssu=num_ssu, n_layers=8,
            batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
            qos_config=static_qos_config(), cross_request_layer0_prefetch=True,
            client_io_config=routing_strategy_specs()[1].client_config(),
            disk_bw_gbps=40, npu_bw_gbps=50, pressure_ttl_ms=5,
            submit_order_seed=17,
        )
        stats = adapter.statistics()
        runtime_manifests = tuple(r.manifest for r in adapter.context.requests.values())
        assignments = list(adapter.assignment_log)
        assert not adapter.reads and not adapter.states and not adapter.grants
        assert all(n.future_arrivals == 0 for n in adapter.context.npus)
    assert continuous_batch_input_fingerprint(requests) == original
    assert placement_fingerprint(requests) == placement_fingerprint(runtime_manifests)
    return result, stats, assignments, runtime_manifests


def assert_conservation(result, stats, requests, num_npu, num_ssu):
    total = sum(8 * len(r.placement[0]) for r in requests)
    assert all(result["invariants"].values())
    assert result["request_count"] == len(requests)
    assert {r["request_id"] for r in result["request_metrics"]} == {r.request_id for r in requests}
    assert stats["reserved_blocks"] == stats["acknowledged_blocks"] == total
    assert stats["ledger_end_counts_by_ssu"] == [0] * num_ssu
    client = stats["global_client"]
    assert client["issued_commands"] == total
    assert client["final_sent_unacked_by_ssu"] == [0] * num_ssu
    assert client["final_sent_unacked_by_npu"] == [0] * num_npu
    disk_limit = max(1, int(40 * client["queue_window_ms"] / (1000 * IO_GIB)))
    link_limit = max(1, int(50 * client["queue_window_ms"] / (1000 * IO_GIB)))
    assert all(0 <= count <= disk_limit for count in client["max_sent_unacked_by_ssu"])
    assert all(0 <= count <= link_limit for count in client["max_sent_unacked_by_npu"])
    assert stats["global_coflow"]["remaining_coflows_at_end"] == 0
    assert not stats["global_coflow"]["instant_other_ssu_queue_access"]
    assert not client["future_requests_visible"]
    assert not client["queued_io_migration"]
    assert all(d["max_backend_active_io"] == 1 for d in result["disk_stats"])
    return total


@pytest.mark.parametrize("policy", ("strategy1", "strategy2", "strategy3"))
def test_global_grants_conserve_exact_commands_and_never_exceed_credits(policy):
    requests = example_trace()
    result, stats, assignments, manifests = run(policy, requests)
    assert_conservation(result, stats, requests, 4, 2)
    assert len(assignments) == len(requests)
    by_id = {r.request_id: r for r in requests}
    executed = {r.request_id: r for r in manifests}
    for decision in assignments:
        original = by_id[decision["request_id"]]
        assert decision["arrival_time_ms"] == original.arrival_time_ms
        assert decision["original_npu_id"] == original.npu_id
        assert decision["assigned_npu_id"] == executed[original.request_id].npu_id
        assert 0 <= decision["arrival_time_ms"] - decision["snapshot_time_ms"] <= 5
        assert decision["decision_wall_us"] >= 0
    client = stats["global_client"]
    assert client["delayed_activation_count"] > 0
    assert client["empty_plans"] > 0
    assert any(event["first_issue_ms"] > event["activation_ms"]
               for event in client["first_issue_examples"])
    # Delay is actual emission delay, not an adjusted completion-time metric.
    for event in client["first_issue_examples"]:
        assert event["activation_ms"] >= by_id[event["request_id"]].arrival_time_ms
        assert event["first_issue_ms"] - event["activation_ms"] == pytest.approx(event["delay_ms"])
    assert stats["global_coflow"]["enabled"] == (policy == "strategy3")


@pytest.mark.parametrize("policy", ("strategy1", "strategy2", "strategy3"))
def test_periodic_telemetry_has_no_hidden_device_reads_or_cir_writes(policy):
    _, stats, _, _ = run(policy)
    times = stats["collector_times_ms"]
    assert len(times) > 2
    assert times == [5.0 * i for i in range(len(times))]
    assert stats["fresh_reads_by_ssu"] == [len(times)] * 2
    assert stats["native_fresh_reads_by_ssu"] == stats["fresh_reads_by_ssu"]
    assert stats["max_snapshot_age_ms"] <= 5 + 1e-9
    assert stats["cir_write_events"] == []
    assert stats["jit_prefetch"]["delay_count"] == 0  # No old per-layer JIT underneath.
    assert stats["reorder_calls"] == 0  # No old shared-adapter sorter underneath.


@pytest.mark.parametrize("policy", ("strategy2", "strategy3"))
def test_native_backend_gets_and_uses_real_cross_path_opportunities(policy):
    requests = example_trace()
    result, stats, _, _ = run(policy, requests)
    total = assert_conservation(result, stats, requests, 4, 2)
    backend = stats["backend_arbitration"]
    assert backend["enqueue_commands"] == backend["selection_calls"] == total
    assert backend["multi_path_opportunities"] > 0
    assert backend["cross_path_changed_vs_native_rr"] > 0
    assert any(size > 1 and count > 0 for size, count in backend["candidate_set_size_histogram"].items())
    assert backend["max_selected_tag_gap"] <= sim._EPS
    assert backend["cir_pir_unchanged"]
    assert (backend["global_priority_calls"] > 0) == (policy == "strategy3")
    assert backend["decision_examples"]
    for event in backend["decision_examples"]:
        assert event["selected_tag"] <= event["minimum_tag"] + sim._EPS
    # Same-Path movement is not guaranteed in this small trace; do not assert
    # nonzero movement merely because the hook exists. Separate disk tests
    # construct and verify its precise pending-command opportunity.


def test_tight_credit_contention_really_defers_but_does_not_deadlock():
    requests = tuple(replace(r, load={**r.load, "per_layer_us": 50.0},
                            placement=(tuple((block % 2, IO_GIB) for block in range(96)),))
                     for r in example_trace())
    result, stats, _, _ = run("strategy3", requests, queue_window_ms=.01)
    assert_conservation(result, stats, requests, 4, 2)
    assert stats["global_client"]["max_sent_unacked_by_ssu"] == [2, 2]
    assert stats["global_client"]["delayed_activation_count"] > 0
    assert stats["global_client"]["empty_plans"] > 0
    assert sum(r["io_stall_ms"] for r in result["request_metrics"]) > 0


@pytest.mark.parametrize("policy,ssus", (("strategy1", 5), ("strategy2", 6), ("strategy3", 7)))
def test_32_card_native_topology_and_arrival_population_are_preserved(policy, ssus):
    requests = tuple(ContinuousBatchRequest(
        npu * 100 + generation, npu, generation * 5.,
        {"category": "SS", "per_layer_us": 2000.0, "seq_len_k": 32, "nql": 128},
        (tuple((block % ssus, IO_GIB) for block in range(16)),))
        for npu in range(32) for generation in range(2))
    result, stats, assignments, _ = run(policy, requests, num_npu=32, num_ssu=ssus)
    assert_conservation(result, stats, requests, 32, ssus)
    assert len(assignments) == 64
    assert stats["native_fresh_reads_by_ssu"] == stats["fresh_reads_by_ssu"]


def test_fixed_assignment_option_does_not_silently_rebind_requests():
    requests = tuple(replace(r, npu_id=0, load={**r.load, "npu_id": 0}) for r in example_trace())
    _, _, decisions, manifests = run("strategy1", requests, assignment="fixed")
    assert all(d["original_npu_id"] == d["assigned_npu_id"] == 0 for d in decisions)
    assert all(r.npu_id == 0 for r in manifests)


def test_assignment_can_rebind_only_arrived_requests_without_changing_storage():
    requests = tuple(replace(r, npu_id=0, load={**r.load, "npu_id": 0}) for r in example_trace())
    result, stats, decisions, manifests = run("strategy1", requests)
    assert_conservation(result, stats, requests, 4, 2)
    assert any(d["assigned_npu_id"] != d["original_npu_id"] for d in decisions)
    assert placement_fingerprint(requests) == placement_fingerprint(manifests)
    assert {d["request_id"]: d["arrival_time_ms"] for d in decisions} == {
        r.request_id: r.arrival_time_ms for r in requests}


def test_unarrived_future_compute_cannot_change_earlier_assignment_or_io():
    current = tuple(r for r in example_trace() if r.arrival_time_ms == 0)
    future = replace(current[0], request_id=99999, arrival_time_ms=40.)
    a = current + (future,)
    b = current + (replace(future, load={**future.load, "per_layer_us": 100000.0}),)
    _, stats_a, assignment_a, _ = run("strategy1", a)
    _, stats_b, assignment_b, _ = run("strategy1", b)
    fields = ("request_id", "arrival_time_ms", "original_npu_id", "assigned_npu_id", "scores_ms")
    early = lambda rows: [tuple(r[k] for k in fields) for r in rows if r["arrival_time_ms"] < 40]
    assert early(assignment_a) == early(assignment_b)
    issues = lambda stats: [r for r in stats["global_client"]["first_issue_examples"]
                            if r["activation_ms"] < 40]
    assert issues(stats_a) == issues(stats_b)


def test_all_native_hooks_restore_after_an_exception():
    native_names = ("_handle_arrival", "_start_layer_io", "_register_submit",
                    "_register_complete", "_schedule_client_event", "_handle_client_submission")
    before = {name: getattr(native, name) for name in native_names}
    init = native._Context.__init__
    select = sim.DiskIOScheduler._select_qos_path
    pressure = sim.DiskIOScheduler.report_path_pressure_analysis
    with pytest.raises(RuntimeError, match="deliberate restoration check"):
        with coflow_adapter("strategy3"):
            raise RuntimeError("deliberate restoration check")
    assert all(getattr(native, name) is method for name, method in before.items())
    assert native._Context.__init__ is init
    assert sim.DiskIOScheduler._select_qos_path is select
    assert sim.DiskIOScheduler.report_path_pressure_analysis is pressure


def test_source_archive_list_replays_without_the_original_workspace(tmp_path):
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    for filename in source_files():
        shutil.copyfile(ROOT / filename, isolated / filename)
    output = isolated / "output"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([
        sys.executable, "-B", str(isolated / "run_coflow_experiments.py"),
        "--strategy", "strategy3", "--small", "--num-npu", "2", "--num-ssu", "5",
        "--output", str(output),
    ], cwd=isolated, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-4000:]
    saved = json.loads(next(output.glob("npu2_*.json")).read_text())
    assert saved["summary"]["invariants"]["all_requests_completed"]
    assert saved["input_placement_fingerprint"] == saved["execution_placement_fingerprint"]
    assert saved["metadata"]["arrival_workload"]["actual_postinitial_gib_s"] == pytest.approx(218.315646777)
    for required in ("coflow_sim_adapter.py", "coflow_client_policy.py", "coflow_disk_adapter.py",
                     "coflow_disk_policy.py", "coflow_joint_policy.py", "shared_path_sim_adapter.py",
                     "run_coflow_experiments.py", "run_shared_path_experiments.py", "data"):
        assert required in saved["source_artifacts"]
