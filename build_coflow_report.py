"""Independent accounting scaffold for the global-coflow, 5-ms experiment.

This is a new experiment, not an alias for the prior shared-Path/JIT report.
Only common event-accounting primitives are reused. In particular the old
Strategy 1/2 JIT requirement must never be applied to these new policies.
The final mechanism contract and PDF renderer are populated after freezing
the new scheduler interfaces; this scaffold does not manufacture results.
"""

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from build_shared_path_report import (
    CORE_FILES, EPS_MS, _same_numeric, audit_adapter, audit_assignment,
    manifest_fingerprint, markdown_table, recompute_metrics,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/coflow_global_5ms_experiments"
EXPERIMENT_ID = "coflow_global_5ms_v1"
REGIME = "arrival218"
POLICIES = ("baseline", "once", "new_once", "strategy1", "strategy2", "strategy3")
SEEDS = (20260906, 20260907)
SSU_COUNTS = (5, 6, 7)
IO_BYTES = 176 * 1024
IO_GIB = IO_BYTES / 2**30
INPUT_CLOCK_VERSION = "canonical_reference_ps_v2"


def audit_input_clock(metadata):
    """Reject the pre-canonical pilot even though its experiment name matched."""
    arrival = metadata["arrival_workload"]
    assert arrival.get("input_clock_version") == INPUT_CLOCK_VERSION, "pre-canonical input clock cannot enter formal comparison"
    assert arrival.get("reference_arrival_quantum_ms") == 1e-9, "reference arrival quantum changed"


def policy_configuration(saved):
    """Read a selected development configuration, never silently assume .25 ms."""
    config = saved["policy_config"]
    window = config["queue_window_ms"]
    assert math.isfinite(window) and window > 0, "invalid global credit window"
    assert config["assignment"] in ("compute", "fixed", "pipeline"), "unknown assignment policy"
    assert config["joint_rule"] in ("least_slack", "urgent_short"), "unknown joint priority rule"
    return window, config["assignment"], config["joint_rule"]


def audit_policy_configurations(saved_cases):
    """Only effective parameters must match across policies.

    S1/S2 retain a recorded inert joint-rule argument. It need not match S3's
    subsequently selected rule, but every actual S3 uses one frozen rule.
    """
    client_configs, joint_rules = set(), set()
    for saved in saved_cases:
        if saved["strategy"] not in ("strategy1", "strategy2", "strategy3"):
            continue
        window, assignment, rule = policy_configuration(saved)
        client_configs.add((window, assignment))
        if saved["strategy"] == "strategy3":
            joint_rules.add(rule)
    assert len(client_configs) <= 1, "new policies used different effective client configurations"
    assert len(joint_rules) <= 1, "Strategy 3 used different effective joint rules"
    if not client_configs:
        return None
    window, assignment = next(iter(client_configs))
    return {"queue_window_ms": window, "assignment": assignment,
            "joint_rule": next(iter(joint_rules)) if joint_rules else None}


def audit_environment(data_dir, selected, *, required=False):
    path = Path(data_dir) / "environment.json"
    if not path.exists():
        assert not required, "formal environment record missing"
        return None
    environment = json.loads(path.read_text())
    assert environment["experiment_id"] == EXPERIMENT_ID, "environment namespace mismatch"
    assert environment["input_clock_version"] == INPUT_CLOCK_VERSION, "environment input clock mismatch"
    hardware = environment["simulated_hardware"]
    for key, expected in (("num_npu", 32), ("ssu_counts", [5, 6, 7]),
                          ("ssd_bandwidth_gib_s", 40), ("npu_link_bandwidth_gib_s", 50),
                          ("io_bytes", IO_BYTES), ("layers", 8), ("collector_ms", 5)):
        assert hardware[key] == expected, ("environment hardware mismatch", key)
    for case in selected.values():
        assert case["saved"]["core_and_policy_sha256"] == environment["source_files_sha256"], "case sources differ from frozen environment"
    tree_hash = hashlib.sha256(json.dumps(environment["source_files_sha256"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert tree_hash == environment["source_tree_sha256"], "environment source tree fingerprint mismatch"
    # Individual source archive contents were already hashed in audit_sources.
    # The workspace tar is not assumed to exist beside the result environment.
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "python_version": environment["python_version"], "platform": environment["platform"],
            "source_tree_sha256": environment["source_tree_sha256"]}


def audit_canonical_reference_order(requests):
    for row in requests:
        reference = row["load"]["reference_arrival_ms"]
        assert math.isfinite(reference) and reference == round(reference, 9), "input reference arrival is not canonical 1 ps"
    later = [row for row in requests if row["arrival_time_ms"] > 0]
    by_reference = sorted(later, key=lambda row: (row["load"]["reference_arrival_ms"], row["request_id"]))
    by_actual = sorted(later, key=lambda row: (row["arrival_time_ms"], row["request_id"]))
    assert [r["request_id"] for r in by_reference] == [r["request_id"] for r in by_actual], "retimed input order differs from canonical reference order"


def complete_run_accounting(summary):
    """Disjoint compute/stall/nonterminal-idle/terminal-tail accounting.

    Tail is makespan minus that card's final request completion. It is not all
    idle: before its final completion a card may have startup/interior gaps.
    A card with no admitted work has its whole run in tail, by convention.
    No simulated result is altered or re-windowed by this calculation.
    """
    npu_count, makespan = summary["num_npu"], summary["makespan_ms"]
    compute, stall = [0.0] * npu_count, [0.0] * npu_count
    active, last = [0.0] * npu_count, [0.0] * npu_count
    for batch in summary["microbatch_metrics"]:
        npu = batch["npu_id"]
        active[npu] += batch["completion_time_ms"] - batch["admission_time_ms"]
        last[npu] = max(last[npu], batch["completion_time_ms"])
        for layer in batch["layer_metrics"]:
            compute[npu] += layer["compute_end_ms"] - layer["compute_start_ms"]
            stall[npu] += layer["io_barrier_wait_ms"]
    tail = [makespan - end for end in last]
    nonterminal_idle = [end - admitted for end, admitted in zip(last, active)]
    for i in range(npu_count):
        _same_numeric(compute[i] + stall[i], active[i], "whole-run active decomposition")
        assert nonterminal_idle[i] >= -EPS_MS and tail[i] >= -EPS_MS, "negative whole-run idle"
        _same_numeric(compute[i] + stall[i] + nonterminal_idle[i] + tail[i],
                      makespan, "whole-run four-way decomposition")
    mean = lambda values: sum(values) / npu_count
    return {
        "whole_compute_ms_by_npu": compute,
        "whole_stall_ms_by_npu": stall,
        "nonterminal_idle_ms_by_npu": [max(0.0, value) for value in nonterminal_idle],
        "terminal_tail_ms_by_npu": tail,
        "last_completion_ms_by_npu": last,
        "mean_whole_compute_ms": mean(compute),
        "mean_whole_stall_ms": mean(stall),
        "mean_nonterminal_idle_ms": mean([max(0.0, value) for value in nonterminal_idle]),
        "mean_terminal_tail_ms": mean(tail),
        "max_terminal_tail_ms": max(tail),
        "last_completion_spread_ms": max(last) - min(last),
    }


def manifest_facts(requests, *, num_ssu, num_npu=32, n_layers=8):
    """Compact immutable input facts; retain counts separately for every SSD.

    This is not a scheduler snapshot. Arrival assigns a request's complete
    known eight-layer work to its arrival time; it does not mean those bytes
    were issued, serviced, or had a deadline at that instant.
    """
    facts = {}
    for row in requests:
        rid, placement = row["request_id"], row["placement"]
        assert rid not in facts, "duplicate input request ID"
        assert 0 <= row["npu_id"] < num_npu, "invalid original NPU"
        assert len(placement) in (1, n_layers), "incomplete input layer placement"
        per_ssu = [0] * num_ssu
        layer_counts = []
        layers = placement * n_layers if len(placement) == 1 else placement
        for layer in layers:
            current_counts = [0] * num_ssu
            for ssu, volume in layer:
                assert isinstance(ssu, int) and 0 <= ssu < num_ssu, "invalid input SSD"
                assert volume == IO_GIB, "I/O must be exactly one 176 KiB block"
                per_ssu[ssu] += 1
                current_counts[ssu] += 1
            layer_counts.append(tuple(current_counts))
        arrival = row["arrival_time_ms"]
        compute = n_layers * row["load"]["per_layer_us"] / 1000
        assert math.isfinite(arrival) and arrival >= 0 and math.isfinite(compute) and compute > 0, "invalid input time"
        facts[rid] = {"request_id": rid, "original_npu_id": row["npu_id"],
                      "arrival_ms": arrival, "ideal_compute_ms": compute,
                      "io_count": sum(per_ssu), "io_by_ssu": tuple(per_ssu),
                      "layer_io_by_ssu": tuple(layer_counts),
                      "seq_len_k": row["load"].get("seq_len_k"), "nql": row["load"].get("nql"),
                      "work_gib": sum(per_ssu) * IO_GIB}
    return facts


def arrival_accounting(facts, *, num_ssu, num_npu=32, target_gib_s=None):
    """Separate zero-time backlog, positive-time arrival work, and ideal demand."""
    initial = [row for row in facts.values() if row["arrival_ms"] == 0]
    later = sorted((row for row in facts.values() if row["arrival_ms"] > 0),
                   key=lambda row: (row["arrival_ms"], row["request_id"]))
    assert later, "no positive-time arrivals to define an arrival rate"
    last = later[-1]["arrival_ms"]
    positive_gib = sum(row["work_gib"] for row in later)
    actual_rate = 1000 * positive_gib / last
    per_ssu_later = [sum(row["io_by_ssu"][s] for row in later) * IO_GIB for s in range(num_ssu)]
    per_ssu_all = [sum(row["io_by_ssu"][s] for row in facts.values()) * IO_GIB for s in range(num_ssu)]
    per_npu_compute = [sum(row["ideal_compute_ms"] for row in facts.values()
                           if row["original_npu_id"] == npu) for npu in range(num_npu)]
    per_ssu_ideal = [sum(1000 * IO_GIB * sum(row["io_by_ssu"][s] for row in facts.values()
                                         if row["original_npu_id"] == npu) / per_npu_compute[npu]
                         for npu in range(num_npu) if per_npu_compute[npu] > 0)
                      for s in range(num_ssu)]
    prefix_max_error = None
    if target_gib_s is not None:
        _same_numeric(actual_rate, target_gib_s, "positive-arrival rate differs from target")
        # The new workload generator uses cumulative carried work / target.
        # Check the complete arrival schedule, not merely its final average.
        prefix, errors = 0.0, []
        for row in later:
            prefix += row["work_gib"]
            expected = 1000 * prefix / target_gib_s
            errors.append(abs(row["arrival_ms"] - expected))
            _same_numeric(row["arrival_ms"], expected, "arrival prefix differs from cumulative-work schedule")
        prefix_max_error = max(errors)
    bucket_gib = defaultdict(float)
    for row in later:
        # Right-closed intervals (0,1000], (1000,2000], ...; initial is separate.
        bucket_gib[max(0, math.ceil(row["arrival_ms"] / 1000) - 1)] += row["work_gib"]
    return {"initial_request_count": len(initial),
            "initial_work_gib": sum(row["work_gib"] for row in initial),
            "positive_request_count": len(later), "positive_work_gib": positive_gib,
            "last_arrival_ms": last, "actual_positive_arrival_gib_s": actual_rate,
            "positive_arrival_gib_s_by_ssu": [1000 * value / last for value in per_ssu_later],
            "total_read_gib_by_ssu": per_ssu_all,
            "ideal_full_compute_demand_gib_s_by_ssu": per_ssu_ideal,
            "ideal_full_compute_demand_gib_s": sum(per_ssu_ideal),
            "arrival_prefix_max_error_ms": prefix_max_error,
            "positive_arrival_work_gib_by_right_closed_second": dict(sorted(bucket_gib.items())),
            "aggregate_arrival_exceeds_ssd_capacity": actual_rate > num_ssu * 40 + EPS_MS,
            "any_ssd_arrival_exceeds_capacity": any(1000 * value / last > 40 + EPS_MS for value in per_ssu_later)}


def audit_request_identity(facts, metrics, policy):
    assert len(facts) == metrics["request_count"], "different input/execution populations"
    for actual in metrics["records"]:
        row = facts[actual["request_id"]]
        _same_numeric(row["arrival_ms"], actual["arrival_ms"], "request arrival changed")
        _same_numeric(row["ideal_compute_ms"], actual["ideal_compute_ms"], "request compute changed")
        assert row["io_count"] == actual["io_count"], "request I/O total changed"
        if policy in ("baseline", "once", "new_once"):
            assert row["original_npu_id"] == actual["npu_id"], "fixed-NPU policy changed binding"


def cross_request_capacity_witnesses(summary, facts, *, start_ms=1000.0, end_ms=2000.0):
    """Find actual, in-window L0 transitions with an isolated-capacity miss.

    This is an after-the-fact certificate for the *recorded* activation and
    predecessor, not a prediction that all placements or earlier schedules
    must create this pair. A lower bound below the budget proves nothing.
    Layer metrics record activation, not first actual SSU command service.
    """
    by_npu = defaultdict(list)
    requests = {row["request_id"]: row for row in summary["request_metrics"]}
    for batch in summary["microbatch_metrics"]:
        by_npu[batch["npu_id"]].append(batch)
    witnesses = []
    for npu, batches in sorted(by_npu.items()):
        batches.sort(key=lambda batch: batch["admission_time_ms"])
        for previous, current in zip(batches, batches[1:]):
            rid = current["member_request_ids"][0]
            previous_id = previous["member_request_ids"][0]
            row, previous_row = facts[rid], facts[previous_id]
            layer, previous_layer = current["layer_metrics"][0], previous["layer_metrics"][-1]
            activation, deadline = layer["io_start_time_ms"], previous["completion_time_ms"]
            if not requests[rid]["layer0_cross_request_prefetched"]:
                continue
            # Restrict this illustrative witness to the full-last-layer budget,
            # not late arrival partway through previous C or a cold layer zero.
            if (abs(activation - previous_layer["compute_start_ms"]) > EPS_MS
                    or row["arrival_ms"] > activation + EPS_MS
                    or not start_ms <= deadline < end_ms):
                continue
            _same_numeric(current["admission_time_ms"], deadline, "cross-request admission gap")
            counts = row["layer_io_by_ssu"][0]
            lower = 1000 * IO_GIB * max(max(counts) / 40, sum(counts) / 50)
            budget = deadline - activation
            if lower <= budget + EPS_MS:
                continue
            actual_stall = layer["io_barrier_wait_ms"]
            assert actual_stall + EPS_MS >= lower - budget, "recorded L0 violates isolated lower bound"
            ready = layer["io_ready_time_ms"]
            _same_numeric(layer["compute_start_ms"], max(deadline, ready), "L0 readiness/barrier mismatch")
            witnesses.append({
                "npu_id": npu, "previous_request_id": previous_id, "request_id": rid,
                "previous_seq_len_k": previous_row["seq_len_k"], "previous_nql": previous_row["nql"],
                "seq_len_k": row["seq_len_k"], "nql": row["nql"],
                "previous_compute_ms": budget, "own_compute_ms": row["ideal_compute_ms"] / 8,
                "layer_io_count": sum(counts), "layer_io_by_ssu": list(counts),
                "layer_read_gib": sum(counts) * IO_GIB,
                "activation_ms": activation, "deadline_ms": deadline, "ready_ms": ready,
                "compute_start_ms": layer["compute_start_ms"],
                "isolated_read_lower_bound_ms": lower, "unavoidable_boundary_stall_ms": lower - budget,
                "actual_layer0_stall_ms": actual_stall,
                "window_layer0_stall_ms": max(0.0, min(end_ms, layer["compute_start_ms"]) - max(start_ms, deadline)),
            })
    return sorted(witnesses, key=lambda row: (-row["window_layer0_stall_ms"], row["deadline_ms"], row["request_id"]))


def load_input_facts(saved, data_dir, cache):
    """Verify each immutable input archive once, then reuse only compact facts."""
    artifact = saved["input_artifact"]
    path = Path(data_dir) / artifact["path_relative_to_data"]
    key = (str(path.resolve()), artifact["sha256"])
    if key not in cache:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == artifact["sha256"], "input artifact SHA mismatch"
        manifest = json.loads(raw)
        audit_canonical_reference_order(manifest["requests"])
        native_hash = manifest_fingerprint(manifest["requests"])
        assert native_hash == manifest["input_fingerprint"], "native input fingerprint mismatch"
        logical = [{name: r[name] for name in ("request_id", "npu_id", "arrival_time_ms", "load")}
                   for r in manifest["requests"]]
        logical_hash = hashlib.sha256(json.dumps(logical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        assert logical_hash == manifest["logical_input_fingerprint"], "logical input fingerprint mismatch"
        physical = [{"request_id": row["request_id"], "placement": row["placement"]}
                    for row in sorted(manifest["requests"], key=lambda row: row["request_id"])]
        placement_hash = hashlib.sha256(json.dumps(physical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        facts = manifest_facts(manifest["requests"], num_ssu=saved["metadata"]["num_ssu"])
        cache[key] = {"facts": facts, "native_hash": native_hash, "logical_hash": logical_hash,
                      "placement_hash": placement_hash}
    result = cache[key]
    assert saved["input_fingerprint"] == result["native_hash"], "case/input fingerprint mismatch"
    assert saved["logical_input_fingerprint"] == result["logical_hash"], "case/logical fingerprint mismatch"
    assert saved["input_placement_fingerprint"] == saved["execution_placement_fingerprint"] == result["placement_hash"], "execution changed per-request ordered SSD placement"
    return result


def audit_sources(saved, data_dir):
    assert set(saved["source_artifacts"]) == set(saved["core_and_policy_sha256"]), "source archive closure incomplete"
    for name, relative in saved["source_artifacts"].items():
        digest = hashlib.sha256((Path(data_dir) / relative).read_bytes()).hexdigest()
        assert digest == saved["core_and_policy_sha256"][name], ("source archive SHA mismatch", name)
    assert saved["core_and_policy_sha256"]["data"] == saved["metadata"]["source"]["source_sha256"], "input data source SHA mismatch"
    for name in CORE_FILES:
        assert saved["core_and_policy_sha256"][name] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), ("native core changed", name)


def audit_coflow_control(saved):
    """Audit only observable counters and logged native-equivalence decisions.

    Unlogged send timestamps and arbitrary counterfactual service traces are
    not reconstructed from configuration values. Scope/nonpreemption claims
    additionally depend on the archived implementation and integration tests.
    """
    policy, summary, stats = saved["strategy"], saved["summary"], saved["adapter_statistics"]
    assert not stats.get("jit_prefetch", {}).get("enabled", False), "old independent JIT policy mixed into global-coflow experiment"
    if policy in ("baseline", "once", "new_once"):
        assert "global_client" not in stats, "reference policy unexpectedly uses global grants"
        return {"global_grants_verified": False, "backend_equivalence_verified": False,
                "logged_backend_examples_verified": 0}
    window, assignment, joint_rule = policy_configuration(saved)
    global_client = stats["global_client"]
    assert global_client["queue_window_ms"] == window and global_client["grant_batch_size"] == 64, "global credit/batch configuration mismatch"
    assert global_client["assignment"] == assignment, "actual assignment differs from declared policy configuration"
    assert global_client["issued_commands"] == summary["submitted_blocks"] == summary["completed_blocks"], "global grant/send conservation failed"
    assert global_client["activation_count"] == 8 * summary["request_count"], "global activated-layer population mismatch"
    assert 0 <= global_client["delayed_activation_count"] <= global_client["activation_count"], "invalid first-issue delay count"
    disk_limit = max(1, int(40 * window / (1000 * IO_GIB)))
    link_limit = max(1, int(50 * window / (1000 * IO_GIB)))
    for field, limit, size in (("ssu", disk_limit, summary["num_ssu"]),
                               ("npu", link_limit, summary["num_npu"])):
        maxima = global_client["max_sent_unacked_by_" + field]
        final = global_client["final_sent_unacked_by_" + field]
        assert len(maxima) == len(final) == size, "global credit topology mismatch"
        assert all(0 <= value <= limit for value in maxima), "global outstanding credit exceeded"
        assert all(value == 0 for value in final), "global outstanding credit did not drain"
    assert global_client["queued_io_migration"] is False and global_client["future_requests_visible"] is False, "global client permission violation"
    _same_numeric(global_client["grant_execution_interval_us"], .1, "global issue interval changed")
    joint = stats["global_coflow"]
    assert joint["priority_rule"] == joint_rule, "actual joint priority differs from declared policy configuration"
    assert joint["enabled"] is (policy == "strategy3"), "joint coflow priority enabled in wrong policy"
    assert joint["remaining_coflows_at_end"] == 0, "activated coflow state did not drain"
    assert joint["instant_other_ssu_queue_access"] is False, "joint policy accessed instant remote-SSD queue"
    for example in global_client["first_issue_examples"]:
        assert example["first_issue_ms"] >= example["activation_ms"] - EPS_MS, "issued before layer activation"
        _same_numeric(example["first_issue_ms"] - example["activation_ms"], example["delay_ms"], "global first-issue delay mismatch")
    backend = stats["backend_arbitration"]
    if policy == "strategy1":
        assert not backend, "Strategy 1 unexpectedly changes queued service"
        return {"global_grants_verified": True, "backend_equivalence_verified": False,
                "logged_backend_examples_verified": 0}
    assert backend["mode"] == policy and backend["cir_pir_unchanged"], "backend QoS configuration changed"
    assert backend["finite_pir_supported"] is False, "native finite-PIR support incorrectly claimed"
    epsilon = backend["native_tag_epsilon"]
    assert 0 < epsilon <= 1e-12, "native tag tolerance changed"
    # Native tests selected <= rounded(minimum + epsilon). Subtracting minimum
    # can report slightly more than epsilon after that addition is rounded.
    # Exact eligibility is asserted in every native adapter decision; below
    # we check the aggregate scale and the recorded decisions themselves.
    assert 0 <= backend["max_selected_tag_gap"] <= 2 * epsilon, "higher-finish Path gap exceeds native tolerance scale"
    histogram = {int(size): count for size, count in backend["candidate_set_size_histogram"].items()}
    assert all(size >= 1 and count >= 0 for size, count in histogram.items()), "invalid equivalent-set histogram"
    assert sum(histogram.values()) == backend["selection_calls"], "backend selection accounting mismatch"
    opportunities = sum(count for size, count in histogram.items() if size > 1)
    assert opportunities == backend["multi_path_opportunities"], "cross-Path opportunity count mismatch"
    assert 0 <= backend["cross_path_changed_vs_native_rr"] <= opportunities, "cross-Path changed without equivalent alternatives"
    assert 0 <= backend["same_path_pending_head_changed"] <= backend["selection_calls"], "invalid pending reorder count"
    assert backend["enqueue_commands"] == summary["completed_blocks"], "backend enqueue byte/command conservation failed"
    assert backend["global_priority_calls"] <= backend["examined_coflow_heads"], "global priority counter mismatch"
    if policy == "strategy2":
        assert backend["global_priority_calls"] == 0, "Strategy 2 consumed joint-coflow priority"
    for example in backend["decision_examples"]:
        assert example["selected_tag"] <= example["minimum_tag"] + epsilon, "logged Path outside native-equivalent set"
        assert 0 <= example["ssu_id"] < summary["num_ssu"] and 0 <= example["selected_path"] < 256, "invalid selected SSD/Path"
    return {"global_grants_verified": True, "backend_equivalence_verified": True,
            "logged_backend_examples_verified": len(backend["decision_examples"])}


def collect_results(data_dir, *, require_complete=False):
    """New namespace only. Old shared-Path cases and old JIT policy aliases fail."""
    selected, input_cache, group_hashes, logical_hashes = {}, {}, defaultdict(set), defaultdict(set)
    for path in sorted(Path(data_dir).glob("npu*.json")):
        saved = json.loads(path.read_text())
        meta, summary, policy = saved["metadata"], saved["summary"], saved["strategy"]
        assert saved.get("experiment_id", meta.get("experiment_id")) == EXPERIMENT_ID, "wrong experiment namespace; old results cannot be relabeled"
        assert policy in POLICIES and meta["regime"] == REGIME, "wrong policy/input regime"
        audit_input_clock(meta)
        assert meta["num_npu"] == summary["num_npu"] == 32, "expected 32 NPUs"
        assert meta["num_ssu"] == summary["num_ssu"] in SSU_COUNTS, "invalid SSU topology"
        assert meta["seed"] == saved["submit_seed"] in SEEDS, "input/submission seed mismatch"
        assert summary["request_count"] == meta["request_count"] == 704, "expected complete 704 requests"
        assert summary["n_layers"] == 8 and summary["batch_size"] == 1, "wrong inference model"
        assert summary["cross_request_layer0_prefetch"] is True, "cross-request prefetch disabled"
        assert all(summary["invariants"].values()), "native invariant failure"
        assert meta["io_bytes"] == IO_BYTES and meta["ssd_bandwidth_gib_s"] == 40 and meta["npu_link_bandwidth_gib_s"] == 50, "hardware/IO units changed"
        metrics = recompute_metrics(summary)
        metrics.update(complete_run_accounting(summary))
        for field in ("start_ms", "end_ms", "mean_npu_utilization", "npu_utilizations",
                      "active_ms_by_npu", "io_stall_ms_by_npu", "idle_ms_by_npu"):
            _same_numeric(metrics[field], saved["common_window"][field], (path.name, field))
        for clock in ("admission", "arrival"):
            slo = saved["slo"]["all_requests"][clock]
            assert slo["count"] == 704, "SLO cohort changed"
            _same_numeric(metrics[clock + "_slo_passed"], slo["passed"], "SLO count mismatch")
            _same_numeric(metrics[clock + "_slo_rate"], slo["rate"], "SLO rate mismatch")
        _same_numeric(saved["slo"]["alpha"], 1.5, "SLO threshold changed")
        telemetry = audit_adapter(saved["adapter_statistics"], summary)
        telemetry.update(audit_coflow_control(saved))
        # Reuse only binding/timestamp checks; do not apply the previous JIT audit.
        assignment_policy = "strategy1" if policy in ("strategy1", "strategy2", "strategy3") else policy
        audit_assignment({**saved, "strategy": assignment_policy}, metrics)
        archived = load_input_facts(saved, data_dir, input_cache)
        facts = archived["facts"]
        fixed_ablation = policy.startswith("strategy") and saved["policy_config"]["assignment"] == "fixed"
        audit_request_identity(facts, metrics, "baseline" if fixed_ablation else policy)
        arrival_meta = meta["arrival_workload"]
        arrival = arrival_accounting(facts, num_ssu=meta["num_ssu"],
                                     target_gib_s=arrival_meta["target_postinitial_gib_s"])
        assert arrival["initial_request_count"] == 128, "initial four-per-card backlog changed"
        assert all(sum(row["arrival_ms"] == 0 and row["original_npu_id"] == npu for row in facts.values()) == 4
                   for npu in range(32)), "initial backlog is not four requests per original card"
        assert math.isclose(arrival["actual_positive_arrival_gib_s"], 218.31565, abs_tol=1e-4), "positive arrival work rate is not the requested 218.31565 GiB/s"
        for actual, cached in (("initial_request_count", "initial_arrived_request_count"),
                               ("initial_work_gib", "initial_arrived_read_gib"),
                               ("positive_request_count", "postinitial_request_count"),
                               ("positive_work_gib", "postinitial_read_gib"),
                               ("last_arrival_ms", "postinitial_last_arrival_ms"),
                               ("actual_positive_arrival_gib_s", "actual_postinitial_gib_s"),
                               ("positive_arrival_gib_s_by_ssu", "postinitial_gib_s_by_ssu")):
            _same_numeric(arrival[actual], arrival_meta[cached], ("arrival metadata mismatch", cached))
        _same_numeric(arrival["ideal_full_compute_demand_gib_s_by_ssu"], meta["input_demand"]["per_ssu_gib_s"], "ideal demand metadata mismatch")
        _same_numeric(summary["completed_blocks"], sum(row["io_count"] for row in facts.values()), "completed block conservation")
        for disk in summary["disk_stats"]:
            _same_numeric(disk["completed_gb"], arrival["total_read_gib_by_ssu"][disk["ssu_id"]], "per-SSD read-byte conservation")
        audit_sources(saved, data_dir)
        key = meta["num_ssu"], meta["seed"], policy
        assert key not in selected, "duplicate experimental case"
        selected[key] = {"saved": saved, "metrics": metrics, "arrival": arrival,
                         "cross_request_witnesses": cross_request_capacity_witnesses(summary, facts),
                         "telemetry": telemetry, "file": path}
        group_hashes[key[:2]].add((archived["native_hash"], saved["input_artifact"]["sha256"],
                                  tuple(saved["static_path_cirs_gib_s"])))
        logical_hashes[key[1]].add(archived["logical_hash"])
    assert all(len(values) == 1 for values in group_hashes.values()), "policies changed input or static CIR"
    assert all(len(values) == 1 for values in logical_hashes.values()), "logical trace differs across SSD topologies"
    new_policy_config = audit_policy_configurations(case["saved"] for case in selected.values())
    expected = {(ssu, seed, policy) for ssu in SSU_COUNTS for seed in SEEDS for policy in POLICIES}
    missing = sorted(expected - set(selected))
    if require_complete:
        assert not missing, ("incomplete 36-case matrix; no final PDF", missing)
    environment = audit_environment(data_dir, selected, required=require_complete)
    audit = {"experiment_id": EXPERIMENT_ID, "case_count": len(selected), "expected_case_count": 36,
             "missing_cases": missing, "complete": not missing,
             "fixed_window_ms": [1000, 2000], "primary_slo": "admission_to_completion <= 1.5*8C",
             "secondary_slo": "arrival_to_completion <= 1.5*8C",
             "input_clock_version": INPUT_CLOCK_VERSION,
             "new_policy_configuration": new_policy_config,
             "environment": environment,
             "whole_run_four_way_accounting_verified": bool(selected),
             "same_logical_input_across_topologies_verified": bool(selected),
             "all_npus_active_every_fixed_window": bool(selected) and all(c["metrics"]["all_npus_active_whole_window"] for c in selected.values()),
             "report_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                      for name in (Path(__file__).name, "build_shared_path_report.py")}}
    return selected, audit


LABELS = {"baseline": "Baseline", "once": "Once/layer", "new_once": "New once",
          "strategy1": "S1", "strategy2": "S2", "strategy3": "S3"}


def control_cost_accounting(saved):
    """Measured Python callback wall time, never added to simulated latency.

    These separately instrumented scopes are not a complete CPU profile.
    Shared ACK timing excludes native completion and new global ACK timing
    starts after that old ACK call; neither is a distributed message measure.
    """
    stats = saved["adapter_statistics"]
    client, backend = stats.get("global_client", {}), stats.get("backend_arbitration", {})
    average = lambda total, calls: total / calls if calls else 0.0
    return {"route_calls": stats["routing_calls"],
            "route_wall_s": stats["routing_wall_us"] / 1e6,
            "mean_route_wall_ms": average(stats["routing_wall_us"], stats["routing_calls"]) / 1000,
            "collector_wall_s": stats["collector_wall_us"] / 1e6,
            "assignment_wall_s": stats["assignment_wall_us"] / 1e6,
            "path_ack_ledger_wall_s": stats["ack_ledger_wall_us"] / 1e6,
            "global_plan_calls": client.get("plans", 0),
            "global_plan_wall_s": client.get("plan_wall_us", 0) / 1e6,
            "mean_global_plan_wall_us": average(client.get("plan_wall_us", 0), client.get("plans", 0)),
            "global_ack_wall_s": client.get("ack_wall_us", 0) / 1e6,
            "backend_selection_calls": backend.get("selection_calls", 0),
            "backend_decision_wall_s": backend.get("decision_wall_us", 0) / 1e6,
            "mean_backend_decision_wall_us": average(backend.get("decision_wall_us", 0), backend.get("selection_calls", 0)),
            "backend_enqueue_wall_s": backend.get("enqueue_wall_us", 0) / 1e6,
            "simulation_makespan_s": saved["summary"]["makespan_ms"] / 1000,
            "experiment_process_wall_s": saved["wall_seconds"]}


def collect_development(folder, selected=()):
    """Audit canonical development artifacts separately, never treat as held-out."""
    rows, fingerprints = [], set()
    for parent in sorted({path.parent for path in Path(folder).rglob("npu32*.json")}):
        cases, _ = collect_results(parent)
        assert len(cases) == 1, "development candidate directory must contain one run"
        (ssu, seed, policy), case = next(iter(cases.items()))
        assert (ssu, seed) == (6, SEEDS[0]), "held-out/topology case was used as development"
        m, saved = case["metrics"], case["saved"]
        fingerprints.add(saved["input_fingerprint"])
        rows.append({"candidate": str(parent.relative_to(folder)), "strategy": policy,
            "policy_config": saved["policy_config"], "input_fingerprint": saved["input_fingerprint"],
            "file": str(case["file"]), "source_sha256": saved["core_and_policy_sha256"],
            "mean_npu_utilization": m["mean_npu_utilization"],
            "admission_slo_passed": m["admission_slo_passed"], "arrival_slo_passed": m["arrival_slo_passed"],
            "makespan_ms": m["makespan_ms"], "full_run_mean_npu_utilization": m["full_run_mean_npu_utilization"],
            "mean_whole_stall_ms": m["mean_whole_stall_ms"], "mean_terminal_tail_ms": m["mean_terminal_tail_ms"]})
    assert len(fingerprints) <= 1, "development candidates changed paired canonical input"
    if selected and fingerprints:
        assert fingerprints == {selected[(6, SEEDS[0], "baseline")]["saved"]["input_fingerprint"]}, "development and formal trace differ"
    return rows


def audit_development_selection(folder, rows, selected=()):
    """Verify the saved choice against measured development objectives.

    Timing/no-held-out fields are provenance declarations; numerical checks
    do not independently prove the human/controller's entire decision history.
    """
    path = Path(folder) / "selection.json"
    record = json.loads(path.read_text())
    assert record["development_seed"] == SEEDS[0] and record["development_ssu"] == 6, "selection used wrong development population"
    assert record["canonical_candidate_count"] == len(rows) == 16, "development selection missing candidates"
    assert record["precanonical_pilot_count_excluded"] == 6, "precanonical exclusion changed"
    assert record["selection_uses_heldout_performance"] is False, "selection declares held-out tuning"
    assert record["client_configuration_frozen_before_formal_client_runs"] is True, "client freeze declaration missing"
    assert record["joint_rule_frozen_before_formal_strategy3_runs"] is True, "joint freeze declaration missing"
    assert record["strategy1_strategy2_joint_field_is_inert"] is True, "inert joint parameter misinterpreted"
    by_file = {str(Path(row["file"]).relative_to(folder)): row for row in rows}
    client = by_file[record["client_selected_case"]]
    joint = by_file[record["joint_selected_case"]]
    assert client["strategy"] == "strategy1" and joint["strategy"] == "strategy3", "selected cases have wrong policy"
    config = record["client_configuration"]
    for row in (client, joint):
        assert {key: row["policy_config"][key] for key in config} == config, "selected client configurations differ"
        assert row["input_fingerprint"] == record["input_fingerprint"], "selection input differs"
    assert joint["policy_config"]["joint_rule"] == record["strategy3_joint_rule"], "selected joint rule differs"
    _same_numeric(client["mean_npu_utilization"], max(row["mean_npu_utilization"] for row in rows if row["strategy"] == "strategy1"), "client choice is not the declared development U maximum")
    eligible_joint = [row for row in rows if row["strategy"] == "strategy3" and
                      all(row["policy_config"][key] == value for key, value in config.items())]
    _same_numeric(joint["mean_npu_utilization"], max(row["mean_npu_utilization"] for row in eligible_joint), "joint choice is not the declared conditional U maximum")
    if selected:
        actual_config = audit_policy_configurations(case["saved"] for case in selected.values())
        assert actual_config == {**config, "joint_rule": record["strategy3_joint_rule"]}, "formal run changed frozen selection"
        for row in (client, joint):
            actual = selected[(6, SEEDS[0], row["strategy"])]["metrics"]
            for field in ("mean_npu_utilization", "admission_slo_passed", "arrival_slo_passed", "makespan_ms"):
                _same_numeric(actual[field], row[field], ("selected development run did not reproduce in formal run", field))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_client_configuration": config, "selected_joint_rule": record["strategy3_joint_rule"],
            "conditional_development_U_maximum_verified": True,
            "selected_development_formal_metrics_reproduced": bool(selected),
            "freeze_history_note": "no-held-out and freeze-order fields are archived provenance declarations, not independently inferable from metric values"}


def make_figure2(selected, figure_dir, *, seed):
    """One seed, three SSD panels, six policy rows, unsorted physical IDs."""
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), constrained_layout=True)
    for axis, ssu in zip(axes, SSU_COUNTS):
        matrix = [[100 * u for u in selected[(ssu, seed, policy)]["metrics"]["npu_utilizations"]]
                  for policy in POLICIES]
        shown = axis.imshow(matrix, vmin=0, vmax=100, cmap="cividis", aspect="auto")
        for row, values in enumerate(matrix):
            for npu, value in enumerate(values):
                axis.text(npu, row, f"{value:.0f}", ha="center", va="center", fontsize=9.5,
                          color="white" if value < 55 else "black")
                if value < 80:
                    axis.add_patch(plt.Rectangle((npu - .48, row - .48), .96, .96,
                                                 fill=False, edgecolor="#E24A33", linewidth=.8))
        axis.set_xticks(range(32), range(32), fontsize=9.5)
        axis.set_yticks(range(6), [LABELS[p] for p in POLICIES], fontsize=9.5)
        axis.set_xticks([i-.5 for i in range(33)], minor=True)
        axis.set_yticks([i-.5 for i in range(7)], minor=True)
        axis.grid(which="minor", color="white", linewidth=.65, alpha=.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_title(f"32 NPU / {ssu} SSU; seed {seed}; [1000, 2000] ms", fontsize=11)
        axis.set_xlabel("Physical NPU ID; not sorted; outline means utilization < 80%", fontsize=9.5)
    colorbar = fig.colorbar(shown, ax=axes, shrink=.8, label="Compute utilization (%)")
    colorbar.ax.tick_params(labelsize=9.5)
    colorbar.ax.yaxis.label.set_size(10)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / f"figure2_per_npu_seed{seed}"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def make_policy_figure(figure_dir):
    """Keep pre-queue control, local device scheduling and ACK separate."""
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, axis = plt.subplots(figsize=(11, 3.3), constrained_layout=True)
    axis.set_xlim(-.15, 11.25)
    axis.set_ylim(-.15, 3.05)
    axis.axis("off")
    def box(x, y, width, height, label, fill):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.035",
                                     facecolor=fill, edgecolor="#303030", linewidth=.8))
        axis.text(x+width/2, y+height/2, label, ha="center", va="center", fontsize=11.5)
    labels = ("Actual arrival\nS1: choose NPU", "Activated layer\nS1: IO grants", "New once\nPath routing",
              "Each SSD: 40 GiB/s\nS2/S3: eligible Path", "One link/NPU\n50 GiB/s -> HBM")
    for i, label in enumerate(labels):
        x = 2.23*i
        box(x, 1.1, 2.02, .86, label, "#DDEAF2" if i<3 else "#F4E3C2")
        if i < 4:
            axis.annotate("", (x+2.21,1.53),(x+2.03,1.53),arrowprops={"arrowstyle":"->","lw":1})
    box(3.0, 2.35, 4.9, .46, "5-ms SSD telemetry -> immutable client copy", "#E9EDF0")
    axis.annotate("", (5.46,1.99),(5.46,2.32),arrowprops={"arrowstyle":"->","lw":.8})
    axis.annotate("", (3.24,.9),(10.0,.9), arrowprops={"arrowstyle":"->","lw":.9})
    axis.text(6.6,.62,"HBM ACK -> client credit + Path ledger + coflow progress",ha="center",fontsize=11.5)
    axis.text(5.5,.18,"S1 changes unissued work. S2/S3 need device cooperation; no active-IO preemption.",ha="center",fontsize=11.5)
    axis.set_title("Control positions and explicit information boundaries", fontsize=12)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "figure1_control_locations.pdf")
    fig.savefig(figure_dir / "figure1_control_locations.png", dpi=200)
    plt.close(fig)


def select_illustrative_witness(selected):
    """Prefer S3 six-SSD case and the cleanest isolated-budget example.

    This is intentionally illustrative, not the worst card or a representative
    sample. Pick minimum unexplained delay above the capacity certificate.
    All witnesses and maximum stalls remain in the audited sidecar.
    """
    keys = [(6, SEEDS[0], "strategy3")]
    keys.extend(key for key in sorted(selected) if key[2] == "strategy3" and key not in keys)
    for key in keys:
        case = selected.get(key)
        if case and case["cross_request_witnesses"]:
            chosen = min(case["cross_request_witnesses"], key=lambda row: (
                row["actual_layer0_stall_ms"] - row["unavoidable_boundary_stall_ms"],
                row["deadline_ms"], row["request_id"]))
            return {"num_ssu": key[0], "seed": key[1], "strategy": key[2], **chosen}
    return None


def make_witness_figure(witness, figure_dir):
    """Activation-to-HBM duration is explicitly not continuous SSD service."""
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / "data/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, d, ready = (witness[name] for name in ("activation_ms", "deadline_ms", "ready_ms"))
    lower_end = a + witness["isolated_read_lower_bound_ms"]
    fig, axis = plt.subplots(figsize=(11, 3.15), constrained_layout=True)
    axis.broken_barh([(a, d - a)], (1.55, .45), facecolors="#E69F00", edgecolors="black")
    axis.broken_barh([(a, ready - a)], (.75, .45), facecolors="#56B4E9", edgecolors="black")
    axis.broken_barh([(d, ready - d)], (-.05, .45), facecolors="#D9D9D9", edgecolors="black", hatch="///")
    for t, y, location, label, align in ((a, 2.0, (.01, .92), "L0 activated", "left"),
                                         (d, 2.0, (.44, .92), "Previous L7 ends", "center"),
                                         (lower_end, -.05, (.32, .1), "Earliest fluid bound", "center"),
                                         (ready, 1.2, (.95, .72), "All data at HBM", "right")):
        axis.vlines(t, -.05, 2.0, color="#444444", linestyle="--", linewidth=.7)
        axis.annotate(f"{label}\n{t:.6f} ms", xy=(t, y), xytext=location,
                      textcoords="axes fraction", ha=align, va="center", fontsize=11,
                      arrowprops={"arrowstyle": "-", "color": "#444444", "lw": .7})
    axis.set_yticks([1.775, .975, .175], ["Previous request L7\ncompute budget",
                       "Next request L0\nactivation to HBM", "Next request L0\nactual NPU stall"], fontsize=11)
    axis.set_xticks([])
    axis.set_ylim(-.85, 2.75)
    axis.set_xlim(a - .03 * (ready - a), ready + .03 * (ready - a))
    axis.set_title(f"S3: NPU {witness['npu_id']}; request {witness['previous_request_id']} -> {witness['request_id']}", fontsize=12)
    axis.set_xlabel("Blue includes issue delay, SSD queue/service and receive link; it is NOT an SSD busy interval.", fontsize=11)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "figure3_cross_request_budget.pdf")
    fig.savefig(figure_dir / "figure3_cross_request_budget.png", dpi=200)
    plt.close(fig)


def build_document(selected, audit, development=()):
    """Write factual tables only after every formal paired run passes audit."""
    assert audit["complete"] and len(selected) == 36, "complete formal matrix required for final document"
    config = audit["new_policy_configuration"]
    first_seed = SEEDS[0]
    primary, heldout, whole, arrival_rows, timing_rows, details = [], [], [], [], [], []
    for (ssu, seed, policy), case in sorted(selected.items(), key=lambda item: (
            item[0][1], item[0][0], POLICIES.index(item[0][2]))):
        m = case["metrics"]
        row = [ssu, LABELS[policy], f"{100*m['mean_npu_utilization']:.2f}",
               f"{m['admission_slo_passed']}/704", f"{m['arrival_slo_passed']}/704", m["active_npu_count"]]
        (primary if seed == first_seed else heldout).append(row)
        whole.append([str(seed)[-2:], ssu, LABELS[policy], f"{m['makespan_ms']:.1f}",
                      f"{100*m['full_run_mean_npu_utilization']:.2f}",
                      f"{m['mean_whole_stall_ms']:.1f}", f"{m['mean_terminal_tail_ms']:.1f}",
                      f"{m['mean_nonterminal_idle_ms']:.1f}"])
        details.append([str(seed)[-2:], ssu, LABELS[policy],
                        f"{m['mean_arrival_latency_ms']:.1f}", f"{m['p99_arrival_latency_ms']:.1f}",
                        f"{m['max_terminal_tail_ms']:.1f}"])
        if seed == first_seed:
            cost = control_cost_accounting(case["saved"])
            timing_rows.append([ssu, LABELS[policy], f"{cost['route_wall_s']:.2f}",
                f"{cost['mean_route_wall_ms']:.3f}", f"{cost['global_plan_wall_s']:.2f}",
                f"{cost['backend_decision_wall_s']:.2f}", f"{cost['simulation_makespan_s']:.3f}"])
        if policy == "baseline":
            a = case["arrival"]
            arrival_rows.append([str(seed)[-2:], ssu, f"{a['initial_work_gib']:.3f}",
                                 f"{a['positive_work_gib']:.3f}", f"{a['last_arrival_ms']:.3f}",
                                 f"{max(a['positive_arrival_gib_s_by_ssu']):.3f}"])
    outcome_rows = []
    for policy in ("strategy1", "strategy2", "strategy3"):
        pairs = [(selected[s, seed, policy]["metrics"], selected[s, seed, "once"]["metrics"])
                 for seed in SEEDS for s in SSU_COUNTS]
        gains = [100 * (a["mean_npu_utilization"] - b["mean_npu_utilization"]) for a, b in pairs]
        wins = sum(gain > 1e-7 for gain in gains)
        slo_wins = sum(a["admission_slo_passed"] > b["admission_slo_passed"] for a, b in pairs)
        faster = sum(a["makespan_ms"] < b["makespan_ms"] - EPS_MS for a, b in pairs)
        outcome_rows.append([LABELS[policy], f"{wins}/6", f"{min(gains):+.2f}…{max(gains):+.2f}",
                             f"{slo_wins}/6", f"{faster}/6"])
    baseline_u = [100 * case["metrics"]["mean_npu_utilization"] for key, case in selected.items() if key[2] == "baseline"]
    mean_u = lambda s, p: 50 * sum(selected[(s, seed, p)]["metrics"]["mean_npu_utilization"] for seed in SEEDS)
    topology_deltas = {s: mean_u(s, "strategy3") - mean_u(s, "once") for s in SSU_COUNTS}
    no_faster_new = all(selected[(s, seed, p)]["metrics"]["makespan_ms"] >
                       selected[(s, seed, "once")]["metrics"]["makespan_ms"]
                       for s in SSU_COUNTS for seed in SEEDS for p in ("strategy1", "strategy2", "strategy3"))
    profiles = selected[(6, first_seed, "baseline")]["saved"]["metadata"]["profiles"]
    profile_rows = [[p["seq_len_k"], p["nql"], p["kv_blocks"], f"{p['compute_ms']:.6f}", p["quota"]]
                    for p in profiles]
    w = select_illustrative_witness(selected)
    witness_text = "没有找到同时满足本节严格条件的 S3 窗口内实例；不以窗口外/冷启动替代证据。"
    if w:
        witness_text = (
            f"实际实例来自 {w['num_ssu']} 盘、种子 {w['seed']}、S3、NPU{w['npu_id']}。"
            f"前请求 {w['previous_request_id']} 是 {w['previous_seq_len_k']}K/NQL{w['previous_nql']}，"
            f"末层计算预算 {w['previous_compute_ms']:.6f} ms；后请求 {w['request_id']} 是 "
            f"{w['seq_len_k']}K/NQL{w['nql']}，自身每层 C={w['own_compute_ms']:.6f} ms，"
            f"但不能用这段尚未开始的计算隐藏它自己的 L0。\n\n"
            f"后请求 L0 共 {w['layer_io_count']} IO（{w['layer_read_gib']:.6f} GiB），"
            f"各盘块数依次为 {w['layer_io_by_ssu']}。激活={w['activation_ms']:.6f} ms，"
            f"预算结束={w['deadline_ms']:.6f} ms，到齐={w['ready_ms']:.6f} ms。"
            f"孤立读取下界 {w['isolated_read_lower_bound_ms']:.6f} ms，"
            f"对应最早到齐时刻的理论下界为 {w['activation_ms']+w['isolated_read_lower_bound_ms']:.6f} ms（不是实际事件），"
            f"所以这次转换至少有 {w['unavoidable_boundary_stall_ms']:.6f} ms 隐藏不了；"
            f"实际 L0 stall={w['actual_layer0_stall_ms']:.6f} ms，"
            f"其中窗口内 {w['window_layer0_stall_ms']:.6f} ms。\n\n"
            "为便于隔离预算原因，此例选实际等待最接近孤立下界的一次，不冒充最差卡或全体请求的典型值。"
            "证书只能说明至少这部分时间隐藏不了，不能把实际等待超过证书的剩余部分全归为固有容量不足。\n\n"
            "![Figure 3：真实跨请求预算。橙色是前请求最后层计算；蓝色是后请求 L0 从激活到 HBM 到齐，包含发射等待、SSD 排队/服务和链路，不是 SSD 持续忙碌线；斜线为后请求实际 NPU stall。虚线标实际事件和明示的理论最早下界，后者不是实际完成事件；不用均匀取样刻度代替事件。](../figures/figure3_cross_request_budget.pdf){width=100%}")
    assignment_text = {
        "compute": "到达选卡取已到达请求剩余计算时间最小的卡；并列依次看未完请求数、原卡和卡号。新请求自己的计算量对候选卡相同，所以在比较中抵消。",
        "fixed": "正式配置保留原输入 NPU 绑定。fixed 是选卡消融，不能把其收益宣称为负载均衡收益。",
        "pipeline": "到达选卡比较已到达的计算、接收链路和各盘均分服务负担：score=max(剩余计算,单卡接收服务,最慢盘估计服务)，加上新请求的全部已知八层工作后取最小。盘共享数量是对该盘有已知未完成工作的 NPU 数，不是未完请求数量，不读未来；该流体均分 score 不是实际完工时间预测。",
    }[config["assignment"]]
    joint_text = ("S3 取最小 slack：deadline−now−剩余服务估计；并列再看剩余量和 deadline。"
                  if config["joint_rule"] == "least_slack" else
                  "S3 与客户端采用相同 urgent-short 键：紧急 coflow 先按剩余服务估计从短到长，未紧急的按 deadline；这不是最小 slack 规则。")
    development_table = "开发记录尚未提供，不能声称完成候选审计。"
    if development:
        short_labels = {"baseline": "Baseline", "once": "Once", "new_once": "New once"}
        dev_rows = []
        for row in development:
            cfg, policy = row["policy_config"], row["strategy"]
            label = short_labels.get(policy, LABELS[policy])
            rule = cfg["joint_rule"].replace("least_slack", "LS").replace("urgent_short", "US") if policy == "strategy3" else "—"
            dev_rows.append([label, cfg["assignment"] if policy.startswith("strategy") else "—",
                str(cfg["queue_window_ms"]) if policy.startswith("strategy") else "—", rule,
                f"{100*row['mean_npu_utilization']:.2f}", row["admission_slo_passed"],
                f"{row['makespan_ms']:.1f}", f"{row['mean_terminal_tail_ms']:.1f}"])
        development_table = markdown_table(["策略", "选卡", "w/ms", "joint", "U/%", "SLO数", "全程/ms", "均尾等/ms"], dev_rows)
    mechanism_table = ""
    if development:
        chain_specs = (("原 Once", "once", None, None), ("加 Path 账本", "new_once", None, None),
                       ("再加全局发射", "strategy1", "fixed", .25),
                       ("再启用选卡", "strategy1", "pipeline", .25),
                       ("信用窗口改 1 ms", "strategy1", "pipeline", 1.0))
        chain_rows, previous = [], None
        for label, policy, assignment, window in chain_specs:
            match = next((r for r in development if r["strategy"] == policy and
                          (assignment is None or (r["policy_config"]["assignment"] == assignment
                                                   and r["policy_config"]["queue_window_ms"] == window))), None)
            if match is None:
                break
            u = 100 * match["mean_npu_utilization"]
            chain_rows.append([label, f"{u:.4f}", "—" if previous is None else f"{u-previous:+.4f}",
                               match["admission_slo_passed"], f"{match['makespan_ms']:.3f}"])
            previous = u
        if len(chain_rows) == len(chain_specs):
            mechanism_table = (markdown_table(["逐步改变", "U/%", "较前项/pp", "接纳SLO数", "全程/ms"], chain_rows) +
                "\n\n这条链使用相同开发输入：New once→fixed 保留原 NPU 绑定；fixed→pipeline 保留 0.25 ms 窗口；"
                "最后只把 pipeline 的信用窗口变为 1 ms。它说明各次受控修改的闭环结果，不是可脱离上下文相加的普遍因果系数；"
                "后续层释放和队列状态随每次修改反馈变化。")
    doc = r'''---
title: "全局 coflow：32 NPU 与 5/6/7 SSU"
subtitle: "同一可变输入、5 ms 遥测、六策略完整对照"
date: "2026-09-07"
documentclass: article
fontsize: 10pt
geometry: a4paper,margin=18mm
CJKmainfont: "Noto Sans CJK JP"
mainfont: "DejaVu Serif"
monofont: "DejaVu Sans Mono"
colorlinks: true
toc: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{pdflscape}
  - \usepackage{caption}
  - \captionsetup{labelformat=empty}
  - \setlength{\emergencystretch}{3em}
---

# 先看结论与适用范围

**@OUTCOME@**

**@PRIMARY_SCOPE@**

@OUTCOME_TABLE@

上表相对 Once，变化范围单位为百分点；全程更快以完整 makespan 判定。能力更多不能直接等同所有输入都更好，也不保证无 stall。

本轮不是旧独立 JIT 的改名：S1 新增全局入队前发射协调，S2/S3 才增加盘内已排队命令的选择权限。完整 36 组均单独运行原生仿真器；不从旧轮搬结果，不按测试种子重新选参数。开发只使用 6 盘、种子 20260906；另一个种子和其他拓扑是泛化检查，不是独立线上分布的保证。

**这不是 218.31565 GiB/s 对三种拓扑都“长期容量可行”的输入。** 后续用户请求携带的八层工作率保持相同，已超过 5 盘总计 200 GiB/s。若此到达率持续，重排不能防止总积压增长；但本次 704 请求是有限输入，仍能排空，不能仅由这个平均值推出固定 [1,2] 秒 U 上限或全部 SLO 必失败。6/7 盘则仍有局部 deadline、跨请求预取预算和调度成本约束。初始 128 请求积压另算，不伪装为后续平均速率的一部分。

主 U：所有 32 卡固定 [1000,2000] ms 的计算时间占比平均。主 SLO：同一完整 704 个请求中 `completion−admission ≤ 1.5×8C`，其中 C 是该请求每层的纯计算时间，8C 是八层纯计算总时间；此 SLO 从接纳起算。用户 SLO 从 arrival 起算，还包括接纳前排队。两种口径不能互换；主 SLO 提高不自动意味着用户端体验同幅提高。

请求参数出自项目 data，但配额、顺序与到达时钟是构造的。它是可复现的变长请求机制实验，不是生产到达分布采样。所有额外控制计算和跨客户端协调延迟均未计入仿真时间，部署收益仍须单独验证。

![Figure 1：控制位置示意。图中主流程是 S1–S3；Baseline/Once 不启用新选卡与全局发射。所有策略都使用同一个周期 collector，只有主机副本可查询；S2/S3 的设备本地已排队信息是另一项明确增加的权限。下方箭头反馈是客户端 HBM ACK，不是额外实时读 SSD。](../figures/figure1_control_locations.pdf){width=100%}

# 主结果：种子 20260906

@PRIMARY@

U 的单位是百分比，接纳/用户 SLO 列都是通过数/704。active 是这一整秒都有已接纳工作（计算或 IO stall）的卡数，不是“恰好在窗口内到达过请求”的卡数。

$$U=\frac{\sum_i |\text{compute}_i\cap[1000,2000]|}{32\times1000\ {\rm ms}}.$$

逐卡还核对 `1000 ms = compute + stall + idle`。若 active<32，低 U 不能全部归因 IO stall；报告保留真实空闲，不移动窗口。S1–S3 可重绑定，所以同一物理卡列未必处理同一请求；请求级对照必须按 request_id 匹配。

```{=latex}
\clearpage
\begin{landscape}
```

![Figure 2A：种子 20260906。三个拓扑面板，每个六策略行，列始终为真实物理 NPU0–31。颜色统一 0–100%，格内四舍五入整数；100 不表示零 stall。红框按未舍入值低于 80% 判断，精确值见 per_npu_accounting.csv。没有逐行排序或跨种子平均卡号。](../figures/figure2_per_npu_seed20260906.pdf){width=255mm}

```{=latex}
\end{landscape}
\clearpage
```

# 泛化检查：种子 20260907

@HELDOUT@

使用相同冻结策略参数；不因这一表出现退化再次调参。开发种子的单点提升不代表稳定提升。全部逐卡精确值、完整请求 SLO、各阶段控制计数另存 CSV/JSON，而不是只留有利案例。

```{=latex}
\clearpage
\begin{landscape}
```

![Figure 2B：种子 20260907，比例尺和真实卡号顺序与 Figure 2A 相同。100 仍为舍入值，不等于零 stall；红框按未舍入值判断，精确 CSV 另存。跨种子输入顺序不同，不先平均同卡值来掩盖差异。](../figures/figure2_per_npu_seed20260907.pdf){width=255mm}

```{=latex}
\end{landscape}
\clearpage
```

# 如何选择参数，以及没选什么

@DEVELOPMENT@

此表只用 6 盘、种子 20260906，同一 canonical 输入。LS=least-slack，US=urgent-short。包括四种信用窗口、固定分卡消融、pipeline 选卡，以及 S3 与选卡/窗口的交互；开发候选是顺序扩展的工程搜索，不是预先注册的穷尽最优解。完整失败项保留在 data/development，修复前 pilot 不混入。

客户端冻结为 @SELECTED_CLIENT@，以用户指定的中间一秒 U 为首要目标；joint-rule 为 @SELECTED_JOINT@。这不是所有指标都最优。尤其 compute-only 选卡可能把短计算、重读取的请求集中到少数卡，增加全程末尾不均衡；必须同时看全程/ms 和均尾等，不能只按最高 U 删掉失败候选。算法可能减小每卡 IO stall，却把尾部等待变长。

@DEVELOPMENT_TRADEOFF@

@MECHANISM_CHAIN@

# 输入究竟是什么

1 IO=1 个 128-token KV Block=176 KiB=180224 byte；每请求 8 层、batch=1。每 SSD 40 GiB/s，每卡只有一条总计 50 GiB/s 的接收链路。SSD→HBM，不经主机 DRAM 缓存。request_id 是请求唯一编号；layer 是 0–7；NPU ID 是 0–31；它们不是同一概念。

本文 coflow 特指“一个请求的一层所需的整组读取”。它按固定 placement 分成多个 SSD 分量，必须所有分量都到 HBM，这一层才能开始计算；只让其中一盘很快并不够。C 表示该请求一层的纯计算毫秒，不包含 IO 等待。每卡独立串行处理自己的请求，但各卡共享 SSD 服务能力；跨请求预取只针对已经真实到达的后继请求。

@PROFILES@

表中长度单位 K token；配额是原输入每卡的请求数，同一批次跨拓扑不改画像。所有实际每层有序 SSD placement 归档；选卡不能改变这些 IO 的盘归属。三个拓扑的逻辑 fingerprint 相同，同一拓扑六策略的完整物理输入 fingerprint 相同。

| 到达量 | 定义 |
|---|---|
| 初始积压 | t=0 已到达 128 请求，原每卡 4 个 |
| 后续携带工作率 | t>0 请求完整八层 GiB / 最后 arrival 秒 |
| SSD 实际吞吐 | 真正已服务的字节 / 观测时长，与上项不同 |
| 满算需求 | 按原分卡 sum V / sum 8C，与 arrival 时钟不同 |

@ARRIVAL@

后续携带工作率六组均为 218.315646777 GiB/s；表的“最热盘”也是后续携带工作率，不是实际 SSD 测量。初始工作只是这些已到达请求未来要读的八层总量，不是 t=0 一次发完。后续每个到达点都按累计工作量/固定速率核验，不仅检验最后平均。

参考到达先规范到 1 ps，再按时间/request_id 排序。修复前浮点排序 pilot 仅保留在 exploratory_precanonical，不进入本报告。相同源码与种子本身不足以证明跨 Python 生成了完全相同的输入，必须检查 manifest。

# 三个新增策略究竟在哪一侧起作用

Baseline 固定每盘 Path0。Once 保留原独立选路纯函数。New once 保留原选路引擎，增加全局客户端 Path 账本。三者均不新选卡、不做新的盘内重排。

所有策略的每盘 256 条共享 Path 采用相同静态 CIR：四类总预算 20/6/8/6 GiB/s；PIR 无限，仍受整盘 40 GiB/s 上限。CIR 是保底而非硬上限，不能说 Baseline 被 Path0 很小的 CIR 锁死。运行期 CIR 写 0，满足写间隔至少 100 ms，但没有测试动态 CIR。原生模型明确不支持有限 PIR，本轮 PIR 均为无限。

## S1：在排进 SSD 以前协调

@ASSIGNMENT@

所有卡共用全局发射器。已激活层提供 deadline、未发射各盘块数、尚未到 HBM 各盘块数。以 $b=176/1048576$ GiB 表示一条 IO，估计

$$h=1000b\max\left(\max_s K_s/40,\ \sum_s K_s/50\right)\quad{\rm ms}.$$

这里的 K 是客户端“未到 HBM”，可能已读完 SSD，故 h 只作优先级启发式，**不是当前 SSD 剩余时间的严格下界**。若 `deadline−now≤h`，该 coflow 被视为紧急，紧急者优先短剩余，其他按 deadline。没有未来请求信息，也不把还没激活的所有后续层塞进当前 deadline。

信用窗口 w=@WINDOW@ ms，单盘最多 $\max(1,\lfloor40w/(1000b)\rfloor)$ 个已发送未 ACK IO；单卡上限将 40 换成 50。一批最多 64 个授权，实际发送消耗信用，HBM 回执归还；每次新层激活都会清空尚未执行的授权尾段并触发重规划，使新紧急 coflow 能参与。授权不是已经入队的 IO，不迁移盘内命令。

盘信用直到 HBM ACK 才归还：已离开 SSD 但滞留接收链路的 IO 仍占盘信用。因此信用限制并不保证 SSD 永远有工作可做，也不是盘真实 pending 数。

三套数必须区分：Path 账本是“已规划−HBM ACK”，信用是“已实际发射−HBM ACK”，coflow 剩余是“已激活层总数−HBM ACK”。前两者不包含未规划未来层；它们都不是 SSD 精确实时队列。

## S2：已入盘后，仅在原生等价 Path 中选

继承 S1，允许每盘在“最小 finish-tag+原生 EPS”候选中改变 Round Robin 选择，并重排选中 Path 的 pending 头。局部优先级用 deadline、该层在本盘的原始块数和入队时间。保持 Path 归属、CIR/PIR、tag 更新和正在服务 IO 不变。每次选择都断言 selected-tag≤min-tag+EPS，没有超出原生等价集合；集合内允许 tag 略高于严格最小值。这不是无限制跨 Path EDF。

## S3：同样的可选范围，用全局 coflow 进度键

@JOINT@

该进度来自客户端已激活工作和 HBM ACK，经共同控制状态供各盘优先级使用，不是读取其他盘即时内部队列。每盘有自己的原生等价候选和忙碌状态，因此同一个共同键不保证所有盘同时服务同一 coflow。S2/S3 需要设备侧接口或固件配合，不能包装为普通 SSD 上仅改客户端就能部署。

S1–S3 到达选卡、发射函数完全相同，但不同历史服务会反馈到后续绑定和层激活。最终差值不能全部归因于某一次跨 Path 换序；同 Path/跨 Path改变次数是机制证据，不是收益的独立因果分解。

# 为什么协调更多也未必消除等待

同请求预取下一层，预算是本请求当前层 C。跨请求 L0 则不同：只有前请求最后层开始时，才激活下一个已到达队首请求，预算是 **C_previous，不是 C_next**。

对于完整尚未读取的一层，孤立读取下界为上一节 h 的公式，但此处 K 必须是真实未读块数。它忽略首块流水启动、盘/链路竞争、QoS和发射开销，故下界小于 C 不证明一定可隐藏。所有本轮画像的孤立下界都小于自身 C，仍不能排除跨请求边界预算不足。

@WITNESS@

这份证书只约束记录中的前后请求与激活时刻。不同选卡、顺序或允许更早多请求预取可能避免该相邻转换；它不是“任何策略 100% 都不可能”的全局证明。

更一般地，对固定当前状态和期限区间 [a,D]，若同盘必须在该区间读完的**真实 SSD 剩余字节**超过 $40(D-a)/1000$ GiB，任何重排都不能全满足；同一 NPU 接收链路用 50。active IO 只能计未完成部分，已在链路的字节不重复计成 SSD 工作。未知 active 剩余可保守省略，而不能按全块充数。无超载证书只是未证伪，不等于已找到可行调度。

## 数学证书和原生仿真到底匹配什么

另外 16 个小型原生测试覆盖 4/32 卡与 5/6/7 盘，验证容量证书与 SSD→HBM 实际到齐时间的方向一致；不是把下界当精确预测器。它们只运行一层，并使用外部指定的读取 deadline；不改变正式八层 SLO。一般 native L0 deadline 是接纳时刻；这些 t=0 接纳的小测试中该值为 0，不能把它当成测试另行指定的外部 deadline。

- 320 条 IO 固定在 SSD0，单盘至少要 1.342773 ms；给 1 ms deadline 时，证书中的请求至少一个实际迟到。其他盘空闲不能读取这批固定 placement。
- 每盘各 100 条发给同一 NPU，每盘本身可在 1 ms 内读完，但 5/6/7 盘合流可能超过这一卡 50 GiB/s 的接收能力；对应链路证书同样出现实际迟到。
- 反例一：只有 1 IO，流体下界约 0.004196 ms；deadline=0.005 ms 没有容量证书，但一条包必须先 SSD 服务再走链路，实际约 0.007553 ms，仍迟到。
- 反例二：200 条普通 IO 排在 1 条短 IO 前，同 Path FIFO 让小请求约 0.846786 ms 才到齐，错过外部 0.012 ms deadline；合法非抢占同 Path 换头后约 0.007553 ms 到齐。两者都没有容量证书，差异来自队序。仅此夹具的 client submit-batch-size=256 用来构造 FIFO 前驱；正式 client submit-batch-size=1，推理 batch-size 始终为 1。

因此有必要容量失败证书能证伪全部按时完成，没证书则仍可能因包流水或队序失约。策略启发式的成功率必须用实际仿真比较，不能声称公式“零误差预测所有 stall”。

# 全程 stall 与尾部不均衡：不能只看中间一秒

@WHOLE@

种子列 06/07 分别是两个完整种子尾号。“均 stall”是每卡全程 IO 等待毫秒；“均尾等”严格为全局 makespan 减该卡最后完工，再取平均；“其他闲”是首次/中途 idle，不偷算成末尾不均衡。

$$T_{\rm makespan}=T_{\rm compute}+T_{\rm stall}+T_{\rm other\ idle}+T_{\rm tail}\quad\text{（逐卡）}.$$

完整 704 请求的计算总量相同，所以全程 U 随 makespan 反向变化；固定 [1,2] 秒 U 可能提高但 makespan 变长。SLO 按请求数计，U 按计算毫秒计，也可能方向不同。每卡四项精确值和最大尾差另存 data/per_npu_accounting.csv。

@DETAILS@

上表用户均值/P99从 arrival 起算；最大尾等表示最早结束卡到全局完工的空隙，不是该卡的 IO stall。

# 控制代价与部署限制

@COST@

成本表为种子 06 的 Python 计时。路由一次是一个 request-layer-SSU 的 Path 规划；全局规划是最多 64 授权的一批；盘决策是一次原生候选集合选择。计时是并行实验机器上的 callback 墙钟，不是硬件 CPU 周期测量。完整两种子的细项，包括 collector、全局 ACK、Path ACK、enqueue 和次数，保存在 control_cost.csv。

**这些时间没有进入上面的仿真时间。** 即使所有压力查询只读 5 ms 副本，也不等于决策免费。若累计路由秒数远大于模拟全程秒数，当前 Python 原型不能自动满足在线预算；需要增量结构、批量化、并行/本地化和实现语言优化后重新计时。计数器计时范围不是完整 profile，不把漏计部分视为 0。

Baseline/Once 的全局 Path ACK 维护仅用于本次守恒审计，不是它们选路所需的部署成本；New once/S1–S3 确实需要全 managed 流量的共享账本、一致性和消息传递。S3 还需要设备使用共同进度。网络延迟为 0、共同状态原子可见是额外理想假设，不从本地 Python ACK 时间推断分布式消息成本很低。

各策略同一个 collector 在 t=0 初始化，之后每 5 ms 每盘一次 fresh 读取，查询只有副本，不按 miss 偷读；CIR 运行写 0。S1–S3 另有全局每 IO 0.1 微秒的发射间隔，参考策略原约束是每 NPU 各自 0.1 微秒；前者更严格，但仍不是现实主机控制延迟模型。

# 函数、归档与复现

- `shared_path_baseline.py`：`baseline_path_ids`，输入 IO 数，输出 Path0 序列。
- `shared_path_once.py`：`once_path_ids`，输入 IO 数、采集副本、允许 Path、QoS，输出原 Once 序列。
- `shared_path_new_once.py`：`new_once_path_ids`，另输入全局已规划未 HBM ACK Path 账本，输出并由调用者原子预留。
- `coflow_client_policy.py`：`choose_npu` / `choose_npu_pipeline` 在真实到达时调用；`plan_global_batch` 输入已激活 ClientRead、各盘/卡实际 outstanding、当前时间，输出有序授权。
- `coflow_disk_policy.py`：`choose_path` 输入本盘原生等价候选与 RR 游标，输出选择；同 Path pending 换头由 adapter 配合。
- `coflow_joint_policy.py`：`coflow_priority` 输入固定 deadline、各盘已激活未 HBM ACK 块数和 now，输出 S3 排序键。
- `coflow_capacity_analysis.py`：`capacity_overloads` 输入真实 remaining SSD/link 分开数与 release/deadline，输出必要容量失败证书，不输出“保证无 stall”。

S1 需要选卡、全局发射、New once 路由和 ACK 维护共同接入；单独调用 choose_npu 不等于运行完整 S1。S2/S3 还需要设备选择 hook。纯函数输出是授权、排序键或必要条件，不是精确未来完工时间。

选卡函数的 backlog/counter 数组必须按物理 NPU ID 0…N−1 排列，返回的数组索引才是 NPU ID。不能把按 U 排过序的列表传进去，再把索引当原物理卡号。各盘向量同样按 SSU ID 0…S−1 排列。

**下列独立模块命令只是最小 demo，保留开发默认，并不自动复现正式配置。** 正式调用需显式指定：

- 选卡：`choose_npu_pipeline`，不是默认的 compute 选卡演示。
- 客户端发射：`queue_window_ms=1.0`。
- S3 纯函数：`coflow_priority` 的参数 `rule="urgent_short"`。
- adapter/runner：选卡参数 `assignment="pipeline"`，窗口参数 `queue_window_ms=1.0`，联合排序参数 `joint_rule="urgent_short"`。

纯函数参数叫 rule；adapter/runner 参数叫 joint_rule。完整正式命令位于 data/formal_results 目录下的 client_commands.json。

```bash
python -B shared_path_baseline.py
python -B shared_path_once.py
python -B shared_path_new_once.py
python -B coflow_client_policy.py
python -B coflow_disk_policy.py
python -B coflow_joint_policy.py
python -B coflow_capacity_analysis.py
python -B -m pytest -q test_coflow_client_policy.py test_coflow_disk_policy.py
python -B -m pytest -q test_coflow_disk_adapter.py test_coflow_sim_adapter.py
python -B -m pytest -q test_coflow_experiment_inputs.py test_coflow_report.py
python -B -m pytest -q test_coflow_capacity_analysis.py
python -B -m pytest -q test_coflow_capacity_simulation.py
```

完整再生成使用 `build_coflow_report.py --data` 指向 `data/formal_results`，加 `--render`。只有 36 组严格审计全部通过才生成最终 PDF。结果各自保存完整输入、实际 ordered placement 指纹和源文件 SHA 归档；不是用当前源码代替当时版本。report_audit.json 核对完成块数、逐层时间、两种 SLO、每盘字节、collector、信用归零和 native tag 权限。

源代码相同、同一机器/解释器仍不免除完整输入核验。正式环境记录在 formal_results/environment.json。输入/源 hash、单测与原生断言证明可复现和检查范围内的正确性，不证明任意真实 SSD 固件、用户分布或低延迟控制实现有相同收益。
'''
    replacements = {
        "@OUTCOME@": f"36 组正式对照已完成。Baseline 在六个拓扑/种子组合中的固定秒 U 为 {min(baseline_u):.2f}%–{max(baseline_u):.2f}%。以下同时报告三个新策略的收益与退化，不只展示最好的一个点。",
        "@PRIMARY_SCOPE@": (f"用户目标目前只部分达到：S3 相对 Once 的两种子平均 U，6 盘改变 {topology_deltas[6]:+.2f} 个百分点，"
            f"5 盘改变 {topology_deltas[5]:+.2f} 个百分点（明显退化），7 盘仅 {topology_deltas[7]:+.2f} 个百分点。"
            + ("三个新策略在每个拓扑/种子组合的完整 makespan 都比 Once 更长，未验证到整批完成吞吐提升。"
               if no_faster_new else "整批 makespan 也须单独比较，不能把窗口 U 当作完成吞吐。")
            + "接纳 SLO 改善与窗口 U/整批时长的取舍必须同时保留，不能称为全面胜出。"),
        "@OUTCOME_TABLE@": markdown_table(["策略", "U提高组数", "U变化/pp", "接纳SLO提高", "全程更快"], outcome_rows),
        "@PRIMARY@": markdown_table(["SSU", "策略", "U/%", "接纳SLO", "用户SLO", "active"], primary),
        "@HELDOUT@": markdown_table(["SSU", "策略", "U/%", "接纳SLO", "用户SLO", "active"], heldout),
        "@PROFILES@": markdown_table(["长度K", "NQL", "IO/层", "C/ms", "原每卡配额"], profile_rows),
        "@ARRIVAL@": markdown_table(["种子", "盘", "初始GiB", "后续GiB", "最后arrival/ms", "最热盘GiB/s"], arrival_rows),
        "@WHOLE@": markdown_table(["种子", "盘", "策略", "全程/ms", "全程U/%", "均stall/ms", "均尾等/ms", "其他闲/ms"], whole),
        "@DETAILS@": markdown_table(["种子", "盘", "策略", "均用户/ms", "P99用户/ms", "最大尾等/ms"], details),
        "@COST@": markdown_table(["盘", "策略", "路由/s", "均路由/ms", "全局规划/s", "盘决策/s", "模拟全程/s"], timing_rows),
        "@WINDOW@": str(config["queue_window_ms"]), "@ASSIGNMENT@": assignment_text,
        "@JOINT@": joint_text, "@WITNESS@": witness_text,
        "@DEVELOPMENT@": development_table,
        "@SELECTED_CLIENT@": f"{config['assignment']}，w={config['queue_window_ms']} ms",
        "@SELECTED_JOINT@": config["joint_rule"],
        "@MECHANISM_CHAIN@": mechanism_table,
    }
    dev_s1 = selected[(6, first_seed, "strategy1")]["metrics"]
    dev_once = selected[(6, first_seed, "once")]["metrics"]
    replacements["@DEVELOPMENT_TRADEOFF@"] = (
        f"正式开发组 S1 相对 Once：固定秒 U 改变 {100*(dev_s1['mean_npu_utilization']-dev_once['mean_npu_utilization']):+.3f} 个百分点，"
        f"但完整 makespan 为 {dev_s1['makespan_ms']:.3f} 对 {dev_once['makespan_ms']:.3f} ms，"
        f"即相对变化 {100*(dev_s1['makespan_ms']/dev_once['makespan_ms']-1):+.3f}%。"
        "正的 makespan 变化表示更慢，不因窗口 U 改善就称整批更快。")
    c0, c1 = dev_once["whole_compute_ms_by_npu"], dev_s1["whole_compute_ms_by_npu"]
    replacements["@DEVELOPMENT_TRADEOFF@"] += (
        f" 原 Once 每卡全程计算范围 {min(c0):.3f}–{max(c0):.3f} ms，"
        f"选卡后为 {min(c1):.3f}–{max(c1):.3f} ms；均尾等从 {dev_once['mean_terminal_tail_ms']:.3f} "
        f"变为 {dev_s1['mean_terminal_tail_ms']:.3f} ms。"
        f"按逐卡四项恒等式，makespan 改变 {dev_s1['makespan_ms']-dev_once['makespan_ms']:+.3f} ms，"
        f"由均 stall 改变 {dev_s1['mean_whole_stall_ms']-dev_once['mean_whole_stall_ms']:+.3f}、"
        f"均尾等改变 {dev_s1['mean_terminal_tail_ms']-dev_once['mean_terminal_tail_ms']:+.3f}、"
        f"其他闲改变 {dev_s1['mean_nonterminal_idle_ms']-dev_once['mean_nonterminal_idle_ms']:+.3f} ms 精确分解（显示数值四舍五入）。"
        "这是时间账目，不把排队与后续调度反馈全部归因为单一因素。")
    joint_by_config = {(row["policy_config"]["assignment"], row["policy_config"]["queue_window_ms"],
                        row["policy_config"]["joint_rule"]): row for row in development if row["strategy"] == "strategy3"}
    if all(("pipeline", w, rule) in joint_by_config for w in (.25, 1.0) for rule in ("least_slack", "urgent_short")):
        parts = []
        for window in (.25, 1.0):
            ls = joint_by_config[("pipeline", window, "least_slack")]
            us = joint_by_config[("pipeline", window, "urgent_short")]
            parts.append(f"pipeline/w={window} ms 时，LS 与 US 的 U 分别为 {100*ls['mean_npu_utilization']:.5f}% / {100*us['mean_npu_utilization']:.5f}%，"
                         f"makespan 分别为 {ls['makespan_ms']:.3f} / {us['makespan_ms']:.3f} ms")
        replacements["@DEVELOPMENT_TRADEOFF@"] += "\n\n" + "；".join(parts) + "。优先级规则与窗口存在交互；一组的排名不能外推为普遍最优。"
    for token, value in replacements.items():
        doc = doc.replace(token, value)
    return doc


def render_report(selected, audit, output, development=()):
    assert audit["complete"], "refuse to publish incomplete formal PDF"
    for folder in ("docs", "data", "figures"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        make_figure2(selected, output / "figures", seed=seed)
    make_policy_figure(output / "figures")
    witness = select_illustrative_witness(selected)
    if witness:
        make_witness_figure(witness, output / "figures")
    source = output / "docs/coflow_global_5ms_report.md"
    source.write_text(build_document(selected, audit, development), encoding="utf-8")
    audit["development_candidates"] = development
    audit["cases"] = [{"num_ssu": key[0], "seed": key[1], "strategy": key[2],
                        "file": str(case["file"]), "input_fingerprint": case["saved"]["input_fingerprint"],
                        "source_sha256": case["saved"]["core_and_policy_sha256"],
                        "metrics": {k: v for k, v in case["metrics"].items() if k != "records"},
                        "arrival": case["arrival"], "telemetry": case["telemetry"],
                        "cross_request_witnesses": case["cross_request_witnesses"]}
                       for key, case in sorted(selected.items())]
    (output / "data/report_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    populations, cards, costs = [], [], []
    for (ssu, seed, policy), case in sorted(selected.items()):
        common = {"num_ssu": ssu, "seed": seed, "strategy": policy}
        populations.extend({**common, **r} for r in case["metrics"]["records"])
        costs.append({**common, **control_cost_accounting(case["saved"])})
        for npu in range(32):
            cards.append({**common, "npu_id": npu,
                          **{name: value[npu] for name, value in case["metrics"].items()
                             if isinstance(value, list) and len(value) == 32}})
    for filename, rows in (("full_population_slo.csv", populations), ("per_npu_accounting.csv", cards),
                            ("control_cost.csv", costs)):
        with (output / "data" / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    destination = output / "coflow_global_5ms_report.pdf"
    subprocess.run(["pandoc", str(source), "--standalone", "--number-sections", "--pdf-engine=xelatex",
                    "--resource-path", str(source.parent), "--output", str(destination)], check=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUT / "data/formal_results")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--render", action="store_true", help="require complete matrix, write audited CSVs/figures/MD/PDF")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--development", type=Path, default=OUT / "data/development")
    args = parser.parse_args()
    selected, audit = collect_results(args.data, require_complete=args.require_complete or args.render)
    if args.render:
        development = collect_development(args.development, selected)
        audit["development_selection"] = audit_development_selection(args.development, development, selected)
        destination = render_report(selected, audit, args.output, development)
        print(json.dumps({"case_count": len(selected), "pdf": str(destination)}, ensure_ascii=False))
    else:
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
