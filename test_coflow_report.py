"""Independent arrival and full-run accounting checks; no large simulation."""

from copy import deepcopy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from build_coflow_report import (
    IO_GIB, arrival_accounting, audit_canonical_reference_order, audit_coflow_control,
    audit_development_selection, audit_environment, audit_input_clock, audit_policy_configurations,
    build_document, control_cost_accounting,
    audit_request_identity, collect_results,
    complete_run_accounting, cross_request_capacity_witnesses, manifest_facts,
    select_illustrative_witness,
)


def gapped_summary():
    batches = []
    for npu, start, first_stall in ((0, 2, 2), (0, 15, 0), (1, 0, 0)):
        now, layers = start, []
        for layer in range(8):
            stall = first_stall if layer == 0 else 0
            layers.append({"compute_start_ms": now + stall, "compute_end_ms": now + stall + 1,
                           "io_barrier_wait_ms": stall})
            now += stall + 1
        batches.append({"npu_id": npu, "admission_time_ms": start,
                        "completion_time_ms": now, "layer_metrics": layers})
    return {"num_npu": 2, "makespan_ms": 23, "microbatch_metrics": batches}


def small_requests():
    requests = []
    cumulative = 0
    for rid, npu, block_disks in ((0, 0, [0]), (1, 1, [1]), (2, 0, [0, 1]), (3, 1, [0, 1, 1])):
        if rid >= 2:
            cumulative += 8 * len(block_disks)
        requests.append({"request_id": rid, "npu_id": npu,
                         "arrival_time_ms": 1000 * cumulative * IO_GIB / 10,
                         "load": {"category": "SS", "per_layer_us": 1000.0},
                         "placement": [[[disk, IO_GIB] for disk in block_disks]]})
    return requests


class CompleteRunTests(unittest.TestCase):
    def test_stall_is_not_tail_imbalance(self):
        result = complete_run_accounting(gapped_summary())
        self.assertEqual(result["whole_compute_ms_by_npu"], [16, 8])
        self.assertEqual(result["whole_stall_ms_by_npu"], [2, 0])
        self.assertEqual(result["nonterminal_idle_ms_by_npu"], [5, 0])
        self.assertEqual(result["terminal_tail_ms_by_npu"], [0, 15])

    def test_mean_four_parts_equal_makespan(self):
        result = complete_run_accounting(gapped_summary())
        self.assertEqual(sum(result[key] for key in (
            "mean_whole_compute_ms", "mean_whole_stall_ms",
            "mean_nonterminal_idle_ms", "mean_terminal_tail_ms")), 23)

    def test_startup_and_interior_idle_are_not_called_terminal_tail(self):
        result = complete_run_accounting(gapped_summary())
        self.assertEqual(result["mean_nonterminal_idle_ms"], 2.5)
        self.assertEqual(result["mean_terminal_tail_ms"], 7.5)

    def test_unused_card_has_explicit_tail_convention(self):
        summary = gapped_summary()
        summary["num_npu"] = 3
        result = complete_run_accounting(summary)
        self.assertEqual(result["terminal_tail_ms_by_npu"][2], 23)
        self.assertEqual(result["nonterminal_idle_ms_by_npu"][2], 0)

    def test_bad_stall_accounting_rejected(self):
        summary = gapped_summary()
        summary["microbatch_metrics"][0]["layer_metrics"][0]["io_barrier_wait_ms"] = 1
        with self.assertRaisesRegex(AssertionError, "decomposition"):
            complete_run_accounting(summary)


class ManifestTests(unittest.TestCase):
    def test_repeated_layer_is_counted_eight_times(self):
        facts = manifest_facts(small_requests(), num_ssu=2, num_npu=2)
        self.assertEqual(facts[2]["io_count"], 16)
        self.assertEqual(facts[2]["io_by_ssu"], (8, 8))
        self.assertEqual(facts[2]["ideal_compute_ms"], 8)

    def test_explicit_eight_layers_are_not_multiplied_again(self):
        requests = small_requests()
        requests[2]["placement"] *= 8
        facts = manifest_facts(requests, num_ssu=2, num_npu=2)
        self.assertEqual(facts[2]["io_count"], 16)

    def test_wrong_block_size_rejected(self):
        requests = small_requests()
        requests[0]["placement"][0][0][1] *= 2
        with self.assertRaisesRegex(AssertionError, "176 KiB"):
            manifest_facts(requests, num_ssu=2, num_npu=2)

    def test_invalid_ssu_rejected(self):
        requests = small_requests()
        requests[0]["placement"][0][0][0] = 2
        with self.assertRaisesRegex(AssertionError, "SSD"):
            manifest_facts(requests, num_ssu=2, num_npu=2)

    def test_partial_layer_manifest_rejected(self):
        requests = small_requests()
        requests[0]["placement"] *= 2
        with self.assertRaisesRegex(AssertionError, "incomplete"):
            manifest_facts(requests, num_ssu=2, num_npu=2)

    def test_duplicate_request_rejected(self):
        requests = small_requests()
        requests.append(deepcopy(requests[0]))
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            manifest_facts(requests, num_ssu=2, num_npu=2)


class ArrivalTests(unittest.TestCase):
    def setUp(self):
        self.facts = manifest_facts(small_requests(), num_ssu=2, num_npu=2)

    def test_initial_work_is_not_divided_into_positive_arrival_rate(self):
        result = arrival_accounting(self.facts, num_ssu=2, num_npu=2, target_gib_s=10)
        self.assertEqual(result["initial_request_count"], 2)
        self.assertEqual(result["initial_work_gib"], 16 * IO_GIB)
        self.assertAlmostEqual(result["actual_positive_arrival_gib_s"], 10)
        self.assertEqual(result["positive_work_gib"], 40 * IO_GIB)

    def test_per_ssu_rates_sum_to_aggregate(self):
        result = arrival_accounting(self.facts, num_ssu=2, num_npu=2)
        self.assertAlmostEqual(sum(result["positive_arrival_gib_s_by_ssu"]), 10)
        self.assertAlmostEqual(result["positive_arrival_gib_s_by_ssu"][0], 4)
        self.assertAlmostEqual(result["positive_arrival_gib_s_by_ssu"][1], 6)

    def test_ideal_compute_demand_is_a_different_rate(self):
        result = arrival_accounting(self.facts, num_ssu=2, num_npu=2)
        self.assertNotAlmostEqual(result["ideal_full_compute_demand_gib_s"],
                                  result["actual_positive_arrival_gib_s"])

    def test_prefix_schedule_not_only_final_average_is_checked(self):
        self.facts[2]["arrival_ms"] /= 2
        with self.assertRaisesRegex(AssertionError, "prefix"):
            arrival_accounting(self.facts, num_ssu=2, num_npu=2, target_gib_s=10)

    def test_wrong_total_rate_rejected(self):
        with self.assertRaisesRegex(AssertionError, "rate"):
            arrival_accounting(self.facts, num_ssu=2, num_npu=2, target_gib_s=11)

    def test_right_closed_arrival_buckets(self):
        self.facts[2]["arrival_ms"] = 1000
        self.facts[3]["arrival_ms"] = 1000.1
        result = arrival_accounting(self.facts, num_ssu=2, num_npu=2)
        self.assertEqual(result["positive_arrival_work_gib_by_right_closed_second"],
                         {0: 16 * IO_GIB, 1: 24 * IO_GIB})

    def test_global_underload_can_still_hide_one_ssd_overload(self):
        for rid in (2, 3):
            self.facts[rid]["arrival_ms"] /= 7
        result = arrival_accounting(self.facts, num_ssu=2, num_npu=2)
        self.assertAlmostEqual(result["actual_positive_arrival_gib_s"], 70)
        self.assertFalse(result["aggregate_arrival_exceeds_ssd_capacity"])
        self.assertTrue(result["any_ssd_arrival_exceeds_capacity"])

    def test_no_positive_arrivals_have_no_defined_rate(self):
        for row in self.facts.values():
            row["arrival_ms"] = 0
        with self.assertRaisesRegex(AssertionError, "no positive"):
            arrival_accounting(self.facts, num_ssu=2, num_npu=2)


class NamespaceTests(unittest.TestCase):
    def test_final_document_cannot_be_populated_from_incomplete_cases(self):
        with self.assertRaisesRegex(AssertionError, "complete formal"):
            build_document({}, {"complete": False})

    def test_reference_values_not_just_metadata_are_canonical(self):
        requests = small_requests()
        for i, row in enumerate(requests):
            row["load"]["reference_arrival_ms"] = i + .1
        audit_canonical_reference_order(requests)
        requests[0]["load"]["reference_arrival_ms"] = .1000000000001
        with self.assertRaisesRegex(AssertionError, "not canonical"):
            audit_canonical_reference_order(requests)

    def test_reference_order_is_preserved_by_byte_clock(self):
        requests = small_requests()
        for i, row in enumerate(requests):
            row["load"]["reference_arrival_ms"] = i
        requests[2]["load"]["reference_arrival_ms"] = 4
        with self.assertRaisesRegex(AssertionError, "reference order"):
            audit_canonical_reference_order(requests)

    def test_precanonical_pilot_clock_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "pre-canonical"):
            audit_input_clock({"arrival_workload": {}})

    def test_canonical_clock_has_explicit_quantum(self):
        audit_input_clock({"arrival_workload": {
            "input_clock_version": "canonical_reference_ps_v2", "reference_arrival_quantum_ms": 1e-9}})
        with self.assertRaisesRegex(AssertionError, "quantum"):
            audit_input_clock({"arrival_workload": {
                "input_clock_version": "canonical_reference_ps_v2", "reference_arrival_quantum_ms": 1e-8}})

    def test_old_round_result_cannot_be_used_as_new_strategy(self):
        with tempfile.TemporaryDirectory(prefix="coflow-audit-") as folder:
            (Path(folder) / "npu32_old.json").write_text(json.dumps({
                "strategy": "strategy1", "metadata": {"experiment_id": "old_shared_path"}, "summary": {}}))
            with self.assertRaisesRegex(AssertionError, "namespace"):
                collect_results(folder)

    def test_incomplete_matrix_cannot_pass_final_gate(self):
        with tempfile.TemporaryDirectory(prefix="coflow-audit-") as folder:
            with self.assertRaisesRegex(AssertionError, "36-case"):
                collect_results(folder, require_complete=True)

    def test_empty_audit_reports_missing_cases_not_zero_metrics(self):
        with tempfile.TemporaryDirectory(prefix="coflow-audit-") as folder:
            selected, audit = collect_results(folder)
            self.assertEqual(selected, {})
            self.assertFalse(audit["complete"])
            self.assertFalse(audit["whole_run_four_way_accounting_verified"])
            self.assertEqual(len(audit["missing_cases"]), 36)


class DocumentTemplateTests(unittest.TestCase):
    def test_complete_template_interpolates_without_publishing_fixture_results(self):
        # Formatting fixture only: never saved as a simulation/audit artifact.
        metrics = {"mean_npu_utilization": .9, "admission_slo_passed": 600,
            "arrival_slo_passed": 30, "active_npu_count": 32, "makespan_ms": 5000,
            "full_run_mean_npu_utilization": .8, "mean_whole_stall_ms": 200,
            "mean_terminal_tail_ms": 700, "mean_nonterminal_idle_ms": 100,
            "mean_arrival_latency_ms": 300, "p99_arrival_latency_ms": 600,
            "max_terminal_tail_ms": 900, "whole_compute_ms_by_npu": [4000] * 32}
        arrival = {"initial_work_gib": 100, "positive_work_gib": 700,
                   "last_arrival_ms": 3500, "positive_arrival_gib_s_by_ssu": [30] * 7}
        saved = {"summary": {"makespan_ms": 5000}, "wall_seconds": 10,
                 "adapter_statistics": {"routing_calls": 1000, "routing_wall_us": 2000000,
                    "collector_wall_us": 1000, "assignment_wall_us": 1000, "ack_ledger_wall_us": 1000},
                 "metadata": {"profiles": [{"seq_len_k": 32, "nql": 128, "kv_blocks": 255,
                                               "compute_ms": 1.178, "quota": 22}]}}
        cases = {(s, seed, policy): {"metrics": metrics, "arrival": arrival, "saved": saved,
                                      "cross_request_witnesses": []}
                 for s in (5, 6, 7) for seed in (20260906, 20260907)
                 for policy in ("baseline", "once", "new_once", "strategy1", "strategy2", "strategy3")}
        audit = {"complete": True, "new_policy_configuration": {
            "queue_window_ms": 1, "assignment": "pipeline", "joint_rule": "urgent_short"}}
        document = build_document(cases, audit)
        self.assertNotRegex(document, r"@[A-Z_]+@")
        self.assertIn("Figure 2A", document)
        self.assertIn("Figure 2B", document)
        self.assertIn("C_previous", document)
        self.assertIn("没有找到同时满足本节严格条件", document)
        self.assertIn("原生模型明确不支持有限 PIR", document)


class EnvironmentAuditTests(unittest.TestCase):
    def environment(self):
        sources = {"sim.py": "frozen-source"}
        return {"experiment_id": "coflow_global_5ms_v1", "input_clock_version": "canonical_reference_ps_v2",
            "simulated_hardware": {"num_npu": 32, "ssu_counts": [5, 6, 7], "ssd_bandwidth_gib_s": 40,
                "npu_link_bandwidth_gib_s": 50, "io_bytes": 176 * 1024, "layers": 8, "collector_ms": 5},
            "source_files_sha256": sources, "python_version": "fixture", "platform": "fixture",
            "source_tree_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    def test_missing_record_cannot_pass_formal_gate(self):
        with tempfile.TemporaryDirectory(prefix="coflow-env-") as folder:
            self.assertIsNone(audit_environment(folder, {}))
            with self.assertRaisesRegex(AssertionError, "missing"):
                audit_environment(folder, {}, required=True)

    def test_environment_sources_match_each_case_not_only_core(self):
        with tempfile.TemporaryDirectory(prefix="coflow-env-") as folder:
            env = self.environment()
            (Path(folder) / "environment.json").write_text(json.dumps(env))
            selected = {"case": {"saved": {"core_and_policy_sha256": dict(env["source_files_sha256"])}}}
            self.assertEqual(audit_environment(folder, selected)["python_version"], "fixture")
            selected["case"]["saved"]["core_and_policy_sha256"]["sim.py"] = "different"
            with self.assertRaisesRegex(AssertionError, "frozen environment"):
                audit_environment(folder, selected)

    def test_environment_tree_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory(prefix="coflow-env-") as folder:
            env = self.environment()
            env["source_tree_sha256"] = "wrong"
            (Path(folder) / "environment.json").write_text(json.dumps(env))
            with self.assertRaisesRegex(AssertionError, "tree fingerprint"):
                audit_environment(folder, {})


class ControlCostTests(unittest.TestCase):
    def test_microseconds_converted_without_claiming_simulated_latency(self):
        saved = {"wall_seconds": 9, "summary": {"makespan_ms": 5000},
            "adapter_statistics": {"routing_calls": 1000, "routing_wall_us": 2300000,
                "collector_wall_us": 5000, "assignment_wall_us": 1000, "ack_ledger_wall_us": 2000,
                "global_client": {"plans": 500, "plan_wall_us": 10000, "ack_wall_us": 4000},
                "backend_arbitration": {"selection_calls": 1000, "decision_wall_us": 80000, "enqueue_wall_us": 20000}}}
        cost = control_cost_accounting(saved)
        self.assertEqual(cost["mean_route_wall_ms"], 2.3)
        self.assertEqual(cost["mean_global_plan_wall_us"], 20)
        self.assertEqual(cost["mean_backend_decision_wall_us"], 80)
        self.assertEqual(cost["simulation_makespan_s"], 5)
        self.assertEqual(cost["experiment_process_wall_s"], 9)


class ConfigurationComparisonTests(unittest.TestCase):
    def cases(self):
        return [{"strategy": p, "policy_config": {"queue_window_ms": 1, "assignment": "pipeline",
                                                   "joint_rule": "least_slack"}}
                for p in ("strategy1", "strategy2", "strategy3", "strategy3")]

    def test_inert_joint_argument_does_not_force_s3_choice(self):
        cases = self.cases()
        for row in cases[2:]:
            row["policy_config"]["joint_rule"] = "urgent_short"
        config = audit_policy_configurations(cases)
        self.assertEqual(config, {"queue_window_ms": 1, "assignment": "pipeline", "joint_rule": "urgent_short"})

    def test_s3_cannot_tune_joint_per_topology_or_seed(self):
        cases = self.cases()
        cases[3]["policy_config"]["joint_rule"] = "urgent_short"
        with self.assertRaisesRegex(AssertionError, "joint rules"):
            audit_policy_configurations(cases)

    def test_client_config_must_stay_same_for_s1_s2_s3(self):
        cases = self.cases()
        cases[1]["policy_config"]["queue_window_ms"] = .25
        with self.assertRaisesRegex(AssertionError, "client configurations"):
            audit_policy_configurations(cases)


class SelectionReceiptTests(unittest.TestCase):
    def make_receipt(self, folder):
        rows = []
        for index in range(16):
            is_client = index < 8
            selected = index in (0, 8)
            rows.append({"strategy": "strategy1" if is_client else "strategy3",
                "file": str(Path(folder) / f"case{index}" / "npu32.json"),
                "policy_config": {"queue_window_ms": 1 if selected else .25, "assignment": "pipeline",
                                  "joint_rule": "urgent_short" if index == 8 else "least_slack"},
                "input_fingerprint": "paired", "mean_npu_utilization": .99 if selected else .9})
        record = {"development_seed": 20260906, "development_ssu": 6,
            "canonical_candidate_count": 16, "precanonical_pilot_count_excluded": 6,
            "selection_uses_heldout_performance": False,
            "client_configuration_frozen_before_formal_client_runs": True,
            "joint_rule_frozen_before_formal_strategy3_runs": True,
            "strategy1_strategy2_joint_field_is_inert": True,
            "client_selected_case": "case0/npu32.json", "joint_selected_case": "case8/npu32.json",
            "client_configuration": {"queue_window_ms": 1, "assignment": "pipeline"},
            "strategy3_joint_rule": "urgent_short", "input_fingerprint": "paired"}
        (Path(folder) / "selection.json").write_text(json.dumps(record))
        return rows

    def test_numeric_choice_is_checked_not_only_declared_frozen(self):
        with tempfile.TemporaryDirectory(prefix="coflow-selection-") as folder:
            rows = self.make_receipt(folder)
            result = audit_development_selection(folder, rows)
            self.assertTrue(result["conditional_development_U_maximum_verified"])
            self.assertFalse(result["selected_development_formal_metrics_reproduced"])
            rows[0]["mean_npu_utilization"] = .8
            with self.assertRaisesRegex(AssertionError, "U maximum"):
                audit_development_selection(folder, rows)

    def test_development_candidates_cannot_be_silently_dropped(self):
        with tempfile.TemporaryDirectory(prefix="coflow-selection-") as folder:
            rows = self.make_receipt(folder)
            with self.assertRaisesRegex(AssertionError, "missing candidates"):
                audit_development_selection(folder, rows[:-1])

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.facts = manifest_facts(small_requests(), num_ssu=2, num_npu=2)
        self.metrics = {"request_count": 4, "records": [
            {**row, "npu_id": row["original_npu_id"]} for row in self.facts.values()]}

    def test_strategy3_may_change_execution_card(self):
        self.metrics["records"][0]["npu_id"] = 1
        audit_request_identity(self.facts, self.metrics, "strategy3")
        with self.assertRaisesRegex(AssertionError, "binding"):
            audit_request_identity(self.facts, self.metrics, "new_once")

    def test_new_policy_still_cannot_change_arrival(self):
        self.metrics["records"][0]["arrival_ms"] += 1
        with self.assertRaisesRegex(AssertionError, "arrival"):
            audit_request_identity(self.facts, self.metrics, "strategy3")

    def test_new_policy_still_cannot_change_io_count(self):
        self.metrics["records"][0]["io_count"] += 1
        with self.assertRaisesRegex(AssertionError, "I/O"):
            audit_request_identity(self.facts, self.metrics, "strategy2")


class CrossRequestWitnessTests(unittest.TestCase):
    def setUp(self):
        self.facts = {
            0: {"arrival_ms": 0, "ideal_compute_ms": 8, "seq_len_k": 32, "nql": 128,
                "layer_io_by_ssu": ((10, 10),) * 8},
            1: {"arrival_ms": 0, "ideal_compute_ms": 80, "seq_len_k": 192, "nql": 512,
                "layer_io_by_ssu": ((500, 500),) * 8},
        }
        self.summary = {"request_metrics": [{"request_id": 0, "layer0_cross_request_prefetched": False},
                                             {"request_id": 1, "layer0_cross_request_prefetched": True}],
            "microbatch_metrics": [
                {"npu_id": 0, "member_request_ids": [0], "admission_time_ms": 999,
                 "completion_time_ms": 1007, "layer_metrics": [{"compute_start_ms": 1006}]},
                {"npu_id": 0, "member_request_ids": [1], "admission_time_ms": 1007,
                 "layer_metrics": [{"io_start_time_ms": 1006, "io_ready_time_ms": 1010,
                                    "compute_start_ms": 1010, "io_barrier_wait_ms": 3}]}]}

    def test_uses_previous_compute_not_next_long_compute(self):
        witness = cross_request_capacity_witnesses(self.summary, self.facts)[0]
        self.assertEqual(witness["previous_compute_ms"], 1)
        self.assertEqual(witness["own_compute_ms"], 10)
        self.assertGreater(witness["unavoidable_boundary_stall_ms"], 2)
        self.assertLess(witness["isolated_read_lower_bound_ms"], 10)
        self.assertEqual(witness["actual_layer0_stall_ms"], 3)

    def test_cold_request_does_not_become_cross_request_witness(self):
        self.summary["request_metrics"][1]["layer0_cross_request_prefetched"] = False
        self.assertEqual(cross_request_capacity_witnesses(self.summary, self.facts), [])

    def test_outside_window_not_used_to_explain_fixed_window(self):
        self.assertEqual(cross_request_capacity_witnesses(self.summary, self.facts, start_ms=1011), [])

    def test_late_activation_not_called_a_full_previous_compute_budget(self):
        self.summary["microbatch_metrics"][1]["layer_metrics"][0]["io_start_time_ms"] = 1006.5
        self.assertEqual(cross_request_capacity_witnesses(self.summary, self.facts), [])

    def test_underload_is_not_a_no_stall_certificate(self):
        self.facts[1]["layer_io_by_ssu"] = ((10, 10),) * 8
        self.assertEqual(cross_request_capacity_witnesses(self.summary, self.facts), [])

    def test_partial_window_stall_is_clipped(self):
        witness = cross_request_capacity_witnesses(self.summary, self.facts, end_ms=1008)[0]
        self.assertEqual(witness["window_layer0_stall_ms"], 1)
        self.assertEqual(witness["actual_layer0_stall_ms"], 3)

    def test_illustration_is_clean_budget_example_not_claimed_worst_card(self):
        clean = cross_request_capacity_witnesses(self.summary, self.facts)[0]
        queued = {**clean, "request_id": 7, "actual_layer0_stall_ms": 50}
        chosen = select_illustrative_witness({(6, 20260906, "strategy3"):
            {"cross_request_witnesses": [queued, clean]}})
        self.assertEqual(chosen["request_id"], 1)


class ControlAuditTests(unittest.TestCase):
    def setUp(self):
        self.saved = {"strategy": "strategy2",
            "policy_config": {"queue_window_ms": .25, "assignment": "compute", "joint_rule": "least_slack"},
            "summary": {"num_ssu": 2, "num_npu": 2, "request_count": 1,
                        "submitted_blocks": 16, "completed_blocks": 16},
            "adapter_statistics": {"jit_prefetch": {"enabled": False},
                "global_client": {"queue_window_ms": .25, "grant_batch_size": 64,
                    "assignment": "compute", "issued_commands": 16, "activation_count": 8,
                    "delayed_activation_count": 4, "max_sent_unacked_by_ssu": [59, 50],
                    "max_sent_unacked_by_npu": [74, 40],
                    "final_sent_unacked_by_ssu": [0, 0], "final_sent_unacked_by_npu": [0, 0],
                    "queued_io_migration": False, "future_requests_visible": False,
                    "grant_execution_interval_us": .1, "first_issue_examples": [
                        {"activation_ms": 1, "first_issue_ms": 2, "delay_ms": 1}]},
                "global_coflow": {"enabled": False, "priority_rule": "least_slack", "remaining_coflows_at_end": 0,
                    "instant_other_ssu_queue_access": False},
                "backend_arbitration": {"mode": "strategy2", "cir_pir_unchanged": True,
                    "finite_pir_supported": False, "native_tag_epsilon": 1e-12,
                    "max_selected_tag_gap": 0, "candidate_set_size_histogram": {"1": 1, "2": 2},
                    "selection_calls": 3, "multi_path_opportunities": 2,
                    "cross_path_changed_vs_native_rr": 1, "same_path_pending_head_changed": 2,
                    "enqueue_commands": 16, "global_priority_calls": 0, "examined_coflow_heads": 5,
                    "decision_examples": [{"minimum_tag": 1, "selected_tag": 1,
                        "ssu_id": 0, "selected_path": 32}]}}}

    def test_complete_global_and_native_equivalence_contract(self):
        result = audit_coflow_control(self.saved)
        self.assertTrue(result["global_grants_verified"])
        self.assertTrue(result["backend_equivalence_verified"])

    def test_credit_exceedance_detected(self):
        self.saved["adapter_statistics"]["global_client"]["max_sent_unacked_by_ssu"][0] = 60
        with self.assertRaisesRegex(AssertionError, "credit"):
            audit_coflow_control(self.saved)

    def test_chosen_credit_window_not_hardcoded_to_pilot(self):
        for window in (.1, .5, 1.0):
            self.saved["policy_config"]["queue_window_ms"] = window
            client = self.saved["adapter_statistics"]["global_client"]
            client["queue_window_ms"] = window
            client["max_sent_unacked_by_ssu"] = [int(40 * window / (1000 * IO_GIB))] * 2
            client["max_sent_unacked_by_npu"] = [int(50 * window / (1000 * IO_GIB))] * 2
            self.assertTrue(audit_coflow_control(self.saved)["global_grants_verified"])

    def test_declared_window_must_match_adapter(self):
        self.saved["policy_config"]["queue_window_ms"] = .5
        with self.assertRaisesRegex(AssertionError, "configuration mismatch"):
            audit_coflow_control(self.saved)

    def test_selected_joint_rule_must_match_adapter(self):
        self.saved["policy_config"]["joint_rule"] = "urgent_short"
        with self.assertRaisesRegex(AssertionError, "actual joint priority"):
            audit_coflow_control(self.saved)
        self.saved["adapter_statistics"]["global_coflow"]["priority_rule"] = "urgent_short"
        self.assertTrue(audit_coflow_control(self.saved)["global_grants_verified"])

    def test_undrained_credit_detected(self):
        self.saved["adapter_statistics"]["global_client"]["final_sent_unacked_by_npu"][0] = 1
        with self.assertRaisesRegex(AssertionError, "drain"):
            audit_coflow_control(self.saved)

    def test_old_jit_cannot_be_mixed(self):
        self.saved["adapter_statistics"]["jit_prefetch"]["enabled"] = True
        with self.assertRaisesRegex(AssertionError, "old independent JIT"):
            audit_coflow_control(self.saved)

    def test_no_cross_path_choice_without_alternatives(self):
        self.saved["adapter_statistics"]["backend_arbitration"]["cross_path_changed_vs_native_rr"] = 3
        with self.assertRaisesRegex(AssertionError, "alternatives"):
            audit_coflow_control(self.saved)

    def test_logged_higher_tag_cannot_be_selected(self):
        self.saved["adapter_statistics"]["backend_arbitration"]["decision_examples"][0]["selected_tag"] = 1.01
        with self.assertRaisesRegex(AssertionError, "outside native"):
            audit_coflow_control(self.saved)

    def test_static_cir_alone_does_not_prove_eligibility(self):
        self.saved["adapter_statistics"]["backend_arbitration"]["max_selected_tag_gap"] = .01
        with self.assertRaisesRegex(AssertionError, "tolerance"):
            audit_coflow_control(self.saved)

    def test_strategy2_may_not_use_joint_priority(self):
        self.saved["adapter_statistics"]["backend_arbitration"]["global_priority_calls"] = 1
        with self.assertRaisesRegex(AssertionError, "Strategy 2"):
            audit_coflow_control(self.saved)

    def test_strategy3_uses_same_contract_plus_client_joint_progress(self):
        self.saved["strategy"] = "strategy3"
        stats = self.saved["adapter_statistics"]
        stats["global_coflow"]["enabled"] = True
        stats["backend_arbitration"].update(mode="strategy3", global_priority_calls=2)
        self.assertTrue(audit_coflow_control(self.saved)["backend_equivalence_verified"])

    def test_strategy3_may_not_read_other_ssd_live_queue(self):
        self.saved["strategy"] = "strategy3"
        self.saved["adapter_statistics"]["global_coflow"].update(
            enabled=True, instant_other_ssu_queue_access=True)
        with self.assertRaisesRegex(AssertionError, "remote-SSD"):
            audit_coflow_control(self.saved)


if __name__ == "__main__":
    unittest.main()
