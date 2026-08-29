from copy import deepcopy

import pytest

import cold_warm_experiment as experiment
from analyze_cold_warm import analyze, merge_compatible_payloads
from cold_warm_metrics import cold_warm_metrics
from test_cold_warm_metrics import _synthetic_trace


def test_modified_strategy_matrix_and_metadata():
    spec = experiment.experiment_spec()
    assert [case.name for case in experiment.CASES] == [
        "modified_baseline",
        "modified_layer_once",
        "modified_refresh8",
        "modified_scheme_b",
        "modified_best_feasible",
    ]
    assert experiment.SSU_LIST == (16, 28, 40, 56, 70)
    assert experiment.LAYER_LIST == (16, 24, 56, 80)
    assert spec["num_npu"] == 128
    assert spec["requests_per_npu"] == 6
    assert spec["batch_size"] == 1
    assert spec["primary_slo_alpha"] == 2.0
    assert "external arrival queue wait excluded" in spec["ttft_definition"]
    assert "all six requests to completion" in spec["survivorship_policy"]


def test_all_modified_strategies_enable_cross_request_layer0_prefetch(monkeypatch):
    calls = []

    def fake_simulate(requests, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(experiment, "simulate_continuous_batch", fake_simulate)
    for case in experiment.CASES:
        assert experiment._simulate(case, (), num_ssu=40, n_layers=16) == {
            "ok": True
        }
    assert len(calls) == 5
    assert all(call["cross_request_layer0_prefetch"] is True for call in calls)
    assert all(call["batch_size"] == 1 for call in calls)
    routing_calls = [call for call in calls if "qos_config" in call]
    assert [call["client_io_config"].pressure_window_io for call in routing_calls] == [
        None,
        None,
        8,
    ]
    assert [
        call["client_io_config"].path_selection_mode for call in routing_calls
    ] == [
        "fixed_path_zero",
        "pressure_aware",
        "pressure_aware",
    ]
    assert len({call["qos_config"] for call in routing_calls}) == 1
    assert all("causal_control" not in call for call in routing_calls)
    full_info_calls = [
        call
        for call in calls
        if call.get("policy") == "per_ssd_full_visible_edf"
    ]
    assert len(full_info_calls) == 1
    assert "qos_config" not in full_info_calls[0]
    assert full_info_calls[0]["client_io_config"].submit_batch_size == 1
    assert full_info_calls[0]["client_io_config"].issue_interval_us == 0.1
    assert (
        full_info_calls[0]["oracle_priority_key"]
        is experiment.best_feasible_priority_key
    )


def test_analysis_uses_paired_complete_cold_and_warm_cohorts(tmp_path):
    workload, summary = _synthetic_trace()
    metrics = cold_warm_metrics(workload, summary)
    shared = {
        "num_ssu": 40,
        "n_layers": 16,
        "wall_time_s": 1.0,
        "assignment_hash": "assignment",
        "workload_hash": "workload",
        "placement_hash": "placement",
        "trace_hash": "trace",
        "simulator_input_fingerprint": "input",
        "representative_profiles": [[64, 128]],
        "fleet_category_counts": {"SS": 12},
        **metrics,
    }
    rows = []
    for case in experiment.CASES:
        row = deepcopy(shared)
        row["strategy"] = case.name
        row["kind"] = case.kind
        if case.name == "modified_refresh8":
            warm = row["cohorts"]["warm"]
            for npu in warm["npu_rows"]:
                npu["utilization"] += 0.1
            warm["mean_npu_utilization"] += 0.1
            warm["slo"]["2"]["pass"] += 1
            warm["slo"]["2"]["fail"] -= 1
            warm["slo"]["2"]["attainment"] += 0.1
        if case.name == "modified_layer_once":
            cold = row["cohorts"]["cold"]
            for npu in cold["npu_rows"]:
                npu["utilization"] += 0.05
            cold["mean_npu_utilization"] += 0.05
        rows.append(row)
    payload = {
        "schema_version": 3,
        "complete": False,
        "experiment": {
            "schema_version": 3,
            "source_fingerprint": "single-source",
            "strategies": [case.name for case in experiment.CASES],
            "num_npu": 2,
            "requests_per_npu": 6,
            "cold_definition": "seq0 admission to seq5 completion",
            "warm_definition": "seq1 admission to seq5 completion",
            "ttft_definition": "completion minus admission",
            "npu_utilization_definition": "per-NPU busy/window then mean",
        },
        "results": rows,
    }

    result = analyze(payload)
    assert result["validation"] == {
        "paired_inputs": True,
        "fixed_complete_cohorts": True,
        "survivorship_bias_excluded": True,
        "external_arrival_queue_wait_excluded_from_ttft": True,
        "result_rows": 5,
    }
    assert result["metrics"]["modified_baseline"]["16"]["40"]["cold"][
        "ttft_slo"
    ]["2"]["total"] == 12
    assert result["metrics"]["modified_baseline"]["16"]["40"]["warm"][
        "ttft_slo"
    ]["2"]["total"] == 10
    assert result["metrics"]["modified_baseline"]["16"]["40"]["warm"][
        "exposed_io_stall"
    ] == {
        "request_count": 10,
        "mean_per_request_ms": 3.0,
        "mean_total_per_npu_ms": 15.0,
    }
    assert result["modified_scheme_b_minus_modified_baseline"]["16"]["40"][
        "warm"
    ]["npu_utilization_delta_pp"] == 0.0
    assert result["modified_refresh8_minus_modified_baseline"]["16"]["40"][
        "warm"
    ]["npu_utilization_delta_pp"] == pytest.approx(10.0)
    assert result["modified_refresh8_minus_modified_baseline"]["16"]["40"][
        "warm"
    ]["shared_window_npu_utilization_delta_pp"] == pytest.approx(0.0)
    assert result["modified_refresh8_minus_modified_baseline"]["16"]["40"][
        "warm"
    ]["ttft_slo_delta_pp"] == pytest.approx(10.0)
    assert result["modified_layer_once_minus_modified_baseline"]["16"]["40"][
        "cold"
    ]["npu_utilization_delta_pp"] == pytest.approx(5.0)
    assert result["modified_layer_once_minus_modified_baseline"]["16"]["40"][
        "cold"
    ]["shared_window_npu_utilization_delta_pp"] == pytest.approx(0.0)

    base = deepcopy(payload)
    base["schema_version"] = 2
    base["experiment"]["schema_version"] = 2
    base["experiment"]["source_fingerprint"] = "base-source"
    base["experiment"]["strategies"] = [
        "modified_baseline",
        "modified_refresh8",
        "modified_scheme_b",
    ]
    base["results"] = [
        row
        for row in base["results"]
        if row["strategy"]
        in {
            "modified_baseline",
            "modified_refresh8",
            "modified_scheme_b",
        }
    ]
    layer_once = deepcopy(payload)
    layer_once["experiment"]["source_fingerprint"] = "layer-source"
    layer_once["results"] = [
        row
        for row in layer_once["results"]
        if row["strategy"] == "modified_layer_once"
    ]
    best_feasible = deepcopy(payload)
    best_feasible["experiment"]["source_fingerprint"] = "oracle-source"
    best_feasible["results"] = [
        row
        for row in best_feasible["results"]
        if row["strategy"] == "modified_best_feasible"
    ]
    base_path = tmp_path / "base.json"
    layer_path = tmp_path / "layer.json"
    oracle_path = tmp_path / "oracle.json"
    base_path.write_text("base")
    layer_path.write_text("layer")
    oracle_path.write_text("oracle")
    second_ssu = deepcopy(payload)
    second_ssu["experiment"]["source_fingerprint"] = "sweep-source"
    for row in second_ssu["results"]:
        row["num_ssu"] = 28
    second_ssu_path = tmp_path / "second_ssu.json"
    second_ssu_path.write_text("second-ssu")
    merged = merge_compatible_payloads(
        (
            (base_path, base),
            (layer_path, layer_once),
            (oracle_path, best_feasible),
            (second_ssu_path, second_ssu),
        ),
        manual_compatibility_audit_note="synthetic fixtures manually matched",
    )
    assert merged["comparison_provenance"]["single_source"] is False
    assert merged["experiment"]["layer_list"] == [16]
    assert merged["experiment"]["ssu_list"] == [28, 40]
    assert merged["experiment"]["source_fingerprints_by_strategy"] == {
        "modified_baseline": ["base-source", "sweep-source"],
        "modified_refresh8": ["base-source", "sweep-source"],
        "modified_scheme_b": ["base-source", "sweep-source"],
        "modified_best_feasible": ["oracle-source", "sweep-source"],
        "modified_layer_once": ["layer-source", "sweep-source"],
    }
    assert merged["comparison_provenance"]["row_source"][
        "modified_baseline|layers=16|ssu=28"
    ].endswith("second_ssu.json")
    compatibility = merged["comparison_provenance"]["compatibility"]
    assert compatibility["behavior_compatibility_verified_by_analyzer"] is False
    assert (
        compatibility["manual_compatibility_audit_note"]
        == "synthetic fixtures manually matched"
    )
    assert analyze(merged)["validation"]["result_rows"] == 10
