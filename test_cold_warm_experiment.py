from copy import deepcopy

import cold_warm_experiment as experiment
from analyze_cold_warm import analyze
from cold_warm_metrics import cold_warm_metrics
from test_cold_warm_metrics import _synthetic_trace


def test_modified_strategy_matrix_and_metadata():
    spec = experiment.experiment_spec()
    assert [case.name for case in experiment.CASES] == [
        "modified_baseline",
        "modified_scheme_b",
    ]
    assert experiment.SSU_LIST == (40, 56, 70)
    assert experiment.LAYER_LIST == (16, 24, 56, 80)
    assert spec["num_npu"] == 128
    assert spec["requests_per_npu"] == 6
    assert spec["batch_size"] == 1
    assert spec["primary_slo_alpha"] == 2.0
    assert "external arrival queue wait excluded" in spec["ttft_definition"]
    assert "all six requests to completion" in spec["survivorship_policy"]


def test_both_modified_strategies_enable_cross_request_layer0_prefetch(monkeypatch):
    calls = []

    def fake_simulate(requests, **kwargs):
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(experiment, "simulate_continuous_batch", fake_simulate)
    for case in experiment.CASES:
        assert experiment._simulate(case, (), num_ssu=40, n_layers=16) == {
            "ok": True
        }
    assert len(calls) == 2
    assert all(call["cross_request_layer0_prefetch"] is True for call in calls)
    assert all(call["batch_size"] == 1 for call in calls)


def test_analysis_uses_paired_complete_cold_and_warm_cohorts():
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
        rows.append(row)
    payload = {
        "complete": False,
        "experiment": {
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
        "result_rows": 2,
    }
    assert result["metrics"]["modified_baseline"]["16"]["40"]["cold"][
        "ttft_slo"
    ]["2"]["total"] == 12
    assert result["metrics"]["modified_baseline"]["16"]["40"]["warm"][
        "ttft_slo"
    ]["2"]["total"] == 10
    assert result["modified_scheme_b_minus_modified_baseline"]["16"]["40"][
        "warm"
    ]["npu_utilization_delta_pp"] == 0.0
