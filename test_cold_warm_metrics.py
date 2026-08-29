from types import SimpleNamespace

import pytest

from cold_warm_metrics import cold_warm_metrics


def _synthetic_trace():
    requests = []
    request_metrics = []
    microbatch_metrics = []
    durations = ((2.0, 3.0, 4.0, 5.0, 6.0, 7.0), (3.0,) * 6)
    starts = (0.0, 10.0)
    batch_id = 0
    for npu_id in range(2):
        admission = starts[npu_id]
        for sequence, duration_ms in enumerate(durations[npu_id]):
            request_id = 1000 + 20 * npu_id + sequence
            requests.append(
                SimpleNamespace(
                    request_id=request_id,
                    npu_id=npu_id,
                    stream_id=sequence,
                    profile_key=(64, 128),
                    category="SS",
                    per_layer_us=1000.0,
                )
            )
            compute_start_ms = admission + 1.0
            compute_end_ms = compute_start_ms + 1.0
            completion_ms = admission + duration_ms
            request_metrics.append(
                {
                    "request_id": request_id,
                    "npu_id": npu_id,
                    "batch_id": batch_id,
                    "admission_time_ms": admission,
                    "completion_time_ms": completion_ms,
                    "processing_latency_ms": duration_ms,
                    "admission_wait_ms": 100.0 + sequence,
                }
            )
            microbatch_metrics.append(
                {
                    "batch_id": batch_id,
                    "npu_id": npu_id,
                    "batch_size": 1,
                    "member_request_ids": [request_id],
                    "layer_metrics": [
                        {
                            "compute_start_ms": compute_start_ms,
                            "compute_end_ms": compute_end_ms,
                        }
                    ],
                }
            )
            admission = completion_ms
            batch_id += 1
    workload = SimpleNamespace(
        requests=tuple(requests),
        num_npu=2,
        n_layers=1,
    )
    summary = {
        "request_metrics": request_metrics,
        "microbatch_metrics": microbatch_metrics,
        "invariants": {"all_requests_completed": True},
    }
    return workload, summary


def test_cold_and_warm_use_independent_per_npu_full_trace_windows():
    workload, summary = _synthetic_trace()
    metrics = cold_warm_metrics(workload, summary, alphas=(2.0, 4.0))

    cold = metrics["cohorts"]["cold"]
    warm = metrics["cohorts"]["warm"]
    assert cold["ttft_request_count"] == 12
    assert warm["ttft_request_count"] == 10
    assert cold["npu_rows"][0]["window_ms"] == 27.0
    assert warm["npu_rows"][0]["window_ms"] == 25.0
    assert cold["npu_rows"][0]["compute_ms"] == 6.0
    assert warm["npu_rows"][0]["compute_ms"] == 5.0
    assert cold["npu_rows"][1]["window_ms"] == 18.0
    assert warm["npu_rows"][1]["window_ms"] == 15.0
    assert cold["mean_npu_utilization"] == pytest.approx(
        ((6.0 / 27.0) + (6.0 / 18.0)) / 2.0
    )
    assert warm["mean_npu_utilization"] == pytest.approx(
        ((5.0 / 25.0) + (5.0 / 15.0)) / 2.0
    )


def test_ttft_uses_processing_window_and_not_external_arrival_queue_wait():
    workload, summary = _synthetic_trace()
    metrics = cold_warm_metrics(workload, summary, alphas=(2.0, 4.0))

    first = metrics["request_rows"][0]
    assert first["ttft_ms"] == 2.0
    assert first["external_arrival_queue_wait_ms"] == 100.0
    assert metrics["cohorts"]["cold"]["slo"]["2"]["attainment"] == pytest.approx(
        1.0 / 12.0
    )
    assert metrics["cohorts"]["warm"]["slo"]["4"]["attainment"] == pytest.approx(
        7.0 / 10.0
    )
    assert metrics["first_request_only"]["request_count"] == 2
    assert metrics["first_request_only"]["mean_ttft_ms"] == 2.5


def test_workload_identity_not_request_id_arithmetic_selects_sequences():
    workload, summary = _synthetic_trace()
    metrics = cold_warm_metrics(workload, summary, alphas=(2.0,))
    warm_ids = {
        row["request_id"]
        for row in metrics["request_rows"]
        if row["sequence"] in metrics["cohorts"]["warm"]["request_sequences"]
    }
    assert warm_ids == {
        1001,
        1002,
        1003,
        1004,
        1005,
        1021,
        1022,
        1023,
        1024,
        1025,
    }
