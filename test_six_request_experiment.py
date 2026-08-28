from collections import Counter, defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest

import sim
from six_request_experiment import cutoff_metrics
from six_request_workload import (
    REQUESTS_PER_NPU,
    prepare_six_request_workload,
)


def _table():
    return sim.load_bw_table_cache(num_npu=128)


def test_six_request_workload_is_balanced_per_npu_and_sequence():
    workload = prepare_six_request_workload(
        _table(), num_npu=128, n_layers=24, num_ssu=8
    )
    by_npu = defaultdict(list)
    for request in workload.requests:
        by_npu[request.npu_id].append(request)
    assert all(len(requests) == REQUESTS_PER_NPU for requests in by_npu.values())
    assert all(
        sorted(Counter(request.category for request in requests).values())
        == [1, 1, 2, 2]
        for requests in by_npu.values()
    )
    fleet_categories = Counter(request.category for request in workload.requests)
    assert fleet_categories == {category: 192 for category in sim.WORKLOAD_CATEGORIES}
    guard_categories = Counter(
        request.category for request in workload.requests if request.stream_id == 5
    )
    assert guard_categories == {category: 32 for category in sim.WORKLOAD_CATEGORIES}
    measured_categories = Counter(
        request.category for request in workload.requests if request.stream_id < 5
    )
    assert measured_categories == {category: 160 for category in sim.WORKLOAD_CATEGORIES}

    compute_totals = [sum(request.per_layer_us for request in requests) for requests in by_npu.values()]
    kv_totals = [sum(request.per_layer_kv_gb for request in requests) for requests in by_npu.values()]
    assert max(compute_totals) / min(compute_totals) - 1.0 < 0.001
    assert max(kv_totals) / min(kv_totals) - 1.0 < 0.001

    other_ssu = prepare_six_request_workload(
        _table(), num_npu=128, n_layers=80, num_ssu=16
    )
    assert workload.statistics["assignment_hash"] == other_ssu.statistics[
        "assignment_hash"
    ]
    assert workload.placement_hash != other_ssu.placement_hash


def _synthetic_input():
    requests = []
    request_rows = []
    batches = []
    completions = (
        (10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        (12.0, 22.0, 32.0, 42.0, 50.0, 70.0),
    )
    admissions = (
        (0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
        (2.0, 12.0, 22.0, 32.0, 42.0, 50.0),
    )
    intervals = (
        ((0.0, 5.0), (10.0, 15.0), (20.0, 25.0), (30.0, 35.0), (40.0, 45.0), (50.0, 60.0)),
        ((2.0, 7.0), (12.0, 17.0), (22.0, 27.0), (32.0, 37.0), (42.0, 47.0), (58.0, 63.0)),
    )
    batch_id = 0
    for npu_id in range(2):
        for sequence in range(6):
            request_id = 100 + npu_id * 20 + sequence
            requests.append(
                SimpleNamespace(
                    request_id=request_id,
                    npu_id=npu_id,
                    stream_id=sequence,
                    per_layer_us=1000.0,
                    profile_key=(64, 128),
                    category="SS",
                )
            )
            admission = admissions[npu_id][sequence]
            completion = completions[npu_id][sequence]
            request_rows.append(
                {
                    "request_id": request_id,
                    "npu_id": npu_id,
                    "batch_id": batch_id,
                    "admission_time_ms": admission,
                    "completion_time_ms": completion,
                    "processing_latency_ms": completion - admission,
                    "batch_compute_ms": intervals[npu_id][sequence][1]
                    - intervals[npu_id][sequence][0],
                    "batch_io_barrier_wait_ms": 0.0,
                    "avg_ssd_queue_wait_ms": 0.0,
                    "avg_npu_link_queue_wait_ms": 0.0,
                }
            )
            start, end = intervals[npu_id][sequence]
            batches.append(
                {
                    "batch_id": batch_id,
                    "npu_id": npu_id,
                    "layer_metrics": [
                        {
                            "compute_start_ms": start,
                            "compute_end_ms": end,
                            "io_barrier_wait_ms": 0.0,
                        }
                    ],
                }
            )
            batch_id += 1
    workload = SimpleNamespace(requests=tuple(requests), num_npu=2, n_layers=1)
    summary = {"request_metrics": request_rows, "microbatch_metrics": batches}
    return workload, summary


def test_cutoff_metrics_clip_compute_and_include_sixth_request():
    workload, summary = _synthetic_input()
    metrics = cutoff_metrics(workload, summary, alphas=(2.0,))
    assert metrics["cutoff_ms"] == 60.0
    assert metrics["cutoff_npu_ids"] == [0]
    assert metrics["slo_cohort_complete"]
    assert metrics["slo_cohort_completed_at_cutoff"] == 10
    assert metrics["slo_guard_margin_ms"] == 10.0
    assert metrics["npu_rows"][0]["compute_ms"] == 35.0
    assert metrics["npu_rows"][1]["compute_ms"] == 27.0
    assert metrics["npu_rows"][1]["completed_requests_at_cutoff"] == 5
    assert metrics["npu_rows"][0]["utilization"] == pytest.approx(35.0 / 60.0)
    assert metrics["npu_rows"][1]["utilization"] == pytest.approx(27.0 / 58.0)


def test_incomplete_first_five_invalidates_ttft_slo():
    workload, summary = _synthetic_input()
    broken = deepcopy(summary)
    target = next(
        row for row in broken["request_metrics"] if row["request_id"] == 124
    )
    target["completion_time_ms"] = 61.0
    target["processing_latency_ms"] = 19.0
    metrics = cutoff_metrics(workload, broken, alphas=(2.0,))
    assert not metrics["slo_cohort_complete"]
    assert metrics["slo_cohort_completed_at_cutoff"] == 9
    assert metrics["slo_guard_margin_ms"] == -1.0
    assert metrics["slo"]["2"]["attainment"] is None
