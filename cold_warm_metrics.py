"""Cold/warm metrics for one completed six-request batch-1 trace."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics


DEFAULT_SLO_ALPHAS = (1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 4.0)
REQUESTS_PER_NPU = 6
FINAL_SEQUENCE = REQUESTS_PER_NPU - 1
COHORTS = {
    "cold": tuple(range(REQUESTS_PER_NPU)),
    "warm": tuple(range(1, REQUESTS_PER_NPU)),
}
_EPS_MS = 1e-9


def _alpha_key(alpha):
    return f"{alpha:g}"


def _overlap_ms(start_ms, end_ms, window_start_ms, window_end_ms):
    return max(
        0.0,
        min(end_ms, window_end_ms) - max(start_ms, window_start_ms),
    )


def _slo_metrics(request_rows, npu_ids, alphas):
    by_npu = defaultdict(list)
    for row in request_rows:
        by_npu[row["npu_id"]].append(row)

    result = {}
    for alpha in alphas:
        key = _alpha_key(alpha)
        per_npu = []
        passed = 0
        for npu_id in npu_ids:
            rows = by_npu[npu_id]
            outcomes = [
                row["ttft_ms"] <= alpha * row["ideal_ttft_ms"] + _EPS_MS
                for row in rows
            ]
            passed += sum(outcomes)
            per_npu.append(statistics.fmean(outcomes))
        total = len(request_rows)
        result[key] = {
            "pass": passed,
            "fail": total - passed,
            "total": total,
            "attainment": statistics.fmean(per_npu),
            "equal_weight_per_npu": True,
        }
    return result


def cold_warm_metrics(workload, summary, *, alphas=DEFAULT_SLO_ALPHAS):
    """Measure cold and warm views from the same completed batch-1 trace.

    Cold spans request 0 admission through request 5 completion and includes
    all six requests.  Warm spans request 1 admission through request 5
    completion and includes requests 1--5.  Utilization is computed per NPU
    before taking an equal-weight fleet mean.
    """
    if not summary["invariants"]["all_requests_completed"]:
        raise ValueError("cold/warm metrics require all requests to complete")

    specs_by_key = {
        (request.npu_id, request.stream_id): request
        for request in workload.requests
    }
    npu_ids = tuple(range(workload.num_npu))
    expected_keys = {
        (npu_id, sequence)
        for npu_id in npu_ids
        for sequence in range(REQUESTS_PER_NPU)
    }
    if set(specs_by_key) != expected_keys:
        raise ValueError("trace must contain sequences 0--5 exactly once per NPU")

    raw_requests = {
        row["request_id"]: row for row in summary["request_metrics"]
    }
    if set(raw_requests) != {
        request.request_id for request in workload.requests
    }:
        raise ValueError("summary request rows do not match the fixed workload")

    batches_by_id = {
        batch["batch_id"]: batch for batch in summary["microbatch_metrics"]
    }
    request_rows = []
    intervals_by_npu_sequence = {}
    for key in sorted(specs_by_key):
        request = specs_by_key[key]
        raw = raw_requests[request.request_id]
        if not all(
            math.isfinite(raw[field])
            for field in ("admission_time_ms", "completion_time_ms")
        ):
            raise ValueError("cold/warm metrics require observed completion times")
        batch = batches_by_id[raw["batch_id"]]
        if batch["npu_id"] != request.npu_id or batch["batch_size"] != 1:
            raise ValueError("cold/warm metrics require batch_size=1")
        intervals = tuple(
            (layer["compute_start_ms"], layer["compute_end_ms"])
            for layer in batch["layer_metrics"]
        )
        intervals_by_npu_sequence[key] = intervals
        ideal_ms = workload.n_layers * request.per_layer_us / 1000.0
        ttft_ms = raw["completion_time_ms"] - raw["admission_time_ms"]
        request_rows.append(
            {
                "request_id": request.request_id,
                "npu_id": request.npu_id,
                "sequence": request.stream_id,
                "profile_key": list(request.profile_key),
                "category": request.category,
                "admission_time_ms": raw["admission_time_ms"],
                "completion_time_ms": raw["completion_time_ms"],
                "ttft_ms": ttft_ms,
                "ideal_ttft_ms": ideal_ms,
                "slowdown": ttft_ms / ideal_ms,
                "external_arrival_queue_wait_ms": raw["admission_wait_ms"],
            }
        )

    request_row_by_key = {
        (row["npu_id"], row["sequence"]): row for row in request_rows
    }
    cohort_metrics = {}
    for cohort_name, sequences in COHORTS.items():
        first_sequence = sequences[0]
        npu_rows = []
        for npu_id in npu_ids:
            window_start_ms = request_row_by_key[
                (npu_id, first_sequence)
            ]["admission_time_ms"]
            window_end_ms = request_row_by_key[
                (npu_id, FINAL_SEQUENCE)
            ]["completion_time_ms"]
            compute_ms = sum(
                _overlap_ms(
                    compute_start_ms,
                    compute_end_ms,
                    window_start_ms,
                    window_end_ms,
                )
                for sequence in sequences
                for compute_start_ms, compute_end_ms in intervals_by_npu_sequence[
                    (npu_id, sequence)
                ]
            )
            window_ms = window_end_ms - window_start_ms
            npu_rows.append(
                {
                    "npu_id": npu_id,
                    "window_start_ms": window_start_ms,
                    "window_end_ms": window_end_ms,
                    "window_ms": window_ms,
                    "compute_ms": compute_ms,
                    "utilization": compute_ms / window_ms,
                }
            )

        cohort_requests = [
            row for row in request_rows if row["sequence"] in sequences
        ]
        cohort_metrics[cohort_name] = {
            "request_sequences": list(sequences),
            "utilization_window": (
                f"sequence {first_sequence} admission to sequence "
                f"{FINAL_SEQUENCE} completion, independently per NPU"
            ),
            "ttft_request_count": len(cohort_requests),
            "mean_ttft_ms": statistics.fmean(
                row["ttft_ms"] for row in cohort_requests
            ),
            "mean_npu_utilization": statistics.fmean(
                row["utilization"] for row in npu_rows
            ),
            "aggregate_window_utilization": sum(
                row["compute_ms"] for row in npu_rows
            )
            / sum(row["window_ms"] for row in npu_rows),
            "slo": _slo_metrics(cohort_requests, npu_ids, alphas),
            "npu_rows": npu_rows,
        }

    first_requests = [row for row in request_rows if row["sequence"] == 0]
    return {
        "metric_scope": (
            "one completed trace; per-NPU windows; TTFT is completion minus "
            "admission and excludes external arrival queue wait"
        ),
        "request_rows": request_rows,
        "cohorts": cohort_metrics,
        "first_request_only": {
            "request_count": len(first_requests),
            "mean_ttft_ms": statistics.fmean(
                row["ttft_ms"] for row in first_requests
            ),
            "slo": _slo_metrics(first_requests, npu_ids, alphas),
        },
        "all_requests_completed": True,
        "survivorship_bias_excluded": True,
    }
