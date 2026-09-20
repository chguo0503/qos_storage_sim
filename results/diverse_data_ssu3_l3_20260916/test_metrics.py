"""Check new metrics against frozen old experiments, including a causal cut.

No simulator runs: reconstruct only the request/layer state visible after the
warm admission cohort has drained and hide all future layer starts/completions.
"""
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from metrics import live_summary, summarize
from inputs.runners.run_baseline_npu32_stress import load_manifest


@lru_cache(maxsize=2)
def frozen(mode):
    directory = ROOT / "results/baseline_ab128_32_ratio12_20260912/validation20s/runs" / f"ssu3_{mode}_k1_sync_seed7/baseline"
    manifests, _ = load_manifest(directory / "manifest.json.gz")
    with gzip.open(directory / "result.json.gz", "rt") as stream:
        result = json.load(stream)
    return manifests, result


@pytest.mark.parametrize("mode,u,passed,count,demand,overload", [
    ("random", 90.67794278896096, 264, 341, 189.79496739979405, 75.88289526353317),
    ("ordered", 71.00894540379132, 160, 224, 318.75035132521884, 34.189512829294245),
])
def test_saved_warm_metrics_match_previous_published_values(mode,u,passed,count,demand,overload):
    manifests, result = frozen(mode)
    row = summarize(result["summary"], manifests, 2000., 4000.)
    assert row["U_percent"] == pytest.approx(u, abs=1e-9)
    assert row["slo"]["passed"] == passed
    assert row["slo"]["count"] == count
    assert row["all_npus_active"]
    assert row["completed_after_window"] > 0  # No right-censoring of SLO.
    assert sum(row["demand"]["per_disk_mean_GiB_s"]) == pytest.approx(demand, abs=1e-8)
    assert row["demand"]["per_disk_overload_percent"] == pytest.approx([overload]*3, abs=1e-9)
    full = summarize(result["summary"], manifests, 0., result["summary"]["makespan_ms"], full=True)
    assert full["U_percent"] == pytest.approx(100*result["summary"]["fleet_npu_compute_utilization"], abs=1e-9)


@pytest.mark.parametrize("mode", ["random", "ordered"])
def test_live_context_cut_matches_final_warm_result(mode):
    manifests, result = frozen(mode)
    summary = result["summary"]
    byid = {q.request_id:q for q in manifests}
    cohort = [r for r in summary["request_metrics"] if 2000 <= r["admission_time_ms"] < 4000]
    cut = max(4100., max(r["completion_time_ms"] for r in cohort) + 0.01)
    live_requests = {}
    for r in summary["request_metrics"]:
        q = byid[r["request_id"]]
        admitted = r["admission_time_ms"] <= cut
        complete = r["completion_time_ms"] <= cut
        live_requests[q.request_id] = SimpleNamespace(
            manifest=q, admitted=admitted, completed=complete,
            admission_time_ms=r["admission_time_ms"] if admitted else math.nan,
            completion_time_ms=r["completion_time_ms"] if complete else math.nan,
            per_layer_compute_ms=q.load["per_layer_us"] / 1000.)
    live_batches = []
    for b in summary["microbatch_metrics"]:
        if b["admission_time_ms"] > cut:
            continue
        layers = [SimpleNamespace(
            compute_start_ms=m["compute_start_ms"] if m["compute_start_ms"] <= cut else math.nan,
            compute_duration_ms=m["compute_duration_ms"] if m["compute_start_ms"] <= cut else 0.)
            for m in b["layer_metrics"]]
        live_batches.append(SimpleNamespace(
            npu_id=b["npu_id"], member_request_ids=b["member_request_ids"],
            admission_time_ms=b["admission_time_ms"],
            completion_time_ms=b["completion_time_ms"] if b["completion_time_ms"] <= cut else math.nan,
            layer_metrics=layers))
    context = SimpleNamespace(requests=live_requests, microbatches=live_batches, n_layers=8)
    projected = summarize(live_summary(context), manifests, 2000., 4000.)
    final = summarize(summary, manifests, 2000., 4000.)
    assert projected["slo"] == final["slo"]
    assert projected["arrival_slo"] == final["arrival_slo"]
    assert projected["slo_by_category"] == final["slo_by_category"]
    assert projected["slo_by_profile"] == final["slo_by_profile"]
    assert projected["cohort_request_ids"] == final["cohort_request_ids"]
    assert projected["completed_after_window"] == final["completed_after_window"]
    assert projected["demand"] == final["demand"]
    assert projected["U_percent"] == pytest.approx(final["U_percent"], abs=1e-9)
    assert projected["per_npu_U_percent"] == pytest.approx(final["per_npu_U_percent"], abs=1e-9)
    assert projected["all_npus_active"]
