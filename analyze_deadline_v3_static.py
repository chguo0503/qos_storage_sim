"""Reproduce the real sequence-0 allocation counterexample for V3."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

import sim
from continuous_batch_sim import (
    CIRControlSnapshot,
    ControlRequestView,
    requests_from_continuous_prefill_workload,
)
from deadline_barrier_scheme_b_v3 import (
    DeadlineBarrierSchemeBControllerV3,
    allocate_deadline_barrier_grants_v3,
)
from scheme_b_prefill import PATH_COUNT, cold_start_hybrid_path_id
from slo_admission_scheme_b import SLOAdmissionSchemeBController
from steady_state_workload import prepare_steady_state_workload


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "steady_state_32npu_deadline_v3"
OUTPUT = OUTPUT_DIR / "static_snapshot.json"
REPORT = OUTPUT_DIR / "static_snapshot.md"
NUM_NPU = 32
NUM_SSU = 10
N_LAYERS = 16
SEED = 42
CATEGORIES = ("SS", "SL", "LS", "LL")


def _real_snapshot():
    table = sim.load_bw_table_cache(num_npu=NUM_NPU)
    workload = prepare_steady_state_workload(
        table,
        num_npu=NUM_NPU,
        num_ssu=NUM_SSU,
        n_layers=N_LAYERS,
        requests_per_npu=4,
        seed=SEED,
    )
    requests = requests_from_continuous_prefill_workload(workload)
    views = []
    for request in requests:
        if request.request_id >= NUM_NPU:
            continue
        work = [0.0] * NUM_SSU
        for ssu_id, size_gb in request.placement[0]:
            work[ssu_id] += size_gb
        compute_ms = float(request.load["per_layer_us"]) / 1000.0
        views.append(
            ControlRequestView(
                request_id=request.request_id,
                npu_id=request.npu_id,
                category=str(request.load["category"]),
                per_layer_compute_ms=compute_ms,
                compute_done_up_to=-1,
                remaining_layers=N_LAYERS,
                next_layer_work_gb_by_ssu=tuple(work),
                waiting_for_io=True,
                remaining_work_gb_by_ssu=tuple(
                    N_LAYERS * amount for amount in work
                ),
                remaining_compute_budget_ms=N_LAYERS * compute_ms,
                prefetch_only=False,
            )
        )
    views.sort(key=lambda view: view.npu_id)
    snapshot = CIRControlSnapshot(
        time_ms=0.0,
        evaluation=1,
        layer_jobs_since_previous=0,
        num_npu=NUM_NPU,
        num_ssu=NUM_SSU,
        active_requests=tuple(views),
        current_path_cirs_by_ssu=tuple(
            (0.0,) * PATH_COUNT for _ in range(NUM_SSU)
        ),
    )
    return workload, snapshot


def _summarize(name, grants, selected, views):
    grants = np.asarray(grants, dtype=float)
    selected = set(int(npu) for npu in selected)
    raw = np.asarray(
        [
            np.asarray(view.next_layer_work_gb_by_ssu)
            / (view.per_layer_compute_ms / 1000.0)
            for view in views
        ]
    )
    ratios = np.divide(
        grants.sum(axis=1),
        raw.sum(axis=1),
        out=np.zeros(NUM_NPU),
        where=raw.sum(axis=1) > 0.0,
    )
    by_category = {}
    for category in CATEGORIES:
        ids = [view.npu_id for view in views if view.category == category]
        values = ratios[ids]
        by_category[category] = {
            "count": len(ids),
            "grant_ratio_mean": float(values.mean()),
            "grant_ratio_min": float(values.min()),
            "grant_ratio_max": float(values.max()),
        }
    return {
        "name": name,
        "selected_count": len(selected),
        "selected_category_counts": dict(
            sorted(Counter(views[npu].category for npu in selected).items())
        ),
        "rejected_category_counts": dict(
            sorted(
                Counter(
                    views[npu].category
                    for npu in range(NUM_NPU)
                    if npu not in selected
                ).items()
            )
        ),
        "ssd_column_gbps": [float(value) for value in grants.sum(axis=0)],
        "ssd_column_min_gbps": float(grants.sum(axis=0).min()),
        "ssd_column_mean_gbps": float(grants.sum(axis=0).mean()),
        "ssd_column_max_gbps": float(grants.sum(axis=0).max()),
        "npu_row_max_gbps": float(grants.sum(axis=1).max()),
        "category_grant_ratios": by_category,
    }


def analyze():
    workload, snapshot = _real_snapshot()
    views = snapshot.active_requests
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(NUM_NPU))

    v3 = DeadlineBarrierSchemeBControllerV3(paths, hysteresis_gbps=0.0)
    v3(snapshot)
    v3_allocation = v3.last_plan.allocation

    v1 = SLOAdmissionSchemeBController(paths)
    v1(snapshot)
    v1_allocation = v1.last_allocation

    raw = np.asarray(
        [
            np.asarray(view.next_layer_work_gb_by_ssu)
            / (view.per_layer_compute_ms / 1000.0)
            for view in views
        ]
    )
    fixed_targets = 0.52 * raw
    normalized = fixed_targets / sim.DISK_BW
    cost_priority = tuple(
        sorted(
            range(NUM_NPU),
            key=lambda npu: (
                float(normalized[npu].sum()),
                float(normalized[npu].max()),
                npu,
            ),
        )
    )
    corrected = allocate_deadline_barrier_grants_v3(
        raw,
        fixed_targets,
        priority_npu_ids=cost_priority,
        background_reserve_fraction=0.05,
        ssd_caps=sim.DISK_BW,
        npu_caps=sim.NPU_BW_LIMIT,
    )
    return {
        "schema_version": 1,
        "num_npu": NUM_NPU,
        "num_ssu": NUM_SSU,
        "n_layers": N_LAYERS,
        "seed": SEED,
        "assignment_hash": workload.statistics["assignment_hash"],
        "workload_hash": workload.workload_hash,
        "placement_hash": workload.placement_hash,
        "trace_hash": workload.trace_hash,
        "snapshot": "sequence-0, all requests waiting, full 16-layer manifest",
        "allocators": [
            _summarize(
                "deadline_barrier_v3",
                v3_allocation.grants_gbps,
                v3_allocation.selected_npu_ids,
                views,
            ),
            _summarize(
                "admission_v1",
                v1_allocation.grants_gbps,
                v1_allocation.selected_npu_ids,
                views,
            ),
            _summarize(
                "cardinality_first_explicit_spill",
                corrected.grants_gbps,
                corrected.selected_npu_ids,
                views,
            ),
        ],
    }


def render(payload):
    lines = [
        "# Real sequence-0 V3 allocation snapshot",
        "",
        "All 32 requests are waiting; every allocator sees the same full "
        "16-layer manifest and fixed SSU10 placement.",
        "",
        "| Allocator | Selected | Selected categories | Rejected categories | "
        "SSD min | mean | max | NPU max |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["allocators"]:
        lines.append(
            f"| {row['name']} | {row['selected_count']} | "
            f"{row['selected_category_counts']} | {row['rejected_category_counts']} | "
            f"{row['ssd_column_min_gbps']:.3f} | "
            f"{row['ssd_column_mean_gbps']:.3f} | "
            f"{row['ssd_column_max_gbps']:.3f} | "
            f"{row['npu_row_max_gbps']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    payload = analyze()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render(payload))


if __name__ == "__main__":
    main()

