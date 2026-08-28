"""Request-known Scheme-B controllers for the batch-1 closed-loop experiment."""

from __future__ import annotations

import numpy as np

import sim
from continuous_batch_sim import CIRControlDecision


def allocate_coflow_grants(work, compute_ms, *, ssd_cap=40.0, npu_cap=50.0):
    """Max-min fill per-NPU demand satisfaction while preserving coflow shape."""
    work = np.asarray(work, dtype=float)
    compute_ms = np.asarray(compute_ms, dtype=float)
    total_work = work.sum(axis=1)
    speed = np.minimum(
        np.divide(
            1000.0,
            compute_ms,
            out=np.zeros_like(compute_ms),
            where=compute_ms > 0.0,
        ),
        np.divide(
            npu_cap,
            total_work,
            out=np.zeros_like(total_work),
            where=total_work > 0.0,
        ),
    )
    demand = work * speed[:, None]
    rho = np.zeros(len(work), dtype=float)
    residual = np.full(work.shape[1], float(ssd_cap))
    active = total_work > 0.0
    while np.any(active):
        rate = demand[active].sum(axis=0)
        resource_steps = np.divide(
            residual,
            rate,
            out=np.full_like(residual, np.inf),
            where=rate > 0.0,
        )
        step = min(float(np.min(resource_steps)), float(np.min(1.0 - rho[active])))
        rho[active] += step
        residual -= step * rate
        active &= rho < 1.0 - 1e-12
        saturated = residual <= 1e-12
        if np.any(saturated):
            active &= ~np.any(demand[:, saturated] > 0.0, axis=1)
    return tuple(
        tuple(float(value) for value in row)
        for row in rho[:, None] * demand
    )


class CoflowSchemeBController:
    """Use only the current request manifest and one fixed Path per NPU."""

    def __init__(self, path_by_npu, path_count):
        self.path_by_npu = tuple(path_by_npu)
        self.path_count = int(path_count)

    def __call__(self, snapshot):
        work = np.zeros((snapshot.num_npu, snapshot.num_ssu), dtype=float)
        compute_ms = np.zeros(snapshot.num_npu, dtype=float)
        for request in snapshot.active_requests:
            work[request.npu_id] = request.next_layer_work_gb_by_ssu
            compute_ms[request.npu_id] = request.per_layer_compute_ms
        grants = allocate_coflow_grants(
            work,
            compute_ms,
            ssd_cap=sim.DISK_BW,
            npu_cap=sim.NPU_BW_LIMIT,
        )
        cirs_by_ssu = []
        for ssu_id in range(snapshot.num_ssu):
            cirs = [0.0] * self.path_count
            for npu_id, path_id in enumerate(self.path_by_npu):
                cirs[path_id] = grants[npu_id][ssu_id]
            cirs_by_ssu.append(tuple(cirs))
        return CIRControlDecision(tuple(cirs_by_ssu))
