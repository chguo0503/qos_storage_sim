"""SLOAwareSchemeBController: pure snapshot-to-CIR decision, no event state access."""

from __future__ import annotations

from typing import Sequence
from simulator.contracts import CIRControlDecision, CIRControlSnapshot, CausalLayerSnapshot
from simulator.policies.policy_logic import (
    SSD_CAP_GBPS, NPU_CAP_GBPS, ManifestDemand,
    CausalLayerObservation as PolicyCausalLayerObservation,
    plan_scheme_b, plan_slo_aware_scheme_b, plan_causal_scheme_b,
)


class SLOAwareSchemeBController:
    """Full-manifest Scheme B with request/coflow-synchronized progress.

    The client-visible snapshot contains all not-yet-ready layers of every
    active request.  A single scalar progress ratio is allocated to the whole
    NPU x SSU demand vector, first toward the warm TTFT target and then toward
    full I/O hiding.  This prevents bandwidth on a request's small SSU flow
    from masking starvation of its barrier-critical flow.
    """

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        slo_alpha: float = 2.0,
        ssd_cap_gbps: float = SSD_CAP_GBPS,
        npu_cap_gbps: float = NPU_CAP_GBPS,
    ):
        if slo_alpha <= 0.0:
            raise ValueError("slo_alpha must be positive")
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.slo_alpha = float(slo_alpha)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self.last_plan = None

    def __call__(self, snapshot: CIRControlSnapshot):
        manifests = []
        for request in snapshot.active_requests:
            work = request.remaining_work_gb_by_ssu
            compute_ms = request.remaining_compute_budget_ms
            # Compatibility for callers that construct the older, next-layer
            # only ControlRequestView directly.
            if not work and request.remaining_layers > 0:
                work = tuple(
                    request.remaining_layers * amount
                    for amount in request.next_layer_work_gb_by_ssu
                )
                compute_ms = request.remaining_layers * request.per_layer_compute_ms
            if compute_ms <= 0.0 or not any(amount > 0.0 for amount in work):
                continue
            manifests.append(
                ManifestDemand(
                    request_id=request.request_id,
                    npu_id=request.npu_id,
                    compute_budget_s=compute_ms / 1000.0,
                    work_by_ssu_gb=tuple(work),
                )
            )
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        plan = plan_slo_aware_scheme_b(
            manifests,
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            slo_alpha=self.slo_alpha,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_count=path_count,
            path_by_npu=self.path_by_npu,
        )
        self.last_plan = plan
        return CIRControlDecision(plan.path_cirs_by_ssu)


def main():
    from simulator.contracts import ControlRequestView
    from simulator.policies.policy_logic import validate_scheme_b_plan
    requests = tuple(ControlRequestView(n, n, "SS", 10.0, -1, 2, (.6, .1), True)
                     for n in range(2))
    snapshot = CIRControlSnapshot(0, 1, 0, 2, 2, requests, ((0.0,) * 256,) * 2)
    controller = SLOAwareSchemeBController((0, 32), slo_alpha=2)
    decision = controller(snapshot)
    assert validate_scheme_b_plan(controller.last_plan)
    assert decision.path_cirs_by_ssu == controller.last_plan.path_cirs_by_ssu
    for demand, grant in zip(controller.last_plan.demands_gbps, controller.last_plan.grants_gbps):
        assert abs(grant[0] / demand[0] - grant[1] / demand[1]) < 1e-12
    print("scheme_b_slo: PASS (whole-coflow progress, legacy snapshot compatibility)")


if __name__ == "__main__":
    main()
