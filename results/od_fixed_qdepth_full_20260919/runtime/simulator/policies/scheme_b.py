"""MaxMinSchemeBController: pure snapshot-to-CIR decision, no event state access."""

from __future__ import annotations

from typing import Sequence
from simulator.contracts import CIRControlDecision, CIRControlSnapshot, CausalLayerSnapshot
from simulator.policies.policy_logic import (
    SSD_CAP_GBPS, NPU_CAP_GBPS, ManifestDemand,
    CausalLayerObservation as PolicyCausalLayerObservation,
    plan_scheme_b, plan_slo_aware_scheme_b, plan_causal_scheme_b,
)


class MaxMinSchemeBController:
    """Manifest-based dynamic Scheme B for NPU-dedicated Paths."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        horizon_layers: int = 1,
        ssd_cap_gbps: float = SSD_CAP_GBPS,
        npu_cap_gbps: float = NPU_CAP_GBPS,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.horizon_layers = int(horizon_layers)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)

    def __call__(self, snapshot: CIRControlSnapshot):
        manifests = []
        for request in snapshot.active_requests:
            horizon = min(request.remaining_layers, self.horizon_layers)
            if horizon <= 0:
                continue
            manifests.append(
                ManifestDemand(
                    request_id=request.request_id,
                    npu_id=request.npu_id,
                    compute_budget_s=(horizon * request.per_layer_compute_ms / 1000.0),
                    work_by_ssu_gb=tuple(
                        horizon * work_gb
                        for work_gb in request.next_layer_work_gb_by_ssu
                    ),
                )
            )
        path_count = len(snapshot.current_path_cirs_by_ssu[0])
        plan = plan_scheme_b(
            manifests,
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_count=path_count,
            path_by_npu=self.path_by_npu,
        )
        return CIRControlDecision(plan.path_cirs_by_ssu)


def main():
    from simulator.contracts import ControlRequestView
    requests = tuple(ControlRequestView(n, n, "SS", 10.0, -1, 2, (.6, .1), True)
                     for n in range(2))
    snapshot = CIRControlSnapshot(0, 1, 0, 2, 2, requests, ((0.0,) * 256,) * 2)
    controller = MaxMinSchemeBController((0, 32))
    decision = controller(snapshot)
    assert decision.path_cirs_by_ssu[0][0] == decision.path_cirs_by_ssu[0][32] == 20
    assert decision.path_cirs_by_ssu[1][0] == decision.path_cirs_by_ssu[1][32] == 10
    assert controller(snapshot) == decision
    print("scheme_b: PASS (snapshot-to-CIR decision, capacity-limited equal grants)")


if __name__ == "__main__":
    main()
