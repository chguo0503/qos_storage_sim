"""CausalMaxMinSchemeBController: pure snapshot-to-CIR decision, no event state access."""

from __future__ import annotations

from typing import Sequence
from simulator.contracts import CIRControlDecision, CIRControlSnapshot, CausalLayerSnapshot
from simulator.policies.policy_logic import (
    SSD_CAP_GBPS, NPU_CAP_GBPS, ManifestDemand,
    CausalLayerObservation as PolicyCausalLayerObservation,
    plan_scheme_b, plan_slo_aware_scheme_b, plan_causal_scheme_b,
)


class CausalMaxMinSchemeBController:
    """Max-min Scheme B driven only by completed previous-layer bytes."""

    def __init__(
        self,
        path_by_npu: Sequence[int],
        *,
        cold_path_id: int,
        cold_path_cir_gbps: float,
        path_count: int,
        ssd_cap_gbps: float = SSD_CAP_GBPS,
        npu_cap_gbps: float = NPU_CAP_GBPS,
    ):
        self.path_by_npu = tuple(int(path_id) for path_id in path_by_npu)
        self.cold_path_id = int(cold_path_id)
        self.cold_path_cir_gbps = float(cold_path_cir_gbps)
        self.path_count = int(path_count)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self._previous_signature = None

    def __call__(self, snapshot: CausalLayerSnapshot):
        warm = tuple(
            observation
            for observation in snapshot.active_batches
            if observation.observed_layer >= 0
        )
        cold_count = len(snapshot.active_batches) - len(warm)
        signature = (
            bool(cold_count),
            tuple(
                (
                    observation.npu_id,
                    observation.observed_work_gb_by_ssu,
                    observation.compute_budget_ms,
                )
                for observation in warm
            ),
        )
        if signature == self._previous_signature:
            return None
        self._previous_signature = signature

        plan = plan_causal_scheme_b(
            tuple(
                PolicyCausalLayerObservation(
                    request_id=observation.batch_id,
                    npu_id=observation.npu_id,
                    observed_layer=observation.observed_layer,
                    compute_budget_ms=observation.compute_budget_ms,
                    observed_work_gb_by_ssu=observation.observed_work_gb_by_ssu,
                )
                for observation in snapshot.active_batches
            ),
            num_npu=snapshot.num_npu,
            num_ssu=snapshot.num_ssu,
            cold_path_id=self.cold_path_id,
            cold_path_cir_gbps=self.cold_path_cir_gbps,
            path_count=self.path_count,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
            path_by_npu=self.path_by_npu,
        )
        return CIRControlDecision(plan.path_cirs_by_ssu)


def main():
    from simulator.contracts import CausalLayerObservation
    snapshot = CausalLayerSnapshot(2, 1, (
        CausalLayerObservation(1, 0, 0, (.2,), 10),
        CausalLayerObservation(2, 1, -1, (0.0,), 10),
    ))
    controller = CausalMaxMinSchemeBController((0, 32), cold_path_id=255,
        cold_path_cir_gbps=4.0, path_count=256)
    decision = controller(snapshot)
    assert decision.path_cirs_by_ssu[0][255] == 4
    assert decision.path_cirs_by_ssu[0][0] == 20
    assert sum(decision.path_cirs_by_ssu[0]) <= 40
    assert controller(snapshot) is None
    print("scheme_b_causal: PASS (observed-layer grants, cold reservation, unchanged-state suppression)")


if __name__ == "__main__":
    main()
