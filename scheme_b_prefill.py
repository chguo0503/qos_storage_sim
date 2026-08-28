"""One-shot Scheme-B planning for a known batch of prefill requests.

All request manifests are known before launch.  The controller computes one
NPU x SSU demand matrix, allocates max-min CIR grants, and programs one
dedicated Path per NPU on every SSU.  The configuration is then reused for all
prefill layers; launch jitter does not trigger another control-plane update.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from continuous_batch_control import GrantMatrix, allocate_grants
from sim import PreparedSimulationInputs, StaticQoSConfig


PATH_COUNT = 256
GROUP_COUNT = 8
PATHS_PER_GROUP = PATH_COUNT // GROUP_COUNT
MAX_NPU = 128
SSD_CAP_GBPS = 40.0
NPU_CAP_GBPS = 50.0


def dedicated_path_id(npu_id: int) -> int:
    """Spread 128 NPU-dedicated Paths evenly over the eight QoS groups."""
    return (npu_id % GROUP_COUNT) * PATHS_PER_GROUP + npu_id // GROUP_COUNT


def cold_start_hybrid_path_id(npu_id: int) -> int:
    """Use the unused half of every Group and keep Path 0 for Layer 0."""
    return dedicated_path_id(npu_id) + MAX_NPU // GROUP_COUNT


def _demand_matrix(prepared: PreparedSimulationInputs) -> GrantMatrix:
    loads = sorted(prepared.request_loads, key=lambda load: load["npu_id"])
    assert tuple(load["npu_id"] for load in loads) == tuple(range(len(loads)))
    assert len(loads) <= MAX_NPU

    rows = []
    for load in loads:
        layers = prepared.placement_by_request[load["request_id"]]
        layer_zero = layers[0]
        assert all(layers[layer] == layer_zero for layer in range(prepared.n_layers))
        work_by_ssu = [0.0] * prepared.num_disk
        for ssu_id, block_gb in layer_zero:
            work_by_ssu[ssu_id] += block_gb
        compute_s = load["per_layer_us"] / 1e6
        rows.append(tuple(work_gb / compute_s for work_gb in work_by_ssu))
    return tuple(rows)


def _qos_configs(grants: GrantMatrix) -> tuple[StaticQoSConfig, ...]:
    npu_count = len(grants)
    ssu_count = len(grants[0]) if grants else 0
    path_by_npu = tuple(dedicated_path_id(npu) for npu in range(npu_count))
    configs = []
    for ssu in range(ssu_count):
        cirs = [0.0] * PATH_COUNT
        for npu, path_id in enumerate(path_by_npu):
            cirs[path_id] = grants[npu][ssu]
        configs.append(
            StaticQoSConfig(
                path_cirs=tuple(cirs),
                path_pirs=(float("inf"),) * PATH_COUNT,
                path_weights=(1.0,) * PATH_COUNT,
                group_weights=(1.0,) * GROUP_COUNT,
                category_paths_per_group=(PATHS_PER_GROUP,),
                category_labels=("NPU",),
            )
        )
    return tuple(configs)


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SchemeBPrefillPlan:
    demands_gbps: GrantMatrix
    grants_gbps: GrantMatrix
    qos_configs: tuple[StaticQoSConfig, ...]
    path_by_npu: tuple[int, ...]
    workload_hash: str
    placement_hash: str
    n_layers: int
    ssd_cap_gbps: float
    npu_cap_gbps: float
    fingerprint: str

    @property
    def num_npu(self) -> int:
        return len(self.demands_gbps)

    @property
    def num_ssu(self) -> int:
        return len(self.qos_configs)

    def constraints(self) -> dict[str, bool]:
        tolerance = 1e-9
        grant_rows = tuple(map(sum, self.grants_gbps))
        grant_columns = tuple(
            sum(self.grants_gbps[npu][ssu] for npu in range(self.num_npu))
            for ssu in range(self.num_ssu)
        )
        return {
            "non_negative": all(
                grant >= -tolerance for row in self.grants_gbps for grant in row
            ),
            "demand_capped": all(
                self.grants_gbps[npu][ssu]
                <= self.demands_gbps[npu][ssu] + tolerance
                for npu in range(self.num_npu)
                for ssu in range(self.num_ssu)
            ),
            "npu_capacity": all(
                total <= self.npu_cap_gbps + tolerance for total in grant_rows
            ),
            "ssd_capacity": all(
                total <= self.ssd_cap_gbps + tolerance for total in grant_columns
            ),
            "dedicated_paths": len(set(self.path_by_npu)) == self.num_npu,
            "cir_matches_grants": all(
                abs(
                    self.qos_configs[ssu].path_cirs[self.path_by_npu[npu]]
                    - self.grants_gbps[npu][ssu]
                )
                <= tolerance
                for npu in range(self.num_npu)
                for ssu in range(self.num_ssu)
            ),
            "cir_capacity": all(
                sum(config.path_cirs) <= self.ssd_cap_gbps + tolerance
                for config in self.qos_configs
            ),
        }

    def summary(self) -> dict:
        demand_rows = tuple(map(sum, self.demands_gbps))
        grant_rows = tuple(map(sum, self.grants_gbps))
        demand_columns = tuple(
            sum(self.demands_gbps[npu][ssu] for npu in range(self.num_npu))
            for ssu in range(self.num_ssu)
        )
        grant_columns = tuple(
            sum(self.grants_gbps[npu][ssu] for npu in range(self.num_npu))
            for ssu in range(self.num_ssu)
        )
        constraints = self.constraints()
        return {
            "scheme": "scheme_b_prefill_one_shot",
            "admission_scope": "all_batch_manifests_known_before_launch",
            "control_plane_updates": 1,
            "reuse_layers": self.n_layers,
            "path_mapping": "(npu_id % 8) * 32 + npu_id // 8",
            "num_npu": self.num_npu,
            "num_ssu": self.num_ssu,
            "ssd_cap_gbps": self.ssd_cap_gbps,
            "npu_cap_gbps": self.npu_cap_gbps,
            "total_demand_gbps": sum(demand_rows),
            "total_grant_gbps": sum(grant_rows),
            "per_npu_demand_gbps": list(demand_rows),
            "per_npu_grant_gbps": list(grant_rows),
            "per_ssu_demand_gbps": list(demand_columns),
            "per_ssu_grant_gbps": list(grant_columns),
            "constraints": constraints,
            "all_constraints_hold": all(constraints.values()),
            "workload_hash": self.workload_hash,
            "placement_hash": self.placement_hash,
            "fingerprint": self.fingerprint,
        }


def build_scheme_b_prefill_plan(
    prepared: PreparedSimulationInputs,
    *,
    ssd_cap_gbps: float = SSD_CAP_GBPS,
    npu_cap_gbps: float = NPU_CAP_GBPS,
) -> SchemeBPrefillPlan:
    """Plan CIR once for a batch=1, 16-layer prefill admission batch."""
    demands = _demand_matrix(prepared)
    grants = allocate_grants(
        demands,
        ssd_caps=ssd_cap_gbps,
        npu_caps=npu_cap_gbps,
    )
    paths = tuple(dedicated_path_id(npu) for npu in range(len(demands)))
    configs = _qos_configs(grants)
    fingerprint = _fingerprint(
        {
            "scheme": "scheme_b_prefill_one_shot_v1",
            "workload_hash": prepared.workload_hash,
            "placement_hash": prepared.placement_hash,
            "n_layers": prepared.n_layers,
            "ssd_cap_gbps": ssd_cap_gbps,
            "npu_cap_gbps": npu_cap_gbps,
            "demands_gbps": demands,
            "grants_gbps": grants,
            "path_by_npu": paths,
        }
    )
    plan = SchemeBPrefillPlan(
        demands_gbps=demands,
        grants_gbps=grants,
        qos_configs=configs,
        path_by_npu=paths,
        workload_hash=prepared.workload_hash,
        placement_hash=prepared.placement_hash,
        n_layers=prepared.n_layers,
        ssd_cap_gbps=float(ssd_cap_gbps),
        npu_cap_gbps=float(npu_cap_gbps),
        fingerprint=fingerprint,
    )
    assert all(plan.constraints().values())
    return plan
