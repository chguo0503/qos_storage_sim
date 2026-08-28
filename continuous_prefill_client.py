"""DES-independent clients and Scheme-B control for continuous prefill."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from continuous_batch_control import allocate_grants
from scheme_b_prefill import GROUP_COUNT, PATH_COUNT, dedicated_path_id
import sim
from strategy_profiles import FINAL_STATIC


SUBMIT_BATCH_SIZE = 1
ISSUE_INTERVAL_US = 0.1
SCHEME_B_MODES = (
    "once",
    "periodic1",
    "periodic2",
    "periodic4",
    "periodic8",
    "membership_event",
)
_PERIOD_LAYERS = {
    "periodic1": 1,
    "periodic2": 2,
    "periodic4": 4,
    "periodic8": 8,
}
_EPS = 1e-9


@dataclass(frozen=True)
class LegacyStrategySpec:
    name: str
    description: str
    path_selection_mode: str
    pressure_window_io: int | None

    def client_config(self):
        return sim.ClientIOConfig(
            name=f"continuous_prefill_{self.name}",
            pressure_window_io=self.pressure_window_io,
            submit_batch_size=SUBMIT_BATCH_SIZE,
            issue_interval_us=ISSUE_INTERVAL_US,
            path_selection_mode=self.path_selection_mode,
        )

    def metadata(self):
        return {
            "name": self.name,
            "description": self.description,
            "policy": sim.POLICY_QOS_STATIC_CIR,
            "path_selection_mode": self.path_selection_mode,
            "pressure_window_io": self.pressure_window_io,
            "submit_batch_size": SUBMIT_BATCH_SIZE,
            "issue_interval_us": ISSUE_INTERVAL_US,
            "qos_profile": FINAL_STATIC.name,
        }


def legacy_strategy_specs():
    """The five retained clients, unchanged apart from the shared batch DES."""
    return (
        LegacyStrategySpec(
            "baseline",
            "All I/Os use Path 0; no pressure reads",
            sim.PATH_SELECTION_FIXED_PATH_ZERO,
            None,
        ),
        LegacyStrategySpec(
            "path_rr",
            "Category-legal deterministic Path round-robin",
            sim.PATH_SELECTION_STATELESS_RR,
            None,
        ),
        LegacyStrategySpec(
            "layer_once",
            "Read Path pressure once per request-layer-SSU",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            None,
        ),
        LegacyStrategySpec(
            "refresh8",
            "Refresh Path pressure every eight planned I/Os",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            8,
        ),
        LegacyStrategySpec(
            "refresh1",
            "Refresh Path pressure before every planned I/O",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            1,
        ),
    )


def legacy_qos_config():
    return FINAL_STATIC.hardware_config()


def scheme_b_client_config(mode):
    return sim.ClientIOConfig(
        name=f"continuous_prefill_scheme_b_{mode}",
        pressure_window_io=None,
        submit_batch_size=SUBMIT_BATCH_SIZE,
        issue_interval_us=ISSUE_INTERVAL_US,
        path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
    )


@dataclass(frozen=True)
class ActiveRequestSnapshot:
    request_id: int
    npu_id: int
    arrival_ms: float
    per_layer_us: float
    work_by_ssu_gb: tuple[float, ...]

    @classmethod
    def from_request(cls, request):
        return cls(
            request_id=int(request.request_id),
            npu_id=int(request.npu_id),
            arrival_ms=float(request.arrival_ms),
            per_layer_us=float(request.per_layer_us),
            work_by_ssu_gb=tuple(map(float, request.work_by_ssu_gb)),
        )


@dataclass(frozen=True)
class SchemeBTarget:
    active_request_ids: tuple[int, ...]
    demands_gbps: tuple[tuple[float, ...], ...]
    grants_gbps: tuple[tuple[float, ...], ...]
    path_cirs_by_ssu: tuple[tuple[float, ...], ...]
    target_hash: str


@dataclass(frozen=True)
class SchemeBPathWrite:
    npu_id: int
    ssu_id: int
    path_id: int
    old_cir_gbps: float
    new_cir_gbps: float


@dataclass(frozen=True)
class SchemeBCommit:
    time_ms: float
    reason: str
    target: SchemeBTarget
    writes: tuple[SchemeBPathWrite, ...]
    qos_configs_by_ssu: tuple[sim.StaticQoSConfig, ...]

    @property
    def path_write_count(self):
        return len(self.writes)


@dataclass(frozen=True)
class SchemeBEvaluation:
    time_ms: float
    reason: str
    active_request_count: int
    target_hash: str
    committed: bool
    path_write_count: int


def _target_hash(active_request_ids, demands, grants):
    encoded = json.dumps(
        {
            "active_request_ids": active_request_ids,
            "demands_gbps": demands,
            "grants_gbps": grants,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_scheme_b_target(
    active_requests,
    *,
    num_npu,
    num_ssu,
    ssd_cap_gbps=sim.DISK_BW,
    npu_cap_gbps=sim.NPU_BW_LIMIT,
):
    """Compute aggregate active-manifest demand and demand-capped max-min grant."""
    snapshots = tuple(sorted(active_requests, key=lambda request: request.request_id))
    if len({request.request_id for request in snapshots}) != len(snapshots):
        raise ValueError("active request IDs must be unique")
    work = np.zeros((num_npu, num_ssu), dtype=np.float64)
    compute_s = np.zeros(num_npu, dtype=np.float64)
    for request in snapshots:
        if len(request.work_by_ssu_gb) != num_ssu:
            raise ValueError("active request work vector has the wrong SSU count")
        work[request.npu_id] += request.work_by_ssu_gb
        compute_s[request.npu_id] += request.per_layer_us / 1e6
    demand = np.divide(
        work,
        compute_s[:, None],
        out=np.zeros_like(work),
        where=compute_s[:, None] > 0.0,
    )
    demands = tuple(tuple(float(value) for value in row) for row in demand)
    grants = allocate_grants(
        demands,
        ssd_caps=ssd_cap_gbps,
        npu_caps=npu_cap_gbps,
    )
    path_cirs = []
    for ssu_id in range(num_ssu):
        cirs = [0.0] * PATH_COUNT
        for npu_id in range(num_npu):
            cirs[dedicated_path_id(npu_id)] = grants[npu_id][ssu_id]
        path_cirs.append(tuple(cirs))
    active_ids = tuple(request.request_id for request in snapshots)
    target = SchemeBTarget(
        active_request_ids=active_ids,
        demands_gbps=demands,
        grants_gbps=grants,
        path_cirs_by_ssu=tuple(path_cirs),
        target_hash=_target_hash(active_ids, demands, grants),
    )
    assert all(sum(row) <= npu_cap_gbps + _EPS for row in grants)
    assert all(
        sum(grants[npu_id][ssu_id] for npu_id in range(num_npu))
        <= ssd_cap_gbps + _EPS
        for ssu_id in range(num_ssu)
    )
    return target


def qos_configs_from_path_cirs(path_cirs_by_ssu):
    return tuple(
        sim.StaticQoSConfig(
            path_cirs=cirs,
            path_pirs=(float("inf"),) * PATH_COUNT,
            path_weights=(1.0,) * PATH_COUNT,
            group_weights=(1.0,) * GROUP_COUNT,
            category_paths_per_group=(PATH_COUNT // GROUP_COUNT,),
            category_labels=("NPU",),
        )
        for cirs in path_cirs_by_ssu
    )


class SchemeBController:
    """Causal Scheme-B target sampler with atomic per-SSD CIR commits.

    The DES should call :meth:`observe` at every admission/completion and at
    :attr:`next_update_time_ms`.  Only snapshots passed to ``observe`` are
    visible; passing a request whose arrival is in the future is rejected.
    """

    def __init__(
        self,
        *,
        num_npu,
        num_ssu,
        t_layer_ms,
        mode,
        membership_debounce_ms=1.0,
        ssd_cap_gbps=sim.DISK_BW,
        npu_cap_gbps=sim.NPU_BW_LIMIT,
    ):
        if mode not in SCHEME_B_MODES:
            raise ValueError(f"mode must be one of {SCHEME_B_MODES}")
        self.num_npu = int(num_npu)
        self.num_ssu = int(num_ssu)
        self.t_layer_ms = float(t_layer_ms)
        self.mode = mode
        self.membership_debounce_ms = float(membership_debounce_ms)
        self.ssd_cap_gbps = float(ssd_cap_gbps)
        self.npu_cap_gbps = float(npu_cap_gbps)
        self.path_by_npu = tuple(
            dedicated_path_id(npu_id) for npu_id in range(self.num_npu)
        )
        self.current_target = None
        self.history = []
        self.commit_count = 0
        self.path_write_count = 0
        self._last_observed_ids = ()
        self._next_periodic_ms = None
        self._pending_membership_ms = None

    @property
    def next_update_time_ms(self):
        if self.current_target is None:
            return 0.0
        if self.mode in _PERIOD_LAYERS:
            return self._next_periodic_ms
        if self.mode == "membership_event":
            return self._pending_membership_ms
        return None

    def _period_ms(self):
        return self.t_layer_ms * _PERIOD_LAYERS[self.mode]

    def _advance_periodic_deadline(self, now_ms):
        period_ms = self._period_ms()
        if self._next_periodic_ms is None:
            self._next_periodic_ms = now_ms + period_ms
            return
        skipped = max(
            1,
            int(math.floor((now_ms - self._next_periodic_ms) / period_ms)) + 1,
        )
        self._next_periodic_ms += skipped * period_ms

    def _writes(self, old, new):
        old_grants = (
            old.grants_gbps
            if old is not None
            else tuple(
                (0.0,) * self.num_ssu for _ in range(self.num_npu)
            )
        )
        return tuple(
            SchemeBPathWrite(
                npu_id=npu_id,
                ssu_id=ssu_id,
                path_id=self.path_by_npu[npu_id],
                old_cir_gbps=old_grants[npu_id][ssu_id],
                new_cir_gbps=new.grants_gbps[npu_id][ssu_id],
            )
            for npu_id in range(self.num_npu)
            for ssu_id in range(self.num_ssu)
            if abs(
                old_grants[npu_id][ssu_id]
                - new.grants_gbps[npu_id][ssu_id]
            )
            > _EPS
        )

    def observe(self, now_ms, active_requests):
        """Observe a causal active set and return a commit when CIRs changed."""
        now_ms = float(now_ms)
        snapshots = tuple(
            sorted(active_requests, key=lambda request: request.request_id)
        )
        if any(request.arrival_ms > now_ms + _EPS for request in snapshots):
            raise ValueError("future arrivals cannot be visible to Scheme B")
        active_ids = tuple(request.request_id for request in snapshots)
        membership_changed = (
            self.current_target is not None
            and active_ids != self._last_observed_ids
        )
        self._last_observed_ids = active_ids

        reason = None
        if self.current_target is None:
            reason = "initial"
        elif self.mode in _PERIOD_LAYERS:
            if now_ms + _EPS >= self._next_periodic_ms:
                reason = self.mode
        elif self.mode == "membership_event":
            if membership_changed and self._pending_membership_ms is None:
                self._pending_membership_ms = (
                    now_ms + self.membership_debounce_ms
                )
            if (
                self._pending_membership_ms is not None
                and now_ms + _EPS >= self._pending_membership_ms
            ):
                reason = "membership_event"
        if reason is None:
            return None

        target = build_scheme_b_target(
            snapshots,
            num_npu=self.num_npu,
            num_ssu=self.num_ssu,
            ssd_cap_gbps=self.ssd_cap_gbps,
            npu_cap_gbps=self.npu_cap_gbps,
        )
        writes = self._writes(self.current_target, target)
        committed = self.current_target is None or bool(writes)
        self.history.append(
            SchemeBEvaluation(
                time_ms=now_ms,
                reason=reason,
                active_request_count=len(snapshots),
                target_hash=target.target_hash,
                committed=committed,
                path_write_count=len(writes),
            )
        )
        if self.mode in _PERIOD_LAYERS:
            self._advance_periodic_deadline(now_ms)
        elif self.mode == "membership_event":
            self._pending_membership_ms = None
        if not committed:
            return None

        self.current_target = target
        self.commit_count += 1
        self.path_write_count += len(writes)
        return SchemeBCommit(
            time_ms=now_ms,
            reason=reason,
            target=target,
            writes=writes,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                target.path_cirs_by_ssu
            ),
        )

    def summary(self):
        return {
            "mode": self.mode,
            "t_layer_ms": self.t_layer_ms,
            "period_layers": _PERIOD_LAYERS.get(self.mode),
            "membership_debounce_ms": (
                self.membership_debounce_ms
                if self.mode == "membership_event"
                else None
            ),
            "commit_count": self.commit_count,
            "path_write_count": self.path_write_count,
            "evaluation_count": len(self.history),
            "target_hashes": [row.target_hash for row in self.history],
            "evaluations": [
                {
                    "time_ms": row.time_ms,
                    "reason": row.reason,
                    "active_request_count": row.active_request_count,
                    "target_hash": row.target_hash,
                    "committed": row.committed,
                    "path_write_count": row.path_write_count,
                }
                for row in self.history
            ],
        }
