"""Millisecond-scale SSU QoS read/CIR-write sensitivity experiment.

The runner is deliberately shardable: every host runs an exact subset of the
same source/config/workload fingerprint, writes an atomic JSON checkpoint, and
the final analysis merges only compatible shards.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import socket
import statistics
import sys
import threading
import time

import numpy as np

from authenticated_workload_inputs import (
    canonical_bw_table,
    load_authenticated_bw_table,
)
import sim
from adaptive_admission_scheme_b_v2_1 import (
    AdaptiveAdmissionSchemeBControllerV2_1,
)
from continuous_batch_sim import (
    CIRControlConfig,
    SteadyStateConfig,
    _steady_accounting_numeric_contract,
    requests_from_continuous_prefill_workload,
    simulate_continuous_batch,
)
from continuous_prefill_client import (
    qos_configs_from_path_cirs,
    routing_strategy_specs,
    scheme_b_client_config,
    static_qos_config,
)
from forecast_hotspot_policy import forecast_frozen_ssu_hotspots
from random_steady_state_workload import (
    IID_UNIFORM_PROFILE_CATALOG_V1,
    build_steady_state_profile_schedule,
    prepare_random_steady_state_workload,
)
from policy_logic import GROUP_COUNT, MAX_NPU, PATH_COUNT, PATHS_PER_GROUP
from protected_floor_scheme_b import (
    REQUIRED_DOWNSTREAM_DEADBAND_GBPS,
    ProtectedFloorSchemeBController,
)
from scheme_b_prefill import cold_start_hybrid_path_id


ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 3
DEFAULT_SEED = 42
SCIENTIFIC_PREFIX_REQUESTS_PER_NPU = 32
THREAD_LIMIT_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    kind: str
    pressure_ttl_ms: float = 0.0
    cir_write_threshold_gbps: float = 0.0
    min_interval_ms: float = 0.0

    def __post_init__(self):
        if not self.name or not self.family:
            raise ValueError("case name and family must be non-empty")
        if self.kind not in (
            "baseline",
            "layer_once",
            "dedicated_wrr",
            "adaptive",
            "pfo",
        ):
            raise ValueError(f"unsupported case kind: {self.kind}")
        values = (
            self.pressure_ttl_ms,
            self.cir_write_threshold_gbps,
            self.min_interval_ms,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("case control parameters must be finite and non-negative")
        if self.cir_write_threshold_gbps > 0.05:
            raise ValueError("CIR write threshold exceeds the simulator safety limit")
        if self.kind == "baseline" and any(value != 0.0 for value in values):
            raise ValueError("baseline cannot carry control parameters")
        if self.kind == "layer_once" and (
            self.cir_write_threshold_gbps != 0.0 or self.min_interval_ms != 0.0
        ):
            raise ValueError("layer_once accepts only a pressure TTL")
        if self.kind == "dedicated_wrr" and any(value != 0.0 for value in values):
            raise ValueError("dedicated_wrr is a zero-control static comparator")
        if self.kind == "adaptive" and self.pressure_ttl_ms != 0.0:
            raise ValueError("Adaptive does not read the pressure table")
        if self.kind == "pfo" and (
            self.pressure_ttl_ms != 0.0
            or self.cir_write_threshold_gbps != REQUIRED_DOWNSTREAM_DEADBAND_GBPS
            or self.min_interval_ms <= 0.0
        ):
            raise ValueError(
                "PFO requires zero pressure TTL/downstream threshold and a "
                "positive external control interval"
            )


@dataclass(frozen=True)
class PFOCase(Case):
    """PFO case whose deadband is owned by the controller, not the DES."""

    pfo_internal_deadband_gbps: float = 0.0
    forecast_hot_fraction: float | None = None
    forecast_requests_per_npu: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.kind != "pfo":
            raise ValueError("PFOCase must use kind='pfo'")
        if (
            not math.isfinite(self.pfo_internal_deadband_gbps)
            or not 0.0 <= self.pfo_internal_deadband_gbps <= 0.05
        ):
            raise ValueError("PFO internal deadband must be in [0, 0.05] GB/s")
        if self.forecast_hot_fraction is None:
            if self.forecast_requests_per_npu != 0:
                raise ValueError("plain PFO cannot carry a forecast prefix")
        elif (
            not math.isfinite(self.forecast_hot_fraction)
            or not 0.0 < self.forecast_hot_fraction <= 1.0
            or self.forecast_requests_per_npu != SCIENTIFIC_PREFIX_REQUESTS_PER_NPU
        ):
            raise ValueError(
                "forecast PFO requires a hot fraction in (0,1] and the frozen "
                "32-request/NPU scientific prefix"
            )


LEGACY32_CASES = (
    Case("baseline", "baseline", "baseline"),
    Case("layer_once_ttl_0ms", "ttl", "layer_once", 0.0),
    Case("layer_once_ttl_0p25ms", "ttl", "layer_once", 0.25),
    Case("layer_once_ttl_1ms", "ttl", "layer_once", 1.0),
    Case("layer_once_ttl_2ms", "ttl", "layer_once", 2.0),
    Case("layer_once_ttl_5ms", "ttl", "layer_once", 5.0),
    Case("adaptive_t0_i25ms", "threshold_interval", "adaptive", 0.0, 0.0, 25.0),
    Case(
        "adaptive_t0p005_i25ms",
        "threshold",
        "adaptive",
        0.0,
        0.005,
        25.0,
    ),
    Case("adaptive_t0p01_i25ms", "threshold", "adaptive", 0.0, 0.01, 25.0),
    Case("adaptive_t0p02_i25ms", "threshold", "adaptive", 0.0, 0.02, 25.0),
    Case("adaptive_t0p05_i25ms", "threshold", "adaptive", 0.0, 0.05, 25.0),
    Case("adaptive_t0_i50ms", "interval", "adaptive", 0.0, 0.0, 50.0),
    Case("adaptive_t0_i100ms", "interval", "adaptive", 0.0, 0.0, 100.0),
    Case("adaptive_t0_i200ms", "interval", "adaptive", 0.0, 0.0, 200.0),
)


@dataclass(frozen=True)
class AdaptiveDefinition:
    """Immutable controller constants shared by every Adaptive case."""

    controller: str = "AdaptiveAdmissionSchemeBControllerV2_1"
    explicit_spill_threshold: float = 0.75
    target_ratio: float = 0.52
    required_ratio: float = 0.50
    background_reserve_fraction: float = 0.05

    def __post_init__(self):
        finite = (
            self.explicit_spill_threshold,
            self.target_ratio,
            self.required_ratio,
            self.background_reserve_fraction,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("Adaptive controller constants must be finite")
        if not 0.0 < self.explicit_spill_threshold <= 1.0:
            raise ValueError("explicit_spill_threshold must be in (0, 1]")
        if not 0.0 <= self.background_reserve_fraction < 1.0:
            raise ValueError("background_reserve_fraction must be in [0, 1)")
        if not 0.0 < self.required_ratio <= self.target_ratio <= 1.0:
            raise ValueError("Adaptive ratios must satisfy 0 < required <= target <= 1")


@dataclass(frozen=True, kw_only=True)
class AdaptiveCase(Case):
    """Adaptive case with an explicit SLO-specific controller profile."""

    tuning_slo_alpha: float = 2.0
    explicit_spill_threshold: float = 0.75
    target_ratio: float = 0.52
    required_ratio: float = 0.50
    background_reserve_fraction: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        if self.kind != "adaptive":
            raise ValueError("AdaptiveCase must use kind='adaptive'")
        if not math.isfinite(self.tuning_slo_alpha) or self.tuning_slo_alpha <= 0.0:
            raise ValueError("Adaptive tuning SLO alpha must be positive")
        expected_required = 1.0 / self.tuning_slo_alpha
        if not math.isclose(
            self.required_ratio,
            expected_required,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Adaptive required_ratio must equal the reciprocal tuning alpha"
            )
        AdaptiveDefinition(
            explicit_spill_threshold=self.explicit_spill_threshold,
            target_ratio=self.target_ratio,
            required_ratio=self.required_ratio,
            background_reserve_fraction=self.background_reserve_fraction,
        )

    def controller_definition(self):
        return AdaptiveDefinition(
            explicit_spill_threshold=self.explicit_spill_threshold,
            target_ratio=self.target_ratio,
            required_ratio=self.required_ratio,
            background_reserve_fraction=self.background_reserve_fraction,
        )


@dataclass(frozen=True)
class ExperimentDefinition:
    """All scale/policy choices that must survive process re-imports."""

    key: str
    experiment_name: str
    num_npu: int
    n_layers: int
    batch_size: int
    default_ssus: tuple[int, ...]
    cases: tuple[Case, ...]
    default_requests_per_npu: int
    default_measurement_ms: float
    adaptive: AdaptiveDefinition = AdaptiveDefinition()
    report_roles: tuple[tuple[str, str], ...] = ()
    require_single_ssu_simulation: bool = False

    def __post_init__(self):
        if not self.key or not self.experiment_name:
            raise ValueError("experiment definition names must be non-empty")
        if not 0 < self.num_npu <= MAX_NPU:
            raise ValueError(f"num_npu must be in [1, {MAX_NPU}]")
        if self.n_layers <= 0 or self.batch_size <= 0:
            raise ValueError("n_layers and batch_size must be positive")
        if not self.default_ssus or any(value <= 0 for value in self.default_ssus):
            raise ValueError("default_ssus must contain positive values")
        if len(set(self.default_ssus)) != len(self.default_ssus):
            raise ValueError("default_ssus must be unique")
        if self.default_requests_per_npu < SCIENTIFIC_PREFIX_REQUESTS_PER_NPU:
            raise ValueError("default finite backing must preserve the prefix")
        if self.default_measurement_ms <= 0.0:
            raise ValueError("default_measurement_ms must be positive")
        names = tuple(case.name for case in self.cases)
        if not names or len(set(names)) != len(names):
            raise ValueError("case names must be non-empty and unique")
        roles = dict(self.report_roles)
        if len(roles) != len(self.report_roles):
            raise ValueError("report roles must be unique")
        if any(name not in names for name in roles.values()):
            raise ValueError("every report role must reference a case")

    @property
    def case_by_name(self):
        return {case.name: case for case in self.cases}

    @property
    def default_keys(self):
        return {
            (case.name, num_ssu) for case in self.cases for num_ssu in self.default_ssus
        }

    def role_for_case(self, case_name):
        return next(
            (role for role, name in self.report_roles if name == case_name), None
        )


LEGACY32_DEFINITION = ExperimentDefinition(
    key="legacy32",
    experiment_name="32npu_ms_scale_ssu_qos_control_v1",
    num_npu=32,
    n_layers=16,
    batch_size=1,
    default_ssus=(6, 10, 18),
    cases=LEGACY32_CASES,
    default_requests_per_npu=64,
    default_measurement_ms=2_000.0,
)

CONFIRM32_CASES = (
    Case("baseline", "baseline", "baseline"),
    Case("adaptive_t0_i25ms", "factorial", "adaptive", 0.0, 0.0, 25.0),
    Case("adaptive_t0p05_i25ms", "factorial", "adaptive", 0.0, 0.05, 25.0),
    Case("adaptive_t0_i50ms", "factorial", "adaptive", 0.0, 0.0, 50.0),
    Case("adaptive_t0p05_i50ms", "factorial", "adaptive", 0.0, 0.05, 50.0),
)
CONFIRM32_DEFINITION = ExperimentDefinition(
    key="confirm32",
    experiment_name="32npu_threshold_interval_factorial_v1",
    num_npu=32,
    n_layers=16,
    batch_size=1,
    default_ssus=(6, 10, 18),
    cases=CONFIRM32_CASES,
    default_requests_per_npu=128,
    default_measurement_ms=8_000.0,
    report_roles=(
        ("B", "baseline"),
        ("A0", "adaptive_t0_i25ms"),
        ("threshold-only", "adaptive_t0p05_i25ms"),
        ("interval-only", "adaptive_t0_i50ms"),
        ("combined", "adaptive_t0p05_i50ms"),
    ),
)

PFO32_CASES = (
    Case("baseline", "pfo32", "baseline"),
    Case("dedicated_wrr_zero_cir", "pfo32", "dedicated_wrr"),
    Case("adaptive_t0_i25ms", "pfo32", "adaptive", 0.0, 0.0, 25.0),
    Case("adaptive_t0p05_i50ms", "pfo32", "adaptive", 0.0, 0.05, 50.0),
    PFOCase("pfo_floor_t0_i25ms", "pfo32", "pfo", 0.0, 0.0, 25.0, 0.0),
    PFOCase(
        "pfo_floor_t0p05_i25ms",
        "pfo32",
        "pfo",
        0.0,
        0.0,
        25.0,
        0.05,
    ),
    PFOCase(
        "pfo_floor_t0p05_i50ms",
        "pfo32",
        "pfo",
        0.0,
        0.0,
        50.0,
        0.05,
    ),
    PFOCase(
        "pfo_astar_h70",
        "pfo32",
        "pfo",
        0.0,
        0.0,
        25.0,
        0.05,
        0.70,
        SCIENTIFIC_PREFIX_REQUESTS_PER_NPU,
    ),
)
PFO32_DEFINITION = ExperimentDefinition(
    key="pfo32",
    experiment_name="32npu_protected_floor_h70_freeze_v2",
    num_npu=32,
    n_layers=16,
    batch_size=1,
    default_ssus=(6, 10, 18),
    cases=PFO32_CASES,
    default_requests_per_npu=256,
    default_measurement_ms=16_000.0,
    report_roles=(
        ("B", "baseline"),
        ("A0", "adaptive_t0_i25ms"),
        ("combined", "adaptive_t0p05_i50ms"),
        ("F0", "pfo_floor_t0_i25ms"),
        ("F25", "pfo_floor_t0p05_i25ms"),
        ("F50", "pfo_floor_t0p05_i50ms"),
        ("H70", "pfo_astar_h70"),
    ),
)


SELECTED128_SSUS = (8, 12, 16, 20, 24, 40, 72)
SELECTED128_INTERVALS_MS = (25.0, 100.0, 200.0)
SELECTED128_ALPHA1P5_REQUIRED_RATIO = 1.0 / 1.5
SELECTED128_ALPHA1P5_TARGET_RATIO = SELECTED128_ALPHA1P5_REQUIRED_RATIO + 0.02
SELECTED128_CAMPAIGN_NAME = "selected128_alpha_tuned_v2"
SELECTED128_FORMAL_MEASUREMENT_MS = 16_000.0
SELECTED128_FORMAL_MAX_WORKERS = 3
SELECTED128_EXPECTED_RUNTIME_IDENTITY = {
    "python_implementation": "CPython",
    "python_version": "3.14.4",
    "numpy_version": "2.5.2",
    "blas_name": "scipy-openblas",
    "blas_version": "0.3.34.0.0",
    "openblas_configuration": (
        "OpenBLAS 0.3.34.0.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY "
        "SkylakeX MAX_THREADS=64"
    ),
}


def _selected128_definition(
    *,
    alpha1p5_target_ratio=SELECTED128_ALPHA1P5_TARGET_RATIO,
    alpha1p5_spill_threshold=0.75,
):
    """Ten cases supporting seven plotted policies at each SLO threshold."""

    cases = [
        Case("baseline", "baseline", "baseline"),
        Case("layer_once_ttl_0ms", "ttl", "layer_once", 0.0),
        Case("layer_once_ttl_2ms", "ttl", "layer_once", 2.0),
        Case("layer_once_ttl_5ms", "ttl", "layer_once", 5.0),
    ]
    for alpha_name, alpha, target_ratio, required_ratio in (
        (
            "a1p5",
            1.5,
            float(alpha1p5_target_ratio),
            SELECTED128_ALPHA1P5_REQUIRED_RATIO,
        ),
        ("a2", 2.0, 0.52, 0.50),
    ):
        for interval_ms in SELECTED128_INTERVALS_MS:
            interval_name = int(interval_ms)
            cases.append(
                AdaptiveCase(
                    name=f"adaptive_{alpha_name}_t0_i{interval_name}ms",
                    family=f"adaptive_alpha_{alpha_name}",
                    kind="adaptive",
                    pressure_ttl_ms=0.0,
                    cir_write_threshold_gbps=0.0,
                    min_interval_ms=interval_ms,
                    tuning_slo_alpha=alpha,
                    explicit_spill_threshold=(
                        float(alpha1p5_spill_threshold) if alpha == 1.5 else 0.75
                    ),
                    target_ratio=target_ratio,
                    required_ratio=required_ratio,
                    background_reserve_fraction=0.05,
                )
            )
    return ExperimentDefinition(
        key="selected128",
        experiment_name="128npu_selected_cir_control_alpha_tuned_v1",
        num_npu=128,
        n_layers=16,
        batch_size=1,
        default_ssus=SELECTED128_SSUS,
        cases=tuple(cases),
        default_requests_per_npu=128,
        default_measurement_ms=8_000.0,
        require_single_ssu_simulation=True,
    )


def _require_exact_keys(value, expected_keys, context):
    if not isinstance(value, dict) or set(value) != set(expected_keys):
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{context} keys differ: expected {sorted(expected_keys)}, got {actual}"
        )


def validate_selected128_campaign_document(document):
    """Strictly validate every semantic field in the formal v2 campaign."""

    top_keys = {
        "schema_version",
        "campaign",
        "purpose",
        "topology",
        "workload",
        "common_cases",
        "adaptive",
        "metrics",
        "execution",
    }
    _require_exact_keys(document, top_keys, "selected128 campaign")
    if document["schema_version"] != 1:
        raise ValueError("selected128 campaign schema_version must equal 1")
    if document["campaign"] != SELECTED128_CAMPAIGN_NAME:
        raise ValueError("selected128 campaign name is not the frozen v2 name")
    if not isinstance(document["purpose"], str) or not document["purpose"].strip():
        raise ValueError("selected128 campaign purpose must be non-empty")

    topology = document["topology"]
    _require_exact_keys(
        topology,
        {"num_npu", "n_layers", "batch_size", "ssu_counts", "ssu_rationale"},
        "selected128 topology",
    )
    if {
        "num_npu": topology["num_npu"],
        "n_layers": topology["n_layers"],
        "batch_size": topology["batch_size"],
        "ssu_counts": topology["ssu_counts"],
    } != {
        "num_npu": 128,
        "n_layers": 16,
        "batch_size": 1,
        "ssu_counts": list(SELECTED128_SSUS),
    }:
        raise ValueError("selected128 topology differs from the frozen definition")
    if (
        not isinstance(topology["ssu_rationale"], str)
        or not topology["ssu_rationale"].strip()
    ):
        raise ValueError("selected128 SSU rationale must be non-empty")

    workload = document["workload"]
    _require_exact_keys(
        workload,
        {
            "seed",
            "requests_per_npu",
            "scientific_prefix_requests_per_npu",
            "warmup_requests_per_npu",
            "settle_ms",
            "measurement_ms",
            "stationarity_block_ms",
        },
        "selected128 workload",
    )
    expected_workload = {
        "seed": 42,
        "requests_per_npu": 128,
        "scientific_prefix_requests_per_npu": 32,
        "warmup_requests_per_npu": 8,
        "settle_ms": 500.0,
        "measurement_ms": SELECTED128_FORMAL_MEASUREMENT_MS,
        "stationarity_block_ms": 500.0,
    }
    if workload != expected_workload:
        raise ValueError("selected128 workload differs from the frozen formal run")

    expected_common = [
        "baseline",
        "layer_once_ttl_0ms",
        "layer_once_ttl_2ms",
        "layer_once_ttl_5ms",
    ]
    if document["common_cases"] != expected_common:
        raise ValueError("selected128 common case list differs from the frozen set")

    adaptive = document["adaptive"]
    _require_exact_keys(
        adaptive,
        {
            "interval_semantics",
            "intervals_ms",
            "cir_write_threshold_gbps",
            "explicit_spill_threshold",
            "background_reserve_fraction",
            "alpha_1p5_profile",
            "alpha_2_profile",
        },
        "selected128 adaptive",
    )
    if (
        not isinstance(adaptive["interval_semantics"], str)
        or not adaptive["interval_semantics"].strip()
        or adaptive["intervals_ms"] != list(SELECTED128_INTERVALS_MS)
        or adaptive["cir_write_threshold_gbps"] != 0.0
        or adaptive["explicit_spill_threshold"] != 0.75
        or adaptive["background_reserve_fraction"] != 0.05
    ):
        raise ValueError("selected128 Adaptive common settings are not frozen v2")
    profile_keys = {
        "tuning_slo_alpha",
        "required_ratio",
        "target_ratio",
        "target_margin_ratio",
    }
    for name, expected in (
        (
            "alpha_1p5_profile",
            {
                "tuning_slo_alpha": 1.5,
                "required_ratio": SELECTED128_ALPHA1P5_REQUIRED_RATIO,
                "target_ratio": SELECTED128_ALPHA1P5_TARGET_RATIO,
                "target_margin_ratio": 0.02,
            },
        ),
        (
            "alpha_2_profile",
            {
                "tuning_slo_alpha": 2.0,
                "required_ratio": 0.50,
                "target_ratio": 0.52,
                "target_margin_ratio": 0.02,
            },
        ),
    ):
        _require_exact_keys(adaptive[name], profile_keys, f"selected128 {name}")
        if adaptive[name] != expected:
            raise ValueError(f"selected128 {name} differs from the frozen profile")

    metrics = document["metrics"]
    _require_exact_keys(
        metrics,
        {
            "primary_recorded_slo_alpha",
            "postprocessed_slo_alphas",
            "slo_aggregation",
            "plots",
        },
        "selected128 metrics",
    )
    if metrics != {
        "primary_recorded_slo_alpha": 2.0,
        "postprocessed_slo_alphas": [1.5, 2.0],
        "slo_aggregation": "equal NPU",
        "plots": [
            "mean NPU utilization versus SSU count",
            "TTFT SLO attainment at 1.5x ideal versus SSU count",
            "TTFT SLO attainment at 2x ideal versus SSU count",
        ],
    }:
        raise ValueError("selected128 metrics differ from the frozen report contract")

    execution = document["execution"]
    _require_exact_keys(
        execution,
        {
            "multiprocessing_start_method",
            "threads_per_blas_runtime",
            "required_environment",
            "runtime_identity",
            "formal_max_workers_per_shard",
            "formal_concurrent_shards",
            "checkpoint_rule",
        },
        "selected128 execution",
    )
    expected_environment = {name: "1" for name in THREAD_LIMIT_ENVIRONMENT}
    if (
        execution["multiprocessing_start_method"] != "spawn"
        or execution["threads_per_blas_runtime"] != 1
        or execution["required_environment"] != expected_environment
        or execution["runtime_identity"] != SELECTED128_EXPECTED_RUNTIME_IDENTITY
        or execution["formal_max_workers_per_shard"] != SELECTED128_FORMAL_MAX_WORKERS
        or execution["formal_concurrent_shards"] != 2
        or not isinstance(execution["checkpoint_rule"], str)
        or not execution["checkpoint_rule"].strip()
    ):
        raise ValueError("selected128 execution settings differ from frozen v2")
    return document


def _report128_definition(
    *,
    astar_threshold_gbps=0.05,
    astar_interval_ms=25.0,
    lstar_ttl_ms=5.0,
):
    values = (astar_threshold_gbps, astar_interval_ms, lstar_ttl_ms)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("report128 policy parameters must be finite")
    if not 0.0 <= astar_threshold_gbps <= 0.05:
        raise ValueError("A* CIR threshold must be in [0, 0.05] GB/s")
    if astar_interval_ms < 1.0:
        raise ValueError("A* control interval must be at least 1 ms")
    if lstar_ttl_ms < 1.0:
        raise ValueError("L* pressure TTL must be at least 1 ms")
    cases = (
        Case("baseline", "baseline", "baseline"),
        Case("layer_once_ttl_0ms", "ttl", "layer_once", 0.0),
        Case("layer_once_lstar", "ttl", "layer_once", float(lstar_ttl_ms)),
        Case("adaptive_t0_i25ms", "adaptive", "adaptive", 0.0, 0.0, 25.0),
        PFOCase(
            "pfo_astar_h70",
            "pfo",
            "pfo",
            0.0,
            0.0,
            float(astar_interval_ms),
            float(astar_threshold_gbps),
            0.70,
            SCIENTIFIC_PREFIX_REQUESTS_PER_NPU,
        ),
    )
    return ExperimentDefinition(
        key="report128",
        experiment_name="128npu_report_curve_random_real_data_h70_v2",
        num_npu=128,
        n_layers=16,
        batch_size=1,
        default_ssus=(8, 12, 16, 20, 24, 32, 40, 48, 72),
        cases=cases,
        default_requests_per_npu=128,
        default_measurement_ms=8_000.0,
        report_roles=(
            ("B", "baseline"),
            ("L0", "layer_once_ttl_0ms"),
            ("L*", "layer_once_lstar"),
            ("A0", "adaptive_t0_i25ms"),
            ("A*", "pfo_astar_h70"),
        ),
        require_single_ssu_simulation=True,
    )


BASE_DEFINITIONS = {
    definition.key: definition
    for definition in (
        LEGACY32_DEFINITION,
        CONFIRM32_DEFINITION,
        PFO32_DEFINITION,
    )
}

# Compatibility aliases for analysis scripts and old ad-hoc imports. Runtime
# paths below never consult these globals to decide experiment scale.
NUM_NPU = LEGACY32_DEFINITION.num_npu
N_LAYERS = LEGACY32_DEFINITION.n_layers
DEFAULT_SSUS = LEGACY32_DEFINITION.default_ssus
DEFAULT_REQUESTS_PER_NPU = LEGACY32_DEFINITION.default_requests_per_npu
CASES = LEGACY32_DEFINITION.cases
CASE_BY_NAME = LEGACY32_DEFINITION.case_by_name
DEFAULT_KEYS = LEGACY32_DEFINITION.default_keys
EXPLICIT_SPILL_THRESHOLD = LEGACY32_DEFINITION.adaptive.explicit_spill_threshold
TARGET_RATIO = LEGACY32_DEFINITION.adaptive.target_ratio
REQUIRED_RATIO = LEGACY32_DEFINITION.adaptive.required_ratio
BACKGROUND_RESERVE_FRACTION = LEGACY32_DEFINITION.adaptive.background_reserve_fraction


@dataclass(frozen=True)
class RunConfig:
    seed: int
    requests_per_npu: int
    warmup_requests_per_npu: int
    settle_ms: float
    measurement_ms: float
    block_ms: float
    slo_alpha: float = 2.0
    timeline_diagnostics: bool = False
    timeline_dispatch_probe_ms: float = 50.0
    timeline_dispatch_probe_limit: int = 10_000
    campaign_spec_sha256: str | None = None
    calibration_mode: bool = False

    def __post_init__(self):
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must fit an unsigned 64-bit integer")
        if self.requests_per_npu < SCIENTIFIC_PREFIX_REQUESTS_PER_NPU:
            raise ValueError("finite backing must preserve the scientific prefix")
        if not 0 < self.warmup_requests_per_npu < self.requests_per_npu:
            raise ValueError("warmup must be positive and below finite backing")
        durations = (
            self.settle_ms,
            self.measurement_ms,
            self.block_ms,
            self.timeline_dispatch_probe_ms,
        )
        if any(not math.isfinite(value) for value in durations):
            raise ValueError("steady-state durations must be finite")
        if (
            self.settle_ms < 0.0
            or self.measurement_ms <= 0.0
            or self.block_ms <= 0.0
            or self.timeline_dispatch_probe_ms < 0.0
        ):
            raise ValueError("steady-state durations are invalid")
        if not math.isfinite(self.slo_alpha) or self.slo_alpha <= 0.0:
            raise ValueError("slo_alpha must be finite and positive")
        if type(self.calibration_mode) is not bool:
            raise ValueError("calibration_mode must be boolean")
        if type(self.timeline_diagnostics) is not bool:
            raise ValueError("timeline_diagnostics must be boolean")
        if (
            isinstance(self.timeline_dispatch_probe_limit, bool)
            or not isinstance(self.timeline_dispatch_probe_limit, int)
            or self.timeline_dispatch_probe_limit < 0
        ):
            raise ValueError("timeline dispatch probe limit must be nonnegative")
        if self.campaign_spec_sha256 is not None and (
            len(self.campaign_spec_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.campaign_spec_sha256
            )
        ):
            raise ValueError("campaign spec SHA256 must be lowercase hexadecimal")

    def steady_state(self):
        return SteadyStateConfig(
            warmup_requests_per_npu=self.warmup_requests_per_npu,
            settle_ms=self.settle_ms,
            measurement_ms=self.measurement_ms,
            slo_alpha=self.slo_alpha,
            block_ms=self.block_ms,
            timeline_diagnostics=self.timeline_diagnostics,
            timeline_dispatch_probe_ms=self.timeline_dispatch_probe_ms,
            timeline_dispatch_probe_limit=self.timeline_dispatch_probe_limit,
        )


@dataclass(frozen=True)
class WorkerTask:
    definition: ExperimentDefinition
    config: RunConfig
    case: Case
    num_ssu: int
    source_fingerprint: str
    config_fingerprint: str
    mp_start_method: str

    def __post_init__(self):
        if self.num_ssu <= 0:
            raise ValueError("worker SSU count must be positive")
        if self.case.name not in self.definition.case_by_name:
            raise ValueError("worker case is outside the experiment definition")
        if self.definition.case_by_name[self.case.name] != self.case:
            raise ValueError("worker case spec differs from the definition")
        if not self.source_fingerprint or not self.config_fingerprint:
            raise ValueError("worker fingerprints must be non-empty")
        if self.mp_start_method not in ("spawn", "forkserver"):
            raise ValueError("worker start method must be spawn or forkserver")


@dataclass(frozen=True)
class WorkerContext:
    definition: ExperimentDefinition
    schedule: object
    prefix_schedule: object
    config: RunConfig
    source_fingerprint: str
    config_fingerprint: str
    input_authentication: dict
    mp_start_method: str


_WORKER_CONTEXT = None
_WORKER_TABLE = None
_WORKER_WORKLOADS = {}
_WORKER_PREFIX_WORKLOADS = {}
_WORKER_REQUESTS = {}


def _canonical_hash(value, namespace=b""):
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(namespace + encoded).hexdigest()


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"campaign spec contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_campaign_spec(path):
    """Authenticate an optional, externally frozen campaign document."""
    if path is None:
        return None, None
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read campaign spec: {path}") from error
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"campaign spec must be valid duplicate-free UTF-8 JSON: {path}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("campaign spec must be a JSON object")
    authentication = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    return document, authentication


def _transitive_local_sources():
    """Return the complete local Python import closure of this runner."""
    pending = [Path(__file__).name]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(name)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=name)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)
        for module in modules:
            candidate = module.split(".", 1)[0] + ".py"
            if (ROOT / candidate).is_file() and candidate not in seen:
                pending.append(candidate)
    return tuple(sorted(seen))


def _source_manifest():
    files = _transitive_local_sources() + ("data",)
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files
    }


def _source_fingerprint():
    return _canonical_hash(_source_manifest(), b"ms-scale-control-source:v1\0")


def _definition_fingerprint(definition):
    return _canonical_hash(asdict(definition), b"ms-scale-control-definition:v1\0")


def _scientific_input_authentication(input_authentication):
    keys = (
        "source",
        "source_sha256",
        "catalog_hash",
        "table_fingerprint",
        "profile_count",
    )
    if set(keys) - set(input_authentication):
        raise ValueError("authenticated workload metadata is incomplete")
    return {key: input_authentication[key] for key in keys}


def _input_loader_environment(input_authentication):
    keys = ("cache_path", "cache_present", "cache_verified_equal")
    return {key: input_authentication.get(key) for key in keys}


def _seed_manifest(schedule, table):
    """Materialize the authenticated seed input for independent analyzers."""
    assignment_rows = [
        [
            int(request_id),
            int(npu_id),
            int(sequence),
            str(category),
            [int(profile_key[0]), int(profile_key[1])],
        ]
        for request_id, npu_id, sequence, category, profile_key in sorted(
            schedule.assignments,
            key=lambda row: int(row[0]),
        )
    ]
    if [row[0] for row in assignment_rows] != list(range(len(assignment_rows))):
        raise AssertionError("schedule request IDs are not contiguous")
    return {
        **schedule.as_fingerprint_dict(),
        "mode": schedule.mode,
        "seed": int(schedule.seed),
        "num_npu": int(schedule.num_npu),
        "requests_per_npu": int(schedule.requests_per_npu),
        "request_id_formula": "sequence * num_npu + npu_id",
        "catalog_rows": canonical_bw_table(table),
        "assignment_rows": assignment_rows,
    }


def _numpy_blas_identity():
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})
    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "openblas_configuration": blas.get("openblas configuration"),
    }


def current_runtime_merge_identity():
    blas = _numpy_blas_identity()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "blas_name": blas["name"],
        "blas_version": blas["version"],
        "openblas_configuration": blas["openblas_configuration"],
    }


def runtime_merge_identity(runtime):
    blas = runtime.get("numpy_blas_identity", {})
    return {
        "python_implementation": runtime.get("python_implementation"),
        "python_version": runtime.get("python"),
        "numpy_version": runtime.get("numpy"),
        "blas_name": blas.get("name"),
        "blas_version": blas.get("version"),
        "openblas_configuration": blas.get("openblas_configuration"),
    }


def _experiment_spec(definition, schedule, config, input_authentication):
    steady_state = asdict(config)
    campaign_spec_sha256 = steady_state.pop("campaign_spec_sha256")
    # Preserve the authenticated schema/fingerprint of every legacy run when
    # the observer is disabled.  Probe settings have no behavioural meaning in
    # that mode; publish them only for an explicitly diagnostic experiment.
    if not config.timeline_diagnostics:
        steady_state.pop("timeline_diagnostics")
        steady_state.pop("timeline_dispatch_probe_ms")
        steady_state.pop("timeline_dispatch_probe_limit")
    spec = {
        "schema_version": SCHEMA_VERSION,
        "experiment": definition.experiment_name,
        "definition": definition.key,
        "definition_fingerprint": _definition_fingerprint(definition),
        "num_npu": definition.num_npu,
        "scale_semantics": {
            "num_npu": definition.num_npu,
            "naked_128_means": "128 NPU" if definition.num_npu == 128 else None,
            "backing_requests_per_npu": config.requests_per_npu,
            "total_assignment_count": definition.num_npu * config.requests_per_npu,
            "rule": (
                "NPU count is fixed by --definition; --requests-per-npu is "
                "finite input backing only"
            ),
        },
        "campaign_spec_sha256": campaign_spec_sha256,
        "n_layers": definition.n_layers,
        "batch_size": definition.batch_size,
        "default_ssu_list": list(definition.default_ssus),
        "cases": [asdict(case) for case in definition.cases],
        "report_roles": {role: case for role, case in definition.report_roles},
        "workload": {
            "mode": schedule.mode,
            "seed": schedule.seed,
            "requests_per_npu": schedule.requests_per_npu,
            **schedule.as_fingerprint_dict(),
            "prefix_32_assignment_hash": schedule.prefix_32_assignment_hash,
            "full_assignment_hash": schedule.full_assignment_hash,
            "sampling": "IID uniform with replacement over all 84 data profiles",
            "per_npu_streams": "independent and prefix-stable",
            "scientific_prefix_requests_per_npu": (SCIENTIFIC_PREFIX_REQUESTS_PER_NPU),
            "backing_prefix_reason": (
                f"{schedule.requests_per_npu} requests preserve the first-"
                f"{SCIENTIFIC_PREFIX_REQUESTS_PER_NPU} "
                "assignment; any suffix only prevents full-load queue exhaustion "
                "during measurement/drain"
            ),
            "authentication": _scientific_input_authentication(input_authentication),
        },
        "steady_state": steady_state,
        "adaptive": {
            **asdict(definition.adaptive),
            "ssd_cap_gbps": sim.DISK_BW,
            "npu_cap_gbps": sim.NPU_BW_LIMIT,
        },
        "cross_request_layer0_prefetch": True,
        "placement": "token-block ring hash reused across all 16 layers",
        "measurement_cost_scope": "true SSU pressure-table reads and CIR writes",
        "pairing_scope": (
            "all strategies share the exact finite schedule/placement/simulator "
            "input within each SSU; wall-time measurement cohorts may differ in "
            "closed loop and receive a separate membership fingerprint"
        ),
        "diagnostics": {
            "control_evaluations": (
                "reported but not required to equal the threshold=0 closed-loop "
                "anchor because filtered CIR writes can change event timing"
            ),
            "source_stable_false": "invalidates the shard",
        },
        "source_files": list(_transitive_local_sources()) + ["data"],
    }
    if definition.key == "selected128":
        spec["adaptive_case_profiles"] = {
            case.name: {
                "tuning_slo_alpha": case.tuning_slo_alpha,
                **asdict(case.controller_definition()),
            }
            for case in definition.cases
            if isinstance(case, AdaptiveCase)
        }
        spec["thread_limit_environment"] = {
            name: os.environ.get(name) for name in THREAD_LIMIT_ENVIRONMENT
        }
        spec["runtime_identity"] = current_runtime_merge_identity()
    pfo_cases = tuple(case for case in definition.cases if case.kind == "pfo")
    if pfo_cases:
        if not all(isinstance(case, PFOCase) for case in pfo_cases):
            raise AssertionError("every PFO definition case must be a PFOCase")
        spec["pfo"] = {
            "controller": "ProtectedFloorSchemeBController",
            "materialized_allocation_stages": ["selected_protected_floor"],
            "request_pin": "stable (npu_id, request_id)",
            "target_ratio": definition.adaptive.target_ratio,
            "required_ratio": definition.adaptive.required_ratio,
            "background_reserve_fraction_for_admission_only": (
                definition.adaptive.background_reserve_fraction
            ),
            "internal_deadband_gbps_by_case": {
                case.name: case.pfo_internal_deadband_gbps for case in pfo_cases
            },
            "downstream_cir_write_threshold_gbps": (REQUIRED_DOWNSTREAM_DEADBAND_GBPS),
            "path_pressure_reads": 0,
            "trigger_ownership": "CIRControlConfig.min_interval_ms",
            "real_register_order": "all decreases before any increase",
        }
    return spec


def _config_fingerprint(spec):
    return _canonical_hash(spec, b"ms-scale-control-config:v1\0")


def _case_fingerprint(case, num_ssu, source_fingerprint, config_fingerprint):
    return _canonical_hash(
        {
            "case": asdict(case),
            "num_ssu": int(num_ssu),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        b"ms-scale-control-case:v1\0",
    )


def _validate_path_abi(definition):
    if PATH_COUNT != 256 or GROUP_COUNT != 8 or PATHS_PER_GROUP != 32:
        raise AssertionError("QoS Path ABI constants changed")
    if MAX_NPU != 128:
        raise AssertionError("QoS Path ABI MAX_NPU changed")
    if definition.num_npu > MAX_NPU:
        raise AssertionError("experiment exceeds the dedicated Path ABI")
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(definition.num_npu))
    if len(set(paths)) != definition.num_npu:
        raise AssertionError("NPU-dedicated Path IDs are not unique")
    if any(not 0 <= path < PATH_COUNT for path in paths) or 0 in paths:
        raise AssertionError("NPU-dedicated Path IDs violate the reserved Path ABI")
    if definition.num_npu == MAX_NPU:
        expected = {
            group * PATHS_PER_GROUP + offset
            for group in range(GROUP_COUNT)
            for offset in range(16, 32)
        }
        if set(paths) != expected:
            raise AssertionError(
                "128-NPU Path ABI must use offsets 16..31 in every group"
            )
    return {
        "path_count": PATH_COUNT,
        "group_count": GROUP_COUNT,
        "paths_per_group": PATHS_PER_GROUP,
        "max_npu": MAX_NPU,
        "assigned_count": len(paths),
        "assigned_unique": len(set(paths)),
        "assigned_min": min(paths),
        "assigned_max": max(paths),
        "path_zero_reserved": 0 not in paths,
        "assigned_paths_sha256": _canonical_hash(
            list(paths), b"ms-scale-control-path-abi:v1\0"
        ),
    }


def _current_rss_bytes():
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _runtime_provenance(mp_start_method, *, include_process=True):
    payload = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "numpy_blas_identity": _numpy_blas_identity(),
        "platform": platform.platform(),
        "multiprocessing_start_method": mp_start_method,
        "cpu_count": os.cpu_count(),
        "thread_limit_environment": {
            name: os.environ.get(name) for name in THREAD_LIMIT_ENVIRONMENT
        },
    }
    if include_process:
        payload.update(
            {
                "pid": os.getpid(),
                "rss_current_bytes": _current_rss_bytes(),
                "rss_peak_bytes": _peak_rss_bytes(),
            }
        )
    return payload


def _init_worker(
    definition,
    schedule,
    config,
    source_fingerprint,
    config_fingerprint,
    input_authentication,
    mp_start_method,
):
    global _WORKER_CONTEXT, _WORKER_TABLE, _WORKER_WORKLOADS
    global _WORKER_PREFIX_WORKLOADS, _WORKER_REQUESTS
    actual_method = mp.get_start_method(allow_none=False)
    if actual_method != mp_start_method:
        raise RuntimeError(
            f"worker start method {actual_method!r} != {mp_start_method!r}"
        )
    _steady_accounting_numeric_contract()
    _validate_path_abi(definition)
    table, authentication = load_authenticated_bw_table(definition.num_npu)
    if authentication != input_authentication:
        raise RuntimeError("worker authenticated a different workload table")
    if schedule.num_npu != definition.num_npu:
        raise RuntimeError("worker schedule topology differs from definition")
    if schedule.seed != config.seed:
        raise RuntimeError("worker schedule seed differs from run config")
    if schedule.requests_per_npu != config.requests_per_npu:
        raise RuntimeError("worker schedule backing differs from run config")
    if _source_fingerprint() != source_fingerprint:
        raise RuntimeError("worker source fingerprint differs from parent")
    worker_spec = _experiment_spec(definition, schedule, config, authentication)
    if _config_fingerprint(worker_spec) != config_fingerprint:
        raise RuntimeError("worker config fingerprint differs from parent")
    prefix_schedule = build_steady_state_profile_schedule(
        table,
        mode=schedule.mode,
        seed=schedule.seed,
        num_npu=schedule.num_npu,
        requests_per_npu=min(
            SCIENTIFIC_PREFIX_REQUESTS_PER_NPU, schedule.requests_per_npu
        ),
    )
    _WORKER_CONTEXT = WorkerContext(
        definition=definition,
        schedule=schedule,
        prefix_schedule=prefix_schedule,
        config=config,
        source_fingerprint=source_fingerprint,
        config_fingerprint=config_fingerprint,
        input_authentication=dict(authentication),
        mp_start_method=mp_start_method,
    )
    _WORKER_TABLE = table
    _WORKER_WORKLOADS = {}
    _WORKER_PREFIX_WORKLOADS = {}
    _WORKER_REQUESTS = {}


def _workload_and_requests(num_ssu):
    if _WORKER_CONTEXT is None or _WORKER_TABLE is None:
        raise RuntimeError("worker was not initialized")
    definition = _WORKER_CONTEXT.definition
    if num_ssu not in _WORKER_REQUESTS:
        workload = prepare_random_steady_state_workload(
            _WORKER_TABLE,
            schedule=_WORKER_CONTEXT.schedule,
            num_ssu=num_ssu,
            n_layers=definition.n_layers,
        )
        _WORKER_WORKLOADS[num_ssu] = workload
        _WORKER_PREFIX_WORKLOADS[num_ssu] = (
            workload
            if _WORKER_CONTEXT.prefix_schedule.requests_per_npu
            == _WORKER_CONTEXT.schedule.requests_per_npu
            else prepare_random_steady_state_workload(
                _WORKER_TABLE,
                schedule=_WORKER_CONTEXT.prefix_schedule,
                num_ssu=num_ssu,
                n_layers=definition.n_layers,
            )
        )
        _WORKER_REQUESTS[num_ssu] = requests_from_continuous_prefill_workload(workload)
    return (
        _WORKER_WORKLOADS[num_ssu],
        _WORKER_PREFIX_WORKLOADS[num_ssu],
        _WORKER_REQUESTS[num_ssu],
    )


def _common_args(definition, config, num_ssu):
    return {
        "num_npu": definition.num_npu,
        "num_ssu": num_ssu,
        "n_layers": definition.n_layers,
        "batch_size": definition.batch_size,
        "submit_order_seed": config.seed,
        "cross_request_layer0_prefetch": True,
        "steady_state": config.steady_state(),
    }


def _pfo_table_hash(table):
    """Hash one complete CIR table without dropping zero-valued Paths."""

    normalized = tuple(tuple(float(value) for value in row) for row in table)
    return _canonical_hash(normalized, b"pfo-complete-cir-table:v1\0")


def _pfo_evaluation_audit(snapshot, reconciliation, path_by_npu):
    """Build a compact, independently checkable safety record for one update."""

    actual = tuple(
        tuple(map(float, row)) for row in reconciliation.actual_path_cirs_by_ssu
    )
    ideal = tuple(
        tuple(map(float, row)) for row in reconciliation.ideal_path_cirs_by_ssu
    )
    required = tuple(
        tuple(map(float, row)) for row in reconciliation.required_path_cirs_by_ssu
    )
    install = tuple(
        tuple(map(float, row)) for row in reconciliation.install_path_cirs_by_ssu
    )
    snapshot_table = tuple(
        tuple(map(float, row)) for row in snapshot.current_path_cirs_by_ssu
    )
    if snapshot_table != actual:
        raise AssertionError(
            "PFO reconciliation pre-state differs from the DES snapshot"
        )

    path_to_npu = {
        int(path_id): int(npu_id) for npu_id, path_id in enumerate(path_by_npu)
    }
    work = [list(row) for row in actual]
    compact_changes = []
    maximum_ssu_excess = 0.0
    maximum_npu_excess = 0.0
    decreases_finished = False
    seen_entries = set()
    for expected_sequence, change in enumerate(reconciliation.ordered_changes):
        key = (int(change.ssu_id), int(change.path_id))
        if int(change.sequence_index) != expected_sequence or key in seen_entries:
            raise AssertionError(
                "PFO register-write sequence is not unique and contiguous"
            )
        seen_entries.add(key)
        if change.phase == "increase":
            decreases_finished = True
            direction = 1
        elif change.phase == "decrease":
            if decreases_finished:
                raise AssertionError("PFO ordered an unsafe decrease after an increase")
            direction = -1
        else:
            raise AssertionError("PFO emitted an unknown register-write phase")
        old_value = float(work[change.ssu_id][change.path_id])
        if not math.isclose(
            old_value, float(change.old_gbps), rel_tol=0.0, abs_tol=1e-12
        ):
            raise AssertionError(
                "PFO register-write pre-state is not sequentially exact"
            )
        work[change.ssu_id][change.path_id] = float(change.new_gbps)
        maximum_ssu_excess = max(
            maximum_ssu_excess,
            max(math.fsum(row) - float(sim.DISK_BW) for row in work),
        )
        maximum_npu_excess = max(
            maximum_npu_excess,
            max(
                math.fsum(work[ssu_id][path_id] for ssu_id in range(len(work)))
                - float(sim.NPU_BW_LIMIT)
                for path_id in path_by_npu
            ),
        )
        compact_changes.append(
            [
                int(change.ssu_id),
                int(change.path_id),
                direction,
                str(change.reason),
                int(bool(change.safety_forced)),
            ]
        )

    replayed = tuple(tuple(row) for row in work)
    if replayed != install:
        raise AssertionError("PFO ordered-write replay differs from its install table")
    maximum_required_shortfall = max(
        (
            required[ssu_id][path_id] - install[ssu_id][path_id]
            for ssu_id in range(len(install))
            for path_id in range(len(install[ssu_id]))
        ),
        default=0.0,
    )
    active_set_hold_violations = sum(
        (hold.installed_gbps > 0.0) != (hold.ideal_gbps > 0.0)
        and abs(hold.delta_gbps) > sim._EPS
        for hold in reconciliation.deadband_holds
    )
    maximum_held_delta = max(
        (abs(float(hold.delta_gbps)) for hold in reconciliation.deadband_holds),
        default=0.0,
    )
    reason_counts = Counter(change.reason for change in reconciliation.ordered_changes)
    return {
        "time_ms": float(snapshot.time_ms),
        "evaluation": int(snapshot.evaluation),
        "pre_state_hash": _pfo_table_hash(actual),
        "ideal_state_hash": _pfo_table_hash(ideal),
        "required_state_hash": _pfo_table_hash(required),
        "install_state_hash": _pfo_table_hash(install),
        "changed_entries": compact_changes,
        "changed_entries_hash": _canonical_hash(
            compact_changes, b"pfo-ordered-changed-entries:v1\0"
        ),
        "decrease_writes": sum(
            change.phase == "decrease" for change in reconciliation.ordered_changes
        ),
        "increase_writes": sum(
            change.phase == "increase" for change in reconciliation.ordered_changes
        ),
        "change_reason_counts": dict(sorted(reason_counts.items())),
        "ordered_sequence_capacity_safe": (
            maximum_ssu_excess <= 1e-12 and maximum_npu_excess <= 1e-9
        ),
        "maximum_ordered_prefix_ssu_excess_gbps": maximum_ssu_excess,
        "maximum_ordered_prefix_npu_excess_gbps": maximum_npu_excess,
        "maximum_required_floor_shortfall_gbps": maximum_required_shortfall,
        "active_set_hold_violations": int(active_set_hold_violations),
        "maximum_deadband_held_delta_gbps": maximum_held_delta,
        "post_des_state_verified": False,
        "post_des_verification": "pending",
    }


def _pfo_materialization_forecast(case, prefix_statistics, num_ssu):
    if not isinstance(case, PFOCase):
        raise TypeError("PFO forecast requires a PFOCase")
    if case.forecast_hot_fraction is None:
        mask = (True,) * num_ssu
        return mask, {
            "policy": "all_ssus_materialized",
            "frozen_for_measurement": True,
            "materialized_ssu_mask": mask,
            "materialized_ssu_count": num_ssu,
            "cold_ssu_count": 0,
            "forecast_requests_per_npu": None,
            "forecast_input_fingerprint": None,
        }
    if prefix_statistics is None:
        raise ValueError("forecast PFO requires prefix workload statistics")
    demand = tuple(map(float, prefix_statistics["demand_gbps_by_ssu"]))
    if len(demand) != num_ssu:
        raise ValueError("forecast demand does not match the SSU topology")
    forecast = forecast_frozen_ssu_hotspots(
        demand,
        ssu_capacity_gbps=sim.DISK_BW,
        hot_fraction=case.forecast_hot_fraction,
        forecast_requests_per_npu=case.forecast_requests_per_npu,
    )
    mask = forecast.materialized_ssu_mask
    diagnostic = {
        **asdict(forecast),
        "policy": "frozen_manifest_hotspot_v1",
        "frozen_for_measurement": True,
        "materialized_ssu_count": sum(mask),
        "cold_ssu_count": len(mask) - sum(mask),
        "forecast_input_fingerprint": forecast.input_fingerprint,
    }
    return mask, diagnostic


def _enrich_pfo_summary(summary, controller, records, case, num_ssu, forecast):
    """Attach PFO controller-owned write diagnostics to one DES summary."""
    if not isinstance(case, PFOCase):
        raise TypeError("PFO summary enrichment requires a PFOCase")
    start_ms = float(summary["measurement_start_ms"])
    end_ms = float(summary["measurement_end_ms"])
    measurement_records = tuple(
        record for record in records if start_ms <= record["time_ms"] < end_ms
    )

    def scalar_total(selected_records, field):
        return sum(int(record[field]) for record in selected_records)

    def vector_total(selected_records, field):
        return [
            sum(int(record[field][ssu_id]) for record in selected_records)
            for ssu_id in range(num_ssu)
        ]

    def min_gap_ms(selected_records):
        times = [float(record["time_ms"]) for record in selected_records]
        return min(
            (later - earlier for earlier, later in zip(times, times[1:])),
            default=None,
        )

    total_writes_by_ssu = vector_total(records, "writes_by_ssu")
    total_transactions_by_ssu = vector_total(records, "transactions_by_ssu")
    total_decrease_transactions_by_ssu = vector_total(
        records, "decrease_transactions_by_ssu"
    )
    total_increase_transactions_by_ssu = vector_total(
        records, "increase_transactions_by_ssu"
    )
    measurement_writes_by_ssu = vector_total(measurement_records, "writes_by_ssu")
    measurement_transactions_by_ssu = vector_total(
        measurement_records, "transactions_by_ssu"
    )
    measurement_decrease_transactions_by_ssu = vector_total(
        measurement_records, "decrease_transactions_by_ssu"
    )
    measurement_increase_transactions_by_ssu = vector_total(
        measurement_records, "increase_transactions_by_ssu"
    )
    measurement_safety_by_ssu = vector_total(
        measurement_records, "safety_forced_writes_by_ssu"
    )
    all_min_gap = min_gap_ms(records)
    measurement_min_gap = min_gap_ms(measurement_records)
    interval_tolerance = 1e-9
    interval_respected = all(
        float(later["time_ms"]) - float(earlier["time_ms"])
        >= case.min_interval_ms - interval_tolerance
        for earlier, later in zip(records, records[1:])
    )
    if controller.last_plan is None:
        raise AssertionError("PFO controller produced no final plan")
    resolved_mask = tuple(controller.last_plan.materialized_ssu_mask)
    evaluation_ids = [int(record["evaluation"]) for record in records]
    evaluation_sequence_contiguous = bool(evaluation_ids) and evaluation_ids == list(
        range(evaluation_ids[0], evaluation_ids[0] + len(evaluation_ids))
    )
    all_update_safety_gates_pass = bool(records) and all(
        record["post_des_state_verified"]
        and record["ordered_sequence_capacity_safe"]
        and record["cold_ssu_install_zero"]
        and tuple(record["materialized_ssu_mask"]) == resolved_mask
        and int(record["active_set_hold_violations"]) == 0
        and float(record["maximum_required_floor_shortfall_gbps"]) <= 1e-12
        for record in records
    )
    last_reconciliation = controller.last_plan.reconciliation
    last_selected = tuple(
        int(npu_id) for npu_id in controller.last_plan.allocation.selected_npu_ids
    )

    enriched = dict(summary)
    enriched.update(
        {
            "pfo_controller": "ProtectedFloorSchemeBController",
            "pfo_materialized_allocation_stages": ["selected_protected_floor"],
            "pfo_internal_deadband_gbps": case.pfo_internal_deadband_gbps,
            "pfo_required_downstream_deadband_gbps": (
                REQUIRED_DOWNSTREAM_DEADBAND_GBPS
            ),
            "pfo_pressure_reads": int(controller.path_pressure_reads),
            "pfo_path_table_ownership": "exclusive_complete_table",
            "pfo_materialization_forecast": forecast,
            "pfo_materialized_ssu_mask": list(resolved_mask),
            "pfo_materialized_ssu_count": sum(resolved_mask),
            "pfo_total_control_evaluations": len(records),
            "pfo_controller_evaluations": int(controller.evaluations),
            "pfo_controller_decisions": int(controller.decisions),
            "pfo_evaluation_sequence_contiguous": evaluation_sequence_contiguous,
            "pfo_all_update_safety_gates_pass": all_update_safety_gates_pass,
            "pfo_total_planned_cir_path_writes": sum(total_writes_by_ssu),
            "pfo_total_planned_cir_path_writes_by_ssu": total_writes_by_ssu,
            "pfo_total_planned_cir_write_transactions": sum(total_transactions_by_ssu),
            "pfo_total_planned_cir_write_transactions_by_ssu": (
                total_transactions_by_ssu
            ),
            "pfo_total_decrease_phase_transactions": sum(
                total_decrease_transactions_by_ssu
            ),
            "pfo_total_increase_phase_transactions": sum(
                total_increase_transactions_by_ssu
            ),
            "pfo_total_decrease_phase_transactions_by_ssu": (
                total_decrease_transactions_by_ssu
            ),
            "pfo_total_increase_phase_transactions_by_ssu": (
                total_increase_transactions_by_ssu
            ),
            "pfo_total_planned_cir_commits": sum(
                any(record["writes_by_ssu"]) for record in records
            ),
            "pfo_total_safety_forced_cir_path_writes": scalar_total(
                records, "safety_forced_writes"
            ),
            "pfo_measurement_control_evaluations": len(measurement_records),
            "pfo_measurement_planned_cir_path_writes": sum(measurement_writes_by_ssu),
            "pfo_measurement_planned_cir_path_writes_by_ssu": (
                measurement_writes_by_ssu
            ),
            "pfo_measurement_hot_ssu_cir_path_writes": sum(
                value
                for selected, value in zip(resolved_mask, measurement_writes_by_ssu)
                if selected
            ),
            "pfo_measurement_cold_ssu_cir_path_writes": sum(
                value
                for selected, value in zip(resolved_mask, measurement_writes_by_ssu)
                if not selected
            ),
            "pfo_measurement_planned_cir_write_transactions": sum(
                measurement_transactions_by_ssu
            ),
            "pfo_measurement_planned_cir_write_transactions_by_ssu": (
                measurement_transactions_by_ssu
            ),
            "pfo_measurement_decrease_phase_transactions": sum(
                measurement_decrease_transactions_by_ssu
            ),
            "pfo_measurement_increase_phase_transactions": sum(
                measurement_increase_transactions_by_ssu
            ),
            "pfo_measurement_decrease_phase_transactions_by_ssu": (
                measurement_decrease_transactions_by_ssu
            ),
            "pfo_measurement_increase_phase_transactions_by_ssu": (
                measurement_increase_transactions_by_ssu
            ),
            "pfo_measurement_planned_cir_commits": sum(
                any(record["writes_by_ssu"]) for record in measurement_records
            ),
            "pfo_measurement_safety_forced_cir_path_writes": sum(
                measurement_safety_by_ssu
            ),
            "pfo_measurement_safety_forced_cir_path_writes_by_ssu": (
                measurement_safety_by_ssu
            ),
            "pfo_measurement_required_floor_increases": scalar_total(
                measurement_records, "required_floor_increases"
            ),
            "pfo_measurement_capacity_compensation_decreases": scalar_total(
                measurement_records, "capacity_compensation_decreases"
            ),
            "pfo_measurement_deadband_holds": scalar_total(
                measurement_records, "deadband_holds"
            ),
            "pfo_measurement_max_required_floor_shortfall_gbps": max(
                (
                    float(record["maximum_required_floor_shortfall_gbps"])
                    for record in measurement_records
                ),
                default=0.0,
            ),
            "pfo_measurement_max_ordered_prefix_ssu_excess_gbps": max(
                (
                    float(record["maximum_ordered_prefix_ssu_excess_gbps"])
                    for record in measurement_records
                ),
                default=0.0,
            ),
            "pfo_measurement_max_ordered_prefix_npu_excess_gbps": max(
                (
                    float(record["maximum_ordered_prefix_npu_excess_gbps"])
                    for record in measurement_records
                ),
                default=0.0,
            ),
            "pfo_measurement_active_set_hold_violations": scalar_total(
                measurement_records, "active_set_hold_violations"
            ),
            "pfo_control_interval_respected": interval_respected,
            "pfo_min_observed_control_interval_ms": all_min_gap,
            "pfo_measurement_min_observed_control_interval_ms": (measurement_min_gap),
            "pfo_evaluation_diagnostics_hash": _canonical_hash(
                records, b"pfo-evaluation-diagnostics:v1\0"
            ),
            "pfo_measurement_evaluation_diagnostics_hash": _canonical_hash(
                measurement_records, b"pfo-measurement-diagnostics:v1\0"
            ),
            "pfo_evaluation_audit_schema": "pfo_update_audit_v2",
            "pfo_evaluation_audit_records": records,
            "pfo_last_selected_npu_ids": list(last_selected),
            "pfo_last_selected_count": len(last_selected),
            "pfo_last_selected_request_by_npu": [
                [int(npu_id), int(request_id)]
                for npu_id, request_id in sorted(
                    controller.selected_request_by_npu.items()
                )
            ],
            "pfo_last_effective_floor_ratio": (
                controller.last_plan.allocation.effective_floor_ratio
            ),
            "pfo_last_deadband_hold_count": len(last_reconciliation.deadband_holds),
            "pfo_last_required_floor_increase_count": (
                last_reconciliation.required_floor_increase_count
            ),
            "pfo_last_capacity_compensation_decrease_count": (
                last_reconciliation.capacity_compensation_decrease_count
            ),
            "pfo_last_logical_forced_path_ids_by_ssu": [
                list(path_ids)
                for path_ids in last_reconciliation.logical_forced_path_ids_by_ssu
            ],
        }
    )
    return enriched


def _simulate(definition, config, case, num_ssu, requests, prefix_statistics=None):
    common = _common_args(definition, config, num_ssu)
    if case.kind in ("baseline", "layer_once"):
        routing = next(
            spec for spec in routing_strategy_specs() if spec.name == case.kind
        )
        return simulate_continuous_batch(
            requests,
            qos_config=static_qos_config(),
            client_io_config=routing.client_config(),
            pressure_ttl_ms=case.pressure_ttl_ms,
            cir_write_threshold_gbps=0.0,
            **common,
        )
    paths = tuple(cold_start_hybrid_path_id(npu) for npu in range(definition.num_npu))
    if case.kind == "dedicated_wrr":
        return simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * num_ssu
            ),
            npu_dedicated_paths=paths,
            layer0_path_id=None,
            client_io_config=scheme_b_client_config(case.name),
            pressure_ttl_ms=0.0,
            cir_write_threshold_gbps=0.0,
            **common,
        )
    adaptive = (
        case.controller_definition()
        if isinstance(case, AdaptiveCase)
        else definition.adaptive
    )
    if case.kind == "adaptive":
        controller = AdaptiveAdmissionSchemeBControllerV2_1(
            paths,
            explicit_spill_threshold=adaptive.explicit_spill_threshold,
            target_ratio=adaptive.target_ratio,
            required_ratio=adaptive.required_ratio,
            background_reserve_fraction=adaptive.background_reserve_fraction,
            ssd_cap_gbps=sim.DISK_BW,
            npu_cap_gbps=sim.NPU_BW_LIMIT,
            record_diagnostics=config.timeline_diagnostics,
        )
        summary = simulate_continuous_batch(
            requests,
            qos_configs_by_ssu=qos_configs_from_path_cirs(
                ((0.0,) * PATH_COUNT,) * num_ssu
            ),
            npu_dedicated_paths=paths,
            layer0_path_id=None,
            client_io_config=scheme_b_client_config(case.name),
            control=CIRControlConfig(
                callback=controller,
                on_batch_boundary=True,
                min_interval_ms=case.min_interval_ms,
            ),
            pressure_ttl_ms=0.0,
            cir_write_threshold_gbps=case.cir_write_threshold_gbps,
            **common,
        )
        enriched = dict(summary)
        enriched["adaptive_residual_mode_evaluations"] = dict(
            controller.residual_mode_evaluations
        )
        enriched["adaptive_controller_profile"] = {
            "controller": type(controller).__name__,
            "explicit_spill_threshold": controller.explicit_spill_threshold,
            "target_ratio": controller.target_ratio,
            "required_ratio": controller.required_ratio,
            "background_reserve_fraction": controller.background_reserve_fraction,
        }
        enriched["adaptive_last_selected_fraction"] = (
            controller.last_allocation.selected_fraction
            if controller.last_allocation is not None
            else None
        )
        enriched["adaptive_decision_diagnostic_schema"] = (
            "adaptive_admission_decision_v1"
            if config.timeline_diagnostics
            else None
        )
        enriched["adaptive_decision_diagnostics"] = (
            [asdict(record) for record in controller.diagnostics]
            if config.timeline_diagnostics
            else []
        )
        return enriched
    if case.kind != "pfo" or not isinstance(case, PFOCase):
        raise ValueError(f"unknown case kind: {case.kind}")

    materialized_ssu_mask, forecast = _pfo_materialization_forecast(
        case, prefix_statistics, num_ssu
    )
    controller = ProtectedFloorSchemeBController(
        paths,
        target_ratio=adaptive.target_ratio,
        required_ratio=adaptive.required_ratio,
        background_reserve_fraction=adaptive.background_reserve_fraction,
        deadband_gbps=case.pfo_internal_deadband_gbps,
        ssd_cap_gbps=sim.DISK_BW,
        npu_cap_gbps=sim.NPU_BW_LIMIT,
        materialized_ssu_mask=materialized_ssu_mask,
    )
    records = []
    pending_install_table = None

    def pfo_callback(snapshot):
        nonlocal pending_install_table
        if pending_install_table is not None:
            observed = tuple(
                tuple(map(float, row)) for row in snapshot.current_path_cirs_by_ssu
            )
            matches = observed == pending_install_table
            records[-1]["post_des_state_verified"] = matches
            records[-1]["post_des_verification"] = "next_snapshot_exact"
            if not matches:
                raise AssertionError(
                    "PFO decision was not the exact pre-state of the next DES snapshot"
                )
        decision = controller(snapshot)
        plan = controller.last_plan
        if plan is None:
            raise AssertionError("PFO callback returned without a diagnostic plan")
        reconciliation = plan.reconciliation
        record = _pfo_evaluation_audit(snapshot, reconciliation, paths)
        record["materialized_ssu_mask"] = plan.materialized_ssu_mask
        record["cold_ssu_install_zero"] = all(
            all(abs(float(value)) <= sim._EPS for value in row)
            for selected, row in zip(
                plan.materialized_ssu_mask,
                reconciliation.install_path_cirs_by_ssu,
            )
            if not selected
        )
        writes_by_ssu = tuple(
            sum(change.ssu_id == ssu_id for change in reconciliation.ordered_changes)
            for ssu_id in range(num_ssu)
        )
        transactions_by_ssu = tuple(int(value > 0) for value in writes_by_ssu)
        decrease_transactions_by_ssu = tuple(
            int(
                any(
                    change.ssu_id == ssu_id and change.phase == "decrease"
                    for change in reconciliation.ordered_changes
                )
            )
            for ssu_id in range(num_ssu)
        )
        increase_transactions_by_ssu = tuple(
            int(
                any(
                    change.ssu_id == ssu_id and change.phase == "increase"
                    for change in reconciliation.ordered_changes
                )
            )
            for ssu_id in range(num_ssu)
        )
        safety_by_ssu = tuple(
            sum(
                change.ssu_id == ssu_id and change.safety_forced
                for change in reconciliation.ordered_changes
            )
            for ssu_id in range(num_ssu)
        )
        record.update(
            {
                "writes_by_ssu": writes_by_ssu,
                "transactions_by_ssu": transactions_by_ssu,
                "decrease_transactions_by_ssu": decrease_transactions_by_ssu,
                "increase_transactions_by_ssu": increase_transactions_by_ssu,
                "safety_forced_writes_by_ssu": safety_by_ssu,
                "safety_forced_writes": sum(safety_by_ssu),
                "required_floor_increases": reconciliation.required_floor_increase_count,
                "capacity_compensation_decreases": reconciliation.capacity_compensation_decrease_count,
                "deadband_holds": len(reconciliation.deadband_holds),
                "selected_npu_count": len(plan.allocation.selected_npu_ids),
            }
        )
        records.append(record)
        pending_install_table = tuple(
            tuple(map(float, row)) for row in reconciliation.install_path_cirs_by_ssu
        )
        return decision

    summary = simulate_continuous_batch(
        requests,
        qos_configs_by_ssu=qos_configs_from_path_cirs(((0.0,) * PATH_COUNT,) * num_ssu),
        npu_dedicated_paths=paths,
        layer0_path_id=None,
        client_io_config=scheme_b_client_config(case.name),
        control=CIRControlConfig(
            callback=pfo_callback,
            on_batch_boundary=True,
            min_interval_ms=case.min_interval_ms,
        ),
        pressure_ttl_ms=0.0,
        cir_write_threshold_gbps=REQUIRED_DOWNSTREAM_DEADBAND_GBPS,
        **common,
    )
    if records:
        # With downstream threshold zero, _apply_cir_decision either installs the
        # complete table synchronously or raises.  Every earlier decision is also
        # observed exactly in the following public snapshot.
        records[-1]["post_des_state_verified"] = True
        records[-1]["post_des_verification"] = "final_des_accept_zero_threshold"
    return _enrich_pfo_summary(summary, controller, records, case, num_ssu, forecast)


def _validate_summary(definition, case, num_ssu, summary):
    if summary.get("mode") != "steady_state_full_load":
        raise AssertionError("runner returned a non-steady-state summary")
    if (
        int(summary["num_npu"]) != definition.num_npu
        or int(summary["num_ssu"]) != num_ssu
    ):
        raise AssertionError("runner returned the wrong topology")
    invariants = summary.get("invariants", {})
    if not invariants or not all(invariants.values()):
        raise AssertionError(f"invalid steady-state run: {invariants}")
    if not math.isclose(
        float(summary["pressure_ttl_ms"]),
        case.pressure_ttl_ms,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("pressure TTL was not propagated")
    if not math.isclose(
        float(summary["cir_write_threshold_gbps"]),
        case.cir_write_threshold_gbps,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("CIR threshold was not propagated")

    pressure = int(summary["measurement_pressure_reports"])
    cache_hits = int(summary["measurement_pressure_cache_hits"])
    pressure_requests = int(summary["measurement_pressure_requests"])
    writes = int(summary["measurement_cir_path_writes"])
    transactions = int(summary["measurement_cir_write_transactions"])
    commits = int(summary["measurement_cir_commits"])
    evaluations = int(summary["measurement_control_evaluations"])
    if (
        pressure != sum(summary["measurement_pressure_reports_by_ssu"])
        or cache_hits != sum(summary["measurement_pressure_cache_hits_by_ssu"])
        or pressure_requests != pressure + cache_hits
        or writes != sum(summary["measurement_cir_path_writes_by_ssu"])
        or transactions != sum(summary["measurement_cir_write_transactions_by_ssu"])
        or transactions > writes
        or commits > transactions
        or commits > evaluations
    ):
        raise AssertionError("measurement control-plane counters are inconsistent")
    if case.kind == "baseline":
        if pressure_requests or writes or transactions or commits or evaluations:
            raise AssertionError("static baseline used a control-plane operation")
    elif case.kind == "layer_once":
        if pressure <= 0 or pressure_requests <= 0 or writes or evaluations:
            raise AssertionError("layer_once control-plane counters are invalid")
    elif case.kind == "dedicated_wrr":
        if pressure_requests or writes or transactions or commits or evaluations:
            raise AssertionError("static dedicated WRR used a control-plane operation")
    elif case.kind == "adaptive":
        if pressure_requests or evaluations <= 0:
            raise AssertionError("Adaptive control-plane counters are invalid")
        if not math.isclose(
            float(summary["control_min_interval_ms"]),
            case.min_interval_ms,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("Adaptive used the wrong decision interval")
        expected_adaptive = (
            case.controller_definition()
            if isinstance(case, AdaptiveCase)
            else definition.adaptive
        )
        expected_profile = {
            "controller": expected_adaptive.controller,
            "explicit_spill_threshold": expected_adaptive.explicit_spill_threshold,
            "target_ratio": expected_adaptive.target_ratio,
            "required_ratio": expected_adaptive.required_ratio,
            "background_reserve_fraction": (
                expected_adaptive.background_reserve_fraction
            ),
        }
        if summary.get("adaptive_controller_profile") != expected_profile:
            raise AssertionError("Adaptive used a controller profile unlike its case")
    elif case.kind == "pfo":
        if not isinstance(case, PFOCase):
            raise AssertionError("PFO summary was paired with a non-PFOCase")
        if pressure_requests or pressure or cache_hits or evaluations <= 0:
            raise AssertionError("PFO control-plane counters are invalid")
        if not math.isclose(
            float(summary["control_min_interval_ms"]),
            case.min_interval_ms,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("PFO used the wrong external decision interval")
        if not math.isclose(
            float(summary["cir_write_threshold_gbps"]),
            REQUIRED_DOWNSTREAM_DEADBAND_GBPS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("PFO downstream CIR threshold must be zero")
        if not math.isclose(
            float(summary["pfo_internal_deadband_gbps"]),
            case.pfo_internal_deadband_gbps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError("PFO internal deadband was not propagated")
        if (
            float(summary["pfo_required_downstream_deadband_gbps"])
            != REQUIRED_DOWNSTREAM_DEADBAND_GBPS
            or int(summary["pfo_pressure_reads"]) != 0
            or summary["pfo_materialized_allocation_stages"]
            != ["selected_protected_floor"]
            or summary["pfo_path_table_ownership"] != "exclusive_complete_table"
        ):
            raise AssertionError("PFO policy contract was not preserved")

        materialized_mask = tuple(summary["pfo_materialized_ssu_mask"])
        forecast = summary.get("pfo_materialization_forecast")
        if (
            len(materialized_mask) != num_ssu
            or any(not isinstance(value, bool) for value in materialized_mask)
            or int(summary["pfo_materialized_ssu_count"]) != sum(materialized_mask)
            or not isinstance(forecast, dict)
            or tuple(forecast["materialized_ssu_mask"]) != materialized_mask
            or int(forecast["materialized_ssu_count"]) != sum(materialized_mask)
            or int(forecast["cold_ssu_count"]) != num_ssu - sum(materialized_mask)
            or not forecast["frozen_for_measurement"]
        ):
            raise AssertionError("PFO SSU materialization forecast is inconsistent")
        if case.forecast_hot_fraction is None:
            if (
                forecast["policy"] != "all_ssus_materialized"
                or not all(materialized_mask)
                or forecast["forecast_input_fingerprint"] is not None
            ):
                raise AssertionError("plain PFO unexpectedly masked an SSU")
        elif (
            forecast["policy"] != "frozen_manifest_hotspot_v1"
            or not math.isclose(
                float(forecast["hot_fraction"]),
                case.forecast_hot_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or int(forecast["forecast_requests_per_npu"])
            != case.forecast_requests_per_npu
            or not isinstance(forecast["forecast_input_fingerprint"], str)
            or len(forecast["forecast_input_fingerprint"]) != 64
        ):
            raise AssertionError("forecast-gated PFO used the wrong frozen forecast")

        audit_records = summary.get("pfo_evaluation_audit_records")
        total_evaluations = int(summary["control_evaluations"])
        if (
            summary.get("pfo_evaluation_audit_schema") != "pfo_update_audit_v2"
            or not isinstance(audit_records, list)
            or len(audit_records) != total_evaluations
            or int(summary["pfo_controller_evaluations"]) != total_evaluations
            or int(summary["pfo_controller_decisions"]) != total_evaluations
            or not summary["pfo_evaluation_sequence_contiguous"]
            or not summary["pfo_all_update_safety_gates_pass"]
        ):
            raise AssertionError("PFO per-evaluation audit coverage is incomplete")
        audit_times = [float(record["time_ms"]) for record in audit_records]
        if any(later < earlier for earlier, later in zip(audit_times, audit_times[1:])):
            raise AssertionError("PFO audit timestamps are not monotonic")
        if any(
            not record["post_des_state_verified"]
            or record["post_des_verification"]
            not in ("next_snapshot_exact", "final_des_accept_zero_threshold")
            or not record["ordered_sequence_capacity_safe"]
            or not record["cold_ssu_install_zero"]
            or tuple(record["materialized_ssu_mask"]) != materialized_mask
            or int(record["active_set_hold_violations"]) != 0
            or float(record["maximum_required_floor_shortfall_gbps"]) > 1e-12
            or float(record["maximum_ordered_prefix_ssu_excess_gbps"]) > 1e-12
            or float(record["maximum_ordered_prefix_npu_excess_gbps"]) > 1e-9
            or any(
                not isinstance(record[field], str) or len(record[field]) != 64
                for field in (
                    "pre_state_hash",
                    "ideal_state_hash",
                    "required_state_hash",
                    "install_state_hash",
                    "changed_entries_hash",
                )
            )
            or len(record["changed_entries"]) != sum(record["writes_by_ssu"])
            or len({(entry[0], entry[1]) for entry in record["changed_entries"]})
            != len(record["changed_entries"])
            for record in audit_records
        ):
            raise AssertionError("PFO per-evaluation update safety audit failed")
        if int(summary["pfo_measurement_cold_ssu_cir_path_writes"]) != 0 or int(
            summary["pfo_measurement_hot_ssu_cir_path_writes"]
        ) + int(summary["pfo_measurement_cold_ssu_cir_path_writes"]) != int(
            summary["pfo_measurement_planned_cir_path_writes"]
        ):
            raise AssertionError(
                "forecast-gated PFO wrote a cold SSU during measurement"
            )

        exact_counter_pairs = (
            ("pfo_total_control_evaluations", "control_evaluations"),
            ("pfo_total_planned_cir_path_writes", "cir_path_writes"),
            (
                "pfo_total_planned_cir_write_transactions",
                "cir_write_transactions",
            ),
            ("pfo_total_planned_cir_commits", "cir_commits"),
            (
                "pfo_measurement_control_evaluations",
                "measurement_control_evaluations",
            ),
            (
                "pfo_measurement_planned_cir_path_writes",
                "measurement_cir_path_writes",
            ),
            (
                "pfo_measurement_planned_cir_write_transactions",
                "measurement_cir_write_transactions",
            ),
            (
                "pfo_measurement_planned_cir_commits",
                "measurement_cir_commits",
            ),
        )
        if any(
            int(summary[pfo_field]) != int(summary[des_field])
            for pfo_field, des_field in exact_counter_pairs
        ):
            raise AssertionError("PFO planned writes differ from installed DES writes")
        exact_vector_pairs = (
            (
                "pfo_total_planned_cir_path_writes_by_ssu",
                "cir_path_writes_by_ssu",
            ),
            (
                "pfo_total_planned_cir_write_transactions_by_ssu",
                "cir_write_transactions_by_ssu",
            ),
            (
                "pfo_measurement_planned_cir_path_writes_by_ssu",
                "measurement_cir_path_writes_by_ssu",
            ),
            (
                "pfo_measurement_planned_cir_write_transactions_by_ssu",
                "measurement_cir_write_transactions_by_ssu",
            ),
        )
        if any(
            list(map(int, summary[pfo_field])) != list(map(int, summary[des_field]))
            for pfo_field, des_field in exact_vector_pairs
        ):
            raise AssertionError("PFO per-SSU planned writes differ from the DES")
        if (
            sum(int(record["decrease_writes"]) for record in audit_records)
            + sum(int(record["increase_writes"]) for record in audit_records)
            != int(summary["pfo_total_planned_cir_path_writes"])
            or int(summary["pfo_total_decrease_phase_transactions"]) < 0
            or int(summary["pfo_total_increase_phase_transactions"]) < 0
        ):
            raise AssertionError("PFO write-phase diagnostics are inconsistent")
        safety_by_ssu = list(
            map(
                int,
                summary["pfo_measurement_safety_forced_cir_path_writes_by_ssu"],
            )
        )
        if (
            len(safety_by_ssu) != num_ssu
            or sum(safety_by_ssu)
            != int(summary["pfo_measurement_safety_forced_cir_path_writes"])
            or int(summary["pfo_measurement_safety_forced_cir_path_writes"])
            != int(summary["pfo_measurement_required_floor_increases"])
            + int(summary["pfo_measurement_capacity_compensation_decreases"])
            or int(summary["pfo_measurement_safety_forced_cir_path_writes"]) > writes
        ):
            raise AssertionError("PFO safety-forced write diagnostics are inconsistent")
        if not summary["pfo_control_interval_respected"]:
            raise AssertionError("PFO control callback violated its minimum interval")
        min_gap = summary["pfo_min_observed_control_interval_ms"]
        if min_gap is not None and float(min_gap) < case.min_interval_ms - 1e-9:
            raise AssertionError("PFO observed control interval is too short")
        maximum_measurement_evaluations = (
            math.ceil(float(summary["measurement_duration_ms"]) / case.min_interval_ms)
            + 1
        )
        if evaluations > maximum_measurement_evaluations:
            raise AssertionError("PFO measurement control frequency is too high")
        last_selected = list(map(int, summary["pfo_last_selected_npu_ids"]))
        last_request_map = [
            (int(npu_id), int(request_id))
            for npu_id, request_id in summary["pfo_last_selected_request_by_npu"]
        ]
        if (
            len(last_selected) != int(summary["pfo_last_selected_count"])
            or len(set(last_selected)) != len(last_selected)
            or any(not 0 <= npu_id < definition.num_npu for npu_id in last_selected)
            or sorted(last_selected) != sorted(npu_id for npu_id, _ in last_request_map)
            or len(summary["pfo_last_logical_forced_path_ids_by_ssu"]) != num_ssu
        ):
            raise AssertionError("PFO final selection diagnostics are inconsistent")
        if case.pfo_internal_deadband_gbps == 0.0 and (
            int(summary["pfo_measurement_deadband_holds"])
            or int(summary["pfo_measurement_capacity_compensation_decreases"])
        ):
            raise AssertionError("zero-deadband PFO unexpectedly held a CIR change")
    else:
        raise AssertionError(f"unvalidated case kind: {case.kind}")
    if summary.get("timeline_diagnostics_enabled"):
        boundaries = summary.get("measurement_stationarity_boundaries", [])
        if not boundaries or any("timeline" not in row for row in boundaries):
            raise AssertionError("timeline diagnostics are missing a boundary")
        if case.kind == "adaptive":
            records = summary.get("adaptive_decision_diagnostics", [])
            if (
                summary.get("adaptive_decision_diagnostic_schema")
                != "adaptive_admission_decision_v1"
                or len(records) != int(summary["control_evaluations"])
                or any(
                    record.get("snapshot_evaluation") != index
                    for index, record in enumerate(records, start=1)
                )
            ):
                raise AssertionError("Adaptive decision timeline is incomplete")
    return summary


def _profile_key_by_request(schedule):
    return {
        int(request_id): tuple(profile_key)
        for request_id, _, _, _, profile_key in schedule.assignments
    }


def _slo_group(values):
    passed = sum(bool(value) for value in values)
    return {
        "count": len(values),
        "passed": passed,
        "slo_attainment": passed / len(values) if values else None,
    }


def _cohort_profile_metrics(definition, summary, schedule, table):
    by_request = _profile_key_by_request(schedule)
    category = defaultdict(list)
    profile_rows = defaultdict(list)
    demand_bins = defaultdict(list)
    profiles_by_npu = defaultdict(list)
    bins = (
        ("le_10", -math.inf, 10.0),
        ("gt_10_le_20", 10.0, 20.0),
        ("gt_20_le_40", 20.0, 40.0),
        ("gt_40_le_50", 40.0, 50.0),
        ("gt_50_le_80", 50.0, 80.0),
        ("gt_80", 80.0, math.inf),
    )
    for row in summary["request_rows"]:
        profile_key = by_request[int(row["request_id"])]
        outcome = bool(row["slo_met"])
        demand = float(table[profile_key][0])
        category[str(row["category"])].append(outcome)
        profile_rows[profile_key].append(outcome)
        profiles_by_npu[int(row["npu_id"])].append(profile_key)
        for name, lower, upper in bins:
            if lower < demand <= upper:
                demand_bins[name].append(outcome)
                break
    by_profile = {}
    for profile_key in sorted(table):
        values = profile_rows.get(profile_key, ())
        by_profile[f"{profile_key[0]},{profile_key[1]}"] = {
            "raw_demand_gbps": float(table[profile_key][0]),
            **_slo_group(values),
        }
    per_npu_demand = []
    per_npu_ms_per_gb = []
    for npu_id in range(definition.num_npu):
        profiles = profiles_by_npu[npu_id]
        compute_s = sum(float(table[key][1]) for key in profiles) / 1e6
        kv_gb = sum(float(table[key][3]) for key in profiles)
        per_npu_demand.append(kv_gb / compute_s)
        per_npu_ms_per_gb.append(1000.0 * compute_s / kv_gb)
    demand_mean = statistics.mean(per_npu_demand)
    ms_per_gb_mean = statistics.mean(per_npu_ms_per_gb)
    return {
        "category": {
            key: _slo_group(values) for key, values in sorted(category.items())
        },
        "raw_demand_bins": {
            name: _slo_group(demand_bins.get(name, ())) for name, _, _ in bins
        },
        "profile": by_profile,
        "profiles_observed": sum(bool(values) for values in profile_rows.values()),
        "realized_cohort": {
            "request_count": sum(len(values) for values in profiles_by_npu.values()),
            "per_npu_raw_demand_gbps": {
                "min": min(per_npu_demand),
                "max": max(per_npu_demand),
                "mean": demand_mean,
                "coefficient_of_variation": (
                    statistics.pstdev(per_npu_demand) / demand_mean
                    if demand_mean > 0.0
                    else 0.0
                ),
            },
            "per_npu_ms_per_gb": {
                "min": min(per_npu_ms_per_gb),
                "max": max(per_npu_ms_per_gb),
                "mean": ms_per_gb_mean,
                "spread": max(per_npu_ms_per_gb) - min(per_npu_ms_per_gb),
            },
            "fleet_raw_demand_gbps": sum(per_npu_demand),
        },
    }


def _measurement_cohort_fingerprint(summary, schedule):
    by_request = _profile_key_by_request(schedule)
    rows = [
        [
            int(row["request_id"]),
            int(row["npu_id"]),
            int(row["sequence"]),
            list(by_request[int(row["request_id"])]),
        ]
        for row in summary["request_rows"]
    ]
    return _canonical_hash(rows, b"ms-scale-control-measurement-cohort:v1\0")


def _stationarity_diagnostics(summary):
    blocks = summary["measurement_blocks"]
    utils = [float(block["npu_utilization"]) for block in blocks]
    requests = [int(block["request_count"]) for block in blocks]
    start = summary["measurement_ssd_outstanding_blocks_at_start"]
    end = summary["measurement_ssd_outstanding_blocks_at_end"]
    drift = [int(b) - int(a) for a, b in zip(start, end)]
    return {
        "block_npu_utilizations": utils,
        "block_request_counts": requests,
        "block_utilization_range": max(utils) - min(utils),
        "first_last_utilization_delta": utils[-1] - utils[0],
        "outstanding_blocks_drift_by_ssu": drift,
        "fleet_outstanding_blocks_drift": sum(drift),
    }


def _run_case(task):
    if not isinstance(task, WorkerTask):
        raise TypeError("worker task must be a WorkerTask")
    if _WORKER_CONTEXT is None:
        raise RuntimeError("worker context is absent")
    definition = task.definition
    config = task.config
    case = task.case
    num_ssu = task.num_ssu
    source_fingerprint = task.source_fingerprint
    config_fingerprint = task.config_fingerprint
    expected = (
        definition,
        config,
        source_fingerprint,
        config_fingerprint,
        task.mp_start_method,
    )
    actual = (
        _WORKER_CONTEXT.definition,
        _WORKER_CONTEXT.config,
        _WORKER_CONTEXT.source_fingerprint,
        _WORKER_CONTEXT.config_fingerprint,
        _WORKER_CONTEXT.mp_start_method,
    )
    if actual != expected:
        raise RuntimeError("worker task differs from its immutable initializer")
    if case.name not in definition.case_by_name:
        raise RuntimeError("worker task contains a case outside its definition")
    if definition.case_by_name[case.name] != case:
        raise RuntimeError("worker task case spec differs from its definition")
    started = time.perf_counter()
    finished = threading.Event()

    def heartbeat():
        while not finished.wait(60.0):
            print(
                f"RUNNING {case.name} ssu={num_ssu}: "
                f"wall={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        workload, prefix_workload, requests = _workload_and_requests(num_ssu)
        summary = _validate_summary(
            definition,
            case,
            num_ssu,
            _simulate(
                definition,
                config,
                case,
                num_ssu,
                requests,
                prefix_workload.statistics,
            ),
        )
        inputs = {
            **_WORKER_CONTEXT.schedule.as_fingerprint_dict(),
            "prefix_32_assignment": workload.statistics["prefix_32_assignment_hash"],
            "full_assignment": workload.statistics["full_assignment_hash"],
            "workload": workload.workload_hash,
            "placement": workload.placement_hash,
            "trace": workload.trace_hash,
            "simulator": summary["input_fingerprint"],
        }
        return {
            "status": "ok",
            "case": case.name,
            "family": case.family,
            "kind": case.kind,
            "role": definition.role_for_case(case.name),
            "num_ssu": num_ssu,
            "num_npu": definition.num_npu,
            "backing_requests_per_npu": config.requests_per_npu,
            "definition": definition.key,
            "definition_fingerprint": _definition_fingerprint(definition),
            "case_spec": asdict(case),
            "source_fingerprint": source_fingerprint,
            "config_fingerprint": config_fingerprint,
            "campaign_spec_sha256": config.campaign_spec_sha256,
            "case_fingerprint": _case_fingerprint(
                case, num_ssu, source_fingerprint, config_fingerprint
            ),
            "input_fingerprints": inputs,
            "workload_statistics": workload.statistics,
            "prefix_32_workload_statistics": prefix_workload.statistics,
            "prefix_32_materialized_fingerprints": {
                "workload": prefix_workload.workload_hash,
                "placement": prefix_workload.placement_hash,
                "trace": prefix_workload.trace_hash,
            },
            "cohort_profile_metrics": _cohort_profile_metrics(
                definition,
                summary,
                _WORKER_CONTEXT.schedule,
                _WORKER_TABLE,
            ),
            "measurement_cohort_fingerprint": _measurement_cohort_fingerprint(
                summary, _WORKER_CONTEXT.schedule
            ),
            "stationarity_diagnostics": _stationarity_diagnostics(summary),
            "runtime": _runtime_provenance(task.mp_start_method),
            "wall_time_s": time.perf_counter() - started,
            "steady_summary": summary,
        }
    finally:
        finished.set()


def _worker_probe():
    if _WORKER_CONTEXT is None or _WORKER_TABLE is None:
        raise RuntimeError("worker probe ran without initializer context")
    return {
        "definition": _WORKER_CONTEXT.definition.key,
        "num_npu": _WORKER_CONTEXT.definition.num_npu,
        "schedule_num_npu": _WORKER_CONTEXT.schedule.num_npu,
        "seed": _WORKER_CONTEXT.schedule.seed,
        "requests_per_npu": _WORKER_CONTEXT.schedule.requests_per_npu,
        "source_fingerprint": _WORKER_CONTEXT.source_fingerprint,
        "config_fingerprint": _WORKER_CONTEXT.config_fingerprint,
        "campaign_spec_sha256": _WORKER_CONTEXT.config.campaign_spec_sha256,
        "input_authentication": _scientific_input_authentication(
            _WORKER_CONTEXT.input_authentication
        ),
        "input_loader_environment": _input_loader_environment(
            _WORKER_CONTEXT.input_authentication
        ),
        "path_abi": _validate_path_abi(_WORKER_CONTEXT.definition),
        "steady_accounting_numeric_contract": _steady_accounting_numeric_contract(),
        "runtime": _runtime_provenance(_WORKER_CONTEXT.mp_start_method),
    }


def _row_key(row):
    return str(row["case"]), int(row["num_ssu"])


def _validate_cached(
    payload,
    definition,
    source_fingerprint,
    spec,
    config_fingerprint,
    campaign_spec_authentication,
    mp_start_method,
):
    if not payload:
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("results"), list)
    ):
        return {}
    if (
        payload.get("source_fingerprint") != source_fingerprint
        or payload.get("ending_source_fingerprint") != source_fingerprint
        or not payload.get("source_stable_during_run")
        or payload.get("experiment_spec") != spec
        or payload.get("config_fingerprint") != config_fingerprint
        or payload.get("ending_config_fingerprint") != config_fingerprint
        or not payload.get("config_stable_during_run")
        or payload.get("campaign_spec_sha256") != spec.get("campaign_spec_sha256")
        or payload.get("num_npu") != spec.get("num_npu")
        or payload.get("backing_requests_per_npu")
        != spec.get("scale_semantics", {}).get("backing_requests_per_npu")
        or payload.get("campaign_spec_authentication") != campaign_spec_authentication
        or payload.get("ending_campaign_spec_authentication")
        != campaign_spec_authentication
        or not payload.get("campaign_spec_stable_during_run")
        or payload.get("execution", {}).get("multiprocessing_start_method")
        != mp_start_method
    ):
        return {}
    rows = {}
    required_inputs = {
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment",
        "full_assignment",
        "workload",
        "placement",
        "trace",
        "simulator",
    }
    expected_schedule_inputs = {
        name: spec["workload"][name]
        for name in ("catalog", "recipe", "schedule", "assignment")
    }
    expected_schedule_inputs.update(
        {
            "prefix_32_assignment": spec["workload"]["prefix_32_assignment_hash"],
            "full_assignment": spec["workload"]["full_assignment_hash"],
        }
    )
    for row in payload.get("results", ()):
        try:
            key = _row_key(row)
            case = definition.case_by_name[key[0]]
        except (KeyError, TypeError, ValueError):
            return {}
        if (
            key in rows
            or type(row.get("case")) is not str
            or type(row.get("num_ssu")) is not int
            or row["num_ssu"] <= 0
            or row.get("status") != "ok"
            or row.get("family") != case.family
            or row.get("kind") != case.kind
            or row.get("definition") != definition.key
            or row.get("definition_fingerprint") != _definition_fingerprint(definition)
            or row.get("num_npu") != definition.num_npu
            or row.get("backing_requests_per_npu")
            != spec.get("scale_semantics", {}).get("backing_requests_per_npu")
            or row.get("role") != definition.role_for_case(case.name)
            or row.get("case_spec") != asdict(case)
            or row.get("source_fingerprint") != source_fingerprint
            or row.get("config_fingerprint") != config_fingerprint
            or row.get("campaign_spec_sha256") != spec.get("campaign_spec_sha256")
            or row.get("case_fingerprint")
            != _case_fingerprint(case, key[1], source_fingerprint, config_fingerprint)
            or set(row.get("input_fingerprints", ())) != required_inputs
            or any(
                row.get("input_fingerprints", {}).get(name) != expected
                for name, expected in expected_schedule_inputs.items()
            )
            or row.get("input_fingerprints", {}).get("simulator")
            != row.get("steady_summary", {}).get("input_fingerprint")
            or not isinstance(row.get("workload_statistics"), dict)
            or not isinstance(row.get("prefix_32_workload_statistics"), dict)
            or not isinstance(row.get("prefix_32_materialized_fingerprints"), dict)
            or not isinstance(row.get("measurement_cohort_fingerprint"), str)
            or row.get("runtime", {}).get("multiprocessing_start_method")
            != mp_start_method
        ):
            return {}
        _validate_summary(definition, case, key[1], row["steady_summary"])
        rows[key] = row
    return rows


def _pairing_audit(rows, selected_ssus):
    fields = (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment",
        "full_assignment",
        "workload",
        "placement",
        "trace",
        "simulator",
    )
    audit = {}
    for num_ssu in selected_ssus:
        group = [row for row in rows.values() if row["num_ssu"] == num_ssu]
        audit[str(num_ssu)] = {
            "cases": sorted(row["case"] for row in group),
            "has_rows": bool(group),
            "all_available_rows_paired": not group
            or all(
                len({row["input_fingerprints"][field] for row in group}) == 1
                for field in fields
            )
            and len({_canonical_hash(row["workload_statistics"]) for row in group}) <= 1
            and len(
                {_canonical_hash(row["prefix_32_workload_statistics"]) for row in group}
            )
            <= 1
            and len(
                {
                    _canonical_hash(row["prefix_32_materialized_fingerprints"])
                    for row in group
                }
            )
            <= 1,
        }
    return audit


def _build_payload(
    *,
    rows,
    definition,
    config,
    schedule,
    spec,
    source_fingerprint,
    config_fingerprint,
    campaign_spec_path,
    campaign_spec_authentication,
    input_authentication,
    seed_manifest,
    mp_start_method,
    max_workers,
    selected_keys,
    selected_ssus,
):
    ending_source = _source_fingerprint()
    (
        _ending_campaign_document,
        ending_campaign_spec_authentication,
    ) = _load_campaign_spec(campaign_spec_path)
    ending_config = _config_fingerprint(
        _experiment_spec(definition, schedule, config, input_authentication)
    )
    ordered = [rows[key] for key in sorted(rows, key=lambda key: (key[1], key[0]))]
    source_stable = ending_source == source_fingerprint
    config_stable = ending_config == config_fingerprint
    campaign_spec_stable = (
        ending_campaign_spec_authentication == campaign_spec_authentication
        and config.campaign_spec_sha256
        == (
            None
            if campaign_spec_authentication is None
            else campaign_spec_authentication["sha256"]
        )
    )
    audited_ssus = tuple(sorted({num_ssu for _, num_ssu in rows}.union(selected_ssus)))
    pairing_audit = _pairing_audit(rows, audited_ssus)
    pairing_valid = all(
        entry["all_available_rows_paired"] for entry in pairing_audit.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": source_stable
        and config_stable
        and campaign_spec_stable
        and pairing_valid
        and definition.default_keys.issubset(rows),
        "selected_complete": source_stable
        and config_stable
        and campaign_spec_stable
        and pairing_valid
        and all(key in rows for key in selected_keys),
        "source_stable_during_run": source_stable,
        "config_stable_during_run": config_stable,
        "campaign_spec_stable_during_run": campaign_spec_stable,
        "source_fingerprint": source_fingerprint,
        "ending_source_fingerprint": ending_source,
        "source_manifest": _source_manifest(),
        "definition": definition.key,
        "definition_fingerprint": _definition_fingerprint(definition),
        "num_npu": definition.num_npu,
        "backing_requests_per_npu": config.requests_per_npu,
        "total_assignment_count": definition.num_npu * config.requests_per_npu,
        "path_abi": _validate_path_abi(definition),
        "input_authentication": _scientific_input_authentication(input_authentication),
        "input_loader_environment": _input_loader_environment(input_authentication),
        "runtime": _runtime_provenance(mp_start_method),
        "execution": {
            "multiprocessing_start_method": mp_start_method,
            "requested_max_workers": int(max_workers),
            "single_ssu_process_pool_required": (
                definition.require_single_ssu_simulation
            ),
        },
        "config_fingerprint": config_fingerprint,
        "ending_config_fingerprint": ending_config,
        "campaign_spec_sha256": config.campaign_spec_sha256,
        "campaign_spec_authentication": campaign_spec_authentication,
        "ending_campaign_spec_authentication": (ending_campaign_spec_authentication),
        "experiment_spec": spec,
        "selected_ssus": list(selected_ssus),
        "selected_cases": sorted({name for name, _num_ssu in selected_keys}),
        "selected_keys": [list(key) for key in sorted(selected_keys)],
        "schedule_metadata": seed_manifest,
        "pairing_audit": pairing_audit,
        "results": ordered,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def acquire_output_lock(path, *, owner="runner"):
    """Hold an advisory lock for an output's complete process lifetime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + f".{owner}.lock")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(
            f"output is already owned by another process: {path}"
        ) from error
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
    handle.flush()
    return handle


def validate_selected128_formal_payload(
    payload,
    *,
    expected_ssu,
    expected_campaign_sha256,
):
    """Validate a completed formal selected128 shard without rerunning it."""

    definition = _selected128_definition()
    expected_cases = definition.case_by_name
    expected_names = set(expected_cases)
    expected_keys = {(name, int(expected_ssu)) for name in expected_names}

    def require(condition, message):
        if not condition:
            raise ValueError(f"selected128 formal shard: {message}")

    require(isinstance(payload, dict), "payload must be an object")
    require(payload.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    require(payload.get("definition") == "selected128", "definition mismatch")
    require(payload.get("num_npu") == 128, "NPU topology mismatch")
    require(
        payload.get("backing_requests_per_npu") == 128,
        "top-level backing request count mismatch",
    )
    require(
        payload.get("total_assignment_count") == 128 * 128,
        "top-level assignment count mismatch",
    )
    require(payload.get("selected_complete") is True, "selected shard incomplete")
    require(payload.get("source_stable_during_run") is True, "source changed")
    require(payload.get("config_stable_during_run") is True, "config changed")
    require(
        payload.get("campaign_spec_stable_during_run") is True,
        "campaign changed",
    )
    source = payload.get("source_fingerprint")
    config = payload.get("config_fingerprint")
    require(isinstance(source, str) and len(source) == 64, "source hash malformed")
    require(
        source == payload.get("ending_source_fingerprint"),
        "ending source differs",
    )
    require(isinstance(config, str) and len(config) == 64, "config hash malformed")
    require(
        config == payload.get("ending_config_fingerprint"),
        "ending config differs",
    )
    require(
        payload.get("campaign_spec_sha256") == expected_campaign_sha256,
        "campaign hash mismatch",
    )
    for field in (
        "campaign_spec_authentication",
        "ending_campaign_spec_authentication",
    ):
        authentication = payload.get(field)
        require(isinstance(authentication, dict), f"{field} missing")
        require(
            authentication.get("sha256") == expected_campaign_sha256,
            f"{field} hash mismatch",
        )

    require(
        payload.get("selected_ssus") == [int(expected_ssu)],
        "selected SSU mismatch",
    )
    require(
        set(payload.get("selected_cases", ())) == expected_names
        and len(payload.get("selected_cases", ())) == len(expected_names),
        "selected cases mismatch",
    )
    selected_keys = payload.get("selected_keys")
    require(isinstance(selected_keys, list), "selected keys missing")
    normalized_selected = {
        (str(key[0]), int(key[1]))
        for key in selected_keys
        if isinstance(key, list) and len(key) == 2
    }
    require(
        len(normalized_selected) == len(selected_keys)
        and normalized_selected == expected_keys,
        "selected key matrix mismatch",
    )

    runtime = payload.get("runtime")
    require(isinstance(runtime, dict), "parent runtime missing")
    require(
        runtime_merge_identity(runtime) == SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "parent runtime identity mismatch",
    )
    expected_thread_environment = {name: "1" for name in THREAD_LIMIT_ENVIRONMENT}
    require(
        runtime.get("thread_limit_environment") == expected_thread_environment,
        "parent thread environment mismatch",
    )
    execution = payload.get("execution", {})
    require(
        execution.get("multiprocessing_start_method") == "spawn",
        "multiprocessing method mismatch",
    )
    require(
        type(execution.get("requested_max_workers")) is int
        and 1 <= execution["requested_max_workers"] <= SELECTED128_FORMAL_MAX_WORKERS,
        "worker limit mismatch",
    )
    require(
        execution.get("single_ssu_process_pool_required") is True,
        "single-SSU process pool contract absent",
    )

    spec = payload.get("experiment_spec")
    require(isinstance(spec, dict), "experiment spec missing")
    require(
        spec.get("experiment") == definition.experiment_name,
        "spec experiment mismatch",
    )
    require(spec.get("definition") == "selected128", "spec definition mismatch")
    require(spec.get("num_npu") == 128, "spec NPU mismatch")
    require(spec.get("n_layers") == 16, "spec layer count mismatch")
    require(spec.get("batch_size") == 1, "spec batch size mismatch")
    require(
        spec.get("campaign_spec_sha256") == expected_campaign_sha256,
        "spec campaign mismatch",
    )
    require(
        spec.get("thread_limit_environment") == expected_thread_environment,
        "spec thread environment mismatch",
    )
    require(
        spec.get("runtime_identity") == SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "spec runtime identity mismatch",
    )
    scale = spec.get("scale_semantics")
    require(isinstance(scale, dict), "spec scale semantics missing")
    require(scale.get("num_npu") == 128, "spec scale NPU mismatch")
    require(
        scale.get("naked_128_means") == "128 NPU",
        "spec scale interpretation mismatch",
    )
    require(
        scale.get("backing_requests_per_npu") == 128,
        "spec backing request count mismatch",
    )
    require(
        scale.get("total_assignment_count") == 128 * 128,
        "spec assignment count mismatch",
    )
    workload = spec.get("workload")
    require(isinstance(workload, dict), "spec workload missing")
    require(workload.get("seed") == 42, "spec workload seed mismatch")
    require(
        workload.get("requests_per_npu") == 128,
        "spec workload backing request count mismatch",
    )
    require(config == _config_fingerprint(spec), "config fingerprint is not derived")
    expected_definition_fingerprint = _definition_fingerprint(definition)
    require(
        payload.get("definition_fingerprint") == expected_definition_fingerprint,
        "definition fingerprint mismatch",
    )
    require(
        spec.get("definition_fingerprint") == expected_definition_fingerprint,
        "spec definition fingerprint mismatch",
    )
    source_manifest = payload.get("source_manifest")
    require(isinstance(source_manifest, dict) and source_manifest, "source manifest")
    require(
        all(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for name, digest in source_manifest.items()
        ),
        "source manifest hashes malformed",
    )
    require(
        source == _canonical_hash(source_manifest, b"ms-scale-control-source:v1\0"),
        "source fingerprint is not derived from its manifest",
    )
    steady = spec.get("steady_state", {})
    require(
        steady
        == {
            "seed": 42,
            "requests_per_npu": 128,
            "warmup_requests_per_npu": 8,
            "settle_ms": 500.0,
            "measurement_ms": SELECTED128_FORMAL_MEASUREMENT_MS,
            "block_ms": 500.0,
            "slo_alpha": 2.0,
            "calibration_mode": False,
        },
        "formal steady-state config mismatch",
    )
    require(
        spec.get("default_ssu_list") == list(SELECTED128_SSUS),
        "spec SSU list mismatch",
    )
    spec_cases = spec.get("cases")
    require(
        isinstance(spec_cases, list)
        and {case.get("name") for case in spec_cases} == expected_names
        and len(spec_cases) == len(expected_names),
        "spec case list mismatch",
    )
    require(
        {case["name"]: case for case in spec_cases}
        == {name: asdict(case) for name, case in expected_cases.items()},
        "spec case parameters mismatch",
    )

    pairing = payload.get("pairing_audit", {}).get(str(expected_ssu), {})
    require(pairing.get("all_available_rows_paired") is True, "inputs unpaired")
    require(
        set(pairing.get("cases", ())) == expected_names
        and len(pairing.get("cases", ())) == len(expected_names),
        "pairing case coverage mismatch",
    )

    rows = payload.get("results")
    require(isinstance(rows, list), "results missing")
    normalized_rows = {
        (row.get("case"), row.get("num_ssu")) for row in rows if isinstance(row, dict)
    }
    require(
        len(rows) == len(expected_keys)
        and len(normalized_rows) == len(rows)
        and normalized_rows == expected_keys,
        "results are missing, duplicated, or unexpected",
    )
    fingerprint_fields = (
        "catalog",
        "recipe",
        "schedule",
        "assignment",
        "prefix_32_assignment",
        "full_assignment",
        "workload",
        "placement",
        "trace",
        "simulator",
    )
    for row in rows:
        inputs = row.get("input_fingerprints")
        require(
            isinstance(inputs, dict) and set(inputs) == set(fingerprint_fields),
            f"{row.get('case')} input fingerprint fields",
        )
        require(
            all(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in inputs.values()
            ),
            f"{row.get('case')} input fingerprint digest",
        )
    for field in fingerprint_fields:
        require(
            len({row["input_fingerprints"][field] for row in rows}) == 1,
            f"input fingerprint {field} is not paired",
        )
    for row in rows:
        name = row["case"]
        require(row.get("status") == "ok", f"{name} status is not ok")
        require(row.get("definition") == "selected128", f"{name} definition")
        require(
            row.get("definition_fingerprint") == expected_definition_fingerprint,
            f"{name} definition fingerprint",
        )
        require(row.get("num_npu") == 128, f"{name} NPU count")
        require(
            row.get("backing_requests_per_npu") == 128,
            f"{name} backing request count",
        )
        require(row.get("source_fingerprint") == source, f"{name} source")
        require(row.get("config_fingerprint") == config, f"{name} config")
        require(
            row.get("campaign_spec_sha256") == expected_campaign_sha256,
            f"{name} campaign",
        )
        require(
            row.get("case_spec") == asdict(expected_cases[name]),
            f"{name} case spec",
        )
        require(
            row.get("case_fingerprint")
            == _case_fingerprint(expected_cases[name], expected_ssu, source, config),
            f"{name} case fingerprint",
        )
        row_runtime = row.get("runtime")
        require(isinstance(row_runtime, dict), f"{name} runtime missing")
        require(
            runtime_merge_identity(row_runtime)
            == SELECTED128_EXPECTED_RUNTIME_IDENTITY,
            f"{name} runtime identity",
        )
        require(
            row_runtime.get("thread_limit_environment") == expected_thread_environment,
            f"{name} thread environment",
        )
        summary = row.get("steady_summary")
        require(isinstance(summary, dict), f"{name} summary missing")
        invariants = summary.get("invariants")
        require(
            isinstance(invariants, dict)
            and invariants
            and all(value is True for value in invariants.values()),
            f"{name} simulator invariant failed",
        )
        require(summary.get("slo_alpha") == 2.0, f"{name} primary alpha")
        require(
            summary.get("measurement_duration_ms") == SELECTED128_FORMAL_MEASUREMENT_MS,
            f"{name} measurement duration",
        )
        request_rows = summary.get("request_rows")
        request_counts = summary.get("request_counts_by_npu")
        require(
            isinstance(request_rows, list)
            and len(request_rows) == summary.get("measurement_request_count"),
            f"{name} request row coverage",
        )
        require(
            isinstance(request_counts, list)
            and len(request_counts) == 128
            and all(type(value) is int and value > 0 for value in request_counts)
            and sum(request_counts) == len(request_rows),
            f"{name} per-NPU request coverage",
        )
        require(
            {request.get("npu_id") for request in request_rows} == set(range(128)),
            f"{name} request NPU coverage",
        )
        if isinstance(expected_cases[name], AdaptiveCase):
            expected_adaptive = expected_cases[name].controller_definition()
            require(
                summary.get("adaptive_controller_profile")
                == {
                    "controller": expected_adaptive.controller,
                    "explicit_spill_threshold": (
                        expected_adaptive.explicit_spill_threshold
                    ),
                    "target_ratio": expected_adaptive.target_ratio,
                    "required_ratio": expected_adaptive.required_ratio,
                    "background_reserve_fraction": (
                        expected_adaptive.background_reserve_fraction
                    ),
                },
                f"{name} actual Adaptive profile",
            )
    return {
        "source_fingerprint": source,
        "config_fingerprint": config,
        "campaign_spec_sha256": expected_campaign_sha256,
        "runtime_identity": SELECTED128_EXPECTED_RUNTIME_IDENTITY,
        "result_count": len(rows),
    }


def _abort_process_pool(executor, futures):
    """Cancel queued work, terminate live workers, and wait for pool teardown."""
    for future in futures:
        future.cancel()

    # Python 3.14 exposes the desired termination operation publicly.  Python
    # 3.10 has the same multiprocessing.Process objects but no public pool-wide
    # method, so use the narrow compatibility fallback before a blocking join.
    terminate_workers = getattr(executor, "terminate_workers", None)
    if callable(terminate_workers):
        terminate_workers()
    else:
        processes = tuple((getattr(executor, "_processes", None) or {}).values())
        for process in processes:
            if process.is_alive():
                process.terminate()
    executor.shutdown(wait=True, cancel_futures=True)


def _select_cases(definition, case_names, families):
    case_by_name = definition.case_by_name
    unknown_cases = sorted(set(case_names) - set(case_by_name))
    if unknown_cases:
        raise ValueError(
            f"unknown cases for {definition.key}: {', '.join(unknown_cases)}"
        )
    valid_families = {case.family for case in definition.cases}
    if definition.key == "legacy32":
        valid_families.update(("threshold", "interval"))
    unknown_families = sorted(set(families) - valid_families)
    if unknown_families:
        raise ValueError(
            f"unknown families for {definition.key}: " + ", ".join(unknown_families)
        )
    selected = set(case_names)
    for family in families:
        if definition.key == "legacy32" and family == "threshold":
            selected.add("adaptive_t0_i25ms")
            selected.update(
                case.name for case in definition.cases if case.family == "threshold"
            )
        elif definition.key == "legacy32" and family == "interval":
            selected.add("adaptive_t0_i25ms")
            selected.update(
                case.name for case in definition.cases if case.family == "interval"
            )
        else:
            selected.update(
                case.name for case in definition.cases if case.family == family
            )
    if not selected:
        selected.update(case_by_name)
    return tuple(case_by_name[name] for name in sorted(selected))


def _preflight(definition, table, schedule, ssus):
    result = {}
    prefix_schedule = build_steady_state_profile_schedule(
        table,
        mode=schedule.mode,
        seed=schedule.seed,
        num_npu=schedule.num_npu,
        requests_per_npu=min(
            SCIENTIFIC_PREFIX_REQUESTS_PER_NPU, schedule.requests_per_npu
        ),
    )
    for num_ssu in ssus:
        workload = prepare_random_steady_state_workload(
            table,
            schedule=schedule,
            num_ssu=num_ssu,
            n_layers=definition.n_layers,
        )
        prefix_workload = (
            workload
            if prefix_schedule.requests_per_npu == schedule.requests_per_npu
            else prepare_random_steady_state_workload(
                table,
                schedule=prefix_schedule,
                num_ssu=num_ssu,
                n_layers=definition.n_layers,
            )
        )
        stats = workload.statistics
        prefix_stats = prefix_workload.statistics
        result[str(num_ssu)] = {
            "workload_hash": workload.workload_hash,
            "placement_hash": workload.placement_hash,
            "trace_hash": workload.trace_hash,
            "prefix_32_assignment_hash": stats["prefix_32_assignment_hash"],
            "full_assignment_hash": stats["full_assignment_hash"],
            "per_npu_demand_gbps_min": stats["per_npu_raw_demand_gbps"]["min"],
            "per_npu_demand_gbps_max": stats["per_npu_raw_demand_gbps"]["max"],
            "per_npu_demand_gbps_mean": stats["per_npu_demand_gbps_mean"],
            "per_npu_demand_gbps_cv": stats["per_npu_demand_gbps_cv"],
            "per_npu_ms_per_gb_min": stats["per_npu_ms_per_gb"]["min"],
            "per_npu_ms_per_gb_max": stats["per_npu_ms_per_gb"]["max"],
            "per_npu_ms_per_gb_spread_pct": stats["per_npu_ms_per_gb"][
                "spread_pct_of_mean"
            ],
            "fleet_demand_gbps": stats["fleet_demand_gbps"],
            "capacity_knee_ssu": stats["capacity_knee_ssu"],
            "demand_gbps_by_ssu": stats["demand_gbps_by_ssu"],
            "max_ssu_demand_gbps": stats["max_ssu_demand_gbps"],
            "ssu_over_40_count": stats["ssu_over_40_count"],
            "fleet_category_counts": stats["fleet_category_counts"],
            "profiles_used": stats["profiles_used"],
            "prefix_32": {
                "workload_hash": prefix_workload.workload_hash,
                "placement_hash": prefix_workload.placement_hash,
                "trace_hash": prefix_workload.trace_hash,
                "per_npu_raw_demand_gbps": prefix_stats["per_npu_raw_demand_gbps"],
                "per_npu_ms_per_gb": prefix_stats["per_npu_ms_per_gb"],
                "fleet_demand_gbps": prefix_stats["fleet_demand_gbps"],
                "capacity_knee_ssu": prefix_stats["capacity_knee_ssu"],
                "demand_gbps_by_ssu": prefix_stats["demand_gbps_by_ssu"],
                "max_ssu_demand_gbps": prefix_stats["max_ssu_demand_gbps"],
                "ssu_over_40_count": prefix_stats["ssu_over_40_count"],
                "fleet_category_counts": prefix_stats["fleet_category_counts_all"],
                "profiles_used": prefix_stats["profiles_used"],
            },
        }
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definition",
        choices=("legacy32", "confirm32", "pfo32", "report128", "selected128"),
        default=None,
        help=(
            "required immutable experiment topology/policy definition; it is "
            "never inferred from requests-per-npu"
        ),
    )
    parser.add_argument("--astar-threshold-gbps", type=float)
    parser.add_argument("--astar-interval-ms", type=float)
    parser.add_argument("--lstar-ttl-ms", type=float)
    parser.add_argument("--selected-alpha1p5-target-ratio", type=float)
    parser.add_argument("--selected-alpha1p5-spill-threshold", type=float)
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--campaign-spec",
        type=Path,
        help=(
            "optional externally frozen JSON campaign document; its exact raw "
            "SHA256 is bound into the config, every result row, and the shard"
        ),
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--family", action="append")
    parser.add_argument("--ssu", action="append", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    available_methods = tuple(
        method
        for method in ("spawn", "forkserver")
        if method in mp.get_all_start_methods()
    )
    parser.add_argument(
        "--mp-start-method",
        choices=available_methods,
        default="spawn",
        help="explicit safe worker import mode; implicit fork is prohibited",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--requests-per-npu", type=int)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--settle-ms", type=float, default=500.0)
    parser.add_argument("--measurement-ms", type=float)
    parser.add_argument("--block-ms", type=float, default=500.0)
    parser.add_argument(
        "--timeline-diagnostics",
        action="store_true",
        help=(
            "record read-only 0.5s NPU/SSU demand, service, queue, Path, "
            "control, and bounded dispatch-replay diagnostics"
        ),
    )
    parser.add_argument(
        "--timeline-dispatch-probe-ms",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--timeline-dispatch-probe-limit",
        type=int,
        default=10_000,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--worker-probe-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="explicit non-formal selected128 tuning run; never accepted as report data",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-definitions", action="store_true")
    return parser.parse_args(argv)


def _definition_from_args(args):
    report128_custom = (
        args.astar_threshold_gbps,
        args.astar_interval_ms,
        args.lstar_ttl_ms,
    )
    selected128_custom = (
        args.selected_alpha1p5_target_ratio,
        args.selected_alpha1p5_spill_threshold,
    )
    if args.definition == "selected128":
        if any(value is not None for value in report128_custom):
            raise ValueError("A*/L* overrides are valid only for report128")
        return _selected128_definition(
            alpha1p5_target_ratio=(
                SELECTED128_ALPHA1P5_TARGET_RATIO
                if args.selected_alpha1p5_target_ratio is None
                else args.selected_alpha1p5_target_ratio
            ),
            alpha1p5_spill_threshold=(
                0.75
                if args.selected_alpha1p5_spill_threshold is None
                else args.selected_alpha1p5_spill_threshold
            ),
        )
    if any(value is not None for value in selected128_custom):
        raise ValueError("selected Adaptive overrides require --definition selected128")
    if args.definition != "report128":
        if any(value is not None for value in report128_custom):
            raise ValueError("A*/L* overrides are valid only for report128")
        return BASE_DEFINITIONS[args.definition]
    return _report128_definition(
        astar_threshold_gbps=(
            0.05 if args.astar_threshold_gbps is None else args.astar_threshold_gbps
        ),
        astar_interval_ms=(
            25.0 if args.astar_interval_ms is None else args.astar_interval_ms
        ),
        lstar_ttl_ms=(5.0 if args.lstar_ttl_ms is None else args.lstar_ttl_ms),
    )


def _pool_initargs(
    definition,
    schedule,
    config,
    source_fingerprint,
    config_fingerprint,
    input_authentication,
    mp_start_method,
):
    return (
        definition,
        schedule,
        config,
        source_fingerprint,
        config_fingerprint,
        input_authentication,
        mp_start_method,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.list_definitions:
        print("\n".join(("legacy32", "confirm32", "pfo32", "report128", "selected128")))
        return 0
    if args.definition is None:
        raise ValueError(
            "--definition is required: choose report128 explicitly for 128 NPUs; "
            "--requests-per-npu controls backing only"
        )
    definition = _definition_from_args(args)
    campaign_document, campaign_spec_authentication = _load_campaign_spec(
        args.campaign_spec
    )
    if args.list_cases:
        print("\n".join(definition.case_by_name))
        return 0
    if definition.key == "selected128":
        invalid_thread_limits = {
            name: os.environ.get(name)
            for name in THREAD_LIMIT_ENVIRONMENT
            if os.environ.get(name) != "1"
        }
        if invalid_thread_limits:
            raise ValueError(
                "selected128 requires every BLAS/OpenMP thread limit to equal 1: "
                f"{invalid_thread_limits}"
            )
        actual_runtime_identity = current_runtime_merge_identity()
        if actual_runtime_identity != SELECTED128_EXPECTED_RUNTIME_IDENTITY:
            raise ValueError(
                "selected128 runtime identity differs from the frozen environment: "
                f"{actual_runtime_identity}"
            )
    elif args.calibration:
        raise ValueError("--calibration is valid only with --definition selected128")
    non_simulation = args.preflight_only or args.dry_run or args.worker_probe_only
    if not non_simulation and args.output is None:
        raise ValueError("--output is required for simulation shards")
    if args.rerun and args.fresh:
        raise ValueError("--rerun and --fresh are mutually exclusive")
    if non_simulation and (args.rerun or args.fresh):
        raise ValueError("--rerun/--fresh apply only to simulation shards")
    if args.max_workers <= 0 or args.warmup_requests <= 0:
        raise ValueError("workers and warmup-requests must be positive")
    if (
        definition.key == "selected128"
        and args.max_workers > SELECTED128_FORMAL_MAX_WORKERS
    ):
        raise ValueError(
            f"selected128 max-workers cannot exceed {SELECTED128_FORMAL_MAX_WORKERS}"
        )
    requests_per_npu = (
        definition.default_requests_per_npu
        if args.requests_per_npu is None
        else args.requests_per_npu
    )
    measurement_ms = (
        definition.default_measurement_ms
        if args.measurement_ms is None
        else args.measurement_ms
    )
    if requests_per_npu < SCIENTIFIC_PREFIX_REQUESTS_PER_NPU:
        raise ValueError("requests-per-npu must retain at least the first 32 requests")
    if requests_per_npu <= args.warmup_requests:
        raise ValueError("requests-per-npu must exceed warmup-requests")
    if not 0 <= args.seed < 2**64:
        raise ValueError("seed must fit an unsigned 64-bit integer")
    if (
        not all(
            math.isfinite(value)
            for value in (args.settle_ms, measurement_ms, args.block_ms)
        )
        or args.settle_ms < 0.0
        or measurement_ms <= 0.0
        or args.block_ms <= 0.0
    ):
        raise ValueError("steady-state durations are invalid")
    selected_ssus = tuple(sorted(set(args.ssu or definition.default_ssus)))
    if any(num_ssu <= 0 for num_ssu in selected_ssus):
        raise ValueError("SSU counts must be positive")
    if definition.key in ("report128", "selected128") and any(
        num_ssu not in definition.default_ssus for num_ssu in selected_ssus
    ):
        raise ValueError(
            f"{definition.key} SSU points are frozen at "
            + ",".join(map(str, definition.default_ssus))
        )
    if (
        definition.require_single_ssu_simulation
        and not non_simulation
        and len(selected_ssus) != 1
    ):
        raise ValueError(
            f"{definition.key} simulation requires exactly one --ssu per invocation "
            "to bound worker RSS and preserve one paired cell per process-pool "
            "lifetime"
        )
    selected_cases = _select_cases(
        definition, tuple(args.case or ()), tuple(args.family or ())
    )
    config = RunConfig(
        seed=args.seed,
        requests_per_npu=requests_per_npu,
        warmup_requests_per_npu=args.warmup_requests,
        settle_ms=args.settle_ms,
        measurement_ms=measurement_ms,
        block_ms=args.block_ms,
        timeline_diagnostics=args.timeline_diagnostics,
        timeline_dispatch_probe_ms=args.timeline_dispatch_probe_ms,
        timeline_dispatch_probe_limit=args.timeline_dispatch_probe_limit,
        campaign_spec_sha256=(
            None
            if campaign_spec_authentication is None
            else campaign_spec_authentication["sha256"]
        ),
        calibration_mode=args.calibration,
    )
    if definition.key == "selected128":
        selected_names = {case.name for case in selected_cases}
        override_requested = any(
            value is not None
            for value in (
                args.selected_alpha1p5_target_ratio,
                args.selected_alpha1p5_spill_threshold,
            )
        )
        if args.calibration:
            allowed_calibration_cases = {
                f"adaptive_a1p5_t0_i{int(interval)}ms"
                for interval in SELECTED128_INTERVALS_MS
            }
            if args.campaign_spec is not None:
                raise ValueError(
                    "selected128 calibration cannot bind the formal campaign"
                )
            if args.fresh:
                raise ValueError("selected128 calibration forbids --fresh")
            if args.seed not in (43, 44):
                raise ValueError(
                    "selected128 calibration must use held-out seed 43 or 44"
                )
            if not args.case or args.family:
                raise ValueError(
                    "selected128 calibration requires explicit --case entries and no family"
                )
            if not selected_names <= allowed_calibration_cases:
                raise ValueError(
                    "selected128 calibration accepts only alpha1.5 Adaptive"
                )
            if (
                config.requests_per_npu != 128
                or config.warmup_requests_per_npu != 8
                or config.settle_ms != 500.0
                or not 0.0 < config.measurement_ms <= 4000.0
                or config.block_ms != 500.0
            ):
                raise ValueError(
                    "selected128 calibration durations/backing are outside bounds"
                )
        else:
            if args.campaign_spec is None or campaign_document is None:
                raise ValueError("formal selected128 requires --campaign-spec")
            validate_selected128_campaign_document(campaign_document)
            if override_requested:
                raise ValueError("formal selected128 forbids Adaptive CLI overrides")
            if args.rerun or args.fresh:
                raise ValueError(
                    "formal selected128 is resume-only; rerun/fresh are forbidden"
                )
            if args.mp_start_method != "spawn":
                raise ValueError("formal selected128 requires spawn")
            if selected_names != set(definition.case_by_name):
                raise ValueError("formal selected128 requires all ten simulation cases")
            if (
                config.seed != 42
                or config.requests_per_npu != 128
                or config.warmup_requests_per_npu != 8
                or config.settle_ms != 500.0
                or config.measurement_ms != SELECTED128_FORMAL_MEASUREMENT_MS
                or config.block_ms != 500.0
                or config.timeline_diagnostics
                or config.slo_alpha != 2.0
            ):
                raise ValueError("formal selected128 run config differs from campaign")
    output_lock = None
    if not non_simulation:
        output_lock = acquire_output_lock(args.output, owner="runner")
    path_abi = _validate_path_abi(definition)
    with redirect_stdout(io.StringIO()):
        table, input_authentication = load_authenticated_bw_table(definition.num_npu)
    schedule = build_steady_state_profile_schedule(
        table,
        mode=IID_UNIFORM_PROFILE_CATALOG_V1,
        seed=config.seed,
        num_npu=definition.num_npu,
        requests_per_npu=config.requests_per_npu,
    )
    seed_manifest = _seed_manifest(schedule, table)
    spec = _experiment_spec(definition, schedule, config, input_authentication)
    source_fingerprint = _source_fingerprint()
    config_fingerprint = _config_fingerprint(spec)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "source_fingerprint": source_fingerprint,
                    "source_manifest": _source_manifest(),
                    "config_fingerprint": config_fingerprint,
                    "campaign_spec_sha256": config.campaign_spec_sha256,
                    "campaign_spec_authentication": (campaign_spec_authentication),
                    "experiment_spec": spec,
                    "input_authentication": _scientific_input_authentication(
                        input_authentication
                    ),
                    "input_loader_environment": _input_loader_environment(
                        input_authentication
                    ),
                    "path_abi": path_abi,
                    "runtime": _runtime_provenance(args.mp_start_method),
                    "preflight": _preflight(definition, table, schedule, selected_ssus),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    selected_keys = {
        (case.name, num_ssu) for case in selected_cases for num_ssu in selected_ssus
    }
    selection = {
        "definition": definition.key,
        "definition_fingerprint": _definition_fingerprint(definition),
        "num_npu": definition.num_npu,
        "seed": config.seed,
        "backing_requests_per_npu": config.requests_per_npu,
        "total_assignment_count": definition.num_npu * config.requests_per_npu,
        "warmup_requests_per_npu": config.warmup_requests_per_npu,
        "settle_ms": config.settle_ms,
        "measurement_ms": config.measurement_ms,
        "stationarity_block_ms": config.block_ms,
        "selected_ssus": list(selected_ssus),
        "selected_cases": [case.name for case in selected_cases],
        "selected_keys": [
            list(key) for key in sorted(selected_keys, key=lambda key: (key[1], key[0]))
        ],
        "mp_start_method": args.mp_start_method,
        "max_workers": args.max_workers,
        "campaign_spec_sha256": config.campaign_spec_sha256,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    **selection,
                    "would_simulate": False,
                    "source_fingerprint": source_fingerprint,
                    "source_manifest": _source_manifest(),
                    "config_fingerprint": config_fingerprint,
                    "campaign_spec_sha256": config.campaign_spec_sha256,
                    "campaign_spec_authentication": (campaign_spec_authentication),
                    "experiment_spec": spec,
                    "input_authentication": _scientific_input_authentication(
                        input_authentication
                    ),
                    "input_loader_environment": _input_loader_environment(
                        input_authentication
                    ),
                    "path_abi": path_abi,
                    "runtime": _runtime_provenance(args.mp_start_method),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    initargs = _pool_initargs(
        definition,
        schedule,
        config,
        source_fingerprint,
        config_fingerprint,
        input_authentication,
        args.mp_start_method,
    )
    pool_context = mp.get_context(args.mp_start_method)
    if args.worker_probe_only:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=pool_context,
            initializer=_init_worker,
            initargs=initargs,
        ) as executor:
            probe = executor.submit(_worker_probe).result()
        ending_source = _source_fingerprint()
        if ending_source != source_fingerprint:
            raise RuntimeError("source changed during worker probe")
        print(
            json.dumps(
                {
                    "mode": "worker_probe",
                    **selection,
                    "source_stable": True,
                    "parent_runtime": _runtime_provenance(args.mp_start_method),
                    "worker": probe,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "event": "simulation_shard_start",
                **selection,
                "output": str(args.output),
                "source_fingerprint": source_fingerprint,
                "config_fingerprint": config_fingerprint,
                "campaign_spec_sha256": config.campaign_spec_sha256,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )

    cached = None
    if args.output.exists() and not args.fresh:
        try:
            cached = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "existing output is unreadable; use --fresh to replace it"
            ) from error
        if not isinstance(cached, dict):
            raise RuntimeError(
                "existing output has an invalid top-level shape; use --fresh "
                "to replace it"
            )
    rows = _validate_cached(
        cached,
        definition,
        source_fingerprint,
        spec,
        config_fingerprint,
        campaign_spec_authentication,
        args.mp_start_method,
    )
    if cached is not None and (
        cached.get("schema_version") != SCHEMA_VERSION
        or not isinstance(cached.get("results"), list)
        or cached.get("source_fingerprint") != source_fingerprint
        or cached.get("ending_source_fingerprint") != source_fingerprint
        or not cached.get("source_stable_during_run")
        or cached.get("experiment_spec") != spec
        or cached.get("config_fingerprint") != config_fingerprint
        or cached.get("ending_config_fingerprint") != config_fingerprint
        or not cached.get("config_stable_during_run")
        or cached.get("campaign_spec_sha256") != config.campaign_spec_sha256
        or cached.get("campaign_spec_authentication") != campaign_spec_authentication
        or cached.get("ending_campaign_spec_authentication")
        != campaign_spec_authentication
        or not cached.get("campaign_spec_stable_during_run")
        or cached.get("schedule_metadata") != seed_manifest
        or cached.get("execution", {}).get("multiprocessing_start_method")
        != args.mp_start_method
        or (cached.get("results") and not rows)
    ):
        raise RuntimeError(
            "existing output is incompatible or invalid; use --fresh to replace it"
        )
    if args.rerun:
        rows = {key: row for key, row in rows.items() if key not in selected_keys}
    pending = [
        WorkerTask(
            definition=definition,
            config=config,
            case=definition.case_by_name[name],
            num_ssu=num_ssu,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
            mp_start_method=args.mp_start_method,
        )
        for name, num_ssu in sorted(selected_keys, key=lambda key: (key[1], key[0]))
        if (name, num_ssu) not in rows
    ]

    def checkpoint():
        payload = _build_payload(
            rows=rows,
            definition=definition,
            config=config,
            schedule=schedule,
            spec=spec,
            source_fingerprint=source_fingerprint,
            config_fingerprint=config_fingerprint,
            campaign_spec_path=args.campaign_spec,
            campaign_spec_authentication=campaign_spec_authentication,
            input_authentication=input_authentication,
            seed_manifest=seed_manifest,
            mp_start_method=args.mp_start_method,
            max_workers=args.max_workers,
            selected_keys=selected_keys,
            selected_ssus=selected_ssus,
        )
        if not all(
            entry["all_available_rows_paired"]
            for entry in payload["pairing_audit"].values()
        ):
            raise RuntimeError("strategy inputs are not paired within an SSU")
        if not payload["source_stable_during_run"]:
            raise RuntimeError("source changed during experiment")
        if not payload["config_stable_during_run"]:
            raise RuntimeError("config changed during experiment")
        if not payload["campaign_spec_stable_during_run"]:
            raise RuntimeError("campaign spec changed during experiment")
        _write_json(args.output, payload)
        return payload

    if pending:
        executor = ProcessPoolExecutor(
            max_workers=min(args.max_workers, len(pending)),
            mp_context=pool_context,
            initializer=_init_worker,
            initargs=initargs,
        )
        futures = {}
        try:
            for task in pending:
                futures[executor.submit(_run_case, task)] = task
            for future in as_completed(futures):
                row = future.result()
                rows[_row_key(row)] = row
                checkpoint()
                summary = row["steady_summary"]
                print(
                    f"DONE {row['case']} ssu={row['num_ssu']} "
                    f"util={summary['mean_npu_utilization']:.4f} "
                    f"slo={summary['ttft_slo_attainment']:.4f} "
                    f"reads={summary['measurement_pressure_reports']} "
                    f"writes={summary['measurement_cir_path_writes']} "
                    f"wall={row['wall_time_s']:.1f}s",
                    flush=True,
                )
        except BaseException:
            _abort_process_pool(executor, futures)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    payload = checkpoint()
    if not payload["selected_complete"]:
        raise RuntimeError("selected experiment shard is incomplete")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "definition": definition.key,
                "num_npu": definition.num_npu,
                "backing_requests_per_npu": config.requests_per_npu,
                "total_assignment_count": (
                    definition.num_npu * config.requests_per_npu
                ),
                "measurement_ms": config.measurement_ms,
                "seed": config.seed,
                "selected_ssus": list(selected_ssus),
                "selected_complete": payload["selected_complete"],
                "source_stable": payload["source_stable_during_run"],
                "config_stable": payload["config_stable_during_run"],
                "campaign_spec_sha256": config.campaign_spec_sha256,
                "campaign_spec_stable": payload["campaign_spec_stable_during_run"],
                "result_count": len(payload["results"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
