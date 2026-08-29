"""Thin adapters from portable policies to the discrete-event simulator.

Policy decisions live in :mod:`policy_logic`. This module only constructs
simulator configuration objects, so a production implementation does not need
to copy any event-queue code.
"""

from __future__ import annotations

from dataclasses import dataclass

from policy_logic import GROUP_COUNT, PATH_COUNT
import sim
from strategy_profiles import FINAL_STATIC


SUBMIT_BATCH_SIZE = 1
ISSUE_INTERVAL_US = 0.1


@dataclass(frozen=True)
class RoutingStrategySpec:
    """Simulator-facing description of a retained client routing policy."""

    name: str
    description: str
    path_selection_mode: str
    pressure_window_io: int | None

    def client_config(self) -> sim.ClientIOConfig:
        return sim.ClientIOConfig(
            name=f"continuous_prefill_{self.name}",
            pressure_window_io=self.pressure_window_io,
            submit_batch_size=SUBMIT_BATCH_SIZE,
            issue_interval_us=ISSUE_INTERVAL_US,
            path_selection_mode=self.path_selection_mode,
        )

    def metadata(self) -> dict:
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


def routing_strategy_specs() -> tuple[RoutingStrategySpec, ...]:
    """Retained routing clients on one identical static-QoS data plane."""
    return (
        RoutingStrategySpec(
            "baseline",
            "All I/Os use Path 0; no pressure reads",
            sim.PATH_SELECTION_FIXED_PATH_ZERO,
            None,
        ),
        RoutingStrategySpec(
            "layer_once",
            "Read Path pressure once per request-layer-SSU",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            None,
        ),
        RoutingStrategySpec(
            "refresh8",
            "Refresh Path pressure every eight planned I/Os",
            sim.PATH_SELECTION_PRESSURE_AWARE,
            8,
        ),
    )


def static_qos_config() -> sim.StaticQoSConfig:
    """The one hardware QoS profile shared by all routing strategies."""
    return FINAL_STATIC.hardware_config()


def scheme_b_client_config(name: str) -> sim.ClientIOConfig:
    """Configure simulator submission; Scheme B Path IDs come from its plan."""
    return sim.ClientIOConfig(
        name=f"continuous_prefill_scheme_b_{name}",
        pressure_window_io=None,
        submit_batch_size=SUBMIT_BATCH_SIZE,
        issue_interval_us=ISSUE_INTERVAL_US,
        path_selection_mode=sim.PATH_SELECTION_FIXED_PATH_ZERO,
    )


def qos_configs_from_path_cirs(
    path_cirs_by_ssu,
) -> tuple[sim.StaticQoSConfig, ...]:
    """Convert portable Scheme-B CIR tables into simulator registers."""
    return tuple(
        sim.StaticQoSConfig(
            path_cirs=tuple(cirs),
            path_pirs=(float("inf"),) * PATH_COUNT,
            path_weights=(1.0,) * PATH_COUNT,
            group_weights=(1.0,) * GROUP_COUNT,
            category_paths_per_group=(PATH_COUNT // GROUP_COUNT,),
            category_labels=("NPU",),
        )
        for cirs in path_cirs_by_ssu
    )
