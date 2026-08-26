"""Static QoS layouts and client telemetry variants used by experiments."""

from __future__ import annotations

from dataclasses import dataclass

from sim import ClientIOConfig, StaticQoSConfig


PATH_COUNT = 256
GROUP_COUNT = 8

CIR_CURRENT = (20.0, 4.0, 12.0, 4.0)
CIR_SHORT_LATENCY = (20.0, 2.0, 16.0, 2.0)
CIR_CAPPED_DEMAND = (17.0, 2.0, 18.0, 3.0)

PATHS_CURRENT = (12, 4, 12, 4)
# Capped per-NPU demand assigns about 90/32/102/32 of the 256 Path tickets.
# The closest common per-Group integer layout is therefore 11/4/13/4.
PATHS_CAPPED_DEMAND = (11, 4, 13, 4)


@dataclass(frozen=True)
class StaticQoSProfile:
    name: str
    category_cir_gbps: tuple
    category_paths_per_group: tuple

    def path_cirs(self):
        cirs = []
        for _ in range(GROUP_COUNT):
            for budget, count in zip(
                self.category_cir_gbps, self.category_paths_per_group
            ):
                cirs.extend([budget / (GROUP_COUNT * count)] * count)
        return tuple(cirs)

    def hardware_config(self):
        return StaticQoSConfig(
            path_cirs=self.path_cirs(),
            path_pirs=(float("inf"),) * PATH_COUNT,
            path_weights=(1.0,) * PATH_COUNT,
            group_weights=(1.0,) * GROUP_COUNT,
            category_paths_per_group=self.category_paths_per_group,
        )


CURRENT_STATIC = StaticQoSProfile(
    "static_20_4_12_4_paths_12_4_12_4",
    CIR_CURRENT,
    PATHS_CURRENT,
)

EQUAL_TICKET_STATIC = StaticQoSProfile(
    "equal_cir_demand_ticket_paths",
    (13.75, 5.0, 16.25, 5.0),
    (11, 4, 13, 4),
)

STATIC_PROFILES = {
    profile.name: profile
    for profile in (
        CURRENT_STATIC,
        EQUAL_TICKET_STATIC,
        # One-factor profiles make the CIR and Path-count effects identifiable.
        StaticQoSProfile(
            "current_cir_20_4_12_4_demand_paths_11_4_13_4",
            CIR_CURRENT,
            PATHS_CAPPED_DEMAND,
        ),
        StaticQoSProfile(
            "short_latency_cir_20_2_16_2_current_paths_12_4_12_4",
            CIR_SHORT_LATENCY,
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "short_latency_cir_20_2_16_2_demand_paths_11_4_13_4",
            CIR_SHORT_LATENCY,
            PATHS_CAPPED_DEMAND,
        ),
        StaticQoSProfile(
            "capped_demand_cir_17_2_18_3_current_paths_12_4_12_4",
            CIR_CAPPED_DEMAND,
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "capped_demand_cir_17_2_18_3_demand_paths_11_4_13_4",
            CIR_CAPPED_DEMAND,
            PATHS_CAPPED_DEMAND,
        ),
        StaticQoSProfile(
            "capped_demand_17_2_18_3_balanced_paths",
            CIR_CAPPED_DEMAND,
            (5, 4, 12, 11),
        ),
        StaticQoSProfile(
            "uncapped_demand_16_2_20_2_balanced_paths",
            (16.0, 2.0, 20.0, 2.0),
            (5, 4, 12, 11),
        ),
        StaticQoSProfile(
            "mean_target_20_1_18_1_balanced_paths",
            (20.0, 1.0, 18.0, 1.0),
            (5, 4, 12, 11),
        ),
        StaticQoSProfile(
            "mean_target_22_1_16_1_balanced_paths",
            (22.0, 1.0, 16.0, 1.0),
            (5, 4, 12, 11),
        ),
        StaticQoSProfile(
            "mean_target_18_1_20_1_balanced_paths",
            (18.0, 1.0, 20.0, 1.0),
            (5, 4, 12, 11),
        ),
        StaticQoSProfile(
            "capped_demand_17_2_18_3_demand_paths",
            CIR_CAPPED_DEMAND,
            (14, 2, 14, 2),
        ),
        StaticQoSProfile(
            "equal_ticket_16p25_1p25_20_2p5",
            (16.25, 1.25, 20.0, 2.5),
            (13, 1, 16, 2),
        ),
        StaticQoSProfile(
            "low_protect_cir_20_4p5_11_4p5_current_paths",
            (20.0, 4.5, 11.0, 4.5),
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "low_protect_cir_20_5_10_5_current_paths",
            (20.0, 5.0, 10.0, 5.0),
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "low_protect_cir_20_6_8_6_current_paths",
            (20.0, 6.0, 8.0, 6.0),
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "balanced_cir_19_5_11_5_current_paths",
            (19.0, 5.0, 11.0, 5.0),
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "ss_plus_cir_21_5_9_5_current_paths",
            (21.0, 5.0, 9.0, 5.0),
            PATHS_CURRENT,
        ),
        StaticQoSProfile(
            "low_protect_cir_20_5_10_5_paths_12_5_10_5",
            (20.0, 5.0, 10.0, 5.0),
            (12, 5, 10, 5),
        ),
    )
}


# Use this fixed, predeclared set for cross-SSU model selection.  It includes
# the current layout plus CIR-only, Path-only, and joint changes; the analysis
# must select one profile across 40/56/80 SSUs rather than one per SSU.
PRIMARY_STATIC_CANDIDATES = tuple(
    STATIC_PROFILES[name]
    for name in (
        CURRENT_STATIC.name,
        "current_cir_20_4_12_4_demand_paths_11_4_13_4",
        "short_latency_cir_20_2_16_2_current_paths_12_4_12_4",
        "short_latency_cir_20_2_16_2_demand_paths_11_4_13_4",
        "capped_demand_cir_17_2_18_3_current_paths_12_4_12_4",
        "capped_demand_cir_17_2_18_3_demand_paths_11_4_13_4",
    )
)


REFINEMENT_STATIC_CANDIDATES = tuple(
    STATIC_PROFILES[name]
    for name in (
        "low_protect_cir_20_4p5_11_4p5_current_paths",
        "low_protect_cir_20_5_10_5_current_paths",
        "low_protect_cir_20_6_8_6_current_paths",
        "balanced_cir_19_5_11_5_current_paths",
        "ss_plus_cir_21_5_9_5_current_paths",
        "low_protect_cir_20_5_10_5_paths_12_5_10_5",
    )
)


CLIENT_VARIANTS = {
    "layer_snapshot_batch8": ClientIOConfig("per_layer_snapshot", None, 8),
    "refresh8_batch8": ClientIOConfig("refresh8", 8, 8),
    "per_io_live_batch8": ClientIOConfig("per_io_live", 1, 8),
    "refresh8_batch16": ClientIOConfig("refresh8_batch16", 8, 16),
    "refresh8_batch32": ClientIOConfig("refresh8_batch32", 8, 32),
    "oracle_layer_submit": ClientIOConfig("oracle_layer_submit", None, 1_000_000),
    "ticket_layer_snapshot": ClientIOConfig(
        "ticket_layer_snapshot", None, 8, "per_npu_demand_tickets"
    ),
    "ticket_refresh8": ClientIOConfig(
        "ticket_refresh8", 8, 8, "per_npu_demand_tickets"
    ),
    "ticket_per_io": ClientIOConfig(
        "ticket_per_io", 1, 8, "per_npu_demand_tickets"
    ),
}
