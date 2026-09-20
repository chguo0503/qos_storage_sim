"""Static QoS profile used by the final routing experiment."""

from __future__ import annotations

from dataclasses import dataclass

from simulator.core.sim import StaticQoSConfig


PATH_COUNT = 256
GROUP_COUNT = 8


@dataclass(frozen=True)
class StaticQoSProfile:
    name: str
    category_cir_gbps: tuple[float, float, float, float]
    category_paths_per_group: tuple[int, int, int, int]

    def path_cirs(self):
        cirs = []
        for _ in range(GROUP_COUNT):
            for budget, count in zip(
                self.category_cir_gbps,
                self.category_paths_per_group,
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


FINAL_STATIC = StaticQoSProfile(
    name="low_protect_cir_20_6_8_6_current_paths",
    category_cir_gbps=(20.0, 6.0, 8.0, 6.0),
    category_paths_per_group=(12, 4, 12, 4),
)


def main():
    import math
    from simulator.policies.policy_logic import category_path_ids
    config = FINAL_STATIC.hardware_config()
    assert len(config.path_cirs) == PATH_COUNT
    assert math.isclose(math.fsum(config.path_cirs), 40.0)
    for category, budget, count in zip(config.category_labels, FINAL_STATIC.category_cir_gbps,
                                       FINAL_STATIC.category_paths_per_group):
        paths = category_path_ids(category, config)
        assert len(paths) == GROUP_COUNT * count
        assert math.isclose(math.fsum(config.path_cirs[p] for p in paths), budget)
    print("profiles: PASS (category CIR budgets 20/6/8/6 across 256 paths)")


if __name__ == "__main__":
    main()
