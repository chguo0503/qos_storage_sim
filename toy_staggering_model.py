"""Small, deterministic kitchen/SSD example used by the staggering tutorial.

This is an illustrative model, not a GLM or hardware performance prediction.
One cook represents one NPU; one FIFO preparation counter represents Path0.
Every bag is one equal-size I/O and occupies the counter for exactly one second.

API example::

    result = simulate([
        {"name": "A", "bags": 1, "compute_s": 4},
        {"name": "B", "bags": 6, "compute_s": 10},
    ])

Orders for the first dish are submitted together at t=0, in configuration
order. Starting a dish submits all bags for the NEXT dish as one contiguous
FIFO batch. There is only one-dish lookahead, no other transfer stage, and
no intentional delay or phase optimization. A cook starts its next dish only
when both current computation and the next dish's entire order have completed.

All returned objects are JSON serializable. Orders/computations released or
started before ``horizon`` are retained with their complete future intervals;
clip intervals to the desired observation window when calculating utilization.
Cold-start waiting is deliberately not classified as a steady-state stall.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def _normalize_configs(configs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(configs, (str, bytes)) or not isinstance(configs, Sequence):
        raise ValueError("configs must be a nonempty sequence of cook mappings")
    if not configs:
        raise ValueError("configs must contain at least one cook")
    normalized = []
    names = set()
    for config in configs:
        if not isinstance(config, Mapping):
            raise ValueError("each cook configuration must be a mapping")
        name = config.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("each cook name must be a unique nonempty string")
        bags = config.get("bags")
        if isinstance(bags, bool) or not isinstance(bags, int) or bags <= 0:
            raise ValueError("bags must be a positive integer")
        compute_s = _positive_number(config.get("compute_s"), "compute_s")
        normalized.append({"name": name, "bags": bags, "compute_s": compute_s})
        names.add(name)
    return normalized


def simulate(
    configs: Sequence[Mapping[str, Any]], horizon: float = 70
) -> dict[str, Any]:
    """Simulate the closed-loop FIFO model from a genuine cold start.

    ``dish`` is zero-based: order A/0 supplies computation A/0; starting
    computation A/0 releases order A/1. The deadline on a non-cold order is
    the END of the computation that released it, not its own subsequent
    computation's end. Consequently ``ready_s - deadline_s``, when positive,
    is exactly the stall before that order's dish can start.

    A queued I/O never overtakes an earlier I/O. Future completion times can
    therefore be calculated exactly when an entire batch is appended. This
    avoids separate I/O-completion events without changing FIFO semantics.
    Simultaneous computation starts use stable event-insertion order.
    """
    configs = _normalize_configs(configs)
    horizon_s = _positive_number(horizon, "horizon")
    by_name = {config["name"]: config for config in configs}
    orders: list[dict[str, Any]] = []
    io_services: list[dict[str, Any]] = []
    computes: list[dict[str, Any]] = []
    stalls: list[dict[str, Any]] = []
    pending: list[tuple[float, int, str, int]] = []
    fifo_tail = 0.0
    event_sequence = 0

    def submit_order(
        name: str, dish: int, release_s: float, deadline_s: float | None
    ) -> dict[str, Any]:
        nonlocal fifo_tail
        config = by_name[name]
        first_service_s = max(release_s, fifo_tail)
        for bag_index in range(1, config["bags"] + 1):
            start_s = max(release_s, fifo_tail)
            fifo_tail = start_s + 1.0
            io_services.append(
                {
                    "name": name,
                    "dish": dish,
                    "bag_index": bag_index,
                    "release_s": release_s,
                    "start_s": start_s,
                    "end_s": fifo_tail,
                }
            )
        order = {
            "name": name,
            "dish": dish,
            "bags": config["bags"],
            "release_s": release_s,
            "first_service_s": first_service_s,
            "ready_s": fifo_tail,
            "deadline_s": deadline_s,
            "cold_start": deadline_s is None,
            "queue_wait_s": first_service_s - release_s,
            "service_s": float(config["bags"]),
            "read_latency_s": fifo_tail - release_s,
            "stall_s": 0.0 if deadline_s is None else max(0.0, fifo_tail - deadline_s),
        }
        orders.append(order)
        return order

    def schedule_compute(name: str, dish: int, start_s: float) -> None:
        nonlocal event_sequence
        event_sequence += 1
        heapq.heappush(pending, (start_s, event_sequence, name, dish))

    # Submit ALL initial orders before allowing even the first cook to run.
    for config in configs:
        first_order = submit_order(config["name"], 0, 0.0, None)
        schedule_compute(config["name"], 0, first_order["ready_s"])

    while pending:
        start_s, _, name, dish = heapq.heappop(pending)
        if start_s >= horizon_s:
            break
        end_s = start_s + by_name[name]["compute_s"]
        computes.append({"name": name, "dish": dish, "start_s": start_s, "end_s": end_s})
        next_order = submit_order(name, dish + 1, start_s, end_s)
        next_start_s = max(end_s, next_order["ready_s"])
        if next_start_s > end_s:
            stalls.append(
                {"name": name, "dish": dish + 1, "start_s": end_s, "end_s": next_start_s}
            )
        schedule_compute(name, dish + 1, next_start_s)

    return {
        "description": "Synthetic one-path FIFO model; one second per equal-size bag/I/O",
        "configs": configs,
        "horizon_s": horizon_s,
        "io_service_s": 1.0,
        "prefetch_dishes": 1,
        "orders": orders,
        "io_services": io_services,
        "computes": computes,
        "stalls": stalls,
    }


def clipped_duration(
    intervals: Sequence[Mapping[str, Any]],
    start_s: float,
    end_s: float,
    name: str | None = None,
) -> float:
    """Sum intersections with [start_s, end_s); no packed-total time axis."""
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        raise ValueError("observation window must have finite start_s < end_s")
    return sum(
        max(0.0, min(end_s, item["end_s"]) - max(start_s, item["start_s"]))
        for item in intervals
        if name is None or item["name"] == name
    )


def _release_times(
    result: Mapping[str, Any], name: str, start_s: float, end_s: float, origin_s: float
) -> list[float]:
    return [
        order["release_s"] - origin_s
        for order in result["orders"]
        if order["name"] == name and start_s <= order["release_s"] < end_s
    ]


def get_story_data() -> dict[str, Any]:
    """Return both full cold-start simulations and audited plotting facts.

    The failure plot's origin is the REAL simulated time 27 seconds. This
    offset only makes axis labels short: it does not initialize or manufacture
    a steady state. The audited 20-second interval is relative [1, 21), i.e.
    absolute [28, 48), and contains two complete repeated bad cycles.
    """
    success = simulate(
        [{"name": "A", "bags": 1, "compute_s": 4},
         {"name": "B", "bags": 1, "compute_s": 4}],
        horizon=70,
    )
    failure = simulate(
        [{"name": "A", "bags": 1, "compute_s": 4},
         {"name": "B", "bags": 6, "compute_s": 10}],
        horizon=70,
    )
    success["origin_s"] = 0.0
    failure["origin_s"] = 27.0
    start_s, end_s = 28.0, 48.0
    a_compute_s = clipped_duration(failure["computes"], start_s, end_s, "A")
    a_stall_s = clipped_duration(failure["stalls"], start_s, end_s, "A")
    b_compute_s = clipped_duration(failure["computes"], start_s, end_s, "B")
    ssu_busy_s = clipped_duration(failure["io_services"], start_s, end_s)
    a_releases = _release_times(failure, "A", 27, 49, 27)
    b_releases = _release_times(failure, "B", 27, 49, 27)
    checks = {
        "success_has_no_post_start_stalls": not success["stalls"],
        "success_A_releases_5_9_13": _release_times(success, "A", 5, 14, 0) == [5, 9, 13],
        "success_B_releases_6_10_14": _release_times(success, "B", 6, 15, 0) == [6, 10, 14],
        "failure_A_relative_releases": a_releases,
        "failure_B_relative_releases": b_releases,
        "failure_repeated_release_pattern": a_releases == [1, 7, 11, 17, 21]
        and b_releases == [0, 10, 20],
        "failure_window_absolute_s": [start_s, end_s],
        "failure_window_relative_s": [1.0, 21.0],
        "failure_A_compute_s": a_compute_s,
        "failure_A_stall_s": a_stall_s,
        "failure_B_compute_s": b_compute_s,
        "failure_ssu_busy_s": ssu_busy_s,
        "failure_ssu_idle_s": end_s - start_s - ssu_busy_s,
        "failure_A_utilization": a_compute_s / (end_s - start_s),
        "failure_B_utilization": b_compute_s / (end_s - start_s),
        "failure_ssu_utilization": ssu_busy_s / (end_s - start_s),
        "failure_nominal_demand_bags_per_s": 1 / 4 + 6 / 10,
        "failure_nominal_demand_bags_per_20_s": (1 / 4 + 6 / 10) * 20,
        "failure_capacity_bags_per_20_s": 20.0,
    }
    if not all(
        [
            checks["success_has_no_post_start_stalls"],
            checks["success_A_releases_5_9_13"],
            checks["success_B_releases_6_10_14"],
            checks["failure_repeated_release_pattern"],
            a_compute_s == 16,
            a_stall_s == 4,
            b_compute_s == 20,
            ssu_busy_s == 16,
            checks["failure_nominal_demand_bags_per_20_s"] == 17,
        ]
    ):
        raise AssertionError(f"Toy-story audit failed: {checks}")
    return {"success": success, "failure": failure, "checks": checks}


if __name__ == "__main__":
    print(json.dumps(get_story_data(), ensure_ascii=False, indent=2))
