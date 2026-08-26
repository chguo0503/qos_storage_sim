"""Pure scheduling helpers for demand-aware and idealized policies."""

from __future__ import annotations


def max_min_work_conserving_rates(capacity_gbps, demands_gbps):
    """Return max-min guarantees, then share unused capacity equally."""
    active = list(demands_gbps)
    remaining = float(capacity_gbps)
    rates = {}
    unsatisfied = set(active)

    while unsatisfied:
        fair_share = remaining / len(unsatisfied)
        satisfied = [
            source
            for source in unsatisfied
            if demands_gbps[source] <= fair_share
        ]
        if not satisfied:
            for source in unsatisfied:
                rates[source] = fair_share
            remaining = 0.0
            break
        for source in satisfied:
            rate = float(demands_gbps[source])
            rates[source] = rate
            remaining -= rate
            unsatisfied.remove(source)

    if remaining > 0.0 and active:
        bonus = remaining / len(active)
        for source in active:
            rates[source] += bonus
    return rates


def capped_proportional_demands(capacity_gbps, compute_time_s, work_by_resource):
    """Cap one NPU's total demand, then split it by each SSD's layer bytes."""
    total_work_gb = sum(work_by_resource.values())
    capped_total_gbps = min(total_work_gb / compute_time_s, capacity_gbps)
    return {
        resource: capped_total_gbps * work_gb / total_work_gb
        for resource, work_gb in work_by_resource.items()
    }


def capped_input_demands(capacity_gbps, input_demand_gbps, work_by_resource):
    """Cap an online NPU demand advert, then split it by current-layer bytes."""
    total_work_gb = sum(work_by_resource.values())
    capped_total_gbps = min(float(input_demand_gbps), float(capacity_gbps))
    return {
        resource: capped_total_gbps * work_gb / total_work_gb
        for resource, work_gb in work_by_resource.items()
    }


def slack_link_guarded_demands(
    capacity_gbps,
    now_ms,
    deadline_ms,
    link_backlog_gb,
    work_by_resource,
):
    """Derive current-layer demands from remaining work and guarded slack.

    Only state already visible to the NPU is consumed: the current layer's
    remaining SSD work, its current compute deadline, and work already committed
    to the NPU's receive link.  The aggregate advert is capped before it is split
    across SSDs, so independent disks cannot request more than the NPU link.
    """
    total_work_gb = sum(work_by_resource.values())
    slack_s = max(0.0, (float(deadline_ms) - float(now_ms)) / 1000.0)
    guarded_slack_s = max(
        0.0,
        slack_s - float(link_backlog_gb) / float(capacity_gbps),
    )
    total_demand_gbps = (
        float(capacity_gbps)
        if guarded_slack_s == 0.0
        else min(total_work_gb / guarded_slack_s, float(capacity_gbps))
    )
    return {
        resource: total_demand_gbps * work_gb / total_work_gb
        for resource, work_gb in work_by_resource.items()
    }


def proportional_capacity_grants(capacity_gbps, demands_gbps):
    """Keep demands unchanged when feasible; proportionally scale overload."""
    total_demand = sum(demands_gbps.values())
    if total_demand == 0.0:
        return {source: 0.0 for source in demands_gbps}
    scale = min(1.0, float(capacity_gbps) / total_demand)
    grants = {
        source: float(demand) * scale
        for source, demand in demands_gbps.items()
    }
    excess = sum(grants.values()) - float(capacity_gbps)
    if excess > 0.0:
        last_source = next(reversed(grants))
        grants[last_source] -= excess
    return grants


def omniscient_edf_key(flow):
    """Static full-layer EDF/SRPT key used by the idealized coordinator."""
    return (
        flow.deadline_time,
        flow.layer_work_gb,
        flow.enqueue_time,
        flow.request_id,
        flow.layer,
        flow.block_idx,
        flow.disk_id,
    )


def global_link_aware_priority(flow, predicted_link_end_ms, pending_layer_blocks):
    """Rank an online SSD candidate by its predicted downstream completion."""
    return (
        max(flow.deadline_time, predicted_link_end_ms),
        predicted_link_end_ms,
        pending_layer_blocks,
        flow.deadline_time,
        flow.enqueue_time,
        flow.request_id,
        flow.layer,
        flow.block_idx,
    )
