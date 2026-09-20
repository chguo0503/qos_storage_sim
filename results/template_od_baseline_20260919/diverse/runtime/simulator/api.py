"""Input-independent entry point for the shared-path simulations.

Pass normalized requests; this module never opens a catalog or experiment
directory. Policy adapters temporarily install module hooks, so use separate
processes, not threads, for concurrent simulations.
"""

from contextlib import nullcontext
import math

from simulator.core import continuous_batch_sim as core
from simulator.core import sim
from simulator import config
from simulator.adapters import shared_path, slo_pool
from simulator.policies.baselines import baseline_qos_config, canonical_baseline_name


STRATEGIES = ("asu_baseline", "od_baseline", "once", "new_once", "static", "mild", "aggressive")


def run_simulation(requests, *, strategy="asu_baseline", num_npu=32, num_ssu=3,
                   n_layers=8, seed=7, disk_bw_gib_s=40.0, npu_bw_gib_s=50.0,
                   collector_interval_ms=5.0, cross_request_layer0_prefetch=True):
    """Run finite, per-NPU FIFO input through the existing data plane.

    All requests run to completion. No warm-window or SLO statistics are
    silently selected here; callers can compute those from the returned layer
    and request metrics. Shared-path policies require equal 176-KiB commands.
    """
    strategy = canonical_baseline_name(strategy)
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if not math.isfinite(disk_bw_gib_s) or disk_bw_gib_s <= 0:
        raise ValueError("disk bandwidth must be finite and positive")
    if not math.isfinite(npu_bw_gib_s) or npu_bw_gib_s <= 0:
        raise ValueError("NPU link bandwidth must be finite and positive")
    if strategy != "od_baseline" and disk_bw_gib_s < 40.0:
        raise ValueError("the existing category CIR profile reserves 40 GiB/s per disk; "
                         "use disk_bw_gib_s >= 40 or explicitly configure the low-level core")
    requests = tuple(requests)
    fingerprint = core.continuous_batch_input_fingerprint(requests)
    extension = strategy in slo_pool.VARIANTS
    stats = slo_pool.make_stats() if extension else None
    manager = slo_pool.install_policy(strategy, stats) if extension else nullcontext()
    backbone = "once" if extension else strategy
    route = "baseline" if backbone in ("asu_baseline", "od_baseline") else "layer_once"
    client = next(s for s in config.routing_strategy_specs() if s.name == route).client_config()
    qos = baseline_qos_config(backbone, num_npu, disk_bw_gib_s)
    with manager:
        # Resolve the adapter after installing SLO hooks, preserving Once's
        # original routing and the scoped monkeypatch restoration order.
        with shared_path.shared_path_adapter(
            strategy=backbone, collector_interval_ms=collector_interval_ms,
        ) as adapter:
            summary = core.simulate_continuous_batch(
                requests, num_npu=num_npu, num_ssu=num_ssu, n_layers=n_layers,
                batch_size=1, policy=sim.POLICY_QOS_STATIC_CIR,
                qos_config=qos, client_io_config=client,
                cross_request_layer0_prefetch=cross_request_layer0_prefetch,
                pressure_ttl_ms=collector_interval_ms, disk_bw_gbps=disk_bw_gib_s,
                npu_bw_gbps=npu_bw_gib_s, submit_order_seed=seed,
            )
            adapter_statistics = adapter.statistics()
    if not all(summary["invariants"].values()):
        raise AssertionError("simulation invariant failed")
    if core.continuous_batch_input_fingerprint(requests) != fingerprint:
        raise AssertionError("simulation mutated the input requests")
    return {"strategy": strategy, "input_fingerprint": fingerprint,
            "summary": summary, "adapter_statistics": adapter_statistics,
            "routing_statistics": stats}
