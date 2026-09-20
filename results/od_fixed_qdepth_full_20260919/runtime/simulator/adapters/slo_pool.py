"""Host-state adapter for the request-budget Once candidate-pool policies.

Uses the existing scoped adapter hooks. Run concurrent simulations in separate
processes because these hooks temporarily patch shared module functions.
"""
from contextlib import contextmanager
from unittest.mock import patch
from simulator.policies.slo_pool import RequestBudget, RouterConfig, PoolDecision, VARIANTS, choose_path_pool, slo_path_ids
from simulator.policies.once import once_path_ids

POLICIES = ("baseline", "mild", "aggressive", "static")


def simulator_strategy(policy_name):
    """Original adapter backbone; the extension only replaces Once routing."""
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    return "baseline" if policy_name == "baseline" else "once"


def make_stats():
    """Bounded categorical aggregates, not a per-block event log."""
    return dict(calls=0, io_blocks=0, unique_paths_sum=0, candidate_paths_sum=0,
                mode_counts={}, decision_counts={}, examples=[])


def upper_budget(context, state, now_ms, policy_name="aggressive"):
    """Read only this request's host-visible profile, admission, and progress.

    Static freezes the initial budget for *any* data profile. The deciding
    initial waiting budget per layer is 0.5*C, independent of category/role.
    Category only determines the legal pool supplied by the existing adapter.
    No current SSD queue, future event, or future request is inspected.
    """
    if policy_name not in VARIANTS:
        raise ValueError("upper_budget is for mild/aggressive/static")
    request = context.requests[state.request_id]
    if not request.admitted:
        return None
    assert request.batch_size == 1
    c = request.per_layer_compute_ms
    if policy_name == "static":
        remaining_compute = context.n_layers * c
        remaining_slo = 1.5 * remaining_compute
        remaining_layers = context.n_layers
    else:
        batch = context.microbatches[request.batch_id]
        if batch.compute_active_layer >= 0:
            k = batch.compute_active_layer
            elapsed = max(0., now_ms - batch.layer_metrics[k].compute_start_ms)
            remaining_compute = max(0., c - elapsed) + (context.n_layers-k-1)*c
        else:
            remaining_compute = (context.n_layers-batch.compute_done_up_to-1)*c
        remaining_slo = request.admission_time_ms + 1.5*context.n_layers*c - now_ms
        remaining_layers = context.n_layers-state.layer
    layout = request.manifest.placement[0 if len(request.manifest.placement)==1 else state.layer]
    work = [0.] * context.num_ssu
    for ssu, size in layout:
        work[ssu] += size
    return RequestBudget(remaining_slo, remaining_compute, remaining_layers, tuple(work))


@contextmanager
def install_policy(policy_name, stats=None):
    """Temporarily install the extension at the original adapter boundary.

    Usage::

        with install_policy(name, stats):
            run_case(..., strategy=simulator_strategy(name))

    The caller still drives the ordinary simulator and original shared adapter.
    Its FIFO, static QoS, sampled count table, L1/L2, submit times, and initial
    unadmitted prefetch are preserved. Policy compute latency remains the same
    as the reference (zero simulated overhead). Each process runs one policy.
    """
    from simulator.core import continuous_batch_sim as native
    from simulator.adapters import shared_path as shared
    from simulator.policies import once as once_module

    strategy = simulator_strategy(policy_name)
    if stats is None:
        stats = make_stats()
    original_manager = shared.shared_path_adapter

    @contextmanager
    def manager(**kwargs):
        assert kwargs.get("strategy", "once") == strategy
        if policy_name == "baseline":
            with original_manager(**kwargs) as adapter:
                yield adapter
            return
        config = VARIANTS[policy_name]
        decision = None

        def route(count, snapshot, allowed_path_ids, qos, start_offset=0,
                  disk_bw_gib_s=40.0):
            assert decision is not None
            ids = once_path_ids(count, snapshot, decision.path_ids, qos,
                                start_offset=start_offset, disk_bw_gib_s=disk_bw_gib_s)
            assert len(ids) == count and set(ids).issubset(allowed_path_ids)
            stats["io_blocks"] += count
            stats["unique_paths_sum"] += len(set(ids))
            return ids

        with patch.object(once_module, "once_path_ids", route):
            with original_manager(**kwargs) as adapter:
                original_plan = native._plan_paths

                def plan(context, state, now_ms):
                    nonlocal decision
                    budget = upper_budget(context, state, now_ms, policy_name)
                    decision = choose_path_pool(
                        state.allowed_path_ids, adapter.qos[state.disk_id],
                        budget=budget, config=config, disk_bw_gib_s=context.disk_bw_gbps,
                        npu_link_bw_gib_s=context.npu_bw_gbps,
                    )
                    load = context.requests[state.request_id].manifest.load
                    category = load["category"]
                    profile = f"{load.get('seq_len_k', '?')}:{load.get('nql', '?')}"
                    key = f"{category}|{profile}|layer{state.layer}|{decision.mode}"
                    stats["calls"] += 1
                    stats["candidate_paths_sum"] += len(decision.path_ids)
                    stats["mode_counts"][decision.mode] = stats["mode_counts"].get(decision.mode, 0) + 1
                    stats["decision_counts"][key] = stats["decision_counts"].get(key, 0) + 1
                    if len(stats["examples"]) < 32:
                        stats["examples"].append(dict(time_ms=now_ms,
                            request_id=state.request_id, npu_id=state.npu_id,
                            layer=state.layer, ssu_id=state.disk_id, category=category,
                            profile=profile, mode=decision.mode,
                            candidate_path_count=len(decision.path_ids),
                            waiting_budget_ms=decision.waiting_budget_ms,
                            waiting_budget_per_layer_ms=decision.waiting_budget_per_layer_ms))
                    return original_plan(context, state, now_ms)

                with patch.object(native, "_plan_paths", plan):
                    yield adapter

    with patch.object(shared, "shared_path_adapter", manager):
        yield stats
