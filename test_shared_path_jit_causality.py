"""Observe native JIT boundaries without adding state to the policy inputs."""

from unittest.mock import patch

import pytest

import continuous_batch_sim as native
import sim
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from shared_path_common import IO_GIB
from shared_path_sim_adapter import shared_path_adapter


def _observed_trace(strategy, ssu_count):
    requests = tuple(native.ContinuousBatchRequest(
        request_id=npu * 100 + generation, npu_id=npu,
        arrival_time_ms=generation * 5.0,
        load={"category": "SS", "per_layer_us": 30000.0 + 10000.0 * (npu % 2),
              "seq_len_k": 32, "nql": 128},
        placement=(tuple((block % ssu_count, IO_GIB) for block in range(64)),))
        for npu in range(4) for generation in range(3))
    activations, first_submit = [], {}
    with shared_path_adapter(strategy) as adapter:
        original_start = native._start_layer_io
        original_submit = native._submit_one_client_batch

        def start(context, request, layer, now, deadline, window, **kwargs):
            fresh = 0 <= layer < context.n_layers and not request.io_started[layer]
            if not fresh:
                return original_start(context, request, layer, now, deadline, window, **kwargs)
            npu = context.npus[request.manifest.npu_id]
            before_id = context.next_submission_id
            ledger_before = sum(map(sum, adapter.pending))
            # Native facts are observed only by this test, not given to policy.
            record = {"request_id": request.manifest.request_id, "layer": layer,
                      "now": now, "deadline": deadline, "arrived": request.arrived,
                      "admitted": request.admitted,
                      "link_pending": len(npu.link_pending),
                      "link_active": npu.link_active_flow is not None,
                      "previous_submission_states": len(context.submission_queues.get(npu.npu_id, ())),
                      "snapshot_time": adapter.collector.sampled_at_ms}
            if npu.compute_active is not None:
                active_layer = npu.compute_active[1]
                record["current_compute_layer"] = active_layer
                record["known_compute_end"] = npu.active_batch.layer_metrics[active_layer].compute_end_ms
            if not request.admitted:
                record["queued_head"] = npu.admission_queue[0]
            result = original_start(context, request, layer, now, deadline, window, **kwargs)
            states = [context.submission_states[s]
                      for s in range(before_id, context.next_submission_id)]
            record["state_ids"] = tuple(state.state_id for state in states)
            record["release_times"] = tuple(state.ready_time_ms for state in states)
            record["activation_time"] = request.io_start_time_ms[layer]
            record["planned_count_at_activation"] = sum(len(state.planned_path_ids) for state in states)
            record["ledger_change_at_activation"] = sum(map(sum, adapter.pending)) - ledger_before
            activations.append(record)
            return result

        def submit(context, state, now):
            first_submit.setdefault(state.state_id, now)
            assert now + 1e-9 >= state.ready_time_ms
            return original_submit(context, state, now)

        with patch.object(native, "_start_layer_io", start), \
             patch.object(native, "_submit_one_client_batch", submit):
            summary = native.simulate_continuous_batch(
                requests, num_npu=4, num_ssu=ssu_count, n_layers=8, batch_size=1,
                policy=sim.POLICY_QOS_STATIC_CIR, qos_config=static_qos_config(),
                cross_request_layer0_prefetch=True,
                client_io_config=routing_strategy_specs()[1].client_config(),
                disk_bw_gbps=40, npu_bw_gbps=50, pressure_ttl_ms=5,
                submit_order_seed=17)
        return summary, adapter.statistics(), activations, first_submit


@pytest.mark.parametrize("strategy,ssu_count", (("strategy1", 6), ("strategy2", 7)))
def test_jit_release_and_cross_request_deadlines_are_causal_native_events(strategy, ssu_count):
    summary, stats, activations, issued = _observed_trace(strategy, ssu_count)
    assert all(summary["invariants"].values())
    assert len(activations) == 12 * 8
    cross_request = [r for r in activations if r["layer"] == 0 and not r["admitted"]]
    assert cross_request  # The test must actually exercise the boundary.
    for record in cross_request:
        assert record["arrived"]
        assert record["queued_head"] == record["request_id"]
        assert record["current_compute_layer"] == 7
        assert record["deadline"] == record["known_compute_end"]
    for record in activations:
        assert record["arrived"]
        assert record["activation_time"] == record["now"]
        assert 0 <= record["now"] - record["snapshot_time"] <= 5 + 1e-9
        assert len(set(record["release_times"])) == 1  # common coflow release
        for state_id, release in zip(record["state_ids"], record["release_times"]):
            assert issued[state_id] + 1e-9 >= release >= record["now"]
    assert stats["reserved_blocks"] == stats["acknowledged_blocks"] == 12 * 8 * 64
    assert stats["ledger_end_counts_by_ssu"] == [0] * ssu_count


def test_one_step_prefetch_has_no_old_link_or_submission_backlog_at_activation():
    _, _, activations, _ = _observed_trace("strategy1", 6)
    # Under this protocol, current-layer data is fully in HBM before next-layer
    # activation. The last layer uses that same opportunity for the next request.
    # This is NOT a claim for deeper prefetch or multiple active batches per NPU.
    assert all(record["link_pending"] == 0 and not record["link_active"]
               for record in activations)
    assert all(record["previous_submission_states"] == 0 for record in activations)


def test_jit_held_states_are_not_yet_path_reservations():
    _, stats, activations, issued = _observed_trace("strategy1", 6)
    delayed = [r for r in activations if r["release_times"][0] > r["now"] + 1e-9]
    assert delayed and stats["jit_prefetch"]["delay_count"] == len(delayed)
    for record in delayed:
        assert record["planned_count_at_activation"] == 0
        assert record["ledger_change_at_activation"] == 0
        assert min(issued[state_id] for state_id in record["state_ids"]) > record["now"]

