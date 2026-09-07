"""Small capacity witnesses checked against native packetized SSD -> HBM.

These are one-layer read-completion tests, NOT middle-one-second utilization
experiments and NOT new TTFT SLO settings. Native L0 carries its usual deadline
at admission; the comparison deadlines below are independent mathematical
read-ready requirements and do not change native compute or arrival times.
No production source is changed. Counts at time zero are wholly unread work.
"""

from contextlib import nullcontext
from dataclasses import replace

import pytest

import sim
from coflow_capacity_analysis import ActivatedCoflow, capacity_overloads, solo_read_lower_bound_ms
from coflow_disk_adapter import coflow_disk_adapter
from continuous_batch_sim import ContinuousBatchRequest, simulate_continuous_batch
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from shared_path_common import IO_GIB


def _simulate(layouts, *, num_npu=4, route="baseline", arrivals=None, reorder=False,
              submit_batch_size=None):
    """One request per listed NPU; extract HBM-ready, not compute completion."""
    num_ssu = len(layouts[0])
    arrivals = arrivals or (0.0,) * len(layouts)
    requests = tuple(ContinuousBatchRequest(
        request_id=npu, npu_id=npu, arrival_time_ms=arrivals[npu],
        load={"category": "SS", "per_layer_us": 1000.0, "seq_len_k": 32, "nql": 128},
        placement=(tuple((s, IO_GIB) for s, count in enumerate(row) for _ in range(count)),))
        for npu, row in enumerate(layouts))
    client = next(spec for spec in routing_strategy_specs() if spec.name == route).client_config()
    if submit_batch_size is not None:
        client = replace(client, submit_batch_size=submit_batch_size)
    manager = coflow_disk_adapter("strategy2") if reorder else nullcontext()
    with manager:
        result = simulate_continuous_batch(
            requests, num_npu=num_npu, num_ssu=num_ssu, n_layers=1, batch_size=1,
            policy=sim.POLICY_QOS_STATIC_CIR, qos_config=static_qos_config(),
            client_io_config=client, cross_request_layer0_prefetch=False,
            disk_bw_gbps=40, npu_bw_gbps=50, submit_order_seed=17)
    assert all(result["invariants"].values())
    expected = sum(map(sum, layouts))
    assert result["submitted_blocks"] == result["completed_blocks"] == expected
    ready = {}
    activated = {}
    for batch in result["microbatch_metrics"]:
        assert batch["batch_size"] == 1
        rid = batch["member_request_ids"][0]
        layer = batch["layer_metrics"][0]
        activated[rid] = layer["io_start_time_ms"]
        ready[rid] = layer["io_ready_time_ms"]
        assert ready[rid] == pytest.approx(layer["compute_start_ms"], abs=1e-12)
        assert layer["compute_end_ms"] > ready[rid]
    return ready, activated


def _cold_coflows(layouts, deadline):
    return tuple(ActivatedCoflow(npu, 0, npu, 0, deadline, tuple(row), sum(row))
                 for npu, row in enumerate(layouts))


def _assert_witness_has_a_late_native_member(witnesses, reads, ready):
    deadline = {(read.request_id, read.layer): read.deadline_ms for read in reads}
    assert witnesses
    for witness in witnesses:
        assert witness.kind == "ready_prefix"
        assert any(ready[rid] > deadline[(rid, layer)] + 1e-9
                   for rid, layer in witness.coflow_ids)


@pytest.mark.parametrize("npus,ssus,route", (
    (4, 5, "baseline"), (4, 6, "baseline"), (4, 7, "baseline"),
    (32, 6, "baseline"), (32, 7, "layer_once")))
def test_shared_ssd_capacity_witness_implies_at_least_one_actual_read_miss(npus, ssus, route):
    # Every case has 320 commands on SSD0; it needs 1.342773 ms at 40 GiB/s.
    # Other disks are configured but cannot read this fixed SSD0 placement.
    blocks = 320 // npus
    layouts = ((blocks,) + (0,) * (ssus - 1),) * npus
    reads = _cold_coflows(layouts, deadline=1.0)
    witnesses = capacity_overloads(reads, now_ms=0)
    assert {w.resource for w in witnesses} == {"ssu"}
    assert {w.resource_id for w in witnesses} == {0}
    ready, activated = _simulate(layouts, num_npu=npus, route=route)
    assert set(activated.values()) == {0.0}
    _assert_witness_has_a_late_native_member(witnesses, reads, ready)


@pytest.mark.parametrize("ssus", (5, 6, 7))
def test_receive_link_witness_matches_actual_hbm_barrier_even_when_disks_fit(ssus):
    # Each SSD needs only 0.419617 ms, but this NPU has ONE 50-GiB/s link.
    layouts = ((100,) * ssus,)
    reads = _cold_coflows(layouts, deadline=1.0)
    witnesses = capacity_overloads(reads, now_ms=0)
    assert {w.resource for w in witnesses} == {"npu_link"}
    ready, _ = _simulate(layouts)
    _assert_witness_has_a_late_native_member(witnesses, reads, ready)


@pytest.mark.parametrize("ssus", (5, 6, 7))
@pytest.mark.parametrize("shape", ("balanced", "hotspot"))
def test_solo_fluid_lower_bound_never_exceeds_native_isolated_hbm_ready(ssus, shape):
    row = (64,) * ssus if shape == "balanced" else (257,) + (3,) * (ssus - 1)
    ready, activated = _simulate((row,))
    bound = solo_read_lower_bound_ms(row)
    assert bound <= ready[0] - activated[0] + 1e-12
    # This is an inequality, not an assertion of exact prediction.
    assert ready[0] - activated[0] > bound


def test_no_capacity_witness_does_not_guarantee_packetized_read_feasibility():
    row = (1, 0, 0, 0, 0)
    deadline = .005
    reads = _cold_coflows((row,), deadline)
    assert capacity_overloads(reads, 0) == ()
    ready, activated = _simulate((row,))
    # A single packet must finish SSD service before starting link service.
    exact_single_packet_ms = 1000 * IO_GIB * (1 / 40 + 1 / 50)
    assert ready[0] - activated[0] == pytest.approx(exact_single_packet_ms, abs=1e-12)
    assert solo_read_lower_bound_ms(row) < deadline < ready[0]


def test_no_witness_can_still_miss_because_of_fifo_and_reordering_can_rescue_it():
    layouts = ((200, 0, 0, 0, 0), (1, 0, 0, 0, 0))
    now = 0.0
    deadlines = {0: 2.0, 1: .012}
    # A test-only bulk client batch puts the 200 ordinary commands ahead of
    # the one short command in Path0 under submit seed17. Every SSD command is
    # still 176 KiB. Both requests are already activated at time zero. The
    # production experiment's client batch size is not changed by this test.
    reads = (ActivatedCoflow(0, 0, 0, 0, deadlines[0], (200, 0, 0, 0, 0), 200),
             ActivatedCoflow(1, 0, 1, now, deadlines[1], (1, 0, 0, 0, 0), 1))
    assert capacity_overloads(reads, now) == ()
    fifo_ready, fifo_activation = _simulate(layouts, submit_batch_size=256)
    reordered_ready, reordered_activation = _simulate(layouts, submit_batch_size=256, reorder=True)
    assert fifo_activation == reordered_activation == {0: 0.0, 1: now}
    assert fifo_ready[1] > deadlines[1]
    assert reordered_ready[1] < deadlines[1]
    assert fifo_ready[0] < deadlines[0] and reordered_ready[0] < deadlines[0]
    # The allowed same-Path sorter sees both native L0 deadlines as zero and
    # favours the one-I/O component as tie-break. These external comparison
    # deadlines are not injected into native scheduling metadata.
