"""OD isolation/equal-CIR service and legacy-ASU behavioral regressions."""

from collections import Counter, defaultdict
from unittest.mock import patch

import pytest

from simulator.core import continuous_batch_sim as native
from simulator.core import sim
from simulator.config import routing_strategy_specs
from inputs.runners.run_shared_path_experiments import logical_input_fingerprint
from simulator.policies.baselines import baseline_qos_config, od_npu_path_ids
from simulator.adapters.shared_path import IO_GIB, shared_path_adapter


def trace(num_npu=4, generations=3, blocks=16):
    result = []
    for npu in range(num_npu):
        for generation in range(generations):
            rid = npu * 100 + generation
            category = ("SS", "SL", "LS", "LL")[(npu + generation) % 4]
            load = dict(request_id=rid, npu_id=(npu + 1) % num_npu,
                        seq_len_k=32, nql=128, category=category,
                        per_layer_us=200.0, per_layer_kv_gb=blocks * IO_GIB,
                        required_bw_input_gbps=blocks * IO_GIB / .0002)
            placement = (tuple((sim.block_ring_hash_disk_id(rid, k, 3), IO_GIB)
                               for k in range(blocks)),)
            # Deliberately stale load.npu_id proves ownership uses execution
            # npu_id, rather than a source annotation inside the request.
            result.append(native.ContinuousBatchRequest(rid, npu, 0., load, placement))
    return tuple(result)


def metadata(requests, num_npu=4):
    return dict(num_npu=num_npu, num_ssu=3, seed=7, equal_176kib_blocks=True,
                logical_input_fingerprint=logical_input_fingerprint(requests))


@pytest.mark.parametrize("num_npu", (1, 4, 8, 32, 255, 256))
def test_exactly_one_equal_cir_path_per_npu(num_npu):
    paths = od_npu_path_ids(num_npu)
    qos = baseline_qos_config("od_baseline", num_npu)
    assert len(set(paths)) == num_npu
    assert sum(qos.path_cirs) == pytest.approx(40.)
    assert {p for p, cir in enumerate(qos.path_cirs) if cir > 0} == set(paths)
    assert {p for p, w in enumerate(qos.path_weights) if w > 0} == set(paths)
    assert all(qos.path_cirs[p] == 40. / num_npu for p in paths)
    groups = Counter(p // 32 for p in paths)
    counts = [groups[g] for g in range(8)]
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("num_npu", (0, 257, 2.5, True))
def test_path_capacity_is_explicit(num_npu):
    with pytest.raises(ValueError):
        od_npu_path_ids(num_npu)


def test_all_backlogged_paths_receive_equal_nonpreemptive_service():
    paths = od_npu_path_ids(32)
    qos = baseline_qos_config("od_baseline", 32)
    disk = sim.DiskState(0)
    scheduler = sim.DiskIOScheduler(disk, sim.POLICY_QOS_STATIC_CIR, 40., qos)
    flows = [sim.BlockIOFlow(n, 0, k, 0, IO_GIB, p, 1, 0.)
             for n, p in enumerate(paths) for k in range(40)]
    scheduler.enqueue_many(flows, 0.)
    rates = sim._static_qos_service_rates(list(scheduler.paths[p] for p in paths),
                                          40., qos.group_weights)
    assert set(rates.values()) == {1.25}
    now, completed, by_npu = 0., [], defaultdict(list)
    for _ in range(32 * 20):
        flow = scheduler._dispatch_one(now)
        assert flow.bw == 40.  # A disk executes one whole command at a time.
        now = flow.end_time
        assert scheduler.complete_ready_flows(now) == [flow]
        completed.append(flow)
        by_npu[flow.npu_id].append(flow.block_idx)
    assert set(map(len, by_npu.values())) == {20}
    assert all(order == list(range(20)) for order in by_npu.values())
    assert now == pytest.approx(len(completed) * IO_GIB / 40. * 1000)


def test_idle_paths_lend_capacity_without_changing_static_cir():
    qos = baseline_qos_config("od_baseline", 32)
    scheduler = sim.DiskIOScheduler(sim.DiskState(0), sim.POLICY_QOS_STATIC_CIR, 40., qos)
    paths = od_npu_path_ids(32)
    active = [scheduler.paths[paths[n]] for n in (0, 8, 1)]
    rates = sim._static_qos_service_rates(active, 40., qos.group_weights)
    assert sum(rates.values()) == pytest.approx(40.)
    assert min(rates.values()) > 1.25
    assert sim._static_qos_service_rates([active[0]], 40., qos.group_weights)[active[0]] == 40.
    assert all(path.cir == 1.25 for path in active)
    # Surplus follows native two-level WRR; unbalanced active groups need
    # not receive an equal *total* rate, despite equal guaranteed CIR.
    assert rates[active[0]] == rates[active[1]] < rates[active[2]]


@pytest.mark.parametrize("runner", ("shared", "coflow", "stress"))
def test_real_wrappers_keep_io_exclusive_and_prefetch_on_owner_path(runner):
    if runner == "shared":
        from inputs.runners.run_shared_path_experiments import run_case
    elif runner == "coflow":
        from inputs.runners.run_coflow_experiments import run_case
    else:
        from inputs.runners.run_baseline_npu32_stress import run_case
    requests = trace()
    sent, received = [], []
    old_submit, old_complete = native._register_submit, native._register_complete

    def submit(context, flow):
        sent.append((flow.disk_id, flow.queue_id, flow.npu_id, flow.request_id,
                     flow.layer, flow.block_idx))
        return old_submit(context, flow)

    def complete(context, flow):
        received.append((flow.disk_id, flow.queue_id, flow.npu_id, flow.request_id,
                         flow.layer, flow.block_idx))
        return old_complete(context, flow)

    with patch.object(native, "_register_submit", submit), patch.object(native, "_register_complete", complete):
        result = run_case(requests, metadata(requests), strategy="od_baseline")
    assert all(result["summary"]["invariants"].values())
    assert len(sent) == len(received) == len(requests) * 8 * 16
    assert Counter(sent) == Counter(received)
    paths = od_npu_path_ids(4)
    assert all(path == paths[npu] for disk, path, npu, *_ in sent)
    for path_key in {(disk, path) for disk, path, *_ in sent}:
        assert [x for x in sent if x[:2] == path_key] == [x for x in received if x[:2] == path_key]
    context_stats = result["adapter_statistics"]
    assert context_stats["reorder_calls"] == 0
    assert context_stats["cir_write_events"] == []
    assert context_stats["ledger_end_counts_by_ssu"] == [0, 0, 0]
    assert context_stats["baseline_configuration"]["usable_path_count_per_ssu"] == 4
    assert sum(row["blocks"] for row in context_stats["routed_blocks_by_ssu_npu_path"]) == len(sent)
    rows = {row["request_id"]: row for row in result["summary"]["request_metrics"]}
    for rid, row in rows.items():
        if rid % 100:
            assert row["layer0_io_start_time_ms"] < row["admission_time_ms"]
    assert "simulator/policies/baselines.py" in result["core_and_policy_sha256"]


def test_asu_is_old_path0_baseline_without_timing_changes():
    from inputs.runners.run_coflow_experiments import run_case
    requests = trace()
    old = run_case(requests, metadata(requests), strategy="baseline")
    new = run_case(requests, metadata(requests), strategy="asu_baseline")
    assert old["summary"]["request_metrics"] == new["summary"]["request_metrics"]
    assert old["summary"]["microbatch_metrics"] == new["summary"]["microbatch_metrics"]
    assert old["summary"]["makespan_ms"] == new["summary"]["makespan_ms"]
    assert {r["path_id"] for r in new["adapter_statistics"]["routed_blocks_by_ssu_npu_path"]} == {0}


def test_od_rejects_accidentally_reusing_category_cirs():
    with pytest.raises(ValueError, match="OD baseline requires"):
        with shared_path_adapter("od_baseline"):
            native.simulate_continuous_batch(
                trace(), num_npu=4, num_ssu=3, n_layers=8,
                policy=sim.POLICY_QOS_STATIC_CIR,
                qos_config=baseline_qos_config("asu_baseline", 4),
                client_io_config=routing_strategy_specs()[0].client_config())
