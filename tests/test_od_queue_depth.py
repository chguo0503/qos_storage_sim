"""Behavioral tests for OD host backpressure, not just counter clipping."""
from collections import Counter
from unittest.mock import patch

import pytest

from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core, sim
from simulator.adapters.shared_path import shared_path_adapter
from simulator.config import routing_strategy_specs
from simulator.policies.od_baseline import qos_config, queue_depth_per_npu

IO = 176*1024/2**30


def request(rid, npu, disks, compute_us=20.):
    load = dict(request_id=rid, npu_id=npu, seq_len_k=32, nql=128,
                category='SS', per_layer_us=compute_us,
                per_layer_kv_gb=len(disks)*IO,
                required_bw_input_gbps=len(disks)*IO/(compute_us/1e6))
    return core.ContinuousBatchRequest(rid,npu,0.,load,
                                       (tuple((s,IO) for s in disks),))


def run(requests, *, npu=2, ssu=2, depth=None, layers=3, link=50.):
    return run_simulation(requests,strategy='od_baseline',num_npu=npu,
                          num_ssu=ssu,n_layers=layers,npu_bw_gib_s=link,
                          od_queue_depth_per_ssu=depth)['summary']


def observe_run(requests, **kwargs):
    submissions=[];ssd=[];hbm=[]
    original_submit=core._register_submit
    original_ssd=core._enqueue_link_io
    original_hbm=core._register_complete
    def submit(context,flow):
        key=(flow.npu_id,flow.disk_id,flow.request_id,flow.layer,flow.block_idx)
        submissions.append((key,context.current_time_ms,
                            [list(r) for r in context.ssd_depth_outstanding],
                            sum(len(s.blocks)-s.cursor for s in context.submission_states.values()
                                if s.npu_id==flow.npu_id and s.disk_id==0)))
        return original_submit(context,flow)
    def device(context,flow,now):
        ssd.append(((flow.npu_id,flow.disk_id,flow.request_id,flow.layer,flow.block_idx),now))
        return original_ssd(context,flow,now)
    def received(context,flow):
        hbm.append(((flow.npu_id,flow.disk_id,flow.request_id,flow.layer,flow.block_idx),context.current_time_ms))
        return original_hbm(context,flow)
    with patch.object(core,'_register_submit',submit),patch.object(core,'_enqueue_link_io',device),patch.object(core,'_register_complete',received):
        result=run(requests,**kwargs)
    return result,submissions,ssd,hbm


def test_total_depth_is_fixed_equal_share_and_od_only():
    assert queue_depth_per_npu(8192,32)==256
    assert queue_depth_per_npu(None,32) is None
    for invalid in [0,-1,8191,1.2,True]:
        with pytest.raises(ValueError):queue_depth_per_npu(invalid,32)
    with pytest.raises(ValueError,match='only supported'):
        run_simulation([request(0,0,[0])],strategy='once',num_npu=1,num_ssu=1,
                       od_queue_depth_per_ssu=8192)


def test_depth_none_and_unreached_limit_preserve_all_business_events():
    qs=[request(n*10+g,n,[0,1]*4) for n in range(2) for g in range(2)]
    old=run(qs)
    explicit_none=run(qs,depth=None)
    assert old==explicit_none
    assert 'ssd_queue_depth' not in old
    limited=run(qs,depth=8192)
    for key in ['request_metrics','microbatch_metrics','makespan_ms','events_processed',
                'completed_blocks','completed_read_gb','fleet_npu_compute_utilization','disk_stats']:
        assert old[key]==limited[key],key
    assert sum(map(sum,limited['ssd_queue_depth']['blocked_state_episodes_by_npu_ssu']))==0


def test_depth_one_pauses_real_issue_and_releases_before_hbm():
    # Slow link makes the release-at-SSD vs release-at-HBM distinction large.
    qs=[request(0,0,[0]*8)]
    limited,sent,device,received=observe_run(qs,npu=1,ssu=1,depth=1,layers=1,link=1.)
    first_device=device[0][1]
    assert sent[1][1]==pytest.approx(first_device)
    assert sent[1][1] < received[0][1]
    assert sent[1][1] > .0001 # It really paused beyond the 0.1-us issue tick.
    assert limited['ssd_queue_depth']['peak_outstanding_blocks_by_npu_ssu']==[[1]]
    assert len(sent)==len(device)==len(received)==8
    assert limited['ssd_queue_depth']['blocked_state_episodes_by_npu_ssu'][0][0]>0
    assert limited['ssd_queue_depth']['host_blocked_state_ms_by_npu_ssu'][0][0]>0
    assert limited['microbatch_metrics'][0]['layer_metrics'][0]['io_start_time_ms']==0.
    assert all(limited['invariants'].values())


def test_full_disk_does_not_block_other_disk_and_issue_pacing_survives():
    qs=[request(0,0,[0,1]*12),request(10,1,[0]*24)]
    result,sent,_,_=observe_run(qs,depth=2,layers=1)
    # NPU0 continues disk1 while disk0 is full and retains an unissued suffix.
    assert any(key[0]==0 and key[1]==1 and counts[0][0]==1 and deferred>0
               for key,t,counts,deferred in sent)
    for npu in [0,1]:
        times=[t for (n,s,r,l,b),t,_,_ in sent if n==npu]
        assert all(b-a>=.0001-1e-12 for a,b in zip(times,times[1:]))
    assert all(result['invariants'].values())


def test_idle_npus_cannot_lend_slots_but_bandwidth_still_borrows():
    # Reserve 32*256 slots, but use just NPU0. Its queue cannot exceed 256.
    result,sent,device,_=observe_run([request(0,0,[0]*300)],npu=32,ssu=1,
                                    depth=8192,layers=1)
    d=result['ssd_queue_depth']
    assert d['peak_outstanding_blocks_by_npu_ssu'][0][0]==256
    assert all(row==[0] for row in d['peak_outstanding_blocks_by_npu_ssu'][1:])
    # SSD runs one command every IO/40 seconds despite guaranteed CIR=1.25.
    assert device[-1][1]-device[0][1]==pytest.approx(299*IO/40*1000)
    assert d['activated_blocks']==d['issued_blocks']==d['ssd_completed_blocks']==300
    assert d['host_deferred_blocks_at_stop']==d['ssd_outstanding_blocks_at_stop']==0


def test_layer_zero_prefetch_uses_same_shared_quota_and_preserves_fifo():
    qs=[request(g,0,[0]*9,compute_us=2.) for g in range(3)]
    result,sent,device,_=observe_run(qs,npu=1,ssu=1,depth=1,layers=3)
    assert result['cross_request_layer0_prefetches']==2
    assert [key for key,*_ in sent]==[key for key,t in device]
    assert {key[2] for key,*_ in sent}=={0,1,2}
    assert {key[3] for key,*_ in sent}=={0,1,2}
    assert result['ssd_queue_depth']['peak_outstanding_blocks_by_npu_ssu']==[[1]]
    assert all(result['invariants'].values())


def test_multi_request_batch_cannot_reset_quota_for_new_submission_state():
    qs=[request(0,0,[0]*6),request(1,0,[0]*6)]
    client=next(s for s in routing_strategy_specs() if s.name=='baseline').client_config()
    with shared_path_adapter(strategy='od_baseline'):
        result=core.simulate_continuous_batch(qs,num_npu=1,num_ssu=1,n_layers=2,
            batch_size=2,qos_config=qos_config(1),client_io_config=client,
            ssd_queue_depth_per_npu=1)
    assert result['ssd_queue_depth']['peak_outstanding_blocks_by_npu_ssu']==[[1]]
    assert result['ssd_queue_depth']['activated_blocks']==24
    assert all(result['invariants'].values())


def test_submission_batch_is_truncated_to_available_slots():
    from dataclasses import replace
    qs=[request(0,0,[0]*11)]
    client=next(s for s in routing_strategy_specs() if s.name=='baseline').client_config()
    client=replace(client,submit_batch_size=8)
    with shared_path_adapter(strategy='od_baseline'):
        result=core.simulate_continuous_batch(qs,num_npu=1,num_ssu=1,n_layers=1,
            batch_size=1,qos_config=qos_config(1),client_io_config=client,
            ssd_queue_depth_per_npu=3)
    assert result['ssd_queue_depth']['peak_outstanding_blocks_by_npu_ssu']==[[3]]
    assert result['ssd_queue_depth']['issued_blocks']==11
    assert all(result['invariants'].values())
