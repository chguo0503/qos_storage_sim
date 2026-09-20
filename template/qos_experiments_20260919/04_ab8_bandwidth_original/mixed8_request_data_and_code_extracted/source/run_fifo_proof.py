#!/usr/bin/env python3
"""Matched-input FIFO versus size-priority causal probe; native Path0 data plane."""
from pathlib import Path
from unittest.mock import patch
import argparse, contextlib, heapq, importlib, json, time
import sim
import continuous_batch_sim as native
import explore_fifo_underload as raw
from run_baseline_npu32_stress import run_case,save_manifest,write_json

class SizePriorityPending:
    """Order only already-enqueued equal-sized I/Os by request layer bytes."""
    def __init__(self,volumes): self.heap=[];self.sequence=0;self.volumes=volumes
    def append(self,flow):
        heapq.heappush(self.heap,(self.volumes[flow.request_id],self.sequence,flow));self.sequence+=1
    def popleft(self): return heapq.heappop(self.heap)[2]
    def __bool__(self): return bool(self.heap)
    def __len__(self): return len(self.heap)
    def __getitem__(self,index):
        if index!=0:raise IndexError(index)
        return self.heap[0][2]
    def __iter__(self):return (x[2] for x in sorted(self.heap))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True)
    p.add_argument('--policy',choices=['fifo','short_first','once'],default='fifo')
    p.add_argument('--horizon-ms',type=float,default=4500.);p.add_argument('--window',nargs=2,type=float,default=[2000.,4000.])
    p.add_argument('--seed',type=int,default=7);p.add_argument('--stage',default='formal')
    a=p.parse_args();case=raw.CASES[a.case];requests,metadata,vectors=raw.build(case,a.horizon_ms,a.seed)
    metadata['queue_order']='FIFO' if a.policy in ['fifo','once'] else 'ascending request per-layer bytes, FIFO ties, reselect each completed IO'
    metadata['probe_policy']=a.policy
    dest=raw.OUT/a.stage/f"{case['name']}_seed{a.seed}_{a.policy}";dest.mkdir(parents=True,exist_ok=True)
    if (dest/'result.json.gz').exists():raise FileExistsError(dest)
    save_manifest(dest/'manifest.json.gz',requests,metadata);write_json(dest/'metadata.json',metadata)
    volumes={q.request_id:q.load['per_layer_kv_gb'] for q in requests}
    original_init=sim.PathQueue.__init__;original_complete=native._register_complete
    trace=[];observed=nonzero_paths=0;started=time.perf_counter();last=started
    trace_window=(a.window[0]-200,a.window[0]+400)
    def initialize(queue,*args,**kwargs):
        original_init(queue,*args,**kwargs)
        if queue.path_id==0:queue.pending=SizePriorityPending(volumes)
    def observe(context,flow):
        nonlocal observed,nonzero_paths,last
        observed+=1;nonzero_paths+=flow.queue_id!=0
        left,right=trace_window
        if (flow.ssd_activation_time<right and flow.link_enqueue_time>left) or (flow.link_start_time<right and flow.link_end_time>left):
            trace.append([flow.request_id,flow.npu_id,flow.layer,flow.block_idx,flow.disk_id,flow.queue_id,flow.total_gb,
                flow.enqueue_time,flow.ssd_activation_time,flow.link_enqueue_time,flow.link_start_time,flow.link_end_time])
        if observed%100000==0 and time.perf_counter()-last>=25:
            print(json.dumps(dict(event='progress',case=case['name'],policy=a.policy,completed_requests=context.completed_requests,
                total_requests=len(requests),simulation_ms=context.current_time_ms,completed_io=observed)),flush=True);last=time.perf_counter()
        return original_complete(context,flow)
    print(json.dumps(dict(event='start',case=case,policy=a.policy,requests=len(requests),static_upper=metadata['per_ssu_static_upper_bound_gib_s'])),flush=True)
    with contextlib.ExitStack() as stack:
        if a.policy=='short_first':stack.enter_context(patch.object(sim.PathQueue,'__init__',initialize))
        stack.enter_context(patch.object(native,'_register_complete',observe))
        result=run_case(requests,metadata,strategy='once' if a.policy=='once' else 'baseline',assignment='fixed',windows=[tuple(a.window)])
    assert observed==8*sum(len(q.placement[0]) for q in requests)
    if a.policy!='once':assert nonzero_paths==0
    assert all(result['summary']['invariants'].values())
    result['probe_policy']=a.policy
    result['demand_audit']=raw.demand_audit(result['summary'],vectors,case['ssu'],*a.window)
    result['io_audit']=dict(observed_completed_io=observed,all_expected_io_completed=True,nonzero_path_io=nonzero_paths)
    write_json(dest/'result.json.gz',result)
    write_json(dest/'trace.json.gz',dict(window_ms=trace_window,columns=['request_id','npu_id','layer','block_idx','ssu_id','path_id','size_gib','enqueue_ms','ssd_start_ms','ssd_end_ms','link_start_ms','link_end_ms'],rows=trace))
    w=result['windows'][0];slo=result['slo']['window_admissions']['admission']
    row=dict(name=case['name'],case_index=a.case,policy=a.policy,seed=a.seed,num_ssu=case['ssu'],window_ms=a.window,
        U_percent=100*w['mean_npu_utilization'],short_U_percent=100*w['by_role']['S']['active_compute_fraction'],
        long_U_percent=100*w['by_role']['L']['active_compute_fraction'],
        short_stall_card_ms=w['by_role']['S']['exposed_stall_ms'],slo_1p5_percent=100*slo['rate'],slo_passed=slo['passed'],slo_count=slo['count'],
        all_active=w['all_npus_active_whole_window'],demand_audit=result['demand_audit'],
        static_upper=metadata['per_ssu_static_upper_bound_gib_s'],average_demand=metadata['input_demand']['per_ssu_gib_s'],
        wall_seconds=time.perf_counter()-started)
    write_json(dest/'metrics.json',row);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
