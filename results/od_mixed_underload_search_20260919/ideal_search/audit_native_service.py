#!/usr/bin/env python3
"""Read-only observer replay to 845 ms; no policy, input or core-source changes."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import contextlib
import hashlib
import io
import json
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from simulator.core import sim
from simulator.core import continuous_batch_sim as native
from inputs.runners.run_baseline_npu32_stress import load_manifest,run_case
from inputs.runners.run_coflow_experiments import source_files

class Stop(Exception):pass

def main():
    src=HERE.parent/'runs/fixedssu1_safe38_od_local/manifest.json.gz'
    reqs,meta=load_manifest(src)
    before={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in source_files()}
    dispatch=sim.DiskIOScheduler._dispatch_one;complete=native._register_complete;enqueue=sim.DiskIOScheduler.enqueue_many
    target_path=1  # OD NPU8 path id.
    rows=[];completed=[];rate_events=[];floor_resets=[];started=time.time()
    def record_rate(self,t,event):
        paths=[p for p in self.paths.values()if p.pending]
        rates=sim._static_qos_service_rates(paths,self.disk_bw,self.group_weights)
        rate_events.append(dict(time_ms=t,event=event,rate=rates.get(self.paths[target_path],0.),backlogged=len(paths)))
    def observe_enqueue(self,flows,t):
        watch=self.state.disk_id==1 and 804.9<=t<843
        if watch:
            changed=any(not self.paths[f.queue_id].pending for f in flows)
            target_new=any(f.queue_id==target_path for f in flows)and not self.paths[target_path].has_work()
            old_finish=self.paths[target_path].virtual_finish
        result=enqueue(self,flows,t)
        if watch and changed:record_rate(self,t,'enqueue_after')
        if watch and target_new:floor_resets.append(dict(time_ms=t,old_finish=old_finish,
            new_finish=self.paths[target_path].virtual_finish,request=flows[0].request_id,layer=flows[0].layer))
        return result
    def observe_dispatch(self,t):
        watch=self.state.disk_id==1 and 804.9<=t<843 and not self.state.active_flows
        if watch:
            paths=[p for p in self.paths.values()if p.pending]
            rates=sim._static_qos_service_rates(paths,self.disk_bw,self.group_weights)
            target=self.paths[target_path];head=target.peek()
            row=dict(start_ms=t,backlogged=len(paths),target_rate=rates.get(target,0.),
                     target_pending=target.pending_io_count,target_finish_before=target.virtual_finish,
                     floor_before=min((p.virtual_finish for p in self.paths.values()if p.has_work()),default=0.),
                     target_request=None if head is None else head.request_id,
                     target_layer=None if head is None else head.layer)
        flow=dispatch(self,t)
        if watch and flow is not None:
            row.update(end_ms=flow.end_time,npu=flow.npu_id,request=flow.request_id,layer=flow.layer,
                       selected_path=flow.queue_id,selected_finish_after=self.paths[flow.queue_id].virtual_finish,
                       GiB=flow.total_gb)
            rows.append(row)
            record_rate(self,t,'dispatch_after')
        return flow
    def observe_complete(context,flow):
        ret=complete(context,flow)
        if flow.npu_id==8 and flow.disk_id==1 and 804.9<=context.current_time_ms<843:
            completed.append(dict(request=flow.request_id,layer=flow.layer,enqueue=flow.enqueue_time,
                ssd_start=flow.ssd_activation_time,ssd_end=flow.link_enqueue_time,
                link_start=flow.link_start_time,link_end=flow.link_end_time,GiB=flow.total_gb))
        if context.current_time_ms>845:raise Stop()
        return ret
    logs=io.StringIO()
    try:
        with contextlib.redirect_stdout(logs),patch.object(sim.DiskIOScheduler,'_dispatch_one',observe_dispatch),patch.object(sim.DiskIOScheduler,'enqueue_many',observe_enqueue),patch.object(native,'_register_complete',observe_complete):
            run_case(reqs,meta,strategy='od_baseline',assignment='fixed',windows=((2000.,4000.),))
    except Stop:pass
    after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in source_files()}
    assert before==after
    output=dict(observer='read-only command dispatch and completion hooks',
                full_run=False,stopped_ms=845,measurement_ssu=1,target_npu=8,
                source_sha256=before,manifest_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                source_unchanged=True,wall_s=time.time()-started,dispatches=rows,completed=completed,
                rate_events=rate_events,floor_resets=floor_resets)
    (HERE/'native_service_observation.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(dispatches=len(rows),completed=len(completed),wall_s=time.time()-started)),flush=True)

if __name__=='__main__':main()
