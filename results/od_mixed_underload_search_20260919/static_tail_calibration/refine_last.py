#!/usr/bin/env python3
"""One causal refinement: only the last cohort's pre-frozen P profiles change."""
import gzip
import hashlib
import json
from experiment import HERE,native,load_manifest,save_manifest,logical_input_fingerprint,run

def main():
    source=HERE/'inputs/tail_gain_0.5.json.gz';trace=HERE/'runs/tail_gain_0.5/live_summary.json.gz'
    qs,meta=load_manifest(source);s=json.load(gzip.open(trace,'rt'));byid={q.request_id:q for q in qs};CB=meta['calibration']['C_B_ms'];anchor=meta['calibration']['target_anchor_ms'];corrections={}
    for n in range(24,32):
        rows=[b for b in s['microbatch_metrics']if b['npu_id']==n];i=next(i for i,b in enumerate(rows)if byid[b['member_request_ids'][0]].load['role']=='A');j=i
        while byid[rows[j]['member_request_ids'][0]].load['role']=='A':j+=1
        target=anchor+(i+1)*8*CB;actual=rows[j]['layer_metrics'][0]['compute_start_ms'];gap=max(0.,target-actual)
        corrections[n]=dict(target_ms=target,observed_ms=actual,advance_ms=gap,extra_C_ms=.3*gap/7)
    out=[]
    for q in qs:
        load=dict(q.load)
        if q.npu_id in corrections and load['input_kind']=='P':
            old=load['per_layer_us']/1000;new=old+corrections[q.npu_id]['extra_C_ms']
            load.update(per_layer_us=new*1000,required_bw_input_gbps=load['per_layer_kv_gb']*1000/new,
                profile_construction=dict(method='offline frozen last-cohort refinement',previous_C_ms=old,C_ms=new,
                    formula='previous C + 0.3 * initial-cycle B advance / 7',source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
        out.append(native.ContinuousBatchRequest.from_normalized(q.request_id,q.npu_id,q.arrival_time_ms,load,q.placement))
    out=tuple(out);label='tail_gain_0.5_group4_refine';meta=dict(meta,label=label,case_id=label,
        input_fingerprint=native.continuous_batch_input_fingerprint(out),logical_input_fingerprint=logical_input_fingerprint(out),
        refinement=dict(source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                        corrections=corrections,cohort='NPU24..31 only',runtime_control=False,gain=.3))
    meta['calibration']=dict(meta['calibration'],refinement=meta['refinement'])
    path=HERE/'inputs'/f'{label}.json.gz';save_manifest(path,out,meta)
    print(json.dumps(dict(refinement=corrections)),flush=True)
    run(path,12)

if __name__=='__main__':main()
