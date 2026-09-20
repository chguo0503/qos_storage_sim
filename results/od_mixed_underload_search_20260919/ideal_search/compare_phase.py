#!/usr/bin/env python3
"""Independent phase analysis of completed native output and the fluid trace."""
import csv
import gzip
import json
import math
import statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    case=HERE.parent/'runs/fixedssu1_safe38_od_local'
    raw=json.load(gzip.open(case/'result.json.gz','rt'));summary=raw['summary']
    manifest=json.load(gzip.open(case/'manifest.json.gz','rt'))
    role={q['request_id']:q['load']['role']for q in manifest['requests']}
    cb=next(q['load']['per_layer_us']/1000 for q in manifest['requests']if q['load']['role']=='B')
    ca=next(q['load']['per_layer_us']/1000 for q in manifest['requests']if q['load']['role']=='A')
    phase=[]
    for lo,hi in [(0,800),(800,1600),(1600,2400),(2400,4000),(4000,8000),(8000,16000)]:
        ts=[l['compute_start_ms']for b in summary['microbatch_metrics']if role[b['member_request_ids'][0]]=='B'
            for l in b['layer_metrics']if lo<=l['compute_start_ms']<hi]
        angles=[(t%cb)/cb*2*math.pi for t in ts]
        R=math.hypot(statistics.mean(math.cos(a)for a in angles),statistics.mean(math.sin(a)for a in angles))
        phase.append(dict(start_ms=lo,end_ms=hi,samples=len(ts),circular_concentration_R=R,
                          min_mod_ms=min(t%cb for t in ts),max_mod_ms=max(t%cb for t in ts)))
    native_blocks=[]
    for n in range(32):
        rows=sorted((b for b in summary['microbatch_metrics']if b['npu_id']==n),key=lambda b:b['admission_time_ms'])
        i=0
        while i<len(rows):
            if role[rows[i]['member_request_ids'][0]]!='A':i+=1;continue
            j=i
            while j<len(rows)and role[rows[j]['member_request_ids'][0]]=='A':j+=1
            if j<len(rows):
                st=rows[i]['layer_metrics'][0]['compute_start_ms'];end=rows[j]['layer_metrics'][0]['compute_start_ms']
                native_blocks.append(dict(npu=n,A_start_ms=st,next_B_compute_start_ms=end,
                                          A_block_through_next_B_ready_ms=end-st,A_requests=j-i,
                                          A_pure_compute_ms=(j-i)*8*ca))
            i=j
    block_windows=[]
    for lo,hi in [(0,1600),(1600,4000),(4000,8000),(8000,16000)]:
        vals=[r['A_block_through_next_B_ready_ms']for r in native_blocks if lo<=r['A_start_ms']<hi]
        block_windows.append(dict(start_ms=lo,end_ms=hi,count=len(vals),min_ms=min(vals),mean_ms=statistics.mean(vals),max_ms=max(vals)))
    proxy=list(csv.DictReader((HERE/'fixedssu1_safe38_proxy_layer_trace.csv').open()))
    pm={(int(r['npu']),int(r['request']),int(r['layer']),r['event']):float(r['time_ms'])for r in proxy}
    early=[]
    for b in summary['microbatch_metrics']:
        rid=b['member_request_ids'][0]
        if b['npu_id']!=8 or rid%1000000 not in (1,2,36):continue
        for layer in b['layer_metrics']:
            key=(8,rid%1000000,layer['layer'])
            p=pm[key+('compute_start',)]
            early.append(dict(request_position=rid%1000000,layer=layer['layer'],native_start_ms=layer['compute_start_ms'],
                proxy_start_ms=p,start_native_minus_proxy_ms=layer['compute_start_ms']-p,
                native_io_duration_ms=layer['io_ready_time_ms']-layer['io_start_time_ms']))
    output=dict(case=str(case.relative_to(HERE.parent)),C_A_ms=ca,C_B_ms=cb,
        phase_metric='R=abs(mean(exp(2*pi*j*(B_compute_start modulo C_B)/C_B))); 1 means same phase, 0 means phases spread around the circle.',
        phase_windows=phase,A_block_windows=block_windows,A_blocks=native_blocks,matched_NPU8_early_layers=early,
        proxy_result=json.loads((HERE/'fixedssu1_safe38_exact_manifest_proxy.json').read_text()),
        service_observation=json.loads((HERE/'native_service_integral.json').read_text()),
        core_references={'nominal_rates':'sim.py:686','new_idle_path_resets_virtual_finish':'sim.py:1152',
                         'packet_virtual_finish_selection':'sim.py:1053','client_issue_interval':'continuous_batch_sim.py:3089'})
    (HERE/'proxy_native_gap.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
