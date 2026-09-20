"""Independently audit frozen calibration manifests and already completed runs."""
from pathlib import Path
import ast,gzip,hashlib,json,math,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from inputs.manifest import load_manifest,write_json
from simulator.core import sim

def audit(name,strategy='od_baseline'):
    reqs,meta=load_manifest(HERE/'inputs'/f'{name}.json.gz')
    byid={r.request_id:r for r in reqs};assert len(byid)==len(reqs)
    physical={};computes={};table=ast.literal_eval((ROOT/'data').read_text())
    for r in reqs:
        l=r.load;n=r.npu_id;p=l['position'];cycle=l['cycle'];key=n,p
        assert r.arrival_time_ms==0 and l['total_tokens']==128*1024
        assert l['physical_prefix_id']==n*1000000+p
        assert r.request_id==n*1000000+cycle*3+p
        assert l['nql']==(256 if p==0 else 4096)
        expected=tuple((sim.block_ring_hash_disk_id(l['physical_prefix_id'],b,3),176*1024/2**30) for b in range((128*1024-l['nql'])//128))
        assert r.placement==(expected,)
        physical.setdefault(key,r.placement);assert physical[key]==r.placement
        c=(meta.get('bootstrap_tail_B_C_ms_by_npu',meta['tail_B_C_ms_by_npu'])[n] if cycle==0 else meta['tail_B_C_ms_by_npu'][n]) if p==2 else table[128,l['nql']][1]/1000
        assert abs(l['per_layer_us']/1000-c)<1e-12
        computes.setdefault((n,cycle),[]).append(c)
    assert len(physical)==96 and len(reqs)==96*meta['cycles']
    directory=HERE/'runs'/(name if strategy=='od_baseline' else name+'_'+strategy)
    command=json.loads((directory/'command.json').read_text())
    assert command['status']=='complete' and command['source_unchanged'] and command['runtime_hooks'] is False
    assert command['manifest_sha256']==hashlib.sha256((HERE/'inputs'/f'{name}.json.gz').read_bytes()).hexdigest()
    raw=json.load(gzip.open(directory/'result.json.gz','rt'))['summary']
    assert all(raw['invariants'].values()) and raw['request_count']==len(reqs)
    expected_blocks=8*sum(len(r.placement[0]) for r in reqs)
    assert raw['completed_blocks']==expected_blocks
    native_compute=sum(l['compute_duration_ms'] for m in raw['microbatch_metrics'] for l in m['layer_metrics'])
    frozen_compute=8*sum(r.load['per_layer_us']/1000 for r in reqs)
    assert abs(native_compute-frozen_compute)<1e-5
    native_byid={m['member_request_ids'][0]:m for m in raw['microbatch_metrics']}
    for r in reqs:
        layers=native_byid[r.request_id]['layer_metrics']
        assert len(layers)==8
        assert all(abs(l['compute_duration_ms']-r.load['per_layer_us']/1000)<1e-10 for l in layers)
    out=dict(name=name,strategy=strategy,unique_logical_ids=True,physical_prefixes=96,
        every_native_hash_block_verified=True,repeated_physical_layout=True,all_arrivals_zero=True,
        A_and_first_B_original_C=True,all_tail_C_from_frozen_boot_or_steady_vector=True,
        native_compute_equals_frozen_compute=True,expected_and_completed_blocks=expected_blocks,
        manifest_sha256=command['manifest_sha256'],simulator_source_unchanged=True,
        no_runtime_hooks=True,all_native_invariants=True)
    if strategy=='od_baseline':
        d=raw['ssd_queue_depth'];assert d['per_npu_per_ssu_slots']==256
        assert max(x for row in d['peak_outstanding_blocks_by_npu_ssu'] for x in row)<=256
        assert all(d[k]==expected_blocks for k in ('activated_blocks','issued_blocks','ssd_completed_blocks','hbm_completed_blocks'))
        assert all(d[k]==0 for k in ('host_deferred_blocks_at_stop','ssd_outstanding_blocks_at_stop','link_outstanding_blocks_at_stop'))
        out['depth_per_npu_per_ssu']=256
    else:
        assert 'ssd_queue_depth' not in raw
        out['depth_per_npu_per_ssu']=None
    cal=meta['calibration']
    if cal and strategy=='od_baseline':
        cyc=cal['cycle'];before=json.load(gzip.open(HERE/'runs'/cal['source']/'result.json.gz','rt'))['summary']
        before_byid={m['member_request_ids'][0]:m for m in before['microbatch_metrics']}
        rids=[n*1000000+cyc*3+2 for n in range(32)]
        starts_before=[before_byid[rid]['layer_metrics'][0]['compute_start_ms'] for rid in rids]
        starts_after=[native_byid[rid]['layer_metrics'][0]['compute_start_ms'] for rid in rids]
        ends_before=[before_byid[rid]['completion_time_ms'] for rid in rids]
        ends_after=[native_byid[rid]['completion_time_ms'] for rid in rids]
        out['calibrated_cycle']=cyc
        out['calibrated_B0_start_max_change_ms']=max(abs(a-b) for a,b in zip(starts_before,starts_after))
        out['calibrated_B_end_spread_ms']=max(ends_after)-min(ends_after)
        out['source_B_start_spread_ms']=max(starts_before)-min(starts_before)
        out['min_new_tail_B_C_ms']=min(byid[rid].load['per_layer_us']/1000 for rid in rids)
        out['causal_sufficient_condition']=out['source_B_start_spread_ms']<out['min_new_tail_B_C_ms']
        assert out['calibrated_B0_start_max_change_ms']<1e-7
    return out

def main():
    rows=[]
    for name in ('abb_base2','abb_cal1_12','abb_cal2_bootsteady12'):
        for strategy in ('od_baseline','once'):
            directory=HERE/'runs'/(name if strategy=='od_baseline' else name+'_'+strategy)
            if (directory/'result.json.gz').exists():rows.append(audit(name,strategy))
    bycase={}
    for row in rows:bycase.setdefault(row['name'],set()).add(row['manifest_sha256'])
    assert all(len(values)==1 for values in bycase.values())
    output=dict(runs=rows,policy_comparisons_use_identical_manifest=True)
    write_json(HERE/'independent_audit.json',output);print(json.dumps(output))
if __name__=='__main__':main()
