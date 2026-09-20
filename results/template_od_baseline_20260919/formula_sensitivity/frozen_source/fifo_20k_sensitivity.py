"""Uncalibrated20K extrapolation sensitivity, never measured/native evidence.

Uses the existing run_fixed128_32.profile allow_low calculation, which with
the actual table extrapolates from32K/48K (length weights1.75/-0.75).
Patches imported proxy profile functions only inside this isolated process.
The data, native builder, and existing proxy source files remain unchanged.
"""
from pathlib import Path
import functools,inspect,itertools,json,time
from unittest.mock import patch
import fast_multitype_probe as fm
import fast_boundary_underload as fb
from run_fixed128_32 import profile as extended_profile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/lower_fifo_followup_20260914'

@functools.lru_cache(None)
def sensitivity_profile(k,nql):
    p=extended_profile(fm.TABLE,int(round(k*1024))-nql,nql)
    return p['compute_us']/1000,p['read_gib']

def main():
    started=time.perf_counter();rows=[]
    # Retain full-deck audits and add the0.5–4s window used in final audits.
    source=inspect.getsource(fb.simulate)
    source=source.replace('windows=dict(warm=(2000,4000),first4500=(0,4500),full=(0,makespan))',
        'windows=dict(warm=(2000,4000),steady=(500,4000),first4500=(0,4500),full=(0,makespan))')
    namespace=dict(fb.__dict__);exec(source,namespace);simulate=namespace['simulate']
    grid=itertools.product((768,960,1152,1280,1344,1536),(1536,2048,3072,4096),(1,2,3,4,6,8,16,32))
    with patch.object(fm,'profile',sensitivity_profile),patch.object(fb,'profile',sensitivity_profile):
        for i,(sq,lq,w) in enumerate(grid):
            spec=dict(name=f'sensitivity20k_{i:03}',ssu=1,groups=[
                dict(id='short',role='S',total_k=20,nql=sq,weight=w),
                dict(id='long',role='L',total_k=200,nql=lq,weight=1)])
            row=simulate(spec,seed=7);rows.append(row)
            if i%24==23:print(json.dumps(dict(progress=i+1,elapsed_seconds=time.perf_counter()-started)),flush=True)
    rows.sort(key=lambda x:x['U_percent'])
    accepted=[x for x in rows if x['passed'] and x['all_active_warm'] and x['all_groups_every_npu_warm']]
    zero=[x for x in accepted if x['metrics']['full']['boundary_hard_lower_ms']<1e-8]
    nominal=[extended_profile(fm.TABLE,20*1024-q,q) for q in (768,960,1152,1280,1344,1536)]
    summarize=lambda x:dict(spec=x['spec'],U_percent=x['U_percent'],passed=x['passed'],
        full_boundary_solo_lower_card_ms=x['metrics']['full']['boundary_hard_lower_ms'],
        warm_boundary_solo_lower_card_ms=x['metrics']['warm']['boundary_hard_lower_ms'],
        warm_internal_stall_card_ms=x['metrics']['warm']['internal_stall_ms'])
    payload=dict(scope='EXTRAPOLATED SENSITIVITY ONLY.20K compute is uncalibrated, whole-layerFIFO proxy; no native evidence.',
        compute_rule='Existing run_fixed128_32.profile allow_low. Actual table first length anchors32K/48K;20K length weights1.75/-0.75, negative weights explicit in nominal_profiles. NQL interpolated inside measured grid.',
        no_original_source_data_or_native_builder_modified=True,seed=7,evaluated=len(rows),accepted=len(accepted),
        accepted_zero_full_boundary_solo_lower=len(zero),elapsed_seconds=time.perf_counter()-started,
        nominal_profiles=nominal,best_accepted=[summarize(x) for x in accepted[:10]],
        best_accepted_zero_full_boundary_solo_lower=[summarize(x) for x in zero[:10]],rows=rows)
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'theory_20k_extrapolated_sensitivity.json';path.write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:v for k,v in payload.items() if k not in ('rows','nominal_profiles')},indent=2),flush=True)
if __name__=='__main__':main()
