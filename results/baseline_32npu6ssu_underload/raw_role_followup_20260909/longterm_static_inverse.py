#!/usr/bin/env python3
"""Pure offline inverse input design; no imports of simulator or runtime hooks.

Constructed 32K profiles use a declared piecewise-linear C(NQL) approximation
between original data's NQL 1024, 2048, 4096 rows. They are NOT direct raw rows.
Caller supplies hypothetical/measured first COMPUTE starts of calibrators.
This function does not claim those starts persist after the input is changed.
"""
import ast
import bisect
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def calibration_table():
    data = ast.literal_eval((ROOT / 'data').read_text())
    knots = [(q, data[(32,q)][1]/1000) for q in (1024,2048,4096)]
    def c(q):
        left,right = (knots[0],knots[1]) if q<=2048 else (knots[1],knots[2])
        return left[1]+(right[1]-left[1])*(q-left[0])/(right[0]-left[0])
    return knots, [(q,c(q)) for q in range(1024,4097)]


def calibrate(first_compute_starts_ms, *, min_layer_c_ms=24.0, target_completion_ms=None):
    """Return integer-NQL profiles without mutating inputs or running a simulator.

    Use the first quantized C >= desired C, so predicted completion never
    precedes target and the requested minimum IO-hiding C is not rounded down.
    Input a_i are first compute starts, not automatically admission timestamps.
    """
    starts = [float(x) for x in first_compute_starts_ms]
    if not starts or any(not math.isfinite(x) for x in starts):
        raise ValueError('finite nonempty compute-start list required')
    knots,table=calibration_table()
    cs=[x[1] for x in table]
    c_min=max(float(min_layer_c_ms),cs[0])
    if c_min>cs[-1]:
        raise ValueError('minimum C exceeds original 32K4096 endpoint')
    lo=max(x+8*c_min for x in starts)
    hi=min(x+8*cs[-1] for x in starts)
    if lo>hi+1e-9:
        raise ValueError(f'one calibrator cannot align this spread: target interval [{lo}, {hi}] ms')
    target=lo if target_completion_ms is None else float(target_completion_ms)
    if not lo-1e-9<=target<=hi+1e-9:
        raise ValueError('requested target outside intersection of feasible C intervals')
    rows=[]
    for n,a in enumerate(starts):
        desired=(target-a)/8
        idx=min(len(table)-1,bisect.bisect_left(cs,desired))
        q,c=table[idx]
        hit_tokens=32768-q
        logical_bytes=hit_tokens*1408
        block_count=math.ceil(hit_tokens/128)
        physical_bytes=block_count*176*1024
        rows.append(dict(index=n,first_compute_start_assumed_ms=a,total_input_tokens=32768,
            nql=q,miss_fraction=q/32768,layer_compute_ms=c,request_compute_ms=8*c,
            desired_layer_compute_ms=desired,predicted_end_without_stall_ms=a+8*c,
            end_minus_target_ms=a+8*c-target,logical_layer_bytes=logical_bytes,
            physical_layer_bytes=physical_bytes,padding_bytes=physical_bytes-logical_bytes,
            blocks_176KiB=block_count,nominal_total_gib_s=physical_bytes/2**30*1000/c,
            constructed_profile=True))
    ends=[r['predicted_end_without_stall_ms'] for r in rows]
    return dict(no_simulation_run=True,source_data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
        model='Piecewise-linear original-data C(NQL) interpolation for fixed total length32K; rounded-up integer NQL. No C scaling independent of this declared model.',
        raw_knots_nql_C_ms=knots,target_feasible_interval_ms=[lo,hi],target_completion_ms=target,
        input_start_spread_ms=max(starts)-min(starts),predicted_completion_spread_ms=max(ends)-min(ends),
        min_layer_C_ms=min(r['layer_compute_ms'] for r in rows),
        max_layer_C_ms=max(r['layer_compute_ms'] for r in rows),
        max_total_nominal_gib_s=max(r['nominal_total_gib_s'] for r in rows),
        total_true_compute_npu_ms=math.fsum(r['request_compute_ms'] for r in rows),rows=rows,
        conditions=['Given first compute start times must be revalidated after changed NQL/IO volume.',
                    'All calibrator internal layers must have zero exposed IO stall.',
                    'Next Long L0 must finish before predicted calibrator end on every NPU.',
                    'Every strategy must receive exactly the same frozen constructed inputs; no per-policy recalibration.',
                    'No runtime synchronization, delay, barrier, event overwrite or adaptive NQL is supplied by this function.'])


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('starts_json',type=Path,help='JSON list of first compute starts in ms')
    ap.add_argument('--min-c',type=float,default=24.0)
    ap.add_argument('--target',type=float)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    result=calibrate(json.loads(args.starts_json.read_text()),min_layer_c_ms=args.min_c,target_completion_ms=args.target)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:args.output.write_text(text)
    else:print(text,end='')
