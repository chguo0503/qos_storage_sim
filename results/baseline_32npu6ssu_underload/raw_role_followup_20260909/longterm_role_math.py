#!/usr/bin/env python3
"""Offline arithmetic for recurring role changes. Never imports/runs a simulator.

The proposed sequences are inputs to experiments, not simulated low-U results.
All C and volumes come from original data rows; placement is explicitly new.
"""
import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IO_BYTES = 176 * 1024
NUM_NPU, NUM_SSU, DISK, LINK, LAYERS = 32, 6, 40.0, 50.0, 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profiles():
    data = ast.literal_eval((ROOT / 'data').read_text())
    out = {}
    for key in ((32, 1024), (48, 1024), (64, 1024), (160, 1024), (176, 1024)):
        bw, c_us, ttft, source_volume = data[key]
        blocks = math.ceil((key[0] * 1024 - key[1]) / 128)
        volume = blocks * IO_BYTES / 2**30
        assert math.isclose(volume, source_volume, abs_tol=1e-12)
        c_ms = c_us / 1000
        out[key] = dict(key=list(key), category='SL' if key[0] <= 80 else 'LL',
            layer_compute_ms=c_ms, request_compute_ms=LAYERS*c_ms,
            layer_volume_gib=volume, layer_blocks=blocks, padding_bytes=0,
            total_V_over_C_gib_s=1000*volume/c_ms,
            source_bandwidth_gib_s=bw, source_78layer_ttft_ms=ttft,
            by_npu_disk_V_gib=[[sum((b+n) % NUM_SSU == s for b in range(blocks))*IO_BYTES/2**30
                              for s in range(NUM_SSU)] for n in range(NUM_NPU)])
    return out


def capacity(ps):
    # Every NPU may select any of these profiles at any time. Max BEFORE sum.
    bounds = [math.fsum(max(p['by_npu_disk_V_gib'][n][s]*1000/p['layer_compute_ms']
                           for p in ps) for n in range(NUM_NPU)) for s in range(NUM_SSU)]
    return dict(placement='SSU=(block_index+npu_id)%6; new placement, not inherited v8 placement',
        arbitrary_current_profile_choice_peak_by_disk_gib_s=bounds,
        peak_gib_s=max(bounds), capacity_pass=max(bounds)<DISK,
        max_link_gib_s=max(p['total_V_over_C_gib_s'] for p in ps),
        link_pass=max(p['total_V_over_C_gib_s'] for p in ps)<LINK,
        definition='Sum over NPU of per-NPU maximum current layer V_s/C across allowed profiles.',
        exclusion='Cross-request prefetched L0 remains in actual IO but is excluded from this requested current-request nominal definition.')


def long_wave(p, count):
    work = [math.fsum(p['by_npu_disk_V_gib'][n][s] for n in range(count))
            for s in range(NUM_SSU)]
    return dict(long_cards=count, service_work_ms_by_disk=[x/DISK*1000 for x in work],
        note='Hypothetical simultaneous Long-layer wave. Full-layer work is not atomic FIFO ownership or a guaranteed Short delay.')


def build():
    ps = profiles()
    s1,s2,s3,l160,l176 = [ps[k] for k in ((32,1024),(48,1024),(64,1024),(160,1024),(176,1024))]
    recipes=[]
    for q in (4,8):
        # One design has two groups of the SAME q, not different q per group.
        a=[1]*q+[0]*(3*q)
        b=[0]*(3*q)+[1]*q
        repeats=32//q
        cs=s1['layer_compute_ms']; cl=l160['layer_compute_ms']
        recipes.append(dict(name=f'balanced16_L160_q{q}', profile_keys=[s1['key'],l160['key']],
            groups=[dict(npus=list(range(16)),cycle_profile_indices=a),
                    dict(npus=list(range(16,32)),cycle_profile_indices=b)],
            repeats=repeats, per_npu_profile_counts=[96,32], per_npu_requests=128,
            per_npu_pure_compute_ms=96*8*cs+32*8*cl,
            long_segment_pure_compute_ms=q*8*cl,
            short_segment_pure_compute_ms=3*q*8*cs,
            conditional_short_U=3*cs/cl,
            conditional_fleet_U=(1+3*cs/cl)/2,
            conditional_scope='Only if the Short cohort sustains three completed compute layers per Long layer period, groups stay phase concentrated, and role swaps retain this relation. Not a prediction, theorem, or observed result.',
            deliberate_synchronization='All sixteen cards within a group have identical profile sequences each cycle. Submission RNG and block-striping remain physical simulator behavior. No delays, barriers, or time-triggered role switches.',
            multiple_short_profiles=False,
            capacity=capacity([s1,l160]), wave=long_wave(l160,16)))
    for nlong in (16,20):
        # Exact parent-proposed population; repetition count here is a sizing example.
        front=[3]*8+[0,1,2]
        rear=[0]*22+[1,2,3]
        pp=[s1,s2,s3,l176]
        front_c=math.fsum(pp[i]['request_compute_ms'] for i in front)
        rear_c=math.fsum(pp[i]['request_compute_ms'] for i in rear)
        nf=math.ceil(14000/front_c); nr=math.ceil(14000/rear_c)
        recipes.append(dict(name=f'mostly_fixed_L176_n{nlong}', profile_keys=[p['key'] for p in pp],
            groups=[dict(npus=list(range(nlong)),cycle_profile_indices=front,repeats_example=nf,
                         pure_compute_ms_per_cycle=front_c,
                         short_fraction_of_cycle_pure_C=sum(p['request_compute_ms'] for p in pp[:3])/front_c),
                    dict(npus=list(range(nlong,32)),cycle_profile_indices=rear,repeats_example=nr,
                         pure_compute_ms_per_cycle=rear_c,
                         long_fraction_of_cycle_pure_C=l176['request_compute_ms']/rear_c)],
            min_per_npu_pure_compute_ms_example=min(nf*front_c,nr*rear_c),
            initial_long_segment_pure_compute_ms=8*l176['request_compute_ms'],
            multiple_short_profiles=True,
            deliberate_synchronization='Each group repeats a common exact profile cycle. Preserve especially the first group Short subsequence across NPUs; independently permuting that subsequence changes the proposed synchronization mechanism.',
            limitations=['Long initial segment pure C exceeds 2000ms, so no proof that every sliding 2-second subwindow contains both roles on every NPU.',
                         'Pure-C role shares are not actual occupied-time role shares or per-window positive-compute shares.',
                         'The two groups have different cycle lengths. Their relative phases can drift even if each group remains internally concentrated.'],
            capacity=capacity(pp), wave=long_wave(l176,nlong)))
    return dict(no_simulation_run=True, data_sha256=sha(ROOT/'data'), script_sha256=sha(__file__),
        old_mechanism_sha256=sha(HERE/'window_sensitivity/repeated_queues/repeated_mechanism.json'),
        v8_generator_sha256=sha(HERE/'mixed_design_v8.py'), constants=dict(npu=32,ssu=6,disk_gib_s=40,link_gib_s=50,layers=8,IO_bytes=IO_BYTES),
        raw_profiles=[{k:v for k,v in p.items() if k!='by_npu_disk_V_gib'} for p in ps.values()],
        recipes=recipes,
        universal_limits=['Nominal underload limits average work per compute budget, not instantaneous released work or every layer deadline.',
            'Identical within-group templates reduce one cause of phase spread; queue feedback and randomized block submission may still destroy or reinforce concentration.',
            'All t=0 finite queues must have enough per-lane pure compute for the target horizon; actual active and both-role conditions still need event validation.',
            'Do not extrapolate low initial-window U to later cycles. Inspect separate 2-second windows, cumulative U, group phase modulo CL, role counts, and Short profile-conditioned utilization.',
            'Balanced single-Short recipes are mechanism probes and do not satisfy an additional several-distinct-Short-profiles requirement.',
            'For an ordering comparison, construct a shuffled complete lane with exactly the same per-NPU population and per-request placement; rerun both policies on each identical manifest.'])


if __name__ == '__main__':
    result=build()
    target=HERE/'longterm_role_math.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(target),'recipes':[{k:r[k] for k in ('name','capacity')} for r in result['recipes']]},ensure_ascii=False))
