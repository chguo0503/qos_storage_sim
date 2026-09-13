#!/usr/bin/env python3
"""Read-only, role-agnostic audit of a complete 32-NPU simulation case.

API: analyze_case(case_dir, windows=None, short_roles=None) -> JSON-safe dict.
CLI writes only the explicitly requested new-research output; never simulates.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOL_MS=1e-7
RATE_TOL=1e-9


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt') as stream:return json.load(stream)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*2**20),b''):digest.update(block)
    return digest.hexdigest()


def close(a,b):
    assert math.isclose(float(a),float(b),abs_tol=TOL_MS,rel_tol=1e-10),(a,b)


def overlap(a,z,left,right):return max(0.,min(z,right)-max(a,left))


def fraction(a,b,scale=1.):return scale*a/b if b else None


def describe(values):
    values=[float(v) for v in values]
    if not values:return dict(count=0,mean=None,min=None,max=None,p50=None,p90=None,p95=None,p99=None)
    return dict(count=len(values),mean=math.fsum(values)/len(values),min=min(values),max=max(values),
                **{f'p{p}':float(np.percentile(values,p)) for p in (50,90,95,99)})


def default_windows():
    return [(2000.,4000.),(2000.,20000.)]+[(float(t),float(t+2000)) for t in range(4000,20000,2000)]


def parse_window(text):
    a,z=map(float,text.split(':'))
    if not(math.isfinite(a) and math.isfinite(z) and 0<=a<z):raise ValueError('Window needs 0 <= start < end')
    return a,z


def profile_info(man,summary):
    """Actual load fields and immutable placements, without A/B constants."""
    npu_count=int(summary['num_npu']);ssu_count=int(summary['num_ssu']);layers=int(summary['n_layers'])
    assert npu_count==32 and ssu_count>0 and layers>0 and summary['batch_size']==1
    meta=man['metadata']
    assert int(meta['num_npu'])==npu_count and int(meta['num_ssu'])==ssu_count and int(meta['n_layers'])==layers
    disk_spec=meta['disk_bw_gib_s']
    disk_caps=[float(x) for x in disk_spec] if isinstance(disk_spec,list) else [float(disk_spec)]*ssu_count
    assert len(disk_caps)==ssu_count and min(disk_caps)>0
    link_cap=float(meta['npu_bw_gib_s'])
    info={};profiles={};placement_cache={}
    for request in man['requests']:
        rid=request['request_id'];q=request['load'];role=str(q.get('role','unknown'))
        assert rid not in info
        C=float(q['per_layer_us'])/1000;V=float(q['per_layer_kv_gb'])
        assert C>0 and V>=0 and math.isfinite(C) and math.isfinite(V)
        index=request['placement_index']
        if index not in placement_cache:
            layer_placements=man['placements'][index]
            assert len(layer_placements) in (1,layers)
            vectors=[]
            for placement in layer_placements:
                totals=[0.]*ssu_count
                for disk,size in placement:
                    assert 0<=int(disk)<ssu_count and float(size)>=0
                    totals[int(disk)]+=float(size)
                vectors.append(totals)
            # This study uses unchanged per-layer placements. Do not silently average
            # different layer layouts into a claimed instantaneous request demand.
            for vector in vectors[1:]:
                for a,b in zip(vector,vectors[0]):close(a,b)
            placement_cache[index]=vectors[0]
        volumes=placement_cache[index]
        close(math.fsum(volumes),V)
        npu=int(request['npu_id']);assert 0<=npu<npu_count
        info[rid]=dict(npu=npu,role=role,C_ms=C,V_GiB=V,B_GiB_s=V*1000/C,
                       disk_V_GiB=volumes,disk_B_GiB_s=[x*1000/C for x in volumes],
                       ideal_request_compute_ms=layers*C)
        key=(role,q.get('seq_len_k'),q.get('nql'),C,V)
        if key not in profiles:
            profiles[key]=dict(role=role,seq_len_k=q.get('seq_len_k'),miss_tokens=q.get('nql'),
                total_tokens=q.get('total_tokens'),hit_tokens=q.get('ssd_prefix_tokens'),C_ms=C,V_GiB=V,
                B_GiB_s=V*1000/C,requests=0,constructed_profile=q.get('constructed_profile'),
                construction=q.get('profile_construction'),source_ttft_ms=q.get('source_ttft_ms'))
        profiles[key]['requests']+=1
    return info,list(profiles.values()),disk_caps,link_cap


def nominal_segments(request_rows,info,ssu_count):
    """Exact intervals between admission/completion events; same-time changes batch."""
    events=defaultdict(lambda:dict(add=[],remove=[]))
    for row in request_rows:
        events[row['admission_time_ms']]['add'].append(row['request_id'])
        events[row['completion_time_ms']]['remove'].append(row['request_id'])
    times=sorted(events);active={};segments=[]
    for index,t in enumerate(times):
        for rid in events[t]['remove']:
            assert active.pop(info[rid]['npu'])==rid,'Overlapping/mismatched request lifetime'
        for rid in events[t]['add']:
            npu=info[rid]['npu'];assert npu not in active
            active[npu]=rid
        if index+1==len(times):break
        end=times[index+1];assert end>t
        roles=defaultdict(int)
        for rid in active.values():roles[info[rid]['role']]+=1
        segments.append(dict(start_ms=t,end_ms=end,
            per_ssu_GiB_s=[math.fsum(info[rid]['disk_B_GiB_s'][s] for rid in active.values()) for s in range(ssu_count)],
            max_card_B_GiB_s=max((info[rid]['B_GiB_s'] for rid in active.values()),default=0.),
            role_card_count=dict(roles),active_count=len(active),request_ids=list(active.values())))
    assert not active
    return segments


def nominal_window(segments,left,right,disk_caps,link_cap,roles):
    count=len(disk_caps);duration=right-left
    peak=[0.]*count;integral=[0.]*count;over=[0.]*count;equal=[0.]*count;witness=[None]*count
    any_over=link_over=link_peak=covered=active_card_ms=0.
    role_card_ms={role:0. for role in roles};role_hist={role:defaultdict(float) for role in roles}
    active_hist=defaultdict(float);interval_count=0
    for seg in segments:
        dt=overlap(seg['start_ms'],seg['end_ms'],left,right)
        if not dt:continue
        interval_count+=1;covered+=dt;active_card_ms+=dt*seg['active_count']
        active_hist[seg['active_count']]+=dt
        for role in roles:
            k=seg['role_card_count'].get(role,0);role_card_ms[role]+=k*dt;role_hist[role][k]+=dt
        for s,rate in enumerate(seg['per_ssu_GiB_s']):
            integral[s]+=rate*dt/1000
            over[s]+=dt if rate>disk_caps[s]+RATE_TOL else 0.
            equal[s]+=dt if abs(rate-disk_caps[s])<=RATE_TOL else 0.
            if rate>peak[s]:
                peak[s]=rate;witness[s]=dict(start_ms=max(left,seg['start_ms']),end_ms=min(right,seg['end_ms']),
                    demand_GiB_s=rate,role_card_count=seg['role_card_count'],active_request_ids=seg['request_ids'])
        any_over+=dt if any(x>cap+RATE_TOL for x,cap in zip(seg['per_ssu_GiB_s'],disk_caps)) else 0.
        link_peak=max(link_peak,seg['max_card_B_GiB_s'])
        link_over+=dt if seg['max_card_B_GiB_s']>link_cap+RATE_TOL else 0.
    missing=max(0.,duration-covered)
    if missing>TOL_MS:
        active_hist[0]+=missing
        for role in roles:role_hist[role][0]+=missing
    return dict(per_ssu_capacity_GiB_s=disk_caps,per_ssu_peak_GiB_s=peak,
        per_ssu_mean_GiB_s=[x*1000/duration for x in integral],per_ssu_nominal_integral_GiB=integral,
        per_ssu_over_capacity_ms=over,per_ssu_equal_capacity_ms=equal,
        any_ssu_over_capacity_ms=any_over,any_ssu_over_capacity_percent=100*any_over/duration,
        peak_disk_capacity_ratio=max((x/cap for x,cap in zip(peak,disk_caps)),default=0.),
        per_ssu_peak_witness=witness,max_card_B_GiB_s=link_peak,any_card_over_link_capacity_ms=link_over,
        strictly_below_disk_capacity=all(x<cap-RATE_TOL for x,cap in zip(peak,disk_caps)),
        within_disk_capacity=all(x<=cap+RATE_TOL for x,cap in zip(peak,disk_caps)),
        within_disk_and_link_capacity=all(x<=cap+RATE_TOL for x,cap in zip(peak,disk_caps)) and link_peak<=link_cap+RATE_TOL,
        numerical_capacity_tolerance_GiB_s=RATE_TOL,positive_event_intervals=interval_count,
        active_card_ms=active_card_ms,mean_active_card_count=active_card_ms/duration,
        role_mean_active_card_count={role:x/duration for role,x in role_card_ms.items()},
        role_active_card_count_duration_ms={role:{str(k):v for k,v in sorted(h.items())} for role,h in role_hist.items()},
        active_card_count_duration_ms={str(k):v for k,v in sorted(active_hist.items())})


def layer_stats(cycles):
    stalls=[c['stall_ms'] for c in cycles];C=math.fsum(c['C_ms'] for c in cycles);D=math.fsum(c['D_ms'] for c in cycles)
    zero=sum(x<=TOL_MS for x in stalls)
    return dict(count=len(cycles),zero_stall_count=zero,late_layer_count=len(cycles)-zero,
        late_layer_percent=fraction(len(cycles)-zero,len(cycles),100),
        stall_ms_including_zeros=describe(stalls),read_lifecycle_ms=describe([c['R_ms'] for c in cycles]),
        C_ms=describe([c['C_ms'] for c in cycles]),cycle_ms=describe([c['D_ms'] for c in cycles]),
        arithmetic_mean_cycle_U_percent=100*math.fsum(c['C_ms']/c['D_ms'] for c in cycles)/len(cycles) if cycles else None,
        duration_weighted_cycle_U_percent=fraction(C,D,100),total_compute_ms=C,total_stall_ms=math.fsum(stalls),
        total_cycle_ms=D,work_implied_mean_b_GiB_s_weighted_by_cycle_duration=fraction(math.fsum(c['V_GiB'] for c in cycles),D,1000))


def physical_window(physical,left,right,disk_caps,link_cap):
    """Exact unions of retained physical bins; never interpolate an unknown bin."""
    if physical is None:return dict(available=False,reason='No physical_service.json supplied; nominal V/C is not physical service.')
    edges=physical['bin_edges_ms']
    if left not in edges or right not in edges:
        return dict(available=False,reason='Window boundaries are not physical-bin edges; exact sub-bin service cannot be reconstructed.')
    a,z=edges.index(left),edges.index(right);duration=right-left
    ssd=physical['ssd_busy_ms'];link=physical['npu_link_busy_ms']
    assert len(ssd)==len(disk_caps) and len(link)==32
    for rows in (ssd,link):
        for bins in rows:
            assert len(bins)==len(edges)-1
            for index,busy in enumerate(bins):assert -TOL_MS<=busy<=edges[index+1]-edges[index]+1e-5
    assert all(abs(cap-physical['ssd_gib_per_second'])<RATE_TOL for cap in disk_caps)
    close(link_cap,physical['npu_link_gib_per_second'])
    disk_busy=[math.fsum(bins[a:z]) for bins in ssd]
    link_busy=[math.fsum(bins[a:z]) for bins in link]
    disk_work=[ms*cap/1000 for ms,cap in zip(disk_busy,disk_caps)]
    received=[ms*link_cap/1000 for ms in link_busy]
    def stock(index):
        return math.fsum(math.fsum(bins[:index])*cap/1000 for bins,cap in zip(ssd,disk_caps))-math.fsum(math.fsum(bins[:index])*link_cap/1000 for bins in link)
    start_stock,end_stock=stock(a),stock(z)
    assert start_stock>=-1e-5 and end_stock>=-1e-5
    difference=math.fsum(disk_work)-math.fsum(received)
    close(difference,end_stock-start_stock)
    return dict(available=True,bin_indices=list(range(a,z)),
        per_ssu_busy_ms=disk_busy,per_ssu_utilization_percent=[100*x/duration for x in disk_busy],
        per_ssu_mean_service_GiB_s=[x*1000/duration for x in disk_work],
        total_mean_service_GiB_s=math.fsum(disk_work)*1000/duration,
        per_npu_link_busy_ms=link_busy,per_npu_received_GiB=received,
        per_npu_mean_received_GiB_s=[x*1000/duration for x in received],
        total_mean_received_GiB_s=math.fsum(received)*1000/duration,
        total_ssd_service_GiB=math.fsum(disk_work),total_npu_received_GiB=math.fsum(received),
        served_minus_received_GiB_at_start=start_stock,served_minus_received_GiB_at_end=end_stock,
        served_minus_received_change_GiB=difference,physical_work_difference_matches_stock_change=True)


def slo_stats(sample,alpha,clock):
    delays=[r['completion_time_ms']-r[clock+'_time_ms'] for r in sample]
    passed=sum(delay<=alpha*r['own_compute_ms']+1e-9 for delay,r in zip(delays,sample))
    return dict(count=len(sample),passed=passed,percent=fraction(passed,len(sample),100),
                latency_ms=describe(delays),slowdown=describe([d/r['own_compute_ms'] for d,r in zip(delays,sample)]))


def slo_window(request_rows,info,roles,left,right,alphas=(1.5,2.)):
    sample=[r for r in request_rows if left<=r['admission_time_ms']<right]
    arrivals=[r for r in request_rows if left<=r['arrival_time_ms']<right]
    by_role={role:[r for r in sample if info[r['request_id']]['role']==role] for role in roles}
    result=dict(admission_cohort_count=len(sample),admission_cohort_request_ids=sorted(r['request_id'] for r in sample),
        completed_after_window_count=sum(r['completion_time_ms']>right for r in sample),
        arrival_cohort_count=len(arrivals),threshold_basis='alpha * own_compute_ms = alpha * n_layers * request per_layer_compute_ms',
        metric='Admission-to-prefill-completion proxy; no first-token event and no pre-admission queue in primary metric.',alphas={})
    for alpha in alphas:
        admission=slo_stats(sample,alpha,'admission')
        per_role={role:slo_stats(rows,alpha,'admission') for role,rows in by_role.items()}
        assert sum(x['count'] for x in per_role.values())==admission['count']
        assert sum(x['passed'] for x in per_role.values())==admission['passed']
        result['alphas'][str(float(alpha))]=dict(admission=admission,per_role=per_role,
            arrival_clock_same_admission_cohort=slo_stats(sample,alpha,'arrival'),
            arrival_cohort=slo_stats(arrivals,alpha,'arrival'))
    return result


def analyze_case(case_dir,windows=None,short_roles=None):
    case=Path(case_dir).resolve();manifest=case/'manifest.json.gz';result=case/'result.json.gz'
    man,raw=read(manifest),read(result);summary=raw['summary']
    assert all(summary['invariants'].values()),summary['invariants']
    assert raw['input_fingerprint']==man['input_fingerprint']
    sources={str(p):sha(p) for p in (manifest,result)}
    command_path=case/'command.json'
    command=None
    if command_path.exists():
        command=read(command_path)
        assert command['status']=='complete' and command.get('completed_simulation',True)
        assert command['manifest_sha256']==sources[str(manifest)]
        assert command['output_sha256']==sources[str(result)]
        sources[str(command_path)]=sha(command_path)
    # Use the repository's canonical manifest loader to verify its fingerprint.
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from run_baseline_npu32_stress import load_manifest
    loaded,metadata=load_manifest(manifest)
    assert len(loaded)==len(man['requests'])
    info,profiles,disk_caps,link_cap=profile_info(man,summary)
    physical=None;physical_path=case/'physical_service.json'
    if physical_path.exists():
        physical=read(physical_path);sources[str(physical_path)]=sha(physical_path)
        if command is not None and 'physical_service_sha256' in command:
            assert sources[str(physical_path)]==command['physical_service_sha256']
        assert physical['input_fingerprint']==man['input_fingerprint'] and physical['core_unchanged']
        assert physical['observed_completed_blocks']==physical['expected_blocks']
        if raw.get('strategy')=='baseline':assert physical['baseline_nonzero_paths']==0
    roles=sorted({q['role'] for q in info.values()})
    if short_roles is None:
        short_roles=metadata.get('short_roles')
        if short_roles is None:
            short_roles=[role for role in roles if role=='S' or (role.startswith('S') and role[1:].isdigit())]
    short_roles=list(short_roles)
    assert set(short_roles)<=set(roles),(short_roles,roles)
    long_roles=[role for role in roles if role not in short_roles]
    request_rows=summary['request_metrics']
    assert len(request_rows)==len(info) and {r['request_id'] for r in request_rows}==set(info)
    for row in request_rows:
        assert math.isfinite(row['completion_time_ms'])
        assert row['arrival_time_ms']<=row['admission_time_ms']<=row['completion_time_ms']
        close(row['own_compute_ms'],info[row['request_id']]['ideal_request_compute_ms'])
    by_npu={n:[] for n in range(32)};internal=[];request_layer_map={}
    for batch in summary['microbatch_metrics']:
        assert len(batch['member_request_ids'])==1
        rid=batch['member_request_ids'][0];q=info[rid];npu=batch['npu_id'];assert q['npu']==npu
        assert rid not in request_layer_map;request_layer_map[rid]=batch
        layers=batch['layer_metrics'];assert len(layers)==int(summary['n_layers'])
        by_npu[npu].append(batch)
        previous_end=batch['admission_time_ms']
        for index,layer in enumerate(layers):
            assert layer['layer']==index
            close(layer['compute_end_ms']-layer['compute_start_ms'],q['C_ms'])
            close(layer['compute_start_ms']-previous_end,layer['io_barrier_wait_ms'])
            previous_end=layer['compute_end_ms']
            if not index:continue
            previous=layers[index-1]
            start=previous['compute_start_ms'];deadline=previous['compute_end_ms'];end=layer['compute_start_ms']
            close(layer['io_start_time_ms'],start)
            close(end,max(deadline,layer['io_ready_time_ms']))
            stall=max(0.,layer['io_ready_time_ms']-deadline);C=deadline-start;D=end-start
            close(stall,layer['io_barrier_wait_ms']);close(C+stall,D)
            internal.append(dict(request_id=rid,npu=npu,role=q['role'],read_layer=index,
                start_ms=start,end_ms=end,deadline_ms=deadline,C_ms=C,D_ms=D,stall_ms=stall,
                R_ms=layer['io_ready_time_ms']-start,V_GiB=q['V_GiB']))
        close(layers[-1]['compute_end_ms'],batch['completion_time_ms'])
    assert set(request_layer_map)==set(info)
    for n,lane in by_npu.items():
        lane.sort(key=lambda b:b['admission_time_ms'])
        for a,z in zip(lane,lane[1:]):assert a['completion_time_ms']<=z['admission_time_ms']+TOL_MS
    segments=nominal_segments(request_rows,info,len(disk_caps))
    chosen=[]
    for a,z in (default_windows() if windows is None else windows):
        pair=(float(a),float(z));assert math.isfinite(a) and math.isfinite(z) and 0<=a<z
        if pair not in chosen:chosen.append(pair)
    results=[];warnings=[]
    for left,right in chosen:
        duration=right-left;denominator=32*duration
        classes={role:dict(compute_ms=0.,active_ms=0.,first_layer_stall_ms=0.,internal_layer_stall_ms=0.,
                           computed_npu_ids=[],admitted_requests=0,completed_requests=0) for role in roles}
        per_npu=[]
        for npu,lane in by_npu.items():
            role_compute={role:0. for role in roles};role_active={role:0. for role in roles}
            for batch in lane:
                rid=batch['member_request_ids'][0];role=info[rid]['role'];g=classes[role]
                active=overlap(batch['admission_time_ms'],batch['completion_time_ms'],left,right)
                compute=math.fsum(overlap(l['compute_start_ms'],l['compute_end_ms'],left,right) for l in batch['layer_metrics'])
                role_compute[role]+=compute;role_active[role]+=active;g['compute_ms']+=compute;g['active_ms']+=active
                g['admitted_requests']+=int(left<=batch['admission_time_ms']<right)
                g['completed_requests']+=int(left<=batch['completion_time_ms']<right)
                previous=batch['admission_time_ms']
                for index,layer in enumerate(batch['layer_metrics']):
                    key='first_layer_stall_ms' if index==0 else 'internal_layer_stall_ms'
                    g[key]+=overlap(previous,layer['compute_start_ms'],left,right)
                    previous=layer['compute_end_ms']
            C=math.fsum(role_compute.values());active=math.fsum(role_active.values())
            assert -TOL_MS<=C<=active+TOL_MS<=duration+2*TOL_MS
            computed=[role for role in roles if role_compute[role]>TOL_MS]
            for role in computed:classes[role]['computed_npu_ids'].append(npu)
            short_C=math.fsum(role_compute[role] for role in short_roles)
            long_C=math.fsum(role_compute[role] for role in long_roles)
            binary_mixed=(short_C>TOL_MS and long_C>TOL_MS) if short_roles and long_roles else None
            per_npu.append(dict(npu=npu,U_percent=100*C/duration,compute_ms=C,active_ms=active,
                io_stall_ms=max(0.,active-C),idle_ms=max(0.,duration-active),
                active_whole_window=abs(active-duration)<=TOL_MS,computed_roles=computed,
                all_roles_computed=set(computed)==set(roles),role_compute_ms=role_compute,role_active_ms=role_active,
                short_roles_compute_ms=short_C,long_roles_compute_ms=long_C,long_short_mixed=binary_mixed))
        for role,g in classes.items():
            g['exposed_stall_ms']=g['active_ms']-g['compute_ms']
            close(g['exposed_stall_ms'],g['first_layer_stall_ms']+g['internal_layer_stall_ms'])
            g['conditional_U_percent']=fraction(g['compute_ms'],g['active_ms'],100)
            g['window_card_time_share_percent']=100*g['active_ms']/denominator
            g['U_contribution_pp']=100*g['compute_ms']/denominator
            g['computed_npu_count']=len(g['computed_npu_ids'])
        U=math.fsum(p['U_percent'] for p in per_npu)/32
        close(U,math.fsum(g['U_contribution_pp'] for g in classes.values()))
        refs=[r for r in raw.get('windows',[]) if r['start_ms']==left and r['end_ms']==right]
        for ref in refs:
            close(U,100*ref['mean_npu_utilization'])
            for n,p in enumerate(per_npu):
                close(p['compute_ms'],ref['compute_ms_by_npu'][n]);close(p['active_ms'],ref['active_ms_by_npu'][n])
        released=[c for c in internal if left<=c['start_ms']<right]
        complete=[c for c in internal if left<=c['start_ms'] and c['end_ms']<=right]
        nominal=nominal_window(segments,left,right,disk_caps,link_cap,roles)
        close(nominal['active_card_ms'],math.fsum(g['active_ms'] for g in classes.values()))
        row=dict(start_ms=left,end_ms=right,duration_ms=duration,denominator_card_ms=denominator,U_percent=U,
            all_32_active=all(p['active_whole_window'] for p in per_npu),
            mixed_card_count=sum(p['all_roles_computed'] for p in per_npu),
            long_short_mixed_card_count=sum(p['long_short_mixed'] is True for p in per_npu) if short_roles and long_roles else None,
            long_short_mixed_pass=all(p['long_short_mixed'] is True for p in per_npu) if short_roles and long_roles else None,
            per_npu=per_npu,classes=classes,
            internal_layers_released_in_window=dict(overall=layer_stats(released),
                per_role={role:layer_stats([c for c in released if c['role']==role]) for role in roles},
                completed_after_window_count=sum(c['end_ms']>right for c in released)),
            complete_internal_cycles_inside_window=dict(overall=layer_stats(complete),
                per_role={role:layer_stats([c for c in complete if c['role']==role]) for role in roles}),
            nominal=nominal,physical=physical_window(physical,left,right,disk_caps,link_cap),
            slo=slo_window(request_rows,info,roles,left,right))
        if short_roles:
            row['short_roles_internal_released']=layer_stats([c for c in released if c['role'] in short_roles])
        if not row['all_32_active']:warnings.append(f'Window [{left},{right}) ms has idle/drained card time; fixed denominator is retained.')
        if row['mixed_card_count']<32:warnings.append(f'Window [{left},{right}) ms: only {row["mixed_card_count"]}/32 cards computed every role.')
        if not nominal['within_disk_capacity']:warnings.append(f'Window [{left},{right}) ms is not per-disk instantaneous nominal underload.')
        results.append(row)
    ideal=[]
    for npu in range(32):
        qs=[q for q in info.values() if q['npu']==npu]
        C=math.fsum(q['ideal_request_compute_ms'] for q in qs)
        volumes=[math.fsum(q['disk_V_GiB'][s]*summary['n_layers'] for q in qs) for s in range(len(disk_caps))]
        ideal.append(dict(npu=npu,pure_compute_ms=C,full_input_GiB=math.fsum(volumes),
                          per_ssu_mean_if_no_stall_GiB_s=[fraction(x,C,1000) for x in volumes]))
    ideal_per_disk=[math.fsum(p['per_ssu_mean_if_no_stall_GiB_s'][s] or 0 for p in ideal) for s in range(len(disk_caps))]
    return dict(schema_version=1,all_technical_checks_passed=True,case_dir=str(case),
        strategy=raw.get('strategy'),order=metadata.get('order',metadata.get('order_mode')),seed=metadata.get('seed'),
        num_npu=32,num_ssu=len(disk_caps),n_layers=summary['n_layers'],roles=roles,short_roles=short_roles,long_roles=long_roles,
        profiles=profiles,metadata_profiles=metadata.get('profiles'),input_fingerprint=man['input_fingerprint'],
        sources=sources,builder_sha256=sha(Path(__file__)),no_new_simulation=True,
        completed_request_count=len(request_rows),makespan_ms=summary['makespan_ms'],
        ideal_no_stall_population=dict(per_npu=ideal,per_ssu_GiB_s=ideal_per_disk,
            total_GiB_s=math.fsum(ideal_per_disk),total_capacity_GiB_s=math.fsum(disk_caps),
            mean_demand_capacity_ratio=math.fsum(ideal_per_disk)/math.fsum(disk_caps)),
        definitions=dict(U='Exact compute overlap / (32 * common window duration), including idle time in denominator.',
            class_U='Class compute overlap / class admitted-active overlap; includes first-layer waiting.',
            nominal='Current admitted request actual per-disk per-layer V/C on every positive event interval; next-request L0 is not added again. This is reference demand, not measured physical I/O.',
            nominal_layer_layout='Requires constant per-request per-disk layer volumes; rejects varying layer layouts instead of averaging them.',
            physical='Exact SSD-service and NPU-link interval integrals from retained 2-second bins; available only for windows whose boundaries are retained bin edges.',
            internal_release_cohort='Next-layer prefetch release occurs within half-open window; retain full C+stall and completion beyond the window; include every zero stall.',
            complete_internal_cycles='Only same-request cycles entirely inside the window; layer-average utilization is distinct from class active-time U.',
            internal_C_over_D='C / (C + exposed next-layer stall); this uses completion logs, without physical trace it is not an independent bandwidth measurement.',
            slo='Admission cohort, followed to final prefill completion; thresholds 1.5 or 2 times own pure compute. Proxy for TTFT, not a first-token event or arrival-to-first-token measurement.',
            role_selection='Roles come from each request load.role. Explicit short_roles wins; otherwise use metadata.short_roles or canonical S/S1/S2 names only. Never infer A/B short roles.',
            mixed_card_count='Count of cards with positive compute from every distinct role; retains the stronger all-role-coverage definition.',
            long_short_mixed='Positive compute from any specified short role and any remaining long role on the same card. For L/S1/S2, requires L and at least one short subtype, not both. Pass means all 32 cards meet this mixture condition; active-whole-window is reported separately.'),
        warnings=warnings,windows=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window',type=parse_window,action='append')
    parser.add_argument('--short-role',action='append',default=None)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(HERE),'Write analysis only inside the new research directory'
    result=analyze_case(args.case,windows=args.window,short_roles=args.short_role)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_name(out.name+'.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(out)
    print(json.dumps(dict(output=str(out),technical_passed=True,strategy=result['strategy'],
        roles=result['roles'],windows=[dict(start_ms=r['start_ms'],end_ms=r['end_ms'],U_percent=r['U_percent'],
            all_32_active=r['all_32_active'],mixed_card_count=r['mixed_card_count'],
            any_disk_over_ms=r['nominal']['any_ssu_over_capacity_ms']) for r in result['windows']]),ensure_ascii=False))


if __name__=='__main__':main()
