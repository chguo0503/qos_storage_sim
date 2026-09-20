#!/usr/bin/env python3
"""Exact 128K/32K mixed inputs with distinct integer NQLs and true partial tail IOs.

C uses NQL interpolation at measured 32K/128K lengths. No length extrapolation,
arrival shifts, compute scaling, padding, or changes to original core files.
"""
from __future__ import annotations
import argparse
import ast
import bisect
from collections import Counter
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

import sim
import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from explore_fifo_underload import demand_audit
from run_baseline_npu32_stress import save_manifest, write_json
from fixed_total_compat import run_case
from run_shared_path_experiments import input_demand, logical_input_fingerprint

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/fixed128_32_underload_20260914'
LAYERS = 8
IO = 176 * 1024 / 2**30


def bracket(grid, value, allow_low=False):
    if value < grid[0]:
        if not allow_low:
            raise ValueError(f'Cannot extrapolate {value} below {grid[0]}')
        lo, hi = grid[:2]
    elif value > grid[-1]:
        raise ValueError(f'Cannot extrapolate {value} above {grid[-1]}')
    else:
        i = bisect.bisect_left(grid, value)
        if grid[i] == value:
            return grid[i], grid[i], 0.
        lo, hi = grid[i-1:i+1]
    return lo, hi, (value-lo)/(hi-lo)


def profile(table, prefix, nql):
    tokens = prefix+nql
    seq = tokens/1024
    if not 20 <= seq <= 200 or prefix <= 0 :
        raise ValueError(f'Invalid total/prefix: {tokens}, {prefix}, {nql}')
    x0,x1,wx = bracket(sorted({x for x,y in table}),seq,allow_low=True)
    y0,y1,wy = bracket(sorted({y for x,y in table}),nql)
    weights = {}
    for x,vx in ((x0,1-wx),(x1,wx)):
        for y,vy in ((y0,1-wy),(y1,wy)):
            if vx*vy:
                weights[x,y] = weights.get((x,y),0.)+vx*vy
    assert math.isclose(sum(weights.values()),1.,abs_tol=1e-12)
    c = math.fsum(table[key][1]*v for key,v in weights.items())
    d = prefix*1408/2**30
    direct = (seq,nql) in table
    assert c > 0
    if direct:
        assert math.isclose(c,table[seq,nql][1],abs_tol=1e-9)
        assert math.isclose(d,table[seq,nql][3],abs_tol=1e-12)
    return dict(total_tokens=tokens,total_length_k=seq,nql=nql,ssd_prefix_tokens=prefix,
        compute_us=c,read_gib=d,B_gib_s=d/(c/1e6),constructed_profile=not direct,
        profile_construction=dict(method='direct_data_row' if direct else
            ('bilinear_length_extrapolation_nql_interpolation' if seq<32 else 'bilinear_interpolation_length_nql'),
            source='data',anchors=[dict(seq_len_k=k[0],nql=k[1],weight=v,compute_us=table[k][1])
                for k,v in weights.items()],extrapolated=seq<32,compute_scale=1.,
            kv_formula='exact hit prefix tokens * 1408 / 2**30; partial final block'))


def pool_for(table, k, nql, jitter):
    if k not in (32,128):
        raise ValueError('This runner fixes total length to exactly 32K or 128K')
    if not 64 <= nql <= 4096:
        raise ValueError('NQL must lie inside measured [64,4096] grid')
    return [profile(table,int(k*1024)-y,y) for y in range(nql,4097)]


def case_name(a):
    label=lambda x:format(x,'g').replace('.','p')
    return (f'n{a.num_npu}_s{a.ssu}_L{label(a.long_k)}n{a.long_nql}_S{label(a.short_k)}n{a.short_nql}'
            f'_r{a.short_per_long}_{a.order}_b{a.block_long}_{a.phase}_j{a.jitter}')


def build(args):
    """Return (frozen request tuple, metadata, request-id -> per-SSD D/C)."""
    a=args
    if a.num_npu < 1 or a.ssu < 1 or a.short_per_long < 1 or a.block_long < 1:
        raise ValueError('NPU, SSD, ratio, and block-long must be positive integers')
    if not 20 <= a.short_k < a.long_k <= 200:
        raise ValueError('Require 20 <= short-k < long-k <= 200')
    table=ast.literal_eval((ROOT/'data').read_text())
    pools={role:pool_for(table,k,nql,a.jitter) for role,k,nql in
        [('L',a.long_k,a.long_nql),('S',a.short_k,a.short_nql)]}
    if any(not p for p in pools.values()):
        raise ValueError('Empty candidate profile pool')
    block=a.block_long if a.order in ('blocked','rotated') else 1
    # Start with nominal compute, then enlarge only if sampled pool minima need it.
    base={r:min(pools[r],key=lambda p:abs(p['nql']-(a.long_nql if r=='L' else a.short_nql))) for r in pools}
    cycles=max(1,math.ceil(a.horizon_ms/(LAYERS*(base['L']['compute_us']+a.short_per_long*base['S']['compute_us'])/1000))+1)
    cycles=math.ceil(cycles/block)*block
    while True:
        counts={'L':cycles,'S':cycles*a.short_per_long}
        selected={}
        for role,pool in pools.items():
            if len(pool)<counts[role]:
                raise ValueError(f'{role}: need {counts[role]} unique profiles, only {len(pool)} available; reduce horizon or use wide jitter')
            size=min(len(pool),max(counts[role]+8,math.ceil(1.25*counts[role])))
            selected[role]=pool[:size] if a.jitter in ('narrow','raw') else pool
        min_ideal=LAYERS*math.fsum(counts[r]*min(p['compute_us'] for p in selected[r]) for r in selected)/1000
        if min_ideal>a.horizon_ms:
            break
        cycles+=block
    requests=[];vectors={};lane_checks=[];maxima=[[0.]*a.ssu for _ in range(a.num_npu)]
    long_start=getattr(a,'long_start_count',None)
    long_start=a.num_npu//2 if long_start is None else long_start
    if not 0<=long_start<=a.num_npu:
        raise ValueError('long-start-count out of range')
    pattern=['L']*block+['S']*(block*a.short_per_long)
    for n in range(a.num_npu):
        rng=random.Random(a.seed+100003*n)
        role_deck=pattern*(cycles//block)
        if a.order=='random':
            rng.shuffle(role_deck)
        if a.phase=='sync':offset=0
        elif a.phase=='split':
            target='L' if n<long_start else 'S'
            offset=role_deck.index(target)
        else:offset=(n*len(pattern)//a.num_npu)%len(role_deck)
        if a.order=='rotated' and a.phase=='sync':
            offset=n%len(pattern)
        role_deck=role_deck[offset:]+role_deck[:offset]
        sampled={r:rng.sample(selected[r],counts[r]) for r in ('L','S')}
        indexes=Counter();unique=set();ideal_ms=0.
        for g,role in enumerate(role_deck):
            p=sampled[role][indexes[role]];indexes[role]+=1
            key=(p['total_tokens'],p['nql'])
            assert key not in unique
            unique.add(key);ideal_ms+=LAYERS*p['compute_us']/1000
            rid=n*1000000+g;blocks=math.ceil(p['ssd_prefix_tokens']/128)
            layer=tuple((sim.block_ring_hash_disk_id(rid,j,a.ssu),min(128,p['ssd_prefix_tokens']-128*j)*1408/2**30) for j in range(blocks))
            assert math.fsum(size for _,size in layer)==p['read_gib']
            load=dict(request_id=rid,npu_id=n,generation=g,original_request_id=rid,
                seq_len_k=p['total_length_k'],total_tokens=p['total_tokens'],nql=p['nql'],role=role,
                ssd_prefix_tokens=p['ssd_prefix_tokens'],category=sim.classify_request(p['total_length_k'],p['nql']),
                per_layer_us=p['compute_us'],per_layer_kv_gb=p['read_gib'],required_bw_input_gbps=p['B_gib_s'],
                arrival_time=0.,arrival_ms=0.,initial=True,constructed_profile=p['constructed_profile'],
                profile_construction=p['profile_construction'],original_compute_us=p['compute_us'],padding_gib_per_layer=0.)
            request=ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,))
            assert all(native._manifest_layer(request,k) is layer for k in range(LAYERS))
            requests.append(request)
            disk_bytes=[math.fsum(size for disk,size in layer if disk==s) for s in range(a.ssu)]
            vector=[size/(p['compute_us']/1e6) for size in disk_bytes]
            vectors[rid]=vector;maxima[n]=[max(x,y) for x,y in zip(maxima[n],vector)]
        assert ideal_ms>a.horizon_ms and set(role_deck)=={'L','S'}
        assert indexes==counts
        lane_checks.append(dict(npu_id=n,request_count=len(role_deck),unique_length_nql_count=len(unique),
            request_counts=dict(indexes),roles_in_execution_order=role_deck,initial_rotation=offset,
            ideal_compute_ms=ideal_ms,shuffle_seed=a.seed+100003*n))
    requests=tuple(requests)
    static=[sum(row[s] for row in maxima) for s in range(a.ssu)]
    specs={}
    for role in ('L','S'):
        actual=[q.load for q in requests if q.load['role']==role]
        specs[role]=dict(base_total_k=a.long_k if role=='L' else a.short_k,
            base_nql=a.long_nql if role=='L' else a.short_nql,
            total_length_k_range=[min(p['seq_len_k'] for p in actual),max(p['seq_len_k'] for p in actual)],
            nql_range=[min(p['nql'] for p in actual),max(p['nql'] for p in actual)],
            compute_ms_range=[min(p['per_layer_us'] for p in actual)/1000,max(p['per_layer_us'] for p in actual)/1000],
            B_gib_s_range=[min(p['required_bw_input_gbps'] for p in actual),max(p['required_bw_input_gbps'] for p in actual)],
            read_gib_range=[min(p['per_layer_kv_gb'] for p in actual),max(p['per_layer_kv_gb'] for p in actual)],
            extrapolated_count=sum(p['profile_construction']['extrapolated'] for p in actual))
    case=dict(name=case_name(a),ssu=a.ssu,num_npu=a.num_npu,mode='all_npu_mixed',long=[a.long_k,a.long_nql],
        short=[a.short_k,a.short_nql],short_per_long=a.short_per_long,order=a.order,phase=a.phase,
        block_long=block,long_start_count=long_start,jitter=a.jitter)
    metadata=dict(experiment='fixed128_32_underload_20260914',case=case,profiles=specs,
        num_npu=a.num_npu,num_ssu=a.ssu,n_layers=LAYERS,seed=a.seed,horizon_ms=a.horizon_ms,
        order=a.order,phase=a.phase,lane_checks=lane_checks,equal_176kib_blocks=all(size==IO for q in requests for _,size in q.placement[0]),
        variable_tail_io_supported=True,exact_fixed_total_tokens=[32*1024,128*1024],
        layout='block_ring_hash',placement_virtual_nodes_per_ssu=256,
        placement_key='(request_id, block_index); layer excluded; all layers reuse exact mapping',
        input_fingerprint=continuous_batch_input_fingerprint(requests),logical_input_fingerprint=logical_input_fingerprint(requests),
        input_demand=input_demand(requests,a.num_npu,a.ssu),
        per_ssu_static_upper_bound_gib_s=static,static_underload_all_request_combinations=max(static)<=40+1e-8,
        static_upper_definition='Per SSU: sum each NPU maximum D/C over its complete frozen mixed deck',
        all_length_nql_unique_within_each_npu=True,all_npus_have_both_roles_in_input=True,
        all_compute_profiles_within_measured_grid=all(p['extrapolated_count']==0 for p in specs.values()),
        data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
        workload_scope='Every NPU has both long and short requests, identical class counts per NPU; no duplicate length/NQL within a NPU. '
        'All requests arrive at t=0, batch size 1; phase only rotates the queue, never changes time. '
        'Total tokens fixed to 32*1024 or 128*1024; only NQL is interpolated in data. Partial last IO preserves exact prefix bytes. No length extrapolation, compute scaling, or padding.')
    return requests,metadata,vectors


def window_lanes(summary,requests,num_npu,left,right):
    loads={q.request_id:q.load for q in requests}
    rows=[dict(npu_id=n,by_role={r:dict(compute_ms=0.,active_ms=0.,active_request_count=0,
        computed_request_count=0,admitted_request_count=0,completed_request_count=0) for r in ('L','S')}) for n in range(num_npu)]
    clip=lambda a,b:max(0.,min(b,right)-max(a,left))
    for batch in summary['microbatch_metrics']:
        rid=batch['member_request_ids'][0];row=rows[batch['npu_id']]['by_role'][loads[rid]['role']]
        active=clip(batch['admission_time_ms'],batch['completion_time_ms'])
        compute=math.fsum(clip(l['compute_start_ms'],l['compute_end_ms']) for l in batch['layer_metrics'])
        row['active_ms']+=active;row['compute_ms']+=compute
        row['active_request_count']+=active>0;row['computed_request_count']+=compute>0
        row['admitted_request_count']+=left<=batch['admission_time_ms']<right
        row['completed_request_count']+=left<=batch['completion_time_ms']<right
    for row in rows:
        for role in row['by_role'].values():
            role['stall_ms']=role['active_ms']-role['compute_ms']
            role['U_percent']=100*role['compute_ms']/role['active_ms'] if role['active_ms'] else None
        row['both_roles_computed']=all(role['compute_ms']>0 for role in row['by_role'].values())
        row['U_percent']=100*sum(role['compute_ms'] for role in row['by_role'].values())/(right-left)
    return rows


class ReceiptObserver:
    """Read-only service accounting during the actual run; no replay required."""
    def __init__(self,requests,num_npu,num_ssu,window,policy='fifo'):
        self.requests=requests;self.left,self.right=window;self.num_npu=num_npu;self.num_ssu=num_ssu
        self.policy=policy;self.path_counts=Counter()
        self.npu=[0.]*num_npu;self.ssu=[0.]*num_ssu;self.full_npu=[0.]*num_npu;self.full_ssu=[0.]*num_ssu
        self.layers={};self.observed=0;self.nonzero=0;self.last=time.perf_counter()
        self.original=native._register_complete
    def __call__(self,context,flow):
        self.observed+=1;self.nonzero+=flow.queue_id!=0
        self.path_counts[flow.queue_id]+=1
        n,s=flow.npu_id,flow.disk_id
        received=max(0.,min(flow.link_end_time,self.right)-max(flow.link_start_time,self.left))*50/1000
        read=max(0.,min(flow.link_enqueue_time,self.right)-max(flow.ssd_activation_time,self.left))*40/1000
        self.npu[n]+=received;self.ssu[s]+=read
        self.full_npu[n]+=flow.total_gb;self.full_ssu[s]+=flow.total_gb
        row=self.layers.setdefault((flow.request_id,flow.layer),dict(request_id=flow.request_id,npu_id=n,
            layer=flow.layer,completed_blocks=0,bytes_gib=0.,first_link_start_ms=math.inf,last_link_end_ms=-math.inf,
            first_ssd_start_ms=math.inf,last_ssd_end_ms=-math.inf,window_received_gib=0.,window_ssd_read_gib=0.))
        row['completed_blocks']+=flow.block_count;row['bytes_gib']+=flow.total_gb
        row['first_link_start_ms']=min(row['first_link_start_ms'],flow.link_start_time)
        row['last_link_end_ms']=max(row['last_link_end_ms'],flow.link_end_time)
        row['first_ssd_start_ms']=min(row['first_ssd_start_ms'],flow.ssd_activation_time)
        row['last_ssd_end_ms']=max(row['last_ssd_end_ms'],flow.link_enqueue_time)
        row['window_received_gib']+=received;row['window_ssd_read_gib']+=read
        if self.observed%100000==0 and time.perf_counter()-self.last>=25:
            print(json.dumps(dict(event='progress',simulation_ms=context.current_time_ms,
                completed_requests=context.completed_requests,total_requests=len(self.requests),completed_io=self.observed)),flush=True)
            self.last=time.perf_counter()
        return self.original(context,flow)
    def payload(self,result,dest):
        checks=dict(all_expected_io_completed=self.observed==LAYERS*sum(len(q.placement[0]) for q in self.requests),
            all_simulator_invariants_passed=all(result['summary']['invariants'].values()),
            all_layer_bytes_and_blocks_match=all(self.layers[q.request_id,l]['bytes_gib']==q.load['per_layer_kv_gb']
                and self.layers[q.request_id,l]['completed_blocks']==len(q.placement[0]) for q in self.requests for l in range(LAYERS)),
            all_layer_receipt_ends_match_io_ready=all(math.isclose(self.layers[b['member_request_ids'][0],l['layer']]['last_link_end_ms'],
                l['io_ready_time_ms'],abs_tol=1e-8,rel_tol=0.) for b in result['summary']['microbatch_metrics'] for l in b['layer_metrics']),
            full_run_byte_conservation=math.isclose(math.fsum(self.full_npu),result['summary']['expected_read_gb'],abs_tol=1e-9))
        if self.policy=='once':
            checks['path_ids_within_256_path_range']=all(0<=p<256 for p in self.path_counts)
        else:
            checks['path0_only']=self.nonzero==0
        assert all(checks.values()),checks
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(schema_version=1,case=dest.name,probe_policy=result['probe_policy'],
            source_result_sha256=digest(dest/'result.json.gz'),source_manifest_sha256=digest(dest/'manifest.json.gz'),
            window_ms=[self.left,self.right],num_npu=self.num_npu,num_ssu=self.num_ssu,
            npu_link_capacity_gib_s=50.,ssu_capacity_gib_s=40.,
            measurement='Read-only hooks in this actual simulation; exact service/window intersections. No replay performed.',
            window_npu_received_gib=self.npu,window_ssu_read_gib=self.ssu,
            window_npu_receive_bandwidth_gib_s=[x*1000/(self.right-self.left) for x in self.npu],
            window_ssu_read_bandwidth_gib_s=[x*1000/(self.right-self.left) for x in self.ssu],
            full_run_npu_received_gib=self.full_npu,full_run_ssu_read_gib=self.full_ssu,
            layers=[self.layers[k] for k in sorted(self.layers)],observer_checks=checks,
            replay_checks=checks,compatibility_note='replay_checks aliases observer_checks for the existing renderer; no replay was performed',
            observer_counters=dict(completed_io=self.observed,nonzero_path_io=self.nonzero,
                completed_io_by_path=dict(sorted(self.path_counts.items()))))


class PhysicalReceiptObserver(ReceiptObserver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dt=2.; self.bin_count=int((self.right-self.left)/self.dt)
        self.ssu_bins=[[0.]*self.bin_count for _ in range(self.num_ssu)]
        self.npu_bins=[[0.]*self.bin_count for _ in range(self.num_npu)]
        self.last_ssd=[-math.inf]*self.num_ssu;self.last_npu=[-math.inf]*self.num_npu
        self.ssd_errors=0;self.npu_errors=0
        self.original_ssd=native._enqueue_link_io
    def add(self,row,start,end,rate):
        start,end=max(start,self.left),min(end,self.right)
        if start>=end:return
        i=min(self.bin_count-1,int((start-self.left)/self.dt))
        while start<end:
            stop=min(end,self.left+(i+1)*self.dt)
            row[i]+=(stop-start)*rate/1000;start=stop;i+=1
    def observe_ssd(self,context,flow,current_time_ms):
        s,e=flow.ssd_activation_time,current_time_ms;disk=flow.disk_id
        self.ssd_errors+=s<self.last_ssd[disk]-1e-8 or abs(e-s-flow.total_gb/40*1000)>1e-8
        self.last_ssd[disk]=e
        self.add(self.ssu_bins[disk],s,e,40.)
        return self.original_ssd(context,flow,current_time_ms)
    def __call__(self,context,flow):
        n=flow.npu_id;s,e=flow.link_start_time,flow.link_end_time
        self.npu_errors+=s<self.last_npu[n]-1e-8 or abs(e-s-flow.total_gb/50*1000)>1e-8
        self.last_npu[n]=e;self.add(self.npu_bins[n],s,e,50.)
        return super().__call__(context,flow)
    def payload(self,result,dest):
        r=super().payload(result,dest)
        r.update(bin_width_ms=self.dt,bin_edges_ms=[self.left+i*self.dt for i in range(self.bin_count+1)],
            ssu_bin_read_gib=self.ssu_bins,npu_bin_received_gib=self.npu_bins)
        r['observer_checks'].update(no_ssd_overlap_or_duration_error=self.ssd_errors==0,
            no_npu_overlap_or_duration_error=self.npu_errors==0,
            ssd_bins_conserve_bytes=all(math.isclose(math.fsum(row),total,abs_tol=1e-7) for row,total in zip(self.ssu_bins,self.ssu)),
            npu_bins_conserve_bytes=all(math.isclose(math.fsum(row),total,abs_tol=1e-7) for row,total in zip(self.npu_bins,self.npu)),
            no_physical_ssd_capacity_breach=all(v*1000/self.dt<=40+1e-7 for row in self.ssu_bins for v in row))
        assert all(r['observer_checks'].values()),r['observer_checks']
        return r


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-npu',type=int,default=8);p.add_argument('--ssu',type=int,default=1)
    p.add_argument('--long-k',type=float,default=128.);p.add_argument('--long-nql',type=int,default=1536)
    p.add_argument('--short-k',type=float,default=32.);p.add_argument('--short-nql',type=int,default=1216)
    p.add_argument('--short-per-long',type=int,default=8)
    p.add_argument('--order',choices=['random','alternating','blocked','rotated'],default='random')
    p.add_argument('--block-long',type=int,default=1)
    p.add_argument('--phase',choices=['split','stagger','sync'],default='sync')
    p.add_argument('--long-start-count',type=int)
    p.add_argument('--horizon-ms',type=float,default=4500.)
    p.add_argument('--window',type=float,nargs=2,default=[2000.,4000.])
    p.add_argument('--stage',default='formal');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--jitter',choices=['narrow','wide','raw'],default='narrow')
    p.add_argument('--policy',choices=['fifo','short_first','once'],default='fifo')
    p.add_argument('--capture-receipts',action='store_true',help='Always enabled; retained as an explicit CLI marker')
    p.add_argument('--prepare-only',action='store_true')
    return p


def main():
    p=parser();a=p.parse_args()
    if not 0<=a.window[0]<a.window[1]<=a.horizon_ms:p.error('Require 0 <= left < right <= horizon')
    started=time.perf_counter();requests,metadata,vectors=build(a)
    metadata['probe_policy']=a.policy
    metadata['queue_order']={'fifo':'FIFO',
        'short_first':'shortest queued request layer first; equal-byte IO, nonpreemptive',
        'once':'FIFO within each QoS path; native once per layer path selection'}[a.policy]
    strategy='once' if a.policy=='once' else 'baseline'
    dest=OUT/a.stage/f"{metadata['case']['name']}_seed{a.seed}_{a.policy}"
    if (dest/'result.json.gz').exists():raise FileExistsError(dest)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metadata.json').exists():
        old=json.loads((dest/'metadata.json').read_text())
        if old['input_fingerprint']!=metadata['input_fingerprint']:raise ValueError('Existing destination has different input')
    save_manifest(dest/'manifest.json.gz',requests,metadata);write_json(dest/'metadata.json',metadata)
    print(json.dumps(dict(event='prepared' if a.prepare_only else 'start',output=str(dest),requests=len(requests),
        case=metadata['case'],profiles=metadata['profiles'],static_upper=metadata['per_ssu_static_upper_bound_gib_s'])),flush=True)
    if a.prepare_only:return
    observer=PhysicalReceiptObserver(requests,a.num_npu,a.ssu,a.window,policy=a.policy)
    with contextlib.ExitStack() as stack:
        if a.policy=='short_first':
            from run_fifo_proof import SizePriorityPending
            volumes={q.request_id:q.load['per_layer_kv_gb'] for q in requests}
            original_init=sim.PathQueue.__init__
            def initialize(queue,*args,**kwargs):
                original_init(queue,*args,**kwargs)
                if queue.path_id==0:queue.pending=SizePriorityPending(volumes)
            stack.enter_context(patch.object(sim.PathQueue,'__init__',initialize))
        stack.enter_context(patch.object(native,'_register_complete',observer))
        stack.enter_context(patch.object(native,'_enqueue_link_io',observer.observe_ssd))
        result=run_case(requests,metadata,strategy=strategy,assignment='fixed',windows=[tuple(a.window)])
    whole=demand_audit(result['summary'],vectors,a.ssu,0.,result['summary']['makespan_ms'])
    audit=demand_audit(result['summary'],vectors,a.ssu,*a.window)
    lanes=window_lanes(result['summary'],requests,a.num_npu,*a.window)
    result.update(probe_policy=a.policy,demand_audit=audit,whole_run_demand_audit=whole,
        mixed_window_by_npu=lanes,exploration_case=metadata['case'])
    write_json(dest/'result.json.gz',result)
    write_json(dest/'receipts.json',observer.payload(result,dest))
    w=result['windows'][0];slo=result['slo']['window_admissions']['admission']
    u=lambda role:None if role not in w['by_role'] or w['by_role'][role]['active_compute_fraction'] is None else 100*w['by_role'][role]['active_compute_fraction']
    row=dict(name=metadata['case']['name'],mode='all_npu_mixed',num_npu=a.num_npu,num_ssu=a.ssu,
        seed=a.seed,strategy=strategy,policy=a.policy,window_ms=a.window,U_percent=100*w['mean_npu_utilization'],
        short_U_percent=u('S'),long_U_percent=u('L'),
        short_stall_card_ms=w['by_role'].get('S',{}).get('exposed_stall_ms',0.),
        slo_1p5_percent=100*slo['rate'] if slo['rate'] is not None else None,slo_passed=slo['passed'],slo_count=slo['count'],
        static_upper_gib_s=metadata['per_ssu_static_upper_bound_gib_s'],
        static_underload_all_request_combinations=metadata['static_underload_all_request_combinations'],
        input_average_gib_s=metadata['input_demand']['per_ssu_gib_s'],demand_audit=audit,whole_run_demand_audit=whole,
        all_npus_both_roles_computed=all(row['both_roles_computed'] for row in lanes),window_by_npu=lanes,
        all_active=w['all_npus_active_whole_window'],all_invariants_passed=all(result['summary']['invariants'].values()),
        all_length_nql_unique_within_each_npu=True,wall_seconds=time.perf_counter()-started)
    write_json(dest/'metrics.json',row)
    print(json.dumps({k:v for k,v in row.items() if k!='window_by_npu'}),flush=True)

if __name__=='__main__':main()
