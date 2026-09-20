#!/usr/bin/env python3
"""Frozen two-profile, 8-NPU native FIFO/Once experiments; decimal GB/s."""
from __future__ import annotations
import argparse, dataclasses, gzip, hashlib, json, math, os, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import sim
import continuous_batch_sim as native
from continuous_prefill_client import routing_strategy_specs, static_qos_config
from fixed_total_compat import exact_tail_adapter

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))

def build(config):
    profiles = {r:config['profile_'+r] for r in ('A','B')}
    s = config['ssu']; seed = config['seed']; horizon = config['horizon_ms']
    requests=[]; metadata={}; decks=[]
    for npu in range(8):
        if config['mode'] == 'fixed':
            role = 'A' if npu < config['n_A'] else 'B'
            deck = [role] * (math.ceil(horizon / (8*profiles[role]['compute_us']/1000)) + 1)
        elif config['mode'] == 'random':
            na, nb = config['input_counts']
            cycle_ms = 8*(na*profiles['A']['compute_us'] + nb*profiles['B']['compute_us'])/1000
            cycles = math.ceil(horizon/cycle_ms)+1
            deck = ['A']*(cycles*na)+['B']*(cycles*nb)
            random.Random(seed+100003*npu).shuffle(deck)
        else:
            raise ValueError(config['mode'])
        decks.append(deck)
        for generation, role in enumerate(deck):
            p=profiles[role]; rid=npu*1000000+generation
            prefix=p['ssd_prefix_tokens']; disk_gib=[0.0]*s
            layer=[]
            for block in range(math.ceil(prefix/128)):
                disk=sim.block_ring_hash_disk_id(rid,block,s)
                size=min(128,prefix-128*block)*1408/2**30
                layer.append((disk,size)); disk_gib[disk]+=size
            assert math.isclose(sum(disk_gib),p['read_gib'],abs_tol=1e-12)
            load=dict(request_id=rid,npu_id=npu,generation=generation,original_request_id=rid,
                total_tokens=p['total_tokens'],seq_len_k=p['total_length_k'],nql=p['nql'],
                role='L' if role=='A' else 'S',profile_group=role,
                ssd_prefix_tokens=prefix,category=sim.classify_request(p['total_length_k'],p['nql']),
                per_layer_us=p['compute_us'],per_layer_kv_gb=p['read_gib'],
                required_bw_input_gbps=p['B_gib_s'],arrival_time=0.,arrival_ms=0.,initial=True,
                constructed_profile=p['constructed_profile'],profile_construction=p['profile_construction'],
                original_compute_us=p['compute_us'],padding_gib_per_layer=0.,disk_gib=disk_gib)
            request=native.ContinuousBatchRequest.from_normalized(rid,npu,0.,load,(tuple(layer),))
            requests.append(request); metadata[rid]=load
    return tuple(requests),metadata,decks

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--strategy',choices=['baseline','once'],required=True)
    ap.add_argument('--force', action='store_true', help='Recompute this local result instead of using the completed cache')
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text())
    out=ROOT/'results'/config['name']/args.strategy; out.mkdir(parents=True,exist_ok=True)
    if (out/'analysis.json').exists() and not args.force:
        print(json.dumps({'cached':str(out)}),flush=True); return
    start=time.time(); requests,metadata,decks=build(config)
    fingerprint=native.continuous_batch_input_fingerprint(requests)
    save(out/'config.json',config); save(out/'requests.json',list(metadata.values()))
    f=1e9/2**30; q0=static_qos_config()
    qos=dataclasses.replace(q0,path_cirs=tuple(v*f for v in q0.path_cirs),path_pirs=tuple(v*f for v in q0.path_pirs))
    route='baseline' if args.strategy=='baseline' else 'layer_once'
    client=next(s for s in routing_strategy_specs() if s.name==route).client_config()
    print(json.dumps({'start':config['name'],'strategy':args.strategy,'requests':len(requests),'ssu':config['ssu']}),flush=True)
    with exact_tail_adapter(args.strategy,collector_interval_ms=5.0) as adapter:
        summary=native.simulate_continuous_batch(requests,num_npu=8,num_ssu=config['ssu'],
            n_layers=8,batch_size=1,policy=sim.POLICY_QOS_STATIC_CIR,qos_config=qos,client_io_config=client,
            cross_request_layer0_prefetch=True,pressure_ttl_ms=5.,disk_bw_gbps=40*f,npu_bw_gbps=50*f,
            submit_order_seed=config['seed'],control=None)
        adapter_stats=adapter.statistics()
    assert all(summary['invariants'].values()),summary['invariants']
    assert native.continuous_batch_input_fingerprint(requests)==fingerprint
    compressed_tmp=out/'native_summary.json.gz.tmp'
    with gzip.open(compressed_tmp,'wt') as stream: json.dump(summary,stream)
    with compressed_tmp.open('rb') as stream: os.fsync(stream.fileno())
    with gzip.open(compressed_tmp,'rt') as stream:
        verified=json.load(stream)
    assert verified['makespan_ms']==summary['makespan_ms']
    assert len(verified['microbatch_metrics'])==len(summary['microbatch_metrics'])
    os.replace(compressed_tmp,out/'native_summary.json.gz')
    save(out/'adapter_statistics.json',adapter_stats)
    from audit_results import analyze
    analysis=analyze(summary,metadata,config)
    analysis.update(name=config['name'],strategy=args.strategy,wall_seconds=time.time()-start,
        input_fingerprint=fingerprint,invariants=summary['invariants'],n_requests=len(requests),
        units={'capacity':'decimal GB/s','native_volume':'GiB','time':'ms'},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').glob('*.py')})
    save(out/'analysis.json',analysis)
    print(json.dumps({'done':config['name'],'strategy':args.strategy,'wall_seconds':analysis['wall_seconds'],
        'analysis':str(out/'analysis.json')},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
