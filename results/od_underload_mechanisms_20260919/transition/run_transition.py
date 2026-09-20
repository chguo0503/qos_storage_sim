"""Bounded AB-transition probes; ordinary Ring Hash, frozen synthetic compute.

No runtime barriers, arrival gating, request-ID search or core edits. Pilot
stops after 4 seconds without SLO; promising inputs replay to complete drain.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,ast,hashlib,json,math,sys,time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from inputs.manifest import save_manifest,load_manifest,write_json
from simulator.api import run_simulation
from simulator.core import continuous_batch_sim as core,sim
import metrics

IO=176*1024/2**30
class StopPilot(Exception):pass

def sources():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'simulator').rglob('*.py'))}

def prepare(cycles=16):
    table=ast.literal_eval((ROOT/'data').read_text())
    for miss in [1024,512]:
        ca=(table[32,miss][1]-22/16*(table[48,miss][1]-table[32,miss][1]))/1000
        cb=75.25
        profiles={'A':dict(total_tokens=10*1024,nql=miss,C_ms=ca),
                  'B':dict(total_tokens=200*1024,nql=2048,C_ms=cb)}
        for role,p in profiles.items():
            p['blocks']=(p['total_tokens']-p['nql'])//128
            p['V_gib']=p['blocks']*IO;p['B_gib_s']=p['V_gib']/(p['C_ms']/1000)
        name=f'ab_a10k{miss}_b200k_cb75p25_c{cycles}'
        requests=[]
        for n in range(32):
            for pos in range(cycles*2):
                role='A' if pos%2==0 else 'B';p=profiles[role];rid=n*1000000+pos
                layer=tuple((sim.block_ring_hash_disk_id(rid,b,3),IO) for b in range(p['blocks']))
                load=dict(request_id=rid,npu_id=n,original_request_id=rid,generation=pos,
                    total_tokens=p['total_tokens'],seq_len_k=p['total_tokens']/1024,nql=p['nql'],
                    role=role,category=sim.classify_request(p['total_tokens']/1024,p['nql']),
                    per_layer_us=p['C_ms']*1000,per_layer_kv_gb=p['V_gib'],
                    required_bw_input_gbps=p['B_gib_s'],arrival_time=0.,arrival_ms=0.,initial=True,
                    constructed_profile=True,compute_calibrated=False,
                    profile_construction=dict(method='32K/48K linear length extrapolation to 10K' if role=='A' else 'explicit compute scaling of data 200K/2048',
                        anchors=[{'length_k':32,'miss':miss,'weight':2.375,'C_us':table[32,miss][1]},
                                 {'length_k':48,'miss':miss,'weight':-1.375,'C_us':table[48,miss][1]}] if role=='A' else [],
                        original_data_C_us=None if role=='A' else table[200,2048][1],
                        scale=1. if role=='A' else cb*1000/table[200,2048][1],
                        caveat='synthetic timing, not measured input profile'))
                requests.append(core.ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,)))
        vectors={q.request_id:[math.fsum(v for disk,v in q.placement[0] if disk==s)*1e6/q.load['per_layer_us'] for s in range(3)] for q in requests}
        static=[sum(max(vectors[q.request_id][s] for q in requests if q.npu_id==n) for n in range(32)) for s in range(3)]
        meta=dict(case=name,profiles=profiles,num_npu=32,num_ssu=3,n_layers=8,cycles=cycles,
            seed=7,all_arrivals_zero=True,queue='(AB)^cycles on each NPU',placement='ordinary Ring Hash; rid=npu*1000000+queuepos; no address selection',
            od_queue_depth_per_ssu=8192,any_combination_demand_bound_gib_s=static,
            original_data_sha256=hashlib.sha256((ROOT/'data').read_bytes()).hexdigest(),
            units='ms and GiB; SSU40GiB/s, NPU50GiB/s; all blocks176KiB',
            scientific_scope='synthetic/extrapolated C; input≥10K; the near-synchronized AB cycle is a hypothesis, not a promised low-U result')
        save_manifest(HERE/'inputs'/f'{name}.json.gz',tuple(requests),meta)
        print(json.dumps(meta),flush=True)

def pilot_measure(ctx,requests):
    live=metrics.live_summary(ctx);byid={q.request_id:q for q in requests}
    cards=[dict(A=0.,B=0.) for _ in range(32)];active=[0.]*32
    for r in live['request_metrics']:active[r['npu_id']]+=metrics.overlap(r['admission_time_ms'],r['completion_time_ms'],2000.,4000.)
    for b in live['microbatch_metrics']:
        role=byid[b['member_request_ids'][0]].load['role']
        for l in b['layer_metrics']:cards[b['npu_id']][role]+=metrics.overlap(l['compute_start_ms'],l['compute_end_ms'],2000.,4000.)
    return dict(U_percent=100*sum(sum(c.values()) for c in cards)/64000,
        per_npu_role_compute_ms=cards,mixed_cards=sum(min(c.values())>1e-7 for c in cards),
        all_npus_active=all(abs(x-2000)<1e-7 for x in active),
        warm_demand=metrics.exact_demand(live['request_metrics'],byid,2000.,4000.),
        initial_4s_demand=metrics.exact_demand(live['request_metrics'],byid,0.,4000.),
        peak_depth=ctx.ssd_depth_peak,pilot_incomplete_no_SLO=True,observed_at_ms=ctx.current_time_ms)

def run(case,policy,pilot):
    requests,meta=load_manifest(HERE/'inputs'/f'{case}.json.gz')
    out=HERE/('pilots' if pilot else 'runs')/f'{case}_{policy}'
    out.mkdir(parents=True,exist_ok=False)
    hashes=sources();start=last=time.perf_counter();observed=0;measurement=None
    original=core._register_complete
    windows=[(2000.,4000.),(4000.,8000.)]
    if meta['cycles']>=100:windows += [(8000.,12000.),(12000.,20000.),(20000.,40000.),(40000.,60000.)]
    busy=[[0.]*3 for _ in windows]
    def observe(ctx,flow):
        nonlocal observed,last,measurement
        ret=original(ctx,flow);observed+=1
        if pilot and ctx.current_time_ms>4000.000001:
            measurement=pilot_measure(ctx,requests);raise StopPilot()
        if not pilot:
            for index,(a,z) in enumerate(windows):busy[index][flow.disk_id]+=metrics.overlap(flow.ssd_activation_time,flow.link_enqueue_time,a,z)
        if observed%20000==0 and time.perf_counter()-last>20:
            row=dict(blocks=observed,simulation_ms=ctx.current_time_ms,wall_seconds=time.perf_counter()-start)
            write_json(out/'progress.json',row);print(json.dumps(row),flush=True);last=time.perf_counter()
        return ret
    command=dict(case=case,policy=policy,pilot=pilot,status='running',source_sha256=hashes,
                 manifest_sha256=hashlib.sha256((HERE/'inputs'/f'{case}.json.gz').read_bytes()).hexdigest())
    write_json(out/'command.json',command)
    try:
        with patch.object(core,'_register_complete',observe):
            result=run_simulation(requests,strategy=policy,num_npu=32,num_ssu=3,n_layers=8,seed=7,
                od_queue_depth_per_ssu=8192 if policy=='od_baseline' else None)
        assert not pilot,'full input unexpectedly ended before4s'
    except StopPilot:
        assert pilot
        write_json(out/'metrics.json',measurement)
        command.update(status='complete_pilot',wall_seconds=time.perf_counter()-start,completed_simulation=False)
    else:
        raw=result['summary'];assert observed==raw['completed_blocks'] and all(raw['invariants'].values())
        rows=[]
        for index,(a,z) in enumerate(windows):
            row=metrics.summarize(raw,requests,a,z)
            row['physical_ssd_gib_s']=[v*40/(z-a) for v in busy[index]]
            row['measurement_definition']['own_compute']='8 × frozen synthetic/extrapolated per-layer C'
            rows.append(row)
        full=metrics.summarize(raw,requests,0.,raw['makespan_ms'],full=True)
        full['measurement_definition']['own_compute']='8 × frozen synthetic/extrapolated per-layer C'
        write_json(out/'result.json.gz',result)
        write_json(out/'analysis.json',dict(windows=rows,full=full,metadata=meta))
        command.update(status='complete',wall_seconds=time.perf_counter()-start,completed_simulation=True)
        measurement=dict(windows=[dict(window_ms=[r['start_ms'],r['end_ms']],U_percent=r['U_percent'],
            slo=r['slo'],strict_underload=r['demand']['strict_underload_all_disks'],
            per_disk_peak=r['demand']['per_disk_max_GiB_s'],any_disk_overload_percent=r['demand']['any_disk_overload_percent'],
            mixed_cards=r['role_and_stall']['npus_with_A_and_B_compute'],all_npus_active=r['all_npus_active'],
            stall_by_kind=r['role_and_stall']['io_stall']['by_kind_card_ms']) for r in rows],
            full_U_percent=full['U_percent'],makespan_ms=raw['makespan_ms'])
        write_json(out/'metrics.json',measurement)
    assert hashes==sources()
    command['source_unchanged']=True;write_json(out/'command.json',command)
    print(json.dumps(dict(case=case,policy=policy,pilot=pilot,measurement=measurement if not pilot else
          {k:v for k,v in measurement.items() if k not in ['warm_demand','initial_4s_demand','per_npu_role_compute_ms','peak_depth']})),flush=True)
    return measurement

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--cycles',type=int,default=16)
    ap.add_argument('--case');ap.add_argument('--policy',choices=['od_baseline','once'],default='od_baseline')
    ap.add_argument('--mode',choices=['pilot','full','auto'],default='auto');a=ap.parse_args()
    if a.build:prepare(a.cycles);return
    if a.mode in ['pilot','auto']:
        result=run(a.case,a.policy,True)
        if a.mode=='pilot' or result['U_percent']>=92.:return
    run(a.case,a.policy,False)

if __name__=='__main__':main()
