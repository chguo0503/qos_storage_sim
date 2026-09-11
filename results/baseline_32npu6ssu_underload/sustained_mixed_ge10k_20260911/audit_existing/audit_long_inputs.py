#!/usr/bin/env python3
"""Audit only the three 7200-second-runner frozen inputs; no simulation."""
import ast,csv,hashlib,json,math,random
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import audit_sustained as audit

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'long_validation_7200s'
ROOT=HERE.parents[3]

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def csvwrite(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    table=ast.literal_eval((ROOT/'data').read_text());planpath=STUDY/'plan.json';plan=audit.read(planpath)
    paths=sorted({Path(j['input']['manifest']) for j in plan['jobs']});assert len(paths)==3 and len(plan['jobs'])==5
    assert all(p.parent.resolve()==(STUDY/'inputs').resolve() for p in paths)
    sources={str(p):audit.sha(p) for p in [Path(__file__),HERE/'audit_sustained.py',ROOT/'data',ROOT/'run_baseline_npu32_stress.py',planpath]}
    sourcechecks={};specpath=Path(plan['spec']);sourcechecks['plan_spec_sha']=audit.sha(specpath)==plan['spec_sha256'];sources[str(specpath)]=audit.sha(specpath)
    commands=[];expected_core=None
    for j in plan['jobs']:
        p=STUDY/'runs'/j['input']['label']/j['strategy']/'command.json';cmd=audit.read(p)
        core=cmd['core_source_sha256']
        if expected_core is None:expected_core=core
        k=j['input']['label']+'/'+j['strategy']
        sourcechecks[k+':same_core_keyset_and_sha']=core==expected_core
        sourcechecks[k+':current_core']=bool(core) and all((ROOT/f).is_file() and audit.sha(ROOT/f)==s for f,s in core.items())
        for f in core:sources[str(ROOT/f)]=audit.sha(ROOT/f)
        wrapper=HERE.parent/'run_candidates.py';arch=HERE.parent/'source_snapshots'/cmd['runner_sha256']/'run_candidates.py'
        sourcechecks[k+':wrapper_and_archive']=audit.sha(wrapper)==audit.sha(arch)==cmd['runner_sha256']
        sources[str(wrapper)]=audit.sha(wrapper);sources[str(arch)]=audit.sha(arch)
        sourcechecks[k+':input_binding']=cmd['manifest_sha256']==j['input']['manifest_sha256']==audit.sha(j['input']['manifest']) and cmd['input_fingerprint']==j['input']['input_fingerprint']
        commands.append(dict(path=str(p),snapshot_sha256=audit.sha(p),status_at_input_audit=cmd['status'],strategy=j['strategy'],label=j['input']['label'],
            core_source_sha256=core,runner_sha256=cmd['runner_sha256'],manifest_sha256=cmd['manifest_sha256'],input_fingerprint=cmd['input_fingerprint']))
    manifests={};checks={};profiles_csv=[];assignment_csv=[]
    for path in paths:
        m,inputs,im=audit.manifest_info(path,table);label=m['metadata']['label'];spec=m['metadata']['construction_spec'];mode=m['metadata']['order'];seed=m['metadata']['seed']
        sources[str(path)]=audit.sha(path);cc=dict(im['checks']);cc['pure_C_ge62000']=True;cc['population_matches_actual_cycles']=True;cc['random_shuffle_exact']=True
        cc['source_data_sha']=m['metadata']['source_data_sha256']==m['metadata']['source']['source_sha256']==audit.sha(ROOT/'data')
        cc['all_direct_data_no_scaling_padding']=all(r['load']['profile_construction']=={'method':'direct_data_row'} and r['load']['constructed_profile'] is False and r['load']['padding_gib_per_layer']==0 and r['load']['original_compute_us']==r['load']['per_layer_us'] for r in m['requests'])
        cc['static_bound_matches_independent']=all(audit.close(a,b) for a,b in zip(im['static_per_ssu_upper_bound_gib_s'],m['metadata']['active_profile_rate_certificate']['per_ssu_upper_bound_gib_s']))
        population_C=math.fsum(i['own_C_ms'] for i in inputs.values());profilecount=Counter(i['profile'] for i in inputs.values());lane_stats=[]
        for n in range(32):
            lane=sorted((r for r in m['requests'] if r['npu_id']==n),key=lambda x:x['request_id']);counts=Counter(inputs[r['request_id']]['profile'] for r in lane)
            C=math.fsum(inputs[r['request_id']]['own_C_ms'] for r in lane);roleC={role:math.fsum(inputs[r['request_id']]['own_C_ms'] for r in lane if inputs[r['request_id']]['role']==role) for role in ('short','long')}
            fast=math.fsum(inputs[r['request_id']]['own_C_ms'] for r in lane if inputs[r['request_id']]['role']=='short' and inputs[r['request_id']]['nql']==1024)
            bridge=roleC['short']-fast;cc['pure_C_ge62000'] &= C>=62000
            cycle=list(spec['long_cycle'] if n<spec['long_cards'] else spec['short_cycle'])
            if n>=spec['long_cards'] and spec.get('short_phase_stagger'):
                shift=((n-spec['long_cards'])*len(cycle))//(32-spec['long_cards']);cycle=cycle[shift:]+cycle[:shift]
            reps=math.ceil(spec['horizon_ms']/(8*math.fsum(m['metadata']['profiles'][i]['per_layer_compute_us']/1000 for i in cycle)))
            order=cycle*reps;expected=list(enumerate(order))
            if mode=='random':random.Random(seed*1000003+n*100003+71923).shuffle(expected)
            cc['population_matches_actual_cycles'] &= len(lane)==len(expected) and Counter(x['load']['profile_index'] for x in lane)==Counter(order)
            exact=all(r['request_id']==n*1000000+pos and r['load']['generation']==pos and r['load']['original_request_id']==n*1000000+orig and r['load']['profile_index']==index and r['load']['original_cycle']==orig//len(cycle) and r['load']['original_cycle_position']==orig%len(cycle) for pos,(r,(orig,index)) in enumerate(zip(lane,expected)))
            cc['random_shuffle_exact'] &= exact
            ids=[r['load']['original_request_id'] for r in lane];cc.setdefault('original_ids_unique',True);cc['original_ids_unique'] &= len(ids)==len(set(ids))
            row=dict(label=label,order=mode,npu=n,queue_requests=len(lane),pure_C_ms=C,short_C_ms=roleC['short'],long_C_ms=roleC['long'],
                short_C_fraction=roleC['short']/C,long_C_fraction=roleC['long']/C,fast_short_nql1024_C_ms=fast,fast_short_C_fraction=fast/C,
                bridge_nql4096_C_ms=bridge,bridge_C_fraction=bridge/C,
                count_S32_nql1024=counts['32:1024'],count_S48_nql1024=counts['48:1024'],count_S64_nql1024=counts['64:1024'],count_bridge32_nql4096=counts['32:4096'],count_Long176_nql1024=counts['176:1024'],
                cycle_repetitions=reps,ordered_cycle_profile_indices=json.dumps(cycle),
                first24_actual_profiles=','.join(inputs[r['request_id']]['profile'] for r in lane[:24]),original_id_queue_sha256=digest(ids))
            assignment_csv.append(row);lane_stats.append(row)
        for p in im['profiles']:
            key=f"{p['seq_len_k']}:{p['nql']}";count=profilecount[key];totalC=8*p['raw_C_ms']*count
            profiles_csv.append(dict(label=label,order=mode,profile=key,role=p['role'],category=p['category'],total_input_tokens=p['total_tokens'],nql=p['nql'],
                per_layer_C_ms=p['raw_C_ms'],request_8layer_C_ms=8*p['raw_C_ms'],per_layer_actual_D_MiB=p['actual_D_MiB'],
                per_layer_block_count=round(p['actual_D_MiB']*1024/176),request_count=count,request_count_fraction=count/len(inputs),population_C_ms=totalC,population_C_fraction=totalC/population_C,
                direct_data_row=True,padding_bytes_per_layer=0))
        checks[label]=cc;manifests[label]=dict(path=str(path),sha256=audit.sha(path),input_fingerprint=im['fingerprint'],request_count=len(inputs),
            mode=mode,minimum_pure_C_ms=min(x['pure_C_ms'] for x in lane_stats),maximum_pure_C_ms=max(x['pure_C_ms'] for x in lane_stats),
            minimum_role_pure_C_fraction=min(x[r+'_C_fraction'] for x in lane_stats for r in ('short','long')),
            minimum_fast_short_pure_C_fraction=min(x['fast_short_C_fraction'] for x in lane_stats),
            static_per_ssu_upper_bound_gib_s=im['static_per_ssu_upper_bound_gib_s'],static_certificate_pass=im['static_sufficient_certificate_pass'],
            population_pure_C_ms=population_C,actual_profile_counts=dict(profilecount))
    ordered=next(p for p in paths if '6l_' in p.name and '_ordered_' in p.name);randomp=next(p for p in paths if '6l_' in p.name and '_random_' in p.name)
    def canonical(p):
        m=audit.read(p);out={}
        for r in m['requests']:
            rid=r['load']['original_request_id'];assert rid not in out
            load={k:v for k,v in r['load'].items() if k not in ('request_id','generation')}
            out[rid]=(r['npu_id'],r['arrival_time_ms'],load,m['placements'][r['placement_index']])
        return out
    left,right=canonical(ordered),canonical(randomp);pair=dict(ordered=str(ordered),random=str(randomp),same_original_id_population=set(left)==set(right),
        per_card_full_load_and_actual_placement_equal=left==right,canonical_population_sha256=digest(left),
        excluded_fields=['outer request_id','load.request_id','load.generation'],explanation='These fields encode queue position; original_request_id restores identity. NPU, arrival, all other load fields and every actual placement block remain equal.')
    passed=all(sourcechecks.values()) and all(all(c.values()) for c in checks.values()) and pair['per_card_full_load_and_actual_placement_equal']
    unchanged=all(audit.sha(p)==s for p,s in sources.items());passed &= unchanged
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=passed,source_checks=sourcechecks,input_checks=checks,manifests=manifests,random_ordered_pair=pair,
        source_sha256=sources,sources_unchanged=unchanged,execution_source_snapshots=commands,
        limits=['Input-only audit: no utilization, completion or warm role-switch claim before raw results complete.',
          'Pure C>=62 seconds implies enough work for the 60-second measurement; runtime activity and role shares are still checked from logs.',
          'NQL4096 bridge is a distinct 28.59ms-compute raw profile; it is retained separately from NQL1024 fast short profiles.',
          'The cancelled long_validation directory is excluded. Command status/hash here is a point-in-time source snapshot while simulations run.'])
    csvwrite('input_assignment.csv',assignment_csv);csvwrite('input_profiles.csv',profiles_csv)
    (HERE/'long_input_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=passed,manifests=manifests,pair_pass=pair['per_card_full_load_and_actual_placement_equal']),ensure_ascii=False))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
