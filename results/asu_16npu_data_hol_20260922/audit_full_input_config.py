#!/usr/bin/env python3
"""Read-only independent audit of the final frozen workload; no simulator imports."""
import ast
from collections import Counter,defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return json.load(f)
def main():
    source=HERE/'configs/full_input_aligned_A10B45.json';oldsource=HERE/'configs/native_aligned_A10B45_60s.json'
    original_sha=sha(source);cfg=read(source);old=read(oldsource);data=ast.literal_eval((ROOT/'data').read_text());data_sha=sha(ROOT/'data')
    assert cfg['num_npu']==16 and cfg['ssu']==1 and cfg['seed']==7 and cfg['mode']=='variant_explicit'
    assert cfg['per_npu_sequences']==old['per_npu_sequences']
    cat=cfg['profiles_catalog'];seqs=cfg['per_npu_sequences'];ks=sorted({k for k,m in data});ms=sorted({m for k,m in data})
    physical=set();changed_keys=[];rates=[]
    for key,p in cat.items():
        t,m=p['total_tokens'],p['nql'];a=p['profile_construction']
        assert isinstance(t,int) and isinstance(m,int) and 32768<=t<=204800 and 64<=m<=4096
        assert p['family'] in ['A','B'] and t>m
        kl=max(k for k in ks if k*1024<=t);kh=min(k for k in ks if k*1024>=t)
        ml=max(x for x in ms if x<=m);mh=min(x for x in ms if x>=m)
        wk=0. if kl==kh else (t/1024-kl)/(kh-kl);wm=0. if ml==mh else (m-ml)/(mh-ml);weights=defaultdict(float)
        for k,x in [(kl,1-wk),(kh,wk)]:
            for n,y in [(ml,1-wm),(mh,wm)]:weights[k,n]+=x*y
        weights={k:w for k,w in weights.items() if w>0};actual={(r['seq_len_k'],r['nql']):r['weight'] for r in a['anchors']}
        assert actual.keys()==weights.keys() and len(actual)==len(a['anchors'])
        assert all(actual[k]>=0 and abs(actual[k]-weights[k])<1e-12 for k in weights)
        assert abs(math.fsum(actual.values())-1)<1e-12
        for r in a['anchors']:
            dk=(r['seq_len_k'],r['nql']);assert r['compute_us']==data[dk][1] and tuple(r['original_data_row'])==data[dk]
        c=math.fsum(data[k][1]*w for k,w in weights.items());v=(t-m)*1408/2**30
        assert abs(p['compute_us']-c)<1e-8 and abs(p['read_gib']-v)<1e-15
        assert p['ssd_prefix_tokens']==t-m and abs(p['B_gib_s']-v/(c/1e6))<1e-12
        assert a['data_sha256']==data_sha and a['extrapolated'] is False and a['compute_scale']==1.
        rates.append(v*2**30/1e9/(c/1e6));physical.add((t,m))
        if p!=old['profiles_catalog'][key]:changed_keys.append(key)
    counts=[];pure=[];changed=[]
    for n,seq in enumerate(seqs):
        count=Counter(cat[k]['family'] for k in seq);assert count=={'A':53,'B':240}
        counts.append(dict(count));pure.append(8*math.fsum(cat[k]['compute_us'] for k in seq)/1000)
        for pos,k in enumerate(seq):
            if k in changed_keys:changed.append(dict(npu=n,position=pos,key=k))
    assert len(seqs)==16 and len(changed)==31 and all(cat[k]['family']=='A' for k in changed_keys)
    assert set(cat)==set(old['profiles_catalog']) and math.fsum(len(s) for s in seqs)==4688
    assert 16*max(rates)<40 and min(pure)>60000
    fullruns=HERE/'runs/full_input_aligned_A10B45';run_data={};manifests=[]
    for policy in ['asu_baseline','once']:
        directory=fullruns/policy;m=read(directory/'manifest.json.gz');c=read(directory/'command.json');stored=read(directory/'config.json')
        assert c['config_sha256']==original_sha and c['manifest_sha256']==sha(directory/'manifest.json.gz')
        assert stored==m['metadata']['config'] and all(stored[k]==v for k,v in cfg.items())
        assert all(r['arrival_time_ms']==0 for r in m['requests']) and len(m['requests'])==4688
        for r in m['requests']:
            n,idx=divmod(r['request_id'],1000000);key=seqs[n][idx];p=cat[key];load=r['load']
            assert r['npu_id']==n and load['profile_key']==key and load['per_layer_us']==p['compute_us']
            assert load['profile_group']==p['family'] and load['nql']==p['nql'] and load['total_tokens']==p['total_tokens']
            blocks=m['placements'][r['placement_index']][0]
            assert len(blocks)==math.ceil(p['ssd_prefix_tokens']/128) and all(d==0 for d,v in blocks)
            assert abs(math.fsum(v for d,v in blocks)-p['read_gib'])<1e-14
        sources=c['source_sha256'];assert all(sha(ROOT/p)==value for p,value in sources.items())
        run_data[policy]=dict(command_sha256=sha(directory/'command.json'),manifest_sha256=sha(directory/'manifest.json.gz'),input_fingerprint=m['input_fingerprint'],source_count=len(sources),all_sources_match=True)
        manifests.append(sha(directory/'manifest.json.gz'))
    assert len(set(manifests))==1
    raw=read(HERE/'runs/native_aligned_A10B45_60s/asu_baseline/native_summary.json.gz')
    batches={b['member_request_ids'][0]:b for b in raw['microbatch_metrics']}
    for row in changed:
        b=batches[row['npu']*1000000+row['position']];row['old_io_start_ms']=b['layer_metrics'][0]['io_start_time_ms'];row['old_admission_ms']=b['admission_time_ms']
    assert sha(source)==original_sha
    output=dict(status='passed',static_only=True,native_performance_not_evaluated_by_this_static_audit=True,config_sha256=original_sha,data_sha256=data_sha,
        unchanged_sequences_and_request_count=True,request_count=4688,per_card_counts=counts,pure_compute_ms_by_card=pure,
        unique_physical_profiles=len(physical),physical_profiles_by_family={f:len({(p['total_tokens'],p['nql']) for p in cat.values() if p['family']==f}) for f in ['A','B']},
        catalog_keys=len(cat),changed_profile_keys=changed_keys,changed_requests_vs_60s=changed,
        original_trajectory_changed_requests_prefetched_before_60s=sum(r['old_io_start_ms']<60000 for r in changed),
        earliest_original_affected_prefetch_ms=min(r['old_io_start_ms'] for r in changed),
        nominal_any_mix_upper_GB_s=16*max(rates),extra_next_request_L0_demand_excluded=True,
        all_profile_anchor_and_C_V_checks=True,ASU_Once_manifest_byte_identical=True,run_initial_artifacts=run_data,
        input_file_unchanged=True,audit_source_sha256=sha(Path(__file__)))
    def physical_pool(c):
        return [Counter((c['profiles_catalog'][k]['total_tokens'],c['profiles_catalog'][k]['nql'],c['profiles_catalog'][k]['compute_us']) for k in seq) for seq in c['per_npu_sequences']]
    reference_pool=physical_pool(cfg);control_checks={}
    for input_seed in [7,17,27]:
        path=HERE/f'configs/full_input_pool_random_inputseed{input_seed}.json';c=read(path)
        same_pool=physical_pool(c)==reference_pool
        same_catalog=c['profiles_catalog']==cat
        changed_cards=sum(x!=y for x,y in zip(c['per_npu_sequences'],seqs))
        assert same_pool and same_catalog and changed_cards==16 and c['seed']==7 and c['num_npu']==16 and c['ssu']==1
        control_checks[path.name]=dict(config_sha256=sha(path),same_per_card_physical_pool=same_pool,same_catalog=same_catalog,
            changed_order_card_count=changed_cards,submit_seed=c['seed'],input_shuffle_seed=input_seed)
    path=HERE/'configs/full_input_aligned_submitseed17.json';c=read(path)
    assert all(c[k]==v for k,v in cfg.items() if k not in ['seed','name']) and c['seed']==17
    control_checks[path.name]=dict(config_sha256=sha(path),same_except_name_and_submit_seed=True,submit_seed=17)
    output['control_config_checks']=control_checks
    (HERE/'audit_full_input_config.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['changed_profile_keys','changed_requests_vs_60s','per_card_counts','pure_compute_ms_by_card']},indent=2))
if __name__=='__main__':main()
