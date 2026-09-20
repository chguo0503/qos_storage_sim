"""Native workload constructor for explicitly cross-card-correlated role decks.

Native simulator and independent baseline builder remain untouched. Native
profile selection is reused; request IDs and ring placement are reconstructed
at each reordered generation. See correlated_fifo_probe.correlated_decks.
"""
from __future__ import annotations
import collections,copy,math
import sim
import continuous_batch_sim as native
from continuous_batch_sim import ContinuousBatchRequest, continuous_batch_input_fingerprint
from run_random_multitype import build as independent_build
from run_shared_path_experiments import input_demand, logical_input_fingerprint
from correlated_fifo_probe import correlated_decks

def build_correlated(spec,seed=7,horizon_ms=4500.):
    base=copy.deepcopy(spec);mode=base.pop('joint_distribution')
    original,meta,_=independent_build(base,seed,horizon_ms)
    lookup={(q.npu_id,q.load['total_tokens'],q.load['nql']):q for q in original}
    lanes=correlated_decks(spec,seed,horizon_ms);S=spec['ssu'];groups=spec['groups'];requests=[];vectors={};maxima=[[0.]*S for _ in range(8)]
    checks=[]
    for n,lane in enumerate(lanes):
        order=[];ideal_ms=0.;seen=set()
        for generation,(gid,c,v,k,y) in enumerate(lane):
            source=lookup[n,int(round(k*1024)),y]
            rid=n*1000000+generation;load=dict(source.load)
            load.update(request_id=rid,generation=generation,original_request_id=rid)
            assert load['profile_group']==groups[gid]['id']
            assert math.isclose(load['per_layer_us']/1000,c,abs_tol=1e-10)
            assert math.isclose(load['per_layer_kv_gb'],v,abs_tol=1e-12)
            prefix=load['ssd_prefix_tokens']
            layer=tuple((sim.block_ring_hash_disk_id(rid,j,S),min(128,prefix-128*j)*1408/2**30) for j in range(math.ceil(prefix/128)))
            q=ContinuousBatchRequest.from_normalized(rid,n,0.,load,(layer,))
            assert all(native._manifest_layer(q,l) is layer for l in range(8))
            requests.append(q)
            vector=[math.fsum(value for disk,value in layer if disk==s)/(load['per_layer_us']/1e6) for s in range(S)]
            vectors[rid]=vector;maxima[n]=[max(a,b) for a,b in zip(maxima[n],vector)]
            order.append(load['profile_group']);ideal_ms+=8*c;seen.add((load['total_tokens'],y))
        assert len(seen)==len(lane) and ideal_ms>horizon_ms
        checks.append(dict(npu_id=n,request_count=len(lane),unique_length_nql_count=len(seen),profile_group_counts=dict(collections.Counter(order)),profile_group_order=order,request_counts=dict(collections.Counter(groups[p[0]]['role'] for p in lane)),ideal_compute_ms=ideal_ms,profile_draw_seed=seed+100003*n))
    requests=tuple(requests);static=[sum(row[s] for row in maxima) for s in range(S)]
    meta.update(experiment='correlated_random_multitype_followup_20260914',specification=spec,order='cross_card_correlated_random',joint_distribution=mode,lane_checks=checks,
        input_fingerprint=continuous_batch_input_fingerprint(requests),logical_input_fingerprint=logical_input_fingerprint(requests),input_demand=input_demand(requests,8,S),
        per_ssu_static_upper_bound_gib_s=static,static_underload_all_request_combinations=max(static)<=40+1e-8,
        workload_scope='NEW joint random input distribution: unique original native profile draws per card, reordered by one shared count-preserving random group sequence with optional swaps or Markov persistence. All requests arrive at zero. No inserted idle. Original 8 layers and exact ring placement by reordered request ID. This is not the independent-permutation input family.')
    meta['case'].update(order='cross_card_correlated_random',joint_distribution=mode)
    return requests,meta,vectors
