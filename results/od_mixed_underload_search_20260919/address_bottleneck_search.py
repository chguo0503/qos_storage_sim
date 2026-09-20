#!/usr/bin/env python3
"""Select unique addresses with an exactly fixed SSU1 bottleneck block count."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT)]
from simulator.core import sim
from address_search import distribution

SPEC={'A':dict(blocks=80,bottleneck_ssu=1,bottleneck_blocks=28,wanted=6720,
               other_disk_min=24,other_disk_max=28),
      'B':dict(blocks=1210,bottleneck_ssu=1,bottleneck_blocks=419,wanted=656,
               other_disk_min=372,other_disk_max=419)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualifies(counts,role):
    spec=SPEC[role]
    return (sum(counts)==spec['blocks'] and counts[1]==spec['bottleneck_blocks']
            and all(spec['other_disk_min']<=counts[s]<=spec['other_disk_max'] for s in (0,2)))


def search_worker(index,cores,quotas,fresh_start,deadline):
    os.sched_setaffinity(0,{cores[index]})
    selected={'A':[],'B':[]};tested=0;hashed=0;scans={'A':0,'B':0};hits={'A':0,'B':0}
    begin=time.perf_counter();cpu=time.process_time();route=sim.block_ring_hash_disk_id
    while any(len(selected[r])<quotas[r] for r in SPEC) and time.perf_counter()<deadline:
        rid=fresh_start+index+tested*len(cores)
        need_b=len(selected['B'])<quotas['B'];limit=1210 if need_b else 80
        counts=[0,0,0];aa=None
        for block in range(limit):
            counts[route(rid,block,3)]+=1
            if block==79:aa=counts.copy()
        tested+=1;hashed+=limit;scans['A']+=1;scans['B']+=need_b
        ah=qualifies(aa,'A');bh=need_b and qualifies(counts,'B')
        hits['A']+=ah;hits['B']+=bh
        # Scarcer B wins when both predicates match, ensuring unique cross-role IDs.
        if bh:selected['B'].append(dict(request_id=rid,counts_by_ssu=counts))
        elif ah and len(selected['A'])<quotas['A']:
            selected['A'].append(dict(request_id=rid,counts_by_ssu=aa))
    return dict(worker=index,cpu_core=cores[index],quota=quotas,pools=selected,
                tested_ids=tested,hash_calls=hashed,predicate_scans=scans,predicate_hits=hits,
                last_tested_id=fresh_start+index+(tested-1)*len(cores) if tested else None,
                cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-begin,
                complete=all(len(selected[r])==quotas[r] for r in SPEC))


def verify_worker(index,cores,rows):
    os.sched_setaffinity(0,{cores[index]});route=sim.block_ring_hash_disk_id
    for role,row in rows:
        counts=[0,0,0]
        for block in range(SPEC[role]['blocks']):counts[route(row['request_id'],block,3)]+=1
        assert counts==row['counts_by_ssu'] and qualifies(counts,role)
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores',default='8,10,12,14')
    parser.add_argument('--max-seconds',type=float,default=120.)
    args=parser.parse_args();cores=tuple(map(int,args.cores.split(',')))
    assert set(cores)<=os.sched_getaffinity(0) and len(set(cores))==len(cores)
    begin=time.perf_counter()
    previous_path=HERE/'address_selection.json'
    protected={'sim.py':sha(ROOT/'simulator/core/sim.py'),'previous_address_selection.json':sha(previous_path),
               'address_search.py':sha(HERE/'address_search.py')}
    previous=json.loads(previous_path.read_text())
    assert previous['status']=='complete' and previous['source_sim_sha256']==protected['sim.py']
    reused={role:[row for row in previous['pools'][role] if qualifies(row['counts_by_ssu'],role)] for role in SPEC}
    remaining={role:SPEC[role]['wanted']-len(reused[role]) for role in SPEC}
    fresh_start=1+max(row['request_id'] for pool in previous['pools'].values() for row in pool)
    sim._block_hash_ring(3)
    with ProcessPoolExecutor(max_workers=len(cores),mp_context=multiprocessing.get_context('fork')) as executor:
        futures=[]
        for i in range(len(cores)):
            quotas={role:remaining[role]//len(cores)+(i<remaining[role]%len(cores)) for role in SPEC}
            futures.append(executor.submit(search_worker,i,cores,quotas,fresh_start,begin+args.max_seconds-5.))
        workers=[future.result() for future in futures]
        pools={role:sorted(reused[role]+[row for w in workers for row in w['pools'][role]],key=lambda row:row['request_id']) for role in SPEC}
        ids=[row['request_id'] for pool in pools.values() for row in pool]
        assert len(ids)==len(set(ids))
        rows=[(role,row) for role,pool in pools.items() for row in pool]
        futures=[executor.submit(verify_worker,i,cores,rows[i::len(cores)]) for i in range(len(cores))]
        verified=sum(future.result() for future in futures)
        assert verified==len(ids)
    assert sha(ROOT/'simulator/core/sim.py')==protected['sim.py']
    assert sha(previous_path)==protected['previous_address_selection.json']
    assert sha(HERE/'address_search.py')==protected['address_search.py']
    scans={r:sum(w['predicate_scans'][r] for w in workers) for r in SPEC}
    hits={r:sum(w['predicate_hits'][r] for w in workers) for r in SPEC}
    output=dict(status='complete' if all(w['complete'] for w in workers) else 'partial_budget_limit',
        classification='adversarial address selection with an exactly fixed bottleneck disk; not ordinary random',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sim_sha256=protected['sim.py'],
        source_and_previous_pool_sha256=protected,script_sha256=sha(__file__),specs=SPEC,
        original_hash_function='sim.block_ring_hash_disk_id(request_id,block_index,3)',
        selection_rule='A: total80,SSU1=28,SSU0/2<=28; B: total1210,SSU1=419,SSU0/2<=419',
        sampling_rule=dict(reused='Only previous same-role rows satisfying the new exact bottleneck predicate',
            reused_counts={r:len(reused[r]) for r in SPEC},fresh_start_request_id=fresh_start,
            fresh_sequence='fresh_start + worker_index + ordinal*worker_count; disjoint arithmetic progressions',
            cores=list(cores),worker_count=len(cores),B_priority_if_both_qualify=True,
            stopping_rule='Each worker stops after its fixed A/B quotas; B blocks are not hashed once its B quota is filled',
            predicate_scans=scans,predicate_hits=hits,
            observed_search_hit_rates={r:hits[r]/scans[r] if scans[r] else None for r in SPEC}),
        pools=pools,pool_counts={r:len(pool) for r,pool in pools.items()},
        selected_distribution={r:distribution([row['counts_by_ssu'] for row in pool]) for r,pool in pools.items()},
        all_ids_unique_within_and_between_this_pool_roles=True,
        reuse_between_old_and_new_pool_is_explicit_and_authorized=True,
        selected_IDs_freshly_verified_with_original_function=verified,
        placement_function_modified=False,source_sim_unchanged=True,old_pool_unchanged=True,
        formal_manifest_generated=False,wall_seconds=time.perf_counter()-begin,
        workers=[{k:v for k,v in w.items() if k!='pools'} for w in workers])
    target=HERE/'address_bottleneck_fixed.json'
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],pool_counts=output['pool_counts'],
                         reused_counts=output['sampling_rule']['reused_counts'],
                         hit_rates=output['sampling_rule']['observed_search_hit_rates'],
                         wall_seconds=output['wall_seconds'],output=str(target))),flush=True)


if __name__=='__main__':main()
