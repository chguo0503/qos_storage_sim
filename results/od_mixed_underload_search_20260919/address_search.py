#!/usr/bin/env python3
"""Bounded adversarial request-ID selection using the unmodified ring hash."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from simulator.core import sim

BASE=100_000_000
SPEC={'A':dict(blocks=80,target=[25,28,27],tolerance=1,wanted=6720),
      'B':dict(blocks=1210,target=[382,419,409],tolerance=2,wanted=656)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matches(counts,role):
    spec=SPEC[role]
    return all(abs(value-target)<=spec['tolerance'] for value,target in zip(counts,spec['target']))


def counts_for(rid,blocks):
    counts=[0,0,0]
    route=sim.block_ring_hash_disk_id
    for block in range(blocks):counts[route(rid,block,3)]+=1
    return counts


def pilot_worker(index,cores,pilot_count):
    os.sched_setaffinity(0,{cores[index]})
    begin=time.perf_counter();cpu=time.process_time()
    a_counts,b_counts=[],[]
    hits={'A':0,'B':0,'both':0}
    route=sim.block_ring_hash_disk_id
    for offset in range(index,pilot_count,len(cores)):
        counts=[0,0,0];aa=None
        for block in range(SPEC['B']['blocks']):
            counts[route(BASE+offset,block,3)]+=1
            if block+1==SPEC['A']['blocks']:aa=counts.copy()
        a_counts.append(aa);b_counts.append(counts)
        ah,bh=matches(aa,'A'),matches(counts,'B')
        hits['A']+=ah;hits['B']+=bh;hits['both']+=ah and bh
    return dict(hits=hits,A_counts=a_counts,B_counts=b_counts,
                wall_seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu)


def collect_worker(index,cores,pilot_count,deadline):
    os.sched_setaffinity(0,{cores[index]})
    begin=time.perf_counter();cpu=time.process_time();route=sim.block_ring_hash_disk_id
    quota={role:spec['wanted']//len(cores)+(index<spec['wanted']%len(cores)) for role,spec in SPEC.items()}
    selected={'A':[],'B':[]};tested=0;blocks_hashed=0;skipped_joint_A=0
    while any(len(selected[role])<quota[role] for role in SPEC) and time.perf_counter()<deadline:
        rid=BASE+pilot_count+index+tested*len(cores)
        need_b=len(selected['B'])<quota['B'];limit=SPEC['B']['blocks'] if need_b else SPEC['A']['blocks']
        counts=[0,0,0];aa=None
        for block in range(limit):
            counts[route(rid,block,3)]+=1
            if block+1==SPEC['A']['blocks']:aa=counts.copy()
        blocks_hashed+=limit;tested+=1
        bh=need_b and matches(counts,'B')
        ah=len(selected['A'])<quota['A'] and matches(aa,'A')
        if bh:
            selected['B'].append(dict(request_id=rid,counts_by_ssu=counts))
            skipped_joint_A+=ah
        elif ah:
            selected['A'].append(dict(request_id=rid,counts_by_ssu=aa))
    return dict(worker=index,cpu_core=cores[index],quota=quota,pools=selected,
                tested_ids=tested,blocks_hashed=blocks_hashed,skipped_joint_A=skipped_joint_A,
                last_tested_id=BASE+pilot_count+index+(tested-1)*len(cores) if tested else None,
                wall_seconds=time.perf_counter()-begin,cpu_seconds=time.process_time()-cpu,
                complete=all(len(selected[role])==quota[role] for role in SPEC))


def verify_worker(index,cores,rows):
    os.sched_setaffinity(0,{cores[index]})
    checked=0
    for role,row in rows:
        actual=counts_for(row['request_id'],SPEC[role]['blocks'])
        assert actual==row['counts_by_ssu'] and matches(actual,role)
        checked+=1
    return checked


def distribution(rows):
    if not rows:return dict(count=0)
    mean=[math.fsum(row[s] for row in rows)/len(rows) for s in range(3)]
    return dict(count=len(rows),mean_counts_by_ssu=mean,
                minimum_counts_by_ssu=[min(row[s] for row in rows) for s in range(3)],
                maximum_counts_by_ssu=[max(row[s] for row in rows) for s in range(3)],
                population_std_counts_by_ssu=[math.sqrt(math.fsum((row[s]-mean[s])**2 for row in rows)/len(rows)) for s in range(3)],
                count_vector_histogram=dict(sorted(Counter('/'.join(map(str,row)) for row in rows).items())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores',default='8,10,12,14')
    parser.add_argument('--pilot-count',type=int,default=2000)
    parser.add_argument('--max-seconds',type=float,default=120.)
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args()
    cores=tuple(map(int,args.cores.split(',')))
    assert len(cores)==len(set(cores)) and set(cores)<=os.sched_getaffinity(0)
    started=time.perf_counter();initial_sha=sha(ROOT/'simulator/core/sim.py')
    # Warming this pure cache preserves exactly the original function and ring.
    sim._block_hash_ring(3)
    context=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=len(cores),mp_context=context) as executor:
        futures=[executor.submit(pilot_worker,i,cores,args.pilot_count) for i in range(len(cores))]
        pilot_results=[f.result() for f in futures]
        pilot_wall=time.perf_counter()-started
        hits={role:sum(r['hits'][role] for r in pilot_results) for role in ('A','B','both')}
        rates={role:value/args.pilot_count for role,value in hits.items()}
        cpu_seconds=sum(r['cpu_seconds'] for r in pilot_results)
        cpu_per_block=cpu_seconds/(args.pilot_count*SPEC['B']['blocks'])
        effective_a=max(rates['A']-rates['both'],1/args.pilot_count)
        needed_a=SPEC['A']['wanted']/effective_a
        needed_b=SPEC['B']['wanted']/rates['B'] if rates['B'] else math.inf
        projected_blocks=needed_b*SPEC['B']['blocks']+max(0,needed_a-needed_b)*SPEC['A']['blocks']
        verify_blocks=sum(s['wanted']*s['blocks'] for s in SPEC.values())
        estimate=(projected_blocks+verify_blocks)*cpu_per_block/len(cores)*1.25+pilot_wall
        result=dict(status='pilot_complete',created_utc=datetime.now(timezone.utc).isoformat(),
            classification='explicit adversarial address selection; not ordinary random inputs',
            original_hash_function='sim.block_ring_hash_disk_id(request_id,block_index,3)',
            source_sim_sha256=initial_sha,script_sha256=sha(__file__),cores=list(cores),
            specs=SPEC,base_request_id=BASE,all_ids_distinct_between_and_within_pools=True,
            pilot=dict(first_id=BASE,last_id=BASE+args.pilot_count-1,count=args.pilot_count,hits=hits,
                       hit_rates=rates,wall_seconds=pilot_wall,cpu_seconds=cpu_seconds,
                       A_distribution=distribution([v for r in pilot_results for v in r['A_counts']]),
                       B_distribution=distribution([v for r in pilot_results for v in r['B_counts']])),
            cost_estimate=dict(expected_total_IDs_for_A=needed_a,expected_total_IDs_for_B=needed_b,
                expected_hash_calls=projected_blocks,expected_total_wall_seconds_with_25pct_margin=estimate,
                includes_fresh_selected_ID_verification=True),
            placement_function_modified=False,formal_manifest_generated=False)
        output=HERE/'address_selection.json'
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(stage='pilot',hits=hits,hit_rates=rates,wall_seconds=pilot_wall,
                              estimated_total_wall_seconds=estimate)),flush=True)
        remaining=args.max_seconds-(time.perf_counter()-started)
        if args.pilot_only or not math.isfinite(estimate) or estimate>args.max_seconds or remaining<10:
            result['status']='pilot_only_cost_exceeds_budget' if not args.pilot_only else 'pilot_only'
        else:
            reserve=max(5.,verify_blocks*cpu_per_block/len(cores)*1.4+1.)
            deadline=started+args.max_seconds-reserve
            jobs=[executor.submit(collect_worker,i,cores,args.pilot_count,deadline) for i in range(len(cores))]
            collected=[job.result() for job in jobs]
            pools={role:sorted([row for worker in collected for row in worker['pools'][role]],key=lambda r:r['request_id']) for role in SPEC}
            ids=[row['request_id'] for pool in pools.values() for row in pool]
            assert len(ids)==len(set(ids)) and all(rid>=BASE for rid in ids)
            all_rows=[(role,row) for role,pool in pools.items() for row in pool]
            jobs=[executor.submit(verify_worker,i,cores,all_rows[i::len(cores)]) for i in range(len(cores))]
            verified=sum(job.result() for job in jobs)
            assert verified==len(ids)
            result.update(status='complete' if all(w['complete'] for w in collected) else 'partial_budget_limit',
                          pools=pools,pool_counts={role:len(pool) for role,pool in pools.items()},
                          selected_distribution={role:distribution([r['counts_by_ssu'] for r in pool]) for role,pool in pools.items()},
                          selected_IDs_freshly_verified_with_original_function=verified,
                          workers=[{k:v for k,v in w.items() if k!='pools'} for w in collected])
        assert sha(ROOT/'simulator/core/sim.py')==initial_sha
        result.update(source_sim_unchanged=True,wall_seconds=time.perf_counter()-started)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=result['status'],pool_counts=result.get('pool_counts',{}),
                              wall_seconds=result['wall_seconds'],output=str(output))),flush=True)


if __name__=='__main__':main()
