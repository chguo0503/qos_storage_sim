#!/usr/bin/env python3
"""Add one 60s role-rate-matched candidate per profile pair; retain every failure."""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import time
from pathlib import Path
from search import HERE, CPUS, work, config, initializer, source_hashes

def main():
    while not (HERE/'checks.json').exists():time.sleep(2)
    rows=json.loads((HERE/'short_results.json').read_text());long=json.loads((HERE/'long_results.json').read_text())
    seen={r['name'] for r in long}; jobs=[];design=[]
    for r in rows:
        if r['kind']!='fixed_roles':continue
        ideal=r['fixed_role_estimate']['q_for_10_A'];q=int(round(ideal));cfg=config(r['A_miss'],r['B_miss'],q)
        design.append(dict(name=cfg['name'],finite_role_q_estimate=ideal,q=q,already_evaluated=cfg['name'] in seen))
        if cfg['name'] in seen:continue
        jobs.append((r['A_miss'],r['B_miss'],q,'periodic',60000.))
        (HERE/(cfg['name']+'_config.json')).write_text(json.dumps(cfg,indent=2)+'\n')
    h=source_hashes();own=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'rate_matched_plan.json').write_text(json.dumps(dict(reason='12s precedes q-dependent role transitions, so compare nearest integer duration ratios directly at60s.',design=design,source_hashes=h,extension_source_sha256=own),indent=2)+'\n')
    ctx=multiprocessing.get_context('fork');queue=ctx.Queue()
    for cpu in CPUS:queue.put(cpu)
    outputs=[]
    with cf.ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=initializer,initargs=(queue,)) as pool:
        futures=[pool.submit(work,job) for job in jobs]
        for f in cf.as_completed(futures):
            row=f.result();outputs.append(row)
            (HERE/'rate_matched_60s_results.json').write_text(json.dumps(sorted(outputs,key=lambda x:x['name']),indent=2)+'\n')
            print('rate_match',len(outputs),len(jobs),row['name'],{k:v['U_percent'] for k,v in row['windows'].items()},flush=True)
    assert h==source_hashes()
    assert own==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    all_long=long+outputs;all_long.sort(key=lambda r:r['windows']['20_60']['U_percent'])
    report=dict(status='complete',approximate_only=True,native_run=False,short_cases=len(rows),long_cases=len(all_long),
        total_model_evaluations=len(rows)+len(all_long),source_unchanged=True,ordered_by_20_60_U=[dict(name=r['name'],windows={k:v['U_percent'] for k,v in r['windows'].items()},warm_mixed=r['windows']['2_4']['all_cards_compute_both'],any_mix_upper_GB_s=r['strict_any_mix_D_upper_GB_s']) for r in all_long])
    (HERE/'final_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
