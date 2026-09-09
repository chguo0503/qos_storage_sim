#!/usr/bin/env python3
"""Replay frozen input and observe completed blocks without altering events."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import time
from unittest.mock import patch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
import continuous_batch_sim as native
from run_baseline_npu32_stress import load_manifest,run_case

LABEL='concurrency_l768_seed7__exact_cohort4_p1'
LEFT,RIGHT=3060.0,3105.0
OUT=BASE/'diagnostics'
OUT.mkdir(exist_ok=True)
MANIFEST=BASE/'inputs'/f'{LABEL}.json.gz'
REFERENCE=next((BASE/'runs'/LABEL/'baseline').glob('*.json.gz'))


def read(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    requests,metadata=load_manifest(MANIFEST)
    original=native._register_complete
    rows=[]
    calls=0
    def observe(context,flow):
        nonlocal calls
        # No event, queue, timing, RNG, request, or flow state is mutated here.
        # The experiment adapter calls this wrapper before updating its ledger.
        if flow.enqueue_time < RIGHT and flow.link_end_time >= LEFT:
            rows.append([flow.request_id,flow.npu_id,flow.layer,flow.block_idx,
                         flow.disk_id,flow.queue_id,flow.total_gb,flow.block_count,
                         flow.enqueue_time,flow.ssd_activation_time,
                         flow.link_enqueue_time,flow.link_start_time,flow.link_end_time])
        calls+=1
        return original(context,flow)
    started=time.perf_counter()
    print(json.dumps({'started':LABEL,'capture_ms':[LEFT,RIGHT],
                      'observer':'completion read-only; all six SSDs; no added simulation events'}),flush=True)
    with patch.object(native,'_register_complete',observe):
        result=run_case(requests,metadata,strategy='baseline',assignment='fixed',windows=((2000,4000),))
    reference=read(REFERENCE)
    checks={k:result['summary'][k]==reference['summary'][k]
            for k in ['microbatch_metrics','request_metrics','makespan_ms',
                      'fleet_npu_compute_utilization','disk_stats','invariants']}
    checks['windows_exactly_equal']=result['windows']==reference['windows']
    checks['core_source_hashes_equal']=result['core_and_policy_sha256']==reference['core_and_policy_sha256']
    checks['input_fingerprint_equal']=result['input_fingerprint']==reference['input_fingerprint']
    checks['all_blocks_path_zero']=all(row[5]==0 for row in rows)
    checks['observer_called_for_all_blocks']=calls==sum(r['io_count'] for r in result['summary']['request_metrics'])
    checks['ssd_service_duration']=all(abs((r[10]-r[9])-1000*r[6]/40)<1e-7 for r in rows)
    checks['link_service_duration']=all(abs((r[12]-r[11])-1000*r[6]/50)<1e-7 for r in rows)
    checks['chronological_block_stages']=all(r[8]<=r[9]+1e-8 and r[9]<r[10]
                                            and r[10]<=r[11]+1e-8 and r[11]<r[12] for r in rows)
    audit={'checks':checks,'passed':all(checks.values()),'captured_blocks':len(rows),
           'total_completed_blocks':calls,'wall_seconds':time.perf_counter()-started,
           'reference_result':str(REFERENCE.relative_to(BASE)),
           'reference_sha256':sha(REFERENCE),'manifest_sha256':sha(MANIFEST),
           'observer_source_sha256':sha(Path(__file__)),'capture_ms':[LEFT,RIGHT],
           'filter':'enqueue < right and HBM/link completion >= left; includes boundary-crossing blocks',
           'meaning':'SSD service is activation -> link enqueue; link service is link start -> link end; all physical durations from actual events.'}
    (OUT/'ordered_block_trace_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert audit['passed'],checks
    with gzip.open(OUT/'ordered_block_trace.json.gz','wt',compresslevel=6) as f:
        json.dump({'audit':audit,'columns':['request_id','npu_id','layer','block_idx','ssu_id','path_id',
                     'size_gib','block_count','enqueue_ms','ssd_start_ms','ssd_end_ms',
                     'link_start_ms','link_end_ms'],'rows':rows},f,separators=(',',':'))
    with gzip.open(OUT/'ordered_trace_replay.json.gz','wt',compresslevel=6) as f:
        json.dump(result,f,separators=(',',':'))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
