"""Read-only postprocessing of the two completed AB replays; never runs simulation."""
from pathlib import Path
import csv,gzip,hashlib,json,statistics,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import metrics
from inputs.manifest import load_manifest,write_json

CASE='ab_a10k1024_b200k_cb75p25_c16'

def main():
    requests,metadata=load_manifest(HERE/'inputs'/f'{CASE}.json.gz')
    byid={r.request_id:r for r in requests}
    output={'case':CASE,'metadata':metadata,'policies':{},'checks':{}}
    rows=[];hashes=[]
    for policy in ('od_baseline','once'):
        directory=HERE/'runs'/f'{CASE}_{policy}'
        raw=json.load(gzip.open(directory/'result.json.gz','rt'))['summary']
        command=json.loads((directory/'command.json').read_text())
        assert command['status']=='complete' and command['source_unchanged']
        assert all(raw['invariants'].values())
        assert raw['request_count']==1024 and raw['completed_blocks']==6782976
        hashes.append(command['source_sha256'])
        hashes.append(command['manifest_sha256'])
        policy_data={'makespan_ms':raw['makespan_ms'],'first_card_drains_ms':min(max(r['completion_time_ms'] for r in raw['request_metrics'] if r['npu_id']==n) for n in range(32)),
            'all_native_invariants_passed':True,'windows':[],
            'manifest_sha256':command['manifest_sha256'],
            'result_sha256':hashlib.sha256((directory/'result.json.gz').read_bytes()).hexdigest()}
        for a,b in [(2000.,4000.),(4000.,6000.),(6000.,8000.),(4000.,8000.),(8000.,10000.),(0.,raw['makespan_ms'])]:
            m=metrics.summarize(raw,requests,a,b,full=a==0.)
            m['measurement_definition']['own_compute']='8 × frozen synthetic/extrapolated per-layer C'
            policy_data['windows'].append(m)
            s=m['role_and_stall']['io_stall']['by_kind_card_ms']
            row=dict(policy=policy,window='full' if a==0. else f'[{a/1000:g},{b/1000:g})',
                U_percent=m['U_percent'],admission_SLO_1p5_percent=m['slo']['percent'],SLO_count=m['slo']['count'],
                all_npus_active=m['all_npus_active'],mixed_cards=m['role_and_stall']['npus_with_A_and_B_compute'],
                strict_underload=m['demand']['strict_underload_all_disks'],any_disk_overload_percent=m['demand']['any_disk_overload_percent'],
                maximum_disk_demand_GiB_s=max(m['demand']['per_disk_max_GiB_s']))
            row.update({f'stall_{k}_card_ms':v for k,v in s.items()});rows.append(row)
        phases=[]
        for cycle in range(16):
            times=[b['layer_metrics'][0]['compute_start_ms'] for b in raw['microbatch_metrics'] if byid[b['member_request_ids'][0]].load['role']=='B' and byid[b['member_request_ids'][0]].load['generation']//2==cycle]
            assert len(times)==32
            phases.append(dict(cycle=cycle,mean_B_L0_compute_start_ms=statistics.mean(times),range_ms=max(times)-min(times),std_ms=statistics.pstdev(times)))
        policy_data['B_L0_phase_spread_by_cycle']=phases
        if policy=='od_baseline':
            d=raw['ssd_queue_depth']
            assert d['per_npu_per_ssu_slots']==256 and d['total_reserved_slots_per_ssu']==8192
            assert all(x==256 for row in d['peak_outstanding_blocks_by_npu_ssu'] for x in row)
            assert d['activated_blocks']==d['issued_blocks']==d['ssd_completed_blocks']==d['hbm_completed_blocks']==6782976
            assert all(d[k]==0 for k in ('host_deferred_blocks_at_stop','ssd_outstanding_blocks_at_stop','link_outstanding_blocks_at_stop','input_not_yet_activated_blocks'))
            policy_data['depth_check']={'all_96_pairs_reached_256':True,'all_end_counts_zero':True,'all_6782976_commands_conserved':True,
                'blocked_state_episodes':sum(map(sum,d['blocked_state_episodes_by_npu_ssu'])),
                'host_blocked_state_ms':sum(map(sum,d['host_blocked_state_ms_by_npu_ssu'])),
                'caveat':'state blocked durations may overlap; not NPU stall'}
        else:
            assert 'ssd_queue_depth' not in raw
            policy_data['depth_check']={'finite_depth_enabled':False,'caveat':'current Once comparison uses its original unbounded submission; not a depth-matched routing-only ablation'}
        write_json(directory/'posthoc_comparison.json',policy_data)
        output['policies'][policy]=policy_data
    assert hashes[0]==hashes[2] and hashes[1]==hashes[3]
    output['checks']={'source_hashes_identical_across_policies':True,'manifest_hashes_identical':True,'all_native_invariants':True,
        'no_new_simulation':True,'only_synthetic_profile_timing':True,'long_term_low_U_candidate':False,
        'strict_underload_candidate':False,'scope':'2 completed replays plus 2 bounded pilots; neither profile is an unmodified measured data row'}
    write_json(HERE/'comparison.json',output)
    with (HERE/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'checks':output['checks'],'rows':rows,'drain_ms':{p:v['first_card_drains_ms'] for p,v in output['policies'].items()}},ensure_ascii=False))
if __name__=='__main__':main()
