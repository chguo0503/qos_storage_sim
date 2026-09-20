"""Freeze untouched archived source and exact archived requests for OD replay."""
from pathlib import Path
import gzip, hashlib, json, shutil, sys

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
TEMPLATE = PROJECT / 'template/qos_experiments_20260919'
FORMULA = TEMPLATE / '03_continuous_underload/formula_ab/original_recovered'
SENS = TEMPLATE / '02_partial_overload/sensitivity20k_076/original_data_code/qos_storage_sim'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def main():
    provenance = {}
    for p in sorted((FORMULA / 'source').glob('*.py')):
        dest = OUT / 'frozen_source' / p.name
        shutil.copyfile(p, dest)
        provenance[str(p.relative_to(PROJECT))] = sha(p)
    # Both archived families use byte-identical native engine and exact-tail adapter.
    for name in ['sim.py', 'continuous_batch_sim.py', 'fixed_total_compat.py',
                 'shared_path_sim_adapter.py', 'continuous_prefill_client.py']:
        assert sha(FORMULA/'source'/name) == sha(SENS/name), name
    shutil.copyfile(PROJECT/'simulator/policies/od_baseline.py', OUT/'od_policy_snapshot.py')
    sys.path.insert(0, str(OUT/'frozen_source'))
    import sim
    import continuous_batch_sim as native
    from run_baseline_npu32_stress import save_manifest, load_manifest
    jobs = []
    groups = ['XY12_32', 'XY12_24', 'XY12_20', 'XY12_16', 'X16']
    for group in groups:
        for seed in [7, 19, 43]:
            name = f'{group}_random_s1_seed{seed}'
            origin = FORMULA/'results'/name/'baseline'
            rows = json.loads((origin/'requests.json').read_text())
            config = json.loads((origin/'config.json').read_text())
            reference = json.loads(gzip.decompress((origin/'native_summary.json.gz').read_bytes()))
            requests = []
            for q in rows:
                prefix = q['ssd_prefix_tokens']
                blocks = tuple((sim.block_ring_hash_disk_id(q['request_id'], b, 1),
                                min(128, prefix-128*b)*1408/2**30)
                               for b in range((prefix+127)//128))
                requests.append(native.ContinuousBatchRequest.from_normalized(
                    q['request_id'], q['npu_id'], q['arrival_ms'], q, (blocks,)))
            requests = tuple(requests)
            assert native.continuous_batch_input_fingerprint(requests) == reference['input_fingerprint']
            assert rows == json.loads((origin.parent/'once/requests.json').read_text())
            metadata = {'family':'formula', 'group':group, 'seed':seed, 'num_npu':8,
                        'num_ssu':1, 'n_layers':8, 'config':config,
                        'disk_bw_gib_s':40*1e9/2**30, 'npu_bw_gib_s':50*1e9/2**30,
                        'capacity_display':'40 GB/s (decimal)', 'window_ms':[2000.,4000.],
                        'source_request_sha256':sha(origin/'requests.json')}
            manifest = OUT/'inputs'/f'{name}.json.gz'
            save_manifest(manifest, requests, metadata)
            jobs.append({'case':name, 'family':'formula', 'group':group, 'seed':seed,
                         'manifest_sha256':sha(manifest), 'request_count':len(requests),
                         'input_fingerprint':reference['input_fingerprint'],
                         'original_directory':str(origin.relative_to(PROJECT))})
            for p in [origin/'requests.json', origin/'config.json', origin/'native_summary.json.gz',
                      origin.parent/'once/native_summary.json.gz']:
                provenance[str(p.relative_to(PROJECT))] = sha(p)
            if group == groups[0] and seed == 7:
                shutil.copyfile(origin/'native_summary.json.gz', OUT/'references'/f'{name}_asu_summary.json.gz')
    origin = SENS/'results/lower_fifo_followup_20260914/native20/sensitivity20k_076_seed7_fifo'
    requests, metadata = load_manifest(origin/'manifest.json.gz')
    comparison, _ = load_manifest(origin.parent/'sensitivity20k_076_seed7_once/manifest.json.gz')
    assert native.continuous_batch_input_fingerprint(requests) == native.continuous_batch_input_fingerprint(comparison)
    metadata.update(family='sensitivity', group='sensitivity20k_076',
                    disk_bw_gib_s=40., npu_bw_gib_s=50., capacity_display='40 GiB/s',
                    window_ms=[2000.,4000.])
    name = 'sensitivity20k_076_seed7'
    manifest = OUT/'inputs'/f'{name}.json.gz'
    save_manifest(manifest, requests, metadata)
    source_result = json.loads(gzip.decompress((origin/'result.json.gz').read_bytes()))
    with gzip.open(OUT/'references'/f'{name}_asu_summary.json.gz', 'wt') as f:
        json.dump(source_result['summary'], f)
    jobs.append({'case':name, 'family':'sensitivity', 'group':'sensitivity20k_076', 'seed':7,
                 'manifest_sha256':sha(manifest), 'request_count':len(requests),
                 'input_fingerprint':source_result['summary']['input_fingerprint'],
                 'original_directory':str(origin.relative_to(PROJECT))})
    for policy in ['fifo','once']:
        for file in ['manifest.json.gz','result.json.gz','metadata.json']:
            p = origin.parent/f'sensitivity20k_076_seed7_{policy}'/file
            provenance[str(p.relative_to(PROJECT))] = sha(p)
    write(OUT/'jobs.json', jobs)
    sys.path.insert(0, str(PROJECT))
    from simulator.policies.od_baseline import qos_config
    from dataclasses import asdict
    write(OUT/'od_qos_configs.json', {family:asdict(qos_config(8, capacity))
          for family,capacity in [('formula',40*1e9/2**30),('sensitivity',40.)]})
    write(OUT/'source_provenance.json', {'archived_file_sha256':provenance,
          'frozen_source_sha256':{p.name:sha(p) for p in (OUT/'frozen_source').glob('*.py')},
          'od_policy_snapshot_sha256':sha(OUT/'od_policy_snapshot.py'),
          'native_engine_modified':False, 'partial_tail_padding':False,
          'formula_capacity_unit':'decimal GB/s; converted to GiB/s for native engine',
          'sensitivity_capacity_unit':'GiB/s', 'exclusions':['mixed8','E1 standalone','sensitivity bandwidth_pair']})
    print(json.dumps({'jobs':len(jobs), 'requests':sum(j['request_count'] for j in jobs)}))

if __name__ == '__main__': main()
