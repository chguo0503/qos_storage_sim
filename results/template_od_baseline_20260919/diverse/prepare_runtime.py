"""Copy frozen template inputs; never regenerate placement or request order."""
from pathlib import Path
import gzip,json,hashlib,shutil,ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
T=ROOT/'template/qos_experiments_20260919'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def main():
 runtime=HERE/'runtime'
 if runtime.exists():raise FileExistsError(runtime)
 runtime.mkdir()
 for package in ('simulator','inputs'):
  shutil.copytree(ROOT/package,runtime/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copy2(ROOT/'data',runtime/'data')
 (HERE/'inputs').mkdir();(HERE/'references').mkdir()
 audit=[]
 oldstudy=T/'repository_source/results/diverse_data_ssu3_l3_20260916'
 near=T/'03_continuous_underload/near35/source/results/diverse_near35_20260916'
 for scenario in ('full','semi','near35'):
  for seed in (7,19,43):
   old=near if scenario=='near35' else oldstudy
   source=old/'inputs'/f'{scenario}_seed{seed}.json.gz'
   target=HERE/'inputs'/source.name;shutil.copy2(source,target)
   m=read(target);meta=m['metadata']
   assert (meta['num_npu'],meta['num_ssu'],meta['n_layers'])==(32,3,8)
   assert meta['seed']==seed and meta['order']=='random'
   assert all(q['arrival_time_ms']==0 for q in m['requests'])
   assert all(len(p)==1 for p in m['placements'])
   assert all(d==(i+q['npu_id'])%3 for q in m['requests'] for i,(d,v) in enumerate(m['placements'][q['placement_index']][0]))
   prior=ROOT/'results/od_baseline_diverse_ssu3_20260918/inputs'/f'{scenario}_seed{seed}_ring_hash.json.gz'
   reuse=prior.exists() and read(prior)['input_fingerprint']==m['input_fingerprint']
   ref=next((old/'runs').glob(f'baseline_seed{seed}/result.json.gz')) if scenario=='near35' else next((old/'runs').glob(f'{scenario}_baseline_seed{seed}_*/result.json.gz'))
   if seed==7:shutil.copy2(ref,HERE/'references'/f'{scenario}_asu_seed7.json.gz')
   audit.append(dict(scenario=scenario,seed=seed,source_manifest=str(source.relative_to(ROOT)),sha256=sha(source),copy_sha256=sha(target),input_fingerprint=m['input_fingerprint'],logical_input_fingerprint=meta['logical_input_fingerprint'],requests=len(m['requests']),blocks=sum(len(m['placements'][q['placement_index']][0])*8 for q in m['requests']),placement='frozen stripe (block_index+npu_id)%3',reused_existing_od=reuse,reference_asu=str(ref.relative_to(ROOT)),source_core_hashes=read(ref)['core_and_policy_sha256']))
 oldmetrics=oldstudy/'metrics.py';shutil.copy2(oldmetrics,HERE/'metrics.py')
 (HERE/'input_audit.json').write_text(json.dumps(dict(cases=audit,policy_only_change=True,original_metrics_sha256=sha(oldmetrics),runtime_sha256={str(p.relative_to(runtime)):sha(p) for p in sorted(runtime.rglob('*')) if p.is_file()},configuration=dict(num_npu=32,num_ssu=3,n_layers=8,batch_size=1,disk_bw_gib_s=40,npu_bw_gib_s=50,arrival_ms=0,cross_request_L0_prefetch=True,collector_ms=5,windows_ms=[[2000,4000],[2000,6000]],cohort='admitted in window; completion observed through drain; three-seed equally weighted metrics/CDF')),indent=2)+'\n')
 print(json.dumps([{k:r[k] for k in ('scenario','seed','requests','blocks','reused_existing_od')} for r in audit],indent=2))
if __name__=='__main__':main()
