"""Preserve the archived full24 input and freeze the approved implementation."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'results/template_od_baseline_20260919/diverse'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
 with gzip.open(path,'rt') as f:return json.load(f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--freeze-runtime',action='store_true');a=ap.parse_args()
 for name in ('inputs','references'): (HERE/name).mkdir(exist_ok=True)
 rows=[]
 for seed in (7,19,43):
  source=OLD/'inputs'/f'full_seed{seed}.json.gz';target=HERE/'inputs'/source.name
  if target.exists():assert sha(target)==sha(source)
  else:shutil.copy2(source,target)
  manifest=read(target)
  reference=OLD/'runs'/f'full_od_baseline_seed{seed}'/'result.json.gz'
  frozen=HERE/'references'/f'full_od_unlimited_seed{seed}.json.gz'
  if frozen.exists():assert sha(frozen)==sha(reference)
  else:shutil.copy2(reference,frozen)
  command=json.loads(reference.with_name('command.json').read_text())
  assert command['status']=='complete' and read(reference)['input_fingerprint']==manifest['input_fingerprint']
  rows.append(dict(seed=seed,input_fingerprint=manifest['input_fingerprint'],manifest_sha256=sha(source),original_manifest=str(source.relative_to(ROOT)),reference_raw=str(reference.relative_to(ROOT)),reference_sha256=sha(reference),requests=len(manifest['requests']),blocks=sum(8*len(manifest['placements'][q['placement_index']][0]) for q in manifest['requests']),previous_wall_seconds=command['wall_seconds']))
 info=dict(cases=rows,configuration=dict(num_npu=32,num_ssu=3,disk_bw_gib_s=40,npu_bw_gib_s=50,n_layers=8,batch_size=1,submission_seed='same as manifest seed',arrival_ms=0,order='unchanged frozen Random',placement='frozen stripe (block_index+npu_id)%3; no generator called',collector_ms=5,cross_request_L0_prefetch=True,windows_ms=[[2000,4000],[2000,6000]],slo='processing/admission clock <=1.5 * request own compute, uncensored full drain',fixed_slot_borrowing=False,SSD_slot_released='SSD service completion, before NPU-link completion',bandwidth_idle_borrowing=True,slot_limit_per_NPU_per_SSU=256,slot_limit_total_per_SSU=8192))
 if a.freeze_runtime:
  runtime=HERE/'runtime'
  if runtime.exists():raise FileExistsError(runtime)
  runtime.mkdir()
  for package in ('simulator','inputs'):shutil.copytree(ROOT/package,runtime/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  shutil.copy2(ROOT/'data',runtime/'data')
  info['runtime_sha256']={str(p.relative_to(HERE)):sha(p) for p in sorted(runtime.rglob('*')) if p.is_file()}
 (HERE/'input_audit.json').write_text(json.dumps(info,indent=2)+'\n')
 print(json.dumps(dict(runtime_frozen=a.freeze_runtime,cases=rows),indent=2))
if __name__=='__main__':main()
