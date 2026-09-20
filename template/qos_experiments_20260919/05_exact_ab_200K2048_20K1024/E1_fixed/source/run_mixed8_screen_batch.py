from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,subprocess,sys
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/fifo_mixed_unique_20260914'
specs=json.loads((OUT/'screen_plan.json').read_text())
def run(spec):
 index=spec['case_index'];command=[sys.executable,'-u','run_fifo_mixed_unique.py']
 for key,value in spec.items():
  if key=='case_index':continue
  command.append('--'+key.replace('_','-'))
  command.extend(str(x) for x in value) if isinstance(value,list) else command.append(str(value))
 command.extend(['--stage',f'screen_{index:02d}'])
 (OUT/f'screen_{index:02d}_command.json').write_text(json.dumps(command))
 with (OUT/f'screen_{index:02d}.log').open('w') as log:
  completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 return dict(index=index,returncode=completed.returncode)
with ThreadPoolExecutor(max_workers=4) as pool:
 for task in as_completed([pool.submit(run,s) for s in specs]):print(json.dumps(task.result()),flush=True)
