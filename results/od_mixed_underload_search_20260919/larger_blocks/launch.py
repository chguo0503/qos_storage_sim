#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import time
HERE=Path(__file__).resolve().parent

def main():
    plan=json.loads((HERE/'plan.json').read_text());jobs=plan['candidates'];start=time.time()
    def worker(cpu,subset):
        for spec in subset:
            label=spec['label'];cmd=['taskset','-c',str(cpu),'python',str(HERE/'run_local_pilot.py'),'--base',str(HERE/plan['base_manifest']),'--spec',str(HERE/'specs'/f'{label}.json')]
            with (HERE/f'{label}.log').open('w')as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
            if p.returncode:print(json.dumps(dict(label=label,error=p.returncode,cpu=cpu)),flush=True);continue
            r=json.loads((HERE/'pilots'/label/'command.json').read_text());m=r['measurement']
            print(json.dumps(dict(label=label,cpu=cpu,U=m['U_percent'],warm_under=m['demand']['strict_underload_all_disks'],all_under=m['initial_4s_demand']['strict_underload_all_disks'],mixed=m['mixed_cards'],active=m['all_npus_active'],wall_s=r['wall_seconds'])),flush=True)
    with ThreadPoolExecutor(max_workers=3)as ex:list(ex.map(lambda x:worker(*x),[(cpu,jobs[i::3])for i,cpu in enumerate(plan['cores'])]))
    print(json.dumps(dict(done=True,wall_s=time.time()-start)),flush=True)

if __name__=='__main__':main()
