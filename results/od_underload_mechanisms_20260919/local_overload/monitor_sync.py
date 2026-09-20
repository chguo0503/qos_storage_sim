"""Read-only remote monitoring plus rsync of new run/log outputs."""
from pathlib import Path
import json,subprocess,time
HERE=Path(__file__).resolve().parent
SSH='ssh -S /tmp/qos_qdepth_live_20260919_mux -o BatchMode=yes'
while True:
 for name in ('runs','logs'):
  subprocess.run(['rsync','-a','-e',SSH,f'chguo@192.168.31.126:/tmp/qos_od_local_overload_20260919/{name}/',str(HERE/name)+'/'],check=True,stdout=subprocess.DEVNULL)
 rows=[]
 for p in sorted((HERE/'runs').glob('*/command.json')):
  d=json.loads(p.read_text());r=dict(label=p.parent.name,status=d['status'])
  for filename,key in [('progress.json','progress'),('window_previews.json','windows')]:
   if p.with_name(filename).exists():
    data=json.loads(p.with_name(filename).read_text())
    r[key]=data if key=='progress' else [{k:v for k,v in row.items() if k in ('start_ms','end_ms','U_percent','mixed_cards','all_npus_active','SSD_GiB_s')}|{'per_disk_overload_percent':row['demand']['per_disk_overload_percent']} for row in data.values()]
  rows.append(r)
 target=HERE/'monitor_current.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),runs=rows),indent=2)+'\n');tmp.replace(target)
 print(time.strftime('%H:%M:%S'),[(r['label'],r['status'],round(r.get('progress',{}).get('simulation_ms',0))) for r in rows if '60s' in r['label']],flush=True)
 if not any(r['status']=='running' for r in rows):break
 time.sleep(30)
