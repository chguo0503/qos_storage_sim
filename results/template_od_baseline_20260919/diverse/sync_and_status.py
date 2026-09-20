from pathlib import Path
import json,subprocess,time
HERE=Path(__file__).resolve().parent
remote='chguo@192.168.31.126:/tmp/qos_template_od_20260919_diverse/diverse/'
for folder in ('runs','execution_logs'):
 subprocess.run(['rsync','-a','-e','ssh -S /tmp/qos_template_od_20260919_mux -o BatchMode=yes',remote+folder+'/',str(HERE/folder)+'/'],check=True)
rows=[]
for path in sorted((HERE/'runs').glob('*/command.json')):
 c=json.loads(path.read_text())
 if c['smoke']:continue
 row={k:c.get(k) for k in ('scenario','policy','seed','status','host','pid')};row['label']=path.parent.name
 for name in ('progress','warm_preview','parity'):
  p=path.with_name(name+'.json')
  if p.exists():
   d=json.loads(p.read_text())
   if name=='warm_preview':row.update(U_percent=d['U_percent'],SLO15_percent=d['slo']['percent'])
   elif name=='parity':row['parity_passed']=d['passed']
   else:row.update(d)
 if c['status']=='complete':row.update(completed_blocks=c['observed_blocks'],expected_blocks=c['expected_blocks'],completed_requests=c['completed_requests'],wall_seconds=c['wall_seconds'])
 if c['status']=='failed':row['error']=c.get('error')
 rows.append(row)
value=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),expected_cases=12,completed=sum(r['status']=='complete' for r in rows),cases=rows)
(HERE/'progress_summary.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
