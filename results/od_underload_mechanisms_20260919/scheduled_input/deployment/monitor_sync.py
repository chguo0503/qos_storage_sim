"""Read-only remote status and completed-artifact synchronization."""
from pathlib import Path
import json,shlex,subprocess,time
LOCAL=Path(__file__).resolve().parents[1]
REMOTE='/tmp/qos_scheduled_interp_20260920/results/od_underload_mechanisms_20260919/scheduled_input'
MUX='/tmp/qos_qdepth_live_20260919_mux';HOST='chguo@192.168.31.126';NAME='abb_interp_unique_50'
SSH=['ssh','-S',MUX,'-o','BatchMode=yes',HOST]
synced=set();previous=None

def fetch():
 code="import json;from pathlib import Path;p=Path("+repr(REMOTE)+");r={};f=p/'long_queue_status.json';r['queue']=json.loads(f.read_text()) if f.exists() else {};f=p/'planning'/"+repr(NAME)+"/'progress.json';r['progress']=json.loads(f.read_text()) if f.exists() else {};r['formal']={s:(json.loads((p/'formal'/('"+NAME+"_'+s)/'command.json').read_text()) if (p/'formal'/('"+NAME+"_'+s)/'command.json').exists() else {}) for s in ('od_baseline','once')};print(json.dumps(r))"
 return json.loads(subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)],text=True))

def copy(relative,destination):
 destination.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['scp','-q','-r','-o','ControlPath='+MUX,HOST+':'+REMOTE+'/'+relative,str(destination)],check=True)

for _ in range(1800):
 try:
  state=fetch();(LOCAL/'deployment/remote_status.json').write_text(json.dumps(state,indent=2)+'\n')
  stage=state['queue'].get('stage');cycle=state['progress'].get('cycle');mark=(stage,cycle)
  if mark!=previous:print(json.dumps({'stage':stage,'cycle':cycle,'progress':state['progress']}),flush=True);previous=mark
  if stage in ('formal','complete','formal_failed') and 'planning' not in synced:
   copy('planning/'+NAME,LOCAL/'planning'/NAME);copy('inputs/'+NAME+'.json.gz',LOCAL/'inputs'/(NAME+'.json.gz'));synced.add('planning');print('SYNCED frozen input and completed planning archive',flush=True)
  for policy,cmd in state['formal'].items():
   if cmd.get('status')=='complete' and policy not in synced:
    copy('formal/'+NAME+'_'+policy,LOCAL/'formal'/(NAME+'_'+policy));synced.add(policy);print('SYNCED formal '+policy,flush=True)
  if stage in ('complete','planning_failed','formal_failed'):
   (LOCAL/'deployment/sync_done.json').write_text(json.dumps({'stage':stage,'synced':sorted(synced)},indent=2)+'\n');break
 except Exception as error:print(type(error).__name__+': '+str(error),flush=True)
 time.sleep(20)
