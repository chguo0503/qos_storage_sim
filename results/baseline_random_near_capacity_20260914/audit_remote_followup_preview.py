#!/usr/bin/env python3
"""Read independent U and every-card compute coverage for follow-up queues before import."""
from datetime import datetime, timezone
import json

from audit_remote_sync import HERE, read, sha, remote_json


def preview(password, scope, roles=('L','S')):
    assert scope in ('load110_confirmation','load105_confirmation','context384_confirmation','low_b_short','heterogeneity','context384_once_confirmation')
    assert roles in (('L','S'),('A','B'))
    setup = read(HERE / 'audit_remote_setup.json')
    path = HERE / ('audit_remote_' + scope + '_preview.json')
    record = read(path) if path.exists() else {'cases': {}}
    plan = read(HERE / ('audit_remote_' + scope + '_plan.json'))
    remaining = [job['label'] for job in plan['jobs'] if job['label'] not in record['cases']]
    code = '''from pathlib import Path
import json,gzip,math,hashlib
D=Path(__DIR__);H=D/'results/baseline_random_near_capacity_20260914'
Q=json.loads((H/('audit_remote_'+__SCOPE__+'_status.json')).read_text());remaining=__REMAINING__
P=json.loads((H/('audit_remote_'+__SCOPE__+'_plan.json')).read_text());jobs={j['label']:j for j in P['jobs']}
def read(p):
 with (gzip.open if p.suffix=='.gz' else open)(p,'rt') as f:return json.load(f)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(H/('audit_remote_'+__SCOPE__+'_plan.json'))==__PLAN_SHA__
expected_roles=__ROLES__
out={}
for label in remaining:
 if Q['jobs'][label]['status']!='complete':continue
 p=H/'runs'/label/jobs[label]['strategy'];c=read(p/'command.json')
 assert c['strategy']==jobs[label]['strategy']
 assert c['status']=='complete' and sha(p/'result.json.gz')==c['output_sha256']
 assert sha(p/'manifest.json.gz')==jobs[label]['manifest_sha256']==c['manifest_sha256']
 r=read(p/'result.json.gz');m=read(p/'manifest.json.gz')
 assert r['input_fingerprint']==jobs[label]['input_fingerprint'] and all(r['summary']['invariants'].values())
 roles={x['request_id']:(x['load']['coarse_role'] if __SCOPE__=='heterogeneity' else x['load']['role']) for x in m['requests']};windows=[]
 if __SCOPE__=='heterogeneity':assert all(x['load']['coarse_role']==x['load']['role'][0] for x in m['requests'])
 for a,z in [(2000.,4000.),(2000.,20000.)]:
  compute=[{role:[] for role in expected_roles} for _ in range(32)];active=[[] for _ in range(32)]
  for b in r['summary']['microbatch_metrics']:
   assert len(b['member_request_ids'])==1
   n=b['npu_id'];role=roles[b['member_request_ids'][0]]
   overlap=max(0.,min(z,b['completion_time_ms'])-max(a,b['admission_time_ms']))
   if overlap<=0:continue
   active[n].append(overlap)
   for layer in b['layer_metrics']:
    overlap=max(0.,min(z,layer['compute_end_ms'])-max(a,layer['compute_start_ms']))
    if overlap>0:compute[n][role].append(overlap)
  cards=[{'npu':n,'compute_ms':{role:math.fsum(v) for role,v in compute[n].items()},'active_ms':math.fsum(active[n])} for n in range(32)]
  U=100*math.fsum(math.fsum(v['compute_ms'].values()) for v in cards)/(32*(z-a))
  ref=next(w for w in c['windows'] if w['start_ms']==a and w['end_ms']==z)
  assert abs(U-100*ref['U'])<1e-7
  missing=[v['npu'] for v in cards if not all(x>1e-7 for x in v['compute_ms'].values())]
  inactive=[v['npu'] for v in cards if abs(v['active_ms']-(z-a))>1e-6]
  windows.append({'start_ms':a,'end_ms':z,'U_percent':U,'long_short_mixed_card_count':32-len(missing),'missing_mixed_cards':missing,'all_32_active':not inactive,'inactive_cards':inactive,'per_npu':cards})
 out[label]={'windows':windows,'input_fingerprint':jobs[label]['input_fingerprint'],'manifest_sha256':c['manifest_sha256'],'result_sha256':c['output_sha256'],'trace_requested':jobs[label]['trace']}
print('AUDIT_JSON:'+json.dumps({'new_cases':out,'complete':Q['complete'],'running':Q['running'],'failed':Q['failed']}))
'''.replace('__DIR__', repr(setup['remote_dir'])).replace('__REMAINING__', repr(remaining)).replace('__SCOPE__',repr(scope)).replace('__ROLES__',repr(roles)).replace('__PLAN_SHA__',repr(sha(HERE / ('audit_remote_' + scope + '_plan.json'))))
    response = remote_json(code, password)
    if response['new_cases']:
        record['cases'].update(response['new_cases'])
        record.update(updated_utc=datetime.now(timezone.utc).isoformat(), role_names=list(roles),
                      role_interpretation='Heterogeneity variants use the independently audited coarse_role L/S; other scopes use the original role directly.',
                      preview_source_sha256=sha(HERE / 'audit_remote_followup_preview.py'),
                      method='Independent clipping of completed raw layer compute and request-active intervals to [2,4) and [2,20) seconds; trace verification follows separately.')
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    brief = dict(complete=response['complete'], running=response['running'], failed=response['failed'],
                 new_cases={label: [dict((k, value) for k, value in window.items() if k != 'per_npu')
                                   for window in case['windows']]
                            for label, case in response['new_cases'].items()})
    print(json.dumps(brief, ensure_ascii=False))
    return brief


if __name__ == '__main__':
    import getpass
    raise SystemExit('Import preview(password, scope, roles) explicitly; no implicit case selection.')
