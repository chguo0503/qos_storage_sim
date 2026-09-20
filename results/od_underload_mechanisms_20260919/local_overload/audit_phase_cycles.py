"""Measure completed A/B cycle phase dispersion without choosing low-U windows."""
from pathlib import Path
import argparse,gzip,json,statistics
HERE=Path(__file__).resolve().parent

def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);a=ap.parse_args();folder=HERE/'runs'/a.label
 command=json.loads((folder/'command.json').read_text());assert command['status']=='complete'
 raw=read(folder/'result.json.gz');mf=read(folder/'manifest.json.gz');req={r['request_id']:r for r in raw['summary']['request_metrics']};nA=mf['metadata']['cycle'].count('A');nB=mf['metadata']['cycle'].count('B')
 sequences=[]
 for n in range(32):
  rows=[r for r in mf['requests'] if r['npu_id']==n];segments=[]
  for r in rows:
   role=r['load']['role']
   if not segments or segments[-1]['role']!=role:segments.append(dict(role=role,rows=[]))
   segments[-1]['rows'].append(r)
  sequences.append([segments[i:i+2] for i in range(len(segments)-1) if segments[i]['role']=='A' and len(segments[i]['rows'])==nA and segments[i+1]['role']=='B' and len(segments[i+1]['rows'])==nB])
 first_finish=min(max(r['completion_time_ms'] for r in req.values() if r['npu_id']==n) for n in range(32))
 count=min(map(len,sequences));output=[]
 for k in range(count):
  pairs=[seq[k] for seq in sequences];astart=[];bstart=[];end=[];own=[];a_wait=[];b_wait=[]
  for left,right in pairs:
   ar=[req[r['request_id']] for r in left['rows']];br=[req[r['request_id']] for r in right['rows']]
   astart.append(ar[0]['admission_time_ms']);bstart.append(br[0]['admission_time_ms']);end.append(br[-1]['completion_time_ms']);own.append(sum(r['own_compute_ms'] for r in ar+br));a_wait.append(sum(r['io_stall_ms'] for r in ar));b_wait.append(sum(r['io_stall_ms'] for r in br))
  dur=[z-x for x,z in zip(astart,end)]
  output.append(dict(cycle=k,A_start_mean_ms=statistics.mean(astart),A_start_min_ms=min(astart),A_start_max_ms=max(astart),A_start_range_ms=max(astart)-min(astart),B_start_mean_ms=statistics.mean(bstart),B_start_min_ms=min(bstart),B_start_max_ms=max(bstart),B_start_range_ms=max(bstart)-min(bstart),cycle_end_mean_ms=statistics.mean(end),cycle_end_latest_ms=max(end),ends_before_first_npu_finishes=max(end)<first_finish,cycle_end_range_ms=max(end)-min(end),mean_A_stage_ms=statistics.mean(y-x for x,y in zip(astart,bstart)),mean_B_stage_ms=statistics.mean(z-y for y,z in zip(bstart,end)),duration_weighted_cycle_U_percent=100*sum(own)/sum(dur),mean_per_npu_cycle_U_percent=statistics.mean(100*c/t for c,t in zip(own,dur)),mean_per_npu_A_stall_ms=statistics.mean(a_wait),mean_per_npu_B_stall_ms=statistics.mean(b_wait)))
 result=dict(label=a.label,status='completed_source',input_fingerprint=mf['input_fingerprint'],A_requests_per_cycle=nA,B_requests_per_cycle=nB,complete_cycles=count,first_npu_final_completion_ms=first_finish,definition='Per card, consecutive full A-group and B-group; time is A-first admission to B-last completion. Initial B prefix and incomplete tail groups excluded. Cycle U sums exact request own compute / corresponding per-card cycle durations; not an instantaneous fleet U.',cycles=output)
 (HERE/(a.label+'_phase_cycles.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
