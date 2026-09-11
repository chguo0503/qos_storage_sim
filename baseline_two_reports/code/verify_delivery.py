#!/usr/bin/env python3
"""Check exported integrity and consistency against independent source audits."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import re

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt',encoding='utf-8') as f:return json.load(f)

def check_close(a,b,label):
    assert abs(a-b)<1e-7,(label,a,b)

def main():
    result={'passed':False,'gzip_files':[],'cases':{},'markdown_local_links_checked':0}
    for p in sorted(ROOT.rglob('*.gz')):
        with gzip.open(p,'rb') as f:
            while f.read(1024*1024):pass
        result['gzip_files'].append(str(p.relative_to(ROOT)))
    high_audit=read(ROOT/'audits/high_source_audit.json')
    for source in high_audit['source_files']:
        p=ROOT/'sources/high'/source['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256'],str(p)
    for key in ['high','low4','low5','low6']:
        root=ROOT/'data'/key
        s=read(root/'statistics.json');c=read(root/'normalized.json.gz')
        audit=read(ROOT/f'audits/{key}_source_audit.json')
        assert all(audit['checks'].values()),key
        check_close(s['fleet_utilization'],audit['fleet_utilization'],key+' independent U')
        check_close(s['compute_ms']+s['stall_ms'],32000,key+' fleet time')
        check_close(s['compute_ms']/32000,s['fleet_utilization'],key+' compute U')
        assert len(c['inputs'])==s['input_count']
        assert len(c['requests'])==s['observed_admitted_count']
        assert len({r['request_id'] for r in c['inputs']})==len(c['inputs'])
        assert len(s['per_npu'])==32
        for row in s['per_npu']:
            check_close(row['active_ms'],1000,key+' active')
            check_close(row['compute_ms']+row['stall_ms'],1000,key+' time')
        for family in ['category_summary','profile_summary']:
            assert sum(r['input_count'] for r in s[family])==len(c['inputs'])
            check_close(math.fsum(r['occupancy_share'] for r in s[family]),1,key+' shares')
            check_close(math.fsum(r['fleet_compute_contribution'] for r in s[family]),s['fleet_utilization'],key+' contributions')
        with gzip.open(root/'all_inputs_and_times.csv.gz','rt',encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==len(c['inputs'])
        for exported,original in zip(rows,c['inputs']):
            for field in ['request_id','npu_id','input_order']:
                assert int(exported[field])==original[field],(key,field)
            for field in ['arrival_ms','admission_ms','first_layer_io_start_ms','completion_ms']:
                value=original.get(field)
                if value is None:assert exported.get(field,'')=='',(key,field)
                else:check_close(float(exported[field]),value,key+' '+field)
        for n in range(32):
            lane=[r for r in c['inputs'] if r['npu_id']==n]
            assert [r['input_order'] for r in lane]==list(range(len(lane)))
            scale=1000000 if key=='high' else 100000
            assert all(r['request_id']==n*scale+r['input_order'] for r in lane)
            if key!='high':
                assert len({(r['total_tokens'],r['nql']) for r in lane})==len(lane)
                assert all(r['arrival_ms']==0 for r in lane)
        if key!='high':
            for name in ['metrics.json','input_metadata.json']:
                p=ROOT/'sources'/key/name
                assert hashlib.sha256(p.read_bytes()).hexdigest()==audit['source_sha256'][name],str(p)
        result['cases'][key]={'passed':True,'input_count':len(rows),'fleet_utilization':s['fleet_utilization'],'window_ms':[s['window_start_ms'],s['window_end_ms']]}
    for p in ROOT.glob('*.md'):
        content=p.read_text(encoding='utf-8')
        assert 'TBD' not in content
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            if target.startswith(('https://','http://','#')):continue
            assert (p.parent/target).is_file(),(p.name,target)
            result['markdown_local_links_checked']+=1
    assert len(list((ROOT/'images').glob('*.png')))==14
    assert len(list((ROOT/'images').glob('*.svg')))==14
    result['passed']=True
    path=ROOT/'audits/final_delivery_check.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
