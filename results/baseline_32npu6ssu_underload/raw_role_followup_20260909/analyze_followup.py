#!/usr/bin/env python3
"""Audit direct-data probes and report all predeclared completed runs."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[1]
sys.path[:0]=[str(ROOT),str(STUDY)]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


auditor=module('followup_parent',STUDY/'analyze.py')
auditor.WINDOWS=((2000.0,4000.0),)
paired=module('followup_paired',STUDY/'paired_once_5seeds/analyze_paired.py')
from run_baseline_npu32_stress import load_manifest, read_json, write_json
from run_coflow_experiments import source_files


def analyze(item,strategy,path):
    signature={str(p):auditor.sha(p) for p in (path,Path(item['manifest']),Path(__file__),
        STUDY/'analyze.py',STUDY/'paired_once_5seeds/analyze_paired.py',ROOT/'data',path.parent/'command.json')}
    cache=HERE/'analysis_cache'/item['label']/f'{strategy}.json'
    if cache.exists():
        saved=read_json(cache)
        if saved.get('signature')==signature:return saved
    info=auditor.load_input(Path(item['manifest']),HERE)
    technical=auditor.analyze_result(path,info,HERE)
    raw=read_json(path)
    command=read_json(path.parent/'command.json')
    paired.EXPECTED_REQUESTS=len(info['_requests'])
    metrics,checks=paired.independent_metrics(raw,info,auditor)
    checks['complete_input_population']=checks.pop('request_metric_population_19456')
    checks.update(paired.policy_checks(raw,info))
    checks['command_completed']=command.get('status')=='complete' and command.get('returncode')==0
    checks['complete_core_hash_set']=set(raw['core_and_policy_sha256'])==set(source_files())
    checks['command_source_hash_match']=all(command['source_sha256'].get(name)==value for name,value in raw['core_and_policy_sha256'].items())
    requests,meta=load_manifest(item['manifest'])
    table=ast.literal_eval((ROOT/'data').read_text())
    checks['direct_data_C_and_V']=all(
        r.load['per_layer_us']==table[(r.load['seq_len_k'],r.load['nql'])][1]
        and math.isclose(math.fsum(v for _,v in r.placement[0]),table[(r.load['seq_len_k'],r.load['nql'])][3],abs_tol=1e-12)
        and r.load['constructed_profile'] is False and r.load['padding_gib_per_layer']==0
        for r in requests)
    checks['frozen_input_sha']=auditor.sha(item['manifest'])==item['manifest_sha256']
    window=technical['windows'][0]
    scan=technical['nominal_demand_scan']['full_run']
    warm=metrics['cohorts']['window_admissions']
    mixed_count=sum(c['by_role']['short']['compute_ms']>0 and c['by_role']['long']['compute_ms']>0 for c in window['per_npu'])
    audit_pass=technical['audit']['passed'] and all(checks.values())
    capacity=scan['max_ssu_gib_s']<40 and scan['max_npu_link_gib_s']<50
    fourth=technical['warmup']['all_fourth_completions_by_1500']
    row=dict(label=item['label'],spec_name=item['spec_name'],seed=item['seed'],mode=item['mode'],strategy=strategy,
        request_count=len(requests),long_cards=meta['long_cards'],short_cards=meta['short_cards'],profile_keys=meta['profile_keys'],
        device_utilization_percent=100*metrics['device_utilization'],warm_slo_percent=100*warm['admission']['rate'],
        warm_admissions=warm['count'],warm_slo_passed=warm['admission']['passed'],
        completion_after_window_count=warm['completion_after_window_end_count'],
        all_npus_active=window['all_npus_active'],warm_mixed_card_count=mixed_count,
        fourth_request_by_1500=fourth,max_fourth_completion_ms=max(technical['warmup']['fourth_completion_by_npu_ms']),
        max_ssu_nominal_gib_s=scan['max_ssu_gib_s'],any_ssu_over_capacity_ms=scan['any_ssu_over_capacity_ms'],
        max_npu_nominal_gib_s=scan['max_npu_link_gib_s'],full_run_capacity_pass=capacity,
        static_max_ssu_gib_s=info['static_max_proof']['max_ssu_gib_s'],audit_pass=audit_pass,
        active_underload_valid=audit_pass and capacity and window['all_npus_active'],
        warmup_underload_valid=audit_pass and capacity and window['all_npus_active'] and fourth,
        original_all_conditions_valid=audit_pass and capacity and window['all_npus_active'] and fourth and mixed_count==32,
        full_run_U_percent=100*technical['full_run']['device_utilization'],makespan_ms=metrics['makespan_ms'],
        l0_stall_ms=window['l0_exposed_stall_ms'],l1_7_stall_ms=window['l1_7_exposed_stall_ms'],
        result_path=str(path),result_sha256=auditor.sha(path),input_sha256=auditor.sha(item['manifest']))
    for role in ('short','long'):
        group=window['by_role'][role]
        row[role+'_pooled_U_percent']=100*group['pooled_utilization']
        row[role+'_warm_slo_percent']=100*metrics['by_role'][role]['window_admissions']['admission']['rate']
        row[role+'_warm_admissions']=metrics['by_role'][role]['window_admissions']['count']
    technical.pop('original_slo',None)
    for w in technical['windows']:w.pop('requests',None)
    out=dict(signature=signature,row=row,checks=checks,technical=technical,metrics=metrics)
    write_json(cache,out)
    return out


def main():
    jobs={}
    for p in sorted((HERE/'plans').glob('*.json')):
        plan=read_json(p)
        assert auditor.sha(plan['spec_file'])==plan['spec_sha256'],p
        assert all(auditor.sha(ROOT/name)==value for name,value in plan['source_sha256'].items()),p
        assert all(auditor.sha(i['manifest'])==i['manifest_sha256'] for i in plan['inputs']),p
        for j in plan['jobs']:
            jobs[j['input']['label'],j['strategy']]=j
    rows=[];pending=[];errors=[]
    for (label,strategy),job in jobs.items():
        paths=list((HERE/'runs'/label/strategy).glob('*.json.gz'))
        if not paths:
            pending.append(dict(label=label,strategy=strategy));continue
        assert len(paths)==1
        command_path=paths[0].parent/'command.json'
        if not command_path.exists() or read_json(command_path).get('status') is None:
            pending.append(dict(label=label,strategy=strategy));continue
        try:rows.append(analyze(job['input'],strategy,paths[0])['row'])
        except Exception as e:errors.append(dict(label=label,strategy=strategy,error=f'{type(e).__name__}: {e}'))
    groups=[]
    for key in sorted({(r['spec_name'],r['mode'],r['strategy']) for r in rows}):
        rr=[r for r in rows if (r['spec_name'],r['mode'],r['strategy'])==key]
        expected=[j for j in jobs.values() if (j['input']['spec_name'],j['input']['mode'],j['strategy'])==key]
        group=dict(spec_name=key[0],mode=key[1],strategy=key[2],completed=len(rr),planned=len(expected),
            seeds=sorted(r['seed'] for r in rr),valid_count=sum(r['active_underload_valid'] for r in rr))
        for field in ('device_utilization_percent','warm_slo_percent','short_pooled_U_percent','long_pooled_U_percent',
                      'short_warm_slo_percent','long_warm_slo_percent'):
            vals=[r[field] for r in rr]
            group[field]=dict(mean=statistics.mean(vals),sample_sd=statistics.stdev(vals) if len(vals)>1 else None)
        groups.append(group)
    summary=dict(completed=len(rows),planned=len(jobs),pending=pending,errors=errors,rows=rows,groups=groups,
        definition='[2000,4000) ms U; admission-clock TTFT proxy <=1.5*8C, all warm admissions followed to completion. All predeclared seeds retained; 4th by1500 reported separately.')
    write_json(HERE/'followup_results.json',summary)
    if rows:
        with (HERE/'followup_per_seed.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['**原始 data 固定与混合：补测结果**','',summary['definition'],'',
        f'已完成 {len(rows)}/{len(jobs)}；解析错误 {len(errors)}。单seed候选探索与五seed确认分开阅读，不删除表现不佳的种子。','',
        '| 配置 | 模式 | seed | 策略 | NPU U% | 暖接纳SLO% | 短角色U% | 全程逐盘峰值GiB/s | 全32卡active | 第4条≤1500ms | 欠载 |',
        '|---|---|---:|---|---:|---:|---:|---:|---|---|---|']
    for r in rows:
        lines.append(f"| {r['spec_name']} | {r['mode']} | {r['seed']} | {r['strategy']} | {r['device_utilization_percent']:.4f} | {r['warm_slo_percent']:.4f} | {r['short_pooled_U_percent']:.4f} | {r['max_ssu_nominal_gib_s']:.6f} | {r['all_npus_active']} | {r['fourth_request_by_1500']} | {r['full_run_capacity_pass']} |")
    lines+=['','fixed/mixed每个配置和seed使用同一批原请求，保留C、V、到达和实际落盘。固定模式长卡只处理长画像，短卡独立打乱其短画像；mixed把同一批请求按画像均匀分给32卡后独立打乱。',
        '', '名义需求沿用当前接纳请求逐盘V/C，不额外叠加下一请求首层预取；真实预取仍完整仿真。4th≤1500ms是此前附加的暖机判据，2048类长计算可能在纯计算上就无法满足，不能把这类结果悄悄并入满足旧判据的结果。','']
    (HERE/'followup_results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(completed=len(rows),planned=len(jobs),errors=errors,
        latest=[{k:r[k] for k in ('label','strategy','device_utilization_percent','warm_slo_percent','active_underload_valid')} for r in rows]),ensure_ascii=False))
    assert not errors
    assert all(r['audit_pass'] for r in rows)


if __name__=='__main__':main()
