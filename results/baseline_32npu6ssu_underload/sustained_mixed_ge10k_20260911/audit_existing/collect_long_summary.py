#!/usr/bin/env python3
"""Summarize the five independently audited long jobs; never fills missing data."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import audit_sustained as audit

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'long_validation_7200s'

def main():
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');args=p.parse_args()
    planpath=STUDY/'plan.json';plan=audit.read(planpath);assert len(plan['jobs'])==5
    rows=[];identity_checks={};sources={str(planpath):audit.sha(planpath),str(Path(__file__)):audit.sha(__file__)}
    for job in plan['jobs']:
        label=job['input']['label'];strategy=job['strategy'];main='_6l_' in label;order=job['input']['mode']
        short='main_random' if main and order=='random' else 'main_ordered' if main else 'backup_ordered'
        target=HERE/f'long_{short}_{strategy}_validation.json';prefetch=HERE/f'long_{short}_{strategy}_prefetch_demand.json';phase=HERE/f'long_{short}_{strategy}_phase.json'
        row=dict(label=label,scope='main' if main else 'backup',order=order,strategy=strategy,seed=7,status='pending',
            technical_passed=None,original_primary_conditions_passed=None,U_2_60_percent=None,U_4_60_percent=None,
            two_second_U_min_percent=None,two_second_U_max_percent=None,all_32_active=None,minimum_role_switches=None,
            min_fast_short_C_ms=None,min_fast_short_fraction=None,cards_fast_and_long_ge5percent=None,
            warm_admission_SLO_passed=None,warm_admission_SLO_count=None,warm_admission_SLO_percent=None,
            full_run_nominal_max_ssu_gib_s=None,full_run_nominal_over40_ms=None,
            prefetch_compute_only_max_ssu_gib_s=None,prefetch_compute_only_over40_ms=None,
            prefetch_through_wait_max_ssu_gib_s=None,prefetch_through_wait_over40_ms=None,
            long_group_noninitial_stall_ms=None,initial_phase_arc_ms=None,final_phase_arc_ms=None,maximum_phase_arc_ms=None,
            manifest_sha256=job['input']['manifest_sha256'],result_sha256=None,audit_path=str(target))
        commandpath=STUDY/'runs'/label/strategy/'command.json'
        cmd=audit.read(commandpath) if commandpath.exists() else {}
        if cmd.get('status')!='complete':row['status']='simulation_'+cmd.get('status','pending');rows.append(row);continue
        if not target.exists():row['status']='audit_pending';rows.append(row);continue
        d=audit.read(target);sources[str(target)]=audit.sha(target);result=Path(d['result']);manifest=Path(d['manifest'])
        assert d['strategy']==strategy and d['submit_seed']==job['input']['seed']==7
        assert d['input_audit']['fingerprint']==job['input']['input_fingerprint']
        assert all(Path(p).is_file() and audit.sha(p)==s for p,s in d['source_sha256'].items())
        identity_checks[label+'/'+strategy]=True
        assert d['source_sha256'][str(result.resolve())]==cmd['output_sha256']==audit.sha(result)
        assert d['source_sha256'][str(manifest.resolve())]==job['input']['manifest_sha256']==audit.sha(manifest)
        assert d['source_sha256'][str((HERE/'audit_sustained.py').resolve())]==audit.sha(HERE/'audit_sustained.py')
        w=d['primary_window'];later=next(x for x in d['windows'] if (x['start_ms'],x['end_ms'])==(4000,60000));bins=[x for x in d['windows'] if x['duration_ms']==2000]
        assert (w['start_ms'],w['end_ms'])==(2000,60000) and len(bins)==29
        row.update(status='complete' if d['technical_audit_passed'] else 'technical_failed',technical_passed=d['technical_audit_passed'],original_primary_conditions_passed=d['primary_conditions_passed'],
            U_2_60_percent=w['U_percent'],U_4_60_percent=later['U_percent'],two_second_U_min_percent=min(x['U_percent'] for x in bins),two_second_U_max_percent=max(x['U_percent'] for x in bins),
            all_32_active=w['all_32_active'],minimum_role_switches=w['minimum_per_card_role_switches'],min_fast_short_C_ms=w['min_fast_short_C_ms'],min_fast_short_fraction=w['min_fast_short_fraction'],
            cards_fast_and_long_ge5percent=w['cards_fast_and_long_ge5percent'],warm_admission_SLO_passed=w['warm_admission_SLO']['passed'],warm_admission_SLO_count=w['warm_admission_SLO']['count'],
            warm_admission_SLO_percent=w['warm_admission_SLO']['rate_percent'],full_run_nominal_max_ssu_gib_s=d['nominal_full_run']['max_ssu_gib_s'],full_run_nominal_over40_ms=d['nominal_full_run']['any_ssu_over40_ms'],result_sha256=cmd['output_sha256'])
        if not prefetch.exists():row['status']='secondary_pending'
        else:
            z=audit.read(prefetch);sources[str(prefetch)]=audit.sha(prefetch);assert z['result_sha256']==cmd['output_sha256'] and z['manifest_sha256']==job['input']['manifest_sha256'] and z['window_ms']==[2000,60000]
            assert z['script_sha256']==audit.sha(HERE.parent/'audit_prefetch_demand.py')
            for variant,prefix in [('compute_budget_only','prefetch_compute_only'),('through_exposed_wait','prefetch_through_wait')]:
                row[prefix+'_max_ssu_gib_s']=z['variants'][variant]['max_ssu_gib_s'];row[prefix+'_over40_ms']=z['variants'][variant]['any_ssu_over40_ms']
        if order=='ordered':
            if not phase.exists():row['status']='phase_pending'
            else:
                z=audit.read(phase);sources[str(phase)]=audit.sha(phase);assert z['source_sha256'][str(result)]==cmd['output_sha256']
                row.update(long_group_noninitial_stall_ms=z['long_group_full_run']['after_initial_L0_exposed_stall_ms'],initial_phase_arc_ms=z['cycles'][0]['circular_cover_arc_ms'],
                    final_phase_arc_ms=z['cycles'][-1]['circular_cover_arc_ms'],maximum_phase_arc_ms=max(c['circular_cover_arc_ms'] for c in z['cycles']))
        rows.append(row)
    complete=sum(x['status']=='complete' for x in rows)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=complete,expected=5,rows=rows,source_sha256=sources,completed_identity_and_current_source_checks=identity_checks,
        definitions=['U clips actual layer compute into the declared window and divides by 32 times window duration.',
            'SLO cohort: admission in [2000,60000), follow full completion, completion-admission <= 1.5*8*raw per-layer C. It is not arrival-clock or hardware TTFT; cohorts can differ by policy and order.',
            'Fast short is short role with NQL1024; NQL4096 bridge is excluded from its per-card C share. This secondary >=5% count does not change original role-based acceptance.',
            'Prefetch variants are additional demand proxies with actual next payload, not actual throughput or an all-deadline feasibility proof.',
            'Only seed7 and a finite 60-second horizon: no multi-seed mean or infinite-horizon stationary claim. Cancelled predecessor runs are excluded.'])
    (HERE/'long_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'long_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 60秒验证：独立审计汇总','',f'完整审计 {complete}/5。所有行均为 seed7；空值表示等待，未用 0 或 pilot 数字补齐。','',
        '| 输入 | 策略 | 状态 | U[2,60) | U[4,60) | 2秒窗 min–max | 暖接纳SLO | 每卡快短+长均≥5% |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.6f}%'
    for r in rows:
        label=('主方案' if r['scope']=='main' else '备选12L')+('随机' if r['order']=='random' else '定序')
        interval='—' if r['two_second_U_min_percent'] is None else f"{r['two_second_U_min_percent']:.6f}–{r['two_second_U_max_percent']:.6f}%"
        counts='—' if r['cards_fast_and_long_ge5percent'] is None else f"{r['cards_fast_and_long_ge5percent']}/32"
        lines.append(f"| {label} | {r['strategy']} | {r['status']} | {fmt(r['U_2_60_percent'])} | {fmt(r['U_4_60_percent'])} | {interval} | {fmt(r['warm_admission_SLO_percent'])} | {counts} |")
    lines+=['','SLO从接纳计时，并随访完整完成；不同策略的窗口接纳人口可能不同。两种预取需求代理与原始容量判据分别保留在JSON/CSV中；次级检查不改变原接受条件。','',*['- '+s for s in result['definitions']]]
    (HERE/'long_summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(complete=complete,expected=5,statuses={r['label']+'/'+r['strategy']:r['status'] for r in rows})))
    if args.require_complete and complete!=5:raise SystemExit(1)

if __name__=='__main__':main()
