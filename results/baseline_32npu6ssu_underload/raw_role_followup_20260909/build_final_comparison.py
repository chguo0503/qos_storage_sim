#!/usr/bin/env python3
"""Require all paired five-seed results before publishing the small final table."""
import csv,json,statistics,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    mixed=json.loads((HERE/'mixed_rebinding/results.json').read_text());fixed=json.loads((HERE/'followup_results.json').read_text())
    assert mixed['all_complete'] and mixed['all_order_pairs_passed'] and not mixed['errors']
    bridge=json.loads((HERE/'mixed_rebinding/fixed_rebinding_population_bridge.json').read_text())
    assert bridge['passed'] and bridge['manifests']==15 and bridge['requests_each']==1212
    f=[r for r in fixed['rows'] if r['spec_name']=='raw176_three_l20' and r['mode']=='fixed']
    m=[r for r in mixed['rows'] if r['spec_name']=='raw176_extendedhot']
    assert len(f)==10 and len(m)==20
    assert all(r['audit_pass'] and r['active_underload_valid'] for r in f)
    assert all(r['audit_pass'] and r['strict_mixed_underload_valid'] for r in m)
    rows=[]
    for name,items in [('固定20长卡/12短卡',f),('每卡混合、完整队列随机',[r for r in m if r['mode']=='random']),('每卡混合、定序延长短段',[r for r in m if r['mode']=='ordered'])]:
        row={'assignment':name,'seeds':5,'population':1212,'long_profile':'176K/1024','window_ms':'2000:4000'}
        for policy in ('baseline','once'):
            rr=[r for r in items if r['strategy']==policy]
            assert sorted(r['seed'] for r in rr)==[7,19,43,67,101]
            for short,key in [('U_percent','device_utilization_percent'),('SLO_percent','warm_slo_percent')]:
                vv=[r[key] for r in rr];row[f'{policy}_{short}_mean']=statistics.mean(vv);row[f'{policy}_{short}_sample_sd']=statistics.stdev(vv)
            row[f'{policy}_max_disk_gib_s']=max(r['max_ssu_nominal_gib_s'] for r in rr)
        rows.append(row)
    drops=[]
    for seed in [7,19,43,67,101]:
        rr={r['mode']:r for r in m if r['seed']==seed and r['strategy']=='baseline'}
        drops.append(rr['random']['device_utilization_percent']-rr['ordered']['device_utilization_percent'])
    ordered=[r for r in m if r['mode']=='ordered' and r['strategy']=='baseline']
    out={'rows':rows,'paired_random_minus_ordered_pp':{'mean':statistics.mean(drops),'sample_sd':statistics.stdev(drops),'per_seed':dict(zip([7,19,43,67,101],drops))},
         'ordered_baseline_mean_long_cards':statistics.mean(r['mean_long_cards'] for r in ordered),
         'ordered_baseline_mean_short_cards':statistics.mean(r['mean_short_cards'] for r in ordered),
         'all_20_mixed_runs_valid':True,'all_30_selected_runs_active_and_underload':True,
         'min_per_card_per_role_compute_ms_among_ordered':min(min(r['min_card_long_compute_ms'],r['min_card_short_compute_ms']) for r in ordered),
         'input_population_bridge':'mixed_rebinding/fixed_rebinding_population_bridge.json',
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'mixed_rebinding/results.json',HERE/'followup_results.json',Path(__file__)]}}
    (HERE/'final_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    with (HERE/'final_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['**同一批1212条原始请求，五种子最终对照**','','32 NPU、6 SSU、warm[2,4)s。表为百分比均值±样本标准差；SLO只统计[2,4)s内接纳的请求，并跟踪到完成，判断完成−接纳≤1.5×8C，不含接纳前排队。实际纳入的请求集合随策略变化。每组均为seeds7/19/43/67/101。','',
        '| 分配/顺序 | Baseline U% | Once U% | Baseline SLO% | Once SLO% |','|---|---:|---:|---:|---:|']
    for r in rows:
        vals=[f"{r[f'{p}_{v}_mean']:.2f} ± {r[f'{p}_{v}_sample_sd']:.2f}" for p,v in [('baseline','U_percent'),('once','U_percent'),('baseline','SLO_percent'),('once','SLO_percent')]]
        lines.append('| '+r['assignment']+' | '+' | '.join(vals)+' |')
    lines+=['',f"Baseline随机→定序下降{statistics.mean(drops):.4f}±{statistics.stdev(drops):.4f}个百分点。定序平均长卡/短卡{out['ordered_baseline_mean_long_cards']:.4f}/{out['ordered_baseline_mean_short_cards']:.4f}，不是每刻固定20/12。",'',
        '全部30格全卡active且全程逐盘欠载；混合的20格都满足每张卡暖窗内长短各有正计算时间。固定与混合保留全局身份、C/V、arrival、逐块placement；两个混合顺序还保留逐卡人口，固定与混合的逐卡人口不同。','',
        '[完整方法与全部探索](mixed_research_report.md)；[最终数值JSON](final_comparison.json)；[输入队列CSV](mixed_rebinding/assignments_seed7.csv)。','']
    (HERE/'final_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},ensure_ascii=False))

if __name__=='__main__':main()
