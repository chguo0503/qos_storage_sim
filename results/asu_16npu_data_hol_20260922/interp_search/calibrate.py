#!/usr/bin/env python3
"""Compare the fast selector against four completed native runs, read only."""
import csv
import gzip
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    quick={r['name']:r for r in json.loads((HERE/'all.json').read_text())['rows']}
    rows=[]
    for profile,ma,mb in [('I1',3302,2448),('I2',3328,2500)]:
        for q in (4,7):
            name=f'{profile}_A{ma}_B{mb}_s1_r1{q}_random_seed7'
            folder=HERE.parent/'runs'/name/'asu_baseline'
            with gzip.open(folder/'native_summary.json.gz','rt') as stream:native=json.load(stream)
            model=quick[f'{profile}_random_q{q}_seed7']
            for label,lo,hi in [('warm',2000,4000),('long',2000,10000)]:
                busy=0
                for batch in native['microbatch_metrics']:
                    for layer in batch['layer_metrics']:
                        busy+=max(0,min(hi,layer['compute_end_ms'])-max(lo,layer['compute_start_ms']))
                actual=busy/(16*(hi-lo))*100
                estimate=model['windows'][0] if label=='warm' else sum(model['windows'][:4])/4
                rows.append(dict(name=name,window=label,native_U_percent=actual,
                                 approximate_U_percent=estimate,
                                 approximation_minus_native_pp=estimate-actual))
    with (HERE/'calibration_native.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 近似筛选器对四场原生结果的校准','',
           '使用同一画像、seed7、12秒工作量的独立洗牌队列。原生U从层计算区间重新积分；近似长窗取前四个2秒子窗口平均，确保比较的都是[2,10)。','',
           '| 配置 | 窗口 | 原生U | 快模U | 快模−原生(百分点) |','|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['window']} | {r['native_U_percent']:.4f}% | {r['approximate_U_percent']:.4f}% | {r['approximation_minus_native_pp']:+.4f} |")
    lines+=['','误差随时间窗、画像变化，不能把这四场误差当作其他相位输入的固定校正量。周期候选必须独立用原生实现验证。','',
            '[机器可读比较](calibration_native.csv) · [复算程序](calibrate.py)','']
    (HERE/'calibration_native.md').write_text('\n'.join(lines));print('\n'.join(lines))


if __name__=='__main__':main()
