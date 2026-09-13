#!/usr/bin/env python3
"""Whole-warm demand and physical NPU receipt means, including gray periods."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import render_per_npu_layer_avg_ssu3 as source

ROOT,HERE,OUT = source.ROOT,source.HERE,source.HERE/'figures/ssu3'
PATH = OUT/'warm_bandwidth_means.json'
LEFT,RIGHT = 2000.,4000.


def compute(order, data):
    case=HERE/'validation20s/runs'/f'ssu3_{order}_k1_sync_seed7'/'baseline'
    trace=source.base.read(case/'trace.json.gz')
    command=source.base.read(case/'command.json')
    assert source.base.sha(case/'trace.json.gz')==command['trace_sha256']
    assert trace['columns']==source.base.COLS and trace['completed_simulation']
    assert all(trace['checks'].values()) and trace['window_ms'][0]<=LEFT<RIGHT<=trace['window_ms'][1]
    arr=np.asarray(trace.pop('rows'),dtype=float)
    for start,end,rate,resource,count in ((9,10,40.,4,3),(11,12,50.,1,32)):
        assert np.allclose(arr[:,end]-arr[:,start],arr[:,6]*1000/rate,atol=1e-8,rtol=1e-9)
        for n in range(count):
            lane=arr[arr[:,resource]==n]
            lane=lane[np.argsort(lane[:,start])]
            assert np.all(lane[1:,start]>=lane[:-1,end]-1e-8)
    # Integrate only the physical part inside [2,4), including partial blocks.
    received_ms=np.maximum(0.,np.minimum(arr[:,12],RIGHT)-np.maximum(arr[:,11],LEFT))
    received=np.bincount(arr[:,1].astype(int),weights=received_ms*50./1000.,minlength=32)
    disk_ms=np.maximum(0.,np.minimum(arr[:,10],RIGHT)-np.maximum(arr[:,9],LEFT))
    disk_bytes=np.bincount(arr[:,4].astype(int),weights=disk_ms*40./1000.,minlength=3)
    assert np.all(disk_bytes*1000/(RIGHT-LEFT)<=40.+1e-7)
    old=source.base.read(OUT/f'bandwidth_{order}.json.gz')
    assert old['num_ssu']==3 and old['window_ms']==[LEFT,RIGHT]
    rows=[]
    for npu,lane in data['lanes'].items():
        weighted=math.fsum(source.base.clip(a,z,LEFT,RIGHT)*B for a,z,B in lane['demands'])
        mean_B=weighted/(RIGHT-LEFT)
        mean_b=float(received[npu])*1000/(RIGHT-LEFT)
        source.base.close(mean_B,old['summary'][npu]['nominal_demand_gib_s'])
        source.base.close(mean_b,old['summary'][npu]['npu_receive_gib_s'])
        source.base.close(lane['U_percent'],old['summary'][npu]['U_percent'])
        rows.append(dict(order=order,npu=npu,U_percent=lane['U_percent'],
                         mean_demand_GiB_s=mean_B,mean_supply_GiB_s=mean_b,
                         received_GiB=float(received[npu]),window_ms=RIGHT-LEFT))
    totals=dict(order=order,U_percent=data['U_percent'],
                sum_mean_demand_GiB_s=math.fsum(r['mean_demand_GiB_s'] for r in rows),
                sum_mean_supply_GiB_s=math.fsum(r['mean_supply_GiB_s'] for r in rows),
                per_card_mean_demand_GiB_s=math.fsum(r['mean_demand_GiB_s'] for r in rows)/32,
                per_card_mean_supply_GiB_s=math.fsum(r['mean_supply_GiB_s'] for r in rows)/32,
                disk_mean_service_GiB_s=(disk_bytes*1000/(RIGHT-LEFT)).tolist())
    return dict(rows=rows,totals=totals)


def load_verified(order):
    a=json.loads(PATH.read_text())
    assert a['all_checks_passed'] and a['num_ssu']==3 and a['window_ms']==[LEFT,RIGHT]
    for section in ('sources','builders'):
        for name,digest in a[section].items():assert source.base.sha(ROOT/name)==digest,name
    return a['results'][order]


def main():
    cache,data=source.verified_data()
    results={order:compute(order,data[order]) for order in ('random','ordered')}
    sources=dict(cache['sources'])
    sources.update(source.base.SOURCES)
    sources[str(source.CACHE.relative_to(ROOT))]=source.base.sha(source.CACHE)
    builders={str(p.relative_to(ROOT)):source.base.sha(p) for p in (Path(__file__),Path(source.__file__))}
    audit=dict(all_checks_passed=True,no_new_simulation=True,num_npu=32,num_ssu=3,seed=7,
               window_ms=[LEFT,RIGHT],sources=sources,builders=builders,
               demand_definition='Time integral of current admitted request B_i=V/C over [2,4)s, divided by 2s; includes stalls and gray periods.',
               supply_definition='Physical NPU received GiB inside [2,4)s divided by 2s; includes cross-request, gray periods and partial blocks.',
               results=results)
    PATH.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    rows=[row for data in results.values() for row in data['rows']]
    with PATH.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md=['# 3 SSU：每卡平均带宽需求与实际供给','',
        '32 NPU、3 SSU × 40 GiB/s、seed 7、warm [2,4) 秒。所有数值使用同一个2秒窗口，包含等待、请求切换和窗口边缘；供给以NPU实际收到的数据为准。','',
        '```text','平均需求 = 每段 B_i × 该段在窗口内的持续时间，加起来后除以 2 秒',
        '平均供给 = 窗口内实际收到的总数据量 / 2 秒',
        '平均利用率 U = 窗口内实际计算时间 / 2 秒','```','',
        '平均需求是当前已接纳请求 V/C 参考值的时间平均，不是新增I/O字节率。平均供给根据逐块接收起止时间积分，边界只计窗口内实际到达的部分；不是只对有传输的时段取平均，也不是把每个层周期均值简单平均。','',
        '图右侧的平均供给包含灰区，不能只把可见蓝线（灰区留空）的面积除以2秒来复现它。图右侧的平均需求则是完整紫线按时间加权的均值。','',
        '**平均供给 / 平均需求不等于平均利用率。** 同一窗口内会切换不同请求，且存在预取、跨请求与窗口边界；不能把两个整窗均值相除当作此前完整内部周期的 b_i/B_i。实际平均供给低于总容量，也不能证明逐盘逐时刻欠载。','',
        '| 顺序 | 每卡平均需求 GiB/s | 每卡平均供给 GiB/s | 32卡需求合计 GiB/s | 32卡供给合计 GiB/s | 整机U |',
        '|---|---:|---:|---:|---:|---:|']
    for order,d in results.items():
        t=d['totals']
        md.append(f'| {order.title()} | {t["per_card_mean_demand_GiB_s"]:.6f} | {t["per_card_mean_supply_GiB_s"]:.6f} | {t["sum_mean_demand_GiB_s"]:.6f} | {t["sum_mean_supply_GiB_s"]:.6f} | {t["U_percent"]:.6f}% |')
    md+=['','每卡均值=32张卡对应数值之和/32；合计表示这32张卡的总和，不要把两种口径混用。','',
         '| NPU | Random U | Random 平均需求 | Random 平均供给 | Ordered U | Ordered 平均需求 | Ordered 平均供给 |',
         '|---:|---:|---:|---:|---:|---:|---:|']
    for n in range(32):
        r,o=[results[k]['rows'][n] for k in ('random','ordered')]
        md.append(f'| {n} | {r["U_percent"]:.2f}% | {r["mean_demand_GiB_s"]:.6f} | {r["mean_supply_GiB_s"]:.6f} | {o["U_percent"]:.2f}% | {o["mean_demand_GiB_s"]:.6f} | {o["mean_supply_GiB_s"]:.6f} |')
    md+=['','逐卡带宽单位均为GiB/s。','',
         '[Random 32卡PNG](random_all_32npu_layer_average.png) · [Ordered 32卡PNG](ordered_all_32npu_layer_average.png) · [CSV](warm_bandwidth_means.csv) · [来源及完整数值](warm_bandwidth_means.json)','']
    PATH.with_suffix('.md').write_text('\n'.join(md))
    for order,d in results.items():print(json.dumps(d['totals'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
