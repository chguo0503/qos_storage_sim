#!/usr/bin/env python3
"""Export native followup evidence, keeping extrapolated compute explicit."""
from pathlib import Path
import csv
import gzip
import json
import zipfile
import hashlib
import shutil

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/lower_fifo_followup_20260914'
DELIVERY = ROOT.parents[1] / 'lower_fifo_followup_deliverables'


def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)


def export_case(case):
    man, result, meta = (read(case/f) for f in ('manifest.json.gz','result.json.gz','metadata.json'))
    loads = {q['request_id']:q['load'] for q in man['requests']}
    request_rows, layer_rows = [], []
    for batch in result['summary']['microbatch_metrics']:
        rid = batch['member_request_ids'][0]; q = loads[rid]
        threshold = 1.5 * meta['n_layers'] * q['per_layer_us']/1000
        execution = batch['completion_time_ms']-batch['admission_time_ms']
        base = dict(request_id=rid,npu_id=batch['npu_id'],role=q['role'],category=q['category'],group=q['profile_group'],policy=meta['probe_policy'],
            total_k=q['total_tokens']/1024,nql=q['nql'],read_mib_per_layer=q['per_layer_kv_gb']*1024,
            compute_ms_per_layer=q['per_layer_us']/1000,demand_gib_s=q['required_bw_input_gbps'],
            pure_single40_gib_s_read_ms=q['per_layer_kv_gb']/40*1000,
            compute_extrapolated=q['profile_construction'].get('extrapolated',False))
        request_rows.append(dict(**base,admission_ms=batch['admission_time_ms'],completion_ms=batch['completion_time_ms'],
            execution_ttft_ms=execution,arrival_ttft_ms=batch['completion_time_ms']-q.get('arrival_ms',0.),
            slo15_ms=threshold,slo15_met=execution<=threshold+1e-8))
        prior = batch['admission_time_ms']
        for layer in sorted(batch['layer_metrics'],key=lambda x:x['layer']):
            layer_rows.append(dict(**base,layer=layer['layer'],io_release_ms=layer['io_start_time_ms'],
                io_ready_ms=layer['io_ready_time_ms'],compute_start_ms=layer['compute_start_ms'],
                compute_end_ms=layer['compute_end_ms'],io_elapsed_ms=layer['io_ready_time_ms']-layer['io_start_time_ms'],
                exposed_stall_ms=layer['compute_start_ms']-prior))
            prior = layer['compute_end_ms']
    for suffix,rows in [('requests',request_rows),('layers',layer_rows)]:
        p=OUT/'csv'/f'{case.name}_{suffix}.csv';p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    DELIVERY.mkdir(parents=True,exist_ok=True)
    old = ROOT/'results/random_multitype_search_20260914/screen/strict_extended_seed7_fifo'
    new = OUT/'native/lower_agent_910074_seed7_fifo'
    sensitivity = [p for p in sorted((OUT/'native20').glob('*')) if (p/'metrics.json').exists()]
    cases=[old,new]+sensitivity
    for case in cases:export_case(case)
    theory=read(OUT/'theory_accounting.json'); search=read(OUT/'search_agent_summary.json')
    corr=read(OUT/'correlated_summary.json'); sens=read(OUT/'theory_20k_extrapolated_sensitivity.json')
    metrics=[read(p/'metrics.json') for p in cases]
    table=[]
    for case,m in zip(cases,metrics):
        label = '原配置' if case==old else ('新增32K配置' if case==new else '20K计算外推')
        table.append(f"| {label} | {m['policy']} | {m['U_percent']:.4f}% | {m['short_U_percent']:.4f}% | {m['long_U_percent']:.4f}% | {m['slo_1p5_percent']:.2f}% |")
    audit20=[]
    for p in sorted((OUT/'audit20').glob('audits/*_audit.json')):
        a=read(p);w=a['windows']['warm_2000_4000ms'];f=a['windows']['full_run']
        audit20.append(dict(policy=a['policy'],passed=a['boundary_exempt_candidate_pass'],
            current_peak=max(f['current']['peak_per_unit_gib_s']),continuing_peak=max(f['continuing']['peak_per_unit_gib_s']),
            max_near=max(x['current']['fleet_near_capacity_percent'] for x in a['windows'].values()),
            internal_stall=w['utilization']['internal_stall_card_ms'],l0_stall=w['utilization']['l0_stall_card_ms'],
            boundary_floor=w['job_stats']['cross_role_boundary']['unavoidable_transfer_stall_lower_bound_card_ms']))
    audit_lines='\n'.join(f"| {a['policy']} | {a['passed']} | {a['current_peak']:.4f} | {a['continuing_peak']:.4f} | {a['max_near']:.3f}% | {a['internal_stall']:.3f} | {a['l0_stall']:.3f} | {a['boundary_floor']:.3f} |" for a in audit20)
    content=f'''# FIFO高利用率：扩展筛选记录

8 NPU；每卡混合；ring hash；原始FIFO Path0；8层；2–4秒固定统计窗口。普通需求全程每盘≤40 GiB/s；只豁免短↔长请求交界预取。所有Stall均计入利用率。

本轮得到更低的20K外推敏感性结果：FIFO整机94.61%、短请求86.70%，同输入Once为99.26%和99.84%。普通需求全程欠载，短↔长交界按既定口径豁免。20K计算时间尚未实测校准，因此这组结果不能当作data中已有20K实测画像的结论。

原来98.31%的结果只能说明局部FIFO阻塞存在，不能作为明显低利用率的坏例。本轮仍未找到满足条件且整机利用率低于90%的已验证案例；这是有限搜索的结果，不是不存在这类输入的证明。

## 原生模拟结果

| 输入 | 策略 | 整机U | 短请求U | 长请求U | TTFT SLO×1.5 |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

原配置：32K/NQL中心1280 ×8 +200K/NQL中心2048 ×1。新增32K配置仅把长请求NQL中心改为2304，比例相同。实际各卡40短+5长，seed7；每卡独立打乱并从邻域无放回取NQL。新32K配置原生U=98.5599%，比原配置更高，不能认为改进了坏例。其全程普通需求峰值36.7599/40，四个审计窗口最大近载占比3.4166%，审计通过。

TTFT按completion−admission计时，阈值1.5×8×原始单层计算时间。所有请求t=0排队，但入场前的卡上队列等待不包含在这一SLO口径；CSV同时提供arrival口径。表内SLO为warm入场请求。U按8×2000ms作分母，8卡在该窗口均有任务，边界等待没有删除。

## 为什么原配置仍高

- 短请求每层计算9.035ms，单个200K长层在40 GiB/s下纯读取6.647ms；单个长层通常能被掩盖，多个长层在前才容易暴露等待。
- warm中短请求内部782层，88层Stall，占11.25%；平均每层Stall0.273ms。长请求内部96层全无Stall。
- 88个短层Stall中，16个发生在L7。其余72个有同请求下一层：40个下一层恢复，32个继续Stall。层间错位会缓解部分堵塞，但不能说下一层总是不堵。
- 短请求占数量88.89%，却只占该窗口计算时间51.40%。长请求的大量高利用率计算稀释了短请求等待对整机均值的影响。
- 8卡×2秒=16000 card-ms。现有总等待270.260 card-ms；90%利用率需要1600 card-ms等待，是目前的5.920倍。若保持当前计算时间配比及长请求利用率，短请求利用率须约82.45%，当前是97.08%。这只是固定配比的条件计算，不是输入变化后的预测。

## 新筛选与排除原因

独立随机初筛{search['warm_evaluated']}组，覆盖2–4类请求；201组做完整代理审计，另做80组邻域审计和60个种子审计。完整代理使用层整体入队，原生是逐IO入队，二者排名可能不同。新增32K候选代理97.06%、原生98.56%，因此不能把代理值当成仿真完成值。

三类候选32K/1152×1、32K/256×4、200K/512×2（4SSU）在均分模型中约96.57–97.12%，实际ring检查3个种子的普通单盘峰值41.51/42.23/42.97 GiB/s，均失败。还有91.69%的代理结果，但普通需求超40持续约2453.86ms，违反全程欠载。

另筛选{corr['approximate_screened']}组跨卡相关的随机混合输入，探索长读取集中出现。其联合分布不同于独立随机，仍每卡混合。最低均分代理94.66%，实际ring代理95.11%，普通单盘峰值41.989，因超载排除。无交界物理下限的相关候选最低代理97.34%，未宣称为原生结果。

60种子是代理稳健性检查，不是60次原生实验；59种子通过，接受集合范围96.45%–99.72%。Seed48只做代理，未作为原生证据。逐盘检查仍必要：整机总需求低于总容量不代表最拥挤盘欠载。

## 20K外推敏感性

用户允许20K–200K；此前限制≥32K是保持计算时间在实测网格内的实现选择。data最小长度32K，20K没有实测行。本轮额外使用原有profile函数，从32K/48K外推20K，长度权重为1.75和−0.75，NQL仍在测量范围内插值。**它是未校准计算时间的敏感性实验，即使使用原生仿真器，也不是20K实测数据验证。**

192组外推代理中77组通过。最低93.4105%：20K/NQL中心1152与200K/NQL中心2048，6:1，1SSU，seed7。warm内部Stall889.230 card-ms；交界独占物理传输下界11.442 card-ms，约占全部等待1.09%，不属于以物理下限为主的候选。这些数值仅用于挑选原生敏感性输入。

原生敏感性使用每卡42个20K短请求+7个200K长请求，共392个请求，每卡独立随机打乱。短NQL范围1126–1178，长NQL范围2042–2054；各卡长度/NQL组合唯一。FIFO/Once输入指纹相同：be0c836adce2d3435fd82ca2edf1d803d96437e079a203eaf4f9a8be0b85eca2。

| 总长度 | NQL中心 | 每层读取MiB | 每层计算ms | 需求GiB/s | 单盘40 GiB/s纯读取ms |
|---|---:|---:|---:|---:|---:|
| 20K（计算外推） | 1152 | 25.953125 | 5.881291 | 4.309402 | 0.633621 |
| 200K | 2048 | 272.250000 | 70.740538 | 3.758370 | 6.646729 |

原生FIFO整机94.6147%、短请求86.7022%；同输入Once整机99.2567%、短请求99.8375%，分别改善4.6419和13.1353个百分点，长请求利用率则从99.4464%降到98.9465%。FIFO内部Stall732.491 card-ms，占全部861.642 card-ms等待约85.01%；交界独占传输下界11.442 card-ms仅为总等待1.33%。单纯调度不能消除这部分物理下界；内部等待也不能仅凭发生时间就全部归因于同一个长请求，原生同输入策略对照支持可改善的调度损失存在。

20K外推输入的warm入场SLO为FIFO126/128、Once137/137。全部相同392请求的SLO为FIFO387/392（98.7245%）、Once392/392；全输入CDF与上表warm入场口径不同，图中单独标注。

20K原生敏感性审计（等待单位card-ms，带宽GiB/s）：

| 策略 | 欠载审计通过 | 普通峰值 | 非交界预取峰值 | 最大近载占比 | 内部Stall | L0 Stall | 交界物理下界 |
|---|---|---:|---:|---:|---:|---:|---:|
{audit_lines}

近载定义沿用上一轮：整机需求≥总容量90%的时间不超过5%，分别检查全程、0–4.5s、0.5–4s、2–4s。current需求包括等待中的当前请求；continuing参考结束于计算deadline，未完成IO仍可能形成排队。不能由参考需求欠载推出队列为空，也不能把两种参考需求相加。每NPU实际接收≤50 GiB/s，每SSU实际服务≤40 GiB/s。

## 文件与复现

代码包内csv目录含逐请求/逐层数据，native与native20目录含原始manifest、result、metrics和2ms真实字节记录。所有原始核心代码保持原有行为，计算时间外推在独立文件中显式标记。图片单独打包，20K外推标注保留在每张图中。

```bash
python run_lower_fifo_followup.py --spec results/lower_fifo_followup_20260914/native_specs.json --case lower_agent_910074 --seed 7 --policy fifo --stage reproduce
python audit_boundary_underload.py --root results/lower_fifo_followup_20260914/reproduce --out results/lower_fifo_followup_20260914/reproduce_audit
python run_20k_sensitivity.py --spec results/lower_fifo_followup_20260914/native_20k_specs.json --case sensitivity20k_076 --seed 7 --policy fifo --stage reproduce20
python run_20k_sensitivity.py --spec results/lower_fifo_followup_20260914/native_20k_specs.json --case sensitivity20k_076 --seed 7 --policy once --stage reproduce20
python audit_boundary_20k_sensitivity.py --root results/lower_fifo_followup_20260914/reproduce20 --out results/lower_fifo_followup_20260914/reproduce20_audit
python newexport_lower_fifo_sensitivity.py
```
'''
    report=DELIVERY/'lower_fifo_followup.md';report.write_text(content,encoding='utf-8')
    image_bundle=DELIVERY/'lower_fifo_followup_images.zip'
    pngs=sorted((OUT/'figures/images').glob('*.png'))
    if pngs:
        with zipfile.ZipFile(image_bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in pngs:z.write(p,p.name)
    bundle=DELIVERY/'lower_fifo_followup_data_code.zip'
    items={}
    def include(p):
        if p.is_file():items['qos_storage_sim/'+str(p.relative_to(ROOT))]=p
    for p in ROOT.glob('*.py'):include(p)
    for p in (ROOT/'assets').glob('*'):include(p)
    for name in ['data','requirements.txt','SOURCE_PROVENANCE.json']:include(ROOT/name)
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix not in ('.log','.pyc','.png'):include(p)
    for p in old.rglob('*'):
        if p.is_file():include(p)
    include(ROOT/'results/boundary_exempt_underload_20260914/boundary_proxy_screen.json')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.write(report,'lower_fifo_followup.md')
        for arc,p in sorted(items.items()):z.write(p,arc)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
    receipts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [report]+([image_bundle] if pngs else [])+[bundle]]
    (DELIVERY/'validation.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(dict(deliverables=receipts,native_cases=len(cases),audit20=audit20),indent=2))


if __name__=='__main__':main()
