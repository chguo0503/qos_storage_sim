#!/usr/bin/env python3
"""Index completed native runs and pilots without treating pilots as full runs."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    formal, pending, pilots = [], [], []
    for path in sorted((HERE/'runs').glob('*/command.json')):
        record = read(path)
        if record['status'] != 'complete':
            pending.append(dict(case=path.parent.name, status=record['status']))
            continue
        assert record['completed_simulation'] and all(record['checks'].values())
        assert not record.get('smoke', False)
        assert record['result_sha256'] == digest(path.parent/'result.json.gz')
        assert record['manifest_sha256'] == digest(path.parent/'manifest.json.gz')
        result = read(path.parent/'result.json.gz')
        assert result['input_fingerprint'] == record['input_fingerprint']
        warm, full = result['analysis'][0], result['analysis'][-1]
        assert (warm['start_ms'], warm['end_ms']) == (2000., 4000.)
        assert full['full_population']
        windows = []
        for r in result['analysis']:
            windows.append(dict(start_ms=r['start_ms'], end_ms=r['end_ms'],
                                full_population=r['full_population'], U=r['U_percent'],
                                slo=r['slo'], strict_underload=r['demand']['strict_underload_all_disks'],
                                all_npus_active=r['all_npus_active'],
                                mixed_cards=r['role_and_stall']['npus_with_A_and_B_compute']))
        formal.append(dict(case=path.parent.name, policy=record['policy'],
                           constructed_profile=result['metadata']['constructed_profile'],
                           selected_addresses=bool(result['metadata'].get('address_selection')),
                           input_fingerprint=record['input_fingerprint'],
                           U=warm['U_percent'], SLO=warm['slo']['percent'],
                           passed=warm['slo']['passed'], count=warm['slo']['count'],
                           warm_under=warm['demand']['strict_underload_all_disks'],
                           full_under=full['demand']['strict_underload_all_disks'],
                           all_npus_active=warm['all_npus_active'],
                           mixed_cards=warm['role_and_stall']['npus_with_A_and_B_compute'],
                           warm_peak=max(warm['demand']['per_disk_max_GiB_s']),
                           full_peak=max(full['demand']['per_disk_max_GiB_s']),
                           role_U={k:v['active_U_percent'] for k,v in warm['by_role'].items()},
                           role_SLO={k:v['admission']['slo']['1.5'] for k,v in warm['by_role'].items()},
                           stall=warm['role_and_stall']['io_stall']['by_kind_card_ms'],
                           windows=windows))
    for base in (HERE/'pilots', HERE/'larger_blocks'/'pilots'):
        for path in sorted(base.glob('*/command.json')):
            r = read(path)
            if r['status'] != 'complete_pilot':
                pending.append(dict(case=str(path.parent.relative_to(HERE)), status=r['status']))
                continue
            assert not r['completed_simulation'] and r['source_unchanged']
            m = r['measurement']
            pilots.append(dict(case=path.parent.name,
                               kind='validation' if path.parent.name.startswith('validation') else 'screen',
                               U=m['U_percent'], warm_under=m['demand']['strict_underload_all_disks'],
                               initial_4s_under=m['initial_4s_demand']['strict_underload_all_disks'],
                               all_npus_active=m['all_npus_active'], mixed_cards=m['mixed_cards'],
                               eligible=(m['initial_4s_demand']['strict_underload_all_disks'] and
                                         m['all_npus_active'] and m['mixed_cards'] == 32),
                               input_fingerprint=r['input_fingerprint'], spec=r['spec']))
    screens = [r for r in pilots if r['kind'] == 'screen']
    data = dict(updated_utc=datetime.now(timezone.utc).isoformat(), formal=formal, pilots=pilots,
                pending=pending, formal_count=len(formal), pilot_screens=len(screens),
                distinct_pilot_screen_inputs=len({r['input_fingerprint'] for r in screens}),
                pilot_validation_count=len(pilots)-len(screens))
    (HERE/'search_index.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    columns = ['case','policy','constructed_profile','selected_addresses','U','SLO','passed','count',
               'warm_under','full_under','all_npus_active','mixed_cards','warm_peak','full_peak']
    with (HERE/'formal_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(formal)
    lines = ['# 全部正式结果与短试验索引', '',
             f'完整排空 {len(formal)} 次；原仿真 4 秒筛选 {len(screens)} 次（{data["distinct_pilot_screen_inputs"]} 种输入）；另有 {len(pilots)-len(screens)} 次截断口径校验。',
             '', '正式结果均核对结果/输入文件 SHA、输入指纹与运行器完整校验。筛选不提供 SLO，不代表长期结果。', '',
             '“全程欠载”包括启动和排空，不表示全程所有卡活跃。利用率和 SLO 以下均为 warm [2,4)；全程 U 不用于证明稳态退化。', '',
             '| 完整实验 | U | SLO×1.5 | warm欠载 | 全程欠载 | 每卡混合 | 输入 |',
             '|---|---:|---:|---|---|---:|---|']
    for r in formal:
        nature = '合成+选址' if r['selected_addresses'] else ('合成' if r['constructed_profile'] else 'data原行')
        lines.append(f'| {r["case"]} | {r["U"]:.4f}% | {r["SLO"]:.4f}% | {r["warm_under"]} | {r["full_under"]} | {r["mixed_cards"]}/32 | {nature} |')
    lines += ['', '## 4秒原仿真筛选', '',
              '“通过”要求前4秒逐盘严格欠载、warm全部32卡活跃且都实际计算过A/B。先后调参属于自适应搜索，不能把最优值当随机总体均值。', '',
              '| 短试验 | warm U | warm欠载 | 前4秒欠载 | warm混合卡数 | 通过 |',
              '|---|---:|---|---|---:|---|']
    for r in pilots:
        lines.append(f'| {r["case"]} | {r["U"]:.4f}% | {r["warm_under"]} | {r["initial_4s_under"]} | {r["mixed_cards"]} | {r["eligible"]} |')
    if pending:
        lines += ['', '尚未完整完成的任务：'+', '.join(r['case']+' ('+r['status']+')' for r in pending)]
    lines += ['', '连续服务代理是候选生成工具，单独保存在 ideal_search；其预测不计入正式结果。额外的静态尾请求校准概念验证如存在，单独报告，不混入上述4秒筛选表。', '']
    (HERE/'search_log.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in data.items() if k not in ('formal','pilots')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
