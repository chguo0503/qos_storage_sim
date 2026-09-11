#!/usr/bin/env python3
"""Build an illustrated input guide from frozen role-separated manifests."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures' / 'separate'
KEYS = [(1, 128), (1, 256), (1, 384), (192, 768)]
NAMES = dict(zip(KEYS, ['S1', 'S2', 'S3', 'L']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(nlong, order):
    path = HERE / 'inputs' / f'role_l{nlong}_s{32-nlong}_seed7_{order}.json.gz'
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    assert len(data['requests']) == 19456
    assert all(r['arrival_time_ms'] == 0 for r in data['requests'])
    lanes = defaultdict(list)
    for r in data['requests']:
        lanes[r['npu_id']].append(r)
    for lane in lanes.values():
        lane.sort(key=lambda r: r['request_id'])
    assert set(lanes) == set(range(32))
    counts = [Counter(NAMES[(r['load']['seq_len_k'], r['load']['nql'])] for r in lanes[n]) for n in range(32)]
    assert all(set(counts[n]) == ({'L'} if n < nlong else {'S1', 'S2', 'S3'}) for n in range(32))
    assert sum((x for x in counts), Counter()) == Counter(S1=6400, S2=6400, S3=6400, L=256)
    return path, data, lanes, counts


def grouped_rows(counts):
    groups = []
    for n, count in enumerate(counts):
        value = tuple(count.get(k, 0) for k in ['S1', 'S2', 'S3', 'L'])
        if groups and groups[-1][2] == value:
            groups[-1][1] = n
        else:
            groups.append([n, n, value])
    return groups


def figure_links(stem, prefix='figures/separate/'):
    return ' / '.join(f'[{ext.upper()}]({prefix}{stem}.{ext})' for ext in ('png', 'pdf', 'svg'))


def main():
    with (HERE/'per_seed.csv').open() as f:
        metrics = {(int(r['long_npu_count']), r['order']): r for r in csv.DictReader(f)
                   if r['seed'] == '7' and r['strategy'] == 'baseline'}
    sources, cases = {}, {}
    figure_names = []
    for n in (11, 6):
        for order in ('random', 'round_robin'):
            cases[n, order] = read_manifest(n, order)
            path = cases[n, order][0]
            sources[str(path.relative_to(HERE))] = sha(path)
            row = metrics[n, order]
            assert row['all_study_conditions_met'] == 'True'
            assert float(row['full_run_any_ssu_over_capacity_ms']) == 0
            for kind in ('baseline', 'demand'):
                stem = f'l{n}_s{32-n}_{order}_{kind}'
                figure_names.append(stem)
    figure_names.append('l11_s21_random_zoom')
    for name in figure_names:
        for ext in ('png', 'pdf', 'svg'):
            assert (FIG/f'{name}.{ext}').is_file(), (name, ext)

    text = ['**固定长短卡：输入如何分配、图应该怎么看**', '',
            '这里的“长卡／短卡”指任务分工，所有 NPU 的硬件配置相同。长卡只处理 L，短卡只处理 S1、S2、S3；'
            '全部 32 张卡仍共享同一组 6 块 SSU。没有为长短卡划分专用磁盘，也没有运行时跨卡迁移。'
            '原混合实验也固定绑定 NPU；这次改变的是长短请求分到不同卡，因此不能把变化单独归因于“固定绑定”。', '',
            '**全部图均为 seed 7 的 Baseline 单次轨迹**，从现有仿真日志绘制，没有重跑仿真、拼接时间或加宽等待。'
            '统一主窗 [2000,4000) ms；每盘 40 GiB/s、每卡接收链路 50 GiB/s、每请求 8 层。'
            '图中值与此前报告的五种子均值略有不同，是统计范围不同。', '',
            '**输入的四种请求**', '',
            '全局仍是原实验的 19,456 条构造请求：三种短请求各 6,400 条，长请求 256 条。'
            '下面的 C、V 都是单层值；每条请求有八层。1K 短画像采用外推，192K/768 长画像采用插值，'
            '这组不是后面的原始 data 四画像实验。', '',
            '| 名称 | 序列长度 / NQL | 单层计算 C（ms） | 单层读取 V（MiB） | 整条8层纯计算（ms） |',
            '|---|---|---:|---:|---:|']
    request_by_key = {}
    for r in cases[11, 'random'][1]['requests']:
        key = (r['load']['seq_len_k'], r['load']['nql'])
        request_by_key[key] = r
    for key in KEYS:
        load = request_by_key[key]['load']
        c = load['per_layer_us']/1000
        text.append(f'| {NAMES[key]} | {key[0]}K / {key[1]} | {c:.6f} | {load["per_layer_kv_gb"]*1024:.6f} | {8*c:.6f} |')

    text += ['', '**每张 NPU 实际分到多少条**', '',
             '下表中的数量都是“每张卡”的数量。同一行的卡具有相同配额，但 Random 下的请求身份和排列不相同。'
             '两种绑定方案分别使用完整的同一全局请求集合，不能把两种方案的请求数量相加。', '',
             '| 绑定方案 | NPU编号 | 卡数 | 每卡 S1 | 每卡 S2 | 每卡 S3 | 每卡 L | 每卡总请求 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for n in (11, 6):
        for first, last, values in grouped_rows(cases[n, 'random'][3]):
            text.append('| ' + ' | '.join([f'{n}长/{32-n}短', f'{first}–{last}', str(last-first+1),
                                         *map(str, values), str(sum(values))]) + ' |')

    text += ['', '分配时，先把每种画像对应的原请求身份独立打乱，再从该角色编号最小的 NPU 开始轮流分发。'
             '不能整除的余数分给前面的卡，所以会出现 23/24、304/305 等配额差别。'
             '分卡完成后才安排卡内顺序；同一绑定方案内的 Random 与轮流排序保持每张卡的请求身份、C、V 和实际落盘位置相同。', '',
             '**卡内怎样输入和执行**', '',
             '所有请求在 t=0 就进入各自卡的有限待处理队列；不是把整条请求的八层读取都在 t=0 发给 SSD。'
             '每卡每次处理一条请求，按既有逐层预取机制推进；计算当前层时可以预取下一层，最后一层计算时可以预取本卡下一请求的首层。', '',
             '- 长卡：队列里全是 L，Random 和轮流排序的长请求身份顺序完全一致。\n'
             '- 短卡 Random：分别独立打乱本卡的整份 S1/S2/S3 清单。\n'
             '- 短卡 round_robin（轮流）：S1 → S2 → S3 → S1 → S2 → S3，直到清单耗尽。'
             '这不是原混合实验的四组 Ordered 相位构造。', '',
             '例如，11长/21短方案中，下面是两张短卡 Random 队列的真实前12条：', '',
             '| NPU | 真实队列前12条（从左向右执行） |', '|---|---|']
    for npu in (11, 12):
        lane = cases[11, 'random'][2][npu]
        sequence = ' → '.join(NAMES[(r['load']['seq_len_k'], r['load']['nql'])] for r in lane[:12])
        text.append(f'| {npu} | {sequence} |')
    text += ['', '短卡始终没有长请求，因此它也不会在短请求末层突然预取本卡下一条长请求。'
             '它仍会与其他长卡共享 SSD 的 FIFO，受到那些长读取的影响。', '',
             '每条请求原有的 176 KiB 分块和 SSU 落盘位置原样保留。原条带规则是 '
             '`SSU = (块序号 + 原NPU编号//4) % 6`；重新分卡后不按新 NPU 编号重新计算 placement。'
             '因此每张新卡的不同请求可能保留不同的原条带起点。', '',
             '[详细分配与随机种子公式](input_assignment.md)；'
             '[逐卡配额CSV](input_assignment_by_npu.csv)。四个完整队列CSV位于 `input_sequences/`，可查到每条请求的新卡、队列位置和原身份。', '',
             '**独立图片下载**', '',
             '| 绑定 | 短卡顺序 | 32卡计算／等待时间线 | 六盘名义需求 |', '|---|---|---|---|']
    for n in (11, 6):
        for order in ('random', 'round_robin'):
            stem = f'l{n}_s{32-n}_{order}'
            text.append(f'| {n}长/{32-n}短 | {order} | {figure_links(stem+"_baseline")} | {figure_links(stem+"_demand")} |')
    text += ['', '**11长/21短的局部放大：** '+figure_links('l11_s21_random_zoom')+'.', '',
             '建议先打开 11长/21短 Random 的时间线：NPU 0–10 的绿色是长计算，NPU 11–31 的蓝色是短计算，橙色是接纳后的真实 I/O 等待。'
             '长卡与短卡分界在 NPU 10 和 11 之间。6长/26短图的分界在 NPU 5 和 6 之间。所有卡在整个主窗都有任务。', '',
             '![11长/21短 Random 时间线](figures/separate/l11_s21_random_baseline.png)', '',
             '再看对应的六盘需求图。它画的是每张卡“当前请求单层在该盘的读取量 / 单层计算时间”之和，'
             '不是磁盘实际吞吐，也不是物理队列长度。', '',
             '![11长/21短 Random 六盘名义需求](figures/separate/l11_s21_random_demand.png)', '',
             '| seed 7 情况 | 整机 U | 短卡 U | 长卡 U | 逐事件扫描的全程名义峰值（GiB/s） | 全程超40时间 |',
             '|---|---:|---:|---:|---:|---:|']
    for n in (11, 6):
        for order in ('random', 'round_robin'):
            r = metrics[n, order]
            text.append(f'| {n}/{32-n} {order} | {float(r["device_utilization_percent"]):.4f}% | '
                        f'{float(r["short_card_utilization_percent"]):.4f}% | {float(r["long_card_utilization_percent"]):.4f}% | '
                        f'{float(r["full_run_max_ssu_gib_s"]):.6f} | 0 ms |')
    text += ['', '不依赖具体排序的逐盘名义需求上界为：11长/21短 **31.735895 GiB/s**，6长/26短 **26.559836 GiB/s**，均低于40。'
             '这与表中逐事件扫描得到的某次运行名义峰值是两个概念。名义需求不额外叠加跨请求首层预取，但仿真保留所有真实读取及其等待。', '',
             '**局部放大图的读法**', '',
             '![11长/21短 Random 局部放大](figures/separate/l11_s21_random_zoom.png)', '',
             '该图在完整落入暖窗的内部短层等待中，选择最大的一次来说明现象，不代表平均等待。短卡 NPU12 的前层计算只有约0.528 ms，'
             '随后等待约11.109 ms，约为自身单层计算的21.04倍。同期长卡 NPU8 的一段读取跨度约12.012 ms，'
             '完整落在25.612 ms计算内，因此该层没有额外等待。精确时间戳见 [zoom_evidence.json](figures/separate/zoom_evidence.json)。', '',
             '读取跨度包含排队、服务和传输，不等于这张卡一直占着SSD。此图用现有层日志作同时间参照，'
             '没有该case的逐块物理服务轨迹，因此不能认定图中的特定长请求就是短请求的FIFO队头阻塞者。', '',
             '这些图说明固定角色的构造输入会持续暴露短卡等待；它们不满足原混合实验“每张卡同时有长短”的条件，'
             '也不能推广为所有原始 data 输入都一样差。6长卡方案的整批尾部还有绑定不均衡，主窗图没有把尾部空卡时间算入。', '',
             '**复现与来源**', '',
             '主图：`plot_role_separate.py`；局部图：`plot_role_zoom.py`；输入导出：`export_input_assignment.py`。'
             '绘图与队列导出不修改冻结输入、策略或仿真结果。图文件自带来源信息，绘图审计JSON和输入审计JSON保留路径、SHA256及逐项核验。', '']
    (HERE/'fixed_assignment_guide.md').write_text('\n'.join(text))
    index = ['**固定长短卡的独立图片**', '', '[输入如何分配、图应该怎么看](../../fixed_assignment_guide.md)。', '',
             '统一为 seed 7 Baseline、32 NPU / 6 SSU、[2,4) 秒。每张图独立提供 PNG、PDF、SVG；不是五种子平均曲线。', '',
             '| 文件主题 | 下载 |', '|---|---|']
    for name in figure_names:
        index.append(f'| {name} | {figure_links(name, "")} |')
    index += ['', '全部图及说明已按文件保存在本目录，可逐项下载；如需ZIP，可运行研究目录的 `build_fixed_figure_guide.py` 按需生成，ZIP不重复提交到Git。', '']
    (FIG/'README.md').write_text('\n'.join(index))

    include = [HERE/'fixed_assignment_guide.md', HERE/'input_assignment.md', HERE/'input_assignment_by_npu.csv',
               HERE/'input_assignment_audit.json', HERE/'per_seed.csv', Path(__file__).resolve(),
               HERE/'plot_role_separate.py', HERE/'plot_role_zoom.py', HERE/'export_input_assignment.py',
               HERE/'run_role_separated.py']
    include += sorted((HERE/'input_sequences').glob('*.csv'))
    include += sorted(p for p in FIG.iterdir() if p.is_file())
    assert all(p.is_file() for p in include)
    files = {str(p.relative_to(HERE)): sha(p) for p in include}
    manifest = {'source_manifests': sources, 'source_per_seed_csv_sha256': sha(HERE/'per_seed.csv'),
                'seed': 7, 'npu': 32, 'ssu': 6, 'window_ms': [2000, 4000], 'simulation_rerun': False,
                'figure_count': len(figure_names), 'files': files,
                'bundle_scope': 'Figures, explanatory Markdown, input queues and plotting/export source; full simulator inputs/results remain in the project at recorded paths.'}
    delivery = HERE/'fixed_figure_delivery.json'
    delivery.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    include.append(delivery)
    bundle = HERE/'fixed_assignment_figures.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in include:
            z.write(p, p.relative_to(HERE))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
    print(json.dumps({'guide': str(HERE/'fixed_assignment_guide.md'), 'bundle': str(bundle),
                      'figure_count': len(figure_names), 'bundle_bytes': bundle.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
