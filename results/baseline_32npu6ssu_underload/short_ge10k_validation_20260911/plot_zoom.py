#!/usr/bin/env python3
"""Draw an independently audited, observed raw176 FIFO episode; no simulation."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures' / 'separate'
DIAG = HERE / 'diagnostics'
SHORT_ID, SHORT_LAYER = 27000017, 1
LEFT, RIGHT = 2000.0, 2036.0
BLUE, GREEN, ORANGE = '#26749B', '#4EA889', '#ECA13B'
INK, GRAY, PURPLE = '#17364B', '#526575', '#8154A8'


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b, c, d):
    return max(0.0, min(b, d)-max(a, c))


def main():
    physical_path = DIAG / 'ordered_physical_equivalence_audit.json'
    physical = read(physical_path)
    assert physical['passed'] and physical['physical_and_control_results_exactly_equal']
    trace_path = DIAG / 'ordered_block_trace.json.gz'
    assert physical['trace_sha256'] == sha(trace_path)
    trace = read(trace_path)
    assert trace['audit']['passed'] is False  # Preserve original strict wall-clock mismatch.
    reference_path = Path(trace['audit']['reference_result'])
    manifest_path = Path(trace['audit']['manifest'])
    assert sha(reference_path) == physical['reference_sha256']
    assert sha(manifest_path) == trace['audit']['manifest_sha256']
    raw, manifest = read(reference_path), read(manifest_path)
    requests = {r['request_id']: r for r in manifest['requests']}
    batches = {b['member_request_ids'][0]: b for b in raw['summary']['microbatch_metrics']}
    rows = [dict(zip(trace['columns'], r)) for r in trace['rows']]
    sb = batches[SHORT_ID]
    sl = sorted(sb['layer_metrics'], key=lambda x:x['layer'])
    sprev, sread = sl[SHORT_LAYER-1], sl[SHORT_LAYER]
    sflows = [r for r in rows if (r['request_id'], r['layer']) == (SHORT_ID, SHORT_LAYER)]
    assert len(sflows) == len(manifest['placements'][requests[SHORT_ID]['placement_index']][0]) == 248
    critical = max(sflows, key=lambda r:r['link_end_ms'])
    assert math.isclose(critical['link_end_ms'], sread['io_ready_time_ms'], abs_tol=1e-8)
    disk, qstart, qend = critical['ssu_id'], critical['enqueue_ms'], critical['ssd_start_ms']
    blockers = sorted((r for r in rows if r['ssu_id'] == disk and
                       overlap(qstart,qend,r['ssd_start_ms'],r['ssd_end_ms']) > 0),
                      key=lambda r:r['ssd_start_ms'])
    service, counts, long_pairs = defaultdict(float), Counter(), defaultdict(float)
    for r in blockers:
        role = requests[r['request_id']]['load']['role']
        dt = overlap(qstart,qend,r['ssd_start_ms'],r['ssd_end_ms'])
        service[role] += dt
        counts[role] += 1
        assert r['path_id'] == 0 and r['enqueue_ms'] <= qstart+1e-8
        if role == 'long': long_pairs[(r['request_id'],r['layer'])] += dt
    assert math.isclose(math.fsum(service.values()), qend-qstart, abs_tol=1e-7)
    assert all(a['ssd_end_ms'] <= b['ssd_start_ms']+1e-8 for a,b in zip(blockers,blockers[1:]))
    # Choose the genuine Long request-layer contributing most observed service
    # in the critical block's queue wait; ties use the smallest request ID.
    long_id, long_layer = min(long_pairs, key=lambda k:(-long_pairs[k],k))
    lb = batches[long_id]
    lread = next(l for l in lb['layer_metrics'] if l['layer'] == long_layer)
    assert long_layer == 0
    lane = sorted((b for b in batches.values() if b['npu_id'] == lb['npu_id']),
                  key=lambda b:b['admission_time_ms'])
    previous_batch = lane[lane.index(lb)-1]
    previous_id = previous_batch['member_request_ids'][0]
    lprev = max(previous_batch['layer_metrics'], key=lambda l:l['layer'])
    assert requests[previous_id]['load']['role'] == requests[long_id]['load']['role'] == 'long'
    assert requests[previous_id]['load']['seq_len_k'] == requests[long_id]['load']['seq_len_k'] == 176
    assert math.isclose(lread['io_start_time_ms'],lprev['compute_start_ms'],abs_tol=1e-8)
    assert math.isclose(lb['admission_time_ms'],lprev['compute_end_ms'],abs_tol=1e-8)
    assert lread['io_ready_time_ms'] < lprev['compute_end_ms'] and lread['io_barrier_wait_ms'] == 0
    lflows = [r for r in rows if (r['request_id'],r['layer']) == (long_id,long_layer)]
    assert len(lflows) == len(manifest['placements'][requests[long_id]['placement_index']][0]) == 1400
    assert math.isclose(max(r['link_end_ms'] for r in lflows),lread['io_ready_time_ms'],abs_tol=1e-8)
    short_c = sprev['compute_duration_ms']
    short_read = sread['io_ready_time_ms']-sread['io_start_time_ms']
    stall = sread['compute_start_ms']-sprev['compute_end_ms']
    assert math.isclose(stall, sread['io_barrier_wait_ms'], abs_tol=1e-8)
    assert math.isclose(short_read-short_c, stall, abs_tol=1e-8)
    long_c, long_read = lprev['compute_duration_ms'], lread['io_ready_time_ms']-lread['io_start_time_ms']
    stages = {'layer_release_to_critical_enqueue_ms':qstart-sread['io_start_time_ms'],
              'ssd_queue_ms':qend-qstart,
              'ssd_service_ms':critical['ssd_end_ms']-qend,
              'link_queue_ms':critical['link_start_ms']-critical['ssd_end_ms'],
              'link_service_ms':critical['link_end_ms']-critical['link_start_ms']}
    assert math.isclose(math.fsum(stages.values()),short_read,abs_tol=1e-8)
    long_fraction = service['long']/(qend-qstart)
    warm = next(w for w in raw['windows'] if (w['start_ms'],w['end_ms']) == (2000,4000))
    checks = {'physical_replay_equivalence_passed':physical['passed'],
              'short_layer_all_248_blocks_captured':len(sflows)==248,
              'long_layer_all_1400_blocks_captured':len(lflows)==1400,
              'critical_HBM_completion_matches_layer_ready':critical['link_end_ms']==sread['io_ready_time_ms'],
              'queue_wait_covered_exactly_by_nonoverlapping_SSD_services':True,
              'all_predecessors_enqueued_before_critical_block':True,
              'selected_long_is_actual_FIFO_predecessor':(long_id,long_layer) in long_pairs,
              'all_long_predecessor_layers_are_L0':all(r['layer']==0 for r in blockers if requests[r['request_id']]['load']['role']=='long'),
              'prefetch_is_long_to_long':True,
              'all_five_critical_block_stages_sum_to_layer_read_lifetime':True}
    assert all(checks.values())
    info = {'checks':checks,'passed':True,'no_simulation_in_plotter':True,
            'window_ms':[LEFT,RIGHT],'block_capture_ms':trace['audit']['capture_ms'],
            'selection':'Prespecified short NPU27/request27000017/L1, maximum full-warm internal Short stall; actual Long predecessor maximizes SSD service overlapping critical block queue wait, tie smallest request ID.',
            'short':{'request_id':SHORT_ID,'npu':sb['npu_id'],'layer':SHORT_LAYER,
                     'profile':[32,1024],'previous_layer':sprev,'read_layer':sread,
                     'C_ms':short_c,'read_lifetime_ms':short_read,'stall_ms':stall},
            'long':{'computing_request_id':previous_id,'prefetched_request_id':long_id,
                    'npu':lb['npu_id'],'profile':[176,1024],'computing_layer':lprev,
                    'prefetched_layer':lread,'prefetch_is_cross_request_L0':True,
                    'C_ms':long_c,'read_lifetime_ms':long_read,'next_request_admission_ms':lb['admission_time_ms'],
                    'own_service_during_target_queue_ms':long_pairs[(long_id,long_layer)]},
            'critical_short_block':critical,'critical_stages':stages,
            'critical_ssu':disk,'blocking_service_ms_by_role':dict(service),
            'blocking_block_count_by_role':dict(counts),'long_service_fraction':long_fraction,
            'blocking_long_request_layers':[{'request_id':k[0],'layer':k[1],'service_overlap_ms':v}
                                           for k,v in sorted(long_pairs.items())],
            'warm_2000_4000_device_U_percent':warm['mean_npu_utilization']*100,
            'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
            'input_fingerprint':raw['input_fingerprint'],'reference_result':str(reference_path),
            'reference_sha256':sha(reference_path),'trace_sha256':sha(trace_path),
            'physical_equivalence_audit_sha256':sha(physical_path),
            'core_source_sha256':raw['core_and_policy_sha256'],'plot_script_sha256':sha(__file__),
            'limitations':['This one selected episode is not a window average or unique attribution of every stall.',
                           'Layer I/O lifetime includes submission, SSD queue/service, link queue/service; it is not continuous SSD service.',
                           'The SSD row shows actual service overlapping the critical block queue, plus that block itself; blank portions are not claims of idle.',
                           'NPU stall starts after the admitted request preceding compute ends; no pre-admission time is called stall.',
                           'Current-request nominal V/C does not add the next request L0 separately; it is not actual throughput or a deadline bound.',
                           'Strict full-record equality remains false for five wall-clock duration fields; all physical/control results are exactly equal.']}

    font = Path('/home/chguo/.fonts/msyh.ttc')
    assert font.exists()
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size':15,'axes.unicode_minus':False,'pdf.fonttype':42,
                         'svg.fonttype':'path','svg.hashsalt':'raw176-observed-FIFO-zoom',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_axes([.265,.315,.71,.49])
    fig.text(.03,.965,'原始画像的真实 FIFO 局部：长读排在前，短卡算完后仍需等数据',
             fontsize=23,fontweight='bold',color=INK,va='top')
    fig.text(.03,.915,'32 NPU / 6 SSU · ordered Baseline · seed 7 · 短 32K / 1024，长 176K / 1024 · 全部 C / V 来自 data',
             fontsize=16,color=GRAY)
    fig.text(.03,.868,f'短卡：读取 {short_read:.3f} ms − 计算 {short_c:.3f} ms = 停工 {stall:.3f} ms；长卡：读取 {long_read:.3f} ms < 计算 {long_c:.3f} ms',
             fontsize=18,color=INK)
    labels=[f'长卡 NPU {lb["npu_id"]}：正在计算\n请求 {previous_id} 的第 8 层（L7）',
            f'同一长卡：预取下一请求\n请求 {long_id} 的第 1 层（L0）',
            f'短卡 NPU 27：计算 / 停工\n请求 {SHORT_ID} 的第 1、2 层',
            '短卡：第 2 层（L1）读取\n从触发到全部数据抵达 HBM',
            '关键短块的实际阶段\n盘排队 → 盘读 → 链路排队 / 传输',
            f'SSU {disk} 的实际盘服务\n只展开目标块排队时段与自身盘读']
    ypos=[5,4,3,2,1,0]

    def bar(a,z,y,color,height=.47,hatch=None,edge=None,alpha=1):
        lo,hi=max(a,LEFT),min(z,RIGHT)
        if hi>lo:ax.broken_barh([(lo,hi-lo)],(y-height/2,height),facecolors=color,
                               edgecolors=edge if edge else 'none',hatch=hatch,alpha=alpha,linewidth=.7)

    bar(lprev['compute_start_ms'],lprev['compute_end_ms'],5,GREEN)
    ax.text((lprev['compute_start_ms']+lprev['compute_end_ms'])/2,5,f'上一条 Long 末层计算：{long_c:.3f} ms',ha='center',va='center',color='white',fontsize=16)
    bar(lread['io_start_time_ms'],lread['io_ready_time_ms'],4,'#DDEFE8',hatch='///',edge=GREEN)
    ax.text((lread['io_start_time_ms']+lread['io_ready_time_ms'])/2,4,f'下一条 Long 首层读取：{long_read:.3f} ms',ha='center',va='center',color=INK,fontsize=14)
    ax.annotate('到齐后仍有计算预算',xy=(lread['io_ready_time_ms'],4),xytext=(2024.0,4.3),
                arrowprops={'arrowstyle':'->','color':GREEN},fontsize=14,color=INK)
    bar(sprev['compute_start_ms'],sprev['compute_end_ms'],3,BLUE)
    bar(sprev['compute_end_ms'],sread['compute_start_ms'],3,ORANGE)
    bar(sread['compute_start_ms'],sread['compute_end_ms'],3,BLUE)
    for a,z,label in [(sprev['compute_start_ms'],sprev['compute_end_ms'],'第 1 层计算'),
                      (sprev['compute_end_ms'],sread['compute_start_ms'],f'IO stall {stall:.3f} ms'),
                      (sread['compute_start_ms'],sread['compute_end_ms'],'第 2 层计算')]:
        ax.text((a+z)/2,3,label,ha='center',va='center',color='white' if '计算' in label else INK,fontsize=13.5)
    bar(sread['io_start_time_ms'],sread['io_ready_time_ms'],2,'#DAEAF4',hatch='///',edge=BLUE)
    ax.text((sread['io_start_time_ms']+sread['io_ready_time_ms'])/2,2,f'整层读取跨度 {short_read:.3f} ms（不是盘服务）',ha='center',va='center',fontsize=13.5,color=INK)
    for t in [sprev['compute_end_ms'],sread['io_ready_time_ms']]:ax.plot([t,t],[1.65,3.5],ls=':',lw=1.0,color=GRAY)
    bar(qstart,qend,1,'#F6D9A3',hatch='///',edge=ORANGE)
    ax.text((qstart+qend)/2,1,f'目标短块盘排队 {stages["ssd_queue_ms"]:.3f} ms',ha='center',va='center',color=INK,fontsize=14)
    bar(qend,critical['ssd_end_ms'],1,PURPLE)
    bar(critical['ssd_end_ms'],critical['link_start_ms'],1,'#D5C5E3')
    bar(critical['link_start_ms'],critical['link_end_ms'],1,PURPLE)
    ax.annotate(f'盘读 {stages["ssd_service_ms"]*1000:.3f} μs\n链路排队 {stages["link_queue_ms"]:.3f} ms\n传输 {stages["link_service_ms"]*1000:.3f} μs',
                xy=(critical['ssd_end_ms'],1),xytext=(2025.0,1.05),va='center',fontsize=13,
                arrowprops={'arrowstyle':'->','color':PURPLE},color=INK)
    for r in blockers:
        bar(max(qstart,r['ssd_start_ms']),min(qend,r['ssd_end_ms']),0,
            GREEN if requests[r['request_id']]['load']['role']=='long' else BLUE,height=.4)
    bar(qend,critical['ssd_end_ms'],0,PURPLE,height=.4)
    ax.add_patch(Rectangle((qstart,-.31),qend-qstart,.62,fill=False,edgecolor=ORANGE,lw=1.8))
    ax.text((qstart+qend)/2,-.52,f'{counts["long"]} 个长块 + {counts["short"]} 个其它短块先获服务',ha='center',va='top',fontsize=13,color=INK)
    ax.set(xlim=(LEFT,RIGHT),ylim=(-.95,5.7),yticks=ypos,yticklabels=labels,
           xticks=list(range(2000,2036,5)),xlabel='真实仿真时间（ms）；区间未移动或拼接')
    ax.tick_params(axis='y',length=0,pad=14,labelsize=14)
    ax.tick_params(axis='x',labelsize=14)
    ax.grid(axis='x',color='#DFE5E8',lw=.6)
    ax.set_axisbelow(True)
    for s in ('left','right','top'):ax.spines[s].set_visible(False)
    fig.text(.03,.242,f'直接证据：目标块盘排队的 {qend-qstart:.3f} ms 中，{service["long"]:.3f} ms（{long_fraction*100:.2f}%）在服务排在前面的 Long 块；盘无空闲间隙。',
             fontsize=17,color=INK,fontweight='bold')
    fig.text(.03,.196,f'这些前驱长块全是“Long → Long”的跨请求 L0 预取；正在计算的请求 {previous_id} 与预取中的请求 {long_id} 不同。',fontsize=16,color=INK)
    fig.text(.03,.150,'下一条 Long 的首层由上一条 Long 的末层计算掩盖；不是短 → 长切换造成计算预算骤减。短卡的等待则超过了自己的计算预算。',fontsize=15,color=GRAY)
    fig.text(.03,.105,'下两行分别表示目标块经历的阶段和该盘实际服务顺序；空白处未展开，不代表盘空闲。读取跨度包含排队、盘读与链路，不能当连续占盘。',fontsize=14,color=GRAY)
    fig.text(.03,.063,f'局部最大短等待示例，不代表平均；原 [2,4)s 设备 U = {warm["mean_npu_utilization"]*100:.4f}%。名义 V/C 不额外叠加下一请求 L0，也不等于实际吞吐。',fontsize=14,color=GRAY)
    fig.text(.03,.025,f'原结果 SHA256 {sha(reference_path)[:16]} · 物理重放完全等价；仅 5 项墙钟耗时不同 · 完整数值和边界见 ordered_zoom_evidence.json',fontsize=12,color=GRAY)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    text_boxes=[t.get_window_extent(renderer) for t in fig.texts+list(ax.get_yticklabels())+list(ax.get_xticklabels())+[ax.xaxis.label]]
    assert all(box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1 for box in text_boxes), 'Text outside figure bounds'
    OUT.mkdir(parents=True,exist_ok=True)
    artifact_paths=[]
    for ext in ('png','pdf','svg'):
        p=OUT/f'ordered_zoom.{ext}'
        metadata = {'Title':'原始 32K / 176K：观察到的 FIFO 局部证据'} if ext!='png' else {
            'Description':json.dumps({'input_fingerprint':raw['input_fingerprint'],
                                      'reference_sha256':sha(reference_path),
                                      'plot_sha256':sha(__file__)})}
        fig.savefig(p,dpi=200,metadata=metadata)
        artifact_paths.append(p)
    plt.close(fig)
    info['figure_files']=[{'path':str(p),'sha256':sha(p)} for p in artifact_paths]
    (OUT/'ordered_zoom_evidence.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':True,'critical_ssu':disk,'short_stall_ms':stall,'long_predecessor':long_id,
                      'previous_computing_request':previous_id,'stages':stages,
                      'service_by_role':dict(service),'output':str(OUT)},ensure_ascii=False))


if __name__ == '__main__':
    main()
