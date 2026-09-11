#!/usr/bin/env python3
"""Publish existing-trace sensitivity separately from extended-input simulations."""
import argparse,json,statistics,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def lookup(d,mode,policy,a,b):return next(r for r in d['groups'] if (r['mode'],r['strategy'],r['start_ms'],r['end_ms'])==(mode,policy,a,b))
def val(r,k='U_percent'):return r[k]['mean']
def label(a,b):return f'[{a/1000:g},{b/1000:g})'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--require-repeated',action='store_true');args=ap.parse_args()
    old=read(HERE/'existing_windows.json');assert len(old['runs'])==30
    repeated=read(HERE/'repeated_windows.json') if (HERE/'repeated_windows.json').exists() else None
    audit=read(HERE/'audit_repeated.json') if (HERE/'audit_repeated.json').exists() else None
    if args.require_repeated:
        assert repeated and len(repeated['runs'])==6
        assert all(r['all_32_active'] for r in repeated['rows'])
        assert all(audit[k] for k in ['all_complete','all_input_audits_passed','all_completed_technical_audits_passed','all_original_prefixes_match'])
    cancelled=(HERE/'repeated_queues/cancellation.json').exists()
    done=min(r['first_card_done_ms'] for r in old['runs'])
    duration=[(2000,4000),(2000,4250),(2000,4350),(1000,4000),(0,4000)]
    text=['**利用率下降有多依赖统计窗口？**','',
      '有明显关系。先前的10.46个百分点是[2,4)秒这段输入与执行时序的结果，不能直接当成持续运行任意长时间的平均损失。I/O等待真实存在，但本输入把较多等待集中在角色交换前；五种子重复不能替代时间窗口敏感性检查。','',
      '**只换统计窗口，输入和仿真完全不变**','',
      '以下重算此前30份完整运行日志：固定长短卡、每卡随机混合、每卡定序混合，分别5seed×Baseline/Once。本表聚焦两个混合顺序，全部仍使用原始data及32 NPU/6 SSU；百分比为五seed的率均值。独立另一脚本重算了其中210个窗口，逐格时间账一致。','',
      '| 窗口（秒） | Baseline 随机U | Baseline 定序U | 差值（百分点） | Once 定序U | 30格全卡持续有请求 |',
      '|---|---:|---:|---:|---:|---|']
    for a,b in duration:
        r=lookup(old,'random','baseline',a,b);o=lookup(old,'ordered','baseline',a,b);once=lookup(old,'ordered','once',a,b)
        active=sum(g['all_active_count'] for g in old['groups'] if (g['start_ms'],g['end_ms'])==(a,b))
        text.append(f'| {label(a,b)} | {val(r):.2f}% | {val(o):.2f}% | {val(r)-val(o):.2f} | {val(once):.2f}% | {active}/30 |')
    text+=['','[0,4)包含冷启动，[1,4)也改变了原暖机起点，不能称为同一个warm条件。更直接的检查是保持起点2秒，延长到4.25或4.35秒。',
      f'所有对照最早有卡耗尽的时刻为{done/1000:.6f}秒；4.35秒是按50ms向下取整得到的共同安全截止，不是为了优化U选出的点。它仍是根据运行结果得到的事后描述性窗口，并非事先注册的独立验证。','',
      '**为什么只加0.25秒，结果就变化**','',
      '在所有卡始终有请求、时间只分为计算与I/O等待的本模型中：','',
      r'$U=1-\frac{\text{所有卡的I/O等待时间之和}}{32\times\text{窗口时长}}$。','',
      '原[2,4)内，定序Baseline平均累计等待约6.70卡·秒；卡·秒就是把各张卡的等待秒数加起来。在[4,4.25)内，这组Baseline已恢复到满计算，没有增加等待。扩大窗口新增8卡·秒计算，因此U升高、与随机输入的差距被摊薄。','',
      r'$U_{[2,4)}=1-6.7033/(32\times2)=89.53\%$；$U_{[2,4.25)}=1-6.7033/(32\times2.25)=90.69\%$。',
      '复核五个seed的完整原日志，定序Baseline在4秒之后直到本批结束都没有新增暴露到计算上的I/O stall。这不代表没有读取或磁盘排队，只代表后续读取及时就绪，计算没有因此停下来。','',
      '机制上，前20张长卡的集中读波在约3.29秒、3.80秒两次角色切换后被打散；后面的短画像和相位也发生变化。所以不能假定后面每两秒都会重现原窗口的损失。','',
      '**直接把有限输入的窗口拉到5、6、8秒，会混入耗尽后的空闲**','',
      '| 窗口 | 随机Baseline U | 定序Baseline U | 随机空闲卡时间占比 | 定序空闲卡时间占比 | 两混合组全卡有请求 |',
      '|---|---:|---:|---:|---:|---|']
    for a,b in [(2000,4500),(2000,5000),(2000,6000),(2000,8000)]:
        r=lookup(old,'random','baseline',a,b);o=lookup(old,'ordered','baseline',a,b)
        text.append(f'| {label(a,b)} | {val(r):.2f}% | {val(o):.2f}% | {val(r,"idle_percent"):.2f}% | {val(o,"idle_percent"):.2f}% | 否 |')
    text+=['','这些空闲不是I/O stall。窗口越长、越多卡已经做完，原“计算时间÷32卡窗口时间”的分母就越难单独反映调度阻塞。','',
      '甚至[2,8)会出现定序U高于随机的反转：8秒之前所有请求都已完成，整个输入的纯计算总量相同；定序只是在2秒前做得更少，把更多计算留到了2秒以后。这个反转不代表它处理更快。若统一取[0,8)，这批输入的总计算相同，U也会全部相同。','',
      '| 原混合顺序 | Baseline 整批最后完成时间（秒，5seed均值） | Once 整批最后完成时间 |',
      '|---|---:|---:|']
    for mode in ['random','ordered']:
        values=[statistics.mean(r['last_card_done_ms'] for r in old['runs'] if r['mode']==mode and r['strategy']==s)/1000 for s in ['baseline','once']]
        text.append(f'| {mode} | {values[0]:.3f} | {values[1]:.3f} |')
    text+=['','因此，同一每卡人口的定序Baseline整批完工仍慢于随机。固定角色与混合的每卡纯计算量分布不同，不能仅凭整个批次的完工时间把这些绑定方案的差异都归因于FIFO。','',
      '**延长输入，再扩大窗口**','',
      '把seed7原有每张卡的完整队列原样重复3遍，arrival仍全为0，保留C/V、落盘、绑定和每周期顺序，只重编唯一请求ID。三种分配均变为3636条请求，两个混合顺序仍有相同的逐卡人口。最小每卡纯计算为13.052秒，所以计划窗口[2,12)内不会因整卡任务做完而空闲。','',
      '这是更长的有限批次，所有请求仍在t=0入队；不是持续到达的稳态工作负载。循环按每张卡自己的执行进度发生，没有插入同步屏障或人工等待，也没有复制拼接旧日志。重复周期可能自然错开，正是需要观察的现象。只测seed7，不能当作五seed长期均值。','']
    if repeated:
        text+=['| 新输入窗口 | 固定Baseline U | 固定Once U | 混合随机Baseline U | 随机Once U | 混合定序Baseline U | 定序Once U | 随机−定序Baseline（百分点） |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for a,b in [(2000,4000),(2000,6000),(2000,8000),(2000,10000),(2000,12000),(4000,12000),(6000,12000)]:
            f=lookup(repeated,'fixed','baseline',a,b);r=lookup(repeated,'random','baseline',a,b);o=lookup(repeated,'ordered','baseline',a,b);q=lookup(repeated,'ordered','once',a,b)
            fq=lookup(repeated,'fixed','once',a,b);rq=lookup(repeated,'random','once',a,b)
            text.append(f'| {label(a,b)} | {val(f):.2f}% | {val(fq):.2f}% | {val(r):.2f}% | {val(rq):.2f}% | {val(o):.2f}% | {val(q):.2f}% | {val(r)-val(o):.2f} |')
        all_active=all(r['all_32_active'] for r in repeated['rows'])
        text+=['',f'7个窗口×6次运行全部32卡持续有请求：{all_active}。该表是seed7结果，与前面五seed均值分开列示。','',
          '全程逐盘需求仍按此前约定的当前请求V/C逐事件核查，不额外叠加下一个请求首层预取；物理读全部保留。这仍不保证瞬时读取不突发。','']
        if audit and audit['all_complete']:
            text+=['| 输入 | 策略 | 全程最高单盘名义需求 GiB/s | 任一盘超40的时长 ms | 全程欠载条件通过 |', '|---|---|---:|---:|---|']
            for r in audit['runs']:
                n=r['full_run_nominal']
                text.append(f'| {r["mode"]} | {r["strategy"]} | {n["max_ssu_gib_s"]:.6f} | {n["any_ssu_over_capacity_ms"]:.6f} | {r["scientific_full_run_nominal_capacity_passed"]} |')
            exact=all(r['original_prefix']['exact_event_times_and_identities_equal'] for r in audit['runs'])
            text+=['',f'扩队列与原seed7在[0,4)内，记录的请求接纳/完成及逐层I/O释放、ready、计算起止时刻完全一致：{exact}；两份独立分析也核对了[2,4)逐卡计算、等待、占用时间。这个验证限于日志中的请求/层时间里程碑，不是SSD逐块或整个离散事件流的等价证明。','']
            ob=next(r for r in audit['runs'] if r['mode']=='ordered' and r['strategy']=='baseline')
            warm=next(w for w in ob['windows'] if (w['start_ms'],w['end_ms'])==(2000,4000))
            late=next(w for w in ob['windows'] if (w['start_ms'],w['end_ms'])==(4000,12000))
            whole=next(w for w in ob['windows'] if (w['start_ms'],w['end_ms'])==(2000,12000))
            stall=lambda w:w['l0_stall_ms']+w['internal_stall_ms']
            text+=['**扩大后的下降还能持续吗？**','',
              f'这次混合定序的10秒观察段里，{100*stall(warm)/stall(whole):.2f}%的I/O等待集中在最早的[2,4)两秒；后面[4,12)八秒仅累计{stall(late)/1000:.3f}卡·秒等待，U为{late["device_U_percent"]:.2f}%。因此，当前定序配方的约10个百分点下降主要是阶段性现象。固定长短卡的扩展结果则仍接近91%，不能把两种分配的窗口敏感性混为一谈。','',
              '还要检查是否只是长请求占比增加，把短请求的问题掩盖了。下表的短类别利用率=短请求计算卡时间÷短请求占用卡时间，分母包含该类请求自己的I/O等待；它与整机U是不同指标。','',
              '| 定序Baseline区间 | 平均长卡/短卡数 | 短类别利用率 | 全部I/O等待 卡·秒 |', '|---|---:|---:|---:|']
            for w in (warm,late):
                short=w['by_role']['short']
                text.append(f'| {label(w["start_ms"],w["end_ms"])} | {w["mean_long_active_cards"]:.2f} / {w["mean_short_active_cards"]:.2f} | {100*short["compute_ms"]/short["active_ms"]:.2f}% | {stall(w)/1000:.3f} |')
            text+=['','后段短类别自身也恢复，不能仅用长类占比增大来解释整机U回升。不过，长短驻留比例、当前短画像和读取相位都在变化，本次扩窗不是将这些变量分别固定的因果消融。','',
              '日志支持的时序解释：固定长卡一直运行相同C，下一层读取又能藏在计算中，发读相位就能持续保持接近。混合卡经过不同长度的短请求后，各自进入下一轮长段的时间错开。循环一遍相同队列，不等于所有卡会同时重新开始。[后续轮次和固定卡的逐层证据](repeated_queues/repeated_mechanism.md)还列出了读取相位、角色数量和短画像组成的变化。','',
              '可支持的结论是：本配方在指定两秒内能造成真实的集中等待；此次更长有限批次不能支持“混合定序持续损失10个百分点”。固定角色在已测10秒内仍维持约9个百分点损失。一次seed7扩展也不足以代表所有到达过程或无限时间的稳态。','']
    else:
        bs=[r for r in (audit or {}).get('runs',[]) if r['strategy']=='baseline' and r['status']=='complete']
        if len(bs)==3:
            text+=['Baseline三组已完成且通过全程逐盘容量审查；Once三组尚待完成。以下仅列已完成的seed7结果，不把它们当成五seed均值。','',
              '| 窗口 | 固定Baseline U | 随机Baseline U | 定序Baseline U | 全卡有请求 |', '|---|---:|---:|---:|---|']
            for a,b in [(2000,4000),(2000,6000),(2000,8000),(2000,10000),(2000,12000),(4000,12000)]:
                ws=[next(w for w in next(r for r in bs if r['mode']==m)['windows'] if (w['start_ms'],w['end_ms'])==(a,b)) for m in ['fixed','random','ordered']]
                text.append(f'| {label(a,b)} | {ws[0]["device_U_percent"]:.2f}% | {ws[1]["device_U_percent"]:.2f}% | {ws[2]["device_U_percent"]:.2f}% | {all(w["all_32_active"] for w in ws)} |')
            text+=['','已完成的Baseline结果表明：混合定序的低利用率主要集中在第一段，固定长短卡的损失持续到扩大后的窗口。[逐层时序证据](repeated_queues/repeated_mechanism.md)。完整策略对照待Once结果，不提前填写。','']
        else:
            text+=['扩展输入的6次仿真正在运行，完成前不推测[2,12)的利用率，也不把旧输入耗尽后的低U当成结果。','']
    text+=['**复核材料**','',
      '[原日志全部重算CSV](existing_windows.csv)、[原日志主分析JSON](existing_windows.json)、[独立重算说明](independent_existing_windows.md)、[扩队列预定计划](repeated_queues/plan.json)。',
      ('[扩展结果CSV](repeated_windows.csv)、[扩展结果JSON](repeated_windows.json)、[逐盘容量及时间账独立审计](audit_repeated.json)、[只读复核记录](review.md)。' if repeated else '新结果完成后另存repeated_windows.json/csv及audit_repeated.json。'),
      '旧实验数值和日志保留；所有新数据位于本window_sensitivity目录。','']
    if cancelled:
        text=[x.replace('Once三组尚待完成。','Once三组已按用户要求终止，未得到完整扩展结果。').replace('完整策略对照待Once结果，不提前填写。','未完成的Once扩展结果不填写，也不补跑。') for x in text]
    (HERE/'report.md').write_text('\n'.join(text))
    plt.rcParams.update({'font.family':FontProperties(fname='/home/chguo/.fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
    fig,ax=plt.subplots(figsize=(11,5.5));ends=[4000,4250,4350,4500,5000,6000,8000]
    for mode,policy,name,color in [('random','baseline','随机 / Baseline','#2a718b'),('ordered','baseline','定序 / Baseline','#cf7228'),('ordered','once','定序 / Once','#638e38')]:
        ax.plot([x/1000 for x in ends],[val(lookup(old,mode,policy,2000,x)) for x in ends],'-o',label=name,color=color,lw=2,ms=4)
    ax.axvspan(done/1000,8,color='#eee',zorder=0);ax.axvline(done/1000,color='#777',ls='--',lw=1)
    ax.text(5.25,97,'灰区开始混入任务耗尽后的空闲',fontsize=10,color='#555')
    ax.set(xlabel='统计窗口终点（秒）；起点固定为2秒',ylabel='32张NPU平均利用率（%）',title='只扩大原日志统计窗口：后期空闲会改变指标含义',xlim=(3.9,8.05),ylim=(50,102))
    ax.legend(loc='lower left');ax.grid(alpha=.15);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'existing_window_sensitivity.{ext}',dpi=170)
    plt.close(fig)
    if repeated:
        fig,ax=plt.subplots(figsize=(11,5.5));ends=[4000,6000,8000,10000,12000]
        for mode,policy,name,color,style in [('random','baseline','混合随机 / Baseline','#2a718b','-'),('ordered','baseline','混合定序 / Baseline','#cf7228','-'),('ordered','once','混合定序 / Once','#638e38','-'),('fixed','baseline','固定长短卡 / Baseline','#8c6499','--')]:
            ax.plot([x/1000 for x in ends],[val(lookup(repeated,mode,policy,2000,x)) for x in ends],style+'o',label=name,color=color,lw=2,ms=4)
        ax.set(xlabel='统计窗口终点（秒）；起点固定为2秒',ylabel='32张NPU平均利用率（%）',title='队列重复三遍后的窗口扩展（seed7，全部卡持续有请求）',xlim=(3.8,12.2),ylim=(85,101))
        ax.legend(loc='best');ax.grid(alpha=.15);fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(HERE/f'repeated_window_sensitivity.{ext}',dpi=170)
        plt.close(fig)
    print(json.dumps({'report':str(HERE/'report.md'),'extended_results_included':bool(repeated)}))

if __name__=='__main__':main()
