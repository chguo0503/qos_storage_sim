#!/usr/bin/env python3
"""Standalone timeline of raw compute/stall intervals, with no simulation."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

FONT=Path('/home/chguo/.fonts/msyh.ttc')
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),
 'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','svg.hashsalt':'sustained-mixed-20260911'})
COLORS={'short':'#26749b','long':'#72b5a5','stall':'#efa640'}

def read(p):
    with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return json.load(f)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def collect(manifest,result,left,right):
    assert manifest['input_fingerprint']==result['input_fingerprint']
    assert all(result['summary']['invariants'].values())
    requests={r['request_id']:r for r in manifest['requests']}
    spans=[{k:[] for k in COLORS} for _ in range(32)]
    C=np.zeros(32);active=np.zeros(32);stall=np.zeros(32)
    roles={k:np.zeros(32) for k in ['short','long']}
    batches=[[] for _ in range(32)]
    def overlap(a,b):return max(0.,min(b,right)-max(a,left))
    for b in result['summary']['microbatch_metrics']:
        assert b['batch_size']==1
        req=requests[b['member_request_ids'][0]];n=b['npu_id'];role=req['load']['role']
        assert n==req['npu_id']
        a,z=b['admission_time_ms'],b['completion_time_ms']
        active[n]+=overlap(a,z)
        if overlap(a,z)>0:batches[n].append((a,role))
        previous=a
        for l in b['layer_metrics']:
            cs,ce=l['compute_start_ms'],l['compute_end_ms']
            assert math.isclose(cs-previous,l['io_barrier_wait_ms'],abs_tol=1e-6)
            for key,start,end in [('stall',previous,cs),(role,cs,ce)]:
                d=overlap(start,end)
                if d:
                    spans[n][key].append((max(start,left),d))
                    if key=='stall':stall[n]+=d
                    else:C[n]+=d;roles[role][n]+=d
            previous=ce
    assert np.allclose(active,C+stall,atol=1e-5,rtol=0)
    assert np.allclose(active,right-left,atol=1e-5,rtol=0),'Timeline must not include exhaustion idle'
    changes=[]
    for lane in batches:
        rr=[v for _,v in sorted(lane)]
        changes.append(sum(a!=b for a,b in zip(rr,rr[1:])))
    return spans,dict(U=float(C.sum()/32/(right-left)),C_ms=C.tolist(),stall_ms=stall.tolist(),
       active_ms=active.tolist(),role_C_ms={k:v.tolist() for k,v in roles.items()},
       min_role_C_fraction=float(min(np.min(v/C) for v in roles.values())),role_switches=changes,
       all_active=True,both_roles_all_cards=all(np.all(v>0) for v in roles.values()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start',type=float,default=2);p.add_argument('--end',type=float,default=30)
    p.add_argument('--caption',default='周期混合输入');args=p.parse_args()
    man,raw=read(args.manifest),read(args.result);left,right=args.start*1000,args.end*1000
    spans,audit=collect(man,raw,left,right)
    profiles=man['metadata']['profiles'];short=sorted({r['load']['seq_len_k'] for r in man['requests'] if r['load']['role']=='short'})
    long=sorted({r['load']['seq_len_k'] for r in man['requests'] if r['load']['role']=='long'})
    policy={'baseline':'Baseline','once':'Once per layer'}.get(raw['strategy'],raw['strategy'])
    fig,ax=plt.subplots(figsize=(16,11.3))
    fig.subplots_adjust(left=.067,right=.985,top=.773,bottom=.193)
    fig.suptitle(f'{args.caption} · {policy}：长时间计算与等待',fontsize=23,y=.973)
    fig.text(.5,.919,f'32 NPU / 6 SSU · 每盘 40 GiB/s · [{args.start:g}, {args.end:g}) 秒 · 平均利用率 {audit["U"]*100:.4f}%',ha='center',fontsize=16)
    fig.text(.5,.875,'短请求总输入 '+ ' / '.join(f'{v}K' for v in short)+'；长请求 '+ ' / '.join(f'{v}K' for v in long)+'；每请求 8 层',ha='center',fontsize=14)
    fig.text(.5,.838,'真实仿真时序；队列按各卡完成进度推进，未移动、拼接时间或插入同步屏障',ha='center',fontsize=12.5,color='#444444')
    fig.legend(handles=[Patch(color=COLORS[k],label=v) for k,v in [('short','短请求计算'),('long','长请求计算'),('stall','接纳后 I/O 等待')]],loc='upper center',bbox_to_anchor=(.5,.819),ncol=3,frameon=False)
    for n in range(32):
        for key in COLORS:
            ax.broken_barh(spans[n][key],(n-.4,.8),facecolors=COLORS[key],linewidth=0,edgecolors='none')
    ax.set(xlim=(left,right),ylim=(31.8,-.8),yticks=range(32),ylabel='NPU 编号',xlabel='仿真时间（秒）')
    ax.tick_params(axis='y',labelsize=10);ax.grid(axis='x',alpha=.16)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1000:g}'))
    fig.text(.067,.135,'全部 32 卡全窗有任务；橙色为已接纳任务等待 I/O，不含到达后等待接纳。',fontsize=12)
    fig.text(.067,.098,f'每卡两类都有计算：{audit["both_roles_all_cards"]}；任一卡任一类最低计算占比 {audit["min_role_C_fraction"]*100:.2f}%；最少切换 {min(audit["role_switches"])} 次。',fontsize=12)
    fig.text(.067,.061,'C 与读取量取自原始 data；人工安排的请求序列。逐盘名义 D/C 不另加下一请求首层预取。',fontsize=11.5)
    fig.text(.067,.025,f'seed {man["metadata"]["seed"]} · 输入指纹 {man["input_fingerprint"][:16]} · 数值为该次运行；非真实硬件测量',fontsize=10.5,color='#59636C')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for ext in ['png','pdf','svg']:
        out=args.out.with_suffix('.'+ext);fig.savefig(out,dpi=175);files.append(dict(path=str(out),sha256=sha(out)))
    plt.close(fig)
    audit.update(manifest=str(args.manifest),result=str(args.result),manifest_sha256=sha(args.manifest),
                 result_sha256=sha(args.result),script_sha256=sha(__file__),window_ms=[left,right],files=files)
    args.out.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(args.out),'U':audit['U'],'min_role_C_fraction':audit['min_role_C_fraction'],
                      'minimum_switches':min(audit['role_switches'])},ensure_ascii=False))

if __name__=='__main__':main()
