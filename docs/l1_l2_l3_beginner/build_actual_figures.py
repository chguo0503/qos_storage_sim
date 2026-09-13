#!/usr/bin/env python3
"""Draw four beginner figures from existing four-SSD layer logs; no simulation.

SVG embeds glyph outlines so Chinese labels do not depend on the PDF reader's
fonts. PNG is rendered from the same Matplotlib artists. No block trace is read.
"""
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import tempfile

# Keep Matplotlib's disposable font cache outside the project.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"qos_beginner_actual_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = ROOT/"results/baseline_ab128_32_ratio12_20260912"
OUT = HERE/"assets"
A, B, RED, DARK = "#2563eb", "#14846b", "#df5748", "#17283d"
WAIT, GRID, MUTED = "#f9d9d4", "#dfe5ec", "#617084"
FONT_PATH = Path("/home/chguo/.fonts/msyh.ttc")
if not FONT_PATH.exists():
    FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT = FontProperties(fname=str(FONT_PATH))
plt.rcParams.update({"svg.fonttype":"path", "axes.unicode_minus":False})
SOURCES, EVIDENCE = {}, {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt") as stream:
        return json.load(stream)


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8), (a,b)


def setup(height):
    fig = plt.figure(figsize=(16,height/100),dpi=100,facecolor="white")
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,1600),ylim=(height,0))
    ax.axis("off")
    return fig,ax


def text(ax,x,y,value,size=24,color=DARK,ha="left",va="center",**kwargs):
    return ax.text(x,y,value,fontproperties=FONT,fontsize=size,color=color,
                   ha=ha,va=va,**kwargs)


def rect(ax,x,y,width,height,color,edge="none",lw=0.0,**kwargs):
    ax.add_patch(Rectangle((x,y),width,height,facecolor=color,edgecolor=edge,linewidth=lw,**kwargs))


def line(ax,x0,y0,x1,y1,color=GRID,width=1,**kwargs):
    ax.plot([x0,x1],[y0,y1],color=color,linewidth=width,**kwargs)


def save(fig,name,height,details):
    # Fail visibly on cropped Chinese labels instead of silently saving them.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for artist in fig.axes[0].texts:
        box = artist.get_window_extent(renderer)
        bounds.append([float(v) for v in box.bounds])
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= 1601 and box.y1 <= height+1, (name,artist.get_text(),box.bounds)
    metadata = {"Description": json.dumps(dict(no_simulation=True,sources=SOURCES,details=details),ensure_ascii=False)}
    fig.savefig(OUT/(name+".svg"),format="svg",metadata=metadata)
    fig.savefig(OUT/(name+".png"),format="png",dpi=100)
    plt.close(fig)
    EVIDENCE[name] = dict(png_pixels=[1600,height],svg_viewbox_points=[0,0,1152,height*.72],
        label_bboxes_inside_canvas=True,details=details,
        files={suffix:dict(path=str((OUT/(name+"."+suffix)).relative_to(ROOT)),
                          sha256=sha(OUT/(name+"."+suffix))) for suffix in ("svg","png")})


def case(order):
    folder = STUDY/"validation20s/runs"/f"ssu4_{order}_k1_sync_seed7"/"baseline"
    m = load(folder/"manifest.json.gz")
    raw = load(folder/"result.json.gz")
    assert raw["strategy"] == "baseline" and raw["submit_seed"] == 7
    assert raw["summary"]["num_ssu"] == 4 and raw["summary"]["num_npu"] == 32
    assert m["input_fingerprint"] == raw["input_fingerprint"]
    assert all(raw["summary"]["invariants"].values())
    info = {r["request_id"]:r for r in m["requests"]}
    batches = {b["member_request_ids"][0]:b for b in raw["summary"]["microbatch_metrics"]}
    layers = []
    for rid,batch in batches.items():
        assert len(batch["member_request_ids"]) == 1
        role = info[rid]["load"]["role"]
        previous = batch["admission_time_ms"]
        for l in batch["layer_metrics"]:
            assert previous <= l["compute_start_ms"]+1e-8
            layers.append(dict(request_id=rid,npu=batch["npu_id"],role=role,
                wait_start_ms=previous,**l))
            previous = l["compute_end_ms"]
    return m,raw,batches,layers


def timeline(order,raw,layers):
    height=630
    fig,ax=setup(height)
    order_cn = "Random（随机顺序）" if order == "random" else "Ordered（A、B、B重复）"
    text(ax,55,40,"4盘 · Baseline · "+order_cn,23)
    text(ax,1540,40,"4卡节选，不等于整机利用率",22,ha="right")
    for x,c,label in ((190,A,"A计算"),(480,B,"B计算"),(770,WAIT,"等数据（IO stall）")):
        rect(ax,x,86,32,27,c,RED if c == WAIT else "none",.7)
        text(ax,x+48,101,label,23)
    left,right=2000.,2400.
    x0,x1=185.,1535.
    x=lambda t:x0+(t-left)/(right-left)*(x1-x0)
    cards=[0,8,16,24]
    for tick in (2000,2100,2200,2300,2400):
        line(ax,x(tick),144,x(tick),486,GRID,.9)
    segments=[]
    rows=[]
    for row,npu in enumerate(cards):
        y=153+86*row
        text(ax,42,y+28,f"NPU {npu}",24)
        rect(ax,x0,y,x1-x0,56,"#f4f6f9")
        c=stall=0.
        for l in layers:
            if l["npu"] != npu:
                continue
            for kind,a,z,color in (("compute",l["compute_start_ms"],l["compute_end_ms"],A if l["role"]=="A" else B),
                                   ("IO_stall",l["wait_start_ms"],l["compute_start_ms"],WAIT)):
                a,z=max(a,left),min(z,right)
                if z<=a:
                    continue
                rect(ax,x(a),y,x(z)-x(a),56,color)
                segments.append(dict(npu=npu,request_id=l["request_id"],layer=l["layer"],
                                     role=l["role"],kind=kind,start_ms=a,end_ms=z))
                if kind=="compute": c+=z-a
                else: stall+=z-a
        close(c+stall,right-left)
        rows.append(dict(npu=npu,compute_ms=c,io_stall_ms=stall,idle_ms=(right-left)-c-stall))
    for tick in (2000,2100,2200,2300,2400):
        line(ax,x(tick),498,x(tick),507,MUTED,1)
        text(ax,x(tick),535,f"{tick/1000:.2f}",23,ha="center")
    text(ax,(x0+x1)/2,583,"仿真绝对时间（秒）",23,ha="center")
    full_C=sum(max(0.,min(right,l["compute_end_ms"])-max(left,l["compute_start_ms"])) for l in layers)
    save(fig,"actual_"+order+"_timeline",height,dict(num_ssu=4,num_npu=32,seed=7,strategy="baseline",
        order=order,absolute_window_ms=[left,right],shown_npus=cards,rows=rows,segments=segments,
        separately_computed_32NPU_window_U_percent=100*full_C/(32*(right-left)),
        accounting="Only post-admission waiting intervals: admission/previous compute end to compute start"))


def local_A(batches,handoff):
    evidence=next(r for r in handoff["A_local"]["per_npu_handoffs"] if r["npu"]==0)
    current=batches[9]["layer_metrics"][1]
    following=batches[9]["layer_metrics"][2]
    start,end=2180.093035688657,2214.400897016782
    deadline=current["compute_end_ms"]
    for a,b in ((start,current["compute_start_ms"]),(start,following["io_start_time_ms"]),
                (end,following["io_ready_time_ms"]),(end,following["compute_start_ms"]),
                (start,evidence["release_ms"]),(end,evidence["ready_ms"]),(deadline,evidence["deadline_ms"])):
        close(a,b)
    C,wait,total=deadline-start,end-deadline,end-start
    close(wait,following["io_barrier_wait_ms"])
    height=500
    fig,ax=setup(height)
    text(ax,55,40,"4盘 Ordered · NPU 0 · A请求 9",23)
    text(ax,55,83,f"真实绝对时间：{start:.3f}–{end:.3f} ms",21,color=MUTED)
    x0,x1=215.,1520.
    x=lambda dt:x0+dt/total*(x1-x0)
    text(ax,42,205,"卡上实际\n执行",23)
    rect(ax,x0,150,x(C)-x0,110,A)
    rect(ax,x(C),150,x1-x(C),110,WAIT)
    text(ax,(x0+x(C))/2,205,f"第2层计算\n{C:.3f} ms",22,color="white",ha="center")
    text(ax,(x(C)+x1)/2,205,f"等第3层数据：{wait:.3f} ms",27,color=RED,ha="center")
    for xx in (x0,x1):
        line(ax,xx,282,xx,300,MUTED,1.3)
    line(ax,x0,293,x1,293,MUTED,1.3)
    text(ax,(x0+x1)/2,327,f"这一轮：{C:.3f} + {wait:.3f} = {total:.3f} ms",25,ha="center")
    for dt in (0,C,total):
        line(ax,x(dt),367,x(dt),377,MUTED,1)
        text(ax,x(dt),399,"0" if dt==0 else f"{dt:.3f}",21,
             ha="right" if dt==total else "center")
    text(ax,(x0+x1)/2,442,"从本次读取发出起经过的时间（ms）",22,ha="center")
    text(ax,55,480,"第2层=日志L1，第3层=日志L2。这里只解释一个真实局部周期。",20,color=MUTED)
    save(fig,"local_A_cycle",height,dict(request_id=9,npu=0,current_compute_layer=1,next_read_layer=2,
        absolute_window_ms=[start,end],compute_ms=C,stall_ms=wait,cycle_ms=total,
        compute_layer_log=current,read_layer_log=following,handoff_evidence=evidence))


def b_handoff(batches,handoff):
    h=handoff["B_first_layer"]
    assert h["request_id"]==15000010 and h["preceding_A_request_id"]==15000009
    a=batches[15000009]["layer_metrics"][7]
    b=batches[15000010]["layer_metrics"][0]
    release,deadline,ready=(h[k] for k in ("release_ms","deadline_ms","ready_ms"))
    for x,y in ((release,a["compute_start_ms"]),(deadline,a["compute_end_ms"]),
                (release,b["io_start_time_ms"]),(ready,b["io_ready_time_ms"]),
                (ready,b["compute_start_ms"]),(deadline,batches[15000010]["admission_time_ms"])):
        close(x,y)
    C,stall,R,BC=deadline-release,ready-deadline,ready-release,b["compute_end_ms"]-ready
    close(stall,b["io_barrier_wait_ms"])
    height=655
    fig,ax=setup(height)
    text(ax,55,40,"4盘 Ordered · NPU 15 · A15000009 → B15000010",23)
    text(ax,55,94,"下一条B的首层读取，在前驱A的末层计算开始时提前发出",22,color=MUTED)
    x0,x1=215.,1515.
    duration=b["compute_end_ms"]-release
    x=lambda dt:x0+dt/duration*(x1-x0)
    text(ax,42,214,"卡上实际\n执行",23)
    rect(ax,x0,160,x(C)-x0,110,A)
    rect(ax,x(C),160,x(R)-x(C),110,WAIT)
    rect(ax,x(R),160,x1-x(R),110,B)
    text(ax,(x0+x(C))/2,215,f"A计算\n{C:.3f}",20,color="white",ha="center")
    text(ax,(x(C)+x(R))/2,215,f"等B首层数据\n{stall:.3f} ms",26,color=RED,ha="center")
    text(ax,(x(R)+x1)/2,215,f"B第1层开始计算\n{BC:.3f} ms",25,color="white",ha="center")
    for dt in (0,C,R):
        line(ax,x(dt),143,x(dt),408,MUTED,.9,linestyle=(0,(4,4)),alpha=.65)
    text(ax,42,365,"B首层\n读取",23)
    rect(ax,x0,333,x(R)-x0,64,"#eff3f7","#9aaabd",.8)
    text(ax,(x0+x(R))/2,366,f"读取生命周期：{R:.3f} ms",24,ha="center")
    text(ax,215,443,"包含排队与传输，不表示SSD一直在服务这条读取。",22,color=MUTED)
    for dt in (0,C,R,duration):
        line(ax,x(dt),488,x(dt),497,MUTED,1)
        text(ax,x(dt),525,"0" if dt==0 else f"{dt:.3f}",21,
             ha="right" if dt==duration else "center")
    text(ax,(x0+x1)/2,574,f"从 {release:.3f} ms 起经过的时间（ms）",22,ha="center")
    text(ax,55,627,"A末层=日志L7；B首层=日志L0。这是A→B交接的局部示例。",20,color=MUTED)
    save(fig,"b_first_layer_handoff",height,dict(npu=15,preceding_A_request_id=15000009,B_request_id=15000010,
        absolute_window_ms=[release,b["compute_end_ms"]],release_ms=release,deadline_ms=deadline,ready_ms=ready,
        preceding_A_compute_ms=C,B_L0_stall_ms=stall,B_L0_read_lifetime_ms=R,B_L0_compute_ms=BC,
        preceding_A_layer_log=a,B_first_layer_log=b,
        scope="Layer/request log reproduction only. Read lifetime includes queueing and transfer, not SSD service time."))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    h=load(STUDY/"formula_review/handoff_checks.json")
    assert h["technical_passed"]
    ordered=None
    for order in ("random","ordered"):
        m,raw,batches,layers=case(order)
        if order=="ordered":
            for key in ("manifest","result"):
                source=h["sources"][key]
                assert sha(source["path"])==source["sha256"]
            ordered=batches
        timeline(order,raw,layers)
    local_A(ordered,h)
    b_handoff(ordered,h)
    output=dict(passed=True,no_new_simulation=True,no_block_trace_read=True,
        script_sha256=sha(__file__),font_file=str(FONT_PATH),sources=SOURCES,
        colors=dict(A_compute=A,B_compute=B,stall_outline=RED,stall_fill=WAIT,text=DARK),figures=EVIDENCE)
    (OUT/"actual_figures_audit.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({name:value["png_pixels"] for name,value in EVIDENCE.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
