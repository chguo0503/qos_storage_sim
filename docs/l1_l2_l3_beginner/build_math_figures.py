#!/usr/bin/env python3
"""Generate three explicitly fictional teaching SVGs; no experiment is run/read."""
from pathlib import Path
from html import escape
import json
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE/"assets"
PURPLE, ORANGE, INK = "#7c3aed", "#df8426", "#17283d"
MUTED, GRID = "#657389", "#dde3eb"


def text(x,y,value,size=22,color=INK,anchor="start",weight="normal"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')


def rect(x,y,w,h,fill,stroke="none",radius=0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')


def line(x1,y1,x2,y2,color=GRID,width=1,dash=None):
    extra=f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{extra}/>')


def document(name,height,title,parts,description):
    assert 0 < height <= 330
    css_class=name.replace("_","-")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" class="{css_class}" width="1000" height="{height}" '
        f'viewBox="0 0 1000 {height}" role="img" aria-labelledby="{name}_title {name}_desc">\n'
        f'<title id="{name}_title">{escape(title)}（教学示意）</title>\n'
        f'<desc id="{name}_desc">{escape(description)} 非仿真或实测结果。</desc>\n'
        f'<style>svg.{css_class} text{{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;}}</style>\n'
        +rect(0,0,1000,height,"white")
        +text(25,31,title,25,weight="bold")+text(974,30,"教学示意",19,MUTED,"end")
        +"\n".join(parts)+"\n</svg>\n")


def build_sum():
    p=[]
    for x,color,label in ((154,PURPLE,"计算"),(283,ORANGE,"等待")):
        p.extend((rect(x,48,19,18,color),text(x+29,64,label,18)))
    x0,width=150,790
    for row,compute in enumerate((8,4)):
        y=82+69*row
        p.extend((text(36,y+30,f"卡{row+1}",24),rect(x0,y,width*compute/10,43,PURPLE),
                  rect(x0+width*compute/10,y,width*(10-compute)/10,43,ORANGE),
                  text(x0+width*compute/20,y+29,f"计算{compute}秒",22,"white","middle"),
                  text(x0+width*(compute+10)/20,y+29,f"等待{10-compute}秒",21,"white","middle")))
    for t in range(0,11,2):
        x=x0+width*t/10
        p.extend((line(x,204,x,211,MUTED),text(x,232,str(t),18,MUTED,"middle")))
    p.extend((text(982,232,"秒",18,MUTED,"end"),rect(25,252,950,52,"#f5f0ff",radius=8),
              text(46,286,"总计算：8+4=12卡·秒",23),
              text(516,286,"整机平均：12÷(2×10)=60%",23,PURPLE)))
    return document("math_sum",320,"把每张卡的计算时间加起来",p,
        "两张卡都观察10秒。卡1计算8秒、等待2秒；卡2计算4秒、等待6秒。计算合计12卡秒，"
        "所有卡可用时间共20卡秒，平均利用率为60%。")


def build_integral():
    p=[]
    x0,x1,y0,scale=105,690,247,14.5
    x=lambda t:x0+(x1-x0)*t/5
    y=lambda rate:y0-scale*rate
    for rate in (0,5,10):
        p.extend((line(x0,y(rate),x1,y(rate),GRID),text(x0-15,y(rate)+6,str(rate),18,MUTED,"end")))
    p.extend((rect(x(0),y(6),x(2)-x(0),y0-y(6),"#b996f7"),
              rect(x(3),y(9),x(5)-x(3),y0-y(9),"#b996f7"),
              line(x0,y(10),x1,y(10),ORANGE,2,"7 5"),
              line(x0,y0,x1,y0,MUTED,1.4),line(x0,85,x0,y0,MUTED,1.4),
              f'<path d="M {x(0)} {y(6)} H {x(2)} V {y0} H {x(3)} V {y(9)} H {x(5)}" '
              f'fill="none" stroke="{PURPLE}" stroke-width="3"/>',
              text(25,74,"速率",18,MUTED),text(25,96,"份/秒",18,MUTED),
              text(175,87,"能力上限：10份/秒",20,ORANGE),
              text(x(1),y(6)-10,"6份/秒",20,PURPLE,"middle"),
              text(x(4),y(9)+31,"9份/秒",20,PURPLE,"middle"),
              text(x(1),212,"12份",25,PURPLE,"middle"),
              text(x(2.5),230,"0份",21,MUTED,"middle"),
              text(x(4),208,"18份",25,PURPLE,"middle")))
    for t in range(6):
        p.extend((line(x(t),y0,x(t),y0+7,MUTED),text(x(t),279,str(t),18,MUTED,"middle")))
    p.extend((text((x0+x1)/2,312,"时间（秒）",21,INK,"middle"),
              rect(745,105,228,169,"#f5f0ff",radius=10),
              text(859,135,"实际交付面积",20,INK,"middle"),
              text(859,169,"12+0+18=30份",21,PURPLE,"middle"),
              line(763,188,955,188,"#d8c9f1"),
              text(859,216,"能力上界",20,INK,"middle"),
              text(859,250,"10×5=50份",23,ORANGE,"middle")))
    return document("math_integral",330,"曲线下的面积，就是实际读出的总量",p,
        "盘能力为每秒10份。0到2秒实际每秒6份，2到3秒为0，3到5秒实际每秒9份。"
        "紫色面积为6乘2加0乘1加9乘2等于30份；能力上界为10乘5等于50份。")


def build_weights():
    # Explicit arrowhead polygons also work in limited SVG/PDF renderers.
    p=[]
    for yy,name,per_layer,rate,total,color,tint in (
            (58,"甲",20,2,40,PURPLE,"#f4edff"),(185,"乙",100,10,200,ORANGE,"#fff2e3")):
        p.extend((rect(25,yy,307,103,tint,color,8),
                  text(178,yy+29,f"画像{name}：r={rate}份/秒",22,color,"middle"),
                  text(178,yy+60,f"每层读{per_layer}份，算10秒",20,INK,"middle"),
                  text(178,yy+89,f"共2层：实际读出{total}份",20,INK,"middle"),
                  f'<path d="M 342 {yy+52} H 374" fill="none" stroke="{MUTED}" stroke-width="2"/>',
                  f'<path d="M 374 {yy+52} L 366 {yy+47} L 366 {yy+57} Z" fill="{MUTED}"/>',
                  rect(384,yy+11,246,81,tint,color,8),
                  text(507,yy+42,f"{total}份 ÷ {rate}份/秒",21,color,"middle"),
                  text(507,yy+74,"=20卡·秒",25,color,"middle")))
    p.extend((f'<path d="M 642 110 H 679 L 710 174 H 735 M 642 237 H 679 L 710 174" '
              f'fill="none" stroke="{MUTED}" stroke-width="2"/>',
              f'<path d="M 735 174 L 727 169 L 727 179 Z" fill="{MUTED}"/>',
              rect(748,124,228,99,"#f1f4f8",GRID,10),
              text(862,152,"总计算",21,INK,"middle"),
              text(862,184,"20+20",25,INK,"middle"),
              text(862,215,"=40卡·秒",26,INK,"middle"),
              text(25,320,"教学示意：先除以各自的速率，再相加；不能给两路都用同一个除数。",20,MUTED)))
    return document("math_weights",330,"不同画像，先各自换算，再相加",p,
        "固定画像甲每层读取20份并计算10秒，速率2份每秒；两层读40份，换算得40除以2等于20卡秒。"
        "画像乙每层读取100份并计算10秒，速率10份每秒；两层读200份，换算得200除以10等于20卡秒。"
        "两路合并计算工作量40卡秒；应使用不同权重1/2和1/10。")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files={}
    for name,build in (("math_sum",build_sum),("math_integral",build_integral),("math_weights",build_weights)):
        svg=build()
        root=ET.fromstring(svg)
        assert int(root.attrib["width"]) == 1000 and int(root.attrib["height"]) <= 330
        path=OUT/(name+".svg")
        path.write_text(svg,encoding="utf-8")
        files[name]=dict(width=int(root.attrib["width"]),height=int(root.attrib["height"]))
    assert 8+4 == 12 and 12/(2*10) == .6
    assert 6*2+0*1+9*2 == 30 and 10*5 == 50
    assert 2*20/2+2*100/10 == 40
    print(json.dumps(dict(no_simulation=True,teaching_only=True,figures=files),ensure_ascii=False))


if __name__=="__main__":
    main()
