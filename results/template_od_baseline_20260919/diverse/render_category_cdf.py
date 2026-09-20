#!/usr/bin/env python3
"""Split the audited full24 warm-admission CDFs by their original SS/LS/SL/LL tags.

No simulator runs, original reports, or input files are changed. Each conditional
ECDF is calculated per seed before the three seed ECDFs are averaged equally.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures/full_category_cdf"
SEEDS = (7, 19, 43)
CLASSES = ("SS", "LS", "SL", "LL")
POLICIES = ("baseline", "od_baseline", "once")
NAMES = {"baseline": "ASU Baseline", "od_baseline": "OD Baseline",
         "once": "流量分配（Once）"}
FILES = {"baseline": "asu_baseline", "od_baseline": "od_baseline", "once": "once"}
CLASS_COLORS = dict(zip(CLASSES, ("#0072B2", "#D55E00", "#009E73", "#AA4499")))
CLASS_STYLES = dict(zip(CLASSES, ("-", "--", "-.", ":")))
POLICY_STYLES = {"baseline": ("#4b5563", "--"), "od_baseline": ("#d45e00", "-."),
                 "once": ("#2474b7", "-")}
DESCRIPTIONS = {"SS": "总输入较短 · miss 少", "LS": "总输入较长 · miss 少",
                "SL": "总输入较短 · miss 多", "LL": "总输入较长 · miss 多"}
SOURCES = {}


def tracked(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(HERE))] = hashlib.sha256(data).hexdigest()
    return data


def read_csv(path):
    return list(csv.DictReader(tracked(path).decode("utf-8-sig").splitlines()))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category(load):
    return ("S" if load["seq_len_k"] <= 80 else "L") + ("L" if load["nql"] >= 512 else "S")


def load_samples():
    manifests = {seed: json.loads(gzip.decompress(tracked(HERE / f"inputs/full_seed{seed}.json.gz")))
                 for seed in SEEDS}
    lookups = {}
    for seed, manifest in manifests.items():
        m = manifest["metadata"]
        assert (m["num_npu"], m["num_ssu"], m["n_layers"]) == (32, 3, 8)
        for q in manifest["requests"]:
            assert q["load"]["category"] == category(q["load"])
            lookups[seed, q["request_id"]] = q
    rows = []
    identities = set()
    for raw in read_csv(HERE / "plot_request_samples.csv"):
        if raw["regime"] != "full":
            continue
        row = dict(raw)
        for key in ("seed", "request_id", "npu_id"):
            row[key] = int(row[key])
        for key in ("admission_ms", "completion_ms", "ideal_ms", "ratio", "raw_ratio"):
            row[key] = float(row[key])
        q = lookups[row["seed"], row["request_id"]]
        load = q["load"]
        assert q["npu_id"] == row["npu_id"]
        assert 2000 <= row["admission_ms"] < 4000
        assert abs(row["ideal_ms"] - 8 * load["per_layer_us"] / 1000) < 1e-8
        passed = row["completion_ms"] - row["admission_ms"] <= 1.5 * row["ideal_ms"] + 1e-9
        assert passed == (row["ratio"] <= 1.5)
        assert passed == (row["passed"] == "True")
        row.update(category=category(load), seq_len_k=load["seq_len_k"], miss_tokens=load["nql"])
        identity = (row["policy"], row["seed"], row["request_id"])
        assert identity not in identities
        identities.add(identity)
        rows.append(row)
    return rows


def ecdf(values, x):
    return np.searchsorted(values, x, side="right") / len(values)


def macro(arrays, x):
    return np.mean([ecdf(a, x) for a in arrays], axis=0)


def summary(arrays):
    return {"count_total": sum(map(len, arrays)),
            **{f"count_seed{seed}": len(a) for seed, a in zip(SEEDS, arrays)},
            "slo_1p5_percent": float(macro(arrays, np.array([1.5]))[0] * 100),
            "ratio_min": float(min(a[0] for a in arrays)),
            "ratio_max": float(max(a[-1] for a in arrays))}


def base_plot(title, ymax_x, gap):
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.subplots_adjust(left=.09, right=.73, bottom=.235, top=.80)
    fig.suptitle(title, x=.09, y=.965, ha="left", fontsize=19, fontweight="bold")
    fig.text(.09, .909, "持续过载 full24 · random · 32 NPU / 3 SSU × 40 GiB/s", fontsize=12, color="#475569")
    ax.set_xlim(.9, ymax_x)
    ax.set_ylim(0, 1.035)
    candidates = [1, 1.5, 2, 3, 4, 5, 6, 7] if ymax_x < 9 else [1, 1.5, 4, 6, 8, 10, 12, 14, 16, 18]
    ticks = [x for x in candidates if x <= ymax_x]
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("本类别中，TTFT 倍率不超过横轴值的请求比例", labelpad=12)
    ax.set_xlabel("归一化 TTFT = 接纳至 prefill 完成耗时 / 纯计算耗时（倍，越小越好）", labelpad=14)
    ax.grid(axis="y", color="#e2e8f0", zorder=0)
    ax.spines[["right", "top"]].set_visible(False)
    ax.axvspan(gap[0], gap[1], color="#64748b", alpha=.10, zorder=0)
    ax.axvline(1.5, color="#b91c1c", linewidth=1.3, linestyle=(0, (2, 3)), zorder=1)
    ax.text(1.5, 1.015, "1.5", color="#b91c1c", ha="center", va="bottom", fontsize=10,
            transform=ax.get_xaxis_transform())
    fig.text(.755, .805, "图例中的达标率：SLO × 1.5\nn：三个种子的样本数之和", fontsize=11, color="#475569")
    fig.text(.09, .128, "分类：第1位 = 总输入长度（S ≤ 80K，L > 80K）；第2位 = miss 数（S < 512，L ≥ 512）。", fontsize=10.5)
    fig.text(.09, .093, "本批：短输入 32/64/80K，长输入 128/160/200K；少 miss = 256，多 miss = 1024/2048/4096。", fontsize=10.5)
    fig.text(.09, .058, "统计：warm [2,4) 秒接纳的请求，跟踪至完成；种子 7/19/43 的类别 CDF 等权平均。", fontsize=10.5)
    fig.text(.09, .023, "灰带：OD 整体 CDF 没有样本的倍率区间（约 1.81–3.59）；它不是一段仿真时间。", fontsize=10.5, color="#475569")
    return fig, ax


def draw(ax, arrays, color, style, label, xmax, width=2.3):
    x = np.unique(np.concatenate(([.9, 1.5, xmax], *arrays)))
    y = macro(arrays, x)
    assert np.all(np.diff(y) >= -1e-14) and y[-1] == 1
    ax.step(x, y, where="post", color=color, linestyle=style, linewidth=width, label=label)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = Path("/home/chguo/.fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False, "savefig.dpi": 200})
    rows = load_samples()
    groups = {(p, c): [np.sort([q["ratio"] for q in rows if q["policy"] == p and q["seed"] == s
                                and (c == "ALL" or q["category"] == c)]) for s in SEEDS]
              for p in POLICIES for c in (*CLASSES, "ALL")}
    assert all(len(a) for arrays in groups.values() for a in arrays)
    stats = {(p, c): summary(arrays) for (p, c), arrays in groups.items()}
    references = read_csv(HERE / "plot_cdf_points.csv")
    original_errors = {}
    for p in POLICIES:
        reference = [r for r in references if r["regime"] == "full" and r["policy"] == p]
        x = np.array([float(r["ratio"]) for r in reference])
        expected = np.array([float(r["macro_cdf"]) for r in reference])
        error = float(np.max(np.abs(macro(groups[p, "ALL"], x) - expected)))
        assert error < 1e-12
        reconstructed = np.mean([sum(len(groups[p, c][i]) * ecdf(groups[p, c][i], x) for c in CLASSES)
                                 / len(groups[p, "ALL"][i]) for i in range(3)], axis=0)
        assert np.max(np.abs(reconstructed - expected)) < 1e-12
        original_errors[p] = error
    od_fast = np.concatenate([*groups["od_baseline", "SL"], *groups["od_baseline", "LL"]])
    od_slow = np.concatenate([*groups["od_baseline", "SS"], *groups["od_baseline", "LS"]])
    gap = float(od_fast.max()), float(od_slow.min())
    assert gap[0] < gap[1]
    files = []
    for p in POLICIES:
        xmax = np.ceil(stats[p, "ALL"]["ratio_max"] * 1.04 * 2) / 2
        fig, ax = base_plot(f"{NAMES[p]}：四类请求的 TTFT CDF", xmax, gap)
        for c in CLASSES:
            s = stats[p, c]
            label = f"{c}  {DESCRIPTIONS[c]}\nn={s['count_total']} · 达标 {s['slo_1p5_percent']:.2f}%"
            draw(ax, groups[p, c], CLASS_COLORS[c], CLASS_STYLES[c], label, xmax)
        s = stats[p, "ALL"]
        draw(ax, groups[p, "ALL"], "#202020", (0, (1, 2)),
             f"整体（原图曲线）\nn={s['count_total']} · 达标 {s['slo_1p5_percent']:.2f}%", xmax, 2.0)
        ax.legend(loc="upper left", bbox_to_anchor=(1.025, .96), borderaxespad=0,
                  frameon=False, fontsize=11, labelspacing=1.5, handlelength=2.8)
        fig.text(.09, .855, "彩线各以本类别为 100%；黑线以全部请求为 100%。曲线越靠左，倍率越低。", fontsize=11)
        name = f"full_{FILES[p]}_ss_ls_sl_ll_cdf.png"
        fig.savefig(OUT / name)
        plt.close(fig)
        files.append(name)
    for c in CLASSES:
        xmax = np.ceil(max(stats[p, c]["ratio_max"] for p in POLICIES) * 1.04 * 2) / 2
        fig, ax = base_plot(f"{c}（{DESCRIPTIONS[c]}）：三种策略的 TTFT CDF", xmax, gap)
        fig.text(.09, .855, "每条曲线各以该策略的本类别样本为 100%；完整尾部均已显示。", fontsize=11)
        for p in POLICIES:
            s = stats[p, c]
            label = f"{NAMES[p]}\nn={s['count_total']} · 达标 {s['slo_1p5_percent']:.2f}%"
            draw(ax, groups[p, c], *POLICY_STYLES[p], label, xmax)
        ax.legend(loc="upper left", bbox_to_anchor=(1.025, .96), borderaxespad=0,
                  frameon=False, fontsize=11, labelspacing=1.6, handlelength=2.8)
        name = f"full_{c.lower()}_three_strategies_cdf.png"
        fig.savefig(OUT / name)
        plt.close(fig)
        files.append(name)
    write_csv("category_request_samples.csv", rows)
    write_csv("category_summary.csv", [dict(policy=p, category=c, **stats[p, c])
                                       for p in POLICIES for c in (*CLASSES, "ALL")])
    points = []
    for (p, c), arrays in groups.items():
        x = np.unique(np.concatenate(([.9, 1.5], *arrays)))
        points.extend(dict(policy=p, category=c, ratio=float(a), macro_cdf=float(b))
                      for a, b in zip(x, macro(arrays, x)))
    write_csv("category_cdf_points.csv", points)
    checks = {"sources_sha256": SOURCES, "regime": "full", "seeds": SEEDS,
              "cohort": "admission in [2000,4000) ms; followed to completion",
              "ratio_definition": "(prefill completion - admission) / pure compute duration",
              "macro": "equal mean of three per-seed conditional category ECDFs",
              "category_definition": "first S: total <=80K; second S: miss <512",
              "original_overall_cdf_max_error": original_errors,
              "original_overall_reconstructed_from_categories": True,
              "od_open_gap": gap, "sample_count": len(rows), "figures": files,
              "note": "The same full inputs are used, but warm admission cohorts can differ by policy."}
    (OUT / "render_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
