#!/usr/bin/env python3
"""Adaptive Admission Scheme B V2.1 —— 独立可运行版本。

这个文件把策略从离散事件仿真里**完全剥离**出来：不 import sim、不 import
continuous_batch_sim，没有任何仿真器状态。它只做一件事：

    输入一个 demand 矩阵  ->  输出一个 grant 矩阵

复制这一个文件就能在任何地方运行策略，也可以直接跑：

    python3 adaptive_policy_standalone.py            # 跑内置的演示场景
    python3 adaptive_policy_standalone.py --selftest # 对拍项目里的正式实现

与正式实现的关系
----------------
本文件把 V2.1 的**决策逻辑**（谁进 floor、走 V1 还是 V2、剩余字节怎么花）
完整重写为可独立阅读的代码，并复用 ``continuous_batch_control`` 里的两个
分配原语。那个模块本身只依赖 math/numpy，不 import 任何仿真器代码，所以
"脱离仿真" 这一点仍然成立，同时保证与正式实现**逐元素一致**。

``--selftest`` 会在随机矩阵上与正式实现对拍。正式实验请继续使用原模块；
本文件用于讲解、移植和快速试参。

依赖：numpy、continuous_batch_control.py（两个文件即可运行）。

配套文档：ADAPTIVE_POLICY_IO.md
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# 只依赖 math/numpy 的分配原语模块，不含任何仿真器代码
from continuous_batch_control import (
    allocate_coflow_grants as _ref_coflow,
    allocate_grants as _ref_equal,
)

_EPS = 1e-12

COFLOW = "v1_coflow_residual"
EXPLICIT = "v2_explicit_selected_spill"


# ---------------------------------------------------------------------------
# 两个底层分配原语
# ---------------------------------------------------------------------------

def _caps(spec, count, name):
    """把标量或向量统一成长度 count 的容量向量。"""
    values = (float(spec),) * count if np.isscalar(spec) else tuple(
        float(v) for v in spec
    )
    if len(values) != count:
        raise ValueError(f"{name} must have {count} entries")
    if any(v < 0.0 or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be finite and non-negative")
    return np.asarray(values, dtype=float)


def allocate_equal(demand, ssd_caps, npu_caps):
    """逐格 max-min：每个 (NPU, SSU) 格子独立被抬高，直到需求满足或容量用尽。

    结果是「需求封顶的等量 max-min 分配」。它**不保证**同一个请求的各个 SSU
    分片进度一致——那正是 coflow 版本要解决的问题。

    直接复用 continuous_batch_control 的实现（该模块只依赖 math/numpy）。
    """
    d = np.asarray(demand, dtype=float)
    if d.size == 0:
        return np.zeros_like(d)
    return np.asarray(
        _ref_equal(d, ssd_caps=tuple(_caps(ssd_caps, d.shape[1], "ssd_caps")),
                   npu_caps=tuple(_caps(npu_caps, d.shape[0], "npu_caps"))),
        dtype=float,
    )


def allocate_coflow(demand, target_ratios, ssd_caps, npu_caps):
    """按 coflow 比例 max-min：每个请求(行)只有**一个标量比例**。

    抬高某行的比例会等比例消耗该行所有 (NPU, SSU) 格子；一旦该行用到的任一
    SSU 或它自己的 NPU 链路饱和，该行就冻结。这保证同一请求的各 SSU 分片
    进度一致——否则请求会卡在最慢的分片上（layer barrier）。

    直接复用 continuous_batch_control 的实现。
    """
    d = np.asarray(demand, dtype=float)
    if d.size == 0:
        return np.zeros_like(d)
    return np.asarray(
        _ref_coflow(d, target_ratios=target_ratios,
                    ssd_caps=tuple(_caps(ssd_caps, d.shape[1], "ssd_caps")),
                    npu_caps=tuple(_caps(npu_caps, d.shape[0], "npu_caps"))),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Stage A：谁进 SLO floor（V1 / V2 完全一致）
# ---------------------------------------------------------------------------

def select_admitted(d, ssd, npu, ratio, required, reserve, pinned):
    """返回 (选中的 NPU 列表, 使用的 target 矩阵, 是否全员入选)。

    先试 "所有人都能拿到 52% floor" -> 再试 "所有人 50%" -> 都不行才做背包。
    背包顺序：先 pinned，再按归一化 SSD 占用从小到大（便宜的先进）。
    """
    preferred = ratio * d
    required_t = required * d

    def fits(t):
        return bool(
            np.all(t.sum(axis=0) <= ssd + _EPS)
            and np.all(t.sum(axis=1) <= npu + _EPS)
        )

    live = np.flatnonzero(d.sum(axis=1) > _EPS)
    if fits(preferred):
        return list(live), preferred, True
    if fits(required_t):
        return list(live), required_t, True

    targets = preferred
    remaining = (1.0 - reserve) * ssd.copy()
    selected, sel_set = [], set()

    def admit(i):
        t = targets[i]
        if d[i].sum() <= _EPS:
            return
        if t.sum() > npu[i] + _EPS:
            return
        if np.any(t > remaining + _EPS):
            return
        remaining[:] -= t
        selected.append(i)
        sel_set.add(i)

    for i in pinned:
        admit(int(i))
    safe = np.maximum(ssd, _EPS)
    cands = []
    for i in range(d.shape[0]):
        if i in sel_set or d[i].sum() <= _EPS:
            continue
        norm = targets[i] / safe
        cands.append((float(norm.sum()), float(norm.max(initial=0.0)), i))
    for _, _, i in sorted(cands):
        admit(i)
    return selected, targets, False


# ---------------------------------------------------------------------------
# Stage B：floor 之后剩下的字节怎么花
# ---------------------------------------------------------------------------

def residual_v1(d, floor, selected, ssd, npu, all_in):
    """V1：剩余容量对**所有**请求做一次统一的残量分配。"""
    rs = np.maximum(0.0, ssd - floor.sum(axis=0))
    rn = np.maximum(0.0, npu - floor.sum(axis=1))
    rd = np.maximum(0.0, d - floor)
    if all_in:
        return floor + allocate_equal(rd, rs, rn)       # 全员达标 -> 逐格填满
    return floor + allocate_coflow(rd, 1.0, rs, rn)     # 过载 -> 按 coflow 推进


def residual_v2(d, floor, selected, ssd, npu, reserve, all_in):
    """V2：三段显式分配 —— 落选者的小额 background，再喂满选中者，最后溢出。"""
    n = d.shape[0]
    sel_set = set(selected)

    rs = np.maximum(0.0, ssd - floor.sum(axis=0))
    rn = np.maximum(0.0, npu - floor.sum(axis=1))
    bg_caps = np.minimum(rs, reserve * ssd)
    rej_d = np.zeros_like(d)
    if not all_in:
        rej = [i for i in range(n) if i not in sel_set]
        if rej:
            rej_d[rej] = np.maximum(0.0, d[rej] - floor[rej])
    background = allocate_coflow(rej_d, 1.0, bg_caps, rn)

    grants = floor + background
    rs = np.maximum(0.0, ssd - grants.sum(axis=0))
    rn = np.maximum(0.0, npu - grants.sum(axis=1))
    sel_res = np.zeros_like(d)
    if selected:
        sel_res[selected] = np.maximum(0.0, d[selected] - grants[selected])
    grants = grants + allocate_equal(sel_res, rs, rn)

    rs = np.maximum(0.0, ssd - grants.sum(axis=0))
    rn = np.maximum(0.0, npu - grants.sum(axis=1))
    return grants + allocate_equal(np.maximum(0.0, d - grants), rs, rn)


# ---------------------------------------------------------------------------
# 顶层：V2.1 选择器
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Allocation:
    grants_gbps: np.ndarray          # (N, S) 每个 NPU 在每块 SSU 上的 CIR
    selected_npu_ids: tuple          # 拿到 SLO floor 的 NPU
    active_npu_count: int            # 有活跃请求的 NPU 数
    selected_fraction: float         # selected / active —— 决策信号
    residual_mode: str               # 实际走了 V1 还是 V2


def allocate(
    demand,
    *,
    explicit_spill_threshold: float = 0.75,
    target_ratio: float = 0.52,
    required_ratio: float = 0.5,
    background_reserve_fraction: float = 0.05,
    pinned_npu_ids: Sequence[int] = (),
    ssd_caps=40.0,
    npu_caps=50.0,
) -> Allocation:
    """输入 demand 矩阵，输出 grant 矩阵。纯函数，无状态。"""
    d = np.asarray(demand, dtype=float)
    if d.size == 0:
        return Allocation(d.reshape(len(demand), 0), (), 0, 1.0, COFLOW)
    if d.ndim != 2:
        raise ValueError("demand must be rectangular")
    if np.any(d < 0.0) or not np.all(np.isfinite(d)):
        raise ValueError("demand must be finite and non-negative")
    thr = float(explicit_spill_threshold)
    if not 0.0 < thr <= 1.0 or not math.isfinite(thr):
        raise ValueError("explicit_spill_threshold must be in (0, 1]")

    n, s = d.shape
    ssd = _caps(ssd_caps, s, "ssd_caps")
    npu = _caps(npu_caps, n, "npu_caps")
    pinned = tuple(dict.fromkeys(int(i) for i in pinned_npu_ids))

    selected, targets, all_in = select_admitted(
        d, ssd, npu, target_ratio, required_ratio,
        background_reserve_fraction, pinned,
    )
    floor = np.zeros_like(d)
    if selected:
        floor[selected] = targets[selected]

    active = int(np.count_nonzero(d.sum(axis=1) > _EPS))
    frac = len(selected) / active if active else 1.0

    if frac < thr:
        grants = residual_v2(d, floor, selected, ssd, npu,
                             background_reserve_fraction, all_in)
        mode = EXPLICIT
    else:
        grants = residual_v1(d, floor, selected, ssd, npu, all_in)
        mode = COFLOW
    return Allocation(grants, tuple(selected), active, float(frac), mode)


# ---------------------------------------------------------------------------
# 演示 / 自检
# ---------------------------------------------------------------------------

CATEGORIES = {                 # (每层 GB, 每层 compute ms) —— 来自真实 data 画像
    "SS": (0.107338, 1.288),
    "SL": (0.106750, 7.729),
    "LS": (0.193024, 6.695),
    "LL": (0.192688, 13.097),
}


def demo_matrix(num_npu, num_ssu, seed=0):
    """构造一个演示用 demand 矩阵：每个 NPU 一个请求，轮转类别。

    简化说明：这里按**单层**构造（每层字节 / 每层 compute）。真实控制器用的是
    「所有剩余未就绪层」的聚合量——分子是剩余总字节，分母是剩余 IO 层数 x 每层
    compute（见 continuous_batch_sim.py:1537）。

    当前项目里层严格同构（sim.py:288，16 层共享同一个 layer_blocks）且请求内预取
    深度为 1（continuous_batch_sim.py:1324），所以两者数值基本等价。差别在于聚合量
    的分子还携带「这个请求总共还剩多少字节」，而 Stage A 的背包排序依赖该信息。
    """
    rng = np.random.default_rng(seed)
    names = list(CATEGORIES)
    d = np.zeros((num_npu, num_ssu))
    cats = []
    for i in range(num_npu):
        c = names[i % len(names)]
        cats.append(c)
        gb, ms = CATEGORIES[c]
        # 该请求这一层的字节按 ring 风格散布到所有 SSU 上
        w = rng.dirichlet(np.full(num_ssu, 40.0)) * gb
        d[i] = w / (ms / 1000.0)
    return d, cats


def _show(num_npu, num_ssu):
    d, cats = demo_matrix(num_npu, num_ssu)
    a = allocate(d)
    g = a.grants_gbps
    print(f"\n=== {num_npu} NPU x {num_ssu} SSU ===")
    print(f"  fleet demand   {d.sum():9.2f} GB/s")
    print(f"  fleet capacity {num_ssu*40:9.2f} GB/s   ({num_ssu*40/d.sum()*100:.1f}%)")
    print(f"  selected       {len(a.selected_npu_ids)}/{a.active_npu_count}"
          f"  fraction={a.selected_fraction:.4f}  ->  {a.residual_mode}")
    print(f"  SSD 用量  max {g.sum(axis=0).max():6.3f} / 40   "
          f"NPU 用量 max {g.sum(axis=1).max():6.3f} / 50")
    by = {}
    for i, c in enumerate(cats):
        by.setdefault(c, []).append(g[i].sum() / d[i].sum())
    print("  各类别拿到的需求满足率：")
    for c in CATEGORIES:
        if c in by:
            v = np.array(by[c])
            print(f"    {c}  n={len(v):3d}  mean={v.mean()*100:6.2f}%  "
                  f"min={v.min()*100:6.2f}%  max={v.max()*100:6.2f}%")


def _selftest():
    """与项目里的正式实现逐元素对拍。"""
    try:
        from adaptive_admission_scheme_b_v2_1 import (
            allocate_adaptive_admission_grants as ref,
        )
    except Exception as exc:                                  # pragma: no cover
        print(f"跳过对拍（无法导入正式实现）：{exc}")
        return 0
    rng = np.random.default_rng(20260830)
    worst, bad = 0.0, 0
    for trial in range(30):
        n = int(rng.integers(4, 40))
        s = int(rng.integers(2, 12))
        d = rng.uniform(0, 6, size=(n, s))
        d[rng.random((n, s)) < 0.3] = 0.0
        mine = allocate(d)
        theirs = ref(d)
        gap = float(np.abs(mine.grants_gbps - np.array(theirs.grants_gbps)).max())
        same_mode = mine.residual_mode == theirs.residual_mode
        same_sel = mine.selected_npu_ids == theirs.selected_npu_ids
        worst = max(worst, gap)
        if gap > 1e-6 or not same_mode or not same_sel:
            bad += 1
            print(f"  trial {trial:2d} {n}x{s}: gap={gap:.2e} "
                  f"mode={'OK' if same_mode else 'DIFF'} "
                  f"sel={'OK' if same_sel else 'DIFF'}")
    print(f"\n对拍 30 组随机矩阵：最大逐元素偏差 {worst:.3e}，不一致 {bad} 组")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true", help="与正式实现对拍")
    p.add_argument("--npu", type=int, default=128)
    p.add_argument("--ssu", type=int, nargs="*", default=[24, 40, 70])
    args = p.parse_args()
    if args.selftest:
        raise SystemExit(1 if _selftest() else 0)
    for s in args.ssu:
        _show(args.npu, s)


if __name__ == "__main__":
    main()
