#!/usr/bin/env python3
"""Plot actual Baseline layer events and admitted-profile V/C, without simulation.

Run from any directory. The default pair is the seed-7 long-768 comparison.
Demand is the current admitted request's per-layer bytes / per-layer compute,
including its compute and stall residency, with no extra cross-request L0 term.
It is not disk throughput, disk busy time, or a deadline-feasibility envelope.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


BASE = Path(__file__).resolve().parent
NPU, SSU = 32, 6
COLORS = {"short": "#26749b", "long": "#72b5a5", "stall": "#efa640"}


def read_json(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def clipped(start, end, left, right):
    a, b = max(start, left), min(end, right)
    return (a, b - a) if b > a else None


def load_case(label, left, right):
    manifest_path = BASE / "inputs" / f"{label}.json.gz"
    matches = sorted((BASE / "runs" / label / "baseline").glob("*.json.gz"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Baseline result for {label}: {matches}")
    result_path = matches[0]
    manifest, result = read_json(manifest_path), read_json(result_path)
    assert manifest["input_fingerprint"] == result["input_fingerprint"]
    assert result["strategy"] == "baseline"
    assert manifest["metadata"]["num_npu"] == NPU
    assert manifest["metadata"]["num_ssu"] == SSU
    assert all(result["summary"]["invariants"].values())
    assert max(abs(r["compute_queue_wait_ms"])
               for r in result["summary"]["request_metrics"]) < 1e-8

    requests = {r["request_id"]: r for r in manifest["requests"]}
    demand_by_id, population = {}, {}
    for rid, request in requests.items():
        load = request["load"]
        layer_c_ms = load["per_layer_us"] / 1000
        placement = manifest["placements"][request["placement_index"]]
        volumes = [[math.fsum(v for s, v in layer if s == disk)
                    for disk in range(SSU)] for layer in placement]
        assert len(placement) in (1, manifest["metadata"]["n_layers"])
        assert all(np.allclose(v, volumes[0], rtol=0, atol=1e-12) for v in volumes)
        assert math.isclose(math.fsum(volumes[0]), load["per_layer_kv_gb"], abs_tol=1e-12)
        demand_by_id[rid] = np.array(volumes[0]) * (1000 / layer_c_ms)
        original_id = load.get("original_request_id", rid)
        identity = (request["npu_id"], original_id)
        stable_load = {k: v for k, v in load.items()
                       if k not in ("request_id", "generation", "original_request_id")}
        population[identity] = {"load": stable_load, "placement": placement,
                                "arrival": request["arrival_time_ms"]}

    spans = [{role: [] for role in COLORS} for _ in range(NPU)]
    active_ms = np.zeros(NPU)
    compute_ms = np.zeros(NPU)
    stall_ms = np.zeros(NPU)
    delta_by_time = defaultdict(lambda: np.zeros(SSU))
    admitted_intervals = defaultdict(list)
    seen = set()
    for batch in result["summary"]["microbatch_metrics"]:
        assert batch["batch_size"] == 1 and len(batch["member_request_ids"]) == 1
        rid = batch["member_request_ids"][0]
        assert rid not in seen
        seen.add(rid)
        request = requests[rid]
        npu = batch["npu_id"]
        assert npu == request["npu_id"], "This figure requires fixed NPU assignment"
        role = request["load"]["role"]
        assert role in ("short", "long")
        admission, completion = batch["admission_time_ms"], batch["completion_time_ms"]
        admitted_intervals[npu].append((admission, completion))
        overlap = clipped(admission, completion, left, right)
        if overlap:
            active_ms[npu] += overlap[1]
        delta_by_time[admission] += demand_by_id[rid]
        delta_by_time[completion] -= demand_by_id[rid]

        previous_end = admission
        for layer in sorted(batch["layer_metrics"], key=lambda x: x["layer"]):
            start, end = layer["compute_start_ms"], layer["compute_end_ms"]
            # Waiting starts at admission or the preceding layer's compute end.
            # In particular, pre-admission L0 read lifetimes are never stall bars.
            wait = start - previous_end
            assert wait >= -1e-8
            assert math.isclose(wait, layer["io_barrier_wait_ms"], abs_tol=1e-7)
            assert math.isclose(end-start, layer["compute_duration_ms"], abs_tol=1e-7)
            overlap = clipped(previous_end, start, left, right)
            if overlap:
                spans[npu]["stall"].append(overlap)
                stall_ms[npu] += overlap[1]
            overlap = clipped(start, end, left, right)
            if overlap:
                spans[npu][role].append(overlap)
                compute_ms[npu] += overlap[1]
            previous_end = end
        assert math.isclose(previous_end, completion, abs_tol=1e-7)

    assert seen == requests.keys()
    for intervals in admitted_intervals.values():
        intervals.sort()
        assert all(a[1] <= b[0] + 1e-8 for a, b in zip(intervals, intervals[1:]))
    assert np.allclose(active_ms, compute_ms + stall_ms, rtol=0, atol=1e-6)
    assert np.allclose(active_ms, right-left, rtol=0, atol=1e-6)
    util = float(compute_ms.sum() / (NPU * (right-left)))
    matching_windows = [w for w in result["windows"]
                        if w["start_ms"] == left and w["end_ms"] == right]
    if matching_windows:
        assert math.isclose(util, matching_windows[0]["mean_npu_utilization"], abs_tol=1e-10)

    # Sum all simultaneous admission/completion changes before evaluating the
    # next half-open interval. No artificial transient at a request handover.
    times, rates = [], []
    current = np.zeros(SSU)
    full_peak = np.zeros(SSU)
    event_times = sorted(delta_by_time)
    for i, time in enumerate(event_times):
        current = current + delta_by_time[time]
        assert current.min() > -1e-8
        current[np.abs(current) < 1e-10] = 0
        if i + 1 < len(event_times):
            end = event_times[i+1]
            full_peak = np.maximum(full_peak, current)
            if end > left and time < right:
                times.append(max(time, left))
                rates.append(current.copy())
    assert times and times[0] == left
    times.append(right)
    rates.append(rates[-1].copy())
    rates = np.array(rates)
    audit = {"label": label, "manifest": str(manifest_path), "result": str(result_path),
             "input_fingerprint": manifest["input_fingerprint"],
             "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
             "window_ms": [left, right], "fleet_compute_utilization": util,
             "all_32_active": True, "window_compute_ms_by_npu": compute_ms.tolist(),
             "window_stall_ms_by_npu": stall_ms.tolist(),
             "full_run_current_profile_peak_by_ssu_gib_s": full_peak.tolist(),
             "window_current_profile_peak_by_ssu_gib_s": rates.max(axis=0).tolist()}
    return {"spans": spans, "times": times, "rates": rates, "population": population,
            "audit": audit, "util": util}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-label", default="concurrency_l768_seed7")
    parser.add_argument("--ordered-label", default="concurrency_l768_seed7__exact_cohort4_p1")
    parser.add_argument("--start", type=float, default=2000)
    parser.add_argument("--end", type=float, default=4000)
    parser.add_argument("--output-stem", default="concurrency_l768_seed7_comparison")
    args = parser.parse_args()
    assert args.end > args.start
    cases = [load_case(label, args.start, args.end)
             for label in (args.random_label, args.ordered_label)]
    assert cases[0]["population"] == cases[1]["population"], "Unmatched request populations"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "baseline-order-study"})
    fig, axes = plt.subplots(2, 2, figsize=(17, 10),
                             gridspec_kw={"height_ratios": [3.25, 1.4]}, sharex=True)
    fig.subplots_adjust(left=.055, right=.985, top=.865, bottom=.17,
                        wspace=.09, hspace=.15)
    drop_pp = (cases[0]["util"] - cases[1]["util"]) * 100
    drop_relative = drop_pp / (cases[0]["util"] * 100) * 100
    fig.suptitle("Same requests, different Baseline queue order  |  32 NPUs / 6 SSUs",
                 fontsize=17, y=.98)
    fig.text(.5, .947,
             f"Device compute utilization: {drop_pp:.3f} percentage-point decrease "
             f"({drop_relative:.3f}% relative); all 32 NPUs active throughout this window",
             ha="center", fontsize=11)
    legend = [Patch(color=COLORS["short"], label="Short compute"),
              Patch(color=COLORS["long"], label="Long compute"),
              Patch(color=COLORS["stall"], label="I/O wait after admission"),
              Patch(color="#eceff1", label="Idle (none in this window)")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=4, frameon=False, columnspacing=2)

    for col, (case, title) in enumerate(zip(cases, ["Random order", "Ordered: four cohorts"])):
        ax, rate_ax = axes[0, col], axes[1, col]
        ax.set_title(f"{title}   |   U = {case['util']*100:.4f}%", fontsize=13, pad=9)
        for npu, spans in enumerate(case["spans"]):
            ax.broken_barh([(args.start, args.end-args.start)], (npu-.41, .82),
                           facecolors="#eceff1", edgecolors="none")
            for role in ("short", "long", "stall"):
                if spans[role]:
                    ax.broken_barh(spans[role], (npu-.41, .82), facecolors=COLORS[role],
                                   edgecolors="none", linewidth=0, antialiased=False,
                                   rasterized=True)
        ax.set_ylim(31.8, -.8)
        ax.set_yticks(range(NPU))
        ax.tick_params(axis="y", labelsize=7, length=2)
        ax.set_ylabel("NPU ID")
        ax.tick_params(axis="x", labelbottom=False)
        # Light cohort separators aid locating physical NPUs without moving bars.
        for border in (7.5, 15.5, 23.5):
            ax.axhline(border, color="#c9cfd2", linewidth=.55, zorder=0)

        palette = plt.get_cmap("tab10").colors
        for ssu in range(SSU):
            rate_ax.step(case["times"], case["rates"][:, ssu], where="post",
                         label=f"SSU {ssu}", color=palette[ssu], linewidth=.8, alpha=.85)
        rate_ax.axhline(40, color="#a52222", linestyle="--", linewidth=1.35,
                        label="40 GiB/s capacity")
        rate_ax.set_ylim(0, 43)
        rate_ax.set_yticks([0, 10, 20, 30, 40])
        rate_ax.set_xlim(args.start, args.end)
        rate_ax.set_xlabel("Simulation time (ms)")
        rate_ax.set_ylabel("Current profile V/C\nper SSU (GiB/s)")
        rate_ax.grid(axis="y", color="#dddddd", linewidth=.5)
        peak = float(case["rates"].max())
        rate_ax.text(.015, .94, f"Window peak: {peak:.3f} GiB/s", transform=rate_ax.transAxes,
                     va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .098),
               ncol=7, frameon=False, fontsize=9, columnspacing=1.25)
    fig.text(.055, .085,
             "Timeline: clipped layer compute intervals and actual I/O barrier waits; "
             "pre-admission waiting is excluded.", fontsize=9)
    fig.text(.055, .063,
             "Demand: sum of admitted requests' per-layer SSD bytes / per-layer compute time. "
             "No extra cross-request L0 term; these curves are not SSD throughput.", fontsize=9)
    fig.text(.055, .041,
             "Profiles: short 1K/128, 1K/256, 1K/384 (constructed); long 192K/768 (interpolated). "
             "Per-NPU quota 25:25:25:1; seed 7.", fontsize=9)
    fig.text(.055, .019,
             "Frozen input fingerprints: " + " / ".join(x["audit"]["input_fingerprint"][:12] for x in cases),
             fontsize=8, color="#555555")
    folder = BASE / "figures"
    folder.mkdir(exist_ok=True)
    stem = folder / args.output_stem
    audit = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "population_and_placement_matched": True, "cases": [x["audit"] for x in cases]}
    fig.savefig(stem.with_suffix(".png"), dpi=240,
                metadata={"Description": json.dumps(audit, sort_keys=True)})
    fig.savefig(stem.with_suffix(".svg"), dpi=240,
                metadata={"Description": json.dumps(audit, sort_keys=True)})
    plt.close(fig)
    print(json.dumps(audit, indent=2))
    print(f"Saved {stem.with_suffix('.png')} and {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
