#!/usr/bin/env python3
"""Show the actual varied request sequence, not a fixed repeating profile."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    study = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=study / "mixed_varied/inputs/raw_size_varied_rho097_seed7.json.gz")
    args = parser.parse_args()
    with gzip.open(args.manifest, "rt") as source:
        data = json.load(source)
    profiles = sorted({(r["load"]["seq_len_k"], r["load"]["nql"]) for r in data["requests"]})
    assert len(profiles) == 5
    lanes = []
    counts = Counter()
    compute = Counter()
    for n in range(32):
        lane = sorted((r for r in data["requests"] if r["npu_id"] == n), key=lambda r: r["load"]["generation"])
        sequence = []
        for request in lane:
            p = (request["load"]["seq_len_k"], request["load"]["nql"])
            sequence.append(profiles.index(p))
            counts[p] += 1
            compute[p] += 8 * request["load"]["per_layer_us"] / 1000
        assert set(sequence) == set(range(5))
        lanes.append(sequence)
    assert len({len(s) for s in lanes}) == 1
    assert len({tuple(s) for s in lanes}) == 32
    colors = ["#88CCEE", "#44AA99", "#117733", "#DDCC77", "#CC6677"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, (ax, lower) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [5, 1.15], "hspace": .40})
    ax.imshow(np.asarray(lanes), aspect="auto", interpolation="nearest", cmap=ListedColormap(colors), vmin=-.5, vmax=4.5)
    ax.set_yticks(range(32)); ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Original / fixed execution NPU")
    ax.set_xlabel("Request position in this NPU's actual input queue (all 92 requests)")
    ax.set_title("Every NPU mixes five profiles; the complete population is independently shuffled", pad=38)
    ax.legend(handles=[Patch(facecolor=c, label=f"{p[0]}K / NQL{p[1]}") for p, c in zip(profiles, colors)],
              loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=5, frameon=False)
    for y, source in [(1, counts), (0, compute)]:
        left = 0
        total = sum(source.values())
        for p, color in zip(profiles, colors):
            width = 100 * source[p] / total
            lower.barh(y, width, left=left, color=color, height=.55)
            if width >= 5:
                lower.text(left + width / 2, y, f"{width:.1f}%", ha="center", va="center", fontsize=8,
                           color="white" if color == "#117733" else "#111111")
            left += width
    lower.set_yticks([1, 0], ["Request count", "Pure compute time"])
    lower.set_xlim(0, 100); lower.set_xlabel("Share of the complete identical input population (%)")
    lower.spines[["top", "right"]].set_visible(False)
    m = data["metadata"]
    fig.text(.5, .018, f"Raw data profiles; no C scaling. Short-profile compute share = {100*m['short_compute_fraction_per_npu']:.2f}%; hottest SSD demand / capacity = {m['input_demand']['hottest_ssu_load_ratio']:.4f}. Synthetic quotas and all-t=0 arrivals.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.90, bottom=.10, left=.14, right=.985)
    target = Path(__file__).resolve().parent / "05_varied_input_sequences"
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(target.with_suffix("." + suffix), dpi=200, facecolor="white")
    plt.close(fig)
    audit = {"input_fingerprint": data["input_fingerprint"], "source": str(args.manifest),
             "source_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "npu_count": len(lanes), "requests_per_npu": len(lanes[0]), "unique_complete_sequences": len({tuple(s) for s in lanes}),
             "profiles_per_npu": 5, "profile_counts": {str(k): v for k, v in counts.items()},
             "profile_compute_ms": {str(k): v for k, v in compute.items()},
             "display": "Actual request order, equal width per request; no simulated execution timing is encoded."}
    target.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print({"figure": str(target), "input_fingerprint": data["input_fingerprint"], "unique_sequences": 32})


if __name__ == "__main__":
    main()
