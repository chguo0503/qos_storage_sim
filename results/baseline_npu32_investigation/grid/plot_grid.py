#!/usr/bin/env python3
"""Plot the complete matched grid, retaining physically overloaded inputs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "paired_metrics.csv").open()))
rows = [r for r in rows if float(r["start_ms"]) == 1000]
nqls = [256, 384, 512, 768, 1024, 1536]
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), sharey=True)
for ax, field, title in zip(axes, ["baseline_fleet_u", "once_fleet_u"],
                            ["Baseline: one FIFO Path per SSD", "Once: native instantaneous telemetry"]):
    data = np.full((3, 6), np.nan)
    for row in rows:
        i = int(row["short_npu_count"]) - 1
        j = nqls.index(int(row["long_nql"]))
        data[i, j] = 100 * float(row[field])
    plot = ax.imshow(data, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    for row in rows:
        i = int(row["short_npu_count"]) - 1
        j = nqls.index(int(row["long_nql"]))
        if row["capacity_feasible"] != "True":
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, hatch="///",
                                   edgecolor=(.15, .15, .15, .45), linewidth=0))
        ink = "white" if data[i,j] >= 80 else "black"
        ax.text(j, i-.09, f"{data[i,j]:.1f}%", ha="center", va="center", weight="bold", color=ink)
        ax.text(j, i+.22, f"load {float(row['rho_ssu']):.2f}", ha="center", va="center", fontsize=8, color=ink)
    ax.set_title(title, pad=12)
    ax.set_xticks(range(6), nqls)
    ax.set_yticks(range(3), ["1 short + 3 large", "2 short + 2 large", "3 short + 1 large"])
    ax.set_xlabel("NQL of the 192K large requests")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
fig.suptitle("4 NPUs / 1 SSD: victim count and read/compute balance both matter", y=.99)
fig.legend(handles=[Patch(facecolor="white", hatch="///", edgecolor="gray",
                         label="Hatched: nominal SSD demand exceeds capacity; not a pure FIFO-loss test")],
           loc="lower center", bbox_to_anchor=(.5, .077), frameon=False)
fig.text(.5, .025, "All cases use [1000,2000] ms, full active coverage, seed 42. 1K short compute is extrapolated; some large NQLs are interpolated.",
         ha="center", fontsize=9)
fig.subplots_adjust(left=.135, right=.93, bottom=.25, top=.81, wspace=.13)
bar = fig.add_axes([.945, .25, .014, .56])
fig.colorbar(plot, cax=bar, label="Fleet compute (%)")
for suffix in ("png", "pdf", "svg"):
    fig.savefig(ROOT / ("parameter_grid." + suffix), dpi=200, bbox_inches="tight")
plt.close(fig)
