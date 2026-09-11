#!/usr/bin/env python3
"""Read long-card phase coherence from existing role-separated traces."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path):
    with gzip.open(path, "rt") as stream:
        result = json.load(stream)
    meta = result["metadata"]
    n_long = meta["long_npu_count"]
    c = meta["profiles"][-1]["per_layer_compute_us"] / 1000
    by_npu = {n: [] for n in range(n_long)}
    for batch in result["summary"]["microbatch_metrics"]:
        if batch["npu_id"] not in by_npu:
            continue
        for layer in batch["layer_metrics"]:
            if 2000 <= layer["compute_start_ms"] < 4000:
                by_npu[batch["npu_id"]].append(layer)
    assert all(by_npu.values())
    phase = [min(l["compute_start_ms"] for l in by_npu[n]) % c for n in range(n_long)]
    ordered = sorted(phase)
    gaps = [b-a for a,b in zip(ordered,ordered[1:])] + [c+ordered[0]-ordered[-1]]
    width = c-max(gaps)
    drift = max(min(abs(l["compute_start_ms"] % c-phase[n]),
                    c-abs(l["compute_start_ms"] % c-phase[n]))
                for n, layers in by_npu.items() for l in layers)
    lifetimes = [l["io_ready_time_ms"]-l["io_start_time_ms"]
                 for layers in by_npu.values() for l in layers]
    return dict(label=meta["label"], seed=meta["seed"], order=meta["order"],
                strategy=result["strategy"], long_npu_count=n_long,
                input_fingerprint=result["input_fingerprint"], result=str(path.relative_to(HERE)),
                result_sha256=sha(path), long_layer_compute_ms=c,
                warm_long_layer_count=len(lifetimes),
                phase_by_npu_mod_C_ms=phase, circular_cohort_phase_width_ms=width,
                max_within_card_phase_drift_ms=drift,
                long_read_lifetime_min_ms=min(lifetimes), long_read_lifetime_max_ms=max(lifetimes),
                long_read_lifetime_mean_ms=math.fsum(lifetimes)/len(lifetimes))


def main():
    plan=json.loads((HERE/"plan.json").read_text())
    rows=[]
    for job in plan["jobs"]:
        paths=list((HERE/"runs"/job["label"]/job["strategy"]).glob("*.json.gz"))
        if len(paths)!=1:
            continue
        rows.append(measure(paths[0]))
    report=dict(expected_count=40, complete_count=len(rows), analyzer_sha256=sha(Path(__file__)),
                definition="For long-layer compute starts in [2000,4000), reduce times modulo the fixed long per-layer C. Circular cohort width is C minus the largest gap between first warm phases; drift checks all warm long layers on each card against that card's first warm phase.",
                caveat="Compute phase coherence is directly observed. Read lifetime is io_ready-io_start; it is not SSD busy time and does not identify the physical FIFO queue head or prove a continuous I/O blackout.",
                rows=rows)
    (HERE/"long_phase_evidence.json").write_text(json.dumps(report,indent=2)+"\n")
    flat=[{k:v for k,v in r.items() if k!="phase_by_npu_mod_C_ms"} for r in rows]
    with (HERE/"long_phase_evidence.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]) if flat else ["label"])
        writer.writeheader();writer.writerows(flat)
    print(json.dumps(dict(complete=len(rows),max_phase_width_ms=max((r["circular_cohort_phase_width_ms"] for r in rows),default=None),
                          max_phase_drift_ms=max((r["max_within_card_phase_drift_ms"] for r in rows),default=None))))


if __name__=="__main__":
    main()
